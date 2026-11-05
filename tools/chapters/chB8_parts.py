"""Parts A and B of the chapter B8 notebook, first half (B8.12 to B8.20): statements and cells (used by build_chB8.py).

Part A: the value, CartPole for a network, your DQN with the ReplayBuffer of rl.py, the colleague's DQN that learns
nothing, the ablations on three seeds (B8.12 to B8.15). Part B, first half: the policy, REINFORCE without and with a
baseline, GAE and the clipped objective in your library, your PPO on vector environments (B8.16 to B8.20). The second
half (B8.21 to B8.28) appends its exercises to PART_B.exercises (B8.21 to B8.24) and adds part C (the preferences); it
reuses TOOLS_A, SETTINGS_20, train_ppo_20, AGENT_KEY_20 and load_agent_20 (chB8_code.GIVEN_20).
The numbers that depend on a calibration (thresholds, the seed of B8.12, the answers of the 🔮 B8.16) come from
tools/chapters/history_chB8.json (history_chB8.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from chapter_kit import Ex, Part, contract_entry, guarded

import chB8_code as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB8.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}

TRACKS = {"rapide": "R", "maths": "M", "code": "C"}


def tracks(ex_id: str) -> str:
    """The « Parcours » of the header, from the contract (R, M, C)."""
    return ", ".join(TRACKS[t] for t in contract_entry(ex_id).get("tracks", []))


def ex(ex_id: str, goal: str, prereq: str, **kwargs) -> Ex:
    """An exercise whose type, stars, minutes and title are those of the contract (docs/syllabus/data/chB8.json)."""
    e = contract_entry(ex_id)
    return Ex(ex_id, e["type"], e["stars"], e["minutes"], e["title"], goal, prereq, tracks=tracks(ex_id), **kwargs)


def fr_number(x: float, digits: int = 0) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


# ------------------------------------------------------------------------------------------------- calibrated values
SEED_12 = 13
assert abs(HISTORY.get("c12", {}).get("means", {}).get(str(SEED_12), 20.06) - 22.3) > 0.1   # not the value of the fiche
C13 = HISTORY.get("c13", {"threshold": 55.0, "worst": 92.89})
C14 = HISTORY.get("c14", {"threshold": 50.0, "worst": 100.0})
P16 = HISTORY.get("p16", {"rows": []})
C20 = HISTORY.get("c20", {"rows": [{"eval": 284.24}], "worst": 284.24})
THRESHOLD_20 = float(5 * int(0.6 * C20["worst"] / 5))
C20_EVALS = " ; ".join(fr_number(r["eval"], 0) for r in C20["rows"])


def p16_rows(baseline: str) -> list:
    return [r for r in P16["rows"] if r["baseline"] == baseline]


WORST_17 = min((r["last50"] for r in p16_rows("mean")), default=300.0)
THRESHOLD_17 = float(5 * int(0.5 * WORST_17 / 5))

PLACEHOLDERS = {"SEED_12": str(SEED_12), "THRESHOLD_13": f"{C13['threshold']:g}",
                "WORST_13": fr_number(C13["worst"], 0), "THRESHOLD_14": f"{C14['threshold']:g}",
                "THRESHOLD_17": f"{THRESHOLD_17:g}", "WORST_17": fr_number(WORST_17, 0),
                "THRESHOLD_20": f"{THRESHOLD_20:g}", "C20_EVALS": C20_EVALS}


def fill(text: str) -> str:
    """Replace the «NAME» placeholders of a cell (an unknown name raises KeyError)."""
    return re.sub(r"«([A-Z0-9_]+)»", lambda m: PLACEHOLDERS[m.group(1)] if m.group(1) != "KIND" else m.group(0), text)


# ------------------------------------------------------------------------------------------- 🔮 B8.16: the predictions
def p16_answers() -> dict:
    """The three answers of the 🔮 B8.16, computed from the precomputed protocol (6 seeds x 3 variants)."""
    import numpy as np

    rows = P16["rows"]
    if not rows:
        return {"a": 1, "b": 3, "c": True}
    seeds = sorted({r["seed"] for r in rows})
    last = {b: [next(r["last50"] for r in rows if r["seed"] == s and r["baseline"] == b) for s in seeds]
            for b in ("none", "mean", "normalize")}
    medians = [float(np.median(last[b])) for b in ("none", "mean", "normalize")]
    best = int(np.argmax(medians))
    reached = sum(1 for r in p16_rows("none") if r["first_200"] is not None)
    beats = all(last["mean"][i] > last["none"][i] for i in range(len(seeds)))
    return {"a": best, "b": reached, "c": beats, "medians": medians, "margin_a": sorted(medians)[-1] - sorted(medians)[-2]}


P16_ANSWERS = p16_answers()


def tests_after(ex_id: str, probe: str, keyword: str, n_tests: str, what: str) -> list:
    """The second check cell of a 🔨: the tests of the exercise (explicit test file, -k group)."""
    cell = (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_DEEP_RL, "{keyword}")   # {n_tests}\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent {what}.")')
    ref = cell.replace('run_pytest_group(TEST_DEEP_RL, "', 'run_pytest_group(TEST_DEEP_RL, impl="ref", keyword="')
    return [("check", cell), ("solution", ref)]


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** : ouvre `mon_travail/mylearn/deep_rl.py` (créé par `python tools/start_chapter.py "
            "B8`), lis la docstring, remplace chaque `raise NotImplementedError(...)` de l'exercice par ton code et "
            "**enregistre**. NumPy seulement, sans PyTorch. La première cellule de vérification recharge ta librairie et "
            "l'essaie sur des cas visibles ; la seconde lance les tests de l'exercice. `python -m pytest "
            "tests/test_chb8_deep_rl.py -q` les lance tous dans un terminal.")


# =====================================================================================================================
PART_A = Part("A", "La valeur : un réseau à la place de la table",
              "Au ch. 26, tu as tenu CartPole avec une table de 12 cases (26.38) et écrit `rl.py` : `ReplayBuffer`, "
              "`returns_to_go`, `evaluate_policy`. Ici, un **réseau** remplace la table. Tu prends d'abord CartPole en "
              "main pour un réseau (B8.12), tu écris ton DQN avec sa mémoire de rejeu et son réseau cible (B8.13), tu "
              "répares le DQN d'un collègue qui apprend mal (B8.14), puis tu mesures ce que chaque stabilisateur "
              "apporte, sur trois graines (B8.15).\n\n"
              "La cellule suivante charge les outils du chapitre : `wbr` (`wb.deep_rl_env`), qui fabrique les "
              "environnements, évalue une politique (`wbr.evaluate`, 500 pas au plus, sans exploration) et contient les "
              "agents de **référence** dont se servent les vérifications ; `EVAL_SEED`, les graines de toutes les "
              "évaluations (jamais celles d'un entraînement) ; `TEST_DEEP_RL`, le fichier des tests de `deep_rl.py`. "
              "Ta librairie `rl.py` du ch. 26 sert telle quelle : si tu ne l'as pas, la référence la remplace (une ligne "
              "ℹ️ le dit). PyTorch tourne sur un seul thread : pour d'aussi petits réseaux, c'est plus rapide. Pendant "
              "qu'une vérification fait tourner ton code, les fonctions de `wbr` qui calculent une réponse (le DQN, "
              "REINFORCE, GAE, PPO de référence et leurs agents) sont surveillées : si ton code en appelle une, la "
              "vérification s'arrête sur un ❌, puisqu'elle mesurerait la référence et pas ton travail.",
              given=C.TOOLS_A, exercises=[
    ex("B8.12", "lire l'interface de CartPole et évaluer honnêtement une politique, aléatoire ou tabulaire, avant d'y "
                "mettre un réseau.",
       "ch. 26 (26.21, 26.36, 26.38) · fiche §208.1 (« CartPole, vu de près »), §208.6 (« Évaluer un agent », piège "
       "« un plafond caché dans l'évaluation »)",
       body=r"""**Contrat** des trois fonctions de la cellule suivante :
- `describe_12(env)` rend un dictionnaire **lu dans `env`**, jamais tapé : `"obs_shape"`, la forme d'une observation (un tuple d'entiers) ; `"n_actions"`, le nombre d'actions (un `int`) ; `"max_steps"`, la limite de temps d'un épisode (un `int`, dans `env.spec`) ; `"threshold"`, le retour moyen qui compte comme résolu (un `float`, dans `env.spec`) ;
- `random_policy_12(seed)` rend une politique `(state, info) -> int` qui joue 0 ou 1 au hasard : **chaque appel** tire `int(rng.integers(2))` d'**un seul** générateur `rng = np.random.default_rng(seed)`, créé une fois par `random_policy_12` (pas à chaque appel). `env.action_space.sample()` ne convient pas : `reset(seed=...)` ne réinitialise pas son générateur (🧠 Q10) ;
- `evaluate_12(env, policy, seed, n_episodes=100)` rend le dictionnaire de `mylearn.rl.evaluate_policy` (ta fonction de 26.21) pour `n_episodes` épisodes **entiers** de `env` : `max_steps` vaut la limite de temps de `env` (dans `env.spec`), pas le défaut d'`evaluate_policy` ; les graines des épisodes viennent de `rng=np.random.default_rng(seed)` (un générateur, jamais un entier).

La cellule donnée fixe `SEED_12` et relit la **table du ch. 26** (la solution de 26.38 : 12 états, l'angle et sa vitesse) : `MAKE_TABULAR_12()` fabrique le CartPole discrétisé, `TABULAR_POLICY_12` est sa politique gloutonne (une action par état), sans réentraînement.

La vérification essaie `describe_12` sur CartPole-v1 et sur un CartPole dont la spec dit 200 pas et un seuil de 195, `random_policy_12` sur 40 appels, `evaluate_12` sur la règle d'une ligne de 208.1, puis évalue ta politique aléatoire (graine `SEED_12`, 100 épisodes) et la table du ch. 26, avec et sans `max_steps`. Le retour moyen de ta politique aléatoire est vérifié par `wb.check` : rien à taper, aucune valeur à arrondir.

Dans tes notes : pourquoi la moyenne de ta politique aléatoire n'est-elle pas celle de la fiche (22,3) ? De combien varie une moyenne de 100 épisodes d'une graine à l'autre (ch. 2 : l'erreur type) ? Que deviendrait le score de la table si l'on oubliait `max_steps` ? Combien de cases faudrait-il à une table qui découperait les **quatre** variables en dix intervalles, et que perd-on à n'en garder que deux ?""",
       given=C.GIVEN_12, todo=C.TODO_12, check=C.CHECK_12, solution=C.SOLUTION_12 + "\n\n\n" + C.CHECK_12,
       record=C.RECORD_12),

    ex("B8.13", "écrire un DQN complet en PyTorch, avec la mémoire de rejeu de ta librairie, un réseau cible, la loss de "
                "Huber et un ε décroissant, et vérifier qu'il apprend.",
       "Ex B8.12 · ✏️ B8.1, B8.2 · 🔨 26.31 (`ReplayBuffer`) · fiche §208.1 (« Apprendre une valeur par régression »), "
       "§208.2",
       body=r"""**Contrat** des quatre fonctions de la cellule suivante (les réglages sont dans `SETTINGS_13`, ceux du DQN de référence du chapitre ; la cellule donnée les affiche) :
- `make_q_net_13(hidden=64)` rend un `nn.Module` neuf : 4 entrées, deux couches cachées de `hidden` neurones avec ReLU, 2 sorties (une valeur par action), en float32 ;
- `dqn_targets_13(q_target, next_states, rewards, terminated, gamma)` rend les cibles d'un mini-batch, un tenseur de forme `(B,)` **sans gradient** : $r + \gamma(1 - d)\max_{a'} Q_{\text{cible}}(s', a')$, avec `next_states` float32 `(B, 4)`, `rewards` `(B,)` et `terminated` booléen `(B,)` ;
- `epsilon_13(step, total_steps, settings=SETTINGS_13)` rend l'ε du pas `step` (0, 1, …) : linéaire de `eps_start` à `eps_end` sur les `eps_fraction × total_steps` premiers pas, puis `eps_end` ;
- `train_dqn_13(env, seed, total_steps, settings=SETTINGS_13)` entraîne un DQN pendant exactement `total_steps` pas de `env` et rend le triplet `(q_net, buffer, history)` :
  - **graines** : `torch.manual_seed(seed)` puis le réseau ; `env.reset(seed=seed)` au premier épisode seulement ; un générateur `np.random.default_rng(seed)` pour l'exploration ; `random_state=seed` pour la mémoire. Deux appels de même graine rendent le même réseau et le même historique ;
  - **la mémoire** : `mylearn.rl.ReplayBuffer(settings["capacity"], obs_shape=(4,), obs_dtype="float32", random_state=seed)`, ta classe du ch. 26 ; on y range `(s, a, r, s', terminated)`, **`terminated` seul** ;
  - **chaque pas** : action ε-greedy (`epsilon_13`), un pas d'environnement, la transition rangée ; un épisode **terminé ou tronqué** se termine par un `env.reset()` ;
  - **les mises à jour** : dès que `step + 1 >= learning_starts` et que la mémoire contient un mini-batch, tous les `train_freq` pas : un mini-batch de `batch_size`, les cibles de `dqn_targets_13` (appelée par ce nom : la vérification l'observe) avec le réseau cible, la loss `F.huber_loss(Q(s, a), y, delta=huber_delta)`, un pas d'Adam (`lr`) après `nn.utils.clip_grad_norm_(..., max_grad_norm)` ;
  - **le réseau cible** : une copie du réseau en ligne au départ, réécrite (`load_state_dict`) tous les `target_every` pas ;
  - **`history`** : `{"returns": [...], "ends": [...]}`, le retour de chaque épisode fini et le nombre de pas joués quand il a fini (1, 2, …).

La vérification essaie `make_q_net_13`, les cibles sur 4 transitions fixées (dont une terminée), ε à cinq pas, puis un entraînement de 600 pas sur un CartPole **coupé à 15 pas** (beaucoup de limites de temps) : le nombre de pas joués, ta mémoire (sa capacité, ce qu'elle range comme fins), l'historique, les graines des `reset`, les copies du réseau cible (le réseau que reçoit `dqn_targets_13`, tous les 100 pas dans cet essai) et la reproductibilité. La cellule suivante entraîne ton DQN sur `STEPS_13` pas (20 000 en mode rapide, ≈ 40 s ; 50 000 en mode complet), trace les retours d'entraînement et évalue la politique **gloutonne** à part, sur 100 épisodes (`wbr.evaluate`, 500 pas au plus). Elle attend deux propriétés, pas un nombre exact : le retour d'entraînement monte, et la politique gloutonne dépasse `THRESHOLD_13`.

Dans tes notes : combien d'épisodes ton DQN a-t-il joués, et combien de mises à jour a-t-il faites ? Pourquoi l'évaluation gloutonne diffère-t-elle des retours d'entraînement de la fin ? Pourquoi la courbe monte-t-elle puis redescend-elle parfois ? Que changerait `F.smooth_l1_loss` à la place de `F.huber_loss` avec ce `huber_delta` (✏️ B8.2) ?""",
       given=C.GIVEN_13, todo=C.TODO_13, check=C.CHECK_13, solution=C.SOLUTION_13 + "\n\n\n" + C.CHECK_13,
       after=[("code", C.EXPERIMENT_13)]),

    ex("B8.14", "retrouver par des diagnostics enchaînés les quatre erreurs d'un DQN qui apprend mal, sans relancer "
                "d'entraînement à chaque essai.",
       "Ex B8.13 · 🐛 26.24 · 🧠 Q3 · fiche §208.1, §208.2 (pièges)",
       body=r"""Un collègue a écrit un DQN en quatre morceaux : `new_memory(seed)`, la mémoire de rejeu (ta `ReplayBuffer`) ; `remember(...)`, qui range une transition et dit si l'épisode est fini ; `exploration_rate(step)`, le planning d'ε ; `bellman_targets(online, frozen, batch)`, les cibles d'un mini-batch ; et quelques constantes. Sa boucle d'entraînement, `run_colleague_14` (cellule donnée), est juste. Son DQN n'apprend pas comme il le devrait : ses retours d'entraînement restent ceux du hasard, et ses valeurs $Q$ ne montent pas comme celles d'un DQN sain. Quatre erreurs se sont glissées dans les morceaux, celles du titre. **Répare la cellule suivante en place** (mêmes noms, mêmes signatures).

La cellule de vérification enchaîne quatre diagnostics, chacun sur un morceau, sans entraîner : 1. les cibles d'un mini-batch fixé, avec deux réseaux différents ; 2. ce que `remember` range pour une limite de temps, une vraie fin et un pas ordinaire ; 3. ε à quatre moments du planning ; 4. de quand datent les transitions d'un mini-batch, après 2 000 pas rangés. Le premier qui échoue arrête la chaîne (⏳ pour les suivants) ; un message dit ce qu'il mesure, pas ce qu'il faut changer. La cellule d'expérience entraîne le DQN tel qu'il est (`STEPS_14` pas, une quinzaine de secondes) et trace ses retours, son ε et la moyenne de $\max_a Q$ sur 256 états de départ : lance-la une première fois **avant** de réparer, pour voir les symptômes. Une fois les quatre diagnostics passés, elle vérifie que le DQN réparé apprend (politique gloutonne au-dessus de `THRESHOLD_14`).

Dans tes notes : pour chaque erreur, le symptôme que tu as vu (sur les courbes, puis dans le message) et la ligne fautive. Lesquelles se voyaient sur les courbes, et lesquelles seulement dans un diagnostic ? Avant la réparation, la politique gloutonne tient déjà plus de 100 pas alors que les retours d'entraînement restent ceux du hasard : pourquoi (pense à ce que veut dire off-policy) ? Pourquoi un diagnostic par morceau plutôt qu'un entraînement complet après chaque correction ? Laquelle de ces erreurs garderait un DQN qui « apprend un peu » ?""",
       given=C.GIVEN_14, todo=C.TODO_14, check=C.CHECK_14, solution=C.SOLUTION_14 + "\n\n\n" + C.CHECK_14,
       after=[("code", C.EXPERIMENT_14)]),

    ex("B8.15", "mesurer sur trois graines ce que le réseau cible, le rejeu et Double DQN changent à l'apprentissage, et "
                "rapporter la dispersion avec le résultat.",
       "Ex B8.13 · ✏️ B8.1 · fiche §208.2 (« La surestimation, et Double DQN »), §208.6 (« Les graines », « Rapporter "
       "plusieurs graines »)",
       body=r"""Quatre variantes du DQN de B8.13 : `"dqn"` ; `"no_target"`, **sans réseau cible** (les cibles viennent du réseau en ligne, toujours sans gradient) ; `"no_replay"`, **sans rejeu** (une mémoire de `batch_size` transitions : chaque mini-batch est fait des derniers pas, dans le désordre) ; `"double"`, **Double DQN**. **Contrat** des deux morceaux de la cellule suivante :
- `targets_15(online, frozen, next_states, rewards, terminated, gamma, variant)` rend les cibles d'un mini-batch pour une variante, un tenseur `(B,)` sans gradient : celles de `dqn_targets_13` avec le réseau cible `frozen` pour `"dqn"` et `"no_replay"` ; la même formule avec le réseau en ligne pour `"no_target"` ; pour `"double"`, l'action choisie par le réseau en ligne (`argmax`) et sa valeur lue dans `frozen` ;
- `memory_size_15(variant, settings=SETTINGS_13)` rend la capacité de la mémoire : `settings["capacity"]`, sauf `settings["batch_size"]` pour `"no_replay"`.

La vérification compare tes cibles à celles de la référence sur 6 transitions (deux réseaux qui ne choisissent pas toujours la même action), pour les quatre variantes. Puis l'expérience : en **mode rapide**, les courbes viennent d'un fichier précalculé (`wbr.precomputed("a15")`) : le DQN de référence du chapitre, 20 000 pas, trois graines par variante, la politique gloutonne évaluée tous les 1 000 pas sur 20 épisodes ; la cellule rejoue d'abord les 3 000 premiers pas d'une des courbes (Double DQN, première graine) pour vérifier qu'elle retombe sur le fichier. En **mode complet** 🚀 (`FAST_MODE = False`), elle relance tout avec tes deux morceaux dans la boucle de référence : 50 000 pas, trois graines par variante, environ 20 minutes de CPU. La figure trace la **médiane** des trois graines et leur **intervalle** (du minimum au maximum) ; un tableau donne la dernière évaluation de chaque graine.

Dans tes notes : quelle variante t'a surpris ? Les intervalles se recouvrent-ils, et que peux-tu conclure avec trois graines seulement (208.6, rliable) ? Pourquoi « sans rejeu » est-il ici une mémoire de 64 transitions plutôt qu'aucune mémoire, et que mesure-t-on alors ? Que faudrait-il pour affirmer qu'une variante est meilleure ?""",
       given=C.GIVEN_15, todo=C.TODO_15, check=C.CHECK_15, solution=C.SOLUTION_15 + "\n\n\n" + C.CHECK_15,
       after=[("code", C.EXPERIMENT_15)]),
])


# =====================================================================================================================
def _record_16() -> str:
    a, b, c = P16_ANSWERS["a"], P16_ANSWERS["b"], P16_ANSWERS["c"]
    names = {0: "sans baseline", 1: "la baseline moyenne", 2: "les retours normalisés"}
    others_a = {k: "lis la médiane de chaque variante sous la figure" for k in (0, 1, 2) if k != a}
    entries_a = ", ".join(f'"{m}": {{{", ".join(str(k) for k, mm in others_a.items() if mm == m)}}}'
                          for m in sorted(set(others_a.values())))
    lines = [f'wb.record("B8.16a", {a}, choices=[0, 1, 2], mistakes={{{entries_a}}}, prediction=True,\n'
             f'          fractional="la réponse est le numéro d\'une variante : 0, 1 ou 2")']
    wrong_b = [k for k in range(7) if k != b]
    lines.append(f'wb.record("B8.16b", {b}, choices=list(range(7)), mistakes={{"compte, ligne « sans baseline », les '
                 f'graines où la moyenne glissante atteint 200": {{{", ".join(str(k) for k in wrong_b)}}}}}, '
                 f'prediction=True,\n          fractional="la réponse est un nombre de graines, de 0 à 6")')
    lines.append(f'wb.record("B8.16c", {c}, mistakes={{"compare, graine par graine, les 50 derniers épisodes avec et '
                 f'sans baseline": {not c}}}, prediction=True)')
    del names
    return "\n".join(lines)


PART_B = Part("B", "La politique : REINFORCE, GAE et PPO",
              "Tu changes de famille : au lieu d'apprendre $Q$ et d'en déduire la politique, tu ajustes **directement** "
              "les probabilités des actions. Tu prédis d'abord ce que change une baseline à REINFORCE (B8.16), tu le "
              "mesures sur le gradient et sur l'apprentissage (B8.17), tu écris dans ta librairie les deux calculs que "
              "les implémentations ratent le plus souvent, GAE et l'objectif clippé (B8.18, B8.19), puis tu assembles "
              "ton PPO sur des environnements vectorisés (B8.20). La suite de la partie lit ses journaux, règle ses "
              "hyperparamètres et teste sa robustesse.",
              exercises=[
    ex("B8.16", "prédire, avant de les voir, ce que la baseline et la normalisation des retours changent à REINFORCE sur "
                "CartPole.",
       "∂ B8.7 · 🧠 Q6 · fiche §208.3 (« REINFORCE », « La baseline et la variance »)",
       hypothesis=True,
       body=r"""Trois variantes de REINFORCE, une mise à jour (un pas d'Adam, learning rate $10^{-2}$) par épisode, $\gamma = 0{,}99$, la politique 4 → 64 → 2 (tanh), 300 épisodes : **sans baseline** (le poids de $\log\pi(a_t \mid s_t)$ est $G_t$) ; avec une **baseline moyenne** ($G_t - b$, $b$ la moyenne des $G_t$ des 20 épisodes précédents, 0 au premier) ; avec des **retours normalisés** ($(G_t - \bar G)/\sigma_G$ sur l'épisode). Chacune a été entraînée avec six graines (le fichier `wbr.precomputed("p16")`, lu par l'expérience). La fiche dit seulement qu'une baseline ne biaise pas le gradient et réduit sa variance sur un **bandit jouet** ; ici, c'est CartPole. **Avant d'exécuter**, écris tes trois prédictions :
- a) `best_16` : laquelle des trois variantes a la plus grande **médiane**, sur les six graines, du retour moyen de ses 50 derniers épisodes ? `0` sans baseline, `1` la baseline moyenne, `2` les retours normalisés ;
- b) `first_200_16` : sur combien des six graines la variante **sans baseline** atteint-elle, au moins une fois, une moyenne glissante de 200 (sur 20 épisodes) en 300 épisodes ? Un entier de 0 à 6 ;
- c) `beats_16` : la baseline moyenne finit-elle **au-dessus** de la variante sans baseline (50 derniers épisodes) sur **chacune** des six graines ? `True` ou `False`.

La cellule d'expérience n'affiche rien tant que tes trois prédictions ne sont pas écrites ; elle trace ensuite la médiane et l'intervalle des six graines pour chaque variante, donne les valeurs par graine et vérifie tes prédictions. Rien n'y dépend de ton code.

Dans tes notes, avant d'exécuter : sur CartPole, tous les retours $G_t$ sont positifs ; que fait alors REINFORCE sans baseline à **chaque** action jouée ? Qu'est-ce qui distingue la normalisation d'une simple baseline ?""",
       todo=C.TODO_16,
       solution=f'best_16, first_200_16, beats_16 = {P16_ANSWERS["a"]}, {P16_ANSWERS["b"]}, {P16_ANSWERS["c"]}',
       record=_record_16(),
       after=[("code", guarded(C.EXPERIMENT_16, ["best_16", "first_200_16", "beats_16"],
                               "⏳ Ex B8.16 : écris d'abord tes trois prédictions (best_16, first_200_16, beats_16) "
                               "dans la cellule précédente.")),
              ("md", "Dans tes notes, après l'expérience : la dispersion d'une graine à l'autre est-elle plus grande "
                     "pour une variante que pour les autres ? Une seule graine aurait-elle suffi à répondre à a) ? "
                     "Qu'est-ce qui, dans la figure, relève de la **variance** du gradient et qu'est-ce qui relève de "
                     "l'**échelle** des pas ?")]),

    ex("B8.17", "écrire REINFORCE en PyTorch, mesurer la dispersion de son gradient avec et sans baseline, et relier "
                "cette dispersion à la vitesse d'apprentissage.",
       "🔮 B8.16 · 🔨 26.14 (`returns_to_go`) · ∂ B8.7 · fiche §208.3",
       body=r"""**Contrat** des trois fonctions de la cellule suivante :
- `make_policy_17(hidden=64)` rend un `nn.Module` neuf : 4 entrées, une couche cachée de `hidden` neurones avec tanh, 2 sorties (les logits des deux actions) ;
- `episode_gradients_17(policy_net, episodes, baseline, gamma=0.99)` rend un tableau NumPy float64 de forme `(M, P)` : la ligne $i$ est le gradient, par rapport à **tous** les paramètres de `policy_net` mis bout à bout dans l'ordre de `policy_net.parameters()` (`P` nombres), de la loss de l'épisode $i$, $-\frac{1}{T}\sum_t \log\pi(a_t \mid s_t)\,(G_t - b)$. `episodes` est une liste de $M$ dictionnaires `"states"` (float32 `(T, 4)`), `"actions"` (int64 `(T,)`) et `"rewards"` (`(T,)`) ; $G_t$ vient de `mylearn.rl.returns_to_go` (ta fonction de 26.14) ; $b = 0$ pour `"none"`, et pour `"mean"` la moyenne des $G_t$ de **tous** les épisodes donnés (une seule constante) ;
- `train_reinforce_17(seed, n_episodes, baseline, lr=1e-2, gamma=0.99)` entraîne REINFORCE sur CartPole-v1 (`wbr.make_env()`) et rend le couple `(policy_net, returns)`, `returns` la liste des retours des `n_episodes` épisodes d'entraînement : `torch.manual_seed(seed)` puis `make_policy_17()` ; `env.reset(seed=seed)` une fois au départ ; chaque épisode joué en entier avec des actions **tirées** dans la politique (`torch.distributions.Categorical`) ; puis un pas d'Adam (`lr`) sur la loss de l'épisode, avec $b = 0$ (`"none"`) ou $b$ la moyenne des $G_t$ des 20 épisodes **précédents** (`"mean"`, 0 au premier épisode).

La cellule donnée fixe deux politiques (une non entraînée ; une entraînée 60 épisodes par la référence, ses poids relus) et 32 épisodes joués par chacune (`EPISODES_17`). La vérification compare tes gradients par épisode à ceux de la référence, pour les deux politiques et les deux baselines, puis affiche leur **dispersion**, $\frac{1}{M}\sum_i \lVert g_i - \bar g\rVert^2$ (`wbr.gradient_spread`), avec et sans baseline. La cellule suivante entraîne ta REINFORCE sans puis avec baseline (300 épisodes chacune, environ une minute : les épisodes longs coûtent) et trace tes courbes sur l'intervalle des six graines de la référence (le protocole de B8.16) ; elle attend qu'avec la baseline, tes 50 derniers épisodes dépassent `THRESHOLD_17`.

Dans tes notes : de combien la baseline réduit-elle la dispersion pour chacune des deux politiques, et pourquoi si peu pour l'une d'elles (pense à la longueur des épisodes et à la taille des $G_t$) ? Une baseline **constante** peut-elle augmenter la variance ? Le gradient moyen change-t-il avec la baseline (∂ B8.7), et le vois-tu sur 32 épisodes ? Ta courbe tombe-t-elle dans l'intervalle de la référence ?""",
       given=C.GIVEN_17, todo=C.TODO_17, check=C.CHECK_17, solution=C.SOLUTION_17 + "\n\n\n" + C.CHECK_17,
       after=[("code", C.EXPERIMENT_17)]),

    ex("B8.18", "écrire GAE à rebours, avec la distinction entre fin d'épisode et limite de temps, et la variance "
                "expliquée d'un critique, en NumPy.",
       "✏️ B8.4 · fiche §208.4 (« GAE », « Fin d'épisode ou limite de temps », « Le critique et sa variance expliquée »)",
       mylearn="deep_rl.py",
       body=mylearn_howto() + r"""

**Contrat** de `compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95)`, qui rend le couple `(advantages, returns)` :
- les cinq tableaux ont la même forme, `(T,)` ou `(T, n_envs)`, le **temps sur le premier axe** ; chaque colonne est un rollout à part, avec sa propre récursion ;
- $\delta_t = r_t + \gamma(1 - \text{terminated}_t)\,\text{next\_values}_t - \text{values}_t$, puis à rebours $\hat A_t = \delta_t + \gamma\lambda(1 - \text{done}_t)\,\hat A_{t+1}$, avec $\text{done}_t$ = `terminated` **ou** `truncated` et $\hat A_T = 0$ après le dernier pas ;
- `next_values[t]` est **toujours** la valeur de l'observation qui suit le pas $t$, au dernier pas du rollout compris : il n'y a pas de valeur d'amorçage à part, et on ne la reprend pas dans `values[t+1]` ; un pas tronqué amorce sur sa `next_values` (la vraie dernière observation) mais arrête la récursion ; `terminated` et `truncated` peuvent être vrais ensemble ;
- `returns` vaut `advantages + values` (pas `next_values`), de la forme de `rewards` ;
- $\gamma$ et $\lambda$ dans $[0, 1]$, bornes comprises ; sinon `ValueError`.

**Contrat** de `explained_variance(y_pred, y_true)`, qui rend un `float` : $1 - \operatorname{Var}(y_{\text{true}} - y_{\text{pred}})/\operatorname{Var}(y_{\text{true}})$, **la prédiction d'abord** (l'ordre inverse de `sklearn.metrics.explained_variance_score`), des variances de population (`np.var`) ; `nan` **seulement** quand la variance de `y_true` vaut exactement 0 ; une prédiction constante donne 0, pas `nan`. Attention : `np.var([0.1, 0.1, 0.1])` vaut $1{,}9 \times 10^{-34}$ et non 0 ; les tests n'emploient que des constantes exactes en binaire (3, 0, −2,5), et acceptent aussi une constante détectée autrement (`np.ptp(y_true) == 0`).

**Pour les deux** : les entrées ne sont jamais modifiées ; des formes différentes lèvent `ValueError`, même quand NumPy pourrait les diffuser (`(T, 1)` contre `(T,)`, un tableau de longueur 1), y compris pour `terminated` et `truncated` ; des listes et des tableaux float32 sont acceptés ; les tests comparent à $10^{-6}$ près.

La première cellule de vérification essaie tes fonctions sur les deux exemples de la docstring, le tableau de 208.4, un rollout `(9, 3)` comparé colonne par colonne et les trois exemples de variance expliquée de 208.4, puis affiche ce que rend ta `compute_gae` sur le rollout de ✏️ B8.4 (compare à tes réponses). La seconde lance les 79 tests de l'exercice (`-k "test_compute_gae_ or test_explained_variance_"`).

Dans tes notes : que donnent λ = 0 et λ = 1 (lis-le sur ta fonction) ? Pourquoi `next_values` plutôt que `values` décalé d'un pas ? Que se passerait-il au dernier pas d'un rollout si l'on prenait 0 pour la valeur suivante ?""",
       check=C.RELOAD + C.CHECK_18, solution=C.CHECK_18,
       after=tests_after("B8.18", "mylearn.deep_rl.compute_gae([1.0], [0.0], [0.0], [True], [False])",
                         "test_compute_gae_ or test_explained_variance_", "79 tests",
                         "tes fonctions compute_gae et explained_variance")),

    ex("B8.19", "écrire l'objectif clippé de PPO et son gradient par rapport aux ratios, cas par cas, sans autograd.",
       "✏️ B8.5 · fiche §208.5 (« L'objectif clippé »)", mylearn="deep_rl.py",
       body=mylearn_howto() + r"""

**Contrat** de `clipped_surrogate(ratio, advantages, clip_eps=0.2)`, qui rend le couple `(loss, gradient)` :
- `loss` $= -\frac{1}{n}\sum_i \min\big(\rho_i \hat A_i,\ \operatorname{clip}(\rho_i, 1 - \varepsilon, 1 + \varepsilon)\,\hat A_i\big)$, un `float` (la convention de la fiche : on minimise l'opposé de $L^{\text{CLIP}}$) ;
- `gradient` est la dérivée de `loss` par rapport à chaque ratio, un tableau `(n,)` : $-\hat A_i/n$ quand le terme non clippé est le minimum, **égalités comprises** (un ratio dans l'intervalle, ou **sur un bord exact**, comme l'autograd de `torch.clamp`), et 0 quand le terme clippé est **strictement** plus petit ; un avantage nul donne un gradient nul. Une façon de vérifier : pour $\hat A > 0$, le gradient est non nul tant que $\rho \le 1 + \varepsilon$ ; pour $\hat A < 0$, tant que $\rho \ge 1 - \varepsilon$ ;
- `ratio` et `advantages` de forme `(n,)`, sans diffusion (des formes différentes lèvent `ValueError`) ; un ratio de 0 est accepté, un ratio négatif lève `ValueError` ; `clip_eps` doit être > 0 (sinon `ValueError`) et **peut dépasser 1** ;
- les entrées ne sont jamais modifiées ; listes et tableaux float32 acceptés ; tolérance $10^{-6}$.

La première cellule de vérification essaie ta fonction sur l'exemple de 208.5, puis la compare à l'autograd de PyTorch (`-torch.min(r * A, r.clamp(1 - eps, 1 + eps) * A).mean()`) sur 9 ratios, dont deux sur un bord exact, pour ε = 0,2, 0,5 et 1,5. La seconde lance les 38 tests de l'exercice (`-k "test_clipped_surrogate_"`).

Dans tes notes : dans quelles zones de la figure 1 de l'article le gradient s'annule-t-il ? Pourquoi un ratio monté au-dessus de $1 + \varepsilon$ avec un avantage **négatif** garde-t-il son gradient ? Que devient l'objectif avec $\varepsilon \ge 1$ du côté bas ?""",
       check=C.RELOAD + C.CHECK_19, solution=C.CHECK_19,
       after=tests_after("B8.19", "mylearn.deep_rl.clipped_surrogate([1.0], [1.0])", "test_clipped_surrogate_",
                         "38 tests", "ta fonction clipped_surrogate")),

    ex("B8.20", "assembler un PPO complet sur des environnements vectorisés, avec ta GAE et ton objectif clippé, et le "
                "voir résoudre CartPole.",
       "Ex B8.17, 🔨 B8.18, B8.19 · ✏️ B8.4, B8.5 · fiche §208.4, §208.5 (« La loss complète et la boucle », « Les "
       "environnements vectorisés ») · 🔁 R2",
       body=r"""**Contrat** des quatre morceaux de la cellule suivante (les réglages sont dans `SETTINGS_20`, ceux du PPO de référence du chapitre, affichés par la cellule donnée) :
- `ActorCritic20(hidden=64)`, un `nn.Module` : un **acteur** (4 → `hidden` → `hidden` → 2 logits, tanh) et un **critique** (4 → `hidden` → `hidden` → 1, tanh), deux réseaux séparés dans les attributs `actor` et `critic` ; `forward(obs)` rend les logits `(N, 2)`, `value(obs)` rend $V$ de forme `(N,)` ;
- `rollout_20(agent, envs, obs, just_reset, n_steps, generator)` joue `n_steps` pas dans chacune des $N$ copies de `envs` (un `gym.make_vec` en autoreset next-step) et rend le triplet `(batch, obs, just_reset)` :
  - les actions sont **tirées** dans la politique de l'agent, sans gradient (avec `generator`, un `torch.Generator`) ;
  - `obs` `(N, 4)` est l'observation courante ; `just_reset` `(N,)` vaut `True` pour une copie dont le pas précédent a fini l'épisode : son prochain pas ne fait que le reset ;
  - `batch` est un dictionnaire de tableaux NumPy de forme `(n_steps, N, ...)` : `"obs"` (avant le pas), `"actions"`, `"log_probs"` (de l'action jouée), `"values"` ($V$ de `obs`), `"rewards"`, `"terminated"`, `"truncated"`, `"next_obs"` (ce que rend le pas), `"next_values"` ($V$ de `next_obs` : au pas où un épisode finit, celle de la **vraie dernière observation**) et `"valid"` (`False` au pas qui ne fait que le reset d'une copie : ce n'est pas une transition) ;
  - les `obs` et `just_reset` rendus servent au rollout suivant ;
- `ppo_loss_20(agent, obs, actions, old_log_probs, advantages, returns, settings=SETTINGS_20)` rend le couple `(loss, parts)` pour un mini-batch de tenseurs : `loss` $= L^{\pi} + c_1 L^V - c_2 H$ (un tenseur, avec son graphe), $L^{\pi} = -\operatorname{mean}\min(\rho \hat A, \operatorname{clip}(\rho, 1 \pm \varepsilon)\hat A)$, $\rho = \exp(\log\pi(a \mid s) - \text{old\_log\_probs})$, $L^V = \frac12 \operatorname{mean}(V(s) - \text{returns})^2$, $H$ l'entropie moyenne de $\pi(\cdot \mid s)$, $c_1$ = `vf_coef`, $c_2$ = `ent_coef`, $\varepsilon$ = `clip_eps` ; les avantages servent tels quels (on les normalise avant) ; `parts` est un dictionnaire de `float` : `"policy_loss"`, `"value_loss"`, `"entropy"`, `"approx_kl"` (la moyenne de $(\rho - 1) - \ln\rho$), `"clipfrac"` (la part des $|\rho - 1| > \varepsilon$) ;
- `train_ppo_20(seed, total_steps, settings=SETTINGS_20, reward_fn=None)` entraîne PPO et rend le couple `(agent, logs)` :
  - **départ** : `torch.manual_seed(seed)`, l'agent, Adam (`lr`, `eps=adam_eps`), un `torch.Generator().manual_seed(seed)` pour les actions et les mélanges, `n_envs` copies de CartPole-v1 (`gym.make_vec(..., vectorization_mode="sync")`, autoreset next-step) remises à zéro une fois avec `seed` ;
  - **les itérations** : `total_steps // (n_envs × n_steps)` ; à chacune, le learning rate `lr × (1 − it/n_iter)` quand `anneal_lr`, un rollout (`rollout_20`, appelée par ce nom), puis les avantages et les retours par `mylearn.deep_rl.compute_gae` (ta fonction de B8.18, **appelée par ce nom** : la vérification l'observe) avec `gamma` et `gae_lambda`, et la variance expliquée par `mylearn.deep_rl.explained_variance`, sur les transitions valides ;
  - **la mise à jour** : `n_epochs` passages sur les transitions **valides**, mélangées puis coupées en `n_minibatches` mini-batches ; dans chacun, les avantages normalisés sur le mini-batch (`norm_adv`), `ppo_loss_20` (appelée par ce nom), `clip_grad_norm_(..., max_grad_norm)` et un pas d'Adam ; et `mylearn.deep_rl.clipped_surrogate` (B8.19) sur les mêmes ratios (détachés) et les mêmes avantages, dont l'écart à `parts["policy_loss"]` entre dans le journal ;
  - **`reward_fn(obs, actions, next_obs)`**, quand elle est donnée, remplace les récompenses du rollout par les siennes (tableau `(N,)` pour un pas) avant GAE : le RLHF de la partie C s'en servira ; `"mean_return"` reste calculé avec les **vraies** récompenses ;
  - **`logs`** : un dictionnaire de listes, une valeur par itération, aux clés de `wbr.LOG_KEYS` (`"step"`, pas d'environnement joués jusque-là, toutes copies comprises ; `"iteration"` ; `"lr"` ; `"policy_loss"`, `"value_loss"`, `"entropy"`, `"approx_kl"`, `"clipfrac"`, moyennes sur les mini-batches de l'itération ; `"explained_variance"` ; `"episodes"`, le nombre d'épisodes finis pendant le rollout ; `"mean_return"`, leur retour moyen, `nan` s'il n'y en a pas) et `"surrogate_gap"` (le plus grand écart de l'itération entre ta `clipped_surrogate` et la loss torch).

La vérification essaie `ActorCritic20` ; puis `rollout_20` avec un agent fixe de la référence, deux rollouts de 10 pas sur deux copies coupées à 12 pas : elle rejoue l'environnement avec **tes** actions et compare chaque champ du batch ; puis `ppo_loss_20` sur un mini-batch de 16 contre la référence ; enfin une boucle de deux itérations sur des copies coupées à 20 pas (pour qu'il y ait des limites de temps), qui observe tes appels à `rollout_20`, `compute_gae`, `ppo_loss_20` et `clipped_surrogate` : GAE reçoit les tableaux du rollout, la mise à jour ne voit que les transitions valides, et les avantages de chaque mini-batch sont normalisés. La cellule suivante entraîne ton PPO sur `STEPS_20` pas (50 000 en mode rapide, environ 25 s), trace le retour de chaque rollout et évalue la politique **déterministe** (l'action la plus probable) sur 100 épisodes ; elle attend que le retour monte et que l'évaluation dépasse `THRESHOLD_20`, puis enregistre ton agent (`AGENT_KEY_20`, à ton nom) pour la suite du chapitre ; un agent sous le seuil est enregistré quand même (ses journaux servent à B8.21), et la suite le dit.

Dans tes notes : combien de pas de reset ton rollout a-t-il écartés sur 50 000 pas, et que se serait-il passé en les gardant (🕰️ Gymnasium 1.x) ? Où ta boucle amorce-t-elle une limite de temps, et que fait `ppo.py` de CleanRL à la place (208.4) ? Pourquoi normaliser les avantages par mini-batch ? Combien de pas d'Adam ton PPO a-t-il faits, et combien de transitions chacun a-t-il servi ?""",
       given=C.GIVEN_20, todo=C.TODO_20, check=C.CHECK_20, solution=C.SOLUTION_20 + "\n\n\n" + C.CHECK_20,
       after=[("code", C.EXPERIMENT_20)]),
])

PARTS = [PART_A, PART_B]

for _part in PARTS:
    _part.given = fill(_part.given)
    for _ex in _part.exercises:
        for _field in ("given", "todo", "check", "solution", "record", "body"):
            setattr(_ex, _field, fill(getattr(_ex, _field)))
        _ex.after = [(kind, fill(text)) for kind, text in _ex.after]
