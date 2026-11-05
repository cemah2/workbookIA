"""End of part B (the policy: B8.21 to B8.24) and part C (the preferences: B8.25 to B8.28) of the chapter B8 notebook,
second half: statements and cells (used by build_chB8.py).

B8.21 to B8.24 are appended to PART_B.exercises (chB8_parts.py); part C is appended to PARTS. The answers of 📈 B8.21
and 🔮 B8.23, the review lines of 🛠️ B8.27 and the thresholds of 📦 B8.26 come from tools/chapters/history_chB8.json
(entries "l21", "c23", "r26", "c28b").
"""

from __future__ import annotations

import json
import re

from chapter_kit import Part, guarded

import chB8_code2 as C2
from chB8_parts import HISTORY, PART_B, PARTS, ex, fr_number, mylearn_howto, tests_after


def fill(text: str, values: dict) -> str:
    """Replace each «NAME» of a cell by its value; every placeholder must be filled (except «KIND», filled per
    notebook by build_chB8.py)."""
    for key, value in values.items():
        text = text.replace(f"«{key}»", str(value))
    left = re.findall(r"«[A-Z0-9_]+»", text.replace("«KIND»", ""))
    if left:
        raise ValueError(f"unfilled placeholder {left[0]}")
    return text


def letters_record(ex_id: str, answer: str, hints: dict, prediction: bool = False) -> str:
    """wb.record of a letter answer, one message per other letter (letters with the same message share it)."""
    grouped = {}
    for letter, message in hints.items():
        if letter != answer:
            grouped.setdefault(message, []).append(letter)
    value = lambda ls: f'"{ls[0]}"' if len(ls) == 1 else "{" + ", ".join(f'"{x}"' for x in sorted(ls)) + "}"  # noqa: E731
    entries = ", ".join(f'"{message}": {value(ls)}' for message, ls in grouped.items())
    return f'wb.record("{ex_id}", "{answer}", mistakes={{{entries}}}' + (", prediction=True)" if prediction else ")")


def repr_review(review: list) -> str:
    """The source of the review of the solutions notebook: one dict per line block, as a learner would write it."""
    lines = ["["]
    for entry in review:
        lines.append("    {" + ",\n     ".join(f"{json.dumps(k)}: {entry[k]!r}" for k in entry) + "},")
    return "\n".join(lines + ["]"])


def choice_record(ex_id: str, answer: int, choices: list, message: str, fractional: str) -> str:
    """wb.record of a 🔮 prediction chosen in a list of integers: one message for every other value."""
    others = "{" + ", ".join(str(c) for c in choices if c != answer) + "}"
    return (f'wb.record("{ex_id}", {answer}, choices={choices}, mistakes={{"{message}": {others}}}, prediction=True,\n'
            f'          fractional="{fractional}")')


# ------------------------------------------------------------------------------------------------- calibrated values
L21 = HISTORY.get("l21", {"letters": {"healthy": "A", "lr_high": "B", "no_clip": "C", "entropy_sign": "D",
                                      "no_critic": "E"}})
LETTERS_21 = L21["letters"]
CAUSES_21 = ("healthy", "lr_high", "no_clip", "entropy_sign", "no_critic")
HINTS_21 = {
    "healthy": "le run aux réglages de référence n'a aucune courbe anormale : reprends tes cinq panneaux un par un",
    "lr_high": "des pas trop grands poussent sans cesse des ratios hors de l'intervalle du clip : quelle courbe le "
               "compte ?",
    "no_clip": "sans clipping, aucun ratio n'est jamais rogné : quelle courbe compte les ratios rognés ?",
    "entropy_sign": "un bonus d'entropie de mauvais signe récompense une politique qui ne doute plus : quelle courbe "
                    "le montre ?",
    "no_critic": "un critique qui n'apprend pas ne prédit pas les retours : quelle courbe le mesure ?",
}


def record_21() -> str:
    return "\n".join(letters_record(f"B8.21{sub}", LETTERS_21[cause], {letter: HINTS_21[cause] for letter in "ABCDE"})
                     for sub, cause in zip("abcde", CAUSES_21))


C23 = HISTORY.get("c23", {})


def c23_answers() -> dict:
    """The answers of 🔮 B8.23 from the six reference agents (first set of evaluation seeds), and whether each holds
    on the second set too."""
    import numpy as np

    rows = C23.get("rows", [])
    if not rows:
        return {"a": 0, "b": 0, "c": 0, "robust": False}
    seeds = C23["protocol"]["eval_seeds"]

    def band(name, es):
        median = float(np.median([r[f"{name}/{es}"]["mean"] for r in rows]))
        return 0 if median >= 475 else (1 if median >= 200 else 2)

    def held(name, es):
        return sum(r[f"{name}/{es}"]["mean"] >= 475 for r in rows)

    a, b, c = band("gravity_x2", seeds[0]), band("length_x2", seeds[0]), held("length_x4", seeds[0])
    robust = (a == band("gravity_x2", seeds[1]) and b == band("length_x2", seeds[1])
              and c == held("length_x4", seeds[1]))
    return {"a": a, "b": b, "c": c, "robust": robust}


A23 = c23_answers()
assert A23["robust"] or not C23, "the answers of B8.23 change with the evaluation seeds"


def record_23() -> str:
    bands = "la réponse est le numéro d'une tranche : 0, 1 ou 2"
    return "\n".join([
        choice_record("B8.23a", A23["a"], [0, 1, 2], "lis la ligne « gravité doublée » du tableau : la médiane des six "
                      "agents", bands),
        choice_record("B8.23b", A23["b"], [0, 1, 2], "lis la ligne « mât deux fois plus long » du tableau : la médiane "
                      "des six agents", bands),
        choice_record("B8.23c", A23["c"], list(range(7)), "compte, ligne « mât quatre fois plus long », les agents "
                      "au-dessus de 475", "la réponse est un nombre d'agents, de 0 à 6")])


VALUES = {"ANSWERS_21": ", ".join(f'"{LETTERS_21[c]}"' for c in CAUSES_21),
          "YSCALE_21": 'axes_21[0, 1].set_yscale("log")\n',
          "ANSWERS_23": f'{A23["a"]}, {A23["b"]}, {A23["c"]}', "RECORD_23": record_23(),
          "GRID_EPISODES_24": str(C23.get("protocol", {}).get("grid_episodes", 50)),
          "DONE_LINES_27": "206", "RESET_LINES_27": "162, 205, 210, 211",
          }


# =====================================================================================================================
PART_B.exercises += [
    ex("B8.21", "reconnaître dans les journaux d'un PPO un learning rate trop grand, un clip absent, un bonus d'entropie de "
                "mauvais signe et un critique qui n'apprend pas.",
       "Ex B8.20 · fiche §208.4 (« Le critique et sa variance expliquée »), §208.5 (« Les journaux de PPO »)",
       body=r"""Cinq runs du PPO de référence du chapitre (`wbr.train_ppo`, 50 000 pas, la même graine), notés A à E dans un ordre tiré au hasard. L'un a les réglages de référence (`SETTINGS_20`) ; chacun des quatre autres en change **un seul** :
- un learning rate dix fois plus grand ($10^{-2}$), qui ne décroît jamais ;
- **pas de clipping** : un `clip_eps` si grand qu'aucun ratio n'est jamais rogné ;
- un bonus d'entropie **de mauvais signe** (`ent_coef = -0.2` : la loss récompense une politique sûre d'elle) ;
- un critique qui n'apprend pas (`vf_coef = 0` : sa loss ne compte plus).

La cellule donnée trace, pour chaque run, les cinq journaux de 208.5 au fil des itérations : le retour des épisodes finis pendant le rollout, la KL approchée (en échelle logarithmique), la fraction clippée, l'entropie et la variance expliquée ; une colonne a la même échelle pour les cinq runs. Ce sont les journaux que rend aussi ton `train_ppo_20` (B8.20). **Lis la figure** et réponds par une lettre entre guillemets :
- a) `healthy_21` : le run aux réglages de référence ;
- b) `lr_high_21` : le learning rate trop grand ;
- c) `no_clip_21` : sans clipping ;
- d) `entropy_sign_21` : le bonus d'entropie de mauvais signe ;
- e) `no_critic_21` : le critique qui n'apprend pas.

Chaque lettre sert une fois. Attention aux effets de **fin** d'un run réussi : le learning rate décroît jusqu'à 0, et quand presque tous les épisodes durent 500 pas, les retours à prédire se ressemblent tous. La cellule suivante trace les mêmes journaux pour **ton** PPO de B8.20.

Dans tes notes : pour chaque run malade, la courbe qui t'a convaincu et ce qu'elle mesure ; quelle autre courbe aurait pu te tromper. Le critique du run sain est-il cassé à la fin, quand sa variance expliquée baisse ? Un run peut-il apprendre à tenir 500 pas avec un critique qui n'apprend rien, et pourquoi ? Ton PPO ressemble-t-il au run sain ?""",
       given=C2.GIVEN_21, todo=C2.TODO_21, check=C2.CHECK_21, solution=C2.SOLUTION_21, record=record_21(),
       after=[("code", C2.EXPERIMENT_21)]),

    ex("B8.22", "mesurer sur trois graines l'effet de l'ε du clip, du λ de GAE et du nombre d'epochs sur l'apprentissage "
                "de PPO, et dire ce que trois graines permettent de conclure.",
       "Ex B8.20, B8.21 · 🔬 B8.15 · fiche §208.4 (« GAE »), §208.5, §208.6 (« Des hyperparamètres qui comptent », "
       "« Rapporter plusieurs graines »)",
       body=r"""Sept réglages de PPO : la référence (`SETTINGS_20`), et six variantes qui changent **un** hyperparamètre : `clip_eps` 0,1 ou 0,4 (au lieu de 0,2), `gae_lambda` 0,8 ou 1 (au lieu de 0,95), `n_epochs` 4 ou 20 (au lieu de 10). **Contrat** de `summary_22(logs)` : elle reçoit un journal au format de `train_ppo_20` (un dictionnaire de listes, une valeur par itération, avec au moins `"step"` et `"mean_return"`) et rend un dictionnaire :
- `"auc"` : la moyenne des valeurs **finies** de `logs["mean_return"]` (l'aire sous la courbe, divisée par sa longueur ; une itération où aucun épisode ne finit vaut `nan` et ne compte pas) ;
- `"first_475"` : le premier `logs["step"]` dont le retour atteint 475 (475 compris), un `int`, ou `None` s'il n'y en a pas ;
- le journal n'est pas modifié.

La vérification essaie ta fonction sur trois petits journaux (des `nan`, un retour qui n'atteint jamais 475, un 475 tout juste) et sur un run précalculé. Puis l'expérience : en **mode rapide**, les runs viennent d'un fichier précalculé (`wbr.precomputed("p22")`) : le PPO de référence, 50 000 pas, trois graines par réglage, évalué à la fin sur 100 épisodes déterministes ; la cellule rejoue d'abord 4 itérations d'un run pour vérifier qu'il retombe sur le fichier. En **mode complet** 🚀 (`FAST_MODE = False`), elle entraîne **ton** PPO de B8.20 sur les sept réglages et les trois graines, `STEPS_20` pas chacun (100 000), soit une vingtaine de minutes de CPU. La figure trace la médiane et l'intervalle des trois graines, un panneau par hyperparamètre ; le tableau donne, par graine, ton `summary_22` et l'évaluation finale.

Dans tes notes : quel hyperparamètre change le plus l'aire sous la courbe, et lequel le moins ? Les intervalles se recouvrent-ils ? Une variante qui apprend plus vite apprend-elle aussi plus sûrement (la dispersion, l'évaluation finale) ? Que changent 20 epochs à la KL approchée et à la fraction clippée (`runs_22[...]["approx_kl"]`), et pourquoi ? Avec trois graines, que peux-tu affirmer, et que faudrait-il pour affirmer plus (208.6) ? Quel réglage choisirais-tu pour le défi B8.28, et pourquoi ?""",
       given=C2.GIVEN_22, todo=C2.TODO_22, check=C2.CHECK_22, solution=C2.SOLUTION_22 + "\n\n\n" + C2.CHECK_22,
       after=[("code", C2.EXPERIMENT_22)]),

    ex("B8.23", "prédire, avant de le mesurer, ce que devient un agent PPO quand la gravité double ou que le mât s'allonge.",
       "Ex B8.20 · fiche §208.1 (« CartPole, vu de près »), §208.8 (« Généraliser au-delà de son environnement »)",
       hypothesis=True,
       body=r"""Ton agent de B8.20 a appris sur CartPole-v1 : $g = 9{,}8$ m/s², un mât de 1 m (`length = 0.5`, la **moitié** du mât, comme dans gymnasium). Que devient-il sur une autre physique, **sans réentraînement** ? Six agents PPO de référence (le protocole de B8.20 : `wbr.train_ppo`, 50 000 pas, graines 0 à 5) ont été évalués sur 100 épisodes déterministes (`wbr.evaluate(..., gravity=...)` ou `length=...`, qui recalcule ce qu'il faut). **Avant d'exécuter**, écris tes trois prédictions sur ces six agents :
- a) `gravity_23` : avec une **gravité doublée** (19,6 m/s²), la **médiane** des six retours moyens : `0` si elle reste au moins à 475, `1` si elle est entre 200 et 475, `2` si elle tombe sous 200 ;
- b) `length_23` : avec un mât **deux fois plus long** (`length = 1.0`), la même question, les mêmes tranches ;
- c) `longer_23` : avec un mât **quatre fois plus long** (`length = 2.0`), combien des six agents tiennent-ils encore au moins 475 ? Un entier de 0 à 6.

La cellule d'expérience n'affiche rien tant que tes trois prédictions ne sont pas écrites ; elle donne ensuite le tableau des six agents (sur deux jeux de graines d'évaluation), évalue **ton** agent de B8.20 sur les quatre physiques (si tu n'en as pas, l'agent de référence le remplace, une ligne ℹ️ le dit), et vérifie tes prédictions.

Dans tes notes, avant d'exécuter : un mât plus long tombe-t-il plus vite ou plus lentement ? Une gravité plus forte rend-elle le problème plus dur pour un humain ? Qu'est-ce qu'un agent entraîné sur une seule physique a pu apprendre qui ne vaut que pour elle ?""",
       todo=C2.TODO_23, solution=C2.SOLUTION_23, record="«RECORD_23»",
       after=[("code", guarded(C2.EXPERIMENT_23, ["gravity_23", "length_23", "longer_23"],
                               "⏳ Ex B8.23 : écris d'abord tes trois prédictions (gravity_23, length_23, longer_23) "
                               "dans la cellule précédente.")),
              ("md", "Dans tes notes, après l'expérience : ton agent se comporte-t-il comme les six agents de référence ? "
                     "L'agent de la graine qui ne résout pas CartPole réagit-il comme les autres, et qu'en conclus-tu sur "
                     "ce qu'il a appris ? Une évaluation sur une seule physique suffit-elle à dire qu'un agent « sait » "
                     "tenir un mât ?")]),

    ex("B8.24", "modifier correctement la physique de CartPole, puis mesurer comment le retour de ton agent chute quand la "
                "gravité ou la longueur du mât s'éloignent de celles de l'entraînement.",
       "🔮 B8.23 · Ex B8.20 · fiche §208.8 (« Généraliser au-delà de son environnement », « De la simulation au réel »)",
       body=r"""**Contrat** des deux fonctions de la cellule suivante :
- `set_physics_24(env, gravity=None, length=None)` reçoit un CartPole fait par `gym.make` (ou `wbr.make_env()`), lui donne une autre gravité et/ou une autre **demi**-longueur de mât (`None` : inchangée), sur `env.unwrapped`, en gardant cohérentes **toutes** les constantes que `CartPoleEnv.__init__` en déduit, et rend `env` ;
- `robustness_24(policy, name, factors, n_episodes=EPISODES_24, seed=EVAL_SEED)` rend la liste des retours moyens de `policy` quand la constante `name` (`"gravity"` ou `"length"`) vaut sa valeur par défaut (`DEFAULTS_24`) multipliée par chaque facteur : un `float` par facteur, chacun `wbr.evaluate(policy, n_episodes, seed, env=env)["mean_return"]` avec `env` un nouveau `wbr.make_env()` modifié par ta `set_physics_24` (les autres constantes inchangées).

La vérification compare tes constantes à celles de `wbr.make_env(gravity=..., length=...)` et rejoue un épisode avec les mêmes actions sur les deux environnements ; puis elle essaie `robustness_24` sur la règle d'une ligne de 208.1. L'expérience balaie la gravité et la longueur du mât de `FACTORS_24` avec ton agent de B8.20 (`EPISODES_24` épisodes par point, une dizaine de secondes), trace tes deux courbes sur la médiane et l'intervalle des six agents de référence de B8.23, donne la chute du retour en pourcentage, puis évalue ton agent sur le mât quatre fois plus long avec et sans le recalcul des constantes.

Dans tes notes : à partir de quel facteur ton agent passe-t-il sous 475, dans chaque direction ? La chute est-elle progressive ou brutale ? Pourquoi un agent peut-il tenir une gravité triplée et tomber avec un mât quatre fois plus long, alors que la règle d'une ligne de 208.1 tient ce mât-là ? Que change l'oubli de `polemass_length`, et pourquoi le voit-on ou non sur le retour ? Comment entraînerais-tu un agent pour qu'il tienne toute la plage (208.8) ?""",
       given=C2.GIVEN_24, todo=C2.TODO_24, check=C2.CHECK_24, solution=C2.SOLUTION_24 + "\n\n\n" + C2.CHECK_24,
       after=[("code", C2.EXPERIMENT_24)]),
]

for _ex in PART_B.exercises[-4:]:
    for _field in ("given", "todo", "check", "solution", "record", "body"):
        setattr(_ex, _field, fill(getattr(_ex, _field), VALUES))
    _ex.after = [(kind, fill(text, VALUES)) for kind, text in _ex.after]


# =====================================================================================================================
# Part C · the preferences
# =====================================================================================================================
REVIEW_SOLUTION_27 = [
    {"sujet": "fin d'épisode et limite de temps",
     "lignes": 206,
     "extrait": "next_done = np.logical_or(terminations, truncations)",
     "chez_moi": "terminated coupe l'amorçage dans δ, terminated ou truncated arrête la récursion, et une troncature "
                 "amorce sur V de la vraie dernière observation (next_values)",
     "effet": "ppo.py traite la limite de 500 pas comme une chute : les états proches de la fin d'un bon épisode "
              "reçoivent une valeur trop basse, alors que l'état ne dit rien du temps écoulé",
     "decision": "garder"},
    {"sujet": "autoreset next-step de gymnasium 1.x",
     "lignes": (162, 164),
     "extrait": "envs = gym.vector.SyncVectorEnv(",
     "chez_moi": "make_vec en autoreset next-step explicite, et un masque valid qui écarte le pas qui ne fait que "
                 "remettre une copie à zéro",
     "effet": "écrit pour gymnasium 0.29 (same-step), ppo.py garde sous la 1.x le pas de reset comme une transition : "
              "une action ignorée, une récompense 0, un faux passage de la dernière observation à la première",
     "decision": "garder"},
    {"sujet": "retours des épisodes dans les journaux",
     "lignes": (210, 211),
     "extrait": 'if "final_info" in infos:',
     "chez_moi": "je suis le retour de chaque copie sur les pas valides et je le range quand l'épisode finit",
     "effet": "en next-step, infos n'a plus de clé final_info : ppo.py n'affiche plus aucun retour sous gymnasium 1.x, "
              "sans message d'erreur",
     "decision": "garder"},
    {"sujet": "valeur de l'état suivant dans GAE",
     "lignes": (226, 228),
     "extrait": "nextvalues = values[t + 1]",
     "chez_moi": "next_values[t] est V de l'observation que rend le pas t, calculée pour chaque pas du rollout",
     "effet": "identique tant que l'épisode continue ; à une fin, values[t + 1] est la valeur d'un autre épisode, que "
              "ppo.py masque, ce qui l'empêche aussi d'amorcer une troncature",
     "decision": "garder"},
    {"sujet": "clip de la loss du critique",
     "lignes": (271, 280),
     "extrait": "v_clipped = b_values[mb_inds] + torch.clamp(",
     "chez_moi": "une loss quadratique simple, 0,5 fois la moyenne des carrés des écarts aux retours de GAE",
     "effet": "le clip limite le changement de V à ± clip_coef par mise à jour ; les 37 détails ne lui trouvent pas "
              "d'avantage net, et il ralentit un critique dont les retours valent des centaines",
     "decision": "tester"},
    {"sujet": "KL approchée dans les journaux",
     "lignes": 305,
     "extrait": 'writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)',
     "chez_moi": "la moyenne de (ρ − 1) − ln ρ sur tous les mini-batches de l'itération",
     "effet": "ppo.py écrit la valeur du dernier mini-batch de la dernière epoch, la plus éloignée de l'ancienne "
              "politique : ses courbes de KL ne se comparent pas directement aux miennes",
     "decision": "garder"},
    {"sujet": "arrêt des epochs sur la KL",
     "lignes": (292, 293),
     "extrait": "if args.target_kl is not None and approx_kl > args.target_kl:",
     "chez_moi": "toujours n_epochs passages, sans test de la KL",
     "effet": "désactivé par défaut dans ppo.py aussi (target_kl = None) ; activé, il protégerait d'un learning rate "
              "ou d'un nombre d'epochs trop grand, comme dans le run malade de B8.21",
     "decision": "tester"},
    {"sujet": "taille des mini-batches",
     "lignes": (246, 248),
     "extrait": "for start in range(0, args.batch_size, args.minibatch_size):",
     "chez_moi": "les transitions valides seulement, coupées en n_minibatches parts égales, le reste de la division "
                 "laissé de côté",
     "effet": "ppo.py coupe toujours 512 transitions en 4 mini-batches de 128, pas de reset compris ; chez moi la "
              "taille varie d'une itération à l'autre de quelques transitions",
     "decision": "garder"},
    {"sujet": "learning rate et nombre d'epochs",
     "lignes": (41, 56),
     "extrait": "learning_rate: float = 2.5e-4",
     "chez_moi": "lr de 1e-3 et 10 epochs (SETTINGS_20), pour 50 000 pas au lieu de 500 000",
     "effet": "quatre fois plus de pas d'Adam par itération et des pas quatre fois plus grands : la KL et la fraction "
              "clippée montent, c'est le prix d'un budget dix fois plus petit (B8.22)",
     "decision": "tester"},
    {"sujet": "graine du mélange des mini-batches",
     "lignes": 245,
     "extrait": "np.random.shuffle(b_inds)",
     "chez_moi": "torch.randperm avec un torch.Generator à graine, le même qui tire les actions",
     "effet": "les deux sont reproductibles à graine fixée ; ppo.py dépend du générateur global de NumPy, qu'un autre "
              "appel du programme peut déplacer",
     "decision": "garder"},
]
VALUES["REVIEW_27"] = repr_review(REVIEW_SOLUTION_27)

C26 = HISTORY.get("r26", {})
C28 = HISTORY.get("c28b", {})


def threshold_26() -> float:
    """The held-out accuracy asked of the learner's broad reward model: the worst of the five seeds of the reference,
    minus 0.1, rounded down to the twentieth."""
    import math
    rows = C26.get("rows", [])
    worst = min((r["rm_broad"]["acc_held"] for r in rows), default=0.85)
    return math.floor((worst - 0.1) * 20) / 20


VALUES.update({"STEPS_26": "30_000 if FAST_MODE else 100_000", "THRESHOLD_26": f"{threshold_26():g}"})

PART_C = Part("C", "Les préférences : Bradley-Terry, DPO, un RLHF miniature, la revue de code et le défi",
              "Les modèles de langue sont alignés avec des **préférences** : un humain compare deux réponses, un modèle de "
              "récompense apprend ses choix, et PPO l'optimise (le RLHF de 208.7) ; ou bien DPO saute le modèle de "
              "récompense. Tu écris d'abord les deux losses de préférences dans ta librairie (B8.25), tu rejoues le RLHF "
              "en miniature sur CartPole, avec une vraie récompense cachée et des préférences simulées, pour voir ce que "
              "l'agent fait du modèle qu'on lui donne (B8.26), tu confrontes ton PPO à celui de CleanRL (B8.27), puis tu "
              "relèves le défi (B8.28).",
              exercises=[
    ex("B8.25", "écrire en NumPy la loss d'un modèle de récompense de Bradley-Terry et la loss DPO, stables pour de très "
                "grands écarts.",
       "✏️ B8.6 · fiche §208.7 (« Le modèle de récompense de Bradley-Terry », « DPO : la récompense est cachée dans la "
       "politique »)", mylearn="deep_rl.py",
       body=mylearn_howto() + r"""

**Contrat** de `bradley_terry_loss(r_chosen, r_rejected)`, qui rend le triplet `(loss, grad_chosen, grad_rejected)` :
- `loss` $= \frac{1}{n}\sum_i -\log\sigma(r^{c}_i - r^{r}_i)$, un `float` ;
- `grad_chosen` et `grad_rejected` sont les gradients de cette loss **moyenne** (divisés par $n$) par rapport à chaque tableau, de forme `(n,)` : `grad_chosen` $\le 0$ et `grad_rejected` $=$ `-grad_chosen` ;
- la loss **et** les gradients restent finis et exacts pour des écarts de ±1 000 à ±10 000 : $-\log\sigma(d)$ = `np.logaddexp(0, -d)` et $\sigma(-d)$ = `np.exp(-np.logaddexp(0, d))` ; une formule comme $1 - e^d/(1 + e^d)$ donne `nan`.

**Contrat** de `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1)`, qui rend le couple `(loss, margins)` :
- `margins` $= \beta\big[(\log\pi^{c} - \log\pi^{c}_{\text{ref}}) - (\log\pi^{r} - \log\pi^{r}_{\text{ref}})\big]$, **avec** $\beta$, de forme `(n,)` ;
- `loss` $= \frac{1}{n}\sum_i -\log\sigma(\text{margins}_i)$, stable comme ci-dessus : c'est Bradley-Terry appliqué aux récompenses implicites $\beta(\log\pi - \log\pi_{\text{ref}})$ ;
- $\beta > 0$, sinon `ValueError`.

**Pour les deux** : des tableaux de forme `(n,)`, sans diffusion (des formes différentes lèvent `ValueError`, même `(n, 1)` contre `(n,)`) ; les entrées ne sont jamais modifiées ; listes et tableaux float32 acceptés ; les tests comparent à $10^{-6}$ près.

La première cellule de vérification essaie tes fonctions sur la docstring, sur 7 paires contre l'autograd de `-F.logsigmoid(r_c - r_r).mean()`, sur des écarts de −6 000 et +8 000, sur l'exemple de la docstring de `dpo_loss`, sur 6 paires contre l'équation 7 (β = 0,1 et 0,5) et sur β = 0, puis affiche ce que rendent tes fonctions sur les nombres de ✏️ B8.6 (compare à tes réponses). La seconde lance les 46 tests de l'exercice (`-k "test_bradley_terry_loss_ or test_dpo_loss_"`).

Dans tes notes : pourquoi ajouter la même constante à toutes les récompenses ne change-t-il rien ? Que vaut la loss DPO tant que la politique égale la référence ? Que règle $\beta$ ? B8.26 se sert de ta `bradley_terry_loss` pour entraîner un modèle de récompense en PyTorch : comment ses gradients NumPy peuvent-ils remonter dans un réseau ?""",
       check=C2.RELOAD + C2.CHECK_25, solution=C2.CHECK_25,
       after=tests_after("B8.25", "mylearn.deep_rl.bradley_terry_loss([0.0], [0.0])",
                         "test_bradley_terry_loss_ or test_dpo_loss_", "46 tests",
                         "tes fonctions bradley_terry_loss et dpo_loss")),

    ex("B8.26", "entraîner un modèle de récompense sur des préférences simulées, l'optimiser avec ton PPO, et mesurer ce "
                "que l'agent gagne selon le modèle et selon la vraie récompense.",
       "🔨 B8.25 · Ex B8.20 · fiche §208.7 (« Le modèle de récompense de Bradley-Terry », « Le RLHF », « Le piratage de "
       "la récompense »)",
       body=r"""Le RLHF de 208.7, en miniature et sans humain. Une **vraie récompense**, cachée à l'agent, préfère un chariot près du centre : $\exp(-(x/0{,}5)^2)$ à chaque pas (1 au centre, 0,37 à 0,5 m, presque 0 au-delà de 1 m). Des paires de trajectoires (100 pas au plus) sont classées par elle : la trajectoire qui en rapporte le plus est « préférée ». Deux jeux de préférences, de 200 paires chacun, plus 100 paires tenues à l'écart (`wbr.preference_pairs`) :
- **larges** (`PAIRS_26`, `HELD_26`) : des trajectoires de la règle d'une ligne de 208.1 avec 0 à 50 % d'actions au hasard, et de la politique aléatoire ; la règle ignore la position du chariot, si bien que certaines trajectoires restent au centre et d'autres s'en éloignent ;
- **étroites** (`NARROW_26`, `NARROW_HELD_26`) : des trajectoires de la politique aléatoire seulement, une vingtaine de pas près du centre.

Une paire est un dictionnaire de listes : `pairs["first"]` et `pairs["second"]`, des tableaux float32 `(T_i, 4)` (l'observation qui **suit** chaque pas), et `pairs["preferred"]`, 0 quand la première trajectoire est préférée, 1 sinon. **Contrat** des quatre fonctions de la cellule suivante :
- `make_reward_model_26(hidden=HIDDEN_26)` rend un `nn.Module` neuf dont `forward(obs)` envoie des observations `(N, 4)` sur des récompenses `(N,)` : un MLP 4 → `hidden` → `hidden` → 1 (tanh) appliqué aux observations divisées par `wbr.OBS_SCALE` ;
- `trajectory_returns_26(model, trajectories)` rend le retour appris de chaque trajectoire, la somme de `model` sur ses pas : un tenseur `(M,)` **avec son graphe** ;
- `train_reward_model_26(pairs, seed, epochs=EPOCHS_26, lr=LR_26)` : `torch.manual_seed(seed)`, puis `make_reward_model_26()` et Adam (`lr`) ; puis `epochs` pas sur **toutes** les paires à la fois (le batch complet) : les retours appris des deux listes, préféré et rejeté selon `pairs["preferred"]`, la loss et ses gradients par rapport aux retours par `mylearn.deep_rl.bradley_terry_loss` (ta fonction de B8.25, **appelée par ce nom** : la vérification l'observe), ces gradients poussés dans le réseau par `torch.autograd.backward`, et un pas d'Adam. Elle rend `(model, losses)`, `losses` la liste des loss de chaque epoch ;
- `reward_fn_26(model)` rend une fonction `(obs, actions, next_obs) -> ` tableau NumPy float `(N,)`, `model(next_obs)` sans gradient : la `reward_fn` de ton `train_ppo_20`.

La vérification essaie `make_reward_model_26`, compare `trajectory_returns_26` à la référence sur un modèle fixé, entraîne 20 epochs sur 30 paires (ta `bradley_terry_loss` appelée à chaque epoch, la loss de la première epoch, une loss qui baisse) et essaie `reward_fn_26`. Puis l'expérience (une minute environ) : tes deux modèles de récompense, leur accuracy sur les paires tenues à l'écart (au moins `THRESHOLD_26` attendu pour le modèle large), puis **quatre** agents de ton PPO, `STEPS_26` pas chacun : sur chaque modèle appris, sur la vraie récompense, et sur le +1 par pas de CartPole. Chacun est mesuré sur 20 épisodes déterministes : longueur, vraie récompense par pas, $|x|$ moyen, récompense apprise par pas. Deux figures : ce que chaque modèle a appris le long de $x$, et où vont les données et les agents.

Dans tes notes : le modèle étroit a-t-il une bonne accuracy sur **ses** paires tenues à l'écart ? Sur les paires larges ? Que récompense-t-il vraiment, et que fait l'agent qui l'optimise ? Compare, pour chaque agent, sa récompense apprise par pas et sa vraie récompense par pas : où vois-tu le piratage de la récompense, et pourquoi une récompense apprise ne se compare-t-elle qu'à elle-même (Bradley-Terry ne fixe que les différences) ? Ces chiffres viennent d'une seule graine : que faudrait-il pour les affirmer ? Qu'ajouterait une pénalité KL vers une politique de référence (208.7) ?""",
       given=fill(C2.GIVEN_26, VALUES), todo=C2.TODO_26, check=C2.CHECK_26,
       solution=C2.SOLUTION_26 + "\n\n\n" + C2.CHECK_26, after=[("code", C2.EXPERIMENT_26)]),

    ex("B8.27", "relire ppo.py de CleanRL ligne à ligne face à ton PPO, et documenter chaque écart : ce qu'il fait, ce que "
                "tu fais, l'effet attendu, et ta décision.",
       "Ex B8.20, B8.21 · 📄 B8.11 · fiche §208.4 (« Fin d'épisode ou limite de temps »), §208.5 (🕰️ Gymnasium 1.x), "
       "§208.6 (🕰️ les bibliothèques de référence, « Déboguer un agent qui n'apprend pas »)",
       body=r"""Une revue de code, comme en entreprise avant de fusionner une implémentation : la tienne (B8.20, ou la solution de `05_solutions.ipynb` si tu préfères partir d'elle) face à `cleanrl/ppo.py` de CleanRL (licence MIT), **au commit épinglé** de la fiche : la cellule donnée affiche le lien et télécharge le fichier (312 lignes) pour la vérification. Lis-le sur GitHub, avec la liste des « 37 détails d'implémentation » de Huang et coll. (2022, liens dans la fiche, 208.6) ; ne l'installe pas (il fige gymnasium 0.29 et Python < 3.11).

**Contrat** de `REVIEW_27` : une liste d'au moins **8 écarts**, chacun un dictionnaire de six clés :
- `"sujet"` : de quoi parle l'écart, en quelques mots ;
- `"lignes"` : le numéro de la ligne de `ppo.py`, ou un couple `(première, dernière)` ;
- `"extrait"` : au moins 8 caractères recopiés de ces lignes (à une ligne près ; la vérification les cherche dans le fichier) ;
- `"chez_moi"` : ce que fait ton PPO à la place (une phrase) ;
- `"effet"` : ce que l'écart change, et dans quel cas (une phrase) ;
- `"decision"` : `"garder"` (ta version), `"adopter"` (celle de CleanRL) ou `"tester"` (sur plusieurs graines, avant de choisir).

La revue doit au moins couvrir les deux écarts que la fiche signale : la ligne où `ppo.py` décide qu'un épisode est fini (208.4), et une ligne qui suppose l'autoreset same-step de gymnasium 0.29 (🕰️ Gymnasium 1.x, 208.5). Les autres sont à trouver : la loss du critique, les journaux, les mini-batches, les réglages, les graines… La cellule de vérification contrôle la **structure** (les clés, les lignes, les extraits, les décisions, les deux écarts de la fiche), écrit la revue en tableau Markdown dans `revue_ppo_cleanrl.md` (à côté de ton notebook) et l'affiche. Le **fond** (chaque écart est-il réel, son effet est-il juste ?) se corrige avec `05_solutions.md`.

Dans tes notes : quel écart changerait le plus un résultat sur CartPole, et lequel ne se verrait jamais ? Lesquels décides-tu de tester, et sur combien de graines ? Lequel des « 37 détails » ton PPO n'a-t-il pas, sans que ce soit un écart avec `ppo.py` ?""",
       given=C2.GIVEN_27, todo=C2.TODO_27, check=C2.CHECK_27, solution=C2.SOLUTION_27 + "\n\n\n" + C2.CHECK_27),

    ex("B8.28", "résoudre CartPole-v1 sur trois graines sur trois, avec un budget de pas compté par l'infrastructure et une "
                "évaluation sur des graines tenues à l'écart.",
       "🔬 B8.22 · Ex B8.20, B8.21 · fiche §208.6 (« Des hyperparamètres qui comptent », « Évaluer un agent »)",
       body=r"""**L'objectif** : CartPole-v1 **résolu**, c'est-à-dire au moins 475 de retour moyen sur 100 épisodes, avec **au plus 100 000 pas** d'environnement par run, sur **trois graines sur trois** (`SEEDS_28`).

**Contrat** de `train_28(seed, budget)` : entraîner un agent **à partir de rien** sur CartPole-v1, avec au plus `budget` pas d'environnement, puis rendre sa politique : un module torch (évalué par son action la plus probable : des logits, ou un agent qui a un `actor`) ou une fonction **déterministe** `(state, info) -> int`. Tu peux reprendre ton `train_ppo_20` avec d'autres réglages, ou écrire autre chose.

**Les règles**, appliquées par `wbr.run_challenge`, qui appelle `train_28(seed, BUDGET_28)` pour chaque graine :
- **tout pas compte** : chaque `step` de chaque CartPole créé pendant le run, quelle que soit la façon de le créer (`gym.make`, `gym.make_vec`, `wbr.make_env`, une copie), toutes copies comprises, tes propres évaluations comprises ; le pas qui dépasserait le budget arrête le run (« budget dépassé ») ; le mode `"async"`, qui joue dans d'autres processus, est refusé ;
- **à partir de rien** : pendant le run, les entraîneurs et les checkpoints de `wbr` et `torch.load` sont interdits, les agents du notebook (ton agent de B8.20, ceux de B8.26) sont masqués, et un agent rendu dont les paramètres sont ceux d'un agent qui existait avant le run est refusé ;
- **l'évaluation** : 100 épisodes déterministes de 500 pas au plus, sur des graines tenues à l'écart (aucun entraînement ni aucune autre évaluation du notebook ne s'en sert) ; un run compte s'il n'enfreint aucune règle ;
- **règle d'honneur** : n'écris pas ton propre simulateur de CartPole (ses équations sont publiques : ce serait un environnement hors du compteur) et ne contourne pas le compteur ; le budget est la mesure de l'efficacité en échantillons (208.6), pas une contrainte de temps.

La cellule de vérification lance les trois runs l'un après l'autre (avec les réglages de B8.20, environ 40 s chacun), affiche pour chacun le retour moyen, le plus court des 100 épisodes, le nombre de pas joués et le temps, puis vérifie par `wb.check` le nombre de graines résolues. Rien à taper.

Dans tes notes : tes réglages, et pourquoi (B8.22) ; le retour de chaque graine ; ce que tu aurais fait avec deux fois moins de pas.""",
       given=C2.GIVEN_28, todo=C2.TODO_28, solution=C2.SOLUTION_28, record=C2.RECORD_28,
       after=[("code", C2.CHECK_28)]),
])

for _ex in PART_C.exercises:
    for _field in ("given", "todo", "check", "solution", "record", "body"):
        setattr(_ex, _field, fill(getattr(_ex, _field), VALUES))
    _ex.after = [(kind, fill(text, VALUES)) for kind, text in _ex.after]

PARTS.append(PART_C)
