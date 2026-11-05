"""End of part B (the policy: B8.21 to B8.24) and part C (the preferences: B8.25 to B8.28) of the chapter B8 notebook,
second half: statements and cells (used by build_chB8.py).

B8.21 to B8.24 are appended to PART_B.exercises (chB8_parts.py); part C is appended to PARTS. The answers of 📈 B8.21
and 🔮 B8.23, the review lines of 🛠️ B8.27 and the thresholds of 📦 B8.26 come from tools/chapters/history_chB8.json
(entries "l21", "c23", "r26", "c28b").
"""

from __future__ import annotations

import json

from chapter_kit import Part, guarded

import chB8_code2 as C2
from chB8_parts import HISTORY, PART_B, PARTS, ex, fr_number, mylearn_howto, tests_after


def fill(text: str, values: dict) -> str:
    """Replace each «NAME» of a cell by its value; every placeholder must be filled (except «KIND», filled per
    notebook by build_chB8.py)."""
    for key, value in values.items():
        text = text.replace(f"«{key}»", str(value))
    left = text.replace("«KIND»", "")
    if "«" in left:
        raise ValueError(f"unfilled placeholder in {left[left.index('«'):left.index('«') + 30]!r}")
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
    "lr_high": "des pas trop grands éloignent chaque politique de la précédente : quelle courbe mesure cet écart ?",
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


C27 = HISTORY.get("c27", {})
REVIEW_SOLUTION_27 = C27.get("review", [])


VALUES = {"ANSWERS_21": ", ".join(f'"{LETTERS_21[c]}"' for c in CAUSES_21),
          "YSCALE_21": 'axes_21[0, 1].set_yscale("log")\n',
          "ANSWERS_23": f'{A23["a"]}, {A23["b"]}, {A23["c"]}', "RECORD_23": record_23(),
          "GRID_EPISODES_24": str(C23.get("protocol", {}).get("grid_episodes", 50)),
          "DONE_LINES_27": "206", "RESET_LINES_27": "162, 205, 210, 211",
          "REVIEW_27": json.dumps(REVIEW_SOLUTION_27, ensure_ascii=False, indent=4) if REVIEW_SOLUTION_27 else "[]"}


# =====================================================================================================================
PART_B.intro = PART_B.intro      # (the intro of part B already announces the journals, the settings and the robustness)
PART_B.exercises += [
    ex("B8.21", "reconnaître dans les journaux d'un PPO un learning rate trop grand, un clip absent, un bonus d'entropie de "
                "mauvais signe et un critique qui n'apprend pas.",
       "Ex B8.20 · fiche §208.4 (« Le critique et sa variance expliquée »), §208.5 (« Les journaux de PPO »)",
       body=r"""Cinq runs du PPO de référence du chapitre (`wbr.train_ppo`, 50 000 pas, la même graine), notés A à E dans un ordre tiré au hasard. L'un a les réglages de référence (`SETTINGS_20`) ; chacun des quatre autres en change **un seul** :
- un learning rate dix fois plus grand ($10^{-2}$), qui ne décroît jamais ;
- **pas de clipping** : un `clip_eps` si grand qu'aucun ratio n'est jamais rogné ;
- un bonus d'entropie **de mauvais signe** (`ent_coef = -0.05` : la loss récompense une politique sûre d'elle) ;
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
