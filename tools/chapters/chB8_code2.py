"""Code cells of the chapter B8 notebook, second half (end of part B, the policy: B8.21 to B8.24; part C, the
preferences: B8.25 to B8.28), shared by build_chB8.py (through chB8_parts2.py) and history_chB8.py (used by Claude).

Same conventions as chB8_code.py: GIVEN_* (both notebooks), TODO_* (exercise notebook), SOLUTION_* (solutions
notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions). Every expected value comes
from the reference of wb.deep_rl_env (``wbr``) or from a precomputed protocol (``wbr.precomputed``), never from another
exercise's code; the agent of B8.20 is read through ``load_agent_20`` (its explicit fallback: the reference agent).
Placeholders «NAME» are filled by chB8_parts2.py from tools/chapters/history_chB8.json.
"""

from chB8_code import RELOAD  # noqa: F401  (re-exported for chB8_parts2.py)

# =====================================================================================================================
# Part B · the policy (second half: B8.21 to B8.24)
# =====================================================================================================================
# --- 📈 B8.21 ---------------------------------------------------------------------------------------------------------
GIVEN_21 = r'''L21 = wbr.precomputed("l21")                 # five runs of the reference PPO, 50 000 steps each (protocol inside)
RUNS_21 = L21["runs"]                        # letter -> its log: one value per iteration, the keys of wbr.LOG_KEYS
PANELS_21 = (("mean_return", "retour (épisodes finis)"), ("approx_kl", "KL approchée"), ("clipfrac", "fraction clippée"),
             ("entropy", "entropie"), ("explained_variance", "variance expliquée"))


def plot_logs_21(axes, logs, **kwargs):
    """The five curves of a PPO log, one per panel, against the environment step (in thousands)."""
    steps = np.asarray(logs["step"]) / 1000
    for ax, (key, title) in zip(axes, PANELS_21):
        ax.plot(steps, np.asarray(logs[key], dtype=float), **kwargs)
        ax.set_title(title, fontsize=9)


fig_21, axes_21 = plt.subplots(len(RUNS_21), len(PANELS_21), figsize=(13, 1.9 * len(RUNS_21)), sharex=True,
                               sharey="col")
for row_21, letter_21 in zip(axes_21, sorted(RUNS_21)):
    plot_logs_21(row_21, RUNS_21[letter_21], color="C0", lw=1.2)
    row_21[0].set_ylabel(f"run {letter_21}", fontsize=11)
for ax_21 in axes_21[-1]:
    ax_21.set_xlabel("milliers de pas")
«YSCALE_21»plt.tight_layout()
plt.show()'''

TODO_21 = r'''# Your readings of the figure: one letter each, between quotes ("A" to "E")
healthy_21 = ...        # a) the run with the reference settings
lr_high_21 = ...        # b) a learning rate ten times larger (1e-2), which never decreases
no_clip_21 = ...        # c) no clipping (clip_eps so large that no ratio is ever clipped)
entropy_sign_21 = ...   # d) the entropy bonus with the wrong sign (ent_coef = -0.2)
no_critic_21 = ...      # e) a critic that does not learn (vf_coef = 0)'''

CHECK_21 = r'''answers_21 = [healthy_21, lr_high_21, no_clip_21, entropy_sign_21, no_critic_21]
for sub_21, answer_21 in zip("abcde", answers_21):
    wb.check(f"B8.21{sub_21}", answer_21)
letters_21 = [str(a).strip().upper() for a in answers_21 if filled(a)]
if len(set(letters_21)) < len(letters_21):
    print("ℹ️ Ex B8.21 : deux de tes réponses ont la même lettre ; chaque run n'a qu'un seul réglage.")'''

SOLUTION_21 = r'''healthy_21, lr_high_21, no_clip_21, entropy_sign_21, no_critic_21 = «ANSWERS_21»'''

EXPERIMENT_21 = r'''# Your PPO of B8.20 read the same way: its five curves (this session's log, or the one saved with your agent)
logs_own_21 = globals().get("LOGS_20")
if logs_own_21 is None:
    try:
        loaded_21 = wbr.load_agent(AGENT_KEY_20, build=ActorCritic20)
    except NotImplementedError:
        loaded_21 = None
    logs_own_21 = loaded_21[1]["logs"] if loaded_21 is not None else None
if logs_own_21 is None or not all(k in logs_own_21 for k, _ in PANELS_21):
    print("⏳ Ex B8.21 : pas de journal de ton PPO (B8.20 n'a pas encore tourné, ou ses journaux n'ont pas les clés de "
          "wbr.LOG_KEYS) : la figure ci-dessus suffit pour l'exercice.")
else:
    fig_own_21, axes_own_21 = plt.subplots(1, len(PANELS_21), figsize=(13, 2.2))
    plot_logs_21(axes_own_21, logs_own_21, color="C1", lw=1.2)
    axes_own_21[0].set_ylabel("ton PPO", fontsize=11)
    plt.tight_layout()
    plt.show()'''

# --- 🔬 B8.22 ---------------------------------------------------------------------------------------------------------
GIVEN_22 = r'''P22 = wbr.precomputed("p22")                   # the reference PPO: 7 settings x 3 seeds x 50 000 steps (protocol inside)
CONFIGS_22 = P22["protocol"]["configs"]        # name -> what it changes in SETTINGS_20 ("reference": nothing)
SEEDS_22 = P22["protocol"]["seeds"]
GROUPS_22 = {"ε du clip": ("clip_0.1", "reference", "clip_0.4"), "λ de GAE": ("lambda_0.8", "reference", "lambda_1"),
             "epochs": ("epochs_4", "reference", "epochs_20")}
REPLAY_ITERATIONS_22 = 4                       # the prefix of one recorded run that this notebook trains again
print(CONFIGS_22)'''

TODO_22 = r'''def summary_22(logs):
    """Two numbers of a PPO log (a dict of lists, one value per iteration, with at least "step" and "mean_return"):
    "auc", the mean of the finite values of logs["mean_return"]; "first_475", the first logs["step"] whose
    logs["mean_return"] is at least 475 (an int), or None if there is none. The log is left unchanged."""
    raise NotImplementedError("summary_22")'''

SOLUTION_22 = r'''def summary_22(logs):
    """Two numbers of a PPO log (a dict of lists, one value per iteration, with at least "step" and "mean_return"):
    "auc", the mean of the finite values of logs["mean_return"]; "first_475", the first logs["step"] whose
    logs["mean_return"] is at least 475 (an int), or None if there is none. The log is left unchanged."""
    returns = np.asarray(logs["mean_return"], dtype=float)          # an iteration without a finished episode: nan
    finite = returns[np.isfinite(returns)]
    first = next((int(step) for step, r in zip(logs["step"], returns) if r >= 475), None)   # nan >= 475 is False
    return {"auc": float(finite.mean()), "first_475": first}'''

CHECK_22 = r'''ok_22 = False
with checked("B8.22"):
    nan_22 = float("nan")
    cases_22 = {"un journal avec des itérations sans épisode fini (nan)": {"step": [512, 1024, 1536, 2048, 2560],
                                                                          "mean_return": [20.0, nan_22, 480.0, nan_22, 500.0]},
                "un journal qui n'atteint jamais 475": {"step": [512, 1024, 1536], "mean_return": [30.0, 474.9, nan_22]},
                "un journal qui atteint 475 tout juste, puis retombe": {"step": [512, 1024, 1536],
                                                                      "mean_return": [100.0, 475.0, 300.0]},
                f"le run précalculé « reference », graine {SEEDS_22[0]}": P22["runs"][f"reference/{SEEDS_22[0]}"]}
    good_22 = []
    for label_22, logs_22 in cases_22.items():
        given_22 = copy.deepcopy(logs_22)
        got_22 = summary_22(given_22)
        want_22 = wbr.log_summary(logs_22)
        if not returned("B8.22", "summary_22", got_22):
            good_22.append(False)
            continue
        if not isinstance(got_22, dict) or set(got_22) != {"auc", "first_475"}:
            print(f"❌ Ex B8.22 : summary_22 doit rendre un dictionnaire aux clés 'auc' et 'first_475', reçu {got_22!r}.")
            good_22.append(False)
            continue
        try:
            auc_22 = float(got_22["auc"])
        except (TypeError, ValueError):
            auc_22 = nan_22
        good_22.append(abs(auc_22 - want_22["auc"]) <= 1e-9 * max(1.0, abs(want_22["auc"])))
        if not good_22[-1]:
            print(f"❌ Ex B8.22 : summary_22, auc : attendu {want_22['auc']:.6g}, reçu {got_22['auc']!r} ({label_22}) ; "
                  "une itération sans épisode fini (nan) ne compte pas dans la moyenne.")
        first_22 = got_22["first_475"]
        good_22.append(first_22 == want_22["first_475"] and (first_22 is None or isinstance(first_22, (int, np.integer))))
        if not good_22[-1]:
            print(f"❌ Ex B8.22 : summary_22, first_475 : attendu {want_22['first_475']!r}, reçu {first_22!r} ({label_22}) "
                  "; c'est le premier logs['step'] dont le retour atteint 475 (475 compris), un entier, None s'il n'y "
                  "en a pas.")
        if not same_log_22(given_22, logs_22):
            print(f"❌ Ex B8.22 : summary_22 a modifié le journal qu'on lui donne ({label_22}).")
            good_22.append(False)
    ok_22 = all(good_22)
    verdict("B8.22", ok_22, f"summary_22 rend l'aire et le premier pas à 475 sur les {len(cases_22)} journaux d'essai.",
            "summary_22 ne rend pas encore ce qu'attend la suite : corrige les lignes ci-dessus.")'''

GIVEN_22 += r'''


def same_log_22(a, b):
    """True when two logs hold the same lists (nan equal to nan)."""
    return a.keys() == b.keys() and all(np.array_equal(np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float),
                                                       equal_nan=True) for k in a)'''

EXPERIMENT_22 = r'''# The 7 settings on three seeds (fast mode: the reference PPO, precomputed, after replaying a prefix; full mode: your
# PPO of B8.20 on every setting and seed, 🚀), then the median and the range of the three seeds, and your summary_22
protocol_22 = P22["protocol"]
if FAST_MODE:
    runs_22 = P22["runs"]
    _, replay_22 = wbr.train_ppo(SEEDS_22[0], protocol_22["steps"], {**SETTINGS_20, **CONFIGS_22["reference"]},
                                 callback=lambda iteration, agent, logs: iteration >= REPLAY_ITERATIONS_22)
    recorded_22 = runs_22[f"reference/{SEEDS_22[0]}"]["approx_kl"][:REPLAY_ITERATIONS_22]
    replayed_22 = [round(v, 4) for v in replay_22["approx_kl"]]
    if np.allclose(replayed_22, recorded_22, atol=2e-4):
        print(f"✅ Ex B8.22 : « reference », graine {SEEDS_22[0]}, rejoué sur {REPLAY_ITERATIONS_22} itérations : KL "
              f"approchée {replayed_22}, comme dans le fichier : les courbes ci-dessous sont celles de ce code.")
    else:
        print(f"ℹ️ « reference », graine {SEEDS_22[0]}, rejoué sur {REPLAY_ITERATIONS_22} itérations : KL approchée "
              f"{replayed_22}, le fichier dit {recorded_22} : une autre machine arrondit autrement, et un écart minuscule "
              "change la suite d'un run de RL (208.6). Les courbes ci-dessous restent celles du fichier.")
elif not (ok_22 and ok_20):
    runs_22 = None
    print("⏳ Ex B8.22 : le mode complet entraîne ton PPO de B8.20 sur chaque réglage ; il attend que les vérifications de "
          "B8.20 et de B8.22 passent.")
else:
    runs_22 = {}
    for name_22, change_22 in CONFIGS_22.items():
        for seed_22 in SEEDS_22:
            start_22 = time.perf_counter()
            agent_22, logs_22 = train_ppo_20(seed_22, STEPS_20, {**SETTINGS_20, **change_22})
            runs_22[f"{name_22}/{seed_22}"] = {**{k: list(logs_22[k]) for k in ("step", "mean_return", "approx_kl",
                                                                                "clipfrac")},
                                              "eval": wbr.evaluate(wbr.greedy_logits(agent_22), 100, EVAL_SEED)["mean_return"]}
            print(f"{name_22}, graine {seed_22} : {time.perf_counter() - start_22:.0f} s")
if runs_22 and ok_22:
    def filled_curve_22(values):
        """The rollout returns with each nan (no finished episode) replaced by the last value seen, for the figure."""
        out, last = [], np.nan
        for v in np.asarray(values, dtype=float):
            last = v if np.isfinite(v) else last
            out.append(last)
        return np.array(out)

    fig_22, axes_22 = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax_22, (group_22, names_22) in zip(axes_22, GROUPS_22.items()):
        for name_22 in names_22:
            hs_22 = [runs_22[f"{name_22}/{s}"] for s in SEEDS_22]
            b_22 = wbr.band([filled_curve_22(h["mean_return"]) for h in hs_22])
            x_22 = np.asarray(hs_22[0]["step"]) / 1000
            line_22, = ax_22.plot(x_22, b_22["median"], label=name_22, lw=2 if name_22 == "reference" else 1.2)
            ax_22.fill_between(x_22, b_22["low"], b_22["high"], color=line_22.get_color(), alpha=0.12)
        ax_22.axhline(wbr.SOLVED, color="gray", ls=":", lw=1)
        ax_22.set(title=group_22, xlabel="milliers de pas")
        ax_22.legend(fontsize=8)
    axes_22[0].set_ylabel("retour des rollouts")
    fig_22.suptitle("Médiane et intervalle (min-max) sur trois graines", fontsize=10)
    plt.tight_layout()
    plt.show()
    print(f"{'réglage':11s} | aire par graine (médiane) | 1er pas à 475 par graine | évaluation déterministe par graine")
    for name_22 in CONFIGS_22:
        hs_22 = [runs_22[f"{name_22}/{s}"] for s in SEEDS_22]
        sums_22 = [summary_22(h) for h in hs_22]
        aucs_22 = [s["auc"] for s in sums_22]
        firsts_22 = ["jamais" if s["first_475"] is None else f"{round(s['first_475'] / 1000)} k" for s in sums_22]
        evals_22 = [round(h["eval"]) for h in hs_22]
        print(f"{name_22:11s} | {[round(a) for a in aucs_22]} ({fr(float(np.median(aucs_22)), 0)}) | {firsts_22} | "
              f"{evals_22}")'''

# =====================================================================================================================
# Part C · the preferences (B8.25 to B8.28)
# =====================================================================================================================
# --- 🔨 B8.25 ---------------------------------------------------------------------------------------------------------
CHECK_25 = r'''with checked("B8.25"):
    loss_25, gc_25, gr_25 = pair("B8.25", "bradley_terry_loss", mylearn.deep_rl.bradley_terry_loss([0.0], [0.0]),
                                 names=("loss", "grad_chosen", "grad_rejected"), floats=("loss",))
    same("B8.25", "bradley_terry_loss([0.0], [0.0]), la loss (log 2)", loss_25, math.log(2))
    same("B8.25", "bradley_terry_loss([0.0], [0.0]), les gradients", np.concatenate([np.ravel(gc_25), np.ravel(gr_25)]),
         [-0.5, 0.5])
    rng_25 = np.random.default_rng(25)
    rc_25, rr_25 = rng_25.normal(0, 3, 7), rng_25.normal(0, 3, 7)
    rc_t, rr_t = torch.tensor(rc_25, requires_grad=True), torch.tensor(rr_25, requires_grad=True)
    want_25 = -F.logsigmoid(rc_t - rr_t).mean()
    want_25.backward()
    loss_25, gc_25, gr_25 = pair("B8.25", "bradley_terry_loss", mylearn.deep_rl.bradley_terry_loss(rc_25.copy(),
                                                                                                    rr_25.copy()),
                                 names=("loss", "grad_chosen", "grad_rejected"), floats=("loss",))
    same("B8.25", "7 paires : la loss contre -F.logsigmoid(...).mean()", loss_25, want_25.item())
    same("B8.25", "7 paires : grad_chosen contre l'autograd (la loss MOYENNE)", gc_25, rc_t.grad.numpy())
    same("B8.25", "7 paires : grad_rejected contre l'autograd", gr_25, rr_t.grad.numpy())
    with np.errstate(all="ignore"):
        loss_25, gc_25, gr_25 = pair("B8.25", "bradley_terry_loss", mylearn.deep_rl.bradley_terry_loss(
            [-3000.0, 4000.0], [3000.0, -4000.0]), names=("loss", "grad_chosen", "grad_rejected"), floats=("loss",))
    same("B8.25", "écarts de -6 000 et +8 000 : la loss (finie, exacte)", loss_25, 3000.0)
    same("B8.25", "écarts de -6 000 et +8 000 : grad_chosen (fini, exact)", gc_25, [-0.5, 0.0])

with checked("B8.25"):
    loss_25, margins_25 = pair("B8.25", "dpo_loss", mylearn.deep_rl.dpo_loss([-10.0, -12.0], [-11.0, -9.0],
                                                                            [-10.5, -11.0], [-10.5, -10.0], beta=0.1),
                               names=("loss", "margins"), floats=("loss",))
    same("B8.25", "l'exemple de la docstring de dpo_loss (marges, avec β)", margins_25, [0.1, -0.2])
    rng_25 = np.random.default_rng(250)
    logps_25 = [rng_25.normal(-40, 5, 6) for _ in range(4)]
    for beta_25 in (0.1, 0.5):
        pc, pr, rc, rr = (torch.tensor(a) for a in logps_25)
        margin_t = beta_25 * ((pc - rc) - (pr - rr))
        loss_25, margins_25 = pair("B8.25", "dpo_loss", mylearn.deep_rl.dpo_loss(*[a.copy() for a in logps_25],
                                                                                beta=beta_25),
                                   names=("loss", "margins"), floats=("loss",))
        same("B8.25", f"6 paires, β = {fr(beta_25, 1)} : la loss contre l'équation 7 (torch)", loss_25,
             (-F.logsigmoid(margin_t).mean()).item())
        same("B8.25", f"6 paires, β = {fr(beta_25, 1)} : les marges", margins_25, margin_t.numpy())
    refuses("B8.25", "dpo_loss(..., beta=0)", lambda: mylearn.deep_rl.dpo_loss([0.0], [0.0], [0.0], [0.0], beta=0.0))
    bt_25 = mylearn.deep_rl.bradley_terry_loss([2.3], [1.1])
    dpo_25 = mylearn.deep_rl.dpo_loss([-35.2], [-38.9], [-36.0], [-37.5])
    print(f"✏️ B8.6 avec tes fonctions : loss {float(bt_25[0]):.4f} · gradient {float(np.ravel(bt_25[1])[0]):.4f} · "
          f"marge DPO (β = 0,1) {float(np.ravel(dpo_25[1])[0]):.4f} ← compare à tes réponses de B8.6")'''

# --- 🛠️ B8.27 ---------------------------------------------------------------------------------------------------------
GIVEN_27 = r'''import tempfile
from pathlib import Path

from IPython.display import Markdown, display

CLEANRL_LINES_27 = wbr.cleanrl_ppo_lines()        # ppo.py of CleanRL at the commit of the fiche (None without network)
FIELDS_27 = ("sujet", "lignes", "extrait", "chez_moi", "effet", "decision")
DECISIONS_27 = ("garder", "adopter", "tester")
print(f"CleanRL ppo.py, commit {wbr.CLEANRL_COMMIT[:7]} : {wbr.CLEANRL_PPO_PAGE}")
print("fichier relu :" if CLEANRL_LINES_27 else "ℹ️ fichier non téléchargé (pas de réseau ?) : lis-le sur GitHub ; les "
      "extraits ne seront pas comparés aux lignes.", f"{len(CLEANRL_LINES_27)} lignes" if CLEANRL_LINES_27 else "")


def line_range_27(lines):
    """(first, last) of the "lignes" field: an int, or a pair (first, last); None if it is neither."""
    if isinstance(lines, (int, np.integer)) and not isinstance(lines, bool):
        return int(lines), int(lines)
    if (isinstance(lines, (tuple, list)) and len(lines) == 2 and all(isinstance(v, (int, np.integer))
                                                                     and not isinstance(v, bool) for v in lines)):
        return int(min(lines)), int(max(lines))
    return None


def check_review_27(review):
    """The problems of a review (a list of dicts with the keys FIELDS_27); an empty list when its structure is right.
    The content (is each gap real, is its effect right?) is corrected with 05_solutions.md."""
    problems = []
    if not isinstance(review, (list, tuple)) or not all(isinstance(e, dict) for e in review):
        return ["REVIEW_27 doit être une liste de dictionnaires (un par écart)."]
    if len(review) < 8:
        problems.append(f"la revue compte {len(review)} écarts : il en faut au moins 8.")
    cited = set()
    for k, entry in enumerate(review, start=1):
        where = f"écart {k} ({str(entry.get('sujet', '?'))[:40]})"
        missing = [f for f in FIELDS_27 if f not in entry]
        extra = [f for f in entry if f not in FIELDS_27]
        if missing or extra:
            problems.append(f"{where} : attendu les clés {list(FIELDS_27)}"
                            + (f", il manque {missing}" if missing else "") + (f", en trop {extra}" if extra else "") + ".")
            continue
        for f in ("sujet", "chez_moi", "effet"):
            if not isinstance(entry[f], str) or len(entry[f].split()) < (2 if f == "sujet" else 6):
                problems.append(f"{where} : « {f} » doit être une phrase en français ("
                                + ("au moins 2 mots" if f == "sujet" else "au moins 6 mots") + ").")
        span = line_range_27(entry["lignes"])
        if span is None or not (1 <= span[0] <= span[1] <= wbr.CLEANRL_PPO_LINES):
            problems.append(f"{where} : « lignes » doit être un numéro de ligne de ppo.py (1 à {wbr.CLEANRL_PPO_LINES}) "
                            "ou un couple (première, dernière).")
            span = None
        if not isinstance(entry["extrait"], str) or len(entry["extrait"].strip()) < 8:
            problems.append(f"{where} : « extrait » doit recopier au moins 8 caractères du code de ces lignes.")
        elif span and CLEANRL_LINES_27:
            text = "\n".join(CLEANRL_LINES_27[max(0, span[0] - 2):span[1] + 1])
            if " ".join(entry["extrait"].split()) not in " ".join(text.split()):
                problems.append(f"{where} : l'extrait « {entry['extrait'].strip()[:50]} » n'est pas dans les lignes "
                                f"{span[0]} à {span[1]} de ppo.py (à une ligne près).")
        if entry["decision"] not in DECISIONS_27:
            problems.append(f"{where} : « decision » vaut {entry['decision']!r} ; attendu l'un de {list(DECISIONS_27)}.")
        if span:
            cited |= set(range(span[0], span[1] + 1))
    if not problems:
        if not cited & {«DONE_LINES_27»}:
            problems.append("la revue ne cite pas la ligne où ppo.py décide qu'un épisode est fini (208.4, « Fin "
                            "d'épisode ou limite de temps »).")
        if not cited & {«RESET_LINES_27»}:
            problems.append("la revue ne cite aucune ligne qui suppose l'autoreset same-step de gymnasium 0.29 "
                            "(🕰️ Gymnasium 1.x, 208.5).")
    return problems


def review_markdown_27(review):
    """The review as a Markdown table (for your notes, your README, a pull request)."""
    rows = ["| # | Sujet | ppo.py | Chez moi | Effet | Décision |", "|---|---|---|---|---|---|"]
    for k, e in enumerate(review, start=1):
        span = line_range_27(e["lignes"])
        where = f"l. {span[0]}" if span and span[0] == span[1] else (f"l. {span[0]}-{span[1]}" if span else "?")
        cells = [str(k), e["sujet"], f"{where} : `{e['extrait'].strip()}`", e["chez_moi"], e["effet"], e["decision"]]
        rows.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    return "\n".join(rows)'''

TODO_27 = r'''# Your review: one dict per gap between your PPO (B8.20) and CleanRL's ppo.py, at least 8. A filled example, which
# you may keep (it does not count among the two gaps that the check asks for):
REVIEW_27 = [
    {"sujet": "epsilon d'Adam",
     "lignes": 168,
     "extrait": "eps=1e-5",
     "chez_moi": "SETTINGS_20 règle aussi adam_eps à 1e-5, passé à torch.optim.Adam",
     "effet": "aucun écart : un epsilon plus grand que le défaut limite la taille des pas quand le second moment est petit",
     "decision": "garder"},
    # ... your gaps, with the same six keys
]'''

SOLUTION_27 = r'''REVIEW_27 = «REVIEW_27»'''

CHECK_27 = r'''if REVIEW_27 is ... or (isinstance(REVIEW_27, list) and len(REVIEW_27) <= 1):
    print("⏳ Ex B8.27 : pas encore fait (complète REVIEW_27).")
else:
    problems_27 = check_review_27(REVIEW_27)
    for problem_27 in problems_27:
        print(f"❌ Ex B8.27 : {problem_27}")
    if not problems_27:
        decisions_27 = {d: sum(e["decision"] == d for e in REVIEW_27) for d in DECISIONS_27}
        print(f"✅ Ex B8.27 : {len(REVIEW_27)} écarts, chacun avec ses lignes, son extrait, ton code, son effet et une "
              f"décision ({decisions_27}) ; les deux écarts de la fiche sont cités. Le fond se corrige avec "
              "05_solutions.md.")
        review_path_27 = (Path("revue_ppo_cleanrl.md") if "mon_travail" in str(Path.cwd())   # next to your notebook
                          else Path(tempfile.mkdtemp(prefix="wb_b8_27_")) / "revue_ppo_cleanrl.md")
        review_path_27.write_text(f"# Revue de code : mon PPO face à CleanRL ppo.py ({wbr.CLEANRL_COMMIT[:7]})\n\n"
                                  + review_markdown_27(REVIEW_27) + "\n", encoding="utf-8")
        print(f"   écrite dans {review_path_27.name}" + (", à côté de ton notebook." if "mon_travail" in str(Path.cwd())
                                                         else "."))
        display(Markdown(review_markdown_27(REVIEW_27)))'''

# --- 🏆 B8.28 ---------------------------------------------------------------------------------------------------------
GIVEN_28 = r'''BUDGET_28 = wbr.CHALLENGE_BUDGET              # 100 000 environment steps per run, counted by the infrastructure
SEEDS_28 = wbr.CHALLENGE_SEEDS                 # the three seeds of the challenge (none of them trains anything else)
print(f"défi : {len(SEEDS_28)} runs (graines {list(SEEDS_28)}), au plus {BUDGET_28} pas d'environnement chacun, puis "
      f"100 épisodes déterministes sur des graines tenues à l'écart ; objectif : {wbr.SOLVED:g} de moyenne sur les "
      "trois runs.")'''

TODO_28 = r'''def train_28(seed, budget):
    """Train an agent on CartPole-v1 from scratch, with at most `budget` environment steps (every copy of every
    CartPole you create counts, your own evaluations too), and return its policy: a torch module (evaluated by its
    most probable action: logits, or an agent with an `actor`) or a deterministic function (state, info) -> int."""
    raise NotImplementedError("train_28")'''

SOLUTION_28 = r'''SETTINGS_28 = {**SETTINGS_20}          # the reference settings of B8.20 (10 epochs, lr 1e-3 decreasing to 0): see 05


def train_28(seed, budget):
    """Train an agent on CartPole-v1 from scratch, with at most `budget` environment steps (every copy of every
    CartPole you create counts, your own evaluations too), and return its policy: a torch module (evaluated by its
    most probable action: logits, or an agent with an `actor`) or a deterministic function (state, info) -> int."""
    agent, _ = train_ppo_20(seed, budget, SETTINGS_28)   # n_iter = budget // 512 iterations: 99 840 steps at most
    return agent'''

CHECK_28 = r'''results_28 = []
for k_28, seed_28 in enumerate(SEEDS_28):
    with checked("B8.28"):
        result_28 = wbr.run_challenge(train_28, seed_28, budget=BUDGET_28, namespace=globals())
        results_28.append(result_28)
        if result_28["error"]:
            print(f"❌ Ex B8.28 : graine {seed_28} : le run ne compte pas : {result_28['error']} "
                  f"({result_28['steps']} pas joués en {result_28['seconds']:.0f} s).")
        else:
            verdict("B8.28", result_28["solved"],
                    f"graine {seed_28} : {fr(result_28['mean_return'], 1)} sur 100 épisodes (le plus court : "
                    f"{result_28['min_return']:.0f} pas), {result_28['steps']} pas d'environnement, "
                    f"{result_28['seconds']:.0f} s.",
                    f"graine {seed_28} : {fr(result_28['mean_return'], 1)} sur 100 épisodes, sous {wbr.SOLVED:g} "
                    f"({result_28['steps']} pas d'environnement, {result_28['seconds']:.0f} s).")
    if len(results_28) <= k_28:            # train_28 not written yet (⏳), or it raised an error (❌): stop there
        break
if len(results_28) == len(SEEDS_28):
    solved_28 = sum(bool(r["solved"]) and r["error"] is None for r in results_28)
    wb.check("B8.28", solved_28, computed=True)'''

RECORD_28 = r'''wb.record("B8.28", 3, choices=[0, 1, 2, 3], mistakes={
    "deux runs sur trois seulement : un réglage qui réussit sur une graine doit tenir sur toutes (relis B8.22)": 2,
    "un run sur trois seulement : regarde la courbe de chaque graine, et relis B8.22": 1,
    "aucun run ne tient 475 : vérifie d'abord ton PPO avec les réglages de B8.20, sur ce budget": 0},
    fractional="la réponse est un nombre de graines, de 0 à 3")'''

# --- 🔮 B8.23 ---------------------------------------------------------------------------------------------------------
TODO_23 = r'''# Your predictions, BEFORE running the next cell (integers)
gravity_23 = ...   # a) gravity doubled: 0 (the median of the six agents stays >= 475), 1 (between 200 and 475), 2 (below 200)
length_23 = ...    # b) a pole twice as long: 0, 1 or 2 (the same bands)
longer_23 = ...    # c) a pole four times as long: how many of the six agents still hold 475? An int from 0 to 6'''

SOLUTION_23 = r'''gravity_23, length_23, longer_23 = «ANSWERS_23»'''

EXPERIMENT_23 = r'''C23 = wbr.precomputed("c23")                  # six reference PPO agents (the protocol of B8.20, seeds 0 to 5)
PHYSICS_23 = C23["protocol"]["physics"]        # name -> the constants it changes (length is HALF the pole)
NAMES_23 = {"normal": "CartPole-v1", "gravity_x2": "gravité doublée", "length_x2": "mât deux fois plus long",
            "length_x4": "mât quatre fois plus long"}
seeds_eval_23 = C23["protocol"]["eval_seeds"]
print(f"Les six agents de référence, 100 épisodes déterministes (graines d'évaluation {seeds_eval_23[0]}, puis "
      f"{seeds_eval_23[1]}) :")
for name_23 in PHYSICS_23:
    means_23 = [row[f"{name_23}/{seeds_eval_23[0]}"]["mean"] for row in C23["rows"]]
    other_23 = [row[f"{name_23}/{seeds_eval_23[1]}"]["mean"] for row in C23["rows"]]
    held_23 = sum(m >= wbr.SOLVED for m in means_23)
    print(f"  {NAMES_23[name_23]:26s} {[round(m) for m in means_23]} · médiane {fr(float(np.median(means_23)), 1)} · "
          f"{held_23} sur 6 au-dessus de {wbr.SOLVED:g} · autres graines : médiane {fr(float(np.median(other_23)), 1)}")
agent_23, source_23 = load_agent_20()
print(f"\nTon agent de B8.20 (source {source_23!r}), sur les mêmes 100 épisodes :")
for name_23, change_23 in PHYSICS_23.items():
    ev_23 = wbr.evaluate(agent_23, 100, seeds_eval_23[0], **change_23)
    print(f"  {NAMES_23[name_23]:26s} {fr(ev_23['mean_return'], 1)} (le plus court : {min(ev_23['returns']):.0f} pas)")
wb.check("B8.23a", gravity_23)
wb.check("B8.23b", length_23)
wb.check("B8.23c", longer_23)'''

RECORD_23 = r'''«RECORD_23»'''

# --- 🔬 B8.24 ---------------------------------------------------------------------------------------------------------
GIVEN_24 = r'''GRID_24 = wbr.precomputed("c23")               # the six reference agents of B8.23, evaluated on the grids below
FACTORS_24 = {"gravity": GRID_24["protocol"]["gravity_factors"], "length": GRID_24["protocol"]["length_factors"]}
DEFAULTS_24 = {"gravity": 9.8, "length": 0.5}  # CartPole-v1: g in m/s², length = HALF the pole, in m
EPISODES_24 = 20                               # episodes per point for your agent (the reference: «GRID_EPISODES_24»)
print(FACTORS_24)'''

TODO_24 = r'''def set_physics_24(env, gravity=None, length=None):
    """Give a CartPole made by gym.make another gravity and/or another half-length of the pole (None: unchanged), on
    env.unwrapped, with every constant that CartPoleEnv.__init__ derives from them kept consistent; returns env."""
    raise NotImplementedError("set_physics_24")


def robustness_24(policy, name, factors, n_episodes=EPISODES_24, seed=EVAL_SEED):
    """The mean return of `policy` on CartPole-v1 whose `name` ("gravity" or "length") is its default value
    (DEFAULTS_24) times each factor: a list of floats, one per factor, each wbr.evaluate(policy, n_episodes, seed,
    env=env)["mean_return"] with env a new wbr.make_env() changed by set_physics_24."""
    raise NotImplementedError("robustness_24")'''

SOLUTION_24 = r'''def set_physics_24(env, gravity=None, length=None):
    """Give a CartPole made by gym.make another gravity and/or another half-length of the pole (None: unchanged), on
    env.unwrapped, with every constant that CartPoleEnv.__init__ derives from them kept consistent; returns env."""
    core = env.unwrapped
    if gravity is not None:
        core.gravity = float(gravity)
    if length is not None:
        core.length = float(length)
    core.total_mass = core.masspole + core.masscart          # computed once by __init__...
    core.polemass_length = core.masspole * core.length      # ...and used by step(): recompute both
    return env


def robustness_24(policy, name, factors, n_episodes=EPISODES_24, seed=EVAL_SEED):
    """The mean return of `policy` on CartPole-v1 whose `name` ("gravity" or "length") is its default value
    (DEFAULTS_24) times each factor: a list of floats, one per factor, each wbr.evaluate(policy, n_episodes, seed,
    env=env)["mean_return"] with env a new wbr.make_env() changed by set_physics_24."""
    out = []
    for factor in factors:
        env = set_physics_24(wbr.make_env(), **{name: DEFAULTS_24[name] * factor})
        out.append(float(wbr.evaluate(policy, n_episodes, seed, env=env)["mean_return"]))
    return out'''

CHECK_24 = r'''ok_24 = False
with checked("B8.24"):
    good_24 = []
    for label_24, change_24 in (("gravité doublée", {"gravity": 19.6}), ("mât deux fois plus long", {"length": 1.0}),
                                ("gravité et mât changés ensemble", {"gravity": 4.9, "length": 0.25}),
                                ("rien de changé", {})):
        env_24 = wbr.make_env()
        out_24 = set_physics_24(env_24, **change_24)
        if not returned("B8.24", "set_physics_24", out_24):
            good_24.append(False)
            continue
        got_24, want_24 = wbr.physics_of(env_24), wbr.physics_of(wbr.make_env(**change_24))
        wrong_24 = {k: (want_24[k], got_24[k]) for k in want_24 if not math.isclose(got_24[k], want_24[k], rel_tol=1e-12)}
        good_24.append(not wrong_24 and out_24 is env_24)
        if wrong_24:
            print(f"❌ Ex B8.24 : set_physics_24, {label_24} : attendu "
                  + ", ".join(f"{k} = {w:g}" for k, (w, _) in wrong_24.items()) + " ; reçu "
                  + ", ".join(f"{k} = {g:g}" for k, (_, g) in wrong_24.items()) + ".")
            continue
        if out_24 is not env_24:
            print(f"❌ Ex B8.24 : set_physics_24 doit rendre l'environnement qu'il a reçu (env), reçu un autre objet.")
            continue
        ref_24 = wbr.make_env(**change_24)                 # the same episode, the same actions: the same states?
        a_24, _ = env_24.reset(seed=24)
        b_24, _ = ref_24.reset(seed=24)
        states_24 = [(a_24, b_24)]
        for t_24 in range(40):
            action_24 = int(t_24 % 3 == 0)
            a_24, _, ta_24, _, _ = env_24.step(action_24)
            b_24, _, tb_24, _, _ = ref_24.step(action_24)
            states_24.append((a_24, b_24))
            if ta_24 or tb_24:
                break
        moves_24 = all(np.allclose(x, y, atol=1e-6) for x, y in states_24)
        good_24.append(moves_24)
        verdict("B8.24", moves_24, f"set_physics_24, {label_24} : les constantes et un épisode de {len(states_24) - 1} "
                                   "pas sont ceux de la référence.",
                f"set_physics_24, {label_24} : un épisode joué avec les mêmes actions s'écarte de la référence.")
    rule_24 = wbr.rule_policy
    for name_24 in ("gravity", "length"):
        got_24 = robustness_24(rule_24, name_24, [1.0, 2.0], n_episodes=5, seed=2424)
        want_24 = [wbr.evaluate(rule_24, 5, 2424, **{name_24: DEFAULTS_24[name_24] * f})["mean_return"] for f in (1.0, 2.0)]
        good_24.append(same("B8.24", f"robustness_24(règle de 208.1, {name_24!r}, [1, 2], 5 épisodes)", got_24, want_24))
    ok_24 = all(good_24)'''

EXPERIMENT_24 = r'''# Your agent of B8.20 on the two grids (EPISODES_24 episodes per point), against the six reference agents of B8.23
if not ok_24:
    print("⏳ Ex B8.24 : l'expérience attend que la vérification ci-dessus passe.")
else:
    with checked("B8.24"):
        agent_24, source_24 = load_agent_20()
        start_24 = time.perf_counter()
        mine_24 = {name: robustness_24(agent_24, name, factors) for name, factors in FACTORS_24.items()}
        fig_24, axes_24 = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
        for ax_24, (name_24, factors_24) in zip(axes_24, FACTORS_24.items()):
            b_24 = wbr.band([GRID_24["grid"][f"{name_24}/{s}"] for s in GRID_24["protocol"]["seeds"]])
            ax_24.plot(factors_24, b_24["median"], color="gray", label="six agents de référence (médiane)")
            ax_24.fill_between(factors_24, b_24["low"], b_24["high"], color="gray", alpha=0.15)
            ax_24.plot(factors_24, mine_24[name_24], marker="o", color="C1", label=f"ton agent ({source_24})")
            ax_24.axhline(wbr.SOLVED, color="k", ls=":", lw=1)
            ax_24.set(xscale="log", xticks=factors_24, xticklabels=[f"×{f:g}" for f in factors_24],
                      xlabel="gravité" if name_24 == "gravity" else "longueur du mât", title=f"{name_24} × facteur")
            ax_24.legend(fontsize=8)
        axes_24[0].set_ylabel("retour moyen")
        plt.tight_layout()
        plt.show()
        for name_24, factors_24 in FACTORS_24.items():
            base_24 = mine_24[name_24][factors_24.index(1.0)]
            drops_24 = [f"×{f:g} : {fr(r, 0)} ({fr(100 * (r - base_24) / base_24, 0)} %)" for f, r in
                        zip(factors_24, mine_24[name_24])]
            print(f"{name_24:8s} " + " · ".join(drops_24))
        naive_24 = wbr.make_env()
        naive_24.unwrapped.length = 2.0                    # only the attribute: polemass_length keeps 0.05
        ev_naive_24 = wbr.evaluate(agent_24, EPISODES_24, EVAL_SEED, env=naive_24)["mean_return"]
        ev_right_24 = wbr.evaluate(agent_24, EPISODES_24, EVAL_SEED, env=set_physics_24(wbr.make_env(), length=2.0))
        print(f"mât quatre fois plus long, polemass_length oublié : {fr(ev_naive_24, 1)} ; recalculé : "
              f"{fr(ev_right_24['mean_return'], 1)} ({time.perf_counter() - start_24:.0f} s en tout)")'''

# --- 📦 B8.26 ---------------------------------------------------------------------------------------------------------
GIVEN_26 = r'''SEED_26 = 26
PAIRS_26 = wbr.preference_pairs(200, seed=SEED_26)          # broad: the rule of 208.1 with 0 to 50 % of random actions,
HELD_26 = wbr.preference_pairs(100, seed=126)               # and the random policy; HELD_26: other pairs, held out
NARROW_26 = wbr.preference_pairs(200, seed=SEED_26, behaviours=(1.0,))     # narrow: the random policy only
NARROW_HELD_26 = wbr.preference_pairs(100, seed=126, behaviours=(1.0,))
HIDDEN_26, EPOCHS_26, LR_26 = 32, 200, 3e-3
STEPS_26 = «STEPS_26»                           # the PPO of each agent (all copies)
THRESHOLD_26 = «THRESHOLD_26»                   # the held-out accuracy asked of your broad reward model (5 seeds of the reference)


def true_reward_26(obs, actions, next_obs):
    """The hidden "true" reward, as a reward_fn of train_ppo_20: exp(-(x / 0.5)^2) on the observation after each step."""
    return wbr.centered_reward(next_obs)


for name_26, pairs_26 in (("larges", PAIRS_26), ("étroites", NARROW_26)):
    trajs_26 = pairs_26["first"] + pairs_26["second"]
    print(f"préférences {name_26} : {len(pairs_26['preferred'])} paires, {np.mean([len(t) for t in trajs_26]):.0f} pas par "
          f"trajectoire en moyenne, |x| jusqu'à {max(float(np.abs(t[:, 0]).max()) for t in trajs_26):.2f} m, la "
          f"première préférée dans {100 * np.mean(pairs_26['preferred'] == 0):.0f} % des paires")'''

TODO_26 = r'''def make_reward_model_26(hidden=HIDDEN_26):
    """A new reward model: an nn.Module whose forward(obs) maps observations (N, 4) to rewards (N,): an MLP
    4 -> hidden -> hidden -> 1 (tanh) on the observations divided by wbr.OBS_SCALE."""
    raise NotImplementedError("make_reward_model_26")


def trajectory_returns_26(model, trajectories):
    """The learned return of each trajectory (a list of float32 arrays (T_i, 4)): the sum of model(states) over its
    steps, a tensor (M,) with its graph."""
    raise NotImplementedError("trajectory_returns_26")


def train_reward_model_26(pairs, seed, epochs=EPOCHS_26, lr=LR_26):
    """Learn a reward model from preferences (the dict of wbr.preference_pairs). Returns (model, losses), losses the
    list of the loss of each epoch (floats)."""
    raise NotImplementedError("train_reward_model_26")


def reward_fn_26(model):
    """A reward function for train_ppo_20: (obs, actions, next_obs) -> a float NumPy array (N,), model(next_obs)
    without gradient."""
    raise NotImplementedError("reward_fn_26")'''

SOLUTION_26 = r'''class RewardModel26(nn.Module):
    def __init__(self, hidden=HIDDEN_26):
        super().__init__()
        self.register_buffer("scale", torch.tensor(wbr.OBS_SCALE, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, obs):
        return self.net(obs / self.scale).squeeze(-1)


def make_reward_model_26(hidden=HIDDEN_26):
    """A new reward model: an nn.Module whose forward(obs) maps observations (N, 4) to rewards (N,): an MLP
    4 -> hidden -> hidden -> 1 (tanh) on the observations divided by wbr.OBS_SCALE."""
    return RewardModel26(hidden)


def trajectory_returns_26(model, trajectories):
    """The learned return of each trajectory (a list of float32 arrays (T_i, 4)): the sum of model(states) over its
    steps, a tensor (M,) with its graph."""
    rewards = model(torch.as_tensor(np.concatenate(trajectories), dtype=torch.float32))   # one pass on all the states
    ends = np.cumsum([len(t) for t in trajectories])
    return torch.stack([part.sum() for part in torch.tensor_split(rewards, ends[:-1].tolist())])


def train_reward_model_26(pairs, seed, epochs=EPOCHS_26, lr=LR_26):
    """Learn a reward model from preferences (the dict of wbr.preference_pairs). Returns (model, losses), losses the
    list of the loss of each epoch (floats)."""
    torch.manual_seed(seed)
    model = make_reward_model_26()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    first_wins = np.asarray(pairs["preferred"]) == 0
    losses = []
    for _ in range(epochs):
        r1 = trajectory_returns_26(model, pairs["first"])
        r2 = trajectory_returns_26(model, pairs["second"])
        chosen = torch.where(torch.as_tensor(first_wins), r1, r2)
        rejected = torch.where(torch.as_tensor(first_wins), r2, r1)
        loss, g_chosen, g_rejected = mylearn.deep_rl.bradley_terry_loss(chosen.detach().double().numpy(),
                                                                        rejected.detach().double().numpy())
        optimizer.zero_grad()
        torch.autograd.backward([chosen, rejected], [torch.as_tensor(g_chosen, dtype=chosen.dtype),
                                                     torch.as_tensor(g_rejected, dtype=rejected.dtype)])
        optimizer.step()
        losses.append(float(loss))
    return model.eval(), losses


def reward_fn_26(model):
    """A reward function for train_ppo_20: (obs, actions, next_obs) -> a float NumPy array (N,), model(next_obs)
    without gradient."""
    def reward(obs, actions, next_obs):
        with torch.no_grad():
            return model(torch.as_tensor(np.asarray(next_obs), dtype=torch.float32)).double().numpy()
    return reward'''

CHECK_26 = r'''ok_26 = False
with checked("B8.26"):
    torch.manual_seed(0)
    model_26 = make_reward_model_26()
    obs_26 = torch.as_tensor(np.random.default_rng(26).normal(0, 0.5, (5, 4)), dtype=torch.float32)
    out_26 = model_26(obs_26)
    n_params_26 = sum(p.numel() for p in model_26.parameters())
    verdict("B8.26", isinstance(model_26, nn.Module) and tuple(out_26.shape) == (5,) and out_26.requires_grad,
            f"make_reward_model_26 : forward rend (5,) pour 5 observations, {n_params_26} paramètres.",
            f"make_reward_model_26 : attendu un nn.Module dont forward rend un tenseur (5,) avec son graphe pour 5 "
            f"observations, reçu {tuple(getattr(out_26, 'shape', ()))}.")
    fixed_26 = wbr.reward_model(HIDDEN_26, seed=1)              # a fixed reward model, so that only your sum is tested
    some_26 = PAIRS_26["first"][:6]
    got_26 = trajectory_returns_26(fixed_26, [t.copy() for t in some_26])
    same("B8.26", "trajectory_returns_26 sur 6 trajectoires (un modèle fixé)", got_26.detach(),
         wbr.trajectory_returns(fixed_26, some_26).detach(), rtol=1e-5, atol=1e-5)
    verdict("B8.26", getattr(got_26, "requires_grad", False), "trajectory_returns_26 garde le graphe (le modèle apprendra).",
            "trajectory_returns_26 rend un tenseur sans graphe : un .detach(), .item() ou NumPy de trop ?")
    calls_26 = {"n": 0}
    real_bt_26 = mylearn.deep_rl.bradley_terry_loss

    def spy_bt_26(*args, **kwargs):
        calls_26["n"] += 1
        return real_bt_26(*args, **kwargs)

    small_26 = {k: v[:30] for k, v in PAIRS_26.items()}
    torch.manual_seed(7)
    start_model_26 = make_reward_model_26()                      # the model your function starts from (same seed)
    with torch.no_grad():
        r1_26 = wbr.trajectory_returns(start_model_26, small_26["first"])
        r2_26 = wbr.trajectory_returns(start_model_26, small_26["second"])
        wins_26 = torch.as_tensor(np.asarray(small_26["preferred"]) == 0)
        want_first_26 = float(-F.logsigmoid(torch.where(wins_26, r1_26 - r2_26, r2_26 - r1_26)).mean())
    mylearn.deep_rl.bradley_terry_loss = spy_bt_26
    try:
        model_s_26, losses_s_26 = pair("B8.26", "train_reward_model_26", train_reward_model_26(small_26, 7, epochs=20),
                                       names=("model", "losses"))
    finally:
        mylearn.deep_rl.bradley_terry_loss = real_bt_26
    good_26 = [calls_26["n"] == 20]
    verdict("B8.26", good_26[-1], "train_reward_model_26 appelle ta bradley_terry_loss à chaque epoch (20 sur 20).",
            f"attendu un appel de mylearn.deep_rl.bradley_terry_loss par epoch (20), reçu {calls_26['n']} : l'appelles-tu "
            "par ce nom (la vérification l'observe) ?")
    good_26.append(len(losses_s_26) == 20 and abs(float(losses_s_26[0]) - want_first_26) <= 1e-4 * max(1.0, want_first_26))
    verdict("B8.26", good_26[-1], f"la loss de la première epoch, {fr(want_first_26, 4)}, est celle de Bradley-Terry sur "
                                  "les paires (le préféré d'abord).",
            f"la loss de la première epoch : attendu {want_first_26:.6g} (Bradley-Terry du modèle de départ, "
            f"torch.manual_seed(seed) puis make_reward_model_26()), reçu "
            f"{(float(losses_s_26[0]) if len(losses_s_26) else None)!r} sur {len(losses_s_26)} valeurs ; le préféré est "
            "pairs['first'] quand pairs['preferred'] vaut 0.")
    good_26.append(len(losses_s_26) == 20 and float(losses_s_26[-1]) < float(losses_s_26[0]))
    verdict("B8.26", good_26[-1], "la loss baisse en 20 epochs.",
            "la loss ne baisse pas en 20 epochs : le gradient atteint-il le réseau (torch.autograd.backward avec les "
            "gradients de ta bradley_terry_loss, puis optimizer.step()) ?")
    fn_26 = reward_fn_26(fixed_26)
    batch_26 = np.random.default_rng(260).normal(0, 0.5, (4, 4)).astype(np.float32)
    r_26 = fn_26(batch_26, np.zeros(4, dtype=np.int64), batch_26.copy())
    with torch.no_grad():
        want_r_26 = fixed_26(torch.as_tensor(batch_26)).double().numpy()
    good_26.append(isinstance(r_26, np.ndarray) and r_26.shape == (4,) and np.allclose(r_26, want_r_26, atol=1e-5))
    verdict("B8.26", good_26[-1], "reward_fn_26 rend un tableau NumPy (4,) : model(next_obs).",
            f"reward_fn_26 : attendu un tableau NumPy (4,) égal à model(next_obs), reçu {type(r_26).__name__} "
            f"{np.shape(r_26)}.")
    ok_26 = all(good_26)'''

EXPERIMENT_26 = r'''# The two reward models (broad, narrow), then four PPO agents: on each learned reward, on the true reward, on the +1 of
# CartPole; each measured by the true reward and by the broad model, on 20 deterministic episodes
if not (ok_26 and ok_20):
    print("⏳ Ex B8.26 : l'expérience attend que les vérifications de B8.26 et de B8.20 (ton PPO) passent.")
else:
    with checked("B8.26"):
        start_26 = time.perf_counter()
        models_26 = {}
        for name_26, pairs_26, held_26 in (("larges", PAIRS_26, HELD_26), ("étroites", NARROW_26, NARROW_HELD_26)):
            models_26[name_26], losses_26 = train_reward_model_26(pairs_26, SEED_26)
            print(f"modèle de récompense, préférences {name_26} : loss {fr(losses_26[0], 3)} → {fr(losses_26[-1], 3)} ; "
                  f"accuracy {fr(wbr.preference_accuracy(models_26[name_26], held_26), 2)} sur ses paires tenues à "
                  f"l'écart, {fr(wbr.preference_accuracy(models_26[name_26], HELD_26), 2)} sur les paires larges")
        accuracy_26 = wbr.preference_accuracy(models_26["larges"], HELD_26)
        verdict("B8.26", accuracy_26 >= THRESHOLD_26,
                f"ton modèle de récompense (préférences larges) classe {fr(100 * accuracy_26, 0)} % des paires tenues à "
                f"l'écart dans le bon ordre (au moins {fr(100 * THRESHOLD_26, 0)} % attendus).",
                f"ton modèle de récompense (préférences larges) ne classe que {fr(100 * accuracy_26, 0)} % des paires tenues "
                f"à l'écart dans le bon ordre (au moins {fr(100 * THRESHOLD_26, 0)} % attendus).")
        rewards_26 = {"RLHF, préférences larges": reward_fn_26(models_26["larges"]),
                      "RLHF, préférences étroites": reward_fn_26(models_26["étroites"]),
                      "vraie récompense": true_reward_26, "+1 par pas (survie)": None}
        AGENTS_26, rows_26 = {}, {}
        for name_26, fn_26 in rewards_26.items():
            AGENTS_26[name_26], _ = train_ppo_20(SEED_26, STEPS_26, reward_fn=fn_26)
            rows_26[name_26] = wbr.rlhf_measure(AGENTS_26[name_26], models_26["larges"], 20, EVAL_SEED)
            rows_26[name_26]["narrow"] = wbr.rlhf_measure(AGENTS_26[name_26], models_26["étroites"], 20,
                                                          EVAL_SEED)["learned_per_step"]
        print(f"\n{'agent (PPO, ' + str(STEPS_26) + ' pas)':28s} | longueur | vraie récompense par pas | |x| moyen | "
              "récompense apprise par pas (modèle large ; étroit)")
        for name_26, m_26 in rows_26.items():
            print(f"{name_26:28s} | {m_26['length']:8.0f} | {fr(m_26['true_per_step'], 3):>24s} | "
                  f"{fr(m_26['abs_x'], 2):>9s} | {fr(m_26['learned_per_step'], 3)} ; {fr(m_26['narrow'], 3)}")
        xs_26 = np.linspace(-2.4, 2.4, 241)
        grid_26 = np.zeros((len(xs_26), 4), dtype=np.float32)
        grid_26[:, 0] = xs_26
        fig_26, (ax1_26, ax2_26) = plt.subplots(1, 2, figsize=(12, 3.4))
        ax1_26.plot(xs_26, wbr.centered_reward(grid_26), color="k", lw=2, label="vraie récompense (cachée)")
        for name_26, model_26 in models_26.items():
            with torch.no_grad():
                r_26 = model_26(torch.as_tensor(grid_26)).numpy()
            ax1_26.plot(xs_26, (r_26 - r_26.min()) / (r_26.max() - r_26.min() + 1e-12), label=f"modèle appris ({name_26})")
        ax1_26.set(xlabel="position du chariot x (les trois autres variables à 0)", ylabel="récompense (min-max)",
                   title="Ce que chaque modèle a appris, le long de x")
        ax1_26.legend(fontsize=8)
        data_x_26 = {"préférences larges": np.concatenate([t[:, 0] for t in PAIRS_26["first"] + PAIRS_26["second"]]),
                     "préférences étroites": np.concatenate([t[:, 0] for t in NARROW_26["first"] + NARROW_26["second"]])}
        for name_26, agent_26 in AGENTS_26.items():
            data_x_26[name_26] = np.concatenate([t[:, 0] for t in wbr.policy_trajectories(agent_26, 5, EVAL_SEED)])
        for name_26, x_26 in data_x_26.items():
            ax2_26.hist(x_26, bins=40, range=(-2.4, 2.4), density=True, histtype="step", lw=1.5, label=name_26)
        ax2_26.set(xlabel="position du chariot x", title="Où vont les données, et où vont les agents")
        ax2_26.legend(fontsize=7)
        plt.tight_layout()
        plt.show()
        print(f"({time.perf_counter() - start_26:.0f} s en tout)")'''
