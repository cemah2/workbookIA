#!/usr/bin/env python
"""Build the notebooks of the final project's starter kit and its bank of interview questions (used by Claude).

    python tools/chapters/build_pf.py              # writes the three notebooks and the bank of SOUTENANCE.md
    python tools/chapters/build_pf.py --check      # exit 1 if the bank of SOUTENANCE.md is out of date

Writes projets/projet_final/depart/notebooks/01_exploration.ipynb, 02_baselines.ipynb and 03_evaluation.ipynb (the
deliverables of PF.3, PF.5 and PF.8 in the contract docs/syllabus/data/pf.json). They belong to the learner's own
repository, which is not the workbook: their setup cell finds the project by its ``configs/fast.yaml`` and adds its
``src/`` to the path, and they call only the learner's package (no ``wb``, no ``mylearn``). Every step runs inside
``with step(...)``: a function not written yet, or an earlier step that did not run, prints ⏳, so that « Run all » on
the empty kit ends without an error. The sealed test opens only with ``OPEN_TEST = True`` (03_evaluation); until
then, the whole evaluation runs on the validation split (a rehearsal that writes nothing in experiments/).

The bank of questions (the 💼 of the chapters of ``chapters`` in pf.json) is read from their 02_exercices.md and
written between the markers ``<!-- pf:bank -->`` and ``<!-- pf:end bank -->`` of projets/projet_final/SOUTENANCE.md:
the ID, the title and the recruiter's question, never the model answer (it stays in the chapter's 05_solutions.md).
It also copies GRILLE.md into the starter kit (depart/docs/grille.md, the learner's self-assessment).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nbbuild import code, md, write_notebook  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "projets" / "projet_final"
NOTEBOOKS = FOLDER / "depart" / "notebooks"
CONTRACT = ROOT / "docs" / "syllabus" / "data" / "pf.json"
BANK_START, BANK_END = "<!-- pf:bank -->", "<!-- pf:end bank -->"
INTERVIEW = re.compile(r"^### (?P<id>[0-9A-Z]+\.E\d+) — (?P<title>.+?) 💼")

SETUP = r'''
# ⚙️ SETUP: run this cell first (on Colab or on your computer)
FAST_MODE = True   # True: configs/fast.yaml (CPU, < 10 min); False: configs/full.yaml (🚀 GPU Colab advised)
REPO_ON_DRIVE = "/content/drive/MyDrive/TODO-your-repository"   # on Colab only: where your repository is cloned

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

try:
    import google.colab  # noqa: F401  (this module only exists on Colab)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    os.chdir(REPO_ON_DRIVE)

ROOT = Path.cwd().resolve()
while not (ROOT / "configs" / "fast.yaml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / "configs" / "fast.yaml").exists():
    raise FileNotFoundError("Projet introuvable : ouvre ce notebook depuis ton dépôt (celui qui a configs/fast.yaml).")
os.chdir(ROOT)                                    # the paths of the configuration are relative to the project
(ROOT / "figures").mkdir(exist_ok=True)           # the figures of the report: plt.savefig("figures/....png")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))         # your package, even without pip install -e


class Pending(Exception):
    """An earlier step of the notebook did not run."""


def need(**values):
    """Raise Pending if one of the values is still None (the step that computes it is not done)."""
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise Pending("attends d'abord : " + ", ".join(missing))


@contextmanager
def step(title):
    """Run one step: a function not written yet, or a missing earlier step, prints ⏳ instead of an error."""
    try:
        yield
    except NotImplementedError as err:
        print(f"⏳ {title} : pas encore écrit ({err})")
    except Pending as err:
        print(f"⏳ {title} : {err}")


from projet.config import default_config_path, load_config  # noqa: E402

CONFIG_PATH = default_config_path(FAST_MODE)
config = load_config(CONFIG_PATH)
print(f"Projet : {ROOT.name} · mode {'rapide' if FAST_MODE else 'complet'} ({CONFIG_PATH.name}) · tâche : {config.task}")
'''

HOW = ("Chaque étape tourne dans `with step(...)` : tant qu'une fonction de ton paquet lève `NotImplementedError`, ou "
       "qu'une étape précédente n'a pas tourné, la cellule affiche ⏳ au lieu d'une erreur, et « Tout exécuter » va "
       "jusqu'au bout. Le notebook n'a pas de code d'apprentissage à lui : il appelle ton paquet (`src/projet/`). "
       "Ajoute tes propres cellules (figures, vérifications) là où le disent les `TODO`.")


def setup():
    return code(SETUP, tags=["setup"])


def exploration() -> list:
    return [
        md("""
# 01 · Exploration et audit de qualité (PF.3)

Le livrable de l'étape PF.3 : ce notebook **exécuté**, et la section « Les données » de `docs/rapport.md`. Il audite le dataset que tu as téléchargé en PF.2 (`python scripts/download_data.py`), fait le découpage **une fois** (`data/splits.json`), puis explore l'entraînement.

**Deux temps, deux règles.** L'**audit structurel** regarde tout le dataset, sans lier les entrées à la cible : tailles, identifiants, doublons, valeurs manquantes, groupes, dates, répartition de la cible. Il sert à choisir le découpage (aléatoire, par groupes, temporel). L'**exploration statistique** (les relations entre les entrées et la cible, les figures qui inspirent des features) ne regarde **que l'entraînement** : ce que tu apprends en regardant le test finit toujours dans tes choix.

""" + HOW),
        setup(),
        md("""
## 1. Charger les données

`data.load_examples(config)` rend une ligne par exemple, avec une colonne `id` unique et stable, la cible `config.target`, et les entrées brutes (le contrat est dans sa docstring).
"""),
        code("""
df = None
with step("charger les données (data.load_examples)"):
    from projet.data import load_examples

    df = load_examples(config)
    print(f"{len(df)} exemples, {df.shape[1]} colonnes")
    display(df.head())
"""),
        md("""
## 2. L'audit structurel (tout le dataset)

Les défauts du ch. 29 (🐛 29.13, 🏆 29.16) : identifiants répétés, doublons exacts et **quasi-doublons** (la même image recadrée, le même texte à la casse près, plusieurs lignes d'une même personne), valeurs manquantes codées par un nombre (`-1`, `999`, `""`), colonne constante, colonne qui trahit la cible, unités incohérentes, labels mal écrits, classe très rare. Note chaque défaut trouvé, la commande qui l'a montré et ce que tu en fais : c'est la section « Limites et biais » de ta data card.
"""),
        code("""
with step("audit structurel"):
    need(df=df)
    print("identifiants uniques :", df["id"].is_unique)
    print("doublons exacts (hors id) :", int(df.drop(columns="id").astype(str).duplicated().sum()))
    missing = df.isna().mean().sort_values(ascending=False)
    print("part de valeurs manquantes :", missing[missing > 0].round(3).to_dict() or "aucune")
    print("valeurs distinctes par colonne :", df.nunique().to_dict())
    if config.target in df:
        print("répartition de la cible :", df[config.target].value_counts(normalize=True).round(3).to_dict())
    # TODO PF.3: the quasi-duplicates of YOUR data, the missing values coded as numbers, the constant columns and the
    # columns that give the target away (29.12, 29.16); the groups (config.group_column) and the dates
"""),
        md("""
## 3. Le découpage, une fois pour toutes

`data.make_splits` applique la règle que l'audit impose (par groupes s'il y a des quasi-doublons ou plusieurs exemples d'une même entité, temporel si le modèle servira dans le futur), puis `data.save_splits` écrit `data/splits.json` et rend son SHA-256. Recopie cette somme dans `docs/protocole.md` (« SHA-256 du découpage »), puis commite et tague le protocole (PF.4). Si le fichier existe déjà, cette cellule le relit : un découpage figé ne se refait pas.
"""),
        code("""
splits = None
with step("découpage (data.make_splits, data.save_splits)"):
    need(df=df)
    from projet.data import load_splits, make_splits, save_splits, sha256_file

    split_file = Path(config.split_file)
    if split_file.exists():
        splits = load_splits(split_file)
        print(f"ℹ️ {split_file} existe déjà : il est relu, jamais refait. SHA-256 du découpage : `{sha256_file(split_file)}`")
    else:
        splits = make_splits(df, config)
        digest = save_splits(splits, split_file)
        print(f"✅ {split_file} écrit. À recopier dans docs/protocole.md : SHA-256 du découpage : `{digest}`")
    print("tailles :", {name: len(ids) for name, ids in splits.items()})
"""),
        md("""
## 4. Le décalage entre découpages

Des comptages et des résumés, sans modèle : la répartition de la cible dans chaque découpage, la moyenne des entrées numériques, la période ou les groupes couverts. Un écart ici (une classe absente du test, une période différente) se dit dans le rapport ; il peut aussi faire revoir le découpage, **avant** le tag du protocole.
"""),
        code("""
with step("décalage entre découpages"):
    need(df=df, splits=splits)
    which = pd.Series({str(i): name for name, ids in splits.items() for i in ids})
    part = df["id"].astype(str).map(which).rename("découpage")
    if config.target in df:
        display(pd.crosstab(part, df[config.target], normalize="index").round(3))
    numeric = df.select_dtypes("number").drop(columns=["id", config.target], errors="ignore")
    if not numeric.empty:
        display(numeric.groupby(part).mean().T.round(3))
    # TODO PF.3: for images or texts, a summary per split (sizes, brightness, length in tokens, sources, dates)
"""),
        md("""
## 5. L'exploration de l'entraînement

Ici seulement, les relations entre les entrées et la cible : distributions par classe, corrélations, exemples de chaque classe, séries tracées. Enregistre les figures du rapport dans `figures/` (`plt.savefig("figures/....png", dpi=150, bbox_inches="tight")`).
"""),
        code("""
train = None
with step("exploration de l'entraînement"):
    need(df=df, splits=splits)
    from projet.data import select

    train = select(df, splits, "train")
    print(f"{len(train)} exemples d'entraînement")
    # TODO PF.3: your figures and your observations (matplotlib), on train only
"""),
        md("""
## Ce que tu retiens

Recopie dans `docs/rapport.md` (« Les données ») et dans `data/card.md` (« Contenu », « Limites et biais ») :

- la taille réelle du dataset et des trois découpages, et la règle du découpage, justifiée par l'audit ;
- chaque défaut trouvé, la commande qui l'a montré, et ce que tu en as fait (corrigé, retiré, gardé et signalé) ;
- le décalage entre découpages ;
- deux ou trois observations de l'exploration qui orientent les modèles (déséquilibre, features fortes, saisonnalité).

**La suite (PF.4)** : remplis `docs/protocole.md`, puis fige-le **avant toute expérience** :

```bash
git add docs/protocole.md data/splits.json && git commit -m "Protocole d'évaluation" && git tag -a protocole -m "Protocole d'évaluation" && git push --tags
```
"""),
    ]


def baselines() -> list:
    return [
        md("""
# 02 · Les baselines (PF.5)

Le livrable de l'étape PF.5 : ce notebook **exécuté** et les premières lignes du journal `experiments/runs.jsonl`. Deux modèles, sous le protocole figé (`docs/protocole.md`, tag `protocole`) : le **prédicteur naïf** (`models.build_naive`), le plancher que tout modèle doit battre, et le **modèle classique** (`models.build_classical`), un `Pipeline` scikit-learn dont le prétraitement est réappris dans chaque fold. La validation croisée (`train.cross_validate`) ne voit que l'entraînement.

""" + HOW),
        setup(),
        md("""
## 1. La validation croisée, journalisée

La cellule lance la commande `projet baselines` (`src/projet/cli.py`) : validation croisée des deux modèles, un résumé JSON par modèle, et une ligne par run dans le journal (`train.log_run`), même si un run échoue.
"""),
        code("""
with step("baselines (projet baselines : validation croisée et journal)"):
    from projet.cli import main as projet

    code = projet(["baselines", "--config", str(CONFIG_PATH)])
    if code == 3:
        raise NotImplementedError("une fonction des baselines (le message ci-dessus la nomme)")
    print("✅ terminé" if code == 0 else f"❌ code de sortie {code} : lis le message ci-dessus")
"""),
        md("""
## 2. Le journal

Les derniers runs du journal, lus par `train.read_runs`. Le tableau de synthèse du rapport ne s'écrit jamais à la main : `python scripts/summarize_runs.py --out experiments/summary.md` (PF.7).
"""),
        code("""
with step("le journal (train.read_runs)"):
    from projet.train import read_runs

    runs = read_runs(config.runs_file)
    need(runs=runs or None)
    rows = []
    for run in runs[-10:]:
        metrics = run.get("metrics") or {}
        cv = metrics.get("cv") or {}
        rows.append({"run_id": run["run_id"], "modèle": run.get("model"), "statut": run["status"],
                     f"{config.metric} (validation croisée)": cv.get("mean"), "écart-type": cv.get("std"),
                     f"{config.metric} (validation)": metrics.get("best_val"), "secondes": run.get("seconds")})
    display(pd.DataFrame(rows))
"""),
        md("""
## Ce que tu retiens

Recopie dans `docs/rapport.md` (« Les baselines ») : les deux scores de validation croisée (moyenne ± écart-type sur les folds), leur score sur la validation (`best_val` : réentraînées sur les mêmes exemples que le modèle deep learning, notées sur la même validation, c'est le chiffre que PF.6 comparera), les `run_id`, et ce qu'ils disent du problème (une baseline classique très au-dessus du naïf : il y a du signal ; à peine au-dessus : le deep learning devra le prouver). Puis commite le journal.

**La suite (PF.6)** : le modèle deep learning, entraîné par `projet train --config configs/fast.yaml`, puis `configs/full.yaml` (🚀 GPU Colab conseillé).
"""),
    ]


def evaluation() -> list:
    return [
        md("""
# 03 · L'évaluation honnête sur le test scellé (PF.8, et l'audit de PF.9)

Le livrable de l'étape PF.8 : ce notebook **exécuté une fois**, `experiments/test_results.json` et la section « Les résultats » de `docs/rapport.md`. Le test s'ouvre **une seule fois**, pour le modèle retenu et les deux baselines, après la fin des expériences : `evaluate.open_sealed_test` vérifie le découpage par son SHA-256 (celui de `docs/protocole.md`), refuse une seconde ouverture et note celle-ci dans `experiments/test_access.jsonl`.

""" + HOW),
        setup(),
        md("""
## 0. Avant d'ouvrir le test

**La répétition générale.** Tant que `OPEN_TEST = False`, tout le notebook tourne sur la **validation** à la place du test : chaque cellule, les tiennes comprises (robustesse, explications), s'essaie sans toucher au test, et rien n'est écrit dans `experiments/`. Le test ne s'ouvre qu'une fois : une cellule qui plante après son ouverture ne se rattrape qu'en le rouvrant, et la réouverture se rapporte.

Coche tout avant de passer `OPEN_TEST = True` :

- « Tout exécuter » en répétition (`OPEN_TEST = False`) va jusqu'au bout, sans ⏳ ni erreur, sur la validation ;
- le modèle retenu est figé : son artefact est dans `config.out_dir` (`projet train --config configs/full.yaml`) et son run dans le journal ;
- les choix sont faits sur la validation (hyperparamètres, seuil, epoch d'arrêt), les ablations aussi ;
- `docs/protocole.md` est tagué `protocole`, et toute modification depuis le tag est écrite dans sa section « Modifications » ;
- tout est commité (`git status` est vide).

Après l'exécution : `open_sealed_test` a déjà commité sa ligne de `experiments/test_access.jsonl` ; commite **aussitôt** `experiments/test_results.json` et `experiments/test_predictions.csv` (une prédiction par exemple du test, pour les trois modèles : une analyse faite plus tard les relit sans rouvrir le test), puis le notebook exécuté. Le relancer ensuite (un relecteur, « Tout exécuter » dans un clone) ne rouvre pas le test : la cellule d'ouverture voit la ligne du journal, le dit, et les cellules suivantes affichent ⏳ ; les sorties qui comptent sont celles du notebook commité. Pour refaire une évaluation interrompue ou fausse, `ALLOW_REOPEN = True` : la réouverture est notée dans le journal du test, et tu la rapportes, avec les deux résultats.
"""),
        code("""
OPEN_TEST = False      # True ONCE, when the model is final (the checklist above)
ALLOW_REOPEN = False   # True only to redo an interrupted or wrong evaluation: it is logged, say it in the report
REASON = "TODO: final evaluation of the model of run <run_id>"
DELIVERED = config.out_dir   # the artifact of the retained run; say in the protocol which seed it is (chosen before
                             # the seeds are compared) and on which rows it was trained
"""),
        code("""
with step("état du protocole (evaluate.protocol_status)"):
    from projet.evaluate import protocol_status, sealed_test_accesses

    status = protocol_status()
    print("tag « protocole » :", status["tag_commit"] or "⚠️ absent")
    print("protocole modifié depuis le tag :", status["protocol_changed_since_tag"])
    print("modifications non commitées du protocole ou du découpage :", status["uncommitted"])
    print("ouvertures précédentes du test :", len(sealed_test_accesses(config)))
"""),
        md("""
## 1. Les trois modèles

Le modèle retenu, rechargé depuis son artefact (`predict.load_artifact`) comme le fera l'API, et les deux baselines, réentraînées sur **les mêmes exemples d'entraînement que lui** (le sous-échantillon du mode rapide compris) avec leurs réglages de PF.5.
"""),
        code("""
predictor = naive = classical = None
with step("le modèle retenu (predict.load_artifact)"):
    from projet.predict import load_artifact

    if not Path(DELIVERED).exists():
        raise Pending(f"l'artefact {DELIVERED} (projet train --config {CONFIG_PATH.as_posix()})")
    predictor = load_artifact(DELIVERED)
with step("les deux baselines, réentraînées"):
    from projet.data import load_examples, load_splits, select, subsample
    from projet.models import build_classical, build_naive

    examples = load_examples(config)
    # the same training rows as the deep model (cli.py): the fast mode subsamples them for both
    fit_df = subsample(select(examples, load_splits(config.split_file), "train"), config.max_train_examples, config.seed)
    # TODO PF.8: pd.concat with the validation split if your protocol refits the baselines on train + val
    naive = build_naive(config).fit(fit_df, fit_df[config.target])
    classical = build_classical(config).fit(fit_df, fit_df[config.target])
"""),
        md("""
## 2. L'ouverture du test (ou la répétition sur la validation)

Avec `OPEN_TEST = False`, `eval_df` est la validation : la répétition générale. Avec `OPEN_TEST = True`, c'est le test, ouvert une fois, et seulement si les trois modèles sont prêts à prédire.
"""),
        code("""
eval_df = None
EVAL_SPLIT = "test" if OPEN_TEST else "val"
if not OPEN_TEST:
    with step("répétition générale sur la validation (le test reste scellé)"):
        need(predictor=predictor, naive=naive, classical=classical)
        eval_df = select(examples, load_splits(config.split_file), "val")
        print(f"ℹ️ Le test reste scellé (OPEN_TEST = False) : la suite tourne sur la validation ({len(eval_df)} "
              "exemples), pour répéter l'évaluation. Rien n'est écrit dans experiments/.")
else:
    with step("ouvrir le test scellé (evaluate.open_sealed_test)"):
        from projet.evaluate import open_sealed_test, sealed_test_accesses

        need(predictor=predictor, naive=naive, classical=classical)   # never spend the one opening on a model not ready
        previous = sealed_test_accesses(config)
        if previous and not ALLOW_REOPEN:
            print(f"🔒 Le test a déjà été ouvert ({len(previous)} fois, la première le {previous[0].get('opened')}) : "
                  "cette exécution ne le rouvre pas. Les résultats de l'ouverture sont dans "
                  "experiments/test_results.json et dans les sorties du notebook commité ; pour refaire une "
                  "évaluation interrompue ou fausse : ALLOW_REOPEN = True, et dis-le dans le rapport.")
        else:
            eval_df = open_sealed_test(config, reason=REASON, allow_reopen=ALLOW_REOPEN)
            print(f"🔓 test ouvert : {len(eval_df)} exemples (l'ouverture est notée et commitée).")
"""),
        md("""
## 3. Les métriques, avec leurs intervalles

Pour chaque modèle : les métriques du protocole (`evaluate.compute_metrics`, avec les probabilités de toutes les classes pour ROC-AUC et PR-AUC ; une métrique principale de probabilités se bootstrape sur elles) et l'intervalle bootstrap à 95 % de la métrique principale (`evaluate.bootstrap_ci`, par groupes si le découpage en a) ; puis la différence entre le modèle retenu et chaque baseline, sur les **mêmes** tirages (`evaluate.paired_bootstrap`). Un chiffre sans intervalle ni baseline ne se rapporte pas.
"""),
        code("""
import json

import numpy as np

predictions = scores = results = None
with step("prédictions des trois modèles"):
    need(eval_df=eval_df, predictor=predictor, naive=naive, classical=classical)
    y_true = eval_df[config.target].to_numpy()
    out = predictor.predict(eval_df.drop(columns=[config.target]).to_dict("records"))
    predictions = {"naive": np.asarray(naive.predict(eval_df)), "classical": np.asarray(classical.predict(eval_df)),
                   "model": np.array([o.get("label", o.get("prediction")) for o in out])}
    confidence = np.array([o.get("score", np.nan) for o in out], dtype=float)   # of the PREDICTED class: sorts errors
    if config.task == "classification" and all("proba" in o for o in out):
        classes = list(classical.classes_)                  # the order of the columns of predict_proba
        scores = {"naive": naive.predict_proba(eval_df), "classical": classical.predict_proba(eval_df),
                  "model": np.array([[o["proba"][str(c)] for c in classes] for o in out])}
        if len(classes) == 2:                                # binary: the probability of the positive class
            scores = {name: s[:, 1] for name, s in scores.items()}
with step("métriques et intervalles bootstrap"):
    need(predictions=predictions)
    from projet.evaluate import bootstrap_ci, compute_metrics, paired_bootstrap

    # a main metric of probabilities is bootstrapped on the scores, any other on the labels
    USES_SCORES = config.metric.startswith(("roc_auc", "average_precision", "neg_log_loss", "neg_brier_score"))
    if USES_SCORES:
        need(scores=scores)                                  # Predictor.predict must give "proba"

    def main_metric(y, p):
        if USES_SCORES:
            return compute_metrics(y, None, config, y_score=p)[config.metric]
        return compute_metrics(y, p, config)[config.metric]

    measured = scores if USES_SCORES else predictions

    groups = eval_df[config.group_column].to_numpy() if config.group_column else None
    boot = {"n_boot": config.n_bootstrap, "seed": config.seed, "groups": groups}
    results = {"split": EVAL_SPLIT, "metric": config.metric, "n_examples": int(len(eval_df)), "models": {},
               "differences": {}}
    for name, p in predictions.items():
        low, high = bootstrap_ci(y_true, measured[name], main_metric, **boot)
        y_score = None if scores is None else scores[name]
        results["models"][name] = {**compute_metrics(y_true, p, config, y_score=y_score), "ci95_low": low,
                                   "ci95_high": high}
    for base in ("naive", "classical"):
        results["differences"][f"model - {base}"] = paired_bootstrap(y_true, measured["model"], measured[base],
                                                                      main_metric, **boot)
    display(pd.DataFrame(results["models"]).T.round(4))
    display(pd.DataFrame(results["differences"]).T.round(4))
"""),
        code("""
with step("écrire experiments/test_results.json"):
    need(results=results)
    if not OPEN_TEST:
        raise Pending("la répétition sur la validation n'écrit rien ; le fichier s'écrit à l'ouverture du test")
    path = Path("experiments/test_results.json")
    if path.exists():                    # never overwrite the first evaluation
        path = path.with_name(f"test_results_reopened_{len(list(path.parent.glob('test_results*.json')))}.json")
        print(f"⚠️ le test a déjà été évalué : ce résultat va dans {path.name}, à signaler dans le rapport")
    path.write_text(json.dumps(results, indent=1, default=float) + "\\n", encoding="utf-8")
    # one row per test example: its id, the truth, the three predictions (and scores): no input, so reading it later
    # is not a new opening (error analysis, subgroups computed afterwards from the ids)
    table = pd.DataFrame({"id": eval_df["id"].astype(str).to_numpy(), "y_true": y_true,
                          **{f"pred_{name}": p for name, p in predictions.items()}})
    if scores is not None and all(np.ndim(s) == 1 for s in scores.values()):
        table = table.assign(**{f"score_{name}": s for name, s in scores.items()})
    table_path = path.with_name(path.name.replace("test_results", "test_predictions").replace(".json", ".csv"))
    table.to_csv(table_path, index=False)
    print(f"✅ {path} et {table_path} écrits : commite-les maintenant, avec le notebook exécuté")
"""),
        md("""
## 4. L'analyse d'erreurs

Les erreurs les plus sûres d'elles (`evaluate.worst_errors`), à lire **une par une** et à ranger par cause : label douteux, cas ambigu, sous-groupe rare, défaut de prétraitement, vrai échec du modèle. Garde trois exemples parlants pour le rapport et la soutenance.
"""),
        code("""
with step("les erreurs les plus sûres d'elles (evaluate.worst_errors)"):
    need(predictions=predictions)
    from projet.evaluate import worst_errors

    errors = worst_errors(eval_df, y_true, predictions["model"], y_score=confidence, k=20)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        display(errors)                                       # every column: the ones that explain an error too
    # TODO PF.8: your reading of the errors, grouped by cause (in the report). Tabular data: read an error by its
    # profile, e.g. errors[features] - train_rows.groupby(config.target)[features].median() for its true class
    # (train_rows: select(examples, splits, "train")), and look at the features that pull it away
"""),
        md("""
## 5. Les sous-groupes et la robustesse

Les sous-groupes décidés dans le protocole (classes, sources, périodes, et une variable sensible si tu as le droit de t'en servir), chacun avec son effectif et son intervalle (`evaluate.subgroup_table`) ; puis les perturbations de robustesse du protocole (bruit, décalage, autre période). Un sous-groupe de moins de `min_size` exemples se signale, il ne se conclut pas.
"""),
        code("""
SUBGROUP_COLUMN = None   # TODO PF.8: the column of eval_df that holds the subgroups of your protocol
with step("métrique par sous-groupe (evaluate.subgroup_table)"):
    need(predictions=predictions, SUBGROUP_COLUMN=SUBGROUP_COLUMN)
    from projet.evaluate import subgroup_table

    display(subgroup_table(y_true, predictions["model"], eval_df[SUBGROUP_COLUMN].to_numpy(), main_metric,
                           n_boot=config.n_bootstrap, seed=config.seed))
with step("robustesse"):
    need(predictions=predictions)
    # TODO PF.8: the perturbations of your protocol (noise, shift, another period): predict, compute, compare
    raise NotImplementedError("TODO PF.8: robustness")
"""),
        md("""
## 6. Explicabilité et équité (PF.9)

Des attributions adaptées au modèle (B6) : SHAP ou importance par permutation pour un modèle tabulaire (B6.12, B6.15), Grad-CAM ou gradients intégrés pour des images (B6.18 à B6.20), attributions par token pour du texte ; avec leur **test de santé** (B6.22 : une explication qui ne change pas quand on brouille les poids n'explique rien). Puis l'audit par sous-groupe (B6.23, B6.24) avec ses intervalles, et ce qu'il ne prouve pas.
"""),
        code("""
with step("explications du modèle retenu"):
    need(predictions=predictions)
    # TODO PF.9: attributions on a few test examples (good predictions AND errors), their sanity check, the figures
    raise NotImplementedError("TODO PF.9: attributions")
"""),
        md("""
## Ce que tu retiens

Recopie dans `docs/rapport.md` (« Les résultats », « Explicabilité et équité ») et dans `model_card.md` : la date et le commit de l'ouverture du test (`experiments/test_access.jsonl`), le tableau des trois modèles avec leurs intervalles, les différences appariées, le critère de succès **atteint ou non** (un échec honnête vaut mieux qu'un succès douteux), l'analyse d'erreurs par cause, les sous-groupes avec leurs effectifs, la robustesse, les explications et leurs limites.
"""),
    ]


# ------------------------------------------------------------------------------------------------ the bank
def chapter_folder(cid: str) -> Path:
    prefix = f"ch{int(cid):02d}" if cid.isdigit() else cid.lower()
    matches = sorted(p for p in (ROOT / "chapitres").iterdir() if p.is_dir() and p.name.startswith(prefix + "_"))
    if not matches:
        raise SystemExit(f"no folder for chapter {cid} in chapitres/")
    return matches[0]


def interview_questions(folder: Path) -> list[dict]:
    """The 💼 of a chapter's 02_exercices.md: ID, title and the recruiter's question (the first « ... » line)."""
    lines = (folder / "02_exercices.md").read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(lines):
        match = INTERVIEW.match(line)
        if not match:
            continue
        question = next((l.strip() for l in lines[i + 1:i + 8] if l.strip().startswith("«")), "")
        out.append({"id": match["id"], "title": match["title"].strip(), "question": question})
    return out


def bank() -> str:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    steps_of = {}
    for ex in contract["exercises"]:
        for cid in ex["covers"]:
            steps_of.setdefault(cid, []).append(ex["id"])
    parts = [BANK_START, "", "*Liste générée par `python tools/chapters/build_pf.py` à partir des `02_exercices.md` "
             "des chapitres mobilisés (champ `chapters` du contrat) : ne la modifie pas à la main.*", ""]
    total = 0
    for cid in contract["chapters"]:
        folder = chapter_folder(cid)
        questions = interview_questions(folder)
        total += len(questions)
        title = re.sub(r"^#+\s*", "", (folder / "01_fiche.md").read_text(encoding="utf-8").splitlines()[0])
        title = re.sub(r"\s+—\s+fiche.*$", "", title)
        steps = ", ".join(steps_of.get(cid, [])) or "—"
        parts += [f"### {title}", "",
                  f"Étapes du projet : {steps} · réponses modèles : "
                  f"[`chapitres/{folder.name}/05_solutions.md`](../../chapitres/{folder.name}/05_solutions.md)", ""]
        for q in questions:
            parts.append(f"- **{q['id']}** — {q['title']}" + (f" · {q['question']}" if q["question"] else ""))
        parts.append("")
    parts += [f"*{total} questions, {len(contract['chapters'])} chapitres.*", "", BANK_END]
    return "\n".join(parts)


GRID_COPY = FOLDER / "depart" / "docs" / "grille.md"
GRID_HEADER = ("<!-- Ta copie de la grille du projet final (projets/projet_final/GRILLE.md du workbook), pour "
               "l'auto-évaluation : coche les critères tenus et remplis la colonne « Preuve ». Elle n'est jamais "
               "écrasée ; si la grille du workbook change, compare et recopie ce qui t'intéresse. -->\n\n")


def grid_copy() -> str:
    """The learner's copy of GRILLE.md in the starter kit (docs/grille.md): the same text, under a header."""
    return GRID_HEADER + (FOLDER / "GRILLE.md").read_text(encoding="utf-8")


def update_bank(check: bool = False) -> bool:
    """Write the bank into SOUTENANCE.md; with ``check``, only say whether it is up to date."""
    path = FOLDER / "SOUTENANCE.md"
    text = path.read_text(encoding="utf-8")
    start, end = text.index(BANK_START), text.index(BANK_END) + len(BANK_END)
    new = text[:start] + bank() + text[end:]
    if check:
        return new == text
    path.write_text(new, encoding="utf-8")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the bank of SOUTENANCE.md is out of date")
    args = parser.parse_args(argv)
    if args.check:
        ok = update_bank(check=True)
        grid_ok = GRID_COPY.exists() and GRID_COPY.read_text(encoding="utf-8") == grid_copy()
        print("✅ banque à jour" if ok else "❌ banque périmée : python tools/chapters/build_pf.py")
        print("✅ depart/docs/grille.md à jour" if grid_ok else "❌ depart/docs/grille.md périmée : build_pf.py")
        return 0 if ok and grid_ok else 1
    for name, cells in (("01_exploration.ipynb", exploration()), ("02_baselines.ipynb", baselines()),
                        ("03_evaluation.ipynb", evaluation())):
        print("✅", write_notebook(NOTEBOOKS / name, cells).relative_to(ROOT))
    update_bank()
    print("✅ projets/projet_final/SOUTENANCE.md (banque de questions)")
    GRID_COPY.write_text(grid_copy(), encoding="utf-8")
    print("✅", GRID_COPY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
