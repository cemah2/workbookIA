"""Part A (the project: B7.9 to B7.15) and the start of part B (🐛 B7.16) of the chapter B7 notebook: statements and
cells (used by build_chB7.py).

The second half of the chapter (B7.17 to B7.25) appends its exercises to PART_B.exercises and adds part C; it reuses
the tools of part A (TOOLS_A: PROJECT, DATA_FILE, SUBSET, CLASSES, DEFAULTS, scratch, run_line) and those of B7.16
(MODEL_16, INFO_16, X_16, Y_16, PATHS_16, FRENCH_16, offline_input_16), all described in EN_COURS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, Part, guarded

import chB7_code as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB7.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}

P12 = HISTORY.get("p12", {}).get("answers", {"a": True, "b": True, "c": False, "d": True})


def package_howto() -> str:
    return ("> **Mode d'emploi du paquet** : chaque fichier de ton paquet est une chaîne de la cellule à compléter "
            "(`DATA_PY = r'''...'''`) ; la cellule de vérification l'**écrit** dans ton projet "
            "(`mon_travail/b7_production/cifar-prod/`, à côté de ce notebook), puis l'importe ou le lance dans un autre "
            "processus. Pour corriger un fichier, corrige sa chaîne et réexécute les deux cellules. Si tu préfères un "
            "éditeur, modifie le fichier sur le disque : la cellule le voit et ne l'écrase plus (elle le dit).")


def prediction_record(ex_id: str, answer: bool, message: str) -> str:
    return f'wb.record("{ex_id}", {answer}, mistakes={{"{message}": {not answer}}}, prediction=True)'


# ---------------------------------------------------------------------------------------------------------------------
PART_A = Part("A", "Le projet : du notebook au paquet, ligne de commande, configuration, reproductibilité, suivi et tests",
              "Au MP5, `cifarkit` rangeait déjà ton code dans un dossier que le notebook importait. Ici, ce code devient un "
              "**projet** : un paquet installable (B7.9), piloté par une ligne de commande (B7.10) et une configuration "
              "validée (B7.11), dont tu mesures la reproductibilité (B7.12, B7.13), qui journalise ses runs (B7.14) et "
              "que des tests protègent (B7.15). Le modèle est volontairement petit (le CNN de 207.6, 2 000 images, une "
              "epoch : quelques secondes) : on juge ici la mécanique du projet, pas l'accuracy.\n\n"
              "La cellule suivante crée ton dossier de projet, `PROJECT` (`mon_travail/b7_production/cifar-prod/` ; le "
              "notebook de solutions travaille dans un dossier temporaire), y écrit `data/cifar10_2000.npz` (200 images "
              "par classe pour l'entraînement, 50 pour la validation, un sous-ensemble figé de CIFAR-10, `uint8` "
              "`(N, 32, 32, 3)`, avec les tableaux `x_train`, `y_train`, `x_val`, `y_val`) et charge `wbp` "
              "(`wb.prod`, l'infrastructure du chapitre : écrire un fichier, lancer une commande, installer dans un "
              "environnement neuf). Toutes les commandes partent du dossier du projet, avec des chemins absolus.",
              given=C.TOOLS_A, exercises=[
    Ex("B7.9", "📦", 3, 60, "Du notebook au paquet : data.py, model.py, train.py, pyproject.toml et installation éditable",
       "transformer le code d'un notebook en paquet Python installable, avec un layout `src/`.",
       "MP5 (`cifarkit`) · ch. 23 · fiche §207.1 (« Une structure de dépôt », « pyproject.toml et installation éditable »), "
       "§207.4 (`seed_everything`, `array_fingerprint`), §207.6", tracks="R, C",
       body=package_howto() + r"""

**Le contrat.** Ton projet doit avoir cette forme (le `.gitignore` et le fichier de données sont fournis) :

```text
cifar-prod/
├── pyproject.toml        .gitignore        data/cifar10_2000.npz
└── src/cifarprod/
    ├── __init__.py       (__version__ = "0.1.0", fourni)
    ├── data.py           model.py           train.py
```

- `pyproject.toml` : le backend de construction de la fiche (`setuptools`, déjà installé : un autre, comme `hatchling`, n'est pas dans l'environnement du workbook) ; le paquet `cifarprod`, version `0.1.0`, Python ≥ 3.11 ; les dépendances avec des bornes basses (`numpy`, `torch`, `pyyaml`, `safetensors`) ; deux extras, `dev` (`pytest`) et `serve` (`fastapi`, `uvicorn`, `pydantic`) ; le point d'entrée `cifarprod = "cifarprod.cli:main"` (le fichier `cli.py` viendra en B7.10) ; et, pour B7.15, une table `[tool.pytest.ini_options]` avec `testpaths = ["tests"]` et le marqueur `slow` déclaré dans `markers`.
- `data.py` : `CLASSES` (les 10 noms, dans l'ordre des sorties) ; `load_npz(path, part="train")` rend `(X, y)`, `X` `uint8` `(N, 32, 32, 3)` et `y` `int64` ; elle **vérifie le schéma** : `ValueError` pour des images qui ne sont pas `uint8` `(N, 32, 32, 3)`, des labels qui ne sont pas $N$ entiers de 0 à 9, ou une autre partie que `"train"` et `"val"` ; `FileNotFoundError` pour un fichier absent ; `fingerprint(*arrays)`, l'empreinte de 207.4 (le type, la forme, puis les octets de chaque tableau) ; `channel_stats(images)` rend `(mean, std)`, trois floats Python chacun, la moyenne et l'écart-type (divisé par $n$) de chaque canal dans $[0, 1]$ ; `preprocess(images, mean, std)` rend un tenseur `float32` `(N, 3, 32, 32)`, chaque canal valant $(\text{pixel}/255 - \text{mean}_c)/\text{std}_c$, accepte aussi **une** image `(32, 32, 3)` (rendue `(1, 3, 32, 32)`) et refuse un tableau qui n'est pas `uint8` (`ValueError`) ; `make_loader(X, y, mean, std, batch_size=128, shuffle=False, seed=0, num_workers=0)` rend un `DataLoader` de mini-batches `(preprocess(X), y)`, `float32` et `int64`, dont l'ordre mélangé vient d'un `torch.Generator().manual_seed(seed)` **à lui** (il ne dépend pas du générateur global) ; `synthetic(n, seed=0)` rend $n$ images `uint8` tirées par `np.random.default_rng(seed).integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)` et les labels `np.arange(n) % 10` (pour les tests et l'intégration continue, qui n'ont pas CIFAR-10).
- `model.py` : `build_model(arch="cnn", widths=(12, 24), n_classes=10)` rend un `nn.Sequential` des couches de la docstring, **dans cet ordre** (`"cnn"`, le CNN de la fiche et de `cifarkit.small_cnn` ; `"vgg"`, celui de `cifarkit.vgg` avec batchnorm et max pooling) : les noms de ses poids (`0.weight`, `1.weight`…) sont alors ceux de `cifarkit`, et le modèle livré du MP5 s'y recharge ; `ValueError` pour une autre architecture ou des largeurs vides ; `n_parameters(model)`.
- `train.py` : `seed_everything(seed, deterministic=True)` (207.4) ; `evaluate(model, loader)` rend `(loss moyenne, accuracy)` en mode `eval()`, sans graphe, puis rend au modèle son mode d'avant ; `train(X, y, X_val, y_val, *, arch="cnn", widths=(16, 32), epochs=1, lr=0.05, batch_size=128, seed=0, deterministic=True, num_workers=0)` appelle d'abord `seed_everything(seed, deterministic)`, apprend la normalisation sur `X` **seulement**, entraîne avec SGD (`momentum=0.9`) et la cross-entropy sur les mini-batches de `make_loader(..., shuffle=True, seed=seed)`, et rend `{"model": le modèle en mode eval(), "mean", "std", "history": {"train_loss", "val_loss", "val_acc"} (un float par epoch), "seconds"}`. **Aucun `print`** : une ligne INFO par epoch et une ligne DEBUG par mini-batch avec `log = logging.getLogger(__name__)` (207.2), que B7.10 affichera.

La cellule de vérification écrit tes fichiers, importe ton paquet **neuf** depuis `src/` et compare chaque fonction : `load_npz` aux tableaux du fichier et quatre fichiers mal formés, `fingerprint` à `hashlib`, `channel_stats` et `preprocess` à NumPy, `make_loader` avec et sans mélange (le générateur global changé entre deux appels), `build_model` à l'architecture du contrat (mêmes tenseurs et mêmes formes, quatre réglages), puis lance `train` deux fois avec la même graine (mêmes poids) et une fois avec une autre, et l'entraînement fumée des réglages par défaut sur les 2 000 images (moins d'une minute, une accuracy de validation au-dessus du hasard). Enfin, elle fait `pip install -e .` de ton projet dans un **environnement virtuel neuf** (qui voit les bibliothèques du notebook, mais pas ton dossier `src/`) et vérifie, depuis un autre dossier, que `import cifarprod` lit ton dossier `src/` (une installation éditable, pas une copie), la version et le point d'entrée. Ton environnement à toi n'est jamais modifié : pour avoir la commande `cifarprod` dans un terminal, lance toi-même `pip install -e .` dans le dossier du projet.

Dans tes notes : pourquoi le layout `src/` empêche-t-il `import cifarprod` depuis la racine du projet tant que le paquet n'est pas installé, et qu'est-ce que cela protège ? Pourquoi `train` apprend-il la normalisation sur `X` seulement, et que deviendrait l'évaluation si on la calculait sur l'entraînement et la validation ensemble (ch. 12) ? Combien de secondes prend l'entraînement fumée, et que juge-t-il (et que ne juge-t-il pas) ?""",
       todo=C.TODO_9, check=C.CHECK_9, solution=C.SOLUTION_9 + "\n\n\n" + C.CHECK_9),

    Ex("B7.10", "📦", 2, 30, "Une interface en ligne de commande : argparse, logging et codes de sortie",
       "écrire la ligne de commande d'un entraînement : options, logging et codes de sortie qu'un autre programme sait lire.",
       "Ex B7.9 · fiche §207.2 (« argparse et le point d'entrée », « Le logging plutôt que print », « Les codes de sortie »)",
       tracks="R, C",
       body=r"""**Le contrat.** Deux nouveaux fichiers ; la configuration, `config.py`, est fournie dans sa version de départ par la cellule suivante (une dataclass `TrainConfig` avec les réglages et leurs valeurs par défaut — `seed`, `epochs`, `lr`, `batch_size`, `n_train`, `arch`, `widths`, `data`, `deterministic`, `num_workers` —, et `load_config(path, overrides)`, sans aucune vérification des types ni des bornes : c'est B7.11). Elle n'est écrite que si `config.py` n'existe pas encore.

- `__main__.py` : `python -m cifarprod ...` appelle `main()` et transmet son code de sortie (207.2).
- `cli.py` : `build_parser()` rend un `ArgumentParser` de `prog="cifarprod"`, avec `--version` (« cifarprod 0.1.0 ») et la sous-commande **obligatoire** `train`, de quatre options : `--config FILE` (défaut `None` : les valeurs par défaut de `TrainConfig`), `--set KEY=VALUE` (répétable, défaut `[]`), `--out DIR` (défaut `runs/latest`) et `-v`/`--verbose`.
- `main(argv=None)` lit `argv` (`None` : `sys.argv`), configure le logging **une fois** (`logging.basicConfig`, niveau INFO, DEBUG avec `-v`, un format qui montre le niveau et le nom du logger : `%(levelname)s %(name)s: %(message)s`, sur stderr), puis : `load_config(args.config, args.set)` ; une `ValueError` ou une `OSError` → une ligne ERROR et le code **2** ; les parties `"train"` et `"val"` de `config.data` (un chemin relatif au dossier courant), dont il garde les `n_train` premières images d'entraînement ; des données illisibles ou trop peu d'images → une ligne ERROR et le code **1** ; `train(...)` avec les réglages de la configuration ; les poids dans `<out>/model.safetensors` (`safetensors.torch.save_file` du `state_dict`) ; une ligne INFO ; et sur stdout **une seule** ligne, le résumé JSON `{"status": "ok", "val_acc": <float>, "weights": <chemin absolu du fichier de poids>}` ; le code **0**. Toute autre exception pendant l'entraînement : `log.exception(...)` et le code 1. `main` **rend** son code (il n'appelle pas `sys.exit`, sauf argparse devant une mauvaise option, qui sort avec 2).

La cellule de vérification lance ta commande dans un autre processus (`python -m cifarprod`, depuis le dossier du projet, ton `src/` dans le `PYTHONPATH`) : `--version`, `--help`, sans sous-commande, avec une option inconnue, `--set lr` (sans `=`), des données absentes, un entraînement sur 200 images (son JSON, son fichier de poids, ses lignes INFO et l'absence de DEBUG), le même avec `-v` ; elle lit chaque fois le code de sortie, stdout et stderr. Puis elle appelle `main(["train", "--set", "lr"])` dans le notebook, qui doit rendre l'entier 2.

Dans tes notes : pourquoi le résumé va-t-il sur stdout et tout le reste sur stderr (pense à un script qui lit ce JSON) ? Pourquoi `main(argv)` reçoit-il ses arguments au lieu de lire `sys.argv`, et pourquoi rend-il son code au lieu d'appeler `sys.exit` ? Une intégration continue lance ta commande : que se passe-t-il si un entraînement qui échoue affiche son erreur mais rend 0 ?""",
       given=C.GIVEN_10, todo=C.TODO_10, check=C.CHECK_10, solution=C.SOLUTION_10 + "\n\n\n" + C.CHECK_10),

    Ex("B7.11", "📦", 2, 30, "Configuration : dataclass + YAML, surcharges en ligne de commande, validation",
       "remplacer les réglages en dur par une configuration typée, validée au chargement et surchargeable.",
       "Ex B7.10 · fiche §207.3 (le code de `TrainConfig` et de `load_config`, « Rappel outil — YAML avec PyYAML »)",
       tracks="R, C",
       body=r"""**Le contrat** de `config.py` (il remplace la version de départ de B7.10 ; `cli.py` ne change pas) :

- `TrainConfig` est une dataclass **figée** (`frozen=True`) aux dix champs et aux valeurs par défaut de la version de départ (`seed=0`, `epochs=1`, `lr=0.05`, `batch_size=128`, `n_train=2000`, `arch="cnn"`, `widths=(16, 32)`, `data="data/cifar10_2000.npz"`, `deterministic=True`, `num_workers=0`) ;
- à la construction (`__post_init__`), chaque réglage a **exactement** le type de sa valeur par défaut (`True` n'est pas un entier, la chaîne `"1e-3"` n'est pas un float), sauf deux conversions permises : un entier pour un float (`lr=1` devient `1.0`) et une liste pour un tuple (`widths=[8, 16]` devient `(8, 16)`) ; puis les bornes : `seed` ≥ 0, `epochs` ≥ 1, $0 <$ `lr` $< 1$, `batch_size` ≥ 1, `n_train` ≥ 1, `num_workers` ≥ 0, `arch` dans `("cnn", "vgg")`, `widths` non vide et fait d'entiers ≥ 1. Toute violation lève une `ValueError` dont le message **nomme le réglage** ;
- `to_dict()` rend un dictionnaire que `json.dumps` accepte (`widths` en liste) ;
- `load_config(path=None, overrides=())` part des valeurs par défaut, puis du fichier YAML `path` (`yaml.safe_load` ; un fichier vide donne les valeurs par défaut ; un fichier qui n'est pas un dictionnaire, `ValueError` ; un fichier absent, `FileNotFoundError`), puis des surcharges `"KEY=VALUE"` (chaque valeur lue par `yaml.safe_load`, donc `widths=[8, 16]` est une liste ; une surcharge sans `=`, `ValueError`) ; une clé inconnue, du fichier ou d'une surcharge, lève une `ValueError` qui la nomme.

Deux fichiers de configuration aussi : `configs/fast.yaml`, l'entraînement fumée du notebook, dont les réglages sont **exactement** les valeurs par défaut (écrits dans le fichier, pour qu'il se suffise) ; `configs/smoke.yaml`, l'essai de bout en bout de l'intégration continue : les mêmes, sauf `n_train: 256`, `batch_size: 64` et `widths: [8, 16]`.

La cellule de vérification essaie ta configuration dans le notebook : valeurs par défaut et `to_dict` en JSON, dataclass figée, conversions, seize surcharges invalides (chacune doit lever une `ValueError` qui nomme le réglage), l'ordre défauts → fichier → surcharges, trois fichiers pièges (`lr: 1e-3`, une faute de frappe, une liste), un fichier vide et un fichier absent, puis tes deux fichiers YAML ; enfin ta commande avec `--set epochs=0` (code 2, avant toute seconde de calcul) et `--config configs/smoke.yaml` (code 0).

Dans tes notes : que lit PyYAML dans `lr: 1e-3`, et pourquoi la validation des types est-elle la seule chose qui te sauve ici ? Pourquoi refuser une clé inconnue plutôt que l'ignorer ? Quels réglages ne mettrais-tu **jamais** dans ce fichier, et où iraient-ils (207.9) ?""",
       todo=C.TODO_11, check=C.CHECK_11, solution=C.SOLUTION_11 + "\n\n\n" + C.CHECK_11),

    Ex("B7.12", "🔮", 1, 10, "Prédire : même graine, puis num_workers = 4, puis GPU — résultats identiques ?",
       "prédire ce que les workers d'un DataLoader et un GPU changent à un entraînement à graine fixée.",
       "Ex B7.11 · ch. 24 (workers) · fiche §207.4 (« Les graines et le déterminisme »)", tracks="C", hypothesis=True,
       body=r"""L'expérience entraîne six fois un petit CNN (`wbp.workers_run`, 512 images fixes, 2 epochs, graine 0) et compare les poids **au bit près** (`torch.equal` sur chaque tenseur). Chaque run fait `torch.manual_seed(0)`, puis lit ses images par un `DataLoader(..., shuffle=True, generator=torch.Generator().manual_seed(0), num_workers=...)`, sans `worker_init_fn` ni `persistent_workers`. Le dataset applique une **augmentation** : dans son `__getitem__`, il retourne l'image (gauche-droite) quand `torch.rand(1) < 0.5`. Avec `num_workers = 0`, `__getitem__` tourne dans le processus du notebook ; avec `num_workers = 4`, dans quatre processus fils (chaque epoch en lance de nouveaux). Avec 4 workers sur une machine à 2 cœurs, PyTorch avertit qu'il en propose au plus 2 : l'avertissement est tu, il ne change rien aux nombres. **Avant d'exécuter**, écris tes quatre prédictions (`True` ou `False`) :
- a) `replay0_12` : deux runs avec `num_workers = 0` donnent-ils exactement les mêmes poids ?
- b) `replay4_12` : deux runs avec `num_workers = 4` donnent-ils exactement les mêmes poids ?
- c) `zero_four_12` : le run à `num_workers = 0` et le run à `num_workers = 4` donnent-ils exactement les mêmes poids ?
- d) `plain_12` : **sans** l'augmentation (aucun tirage dans `__getitem__`), `num_workers = 0` et `num_workers = 4` donnent-ils exactement les mêmes poids ?

Et, dans ta cellule 📝 seulement (aucune vérification) : e) sur un GPU, avec la même graine et `num_workers = 0`, les poids seraient-ils ceux du CPU ? Ceux d'un second run sur le même GPU ? La cellule d'expérience le mesure si un GPU est là (sur Colab), et le dit sinon.

La cellule d'expérience n'affiche rien tant que tes quatre prédictions ne sont pas écrites ; elle les vérifie à la fin. Rien n'y dépend de ton paquet.""",
       todo=C.TODO_12,
       solution=(f"replay0_12, replay4_12, zero_four_12, plain_12 = {P12['a']}, {P12['b']}, {P12['c']}, {P12['d']}"),
       record="\n".join([
           prediction_record("B7.12a", P12["a"], "relis le tableau : num_workers 0, essai 1 contre essai 2"),
           prediction_record("B7.12b", P12["b"], "relis le tableau : num_workers 4, essai 1 contre essai 2"),
           prediction_record("B7.12c", P12["c"], "relis le tableau : num_workers 0 contre 4, avec l'augmentation"),
           prediction_record("B7.12d", P12["d"], "relis le tableau : num_workers 0 contre 4, sans l'augmentation")]),
       after=[("code", guarded(C.EXPERIMENT_12, ["replay0_12", "replay4_12", "zero_four_12", "plain_12"],
                               "⏳ Ex B7.12 : écris d'abord tes quatre prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? D'où chaque worker tient-il "
                     "son générateur, et pourquoi la graine du `DataLoader` suffit-elle à le rejouer ? Que faudrait-il "
                     "pour qu'un run à 0 worker et un run à 4 workers tirent les mêmes retournements ? Que ferait "
                     "`worker_init_fn` (207.4) pour les tirages de NumPy et de `random` dans un worker ? Pourquoi ces "
                     "résultats ne disent-ils rien d'un GPU ?")]),

    Ex("B7.13", "🔬", 2, 30, "Mesurer le non-déterminisme : graines fixées ou non, algorithmes déterministes, cinq exécutions",
       "mesurer, sur ton propre entraînement, ce que changent la graine et les algorithmes déterministes, et rapporter la "
       "dispersion d'un score.",
       "Ex B7.12 · ✏️ B7.2 · fiche §207.4 (« Les graines et le déterminisme », « Rapporter un score »)", tracks="C",
       body=r"""Tu lances **ta** commande cinq fois et tu compares les poids qu'elle écrit. **Contrat** des trois fonctions de la cellule suivante :
- `train_run_13(seed, deterministic=True)` lance une fois, par `wbp.cli(PROJECT, ...)`, la commande `train --config configs/fast.yaml --set seed=<seed> --set deterministic=<true|false> --out <un nouveau dossier temporaire>` ; elle rend `{"weights": le dictionnaire de tenseurs du fichier de poids écrit, "val_acc": la val_acc de la ligne JSON, "seconds": la durée de la commande}`, et lève une `RuntimeError` avec la fin de stderr si le code de sortie n'est pas 0 ;
- `same_weights_13(a, b)` rend `True` quand les deux dictionnaires ont les mêmes noms et exactement les mêmes tenseurs, `False` sinon (un booléen) ;
- `max_gap_13(a, b)` rend le plus grand écart absolu entre les tenseurs de même nom, un float Python.

La cellule de vérification essaie `same_weights_13` et `max_gap_13` sur trois petits cas (des copies, un poids qui diffère de $10^{-6}$, un tenseur en moins). L'expérience lance ensuite cinq runs de 2 000 images (≈ 4 s chacun) : A et B, graine 0 ; C, graine 0 sans les algorithmes déterministes ; D et E, graines 1 et 2. Elle affiche les écarts A-B, A-C, A-D et D-E, l'accuracy des graines 0, 1 et 2 avec son intervalle à 95 % (la formule de ✏️ B7.2, $t_{0{,}975;\,2} = 4{,}303$, donné par le tableau de 207.4), et vérifie par `wb.check` si A et B ont exactement les mêmes poids (rien à taper : la valeur vient de ton code).

Dans tes notes : que donne A contre C sur ce processeur, et pourquoi ce résultat ne vaut-il que pour **ce** matériel et **ce** logiciel (207.4) ? De quel ordre est l'écart d'une graine à l'autre, et l'écart d'accuracy ? Avec trois graines, que vaut l'intervalle, et que dirais-tu d'un concurrent annoncé 1 point au-dessus sur une seule graine ? Qu'est-ce qui, dans ton paquet, rend A et B identiques : liste les générateurs que la graine fixe.""",
       todo=C.TODO_13, check=C.CHECK_13, solution=C.SOLUTION_13 + "\n\n\n" + C.CHECK_13, record=C.RECORD_13,
       after=[("code", C.EXPERIMENT_13)]),

    Ex("B7.14", "📦", 2, 30, "Suivi d'expériences maison : identifiant de run, configuration, métriques et artefacts en JSONL",
       "faire laisser à chaque run une ligne qui dit ce qui le définit, ce qu'il mesure et ce qu'il produit.",
       "Ex B7.11 · ch. 24 (journal d'expériences) · 🔁 R1 · fiche §207.4 (versions, empreinte), §207.5", tracks="R, C",
       body=package_howto() + r"""

**Le contrat.** Un nouveau fichier, `tracking.py` (la cellule suivante : exécute-la d'abord), et `cli.py` à **compléter dans la cellule de B7.10** ; réexécute ensuite la cellule de vérification de B7.10 (qui écrit aussi `tracking.py`, et vérifie que rien n'a cassé), puis celle-ci :
- `tracking.py` : `new_run_id()` (la date et l'heure UTC puis six caractères hexadécimaux au hasard : `20261104-161731-e06339`, comme 207.5) ; `file_sha256(path)` ; `artifact(path)` rend `{"path": str(path), "bytes": sa taille, "sha256": file_sha256(path)}` ; `environment()` rend `{"python", "numpy", "torch", "platform", "commit", "dirty"}` (`platform.python_version()`, `importlib.metadata.version(...)`, `platform.platform()`, le commit court `git rev-parse --short HEAD` et si `git status --porcelain` n'est pas vide, `None` hors d'un dépôt git) ; `log_run(path, record)` ajoute **une** ligne JSON (clés triées) à la fin du journal, dont il crée le dossier ; `read_runs(path)` rend la liste des runs, du plus ancien au plus récent (`[]` sans fichier) ;
- `cli.py` : une option de plus, `--runs FILE` (défaut `runs/runs.jsonl`) ; sans `--out`, les poids vont dans `runs/<run_id>/` ; **chaque** run qui a commencé (après une configuration valide) laisse une ligne, réussi ou non (un `finally`), avec les champs `run_id`, `started` (la date ISO 8601 UTC), `status` (`"ok"` ou `"failed"`), `seconds`, `config` (`to_dict()`), `environment` ; un run réussi ajoute `data` (`{"path", "n_train", "sha256"}` : l'empreinte `fingerprint(X, y)` des images **et** des labels d'entraînement utilisés), `metrics` (les trois courbes de `history` et `val_acc_final`), `normalization` (`{"mean", "std"}`) et `artifacts` (la liste des `artifact(...)` des fichiers écrits : ici, les poids) ; un run raté garde son `error` (le type et le message de l'exception) ; le résumé JSON de stdout ajoute le `run_id`.

La cellule de vérification essaie `tracking.py` dans le notebook (trois identifiants, l'empreinte d'un fichier de 300 000 octets contre `hashlib`, l'environnement, deux lignes ajoutées puis relues), puis lance trois runs dans un même journal : 200 images, puis `n_train=99999` (plus que les données : code 1), puis 200 images avec la graine 1 sans `--out`. Elle relit le journal (trois lignes, la première jamais réécrite) et compare chaque champ à ce qu'elle sait : la configuration demandée, l'empreinte des 200 premières images et de leurs labels, le fichier de poids (taille et SHA-256), la val_acc de stdout, l'erreur du run raté ; elle affiche enfin le journal par `pandas.json_normalize`.

Dans tes notes : six mois plus tard, quelle ligne du journal te dit avec quelles données, quel code et quels réglages le modèle livré a été entraîné ? Pourquoi garder les runs ratés ? Pourquoi l'empreinte porte-t-elle aussi sur les labels ? Que manque-t-il encore pour rejouer un run sur une autre machine (207.4) ?""",
       todo=C.TODO_14, check=C.CHECK_14, solution=C.SOLUTION_14 + "\n\n\n" + C.CHECK_14),

    Ex("B7.15", "📦", 3, 45, "Tests pytest pour le ML : formes, overfitting d'un mini-batch, prétraitement, schéma des données",
       "écrire les tests qui protègent un projet de ML, et vérifier qu'ils attrapent de vraies erreurs.",
       "Ex B7.11 (et B7.14) · ch. 21 (tests paramétrés, 21.16) · fiche §207.6 (« Quatre familles de tests »)", tracks="R, C",
       body=r"""La fiche donne en entier deux tests du modèle et sa fixture : ils sont dans `TEST_MODEL_PY`, avec les fixtures de `CONFTEST_PY` (`model`, le CNN de la fiche ; `data_file`, le chemin de `data/cifar10_2000.npz`, qui fait sauter le test sans données, comme dans l'intégration continue ; `synthetic_file`, un petit `.npz` au même schéma, 96 images d'entraînement et 32 de validation). **À toi** `tests/test_data.py` et `tests/test_cli.py`, avec **au moins 8 tests** (les cas d'un `parametrize` comptent chacun) dans ces deux fichiers, qui couvrent :
- le **prétraitement** d'une image constante (tout à 0, tout à 255 : la valeur attendue de chaque canal se calcule à la main) et l'ordre des canaux ;
- le **schéma des données** : types, formes, labels de 0 à 9, chaque classe présente, et le refus d'un fichier mal formé ;
- **aucun recouvrement** entre l'entraînement et la validation, par l'empreinte de chaque image ;
- une **configuration invalide refusée** (au moins `epochs=0`, un learning rate hors de $]0, 1[$, une clé inconnue) ;
- le **code de sortie** de `main(argv)`, appelé dans le test (2 pour une configuration invalide, 1 sans données) ;
- un **entraînement fumée** de bout en bout, marqué `@pytest.mark.slow`, sur `synthetic_file` : code 0, le fichier de poids écrit, et le run dans le journal si tu as fait B7.14.

Chaque `assert` a un message dont la **première ligne dit ce qui était attendu** (`f"expected ..., got ..."`), comme dans la fiche : quand l'intégration continue échoue, c'est tout ce que tu lis. Passe `--runs` et `--out` vers `tmp_path` (la fixture de pytest) pour ne rien écrire dans le projet. Si tu reviens après B7.18, ajoute un test de l'API (une réponse 422 pour une image de 28 × 28).

La cellule de vérification écrit tes tests dans `tests/`, les lance par pytest sur ton paquet, fichier par fichier (le nom de chaque test en échec, puis sa raison), et compte tes tests et ceux marqués `slow`. Puis elle les lance sur le **paquet de référence** du chapitre (`solutions/b7_cifarprod`, qui suit le contrat de B7.9 à B7.14 : un test qui y échoue exige autre chose que le contrat) et sur **six versions fautives** de ce paquet, chacune avec une erreur que tes tests doivent voir : le prétraitement qui ne divise plus par 255, `load_npz` qui rend des floats sans vérifier le schéma, une validation qui recouvre l'entraînement, une configuration qui accepte un learning rate hors de $]0, 1[$, `main` qui rend 0 sur une configuration invalide, et une commande `train` qui rend 0 sans écrire ses poids (vue seulement par un test `slow`). Pour chacune, elle dit quel test l'a attrapée, et si la première ligne de son message dit ce qui était attendu.

Dans tes notes : lequel de tes tests attrape le plus de versions fautives, et pourquoi ? Pourquoi un test qui ne fait que « ça ne plante pas » n'en attrape-t-il aucune ? Que verrait l'overfitting d'un mini-batch que les autres ne voient pas (pense à un `zero_grad` oublié) ? Pourquoi l'entraînement fumée est-il marqué `slow`, et quand le lancerais-tu quand même ?""",
       todo=C.TODO_15, check=C.CHECK_15, solution=C.SOLUTION_15 + "\n\n\n" + C.CHECK_15),
])

# ---------------------------------------------------------------------------------------------------------------------
PART_B = Part("B", "Le service : les écarts entre entraînement et production, l'artefact, l'API, la latence et l'export",
              "Le modèle sort maintenant du projet pour servir. Tu commences par réparer le service d'un collègue, dont "
              "le modèle marche dans son notebook et pas en production (B7.16) : cinq écarts entre la préparation de "
              "l'entraînement et celle du service, le training-serving skew du ch. 12. Le modèle du collègue est le VGG "
              "16-32-64 livré par la solution du MP5 (`model.pt` et `model.json`, versionnés avec le dépôt), entraîné "
              "sur 45 000 images : la partie B sert un vrai modèle, pas le petit CNN de la partie A.",
              exercises=[
    Ex("B7.16", "🐛", 2, 30, "Le modèle qui marche dans le notebook et pas en production : cinq écarts à retrouver",
       "retrouver, par des diagnostics enchaînés, les écarts entre la préparation de l'entraînement et celle d'un service.",
       "Ex B7.15 · ch. 12 (training-serving skew) · ch. 20 (`eval()`) · fiche §207.7 (« L'artefact du modèle »), §207.10",
       tracks="C",
       body=r"""Un collègue a entraîné le VGG du MP5 dans son notebook : 85 % sur ces images de test. Il l'a ensuite emballé en service : `save_artifact_16` écrit l'artefact à la fin de l'entraînement ; le service le recharge (`load_service_16`), lit chaque image envoyée par un client (`read_upload_16`, sur un fichier PNG), la prépare (`preprocess_16`) et répond le nom français de la classe (`predict_16`). **Ce que faisait son notebook**, la référence : des images `uint8` RGB `(N, 32, 32, 3)`, divisées par 255, normalisées canal par canal par la moyenne et l'écart-type de **ses images d'entraînement** (`INFO_16["mean"]`, `INFO_16["std"]`), en `float32`, le modèle en mode `eval()`, et la sortie $k$ du modèle est la classe `CLASSES[k]`. En production, le service répond n'importe quoi. **Répare son code sur place** (mêmes noms, mêmes signatures, les docstrings sont le contrat) : cinq écarts s'y cachent, et un écart peut en cacher un autre.

La cellule donnée prépare le modèle du collègue (`MODEL_16`, `INFO_16`, une copie neuve à chaque appel de `wbp.reference_model()`), 200 images de test (`X_16`, `Y_16`) écrites en PNG comme les enverrait un client (`PATHS_16`), ce que le notebook du collègue répond pour elles (`WANT_16`), et les cinq diagnostics. Chaque diagnostic écrit un **nouvel** artefact avec ton `save_artifact_16`, le recharge avec ton `load_service_16`, et ne regarde qu'une partie du service : 1. le service répond-il, et en quel type ; 2. l'image lue d'un fichier ; 3. le prétraitement, sur des images bien lues ; 4. le modèle rechargé, sur des images bien préparées ; 5. les noms rendus pour les 200 images, comparés à ceux du notebook (un **test de non-régression**, ch. 7), puis pour un modèle dont les sorties sont rangées dans un autre ordre. Ils s'enchaînent : le premier qui échoue arrête les suivants (⏳). La cellule suivante montre le service tel qu'il est sur 8 images, puis son accuracy sur les 200.

Dans tes notes : pour chaque écart, le symptôme, la cause, la ligne corrigée, et le test qui l'aurait attrapé avant la mise en production. Pourquoi certains écarts ne faisaient-ils **pas** planter le service ? Lequel en cachait un autre, et pourquoi (pense à ce que fait une normalisation par les statistiques du batch à des pixels de 0 à 255) ? Qu'aurait-il fallu mettre dans l'artefact dès le départ (207.7) ?""",
       given=C.GIVEN_16, todo=C.TODO_16, check=C.CHECK_16, solution=C.SOLUTION_16 + "\n\n\n" + C.CHECK_16,
       after=[("code", C.EXPERIMENT_16)]),
])

PARTS = [PART_A, PART_B]
