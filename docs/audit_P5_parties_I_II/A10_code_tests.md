# Audit P5, tour 2 : code, tests et garde-fous (relecteur « code », A10)

*Périmètre : `solutions/mylearn_ref/` et `templates/mylearn_stubs/` (0A à 11), `tests/`, `src/wb/`, `tools/` (constructeurs et kit), `projets/*/solution/*.py`. Lecture seule ; toutes les exécutions dans une copie `tar` (`…/audit2/code/repo`), machine partagée, `OMP/OPENBLAS/MKL_NUM_THREADS=1`. Les constats déjà triés (`docs/AUDIT_parties_I_II.md`, sections 7 et 8) et les décisions du §22 ne sont pas re-signalés.*

## Ce qui a été exécuté

| Contrôle | Résultat |
|---|---|
| `pytest -q --impl=ref --durations=25` | 1 579 passés, 3 ignorés (réseau), **31 s** ; test le plus lent 2,2 s (`test_run_notebooks_engines[nbclient]`) ; aucun test de module au-delà de 1,6 s |
| `pytest -q --impl=stubs tests/test_ch*.py` | **1 324 échecs sur 1 324** : aucun test ne passe sur les squelettes |
| `pytest tests/ -q` en mode apprenant, librairie vide | 250 passés (infra), 1 332 ignorés avec leur message, 20 s |
| `tools/syllabus.py check` · `tools/build_answers.py --check` | 0 problème · « à jour » (mais cohérent avec une réponse fausse : constat 1) |
| Signatures stubs ↔ référence (AST, 14 modules) | identiques partout ; aucune fonction `Provided:` en 0A–11 (les constructeurs des bandits sont « already written », identiques) ; seule docstring différente : `perceptron.neuron_forward` (2 exemples ajoutés = solution de 🛠️ 10.20, voulu) |
| « Run all » des 13 `03_notebook.ipynb` vides (nbclient, sans `--inplace`, puis avec `--inplace` dans la copie pour lire les sorties) | **13/13 sans erreur**, de 3,6 s (0B) à 9,6 s (ch. 1) ; aucune ❌ ; seul ✅ : 0A.19a, l'exemple fourni |
| Avertissements visibles dans les sorties des `05_solutions.ipynb` (13 chapitres, CP1, CP2, MP1, MP2) | 3, tous voulus et annoncés par l'énoncé ou le corrigé : `SettingWithCopyWarning` (0A.58), 3 `RuntimeWarning` (0B.37), `ConvergenceWarning` (1.23) |
| Contrôle par mutation de la référence (tableau 2) | 108 mutants : 97 attrapés, 2 acceptés par décision du §22, 6 équivalents, **3 vrais trous** (tous dans `stats`, ch. 2) |
| Cellules de vérification remplies avec une fonction sans `return` ou un résultat de mauvaise forme (0A, 0B, 2, 3, 5 ; notebooks construits dans la copie, exécutés avec `allow_errors`) | **10 cellules plantent**, 6 vérifications affichent un ⏳ trompeur (constats 3 et 4) |
| Piste 0B.40 e du 1ᵉʳ tour | **cause trouvée** : une mesure de temps est enregistrée comme réponse, et la réponse stockée est devenue `False` (constat 1) |

## Constats

### 1. MAJEUR — 0B.40 e : la réponse enregistrée est `False`, une bonne réponse est refusée (cause de la piste du 1ᵉʳ tour, trouvée aussi par un autre relecteur)

- **Où** : `tools/chapters/build_ch00b.py:878` (vérification) et `:901` (enregistrement) ; `chapitres/ch00b_maths/05_solutions.ipynb`, cellules 93-94 (construites par ces lignes) ; `src/wb/answers.json:4867` (`"0B.40e"`).
- **Extraits** : `    wb.check("0B.40e", bool(speedup >= 10))` · `wb.record("0B.40e", bool(speedup >= 10))` · sortie enregistrée de la cellule 93 : `-7.0 [4.61 7.5  4.  ] -0.277 0.0 NumPy is 1 times faster` · `05_solutions.md:747` : « e) **True** (NumPy est ici environ 300 fois plus rapide) ».
- **Problème** : 0B.40 e n'est pas une question à choix mais une **mesure de temps** hachée comme une réponse. Le notebook de solutions a été réexécuté au commit `e96506d` (2026-10-04, 00 h 29), pendant que les relecteurs du 1ᵉʳ tour chargeaient la machine : il a mesuré « 1 times faster » et enregistré `False`. Vérifié : le hachage de `answers.json` est celui de `"false"` (`hash_answer("0B.40e", "bool", "false")`) ; il valait `"true"` jusqu'à `384b50c`. C'est **la seule bonne réponse qui ait jamais changé** dans l'historique de `answers.json` (parcours de tous les commits). Conséquence : un apprenant dont NumPy est bien ≥ 10 fois plus rapide (mesuré ici : 36 à 257 fois, 10 mesures) reçoit `❌ Ex 0B.40e : Ce n'est pas la bonne réponse : relis l'énoncé et justifie ton choix.`, et seule une machine lente obtient ✅. Explication de la piste du 1ᵉʳ tour : avant `e96506d`, l'échec puis la réussite venaient de la mesure elle-même (machine chargée), comme 0B.54 c ; depuis, la vérification est inversée pour tout le monde. Le même notebook de solutions affiche aussi, pour 0B.54 c, `left to right: 108.0 ms, right to left: 22.23 ms, speedup: 5` (objectif ≥ 20 ; le corrigé dit ≈ 50) : une sortie de corrigé qui contredit le défi.
- **Correction** : (a) remplacer la vérification hachée par un verdict, comme 0B.54 c — ligne 878 : `    verdict("0B.40e", speedup >= 10, f"NumPy est {speedup:.0f} fois plus rapide que ta boucle : objectif atteint.", f"NumPy n'est que {speedup:.0f} fois plus rapide : relance la cellule (une mesure de temps fluctue quand la machine est occupée) ; si l'écart reste faible, vérifie que ton dot reçoit des listes et np.dot des arrays.")` ; supprimer la ligne 901 (0B.40e sort de `answers.json`, `build_answers.py` le signale « supprimée ») ; (b) reconstruire, réexécuter `05_solutions.ipynb` de 0B **sur une machine au repos** (vérifier à l'œil les vitesses affichées de 0B.40 et 0B.54) et relancer `build_answers.py` ; (c) règle à consigner au §22 : « une valeur qui dépend d'une mesure de temps ne passe jamais par `wb.record` (verdict seulement) ». Voir aussi le constat 8 (alerte de `build_answers.py`).

### 2. MAJEUR (conditionnel) — `mylearn.stats` : une modification en place des données par `zscore`, `covariance` ou `covariance_matrix` passe les 113 tests et corrompt la suite du notebook du ch. 2

- **Où** : `tests/test_ch02_stats.py:207-215` (`zscore`), `:291-301` (`covariance`), `:356-368` (`covariance_matrix`) ; seules `median` et `percentile` ont un test `..._does_not_modify_...` (`:95`, `:195`). Notebook : `tools/chapters/build_ch02.py:231` et `:475`.
- **Extraits** : `assert_close(st.zscore(x, ddof=ddof), scipy_stats.zscore(x, axis=None, ddof=ddof))` (l'oracle est calculé **après** l'appel, sur le même `x`) · `mass = measured["body_mass_g"].to_numpy()                 # in grams` · `    z_mass = st.zscore(mass)`.
- **Problème** : une erreur classique (enseignée en 0A.56, « vue modifiée ») — `arr = np.asarray(x, dtype=float)` puis `arr -= moyenne ; arr /= écart_type` — renvoie le même objet que `x` quand `x` est déjà un tableau de flottants. Les mutants « `zscore` en place », « `covariance` en place » et « `covariance_matrix` en place » passent **les 113 tests** : le z-score d'un tableau déjà standardisé est lui-même, et centrer ne change pas une covariance, donc l'oracle calculé après coup est trompé. (« `variance` en place » n'est attrapé qu'indirectement, par les tests de `zscore`, avec un message qui désigne la mauvaise fonction.) Dans le notebook, `mass` partage sa mémoire avec le DataFrame (`np.shares_memory` → `True`) : après les vérifications de 2.15 (toutes ✅), `measured["body_mass_g"]` contient des z-scores. Simulé : `❌ Ex 2.22a : Le signe de ta réponse n'est pas le bon.` pour un `mean` juste (`chinstrap_mass`, `PART_C_GIVEN`, `build_ch02.py:1004`), et `❌ Ex 2.26a : Ta valeur est trop petite d'au moins un facteur 10…` (`:1456`). L'apprenant n'a aucun moyen de remonter à la cause.
- **Correction** : (a) tests (ordre de la règle du ch. 9) : `test_zscore_does_not_modify_its_input`, `test_covariance_does_not_modify_its_inputs`, `test_covariance_matrix_does_not_modify_its_input`, `test_variance_does_not_modify_its_input`, chacun avec un tableau de flottants et le message `"zscore must not change the caller's array: compute (x - mean) / std in a NEW array (np.asarray(x, dtype=float) is x itself when x is already a float array, so -= and /= change it)"` ; dans les autres tests, appeler la fonction sur des copies (helper `call(..., copy_inputs=True)` de `tests/test_ch09_linear.py:146`). (b) Le stub étant figé, une phrase dans les énoncés de 2.15 et 2.26 (`build_ch02.py`, corps de l'exercice) : « Aucune fonction ne modifie ses arguments : `np.asarray(x, dtype=float)` renvoie **le même** tableau quand `x` en est déjà un ; calcule `x - moyenne` dans un nouveau tableau, jamais avec `-=` (0A.56). » (c) Défensif : `.to_numpy(copy=True)` aux lignes 230-232 de `build_ch02.py`.

### 3. MINEUR — `wb.check` : une fonction sans `return` reçoit « ⏳ pas encore fait (remplace `...` ou `None`…) », même avec `computed=True`

- **Où** : `src/wb/checker.py:954-955`.
- **Extrait** : `    if any(value is pending for pending in _PENDING_VALUES):` / ``        return done(False, "pending", "pas encore fait (remplace `...` ou `None` par ta réponse).", "⏳")``.
- **Problème** : le test de `None` passe avant `computed`. Vérifié dans les notebooks remplis : 0A.17 a (`min_max` sans `return`), 0A.56 a, 0B.53 a, 3.16 a (`mylearn.metrics.accuracy` sans `return`, `computed=True`), 5.11 a-b (`numerical_derivative` sans `return`, `computed=True`) affichent tous ⏳ « pas encore fait » alors que l'apprenant a écrit sa fonction. Les ch. 3 à 6 comptent 73 occurrences de `computed=True` (14 + 18 + 22 + 19 dans `build_ch03` à `build_ch06`) et aucun garde `returned` (introduit au ch. 7) ; dans les notebooks d'exercices de 0A à 6, `wb.check(id, fonction_de_l_apprenant(...))` apparaît 115 fois (0A : 31, 0B : 33, ch. 1 : 4, ch. 2 : 18, ch. 3 : 7, ch. 4 : 7, ch. 5 : 10, ch. 6 : 5).
- **Correction** (un seul endroit, aucun notebook à reconstruire) : avant la ligne 954, ``if value is None and computed: return done(False, "wrong", "ta fonction a renvoyé None : as-tu oublié le `return` ?", "❌")`` ; et message de la ligne 955 : ``"pas encore fait (remplace `...` ou `None` par ta réponse ; si cette valeur vient d'une fonction que tu as écrite, vérifie qu'elle se termine par `return`)."`` ; un test dans `tests/infra/test_wb_check.py`.

### 4. MINEUR — Chapitres 0A à 6 : les cellules de vérification plantent pour une fonction sans `return` ou un résultat de mauvaise forme (protections du ch. 7 non reportées)

- **Où et extraits** (exécutés dans la copie ; chaque trace arrête la cellule, les vérifications et les tests suivants ne s'affichent pas) :
  - 0A.15, `build_ch00a.py:351` `wb.check("0A.15a", ", ".join(short_name(r) for r in raw_names))` → `TypeError: sequence item 0: expected str instance, NoneType found` ;
  - 0A.25, `build_ch00a.py:754` `parsed = parse_all(texts)` puis `[m for m in parsed …]` → `TypeError: 'NoneType' object is not iterable` ;
  - 0B.49 a, `build_ch00b.py:1450` `list(grad_f49(3.0, 1.0))` → `TypeError: 'NoneType' object is not iterable` (0B.49 b-d non vérifiés) ;
  - 0B.54 a, `build_ch00b.py:1852` `np.allclose(left_to_right(...), right_to_left(...))` → `TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'` (0B.54 b, c, e non vérifiés) ;
  - 2.13, `build_ch02.py:349` `mass == mode_13[0]` avec un `mode` qui renvoie un nombre → `❌ … j'ai reçu un objet de type float`, puis `TypeError: 'float' object is not subscriptable` : 2.13 d-f **et les tests** ne tournent pas ;
  - 2.23, `build_ch02.py:1137` `widths_23 = widths_by_size()` puis `zip(...)` → `TypeError: 'NoneType' object is not iterable` ;
  - 2.31, `build_ch02.py:1835` `list(np.asarray(flagged_31))` → `TypeError: iteration over a 0-d array` (message incompréhensible pour un débutant) ;
  - 3.21, `build_ch03.py:1096` `sick_21, positive_21 = simulate_screening(...)` → `TypeError: cannot unpack non-iterable NoneType object` ;
  - 3.29, `build_ch03.py:1337` `column_29, threshold_29 = choose_29(val_29)` avec un dict → `KeyError: 'column'` ;
  - 5.25, `build_ch05.py:1266` `[tuple(phase) for phase in schedule_25()]` avec `return (0.0019, 8500)` (une phase hors liste) → `TypeError: 'float' object is not iterable` (le contrôle de forme écrit juste après n'est jamais atteint) ;
  - même motif, non exécuté : 1.16 c `list(train_line(x_line, y_line, 0.01, 1)[:2])` (`build_ch01.py:609`), 1.20 `model_fixed, accuracy_fixed = train_and_evaluate_20()` (`:852`).
- **Problème** : PROGRESS (« Toujours valable ») exige qu'une cellule de vérification survive à une fonction qui renvoie `None` et à un résultat de mauvaise forme ; les ch. 7 à 11 le font (`returned`, messages de forme, fonctions de notation qui renvoient `(résultat, raison)`), pas 0A à 6. Contraste vérifié : 5.19, qui passe par `np.asarray(..., dtype=float)` puis teste la forme, ne plante pas.
- **Correction** : dans `build_ch00a`, `build_ch00b`, `build_ch01` à `build_ch05`, ajouter le helper `returned(ex_id, name, value)` du ch. 7 dans la cellule d'outils et l'appeler avant toute opération sur un résultat ; contrôler la forme avant de dépaqueter (5.25 : `phases = schedule_25(); if not (isinstance(phases, (list, tuple)) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in phases)): verdict("5.25", False, "", "…une liste de 1 à 3 phases (lr, n_steps)…")` ; 3.29 : `choice = choose_29(val_29)` puis vérifier un tuple de longueur 2 ; 2.13 : `if returned("2.13", "mode", mode_13) and np.ndim(mode_13) == 1 and len(mode_13):` avant le `print`). Reconstruire, réexécuter, `build_answers.py`. Aucun énoncé ni réponse ne change.

### 5. MINEUR — Helpers pytest de 0A, 0B et 2 : raisons tronquées à 200 caractères, et sept noms différents pour le même helper

- **Où** : `build_ch00a.py:2907-2918` (`run_utils_tests`), `build_ch00b.py:740-751` (`run_linalg_tests`), `build_ch02.py:247-258` (`run_stats_tests`).
- **Extraits** : `env={**os.environ, "COLUMNS": "200"})` · `"-rf", "--tb=line"]` · `        print(line[:200])`.
- **Problème** : la règle de `CLAUDE.md` (« COLUMNS=1000, et le nom du test puis sa raison sur deux lignes ») est appliquée à partir du ch. 3 (`run_metrics_tests`), pas avant. Démonstration (librairie d'apprenant avec `bootstrap_ci` faux) : `FAILED tests/test_ch02_stats.py::test_bootstrap_ci_follows_the_documented_algorithm[0.8] - AssertionError: Not equal to tolerance rtol=1e-10, atol=1e-10 | low and high are the percentiles 50 * (1 -...` : l'explication est coupée. Pas non plus de « … and N other failed test(s) ». Noms : `run_utils_tests`, `run_linalg_tests`, `run_stats_tests`, `run_metrics_tests`, `run_bayes_tests`, `run_calculus_tests`, `run_info_tests`, puis `run_mylearn_tests` (ch. 7 : `(module, keyword)` ; ch. 8 à 11 : `(keyword)`).
- **Correction** : recopier le corps de `run_metrics_tests` (`build_ch03.py`, `--tb=no`, `COLUMNS=1000`, `❌ {name}\n   {reason[:800]}`, « … and N other ») dans les trois helpers ; à la prochaine reconstruction, un seul nom, `run_mylearn_tests(keyword)`, avec le fichier de test en constante (`TEST_FILE`, comme au ch. 9).

### 6. MINEUR — Tests de 0A, 0B et du ch. 2 : premières lignes muettes, entrées invalides sans id ni `why`, oracle calculé après un appel qui peut modifier les données

- **Où et extraits** (preuves : mutants du tableau 2) :
  - `tests/test_ch00a_utils.py:54` `assert all(isinstance(c, int) for c in result.values())` → ligne `-rf` : `assert False` (comptes `np.int64`) ; `:77` `assert isinstance(result, int)` → `assert False` (`argmax` renvoie un `np.int64`) ; `:183` (`assert all(isinstance(b, np.ndarray) and b.dtype == np.int64 …)`) ; `:168` `[batch.tolist() for batch in result]` → `AttributeError: 'list' object has no attribute 'tolist'` si l'apprenant renvoie des listes ;
  - `tests/test_ch00b_linalg_basics.py:206` `assert isinstance(result, tuple)` → `assert False` (`shape` renvoie une liste) ; `:78` `assert result is not u and result is not v` → `assert ([1.0, 2.0] is not [1.0, 2.0])` ; `:64` `assert_float_list(function(u, v), oracle(u, v))` : pour un `vector_add` qui modifie `u` en place, la première ligne dit `expected array([-9., 12., …]), got array([-3., 6., …])` alors que la somme rendue est juste (l'oracle a été calculé sur `u` déjà modifié) ;
  - entrées invalides sans id lisible ni `why` : 5 tests en 0A (`:59`, `:65`, `:108`, `:155`, `:198`), 6 en 0B, 19 au ch. 2 (ex. `test_variance_rejects_bad_inputs[bad1-1] - Failed: DID NOT RAISE <class 'ValueError'>` : l'apprenant ne sait pas quelle entrée).
- **Problème** : ces fichiers précèdent les règles de `CLAUDE.md` (première ligne qui dit l'attendu, `pytest.param(..., id=...)` et `why`, copies passées à la fonction de l'apprenant), appliquées à partir du ch. 3 (ids, messages) et du ch. 9 (copies).
- **Correction** : messages explicites (`assert isinstance(result, int), f"argmax must return a Python int (int(...)), got {type(result).__name__}"`, `f"shape must return a tuple (n_rows, n_cols), got {type(result).__name__}"`, `f"{name} must return a NEW list, not one of its arguments"`, `f"iterate_minibatches must return a list of NumPy arrays, got {type(batch).__name__}"`), `pytest.param(..., id="nan")` + `why` (modèle `assert_raises_value_error` de `test_ch03_metrics.py`), et appel sur des copies (`u_before, v_before` d'abord, oracle calculé sur les copies) ; les stubs ne changent pas.

### 7. MINEUR — Convention « float Python » : le ch. 2 à 9 accepte `np.float64`, les ch. 10 et 11 le refusent, sans ligne au §22

- **Où** : `tests/test_ch02_stats.py:44` (`isinstance(value, float)`, comme 0B, ch. 3 à 7) ; `tests/test_ch09_linear.py:118-119` (« a Python float or np.float64 ») ; `tests/test_ch10_perceptron.py:418-421` (`repr(model.intercept_) != "-4.0"`) ; `tests/test_ch11_bandit.py:113-119`.
- **Extrait** : `    if type(value) is not float:` / `raise AssertionError(f"expected {what} to be a Python float (convert with float(...)): the docstring examples print 1.0, not np.float64(1.0); …`.
- **Problème** : le §22 (session 10) dit « `np.float64` est accepté pour “un float Python” (comme en 0B et au ch. 2) » ; les ch. 10 (`intercept_`) et 11 (`pull`, `python_int` pour les indices de bras) exigent le type exact, pour que les exemples des docstrings s'affichent comme écrit. Mutants : `stats.mean` en `np.float64` et `Perceptron.score` en `np.float64` passent ; `BernoulliBandit.pull` en `np.float64` échoue. Un apprenant habitué à renvoyer `np.mean(...)` découvre la règle au ch. 11. Le message est clair ; c'est la règle qui n'est écrite nulle part.
- **Correction** (sans changer de comportement) : une ligne au §22 : « Type des scalaires : `np.float64` est accepté là où la docstring dit `float`, sauf quand un exemple de la docstring affiche la valeur (ch. 10 et suivants) : float ou int Python exigé, message « convertis avec float(...) ». » Aligner les ch. 2 à 9 serait un changement de contrat (à éviter).

### 8. MINEUR — `tools/build_answers.py` ne distingue pas une bonne réponse qui change d'une simple mise à jour des métadonnées

- **Où** : `tools/build_answers.py:117` et `:123-125`.
- **Extrait** : `    changed = sorted((k for k in set(merged) & set(old_answers) if merged[k] != old_answers[k]), key=natural_key)` puis `out(f"  {label} : {', '.join(items)}")`.
- **Problème** : au commit `e96506d`, 46 réponses « modifiées » ont été listées (elles gagnaient leurs `hash_coarse`) ; une seule avait un `hash` différent, 0B.40e (constat 1), noyée dans la liste. Une reconstruction qui ne devait changer que le texte a changé une réponse sans alerte.
- **Correction** : `answer_changed = [k for k in changed if merged[k]["hash"] != old_answers[k]["hash"]]` ; afficher d'abord `⚠️ bonne réponse changée : …` (puis « métadonnées mises à jour : … ») ; avec `--check`, échouer avec ce message ; un test dans `tests/infra/test_tools.py`.

### 9. MINEUR — `tests/conftest.py` : la docstring du module dit encore d'utiliser la fixture `mylearn`

- **Où** : `tests/conftest.py:9-10`.
- **Extrait** : « Tests of mylearn use the ``mylearn`` fixture (the whole package) or the ``mylearn_module`` fixture (one module, skipped if you have not created it yet). »
- **Problème** : contredit `CLAUDE.md` (« jamais la fixture `mylearn` ») et la docstring de la fixture elle-même (`:84-90`, « For the tests of the infrastructure only »), qui explique pourquoi (en mode apprenant, elle testerait la référence).
- **Correction** : « Tests of mylearn modules use the ``mylearn_module`` fixture (one module, skipped if you have not created it yet); the ``mylearn`` fixture (the whole package) is for the infrastructure tests only. »

### 10. SUGGESTION — Politique des NaN hétérogène d'un module à l'autre (à fixer pour les modules à venir)

- **Constat** (référence, sondes) : refus par `ValueError` dans `utils.count_values`, `stats` (sauf `mode`), `bayes`, `info`, les scores de `metrics`, `bandit.argmax_random_tie` ; propagation silencieuse ailleurs : `stats.mode([1.0, nan, nan])` → `array([nan])` alors que la docstring du module dit « Every function refuses NaN: clean the data first (e.g. with ``dropna``) » ; `cluster.NearestCentroid` → centroïde NaN ; `linear.mean_squared_error` → `nan` ; `Ridge` → `coef_ = [nan]` ; `LinearRegression` → `LinAlgError: SVD did not converge in Linear Least Squares` précédé de ` ** On entry to DLASCL parameter number  4 had an illegal value` sur stderr ; `Perceptron.fit` → poids quelconques ; `bandit.ucb_action([nan, 1.0], …)` → 0. Aucun de ces cas n'est promis par les docstrings figées : pas de défaut du contrat, mais aucune règle commune, alors que le ch. 12 (préparation des données, valeurs manquantes) arrive.
- **Proposition** : une ligne au §22 « Conventions de `mylearn` » : « Les fonctions et les estimateurs refusent NaN (`ValueError` « … contains NaN: remove the missing values first ») sauf quand leur rôle est de les traiter (imputation, ch. 12), comme scikit-learn. »

### 11. SUGGESTION — Tolérance « somme à 1 » : 1e-8 (`stats`, `bayes`) contre 1e-6 (`info`)

- **Où** : `templates/mylearn_stubs/stats.py:603` « (tolerance 1e-8) », `bayes.py:29` « summing to 1 (tolerance 1e-8) », `info.py:74` « summing to 1 (tolerance 1e-6) ».
- **Constat** : documenté dans chaque docstring (figée), mais une distribution acceptée par `info.entropy` (somme 1 + 5e-7, par exemple des probabilités arrondies à 6 décimales) est refusée par `bayes.bayes_posterior` et `stats.sample_categorical`.
- **Proposition** : retenir une seule tolérance pour les modules à venir (1e-6, la plus tolérante) et l'écrire au §22.

### 12. SUGGESTION — `rng=42` (l'habitude de scikit-learn) donne une erreur obscure dans toutes les fonctions de `mylearn`

- **Constat** : `model_selection.train_test_split(np.arange(10), rng=42)` → `AttributeError: 'int' object has no attribute 'permutation'` ; même chose pour `stats.sample`, `cluster.kmeans_plusplus`, `bandit.argmax_random_tie`, `utils.iterate_minibatches`. Cohérent avec les docstrings (`rng : np.random.Generator or None`), mais déroutant juste après `random_state=42`.
- **Proposition** (modules à venir) : `rng = np.random.default_rng(rng)` accepte `None`, un entier ou un `Generator` (un `Generator` est renvoyé tel quel), et l'écrire dans les conventions du §22 ; pour 0A à 11 (stubs figés), une ligne ⚠️ dans la fiche du ch. 8 ou un message ciblé dans les cellules de vérification qui appellent ces fonctions.

### 13. SUGGESTION — `wb.run_pytest` : un test de l'apprenant qui boucle fait planter la cellule après 120 s

- **Où** : `src/wb/testing.py`, `completed = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=timeout)`.
- **Constat** : `subprocess.TimeoutExpired` n'est pas rattrapé (trace Python au lieu d'un message) ; aucun test dans `tests/infra/test_wb_testing.py`.
- **Proposition** : `except subprocess.TimeoutExpired: print(f"❌ tes tests ont dépassé {timeout:.0f} s : une boucle infinie ?"); return PytestResult(0, 0, 1, "timeout", "")`, et un test avec une fonction qui boucle (`timeout=2`).

## Tableau 1 — Cohérence de l'API `mylearn` d'un module à l'autre (référence sondée, tests lus)

✅ conforme · ✗ écart (constat indiqué) · — sans objet. Classes sondées : `NearestCentroid`, `KMeans`, `OneVsRestClassifier`, `OneVsOneClassifier`, `LinearRegression`, `Ridge`, `Lasso`, `Perceptron` : `__init__` ne fait que stocker ses arguments sous leur nom, `fit` renvoie `self`, attributs appris suffixés `_` (aucun autre attribut public créé par `fit`), `score` présent, `model_selection.clone` fonctionne sur chacune ; `predict` avant `fit` → `AttributeError` partout ; `X` à une dimension → `ValueError` partout.

| Règle | 0A `utils` | 0B `linalg_basics` | 2 `stats` | 3 `metrics` | 4 `bayes` | 5 `calculus` | 6 `info` | 7 `cluster`, `multiclass` | 8 `model_selection` | 9 `linear` | 10 `perceptron` | 11 `bandit` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `rng` pour les fonctions, `random_state` pour les classes | ✅ | — | ✅ | — | — | — | — | ✅ | ✅ | — | ✅ | ✅ |
| Scalaires renvoyés en `float`/`int` Python (référence) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Les tests acceptent `np.float64` pour « float » | — (`int` exigé) | oui | oui | oui | oui | oui | oui | oui | — | oui (écrit) | `score` oui, `intercept_` non | non (✗ 7) |
| NaN refusé par `ValueError` | ✅ | — | ✅ sauf `mode` (✗ 10) | scores ✅, labels non | ✅ | non (`find_local_extrema`) | ✅ | non (✗ 10) | — | non : `nan`, `LinAlgError` (✗ 10) | non (✗ 10) | `argmax_random_tie` ✅, `ucb_action` non |
| Tolérance « somme à 1 » | — | — | 1e-8 | — | 1e-8 | — | 1e-6 (✗ 11) | — | — | — | — | — |
| Entrées laissées intactes, testé | — | ✅ (oracle calculé après l'appel, ✗ 6) | partiel : `median`, `percentile` ; trous `zscore`, `covariance`, `covariance_matrix` (✗ 2) | indirect (valeurs) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ + copies | ✅ + copies | ✅ |
| Entrées invalides : id lisible et `why` | ✗ 6 | ✗ 6 | ✗ 6 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 1ʳᵉ ligne d'un échec qui dit l'attendu | partiel (`assert False`, ✗ 6) | partiel (✗ 6) | ✅ (résumé NumPy du `conftest`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Helper pytest du notebook (COLUMNS=1000, nom puis raison) | ✗ 5 | ✗ 5 | ✗ 5 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cellules protégées (`returned`, forme, `(résultat, raison)`) ; ch. 1 sans module : ✗ 4 | ✗ 4 | ✗ 4 | ✗ 4 | ✗ 4 | ✗ 4 | ✗ 4 | ✗ 3 (`computed=True`) | ✅ | ✅ | ✅ | ✅ | ✅ |

Exceptions : `ValueError` partout pour une entrée invalide, sauf `BernoulliBandit.pull` et `GaussianBandit.pull` (`IndexError`, décision du §22) et `clone` (`TypeError` propagée, documentée) ; noms d'hyperparamètres alignés sur les bibliothèques oracles (`eta0`, `alpha`, `n_clusters`, `n_init`, `max_iter`, `tol`, `lr`, `maximize`, `fit_intercept`, `average`, `pos_label`, `zero_division`). Mini-projets : `HousingModel(random_state=0)` suit la convention ; `make_test_indices(..., seed=2026)` et `learning_curve(..., seed=0)` prennent un entier `seed` (module de projet, pas de `mylearn` : sans conséquence).

## Tableau 2 — Contrôle par mutation de la référence (dans la copie, `--impl=ref`, tests du module seulement)

108 mutants (erreurs plausibles d'apprenant : comparateur, axe, ddof, signe, copie oubliée ou modification en place, borne, type de retour, ordre, oubli de `return self`). Résultat : **97 attrapés** ; 2 survivants acceptés par décision (§22) ; 6 équivalents ; **3 vrais trous** (constat 2). Deux mutants mal formés ont été écartés et refaits (`median` en place, `Perceptron.fit` « en place » sans effet) ; quatre mutants « en place » de `gradient_descent` et de `KMeans` étaient équivalents pris isolément (copie faite ailleurs) : leur version combinée est attrapée (lignes `gradient_descent déplace x0` et `KMeans écrit les centres finaux dans init`). La dernière colonne est la première ligne `-rf` que voit l'apprenant (les lignes `assert False` du 0A et du 0B sont le constat 6).

| Module | Erreur simulée | Résultat | Tests en échec | Première ligne `-rf` (abrégée) |
|---|---|---|---|---|
| `_example` | pas de contrôle de la liste vide (ZeroDivisionError) | attrapé | 1 | `ZeroDivisionError: float division by zero` |
| `_example` | divise par n − 1 | attrapé | 2 | `assert -0.02299435432578868 == -0.02244686969898421 ± 2.2e-08` |
| `utils` | `count_values` accepte NaN | attrapé | 2 | `Failed: DID NOT RAISE <class 'ValueError'>` |
| `utils` | `count_values` via `np.unique` (comptes `np.int64`) | attrapé | 3 | `assert False` |
| `utils` | `argmax` garde le dernier maximum (`>=`) | attrapé | 6 | `assert 5 == np.int64(1)` |
| `utils` | `argmax` : axes 0 et 1 inversés | attrapé | 3 | `AssertionError: assert (True and (5,) == (2,)` |
| `utils` | `one_hot` sans contrôle de plage (IndexError) | attrapé | 1 | `IndexError: index 3 is out of bounds for axis 1 with size 3` |
| `utils` | `iterate_minibatches` ignore `drop_last` | attrapé | 3 | `assert [[8, 0, 7], [...2, 4, 5], [9]] == [[8, 0, 7], [...6], [2, 4, 5]]` |
| `utils` | `one_hot` ignore `dtype` | attrapé | 1 | `AssertionError: assert dtype('float64') == dtype('int64')` |
| `utils` | `argmax` renvoie un `np.int64` | attrapé | 2 | `assert False` |
| `utils` | `iterate_minibatches` renvoie des listes | attrapé | 9 | `AttributeError: 'list' object has no attribute 'tolist'` |
| `utils` | `one_hot` en entiers par défaut | attrapé | 1 | `AssertionError: assert dtype('int64') == <class 'numpy.float64'>` |
| `linalg_basics` | `vector_add` modifie `u` en place et le renvoie | attrapé | 2 | `AssertionError: Not equal to tolerance rtol=1e-12, atol=1e-12 \| expected array([-9., 12., 2., 12., -9., -4., 9.]), got array([-3., 6., -…` |
| `linalg_basics` | norme L∞ sans valeur absolue | attrapé | 2 | `assert 3.0 == 9.0 ± 9.0e-12` |
| `linalg_basics` | cosinus non borné à [−1, 1] | attrapé | 1 | `AssertionError: rounding errors must not push the value outside [-1, 1]` |
| `linalg_basics` | `identity` avec lignes partagées (`[ligne] * n`) | attrapé | 4 | `AssertionError: Not equal to tolerance rtol=1e-12, atol=1e-12 \| expected array([[1., 0.], [0., 1.]]), got array([[1., 1.], [1., 1.]])` |
| `linalg_basics` | `matmul` garde des entiers | attrapé | 1 | `AssertionError: the entries must be Python floats` |
| `linalg_basics` | `transpose` sans conversion en float | attrapé | 1 | `AssertionError: the entries must be Python floats` |
| `linalg_basics` | `shape` renvoie une liste | attrapé | 2 | `assert False` |
| `linalg_basics` | `dot` : total initialisé à 0 (int) | survit — équivalent (`float(total)` à la fin) | 0 | — |
| `stats` | `variance` ignore `ddof` | attrapé | 4 | `AssertionError: Not equal to tolerance rtol=1e-10, atol=1e-10 \| expected array(148.347619), got array(141.283447)` |
| `stats` | `percentile` sans interpolation | attrapé | 4 | `AssertionError: Not equal to tolerance rtol=1e-10, atol=1e-10 \| expected array(13.884431), got array(14.)` |
| `stats` | `histogram` exclut le bord droit du dernier intervalle | attrapé | 6 | `AssertionError: Not equal to tolerance rtol=1e-10, atol=1e-10 \| expected array([ 5., 5., 18., 27., 27., 35., 23., 19., 11.]), got array(…` |
| `stats` | `bootstrap_ci` : percentiles 2,5/97,5 en dur | attrapé | 1 | `AssertionError: Not equal to tolerance rtol=1e-10, atol=1e-10 \| low and high are the percentiles 50 * (1 - confidence) and 50 * (1 + con…` |
| `stats` | `mean` renvoie un `np.float64` | survit — accepté (§22, session 10) | 0 | — |
| `stats` | `sample` sans remise via `rng.choice` | attrapé | 1 | `AssertionError: with the same seed, your draws must be exactly those of the algorithm of the docstring` |
| `stats` | `median` trie le tableau de l'appelant en place | attrapé | 1 | `AssertionError: median must not sort its input in place` |
| `stats` | `zscore` standardise le tableau de l'appelant en place | **survit — vrai trou** | 0 | — |
| `stats` | `covariance_matrix` centre `X` en place | **survit — vrai trou** | 0 | — |
| `stats` | `bootstrap_distribution` mélange les données en place | attrapé | 6 | `AssertionError: Not equal to tolerance rtol=1e-10, atol=1e-10 \| expected array([19.60137 , 18.975676, 19.972486, 20.204704, 20.669826, 1…` |
| `stats` | `variance` centre `x` en place | attrapé indirectement (par les tests de `zscore`) | 3 | `AssertionError: Not equal to tolerance rtol=1e-10, atol=1e-10 \| expected array([[ 0.493251, 0.183098, 1.112381], [ 0.468194, -0.302355, …` |
| `stats` | `covariance` centre `x` et `y` en place | **survit — vrai trou** | 0 | — |
| `stats` | `histogram` borne les données en place (hors de l'intervalle) | survit — équivalent (valeurs hors intervalle : comptes inchangés) | 0 | — |
| `metrics` | matrice de confusion transposée | attrapé | 28 | `AssertionError: C[i, j] counts the samples of true label i predicted as label j: expected [[14, 2], [ 4, 11]], got [[14, 4], [ 2, 11]]` |
| `metrics` | `zero_division` ignoré (NaN) | attrapé | 21 | `AssertionError: precision = zero_division (0.0) only when TP + FP = 0: expected 0, got nan` |
| `metrics` | ROC sans fusion des ex æquo | attrapé | 17 | `AssertionError: one point per distinct score, plus the starting point at +inf` |
| `metrics` | AP par trapèzes | attrapé | 9 | `AssertionError: AP = sum of (R_n - R_(n-1)) P_n over the thresholds, from the highest {}: expected 0.865636, got 0.86375` |
| `metrics` | calibration : bord intérieur vers l'intervalle du haut | attrapé | 5 | `AssertionError: share of positives per bin (n_bins=2; a value on an inner edge goes to the lower bin): expected [0.142857, 0.757576], got…` |
| `metrics` | `average='weighted'` sans poids | attrapé | 7 | `AssertionError: precision(average='weighted'): expected 0.72549, got 0.724537` |
| `metrics` | ordre de `labels` ignoré | attrapé | 1 | `AssertionError: labels=['spam', 'ham', 'eggs']: rows and columns follow this order: expected [[1, 1, 0], [1, 1, 1], [0, 0, 1]], got [[1, …` |
| `metrics` | `roc_curve` trie les scores en place | attrapé | 16 | `AssertionError: false positive rates FP / (FP + TN), predicting positive when score >= t: expected [0. , 0. , 0. , ..., 0.965517, 0.96551…` |
| `metrics` | `brier_score` modifie les probabilités en place | attrapé | 1 | `ValueError: y_proba contains values less than 0.` |
| `bayes` | `update_discrete` normalise à la fin (underflow) | attrapé | 2 | `AssertionError: got NaN after 5 000 observations: normalize the posterior at EVERY step (it becomes the next prior), do not multiply all …` |
| `bayes` | `coin_bias_posterior` sans logarithmes | attrapé | 2 | `AssertionError: coin_bias_posterior returned NaN values: [nan, nan, nan, ..., nan, nan, nan]` |
| `bayes` | `credible_interval` avec `side='right'` | attrapé | 1 | `AssertionError: mass 0.5: the first grid value whose cumulative probability is >= each level (reaching a level exactly counts): expected …` |
| `bayes` | `evidence` sans contrôle de la somme du prior | attrapé | 6 | `AssertionError: evidence must raise ValueError here: the prior sums to 1.1, it is not a distribution` |
| `bayes` | `bayes_posterior` modifie le prior en place | attrapé | 7 | `AssertionError: prior was modified in place: [0.125, 0.37499999999999994, 0.5] (work on a copy)` |
| `bayes` | `update_discrete` divise le prior par sa somme, en place | survit — quasi équivalent (division par 1 ± 1e-16) | 0 | — |
| `calculus` | différence centrée divisée par h | attrapé | 14 | `AssertionError: the central difference of sin must approach its derivative (h = 1e-5): expected -0.128844, got -0.257689` |
| `calculus` | `numerical_gradient` modifie `x` de l'appelant | attrapé | 1 | `AssertionError: the caller's x was modified while f was called (seen: [0.50001, -1.25, 2.0]): nudge the coordinates of a float COPY (np.a…` |
| `calculus` | `gradient_descent` : le chemin stocke le même tableau | attrapé | 8 | `AssertionError: the path must match torch.optim.SGD(lr=0.1, maximize=False) step by step (x - lr * grad(x) to descend, x + lr * grad(x) t…` |
| `calculus` | extrema non stricts (`<=`) | attrapé | 6 | `AssertionError: local minima (order=1), as scipy.signal.argrelextrema: expected shape (2,), got shape (6,): expected [11, 13], got [ 3, 4…` |
| `calculus` | point critique : axes seulement | attrapé | 2 | `AssertionError: expected 'saddle', got 'flat' (at x=[0.0, 0.0])` |
| `calculus` | `gradient_descent` ignore `tol` | attrapé | 1 | `AssertionError: with tol=0.3 the loop stops when \|grad\| < 0.3 (at x = 0.125, after 3 steps): expected a path of shape (4, 1), got (101, 1)` |
| `calculus` | dérivée seconde divisée par (2h)² | attrapé | 7 | `AssertionError: the second difference of sin must approach its second derivative (h = 1e-4): expected 0.991665, got 0.247916` |
| `calculus` | `gradient_descent` déplace `x0` de l'appelant en place | attrapé | 1 | `AssertionError: the caller's x0 must not change (work on a float copy), got [1.50848, -0.99232]` |
| `info` | entropie en nats par défaut | attrapé | 34 | `AssertionError: the surprise of an event of probability 0.5, in bits, is -log2(p): expected 1, got 0.693147` |
| `info` | KL sans test de q = 0 (avertissement NumPy) | attrapé | 1 | `AssertionError: expected kl_divergence to return inf without any NumPy warning (test q == 0 before taking the log), got the warning: divi…` |
| `info` | `log_loss` sans écrêtage | attrapé | 1 | `AssertionError: a confident mistake must give a large but finite loss, got inf` |
| `info` | vocabulaire trié (ordre perdu) | attrapé | 3 | `AssertionError: a string alphabet must come back as a list of characters: expected ['h', 'e', 'l', 'o'], got ['e', 'h', 'l', 'o']` |
| `info` | Huffman : égalités départagées par le symbole | attrapé | 4 | `TypeError: '<' not supported between instances of 'int' and 'str'` |
| `info` | perplexité calculée en bits (2 ** H) | survit — équivalent (2^H en bits = e^H en nats) | 0 | — |
| `info` | `char_distribution` ignore `lowercase=False` | attrapé | 1 | `AssertionError: with lowercase=False, 'A' and 'a' are different characters: expected ['A', 'a', 'b'], got ['a', 'b']` |
| `info` | `entropy` modifie `p` en place | attrapé | 2 | `AssertionError: outcomes with p_i = 0 contribute 0 (0 log 0 = 0), not NaN: expected 2.52508, got 2.10032` |
| `info` | `token_distribution` garde un alias du vocabulaire | survit — équivalent (alias jamais modifié) | 0 | — |
| `cluster` | distances non élevées au carré | attrapé | 26 | `AssertionError: squared distances between 5 and 3 points in 2-D (scipy cdist 'sqeuclidean'): first difference at index (0, 0): expected 3…` |
| `cluster` | cluster vide → centre NaN | attrapé | 1 | `AssertionError: an empty cluster keeps its previous centre (the mean of no point is undefined): expected [[ 0.5], [ 10.5], [100. ]], got …` |
| `cluster` | `KMeans.fit` sans `return self` | attrapé | 23 | `AttributeError: 'NoneType' object has no attribute 'n_iter_'` |
| `cluster` | k-means++ proportionnel à D (pas D²) | attrapé | 1 | `AssertionError: after the first centre 0, expected the next one to be drawn with probability D(x)²/ΣD²: about [4, 38, 205, 940] times for…` |
| `cluster` | silhouette : a divisé par la taille du cluster | attrapé | 11 | `AssertionError: silhouette of every sample (scikit-learn silhouette_samples) [ints labels]: first difference at index 0: expected 0.30280…` |
| `cluster` | `labels_` des centres d'avant la dernière mise à jour | attrapé | 4 | `AssertionError: labels_ (scikit-learn KMeans, same init; the nearest FINAL centre of every sample, recomputed when tol or max_iter stops …` |
| `cluster` | `decision_function` binaire de signe inversé | attrapé | 2 | `AssertionError: with 2 classes, decision_function is d²(x, c0) - d²(x, c1), shape (n_samples,): first difference at index 0: expected 35.…` |
| `cluster` | `NearestCentroid.fit` modifie `X` en place | attrapé | 4 | `AssertionError: fit must not modify X` |
| `cluster` | `KMeans` écrit les centres finaux dans `init` | attrapé | 1 | `AssertionError: fit must not modify the init array (work on a copy)` |
| `multiclass` | OvR : un seul modèle partagé (pas de copie) | attrapé | 9 | `AssertionError: decision_function: column k is the score of the model 'class k against the rest' (scikit-learn OneVsRestClassifier, linea…` |
| `multiclass` | OvO : égalités vers la dernière classe | attrapé | 1 | `AssertionError: a three-way tie must go to the smallest class index (np.argmax keeps the first): expected [0, 0, 0, 0, 0, 0], got [2, 2, …` |
| `multiclass` | OvR : scores = `predict` (0/1) | attrapé | 6 | `AssertionError: decision_function: column k is the score of the model 'class k against the rest' (scikit-learn OneVsRestClassifier, linea…` |
| `multiclass` | OvO : paire inversée | attrapé | 8 | `AssertionError: votes (scikit-learn OneVsOneClassifier with linear_svc: decision_function rounded): expected [[2, 1, 0], [1, 0, 2], ..., …` |
| `model_selection` | taille du test arrondie vers le bas | attrapé | 10 | `AssertionError: expected 3 test and 7 train samples for n = 10 and test_size = 0.25 (n_test = ceil(test_size * n), exactly (an integer pr…` |
| `model_selection` | k-fold : le reste dans le dernier fold | attrapé | 14 | `AssertionError: docstring example, split 0: train_idx: expected shape (2,), got shape (3,): expected [3, 4], got [2, 3, 4]` |
| `model_selection` | `clone` sans copie profonde | attrapé | 1 | `AssertionError: expected the original estimator unchanged when the clone's lists are modified, got degrees=[1, 2, 3], options={'weights':…` |
| `model_selection` | `cross_val_score` sans `clone` | attrapé | 1 | `AssertionError: expected a new estimator for every split (clone(estimator) inside the loop), got the same object twice` |
| `model_selection` | une permutation par tableau (lignes désalignées) | attrapé | 1 | `AssertionError: y_train[i] must be the label of row X_train[i] (rows of X and y must be taken with the same indices): expected [ 40, 110,…` |
| `model_selection` | tri non stable des classes | attrapé | 4 | `AssertionError: split 0: val_idx (indices sorted by class with a stable sort, then fold 0 takes the positions 0, 0 + 5, 0 + 2*5... of tha…` |
| `model_selection` | `train_test_split` mélange `X` en place | attrapé | 2 | `AssertionError: two calls with np.random.default_rng(7) must give the same test part: expected [88, 42, 26, ..., 48, 73, 80], got [38, 0,…` |
| `linear` | Ridge pénalise l'ordonnée à l'origine | attrapé | 10 | `AssertionError: coef_ with alpha = 0.01 (objective \|\|y - Xw - b\|\|² + alpha \|\|w\|\|²: a SUM of squares, no 1/n, intercept not penali…` |
| `linear` | R² : SS_tot autour de la moyenne des prédictions | attrapé | 9 | `AssertionError: R² = 1 - SS_res / SS_tot, SS_tot computed around the mean of y_true: expected 0.539717, got 0.595677` |
| `linear` | Lasso : seuil α·n | attrapé | 14 | `AssertionError: coef_ with alpha = 0.05, tol = 1e-10 (objective (1/2n) \|\|y - Xw - b\|\|² + alpha \|\|w\|\|₁: w_j = soft_threshold(rho_j…` |
| `linear` | colonnes polynomiales dans un autre ordre | attrapé | 8 | `AssertionError: 2 feature(s), degree 2, include_bias=False (columns in scikit-learn's order: x0, x1, x0^2, x0 x1, x1^2): largest differen…` |
| `linear` | variance des modèles avec ddof = 1 | attrapé | 8 | `AssertionError: docstring example: (bias2, variance): expected [1., 1.], got [1., 2.] (first difference at index 1: expected 1, got 2)` |
| `linear` | `fit_intercept=False` ignoré | attrapé | 3 | `AssertionError: fit_intercept=False: no centring, the line goes through the origin: expected [0.662915], got [2.288825] (first difference…` |
| `linear` | `LinearRegression` centre `X` en place | attrapé | 1 | `AssertionError: expected fit to leave X and y unchanged (centre copies: X - X.mean(axis=0) creates a new array)` |
| `linear` | `Ridge` centre `X` en place | attrapé | 1 | `AssertionError: expected fit to leave X and y unchanged` |
| `perceptron` | erreur si `< 0` (z = 0 n'est pas une erreur) | attrapé | 23 | `AssertionError: coef_ = scikit-learn's coef_[0]: expected [0.514692, 0.237005], got [0., 0.] (first difference at index 0: expected 0.514…` |
| `perceptron` | labels 0/1 au lieu de −1/+1 | attrapé | 24 | `AssertionError: coef_ = scikit-learn's coef_[0]: expected [0.514692, 0.237005], got [-0.618643, 1.618824] (first difference at index 0: e…` |
| `perceptron` | une seule permutation pour toutes les époques | attrapé | 1 | `AssertionError: expected shuffle=True to draw a NEW order of visit at each epoch: on XOR, any fixed order (even a random one drawn once, …` |
| `perceptron` | `intercept_` en `np.float64` | attrapé | 1 | `AssertionError: expected clf.intercept_ to display -4.0, as in the docstring (a Python float: float(b)), got np.float64(-4.0) (a NumPy sc…` |
| `perceptron` | `score` renvoie un `np.float64` | survit — accepté (§22, session 21) | 0 | — |
| `perceptron` | colonne de 1 ajoutée à la fin | attrapé | 7 | `AssertionError: a column of ones FIRST, then the 1 column(s) of X: expected [[1. , 0.189053]], got [[0.189053, 1. ]] (first difference at…` |
| `perceptron` | `fit` mélange `X` et `y` en place | attrapé | 1 | `AssertionError: expected fit to leave X and y unchanged (shuffle the ORDER of visit, never the arrays), but it modified them` |
| `bandit` | toujours le premier maximum | attrapé | 4 | `AssertionError: expected each of the 3 tied maxima to be chosen about 1/3 of the time, got the counts [30000, 0, 0] out of 30000 (chi-squ…` |
| `bandit` | ε-greedy explore hors du meilleur bras | attrapé | 3 | `AssertionError: expected the best action to be chosen with frequency 1 - epsilon + epsilon/K = 0.9250 (±0.0076), got 0.8987: explore with…` |
| `bandit` | Thompson : Beta(s, f) sans le +1 | attrapé | 3 | `AssertionError: expected each arm chosen with the probability that its draw from Beta(1 + s, 1 + f) is the largest, [0.15 , 0.531, 0.319]…` |
| `bandit` | UCB avec ln(t + 1) | attrapé | 2 | `AssertionError: expected the arm 2 = argmax of q + c*sqrt(ln t / counts) (natural log, c=2, t=57), got 3` |
| `bandit` | `pull` renvoie un `np.float64` | attrapé | 2 | `AssertionError: expected the reward to be a Python float (convert with float(...)): the docstring examples print 1.0, not np.float64(1.0)…` |
| `bandit` | regret calculé avec les récompenses | attrapé | 2 | `AssertionError: regret = cumulative sum of best_mean - means[a_t], with means [0.25, 0.75] and arm 0 always: expected [0.5, 1. , 1.5, 2. …` |
| `bandit` | `pull` avec `<=` | survit — équivalent (`random()` vaut 0 avec une probabilité 2⁻⁵³) | 0 | — |
| `bandit` | `incremental_update` modifie l'estimation en place | attrapé | 1 | `AssertionError: expected the estimate array to stay [0.0, 1.0, 2.0, -4.0] (return a new array), got [0.25, 1.0, 1.5, -2.0]: += changes th…` |

## Bilan

**Ce qui est solide.** Les signatures des 14 modules sont identiques entre squelettes et référence, et aucun test ne passe sur les squelettes (1 324/1 324 échouent). L'API est homogène (classes à la scikit-learn conformes, `rng` / `random_state`, types de retour Python, `ValueError` partout sauf les deux décisions documentées). Les tests sont rapides (31 s pour 1 579), fondés sur des oracles extérieurs (NumPy, SciPy, scikit-learn, PyTorch, `statistics`, `fractions`), à aléa fixé, et, à partir du ch. 3, leurs messages sont remarquables : 97 des 108 mutants sont attrapés, presque toujours avec une première ligne qui nomme la règle violée et les valeurs attendues. `wb.check` est couvert par 70 tests d'infrastructure qui suivent fidèlement les décisions du §22 ; le mécanisme de repli (`impl.py`) et le `conftest` (lignes `-rf` lisibles, fuite d'implémentation corrigée) tiennent. « Run all » des 13 notebooks vides passe, et les seuls avertissements visibles des corrigés sont voulus et expliqués.

**Les trois risques principaux.**
1. **Une bonne réponse refusée à tout le monde (0B.40 e)**, parce qu'une mesure de temps est enregistrée comme réponse et qu'un notebook de solutions a été réexécuté sur une machine chargée ; rien dans la chaîne (`record`, `build_answers --check`) ne l'a signalé. À corriger tout de suite (constat 1), avec l'alerte de `build_answers.py` (constat 8) pour que cela ne se reproduise pas pendant les reconstructions de P1 à P5.
2. **Des trous de tests là où les règles sont arrivées après coup** : les modules des ch. 0A, 0B et 2 ont été testés avant les règles « copies » et « première ligne », et le ch. 2 laisse passer une modification en place qui corrompt silencieusement les données du notebook (constat 2) ; ses messages et ceux de 0A/0B restent parfois muets (constats 5 et 6).
3. **La robustesse des cellules de vérification des chapitres 0A à 6**, ceux que l'apprenant débutant fait en premier : une fonction sans `return` ou de mauvaise forme y produit une trace Python ou un ⏳ trompeur (constats 3 et 4). Le constat 3 se corrige en une ligne dans `checker.py` ; le constat 4 demande de reconstruire les notebooks de 0A, 0B et des ch. 1 à 5 avec le helper `returned` du ch. 7.
