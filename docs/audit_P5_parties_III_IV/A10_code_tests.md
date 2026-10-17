# Audit P5 des parties III et IV — rapport A10 « code »

Relecteur A10 (code et tests). Dépôt `/home/claude/workbookia` au commit `5a17b91`, en lecture seule ; tout ce qui s'exécute tourne dans deux copies `tar` (`A10_code/copy/` pour les mutations, `A10_code/nb/` pour les notebooks, `WB_ROOT` de la copie). Scripts dans `A10_code/scripts/` (`sigs.py`, `provided.py`, `probe.py`, `mutants.py`, `mutants2.py`, `run_mutants.py`, `scenario.py`, `scenarios.py`, `nbhelpers.py`, `nbdiff.py`, `helpers_diff.py`, `msgs.py`, `scan_empty.py`, `warn_scan.py`) ; sorties brutes : `ref_durations.txt`, `probe_ref.txt`, `mutants_out.txt`, `mutants2_out.txt`, `nb_empty_out.txt`, `scenarios_out.txt`.

## Ce qui a été vérifié

| Contrôle | Résultat |
|---|---|
| Stubs ↔ référence (12 modules) : noms publics, signatures, classes de base, docstrings (AST) | identiques partout (0 écart) ; les 32 corps « Provided » (`__init__`, `sigmoid`, `_check_*` d'`optim`) recopiés à l'identique |
| `pytest --impl=ref` par chapitre | 12 : 202 ✅ (1,2 s) · 13 : 474 ✅ (3,8 s) · 14 : 179 ✅ (15,4 s ; le plus lent : `test_random_forest_classifier_and_extra_trees_close_to_sklearn[moons-ExtraTreesClassifier]` 4,0 s) · 16 : 120 ✅ (3,0 s ; `test_mlp_forward_rebuilds_the_predictions_of_sklearn` 1,2 s) · 17 : 259 ✅ · 18 : 232 ✅ (`test_dense_backward_matches_torch_autograd` 0,5 s, import de torch) · 19 : 207 ✅ (5,6 s) · 20 : 102 ✅ (`test_l2_penalty_gradients_reproduce_sgd_weight_decay` 0,6 s) |
| `pytest --impl=stubs` sur les 12 fichiers | 1 774 échecs, 1 succès (`test_sigmoid_matches_scipy_expit` : `logistic.sigmoid` est « Provided ») |
| Mutations (copie, 1ᵉʳ tour) : 77 mutants plausibles de débutant, 4 à 9 par module | 76 tués, 1 survivant (Y6, constat 9) |
| Mutations, 2ᵉ tour (plus subtils) : 17 mutants | 15 tués, 1 équivalent (R11 : en place sur la copie interne), 1 survivant réel (R12, constat 8) |
| Conventions mesurées sur la référence (`probe.py`, ≈ 260 sondes) | tableau 1 ci-dessous |
| « Run all » des 9 notebooks d'exercices et des 2 examens, vides, avec les stubs copiés par `start_chapter.py` | 11/11 sans erreur (de 4 à 80 s) ; seuls des ⏳ et ⏸️, sauf les 🐛 dont le code fautif est donné (❌ attendus : 14.23, 15.12, 16.20, 17.17, 20.24) et ✅ 12.33 (le point de départ respecte les règles du défi) |
| Sorties des 13 notebooks de solutions (ch. 12–20, CP3, CP4, MP3, MP4) | aucune erreur, aucun ❌/⏳ ; un seul avertissement visible (constat 6) |
| Cellules de vérification sous 29 scénarios (nbclient, cellules de préambule + celles de l'exercice ; 25 exercices des 9 chapitres et des 2 examens) | constats 14 à 19 et 21 ; annexe 2 |

### Tableau 1 — module × convention (référence mesurée ; « D » = écrit dans la docstring, « T » = testé)

| Module | Style | `fit` → `self`, attributs `_` | Avant `fit` | NaN dans X | ±inf dans X | NaN dans y | `rng` entier | Entrées intactes | Sortie float64 | Scalaires Python |
|---|---|---|---|---|---|---|---|---|---|---|
| preprocessing | classes sklearn | oui (T) | `RuntimeError` (D, T) | `ValueError` « remove the missing values first (SimpleImputer) » (D, T) ; l'imputer les accepte | `ValueError` (D, T) | — | — | oui (T) | oui (D, T) | `n_components_` int |
| neighbors | fonction + classe | oui (T) | `RuntimeError` (D, T) | `ValueError` (énoncé, T) | **acceptés** | **acceptés comme une classe** | — | oui (T) | distances float64 | `score` float |
| tree | fonctions + classes | oui (T) | `RuntimeError` (D, T) | `ValueError` (D, T) | `ValueError` (ni D ni T) | `ValueError` | oui (`best_split`) | oui (T) | oui | impuretés et gains float (T) |
| naive_bayes | fonctions + classes | oui (T) | `RuntimeError` pour `predict_*_proba` (D, T) ; **`AttributeError`** pour `predict` et `score` | `ValueError` (énoncé, T) | **acceptés** | **acceptés comme une classe** | — | oui (T) | oui | `score` float |
| logistic | classe | oui (T) | `RuntimeError` (D, T) | `ValueError` (énoncé, T) | **acceptés** | `ValueError` « exactly two classes, got 3: [0.0, 1.0, nan] » | — | oui (T) | oui | `score` float |
| ensemble | fonctions + classes | oui (T) | `RuntimeError` (D, T) | `ValueError` (§22, T) | `ValueError` (message de l'arbre) | `ValueError` | oui (D : non ; §22 : oui ; T) | oui (T) | oui | `oob_score_`, `score` float |
| nn/layers | fonctions | — | — | propagés | propagés | — | oui | oui (T) | oui (D, T) | `count_parameters` int (T) |
| nn/activations | fonctions | — | — | propagés (§22, choix assumé) | finis (T) | — | — | copie neuve (T) | oui (T) | tableau 0-D |
| nn/backward | fonctions | — | — | propagés | finis à ±1 000 (T) | cible BCE hors [0, 1] refusée ; cibles de probabilités du CE **non vérifiées** | oui (`gradient_check`) | oui (T) | oui (T) | loss float (T) |
| autograd_mini | classes | — | — | `Value(nan)` accepté | — | — | `random_state` | — | — | `Value.data` float |
| optim | classes + fonctions | — (`step` en place, D) | — | propagés | — | — | — | grads intacts (T), sauf `clip_grad_norm` (en place, D) | tampons float64 (D) | lr et normes float (T) |
| nn/regularization | fonctions + classe | — | — | propagés | — | — | **`AttributeError: 'int' object has no attribute 'random'`** | oui (T) | oui (T) | `l2_penalty` float |

Lecture : les modules sont très homogènes (validation, messages en une ligne qui disent l'attendu, copies, float64, scalaires Python). Les écarts en gras font les constats 1 à 4.

## Constats

### A. API `mylearn` : cohérence entre modules

**1. MINEUR · `solutions/mylearn_ref/nn/regularization.py:97` (et `chapitres/ch20_deep_learning/04_indices.md:691`, indice 3 de 20.16) · `rng = np.random.default_rng() if rng is None else rng`**
Problème : c'est le seul `rng` des parties III et IV qui refuse un entier, contre la convention du §22 (2026-10-04, Q9 : « un paramètre `rng` accepte `None`, un entier ou un `Generator` (`np.random.default_rng(rng)`) »), que suivent `tree.best_split` (l. 621), `ensemble.bootstrap_indices` (l. 218), `nn.layers.init_weights`/`init_mlp` (l. 229, 291) et `nn.backward.gradient_check` (l. 432). Mesuré : `dropout_forward(X, 0.5, rng=0)` lève `AttributeError: 'int' object has no attribute 'random'`, un message brut. La docstring figée dit « Generator or None », donc aucun test n'est à ajouter.
Correction : dans la référence et dans le squelette de l'indice 3, `rng = np.random.default_rng(rng)` (qui rend le même objet pour un `Generator`) ; dans l'énoncé de 20.16 (`tools/chapters/build_ch20.py:991`), « avec le générateur `rng` reçu (`np.random.default_rng(rng)` accepte `None`, un entier ou un `Generator`) ».

**2. MINEUR · `solutions/mylearn_ref/naive_bayes.py:321` et `:502` · `return self.classes_[np.argmax(self.predict_joint_log_proba(X), axis=1)]`**
Problème : avant `fit`, `GaussianNB().predict(X)` et `.score(X, y)` (de même pour `MultinomialNB`) lèvent `AttributeError: 'GaussianNB' object has no attribute 'classes_'`, parce que `self.classes_` est évalué avant l'appel qui vérifie l'ajustement ; `predict_proba`, `predict_log_proba` et `predict_joint_log_proba` lèvent bien `RuntimeError("this GaussianNB is not fitted yet: call fit(X, y) first")`, comme les quatre autres modules (tableau 1). Le test `check_unfitted` (`tests/test_ch13_naive_bayes.py:553`, `:696`) ne vérifie que `predict_joint_log_proba`.
Correction : `joint = self.predict_joint_log_proba(X)` puis `return self.classes_[np.argmax(joint, axis=1)]` (deux lignes, dans les deux classes) ; facultatif : ajouter `"predict"` à la liste de `check_unfitted` (la docstring de la classe ne le promet pas : à ne faire que si l'énoncé de 13.28/13.29 le dit).

**3. MINEUR · `solutions/mylearn_ref/neighbors.py:31-32`, `naive_bayes.py:29-30`, `logistic.py:31-32` · `if np.isnan(X).any(): raise ValueError(...)`**
Problème : les valeurs infinies passent dans le kNN, Naïve Bayes et la régression logistique (`KNeighborsClassifier().fit(X, y).predict(X_avec_inf)` rend des labels), alors que `preprocessing`, `tree` (`tree.py:49`, « X contains infinite values ») et `ensemble` les refusent, comme scikit-learn partout. De même, un `y` qui contient NaN devient une classe de plus dans `KNeighborsClassifier`, `GaussianNB` et `MultinomialNB` (`np.unique` garde NaN), quand `tree` et `ensemble` le refusent (`tree.py:65`) et que `LogisticRegression` répond « y must contain exactly two classes, got 3: [0.0, 1.0, nan] ». Rien n'est documenté ni testé dans un sens ou dans l'autre.
Correction (référence seulement, sans test : le contrat ne change pas) : dans les trois `_check_2d`, ajouter `if np.isinf(X).any(): raise ValueError(f"{name} contains infinite values")` ; refuser un `y` flottant contenant NaN avec le message du §22 (« y contains NaN values: remove the missing values first »). Une ligne au §22 : « ±inf refusés par les estimateurs, comme NaN ».

**4. MINEUR · `solutions/mylearn_ref/tree.py:53` · `raise ValueError(f"X has {X.shape[1]} feature(s), but the tree was fitted with {n_features}")`**
Problème : appelé depuis un ensemble, le message parle de « the tree » : `RandomForestClassifier(...).fit(X3, y).predict(X3[:, :2])` dit « X has 2 feature(s), but the tree was fitted with 3 » (mesuré pour `BaggingClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `AdaBoostClassifier`, `GradientBoostingRegressor`), alors que les autres modules nomment la classe appelée (« but KNeighborsClassifier was fitted with 3 »).
Correction : dans `ensemble.py`, enregistrer `self.n_features_in_` dans chaque `fit` et vérifier `X.shape[1]` au début de `predict_proba`/`predict`/`decision_function`/`staged_predict` avec `f"X has {X.shape[1]} feature(s), but {type(self).__name__} was fitted with {self.n_features_in_}"`.

**5. SUGGESTION · `solutions/mylearn_ref/nn/backward.py` (`softmax_cross_entropy`, docstring : « target probabilities (each row sums to 1) ») · `softmax_cross_entropy(np.zeros((1, 2)), [[0.5, 0.6]])` rend `(0.7625, [[0., -0.1]])`, et `[[1.5, -0.5]]` est accepté**
Problème : la BCE refuse une cible hors de [0, 1] (« every target must lie in [0, 1] »), le cross-entropy n'impose rien à ses cibles de probabilités, alors que sa docstring les dit de somme 1 et que le §22 fixe une tolérance unique de `1e-6` pour « somme à 1 ». PyTorch ne vérifie pas non plus : choix défendable, mais à écrire.
Correction : soit une ligne au §22 (« cibles de probabilités du cross-entropy non vérifiées, comme `F.cross_entropy` »), soit, dans la référence seulement, `ValueError` si une cible est négative ou si `abs(Y.sum(axis=1) - 1) > 1e-6`.

### B. Tests

**6. MINEUR · `tools/chapters/build_ch17.py:64` (cellule 4 de `chapitres/ch17_activations/05_solutions.ipynb`) · `B_EQ_171 = float((B1_171 @ W2_171 + B2_171) @ W3_171 + B3_171)`**
Problème : seul avertissement visible des 13 notebooks de solutions audités : `DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future.` (le produit par `W3_171`, de forme `(2, 1)`, rend un tableau `(1,)`). Une future version de NumPy en fera une `TypeError`.
Correction : `B_EQ_171 = float(((B1_171 @ W2_171 + B2_171) @ W3_171).item() + B3_171)` (comme la ligne 70 du même script), puis réexécuter le notebook de solutions.

**7. MINEUR · `tests/test_ch20_regularization.py:174, 175, 269, 372, 408, 501, 522, 585, 666, 667, 690` · `assert np.allclose(tx.grad.numpy(), want)` (et dix autres `assert` sans message)**
Problème : ces `assert` vérifient l'oracle NumPy du fichier contre PyTorch ; s'ils échouent (une autre version de torch sur Colab), l'apprenant lit `assert False` dans la ligne `FAILED` de `pytest -rf` et croit son code fautif. `tests/test_ch19_optim.py:262` fait la même vérification avec un message qui le dit : « bug in tests/test_ch19_optim.py (not in your code): … ».
Correction : `assert …, "bug in tests/test_ch20_regularization.py (not in your code): the NumPy oracle disagrees with torch"` sur les onze lignes.

**8. MINEUR · `tests/test_ch20_regularization.py:204-207` · `def test_dropout_forward_rng_none_works(reg):`**
Problème : contrôle par mutation (2ᵉ tour, R12) : une référence où `rng=None` devient `np.random.default_rng(0)` (le même masque à chaque appel sans générateur) passe les 102 tests, alors que la docstring (« None means `np.random.default_rng()` ») et l'énoncé de 20.16 (« un **nouveau** masque à chaque appel ») l'excluent. Le ch. 16 a le test correspondant (`tests/test_ch16_layers.py:427`, `test_init_weights_without_rng_is_random_and_leaves_the_global_state`).
Correction : ajouter `test_dropout_forward_without_rng_draws_a_new_mask` : deux appels `dropout_forward(np.ones(1000), p=0.5)` sans `rng` rendent deux masques différents, et `np.random.get_state()` est inchangé ; première ligne : « expected two different masks from two calls without rng (rng=None means a fresh np.random.default_rng(), never a fixed seed) ».

**9. SUGGESTION · `tests/test_ch16_layers.py:539` · `test_init_mlp_draws_the_layers_in_order`**
Problème : survivant du 1ᵉʳ tour (Y6) : un `init_mlp` qui crée un nouveau générateur `np.random.default_rng(rng.integers(2**31))` au lieu de tirer dans celui qu'il reçoit passe les 120 tests ; la docstring dit « Generator shared by all layers, drawn in layer order ». Effet réel limité (les couches restent dans l'ordre et reproductibles), mais l'état du générateur de l'appelant n'avance que d'un tirage.
Correction : un test exact demanderait un oracle du flux de tirage d'`init_weights`, que le contrat ne fixe pas (§22, ch. 16 : 16.23 en fourchette pour cette raison), ou la fonction `init_weights` de l'apprenant comme oracle (interdit par CLAUDE.md). Proposition : une ligne au §22, « `init_mlp` : le partage du générateur entre les couches n'est pas vérifié au-delà de l'ordre des couches ».

**10. MINEUR · `tests/test_ch16_layers.py:64-70` (`as_array`) et `tests/test_ch19_optim.py:74-80` (`assert_close`) · `return np.asarray(result, dtype=float)`**
Problème : `np.asarray(None, dtype=float)` vaut `array(nan)`, de forme `()` : une fonction sans `return` reçoit « expected shape (1, 3), got shape () » (mesuré, scénario 16.13 : `test_dense_forward_docstring_examples`, « expected shape (1, 3), got shape () »). Les helpers des ch. 17, 18 et 20 disent « got None (did you forget the return?) » (`tests/test_ch17_activations.py:79`, `tests/test_ch20_regularization.py:70`).
Correction : en tête des deux fonctions, `if result is None: _fail(msg, "expected a number or an array, got None (did you forget the return?)")`.

**11. MINEUR (latent) · `tests/test_ch16_layers.py:87`, `test_ch17_activations.py:108`, `test_ch18_autograd_mini.py:87`, `test_ch18_backward.py:102`, `test_ch19_optim.py:95`, `test_ch20_regularization.py:90` · `except error:` placé avant `except NotImplementedError:`**
Problème : les versions des ch. 12 à 14 mettent `except NotImplementedError: raise` **avant** `except error`, avec ce commentaire : « NotImplementedError is a RuntimeError; shown as "⏳" » (`tests/test_ch13_neighbors.py:158`). Dans les six fichiers de la partie IV, l'ordre est inversé : un futur `assert_raises(RuntimeError, …)` (le cas « non ajusté » des estimateurs) passerait sur un squelette vide. Aucun test actuel n'est concerné (aucun `assert_raises(RuntimeError` dans ces fichiers), d'où « latent ».
Correction : inverser les deux clauses dans les six fichiers (texte des ch. 12 à 14). Plus largement (SUGGESTION) : les outils des tests sont recopiés dans les 12 fichiers et ont divergé (empreintes AST, `scripts/helpers_diff.py` : `assert_close` en 6 versions, `assert_raises` en 6, `call` en 5, `_short` en 3) ; un `tests/_helpers.py` (le dossier `tests/` est sur le `sys.path` de pytest) les garderait alignés.

**12. SUGGESTION · `tests/test_ch16_layers.py:379-387` · `@pytest.mark.parametrize("method, torch_init", [("glorot_uniform", lambda t: …), …])`**
Problème : ids illisibles `test_init_weights_matches_the_spread_of_torch_init[glorot_uniform-<lambda>]` (règle des ids lisibles de CLAUDE.md).
Correction : `ids=lambda v: v if isinstance(v, str) else ""` ou `pytest.param(..., id="glorot_uniform")`.

**13. SUGGESTION · `tests/test_ch19_optim.py:45` · `from transformers import get_cosine_schedule_with_warmup` au chargement du module**
Problème : chaque cellule 🔨 du ch. 19 relance pytest (`run_optim_tests`) et réimporte `transformers` : mesuré sur la machine de l'audit, `import torch` 4,7 s puis `transformers` + 13,4 s ; `pytest tests/test_ch19_optim.py -k cosine` prend 22,8 s, contre 10,8 s pour un `-k` du ch. 20. Le module ne sert qu'à deux tests de `cosine_lr`.
Correction : importer `transformers` dans une fixture (ou dans `test_cosine_lr_matches_hugging_face_with_warmup`) et garder l'oracle NumPy de la ligne 793 pour les autres.

### C. Cellules de vérification des notebooks

Scénarios exécutés avec nbclient dans la copie (préambule du notebook + cellules de l'exercice seulement, ce que fait un apprenant qui rouvre le notebook ; module de l'apprenant = référence + une erreur) : 12.16, 12.18, 12.20, 12.27, 13.14, 13.24, 13.32 (relancée), 14.12, 14.24, 15.13, 16.12, 16.13 (deux fois, dont noyau redémarré), 16.17, 17.13, 17.15, 18.12 (deux fois), 18.16, 18.26, 19.16, 19.18 (relancée), 19.19, 20.16, 20.22 (trois fois, dont relancée), CP3.7, CP4.9 (relancée) : 29 scénarios, 25 exercices, 9 chapitres et les deux examens. Ce qui tient : une erreur de l'apprenant devient une ligne ❌ et les tests tournent ensuite dans tous les chapitres 13 à 20 ; les cellules relancées donnent exactement les mêmes sorties (19.18, 20.22, 13.32, CP4.9) ; les examens restent exécutables jusqu'au bout avec un code faux (`safe`, `ready`, `restore_data`) ; les messages des tests disent l'attendu sur leur première ligne. Défauts :

**14. MINEUR · `tools/chapters/build_ch12.py:968, 987, 1009, 1597, 1616` (`CHECK_16`, `CHECK_17`, `CHECK_18`, `CHECK_26`, `CHECK_27`) · `CHECK_27 = r'''with wb.attempt("12.27"):` … `    run_mylearn_tests("test_pca_")'''`**
Problème : les cinq 🔨 du ch. 12 sont restés au schéma d'avant le ch. 13 (§22, 2026-10-16 : « `with checked(id)` … et les tests se lancent dans un bloc séparé `if written(sonde)` », « ch. 13, puis chapitres suivants ») : tout tient dans un seul `with wb.attempt`, avec `run_mylearn_tests` à la fin. Mesuré : (a) un `PCA.transform` qui rend la transposée fait planter la cellule sur la trace brute `ValueError: Z has 200 column(s), but the PCA keeps 5 component(s)` (levée par `inverse_transform`) et **les tests ne tournent pas** ; (b) un `SimpleImputer.transform` sans `return` donne `TypeError: ufunc 'isnan' not supported for the input types…` (le message ne contient pas « NoneType », donc `wb.attempt` laisse passer la trace) et pas de tests non plus ; (c) une classe à moitié écrite (`inverse_transform` encore en `NotImplementedError`) affiche ⏳ et masque les tests de `fit` et `transform` déjà justes.
Correction : reconstruire ces cinq cellules sur le modèle du ch. 13 (`build_ch13.py:631-660`) : comparaisons dans `with checked("12.27"):`, puis `if written(lambda: mylearn.preprocessing.PCA().fit(np.eye(3))): run_mylearn_tests("test_pca_") else: print("⏳ Ex 12.27 : les tests attendent le fit de ta PCA.")` ; ajouter `checked` et `written` à la cellule d'outils du ch. 12 ; réexécuter les solutions (`ref_check` existe déjà au ch. 13).

**15. MINEUR · `tools/chapters/build_ch20.py:835` et `:1456` · `out_16, mask_16 = R.dropout_forward(np.ones((2, 4)), p=0.5, rng=np.random.default_rng(0))` ; `out_22, cache_22 = R.batchnorm_forward(x_22, …)`**
Problème : un résultat qui n'est pas un couple est déballé en silence quand c'est un tableau à deux lignes : un `dropout_forward` qui rend seulement `out` (forme `(2, 4)`) donne d'abord « ❌ Ex 20.16 : l'exemple de la docstring : attendu des 0 et des 2 = 1/(1 - 0,5), avec out = x * mask ; reçu [0.0, 2.0]. » (la valeur reçue est exactement celle attendue : le message contredit le diagnostic), puis « ton code lève ValueError : too many values to unpack (expected 2) » ; même chose en 20.22 (« attendu la forme (2, 2), reçu la forme (2,) »). Le ch. 18 a l'outil qu'il faut (`pair`, `build_ch18.py:522`), et 18.12 dit bien « mse_loss([1, 2], [0, 0]) doit rendre un tuple (loss, grad), reçu float ».
Correction : copier `pair` dans la cellule d'outils du ch. 20 et écrire `out_16, mask_16 = pair("20.16", "dropout_forward", R.dropout_forward(...))` (de même pour les appels de `batchnorm_forward` en 20.22), qui affiche « dropout_forward doit rendre le couple (out, mask), reçu ndarray » et arrête la vérification.

**16. MINEUR · `tools/chapters/build_ch13.py:1541-1543` (13.24) · `elif not np.array_equal(np.asarray(mine_24.predict(grid_24)), oracle_24.predict(grid_24)):` → « même profondeur et mêmes feuilles que scikit-learn, mais d'autres prédictions sur la grille : un seuil ou une feature diffère »**
Problème : un `predict` qui rend une liste de forme `(n, 1)` (mesuré) reçoit ce diagnostic faux : l'arbre est juste, seule la forme de la sortie diffère.
Correction : avant la comparaison, `pred_24 = np.asarray(mine_24.predict(grid_24))` puis `if pred_24.shape != (len(grid_24),): print(f"❌ Ex 13.24 : predict doit rendre un tableau ({len(grid_24)},), un label par ligne ; reçu la forme {pred_24.shape}.")`.

**17. MINEUR · `tools/chapters/build_ch16.py:667` (16.13 ; même schéma en 16.14, l. 783 et 792) · `z_13 = np.asarray(mylearn.nn.layers.dense_forward(X_13, W_13, b_13))`**
Problème : envelopper le résultat dans `np.asarray` cache un `return` oublié : mesuré, « docstring examples: None and None » puis « ❌ Ex 16.13 : attendu : un tableau (50, 30) en float64, pas () en object. », jamais « as-tu oublié le return ? » (que donnent `returned` aux ch. 12 à 16 et `same` aux ch. 17 à 20). Les tests du ch. 16 ne le disent pas non plus (constat 10).
Correction : `z_13 = mylearn.nn.layers.dense_forward(X_13, W_13, b_13)` puis `if not returned("16.13", "dense_forward", z_13): ...` (l'outil existe dans la cellule d'outils du ch. 16), avant tout `np.asarray`.

**18. SUGGESTION · `tools/chapters/build_ch16.py` (cellule « Tools » de la partie A) et tous les notebooks des ch. 13 à 20 · après un redémarrage du noyau, la cellule de setup puis directement celle de 16.13 donnent `NameError: name 'checked' is not defined`**
Problème : les examens disent quoi faire (`ready()` : « exécute d'abord les cellules de la partie A ») ; les chapitres laissent une trace brute à un débutant.
Correction : en tête de chaque cellule de vérification, rien à changer ; il suffit d'ajouter dans le Markdown d'introduction de la partie A (ou dans `mylearn_howto()`) : « Après un redémarrage du noyau : exécute la cellule de configuration, puis la cellule d'outils de la partie A (`# Tools for the notebook exercises`), avant toute vérification. » ; ou, plus robuste, mettre les outils dans un module (constat 20) importé par la cellule de setup.

**19. SUGGESTION · `tools/chapters/build_ch18.py:882, 885, 888` (18.16) · `err_16 = float(B.gradient_check(loss_16, W_16, exact_16))`**
Problème : quand la fonction rend un tableau `(1,)` (erreur signalée juste avant par « gradient_check doit rendre un float Python »), ces conversions affichent trois `DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated` dans la sortie de l'apprenant.
Correction : `err_16 = float(np.asarray(B.gradient_check(...)).ravel()[0])`, ou sauter ces lignes quand le premier contrôle de type a échoué.

### D. Outils des notebooks, scripts `tools/chapters` et `src/wb`

**20. MAJEUR (restructuration, à valider) · outils de notebook recopiés dans chaque cellule d'outils (`tools/chapters/build_ch12.py` … `build_ch20.py`) · versions divergentes mesurées par empreinte AST (`scripts/nbhelpers.py`) :**

| Outil | Copies | Variantes |
|---|---|---|
| `verdict`, `filled` | ch. 12–20 | 1 |
| `checked` | ch. 13–20 | 4 : ch. 13–14 (indice « une fonction ou une méthode oublie-t-elle son return ? » pour toute erreur qui contient `NoneType`), ch. 15, ch. 16–19 et ch. 20 (sans cet indice) |
| `same` | ch. 17–20 | 3 : ch. 17–18 (« attendu autre chose, reçu un tableau (4, 6) (plus grand écart 0.759) », mesuré en 17.15), ch. 19 et 20 (« première différence à l'indice (0, 0) : attendu …, reçu … », ajouté en session 39) |
| `print_answer` | ch. 12–19 | 3 |
| `run_mylearn_tests` / `run_optim_tests` / `run_reg_tests` | ch. 12–20 | 6 (le 💡 « un attribut est lu sur None : une méthode oublie-t-elle son return ? » des ch. 12–14 a disparu aux ch. 16–20) |
| `ref_check` | ch. 13–20 | 4 · `as_printed` : 3 · `mylearn_howto` : 7 |
| `pair`, `refuses`, `letters`, `shape_text`, `quiet`, `as_numpy` | 1 à 3 chapitres chacun | — |

Problème : chaque amélioration apportée par une vérification (session 39 pour `same`, ch. 13 pour l'indice `NoneType`) ne vaut que pour les chapitres reconstruits ensuite ; les ch. 17 et 18 gardent le message le plus pauvre (mesuré), les ch. 16 à 20 ont perdu l'indice du `return` oublié pour une `AttributeError` (mesuré en 18.26 : « ton code lève AttributeError : 'NoneType' object has no attribute 'backward' », sans piste ; les tests le disent ensuite). Les cellules d'outils font 150 à 200 lignes que l'apprenant fait défiler à chaque chapitre.
Correction : (a) tout de suite, sans changer de structure : reporter dans les scripts des ch. 16 à 20 l'indice `NoneType` de `checked` (`build_ch13.py:631-645`) et le 💡 de `run_mylearn_tests` (`build_ch13.py:661-688`), et dans ceux des ch. 17 et 18 le `same` du ch. 19 (`build_ch19.py:638-663`) ; reconstruire et réexécuter ; (b) à valider : un module `wb.nbtools` (testé dans `tests/infra/`) qui regroupe `verdict`, `filled`, `fr`, `returned`, `checked`, `written`, `same`, `pair`, `refuses` et un `run_mylearn_tests(test_file, keyword, impl)` unique, importé par une ligne de la cellule d'outils ; les fonctions `build`, `main`, `ref_check` et `as_printed` des scripts (identiques ou presque dans les ch. 12 à 20) passent dans `tools/chapters/chapter_kit.py`.

**21. MINEUR · `src/wb/checker.py:1007-1012` (`attempt`) · `except TypeError as exc: … if "NoneType" not in str(exc): raise`**
Problème : la règle « une valeur `None` là où un résultat était attendu » ne couvre que `TypeError` ; l'autre forme, tout aussi fréquente, `AttributeError: 'NoneType' object has no attribute 'transform'` (un `fit` sans `return self` suivi de `.transform`, un `__add__` sans `return`), sort en trace brute dans toute cellule protégée par `wb.attempt` seul (les 🔨 du ch. 12, constat 14 ; les cellules d'essai des examens, par exemple `checkpoints/partie_3/02_examen_notebook.ipynb`, cellule 10).
Correction : ajouter `except AttributeError as exc: if "'NoneType' object has no attribute" not in str(exc): raise` puis le même message ❌ ; un test dans `tests/infra/test_wb_check.py` (une fonction sans `return` suivie d'un attribut).

**22. MINEUR · `src/wb/synth.py:250-251` · `b[:30] = …` ; `b[30:70] = start + np.abs(6 * rng.standard_cauchy(40)).clip(0, 25)`**
Problème : `optimizer_curves(n_updates=…)` est un paramètre public, mais toute valeur inférieure à 70 plante : `n_updates=69` → `ValueError: could not broadcast input array from shape (40,) into shape (39,)`, `n_updates=20` → `operands could not be broadcast together with shapes (20,) (30,)`. Le test (`tests/infra/test_wb_synth.py`) n'essaie que 1 200 et 600.
Correction : `if n_updates < 100: raise ValueError(f"n_updates must be >= 100, got {n_updates}")` en tête de la fonction, et un `pytest.raises` dans le test.

**23. MINEUR · `src/wb/synth.py:255-256` et `tests/infra/test_wb_synth.py:102-103, 119-124` · `warm = int(0.12 * n_updates)` ; `lr = np.where(t < warm, t / warm, 0.5 * (1 + np.cos(…)))` ; `# overfitting`, `# underfitting`, `# the dead network stays at log 10`, `# one example per update: the noisiest`, `# a noisy validation`, `# validation below training`**
Problème : le §22 (sessions 37, 39 et 41) a déplacé les courbes des 📈 18.22, 19.23 et 20.25 dans `wb.synth` « sans commentaire qui nomme ses cas », pour qu'aucun code lisible ne donne la réponse ; mais les noms de variables de la courbe D (`warm`, `lr`, cosinus) donnent la réponse de 19.23 a) (« warmup_23 : la lettre de la courbe »), et les commentaires du test d'infrastructure nomment les cas de 19.23 et 20.25. Le source est à un `wb.synth.optimizer_curves??` de l'apprenant.
Correction : renommer en `k = int(0.12 * n_updates)` et `s = …` ; remplacer les commentaires du test par des repères neutres (`# curve B: flat after the burst`, etc.) ou les retirer.

**24. SUGGESTION · `src/wb/datasets.py:389-397` (`_cifar_from_huggingface`), appelé par 🔮 20.26 (`build_ch20.py:1961`) · la première exécution affiche sur stderr `TqdmWarning: IProgress not found…`, `Warning: You are sending unauthenticated requests to the HF Hub…` et sept barres « Generating train split »**
Problème : bruit en rouge dans le notebook d'un débutant, au milieu d'un 🔮 (mesuré au « Run all » du ch. 20 dans une copie sans cache).
Correction : dans `_cifar_from_huggingface`, `from datasets.utils.logging import disable_progress_bar, set_verbosity_error` puis les deux appels, et `os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")` avant `load_dataset` ; ou un `print("ℹ️ Premier téléchargement de CIFAR-10 (≈ 170 Mo), une seule fois.")` qui annonce le reste.

**25. SUGGESTION · `tools/chapters/build_ch12.py:530`, `build_ch13.py:668`, `build_ch14.py:647`, `build_ch16.py:574`, `build_ch17.py:543`, `build_ch18.py:591`, `build_ch19.py:745`, `build_ch20.py:585` (et les ch. 0A à 11) · `result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=…)` sans `timeout`**
Problème : `wb.run_pytest` a un délai de 120 s depuis l'audit précédent (A10, suggestion 13 appliquée), pas les `run_*_tests` des notebooks : une boucle infinie dans le code de l'apprenant (un `while` de gradient sans `max_iter`, un ordre topologique qui boucle sur un cycle) bloque la cellule sans message.
Correction : `timeout=300` et `except subprocess.TimeoutExpired: print("❌ les tests n'ont pas fini en 5 min : une boucle de ton code ne s'arrête-t-elle pas ?")` (dans `wb.nbtools` si le constat 20 b est retenu).

## Annexe 1 — mutations (copie `A10_code/copy/`, `scripts/mutants.py` et `mutants2.py`, `pytest -x --impl=ref` sur le fichier du module)

| Module | Mutants (tous tués sauf mention) |
|---|---|
| preprocessing | ddof = 1 du scaler ; colonne constante repérée par `var == 0` ; `clip` ignoré ; égalités de `most_frequent` → la plus grande ; variance de la PCA sur n ; seuil `n_components` en ≥ ; blanchiment non défait ; `transform` recentré sur les nouvelles données (fuite) ; colonne constante du min-max divisée par `range + 1e-12` (9/9) |
| neighbors | euclidienne sans racine ; `argsort` non stable ; poids 1/d² ; distance nulle → poids 1 pour tous ; égalités de vote → dernière classe (5/5) |
| tree | entropie en nats ; gain non pondéré (l'erreur du livre) ; seuil = valeur gauche ; `max_depth` décalé de 1 ; `min_samples_split` en ≤ ; importances non pondérées (6/6) |
| naive_bayes | ddof = 1 ; `epsilon_` sur les variances par classe ; dénominateur de Laplace sans α·p ; prior uniforme log 1 ; `logsumexp` sans décalage ; ½ oublié dans log σ² (6/6) |
| logistic | ordonnée pénalisée ; gradient sommé ; ½ de la pénalité oublié dans l'objectif ; `tol` en ≤ ; p = 0,5 → `classes_[1]` (5/5) |
| ensemble | égalités du vote → plus grand label ; bootstrap sans remise ; SAMME sans ln(K − 1) ; poids d'AdaBoost jamais renormalisés ; résidus depuis `init_` ; `train_score_` sur toutes les lignes ; sous-échantillon avec remise ; même `random_state` pour tous les membres ; `fit` du boosting sans learning rate (9/9) |
| nn/layers | He avec n_out ; activation cachée sur la sortie ; biais compté par entrée ; ordre topologique FIFO ; biais à 0,01 ; **Y6 : `init_mlp` qui recrée un générateur (survit, constat 9)** (5/6) |
| nn/activations | ReLU′(0) = 1 ; leaky ReLU′(0) = 1 ; seuil de softplus en ≥ ; softmax sans décalage ; ELU′(0) = 1 au lieu de α ; tanh′ calculée sur z ; `log_softmax` = log(softmax) ; constante 0,0447 de GELU (8/8) |
| nn/backward | gradient de la MSE sans 2 ; BCE et CE non divisés par n ; db moyenné ; dérivée cachée sur la sortie ; restauration par `+ eps` ; BCE naïve ; différence avant ; `n_checks` avec remise (9/9) |
| autograd_mini | `*` qui écrase les gradients ; puissance sans l'exposant ; tanh′ sur l'entrée ; relu′(0) = 1 ; ordre inverse ; gradient de la racine cumulé ; `__rsub__` de signe faux ; `zero_grad` vide ; activation sur la dernière couche de `MiniMLP` (9/9) |
| optim | momentum amorti (moyenne mobile) ; ε sous la racine (Adam, RMSprop) ; AdamW couplé ; clipping sans eps ; warmup décalé de 1 ; `initial_accumulator_value` ignoré ; Adam sans correction du biais ; step decay en avance d'un pas ; bold driver qui baisse au bord (10/10) |
| nn/regularization | dropout non inversé ; garder avec la probabilité p ; variance glissante biaisée ; statistiques réaffectées (pas en place) ; ε hors de la racine ; L2 sans ½ ; early stopping non strict ; références au lieu de copies ; passe arrière de la batchnorm sans les statistiques du batch ; layer norm à ddof = 1 ; R11 `dropout_backward` en place sur `dout` (**équivalent** : la référence travaille sur sa copie `float64`) ; **R12 : `rng=None` → graine fixe (survit, constat 8)** (10/12, dont 1 équivalent) |

Total : 94 mutants, 91 tués, 1 équivalent, 2 survivants réels (constats 8 et 9).

## Annexe 2 — scénarios des cellules de vérification (`scripts/scenario.py`, `scenarios.py`, `exam_scenarios.py` ; sortie complète dans `scenarios_out.txt` et `exam_scen_out.txt`)

| Exercice | Erreur injectée | Ce qu'affiche la cellule | Verdict |
|---|---|---|---|
| 12.16 | `fit` sans `return self` | comparaisons ✅ (via `fitted`), puis 💡 « une méthode oublie-t-elle son return ? » et les tests | ✅ |
| 12.18 | `transform` sans `return` | trace `TypeError: ufunc 'isnan'…`, pas de tests | constat 14 |
| 12.20 | `prepare_20` rend 4 colonnes | « prepare_20 doit renvoyer une matrice (100, 7) … reçu (100, 4) » | ✅ |
| 12.27 | `transform` transposé | trace `ValueError`, pas de tests | constat 14 |
| 13.14 | `IndexError` dans `predict_proba` | ❌ « ton code lève IndexError », puis les tests | ✅ |
| 13.24 | `predict` rend une liste (n, 1) | « un seuil ou une feature diffère » | constat 16 |
| 13.32 | aucune, cellule relancée | sorties identiques | ✅ |
| 14.12 | `plurality_vote` sans `return` | « renvoie None : as-tu oublié le return ? », tests | ✅ |
| 14.24 | `decision_function` (n,) | « doit rendre un tableau (100, 3) … pas (100,) », tests | ✅ |
| 15.13 | listes imbriquées, dict, chaîne, tuple de 3 | messages de forme et de type de `wb.check` | ✅ |
| 16.12 | `count_parameters` rend `np.float64` | ✅ « 32.0 paramètres » dans la cellule, ❌ du test « expected a Python int » | ✅ (le test tranche) |
| 16.13 | `dense_forward` sans `return` | « pas () en object » | constat 17 |
| 16.13 | noyau redémarré, cellule d'outils sautée | `NameError: name 'checked' is not defined` | constat 18 |
| 16.17 | activation aussi sur la sortie | ❌ précis, tests | ✅ |
| 17.13 | `sigmoid` rend une liste | ✅ dans la cellule, ❌ du test « new float64 array » | ✅ |
| 17.15 | softmax sur l'axe 0 | « attendu autre chose, reçu un tableau (4, 6) » | constat 20 |
| 18.12 | ne rend que la loss / pas de `return` | « doit rendre un tuple (loss, grad), reçu float » / « renvoie None » | ✅ |
| 18.16 | rend un tableau (1,) | ❌ juste, puis trois `DeprecationWarning` | constat 19 |
| 18.26 | `__add__` sans `return` | « ton code lève AttributeError : 'NoneType'… » sans piste ; les tests disent « did you forget the return? » | constat 20 |
| 19.16 | `p = p - lr * g` | « les paramètres n'ont pas bougé en place (écris p -= …) » | ✅ |
| 19.18 | aucune, cellule relancée | sorties identiques | ✅ |
| 19.19 | `ZeroDivisionError` sans warmup | ❌ une ligne, tests | ✅ |
| 20.16 | ne rend que `out` | « attendu des 0 et des 2 … reçu [0.0, 2.0] » puis « too many values to unpack » | constat 15 |
| 20.22 | statistiques réaffectées / ne rend que `out` / cellule relancée | ❌ précis / constat 15 / sorties identiques | ✅ / constat 15 / ✅ |
| CP3.7 | `min_cost_threshold` sans `return` | « ne renvoie rien : as-tu oublié le return ? » | ✅ |
| CP4.9 | `adamw_step` pas en place, `train_epoch` rend une liste ; cellule relancée | sept ❌ précis, partie B jusqu'au bout, sorties identiques | ✅ |

## Bilan

**Ce qui est solide.** Les stubs, la référence et leurs docstrings sont identiques à l'octet près pour les 12 modules ; les conventions du §22 (classes à la scikit-learn, `RuntimeError` avant `fit`, NaN refusés, copies, float64, scalaires Python, messages qui disent l'attendu) sont appliquées presque partout, et les énoncés des 🔨 écrivent le contrat que les tests vérifient (« Contrat (la docstring, plus ces précisions que vérifient les tests) »). Les tests sont d'un niveau rare : 1 775 tests, oracles extérieurs (scikit-learn, SciPy, PyTorch en float64, pseudo-code de `torch.optim` vérifié contre torch à chaque test), bords exacts, ids lisibles, et 91 mutants sur 94 tués ; ils tournent en 37 s en tout. Les 11 notebooks vides vont au bout, les cellules relancées sont idempotentes, et les examens restent exécutables avec un code faux.

**Les trois risques principaux.**
1. **La dérive des outils de notebook** (constat 20) : copiés dans chaque chapitre, ils divergent (4 versions de `checked`, 3 de `same`, 6 de `run_*_tests`) et chaque correction faite par une vérification ne vaut que pour les chapitres suivants ; les ch. 17 et 18 gardent les messages les plus pauvres, le ch. 12 le schéma d'avant `checked`/`written` (constat 14). Le risque grandit avec chaque chapitre de la partie V.
2. **Les retours de mauvaise forme mal diagnostiqués** dans les cellules (constats 15, 16, 17, 21) : un `return` oublié ou un tuple manquant donne parfois un message qui contredit la valeur reçue ; les tests rattrapent toujours, mais l'apprenant lit d'abord la cellule.
3. **De petites divergences de la référence aux conventions** (constats 1 à 4 : `rng` entier du dropout, `AttributeError` de Naïve Bayes, ±inf et NaN dans `y`, message « the tree » des ensembles) : invisibles pour les tests, elles apparaissent à l'apprenant qui compare sa librairie à la référence ou lit `solutions/`.

Vérification finale : `git -C /home/claude/workbookia status --short` vide (aucun fichier du dépôt modifié ; tout a tourné dans les copies).
