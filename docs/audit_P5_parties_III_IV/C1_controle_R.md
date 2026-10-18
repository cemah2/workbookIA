# Contrôle de l'application de R1 à R11 (lecture seule)

J'ai tout lu à l'état du dépôt à `0546e51`, plus les réexécutions en cours. Je n'ai rien modifié dans le dépôt. Les scripts de lecture sont dans `/tmp/claude-0/-home-claude/d0ad1154-96ff-53e5-aebd-0edf604016be/scratchpad/s43/ctl/`.

Ce que j'ai lancé :
- `syllabus.py check` : 46 chapitres, 2 024 exercices, 0 problème ;
- `pytest tests/infra/test_terminology.py tests/infra/test_wb_nbtools.py` : 70 passed ;
- `git status --short` à la fin : seuls `chapitres/ch18_backprop/05_solutions.ipynb` et `chapitres/ch19_optimiseurs/05_solutions.ipynb` sont modifiés. Ce sont les réexécutions du coordinateur, pas mes fichiers.

## Restes

1. **`src/wb/answers.json` n'est pas à jour : R11 n'est pas encore appliqué là où l'apprenant le voit.** J'ai comparé les lignes `WB_ANSWER` des notebooks de solutions exécutés avec `answers.json`. Il y a exactement trois écarts, et ce sont des messages seulement : 14.23a, 14.23b et 18.R2d.
   - `answers.json:27329` (14.23a) : « … Tes souches sont-elles toutes la même (sample_weight) ? Tes poids somment-ils à 1 à chaque tour ? »
   - `answers.json:27341` (14.23b) : « α nul : … reçoit-elle les poids du tour (sample_weight) ? »
   - `answers.json:35820` (18.R2d) : « relis la documentation de MLPClassifier : le solveur lbfgs a-t-il un learning rate ? »

   Tant que ce fichier n'est pas régénéré, `wb.check` affiche à l'apprenant le nom du correctif (`sample_weight`), et l'ancien message de 18.R2d qui suppose `lbfgs` connu.

   **Correction** : une fois les réexécutions finies, lancer `python tools/build_answers.py` (aucune bonne réponse ne change), puis `--check`.

2. **🔨 14.25 : un appel que les tests comparent pas à pas est sorti de l'énoncé.**
   - Où : `tools/chapters/build_ch14.py:2048`, repris dans les deux notebooks du ch. 14.
   - L'énoncé dit maintenant : « `max(1, int(subsample * n))` lignes tirées sans remise par **un** générateur créé dans `fit` à partir de `random_state` ». L'appel exact n'est plus que dans l'indice 2 (`04_indices.md`, vers la ligne 896).
   - Or `test_gradient_boosting_regressor_subsample_updates_every_sample` (`tests/test_ch14_ensemble.py:1489-1505`) refait `rng = np.random.default_rng(seed)` puis `rng.choice(len(y), size=60, replace=False)` à chaque étape, et compare à 10⁻⁹ près. Un tirage sans remise écrit autrement échoue : `rng.permutation(n)[:k]`, par exemple. D'après R2 (§22), c'est du contrat.
   - **Correction de l'énoncé**, remettre : « les lignes de l'étape (toutes, ou `rng.choice(n, size=max(1, int(subsample * n)), replace=False)` si `subsample < 1`, avec **un** générateur `rng = np.random.default_rng(random_state)` créé dans `fit` : les tests comparent ces tirages étape après étape) ». Puis reconstruire le ch. 14.
   - **Correction de la bible** : la dernière phrase de `docs/BIBLE.md:662`, « Pour un 🔨 dont les tests ne comparent pas le flux aléatoire (ch. 14), l'ordre des tirages n'est pas du contrat », généralise trop. Écrire : « (bagging et forêts, 14.17 et 14.18), l'ordre des tirages n'est pas du contrat ; quand les tests le comparent (sous-échantillon de 14.25, arbres du ch. 13), il l'est ».

3. **Ch. 12 : la glose de scaler n'est pas à la première occurrence.**
   - `chapitres/ch12_preparation/01_fiche.md:11` : « Les ch. 22 à 24 réutiliseront tes scalers » (tableau d'en-tête). La glose n'arrive qu'à la ligne 82 (« un `fit` du scaler (*l'objet qui met à l'échelle*) »).
   - **Correction** : à la ligne 11, écrire « tes scalers (*objets qui mettent à l'échelle*) » ; à la ligne 82, garder « du scaler » sans la glose.

4. **« split » au sens écarté (découpage entraînement/test).**
   - `docs/syllabus/data/cp3.json:353` : « Les sources de fuite de données et leur parade (Pipeline, split avant tout fit, CV groupée ou temporelle) ». C'est repris dans `docs/SYLLABUS.md:2747`.
   - **Correction** : écrire « découpage avant tout fit », puis lancer `syllabus.py build`.

5. **« variable de substitution » sans la glose (*proxy*) dans des documents autonomes.** Les cinq documents de MP3, qui sont autonomes eux aussi, ont la glose.
   - `projets/partie_2_california_validation/README.md:78` : « l'emplacement comme variable de substitution » → « l'emplacement comme variable de substitution (*proxy*) ».
   - `projets/partie_2_california_validation/solution/README.md:65` : « **L'emplacement comme variable de substitution** : » → « **L'emplacement comme variable de substitution** (*proxy*) : ».
   - `data/cards/adult.md:39` : « des **variables de substitution** : » → « des **variables de substitution** (*proxy*) : ».

6. **R1, une petite incohérence chiffrée** dans `chapitres/ch16_feedforward/05_solutions.md:243` (erreurs fréquentes de 16.19).
   - Le texte dit : « l'écart final va de 0 à $10^{-6}$ environ ». La démarche (l. 241), la note du notebook et la mesure de G3 donnent environ 2 × 10⁻⁶.
   - **Correction** : « l'écart final va de 0 à $2 \times 10^{-6}$ environ ».

7. **R3 : le mode d'emploi n'est pas dans `CLAUDE.md`.** G8 demandait de l'y recopier. Une session qui écrit le ch. 21 lit `CLAUDE.md` en premier. Or « Écrire un chapitre » ne cite que `wb.run_pytest` et la règle du helper pytest ; rien sur `wb.nbtools`.
   - **Correction** : y ajouter une puce. « Cellules de vérification (à partir du ch. 21) : `from wb.nbtools import checked, written, same, verdict, returned, pair, StopCheck, refuses, fitted, run_pytest_group`, sans recopier d'outils (BIBLE §22, R3) ; une amélioration va dans `src/wb/nbtools.py` avec son test dans `tests/infra/test_wb_nbtools.py`. »
   - C'est à toi de décider : ni le §22 ni ma consigne ne l'exigent.

8. **Facultatifs, pas des écarts à une décision :**
   - le glossaire attribue baseline au seul ch. 0A (`annexes/glossaire.md:150`) ; on pourrait ajouter 13, où elle est glosée aussi ;
   - « nuages gaussiens » n'est pas glosé à sa seule occurrence du ch. 14 (`ch14_ensembles/05_solutions.md:286`) ;
   - 🐛 15.29 (signalé par G2) est lui aussi « corrigé sur place » et montre un ❌ au premier « Tout exécuter » sans `neighbors.py`. Il n'est pas dans la liste de R11.

## Ce qui est conforme

- **R4 et R5, titres.** Les 15 titres changés sont identiques sur tous les supports où ils figurent :
  - le contrat ;
  - `SYLLABUS.md` ;
  - le tableau de bord (les quiz et 💼 n'y sont pas) ;
  - les scripts ;
  - les cellules Markdown de `03_notebook.ipynb` et `05_solutions.ipynb` (en-têtes et tableaux) ;
  - `02_exercices.md`, `04_indices.md`, `05_solutions.md` et `06_mes_reponses.md`.

  Il ne reste aucune ancienne forme : « & » dans ces titres, « LayerNorm » dans un titre, « coût du backward », « avant le split », « blobs », « (forward/backward) », « Référence naïve ». Les tables d'étapes des projets gardent leurs libellés courts, déjà courts avant ; MP2.2 y devient « Une baseline naïve ».
- **R5, prose.** Sur les `.md`, les JSON, les cellules Markdown, les chaînes du code et les sorties des notebooks, il n'y a plus aucun « Naive Bayes », « couche pleine », « indicateur indirect », « gradient check » non glosé, « LayerNorm » hors code, « le/du/au forward/backward » ni « substitut » au sens de *proxy*. Ce qui reste est voulu :
  - le glossaire en 2ᵉ colonne ;
  - des docstrings et messages en anglais ;
  - le *surrogate* du ch. 11 et de LIME ;
  - le gradient de substitution du ch. 10.

  Les « référence » restantes des ch. 12 à 20 désignent la référence `mylearn`, une solution de référence, une catégorie de référence ou une méthode de référence, jamais une baseline. Les gloses de split (fiche 13, l. 3), de baseline (fiche 13, l. 54), de nuage gaussien (fiche 13, l. 167) et de la Jacobienne (📄 16.10, question 4) sont à leur première occurrence.
- **R6.** CP4.3 a ★★ dans `cp4.json`, le sujet (en-tête), `SYLLABUS.md` et le tableau de bord ; les notebooks, le corrigé et le README n'affichent pas les étoiles.
- **R7.** Dans `ch20.json`, les deux notebooks (« Parcours : R, M, C ») et `SYLLABUS.md` (RMC), 20.13, 20.15, 20.18, 20.19 et 20.23 sont au parcours maths ; `PARCOURS.md` les liste aussi.
- **R8.** 150, 70 et 75 min dans le contrat, les en-têtes et tableaux des deux notebooks, `02_exercices.md` et `SYLLABUS.md`. La fiche 13 affiche « ≈ 29 h », comme `SYLLABUS.md`.
- **R1 et R9.** Les énoncés sont précisés sur tous leurs supports, et les réponses enregistrées sont inchangées :
  - 16.19 a : 10⁻⁵ dans l'énoncé, la vérification, l'indice 3 et la solution ;
  - 18.R2 d, 13.18 (prérequis « ch. 8 (validation croisée) »), 13.14, 18.16 (cas $k$ quelconque, corrigé et variante) et 19.Q12 e ;
  - 20.15 (corrigé de 18.4), CP3.6 c (même phrase au sujet et au notebook) et CP4.9 a (sujet et notebooks, cohérent avec la docstring de `clip_grad_norm` « IN PLACE »).
- **R2.** Les conseils de 19.16 (`np.asarray`, `g.copy()`, `p -= …`, pièges du 0-D) sont bien dans les indices 1 et 2. L'énoncé garde ce que testent les tests ; le seul écart est 14.25 (reste 2).
- **R11.** Les cinq 🐛 (14.23, 15.12, 16.20, 17.17, 20.24) enchaînent leurs points : ⏳ « corrige d'abord le point k ». Les `wb.check` de 14.23 et 15.12, et l'entraînement de contrôle de 20.24, attendent que tous les points passent. Aucun message de vérification ne nomme le correctif ; `sample_weight` n'est plus que dans la note du corrigé. Les notebooks reconstruits contiennent bien ces versions.
- **R3.** `src/wb/nbtools.py` exporte les dix noms de la ligne du §22, plus `filled`, `fr`, `as_numpy`, `shape_text`, `print_answer` et `GroupResult`.
  - `run_pytest_group` refuse un dossier ou un fichier autre que `test_*.py` (`ValueError`), pose `COLUMNS=1000`, a un délai de 300 s (une ligne ❌ s'il est dépassé) et rend un `GroupResult` ;
  - `wb/__init__.py` importe le module.
- **R10.** `neighbors.py`, `naive_bayes.py` et `logistic.py` refusent ±inf (« X contains infinite values », le message de `tree` et de `preprocessing`) et NaN dans `y` (`_check_no_nan_labels`, dans `fit` et `score`). `git diff d021664 --stat -- templates/` est vide : les stubs n'ont pas changé.
