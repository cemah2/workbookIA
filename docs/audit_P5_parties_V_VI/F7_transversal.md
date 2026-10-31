# F7 (transversal) — audit P5 des parties V et VI

## En-tête

**Fichiers touchés**
- Annexes : `annexes/glossaire.md`, `annexes/formulaire.md`, `annexes/erreurs_frequentes.md`, `annexes/cheatsheets/pytorch.md`, `annexes/ressources.md`, `annexes/metiers.md`.
- Référence `mylearn` : `solutions/mylearn_ref/rl.py` (`rng`), `sequences.py` (NaN), `conv.py` (messages des poids).
- Tests des modules : `tests/test_ch22_rnn.py` (un test par cellule), `tests/test_ch22_sequences.py` (type de `temperature_softmax`), `tests/test_ch25_vae.py` (slerp à 10⁻³ rad), `tests/test_ch26_rl.py` (messages de type).
- `wb` : `src/wb/checker.py` (`attempt`, `NONE_TYPE_ERROR`), `src/wb/nbtools.py` (`checked`, `as_numpy`, `same`, `refuses`, `run_pytest_group`).
- Tests d'infrastructure : `tests/infra/test_wb_nbtools.py`, `tests/infra/test_wb_check.py`, `tests/infra/test_terminology.py` (sources), `tests/infra/test_repo_structure.py` (groupes `-k`), `tests/infra/test_chapter_kit.py` (nouveau).
- Kit : `tools/chapters/chapter_kit.py` (🚀 et « Fil rouge » lus dans le contrat, `Ex.table_row`).
- `README.md`, `00_setup/INSTALL_LOCAL.md`, `00_setup/COLAB.md`.

**Changement d'affichage des notebooks (à reconstruire, ch. 21 à 29)** : `Ex.header` du kit ajoute « 🚀 » après « ⏱️ N min » quand le contrat a `"gpu": true`, et affiche le « Fil rouge » du contrat quand le script n'en donne pas (avec « synth » écrit « synthétique »). Le tableau de la cellule 0 n'est pas dans le kit : chaque script le construit ; `Ex.table_row(part_key)` le fait désormais (voir « Pour une autre zone »). Les messages de `checked`, `same` et l'ordre des échecs de `run_pytest_group` changent aussi (sorties des cellules de vérification seulement).

**Reconstructions** : aucune dans le dépôt. Dans une copie (`F7_transversal/COPIE/`), `build_ch21.py` à `build_ch29.py` : les neuf tournent sans erreur ; 20 en-têtes portent 🚀 (les 20 exercices `gpu` des parties V et VI), plus aucun « Fil rouge : synth ». « Tout exécuter » à vide des `03_notebook.ipynb` reconstruits des ch. 22 et 26 (outils `nbtools` modifiés) : ✅, aucune erreur.

**Vérifications** (sous le verrou) : 
- `pytest --impl=ref tests/test_ch2*.py` : **951 passed** (ch. 21 à 27 ; les ch. 23, 24, 28 et 29 n'ont pas de module).
- `pytest --impl=stubs tests/test_ch2*.py` : **951 failed**, 0 réussi (les tests ne sont pas vides).
- `pytest tests/infra` : 501 passed, 5 skipped, **3 failed, aucun de ma zone** : `test_answers_json_is_valid_and_up_to_date` (le `05_solutions.ipynb` du ch. 21 reconstruit par F1 attend sa réexécution : « cellule taguée answer sans ligne WB_ANSWER »), et `test_generated_documents_are_up_to_date` pour `SYLLABUS.md` et `PARCOURS.md` (549 h → 550 h : temps de lecture changés par d'autres zones ; `syllabus.py build` revient au coordinateur).
- `test_terminology.py` : 21 passed (avec les nouvelles sources). `test_wb_nbtools.py`, `test_wb_check.py`, `test_chapter_kit.py` (nouveau), `test_repo_structure.py` : verts.
- Mutants : le slerp à seuil `1e-2` et un `temperature_softmax` qui rend un tenseur échouent désormais ; la référence passe.
- `syllabus.py check` : 0 problème ; `export_flashcards.py --check` : 757 cartes.
- `git status --short mon_travail` : vide.

**Réponses enregistrées : aucune ne change** (aucune cellule `answer` touchée ; `build_answers.py --check` : ❌ pour la seule raison du ch. 21 ci-dessus, hors zone).

## Tableau de décision

| Constat (relecteur et numéro) | Décision | Fichiers touchés |
|---|---|---|
| A10 1 (`rng` entier refusé par `rl.py`) | appliqué : `np.random.default_rng(rng)` aux trois endroits, référence seule (la docstring dit « Generator or None » : aucun test) | `solutions/mylearn_ref/rl.py` |
| A10 2 (NaN de `sample_from_logits`) | appliqué : `ValueError` sur des logits NaN, référence seule ; ligne du §22 proposée (règle des NaN et des infinis des modules V et VI). Refus de ±inf dans `vae` : non retenu (proposé seulement : des expériences du ch. 25 peuvent produire des `log_var` infinies) | `solutions/mylearn_ref/sequences.py` |
| A10 3 (message d'un poids 3-D) | appliqué : `_batch(..., layout=...)`, disposition propre à chaque poids | `solutions/mylearn_ref/conv.py` |
| A10 4, 5, 7, 18 (ch. 24 : ✅ précoces de 24.16, 🐛 24.19, `UserWarning`, dépendance de 24.18) | hors zone (F2) ; la partie `checked` de A10 18 (première ligne vide) est appliquée | `src/wb/nbtools.py` |
| A10 6 (🐛 28.21) | hors zone (F4) | — |
| A10 7 (`UserWarning` de 25.38) | hors zone (F3) | — |
| A10 8 (mutant du slerp) | appliqué : test paramétré à 0,05 et 10⁻³ rad ; le mutant `< 1e-2` échoue désormais (vérifié), la référence passe. L'énoncé de 25.20 (« de l'ordre des erreurs d'arrondi ») et la docstring (« parallel or anti-parallel ») le disent | `tests/test_ch25_vae.py` |
| A10 9 (3 tests de `rnn` hors des groupes `-k`) | appliqué : trois tests nommés d'après leur fonction (`test_rnn_cell_forward_…`, `test_lstm_cell_forward_…`, `test_gru_cell_forward_…`), attrapés par les groupes publiés : **le notebook du ch. 22 ne change pas** (seuls ses comptes « N passed » bougent à la réexécution) ; test d'infrastructure qui vérifie que les groupes d'un notebook couvrent tout son fichier de tests | `tests/test_ch22_rnn.py`, `tests/infra/test_repo_structure.py` |
| A10 10 (`as_numpy` d'une liste de tenseurs) | appliqué : éléments détachés un par un, `RuntimeError` rattrapé, tableaux complexes refusés ; tests | `src/wb/nbtools.py`, `tests/infra/test_wb_nbtools.py` |
| A10 11 (`same` et un `float32`) | appliqué : quand l'écart ne vient que de la simple précision, la ligne ❌ le dit ; tests (NumPy et torch, et un vrai écart sans la remarque) | idem |
| A10 12 (type rendu par `temperature_softmax`) | appliqué : un test exige ce que dit la docstring (« ndarray of the shape of logits », exemple sans `dtype` : `float64`) ; un tenseur échoue désormais (vérifié). Le contrat de 22.20 n'est pas modifié (proposition plus bas) | `tests/test_ch22_sequences.py` |
| A10 13, 16 (22.12, 21.12, 21.15, 22.13, 22.20, 25.28) | hors zone (F1, F3) ; le helper `unchanged` suggéré n'est pas écrit (aucun chapitre ne l'importerait avant reconstruction) | — |
| A10 14 (`TypeError` de PyTorch pris pour un `return` oublié) | appliqué : `NONE_TYPE_ERROR` (six messages de None reconnus, ceux de `np.mean(tensor)` non), dans `attempt` et `checked` ; `checked` ajoute une piste « une fonction NumPy a-t-elle reçu un tenseur PyTorch ? » ; tests | `src/wb/checker.py`, `src/wb/nbtools.py`, `tests/infra/test_wb_check.py`, `tests/infra/test_wb_nbtools.py` |
| A10 15, 17 (26.19, 27.11, 27.12) | hors zone (F3, F4) | — |
| A10 19 (`refuses` et `NotImplementedError`) | appliqué : `except _NOT_WRITTEN` en premier ; test | `src/wb/nbtools.py`, `tests/infra/test_wb_nbtools.py` |
| A10 20 (kit sans tests) | appliqué : `tests/infra/test_chapter_kit.py` (`guarded`, `bug_letters_mistakes`, `check_unique_keys`, en-tête et ligne de tableau) | `tests/infra/test_chapter_kit.py` |
| A10 21 (ordre d'affichage de `run_pytest_group`) | appliqué : `_CLEAR_FIRST` (`does_not_modify`, `new_array`, `python_int`…) ; cas ajouté au test | `src/wb/nbtools.py`, `tests/infra/test_wb_nbtools.py` |
| A10 22 (messages de type de `test_ch26_rl.py`) | appliqué : type puis forme ; conseil sans la remarque NumPy pour un tenseur | `tests/test_ch26_rl.py` |
| A1 1, 2, 4, 5 (formats des 04, 05, rappels) | hors zone (F1 à F6) ; les tests d'infrastructure suggérés ne sont pas écrits : ils échoueraient tant que les chapitres ne sont pas corrigés (proposés plus bas) | — |
| A1 3 (« synth » affiché) | appliqué dans le kit : « synth » s'affiche « synthétique » (aussi dans « synth/MNIST ») | `tools/chapters/chapter_kit.py` |
| A1 6 à 9 (titres, carte d'activation, on-policy, transfer learning) | reporté (R1) | — |
| A1 10 (« à pas », « mini-batchs », « sur-échantillonnage ») | glossaire appliqué (DCGAN : « convolutions de stride 2 » ; une seule entrée « suréchantillonnage ») ; prose des ch. 25, 27, 28 hors zone ; motifs du test : voir A1 13 | `annexes/glossaire.md` |
| A1 11 (doublons du glossaire) | appliqué : maximisation d'activation (21, 28), décodage glouton (22, 28), apprentissage auto-supervisé (1, 25), générateur (1, 27) fusionnés | `annexes/glossaire.md` |
| A1 12 (data card et fiche modèle) | proposé : ligne du §22 | — |
| A1 13 (`test_terminology.py`) | appliqué en partie : `ch*_parts.py` et `ch*_code.py` lus (aucune forme écartée). Motifs « mini-batchs », « convolutions à pas », « sur-échantillonnage », « affiner » : non ajoutés, les formes écartées restent dans d'autres zones ou dans des titres publiés (liste plus bas) ; entrées exactes prêtes dans « Propositions » | `tests/infra/test_terminology.py` |
| A1 14 (« Fil rouge » absent de 16 en-têtes) | appliqué dans le kit : le « Fil rouge » du contrat quand `Ex` n'en reçoit pas (ch. 22, 27, 29 à la reconstruction) ; les `02_exercices.md` (26.2, 26.9, 27.1, 29.1) restent hors zone | `tools/chapters/chapter_kit.py` |
| A1 20, A2 5, A3 15, A11 6 (README) | appliqué : partie VI ✅ avec son checkpoint, renvois vers CP5 et CP6, 920 h et 550 h | `README.md` |
| A1 21 (run, worker, workflow) | appliqué au glossaire pour run et worker ; « workflow » de CP5 : déjà retiré par F6 (pas d'entrée au glossaire) ; la glose à la première occurrence (fiche du ch. 24) hors zone (F2) | `annexes/glossaire.md` |
| A1 22 (état caché) | l'entrée existe (ch. 22) ; la fiche qui ne le définit pas est hors zone (F1) | — |
| A1 24 (« affiner ») | les occurrences de CP5 et du MP5 ont disparu ; motif du test non ajouté (A1 13) | — |
| A1 25 (σ deux fois) | appliqué au formulaire (norme spectrale $\lVert W_{hh}\rVert_2$) ; encadré du ch. 22 hors zone (F1) | `annexes/formulaire.md` |
| A2 3 (liens `stable` de `ressources.md`) | appliqué : versions figées (NumPy 2.1, pandas 2.2, Matplotlib 3.10.0, scikit-learn 1.6, PyTorch 2.11, Gymnasium 1.4.0 ; tous en 200) et la phrase sur les versions figées ; liens des fiches hors zone | `annexes/ressources.md` |
| A2 6, A8 12 (ressources V et VI) | appliqué : quinze ressources, toutes citées dans les fiches des ch. 21 à 29 et vérifiées (HTTP 200) | `annexes/ressources.md` |
| A2 15 (renvois et *distribution shift*) | appliqué : doublons fusionnés ; « dérive des données » réservée au *data drift*, « décalage de distribution » terme général, chacun renvoie à l'autre ; la phrase de la fiche du ch. 29 hors zone (F5) | `annexes/glossaire.md` |
| A3 5, A11 5 (🚀 absent des notebooks) | appliqué dans le kit (en-tête) ; tableau : `Ex.table_row` à adopter par les scripts (F1 à F4) ; durées du mode complet hors zone | `tools/chapters/chapter_kit.py` |
| A3 16 (`PROGRESS.md:171`) | proposé (ligne plus bas) | — |
| A3 17 (kernel, erreur TD, termes manquants) | appliqué : *kernel* (ch. 21, renvoi à la DÉROGATION), taille de l'état, schéma enroulé/déroulé, détecteur de motif, MMD ; « erreur TD » gardé, la fiche du ch. 26 hors zone (F3) | `annexes/glossaire.md` |
| A3 22 (cheatsheet sans ch. 26 ni 29) | appliqué : deux sections courtes, extraits exécutés | `annexes/cheatsheets/pytorch.md` |
| A4 4 (biais de Keras et GRU) | appliqué au formulaire ; flashcard du ch. 22 hors zone (F1) | `annexes/formulaire.md` |
| A4 5 (biais d'oubli à 1) | appliqué (`bias_ih` à 1 et `bias_hh` à 0, sous `no_grad`, vérifié) | `annexes/cheatsheets/pytorch.md` |
| A4 6 (lettre D) | appliqué : `n_dir` | idem |
| A4 11 (entrée *kernel*) | appliqué | `annexes/glossaire.md` |
| A4 12 (gradient évanescent deux fois) | appliqué : une entrée « 0B, 22 » | idem |
| A5 1 (`view` et non contigu) | appliqué au glossaire, à la cheatsheet et à l'annexe des erreurs (vérifié : `view(2, 3, 30)` réussit sur un tenseur permuté, `view(2, -1)` échoue) ; flashcard du ch. 23 hors zone (F2) | `annexes/glossaire.md`, `annexes/cheatsheets/pytorch.md`, `annexes/erreurs_frequentes.md` |
| A5 2 (en-tête de la cheatsheet) | appliqué : « 5 et 10 à 29 » (avec les sections ajoutées) | `annexes/cheatsheets/pytorch.md` |
| A6 1 (ligne 278) | appliqué : « mémorisée : dans ses biais selon le livre, dans les poids du décodeur d'après le 🔮 25.14 » | `annexes/erreurs_frequentes.md` |
| A6 2 (`TypeError` de Gymnasium) | appliqué : `gymnasium.error.Error: Seed must be a python integer` (vérifié avec gymnasium 1.4.0) | idem |
| A6 3 (BCE et PSNR) | appliqué : `reduction="sum") / len(x)` ; PSNR d'une image et d'un batch (MSE image par image) | `annexes/formulaire.md` |
| A6 11 (sur-échantillonnage) | glossaire appliqué ; fiche et notebook du ch. 25 hors zone (F3), titre de 25.27 reporté (R1) | `annexes/glossaire.md` |
| A8 8, A1 11 (doublons, orthographes) | appliqué | `annexes/glossaire.md` |
| A8 9 (termes absents) | appliqué : gradient de politique, pénalité de gradient (WGAN-GP), règlement européen sur l'IA, RGPD (ch. 20, première fiche qui les glose) ; l'alignement de la prose sur un seul nom (« règlement européen sur l'IA » ou « AI Act ») est hors zone | `annexes/glossaire.md` |
| A8 10 (transfer learning, model card du ch. 29) | reporté (R1) pour *transfer learning* ; « fiche modèle » du ch. 29 hors zone (F5) | — |
| A8 11 (doublons de l'annexe des erreurs) | appliqué : cinq paires fusionnées, chapitres listés ; « split aléatoire » devenu « découpage aléatoire » (R5) | `annexes/erreurs_frequentes.md` |
| A8 13 (métiers) | appliqué : fuites (8, 12, 15, 29), RL, datasets, MP5 et MP6 au portfolio | `annexes/metiers.md` |
| A8 15 (`PROGRESS.md:1101`) | proposé (ligne plus bas) | — |
| A11 7 (commande de reproduction du MP6) | hors zone (F6) : la commande `nbconvert` ; l'option `--mine` de `run_all_notebooks.py` non retenue (elle changerait la règle du §22 du 2026-09-29 « ne touche jamais `mon_travail/` ») | — |
| A11 9 (téléchargements, hors ligne) | appliqué : section « Travailler hors ligne (à partir du ch. 20) » (tableau des téléchargements, commande unique) ; Colab §4 : poids pré-entraînés et Hub | `00_setup/INSTALL_LOCAL.md`, `00_setup/COLAB.md` |
| A11 10 (coupure de Colab pendant un mode complet) | appliqué : paragraphe « Entraînements longs » (ce qui survit dans le Drive, quoi relancer, la reprise de 27.27 et du MP6.4) ; la variable `RESUME` du MP6 hors zone (F6) | `00_setup/COLAB.md` |

## Propositions (texte exact, à valider)

1. **Contrat de 22.20** (énoncé, `build_ch22.py`, F1) : après « $T \leq 0$ lève `ValueError` ; », ajouter « elle rend un **nouveau** tableau NumPy `float64` de la forme des logits ; ». Le test ajouté n'exige que ce que dit déjà la docstring (`ndarray` de la forme des logits, `float64` pour des logits `float64` ou une liste) ; la phrase aiderait l'apprenant.
2. **27.Q7 dans R1** : le titre publié « Pourquoi juger des mini-batchs plutôt qu'un point isolé » emploie la forme écartée du pluriel ; à ajouter à la liste des titres de R1 A : « Pourquoi juger des mini-batches plutôt qu'un point isolé ».
3. **Entrées de `FORBIDDEN`** (à ajouter quand les zones citées dans « Pour une autre zone » ont corrigé, et après R1 pour les titres) :
   ```python
   "mini-batches": (r"\b[Mm]ini-batchs\b", []),
   "stride": (r"\b[Cc]onvolutions? à pas\b", []),
   "suréchantillonnage": (r"\b[Ss]ur-échantillonn", []),
   "fine-tuning (affiner)": (r"\b[Aa]ffin(?:er|é|ée|és|ées|age|ons|ez|ent)\b",
                             ["affiner encore la grille", "affinent le dosage", "« affinage des données »",
                              "s'affinent ensuite"]),
   ```
   (« affine », « affines » restent permis : la fonction affine.) À ajouter aussi au test qui vérifie les exemples : `assert list(_violations("312 mini-batchs pleins", *FORBIDDEN["mini-batches"])) == ["mini-batchs"]`.
4. **Tests de format** (A1 1, A1 2, A3 2, A3 « soit environ X h ») : à écrire dans `tests/infra/test_syllabus.py` une fois les chapitres corrigés, sinon ils échouent : (a) l'icône du type du contrat à la fin de chaque en-tête `### Ex N.k — Titre` de `04_indices.md` et `05_solutions.md` ; (b) le préfixe « Ch. N : » (N = `recall_from`) suivi d'une minuscule, et « 🔁 ★ » dans les en-têtes des rappels ; (c) le « soit environ X h » du guide de lecture égal à `rapide_reading`.
5. **Référence `vae`** (option, A10 2) : refuser ±inf dans `_pair` comme `gen_metrics` (cohérent avec R10 A). Non appliqué : à vérifier d'abord contre les expériences du ch. 25 qui font diverger `log_var`.

## Pour une autre zone

- **F1 (ch. 21, 22)**
  - `build_ch21.py`, `build_ch22.py` : construire les lignes du tableau de la cellule 0 avec `ex.table_row(part.key if i == 0 else "")` (🚀 de 21.26, 22.25, 22.27).
  - `chapitres/ch22_rnn/flashcards.csv:7` : « … Le Keras du livre n'en compte qu'un (Keras 3 en garde deux pour le GRU). » (A4 4).
  - `chapitres/ch22_rnn/01_fiche.md:112` (encadré des conventions) : « $\sigma$ est la sigmoïde ; $\sigma_{\max}(W)$, dans ∂ 22.6, la plus grande valeur singulière » (A1 25).
  - `chapitres/ch22_rnn/01_fiche.md:86` : définir « état caché » (A1 22).
  - `chapitres/ch21_cnn/01_fiche.md:258` : « convolution à pas » → « convolution de stride 2 » (motif du test).
  - Notebook du ch. 22 : aucun changement pour A10 9 (les groupes `-k` publiés attrapent les trois nouveaux tests).
- **F2 (ch. 23, 24)**
  - `build_ch24.py` : `ex.table_row(...)` pour le tableau (🚀 de 24.18, 24.20, 24.21, 24.23, 24.30, 24.31) ; `build_ch23.py` aussi (aucun 🚀 aujourd'hui, pour l'uniformité).
  - `chapitres/ch23_pytorch_1/flashcards.csv:7` : la formule de A5 1 sur `view`.
  - `chapitres/ch24_pytorch_2/01_fiche.md:113`, `:416` : gloses de *run* et *worker* à leur première occurrence (A1 21 ; les entrées du glossaire existent).
- **F3 (ch. 25, 26)**
  - `build_ch25.py` : `ex.table_row(...)` (25.27, 25.39) ; « sur-échantillonnage » → « suréchantillonnage » et « convolutions à pas » → « convolutions de stride 2 » dans la fiche, 02, 04, 05 et le script, hors du titre de 25.27 (R1).
  - `chapitres/ch26_rl/01_fiche.md:192` : « l'**erreur TD** (erreur de différence temporelle, *temporal-difference error*) » (A3 17).
  - `docs/syllabus/data/ch26.json:341` (champ `today`, hors contrat) : « mini-batchs » → « mini-batches ».
- **F4 (ch. 27, 28)**
  - « mini-batchs » → « mini-batches » dans tout le ch. 27 (`build_ch27.py:867`, `:2283`, `:2333`, `:2442`, `:2539`, `:2558`, `:2563`, 02, 04, 05, 06 ; le titre de 27.Q7 : proposition 2) ; `docs/syllabus/data/ch27.json:103` (concept, hors contrat).
  - « convolution(s) à pas » → « … de stride 2 » : `ch27_gan/01_fiche.md:9`, `:59`, `:316`, `02_exercices.md:87` (sans la glose), `build_ch27.py:2435` ; `docs/syllabus/data/ch27.json:215`.
  - « sur-échantillonnage » → « suréchantillonnage » : `build_ch27.py:1523`, `:1601`, `:2450`, `:2453`.
  - `build_ch27.py:2571` et `build_ch28.py:264` : remplacer le calcul local de `rocket` par `ex.table_row(...)` (le contrat a les mêmes ID).
- **F5 (ch. 29)** : `chapitres/ch29_datasets/01_fiche.md` : relier « décalage de distribution » et « dérive des données » (A2 15) ; « fiche modèle » (A8 10, A1 12).
- **F6 (CP5, MP5, MP6)** : (« workflow » de CP5 : déjà corrigé dans le dépôt) ; `projets/partie_6_generatif/README.md:51` : la commande `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 mon_travail/projets/partie_6_generatif/mp6_generatif.ipynb` (`WB_FAST_MODE=0` devant pour le mode complet) (A11 7).
- **Coordinateur (hors des zones)**
  - `chapitres/ch11_raisonnement/01_fiche.md:281` : « InstructGPT a été affiné par apprentissage par renforcement… » → « InstructGPT a reçu un fine-tuning par apprentissage par renforcement… » (seule occurrence de « affiner » au sens de fine-tuning ; elle bloque l'entrée du test).
  - `chapitres/ch20_deep_learning/01_fiche.md:183`, `flashcards.csv:23`, `docs/syllabus/data/ch20.json:318-319` : « convolutions à pas » → « convolutions de stride 2 ».
  - `docs/syllabus/data/chB5.json:193` (bonus, hors contrat publié) : « sur-échantillonnage » → « suréchantillonnage ».
  - `docs/SYLLABUS.md` et `suivi/tableau_de_bord.md` suivront `syllabus.py build`.

## Lignes proposées

**BIBLE §22**
- | 2026-10-31 | **NaN et infinis des modules des parties V et VI** (A10 2) : une opération de tableau propage NaN et ±inf (`conv`, `rnn`, `temperature_softmax`, comme §22 2026-10-18 pour `conv`) ; une fonction qui rend un résultat plausible refuse les NaN (`ValueError`) : `vae`, `rl`, `gen_metrics` et `sample_from_logits` (un NaN y donnait l'id 0 sans rien dire ; −inf, le masque d'un logit, reste permis) ; ±inf : refusés par `gen_metrics`, acceptés par `vae` et `rl`. La référence de `sample_from_logits` refuse les NaN ; aucun test ne l'exige (la docstring figée ne le dit pas). `rl.py` accepte un `rng` entier (Q9), référence seule. | Audit P5 V-VI, A10 1 et 2. |
- | 2026-10-31 | **`wb.nbtools` et `wb.attempt` face à PyTorch** : un `TypeError` n'est un `return` oublié que si son message parle d'un None (`wb.checker.NONE_TYPE_ERROR`), pas le « got (dtype=NoneType, …) » d'une fonction NumPy qui reçoit un tenseur (piste dédiée dans `checked`) ; `as_numpy` détache les tenseurs d'une liste et refuse un tableau complexe ; `same` dit quand un résultat `float32` n'est faux qu'à la simple précision ; `checked` n'affiche jamais une ligne vide ; `refuses` laisse passer un code pas encore écrit quel que soit `error` ; `run_pytest_group` montre d'abord les tests les plus simples à corriger (`_CLEAR_FIRST`). Tests dans `tests/infra/`. | A10 10, 11, 14, 18, 19, 21. |
- | 2026-10-31 | **🚀 et « Fil rouge » lus dans le contrat** : `Ex.header` ajoute « 🚀 » après « ⏱️ N min » quand le contrat a `"gpu": true`, et le « Fil rouge » du contrat quand le script n'en donne pas (« synth » s'affiche « synthétique ») ; `Ex.table_row` met « 🚀 » après le titre dans le tableau de la cellule 0 (forme des ch. 27-28), ce qui tranche la colonne 🚀 laissée ouverte le 2026-10-18 (A3 15). Les chapitres l'adoptent à leur reconstruction. | A3 5, A11 5, A1 3 et 14. |
- | 2026-10-31 | **Groupes `-k` des notebooks** : un test de `mylearn` porte le nom de la fonction qu'il teste ; `test_repo_structure.py::test_notebook_pytest_groups_cover_every_test_of_their_file` vérifie que les groupes d'un notebook couvrent tout son fichier de tests. | A10 9. |
- | 2026-10-31 | **data card en anglais, fiche modèle en français** : les deux décisions du glossaire (ch. 1 et ch. 8) sont gardées, et écrites ici ; « model card » est la glose de « fiche modèle ». | A1 12 (sauf décision contraire de R1). |

**PROGRESS, « Écarts » et lignes à corriger**
- `suivi/PROGRESS.md:171` : « | 23 | PyTorch en pratique 1 | ✅ | 48-49 | 49 exercices (dont 18 dans le notebook, parties A à D) … » (A3 16 ; vérifié : 49 dans `ch23.json`).
- `suivi/PROGRESS.md:1101` : « | ch. 29 | `wb.datasets.make_trap_dataset` et `trap_dataset_answers` (huit défauts plantés) | 🏆 29.16 ; ✅ fait en session 60 | » (A8 15).
- Écart : « Tests des modules V et VI (session 62, F7) : `test_rnn_cells_return_float64_…` remplacé par trois tests nommés d'après leur fonction ; `test_interpolate_latents_slerp_at_a_small_angle` paramétré (0,05 et 10⁻³ rad) ; `test_temperature_softmax_returns_a_numpy_array_of_the_shape_of_logits` ajouté. Stubs et docstrings inchangés. »
