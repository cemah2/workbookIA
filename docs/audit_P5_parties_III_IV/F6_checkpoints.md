# Correcteur F6 : checkpoints III et IV, MP3 et MP4 (et le coffre de MP2)

**Réponses enregistrées (`wb.record`) dont la valeur change : aucune.** Un seul `wb.record` est touché : le message d'une mauvaise réponse de CP3.8e (`mistakes`), pas sa valeur.

**À signaler au coordinateur :**
- La machine était très chargée (charge de 5 à 11 sur 2 cœurs) pendant mes exécutions.
- Les temps d'entraînement de MP3 (`fit_time`, `score_time` de `results.json`, colonnes « s/fold » du notebook) sont donc 2 à 20 fois plus longs que ceux que cite `solution/README.md` (l. 35 à 42).
- À relancer sur une machine calme avant le commit final : `run_all_notebooks.py projets/partie_3_revenus/solution/mp3_revenus.ipynb --inplace`. Rien d'autre ne change dans `results.json` (comparé clé par clé), à part les deux clés ajoutées.

## Constats

| Constat (relecteur et numéro) | Décision | Fichiers touchés |
|---|---|---|
| A9 1 (étoiles de CP4.3) | reporté R6 | — |
| A9 2 (CP3.3, erreur fréquente « 5,5 → 0,4 ») | appliqué. Vérifié : 5,5 donne 0,5. Le milieu des $x$ vaut 4,5, ce qui coupe comme $s = 4$ (0,444) | `checkpoints/partie_3/03_examen_corrige.md` |
| A9 3 (phrase de CP3.6 c) | reporté R9 | — |
| A9 4 (structure du corrigé de CP4) | appliqué : « Barème récapitulatif et remédiation » (14 lignes, total 20, Ch. = `covers`), « Lire ta note » en quatre tranches (la consigne « sous 12 » est gardée dans la tranche 8-12) ; « Erreurs fréquentes » ajoutées pour CP4.10, CP4.13 (une par extrait, d'après les distracteurs du sujet) et CP4.14 | `checkpoints/partie_4/03_examen_corrige.md` |
| A9 5 (coffre de MP3 et MP2 sans contrôle du commit) | appliqué : avant `vault.open`, `committed(INDICES)` ; s'il n'est pas vrai, le coffre reste fermé (⏳ « test_indices.npy {raison}. Commite-le (MP3.1, consigne d), puis relance cette cellule. ») ; la consigne de MP3.6 c) et de MP2.6 a) le dit. Vérifié sur la vraie cellule (`gate_check.py`, avec des mocks) : fermé sans dépôt git, fermé tant que le fichier n'est pas commité, ouvert une fois commité, fermé quand il a changé depuis son commit, pour MP3 et pour MP2. Solutions (fichier commité) : coffre « déjà ouvert … pour ce même modèle », sorties justes | `tools/chapters/build_mp3.py`, `tools/chapters/build_mp2.py` (+ les 4 notebooks) |
| A9 6 = A1 25 (message du test de MP4 en anglais) | appliqué : messages en français, sur le modèle de MP3 : ⚠️ autre modèle, 🔓 « déjà ouvert … pour ce même modèle », et « 🔓 Test ouvert (n ouverture(s)…) » | `projets/partie_4_mini_framework/{depart,solution}/data.py` (identiques) |
| A9 7 (valeur du README de MP4 absente de `results.json`) | appliqué, sans rouvrir le test (détail sous le tableau) | `solution/results.json`, `solution/README.md`, `tools/chapters/build_mp4.py` |
| A9 8 (CP4.9 a, clipping sur place) | appliqué pour le corrigé seulement : l'erreur fréquente `g *= coef` dit que c'est la version sur place de `clip_grad_norm` (ch. 19) et de `clip_grad_norm_`, mais que le contrat demande de nouveaux tableaux ; sujet et notebook : reporté R9 | `checkpoints/partie_4/03_examen_corrige.md` |
| A9 9 (RMSprop écrit avec $\rho$) | appliqué : $\alpha$ (`alpha` = 0,99) ; la remarque de la batchnorm renvoie à l'`alpha` de RMSprop et aux $\beta$ d'Adam | `checkpoints/partie_4/05_synthese.md` |
| A9 10 (seuil de coût, > ou ≥, $c$ ou $C$) | appliqué pour CP3 ($C_{FP}$, $C_{FN}$ dans le sujet, le corrigé et la synthèse ; ≥ déjà partout) ; formulaire et fiche du ch. 7 : hors zone (voir plus bas) | `checkpoints/partie_3/01_examen_sujet.md`, `03_examen_corrige.md`, `05_synthese.md` |
| A9 11 (la synthèse de CP4 amorce l'examen) | appliqué : mode d'emploi « un petit réseau de ton choix, quelques pas d'un optimiseur » ; la ligne « Chaque epoch commence par un saut » (le bug D de CP4.7) est remplacée par « la loss de validation devient `nan` » (données non finies, `log(0)`). Les lignes « softmax avant `CrossEntropyLoss` » et « `total += loss` » restent : ce sont des pièges que les ch. 18 à 20 enseignent | `checkpoints/partie_4/05_synthese.md` |
| A9 12 (CP3.8 : « baisse encore à 400 arbres ») | appliqué : courbe recalculée (`figures_cp3.boosting_curves`) : 0,4984 à 300 arbres, 0,4934 à 380, minimum 0,4932 à 395, 0,4933 à 400. e) et g) disent : elle ne remonte pas, elle baisse de plus en plus lentement et finit presque plate. Message de la mauvaise réponse de CP3.8e reformulé | `checkpoints/partie_3/03_examen_corrige.md`, `tools/chapters/build_cp3.py` |
| A9 13 (seuils et expérience « sans sex » absents de `results.json`) | appliqué : `thresholds_out_of_fold` (0,5, meilleur F1 0,3607, part de la prévalence 0,4158 → F1 0,7261) et `without_sex_out_of_fold` (PR-AUC 0,8286, 0,8288, 0,8275 ; écarts de sélection 0,2435, 0,2344, 0,2469). Ce sont les chiffres du README et de la model card | `tools/chapters/build_mp3.py`, `projets/partie_3_revenus/solution/results.json` |
| A9 14 (MP4 sans section « Pistes ») | appliqué : un paragraphe TODO dans le kit, cinq pistes dans la solution | `projets/partie_4_mini_framework/{depart,solution}/README.md` |
| A9 15 (filtre `cell < "MP3.6 test"`) | appliqué : liste explicite `BEFORE_VAULT` | `tools/chapters/build_mp3.py` |
| A9 16 (calibrage de MP3.8) | hors zone : une ligne pour PROGRESS (plus bas) | — |
| A1 26 (« Exactitude » dans la grille de MP4) | appliqué : « **Justesse du code** » (cahier des charges et bilan du notebook). Le libellé de la grille dans `cp4.json` (`miniproject.grading`, repris par SYLLABUS l. 3489) : proposé, champ hors mandat | `projets/partie_4_mini_framework/README.md`, `tools/chapters/build_mp4.py` |
| A1 27 (corrigé de CP4 : récapitulatif, libellés) | appliqué (avec A9 4) ; libellés alignés sur CP3 : « **Réponse modèle en 60 secondes** », « **Relances possibles** », « **Réponse modèle** » (CP4.14) | `checkpoints/partie_4/03_examen_corrige.md` |
| A1 33 (titres courts des checkpoints) | reporté R4 ; hors titres, rien à appliquer | — |
| A2 8 (Ridge du ch. 9 : $\lambda = 2\alpha$) | appliqué. Vérifié : le Ridge du ch. 9 minimise une somme ($\lVert y - Xw - b\rVert^2 + \alpha\lVert w\rVert^2$, `linear.py`) ; à loss moyenne, $\lambda = 2\alpha/n$ | `checkpoints/partie_4/05_synthese.md` |
| A2 15 (LIME et contrefactuels annoncés pour B6) | appliqué : concept ajouté à `chB6.json` ; `syllabus.py check` (0 problème) puis `build` | `docs/syllabus/data/chB6.json` (+ fichiers générés) |
| A2 17 (CP4.5 d porte sur le ch. 20) | appliqué : `prereq` et `covers` = ["17", "18", "20"] ; répartition de `study_notes` recalculée (ch. 17 : 2,1 ; ch. 18 : 4,2 ; ch. 20 : 5,7) ; `check` puis `build` | `docs/syllabus/data/cp4.json` (+ fichiers générés) |
| A2 19 (`log_softmax` dans « Ta librairie » de MP4) | appliqué : « 18.14 (et `log_softmax`, 17.15), 18.12, 18.16 » | `projets/partie_4_mini_framework/README.md` |
| A3 9 (titres des sujets CP3.1, CP3.8, CP4.9) | reporté R4 | — |
| A11 3 (MP4 vierge : `NotImplementedError` en MP4.6 c) | appliqué : le bloc PyTorch et la figure attendent la fin de l'entraînement minifw (`main_ok`) ; le bloc PyTorch est en plus sous `attempt` | `tools/chapters/build_mp4.py` |
| A11 4 (MP4 vierge : 14 ❌) | appliqué : `pending()` (⏳, qui garde le test fermé) ; tests en échec « pas encore écrit » → ⏳ ; MP4.4 : ⏳ tant que des tests attendent minifw ou qu'il n'y a que les deux tests d'exemple (thèmes et mutants attendent) ; MP4.7 : sections ou paragraphes encore TODO, figures et `results.json` pas encore créés, test pas encore ouvert, tests qui attendent minifw → ⏳ | `tools/chapters/build_mp4.py` |

**A9 7, le détail.**
- La valeur 0,8994 vient de la sortie du mode complet de la session 42 (`scratchpad/s42/mp4_full.ipynb`, cellule 23 : « minifw : 15 epochs, 3.2 s par epoch, meilleure epoch 13, accuracy de validation 0.8994 »). Le source de cette cellule imprime `max(main_history['val_acc'])`. Sur 10 000 images, une accuracy vaut k/10 000 : la valeur est exacte, et les 15 epochs ont toutes tourné.
- Elle est recopiée à la main dans `results.json` (`"main_config": "les deux"`, `"minifw_best_val_accuracy": 0.8994`). Les autres clés ne changent pas (vérifié par script). Le dictionnaire `results` de `build_mp4.py` écrit désormais ces deux clés.
- README : « (15 epochs au plus) ».

## Pour une autre zone

1. **A9 10, formulaire et fiche du ch. 7.**
   - `annexes/formulaire.md:293` : « probabilités calibrées : positif si $p \ge t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ (à l'égalité, les deux décisions coûtent autant) ».
   - `chapitres/ch07_classification/01_fiche.md:105` : `$$p \ge t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$`, et la même parenthèse dans la phrase qui suit.
2. **`suivi/PROGRESS.md`, § Écarts.**
   - « Audit P5 (F6) : CP4.5 couvre aussi le ch. 20 (`prereq` et `covers` = 17, 18, 20 : la question d) attend les connexions résiduelles) ; la répartition des points de CP4 (`study_notes`) devient ch. 17 : 2,1, ch. 18 : 4,2, ch. 20 : 5,7. »
   - « Audit P5 (F6) : `chB6.json`, concept ajouté : explications locales par substitut (LIME) et explications contrefactuelles (Wachter et coll., 2017), que la fiche du ch. 20 annonce pour B6. »
   - « MP4 : `results.json` reçoit `main_config` et `minifw_best_val_accuracy` (0,8994), recopiés de la sortie du mode complet de la session 42 (MP4.6 c, `max(main_history["val_acc"])`), sans rouvrir le test ; le notebook les écrit désormais. »
3. **`suivi/PROGRESS.md`, § Calibrage (A9 16).** « MP3.8 (60 min) est la plus dense des huit étapes : six tests, trois fonctions, trois documents, la relance complète et le commit. À recaler avec le temps réel de l'apprenant. »
4. **`docs/BIBLE.md` §22.** « 2026-10-18 · Audit P5 (F6) : le coffre de MP2 et de MP3 ne s'ouvre que si `test_indices.npy` est commité et inchangé depuis son commit (sinon un ⏳ dit de le commiter). Dans le kit MP4, ce qui est encore au modèle (README en TODO, figures et `results.json` du mode complet, tests qui attendent minifw, test pas encore ouvert) s'affiche ⏳, pas ❌. »
5. **`tests/infra/test_projects.py`** (proposés).
   - (a) Pour chaque projet qui a un `TestVault`, vérifier que la cellule de `vault.open` appelle `committed(INDICES)` avant d'ouvrir (test de source).
   - (b) A11 3 : un « Tout exécuter » du kit MP4 avec les squelettes de `templates/mylearn_stubs` copiés dans `mon_travail/mylearn` (`nn/`, `optim.py`) : aucune erreur, aucun ❌. Il est lourd (nbclient, ≈ 10 s), donc marqué lent.
6. **`docs/syllabus/data/cp4.json`, `miniproject.grading`** (proposé, A1 26) : « Exactitude » → « Justesse du code », comme le cahier des charges.

## Scripts reconstruits, notebooks réexécutés

Scripts reconstruits :
- `build_cp3.py` ;
- `build_mp2.py`, `build_mp3.py` ;
- `build_mp4.py`, deux fois (la seconde après la reformulation des messages de figures).

`build_cp4.py` et les deux scripts de figures n'ont pas été modifiés. `syllabus.py check` donne 0 problème ; `syllabus.py build` a été lancé.

**Solutions, en mode rapide, `--inplace`** (charge de la machine entre 5 et 11 sur 2 cœurs) :

| Notebook | Durée | Sorties |
|---|---|---|
| `checkpoints/partie_3/03_examen_solutions.ipynb` | 20,3 s | ✅, aucun ❌ ni ⏳ ni traceback |
| `projets/partie_2_california_validation/solution/mp2_california.ipynb` | 1 386 s, cellule 18 à 892 s (hors budget à cause de la charge) | 45 ✅, 0 ❌, 0 ⏳ |
| `projets/partie_3_revenus/solution/mp3_revenus.ipynb` | 1 244 s, puis 428,7 s (cellule 24 à 195 s, hors budget à cause de la charge) | 56 ✅, 0 ❌, 0 ⏳ |
| `projets/partie_4_mini_framework/solution/mp4_mini_framework.ipynb` | 264,6 s, puis 187,1 s (OK) | 69 ✅, 0 ❌, 0 ⏳ |

Détail des sorties :
- MP2 : coffre « déjà ouvert le 2026-10-03 pour ce même modèle » ; `results.json` inchangé.
- MP3 : coffre déjà ouvert pour ce même modèle ; les deux nouvelles clés dans `results.json` ; temps à refaire (voir en tête).
- MP4 : 23 tests, 6 mutants sur 6, test fermé en mode rapide (⏸️), `results.json` et `test_log.json` intacts.

**« Tout exécuter » à vide**, dans une copie (`F6_checkpoints/copie`, `WB_ROOT` de la copie, `start_chapter.py --init`, puis `CP2`, `CP3`, `CP4`) :

| Notebook | Durée | ⏳ | ✅ | ❌ | Erreurs |
|---|---|---|---|---|---|
| MP4, `mylearn` vide (repli sur la référence) | 8 s | 29 | 3 | 0 | 0 |
| MP4, squelettes de `nn/`, `optim.py` et `autograd_mini.py` copiés (le cas d'A11 3) | 12 s | 29 | 2 | 0 | 0 |
| MP3 | 6 s | 16 | — | 0 | 0 |
| MP2 | 6 s | 13 | — | 0 | 0 |
| Examen CP3 | 5 s | 3 | — | 0 | 0 |

Tests :
- `python -m pytest -q tests/infra/test_projects.py` : 12 passés ;
- `tests/infra/test_terminology.py` : 11 passés.

Outils de vérification, dans `F6_checkpoints/` : `gate_check.py` (le coffre de MP2 et de MP3 dans ses quatre états), `scan.py` (le bilan des sorties d'un notebook), `runnb.py` (copié d'A11), et les notebooks exécutés `mp4_vide_ref.ipynb`, `mp4_vide_stubs.ipynb` et `vide_*.ipynb`.
