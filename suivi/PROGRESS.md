# PROGRESS : état de la génération du workbook

*Tenu par Claude à chaque session (BIBLE §18). Dernière mise à jour : 2026-10-01, session 10.*

## Prochaine étape

➡️ **Session 11 : prompt P2, chapitre 3 (Probabilités et mesure de la qualité), 2ᵉ session sur 2** ; pièce jointe : le **Volume 1** du livre (ch. 3, p. 97-152). À faire : le notebook, parties A (3.12–3.14), B (3.15–3.19), C (3.20–3.23) et D (3.24–3.29), dans `tools/chapters/build_ch03.py` (la partie 0 et `NEXT_SESSION` y sont déjà) ; les indices (3 niveaux) et les solutions de 3.12–3.29 dans `04_indices.md` et `05_solutions.md` ; la vérification (apprenant simulé sur le notebook, relecture) ; puis la suppression d'`EN_COURS.md`, qui fera copier le notebook par `start_chapter.py`. La référence et les tests de `mylearn.metrics` sont prêts : ne pas les réécrire. Contraintes : voir « Session 10 » ci-dessous.

Côté apprenant : le **chapitre 2 est complet** (relance `python tools/start_chapter.py 2` pour obtenir son notebook). Le **chapitre 3 est à moitié prêt** : `python tools/start_chapter.py 3` copie ta feuille de réponses et ton squelette `mylearn/metrics.py`, pas encore le notebook. Tu peux lire la fiche, faire les quiz, les rappels, tous les exercices papier (vérifiés en partie 0 du notebook dès qu'il sera copié) et les questions d'entretien, et même commencer `metrics.py` (`python -m pytest tests/test_ch03_metrics.py -q`).

## Statut des chapitres

Légende : 📅 planifié · 🛠️ en cours (étape exacte indiquée) · ✅ généré · 🔍 audité

| ID | Chapitre | Statut | Session | Remarques |
|---|---|---|---|---|
| setup | Mise en place (dépôt, outils, datasets, documentation) | ✅ | 1 | voir « Session 1 » |
| — | Syllabus et parcours | ✅ | 2 | 2 019 exercices, stubs de tous les modules ; voir « Session 2 » |
| 0A | Python, notebooks et outils | ✅ | 3, 4 | 84 exercices (dont 55 dans le notebook, parties A à H), 290 vérifications `wb.check`, 30 flashcards, `mylearn.utils` (référence + 44 tests) ; solutions exécutées en ≈ 20 s |
| 0B | Maths du lycée au ML | ✅ | 5, 6 | 74 exercices (dont 22 dans le notebook, parties A à D, et 174 vérifications papier en partie 0), 281 vérifications `wb.check` au total, 13 figures, 30 flashcards, `mylearn.linalg_basics` (référence + 53 tests) ; solutions exécutées en ≈ 15 s |
| 1 | Introduction | ✅ | 7 | 43 exercices (dont 17 dans le notebook, parties A à D), 70 vérifications `wb.check` (23 pour les exercices papier en partie 0) et 11 vérifications de propriétés, 5 figures, 20 flashcards, aucun module `mylearn` ; solutions exécutées en ≈ 25 s |
| 2 | Hasard et statistiques | ✅ | 8, 9 | 52 exercices (dont 20 dans le notebook, parties A à D), 115 vérifications `wb.check` (52 pour les exercices papier en partie 0) et 34 vérifications de propriétés ou de tests, 7 figures, 28 flashcards, `mylearn.stats` (référence + 113 tests) ; solutions exécutées en ≈ 26 s |
| 3 | Probabilités et mesure de la qualité | 🛠️ | 10 | session 10 faite : fiche (7 figures), 32 exercices hors notebook avec indices et solutions, 30 flashcards, partie 0 du notebook (60 vérifications `wb.check`, 104 erreurs classiques), `mylearn.metrics` (référence + 230 tests) ; **reste** : notebook 3.12–3.29, ses indices et solutions (session 11) |
| 4 | Règle de Bayes | 📅 | | |
| 5 | Courbes et surfaces | 📅 | | |
| 6 | Théorie de l'information | 📅 | | |
| CP-I | Checkpoint partie I | 📅 | | |
| 7 | Classification | 📅 | | |
| 8 | Entraînement et test | 📅 | | |
| 9 | Overfitting et underfitting | 📅 | | |
| 10 | Neurones | 📅 | | |
| 11 | Apprentissage et raisonnement | 📅 | | |
| CP-II | Checkpoint partie II (+ audit P5) | 📅 | | |
| 12 | Préparation des données | 📅 | | |
| 13 | Classifieurs | 📅 | | |
| 14 | Ensembles | 📅 | | |
| 15 | scikit-learn | 📅 | | |
| CP-III | Checkpoint partie III | 📅 | | |
| 16 | Réseaux feed-forward | 📅 | | |
| 17 | Fonctions d'activation | 📅 | | |
| 18 | Rétropropagation | 📅 | | peut prendre 2 sessions |
| 19 | Optimiseurs | 📅 | | |
| 20 | Deep learning et premiers pas en PyTorch | 📅 | | |
| CP-IV | Checkpoint partie IV (+ audit P5) | 📅 | | |
| 21 | CNN | 📅 | | peut prendre 2 sessions |
| 22 | RNN | 📅 | | peut prendre 2 sessions |
| 23 | PyTorch en pratique 1 | 📅 | | |
| 24 | PyTorch en pratique 2 | 📅 | | |
| CP-V | Checkpoint partie V | 📅 | | |
| 25 | Autoencodeurs et VAE | 📅 | | |
| 26 | Apprentissage par renforcement | 📅 | | peut prendre 2 sessions ; Flippers et morpion à coder dans `wb` |
| 27 | GAN | 📅 | | |
| 28 | Applications créatives | 📅 | | |
| 29 | Datasets et préparation du projet final | 📅 | | |
| CP-VI | Checkpoint partie VI (+ audit P5) | 📅 | | |
| B1-B8 | Chapitres bonus | 📅 | | |
| PF | Projet final et finalisation (P10) | 📅 | | |

## Session 1 (2026-09-29) : mise en place ✅

**Fait** :
- Dépôt `cemah2/workbookIA` : arborescence du §7, `.gitignore` (PDF, données lourdes, checkpoints, caches), `.gitattributes`.
- Versions vérifiées et figées (§21) : Python 3.13, alignement sur Colab (torch 2.11.0, scikit-learn 1.6.1, numpy 2.1.3, pandas 2.2.3…). Installation complète testée sous Linux, Python 3.13 et 3.12, torch CPU.
- Package `wb` : `setup`, `check`/`record`/`attempt`, `datasets` (avec fallbacks), `synth`, `plot`, `load_mylearn`, `ensure`, `by_mode`, `timer`.
- Outils : `start_chapter.py`, `build_answers.py`, `run_all_notebooks.py`, `export_flashcards.py`, plus `build_datasets.py` et `nbbuild.py`.
- mylearn : `solutions/mylearn_ref/`, `templates/mylearn_stubs/` (+ `MANIFEST.json`), `tests/conftest.py` avec `--impl=learner|ref|stubs` ; test d'exemple (`_example.mean`) qui prouve les modes.
- Datasets versionnés : Penguins (+ brut), California (CSV identique à scikit-learn), MNIST (`.npz`, 11,5 Mo), Holmes, Verne, taches solaires SILSO ; 10 data cards.
- Documentation : README, guides d'installation locale et Colab, `check_env.py`, `demo.ipynb` + `demo_solutions.ipynb` (exécutés), annexes, suivi, `CLAUDE.md`, bible (§21-22 remplis) et méthode dans `docs/`.
- Fichiers de suivi personnels : les modèles restent dans `suivi/`, ta copie est créée dans `mon_travail/suivi/` par `start_chapter.py --init` puis complétée à chaque chapitre (décision §22 : évite les conflits git).
- Relecture indépendante par un sous-agent : 21 défauts confirmés, tous corrigés et couverts par des tests.
- Vérification finale : 173 tests verts avec les tests qui téléchargent (5 ignorés normalement : ta librairie n'existe pas encore) ; `--impl=ref` : 175 verts ; `--impl=stubs` : le test d'exemple échoue bien (5 ⏳) ; mêmes tests verts avec les toutes dernières versions (numpy 2.5, pandas 3.0, scikit-learn 1.9) ; `check_env.py` OK ; démo exécutée de bout en bout en ~10 s sur CPU.

## Session 2 (2026-09-30) : syllabus détaillé ✅

**Fait** :
- Lecture des deux volumes par six sous-agents (un par partie), consolidation en une fiche JSON par chapitre (`docs/syllabus/data/`, 46 fiches : 0A, 0B, 1-29, B1-B8, 6 checkpoints, projet final) et outil `tools/syllabus.py` (`check`, `build`, `stats`).
- `docs/SYLLABUS.md` (généré) : conventions, vue d'ensemble, totaux par type et par partie, graphes de dépendances (chapitres et modules, Mermaid), calendrier indicatif à 10 h par semaine, plan détaillé de chaque chapitre (objectifs, sections du livre, exercices avec ID, type, ★, ⏱️, fil rouge, fichier, prérequis, parcours ; signatures `mylearn` ; points 🕰️ ; thèmes 💼 ; compétence 🛠️ ; temps ; sessions de génération), matrice de couverture (aucune section à zéro).
- `docs/PARCOURS.md` : parcours complet (≈ 889 h), rapide (≈ 518 h), maths (≈ 526 h), code (≈ 706 h), listes d'ID, corrigés à lire pour les prérequis hors parcours.
- Stubs de **tous** les modules `mylearn` (38 modules + `nn/__init__.py`, docstrings NumPy en anglais, exemples vérifiés contre les oracles) et `MANIFEST.json` complet.
- Relecture pédagogique indépendante (sous-agent) : notions utilisées avant d'être enseignées, trous de couverture, doublons, ruptures de difficulté, volume. Corrections appliquées (BIBLE §22, 2026-09-30), puis vérifiées par une seconde relecture indépendante (8 défauts relevés, corrigés).
- Non retenu (à arbitrer par toi, voir le rapport) : un seul indice pour les 🧠/🔁/💼, 🧠 ramenés à 8-10, fonctions d'aide `mylearn` en double (contrats différents), parcours rapide ramené vers 350 h.
- Repli sur la référence pour les modules des chapitres sautés (`wb.load_mylearn(..., fallback="ref", chapter=…)`, `pytest`, `start_chapter.py`), avec tests.
- Tableau de bord (modèle) : une section par chapitre avec toutes ses cases.
- Vérification : `tools/syllabus.py check` sans problème ; tests verts (mode apprenant, `--impl=ref`) ; `--impl=stubs` échoue comme prévu.

**Chiffres** : 2 019 exercices ; ≈ 889 h d'étude (659 h d'exercices dont 41 h de projet final, 134 h de lecture, 32 h de flashcards, 63 h de synthèses et de mini-projets) ; 67 sessions de génération de chapitres (≈ 80 au total avec checkpoints, audits et finalisation).

## Session 3 (2026-09-30) : chapitre 0A, 1ʳᵉ partie ✅

**Fait** :
- `01_fiche.md` : **tout** le cours (sections 100.1 à 100.11, ≈ 15 000 mots), exemples exécutés dans un vrai interpréteur (sorties recopiées automatiquement), encadrés 🧮 (division euclidienne, flottants, vecteurs et matrices, formule des mini-lots), 6 encadrés 🕰️ vérifiés par recherche web avec sources (annotations `X | None`, `pathlib`, `torch.load(weights_only=True)`, `default_rng`, Copy-on-Write de pandas, `main`/`git switch`), pièges, liens, guide de lecture (parcours rapide), ressources.
- `02_exercices.md` : 12 quiz, 8 exercices papier (vérifiés par `wb.check` dans la partie 0 du notebook), 🗣️ 0A.9, 🛠️ git 0A.10–0A.12, 5 questions 💼 ; `06_mes_reponses.md`.
- Notebook (`tools/chapters/build_ch00a.py`, source unique des deux notebooks) : parties 0 et A–E, 0A.13–0A.36, 146 vérifications ; `05_solutions.ipynb` exécuté en ≈ 5 s ; « Run all » de `03_notebook.ipynb` sans erreur.
- `04_indices.md` (3 niveaux pour les 53 exercices et quiz de cette session), `05_solutions.md` (réponses, pourquoi, erreurs fréquentes, variantes, réponses modèles 💼 en 60 s), `flashcards.csv` (30 cartes).
- mylearn : référence `utils.py` (count_values, argmax, one_hot, iterate_minibatches) et 44 tests à oracle, en avance sur la session 4.
- Infrastructure : `wb.record` n'affiche plus l'entrée dans le notebook ; le repli sur la référence ne sert jamais les fichiers de base (`_example.py`) ; marqueur `EN_COURS.md` et option `--force` de `start_chapter.py` ; messages d'aide avec le vrai numéro de chapitre.
- Annexes : glossaire (45 termes), formulaire (section 0A), cheatsheets NumPy, pandas et git, erreurs fréquentes Python/NumPy/pandas.
- **Vérifications indépendantes** (sous-agents) : 116 sous-questions papier et quiz re-résolues à l'aveugle, 116 identiques ; 2 erreurs dans les solutions (variante de 0A.3, couleurs de `git status`) et 6 indices corrigés (dont 3 qui donnaient la réponse trop tôt), 2 imprécisions de la fiche corrigées. Test « apprenant » des 91 vérifications de 0A.13–0A.35 avec des méthodes différentes des corrigés : 90 acceptées ; 9 défauts d'énoncé corrigés (seuil « heavy » unifié à ≥ 4500 g, méthode imposée en 0A.31e, messages d'erreur ciblés pour `dropna(subset=…)`, cellule 0A.17 relançable, 0A.36 reformulé…).
- Vérification finale : `syllabus.py check` 0 problème ; tests verts (apprenant 191, `--impl=ref` 240) ; `--impl=stubs` : les 44 tests de `utils` échouent bien ; `build_answers.py --check` et `export_flashcards.py --check` OK.

## Session 4 (2026-09-30) : chapitre 0A, 2ᵉ partie ✅

**Fait** :
- Notebook, parties F (Python intermédiaire et avancé : traceback, fichiers, JSON et pickle, `*args`/`**kwargs`, `lambda`, fermetures, récursivité, regex, `itertools`/`heapq`, classes, méthodes spéciales, héritage, générateurs), G (NumPy, pandas et matplotlib avancés : axes, broadcasting, `reshape`, MNIST, vectorisation mesurée, bugs silencieux, `groupby`, filtres, figures à panneaux, lecture de graphique) et H (docstring et doctest, tests pytest, `utils.count_values`, `argmax`, `one_hot`, `iterate_minibatches`, défi 🏆 en dix questions) : 0A.37–0A.67, 144 nouvelles vérifications.
- `04_indices.md` et `05_solutions.md` complétés (84 exercices, 3 indices chacun) ; 5 flashcards remplacées pour couvrir fermetures, récursivité, générateurs, `fit` qui renvoie `self` et tests (toujours 30 cartes).
- Infrastructure : `wb.run_pytest` (lance pytest pour de vrai sur des tests écrits dans un notebook, avec tests dans `tests/infra/test_wb_testing.py`) ; les cellules des exercices mylearn rechargent la librairie de l'apprenant (plus besoin de redémarrer le noyau) et affichent les tests en échec.
- Fiche : `idxmax`/`idxmin` ajoutés (§100.9.4) ; `EN_COURS.md` supprimé.
- **Vérifications indépendantes** : un « apprenant » simulé a résolu les 31 exercices avec d'autres méthodes (153 vérifications sur 153 acceptées ; deux implémentations différentes de `utils` passent les 44 tests) ; un relecteur a recalculé toutes les valeurs de `05_solutions.md` (toutes justes). Corrections : 6 défauts de vérification (chaîne acceptée en 0A.46, erreurs de collecte en 0A.62, docstring factice acceptée en 0A.61, squelettes incomplets en 0A.48 et 0A.50, figures vérifiées séparément en 0A.59), variables renommées pour que chaque cellule reste relançable, point ambigu remplacé en 0A.60c, lettres des énoncés alignées sur les vérifications (0A.43, 0A.53), 5 indices trop explicites reformulés, 8 phrases des solutions corrigées, 0A.64 passé à ★★★ (35 min).
- Vérification finale : `syllabus.py check` 0 problème ; tests verts (apprenant 201, `--impl=ref` 250) ; `--impl=stubs` : les 44 tests de `utils` échouent bien ; simulation « apprenant = référence » : 300 ✅, 0 ❌ ; « Run all » du notebook vide : 225 ⏳, aucune erreur.

## Session 5 (2026-09-30) : chapitre 0B, 1ʳᵉ partie ✅

**Fait** :
- `01_fiche.md` : tout le cours (101.1 à 101.7, 35 sous-sections), un exemple chiffré par notion, 71 blocs `>>>` exécutés, encadrés 🧮 et 🕰️ (log = ln en ML ; vecteurs colonnes des manuels contre lignes du code), 13 figures calculées par `tools/chapters/figures_ch00b.py`.
- `02_exercices.md` : 12 quiz, 3 rappels, 29 exercices papier (23 ✏️, 6 ∂), 🧮 0B.30, 🗣️ 0B.31, 🛠️ 0B.32 (LaTeX), 5 questions d'entretien ; `06_mes_reponses.md`.
- `04_indices.md` (3 indices pour chacun des 52 exercices) et `05_solutions.md` (réponses, démarches, erreurs fréquentes, variantes ; démonstrations ∂ complètes ; réponses d'entretien en 60 secondes avec relances) ; 30 flashcards.
- Notebook : partie 0 (174 vérifications des ✏️, avec 80 erreurs classiques reconnues) via le nouveau kit `tools/chapters/chapter_kit.py` ; `EN_COURS.md` créé.
- En avance : `solutions/mylearn_ref/linalg_basics.py` (Python pur, 13 fonctions) et `tests/test_ch00b_linalg_basics.py` (51 tests, oracles NumPy, `math.dist`, scikit-learn, SciPy ; 10 implémentations fautives testées, toutes attrapées).
- Annexes : formulaire §0B, 40 termes au glossaire, cheatsheet NumPy (algèbre linéaire, fonctions mathématiques), erreurs fréquentes (maths et algèbre linéaire).
- `wb.check` : messages d'ordre de grandeur justes (« facteur 10 » seulement quand c'est vrai), nombre non entier pour une réponse entière traité comme une mauvaise valeur (tests ajoutés).
- **Vérifications indépendantes** : ✏️, ∂ et 🧮 re-résolus à l'aveugle (174/174 identiques, variantes de saisie acceptées) ; fiche relue (aucune erreur de maths, blocs `>>>` rejoués) ; indices, solutions, notebook et annexes relus. Corrections : 11 énoncés, 9 erreurs classiques ajoutées, 10 messages trop explicites reformulés, 5 exemples de format qui donnaient des réponses, 4 renvois de chapitre, 3 indices, 12 points de la fiche, 6 figures (BIBLE §22).
- Vérification finale : `syllabus.py check` 0 problème ; tests verts (apprenant 206, `--impl=ref` 306) ; `--impl=stubs` : tous les tests de chapitre échouent ; saisie simulée des ✏️ : 243 ✅ sur 243 ; « Run all » du notebook vide : 174 ⏳, aucune erreur ; solutions exécutées en ≈ 5 s.

## Session 6 (2026-09-30) : chapitre 0B, 2ᵉ partie ✅

**Fait** :
- Notebook, parties A (nombres et fonctions : calculs, 🔮 0,99¹⁰⁰⁰, Σ/Π et moyenne mobile, galerie des fonctions, 🐛 `-inf`/`nan`/dépassements), B (`linalg_basics` en quatre étapes, comparaison et vitesse contre NumPy, 🔬 distance contre cosinus sur des sacs de mots, 🔮 règles du produit matriciel, 🐛 `*`/`@`/formes, `inv` et `solve`), C (pentes centrées et *gradient checking*, variations, carte de lignes de niveau et gradient, 🔮 directions de descente, somme sur les chemins) et D (🔬 dés simulés et vitesse $1/\sqrt{n}$, espérance et variance simulées, 🏆 ordre des produits) : 0B.33–0B.54, 107 nouvelles vérifications.
- `04_indices.md` et `05_solutions.md` complétés (74 exercices, 3 indices chacun) ; 3 flashcards remplacées (formes `*`/`@`, *gradient checking*, précision d'une simulation) ; annexes (glossaire, formulaire, cheatsheet NumPy, erreurs fréquentes) ; `EN_COURS.md` supprimé.
- Infrastructure : `chapter_kit.py` (cellules de suite après une expérience) ; `wb.check` (tableaux d'entiers calculés en flottants acceptés, messages des réponses à choix) ; raisons des tests en échec affichées dans les notebooks 0A et 0B ; 2 tests de `linalg_basics` ajoutés, 2 exemples qui donnaient des réponses retirés.
- 0B.54 renommé (« des dizaines de fois plus vite ») : voir § Écarts.
- **Vérifications indépendantes** : un « apprenant » simulé a écrit sa propre `linalg_basics` (elle passe les 53 tests, tout comme le code recopié des indices) et résolu le notebook avec d'autres méthodes (121/121 vérifications acceptées, environ 150 formats de saisie) ; un relecteur a recalculé toutes les valeurs des solutions du notebook (toutes justes). Corrections : voir BIBLE §22.
- Vérification finale : `syllabus.py check` 0 problème ; tests verts (apprenant 207, `--impl=ref` 309) ; `--impl=stubs` : les 97 tests de chapitre échouent bien ; `build_answers.py --check` (571 réponses) et `export_flashcards.py --check` (60 cartes) OK ; simulation « apprenant = référence » : 121 ✅ (notebook) et 243 ✅ (saisies papier) ; « Run all » du notebook vide : aucune erreur ; solutions exécutées en ≈ 15 s.

## Session 7 (2026-09-30) : chapitre 1 ✅

**Fait** :
- `01_fiche.md` : le cours du ch. 1 (§1.1 à §1.8) raconté avec les données du workbook (les quatre fils rouges) et un mini-exemple chiffré par notion ; les anecdotes du livre sont des renvois ; guide de lecture avec les sections ⏩ du parcours rapide ; 6 encadrés 🕰️ vérifiés par recherche web (auto-supervisé et RLHF, MNIST résolu, NPU et TPU, Keras 3 et PyTorch, panorama 2026 : Transformers, LLM, diffusion, foundation models ; reconnaissance faciale, RGPD et AI Act) ; 5 figures calculées par `tools/chapters/figures_ch01.py`.
- `02_exercices.md` : 11 quiz, 3 rappels, 4 exercices papier (vérifiés en partie 0 du notebook), 🧮 1.5 (coût des étiquettes de MNIST), 🗣️ 1.6, ⚖️ 1.7, 📄 1.8 (Galton), 4 questions d'entretien ; `06_mes_reponses.md`.
- Notebook (`tools/chapters/build_ch01.py`) : partie 0 ; A, les fils rouges (Penguins, MNIST, Holmes et Verne, taches solaires, data cards) ; B, apprendre et évaluer sur Penguins (mémoriser n'est pas apprendre, système expert, boucle d'entraînement d'une droite, learning rate, arbre de décision, espèce inconnue, score trop beau), avec un découpage entraînement/test imposé (graine 42) ; C, clustering, agent cuisinier (renforcement), réseau de neurones sur MNIST, faux Holmes et faux Verne ; D, 🏆 battre l'expert.
- `04_indices.md` (3 niveaux pour les 43 exercices), `05_solutions.md` (réponses, pourquoi, erreurs fréquentes, réponses d'entretien en 60 s), 20 flashcards.
- Infrastructure : `wb.check` (facteur 100 exact, réponses entières arrondies, dates refusées ; 3 tests), kit (`todo_md`, `solution_md` pour les réponses rédigées), vérifications de propriétés ✅/❌ pour les exercices sans valeur unique.
- Annexes : glossaire (45 termes du ch. 1, colonne « Ch. » remplie, lignes cassées réparées), formulaire, erreurs fréquentes (scikit-learn, raisonnement), cheatsheet scikit-learn ; data cards (usage au ch. 1) ; modèle d'auto-évaluation complété pour 0A, 0B et 1.
- **Vérifications indépendantes** : papier et 🧮 re-résolus à l'aveugle (aucun désaccord) ; notebook résolu deux fois par un « apprenant » simulé avec d'autres méthodes (81 ✅, 0 ❌ au second passage) ; fiche relue trois fois contre le texte du livre : la 1ʳᵉ version suivait le livre de trop près et a été **réécrite entièrement**, puis les reformulations encore proches, 5 réponses données d'avance et 4 inexactitudes ont été corrigées (BIBLE §22).
- Vérification finale : `syllabus.py check` 0 problème ; tests verts (apprenant 210, `--impl=ref` 312) ; `--impl=stubs` : les 97 tests de chapitre échouent bien ; simulation « apprenant = référence » : 81 ✅, 0 ❌ ; « Run all » du notebook vide : 71 ⏳, aucune erreur ; `build_answers.py --check` (641 réponses) et `export_flashcards.py --check` (80 cartes) OK.

## Session 8 (2026-09-30) : chapitre 2, 1ʳᵉ partie ✅

**Fait** :
- `01_fiche.md` : tout le cours (§2.1 à §2.9) avec les manchots, MNIST et des exemples synthétiques, un mini-exemple chiffré par notion, 15 blocs `>>>` exécutés, 5 encadrés 🧮 (densité, variance corrigée et ddof, percentiles, intervalle de confiance, matrice de covariance), 5 encadrés 🕰️ vérifiés par recherche web (générateurs NumPy, taille des rééchantillons bootstrap et `scipy.stats.bootstrap`, pmf/pdf et loi catégorielle, conventions ddof, Datasaurus et EDA), 5 encadrés ⚠️ qui corrigent le livre (mode, variable aléatoire, écart-type et 68 %, covariance, corrélation nulle) ; 7 figures calculées par `tools/chapters/figures_ch02.py`.
- `02_exercices.md` : 12 quiz, 3 rappels, 7 exercices ✏️ et une démonstration ∂ (2.8), 🗣️ 2.9, ⚖️ 2.10, 🧮 2.11, 📄 2.12 (Anscombe 1973), 5 questions d'entretien ; `06_mes_reponses.md`.
- `04_indices.md` et `05_solutions.md` pour ces 32 exercices (3 indices chacun ; réponses, démarches, erreurs fréquentes, réponses d'entretien en 60 s) ; 28 flashcards.
- Notebook (`tools/chapters/build_ch02.py`) : partie 0, 52 vérifications des ✏️ avec 45 erreurs classiques reconnues ; `EN_COURS.md` créé.
- En avance : `solutions/mylearn_ref/stats.py` (16 fonctions, NumPy) et `tests/test_ch02_stats.py` (113 tests à oracle NumPy, SciPy, pandas et `statistics`).
- Annexes : 31 termes au glossaire, formulaire §2, cheatsheets NumPy (tirages, statistiques) et pandas (statistiques), 12 erreurs de raisonnement statistique ; data cards Penguins et MNIST ; section 2 de l'auto-évaluation. Au passage : une flashcard du ch. 1 qui donnait la réponse de 1.9 f corrigée ; `wb.check` accepte le signe moins typographique.
- **Vérifications indépendantes** : ✏️, ∂ et 🧮 re-résolus à l'aveugle (52/52 identiques) ; un apprenant simulé a écrit son propre `stats.py` (113/113 dès le premier essai, 29 variantes essayées) ; fiche relue deux fois contre le texte du livre (une quinzaine de passages réécrits, critiques du livre rendues plus justes, 6 faits corrigés, 5 fuites supprimées, 2.6 refait avec cinq exemples). Détail : BIBLE §22.
- Vérification finale : voir la ligne du tableau de bord ci-dessus et BIBLE §22 ; `syllabus.py check` 0 problème ; tests verts (apprenant 210, `--impl=ref` 425) ; `--impl=stubs` : les 210 tests de chapitre échouent bien ; « Run all » du notebook vide : 52 ⏳, aucune erreur ; `build_answers.py --check` et `export_flashcards.py --check` OK.

## Session 9 (2026-09-30) : chapitre 2, 2ᵉ partie ✅

**Fait** :
- Notebook, parties A (`mylearn.stats` : `mean`, `median`, `mode`, `variance`, `std`, `percentile`, `zscore`, `histogram` ; 🔮 graines ; 🎨 galerie des lois ; 🔬 règle 68-95-99,7 sur des tirages et sur les manchots), B (roue de la fortune `sample_categorical`, 🔮 pelage des animaux, tirages avec et sans remise, mini-batches d'une epoch), C (bootstrap et intervalle de confiance, 🔬 rééchantillons de 20 contre $n$ et couverture, 📦 `scipy.stats.bootstrap`) et D (distances entre images de MNIST, covariance et corrélation, 📈 deviner une corrélation, matrices des manchots et paradoxe de Simpson, 🐛 ddof, 🎨 Anscombe, 🛠️ docstring et tests confrontés à trois versions boguées de `zscore`, 🏆 fabriquer son quartet) : 2.13–2.32, 63 réponses `wb.check` et 34 vérifications de propriétés ou de tests.
- `04_indices.md` (3 niveaux pour les 20 exercices) et `05_solutions.md` (réponses, démarches, erreurs fréquentes, variantes) complétés ; `EN_COURS.md` supprimé.
- Infrastructure : `wb.check` (le message « exactement 100 fois » n'est affirmé que si la réponse garde deux chiffres significatifs ; il est aussi donné pour un tableau de pourcentages au lieu de proportions ; 2 tests) ; `run_all_notebooks.py` ignore un `MPLBACKEND` hérité du terminal (sans cela, le notebook de solutions du ch. 2 avait été enregistré sans ses figures à la session 8 ; réexécuté, test ajouté) ; cellules d'expérience des 🔮 « gardées » (rien ne s'affiche avant les prédictions).
- Annexes : 7 termes au glossaire (erreur type, couverture, bruit de Monte-Carlo, BCa, contraste des distances, paradoxe de Simpson, point influent), 10 erreurs fréquentes, 6 lignes de cheatsheet NumPy ; `pyproject.toml` : `scipy>=1.15` (argument `rng=` de `scipy.stats.bootstrap`).
- **Vérifications indépendantes** : un apprenant simulé (son propre `stats.py`, d'autres méthodes que le corrigé, ≈ 250 saisies supplémentaires en direct) a trouvé 3 bugs de cellule (pytest non importé pour 2.31, 2.24 sans `.confidence_interval`, 2.21 écrit avec `yield`), 6 fuites et 19 autres remarques ; au dernier passage, 96 ✅ et aucune bonne réponse refusée. Un relecteur a recalculé toutes les valeurs (NumPy, pandas, SciPy) et relevé 43 points (9 erreurs, 5 fuites, 7 règles de la bible, 19 manques de clarté, 3 de style), tous corrigés. Détail : BIBLE §22.
- Vérification finale : `syllabus.py check` 0 problème ; tests verts (apprenant 212, `--impl=ref` 427) ; `--impl=stubs` : les 210 tests de chapitre échouent bien ; simulation « apprenant = référence » : 97 ✅, 0 ❌ ; « Run all » du notebook vide : 110 ⏳ (dont 52 en partie 0), aucune erreur ni réponse affichée ; solutions exécutées en ≈ 26 s ; `build_answers.py --check` (756 réponses) et `export_flashcards.py --check` (108 cartes) OK.

**Contraintes issues de la session 8, toutes respectées** :
- ne pas faire vérifier des valeurs que la fiche imprime déjà : effectifs par île (168, 124, 52) et donc le mode de `island`, moyennes de nageoire des Gentoo et des Adélie (≈ 217 et 190 mm), sorties des graines 42 et 0 de la fiche, chiffres d'Anscombe (0,816, 3,16, 11, 4,125), valeurs de la figure bootstrap ;
- 2.23 fait **mesurer** le rapport des largeurs (rééchantillons de 20 contre $n$) : la fiche ne le chiffre pas ;
- 2.27 ne doit pas réutiliser les nuages de la fiche (graine 3 de `figures_ch02.py`) ;
- 2.18 : sur les vraies données, 151 Adélie mesurés, 70,9 %, 95,4 % et 99,3 % à moins de 1, 2 et 3 écarts-types (ddof = 0), et 3 nageoires au-dessus de 203 mm (le ✏️ 2.3 en prévoit 4) ;
- idée pour 2.28 (non citée dans la fiche) : corrélation bec (longueur, épaisseur) de −0,235 sur tous les manchots, mais de +0,64 chez les Gentoo : un paradoxe de Simpson, à faire prédire avant de calculer ;
- 2.13 : les masses sont des multiples de 25 g (le mode existe et dépend de l'arrondi, comme le dit la fiche).

## Session 10 (2026-09-30 → 10-01) : chapitre 3, 1ʳᵉ partie ✅

**Fait** :
- `01_fiche.md` : tout le cours (§3.1 à §3.8) avec le filtre anti-spam comme fil conducteur, les manchots et des exemples synthétiques ; un mini-exemple chiffré par notion, choisi pour ne répondre à aucun quiz ni exercice ; sections « au-delà du livre » (plusieurs classes et moyennes macro, micro, pondérée ; courbe ROC et AUC ; courbe precision-recall et average precision ; calibration et score de Brier) ; 4 encadrés 🧮 (`pd.crosstab`, seuil de décision, moyenne harmonique, méthode des trapèzes) ; 5 encadrés 🕰️ vérifiés par recherche web (conventions de `confusion_matrix`, balanced accuracy et MCC, ROC et `roc_curve`, PR et AP nuancés par McDermott et coll. 2024, calibration et `temperature` depuis scikit-learn 1.8, nuancé par Minderer et coll. 2021) ; critiques du livre (deux phrases fausses sur la spécificité, sans nommer la bonne mesure pour ne pas donner 3.7 m ; légendes des fig. 3.23 et 3.27 ; phrase informelle sur l'indépendance) ; 7 figures (`tools/chapters/figures_ch03.py`), 23 exemples `>>>` exécutés.
- `02_exercices.md` : 12 quiz, 3 rappels, ✏️ 3.1–3.3 et 3.5–3.7, ∂ 3.4 et 3.8, 🗣️ 3.9, ⚖️ 3.10, 📄 3.11 (Fawcett 2006), 5 questions d'entretien ; `06_mes_reponses.md` ; `04_indices.md` et `05_solutions.md` pour ces 32 exercices ; 30 flashcards.
- Notebook (`tools/chapters/build_ch03.py`) : partie 0, 60 vérifications et 104 erreurs classiques reconnues ; `EN_COURS.md` créé.
- `mylearn.metrics` : référence (14 fonctions, `solutions/mylearn_ref/metrics.py`) et 230 tests à oracle scikit-learn (`tests/test_ch03_metrics.py`), dont les groupes `-k` des exercices 🔨 sont prêts (ci-dessous).
- Infrastructure : `wb.check` (« presque » plus prudent, fractions « 41/76 », message pour « 30 % », nombres écrits en texte dans une liste, indice d'arrondi pour un tableau ; 4 tests) ; lignes `FAILED` de `pytest -rf` qui donnent toujours leur raison (⏳ ou message de `np.testing`, ce qui profite aussi aux notebooks de 0B et du ch. 2 ; 2 tests) ; consigne ajoutée à `CLAUDE.md` (première ligne des messages d'échec, helper pytest des notebooks).
- Annexes : 37 entrées au glossaire (dont precision, recall et F1), formulaire §3, cheatsheets scikit-learn (mesures, courbes, calibration) et pandas (`crosstab`), erreurs fréquentes (lecture de la matrice, prévalence, F1, calibration), data card Penguins (usage au ch. 3), section 3 de l'auto-évaluation.
- **Vérifications indépendantes** : ✏️, ∂ et 🧮 re-résolus à l'aveugle (60/60 ; 24 remarques, dont 5 erreurs des solutions) ; fiche relue contre le texte du livre (45 remarques, dont 8 paraphrases réécrites et 6 fuites) ; apprenant simulé qui écrit son propre `metrics.py` (206/206 au premier essai ; tests renforcés : 13 alternatives justes passent, 65 erreurs sur 65 attrapées) ; seconde relecture des corrections (16 points, corrigés). Détail : BIBLE §22.
- Vérification finale : `syllabus.py check` 0 problème ; tests verts (apprenant 217, `--impl=ref` 662) ; `--impl=stubs` : les 440 tests de chapitre échouent bien ; « Run all » du notebook vide : 60 ⏳, aucune erreur ; solutions exécutées en ≈ 2 s ; `build_answers.py --check` (816 réponses) et `export_flashcards.py --check` (138 cartes) OK.

**Contraintes pour la session 11** :
- **Valeurs déjà imprimées par la fiche, à ne pas faire vérifier** : les fléchettes de la figure (graine 30, 500 fléchettes : 155 dans B, 64 dans A, 46 dans les deux) ; la table sexe × espèce des manchots et ses deux `crosstab` normalisés ; les 16 mesures du filtre anti-spam (TP 8, FN 4, FP 2, TN 36) ; le rapport chat/chien/lapin (F1 macro 0,704, pondéré 0,756, accuracy 0,767) ; l'arbre du test à 0,90/0,95 et 2 % (precision 0,27) ; l'exemple à six scores (points ROC, AUC 7/9, AP 29/36, precision qui baisse de 3/4 à 2/3 entre les seuils 0,4 et 0,7) ; le Brier de la météo (0,1375) ; les figures `seuil.png` (10 % de positifs, N(2,2 ; 1) contre N(0 ; 1), seuil 1,5), `prevalence.png`, `roc_pr.png` (N(1,5 ; 1) contre N(0 ; 1), 50 % et 2 % de positifs : AUC 0,86 et 0,85, AP 0,86 et 0,18) et `calibration.png` (graine 9, 4 000 cas tirés d'une loi bêta(2, 2) : même AUC ≈ 0,77 pour les deux modèles, Brier 0,196 et 0,216) ; les faits de documentation des 🕰️ (`thresholds[0] = np.inf`, `drop_intermediate=True`, dernier point PR (1, 0), AP sans interpolation, isotonique à partir d'environ 1 000 exemples, `method="temperature"` depuis 1.8).
- 3.21 simule un autre test que celui de la fiche (0,90/0,95) : celui du livre (0,99/0,98), comme prévu. 3.27 prend d'autres lois de scores ou d'autres prévalences que `roc_pr.png`. 🛠️ 3.23 interroge d'autres paramètres que ceux des 🕰️ (`zero_division`, `labels`, `pos_label`, `normalize`, `sample_weight`, `average=None`). 🔮 3.18 : la fiche dit déjà recall = 1 et precision = prévalence (tout positif), precision = 1 et recall minuscule (un seul positif sûr) : faire prédire d'autres valeurs (accuracy, F1, balanced accuracy des deux classifieurs extrêmes sur ses données).
- Énoncés des 🔨 : en 3.16, « les étiquettes présentes » sont celles de `y_true` **et** de `y_pred` (c'est testé) ; les tests acceptent `np.float64` là où la docstring dit « float Python » (comme en 0B et au ch. 2), donc ne pas lancer de doctest sur le code de l'apprenant (NumPy 2 affiche `np.float64(0.75)`) ; en 3.19, la règle `zero_division` vaut pour tout rapport 0/0, même là où `balanced_accuracy_score` et `matthews_corrcoef` ont leur propre convention ; en 3.24 et 3.26, convertir les entrées avec `np.asarray` (une Series à index mélangé est testée) ; en 3.28, trouver l'intervalle en comparant p aux bords (`np.searchsorted(edges[1:-1], p)` ou `np.digitize(p, edges[1:-1], right=True)`), pas par un calcul sur `p * n_bins`, et une valeur posée sur un bord intérieur va dans l'intervalle du bas.
- Groupes `-k` (sur le fichier `tests/test_ch03_metrics.py`, jamais `tests/`) : 3.15 `test_confusion_matrix_` ; 3.16 `(test_accuracy_ or test_precision_ or test_recall_ or test_fbeta_ or test_f1_) and not multiclass and not curve` ; 3.19 `test_classification_rates_` ; 3.24 `test_roc_curve_ or test_auc_ or test_roc_auc_` ; 3.25 `(test_precision_ or test_recall_ or test_fbeta_ or test_f1_) and multiclass and not curve` ; 3.26 `test_precision_recall_curve_ or test_average_precision_` ; 3.28 `test_calibration_curve_ or test_brier_score_`. Le helper imprime les lignes `FAILED` avec `COLUMNS=1000`, le nom du test puis sa raison sur deux lignes (elles donnent maintenant « ⏳ pas encore implémenté » ou « explication : expected …, got … »). Un nouveau test garde ces préfixes (et `binary`, `multiclass`, `curve`).
- 3.25 réutilise les règles expertes du ch. 1 (1.15) ; 3.20 et 3.24 : score = longueur de nageoire pour « Gentoo contre le reste » ; 🏆 3.29 : seuil choisi sur la validation (recall ≥ 0,99 sur le test), seuil de precision fixé à la génération (notes de `ch03.json`).

## Infrastructure à coder pendant les sessions de chapitres

| Pour | Élément | Remarque |
|---|---|---|
| ch. 26 | `wb.envs.Flippers`, `wb.envs.TicTacToe`, `wb.envs.minimax_policy` (+ tests `tests/infra/`) | spécification dans les notes de `ch26.json` |
| ch. 28 | `wb.datasets.load_sample_image` (photos libres, peintures du domaine public, licences dans une data card) | notes de `ch28.json` |
| ch. 29 | `wb.datasets.make_trap_dataset` (doublons entre train et test, fuite) | exercice 29.13 |
| CP3 (MP3), B6 | loader Adult (`fetch_openml(data_id=1590)` + copie de secours ≈ 1 Mo) et sa data card | décision §22 |
| B4 | 20 Newsgroups (scikit-learn, cache) ; 30 questions annotées sur Holmes (B4.20) | |
| B5, B7 | dépendances `diffusers` (oracle), `fastapi`, `uvicorn`, `pydantic` : versions Colab à vérifier et figer (§21) | |

## ⚠️ À valider sur Colab

| # | Élément | Pourquoi | Statut |
|---|---|---|---|
| 1 | `00_setup/demo.ipynb` : cellule de setup (montage Drive, clonage dans `MyDrive/workbookIA`, puis `git pull` à la 2ᵉ ouverture) | le montage Drive et git sur Drive n'existent pas dans l'environnement de génération | à faire |
| 2 | Rapport de `wb.setup` sur Colab : Python 3.13.x, torch `2.11.0+cu…`, aucune ligne ⚠️ de version | vérifie l'alignement des versions (§21) | à faire |
| 3 | Téléchargements Fashion-MNIST et CIFAR-10 sur Colab (cache `/content/wb_cache`) | testés ici via les sources de secours (GitHub, Hugging Face) ; la source torchvision n'a pas pu l'être | à faire |
| 4 | GPU T4 : `Device : cuda` après changement du type d'exécution | pas de GPU dans l'environnement de génération | à faire |
| 5 | Installation locale Windows et Mac (guide `INSTALL_LOCAL.md`) | seule l'installation Linux a été testée | à faire quand tu installeras |
| 6 | Notebook 0A sur Colab : cellule de setup avec `chapter="0A"`, `pytest` lancé par `subprocess` en 0A.26, lecture de `penguins.csv` par `wb.datasets.data_dir()` | exécuté ici en local seulement | à faire au 1ᵉʳ chapitre |
| 7 | Exercices git 0A.10–0A.12 depuis Colab (section 5 de `00_setup/COLAB.md`, jeton secret) | pas de compte GitHub de test ici | à faire |
| 8 | Notebook 0B sur Colab : tests de `linalg_basics` lancés par `subprocess` (0B.38–0B.43), mesures de vitesse de 0B.40 et 0B.54 (objectifs : NumPy ≥ 10 fois, ordre des produits ≥ 20 fois plus rapide) | exécuté ici sur 2 cœurs seulement | à faire |
| 9 | Notebook du ch. 1 sur Colab : chargement de MNIST, des taches solaires, de Holmes et de Verne par `wb.datasets` ; durée de 1.23 (`MLPClassifier`, 5 000 images en `FAST_MODE`, 60 000 sinon) | exécuté ici en local seulement (≈ 25 s en `FAST_MODE`) | à faire |
| 10 | Notebook du ch. 2 sur Colab : tests de `mylearn.stats` lancés par `subprocess`, `scipy.stats.bootstrap(..., rng=)` (SciPy ≥ 1.15 ; Colab : 1.16.3), durée de 2.23 en mode complet (≈ 5 s ici) | exécuté ici en local seulement (≈ 26 s) | à faire |

## Calibrage

*(retours de l'apprenant via P7 et ajustements appliqués aux chapitres suivants)*

| Date | Retour | Ajustement |
|---|---|---|
| | Niveau Python : « débutant » à confirmer (BIBLE §1) | à préciser après le ch. 0A |
| 2026-09-30 | (estimation, pas encore de retour) Temps d'étude : 4 min par page, 2 min par flashcard, durées des exercices selon ★ | à recalibrer avec tes temps réels (journal) après 0A et 0B |

## Écarts par rapport au SYLLABUS

| Session | Chapitre | Écart | Raison |
|---|---|---|---|
| 3 | 0A | aucun écart d'exercice (ID, titres, types, parcours et durées conformes à `ch0A.json`) ; la référence et les tests de `utils` (prévus en session 4) sont faits en avance | les tests fixent le contrat avant d'écrire les exercices 0A.63–0A.66 |
| 4 | 0A | 0A.64 (`utils.argmax`) passe de ★★ / 26 min à ★★★ / 35 min (`ch0A.json` corrigé, `syllabus.py build`) ; titres mis en forme (code entre accents graves, « transposée ») sans changement de sens | la relecture indépendante a jugé 0A.64 plus difficile que 0A.66 (quatre cas d'axe, cinq erreurs, types de retour) |
| 5 | 0B | aucun écart d'exercice (ID, titres, types, parcours et durées conformes à `ch0B.json`) ; la référence et les tests de `linalg_basics` (prévus en session 6) sont faits en avance ; énoncés précisés après vérification (0B.Q4 : q = −1 au lieu de −2, pour que la catégorie « oscille » soit sans ambiguïté) | même méthode qu'en 0A |
| 6 | 0B | 0B.54 : titre « … des dizaines de fois plus vite » au lieu de « … cent fois plus vite » (`ch0B.json` corrigé, `syllabus.py build`) ; ID, type, durée et parcours inchangés | mesuré ≈ 50 fois plus rapide sur 2 cœurs : un produit matrice-vecteur est limité par la mémoire, « cent fois » n'est pas tenable sur CPU |
| 7 | 1 | 1.4 et 1.11 : ★ → ★★ et 15 min ; 1.22 : 25 → 30 min, vérification manuelle → `wb.check` (graine imposée) ; `ch01.json` corrigé, `syllabus.py build` ; ID, types, titres et parcours inchangés | relectures indépendantes : 1.4 (projection sur un axe) et 1.11 (textes en vecteurs de fréquences) plus longs que prévu ; 1.22 vérifiable exactement avec une graine |
| 8 | 2 | 2.6 : cinq exemples au lieu des trois du livre (d : 7 exemples ; h : 4 décimales) ; `ch02.json` (champ `examples`) mis à jour ; ID, titre, type, durée et parcours inchangés | le livre imprime les réponses de la version à trois exemples (§2.5.3) |
| 9 | 2 | 2.13 : ★ → ★★ et 15 → 30 min ; 2.15 : ★★ → ★★★ et 25 → 40 min ; 2.25 : 20 → 25 min ; 2.31 : 20 → 30 min (`ch02.json` corrigé, note sur l'ordre des ★ réécrite, `syllabus.py build` : chapitre ≈ 18 h au lieu de 17) ; ID, titres, types et parcours inchangés | relecture indépendante : 2.13 et 2.15 demandent trois ou quatre fonctions avec `axis`, validation et cas limites (loin d'une « application directe ») ; 2.25 a six sous-questions, 2.31 une docstring, des doctests et des tests confrontés à trois versions boguées |
| 10 | 3 | aucun écart d'ID, de titre, de type, de durée ou de parcours. Chiffres du workbook là où le livre imprime ses résultats : Q4 (40 %, un quart, 200 fléchettes), 3.2 (20 autres points), 3.3 (glacier), 3.7 (partie 1 : un autre test ; partie 2 : les chiffres du livre) ; champ `examples` de `ch03.json` mis à jour (sans les réponses de 3.2). Énoncés précisés après vérification : n'arrondir qu'à la fin (papier, 3.6), 3.4 q3–q5, 3.5 n, 3.7 m, 3.8 q5 (deux modèles sur le même jeu de test), 3.11 (sections 1 à 5, 7 et 9) ; objectif « prévalence » et 🕰️ precision-recall de `ch03.json` réécrits comme dans la fiche | le livre imprime ses résultats ; relectures indépendantes (fuites, consignes ambiguës, nuance de McDermott et coll. 2024) |
