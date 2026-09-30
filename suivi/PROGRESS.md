# PROGRESS : état de la génération du workbook

*Tenu par Claude à chaque session (BIBLE §18). Dernière mise à jour : 2026-09-30, session 6.*

## Prochaine étape

➡️ **Session 7 : prompt P2, chapitre 1 (Introduction au machine learning et au deep learning)**, une session, 43 exercices ; pièce jointe : le **Volume 1** du livre. Contrat : section 1 de `docs/SYLLABUS.md` et `docs/syllabus/data/ch01.json`. Premier chapitre adossé au livre : fiche avec le guide de lecture ⏩ du parcours rapide, encadrés 🕰️ vérifiés par recherche web, aucun passage recopié. Constructeur `tools/chapters/build_ch01.py` avec le kit `tools/chapters/chapter_kit.py` (modèle : `build_ch00b.py`).

Côté apprenant : le chapitre 0B est complet. Relance `python tools/start_chapter.py 0B` : la commande copie maintenant le notebook (tes fichiers déjà copiés ne sont pas touchés ; dans ta copie du tableau de bord, corrige à la main le titre de 0B.54 si elle porte encore « cent fois plus vite »). Toujours en suspens : les questions du rapport de session 2 et la liste « À valider sur Colab » (point 8 ajouté pour 0B).

## Statut des chapitres

Légende : 📅 planifié · 🛠️ en cours (étape exacte indiquée) · ✅ généré · 🔍 audité

| ID | Chapitre | Statut | Session | Remarques |
|---|---|---|---|---|
| setup | Mise en place (dépôt, outils, datasets, documentation) | ✅ | 1 | voir « Session 1 » |
| — | Syllabus et parcours | ✅ | 2 | 2 019 exercices, stubs de tous les modules ; voir « Session 2 » |
| 0A | Python, notebooks et outils | ✅ | 3, 4 | 84 exercices (dont 55 dans le notebook, parties A à H), 290 vérifications `wb.check`, 30 flashcards, `mylearn.utils` (référence + 44 tests) ; solutions exécutées en ≈ 20 s |
| 0B | Maths du lycée au ML | ✅ | 5, 6 | 74 exercices (dont 22 dans le notebook, parties A à D, et 174 vérifications papier en partie 0), 281 vérifications `wb.check` au total, 13 figures, 30 flashcards, `mylearn.linalg_basics` (référence + 53 tests) ; solutions exécutées en ≈ 15 s |
| 1 | Introduction | 📅 | | |
| 2 | Hasard et statistiques | 📅 | | |
| 3 | Probabilités et mesure de la qualité | 📅 | | |
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
