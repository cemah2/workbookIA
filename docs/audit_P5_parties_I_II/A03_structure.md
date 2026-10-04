# Audit P5, tour 2 : A3, conformité structurelle (BIBLE §11, §12, §17, §19)

**Périmètre** : les 13 chapitres publiés (0A à 11, soit 665 exercices), CP1, CP2, MP1 et MP2, au commit `0863f10`. Le dépôt a été lu sans être modifié (`git status --short` vide à la fin). Les reconstructions de notebooks et les tests ont tourné dans une copie `tar` (`structure/copy/`).
**Méthode** : un script par contrôle (`structure/check1_exercises.py` à `check9_checkpoints.py`, plus `check1b`, `check1c`, `check6b` et `compare_rebuild.py`), avec leurs sorties brutes (`structure/raw_*.txt`). Chaque écart signalé a été relu dans le fichier. Les faux positifs des scripts ont été écartés à la main : verbes d'action, temps « 1 h 25 » lus « 1 h », numéros pris pour des chapitres dans les data cards, tags avec `_`, tracebacks voulus de 0A.37 et 0A.61, cellules vérifiées par `verdict` en 2.23 et 6.21.

## Contrôles globaux

- `python tools/syllabus.py check` : 46 chapitres, 2 019 exercices, 0 problème ; les fichiers générés (SYLLABUS, PARCOURS, tableau de bord, MANIFEST) sont à jour.
- `python tools/build_answers.py --check` : ✅, 1 827 réponses, `answers.json` à jour.
- `python tools/export_flashcards.py --check` : ❌, code de sortie 1, 320 cartes (voir le constat 3).
- `pytest --impl=ref` (copie) : 1 578 réussis, 4 ignorés (réseau). Le test `answers.json` a d'abord échoué, parce que ma reconstruction avait vidé les sorties de la copie. Après restauration, `tests/infra/test_repo_structure.py` donne 47 réussis et 1 ignoré (`raw_pytest_ref_rerun_repo_structure.txt`).
- `pytest --impl=stubs` : 1 324 échecs. C'est attendu : les tests ne sont pas vides.
- Mode apprenant : 249 réussis, 1 333 ignorés.
- Reconstruction par les 17 `build_*.py` (13 chapitres, CP1, CP2, MP1, MP2) : sources et tags identiques aux notebooks publiés, sauf au ch. 2 (constat 4).
- Durées des notebooks de solutions, lues dans les métadonnées d'exécution publiées :
  - chapitres : de 11,9 s (0B) à 57,5 s (ch. 11) ; la cellule la plus longue prend 13,4 s ;
  - CP1 2,0 s, CP2 2,5 s, MP1 20,3 s, MP2 116,5 s (cellule la plus longue : 44,9 s).
  - Le budget FAST_MODE (moins de 10 min par notebook, aucune cellule au-delà de 3 min) est tenu partout, avec une marge d'au moins 4×. Aucune mesure n'a échoué : je n'ai rien relancé pour le temps.

## Tableau récapitulatif

✅ = aucun écart ; un nombre = écarts non justifiés, sauf mention contraire. Le constat qui détaille chaque écart est indiqué entre parenthèses.

| Ch. | 1a · exercices ↔ contrat (02, 03, tableau de bord, 04, 05, 06) | 1b · 05 : rubriques §12 manquantes | 2 · fiche (10 rubriques, ⏩) | 3 · ordre de 02 | 4 · notebooks 03 et 05 | 5 · composition §11 | 6 · indice 3 : T / T+P ; 🔨 avec code complet | 7 · flashcards | 8 · annexes, suivi, data cards |
|---|---|---|---|---|---|---|---|---|---|
| 0A | 123 (titres 117 ; en-tête `mylearn` 5 ; `check` 1) | 64 | ✅ | ✅ | 1 (🚀) | 3 justifiés (note périmée, C14) | 5 % / 65 % ; 2/29 | ✅ | 1 (MNIST) |
| 0B | 81 (titres 68 ; `mylearn` 4 ; `check` 6 ; préfixe 🔁 3) | 58 | ✅ | ✅ | 1 (🚀) | 2 justifiés | 11 % / 33 % ; 0/7 | 2 doublons | 4 (3 data cards, glossaire) |
| 1 | 5 (titres) | 22 | ✅ | ✅ | 1 (🚀) | ✅ | **42 % / 74 %** ; 0/3 | 2 doublons | ✅ |
| 2 | 8 (`mylearn`) | 11 | ✅ | ✅ | 2 (🚀 ; jamais reconstruit) | ✅ | **50 % / 79 %** ; 1/8 | ✅ | ✅ |
| 3 | 6 (`check`) | 5 | ✅ | ✅ | 1 (🚀) | ✅ | 4 % / 43 % ; **7/7** | 1 doublon | ✅ |
| 4 | 3 (`check`) | 4 | ✅ | ✅ | 1 (🚀) | ✅ | 19 % / 52 % ; **4/4** | ✅ | ✅ |
| 5 | ✅ | 4 | ✅ | ✅ | 1 (🚀) | ✅ | **52 % / 76 %** ; **5/5** | 1 doublon | ✅ |
| 6 | ✅ | 8 | ✅ | ✅ | 1 (🚀) | ✅ | **50 % / 75 %** ; **5/5** | ✅ | ✅ |
| 7 | 5 (`check`) | 5 | ✅ | ✅ | 1 (🚀) | ✅ | **43 % / 62 %** ; **7/7** | ✅ | ✅ |
| 8 | 2 (libellé de partie 0 ; Fil rouge) | 5 | ✅ | ✅ | 1 (🚀) | ✅ | **62 % / 71 %** ; **6/6** | 1 (2 tags cassés) | ✅ |
| 9 | 1 (Fil rouge) | 5 | ✅ | ✅ | 1 (🚀) | 1 justifié | 5 % / 36 % ; **8/8** | ✅ | ✅ |
| 10 | 7 (Fil rouge 4 ; préfixe 🔁 3) | 6 | ✅ | ✅ | 1 (🚀) | ✅ | 100 % / 100 % (connu) ; 2/5 | ✅ | 1 (cheatsheet) |
| 11 | 7 (Fil rouge 4 ; préfixe 🔁 3) | 16 | 1 (6 ressources) | ✅ | 1 (🚀) | ✅ | 100 % / 100 % (connu) ; 0/6 | ✅ | 1 (cheatsheet) |

**Lecture des colonnes**

- **1a.** Les ID, types, ★, durées et parcours concordent partout entre les contrats et 02, 03, 04, 05, 06 et le tableau de bord. Aucun ID ne manque ni n'est en trop. Chaque exercice a ses 3 indices `<details>`. 06 a une place pour chaque sous-question, et un « Réponse rédigée » pour les questions ouvertes.
  - Ne sont pas comptés : le titre de 0A.53 (Écarts, session 4) et le préfixe « Ch. N : » ajouté aux rappels (convention des ch. 4 à 9).
  - Les quiz de 02 n'ont jamais d'étoiles (constat 17).
- **1b.** Rubriques « Erreurs fréquentes », « Variante » et « Pourquoi / Démarche » manquantes dans 05. Les quiz, rappels et 🗣️ sont exclus ; les 💼 ont tous une réponse modèle et des relances.
- **3.** L'ordre des sections est respecté partout. La place de 📈 et 🛠️ (0A, 0B, 7, 8, 9, 11) n'est fixée nulle part (constat 17).
- **4.** Tous les notebooks d'exercices :
  - ont un titre, un badge Colab au bon chemin et une seule cellule de setup, identique au modèle de `nbbuild` ;
  - suivent l'ordre en-tête → TODO → vérification ;
  - finissent par un bilan et 3 questions d'auto-évaluation ;
  - n'ont ni sortie, ni `!`, ni `%`.
  Tous les notebooks de solutions sont exécutés et gardent leurs sorties, sans erreur non voulue ni ligne ❌ ou ⏳.
- **6.** T = l'indice 3 donne la réponse finale de toutes les sous-questions, ou de toutes sauf une. P = il en donne au moins une. Base : les exercices à réponse courte de 02 (🧠, 🔁, ✏️, ∂, 🧮, 📈), lus un par un contre 05. « Code complet » = une fonction entière (`def` … `return`, au moins 5 lignes).

### Checkpoints et mini-projets (contrôle 9)

| | Durée | Types | Partie antérieure | Barème sur 20 (sujet = corrigé, question par question) | Synthèse | Notebooks | Mini-projet | Data cards |
|---|---|---|---|---|---|---|---|---|
| CP1 + MP1 | 117 min ✅ | 10 types (§22) ✅ | 2 points (10 %) sur la partie 0 : CP1.2, CP1.3 ✅ | 14 questions, total 20 ✅ | carte Mermaid et fiche d'une page (20 formules) ✅ | sujet sans sortie ; solutions exécutées en 2,0 s ✅ | README complet ; départ et solution ; grille 2+4+4+2+3+3+2 = 20 ; extensions ✅ | ❗ Penguins, Holmes et Verne ne citent ni CP1 ni MP1 (C10) |
| CP2 + MP2 | 117 min ✅ | 7 types (§22) ✅ | 2 points (10 %) sur la partie I : CP2.13 ✅ | 13 questions, total 20 ✅ | carte Mermaid et fiche d'une page (22 formules) ✅ | sujet sans sortie ; solutions en 2,5 s ✅ | grille 4+3+3+2+3+3+2 = 20 ; extensions ; solution en 116,5 s ✅ | ❗ California ne cite ni CP2 ni MP2 ; Penguins ne cite pas CP2 (C10) |

Deux titres du sujet de CP1 sont raccourcis (constat 6).

### Indices de niveau 3 des exercices à réponse courte (contrôle 6)

| Ch. | exercices | T | P | N | T | T + P | T parmi les 🧠 | T parmi ✏️/∂/🧮/📈 |
|---|---|---|---|---|---|---|---|---|
| 0A | 20 | 1 | 12 | 7 | 5 % | 65 % | 1/12 | 0/8 |
| 0B | 45 | 5 | 10 | 30 | 11 % | 33 % | 4/12 | 0/30 |
| 1 | 19 | 8 | 6 | 5 | 42 % | 74 % | 8/11 | 0/5 |
| 2 | 24 | 12 | 7 | 5 | 50 % | 79 % | 11/12 | 0/9 |
| 3 | 23 | 1 | 9 | 13 | 4 % | 43 % | 1/12 | 0/8 |
| 4 | 21 | 4 | 7 | 10 | 19 % | 52 % | 4/10 | 0/8 |
| 5 | 21 | 11 | 5 | 5 | 52 % | 76 % | 8/10 | 2/8 |
| 6 | 24 | 12 | 6 | 6 | 50 % | 75 % | 11/12 | 1/9 |
| 7 | 21 | 9 | 4 | 8 | 43 % | 62 % | 9/11 | 0/7 |
| 8 | 21 | 13 | 2 | 6 | 62 % | 71 % | 11/11 | 0/7 |
| 9 | 22 | 1 | 7 | 14 | 5 % | 36 % | 0/11 | 0/8 |
| 10 | 20 | 20 | 0 | 0 | 100 % | 100 % | 9/9 | 8/8 |
| 11 | 24 | 24 | 0 | 0 | 100 % | 100 % | 12/12 | 9/9 |

N = l'indice 3 ne donne aucune réponse finale (méthode, calcul posé, première ligne, renvoi à la fiche). Le classement de chaque exercice est dans `manual_hints.py` ; les textes comparés sont dans `hints_ch*.txt` et `raw_check6_hints.json`.

## Constats

### MAJEUR

**1. MAJEUR** · `chapitres/ch02_stats/04_indices.md:61` (2.Q3) · « Deux affirmations sont vraies : la 1 et la 2. »

- **Problème.** Le §12 dit : « **Indice 3** (presque la solution : pseudo-code ou première ligne) ». Le défaut, connu pour les ch. 10 et 11 (tri A8, constat 13), touche aussi six autres chapitres : aux ch. 1, 2, 5, 6, 7 et 8, l'indice 3 donne toutes les réponses de presque tous les quiz (T : 8/11, 11/12, 8/10, 11/12, 9/11, 11/11).
  - Aux ch. 7 et 8, ces réponses sont vérifiées par `wb.check` dans la partie 0 (44 empreintes `7.Q*` et 31 empreintes `8.Q*` dans `answers.json`). La vérification se réduit alors à recopier l'indice.
  - 0A, 3 et 9 respectent la règle : l'indice 3 y donne le critère ou le calcul posé. Exemples : `chapitres/ch09_overfitting/04_indices.md:25` (9.Q1) et `chapitres/ch03_probabilites/04_indices.md:321` (3.2 : « Tu dois trouver 10 positifs et 11 prédictions positives. »).
- **Occurrences (T, 77 indices)** :
  - 0A : Q1 ;
  - 0B : Q1, Q2, Q7, Q11, R1 ;
  - ch. 1 : Q1 à Q5, Q7, Q8, Q10 ;
  - ch. 2 : Q2 à Q12, R1 ;
  - ch. 3 : Q8 ;
  - ch. 4 : Q2, Q3, Q6, Q9 ;
  - ch. 5 : Q1 à Q3, Q6 à Q10, R2, 5.4, 5.9 ;
  - ch. 6 : Q1 à Q11, 6.10 ;
  - ch. 7 : Q1 à Q8, Q10 ;
  - ch. 8 : Q1 à Q11, R1, R3.
- **Exemples** :
  - `chapitres/ch01_introduction/04_indices.md:25` (1.Q1) : « Deux programmes apprennent (le 2 et le 4) ; les trois autres appliquent des règles écrites. »
  - `chapitres/ch02_stats/04_indices.md:169` (2.Q9) : « Sont **avec** remise : le rééchantillon bootstrap, les commandes du café et `rng.choice` par défaut. Les autres sont sans remise, et l'affirmation 7 est fausse. »
  - `chapitres/ch08_train_test/04_indices.md:25` (8.Q1) : les quatre réponses (a) « La boucle passe à l'exemple suivant. », b) les trois ingrédients…).
  - `chapitres/ch05_courbes/04_indices.md:321` (5.4) : « $f(-1) = 2$, $f(1) = -2$, $f(2{,}5) = 8{,}125$ … la valeur maximale 2 est atteinte deux fois ».
  - `chapitres/ch05_courbes/04_indices.md:415` (5.9) : « … soit environ 40 minutes par pas ».
  - `chapitres/ch06_information/04_indices.md:79` (6.Q4) et `chapitres/ch07_classification/04_indices.md:61` (7.Q3).
- **Sévérité.** Il s'agit d'une fuite systématique de réponses, et sa correction fixe une convention transversale. Le tri du tour 1 a classé les ch. 10 et 11 « mineur » : à harmoniser.
- **Correction.**
  1. Une règle au §22 : « pour un exercice à réponse courte, l'indice 3 résout le premier item comme modèle (la « première ligne » du §12) et donne pour les autres le critère ou le calcul posé, jamais le verdict, la lettre ni la valeur ».
  2. Réécrire les 77 indices. Exemples :
     - 2.Q3 : « Une graine fixe le point de départ d'un calcul déterministe : elle rend la suite reproductible, rien de plus. L'affirmation 1 est donc vraie ; juge les cinq autres avec cette seule phrase. »
     - 1.Q1 : « Cherche qui a fixé la règle : un humain qui l'a écrite (un taux, un seuil, une liste) ou des exemples dont le programme l'a tirée. Le 1 applique un taux de TVA écrit dans la loi : règle écrite. Fais de même pour les quatre autres. »
  3. Garder `structure/check6_hints.py` comme détecteur : sa détection automatique ne voit qu'une partie des cas, d'où le classement manuel.

**2. MAJEUR** · `chapitres/ch03_probabilites/04_indices.md:702-720` (indice 3 de 3.16) · « def _binary_counts(y_true, y_pred, pos_label): » … « def precision(y_true, y_pred, pos_label=1, average="binary", zero_division=0.0): » … « return _ratio(tp, tp + fp, zero_division) »

- **Problème.** Selon le §12, l'indice 3 est du « pseudo-code ou [une] première ligne ». Or, aux ch. 3 à 9, l'indice 3 donne une fonction complète pour les **42 🔨 sur 42**, de 5 à 49 lignes. Le §22 (2026-10-01, vérification du ch. 3) note même comme une qualité qu'« une librairie écrite d'après les seuls indices de niveau 3 passe les 230 tests ».
  - Ailleurs, la règle du §12 est suivie : 0A (2/29), 0B (0/7), ch. 1 (0/3), ch. 2 (1/8) et ch. 11 (0/6), où l'indice 3 se limite aux lignes clés. Exemple : `chapitres/ch02_stats/04_indices.md:665` (2.15 : `values = np.moveaxis(np.sort(arr, axis=axis), axis, 0)` …). Le ch. 10 est à 2/5.
  - D'autres exercices de notebook des ch. 4 à 9 reçoivent aussi du code complet (5, 3, 1, 8, 6 et 6 exercices).
  - L'apprenant qui ouvre l'indice 3 recopie la solution, et le 🔨 « from scratch » devient une lecture.
  - Deux conventions coexistent. Les parties III à VII, les plus riches en 🔨, en attendent une seule.
- **Correction.** Trancher et l'écrire au §22.
  - **A (recommandé)** : garder le §12. L'indice 3 d'un 🔨 donne la signature, le squelette et les deux ou trois lignes clés (modèle : 2.15, ch. 11). Réduire ainsi les 42 indices des ch. 3 à 9 ; le code complet reste dans 05.
  - **B** : légaliser au §12 « indice 3 = implémentation complète lisible » pour les 🔨, en disant ce qui le distingue alors de la solution.
  - Liste des exercices : `raw_check6b_hints_code_detail.txt`.

### MINEUR

**3. MINEUR** · `chapitres/ch08_train_test/flashcards.csv:18` et `:19` · « …;dlwb::ch08::erreur type »

- **Problème.** Le commit `e96506d` a appliqué le constat A1 8 (« erreur type » sans trait d'union) et a aussi remplacé le tag `dlwb::ch08::erreur-type`.
  - Anki coupe les tags aux espaces : chaque carte reçoit deux tags, `dlwb::ch08::erreur` et un tag racine parasite `type`.
  - `python tools/export_flashcards.py --check` sort maintenant en code 1 (« tags attendus sous la forme dlwb::chXX::concept », lignes 18 et 19). Il sortait en code 0 au tour 1 (AUDIT §5).
  - Aucun test ne lance ce contrôle sur le dépôt réel : `tests/infra/test_tools.py:266` travaille dans un dossier temporaire. `pytest` reste donc vert.
- **Correction.**
  1. Remettre `dlwb::ch08::erreur-type` sur les deux lignes.
  2. Ajouter à `tests/infra/test_repo_structure.py` un test sur le dépôt réel : `assert export_flashcards.export(tmp_path / "deck.csv", check_only=True, out=lambda *a: None) == 0`.

**4. MINEUR** · `chapitres/ch02_stats/03_notebook.ipynb`, cellules 27, 54, 58, 89 et 100 (et `05_solutions.ipynb`, cellules 29, 61, 65, 104 et 116) · « if any(answer is ... for answer in [prediction_2_14a, prediction_2_14b, … ]): »

- **Problème.** Le §22 (2026-10-01) fait traiter `None` comme une prédiction pas encore faite (`tools/chapters/chapter_kit.py:161` : « if any(answer is ... or answer is None for answer in … »). Deux promesses de cette ligne ne sont pas tenues.
  - **Les notebooks du ch. 2 n'ont jamais été reconstruits** (exécutés le 2026-09-30). Une reconstruction par `build_ch02.py` ne change que ces 5 cellules : 2.14, 2.20 (deux cellules), 2.27 et 2.28. Aujourd'hui, un apprenant qui écrit `None` voit l'expérience avant d'avoir prédit.
  - **L'en-tête « mylearn » ne peut pas apparaître.** La même ligne promet « **mylearn :** `fichier` » dans l'en-tête des 🔨 de 0A, 0B et du ch. 2 « à leur prochaine reconstruction ». Une reconstruction ne l'ajoutera pas :
    - `tools/chapters/build_ch00a.py:34-59` a sa propre classe `Ex`, sans champ `mylearn` ;
    - `build_ch00b.py` et `build_ch02.py` ne passent jamais `mylearn=` (0 occurrence, contre 3 à 7 dans `build_ch03.py` à `build_ch11.py`).
  - Les 17 🔨 dont le contrat nomme un module n'ont donc pas ce champ : 0A.26 et 0A.63 à 0A.66 ; 0B.38, 0B.39, 0B.42, 0B.43 ; 2.13, 2.15, 2.16, 2.19, 2.21, 2.22, 2.26, 2.28.
- **Correction.**
  1. Ajouter les champs manquants :
     - `mylearn="stats.py"` aux 8 `Ex` de `build_ch02.py` ;
     - `mylearn="linalg_basics.py"` aux 4 `Ex` de `build_ch00b.py` ;
     - un champ `mylearn` et sa ligne d'en-tête à l'`Ex` de `build_ch00a.py`, ou passer à `chapter_kit.Ex` : `_example.py` pour 0A.26, `utils.py` pour 0A.63 à 0A.66.
  2. Reconstruire 0A, 0B et le ch. 2, puis lancer `run_all_notebooks.py … --inplace` et `build_answers.py`.
  3. Ajouter un test : tout 🔨 qui a un champ `mylearn` a la ligne dans 03.

**5. MINEUR** · `docs/syllabus/data/ch03.json:907` (3.16) · « "check": "pytest", », affiché dans `docs/SYLLABUS.md:947` : « | 3.16 | 🔨 | accuracy, precision, recall, F-beta et F1 (cas binaire) | ★★★ | 40 | synth | 03 | 3.15 | RMC | pytest | » ; or `chapitres/ch03_probabilites/03_notebook.ipynb`, cellule 39, contient « wb.check("3.16a", mm.accuracy(spam_true, spam_pred), computed=True) »

- **Problème.** 21 exercices sont vérifiés par `wb.check` (avec leurs réponses dans `answers.json`), mais leur contrat dit encore `pytest` ou `manual`. Les ch. 5 et 9 ont mis ce champ à jour (Écarts, sessions 13 et 20) ; les autres non.
  - `pytest` → `wb.check+pytest` :
    - `ch0A.json:1268` (0A.26) ;
    - `ch0B.json:1474`, `:1499`, `:1572`, `:1596` (0B.38, 0B.39, 0B.42, 0B.43) ;
    - `ch03.json:907`, `:971`, `:1083`, `:1107`, `:1130`, `:1175` (3.16, 3.19, 3.24, 3.25, 3.26, 3.28) ;
    - `ch04.json:792`, `:963` (4.16, 4.24) ;
    - `ch07.json:790`, `:965`, `:987`, `:1057`, `:1101` (7.14, 7.22, 7.23, 7.26, 7.28).
  - `manual` → `wb.check` : `ch0B.json:1428` (0B.36), `:1547` (0B.41) et `ch04.json:941` (4.23).
- **Correction.**
  1. Corriger ces 21 valeurs, lancer `python tools/syllabus.py build` et ajouter une ligne dans `suivi/PROGRESS.md` § Écarts.
  2. Faire refuser par `syllabus.py check` un exercice qui a des empreintes dans `answers.json` sans `wb.check` dans son champ `check`. Sans garde-fou, la dérive reviendra.

**6. MINEUR** · Titres raccourcis ou reformulés dans 04, 05, la partie 0 et le sujet de CP1. Exemples :

- `chapitres/ch00a_python/05_solutions.md:306` : « ### Ex 0A.15 — Chaînes » (contrat et 03 : « Chaînes : nettoyer les noms d'espèces de Penguins brut ») ;
- `chapitres/ch00a_python/05_solutions.md:353` : « ### Ex 0A.24 — Modules de la bibliothèque standard » (04, ligne 784 : « Importer des modules » ; contrat : « Importer des modules : math, random, statistics et Counter ») ;
- `chapitres/ch00b_maths/05_solutions.md:309` : « ### Ex 0B.12 — Dénombrer » (02, ligne 338 : « Dénombrer : choix successifs, factorielle et C(n, k) ») ;
- `chapitres/ch01_introduction/04_indices.md:401` : « ### Ex 1.8 — Galton (1886) » (02, ligne 266 : « Galton (1886) : l'origine du mot « régression » »).

- **Problème.** Le titre fait partie du contrat (§11). Aux ch. 2 à 11, 04 et 05 reprennent le titre du contrat ; en 0A, 0B et au ch. 1, non.
  - **0A** : 117 titres, dont 98 raccourcis (34 dans 04, 60 dans 05, 4 libellés de partie 0), 8 réduits à un extrait et 11 reformulés.
  - **0B** : 62 titres raccourcis, plus 0B.53 reformulé dans 05.
  - **Ch. 1** : 5 titres dans 04.
  - **Libellés de partie 0** : 0A.2, 0A.3, 0A.6 et 0A.7 (`tools/chapters/build_ch00a.py:85, 95, 119, 127`) ; 8.4 (`tools/chapters/build_ch08.py:264` : « Quelle confiance accorder à une accuracy de test ? », où le contrat ajoute « Erreur type et taille du test »).
  - **Sujet de CP1** : `checkpoints/partie_1/01_examen_sujet.md:39` « ### CP1.1 — Questions flash » et `:156` « ### CP1.12 — Un LLM expliqué en cinq lignes », alors que le tableau du même fichier (lignes 21 et 32) donne les titres complets.
  - **0B.33** : « Calculer avec Python : puissances, arrondis, `abs`, signe et `C(n, k)` » partout (02, ligne 699 ; 03 ; 04, ligne 969 ; 05, ligne 701), contre « … |x|, signe et C(n, k) » au contrat. Ce changement n'est pas consigné : les Écarts des sessions 5 et 6 disent « aucun écart … titres ». Celui de 0A.53, lui, l'est (session 4).
  - Liste complète : `raw_check1_titles_summary.txt`.
- **Correction.**
  - Ce constat complète P5 et ne le répète pas. Le passage qui harmonise les en-têtes « Ex N.k — Titre [icône] » de 04 et 05 doit prendre le titre dans le JSON, pas dans l'en-tête existant (0A, 0B, ch. 1).
  - Faire de même pour les libellés `Paper(...)` et les en-têtes du sujet de CP1.
  - Pour 0B.33 : mettre le titre publié dans `ch0B.json` (le `|` de |x| cassait les tableaux ; le kit l'échappe désormais), puis ajouter une ligne aux Écarts.

**7. MINEUR** · `docs/syllabus/data/ch10.json:346` · « "title": "Ch. 9 — Régularisation L2 : que deviennent les poids ?", » ; publié dans `chapitres/ch10_neurones/02_exercices.md:94` sous la forme « ### 10.R1 — Ch. 9 : régularisation L2, que deviennent les poids ? 🔁 ★ ⏱️ 5 min »

- **Problème.** Aux ch. 4 à 9, les sessions ont fixé la règle « R1 à R3 sans le préfixe « Ch. N — » » dans le contrat ; le préfixe est ajouté à la publication (`suivi/PROGRESS.md:464` et `:467`).
  - Le préfixe reste dans 9 titres publiés : 10.R1 à R3, 11.R1 à R3 (`ch11.json:461`…) et 0B.R1 à R3 (« 0A : … » ; les 04 et 05 de 0B l'enlèvent, les 02 et 06 le gardent). SYLLABUS, PARCOURS et le tableau de bord affichent donc une autre forme que le chapitre.
  - 62 titres de rappels des chapitres à venir le portent aussi : 12 à 15, 21 à 29, B1 à B8.
  - La règle n'est écrite que dans PROGRESS, pas au §22.
- **Correction.**
  1. Retirer le préfixe de ces 71 titres (par exemple `"title": "Régularisation L2 : que deviennent les poids ?"`), puis lancer `syllabus.py build`.
  2. Écrire la règle au §22.
  3. Faire refuser par `syllabus.py check` un titre 🔁 qui commence par `^(Ch\. \d+|0A|0B|B\d) *[—:]`.

**8. MINEUR** · `chapitres/ch08_train_test/02_exercices.md:154` (8.1) · « **Prérequis :** ch. 2 (proportions) · fiche §8.3, §8.4, §8.5.1 · **Parcours :** R, M » ; le contrat de 8.1 donne le fil « Penguins »

- **Problème.** L'en-tête du §11 comporte « **Fil rouge :** … » quand l'exercice en a un. Ce champ manque dans 10 en-têtes dont le contrat donne un fil ; d'ailleurs, aucun 02 des ch. 7 à 11 n'a ce champ.
  - 8.1 (Penguins) ;
  - 10.3, 10.4, 10.5 et 10.6 (portes logiques) : `chapitres/ch10_neurones/02_exercices.md:173`, `:187`, `:200`, `:219` ;
  - 11.2, 11.8, 11.10 et 11.11 (bandit) : `chapitres/ch11_raisonnement/02_exercices.md:212`, `:326`, `:366`, `:383` ;
  - l'en-tête de 9.15 dans le notebook : `chapitres/ch09_overfitting/03_notebook.ipynb`, cellule 58, produite par `tools/chapters/build_ch09.py:946` (`Ex("9.15", …)` sans `thread=` ; contrat : « synth »).
- **Correction.**
  1. Ajouter après les prérequis « · **Fil rouge :** Penguins » (8.1), « · **Fil rouge :** portes logiques » (10.3 à 10.6) et « · **Fil rouge :** bandit » (11.2, 11.8, 11.10, 11.11).
  2. Ajouter `thread="synthétique"` à l'`Ex` de 9.15 (libellé de P5), puis reconstruire le ch. 9.

**9. MINEUR** · `chapitres/ch00a_python/05_solutions.md:381-383` (0A.29) · « a) **100** : `b = a[2:5]` est une vue, `b[0]` est `a[2]` · b) **0** : `c = a[[0, 1]]` est une copie · c) **113** ($0 + 1 + 100 + 3 + 4 + 5$). **À retenir** : tranche = vue ; masque ou liste d'indices = copie. » ; autre exemple : `chapitres/ch11_raisonnement/05_solutions.md:105-112` (11.4 : six réponses commentées et un « À retenir », sans erreurs fréquentes ni variante)

- **Problème.** Le §12 demande, pour chaque exercice, « la réponse, la démarche détaillée, … les erreurs fréquentes, une variante ou une piste pour aller plus loin ». Il manque **213 rubriques** :
  - 0A : 64 (17 sans erreurs fréquentes, 40 sans variante, 7 sans « pourquoi ») ;
  - 0B : 58 ;
  - ch. 1 : 22 ;
  - ch. 2 : 11 ;
  - ch. 11 : 16, dont les exercices papier 11.1 à 11.7, sans variante ;
  - ch. 3 à 10 : de 4 à 8 par chapitre, presque tous des exercices de réflexion (⚖️, 📄, 🧮).
  - Liste : `raw_check1c_solutions.txt`.
  - Ce constat est distinct de A1 41 et de P5, qui portent sur le nom des rubriques.
- **Correction.**
  1. Compléter d'abord les exercices papier (✏️ et ∂ de 0B, 11.1 à 11.7) et les 🔨/📦 de 0A.
  2. Pour les items très courts de la partie 0 (★, 10 min au plus), soit compléter, soit écrire au §22 que « À retenir » remplace ces rubriques.

**10. MINEUR** · `data/cards/mnist.md:29` · « 1 (une image = 784 nombres, premier réseau de neurones en boîte noire), 2 (… ) »

- **Problème.** Une data card doit citer les « chapitres qui l'utilisent » (`data/cards/README.md:3`). Comparées aux fils du contrat et aux appels des loaders, plusieurs cartes ont des oublis :
  - **MNIST** : 0A manque (0A.54, 0A.55, 0A.59 ; `load_mnist` à `tools/chapters/build_ch00a.py:2321`), ainsi que 0B (0B.30) ;
  - **Penguins** (`data/cards/penguins.md:38`) : 0B (0B.8), CP1 (CP1.3, CP1.5) et CP2 (CP2.5) manquent ;
  - **California** (`data/cards/california_housing.md:38`) : CP2 (CP2.10) et MP2 (`projets/partie_2_california_validation/depart/data.py:28`) manquent ;
  - **Holmes et Verne** (`data/cards/holmes.md:17`, `data/cards/verne.md:17`) : CP1 (CP1.11) et MP1 (`projets/partie_1_detecteur_langue/depart/data.py:35` et `:47`) manquent ;
  - **Données synthétiques** : `data/cards/synthetic.md:15` donne `noisy_sine` « (ch. 22) » seulement, alors que 0B.35 l'utilise (`tools/chapters/build_ch00b.py:514`).
- **Correction.** Ajouter :
  - à mnist : « 0A (un tableau (N, 28, 28), boucle contre NumPy, `show_images` : 0A.54, 0A.55, 0A.59), 0B (ordre de grandeur d'un produit matriciel, 0B.30) » ;
  - à penguins : « 0B (deux manchots vus comme des vecteurs, 0B.8) », « CP1 (CP1.3, CP1.5) », « CP2 (CP2.5) » ;
  - à california : « CP2 (CP2.10, la fuite d'une validation croisée), MP2 (protocole d'évaluation honnête) » ;
  - à holmes et verne : « CP1 (CP1.11), MP1 (détecteur de langue) » ;
  - pour `noisy_sine` : « (0B.35, ch. 22) ».

**11. MINEUR** · `annexes/cheatsheets/numpy.md:223` · « ## Moindres carrés, pénalités et grilles de droites (ch. 9) »

- **Problème.** Chaque chapitre, du ch. 2 au ch. 9, a sa propre section. Sous ce titre du ch. 9 sont pourtant rangées les lignes du ch. 10 (lignes 239 à 242 : `np.where(z > 0, 1.0, -1.0)`, astuce du biais, `rng.permutation`, perceptron moyenné) et celles du ch. 11 (lignes 243 à 249 : argmax avec ex aequo, `rng.random() < epsilon`, `rng.beta`, tirage pondéré, regret, `np.rint`, ajustement d'un cercle). La colonne « Ch. » de ces lignes est juste.
- **Correction.** Insérer, avec leur en-tête de tableau `| Code | Effet | Ch. |` / `|---|---|---|` :
  - avant la ligne 239 : « ## Seuils, biais et ordre de parcours (ch. 10) » ;
  - avant la ligne 243 : « ## Bandits et tirages pondérés (ch. 11) ».

**12. MINEUR** · `chapitres/ch11_raisonnement/01_fiche.md:330` · « ## Pour aller plus loin », suivi de 6 puces (lignes 332 à 337 : Domingos ; Sutton et Barto ; Lattimore et Szepesvári ; Russo et al. ; Henderson ; Zach et coll.)

- **Problème.** Le §12, rubrique 10, demande « 3–5 ressources ». Les autres fiches en ont 5 au plus.
- **Correction.** Supprimer la puce Lattimore et Szepesvári (ligne 334) et ajouter à la fin de la puce Sutton et Barto : « ; pour la théorie complète, T. Lattimore et C. Szepesvári, [*Bandit Algorithms*](https://tor-lattimore.com/downloads/book/book.pdf), 2020 (en ligne) ».

**13. MINEUR** · `annexes/glossaire.md:56` · « | descente de gradient | gradient descent | méthode qui ajuste les poids par petits pas dans la direction qui fait baisser la loss | | »

- **Problème.** La colonne Ch. (« chapitre où il est introduit », `annexes/glossaire.md:5`) est vide. Or 0B définit un pas de descente de gradient (`chapitres/ch00b_maths/01_fiche.md:760` : « Un **pas de descente de gradient**, avec un learning rate $\eta$, s'écrit : »), et le ch. 5 l'implémente. Les entrées voisines « learning rate » et « loss » sont datées 0B (lignes 17 et 18).
- **Correction.** Mettre « 0B » dans la dernière cellule.

**14. MINEUR** · `docs/syllabus/data/ch0A.json:2363` (affiché dans `docs/SYLLABUS.md:490`) · « 🔨/📦 = 43 (cours de programmation … » et « Durée ≈ 19 h d'exercices »

- **Problème.** Le contrat compte 44 🔨/📦 et 1 348 min (22,5 h) d'exercices. La fiche (`chapitres/ch00a_python/01_fiche.md:8`) dit « exercices ≈ 22 h », et la même note dit plus loin que « 19 h de Python pour un débutant complet était optimiste ».
- **Correction.** Écrire « 🔨/📦 = 44 » et « Durée ≈ 22,5 h d'exercices (19 h avant la hausse des durées des 🔨/📦) », puis lancer `syllabus.py build`.

**15. MINEUR** · `tools/chapters/build_ch03.py:1698` · « rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|", »

- **Problème.** Le §12 (03, point 1) demande un tableau des exercices « (ID, type, ★, ⏱️, 🚀) ». Aucun des 13 notebooks n'a de colonne 🚀.
  - Le code du tableau est recopié dans chaque constructeur : `build_ch00a.py:3313`, `build_ch00b.py:1920`, `build_ch01.py:1243`, `build_ch02.py:2008`, `build_ch04.py:1435`, `build_ch05.py:1525`, `build_ch06.py:1303`, `build_ch07.py:2354`, `build_ch08.py:2121`, `build_ch09.py:2752`, `build_ch10.py:1750`, `build_ch11.py:2128`.
  - Sans effet aujourd'hui : aucun exercice publié n'a `gpu: true`. Le premier est 21.26 ; 38 exercices en ont un dans les ch. 21 à 28, B1 à B8 et le projet final.
- **Correction.** Une seule fonction `exercise_table(exercises, parts)` dans `chapter_kit.py`, avec une colonne 🚀 (« 🚀 » si `gpu`). L'utiliser dès le ch. 12, puis à la prochaine reconstruction des autres chapitres.

### SUGGESTION

**16. SUGGESTION** · `chapitres/ch00b_maths/flashcards.csv:31` · « Deux événements incompatibles, de probabilités non nulles, peuvent-ils être indépendants ? », presque identique à `chapitres/ch03_probabilites/flashcards.csv:8` · « Deux événements disjoints (de probabilités non nulles) peuvent-ils être indépendants ? »

- **Problème.** Trois paires posent la même question avec la même réponse dans deux paquets ; Anki les fait réviser deux fois :
  - la paire ci-dessus ;
  - `chapitres/ch00b_maths/flashcards.csv:18` « Pourquoi standardiser les features avant de calculer des distances ? » et `chapitres/ch01_introduction/flashcards.csv:15` « Pourquoi mettre les features à la même échelle avant de calculer des distances ? » ;
  - `chapitres/ch01_introduction/flashcards.csv:7` « Learning rate trop petit ? Trop grand ? » et `chapitres/ch05_courbes/flashcards.csv:14` « Learning rate trop grand, trop petit : que voit-on ? ».

  La dernière paire montre une autre dérive : `dlwb::ch01::learning_rate` contre `dlwb::ch05::learning-rate`. Sur les 320 tags, 51 utilisent `_` et les autres `-`, ce qui empêche de filtrer un concept d'un chapitre à l'autre.
- **Correction.**
  1. Faire avancer la carte la plus récente de chaque paire : ch. 3, un exemple sur un dé à vérifier avec \(P(A \cap B)\) ; ch. 1, l'effet sur les plus proches voisins des manchots ; ch. 5, la forme de la courbe de loss dans les trois cas. Sinon, supprimer la carte.
  2. Choisir `-` et faire respecter `^dlwb::ch[0-9a-z]+::[a-z0-9-]+$` par `export_flashcards.py`.

**17. SUGGESTION** · `chapitres/ch08_train_test/02_exercices.md:148` « ## ✏️ ∂ 📈 Papier-crayon », contre `chapitres/ch11_raisonnement/02_exercices.md:351` « ## 🗣️ 📈 ⚖️ 📄 Réflexion »

- **Problème.** Le §12 ne dit pas où vont 📈 et 🛠️ dans 02.
  - 📈 est dans « Papier-crayon » aux ch. 8 (8.8, ligne 252) et 9 (9.9, ligne 300), mais dans « Réflexion » au ch. 11 (11.10, ligne 364).
  - 🛠️ est toujours dans « Réflexion » : 0A.10 à 0A.12, 0B.32 et 7.9.
  - Les en-têtes de quiz de 02 n'ont pas d'étoiles, contrairement au modèle du §11, dans tous les chapitres (135 items). Exemple : `chapitres/ch08_train_test/02_exercices.md:17` « ### 8.Q1 — La boucle d'entraînement : prédire, comparer, corriger 🧠 ⏱️ 3 min ». C'est cohérent, mais écrit nulle part.
- **Correction.** Ajouter une ligne au §22 : « dans 02, 📈 va avec le papier-crayon et 🛠️ dans « Réflexion et outils » ; les en-têtes de quiz n'ont pas d'étoiles ». Déplacer 11.10 dans la section papier, ou l'inverse. Ce point complète P5 (intitulés des sections).

## Bilan

**Ce qui est solide**

- **Contrat.** Les 665 exercices concordent avec leur contrat (ID, types, ★, durées, parcours) dans 02, 03, 04, 05, 06 et le tableau de bord. Chaque exercice a ses 3 indices. 06 prévoit une place pour chaque sous-question. Les fichiers générés et `answers.json` sont à jour.
- **Notebooks.** Les constructeurs reproduisent exactement les notebooks publiés, sauf au ch. 2.
  - Notebooks d'exercices : sans sortie ni commande magique ; badge et cellule de setup conformes.
  - Notebooks de solutions : exécutés, sans erreur non voulue ni ❌ ou ⏳, largement sous le budget FAST_MODE.
- **Fiches.** Les 10 rubriques y sont, avec des ⏩ conformes au SYLLABUS.
- **Composition et annexes.** La composition du §11 est respectée ou justifiée. Les flashcards vont de 15 à 30 cartes par chapitre, sans série de « Qu'est-ce que ». Formulaire, auto-évaluation et tableau de bord sont complets pour 0A à 11, CP1 et CP2.
- **Checkpoints.** Examens de 117 min, barème sur 20 identique au sujet et au corrigé question par question, 10 % sur la partie antérieure, synthèse avec carte Mermaid et fiche d'une page.
- **Mini-projets.** Cahier des charges, kit de départ, solution, grille sur 20 et extensions.

**Les 3 risques principaux**

1. **Les conventions de l'indice 3** (constats 1 et 2). Six chapitres de plus que ceux déjà connus donnent les réponses des quiz, et les ch. 3 à 9 donnent le code complet des 🔨 : l'échelle d'aide est court-circuitée. Il faut trancher avant la partie III, où les 🔨 et les quiz vérifiés par `wb.check` se multiplient.
2. **Le contrat dérive par rapport au contenu publié** (constats 5, 6, 7, 10 et 14 : champ `check`, préfixes des rappels, titres, notes, data cards). `syllabus.py check` ne voit rien de tout cela. Or le contrat sert de référence à toutes les sessions suivantes, et chaque écart s'y recopie.
3. **Des remplacements globaux sans test sur le dépôt réel** (constats 3 et 4). Le tag des flashcards a été cassé par `e96506d`, le ch. 2 n'a jamais été reconstruit après un changement du kit, et une promesse du §22 ne peut pas être tenue par une simple reconstruction. Les passes P1 et P5 en cours feront beaucoup de remplacements de ce genre. Avant, il faut ajouter des tests sur le dépôt réel :
   - `export_flashcards --check` ;
   - « une reconstruction redonne les notebooks publiés » (le principe de `structure/compare_rebuild.py`) ;
   - « tout 🔨 avec `mylearn` a sa ligne d'en-tête ».

**Sorties brutes** (dans `audit2/structure/`) : `raw_check1_exercises.txt`, `raw_check1_titles_summary.txt`, `raw_check1b_subquestions.txt`, `raw_check1c_solutions.txt`, `raw_check1d_headers.txt`, `raw_check2_fiche.txt`, `raw_check3_order02.txt`, `raw_check4_notebooks.txt`, `raw_check4_rebuild_sync.txt`, `raw_check4b_checkfield.txt`, `raw_check5_composition.txt`, `raw_check6_hints.txt`, `raw_check6_hints_manual.txt`, `raw_check6b_hints_code_detail.txt`, `raw_check7_flashcards.txt`, `raw_check7b_flashcards_near.txt`, `raw_check8_annexes.txt`, `raw_check9_checkpoints.txt`, `raw_check9_notebooks.txt`, `raw_syllabus_check.txt`, `raw_export_flashcards_check.txt`, `raw_build_answers_check.txt` et `raw_pytest_*.txt`.

Notes de lecture :

- Dans `raw_check4b_checkfield.txt`, les lignes 2.23 et 6.21 sont des faux positifs : leurs vérifications se font par `verdict`, pas par `wb.check`.
- Dans `raw_check1d_headers.txt`, la fenêtre de 3 lignes du script signalait à tort des « mylearn » manquants ; le décompte juste figure au constat 4.
