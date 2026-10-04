# Audit P5, tour 2 : le parcours de l'apprenant de bout en bout (relecteur A11 « parcours »)

Relecteur « parcours » · 2026-10-04 · lecture seule : aucun fichier du dépôt modifié (`git -C /home/claude/workbookia status --short` vide à la fin) ; aucun processus laissé en cours.

## Méthode et résultats de la simulation

- **Copie** : `tar` du dépôt (sans `data/downloads/`) dans `scratchpad/s24/audit2/parcours/workbookia/`, `__pycache__` copiés supprimés ; venv `/home/claude/venv313` (Python 3.13.13, versions figées). Scripts conservés dans `parcours/scripts/` (`runnb.py` : exécution nbclient d'un notebook dans son dossier ; `tryfill.py` : remplissage de cellules ; `simulate_path.py` : apprenant d'un parcours), sorties dans `parcours/runs/` et `parcours/sim/`.
- **`start_chapter.py`** : `--init`, puis `0A`, `0B`, `1` à `11`, `CP1`, `CP2` : tout est copié au bon endroit, rien n'est écrasé (🔒 à chaque relance), les messages sont clairs ; un identifiant inexistant (`12`, `CP3`, `3.1`, `partie_1`) donne un message utile ; `ch03`, `cp1`, `0a` sont acceptés.
- **Notebooks exécutés depuis `mon_travail/` de la copie** (nbclient, FAST_MODE, sans `PYTHONPATH` ni `pip install -e .` : la cellule de setup retrouve seule le dépôt) : 0A (9 s, 225 ⏳, 0 erreur), ch. 3 (5 s, 130 ⏳), ch. 9 (5 s, 137 ⏳), examens CP1 et CP2, kits MP1 et MP2 : 0 erreur. Sans librairie (`mon_travail/mylearn` absent), le message « Ta librairie mylearn n'existe pas encore … lance d'abord : python tools/start_chapter.py 3 » est clair (en local).
- **Cellules remplies juste puis faux** : 0A.1 (7 valeurs), 0A.14 (f-string), 3.15 (`confusion_matrix` juste, puis transposée dans `mylearn/metrics.py`), 9.1 (7 valeurs) : toutes les bonnes réponses ✅ ; les messages d'erreur sont bons (« le tableau est transposé (.T) », « c'est la somme des carrés : la MSE en est la moyenne », virgule décimale détectée…), sauf deux cas en 0A (constat 10).
- **Parcours simulés** (`simulate_path.py` : librairie = fonctions de la référence écrites dans les exercices du parcours, squelettes pour les autres, tous chapitres jusqu'au chapitre visé ; notebook = solutions pour les exercices du parcours, TODO pour les autres) : rapide aux ch. 3, 8, 9 et 10, maths au ch. 9, et témoin « complet » au ch. 3. **Aucun exercice du parcours ne reste en ⏳ ou ❌** : les prérequis de code des chapitres sont bien dans les parcours. Le défaut est dans les mini-projets (constat 1).

## Constats

### MAJEUR

**1. MAJEUR · mini-projets communs à tous les parcours mais qui exigent des fonctions `mylearn` hors parcours**
- Fichiers : `projets/partie_1_detecteur_langue/depart/notebook.ipynb` cellule 22 (source `tools/chapters/build_mp1.py:464`) et cellule 24 (`build_mp1.py:540`) ; `projets/partie_2_california_validation/depart/mp2_california.ipynb` cellules 21 et 27 (`build_mp2.py:709`, `:872`) ; `projets/partie_2_california_validation/README.md:9` et `:74` ; `projets/partie_1_detecteur_langue/README.md:10`, `projets/partie_2_california_validation/README.md:10`.
- Extraits : MP1.6 « b) Écris `reliability_data(model, texts, labels)` … (`mylearn.metrics.calibration_curve`, 10 intervalles) et le score de Brier » et, dans la cellule de vérification fournie, `f"Brier {mylearn.metrics.brier_score(site_y, p_site):.4f}")` ; MP2 : `labels = np.asarray(mylearn.cluster.KMeans(n_clusters=k, n_init=1, random_state=ZONE_SEED).fit(geo_z).labels_)` ; « **Méthode imposée** | des modèles linéaires de ta librairie `mylearn` (moindres carrés, Ridge, Lasso) » ; grille : « moindres carrés, Ridge et Lasso (et toute la chaîne) donnent les prédictions de scikit-learn » ; prérequis : « si tu n'as pas écrit l'un de ces modules, celui de la référence est utilisé à sa place ».
- Problème : PARCOURS.md et le README disent que les mini-projets font partie des quatre parcours, mais `calibration_curve` et `brier_score` ne s'écrivent qu'en 3.28 (parcours M et C), `Lasso` et `soft_threshold` qu'en 9.23 (M et C), `KMeans` et `kmeans_plusplus` qu'en 7.26 et 7.25 (R et C). Le repli sur la référence ne joue que pour un **module absent** ; or `start_chapter.py` a copié `metrics.py`, `cluster.py` et `linear.py` avec ces fonctions en squelette. Vérifié dans la copie : `mylearn.cluster.KMeans(...).fit(X)`, `mylearn.linear.Lasso(alpha=0.1).fit(...)` et `mylearn.metrics.brier_score(...)` lèvent `NotImplementedError` (aucun repli). Conséquence : MP1.6 reste ⏳ pour le parcours rapide ; MP2.3 (Lasso) pour le rapide ; MP2.4 et MP2.5 (zones k-means) pour le parcours maths. `tools/syllabus.py:340-347` ne vérifie les prérequis de code que d'exercice à exercice, pas pour les mini-projets.
- Correction (contrat : parcours d'exercices publiés) : soit ajouter 3.28 au parcours rapide, 9.23 au parcours rapide, 7.25 et 7.26 au parcours maths (≈ +1 h 35 pour chacun des deux parcours) ; soit déclarer dans `cp1.json` et `cp2.json` un champ `requires` (fonctions `mylearn` utilisées) et faire vérifier par `syllabus.py check` qu'elles sont écrites dans chaque parcours ; dans tous les cas, remplacer la parenthèse des deux README par : « (si tu n'as pas écrit l'un de ces modules, celui de la référence est utilisé ; si le module existe mais qu'une fonction n'est pas écrite, écris-la d'abord : `calibration_curve` et `brier_score` en 3.28 pour le parcours rapide) » (MP2 : « `Lasso` en 9.23 pour le parcours rapide, `kmeans_plusplus` et `KMeans` en 7.25 et 7.26 pour le parcours maths »).

**2. MAJEUR · la copie du tableau de bord de l'apprenant est figée dès `--init`, chapitres futurs compris**
- Fichiers : `tools/syllabus.py:925-955` (`build_dashboard` : `for cid, ch in chapters.items():`, sans filtre sur les chapitres publiés) ; `suivi/tableau_de_bord.md:3` ; `tools/start_chapter.py:175-206` (`sync_trackers`).
- Extrait (`tableau_de_bord.md:3`) : « À chaque nouveau chapitre, `start_chapter.py` ajoute à la fin de ta copie les sections publiées depuis, sans toucher à tes cases déjà cochées. »
- Problème : le modèle contient déjà les 47 sections (mise en place, 39 chapitres, 6 checkpoints, projet final) ; la copie créée par `--init` les contient donc toutes, et `sync_trackers` n'ajoute que les sections **absentes** : il n'ajoutera jamais rien, et aucune correction ultérieure n'atteindra la copie. Mesuré avec git : entre la fin de 0A (commit `0e100ee`) et aujourd'hui, 27 lignes du modèle ont changé pour 0B à 9 (par exemple « 2.13 … ★ 15 min » → « ★★ 30 min », « 3.16 … ★★ 25 min » → « ★★★ 40 min », « 9.5 ✏️ Arrêt anticipé avec patience sur une suite de pertes » → « 9.5 ✏️ Early stopping avec patience sur une courbe de loss ») ; P1 et le constat 34 d'A1 changeront encore des dizaines de titres des ch. 12 à B8 avant leur publication. Le test `tests/infra/test_tools.py:333` montre l'intention (« Claude publishes ch 18 » ajoute une section), que le modèle réel contredit.
- Correction : `build_dashboard` n'écrit que les sections des chapitres publiés (dossier présent dans `chapitres/` sans `EN_COURS.md`, ou checkpoint présent), comme `auto_evaluation.md` ; et `sync_trackers` signale (sans rien modifier) une section de la copie qui diffère du modèle : « ℹ️ La section N de ton tableau de bord a changé (titres ou durées) : compare avec `suivi/tableau_de_bord.md`. » Une ligne au §22.

**3. MAJEUR · `git pull --rebase --autostash` corrompt en silence un fichier modifié hors de `mon_travail/`, et le remède documenté échoue**
- Fichiers : `templates/notebook_setup_cell.py:32-35` (repris dans les 36 notebooks) ; `00_setup/COLAB.md:80` ; `00_setup/INSTALL_LOCAL.md:176`.
- Extraits : `if pull.returncode != 0:` puis « ⚠️ git pull a échoué » ; COLAB.md : « copie ta modification ailleurs si tu y tiens, puis `!git -C /content/drive/MyDrive/workbookIA restore <fichier>` ».
- Problème : simulé avec git 2.43 (`parcours/gitsim/`) : quand le fichier local modifié a aussi été modifié en amont, `git pull --rebase --autostash` **sort avec le code 0**, affiche « Applying autostash resulted in conflicts » et laisse des marqueurs `<<<<<<<` dans le fichier (un `.ipynb` devient illisible) ; la cellule de setup n'affiche rien. Au pull suivant : code 128, « Pulling is not possible because you have unmerged files » ; le remède `git restore <fichier>` échoue alors (« error: path 'nb.ipynb' is unmerged ») ; il faut `git restore --source=HEAD --staged --worktree <fichier>`, puis récupérer le travail dans `git stash list`. Les déclencheurs sont réalistes : un notebook du dépôt ouvert depuis Drive est enregistré automatiquement par Colab (METHODE L50 « Ouvre `03_notebook.ipynb` dans Colab et fais « Exécuter tout » » ; fiche 0A L18 « ou ouvre le notebook dans Colab ») ; METHODE L63 et L68 font écrire dans `06_mes_reponses.md` et `suivi/` sans préciser `mon_travail/` (constat 6), fichiers que Claude modifie (P1 change les titres des `06_mes_reponses.md`).
- Correction (cellule de setup de tous les notebooks, d'où la sévérité) : après le pull, tester `"conflicts" in pull.stdout + pull.stderr` ou `git diff --name-only --diff-filter=U`, et afficher « ⚠️ Mise à jour faite, mais tes modifications de <fichiers> (hors de `mon_travail/`) entrent en conflit avec la nouvelle version ; elles sont gardées dans `git stash`. Pour reprendre la version du dépôt : `git restore --source=HEAD --staged --worktree <fichier>` ; voir 00_setup/COLAB.md. » ; corriger le remède de COLAB.md L80 et d'INSTALL_LOCAL L176 de la même façon (et `git stash show -p` pour retrouver sa modification).

**4. MAJEUR · durées sous-estimées des grosses implémentations (contrat : durées publiées)**
- Fichiers : `docs/syllabus/data/ch03.json`, `ch06.json`, `ch07.json`, `ch08.json`, `ch09.json`, `ch10.json`, `ch11.json` (repris par les en-têtes, le tableau de bord, SYLLABUS et PARCOURS).
- Échantillon (13 exercices de 0A à 11 ; « lignes » = code de la référence hors docstrings et commentaires ; médiane du workbook : 1,3 min par ligne de référence, et 1,9 min pour 0A.66, calibré sur un débutant) :

| Ex | Contrat | Lignes | Tests | Estimation débutant | Verdict |
|---|---|---|---|---|---|
| 0B.43 `matmul` | ★★ 25 min | 6 | 4 | 25 min | juste |
| 0A.66 `iterate_minibatches` | ★★★ 39 min | 21 | 5 | 45 min | juste |
| 1.16 boucle d'entraînement | ★★ 30 min | 4 fonctions | — | 40 min | léger |
| 2.22 bootstrap | ★★ 30 min | 40 | 9 | 40 min | léger |
| 5.18 `gradient_descent` | ★★ 30 min | 24 | 13 | 35 min | juste |
| 8.22 `clone`, `cross_val_score` | ★★★ 45 min | 30 | 18 | 50 min | juste |
| 3.16 cinq mesures binaires | ★★★ 40 min | 39 + aides | 26 | 60 min | sous-estimé |
| 11.21 `run_bandit` + `testbed_21` | ★★★ 40 min | 34 + banc | 12 | 55 min | sous-estimé |
| 10.21 `Perceptron` | ★★★ 45 min | 58 | 22 | 60 min | sous-estimé |
| 8.13 `train_test_split` stratifié | ★★ 30 min | 48 | 14 | 50 min | sous-estimé |
| 6.23 Huffman (3 fonctions) | ★★★ 45 min | 59 | 19 | 70 min | sous-estimé |
| 9.23 `soft_threshold`, `Lasso` | ★★★ 60 min | 54 | 25 | 90 min | sous-estimé |
| 7.26 `KMeans` complet | ★★★ 60 min | 58 | 22 | 120 min | très sous-estimé |

- Problème : les classes et fonctions « à la scikit-learn » validées contre un oracle exact (7.26 : `n_init`, trois initialisations, deux règles d'arrêt, `n_iter_` identique à scikit-learn, cinq méthodes) demandent à un débutant deux fois le temps prévu ; le reste de l'échantillon est juste. Autres candidats du même indicateur : 3.19 (★★ 20 min, 31 lignes, 13 mesures), 11.19 (★★ 20 min, deux bandits).
- Correction : 7.26 → ★★★★ 120 min ; 9.23 → 90 min ; 6.23 → 70 min ; 3.16 → 60 min ; 10.21 → 60 min ; 11.21 → 55 min ; 8.13 → ★★★ 50 min ; puis `syllabus.py build` et une ligne PROGRESS § Écarts. À recouper avec les temps réels du journal (Calibrage, P7) avant la partie III.

**5. MAJEUR (sans PyTorch : Mac Intel, cas prévu par le guide) · « PyTorch inutile avant le ch. 20 » est faux, et `pytest tests/` ne tourne plus du tout**
- Fichiers : `00_setup/check_env.py:34` (`"torch": "ch. 20"`) et `:104` (« PyTorch absent : inutile avant le ch. 20 ») ; `00_setup/INSTALL_LOCAL.md:43` (« pour les chapitres PyTorch (20 et suivants), tu travailleras sur Colab ») et `:113` (« **Mac Intel** : saute cette étape ») ; `tests/test_ch05_calculus.py:19` et `tests/test_ch06_info.py:25` (`import torch` au niveau du module) ; cellules d'outils des notebooks des ch. 5 (cellule 15, `tools/chapters/build_ch05.py:227`) et 10 (cellule 35, `build_ch10.py:314`) : `import torch`.
- Problème : vérifié dans la copie en rendant `torch` introuvable : `python -m pytest tests/` s'arrête net, « Interrupted: 2 errors during collection » (aucun test de `mylearn` ne tourne, dès 0A) ; le notebook du ch. 5 échoue dès sa cellule d'outils, puis 7 cellules en `NameError` (« Exécuter tout » s'arrête à la première) ; le ch. 10 importe aussi torch dans sa cellule d'outils. Le rapport de `wb.setup` affiche alors « torch ? » et « ⚠️ Versions différentes de celles du workbook : torch ? (attendu 2.11.0) », peu clair.
- Correction : dans les deux fichiers de tests, `torch = pytest.importorskip("torch")` dans les seuls tests qui s'en servent (comme `test_ch00a_utils.py:134` et `test_ch10_perceptron.py:277`) ; dans les cellules d'outils des ch. 5 et 10, `try: import torch` / `except ImportError: torch = None`, et un ⏳ « PyTorch absent : fais 5.21 (10.14 c, 10.15) sur Colab » dans les exercices concernés ; check_env et INSTALL_LOCAL : « PyTorch sert dès le ch. 5 (5.21) et le ch. 10 (10.14, 10.15), et aux tests des ch. 5 et 6 ; sans PyTorch, fais ces exercices sur Colab » ; `wb.setup` : « torch absent » au lieu de « torch ? ».

### MINEUR

**6. MINEUR · `docs/METHODE.md` (§5, guide de l'apprenant) : où écrire, et des fichiers qui n'existent pas**
- `docs/METHODE.md:68` « 10. **Journal et auto-évaluation** dans `suivi/`. » contredit `00_setup/COLAB.md:35` (« Les modèles de `suivi/` sont tenus par Claude : n'y écris pas »), le README L81 et le §22 ; `:63` « en écrivant tes réponses dans `06_mes_reponses.md` » (sans chemin : le fichier du dépôt est modifié par Claude) ; `:59` ne cite pas `06_mes_reponses.md` parmi les copies ; `:3` « Les prompts à copier-coller sont dans `02_PROMPTS.md`, la spécification dans `01_BIBLE_WORKBOOK.md` » et `:20` « `01_BIBLE_WORKBOOK.md` et `00_METHODE.md` (ces fichiers) » : aucun de ces fichiers n'est dans le dépôt (la spécification est `docs/BIBLE.md`, BIBLE L5).
- Correction : L68 « **Journal et auto-évaluation** dans `mon_travail/suivi/` (tes copies ; `suivi/` contient les modèles tenus par Claude). » ; L63 « … dans ta copie, `mon_travail/<chapitre>/06_mes_reponses.md`. » ; L59 « … copie le notebook, `06_mes_reponses.md` et les squelettes mylearn dans `mon_travail/` » ; L3 « … la spécification dans `docs/BIBLE.md` (la version d'origine, `01_BIBLE_WORKBOOK.md`, n'était qu'une pièce jointe de la session 1) ».

**7. MINEUR · « les 🔮 d'abord » contredit les prérequis de code et l'ordre des fiches**
- `README.md:78` « 4. Notebook : les 🔮 d'abord, à l'instinct. » ; `docs/METHODE.md:64` « **Notebook** : les 🔮 d'abord à l'instinct, puis le reste. »
- Problème : 20 des 29 🔮 de 0A à 11 ont un prérequis de code (3.18 ← 3.16, 5.12 ← 5.11, 6.25 ← 6.23 et 6.18, 9.13 ← 9.12, 9.19 ← 9.17, 11.22 ← 11.21…) : faits d'abord, ils restent en ⏳ ; et les fiches placent chaque 🔮 à un moment précis (ch. 5 : « la prédiction 🔮 5.12, **avant** de lire « Au-delà du livre (2) » » ; ch. 7 : « le 🔮 7.12 se fait **avant** de lire la section « Quand k-means échoue » »).
- Correction : « 4. Notebook, dans l'ordre conseillé par la fiche : à chaque 🔮, écris ta prédiction **avant** d'exécuter. `wb.check` te dit si ta réponse est juste sans la révéler. » (de même dans METHODE).

**8. MINEUR · deux règles des 15 minutes**
- `README.md:94` « bloqué 15 min → indice 1 ; encore 15 min → indice 2 » et `docs/METHODE.md:64` « encore 15 minutes, l'indice 2 », contre les 13 `chapitres/*/04_indices.md:3` « Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 ».
- Correction : une seule règle ; par exemple, dans README et METHODE : « bloqué 15 min → indice 1 ; encore 5 min → indice 2, puis l'indice 3 », comme les indices.

**9. MINEUR · la sortie de `pytest` en mode apprenant noie l'information utile**
- Fichiers : `pyproject.toml:45` (`addopts = "-ra"`) ; `tests/conftest.py:135` et `:142` (`f"{full} n'existe pas encore : lance python tools/start_chapter.py <chapitre>"`) ; `README.md:79` et `:107`, `chapitres/ch00a_python/01_fiche.md:150` et `:1892`, `tools/start_chapter.py:286` et `:293` (`python -m pytest tests/ -q`).
- Problème : mesuré dans la copie, juste après `--init` et `start_chapter 0A` : « 49 failed, 250 passed, 1283 skipped », 823 lignes, dont 651 lignes « SKIPPED [n] … n'existe pas encore : lance python tools/start_chapter.py <chapitre> » (une par test, `<chapitre>` jamais remplacé) et 250 tests d'infrastructure qui ne concernent pas l'apprenant ; avec tous les squelettes copiés : 1 329 échecs, 4 015 lignes, 80 s. La fiche 0A (L1892) promet une dernière ligne du type « 5 passed, 1 failed ».
- Correction : `addopts = "-rfE"` ; dans `conftest.py`, le chapitre réel (MANIFEST, comme `wb.impl` le fait déjà : « mylearn.linalg_basics (ch. 0B) n'existe pas encore : python tools/start_chapter.py 0B ») et une seule ligne par module absent (hook `pytest_terminal_summary`) ; conseiller la commande par chapitre (`python -m pytest tests/test_ch03_metrics.py -q`, déjà donnée par les notebooks) dans README L79, la fiche 0A L1892 et le message final de `start_chapter.py`, et dire que `tests/infra/` teste l'outillage.

**10. MINEUR · deux messages de `wb.check` en 0A qui n'aident pas, dont un qui enseigne la mauvaise règle**
- `src/wb/checker.py:794-796` : pour une réponse entière, une valeur décimale dont l'arrondi tombe juste reçoit « Presque : la réponse attendue est un nombre entier ; arrondis ton résultat à l'entier le plus proche (round). » Vérifié : 0A.1 a) `17 // 5` saisi `3.4` (la division `/`) → ce message ; idem 0A.1 d) (`21.67`, où `10 // 3` a été calculé `3,33`) et 0A.5 c) (`333 / 64` au lieu de `333 // 64`). L'apprenant apprend à arrondir là où `//` prend la partie entière par défaut (la fiche le dit, et 0A.1 c en fait une erreur classique). Les entrées `tools/chapters/build_ch00a.py:75`, `:78`, `:114` n'ont pas d'indication `fractional` (mécanisme employé à partir du ch. 6).
- 0A.14 b) (`build_ch00a.py:324`, `wb.record("0A.14b", card)` sans `mistakes`) : la fiche fautive « Gentoo (Biscoe): 5.08 kg, flipper 217.0 mm » reçoit « Si tu as choisi parmi les réponses proposées dans l'énoncé, ton choix n'est pas le bon … vérifie l'orthographe du terme demandé », hors sujet pour une f-string.
- Correction : `fractional="// garde le quotient entier, arrondi vers le bas : 17 // 5, c'est combien de fois 5 tient entièrement dans 17"` pour 0A.1 a et d (et `fractional="drop_last=True jette le lot incomplet : 333 // 64"` pour 0A.5 c) ; `wb.record("0A.14b", card, mistakes={"la masse s'écrit avec une seule décimale (:.1f)": "Gentoo (Biscoe): 5.08 kg, flipper 21.7 cm", "la nageoire est en cm : 217 mm = 21.7 cm": "Gentoo (Biscoe): 5.1 kg, flipper 217.0 cm"})` ; puis reconstruire 0A et `build_answers.py`.

**11. MINEUR · Colab : travailler dans sa copie n'est expliqué que dans `00_setup/COLAB.md` §2**
- Fichiers : en-têtes des 17 notebooks d'exercices, de checkpoint et de kit (« > Travaille dans **ta copie** (`mon_travail/ch03_probabilites/03_notebook.ipynb`, créée par `python tools/start_chapter.py 3`) : ce fichier-ci est mis à jour par Claude. », par exemple `tools/chapters/build_ch03.py:1712`, de même dans les 16 autres `build_*.py`) ; `README.md:75` ; `chapitres/ch00a_python/01_fiche.md:18` « Lance `python tools/start_chapter.py 0A` (ou ouvre le notebook dans Colab) ».
- Problème : sur Colab gratuit, il n'y a pas de terminal ; la fiche 0A propose même d'ouvrir « le notebook » au lieu de lancer `start_chapter`. Le badge de chaque notebook, y compris dans la copie, ouvre la version GitHub du dépôt : le travail n'est ni dans Drive ni dans `mon_travail/` (Claude ne peut pas le corriger avec P9), et il est perdu si l'on ne fait pas « Enregistrer une copie ». La cellule de setup, elle, fonctionne bien depuis la copie ouverte dans Drive (racine fixe `/content/drive/MyDrive/workbookIA`).
- Correction : ajouter à l'en-tête (dans le kit, une fois pour tous) : « Sur Colab : lance `!python /content/drive/MyDrive/workbookIA/tools/start_chapter.py 3` dans une cellule, puis ouvre ta copie depuis Google Drive (`workbookIA/mon_travail/…` → clic droit → *Ouvrir avec → Google Colaboratory*) ; le badge ci-dessus ouvre la version du dépôt, qui n'est pas enregistrée (00_setup/COLAB.md §2). » ; fiche 0A L18 : « Lance `python tools/start_chapter.py 0A` (sur Colab : 00_setup/COLAB.md §2) » ; README L75 : « … (sur Colab : voir 00_setup/COLAB.md §2) ».

**12. MINEUR · Colab : modifier un `.py` (mylearn) ou un `.md` (réponses, suivi) n'est expliqué nulle part, et l'indication donnée est inexacte**
- `00_setup/COLAB.md:35` « … ton tableau de bord, ton journal et ton auto-évaluation, que tu modifies directement dans Drive (ou dans Colab via *Fichier → Ouvrir*). »
- Problème : Drive ne sait pas modifier un `.md` sans application tierce, et *Fichier → Ouvrir un notebook* de Colab n'ouvre que des notebooks. Or l'apprenant doit modifier `mon_travail/mylearn/*.py` dès 0A.26 et `06_mes_reponses.md` dès 0A (0B.32 : en LaTeX). Ce qui marche sur Colab : le panneau **Fichiers** (icône 📁) → `drive/MyDrive/workbookIA/mon_travail/…` → double-clic : le fichier s'ouvre dans un éditeur, enregistré dans Drive.
- Correction : une section « Modifier tes fichiers (.py, .md) » dans COLAB.md (panneau Fichiers, double-clic, `Ctrl+S`, puis relancer la cellule de vérification qui recharge `mylearn`), et L35 : « … que tu modifies dans Colab : panneau **Fichiers** (📁) → `drive/MyDrive/workbookIA/mon_travail/suivi/` → double-clic sur le fichier. »

**13. MINEUR · `!` « jamais utilisé » selon la fiche 0A, mais demandé par COLAB.md**
- `chapitres/ch00a_python/01_fiche.md:161` « (le workbook n'utilise jamais les raccourcis `!` et `%`, qui ne fonctionnent pas partout) » ; `00_setup/COLAB.md:30` `!python /content/drive/MyDrive/workbookIA/tools/start_chapter.py 3`, `:41` `!cd /content/drive/MyDrive/workbookIA && python -m pytest tests/ -q`, `:80` `!git -C … status`, `:81` `!rm …/.git/index.lock`.
- Correction : nuancer la fiche : « les notebooks du workbook n'utilisent pas `!` et `%` (ils doivent tourner partout) ; sur Colab, une cellule `!commande` lance une commande du terminal, comme dans 00_setup/COLAB.md » (ou réécrire COLAB.md avec `subprocess.run`).

**14. MINEUR (sécurité) · le code de publication de COLAB.md §5 affiche le jeton GitHub en cas d'échec**
- `00_setup/COLAB.md:66` `git = lambda *args: subprocess.run(["git", "-C", repo, *args], check=True)` et `:72` `git("push", f"https://x-access-token:{token}@github.com/cemah2/workbookIA.git", "main")`, alors que L74 dit « N'affiche jamais le jeton ».
- Problème : si le push échoue (réseau, droits, push refusé), `CalledProcessError` affiche la commande complète ; vérifié : « Command '['git', …, 'push', 'https://x-access-token:ghp_SECRET123@github.com/cemah2/workbookIA.git', 'main']' returned non-zero exit status 128. » Colab enregistre cette sortie dans le notebook, que l'apprenant commite ensuite dans `mon_travail/`.
- Correction : `def git(*args):` / `result = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)` / `if result.returncode != 0: raise RuntimeError(f"git {args[0]} a échoué (code {result.returncode}) : {result.stderr.replace(token, '***')}") from None`.

**15. MINEUR · `--init` décrit comme ne créant que `mylearn`**
- `README.md:106` « (`--init` : crée seulement `mylearn`) » ; `tools/start_chapter.py:8` « # only create mon_travail/mylearn (base files) », `:302` (aide de `--init`) ; titre affiché `:260` « Initialisation de mylearn » ; `README.md:32` « python 00_setup/check_env.py        # tout doit être ✅ », lancé avant `--init`, qui donne un ⚠️ (« mon_travail/mylearn pas encore créé », comme le dit INSTALL_LOCAL L135).
- Problème : `--init` crée aussi `mon_travail/suivi/` (vérifié ; INSTALL_LOCAL L142 et COLAB.md L35 le disent).
- Correction : README « (`--init` : crée `mon_travail/mylearn` et tes fichiers de suivi `mon_travail/suivi`) » ; docstring et aide « create mon_travail/mylearn (base files) and mon_travail/suivi » ; titre « Initialisation de mylearn et du suivi » ; README L32 « # tout doit être ✅ (un ⚠️ normal : mylearn pas encore créé) ».

**16. MINEUR · `suivi/remediation.md:15` fait cocher une case qui n'existe pas**
- Extrait : « Coche l'exercice fait dans ta copie du tableau de bord (`mon_travail/suivi/tableau_de_bord.md`), section du chapitre concerné. »
- Problème : les exercices `R-18.1` n'ont pas de ligne dans le tableau de bord, et `start_chapter.py` ne modifie jamais une section existante.
- Correction : « Ajoute une ligne `- [x] R-18.1 Titre` dans la section du chapitre concerné de ta copie du tableau de bord (`mon_travail/suivi/tableau_de_bord.md`). »

**17. MINEUR · la mise en place fait écrire `mean` (avec sa `ValueError`) avant 0A, en doublon de 0A.26**
- `suivi/tableau_de_bord.md:30` « `python tools/start_chapter.py --init`, puis implémenter `mean` dans `mon_travail/mylearn/_example.py` jusqu'à 5 tests verts (⏱️ 15 min) » ; `00_setup/demo.ipynb` cellule 40 « Implémente ensuite `mean` … les 5 tests doivent passer » ; `:27` « Refaire demo.3 et demo.4 jusqu'à obtenir ✅ (⏱️ 5 min) ».
- Problème : c'est exactement 0A.26 (« C'est ton premier contact avec **mylearn** », 15 min, comptée deux fois), qui vient après les fonctions (0A.23) et les exceptions (0A.25) ; demo.3 et demo.4 demandent `groupby` et `nunique` de pandas (0A.32 à 0A.35). Pour le débutant du §1, la mise en place exige donc ce que 0A enseigne.
- Correction : tableau de bord (modèle, dans `syllabus.py`) : « `python tools/start_chapter.py --init` (⏱️ 2 min) ; `mean` s'écrit en 0A.26 » et « demo.3 et demo.4 : à refaire après la partie E de 0A (facultatif) » ; démo, cellule 40 : « Tu écriras `mean` en 0A.26 ».

**18. MINEUR · lecture de la fiche comptée 30 minutes, quelle que soit sa longueur**
- `tools/syllabus.py:60` `FICHE_MINUTES = 30         # the chapter sheet (01_fiche.md), read in full by every track` (et les `reading_minutes` des `chXX.json`, ≈ 4 min par page du livre + 20 à 30 min).
- Problème : les fiches des ch. 1 à 11 font de 6 800 à 11 000 mots hors code (ch. 9 : 11 000, ch. 11 : 10 400, ch. 7 : 9 600), soit 14 à 22 pages : 54 à 88 minutes au rythme du syllabus lui-même (4 min par page, PROGRESS § Calibrage). L'écart cumulé est d'environ 6,7 h pour les ch. 1 à 11, et il pèse surtout sur le parcours rapide, qui lit toute la fiche (« environ 1,8 h avec la fiche entière » au ch. 9, dont 30 min pour une fiche de 22 pages).
- Correction : calculer la lecture de la fiche à partir de sa longueur une fois écrite (par exemple mots / 125 par minute), puis `syllabus.py build` (totaux de SYLLABUS, PARCOURS, tableau de bord et en-têtes des fiches) ; ligne PROGRESS § Calibrage.

**19. MINEUR · import des flashcards dans Anki non expliqué**
- `docs/METHODE.md:67` « importe `flashcards.csv` dans Anki » ; `README.md:80` « importe les flashcards dans Anki » ; chaque `chapitres/*/flashcards.csv` commence par la ligne `front;back;tags`, sans directive Anki ; seul `tools/export_flashcards.py` (deck global, `exports/dlwb_anki.csv`) écrit `#separator:semicolon`, `#html:true`, `#tags column:3` et donne la marche à suivre (« Fichier > Importer, type de note « Basique » »).
- Problème : importé tel quel, un fichier de chapitre demande de régler à la main séparateur, HTML et colonne des tags, et sa ligne d'en-tête devient une carte ; le deck global, lui, mélange les chapitres pas encore étudiés.
- Correction : une option `--chapter 3` à `export_flashcards.py` (mêmes directives) et, dans README L80 et METHODE L67 : « `python tools/export_flashcards.py --chapter 3`, puis dans Anki *Fichier > Importer* `exports/…`, type de note « Basique » ».

### SUGGESTIONS

**20. SUGGESTION · un échafaudage qui ne décroît pas (BIBLE §12.8)** — Les énoncés 🔨/📦 s'allongent (moyenne 118 mots en 0A, de 198 à 274 mots aux ch. 5 à 11) et donnent souvent l'algorithme pas à pas : 7.26 (toute la boucle de Lloyd et ses arrêts), 8.22 (« copie chaque valeur avec `copy.deepcopy`, et renvoie `type(estimator)(**params)` »), 10.21 ; les cellules TODO de la partie II donnent encore signature, docstring et première ligne (11.21 : `rng = np.random.default_rng(seed)       # draw the means, the seed of each bandit and of each policy from it`). À partir de la partie III : séparer le **contrat** (les conventions que les tests vérifient, à garder dans l'énoncé) de la **méthode** (à placer dans les indices 1 et 2).

**21. SUGGESTION · tableau de bord** — Ajouter les lettres de parcours à chaque ligne (`· R M C`, comme le champ « Parcours » des énoncés) : aujourd'hui l'apprenant du parcours rapide doit croiser `docs/PARCOURS.md` ; ajouter une case « Synthèse (≈ 1 h 30) » aux sections CP1 et CP2, et les totaux d'exercices par partie dans « Progression globale » (seule la mise en place a « / 6 »).

**22. SUGGESTION · message final de `start_chapter.py`** (`tools/start_chapter.py:292-293`) — « 👉 Ouvre mon_travail/…/03_notebook.ipynb » passe avant la fiche, alors que toutes les fiches commencent par la lecture : « 👉 Commence par `chapitres/ch03_probabilites/01_fiche.md` (« Ordre conseillé »), puis ta copie `mon_travail/ch03_probabilites/03_notebook.ipynb` » ; pour les tests : `python -m pytest tests/test_ch03_metrics.py -q`.

**23. SUGGESTION · prévenir quand un notebook publié change après la copie** — Les commits d'audit du 2026-10-04 ont reconstruit les notebooks de 0B, 3, 5, 6, 9, 10 et 11, et P1 à P5 en modifieront d'autres ; la copie de l'apprenant ne reçoit rien et rien ne le lui dit. `start_chapter.py` pourrait enregistrer l'empreinte de la source dans les métadonnées de la copie et afficher « ℹ️ Le notebook du ch. 3 a été corrigé depuis ta copie : `--as 03_notebook_v2.ipynb` pour en avoir une nouvelle à côté. »

**24. SUGGESTION · messages pensés pour le terminal** — Sur Colab, « lance d'abord : python tools/start_chapter.py 3 » (`src/wb/impl.py:223-224`) et « 👉 Teste ta librairie : python -m pytest tests/ -q » ne se tapent pas tels quels ; si `IN_COLAB`, donner la forme `!python /content/drive/MyDrive/workbookIA/tools/start_chapter.py 3`.

**25. SUGGESTION · outils du README** (`README.md:108`) — `python tools/run_all_notebooks.py` est un outil de génération : il n'exécute jamais les copies de `mon_travail/` et, avec `--inplace`, modifierait des fichiers suivis (conflits, constat 3). Le retirer du tableau de l'apprenant ou préciser « (outil de Claude) ».

## Points de friction d'un débutant, par gravité

**Bloquants** (l'apprenant ne peut pas avancer seul)
1. Mini-projets MP1 (étape 6) et MP2 (étapes 3, 4 et 5) en ⏳ hors parcours complet et code : fonctions jamais demandées dans son parcours (constat 1).
2. Après une modification hors de `mon_travail/` (notebook du dépôt ouvert depuis Drive, `suivi/` ou `06_mes_reponses.md` du dépôt selon METHODE), un fichier corrompu sans alerte, puis plus aucune mise à jour, et un remède documenté qui échoue (constats 3 et 6).
3. Sans PyTorch (Mac Intel), `pytest tests/` interrompu dès 0A et notebooks des ch. 5 et 10 inutilisables, alors que la documentation annonce PyTorch inutile avant le ch. 20 (constat 5).

**Gênants** (temps perdu, confusion, découragement)
4. Colab : comment créer sa copie, l'ouvrir (le badge mène à la version du dépôt, non enregistrée) et modifier un `.py` ou un `.md` (constats 11, 12, 13).
5. Durées trop courtes pour les grosses classes (KMeans, Lasso, Huffman, Perceptron…) et pour la lecture des fiches : le débutant se croit lent (constats 4 et 18).
6. Tableau de bord figé à l'initialisation : titres, ★ et durées périmés, corrections jamais reçues (constat 2).
7. `python -m pytest tests/` : des centaines de lignes « SKIPPED … <chapitre> » et 250 tests d'infrastructure dès 0A (constat 9).
8. Consignes contradictoires : où écrire (METHODE), « les 🔮 d'abord », règle des 15 minutes (15 ou 5 minutes entre deux indices) (constats 6, 7, 8).
9. Jeton GitHub affiché, et enregistré dans le notebook, si le push depuis Colab échoue (constat 14).
10. Message « Presque … arrondis (round) » sur la division entière `//` en 0A (constat 10).

**Mineurs**
11. Mise en place : `mean` avant d'avoir appris les fonctions et les exceptions, et compté deux fois avec 0A.26 (constat 17).
12. Remédiation : cocher une case qui n'existe pas (constat 16).
13. Import des flashcards dans Anki non expliqué (constat 19).
14. `--init` mal décrit, et « tout doit être ✅ » alors qu'un ⚠️ est normal (constat 15).

## Bilan

**Ce qui est solide.** La mécanique de l'apprenant tient : `start_chapter.py` copie tout au bon endroit pour 0A à 11, CP1 et CP2, n'écrase jamais rien et explique chaque cas ; la cellule de setup, identique dans les 36 notebooks (seul le chargement de `mylearn` change), retrouve le dépôt depuis `mon_travail/` en local et ne dépend pas du dossier sur Colab ; les badges Colab pointent tous vers le bon chemin de `cemah2/workbookIA`, branche `main` ; les 13 notebooks, les examens et les kits s'exécutent depuis `mon_travail/` sans erreur ; les messages de `wb.check` et des tests sont presque toujours précis (transposition, somme au lieu de moyenne, virgule décimale) ; les parcours sont cohérents à l'intérieur des chapitres (aucun exercice d'un parcours bloqué par un exercice hors parcours aux ch. 3, 8, 9 et 10) ; les fiches, `06_mes_reponses.md`, les guides de lecture (⏩ conformes au SYLLABUS) et les README des checkpoints donnent les mêmes consignes d'un chapitre à l'autre ; l'auto-évaluation a une section par chapitre publié, alignée sur les objectifs des fiches.

**Les trois risques principaux.**
1. **Les mini-projets ne respectent pas la règle des parcours** : communs à tous, ils exigent des fonctions que le parcours rapide (3.28, 9.23) ou maths (7.25, 7.26) n'a jamais écrites, et rien ne le vérifie ; le risque grandira avec MP3 à MP6.
2. **Les fichiers de l'apprenant ne suivent pas les corrections** : tableau de bord figé à `--init` (chapitres futurs compris), copies de notebooks jamais signalées comme périmées, et conflits `git` silencieux dès qu'un fichier hors de `mon_travail/` a été touché ; les changements P1 à P5 en cours rendent ce risque immédiat.
3. **L'expérience Colab (plateforme recommandée) repose sur un seul document** : créer, ouvrir et modifier ses copies, lancer les tests, publier sur GitHub ne sont expliqués que dans COLAB.md, avec des inexactitudes (`Fichier → Ouvrir`, `!` interdit par la fiche 0A, jeton affiché en cas d'échec) ; un débutant qui suit le badge travaille dans un notebook qui n'est pas enregistré.
