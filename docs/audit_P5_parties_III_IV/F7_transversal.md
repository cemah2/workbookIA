# Correcteur F7 (transversal) : rapport

**Réponses enregistrées dont la valeur change : aucune.** Je n'ai modifié aucun script `tools/chapters/` et je n'ai réexécuté aucun notebook. Les changements de la référence ne changent aucune valeur :
- `dropout_forward` : `np.random.default_rng(g)` rend le même `Generator` ;
- Naïve Bayes et ensembles : seuls les chemins d'erreur changent.

## Constats

| Constat | Décision | Fichiers touchés |
|---|---|---|
| A10 1 (`rng` entier du dropout) | appliqué dans la référence : `rng = np.random.default_rng(rng)`. L'indice 3 de 20.16 était déjà corrigé par F5 ; l'énoncé de `build_ch20.py` relève de F5 | `solutions/mylearn_ref/nn/regularization.py` |
| A10 2 (`predict`/`score` de Naïve Bayes avant `fit`) | appliqué : `joint = self.predict_joint_log_proba(X)` d'abord, dans les deux classes, ce qui donne une `RuntimeError`. Le test facultatif (`"predict"` dans `check_unfitted`) est seulement proposé : l'énoncé de 13.28 dit « les autres méthodes passent par elle », sans promettre l'erreur | `solutions/mylearn_ref/naive_bayes.py` |
| A10 4 (message « the tree » dans les ensembles) | appliqué : `_check_X(X, model)` compare le nombre de colonnes à `n_features_in_` du premier membre. Le message nomme la classe (« X has 2 feature(s), but RandomForestClassifier was fitted with 3 »), vérifié pour les 5 ensembles. Pas de nouvel attribut public | `solutions/mylearn_ref/ensemble.py` |
| A10 3, 5 | reporté (R10) | — |
| A10 7 (`assert` d'oracle sans message) | appliqué : constante `ORACLE_BUG` (« bug in tests/test_ch20_regularization.py (not in your code): … ») sur les 11 `assert` qui comparent l'oracle à torch | `tests/test_ch20_regularization.py` |
| A10 8 (mutant `rng=None` → graine fixe) | appliqué : nouveau test `test_dropout_forward_without_rng_draws_a_new_mask` (deux masques différents, état global de NumPy intact). Il tue le mutant R12 (vérifié sur une copie) | `tests/test_ch20_regularization.py` |
| A10 9 (`init_mlp` qui recrée un générateur) | non retenu : un test exact demanderait un oracle du flux de tirage. La ligne du §22 proposée par A10 est dans « Pour une autre zone » | — |
| A10 10 (`None` lu comme « shape () ») | appliqué : « got None (did you forget the return?) » en tête de `as_array` (ch. 16) et de `assert_close` (ch. 19) | `tests/test_ch16_layers.py`, `tests/test_ch19_optim.py` |
| A10 11 (ordre des `except`) | appliqué : `except NotImplementedError` passe avant `except error` dans les 6 fichiers de la partie IV, avec le commentaire des ch. 12 à 14. Le `tests/_helpers.py` commun est proposé, pas appliqué (restructuration) | `tests/test_ch16_layers.py`, `test_ch17_activations.py`, `test_ch18_autograd_mini.py`, `test_ch18_backward.py`, `test_ch19_optim.py`, `test_ch20_regularization.py` |
| A10 12 (ids `<lambda>`) | appliqué : `ids=lambda value: value if isinstance(value, str) else "torch"` | `tests/test_ch16_layers.py` |
| A10 13 (import de `transformers` au chargement) | appliqué : `hugging_face_cosine()`, importé au premier usage et mis en cache (`lru_cache`). Seul le test cosinus le charge | `tests/test_ch19_optim.py` |
| A10 21 (`AttributeError: 'NoneType' …` dans `wb.attempt`) | appliqué : une ligne ❌ (« … comme fit qui doit rendre self … ») ; les autres `AttributeError` gardent leur trace. Docstring mise à jour, test ajouté | `src/wb/checker.py`, `tests/infra/test_wb_check.py` |
| A10 22 (`optimizer_curves` avec `n_updates` < 70) | appliqué : `ValueError` si `n_updates` n'est pas un entier ≥ 100. Le test essaie 100, 99, 20, 600.0 et True | `src/wb/synth.py`, `tests/infra/test_wb_synth.py` |
| A10 23 (noms qui donnent la réponse de 19.23 a) | appliqué : `warm`/`lr` deviennent `k`/`s`, avec des courbes identiques à l'octet (comparées avant et après pour 1 200 et 600). Les commentaires du test sont neutres (« curve B: flat at log 10 after its burst », « C: both losses high »…) | `src/wb/synth.py`, `tests/infra/test_wb_synth.py` |
| A10 24 (bruit de Hugging Face au premier téléchargement de CIFAR-10) | appliqué : contexte `_quiet_huggingface()`. Il coupe les barres de progression et met `datasets` et `huggingface_hub` en niveau « error », puis les restaure, et filtre les avertissements tqdm et « unauthenticated ». Il est précédé d'une ligne ℹ️ « Premier chargement de CIFAR-10 (≈ 170 Mo…) ». Test de restauration ajouté | `src/wb/datasets.py`, `tests/infra/test_wb_datasets.py` |
| A10 25 (délai des `run_*_tests`) | non retenu dans ma zone : `wb.run_pytest` a déjà `timeout=120`. Les helpers en cause sont dans `tools/chapters/` (« Pour une autre zone ») | — |
| A10 14 à 20 | hors zone (`tools/chapters/`) | — |
| A11 6 (`SyntaxError` dans un module mylearn) | appliqué : dans `mylearn_module`, toute erreur à l'import (hors module absent, qui reste un skip) appelle `pytest.exit` une fois, avec une ligne. Exemple : « ❌ mon_travail/mylearn/nn/layers.py ne s'importe pas : SyntaxError ligne 292 : … Corrige ce fichier, puis relance pytest. » Vérifié : 120 tests du ch. 16 avec une parenthèse oubliée donnent 2 lignes au lieu de 5 400. Tests d'infrastructure paramétrés : `SyntaxError` et `NameError` à l'import | `tests/conftest.py`, `tests/infra/test_impl_modes.py` |
| A11 10 / A3 10 (ordre de `auto_evaluation.md`) | appliqué : le bloc CP2 passe avant la section 12. Nouveau test : l'ordre des `wb:section` des modèles `auto_evaluation.md` et `tableau_de_bord.md` suit `syllabus.ORDER` (il échoue sur l'ancien fichier, vérifié) | `suivi/auto_evaluation.md`, `tests/infra/test_syllabus.py` |
| A11 11 (METHODE et README : temps et ordre de lecture de la fiche) | appliqué, d'après les fiches réelles : 55 à 123 min à 125 mots/min, et un « Ordre conseillé » bloc par bloc dans les 22 fiches. METHODE §5, étapes 2 à 5 : lire « Comment utiliser ce chapitre », les objectifs et « L'essentiel en 10 lignes » (10 min), puis suivre l'ordre conseillé bloc par bloc (livre → fiche → 🧠 🔁 → ✏️ ∂). Le README, étapes 2 et 3, dit la même chose | `docs/METHODE.md`, `README.md` |
| A11 16 (temps réel par exercice) | appliqué au journal : « exercices avec leur temps réel », avec un exemple. La légende du tableau de bord est dans « Pour une autre zone » | `suivi/journal.md` |
| A11 17 (`start_chapter.py MP4`) | appliqué : `OtherCommand` (sous-classe de `ValueError`). `main` affiche « ℹ️ Le kit du mini-projet MP4 est copié avec son checkpoint : python tools/start_chapter.py CP4 » ; test ajouté | `tools/start_chapter.py`, `tests/infra/test_tools.py` |
| A11 18 (les docstrings disent `pytest tests/ -q`) | appliqué côté README (« Ta librairie ») : préférer la commande du chapitre ou `-m mylearn`. L'encadré « Mode d'emploi mylearn » relève des scripts de chapitre | `README.md` |
| A11 3 (variante du test des kits avec les squelettes) | proposé : aucun test « Run all » des kits n'existe dans `tests/infra/`, et le test dépend de la correction de `build_mp4.py` (F6) | — |
| A11 15, A3 11 (`PROGRESS.md`) | hors zone (« Pour une autre zone ») | — |
| A1 16 (règle de la glose en tête du glossaire) | appliqué (texte du §5 réécrit par P2) | `annexes/glossaire.md` |
| A1 17 (colonne « Ch. » du glossaire) | appliqué : passe avant/arrière → 16 (la glose « passe avant » est déjà dans la fiche 16) ; gradient évanescent → 0B ; LayerNorm → 20 (le terme lui-même n'a pas changé : R5). « gradient explosif », absent de tout fichier, n'est plus un terme retenu : la définition dit « font exploser le gradient (*exploding gradient*) » | `annexes/glossaire.md` |
| A1 32 (`test_terminology.py`) | appliqué en partie, vert sur tout le dépôt (15 tests). `SOURCES` gagne `projets/*/*/*.md` et `projets/*/*/*/*.py` (`projets/*/*/*.py` y était déjà). `FORBIDDEN` gagne early stopping (`arrêts? anticipés?/précoces?`, avec le sens courant « l'arrêt précoce les manque » du pré-élagage au ch. 13), weight decay (`décroissance des poids`), warmup (`échauffement`, avec le sens courant « lancers d'échauffement » au ch. 4) et layer norm (`normalisation par couche`). J'ai corrigé un vrai écart, « arrêt précoce » → « early stopping » dans `erreurs_frequentes.md`. Non ajoutés : voir « Pour une autre zone » | `tests/infra/test_terminology.py`, `annexes/erreurs_frequentes.md` |
| A1 8 à 11, 14, 15, 28 (choix de termes) | reporté (R5) : aucune forme changée | — |
| A3 12 / A2 16 (Penguins dans CP3 et CP4) | appliqué : « CP3 (examen : … CP3.2 ; … CP3.4), CP4 (examen : standardisation sans fuite d'un manchot de test, CP4.12) », vérifié dans `cp3.json` et `cp4.json` | `data/cards/penguins.md` |
| A3 13 (titre « Ch. 15 » du formulaire) | appliqué : « ### Ch. 15 · scikit-learn » | `annexes/formulaire.md` |
| A3 14 (colonne « Ch. » vide) | appliqué : pooling, padding, stride → 20 ; framework → 15 (première fiche qui l'emploie) ; kernel reste vide (ch. 21) | `annexes/glossaire.md` |
| A3 15 (`exercise_table` avec la colonne 🚀) | hors zone (`tools/chapters/chapter_kit.py`) | — |
| A3 20 (XGBoost et LightGBM absents des cheatsheets) | appliqué : deux lignes après `HistGradientBoostingRegressor`. Vérifié avec xgboost 3.4.1 et lightgbm 4.6.0 : `fit` refuse `early_stopping_rounds` ; `best_iteration` de XGBoost compte à partir de 0 (51 = argmin) ; `best_iteration_` de LightGBM à partir de 1 (59 = argmin 58 + 1) ; le `predict` de XGBoost s'arrête au meilleur | `annexes/cheatsheets/sklearn.md` |
| A4 1 (côté glossaire) | appliqué : la ligne « un-contre-un » du ch. 13 est supprimée ; « c'est la stratégie de `SVC` (ch. 13) » est ajouté à celle du ch. 7 | `annexes/glossaire.md` |
| A4 2 (colonne constante standardisée) | appliqué après recalcul : ±1 pour n = 3, 7, 13 (0/0 pour n = 10) ; (0,11 − 0,1)/1,39e-17 ≈ 7,2 × 10¹⁴ | `annexes/erreurs_frequentes.md` |
| A4 4 (lissage additif) | appliqué (texte du relecteur) | `annexes/glossaire.md` |
| A5 4 (`inverse_transform` « quand il existe ») | appliqué | `annexes/glossaire.md` |
| A6 7 (forme de la couche dense avec un biais en colonne) | appliqué après recalcul : (5,5) pour un exemple, (5,5) pour n = n_out, erreur pour n = 3 | `annexes/erreurs_frequentes.md` |
| A6 8 (`nn.Threshold`, « détails au ch. 17 ») | sans objet : la fiche 17 (l. 116) parle maintenant de `torch.nn.Threshold` | — |
| A6 9 (SwiGLU, échantillonneur-bloqueur, fonction de transfert) | appliqué : trois lignes, chapitres 16-17, 16 et 17 | `annexes/glossaire.md` |
| A6 10 (affine par morceaux) | appliqué | `annexes/glossaire.md` |
| A7 7 (`loss_curves` dans la data card) | appliqué : ligne ajoutée au tableau. La phrase du ch. 18 ne liste plus les six cas, ce qui supprime une fuite de la réponse de 18.22. `n_updates ≥ 100` est noté pour `optimizer_curves` | `data/cards/synthetic.md` |
| A8 4 (`dim`, pas `axis`) | appliqué. Vérifié avec torch 2.11 : `axis` **et** `keepdims` sont acceptés tous les deux ; la correction proposée (« au lieu du `keepdims` de NumPy ») aurait été fausse. Texte retenu : « les paramètres s'appellent `dim` et `keepdim` (PyTorch accepte aussi `axis` et `keepdims`…) » | `annexes/cheatsheets/pytorch.md` |
| A8 6 (côté data card) | appliqué : 256 × 256 puis recadrage au centre en 224 × 224 | `data/cards/cifar10.md` |
| A8 7 (20.28, un MLP et pas un CNN) | appliqué : « MLP 784-256-256-10 avec batchnorm et dropout » (vérifié dans `build_ch20.py`) | `data/cards/fashion_mnist.md` |
| A8 8 (20.23 : 12 000 / 50 000) | appliqué (vérifié : `wb.by_mode(fast=12_000, full=50_000)`, validation sur les 10 000 dernières) | `data/cards/mnist.md` |
| A8 10 ($\frac{n}{n-1}$ → $\frac{m}{m-1}$) | appliqué | `annexes/formulaire.md` |
| A8 16 (note « complété à partir du ch. 20 ») | appliqué (supprimée) | `annexes/erreurs_frequentes.md` |
| A8 20 (`BatchNorm1d` et un mini-batch d'un seul exemple) | appliqué après vérification : `BatchNorm2d(3)` passe sur (1, 3, 2, 2), échoue sur (1, 3, 1, 1) | `annexes/cheatsheets/pytorch.md`, `annexes/erreurs_frequentes.md` |
| A9 10 (`>` contre `>=` dans le formulaire) | appliqué au formulaire : « $p \ge t^*$ (à l'égalité, les deux décisions coûtent autant) ». La fiche du ch. 7 et la synthèse de CP3 sont hors zone | `annexes/formulaire.md` |
| A2 9 (dépôt du code du volume 1) | appliqué. `git ls-remote` sur `blueberrymusic/DeepLearningBookCode-Volume1` échoue comme pour un dépôt inexistant (2026-10-18). Une copie publique existe, `bssrdf/DeepLearningBookCode-Volume1` (vérifiée : historique d'Andrew Glassner, licence MIT, dossiers Chapter01 à Chapter19). Le README et les ressources renvoient à cette copie et aux ressources de l'auteur | `README.md`, `annexes/ressources.md` |

## Pour une autre zone

1. **`suivi/PROGRESS.md`** (coordinateur) :
   - A3 11, ligne 142 : « 52 exercices (dont 23 dans le notebook, parties A à D) » ;
   - A11 15, tableau « À valider sur Colab » : une ligne par notebook (ch. 14, 15, 16, 18, 19, 20, CP3, CP4, MP3, MP4). Pour le ch. 20 : « tests de `mylearn.nn.regularization` lancés par `subprocess`, « Tout exécuter » avec le GPU T4 (`Device : cuda` ; 20.13, 20.23, 20.28, 20.30 et 20.31 sur `cuda`, réponses 20.23 a et b acceptées), téléchargement de MobileNet V3 Small (20.26) et repli ⏳ sans réseau, temps du défi 20.31 » ; ajouter aussi « premier téléchargement de CIFAR-10 sans barres ni avertissements de Hugging Face (`wb.datasets._quiet_huggingface`) ».
2. **`suivi/tableau_de_bord.md`** (A11 16) : ajouter dans l'en-tête, après « Coche les cases au fur et à mesure (`- [x]`). », la phrase « Note ton temps réel au bout de la ligne quand il s'écarte de plus de moitié du temps annoncé (ex. `- [x] 13.24 … 100 min → 2 h 40`). ». L'en-tête est conservé par `syllabus.py build`, qui ne réécrit que les sections.
3. **`docs/BIBLE.md` §22** :
   - « 2026-10-18 | Audit P5 III-IV, outillage : une erreur à l'import d'un module `mylearn` (`SyntaxError`…) arrête `pytest` sur une ligne (fichier, erreur, ligne) ; `wb.attempt` traite aussi `AttributeError: 'NoneType' object has no attribute …` en une ligne ❌ ; `wb.synth.optimizer_curves` exige `n_updates ≥ 100` ; `start_chapter.py MPn` renvoie vers `CPn` ; le premier téléchargement de CIFAR-10 se fait sans barres ni avertissements de Hugging Face. | Constats A10 21-24, A11 6 et 17. » ;
   - « 2026-10-18 | Référence `mylearn` alignée sur ses conventions : `dropout_forward` accepte un `rng` entier ; `predict` et `score` de Naïve Bayes lèvent `RuntimeError` avant `fit` ; les ensembles nomment leur classe quand le nombre de colonnes diffère. `init_mlp` : le partage du générateur entre les couches n'est pas vérifié au-delà de l'ordre des couches. | A10 1, 2, 4, 9. » ;
   - « 2026-10-18 | Le dépôt officiel du code du volume 1 n'est plus en ligne (vérifié le 2026-10-18) : README et ressources renvoient à la copie `bssrdf/DeepLearningBookCode-Volume1`, comme pour le volume 2. | A2 9. » ;
   - « 2026-10-18 | `test_terminology.py` vérifie aussi early stopping, weight decay, warmup et layer norm (formes françaises), et lit les data cards, model cards et `minifw/` des mini-projets. | A1 32. »
4. **`chapitres/ch07_classification/01_fiche.md:105`** (A9 10) : « positif si $p \ge t^*$ (à l'égalité, les deux décisions coûtent autant) ». Dans le sujet et la synthèse de CP3 (F6) : $C_{FP}$, $C_{FN}$ en majuscules.
5. **`tests/infra/test_terminology.py`, formes non ajoutées** (A1 32). Elles échouent aujourd'hui ou relèvent de R4/R5 :
   - passe avant/arrière, `\b(le|du|au) (forward|backward)\b` : 8 occurrences, dont le titre de 18.Q11 (R4) dans `ch18/02_exercices.md:105`, `04_indices.md:191`, `05_solutions.md:52`, `06_mes_reponses.md:85`, `docs/SYLLABUS.md:3062`, `ch18.json:149, 511` et `build_ch18.py:223` ;
   - couche dense contre `couches? pleines?` : 9 occurrences, R5 ;
   - `LayerNorm` hors code : R5.
   Une fois R4 et R5 tranchés, ajouter `"passe avant, passe arrière": (r"\b(?:le|du|au) (?:forward|backward)\b", [])` et, selon R5, la forme écartée de couche dense ou pleine.
6. **`tools/chapters/`** :
   - A10 25 : `timeout=300` dans les `run_*_tests`, plus `except subprocess.TimeoutExpired: print("❌ les tests n'ont pas fini en 5 min : une boucle de ton code ne s'arrête-t-elle pas ?")` ;
   - A3 15 : `exercise_table` dans `chapter_kit.py` avec la colonne 🚀 ;
   - A11 18 : la phrase sur `pytest tests/ -q` dans l'encadré « Mode d'emploi mylearn » ;
   - A10 1 : la parenthèse « (`np.random.default_rng(rng)` accepte `None`, un entier ou un `Generator`) » dans l'énoncé de 20.16, `build_ch20.py` (F5).
7. **Pour le coordinateur** : les deux échecs du `pytest --impl=ref` complet viennent de `chapitres/ch14_ensembles/05_solutions.ipynb` (zone F2). La cellule 132 contient un `KeyboardInterrupt`, et la cellule 6, taguée « answer », n'a pas de `WB_ANSWER`. Ce notebook était en cours de réexécution par un autre correcteur.

## Scripts reconstruits et notebooks réexécutés

Aucun : ma zone ne contient pas de script `tools/chapters/` ni de notebook.

Vérifications :
- `python -m pytest -q tests/infra` : 299 réussis, 4 ignorés, 1 échec (`test_answers_json_is_valid_and_up_to_date`, dû au notebook du ch. 14 de F2), 2 min 50 s ;
- `python -m pytest -q --impl=ref` complet : 3 409 réussis, 2 échecs, les deux dus à ce même notebook (`test_answers_json_is_valid_and_up_to_date`, `test_executed_notebooks_have_no_errors`), 4 min 07 s ;
- les 8 fichiers de tests de chapitre que j'ai touchés : 1 174 réussis, 19 s ;
- `python -m pytest -q --impl=stubs tests/test_ch20_regularization.py` : 103 échecs, comme attendu ;
- ch. 16 et 19 en mode stubs : 327 échecs, comme attendu ;
- `tests/infra/test_terminology.py` : 15 réussis.
