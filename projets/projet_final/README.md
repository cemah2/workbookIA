# Projet final — Un projet de bout en bout sur ton propre dataset

> La dernière brique du workbook, environ 41 h (35 à 60 h selon le dataset et l'accès à un GPU de Colab), à étaler sur **4 à 6 semaines**. Tu choisis un problème réel et ses données, tu fixes un protocole d'évaluation **avant** toute expérience, tu compares honnêtement une baseline classique et un modèle deep learning adapté à tes données, tu évalues **une seule fois** sur un test scellé, puis tu livres un dépôt GitHub professionnel et tu le défends comme en entretien. Il n'y a pas de solution de référence : c'est ton projet. Il y a une grille, des indices et un exemple de calibrage.

| | |
|---|---|
| **Problème** | le tien : une décision réelle, des utilisateurs, une cible claire (cadré en PF.1, préparé par 🗣️ 29.4) |
| **Données** | un dataset public choisi par toi, sous une licence compatible avec un dépôt public, documenté par une data card (🛠️ 29.15) |
| **Méthode imposée** | un protocole figé par un tag git avant les expériences ; une baseline naïve, une baseline classique et un modèle deep learning ; un test scellé ouvert une fois ; un paquet Python testé, avec intégration continue et une démonstration |
| **Prérequis** | les chapitres 2, 3, 8, 12 à 15, 19 à 24 et 29, les bonus B1 à B4, B6 à B8 (selon tes données : la colonne « Chapitres mobilisés » de chaque étape) |
| **Évaluation** | sur 20, avec la grille [`GRILLE.md`](GRILLE.md) : auto-évaluation, puis correction par Claude (prompt P9) |
| **Livrables** | ton dépôt `mon_travail/projet_final/`, poussé sur GitHub ; une soutenance de 10 minutes ([`SOUTENANCE.md`](SOUTENANCE.md)) |

**Par où commencer** : lis [Choisir ton dataset](#dataset) et [La règle du test scellé](#test-scelle), installe le kit dans ton dépôt ([Démarrer](#demarrer)), puis fais PF.1. **Ce que tu rends** : ton dépôt, avec le livrable de chaque étape (la ligne « Livrable » de chacune des [13 étapes](#etapes)), et ta soutenance. **Comment tu es noté** : [la grille](#grille), critère par critère, avec ses pénalités et ses seuils.

**Sommaire** : [Le contexte](#contexte) · [Choisir ton dataset](#dataset) · [Le test scellé](#test-scelle) · [Le budget](#budget) · [Les jalons](#jalons) · [Démarrer](#demarrer) · [Le kit de départ](#kit) · [Les 13 étapes](#etapes) · [Le parcours rapide](#parcours-rapide) · [La grille](#grille) · [La correction](#correction) · [La soutenance](#soutenance) · [L'exemple de calibrage](#exemple)

<a id="contexte"></a>

## Le contexte

Les mini-projets des parties I à VI t'ont donné un problème, ses données et son protocole. Le projet final te les retire : c'est le travail réel d'un data scientist ou d'un ML engineer, et c'est ce que tu montreras en entretien. Un recruteur qui ouvre ton dépôt y passe deux minutes avant de décider s'il lit la suite. Il cherche, dans cet ordre :

1. **Un README qui dit en 30 secondes** ce que fait le modèle, à quel point il marche (un chiffre, son intervalle, la baseline) et comment le lancer.
2. **De l'honnêteté** : une baseline naïve et une baseline classique, un test qui n'a servi qu'une fois, des intervalles, des limites écrites, un critère de succès fixé d'avance, même s'il n'est pas atteint. Un résultat modeste et vrai vaut plus qu'un 99 % qui ne tient pas à la première question (💼 29.E3).
3. **Des données comprises** : une data card, une licence vérifiée, un audit qui a trouvé quelque chose.
4. **Du code qu'il pourrait reprendre** : un paquet installable, des tests qui attrapent de vraies erreurs, une intégration continue verte, une commande pour tout reproduire, une démonstration.
5. **Quelqu'un qui sait l'expliquer** : la soutenance, et les questions qui suivent.

Le score brut compte peu. Ce qui compte, c'est la démarche, et qu'on puisse la vérifier dans le dépôt.

<a id="dataset"></a>

## Choisir ton dataset

Tu l'as préparé au ch. 29 : le pitch de cinq lignes (🗣️ 29.4) et la data card avec son téléchargement vérifié (🛠️ 29.15). Si tu ne l'as pas fait, fais-le d'abord : PF.1 et PF.2 en partent. La fiche du ch. 29 (§29.1, « Au-delà du livre : auditer un dataset », « documenter et télécharger ») et ⚖️ 29.5 donnent la liste de vérifications.

**Les règles**

- **Licence** compatible avec un dépôt public : CC0, CC BY, CC BY-SA, MIT, Apache 2.0, ODbL… Lis la licence sur la page **officielle** du dataset, pas sur un miroir (un dataset republié sur Kaggle sans sa licence d'origine n'en a pas pour autant une nouvelle). « Recherche non commerciale seulement » se respecte : tu peux publier ton code et tes résultats, pas les données.
- **Taille** : au moins quelques milliers d'exemples (assez pour un test de quelques centaines, donc un intervalle qui dit quelque chose : 💼 2.E3), et un mode rapide (`FAST_MODE`) qui entraîne et évalue tout en **moins de 10 minutes sur CPU**, sur un sous-échantillon si besoin.
- **Une cible claire** : une classe, une valeur, une prévision, une récompense ; mesurée, pas devinée. Un problème sans cible (du clustering seul, de la génération sans mesure) se transforme en problème mesurable, ou se choisit autrement.
- **Une version figée** : une révision du Hugging Face Hub (le `sha` d'un commit), une version Kaggle ou OpenML, une archive datée ; et une somme SHA-256 par fichier.

**Les interdits** : des données personnelles non anonymisées (noms, visages identifiables, adresses, dossiers médicaux, messages privés) ; un dataset **retiré** par ses auteurs ou à licence incompatible (MS-Celeb-1M, *80 Million Tiny Images*, Boston Housing hors d'une discussion de son retrait : 29.5, 15.7) ; un fil rouge du workbook tel quel (Penguins, California, MNIST, Fashion-MNIST, CIFAR-10, Holmes, Verne, les taches solaires, Adult) : tu as déjà tout fait dessus, le recruteur aussi ; ni le dataset de l'exemple de calibrage (Bike Sharing, [`exemple/`](exemple/)), dont le cadrage et le protocole sont déjà écrits.

**Les variantes acceptées**

| Données | Baseline classique (PF.5) | Modèle deep learning (PF.6) | Le piège du découpage | Chapitres |
|---|---|---|---|---|
| **Images** | linear probe ou k plus proches voisins sur des embeddings figés | transfer learning : fine-tuning partiel puis complet d'un CNN ou d'un ViT pré-entraîné | plusieurs photos d'un même objet, d'un même patient, d'une même rafale : découper **par groupes** | 21, 23, 24, B1, B6 |
| **Texte** | TF-IDF + régression logistique (ou Naïve Bayes) | un encodeur pré-entraîné fine-tuné, ou un petit LLM adapté par LoRA | des textes dupliqués ou traduits, un même auteur des deux côtés | 13, B2, B3, B4 |
| **Séries temporelles** | persistance (et persistance saisonnière), gradient boosting sur des retards | LSTM, GRU ou petit Transformer sur des fenêtres | jamais de découpage aléatoire : le test vient **après** l'entraînement, et les fenêtres ne se chevauchent pas d'un découpage à l'autre | 22, 24 |
| **Tabulaire** | gradient boosting (HistGradientBoosting, XGBoost, LightGBM) : l'état de l'art du tabulaire | un réseau tabulaire (MLP avec embeddings des variables catégorielles) ; il a le droit de perdre, et tu dis pourquoi | plusieurs lignes d'une même entité ; une colonne qui trahit la cible | 12, 14, 15 |
| **Apprentissage par renforcement** | la politique aléatoire et une règle écrite à la main | DQN ou PPO (B8) sur un environnement gymnasium **autre que CartPole** | le « test scellé » est un jeu de graines d'évaluation tenu à l'écart, et une physique modifiée pour la robustesse | 26, B8 |

Un projet multimodal ou génératif est possible s'il reste mesurable (une métrique, une baseline, un test) ; parles-en dans le cadrage.

<a id="test-scelle"></a>

## La règle du test scellé

Le test scellé est la seule mesure de ton projet qui ne soit pas optimiste : tous les choix (features, architecture, hyperparamètres, seuil, epoch d'arrêt) se font sans lui. Il obéit à quatre règles.

1. **Un découpage figé et haché.** Le découpage (`data/splits.json`, les identifiants de chaque exemple en entraînement, validation et test) s'écrit **une fois** (`projet split`), se commite, et son SHA-256 se recopie dans `docs/protocole.md`. Toutes les expériences le relisent ; `evaluate.open_sealed_test` refuse d'ouvrir le test si le fichier a changé.
2. **Le protocole tagué dans git avant les expériences.** `docs/protocole.md` (découpage, métriques, baselines, critère de succès, sous-groupes, budget) se commite puis se tague : `git tag protocole && git push --tags`. Le tag date le protocole : sa date doit précéder la première ligne de `experiments/runs.jsonl`, et le commit de chaque run (le champ `environment.git_commit` du journal) doit descendre du commit tagué (`git merge-base --is-ancestor protocole <commit>`), ce qu'un tag déplacé après coup ne peut pas simuler.
3. **Une seule évaluation finale.** Le test s'ouvre une fois, à la fin, pour le modèle retenu et les deux baselines, dans `notebooks/03_evaluation.ipynb` (`OPEN_TEST = True`) ; tant que `OPEN_TEST = False`, ce notebook répète toute l'évaluation sur la validation, sans rien écrire, pour que l'unique passe sur le test ne bute sur aucune cellule. `evaluate.open_sealed_test` note chaque ouverture dans `experiments/test_access.jsonl` (date, commit, commit du tag, raison) et refuse une seconde ouverture ; relancé, le notebook ne rouvre pas le test. Commite ce fichier et `experiments/test_results.json` aussitôt : le journal du test ne perd jamais une ligne (son historique git le montre), et le supprimer pour rouvrir le test compte comme un test consulté.
4. **Toute modification signalée.** Changer le protocole après le tag n'est pas interdit (une métrique mal choisie, un défaut découvert tard) : la modification se fait dans un nouveau commit, s'écrit dans la section « Modifications » du protocole et dans le rapport, **avant** l'ouverture du test. Une réouverture du test (`allow_reopen=True`) se rapporte de même, avec les deux résultats.

Pénalité : un test consulté avant l'évaluation finale, ou un protocole modifié après coup sans le signaler, coûte **3 points** (la grille le vérifie par `git log`, le tag et `experiments/test_access.jsonl`).

**L'exploration et le test.** L'audit de PF.3 se fait avant le découpage, sur tout le dataset, mais seulement sur sa structure : tailles, identifiants, doublons, valeurs manquantes, groupes, dates, répartition de la cible. Il sert à choisir la règle du découpage. Les relations entre les entrées et la cible, celles qui inspirent tes choix, ne s'explorent que sur l'entraînement. Comparer la répartition de la cible ou des entrées entre découpages (le décalage) est permis : ce sont des comptages, sans modèle.

<a id="budget"></a>

## Le budget

- **FAST_MODE** (`configs/fast.yaml`, `FAST_MODE = True` dans les notebooks) : tout le projet s'exécute sur un CPU en **moins de 10 minutes** (sous-échantillon d'entraînement, peu d'epochs, moins de tirages bootstrap). C'est lui que lancent les tests, l'intégration continue et le correcteur. Le protocole est le même dans les deux modes : seul le budget change (la quantité de données d'entraînement, les epochs, la patience, la taille de batch, les tirages bootstrap, la machine) ; la tâche, la cible, le découpage, la graine et la métrique restent les mêmes (un test de structure du kit le vérifie).
- **Le mode complet** (`configs/full.yaml`) : toutes les données, assez d'epochs. PF.6 et PF.7 dépassent en général 15 minutes de CPU : 🚀 **GPU Colab conseillé** (*Exécution → Modifier le type d'exécution → GPU*). Note la durée et la machine de chaque run (le journal les garde).
- **Un projet fait en mode rapide seulement** peut avoir tous les points s'il le dit, et si ses chiffres sont ceux du mode rapide, avec leurs intervalles. Le mode complet sur Colab est recommandé pour le portfolio, pas exigé.
- **Le budget de calcul est une contrainte du protocole** : une vingtaine de runs pour la recherche d'hyperparamètres, trois graines pour le modèle retenu et ses ablations. Au-delà, dis pourquoi (⚖️ 19.11).

<a id="jalons"></a>

## Les jalons (4 à 6 semaines)

| Semaine | Étapes | À la fin de la semaine | ⏱️ |
|---|---|---|---|
| 1 | PF.1 à PF.4 : cadrage, données, exploration, protocole | `docs/cadrage.md`, `data/card.md`, `data/splits.json`, `notebooks/01_exploration.ipynb` exécuté, **le tag `protocole` poussé** | 7 h |
| 2 et 3 | PF.5 à PF.7 : baselines, modèle, expériences | les baselines et le modèle dans le journal, les ablations sur trois graines, le tableau de synthèse généré | 17 h |
| 4 | PF.8 et PF.9 : évaluation sur le test scellé, explicabilité et équité | `experiments/test_access.jsonl` (une ligne), `experiments/test_results.json`, les sections « Résultats » et « Explicabilité et équité » | 5 h 30 |
| 5 et 6 | PF.10 à PF.13 : industrialisation, démonstration, rapport, soutenance | l'intégration continue verte, l'API, le README, la model card, la relecture P9, la soutenance enregistrée | 11 h 30 |

Le seul jalon qui ne se rattrape pas est le tag de la semaine 1 : sans lui, rien ne prouve que le protocole précède les résultats.

<a id="demarrer"></a>

## Démarrer

```bash
python tools/start_chapter.py PF
```

copie le kit de départ dans `mon_travail/projet_final/`, sans jamais écraser un fichier existant (relance-la après une mise à jour du workbook : seuls les fichiers manquants arrivent). Ce dossier devient **ton dépôt**, distinct du workbook :

```bash
cd mon_travail/projet_final
git init -b main && git add -A && git commit -m "Kit de départ du projet final"
echo "mon_travail/projet_final/" >> ../../.git/info/exclude    # le workbook ignore ce dépôt imbriqué, sans toucher à son .gitignore
# crée un dépôt vide sur GitHub, puis :
git remote add origin https://github.com/<ton-compte>/<ton-depot>.git && git push -u origin main
```

Puis, dans cet ordre :

1. **Renomme le paquet** : `src/projet/` devient un nom court (`velo_demande`, `birdclf`) ; remplace `projet` dans `pyproject.toml`, `scripts/`, `tests/` et `notebooks/` (`git grep -n "projet"` les liste). Garder `projet` est permis, mais moins parlant.
2. **Installe** : `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu` (sur CPU), puis `pip install -e ".[dev,serve,notebooks]"`. Sur Colab, clone ton dépôt dans Drive et donne son chemin à `REPO_ON_DRIVE` dans la cellule de setup des notebooks.
3. **Lance les tests** : `python -m pytest -q`. Les tests de structure passent dès le départ ; les tests marqués `todo` sont sautés avec ⏳ tant que leurs fonctions ne sont pas écrites.
4. **Pousse** : l'intégration continue (`.github/workflows/ci.yml`) tourne sur GitHub à chaque push ; ajoute son badge en haut du README.

Le correcteur (P9) lit `mon_travail/projet_final/` : garde-y ton dépôt (ou un clone à jour).

<a id="kit"></a>

## Le kit de départ

Le kit (`depart/`) est un dépôt modèle, généralisé du paquet `cifarprod` du bonus B7 : rien n'y est propre à un dataset. Chaque fonction à écrire a sa docstring (le contrat), un `TODO PF.n` et lève `NotImplementedError` ; celles qui disent « Provided » sont fournies et peuvent s'adapter.

| Fichier | Contenu | Étapes |
|---|---|---|
| `docs/cadrage.md`, `docs/protocole.md`, `docs/rapport.md` | les modèles du cadrage (une page), du protocole (avec la ligne « SHA-256 du découpage ») et du rapport technique | PF.1, PF.4, toutes |
| `docs/grille.md` | ta copie de la grille, pour l'auto-évaluation (cases et colonne de preuve) | toutes |
| `data/card.md` | le modèle de data card de 🛠️ 29.15 (six sections) | PF.2 |
| `model_card.md`, `README.md` | les modèles de la model card et du README de portfolio | PF.9, PF.12 |
| `experiments/journal.md` | le carnet d'expériences (hypothèse avant, résultat après) ; `runs.jsonl` est écrit par le code | PF.7 |
| `src/projet/config.py` | la configuration (dataclass figée, YAML, surcharges `--set`, validation) : **fournie** | PF.10 |
| `src/projet/data.py` | `sha256_file`, `download`, `load_examples`, `make_splits`, `save_splits`, `load_splits`, `select`, `subsample` | PF.2 à PF.5 |
| `src/projet/preprocess.py` | `build_preprocessor` (appris sur l'entraînement seul), `make_dataset` | PF.5, PF.6 |
| `src/projet/models.py` | `build_naive`, `build_classical` (un `Pipeline`), `build_model` (un `nn.Module`) ; `n_parameters` fournie | PF.5, PF.6 |
| `src/projet/train.py` | `set_seed`, `cross_validate`, `fit`, `save_checkpoint`, `load_checkpoint`, `log_run`, `read_runs` ; `new_run_id`, `environment` fournies | PF.5 à PF.7 |
| `src/projet/evaluate.py` | `compute_metrics`, `bootstrap_ci`, `paired_bootstrap`, `subgroup_table`, `worst_errors` ; `open_sealed_test`, `protocol_status` **fournies** | PF.8, PF.9 |
| `src/projet/predict.py`, `serve.py` | `save_artifact`, `load_artifact`, `Predictor.predict`, `create_app` (FastAPI), `measure_latency` | PF.11 |
| `src/projet/cli.py` | la ligne de commande `projet` et ses sous-commandes `download`, `split`, `baselines`, `train`, `predict`, `latency`, `serve` : **fournie** (le câblage, à adapter) | PF.10 |
| `scripts/` | `download_data.py`, `predict.py` (des appels à la ligne de commande), `summarize_runs.py` (le tableau du journal, à écrire) | PF.2, PF.7, PF.11 |
| `notebooks/` | `01_exploration`, `02_baselines`, `03_evaluation` : ils n'appellent que ton paquet, et « Tout exécuter » à vide se termine en ⏳ ; ils créent `figures/`, où vont les figures du rapport | PF.3, PF.5, PF.8 |
| `tests/` | `test_structure.py` (passe dès le départ) ; `test_data.py`, `test_models.py`, `test_evaluate.py`, `test_serve.py` (marqués `todo`, à compléter) | PF.10 |
| `configs/fast.yaml`, `configs/full.yaml` | le mode rapide et le mode complet, même protocole | PF.4, PF.6 |
| `.github/workflows/ci.yml`, `Dockerfile` (facultatif), `.gitignore` | l'intégration continue, l'image de l'API, les données et les checkpoints hors de git | PF.10, PF.11 |

Le câblage de `cli.py` et des notebooks suppose une table d'exemples (`load_examples`) et un découpage par identifiants. Un projet d'apprentissage par renforcement le remplace : `load_examples` rend les graines d'évaluation, `make_splits` sépare les graines de réglage de celles du test, et `build_model` rend l'agent ; adapte les cellules, garde la règle.

<a id="etapes"></a>

## Les 13 étapes

Chaque étape a son livrable exact (le chemin du fichier dans ton dépôt), ses critères de réussite vérifiables et ses points. Les critères détaillés, case par case, sont dans [`GRILLE.md`](GRILLE.md) ; les indices dans [`04_indices.md`](04_indices.md).

### PF.1 — Cadrage : problème, utilisateurs, métrique métier et métrique technique, critère de succès chiffré, risques 🗣️ ★★★ ⏱️ 90 min
**Objectif :** dire en une page quel problème tu résous, pour qui, et à quel chiffre tu sauras que c'est réussi, avant d'avoir vu un seul résultat.
**Prérequis :** 🗣️ 29.4 · ch. 29 · **Points :** 1

**Chapitres mobilisés** : ch. 3, la métrique qui suit le coût des erreurs (🗣️ 3.9, ⚖️ 3.10, 🏆 3.29, 💼 3.E1 et 3.E2) ; ch. 29, le pitch (🗣️ 29.4) et le choix du dataset (💼 29.E1).

**Livrable** : `docs/cadrage.md` (une page) et pitch de cinq lignes repris de l'Ex 29.4.

**Critères de réussite**
- une page au plus, le pitch de cinq lignes en tête ;
- une métrique métier (ce que compte l'utilisateur) et une métrique technique qui la suit, avec la raison du choix (déséquilibre, coûts asymétriques) ;
- un critère de succès **chiffré** (une différence avec la baseline classique, un seuil, son intervalle) et une contrainte (latence, taille, CPU) ;
- le périmètre (ce que le projet ne fait pas) et deux ou trois risques, chacun avec une parade ;
- le commit du cadrage précède ceux des données (`git log`).

**Erreurs à éviter** : l'accuracy sur des classes déséquilibrées ; un critère vague (« le meilleur possible ») ; un problème trop large pour 41 h ; un critère écrit après les premiers résultats.

### PF.2 — Données : data card complète, licence vérifiée, script de téléchargement reproductible (révision figée, SHA-256) 🛠️ ★★★ ⏱️ 90 min
**Objectif :** documenter tes données et les rendre téléchargeables par n'importe qui, à l'octet près.
**Prérequis :** 🛠️ 29.15 · PF.1 · **Points :** 1

**Chapitres mobilisés** : ch. 29, la data card et le téléchargement vérifié (🛠️ 29.15), la liste de vérifications (⚖️ 29.5), les datasheets (📄 29.6), le Hugging Face Hub et OpenML (📦 29.10, 29.11), la licence et l'usage (💼 29.E2, 29.E4).

**Livrable** : `data/card.md` et `scripts/download_data.py` (repris et complétés depuis l'Ex 29.15).

**Critères de réussite**
- les six sections de la data card remplies (Provenance, Contenu, Licence, Usage prévu, Limites et biais, Téléchargement), et ce que tu n'as pas pu vérifier, dit ;
- la licence lue sur la page officielle, son nom exact et son lien, compatible avec un dépôt public ;
- une version figée (révision, version, date d'archive) et un SHA-256 par fichier ;
- `python scripts/download_data.py` télécharge et vérifie ; relancé, il ne télécharge rien ; une somme fausse arrête tout ;
- aucune donnée dans git, aucune clé d'API dans le code ;
- `load_examples` rend le nombre d'exemples écrit dans la data card.

**Erreurs à éviter** : la licence d'un miroir ; une source « latest » qui change sous tes pieds ; des données commitées ; une clé Kaggle ou Hugging Face dans un notebook.

### PF.3 — Exploration et audit de qualité : doublons, fuites, déséquilibre, valeurs manquantes, décalage entre découpages 📈 ★★★★ ⏱️ 180 min
**Objectif :** trouver les défauts de tes données avant qu'ils ne trouvent ton modèle, et en déduire la règle du découpage.
**Prérequis :** 📈 29.12 · 🐛 29.13 · PF.2 · **Points :** 1,5

**Chapitres mobilisés** : ch. 2, regarder avant de calculer (📄 2.12, 🎨 2.30) ; ch. 12, le diagnostic d'un fichier brut et les fuites du prétraitement (📦 12.11, 🔬 12.31) ; ch. 29, le profil express (📈 29.12), les doublons entre entraînement et test (🐛 29.13), le décalage de distribution (🔬 29.14), les huit défauts (🏆 29.16) ; ch. 8, les données dépendantes (📦 8.19).

**Livrable** : `notebooks/01_exploration.ipynb` et section « Données » du rapport.

**Critères de réussite**
- l'audit structurel chiffré : identifiants, doublons exacts **et** quasi-doublons (avec ta règle), valeurs manquantes (y compris codées par un nombre), déséquilibre de la cible, colonnes constantes ou qui trahissent la cible ;
- chaque défaut trouvé, la commande qui l'a montré et ce que tu en as fait ;
- la règle du découpage (aléatoire stratifié, par groupes, temporel) justifiée par l'audit, et le décalage entre découpages mesuré ;
- l'exploration statistique (entrées contre cible) faite sur l'entraînement seulement ;
- le notebook exécuté de bout en bout, des figures légendées dans `figures/`.

**Erreurs à éviter** : explorer le test ; dédoublonner **après** le découpage ; un découpage aléatoire d'une série temporelle ou de photos en rafale ; un audit sans conséquence sur le découpage.

### PF.4 — Protocole d'évaluation écrit avant toute expérience : découpage, métriques, test scellé, baseline naïve, budget ✏️ ★★★ ⏱️ 60 min
**Objectif :** écrire et figer, avant le premier entraînement, la façon dont ton projet sera jugé.
**Prérequis :** ch. 8 · ch. 3 · PF.3 · **Points :** 1,5

**Chapitres mobilisés** : ch. 8, le test sous clé (🗣️ 8.7, 🔮 8.17, 💼 8.E1), les fuites (🐛 8.24), comparer deux modèles honnêtement (🔬 8.26, 💼 8.E2) ; ch. 3, les moyennes macro, micro et pondérée (🔨 3.25) ; ch. 12, la fuite par le prétraitement (🗣️ 12.9, 🔬 12.31).

**Livrable** : `docs/protocole.md` figé par un commit tagué « protocole ».

**Critères de réussite**
- la règle du découpage, ses tailles et le SHA-256 de `data/splits.json` ;
- la règle du test scellé, et ce que tu feras s'il faut le rouvrir ;
- une métrique principale, des secondaires, la méthode d'intervalle (bootstrap, par groupes s'il y en a) ;
- la baseline naïve et la baseline classique, la validation croisée et ses folds ;
- le critère de succès chiffré, les sous-groupes et les perturbations de robustesse, le budget ;
- le tag `protocole` poussé, daté **avant** la première ligne de `experiments/runs.jsonl` ;
- la section « Modifications » tenue à jour.

**Erreurs à éviter** : taguer après les premiers runs ; changer de métrique en silence ; régler l'early stopping sur le test ; un protocole qui ne nomme pas la baseline.

### PF.5 — Baselines : prédicteur naïf et modèle classique dans un Pipeline sans fuite, en validation croisée 📦 ★★★★ ⏱️ 180 min
**Objectif :** poser le plancher (le naïf) et la barre (le classique) que ton modèle deep learning devra franchir.
**Prérequis :** ch. 15 · ch. 14 · PF.4 · **Points :** 1

**Chapitres mobilisés** : ch. 13, la régression logistique et le texte (📦 13.31, 13.30) ; ch. 14, le gradient boosting et le benchmark reproductible (📦 14.26, 🛠️ 14.27) ; ch. 15, le `Pipeline` sans fuite (🔬 15.24), la même validation croisée pour plusieurs familles (📦 15.27), le `ColumnTransformer` (📦 15.28), les classes déséquilibrées (📦 15.26) ; ch. 24, la validation croisée avec la normalisation dans un `Pipeline` (📦 24.17).

**Livrable** : `notebooks/02_baselines.ipynb` et première ligne du journal d'expériences.

**Critères de réussite**
- la baseline naïve rapportée (classe majoritaire, moyenne, persistance) ;
- la baseline classique dans un `Pipeline` : le prétraitement est réappris dans chaque fold ;
- des folds qui suivent le protocole (groupes, temps), sur l'entraînement seulement ; moyenne ± écart-type ;
- chaque run dans `experiments/runs.jsonl` (configuration, environnement, commit) ;
- le mode rapide en moins de 10 minutes sur CPU.

**Erreurs à éviter** : un scaler appris sur tout le dataset ; une `StratifiedKFold` quand il y a des groupes ; une baseline volontairement faible, pour que le deep learning brille (un recruteur le voit).

### PF.6 — Modèle deep learning adapté aux données : transfer learning, Transformer ou LLM avec LoRA, réseau séquentiel ou tabulaire 📦 ★★★★ ⏱️ 480 min 🚀
**Objectif :** choisir, justifier et entraîner de façon reproductible le modèle deep learning qui convient à tes données.
**Prérequis :** ch. 23 · ch. 24 · B1 · PF.5 · **Points :** 3

**Chapitres mobilisés** : ch. 20, la boucle d'entraînement et sa reproductibilité (📦 20.23, 🛠️ 20.27) ; ch. 21, le CNN en PyTorch (📦 21.22) ; ch. 22, le LSTM face à la persistance (📦 22.23) ; ch. 23, `fit()`, le modèle pré-entraîné, les callbacks et la reprise (📦 23.18, 23.24, 23.25, 23.23) ; ch. 24, CNN et augmentation (📦 24.20, 24.23) ; B1, embeddings figés puis fine-tuning (📦 B1.13, B1.17, B1.19) ; B2, les embeddings de phrases (📦 B2.28) ; B3, le Transformer et le ViT (📦 B3.19, B3.28) ; B4, LoRA (📦 B4.25 à B4.28) ; B8, DQN et PPO (📦 B8.13, B8.20).

**Livrable** : `src/<projet>/models.py`, entraînement reproductible, meilleur checkpoint.

**Critères de réussite**
- le choix du modèle justifié par le type de données, la taille du dataset et le budget (pourquoi pas plus gros, pourquoi pas plus petit) ;
- `projet train --config configs/fast.yaml` reproduit le run, la même graine donne le même résultat (ou le non-déterminisme du GPU est mesuré et dit) ;
- le meilleur checkpoint choisi sur la validation, l'early stopping, le prétraitement du modèle pré-entraîné respecté ;
- un test qui fait apprendre par cœur un mini-batch au modèle ;
- le mode rapide en moins de 10 minutes sur CPU, le mode complet documenté (durée, machine) ;
- la comparaison avec la baseline classique, sur la validation.

**Erreurs à éviter** : oublier la normalisation du modèle pré-entraîné ; un learning rate trop grand sur les couches pré-entraînées, ou l'optimiseur créé avant la nouvelle tête (🐛 B1.18) ; un LLM trop gros pour le budget ; `eval()` oublié à la validation.

### PF.7 — Expériences : journal, recherche d'hyperparamètres ciblée, ablations, trois graines 🔬 ★★★★ ⏱️ 360 min 🚀
**Objectif :** apprendre quelque chose de chaque run, et ne conclure que ce que trois graines permettent de conclure.
**Prérequis :** ch. 24 · B7 · PF.6 · **Points :** 1,5

**Chapitres mobilisés** : ch. 19, les optimiseurs, le warmup et le cosinus (🔨 19.19, 🔬 19.27, ⚖️ 19.11) ; ch. 24, un hyperparamètre à la fois, le journal, la recherche en trois temps, une seule graine suffit-elle (🔬 24.11, 🛠️ 24.15, 📦 24.18, ⚖️ 24.9) ; B7, le non-déterminisme et le suivi d'expériences (🔬 B7.13, 📦 B7.14) ; les ablations de B3 et B8 (🔬 B3.25, B8.15).

**Livrable** : `experiments/runs.jsonl` et tableau de synthèse généré par script.

**Critères de réussite**
- chaque run dans le journal, les ratés compris (configuration, environnement, commit, durée) ;
- une recherche d'hyperparamètres ciblée (une vingtaine de runs au plus), sur la validation ;
- au moins deux ablations, une variable à la fois, et l'hypothèse écrite **avant** dans `experiments/journal.md` ;
- trois graines pour le modèle retenu et pour chaque ablation, moyenne ± écart-type ;
- le tableau de synthèse généré par `scripts/summarize_runs.py`, jamais à la main.

**Erreurs à éviter** : rapporter la meilleure graine ; changer deux choses à la fois ; un tableau recopié à la main ; un coup d'œil au test « pour voir ».

### PF.8 — Évaluation honnête sur le test scellé : une seule passe, intervalles bootstrap, analyse d'erreurs, sous-groupes, robustesse 📈 ★★★★ ⏱️ 180 min
**Objectif :** mesurer une seule fois ce que vaut vraiment ton modèle, avec son incertitude, et comprendre où il se trompe.
**Prérequis :** ch. 2 · ch. 3 · 🔬 29.14 · PF.7 · **Points :** 2,5

**Chapitres mobilisés** : ch. 2, le bootstrap (🔨 2.22, 📦 2.24, 💼 2.E3) ; ch. 3, ROC, PR et calibration (🔨 3.24, 3.26, 3.28) ; ch. 8, le bootstrap apparié (🔬 8.26) ; ch. 23, l'analyse d'erreurs (📦 23.20) ; ch. 29, le décalage de distribution (🔬 29.14) ; B6, les intervalles par groupe et la lecture d'un audit (📦 B6.24, 📈 B6.25).

**Livrable** : `notebooks/03_evaluation.ipynb` et section « Résultats » du rapport.

**Critères de réussite**
- une seule ligne dans `experiments/test_access.jsonl`, après le tag `protocole`, avec le protocole et le découpage commités (les champs `protocol_tag_commit`, `protocol_changed_since_tag` et `uncommitted_protocol_or_split` de la ligne) ;
- les trois modèles sur le test, chacun avec l'intervalle bootstrap à 95 % de la métrique principale ;
- la différence avec la baseline classique et son intervalle apparié ; le critère de succès atteint ou non, dit sans détour ;
- vingt erreurs lues et rangées par cause, trois exemples commentés ;
- les sous-groupes du protocole, avec leurs effectifs et leurs intervalles ; au moins une perturbation de robustesse ;
- `experiments/test_results.json` commité dans la foulée.

**Erreurs à éviter** : relancer l'évaluation après un « petit réglage » ; des intervalles calculés sur la validation ; un bootstrap exemple par exemple quand les exemples sont groupés ; ne rapporter que les sous-groupes flatteurs.

### PF.9 — Explicabilité et équité : attributions adaptées au modèle, audit par sous-groupe, limites et usages à proscrire ⚖️ ★★★★ ⏱️ 150 min
**Objectif :** montrer sur quoi ton modèle s'appuie, qui il sert moins bien, et à quoi il ne doit pas servir.
**Prérequis :** B6 · PF.8 · **Points :** 1,5

**Chapitres mobilisés** : B6, les explications globales (📦 B6.12), SHAP (📦 B6.15, 🔬 B6.16, 📦 B6.17), les gradients intégrés et Grad-CAM (📦 B6.19, B6.20), le test de santé d'Adebayo (🔬 B6.22), les indicateurs d'équité et l'audit (🔨 B6.23, 📦 B6.24, 🔮 B6.26, 🔬 B6.27), la fiche modèle (🛠️ B6.28), les cas ⚖️ B6.9 et B6.10.

**Livrable** : section « Explicabilité et équité » du rapport et de la model card.

**Critères de réussite**
- une méthode d'attribution adaptée au modèle, et son test de santé ;
- des explications sur de bonnes prédictions **et** sur des erreurs ;
- l'audit par sous-groupe, avec ses indicateurs, ses effectifs et ses intervalles (ou, sans variable sensible, pourquoi, et les sous-groupes que tu peux auditer) ;
- les limites des explications, et les usages à proscrire, écrits dans la model card.

**Erreurs à éviter** : lire une valeur SHAP comme une cause ; conclure sur un sous-groupe de douze exemples ; croire qu'un modèle devient équitable quand on retire la variable sensible, sans regarder ses variables de substitution (fiche B6, §206.8).

### PF.10 — Industrialisation : paquet installable, configuration, ligne de commande, tests pytest, intégration continue 🛠️ ★★★★ ⏱️ 240 min
**Objectif :** transformer ton projet en un dépôt qu'un collègue installe, teste et relance en une commande.
**Prérequis :** B7 · PF.6 · **Points :** 2

**Chapitres mobilisés** : B7, du notebook au paquet, la ligne de commande, la configuration (📦 B7.9 à B7.11), les tests du ML (📦 B7.15), l'intégration continue (🛠️ B7.23) ; les tests écrits au fil du workbook (🛠️ 2.31, 8.20, 21.16, B3.29).

**Livrable** : `pyproject.toml`, `src/`, `configs/`, `tests/`, `.github/workflows/ci.yml`.

**Critères de réussite**
- `pip install -e .` sur une machine propre, puis `python -m pytest -q` vert ;
- la configuration validée (une clé inconnue refusée), la ligne de commande et ses codes de sortie ;
- au moins cinq tests à toi qui attrapent de vraies erreurs (découpage, prétraitement sans fuite, formes, apprentissage par cœur d'un mini-batch, API), avec un message qui dit ce qui était attendu ;
- `PF_STRICT=1 python -m pytest -q` ne saute plus rien (sauf les tests marqués `data`, là où le dataset n'est pas téléchargé : dans l'intégration continue) ;
- l'intégration continue verte sur GitHub, son badge dans le README ;
- aucun code qui ne vit que dans un notebook.

**Erreurs à éviter** : des tests qui ne peuvent pas échouer ; une intégration continue qui télécharge le dataset ; des chemins absolus.

### PF.11 — Démonstration : prédiction en ligne de commande et API HTTP (ou démo Gradio), latence mesurée 📦 ★★★★ ⏱️ 150 min
**Objectif :** servir ton modèle comme il le serait en production, et mesurer ce que coûte une prédiction.
**Prérequis :** B7 · PF.10 · **Points :** 1

**Chapitres mobilisés** : ch. 23, prédire sur de nouvelles entrées et faire voyager le prétraitement (📦 23.22, 🗣️ 23.9) ; B7, l'artefact (📦 B7.17), l'API (📦 B7.18), la latence (🔮 B7.19, 🔬 B7.20), le Dockerfile (📦 B7.24), le service qui ne marche qu'au notebook (🐛 B7.16).

**Livrable** : `scripts/predict.py`, API avec `/health` et `/predict`, mesure de latence.

**Critères de réussite**
- un artefact qui se suffit : un processus neuf prédit sans le code d'entraînement ni les données ;
- `scripts/predict.py` prédit depuis une entrée brute, celles d'un fichier `examples.jsonl` versionné (trois ou quatre entrées choisies) ;
- l'API répond sur `/health` et `/predict`, refuse une entrée invalide en 422 (ou la démo Gradio) ;
- la même prédiction par la ligne de commande, l'API et le notebook ;
- la latence p50 et p95, avec la machine.

**Erreurs à éviter** : un prétraitement différent en production ; `eval()` oublié ; le module entier sérialisé par pickle au lieu des poids et de la configuration.

### PF.12 — Rapport et portfolio : README, model card, data card, figures, dépôt GitHub propre et reproductible 🛠️ ★★★★ ⏱️ 180 min
**Objectif :** livrer un dépôt public qu'un recruteur comprend en 30 secondes et qu'un relecteur peut vérifier.
**Prérequis :** ch. 23 · ch. 29 · PF.9 · PF.11 · **Points :** 1,5

**Chapitres mobilisés** : ch. 23, la fiche modèle (🛠️ 23.28) ; ch. 29, la data card (🛠️ 29.15) ; B6, la fiche modèle avec équité (🛠️ B6.28) ; B7, le dépôt prêt pour la production (🏆 B7.25).

**Livrable** : `README.md`, `model_card.md`, `figures/` ; dépôt public relu par un tiers (ou par Claude, P9).

**Critères de réussite**
- le README dit en 30 secondes le problème, le résultat (avec son intervalle et la baseline) et comment lancer ;
- la reproduction en une suite de commandes, chacune avec sa durée mesurée ;
- la data card et la model card complètes, cohérentes avec `experiments/test_results.json` ;
- des figures légendées ; un rapport (`docs/rapport.md`) dont chaque chiffre vient d'un fichier du dépôt ;
- un dépôt propre : ni données, ni secrets (historique compris), une licence pour ton code, un historique git lisible ;
- un `git clone` neuf qui s'installe et passe les tests ; une relecture par un tiers ou par Claude (P9), ses remarques traitées.

**Erreurs à éviter** : un README sans chiffres, ou des chiffres sans intervalle ; une clé d'API restée dans l'historique git ; des liens morts vers des figures.

### PF.13 — Soutenance : dix minutes de présentation, cinq minutes de questions façon entretien, démonstration en direct 🗣️ ★★★★ ⏱️ 120 min
**Objectif :** présenter et défendre ton projet comme en entretien technique.
**Prérequis :** PF.12 · **Points :** 1

**Chapitres mobilisés** : ch. 29 et B7 (🗣️ B7.6 : convaincre un manager d'industrialiser) ; les questions 💼 de tous les chapitres mobilisés, dans la banque de [`SOUTENANCE.md`](SOUTENANCE.md).

**Livrable** : diapositives (PDF) et répétition chronométrée enregistrée.

**Critères de réussite** : le plan, la démonstration et les critères de [`SOUTENANCE.md`](SOUTENANCE.md) (temps tenu, chiffres avec intervalles et baseline, démonstration qui marche ou plan B, réponses précises qui renvoient au dépôt).

**Erreurs à éviter** : vingt diapositives ; un entraînement en direct ; des chiffres sans incertitude ; esquiver une question au lieu de dire « je ne sais pas, voici comment je le vérifierais ».

<a id="parcours-rapide"></a>

## Le parcours rapide

Les quatre parcours font le projet final (BIBLE §22). Le **parcours rapide** fait en entier PF.1 à PF.6, PF.8, PF.10, PF.12 et PF.13 (≈ 30 h), et **allège** les trois autres étapes (quelques heures de plus) :

- PF.7 : **une** ablation (sur trois graines), le journal et le tableau généré ;
- PF.9 : **une** explication adaptée au modèle, avec son test de santé, et les usages à proscrire ;
- PF.11 : `scripts/predict.py` seul (pas d'API).

La grille note alors ces trois étapes sur leur version allégée : tous leurs points restent atteignables.

<a id="grille"></a>

## La grille (sur 20)

| Étape | Points |
|---|---|
| PF.1 Cadrage | 1 |
| PF.2 Données et data card | 1 |
| PF.3 Exploration et audit | 1,5 |
| PF.4 Protocole figé avant les expériences | 1,5 |
| PF.5 Baselines | 1 |
| PF.6 Modèle deep learning adapté et justifié | 3 |
| PF.7 Expériences (journal, ablations, trois graines) | 1,5 |
| PF.8 Évaluation honnête sur le test scellé | 2,5 |
| PF.9 Explicabilité et équité | 1,5 |
| PF.10 Industrialisation (paquet, tests, intégration continue) | 2 |
| PF.11 Démonstration ou API | 1 |
| PF.12 Rapport, README, model card | 1,5 |
| PF.13 Soutenance | 1 |
| **Total** | **20** |

**Pénalités** : test consulté avant l'évaluation finale, ou protocole modifié après coup sans le signaler (**−3**) ; notebook qui ne s'exécute pas de bout en bout (**−2**) ; résultat sans intervalle ni baseline (**−1**).

**Seuils** : **14/20 ou plus**, le projet est présentable en portfolio ; **moins de 10/20**, reprends les étapes que signale la correction.

Le détail, critère par critère, avec une colonne de preuve à remplir : [`GRILLE.md`](GRILLE.md) (ta copie est `docs/grille.md` dans ton dépôt).

<a id="correction"></a>

## La correction

1. **L'auto-évaluation** : remplis `docs/grille.md` dans ton dépôt, une case par critère, et la colonne de preuve (le fichier, le commit, la cellule). Un critère sans preuve ne compte pas.
2. **La correction par Claude (prompt P9)** : il lit `mon_travail/projet_final/` (le dépôt, son historique git, sa grille remplie), sans jamais y écrire, refait les vérifications de la grille (le tag et sa date, le journal du test, un clone neuf qui s'installe et passe les tests, le mode rapide) et note chaque étape. Il signale les étapes à reprendre ; à moins de 10/20, reprends-les, puis redemande une correction.
3. **La soutenance simulée** : avec les questions de [`SOUTENANCE.md`](SOUTENANCE.md).

<a id="soutenance"></a>

## La soutenance

Le plan des dix minutes, les conseils de démonstration et la banque de questions d'entretien sont dans [`SOUTENANCE.md`](SOUTENANCE.md).

<a id="exemple"></a>

## L'exemple de calibrage

Le dossier [`exemple/`](exemple/) contient un cadrage (PF.1) et un protocole (PF.4) remplis pour un dataset qui n'est pas un fil rouge du workbook (la demande horaire de vélos en libre-service de Washington). **Ce n'est pas une solution** et ce n'est pas ton projet : il montre le niveau de précision attendu, une page chacun. Lis-le après avoir écrit ta première version, pas avant.

## Pour aller plus loin

- **Le suivi de la dérive** : un rapport PSI et KS sur des données plus récentes que ton test, avec un seuil d'alerte (📈 B7.22).
- **Un Space Hugging Face** : ta démo Gradio publiée, avec la model card.
- **L'export** : `torch.export` et la vérification numérique de l'équivalence (📦 B7.21).
- **Un second dataset** : le même code sur des données voisines (une autre ville, une autre période), et ce que le modèle perd.
