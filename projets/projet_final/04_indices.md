# Projet final · Indices

> **Mode d'emploi.** Cherche d'abord seul : sur un projet ouvert, la bonne durée est plutôt une demi-journée qu'un quart d'heure. Si tu bloques, ouvre **l'indice 1** (la direction : quel chapitre, quelle idée), cherche encore, puis l'indice 2 (la méthode, les étapes), puis l'indice 3. L'indice 3 ne fait jamais le travail à ta place : il donne un exemple sur **un autre dataset** que le tien, un squelette ou une liste de contrôle. Les critères de réussite de chaque étape sont dans `GRILLE.md`.

**Sommaire** : [PF.1](#pf1) · [PF.2](#pf2) · [PF.3](#pf3) · [PF.4](#pf4) · [PF.5](#pf5) · [PF.6](#pf6) · [PF.7](#pf7) · [PF.8](#pf8) · [PF.9](#pf9) · [PF.10](#pf10) · [PF.11](#pf11) · [PF.12](#pf12) · [PF.13](#pf13)

<a id="pf1"></a>

### PF.1 — Cadrage : problème, utilisateurs, métrique métier et métrique technique, critère de succès chiffré, risques

<details><summary>Indice 1</summary>

Pars de la **décision** que le modèle aide à prendre, pas du modèle : qui la prend, à quelle fréquence, et ce qu'une erreur lui coûte dans chaque sens. Relis ton pitch de 🗣️ 29.4 et la fiche du ch. 3 (precision, recall, le coût des erreurs, 🏆 3.29).

</details>
<details><summary>Indice 2</summary>

Dans l'ordre : (1) l'utilisateur et sa décision ; (2) ce qu'il fait aujourd'hui sans modèle (c'est souvent ta baseline naïve) ; (3) la métrique métier, dans son unité à lui (des heures, des euros, des alertes manquées) ; (4) la métrique technique qui la suit le mieux, et pourquoi pas l'accuracy ; (5) le critère de succès : un écart à la baseline **classique**, sur le test, avec son intervalle ; (6) ce que le projet ne fait pas ; (7) trois risques au plus, chacun avec ce que tu feras. Une page.

</details>
<details><summary>Indice 3</summary>

Un exemple complet sur un autre dataset (la demande de vélos de Washington) : `exemple/cadrage.md`. Liste de contrôle du tien : le pitch tient en cinq lignes lisibles par quelqu'un qui ne fait pas de deep learning · la métrique métier a une unité · la métrique technique est nommée exactement (« F1 macro », « MAE en vélos par heure ») · le critère de succès contient un nombre **et** une baseline · une contrainte (latence, CPU, taille) · le périmètre dit au moins une chose que tu ne feras pas · chaque risque a sa parade · le fichier est commité avant que tu remplisses la data card et `data.py`.

</details>

<a id="pf2"></a>

### PF.2 — Données : data card complète, licence vérifiée, script de téléchargement reproductible (révision figée, SHA-256)

<details><summary>Indice 1</summary>

Reprends ce que tu as écrit en 🛠️ 29.15 (`sha256_15`, `download_verified_15`, `CARD_15`) : il reste à le ranger dans `data.py` et `data/card.md`, et à figer la version de la source. La liste de vérifications de ⚖️ 29.5 dit ce qu'il faut chercher sur la page de la source.

</details>
<details><summary>Indice 2</summary>

(1) Trouve la page **officielle** du dataset et sa licence (le fichier `LICENSE`, la carte du Hugging Face Hub, la page UCI ou OpenML) ; (2) fige la version : `HfApi().dataset_info(nom).sha` pour le Hub, le numéro de version de Kaggle ou d'OpenML, sinon la date et l'URL exacte de l'archive ; (3) télécharge une fois, calcule le SHA-256 de chaque fichier, écris-les dans `data.py` (une constante) et dans la data card ; (4) écris `download` : un fichier présent et juste n'est pas retéléchargé, sinon `.part`, vérification, renommage ; (5) écris `load_examples` et compte les exemples ; (6) remplis les six sections.

</details>
<details><summary>Indice 3</summary>

Le squelette de `download`, à adapter à ta source :

```python
FILES = {"hour.csv": "<sha256 en 64 caractères>"}          # one line per file, the same in data/card.md
SOURCE = "https://<page officielle>/<archive figée>"       # or hf_hub_download(..., revision="<commit sha>")

def download(config):
    # for each file: already there with the right SHA-256 -> keep it
    # otherwise: download into <name>.part, check sha256_file(part) == FILES[name] (ValueError if not, part deleted)
    # then part.replace(final)
    # return the list of the final paths
```

Liste de contrôle : la licence est celle de la page d'origine, pas d'un miroir · la data card dit ce que tu n'as pas pu vérifier (consentement, droits des images) · `python scripts/download_data.py` relancé ne télécharge rien · `git status` ne montre aucun fichier de `data/raw/` · aucune clé d'API dans le code (`os.environ`).

</details>

<a id="pf3"></a>

### PF.3 — Exploration et audit de qualité : doublons, fuites, déséquilibre, valeurs manquantes, décalage entre découpages

<details><summary>Indice 1</summary>

Les huit défauts du 🏆 29.16 sont une liste de recherche : doublons exacts, quasi-doublons entre entraînement et test, valeurs manquantes codées par un nombre, colonne sans information, colonne qui trahit le label, unités incohérentes, labels mal écrits, classe très rare. Le 🐛 29.13 montre pourquoi les quasi-doublons imposent un découpage par groupes, et 📦 8.19 comment le faire.

</details>
<details><summary>Indice 2</summary>

Deux temps. **L'audit structurel**, sur tout le dataset : `df.shape`, `df["id"].is_unique`, `duplicated()` sur les entrées (pour des images ou des textes, une empreinte : le SHA-256 du fichier, le texte normalisé), `isna().mean()`, `nunique()`, `value_counts()` de la cible, les groupes et les dates. Puis **le choix du découpage** (stratifié, par groupes, temporel), `make_splits`, `save_splits`. Puis **le décalage** entre découpages (la cellule 4 du notebook) et **l'exploration de l'entraînement** seul. Pour chaque défaut : la commande qui l'a montré, et ce que tu en fais (corrigé, retiré, gardé et signalé).

</details>
<details><summary>Indice 3</summary>

Un exemple sur un autre dataset (la demande de vélos de `exemple/`) : l'audit y trouve que `casual + registered = cnt` pour chaque ligne (deux colonnes qui trahissent la cible : retirées), que les dates font du découpage aléatoire une fuite (des heures voisines des deux côtés : découpage temporel), qu'il manque quelques heures sans location (des trous, pas des zéros : signalé). Liste de contrôle du tien : un chiffre pour chaque défaut cherché, même s'il vaut 0 · une règle de quasi-doublon écrite · une phrase « le découpage est … parce que l'audit a montré … » · aucune cellule ne calcule une relation entre entrées et cible sur le test · le notebook exécuté dans l'ordre.

</details>

<a id="pf4"></a>

### PF.4 — Protocole d'évaluation écrit avant toute expérience : découpage, métriques, test scellé, baseline naïve, budget

<details><summary>Indice 1</summary>

Le protocole répond, avant les résultats, à la question qu'un relecteur posera après : « comment sais-tu que ce chiffre n'est pas optimiste ? ». Relis 🗣️ 8.7 (le test sous clé), 🔮 8.17 (validation ou test : lequel est optimiste) et 🔬 8.26 (comparer deux modèles honnêtement).

</details>
<details><summary>Indice 2</summary>

Remplis les sections du modèle `docs/protocole.md` dans l'ordre : le découpage (la règle de PF.3, les tailles, le SHA-256 que `save_splits` a rendu) ; le test scellé ; une métrique principale et deux ou trois secondaires ; l'intervalle (bootstrap percentile à 95 %, par groupes s'il y en a) ; les deux baselines et la validation croisée ; le critère de succès de ton cadrage ; les sous-groupes et la robustesse ; le budget. Puis commit, tag `protocole`, push des tags, **avant** `projet baselines`.

</details>
<details><summary>Indice 3</summary>

Un exemple complet sur un autre dataset : `exemple/protocole.md`. Liste de contrôle du tien : `sha256sum data/splits.json` égale la ligne « SHA-256 du découpage » · la baseline naïve est nommée (pas « une baseline ») · la métrique principale est unique · l'intervalle dit combien de tirages et s'il est par groupes · le critère de succès est un nombre · les sous-groupes ont une taille minimale · `git tag` liste `protocole` et `git push --tags` est fait · `experiments/runs.jsonl` n'existe pas encore.

</details>

<a id="pf5"></a>

### PF.5 — Baselines : prédicteur naïf et modèle classique dans un Pipeline sans fuite, en validation croisée

<details><summary>Indice 1</summary>

Le naïf est ce que ferait quelqu'un sans modèle (classe majoritaire, moyenne, persistance) ; le classique est le meilleur outil du ch. 13 à 15 pour ton type de données (le tableau « Les variantes acceptées » du README). Tout prétraitement appris va **dans** le `Pipeline` : c'est ce qui le réapprend à chaque fold (🔬 15.24).

</details>
<details><summary>Indice 2</summary>

(1) `set_seed` (la commande `projet baselines` l'appelle en premier) et `build_naive` : `DummyClassifier`, `DummyRegressor`, ou un petit estimateur de persistance (`fit` retient la dernière valeur, `predict` la recopie) ; (2) `build_preprocessor` puis `build_classical` : `make_pipeline(préprocesseur, modèle)` ; (3) `cross_validate` : les folds du protocole (`GroupKFold`, `TimeSeriesSplit`, `StratifiedKFold`) et le `scoring` de `config.metric` ; (4) `log_run` et `read_runs` ; (5) `projet baselines --config configs/fast.yaml`, puis le notebook 02.

</details>
<details><summary>Indice 3</summary>

Le squelette de `cross_validate`, sans le choix des folds (c'est ton protocole) :

```python
from sklearn.model_selection import cross_val_score

def cross_validate(estimator, df, config):
    cv, groups = ..., ...                     # the folds of docs/protocole.md (groups, time, stratification)
    scores = cross_val_score(estimator, df, df[config.target], cv=cv, groups=groups, scoring=config.metric)
    return {"metric": config.metric, "scores": scores.tolist(), "mean": ..., "std": ...}
```

Liste de contrôle : le prétraitement n'est appris nulle part en dehors du `Pipeline` · avec des groupes, aucune `StratifiedKFold` · le naïf apparaît dans le tableau du rapport · deux lignes `"status": "ok"` dans le journal.

</details>

<a id="pf6"></a>

### PF.6 — Modèle deep learning adapté aux données : transfer learning, Transformer ou LLM avec LoRA, réseau séquentiel ou tabulaire

<details><summary>Indice 1</summary>

Le bon modèle est le plus petit qui a une chance de battre ta baseline classique dans ton budget. Images : un réseau pré-entraîné (B1) ; texte : un encodeur pré-entraîné ou un petit LLM avec LoRA (B4) ; série : un LSTM face à la persistance (22.23) ; tabulaire : un MLP avec embeddings, qui a le droit de perdre contre le gradient boosting ; RL : DQN ou PPO (B8).

</details>
<details><summary>Indice 2</summary>

(1) Commence par la version la moins chère qui marche (un linear probe sur des embeddings figés, un LSTM d'une couche) ; (2) écris `set_seed`, `make_dataset`, `build_model`, puis `fit` à partir de ton `fit()` du ch. 23 ou de MP5 (meilleur checkpoint, early stopping, planning) ; (3) écris le test d'apprentissage par cœur d'un mini-batch **avant** le premier vrai entraînement ; (4) `projet train --config configs/fast.yaml`, deux fois, même graine : même résultat ; (5) seulement ensuite, le mode complet (🚀 Colab) et la version plus ambitieuse (fine-tuning partiel, puis complet).

</details>
<details><summary>Indice 3</summary>

Un exemple sur un autre dataset (CIFAR-10, ce que tu as fait en B1) : le linear probe sur ResNet-18 figé sert de premier modèle deep learning ; la marche suivante dégèle le dernier bloc avec deux learning rates (1e-3 pour la tête, 1e-4 pour `layer4`, comme en B1.17) et mesure ce qu'elle gagne, sur trois graines : en B1.17, sur 500 images, le gain restait de l'ordre du bruit d'une graine à l'autre. Chaque marche est un run du journal, gardée seulement si elle bat la précédente de plus que ce bruit. Liste de contrôle du tien : le prétraitement du modèle pré-entraîné (normalisation, taille, tokenizer de sa révision) · l'optimiseur créé **après** le remplacement de la tête · `model.eval()` et `torch.inference_mode()` à la validation · le meilleur checkpoint rechargé à la fin de `fit` · le mode rapide en moins de 10 min · le nombre de paramètres (`n_parameters`) dans le journal · la comparaison avec la baseline classique sur la même validation.

</details>

<a id="pf7"></a>

### PF.7 — Expériences : journal, recherche d'hyperparamètres ciblée, ablations, trois graines

<details><summary>Indice 1</summary>

Une expérience est une question posée **avant** de lancer : « le dégel du dernier bloc gagne-t-il au moins 2 points ? ». Relis 🔬 24.11 (une seule variable), 📦 24.18 (grille, hasard, puis zoom), ⚖️ 24.9 (une seule graine suffit-elle ?) et 📦 B7.14 (le journal).

</details>
<details><summary>Indice 2</summary>

(1) Écris l'hypothèse dans `experiments/journal.md` ; (2) lance les runs avec `--set` (une variable change, le reste non) et trois graines : `for s in 0 1 2; do projet train --set seed=$s --set lr=3e-4; done` ; (3) écris `scripts/summarize_runs.py` : grouper les runs par configuration sans la graine, moyenne et écart-type ; (4) note la conclusion dans le carnet ; (5) la recherche d'hyperparamètres : peu d'hyperparamètres, une échelle logarithmique pour le learning rate, une vingtaine de runs au plus.

</details>
<details><summary>Indice 3</summary>

Le cœur de `summarize` : la clé qui regroupe les graines d'une même configuration. Les deux baselines ont la même configuration : c'est le champ `model` du run qui les sépare.

```python
def config_key(run):
    settings = {k: v for k, v in run["config"].items() if k != "seed"}
    return run["command"], run["model"], json.dumps(settings, sort_keys=True)
# then: group the "ok" runs by config_key, count the failed ones apart, keep one run per seed (a second run of
# the same seed checks reproducibility, it is not a new seed), the validation metric metrics["best_val"] (the
# same validation split for baselines and train), its mean and std per group,
# one Markdown row per group, best first, a mark when there are fewer than three seeds
```

Liste de contrôle : chaque ablation change une seule chose · trois graines partout où tu conclus · le tableau du rapport est la sortie du script · aucune métrique de test dans le journal.

</details>

<a id="pf8"></a>

### PF.8 — Évaluation honnête sur le test scellé : une seule passe, intervalles bootstrap, analyse d'erreurs, sous-groupes, robustesse

<details><summary>Indice 1</summary>

Une seule passe, préparée comme une expérience : tout est écrit et testé **avant** `OPEN_TEST = True`. Relis 🔨 2.22 (le bootstrap), 🔬 8.26 (le bootstrap apparié), 📦 23.20 (l'analyse d'erreurs) et 📦 B6.24 (les intervalles par groupe).

</details>
<details><summary>Indice 2</summary>

(1) Écris et teste `compute_metrics`, `bootstrap_ci`, `paired_bootstrap`, `subgroup_table`, `worst_errors` sur la **validation** (les tests de `test_evaluate.py`, puis le notebook 03 en répétition : avec `OPEN_TEST = False`, il tourne en entier sur la validation, tes cellules de robustesse et d'explication comprises) ; (2) fige le modèle (artefact, run du journal), commite tout ; (3) passe `OPEN_TEST = True` et « Tout exécuter », une fois ; (4) commite `experiments/test_access.jsonl` et `experiments/test_results.json` ; (5) lis les vingt erreurs et range-les par cause (sur un tabulaire, une erreur se lit par son profil : les features qui l'éloignent de la médiane de sa vraie classe) ; (6) écris la section « Résultats ».

</details>
<details><summary>Indice 3</summary>

Le cœur de `bootstrap_ci` (le bootstrap par groupes en est une variante : tirer des groupes, puis prendre tous leurs exemples) :

```python
rng = np.random.default_rng(seed)
scores = []
for _ in range(n_boot):
    idx = rng.integers(0, n, size=n)          # n examples drawn with replacement
    scores.append(metric_fn(y_true[idx], y_pred[idx]))
low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
```

Liste de contrôle : `experiments/test_access.jsonl` a une ligne · trois modèles et trois intervalles · la différence appariée avec la baseline classique · le critère de succès dit atteint ou non · vingt erreurs rangées par cause · des effectifs à côté de chaque sous-groupe · une perturbation de robustesse · tout est commité le jour même.

</details>

<a id="pf9"></a>

### PF.9 — Explicabilité et équité : attributions adaptées au modèle, audit par sous-groupe, limites et usages à proscrire

<details><summary>Indice 1</summary>

L'attribution dépend du modèle : SHAP ou permutation pour un modèle tabulaire (B6.12, B6.15), Grad-CAM ou gradients intégrés pour un CNN (B6.18 à B6.20), des attributions par token pour un Transformer. Et toute explication passe un test de santé (B6.16, B6.22) avant d'être montrée.

</details>
<details><summary>Indice 2</summary>

(1) Choisis une méthode et applique-la à quelques exemples du test : de bonnes prédictions **et** des erreurs ; (2) fais son test de santé (poids brouillés couche par couche, autre fond pour SHAP) ; (3) l'audit : les indicateurs de B6.23 (taux par groupe, écart de parité démographique, d'equalized odds) avec leurs intervalles et effectifs ; sans variable sensible, dis pourquoi et audite les sous-groupes que tu as (sources, périodes, classes) ; (4) écris les limites et les usages à proscrire dans la model card.

</details>
<details><summary>Indice 3</summary>

Un exemple sur un autre dataset (Adult, que tu as audité en B6.24) : les indicateurs par sexe et par âge s'y donnent avec leurs intervalles bootstrap et les effectifs de chaque groupe, et la fiche du B6 (§206.8) rappelle qu'une variable sensible retirée peut revenir par ses variables de substitution. Liste de contrôle du tien : la méthode est adaptée au modèle · son test de santé est fait et montré · au moins une explication porte sur une erreur · chaque indicateur a son effectif et son intervalle · un sous-groupe trop petit est signalé, pas conclu · la model card a sa section « Explicabilité et équité » et ses usages à proscrire.

</details>

<a id="pf10"></a>

### PF.10 — Industrialisation : paquet installable, configuration, ligne de commande, tests pytest, intégration continue

<details><summary>Indice 1</summary>

Le kit te donne la structure du paquet de B7 : il reste à le remplir et à le prouver. Relis 📦 B7.15 (des tests qui attrapent de vraies erreurs) et 🛠️ B7.23 (le workflow d'intégration continue).

</details>
<details><summary>Indice 2</summary>

(1) Ajoute à `Config` les réglages de ton projet (des champs avec une valeur par défaut) ; (2) écris tes tests : le découpage sans recouvrement, le prétraitement appris sur l'entraînement seul, les formes, l'apprentissage par cœur d'un mini-batch, l'API ; chacun avec un message qui dit ce qui était attendu ; (3) vérifie qu'ils échouent quand tu casses la fonction à la main ; (4) pousse et regarde l'onglet *Actions* ; (5) à la fin, `PF_STRICT=1` dans le workflow.

</details>
<details><summary>Indice 3</summary>

Un test d'absence de fuite, sur un autre dataset (un tableau fabriqué dans le test, avec une colonne numérique `x` standardisée) : un test très différent de l'entraînement doit être transformé avec les statistiques de l'entraînement, pas les siennes. Il échoue si le prétraitement est appris sur le test, ou sur l'entraînement et le test réunis.

```python
def test_the_test_is_scaled_with_the_training_statistics():
    train, test = make_toy_frame(seed=0), make_toy_frame(seed=1)     # a helper of your tests
    test["x"] = test["x"] * 1000 + 50                                 # a test set far from the training
    pre = build_preprocessor(config).fit(train)
    expected = (test["x"] - train["x"].mean()) / train["x"].std(ddof=0)
    got = ...                                                         # the column of x in pre.transform(test)
    assert np.allclose(got, expected), "expected x scaled with the mean and std of the training rows only"
```

Liste de contrôle : `pip install -e .` dans un environnement neuf · `projet --help` · une clé inconnue refusée · cinq tests à toi au moins, dont un sur la cible et l'identifiant (les mélanger ne change pas les features, sauf un encodage par la cible) · chacun échoue sur une version cassée · intégration continue verte et badge · `PF_STRICT=1` vert.

</details>

<a id="pf11"></a>

### PF.11 — Démonstration : prédiction en ligne de commande et API HTTP (ou démo Gradio), latence mesurée

<details><summary>Indice 1</summary>

Un artefact qui se suffit : un processus neuf, sans les notebooks ni les données, doit prédire depuis une entrée brute. Relis 📦 B7.17 (l'artefact), 📦 B7.18 (l'API) et 🐛 B7.16 (les cinq écarts entre le notebook et la production).

</details>
<details><summary>Indice 2</summary>

(1) `save_artifact` et `load_artifact` : les poids (`safetensors` ou `state_dict`), le prétraitement appris, `meta.json` (configuration, classes, métriques, SHA-256) ; (2) `Predictor.predict` depuis des dictionnaires bruts ; (3) `scripts/predict.py` sur un `examples.jsonl` ; (4) `create_app` : `/health`, `/predict`, un schéma pydantic, 422 ; (5) `measure_latency`, puis `projet latency` ; (6) vérifie que la ligne de commande, l'API et le notebook donnent la même prédiction.

</details>
<details><summary>Indice 3</summary>

Le squelette de `create_app` (le schéma d'un enregistrement est le tien) :

```python
def create_app(predictor):
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    class Request(BaseModel):
        records: list[dict] = Field(min_length=1, max_length=64)    # or a model of YOUR fields

    app = FastAPI()

    @app.exception_handler(RequestValidationError)                 # the default 422 echoes the input back
    def invalid(request, exc):
        return JSONResponse(status_code=422, content={"detail": [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]})

    @app.get("/health")
    def health(): ...

    @app.post("/predict")
    def predict(request: Request): ...      # predictor.predict(request.records), 500 on an error of the model
    return app
```

Liste de contrôle : un processus neuf prédit · le prétraitement vient de l'artefact · `eval()` · `proba` de toutes les classes à côté du `score` · 422 sur une entrée invalide, sans l'entrée dans la réponse · même prédiction par les trois chemins · p50 et p95 avec la machine.

</details>

<a id="pf12"></a>

### PF.12 — Rapport et portfolio : README, model card, data card, figures, dépôt GitHub propre et reproductible

<details><summary>Indice 1</summary>

Écris pour deux lecteurs : le recruteur pressé (le README, 30 secondes) et le relecteur qui vérifie (le rapport, chaque chiffre avec sa source). Relis 🛠️ 23.28 et B6.28 (la model card) et 🏆 B7.25 (le dépôt prêt pour la production).

</details>
<details><summary>Indice 2</summary>

(1) Le README : la phrase de résumé avec le chiffre et son intervalle, le tableau des trois modèles, une figure, les commandes de reproduction avec leurs durées ; (2) la model card et la data card relues contre `experiments/test_results.json` ; (3) le rapport : chaque section remplie, chaque chiffre suivi de son fichier ; (4) le ménage : `git status` vide, aucune donnée ni clé dans l'historique, un fichier `LICENSE` ; (5) un clone neuf, l'installation, les tests ; (6) une relecture par un tiers ou par Claude (P9), et ses remarques traitées.

</details>
<details><summary>Indice 3</summary>

La structure d'une phrase de résumé, sur un autre dataset : « Un modèle qui prévoit la demande horaire de vélos à Washington 24 h à l'avance, avec une MAE de 41 vélos [38 ; 44] sur les trois derniers mois de 2012, contre 52 pour un gradient boosting et 118 pour la persistance hebdomadaire. » (chiffres inventés pour montrer la forme). Liste de contrôle du tien : le chiffre principal avec son intervalle · la baseline dans la même phrase · les commandes marchent dans un clone neuf · chaque figure a une légende · aucun lien mort · `git log --all -S "api_key"` vide · un fichier `LICENSE`.

</details>

<a id="pf13"></a>

### PF.13 — Soutenance : dix minutes de présentation, cinq minutes de questions façon entretien, démonstration en direct

<details><summary>Indice 1</summary>

Raconte le projet dans l'ordre où un recruteur pose ses questions : le problème, les données, la façon dont tu as jugé, ce que tu as trouvé, ses limites. Le plan minuté et les conseils de démonstration sont dans `SOUTENANCE.md`.

</details>
<details><summary>Indice 2</summary>

(1) Dix diapositives au plus, un titre qui dit la conclusion ; (2) la démonstration préparée (entrées choisies, serveur chauffé, plan B) ; (3) une répétition chronométrée enregistrée, regardée, corrigée ; (4) les questions « Sur ton projet » de `SOUTENANCE.md`, chacune en 60 secondes ; (5) une soutenance simulée avec Claude sur la banque de questions des chapitres de tes étapes.

</details>
<details><summary>Indice 3</summary>

Liste de contrôle : la soutenance tient en 10 minutes ± 1 à la répétition · chaque chiffre a son intervalle et sa baseline · le critère de succès est dit atteint ou non · la démonstration a un plan B · tu as une réponse de 60 secondes aux huit questions « Sur ton projet » · tu as répondu à au moins dix questions de la banque et comparé tes réponses aux réponses modèles des `05_solutions.md` · tu sais dire « je ne sais pas, voici comment je le vérifierais ».

</details>
