<!-- Ta copie de la grille du projet final (projets/projet_final/GRILLE.md du workbook), pour l'auto-évaluation : coche les critères tenus et remplis la colonne « Preuve ». Elle n'est jamais écrasée ; si la grille du workbook change, compare et recopie ce qui t'intéresse. -->

# Projet final · La grille d'évaluation (sur 20)

> Elle sert deux fois. **En auto-évaluation** : ta copie est `docs/grille.md` dans ton dépôt ; coche chaque critère tenu (`[x]`) et remplis la colonne « Preuve » (le fichier, le commit, la cellule du notebook) ; un critère sans preuve ne compte pas. **En correction** : Claude (prompt P9) refait chaque vérification de la colonne « Comment le vérifier » sur `mon_travail/projet_final/`, sans y écrire, et note chaque étape. Les points d'une étape se donnent critère par critère ; un critère à moitié tenu vaut la moitié de ses points.

**Comment lire la grille** : chaque critère est **observable**, c'est-à-dire vérifiable dans le dépôt par une commande ou une lecture, sans croire l'auteur sur parole. La mention **(R)** donne la version allégée du parcours rapide, pour PF.7, PF.9 et PF.11 : un projet du parcours rapide est noté sur elle, et peut avoir tous les points.

**Les vérifications communes** (à faire une fois, avant la grille) :

```bash
git clone <ton dépôt> /tmp/pf && cd /tmp/pf          # un clone neuf, pas ta copie de travail
git log --reverse --format="%h %ad %s" --date=iso | head -20
git show --no-patch --format="%h %ad" protocole      # la date du tag
head -1 experiments/runs.jsonl ; cat experiments/test_access.jsonl
pip install -e ".[dev,serve]" && python -m pytest -q && PF_STRICT=1 python -m pytest -q
```

## PF.1 — Cadrage (1 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | `docs/cadrage.md` tient en une page et commence par le pitch de cinq lignes (problème et utilisateurs, données, entrée et sortie, métrique et baseline, risque) | 0,25 | lecture ; les cinq points du pitch de 29.4 sont là | |
| [ ] | une métrique métier et une métrique technique qui la suit, et la raison du choix (déséquilibre, coût des erreurs dans chaque sens) | 0,25 | lecture ; l'accuracy seule sur des classes déséquilibrées ne compte pas | |
| [ ] | un critère de succès **chiffré** (écart à la baseline classique et son intervalle, ou un seuil) et une contrainte (latence, taille, CPU) | 0,25 | un nombre écrit, pas « le meilleur possible » | |
| [ ] | le périmètre et deux ou trois risques avec leur parade ; le cadrage commité avant les données | 0,25 | `git log -- docs/cadrage.md` précède `git log -- data/` | |

## PF.2 — Données et data card (1 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | `data/card.md` a ses six sections remplies (Provenance, Contenu, Licence, Usage prévu, Limites et biais, Téléchargement), et dit ce que tu n'as pas pu vérifier | 0,25 | lecture | |
| [ ] | la licence est nommée exactement, avec son lien officiel, et compatible avec un dépôt public ; aucune donnée personnelle non anonymisée | 0,25 | ouvrir le lien ; comparer à la page de la source | |
| [ ] | une version figée (révision, version, date) et un SHA-256 par fichier, les mêmes dans la data card et dans `data.py` | 0,25 | `grep -n sha` dans les deux fichiers | |
| [ ] | `python scripts/download_data.py` télécharge et vérifie ; relancé, il ne télécharge rien ; aucune donnée ni clé d'API dans git | 0,25 | deux lancements dans le clone ; `git log --all --name-only -- data/raw kaggle.json` vide | |

## PF.3 — Exploration et audit de qualité (1,5 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | doublons exacts **et** quasi-doublons comptés, avec la règle des quasi-doublons écrite | 0,25 | cellule de `notebooks/01_exploration.ipynb` | |
| [ ] | valeurs manquantes (y compris codées par un nombre), déséquilibre de la cible, colonnes constantes ou qui trahissent la cible : cherchés et chiffrés | 0,25 | cellules du notebook ; section « Données » du rapport | |
| [ ] | la règle du découpage (stratifié, par groupes, temporel) est justifiée par ce que l'audit a trouvé | 0,25 | rapport et protocole : la phrase « parce que… » | |
| [ ] | le décalage entre découpages est mesuré (répartition de la cible, des entrées, des périodes ou des sources) | 0,25 | cellule « décalage entre découpages » exécutée | |
| [ ] | l'exploration statistique (entrées contre cible) ne regarde que l'entraînement | 0,25 | aucune figure ni corrélation calculée sur `test` | |
| [ ] | le notebook est exécuté de bout en bout, ses figures légendées (dans `figures/`) et reprises dans la section « Données » | 0,25 | sorties présentes, numéros d'exécution dans l'ordre | |

## PF.4 — Protocole figé avant les expériences (1,5 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | le découpage : sa règle, ses tailles, et la ligne « SHA-256 du découpage » égale au SHA-256 de `data/splits.json` | 0,25 | `sha256sum data/splits.json` | |
| [ ] | une métrique principale, des secondaires, la méthode d'intervalle (bootstrap, par groupes s'il y en a) | 0,25 | lecture de `docs/protocole.md` | |
| [ ] | la baseline naïve **nommée** et la baseline classique, la validation croisée et la règle de ses folds | 0,25 | lecture | |
| [ ] | le critère de succès chiffré, les sous-groupes et la robustesse, le budget (runs, graines, machine) | 0,25 | lecture | |
| [ ] | le tag `protocole` existe, poussé, et sa date précède la première ligne de `experiments/runs.jsonl` | 0,25 | `git show protocole` contre `head -1 experiments/runs.jsonl` | |
| [ ] | la section « Modifications » liste chaque changement après le tag (date, commit, pourquoi), ou dit « Aucune » et `git diff protocole -- docs/protocole.md` est vide | 0,25 | `git log protocole..HEAD -- docs/protocole.md` | |

## PF.5 — Baselines (1 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | la baseline naïve est rapportée (classe majoritaire, moyenne, persistance), dans le journal et dans le rapport | 0,25 | `grep naive experiments/runs.jsonl` | |
| [ ] | la baseline classique est un `Pipeline` : le prétraitement est réappris dans chaque fold | 0,25 | `models.build_classical` ; aucun `fit` du prétraitement hors du `Pipeline` | |
| [ ] | les folds suivent le protocole (groupes, temps) et ne voient que l'entraînement ; moyenne ± écart-type rapportés | 0,25 | `train.cross_validate` ; `notebooks/02_baselines.ipynb` | |
| [ ] | `projet baselines --config configs/fast.yaml` tourne en moins de 10 min sur CPU et écrit ses runs | 0,25 | lancement dans le clone (le champ `seconds` du journal) | |

## PF.6 — Modèle deep learning adapté et justifié (3 points)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | le choix du modèle est justifié par le type de données, la taille du dataset et le budget, et comparé à au moins une autre option écartée | 0,5 | section « Le modèle » du rapport | |
| [ ] | `projet train --config configs/fast.yaml` reproduit le run : même graine, même métrique de validation (ou le non-déterminisme mesuré et dit) | 0,5 | deux lancements, deux lignes du journal à comparer | |
| [ ] | le meilleur checkpoint est choisi sur la validation, avec early stopping, et rechargé à la fin | 0,5 | `train.fit` ; l'historique du journal (`best_epoch`) | |
| [ ] | le prétraitement du modèle pré-entraîné est respecté (normalisation, taille d'entrée, tokenizer de sa révision) ; `eval()` et `inference_mode()` à la validation | 0,5 | `preprocess.py`, `train.py` | |
| [ ] | un test fait apprendre par cœur un mini-batch au modèle, et un autre vérifie les formes de sortie | 0,5 | `tests/test_models.py` passe avec `PF_STRICT=1` | |
| [ ] | le mode rapide tient en moins de 10 min sur CPU ; le mode complet est documenté (durée, machine, 🚀) ; la comparaison avec la baseline classique est faite sur la validation | 0,5 | le journal (`seconds`, `environment`) ; le rapport | |

## PF.7 — Expériences : journal, ablations, trois graines (1,5 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | chaque run est dans `experiments/runs.jsonl`, les ratés compris, avec configuration, environnement (versions, commit) et durée | 0,25 | `wc -l` ; des lignes `"status": "failed"` s'il y a eu des échecs | |
| [ ] | une recherche d'hyperparamètres ciblée (une vingtaine de runs au plus), sur la validation, son espace écrit **(R : trois à cinq runs suffisent)** | 0,25 | le journal ; la section « Les expériences » | |
| [ ] | au moins deux ablations, une variable à la fois, l'hypothèse écrite avant dans `experiments/journal.md` **(R : une ablation)** | 0,5 | le carnet daté avant les runs correspondants | |
| [ ] | trois graines pour le modèle retenu et pour chaque ablation, moyenne ± écart-type, jamais la meilleure seule | 0,25 | le tableau de synthèse | |
| [ ] | le tableau est généré par `scripts/summarize_runs.py` et identique à celui du rapport | 0,25 | relancer le script, comparer | |

## PF.8 — Évaluation honnête sur le test scellé (2,5 points)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | une seule ligne dans `experiments/test_access.jsonl`, après le tag (`protocol_tag_commit` présent, `protocol_changed_since_tag` faux ou signalé), commitée avec `experiments/test_results.json` | 0,5 | `cat experiments/test_access.jsonl` ; `git log -- experiments/` | |
| [ ] | les trois modèles (naïf, classique, retenu) évalués sur le test, chacun avec l'**intervalle bootstrap à 95 %** de la métrique principale | 0,5 | `experiments/test_results.json` ; le tableau du rapport | |
| [ ] | la différence avec la baseline classique et son intervalle apparié ; le critère de succès déclaré atteint ou non, sans détour | 0,5 | la section « Résultats » | |
| [ ] | l'analyse d'erreurs : vingt erreurs lues, rangées par cause, trois exemples commentés | 0,5 | la cellule `worst_errors` et le rapport | |
| [ ] | les sous-groupes du protocole avec leurs effectifs et leurs intervalles ; au moins une perturbation de robustesse | 0,5 | la cellule `subgroup_table` ; la section « Résultats » | |

## PF.9 — Explicabilité et équité (1,5 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | une méthode d'attribution adaptée au modèle (SHAP ou permutation, Grad-CAM ou gradients intégrés, attributions par token) **(R : une explication)** | 0,5 | la section du notebook 03 et du rapport | |
| [ ] | son test de santé (poids brouillés, fond changé : B6.16, B6.22), et des explications sur de bonnes prédictions **et** sur des erreurs | 0,25 | figures et texte | |
| [ ] | l'audit par sous-groupe : indicateurs, effectifs, intervalles ; ou, sans variable sensible, pourquoi, et les sous-groupes audités à la place | 0,5 | le tableau du rapport et de la model card | |
| [ ] | les limites des explications et les usages à proscrire, écrits dans la model card | 0,25 | `model_card.md`, section « Explicabilité et équité » | |

## PF.10 — Industrialisation : paquet, tests, intégration continue (2 points)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | `pip install -e .` dans un environnement neuf, puis `python -m pytest -q` vert | 0,5 | dans le clone | |
| [ ] | la configuration est validée (une clé inconnue refusée), la ligne de commande a ses codes de sortie, aucun code ne vit que dans un notebook | 0,25 | `projet train --set epoch=3` rend 2 ; les notebooks n'importent que le paquet | |
| [ ] | au moins cinq tests à toi qui attrapent de vraies erreurs (découpage, prétraitement sans fuite, formes, apprentissage par cœur, API), chacun avec un message qui dit l'attendu | 0,5 | lecture de `tests/` ; casser une fonction à la main fait échouer un test | |
| [ ] | `PF_STRICT=1 python -m pytest -q` ne saute plus rien | 0,25 | dans le clone | |
| [ ] | l'intégration continue est verte sur GitHub, son badge est dans le README, elle ne télécharge pas le dataset | 0,5 | l'onglet *Actions* du dépôt ; `.github/workflows/ci.yml` | |

## PF.11 — Démonstration ou API (1 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | l'artefact se suffit : dans un processus neuf, `scripts/predict.py` prédit depuis une entrée brute sans le code d'entraînement ni les données | 0,5 | `python scripts/predict.py --model ... --input examples.jsonl` dans le clone | |
| [ ] | l'API répond sur `/health` et `/predict` et refuse une entrée invalide en 422 (ou la démo Gradio) ; la même prédiction que la ligne de commande **(R : la même prédiction par `scripts/predict.py` et par le notebook 03)** | 0,25 | `projet serve`, puis `curl` | |
| [ ] | la latence p50 et p95 est mesurée, avec la machine **(R : la durée d'une prédiction, avec la machine)** | 0,25 | `projet latency` ; le README | |

## PF.12 — Rapport, README, model card (1,5 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | le README dit en 30 secondes le problème, le résultat (avec son intervalle et la baseline) et comment lancer | 0,25 | lecture chronométrée | |
| [ ] | la reproduction en une suite de commandes, chacune avec sa durée mesurée, qui marche dans un clone neuf | 0,25 | dans le clone | |
| [ ] | la data card et la model card complètes et cohérentes avec `experiments/test_results.json` | 0,25 | comparer les chiffres | |
| [ ] | `docs/rapport.md` : chaque chiffre vient d'un fichier du dépôt ; les figures sont légendées | 0,25 | prendre trois chiffres au hasard et retrouver leur source | |
| [ ] | un dépôt propre : ni données, ni secrets (historique compris), une licence pour ton code, des messages de commit lisibles | 0,25 | `git log --all -S "api_key" --oneline` vide (de même pour `password`, `HF_TOKEN`) ; un fichier `LICENSE` | |
| [ ] | une relecture par un tiers ou par Claude (P9), et ses remarques traitées (commits qui les citent) | 0,25 | `git log` | |

## PF.13 — Soutenance (1 point)

| Fait | Critère | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | le temps tenu (10 minutes ± 1) et le plan suivi : problème, données, protocole, modèle, résultats, limites, démonstration | 0,25 | la répétition enregistrée ; `docs/soutenance.pdf` | |
| [ ] | des résultats chiffrés **avec** leurs intervalles et la baseline, le critère de succès dit | 0,25 | les diapositives | |
| [ ] | la démonstration en direct marche, ou le plan B est déroulé sans perdre le fil | 0,25 | l'enregistrement | |
| [ ] | des réponses précises aux questions (`SOUTENANCE.md`), qui disent « je ne sais pas » quand c'est le cas et renvoient à une preuve du dépôt | 0,25 | la soutenance simulée | |

## Les pénalités

| Fait | Pénalité | Pts | Comment le vérifier | Preuve |
|---|---|---|---|---|
| [ ] | le test consulté avant l'évaluation finale (plus d'une ligne dans `experiments/test_access.jsonl` sans réouverture signalée, une métrique de test dans les `metrics` de `runs.jsonl`, un `select(..., "test")` contourné), **ou** le protocole modifié après le tag sans le signaler | −3 | `git log protocole..HEAD -- docs/protocole.md data/splits.json`  ; les clés des `metrics` du journal ; `git grep -n "_sealed=True"` hors de `evaluate.py` | |
| [ ] | un notebook livré qui ne s'exécute pas de bout en bout (« Tout exécuter » en mode rapide, dans le clone) | −2 | `jupyter nbconvert --to notebook --execute notebooks/*.ipynb` | |
| [ ] | un résultat rapporté sans intervalle **ni** baseline (README, rapport, model card ou soutenance) | −1 | lecture | |

## Le total et les seuils

| | |
|---|---|
| Total des étapes | / 20 |
| Pénalités | − |
| **Note** | **/ 20** |

- **14/20 ou plus** : le projet est présentable en portfolio. Mets le lien du dépôt dans ton CV.
- **De 10 à 14/20** : reprends d'abord les critères à zéro des étapes les mieux dotées (PF.6, PF.8, PF.10).
- **Moins de 10/20** : reprends les étapes que signale la correction, dans l'ordre du projet, puis redemande une correction.

**Pour le correcteur (P9)** : noter sur les preuves du dépôt seulement, dans un clone neuf ; citer pour chaque critère non tenu le fichier ou la commande qui le montre ; ne jamais réécrire le travail de l'apprenant ; appliquer une pénalité seulement si la vérification de sa ligne la montre ; pour le parcours rapide, noter PF.7, PF.9 et PF.11 sur leur version **(R)**.
