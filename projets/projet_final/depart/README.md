# TODO : le titre du projet

<!-- Ce fichier est le modèle du README de ton dépôt (PF.12) : c'est la première page qu'ouvre un recruteur, il doit dire en 30 secondes ce que fait le modèle, à quel point il marche et comment le lancer. Remplace chaque paragraphe qui commence par TODO par ton texte, avec tes propres chiffres (ceux du test scellé, avec leurs intervalles), puis supprime ce commentaire et la section « Le kit de départ » en bas. Tu peux l'écrire en anglais si tu vises des postes à l'international, avec le même plan. -->

![CI](https://github.com/TODO-ton-compte/TODO-ton-depot/actions/workflows/ci.yml/badge.svg)

> TODO : une phrase. Le problème, les données, le modèle, le résultat principal sur le test scellé avec son intervalle, et la baseline qu'il bat (ou ne bat pas).

## Le résultat en un coup d'œil

| Modèle | TODO : métrique principale [IC 95 %] | TODO : métrique secondaire | Latence p95 (CPU) |
|---|---|---|---|
| Baseline naïve | TODO | | |
| Baseline classique | TODO | | |
| **Modèle retenu** | TODO | | TODO |

TODO : une figure qui parle (les erreurs typiques, une explication, la prévision contre la réalité), avec sa légende.

## Le problème

TODO : trois lignes reprises de [docs/cadrage.md](docs/cadrage.md) : pour qui, quelle décision, la métrique métier, le critère de succès fixé avant les expériences.

## Les données

TODO : la source, la licence, la taille, la version figée ; le lien vers la [data card](data/card.md). Les données ne sont pas dans le dépôt : `python scripts/download_data.py` les télécharge et vérifie leur SHA-256.

## La méthode

TODO : le protocole ([docs/protocole.md](docs/protocole.md), figé par le tag `protocole` avant les expériences) ; les baselines ; le modèle et pourquoi lui ; les expériences (journal, ablations, trois graines) ; l'évaluation unique sur le test scellé. Le détail est dans le [rapport](docs/rapport.md).

## Reproduire

```bash
git clone https://github.com/TODO-ton-compte/TODO-ton-depot && cd TODO-ton-depot
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu   # sur CPU
pip install -e ".[dev,serve]"
python scripts/download_data.py                 # les données, vérifiées (SHA-256)
projet baselines --config configs/fast.yaml     # FAST_MODE : moins de 10 min sur CPU
projet train --config configs/fast.yaml         # le mode complet : configs/full.yaml (GPU conseillé)
python -m pytest -q
```

TODO : la durée mesurée de chaque commande, et la machine.

## La démonstration

```bash
python scripts/predict.py --model artifacts/model --input examples.jsonl
projet serve --model artifacts/model --port 8000      # GET /health, POST /predict
```

TODO : un exemple de requête et de réponse, la latence mesurée (p50, p95), où télécharger le modèle livré (une release GitHub, le Hugging Face Hub) si l'artefact n'est pas dans le dépôt.

## Les limites et l'usage responsable

TODO : ce que le modèle ne sait pas faire, où il se trompe, les sous-groupes moins bien servis, les usages à proscrire ; le lien vers la [model card](model_card.md).

## Structure du dépôt

```
README.md       ce fichier ; model_card.md (la model card) ; examples.jsonl (les entrées de la démonstration) ; LICENSE
configs/        fast.yaml (CPU, moins de 10 min) et full.yaml (le mode complet)
data/           card.md (la data card) et splits.json (le découpage figé) ; le reste de data/ hors de git
docs/           cadrage.md, protocole.md, rapport.md, grille.md (l'auto-évaluation), soutenance.pdf
experiments/    runs.jsonl (le journal), journal.md, test_access.jsonl, test_results.json
figures/        les figures du rapport et du README (créé par les notebooks)
notebooks/      01_exploration, 02_baselines, 03_evaluation
scripts/        download_data.py, predict.py, summarize_runs.py
src/projet/     le paquet (données, prétraitement, modèles, entraînement, évaluation, prédiction, API)
tests/          pytest ; .github/workflows/ci.yml les lance à chaque push
```

## Ce que j'ai appris

TODO : trois lignes honnêtes (ce qui a marché moins bien que prévu, ce que tu referais autrement).

---

## Le kit de départ (à supprimer une fois le projet lancé)

Ce dépôt part du kit du projet final du workbook *Deep Learning* (d'après A. Glassner). Le cahier des charges, la grille, les indices et la banque de questions de la soutenance sont dans le workbook, `projets/projet_final/`.

- **Renommer le paquet** : choisis un nom court (`velo_demande`, `birdclf`), puis, depuis la racine du dépôt :
  `N=velo_demande && git mv src/projet src/$N && git grep -lwz projet -- pyproject.toml configs src scripts tests | xargs -0 sed -i -E "s/\bprojet\b/$N/g" && git grep -lz -E '\bprojet\.' -- notebooks | xargs -0 sed -i -E "s/\bprojet\./$N./g"` (le code seulement : la prose française reste).
- **Installer** : d'abord `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu` (sinon pip prend la version CUDA, plusieurs Go ; `torch.__version__` doit finir par `+cpu`), puis `pip install -e ".[dev,serve,notebooks]"` ; les notebooks et les tests marchent aussi sans installation (ils ajoutent `src/` au chemin).
- **Tester** : `python -m pytest -q`. Les tests de structure passent dès le départ ; les tests marqués `todo` sont sautés avec ⏳ tant que leurs fonctions lèvent `NotImplementedError`, puis doivent passer. À la fin : `PF_STRICT=1 python -m pytest -q` ne doit plus rien sauter.
- **Les fonctions à écrire** : chacune a sa docstring (le contrat) et lève `NotImplementedError("TODO PF.n: nom")` ; celles qui disent « Provided » sont fournies et peuvent être adaptées.
