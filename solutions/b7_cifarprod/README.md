# cifar-prod · un classifieur CIFAR-10, du notebook à la production

Un classifieur d'images 32 × 32 (les dix classes de CIFAR-10) livré comme un projet : un paquet Python installable
(`cifarprod`), une ligne de commande d'entraînement pilotée par une configuration validée, des runs reproductibles et
journalisés, des tests, un artefact de modèle qui se suffit, une API HTTP, un export `torch.export`, une intégration
continue et un Dockerfile. Projet du chapitre bonus B7 du workbook ; le modèle servi par défaut est le VGG 16-32-64 du
mini-projet 5 (85,0 % d'accuracy sur les 10 000 images de test de CIFAR-10, voir `MODEL_CARD.md`).

## Installer

```bash
git clone <ce dépôt> cifar-prod && cd cifar-prod
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu   # PyTorch CPU d'abord (Linux, Windows)
pip install -e ".[dev,serve]"                                                 # le paquet, pytest, FastAPI et uvicorn
```

Python 3.11 ou plus (testé avec 3.13 et PyTorch 2.11). Sans l'index CPU, `pip install torch` télécharge sous Linux la
version CUDA (plus de 2 Go).

## De git clone à une prédiction servie, en une commande

```bash
python scripts/demo.py        # ou : make demo
```

La démo prend `data/cifar10_2000.npz` s'il est là (2 000 images de CIFAR-10 ; les données ne sont pas dans git), sinon
un fichier synthétique ; elle lance un entraînement fumée (`configs/smoke.yaml`), écrit l'artefact dans `runs/demo/`,
sert l'API sur un port libre, l'interroge, puis l'arrête. Sa dernière ligne est un résumé JSON (classe, probabilité,
p50 et p95 de la latence). Sur un CPU à deux cœurs : ≈ 7 s, installation de PyTorch non comprise.

## Tester, entraîner, servir

```bash
pytest -q                                                    # 34 tests (les tests de données sautent sans le fichier)
cifarprod train --config configs/fast.yaml --out artifact    # 2 000 images, 1 epoch, ≈ 4 s ; --set lr=0.01 pour surcharger
python -m cifarprod.serve --artifact artifact --port 8000    # GET /health, POST /predict, POST /predict_batch
curl -s localhost:8000/health
```

Chaque run ajoute une ligne à `runs/runs.jsonl` (configuration, environnement, empreinte des données, métriques,
artefacts) ; codes de sortie : 0 succès, 2 configuration invalide, 1 données illisibles ou entraînement en échec.
L'artefact (`model.safetensors` et `meta.json`) porte l'architecture, la normalisation, les classes dans l'ordre des
sorties, la version du paquet, l'empreinte SHA-256 des données d'entraînement et celle des poids, que le service
vérifie au chargement. `python -c "from cifarprod.export import export_artifact; export_artifact('artifact')"` écrit
`artifact/model.pt2`, rechargeable sans ce code (écart avec le modèle : 0,0 sur 1, 7 et 64 images).

## Résultats et latence

| Modèle | Données | Accuracy |
|---|---|---|
| VGG 16-32-64 (MP5), servi par l'API | 45 000 images, 30 epochs | 85,0 % (test, 10 000 images) |
| CNN 16-32, `configs/fast.yaml` | 2 000 images, 1 epoch | 30,2 % (validation, 500 images) ; 31,6 % ± 2,6 sur 5 graines |

Latence du VGG servi par FastAPI et uvicorn sur un CPU à deux cœurs (200 requêtes après 20 appels à blanc, client
`requests` sur la même machine) : `POST /predict` d'une image, p50 ≈ 5 à 8 ms, **p95 ≈ 8 à 10 ms** ; `POST
/predict_batch` de 32 images, p50 ≈ 70 à 90 ms, p95 ≈ 105 à 120 ms, soit 2 à 3,5 fois le débit des requêtes d'une
image. L'appel direct du modèle à une image prend ≈ 1 ms : l'essentiel du temps d'une requête est le HTTP et le JSON.

## Déployer

```bash
cifarprod train --config configs/fast.yaml --out artifact
docker build -t cifar-api . && docker run -p 8000:8000 cifar-api
```

L'image part de `python:3.13.18-slim-trixie`, installe PyTorch depuis l'index CPU et les versions figées de
`requirements-serve.txt`, et fait tourner le serveur sans privilèges. L'intégration continue
(`.github/workflows/ci.yml`) installe, teste et lance un entraînement fumée à chaque push.

## Organisation

```text
pyproject.toml   configs/   scripts/ (demo.py, make_synthetic.py)   Dockerfile   Makefile   MODEL_CARD.md
src/cifarprod/   data.py model.py train.py config.py cli.py tracking.py artifact.py serve.py export.py
tests/           test_data.py test_model.py test_cli.py test_serve.py
```
