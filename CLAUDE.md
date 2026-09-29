# Consignes pour Claude (sessions de génération du workbook)

1. **Lis `docs/BIBLE.md` en entier** avant toute action : c'est la spécification qui fait foi (en cas de conflit avec un prompt, la bible l'emporte sauf « DÉROGATION »). Puis lis `suivi/PROGRESS.md` (état, étape en cours, calibrage) et, à partir de la session 2, `docs/SYLLABUS.md`.
2. **N'écris jamais dans `mon_travail/`** : c'est l'espace de l'apprenant. Tu peux seulement y lire (correction, prompt P9).
3. Un stub `mylearn` publié n'est jamais modifié ; `templates/mylearn_stubs/MANIFEST.json` ne fait que grandir. Les ID d'exercices sont stables.
4. Toute décision ou dérogation → une ligne dans `docs/BIBLE.md` §22.

## Environnement de génération

```bash
# Python 3.13 comme Colab ; PyPI n'est joignable qu'en passant par le proxy
export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt REQUESTS_CA_BUNDLE=/root/.ccr/ca-bundle.crt
unset NO_PROXY no_proxy
uv venv --python /usr/bin/python3.13 ~/venv313 && source ~/venv313/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt && uv pip install -e .
```
- Si le réseau est coupé (proxy 403), le dire à l'utilisateur ; les dépôts GitHub publics restent accessibles par `git clone`.
- Téléchargements de données : `data/downloads/` (ignoré par git). CIFAR-10 : Hugging Face d'abord (le serveur de Toronto est très lent ici). Fashion-MNIST : l'URL torchvision est en HTTP simple et échoue ici, le miroir GitHub prend le relais.

## Commandes de vérification

```bash
python -m pytest -q                         # infra + mylearn (mode learner : skips normaux)
python -m pytest -q --impl=ref              # la référence doit être verte
python -m pytest -q --impl=stubs tests/test_ch*.py   # doit ÉCHOUER (⏳) : les tests ne sont pas vides
python -m pytest -q --run-network           # + tests qui téléchargent
python tools/run_all_notebooks.py chapitres/chXX_*/05_solutions.ipynb --inplace   # exécute, garde les sorties
python tools/build_answers.py               # régénère src/wb/answers.json depuis les solutions exécutées
python tools/build_answers.py --check
python tools/run_all_notebooks.py chapitres/chXX_*/03_notebook.ipynb   # « Run all » sans rien remplir
python tools/export_flashcards.py --check
python 00_setup/check_env.py
```

## Écrire un chapitre

- Construire les notebooks avec `tools/nbbuild.py` (`md`, `code`, `setup_cell("exercise" | "solution")`, `badge`, `write_notebook`) : la cellule de setup vient de `templates/notebook_setup_cell.py`.
- Notebook d'exercices : cellules TODO qui ne font que **définir** (fonctions avec `raise NotImplementedError`, variables `= ...`) ; tout appel au code de l'apprenant dans `with wb.attempt("18.4"):` ; vérification par `wb.check("18.4", valeur)`. Pas de `!` ni de `%` : utiliser Python (`subprocess`). Finir les cellules graphiques par `plt.show()`.
- Notebook de solutions : `mylearn = wb.load_mylearn("ref")` ; chaque réponse vérifiable dans une cellule taguée `answer` avec `wb.record("18.4", valeur, decimals=3, mistakes={...})`.
- Budget FAST_MODE (§4) : notebook de solutions < 10 min sur CPU, aucune cellule > 3 min ; `run_all_notebooks.py` signale les dépassements.
- Données : `wb.datasets.*`, `wb.synth.*` ; graphiques : `wb.plot.*`. Nouvelles fonctions de `wb` : ajouter des tests dans `tests/infra/`.
- Stubs mylearn : un fichier par chapitre dans `templates/mylearn_stubs/` + la même chose implémentée dans `solutions/mylearn_ref/` (imports relatifs uniquement) + entrée dans `MANIFEST.json` + tests `tests/test_chXX_<module>.py` fondés sur un oracle (fixture `mylearn_module("metrics")`).
- Définition du « terminé » : BIBLE §19. Commit `chXX: …`, puis mise à jour de `suivi/PROGRESS.md` et `suivi/tableau_de_bord.md`.

## Git

- Branche `main`, dépôt `cemah2/workbookIA` (remote `https://github.com/cemah2/workbookia`).
- Avant de pousser depuis un clone superficiel : `git fetch origin main` puis rebase si besoin.
- Ne jamais versionner de PDF du livre ni de données lourdes (voir `.gitignore`).
