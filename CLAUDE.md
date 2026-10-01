# Consignes pour Claude (sessions de génération du workbook)

1. **Lis `docs/BIBLE.md` en entier** avant toute action : c'est la spécification qui fait foi (en cas de conflit avec un prompt, la bible l'emporte sauf « DÉROGATION »). Puis lis `suivi/PROGRESS.md` (état, étape en cours, calibrage) et, à partir de la session 2, `docs/SYLLABUS.md` (le contrat : pour un chapitre, sa section du SYLLABUS et sa fiche `docs/syllabus/data/chXX.json`).
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
python tools/syllabus.py check              # le syllabus (données + stubs) est cohérent
python tools/syllabus.py build              # régénère SYLLABUS, PARCOURS, tableau de bord, MANIFEST
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

- Suivre le syllabus : exercices, ID, titres, types, parcours et signatures `mylearn` de `docs/syllabus/data/chXX.json`. Tout écart (exercice ajouté, retiré ou modifié) : corriger le JSON, `python tools/syllabus.py build`, et une ligne dans `suivi/PROGRESS.md` § Écarts. Un ID publié ne change plus.
- Construire les notebooks avec `tools/nbbuild.py` (`md`, `code`, `setup_cell("exercise", chapter="18")` ou `setup_cell("solution")`, `badge`, `write_notebook`) : la cellule de setup vient de `templates/notebook_setup_cell.py` ; `chapter` active le repli sur la référence pour les modules des chapitres antérieurs (BIBLE §22).
- Un script `tools/chapters/build_chXX.py` par chapitre (modèles : `build_ch00a.py` pour les exercices du notebook, `build_ch00b.py` pour le kit commun `tools/chapters/chapter_kit.py`, classes `Ex`, `Part`, `Paper`) écrit les **deux** notebooks à partir d'une seule source. Figures de la fiche : `tools/chapters/figures_chXX.py` → `chapitres/<dossier>/figures/`. Après chaque reconstruction : `run_all_notebooks.py …05_solutions.ipynb --inplace`, puis `build_answers.py`. Plusieurs réponses → sous-ID `0A.14a`, `0A.14b`… ; exercices ✏️ vérifiés dans une « Partie 0 » du notebook.
- Chapitre sur plusieurs sessions : créer `chapitres/<dossier>/EN_COURS.md` (état des fichiers) à la 1ʳᵉ session, le supprimer à la dernière (`start_chapter.py` ne copie pas le notebook tant qu'il existe).
- Vérification indépendante : un sous-agent re-résout ✏️/∂/🧮 à l'aveugle, un autre joue l'apprenant sur le notebook avec d'autres méthodes que le corrigé (chaque bonne réponse doit passer).
- Fiche : le guide de lecture marque ⏩ les sections du parcours rapide (ligne « Lecture du parcours rapide » du SYLLABUS).
- Notebook d'exercices : cellules TODO qui ne font que **définir** (fonctions avec `raise NotImplementedError`, variables `= ...`) ; tout appel au code de l'apprenant dans `with wb.attempt("18.4"):` ; vérification par `wb.check("18.4", valeur)`. Pas de `!` ni de `%` : utiliser Python (`subprocess`). Finir les cellules graphiques par `plt.show()`. Expérience d'un 🔮 : l'envelopper dans `guarded(code, [prédictions], message)` du kit, pour que « Run all » ne montre rien avant les prédictions ; une prédiction à deux issues se saisit en booléen ou en entier, pas en mots.
- Notebook de solutions : `mylearn = wb.load_mylearn("ref")` ; chaque réponse vérifiable dans une cellule taguée `answer` avec `wb.record("18.4", valeur, decimals=3, mistakes={...})` ; `mistakes` associe un **message** (une phrase qui dit où est l'erreur) à la mauvaise valeur, jamais l'inverse (`record` refuse un dictionnaire inversé). Une valeur calculée par la fonction de l'apprenant dans la cellule de vérification se vérifie avec `wb.check(id, valeur, computed=True)` (pas de conseil d'arrondi) et s'enregistre à 4 décimales ; un exercice 🔨 déclare son fichier avec `Ex(..., mylearn="metrics.py")`.
- Budget FAST_MODE (§4) : notebook de solutions < 10 min sur CPU, aucune cellule > 3 min ; `run_all_notebooks.py` signale les dépassements.
- Données : `wb.datasets.*`, `wb.synth.*` ; graphiques : `wb.plot.*` ; tests écrits par l'apprenant dans un notebook : `wb.run_pytest`. Nouvelles fonctions de `wb` : ajouter des tests dans `tests/infra/`.
- Stubs mylearn : **ils existent tous depuis la session 2 et ne changent plus** (`templates/mylearn_stubs/`, `MANIFEST.json`). Pour un chapitre : écrire la référence dans `solutions/mylearn_ref/` (imports relatifs uniquement, mêmes signatures) + tests `tests/test_chXX_<module>.py` fondés sur l'oracle indiqué dans la docstring (fixture `mylearn_module("metrics")`). Une fonction `Provided:` est recopiée telle quelle dans la référence. Les tests passent toujours par `mylearn_module(...)` (jamais la fixture `mylearn` ni `import mylearn`) et n'utilisent jamais une autre fonction de l'apprenant comme oracle : l'oracle vient de numpy, scipy, scikit-learn ou PyTorch. Chaque message d'échec dit sur sa **première ligne** ce qui était attendu (`pytest -rf`, que les notebooks affichent, n'imprime qu'elle ; modèle : `assert_close` de `tests/test_ch03_metrics.py`) ; les entrées invalides ont un id lisible et un `why` ; une valeur posée sur un bord n'utilise que des bords exacts (k/n, pas les artefacts de `np.linspace`). Helper de notebook qui lance pytest : le fichier de test explicite (jamais `tests/`), `COLUMNS=1000`, et le nom du test puis sa raison sur deux lignes.
- Définition du « terminé » : BIBLE §19. Commit `chXX: …`, puis mise à jour de `suivi/PROGRESS.md` et `suivi/tableau_de_bord.md`.

## Git

- Branche `main`, dépôt `cemah2/workbookIA` (remote `https://github.com/cemah2/workbookia`).
- Avant de pousser depuis un clone superficiel : `git fetch origin main` puis rebase si besoin.
- Ne jamais versionner de PDF du livre ni de données lourdes (voir `.gitignore`).
