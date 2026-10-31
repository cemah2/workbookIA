# Audit P5 des parties V et VI : consignes communes des correcteurs (F1 à F7)

**Contexte.** Le dépôt `/home/claude/workbookia` est un workbook d'exercices en français (tutoiement) pour le livre *Deep Learning: From Basics to Practice* (A. Glassner). Onze relecteurs indépendants viennent d'auditer les parties V et VI (ch. 21 à 29, CP5, CP6, MP5, MP6) en lecture seule.

**Les rapports des relecteurs.**
- Versions versionnées : `docs/audit_P5_parties_V_VI/A01_….md` à `A11_….md`.
- Versions brutes, avec les citations du livre balisées `[[EN: …]]` : `…/s62/A0x_*/rapport.md`. Lis-les quand un constat porte sur le livre.

Le livre se trouve dans `…/scratchpad/s62/book/` (`book_ch21.txt` à `book_ch29.txt`, avec des repères `[livre p. N]`). Il est à usage interne : ne le recopie jamais dans le dépôt (BIBLE §3).

Le rapport de synthèse, `docs/AUDIT_parties_V_VI.md`, liste au §3 les propositions **R1 à R7**. **Elles te sont interdites** : elles attendent la décision de l'utilisateur.

**Ta mission : vérifier, puis appliquer, les constats de ta zone.**
- Lis les constats de ta zone dans **les onze rapports** : chaque rapport touche plusieurs zones.
- Vérifie chaque constat dans les fichiers : refais le calcul, rejoue le code, relis le livre. Un relecteur peut se tromper.
- Applique ce qui ne change pas le contrat : les MINEURS, les SUGGESTIONS que tu retiens, et le majeur A11 1 (messages `mistakes` du ch. 21), qui ne touche pas au contrat.

## Lectures et règles

1. **Lis d'abord** :
   - `CLAUDE.md` du dépôt (les règles d'écriture d'un chapitre, à suivre) ;
   - `docs/BIBLE.md` : §5, §6, §11 à §15, §17, §19 et les lignes du §22 de ta zone ;
   - `suivi/PROGRESS.md` : la liste « Toujours valable » de la section « Prochaine étape ».
2. **Ne change jamais le contrat.**
   - Le contrat, c'est l'énoncé d'un exercice publié (sens et données), sa réponse enregistrée, son ID, son titre, son type, ses ★, sa durée, ses parcours, les stubs `templates/mylearn_stubs/` (figés) et les règles de la bible. Il comprend aussi les champs correspondants de `docs/syllabus/data/*.json`.
   - Une coquille ou une formulation d'énoncé qui ne change ni le sens ni les données, c'est permis. Un énoncé à préciser, en revanche, est **proposé** : texte exact de remplacement, dans ta section « Propositions », sans l'appliquer.
   - Un titre de support qui diffère du contrat se corrige **sur le support**, c'est-à-dire qu'on aligne le support sur le contrat, sauf pour les titres que vise R1 ou R2.
3. **N'écris jamais dans `mon_travail/`.** Ne lance ni `python tools/syllabus.py build`, ni `tools/build_answers.py` (sauf `--check`), ni `export_flashcards.py` sans `--check` : le coordinateur les lance à la fin. Ne commite rien et ne fais aucune opération git qui écrit.
4. **Notebooks.**
   - Un notebook se corrige **dans son script** `tools/chapters/build_*.py` (ou dans `chXX_parts.py`, `chXX_code.py`, `figures_*.py`), puis se reconstruit : `python tools/chapters/build_chXX.py`.
   - **Ne réexécute pas les notebooks de solutions dans le dépôt.** Le coordinateur les réexécutera tous à la fin, sur une machine calme.
   - Pour vérifier ton travail, exécute dans une **copie** de ton dossier de travail :
     `mkdir -p COPIE && tar -C /home/claude/workbookia --exclude=.git --exclude=data/downloads -cf - . | tar -C COPIE -xf - && ln -s /home/claude/workbookia/data/downloads COPIE/data/downloads`
     Fais au minimum un « Tout exécuter » à vide du `03_notebook.ipynb` reconstruit, et l'exécution du `05_solutions.ipynb` si tu as touché une cellule qui calcule ou enregistre une réponse. Compare alors les `wb.record` aux empreintes (`python tools/build_answers.py --check` dans la copie, après l'exécution `--inplace` de la copie).
   - Une figure se régénère par son script `figures_*.py` : c'est une figure par fichier, et c'est permis dans le dépôt.
5. **La machine est partagée** : 2 cœurs, 7 Go de mémoire, sept correcteurs en même temps.
   - **Toute commande lourde** (notebook, pytest de plus de quelques secondes, entraînement) **passe par le verrou commun** :
     `flock …/s62/heavy.lock nice -n 10 timeout 1500 <commande>`
     Ainsi, une seule tourne à la fois.
   - Ajoute `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` à ces commandes.
   - Ne juge aucune durée mesurée.
6. **Environnement** :
   `cd /home/claude/workbookia && export PATH=/root/venv313/bin:$PATH PYTHONPATH=src:tools:tools/chapters SSL_CERT_FILE=/root/.ccr/ca-bundle.crt REQUESTS_CA_BUNDLE=/root/.ccr/ca-bundle.crt && unset MPLBACKEND`
   - Ajoute `solutions` au `PYTHONPATH` pour `mylearn_ref`.
   - Tests de la référence : `python -m pytest -q -p no:cacheprovider --impl=ref tests/test_ch21_conv.py`.
   - Vérifications rapides permises dans le dépôt : `python tools/syllabus.py check`, `python tools/build_answers.py --check`, `python tools/export_flashcards.py --check`, `python -m pytest -q -p no:cacheprovider tests/infra/test_terminology.py`.
7. **Zones.** Ne modifie que les fichiers de ta zone. Un constat d'une autre zone va dans ta section « Pour une autre zone », avec le fichier et la correction.
   - **F1** : `chapitres/ch21_cnn`, `chapitres/ch22_rnn`, leurs scripts (`build_ch21.py`, `build_ch22.py`, `ch21_*.py`, `ch22_*.py`, `figures_ch21.py`, `figures_ch22.py`, `history_ch2[12]*`) et les champs hors contrat de `ch21.json` et `ch22.json` (`notes`, `prereq`).
   - **F2** : idem pour les ch. 23 et 24.
   - **F3** : idem pour les ch. 25 et 26.
   - **F4** : idem pour les ch. 27 et 28.
   - **F5** : le ch. 29 (idem), `data/cards/` en entier, `src/wb/datasets.py` et ses tests (`tests/infra/test_wb_datasets.py`, `test_wb_trap.py`).
   - **F6** :
     - `checkpoints/partie_5`, `checkpoints/partie_6` ;
     - `projets/partie_5_classifieur_cifar10`, `projets/partie_6_generatif` ;
     - `build_cp5.py`, `build_cp6.py`, `build_mp5.py`, `build_mp6.py`, `figures_cp5.py`, `figures_cp6.py` ;
     - les champs hors contrat de `cp5.json` et `cp6.json` ;
     - `tests/infra/test_projects.py`.
   - **F7** (transversal) :
     - `annexes/` (glossaire, formulaire, erreurs fréquentes, aide-mémoires, ressources, métiers) ;
     - `solutions/mylearn_ref/{conv,rnn,sequences,vae,rl,gen_metrics}.py` et `tests/test_ch2[1-9]_*.py` ;
     - `src/wb/` sauf `datasets.py`, avec ses tests dans `tests/infra/` (dont `test_terminology.py` et `test_wb_nbtools.py`) ;
     - `tools/chapters/chapter_kit.py`, `tools/syllabus.py` (hors contrat, sans lancer `build`), `tools/start_chapter.py`, `tools/run_all_notebooks.py` ;
     - `README.md`, `METHODE.md`, `00_setup/`.

     Une modification de `chapter_kit.py` ou de `src/wb` qui change l'affichage des notebooks est permise ; dis-le dans ton rapport (le coordinateur reconstruira tous les chapitres à la fin). Elle ne doit casser aucun script de chapitre : `python tools/chapters/build_ch21.py` … `build_ch29.py` doivent tourner dans une copie.

   `suivi/PROGRESS.md`, `docs/BIBLE.md` et `docs/AUDIT_parties_V_VI.md` sont tenus par le coordinateur : propose tes lignes, ne les écris pas.
8. **Ton rapport** s'écrit au fur et à mesure dans `rapport.md`, dans ton dossier de travail. Il contient :
   - un en-tête : les fichiers touchés, les reconstructions faites, les vérifications (avec leurs résultats), et la mention « Réponses enregistrées : aucune ne change » ou la liste des réponses qui changent ;
   - un **tableau de décision**, `| Constat (relecteur et numéro) | Décision | Fichiers touchés |`. La décision vaut « appliqué : … », « non retenu : raison », « reporté (Rn) », « proposé (voir plus bas) » ou « hors zone ».
   - une section « **Propositions** » (texte exact, à valider) ;
   - une section « **Pour une autre zone** » ;
   - des lignes proposées pour le §22 de la bible et pour les « Écarts » de PROGRESS.

   À la fin, vérifie `git -C /home/claude/workbookia status --short mon_travail` (vide). Ta réponse finale tient en 10 lignes au plus.
