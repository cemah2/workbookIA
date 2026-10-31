# Audit P5 des parties V et VI — consignes communes des relecteurs

**Contexte.** Le dépôt `/home/claude/workbookia` est un workbook d'exercices en français (tutoiement). Il accompagne le livre *Deep Learning: From Basics to Practice* (A. Glassner, 2018, deux volumes). Le code, les docstrings et les commentaires sont en anglais.

**Documents de référence.**
- `docs/BIBLE.md` est la spécification qui fait foi. Lis au minimum les §1, §3, §5, §6, §11 à §15, §17 et §19. Lis aussi les lignes du §22 utiles à ton périmètre : elles consignent des décisions déjà prises, qui ne sont pas des défauts. Les lignes datées de 2026-10-18 (fin de l'audit des parties III et IV) et suivantes s'appliquent aux parties V et VI.
- Le contrat de chaque chapitre est `docs/syllabus/data/<id>.json`, avec sa section de `docs/SYLLABUS.md`.
- `suivi/PROGRESS.md` donne l'état et les règles accumulées : la liste « Toujours valable » de la section « Prochaine étape », la section « Écarts », et les lignes « À valider sur Colab ».
- Les deux audits précédents (`docs/AUDIT_parties_I_II.md` et `docs/AUDIT_parties_III_IV.md`) montrent ce qui a déjà été corrigé et quelles conventions en sont sorties. C'est notamment le cas des propositions R1 à R11 de la session 43, appliquées au §22 et dans `CLAUDE.md`. **Une dérive de ces conventions dans les parties V et VI est un constat.**

**Le périmètre audité** (parties V et VI) :
- **Chapitres** : `chapitres/ch21_cnn`, `ch22_rnn`, `ch23_pytorch_1`, `ch24_pytorch_2`, `ch25_autoencodeurs`, `ch26_rl`, `ch27_gan`, `ch28_creatif`, `ch29_datasets`.
- **Checkpoints** : `checkpoints/partie_5` (CP5) et `checkpoints/partie_6` (CP6).
- **Mini-projets** :
  - `projets/partie_5_classifieur_cifar10` (MP5, package `cifarkit`) ;
  - `projets/partie_6_generatif` (MP6, package `genbench`).
- **Librairie** : les modules `mylearn` de ces chapitres, à savoir `conv.py`, `rnn.py`, `sequences.py`, `vae.py`, `rl.py` et `gen_metrics.py`. Ils existent en trois versions :
  - la référence, dans `solutions/mylearn_ref/` ;
  - les stubs publiés et figés, dans `templates/mylearn_stubs/` ;
  - les tests `tests/test_ch2[1-9]_*.py`.
- **Package `wb`** (`src/wb`) : surtout `nbtools.py`, `envs.py`, et les ajouts des parties V et VI à `datasets.py`, `synth.py` et `plot.py`. Leurs tests sont dans `tests/infra/`.
- **Scripts qui génèrent les notebooks** : `tools/chapters/build_ch21.py` … `build_ch29.py`, `build_cp5.py`, `build_cp6.py`, `build_mp5.py` et `build_mp6.py`, avec leurs modules `ch2X_parts.py` et `ch2X_code.py` et le kit commun `chapter_kit.py`. Les figures viennent de `tools/chapters/figures_*.py`.
- **Annexes** (`annexes/`) : le glossaire, le formulaire, les erreurs fréquentes, `cheatsheets/pytorch.md`, les ressources et les métiers, pour leurs lignes qui concernent les ch. 21 à 29.
- **Data cards** : `data/cards/cifar10.md`, `fashion_mnist.md`, `trap.md`, `rl_envs.md`, `pretrained.md`, `images.md`, `holmes.md`, `verne.md` et `sunspots.md` (pour le ch. 22).
- **Suivi** : `suivi/tableau_de_bord.md` et `suivi/auto_evaluation.md`.

Les parties 0 à IV ne se relisent que pour la cohérence : renvois, notions déjà vues, termes, contradictions.

**Organisation d'un chapitre.** Chaque chapitre contient :
- `01_fiche.md` : le cours ;
- `02_exercices.md` : quiz, rappels 🔁, exercices papier, réflexion, entretien ;
- `03_notebook.ipynb` : les exercices, sans sorties ;
- `04_indices.md` : trois indices par exercice ;
- `05_solutions.md` et `05_solutions.ipynb` (ce dernier exécuté) ;
- `06_mes_reponses.md` : une copie vierge ;
- `flashcards.csv` et `figures/`.

Les réponses courtes sont vérifiées par `wb.check`. Leurs empreintes sont dans `src/wb/answers.json`, générées depuis les notebooks de solutions. **Un défaut d'un notebook se corrige dans son script `tools/chapters/build_*.py`** : indique toujours la cellule et le script qui la produit.

**Le texte du livre** est disponible pour l'audit, à usage interne, et ne doit jamais être versionné.
- Il se trouve dans `…/s62/book/`.
- Les fichiers `book_ch21.txt` à `book_ch29.txt` portent des repères `===== [livre p. N] =====`.
- `vol1.txt` et `vol2.txt` contiennent les volumes complets, sans repères de page.
- Sers-t'en pour trois choses : juger les encadrés « Le livre : » et « Le livre, corrigé », vérifier les numéros de page et les renvois au livre, et vérifier la règle du §3 (ne jamais recopier ni paraphraser de près le livre).
- **Toute citation du livre en anglais dans ton rapport s'écrit entre `[[EN: ` et `]]`**, par exemple `[[EN: the discriminator is…]]`. Ces citations seront retirées de la version versionnée.

**Ta mission : un audit qualité indépendant** (prompt P5).
- Chaque chapitre a déjà été vérifié à sa génération : re-résolution à l'aveugle, apprenant simulé, relecture.
- L'audit cherche surtout ce qu'une vérification chapitre par chapitre ne voit pas :
  - la **cohérence d'ensemble** ;
  - la **dérive du style et des conventions** d'un chapitre à l'autre et par rapport aux parties I à IV ;
  - les **erreurs restantes** ;
  - les **liens entre chapitres** ;
  - la **solidité des tests et des garde-fous** ;
  - l'**expérience de l'apprenant**.
- Profil de l'apprenant : en reconversion, maths de niveau lycée, Python débutant au départ, travaille sur Colab et sur un PC sans GPU.

## Règles

1. **Lecture seule.**
   - Ne modifie aucun fichier de `/home/claude/workbookia`, et n'écris **jamais** dans `mon_travail/`.
   - Pour exécuter du code qui écrit des fichiers (notebooks, `start_chapter.py`, mutations…), travaille dans une copie de ton dossier de travail :
     `mkdir -p COPIE && tar -C /home/claude/workbookia --exclude=.git --exclude=data/downloads -cf - . | tar -C COPIE -xf - && ln -s /home/claude/workbookia/data/downloads COPIE/data/downloads`
     (`rsync` n'est pas installé). Enchaîne tes commandes par `&&`.
   - À la fin, vérifie que `git -C /home/claude/workbookia status --short` est vide.
2. **La machine est partagée.** Elle a 2 cœurs et 7 Go de mémoire, et onze relecteurs y travaillent en même temps. Une saturation de la mémoire redémarre le conteneur et fait perdre le travail de tous.
   - **Un seul processus Python lourd à la fois** : notebook, pytest, entraînement.
   - Préfixe chaque commande lourde par `nice -n 10`, avec `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Mets un délai à tout ce qui peut être long (`timeout 900`).
   - N'exécute pas en entier les notebooks de solutions : ils sont exécutés et leurs sorties sont dans le dépôt. Lis les sorties.
   - **Ne juge aucune durée mesurée ici** : sous charge, les temps sont multipliés par 5 à 10.
3. **Environnement Python** :
   `cd COPIE && export PATH=/root/venv313/bin:$PATH PYTHONPATH=src:tools:tools/chapters SSL_CERT_FILE=/root/.ccr/ca-bundle.crt REQUESTS_CA_BUNDLE=/root/.ccr/ca-bundle.crt && unset MPLBACKEND`
   - Ajoute `solutions` au `PYTHONPATH` pour importer `mylearn_ref`.
   - Pour lire le Markdown d'un notebook, extrais ses cellules avec un petit script `json` plutôt que de lire le JSON brut.
   - Tests : `python -m pytest -q -p no:cacheprovider --impl=ref tests/test_ch21_conv.py`.
4. **Chaque constat est vérifié.**
   - Cite le fichier et la ligne, ou la cellule du notebook et le script `build_*.py` qui la produit, et recopie l'extrait exact.
   - Pour une erreur de fait ou de calcul, montre le calcul (Python) ou la source.
   - Pas de constat vague (« pourrait être plus clair ») sans correction précise.
   - Ne signale ni ce que le §22 de la bible a décidé, ni les points déjà listés « à valider sur Colab » dans `suivi/PROGRESS.md`, ni les écarts déjà consignés dans sa section « Écarts ».
5. **Sévérité.**
   - **MAJEUR**, dans deux cas :
     - une erreur de fond qui trompe ou bloque l'apprenant : fait ou formule faux, réponse vérifiée fausse, exercice infaisable ou à deux réponses, notion utilisée avant d'être enseignée, fuite systématique de réponses, deux chapitres qui enseignent des choses contradictoires ;
     - une correction qui changerait le contrat : énoncé ou réponse d'un exercice publié, ID, titre, type, ★, durée, parcours, stub, règle de la bible, convention transversale, restructuration.

     Pour un majeur, propose des options (A, B…) et ta recommandation.
   - **MINEUR** : un défaut local sans effet sur le contrat, par exemple une coquille, une formulation, un renvoi ou un lien faux, un terme non conforme à la bible, la mise en forme, un message d'erreur ou un indice améliorable, un test qui manque.
   - **SUGGESTION** : une amélioration facultative.
6. **Le rapport s'écrit au fur et à mesure** dans le fichier `rapport.md` de ton dossier de travail, pour qu'une interruption ne perde rien. Ajoute chaque constat dès qu'il est vérifié. C'est le seul fichier que tu écris hors d'une copie, avec tes scripts.
   - Langue et format : en français, en Markdown. Le rapport commence par une ligne « Périmètre », qui donne aussi la liste des scripts de vérification, puis une liste numérotée de constats.
   - Chaque constat suit ce format :

     ```
     ### N. SÉVÉRITÉ · `fichier:ligne` (repère)
     **Extrait :** « … »
     **Problème :** …
     **Correction :** … (texte de remplacement précis quand c'est possible)
     ```

   - Regroupe les défauts répétitifs (le même défaut dans plusieurs fichiers) en un seul constat, avec la liste des occurrences.
   - Termine par un **bilan** court : ce qui est solide, les 3 risques principaux, et un tableau du nombre de constats par sévérité.
   - Priorise : au plus ~120 constats.
   - Ta réponse finale est un résumé de 10 lignes au plus, avec les comptes et les majeurs en une ligne chacun. Le rapport complet reste dans le fichier.
