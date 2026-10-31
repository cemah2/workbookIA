# Audit P5 des parties V et VI : application des propositions R1 à R9 (agents G1 à G7)

**Contexte.** Le dépôt est `/home/claude/workbookia`. L'audit P5 des parties V et VI est terminé : ses constats mineurs sont appliqués. Les propositions **R1 à R9** du §3 de `docs/AUDIT_parties_V_VI.md` sont **validées** par l'utilisateur, qui a retenu les recommandations, avec les choix précis ci-dessous.

**Ta mission :** appliquer, dans ta zone, la partie de ces décisions qui la concerne. Ici, les changements de contrat sont permis : titres, termes, types, ★, durées, parcours et énoncés, mais **seulement ceux qui sont listés ci-dessous**. Les réponses enregistrées ne doivent pas changer. Les ID ne changent jamais. Les stubs `templates/mylearn_stubs/` ne changent jamais.

Lis d'abord `CLAUDE.md` du dépôt, puis le §3 de `docs/AUDIT_parties_V_VI.md`, puis les constats sources dans `docs/audit_P5_parties_V_VI/`. Pour R1, ce sont `A01_terminologie_style.md`, constats 6 à 10, et `F7_transversal.md` (« Pour une autre zone » et « Propositions »). Pour R8, ce sont les sections « Propositions » des fichiers `F*.md`, qui donnent le texte exact.

## Les décisions

**R1, termes et titres** (ID inchangés ; un titre change dans le JSON et sur **tous** les supports : 02, notebook, 04, 05, 06, tableau de bord par `syllabus.py build`) :
- 24.8 « L'augmentation de données en cinq lignes » ; 24.E2 « Pourquoi l'augmentation de données améliore-t-elle la généralisation ? … » (le reste du titre inchangé) ;
- 26.E4 « Pourquoi DQN utilise-t-il une mémoire de rejeu et un réseau cible ? » ; 26.31 « Une mémoire de rejeu » (la glose *replay buffer* reste dans l'énoncé) ;
- 23.28 « Rédiger la fiche modèle » ; MP6.7 « README, fiche modèle, tests pytest et nettoyage du dépôt » (et la grille de MP6 : « README et fiche modèle ») ;
- 25.27 « Suréchantillonnage explicite ou convolutions à stride : temps et qualité » ;
- 23.15 « … : découpage stratifié, float32, normalisation, DataLoader » ; MP5.1 « Cadrage, protocole de validation (découpage stratifié) et baselines (MLP puis petit CNN) » ; et « découpage » au lieu de « split » pour le découpage des données dans la prose de MP5, de CP5 et du ch. 23 (R5 de la session 43 réserve *split* au nœud d'un arbre) ;
- 27.R1 « Deux agents qui jouent l'un contre l'autre (auto-apprentissage) » ; 27.Q7 « Pourquoi juger des mini-batches plutôt qu'un point isolé » ;
- 29.R3 : « : » au lieu du tiret cadratin dans le contrat (forme des supports).
- Termes retenus :
  - **carte d'activation** (*feature map*) au lieu de « carte de caractéristiques », partout ;
  - ***on-policy* et *off-policy*** gardés en anglais, glosés « sur politique » et « hors politique » à leur première occurrence (ch. 26) ;
  - ***transfer learning*** gardé en anglais, glosé « apprentissage par transfert » à sa première occurrence (ch. 23) ;
  - **stride** : « convolution(s) de stride 2 » au lieu de « convolution(s) à pas » et de « de pas 2 », là où « pas » désigne un stride (ch. 20, 21, 25, 27, 28 ; pas le « pas » d'un optimiseur ni le « pas de temps ») ;
  - **suréchantillonnage** (pas « sur-échantillonnage ») ;
  - **mini-batches** (pas « mini-batchs ») ;
  - **fine-tuning** (pas « affiner ») : ch. 11, `01_fiche.md:281`.
- Titre du MP6 inchangé (« Banc d'essai génératif… ») : « banc d'essai » est glosé *benchmark* dans la prose, à sa première occurrence, jamais dans un titre.

**R2, CP6** :
- CP6.8 devient un 🔨 « au contrat écrit », comme CP5.6 (énoncé, réponses et vérifications inchangés) ;
- CP6.4 et CP6.9 passent en ★★ ;
- le contrat prend le titre « Réparer un pas d'entraînement de GAN » ;
- `cp6.json` reçoit un champ `notes` sur le modèle de `cp5.json`, qui justifie les types absents : 🗣️ (CP6.12 💼 en tient lieu), 📄 (lire un article demande l'article), et 📦, 🔬, 🛠️, 🔁, 🎨, 🏆 (exclus par le §17) ;
- `study_notes` est réécrit (types réels de l'examen).

**R3, examens** :
- les oracles de code des parties B de CP4, CP5 et CP6 (`oracle_step` de CP4.9, `reference_fit` et l'oracle RNN de CP5, `oracle_8` et `reference_step` de CP6, et toute autre fonction qui contient une solution) passent dans un nouveau module **`src/wb/exams.py`**, testé dans `tests/infra/test_wb_exams.py` ;
- les notebooks d'examen les importent ; leurs messages restent des pistes ;
- les lettres de CP6.9 a et leurs corrections ne doivent plus apparaître en clair dans le notebook (un `wb.check` les vérifie déjà, ou une empreinte) ;
- aucun résultat de vérification ne doit changer (rejoue la partie B avec la solution et avec des erreurs : même verdicts qu'avant).

**R4** :
- dans `.gitignore`, les exceptions `!projets/partie_5_classifieur_cifar10/solution/model.pt` et `!mon_travail/projets/*/model.pt` (vérifie avec `git check-ignore -v`) ;
- le `model.pt` de la solution (300 Ko, présent sur le disque) est à ajouter (le coordinateur commitera) ;
- dans le cahier des charges de MP5, une phrase qui dit que `model.pt` se commite (300 Ko), alors que les autres `*.pt` restent ignorés.

**R5** :
- (A) `syllabus.py stats` exclut le paragraphe « **Contrat commun** … » des énoncés, comme l'encadré « Mode d'emploi » ;
- (B) 24.27 : retirer de l'énoncé et de la docstring la solution de `last_steps_27` (`output[:, -1, :H]`, `h_n[-2]`). Le contrat dit seulement : « pour un LSTM bidirectionnel, le sens avant seulement ; les deux lectures doivent être égales ». Les deux expressions vont à l'indice 3 ;
- (B) 24.24 : retirer de l'énoncé et de la docstring la recette du dessin (coordonnées, bruit, facteur, décalage, épaisseur). Garder le contrat vérifié (image `float32` (size, size) de 0 et 1, label imposé ou tiré, cinq formes distinctes, tout le hasard de `rng`), et mettre la recette dans l'indice 2.
- La règle « un contrat commun à un module s'écrit une fois » s'appliquera à la prochaine reconstruction des autres chapitres : n'y touche pas.

**R6, durées et étoiles** :
- recale les durées (contrat JSON et tous les supports, en-têtes des notebooks compris), en gardant chaque durée dans la borne de son étoile (BIBLE §11 : ★ 5–15, ★★ 15–30, ★★★ 30–90 min). L'étoile est relevée quand la durée l'exige :
  - 21.12 ★★ 30 ; 21.23 ★★ 25 ;
  - 22.12 ★★ 30 ; 22.13 ★★ 30 ;
  - 25.28 ★★★ 35 ;
  - 26.14 ★★ 30 ; 26.19 ★★★ 35 ; 26.20 ★★★ 75 ; 26.21 ★★★ 90 ; 26.31 ★★★ 40.
- MP5 garde 10 h : l'étape MP5.8 (Livraison) gagne 30 min (la somme des étapes fait alors 10 h).
- Puis les en-têtes « Temps total estimé » des fiches (après `syllabus.py build`, lancé par le coordinateur ; signale-le).

**R7, parcours maths** : ajoute `"maths"` aux `tracks` de 23.15, 23.16, 23.18, 23.23, 23.25, 24.23, 27.20 et 27.27.
- La ligne « Parcours » des notebooks et des 02 suit.
- Vérifie qu'aucun de ces exercices ne dépend d'un exercice absent du parcours maths. Si c'est le cas, signale-le ; n'ajoute rien d'autre sans le dire.

**R8, énoncés précisés** (texte exact dans les « Propositions » de F1 à F7 ; réponses inchangées) : 21.Q8 a (sans dropout) ; 22.20 (« un **nouveau** tableau NumPy `float64` de la forme des logits ») ; 24.R1 a, option D ; 26.17 (règle d'honneur sur `Flippers().min_moves`) ; 26.35 (minimax, adversaire parfait) ; 29.15 (les six sections) ; CP6.8 (logits tirés au hasard) ; CP6.7 (1/3 de point par panneau, dans le sujet et le corrigé). **Pas 25.25.**

**R9, conventions de forme** : le coordinateur écrit les §14, §5 et §12 de la bible. À toi :
- les relances des 💼 sur **une ligne**, partout (forme des ch. 12 à 28 : « « … » (réponse) · « … » (…) ») ; seul le ch. 29 est à aligner ;
- l'encadré du ch. 22 « 🧮 Rappel maths — les conventions de ce chapitre » devient « 📐 **Conventions de la fiche** — » ;
- trois tests de format dans `tests/infra/test_syllabus.py` (G7) :
  - (a) l'icône du type du contrat à la fin de chaque en-tête `### Ex N.k — Titre` de `04_indices.md` et de `05_solutions.md` ;
  - (b) le préfixe « Ch. N : » (N = source du rappel), suivi d'une minuscule, dans les en-têtes des rappels de 02, 04, 05 et 06 ;
  - (c) « 🔁 ★ » dans les en-têtes des rappels de `02_exercices.md`.

  Écris-les pour les chapitres où le format est déjà en place : vérifie chapitre par chapitre, de 12 à 29. Si un chapitre ancien ne suit pas le format, corrige-le s'il s'agit de Markdown (02, 04, 05, 06), sinon exclus-le avec une raison.

## Règles

1. **Reprends les règles 3 à 8 de `…/s62/consignes_F.md`** (verrou commun, copies, environnement, rapport).
   - Les notebooks se corrigent par leurs scripts.
   - Ne réexécute pas les solutions dans le dépôt : le coordinateur les réexécute toutes à la fin. Vérifie dans une copie, sous le verrou.
   - Pas de `syllabus.py build`, pas de `build_answers.py` (sauf `--check`), pas de git qui écrit, rien dans `mon_travail/`.
2. Les contrats JSON de ta zone se modifient pour les champs listés ci-dessus (`title`, `type`, `stars`, `minutes`, `tracks`, `notes`, `study_notes`) et nulle part ailleurs.
3. **Zones :**
   - **G1** : ch. 21, 22 ;
   - **G2** : ch. 23, 24 ;
   - **G3** : ch. 25, 26 ;
   - **G4** : ch. 27, 28 ;
   - **G5** : ch. 29, et la prose des ch. 11 et 20 (termes de R1 : `chapitres/ch11_raisonnement/01_fiche.md:281`, `chapitres/ch20_deep_learning/01_fiche.md:183`, `flashcards.csv:23`, champs hors contrat de `ch20.json`), `data/cards/` ;
   - **G6** : CP4, CP5, CP6, MP5, MP6 (`build_cp4.py`, `build_cp5.py`, `build_cp6.py`, `build_mp5.py`, `build_mp6.py`, `cp4.json`, `cp5.json`, `cp6.json`), `src/wb/exams.py` et son test, `.gitignore` ;
   - **G7** : `annexes/` (glossaire : termes retenus, *benchmark*/banc d'essai, découpage), `tests/infra/test_terminology.py` (les motifs prêts dans F7, « Propositions » 3, plus « carte de caractéristiques » et « convolution à pas » ; à n'activer qu'à la fin, après avoir vérifié qu'aucune forme écartée ne reste dans le dépôt ; s'il en reste dans une autre zone, attends ou liste-les), `tests/infra/test_syllabus.py` (R9), `tools/syllabus.py` (R5 A), `src/wb` (sauf `exams.py`).
4. **Rapport** dans `rapport.md` de ton dossier de travail : ce qui est appliqué, décision par décision, avec les fichiers touchés, les vérifications et la mention « réponses enregistrées : aucune ne change ». Ta réponse finale tient en 10 lignes.
