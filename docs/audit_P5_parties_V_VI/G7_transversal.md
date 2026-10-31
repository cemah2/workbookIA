# G7 (transversal) — R1 (glossaire, `test_terminology.py`), R5 A (`syllabus.py stats`), R9 (tests de format)

## En-tête

**Fichiers touchés**
- `annexes/glossaire.md` (R1) ;
- `tests/infra/test_terminology.py` (R1, motifs activés en dernier) ;
- `tools/syllabus.py` et `tests/infra/test_syllabus.py` (R5 A : `stats` ; R9 : trois tests de format) ;
- `chapitres/ch21_cnn/02_exercices.md`, `04_indices.md`, `05_solutions.md`, `06_mes_reponses.md` : une ligne chacun, l'en-tête de 21.R3 (R9 b ; Markdown seulement, le notebook écrit le titre du contrat sans préfixe) ;
- `src/wb/__init__.py` : une ligne de la docstring du paquet cite `wb.exams` (suggestion de G6).

**Reconstructions** : aucune (aucun notebook ni script de chapitre touché).

**Vérifications**
- `pytest tests/infra/test_syllabus.py` : les trois tests de format et le test de `stats` passent ; seuls échouent `test_generated_documents_are_up_to_date` (SYLLABUS, PARCOURS, tableau de bord : `syllabus.py build`, coordinateur).
- `syllabus.py check` : 0 problème ; `build_answers.py --check` : ❌ pour la seule raison des notebooks à réexécuter (ch. 21, hors zone).
- `syllabus.py stats` avant et après R5 A : voir plus bas.
- `pytest tests/infra` (sous le verrou) : 521 passed, 5 skipped, 9 failed, détaillés dans « État final » (aucun dans ma zone : documents générés, notebooks à réexécuter, et dix formes écartées dans d'autres zones).
- `git status --short mon_travail` : vide.

**Réponses enregistrées : aucune ne change** (aucune cellule de notebook ni `answers.json` touchés).

## Tableau de décision

| Décision | Ce qui est appliqué | Fichiers touchés |
|---|---|---|
| R1, carte d'activation | entrée « carte d'activation » (*feature map, activation map* ; « carte de caractéristiques » écarté, en 2ᵉ colonne) | `annexes/glossaire.md` |
| R1, *on-policy* / *off-policy* | entrée inversée : retenu « on-policy, off-policy », 2ᵉ colonne « sur politique, hors politique » ; définition précisée (exploration comprise pour SARSA ; Q-learning apprend la politique gloutonne) | idem |
| R1, *transfer learning* | entrée inversée : retenu « transfer learning », 2ᵉ colonne « apprentissage par transfert » | idem |
| R1, *benchmark* et « banc d'essai » | « benchmark » : jeu de test standard et public (ImageNet), renvoi à « banc d'essai » ; nouvelle entrée « banc d'essai » (*benchmark*, MP6) : le montage d'une comparaison, protocole fixé d'avance ; la glose *benchmark* à sa première occurrence dans la prose, jamais dans un titre. Section « Termes des checkpoints et des mini-projets » renommée « (parties I à VI) » | idem |
| R1, découpage | nouvelle entrée « découpage (des données) » (*data split*, ch. 8 : stratifié, par groupes, temporel ; « split » réservé au nœud d'un arbre, ch. 13). L'entrée « split » disait déjà « rien à voir avec le découpage entraînement / test » | idem |
| R1, stride | définition : « on dit « une convolution de stride 2 » ; le mot « pas » reste au pas d'un optimiseur et au pas de temps d'une séquence » (l'entrée DCGAN disait déjà « convolutions de stride 2 ») | idem |
| R1, suréchantillonnage | rien à faire : une seule entrée, « suréchantillonnage » | — |
| R1, mini-batches | définition : « au pluriel, « des mini-batches » (comme « des batches ») » | idem |
| R1, fine-tuning | 2ᵉ colonne « réglage fin, affinage, affiner » ; définition : « on dit « faire du fine-tuning » ou « un modèle qui a reçu un fine-tuning » (le verbe français de la 2ᵉ colonne est écarté) » | idem |
| R1, formulaire, erreurs fréquentes, cheatsheets | relus : aucune forme écartée (« split » n'y désigne que le nœud d'un arbre ou un identifiant : `cv.split`, `train_test_split`, `split="test"` ; « stride » partout ; « mini-batch(es) » ; aucune carte de caractéristiques, *on-policy*, *transfer learning*) : rien à changer | — |
| R1, `test_terminology.py` | motifs ajoutés en dernier : voir « Motifs ajoutés » et « État final » | `tests/infra/test_terminology.py` |
| R5 A | `statement_words` exclut le paragraphe qui commence par « **Contrat commun » (jusqu'à la ligne vide suivante), comme l'encadré « Mode d'emploi » ; la légende du tableau le dit ; le test de `stats` a un paragraphe « Contrat commun » sur deux lignes, non compté | `tools/syllabus.py`, `tests/infra/test_syllabus.py` |
| R9 a | `test_exercise_headers_of_hints_and_solutions_end_with_the_icon_of_their_type` : tous les chapitres publiés (0A à 29) | `tests/infra/test_syllabus.py` |
| R9 b | `test_recall_headers_name_their_source_chapter_then_a_lowercase` : à partir du ch. 12 ; un seul en-tête hors format, 21.R3 « 0B : » → « Ch. 0B : » (comme 16.R3), corrigé dans les quatre Markdown | `tests/infra/test_syllabus.py`, `chapitres/ch21_cnn/0[2456]_*.md` |
| R9 c | `test_recall_headers_of_the_exercises_show_their_stars_and_minutes` : tous les chapitres publiés ; chaque 🔁 du contrat a son en-tête dans 02, qui finit par « 🔁 ★ ⏱️ 5 min » (étoiles et minutes du contrat) | `tests/infra/test_syllabus.py` |

### R5 A : `syllabus.py stats`, avant et après

| Partie | 🔨 mots avant | 🔨 mots après | Le plus long, après |
|---|---|---|---|
| IV · Réseaux | 251 | 215 | 18.19 (329) |
| V · Architectures | 310 | 200 | 24.30 (477, un 📦) |
| VI · Génératif et RL | 313 | 227 | 27.20 (389) |

Les 📦 ne changent pas (aucun n'a de « Contrat commun ») ; les parties 0 à III non plus. Le ch. 20 (partie IV) avait déjà un contrat commun. L'audit donnait 152 mots pour la partie V sans le contrat commun : il comptait le contrat **propre** seulement ; `stats` garde, lui, tout le reste de l'énoncé (consignes de vérification, « Dans tes notes »). Les chiffres sont mesurés sur les notebooks du dépôt au moment du calcul (avant la reconstruction de 24.24 et 24.27 par G2, R5 B).

### R9 : vérification chapitre par chapitre

| Format | Ch. 12 à 20 | Ch. 21 à 29 | Avant le ch. 12 |
|---|---|---|---|
| (a) icône à la fin de « ### Ex N.k — Titre » (04, 05) | conformes | conformes (corrigés par F1 à F5) | conformes : le test les couvre |
| (b) « Ch. N : » + minuscule (02, 04, 05, 06) | conformes (16.R3 : « Ch. 0B : ») | 21.R3 corrigé (« 0B : » → « Ch. 0B : ») | ch. 0B à 7 : « 0A : », « 0B : » pour un rappel de la partie 0 (forme antérieure à la session 43) : exclus, raison écrite dans le test (`RECALL_PREFIX_FROM = "12"`) ; ch. 8 à 11 conformes |
| (c) « 🔁 ★ ⏱️ 5 min » (02) | conformes | conformes | conformes : le test les couvre |

Le test (b) accepte une majuscule seulement pour un nom à plusieurs capitales (« VGG16 », « PyTorch ») ; aucun cas aujourd'hui.

## Motifs ajoutés à `FORBIDDEN`

Entrées de `FORBIDDEN` (`tests/infra/test_terminology.py`), ajoutées après 50 minutes d'attente et de nouvelles recherches toutes les 4 à 5 minutes (de 11 h 33 à 12 h 25) :

| Terme retenu | Formes refusées | Exceptions (sens courant) |
|---|---|---|
| mini-batches | « mini-batchs » | — |
| stride | « convolution(s) à pas », « convolution(s) (transposée(s)) de pas 2 » (« un pas de 2 » n'est pas refusé : c'est aussi le pas d'un optimiseur ou d'une fenêtre) | — |
| suréchantillonnage | « sur-échantillonn… » | — |
| fine-tuning (affiner) | « affiner », « affiné(e)(s) », « affinage », « affinons », « affinez », « affinent », « affinera(it) » (« affine », « affines » restent : la fonction affine) | « affiner encore la grille » (ch. 26), « affinent le dosage » (ch. 28), « « affinage des données » » (ch. 15, le *data refinement* du livre), « s'affinent ensuite » (README de la solution de MP6) |
| carte d'activation | « carte(s) de caractéristiques », « carte(s) de features » | — |
| découpage (des données) | « split(s) » suivi de « stratifié », « temporel », « aléatoire », « chronologique », « par groupes », « de validation », « des données », « train/ », « entraînement/ » (avec ou sans gras, ou « non » devant), et « des deux côtés du split » | — |

**« split » : un motif sûr, mais partiel.** « le split », « un split », « même split », « du split » désignent aussi le nœud d'un arbre (≈ 50 fois aux ch. 13 et 14) : on ne peut pas les refuser. Le motif ne refuse que les formes qui ne peuvent pas être un nœud ; « sur le même split » au sens du découpage passerait (il n'en reste aucun : G6 les a corrigés). Les identifiants (`train_test_split`, `stratified_split`) et l'anglais du code ne sont pas touchés (`\b` ne coupe pas `_split`, et le motif demande un adjectif français).

Le test des exemples vérifie chaque nouveau motif sur une phrase (« 312 mini-batchs pleins », « convolutions à pas », « split stratifié », « affiné ») et deux phrases permises (« une convolution de stride 2 ; un pas de l'optimiseur », « le meilleur split d'un nœud ; `train_test_split` stratifié », « une fonction affine »).

Non ajoutés (hors de la demande, et trop larges aujourd'hui) : les formes anglaises des termes retenus en français, « model card » (104 occurrences, MP3 compris), « self-play » (20), « replay buffer » (6), « data augmentation » (9), presque toutes dans des champs hors contrat des JSON et des documents générés ; « sur politique / hors politique » et « apprentissage par transfert » sans glose : propres dans les chapitres publiés, mais un titre de `chB8.json` (« … apprentissage hors politique ») et une notion de `chB1.json` les emploient (bonus pas encore publiés). À ajouter quand ces fichiers seront alignés : `"on-policy, off-policy": (r"\b[Ss]ur politique\b|\b[Hh]ors politique\b", [])`, `"transfer learning": (r"\b[Aa]pprentissage par transfert\b", [])`.

## État final

`pytest tests/infra` (sous le verrou, 12 h 30) : **521 passed, 5 skipped, 9 failed**.

- `test_syllabus.py::test_generated_documents_are_up_to_date` × 3 (SYLLABUS, PARCOURS, tableau de bord) : `syllabus.py build`, coordinateur.
- `test_repo_structure.py::test_answers_json_is_valid_and_up_to_date` : « `ch21_cnn/05_solutions.ipynb` cellule 6 : cellule taguée answer sans ligne WB_ANSWER » ; les notebooks reconstruits par G1 à G6 attendent leur réexécution (coordinateur), puis `build_answers.py`.
- `test_terminology.py` × 5 (`stride`, `suréchantillonnage`, `mini-batches`, `découpage (des données)`, `carte d'activation`) ; ce qui reste, en trois groupes :
  1. **les documents générés** (`docs/SYLLABUS.md`, `suivi/tableau_de_bord.md`) : ils disparaissent avec `syllabus.py build`, sauf ce que donnent les deux JSON du point 2 (vérifié en construisant SYLLABUS, PARCOURS et le tableau de bord en mémoire : il ne resterait que les lignes 3778, 3790 et 5210 de SYLLABUS) ;
  2. **des champs hors contrat que la règle 2 interdisait aux correcteurs** : `ch23.json` `objectives[0]`, `concepts[12]`, `teaches[2]` (« split stratifié », déjà signalé par G2) et `chB5.json:193` (« sur-échantillonnage », signalé par F7) ;
  3. **des fichiers d'autres zones que leurs correcteurs n'ont pas corrigés** (G5 et G6, qui ont fini ou finissent) : ch. 20 « carte de features » (2), CP5 « des deux côtés du split » (2), MP5 « split (non) stratifié » (2), MP6 « convolutions (transposées) de pas 2 » (4).
  Les corrections exactes sont dans « Pour une autre zone ». Une fois faites, puis `syllabus.py build`, les cinq tests passent (aucune exception à ajouter : toutes ces occurrences sont des formes écartées, pas des sens courants).
- Tout le reste de `tests/infra` est vert, dont les trois tests de format (R9) et le test de `stats` (R5 A).

## Pour une autre zone

Les occurrences que `test_terminology.py` refuse maintenant (texte exact) :

- **Coordinateur (ch. 20, Markdown seulement, aucun notebook)** :
  - `chapitres/ch20_deep_learning/04_indices.md:505` : « 4. Toutes les positions d'une carte de features partagent les statistiques. » → « … d'une carte d'activation … » ;
  - `chapitres/ch20_deep_learning/05_solutions.md:175` : « toutes les activations d'une même carte de features » → « … d'une même carte d'activation ».
- **G6 / coordinateur (CP5, MP5)** :
  - `checkpoints/partie_5/03_examen_corrige.md:165` et `:167` : « des deux côtés du split » → « des deux côtés du découpage » ;
  - `projets/partie_5_classifieur_cifar10/solution/README.md:28` : « un split non stratifié » → « un découpage non stratifié » ;
  - `projets/partie_5_classifieur_cifar10/solution/MODEL_CARD.md:15` : « (split stratifié, graine 0) » → « (découpage stratifié, graine 0) ».
- **G6 / coordinateur (MP6)** :
  - `tools/chapters/build_mp6.py:801` (message de la vérification de MP6.4) : « convolutions de pas 2 » → « convolutions de stride 2 », puis `build_mp6.py` (un message, aucune réponse enregistrée) ;
  - `projets/partie_6_generatif/solution/MODEL_CARD.md:5` « un encodeur à deux convolutions de pas 2 » → « … de stride 2 » ; `:6` « convolutions transposées de pas 2 » et « convolutions de pas 2 » → « … de stride 2 » ;
  - `projets/partie_6_generatif/solution/README.md:27` : « deux convolutions de pas 2 », « convolutions de pas 2 et transposées » → « … de stride 2 … ».
- **Coordinateur (champs hors contrat)** :
  - `docs/syllabus/data/ch23.json` : `objectives[0]` « …, découpage stratifié, DataLoader, graines » ; `concepts[12]` « découpage stratifié » ; `teaches[2]` « normalisation et découpage stratifié » (aucun `requires` ne cite ces textes) ;
  - `docs/syllabus/data/chB5.json:193` : « convolutions, convolution transposée, suréchantillonnage » ;
  - puis `syllabus.py build`.
- **Coordinateur (R6)** : après `syllabus.py build`, les en-têtes « Temps total estimé » des fiches (R6 change des durées ; les temps de lecture aussi, puisque des fiches ont changé).
- **Pour plus tard** (bonus) : `chB8.json:307` (titre « … apprentissage hors politique » → « … apprentissage off-policy »), `chB1.json:61` (« transfer learning (*apprentissage par transfert*) »), puis les deux motifs donnés dans « Motifs ajoutés ».

## Lignes proposées

**BIBLE §22**
- | 2026-10-31 | **Termes des parties V et VI (R1)** : carte d'activation (*feature map* ; « carte de caractéristiques » écarté) ; *on-policy* et *off-policy*, *transfer learning* gardés en anglais (glosés « sur politique », « hors politique », « apprentissage par transfert » à leur première occurrence) ; « banc d'essai » (le montage d'une comparaison, MP6), glosé *benchmark* dans la prose, jamais dans un titre ; *benchmark* reste le jeu de test public ; « découpage » des données (*split* : le nœud d'un arbre) ; « convolution de stride 2 » ; suréchantillonnage ; mini-batches ; fine-tuning (pas « affiner »). `test_terminology.py` refuse « mini-batchs », « convolution(s) (transposée(s)) à pas / de pas 2 », « sur-échantillonn… », « affiner » (quatre sens courants en exception), « carte(s) de caractéristiques / de features » et les formes sans ambiguïté de « split » pour le découpage (« split stratifié, temporel, aléatoire… », « des deux côtés du split ») ; « le split », « même split » restent permis (nœud d'arbre, ch. 13 et 14). | Audit P5 V-VI, R1 (A1 6 à 10, 13). |
- | 2026-10-31 | **`syllabus.py stats` sans le contrat commun (R5 A)** : la longueur d'un énoncé 🔨/📦 ne compte plus le paragraphe « **Contrat commun** … » de son module (recopié dans chaque 🔨), comme l'encadré « Mode d'emploi ». | A3 12. |
- | 2026-10-31 | **Formats des supports vérifiés (R9)** : `test_syllabus.py` vérifie (a) l'icône du type du contrat à la fin de chaque « ### Ex N.k — Titre » de 04 et 05 (tous les chapitres publiés) ; (b) « N.Rk — Ch. M : » (M = source du rappel) suivi d'une minuscule dans 02, 04, 05 et 06, à partir du ch. 12 (les ch. 0B à 7 écrivent « 0A : », « 0B : ») ; (c) chaque rappel du contrat a son en-tête dans 02, qui finit par « 🔁 ★ ⏱️ 5 min ». | A1 1 et 2 ; troisième dérive évitée. |

**PROGRESS, « Écarts »**
- « R9 (session 62, G7) : en-tête de 21.R3 « Ch. 0B : » dans 02, 04, 05 et 06 (forme de 16.R3). »
