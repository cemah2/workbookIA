# G6 : checkpoints III et IV, MP3 et MP4. Application de R1 à R11

**Réponses enregistrées dont la valeur ou le message change : aucune.** Aucun `wb.record` ni aucun message de `mistakes` n'a été touché. Les sorties des deux notebooks de solutions des checkpoints sont identiques à celles de HEAD (comparées cellule par cellule, durées exceptées). Celles de MP4 aussi, à une durée affichée près (« en 13 s » devient « en 16 s »). `build_answers.py` n'a pas été lancé.

## Ce qui a été fait

| Proposition (ID ou fichier) | Ce qui a été fait | Fichiers touchés |
|---|---|---|
| R4 (titres de 8.10, 16.10, 17.10, 18.11, 19.12, 20.12, 20.7, 20.30, 20.E2, 18.Q11) | Ma zone ne cite aucun de ces titres : elle ne cite que des ID (remédiations). Rien à changer. Les titres courts des questions de CP3 et CP4 restent. « Becker, B. & Kohavi, R. (1996) » (référence APA de la data card) et « *Restart & Run all* » (nom d'un menu) restent. | — |
| R5 : variable de substitution (*proxy*) | CP4.11, corrigé : « jamais sur un substitut d'une caractéristique protégée » devient « jamais sur une variable de substitution (*proxy*) d'une caractéristique protégée, comme le code postal ». MP3 : « indicateurs indirects » devient « variables de substitution (*proxy*) » dans les cinq documents. Ce sont des documents autonomes, chacun avec une seule occurrence, d'où la glose dans chacun. | `checkpoints/partie_4/03_examen_corrige.md`, `projets/partie_3_revenus/depart/DATA_CARD.md`, `depart/README.md`, `solution/DATA_CARD.md`, `solution/MODEL_CARD.md`, `solution/README.md` |
| R5 : vérification du gradient | Carte Mermaid de la synthèse de CP4 : « gradient check » devient « vérification du gradient ». | `checkpoints/partie_4/05_synthese.md` |
| R5 : baseline | Synthèse de CP3, tableau des familles : « une référence rapide à écrire » (kNN), « une référence » (Naïve Bayes) et « la référence à battre » (régression logistique) deviennent « baseline ». Ont été vérifiées et gardées : « solution de référence », « repli sur la référence », « librairie de référence », « des références vers les poids », et « le kNN, référence des embeddings », qui ne désigne pas une baseline. | `checkpoints/partie_3/05_synthese.md` |
| R5 : layer norm | `cp4.json`, extension de MP4 : « Connexions résiduelles et LayerNorm dans minifw » devient « … et layer norm dans minifw ». Le README et le notebook disaient déjà « layer norm ». | `docs/syllabus/data/cp4.json` |
| R5 : passe avant et passe arrière | Titre de MP4.2 (ID inchangé), sur le contrat, la table des étapes et l'en-tête de l'étape dans les deux notebooks : « (forward/backward) » devient « (passes avant et arrière) ». Brief de MP4 : « qui exposent forward/backward » devient « qui ont chacune leur passe avant et leur passe arrière (forward, backward) ». README de la solution, deux commentaires du schéma de boucle : « # forward, … » et « # backward, … » deviennent « # passe avant, … » et « # passe arrière, … ». Les noms de méthodes (`forward`, `backward()`, `loss.backward()`), les docstrings et les messages en anglais restent. | `docs/syllabus/data/cp4.json`, `tools/chapters/build_mp4.py`, `projets/partie_4_mini_framework/solution/README.md` |
| R5 : Naïve Bayes, couche dense, split, scaler, blob | Ma zone n'a aucune forme écartée : elle écrit déjà partout « Naïve Bayes » et « couche dense », et n'emploie ni « blob » ni « couche pleine ». Les gloses de split et de scaler vont aux ch. 12 et 13, pas aux checkpoints. | — |
| R6 : CP4.3 ★ devient ★★ | Contrat (`"stars": 2`, et la note des étoiles complétée : « CP4.3 passe de ★ à ★★ à l'audit P5 des parties III et IV (R6), comme CP3.5 : f et g combinent Adam et weight decay, h est une interprétation »), puis l'en-tête de CP4.3 dans le sujet. Le corrigé, la copie, le README et les notebooks n'affichent pas les étoiles de CP4.3, donc rien à y changer. `syllabus.py check` : 0 problème, donc les règles du §17 passent. | `docs/syllabus/data/cp4.json`, `checkpoints/partie_4/01_examen_sujet.md` |
| R9 : CP3.6 c) | Même phrase neutre au sujet et au notebook : « garde la transformation de la ligne (B) telle quelle (la valeur attendue à la question d est calculée avec elle) et retire les fuites ». Le notebook ne dit plus que (B) n'est pas une fuite. Le message de `mistakes` de CP3.6d (partie B, après l'examen) reste inchangé. | `checkpoints/partie_3/01_examen_sujet.md`, `tools/chapters/build_cp3.py` |
| R9 : CP4.9 a) | Sujet et notebook : « Elle ne modifie pas `grads`, à la différence de `clip_grad_norm` (ch. 19) et de `torch.nn.utils.clip_grad_norm_`, qui modifient les gradients sur place. » Dans le sujet, « (la règle de `clip_grad_norm_`) » devient « (le coefficient de `clip_grad_norm_`) », pour ne pas se contredire. Le corrigé le disait déjà, dans ses erreurs fréquentes. | `checkpoints/partie_4/01_examen_sujet.md`, `tools/chapters/build_cp4.py` |
| R1, R2, R3, R7, R8, R10, R11 | Hors de ma zone. Ma zone ne cite ni les titres ni les durées de 13.24, 13.32, 14.24, 16.19, 18.16, 20.15, etc. : seulement des ID de remédiation. | — |

## Pour une autre zone

- **Coordinateur, `suivi/PROGRESS.md` § Écarts** (à toi d'ajuster le texte) : « CP4.3 : ★ → ★★ (audit P5 III-IV, R6), comme CP3.5 : f et g combinent Adam et weight decay, h est une interprétation. MP4.2 : titre « (forward/backward) » → « (passes avant et arrière) » (R5), ID inchangé. »
- **Coordinateur** : `python tools/syllabus.py build` régénère `docs/SYLLABUS.md` (étoiles de CP4.3, titre de MP4.2, brief de MP4), `docs/PARCOURS.md` et `suivi/tableau_de_bord.md`. `test_generated_documents_are_up_to_date` échoue tant que ce build n'est pas fait, et c'est normal.
- **Une remarque, sans changement** : dans `projets/partie_4_mini_framework/README.md` (l. 58), la table des étapes donne pour MP4.2 un libellé plus court que le titre du contrat (« Les couches `Dense`, `Activation`, `Dropout` et `BatchNorm1d`, sur ta librairie, comparées à PyTorch »). C'était déjà le cas avant ces corrections, je l'ai laissé.

## Scripts reconstruits et notebooks réexécutés

- `build_cp3.py`, `build_cp4.py` et `build_mp4.py` ont été relancés. `build_mp4.py` régénère `depart/minifw/` à l'identique : aucun diff. `build_mp3.py` n'a pas été modifié ni relancé, donc le coffre de MP3 n'a pas été rouvert. Aucun mode complet n'a été lancé.
- Réexécution unique, en mode rapide, avec `--inplace` :
  - `checkpoints/partie_3/03_examen_solutions.ipynb` ✅ (6,3 s) : 0 ❌, 0 ⏳, 0 traceback ; sorties identiques à HEAD.
  - `checkpoints/partie_4/03_examen_solutions.ipynb` ✅ (3,9 s) : de même.
  - `projets/partie_4_mini_framework/solution/mp4_mini_framework.ipynb` ✅ (64 s) : 69 ✅, 0 ❌, 0 ⏳, 0 traceback. Le test reste fermé (⏸️, mode rapide). `results.json`, `test_log.json` et les figures ne bougent pas.
- « Tout exécuter » à vide, dans une copie (`G6/copy/workbookia`, `WB_ROOT` de la copie, `start_chapter.py --init`, puis `CP3` et `CP4`) : les notebooks `02_examen_notebook.ipynb` de CP3 et de CP4 et `mp4_mini_framework.ipynb` vont tous au bout. Ils n'affichent que les ⏳ attendus (« pas encore fait », « il manque main_model », README encore en TODO), sans ❌ ni traceback.
- Autres vérifications :
  - `python tools/syllabus.py check` : 46 chapitres, 2024 exercices, 0 problème. Il n'affiche aucun total par parcours, donc je ne vois pas les totaux de R7 et R8.
  - `tests/infra/test_terminology.py` et `tests/infra/test_projects.py` : 27 passed.
  - `tests/infra/test_repo_structure.py` : 50 passed.
