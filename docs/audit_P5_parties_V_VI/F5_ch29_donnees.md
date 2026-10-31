# F5 — ch. 29, data cards et `wb.datasets` : rapport du correcteur

## En-tête

**Fichiers touchés (dépôt)**
- `src/wb/datasets.py` : `make_trap_dataset` (défaut 7), `CIFAR10_HF_REVISION` et `revision=` dans `_cifar_from_huggingface`, message « ≈ 145 Mo », docstrings de `load_cifar10` et `load_pretrained` (Mio).
- `tests/infra/test_wb_trap.py` (+1 test : les labels mal écrits écorchent le vrai fruit, graines 2916 et 7), `tests/infra/test_wb_datasets.py` (+2 tests : empreintes SHA-256 des data cards = fichiers versionnés ; révision CIFAR-10 figée ; `import re`).
- `tools/chapters/build_ch29.py` (titres des rappels avec « Ch. N : », message de 29.Q7 e, auto-évaluation à 3 questions), `tools/chapters/ch29_parts.py` (fil rouge de 29.8, 29.9, 29.10, 29.14 ; prérequis de 29.10 « ch. 20 et 23 »), `tools/chapters/ch29_code.py` (`clean_16` : `"coings"` ; une vérification de plus en 29.16 : chaque label corrigé est le fruit de sa ligne).
- `chapitres/ch29_datasets/` : `01_fiche.md`, `02_exercices.md`, `04_indices.md`, `05_solutions.md`, `06_mes_reponses.md`, `flashcards.csv`, et les deux notebooks reconstruits (`03_notebook.ipynb`, `05_solutions.ipynb`, **sans sorties dans le dépôt** : à réexécuter par le coordinateur).
- `data/cards/` : `README.md`, `adult.md`, `cifar10.md`, `fashion_mnist.md`, `holmes.md`, `images.md`, `mnist.md`, `penguins.md`, `pretrained.md` (réécrit), `rl_envs.md`, `sunspots.md`, `verne.md`.

**Reconstructions** : `python tools/chapters/build_ch29.py` dans le dépôt (13 exercices papier, 67 vérifications, 10 exercices de notebook).

**Vérifications**
- Copie `COPIE/` (tar du dépôt après reconstruction), sous le verrou : `run_all_notebooks.py chapitres/ch29_datasets/05_solutions.ipynb --inplace` → ✅ 0 erreur ; toutes les vérifications ✅ (29.16 compris, avec la nouvelle : « chaque label corrigé est bien le fruit de sa ligne »).
- `build_answers.py --check chapitres/ch29_datasets/05_solutions.ipynb` dans la copie : 88 réponses lues ; **une seule différence, 29.Q7e, « modifiées (messages, métadonnées) »** (le message « une model card » devenu « une fiche modèle »). Comparaison champ par champ de `answers.json` : aucun `hash` ne change. (`--check` sans argument échoue dans la copie sur `ch25_autoencodeurs/05_solutions.ipynb`, cellule 6 sans sortie : notebook reconstruit par F3, hors de ma zone.)
- « Tout exécuter » à vide de `03_notebook.ipynb` dans la copie (sous le verrou) : ✅ jusqu'au bout, 0 ❌, 0 Traceback, 95 ⏳.
- Dépôt : `pytest tests/infra/test_wb_trap.py tests/infra/test_wb_datasets.py tests/infra/test_terminology.py` → 42 passed, 4 skipped (réseau) ; `syllabus.py check` → 0 problème ; `export_flashcards.py --check` → 757 cartes, 31 fichiers, sans avertissement.
- Le dataset piégé, avant et après la correction (`scripts/trap_probe.py`) : graine 2916, les 11 labels invalides sont 7 pommes et 4 poires mal écrites (avant : une « Poire » et un « poire  » sur des pommes, un « pommes » sur une poire) ; `trap_dataset_answers` inchangé (13, 17, acidite, pays, code_qualite, diametre_cm, 11, coing ; 30 manquants codés, 39 diamètres en mm) ; graine 7 : 13, 17, 38, 12 inchangés. Le flux de `rng` n'est pas touché.
- CP6 et MP6 n'appellent ni `make_trap_dataset` ni `trap_dataset_answers` (grep sur tout le dépôt).

**Réponses enregistrées : aucune ne change** (seul le message d'erreur de 29.Q7 e change). Une sortie imprimée change : l'accuracy de la forêt sur les données nettoyées de 29.16, 0,963 → **0,967** (reportée dans `05_solutions.md`) ; les données brutes restent à 0,991.

## Tableau de décision

| Constat (relecteur et numéro) | Décision | Fichiers touchés |
|---|---|---|
| A8 1 (labels faux du dataset piégé) | appliqué : chaque faute écorche le vrai fruit (`pomme` → pome, pommes, POMME ; `poire` → « poire  », Poire, poir ; `coing` → Coing, coings), sans tirage ; `clean_16` connaît `coings` ; vérification de plus en 29.16 ; test d'infrastructure | `datasets.py`, `ch29_code.py`, `test_wb_trap.py`, `05_solutions.md` |
| A8 2, A5 8 (ch. 23 « sur MNIST ») | appliqué : prérequis « ch. 20 et 23 (`Dataset`, `DataLoader`, un classifieur PyTorch sur Fashion-MNIST) » ; liens : « Ch. 20 et 23 … `train_mlp`, le petit réseau de 29.8, 29.9 et 29.14 » | `01_fiche.md` |
| A8 3 (« deux précautions » qui sont trois) | appliqué : « trois points », « Les trois ont pris… » ; les indices 1 de Q2, 29.5 et E4 renvoient aux « trois points du livre » | `01_fiche.md`, `04_indices.md` |
| A8 4 (29.11 contre 29.E4 sur CC BY-SA) | appliqué (texte d'A8) ; flashcard « licence » alignée (CC BY-SA aussi) | `05_solutions.md`, `flashcards.csv` |
| A8 5, A2 16 (chapitres des data cards) | appliqué : CIFAR-10 sans « 28 » ; Fashion-MNIST + ch. 24 (partie A, 6 000 / 54 000 images) et checkpoint V (CP5.6, figure de CP5.7) ; MNIST + ch. 29 et checkpoint VI, 17 remis avant 18 ; Adult + 29.12 ; Penguins + 29.R3, 29.11, 29.12, 29.15 ; aussi images (29.R1) et FrozenLake (29.R2) ; liens de provenance (Toronto, Hub, Zalando, Gutenberg 1661 et 800, SILSO, Gymnasium 1.4.0, torchvision 0.26) | `data/cards/*.md` |
| A8 6, A5 10 (poids sans fiche) | appliqué : `pretrained.md` réécrit, « Réseaux pré-entraînés (poids torchvision) », quatre jeux de poids (MobileNetV3 Small, ResNet-18, SqueezeNet, VGG16), chiffres lus dans `weights.meta` de torchvision 0.26, utilisation par chapitre ; README mis à jour | `pretrained.md`, `README.md` |
| A8 7 (tailles) | appliqué : CIFAR-10 ≈ 145 Mo depuis le Hub (143 646 105 octets, `dataset_info.json`), ≈ 170 Mo depuis Toronto (170 498 071 octets, vérifié par curl), cache ≈ 173 Mo ; poids en « Mio (Mo) » : 4,7 Mio = 5,0 Mo, 44,7 Mio = 46,8 Mo, 9,8 Mio = 10,3 Mo, 528 Mio = 553 Mo | `cifar10.md`, `README.md`, `pretrained.md`, `datasets.py` |
| A8 10, A1 12 (termes retenus au ch. 29) | appliqué pour « fiche modèle » (fiche, Q7 e et son message, 05 E2) et « baseline » (fiche, 29.4, indices, solutions ; le titre de 29.2 n'est pas touché) ; *transfer learning* : reporté (R1) | `01_fiche.md`, `02_exercices.md`, `04_indices.md`, `05_solutions.md`, `build_ch29.py` |
| A8 14 (backticks des flashcards) | appliqué : `<code>…</code>` (cartes 5, 12, 13, 14) | `flashcards.csv` |
| A4 7 (MathJax des flashcards, « même dérive au ch. 29 ») | appliqué : la formule de la mémoire d'un dataset d'images en `\( N \times H \times W \times C \)` (carte 10) ; le chapitre n'a pas d'autre formule | `flashcards.csv` |
| A8 16 (`DataLoader` « ch. 23 ») | appliqué : « (ch. 20 et 23) » ; aussi le prérequis de 29.10 | `01_fiche.md`, `ch29_parts.py` |
| A8 17 (deux taxonomies des fuites) | appliqué : une phrase dans « Ce qui compte » de 29.E3 et dans la flashcard « fuites » (le corps de la réponse en 60 secondes ne s'allonge pas) | `05_solutions.md`, `flashcards.csv` |
| A8 18 (la leçon du ch. 29 appliquée au workbook) | appliqué : `revision=` figée pour CIFAR-10 (vérifiée sur le Hub le 2026-10-31 : c'est la dernière) ; lignes « Empreinte » SHA-256 dans mnist, penguins (deux fichiers), holmes, verne, sunspots, adult, vérifiées par un test ; l'énoncé de 29.15 : proposé (voir plus bas) | `datasets.py`, cards, `test_wb_datasets.py` |
| A8 19 (licence de MNIST sur le Hub) | appliqué (texte d'A8) | `mnist.md` |
| A8 8, 9, 11, 12, 13 (glossaire, erreurs fréquentes, ressources, métiers) | hors zone (F7) | — |
| A8 15 (ligne de PROGRESS) | hors zone (coordinateur) : ligne proposée plus bas | — |
| A2 3 (liens `stable` et `main` du ch. 29) | appliqué : `vision/0.26/datasets.html`, `blob/v0.26.0/…/mnist.py` | `01_fiche.md` |
| A2 4 (Boston : page de la 1.2) | appliqué : notes de la 1.0 (en 1.6) et page de `load_boston` en 1.1 | `01_fiche.md` |
| A2 10, A1 2 (préfixe, ★ et source des rappels) | appliqué : « 29.R1 — Ch. 28 : qui possède… 🔁 ★ ⏱️ 5 min », ligne « *Ch. 28 (…) · fiche §29.1 · parcours R* » ; préfixe dans 02, 04, 05, 06 et la partie 0 des notebooks. Le corps du titre de 29.R3 n'est pas touché (R1) : d'où « Ch. 2 : biais d'échantillonnage : un dataset… » sur les supports Markdown et « … — un dataset… » dans le notebook, jusqu'à la décision R1 | 02, 04, 05, 06, `build_ch29.py` |
| A3 2 (parcours des quiz et des 💼) | appliqué : « · parcours R » aux 8 quiz ; ligne « *Fiche … · parcours R* » aux 4 entretiens | `02_exercices.md` |
| A1 1, A3 3 (icônes des en-têtes de 04) | appliqué : les 16 « ### Ex 29.k » ont l'icône du contrat | `04_indices.md` |
| A1 4, A3 4 (rubriques des 🗣️, ⚖️ et 📄 ; 💼 dans les titres) | appliqué : 29.4 + « Erreurs fréquentes » et « Variante » ; 29.5 et 29.6 : « Réponse », « Erreurs fréquentes » (le contenu de « Ce qui compte » réécrit en erreurs), « Variante » avec sa réponse ; « 💼 » retiré des quatre titres de 05 | `05_solutions.md` |
| A1 5 a (format des relances) | proposé (voir plus bas) : le ch. 29 garde la liste à puces en attendant la règle du §12 | — |
| A1 14, A3 6 (fil rouge des en-têtes) | appliqué : 29.8, 29.9, 29.14 « MNIST », 29.10 « Fashion-MNIST » (notebooks), 29.1 « MNIST » (02) | `ch29_parts.py`, `02_exercices.md` |
| A1 16 (guillemets droits) | appliqué : “recherche seulement” (04, 05), noms de colonnes de 29.12 et 29.16 en backticks | `04_indices.md`, `05_solutions.md` |
| A3 11 (commentaires de code en français) | appliqué : les commentaires des squelettes de 29.7, 29.10, 29.13 et 29.15 en anglais | `04_indices.md` |
| « L'indice 3 de 29.* » (A3 10, relu) | relu, rien à corriger : chaque indice 3 résout le premier item puis pose le critère, ou donne un squelette à `...` ; l'alerte 29.Q4 c (« 10 ») d'A3 vient de « 10 000 » ; le 🔮 29.8 et 29.9 ne donnent pas l'issue | — |
| A3 20 (quatre questions d'auto-évaluation) | appliqué : la 4ᵉ devient « **Avant le checkpoint** : choisis le dataset de ton projet final… (🛠️ 29.15) » | `build_ch29.py` |
| A3 22 (aide-mémoire PyTorch sans le ch. 29) | hors zone (F7) | — |
| A3 24, A1 6 (titre de 29.R3) | reporté (R1) | — |
| A1 9, A8 10 (*transfer learning*) ; A1 6 (« banc d'essai », fiche l. 48) | reporté (R1) | — |
| A1 18 (encadré 📐 hors du §14) | hors zone (règle du §14, à trancher par le coordinateur) ; l'encadré du ch. 29 a déjà la forme « 📐 **Conventions de la fiche** — » | — |
| A11 9 (téléchargements des parties V et VI) | hors zone (F7) ; la taille de CIFAR-10 à y écrire : ≈ 145 Mo depuis le Hub | — |

## Propositions (à valider, non appliquées)

1. **Énoncé de 🛠️ 29.15** (A8 18), dernière phrase du paragraphe « Puis remplis `CARD_15` » : remplacer « Modèle : les fiches de `data/cards/`. » par « Modèle : les six sections ci-dessus ; les fiches de `data/cards/` montrent le niveau de détail attendu (elles n'ont pas toutes ces six titres). » (`tools/chapters/ch29_parts.py`, puis reconstruction). Raison : `check_card` exige six sections que les fiches du workbook n'ont pas ; un apprenant qui recopie `cifar10.md` échoue.
2. **Format des relances des 💼** (A1 5 a) : écrire au §12 « Relances possibles : une liste à puces, une relance par ligne, « question » → réponse » (forme des ch. 27 à 29, plus lisible), et l'appliquer aux ch. 12 à 26 à leur prochaine reconstruction. Le ch. 29 est déjà à ce format.
3. **Data cards et six sections** (A8 18, suite) : si la proposition 1 n'est pas retenue, ajouter une section « Téléchargement » (source, révision, empreinte) aux fiches de `data/cards/`, pour qu'elles suivent le modèle qu'enseigne le ch. 29.

## Pour une autre zone

- **F6** (MP5) : `projets/partie_5_classifieur_cifar10/depart/data.py:26` et `solution/data.py:26` (« downloads about 170 MB ») et `README.md:116` (« environ 170 Mo ») : « about 145 MB from Hugging Face (170 MB from Toronto) » / « environ 145 Mo depuis le Hugging Face Hub ».
- **F7** (`00_setup/demo.ipynb:21` et `:842`) : « CIFAR-10 (≈ 145 Mo depuis Hugging Face) ». Pour A11 9, les tailles exactes : CIFAR-10 ≈ 145 Mo (Hub) ou 170 Mo (Toronto), cache 173 Mo ; ResNet-18 46,8 Mo ; MobileNetV3 Small 10,3 Mo ; SqueezeNet 5,0 Mo ; VGG16 553 Mo (`data/cards/pretrained.md`).
- **F2** (`tools/chapters/build_ch24.py:2457`) : « le premier chargement télécharge 170 Mo » → « environ 145 Mo ».
- **F7** (`annexes/cheatsheets/pandas.md:112`) : rien à changer, la règle « chaque valeur n'apparaît que pour une classe » est maintenant vraie sur `FRUITS_16`.
- **F7** (glossaire, A8 9) : les termes du ch. 29 restent « règlement européen sur l'IA » et « RGPD » dans la prose ; l'entrée du glossaire proposée par A8 9 leur convient.
- **Coordinateur** : `src/wb/datasets.py` a un nouveau nom public, `CIFAR10_HF_REVISION` ; `load_cifar10` lit d'abord `cifar10.npz` : le cache existant ne change pas.

## Lignes proposées

**BIBLE §22**
| 2026-10-31 | **Audit P5 V-VI, F5 (ch. 29, data cards, `wb.datasets`)** : le défaut 7 de `make_trap_dataset` écorche le vrai fruit (les labels mal écrits se réparent par normalisation ; le nettoyage de référence introduisait trois erreurs de label), sans tirage, donc à réponses inchangées, et un test le garde ; CIFAR-10 se charge du Hub à une révision figée (`CIFAR10_HF_REVISION`) ; les data cards donnent l'empreinte SHA-256 de leurs fichiers versionnés, vérifiée par un test ; tailles en « Mio (Mo) » (1 Mo = $10^6$ octets, ch. 29) ; `pretrained.md` couvre tous les poids torchvision du workbook. | A8 1, 5, 6, 7, 18, 19 ; A2 16 ; A5 10 ; BIBLE §9. |

**PROGRESS, « Écarts »**
- ch. 29 : aucun écart de contrat ; 29.Q7 e (« une fiche modèle » au lieu d'« un *model card* ») et 29.4 (« une baseline simple à battre ») reformulés sans changer le sens ; en-têtes des rappels préfixés « Ch. N : ».

**PROGRESS, l. 1101** (A8 15) : « | ch. 29 | `wb.datasets.make_trap_dataset` et `trap_dataset_answers` (huit défauts plantés) | 🏆 29.16 ; ✅ fait en session 60 (défaut 7 corrigé en session 62) | ».

**Temps de lecture** : la fiche a pris quelques mots (≈ 40) : `syllabus.py build` puis l'en-tête « Temps total estimé » (≈ 1,1 h de lecture, inchangé à l'arrondi probable).
