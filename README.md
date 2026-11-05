# Workbook Deep Learning : de la théorie à la pratique

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cemah2/workbookIA/blob/main/00_setup/demo.ipynb)

Un cahier d'exercices complet, en français, pour étudier le livre **« Deep Learning: From Basics to Practice »** d'Andrew Glassner (2018, 2 volumes), le mettre à jour (2026) et le compléter par huit chapitres bonus modernes (transfer learning, Transformers, LLM, diffusion, explicabilité, mise en production, RL moderne).

Objectif : une **reconversion professionnelle** vers les métiers de l'IA et de la data. Chaque chapitre apporte de la compréhension (intuition, maths, code), une librairie construite de ses propres mains (`mylearn`), des réflexes professionnels (tests, git, reproductibilité) et des questions d'entretien. Le workbook part du niveau lycée en maths et du niveau débutant en Python : les chapitres 0A et 0B posent tout ce qui sert ensuite.

> Le livre n'est pas inclus dans ce dépôt (droits d'auteur). Les exercices et les fiches sont originaux ; ils renvoient aux sections du livre sous la forme « (livre §18.5) ». Voir [Crédits et licences](#crédits-et-licences).

**Pour un recruteur** : ce dépôt montre un parcours complet, du calcul à la main jusqu'au modèle livré. Le plus parlant : les [six mini-projets de portfolio](#portfolio--mini-projets-et-projet-final), la librairie [`mylearn`](#ta-librairie-mylearn) écrite en NumPy et testée contre scikit-learn et PyTorch, et le [projet final](projets/projet_final/).

## En chiffres

<!-- wb:readme chiffres -->
| | |
|---|---|
| **Exercices** | **2 028** : 1 933 dans les 39 chapitres (0A, 0B, 1 à 29, B1 à B8), 82 dans les 6 examens blancs, 13 étapes du projet final |
| **Par type** | 🧠 448 · 🔁 114 · ✏️ 250 · ∂ 71 · 🔨 197 · 📦 191 · 🔬 116 · 🔮 87 · 🐛 50 · 📈 47 · 🧮 37 · 🗣️ 45 · ⚖️ 39 · 📄 34 · 🎨 27 · 🏆 39 · 💼 190 · 🛠️ 46 |
| **Vérifications automatiques** | 5 671 réponses vérifiées par `wb.check` (empreinte, jamais la réponse en clair) |
| **Tests** | **5 855 tests** pytest : 5 003 pour `mylearn` (ton code, comparé à NumPy, scikit-learn ou PyTorch) et 852 pour l'outillage du workbook |
| **Notebooks** | 107 : 78 dans les chapitres (exercices et solutions exécutées), 12 pour les examens blancs, 15 pour les mini-projets et le projet final, 2 pour la mise en place |
| **Flashcards** | 967 cartes en 39 sous-paquets Anki |
| **mylearn** | 38 modules, 273 fonctions et classes publiques aux signatures figées |
| **Index et glossaire** | 1 565 notions et 1 602 renvois, 1 261 termes au glossaire |

*Compté le 2026-11-05 par `python tools/repo_stats.py --write` : exercices du contrat (`docs/syllabus/data/*.json`), réponses de `src/wb/answers.json`, cartes acceptées par `export_flashcards.py --check`, tests de `pytest --collect-only --impl=ref`, fichiers `.ipynb` hors `mon_travail/`, stubs de `MANIFEST.json`, en-tête de l'index.*
<!-- wb:end chiffres -->

---

## 🚀 Démarrage rapide sur Google Colab (recommandé)

1. Clique sur le badge **Open in Colab** ci-dessus (notebook `00_setup/demo.ipynb`).
2. Lance la première cellule (**setup**) et autorise l'accès à Google Drive. Le dépôt est cloné dans `Mon Drive/workbookIA` ; aux ouvertures suivantes, la même cellule fait un `git pull` pour récupérer les mises à jour.
3. Menu **Exécution → Tout exécuter** : tout doit aller jusqu'au bout.

Guide détaillé (GPU, sauvegarde de ton travail, mises à jour, problèmes fréquents) : [`00_setup/COLAB.md`](00_setup/COLAB.md).

## 💻 Démarrage en local (ordinateur sans GPU)

Pas à pas pour grand débutant, Windows, Mac et Linux : [`00_setup/INSTALL_LOCAL.md`](00_setup/INSTALL_LOCAL.md). En résumé (Linux/macOS, Python 3.13) :

```bash
git clone https://github.com/cemah2/workbookIA.git
cd workbookIA
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu   # sur Mac : sans --index-url
pip install -r requirements.txt
pip install -e .
python 00_setup/check_env.py        # tout doit être ✅ (un ⚠️ normal : mylearn pas encore créé)
python tools/start_chapter.py --init  # crée ta librairie mon_travail/mylearn et tes fichiers de suivi
jupyter lab                         # ouvre 00_setup/demo.ipynb
```

Ensuite : `python tools/start_chapter.py 0A` pour commencer le premier chapitre.

## 🔁 Routine pour chaque chapitre

1. `python tools/start_chapter.py 18` : copie le notebook et les squelettes `mylearn` du chapitre dans `mon_travail/` et complète tes fichiers de suivi, sans jamais écraser ton travail (sur Colab : [00_setup/COLAB.md §2](00_setup/COLAB.md#2-démarrer-un-chapitre)).
2. Ouvre la fiche (`01_fiche.md`) et suis son **ordre conseillé** : un bloc du livre, la partie de la fiche qui lui correspond (la fiche entière compte 1 à 2 heures de lecture), puis les quiz 🧠, les rappels 🔁 et les exercices papier ✏️ et ∂ de ce bloc, sans le livre.
3. Écris tes réponses papier dans `mon_travail/<chapitre>/06_mes_reponses.md`, puis passe au bloc suivant.
4. Notebook (ta copie, `mon_travail/<chapitre>/03_notebook.ipynb`), dans l'ordre conseillé par la fiche : à chaque 🔮, écris ta prédiction **avant** d'exécuter. `wb.check` te dit si ta réponse est juste sans la révéler.
5. `python -m pytest tests/test_ch18_<module>.py -q` (la commande exacte est dans le notebook) : tes fonctions `mylearn` comparées à des bibliothèques de confiance ; `python -m pytest tests/ -q -m mylearn` teste toute ta librairie.
6. Corrige avec les indices (`04_indices.md`) puis les solutions, réponds aux questions 💼 à voix haute, importe les flashcards du chapitre dans Anki (`python tools/export_flashcards.py --chapter 18`, pas à pas dans [`annexes/anki.md`](annexes/anki.md)).
7. Note ta séance dans `mon_travail/suivi/journal.md`, ton niveau dans `mon_travail/suivi/auto_evaluation.md`, et coche tes exercices dans `mon_travail/suivi/tableau_de_bord.md`.

À la fin d'une partie, `python tools/start_chapter.py CP1` (puis `CP2`…) copie l'examen blanc, ta copie et le kit de départ du mini-projet ; le déroulé est dans le `README.md` du checkpoint. Pour finir, `python tools/start_chapter.py PF` copie le kit du projet final (`projets/projet_final/depart/`) dans `mon_travail/projet_final/`, qui deviendra ton propre dépôt git (le pas à pas est dans [`projets/projet_final/README.md`](projets/projet_final/README.md), section « Démarrer »). La méthode complète (comment étudier, comment demander de l'aide à Claude en mode tuteur) est dans [`docs/METHODE.md`](docs/METHODE.md).

## 📚 Plan et statuts

<!-- wb:readme plan -->
| Partie | Chapitres | Checkpoint : examen blanc, synthèse, mini-projet |
|---|---|---|
| **0 · Prérequis** | ✅ [0A · Python, notebooks et outils](chapitres/ch00a_python/01_fiche.md)<br>✅ [0B · Maths du lycée au ML](chapitres/ch00b_maths/01_fiche.md) | — |
| **I · Fondations** | ✅ [1 · Introduction au machine learning et au deep learning](chapitres/ch01_introduction/01_fiche.md)<br>✅ [2 · Hasard et statistiques de base](chapitres/ch02_stats/01_fiche.md)<br>✅ [3 · Probabilités et mesure de la qualité](chapitres/ch03_probabilites/01_fiche.md)<br>✅ [4 · Règle de Bayes](chapitres/ch04_bayes/01_fiche.md)<br>✅ [5 · Courbes et surfaces](chapitres/ch05_courbes/01_fiche.md)<br>✅ [6 · Théorie de l'information](chapitres/ch06_information/01_fiche.md) | ✅ [CP1](checkpoints/partie_1/README.md) · mini-projet [MP1 Détecteur de langue anglais / français from scratch](projets/partie_1_detecteur_langue/) |
| **II · Concepts** | ✅ [7 · Classification](chapitres/ch07_classification/01_fiche.md)<br>✅ [8 · Entraînement et test](chapitres/ch08_train_test/01_fiche.md)<br>✅ [9 · Overfitting et underfitting](chapitres/ch09_overfitting/01_fiche.md)<br>✅ [10 · Neurones](chapitres/ch10_neurones/01_fiche.md)<br>✅ [11 · Apprentissage et raisonnement](chapitres/ch11_raisonnement/01_fiche.md) | ✅ [CP2](checkpoints/partie_2/README.md) · mini-projet [MP2 Prix des logements californiens : un protocole d'évaluation honnête](projets/partie_2_california_validation/) |
| **III · ML classique** | ✅ [12 · Préparation des données](chapitres/ch12_preparation/01_fiche.md)<br>✅ [13 · Classifieurs](chapitres/ch13_classifieurs/01_fiche.md)<br>✅ [14 · Ensembles](chapitres/ch14_ensembles/01_fiche.md)<br>✅ [15 · scikit-learn](chapitres/ch15_sklearn/01_fiche.md) | ✅ [CP3](checkpoints/partie_3/README.md) · mini-projet [MP3 Prédire un revenu à partir du recensement : pipeline tabulaire complet et audit d'équité](projets/partie_3_revenus/) |
| **IV · Réseaux** | ✅ [16 · Réseaux feed-forward](chapitres/ch16_feedforward/01_fiche.md)<br>✅ [17 · Fonctions d'activation](chapitres/ch17_activations/01_fiche.md)<br>✅ [18 · Rétropropagation](chapitres/ch18_backprop/01_fiche.md)<br>✅ [19 · Optimiseurs](chapitres/ch19_optimiseurs/01_fiche.md)<br>✅ [20 · Deep learning et premiers pas en PyTorch](chapitres/ch20_deep_learning/01_fiche.md) | ✅ [CP4](checkpoints/partie_4/README.md) · mini-projet [MP4 Un mini-framework de deep learning en NumPy, validé contre PyTorch](projets/partie_4_mini_framework/) |
| **V · Architectures** | ✅ [21 · Réseaux convolutifs (CNN)](chapitres/ch21_cnn/01_fiche.md)<br>✅ [22 · Réseaux récurrents (RNN, LSTM, GRU)](chapitres/ch22_rnn/01_fiche.md)<br>✅ [23 · PyTorch en pratique 1 : du dataset au modèle sauvegardé](chapitres/ch23_pytorch_1/01_fiche.md)<br>✅ [24 · PyTorch en pratique 2 : améliorer, chercher, CNN et RNN](chapitres/ch24_pytorch_2/01_fiche.md) | ✅ [CP5](checkpoints/partie_5/README.md) · mini-projet [MP5 Classifieur d'images CIFAR-10 de bout en bout](projets/partie_5_classifieur_cifar10/) |
| **VI · Génératif et RL** | ✅ [25 · Autoencodeurs et VAE](chapitres/ch25_autoencodeurs/01_fiche.md)<br>✅ [26 · Apprentissage par renforcement](chapitres/ch26_rl/01_fiche.md)<br>✅ [27 · Réseaux antagonistes génératifs (GAN)](chapitres/ch27_gan/01_fiche.md)<br>✅ [28 · Applications créatives](chapitres/ch28_creatif/01_fiche.md)<br>✅ [29 · Datasets et préparation du projet final](chapitres/ch29_datasets/01_fiche.md) | ✅ [CP6](checkpoints/partie_6/README.md) · mini-projet [MP6 Banc d'essai génératif : VAE contre GAN sur Fashion-MNIST](projets/partie_6_generatif/) |
| **VII · Bonus** | ✅ [B1 · Transfer learning et modèles pré-entraînés](chapitres/b1_transfer/01_fiche.md)<br>✅ [B2 · Tokenisation et embeddings](chapitres/b2_tokenisation/01_fiche.md)<br>✅ [B3 · Attention et Transformers : un mini-GPT from scratch](chapitres/b3_transformers/01_fiche.md)<br>✅ [B4 · LLM en pratique : Hugging Face, prompting, RAG et LoRA](chapitres/b4_llm/01_fiche.md)<br>✅ [B5 · Modèles de diffusion : un DDPM minimal](chapitres/b5_diffusion/01_fiche.md)<br>✅ [B6 · Explicabilité, équité et éthique](chapitres/b6_explicabilite/01_fiche.md)<br>✅ [B7 · Du notebook à la production](chapitres/b7_production/01_fiche.md)<br>✅ [B8 · RL moderne : DQN, gradient de politique, PPO et RLHF](chapitres/b8_rl_moderne/01_fiche.md) | — |
| **Fin** | ✅ [Projet final : un projet de bout en bout sur ton propre dataset](projets/projet_final/) | — |

✅ publié · 🛠️ en cours de génération · 📅 planifié. Les liens des chapitres mènent à leur fiche (`01_fiche.md`) ; le dossier contient aussi les exercices, le notebook, les indices, les solutions, ta feuille de réponses et les flashcards.
<!-- wb:end plan -->

Le plan détaillé (exercices, signatures `mylearn`, sections du livre couvertes, calendrier indicatif) est dans [`docs/SYLLABUS.md`](docs/SYLLABUS.md) ; l'historique de la génération dans [`suivi/PROGRESS.md`](suivi/PROGRESS.md).

## 🧭 Quatre parcours

<!-- wb:readme parcours -->
| Parcours | Pour qui | Exercices | Temps total | À 10 h par semaine |
|---|---|---|---|---|
| [Parcours complet](docs/PARCOURS.md#complet) | tout le workbook, dans l'ordre | 2028 | ≈ 923 h | ≈ 92 semaines |
| [Parcours rapide](docs/PARCOURS.md#rapide) | l'essentiel pour être employable | 1312 | ≈ 553 h | ≈ 55 semaines |
| [Parcours orienté maths](docs/PARCOURS.md#maths) | comprendre en profondeur (calculs, démonstrations) | 814 | ≈ 571 h | ≈ 57 semaines |
| [Parcours orienté code](docs/PARCOURS.md#code) | devenir solide en implémentation (from scratch, bibliothèques, débogage) | 890 | ≈ 740 h | ≈ 74 semaines |

Les temps comptent la lecture (livre et fiche), les exercices, les corrigés des prérequis hors parcours, les flashcards, les synthèses, les mini-projets et le projet final, communs aux quatre parcours (détail : [`docs/PARCOURS.md`](docs/PARCOURS.md), généré par `python tools/syllabus.py build`).
<!-- wb:end parcours -->

Chaque exercice indique ses parcours (R, M, C) ; les checkpoints, les mini-projets et le projet final font partie des quatre.

## Portfolio : mini-projets et projet final

Chaque partie se termine par un mini-projet prêt pour un portfolio : cahier des charges, kit de départ (`depart/`), solution de référence (`solution/`), grille d'évaluation, README type, et des tests qui vérifient ton code.

| Mini-projet | Ce qu'il montre |
|---|---|
| [MP1 · Détecteur de langue anglais / français](projets/partie_1_detecteur_langue/) | un classifieur Naïve Bayes *from scratch*, des probabilités jusqu'à l'évaluation |
| [MP2 · Prix des logements californiens](projets/partie_2_california_validation/) | un protocole d'évaluation honnête : validation croisée, test gelé, choix justifié |
| [MP3 · Prédire un revenu à partir du recensement](projets/partie_3_revenus/) | un pipeline scikit-learn sans fuite, puis un audit d'équité, avec model card et data card |
| [MP4 · Un mini-framework de deep learning en NumPy](projets/partie_4_mini_framework/) | couches, rétropropagation et optimiseurs, prouvés justes contre PyTorch |
| [MP5 · Classifieur d'images CIFAR-10 de bout en bout](projets/partie_5_classifieur_cifar10/) | des baselines au modèle livré avec sa ligne de commande et sa fiche modèle |
| [MP6 · Banc d'essai génératif : VAE contre GAN](projets/partie_6_generatif/) | deux modèles génératifs comparés sous un même protocole, sur Fashion-MNIST |

**Projet final** ([`projets/projet_final/`](projets/projet_final/)) : un projet de bout en bout sur un dataset que tu choisis, en 13 étapes (4 à 6 semaines). Tu cadres le problème avec une métrique et un critère de succès fixés à l'avance, tu documentes ton dataset (data card du ch. 29), tu compares une baseline classique et un modèle de deep learning sous un protocole figé, tu évalues une seule fois sur un test scellé, puis tu livres un dépôt professionnel (paquet testé, intégration continue, model card) et tu le présentes comme en entretien technique. Le dossier contient :

| Fichier | Contenu |
|---|---|
| [`README.md`](projets/projet_final/README.md) | le cahier des charges : choix du dataset, règle du test scellé, budget, jalons, les 13 étapes et leurs livrables |
| [`GRILLE.md`](projets/projet_final/GRILLE.md) | la grille sur 20 : critères observables, comment les vérifier, pénalités |
| [`04_indices.md`](projets/projet_final/04_indices.md) | trois indices par étape, sur d'autres datasets que le tien |
| [`SOUTENANCE.md`](projets/projet_final/SOUTENANCE.md) | le plan des dix minutes de présentation, la démonstration et une banque de questions d'entretien |
| [`exemple/`](projets/projet_final/exemple/) | un exemple de cadrage et de protocole sur Bike Sharing (UCI), sans aucun résultat de modèle |
| [`depart/`](projets/projet_final/depart/) | le kit de départ, un dépôt modèle : paquet `projet` (configuration et ligne de commande fournies, ouverture unique du test scellé), tests, notebooks, intégration continue, gabarits des documents |

Il n'y a pas de solution de référence : le projet est le tien, et la grille sert à l'auto-évaluation puis à la correction par Claude (P9).

Le lien entre ces compétences et les métiers visés (data analyst, data scientist, ML engineer, AI engineer) est dans [`annexes/metiers.md`](annexes/metiers.md).

## Ta librairie mylearn

Chapitre après chapitre, tu écris ta propre librairie de machine learning dans `mon_travail/mylearn/` : statistiques, métriques, k-means, régressions, arbres et forêts, couches, activations, rétropropagation, optimiseurs, convolution, RNN, VAE, Q-learning, tokenizer BPE, attention, diffusion, explicabilité… `start_chapter.py` y copie les squelettes (signatures, docstrings, `raise NotImplementedError`) et `pytest` compare ton code à des bibliothèques de confiance (NumPy, SciPy, scikit-learn, PyTorch). Les signatures sont figées dans [`docs/SYLLABUS.md`](docs/SYLLABUS.md), qui montre aussi le graphe des modules (un module réutilise ceux des chapitres précédents).

Préfère la commande du chapitre (`python -m pytest tests/test_ch18_backward.py -q`, donnée par le notebook et par `start_chapter.py`) ou `python -m pytest tests/ -q -m mylearn` pour toute ta librairie ; `python -m pytest tests/ -q` lance aussi les tests de l'outillage (`tests/infra/`).

**Tu as sauté un chapitre ?** Pas de problème : quand un module d'un chapitre **précédent** te manque (par exemple `tree.py` du ch. 13, que `ensemble.py` du ch. 14 réutilise), les notebooks et les tests prennent la version de référence (`solutions/mylearn_ref/`) à sa place, avec un message. Les modules du chapitre que tu étudies ne sont jamais remplacés : une vérification ✅ vient toujours de ton code. Pour écrire toi-même un module sauté : `python tools/start_chapter.py 13`.

## 🗂️ Structure du dépôt

```
workbookIA/
├── 00_setup/            installation (Colab, local), check_env.py, notebook de démonstration
├── chapitres/           un dossier par chapitre (ch00a_python … ch29_datasets, b1_transfer … b8_rl_moderne) :
│                        01_fiche, 02_exercices, 03_notebook, 04_indices, 05_solutions (.md et .ipynb exécuté),
│                        06_mes_reponses, flashcards.csv, figures/
├── checkpoints/         partie_1 … partie_6 : examen blanc (sujet, notebook, corrigé, copie) et synthèse
├── projets/             les six mini-projets (depart/, solution/) et le projet final (cahier des charges, grille, indices, soutenance, exemple/, depart/)
├── annexes/             glossaire, index des notions, formulaire, cheatsheets, erreurs fréquentes, Anki, ressources, métiers
├── src/wb/              utilitaires du workbook : setup, wb.check, datasets, données synthétiques, graphiques…
├── templates/mylearn_stubs/   squelettes de mylearn (figés, MANIFEST.json)
├── solutions/mylearn_ref/     implémentation de référence de mylearn
├── tests/               tests de mylearn (test_chXX_<module>.py) et de l'outillage (tests/infra/)
├── data/                datasets légers versionnés et leurs fiches (data/cards/)
├── docs/                BIBLE.md (la spécification), METHODE.md, SYLLABUS.md et PARCOURS.md (générés), syllabus/data/
├── suivi/               PROGRESS.md (génération), modèles de ton tableau de bord, journal, auto-évaluation, remédiation
├── tools/               scripts (voir Outils) ; tools/chapters/ : un script de construction par chapitre
├── exports/             decks Anki générés (non versionné)
└── mon_travail/         TON espace : copies des notebooks, ta librairie mylearn, ton suivi (Claude n'y écrit jamais)
```

## 🧰 Outils

| Commande | Rôle |
|---|---|
| `python 00_setup/check_env.py` | vérifie ton installation |
| `python tools/start_chapter.py 18` | démarre un chapitre, un checkpoint (`CP1`) ou le projet final (`PF`) ; `--init` crée `mon_travail/mylearn` et tes fichiers de suivi |
| `python -m pytest tests/test_ch18_<module>.py -q` | teste ton module du chapitre (`tests/ -m mylearn` : toute ta librairie ; `--impl=ref` : la référence) |
| `python tools/export_flashcards.py` | fusionne les flashcards en un deck Anki à sous-paquets (`--chapter 3` : un chapitre ; `--apkg` : un paquet `.apkg`, avec genanki) |
| `python tools/syllabus.py check` | vérifie le contrat (exercices, prérequis, parcours, stubs) ; `build` régénère le SYLLABUS, les parcours, le tableau de bord, l'index des notions et les tableaux de ce README |
| `python tools/repo_stats.py --write` | recompte les chiffres de ce README |
| `python tools/run_all_notebooks.py` | outil de génération : exécute les notebooks du dépôt (jamais tes copies de `mon_travail/`) et mesure leur durée |
| `python tools/build_answers.py` | outil de génération : construit `src/wb/answers.json` à partir des notebooks de solutions exécutés |

## 📎 Annexes

| Annexe | Contenu |
|---|---|
| [Index des notions](annexes/index_notions.md) | chaque notion, de A à Z : le chapitre et la section de la fiche qui l'enseignent, les exercices qui la pratiquent, les chapitres qui la réutilisent ; renvois depuis l'anglais (généré) |
| [Glossaire](annexes/glossaire.md) | les termes français ↔ anglais, avec leur définition courte et leur chapitre (BIBLE §5) |
| [Formulaire](annexes/formulaire.md) | toutes les formules, par chapitre, et les conventions des bibliothèques |
| [Cheatsheets](annexes/cheatsheets/) | NumPy, pandas, scikit-learn, PyTorch, git |
| [Erreurs fréquentes](annexes/erreurs_frequentes.md) | les messages d'erreur et les bugs classiques, avec leur cause et leur remède |
| [Anki](annexes/anki.md) | importer le deck global ou un chapitre, réviser au fil du parcours, suspendre les chapitres pas encore étudiés |
| [Ressources](annexes/ressources.md) | documentation officielle, cours et livres vérifiés |
| [Métiers](annexes/metiers.md) | les compétences du workbook, métier par métier |
| [Fiches des datasets](data/cards/) | provenance, licence, biais et limites de chaque dataset |

## ⭐ Règles d'or

1. **Essaie vraiment avant de regarder une solution.** L'effort de récupération est ce qui fait apprendre.
2. **Règle des 15 minutes** : bloqué 15 min → indice 1 ; encore 5 min → indice 2, puis l'indice 3 (le mode d'emploi de `04_indices.md`).
3. **Prédis avant d'exécuter** (🔮) : écris ton hypothèse, puis compare.
4. **Travaille uniquement dans `mon_travail/`** : les autres dossiers sont mis à jour par `git pull`.
5. **FAST_MODE d'abord** : tout tourne sur CPU en quelques minutes ; le mode complet est optionnel (🚀 : GPU Colab conseillé).
6. **Révise tes flashcards chaque jour**, même 10 minutes.
7. **Commite ton travail** régulièrement (`mon_travail/`) : ton dépôt GitHub est ton portfolio.

## Versions

Python 3.13 et les versions préinstallées sur Colab (PyTorch 2.11, scikit-learn 1.6, NumPy 2.1, pandas 2.2…) : détail et justification dans [`docs/BIBLE.md` §21](docs/BIBLE.md#21-versions-figées). Les extras de `pyproject.toml` (`torch`, `hf`, `rl`…) installent les dépendances d'un groupe de chapitres ; `anki`, celle de l'export `.apkg` des flashcards (genanki, facultatif).

## Crédits et licences

- **Le livre** : Andrew Glassner, *Deep Learning: From Basics to Practice*, 2 volumes, 2018. Le workbook n'en recopie aucun passage : les fiches sont des résumés originaux, les exercices sont originaux, et les idées d'exemples reprises (la pièce truquée, Flippers, le réseau minuscule du ch. 18…) sont réécrites. Les PDF du livre ne sont jamais versionnés. Le code et les figures du livre sont publiés par l'auteur sous licence MIT ; ses dépôts de notebooks ne sont plus en ligne, mais des copies existent : [volume 1](https://github.com/bssrdf/DeepLearningBookCode-Volume1) et [volume 2](https://github.com/bssrdf/DeepLearningBookCode-Volume2) (voir aussi les [ressources de l'auteur](https://github.com/blueberrymusic/DeepLearningBook-Resources)).
- **Les datasets** : chacun a sa fiche dans [`data/cards/`](data/cards/), avec sa provenance et sa licence. Palmer Penguins (CC0), MNIST (CC BY-SA 3.0), Fashion-MNIST (MIT), CIFAR-10 (sans licence formelle, citer le rapport technique), California Housing (recensement américain de 1990, via StatLib), *Adult* et 20 Newsgroups (CC BY 4.0, UCI), les taches solaires (SILSO, **CC BY-NC 4.0** : usage non commercial), *The Adventures of Sherlock Holmes* et *Le Tour du monde en quatre-vingts jours* (domaine public, Project Gutenberg), des photos sous CC BY 2.0 et des tableaux du domaine public ([`data/cards/images.md`](data/cards/images.md)). L'exemple de calibrage du projet final décrit Bike Sharing (UCI, CC BY 4.0), qui n'est pas versionné.
- **CleanRL** : le chapitre B8 fait relire `ppo.py` de [CleanRL](https://github.com/vwxyzjn/cleanrl) (licence MIT), téléchargé à un commit figé et jamais installé.
- **nanoGPT** : le mini-GPT du chapitre B3 suit la structure de [nanoGPT](https://github.com/karpathy/nanoGPT) d'A. Karpathy (licence MIT), crédité dans la fiche.
- **Les modèles pré-entraînés** (ResNet-18, MobileNetV3, SqueezeNet et VGG16 de torchvision ; DINOv2, CLIP, MiniLM, SmolLM2, Qwen2.5… du Hugging Face Hub) sont téléchargés à des révisions figées, jamais versionnés ; leurs licences (BSD 3-Clause pour torchvision, Apache 2.0 ou MIT pour la plupart des modèles du Hub, les conditions d'ImageNet, les restrictions de la fiche de CLIP) sont détaillées dans [`data/cards/pretrained.md`](data/cards/pretrained.md).
- **Les bibliothèques** : NumPy, pandas, Matplotlib, scikit-learn, PyTorch, Hugging Face (transformers, datasets, tokenizers, peft), Gymnasium, XGBoost, LightGBM, SHAP, FastAPI, pytest et genanki (facultatif), chacune sous sa propre licence.
