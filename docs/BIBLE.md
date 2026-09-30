# BIBLE DU WORKBOOK — « Deep Learning: From Basics to Practice » (A. Glassner)

> **Statut : document de référence.** Ce fichier est la spécification unique du workbook. Toute session de génération doit le lire en entier avant de produire quoi que ce soit.
> En cas de conflit entre un prompt et cette bible, **la bible l'emporte**, sauf si le prompt écrit explicitement « DÉROGATION ». Toute dérogation acceptée est consignée en §22.
> Après la session 1, la version qui fait foi est `docs/BIBLE.md` dans le dépôt.

---

## 1. Profil de l'apprenant

| Élément | Valeur | Conséquence pour la génération |
|---|---|---|
| Objectif | **Reconversion professionnelle** vers les métiers de l'IA et de la data | Compétences employables : portfolio GitHub, questions d'entretien, bonnes pratiques pro |
| Niveau maths | **Lycée** | Toute notion au-delà du lycée est d'abord introduite (chap. 0B ou encadré 🧮 *Rappel maths*) avant d'être utilisée. Aucun saut. |
| Niveau Python | Débutant (à confirmer ; ajuster selon `suivi/PROGRESS.md` §Calibrage) | Chap. 0A complet ; échafaudage de code fort au début, puis dégressif |
| Temps disponible | **Illimité** ; l'objectif est l'exhaustivité | Pas d'économie de contenu ; on vise la couverture totale du livre et au-delà |
| Matériel | **Google Colab** (GPU gratuit parfois disponible) + **ordinateur personnel sans GPU** | Chaque notebook doit tourner sur CPU en mode rapide ; les usages GPU sont signalés 🚀 |
| Langue | **Explications en français, code en anglais** | Voir §5 |

## 2. Objectifs du workbook

1. **Comprendre** en profondeur chaque chapitre du livre (intuition, maths, code).
2. **Savoir implémenter** : from scratch (numpy), puis avec les outils professionnels (scikit-learn, PyTorch, Hugging Face).
3. **Être employable** : un dépôt GitHub présentable, des mini-projets de portfolio, des réponses solides aux questions d'entretien et des réflexes de pro (tests, git, reproductibilité, lecture de documentation).
4. **Être à jour** : signaler tout ce que le livre (2018) présente et qui est dépassé ou fait autrement aujourd'hui, et couvrir les sujets modernes absents du livre (chapitres bonus).

## 3. Livre source et droits

- **Volume 1** : chapitres 1 à 19. **Volume 2** : chapitres 20 à 29, plus le glossaire (ch. 30).
- **Interdit** : recopier le texte du livre, même paraphrasé de près. Les fiches sont des résumés originaux ; on renvoie aux sections sous la forme « (livre §18.5) ».
- **Autorisé** : reprendre les *idées* d'exemples du livre (pièce truquée, Flippers, réseau minuscule du ch. 18…) sous forme d'exercices originaux. Le code et les figures du livre sont sous licence MIT sur le GitHub de l'auteur : on peut y renvoyer et s'en inspirer, en le créditant.
- Les PDF du livre ne sont **jamais** versionnés dans le dépôt (`.gitignore`).

## 4. Stack technique

- **Python** : la version par défaut de Colab (figée en §21).
- **Bibliothèques principales** : numpy, pandas, matplotlib (seaborn en option), scikit-learn, **PyTorch + torchvision** (framework deep learning principal), pytest.
- **Selon les chapitres** : xgboost et/ou lightgbm (ch. 14-15), gymnasium (ch. 26, B8), transformers, datasets et tokenizers de Hugging Face (bonus), umap-learn (ch. 12), shap (B6).
- **Keras** : n'est pas utilisé pour les exercices. Le livre s'en sert (API Keras de 2018, qui ne tourne plus telle quelle). Chaque fois que c'est pertinent, un encadré 🕰️ explique l'équivalent PyTorch et le statut actuel de Keras 3.
- **Versions** : figées dans `requirements.txt` lors de la session 1, après vérification par recherche web, et recopiées en §21. Elles doivent s'installer **à la fois sur Colab et en local sur CPU**.
- **Budget d'exécution** :
  - chaque notebook comporte `FAST_MODE = True` par défaut (sous-échantillons, moins d'epochs) ;
  - le notebook de solutions en FAST_MODE doit s'exécuter **en entier en moins de 10 minutes sur CPU**, et aucune cellule ne doit dépasser 3 minutes sur CPU ;
  - le mode complet (`FAST_MODE = False`) est documenté avec les résultats attendus et un temps estimé sur CPU et sur GPU Colab ; au-delà de 15 minutes sur CPU, la cellule est marquée 🚀 *GPU Colab conseillé* ;
  - device : `cuda` si disponible, sinon `cpu`, géré par `wb.setup()` ;
  - graines aléatoires fixées par `wb.setup(seed=42)` (numpy, random, torch).

## 5. Langue et terminologie

- **Markdown** (énoncés, fiches, solutions) : **en français**, tutoiement, ton clair et encourageant, phrases courtes.
- **Code** : identifiants, docstrings et commentaires **en anglais**, comme en entreprise. Toute explication pédagogique va dans les cellules Markdown, en français.
- **Termes techniques** : on garde l'anglais quand c'est l'usage professionnel. À la première occurrence dans un chapitre : « terme retenu (*autre langue*) », puis on s'en tient au terme retenu. Tout terme ajouté va dans `annexes/glossaire.md`.

| On garde l'anglais | On dit en français | Remarque |
|---|---|---|
| dataset, feature, label, batch, mini-batch, epoch, learning rate, loss, pipeline, framework, fine-tuning, embedding, dropout, pooling, padding, stride, kernel, token, prompt, overfitting, underfitting, benchmark, notebook, accuracy, precision, recall, F1-score | réseau de neurones, couche, poids, biais, neurone, fonction d'activation, descente de gradient, rétropropagation (*backpropagation*), entraînement, validation croisée (*cross-validation*), apprentissage supervisé ou non supervisé, apprentissage par renforcement, matrice de confusion | **accuracy / precision / recall restent en anglais** : « précision » est ambigu en français |

## 6. Notations mathématiques

- LaTeX dans le Markdown (`$…$`, `$$…$$`), compatible avec le rendu GitHub, Colab et Jupyter.
- Scalaires en italique ($x$), vecteurs en gras minuscule ($\mathbf{x}$), matrices en gras majuscule ($\mathbf{W}$).
- Prédiction $\hat{y}$, cible $y$, loss $L$, learning rate $\eta$, activation $\sigma$ ou $f$, somme pondérée $z$, sortie activée $a$ ; indices : $i$ pour les exemples, $j$ et $k$ pour les neurones, $\ell$ pour la couche.
- On reprend les notations du livre quand elles existent ; un écart éventuel est signalé une fois dans la fiche.
- `annexes/formulaire.md` recense toutes les formules du workbook, par chapitre.

## 7. Architecture du dépôt

```
dl-workbook/
├── README.md                     ← présentation, démarrage rapide Colab et local, parcours, sommaire
├── requirements.txt / pyproject.toml
├── .gitignore                    ← *.pdf, données lourdes, checkpoints, caches
├── docs/
│   ├── BIBLE.md                  ← ce document (source de vérité)
│   ├── METHODE.md
│   ├── SYLLABUS.md               ← plan détaillé de tous les chapitres (session 2)
│   ├── PARCOURS.md               ← parcours complet, rapide, orienté maths, orienté code
│   └── AUDIT_*.md                ← rapports d'audit
├── 00_setup/                     ← guide d'installation locale (CPU), guide Colab + Google Drive, check_env.py, demo.ipynb
├── src/wb/                       ← utilitaires du workbook (setup, check, datasets, synth, plot)
├── templates/mylearn_stubs/      ← squelettes de la librairie à compléter, un fichier par chapitre
├── solutions/mylearn_ref/        ← implémentation de référence (package `mylearn_ref`)
├── tests/                        ← tests pytest de mylearn : --impl=learner (défaut) ou --impl=ref
├── data/                         ← datasets légers versionnés + cards/ (fiche descriptive de chaque dataset)
├── chapitres/
│   └── chXX_nom-court/           ← ex. ch00a_python, ch18_backprop, b3_transformers
│       ├── 01_fiche.md
│       ├── 02_exercices.md
│       ├── 03_notebook.ipynb
│       ├── 04_indices.md
│       ├── 05_solutions.md
│       ├── 05_solutions.ipynb    ← exécuté, sorties conservées
│       ├── 06_mes_reponses.md    ← modèle vide où l'apprenant écrit ses réponses papier
│       └── flashcards.csv
├── checkpoints/partie_X/         ← examen blanc (sujet, corrigé, barème) + synthèse de la partie
├── projets/                      ← mini-projets de portfolio (un par partie) + projet final
├── annexes/                      ← glossaire FR↔EN, formulaire, cheatsheets (numpy, pandas, sklearn, pytorch, git), erreurs fréquentes, ressources
├── suivi/
│   ├── PROGRESS.md               ← état de la génération (tenu par Claude)
│   ├── tableau_de_bord.md        ← cases à cocher pour chaque exercice (tenu par l'apprenant)
│   ├── journal.md                ← modèle de journal d'apprentissage
│   ├── auto_evaluation.md        ← grille de compétences
│   └── remediation.md            ← exercices de remédiation ajoutés après correction
├── tools/                        ← build_answers.py, start_chapter.py, run_all_notebooks.py, export_flashcards.py
└── mon_travail/                  ← ESPACE DE L'APPRENANT : Claude n'y écrit jamais
    ├── mylearn/                  ← la librairie de l'apprenant (copiée depuis les stubs)
    └── chXX/                     ← copies de travail des notebooks
```

**Règles anti-conflit** (Claude et l'apprenant travaillent dans le même dépôt) :
- Claude **n'écrit jamais** dans `mon_travail/`. Il ne fait qu'y lire, pour corriger le travail de l'apprenant (prompt P9).
- `python tools/start_chapter.py 18` copie le notebook d'exercices et les nouveaux stubs mylearn vers `mon_travail/`, **sans jamais écraser un fichier existant**.
- Un stub mylearn publié n'est plus jamais modifié. Une fonctionnalité nouvelle va dans un nouveau fichier. En cas d'erreur dans un stub, on publie une version corrigée sous un autre nom et on documente le correctif dans le chapitre.
- Le notebook d'exercices importe `mylearn` (le code de l'apprenant) ; le notebook de solutions importe `mylearn_ref as mylearn`.

## 8. Plan : parties, chapitres, fils rouges

Les chapitres 23 et 24 du livre (Keras) deviennent « PyTorch en pratique ». Les bases de PyTorch sont introduites dès le ch. 20, pour que les ch. 21 et 22 se pratiquent directement en PyTorch.

| Partie | ID | Titre workbook | Livre | Fils rouges | Module mylearn | Points de modernisation 🕰️ (liste indicative, à vérifier) |
|---|---|---|---|---|---|---|
| **0 Prérequis** | 0A | Python, notebooks & outils (Colab, numpy, pandas, matplotlib, bases de git) | — | Penguins | structure du package, 1er test | — |
| | 0B | Maths du lycée au ML (vecteurs, matrices, produit matriciel, fonctions, dérivées, règle de la chaîne, dérivées partielles, Σ, exp/log, bases de probas) | — | synthétiques | `linalg_basics` | — |
| **I Fondations** | 1 | Introduction | V1 ch1 | les 4 fils rouges (découverte) | — | panorama 2026 (LLM, diffusion, foundation models) |
| | 2 | Hasard et statistiques | V1 ch2 | Penguins, synth | `stats` (mean, var, cov, corr, bootstrap) | |
| | 3 | Probabilités et mesure de la qualité | V1 ch3 | synth, Penguins | `metrics` (confusion_matrix, accuracy, precision, recall, f1) | ROC-AUC, PR-AUC, calibration |
| | 4 | Règle de Bayes | V1 ch4 | synth (pièces), Penguins | `bayes` | |
| | 5 | Courbes et surfaces | V1 ch5 | Rosenbrock, synth | `calculus` (dérivée et gradient numériques) | autodiff |
| | 6 | Théorie de l'information | V1 ch6 | Holmes (EN) + Verne (FR) | `info` (entropy, cross_entropy, kl, huffman) | lien avec la loss des LLM, perplexité |
| **II Concepts** | 7 | Classification | V1 ch7 | moons/blobs, Penguins | `cluster` (k-means) | DBSCAN/HDBSCAN |
| | 8 | Entraînement et test | V1 ch8 | Penguins, California | `model_selection` (split, k-fold, stratification) | |
| | 9 | Overfitting et underfitting | V1 ch9 | polynômes synth, California | `linear` (régression linéaire, ridge, lasso) | double descente (mention) |
| | 10 | Neurones | V1 ch10 | portes logiques | `perceptron` | |
| | 11 | Apprentissage et raisonnement | V1 ch11 | simulation de bandit | `bandit` | |
| **III ML classique** | 12 | Préparation des données | V1 ch12 | Penguins (valeurs manquantes), California, MNIST (PCA) | `preprocessing` (scalers, one-hot, PCA) | UMAP, t-SNE, `set_output`, fuite de données |
| | 13 | Classifieurs | V1 ch13 | Penguins, sous-ensemble MNIST, Holmes/Verne (Naive Bayes) | `neighbors`, `tree`, `naive_bayes` | |
| | 14 | Ensembles | V1 ch14 | California, Penguins | `ensemble` (bagging, random forest simple) | **gradient boosting (XGBoost, LightGBM, CatBoost, HistGradientBoosting) = état de l'art sur le tabulaire** |
| | 15 | scikit-learn | V1 ch15 | tous les datasets tabulaires | comparaison mylearn / sklearn | Pipeline, ColumnTransformer, API récente |
| **IV Réseaux** | 16 | Réseaux feed-forward | V1 ch16 | synth, MNIST | `nn/layers` | initialisation He/Xavier par défaut |
| | 17 | Fonctions d'activation | V1 ch17 | synth | `nn/activations` | GELU, SiLU ; sigmoïde/tanh abandonnées en couches cachées |
| | 18 | Rétropropagation | V1 ch18 | synth, MNIST | `nn/backward`, `autograd_mini` (micro-autograd) | autodiff : plus de dérivation à la main en pratique |
| | 19 | Optimiseurs | V1 ch19 | Rosenbrock, MNIST | `optim` (SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW) | AdamW, warmup + cosine, gradient clipping |
| | 20 | Deep learning + **premiers pas en PyTorch** | V2 ch20 | MNIST, Fashion-MNIST | `nn/regularization` (dropout, batchnorm) | LayerNorm, connexions résiduelles |
| **V Architectures** | 21 | CNN | V2 ch21 | Fashion-MNIST, CIFAR-10 | `conv` (conv2d, maxpool en numpy) | ResNet, modèles pré-entraînés, ViT |
| | 22 | RNN | V2 ch22 | Holmes/Verne (génération caractère par caractère), série temporelle | `rnn` (cellule RNN en numpy) | Transformers pour le texte ; RNN encore utiles sur séries légères ; SSM (mention) |
| | 23 | PyTorch en pratique 1 | V2 ch23 (Keras 1) | Fashion-MNIST | — | Keras 3 multi-backend (encadré) |
| | 24 | PyTorch en pratique 2 | V2 ch24 (Keras 2) | CIFAR-10, Holmes | — | data augmentation, mixed precision, `torch.compile` (si stable) |
| **VI Génératif & RL** | 25 | Autoencodeurs et VAE | V2 ch25 | MNIST, Fashion-MNIST | — | espaces latents des modèles de diffusion |
| | 26 | Apprentissage par renforcement | V2 ch26 | Flippers, morpion, FrozenLake, CartPole | `rl` (Q-learning, SARSA) | DQN, PPO, RLHF |
| | 27 | GAN | V2 ch27 | gaussienne 1D, MNIST/Fashion-MNIST | — | la diffusion a supplanté les GAN pour la plupart des usages |
| | 28 | Applications créatives | V2 ch28 | CIFAR-10 / images libres, Holmes | — | deep dream et style transfer sont historiques ; génération moderne |
| | 29 | Datasets → préparation du projet final | V2 ch29 | au choix de l'apprenant | — | Hugging Face Hub, Kaggle, OpenML, UCI ; data cards |
| **VII Bonus** | B1 | Transfer learning et modèles pré-entraînés | — | CIFAR-10 / petit dataset d'images | — | |
| | B2 | Tokenisation et embeddings | — | Holmes/Verne | — | |
| | B3 | Attention et Transformers (mini-GPT from scratch) | — | Holmes/Verne | `attention` | |
| | B4 | LLM en pratique (Hugging Face, prompting, RAG, fine-tuning léger/LoRA) | — | Holmes/Verne + corpus au choix | — | |
| | B5 | Modèles de diffusion (DDPM minimal) | — | MNIST/Fashion-MNIST | — | |
| | B6 | Explicabilité, équité, éthique (SHAP, Grad-CAM, biais) | — | Penguins, California, CIFAR | — | |
| | B7 | Du notebook à la production (scripts, config, suivi d'expériences, API, reproductibilité) | — | projet de la partie V | — | |
| | B8 | RL moderne (DQN, policy gradient, PPO, idée du RLHF) | — | CartPole | — | |
| **Fin** | PF | Projet final | — | dataset choisi par l'apprenant | — | |

Un checkpoint (examen blanc, synthèse et mini-projet) clôt chacune des parties I à VI.

*Le détail qui fait foi (exercices, modules `mylearn` effectifs et leurs signatures) est dans `docs/SYLLABUS.md`, généré en session 2 ; les modules ajoutés à ce tableau sont consignés au §22.*

## 9. Datasets (fils rouges)

Les loaders sont centralisés dans `src/wb/datasets.py`, chacun avec un fallback. Chaque dataset a sa fiche `data/cards/<nom>.md` : provenance, licence, taille, variables, biais et limites connus, chapitres qui l'utilisent.

| Nom | Contenu | Stockage | Rôle |
|---|---|---|---|
| **Palmer Penguins** | 344 manchots, 3 espèces, mesures, valeurs manquantes | CSV versionné dans `data/` | classification tabulaire, stats, nettoyage |
| **California Housing** | prix immobiliers (recensement de 1990) | téléchargé par sklearn, CSV de secours dans `data/` | régression tabulaire. Remplace Boston Housing, **retiré de scikit-learn pour raisons éthiques** : à exploiter dans un exercice ⚖️ |
| **MNIST** | chiffres manuscrits 28×28 | torchvision (téléchargement) ; copie `.npz` compressée dans `data/` si la taille le permet | fil rouge images |
| **Fashion-MNIST** | vêtements 28×28 | torchvision | images, plus difficile que MNIST |
| **CIFAR-10** | 60 000 images couleur 32×32 | torchvision (jamais versionné) | CNN, augmentation, transfer learning |
| **Holmes** | *The Adventures of Sherlock Holmes* (Project Gutenberg, domaine public) | `.txt` versionné | fil rouge texte (EN) |
| **Verne** | *Le Tour du monde en quatre-vingts jours* (Gutenberg, domaine public) | `.txt` versionné | fil rouge texte (FR), comparaisons de langues |
| **Série temporelle** | série réelle légère, choisie en session 1 (ex. taches solaires mensuelles) + sinus bruité synthétique | CSV versionné | RNN, prévision |
| **Environnements RL** | Flippers et morpion (recodés d'après la description du livre), FrozenLake, CartPole (gymnasium) | code | RL |
| **Synthétiques** | `wb.synth` : moons, spirals, blobs, xor, polynomial, noisy_sine, rosenbrock, coin_flips… | générés, avec graine | expériences contrôlées |

Si un téléchargement est bloqué dans l'environnement de génération : utiliser la copie versionnée, sinon un substitut de même forme pour valider le code, **et** marquer la cellule « à valider sur Colab » dans `suivi/PROGRESS.md`.

## 10. mylearn : la librairie que l'apprenant construit

- Un fichier par chapitre dans `templates/mylearn_stubs/` (ex. `metrics.py` au ch. 3, `nn/activations.py` au ch. 17). Signatures, docstrings (en anglais) et `raise NotImplementedError` dans chaque corps de fonction.
- Référence complète dans `solutions/mylearn_ref/`.
- **Des tests fondés sur un oracle, qui ne révèlent pas la solution** : chaque fonction est comparée à numpy, scipy, scikit-learn ou PyTorch (ex. `mylearn.metrics.f1` contre `sklearn.metrics.f1_score`, `mylearn.optim.Adam` contre `torch.optim.Adam` sur un même problème), avec en plus des tests de propriétés (formes, invariants, cas limites).
- `pytest tests/test_chXX_*.py` teste le code de l'apprenant ; `pytest --impl=ref` valide la référence.
- Les signatures sont fixées dans `docs/SYLLABUS.md` et ne changent plus ensuite.

## 11. Types d'exercices

| Icône | Type | Où |
|---|---|---|
| 🧠 | Quiz de lecture (QCM, vrai/faux justifié) | 02_exercices |
| 🔁 | Rappel espacé (question sur un chapitre antérieur) | 02_exercices |
| ✏️ | Calcul à la main | 02_exercices |
| ∂ | Dérivation ou démonstration mathématique | 02_exercices |
| 🔨 | Implémentation from scratch (numpy, dans mylearn) | notebook |
| 📦 | Implémentation avec une librairie (sklearn, PyTorch, HF) | notebook |
| 🔬 | Expérience ou ablation (faire varier, tracer, conclure) | notebook |
| 🔮 | Prédire avant d'exécuter : l'hypothèse est écrite dans une cellule Markdown **avant** de lancer le code | notebook |
| 🐛 | Chasse au bug : code piégé fourni, à diagnostiquer et corriger | notebook |
| 📈 | Lecture de graphique ou diagnostic (courbes de loss, frontières…) | notebook ou 02 |
| 🧮 | Estimation de Fermi (paramètres, mémoire, temps de calcul) | 02_exercices |
| 🗣️ | Explication façon Feynman (expliquer en 5 lignes à un débutant) | 02_exercices |
| ⚖️ | Cas pratique, éthique, biais | 02_exercices |
| 📄 | Lecture d'un article fondateur (abstract + une figure, questions guidées) | 02_exercices |
| 🎨 | Reproduire une figure du livre | notebook |
| 🏆 | Défi à objectif chiffré | notebook |
| 💼 | Question d'entretien | 02_exercices (section dédiée) |
| 🛠️ | Compétence pro (git, docstring, test, README, refactoring) | notebook ou 02 |

**Composition minimale d'un chapitre standard** (à adapter si le chapitre est conceptuel, comme le ch. 11, en le justifiant dans le rapport) :
🧠 8–12 · 🔁 3 · ✏️/∂ 4–8 · 🔨/📦 6–10 · 🔬 ≥ 1 · 🔮 ≥ 2 · 🐛 ≥ 1 · 📈 ≥ 1 si pertinent · 🧮 ≥ 1 à partir du ch. 16 · 🗣️ 1 · ⚖️ si pertinent · 📄 1 à partir du ch. 16 si un article fondateur existe · 🏆 1 · 💼 3–5 · 🛠️ 1 · flashcards 15–30.

**Difficulté et temps** : ★ application directe (5–15 min) · ★★ standard (15–30 min) · ★★★ approfondi (30–90 min) · ★★★★ défi (plus de 90 min) · 🚀 GPU Colab conseillé. Dans chaque chapitre, la difficulté est croissante.

**En-tête obligatoire de chaque exercice** :
```markdown
### Ex 18.4 — Vérifier son gradient numériquement 🔨 ★★★ ⏱️ 45 min
**Objectif :** une phrase, un seul objectif.
**Prérequis :** Ex 18.2 · livre §18.8 · 0B (dérivées partielles)
**Fil rouge :** MNIST · **mylearn :** `nn/backward.py`
```
Les ID (`chapitre.numéro`) sont **stables** : ils ne changent jamais une fois publiés.

**Règles de rappel espacé** : chaque chapitre N contient 3 questions 🔁, idéalement tirées de N−1, N−3 et d'un chapitre plus ancien (≈ N−7 ou partie 0).

## 12. Modèles de fichiers

### 01_fiche.md
1. En-tête : titre, sections du livre couvertes, temps total estimé, prérequis.
2. Objectifs d'apprentissage (4–8, verbes d'action : calculer, implémenter, expliquer, choisir, diagnostiquer…).
3. L'essentiel en 10 lignes (résumé original, jamais recopié du livre).
4. Concepts clés, avec une intuition, une formule si besoin et un mini-exemple chiffré.
5. Encadrés 🧮 *Rappel maths* nécessaires.
6. Encadrés 🕰️ *Mise à jour* (voir §14).
7. Pièges classiques ⚠️.
8. Liens avec les autres chapitres 🔗.
9. Guide de lecture : dans quel ordre lire le livre et faire les exercices ; les sections lues par le parcours rapide sont marquées ⏩ (liste dans le SYLLABUS).
10. Pour aller plus loin (3–5 ressources de qualité, vérifiées).

### 02_exercices.md
Sections, dans cet ordre : 🧠 Quiz → 🔁 Rappels → ✏️/∂ Papier-crayon → 🧮/🗣️/⚖️/📄 Réflexion → 💼 Entretien → liste des exercices du notebook (renvoi).

### 03_notebook.ipynb (exercices)
1. Titre, badge « Open in Colab », tableau des exercices (ID, type, ★, ⏱️, 🚀).
2. Cellule de setup unique : détection de Colab, montage de Drive et `git pull` ou clonage, installation, imports, `wb.setup(seed=42, fast=FAST_MODE)`.
3. Objectifs et rappel express.
4. Pour chaque exercice : cellule Markdown (en-tête + énoncé) → cellule de code avec `# TODO` et squelette → cellule de vérification (`wb.check(...)` ou appel pytest).
5. Pour 🔮 : une cellule Markdown « Mon hypothèse : … » à remplir **avant** la cellule d'exécution.
6. Fin : bilan, auto-évaluation (3 questions), « pour aller plus loin ».
7. **« Run all » doit aller jusqu'au bout sans planter**, même si rien n'est rempli : les TODO lèvent `NotImplementedError`, que les cellules de vérification capturent en affichant « ⏳ pas encore fait ».
8. L'échafaudage (squelettes, noms de variables fournis) est fort en partie 0 et I, puis diminue progressivement.

### 04_indices.md
Pour **chaque** exercice, trois niveaux dans des blocs `<details>` repliés : **Indice 1** (la direction, quel concept mobiliser) → **Indice 2** (la méthode, les étapes) → **Indice 3** (presque la solution : pseudo-code ou première ligne).

### 05_solutions.md et 05_solutions.ipynb
Pour chaque exercice : la réponse, la démarche détaillée (le *pourquoi*, pas seulement le *comment*), les erreurs fréquentes, une variante ou une piste pour aller plus loin. Les questions 💼 reçoivent une « réponse modèle en 60 secondes » et les relances possibles du recruteur. Le notebook de solutions est **exécuté, sorties conservées**.

### 06_mes_reponses.md
Modèle vide avec un titre par exercice papier, où l'apprenant écrit ses réponses. Il sert de support à la correction (P9).

### flashcards.csv
Format d'import Anki : séparateur `;`, colonnes `front;back;tags`, HTML autorisé, maths en `\( … \)`. Tags : `dlwb::chXX::concept`. 15 à 30 cartes, une idée par carte, des questions actives (pas « Qu'est-ce que X ? » en série).

## 13. Vérification automatique : `wb.check`

- Réponses numériques ou courtes : `wb.check("18.4", value)` compare un **hash** SHA-256 de la valeur normalisée (arrondie au nombre de décimales indiqué dans l'énoncé, chaînes en minuscules sans espaces) au contenu de `src/wb/answers.json`. La solution n'est jamais visible.
- Fonctions et objets : tests fondés sur un oracle (§10) ou vérifications de propriétés dans la cellule.
- `answers.json` est généré par `tools/build_answers.py` **à partir du notebook de solutions exécuté**, jamais à la main.
- Pour chaque check, on vérifie qu'une bonne valeur passe **et** qu'une mauvaise valeur échoue.
- Les messages d'échec sont pédagogiques (« ordre de grandeur correct mais signe inversé ? »), sans révéler la réponse.

## 14. Encadrés standard

```markdown
> 🕰️ **Mise à jour (2026)** — **Le livre :** … · **Aujourd'hui :** … · **Faut-il quand même l'apprendre ?** Oui/Non, parce que … · *Source :* [lien]
> 🧮 **Rappel maths** — …
> 💼 **En entreprise** — …
> ⚠️ **Piège classique** — …
> 🔗 **Lien avec le ch. X** — …
```
Toute affirmation 🕰️ est **vérifiée par recherche web au moment de la génération**, avec une source (documentation officielle, article, référence reconnue). Au-delà de la liste indicative du §8, Claude signale **toute** notion du chapitre dépassée ou qui se fait autrement aujourd'hui : API, bonnes pratiques, conventions, jeux de données retirés…

## 15. Règles pédagogiques

1. Un exercice = un objectif clair ; l'énoncé se suffit à lui-même.
2. **Aucun exercice ne mobilise une notion pas encore vue** (vérifier dans le SYLLABUS ; sinon ajouter un 🧮 ou un renvoi à la partie 0).
3. Progression de difficulté dans chaque chapitre et d'un chapitre à l'autre.
4. On relie toujours l'intuition, les maths et le code.
5. Les fils rouges sont réutilisés, en rappelant ce qu'on a déjà fait sur ces données (« au ch. 12 tu as standardisé Penguins ; ici… »).
6. Les solutions expliquent **pourquoi**, montrent les erreurs fréquentes et proposent une alternative quand elle existe.
7. Le vocabulaire est accessible à un niveau lycée ; tout terme nouveau est défini à sa première apparition.
8. Pas de remplissage : chaque exercice doit apprendre quelque chose de nouveau ou consolider quelque chose d'important.

## 16. Volet reconversion

- 💼 : 3 à 5 questions d'entretien par chapitre, avec réponses modèles.
- 🛠️ : une compétence pro par chapitre (commit git propre, docstring, test unitaire, README, refactoring, lecture de documentation officielle…).
- **Mini-projets de fin de partie** prêts pour le portfolio : cahier des charges, README type, résultats, notebook propre et grille d'évaluation.
- `annexes/metiers.md` relie les compétences du workbook aux métiers (data analyst, data scientist, ML engineer, AI engineer).
- Le projet final se fait sur un dataset choisi par l'apprenant, avec une data card (ch. 29).

## 17. Checkpoints de partie

Dans `checkpoints/partie_X/` :
- **Examen blanc** de 60 à 120 minutes : tous les types d'exercices, toute la partie, plus 10 % de questions sur les parties antérieures ; sujet, corrigé détaillé et barème sur 20.
- **Synthèse** : carte mentale (Mermaid) et fiche de révision d'une page.
- **Mini-projet** dans `projets/partie_X_nom/` : cahier des charges, notebook de départ, solution de référence, grille d'évaluation, extensions possibles.

## 18. Suivi

- `suivi/PROGRESS.md`, tenu par Claude à chaque session : statut de chaque chapitre (planifié, en cours avec l'étape exacte, généré, audité), éléments « à valider sur Colab », section **Calibrage** (retours de l'apprenant et ajustements appliqués), écarts par rapport au SYLLABUS, prochaine étape.
- `suivi/tableau_de_bord.md` : une case à cocher par exercice, groupée par chapitre, avec le temps estimé ; complété par Claude à chaque nouveau chapitre.
- `suivi/journal.md` : pour chaque séance, ce que j'ai fait, compris, pas compris, et la question à poser.
- `suivi/auto_evaluation.md` : grille de compétences (0 à 3) par chapitre.

## 19. Définition du « terminé » pour un chapitre

- [ ] Les 8 fichiers du chapitre sont présents et conformes au §12.
- [ ] La composition minimale du §11 est respectée (ou l'écart est justifié).
- [ ] Toutes les sections du chapitre du livre sont couvertes par au moins un exercice (matrice de couverture dans le rapport).
- [ ] `05_solutions.ipynb` s'exécute en entier en FAST_MODE sur CPU en moins de 10 minutes, sans erreur, sorties conservées.
- [ ] `03_notebook.ipynb` passe « Run all » sans planter.
- [ ] `pytest --impl=ref` est vert ; les tests échouent bien sur les stubs.
- [ ] `answers.json` a été régénéré ; chaque check passe avec la bonne valeur et échoue avec une mauvaise.
- [ ] Les ✏️, ∂ et 🧮 ont été re-résolus par un vérificateur indépendant (sous-agent) ; les écarts sont corrigés.
- [ ] Les 🕰️ ont été vérifiés par recherche web, avec sources.
- [ ] Aucun passage recopié du livre.
- [ ] Glossaire, formulaire, cheatsheets, `tableau_de_bord.md` et `PROGRESS.md` sont à jour.
- [ ] Commit « chXX: … » poussé.

## 20. Sources externes

On privilégie la documentation officielle (PyTorch, scikit-learn, Hugging Face), les articles originaux (arXiv) et les cours reconnus. Chaque lien est vérifié avant d'être cité.

## 21. Versions figées
*Vérifiées le 2026-09-29 (session 1). Sources : dépôt officiel [googlecolab/backend-info](https://github.com/googlecolab/backend-info) (runtime du 2026-09-28), pages PyPI et tags GitHub de chaque projet.*

**Politique** : on fige chaque paquet sur la version **préinstallée sur Colab**, pas sur la toute dernière version. Résultat : les notebooks donnent les mêmes résultats sur Colab et en local, et Colab n'a rien à réinstaller (réinstaller torch sur Colab casse souvent CUDA). Ces versions s'installent sur CPU sous Linux, Windows et macOS Apple Silicon avec Python 3.13 (installation complète testée sous Linux avec Python 3.13 et 3.12, torch CPU). On revérifie à chaque checkpoint de partie (P3) : si Colab a changé, on met à jour `requirements.txt`, `src/wb/_versions.py` et ce tableau.

| Paquet | Version figée (= Colab) | Dernière stable (info) | Vérifié le |
|---|---|---|---|
| Python (Colab) | **3.13** (Colab : 3.13.15, passage à 3.13 le 2026-08-19) | 3.14 | 2026-09-29 |
| torch / torchvision | **2.11.0 / 0.26.0** (CPU en local : index `download.pytorch.org/whl/cpu`) | 2.14.0 / 0.29.0 (2026-09-02) | 2026-09-29 |
| scikit-learn | **1.6.1** | 1.9.1 (2026-09-10) | 2026-09-29 |
| numpy / pandas / matplotlib | **2.1.3 / 2.2.3 / 3.10.0** | 2.5.3 / 3.0.6 / 3.11.2 | 2026-09-29 |
| scipy / seaborn | **1.16.3 / 0.13.2** | 1.18.1 / 0.13.2 | 2026-09-29 |
| gymnasium | **1.3.0** | 1.3.0 | 2026-09-29 |
| transformers / datasets | **5.17.0 / 4.8.5** | 5.17.0 / 5.0.1 | 2026-09-29 |
| tokenizers / huggingface_hub / accelerate / peft | **0.23.2 / 1.31.0 / 1.15.0 / 0.21.0** | | 2026-09-29 |
| xgboost / lightgbm | **3.4.1 / 4.6.0** | 3.4.2 / 4.7.0 | 2026-09-29 |
| umap-learn (+ numba / llvmlite / pynndescent) | **0.5.12** (0.61.2 / 0.44.0 / 0.6.0) | 0.5.12 (numba 0.67.0) | 2026-09-29 |
| shap | **0.52.0** | | 2026-09-29 |
| pytest / nbformat / nbclient | **8.4.2 / 5.11.1 / 0.10.4** | 9.1.1 / 5.11.1 / 0.11.0 | 2026-09-29 |
| jupyterlab (local uniquement) | **4.5.4** | 4.5.4 | 2026-09-29 |
| Keras (non utilisé, §4) | Colab : 3.13.2 | 3.15.1 (2026-07-29) | 2026-09-29 |

**Statut de Keras 3** (pour les encadrés 🕰️ des ch. 23-24) : Keras 3 est un framework **multi-backend** (JAX, TensorFlow, PyTorch ; OpenVINO pour l'inférence seulement), toujours activement développé (3.15.1 en juillet 2026). Le code Keras 2 du livre (2018, `keras` autonome + TensorFlow 1.x) ne tourne pas tel quel : voir le [guide de migration vers Keras 3](https://keras.io/guides/migrating_to_keras_3/). Source : [PyPI keras](https://pypi.org/project/keras/).

**Points d'attention** :
- **Mac Intel** : PyTorch ne publie plus de versions pour macOS x86_64 depuis la 2.3 (dernière : 2.2.2, dépréciation annoncée en [janvier 2024](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690)). Sur un Mac Intel, les chapitres PyTorch (20 et suivants) se font sur Colab.
- Linux : sans l'index CPU, `pip install torch` télécharge la version CUDA (plus de 2 Go).
- Sur Colab, ne jamais lancer `pip install -r requirements.txt` ; `wb.ensure("paquet")` installe un paquet manquant à la version figée si Colab en retire un un jour.

## 22. Journal des décisions et dérogations
*(une ligne par décision : date, décision, raison)*

| Date | Décision | Raison |
|---|---|---|
| 2026-09-29 | Le dépôt s'appelle `cemah2/workbookIA` (public) au lieu de `dl-workbook/` (§7). | Nom choisi par l'apprenant ; le contenu suit exactement l'arborescence du §7. |
| 2026-09-29 | Versions figées = versions préinstallées sur Colab, pas les dernières (§21). Python 3.13. | Résultats identiques en local et sur Colab, rien à réinstaller sur Colab. |
| 2026-09-29 | `requirements.txt` = versions exactes ; `pyproject.toml` = bornes minimales souples + extras (`torch`, `boost`, `umap`, `rl`, `hf`, `xai`, `dev`). | Pratique standard : reproductibilité d'un côté, installabilité de l'autre. |
| 2026-09-29 | Installation locale de torch en deux temps : torch CPU depuis `download.pytorch.org/whl/cpu`, puis `requirements.txt`. | Évite 2 Go de CUDA inutiles sous Linux ; la version installée (`2.11.0+cpu`) satisfait l'épinglage. |
| 2026-09-29 | Sur Colab, aucune installation : le setup vérifie les versions ; `wb.ensure(...)` installe un paquet manquant à la version figée. | Tout est préinstallé ; réinstaller torch casse CUDA. |
| 2026-09-29 | Série temporelle fil rouge = taches solaires mensuelles SILSO (1749-2026, licence CC BY-NC 4.0) + sinus bruité synthétique. | Série réelle, légère, cycle d'environ 11 ans d'amplitude variable ; licence compatible avec un usage éducatif. |
| 2026-09-29 | `penguins_raw.csv` versionné en plus de `penguins.csv`. | Données brutes réelles pour le nettoyage (ch. 12), 53 Ko. |
| 2026-09-29 | MNIST versionné (`data/mnist.npz`, 11,5 Mo) ; Fashion-MNIST et CIFAR-10 téléchargés dans un cache non versionné : `data/downloads/` en local, `/content/wb_cache` sur Colab. | Taille (§9) ; sur Colab, lire depuis Drive est très lent. |
| 2026-09-29 | Sources de secours : Fashion-MNIST via le dépôt GitHub de Zalando (l'URL torchvision est en HTTP simple), CIFAR-10 via Hugging Face en premier (CDN rapide) puis torchvision. Substitut aléatoire seulement avec `allow_substitute=True`, avec un avertissement. | Robustesse ; aucun résultat faux sans avertissement explicite. |
| 2026-09-29 | Textes Gutenberg versionnés en entier (licence Gutenberg conservée), en-tête et pied de page retirés au chargement. `.gitattributes` empêche toute conversion de fins de ligne dans `data/`. | Respect de la licence ; mêmes nombres de caractères sur tous les OS. |
| 2026-09-29 | Outil supplémentaire `tools/build_datasets.py` : reconstruit les fichiers de `data/` depuis leurs sources officielles. | Traçabilité (provenance documentée dans les data cards). |
| 2026-09-29 | `wb.check` : SHA-256 salé par l'ID d'exercice ; tolérance d'un dixième de la dernière décimale, et les deux arrondis acceptés quand la réponse tombe pile sur une limite (0,125) ; `decimals` est stocké avec la réponse (l'argument de `check` est facultatif) ; chaînes : casse, accents, espaces et tirets ignorés, apostrophes typographiques unifiées ; nombres « 3,14 » et « 1 000,5 » acceptés ; un conteneur à un seul élément (Series de `.mode()`, tenseur) accepté pour une réponse scalaire ; tableaux d'entiers qui refusent les nombres à virgule ; booléens acceptant « vrai/faux/oui/non ». `record` refuse une réponse qui s'arrondit à 0. | Hash non réutilisable d'un exercice à l'autre ; pas d'échec injuste pour une question de forme ; une faute d'accent n'est pas une faute de ML. |
| 2026-09-29 | Messages d'échec : signe inversé, pourcentage/proportion, complément 1−p, erreur de 1 (entiers), juste à une décimale près, ordre de grandeur, éléments faux d'un tableau (≤ 100 éléments), forme ou transposition ; « erreurs classiques » déclarées dans le corrigé (`mistakes={...}`, stockées hachées). | Messages pédagogiques du §13 sans révéler la réponse. |
| 2026-09-29 | Côté solutions : `wb.record(id, valeur, decimals=…)` dans une cellule taguée `answer` ; il vérifie lui-même qu'une bonne valeur passe et qu'une mauvaise échoue, puis imprime une ligne `WB_ANSWER` que `tools/build_answers.py` récolte. | Automatise le §13 (« jamais à la main », « bonne valeur passe, mauvaise échoue »). |
| 2026-09-29 | Le module s'appelle `wb.checker` (la fonction reste `wb.check`). | Éviter qu'une fonction et un module portent le même nom. |
| 2026-09-29 | Convention des notebooks d'exercices : les cellules TODO ne font que définir (fonctions, variables `= ...`) ; tout appel au code de l'apprenant se fait dans `with wb.attempt("id"):`, qui affiche « ⏳ pas encore fait » sur `NotImplementedError` ou module mylearn absent. Pas de commandes magiques IPython (`!`, `%`) : on utilise Python (`subprocess`). | « Run all » va au bout sans rien remplir (§12.7) ; les notebooks tournent avec les deux moteurs d'exécution. |
| 2026-09-29 | Cellule de setup standard dans `templates/notebook_setup_cell.py` : Colab → Drive monté, dépôt cloné dans `MyDrive/workbookIA` ou `git pull --ff-only` (remplacé le même jour par `git pull --rebase --autostash`, voir plus bas) ; `WB_ROOT` défini ; `cfg = wb.setup(...)` puis `FAST_MODE = cfg.fast`. La variable d'environnement `WB_FAST_MODE` force le mode. | Un seul modèle pour tous les notebooks ; `run_all_notebooks.py --full` peut forcer le mode complet. |
| 2026-09-29 | mylearn est chargé par `wb.load_mylearn(impl)` (impl = `"learner"`, `"ref"` ou `"stubs"`), qui l'enregistre sous le nom `mylearn` (équivalent à `import mylearn_ref as mylearn`, mais `from mylearn.x import y` fonctionne aussi) ; sous-modules importés à la demande ; imports **relatifs** obligatoires dans les packages. | Même code dans les notebooks d'exercices et de solutions (§7). |
| 2026-09-29 | Tests : 3ᵉ mode `--impl=stubs` (prouve que les tests échouent sur les squelettes, §19), options `--learner-dir` (tests du mécanisme) et `--run-network` (tests qui téléchargent, ignorés par défaut). `NotImplementedError` apparaît comme « ⏳ pas encore implémenté » ; un module pas encore copié est ignoré (skip) avec un message. | Vérification du §19 ; un `pytest` sans option reste rapide et lisible pour l'apprenant. |
| 2026-09-29 | `templates/mylearn_stubs/MANIFEST.json` associe chaque chapitre à ses stubs (ajouts seulement) ; fichiers de base : `__init__.py` et `_example.py` (fonction `mean`, exemple du mécanisme, réutilisable comme tout premier exercice en 0A). | `start_chapter.py` sait quoi copier sans jamais modifier un stub publié (§7). |
| 2026-09-29 | `start_chapter.py` copie dans `mon_travail/<dossier du chapitre>/` (ex. `mon_travail/ch18_backprop/`) le notebook **et** `06_mes_reponses.md` ; `--init` crée `mon_travail/mylearn/` et `mon_travail/suivi/` ; création exclusive, lecture de la source avant écriture et nettoyage en cas d'échec (jamais d'écrasement ni de fichier tronqué). | L'apprenant écrit ses réponses papier dans sa copie, jamais dans un fichier que Claude met à jour. |
| 2026-09-29 | `mon_travail/` n'est matérialisé que par des fichiers `.gitkeep` créés en session 1 ; Claude n'y écrit plus jamais ensuite. | Arborescence du §7 sans empiéter sur l'espace de l'apprenant. |
| 2026-09-29 | `run_all_notebooks.py` : moteur `nbclient` (vrai noyau Jupyter) par défaut, moteur de secours `inprocess` si Jupyter est absent ; budgets du §4 signalés (10 min par notebook, 3 min par cellule) ; ne touche jamais `mon_travail/`. | Exécution possible même dans un environnement restreint. |
| 2026-09-29 | Device : `cuda` sinon `cpu` ; le GPU Apple (`mps`) seulement avec `wb.setup(prefer_mps=True)`. | Opérations manquantes et non-déterminisme sur MPS : déroutant pour un débutant. |
| 2026-09-29 | Style des graphiques : palette catégorielle de 8 couleurs lisible par les daltoniens, ordre fixe ; au-delà de 8 classes, `tab10` + étiquettes directes. | Lisibilité et accessibilité. |
| 2026-09-29 | Export Anki dans `exports/dlwb_anki.csv` (non versionné, régénérable), avec les lignes d'en-tête Anki (`#separator`, `#html`, `#tags column`, `#deck`). | Import en un clic dans Anki. |
| 2026-09-29 | Ajouts hors §7 : `CLAUDE.md` (consignes pour les sessions suivantes), `templates/notebook_setup_cell.py`, `.gitattributes`, `tools/build_datasets.py`, `tests/infra/` (tests de `wb` et des outils). | Continuité entre les ~50 sessions ; l'arborescence du §7 reste intacte. |
| 2026-09-29 | Environnement de génération : PyPI n'est joignable qu'en passant par le proxy (`env -u NO_PROXY -u no_proxy pip …`) ; le serveur CIFAR de Toronto y est très lent. Consigné dans `CLAUDE.md`. | Éviter de perdre du temps aux sessions suivantes. |
| 2026-09-29 | **Fichiers de suivi de l'apprenant dans `mon_travail/suivi/`** (tableau de bord, journal, auto-évaluation). Les fichiers de `suivi/` deviennent des **modèles** tenus par Claude, découpés en sections `<!-- wb:section ID -->…<!-- wb:end ID -->` ; `start_chapter.py` copie les modèles une fois, puis ajoute à la fin de la copie de l'apprenant les sections manquantes, sans rien modifier d'autre. `suivi/PROGRESS.md` et `suivi/remediation.md` restent tenus par Claude. Précise §7 et §18. | Le §18 faisait écrire l'apprenant (cases cochées) et Claude (nouvelles sections) dans les mêmes fichiers : conflit git garanti au `git pull` (constaté par simulation lors de la relecture indépendante). |
| 2026-09-29 | Mise à jour du dépôt : `git pull --rebase --autostash` (cellule de setup Colab, avec `safe.directory` et une identité git de secours) ; en local, `git config --global pull.rebase true` et `rebase.autoStash true`. | Avec `--ff-only`, un simple commit local de l'apprenant bloquait toute mise à jour ; comme Claude n'écrit jamais dans `mon_travail/`, le rebase ne peut pas créer de conflit (vérifié par simulation). |
| 2026-09-29 | Nuance du §13 : le hash est un **garde-fou pédagogique, pas un coffre-fort**. Une réponse simple (petit entier, mot court) peut être retrouvée par force brute ; la discipline reste celle de l'apprenant (METHODE §2). Les empreintes par élément d'un tableau sont liées à leur position. | Honnêteté sur ce que le mécanisme garantit (relevé par la relecture indépendante). |
| 2026-09-29 | Relecture indépendante de l'infrastructure par un sous-agent ; 21 défauts confirmés puis corrigés et couverts par des tests (tenseurs PyTorch particuliers, sous-paquets `mylearn.nn.*`, `from mylearn import x` dans `wb.attempt`, torch installé mais cassé → repli CPU avec message, consoles Windows non UTF-8, motifs non développés par PowerShell, frontières de décision avec étiquettes −1/+1 ou textuelles, délai maximal de 30 s sur le téléchargement scikit-learn, type de note Anki non imposé…). | Rien n'est livré sans vérification indépendante (METHODE §1.4). |
| 2026-09-30 | **Syllabus = données + outil.** Une fiche JSON par chapitre dans `docs/syllabus/data/` (exercices, sections du livre, notions requises et enseignées, points 🕰️, API `mylearn`, parcours, temps) ; `tools/syllabus.py check` valide tout, `build` génère `docs/SYLLABUS.md`, `docs/PARCOURS.md`, le modèle `suivi/tableau_de_bord.md` et complète `MANIFEST.json` (ajouts seulement). Les signatures affichées sont lues dans les stubs. On n'édite jamais les fichiers générés à la main ; un test vérifie qu'ils sont à jour. | Un contrat de 2 000 exercices ne se maintient pas à la main : ID uniques, prérequis antérieurs, couverture de chaque section du livre, stubs cohérents, tout est vérifié à chaque session. |
| 2026-09-30 | **Conventions d'ID** (précise §11) : `N.Qk` 🧠, `N.Rk` 🔁, `N.Ek` 💼, `N.k` consécutifs pour les autres types dans l'ordre papier (✏️, ∂) → réflexion → notebook ; checkpoints `CPk.n` et projet final `PF.n` pour **tous** les types (🧠 et 💼 compris) ; étapes de mini-projet `MPk.n`. Sections internes des chapitres sans livre : 0A → `100.x`, 0B → `101.x`, Bn → `20n.x`. Un ID devient définitif quand son chapitre est publié ; avant, le syllabus peut renuméroter (session 2 : 49 ID décalés par des insertions en 0A, 8, 12 et 15). | Numérotation lisible et stable ; §17 (« tous les types ») respecté dans les checkpoints. |
| 2026-09-30 | **Tous les stubs `mylearn` existent dès la session 2** (38 modules, interface figée) ; la référence `solutions/mylearn_ref/` s'écrit avec chaque chapitre. Les tests n'exigent le miroir de référence que pour les chapitres publiés (dossier `chapitres/…` présent). Une fonction d'aide fournie sans exercice est implémentée dans le stub et marquée `Provided:` dans sa docstring. | Demande de la session 2 (figer l'interface) ; un stub publié ne change plus (§7). |
| 2026-09-30 | **Modules ajoutés au §8** : `utils` (0A), `multiclass` (7), `logistic` (13), `sequences` (22), `vae` (25), `gen_metrics` (27), `tokenizer` et `embeddings` (B2), `retrieval` (B4), `diffusion` (B5), `explain` (B6), `deep_rl` (B8). Pas de module aux ch. 1, 15, 23, 24, 28, 29, B1 et B7 (pratique des bibliothèques). | Chaque exercice 🔨 doit pouvoir être testé contre un oracle (§10). |
| 2026-09-30 | **Conventions de `mylearn`** : classes à la scikit-learn (`__init__` ne fait que stocker, `fit` renvoie `self`, attributs appris suffixés `_`, `score`) ; `random_state` pour les classes, `rng` pour les fonctions ; `X` `(n_samples, n_features)`, images `(N, C, H, W)`, séquences `(N, T, D)` ; poids denses `(n_in, n_out)` avec `z = x @ W + b`, filtres de convolution `(C_out, C_in, kh, kw)` comme PyTorch ; noms d'hyperparamètres = ceux de la bibliothèque oracle (`eta0` du Perceptron, `learning_rate` de la logistique et du boosting, `lr` des optimiseurs, `alpha` de Ridge et Lasso, `lam` de `l2_penalty`) ; `info.*` en bits par défaut, sauf `log_loss` en nats comme scikit-learn (paramètre `base`) ; `cross_val_score(…, scoring)` suit la convention scikit-learn `scoring(estimateur, X, y)`. Tableau des conventions de régularisation dans `annexes/formulaire.md`. | Relecture indépendante : ces écarts auraient été figés à la publication ; un tableau unique évite de confondre α, λ/2 et `weight_decay`. |
| 2026-09-30 | **Parcours** (rapide, maths, code ; le complet contient tout) : chaque exercice porte ses parcours. Un prérequis **de code** (exercice de notebook qui réutilise un exercice de notebook antérieur) est toujours dans le même parcours : `tools/syllabus.py` le vérifie. Les autres prérequis sont des connaissances : hors parcours, on lit le corrigé (compté pour un tiers du temps). Checkpoints, mini-projets et projet final font partie des quatre parcours. Le parcours rapide lit seulement les sections du livre couvertes par ses exercices pratiques (marquées ⏩ dans la fiche, §12). | Relecture indépendante : 104 exercices du parcours rapide dépendaient d'exercices absents du parcours, dont une trentaine de chaînes `mylearn` qui auraient fait échouer les tests. |
| 2026-09-30 | **Repli sur la référence** : `wb.load_mylearn("learner", fallback="ref", chapter="N")` (cellule de setup des notebooks d'exercices, `nbbuild.setup_cell("exercise", chapter=…)`) prend dans `solutions/mylearn_ref/` un module d'un chapitre **antérieur** absent de la librairie de l'apprenant, jamais un module du chapitre en cours ; un message l'indique. `pytest` (mode apprenant) utilise le même repli pour les dépendances, mais ne teste que les modules de l'apprenant. `start_chapter.py` liste les modules pris dans la référence, sans les copier. | Sauter un chapitre (parcours, reprise en cours de route) ne casse pas les suivants (ex. `ensemble.py` du ch. 14 importe `tree.py` du ch. 13), et une vérification ne peut jamais réussir avec le code de la référence à la place de celui de l'apprenant. |
| 2026-09-30 | **Temps d'étude** = lecture (≈ 4 min par page du livre, fiche comprise ; listings Keras des ch. 23-24 survolés) + exercices + 2 min par flashcard ; checkpoint = examen + 90 min de synthèse + mini-projet. Calendrier indicatif à 10 h par semaine à partir du 5 octobre 2026. | Estimation explicite et recalculée automatiquement ; à calibrer avec les retours de l'apprenant (PROGRESS, Calibrage). |
| 2026-09-30 | **Relecture indépendante du syllabus** (sous-agent) et corrections : notions enseignées avant usage (récursivité 0A.44, `@dataclass` en 0A.47, TF-IDF en 13.30, encadrés locaux PMI en B2, Huber en B8, `nn.Embedding` en 24, projection en 25 ; 5.6 sans Taylor) ; briques d'employabilité (MAE, test par permutation et p-valeur en 8.26, pandas dates/`merge`/`pivot_table` en 12.14, SQL en 12.15, classes déséquilibrées en 15.26) ; B7 faisable dès la fin de la partie V ; 24 exercices en doublon réorientés (boucles PyTorch des ch. 20/23, sauvegarde, décodage, détection de langue, questions 💼…) ; durées rehaussées (0A, 18.26, 20.22, 21.15, B8.20) ; rappels R3 des ch. 21-29 tournés vers les parties 0 et I ; parcours rapide allégé (≈ 520 h au lieu de 528 h, malgré l'ajout des prérequis manquants). Une seconde relecture indépendante a vérifié ces corrections et relevé 8 défauts, corrigés : numéros de figures du livre touchés par la renumérotation, contournement du repli par un sous-paquet (`nn`), fixture `mylearn` qui aurait testé la référence, dépendances `mylearn` implicites (méthodes `score` qui réutilisent `r2_score`), champs périmés d'exercices réorientés, prérequis de B7 et lecture de la fiche dans le parcours rapide. | Vérification exigée par la session 2 ; détail dans les notes des chapitres concernés. |
| 2026-09-30 | METHODE §3 : environ **80 sessions** de génération au lieu d'environ 50 (67 sessions de chapitres prévues par le syllabus, plus 6 checkpoints, 3 audits, la mise en place, le syllabus et la finalisation). | Le syllabus compte environ 2 000 exercices ; la moitié des chapitres demande deux sessions. |
| 2026-09-30 | **Hors bible, ajouté** : un exercice SQL (12.15, `sqlite3` et `pd.read_sql`) et le dataset **Adult** (MP3, B6 ; `fetch_openml` + copie de secours à versionner) ; 20 Newsgroups (scikit-learn) en B4 ; `load_digits` (scikit-learn) en 29.14 ; CliffWalking (gymnasium) au ch. 26. Dépendances prévues, à figer au moment du chapitre : `fastapi`, `uvicorn`, `pydantic` (B7), `diffusers` (oracle du B5). | Compétences attendues en entretien ; aucun dataset lourd versionné (§9). |
| 2026-09-30 | **Constructeur de chapitre** `tools/chapters/build_chXX.py` : une seule source (énoncé, squelette TODO, vérification, solution, `wb.record`) produit `03_notebook.ipynb` **et** `05_solutions.ipynb`. Reconstruire efface les sorties des solutions : on relance toujours `run_all_notebooks.py … --inplace` puis `build_answers.py`. | Énoncé, vérification et corrigé ne peuvent plus diverger ; une correction se fait à un seul endroit. |
| 2026-09-30 | **Sous-ID de vérification** : un exercice à plusieurs réponses est vérifié par `wb.check("0A.14a", …)`, `"0A.14b"`… ; l'ID du syllabus reste celui de l'exercice. Les exercices ✏️ se vérifient dans une « Partie 0 » du notebook (variables `answer_0A_1a`…) ; une forme ou « erreur » se donne en chaîne (`"(5, 3)"`). | Un message ✅/❌ par sous-question ; les exercices papier gardent une vérification automatique sans révéler la réponse (§13). |
| 2026-09-30 | **Chapitre généré en plusieurs sessions** : un fichier `chapitres/<dossier>/EN_COURS.md` signale le chantier ; tant qu'il existe, `start_chapter.py` copie tout **sauf** le notebook (⏳ « plus tard » ; `--force` pour le copier quand même) ; on supprime le fichier à la dernière session, et la même commande copie alors le notebook. | Une copie à moitié écrite ne serait jamais complétée, puisque les fichiers de l'apprenant ne sont jamais écrasés. |
| 2026-09-30 | `wb.record` renvoie un `RecordedAnswer` (un `dict`) qui ne s'affiche pas dans un notebook ; le repli sur la référence (§22, « Repli ») ne sert **jamais** les fichiers de base (`__init__.py`, `_example.py`), seulement les modules des chapitres antérieurs ; les messages d'aide citent le vrai numéro de chapitre. | La dernière ligne d'une cellule `answer` affichait l'empreinte ; 0A.26 aurait pu passer avec le `mean` de la référence. |
| 2026-09-30 | **0A en deux sessions** : session 3 = fiche complète, `02_exercices.md`, notebook parties 0 et A–E (0A.1–0A.36), indices, solutions, 30 flashcards, référence et tests de `utils` ; session 4 = parties F–H (0A.37–0A.67). Seuil « heavy » unifié à **≥ 4500 g** dans tout le chapitre (0A.20, 0A.28, 0A.30, fiche, flashcards). | Découpage prévu par le syllabus ; le test « apprenant » indépendant a montré que deux seuils différents (≥ et >) piégeaient un apprenant qui réutilise sa propre règle. |
| 2026-09-30 | **Vérification indépendante de 0A (session 3)** : exercices papier et quiz re-résolus à l'aveugle (116/116 identiques) ; notebook résolu par un « apprenant » simulé avec d'autres méthodes que le corrigé (90/91 acceptées). Corrections : 2 solutions, 6 indices, 2 phrases de la fiche, 9 énoncés ou messages (méthode imposée en 0A.31e, messages ciblés pour `dropna(subset=…)`, `mass_known` au lieu de `X`, cellule 0A.17 relançable…). Règle retenue pour la suite : quand plusieurs méthodes correctes donnent des valeurs différentes (tirages aléatoires), l'énoncé impose la méthode. | §19 ; §13 (une bonne réponse doit passer). |
| 2026-09-30 | **`wb.run_pytest(tests, subject=…, name=…)`** : copie les tests écrits dans un notebook (et la fonction testée, renommée au besoin) dans un fichier temporaire, puis lance **vraiment** pytest dans un autre processus ; renvoie `passed`, `failed`, `errors`. Sert aux exercices 🛠️ « écrire ses tests » (0A.62) et au *mutation testing* : les tests de l'apprenant doivent passer sur une version juste et échouer sur des versions buggées fournies. | Écrire des tests est une compétence pro clé ; un vrai pytest (collecte, `parametrize`, `raises`) vaut mieux qu'un mini-lanceur maison. |
| 2026-09-30 | Exercices mylearn dans un notebook : la cellule de vérification **recharge** la librairie de l'apprenant (`wb.load_mylearn("learner", …)`) avant de vérifier, puis lance `pytest -k <fonction>` (avec la liste des tests en échec) ; le notebook de solutions lance les mêmes tests avec `--impl=ref`. | Pas de redémarrage du noyau à chaque essai ; le retour des tests est lisible, même sur Colab. |
| 2026-09-30 | Règles de rédaction issues de la vérification de 0A : chaque cellule de vérification reste **relançable** (noms de variables propres à chaque exercice) ; les squelettes de classe déclarent **toutes** les méthodes attendues avec `raise NotImplementedError` (sinon une `TypeError` arrête « Run all ») ; une réponse « texte » demandée au format `"A, B"` est acceptée aussi bien comme chaîne que comme tuple ; les lettres d'un énoncé correspondent exactement aux sous-ID vérifiés ; un point « à lire sur un graphique » est choisi loin de toute frontière entre classes. | Défauts trouvés par le test « apprenant » indépendant de la session 4. |
| 2026-09-30 | **Figures des fiches** : un script `tools/chapters/figures_chXX.py` les calcule (jamais copiées du livre) dans `chapitres/<dossier>/figures/` (PNG versionnés, 20 à 120 ko chacun), avec le style `wb.plot`, la virgule décimale sur les graduations linéaires et dans les légendes, et les vecteurs en gras. **Kit commun des constructeurs** : `tools/chapters/chapter_kit.py` (`Ex`, `Part`, `Paper`, cellules des deux notebooks) sert à partir de 0B ; `build_ch00a.py` garde sa copie. | Figures reproductibles et cohérentes avec la typographie du texte ; un seul endroit pour la mécanique des notebooks. |
| 2026-09-30 | **0B en deux sessions** : session 5 = fiche et figures, `02_exercices.md`, indices et solutions du quiz, des rappels, des exercices papier (0B.1–0B.32) et de l'entretien, flashcards, partie 0 du notebook (174 vérifications ✏️), référence et tests de `linalg_basics` (faits en avance) ; session 6 = notebook, parties A–D (0B.33–0B.54). | Découpage prévu par le syllabus (`generation_sessions = 2`). |
| 2026-09-30 | Exemples de format dans les énoncés et les notebooks (« écris un vecteur comme `[7, -2]` ») : **toujours neutres**, jamais une vraie réponse. Messages des erreurs classiques (`mistakes`) : ils disent **où** est l'erreur, sans donner la bonne valeur ni un calcul qui y mène directement. | La relecture de 0B a trouvé 5 réponses données en exemple et 10 messages trop explicites. |
| 2026-09-30 | **`wb.check`, diagnostics** : « trop grande / trop petite d'au moins un facteur 10 » seulement à deux puissances de 10 d'écart (9 contre 11 donne « trop grande ») ; un nombre non entier pour une réponse entière est une **mauvaise valeur** (« la réponse attendue est un nombre entier »), plus un mauvais type ; le message « à 1 près » évoque aussi une erreur de calcul, pas seulement des bornes. Tests dans `tests/infra/test_wb_check.py`. | Relevé par la vérification à l'aveugle de 0B : 11 contre 9 recevait « au moins un facteur 10 », ce qui est faux. |
| 2026-09-30 | **Vérification indépendante de 0B (session 5)** : ✏️, ∂ et 🧮 re-résolus à l'aveugle (174/174 identiques, variantes de saisie acceptées : fractions, tuples, `float`) ; fiche relue (71 blocs `>>>` rejoués, aucune erreur de maths) ; indices et solutions relus. Corrections : 11 énoncés précisés (0B.Q4 : q = −1 au lieu de −2 ; 0B.2i « entiers relatifs » ; 0B.7 rappel cos(π − x) ; 0B.10f valeur exacte et vraie valeur à 4 décimales ; 0B.26 fractions exactes ; 0B.27 « dé équilibré aux faces 1, 1, 2, 3, 3, 3 »…), 9 erreurs classiques ajoutées, 4 renvois de chapitre, 3 indices, 12 points de la fiche (loss, ŷ et learning rate définis avant usage, Iₙ pour une matrice non carrée, E[X + Y] admis…), 6 figures. | Même protocole qu'en 0A. |
| 2026-09-30 | **Kit des constructeurs, cellules « après »** : dans `Ex.after`, les types `todo`/`check` (notebook d'exercices) et `solution`/`record` (notebook de solutions) permettent une question de suite **après** l'expérience d'un 🔮 (0B.34 g, 0B.50 e–f). Les aides des cellules de tests (`run_utils_tests` en 0A, `run_linalg_tests` en 0B) lancent pytest avec `COLUMNS=200`, pour que chaque test en échec affiche sa raison. | Un 🔮 prédit avant d'exécuter, puis creuse le résultat ; un « FAILED » sans raison n'aide pas un débutant. |
| 2026-09-30 | **`wb.check`, suite** : un tableau d'entiers attendu accepte des flottants à $10^{-6}$ près d'entiers (racines calculées, `2.9999999999999996`), et le message « valeurs entières » suggère `round()` ; le message d'une chaîne fausse distingue « choix possible mais pas le bon » et faute d'orthographe. Les réponses à choix (🔮) reçoivent des `mistakes` pour les autres choix plausibles, et les valeurs théoriques demandées « simulées » ont leur message. | Test « apprenant » de 0B : `np.roots` refusé avec un message sur les probabilités ; une prédiction fausse renvoyait vers l'orthographe. |
| 2026-09-30 | **0B.54 renommé** « L'ordre des produits : calculer A·B·C·v des dizaines de fois plus vite » (au lieu de « cent fois ») ; objectif 🏆 : au moins 20 fois plus rapide (mesuré : ≈ 50 sur 2 cœurs), pour ≈ 670 fois moins de multiplications. | Un produit matrice-vecteur est limité par la mémoire : le gain en temps reste très inférieur au gain en calculs, et « cent fois » n'est pas atteint sur un CPU ordinaire. |
| 2026-09-30 | **Vérification indépendante de 0B (session 6)** : notebook résolu par un « apprenant » simulé avec sa propre librairie (51/51 tests) et d'autres méthodes que le corrigé (121/121 vérifications acceptées, environ 150 formats de saisie essayés) ; indices et solutions du notebook relus, valeurs recalculées. Corrections : 5 réponses données par un énoncé, un commentaire ou une sortie de cellule (0B.41 a, 0B.45 a–c, 0B.47 b, 0B.50 e, 0B.52 a), 0B.50 e–f repensé (petit pas contre grand pas), frontière du ch. 5 respectée (plus rien sur le choix de $h$), 6 erreurs classiques corrigées ou ajoutées, 5 messages trop explicites, 2 indices 1 trop explicites, indices de `linalg_basics` complétés (ils ne passaient pas les tests), tests de `linalg_basics` resserrés (types `float`, sans réponses d'exercices). | §19. |
| 2026-09-30 | **Chapitre 1 en une session** (session 7, comme prévu par `ch01.json`) : fiche et 5 figures, `02_exercices.md`, notebook complet (partie 0 et parties A–D), indices, solutions, 20 flashcards. Aucun module `mylearn` : c'est un chapitre de découverte, et la première boucle d'entraînement (1.16) s'écrit dans le notebook. | Chapitre de vocabulaire, sans maths nouvelles ni API à figer. |
| 2026-09-30 | **Découpage Penguins du ch. 1** : `load_penguins(dropna=True)` (333 manchots), `np.random.default_rng(42).permutation(333)`, les 100 premiers pour le test et les 233 autres pour l'entraînement ; donné tout fait en tête de la partie B du notebook et repris par la figure de la fiche (entraînement seulement). | Les réponses vérifiées (1.14–1.20, 1.25) dépendent du découpage : un seul, imposé, pour tout le chapitre. |
| 2026-09-30 | **Kit des constructeurs, suite** : `Ex.after` accepte aussi `todo_md` (Markdown du notebook d'exercices seulement, la cellule « 📝 Mes réponses ») et `solution_md` (Markdown du notebook de solutions seulement, les réponses rédigées). Les questions « pourquoi » sans cellule 📝 se notent dans « Notes sur le notebook » de `06_mes_reponses.md`. Les exercices 🔬/🐛 non vérifiables par une valeur ont des vérifications de propriétés (`verdict`, ✅/❌) en plus des réponses rédigées. | Les expériences du ch. 1 (learning rate, fuite, apprentissage par cœur) appellent des réponses écrites à côté de l'expérience. |
| 2026-09-30 | **`wb.check`, suite** : un facteur 100 exact donne un message neutre (« exactement 100 fois trop grande / petite : un pourcentage au lieu d'une proportion, ou une division par 100 oubliée ? ») ; une réponse entière saisie avec des décimales est arrondie avant la recherche des erreurs classiques (« Presque : … arrondis (round) » si l'arrondi est juste) ; une date (`Timestamp`, `datetime`…) est refusée avec un message qui montre `d.year`. Tests dans `tests/infra/test_wb_check.py`. | Vérification à l'aveugle et « apprenant » du ch. 1 : 99,05 au lieu de 0,9905, un effectif calculé en flottant, une année saisie comme une date. |
| 2026-09-30 | **Rattrapage des fichiers communs** : sections 0A, 0B et 1 ajoutées au modèle `suivi/auto_evaluation.md` (oubliées aux sessions 3 à 6) ; glossaire : lignes cassées réparées, colonne « Ch. » remplie pour les termes existants, 45 termes du ch. 1 ; annexes `formulaire.md`, `erreurs_frequentes.md` et `cheatsheets/sklearn.md` complétées ; data cards mises à jour (usage au ch. 1). | §19 : chaque chapitre enrichit les annexes et le suivi. |
| 2026-09-30 | **`ch01.json` corrigé** : 1.4 et 1.11 passent de ★ à ★★ (15 min), 1.22 de 25 à 30 min et d'une correction manuelle à `wb.check` ; deux notes 🕰️ rectifiées (en 2018, les GAN étaient déjà un sujet de recherche en plein essor ; « semi-supervisé » avait déjà son sens actuel). | Relectures indépendantes : durées sous-estimées, 1.22 vérifiable par une graine imposée, faits inexacts. |
| 2026-09-30 | **Vérification indépendante du ch. 1 (session 7)** : papier et 🧮 re-résolus à l'aveugle (aucun désaccord ; suggestions de messages reprises) ; notebook résolu deux fois par un « apprenant » simulé avec d'autres méthodes que le corrigé (79 puis 81 ✅, 0 ❌) ; fiche relue trois fois contre le texte du livre. La 1ʳᵉ version suivait le livre de trop près : **réécrite entièrement** (exemples du workbook, anecdotes du livre remplacées par des renvois) ; les relectures suivantes ont corrigé des reformulations encore proches, des réponses données d'avance (1.4 c, 1.9 f, 1.14, 1.15 c, 1.19 a) et des faits : le jeu de validation vient du livre lui-même (§8.4 ; l'encadré 🕰️ est devenu une remarque dans le piège ⚠️), et « fuite de données » couvre aussi le test utilisé pour entraîner ou régler le modèle, comme au §8.3 du livre. | §3 (ne pas recopier le livre, même paraphrasé de près) et §19. |
