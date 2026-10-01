# Workbook Deep Learning : de la théorie à la pratique

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cemah2/workbookIA/blob/main/00_setup/demo.ipynb)

Un cahier d'exercices complet, en français, pour étudier le livre **« Deep Learning: From Basics to Practice »** d'Andrew Glassner (2018, 2 volumes), le mettre à jour (2026) et le compléter par des chapitres bonus modernes (Transformers, LLM, diffusion…).

Objectif : une **reconversion professionnelle** vers les métiers de l'IA et de la data. Chaque chapitre apporte de la compréhension (intuition, maths, code), une librairie construite de ses propres mains (`mylearn`), des réflexes professionnels (tests, git, reproductibilité) et des questions d'entretien.

> Le livre n'est pas inclus dans ce dépôt (droits d'auteur). Les exercices sont originaux ; ils renvoient aux sections du livre sous la forme « (livre §18.5) ». Le code et les figures du livre sont publiés par l'auteur sous licence MIT : [notebooks du volume 1](https://github.com/blueberrymusic/DeepLearningBookCode-Volume1) (le dépôt officiel du volume 2 n'est plus en ligne ; une copie existe : [bssrdf/DeepLearningBookCode-Volume2](https://github.com/bssrdf/DeepLearningBookCode-Volume2)).

---

## 🚀 Démarrage rapide sur Google Colab (recommandé)

1. Clique sur le badge **Open in Colab** ci-dessus (notebook `00_setup/demo.ipynb`).
2. Lance la première cellule (**setup**) et autorise l'accès à Google Drive. Le dépôt est cloné dans `Mon Drive/workbookIA` ; aux ouvertures suivantes, la même cellule fait un `git pull` pour récupérer les nouveaux chapitres.
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
python 00_setup/check_env.py        # tout doit être ✅
python tools/start_chapter.py --init  # crée ta librairie mon_travail/mylearn
jupyter lab                         # ouvre 00_setup/demo.ipynb
```

## 📚 Sommaire

| Partie | Chapitres | Statut |
|---|---|---|
| **Mise en place** | [`00_setup/`](00_setup/) : installation, démo, vérification de l'environnement | ✅ |
| **0 · Prérequis** | 0A Python, notebooks et outils · 0B Maths du lycée au ML | ✅ |
| **I · Fondations** | 1 Introduction · 2 Hasard et statistiques · 3 Probabilités et qualité · 4 Règle de Bayes · 5 Courbes et surfaces · 6 Théorie de l'information | 🛠️ (1 à 5 ✅) |
| **II · Concepts** | 7 Classification · 8 Entraînement et test · 9 Overfitting et underfitting · 10 Neurones · 11 Apprentissage et raisonnement | 📅 |
| **III · ML classique** | 12 Préparation des données · 13 Classifieurs · 14 Ensembles · 15 scikit-learn | 📅 |
| **IV · Réseaux** | 16 Réseaux feed-forward · 17 Fonctions d'activation · 18 Rétropropagation · 19 Optimiseurs · 20 Deep learning et premiers pas en PyTorch | 📅 |
| **V · Architectures** | 21 CNN · 22 RNN · 23-24 PyTorch en pratique | 📅 |
| **VI · Génératif et RL** | 25 Autoencodeurs et VAE · 26 Apprentissage par renforcement · 27 GAN · 28 Applications créatives · 29 Datasets et projet final | 📅 |
| **VII · Bonus** | B1 Transfer learning · B2 Tokenisation et embeddings · B3 Attention et Transformers · B4 LLM en pratique · B5 Diffusion · B6 Explicabilité et éthique · B7 Du notebook à la production · B8 RL moderne | 📅 |
| **Fin** | Projet final | 📅 |

✅ disponible · 🛠️ en cours · 📅 planifié. Le plan détaillé de chaque chapitre (≈ 2 000 exercices, temps d'étude, calendrier indicatif) est dans [`docs/SYLLABUS.md`](docs/SYLLABUS.md) ; l'état d'avancement est suivi dans [`suivi/PROGRESS.md`](suivi/PROGRESS.md). Un checkpoint (examen blanc, synthèse, mini-projet) clôt chacune des parties I à VI.

**Quatre parcours** ([`docs/PARCOURS.md`](docs/PARCOURS.md)) : complet (≈ 890 h), **rapide**, l'essentiel pour être employable (≈ 520 h), orienté maths et orienté code. Chaque exercice indique ses parcours (R, M, C) ; checkpoints, mini-projets et projet final sont communs à tous.

## 🗂️ Organisation du dépôt

| Dossier | Contenu |
|---|---|
| `chapitres/chXX_nom/` | pour chaque chapitre : fiche, exercices, notebook, indices, solutions, modèle de réponses, flashcards |
| `mon_travail/` | **ton espace** : copies de travail des notebooks, ta librairie `mylearn` et tes fichiers de suivi (Claude n'y écrit jamais) |
| `src/wb/` | utilitaires du workbook : `wb.setup`, `wb.check`, datasets, données synthétiques, graphiques |
| `solutions/mylearn_ref/` | implémentation de référence de `mylearn` |
| `templates/mylearn_stubs/` | squelettes de `mylearn`, copiés dans ton espace chapitre par chapitre |
| `tests/` | tests de `mylearn` (et de l'infrastructure dans `tests/infra/`) |
| `data/` | datasets légers versionnés et leurs fiches (`data/cards/`) |
| `annexes/` | glossaire FR↔EN, formulaire, cheatsheets, erreurs fréquentes, métiers, ressources |
| `suivi/` | avancement de la génération (`PROGRESS.md`), remédiation, et les modèles de tes fichiers de suivi |
| `checkpoints/`, `projets/` | examens blancs et mini-projets de portfolio |
| `tools/` | scripts : démarrer un chapitre, exécuter les notebooks, construire les réponses, exporter les flashcards |
| `docs/` | la spécification du workbook (`BIBLE.md`), la méthode, le syllabus |

## 🔁 Routine pour chaque chapitre

1. `python tools/start_chapter.py 18` : copie le notebook et les squelettes `mylearn` du chapitre dans `mon_travail/` et complète tes fichiers de suivi, sans jamais écraser ton travail.
2. Lis la fiche (`01_fiche.md`), puis le chapitre du livre en suivant son guide de lecture.
3. Quiz 🧠 et rappels 🔁 sans le livre, puis exercices papier ✏️ et ∂ dans `mon_travail/<chapitre>/06_mes_reponses.md`.
4. Notebook : les 🔮 d'abord, à l'instinct. `wb.check` te dit si ta réponse est juste sans la révéler.
5. `python -m pytest tests/` : tes fonctions `mylearn` comparées à des bibliothèques de confiance.
6. Corrige avec les indices (`04_indices.md`) puis les solutions, réponds aux questions 💼 à voix haute, importe les flashcards dans Anki.
7. Note ta séance dans `mon_travail/suivi/journal.md`, ton niveau dans `mon_travail/suivi/auto_evaluation.md`, et coche tes exercices dans `mon_travail/suivi/tableau_de_bord.md`.

## 🧩 Ta librairie mylearn

Chapitre après chapitre, tu écris ta propre librairie de machine learning dans `mon_travail/mylearn/` : `start_chapter.py` y copie les squelettes (signatures, docstrings, `raise NotImplementedError`) et `pytest` compare ton code à des bibliothèques de confiance (NumPy, scikit-learn, PyTorch). Toutes les signatures sont fixées dans [`docs/SYLLABUS.md`](docs/SYLLABUS.md).

**Tu as sauté un chapitre ?** Pas de problème : quand un module d'un chapitre **précédent** te manque (par exemple `tree.py` du ch. 13, que `ensemble.py` du ch. 14 réutilise), les notebooks et les tests prennent la version de référence à sa place, avec un message. Les modules du chapitre que tu étudies ne sont jamais remplacés : une vérification ✅ vient toujours de ton code. Pour écrire toi-même un module sauté : `python tools/start_chapter.py 13`.

## ⭐ Règles d'or

1. **Essaie vraiment avant de regarder une solution.** L'effort de récupération est ce qui fait apprendre.
2. **Règle des 15 minutes** : bloqué 15 min → indice 1 ; encore 15 min → indice 2, et ainsi de suite.
3. **Prédis avant d'exécuter** (🔮) : écris ton hypothèse, puis compare.
4. **Travaille uniquement dans `mon_travail/`** : les autres dossiers sont mis à jour par `git pull`.
5. **FAST_MODE d'abord** : tout tourne sur CPU en quelques minutes ; le mode complet est optionnel.
6. **Révise tes flashcards chaque jour**, même 10 minutes.
7. **Commite ton travail** régulièrement (`mon_travail/`) : ton dépôt GitHub est ton portfolio.

## 🧰 Outils

| Commande | Rôle |
|---|---|
| `python 00_setup/check_env.py` | vérifie ton installation |
| `python tools/start_chapter.py 18` | démarre un chapitre (`--init` : crée seulement `mylearn`) |
| `python -m pytest tests/` | teste ta librairie (`--impl=ref` : teste la référence) |
| `python tools/run_all_notebooks.py` | exécute les notebooks et mesure leur durée |
| `python tools/export_flashcards.py` | fusionne toutes les flashcards en un deck Anki |

## Versions

Python 3.13 et les versions préinstallées sur Colab (PyTorch 2.11, scikit-learn 1.6, NumPy 2.1, pandas 2.2…) : détail et justification dans [`docs/BIBLE.md` §21](docs/BIBLE.md).
