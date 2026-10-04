# ⏳ Chapitre 12 en cours de génération

Ce chapitre est généré en deux sessions (voir `suivi/PROGRESS.md`).

| Fichier | État |
|---|---|
| `01_fiche.md` + `figures/` | ✅ complet (tout le cours, §12.1 à §12.10 : règle d'or, types de données et encodages, nettoyage et valeurs manquantes, mises à l'échelle, sélection de features, PCA avec ses encadrés de maths, transformations inverses, découpes, fuites en validation croisée ; au-delà du livre : t-SNE et UMAP, pandas et SQL) |
| `02_exercices.md`, `06_mes_reponses.md` | ✅ complets (quiz, rappels, papier, réflexion, entretien) ; la liste des exercices du notebook est déjà dans `02_exercices.md` |
| `03_notebook.ipynb`, `05_solutions.ipynb` | 🛠️ partie 0 (vérification des quiz, des rappels et des exercices papier 12.1 à 12.5 et 12.7) ; parties A à D (12.11 à 12.33, `mylearn.preprocessing`) à la prochaine session |
| `04_indices.md`, `05_solutions.md` | 🛠️ quiz, rappels, exercices 12.1 à 12.10 et entretien ; 12.11 à 12.33 à la prochaine session |
| `flashcards.csv` | ✅ complet |
| `mylearn.preprocessing` | ✅ référence et tests prêts (`tests/test_ch12_preprocessing.py`) ; les exercices qui te le font écrire arrivent avec le notebook |

Tant que ce fichier existe, `python tools/start_chapter.py 12` copie dans `mon_travail/` ton fichier de réponses (`06_mes_reponses.md`) et ton squelette `mylearn/preprocessing.py`, mais **pas encore le notebook** : une copie à moitié écrite ne serait jamais complétée, puisque tes fichiers ne sont jamais écrasés. Tu peux déjà lire toute la fiche, faire tous les exercices papier et même commencer le module (ses tests : `python -m pytest tests/test_ch12_preprocessing.py -q`). Quand le chapitre sera terminé, ce fichier disparaîtra : relance alors la même commande pour obtenir le notebook complet, dont la partie 0 vérifie tes réponses courtes.
