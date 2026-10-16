# ⏳ Chapitre 15 en cours de génération

Ce chapitre est généré en deux sessions (voir `suivi/PROGRESS.md`).

| Fichier | État |
|---|---|
| `01_fiche.md` + `figures/` | ✅ complet (tout le cours, §15.1 à §15.12, modernisé pour scikit-learn 1.6 : l'interface des estimateurs, `predict`, `decision_function` et `predict_proba` ; le clustering ; les transformateurs et `inverse_transform` ; la PCA ; les ensembles ; la validation croisée, les recherches en grille et au hasard, les pipelines, la frontière de décision ; les datasets et `train_test_split` ; au-delà du livre : `ColumnTransformer`, seuil, calibration et classes déséquilibrées, sauvegarde et versions, brancher ton propre estimateur) |
| `02_exercices.md`, `06_mes_reponses.md` | ✅ complets (quiz, rappels, papier, réflexion, entretien) ; la liste des exercices du notebook est déjà dans `02_exercices.md` |
| `03_notebook.ipynb`, `05_solutions.ipynb` | 🛠️ partie 0 (vérification des quiz, des rappels, des exercices papier 15.1 à 15.4 et de l'estimation 15.5) ; parties A à D (15.9 à 15.31) à la prochaine session |
| `04_indices.md`, `05_solutions.md` | 🛠️ quiz, rappels, exercices 15.1 à 15.8 et entretien ; 15.9 à 15.31 à la prochaine session |
| `flashcards.csv` | ✅ complet |
| `mylearn` | aucun module nouveau dans ce chapitre (le notebook compare ta librairie à scikit-learn) |

Tant que ce fichier existe, `python tools/start_chapter.py 15` copie dans `mon_travail/` ton fichier de réponses (`06_mes_reponses.md`), mais **pas encore le notebook** : une copie à moitié écrite ne serait jamais complétée, puisque tes fichiers ne sont jamais écrasés. Tu peux déjà lire toute la fiche et faire tous les exercices papier. Quand le chapitre sera terminé, ce fichier disparaîtra : relance alors la même commande pour obtenir le notebook complet, dont la partie 0 vérifie tes réponses courtes.
