# ⏳ Chapitre 14 en cours de génération

Ce chapitre est généré en deux sessions (voir `suivi/PROGRESS.md`).

| Fichier | État |
|---|---|
| `01_fiche.md` + `figures/` | ✅ complet (tout le cours, §14.1 à §14.7 : pourquoi un ensemble bat ses membres et quand il échoue ; le vote à la pluralité, pondéré, dur et souple, le voting et le stacking ; le bootstrap, les exemples out-of-bag et le bagging ; les forêts aléatoires et l'importance des features ; ExtraTrees ; le boosting et AdaBoost (SAMME) ; le gradient boosting, ajouté par le workbook, avec HistGradientBoosting, XGBoost et LightGBM) |
| `02_exercices.md`, `06_mes_reponses.md` | ✅ complets (quiz, rappels, papier, réflexion, entretien) ; la liste des exercices du notebook est déjà dans `02_exercices.md` |
| `03_notebook.ipynb`, `05_solutions.ipynb` | 🛠️ partie 0 (vérification des quiz, des rappels et des exercices papier 14.1 à 14.3 et 14.5 à 14.7) ; parties A à D (14.11 à 14.28, ta librairie) à la prochaine session |
| `04_indices.md`, `05_solutions.md` | 🛠️ quiz, rappels, exercices 14.1 à 14.10 et entretien ; 14.11 à 14.28 à la prochaine session |
| `flashcards.csv` | ✅ complet |
| `mylearn` : `ensemble` | ✅ référence et tests prêts (`tests/test_ch14_ensemble.py`) ; les exercices qui te le font écrire arrivent avec le notebook |

Tant que ce fichier existe, `python tools/start_chapter.py 14` copie dans `mon_travail/` ton fichier de réponses (`06_mes_reponses.md`) et ton squelette `mylearn/ensemble.py`, mais **pas encore le notebook** : une copie à moitié écrite ne serait jamais complétée, puisque tes fichiers ne sont jamais écrasés. Tu peux déjà lire toute la fiche, faire tous les exercices papier et même commencer le module (ses tests : `python -m pytest tests/test_ch14_ensemble.py -q`). Tes ensembles s'appuient sur **ton** `tree.py` (ch. 13) et **ton** `model_selection.py` (ch. 8, `clone`) : si tu ne les as pas encore, la référence les remplace. Quand le chapitre sera terminé, ce fichier disparaîtra : relance alors la même commande pour obtenir le notebook complet, dont la partie 0 vérifie tes réponses courtes.
