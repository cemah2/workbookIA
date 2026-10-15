# ⏳ Chapitre 13 en cours de génération

Ce chapitre est généré en deux sessions (voir `suivi/PROGRESS.md`).

| Fichier | État |
|---|---|
| `01_fiche.md` + `figures/` | ✅ complet (tout le cours, §13.1 à §13.7 : paramétrique ou non paramétrique ; la régression logistique, ajoutée par le workbook ; les k plus proches voisins ; les SVM, la marge, C et l'astuce du noyau ; les arbres de décision, leurs splits, Gini, l'entropie et l'élagage ; Naïve Bayes gaussien et multinomial, le lissage et logsumexp ; le choix d'un classifieur) |
| `02_exercices.md`, `06_mes_reponses.md` | ✅ complets (quiz, rappels, papier, réflexion, entretien) ; la liste des exercices du notebook est déjà dans `02_exercices.md` |
| `03_notebook.ipynb`, `05_solutions.ipynb` | 🛠️ partie 0 (vérification des quiz, des rappels et des exercices papier 13.1 à 13.7 et 13.9) ; parties A à F (13.12 à 13.36, ta librairie) à la prochaine session |
| `04_indices.md`, `05_solutions.md` | 🛠️ quiz, rappels, exercices 13.1 à 13.11 et entretien ; 13.12 à 13.36 à la prochaine session |
| `flashcards.csv` | ✅ complet |
| `mylearn` : `neighbors`, `tree`, `naive_bayes`, `logistic` | ✅ références et tests prêts (`tests/test_ch13_*.py`) ; les exercices qui te les font écrire arrivent avec le notebook |

Tant que ce fichier existe, `python tools/start_chapter.py 13` copie dans `mon_travail/` ton fichier de réponses (`06_mes_reponses.md`) et tes squelettes `mylearn/neighbors.py`, `tree.py`, `naive_bayes.py` et `logistic.py`, mais **pas encore le notebook** : une copie à moitié écrite ne serait jamais complétée, puisque tes fichiers ne sont jamais écrasés. Tu peux déjà lire toute la fiche, faire tous les exercices papier et même commencer les modules (leurs tests : `python -m pytest tests/test_ch13_neighbors.py -q`, et de même pour `tree`, `naive_bayes` et `logistic`). Quand le chapitre sera terminé, ce fichier disparaîtra : relance alors la même commande pour obtenir le notebook complet, dont la partie 0 vérifie tes réponses courtes.
