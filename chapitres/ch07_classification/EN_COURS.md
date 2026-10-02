# ⏳ Chapitre 7 en cours de génération

Ce chapitre est généré en deux sessions (voir `suivi/PROGRESS.md`).

| Fichier | État |
|---|---|
| `01_fiche.md` + `figures/` | ✅ complet (tout le cours, §7.1 à §7.6.1, avec le centroïde le plus proche, l'algorithme de Lloyd, k-means++, la silhouette et le clustering par densité) |
| `02_exercices.md`, `06_mes_reponses.md` | ✅ complets (quiz, rappels, papier, réflexion, entretien) |
| `03_notebook.ipynb`, `05_solutions.ipynb` | 🛠️ partie 0 (vérification des quiz, des rappels et des exercices papier 7.1 à 7.6) ; parties A à D (7.11 à 7.31, `mylearn.cluster` et `mylearn.multiclass`) à la prochaine session |
| `04_indices.md`, `05_solutions.md` | 🛠️ quiz, rappels, exercices 7.1 à 7.10 et entretien ; 7.11 à 7.31 à la prochaine session |
| `flashcards.csv` | ✅ complet |
| `mylearn.cluster`, `mylearn.multiclass` | ✅ référence et tests prêts (`tests/test_ch07_cluster.py`, `tests/test_ch07_multiclass.py`) ; les exercices qui te les font écrire arrivent avec le notebook |

Tant que ce fichier existe, `python tools/start_chapter.py 7` copie dans `mon_travail/` ton fichier de réponses (`06_mes_reponses.md`) et tes squelettes `mylearn/cluster.py` et `mylearn/multiclass.py`, mais **pas encore le notebook** : une copie à moitié écrite ne serait jamais complétée, puisque tes fichiers ne sont jamais écrasés. Tu peux déjà lire la fiche, faire tous les exercices papier et même commencer les deux modules (leurs tests : `python -m pytest tests/test_ch07_cluster.py tests/test_ch07_multiclass.py -q`). Quand le chapitre sera terminé, ce fichier disparaîtra : relance alors la même commande pour obtenir le notebook complet, dont la partie 0 vérifie tes réponses courtes.
