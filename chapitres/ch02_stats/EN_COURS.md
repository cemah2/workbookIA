# ⏳ Chapitre 2 en cours de génération

Ce chapitre est généré en deux sessions (voir `suivi/PROGRESS.md`).

| Fichier | État |
|---|---|
| `01_fiche.md` + `figures/` | ✅ complet (tout le cours, §2.1 à §2.9) |
| `02_exercices.md`, `06_mes_reponses.md` | ✅ complets (quiz, rappels, papier, démonstration, réflexion, entretien) |
| `03_notebook.ipynb`, `05_solutions.ipynb` | 🛠️ partie 0 (vérification des exercices papier 2.1 à 2.7) ; parties A à D (2.13 à 2.32, `mylearn.stats`) à la prochaine session |
| `04_indices.md`, `05_solutions.md` | 🛠️ quiz, rappels, exercices 2.1 à 2.12 et entretien ; 2.13 à 2.32 à la prochaine session |
| `flashcards.csv` | ✅ complet |
| `mylearn.stats` | ✅ référence et tests prêts (`tests/test_ch02_stats.py`) ; les exercices qui te la font écrire arrivent avec le notebook |

Tant que ce fichier existe, `python tools/start_chapter.py 2` copie dans `mon_travail/` ton fichier de réponses (`06_mes_reponses.md`) et ton squelette `mylearn/stats.py`, mais **pas encore le notebook** : une copie à moitié écrite ne serait jamais complétée, puisque tes fichiers ne sont jamais écrasés. Tu peux déjà lire la fiche, faire tous les exercices papier et même commencer `stats.py` (ses tests : `python -m pytest tests/test_ch02_stats.py -q`). Quand le chapitre sera terminé, ce fichier disparaîtra : relance alors la même commande pour obtenir le notebook complet, dont la partie 0 vérifie tes réponses papier.
