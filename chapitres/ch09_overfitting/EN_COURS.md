# ⏳ Chapitre 9 en cours de génération

Ce chapitre est généré en deux sessions (voir `suivi/PROGRESS.md`).

| Fichier | État |
|---|---|
| `01_fiche.md` + `figures/` | ✅ complet (tout le cours, §9.1 à §9.7, avec les mesures d'erreur, les moindres carrés, l'early stopping avec patience, Ridge et le Lasso, la décomposition biais-variance, la double descente et la droite bayésienne) |
| `02_exercices.md`, `06_mes_reponses.md` | ✅ complets (quiz, rappels, papier, réflexion, entretien) ; la liste des exercices du notebook est déjà dans `02_exercices.md` |
| `03_notebook.ipynb`, `05_solutions.ipynb` | 🛠️ partie 0 (vérification des quiz, des rappels et des exercices papier 9.1 à 9.7 et 9.9) ; parties A à D (9.12 à 9.31, `mylearn.linear`) à la prochaine session |
| `04_indices.md`, `05_solutions.md` | 🛠️ quiz, rappels, exercices 9.1 à 9.11 et entretien ; 9.12 à 9.31 à la prochaine session |
| `flashcards.csv` | ✅ complet |
| `mylearn.linear` | ✅ référence et tests prêts (`tests/test_ch09_linear.py`) ; les exercices qui te le font écrire arrivent avec le notebook |

Tant que ce fichier existe, `python tools/start_chapter.py 9` copie dans `mon_travail/` ton fichier de réponses (`06_mes_reponses.md`) et ton squelette `mylearn/linear.py`, mais **pas encore le notebook** : une copie à moitié écrite ne serait jamais complétée, puisque tes fichiers ne sont jamais écrasés. Tu peux déjà lire toute la fiche, faire tous les exercices papier et même commencer le module (ses tests : `python -m pytest tests/test_ch09_linear.py -q`) : les prédictions que demanderont les 🔮 du notebook portent sur des détails que la fiche ne donne pas (des valeurs pour 9.13, les coefficients pris un à un pour 9.19). Quand le chapitre sera terminé, ce fichier disparaîtra : relance alors la même commande pour obtenir le notebook complet, dont la partie 0 vérifie tes réponses courtes.
