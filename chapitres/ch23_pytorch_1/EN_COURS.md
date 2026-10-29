# Chapitre 23 : génération en cours

Ce chapitre est généré en deux sessions. Tant que ce fichier existe, `python tools/start_chapter.py 23` copie la fiche et `06_mes_reponses.md`, mais pas encore le notebook.

| Fichier | État |
|---|---|
| `01_fiche.md` et `figures/` | complets (session 48) |
| `02_exercices.md`, `06_mes_reponses.md` | complets (session 48) |
| `04_indices.md`, `05_solutions.md` | quiz, rappels, papier, réflexion et entretien (session 48) ; notebook 23.12 à 23.29 à la session 49 |
| `flashcards.csv` | complet (session 48) |
| `03_notebook.ipynb`, `05_solutions.ipynb` | partie 0 (session 48) ; parties A à D (23.12 à 23.29) à la session 49 |

Pas de module `mylearn` dans ce chapitre (contrat) : le notebook réutilise `mylearn.metrics.confusion_matrix` (ch. 3) et `mylearn.nn.regularization.EarlyStopping` (ch. 20).

## Pour la session 49

À reprendre : `PARTS` et `NEXT_SESSION` de `tools/chapters/build_ch23.py` (les quatre parties annoncées), les sections « Notebook » de `04_indices.md` et `05_solutions.md`, puis supprimer ce fichier, lancer `syllabus.py build` et revérifier l'en-tête de la fiche.
