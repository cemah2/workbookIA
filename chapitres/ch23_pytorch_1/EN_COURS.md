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

**Contraintes pour la session 49** (vérifications de la session 48) :
- 🔮 23.14 : la fiche dit déjà qu'oublier la conversion en float donne une erreur de type (§23.3.3, §23.5.5, pièges) et que la normalisation aide l'optimisation (§23.5.6) ; la prédiction doit porter sur un détail que la fiche ne donne pas (par exemple l'accuracy après une ou trois epochs avec des pixels de 0 à 255 non normalisés, au-dessus ou au-dessous d'un seuil, ou l'apparition de `nan`), saisie en booléen ou en entier.
- **Valeurs déjà imprimées, à ne pas faire vérifier** : les réponses des exercices papier (623 290 et le compte par couche, 211/210/240 mini-batches, 4 220 mises à jour, 0,408 et 0,797, 224,4 milliards d'opérations, 4,5 s, 2,5 et 7,5 Mo, 802 epochs, les epochs de 23.6), les variantes (101 770, 422 et 112, 0,240, 73,3) et les exemples de la fiche (2 020 paramètres pour 100 → 20, $6 \times 10^9$).
- Le `preprocess` de la fiche (§23.5.8) convertit $\mu$ et $\sigma$ en `float` Python et rend du `float32` : avec des scalaires NumPy `float64`, NumPy 2 rendrait du `float64` et `nn.Linear` refuserait. Le notebook (📦 23.15) doit garder ce piège en tête (un test : `dtype` du tenseur rendu).
- `EarlyStopping` de Keras 2.0 (2.0.6 à 2.0.8) attendait une epoch de plus que sa patience (corrigé en 2.1.0) : 📈 23.19 ou 🔬 23.27 ne doivent pas reprendre le décompte du livre tel quel.
- Data card `data/cards/fashion_mnist.md` : détailler l'usage du ch. 23 (découpage, tailles du mode rapide) une fois le notebook écrit.

