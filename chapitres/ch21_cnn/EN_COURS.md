# Chapitre 21 : génération en cours

Ce chapitre est généré en deux sessions. Tant que ce fichier existe, `python tools/start_chapter.py 21` copie la fiche, `06_mes_reponses.md` et le squelette `mylearn/conv.py`, mais pas encore le notebook.

| Fichier | État |
|---|---|
| `01_fiche.md` et `figures/` | complets (session 44) |
| `02_exercices.md`, `06_mes_reponses.md` | complets (session 44) |
| `04_indices.md`, `05_solutions.md` | quiz, rappels, papier, réflexion et entretien (session 44) ; notebook 21.12 à 21.32 à la session 45 |
| `flashcards.csv` | complet (session 44) |
| `03_notebook.ipynb`, `05_solutions.ipynb` | partie 0 (session 44) ; parties A à D (21.12 à 21.32) à la session 45 |
| `mylearn.conv` | référence et tests complets (session 44) |

## Pour la session 45 : précisions de contrat à écrire dans les énoncés des 🔨

Les tests de `tests/test_ch21_conv.py` (210 tests, groupes `-k` dans son en-tête) exigent, au-delà des docstrings :

1. Les entrées ne sont jamais modifiées et peuvent être des listes d'entiers ; les tableaux rendus sont toujours **neufs** et en float64 (même `pad2d(x, 0)`, `im2col` avec k = 1, `max_pool2d` avec k = 1).
2. `conv_output_size` rend un `int` Python (pas `np.int64`, pas un float) et refuse `size = 0`, même avec un padding.
3. Un argument « int ou tuple de 2 ints » : le 1er nombre vaut pour H, le 2e pour W.
4. Noyau, stride ou dilatation < 1, padding négatif, fenêtre qui ne tient pas : `ValueError` dans `im2col`, `conv2d` et les poolings (appeler `conv_output_size` suffit) ; stride 0 refusé aussi par `conv_transpose2d`.
5. `conv2d` refuse : `groups = 0` ou qui ne divise pas C_in ou C_out ; un poids 3-D ou de `shape[1]` ≠ C_in // groups ; un biais de forme ≠ (C_out,), même diffusable ; une chaîne autre que `"same"` et `"valid"` ; `"same"` avec un stride > 1 sur un seul axe.
6. `"same"` avec une étendue paire, d(k − 1) impair : le zéro en trop en bas et à droite, dilatation comprise.
7. Poolings : padding refusé au-dessus de `kernel_size // 2` sur chaque axe (p = k // 2 accepté) ; avec des égalités, seules les valeurs comptent.
8. `conv2d_backward` : une cellule qu'aucune fenêtre ne voit reçoit un gradient nul ; un `dout` d'une autre forme que la sortie est refusé, même avec le même nombre de valeurs.
9. `conv_transpose2d` : padding hors de [0, k − 1] refusé sur chaque axe ; biais non (C_out,) et sortie vide refusés.
10. Piège à mettre dans les indices ou les erreurs fréquentes de 21.30 et 21.31 : calculer dx par `conv_transpose2d(dout, W, padding=p)` est faux quand le stride laisse des cellules non vues (il manque des lignes en bas : il faut compléter la sortie avant de retirer le padding). La propriété « transposée de dout = dx » ne vaut que si (H + 2p − k) est divisible par s.
