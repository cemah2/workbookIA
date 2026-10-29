# Chapitre 24 : génération en cours

Ce chapitre est généré en deux sessions. Tant que ce fichier existe, `python tools/start_chapter.py 24` copie la fiche et `06_mes_reponses.md`, mais pas encore le notebook.

| Fichier | État |
|---|---|
| `01_fiche.md` et `figures/` | complets (session 50) |
| `02_exercices.md`, `06_mes_reponses.md` | complets (session 50) |
| `04_indices.md`, `05_solutions.md` | quiz, rappels, papier, réflexion et entretien (session 50) ; notebook 24.11 à 24.31 à la session 51 |
| `flashcards.csv` | complet (session 50) |
| `03_notebook.ipynb`, `05_solutions.ipynb` | partie 0 (session 50) ; parties A à E (24.11 à 24.31) à la session 51 |

Pas de module `mylearn` dans ce chapitre (contrat) : le notebook réutilise `mylearn.sequences` (ch. 22 : `make_windows`, `make_sequence_pairs`, `build_vocab`, `encode`, `decode`, `sample_from_logits`) et `mylearn.preprocessing.MinMaxScaler` (ch. 12).

## Pour la session 51

À reprendre : `PARTS` et `NEXT_SESSION` de `tools/chapters/build_ch24.py` (les cinq parties annoncées : A 24.11-24.15, B 24.16-24.18, C 24.19-24.24, D 24.25-24.27, E 24.28-24.31), les sections « Notebook » de `04_indices.md` et `05_solutions.md`, la data card CIFAR-10 (usage du ch. 24), puis supprimer ce fichier, lancer `syllabus.py build` et revérifier l'en-tête de la fiche (temps total, parcours rapide).

**Contraintes pour la session 51** :
- Budget : solutions < 10 min sur CPU, aucune cellule > 3 min ; un seul entraînement PyTorch à la fois (vérificateurs compris). Le contrat prévoit 6 000 images Fashion-MNIST, une grille de 4 candidats × 3 folds × 3 epochs, CIFAR-10 réduit (5 000 images, 2 epochs), des sinus de 200 à 2 000 points (20 000 en mode complet), 150 000 caractères de Holmes. Les historiques longs (CNN CIFAR-10 complets, générateur de mots) se précalculent par un script `tools/chapters/history_ch24.py`, comme aux ch. 21 à 23.
- 🔮 24.12 : la fiche (§24.2.4, §24.2.5) résume déjà les résultats du livre sur MNIST (784 + 784 sur-apprend plus vite ; 64 un peu moins bon ; 32 + 32 encore un peu moins). Le 🔮 doit faire prédire ce qu'elle ne dit pas, sur Fashion-MNIST, avec des modèles entraînés par une cellule donnée (graines fixes) : par exemple l'epoch du minimum de la loss de validation de chaque modèle, ou si 32 + 32 bat 64, saisis en entiers ou en booléens, et refaits sur plusieurs graines avant d'être enregistrés.
- 📈 24.14 : la fiche ne donne pas l'explication de la validation au-dessus du train avec dropout (elle renvoie au 📈) ; le quiz Q3 et R1 c) touchent la moyenne pendant l'epoch, sans faire le lien.
- 🔮 24.22 : la fiche donne la règle de plausibilité et l'exemple du « 2 » en miroir ; le 🔮 doit porter sur des mesures (par exemple l'accuracy d'un modèle entraîné, sur des images de test transformées), pas sur la règle.
- 📦 24.16 : la fiche donne le squelette de `TorchMLPClassifier` (`__init__` seulement) ; `fit` et `predict` restent à écrire. Ordre des classes mères : `ClassifierMixin` avant `BaseEstimator` (sinon `is_classifier` est faux sans message, sklearn 1.6.1). Le contrat doit fixer ce que vérifie la cellule (`clone`, `get_params`, `set_params`, `classes_`, labels non entiers, `predict` sur des labels d'origine).
- 📦 24.17 : `MinMaxScaler` par pixel est un mauvais choix pour des images (fiche §24.3.3) ; la cellule peut le montrer ou utiliser une mise à l'échelle globale (`FunctionTransformer`).
- 📦 24.25 et 24.26 : les sinus du livre (calme : freqs (1, 2), amps (1, 2) ; agitée : freqs (1,1 ; 1,7 ; 3,1 ; 7), amps (1, 2, 2, 3) ; Δθ = 0,057 ; test à partir du point 400) ; la série « toujours montante » du livre change le signe d'un point quand la somme baisse (listing 24.66). Comparer chaque RMSE à la persistance.
- 📦 24.27 (taches solaires, `wb.check`) : vérifier les formes d'`output`, `h_n`, `c_n` et l'égalité `output[:, -1] == h_n[-1]` ; l'état porté avec `detach()` sur des flux parallèles (la ligne i continue la ligne i).
- 📦 24.29 : mesurer sans jamais enregistrer un temps comme réponse (verdict à seuil ou pas de vérification) ; `torch.compile` facultatif (sous Windows, MSVC requis ; sur Linux, un compilateur C++) ; `bfloat16` sur CPU, `float16` + `GradScaler` sur GPU.
- 🏆 24.31 : contrat « ≥ 80 % de test accuracy avec moins de 100 000 paramètres » en mode complet (GPU Colab) ; en mode rapide, objectif affiché ≈ 60 % sur le sous-ensemble ; à calibrer sur plusieurs graines ; test révélé une fois par architecture et par session.
- **Valeurs déjà imprimées, à ne pas faire vérifier** : les réponses des exercices papier (114 502 paramètres et le compte par couche, 784, 588 000 / 846 720 / 109 184 / 358 680 MAC, 3,30, les formes de 24.4 et ses 304 paramètres, 187 / 19 987 / 625 fenêtres et pas, 8,3 min, 1 008 s, les durées de 24.7) ; les variantes (16 198 paramètres, 299 tirages, 2 304 000 et 576 000 MAC, 228 paramètres de GRU, 1 995 et 63, 440 min) ; les exemples de la fiche (224 paramètres et 221 184 MAC de `Conv2d(3, 8, 3)`, $4^5 = 1\,024$, $1 - 0{,}9^{10} \approx 0{,}65$) ; les moyennes et statistiques de CIFAR-10 de la fiche (0,491, 0,482, 0,447 ; 0,247, 0,243, 0,262).
- Les chiffres du livre cités par la fiche (98,62 %, 98,79 %, 99,39 %, 99,12 %, 99,11 % ; 66, 127, 147, 36, 45 s par epoch) ne sont pas à faire reproduire : MNIST n'est pas le fil rouge des CNN du notebook (CIFAR-10).
