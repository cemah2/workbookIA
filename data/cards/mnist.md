# MNIST

| | |
|---|---|
| **Fichier** | `data/mnist.npz` (compressé, 11,5 Mo) : clés `x_train`, `y_train`, `x_test`, `y_test` |
| **Chargement** | `wb.datasets.load_mnist(split="train", n=None, flatten=False, normalize=False, seed=0)` |
| **Secours** | torchvision, puis OpenML (`mnist_784`) |
| **Taille** | 60 000 images d'entraînement + 10 000 de test, 28 × 28 pixels, niveaux de gris `uint8` (0-255) |
| **Tâche type** | classification d'images en 10 classes (chiffres 0 à 9) |
| **Licence** | Creative Commons Attribution-Share Alike 3.0 ; copyright Yann LeCun et Corinna Cortes (œuvre dérivée des bases du NIST) |
| **Construit le** | 2026-09-29 depuis torchvision (`tools/build_datasets.py`) |

## Provenance
Y. LeCun, C. Cortes, C. J. C. Burges, *The MNIST database of handwritten digits* (1998). Chiffres écrits par des employés du recensement américain et des lycéens, recentrés et normalisés en 28 × 28.

## Répartition des classes
Entraînement : de 5 421 (chiffre 5) à 6 742 (chiffre 1) exemples par classe. Test : de 892 à 1 135. Légèrement déséquilibré.

## Options du loader
- `n=5000` : sous-échantillon aléatoire reproductible (`seed`), pratique en FAST_MODE ;
- `flatten=True` : vecteurs de 784 valeurs (pour les modèles non convolutifs) ;
- `normalize=True` : `float32` entre 0 et 1.

## Biais et limites
- Dataset « résolu » : les meilleurs modèles dépassent 99,7 % d'accuracy. Idéal pour apprendre, trompeur pour juger une méthode (un modèle médiocre y paraît bon). Fashion-MNIST est plus exigeant.
- Écritures d'une population particulière (États-Unis, années 1990) : un modèle entraîné ici généralise mal à d'autres styles d'écriture.

## Chapitres
12 (PCA, UMAP), 13 (k-NN, arbres), 16-20 (réseaux, rétropropagation, optimiseurs, PyTorch), 25 (autoencodeurs, VAE), 27 (GAN), B5 (diffusion).
