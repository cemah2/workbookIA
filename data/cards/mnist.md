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
0A (un tableau de forme (N, 28, 28), boucle Python contre NumPy, `show_images` : 0A.54, 0A.55, 0A.59), 0B (ordre de grandeur d'un produit matriciel, 0B.30), 1 (une image = 784 nombres, premier réseau de neurones en boîte noire), 2 (un point d'un espace à 784 dimensions, distances entre chiffres), 7 (contraste des distances face à des points uniformes, 7.20 ; un-contre-tous et un-contre-un sur 10 chiffres, 7.24), 10 (défi « Mark I » : le carré central de 20 × 20 pixels, 97 % de l'encre, et un perceptron sur les 3 contre les 5), 12 (variance expliquée des 1 et des 8, nombre de composantes de tout MNIST à prédire, chiffres propres et reconstructions jugées sur des images de test, cartes en 2D par PCA, t-SNE et UMAP), 13 (mémoire et temps d'un kNN sur MNIST complet, Naïve Bayes gaussien et le produit des probabilités qui donne `nan`, trois façons de calculer les distances et leurs temps, comparatif de ta librairie et de scikit-learn sur 500 images réduites par PCA, défi : 96 % sur les 2 000 premières images de test avec une PCA et un SVM RBF), 16 (formes d'une passe avant sur un mini-batch, un `MLPClassifier` relu par ta passe avant, histogrammes d'activations de réseaux tanh profonds, profils de 20 couches selon l'initialisation, défi : un réseau ReLU de 50 couches nourri de pixels bruts), 18 (neurones ReLU morts et valeurs sigmoïdes saturées pendant l'entraînement d'un MLP 784-100-100-10 sur 10 000 images, 18.24 ; défi : au moins 97 % sur les 10 000 images de test avec un MLP entraîné par ta propre rétropropagation NumPy, 18.29), 19 (les courbes de loss fabriquées de 19.23 imitent celles d'un MLP sur MNIST ; les solveurs de `MLPClassifier` sur 5 000 images, 19.24 ; neuf optimiseurs et plannings sur un MLP 784-128-10, validés sur les 10 000 dernières images d'entraînement, 19.27 ; défi : au moins 98 % sur le test en cinq epochs avec tes optimiseurs, 19.28), 17, 20 (des tableaux NumPy devenus tenseurs, 20.13 ; le mini-batch d'un `DataLoader`, 20.19 ; les formes d'une couche de convolution, 20.20 ; une première boucle d'entraînement PyTorch sur 12 000 images (50 000 en mode complet), validée sur les 10 000 dernières, 20.23 ; un script à cinq bugs, 20.24 ; la reprise d'un checkpoint, 20.27 ; un réseau de 30 couches avec et sans connexions résiduelles, 20.30), checkpoint IV (quatre paires de courbes de loss de MLP 784-256-10 entraînés sur 300 à 6 000 images et validés sur 2 000 autres, CP4.8), 25 (autoencodeurs, VAE), 27 (GAN), B5 (diffusion).
