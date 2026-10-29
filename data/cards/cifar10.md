# CIFAR-10

| | |
|---|---|
| **Stockage** | téléchargé (≈ 170 Mo) puis mis en cache (`cifar10.npz`), **jamais versionné** |
| **Chargement** | `wb.datasets.load_cifar10(split="train", n=None, normalize=False, channels_first=False, seed=0)` ; objet torchvision : `wb.datasets.torchvision_dataset("CIFAR10")` |
| **Sources** | Hugging Face (`uoft-cs/cifar10`, rapide) → torchvision (serveur de l'université de Toronto, parfois lent) |
| **Taille** | 50 000 images d'entraînement + 10 000 de test, 32 × 32 pixels, couleur (RGB), 6 000 images par classe |
| **Classes** | `wb.datasets.CIFAR10_CLASSES` : airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| **Licence** | pas de licence formelle ; les auteurs demandent de citer le rapport technique |
| **Vérifié le** | 2026-09-29 |

## Provenance
A. Krizhevsky (2009). *Learning Multiple Layers of Features from Tiny Images*. Rapport technique, Université de Toronto. CIFAR-10 est un sous-ensemble étiqueté de la base « 80 Million Tiny Images ».

## Format
`X` en `uint8` de forme (N, 32, 32, 3) (« canaux en dernier », comme matplotlib). PyTorch attend (N, 3, 32, 32) : utilise `channels_first=True` ou `X.transpose(0, 3, 1, 2)`.

## Biais et limites
- Très basse résolution : même un humain se trompe parfois (chat ou chien ?).
- La base mère « 80 Million Tiny Images » a été retirée par ses auteurs en 2020 (labels offensants découverts). CIFAR-10, sélectionné et vérifié à la main, reste utilisé, mais c'est un bon sujet de réflexion ⚖️ sur la provenance des données.
- Photos du web des années 2000 : biais culturels (types de voitures, d'animaux…).

## Chapitres
20 (les 200 premières photos de test, agrandies à 256 × 256 puis recadrées au centre en 224 × 224, données à MobileNetV3 pré-entraîné sur ImageNet, avec du bruit pur pour comparer les confiances : 🔮 20.26), 21 (des photos de test aux couleurs choisies pour un filtre 1 × 1, 🔮 21.13, et une photo de chat pour des filtres faits main, 21.19 ; stride ou pooling, entraînés sur 5 000 photos d'entraînement, 50 000 en mode complet, et testés sur 1 000 photos de test, 21.26 ; quatre photos de test agrandies à 224 × 224 pour les filtres et les cartes d'activation d'un ResNet-18 pré-entraîné sur ImageNet, 21.27), 24 (les 5 000 premières photos d'entraînement, 45 000 en mode complet, et les 5 000 dernières du fichier pour la validation, standardisées par canal avec 0,491, 0,482, 0,447 et 0,247, 0,243, 0,262 : un CNN traduit du Keras à réparer, 🐛 24.19 ; du CNN simple au CNN profond, 24.20 ; batchnorm et dropout, 🔬 24.21, avec un historique de 20 epochs sur 20 000 photos ; un CNN entraîné sur 10 000 photos testé sur 2 000 photos de test retournées, décalées, tournées ou en gris, 🔮 24.22 ; l'augmentation avec `torchvision.transforms.v2`, recadrages décalés et miroirs gauche-droite, 24.23 ; la précision mixte, 24.29 ; le défi frugal, moins de 100 000 paramètres et 80 % sur les 10 000 photos de test, 🏆 24.31), checkpoint V (le mini-projet MP5 : 45 000 photos d'entraînement et 5 000 de validation tirées par un split stratifié des 50 000, 10 000 photos de test ouvertes une fois en mode complet ; en mode rapide, au plus 12 000 photos d'entraînement et 2 000 de validation, sans le test ; MLP, petit CNN, VGG, petit ResNet, ResNet-18 pré-entraîné gelé, attaque FGSM), 28 (applications créatives), B1 (transfer learning), B6 (Grad-CAM), B7 (du notebook à la production : le classifieur du mini-projet de la partie V).
