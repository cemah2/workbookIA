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
21 (CNN), 24 (augmentation de données), 28 (applications créatives), B1 (transfer learning), B6 (Grad-CAM), B7 (du notebook à la production : le classifieur du mini-projet de la partie V).
