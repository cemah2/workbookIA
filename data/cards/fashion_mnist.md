# Fashion-MNIST

| | |
|---|---|
| **Stockage** | téléchargé puis mis en cache (`fashion_mnist.npz`, ≈ 30 Mo), **jamais versionné** |
| **Chargement** | `wb.datasets.load_fashion_mnist(split="train", n=None, flatten=False, normalize=False, seed=0)` |
| **Sources** | torchvision → dépôt GitHub `zalandoresearch/fashion-mnist` → OpenML |
| **Taille** | 60 000 + 10 000 images 28 × 28 en niveaux de gris (même format que MNIST) |
| **Classes** | `wb.datasets.FASHION_CLASSES` : T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot |
| **Licence** | MIT |
| **Vérifié le** | 2026-09-29 (téléchargement testé via GitHub ; l'URL torchvision est en HTTP simple, bloquée dans l'environnement de génération mais accessible sur Colab) |

## Provenance
H. Xiao, K. Rasul, R. Vollgraf (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747. Photos d'articles du catalogue Zalando, réduites en 28 × 28.

## Biais et limites
- Plus difficile que MNIST (≈ 90-94 % pour un bon CNN simple) : certaines classes se ressemblent beaucoup (Shirt, T-shirt, Pullover, Coat).
- Images de catalogue sur fond neutre : très loin de photos prises dans la rue.

## Chapitres
20 (premiers pas PyTorch), 21 (CNN), 23 (PyTorch en pratique), 25 (autoencodeurs), 27 (GAN), B5 (diffusion).
