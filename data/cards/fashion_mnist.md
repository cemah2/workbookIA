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
20 (early stopping d'un MLP 784-256-256-10 avec batchnorm et dropout, validé sur les 10 000 dernières images d'entraînement, 20.28 ; ablation dropout, weight decay et batchnorm sur 2 000 images, 20.29 ; défi : au moins 90 % sur le test avec un MLP de 300 000 paramètres au plus, 20.31), checkpoint IV (une epoch d'AdamW écrite en NumPy sur 2 048 images d'entraînement, CP4.9 ; mini-projet MP4 : un MLP 784-256-128-10 de minifw, ton framework NumPy, contre son jumeau PyTorch, appris sur les 50 000 premières images d'entraînement, validé sur les 10 000 dernières, le test ouvert une seule fois), 21 (le convnet du livre traduit en PyTorch, entraîné sur 10 000 images d'entraînement et validé sur 2 000 autres, tirées par une permutation de graine 2122 : 21.22, puis 21.24, 21.25, 21.28 et 21.29 ; deux historiques de 15 epochs, avec et sans dropout, appris sur 50 000 images et validés sur les 10 000 autres, précalculés par `tools/chapters/history_ch21.py` : 📈 21.23 ; défi : au moins 92 % sur le test avec un CNN de moins de 100 000 paramètres, appris sur les 60 000 images d'entraînement, 21.32), 23 (PyTorch en pratique 1 : chargée dans l'ordre des fichiers, `seed=None` ; découpage stratifié 54 000 + 6 000, `test_size=0.1`, `random_state=0`, moyenne 0,2858 et écart-type 0,3529 des pixels divisés par 255 ; 12 000 images d'entraînement en mode rapide pour les exercices qui entraînent longtemps, toutes sinon ; le MLP du livre, 623 290 paramètres ; un historique de 50 epochs précalculé par `tools/chapters/history_ch23.py` pour 📈 23.19 et 🔬 23.27 ; des images de validation décalées, inversées ou en miroir pour 🔮 23.21 ; 2 000 images agrandies pour les features d'un ResNet-18 pré-entraîné, 23.24 ; défi 🏆 23.29 : cinq graines au moins à 88 % sur le test, révélé une fois par session), 25 (autoencodeurs et VAE : le même ordre fixe que MNIST, permutation de graine 2500 ; 2 000 images de test, graine 0, dont 1 000 de validation et 1 000 de test caché pour les débruiteurs ; une botte et un T-shirt pour l'autoencodeur d'une seule image, 🔮 25.14 ; les vêtements donnés à un autoencodeur de chiffres, 🔮 25.17 ; débruiteurs à bruit gaussien σ = 0,3 écrêté, 25.26, 25.27 et le défi 🏆 25.39 : au moins le PSNR de la référence moins 0,5 dB, en 300 s d'entraînement en mode rapide, avec au plus 100 000 paramètres ; β-VAE, 25.36), 27 (défi 🏆 27.28 : un DCGAN de vêtements, 10 000 images d'entraînement en mode rapide dans l'ordre d'une permutation de graine 2800, ramenées dans $[-1, 1]$ ; 4 000 images de test tirées par une permutation de graine 2801, 2 000 publiques et 2 000 cachées, révélées une fois par session ; les features d'un petit CNN de vêtements, ≈ 84 % d'accuracy après 3 epochs ; objectif : la distance de Fréchet de la référence, la pire de trois graines, plus 10 %, en 150 s d'entraînement), B5 (diffusion).
