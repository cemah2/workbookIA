# Réseaux pré-entraînés (poids torchvision)

| | |
|---|---|
| **Chargement** | ch. 28 : `net, mean, std = wb.datasets.load_pretrained("squeezenet1_1" \| "vgg16")` : la partie convolutive (`model.features`), gelée (`requires_grad=False`), en mode `eval()`, ReLU sans `inplace` ; `mean` et `std` d'ImageNet en tenseurs `(1, 3, 1, 1)`. Ch. 20, 21, 23 et MP5 : directement par `torchvision.models` (`mobilenet_v3_small(weights=...)`, `resnet18(weights=...)`), avec le prétraitement des poids, `weights.transforms()` (agrandir à 256, recadrer au centre en 224 × 224, normaliser avec les moyennes et écarts-types d'ImageNet) |
| **Poids** | `MobileNet_V3_Small_Weights.IMAGENET1K_V1` (2,5 millions de paramètres, 9,8 Mio soit 10,3 Mo, 0,06 GFLOPS, 67,7 % top-1) ; `ResNet18_Weights.IMAGENET1K_V1` (11,7 millions de paramètres, 44,7 Mio soit 46,8 Mo, fichier `resnet18-f37072fd.pth`, 1,8 GFLOPS, 69,8 % top-1) ; `SqueezeNet1_1_Weights.IMAGENET1K_V1` (1,2 million de paramètres, 4,7 Mio soit 5,0 Mo, 0,35 GFLOPS) ; `VGG16_Weights.IMAGENET1K_V1` (138 millions de paramètres, 528 Mio soit 553 Mo, 15,5 GFLOPS ; mode complet seulement). `DEFAULT` désigne ces poids dans torchvision 0.26 |
| **Stockage** | cache de torch (`~/.cache/torch/hub/checkpoints`), téléchargés une fois ; jamais versionnés |
| **Licence** | poids publiés avec torchvision (BSD 3-Clause pour le code) ; ImageNet, sur lequel ils ont été entraînés, a ses propres conditions (recherche et enseignement, non commercial) : à écrire dans la fiche modèle d'un modèle qui les réutilise (🛠️ 23.28) |

## Provenance
A. Howard et coll., « Searching for MobileNetV3 » (2019) ; K. He et coll., « Deep Residual Learning for Image Recognition » (2015) ; F. Iandola et coll., « SqueezeNet » (2016), version 1.1 ; K. Simonyan et A. Zisserman, « Very Deep Convolutional Networks » (2014). Poids, nombres de paramètres, tailles et prétraitement : [documentation de torchvision 0.26](https://docs.pytorch.org/vision/0.26/models.html) (*Models and pre-trained weights*) et `weights.meta`.

## Utilisation
- Ch. 20 : MobileNetV3 Small face à des photos de CIFAR-10 et à du bruit pur (🔮 20.26).
- Ch. 21 : les filtres et les cartes d'activation d'un ResNet-18 (📦 21.27).
- Ch. 23 : un ResNet-18 gelé, extracteur de features pour Fashion-MNIST, avec une nouvelle tête (§23.9.4, 📦 23.24), et sa fiche modèle (🛠️ 23.28).
- Checkpoint V : le ResNet-18 pré-entraîné gelé du mini-projet MP5.
- Ch. 28 : extraction d'activations par hooks, visualisation de filtres, deep dream, transfert de style (SqueezeNet, VGG16 en mode complet). En mode rapide, SqueezeNet sur des images de 128 à 192 pixels : ≈ 30 ms par pas d'Adam sur deux cœurs.

## Biais et limites
- Un réseau entraîné sur ImageNet-1k « rêve » de ce qu'il a vu : beaucoup d'animaux (118 classes de chiens), des objets occidentaux du quotidien.
- Ses 1 000 classes ne contiennent pas « rien » : il donne une classe à toute image, même hors de sa distribution.
- Les modules `Fire` de SqueezeNet concatènent deux branches (`expand1x1`, `expand3x3`) : les canaux de leur sortie ne sont pas tous du même genre.
