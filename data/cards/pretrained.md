# Réseaux pré-entraînés du ch. 28 (poids torchvision)

| | |
|---|---|
| **Chargement** | `net, mean, std = wb.datasets.load_pretrained("squeezenet1_1" \| "vgg16")` : la partie convolutive (`model.features`), gelée (`requires_grad=False`), en mode `eval()`, ReLU sans `inplace` ; `mean` et `std` d'ImageNet en tenseurs `(1, 3, 1, 1)` |
| **Poids** | `SqueezeNet1_1_Weights.IMAGENET1K_V1` (1,2 million de paramètres, 4,7 Mo, 0,35 GFLOPS) ; `VGG16_Weights.IMAGENET1K_V1` (138 millions de paramètres, 528 Mio, 15,5 GFLOPS ; mode complet seulement) |
| **Stockage** | cache de torch (`~/.cache/torch/hub/checkpoints`), téléchargés une fois ; jamais versionnés |
| **Licence** | poids publiés avec torchvision (BSD 3-Clause pour le code) ; ImageNet, sur lequel ils ont été entraînés, a ses propres conditions (recherche et enseignement, non commercial) |

## Provenance
F. Iandola et coll., « SqueezeNet » (2016), version 1.1 ; K. Simonyan et A. Zisserman, « Very Deep Convolutional Networks » (2014). Poids et prétraitement : documentation de torchvision 0.26 (*Models and pre-trained weights*).

## Utilisation
Ch. 28 : extraction d'activations par hooks, visualisation de filtres, deep dream, transfert de style. En mode rapide, SqueezeNet sur des images de 128 à 192 pixels : ≈ 30 ms par pas d'Adam sur deux cœurs.

## Biais et limites
- Un réseau entraîné sur ImageNet-1k « rêve » de ce qu'il a vu : beaucoup d'animaux (118 classes de chiens), des objets occidentaux du quotidien.
- Les modules `Fire` de SqueezeNet concatènent deux branches (`expand1x1`, `expand3x3`) : les canaux de leur sortie ne sont pas tous du même genre.
