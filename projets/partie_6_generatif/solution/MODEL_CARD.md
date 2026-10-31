# Fiche modèle · Générateurs de vêtements Fashion-MNIST (`genbench`)

## Les modèles

- **VAE convolutif** (`ConvVAE`, 220 385 paramètres) : un encodeur à deux convolutions de pas 2 (32 puis 64 canaux) et deux têtes vers 16 latents (moyenne et log-variance), un décodeur symétrique à deux convolutions transposées qui rend des logits ; loss = cross-entropy binaire sommée sur les 784 pixels + β · KL, avec **β = 1** (choisi parmi 0,5, 1 et 4 par une règle écrite avant la comparaison).
- **DCGAN** (`Generator` de 385 057 paramètres, bruit de 64 valeurs ; `Discriminator` de 79 441 paramètres ; largeur 48) selon les règles de Radford : convolutions transposées de pas 2, batchnorm et ReLU, tanh en sortie ; convolutions de pas 2 et Leaky ReLU (0,2) dans le discriminateur ; initialisation $\mathcal{N}(0 ;\, 0{,}02^2)$ ; Adam (2·10⁻⁴, β₁ = 0,5), loss non saturante. Un **GAN conditionnel** de même taille (le label entre par un embedding) est fourni en option.
- **Entraînement** (le 2026-10-31, mode complet) : les 50 000 images d'entraînement, mini-batches de 128 (VAE) et 64 (GAN), 15 epochs pour le VAE, 10 pour le GAN ; trois graines (0, 1, 2) pour chaque générateur ; Python 3.13, PyTorch 2.11 (CPU). La configuration complète est dans `config.json`.
- **L'instrument de mesure** : un classifieur CNN de 218 586 paramètres, entraîné sur les mêmes images, **92,5 %** d'accuracy de validation ; ses 128 features servent à la distance de Fréchet.

Les poids ne sont pas livrés (les checkpoints du DCGAN, `runs/`, se régénèrent avec le notebook) ; `genbench.sample_images(model, n, seed)` produit des images à partir d'un modèle entraîné.

## Usage prévu

Un banc d'essai de portfolio et d'enseignement : comparer deux familles de générateurs avec un protocole d'évaluation fixé à l'avance, sur des vignettes de vêtements 28 × 28. **Usages exclus** : présenter ces images comme des photos de produits réels ; s'en servir pour entraîner un autre modèle sans dire qu'elles sont générées ; toute application qui a besoin d'images à une autre résolution ou d'une autre nature (photos portées, personnes). Les modèles ne savent produire que ce que montrent les données : des articles de catalogue centrés sur fond noir.

## Les données

Fashion-MNIST (H. Xiao, K. Rasul et R. Vollgraf, Zalando Research, 2017, [arXiv:1708.07747](https://arxiv.org/abs/1708.07747) ; licence MIT) : 70 000 photos d'articles du catalogue Zalando réduites en 28 × 28 niveaux de gris, dix classes (T-shirt, pantalon, pull, robe, manteau, sandale, chemise, basket, sac, bottine), environ 6 000 images par classe à l'entraînement. Le projet coupe les 60 000 images d'entraînement en 50 000 pour l'entraînement et 10 000 pour la validation (permutation de graine 2026) ; le test n'est pas utilisé. Les images ne contiennent aucune personne ; elles sont centrées, sur fond neutre, très loin de photos prises dans la rue. Voir aussi la data card du workbook, `data/cards/fashion_mnist.md`.

## Les métriques

Mesurées sur 10 000 images générées par modèle et par graine, contre les 10 000 images de validation :

| Modèle | Distance de Fréchet (moyenne ± écart-type, 3 graines) | Entropie des classes | Quasi-copies |
|---|---|---|---|
| Deux échantillons d'images réelles (plancher) | 1,1 | 0,999 | — |
| VAE (β = 1) | 214,9 ± 3,2 | 0,96 | 0 % |
| DCGAN | 95,1 ± 4,8 | 0,97 à 0,98 | 0 % |
| GAN conditionnel (une graine) | 37,3 | 0,994 | — |

Repères de l'échelle : des images réelles floutées (moyenne 3 × 3) sont à 212, bruitées (σ = 0,2) à 491. Le GAN conditionnel produit la classe demandée pour 87,1 % de ses images (selon le classifieur). Une **quasi-copie** est un échantillon plus proche de l'entraînement (en distance entre pixels) que 99 % des images de validation ; la part attendue pour des images réelles nouvelles est 1 %. Ces distances sont calculées sur les features d'un petit CNN : **elles ne sont pas comparables aux FID publiés**, calculés avec Inception v3.

## Les limites et risques

- **Qualité** : les images du VAE sont floues et sans texture ; celles du DCGAN sont nettes mais parfois déformées (sandales, contours). Les proportions des classes diffèrent de celles des données (le VAE produit trop de chemises et de sacs, trop peu de sandales ; le DCGAN trop peu de pulls).
- **Mesure** : un seul extracteur de features, de 92,5 % d'accuracy, qui confond surtout chemise, T-shirt, pull et manteau ; trois graines seulement ; β choisi sur la validation qui sert aussi aux mesures (avec d'autres images tirées).
- **Mémorisation** : aucune quasi-copie détectée, mais le contrôle ne compare que des pixels, sur 2 000 échantillons par modèle ; une copie décalée ou retournée lui échapperait.
- **Risques et usages responsables** : des images générées présentées comme réelles (une fausse photo de produit) ; un générateur entraîné sur des données personnelles pourrait en restituer, d'où le contrôle des voisins, à refaire sur toute nouvelle donnée ; l'article 50 du règlement européen sur l'IA (AI Act) demande aux fournisseurs de systèmes qui génèrent des images de marquer leurs sorties comme artificielles. Avant tout usage réel : un extracteur de features reconnu, plus de graines, un contrôle de mémorisation plus robuste (features plutôt que pixels), et un marquage des images produites.
