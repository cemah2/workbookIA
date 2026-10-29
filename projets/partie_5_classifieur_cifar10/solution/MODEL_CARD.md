# Fiche modèle · Classifieur CIFAR-10 (`cifarkit`)

## Le modèle

Un CNN de type VGG : trois étages de deux convolutions 3 × 3 (16, 32 puis 64 canaux), chacune suivie d'une batchnorm et d'une ReLU, un max pooling 2 × 2 après les deux premiers étages, un pooling global moyen et une couche dense vers les dix logits ; **72 954 paramètres** (`{"arch": "vgg", "widths": [16, 32, 64], "downsample": "pool"}`). Entraîné le 2026-10-29 sur 45 000 images de CIFAR-10, 30 epochs, SGD de Nesterov (momentum 0,9, weight decay 10⁻⁴), `OneCycleLR` jusqu'à un learning rate de 0,145, cross-entropy avec label smoothing de 0,1, mini-batches de 128, augmentation (recadrage après padding de 4 pixels, symétrie horizontale), graine 0 ; 28 minutes sur un CPU à deux cœurs (Python 3.13, PyTorch 2.11, torchvision 0.26). Le prétraitement est enregistré avec lui : chaque image est ramenée dans $[0, 1]$, puis normalisée par canal avec la moyenne (0,4916 ; 0,4823 ; 0,4466) et l'écart-type (0,2471 ; 0,2435 ; 0,2615) des images d'entraînement.

Les fichiers : `model.pt` (le `state_dict`) et `model.json` (la configuration, la normalisation, les classes, les métriques, les versions). Pour le charger : `cifarkit.load_model(dossier)` ; pour classer des images : `python -m cifarkit.predict image.jpg --model dossier`. `model.pt` se charge avec `torch.load(..., weights_only=True)` : un fichier de poids, pas du code à exécuter.

## Usage prévu

Un exercice de portfolio et d'enseignement : classer des photos 32 × 32 dans les dix classes de CIFAR-10, et servir de point de départ pour apprendre à livrer un modèle (chapitre bonus B7 du workbook). **Usages exclus** : toute décision qui compte pour quelqu'un (sécurité, surveillance, tri de personnes ou de biens), des images d'autres classes ou d'une autre nature (le modèle choisit toujours une des dix classes, même pour une image qui n'en contient aucune), et tout contexte où une perturbation adversariale est possible (voir les limites).

## Les données

CIFAR-10 (A. Krizhevsky, « Learning Multiple Layers of Features from Tiny Images », rapport technique, 2009) : 60 000 photos couleur 32 × 32, tirées de la base *80 Million Tiny Images* et étiquetées à la main, 6 000 par classe ; pas de licence formelle, les auteurs demandent de citer le rapport. La base d'origine, *80 Million Tiny Images*, a été retirée en 2020 pour des images et des mots offensants ; CIFAR-10 en est un sous-ensemble filtré, mais ses images viennent du web des années 2000. Découpage : 45 000 images d'entraînement et 5 000 de validation (split stratifié, graine 0) dans les 50 000 d'entraînement ; les 10 000 images de test n'ont servi qu'une fois, pour la mesure finale.

## Les performances

**Accuracy de test : 85,0 %** (10 000 images, ouvert une fois ; ±0,7 point à 95 %), loss 0,484 ; 85,8 % sur la validation, qui a servi aux choix. Par classe : bateau 93,4 %, automobile 93,0 %, camion 91,5 %, cheval 86,9 %, cerf 86,3 %, grenouille 86,2 %, avion 85,9 %, oiseau 78,0 %, chien 77,5 %, chat 71,2 %. Les confusions principales : chat et chien (213 images dans les deux sens), avion et bateau (88), automobile et camion (85). Repères : toujours prédire la même classe donne 10 %, un MLP 52,8 %, un petit CNN 72,1 % (validation) ; un ResNet-18 pré-entraîné sur ImageNet, gelé, avec une nouvelle tête entraînée sur 10 000 images agrandies en 224 × 224, fait 86,1 % en validation, pour environ 50 fois plus de calcul par image.

## Les limites et risques

- **Aucune robustesse adversariale** : l'attaque FGSM fait tomber l'accuracy (1 000 images de test) de 86,5 % à 38,9 % avec ε = 1/255, 19,5 % avec 2/255, 7,3 % avec 4/255 et 3,9 % avec 8/255, des perturbations invisibles.
- **Des probabilités trop sûres d'elles** : des erreurs reçoivent 94 à 98 % ; les probabilités ne sont pas calibrées (aucune calibration n'a été faite).
- **Hors distribution** : la taille (32 × 32), le style et l'époque des photos ; aucune mesure sur d'autres images (CIFAR-10.1, photos actuelles réduites).
- **Une seule graine** : les écarts de moins d'un point entre réglages ne sont pas établis.
- **Les labels** : quelques images du test semblent mal étiquetées ; le modèle s'appuie aussi sur le fond (l'herbe pour un cerf, l'eau pour un bateau).
- Avant tout usage réel : mesurer sur des données du terrain, calibrer les probabilités, ajouter une option « aucune des dix classes », et évaluer la robustesse (entraînement adversarial si l'usage l'exige).
