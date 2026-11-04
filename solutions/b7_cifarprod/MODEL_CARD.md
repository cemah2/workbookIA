# Fiche modèle · cifar-prod (VGG 16-32-64 servi par une API)

## Le modèle

Le VGG du mini-projet 5 : trois étages de deux convolutions 3 × 3 (16, 32 puis 64 canaux), chacune suivie d'une
batchnorm et d'une ReLU, un max pooling après les deux premiers étages, un pooling global moyen et une couche dense
vers les dix logits ; **72 954 paramètres**. Entraîné le 2026-10-29 sur 45 000 images de CIFAR-10 (30 epochs, SGD de
Nesterov, `OneCycleLR`, label smoothing 0,1, augmentation), 28 minutes sur un CPU à deux cœurs. L'artefact
(`model.safetensors`, `meta.json`, version du paquet 0.1.0) contient la normalisation apprise sur l'entraînement
(moyenne 0,4916 ; 0,4823 ; 0,4466 ; écart-type 0,2471 ; 0,2435 ; 0,2615, pixels ramenés dans [0, 1]), l'ordre des
classes, l'empreinte SHA-256 des images et des labels d'entraînement (`b76c6f2c…`) et celle des poids.

## Usage prévu

Classer des photos couleur 32 × 32 dans les dix classes de CIFAR-10, par une API HTTP (`POST /predict`), pour
l'enseignement et un portfolio. **Usages exclus** : toute décision qui compte pour une personne, des images d'autres
classes ou d'une autre nature (le modèle choisit toujours une des dix classes), un contexte adversarial.

## Les données

CIFAR-10 (A. Krizhevsky, 2009) : 60 000 photos 32 × 32 étiquetées à la main, 6 000 par classe, tirées du web des
années 2000 (base *80 Million Tiny Images*, retirée en 2020). Entraînement : 45 000 des 50 000 images d'entraînement
(découpage stratifié, graine 0) ; validation : les 5 000 autres ; test : 10 000 images, ouvert une fois.

## Les performances

Accuracy de test **85,0 %** (± 0,7 point à 95 %) ; de 71,2 % (chat) à 93,4 % (bateau) selon la classe. Servi par
FastAPI sur un CPU à deux cœurs : p95 ≈ 8 à 10 ms par requête d'une image. L'export `torch.export` donne exactement les
mêmes logits que le modèle.

## Les limites, la surveillance et les risques

- **Sensible au bruit** : un bruit gaussien d'écart-type 25 (sur 255) sur les pixels fait tomber l'accuracy de 83 % à
  36 % sur 500 images de test, sans changer la luminosité moyenne : la surveillance des entrées par la seule
  luminosité ne le voit pas, celle des classes prédites (PSI) le voit.
- **Peu sensible à l'éclairage** : des images assombries (pixels × 0,6) changent beaucoup la distribution des entrées
  (PSI de la luminosité au-dessus de 3) pour une perte d'accuracy de 3 points.
- Des probabilités non calibrées, aucune robustesse adversariale (FGSM à ε = 1/255 : 39 % d'accuracy), une seule graine.
- **Surveiller** : la distribution des classes prédites et la confiance moyenne par fenêtre (PSI à 10 intervalles,
  seuils 0,1 et 0,25), la luminosité et le contraste des entrées, la latence p95 et le taux de réponses 4xx et 5xx ;
  garder l'artefact précédent pour revenir en arrière.
- **Données personnelles** : une photo envoyée peut montrer une personne ; les réponses 422 ne renvoient pas l'entrée,
  les journaux ne gardent que des statistiques d'images, pour une durée écrite.
