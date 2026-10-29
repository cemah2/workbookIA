# Fiche modèle · Classifieur CIFAR-10 (`cifarkit`)

<!-- Le modèle d'une fiche modèle (model card : Mitchell et coll., 2019, « Model Cards for Model Reporting ») : remplace chaque paragraphe qui commence par TODO. Une fiche modèle dit à quoi sert un modèle, à quoi il ne doit pas servir, et comment il se comporte, globalement et classe par classe. -->

## Le modèle

TODO : l'architecture (la configuration de `model.json`), le nombre de paramètres, l'entraînement (données, epochs, optimiseur, planning, augmentation, graine), le prétraitement (la normalisation enregistrée), la date et les versions ; les fichiers (`model.pt`, `model.json`) et comment les charger (`cifarkit.load_model`, `python -m cifarkit.predict`).

## Usage prévu

TODO : l'usage pour lequel il a été construit (un exercice de portfolio, des photos 32 × 32 des dix classes de CIFAR-10) ; les **usages exclus** (toute décision qui compte, des images d'autres classes ou d'une autre nature) et pourquoi.

## Les données

TODO : CIFAR-10 (la source, la date, la licence, la citation demandée), le découpage, ce que les images contiennent et ce qu'elles ne contiennent pas.

## Les performances

TODO : l'accuracy de test (et son incertitude : environ ±0,7 point à 95 % sur 10 000 images, vers 85 %), l'accuracy par classe, les confusions principales, la comparaison aux baselines.

## Les limites et risques

TODO : la robustesse (FGSM : l'accuracy pour quelques ε), les images hors distribution (taille, style, classes inconnues : le modèle choisit toujours une des dix classes), une seule graine, ce qu'il faudrait faire avant tout usage réel.
