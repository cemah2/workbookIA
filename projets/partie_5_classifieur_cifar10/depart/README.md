# cifarkit : un classifieur d'images CIFAR-10 en PyTorch, de la baseline au modèle livré

> TODO : une phrase qui résume le projet et son résultat principal (le modèle retenu, son accuracy de test, ce qu'il a fallu pour y arriver, et ce que coûte un ResNet-18 pré-entraîné pour faire mieux ou moins bien).

Mini-projet du checkpoint V du workbook *Deep Learning* (d'après A. Glassner) : un package PyTorch testé (`cifarkit` : le pipeline de données, les modèles, la boucle d'entraînement, la livraison), une démarche expérimentale notée dans un journal, un modèle livré avec son prétraitement, et une analyse d'erreurs.

<!-- Ce fichier est le modèle de ton README de portfolio : remplace chaque paragraphe qui commence par TODO (et la ligne TODO des tableaux) par ton texte, avec tes propres chiffres. Tu peux l'écrire en anglais si tu vises des postes à l'international, avec le même plan (## The goal, ## Protocol, ## Results, ## Limitations, ## Reproduce, ## What I learned). -->

## L'objectif

TODO : le problème (dix classes, des photos 32 × 32), la métrique et pourquoi elle convient, l'objectif chiffré, ce que le projet ne cherche pas à faire (battre l'état de l'art).

## Le protocole

TODO : le découpage (entraînement, validation stratifiée, test ouvert une fois), la normalisation apprise sur l'entraînement, l'augmentation, les graines, la règle « une variable à la fois » et le journal.

## L'architecture du code

TODO : les quatre modules de `cifarkit` et ce qu'ils contiennent ; ce que vérifient les tests (et combien de mutants ils attrapent) ; comment le modèle livré se recharge.

## Les résultats

TODO : les baselines ; la comparaison des architectures (un tableau) ; la recherche d'hyperparamètres ; le modèle retenu, sa règle de choix, son accuracy de validation et de test (le test ouvert une seule fois) ; la comparaison avec le ResNet-18 pré-entraîné.

| Modèle | Paramètres | Accuracy de validation | Accuracy de test |
|---|---|---|---|
| TODO | | | |

![Quelques tirages de l'augmentation de données](figures/augmentation.png)

![L'accuracy de validation des cinq architectures comparées](figures/architectures.png)

![Les courbes d'entraînement du modèle livré](figures/courbes_final.png)

![La matrice de confusion du test](figures/confusion.png)

![Les 20 erreurs les plus sûres d'elles](figures/erreurs.png)

![L'accuracy sous l'attaque FGSM](figures/fgsm.png)

## L'analyse d'erreurs

TODO : les classes les plus confondues et pourquoi ; ce que montrent les 20 erreurs (lesquelles sont excusables) ; la robustesse face à FGSM.

## Les limites

TODO : honnêtement : la taille des images, une seule graine pour le modèle final, la validation qui a servi à tous les choix, la fragilité face à FGSM, les images hors de la distribution de CIFAR-10.

## Reproduire

TODO : les commandes pour lancer les tests, entraîner (en mode rapide et en mode complet, sur Colab) et classer une image avec `python -m cifarkit.predict`.

## Ce que j'ai appris

TODO : trois ou quatre points concrets (une erreur que les tests ont attrapée, l'effet réel d'un choix, ce qu'apporte la reprise exacte…).
