# minifw : un mini-framework de deep learning en NumPy, validé contre PyTorch

> TODO : une phrase qui résume le projet et son résultat principal (ce que fait minifw, l'écart à PyTorch dans les tests et sur un entraînement entier, l'accuracy de test sur Fashion-MNIST face à celle de PyTorch).

Mini-projet du checkpoint IV du workbook *Deep Learning* (d'après A. Glassner) : des couches (`Dense`, activations, `Dropout`, `BatchNorm1d`), un conteneur `Sequential`, des losses et une boucle d'entraînement (AdamW, warmup + cosinus, clipping, early stopping), écrits en NumPy à partir de ma librairie `mylearn`, et prouvés justes par des tests d'équivalence avec PyTorch.

<!-- Ce fichier est le modèle de ton README de portfolio : remplace chaque paragraphe qui commence par TODO (et la ligne TODO du tableau) par ton texte, avec tes propres chiffres. Tu peux l'écrire en anglais si tu vises des postes à l'international, avec le même plan (## The goal, ## Code architecture, ## Results, ## Limitations, ## What I learned). -->

## L'objectif

TODO : ce que fait minifw, ce qu'il ne fait pas (pas de GPU, pas d'autograd, pas de convolution), et comment tu prouves qu'il est juste.

## L'architecture du code

TODO : les quatre modules et ce qu'ils contiennent ; le contrat de chaque couche (`forward`, `backward`, `parameters`, `gradients`, `training`) ; comment `Sequential` enchaîne les passes et comment l'optimiseur de `mylearn` modifie les poids sur place ; un petit schéma (un bloc de code ou une liste suffit).

```
TODO : le schéma des modules
```

## Les tests

TODO : ce que vérifient tes tests (une ligne par famille), contre quoi (PyTorch, en float64), avec quelle tolérance ; le nombre de tests et de mutants attrapés.

## Les résultats

TODO : l'entraînement identique à PyTorch (l'écart des poids après deux epochs) ; minifw contre PyTorch sur Fashion-MNIST (accuracy, temps par epoch, nombre de paramètres) ; l'ablation ; la configuration retenue, sa règle de choix et son accuracy de test (le test ouvert une seule fois).

| | minifw | PyTorch |
|---|---|---|
| TODO | | |

![Les deux demi-lunes : la frontière apprise par minifw et ses courbes de loss](figures/moons_frontiere.png)

![Fashion-MNIST : les courbes de minifw et de PyTorch](figures/fashion_courbes.png)

![La matrice de confusion du test](figures/fashion_confusion.png)

## Les limites

TODO : honnêtement : la vitesse face à PyTorch et pourquoi, le CPU seulement, le float64, pas d'autograd (chaque couche a sa passe arrière écrite à la main), une seule graine, l'accuracy d'un MLP face à un réseau convolutif.

## Ce que j'ai appris

TODO : trois ou quatre points concrets (une erreur que les tests ont attrapée, ce que fait vraiment `model.eval()`, pourquoi les losses se calculent sur les logits…).

## Reproduire

```bash
python tools/start_chapter.py CP4                               # from the workbook repository
python -m pytest mon_travail/projets/partie_4_mini_framework -q  # the equivalence tests
```

Puis exécute `mp4_mini_framework.ipynb` de bout en bout (`FAST_MODE = False` pour les chiffres de ce README).
