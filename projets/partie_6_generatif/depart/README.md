# genbench : un banc d'essai génératif, VAE contre DCGAN sur Fashion-MNIST

> TODO : une phrase qui résume le projet et son résultat principal (lequel des deux générateurs gagne sur la distance de Fréchet, de combien, sur trois graines, et ce que disent les contrôles d'effondrement de mode et de mémorisation).

Mini-projet du checkpoint VI du workbook *Deep Learning* (d'après A. Glassner) : un package PyTorch testé (`genbench` : les modèles, l'entraînement, l'évaluation), un protocole d'évaluation écrit avant les expériences, trois graines par générateur, une galerie et une fiche modèle.

<!-- Ce fichier est le modèle de ton README de portfolio : remplace chaque paragraphe qui commence par TODO (et la ligne TODO des tableaux) par ton texte, avec tes propres chiffres. Écris les sections « L'objectif » et « Le protocole d'évaluation » dès l'étape MP6.1, AVANT les expériences, et commite-les. Les liens des figures pointent vers celles du mode complet : si tu n'as que le mode rapide, remplace-les par les fichiers « …_rapide.png » et dis-le. Tu peux l'écrire en anglais si tu vises des postes à l'international, avec le même plan (## The goal, ## Evaluation protocol, ## Method, ## Results, ## Gallery, ## Limitations and responsible use, ## Reproduce, ## What I learned). -->

## L'objectif

TODO : le problème (générer des vêtements 28 × 28), les deux générateurs comparés, la question posée (lequel imite le mieux la distribution des images, sans effondrement de mode ni mémorisation), ce que le projet ne cherche pas à faire (l'état de l'art).

## Le protocole d'évaluation

TODO : écrit AVANT les expériences. L'instrument (le classifieur de référence et ses features), les images réelles de comparaison, le nombre d'images générées, les graines, la règle de choix de β, la règle de mémorisation (le quantile), et ce que tu regarderas à l'œil.

## La méthode

TODO : le classifieur de référence ; le VAE (architecture, loss, β) ; le DCGAN (règles de Radford, loss non saturante, checkpoint complet et reprise), le GAN conditionnel si tu l'as fait ; ce que vérifient les tests (et combien de mutants ils attrapent).

## Les résultats

TODO : l'accuracy du classifieur et les repères de la distance (plancher, flou, bruit) ; le choix de β (un tableau) ; la distance de Fréchet de chaque générateur sur trois graines (moyenne et écart) ; les classes générées ; la mémorisation.

| Modèle | Distance de Fréchet (moyenne ± écart, 3 graines) | Entropie des classes | Quasi-copies |
|---|---|---|---|
| TODO | | | |

![Huit images de validation par classe](figures/donnees.png)

![Les courbes de reconstruction et de KL du VAE pour chaque β](figures/vae_beta_courbes.png)

![Reconstructions et échantillons du VAE pour chaque β](figures/vae_beta_echantillons.png)

![Les courbes d'entraînement du DCGAN](figures/gan_courbes.png)

![Le même bruit après chaque epoch du DCGAN](figures/gan_progression.png)

![La distance de Fréchet par modèle et par graine](figures/frechet.png)

![Les classes prédites des images générées](figures/classes.png)

![Les échantillons les plus proches de l'entraînement et leurs voisins](figures/voisins.png)

## La galerie

TODO : une légende pour chaque figure : ce qu'on y voit, et ce qu'il faut en retenir.

![Échantillons du VAE et du DCGAN](figures/galerie.png)

![Interpolations sphériques et linéaires](figures/interpolations.png)

![Le GAN conditionnel : une ligne par classe](figures/conditionnel.png)

## Les limites et les usages responsables

TODO : honnêtement : la taille des modèles et des images, une distance calculée sur les features d'un petit CNN (non comparable aux FID publiés), trois graines seulement, la validation qui a servi au choix de β ; les risques des images générées et les usages à éviter.

## Reproduire

TODO : la commande unique qui refait tout (le notebook en mode rapide ou complet), celle des tests, et où trouver `config.json` et `results.json`.

## Ce que j'ai appris

TODO : trois ou quatre points concrets (une erreur que les tests ont attrapée, ce que la distance de Fréchet ne voit pas, ce qu'apporte la reprise exacte…).
