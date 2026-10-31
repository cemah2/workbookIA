# Fiche modèle · Générateurs de vêtements Fashion-MNIST (`genbench`)

<!-- Le modèle d'une fiche modèle (model card : Mitchell et coll., 2019, « Model Cards for Model Reporting ») : remplace chaque paragraphe qui commence par TODO. Une fiche modèle dit à quoi sert un modèle, à quoi il ne doit pas servir, comment on l'a mesuré, et ce qu'il peut faire de travers. Ici, elle décrit les deux générateurs (le VAE et le DCGAN) et le classifieur qui sert d'instrument. -->

## Les modèles

TODO : les architectures (le VAE convolutif et son β, le DCGAN et sa largeur, le GAN conditionnel si tu l'as fait), leurs nombres de paramètres, l'entraînement (images, epochs, optimiseurs, graines), la date et les versions ; le classifieur de référence et son accuracy de validation.

## Usage prévu

TODO : l'usage pour lequel ils ont été construits (un banc d'essai de portfolio et d'enseignement, des vignettes de vêtements en 28 × 28) ; les **usages exclus** (présenter les images comme de vraies photos de produits, entraîner un autre modèle sur ces images sans le dire…) et pourquoi.

## Les données

TODO : Fashion-MNIST (la source, Zalando, la date, la licence MIT, la citation), le découpage (50 000 et 10 000 images), ce que les images contiennent et ce qu'elles ne contiennent pas (des photos de catalogue, centrées, sur fond neutre, aucune personne).

## Les métriques

TODO : la distance de Fréchet de chaque générateur (moyenne et écart sur trois graines, le plancher, le nombre d'images), la répartition des classes générées, la part de quasi-copies ; ce que ces mesures ne disent pas (une distance calculée sur les features d'un petit CNN, non comparable aux FID publiés).

## Les limites et risques

TODO : les limites (images floues du VAE, défauts du DCGAN, petits modèles, une seule résolution), les risques et les usages responsables (des images générées présentées comme réelles, la mémorisation de données d'entraînement, le marquage des contenus générés), et ce qu'il faudrait faire avant tout usage réel.
