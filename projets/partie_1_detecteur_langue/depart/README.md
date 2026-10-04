# Détecteur de langue anglais / français *from scratch*

> TODO : une phrase qui résume le projet et son résultat principal (par exemple l'accuracy obtenue sur des extraits de 20 caractères).

Mini-projet du checkpoint I du workbook *Deep Learning* (d'après A. Glassner) : un classifieur *Naive Bayes* sur les lettres, écrit sans bibliothèque de machine learning, avec ma librairie `mylearn` (statistiques, mesures de qualité, règle de Bayes, descente de gradient, théorie de l'information).

<!-- Ce fichier est le modèle de ton README de portfolio : remplace chaque paragraphe qui commence par TODO par ton texte, avec tes propres chiffres. Tu peux l'écrire en anglais si tu vises des postes à l'international, avec le même plan (## The problem, ## The data, ## The method, ## Results, ## Limitations). -->

## Le problème

TODO : qui utiliserait ce détecteur, sur quels textes (et de quelle longueur), quelle erreur coûte le plus, et la mesure principale que je suis.

## Les données

TODO : les deux romans (source, licence), le découpage en chapitres d'entraînement, de validation et de test, et pourquoi il évite les fuites ; les longueurs des extraits et leur nombre.

## La méthode

TODO : en cinq à huit lignes : distributions de lettres lissées, log-vraisemblance (hypothèse i.i.d.), règle de Bayes avec un prior, astuce log-sum-exp, température ajustée par descente de gradient.

## Les résultats

TODO : le tableau par longueur d'extrait (accuracy et ROC-AUC avec leurs intervalles bootstrap, F1), la figure principale, la calibration (température trouvée, Brier et log loss avant et après) et l'effet du prior.

![Accuracy et ROC-AUC selon la longueur de l'extrait](figures/accuracy_auc_longueur.png)

## Les limites

TODO : par exemple des textes du XIXᵉ siècle, deux langues et deux auteurs seulement, l'hypothèse i.i.d., les noms propres, les extraits très courts.

## Pistes

TODO : ce que je ferais ensuite (bigrammes, troisième langue, comparaison avec un détecteur pré-entraîné…).

## Reproduire

Depuis la racine du dépôt :

```bash
python -m pytest mon_travail/projets/partie_1_detecteur_langue -q   # les tests du module
```

puis ouvrir `mp1_detecteur_langue.ipynb` et tout exécuter (environ une minute sur un CPU).
