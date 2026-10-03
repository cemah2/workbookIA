# Prix des logements californiens : un protocole d'évaluation honnête

> TODO : une phrase qui résume le projet et son résultat principal (la RMSE du test et son intervalle, comparées à la référence naïve).

Mini-projet du checkpoint II du workbook *Deep Learning* (d'après A. Glassner) : des modèles linéaires (moindres carrés, Ridge, Lasso) et des zones géographiques par k-means, écrits avec ma librairie `mylearn`, choisis par validation croisée, et un jeu de test gelé dès le départ, ouvert une seule fois.

<!-- Ce fichier est le modèle de ton README de portfolio : remplace chaque paragraphe qui commence par TODO (et la ligne TODO du tableau) par ton texte, avec tes propres chiffres. Tu peux l'écrire en anglais si tu vises des postes à l'international, avec le même plan (## The problem, ## The data, ## The protocol, ## Results, ## Limitations). -->

## Le problème

TODO : qui utiliserait ces prédictions et pour quelle décision ; ce que vaut une erreur de 0,5 (en centaines de milliers de dollars) ; la mesure suivie (la RMSE) et pourquoi.

## Les données

TODO : la source (le recensement américain de 1990, diffusé par StatLib et scikit-learn ; Pace et Barry, 1997), les 8 mesures et la cible, le plafond à 500 000 dollars, les valeurs extrêmes ; le jeu de test : 20 % des districts, tirés avec une graine fixe et commités avant toute modélisation (donne la date du commit).

## Le protocole

TODO : les 5 folds communs à tous les modèles ; le prétraitement (bornes, z-scores, zones) appris dans chaque fold ; la règle de choix (une erreur type) ; le coffre ouvert une seule fois (`vault.json`).

## Les résultats

TODO : les choix (degré, α, nombre de zones) et pourquoi ; le tableau des scores ; ce que montrent les figures ; le score du test face à la validation croisée.

| Modèle | RMSE en validation croisée (moyenne ± écart-type, 5 folds) | RMSE du test |
|---|---|---|
| TODO | | |

![Courbes de validation de Ridge selon α, pour les degrés 1 à 4](figures/courbes_validation.png)

![La silhouette et la validation croisée selon le nombre de zones](figures/zones.png)

![Courbes d'apprentissage](figures/courbes_apprentissage.png)

![Les zones du modèle retenu et ses résidus hors fold](figures/carte_zones_residus.png)

## Les limites et l'équité

TODO : par exemple des données de 1990, la cible plafonnée, des districts voisins répartis entre entraînement et validation (dépendance spatiale), qui le modèle dessert mal (les résidus par région, les districts plafonnés), et ce qu'un usage réel demanderait.

## Reproduire

Depuis la racine du dépôt :

```bash
python -m pytest mon_travail/projets/partie_2_california_validation -q   # les tests du module
```

puis ouvrir `mp2_california.ipynb` et tout exécuter (quelques minutes sur un CPU). Les graines, les hyperparamètres retenus, les scores par fold et les versions des bibliothèques sont dans `results.json`.
