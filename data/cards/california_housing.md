# California Housing

| | |
|---|---|
| **Fichier de secours** | `data/california_housing.csv` (1,9 Mo) |
| **Chargement** | `wb.datasets.load_california(as_frame=True, return_X_y=False, source="auto")` |
| **Source principale** | `sklearn.datasets.fetch_california_housing` (téléchargé, puis mis en cache) ; la copie CSV est utilisée si le téléchargement échoue |
| **Taille** | 20 640 lignes × 9 colonnes (8 features + la cible) |
| **Tâche type** | régression |
| **Licence** | données publiques du recensement américain, diffusées par StatLib ; pas de licence explicite, usage éducatif courant |
| **Téléchargé le** | 2026-09-29 (le CSV est identique octet pour octet aux valeurs de scikit-learn, vérifié par un test) |

## Provenance
Recensement américain de 1990, une ligne par *block group* (quartier de 600 à 3 000 habitants environ). Référence : Pace, R. K. & Barry, R. (1997). *Sparse Spatial Autoregressions*. Statistics & Probability Letters, 33, 291-297.

## Variables
| Colonne | Description |
|---|---|
| `MedInc` | revenu médian du quartier (en dizaines de milliers de dollars) |
| `HouseAge` | âge médian des logements (années) |
| `AveRooms` | nombre moyen de pièces par ménage |
| `AveBedrms` | nombre moyen de chambres par ménage |
| `Population` | population du quartier |
| `AveOccup` | nombre moyen de personnes par ménage |
| `Latitude`, `Longitude` | position du quartier |
| `MedHouseVal` | **cible** : valeur médiane des logements, en centaines de milliers de dollars (3,5 = 350 000 $) |

## Biais et limites
- **Valeurs plafonnées** : la cible est tronquée à 5,0 (≥ 500 000 $), l'âge à 52 ans et le revenu à 15. Un modèle ne peut pas apprendre au-delà : à repérer sur un histogramme (ch. 12).
- Moyennes par ménage parfois aberrantes (`AveOccup` jusqu'à 1 243) : quartiers atypiques (foyers, résidences).
- Données de 1990 : les prix d'aujourd'hui n'ont plus rien à voir. C'est un jeu d'entraînement, pas un outil d'estimation.

## ⚖️ Pourquoi pas Boston Housing ?
Le dataset historique « Boston Housing » a été **déprécié dans scikit-learn 1.0 puis retiré en 1.2**. Ses auteurs avaient construit une variable « B » à partir de la proportion d'habitants noirs, en supposant que la ségrégation raciale avait un effet positif sur les prix. Les mainteneurs de scikit-learn déconseillent son usage, sauf pour étudier les questions d'éthique en data science. Ce cas fera l'objet d'un exercice ⚖️.
Source : [documentation scikit-learn 1.1, `load_boston`](https://scikit-learn.org/1.1/modules/generated/sklearn.datasets.load_boston.html).

## Chapitres
8 (400 districts tirés au hasard, revenu médian entre 1 et 8 et valeur non plafonnée : régression polynomiale `PolyFit` du revenu, choix du degré sur un jeu de validation, optimisme du score de validation du degré retenu), 9 (overfitting, régularisation), 12 (préparation), 14 (ensembles, gradient boosting), 15 (scikit-learn), B6 (explicabilité).
