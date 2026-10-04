# Palmer Penguins

| | |
|---|---|
| **Fichiers** | `data/penguins.csv` (version simplifiée), `data/penguins_raw.csv` (version brute) |
| **Chargement** | `wb.datasets.load_penguins(dropna=False)`, `wb.datasets.load_penguins_raw()` |
| **Taille** | 344 lignes × 8 colonnes (brute : 344 × 17) ; 15 Ko et 53 Ko |
| **Tâche type** | classification (espèce), statistiques descriptives, nettoyage |
| **Licence** | CC0 1.0 (domaine public), conformément à la politique de données de Palmer Station LTER |
| **Source** | R package *palmerpenguins*, fichiers `inst/extdata/` du dépôt GitHub `allisonhorst/palmerpenguins` |
| **Téléchargé le** | 2026-09-29 |

## Provenance
Mesures prises entre 2007 et 2009 par la Dr. Kristen Gorman et la station Palmer (Antarctique, réseau LTER) sur trois îles de l'archipel Palmer.
Citation : Horst AM, Hill AP, Gorman KB (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. R package version 0.1.0. doi:10.5281/zenodo.3960218.

## Variables (`penguins.csv`)
| Colonne | Type | Description |
|---|---|---|
| `species` | texte | Adelie (152), Gentoo (124), Chinstrap (68) |
| `island` | texte | Biscoe (168), Dream (124), Torgersen (52) |
| `bill_length_mm` | réel | longueur du bec (mm) |
| `bill_depth_mm` | réel | épaisseur du bec (mm) |
| `flipper_length_mm` | réel | longueur de la nageoire (mm) |
| `body_mass_g` | réel | masse (g) |
| `sex` | texte | `male` / `female` |
| `year` | entier | 2007, 2008 ou 2009 |

**Valeurs manquantes** : 2 manchots sans aucune mesure, 11 sans sexe renseigné. `dropna=True` garde 333 lignes.
La version brute (`penguins_raw.csv`) garde les noms d'origine (« Culmen Length (mm) »…), les dates, les isotopes (δ15N, δ13C) et des commentaires : idéale pour les exercices de nettoyage (ch. 12).

## Biais et limites
- Petit échantillon (344 individus) d'une seule région : les conclusions ne se généralisent pas à tous les manchots.
- Classes déséquilibrées (Chinstrap deux fois moins représentés) et espèces liées aux îles (les Gentoo ne viennent que de Biscoe) : risque de « raccourci » où le modèle apprend l'île plutôt que la morphologie.
- Le sexe est codé de façon binaire à partir d'analyses moléculaires.
- Dans la version brute, `Individual ID` n'identifie pas un oiseau : la même valeur revient d'une saison à l'autre, parfois pour une autre espèce ou une autre île (N11A1 est un Adélie en 2007-08, puis un Gentoo en 2008-09). Seule la paire (`Individual ID`, `studyName`) désigne une mesure ; ne pas s'en servir pour regrouper des mesures d'un même individu.
- Valeurs manquantes de la version brute concentrées sur la première saison (prélèvements sanguins) : une suppression des lignes incomplètes frappe surtout les Adélie de 2007-08 (⚖️ 12.10).

## Chapitres
0A (découverte pandas), 0B (deux manchots vus comme des vecteurs, 0B.8), 1 (vocabulaire du ML, règles d'expert, arbre de décision, clustering), 2 (statistiques descriptives, règle 68-95-99,7, bootstrap, covariance et corrélation), 3 (tables de contingence, métriques de classification), 4 (Bayes : l'espèce sachant l'île, prior et vraisemblance séparés ; refactorisation testée), 7 (clustering sans labels : k-means face aux espèces, avec des features standardisées ; pureté ; features de bruit ajoutées pour le phénomène de Hughes ; un-contre-tous et un-contre-un sur les trois espèces), 8 (fichier trié par espèce : piège du découpage sans mélange ; hold-out stratifié, k-fold simple et stratifiée ; le plus proche voisin qui apprend par cœur, même des espèces mélangées au hasard ; l'île comme cible d'une étude « trop belle » à corriger ; comparaison appariée de deux modèles ; le sexe comme cible du défi, avec le découpage 233 + 100 du ch. 1), 10 (perceptrons : Adélie contre Chinstrap, mesures brutes contre standardisées ; quelles paires d'espèces une droite sépare ; trois espèces en un-contre-tous et en un-contre-un), 11 (l'induction mise à l'épreuve : la part de Gentoo estimée sur 5, 20 ou 100 manchots, sur la seule île Biscoe, ou avec une capture proportionnelle à la masse), CP1 (examen : standardisation colonne par colonne en CP1.3, bootstrap de la masse des 119 Gentoo en CP1.5), CP2 (CP2.5 : plan d'évaluation sur les 333 manchots), 12 (la version brute : diagnostic, valeurs manquantes par espèce et par saison, encodages one-hot et ordinal à la main, préparation complète sans fuite), 13 (classifieurs), 14 (ensembles), 15 (scikit-learn), B6 (explicabilité).
