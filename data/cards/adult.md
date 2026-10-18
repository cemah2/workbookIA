# Adult (Census Income)

| | |
|---|---|
| **Fichier de secours** | `data/adult.csv.gz` (0,6 Mo compressé, un CSV de 5,3 Mo) |
| **Chargement** | `wb.datasets.load_adult(source="auto")` |
| **Source principale** | OpenML, `sklearn.datasets.fetch_openml(data_id=1590)` (téléchargé, puis mis en cache) ; la copie versionnée est utilisée si le téléchargement échoue (`source="csv"` pour la forcer) |
| **Taille** | 48 842 lignes × 15 colonnes (14 features + la cible `class`) |
| **Tâche type** | classification binaire : le revenu annuel dépasse-t-il 50 000 $ ? |
| **Licence** | Creative Commons Attribution 4.0 (CC BY 4.0), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) |
| **Téléchargé le** | 2026-10-16 (la copie est identique, valeur par valeur, à ce que rend OpenML, vérifié par un test) |

## Provenance
Extrait par Barry Becker de la « 1994 Census database » du US Census Bureau (le supplément annuel ASEC de la *Current Population Survey*, d'après Ding et coll., 2021 : la documentation de l'UCI ne nomme pas l'enquête), avec les conditions âge > 16, revenu > 100 $, poids d'échantillonnage > 1 et au moins une heure de travail par semaine ; publié en 1996 sur le dépôt de l'UCI. Référence : Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5XW20](https://doi.org/10.24432/C5XW20). OpenML 1590 réunit en une seule table les fichiers d'entraînement (32 561 lignes) et de test (16 281 lignes) d'origine.

## Variables
| Colonne | Type | Description |
|---|---|---|
| `age` | entier | âge (17 à 90 ans) |
| `workclass` | texte | secteur d'emploi (Private, Self-emp-not-inc, Local-gov…) ; **2 799 manquants** |
| `fnlwgt` | entier | poids d'échantillonnage de l'enquête (le nombre de personnes que la ligne représente) : pas une caractéristique de la personne |
| `education` | texte | dernier diplôme (16 niveaux, de Preschool à Doctorate) |
| `education-num` | entier | le même niveau, codé de 1 à 16 : **une copie ordinale** de `education` |
| `marital-status` | texte | situation matrimoniale (7 modalités) |
| `occupation` | texte | métier (14 modalités) ; **2 809 manquants** |
| `relationship` | texte | place dans le foyer (Husband, Wife, Own-child…) : mêle la situation familiale et le sexe |
| `race` | texte | 5 catégories du recensement de 1994 |
| `sex` | texte | Male ou Female (deux modalités seulement, celles du recensement) |
| `capital-gain`, `capital-loss` | entiers | plus-values et moins-values de l'année ; nulles pour 92 % et 95 % des lignes ; `capital-gain` plafonné à 99 999, `capital-loss` va jusqu'à 4 356 |
| `hours-per-week` | entier | heures de travail par semaine (1 à 99, valeur plafond) |
| `native-country` | texte | pays de naissance (41 pays) ; **857 manquants** |
| `class` | texte | **cible** : `"<=50K"` (37 155 lignes, 76 %) ou `">50K"` (11 687 lignes, 24 %) |

Les valeurs manquantes sont des « ? » dans les fichiers d'origine ; OpenML et `load_adult` les rendent en `NaN`. Le fichier contient 52 lignes en double.

## Biais et limites
- **Une photographie de 1994** : 50 000 $ de 1994 valent environ le double en dollars d'aujourd'hui, et le marché du travail a changé. Le modèle n'apprend rien d'utilisable sur les revenus actuels : c'est un jeu d'entraînement, pas un outil de décision.
- **Un seuil arbitraire et haut** : 50 000 $ était le 76ᵉ centile des revenus individuels de 1994, et les 88ᵉ et 89ᵉ pour les personnes noires et les femmes (Ding et coll., 2021) : la plupart d'entre elles sont sous le seuil, et une accuracy élevée sur ces groupes ne dit presque rien.
- **Des attributs sensibles** (`sex`, `race`, `native-country`, `marital-status`, `relationship`) et des **variables de substitution** : `relationship` code « Husband » et « Wife », qui révèlent le sexe ; retirer `sex` ne suffit pas à rendre un modèle « aveugle » au sexe.
- **Des catégories datées ou réductrices** (les cinq « races » du recensement de 1994, deux sexes), à présenter comme telles.
- **Un benchmark fatigué** : utilisé dans des milliers d'articles d'équité algorithmique, Adult a été « mis à la retraite » par Ding, Hardt, Miller et Schmidt (2021), qui proposent des tâches plus récentes et plus grandes tirées de l'enquête ACS (paquet `folktables`, tâche `ACSIncome`).
- `fnlwgt` n'est pas une caractéristique de la personne : un modèle qui s'en sert apprend une particularité de l'échantillonnage.

Sources : [UCI, Adult](https://archive.ics.uci.edu/dataset/2/adult) ; Ding, F., Hardt, M., Miller, J. & Schmidt, L. (2021). [*Retiring Adult: New Datasets for Fair Machine Learning*](https://arxiv.org/abs/2108.04884). NeurIPS 2021.

## Chapitres
MP3 (le mini-projet de la partie III : pipeline tabulaire complet, classes déséquilibrées, seuil de décision et audit des erreurs par sous-groupe), B6 (explicabilité et équité).
