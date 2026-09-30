# Cheatsheet pandas

> Aide-mémoire rempli au fil des chapitres (0A, 2, 12). Une ligne = une commande utile + ce qu'elle fait.

## Lire et écrire (CSV)

| Code | Effet | Ch. |
|---|---|---|
| `df = pd.read_csv(chemin)` | lit un fichier CSV dans un DataFrame | 0A |
| `wb.datasets.data_dir() / "penguins.csv"` | chemin portable vers les données du workbook (`pathlib`) | 0A |
| `df.to_csv(chemin, index=False)` | écrit le DataFrame (sans la colonne d'index) | 0A |

## Explorer un DataFrame

| Code | Effet | Ch. |
|---|---|---|
| `df.head()`, `df.tail(3)`, `df.sample(5, random_state=0)` | premières, dernières, lignes au hasard | 0A |
| `df.shape`, `df.columns`, `df.dtypes` | (lignes, colonnes), noms et types des colonnes | 0A |
| `df.info()` | types et nombre de valeurs **non manquantes** par colonne | 0A |
| `df.describe()` | statistiques des colonnes numériques (count, mean, std, min, quartiles, max) | 0A |
| `df["species"].unique()`, `.nunique()` | valeurs distinctes ; leur nombre | 0A |
| `df["species"].value_counts(normalize=True)` | effectifs (ou proportions) de chaque valeur, triés | 0A |
| `df.sort_values("body_mass_g", ascending=False)` | trier les lignes | 0A |

## Sélectionner (loc, iloc, filtres)

| Code | Effet | Ch. |
|---|---|---|
| `df["island"]` / `df[["island", "sex"]]` | une colonne (Series) / plusieurs colonnes (DataFrame) | 0A |
| `df.loc[100, "island"]` | par **étiquette** (index, nom de colonne) | 0A |
| `df.iloc[-1]`, `df.iloc[:5, :2]` | par **position** | 0A |
| `df[df["body_mass_g"] > 5500]` | les lignes qui vérifient une condition | 0A |
| `df[(df["species"] == "Gentoo") & (df["sex"] == "male")]` | deux conditions : `&`, `\|`, `~` et des parenthèses | 0A |
| `df.loc[masque, "col"] = v` | modifier les lignes filtrées (jamais `df[masque]["col"] = v`) | 0A |
| `df["island"].isin(["Dream", "Biscoe"])` | appartenance à une liste de valeurs | 0A |

## Valeurs manquantes

| Code | Effet | Ch. |
|---|---|---|
| `df.isna().sum()` / `.sum().sum()` | valeurs manquantes par colonne / au total | 0A |
| `df.dropna()` | retire toute ligne qui a **au moins une** valeur manquante | 0A |
| `df.dropna(subset=["body_mass_g"])` | ne regarde que certaines colonnes | 0A |
| `df["sex"].fillna("unknown")`, `s.fillna(s.median())` | remplacer les manquants par une valeur | 0A |
| `df.duplicated().sum()`, `df.drop_duplicates()` | compter, retirer les lignes en double | 0A |
| `pd.isna(x)` | tester une valeur isolée (jamais `x == np.nan`) | 0A |

## Grouper et agréger (groupby)

| Code | Effet | Ch. |
|---|---|---|
| `df.groupby("species")["body_mass_g"].mean()` | une moyenne par groupe (séparer, appliquer, combiner) | 0A |
| `df.groupby("island")["flipper_length_mm"].agg(["count", "mean", "max"])` | plusieurs statistiques à la fois | 0A |
| `df.groupby(["year", "species"])["body_mass_g"].mean().unstack()` | deux clés, puis un tableau croisé (une colonne par espèce) | 0A |
| `.idxmax()`, `.idxmin()` | **l'étiquette** du maximum, du minimum | 0A |
| `for name, group in df.groupby("species"):` | parcourir les groupes (un graphique par espèce, par exemple) | 0A |

## Statistiques (ch. 2)

| Code | Effet | Ch. |
|---|---|---|
| `s.mean()`, `s.median()`, `s.mode()` | moyenne, médiane, mode (`mode()` renvoie **toutes** les valeurs ex æquo, dans une Series) | 2 |
| `s.var()`, `s.std()` | variance et écart-type, divisés par `n - 1` par défaut (`ddof=1`) ; `ddof=0` pour faire comme NumPy | 2 |
| `s.quantile([0.25, 0.5, 0.75])` | quartiles (quantiles entre 0 et 1) | 2 |
| `df.cov()`, `df.corr()` | matrices de covariance (`ddof=1`) et de corrélation de Pearson des colonnes numériques | 2 |
| `df["body_mass_g"].hist(bins=20)` | histogramme rapide d'une colonne | 2 |
| `pd.plotting.scatter_matrix(df[cols])` | nuages de points de toutes les paires de colonnes (analyse exploratoire) | 2 |
| `df.sample(n=len(df), replace=True, random_state=0)` | un rééchantillon bootstrap des lignes | 2 |

## Tables de contingence (ch. 3)

| Code | Effet | Ch. |
|---|---|---|
| `pd.crosstab(df["sex"], df["species"])` | comptages croisés de deux colonnes catégorielles | 3 |
| `pd.crosstab(..., margins=True)` | ajoute les totaux (ligne et colonne `All`) : les marges | 3 |
| `pd.crosstab(..., normalize="all")` | probabilités jointes (divisées par le total) | 3 |
| `pd.crosstab(..., normalize="index")` | chaque ligne somme à 1 : P(colonne \| ligne) | 3 |
| `pd.crosstab(..., normalize="columns")` | chaque colonne somme à 1 : P(ligne \| colonne) | 3 |
| `pd.crosstab(y_true, y_pred, rownames=["vérité"], colnames=["prédiction"])` | une matrice de confusion lisible, avec les noms des axes | 3 |

## Joindre et remodeler

| Code | Effet | Ch. |
|---|---|---|
| `pd.concat([df1, df2], ignore_index=True)` | empiler des DataFrames (lignes à la suite) | 0A |
| `pd.get_dummies`, `merge`, `pivot_table` | *(ch. 12)* | |

