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

## Joindre et remodeler

| Code | Effet | Ch. |
|---|---|---|
| `pd.concat([df1, df2], ignore_index=True)` | empiler des DataFrames (lignes à la suite) | 0A |
| `pd.get_dummies`, `merge`, `pivot_table` | *(ch. 2 et 12)* | |

