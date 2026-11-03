"""Parts A and B of the chapter B6 notebook, first half (B6.12 to B6.20): statements and cells (used by build_chB6.py).

Part A: the tabular models, global explanations, Shapley values with your library, shap, the background, Penguins
(B6.12 to B6.17). Part B: the networks, your integrated gradients and Grad-CAM, a small CIFAR-10 CNN, the ResNet of
chapter B1 (B6.18 to B6.20). The second half (B6.21 to B6.29) appends its exercises to PART_B.exercises (B6.21,
B6.22) and adds part C (fairness); it reuses TOOLS_B (CIFAR-10, CNN, show_rows) and RESNET, RESNET_INFO, SIZE_20 of
LOAD_20.
"""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, Part, guarded

import chB6_code as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB6.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}

P13 = HISTORY.get("p13", {}).get("answers", {"medinc_rise": "C", "lat_trend": "C", "lat_up": "A", "spread": True})

TEST = "tests/test_chb6_explain.py"


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** : ouvre `mon_travail/mylearn/explain.py` (créé par `python tools/start_chapter.py "
            "B6`), lis la docstring, remplace chaque `raise NotImplementedError(...)` de l'exercice par ton code et "
            "**enregistre**. NumPy seulement. La première cellule de vérification recharge ta librairie et l'essaie sur des "
            "cas visibles ; la seconde lance les tests de l'exercice. `python -m pytest tests/test_chb6_explain.py -q` "
            "les lance tous dans un terminal.")


def tests_after(ex_id: str, probe: str, keyword: str, n_tests: str, what: str = "tes fonctions") -> list:
    """The second check cell of a 🔨: the tests of the exercise (explicit test file, -k group)."""
    cell = (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_EXPLAIN, "{keyword}")   # {n_tests}\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent {what}.")')
    ref = cell.replace('run_pytest_group(TEST_EXPLAIN, "', 'run_pytest_group(TEST_EXPLAIN, impl="ref", keyword="')
    return [("check", cell), ("solution", ref)]


def letters_record(ex_id: str, answer: str, hints: dict, prediction: bool = False) -> str:
    """wb.record of a letter answer, with a message for each other letter; letters with the same message share it (a
    set of wrong values), so a message never has to name its letter (« Piste : « B » : … » read as the hint)."""
    grouped = {}
    for letter, message in hints.items():
        if letter != answer:
            grouped.setdefault(message, []).append(letter)
    value = lambda ls: f'"{ls[0]}"' if len(ls) == 1 else "{" + ", ".join(f'"{x}"' for x in ls) + "}"
    entries = ", ".join(f'"{message}": {value(ls)}' for message, ls in grouped.items())
    return f'wb.record("{ex_id}", "{answer}", mistakes={{{entries}}}' + (", prediction=True)" if prediction else ")")


# ---------------------------------------------------------------------------------------------------------------------
PART_A = Part("A", "Modèles tabulaires : explications globales, valeurs de Shapley et SHAP",
              "Tu expliques d'abord un gradient boosting de California **globalement** : coefficients d'une régression, "
              "importance par permutation, dépendance partielle et courbes ICE (B6.12), dont tu prédis la forme pour deux "
              "features (B6.13). Puis **localement** : tu écris le Shapley exact de ta librairie (B6.14), tu le compares à "
              "TreeSHAP sur un gradient boosting et tu lis les trois graphiques de shap (B6.15), tu mesures ce que changent "
              "le fond et une feature dupliquée (B6.16), et tu expliques un classifieur à trois classes de Penguins "
              "(B6.17). Au ch. 14, tu as entraîné des gradient boostings sur California et comparé `feature_importances_` "
              "à la permutation ; au ch. 15, tu as construit des pipelines : ici, tu ouvres la boîte.\n\n"
              "La cellule suivante charge les outils de la partie : `wbx` (`wb.explain_data`, l'infrastructure du "
              "chapitre), le découpage figé de California (`X_TRAIN`, `Y_TRAIN` : 12 384 districts ; `X_VAL`, `Y_VAL` : "
              "4 128 ; les mêmes à chaque exécution), `FEATURES` (les 8 noms de colonnes), `GBR`, le gradient boosting "
              "de référence (`HistGradientBoostingRegressor(random_state=0)` ajusté sur `X_TRAIN`), dont se servent les "
              "vérifications et les expériences données, `TEST_EXPLAIN` (le fichier des tests de `explain.py`) et "
              "`shap`. Les features de California y sont arrondies en float32 (B6.15 dit pourquoi).",
              given=C.TOOLS_A, exercises=[
    Ex("B6.12", "📦", 2, 30, "Explications globales sur California : coefficients, permutation, dépendance partielle et ICE",
       "comparer trois explications globales d'un même problème et savoir ce que chacune mesure.",
       "ch. 14 et 15 · fiche §206.2 (« Coefficients et arbres », « L'importance par permutation », « La dépendance "
       "partielle », « Les courbes ICE ») · 🔁 R2", tracks="R, C",
       body=r"""**Contrat** des cinq fonctions de la cellule suivante :
- `linear_12(X, y)` rend un `Pipeline` `StandardScaler` → `LinearRegression`, ajusté sur `(X, y)` ;
- `coefs_12(pipeline)` rend une `pd.Series` des 8 coefficients de la `LinearRegression` du pipeline, indexée par les features, dans l'ordre de `FEATURES` : l'effet d'**un écart-type** de chaque feature, les autres fixées ;
- `boosting_12(X, y)` rend un `HistGradientBoostingRegressor(random_state=0)`, tous les autres réglages par défaut, ajusté sur `(X, y)` ;
- `permutation_12(model, X, y)` rend une `pd.Series` : la baisse moyenne du R² de `permutation_importance(model, X, y, n_repeats=5, random_state=0)`, indexée par les colonnes de `X`, triée de la plus grande à la plus petite ;
- `pd_ice_12(model, X, feature)` rend le dictionnaire `{"grid": (50,), "average": (50,), "individual": (len(X), 50)}` de `partial_dependence(model, X, [feature], kind="both", grid_resolution=50)` (la grille va du 5ᵉ au 95ᵉ centile de la feature).

La cellule de vérification appelle `linear_12` et `boosting_12` sur `X_TRAIN`, `permutation_12` sur le gradient boosting de référence `GBR` et la **validation**, `pd_ice_12` sur `GBR`, 300 districts de validation et `AveOccup`, et compare chaque résultat à scikit-learn. Une fois les trois résultats justes, elle trace les coefficients en valeur absolue, les importances, puis les courbes ICE de `AveOccup` et de `HouseAge` avec leur moyenne. Rien à taper.

Dans tes notes : range les features par |coefficient| et par importance : lesquelles changent de rang, et pourquoi une régression linéaire et un gradient boosting peuvent-ils ne pas s'appuyer sur les mêmes features ? Pourquoi la permutation se calcule-t-elle sur la validation, et que perdrait-on à la calculer sur l'entraînement (🔁 R2) ? Sur les courbes ICE de `AveOccup`, où la dépendance partielle résume-t-elle bien les courbes, et où les cache-t-elle ? Que deviennent ces courbes au-delà de la grille, là où il y a peu de districts (piège « une dépendance partielle hors des données ») ?""",
       todo=C.TODO_12, check=C.CHECK_12, solution=C.SOLUTION_12 + "\n\n\n" + C.CHECK_12),

    Ex("B6.13", "🔮", 1, 10, "Prédire la forme des courbes ICE du revenu médian et de la latitude",
       "prédire ce que les courbes ICE d'un gradient boosting montrent de deux features, avant de les tracer.",
       "Ex B6.12 · fiche §206.2 (« La dépendance partielle », « Les courbes ICE »)", tracks="C", hypothesis=True,
       body=r"""L'expérience trace, pour le gradient boosting de référence `GBR`, les courbes ICE de `MedInc` (le revenu médian du district, en dizaines de milliers de dollars) et de `Latitude` (de ≈ 33° au sud, vers San Diego, à ≈ 39° au nord de la grille) sur 500 districts de validation, avec leur moyenne, la dépendance partielle ; chaque grille a 50 valeurs, du 5ᵉ au 95ᵉ centile de la feature. La prédiction est en `MedHouseVal`, en centaines de milliers de dollars. **Avant d'exécuter**, écris tes quatre prédictions :
- a) `medinc_rise_13` : de combien la dépendance partielle de `MedInc` monte-t-elle du début à la fin de sa grille ? `"A"` moins de 0,5 ; `"B"` de 0,5 à 1,5 ; `"C"` de 1,5 à 3 ; `"D"` plus de 3 ;
- b) `lat_trend_13` : du sud au nord, la dépendance partielle de `Latitude` : `"A"` monte (de plus de 0,3) ; `"B"` reste à peu près plate (entre −0,3 et +0,3) ; `"C"` descend (de plus de 0,3) ;
- c) `lat_up_13` : quelle part des 500 courbes ICE de `Latitude` **finissent plus haut** qu'elles ne commencent ? `"A"` moins de 5 % ; `"B"` de 5 à 30 % ; `"C"` de 30 à 70 % ; `"D"` plus de 70 % ;
- d) `spread_13` : les courbes ICE de `Latitude` varient-elles **plus** d'un district à l'autre que celles de `MedInc` (un écart-type plus grand de leur variation, fin moins début) ? `True` ou `False`.

La cellule d'expérience n'affiche rien tant que tes quatre prédictions ne sont pas écrites ; elle les vérifie à la fin. Rien n'y dépend de ton code.

Dans tes notes, avant d'exécuter : ce que tu sais de la Californie (où sont les villes, la côte, les grandes régions) et ce que tu attends d'un modèle qui voit la latitude **et** la longitude.""",
       given=C.GIVEN_13, todo=C.TODO_13,
       solution=(f'medinc_rise_13, lat_trend_13, lat_up_13, spread_13 = "{P13["medinc_rise"]}", "{P13["lat_trend"]}", '
                 f'"{P13["lat_up"]}", {P13["spread"]}'),
       record="\n".join([
           letters_record("B6.13a", P13["medinc_rise"], {
               "A": "lis la variation de la dépendance partielle de MedInc, sous la figure",
               "B": "lis la variation de la dépendance partielle de MedInc, sous la figure",
               "C": "lis la variation de la dépendance partielle de MedInc, sous la figure",
               "D": "lis la variation de la dépendance partielle de MedInc, sous la figure"}, prediction=True),
           letters_record("B6.13b", P13["lat_trend"], {
               "A": "suis la courbe épaisse de la latitude, du sud (à gauche) au nord (à droite)",
               "B": "lis la variation de la dépendance partielle de la latitude, sous la figure",
               "C": "suis la courbe épaisse de la latitude, du sud (à gauche) au nord (à droite)"}, prediction=True),
           letters_record("B6.13c", P13["lat_up"], {
               "A": "lis la part des courbes ICE de la latitude qui finissent plus haut, sous la figure",
               "B": "lis la part des courbes ICE de la latitude qui finissent plus haut, sous la figure",
               "C": "lis la part des courbes ICE de la latitude qui finissent plus haut, sous la figure",
               "D": "lis la part des courbes ICE de la latitude qui finissent plus haut, sous la figure"},
               prediction=True),
           f'wb.record("B6.13d", {P13["spread"]}, mistakes={{"compare les deux écarts-types de la variation des ICE, '
           f'sous la figure": {not P13["spread"]}}}, prediction=True)']),
       after=[("code", guarded(C.EXPERIMENT_13, ["medinc_rise_13", "lat_trend_13", "lat_up_13", "spread_13"],
                               "⏳ Ex B6.13 : écris d'abord tes quatre prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Que te disent les courbes "
                     "ICE de la latitude que la dépendance partielle seule ne dit pas, et avec quelle autre feature la "
                     "latitude interagit-elle sans doute ? Imposer une latitude à un district sans toucher à sa "
                     "longitude : quelles combinaisons crée-t-on, et ces districts existent-ils (piège « une dépendance "
                     "partielle hors des données ») ? Une courbe ICE dit-elle ce que **ferait** un déménagement du "
                     "district vers le nord ?")]),

    Ex("B6.14", "🔨", 3, 45, "shapley_values : Shapley exact par énumération des coalitions",
       "écrire le calcul exact des valeurs de Shapley interventionnelles, que shap approche.",
       "✏️ B6.1 · ∂ B6.2 · fiche §206.3 (« Les features comme joueurs », « Un exemple à deux features », « Combien ça "
       "coûte »)", tracks="R, M, C", mylearn="explain.py",
       body=mylearn_howto() + r"""

**Contrat** de `shapley_values(f, x, background, max_features=12)`, qui rend le couple `(phi, base_value)` :
- `f` ne reçoit que des tableaux 2-D de forme `(n, M)`, jamais une ligne seule, et rend une sortie de forme `(n,)`, par exemple `model.predict(X)` ou `model.predict_proba(X)[:, 1]` ; elle peut être appelée plusieurs fois ;
- $v(S)$ est la **moyenne de $f$** sur les $B$ lignes du fond dont les colonnes de $S$ prennent les valeurs de `x` ($f$ d'abord, la moyenne ensuite) ; **toutes** les lignes du fond comptent, sans sous-échantillonnage (contrairement au `max_samples=100` par défaut de shap) ;
- `phi` est un tableau de forme `(M,)` ; `base_value` vaut $v(\varnothing)$, la moyenne de $f$ sur le fond, un `float` Python ; l'efficacité `phi.sum() + base_value == f(x)` tient aux arrondis près ;
- `ValueError` si $M >$ `max_features` ($M =$ `max_features` est permis), si `x` n'est pas 1-D, ou si `background` n'est pas 2-D à $M$ colonnes ; `x` et `background` peuvent être des listes et ne sont jamais modifiés ;
- **coût** : la docstring prescrit l'énumération des $2^M$ coalitions, chaque $v(S)$ calculé **une seule fois** (un appel à `f` sur les $B$ lignes) ; les tests comptent les lignes passées à `f` et arrêtent ta fonction au-delà de `max(2·M·2**M·B, 500 000)` lignes, avec un message qui le dit. Ce budget laisse passer un $v(S)$ recalculé pour chaque feature, ou `f` appelée ligne par ligne ; il arrête une énumération des $M!$ ordres d'arrivée sans mémoire des $v(S)$ (40 320 ordres à $M = 8$), qui prendrait des minutes.

La première cellule de vérification regarde d'abord ce que reçoit `f`, puis essaie ta fonction sur les deux exemples de la docstring, sur un modèle linéaire à 5 features et un fond aléatoire de 20 lignes ($\phi_i = w_i(x_i - \bar b_i)$, ∂ B6.2), sur le gradient boosting de référence de California (8 features, fond de 50 lignes : efficacité, valeur de base, comparaison à `shap.explainers.Exact`, durée affichée), et sur quatre gardes. La seconde lance les 33 tests de l'exercice (`-k "test_shapley_values_"`).

Dans tes notes : combien de lignes ta fonction passe-t-elle à `f` pour $M = 8$ et $B = 50$, et combien en passerait une version qui recalcule chaque $v(S)$ pour chaque feature ? Pourquoi `f` doit-elle recevoir des tableaux 2-D (pense au coût d'un appel à `predict`) ? Que deviendrait la durée affichée avec 12 features (🧮 B6.7) ?""",
       check=C.RELOAD + C.CHECK_14, solution=C.CHECK_14,
       after=tests_after("B6.14", "mylearn.explain.shapley_values(lambda X: X.sum(axis=1), [0.0], [[0.0]])",
                         "test_shapley_values_", "33 tests", "ta fonction shapley_values")),

    Ex("B6.15", "📦", 2, 30, "SHAP sur un gradient boosting California : TreeExplainer, waterfall, beeswarm, dépendance",
       "expliquer un gradient boosting avec TreeSHAP, vérifier qu'il calcule le même Shapley que ta fonction, et lire "
       "les trois graphiques de shap.",
       "Ex B6.14 · ch. 14 · fiche §206.4 (« L'API de shap », « Les trois graphiques à connaître »)", tracks="R, C",
       body=r"""**Contrat** des deux fonctions de la cellule suivante :
- `fit_15(X, y)` rend un `HistGradientBoostingRegressor(max_depth=3, random_state=0)` (des arbres peu profonds, les autres réglages par défaut), ajusté sur `(X, y)` ;
- `tree_shap_15(model, background, rows)` rend le `shap.Explanation` de `shap.TreeExplainer(model, data=background, feature_perturbation="interventional")` appliqué au DataFrame `rows` : `.values` de forme `(len(rows), 8)`, `.base_values`, `.feature_names`.

La cellule donnée fixe `BG_15`, le fond (les 50 premiers districts d'entraînement), et `ROWS_15`, les 1 000 premiers districts de validation à expliquer. La cellule de vérification ajuste ton modèle sur `X_TRAIN`, l'explique, contrôle l'efficacité sur les 1 000 districts, puis vérifie quatre grandeurs que tes fonctions calculent (rien à taper, aucune à arrondir) : a) la valeur de base ; b) la valeur SHAP de `MedInc` pour le premier district ; c) la moyenne des $\lvert\phi_{\text{MedInc}}\rvert$ sur les 1 000 districts ; d) si ta `shapley_values` (B6.14), avec le même fond de 50 lignes, redonne les valeurs de TreeSHAP sur le premier district à $10^{-6}$ près (`True` ou `False`). La cellule suivante trace le waterfall du premier district, le beeswarm et la dépendance de `MedInc`.

> ⚠️ **Piège classique — TreeSHAP et les données en float64** — `TreeExplainer` parcourt les arbres en **float32**. Un `HistGradientBoosting` pose certains seuils exactement sur des valeurs répétées des données (la latitude et la longitude de California en ont beaucoup) : une valeur égale à un seuil peut alors passer de l'autre côté, et TreeSHAP perd l'efficacité (des écarts jusqu'à 0,5 sur les données d'origine de scikit-learn, mesurés avec shap 0.54.1). C'est pourquoi `wbx.california()` arrondit les features en float32 : TreeSHAP interventionnel y redonne le Shapley exact à $10^{-8}$ près. Sur tes propres données, vérifie toujours l'efficacité (shap le fait : `check_additivity`).

Dans tes notes : pourquoi la valeur de base a) n'est-elle ni la moyenne des cibles d'entraînement, ni la prédiction moyenne des 1 000 districts expliqués ? Sur le waterfall, quelles features font monter et descendre la prédiction du premier district, et de combien ? Sur le beeswarm, dans quel sens joue un grand revenu, et quelles features ont une grande dispersion ? Sur la dépendance de `MedInc`, qu'indique la dispersion verticale à revenu égal ? Combien de temps a pris TreeSHAP sur 1 000 districts, et combien prendrait ta `shapley_values` (sa durée pour un district est affichée en B6.14) ?""",
       given=C.GIVEN_15, todo=C.TODO_15, check=C.CHECK_15, solution=C.SOLUTION_15 + "\n\n\n" + C.CHECK_15,
       record=C.RECORD_15, after=[("code", C.PLOTS_15)]),

    Ex("B6.16", "🔬", 3, 40, "Le fond change tout : trois fonds différents, puis une feature dupliquée",
       "mesurer ce que le choix du fond et une feature dupliquée font aux valeurs de Shapley d'un gradient boosting.",
       "Ex B6.15 · fiche §206.3 (« Le fond fait partie de la question ») et §206.4 (« Features corrélées : ce que SHAP "
       "répartit ») · 🧠 Q5, Q6", tracks="C",
       body=r"""Deux expériences sur le gradient boosting de référence `GBR`. **Ne présume pas l'issue : mesure-la.**

**1. Trois fonds.** La cellule donnée choisit `X_16`, le district de validation le plus proche du 75ᵉ centile du revenu, et trois fonds (`BACKGROUNDS_16`) : `"population"`, 50 districts d'entraînement tirés au hasard ; `"riches"`, 50 districts du décile de revenu le plus haut ; `"ligne type"`, **une** ligne, la médiane de chaque feature. Contrat de `explain_16(model, x, background)` : il rend le couple `(phi, base)`, les valeurs de Shapley interventionnelles de `model.predict` au district `x` (une `pd.Series` des 8 features) avec le fond `background` (un DataFrame), un tableau `(8,)`, et la valeur de base, un `float` ; ta `shapley_values` ou `shap.TreeExplainer` (les mêmes nombres, B6.15). La cellule de vérification compare tes valeurs à TreeSHAP pour les trois fonds, puis affiche le tableau des valeurs de base et des $\phi$.

**2. Une feature dupliquée.** `add_copy_16(X, noise, rng)` (donnée) ajoute à un DataFrame, juste après `MedInc`, la colonne `"MedInc_copy"` = `MedInc` + `noise` × (écart-type de `MedInc`) × un bruit gaussien tiré par `rng`. Contrat de `copy_experiment_16(noise, seed, n_rows=200)` :
- `rng = np.random.default_rng(seed)`, puis `A = add_copy_16(X_TRAIN, noise, rng)` et `V = add_copy_16(X_VAL, noise, rng)`, dans cet ordre, avec le même `rng` ;
- un `HistGradientBoostingRegressor(random_state=seed)` ajusté sur `(A, Y_TRAIN)` ;
- `shap.TreeExplainer` interventionnel, de fond les 50 premières lignes de `A`, sur les `n_rows` premières lignes de `V` ;
- il rend `{"MedInc": moyenne des |φ_MedInc|, "MedInc_copy": moyenne des |φ_copie|, "sum": moyenne des |φ_MedInc + φ_copie|}` (des floats).

La cellule d'expérience essaie d'abord ta fonction sur un petit cas (bruit 0,01, graine 0, 50 districts), puis la lance pour trois niveaux de bruit (0, 0,01 et 0,1 écart-type) et trois graines (≈ 10 s), à côté de la moyenne des $\lvert\phi_{\text{MedInc}}\rvert$ du modèle sans copie.

Dans tes notes : avec quel fond la valeur SHAP de `MedInc` change-t-elle de signe, et pourquoi ? Que change le fond « ligne type » par rapport à « population » ? Quelle phrase écrirais-tu sous un waterfall pour dire quel fond a servi ? Pour la copie : que reçoit-elle quand elle est exacte (bruit 0), et pourquoi ? Avec un bruit de 0,01, le partage entre `MedInc` et sa copie est-il stable d'une graine à l'autre ? Et la somme ? Que rapporterais-tu à un métier qui demande « quelle est l'importance du revenu ? » quand deux colonnes le portent ?""",
       given=C.GIVEN_16, todo=C.TODO_16, check=C.CHECK_16, solution=C.SOLUTION_16 + "\n\n\n" + C.CHECK_16,
       after=[("code", C.EXPERIMENT_16)]),

    Ex("B6.17", "📦", 2, 25, "Penguins : SHAP par classe pour une régression logistique et une forêt aléatoire",
       "expliquer un classifieur à trois classes, une sortie à la fois, en logit puis en probabilité.",
       "Ex B6.15 · ∂ B6.2 · fiche §206.4 (« L'API de shap ») · ch. 13 et 14", thread="Penguins", tracks="C",
       body=r"""Pourquoi ce manchot est-il un Gentoo ? La cellule donnée charge le découpage figé de Penguins (`wbx.penguins()` : 249 manchots d'entraînement, 84 de test, les 4 mesures en colonnes, l'espèce en label) et `GENTOO_17`, le premier Gentoo du test ; au ch. 0A tu as découvert ce dataset, au ch. 13 tu l'as classé. Un classifieur multiclasse a **une sortie par classe** : on explique une sortie à la fois, la probabilité de la classe $k$ ou son logit, et ces deux sorties ne donnent pas les mêmes valeurs (206.4). **Contrat** :
- `fit_logistic_17(X, y)` rend `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))` ajusté ; `fit_forest_17(X, y)`, `RandomForestClassifier(n_estimators=200, random_state=0)` ajusté ;
- `class_output_17(model, k, kind="proba")` rend la fonction `f(X) -> (n,)` d'**une** sortie : la probabilité de la classe d'indice `k` (`predict_proba`) pour `kind="proba"`, son logit (`decision_function`) pour `kind="logit"` ;
- `shap_by_class_17(model, x, background, kind="proba")` rend le couple `(phi, base)` : `phi` de forme `(3, 4)`, une ligne par classe dans l'ordre de `model.classes_` (`SPECIES`), calculée par ta `shapley_values` sur la sortie de cette classe, et `base` de forme `(3,)`.

La cellule de vérification compare tes deux modèles à scikit-learn, puis explique `GENTOO_17` avec les modèles de référence et tout l'entraînement pour fond : en logit pour la régression logistique, elle compare tes valeurs au cas linéaire ($\phi_{ki} = w_{ki}(x_i - \bar b_i)$ dans les unités des mesures, ∂ B6.2) ; en probabilité, elle vérifie l'efficacité de chaque classe et les valeurs de base ; pour la forêt, elle compare tes valeurs à `shap.TreeExplainer` interventionnel, avec tout le fond (`shap.maskers.Independent(XP_TRAIN, max_samples=249)` : sinon il n'en garde que 100), à $2 \cdot 10^{-3}$ près, parce que sur une forêt TreeSHAP s'écarte lui-même de `predict_proba` de quelques $10^{-4}$. Puis elle trace les trois explications et affiche, pour chacune, la somme sur les trois classes de chaque feature.

Dans tes notes : quelles mesures font du manchot un Gentoo selon chaque explication ? Les logits et les probabilités classent-ils les mesures dans le même ordre ? Pourquoi les sommes sur les classes affichées valent-elles 0 pour les probabilités (pense à l'additivité et à ce que vaut la somme des trois probabilités) ? Elles valent 0 aussi pour les logits : que dit cela des coefficients de la régression logistique multinomiale de scikit-learn (affiche `LR_17[-1].coef_.sum(axis=0)`) ? La forêt et la régression logistique s'appuient-elles sur les mêmes mesures ?""",
       given=C.GIVEN_17, todo=C.TODO_17, check=C.CHECK_17, solution=C.SOLUTION_17 + "\n\n\n" + C.CHECK_17),
])

# ---------------------------------------------------------------------------------------------------------------------
PART_B = Part("B", "Réseaux : gradients intégrés et Grad-CAM",
              "Tu écris d'abord en NumPy les deux méthodes de ta librairie qui ne dépendent pas du réseau : les gradients "
              "intégrés à partir d'une fonction de gradient, et la combinaison Grad-CAM à partir d'activations et de "
              "gradients déjà extraits (B6.18). Puis tu les branches sur des réseaux : saillance, gradient × entrée et "
              "gradients intégrés sur un petit CNN CIFAR-10 (B6.19), Grad-CAM par hooks sur le ResNet fine-tuné du B1, "
              "pour des bonnes prédictions et des erreurs (B6.20). Au ch. 21, FGSM et la maximisation d'activation "
              "dérivaient déjà par rapport à l'entrée (🔁 R1) ; au ch. 20 et au B1, tu as posé des hooks.\n\n"
              "La cellule suivante charge 500 images de test de CIFAR-10 (`X_CIFAR`, `Y_CIFAR` : 50 par classe, en "
              "uint8) et `CNN`, le petit CNN de référence de B6.19 : trois blocs de deux convolutions, entraîné une fois "
              "par `wbx.cifar_cnn` (≈ 40 s sur CPU la première fois, puis relu du disque), **gelé** et en mode `eval()` ; "
              "il reçoit des pixels dans $[0, 1]$, de forme `(N, 3, 32, 32)`, et se normalise lui-même : l'image noire "
              "est le tenseur nul. `show_rows` affiche une grille d'images et de cartes.",
              given=C.TOOLS_B, exercises=[
    Ex("B6.18", "🔨", 2, 30, "integrated_gradients et grad_cam",
       "écrire les gradients intégrés à partir d'une fonction de gradient, et la combinaison Grad-CAM de cartes et de "
       "gradients déjà extraits.",
       "✏️ B6.3, B6.4 · fiche §206.5 (« Les gradients intégrés », « En pratique : une somme à la place de l'intégrale ») "
       "et §206.6 (« Un exemple à la main », « Avec PyTorch : deux hooks »)", tracks="R, M, C", mylearn="explain.py",
       body=mylearn_howto() + r"""

**Contrat** de `integrated_gradients(grad_fn, x, baseline=None, n_steps=50, method="trapezoid")` :
- les points du chemin sont $\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}')$, $\mathbf{x}'$ la ligne de référence `baseline` (des zéros si `None`) ; la moyenne pondérée des gradients est multipliée par $\mathbf{x} - \mathbf{x}'$, et le résultat a la forme de `x`, en float, même pour un `x` entier ;
- `n_steps` compte les **intervalles**, un entier ≥ 1 : `'left'`, `'right'` et `'midpoint'` utilisent $n$ points de poids $1/n$, `'trapezoid'` les $n + 1$ points $k/n$ de poids $1/n$, divisés par deux aux extrémités (la docstring) ;
- `grad_fn` ne reçoit que des **batches** de forme `(n, *x.shape)`, même avec `n_steps = 1` (pour un seul point : `point[None]`, puis `[0]` sur le résultat) ; elle peut être appelée une fois ou sur plusieurs morceaux ;
- `ValueError` si `n_steps < 1`, si `method` est inconnue (les noms de captum, comme `'riemann_trapezoid'`, sont refusés), ou si `baseline` n'a pas **exactement** la forme de `x` (une forme seulement diffusable est refusée) ; `x` et `baseline` ne sont pas modifiés ;
- ⚠️ la note de la docstring sur captum est inexacte : son `riemann_trapezoid` à $n + 1$ points vaut ton `'trapezoid'` **multiplié par $n/(n+1)$**, parce que ses poids somment à moins de 1 (fiche, 🕰️ de 206.6 ; ✏️ B6.3 k) ; `'left'`, `'right'` et `'midpoint'` coïncident avec `riemann_left`, `riemann_right` et `riemann_middle` au même nombre de points.

**Contrat** de `grad_cam(activations, gradients, relu=True, normalize=True)` :
- les deux entrées ont exactement la même forme 4-D `(N, C, H, W)`, sinon `ValueError` ; le résultat a la forme `(N, H, W)` ;
- le poids d'un canal est la **moyenne** spatiale de ses gradients ; la carte est la somme pondérée des canaux, la ReLU appliquée **une fois**, à cette somme (`relu=True`) ;
- `normalize=True` divise **chaque** carte par sa propre plus grande valeur absolue, sans epsilon ; une carte nulle, y compris après la ReLU, reste nulle (pas de NaN) ;
- les entrées ne sont pas modifiées.

La première cellule de vérification essaie `integrated_gradients` sur les exemples de la docstring, sur $x_0 x_1 + x_2^3$ en $(1, 2, -1)$ avec les quatre règles et 6 intervalles (formes fermées), sur ce que reçoit `grad_fn`, sur la complétude de la sigmoïde de 206.5, un `x` entier et trois refus ; la seconde partie essaie `grad_cam` sur la docstring, sur trois images de 4 canaux (contre `np.einsum`), l'invariance à un facteur positif des gradients, `relu=False`, des gradients nuls et deux refus. La seconde cellule lance les 111 tests de l'exercice (`-k "test_integrated_gradients_ or test_grad_cam_"`) ; sans captum, 12 sont ignorés.

Dans tes notes : avec 6 intervalles, quelle règle donne la valeur exacte pour $x_0 x_1$, et pourquoi pas pour $x_2^3$ ? Pourquoi la complétude sert-elle de test à une implémentation ? Pourquoi `grad_cam` ne fait-elle que la combinaison, et pas l'extraction (qui reste côté PyTorch) ?""",
       check=C.RELOAD + C.CHECK_18, solution=C.CHECK_18,
       after=tests_after("B6.18", "(mylearn.explain.integrated_gradients(lambda X: X, [1.0], n_steps=1), "
                                  "mylearn.explain.grad_cam(np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1))))",
                         "test_integrated_gradients_ or test_grad_cam_", "111 tests, dont 12 ignorés sans captum")),

    Ex("B6.19", "📦", 2, 30, "Saillance, gradient × entrée et gradients intégrés sur un CNN CIFAR-10",
       "calculer trois attributions pixel à pixel sur un CNN, vérifier la complétude et comparer deux lignes de "
       "référence.",
       "Ex B6.18 · ch. 24 · fiche §206.5 (« Le gradient comme explication », « Les gradients intégrés », « Le choix de "
       "la ligne de référence »)", thread="CIFAR-10", tracks="C",
       body=r"""La cellule donnée prend 8 images de test (`IMAGES_19`, un tenseur `(8, 3, 32, 32)` dans $[0, 1]$), leurs classes (`LABELS_19`), les classes que prédit le CNN (`PRED_19`) et `blurred_19`, qui floute une image (la moyenne sur une fenêtre de 7 × 7 pixels). **Contrat** :
- `class_gradient_19(model, images, target)` rend un tableau NumPy `(N, 3, 32, 32)` : le gradient du **logit** (le score avant la softmax) de la classe `target[n]` de chaque image par rapport à ses pixels ; `images` est un tenseur ou un tableau `(N, 3, 32, 32)`, `target` une classe par image, pas forcément la prédite ; le modèle est gelé et en mode `eval()`, et doit le rester ; deux appels successifs rendent le même gradient ;
- `saliency_19(gradients)` rend la carte de saillance `(N, 32, 32)` : le plus grand $\lvert$gradient$\rvert$ des trois canaux de chaque pixel ;
- `grad_times_input_19(gradients, images)` rend `(N, 32, 32)` : gradient × entrée, sommé sur les trois canaux de chaque pixel, avec son signe ;
- `integrated_19(model, image, target, baseline, n_steps=50)` rend `(3, 32, 32)` : les gradients intégrés du logit de la classe `target` (un entier) pour **une** image `(3, 32, 32)`, depuis la ligne de référence `baseline` `(3, 32, 32)`, calculés par **ta** `integrated_gradients` (règle `'trapezoid'`) avec une `grad_fn` construite sur `class_gradient_19`.

La cellule de vérification compare chaque fonction à une référence (l'autograd de `wbx`) sur quatre images, avec des classes que le CNN ne prédit pas. L'expérience trace, pour 6 images et la vraie classe, la saillance, $\lvert$gradient × entrée$\rvert$, et $\lvert$IG$\rvert$ depuis l'image noire et depuis l'image floutée (la somme des valeurs absolues des trois canaux), puis compare la somme des attributions à $f(\mathbf{x}) - f(\mathbf{x}')$.

Dans tes notes : laquelle des trois cartes est la plus bruitée, et laquelle se concentre le plus sur l'objet ? La complétude tient-elle pour les gradients intégrés, et pour gradient × entrée ? Avec 50 intervalles, à combien près ? Que change la ligne de référence floutée : que vaut $f(\mathbf{x}')$, quels pixels reçoivent une attribution presque nulle, et pourquoi (le facteur $x_i - x'_i$) ? Quelle référence choisirais-tu pour un objet noir sur fond clair ?""",
       given=C.GIVEN_19, todo=C.TODO_19, check=C.CHECK_19, solution=C.SOLUTION_19 + "\n\n\n" + C.CHECK_19,
       after=[("code", C.EXPERIMENT_19)]),

    Ex("B6.20", "📦", 3, 45, "Grad-CAM par hooks sur le ResNet fine-tuné du B1 : bonnes prédictions et erreurs",
       "extraire activations et gradients par deux hooks, construire les cartes Grad-CAM d'un ResNet et les lire sur "
       "des bonnes prédictions et des erreurs.",
       "Ex B6.18 · B1 (B1.17, B1.27) · fiche §206.6 (« Avec PyTorch : deux hooks ») · ch. 20 (hooks)", thread="CIFAR-10",
       tracks="R, C",
       body=r"""Le modèle est **ton** ResNet-18 du B1 si tu l'as publié en B1.27 : `wbx.find_b1_folder()` cherche le dossier `resnet18_cifar10_b1` (avec `model.safetensors` et `config.json`) à côté de tes notebooks de `mon_travail/` ; sinon, ou si tu mets `B1_FOLDER = None`, c'est un ResNet-18 de référence, fine-tuné une fois par la recette de B1.17 (`layer4` et une nouvelle tête, 100 images par classe à 96 pixels, 3 epochs : ≈ 35 s sur CPU la première fois, puis relu du disque). La cellule de chargement dit lequel est utilisé. Le modèle est **gelé** (`requires_grad=False` partout) et en mode `eval()`, comme un backbone du B1 ; il attend des images préparées par `wbx.model_input` (agrandies à `SIZE_20` pixels, normalisées par les statistiques d'ImageNet), et `layer4` y est en `SIZE_20 // 32` × `SIZE_20 // 32` (3 × 3 à 96 pixels). La cellule de chargement choisit aussi, parmi 200 images de test, 8 bonnes prédictions et 8 erreurs (`X_20`, 16 images préparées).

**Contrat** :
- `activations_and_gradients_20(model, layer, images, target)` rend le couple `(A, G)` de tableaux NumPy `(N, C, H, W)` : `A`, la sortie du module `layer` pour le batch préparé `images`, et `G`, le gradient de la somme, sur les images, du logit de `target[n]` par rapport à `A` (une classe par image, pas forcément la prédite) ; deux hooks, l'un sur le module (passe avant), l'autre sur son tenseur de sortie ; le hook de passe avant ne rend rien, et il est retiré avant de rendre la main, même en cas d'erreur ;
- `cam_20(model, layer, images, target, size)` rend `(N, size, size)` : les cartes de **ta** `grad_cam` (ReLU et normalisation par défaut), agrandies par `F.interpolate(..., mode="bilinear", align_corners=False)`, la convention qui a agrandi les images.

La cellule de vérification compare `A`, `G` et les cartes à une référence (`wbx`) sur 4 images, dont une avec une autre classe que la vraie, et vérifie qu'aucun hook ne reste posé. L'expérience trace la carte de la classe prédite sur les 8 bonnes prédictions, puis, sur les 8 erreurs, celle de la classe prédite et celle de la vraie classe, et affiche la corrélation entre ces deux cartes.

Dans tes notes : sur les bonnes prédictions, la carte tombe-t-elle sur l'objet, sur le fond, sur un bord ? Sur les erreurs, les cartes de la classe prédite et de la vraie classe regardent-elles au même endroit, et qu'est-ce que cela suggère de l'erreur (un objet ambigu, un contexte trompeur) ? Que peut-on localiser avec une carte de 3 × 3 sur une image de 96 pixels ? Que verrais-tu avec `layer3`, et pourquoi Grad-CAM vise-t-il la dernière couche convolutive (🧠 Q8) ? Une carte qui « tombe sur l'objet » prouve-t-elle que le modèle a raison pour de bonnes raisons (206.7) ?""",
       todo=C.TODO_20, solution=C.SOLUTION_20,
       after=[("code", C.LOAD_20), ("code", C.CHECK_20), ("code", C.EXPERIMENT_20)]),
])

PARTS = [PART_A, PART_B]
