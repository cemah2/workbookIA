"""Code cells of the chapter B6 notebook, first half (part A, tabular models: B6.12 to B6.17; part B, networks: B6.18
to B6.20), shared by build_chB6.py and history_chB6.py (used by Claude).

Each constant is the source of one cell: TOOLS_* (both notebooks), GIVEN_* (both), TODO_* (exercise notebook),
SOLUTION_* (solutions notebook), CHECK_* and EXPERIMENT_* (both, after the learner's cell), RECORD_* (solutions).
The second half (B6.21 to B6.29) reuses TOOLS_B (CIFAR-10, the small CNN, the hooks) and the ResNet loaded by
LOAD_20 (RESNET, RESNET_INFO).
"""

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="B6")   # reload your saved file\n'

# =====================================================================================================================
# Part A · tabular models
# =====================================================================================================================
TOOLS_A = r'''# Tools of part A: California and Penguins (frozen splits), the reference model of the checks, your library explain.py
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from wb import explain_data as wbx
from wb.nbtools import (StopCheck, as_numpy, checked, filled, fr, pair, refuses, returned, run_pytest_group, same,
                        verdict, written)

warnings.filterwarnings("ignore", message=".*IProgress not found.*")              # tqdm, imported by shap
warnings.filterwarnings("ignore", message="X does not have valid feature names")   # a model fitted on a DataFrame,
                                                                                   # called on an array (shapley_values)
try:
    import shap
except ImportError:                       # shap is on Colab and in requirements.txt; wb.ensure("shap") installs it
    shap = None
    print("⏳ shap n'est pas installé : les cellules qui s'en servent le diront (wb.ensure('shap') l'installe).")

TEST_EXPLAIN = "tests/test_chb6_explain.py"            # the tests of your library explain.py
CAL = wbx.california()                                 # the frozen split of California (features rounded to float32)
FEATURES = CAL.features
X_TRAIN, Y_TRAIN, X_VAL, Y_VAL = CAL.X_train, CAL.y_train, CAL.X_val, CAL.y_val
GBR = HistGradientBoostingRegressor(random_state=0).fit(X_TRAIN, Y_TRAIN)   # the reference gradient boosting


def pct(x, digits=0):
    """A share written the French way: 0.734 -> '73 %'."""
    return f"{100 * x:.{digits}f} %".replace(".", ",")


def shapley_ready(ex_id):
    """True when your shapley_values (Ex B6.14) runs on a tiny case where f only accepts 2-D arrays (one feature,
    a background of one row); otherwise one ⏳ or ❌ line that sends you back to B6.14."""
    tiny = lambda: mylearn.explain.shapley_values(lambda X: np.asarray(X, dtype=float)[:, 0], [1.0], [[0.0]])
    if not written(tiny):
        print(f"⏳ Ex {ex_id} : cette vérification attend ta shapley_values (Ex B6.14).")
        return False
    try:
        tiny()
    except Exception as exc:  # noqa: BLE001 - the learner's code
        print(f"❌ Ex {ex_id} : ta shapley_values (Ex B6.14) lève {type(exc).__name__} sur un cas minuscule (une "
              "feature, un fond d'une ligne, une f qui n'accepte que des tableaux 2-D) : fais d'abord passer ses tests.")
        return False
    return True


print(f"California : {len(X_TRAIN)} districts d'entraînement, {len(X_VAL)} de validation, {len(CAL.X_test)} de test, "
      f"{len(FEATURES)} features ({', '.join(FEATURES)}) ; cible MedHouseVal (centaines de milliers de dollars). "
      f"Gradient boosting de référence : R² de validation {fr(GBR.score(X_VAL, Y_VAL))}.")'''

# --- 📦 B6.12 ---------------------------------------------------------------------------------------------------------
TODO_12 = r'''def linear_12(X, y):
    """A Pipeline StandardScaler -> LinearRegression, fitted on (X, y), returned."""
    raise NotImplementedError("linear_12")


def coefs_12(pipeline):
    """pd.Series of the 8 coefficients of the LinearRegression of the pipeline, indexed by the features."""
    raise NotImplementedError("coefs_12")


def boosting_12(X, y):
    """HistGradientBoostingRegressor(random_state=0), every other setting by default, fitted on (X, y), returned."""
    raise NotImplementedError("boosting_12")


def permutation_12(model, X, y):
    """pd.Series: the mean drop of R² of permutation_importance(model, X, y, n_repeats=5, random_state=0), indexed by
    the columns of X, sorted from the largest to the smallest."""
    raise NotImplementedError("permutation_12")


def pd_ice_12(model, X, feature):
    """dict {"grid": (50,), "average": (50,), "individual": (len(X), 50)} of
    partial_dependence(model, X, [feature], kind="both", grid_resolution=50)."""
    raise NotImplementedError("pd_ice_12")'''

SOLUTION_12 = r'''def linear_12(X, y):
    """A Pipeline StandardScaler -> LinearRegression, fitted on (X, y), returned."""
    return make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)


def coefs_12(pipeline):
    """pd.Series of the 8 coefficients of the LinearRegression of the pipeline, indexed by the features."""
    return pd.Series(pipeline[-1].coef_, index=pipeline.feature_names_in_)


def boosting_12(X, y):
    """HistGradientBoostingRegressor(random_state=0), every other setting by default, fitted on (X, y), returned."""
    return HistGradientBoostingRegressor(random_state=0).fit(X, y)


def permutation_12(model, X, y):
    """pd.Series: the mean drop of R² of permutation_importance(model, X, y, n_repeats=5, random_state=0), indexed by
    the columns of X, sorted from the largest to the smallest."""
    result = permutation_importance(model, X, y, n_repeats=5, random_state=0)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def pd_ice_12(model, X, feature):
    """dict {"grid": (50,), "average": (50,), "individual": (len(X), 50)} of
    partial_dependence(model, X, [feature], kind="both", grid_resolution=50)."""
    result = partial_dependence(model, X, [feature], kind="both", grid_resolution=50)
    return {"grid": result["grid_values"][0], "average": result["average"][0], "individual": result["individual"][0]}'''

CHECK_12 = r'''ROWS_12 = X_VAL.iloc[:300]                     # the districts of the curves of the check
REF_LIN_12 = make_pipeline(StandardScaler(), LinearRegression()).fit(X_TRAIN, Y_TRAIN)
ok_12 = {}
with checked("B6.12"):
    lin_12 = linear_12(X_TRAIN.copy(), Y_TRAIN.copy())
    if returned("B6.12", "linear_12", lin_12):
        steps_12 = [step for _, step in getattr(lin_12, "steps", [])]
        verdict("B6.12", isinstance(lin_12, Pipeline) and steps_12 and isinstance(steps_12[-1], LinearRegression)
                and any(isinstance(s, StandardScaler) for s in steps_12),
                "linear_12 rend un Pipeline StandardScaler → LinearRegression.",
                f"linear_12 doit rendre un Pipeline (StandardScaler, puis LinearRegression), reçu {type(lin_12).__name__} "
                f"{[type(s).__name__ for s in steps_12]}.")
        scaler_12 = next((s for s in steps_12 if isinstance(s, StandardScaler)), None)
        if scaler_12 is not None and not (scaler_12.with_mean and scaler_12.with_std):
            print("❌ Ex B6.12 : ton StandardScaler ne centre pas ou ne réduit pas (with_mean="
                  f"{scaler_12.with_mean}, with_std={scaler_12.with_std}) : le contrat veut ses réglages par défaut, "
                  "pour des coefficients qui mesurent l'effet d'un écart-type.")
            raise StopCheck
        if scaler_12 is not None and getattr(scaler_12, "mean_", None) is not None:
            verdict("B6.12", np.allclose(scaler_12.mean_, X_TRAIN.mean().to_numpy()),
                    "le StandardScaler a appris les moyennes de X_TRAIN.",
                    "le StandardScaler n'a pas appris les moyennes de X_TRAIN : sur quelles lignes est-il ajusté ?")
        c_12 = coefs_12(lin_12)
        want_c_12 = pd.Series(REF_LIN_12[-1].coef_, index=FEATURES)
        raw_12 = LinearRegression().fit(X_TRAIN, Y_TRAIN).coef_
        if not isinstance(c_12, pd.Series) or list(c_12.index) != FEATURES:
            print("❌ Ex B6.12 : coefs_12 doit rendre une pd.Series indexée par les 8 features, dans l'ordre de FEATURES "
                  f"(reçu {type(c_12).__name__}).")
        elif np.allclose(c_12.to_numpy(), raw_12):
            print("❌ Ex B6.12 : ce sont les coefficients des features brutes : chacun dépend de l'unité de sa feature ; "
                  "le contrat les veut sur des features standardisées.")
        else:
            ok_12["coefs"] = same("B6.12", "coefficients sur features standardisées", c_12.to_numpy(),
                                  want_c_12.to_numpy(), rtol=1e-6, atol=1e-9)

with checked("B6.12"):
    gb_12 = boosting_12(X_TRAIN.copy(), Y_TRAIN.copy())
    if returned("B6.12", "boosting_12", gb_12):
        if not isinstance(gb_12, HistGradientBoostingRegressor):
            print(f"❌ Ex B6.12 : boosting_12 doit rendre un HistGradientBoostingRegressor, reçu {type(gb_12).__name__}.")
            raise StopCheck
        want_p_12 = HistGradientBoostingRegressor(random_state=0).get_params()
        diff_12 = {k: v for k, v in gb_12.get_params().items() if want_p_12.get(k) != v}
        verdict("B6.12", not diff_12, "réglages : random_state=0, le reste par défaut.",
                f"réglages différents du contrat : {diff_12}.")
        same("B6.12", "prédictions de boosting_12 sur 200 districts de validation", gb_12.predict(X_VAL.iloc[:200]),
             GBR.predict(X_VAL.iloc[:200]), rtol=1e-9, atol=1e-9)

with checked("B6.12"):
    imp_12 = permutation_12(GBR, X_VAL.copy(), Y_VAL.copy())      # the reference model: a slip above does not spread
    want_i_12 = pd.Series(permutation_importance(GBR, X_VAL, Y_VAL, n_repeats=5, random_state=0).importances_mean,
                          index=FEATURES).sort_values(ascending=False)
    if not isinstance(imp_12, pd.Series) or sorted(imp_12.index) != sorted(FEATURES):
        print("❌ Ex B6.12 : permutation_12 doit rendre une pd.Series indexée par les 8 features "
              f"(reçu {type(imp_12).__name__}).")
        raise StopCheck
    verdict("B6.12", list(imp_12.index) == list(want_i_12.index) or bool((np.diff(imp_12.to_numpy()) <= 0).all()),
            "importances triées de la plus grande à la plus petite.",
            "permutation_12 doit trier les importances de la plus grande à la plus petite.")
    if not np.allclose(imp_12[FEATURES].to_numpy(), want_i_12[FEATURES].to_numpy(), atol=1e-9):
        on_train_12 = permutation_importance(GBR, X_TRAIN, Y_TRAIN, n_repeats=5, random_state=0).importances_mean
        if np.allclose(imp_12[FEATURES].to_numpy(), on_train_12, atol=1e-9):
            print("❌ Ex B6.12 : ces importances sont celles des districts d'entraînement : la permutation mesure ce que "
                  "le modèle perd sur des données qu'il n'a pas vues (relis 206.2).")
        else:
            same("B6.12", "importances par permutation (validation, 5 répétitions, random_state=0)",
                 imp_12[FEATURES].to_numpy(), want_i_12[FEATURES].to_numpy(), atol=1e-9)
    else:
        ok_12["perm"] = True
        print("✅ Ex B6.12 : importances par permutation sur la validation, comme attendu.")

with checked("B6.12"):
    out_12 = pd_ice_12(GBR, ROWS_12.copy(), "AveOccup")
    if not isinstance(out_12, dict) or set(out_12) != {"grid", "average", "individual"}:
        print("❌ Ex B6.12 : pd_ice_12 doit rendre un dictionnaire aux trois clés 'grid', 'average', 'individual'.")
        raise StopCheck
    ref_12 = partial_dependence(GBR, ROWS_12, ["AveOccup"], kind="both", grid_resolution=50)
    same("B6.12", "grid de AveOccup (50 valeurs)", out_12["grid"], ref_12["grid_values"][0])
    ind_12 = as_numpy(out_12["individual"])
    if ind_12 is not None and ind_12.ndim == 2 and ind_12.shape[1] > 1 and np.allclose(ind_12[:, 0], 0, atol=1e-9):
        print("❌ Ex B6.12 : tes courbes ICE partent toutes de 0 : c'est la version centrée (chaque courbe moins sa "
              "première valeur) ; le contrat veut les prédictions elles-mêmes.")
        raise StopCheck
    same("B6.12", "courbes ICE de AveOccup (300 districts × 50 valeurs)", out_12["individual"],
         ref_12["individual"][0], rtol=1e-9, atol=1e-9)
    if same("B6.12", "dépendance partielle de AveOccup", out_12["average"], ref_12["average"][0], rtol=1e-9, atol=1e-9):
        ok_12["pd"] = True
        verdict("B6.12", np.allclose(out_12["average"], np.asarray(out_12["individual"]).mean(axis=0)),
                "la dépendance partielle est la moyenne des courbes ICE.",
                "la dépendance partielle devrait être la moyenne des courbes ICE.")

if len(ok_12) == 3:                              # the figure, once your three results are right
    fig_12, axes_12 = plt.subplots(1, 4, figsize=(17, 3.8))
    order_12 = imp_12.index
    axes_12[0].barh(order_12[::-1], c_12[order_12[::-1]].abs(), color="tab:gray")
    axes_12[0].set_title("|coefficient| (features standardisées)", fontsize=10)
    axes_12[1].barh(order_12[::-1], imp_12[order_12[::-1]], color="tab:green")
    axes_12[1].set_title("permutation : baisse du R² (validation)", fontsize=10)
    for ax_12, feat_12 in zip(axes_12[2:], ("AveOccup", "HouseAge")):
        curves_12 = pd_ice_12(GBR, ROWS_12, feat_12)
        ax_12.plot(curves_12["grid"], np.asarray(curves_12["individual"])[:80].T, color="tab:blue", alpha=0.15, lw=0.8)
        ax_12.plot(curves_12["grid"], curves_12["average"], color="black", lw=3, label="dépendance partielle")
        ax_12.set_xlabel(f"{feat_12} imposé")
        ax_12.set_ylabel("prédiction (MedHouseVal)")
        ax_12.set_title(f"80 courbes ICE de {feat_12} et leur moyenne", fontsize=10)
        ax_12.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    print("coefficients standardisés :", ", ".join(f"{k} {fr(v, 2)}" for k, v in c_12.items()))
    print("importances par permutation :", ", ".join(f"{k} {fr(v, 3)}" for k, v in imp_12.items()))'''

# --- 🔮 B6.13 ---------------------------------------------------------------------------------------------------------
GIVEN_13 = r'''ROWS_13 = X_VAL.iloc[:500]                        # the 500 validation districts of the curves


def curves_13(feature):
    """(grid, average, individual) of the reference gradient boosting GBR for `feature` on ROWS_13: a grid of 50 values
    between the 5th and the 95th percentiles of the feature (written here without your code)."""
    result = partial_dependence(GBR, ROWS_13, [feature], kind="both", grid_resolution=50)
    return result["grid_values"][0], result["average"][0], result["individual"][0]'''

TODO_13 = r'''medinc_rise_13 = ...   # a) by how much does the partial dependence of MedInc rise from the start to the end of its
                       #    grid? "A" less than 0.5, "B" 0.5 to 1.5, "C" 1.5 to 3, "D" more than 3 (units of MedHouseVal)
lat_trend_13 = ...     # b) from the south to the north of its grid, the partial dependence of Latitude: "A" rises
                       #    (by more than 0.3), "B" stays about flat (between -0.3 and +0.3), "C" falls (by more than 0.3)
lat_up_13 = ...        # c) which share of the 500 ICE curves of Latitude END HIGHER than they start? "A" less than 5 %,
                       #    "B" 5 to 30 %, "C" 30 to 70 %, "D" more than 70 %
spread_13 = ...        # d) do the ICE curves of Latitude vary more from one district to another than those of MedInc
                       #    (a larger standard deviation of their change, end minus start)? True or False'''

EXPERIMENT_13 = r'''fig_13, axes_13 = plt.subplots(1, 2, figsize=(12, 4))
seen_13 = {}
for ax_13, feature_13 in zip(axes_13, ("MedInc", "Latitude")):
    grid_13, avg_13, ind_13 = curves_13(feature_13)
    ax_13.plot(grid_13, ind_13[:100].T, color="tab:blue", alpha=0.15, lw=0.8)
    ax_13.plot(grid_13, avg_13, color="black", lw=3, label="dépendance partielle")
    ax_13.set_xlabel(f"{feature_13} imposé (du 5e au 95e centile)")
    ax_13.set_ylabel("prédiction (MedHouseVal)")
    ax_13.set_title(f"100 courbes ICE de {feature_13} (sur 500) et leur moyenne", fontsize=10)
    ax_13.legend(fontsize=8)
    change_13 = ind_13[:, -1] - ind_13[:, 0]
    seen_13[feature_13] = {"rise": float(avg_13[-1] - avg_13[0]), "up": float(np.mean(change_13 > 0)),
                           "spread": float(np.std(change_13))}
plt.tight_layout()
plt.show()
for feature_13, s_13 in seen_13.items():
    print(f"{feature_13:>8} : variation de la dépendance partielle (fin − début) {fr(s_13['rise'], 2)} ; "
          f"courbes ICE qui finissent plus haut {pct(s_13['up'])} ; écart-type de la variation des ICE {fr(s_13['spread'], 2)}")
wb.check("B6.13a", medinc_rise_13)
wb.check("B6.13b", lat_trend_13)
wb.check("B6.13c", lat_up_13)
wb.check("B6.13d", spread_13)'''

# --- 🔨 B6.14 ---------------------------------------------------------------------------------------------------------
CHECK_14 = r'''with checked("B6.14"):
    E = mylearn.explain
    rng_14 = np.random.default_rng(14)                 # a linear model on a random background: w_i (x_i - mean_i)
    w5_14, x5_14, bg5_14 = rng_14.normal(size=5), rng_14.normal(size=5), rng_14.normal(size=(20, 5))
    shapes_14 = []

    def spy_14(X):                                     # first: what does f receive?
        shapes_14.append(np.shape(X))
        return np.asarray(X, dtype=float) @ w5_14

    E.shapley_values(spy_14, x5_14, bg5_14)
    bad_14 = sorted({s for s in shapes_14 if len(s) != 2 or s[1] != 5})
    verdict("B6.14", not bad_14, "f ne reçoit que des tableaux 2-D de 5 colonnes.",
            f"f a reçu les formes {bad_14[:3]} : appelle-la sur des tableaux (n, M), jamais sur une ligne seule.")
    rows_14 = sum(s[0] for s in shapes_14 if len(s) == 2)
    print(f"ℹ️ f a été appelée {len(shapes_14)} fois, sur {rows_14} lignes en tout (2**5 coalitions × 20 lignes = "
          f"{2 ** 5 * 20}).")
    w_14 = np.array([2.0, -1.0, 0.5])
    bg_14 = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    phi_14, base_14 = pair("B6.14", "shapley_values", E.shapley_values(lambda X: X @ w_14, [3.0, 1.0, 1.0], bg_14.copy()),
                           names=("phi", "base_value"), floats=("base_value",))
    same("B6.14", "phi du modèle linéaire de la docstring", phi_14, [4.0, 0.0, 0.0])
    same("B6.14", "base_value du modèle linéaire de la docstring", base_14, 1.5)
    same("B6.14", "le produit x0 x1 de la docstring : l'interaction partagée",
         E.shapley_values(lambda X: X[:, 0] * X[:, 1], [1.0, 1.0], [[0.0, 0.0]])[0], [0.5, 0.5])

    same("B6.14", "modèle linéaire à 5 features, fond de 20 lignes : w_i (x_i − moyenne du fond)",
         E.shapley_values(lambda X: X @ w5_14 + 1.0, x5_14, bg5_14)[0], w5_14 * (x5_14 - bg5_14.mean(axis=0)),
         rtol=1e-8, atol=1e-10)

    X0_14, BG_14 = X_VAL.iloc[0].to_numpy(), X_TRAIN.iloc[:50].to_numpy()
    copies_14 = (X0_14.copy(), BG_14.copy())
    start_14 = time.perf_counter()
    phi_c_14, base_c_14 = pair("B6.14", "shapley_values", E.shapley_values(GBR.predict, X0_14, BG_14),
                               names=("phi", "base_value"), floats=("base_value",))
    seconds_14 = time.perf_counter() - start_14
    same("B6.14", "California, efficacité : somme des phi + base_value = f(x)", float(np.sum(phi_c_14)) + base_c_14,
         float(GBR.predict(X0_14[None])[0]), rtol=1e-9, atol=1e-9)
    same("B6.14", "California : base_value = moyenne de f sur les 50 lignes du fond", base_c_14,
         float(GBR.predict(BG_14).mean()), rtol=1e-9, atol=1e-9)
    if shap is not None:
        exact_14 = shap.explainers.Exact(GBR.predict, shap.maskers.Independent(BG_14, max_samples=len(BG_14)))(X0_14[None])
        same("B6.14", "California : phi contre shap.explainers.Exact (fond complet)", phi_c_14, exact_14.values[0],
             rtol=1e-7, atol=1e-9)
    print(f"ℹ️ California, 8 features, fond de 50 lignes (256 coalitions × 50 = 12 800 lignes) : {fr(seconds_14, 2)} s.")
    verdict("B6.14", np.array_equal(X0_14, copies_14[0]) and np.array_equal(BG_14, copies_14[1]),
            "x et le fond n'ont pas changé.", "shapley_values a modifié x ou le fond reçus.")

    try:
        E.shapley_values(lambda X: X.sum(axis=1), np.ones(3), np.zeros((2, 3)), max_features=3)
        print("✅ Ex B6.14 : M = max_features est permis.")
    except ValueError as exc_14:
        print(f"❌ Ex B6.14 : M = max_features doit être permis (seul M > max_features lève) : reçu ValueError ({exc_14}).")
    refuses("B6.14", "M = 4 > max_features = 3", lambda: E.shapley_values(lambda X: X.sum(axis=1), np.ones(4),
                                                                        np.zeros((2, 4)), max_features=3))
    refuses("B6.14", "x en 2-D", lambda: E.shapley_values(lambda X: X.sum(axis=1), np.ones((1, 3)), np.zeros((2, 3))))
    refuses("B6.14", "un fond de 2 colonnes pour x de 3", lambda: E.shapley_values(lambda X: X.sum(axis=1), np.ones(3),
                                                                                   np.zeros((2, 2))))'''

# --- 📦 B6.15 ---------------------------------------------------------------------------------------------------------
GIVEN_15 = r'''BG_15 = X_TRAIN.iloc[:50]          # the background: the first 50 training districts
ROWS_15 = X_VAL.iloc[:1000]        # the 1 000 validation districts to explain
MEDINC = FEATURES.index("MedInc")  # the column of MedInc (0)'''

TODO_15 = r'''def fit_15(X, y):
    """HistGradientBoostingRegressor(max_depth=3, random_state=0), every other setting by default, fitted on (X, y)."""
    raise NotImplementedError("fit_15")


def tree_shap_15(model, background, rows):
    """The shap.Explanation of shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    on the DataFrame `rows`: .values of shape (len(rows), 8), .base_values, .feature_names."""
    raise NotImplementedError("tree_shap_15")'''

SOLUTION_15 = r'''def fit_15(X, y):
    """HistGradientBoostingRegressor(max_depth=3, random_state=0), every other setting by default, fitted on (X, y)."""
    return HistGradientBoostingRegressor(max_depth=3, random_state=0).fit(X, y)


def tree_shap_15(model, background, rows):
    """The shap.Explanation of shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    on the DataFrame `rows`: .values of shape (len(rows), 8), .base_values, .feature_names."""
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    return explainer(rows)'''

CHECK_15 = r'''SV_15 = MODEL_15 = None
BASE_OK_15 = TREE_OK_15 = False
with checked("B6.15"):
    if shap is None:
        print("⏳ Ex B6.15 : cet exercice a besoin de shap (wb.ensure('shap')).")
        raise StopCheck
    model_15 = fit_15(X_TRAIN.copy(), Y_TRAIN.copy())
    if not returned("B6.15", "fit_15", model_15):
        raise StopCheck
    if not isinstance(model_15, HistGradientBoostingRegressor):
        print(f"❌ Ex B6.15 : fit_15 doit rendre un HistGradientBoostingRegressor, reçu {type(model_15).__name__}.")
        raise StopCheck
    want_p_15 = HistGradientBoostingRegressor(max_depth=3, random_state=0).get_params()
    diff_15 = {k: v for k, v in model_15.get_params().items() if want_p_15.get(k) != v}
    verdict("B6.15", not diff_15, "réglages : max_depth=3, random_state=0, le reste par défaut.",
            f"réglages différents du contrat : {diff_15}.")
    ref_15 = HistGradientBoostingRegressor(max_depth=3, random_state=0).fit(X_TRAIN, Y_TRAIN)
    if not same("B6.15", "prédictions de fit_15 sur 200 districts de validation", model_15.predict(X_VAL.iloc[:200]),
                ref_15.predict(X_VAL.iloc[:200]), rtol=1e-9, atol=1e-9):
        raise StopCheck
    MODEL_15 = model_15
    start_15 = time.perf_counter()
    sv_15 = tree_shap_15(model_15, BG_15.copy(), ROWS_15.copy())
    print(f"ℹ️ tree_shap_15 sur les 1 000 districts : {fr(time.perf_counter() - start_15, 1)} s.")
    if not returned("B6.15", "tree_shap_15", sv_15):
        raise StopCheck
    if isinstance(sv_15, np.ndarray):
        print("❌ Ex B6.15 : tree_shap_15 rend un tableau NumPy : le contrat veut le shap.Explanation de explainer(rows) "
              "(avec .base_values et .feature_names), pas explainer.shap_values(rows).")
        raise StopCheck
    if np.shape(getattr(sv_15, "values", None)) != (len(ROWS_15), len(FEATURES)):
        print("❌ Ex B6.15 : tree_shap_15 doit rendre un shap.Explanation dont .values a la forme (1000, 8), reçu "
              f"{type(sv_15).__name__} de forme {np.shape(getattr(sv_15, 'values', None))}.")
        raise StopCheck
    if not getattr(sv_15, "feature_names", None):
        sv_15.feature_names = list(FEATURES)               # rows given as an array: the names for the plots
    SV_15 = sv_15
    gap_eff_15 = float(np.abs(sv_15.values.sum(axis=1) + np.ravel(sv_15.base_values) - model_15.predict(ROWS_15)).max())
    verdict("B6.15", gap_eff_15 < 1e-6, f"efficacité sur les 1 000 districts : écart maximal {gap_eff_15:.1e}.",
            f"efficacité : somme des valeurs SHAP + valeur de base s'écarte de la prédiction de {gap_eff_15:.2g} : "
            "l'explication porte-t-elle sur ce modèle et sur ces lignes ?")
    BASE_OK_15 = bool(abs(float(np.ravel(sv_15.base_values)[0]) - float(model_15.predict(BG_15).mean())) < 1e-6)
    TREE_OK_15 = BASE_OK_15 and bool(np.allclose(sv_15.values[:5], shap.TreeExplainer(
        model_15, data=BG_15, feature_perturbation="interventional")(ROWS_15.iloc[:5]).values, rtol=1e-6, atol=1e-8))
    verdict("B6.15", BASE_OK_15, "la valeur de base est la prédiction moyenne de ton modèle sur les 50 lignes de BG_15.",
            "la valeur de base n'est pas la prédiction moyenne de ton modèle sur les 50 lignes de BG_15 : tree_shap_15 "
            "reçoit-il tout le fond, en mode interventionnel ?")
    wb.check("B6.15a", float(np.ravel(sv_15.base_values)[0]), computed=True)
    wb.check("B6.15b", float(sv_15.values[0, MEDINC]), computed=True)
    wb.check("B6.15c", float(np.mean(np.abs(sv_15.values[:, MEDINC]))), computed=True)

with checked("B6.15"):
    if SV_15 is None:
        raise StopCheck
    if not TREE_OK_15:
        print("⏳ Ex B6.15d : corrige d'abord tree_shap_15 (ci-dessus) : la comparaison à ta shapley_values suppose les "
              "valeurs de TreeSHAP interventionnel, avec le même fond de 50 lignes.")
        raise StopCheck
    if not shapley_ready("B6.15d"):
        raise StopCheck
    x0_15 = ROWS_15.iloc[0].to_numpy()
    phi_15, base_15 = pair("B6.15", "shapley_values", mylearn.explain.shapley_values(MODEL_15.predict, x0_15,
                                                                                     BG_15.to_numpy()),
                           names=("phi", "base_value"), floats=("base_value",))
    gap_15 = max(float(np.abs(np.asarray(phi_15, dtype=float) - SV_15.values[0]).max()),
                 abs(base_15 - float(np.ravel(SV_15.base_values)[0])))
    print(f"ℹ️ écart maximal entre TreeExplainer et ta shapley_values sur le premier district : {gap_15:.1e}")
    wb.check("B6.15d", bool(gap_15 <= 1e-6))'''

PLOTS_15 = r'''if SV_15 is not None:                       # the three plots of 206.4, on your explanation
    shap.plots.waterfall(SV_15[0], show=False)
    plt.title("waterfall du premier district", fontsize=10)
    plt.show()
    shap.plots.beeswarm(SV_15, show=False)
    plt.title("beeswarm des 1 000 districts", fontsize=10)
    plt.show()
    shap.plots.scatter(SV_15[:, MEDINC], color=SV_15, show=False)
    plt.title("dépendance de MedInc (couleur : la feature qui interagit le plus)", fontsize=10)
    plt.show()
else:
    print("⏳ Ex B6.15 : les graphiques attendent ton explication (cellule précédente).")'''

RECORD_15 = r'''# The right values, computed above with the solution; the classic slips, recomputed here
tree_path_15 = shap.TreeExplainer(MODEL_15)(ROWS_15.iloc[:1])                # without background: path-dependent
default_15 = shap.TreeExplainer(GBR, data=BG_15, feature_perturbation="interventional")(ROWS_15)   # the default boosting
bg100_15 = shap.TreeExplainer(MODEL_15, data=X_TRAIN.iloc[:100], feature_perturbation="interventional")(ROWS_15.iloc[:1])
path_bg_15 = shap.TreeExplainer(MODEL_15, data=BG_15, feature_perturbation="tree_path_dependent")(ROWS_15)   # same base
PATH_BG_15 = ("c'est le mode « chemins de l'arbre » (feature_perturbation=\"tree_path_dependent\"), même avec un fond : "
              "relis les deux modes de TreeExplainer (§206.4)")
base_r_15 = float(np.ravel(SV_15.base_values)[0])
phi_r_15 = float(SV_15.values[0, MEDINC])
mean_r_15 = float(np.mean(np.abs(SV_15.values[:, MEDINC])))


def distinct_15(right, slips):
    """The slips whose value differs from the right one, and from each other, at 4 decimals."""
    kept, seen = {}, {round(right, 4)}
    for message, value in slips.items():
        if round(value, 4) not in seen:
            kept[message] = value
            seen.add(round(value, 4))
    return kept


wb.record("B6.15a", base_r_15, decimals=4, mistakes=distinct_15(base_r_15, {
    "c'est la valeur de base sans fond (mode « chemins de l'arbre ») : relis les deux modes de TreeExplainer (§206.4)":
        float(np.ravel(tree_path_15.base_values)[0]),
    "c'est la moyenne des cibles d'entraînement : la valeur de base est la prédiction moyenne sur le fond": float(Y_TRAIN.mean()),
    "c'est la prédiction moyenne sur les districts expliqués : la valeur de base se calcule sur le fond": float(MODEL_15.predict(ROWS_15).mean()),
    "c'est le gradient boosting par défaut : le contrat veut max_depth=3": float(np.ravel(default_15.base_values)[0])}))
wb.record("B6.15b", phi_r_15, decimals=4, mistakes=distinct_15(phi_r_15, {
    "c'est la valeur sans fond (mode « chemins de l'arbre ») : relis les deux modes de TreeExplainer (§206.4)":
        float(tree_path_15.values[0, MEDINC]),
    "c'est le gradient boosting par défaut : le contrat veut max_depth=3": float(default_15.values[0, MEDINC]),
    "c'est un fond de 100 lignes : le contrat en prend 50": float(bg100_15.values[0, MEDINC]),
    PATH_BG_15: float(path_bg_15.values[0, MEDINC]),
    "vérifie le signe": -phi_r_15}))
tree_path_all_15 = shap.TreeExplainer(MODEL_15)(ROWS_15)
wb.record("B6.15c", mean_r_15, decimals=4, mistakes=distinct_15(mean_r_15, {
    "c'est la valeur sans fond (mode « chemins de l'arbre ») : relis les deux modes de TreeExplainer (§206.4)":
        float(np.mean(np.abs(tree_path_all_15.values[:, MEDINC]))),
    "c'est la moyenne des valeurs SHAP signées : on demande la moyenne de leurs valeurs absolues":
        float(np.mean(SV_15.values[:, MEDINC])),
    "c'est la moyenne des carrés : on demande la moyenne des valeurs absolues": float(np.mean(SV_15.values[:, MEDINC] ** 2)),
    "c'est le gradient boosting par défaut : le contrat veut max_depth=3": float(np.mean(np.abs(default_15.values[:, MEDINC]))),
    PATH_BG_15: float(np.mean(np.abs(path_bg_15.values[:, MEDINC])))}))
wb.record("B6.15d", True, mistakes={
    "TreeExplainer interventionnel et ta shapley_values diffèrent de plus de 1e-6 : ta fonction passe-t-elle ses tests "
    "(Ex B6.14), et garde-t-elle toutes les lignes du fond ?": False})'''

# --- 🔬 B6.16 ---------------------------------------------------------------------------------------------------------
GIVEN_16 = r'''# the district to explain: the validation district closest to the 75th percentile of MedInc
X_16 = X_VAL.iloc[int(np.argmin(np.abs(X_VAL["MedInc"].to_numpy() - X_VAL["MedInc"].quantile(0.75))))]
RICH_16 = X_TRAIN[X_TRAIN["MedInc"] >= X_TRAIN["MedInc"].quantile(0.9)]          # the top decile of income
BACKGROUNDS_16 = {"population": X_TRAIN.sample(50, random_state=0),             # 50 districts drawn at random
                  "riches": RICH_16.sample(50, random_state=0),                  # 50 districts of the top decile
                  "ligne type": X_TRAIN.median().to_frame().T.astype("float32").astype("float64")}   # one row: medians
SD_MEDINC_16 = float(X_TRAIN["MedInc"].std())
NOISES_16, SEEDS_16 = (0.0, 0.01, 0.1), (0, 1, 2)


def add_copy_16(X, noise, rng):
    """A copy of the DataFrame X with the column "MedInc_copy" = MedInc + noise * SD_MEDINC_16 * N(0, 1), drawn by
    rng.standard_normal(len(X)), placed right after MedInc and rounded to float32 like the other features (B6.15)."""
    out = X.copy()
    copy = X["MedInc"].to_numpy() + noise * SD_MEDINC_16 * rng.standard_normal(len(X))
    out.insert(1, "MedInc_copy", copy.astype("float32").astype("float64"))
    return out


print(f"district expliqué : MedInc {fr(X_16['MedInc'], 2)}, Latitude {fr(X_16['Latitude'], 2)}, prédiction "
      f"{fr(GBR.predict(X_16.to_frame().T)[0], 2)} ; fonds : " + ", ".join(f"{k} ({len(v)} ligne{'s' * (len(v) > 1)})"
                                                                            for k, v in BACKGROUNDS_16.items()))'''

TODO_16 = r'''def explain_16(model, x, background):
    """(phi, base): the interventional Shapley values of model.predict at the district x (a pd.Series of the 8
    features) with the DataFrame `background`, as an array (8,), and the base value, a float."""
    raise NotImplementedError("explain_16")


def copy_experiment_16(noise, seed, n_rows=200):
    """The duplicated feature: rng = np.random.default_rng(seed); A = add_copy_16(X_TRAIN, noise, rng), then
    V = add_copy_16(X_VAL, noise, rng) (the same rng, in this order); HistGradientBoostingRegressor(random_state=seed)
    fitted on (A, Y_TRAIN); shap.TreeExplainer (interventional, background: the first 50 rows of A) on the first
    n_rows rows of V. Returns {"MedInc": mean |phi_MedInc|, "MedInc_copy": mean |phi_copy|,
    "sum": mean |phi_MedInc + phi_copy|} over these rows (floats)."""
    raise NotImplementedError("copy_experiment_16")'''

SOLUTION_16 = r'''def explain_16(model, x, background):
    """(phi, base): the interventional Shapley values of model.predict at the district x (a pd.Series of the 8
    features) with the DataFrame `background`, as an array (8,), and the base value, a float."""
    explanation = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")(x.to_frame().T)
    return explanation.values[0], float(np.ravel(explanation.base_values)[0])


def copy_experiment_16(noise, seed, n_rows=200):
    """The duplicated feature: rng = np.random.default_rng(seed); A = add_copy_16(X_TRAIN, noise, rng), then
    V = add_copy_16(X_VAL, noise, rng) (the same rng, in this order); HistGradientBoostingRegressor(random_state=seed)
    fitted on (A, Y_TRAIN); shap.TreeExplainer (interventional, background: the first 50 rows of A) on the first
    n_rows rows of V. Returns {"MedInc": mean |phi_MedInc|, "MedInc_copy": mean |phi_copy|,
    "sum": mean |phi_MedInc + phi_copy|} over these rows (floats)."""
    rng = np.random.default_rng(seed)
    A, V = add_copy_16(X_TRAIN, noise, rng), add_copy_16(X_VAL, noise, rng)
    model = HistGradientBoostingRegressor(random_state=seed).fit(A, Y_TRAIN)
    values = shap.TreeExplainer(model, data=A.iloc[:50], feature_perturbation="interventional")(V.iloc[:n_rows]).values
    i, j = list(A.columns).index("MedInc"), list(A.columns).index("MedInc_copy")
    return {"MedInc": float(np.mean(np.abs(values[:, i]))), "MedInc_copy": float(np.mean(np.abs(values[:, j]))),
            "sum": float(np.mean(np.abs(values[:, i] + values[:, j])))}'''

CHECK_16 = r'''TABLE_16 = {}
with checked("B6.16"):
    if shap is None:
        print("⏳ Ex B6.16 : cet exercice a besoin de shap (wb.ensure('shap')).")
        raise StopCheck
    for name_16, bg_16 in BACKGROUNDS_16.items():
        got_16 = explain_16(GBR, X_16.copy(), bg_16.copy())
        phi_16, base_16 = pair("B6.16", "explain_16", got_16, names=("phi", "base"), floats=("base",))
        if len(bg_16) > 1 and abs(base_16 - float(GBR.predict(bg_16.iloc[:1])[0])) < 1e-9:
            print(f"❌ Ex B6.16 : fond « {name_16} » : ta valeur de base est la prédiction de la première ligne du fond : "
                  f"v(∅) est la moyenne de f sur ses {len(bg_16)} lignes.")
            continue
        want_16 = shap.TreeExplainer(GBR, data=bg_16, feature_perturbation="interventional")(X_16.to_frame().T)
        ok_16 = same("B6.16", f"fond « {name_16} » : phi", phi_16, want_16.values[0], rtol=1e-6, atol=1e-6)
        ok_16 &= same("B6.16", f"fond « {name_16} » : valeur de base", base_16, float(np.ravel(want_16.base_values)[0]),
                      rtol=1e-6, atol=1e-6)
        if ok_16:
            TABLE_16[name_16] = (base_16, np.asarray(phi_16, dtype=float))
if len(TABLE_16) == len(BACKGROUNDS_16):
    f_16 = float(GBR.predict(X_16.to_frame().T)[0])
    print(f"\nprédiction du district : {fr(f_16, 3)}")
    print(f"{'fond':>12} | {'base':>6} | " + " | ".join(f"{name[:9]:>9}" for name in FEATURES))
    for name_16, (base_16, phi_16) in TABLE_16.items():
        print(f"{name_16:>12} | {fr(base_16, 3):>6} | " + " | ".join(f"{fr(v, 3):>9}" for v in phi_16))'''

EXPERIMENT_16 = r'''COPIES_16 = {}
with checked("B6.16"):
    if shap is None:
        raise StopCheck
    got_16 = copy_experiment_16(0.01, 0, n_rows=50)      # a small first run, checked before the whole grid
    if not isinstance(got_16, dict) or set(got_16) != {"MedInc", "MedInc_copy", "sum"}:
        print("❌ Ex B6.16 : copy_experiment_16 doit rendre un dictionnaire aux clés 'MedInc', 'MedInc_copy', 'sum'.")
        raise StopCheck
    rng_16 = np.random.default_rng(0)
    A_16, V_16 = add_copy_16(X_TRAIN, 0.01, rng_16), add_copy_16(X_VAL, 0.01, rng_16)
    m_16 = HistGradientBoostingRegressor(random_state=0).fit(A_16, Y_TRAIN)
    v_16 = shap.TreeExplainer(m_16, data=A_16.iloc[:50], feature_perturbation="interventional")(V_16.iloc[:50]).values
    want_16 = [np.mean(np.abs(v_16[:, 0])), np.mean(np.abs(v_16[:, 1])), np.mean(np.abs(v_16[:, 0] + v_16[:, 1]))]
    if not same("B6.16", "copy_experiment_16(0.01, 0, n_rows=50)", [got_16[k] for k in ("MedInc", "MedInc_copy", "sum")],
                want_16, rtol=1e-6, atol=1e-6):
        raise StopCheck
    start_16 = time.perf_counter()
    for noise_16 in NOISES_16:
        for seed_16 in SEEDS_16:
            COPIES_16[noise_16, seed_16] = copy_experiment_16(noise_16, seed_16)
    print(f"ℹ️ {len(COPIES_16)} entraînements et explications en {time.perf_counter() - start_16:.0f} s.\n")
    alone_16 = shap.TreeExplainer(GBR, data=X_TRAIN.iloc[:50], feature_perturbation="interventional")(X_VAL.iloc[:200])
    print(f"sans copie : moyenne de |phi_MedInc| sur les 200 districts = {fr(np.mean(np.abs(alone_16.values[:, 0])), 3)}")
    for noise_16 in NOISES_16:
        cells_16 = [COPIES_16[noise_16, s] for s in SEEDS_16]
        print(f"bruit {fr(noise_16, 2)} écart-type : " + " ; ".join(
            f"graine {s} : MedInc {fr(c['MedInc'], 3)}, copie {fr(c['MedInc_copy'], 3)}, somme {fr(c['sum'], 3)}"
            for s, c in zip(SEEDS_16, cells_16)))'''

# --- 📦 B6.17 ---------------------------------------------------------------------------------------------------------
GIVEN_17 = r'''PENG = wbx.penguins()                               # frozen split of Penguins: 249 penguins, 84 for the test
XP_TRAIN, XP_TEST = PENG.X_train.to_numpy(), PENG.X_test.to_numpy()   # the 4 measures, as arrays (columns PENG.features)
YP_TRAIN, YP_TEST = PENG.y_train.to_numpy(), PENG.y_test.to_numpy()   # the species, strings
SPECIES = PENG.classes                               # ['Adelie', 'Chinstrap', 'Gentoo']: the order of predict_proba
GENTOO_17 = XP_TEST[np.flatnonzero(YP_TEST == "Gentoo")[0]]           # the penguin to explain: the first Gentoo
print("le manchot expliqué :", ", ".join(f"{name} {fr(v, 1)}" for name, v in zip(PENG.features, GENTOO_17)))'''

TODO_17 = r'''def fit_logistic_17(X, y):
    """make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), fitted on (X, y), returned."""
    raise NotImplementedError("fit_logistic_17")


def fit_forest_17(X, y):
    """RandomForestClassifier(n_estimators=200, random_state=0), fitted on (X, y), returned."""
    raise NotImplementedError("fit_forest_17")


def class_output_17(model, k, kind="proba"):
    """The function f(X) -> (n,) of ONE output of a classifier: the probability of the class of index k
    (predict_proba) for kind="proba", its logit (decision_function) for kind="logit"."""
    raise NotImplementedError("class_output_17")


def shap_by_class_17(model, x, background, kind="proba"):
    """(phi, base): phi of shape (3, 4), the Shapley values of the 4 measures for each class (a row per class, in the
    order of model.classes_), base of shape (3,), with your shapley_values, one output at a time."""
    raise NotImplementedError("shap_by_class_17")'''

SOLUTION_17 = r'''def fit_logistic_17(X, y):
    """make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), fitted on (X, y), returned."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)


def fit_forest_17(X, y):
    """RandomForestClassifier(n_estimators=200, random_state=0), fitted on (X, y), returned."""
    return RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)


def class_output_17(model, k, kind="proba"):
    """The function f(X) -> (n,) of ONE output of a classifier: the probability of the class of index k
    (predict_proba) for kind="proba", its logit (decision_function) for kind="logit"."""
    if kind == "proba":
        return lambda X: model.predict_proba(X)[:, k]
    if kind == "logit":
        return lambda X: model.decision_function(X)[:, k]
    raise ValueError(f"kind must be 'proba' or 'logit', got {kind!r}")


def shap_by_class_17(model, x, background, kind="proba"):
    """(phi, base): phi of shape (3, 4), the Shapley values of the 4 measures for each class (a row per class, in the
    order of model.classes_), base of shape (3,), with your shapley_values, one output at a time."""
    results = [mylearn.explain.shapley_values(class_output_17(model, k, kind), x, background)
               for k in range(len(model.classes_))]
    return np.array([phi for phi, _ in results]), np.array([base for _, base in results])'''

CHECK_17 = r'''LR_17 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(XP_TRAIN, YP_TRAIN)   # references
RF_17 = RandomForestClassifier(n_estimators=200, random_state=0).fit(XP_TRAIN, YP_TRAIN)
G_17 = GENTOO_17[None]
with checked("B6.17"):
    lr_17 = fit_logistic_17(XP_TRAIN.copy(), YP_TRAIN.copy())
    if returned("B6.17", "fit_logistic_17", lr_17):
        steps_17 = [type(s).__name__ for _, s in getattr(lr_17, "steps", [])]
        verdict("B6.17", steps_17 == ["StandardScaler", "LogisticRegression"],
                "fit_logistic_17 : StandardScaler puis LogisticRegression.",
                f"fit_logistic_17 doit rendre le Pipeline (StandardScaler, LogisticRegression), reçu {steps_17 or type(lr_17).__name__}.")
        same("B6.17", "probabilités de fit_logistic_17 sur le test", lr_17.predict_proba(XP_TEST),
             LR_17.predict_proba(XP_TEST), rtol=1e-6, atol=1e-8)
    rf_17 = fit_forest_17(XP_TRAIN.copy(), YP_TRAIN.copy())
    if returned("B6.17", "fit_forest_17", rf_17):
        same("B6.17", "probabilités de fit_forest_17 sur le test", rf_17.predict_proba(XP_TEST), RF_17.predict_proba(XP_TEST))

LOGIT_OK_17 = False
with checked("B6.17"):
    f_p_17, f_l_17 = class_output_17(LR_17, 2, "proba"), class_output_17(LR_17, 2, "logit")
    if not (callable(f_p_17) and callable(f_l_17)):
        print("❌ Ex B6.17 : class_output_17 doit rendre une fonction f(X) -> (n,).")
        raise StopCheck
    same("B6.17", "class_output_17(LR, 2, 'proba') sur le test", f_p_17(XP_TEST), LR_17.predict_proba(XP_TEST)[:, 2])
    out_l_17 = f_l_17(XP_TEST)
    arr_l_17 = as_numpy(out_l_17)
    if (arr_l_17 is not None and arr_l_17.shape == (len(XP_TEST),)
            and np.allclose(arr_l_17, np.log(LR_17.predict_proba(XP_TEST))[:, 2], rtol=1e-6, atol=1e-8)):
        print("❌ Ex B6.17 : class_output_17(LR, 2, 'logit') rend le log de la probabilité : il diffère du logit "
              "(decision_function) d'une quantité qui change d'un manchot à l'autre, le log de la somme des "
              "exponentielles des trois logits.")
    else:
        LOGIT_OK_17 = same("B6.17", "class_output_17(LR, 2, 'logit') sur le test", out_l_17,
                           LR_17.decision_function(XP_TEST)[:, 2])

RESULTS_17 = {}
ORDERS_17 = [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]   # the other orders of the three classes


def shifted_17(got, want):
    """True when the rows of `got` are those of `want` in another order of the classes."""
    arr = as_numpy(got)
    want = np.asarray(want, dtype=float)
    return (arr is not None and arr.shape == want.shape and not np.allclose(arr, want, rtol=1e-6, atol=1e-8)
            and any(np.allclose(arr, want[order], rtol=1e-6, atol=1e-8) for order in ORDERS_17))


with checked("B6.17"):
    if not shapley_ready("B6.17"):
        raise StopCheck
    phi_l_17, base_l_17 = pair("B6.17", "shap_by_class_17", shap_by_class_17(LR_17, GENTOO_17.copy(), XP_TRAIN.copy(), "logit"),
                               names=("phi", "base"))
    if np.shape(phi_l_17) != (3, 4) or np.shape(base_l_17) != (3,):
        print(f"❌ Ex B6.17 : attendu phi de forme (3, 4) et base de forme (3,), reçu {np.shape(phi_l_17)} et {np.shape(base_l_17)}.")
        raise StopCheck
    scaler_17, logreg_17 = LR_17[0], LR_17[-1]
    linear_17 = logreg_17.coef_ / scaler_17.scale_ * (GENTOO_17 - XP_TRAIN.mean(axis=0))   # w_ki (x_i - mean_i), ∂ B6.2
    if not LOGIT_OK_17:
        print("⏳ Ex B6.17 : les valeurs de Shapley des logits attendent class_output_17(..., 'logit') (ci-dessus).")
    elif shifted_17(phi_l_17, linear_17):
        print("❌ Ex B6.17 : tes lignes sont celles d'autres classes : la ligne k doit expliquer la classe "
              "model.classes_[k] (SPECIES[k]).")
        raise StopCheck
    elif same("B6.17", "logits : phi = w_ki (x_i − moyenne du fond), le cas linéaire", phi_l_17, linear_17, rtol=1e-6, atol=1e-8):
        RESULTS_17["régression logistique, logit"] = np.asarray(phi_l_17)
    phi_p_17, base_p_17 = pair("B6.17", "shap_by_class_17", shap_by_class_17(LR_17, GENTOO_17.copy(), XP_TRAIN.copy(), "proba"),
                               names=("phi", "base"))
    if shifted_17(base_p_17, LR_17.predict_proba(XP_TRAIN).mean(axis=0)):
        print("❌ Ex B6.17 : tes lignes sont celles d'autres classes : la ligne k doit expliquer la classe "
              "model.classes_[k] (SPECIES[k]).")
        raise StopCheck
    ok_p_17 = same("B6.17", "probabilités : efficacité de chaque classe", np.sum(phi_p_17, axis=1) + base_p_17,
                   LR_17.predict_proba(G_17)[0], rtol=1e-6, atol=1e-8)
    ok_p_17 &= same("B6.17", "probabilités : valeurs de base = probabilités moyennes sur le fond", base_p_17,
                    LR_17.predict_proba(XP_TRAIN).mean(axis=0), rtol=1e-6, atol=1e-8)
    if ok_p_17:
        RESULTS_17["régression logistique, probabilité"] = np.asarray(phi_p_17)
    phi_rf_17, base_rf_17 = pair("B6.17", "shap_by_class_17", shap_by_class_17(RF_17, GENTOO_17.copy(), XP_TRAIN.copy(), "proba"),
                                 names=("phi", "base"))
    if shap is not None:                  # the whole background: TreeExplainer keeps at most 100 rows by default
        tree_17 = shap.TreeExplainer(RF_17, data=shap.maskers.Independent(XP_TRAIN, max_samples=len(XP_TRAIN)),
                                     feature_perturbation="interventional")(G_17)
        own_17 = float(np.abs(tree_17.values[0].sum(axis=0) + tree_17.base_values[0] - RF_17.predict_proba(G_17)[0]).max())
        print(f"ℹ️ forêt : TreeSHAP s'écarte lui-même de predict_proba de {own_17:.1e} (son efficacité) ; la comparaison "
              "se fait donc à 2e-3 près.")
        if same("B6.17", "forêt : tes valeurs contre TreeExplainer (interventionnel, fond complet)", phi_rf_17,
                tree_17.values[0].T, rtol=0, atol=2e-3):
            RESULTS_17["forêt aléatoire, probabilité"] = np.asarray(phi_rf_17)
if len(RESULTS_17) == 3:
    fig_17, axes_17 = plt.subplots(1, 3, figsize=(16, 3.8))
    for ax_17, (title_17, phi_17) in zip(axes_17, RESULTS_17.items()):
        for k_17, species_17 in enumerate(SPECIES):
            ax_17.bar(np.arange(4) + 0.27 * (k_17 - 1), phi_17[k_17], width=0.27, label=species_17)
        ax_17.axhline(0, color="black", lw=0.8)
        ax_17.set_xticks(range(4), [name.replace("_mm", "").replace("_g", "") for name in PENG.features], fontsize=8)
        ax_17.set_title(title_17, fontsize=10)
        ax_17.legend(fontsize=8)
    plt.suptitle("valeurs de Shapley du premier Gentoo du test, une sortie à la fois (fond : les 249 manchots d'entraînement)",
                 fontsize=10)
    plt.tight_layout()
    plt.show()
    for title_17, phi_17 in RESULTS_17.items():
        print(f"{title_17:>36} : somme sur les classes de chaque feature "
              + ", ".join(fr(0.0 if abs(v) < 5e-5 else v, 4) for v in phi_17.sum(axis=0)))'''

# =====================================================================================================================
# Part B · networks
# =====================================================================================================================
CHECK_18 = r'''with checked("B6.18"):
    E = mylearn.explain
    grad_q_18 = lambda X: np.stack([2 * X[:, 0], np.full(len(X), 3.0)], axis=1)       # f = x0**2 + 3 x1 (docstring)
    same("B6.18", "docstring : x0² + 3 x1 en (1, 1), n_steps=4, 'left'",
         E.integrated_gradients(grad_q_18, [1.0, 1.0], n_steps=4, method="left"), [0.75, 3.0])
    same("B6.18", "docstring : la même avec 'trapezoid'", E.integrated_gradients(grad_q_18, [1.0, 1.0], n_steps=4), [1.0, 3.0])

    # f = x0 x1 + x2**3 at x = (1, 2, -1) from a zero baseline: on the path a x, the gradient is (a x1, a x0, 3 a² x2²)
    grad_c_18 = lambda X: np.stack([X[:, 1], X[:, 0], 3 * X[:, 2] ** 2], axis=1)
    x_18 = np.array([1.0, 2.0, -1.0])
    k_18 = np.arange(7) / 6                                       # n = 6 intervals
    means_18 = {"left": (k_18[:-1].mean(), (k_18[:-1] ** 2).mean()), "right": (k_18[1:].mean(), (k_18[1:] ** 2).mean()),
                "midpoint": (((np.arange(6) + 0.5) / 6).mean(), (((np.arange(6) + 0.5) / 6) ** 2).mean()),
                "trapezoid": (0.5, 1 / 3 + 1 / (6 * 36))}         # the means of a and a² for each rule
    for method_18, (m1_18, m2_18) in means_18.items():
        want_18 = np.array([x_18[0] * x_18[1] * m1_18, x_18[1] * x_18[0] * m1_18, x_18[2] * 3 * x_18[2] ** 2 * m2_18])
        same("B6.18", f"x0 x1 + x2³ en (1, 2, −1), n_steps=6, '{method_18}'",
             E.integrated_gradients(grad_c_18, x_18.copy(), n_steps=6, method=method_18), want_18)
    shapes_18 = []

    def spy_18(X):
        shapes_18.append(np.shape(X))
        return grad_c_18(np.asarray(X, dtype=float))

    E.integrated_gradients(spy_18, x_18, n_steps=1, method="left")
    verdict("B6.18", shapes_18 and all(len(s) == 2 and s[1] == 3 for s in shapes_18),
            "grad_fn ne reçoit que des batches (n, 3), même avec n_steps=1.",
            f"grad_fn a reçu les formes {shapes_18} : elle attend un batch (n, *x.shape), même pour un seul point.")
    sig_18 = lambda X: (lambda s: np.stack([s * (1 - s) * X[:, 1], s * (1 - s) * X[:, 0]], axis=1))(
        1 / (1 + np.exp(-(X[:, 0] * X[:, 1] - 2))))                # the sigmoid of 206.5: sigma(x0 x1 - 2)
    gap_18 = 1 / (1 + np.exp(-4.0)) - 1 / (1 + np.exp(2.0))
    total_18 = float(np.sum(E.integrated_gradients(sig_18, [2.0, 3.0], n_steps=200)))
    verdict("B6.18", abs(total_18 - gap_18) < 1e-4, f"complétude sur la sigmoïde de 206.5 (200 intervalles) : somme {fr(total_18, 5)}.",
            f"complétude sur la sigmoïde de 206.5 : la somme vaut {total_18:.5g}, attendu f(x) − f(x′) ≈ {gap_18:.5g}.")
    ig_int_18 = E.integrated_gradients(grad_q_18, np.array([1, 1]), n_steps=4)
    verdict("B6.18", np.asarray(ig_int_18).dtype.kind == "f", "un x entier donne des attributions en float.",
            f"un x entier doit donner des attributions en float, reçu {np.asarray(ig_int_18).dtype}.")
    refuses("B6.18", "n_steps=0", lambda: E.integrated_gradients(grad_q_18, [1.0, 1.0], n_steps=0))
    refuses("B6.18", "method='riemann_trapezoid' (un nom de captum)",
            lambda: E.integrated_gradients(grad_q_18, [1.0, 1.0], method="riemann_trapezoid"))
    refuses("B6.18", "une ligne de référence de forme (1,) pour x de forme (2,)",
            lambda: E.integrated_gradients(grad_q_18, [1.0, 1.0], baseline=[0.0]))

with checked("B6.18"):
    E = mylearn.explain
    A_18 = np.array([[[[1.0, 2.0], [0.0, 1.0]], [[0.0, 1.0], [3.0, 0.0]]]])           # the docstring: N=1, C=2, 2 x 2
    G_18 = np.array([[[[1.0, 1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, -1.0]]]])
    copies_18 = (A_18.copy(), G_18.copy())
    same("B6.18", "docstring : grad_cam(A, G, relu=False, normalize=False)", E.grad_cam(A_18, G_18, relu=False, normalize=False),
         [[[1.0, 1.0], [-3.0, 1.0]]])
    same("B6.18", "docstring : grad_cam(A, G)", E.grad_cam(A_18, G_18), [[[1.0, 1.0], [0.0, 1.0]]])
    verdict("B6.18", np.array_equal(A_18, copies_18[0]) and np.array_equal(G_18, copies_18[1]),
            "les activations et les gradients reçus n'ont pas changé.", "grad_cam a modifié ses entrées.")
    rng_18 = np.random.default_rng(18)
    A2_18, G2_18 = rng_18.random((3, 4, 5, 5)), rng_18.normal(size=(3, 4, 5, 5))
    want_raw_18 = np.einsum("nc,nchw->nhw", G2_18.mean(axis=(2, 3)), A2_18)
    want_18 = np.maximum(want_raw_18, 0)
    peak_18 = want_18.max(axis=(1, 2), keepdims=True)
    same("B6.18", "3 images de 4 canaux 5 × 5 : ReLU de la somme pondérée, chaque carte ÷ son maximum",
         E.grad_cam(A2_18, G2_18), np.divide(want_18, peak_18, out=np.zeros_like(want_18), where=peak_18 > 0))
    same("B6.18", "les mêmes, gradients multipliés par 7 (même carte)", E.grad_cam(A2_18, 7 * G2_18),
         np.divide(want_18, peak_18, out=np.zeros_like(want_18), where=peak_18 > 0))
    same("B6.18", "relu=False : dans [−1, 1], ÷ le plus grand |.|", E.grad_cam(A2_18, G2_18, relu=False),
         want_raw_18 / np.abs(want_raw_18).max(axis=(1, 2), keepdims=True))
    same("B6.18", "gradients nuls : une carte nulle, sans NaN", E.grad_cam(A2_18, np.zeros_like(G2_18)), np.zeros((3, 5, 5)))
    refuses("B6.18", "activations et gradients de formes différentes", lambda: E.grad_cam(A2_18, G2_18[:, :3]))
    refuses("B6.18", "des tableaux 3-D", lambda: E.grad_cam(A2_18[0], G2_18[0]))'''

TOOLS_B = r'''# Tools of part B: CIFAR-10 and the small CNN of B6.19 (the ResNet of B6.20 is loaded there)
import torch
import torch.nn.functional as F

CLASSES = wb.datasets.CIFAR10_CLASSES
X_CIFAR, Y_CIFAR = wb.datasets.load_cifar10_subset(50, "test")   # 500 test images, uint8 (N, 32, 32, 3), int labels
CNN, CNN_INFO = wbx.cifar_cnn(FAST_MODE)   # trained once (≈ 40 s on CPU), then read back; frozen, in eval() mode


def show_rows(rows, row_titles, col_titles, cmaps, size=1.7, suptitle=None):
    """A grid of images: rows[r][c] is an image (H, W, 3) in [0, 1] or a map (H, W); cmaps[c] the colour map of the
    column c ("gray" for the images is ignored), None for an image."""
    nrow, ncol = len(rows), len(rows[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow + (0.5 if suptitle else 0.1)), squeeze=False)
    for r in range(nrow):
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(rows[r][c], cmap=cmaps[c], vmin=None if cmaps[c] is None else 0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
        axes[r, 0].set_ylabel(row_titles[r], fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.show()


print(f"CIFAR-10 : {len(X_CIFAR)} images de test (50 par classe). Petit CNN de référence : "
      f"{sum(p.numel() for p in CNN.parameters())} paramètres, accuracy de test {pct(CNN_INFO['test_accuracy'], 1)}, "
      "entrée : des pixels dans [0, 1], (N, 3, 32, 32).")'''

# --- 📦 B6.19 ---------------------------------------------------------------------------------------------------------
GIVEN_19 = r'''IMAGES_19 = wbx.pixels(X_CIFAR[:8])               # 8 test images, float32 tensor (8, 3, 32, 32) in [0, 1]
LABELS_19 = Y_CIFAR[:8]
PRED_19 = wbx.predict(CNN, IMAGES_19).argmax(axis=1)  # the classes the CNN predicts (computed without your code)


def blurred_19(x):
    """A blurred copy of images (N, 3, 32, 32): the mean over a 7 x 7 window around each pixel."""
    return F.avg_pool2d(torch.as_tensor(x), 7, stride=1, padding=3, count_include_pad=False)


print("classes vraies :", [CLASSES[k] for k in LABELS_19])
print("classes prédites :", [CLASSES[k] for k in PRED_19])'''

TODO_19 = r'''def class_gradient_19(model, images, target):
    """np.ndarray (N, 3, 32, 32): the gradient of the logit of the class target[n] of each image with respect to its
    pixels. images: a float tensor or array (N, 3, 32, 32) in [0, 1]; target: one class per image (N ints)."""
    raise NotImplementedError("class_gradient_19")


def saliency_19(gradients):
    """(N, 32, 32): the saliency map, the largest |gradient| over the 3 channels of each pixel."""
    raise NotImplementedError("saliency_19")


def grad_times_input_19(gradients, images):
    """(N, 32, 32): gradient × input, summed over the 3 channels of each pixel (signed)."""
    raise NotImplementedError("grad_times_input_19")


def integrated_19(model, image, target, baseline, n_steps=50):
    """(3, 32, 32): the integrated gradients of the logit of the class `target` (an int) for ONE image (3, 32, 32),
    from `baseline` (3, 32, 32), with your integrated_gradients (rule 'trapezoid') and a grad_fn built on
    class_gradient_19."""
    raise NotImplementedError("integrated_19")'''

SOLUTION_19 = r'''def class_gradient_19(model, images, target):
    """np.ndarray (N, 3, 32, 32): the gradient of the logit of the class target[n] of each image with respect to its
    pixels. images: a float tensor or array (N, 3, 32, 32) in [0, 1]; target: one class per image (N ints)."""
    x = torch.as_tensor(np.asarray(images), dtype=torch.float32).clone().requires_grad_(True)
    t = torch.as_tensor(np.asarray(target)).long().reshape(-1)
    scores = model(x).gather(1, t[:, None])          # (N, 1): the logit of the class of each image
    scores.sum().backward()                          # each logit depends on its own image only
    return x.grad.numpy()


def saliency_19(gradients):
    """(N, 32, 32): the saliency map, the largest |gradient| over the 3 channels of each pixel."""
    return np.abs(np.asarray(gradients)).max(axis=1)


def grad_times_input_19(gradients, images):
    """(N, 32, 32): gradient × input, summed over the 3 channels of each pixel (signed)."""
    return (np.asarray(gradients) * np.asarray(images)).sum(axis=1)


def integrated_19(model, image, target, baseline, n_steps=50):
    """(3, 32, 32): the integrated gradients of the logit of the class `target` (an int) for ONE image (3, 32, 32),
    from `baseline` (3, 32, 32), with your integrated_gradients (rule 'trapezoid') and a grad_fn built on
    class_gradient_19."""
    grad_fn = lambda points: class_gradient_19(model, points, [target] * len(points))
    return mylearn.explain.integrated_gradients(grad_fn, np.asarray(image, dtype=float), np.asarray(baseline, dtype=float),
                                                n_steps=n_steps, method="trapezoid")'''

CHECK_19 = r'''ok_19 = {}
with checked("B6.19"):
    x_19 = IMAGES_19[:4]
    t_19 = np.array([(p + 3) % 10 for p in PRED_19[:4]])     # classes that the CNN does NOT predict
    want_19 = wbx.input_gradients(CNN, x_19, t_19)
    g_19 = class_gradient_19(CNN, x_19.clone(), t_19.copy())
    if CNN.training or any(p.requires_grad for p in CNN.parameters()):
        print("❌ Ex B6.19 : après class_gradient_19, le modèle n'est plus gelé ou n'est plus en mode eval() (ses "
              "batchnorms changent alors le calcul) : ta fonction ne doit pas le changer.")
        raise StopCheck
    if not returned("B6.19", "class_gradient_19", g_19):
        raise StopCheck
    arr_19 = np.asarray(g_19.detach() if torch.is_tensor(g_19) else g_19, dtype=float)
    if arr_19.shape != want_19.shape:
        print(f"❌ Ex B6.19 : class_gradient_19 doit rendre un tableau (4, 3, 32, 32), reçu la forme {arr_19.shape}.")
        raise StopCheck
    if not np.allclose(arr_19, want_19, rtol=1e-4, atol=1e-6):
        probe_19 = x_19.clone().requires_grad_(True)
        probs_19 = torch.softmax(CNN(probe_19), dim=1)
        (g_prob_19,) = torch.autograd.grad(probs_19.gather(1, torch.as_tensor(t_19)[:, None]).sum(), probe_19)
        if np.allclose(arr_19, wbx.input_gradients(CNN, x_19, PRED_19[:4]), rtol=1e-4, atol=1e-6):
            print("❌ Ex B6.19 : c'est le gradient de la classe prédite : le contrat veut celui de target[n], qui n'est "
                  "pas forcément la prédite.")
        elif np.allclose(arr_19, g_prob_19.numpy(), rtol=1e-4, atol=1e-6):
            print("❌ Ex B6.19 : c'est le gradient de la probabilité (après la softmax) : le contrat veut celui du logit.")
        elif np.allclose(arr_19, sum(wbx.input_gradients(CNN, x_19, k) for k in range(10)), rtol=1e-4, atol=1e-5):
            print("❌ Ex B6.19 : c'est le gradient de la somme des 10 logits : ne garde que celui de target[n].")
        else:
            same("B6.19", "class_gradient_19 contre l'autograd de référence", arr_19, want_19, rtol=1e-4, atol=1e-6)
        raise StopCheck
    print("✅ Ex B6.19 : class_gradient_19 : le gradient du logit de la classe demandée, image par image.")
    again_19 = np.asarray(class_gradient_19(CNN, x_19.clone(), t_19.copy()), dtype=float)
    verdict("B6.19", np.allclose(again_19, arr_19, rtol=1e-6, atol=1e-8), "un second appel rend le même gradient (rien ne s'accumule).",
            "un second appel ne rend pas le même gradient : un .grad s'accumule-t-il d'un appel à l'autre ?")
    print("✅ Ex B6.19 : le modèle reste gelé, en mode eval().")
    ok_19["grad"] = True
CNN.eval().requires_grad_(False)                       # restored for the cells that follow, whatever your code did

with checked("B6.19"):
    if not ok_19:
        raise StopCheck
    s_19 = saliency_19(want_19.copy())
    arr_s_19 = as_numpy(s_19)
    if arr_s_19 is not None and arr_s_19.shape == (4, 32, 32) and np.allclose(arr_s_19, np.abs(want_19).mean(axis=1)):
        print("❌ Ex B6.19 : c'est la moyenne des |gradients| des canaux : la carte de saillance de 206.5 prend le maximum.")
    elif same("B6.19", "saliency_19 : le plus grand |gradient| des 3 canaux", s_19, np.abs(want_19).max(axis=1)):
        ok_19["saliency"] = True
    gi_19 = grad_times_input_19(want_19.copy(), x_19.numpy().copy())
    if same("B6.19", "grad_times_input_19 : somme sur les canaux de gradient × pixel", gi_19, (want_19 * x_19.numpy()).sum(axis=1),
            rtol=1e-6, atol=1e-8):
        ok_19["gxi"] = True

with checked("B6.19"):
    if not ok_19.get("grad"):
        raise StopCheck
    ig_19 = integrated_19(CNN, IMAGES_19[0].numpy(), int(LABELS_19[0]), np.zeros((3, 32, 32)), n_steps=20)
    if not returned("B6.19", "integrated_19", ig_19):
        raise StopCheck
    ref_ig_19 = wbx.integrated_gradients(CNN, IMAGES_19[:1], [int(LABELS_19[0])], n_steps=20)[0]
    arr_ig_19 = as_numpy(ig_19)
    if (arr_ig_19 is not None and arr_ig_19.shape == ref_ig_19.shape and not np.allclose(arr_ig_19, ref_ig_19, rtol=1e-4, atol=1e-6)
            and np.allclose(arr_ig_19 * IMAGES_19[0].numpy(), ref_ig_19, rtol=1e-4, atol=1e-6)):
        print("❌ Ex B6.19 : integrated_19 rend la moyenne des gradients sur le chemin : les gradients intégrés la "
              "multiplient par (x − x′).")
    elif same("B6.19", "integrated_19 (image noire, 20 intervalles) contre la référence", ig_19, ref_ig_19, rtol=1e-4, atol=1e-6):
        ok_19["ig"] = True
    else:
        print("ℹ️ Ex B6.19 : la ligne de référence est ici l'image noire, le tenseur nul (le CNN reçoit des pixels dans "
              "[0, 1] et se normalise lui-même), et la règle celle des trapèzes, en 20 intervalles.")'''

EXPERIMENT_19 = r'''if len(ok_19) == 4:
    start_19 = time.perf_counter()
    black_19, blur_19 = torch.zeros_like(IMAGES_19), blurred_19(IMAGES_19)
    grads_19 = class_gradient_19(CNN, IMAGES_19, LABELS_19)
    sal_19, gxi_19 = saliency_19(grads_19), grad_times_input_19(grads_19, IMAGES_19.numpy())
    ig_black_19 = np.stack([integrated_19(CNN, IMAGES_19[i].numpy(), int(LABELS_19[i]), black_19[i].numpy())
                            for i in range(6)])
    ig_blur_19 = np.stack([integrated_19(CNN, IMAGES_19[i].numpy(), int(LABELS_19[i]), blur_19[i].numpy())
                           for i in range(6)])
    rows_19 = []
    for i in range(6):
        rows_19.append([IMAGES_19[i].permute(1, 2, 0).numpy(), sal_19[i], np.abs(gxi_19[i]),
                        np.abs(ig_black_19[i]).sum(axis=0), blur_19[i].permute(1, 2, 0).numpy(),
                        np.abs(ig_blur_19[i]).sum(axis=0)])
    show_rows(rows_19, [f"{CLASSES[LABELS_19[i]]} (prédit {CLASSES[PRED_19[i]]})" for i in range(6)],
              ["image", "saillance", "|gradient × entrée|", "|IG|, réf. noire", "réf. floutée", "|IG|, réf. floutée"],
              [None, "magma", "magma", "magma", None, "magma"],
              suptitle="attributions du logit de la vraie classe (IG : somme des |attributions| sur les 3 canaux)")
    with torch.no_grad():
        logit_19 = lambda x: CNN(x).gather(1, torch.as_tensor(LABELS_19[:6])[:, None])[:, 0].double().numpy()
        f_x_19, f_black_19, f_blur_19 = logit_19(IMAGES_19[:6]), logit_19(black_19[:6]), logit_19(blur_19[:6])
    print("complétude (somme des attributions contre f(x) − f(x′), logit de la vraie classe) :")
    for i in range(6):
        print(f"  {CLASSES[LABELS_19[i]]:>10} : réf. noire {fr(ig_black_19[i].sum(), 3)} contre {fr(f_x_19[i] - f_black_19[i], 3)} ; "
              f"réf. floutée {fr(ig_blur_19[i].sum(), 3)} contre {fr(f_x_19[i] - f_blur_19[i], 3)} ; "
              f"somme de gradient × entrée {fr(gxi_19[i].sum(), 3)}")
    print(f"ℹ️ {time.perf_counter() - start_19:.1f} s")
else:
    print("⏳ Ex B6.19 : l'expérience attend tes quatre fonctions (cellule précédente).")'''

# --- 📦 B6.20 ---------------------------------------------------------------------------------------------------------
TODO_20 = r'''B1_FOLDER = wbx.find_b1_folder()   # your model published in Ex B1.27, if it is there (None: the reference ResNet)


def activations_and_gradients_20(model, layer, images, target):
    """(A, G): np.ndarray (N, C, H, W) twice, the output A of `layer` (a module of model) for the prepared batch
    `images`, and the gradient G of the sum over the images of the logit of target[n] with respect to A; two hooks
    (one on the module, one on its output tensor), removed before returning."""
    raise NotImplementedError("activations_and_gradients_20")


def cam_20(model, layer, images, target, size):
    """(N, size, size): the Grad-CAM maps of your grad_cam (relu and normalize by default), resized to size x size by
    F.interpolate(..., mode="bilinear", align_corners=False)."""
    raise NotImplementedError("cam_20")'''

SOLUTION_20 = r'''B1_FOLDER = None   # the solutions notebook never reads the learner's model: the reference ResNet


def activations_and_gradients_20(model, layer, images, target):
    """(A, G): np.ndarray (N, C, H, W) twice, the output A of `layer` (a module of model) for the prepared batch
    `images`, and the gradient G of the sum over the images of the logit of target[n] with respect to A; two hooks
    (one on the module, one on its output tensor), removed before returning."""
    store = {}

    def save(module, inputs, output):                 # returns None: the output is left unchanged
        store["A"] = output.detach()
        output.register_hook(lambda grad: store.__setitem__("G", grad.detach()))

    x = torch.as_tensor(images, dtype=torch.float32).clone().requires_grad_(True)   # frozen model: the input needs grad
    t = torch.as_tensor(np.asarray(target)).long().reshape(-1)
    handle = layer.register_forward_hook(save)
    try:
        scores = model(x)
        scores.gather(1, t[:, None]).sum().backward()
    finally:
        handle.remove()
    return store["A"].numpy(), store["G"].numpy()


def cam_20(model, layer, images, target, size):
    """(N, size, size): the Grad-CAM maps of your grad_cam (relu and normalize by default), resized to size x size by
    F.interpolate(..., mode="bilinear", align_corners=False)."""
    A, G = activations_and_gradients_20(model, layer, images, target)
    cams = torch.as_tensor(mylearn.explain.grad_cam(A, G), dtype=torch.float32)
    return F.interpolate(cams[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()'''

LOAD_20 = r'''RESNET, RESNET_INFO = wbx.try_b1_resnet(FAST_MODE, folder=B1_FOLDER)   # frozen, in eval() mode
SIZE_20 = RESNET_INFO["size"]                                          # the side of its input images (96 in fast mode)
POOL_20 = X_CIFAR[:200], Y_CIFAR[:200]                                 # 200 test images (20 per class)
if wbx.is_missing(RESNET):                     # no network the first time (⏳ above): the cells of RESNET say ⏳
    PRED_20 = RIGHT_20 = WRONG_20 = X_20 = None
else:
    PRED_20 = wbx.predict(RESNET, POOL_20[0], RESNET_INFO).argmax(axis=1)
    RIGHT_20 = np.flatnonzero(PRED_20 == POOL_20[1])[:8]               # 8 well-classified images
    WRONG_20 = np.flatnonzero(PRED_20 != POOL_20[1])[:8]               # 8 errors
    X_20 = wbx.model_input(POOL_20[0][np.concatenate([RIGHT_20, WRONG_20])], RESNET_INFO)   # (16, 3, SIZE_20, SIZE_20)
    print(f"ResNet-18 {'de référence' if RESNET_INFO['source'] == 'reference' else 'du B1 (le tien)'} : {SIZE_20} pixels, layer4 en {SIZE_20 // 32} × {SIZE_20 // 32} ; accuracy sur "
          f"les 200 images : {pct(np.mean(PRED_20 == POOL_20[1]), 1)} ; 8 bonnes prédictions et 8 erreurs retenues.")'''

CHECK_20 = r'''ok_20 = {}
wbx.clear_hooks(RESNET)                                # hooks left on RESNET by an earlier run of your code
with checked("B6.20"):
    x_20 = X_20[[0, 1, 8, 9]]
    true_20 = POOL_20[1][np.concatenate([RIGHT_20, WRONG_20])][[0, 1, 8, 9]]
    t_20 = np.array([true_20[0], (true_20[1] + 1) % 10, true_20[2], true_20[3]])      # true classes, and one other
    A_ref_20, G_ref_20, _ = wbx.activations_and_gradients(RESNET, RESNET.layer4, x_20, t_20)
    try:
        got_20 = activations_and_gradients_20(RESNET, RESNET.layer4, x_20.clone(), t_20.copy())
    except RuntimeError as exc_20:
        if "require" in str(exc_20) and "grad" in str(exc_20):
            print("❌ Ex B6.20 : ton code lève RuntimeError : " + str(exc_20)[:160] + " ; deux causes possibles : les "
                  "poids du modèle sont gelés (requires_grad=False), relis le dernier détail de « Avec PyTorch : deux "
                  "hooks » (§206.6) ; ou ton hook de passe avant rend une valeur, qui remplace la sortie du module, et "
                  "un tenseur détaché coupe le graphe (le hook ne rend rien, 206.6).")
            raise StopCheck
        raise
    A_20, G_20 = pair("B6.20", "activations_and_gradients_20", got_20, names=("A", "G"))
    left_20 = wbx.leftover_hooks(RESNET)
    if left_20:
        print("❌ Ex B6.20 : un hook reste posé sur RESNET après l'appel ("
              + ", ".join(f"{name or 'le modèle'} : {', '.join(kinds)}" for name, kinds in left_20.items())
              + ") : retire chaque hook de module (handle.remove(), dans un finally) ; sinon chaque passe suivante le "
              "déclenche encore.")
        wbx.clear_hooks(RESNET)
        raise StopCheck
    if same("B6.20", "A : la sortie de layer4", A_20, A_ref_20, rtol=1e-5, atol=1e-6):
        ok_20["A"] = True
    if np.shape(G_20) == G_ref_20.shape and not np.allclose(np.asarray(G_20, dtype=float), G_ref_20, rtol=1e-4, atol=1e-7):
        pred_t_20 = wbx.predict(RESNET, x_20).argmax(axis=1)
        _, G_pred_20, _ = wbx.activations_and_gradients(RESNET, RESNET.layer4, x_20, pred_t_20)
        if np.allclose(np.asarray(G_20, dtype=float), G_pred_20, rtol=1e-4, atol=1e-7):
            print("❌ Ex B6.20 : G est le gradient de la classe prédite : le contrat veut celui de target[n] "
                  "(l'image 2 reçoit une autre classe).")
            raise StopCheck
    if same("B6.20", "G : le gradient du logit de target[n] par rapport à A", G_20, G_ref_20, rtol=1e-4, atol=1e-7):
        ok_20["G"] = True
RESNET.eval().requires_grad_(False)                    # restored for the cells that follow, whatever your code did
wbx.clear_hooks(RESNET)

with checked("B6.20"):
    if len(ok_20) < 2:
        raise StopCheck
    cams_20 = cam_20(RESNET, RESNET.layer4, x_20.clone(), t_20.copy(), SIZE_20)
    want_cam_20 = wbx.grad_cam(RESNET, RESNET.layer4, x_20, t_20, size=SIZE_20)
    arr_cam_20 = np.asarray(cams_20, dtype=float)
    if arr_cam_20.shape == want_cam_20.shape and not np.allclose(arr_cam_20, want_cam_20, rtol=1e-4, atol=1e-5):
        small_20 = torch.as_tensor(wbx.grad_cam(RESNET, RESNET.layer4, x_20, t_20))[:, None].float()
        corners_20 = F.interpolate(small_20, size=(SIZE_20, SIZE_20), mode="bilinear", align_corners=True)[:, 0].numpy()
        if np.allclose(arr_cam_20, corners_20, rtol=1e-4, atol=1e-5):
            print("❌ Ex B6.20 : la carte est agrandie avec align_corners=True : les images le sont avec "
                  "align_corners=False (wb.transfer.to_input) ; avec deux conventions, la carte est décalée sur l'image.")
            raise StopCheck
        if arr_cam_20.min() < -1e-9:
            print("❌ Ex B6.20 : la carte a des valeurs négatives : grad_cam garde-t-il sa ReLU (relu=True par défaut) ?")
            raise StopCheck
        nearest_20 = F.interpolate(small_20, size=(SIZE_20, SIZE_20), mode="nearest")[:, 0].numpy()
        if np.allclose(arr_cam_20, nearest_20, rtol=1e-4, atol=1e-5):
            print("❌ Ex B6.20 : la carte est agrandie par plus proche voisin (des blocs) : le contrat veut "
                  "mode=\"bilinear\", align_corners=False, la convention qui a agrandi les images.")
            raise StopCheck
        wbx.clear_hooks(RESNET)
        if np.allclose(arr_cam_20, wbx.grad_cam(RESNET, RESNET.layer3, x_20, t_20, size=SIZE_20), rtol=1e-4, atol=1e-5):
            print("❌ Ex B6.20 : ce sont les cartes de layer3 : cam_20 doit pondérer les activations du module `layer` "
                  "qu'elle reçoit.")
            raise StopCheck
    if same("B6.20", f"cam_20 : cartes agrandies à {SIZE_20} × {SIZE_20}", arr_cam_20, want_cam_20, rtol=1e-4, atol=1e-5):
        ok_20["cam"] = True
_ = wbx.clear_hooks(RESNET)                            # no hook stays on RESNET for the cells that follow'''

EXPERIMENT_20 = r'''if len(ok_20) == 3:
    labels_20 = POOL_20[1][np.concatenate([RIGHT_20, WRONG_20])]
    preds_20 = PRED_20[np.concatenate([RIGHT_20, WRONG_20])]
    cam_pred_20 = cam_20(RESNET, RESNET.layer4, X_20, preds_20, SIZE_20)        # the predicted class
    cam_true_20 = cam_20(RESNET, RESNET.layer4, X_20, labels_20, SIZE_20)       # the true class (the same when right)
    shown_20 = wbx.model_input(POOL_20[0][np.concatenate([RIGHT_20, WRONG_20])], {"size": SIZE_20, "mean": (0, 0, 0),
                                                                                  "std": (1, 1, 1)})
    pics_20 = shown_20.permute(0, 2, 3, 1).clamp(0, 1).numpy()                  # the images at SIZE_20 pixels, in [0, 1]

    def overlay_20(image, cam):
        heat = plt.cm.jet(cam)[..., :3]
        return np.clip(0.55 * image + 0.45 * heat, 0, 1)

    def corr_20(a, b):
        """The correlation of two maps, or None when one of them is constant (all zero after the ReLU)."""
        a, b = a.ravel(), b.ravel()
        return None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])

    rows_20 = [[pics_20[i], overlay_20(pics_20[i], cam_pred_20[i])] for i in range(8)]
    show_rows(rows_20, [f"{CLASSES[labels_20[i]]}" for i in range(8)], ["image", "Grad-CAM (prédite)"],
              [None, None], size=1.6, suptitle="bonnes prédictions : Grad-CAM de la classe prédite (layer4)")
    rows_w_20 = [[pics_20[i], overlay_20(pics_20[i], cam_pred_20[i]), overlay_20(pics_20[i], cam_true_20[i])]
                 for i in range(8, 16)]
    show_rows(rows_w_20, [f"{CLASSES[labels_20[i]]} → {CLASSES[preds_20[i]]}" for i in range(8, 16)],
              ["image", "classe prédite", "vraie classe"], [None, None, None], size=1.6,
              suptitle="erreurs (vraie → prédite) : Grad-CAM de la classe prédite, puis de la vraie classe")
    same_20 = [corr_20(cam_pred_20[i], cam_true_20[i]) for i in range(8, 16)]
    print("erreurs : corrélation entre la carte de la classe prédite et celle de la vraie classe :",
          ", ".join("carte nulle" if v is None else fr(v, 2) for v in same_20))
    empty_20 = [f"{CLASSES[labels_20[i]]} ({which})" for i in range(16)
                for which, cams in (("classe prédite", cam_pred_20), ("vraie classe", cam_true_20)) if cams[i].max() == 0]
    if empty_20:
        print("cartes nulles (la ReLU a tout éteint) :", ", ".join(empty_20))
else:
    print("⏳ Ex B6.20 : l'expérience attend tes deux fonctions (cellules précédentes).")'''


def needs_resnet(code: str, ex_id: str) -> str:
    """A cell that needs RESNET, wrapped: while the ResNet could not be loaded (no network the first time, a
    wbx.MissingModel), it prints a ⏳ line instead of running (the behaviour of chapter B1)."""
    body = "\n".join(("    " + line) if line.strip() else "" for line in code.splitlines())
    message = (f"⏳ Ex {ex_id} : cette cellule attend RESNET, le ResNet-18 de B6.20, qui n'a pas pu être chargé (la "
               "cellule de chargement de B6.20 dit pourquoi) : il faut le réseau une première fois, ou Colab.")
    return f"if wbx.is_missing(RESNET):\n    print({message!r})\nelse:\n{body}"


CHECK_20 = needs_resnet(CHECK_20, "B6.20")
EXPERIMENT_20 = needs_resnet(EXPERIMENT_20, "B6.20")
