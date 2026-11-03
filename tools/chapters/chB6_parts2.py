"""End of part B (B6.21, B6.22) and part C (fairness, B6.23 to B6.29) of the chapter B6 notebook, second half:
statements and cells (used by build_chB6.py).

B6.21 and B6.22 are appended to PART_B.exercises (chB6_parts.py); part C is appended to PARTS. The answers of 📈
B6.25, the predictions of 🔮 B6.26 and the goal of 🏆 B6.29 come from tools/chapters/history_chB6.json (entries
"t25", "p26", "c29").
"""

from __future__ import annotations

import json

from chapter_kit import Ex, Part, guarded

import chB6_code as C
import chB6_code2 as C2
from chB6_parts import HISTORY, PART_B, PARTS, letters_record, mylearn_howto, tests_after


def fill(text: str, values: dict) -> str:
    """Replace each «NAME» of a cell by its value; every placeholder must be filled."""
    for key, value in values.items():
        text = text.replace(f"«{key}»", str(value))
    if "«" in text:
        raise ValueError(f"unfilled placeholder in {text[text.index('«'):text.index('«') + 30]!r}")
    return text


def fr_num(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


# ---------------------------------------------------------------------------------------------------------------------
# History: the table of B6.25, the predictions of B6.26, the goal of B6.29
# ---------------------------------------------------------------------------------------------------------------------
T25 = HISTORY.get("t25")
P26 = HISTORY.get("p26", {}).get("answers", {"proxy_auc": "D", "proxy_auc_few": "C", "eo_ratio": "B", "four_fifths": "A"})
C29 = HISTORY.get("c29", {"eo": [0.0, 0.05], "loss": [0.0, 0.02], "rows": [], "shuffled_balanced": 0.5})
AGE_LETTERS = {"<25": "A", "25-34": "B", "35-44": "C", "45-54": "D", "55-64": "E", "65+": "F"}


def table_25():
    """The rows of the table of B6.25 (age bands, then the sexes) and of its gaps, as Python literals, and the answers."""
    if T25 is None:
        return "{}", "{}", {"A25": "A", "B25": 0, "C25": "A", "D25": True, "E25": True, "F25": "A"}
    rows = {"groupe": [], "n": [], "positifs": [], "taux de base": [], "sélection [IC 95 %]": [], "TPR [IC 95 %]": [],
            "FPR [IC 95 %]": []}
    ci = lambda r, k, d=3: f"{fr_num(r[k], d)} [{fr_num(r[k + '_low'], d)} ; {fr_num(r[k + '_high'], d)}]"
    order = {"age": list(AGE_LETTERS), "sex": ["Female", "Male"]}            # the age bands in their natural order
    for part in ("age", "sex"):
        for group in order[part]:
            r = T25["tables"][part][group]
            rows["groupe"].append({"Female": "femmes", "Male": "hommes"}.get(group, group))
            rows["n"].append(r["n"])
            rows["positifs"].append(r["positives"])
            rows["taux de base"].append(fr_num(r["base_rate"], 3))
            rows["sélection [IC 95 %]"].append(ci(r, "selection_rate"))
            rows["TPR [IC 95 %]"].append(ci(r, "tpr"))
            rows["FPR [IC 95 %]"].append(ci(r, "fpr"))
    names = {"dp": "ΔDP (sélection)", "eo": "ΔEO", "tpr": "écart de TPR", "fpr": "écart de FPR"}
    gaps = {"écart": [], "par sexe [IC 95 %]": [], "par âge [IC 95 %]": []}
    for kind, label in names.items():
        gaps["écart"].append(label)
        for part, column in (("sex", "par sexe [IC 95 %]"), ("age", "par âge [IC 95 %]")):
            g, lo, hi = T25["gaps"][f"{part}_{kind}"]
            gaps[column].append(f"{fr_num(g, 3)} [{fr_num(lo, 3)} ; {fr_num(hi, 3)}]")
    age = T25["tables"]["age"]
    lowest = min(age, key=lambda k: age[k]["tpr"])
    widest = max((k for k in age if k != lowest), key=lambda k: age[k]["tpr_high"] - age[k]["tpr_low"])
    sex = T25["tables"]["sex"]
    overlap = sex["Female"]["tpr_high"] >= sex["Male"]["tpr_low"] and sex["Male"]["tpr_high"] >= sex["Female"]["tpr_low"]
    answers = {"A25": AGE_LETTERS[lowest], "B25": age[lowest]["positives"], "C25": AGE_LETTERS[widest],
               "D25": T25["gaps"]["sex_tpr"][1] > 0, "E25": bool(overlap), "F25": "C"}
    return json.dumps(rows, ensure_ascii=False), json.dumps(gaps, ensure_ascii=False), answers


ROWS_25, GAPS_25, A25 = table_25()
WIDTH_25 = ({k: r["tpr_high"] - r["tpr_low"] for k, r in T25["tables"]["age"].items()} if T25 else {})

GOAL_29 = 0.07                       # the title of the contract (0.05 at first: the worst split of the reference gave 0.058)
CONTROL_29 = 1                       # wbx.adult(seed=1): another test part (one of the calibration splits)
REF_29 = {"eo": [round(v, 4) for v in C29["eo"]], "loss": [round(v, 4) for v in C29["loss"]]}
if C29["rows"] and not (C29["eo"][1] + 0.01 <= GOAL_29 and C29["loss"][1] + 0.005 <= 0.02):
    raise SystemExit(f"B6.29: the reference does not hold the goal with a margin: {C29['eo']}, {C29['loss']}")

# ---------------------------------------------------------------------------------------------------------------------
EX_21 = Ex("B6.21", "🐛", 2, 25, "Grad-CAM à l'envers : mauvaise couche, mauvaise classe, ReLU oubliée, carte mal alignée",
           "trouver dans le code, par des diagnostics enchaînés, les quatre erreurs classiques d'un Grad-CAM.",
           "Ex B6.20 · ✏️ B6.4 · fiche §206.6 (« L'idée », « Avec PyTorch : deux hooks ») et les pièges de la fiche",
           thread="CIFAR-10", tracks="C",
           body=r"""Un collègue a écrit le Grad-CAM de 206.6 pour `RESNET` (le modèle de B6.20) en quatre morceaux : `LAYER_21`, la couche dont on pondère les activations ; `score_21(logits, target)`, le scalaire que l'on dérive ; `combine_21(A, G)`, la combinaison des activations et des gradients ; `upsample_21(cam, size)`, l'agrandissement de la carte à la taille des images. L'extraction par hooks (`extract_21`, dans la cellule donnée) est juste, comme ta fonction de B6.20. Quatre défauts se sont glissés dans les morceaux, ceux du titre. **Répare la cellule suivante en place** (mêmes noms, mêmes signatures).

La cellule de vérification enchaîne cinq diagnostics, dans l'ordre où un défaut peut en masquer un autre : 1. la résolution des activations de la couche ; 2. la dépendance des gradients à la classe demandée (la vraie classe, puis une autre) ; 3. le signe des cartes avant agrandissement ; 4. l'alignement de la carte agrandie sur l'image, mesuré sur une carte « rampe » dont les cases valent 0, 1, 2 de gauche à droite ; 5. les cartes finales, comparées à la référence `wbx.grad_cam`. Le premier qui échoue arrête la chaîne (⏳ pour les suivants) ; un message dit ce qu'il mesure, pas ce qu'il faut changer. La cellule d'expérience affiche tes cartes à côté de la référence, sur 6 images (3 bonnes prédictions et 3 erreurs de B6.20, la vraie classe) : lance-la une première fois **avant** de réparer, pour voir les symptômes.

Dans tes notes : pour chaque défaut, le symptôme que tu as vu (sur les cartes, puis dans le message) et la ligne fautive. Pourquoi le point 2 compare-t-il les gradients de deux classes plutôt que de regarder une seule carte ? Pourquoi chercher un décalage avec une rampe plutôt qu'avec la carte d'une vraie image ? Lequel de ces défauts une carte « qui a l'air juste » cacherait-elle le mieux ?""",
           given=C2.GIVEN_21, todo=C2.TODO_21, check=C2.CHECK_21, solution=C2.SOLUTION_21 + "\n\n\n" + C2.CHECK_21,
           after=[("code", C2.EXPERIMENT_21)])

EX_22 = Ex("B6.22", "🔬", 3, 40, "Test de santé d'Adebayo : randomiser les poids couche par couche, la carte change-t-elle ?",
           "soumettre quatre cartes d'attribution au test de santé par randomisation des poids, et lire leurs corrélations "
           "avec leur dispersion.",
           "Ex B6.19, B6.20 · fiche §206.7 (« Une explication peut avoir l'air juste et être fausse », « Les tests de "
           "santé ») · 🧠 Q9", thread="CIFAR-10", tracks="C",
           body=r"""Une carte qui décrit le modèle doit **changer quand le modèle change** (206.7). Tu fais le test d'Adebayo et coll. sur le petit CNN de B6.19, puis sur le ResNet de B6.20. **Ne présume pas l'issue : mesure-la.** **Contrat** :
- `randomize_22(model, names, seed=0)` rend une **copie** gelée (mode `eval()`, `requires_grad=False` partout) de `model`, dans laquelle chaque `Conv2d` et chaque `Linear` des parties nommées (`model.get_submodule(name)` : des noms comme `"fc"` ou `"features.17"`) reprend son `reset_parameters()`, l'initialisation par défaut de PyTorch, tirée après `torch.manual_seed(seed)` **dans** `torch.random.fork_rng()` : `model` et le générateur aléatoire global ne bougent pas, les batchnorms et les autres parties gardent leurs valeurs, et la même graine redonne les mêmes poids ;
- `sanity_22(model, explain, names, mode="cascade", seed=0, absolute=False)` rend un tableau `(len(names), N)` : sa ligne $k$ est la corrélation de rangs de Spearman, image par image, entre les cartes `explain(model)` et `explain(copie)`, la copie randomisant les $k + 1$ premières parties de `names` (`mode="cascade"`, de la sortie vers l'entrée) ou la partie $k$ seule (`mode="independent"`) ; `explain(m)` rend les cartes `(N, …)` des $N$ images pour le modèle `m` ; `absolute=True` compare leurs valeurs absolues ; une carte constante (par exemple toute nulle après la ReLU de Grad-CAM) donne `nan`. `scipy.stats.spearmanr` ou `wbx.spearman_per_image` font la corrélation.

La cellule donnée choisit 16 images de test que le CNN classe bien (`X_22`, `T_22`) et `NAMES_22`, les parties du CNN qui ont des poids, de la sortie vers l'entrée (`wbx.cascade_layers`). La vérification essaie `randomize_22` sur deux parties du CNN (une copie, le générateur, le gel, les parties changées et les autres intactes, la graine) et sur un bloc du ResNet (ses batchnorms restent), puis compare `sanity_22` à un calcul direct sur le gradient du logit de 4 images, dans les deux modes. L'expérience (≈ 25 s) mesure quatre cartes du CNN, pour la vraie classe : Grad-CAM du dernier bloc convolutif (`features[19]`, la ReLU qui suit la dernière convolution, 8 × 8), les gradients intégrés (20 intervalles, valeurs absolues sommées sur les canaux), le gradient (la saillance de B6.19) et un **contrôle** qui ne regarde pas le modèle, les bords de l'image (un filtre de Sobel) ; en cascade et une partie à la fois, pour trois graines des poids aléatoires ; puis le Grad-CAM de `layer4` du ResNet, en cascade, sur les 16 images de B6.20. Elle trace la médiane et les quartiles des corrélations (16 images × 3 graines) et affiche un tableau, avec l'étendue des médianes d'une graine à l'autre et le nombre de cartes constantes.

Dans tes notes : quelles cartes changent quand on randomise, lesquelles non, et dès quelle partie ? Que montre le contrôle, et pourquoi le mettre ? Les gradients intégrés gardent-ils une part de l'image elle-même (pense au facteur $x_i - x'_i$) ? Que veut dire une carte de Grad-CAM qui devient constante ? La dispersion d'une image à l'autre et d'une graine à l'autre change-t-elle tes conclusions ? Qu'est-ce qu'un test réussi prouve, et qu'est-ce qu'il ne prouve pas (🧠 Q9) ?""",
           given=C2.GIVEN_22, todo=C2.TODO_22, check=C2.CHECK_22, solution=C2.SOLUTION_22 + "\n\n\n" + C2.CHECK_22,
           after=[("code", C2.EXPERIMENT_22)])

PART_B.exercises += [EX_21, EX_22]

# ---------------------------------------------------------------------------------------------------------------------
PART_C = Part("C", "Équité : auditer, corriger, documenter",
              "Au MP3, tu as prédit un revenu de plus de 50 000 $ sur *Adult* avec un pipeline sans fuite, audité ses erreurs "
              "par sexe et par âge, et mesuré sur les folds le retrait de `sex`. Ici, tu écris d'abord les indicateurs de "
              "ta librairie (B6.23), tu refais l'audit **avec des intervalles** (B6.24), tu lis un tableau d'audit (B6.25), "
              "tu prédis ce que change le retrait de `sex` (B6.26), tu essaies trois corrections et tu chiffres leur coût "
              "(B6.27), tu documentes le modèle dans une fiche (B6.28), puis tu relèves le défi (B6.29).\n\n"
              "La cellule suivante charge le découpage figé d'*Adult* (`wbx.adult()` : `XA_TRAIN`, `YA_TRAIN`, 29 304 "
              "personnes d'entraînement ; `XA_VAL`, `YA_VAL`, 9 769 de validation ; le test, celui du MP3, reste à part "
              "jusqu'au défi), `adult_pipeline(model=None, drop=())`, le pipeline du MP3 en encodage ordinal "
              "(`income.build_pipeline(..., encoding=\"ordinal\")` : `fnlwgt` et `education` écartés, et les colonnes de "
              "`drop`), `age_band`, les tranches d'âge du MP3, et `HGB_C`, le gradient boosting de référence de la partie "
              "(pipeline du MP3, `random_state=0`, ajusté sur l'entraînement), avec ses scores `P_VAL`, ses décisions "
              "`PRED_VAL` au seuil 0,5 sur la validation, `SEX_VAL` et `AGE_VAL`. `sk_audit(y, p, g)` rend les taux par "
              "groupe et les écarts, calculés avec scikit-learn : c'est la référence des vérifications et des expériences "
              "données, qui ne dépendent jamais de ton code. ≈ 5 s.",
              given=C2.TOOLS_C)

EX_23 = Ex("B6.23", "🔨", 2, 25, "group_rates, demographic_parity_difference et equalized_odds_difference",
           "écrire les taux par groupe et les deux écarts d'équité de ta librairie, avec leurs cas limites.",
           "✏️ B6.5 · ch. 3 (`confusion_matrix`) · 🔁 R3 · fiche §206.9 (« Des taux par groupe », « Les critères »)",
           tracks="R, C", mylearn="explain.py",
           body=mylearn_howto() + r"""

**Contrat** de `group_rates(y_true, y_pred, groups)`, qui rend `{groupe: {'n', 'base_rate', 'selection_rate', 'tpr', 'fpr', 'precision', 'accuracy'}}` :
- les clés sont les valeurs des groupes, dans l'ordre de `np.unique` (`rates[10]` pour le groupe entier 10 ; un scalaire Python ou NumPy convient) ; `'n'` est un entier, les taux des floats, et un taux dont le dénominateur est nul vaut `nan` (un groupe sans positif a un TPR `nan`) ;
- chaque groupe a sa matrice de confusion 2 × 2, celle de ta `confusion_matrix` du ch. 3 avec `labels=[0, 1]` : sans `labels`, un groupe qui n'a qu'un seul label donne une matrice 1 × 1, que `tn, fp, fn, tp = C.ravel()` ne sait pas défaire ; `C.ravel()` rend `TN, FP, FN, TP` (lignes = vérité, colonnes = prédiction, la disposition de scikit-learn ; le livre met TP en haut à gauche) ;
- un seul groupe est permis ; `ValueError` si les longueurs diffèrent ou si un label ou une prédiction n'est pas 0 ou 1 (NaN, 0,5, 2 ou −1 sont refusés) ; les entrées ne sont pas modifiées.

**Contrat** de `demographic_parity_difference(y_pred, groups)` et de `equalized_odds_difference(y_true, y_pred, groups)` :
- le plus grand moins le plus petit taux de sélection, sur **tous** les groupes ; pour l'equalized odds, le plus grand de deux écarts, celui des TPR et celui des FPR, chacun le plus grand moins le plus petit sur tous les groupes ;
- `ValueError` s'il y a moins de deux groupes, et, pour l'equalized odds, si un groupe n'a aucun positif ou aucun négatif (son TPR ou son FPR est indéfini ; fairlearn rend alors un nombre, fiche 🕰️) ; les mêmes refus que `group_rates`.

La première cellule de vérification essaie tes fonctions sur l'exemple de la docstring, sur trois groupes entiers non triés (contre `sklearn.metrics.confusion_matrix`), sur un groupe sans positif, sur un groupe dont les labels et les prédictions valent tous 0, sur les deux écarts à trois groupes et sur trois refus ; la seconde lance les 42 tests de l'exercice (`-k "test_group_rates_ or test_demographic_parity_difference_ or test_equalized_odds_difference_"`).

Dans tes notes : pourquoi l'ordre `TN, FP, FN, TP` est-il un piège quand on passe d'un livre à scikit-learn ? Pourquoi ton equalized odds refuse-t-il un groupe sans positif, quand fairlearn rend un nombre ? Avec trois groupes, l'écart dit-il entre quels groupes il se trouve ?""",
           check=C.RELOAD + C2.CHECK_23,
           solution=C2.CHECK_23,
           after=tests_after("B6.23", "mylearn.explain.group_rates([0, 1], [0, 1], [0, 0])",
                             "test_group_rates_ or test_demographic_parity_difference_ or test_equalized_odds_difference_",
                             "42 tests", "tes trois fonctions"))

EX_24 = Ex("B6.24", "📦", 3, 45, "Audit d'équité sur Adult : indicateurs par sexe et par âge, intervalles de confiance bootstrap",
           "auditer un classifieur groupe par groupe, avec un intervalle bootstrap pour chaque taux et chaque écart.",
           "Ex B6.23 · ch. 2 (`bootstrap_ci`, Ex 2.22) · CP3 (MP3.7) · fiche §206.8 et §206.9 (« Des écarts avec leurs "
           "intervalles »)", tracks="R, C",
           body=r"""Au MP3, ton audit donnait des taux par groupe sans intervalles (tes intervalles bootstrap portaient sur le test entier). Un écart entre deux taux ne dit rien sans son incertitude, surtout pour un petit groupe (206.9, « Des écarts avec leurs intervalles »). **Contrat** :
- `rate_ci_24(y_true, y_pred, rate, n_boot=500, rng=None)` rend `(low, high)`, l'intervalle bootstrap à 95 % (centiles 2,5 et 97,5) du taux `rate` (`"selection_rate"`, `"tpr"` ou `"fpr"`) d'**un** groupe, calculé par ta `bootstrap_ci` (ch. 2) sur les paires `(y_true, y_pred)` du groupe, tirées ensemble ; sur un tirage sans dénominateur (aucun positif tiré, pour un TPR), la statistique rend `nan`, que ta `bootstrap_ci` refuse (`ValueError`, ch. 2) : sur les groupes d'*Adult*, cela n'arrive pas (le plus petit dénominateur, les 12 positifs des moins de 25 ans, n'est presque jamais tiré vide) ; pour un petit groupe, il faudrait `bootstrap_distribution` puis `np.nanpercentile`, en disant la part des tirages écartés ;
- `audit_24(y_true, y_pred, groups, n_boot=500, seed=0)` rend un `pd.DataFrame`, une ligne par groupe (dans l'ordre de `np.unique`, index nommé `"group"`), les colonnes `COLUMNS_24` : `n`, `positives` (les y = 1 du groupe), les taux de ta `group_rates` (`base_rate`, `selection_rate`, `tpr`, `fpr`, `precision`, `accuracy`), et les bornes `selection_rate_low`, `selection_rate_high`, `tpr_low`, `tpr_high`, `fpr_low`, `fpr_high` de `rate_ci_24` sur les lignes du groupe, avec un seul `rng = np.random.default_rng(seed)` pour tout le tableau ;
- `gap_ci_24(y_true, y_pred, groups, kind, n_boot=500, seed=0)` rend `(gap, low, high)` : l'écart `kind` sur les groupes (`"dp"` : ta `demographic_parity_difference` ; `"eo"` : ta `equalized_odds_difference` ; `"tpr"` ou `"fpr"` : le plus grand moins le plus petit de ce taux) et son intervalle à 95 % ; à chaque tirage, les lignes de **chaque groupe** sont tirées avec remise dans leur groupe (un bootstrap stratifié par groupe : chaque tirage garde les effectifs ; un seul `rng = np.random.default_rng(seed)` pour tous les tirages), puis l'écart est recalculé.

La cellule de vérification audite `PRED_VAL` (le modèle de référence, seuil 0,5, sur la validation) par sexe et par tranche d'âge : les taux sont comparés à scikit-learn ; les intervalles doivent contenir leur taux et avoir la largeur attendue, celle de l'approximation normale $\pm 1{,}96\sqrt{r(1-r)/m}$ à 30 % près, quand le dénominateur $m$ dépasse 100 ; la même graine doit redonner le même tableau, et deux groupes identiques des intervalles différents (un seul générateur) ; puis `gap_ci_24` est comparé aux écarts de scikit-learn, et ses tirages doivent garder l'effectif de chaque groupe. L'expérience affiche les deux tableaux avec leurs intervalles et les quatre écarts par sexe et par âge, et trace les taux par groupe. `N_BOOT_24` vaut 500 en mode rapide (≈ 30 s pour la vérification et l'expérience), 2 000 sinon.

Dans tes notes : quel groupe a les intervalles les plus larges, et pourquoi ? Où sont les écarts par sexe : dans les TPR, dans les FPR, dans la sélection ? Leurs intervalles excluent-ils 0 ? Les écarts par âge viennent-ils plutôt des erreurs du modèle ou des taux de base ? Pourquoi tirer dans chaque groupe plutôt que dans toute la validation ? Que deviendraient tes conclusions avec 24 sous-groupes croisés (sexe × âge × deux catégories raciales) ?""",
           given=C2.GIVEN_24, todo=C2.TODO_24, check=C2.CHECK_24, solution=C2.SOLUTION_24 + "\n\n\n" + C2.CHECK_24,
           after=[("code", C2.EXPERIMENT_24)])

LOWEST_HINT = "suis la colonne du TPR, tranche par tranche"
EX_25 = Ex("B6.25", "📈", 2, 15, "Lire un tableau d'audit : où est l'écart, est-il significatif, que conclure ?",
           "lire un tableau d'audit avec effectifs et intervalles, et en tirer une conclusion prudente.",
           "Ex B6.24 · fiche §206.9 (« Des écarts avec leurs intervalles », « L'impossibilité »)", tracks="C",
           body=r"""Le tableau suivant est l'audit du modèle de référence de la partie (`HGB_C`, seuil 0,5) sur la validation, par tranche d'âge et par sexe, avec 2 000 tirages bootstrap (la solution de B6.24, calculée une fois) ; le second tableau donne les écarts avec leur intervalle à 95 %. Réponds **à partir du tableau** :
- a) `lowest_tpr_25` : la lettre de la tranche d'âge au plus petit TPR : `"A"` <25, `"B"` 25-34, `"C"` 35-44, `"D"` 45-54, `"E"` 55-64, `"F"` 65+ ;
- b) `positives_25` : sur combien de positifs de la validation ce TPR repose-t-il ? (un entier)
- c) `widest_25` : parmi les **cinq autres** tranches, la lettre de celle dont l'intervalle du TPR est le plus **large** (mêmes lettres) ;
- d) `tpr_sex_25` : l'intervalle à 95 % de l'écart de TPR entre les sexes exclut-il 0 ? `True` ou `False` ;
- e) `overlap_25` : les intervalles du TPR des femmes et des hommes se chevauchent-ils ? `True` ou `False` ;
- f) `conclusion_25` : la conclusion que le tableau permet :
  - `"A"` le modèle est équitable par âge, puisque les intervalles des TPR des tranches se chevauchent deux à deux ;
  - `"B"` il faut égaliser les taux de sélection des tranches d'âge, puisque ΔDP par âge est le plus grand écart du tableau ;
  - `"C"` le TPR le plus bas repose sur peu de positifs et son intervalle est large : l'écart de TPR par âge est mesurable, sa taille très incertaine, et les écarts de sélection suivent surtout les taux de base ;
  - `"D"` les écarts par sexe ne sont pas significatifs, car les deux groupes sont grands.

Dans tes notes : pourquoi un intervalle large ne veut-il pas dire « pas d'écart » ? Quelle ligne rapporterais-tu en premier à un métier, avec quelle phrase ? Avec six tranches, combien de comparaisons deux à deux ferais-tu, et qu'en dit 206.9 (les comparaisons multiples) ?""",
           given=fill(C2.GIVEN_25, {"ROWS_25": ROWS_25, "GAPS_25": GAPS_25}), todo=C2.TODO_25, check=C2.CHECK_25,
           solution=fill(C2.SOLUTION_25, A25),
           record="\n".join([
               letters_record("B6.25a", A25["A25"], {letter: f"{LOWEST_HINT}" for letter in "ABCDEF"}),
               f'wb.record("B6.25b", {A25["B25"]}, fractional="un nombre de personnes est un entier : lis la colonne des '
               f'positifs", mistakes={{"relis la question : on demande les positifs (y = 1), pas tout l\'effectif de la '
               f'tranche": {T25["tables"]["age"]["<25"]["n"] if T25 else 1}}})',
               letters_record("B6.25c", A25["C25"], {letter: (f"relis la question : on cherche parmi les "
                                                               "cinq autres tranches que celle de a)" if letter == A25["A25"]
                                                               else f"compare la largeur (borne haute moins "
                                                               "borne basse) des intervalles du TPR des cinq autres tranches")
                                                     for letter in "ABCDEF"}),
               f'wb.record("B6.25d", {A25["D25"]}, mistakes={{"lis les deux bornes de l\'intervalle de l\'écart de TPR '
               f'par sexe : 0 est-il entre elles ?": {not A25["D25"]}}})',
               f'wb.record("B6.25e", {A25["E25"]}, mistakes={{"compare la borne haute de l\'un et la borne basse de '
               f'l\'autre, dans la colonne du TPR": {not A25["E25"]}}})',
               letters_record("B6.25f", A25["F25"], {
                   "A": "des intervalles qui se chevauchent prouvent-ils l'égalité ? Regarde l'intervalle de l'écart",
                   "B": "un écart de sélection suit aussi les taux de base : le tableau dit-il qu'il faut les égaliser ?",
                   "D": "lis l'intervalle des écarts par sexe : 0 y est-il ?"})]))

EX_26 = Ex("B6.26", "🔮", 1, 10, "Prédire : retirer la variable « sexe » rend-il le modèle équitable ?",
           "prédire de combien le retrait d'un attribut sensible change les écarts d'un modèle, et ce qui le remplace.",
           "Ex B6.24 · MP3 (MP3.7) · fiche §206.8 (« L'équité par aveuglement ») et §206.10 · 💼 B6.E5", tracks="R, C",
           hypothesis=True,
           body=r"""Retirer `sex` des features et réentraîner : le modèle devient-il équitable ? La fiche dit seulement que cela se **mesure** (206.8, « L'équité par aveuglement »). L'expérience mesure, sur la validation, quatre grandeurs avec la recette du modèle de référence (`adult_pipeline`). **Avant d'exécuter**, écris tes quatre prédictions :
- a) `proxy_auc_26` : la ROC-AUC d'un gradient boosting (la même recette) qui prédit le **sexe** à partir des 11 autres features du pipeline (`fnlwgt` et `education` en sont déjà écartés) : `"A"` moins de 0,6 ; `"B"` de 0,6 à 0,75 ; `"C"` de 0,75 à 0,9 ; `"D"` plus de 0,9 ;
- b) `proxy_auc_few_26` : la même chose sans `relationship` ni `marital-status` (9 features) : mêmes lettres ;
- c) `eo_ratio_26` : le ΔEO par sexe du modèle réentraîné sans `sex`, divisé par celui du modèle de référence : `"A"` moins de 0,5 (l'écart baisse de plus de moitié) ; `"B"` de 0,5 à 1,5 ; `"C"` plus de 1,5 ;
- d) `four_fifths_26` : pour le modèle sans `sex`, le taux de sélection des femmes divisé par celui des hommes (la règle des quatre cinquièmes, 206.9) : `"A"` moins de 0,4 ; `"B"` de 0,4 à 0,6 ; `"C"` de 0,6 à 0,8 ; `"D"` 0,8 ou plus.

La cellule d'expérience n'affiche rien tant que tes quatre prédictions ne sont pas écrites ; elle entraîne trois modèles (≈ 10 s), affiche les deux AUC et un tableau des taux par sexe avec et sans `sex`, puis vérifie tes prédictions. Rien n'y dépend de ton code.

Dans tes notes, avant d'exécuter : quelles features d'*Adult* portent, selon toi, l'information du sexe, et comment le vérifierais-tu ?""",
           todo=C2.TODO_26,
           solution=(f'proxy_auc_26, proxy_auc_few_26, eo_ratio_26, four_fifths_26 = "{P26["proxy_auc"]}", '
                     f'"{P26["proxy_auc_few"]}", "{P26["eo_ratio"]}", "{P26["four_fifths"]}"'),
           record="\n".join([
               letters_record("B6.26a", P26["proxy_auc"], {k: f"lis la ROC-AUC du modèle qui prédit le sexe à partir "
                                                           "des 11 autres features" for k in "ABCD"}, prediction=True),
               letters_record("B6.26b", P26["proxy_auc_few"], {k: f"lis la ROC-AUC sans relationship ni "
                                                               "marital-status" for k in "ABCD"}, prediction=True),
               letters_record("B6.26c", P26["eo_ratio"], {k: f"lis le rapport des deux ΔEO, sous le tableau"
                                                          for k in "ABC"}, prediction=True),
               letters_record("B6.26d", P26["four_fifths"], {k: f"lis la colonne F / M du modèle sans sex"
                                                             for k in "ABCD"}, prediction=True)]),
           after=[("code", guarded(C2.EXPERIMENT_26, ["proxy_auc_26", "proxy_auc_few_26", "eo_ratio_26", "four_fifths_26"],
                                   "⏳ Ex B6.26 : écris d'abord tes quatre prédictions dans la cellule précédente.")),
                  ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Quelles colonnes portent "
                         "l'information du sexe, et que reste-t-il sans elles ? Si le modèle n'utilise pas `sex`, d'où viennent "
                         "les écarts qui restent : des variables de substitution, des taux de base, des labels ? Que "
                         "perdrait-on, pour l'audit et pour la correction, en effaçant `sex` des données ?")])

EX_27 = Ex("B6.27", "🔬", 3, 50, "Atténuer : retrait des variables de substitution, repondération, seuils par groupe, et ce que cela coûte",
           "comparer trois familles de corrections d'équité sur un même modèle, réauditer tous les groupes et chiffrer le "
           "coût en accuracy.",
           "Ex B6.24, B6.26 · fiche §206.10 (« Trois moments pour agir », « Le prix à payer »)", tracks="R, C",
           body=r"""Trois familles de corrections (206.10), essayées sur le même modèle, et leur coût. **Contrat** :
- `reweigh_27(groups, y)` rend le poids $w(g, y) = P(g)\,P(y) / P(g, y)$ de chaque exemple (Kamiran et Calders), les probabilités estimées par les proportions dans ces données ;
- `tpr_thresholds_27(scores, y, groups, target_tpr)` rend `{groupe: seuil}` (les groupes de `np.unique`) : pour chaque groupe, le quantile `1 - target_tpr` des scores de ses **positifs** (`np.quantile`, comme dans 206.10), pour que son TPR vaille à peu près `target_tpr` sur ces données ;
- `decide_27(scores, groups, thresholds)` rend les décisions 0/1 (des entiers) : 1 quand le score atteint le seuil du groupe de l'exemple (`>=`).

La cellule donnée coupe la validation en deux moitiés : `CAL_27`, où se choisissent les seuils, et `EVAL_27`, où tout se mesure (choisir et mesurer sur les mêmes personnes flatterait les seuils). La vérification essaie tes trois fonctions sur de petits cas. L'expérience (≈ 20 s) compare cinq stratégies, toutes mesurées sur `EVAL_27` : le modèle de référence au seuil 0,5 ; réentraîné sans `sex` ; sans `sex`, `relationship` ni `marital-status` (des variables de substitution, B6.26) ; repondéré sur le sexe et le label (`fit(..., model__sample_weight=w)`) ; des seuils par sexe choisis sur `CAL_27` pour le TPR qu'y obtient la référence. Pour chacune : l'accuracy, ΔDP et ΔEO par sexe (avec leur intervalle, ta `gap_ci_24` de B6.24, 200 tirages), ΔDP et ΔEO par tranche d'âge (l'audit refait après chaque correction), et les TPR et FPR des femmes et des hommes. **Ne présume pas l'issue : mesure-la.**

Dans tes notes : quelle stratégie réduit le plus ΔEO par sexe, et à quel coût en accuracy ? Les seuils par sexe rapprochent les TPR : que deviennent les FPR, et donc ΔEO ? La repondération change-t-elle les écarts autant que tu l'attendais ? Une correction par sexe déplace-t-elle les écarts par âge ? Lesquelles de ces stratégies utilisent le sexe dans la **décision**, et qu'en dit 206.10 ? Laquelle recommanderais-tu pour la statistique publique du MP3, et pourquoi ?""",
           given=C2.GIVEN_27, todo=C2.TODO_27, check=C2.CHECK_27, solution=C2.SOLUTION_27 + "\n\n\n" + C2.CHECK_27,
           after=[("code", C2.EXPERIMENT_27)])

EX_28 = Ex("B6.28", "🛠️", 2, 25, "Fiche modèle avec sections explicabilité et équité : indicateurs par groupe, limites, usage prévu",
           "documenter un modèle de décision dans une fiche modèle complète, chiffres par groupe et limites compris.",
           "Ex B6.24, B6.27 · ch. 23 (Ex 23.28) · MP3 (fiche modèle) · fiche §206.11 (« La fiche modèle »)", tracks="R, C",
           body=r"""Au ch. 23, tu as rédigé la fiche modèle d'un MLP ; au MP3, celle du modèle de revenu. Rédige ici celle du modèle de référence de la partie (`HGB_C`), avec les deux rubriques que la fiche ajoute pour un modèle de décision, **explicabilité** et **équité** (206.11). La cellule suivante affiche ce qu'il faut savoir (`FACTS_28` : le modèle, les données, l'accuracy et la ROC-AUC de validation, l'effectif, le TPR et le FPR de chaque sexe et de chaque tranche d'âge, l'importance par permutation sur la validation) et les onze sections (`REQUIRED_28`) ; ≈ 5 s.

**Contrat** : `CARD_28` est une chaîne Markdown d'au moins 300 mots, avec les onze sections de `REQUIRED_28` comme titres `## …`, dans cet ordre (les accents et la casse des titres ne comptent pas, et un titre peut continuer : « ## Usage prévu et hors périmètre ») ; « Usage prévu » dit aussi les usages **hors** périmètre ; « Métriques » donne le seuil (0,5) ; l'accuracy de validation y est, à 4 décimales ; les sections « Analyses quantitatives » ou « Équité » ont, pour chaque groupe de `FACTS_28["groupes"]`, une ligne (de tableau, par exemple) avec son nom, son effectif, son TPR et son FPR à 2 décimales ; « Explicabilité » nomme une méthode et les données sur lesquelles elle est calculée (la validation, ou le fond d'une valeur SHAP) ; « Équité » nomme le critère retenu (parité démographique, égalité des chances ou equalized odds) et parle des intervalles. Une f-string à triples guillemets (`f` devant trois guillemets) t'évite de recopier les nombres. Sers-toi de ce que tu as mesuré en B6.24, B6.26 et B6.27.

La vérification relit ta fiche, dit ce qui manque, puis l'écrit dans `fiche_modele_adult_b6.md`, à côté de ton notebook (dans `mon_travail`) : elle a sa place dans ton portfolio, avec la data card d'*Adult* (`data/cards/adult.md`).

Dans tes notes : qu'est-ce qu'une vérification de structure ne peut pas juger dans ta fiche ? Qui la lirait (un métier, un auditeur, une autorité), et que chercherait chacun d'abord ?""",
           given=C2.GIVEN_28, todo=C2.TODO_28, check=C2.CHECK_28, solution=C2.SOLUTION_28 + "\n\n\n" + C2.CHECK_28)

N_CAL_29 = len(C29["rows"]) or 11
EX_29 = Ex("B6.29", "🏆", 4, 90, "Défi : écart d'equalized odds sous 0,07 sur Adult, en perdant moins de 2 points d'accuracy",
           "concevoir une méthode qui rapproche les taux d'erreur des deux sexes sur des personnes jamais vues, pour un coût "
           "en accuracy borné.",
           "Ex B6.24, B6.27 · tout le chapitre · fiche §206.10", tracks="C",
           body=r"""**Le défi** : écris `method_29(X_train, y_train, X_val, y_val)`, qui ajuste **un modèle neuf** sur ces seules données (des DataFrames des 14 colonnes d'*Adult*, `sex` compris, et des tableaux de labels 0/1) et rend une fonction `decide(X)`, qui rend les décisions 0/1 des personnes du DataFrame `X` (un tableau d'entiers de forme `(len(X),)`), de façon déterministe. La mesure, sur le **test du MP3** (9 769 personnes que ta méthode ne voit jamais) : un ΔEO par sexe sous `GOAL_EO_29` (0,07), et une accuracy inférieure de moins de 2 points (`MAX_LOSS_29`) à celle du gradient boosting de référence **sans correction** (seuil 0,5) sur le même test ; **et de même** sur un découpage de contrôle (`wbx.adult(seed=CONTROL_SEED_29)`, dont le test contient d'autres personnes), pour qu'une méthode réglée sur le test du MP3 ne passe pas par chance.

**Le sexe dans la décision.** Dans ce défi, `decide` **peut** utiliser `sex` : des seuils par sexe sont permis. C'est un choix d'exercice : en entreprise, utiliser l'attribut sensible dans la décision peut être une discrimination directe, même pour corriger (206.10) ; dis dans tes notes ce que tu ferais s'il était interdit.

**Les règles**, contrôlées par `evaluate_29(method_29)` :
- **à froid et sans fuite** : pendant que ta méthode s'ajuste et décide, les variables du notebook qui contiennent des données (un DataFrame, une Series ou un tableau d'au moins 1 000 éléments, un découpage de `wbx`) ou un modèle scikit-learn déjà ajusté, directement ou dans une liste, un tuple ou un dictionnaire, sont masquées, et les chargeurs de `wbx` et de `wb.datasets` neutralisés ; `method_29` ne garde rien de tel dans ses arguments par défaut ; elle ne reçoit que l'entraînement et la validation (leurs lignes numérotées à partir de 0 : l'index d'origine d'*Adult* est retiré), jamais les labels du test ;
- **chaque personne est décidée seule** : sur les 200 premières personnes du test, `decide` appliquée à une ligne à la fois, avant et après le test entier, doit redonner les décisions du test entier, et le test présenté dans un autre ordre, les mêmes décisions dans cet ordre (aucun réglage sur les features du test) ;
- **contrôle des labels mélangés** : ta méthode, ajustée sur des labels d'entraînement et de validation mélangés, ne doit plus rien savoir (une balanced accuracy d'au plus `MAX_BALANCED_29` = 0,56 sur le test ; le hasard : 0,5) ; et aucune accuracy ne dépasse de plus de 2 points celle de la référence sans correction (une correction d'équité ne fait pas mieux qu'elle d'autant) ;
- **révélé une fois** : le premier résultat sur le test du MP3 est gardé pour la session, et c'est lui qui compte. Règle d'honneur : ne règle pas ta méthode sur le test, que tu pourrais lire dans `ADULT` ou dans le fichier d'*Adult* sur le disque.

La cellule suivante donne `evaluate_29`, `GOAL_EO_29`, `MAX_LOSS_29`, `CONTROL_SEED_29`, `MAX_BALANCED_29` et `REFERENCE_29` (le meilleur et le pire ΔEO, et la plus petite et la plus grande baisse d'accuracy de la solution de référence sur «N_CAL_29» découpages) ; elle ajuste les deux gradient boostings sans correction (≈ 10 s). L'évaluation ajuste ta méthode trois fois (≈ 20 s pour la référence). Tu peux reprendre tes fonctions du chapitre, ou faire autrement : d'autres seuils, une repondération, une contrainte pendant l'entraînement, un autre modèle.

Dans tes notes : ta méthode, ce que tu as essayé, ce qui a aidé et ce qui n'a rien changé, tes deux ΔEO et tes deux baisses d'accuracy. Pourquoi viser sur la validation un écart plus petit que l'objectif ? De combien le ΔEO de ta méthode varie-t-il d'un découpage à l'autre, comparé à l'objectif ? Et si `sex` était interdit dans la décision ? La réussite est affichée ; elle n'est pas enregistrée : c'est une règle d'honneur.""".replace("«N_CAL_29»", str(N_CAL_29)),
           given=fill(C2.GIVEN_29, {"GOAL_29": repr(GOAL_29), "REF_29": repr(REF_29), "CONTROL_29": CONTROL_29,
                                    "N_CAL_29": N_CAL_29}),
           todo=C2.TODO_29, check=C2.CHECK_29, solution=C2.SOLUTION_29 + "\n\n\n" + C2.CHECK_29)

PART_C.exercises = [EX_23, EX_24, EX_25, EX_26, EX_27, EX_28, EX_29]
PARTS.append(PART_C)
