"""Parts A and B and the start of part C of the chapter B5 notebook (B5.12 to B5.21): statements and cells (used by
build_chB5.py).

Part A: the mathematics of your NumPy library diffusion.py and their checks (B5.12 to B5.18). Part B: 2-D toys, a
denoising MLP and flow matching (B5.19, B5.20). Part C: the U-Net on MNIST and Fashion-MNIST; this file holds its
introduction and its first exercise, the tiny U-Net (B5.21), whose class (UNET_CLASS, the learner's or the contract's)
the second half (chB5_parts2.py: B5.22 to B5.31) trains, samples and evaluates; it appends its exercises to
PART_C.exercises.
"""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, Part, guarded

import chB5_code as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB5.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}

P13 = HISTORY.get("p13", {}).get("answers", {"strong": 3, "cos600": False})
P15 = HISTORY.get("p15", {}).get("answers", {"same_curve": "S", "largest_end": "D", "cos_above": "C"})
B18 = HISTORY.get("b18", {}).get("answers", {"a": False, "b": True, "c": True, "d": False})
TOY = HISTORY.get("toy20", {})
A20 = TOY.get("answers", {"fewest": 5, "better5": "E"})
STEPS_20 = tuple(TOY.get("steps", (1, 2, 5, 10, 20, 50, 100)))

TEST = "tests/test_chb5_diffusion.py"


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** : ouvre `mon_travail/mylearn/diffusion.py` (créé par `python tools/start_chapter.py "
            "B5`), lis la docstring, remplace chaque `raise NotImplementedError(...)` de l'exercice par ton code et "
            "**enregistre**. NumPy seulement. La première cellule de vérification recharge ta librairie et l'essaie sur des "
            "cas visibles ; la seconde lance les tests de l'exercice. `python -m pytest tests/test_chb5_diffusion.py -q` "
            "les lance tous dans un terminal.")


def tests_after(ex_id: str, probe: str, keyword: str, n_tests: int, what: str = "tes fonctions") -> list:
    """The second check cell of a 🔨: the tests of the exercise (explicit test file, -k group)."""
    cell = (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_DIFF, "{keyword}")   # {n_tests} tests\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent {what}.")')
    ref = cell.replace('run_pytest_group(TEST_DIFF, "', 'run_pytest_group(TEST_DIFF, impl="ref", keyword="')
    return [("check", cell), ("solution", ref)]


COMMON_DIFF = (
    "**Contrat commun de `diffusion.py`** (les docstrings, plus ces précisions que vérifient les tests) :\n"
    "- les pas sont des **indices** de 0 à $T - 1$ : le pas $t$ de l'article, celui des formules de la fiche, est "
    "l'indice $t - 1$ ; un indice $-1$ ou $T$ lève `ValueError`, jamais `IndexError` (−1 n'est pas « le dernier "
    "pas ») ;\n"
    "- tes fonctions ne modifient jamais leurs entrées (pas de `-=` ni de `*=` sur un tableau reçu) ; les entrées peuvent "
    "être des listes ;\n"
    "- une valeur hors de son intervalle lève `ValueError`, **`nan` compris** : `nan` ne vérifie aucune condition « 0 < "
    "β < 1 », il doit donc être refusé ;\n"
    "- tolérances des tests : $10^{-9}$ en relatif pour les plannings (des float64, par contrat), $10^{-6}$ en relatif "
    "et en absolu ailleurs (un calcul en float32 passe) ;\n"
    "- si la bibliothèque diffusers est installée (elle l'est sur Colab), 25 tests comparent aussi tes fonctions à ses "
    "`DDPMScheduler` et `DDIMScheduler` ; sans elle, ils sont ignorés et les formes fermées restent testées.")

# ---------------------------------------------------------------------------------------------------------------------
PART_A = Part("A", "Les mathématiques de ta librairie et leurs contrôles",
              "Tu écris dans `mylearn/diffusion.py` tout ce qui, dans un modèle de diffusion, ne dépend pas du réseau : les "
              "plannings de bruit et leurs constantes (B5.12), le bruitage en forme fermée (B5.14), le posterior et "
              "l'estimation de $\\mathbf{x}_0$ (B5.16), puis un pas DDPM et un pas DDIM (B5.17). Entre deux, tu prédis "
              "ce qu'un classifieur voit encore d'un chiffre bruité (B5.13) et tu lis les plannings sur une figure "
              "(B5.15). Pour finir, tu enchaînes tes pas sur des points 2D avec le vrai bruit, le contrôle qui "
              "attrape les erreurs d'indice et de racine avant tout entraînement (B5.18).\n\n" + COMMON_DIFF + "\n\n"
              "La cellule suivante charge les outils des parties A et B : `TEST_DIFF` (le fichier des tests), `T_STEPS` "
              "(le nombre de pas du notebook : 400 en mode rapide, 1 000 en mode complet), `BETAS` (le planning linéaire "
              "du notebook, aux bornes multipliées par $1\\,000/T$ : avec les bornes par défaut et $T = 400$, il "
              "resterait trop d'image au dernier pas, le piège « SNR final non nul » de la fiche), `DIGITS` (1 000 "
              "chiffres de test de MNIST, ramenés dans $[-1, 1]$ par `wbd.to_signed`, toujours les mêmes), "
              "`toy_schedule` (les neuf constantes de `noise_schedule`, écrites sans ton code, pour les vérifications) "
              "et `show_images`. Le module `wbd` (`wb.diffusion`) contient l'infrastructure du chapitre : données, "
              "plannings, jouets 2D et modèles de référence ; le notebook ne s'en sert que pour vérifier ton code ou "
              "quand il manque.",
              given=C.TOOLS_A, exercises=[
    Ex("B5.12", "🔨", 2, 20, "linear_beta_schedule, cosine_beta_schedule et noise_schedule",
       "écrire les deux plannings de bruit et toutes les constantes qui en dérivent.",
       "✏️ B5.2 · fiche §205.2 (« Le planning linéaire de Ho et coll. », « Le planning cosinus de Nichol et Dhariwal », "
       "« Bruiter un batch : `noise_schedule` et `q_sample` »)", tracks="R, M, C", mylearn="diffusion.py",
       body=mylearn_howto() + r"""

**Contrat**
- `linear_beta_schedule(n_steps, beta_start=1e-4, beta_end=0.02)` : un tableau float64 de forme `(T,)`, de `beta_start` (indice 0) à `beta_end` (indice $T - 1$), régulièrement espacés ; pour `n_steps = 1`, `[beta_start]`, comme `np.linspace` (aucune division par $T - 1$) ; `ValueError` si `n_steps < 1` ou si $0 < \text{beta\_start} \le \text{beta\_end} < 1$ n'est pas vérifié (`nan` compris).
- `cosine_beta_schedule(n_steps, s=0.008, max_beta=0.999)` : la fonction $f$ de la docstring évaluée aux $T + 1$ points $t/T$, $t = 0 \dots T$ (ni $T$ points, ni des points de $[0, 1)$), puis $\beta_t = \min(1 - f(t+1)/f(t),\ \text{max\_beta})$ pour $t = 0 \dots T - 1$, en float64 ; l'écrêtage à `max_beta` est atteint au dernier indice ; `ValueError` si `n_steps < 1`, `s < 0` ou `max_beta` hors de $(0, 1)$ (`nan` compris).
- `noise_schedule(betas)` : un dictionnaire aux neuf clés de la docstring, chacune un tableau float64 de forme `(T,)`, **même pour des betas en float32** ; `alpha_bars_prev[0]` vaut 1, donc `posterior_variance[0]` vaut 0 **exactement** ; `ValueError` si `betas` n'est pas 1-D ou si un $\beta$ est hors de $(0, 1)$, `nan` compris.

La première cellule de vérification essaie tes trois fonctions sur les exemples des docstrings, contre `np.linspace` et la forme fermée du cosinus, sur le cas $T = 1$, l'écrêtage et six refus, puis affiche $\bar\alpha$ au dernier indice du planning linéaire de $T$ = `T_STEPS` pas, avec les bornes par défaut puis multipliées par $1\,000/T$ (rien à taper). La seconde lance les 60 tests de l'exercice (`-k "test_linear_beta_schedule_ or test_cosine_beta_schedule_ or test_noise_schedule_"`).

Dans tes notes : pourquoi `posterior_variance[0]` vaut-il 0, et qu'en conclure pour le dernier pas de l'échantillonnage ? Avec $T = 400$, que vaut $\bar\alpha$ au dernier indice avec les bornes par défaut, quelle part de l'image reste-t-il (son coefficient $\sqrt{\bar\alpha}$), et pourquoi multiplier les bornes par $1\,000/T$ règle-t-il le problème (✏️ B5.2) ? Pourquoi un test « aucun $\beta$ n'est $\le 0$ ni $\ge 1$ » laisse-t-il passer un `nan` ?""",
       check=C.RELOAD + C.CHECK_12, solution=C.CHECK_12,
       after=tests_after("B5.12", "mylearn.diffusion.linear_beta_schedule(2)",
                         "test_linear_beta_schedule_ or test_cosine_beta_schedule_ or test_noise_schedule_", 60)),

    Ex("B5.13", "🔮", 1, 10, "Prédire l'allure d'un chiffre bruité aux pas 100, 300 et 600, planning linéaire puis cosinus",
       "prédire jusqu'à quel niveau de bruit un classifieur entraîné sur des chiffres propres reconnaît encore un chiffre "
       "bruité, et comparer à la convention de lisibilité de ✏️ B5.3.",
       "Ex B5.12 · ✏️ B5.3 · fiche §205.2 (« Le rapport signal sur bruit ») · ch. 27 (le classifieur de la partie B)",
       tracks="R, C", hypothesis=True,
       body=r"""En ✏️ B5.3, tu as adopté une **convention** : un chiffre reste lisible tant que son rapport signal sur bruit vaut au moins 0,1. Qu'en pense un vrai lecteur ? La cellule suivante prépare `CLF`, le petit CNN de la partie B du ch. 27 (`wbd.classifier`), entraîné une fois sur 10 000 chiffres **propres** de MNIST ; il en reconnaît environ 96 % sur les 1 000 chiffres de test. Elle donne aussi `noisy_13(planning, pas)`, qui bruite ces 1 000 chiffres par la forme fermée, au **pas de l'article** 100, 300 ou 600 (les indices 99, 299 et 599), du planning linéaire ou cosinus de $T = 1\,000$, toujours avec le même bruit.

L'expérience montre quatre chiffres à ces trois pas pour chaque planning, puis, pour chacun des six cas, le SNR et l'**accuracy** du classifieur sur les chiffres bruités. Tu peux calculer les six SNR (✏️ B5.3), pas l'accuracy. **Avant d'exécuter**, écris tes deux prédictions :
- a) `strong_13` : dans combien des six cas (3 pas × 2 plannings) le classifieur reconnaît-il **au moins 70 %** des chiffres ? Un entier de 0 à 6 ;
- b) `cos600_13` : au pas 600 du planning cosinus, où le SNR dépasse largement le seuil 0,1 de ta convention, le classifieur reconnaît-il **plus de la moitié** des chiffres ? `True` ou `False`.

La cellule d'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites ; elle les vérifie à la fin. Rien n'y dépend de ton code.""",
       given=C.GIVEN_13, todo=C.TODO_13,
       solution=f'strong_13, cos600_13 = {P13["strong"]}, {P13["cos600"]}',
       record=(f'wb.record("B5.13a", {P13["strong"]}, mistakes={{'
               + ", ".join(f'"{msg}": {v}' for v, msg in {
                   0: "relis le tableau : au pas 100, le classifieur reconnaît-il encore les chiffres ?",
                   1: "relis le tableau : compte le pas 100 des deux plannings",
                   2: "relis le tableau : le cosinus au pas 300 atteint-il 70 % ?",
                   3: "relis le tableau : le linéaire au pas 300 atteint-il 70 % ?",
                   4: "relis le tableau : compte un à un les cas à 70 % ou plus",
                   5: "relis le tableau : au pas 600, le cosinus atteint-il 70 % ?",
                   6: "relis le tableau : que vaut l'accuracy au pas 600 du linéaire ?",
               }.items() if v != P13["strong"])
               + '}, fractional="on demande un nombre de cas, un entier de 0 à 6", choices=[0, 1, 2, 3, 4, 5, 6], '
                 'prediction=True)\n'
               f'wb.record("B5.13b", {P13["cos600"]}, mistakes={{"relis le tableau : au pas 600 du cosinus, '
               f'l\'accuracy dépasse-t-elle 50 % ?": {not P13["cos600"]}}}, prediction=True)'),
       after=[("code", guarded(C.EXPERIMENT_13, ["strong_13", "cos600_13"],
                               "⏳ Ex B5.13 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? À partir de quel SNR, à "
                     "peu près, le classifieur décroche-t-il, et ton œil, sur les images ? Quelle classe le classifieur "
                     "prédit-il quand il ne sait plus, et pourquoi un réseau entraîné sur des images propres se trompe-t-il "
                     "ainsi sur du bruit qu'il n'a jamais vu ? Qu'est-ce que cela dit du débruiteur d'un modèle de "
                     "diffusion, qui, lui, voit des images de tous les niveaux de bruit pendant son entraînement ?")]),

    Ex("B5.14", "🔨", 2, 20, "q_sample : bruiter un batch d'images à des pas différents",
       "écrire le bruitage en forme fermée d'un batch où chaque exemple a son propre pas.",
       "Ex B5.12 · ∂ B5.6 · fiche §205.2 (« La forme fermée », « Bruiter un batch » et le piège « `alpha_bars[t]` sans "
       "remise en forme »)", thread="MNIST", tracks="R, M, C", mylearn="diffusion.py",
       body=mylearn_howto() + r"""

**Contrat** de `q_sample(x0, t, alpha_bars, noise=None, rng=None)`, qui rend le couple `(x_t, noise)` :
- $\mathbf{x}_t = \sqrt{\bar\alpha_{t}}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_{t}}\,\boldsymbol{\varepsilon}$ pour chaque exemple, avec **son** indice `t[n]` ; le coefficient est remis en forme `(N, 1, …, 1)`, quel que soit le nombre d'axes de `x0` (1-D, 2-D, `(N, C, H, W)`) ;
- `t` a **exactement un indice entier par exemple**, de forme `(N,)` : un `t` de longueur 1 pour un batch de plusieurs exemples lève `ValueError`, **même si NumPy saurait le diffuser** ; un indice hors de $[0, T)$ aussi ;
- `noise`, s'il est donné, a **exactement** la forme de `x0` (un bruit de forme `(2,)` pour `x0` de forme `(3, 2)` est refusé, même s'il se diffuse) ; sinon, **un seul** appel `rng.standard_normal(x0.shape)`, et c'est ce bruit qui est rendu (la cible de l'entraînement) ; `rng=None` : `np.random.default_rng()`, sans graine.

La première cellule de vérification essaie ta fonction sur l'exemple de la docstring, sur 6 chiffres de MNIST `(6, 1, 28, 28)` à 6 indices différents, sur un `x0` en 1-D, sur le bruit tiré par `rng`, et quatre refus ; elle montre les 6 chiffres bruités. La seconde lance les 19 tests de l'exercice (`-k "test_q_sample_"`).

Dans tes notes : que se passe-t-il, sans erreur, si tu multiplies `x0` de forme `(3, 1, 4, 3)` par `np.sqrt(alpha_bars[t])` de forme `(3,)` sans remise en forme (fiche, piège « `alpha_bars[t]` sans remise en forme ») ? Pourquoi rendre le bruit tiré, et pas seulement $\mathbf{x}_t$ ? Pourquoi refuser un `t` de longueur 1 que NumPy diffuserait volontiers ?""",
       check=C.RELOAD + C.CHECK_14, solution=C.CHECK_14,
       after=tests_after("B5.14", "mylearn.diffusion.q_sample([0.0], [0], [0.5], noise=[0.0])", "test_q_sample_", 19,
                         "ta fonction q_sample")),

    Ex("B5.15", "📈", 1, 15, "Lire ᾱ_t, le rapport signal sur bruit et la grille de bruitage des deux plannings",
       "lire sur une figure ce qui distingue les plannings, et voir un même chiffre se dissoudre dans le bruit.",
       "Ex B5.14 · ✏️ B5.3 · fiche §205.2 (« Le planning linéaire… », « Le planning cosinus… », « Le rapport signal sur "
       "bruit » et le piège « un SNR final qui n'est pas nul »)", thread="MNIST", tracks="C",
       body=r"""La figure trace $\bar\alpha$ (à gauche) et $\ln \operatorname{SNR}$ (à droite) de quatre plannings, en fonction de la **fraction de la chaîne** parcourue, (indice + 1)/$T$, pour les comparer malgré leurs $T$ différents : **L**, le linéaire de Ho et coll. ($T = 1\,000$) ; **C**, le cosinus ($T = 1\,000$) ; **D**, le linéaire raccourci à $T = 400$ avec les bornes par défaut ; **S**, le linéaire à $T = 400$ avec les bornes multipliées par $1\,000/T$, celui du notebook en mode rapide. Dessous, le même chiffre et le même bruit aux indices 0, 99, 199, …, 999 de L, puis de C. Rien ne dépend de ton code. Réponds en lisant la figure :
- a) `same_curve_15` : laquelle des deux courbes à $T = 400$ suit celle de L : `"D"` ou `"S"` ?
- b) `largest_end_15` : quel planning garde **le plus** de l'image à son dernier indice : `"L"`, `"C"`, `"D"` ou `"S"` ?
- c) `cos_above_15` : sur quelle part de la chaîne le $\ln \operatorname{SNR}$ de C est-il au-dessus de celui de L ? `"A"` moins de la moitié ; `"B"` entre la moitié et 90 % ; `"C"` plus de 95 %.

Dans tes notes : avec L, à partir de quelle colonne de la grille ne distingues-tu plus le chiffre, et avec C ? Que gagne le planning cosinus, en nombre de pas « utiles » (où il reste quelque chose à débruiter), d'après la figure et ✏️ B5.3 ? Pourquoi D est-il un piège, et que verrait un modèle entraîné avec lui, puis échantillonné à partir d'un bruit pur ?""",
       given=C.GIVEN_15, todo=C.TODO_15, check=C.CHECK_15,
       solution=(f'same_curve_15, largest_end_15, cos_above_15 = "{P15["same_curve"]}", "{P15["largest_end"]}", '
                 f'"{P15["cos_above"]}"\n\n' + C.CHECK_15),
       record=(f'wb.record("B5.15a", "{P15["same_curve"]}", mistakes={{"compare les deux courbes à T = 400 à la courbe L '
               f'du graphique de gauche : laquelle se superpose ?": "D"}})\n'
               f'wb.record("B5.15b", "{P15["largest_end"]}", mistakes={{'
               '"lis la fin des courbes du graphique de droite : la courbe L finit-elle le plus haut ?": "L", '
               '"lis la fin des courbes du graphique de droite : la courbe C finit-elle le plus haut ?": "C", '
               '"lis la fin des courbes du graphique de droite : la courbe S finit-elle le plus haut ?": "S"})\n'
               f'wb.record("B5.15c", "{P15["cos_above"]}", mistakes={{'
               '"suis les courbes C et L du graphique de droite : où se croisent-elles ?": "A", '
               '"suis les courbes C et L du graphique de droite jusqu\'au dernier indice : où se croisent-elles ?": "B"})')),

    Ex("B5.16", "🔨", 2, 30, "q_posterior_mean_variance et predict_x0_from_eps",
       "écrire la moyenne et la variance du posterior d'un pas, et l'estimation de $\\mathbf{x}_0$ à partir d'un bruit.",
       "Ex B5.14 · ∂ B5.7 · fiche §205.3 (« Le posterior : un pas en arrière quand on connaît $\\mathbf{x}_0$ », « Prédire "
       "le bruit »)", tracks="R, M, C", mylearn="diffusion.py",
       body=mylearn_howto() + r"""

**Contrat**
- `q_posterior_mean_variance(x0, x_t, t, schedule)` rend le couple `(mean, variance)` :
  - `schedule` est le dictionnaire de `noise_schedule` (B5.12) ; tu y lis `posterior_mean_coef1`, `posterior_mean_coef2` et `posterior_variance` à l'indice de chaque exemple ;
  - `mean` a la forme de `x0` ; `variance` a la forme `(N, 1, …, 1)`, avec **autant d'axes que `x0`** (`(N,)` si `x0` est 1-D), et vaut 0 à l'indice 0 ;
  - mêmes règles que `q_sample` pour `t` (un indice entier par exemple, dans $[0, T)$) ; `x_t` a **exactement** la forme de `x0`, sans diffusion.
- `predict_x0_from_eps(x_t, t, eps, alpha_bars)` : $\hat{\mathbf{x}}_0 = \big(\mathbf{x}_t - \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}\big) / \sqrt{\bar\alpha_t}$, de la forme de `x_t` ; `eps` a exactement la forme de `x_t` ; mêmes règles pour `t`. L'aller-retour `predict_x0_from_eps(q_sample(x0, t, ab, noise)[0], t, noise, ab)` redonne `x0`.

La première cellule de vérification essaie tes deux fonctions sur les exemples des docstrings, les formes de la variance en 4-D et en 1-D, un **Bayes numérique** en dimension 1 (le produit des deux densités gaussiennes de ∂ B5.7, normalisé sur une grille de 600 001 points, dont on lit la moyenne et la variance), l'aller-retour avec le vrai bruit et quatre refus. La seconde lance les 29 tests de l'exercice (`-k "test_q_posterior_mean_variance_ or test_predict_x0_from_eps_"`).

Dans tes notes : à l'indice 0, que valent la moyenne et la variance du posterior, et pourquoi ? À combien près la moyenne du Bayes numérique et ta formule s'accordent-elles (la ligne sous les vérifications), et pourquoi une grille suffit-elle ici ? Que devient l'erreur de $\hat{\mathbf{x}}_0$ quand le bruit prédit se trompe de 0,1, au pas 10 puis au dernier pas du planning du notebook ?""",
       check=C.RELOAD + C.CHECK_16, solution=C.CHECK_16,
       after=tests_after("B5.16", "mylearn.diffusion.predict_x0_from_eps([0.0], [0], [0.0], [0.5])",
                         "test_q_posterior_mean_variance_ or test_predict_x0_from_eps_", 29)),

    Ex("B5.17", "🔨", 3, 40, "ddpm_step et ddim_step",
       "écrire un pas d'échantillonnage DDPM et un pas DDIM, avec leurs cas particuliers.",
       "Ex B5.16 · ✏️ B5.4, B5.5 · fiche §205.5 (« L'algorithme », « Un pas à la main », le piège « l'échelle et le "
       "dernier pas »), §205.6 (« L'idée de DDIM », « Le paramètre η »)", tracks="R, M, C", mylearn="diffusion.py",
       body=mylearn_howto() + r"""

**Contrat**
- `ddpm_step(x_t, t, eps_pred, schedule, rng=None, variance="posterior", clip_x0=None)` rend $\mathbf{x}_{t-1}$, de la forme de `x_t` :
  - `t` est un **entier Python** de $[0, T)$, le même pour tout le batch ; −1 est refusé ;
  - `variance` vaut exactement `"posterior"` ($\tilde\beta_t$) ou `"beta"` ($\beta_t$), et se vérifie **même à $t = 0$** : la clause *Raises* de la docstring est inconditionnelle ;
  - sans `clip_x0`, la moyenne de l'algorithme 2 de Ho et coll. ; avec `clip_x0`, $\hat{\mathbf{x}}_0$ écrêté à `[-clip_x0, clip_x0]` (la borne reçue, pas 1), puis la moyenne du posterior ;
  - à $t > 0$, on ajoute `sqrt(var) * z` : **l'écart-type**, pas la variance ; $\mathbf{z}$ est tiré de `rng`, de la forme de `x_t`, sans dépendre des valeurs de `x_t` ni de `eps_pred` (deux appels avec la même graine tirent le même $\mathbf{z}$ : c'est ainsi que les tests isolent la moyenne) ; **aucun bruit à $t = 0$**.
- `ddim_step(x_t, t, t_prev, eps_pred, alpha_bars, eta=0.0, rng=None)` rend $\mathbf{x}_{t_{\text{prev}}}$ :
  - `t_prev = -1` veut dire « l'image propre » : $\bar\alpha_{\text{prev}} = 1$ ; **ne lis pas `alpha_bars[-1]`** ;
  - `ValueError` si $-1 \le t_{\text{prev}} < t < T$ n'est pas vérifié (`t_prev >= t` refusé), ou si `eta` est hors de $[0, 1]$, `nan` compris ;
  - avec `eta = 0`, **aucun nombre n'est tiré de `rng`** (les tests vérifient l'état du générateur).

La première cellule de vérification essaie `ddpm_step` à $t = 0$ (l'exemple de la docstring), sa moyenne à l'indice 2 (deux appels à la même graine, l'un avec `x_t` et `eps_pred` nuls, dont la différence est la moyenne), le rapport des bruits des deux variances, l'écrêtage à 0,5 et trois refus ; puis `ddim_step` avec le vrai bruit (vers −1, puis vers l'indice 0), l'état de `rng` avec `eta = 0` et trois refus. La seconde lance les 101 tests de l'exercice (`-k "test_ddpm_step_ or test_ddim_step_"`).

Dans tes notes : pourquoi n'ajoute-t-on pas de bruit au dernier pas, et que verrais-tu sur des images si tu en ajoutais (fiche, piège « l'échelle et le dernier pas ») ? Pourquoi `alpha_bars[-1]` serait-il une erreur silencieuse pour `t_prev = -1` ? Avec `eta = 1` et `t_prev = t - 1`, que vaut $\sigma^2$, et qu'en conclus-tu sur le lien entre DDIM et DDPM ?""",
       check=C.RELOAD + C.CHECK_17, solution=C.CHECK_17,
       after=tests_after("B5.17", "mylearn.diffusion.ddim_step([0.0], 1, 0, [0.0], [0.9, 0.8])",
                         "test_ddpm_step_ or test_ddim_step_", 101)),

    Ex("B5.18", "🔬", 2, 25, "Avec le vrai bruit, DDPM et DDIM retrouvent-ils x_0 ? Contrôle sur des données 2D",
       "enchaîner tes pas DDPM et DDIM avec le vrai bruit, pour savoir lesquels doivent retrouver $\\mathbf{x}_0$ "
       "exactement : le contrôle de ta librairie avant tout entraînement.",
       "Ex B5.17 · fiche §205.5 (« L'algorithme »), §205.6 (« L'idée de DDIM », « Le paramètre η »)", tracks="M, C",
       body=r"""Un réseau parfait prédirait le **vrai** bruit. Sans réseau, on peut donc éprouver une chaîne d'échantillonnage : partir d'une donnée $\mathbf{x}_0$ connue, la bruiter jusqu'au dernier indice, puis remonter en donnant aux pas « le vrai bruit » à la place de $\boldsymbol{\varepsilon}_\theta$. La fiche dit qu'**un** pas DDIM vers l'indice −1 rend alors $\mathbf{x}_0$. Mais quel est « le vrai bruit » au milieu d'une chaîne, et qu'en est-il des chaînes entières ?

**Contrat**
- `true_noise_18(x_t, t, x0, alpha_bars)` : le bruit $\boldsymbol{\varepsilon}$ tel que $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}$, pour un indice `t` entier (le même pour tout le batch) ; un nouveau tableau, de la forme de `x_t` ;
- `ddpm_chain_18(x0, eps, schedule, rng, oracle=False)` : $\mathbf{x}$ à l'indice $T - 1$ fait de `x0` et `eps` par la forme fermée, puis ta `ddpm_step` de l'indice $T - 1$ à 0 (avec `rng`, sans écrêtage, la variance par défaut), avec pour `eps_pred` le bruit **initial** `eps` à chaque pas (`oracle=False`), ou le vrai bruit du $\mathbf{x}$ **courant**, `true_noise_18(x, t, x0, …)` (`oracle=True`) ; rend le $\mathbf{x}$ final ;
- `ddim_chain_18(x0, eps, alpha_bars, timesteps, eta=0.0, rng=None)` : $\mathbf{x}$ à l'indice `timesteps[0]` fait de `x0` et `eps`, puis ta `ddim_step` de chaque indice de la sous-séquence décroissante `timesteps` au suivant, et du dernier à −1, toujours avec `eps_pred = eps`, le bruit initial ; rend le $\mathbf{x}$ final.

La cellule de vérification essaie tes trois fonctions sur un planning de 3 pas, contre la chaîne écrite pas à pas. Puis l'expérience prend 500 points de l'anneau de 8 gaussiennes (`wbd.ring`), le planning du notebook (ta `noise_schedule` de `BETAS`), et mesure l'écart maximal entre $\mathbf{x}_0$ et la fin de cinq chaînes : DDPM avec le bruit initial ; DDPM avec le vrai bruit du $\mathbf{x}$ courant ; DDIM ($\eta = 0$) sur tous les indices ; DDIM ($\eta = 0$) sur 10 indices (`wbd.ddim_timesteps(T_STEPS, 10)`) ; DDIM ($\eta = 1$) sur tous les indices. Réponds ensuite, d'après le tableau, par `True` ou `False` : la chaîne retrouve-t-elle $\mathbf{x}_0$ à $10^{-6}$ près ?
- a) `ddpm_fixed_18` : DDPM, bruit initial à chaque pas ;
- b) `ddpm_oracle_18` : DDPM, vrai bruit du $\mathbf{x}$ courant ;
- c) `ddim_ten_18` : DDIM, $\eta = 0$, 10 pas ;
- d) `ddim_eta1_18` : DDIM, $\eta = 1$, tous les pas.

Dans tes notes : pourquoi le bruit initial n'est-il plus « le vrai bruit » d'un $\mathbf{x}_t$ de la chaîne DDPM après le premier pas, alors qu'il le reste tout au long de la chaîne DDIM avec $\eta = 0$ ? Pourquoi la chaîne DDPM avec le vrai bruit courant finit-elle exactement sur $\mathbf{x}_0$, quelle que soit sa trajectoire (regarde le dernier pas) ? Pourquoi le nombre de pas de DDIM ne change-t-il rien ici, alors qu'il comptera avec un réseau ? Quelle erreur de ta librairie ce contrôle aurait-il attrapée (un indice décalé, $\bar\alpha$ au lieu de $\sqrt{\bar\alpha}$), et laquelle pas ?""",
       todo=C.TODO_18, check=C.CHECK_18, solution=C.SOLUTION_18 + "\n\n\n" + C.ANSWERS_18 + "\n\n\n" + C.CHECK_18,
       record=(f'wb.record("B5.18a", {B18["a"]}, mistakes={{"relis l\'écart de la première ligne du tableau": {not B18["a"]}}})\n'
               f'wb.record("B5.18b", {B18["b"]}, mistakes={{"relis l\'écart de la deuxième ligne du tableau": {not B18["b"]}}})\n'
               f'wb.record("B5.18c", {B18["c"]}, mistakes={{"relis l\'écart de la quatrième ligne du tableau": {not B18["c"]}}})\n'
               f'wb.record("B5.18d", {B18["d"]}, mistakes={{"relis l\'écart de la dernière ligne du tableau": {not B18["d"]}}})')),
])

# ---------------------------------------------------------------------------------------------------------------------
def _full_20() -> str:
    ddim_fast = TOY.get("ddim_400", [])
    ddim_full = TOY.get("ddim_1000", [])
    euler = TOY.get("euler", [])
    return ("FULL_20 = {\n"
            f'    "seeds": {TOY.get("seeds", [])},\n'
            f'    "DDIM (bruit prédit, B5.19)": {[[round(v, 4) for v in row] for row in ddim_fast]} if FAST_MODE else '
            f'{[[round(v, 4) for v in row] for row in ddim_full]},\n'
            f'    "Euler (vitesse prédite)": {[[round(v, 4) for v in row] for row in euler]},\n'
            "}")


GIVEN_20 = (f"STEPS_20 = {STEPS_20}                       # numbers of sampling steps of the comparison\n"
            "# The full protocol (tools/chapters/history_chB5.py): the same cells with 5 seeds (seed 0 is yours), the\n"
            "# distance of 2 000 points to REAL_B for each number of steps; DDIM depends on T, so on the mode\n"
            + _full_20())

PART_B = Part("B", "Jouets 2D : un MLP débruiteur, puis le flow matching",
              "Avant les images, des points du plan : l'**anneau de 8 gaussiennes** (`wbd.ring` : 8 centres sur un cercle "
              "de rayon 1,5, écart-type 0,15 autour de chacun ; sa variance par coordonnée, 1,15, est proche de celle "
              "du bruit). Tout s'y voit d'un coup d'œil et s'entraîne en une dizaine de secondes. Tu y entraînes un MLP "
              "qui prédit le bruit, conditionné par le pas comme le U-Net de la fiche, puis tu l'échantillonnes avec "
              "ta `ddpm_step` (B5.19) ; puis tu entraînes le **même** MLP à prédire une vitesse (flow matching, fiche "
              "§205.9) et tu compares les deux échantillonneurs à nombre de pas égal (B5.20).\n\n"
              "**Mesurer la qualité.** Une distance de Fréchet en 2D ne compare que les moyennes et les covariances : "
              "un nuage gaussien de variance 1 a presque celles de l'anneau. On prend la **distance de Wasserstein en "
              "tranches** (`wbd.sliced_wasserstein`) : la moyenne, sur 64 directions du plan, de la distance de "
              "Wasserstein du ch. 27 entre les deux projections (l'écart moyen de leurs valeurs triées). La fonction "
              "`quality(points)` rend cette distance à `REAL_B` (2 000 points de l'anneau), la part des points à moins "
              "de trois écarts-types d'un centre et le nombre de modes atteints (`wbd.mode_report`) ; la cellule "
              "suivante affiche ses deux repères, deux échantillons de l'anneau et du bruit pur.\n\n"
              "**Encoder le pas.** Le MLP reçoit une **position entière**, comme le U-Net : l'indice $t$ du pas pour la "
              "diffusion ; pour le flow matching, dont le temps $s$ est continu sur $[0, 1]$, on le prend sur une "
              "grille de 1 000 valeurs, $s = u / 1\\,000$ avec $u$ entier de 1 à 1 000 (`wbd.FM_POSITIONS` = 1 001 "
              "positions). La position $p$ est encodée par la ligne $p$ de ta `positional_encoding` de B3 "
              "(`mylearn.attention`) : si tu n'as pas écrit `attention.py`, la référence la remplace (une ligne ℹ️ le "
              "dit) ; s'il existe sans cette fonction, la vérification te le signale. La référence d'`attention.py` "
              "s'appuie elle-même sur celles de `nn/activations.py` (ch. 17) et de `nn/regularization.py` (ch. 20) : "
              "si tu n'as pas ces modules, deux autres lignes ℹ️ le disent, sans effet sur la partie B.\n\n"
              "`SETTINGS_B` donne le budget des deux entraînements : 3 000 itérations, des batches de 512 points, "
              "AdamW au learning rate 2e-3 (environ 10 s chacun sur un CPU à deux cœurs). `wbd` contient une solution "
              "de référence (`Denoiser2D`, `train_toy`, `sample_toy_ddpm`…) : la vérification s'en sert pour comparer "
              "tes fonctions ; ne la lis qu'une fois les tiennes écrites.",
              given=C.GIVEN_B, exercises=[
    Ex("B5.19", "📦", 2, 30, "Diffusion en 2D : un MLP débruiteur conditionné par le pas de temps, entraîné puis échantillonné",
       "entraîner un MLP à prédire le bruit de points 2D, puis générer de nouveaux points avec ta `ddpm_step`.",
       "Ex B5.17 · B3 (`positional_encoding`, B3.15) · fiche §205.3 (« La loss simplifiée »), §205.4 (« Dire au réseau "
       "quel est le pas »), §205.5 (« L'algorithme »)", tracks="R, M, C",
       body=r"""**Contrat**
- `Denoiser2D_19(n_positions, t_dim=32, hidden=128)`, un `nn.Module` :
  - un buffer **non persistant** `pe` (`register_buffer(..., persistent=False)` : il n'entre pas dans le `state_dict`), la table `positional_encoding(n_positions, t_dim)` de ta librairie (B3) en float32 ;
  - `net = nn.Sequential(nn.Linear(2 + t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))` ;
  - `forward(x, t)` : `x` de forme `(N, 2)` en float32, `t` de forme `(N,)` en int64 → `net` appliqué aux coordonnées **suivies** de l'encodage de la position, `(N, 2)`.
- `train_19(model, data, alpha_bars, n_iters, batch_size, lr, generator)` entraîne `model` sur place et rend la liste des `n_iters` loss (des floats Python) :
  - un `torch.optim.AdamW(model.parameters(), lr=lr)`, avec ses autres réglages par défaut ;
  - à chaque itération, **dans cet ordre**, trois tirages par `generator` : les indices du batch `torch.randint(len(data), (batch_size,))`, les pas `torch.randint(0, T, (batch_size,))` ($T$ = `len(alpha_bars)`), le bruit `torch.randn((batch_size, 2))` ; puis $\mathbf{x}_t$ par la forme fermée en float32, la loss `F.mse_loss(model(x_t, t), eps)`, les gradients et un pas de l'optimiseur.
- `sample_19(model, n, schedule, generator, rng)` : `n` points par l'échantillonnage ancestral : $\mathbf{x}$ = `torch.randn((n, 2), generator=generator)` converti en float64 NumPy, puis, pour $t$ de $T - 1$ à 0, la prédiction du réseau (sans gradient, en float32) et ta `ddpm_step(x, t, eps, schedule, rng=rng)` (variance du posterior, sans écrêtage) ; rend un tableau float64 `(n, 2)`.

La cellule de vérification contrôle ton MLP (noms et formes du `state_dict`, le buffer `pe`, la sortie aux mêmes poids que le modèle du contrat), 5 itérations de `train_19` contre la recette du contrat (mêmes tirages, mêmes loss) et `sample_19` sur 6 points. La cellule suivante entraîne ton MLP avec les réglages de `SETTINGS_B` sur `DATA_B`, génère 2 000 points, trace la loss et les points, et vérifie qu'ils couvrent les 8 modes (une distance en tranches sous 0,2, plus de 80 % des points sur un mode). Le temps affiché est une mesure, pas une réponse.

Dans tes notes : la loss descend-elle vers 0 ? Pourquoi pas, alors que le modèle génère de bons points (pense à ce que vaut l'erreur sur $\boldsymbol{\varepsilon}$ aux petits indices, où $\mathbf{x}_t$ ne dit presque rien du bruit) ? Combien de passes du réseau a coûté l'échantillonnage ? Que se passe-t-il si tu supprimes l'encodage du pas (des zéros à la place) ?""",
       todo=C.TODO_19, check=C.CHECK_19, solution=C.SOLUTION_19 + "\n\n\n" + C.CHECK_19,
       after=[("code", C.EXPERIMENT_19)]),

    Ex("B5.20", "🔬", 3, 40, "Flow matching en 2D : même MLP, cible « vitesse » et intégration d'Euler",
       "entraîner le même MLP à prédire une vitesse, l'intégrer par la méthode d'Euler, et comparer la qualité au DDIM "
       "du modèle de B5.19 selon le nombre de pas, sans présumer du gagnant.",
       "Ex B5.19 · fiche §205.9 (« Flow matching et rectified flow », encadré « la méthode d'Euler »), §205.6 (« L'idée "
       "de DDIM »)", tracks="M, C",
       body=r"""Le flow matching relie la donnée et le bruit par la ligne droite $\mathbf{x}_s = (1 - s)\,\mathbf{x}_0 + s\,\boldsymbol{\varepsilon}$ (fiche §205.9) et fait prédire au réseau la vitesse $\mathbf{v} = \boldsymbol{\varepsilon} - \mathbf{x}_0$ ; on génère en suivant le champ des vitesses de $s = 1$ (le bruit) à $s = 0$. La question : à nombre de pas égal, ces points valent-ils mieux que ceux de DDIM avec le modèle de B5.19, qui prédit le bruit ? Même MLP, même budget d'entraînement, mêmes points de départ.

**Contrat**
- `train_fm_20(model, data, n_iters, batch_size, lr, generator)` : la recette de `train_19`, à trois différences près : le deuxième tirage est la position `u = torch.randint(1, 1001, (batch_size,))`, et $s = u / 1\,000$ ; l'entrée est $\mathbf{x}_s$ ; la cible est $\boldsymbol{\varepsilon} - \mathbf{x}_0$, et le modèle reçoit la position `u` (un `Denoiser2D_19(wbd.FM_POSITIONS)`, 1 001 positions). Rend la liste des loss.
- `euler_20(model, n, n_steps, generator)` : $\mathbf{x}$ = `torch.randn((n, 2), generator=generator)`, puis `n_steps` pas d'Euler égaux, de $s = 1$ vers $s = 0$ : au pas $k$ (de 0 à `n_steps` − 1), la position `round(1000 * (1 - k / n_steps))` et $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{v}/\text{n\_steps}$ ; sans gradient ; rend un tableau float64 `(n, 2)`.
- `ddim_20(model, n, n_steps, alpha_bars, generator)` : $\mathbf{x}$ = `torch.randn((n, 2), generator=generator)` en float64 NumPy, puis ta `ddim_step` ($\eta = 0$) d'un indice au suivant de `wbd.ddim_timesteps(T, n_steps)` (de $T - 1$ à 0, `n_steps` indices), et du dernier à −1 ; rend un tableau float64 `(n, 2)`.

La cellule de vérification compare tes trois fonctions aux fonctions du contrat (5 itérations, puis 1, 3 et 8 pas d'Euler, 1, 4 et 50 pas de DDIM). La cellule suivante (environ 15 s) entraîne le modèle de flow matching avec `SETTINGS_B`, puis mesure la distance en tranches de 2 000 points de chaque échantillonneur pour chaque nombre de pas de `STEPS_20`, avec le modèle de B5.19 pour DDIM ; elle trace les deux courbes, avec, en bandes, les **cinq graines** du protocole complet (`FULL_20`, précalculé avec les mêmes cellules ; ta graine est la première), et montre les points à 5 pas. Réponds ensuite d'après `FULL_20` (une liste par graine, une distance par nombre de pas de `STEPS_20` ; le tableau imprimé n'en donne que les médianes, les bandes du graphique l'étendue), **pour toutes ses graines** :
- a) `fewest_20` : le plus petit nombre de pas de `STEPS_20` à partir duquel la distance d'Euler est au plus le triple de sa distance à 100 pas ;
- b) `better5_20` : à 5 pas, lequel a la plus petite distance : `"D"` (DDIM) ou `"E"` (Euler) ?

Dans tes notes : pourquoi DDIM à 1 ou 2 pas rend-il des points aussi loin (regarde $\hat{\mathbf{x}}_0$ au dernier indice, et la division par $\sqrt{\bar\alpha_t}$) ? Et Euler à 1 pas, où envoie-t-il tous les points (fiche, encadré 🧮 « la MSE apprend une moyenne ») ? Le gagnant change-t-il avec le nombre de pas ? À 100 pas, compare le DDIM et le DDPM de B5.19 sur le même modèle : que conclus-tu de l'aléa de l'échantillonnage ancestral ? Qu'est-ce que la comparaison ne dit pas (un seul dataset, des réglages choisis pour l'un ou l'autre, la loss de chaque modèle) ?""",
       given=GIVEN_20, todo=C.TODO_20, check=C.CHECK_20, solution=C.SOLUTION_20 + "\n\n\n" + C.CHECK_20,
       after=[("code", C.EXPERIMENT_20), ("todo", C.TODO_20B), ("check", C.CHECK_20B),
              ("solution", f'fewest_20, better5_20 = {A20["fewest"]}, "{A20["better5"]}"\n\n' + C.CHECK_20B),
              ("record", f'wb.record("B5.20a", {A20["fewest"]}, mistakes={{'
                         + ", ".join(f'"relis les distances d\'Euler de FULL_20 : à {n} pas, la distance est-elle au plus '
                                     f'le triple de celle à 100 pas pour toutes les graines ?": {n}'
                                     for n in STEPS_20 if n != A20["fewest"])
                         + f'}}, fractional="on demande un nombre de pas de STEPS_20", choices={list(STEPS_20)})\n'
                         f'wb.record("B5.20b", "{A20["better5"]}", mistakes={{"relis la colonne des 5 pas de FULL_20, '
                         f'graine par graine": "{"D" if A20["better5"] == "E" else "E"}"}})')]),
])

# ---------------------------------------------------------------------------------------------------------------------
PART_C = Part("C", "Le U-Net sur MNIST et Fashion-MNIST",
              "Place aux images. Tu écris d'abord le **U-Net minuscule** de la fiche (B5.21), que la suite de la partie "
              "entraîne sur MNIST, échantillonne, conditionne par la classe et évalue. Les images sont ramenées dans "
              "$[-1, 1]$ par une seule fonction, `wbd.to_signed` ($x / 127{,}5 - 1$ pour des octets) : le réseau, la "
              "forme fermée et l'écrêtage de $\\hat{\\mathbf{x}}_0$ supposent tous cette échelle, et `wbd.to_unit` les "
              "ramène dans $[0, 1]$ pour l'affichage. Les données d'entraînement sont un sous-ensemble **figé** "
              "(`wbd.image_subset`) : 10 000 images en mode rapide, les 60 000 en mode complet, toujours dans le même "
              "ordre, de sorte que les petits sous-ensembles sont inclus dans les grands.",
              exercises=[
    Ex("B5.21", "📦", 3, 50, "Un U-Net minuscule en PyTorch : blocs convolutifs, connexions de saut, encodage du pas de temps",
       "écrire le U-Net de la fiche, avec son encodage du pas et une classe facultative, et le valider étage par étage "
       "contre le modèle du contrat.",
       "Ex B5.19 · B3 (`positional_encoding`) · ch. 21 (convolutions de stride 2, convolution transposée) · fiche §205.4 "
       "(« Le U-Net et ses connexions de saut », « Un bloc résiduel conditionné par le pas ») · §205.7 (« Donner la "
       "classe au réseau »)", thread="MNIST", tracks="R, C",
       body=r"""**Contrat.** Les noms des modules comptent : la suite du chapitre sauvegarde ton modèle par son `state_dict` et le recharge.
- `ResBlock_21(c_in, c_out, t_dim)` : le bloc de la fiche (§205.4), avec ses six sous-modules `norm1` (`nn.GroupNorm(8, c_in)`), `conv1` (3×3, `padding=1`), `time` (`nn.Linear(t_dim, c_out)`), `norm2`, `conv2` et `skip` (une convolution 1×1 si `c_in != c_out`, sinon `nn.Identity()`) ; `forward(x, temb)` : $\mathbf{h}$ = `conv1(silu(norm1(x)))`, plus `time(temb)` en un biais par canal, puis `conv2(silu(norm2(h)))`, plus `skip(x)`.
- `TinyUNet_21(n_steps, channels=(32, 64, 128), t_dim=128, n_classes=None)`, avec `c1, c2, c3 = channels` :
  - `self.config`, le dictionnaire `{"n_steps", "channels" (une liste), "t_dim", "n_classes"}` de ses arguments : il suffit à reconstruire le modèle (`TinyUNet_21(**config)`) ;
  - un buffer **non persistant** `pe` : la table `positional_encoding(n_steps, t_dim)` de B3, en float32 ; `t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))` ;
  - si `n_classes` est un entier : `class_emb = nn.Embedding(n_classes + 1, t_dim)`, dont la **dernière** ligne, d'indice `n_classes`, est la classe vide $\varnothing$ de la guidance sans classifieur ;
  - `inp` (1 → `c1`, 3×3) ; l'encodeur `d1 = ResBlock_21(c1, c1)`, `down1` (convolution 3×3 de stride 2, `padding=1`, `c1` → `c1` : 28 → 14), `d2 = ResBlock_21(c1, c2)`, `down2` (la même convolution 3×3 de stride 2, `padding=1`, `c2` → `c2` : 14 → 7) ; `mid = ResBlock_21(c2, c3)` ; le décodeur `up2` (`nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)` : 7 → 14), `u2 = ResBlock_21(2 * c2, c2)`, `up1` (la même convolution transposée, `nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)` : 14 → 28), `u1 = ResBlock_21(2 * c1, c1)` ; `out` (`c1` → 1, 3×3) ; tous les blocs reçoivent `t_dim` ;
  - `forward(x, t, y=None)` : `x` de forme `(N, 1, 28, 28)`, `t` des indices int64 `(N,)`, `y` des classes int64 `(N,)` de 0 à `n_classes` (la classe vide si `y` est `None`) ; `temb = t_mlp(pe[t])`, plus `class_emb(y)` si le modèle a des classes ; puis `h1 = d1(inp(x))`, `h2 = d2(down1(h1))`, `h = mid(down2(h2))`, `h = u2(cat([up2(h), h2]))`, `h = u1(cat([up1(h), h1]))` (la concaténation sur l'axe des canaux, **la sortie du décodeur d'abord**), et `out(h)` : la prédiction du bruit, `(N, 1, 28, 28)`.

Puis, dans la même cellule, avec ton modèle :
- a) `n_small_21` : le nombre de paramètres de `TinyUNet_21(400, channels=(16, 32, 64), t_dim=64)` ;
- b) `u1_in_21` : la forme de l'entrée de `u1` (après la concaténation) pour ce modèle et un batch de 8 images, un tuple de 4 entiers.

La cellule de vérification compare ton bloc à celui du contrat aux mêmes poids, puis, pour trois réglages (celui de la fiche, `(16, 32, 64)` avec `t_dim=64`, et 10 classes), les noms et les formes du `state_dict`, le buffer `pe`, et la sortie de ton U-Net à celle du U-Net du contrat (`wbd.TinyUNet`) aux mêmes poids, éloignés de l'initialisation ; elle compare enfin la forme de chaque étage (par des *hooks*, `wbd.stage_shapes`) et vérifie tes deux réponses. La cellule suivante choisit le modèle de la suite du chapitre, `UNET_CLASS` : le tien s'il a tout passé, sinon celui du contrat, et le dit ; elle mesure une passe sur un batch de 64 images (une mesure, pas une réponse).

Dans tes notes : combien de canaux reçoit `u2`, et pourquoi la convolution qui suit doit-elle les attendre (fiche, piège « deux *skip connections* ») ? Que perdrait le U-Net sans ses connexions de saut ? Pourquoi une group norm plutôt qu'une batchnorm ici ? Pourquoi `pe` est-il un buffer non persistant, et que se passerait-il en rechargeant un `state_dict` avec un autre `n_steps` ? D'après ta mesure, combien de temps coûteraient 400 pas de DDPM pour 64 images, et 50 pas de DDIM ?""",
       todo=C.TODO_21, check=C.CHECK_21, solution=C.SOLUTION_21 + "\n\n\n" + C.CHECK_21,
       record="RECORD_21_PLACEHOLDER",
       after=[("code", C.EXPERIMENT_21)]),
])

PARTS = [PART_A, PART_B, PART_C]
