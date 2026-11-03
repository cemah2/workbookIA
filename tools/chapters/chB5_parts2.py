"""Second half of part C of the chapter B5 notebook (B5.22 to B5.31): statements and cells (used by build_chB5.py, which
appends EXERCISES_C2 to PART_C.exercises).

B5.22 repairs a DDPM pipeline with a perfect denoiser (no training); B5.23 trains the U-Net of B5.21 on MNIST and saves
it with its provenance; B5.24 to B5.30 reload it (TOOLS_C2, with an explicit fallback on the reference model), sample it,
evaluate it and wrap it in a script; B5.26 and B5.27 condition a second U-Net on the classes of Fashion-MNIST; B5.31 is
the challenge. The long protocols are precomputed by history_chB5.py (history_chB5.json; the table of the 🔮 B5.26 is
read at run time from wb.diffusion.precomputed, so that its source shows no answer).
"""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, guarded

import chB5_code2 as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB5.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}


def fill(text: str, values: dict) -> str:
    """Replace each «NAME» of a cell by its value (repr of floats rounded to 4 significant digits)."""
    for name, value in values.items():
        text = text.replace(f"«{name}»", value if isinstance(value, str) else repr(value))
    if "«" in text:
        raise ValueError(f"unfilled placeholder in a cell: {text[text.index('«'):text.index('«') + 40]}")
    return text


EX_22 = Ex("B5.22", "🐛", 2, 25, "Le DDPM qui ne produit que du gris : quatre bugs d'indices, de racines et d'échelle",
           "trouver par des diagnostics enchaînés les quatre bugs d'un pipeline DDPM qui ne plante jamais.",
           "Ex B5.18, B5.21 · fiche §205.2 (« La forme fermée »), §205.5 (« L'algorithme », piège « l'échelle et le "
           "dernier pas ») · tableau des pièges", thread="MNIST", tracks="R, C",
           body=r"""Un collègue a écrit le pipeline d'un DDPM : `prepare_22` prépare les images, `noisy_22` les bruite par la forme fermée (c'est ce que verrait le réseau pendant l'entraînement), `sample_22` échantillonne par l'algorithme 2 de Ho et coll. (variance $\beta_t$). Rien ne plante, mais ses échantillons ne sont que du gris. Quatre bugs s'y cachent, parmi les familles du tableau des pièges de la fiche : un décalage d'indice, $\bar\alpha$ là où il faut $\sqrt{\bar\alpha}$, l'échelle des images, le dernier pas.

Pour voir l'effet du pipeline sans entraîner de réseau, la cellule de l'expérience utilise un **débruiteur parfait** (`wbd.ideal_denoiser`) : pour un dataset fini (ici 500 chiffres de MNIST, `RAW_22`, tels qu'ils sont stockés, des octets de 0 à 255), le bruit qu'un réseau entraîné sans fin sur ces images apprendrait, calculé exactement pour le processus direct que **ton** `noisy_22` définit. Elle montre 8 échantillons de ton pipeline, tel quel puis réparé.

**Ce que tu fais** : répare les trois fonctions **sur place** (mêmes noms, mêmes signatures ; garde la variance $\beta_t$ de l'algorithme 2), puis relance la cellule de vérification. Elle enchaîne quatre diagnostics, dans l'ordre où les bugs se masquent les uns les autres : le premier qui échoue arrête la chaîne et dit ce qu'il a mesuré, jamais ce qu'il faut écrire.
1. **échelle** : les pixels préparés de 64 chiffres vont-ils de −1 à 1 ?
2. **indices** : à chaque indice $t$, la part du bruit dans `noisy_22` est-elle $\sqrt{1 - \bar\alpha_t}$, celle de la forme fermée ?
3. **coefficients** : à chaque indice, les deux coefficients de `noisy_22` préservent-ils la variance ($c_1^2 + c_2^2 = 1$) ?
4. **aller-retour** : partie de `noisy_22` au dernier indice, la chaîne de `sample_22` à qui l'on donne à chaque pas le **vrai** bruit du $\mathbf{x}$ courant retrouve-t-elle les images (🔬 B5.18) ?

Dans tes notes : pour chaque bug, la ligne fautive et ce que tu as changé. Après la seule réparation du point 1, les échantillons du débruiteur parfait ont-ils l'air justes ? Pourquoi un débruiteur parfait cache-t-il les points 2 et 3, et pourquoi un réseau entraîné avec ce pipeline ne les cacherait-il pas aussi bien ? Les 8 images réparées, les reconnais-tu (compare-les au dataset) : qu'est-ce que cela annonce pour 🔬 B5.28 ?""",
           given=C.GIVEN_22, todo=C.TODO_22, check=C.CHECK_22, solution=C.SOLUTION_22 + "\n\n\n" + C.CHECK_22,
           after=[("code", C.EXPERIMENT_22)])


# ---------------------------------------------------------------------------------------------------------------------
U = HISTORY.get("unet25", {})
_L_FD50 = [row[U.get("steps", [5, 10, 20, 50, 100]).index(50)] for row in U.get("L", [])] or [100.0]
FLOOR_23 = U.get("floor", 15.0)
REF_23 = (min(_L_FD50), max(_L_FD50))
GOAL_23 = float(5 * round(1.3 * REF_23[1] / 5 + 0.5))          # worst seed of the reference + 30 %, rounded up to 5

EX_23 = Ex("B5.23", "📦", 3, 60, "Entraîner un DDPM sur MNIST : boucle, moyenne exponentielle des poids (EMA), courbe de loss",
           "écrire la boucle d'entraînement de la loss simplifiée avec sa copie EMA, choisir $\\rho$, entraîner ton U-Net "
           "et le juger sur des échantillons.",
           "Ex B5.21, B5.22 · ch. 19 (moyenne mobile exponentielle) · fiche §205.3 (« La loss simplifiée », « En "
           "pratique : la moyenne exponentielle des poids », piège « évaluer le mauvais réseau »)", thread="MNIST",
           tracks="R, C",
           body=r"""La cellule suivante donne la recette (`SETTINGS_23`, sans le $\rho$ de l'EMA, que tu choisis), la configuration du U-Net (`CONFIG_23`), le sous-ensemble figé de MNIST (`DATA_23`, dans $[-1, 1]$), `AB` ($\bar\alpha$ du planning du notebook), `N_UPDATES_23` (le nombre de pas d'optimisation de la recette), 2 000 chiffres de test et leurs features (`REAL_MNIST`, `REAL_FEATS`), `frechet_mnist(images)` (la distance de Fréchet de 256 images ou plus à ces chiffres, avec ta `frechet_distance_from_features` du ch. 27) et `X_FIXED`, 16 bruits de départ fixés. En mode rapide, le U-Net a les canaux (16, 32, 64) : celui de la fiche ferait quatre fois moins de pas dans le même temps. 🚀 En mode complet, le U-Net de la fiche, 20 epochs sur les 60 000 images, $T = 1\,000$ : plusieurs heures sur CPU, un GPU de Colab est conseillé.

**Contrat**
- `RHO_23` : le $\rho$ de ta copie EMA, $\theta_{\text{EMA}} \leftarrow \rho\,\theta_{\text{EMA}} + (1 - \rho)\,\theta$ après chaque pas. La fiche cite $0{,}9999$ pour 800 000 pas ; choisis-le pour les `N_UPDATES_23` pas de la recette. La vérification demande qu'à la fin il reste au plus 1 % des poids de départ dans la copie ($\rho^{N} \le 0{,}01$) et qu'elle moyenne au moins 20 pas ($1/(1 - \rho) \ge 20$).
- `ema_update_23(ema, model, rho)` : la formule, pour chaque paramètre de `ema`, **sur place** (les mêmes objets `Parameter`) et sans graphe ; les buffers de `model` sont recopiés dans ceux de `ema`.
- `train_23(model, ema, images, settings, generator, alpha_bars)` entraîne `model` et sa copie `ema` sur place et rend `{"loss": [...]}`, la loss de chaque pas (des floats Python) :
  - `torch.optim.AdamW(model.parameters(), lr=settings["lr"])`, ses autres réglages par défaut ;
  - pour chacune des `settings["epochs"]` epochs, l'ordre `torch.randperm(N, generator=generator)`, découpé en batches de `settings["batch_size"]` (le dernier peut être plus petit) ;
  - pour chaque batch, **dans cet ordre**, deux tirages par `generator` : les indices des pas `torch.randint(0, T, (b,))`, puis le bruit `torch.randn(x0.shape)` ; $\mathbf{x}_t$ par la forme fermée en float32 ; la loss `F.mse_loss(model(x_t, t), eps)` ; les gradients ; un pas de l'optimiseur ; puis `ema_update_23(ema, model, settings["ema"])` ;
  - à la fin, les deux modèles en mode eval.

La cellule de vérification compare `ema_update_23` à la formule, puis 4 pas de `train_23` (2 epochs de 6 images en batches de 4) à la recette du contrat, sur un petit U-Net du contrat : les loss, les poids, la copie EMA ; et elle contrôle ton $\rho$. La cellule suivante entraîne ton U-Net (`UNET_CLASS`, environ 2 min sur un CPU à deux cœurs en mode rapide) et le **sauvegarde** une fois, avec sa copie EMA et sa loss, dans `wbd.checkpoints_dir()` (hors du suivi de git ; sur Colab, `/content/wb_cache/diffusion/`, effacé avec la session), sous une clé qui dit tout de l'entraînement (la configuration, la recette, ton $\rho$, le mode, les images) ; un nouvel entraînement efface d'abord l'ancienne sauvegarde. La dernière **juge** l'entraînement comme la fiche le demande, sur des échantillons et non sur la loss : la courbe de loss, 16 échantillons du même bruit `X_FIXED` avec la copie EMA puis avec les poids bruts (DDIM, 50 pas, $\hat{\mathbf{x}}_0$ écrêté), et la distance de Fréchet de 256 échantillons de la copie EMA, qui doit passer sous `GOAL_23` (la pire graine de la référence plus 30 %). Le temps affiché est une mesure, pas une réponse.

Dans tes notes : pourquoi ce $\rho$ ? Que vaudrait $0{,}9999^{N}$ ici, et que généreraient ces poids ? La loss descend-elle encore à la fin, et pourquoi ne suffit-elle pas à juger le modèle (fiche, piège « juger un entraînement à sa loss ») ? Lequel des deux réseaux génère le mieux, et pourquoi ? Les 16 échantillons ressemblent-ils à des chiffres ? Combien d'entre eux sont lisibles ?""",
           given=C.GIVEN_23 + "\n" + f"FLOOR_23 = {round(FLOOR_23, 1)}    # two sets of 256 real digits (history_chB5.py)\n"
           f"REF_23 = ({round(REF_23[0], 1)}, {round(REF_23[1], 1)})   # the reference (rho = 0.99): best and worst of "
           f"{len(_L_FD50)} seeds\nGOAL_23 = {GOAL_23}     # the distance below which the check accepts the EMA copy",
           todo=C.TODO_23, check=C.CHECK_23,
           solution='CKPT_SOURCE_23 = "solution"   # the solutions notebook saves its models apart: it never overwrites, nor ' \
                    'reloads, the learner\'s checkpoints\n\n' + C.SOLUTION_23 + "\n\n\n" + C.CHECK_23,
           after=[("code", C.EXPERIMENT_23), ("code", C.EXPERIMENT_23B)])

EX_24 = Ex("B5.24", "📦", 2, 30, "Échantillonner : DDPM à 1 000 pas contre DDIM à 50 pas, qualité et temps",
           "écrire les deux échantillonneurs avec ta librairie, et comparer leurs images et leur coût sur ton modèle.",
           "Ex B5.17, B5.23 · fiche §205.5 (« L'algorithme », « Ce que coûte un échantillon »), §205.6 (« L'idée de "
           "DDIM », « Pourquoi c'est plus rapide »)", thread="MNIST", tracks="R, C",
           body=r"""La première cellule relit ton modèle : la **copie EMA** de ton U-Net de B5.23, `MODEL`, depuis sa sauvegarde (elle n'entraîne rien). S'il manque, elle le dit ; `REFERENCE_FALLBACK = True` dans cette cellule prend le U-Net de référence de `wb.diffusion` (entraîné une fois avec la recette de B5.23, environ 2 min). Elle donne aussi `need_model` et `PassCounter`, qui compte les passes d'un réseau. En mode rapide, la chaîne a $T = 400$ pas (1 000 en mode complet) : « DDPM à 1 000 pas » devient DDPM à 400 pas.

**Contrat** (le réseau est appelé `model(x, torch.full((N,), t))` en float32, sans graphe ; entre deux pas, `x` est un tableau float64 de NumPy ; les deux fonctions rendent un tenseur float32 `(N, 1, 28, 28)` et ne modifient pas `x_T`) :
- `sample_ddpm_24(model, x_T, schedule, rng, clip_x0=1.0)` : de l'indice $T - 1$ à 0, une passe du réseau puis **ta** `ddpm_step(x, t, eps, schedule, rng=rng, clip_x0=clip_x0)` (B5.17 ; `schedule` = ta `noise_schedule(BETAS)`) : $T$ passes ;
- `sample_ddim_24(model, x_T, alpha_bars, n_steps, clip_x0=1.0)` : sur la sous-séquence `wbd.ddim_timesteps(T, n_steps)` puis vers −1, à chaque pas une passe du réseau, $\hat{\mathbf{x}}_0$ par **ta** `predict_x0_from_eps`, écrêté à $[-1, 1]$ (si `clip_x0` n'est pas `None`), le bruit **recalculé** à partir de $\hat{\mathbf{x}}_0$ écrêté, $\boldsymbol{\varepsilon} = (\mathbf{x} - \sqrt{\bar\alpha_t}\,\hat{\mathbf{x}}_0)/\sqrt{1 - \bar\alpha_t}$, puis **ta** `ddim_step` avec ce bruit : `n_steps` passes.

La cellule de vérification compare tes deux fonctions aux échantillonneurs du contrat (`wbd.sample_ddim`, `wbd.sample_ddpm`) sur 4 images et le même bruit de départ, et compte les passes du réseau. La suivante échantillonne 32 images d'un même bruit de départ par les deux méthodes (DDPM est limité à 32 images : ses $T$ passes coûtent cher), les montre, et mesure le temps, la confiance moyenne du classifieur `CLF` (la probabilité de sa classe préférée) et la part d'images qu'il reconnaît avec une probabilité de plus de 90 % (environ 15 s en mode rapide). Une distance de Fréchet sur si peu d'images ne voudrait rien dire (fiche §205.8) : 🔬 B5.25 la mesure sur plus d'images. Les durées sont des mesures, pas des réponses.

Dans tes notes : combien de fois DDIM est-il plus rapide ici, et pourquoi ce rapport est-il proche du rapport des nombres de passes ? À qualité voisine, lequel choisirais-tu, et dans quel cas l'autre ? Partis du même bruit, les deux échantillonneurs donnent-ils les mêmes chiffres ? Pourquoi DDIM rend-il toujours les mêmes images pour un même bruit, et DDPM pas ? Que se passe-t-il, sur quelques images, si tu passes `clip_x0=None` à DDIM ?""",
           given=C.TOOLS_C2, todo=C.TODO_24, check=C.CHECK_24, solution=C.SOLUTION_24 + "\n\n\n" + C.CHECK_24,
           after=[("code", C.EXPERIMENT_24)])

# ---------------------------------------------------------------------------------------------------------------------
STEPS_25 = tuple(U.get("steps", (5, 10, 20, 50, 100)))
RATIO_25 = 1.2                       # a) "at most RATIO_25 times the distance at 100 steps"


def _answers_25() -> dict:
    """a) the smallest number of steps from which every linear seed is within RATIO_25 of its distance at 100 steps;
    b) at 10 steps, the schedule with the smaller distance for every seed (None when the seeds disagree)."""
    if not U:
        return {"enough": 20, "better5": "C"}
    enough = []
    for row in U["L"]:
        ok = [n for n, v in zip(STEPS_25, row) if v <= RATIO_25 * row[-1]]
        enough.append(min(n for n in ok if all(v <= RATIO_25 * row[-1] for v in row[STEPS_25.index(n):])))
    i5 = STEPS_25.index(5)
    lin, cos = [r[i5] for r in U["L"]], [r[i5] for r in U["C"]]
    per_seed = {"C" if c < l else "L" for l, c in zip(lin, cos)}
    if len(per_seed) != 1:
        raise ValueError(f"B5.25 b): the seeds disagree at 5 steps: {lin} {cos}")
    return {"enough": max(enough), "enough_per_seed": enough, "better5": per_seed.pop(), "lin5": lin, "cos5": cos}


A25 = _answers_25()


def _full_25() -> str:
    if not U:
        return "{}"
    return "{\n" + "\n".join(f'    "{k}": {[[round(v, 1) for v in row] for row in U[k]]},' for k in ("L", "C")) + "\n}"


EX_25 = Ex("B5.25", "🔬", 3, 45, "Nombre de pas DDIM, planning linéaire ou cosinus : distance de Fréchet contre temps de calcul",
           "mesurer ce que coûte et ce que rapporte chaque pas de DDIM, et comparer deux plannings à nombre de pas égal.",
           "Ex B5.24 · ✏️ B5.3 · ch. 27 (distance de Fréchet) · fiche §205.6 (« Pourquoi c'est plus rapide », piège « une "
           "sous-séquence qui ne part pas du dernier indice »), §205.8 (« La distance de Fréchet »)", thread="MNIST",
           tracks="C",
           body=r"""Combien de pas DDIM faut-il à ton modèle, et le planning cosinus, qui garde l'image lisible plus longtemps (✏️ B5.3), en demande-t-il moins ? La cellule suivante affiche les sous-séquences de 5 et 10 pas (`wbd.ddim_timesteps`, de l'indice $T - 1$ à 0 : jamais l'espacement « leading ») et donne `STEPS_25`, `N_25` et `FULL_25`, le **protocole complet** précalculé : la recette de B5.23 avec le planning linéaire du notebook (L) et avec le planning cosinus (C), sur plusieurs graines, et la distance de Fréchet de 256 échantillons pour chaque nombre de pas de `STEPS_25`.

**Contrat**
- `fd_curve_25(model, alpha_bars, steps_list, x_T)` : pour chaque nombre de pas $n$ de `steps_list`, le couple (`frechet_mnist` des images de ta `sample_ddim_24(model, x_T, alpha_bars, n)`, la durée de cet échantillonnage en secondes) ; une liste de couples de floats.

Puis, d'après le tableau `FULL_25`, **pour toutes ses graines** :
- a) `enough_25` : le plus petit nombre de pas de `STEPS_25` à partir duquel la distance du planning linéaire reste au plus """ + str(RATIO_25).replace(".", ",") + r""" fois sa distance à 100 pas ;
- b) `better5_25` : à 5 pas, quel planning a la plus petite distance, graine par graine, `"L"` ou `"C"` ?

La cellule de vérification compare `fd_curve_25` à DDIM du contrat sur 80 images et 2 et 4 pas, puis vérifie tes deux réponses. La suivante trace la courbe distance contre secondes de **ton** modèle (`N_25` images, environ 20 s en mode rapide), et, à côté, les bandes du protocole complet pour les deux plannings. 🚀 En mode complet, elle entraîne aussi un modèle au planning cosinus avec ta `train_23` (aussi long que B5.23) et trace sa courbe.

Dans tes notes : à partir de combien de pas la courbe de ton modèle se met-elle à plat, et combien de secondes coûte chaque point ? Pourquoi la distance remonte-t-elle parfois un peu quand on ajoute des pas ? Pourquoi les distances de ton modèle ne se comparent-elles pas directement à celles du protocole complet (nombre d'images) ? Le planning cosinus rend-il le petit nombre de pas moins coûteux en qualité ? Relie la réponse à la répartition des pas « utiles » de ✏️ B5.3 et 📈 B5.15.""",
           given=fill(C.GIVEN_25, {"N_SEEDS_25": str(len(U.get("L", [])) or 3), "FULL_25": _full_25()}),
           todo=fill(C.TODO_25, {"RATIO_TEXT_25": str(RATIO_25)}), check=C.CHECK_25,
           solution=C.SOLUTION_25 + "\n\n\n" + f'enough_25, better5_25 = {A25["enough"]}, "{A25["better5"]}"' + "\n\n\n"
           + C.CHECK_25,
           record=(f'wb.record("B5.25a", {A25["enough"]}, mistakes={{'
                   + ", ".join(f'"relis la ligne L du protocole complet : à {n} pas, toutes les graines restent-elles au plus à '
                               f'{str(RATIO_25).replace(".", ",")} fois leur distance à 100 pas, et ensuite aussi ?": {n}'
                               for n in STEPS_25 if n != A25["enough"])
                   + f'}}, fractional="on demande un nombre de pas de STEPS_25", choices={list(STEPS_25)})\n'
                   f'wb.record("B5.25b", "{A25["better5"]}", mistakes={{"relis la colonne des 5 pas du protocole complet, '
                   f'graine par graine": "{"L" if A25["better5"] == "C" else "C"}"}})'),
           after=[("code", C.EXPERIMENT_25)])

# ---------------------------------------------------------------------------------------------------------------------
G26 = HISTORY.get("guidance26", {})


def _answers_26() -> dict:
    """The three predictions of 🔮 B5.26 from the table of the protocol, each the same for every seed with a margin."""
    if not G26:
        return {"fd7": True, "acc0": False, "div0": True}
    by_w, real = G26["table"]["by_w"], G26["table"]["real"]
    fd0, fd7 = np.array(by_w["0"]["fd"]), np.array(by_w["7"]["fd"])
    acc0, div0 = np.array(by_w["0"]["acc"]), np.array(by_w["0"]["div"])
    answers = {}
    for name, ratios in (("fd7", fd7 / fd0), ("acc0", acc0 / 0.7), ("div0", div0 / real["div"])):
        smaller_is_true = name == "fd7"                 # a): "smaller"; b) and c): "more", "larger"
        if (ratios < 0.9).all():
            answers[name] = smaller_is_true
        elif (ratios > 1.1).all():
            answers[name] = not smaller_is_true
        else:
            raise ValueError(f"B5.26 {name}: the ratios {ratios} are too close to 1")
    return answers


import numpy as np  # noqa: E402

A26 = _answers_26()

EX_26 = Ex("B5.26", "🔮", 1, 10, "Prédire l'effet de w = 0, 1, 3 et 7 sur la netteté et la diversité",
           "prédire ce que la guidance change sur Fashion-MNIST, là où la fiche ne donne que le compromis général.",
           "✏️ B5.1 · fiche §205.7 (« La guidance sans classifieur », piège « deux conventions pour l'échelle de "
           "guidance »), §205.8 (« Fidélité et diversité »)", thread="Fashion-MNIST", tracks="C", hypothesis=True,
           body=r"""La fiche donne le compromis général mesuré par Ho et Salimans sur ImageNet. Et sur Fashion-MNIST, avec un petit U-Net conditionné par la classe ? Le **protocole complet** (précalculé, la recette de 📦 B5.27 en mode rapide sur plusieurs graines) génère 200 images (20 par classe, du même bruit pour chaque $w$, DDIM 20 pas, $\hat{\mathbf{x}}_0$ écrêté) pour $w = 0, 1, 3$ et 7, dans la convention de Ho et Salimans (`guidance_scale` $= 1 + w$), et les mesure de quatre façons :
- la **distance de Fréchet** aux 2 000 images de test (features du CNN du ch. 27 entraîné sur Fashion-MNIST) ;
- la part d'images dont le classifieur reconnaît la **classe demandée** ;
- la **diversité** : la moyenne, sur les classes, de l'écart-type moyen des features des images de la classe ;
- la part de **pixels saturés** ($|x| > 0{,}98$, du noir ou du blanc presque purs).

**Avant d'exécuter**, écris tes trois prédictions (`True` ou `False`, pour **toutes** les graines) :
- a) `fd7_26` : à $w = 7$, la distance de Fréchet est-elle **plus petite** qu'à $w = 0$ ?
- b) `acc0_26` : à $w = 0$ (le modèle conditionné, sans guidance), le classifieur reconnaît-il la classe demandée dans **plus de 70 %** des images ?
- c) `div0_26` : à $w = 0$, la diversité des images générées est-elle **plus grande** que celle des vraies images de test ?

La cellule d'expérience n'affiche le tableau du protocole qu'une fois tes trois prédictions écrites, et les vérifie à la fin. Rien n'y dépend de ton code ; tu mesures ton propre modèle en 📦 B5.27.""",
           given=C.GIVEN_26, todo=C.TODO_26,
           solution=f'fd7_26, acc0_26, div0_26 = {A26["fd7"]}, {A26["acc0"]}, {A26["div0"]}',
           record=(f'wb.record("B5.26a", {A26["fd7"]}, mistakes={{"compare la colonne des distances aux lignes w = 0 et w = 7, '
                   f'graine par graine": {not A26["fd7"]}}}, prediction=True)\n'
                   f'wb.record("B5.26b", {A26["acc0"]}, mistakes={{"lis la ligne w = 0 de la colonne de la classe reconnue": '
                   f'{not A26["acc0"]}}}, prediction=True)\n'
                   f'wb.record("B5.26c", {A26["div0"]}, mistakes={{"compare la ligne w = 0 de la colonne de la diversité à '
                   f'celle des vraies images, sous le tableau": {not A26["div0"]}}}, prediction=True)'),
           after=[("code", guarded(C.EXPERIMENT_26, ["fd7_26", "acc0_26", "div0_26"],
                                   "⏳ Ex B5.26 : écris d'abord tes trois prédictions dans la cellule précédente.")),
                  ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Quel $w$ donne la plus "
                         "petite distance, et pourquoi n'est-ce ni 0 ni le plus grand ? Que deviennent la classe reconnue et "
                         "la diversité quand $w$ augmente ? La part de pixels saturés des vraies images est affichée sous le "
                         "tableau : que dit l'écart avec les images générées ? Que mesure la distance de Fréchet, ici, que "
                         "ne mesure pas la classe reconnue, et inversement ?")])

EX_27 = Ex("B5.27", "📦", 3, 60, "Diffusion conditionnelle et guidance sans classifieur sur Fashion-MNIST",
           "conditionner ton U-Net par la classe, l'entraîner avec l'abandon du label, et générer avec la guidance.",
           "Ex B5.23, B5.24, B5.26 · fiche §205.7 (« Donner la classe au réseau », « La guidance sans classifieur »)",
           thread="Fashion-MNIST", tracks="C",
           body=r"""La cellule suivante donne `FASHION` (10 000 images de Fashion-MNIST en mode rapide, les 60 000 en mode complet), `CLF_F` (le CNN du ch. 27 entraîné une fois sur Fashion-MNIST, comme `CLF` sur MNIST), `REAL_F` et `REAL_F_FEATS` (2 000 images de test et leurs features), `SETTINGS_27` (la recette de B5.23 sur 2 epochs en mode rapide, avec un $\rho$ fixé pour son nombre de pas et `p_uncond = 0.1`), `CONFIG_27` (ton U-Net avec `n_classes = 10` : sa ligne 10 est la classe vide $\varnothing$, B5.21), `Y_27` (les classes demandées, autant par classe), `NAMES_F` et `guidance_measures(images, y)`, les quatre mesures de 🔮 B5.26.

**Contrat**
- `train_cond_27(model, ema, images, labels, settings, generator, alpha_bars)` : `train_23` avec les classes. Pour chaque batch, **après** les pas et le bruit, un troisième tirage `torch.rand(b, generator=generator) < settings["p_uncond"]` remplace le label par la classe vide `model.n_classes` ; le modèle reçoit `(x_t, t, y)`. La copie EMA par ta `ema_update_23`. Rend `{"loss": [...]}`.
- `guided_eps_27(model, x, t, y, w)` : au pas d'indice `t` (un entier), $(1 + w)\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}, c) - w\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}, \varnothing)$, les deux prédictions en **un seul** appel du réseau sur un batch de $2N$ images (les $N$ avec `y`, puis les $N$ avec la classe vide) ; pour `w == 0`, un seul appel sur les $N$ images avec `y`. Sans graphe.
- `sample_cond_27(model, x_T, y, w, alpha_bars, n_steps)` : ta `sample_ddim_24` ($\hat{\mathbf{x}}_0$ écrêté à $[-1, 1]$, le bruit recalculé), avec `guided_eps_27(model, x, t, y, w)` à la place du bruit du réseau.

La cellule de vérification compare tes fonctions au contrat sur un petit U-Net conditionné : 4 pas de `train_cond_27` (avec une probabilité d'abandon de 0,4, pour que l'abandon agisse), `guided_eps_27` à $w = 0$ et 2,5 avec le nombre d'appels du réseau, et 5 pas de `sample_cond_27`. La suivante entraîne ton U-Net conditionné (environ 1 min en mode rapide ; 🚀 en mode complet) et le sauvegarde, comme en B5.23. La dernière génère 200 images pour chaque $w$ de `W_26` (DDIM 20 pas), montre une image par classe et par $w$, et compare tes quatre mesures aux médianes du protocole de B5.26 (environ 25 s en mode rapide).

Dans tes notes : les images à $w = 0$ ressemblent-elles à leur classe ? Et à $w = 7$ ? Quel $w$ choisirais-tu pour un catalogue de vêtements, et pour un jeu de données d'entraînement ? Combien de passes du réseau coûte une image à 20 pas avec guidance ? Que se passerait-il avec `p_uncond = 0` (le réseau n'aurait jamais vu $\varnothing$) ? Avec `guidance_scale = 4` dans diffusers, quel $w$ utiliserais-tu (fiche, piège des deux conventions) ?""",
           given=C.GIVEN_27, todo=C.TODO_27, check=C.CHECK_27, solution=C.SOLUTION_27 + "\n\n\n" + C.CHECK_27,
           after=[("code", C.EXPERIMENT_27), ("code", C.EXPERIMENT_27B)])

# ---------------------------------------------------------------------------------------------------------------------
M28 = HISTORY.get("mem28", {})


def _category(shares) -> str | None:
    """"A" below 10 %, "B" from 10 to 50 %, "C" above 50 %, the same for every seed with a margin of 2 points."""
    cats = set()
    for s in shares:
        if s < 0.08:
            cats.add("A")
        elif 0.12 < s < 0.48:
            cats.add("B")
        elif s > 0.52:
            cats.add("C")
        else:
            return None
    return cats.pop() if len(cats) == 1 else None


def _answers_28() -> dict:
    """a) the category of the share of copies for 500 images; b) the median r of 500 images below that of 60 000 images,
    seed by seed (with a margin of 3 %)."""
    if not M28:
        return {"500": "A", "closer": True}
    cat = _category(M28["copies"]["500"])
    r500, r60k = M28["median_r"]["500"], M28["median_r"]["60000"]
    if all(a < 0.97 * b for a, b in zip(r500, r60k)):
        closer = True
    elif all(a > 1.03 * b for a, b in zip(r500, r60k)):
        closer = False
    else:
        raise ValueError(f"B5.28 b): the median ratios are too close: {r500} {r60k}")
    if cat is None:
        raise ValueError(f"B5.28 a): the shares of copies straddle a bound: {M28['copies']['500']}")
    return {"500": cat, "closer": closer}


A28 = _answers_28()


def _full_28() -> str:
    if not M28:
        return "{}"
    copies = {"500": M28["copies"]["500"], "10000": U.get("copies", [])[:2], "60000": M28["copies"]["60000"]}
    median_r = {"500": M28["median_r"]["500"], "10000": U.get("median_r", [])[:2], "60000": M28["median_r"]["60000"]}
    return ("{\n    \"copies\": " + json.dumps({k: [round(v, 4) for v in vs] for k, vs in copies.items()}) + ",\n"
            "    \"median_r\": " + json.dumps({k: [round(v, 4) for v in vs] for k, vs in median_r.items()}) + ",\n}")


_CAT_MSG = {"A": "moins de 10 %", "B": "de 10 à 50 %", "C": "plus de 50 %"}

EX_28 = Ex("B5.28", "🔬", 3, 40, "Mémorisation : plus proches voisins des échantillons, entraînement sur 500 images contre 60 000",
           "chercher les copies parmi les échantillons, et mesurer comment la taille du dataset d'entraînement les change.",
           "Ex B5.22, B5.23 · ch. 13 (plus proches voisins) · fiche §205.8 (« La mémorisation »)", thread="MNIST",
           tracks="C",
           body=r"""Une distance faible à son plus proche voisin ne suffit pas à déclarer une copie : un « 1 » bien droit est proche de beaucoup d'autres « 1 ». Comme Carlini et coll. (fiche §205.8), on compare la distance d'un échantillon à son plus proche voisin $a$ dans le dataset d'entraînement à la distance de $a$ à **ses propres** voisins :
$$r = \frac{d(\text{échantillon},\ a)}{d(a,\ b)},$$
où $b$ est l'image d'entraînement la plus proche de $a$ (autre que $a$). Un échantillon de $r < 1/3$ est anormalement proche de $a$ : une copie. La cellule suivante donne `FULL_28`, le **protocole complet** précalculé (la recette de B5.23 avec le **même nombre de pas d'optimisation**, `N_UPDATES_23`, sur 500, 10 000 et 60 000 images, deux graines) : pour chaque modèle, la part de copies parmi 256 échantillons et le $r$ médian ; et `IdealNet`, le débruiteur parfait de B5.22 vu comme un réseau.

**Contrat**
- `nearest_28(samples, train)` : pour chaque image de `samples` `(N, 1, 28, 28)`, son plus proche voisin dans `train` `(M, 1, 28, 28)` en pixels : le couple (distances, indices), deux tableaux NumPy `(N,)` ; la distance euclidienne des images aplaties ; à égalité, le premier indice.
- `copy_ratio_28(samples, train)` : le $r$ de chaque échantillon, un tableau float64 `(N,)`.

Puis, d'après `FULL_28` :
- a) `copies500_28` : pour les modèles entraînés sur 500 images, la part des échantillons de $r < 1/3$, pour **toutes** les graines : `"A"` moins de 10 %, `"B"` de 10 à 50 %, `"C"` plus de 50 % ;
- b) `closer500_28` : le $r$ médian des modèles entraînés sur 500 images est-il **plus petit** que celui des modèles entraînés sur 60 000 images, graine par graine ? `True` ou `False`.

La cellule de vérification essaie tes deux fonctions sur quatre points du plan (où tu peux refaire les calculs à la main) et sur 5 chiffres à peine bruités, puis vérifie tes réponses. La suivante génère 128 échantillons (DDIM, 50 pas) de ton U-Net de B5.23 et du débruiteur parfait des 500 premières images de `DATA_23`, montre pour chacun les 8 échantillons de plus petit $r$ et leur voisin, et trace les deux distributions de $r$ (environ 15 s en mode rapide). 🚀 En mode complet, elle entraîne aussi ton U-Net sur ces 500 images avec le même nombre de pas que B5.23, et le mesure.

Dans tes notes : que vaut $r$ pour le débruiteur parfait, et pourquoi ? Ton U-Net copie-t-il ? Que change la taille du dataset, à nombre de pas égal (combien de fois chaque image est-elle vue) ? Pourquoi la distance de Fréchet ne verrait-elle rien (🧠 Q10) ? Pourquoi comparer en pixels, et que changerait une comparaison dans l'espace des features de `CLF` ? Que t'apprennent Somepalli et coll., et Kadkhodaie et coll. (2024), qui étudient la même question (cherche leurs articles) ?""",
           given=fill(C.GIVEN_28, {"N_SEEDS_28": "deux", "FULL_28": _full_28()}),
           todo=C.TODO_28, check=C.CHECK_28,
           solution=C.SOLUTION_28 + "\n\n\n" + f'copies500_28, closer500_28 = "{A28["500"]}", {A28["closer"]}' + "\n\n\n"
           + C.CHECK_28,
           record=(f'wb.record("B5.28a", "{A28["500"]}", mistakes={{'
                   + ", ".join(f'"relis la ligne des 500 images de FULL_28 : la part de copies y est-elle {_CAT_MSG[c]} pour '
                               f'toutes les graines ?": "{c}"' for c in "ABC" if c != A28["500"])
                   + '})\n'
                   f'wb.record("B5.28b", {A28["closer"]}, mistakes={{"compare les r médians des lignes 500 et 60 000 de '
                   f'FULL_28, graine par graine": {not A28["closer"]}}})'),
           after=[("code", C.EXPERIMENT_28)])

EX_29 = Ex("B5.29", "📦", 3, 30, "Distance de Fréchet : situer le DDPM face au VAE (ch. 25) et au DCGAN (ch. 27) sur MNIST, avec le "
           "code d'évaluation du mini-projet MP6",
           "évaluer trois générateurs avec le même protocole et le même code, et lire ce que les mesures disent ou non.",
           "Ex B5.24 · ch. 25 (VAE), ch. 27 (DCGAN, distance de Fréchet) · mini-projet MP6 (`genbench`) · fiche §205.8",
           thread="MNIST", tracks="C",
           body=r"""Le mini-projet MP6 (`projets/partie_6_generatif/`) compare un VAE et un DCGAN avec son package `genbench` : des modèles, leurs entraînements et une évaluation (`genbench.evaluate` : `classifier_features`, `frechet_distance`, `class_report`, `nearest_neighbors`). La cellule suivante importe la **solution** de ce package (`projets/partie_6_generatif/solution/genbench` : elle ne dépend pas de ton avancement dans le MP6), entraîne une fois sur `DATA_23` son VAE (`ConvVAE`, 16 latents, $\beta = 0{,}5$, 3 epochs) et son DCGAN (3 epochs), puis les relit de `wbd.checkpoints_dir()` (environ 35 s la première fois). `genbench` travaille sur des images dans $[0, 1]$ : `CLF01` est `CLF` vu par `genbench` (features et logits d'images dans $[0, 1]$), `TRAIN01_29` les images de `DATA_23` dans $[0, 1]$. Le MP6 mesurait des images de Fashion-MNIST avec son propre classifieur (128 features) ; ici, sur MNIST, les features sont celles de `CLF` (64 valeurs), comme en B5.23 à B5.25.

**Contrat**
- `evaluate_29(images01, train01)` : l'évaluation du MP6 d'images générées dans $[0, 1]$, avec les fonctions de `genbench.evaluate` (`gb_eval`) et `CLF01` ; un dictionnaire :
  - `"fd"` : `frechet_distance` entre les features des 2 000 chiffres de test (`REAL_MNIST`, ramenés dans $[0, 1]$) et celles de `images01` ;
  - `"entropy"` et `"confidence"` : celles de `class_report(CLF01, images01)` ;
  - `"nn"` : la médiane des distances de `nearest_neighbors(images01, train01)`.

La cellule de vérification compare `evaluate_29` à un calcul direct sur 100 vrais chiffres. La suivante prend 256 images de chaque générateur (ton DDPM de B5.23 par DDIM à 50 pas, le VAE, le DCGAN) et 256 vrais chiffres d'entraînement tenus à l'écart, les mesure avec ta fonction (les mêmes features, le même nombre d'images, la même référence) et les montre.

Dans tes notes : classe les trois générateurs selon la distance de Fréchet ; ce classement suit-il ton jugement à l'œil ? Que disent l'entropie des classes (effondrement de mode, ch. 27) et la confiance du classifieur ? Pourquoi le VAE est-il si loin (pense au flou et aux features) ? Que mesure `nn`, et que vaut-il pour les vraies images ? À coût d'entraînement voisin, qu'est-ce qui rend la comparaison inégale (nombre de passes à la génération, taille des modèles, epochs) ?""",
           given=C.GIVEN_29, todo=C.TODO_29, check=C.CHECK_29, solution=C.SOLUTION_29 + "\n\n\n" + C.CHECK_29,
           after=[("code", C.EXPERIMENT_29)])

EX_30 = Ex("B5.30", "🛠️", 2, 25, "Un échantillonneur autonome : sample.py avec --steps, --seed et --guidance, images et paramètres "
           "sauvegardés",
           "livrer un script en ligne de commande qui génère des images d'un checkpoint, reproductible et documenté.",
           "Ex B5.24, B5.27 · ch. 23 (livrer un modèle) · fiche §205.6", thread="MNIST", tracks="C",
           body=r"""Un modèle ne sert qu'une fois sorti du notebook. Écris `sample.py`, un script qu'on lance dans un terminal : il relit un checkpoint de B5.23 ou de B5.27 (un fichier de `wbd.save_checkpoint`), génère des images par DDIM et sauvegarde les images **et** les paramètres qui permettent de les refaire. Tu écris son code dans la chaîne `SAMPLE_PY_30` ; la cellule de vérification l'enregistre dans `OUT_30 / "sample.py"` et le lance par `subprocess` (un vrai processus, sans les variables du notebook).

**Contrat** : `python sample.py CHECKPOINT [--steps 50] [--seed 0] [--guidance 0] [--label K] [--n 16] [--out samples]`
- le checkpoint se lit par `torch.load(..., weights_only=True)` : un dictionnaire avec `"config"` (de quoi reconstruire `wbd.TinyUNet(**config)`, mêmes noms que ton U-Net), `"ema_state_dict"` (les poids à utiliser) et `"key"`, la clé à provenance en JSON (`json.loads(payload["key"])["settings"]` donne `"T"` et `"schedule"`, le nom du planning pour `wbd.schedule_betas`) ;
- le bruit de départ `torch.randn((n, 1, 28, 28), generator=torch.Generator().manual_seed(seed))` ; les classes : toutes `--label` s'il est donné, sinon `torch.arange(n) % 10` pour un modèle conditionné, aucune pour un modèle sans classes ; les images par `wbd.sample_ddim(model, x_T, alpha_bars, steps, y=y, w=guidance)`, ramenées dans $[-1, 1]$ ;
- dans le dossier `--out` (créé s'il manque) : `samples.npy` (les images, float32 `(n, 1, 28, 28)`), `samples.png` (une grille) et `params.json`, avec au moins `checkpoint_sha256` (l'empreinte du fichier), `steps`, `subsequence` (les indices de `wbd.ddim_timesteps`), `seed`, `guidance` et `n` ;
- **refuse**, par un message sur la sortie d'erreur et un code de sortie non nul, sans rien générer : `--steps` hors de $[1, T]$, `--guidance` ou `--label` sur un modèle sans classes, `--label` hors des classes ; le code de sortie vaut 0 sinon.

La cellule de vérification lance ton script plusieurs fois (environ 20 s) : 4 images de ton modèle de B5.23 à 10 pas, comparées à `wbd.sample_ddim` ; `params.json` ; deux appels de même graine (les mêmes images au bit près) et une autre graine ; deux refus ; puis, si le modèle conditionné de B5.27 est sauvegardé, `--guidance 3 --label 7`.

Dans tes notes : pourquoi l'empreinte du checkpoint dans `params.json`, et pas seulement son chemin ? Qu'est-ce qui pourrait encore changer les images d'une machine à l'autre, à graine égale (versions, CPU ou GPU) ? Pourquoi un code de sortie non nul plutôt qu'un simple message ? Quelle option ajouterais-tu pour livrer ce script à un collègue ?""",
           todo=C.TODO_30, check=C.CHECK_30, solution=C.SOLUTION_30 + "\n\n\n" + C.CHECK_30)

# ---------------------------------------------------------------------------------------------------------------------
D31 = HISTORY.get("defi31", {})


def _defi_values() -> dict:
    """The reference of the challenge for each mode: the guidance w whose worst seed is the best, the (best, worst)
    distances with it, the goal (worst + 15 %, rounded up to the unit), the largest share of copies (+ 5 points)."""
    out = {}
    for mode in ("fast", "full"):
        rows = D31.get(mode)
        if not rows:
            out[mode] = {"w": 2.0, "ref": (40.0, 50.0), "goal": 60.0, "copies": 0.05}
            continue
        ws = list(rows[0]["fd"])
        w = min(ws, key=lambda k: max(r["fd"][k] for r in rows))
        fds = [r["fd"][w] for r in rows]
        copies = max(r["copies"][w] for r in rows)
        out[mode] = {"w": float(w), "ref": (round(min(fds), 1), round(max(fds), 1)),
                     "goal": float(math.ceil(1.15 * max(fds))), "copies": round(copies + 0.05, 2)}
    return out


import math  # noqa: E402

V31 = _defi_values()
_W_31 = V31["fast"]["w"]

EX_31 = Ex("B5.31", "🏆", 4, 120, "Défi : la plus petite distance de Fréchet sur Fashion-MNIST en au plus 50 pas d'échantillonnage",
           "concevoir de bout en bout un générateur de Fashion-MNIST qui s'entraîne dans un temps donné et génère en au "
           "plus 50 passes de réseau par image.",
           "Ex B5.25, B5.27, B5.28, B5.29 · tout le chapitre", thread="Fashion-MNIST", tracks="C",
           body=r"""**Le défi** : écris `method_31(images, labels, seconds)`, qui construit et entraîne **un modèle neuf**, en au plus `seconds` secondes, sur les seules images d'entraînement de Fashion-MNIST qu'elle reçoit (`images`, float32 `(N, 1, 28, 28)` dans $[-1, 1]$ ; `labels`, int64 `(N,)`), et rend une fonction `generate(n, generator)` qui rend `n` images, un tenseur float32 `(n, 1, 28, 28)` dans $[-1, 1]$. La mesure : la distance de Fréchet de `N_31` images générées aux 2 000 images de **test** de Fashion-MNIST, que ta méthode ne voit jamais (features de `CLF_F`). Objectif : au plus `GOAL_31`.

**Les règles**, contrôlées par `evaluate_31(method_31)` :
- **au plus 50 passes de réseau par image** : chaque appel d'un réseau (un `nn.Module`, quel qu'il soit) pendant `generate` compte autant de passes que d'images dans son batch ; la guidance compte donc double (25 pas DDIM guidés = 50 passes) ;
- l'entraînement tient dans `seconds` (`LIMIT_31` : une course réduite de 45 s en mode rapide, sur 10 000 images ; **le défi** en mode complet, 🚀, 10 min sur les 60 000 images) ; la génération n'est pas chronométrée, les 50 passes la bornent ;
- **à froid** et sans fuite : pendant la course, les variables du notebook qui contiennent des images, des features, un modèle ou une sauvegarde sont masquées, les chargeurs de `wb` (`wbd.image_subset`, `wb.datasets`), les sauvegardes, `torch.load` et le classifieur des mesures sont neutralisés ; `generate` doit appeler un réseau entraîné pendant la course, pas un modèle du notebook ;
- **pas de copie** : la part des images générées anormalement proches d'une image d'entraînement ($r < 1/3$, 🔬 B5.28) ne dépasse pas `COPIES_31` (celle de la référence plus une marge), et aucune n'est une image de test.

La cellule suivante donne `evaluate_31`, `LIMIT_31`, `N_31`, `GOAL_31`, `REFERENCE_31` (la meilleure et la pire graine de la solution de référence) et `COPIES_31`. L'objectif est la pire graine de la référence plus 15 % : un U-Net conditionné de canaux (16, 32, 64), entraîné pendant le temps donné, avec une EMA, et échantillonné par DDIM guidé. Tu peux reprendre tes fonctions du chapitre, ou faire autrement (un autre planning, une autre paramétrisation, un réseau plus petit ou plus gros, moins de pas mais une guidance différente…).

Dans tes notes : ta méthode, ce que tu as essayé, ce qui a aidé et ce qui n'a rien changé, et ta distance. Combien de pas d'optimisation ta méthode fait-elle dans le temps donné ? À 50 passes, vaut-il mieux 50 pas sans guidance ou 25 pas guidés ? Et la course réduite du mode rapide, prédit-elle le classement du défi complet ? La réussite est affichée ; elle n'est pas enregistrée : c'est une règle d'honneur.""",
           given=fill(C.GIVEN_31, {"LIMIT_FAST_31": "45.0", "LIMIT_FULL_31": "600.0",
                                   "GOAL_FAST_31": repr(V31["fast"]["goal"]), "GOAL_FULL_31": repr(V31["full"]["goal"]),
                                   "REF_FAST_31": repr(V31["fast"]["ref"]), "REF_FULL_31": repr(V31["full"]["ref"]),
                                   "COPIES_FAST_31": repr(V31["fast"]["copies"]),
                                   "COPIES_FULL_31": repr(V31["full"]["copies"])}),
           todo=C.TODO_31, check=C.CHECK_31,
           solution=fill(C.SOLUTION_31, {"W_31": repr(_W_31), "STEPS_31": "25"}) + "\n\n\n" + C.CHECK_31)

EXERCISES_C2 = [EX_22, EX_23, EX_24, EX_25, EX_26, EX_27, EX_28, EX_29, EX_30, EX_31]
