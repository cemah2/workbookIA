"""Parts A to D of the chapter 28 notebook: statements and cells (used by build_ch28.py)."""

from __future__ import annotations

import base64
import json
import math
import zlib
from pathlib import Path

from chapter_kit import Ex, Part, bug_letters_mistakes, guarded

import ch28_code as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_ch28.json"
HISTORY = (json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists()
           else {"p12": {}, "recon19": {"runs": []}, "curves20": {"runs": {}}, "calib29": {"runs": []}})


def _packed(data) -> str:
    """JSON data packed (zlib + base64) so that a code cell does not show the numbers before the prediction."""
    return base64.b64encode(zlib.compress(json.dumps(data, ensure_ascii=False).encode("utf-8"), 9)).decode("ascii")


def _median(values):
    v = sorted(values)
    return (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2 if len(v) % 2 == 0 else v[len(v) // 2]


def _cat(value, low, high):
    return "A" if value < low else ("B" if value <= high else "C")


# --- answers computed from the history (checked again by main() of build_ch28.py) ----------------------------------------
def answers() -> dict:
    p12 = HISTORY.get("p12") or {}
    early = _median([r["period"] for r in p12.get("conv1", [])]) if p12 else 2.0
    deep = _median([r["period"] for r in p12.get("fire8", [])]) if p12 else 13.0
    runs19 = HISTORY.get("recon19", {}).get("runs", [])
    ratios = [r["mse_fire7"] / r["mse_conv1"] for r in runs19] or [30.0]
    corrs = [abs(r["corr_style"]) for r in runs19] or [0.0]
    runs20 = HISTORY.get("curves20", {}).get("runs", {})
    noise = next((k for k, r in runs20.items() if r["content"][0] > 0), "A")
    heavy = next((k for k, r in runs20.items() if r["style"][-1] > 10 * r["content"][-1]), "C")
    first = round(math.log10(runs20["A"]["style"][0] / runs20["A"]["style"][-1])) if runs20 else 3
    calib = HISTORY.get("calib29", {}).get("runs", [])
    goal = round(1.15 * next((r["content"] for r in calib if r["style"] == "great_wave"), 100.0), 1)
    return {"early": _cat(early, 4, 8), "deep": _cat(deep, 4, 8), "ratio": _cat(min(ratios), 1.5, 5),
            "corr": all(c < 0.1 for c in corrs) and all(_cat(r, 1.5, 5) == _cat(min(ratios), 1.5, 5) for r in ratios),
            "noise": noise, "heavy": heavy, "first": first, "goal": goal, "early_value": early, "deep_value": deep}


A = answers()

GIVEN_12 = r'''# The images that excite 8 filters of conv1 and 8 filters of fire8 (the solution of 28.13, 128 pixels, 100 steps;
# tools/chapters/history_ch28.py), summed up by their dominant period in pixels (dominant_period, in the tools) and
# their share of high frequencies (high_freq_share)
FILTERS_12 = unpack("{P12}")   # packed, so that the numbers stay hidden until the experiment cell'''

TODO_12 = r'''early_12 = ...   # a) the MEDIAN dominant period of the 8 conv1 images: "A" below 4 pixels, "B" from 4 to 8, "C" above 8
deep_12 = ...    # b) the median dominant period of the 8 fire8 images: "A" below 4 pixels, "B" from 4 to 8, "C" above 8'''

SOLUTION_12 = f'''early_12, deep_12 = "{A['early']}", "{A['deep']}"'''

EXPERIMENT_12 = r'''for name_12, rows_12 in FILTERS_12.items():
    periods_12 = [r["period"] for r in rows_12]
    print(f"{name_12} : périodes dominantes {', '.join(fr(p, 1) for p in periods_12)} pixels ; médiane "
          f"{fr(float(np.median(periods_12)), 1)} ; hautes fréquences {pct(float(np.mean([r['high'] for r in rows_12])), 0)} en moyenne")
wb.check("28.12a", early_12)
wb.check("28.12b", deep_12)'''

RECORD_12 = (f'wb.record("28.12a", "{A["early"]}", mistakes={{"regarde la médiane des périodes de conv1 : les motifs '
             f'mesurent-ils plus de 4 pixels ?": "B", "regarde la médiane des périodes de conv1, en pixels": "C"}})\n'
             f'wb.record("28.12b", "{A["deep"]}", mistakes={{"regarde la médiane des périodes de fire8 : un filtre profond '
             f'cherche-t-il des motifs de quelques pixels ?": "A", "regarde la médiane des périodes de fire8, en pixels": "B"}})')

RECORD_19 = (f'wb.record("28.19a", "{A["ratio"]}", mistakes={{"compare les deux colonnes d\'erreur : le rapport est-il '
             f'proche de 1 ?": "A", "calcule le rapport des deux erreurs, graine par graine": "B"}})\n'
             f'wb.record("28.19b", {A["corr"]}, mistakes={{"regarde la colonne des corrélations, en valeur absolue": '
             f'{not A["corr"]}}})')

RECORD_20 = (f'wb.record("28.20a", "{A["noise"]}", mistakes={{"partir de la photo, c\'est un terme de contenu nul au départ : '
             f'cherche l\'essai où il ne l\'est pas": "{"B" if A["noise"] != "B" else "C"}", "regarde le terme de contenu au '
             f'pas 0 de chaque essai": "{"C" if A["noise"] != "C" else "B"}"}})\n'
             f'wb.record("28.20b", "{A["heavy"]}", mistakes={{"compare les deux courbes à la fin de chaque essai, sur '
             f'l\'échelle logarithmique : un facteur 10, c\'est une graduation": "{"A" if A["heavy"] != "A" else "B"}", '
             f'"regarde l\'écart entre style et contenu au dernier pas": "{"B" if A["heavy"] not in "AB" else "C"}"}})\n'
             f'wb.record("28.20c", {A["first"]}, mistakes={{"compte les graduations (puissances de dix) que descend la courbe '
             f'de style de l\'essai A": {A["first"] - 1}, "l\'axe est logarithmique : une graduation, c\'est un facteur 10": '
             f'{A["first"] + 1}}})')

RECORD_21 = ('wb.record("28.21", "BEKS", mistakes=' + bug_letters_mistakes("BEKS", "ACDFGHIJLMNOPQR", {
    "B": "une ligne fautive manque : que reçoit la première couche du réseau (diagnostic 2) ?",
    "E": "une ligne fautive manque : compare l'échelle des matrices de Gram à celle de la fiche (diagnostic 3)",
    "K": "une ligne fautive manque : qu'est-ce que l'optimiseur met à jour (diagnostic 1) ?",
    "S": "une ligne fautive manque : dans quelle plage doivent rester les pixels (diagnostic 4) ?"}) + ")")

RECORD_27 = C.RECORD_27

# ---------------------------------------------------------------------------------------------------------------------
PART_A = Part("A", "Visualiser ce que cherchent les filtres et les couches",
              "Un réseau pré-entraîné sur ImageNet, gelé : SqueezeNet 1.1 (`NET`, dans les outils), petit et rapide sur "
              "CPU. Tu écris d'abord un extracteur d'activations par *hooks*, que tout le chapitre réutilise, puis la "
              "montée de gradient sur l'image, pour un filtre, pour une couche entière, et avec les régularisations qui "
              "rendent les images lisibles.", exercises=[
    Ex("28.11", "📦", 2, 25, "Un extracteur multi-couches par hooks sur un réseau pré-entraîné gelé",
       "récupérer en une passe les activations de plusieurs couches d'un réseau pré-entraîné, sans le modifier.",
       "ch. 21 (cartes d'activation) · ch. 23 (modèles de torchvision) · fiche §28.2.1 (encadré 🧮 sur les hooks)",
       tracks="R, M, C",
       body=r"""`NET` est la partie convolutive de SqueezeNet 1.1 : un `nn.Sequential` de 13 modules, `NET[0]` à `NET[12]`. Le dictionnaire `LAYERS` nomme ceux dont on lira la sortie : `"conv1"` (la ReLU qui suit la première convolution, `NET[1]`), puis les modules `Fire` (`"fire2"` = `NET[3]`… `"fire9"` = `NET[12]`). Le réseau attend des images normalisées comme celles d'ImageNet : `normalize(x)` (dans les outils) le fait pour un batch `(N, 3, H, W)` dans $[0, 1]$.

**Contrat** de `extract_11(net, x, layers)` :
- `x` est un batch **déjà normalisé** ; `layers` un dictionnaire `{nom: indice}` ;
- une seule passe `net(x)`, pendant laquelle un *hook* par couche (`register_forward_hook`) range la sortie du module dans un dictionnaire `{nom: tenseur}`, rendu à la fin ;
- les tenseurs gardent leur graphe (aucun `detach`, aucun `torch.no_grad()`) : on dérivera par rapport à l'image ;
- les hooks sont retirés après la passe, **même si elle échoue** (`try`/`finally`) : aucun hook ne doit rester accroché à `NET`.

La cellule de vérification compare une activation à la sortie de `NET[:8](x)`, compte les hooks qui restent, vérifie le graphe, puis te fait reporter deux tailles :
a) le nombre de valeurs de l'activation `"fire4"` pour la photo `x_11` (128 pixels de large ; `numel()`) ;
b) le côté des cartes de `"fire9"` pour une image de 192 × 192.

Dans tes notes : pourquoi accrocher le hook à la ReLU (ou ici au module `Fire`, qui finit par une ReLU) plutôt qu'à la convolution, surtout si la ReLU est `inplace` ? Que se passe-t-il si l'on relance une cellule qui accroche des hooks sans les retirer ?""",
       todo=C.TODO_11, check=C.CHECK_11, solution=C.SOLUTION_11 + "\n\n\n" + C.CHECK_11, record=C.RECORD_11),

    Ex("28.12", "🔮", 2, 15, "Prédire l'image qui excite un filtre précoce, puis un filtre profond",
       "prédire, en pixels, la taille des motifs que cherchent les filtres d'une couche précoce et d'une couche profonde.",
       "Ex 28.11 · fiche §28.2.2, §28.2.3", tracks="R, C",
       body=r"""La fiche dit que les couches précoces cherchent des traits et des textures fines, les couches profondes des motifs plus grands. De combien ? La cellule suivante charge, pour 8 filtres de `conv1` et 8 filtres de `fire8`, l'image qui les excite le plus (la montée de gradient de 28.13, à partir d'un bruit, 128 × 128 pixels, 100 pas), résumée par sa **période dominante** : la longueur, en pixels, du motif qui se répète le plus dans l'image (le pic de son spectre de Fourier, fonction `dominant_period` des outils).

**Avant d'exécuter**, écris tes deux prédictions :
a) la médiane des 8 périodes de `conv1` : `"A"` moins de 4 pixels, `"B"` de 4 à 8, `"C"` plus de 8 ;
b) la médiane des 8 périodes de `fire8`, avec les mêmes catégories.

Rappel utile : la première convolution de SqueezeNet a un noyau de 3 × 3 et un pas de 2 ; `fire8` vient après trois max-poolings.

La cellule d'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites.""",
       given=GIVEN_12.replace("{P12}", _packed(HISTORY.get("p12", {}))), todo=TODO_12, solution=SOLUTION_12,
       hypothesis=True, record=RECORD_12,
       after=[("code", guarded(EXPERIMENT_12, ["early_12", "deep_12"],
                               "⏳ Ex 28.12 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : tes prédictions étaient-elles justes ? Relie la période des images de `fire8` au "
                     "champ récepteur de cette couche (🔁 R3). Pourquoi certaines images de `fire8` ont-elles quand même "
                     "une période de 2 à 4 pixels (pense à ce que la fiche dit du bruit de haute fréquence) ?")]),

    Ex("28.13", "📦", 2, 30, "Visualiser un filtre d'un réseau ImageNet : la montée de gradient du ch. 21 passe à l'échelle",
       "écrire la montée de gradient sur l'image qui fait apparaître ce que cherche un filtre d'un réseau pré-entraîné.",
       "Ex 28.11 · ch. 21 (montée de gradient sur un petit CNN) · fiche §28.2.2 (encadré 🧮)", tracks="R, C",
       body=r"""**Contrat**
- `activation_13(net, x, layer_index, channel)` : pour un batch `x` dans $[0, 1]$ (normalisé **dans** la fonction), la moyenne des carrés de la carte du filtre `channel` à la sortie de `net[layer_index]`, $a(\mathbf{x}) = \frac{1}{HW}\sum A_{c,h,w}^2$ ; un tenseur à 0 dimension, avec son graphe (`net[:layer_index + 1]` suffit : inutile de calculer les couches suivantes) ;
- `visualize_filter_13(net, layer_index, channel, size=128, steps=100, lr=0.05, seed=0)` : part du bruit gris `torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(seed)) * 0.2 + 0.4`, fait `steps` pas d'`Adam([x], lr)` sur $-a(\mathbf{x})$, ramène les pixels dans $[0, 1]$ après chaque pas, et rend l'image `(3, size, size)` en `float32` (`to_image`). Les poids du réseau ne changent jamais.

La cellule de vérification contrôle l'activation sur un bruit, puis une visualisation courte : forme, plage des pixels, poids intacts, activation multipliée par plus de 5. La cellule suivante visualise six filtres, du premier au dernier module (quelques secondes).

Dans tes notes : décris ce que tu vois de `conv1` à `fire9`. Pourquoi les images de `conv1` sont-elles presque uniformes, avec une grille fine ? Que se passerait-il sans le `clamp_` ?""",
       todo=C.TODO_13, check=C.CHECK_13, solution=C.SOLUTION_13 + "\n\n\n" + C.CHECK_13, after=[("code", C.EXPERIMENT_13)]),

    Ex("28.14", "🎨", 2, 25, "Une galerie couche par couche (fig. 28.7)",
       "refaire la figure 28.7 du livre : une image par couche, qui l'excite en entier.",
       "Ex 28.13 · fiche §28.2.3", tracks="C",
       body=r"""Le livre montre une image par couche de VGG16 (fig. 28.7). Fais-en autant pour les neuf couches de `LAYERS`.

**Contrat** de `visualize_layer_14(net, layer_index, size=128, steps=100, lr=0.05, seed=0)` : comme `visualize_filter_13`, mais la quantité qui monte est la moyenne des carrés des activations de **tous** les filtres de `net[layer_index]`.

La cellule de vérification produit la galerie (une dizaine de secondes), vérifie que chaque couche s'active plus fort qu'au départ, et affiche sous chaque image le gain obtenu, puis la part de hautes fréquences de chaque image (`high_freq_share`).

Dans tes notes : compare la galerie à la fig. 28.7. Les images de toutes les couches sont-elles dominées par les hautes fréquences ? Qu'est-ce que cela annonce pour 28.15 ?""",
       todo=C.TODO_14, check=C.CHECK_14, solution=C.SOLUTION_14 + "\n\n\n" + C.CHECK_14),

    Ex("28.15", "🔬", 3, 40, "Régulariser la visualisation : décalages, flou, octaves",
       "mesurer ce que chaque régularisation change à une visualisation : la part de hautes fréquences, et l'activation atteinte.",
       "Ex 28.13 · fiche §28.2.2 (encadré 🕰️) · C. Olah et coll., « Feature Visualization » (Distill, 2017)", tracks="C",
       body=r"""**Contrat** de `visualize_reg_15(net, layer_index, channel, ..., jitter=0, blur_every=0, octaves=1, octave_scale=1.4)` : la docstring de la cellule suivante. Sans régularisation (`jitter=0, blur_every=0, octaves=1`), elle doit rendre **la même image** que `visualize_filter_13` ; les trois régularisations s'ajoutent indépendamment :
- les **décalages** (*jitter*) : avant chaque pas, l'image est roulée de $(d_y, d_x)$ tirés uniformément dans $[-j, j]$ (`torch.roll`, générateur de graine `seed + 1`), le pas est fait sur l'image roulée ;
- le **flou** : tous les `blur_every` pas, un flou 3 × 3 (`F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)`), sans graphe ;
- les **octaves** : on optimise d'abord une petite image, de côté `size / octave_scale**(octaves - 1)`, puis on l'agrandit d'un facteur `octave_scale` (interpolation bilinéaire) jusqu'à `size`, `steps` pas à chaque échelle.

La cellule d'expérience compare cinq réglages sur un filtre de `fire7` : l'activation atteinte et la part de hautes fréquences.

Dans tes notes : quelle régularisation réduit le plus les hautes fréquences ? Laquelle coûte le plus d'activation ? Une image régularisée excite moins le filtre : pourquoi est-elle pourtant **plus** informative ? Pourquoi les décalages empêchent-ils le réseau de « tricher » avec des motifs de quelques pixels collés à la grille de ses pas ?""",
       todo=C.TODO_15, check=C.CHECK_15, solution=C.SOLUTION_15 + "\n\n\n" + C.CHECK_15, after=[("code", C.EXPERIMENT_15)]),
])

PART_B = Part("B", "Le deep dream", "Le rêve part d'une **photo** libre (fournie avec scikit-learn, sous licence CC BY "
              "2.0 : on la crédite) et fait monter les activations de plusieurs couches à la fois, à plusieurs échelles.",
              given=C.GIVEN_B, exercises=[
    Ex("28.16", "📦", 3, 40, "Deep dream sur une photo libre",
       "écrire un deep dream à plusieurs couches et plusieurs octaves, avec un masque optionnel.",
       "Ex 28.11, 28.13 · fiche §28.3", tracks="C",
       body=r"""**Contrat**
- `dream_loss_16(net, x, layer_weights)` : $L_{\text{dream}} = \sum_\ell w_\ell \cdot \operatorname{moyenne}(A_\ell^2)$ pour un batch `x` dans $[0, 1]$ (normalisé dans la fonction), les activations lues par **ton** `extract_11` en une passe ; `layer_weights` = `{nom de couche: poids}` ;
- `deep_dream_16(net, photo, layer_weights, steps=40, lr=0.01, octaves=3, octave_scale=1.4, mask=None)` : la docstring de la cellule suivante. Retiens trois points : on **monte** (`x += lr * g`) avec un gradient **normalisé** par sa valeur absolue moyenne (`g / (g.abs().mean() + 1e-8)`) ; à chaque échelle, on réinjecte les détails que la réduction a fait perdre à la photo ; le gradient est multiplié par le masque (redimensionné, plus proche voisin) quand il y en a un.

La cellule de vérification contrôle la loss de rêve sur une petite fleur, un rêve court (la loss monte, les poids ne bougent pas), puis le masque : avec une seule octave, la moitié masquée de la photo ne doit pas changer du tout. La cellule suivante fait rêver la photo avec des couches précoces, intermédiaires et profondes. 🚀 En mode complet, les images sont plus grandes et les pas plus nombreux ; tu peux aussi essayer VGG16 (`wb.datasets.load_pretrained("vgg16")`, 528 Mio, et ses couches `VGG_LAYERS`), le réseau du livre.

Dans tes notes : décris les trois rêves. Avec quelles couches vois-tu des formes d'animaux ? Pourquoi normaliser le gradient (pense à l'échelle très différente des activations d'une couche à l'autre) ?""",
       todo=C.TODO_16, check=C.CHECK_16, solution=C.SOLUTION_16 + "\n\n\n" + C.CHECK_16, after=[("code", C.EXPERIMENT_16)]),

    Ex("28.17", "🔬", 3, 30, "Choisir couches et poids, ne rêver que dans une zone",
       "composer un rêve : choisir les couches et leurs poids, et limiter le rêve à une zone de la photo par un masque.",
       "Ex 28.16 · fiche §28.3", tracks="C",
       body=r"""Choisis tes couches et leurs poids (`layers_17`, au moins deux couches) et un masque (`mask_17`, un tenseur de 0 et de 1 à la taille de `PHOTO_B` : 1 là où le rêve a le droit de changer la photo). Par exemple : le ciel seulement, ou tout sauf le bâtiment. La cellule de vérification fait rêver la photo dans ta zone, vérifie que rien n'a bougé ailleurs et montre la carte des changements.

Dans tes notes : essaie au moins deux jeux de poids. Qu'est-ce qui change quand on donne plus de poids à une couche profonde ? Pourquoi un masque appliqué au **gradient** laisse-t-il quand même voir, au bord de la zone, des motifs qui débordent ? (Pense au champ récepteur.)""",
       todo=C.TODO_17, check=C.CHECK_17, solution=C.SOLUTION_17 + "\n\n" + C.CHECK_17),
])

PART_C = Part("C", "Le transfert de style",
              "Le contenu : une photo libre ; le style : *La Nuit étoilée* (domaine public). D'abord la matrice de Gram, "
              "puis trois exercices pour **lire** et **déboguer** avant d'écrire : une prédiction, des courbes, un script "
              "fautif. Ensuite les reconstructions couche par couche, ton transfert de style complet, ses réglages, une "
              "alternative à la matrice de Gram, une galerie documentée et le défi. Conventions de ce notebook (✏️ 28.4) : "
              "la loss de contenu est une **moyenne** (`F.mse_loss`), la loss de style une **somme** sur les coefficients "
              "des matrices de Gram **normalisées** ; les poids `ALPHA`, `BETA`, `TV` sont réglés pour ces conventions.",
              given=C.GIVEN_C, exercises=[
    Ex("28.18", "📦", 1, 15, "La matrice de Gram en PyTorch, vérifiée contre NumPy",
       "écrire la matrice de Gram normalisée et la variation totale en PyTorch, et les vérifier contre NumPy.",
       "✏️ 28.2, 28.4 · fiche §28.4.1, §28.4.5 (encadrés 🧮)", tracks="R, M, C",
       body=r"""**Contrat**
- `gram_matrix_18(A)` : pour un tenseur `(C, H, W)`, la matrice `(C, C)` $G = \frac{1}{CHW} F F^\top$ ($F$ : les cartes mises à plat en lignes) ; pour un batch `(N, C, H, W)`, une matrice par image, `(N, C, C)` ; sans NumPy, pour que le gradient passe ;
- `tv_loss_18(x)` : pour un batch d'images `(N, 3, H, W)`, la somme des carrés des écarts entre pixels voisins, verticalement et horizontalement, sur tous les canaux et toutes les images.

Exemple (✏️ 28.2) : pour $F = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$ (deux filtres, deux positions, rangées dans un tenseur `(2, 1, 2)`), $G = \begin{pmatrix} 2{,}5 & 0{,}5 \\ 0{,}5 & 1 \end{pmatrix}$.

La cellule de vérification compare tes deux fonctions à `np.einsum` et à `np.diff` sur des tableaux au hasard, et vérifie que le gradient traverse `gram_matrix_18`.""",
       todo=C.TODO_18, check=C.CHECK_18, solution=C.SOLUTION_18 + "\n\n\n" + C.CHECK_18),

    Ex("28.19", "🔮", 1, 10, "Prédire une reconstruction depuis le bruit : contenu profond seul, style seul",
       "prédire, en chiffres, ce qu'une reconstruction depuis le bruit garde de la photo, selon la couche et la loss.",
       "Ex 28.18 · fiche §28.4.3, §28.4.4", tracks="C",
       body=r"""On part d'un bruit et l'on optimise les pixels pour égaler, au choix, les activations d'une **seule** couche de la photo (loss de contenu), ou les matrices de Gram du tableau (loss de style), comme le livre aux fig. 28.19 et 28.23. La cellule suivante charge les résultats (la solution de 28.22, trois graines) : l'erreur quadratique moyenne, pixel par pixel, entre la photo et ses reconstructions depuis `conv1` et depuis `fire7`, et la corrélation, pixel par pixel, entre la reconstruction du **style** et la photo.

**Avant d'exécuter**, écris tes deux prédictions :
a) l'erreur de la reconstruction depuis `fire7`, divisée par celle depuis `conv1` : `"A"` moins de 1,5, `"B"` de 1,5 à 5, `"C"` plus de 5 ;
b) `True` ou `False` : la corrélation entre la reconstruction du style et la photo reste sous 0,1 en valeur absolue, pour les trois graines.

La cellule d'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites.""",
       given=C.GIVEN_19.replace("{RECON19}", _packed(HISTORY.get("recon19", {"runs": []}))), todo=C.TODO_19,
       solution=C.SOLUTION_19.replace('"{RATIO}"', f'"{A["ratio"]}"').replace("{CORR}", str(A["corr"])),
       hypothesis=True, record=RECORD_19,
       after=[("code", guarded(C.EXPERIMENT_19, ["ratio_19", "corr_19"],
                               "⏳ Ex 28.19 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : tes prédictions étaient-elles justes ? Pourquoi une reconstruction depuis `fire7` "
                     "peut-elle avoir une grosse erreur pixel et garder pourtant la scène (tu le verras en 28.22) ? Pourquoi "
                     "la corrélation du style avec la photo ne pouvait-elle pas être forte (∂ 28.3) ?")]),

    Ex("28.20", "📈", 2, 15, "Lire les courbes de losses d'un transfert de style",
       "lire les trois termes pondérés d'un transfert de style, et reconnaître un départ et un réglage à leurs courbes.",
       "Ex 28.18 · fiche §28.4.5 · ✏️ 28.4", tracks="R, C",
       body=r"""La cellule suivante trace trois transferts de style réels (la solution de 28.23, 150 pas), sous des lettres mélangées : les trois termes **pondérés**, $\alpha L_c$, $\beta L_s$ et $\lambda L_{TV}$, tous les 5 pas, sur une échelle logarithmique. L'un a les réglages de la partie C ; les deux autres changent **une** chose chacun.

Lis les courbes et écris :
a) la lettre de l'essai qui part d'un **bruit** (et non de la photo) ;
b) la lettre de l'essai dont le terme de style pondéré reste, à la fin, plus de 10 fois au-dessus du terme de contenu pondéré ;
c) pour l'essai A : de combien de puissances de dix le terme de style pondéré descend-il entre le pas 0 et la fin ? (un entier, lu sur les graduations)

Dans tes notes : qu'a-t-on changé dans chacun des deux autres essais ? Pourquoi la variation totale **monte**-t-elle dans deux essais ? Pourquoi un terme de contenu nul au départ n'est-il pas un bon signe en soi ?""",
       given=C.GIVEN_20.replace("{CURVES20}", _packed(HISTORY.get("curves20", {"runs": {}}))), todo=C.TODO_20,
       solution=C.SOLUTION_20.replace('"{NOISE}"', f'"{A["noise"]}"').replace('"{HEAVY}"', f'"{A["heavy"]}"').replace("{FIRST}", str(A["first"])),
       check=C.CHECK_20, record=RECORD_20),

    Ex("28.21", "🐛", 2, 25, "Le transfert de style qui ne produisait que du gris",
       "trouver et corriger les quatre erreurs d'un script de transfert de style, guidé par des diagnostics enchaînés.",
       "Ex 28.18, 28.11 · fiche §28.4.5, pièges ⚠️", tracks="C",
       body=r"""Le script `style_transfer_buggy_21` part d'une image grise (c'est voulu) et devrait la transformer en une photo peinte. Il ne rend que du gris. La cellule suivante le définit et lance `diagnose_21`, quatre diagnostics enchaînés, chacun ne s'exécutant que si le précédent passe (une erreur peut en cacher une autre) : les pixels bougent-ils ? que reçoit la première couche ? quel est le rapport entre le terme de style et le terme de contenu au premier pas ? dans quelle plage finissent les pixels ?

1. Écris dans `faulty_21` les lettres des lignes fautives (dans n'importe quel ordre). Il y en a **quatre**.
2. Recopie la fonction dans `style_transfer_fixed_21` et corrige-la : même signature, mêmes sorties, même départ gris. La cellule de vérification relance les diagnostics sur ta version.

Dans tes notes : pour chaque erreur, le symptôme qu'elle produit seule. Laquelle expliquait le gris ? Pourquoi `clamp_(-3, 3)` est-il une confusion entre deux espaces de pixels ?""",
       given=C.GIVEN_21, todo=C.TODO_21, solution=C.SOLUTION_21 + "\n\n\n" + C.CHECK_21, check=C.CHECK_21, record=RECORD_21),

    Ex("28.22", "🎨", 3, 45, "Reconstructions de contenu et de style couche par couche (fig. 28.19, 28.23, 28.24)",
       "refaire les trois figures du livre qui montrent ce que gardent les activations et les matrices de Gram de chaque couche.",
       "Ex 28.18, 28.11 · fiche §28.4.3, §28.4.4", tracks="C",
       body=r"""**Contrat** de `reconstruct_22(net, target, layer_names, kind="content", steps=150, lr=0.05, seed=0)` : la docstring de la cellule suivante. Départ : le bruit `torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(seed))` ; `kind="content"` égale les activations de `target` (somme, sur les couches, des `F.mse_loss`), `kind="style"` ses matrices de Gram (somme des carrés des écarts) ; Adam sur les pixels, ramenés dans $[0, 1]$.

La cellule de vérification contrôle deux reconstructions courtes sur une petite fleur. La suivante produit trois rangées : le contenu de la photo couche par couche (fig. 28.19), le style du tableau couche par couche (fig. 28.23), puis cumulé (fig. 28.24). 🚀 En mode complet, plus de pas ; avec VGG16, tu retrouves les blocs du livre.

Dans tes notes : à partir de quelle couche le contenu ne garde-t-il plus que la disposition ? Quelle couche donne le style le plus proche du tableau ? Le livre dit que les blocs 4 et 5 de VGG16 n'ajoutent presque rien au style : est-ce aussi ce que tu vois pour les dernières couches de SqueezeNet ?""",
       todo=C.TODO_22, check=C.CHECK_22, solution=C.SOLUTION_22 + "\n\n\n" + C.CHECK_22, after=[("code", C.EXPERIMENT_22)]),

    Ex("28.23", "📦", 3, 45, "Transfert de style complet sous budget CPU",
       "écrire le transfert de style de Gatys en PyTorch, avec ses trois termes, et l'appliquer à trois tableaux du domaine public.",
       "Ex 28.18, 28.11 · ✏️ 28.4 · fiche §28.4.2 à §28.4.5", tracks="R, M, C",
       body=r"""**Contrat** de `style_transfer_23(net, content, style, content_layers, style_layers, alpha, beta, tv, steps=150, lr=0.05, init="content", seed=0, every=1)` (la docstring de la cellule suivante le détaille) :
- les cibles sont calculées **une fois**, sans graphe : les activations du contenu aux `content_layers`, les matrices de Gram (`gram_matrix_18`) du style aux `style_layers` ; contenu et style peuvent avoir des tailles différentes ;
- départ : une copie du contenu (`init="content"`), ou le bruit `torch.rand(...)` de graine `seed` (`init="noise"`) ;
- à chaque pas : **une** passe de l'image par `extract_11` pour toutes les couches utiles ; $L_c$ = somme des `F.mse_loss` (des moyennes) ; $L_s$ = somme, sur les couches, des sommes des carrés des écarts de matrices de Gram ; $L_{TV}$ = `tv_loss_18` ; loss $= \alpha L_c + \beta L_s + \lambda L_{TV}$, un pas d'`Adam([x], lr)`, pixels ramenés dans $[0, 1]$ ;
- rend `(image, history)` : `history` = `{"content": [...], "style": [...], "tv": [...]}`, les trois termes **pondérés**, en `float`, enregistrés avant chaque pas dont l'indice est un multiple de `every`, et une fois de plus après le dernier pas.

La cellule de vérification fait un transfert de 12 pas sur de petites images : forme, plage, historique (13 valeurs par terme), loss qui baisse, poids intacts, couches de style additionnées, départ `"noise"` exact. La suivante applique trois styles à la photo (une quinzaine de secondes). 🚀 En mode complet, des images de 256 pixels et 400 pas ; VGG16 et les couches du livre sont à ta portée (`VGG_LAYERS`).

Dans tes notes : compare les trois résultats. Qu'est-ce qui passe du tableau à la photo : la palette, les touches, les contours ? Quelle part du temps d'un pas prend la passe dans le réseau (🧮 28.6) ?""",
       todo=C.TODO_23, check=C.CHECK_23, solution=C.SOLUTION_23 + "\n\n\n" + C.CHECK_23, after=[("code", C.EXPERIMENT_23)]),

    Ex("28.24", "🔬", 3, 35, "Poids contenu/style, variation totale et image de départ",
       "explorer le compromis entre contenu et style, et l'effet de la variation totale et du point de départ, avec des mesures fixes.",
       "Ex 28.23 · fiche §28.4.5, §28.4.6", tracks="C",
       body=r"""Pour comparer des réglages, il faut des **mesures qui ne changent pas avec eux** : `measure_24(img)` (cellule suivante) rend la distance de contenu (à la couche `fire6`) et la distance de style (matrices de Gram de `STYLE_LAYERS`) entre ton image et la photo, le tableau, sans pondération.

Écris dans `settings_24` au moins quatre réglages (des dictionnaires d'arguments de `style_transfer_23`, par exemple `{"beta": 100.0}` ou `{"init": "noise"}`) ; les autres arguments prennent les valeurs de la partie C. La cellule de vérification les fait tourner (≈ 4 s chacun en mode rapide) et affiche, pour chacun, les deux distances et la part de hautes fréquences.

Dans tes notes : trace (même à la main) la distance de style en fonction de la distance de contenu pour tes réglages de $\beta$ : une frontière se dessine-t-elle ? Que change $\lambda = 0$ ? Partir du bruit : meilleure ou pire distance de style, à nombre de pas égal ? Pourquoi « $\beta / \alpha = 40$ » du livre ne se transpose-t-il pas ici (Q9) ?""",
       given=C.GIVEN_24, todo=C.TODO_24, solution=C.SOLUTION_24 + "\n\n\n" + C.CHECK_24, check=C.CHECK_24),

    Ex("28.25", "🔬", 3, 40, "Au-delà de Gram : moyennes et écarts-types des canaux",
       "remplacer la matrice de Gram par les moyennes et les écarts-types des canaux (l'idée d'AdaIN) et comparer les résultats.",
       "Ex 28.23 · ∂ 28.3 (5 et 6) · fiche §28.4.6 (encadrés)", tracks="M, C",
       body=r"""∂ 28.3 a montré que la matrice de Gram mêle moyenne et écart-type de chaque canal ($C\,G_{ii} = \mu_i^2 + \sigma_i^2$) et y ajoute les co-activations. Huang et Belongie (AdaIN, 2017) n'utilisent **que** les moyennes et les écarts-types des canaux.

**Contrat**
- `channel_stats_25(A)` : pour des activations `(N, C, H, W)`, deux tenseurs `(N, C)` : la moyenne de chaque canal sur les $H \times W$ positions, et $\sqrt{\text{variance} + 10^{-5}}$, la variance de population (`unbiased=False`) ;
- `stats_loss_25(A, target_mean, target_std)` : la somme des carrés des écarts entre les moyennes de `A` et `target_mean`, plus celle des écarts entre les écarts-types et `target_std`.

La cellule de vérification compare tes deux fonctions à NumPy. La suivante fait deux transferts avec la même boucle (`transfer_25`, fournie : elle utilise ton `extract_11` et ton `tv_loss_18`), l'un avec la loss de Gram, l'autre avec ta loss de statistiques, et les mesure par `measure_24`.

Dans tes notes : les deux images se ressemblent-elles ? La loss de statistiques fait-elle baisser la distance de style **mesurée par Gram** ? Qu'est-ce que la matrice de Gram contient de plus (pense aux coefficients hors de la diagonale) ? Pourquoi AdaIN peut-il alors appliquer un style en une seule passe ?""",
       todo=C.TODO_25, check=C.CHECK_25, solution=C.SOLUTION_25 + "\n\n\n" + C.CHECK_25,
       after=[("code", C.GIVEN_25), ("code", C.EXPERIMENT_25)]),

    Ex("28.26", "🛠️", 3, 30, "Crédits, licences et paramètres : documenter une galerie générée",
       "publier une galerie d'images générées qu'un autre peut refaire et réutiliser légalement : crédits, licences, réglages, versions.",
       "Ex 28.23 · ⚖️ 28.9 · fiche (encadré ⚠️ du §28.4.5 et 🕰️ sur le droit d'auteur)", tracks="C",
       body=r"""Une image générée hérite de questions : d'où viennent la photo et le tableau ? sous quelle licence ? comment la refaire ? **Contrat** de `write_gallery_26(folder, entries)` : la docstring de la cellule suivante. Pour chaque entrée, un PNG (`plt.imsave`) et, dans un `README.md` : le nom du fichier (une image Markdown), les lignes de crédit de la photo et du tableau (`wb.datasets.image_credit`), chaque réglage de `params` sous la forme `clé: valeur` ; puis, une fois, le réseau et ses poids (« SqueezeNet 1.1, IMAGENET1K_V1 »), les versions de torch et de torchvision, la date (AAAA-MM-JJ), et une phrase qui dit que les images sont **générées**.

La cellule de vérification fait deux petits transferts, appelle ta fonction et passe le dossier à `validate_gallery_26`, qui liste ce qui manque (et refuse toute œuvre encore protégée).

Dans tes notes : quelle licence choisirais-tu pour tes images, sachant que la photo est sous CC BY 2.0 ? Que faudrait-il ajouter pour publier la galerie dans l'Union européenne (transparence, ⚖️ 28.9) ?""",
       given=C.GIVEN_26, todo=C.TODO_26, solution=C.SOLUTION_26 + "\n\n\n" + C.CHECK_26, check=C.CHECK_26),
])

PART_D = Part("D", "Le texte mot à mot sur Verne, et le défi",
              "Le livre termine par un LSTM qui écrit la suite du livre, mot à mot, à partir du mot le plus probable. Tu "
              "reprends ton code de 24.30 (Holmes) sur *Le Tour du monde en quatre-vingts jours*, avec le modèle du livre "
              "(deux couches de LSTM de 128 cellules), puis tu compares les stratégies de décodage. Le défi, enfin, "
              "revient au transfert de style.", given=C.GIVEN_D, exercises=[
    Ex("28.27", "📦", 3, 45, "Plus de Verne : réutiliser le code du LSTM mot à mot de 24.30 sur Verne et l'entraîner plus longtemps, du charabia aux phrases",
       "adapter un générateur de texte mot à mot au français et suivre ce qu'il écrit au fil des epochs.",
       "ch. 22 (`mylearn.sequences`) · Ex 24.30 · fiche §28.5", thread="Verne", tracks="C",
       body=r"""Recopie tes fonctions de 24.30 en les renommant `_27`, puis adapte-les. **Contrat** (les docstrings de la cellule suivante) :
- `tokenize_words_27(text)` : le texte en minuscules ; les mots sont des suites de **lettres de tout alphabet** (la classe `[^\W\d_]` du module `re`, qui inclut é, à, ç…), une apostrophe `'` ou `’` qui suit des lettres leur reste collée (`"l'"`, `"qu'"`, `"aujourd'"`) ; tout autre caractère qui n'est pas une espace forme un token (ponctuation, chiffres un par un). L'expression de 24.30, limitée aux lettres de a à z, couperait « été » en trois tokens ;
- `make_vocab_27`, `split_pairs_27` (la validation est la **fin** du roman), `train_lm_27` (comme `train_lm_30`, plus un rappel `on_epoch(epoch, model)` après chaque epoch) et `mean_loss_27` : comme en 24.30.

Le modèle est fourni : `WordLM2`, un embedding de 64, **deux** couches de LSTM de 128 cellules, une couche dense vers le vocabulaire (le modèle du livre). Vocabulaire : les 5 000 mots les plus fréquents et `<unk>`.

La cellule de vérification essaie ton tokenizer sur une phrase piégée, construit le vocabulaire et les paires, puis te fait reporter : a) le nombre de tokens de Verne ; b) la part de `<unk>` dans le texte encodé. La suivante entraîne `MODEL_27` (`EPOCHS_27` epochs, 12 en mode rapide, ≈ 4 s chacune, learning rate 0,01) et affiche la perplexité de validation après chaque epoch et trois extraits générés (température 0,8), après la première, la moitié et la dernière epoch. 🚀 En mode complet, 40 epochs.

Dans tes notes : compare les trois extraits (le livre parle de « charabia » puis de phrases presque sensées). La perplexité de validation remonte-t-elle à la fin (overfitting, 🧮 28.7) ? Quels mots les conjugaisons et les accords du français ajoutent-ils au vocabulaire, par rapport à l'anglais de Holmes ?""",
       todo=C.TODO_27, check=C.CHECK_27, solution=C.SOLUTION_27 + "\n\n\n" + C.CHECK_27, record=RECORD_27,
       after=[("code", C.EXPERIMENT_27)]),

    Ex("28.28", "🔬", 3, 30, "Glouton, température, top-k ou top-p : répétitions et diversité",
       "écrire les quatre stratégies de décodage et mesurer leurs effets sur les répétitions et la diversité du texte généré.",
       "Ex 28.27 · ch. 22 (`sample_from_logits`) · fiche §28.5 (encadré 🧮)", thread="Verne", tracks="C",
       body=r"""**Contrat**
- `pick_28(logits, strategy, rng, temperature=1.0, k=None, p=None)` : un identifiant (un `int` Python) tiré des logits `(V,)` du mot suivant : `"greedy"` (l'argmax, sans hasard), `"temperature"` (un tirage selon la softmax de `logits / temperature`), `"top_k"` (un tirage parmi les `k` plus grands logits seulement, probabilités renormalisées), `"top_p"` (un tirage parmi le plus petit ensemble des mots les plus probables dont la probabilité cumulée atteint `p`, renormalisé) ; un seul `rng.choice(ids, p=probs)` par tirage ;
- `repeated_share_28(ids, n=3)` : la part des n-grammes de `ids` qui sont déjà apparus plus tôt dans `ids` (0 s'il n'y en a pas).

La cellule de vérification contrôle les cas limites (glouton, top-k avec $k = 1$, top-p avec un petit $p$) et les fréquences de 6 000 tirages contre les probabilités attendues. La suivante génère, avec ton `MODEL_27`, 120 mots après « Phileas Fogg » pour cinq graines et cinq stratégies, et affiche la part de trigrammes répétés, la part de mots distincts et un extrait.

Dans tes notes : le glouton tombe-t-il dans une boucle, comme l'annonce la fiche ? Quelle stratégie donne le texte le plus varié, et à quel prix ? Pourquoi le top-p s'adapte-t-il mieux que le top-k à la confiance du modèle ?""",
       todo=C.TODO_28, check=C.CHECK_28, solution=C.SOLUTION_28 + "\n\n\n" + C.CHECK_28, after=[("code", C.EXPERIMENT_28)]),

    Ex("28.29", "🏆", 4, 90, "Défi : un transfert de style en moins de 2 minutes de CPU, loss de style divisée par 10 et contenu reconnaissable",
       "trouver le meilleur compromis entre style et contenu sous une contrainte de temps, mesuré par un juge fixe.",
       "Ex 28.23, 28.24", tracks="C",
       body=r"""Le défi porte sur une photo et un tableau que tu n'as pas encore utilisés pour régler : la fleur et *La Grande Vague* d'Hokusai. Écris `make_29(content, style)`, qui rend une image `(3, H, W)` dans $[0, 1]$ (tu peux appeler ton `style_transfer_23`, ou écrire autre chose). `evaluate_29` la lance une fois, la chronomètre et la mesure avec `measure_24`. Il faut **les trois** :
1. moins de 2 minutes de calcul ;
2. une distance de style au tableau divisée par **10** au moins, par rapport à celle de la photo elle-même ;
3. une distance de contenu à la photo sous `CONTENT_GOAL_29` (le contenu reste reconnaissable) : la valeur d'une solution de référence, plus 15 %.

Les réglages de la partie C passent le critère 2 haut la main, mais pas le 3 : il faut doser. Ne règle pas tes paramètres en regardant `CONTENT_29` et `STYLE_29` pas à pas : règle-les sur d'autres images (la photo `"china"`, les autres tableaux), puis lance `evaluate_29` une fois.

Dans tes notes : ta démarche de réglage, tes essais et leur temps ; ce que tu as appris du compromis entre style et contenu.""",
       given=C.GIVEN_29.replace("{GOAL}", str(A["goal"])), todo=C.TODO_29, solution=C.SOLUTION_29 + "\n\n\n" + C.CHECK_29, check=C.CHECK_29),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]
ROCKETS = ("28.16", "28.22", "28.23", "28.24", "28.27")
