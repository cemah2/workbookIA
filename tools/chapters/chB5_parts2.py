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

EXERCISES_C2 = [EX_22]
