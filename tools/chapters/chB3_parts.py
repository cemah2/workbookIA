"""Parts A and B of the chapter B3 notebook, first half (B3.12 to B3.20): statements and cells (used by build_chB3.py).

Part A: your NumPy library attention.py (B3.12 to B3.17). Part B: the mini-GPT in PyTorch, its module, the model and
its training on Holmes (B3.18 to B3.20); the second half of the chapter (B3.21 to B3.30) reloads the model that B3.20
saves (wb.gpt.learner_key) and appends its exercises to PART_B (or adds parts).
"""

from __future__ import annotations

import json
from pathlib import Path

from chapter_kit import Ex, Part, guarded

import chB3_code as C

_HISTORY_FILE = Path(__file__).resolve().parent / "history_chB3.json"
HISTORY = json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists() else {}

for _row in HISTORY.get("bug17", {}).get("versions", []):      # 🐛 B3.17 replayed on its 8 versions (history_chB3.py)
    if _row["first_failure"] != _row["expected"]:
        raise ValueError(f"🐛 B3.17: the version with the fixes {_row['fixed']!r} stops at the wrong point: {_row}")

P13 = HISTORY.get("p13", {}).get("answers", {"exponent": 3, "keeps_half": True})
TARGETS_20 = {"fast": HISTORY.get("gpt_fast", {}).get("target", 1.75), "full": HISTORY.get("gpt_full", {}).get("target", 1.6)}
LSTM_ROWS = HISTORY.get("lstm", {}).get("rows", [])
GPT_FULL = HISTORY.get("gpt_full", {})

RELOAD = C.RELOAD
TEST = "tests/test_chb3_attention.py"


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** : ouvre `mon_travail/mylearn/attention.py` (créé par `python tools/start_chapter.py "
            "B3`), lis la docstring, remplace chaque `raise NotImplementedError(...)` de l'exercice par ton code et "
            "**enregistre**. NumPy seulement. La première cellule de vérification recharge ta librairie et l'essaie sur des "
            "cas visibles ; la seconde lance les tests de l'exercice. `python -m pytest tests/test_chb3_attention.py -q` "
            "les lance tous dans un terminal.")


def tests_after(ex_id: str, probe: str, keyword: str, n_tests: int, what: str = "tes fonctions") -> list:
    """The second check cell of a 🔨: the tests of the exercise (explicit test file, -k group)."""
    cell = (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_ATT, "{keyword}")   # {n_tests} tests\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent {what}.")')
    ref = cell.replace('run_pytest_group(TEST_ATT, "', 'run_pytest_group(TEST_ATT, impl="ref", keyword="')
    return [("check", cell), ("solution", ref)]


COMMON_ATT = (
    "**Contrat commun de `attention.py`** (les docstrings, plus ces précisions que vérifient les tests) :\n"
    "- conventions : couches denses comme partout dans `mylearn`, `W` de forme `(n_in, n_out)` et $z = xW + b$ ; masques "
    "booléens comme `F.scaled_dot_product_attention`, `True` = la requête **peut** regarder la clé (l'inverse de "
    "`nn.MultiheadAttention`) ;\n"
    "- tes fonctions ne modifient jamais leurs entrées, ni les tableaux ni `params` ; les entrées peuvent être des listes "
    "(`lengths`, `q`, `k`, `v`, `mask`, `positions`, le `x` de `split_heads`) ;\n"
    "- les masques rendus sont de dtype `bool` ; une entrée invalide lève `ValueError`, jamais `KeyError` ;\n"
    "- les valeurs sont comparées à rtol = atol = $10^{-6}$ (un calcul en float32 passe ; $10^{-5}$ contre Hugging Face), "
    "les masques et les têtes exactement ;\n"
    "- tu réutilises tes fonctions des chapitres précédents : `softmax` et `gelu` (`nn/activations.py`, ch. 17), "
    "`layer_norm` (`nn/regularization.py`, ch. 20). Le stub les importe. Si l'un de ces fichiers manque dans ta "
    "librairie, la référence le remplace (une ligne ℹ️ le dit) ; s'il existe mais que la fonction n'est pas écrite, "
    "les tests qui en dépendent affichent ⏳ en nommant son fichier et son exercice.")

# ---------------------------------------------------------------------------------------------------------------------
PART_A = Part("A", "Ta librairie NumPy : attention.py",
              "Tu écris dans `mylearn/attention.py` les briques d'un Transformer, en NumPy, chacune comparée à PyTorch : "
              "les masques et l'attention par produit scalaire mise à l'échelle (B3.12), les têtes (B3.14), les "
              "encodages de position (B3.15), puis un bloc complet (B3.16, ★★★★). Entre deux, tu prédis ce que fait la "
              "division par $\\sqrt{d_k}$ quand une clé copie la requête (B3.13), et tu répares un petit modèle de langue "
              "dont trois masques mentent (B3.17). `positional_encoding` resservira en B5, pour l'encodage du pas de "
              "temps d'un modèle de diffusion.\n\n" + COMMON_ATT + "\n\n"
              "La cellule suivante charge les outils de la partie : `TEST_ATT` (le fichier des tests), `DATA` (Holmes en "
              "identifiants de caractères, le découpage figé de `wb.text.train_val(\"holmes\")`, que tu as découpé en "
              "caractères au ch. 22), `causal_np` (le masque causal des vérifications, écrit sans ton code), "
              "`params_from_mha` et `block_params_from_torch` (les poids d'un module PyTorch dans ta convention).",
              given=C.TOOLS_A, exercises=[
    Ex("B3.12", "🔨", 2, 30, "causal_mask, padding_mask et scaled_dot_product_attention",
       "écrire les deux masques et l'attention par produit scalaire mise à l'échelle de ta librairie, comparés à PyTorch.",
       "✏️ B3.2 · fiche §203.2 (« La formule »), §203.3 (« Le masque causal », « Le masque de padding », « Le masque "
       "causal quand il y a moins de requêtes que de clés »)", tracks="R, M, C", mylearn="attention.py",
       body=mylearn_howto() + r"""

**Contrat**
- `causal_mask(q_len, kv_len=None)` : le masque `(L, S)` de booléens $M_{ij} = (j \le i + S - L)$, aligné **en bas à droite** : la dernière requête voit toutes les clés (cas du KV cache, où les $S - L$ premières clés sont en cache) ; il coïncide avec `is_causal=True` de `F.scaled_dot_product_attention` quand $L = S$, pas quand $L < S$. `ValueError` si `q_len < 1` ou `kv_len < q_len`.
- `padding_mask(lengths, max_len)` : le masque `(N, S)` de booléens, `True` sur les vrais tokens, `False` sur le padding ; `ValueError` pour une longueur < 1 ou > `max_len`.
- `scaled_dot_product_attention(q, k, v, mask=None, scale=None)` rend le couple `(output, weights)` :
  - $\operatorname{softmax}(\mathbf{q}\mathbf{k}^\top \cdot \text{scale} + \mathbf{M})\,\mathbf{v}$ sur les deux derniers axes ; les axes de tête (batch, têtes) sont gardés et diffusés, et `mask` se diffuse sur `(..., L, S)` : `(L, S)`, `(N, 1, 1, S)`… ;
  - une clé masquée reçoit un poids de 0 **exactement**, et chaque ligne de poids somme à 1 ;
  - `scale` : seul `None` donne $1/\sqrt{d_k}$ ; `scale=0.0` est une échelle comme une autre (des scores nuls) ;
  - `ValueError` si les formes sont incompatibles, ou si une ligne du masque n'autorise **aucune** clé, y compris quand cette ligne vient d'un masque 4-D `(N, 1, 1, S)` (une séquence entièrement masquée). `F.scaled_dot_product_attention` rendrait une ligne de zéros et `nn.MultiheadAttention` des `nan` : ta librairie refuse.

La première cellule de vérification essaie tes masques sur des cas visibles (dont ceux des docstrings) et contre `torch.tril`, refait l'attention à la main de la fiche (§203.2 : une requête, quatre clés), compare ton attention à `F.scaled_dot_product_attention` sur un batch à têtes avec un masque de padding `(2, 1, 1, 7)` et avec le masque causal, essaie `scale=0.0`, puis trois refus. La seconde lance les 55 tests de l'exercice (`-k "test_causal_mask_ or test_padding_mask_ or test_scaled_dot_product_attention_"`).

Dans tes notes : avec `causal_mask(2, 5)`, quelle requête est le token de quelle position, et que donnerait `is_causal=True` à la place ? Pourquoi ta fonction refuse-t-elle une ligne sans aucune clé autorisée, plutôt que de rendre des zéros comme SDPA ?""",
       check=RELOAD + C.CHECK_12, solution=C.CHECK_12,
       after=tests_after("B3.12", "mylearn.attention.causal_mask(2)",
                         "test_causal_mask_ or test_padding_mask_ or test_scaled_dot_product_attention_", 55)),

    Ex("B3.13", "🔮", 1, 10, "Prédire la carte d'attention quand une clé copie la requête, puis quand on retire la "
                            "division par √d_k",
       "prédire, en ordre de grandeur, combien de clés aléatoires il faut pour qu'une clé égale à la requête ne prenne "
       "plus la moitié du poids, avec et sans la division par $\\sqrt{d_k}$.",
       "Ex B3.12 · ∂ B3.6 · fiche §203.2 (« Pourquoi diviser par $\\sqrt{d_k}$ »)", tracks="C", hypothesis=True,
       body=r"""Une requête $\mathbf{q}$ de dimension $d_k = 64$, de composantes indépendantes tirées selon $\mathcal{N}(0;\ 1)$. Parmi ses $S$ clés, l'une est $\mathbf{q}$ elle-même : une clé qui **copie** la requête ; les $S - 1$ autres sont tirées comme $\mathbf{q}$, indépendamment. La cellule suivante donne `copied_weight_13(S, scaled)`, qui calcule pour 200 requêtes le poids d'attention de la clé copiée parmi $S$ clés, avec (`scaled=True`) ou sans (`scaled=False`) la division par $\sqrt{d_k}$.

L'expérience trace d'abord deux cartes d'attention avec ta `scaled_dot_product_attention` (6 requêtes et 32 clés, dont les 6 premières copient les requêtes), avec puis sans la division ; puis le poids **médian** de la clé copiée, sur les 200 requêtes, pour $S$ = 10, 30, 100, 300, …, 100 000. **Avant d'exécuter**, écris tes deux prédictions :
- a) `exponent_13` : **avec** la division, l'entier $k$ tel que le poids médian de la clé copiée passe sous 1/2 entre $10^k$ et $10^{k+1}$ clés (par exemple `1` si la bascule a lieu entre 10 et 100 clés) ;
- b) `keeps_half_13` : **sans** la division, avec 100 000 clés, le poids médian de la clé copiée est-il encore au-dessus de 1/2 ? `True` ou `False`.

La cellule d'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites ; elle les vérifie à la fin. La courbe ne dépend pas de ton code ; les deux cartes, si (B3.12).""",
       given=C.GIVEN_13, todo=C.TODO_13,
       solution=f'exponent_13, keeps_half_13 = {P13["exponent"]}, {P13["keeps_half"]}',
       record=(f'wb.record("B3.13a", {P13["exponent"]}, mistakes={{'
               + ", ".join(f'"{msg}": {v}' for v, msg in {
                   0: "relis le tableau sous la courbe : à 10 clés, le poids médian avec la division est-il déjà sous 1/2 ?",
                   1: "relis le tableau sous la courbe : à 100 clés, le poids médian avec la division est-il déjà sous 1/2 ?",
                   2: "relis le tableau sous la courbe : à 1 000 clés, le poids médian avec la division est-il déjà sous 1/2 ?",
                   4: "relis le tableau sous la courbe : à 10 000 clés, le poids médian avec la division est-il encore au-dessus de 1/2 ?",
                   5: "relis le tableau sous la courbe : à 100 000 clés, le poids médian avec la division est-il encore au-dessus de 1/2 ?",
               }.items() if v != P13["exponent"])
               + '}, fractional="on demande l\'exposant k de la puissance de 10, un entier")\n'
               f'wb.record("B3.13b", {P13["keeps_half"]}, mistakes={{"relis le tableau sous la courbe : sans la '
               f'division, que vaut le poids médian à 100 000 clés ?": {not P13["keeps_half"]}}})'),
       after=[("code", guarded(C.EXPERIMENT_13, ["exponent_13", "keeps_half_13"],
                               "⏳ Ex B3.13 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Relie la courbe « avec la "
                     "division » à ∂ B3.6 : quel est le score typique de la clé copiée, celui des autres clés, et que "
                     "compare la softmax ? Refais la courbe avec `d_k = 16`, puis `d_k = 256` "
                     "(`copied_weight_13(n, True, d_k=…)`) : où se place la bascule, et que corrige exactement la "
                     "division par $\\sqrt{d_k}$ ?")]),

    Ex("B3.14", "🔨", 3, 45, "split_heads, merge_heads et multi_head_attention",
       "découper en têtes, recoller, puis assembler une attention multi-têtes complète, validée contre "
       "`nn.MultiheadAttention`.",
       "Ex B3.12 · ✏️ B3.3 · fiche §203.4 (« Une seule matrice pour toutes les têtes », piège du reshape direct)",
       tracks="R, M, C", mylearn="attention.py",
       body=mylearn_howto() + r"""

**Contrat**
- `split_heads(x, n_heads)` : `(N, T, D)` → `(N, H, T, D // H)` ; la tête $h$ reçoit les features $h\,d_h$ à $(h + 1)\,d_h - 1$ de chaque token ($d_h = D / H$). `ValueError` si `x` n'est pas 3-D ou si $H$ ne divise pas $D$.
- `merge_heads(x)` : l'inverse, `(N, H, T, d_h)` → `(N, T, H d_h)`, la tête 0 d'abord ; `merge_heads(split_heads(x, H))` redonne `x`. `ValueError` si `x` n'est pas 4-D.
- `multi_head_attention(query, key, value, params, n_heads, mask=None)` rend le couple `(output, weights)`, de formes `(N, L, D)` et `(N, H, L, S)` :
  - $\mathbf{Q} = \text{query}\,\mathbf{W}_q + \mathbf{b}_q$ (de même $\mathbf{K}$ avec `key` et $\mathbf{V}$ avec `value`), le découpage en têtes, ta `scaled_dot_product_attention` (B3.12) dans chaque tête avec son échelle par défaut, le recollage, puis la projection de sortie $\mathbf{W}_o$, $\mathbf{b}_o$ ; l'auto-attention est `query = key = value = x` ;
  - `params` : `W_q`, `W_k`, `W_v`, `W_o` de forme `(D, D)` et `b_q`, `b_k`, `b_v`, `b_o` de forme `(D,)`, dans un mapping quelconque (aussi en lecture seule, `MappingProxyType`) ; ses autres clés sont ignorées (les paramètres d'un bloc entier passent) ; un paramètre manquant lève `ValueError`, pas `KeyError` ;
  - `mask` (`True` = autorisé) de forme `(L, S)`, `(N, 1, 1, S)`, `(N, 1, L, S)` ou `(N, H, L, S)` ; les tests prennent aussi de l'attention croisée, où $L \ne S$.

La première cellule de vérification essaie tes deux fonctions de têtes sur l'exemple de la docstring, contre `view(N, T, H, d_h).transpose(1, 2)` de PyTorch, puis l'aller-retour et deux refus. Elle compare ensuite ton attention multi-têtes à `nn.MultiheadAttention` aux mêmes poids (la fonction `params_from_mha` de la cellule des outils montre la conversion : PyTorch range $\mathbf{W}_q^\top$, $\mathbf{W}_k^\top$, $\mathbf{W}_v^\top$ empilées dans `in_proj_weight`, et attend le masque inversé, `~mask`), en auto-attention causale puis en attention croisée avec un masque de padding, et essaie un paramètre manquant. La seconde lance les 48 tests de l'exercice (`-k "test_split_heads_ or test_merge_heads_ or test_multi_head_attention_"`).

Dans tes notes : un reshape direct `(N, T, D)` → `(N, H, T, d_h)`, sans échange d'axes, donne la bonne forme : quelles features reçoit alors chaque tête (fiche §203.4) ? Pourquoi `params_from_mha` transpose-t-elle chaque bloc de `in_proj_weight` ? Que changerait une échelle $1/\sqrt{D}$ au lieu de $1/\sqrt{d_h}$ ?""",
       check=RELOAD + C.CHECK_14, solution=C.CHECK_14,
       after=tests_after("B3.14", "mylearn.attention.split_heads(np.ones((1, 2, 4)), 2)",
                         "test_split_heads_ or test_merge_heads_ or test_multi_head_attention_", 48)),

    Ex("B3.15", "🔨", 2, 30, "positional_encoding et apply_rope",
       "écrire l'encodage sinusoïdal et RoPE, et vérifier leurs propriétés et la convention de Hugging Face.",
       "✏️ B3.4 · ∂ B3.7 · fiche §203.5 (« L'encodage sinusoïdal (2017) », « RoPE : tourner les requêtes et les clés »)",
       tracks="R, M, C", mylearn="attention.py",
       body=mylearn_howto() + r"""

**Contrat**
- `positional_encoding(seq_len, d_model, base=10000.0)` : le tableau `(T, D)` en float64 de $PE_{p, 2i} = \sin(p / \text{base}^{2i/D})$ et $PE_{p, 2i+1} = \cos(p / \text{base}^{2i/D})$, pour $i = 0 \dots D/2 - 1$. `base = 1` est accepté ; `ValueError` si `d_model` est impair, ou si `seq_len`, `d_model` ou `base` est < 1. Les tests vont jusqu'à $T = 1\,000$ : B5 s'en resservira.
- `apply_rope(x, positions=None, base=10000.0)` : `x` de forme `(..., T, d)`, $d$ pair, une requête ou une clé par position ; la convention **« rotate half »** de Hugging Face (Llama) : la dimension $i$ est appariée avec $i + d/2$ (pour $i < d/2$), pas avec $i + 1$ comme dans l'article de RoFormer, et la paire $(x_1, x_2)$ tourne de l'angle $\theta = \text{position} \cdot \text{base}^{-2i/d}$ :
  $(x_1 \cos\theta - x_2 \sin\theta,\ x_2 \cos\theta + x_1 \sin\theta)$ ;
  - `positions` : exactement $T$ positions (`np.arange(T)` si `None`) ; une seule position pour $T$ vecteurs est refusée, même si NumPy la diffuserait ;
  - rend un tableau de la forme de `x` ; `ValueError` si $d$ est impair ou si `positions` n'a pas la longueur $T$ ;
  - comparé à `LlamaRotaryEmbedding` et `apply_rotary_pos_emb` de transformers, à $10^{-5}$ (ils calculent les angles en float32).

La première cellule de vérification essaie ton encodage sur l'exemple de la docstring, sur 1 000 positions et sur la propriété du décalage (§203.5 : décaler de 7 positions applique la même rotation à chaque paire (sin, cos)), puis RoPE sur l'exemple de la docstring, la conservation des normes, la propriété relative (le produit scalaire ne dépend que de $m - n$, ∂ B3.7) et contre `LlamaRotaryEmbedding` ; enfin quatre refus. La seconde lance les 40 tests de l'exercice (`-k "test_positional_encoding_ or test_apply_rope_"`).

Dans tes notes : pourquoi RoPE tourne-t-il les requêtes et les clés, mais pas les valeurs ? Le produit scalaire d'une requête et d'une clé change-t-il quand on décale toutes les positions de 1 000 (`positions = np.arange(T) + 1000`) ? Mesure-le. Qu'apporte la convention « rotate half » par rapport aux paires $(2i, 2i + 1)$, et que perd-on à confondre les deux ?""",
       check=RELOAD + C.CHECK_15, solution=C.CHECK_15,
       after=tests_after("B3.15", "mylearn.attention.positional_encoding(2, 2)",
                         "test_positional_encoding_ or test_apply_rope_", 40)),

    Ex("B3.16", "🔨", 4, 100, "transformer_block en NumPy, validé contre nn.TransformerEncoderLayer",
       "assembler un bloc Transformer pré-LN complet en NumPy, et le valider au millionième contre PyTorch.",
       "Ex B3.14 · ch. 20 (layer norm, 20.17) · ch. 17 (GELU, 17.14) · fiche §203.6 (« Pré-LN ou post-LN », « Le bloc "
       "en équations, et sa vérification »)", tracks="C", mylearn="attention.py",
       body=mylearn_howto() + r"""

**Contrat** de `transformer_block(x, params, n_heads, mask=None, eps=1e-5)`, le bloc pré-LN de GPT-2, sans dropout :
- $\mathbf{h} = \mathbf{x} + \operatorname{MHA}(\operatorname{LN}_1(\mathbf{x}))$ en auto-attention (ta `multi_head_attention`, avec `mask`), puis $\text{out} = \mathbf{h} + \operatorname{gelu}(\operatorname{LN}_2(\mathbf{h})\,\mathbf{W}_1 + \mathbf{b}_1)\,\mathbf{W}_2 + \mathbf{b}_2$ ;
- la GELU **exacte** (avec la fonction d'erreur), pas l'approximation tanh ; `eps` sert aux **deux** layer norms ;
- `params` : les clés de `multi_head_attention`, plus `ln1_gamma`, `ln1_beta`, `ln2_gamma`, `ln2_beta` de forme `(D,)`, `W_1` `(D, F)`, `b_1` `(F,)`, `W_2` `(F, D)` et `b_2` `(D,)` ; la largeur $F$ du MLP se lit dans `W_1` ; un paramètre manquant lève `ValueError` ;
- rend `(N, T, D)` ; `x` et `params` ne sont pas modifiés.

**Tes fonctions des ch. 17 et 20.** Le stub importe ta `gelu` (`nn/activations.py`, Ex 17.14) et ta `layer_norm` (`nn/regularization.py`, Ex 20.17) : réutilise-les. Si l'un de ces fichiers manque dans ta librairie, la référence le remplace (une ligne ℹ️ le dit). S'il existe mais que la fonction n'est pas encore écrite, la première cellule de vérification te le signale : écris-la (son exercice), ou recode la layer norm et la GELU dans `attention.py`, ce que les tests acceptent aussi. Les tests passent par tes fonctions si tu les réutilises : si l'un d'eux échoue, lance d'abord les tests de 17.14 et de 20.17 (le message de l'échec le dit).

La première cellule de vérification construit le même bloc en PyTorch, `nn.TransformerEncoderLayer(16, 4, 64, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)` en float64, recopie ses poids dans ta convention (`block_params_from_torch` dans la cellule des outils : `W = weight.T` pour chaque couche dense), et compare ta sortie à la sienne avec le masque causal (passé à PyTorch en `~mask`), sans masque, puis avec `eps = 0.1` ; elle essaie enfin un paramètre manquant. La seconde lance les 25 tests de l'exercice (`-k "test_transformer_block_"`).

Dans tes notes : avec tous les poids à zéro (l'exemple de la docstring), que rend ton bloc, et que dit ce résultat des connexions résiduelles ? En quoi ton bloc changerait-il pour un post-LN (§203.6) ? Remplace ta GELU par l'approximation tanh : de combien ta sortie s'écarte-t-elle de celle de PyTorch, et le test l'aurait-il vu ?""",
       check=RELOAD + C.CHECK_16, solution=C.CHECK_16,
       after=tests_after("B3.16", "mylearn.attention.transformer_block(np.zeros((1, 1, 2)), {}, 1)",
                         "test_transformer_block_", 25, "ta fonction transformer_block")),

    Ex("B3.17", "🐛", 2, 25, "Trois masques qui mentent : convention inversée, masque transposé, padding compté dans la loss",
       "trouver et corriger les trois erreurs de masque d'un petit modèle de langue, guidé par des diagnostics enchaînés.",
       "Ex B3.12 · fiche §203.3 (encadrés 🕰️ sur `F.scaled_dot_product_attention` et `nn.MultiheadAttention`), §203.8 "
       "(« La loss et la perplexité ») · tableau des pièges ⚠️", thread="Holmes", tracks="R, C",
       body=r"""Une collègue a écrit un petit modèle de langue au niveau caractère : une couche d'attention causale sur les embeddings des caractères et de leurs positions, puis une couche dense. Elle calculait d'abord l'attention avec `F.scaled_dot_product_attention`, puis elle est passée à `nn.MultiheadAttention` pour lire les poids d'attention. Le modèle s'entraîne sur des lignes de Holmes de longueurs différentes, complétées par du padding. Son code contient **trois** erreurs. Corrige-le **sur place**, dans sa cellule, sans changer les noms, les signatures ni ce que rendent les fonctions.

Ce qu'il devrait faire :
- `causal_mask_17(T)` : le masque causal `(T, T)` dans la convention de `F.scaled_dot_product_attention`, `True` là où la requête $i$ peut regarder la clé $j$ ;
- `TinyLM_17` : des ids `(N, T)` → des logits `(N, T, V)`, chaque position ne voyant que les caractères jusqu'à elle ; `self.weights` garde les poids d'attention (la moyenne des deux têtes) ;
- `lm_loss_17(logits, targets)` : la cross-entropy moyenne sur les **vrais** caractères du batch ; les cibles valent `PAD_17` sur le padding.

La cellule donnée prépare les données : `LINES_17` (des lignes de l'entraînement de Holmes, de 24 à 64 caractères), `batch_17` (un batch de lignes, entrées et cibles décalées d'un caractère, complétées par `PAD_17`), `VAL_17` (64 fenêtres de 64 caractères de la validation, sans padding). Elle définit aussi `diagnose_17`. La cellule de vérification lance `diagnose_17`, trois diagnostics enchaînés, chacun ne s'exécutant que si le précédent passe, car une erreur peut en cacher une autre :
- la loss d'un batch est-elle un nombre ?
- le modèle voit-il le futur ? (un caractère changé, puis 200 itérations d'entraînement, quelques secondes)
- la loss d'un batch dépend-elle de son padding ?

Elle tourne dès le premier « Tout exécuter » : ses messages sont ton point de départ. Une fois les trois points passés, la cellule suivante trace les poids d'attention de ta version réparée sur une ligne de Holmes.

Dans tes notes : les trois erreurs, et pour chacune le symptôme qu'elle produirait seule. Pourquoi les `nan` n'apparaissent-ils qu'à une seule position, et laquelle, selon que l'autre erreur de masque est encore là ou non ? Laquelle des trois erreurs fait baisser la loss d'entraînement sans que le modèle prédise mieux le texte ?""",
       given=C.GIVEN_17, todo=C.TODO_17, check=C.CHECK_17, solution=C.SOLUTION_17 + "\n\n\n" + C.CHECK_17,
       after=[("code", C.EXPERIMENT_17)]),
])

# ---------------------------------------------------------------------------------------------------------------------
def _lstm_rows() -> str:
    rows = [(r["name"], r["params"], r["budget"], r["char_loss"]) for r in LSTM_ROWS]
    if GPT_FULL:
        rows.append((GPT_FULL["name"], GPT_FULL["params"], GPT_FULL["budget"], GPT_FULL["char_loss"]))
    return "[\n" + "".join(f"    {row!r},\n" for row in rows) + "]"


GIVEN_20 = (C.GIVEN_20 + "\n"
            f"TARGET_20 = {TARGETS_20['fast']} if FAST_MODE else {TARGETS_20['full']}   # nats per character on the "
            "whole validation (char_loss)\n"
            "# The comparison (precomputed by tools/chapters/history_chB3.py on the same split, the same vocabulary and the "
            "same\n# validation targets, every character from index 128 on): (model, parameters, budget, nats per character)\n"
            f"LSTM_20 = {_lstm_rows()}")

PART_B = Part("B", "Le mini-GPT en PyTorch",
              "Tu passes de NumPy à PyTorch : le module d'attention causale du mini-GPT, comparé à ta version NumPy "
              "(B3.18), le modèle complet (B3.19), puis son entraînement sur Holmes, face au LSTM du ch. 22 (B3.20). Le "
              "modèle suit la structure de nanoGPT (A. Karpathy, licence MIT, fiche §203.8) ; le contrat interdit "
              "`nn.Transformer…` et `nn.MultiheadAttention` dans ton modèle.\n\n"
              "Le modèle que B3.20 entraîne est **sauvegardé une fois** (son `state_dict`, sa configuration, le "
              "vocabulaire et l'historique de l'entraînement), hors du dépôt, sous une clé qui dit tout ce qui l'a produit "
              "(configuration, réglages, graine, mode rapide ou complet, empreinte du découpage) : la suite du chapitre "
              "le recharge sans réentraîner. Ton modèle doit donc suivre les noms du contrat de B3.19.\n\n"
              "La cellule suivante charge les outils de la partie : `MINI_CONFIG` (4 blocs, 4 têtes, $d = 128$, contexte "
              "de 128 caractères, 88 caractères), `n_params`, `forbidden_in` (les modules interdits trouvés dans un "
              "modèle), `params_from_qkv` (les poids de ton module d'attention dans la convention de "
              "`multi_head_attention`). Le vocabulaire est celui de ton `build_vocab` (ch. 22) : les caractères par "
              "fréquence décroissante, à égalité dans l'ordre de `sorted` ; la cellule vérifie que ta fonction donne le "
              "même. `wb.gpt` contient aussi une solution de référence de B3.18 à B3.20 (`ReferenceGPT`, `train`) : le "
              "notebook ne s'en sert que pour vérifier tes fonctions ou quand ton modèle manque ; ne la lis qu'une fois "
              "ta version écrite.",
              given=C.TOOLS_B, exercises=[
    Ex("B3.18", "📦", 2, 30, "De NumPy à PyTorch : un module CausalSelfAttention avec F.scaled_dot_product_attention",
       "écrire le module d'attention causale du mini-GPT en PyTorch, et mesurer son écart à ta version NumPy.",
       "Ex B3.14 · ch. 20 (`nn.Module`, dropout) · fiche §203.3 (encadré 🕰️ sur `F.scaled_dot_product_attention`), "
       "§203.4", tracks="R, M, C",
       body=r"""**Contrat** de `CausalSelfAttention_18(d_model, n_head, dropout=0.0)`, un `nn.Module` :
- exactement deux couches denses, avec biais : `qkv = nn.Linear(d_model, 3 * d_model)`, dont les sorties sont, dans l'ordre, les requêtes, les clés puis les valeurs ; et `proj = nn.Linear(d_model, d_model)`, la projection de sortie $\mathbf{W}_O$ ; `ValueError` si `n_head` ne divise pas `d_model` ;
- `forward(x)` : `x` de forme `(N, T, d_model)` → `(N, T, d_model)` : le découpage en `n_head` têtes comme ton `split_heads` (B3.14), l'attention causale par `F.scaled_dot_product_attention` avec son échelle par défaut, le recollage des têtes comme `merge_heads`, puis `proj` ;
- dropout : `dropout` sur les poids d'attention (`dropout_p`) et un `nn.Dropout(dropout)` sur la sortie de `proj`, en mode train seulement ; en mode eval, aucun dropout (`dropout_p = 0`, fiche §203.3) ;
- ni `nn.MultiheadAttention` ni `nn.Transformer…`.

La cellule de vérification contrôle les noms et les formes des paramètres, puis compare ta sortie, en mode eval, à celle de `nn.MultiheadAttention` aux mêmes poids, avec le masque causal (l'oracle de PyTorch), et à ta `multi_head_attention` NumPy de B3.14 avec les mêmes poids, convertis par `params_from_qkv` (`W = weight.T`). L'écart maximal entre ton module (float32) et ta version NumPy (float64) est vérifié : il doit rester sous $10^{-5}$ (rien à taper). Elle vérifie enfin la causalité, puis le dropout en mode train et en mode eval.

Dans tes notes : quel écart maximal mesures-tu entre tes deux versions, et d'où vient-il ? Pourquoi `view` juste après `transpose` lève-t-il une erreur, et que fait `reshape` à la place (fiche §203.4) ? Que se passerait-il en passant à la fois un masque et `is_causal=True` à SDPA (fiche §203.3) ?""",
       todo=C.TODO_18, check=C.CHECK_18, solution=C.SOLUTION_18 + "\n\n\n" + C.CHECK_18, record=C.RECORD_18),

    Ex("B3.19", "📦", 3, 60, "Mini-GPT caractère par caractère : embeddings de tokens et de positions, blocs pré-LN, "
                            "tête liée",
       "assembler le mini-GPT complet en PyTorch, avec la structure et l'initialisation de nanoGPT.",
       "Ex B3.18 · Ex B3.15 · ✏️ B3.5 j) · fiche §203.6, §203.7 (« Le décodeur : GPT »), §203.8 (« Le modèle », « La "
       "recette d'entraînement »)", thread="Holmes", tracks="R, M, C",
       body=r"""**Contrat.** Les noms des modules comptent : B3.20 sauvegarde ton modèle par son `state_dict`, et la suite du chapitre le recharge et lit ses blocs.
- `Block_19(d_model, n_head, dropout=0.0)`, un bloc pré-LN : `ln1` et `ln2` (`nn.LayerNorm(d_model)`), `attn` (ton `CausalSelfAttention_18`), `mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))` ; `forward(x)` : d'abord `x + attn(ln1(x))`, puis on ajoute `mlp(ln2(·))` du résultat.
- `MiniGPT_19(vocab_size, block_size=128, n_layer=4, n_head=4, d_model=128, dropout=0.0)` :
  - `tok_emb` (`nn.Embedding(vocab_size, d_model)`), `pos_emb` (des positions apprises, `nn.Embedding(block_size, d_model)`), un `nn.Dropout(dropout)` sur leur somme, `blocks` (un `nn.ModuleList` de `n_layer` `Block_19`), `ln_f` (`nn.LayerNorm(d_model)`) et `head` (`nn.Linear(d_model, vocab_size, bias=False)`) ;
  - **poids liés** : `head.weight` **est** `tok_emb.weight`, le même tenseur, compté une fois ;
  - **initialisation de nanoGPT** (fiche §203.8) : les poids des couches denses et des deux tables selon $\mathcal{N}(0;\ 0{,}02^2)$, les biais à zéro, et les deux projections qui écrivent dans le chemin résiduel de chaque bloc, `attn.proj` et `mlp[2]`, selon $\mathcal{N}\big(0;\ (0{,}02 / \sqrt{2\,n_{\text{layer}}})^2\big)$ ; les layer norms gardent $\gamma = 1$ et $\beta = 0$ ;
  - `self.block_size` garde `block_size` ; `forward(idx)` : des ids `(N, T)` int64 → des logits `(N, T, vocab_size)` ; `ValueError` si `T > block_size` ;
  - le `state_dict` a les noms et les formes de `wbgpt.expected_state_shapes()` (affiche-le pour comparer).
- Ni `nn.Transformer…` ni `nn.MultiheadAttention`.

La cellule de vérification construit ton modèle avec la configuration du contrat (`MINI_CONFIG`) et contrôle : les noms et les formes de son `state_dict`, les poids liés, son nombre de paramètres (celui que tu as calculé en ✏️ B3.5 j ; la cellule ne le redonne pas), la présence de tes `Block_19` et `CausalSelfAttention_18`, l'initialisation, la forme des logits, la loss à l'initialisation (le contrôle de départ de la fiche), la causalité et le refus d'une séquence trop longue.

Dans tes notes : mesure la loss de départ d'un modèle dont les tables gardent l'initialisation par défaut de PyTorch, $\mathcal{N}(0;\ 1)$ pour `nn.Embedding` : que se passe-t-il, et pourquoi la tête liée l'aggrave-t-elle ? Pourquoi réduire l'initialisation des deux projections résiduelles quand le nombre de blocs grandit ?""",
       todo=C.TODO_19, check=C.CHECK_19, solution=C.SOLUTION_19 + "\n\n\n" + C.CHECK_19),

    Ex("B3.20", "📦", 3, 60, "Entraîner le mini-GPT sur Holmes : AdamW, warmup + cosinus, clipping, perplexité, face au "
                            "LSTM du ch. 22",
       "entraîner le mini-GPT avec la recette de nanoGPT, mesurer sa perplexité de validation et la comparer à celle du "
       "LSTM du ch. 22.",
       "Ex B3.19 · ch. 19 (AdamW, warmup + cosinus, clipping ; 19.19) · ch. 22 (22.25, 22.26) · fiche §203.8 (« Les "
       "données », « La loss et la perplexité », « La recette d'entraînement »)", thread="Holmes", tracks="R, M, C",
       body=r"""Les données : `DATA.train_ids` et `DATA.val_ids`, deux tenseurs 1-D d'ids (le découpage figé de `wb.text.train_val("holmes")` : 90 % puis 10 %, coupé à une fin de ligne, jamais mélangé : la validation n'est jamais vue) ; `get_batch`, celui de la fiche ; `TRAIN_20`, les réglages (affichés par la cellule suivante).

**Contrat**
- `configure_optimizer_20(model, lr, weight_decay, betas=(0.9, 0.99))` : un `torch.optim.AdamW` à deux groupes, `weight_decay` sur les paramètres à deux axes ou plus (les matrices et les tables d'embedding), 0 sur les autres (les biais, les $\gamma$ et $\beta$ des layer norms) ; chaque paramètre une seule fois.
- `estimate_loss_20(model, data, n_batches, batch_size, block_size, generator)` : la moyenne, en float Python, des loss de `n_batches` batches `get_batch(data, batch_size, block_size, generator)`, en mode eval et sans gradient ; le modèle revient ensuite en mode train.
- `train_20(model, train_ids, val_ids, settings, generator, block_size=128)` entraîne `model` sur place :
  - avec l'optimiseur de `configure_optimizer_20(model, settings["lr"], settings["weight_decay"], (0.9, settings["beta2"]))` ;
  - à chaque itération `it`, de 0 à `iters − 1`, dans cet ordre : le learning rate de tous les groupes, `cosine_lr(it, iters, lr, warmup, min_lr)` de ton ch. 19 (19.19 : montée linéaire depuis 0 pendant `warmup` itérations, puis cosinus jusqu'à `min_lr`) ; un batch `get_batch(train_ids, batch_size, block_size, generator)` ; la loss, la cross-entropy moyenne sur les $N \times T$ positions (fiche §203.8) ; les gradients ; le clipping à la norme `settings["clip"]` ; le pas de l'optimiseur ;
  - l'évaluation, `estimate_loss_20` sur `val_ids` avec `eval_batches` batches, avant la première itération, après chaque multiple de `eval_interval` itérations et après la dernière ; chacune tire ses batches avec un générateur neuf, `torch.Generator().manual_seed(settings["eval_seed"])` (les mêmes batches de validation à chaque fois ; le générateur de l'entraînement ne sert jamais à l'évaluation) ;
  - rend l'historique, un dictionnaire de listes : `"train_loss"` et `"lr"` (une valeur par itération), `"val_iter"` (le nombre d'itérations faites à chaque évaluation : 0, 50, 100…) et `"val_loss"`.

La cellule de vérification contrôle d'abord tes trois fonctions sur un petit modèle (celui de référence de `wb.gpt`, pour ne pas dépendre de B3.19) : les groupes de l'optimiseur, paramètre par paramètre ; `estimate_loss_20` (le mode, le gradient, la valeur) ; puis 6 itérations de `train_20`, comparées à la recette du contrat, avec un clipping qui agit à chaque itération. La cellule suivante entraîne ton `MiniGPT_19` sur Holmes : environ 2 min 20 sur un CPU à deux cœurs en mode rapide (600 itérations) ; 🚀 en mode complet (5 000 itérations, 15 à 20 min sur CPU), un GPU de Colab est conseillé. Elle mesure ensuite sa loss sur **toute** la validation (`wbgpt.char_loss` : chaque caractère à partir du 129ᵉ, prédit à partir des 97 à 128 caractères qui le précèdent), la compare au seuil `TARGET_20` et au LSTM du ch. 22 sur les mêmes caractères de validation (le tableau `LSTM_20`, précalculé), puis **sauvegarde** ton modèle. Le temps affiché est une mesure, pas une réponse : il dépend de la machine.

Dans tes notes : le mini-GPT bat-il le LSTM du ch. 22 à temps de calcul voisin ? Qu'est-ce qui rend la comparaison inégale (paramètres, contexte, caractères vus) ? Pourquoi la dernière loss estimée pendant l'entraînement (8 batches de validation) et celle de `char_loss` (toute la validation) diffèrent-elles un peu ?""",
       given=GIVEN_20, todo=C.TODO_20, check=C.CHECK_20, solution=C.SOLUTION_20 + "\n\n\n" + C.CHECK_20,
       after=[("code", C.EXPERIMENT_20)]),
])

PARTS = [PART_A, PART_B]
