# B3 · Notes de rédaction de la fiche (session 67)

Notes pour le rédacteur des exercices (`02_exercices.md`, indices, solutions, partie 0, notebook). Elles ne sont pas destinées à l'apprenant. Tout ce qui suit a été **exécuté** avec `/root/venv313` (Python 3.13, torch 2.11.0+cpu, NumPy 2.1.3, transformers 5.21.0) ou **lu dans la source citée**, le 2026-11-01.

## 1. Fichiers produits

- `chapitres/b3_transformers/01_fiche.md` : 15 299 mots (`len(text.split())`), soit **120 min** à 125 mots/min (5 × round(122,4 / 5)). C'est **à la limite** : 14 mots de plus font passer à 125 min. L'en-tête dit « lecture de la fiche ≈ 2,0 h » et « ≈ 22 h » au total (valeurs du SYLLABUS). Les dix sections 203.x sont marquées ⏩ (bonus : tous les parcours lisent toute la fiche).
- `tools/chapters/figures_chB3.py` → `figures/` (7 PNG, ≈ 5 s, sans réseau ni entraînement) : `goulot.png`, `qkv.png`, `masques.png`, `tetes.png`, `positions.png`, `bloc.png`, `cout.png`. Rien d'appris ni de mesuré : les poids de `goulot.png` sont marqués « illustratifs » ; `cout.png` est théorique (mémoire $32 T^2 \times 4$ octets ; part de l'attention $T / (T + 12d)$ pour $d$ = 512, 2 048, 8 192).
- `tests/infra/test_terminology.py` et `tests/infra/test_syllabus.py` passent (48 tests) ; `syllabus.py check` : 0 problème. Tous les blocs de code de la fiche ont été rejoués d'un seul tenant (17 blocs) : les sorties en commentaire sont celles de l'exécution.

## 2. Chiffres vérifiés en exécutant

### Conventions PyTorch (2.11, CPU)
- `F.scaled_dot_product_attention` : masque booléen `True` = participe ; masque flottant ajouté. Vérifié contre une softmax explicite.
- `is_causal=True` avec $L \ne S$ : triangle **en haut à gauche** (`tril(diagonal=0)`), pas en bas à droite. `torch.nn.attention.bias.causal_lower_right(L, S)` = `tril(diagonal=S-L)` = la convention de `causal_mask`.
- `attn_mask` **et** `is_causal=True` ensemble : avec `sdpa_kernel(SDPBackend.MATH)` → `RuntimeError: _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True` ; avec le noyau par défaut du CPU (FLASH_ATTENTION), **aucune erreur** et le résultat est `mask & causal`. La doc dit « An error is thrown if both… ». La fiche dit de ne jamais les passer ensemble.
- Ligne entièrement masquée : SDPA rend une **ligne de zéros** (noyaux MATH et FLASH sur CPU) ; `nn.MultiheadAttention` rend des `nan`. La docstring de `scaled_dot_product_attention` (mylearn) exige une `ValueError`.
- **Noyaux sur CPU** : `torch._fused_sdp_choice(q, k, v, ...)` rend 1 = `FLASH_ATTENTION`, même avec un masque booléen et en float64 ; `EFFICIENT_ATTENTION` n'existe pas sur CPU (« No viable backend »). `MATH` = 0. Écart flash − math en float32, (1, 4, 512, 32), causal : 7,2·10⁻⁷. ⚠️ Pour 🔬 B3.27 (« noyau math et noyau efficace de SDPA sur CPU ») : le noyau « efficace » à comparer sur CPU est `SDPBackend.FLASH_ATTENTION` ; la fiche l'appelle « noyau de type FlashAttention écrit pour CPU ». La doc 2.11 dit pourtant « For all other backends, the PyTorch implementation will be used » : c'est l'exécution qui fait foi.
- `enable_gqa=True` = `repeat_interleave` des têtes de clés sur l'axe 1 (pas `repeat`).
- `nn.MultiheadAttention` : `batch_first=False` par défaut ; `attn_mask` et `key_padding_mask` booléens `True` = masqué ; `average_attn_weights=True` par défaut. Avec le masque causal **non inversé** : la ligne 0 voit toutes les clés sauf la 0, la dernière ligne est `nan`. Paramètres de `nn.MultiheadAttention(8, 2)` : 288 (`in_proj_weight` (24, 8), `in_proj_bias` (24,), `out_proj.weight` (8, 8), `out_proj.bias` (8,)).
- `view` après `transpose` : `RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.`
- GELU exacte contre GELU tanh : écart maximal 4,7·10⁻⁴ sur [−10, 10] (la fiche dit « moins de 10⁻³ » et « environ 5 × 10⁻⁴ »).
- GPT-2 de transformers range ses couches en `Conv1D` de poids `(n_in, n_out)` : la convention `mylearn` (`z = x @ W + b`) ; `lm_head.weight is transformer.wte.weight` → True.
- transformers 5.21 : `transformers.models.llama.modeling_llama.rotate_half`, `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)` et `LlamaRotaryEmbedding(config)` existent ; la base se lit dans `config.rope_parameters["rope_theta"]` (API 5.x) et les angles sont calculés en float32 ; `rotate_half(arange(1., 7.))` = `[-4, -5, -6, 1, 2, 3]`.

### Nombres de paramètres
- **Mini-GPT du contrat** (4 blocs, 4 têtes, $d$ = 128, contexte 128, biais partout comme nanoGPT `bias=True`, positions apprises, tête liée sans biais, layer norm finale), vocabulaire des caractères de `wb.datasets.load_holmes()` ($V$ = 88) : **820 992** paramètres = 4 × 198 272 (bloc) + 11 264 (tokens) + 16 384 (positions) + 256 (LN finale). Un bloc : $12d^2 + 13d$ = 196 608 + 1 664 = 198 272. Sans les embeddings (convention de Kaplan) : 793 344. **Non donné par la fiche** (elle dit seulement « environ 0,8 million »).
- Bloc `nn.TransformerEncoderLayer(256, 8, 1024, norm_first=True)` : 789 760 (**utilisé par la fiche**).
- GPT-2 small (config du Hub) : **124 439 808** = 12 × 7 087 872 + 38 597 376 (wte) + 786 432 (wpe) + 1 536 (ln_f). **Non donné par la fiche** (✏️ B3.5). ⚠️ Le tableau 2 de l'article GPT-2 dit « 117M » pour ce modèle et « 345M » pour medium : écart connu, à exploiter ou à signaler dans B3.5.
- GPT-2 medium : **354 823 168** = 24 × 12 596 224 + 51 463 168 + 1 048 576 + 2 048 (**utilisé par la fiche**, décomposition complète). B3.5 ne doit donc pas porter sur medium.
- BERT-base (`BertModel`, pooler compris) 109 482 240 ; ViT-B/16 (`ViTModel`, sans pooler) 85 798 656 ; T5-base (`T5ForConditionalGeneration`) 222 903 552. La fiche ne cite que « environ 110 millions » (BERT) et « environ 86 millions » (ViT, chiffre de l'article).

### Données
- `load_holmes()` : 562 203 caractères, 88 distincts ; `load_verne()` : 421 336 caractères, 103 distincts (la fiche B2 cite déjà 88 pour Holmes).

### Coûts
- KV cache (formule du contrat, GQA : $d \to n_{kv} d_h$). **Fiche** : Llama 3.1 8B (32 couches, 8 × 128, bf16) à 131 072 tokens = 17 179 869 184 octets = 16 Gio, 128 Kio par token, 64 Gio sans GQA.
- Mini-GPT, pour info (non donné) : $2 \times 4 \times 128 \times 128 \times 4$ octets (float32, contexte plein) = 524 288 octets = 512 Kio par séquence.
- FLOPs $6ND$. **Fiche** : Llama 3 405B, 15,6·10¹² tokens → 3,79·10²⁵ (rapport : 3,8·10²⁵) ; Chinchilla 70 Md × 1,4·10¹² = 5,88·10²³ (budget Gopher 5,76·10²³, cité). Pour 🧮 B3.8 (non donné) : GPT-3, 175 Md × 300 Md tokens → 3,15·10²³ (tableau D.1 de Brown et coll. : 3,14E+23, 3,64E+03 pétaflop/s-jours) ; mini-GPT en mode rapide (600 itérations × 32 × 128 = 2 457 600 tokens) → 6 × 820 992 × 2 457 600 ≈ 1,21·10¹³ FLOPs (≈ 1,17·10¹³ sans les embeddings), mode complet (5 000 itérations, 20,48 M tokens) ≈ 1,01·10¹⁴. Le terme d'attention ajoute $T/(12d)$ = 128/1 536 ≈ 8 % pour le mini-GPT.
- Mémoire des poids d'attention d'une couche (fiche) : 1 séquence, 32 têtes, float32 : 8 Gio à $T$ = 8 192, 2 Tio à 131 072.
- Seuil où l'attention domine un décodeur causal : $T = 12d$ (fiche : 49 152 pour $d$ = 4 096), cohérent avec Kaplan et coll. (« $d_{model} > n_{ctx}/12$ »).

### Modèles cités (`config.json` lus sur le Hub le 2026-11-01)
| Modèle | couches | $d$ | têtes q / kv | $d_h$ | MLP | positions | autres |
|---|---|---|---|---|---|---|---|
| openai-community/gpt2 | 12 | 768 | 12 / 12 | 64 | 3 072, `gelu_new` | 1 024 apprises | V = 50 257, LN eps 1e-5 |
| gpt2-medium | 24 | 1 024 | 16 | 64 | 4 096 | 1 024 | |
| gpt2-xl | 48 | 1 600 | 25 | 64 | 6 400 | 1 024 | |
| bert-base-uncased | 12 | 768 | 12 | 64 | 3 072 gelu | 512 | V = 30 522 |
| t5-base | 12 + 12 | 768 | 12 | 64 | 3 072 | relatives | V = 32 128 |
| vit-base-patch16-224 | 12 | 768 | 12 | 64 | 3 072 | 196 + 1 | patch 16, image 224 |
| Qwen3-0.6B | 28 | 1 024 | 16 / 8 | 128 | 3 072 SwiGLU | 40 960, θ = 10⁶ | liés, sans biais |
| Qwen3-8B | 36 | 4 096 | 32 / 8 | 128 | 12 288 | 40 960, θ = 10⁶ | non liés |
| Qwen3-30B-A3B | 48 | 2 048 | 32 / 4 | 128 | 128 experts, 8 actifs, 768 | 40 960 | |
| Llama 3 8B (copie NousResearch) | 32 | 4 096 | 32 / 8 | 128 | 14 336 | 8 192, θ = 5·10⁵ | V = 128 256 |
| Llama 3.1 8B (copie unsloth) | 32 | 4 096 | 32 / 8 | 128 | 14 336 | 131 072, θ = 5·10⁵, `llama3` × 8 | |
| Mistral-7B-v0.1 | 32 | 4 096 | 32 / 8 | 128 | 14 336 | 32 768, θ = 10⁴ | fenêtre glissante 4 096 |
| gpt-oss-20b | 24 | 2 880 | 64 / 8 | 64 | 32 experts, 4 actifs | 131 072, YaRN × 32 depuis 4 096 | fenêtre 128 une couche sur deux, biais d'attention |

Le dépôt `meta-llama/Meta-Llama-3-8B` est à accès contrôlé (401) : copies publiques lues à la place, comme en B2.

## 3. Valeurs déjà utilisées par la fiche (à ne pas reprendre comme réponses)

- Attention à la main (203.2-203.3) : $d_k = 4$, $d_v = 2$, $\mathbf{q} = (1, 0, 1, 0)$, clés $(1,1,0,0)$, $(0,1,1,0)$, $(2,0,2,0)$, $(0,0,0,2)$, valeurs $(1,0)$, $(0,1)$, $(2,2)$, $(-1,1)$ ; scores 0,5 0,5 2 0 ; poids 0,1411 0,1411 0,6323 0,0856 ; sortie (1,3201 ; 1,4912) ; avec masque causal (token 2 ne voit pas le token 3) : poids 0,1543 0,1543 0,6914 0, sortie (1,5372 ; 1,5372). ✏️ B3.1 et B3.2 (trois tokens, $d_k = 2$) doivent prendre d'autres nombres.
- Variance des scores simulée ($d_k$ = 4, 64, 512 ; graine 0).
- Masques : `tril(4×4)`, padding `[3, 5]` sur 5, $L$ = 2 et $S$ = 5 (code), $L$ = 2 et $S$ = 6, longueurs 4 et 6 (figure). Les docstrings du stub publient déjà `causal_mask(3)`, `causal_mask(2, 4)`, `padding_mask([2, 3], 4)`.
- MHA : `nn.MultiheadAttention(8, 2)` = 288 paramètres ; formule $4d^2 + 4d$ ; GQA 8 têtes de requêtes / 2 de clés.
- Positions : $d$ = 16 (périodes, $PE_5 \cdot PE_9 = PE_{20} \cdot PE_{24} \approx 5{,}5597$, $PE_p \cdot PE_p = 8$) ; figure : 64 × 64 ; la matrice du décalage sur (sin, cos) est donnée en général. Le stub publie déjà `positional_encoding(3, 4)` et `apply_rope(np.ones((3, 4)))`.
- RoPE 2D : θ = 0,5 rad, $\mathbf{q} = (1, 2)$, $\mathbf{k} = (3, -1)$, paires (0, 3), (4, 7), (10, 13) → 7,0532 ; (7, 4) → −6,9117. Le rappel 🧮 donne la matrice de rotation, $R_\beta R_\alpha = R_{\alpha+\beta}$, $R_\theta^\top = R_{-\theta}$ et la conservation des normes, **pas** l'assemblage $\langle R_m q, R_n k\rangle = q^\top R_m^\top R_n k$ (∂ B3.7).
- Variance : le rappel 🧮 donne $\operatorname{Var}(XY) = \operatorname{Var}X \operatorname{Var}Y$ (centrées, indépendantes) avec l'exemple des ±1 ; la preuve complète de $\operatorname{Var}(\mathbf{q}\cdot\mathbf{k}) = d_k$ reste à ∂ B3.6.
- Bloc : $d$ = 256, $F$ = 1 024 (789 760) ; GPT-2 medium complet (354 823 168).
- Loss initiale : $\log 50\,257 \approx 10{,}825$ (pas $\log 88$ du mini-GPT, pour 📈 B3.21).
- Perplexité par caractère : exemple 2,8 nats/token, 4 caractères/token → 0,7 nat/caractère, ≈ 1,01 bit, $e^{0,7} \approx 2{,}01$, $e^{2,8} \approx 16{,}4$.
- Échantillonnage : logits (2, 1, 0,5, 0, −1), T = 0,8, top-k = 3 → (0,6945 ; 0,1990 ; 0,1065 ; 0 ; 0).
- KV cache sur une couche ($d$ = 16, 10 tokens) ; Llama 3.1 8B (16 Gio, 128 Kio/token, 64 Gio sans GQA).
- Coût : $T = 12d$, 49 152 pour $d$ = 4 096 ; 8 Gio / 2 Tio ; FlashAttention : softmax en ligne sur 1 000 clés par blocs de 128 ; chiffres de l'article (15 % BERT-large, ×3 GPT-2 à 1 024 tokens).
- $6ND$ : Llama 3 405B ; Chinchilla (70 Md, 1 400 Md tokens, ≈ 20 tokens/paramètre, budget 5,76·10²³).
- ViT : CIFAR 32 × 32, $P$ = 4 → 64 patchs de 48 valeurs, $d$ = 192 ; ViT-B/16 : 224 px → 196 patchs. 📦 B3.28 (Fashion-MNIST, 16 patchs de 7 × 7) n'est pas traité.

## 4. Ce que la fiche ne dit pas (réservé aux exercices)

- Aucune réponse de ✏️ B3.1 à B3.5, ∂ B3.6-B3.7, 🧮 B3.8 : ni le compte de GPT-2 small, ni ses 124 M décomposés, ni les FLOPs de GPT-3 ou du mini-GPT, ni le KV cache du mini-GPT, ni le nombre exact de paramètres du mini-GPT.
- 🔮 B3.13 : la fiche explique la saturation de la softmax **en général** (variance $d_k$) et ne montre **aucune** carte où une clé copie la requête, ni l'effet chiffré du retrait de $\sqrt{d_k}$.
- 🔮 B3.22 : rien sur ce que fait un modèle de Holmes amorcé en français.
- 📈 B3.21 et B3.24 : aucune courbe, aucune carte apprise ; la fiche ne décrit pas les motifs typiques des têtes (token précédent, ponctuation, « puits d'attention »).
- 🔬 B3.25 : aucun chiffre d'ablation. ⚠️ La fiche dit (203.5) que le masque causal casse un peu la symétrie et cite Haviv et coll. (2022) : des modèles causaux sans encodage de position apprennent quand même une information de position. Le 🔬 doit donc mesurer **combien** on perd, pas demander « perd-on quelque chose ? ». Elle dit aussi (203.6, Xiong et coll.) que le post-LN réclame un warmup soigné : l'issue « post-LN instable sans warmup » est une conclusion attendue, pas une découverte.
- 🔬 B3.26 : la conversion loss/token → loss/caractère est donnée avec des nombres fictifs ; aucune mesure sur Holmes.
- 🔬 B3.27 et 📦 B3.23 : aucune mesure de temps ni de gain du cache ; seulement les ordres de grandeur ($n^2$, $n^3$ sans cache ; $t$ par token avec).
- 📦 B3.18-B3.19 : la fiche ne donne ni le module `CausalSelfAttention` ni le modèle ; seulement `get_batch`, la loss aplatie, l'initialisation et la recette de nanoGPT (init 0,02, $1/\sqrt{2 n_{\text{blocs}}}$, weight decay sur les paramètres à deux axes, $\beta_2$ = 0,99).
- 🛠️ B3.29 : l'invariant (« changer un token futur ne change aucune sortie passée ») est énoncé trois fois ; le test pytest n'est pas écrit.
- ⚖️ B3.10 : seulement Carlini et coll. (USENIX Security 2021 : des centaines de passages extraits de GPT-2, dont des noms, numéros de téléphone, adresses électroniques). Le résumé ajoute deux faits non repris par la fiche (une seule occurrence suffit ; les grands modèles sont plus vulnérables) : utilisables par le ⚖️.
- 📄 B3.11 : la fiche cite l'article (2017, NeurIPS, $d$ = 512, $H$ = 8, $d_h$ = 64, ReLU dans le MLP, post-LN, positions sinusoïdales, « presque identiques » aux positions apprises, argument de la variance) mais ne donne ni les BLEU (28,4 et 41,8), ni les durées (12 h base sur 8 P100, 3,5 jours big), ni le tableau 1 de complexité, ni les tailles 65 M et 213 M (tableau 3), ni le warmup de 4 000 pas, ni le label smoothing 0,1, ni le partage des poids d'embedding multipliés par $\sqrt{d_{model}}$.

## 5. Points d'attention pour la suite

- **Notations (📐)** : $L$ = nombre de requêtes (203.2-203.4), loss (203.8), nombre de couches (formules de coût 203.9-203.10) ; dans $6ND$, $N$ = paramètres, $D$ = tokens. Les énoncés doivent dire laquelle.
- Termes nouveaux (à ajouter au glossaire, ch. B3) : requête, clé, valeur (*query, key, value*) ; attention par produit scalaire mis à l'échelle ; carte d'attention ; attention additive ; alignement (au sens de la traduction, distinct de l'alignement RLHF du glossaire) ; vecteur de contexte ; auto-attention (*self-attention*) ; attention croisée (*cross-attention*) ; masque causal ; masque de padding ; tête d'attention (distincte de « tête (d'un modèle) ») ; attention multi-têtes (MHA) ; projection de sortie ; MQA, GQA ; encodage de position, positions apprises ; RoPE ; équivariance par permutation ; pré-LN, post-LN ; modèle de langue causal (auto-régressif), modèle de langue masqué (MLM) ; corruption de segments ; teacher forcing (*forçage par l'enseignant*) ; biais d'exposition ; KV cache ; prefill ; softmax en ligne ; FlashAttention ; fenêtre glissante ; mélange d'experts (MoE) ; QK-Norm ; lois d'échelle ; PagedAttention ; décodage spéculatif ; YaRN ; ALiBi ; patch (ViT), token `[CLS]` ; biais inductif.
- Formulaire : les neuf formules du contrat sont dans la fiche (attention avec $\mathbf{M}$, $\operatorname{Var} = d_k$, MHA, PE, RoPE, bloc pré-LN, $12d^2$, $6ND$ et KV cache, $L$ et perplexité), plus post-LN, $4d^2 + 4d$, la part de l'attention $T/(12d)$.
- Encadrés 🕰️ (5, ceux du contrat) : RNN contre Transformers (203.1), `F.scaled_dot_product_attention` (203.3), `nn.MultiheadAttention` (203.3), architecture des LLM en 2026 (203.6), inférence des LLM (203.9). Sources vérifiées : arXiv (API export et PDF : 1706.03762, 1409.0473, 2104.09864, 2002.04745, 1810.04805, 2205.14135, 2203.15556, 2010.11929, 2001.08361, 2305.13245, 1911.02150, 2312.00752, 2309.06180, 2211.17192, 2302.01318, 1910.10683, 2002.05202, 2307.08691, 2407.08608, 2108.12409, 2403.19887, 2309.00071, 1508.04025, 1409.3215, 1406.1078, 1904.10509, 2004.05150, 1905.10650, 2203.16634, 2012.07805, 2407.21783, 2505.09388, 2005.14165), PDF de GPT-2 (cdn.openai.com), doc PyTorch 2.11 (SDPA, `nn.MultiheadAttention`, `torch.nn.attention.bias`), README et `model.py` de nanoGPT (licence MIT, « now very old and deprecated », renvoi à nanochat), doc transformers « Cache strategies », PyPI de vLLM (0.32.0 du 18 octobre 2026), annonce IBM Granite 4.0 (2 octobre 2025), arXiv 2504.03624 (Nemotron-H). `git ls-remote` : nanoGPT, nanochat, annotated-transformer joignables ; pages Annotated Transformer, Illustrated Transformer, cours HF (fr, chapitre 1.4) en 200. La vidéo YouTube de Karpathy a répondu 429 (limite de débit) : lien repris du README de nanoGPT.
- `WebFetch` sur arxiv.org/abs a été refusé (permission non accordée à temps) : les vérifications d'articles sont passées par l'API export d'arXiv et les PDF (curl).
- Le contrat de `apply_rope` dit « import path to re-check in transformers 5.17 » : en 5.21, le chemin `transformers.models.llama.modeling_llama` est valable (voir §2).
