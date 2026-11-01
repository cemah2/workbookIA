# B2 — notes de rédaction (session 65, pour le rédacteur des exercices)

Fiche : `01_fiche.md`, 12 534 mots (`split()`), soit ≈ 100 min à 125 mots/min. Figures : `python tools/chapters/figures_chB2.py` (≈ 5 s, hors ligne) ; `--measure` relit les constantes sur le Hub et clone la DUDH dans `data/downloads/udhr` (ignoré par git). Tout a été exécuté avec `/root/venv313` (tokenizers 0.23.3, transformers 5.21.0, torch 2.11.0, NumPy 2.1.3, scikit-learn 1.6.1), le 2026-11-01.

## 1. Valeurs que la fiche utilise déjà (à ne pas reprendre dans un exercice)

| Où | Valeur |
|---|---|
| 202.1, figure `niveaux.png` | « Les chatons s'endorment paisiblement. » : 37 caractères, 7 pré-tokens `Whitespace`, GPT-2 11 tokens, CamemBERT 11 ; fertilité GPT-2 2,75 (4 mots) ; « The kittens fall asleep peacefully. » 6 tokens / 5 mots = 1,2 |
| 202.1 | Holmes : 88 caractères, ≈ 562 000 car., ≈ 106 000 mots, 7 819 mots distincts ; Verne 8 793 ; 13,5 % d'inconnus avec 2 000 mots (chiffres déjà publiés aux ch. 6 et 22) |
| 202.2 | « Ça coûte 15 € » : 13 points de code, 17 octets, NFD 15 points de code ; « ç » U+00E7 → C3 A7 (calcul binaire détaillé), « € » E2 82 AC, « 🙂 » F0 9F 99 82 ; « Straße » → strasse ; NFKC « ﬁn x² » → « fin x2 » ; « û » U+00FB / u + U+0302 |
| 202.3, figure `fusions_bpe.png` | corpus chat × 5, chats × 2, chaton × 3, rat × 4, raton × 2 ; alphabet a c h n o r s t ; fusions a+t (16), c+h (10, **égalité avec h+at**, départage (1, 2) < (2, 8)), ch+at (10), r+at (6), o+n (5), chat+on (3) ; `vocab_size=14` ; encodage de « ratons » → [rat, on, s] pas à pas, « achat » → [a, chat], « chatte » → [chat, t] ; exemple abstrait « abc » avec « b c » rang 0 et « a b » rang 1 |
| 202.4 | GPT-2 sur « Ça coûte 15 € » : ids [127, 229, 64, 763, 42324, 660, 1315, 10432] ; « the » 1169, « Ġthe » 262 ; `<|endoftext|>` 50256 ; `tok.encode("hi<|endoftext|>")` → [5303, 50256] ; pré-tokenisation de « Le chat, l'été ! » (Whitespace et ByteLevel, avec offsets) |
| 202.5 | BERT uncased : « Ça coûte 15 € » → [CLS] ca co ##ute 15 € [SEP] ; « unbelievably » → un ##bel ##ie ##va ##bly ; « café☕ » → [UNK] ; padding avec « a short one » ; BPE octet `tokenizers` vocab 300 sur « Le chat dort… » → ['Ãĩ', 'a', 'ĠcoÃ»te', '<|endoftext|>'] [273, 65, 283, 0] ; piège : `models.BPE()` sans `unk_token` jette « L », « e », la ponctuation (« Le chat, le rat. » → ['chat', 'rat']) |
| 202.5, tableau | tailles `len(tok)` : GPT-2 50 257, BERT uncased 30 522, T5 32 100, CamemBERT 32 006 (config 32 005), XLM-R/e5 250 002, Mistral 7B 32 000, SmolLM2 49 152, Llama 3 128 256, Qwen3 151 669 (table 151 936), gpt-oss 200 019 |
| 202.6, figure `fertilite.png` | DUDH (préambule + 30 articles) : EN 1 681 mots / GPT-2 1 935 tokens (1,15 ; 5,3 o/token) ; FR 1 882 mots / 3 900 tokens (2,07 ; 3,1 o/token) ; rapports à l'anglais pour 15 langues × 4 tokenizers (constantes dans le script) ; Qwen3 fertilités EN 1,12, FR 1,57 → fenêtre 32 768 ≈ 29 000 mots EN / 21 000 FR ; nombres « 1234567 » et « 1234656 » avec GPT-2, Llama 3, Qwen3 |
| 202.6 | exemple de coût : 10 000 tickets × 150 mots, fertilité 1,2 → 1,8 M tokens, 0,18 $ à 18 $ ; FR fertilité 1,5 → 2,25 M (+ 25 %) |
| 202.7 | `nn.Embedding(6, 2)`, ids [4, 1, 4], `manual_seed(0)`, gradient de la somme : lignes 1 → 1, 4 → 2 ; `TinyLM(1000, 64)` lié → 64 000 paramètres ; tableau des tables : MiniLM 11 720 448 (52 %), e5-small 96 014 208 (82 % de 117 653 760), Qwen3-0.6B 155 582 464 (26 % de 596 049 920) |
| 202.8 | PMI sur la matrice 3 × 3 chien/chat/os [[0,4,1],[4,0,1],[1,1,2]] : 0,807 ; −0,357 ; 0,560 ; avec α = 0,75 : p_α 0,351 / 0,297, PPMI(chien, chat) 0,823 ; phrase « le vieux phare éclaire la côte », fenêtre 2 ; loss SGNS : v = (0,5 ; −1 ; 2), u_c = (1 ; 0 ; 0,5), négatifs (0 ; 1 ; 0) et (−1 ; 0,5 ; 1) → scores 1,5, −1, 1, L ≈ 1,828 ; loi des négatifs 100 vs 1 → 0,969 / 0,031 ; sous-échantillonnage f = 1 % → 0,968 |
| 202.9, figure `carte_embeddings.png` | PPMI lissée + SVD (U S^½) sur Holmes : 4 000 mots, fenêtre 2, d = 100, 5 groupes (nombres, famille, corps, moments, couleurs) ; voisins de « father » (sister 0,59, son 0,50, wife 0,47) et de « three » (four 0,67, hundred 0,60, smack 0,53) ; analogie jouet France/Paris/Italie/Rome/Milan/Lyon (requête (1 ; 0 ; 1,1), Rome 0,996, Milan 0,981, Italie 0,803, Paris 0,527, Lyon 0,346, France 0,067) |
| 202.10 | MiniLM (révision `1110a243…`) sur 5 phrases anglaises, requête « I can't log in to my account » : 0,502 / 0,293 / 0,042 ; e5-small (révision `614241f6…`) sur leur traduction française : cosinus 0,76 à 0,86, l'ordinateur (0,864) devant le mot de passe (0,844) ; « glace » comme mot polysémique (le quiz Q11 garde « avocat ») |

## 2. Ce que la fiche ne dit pas (laissé aux exercices)

- **✏️ B2.1** : ni « Noël à Paris », ni « é », ni « ë » (la fiche prend « ç », « û », « € », « 🙂 ») ; **🧠 Q2** : la fiche ne compte pas « é ».
- **✏️ B2.2** : pas de roi/homme/femme (la fiche cite Madrid − Espagne + France de Mikolov et un jouet de capitales en dimension 3).
- **✏️ B2.3, B2.4** : autre corpus que « chat/rat » et que « low lower lowest newer newest » (exemple des docstrings des stubs, déjà public).
- **✏️ B2.5** : la fiche montre V = 6, d = 2, ids [4, 1, 4] ; ch. 24 avait V = 4, d = 2.
- **✏️ B2.6** : la fiche donne une matrice 3 × 3 toute faite, ne construit pas de matrice avec une fenêtre de 1.
- **∂ B2.7** : aucun gradient de la loss SGNS n'est écrit (seulement la loss et sa valeur sur un exemple).
- **🧮 B2.9** : la fiche ne donne **pas** d = 768 pour GPT-2, ni la taille de sa table, ni de LLM à 128 000 tokens (le tableau de 202.7 prend MiniLM, e5-small, Qwen3-0.6B). Elle donne V = 50 257 pour GPT-2 (tableau 202.5).
- **📦 B2.13 / 🔮 B2.14 / 🔬 B2.21** : aucun compte de tokens sur Holmes ou Verne. ⚠️ La fiche montre en revanche que le **français coûte ≈ 2 fois l'anglais avec GPT-2** sur la DUDH (et 1,3 à 1,6 avec les autres) : le 🔮 B2.14 ne doit pas reposer sur le seul sens de l'écart Holmes/Verne ; ce que la fiche **ne** dit pas : le coût des **nombres** et des **emoji** rapporté à un texte (elle dit seulement qu'un emoji fait 4 octets, que GPT-2 découpe les nombres au gré de ses fusions et que Qwen3/SmolLM2 les isolent).
- **🔬 B2.20** : aucune courbe compression / taille du vocabulaire, aucune entropie par token.
- **📦 B2.24** : pas de voisins de « Holmes », « Watson », « letter » (exclus de la carte).
- **🔮 B2.26** : rien sur Fogg ni Passepartout ; **📈 B2.27** : la fiche ne parle pas de l'effet de la fréquence sur une carte.
- **🔬 B2.29** : concept seulement, aucun cosinus contextuel mesuré.
- **🏆 B2.31** : aucune compression de Verne avec 2 000 tokens.
- **🧠 Q6** : la fiche donne les ids 1169 et 262 de « the »/« Ġthe » ; **Q12** : l'identité ‖a − b‖² = 2 − 2 cos est écrite (déjà en B1).

## 3. Chiffres vérifiés, utiles aux exercices (pas dans la fiche)

Extraits `H[100000:150000]` et `V[100000:150000]` (50 000 caractères) de `load_holmes()` et `load_verne()`, sans tokens spéciaux, mots = `split()` :

| Tokenizer | Holmes : tokens / fertilité / octets par token | Verne : tokens / fertilité / octets par token |
|---|---|---|
| GPT-2 | 13 419 / 1,453 / 3,80 | 19 132 / 2,407 / 2,69 |
| BERT uncased | 11 781 / 1,276 / 4,33 | 16 944 / 2,132 / 3,04 |
| e5 (XLM-R) | 12 699 / 1,375 / 4,01 | 14 287 / 1,798 / 3,60 |
| CamemBERT | 16 806 / 1,820 / 3,03 | 12 741 / 1,603 / 4,04 |
| Qwen3 | 12 151 / 1,316 / 4,20 | 15 458 / 1,945 / 3,33 |
| SmolLM2 | 12 823 / 1,388 / 3,98 | 18 989 / 2,389 / 2,71 |

(9 236 mots dans l'extrait de Holmes, 7 948 dans celui de Verne.) Nombres : `" ".join(str(i*7919 % 100000) for i in range(2000))` → GPT-2 2,56 octets/token, Qwen3 et SmolLM2 1,00 (chiffre par chiffre). Emoji : `"🙂🚀🎉👍🏽🐍❤️" * 200` (6 000 octets) → GPT-2 3 800 tokens (1,58 o/token), SmolLM2 4 200, Qwen3 et e5 1 600 ; **BERT uncased : un seul `[UNK]`** pour toute la chaîne (aucune espace), CamemBERT 2 tokens.

Autres faits vérifiés :
- **Départage de `BpeTrainer`** (tag v0.23.3, commit 3012770) : `impl Ord for Merge` (trainer.rs l. 30-38) compare `count`, puis `other.pair.cmp(&self.pair)` (« ascending order ») : la plus petite paire d'ids gagne ; `compute_alphabet` trie par `*k.0 as u32` (point de code) ; `add_special_tokens` passe avant l'alphabet (ids 0…). La boucle s'arrête si `word_to_id.len() >= vocab_size` ou `min_frequency > top.count`. Le contrat cite v0.23.2 : le fichier est identique sur ce point.
- **Modèles d'embedding** : all-MiniLM-L6-v2 (révision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), Apache 2.0, 22 713 216 paramètres F32, 384 dimensions, 6 couches, anglais, tronque à 256 word pieces (fiche) alors que `tok.model_max_length` = 512 ; multilingual-e5-small (révision `614241f622f53c4eeff9890bdc4f31cfecc418b3`), MIT, 117 653 760 paramètres, 12 couches, 384 dimensions, `max_length=512`, préfixes « query: » / « passage: » obligatoires (« query: » pour les tâches hors recherche) ; table 250 037 × 384 alors que le tokenizer a 250 002 tokens.
- **Licences et accès** : `meta-llama/Meta-Llama-3-8B` et `Llama-3.1-8B` gated (manual), `google/gemma-3-270m` gated ; chiffres de Llama 3 lus dans `unsloth/Llama-3.2-1B` (128 000 + 256, 280 147 fusions) et confirmés par le billet Hugging Face (128 256). GPT-2 MIT, BERT Apache 2.0, CamemBERT MIT, Qwen3 Apache 2.0, gpt-oss Apache 2.0, Mistral 7B Apache 2.0.
- **transformers 5.21** : classes `GPT2Tokenizer`, `BertTokenizer`, `Qwen2Tokenizer`, `CamembertTokenizer`, `T5Tokenizer`, `TokenizersBackend` (Llama 3, gpt-oss) ; `PreTrainedTokenizerFast(tokenizer_object=…)` existe encore (rend un `TokenizersBackend`).
- **Pré-tokenizers** : Llama 3 `\p{N}{1,3}`, Qwen3 `\p{N}` (un chiffre), SmolLM2 `Digits(individual_digits=True)`, Qwen normaliseur NFC, Mistral `byte_fallback: true`.
- **Prix** (fin octobre 2026, deux comparateurs ; les pages officielles d'OpenAI et d'Anthropic n'ont pas pu être lues) : entrée 0,05 $ à 30 $ / M tokens, sortie 0,40 $ à 180 $ / M, rapport sortie/entrée 3 à 8, la plupart 0,10 $ à 10 $ en entrée. Sources : benchlm.ai (données du 2026-10-29), geotoolbox.ai (mise à jour du 2026-10-28). Pour 🧮 B2.8, donner des fourchettes datées dans l'énoncé.
- **Articles vérifiés en ligne** : Sennrich et coll. (arXiv 1508.07909, ACL 2016 : +1,1 et +1,3 BLEU EN→DE, EN→RU WMT15 ; 59 500 fusions indépendantes, 89 500 jointes ; `</w>`) ; Mikolov et coll. 2013 (arXiv 1310.4546 : k 5-20 petits corpus, 2-5 grands ; U(w)^{3/4}/Z ; sous-échantillonnage 1 − √(t/f), t ≈ 10⁻⁵ ; 1 milliard de mots, 300 dimensions ; Madrid − Spain + France ≈ Paris) ; GloVe (EMNLP 2014 : x_max = 100, α = 3/4, 1/d, W + W̃) ; Kudo 2018 Unigram (ACL, η = 80 %, caractères toujours gardés, Viterbi) ; Kudo et Richardson 2018 (arXiv 1808.06226, ▁ U+2581, BPE et Unigram, texte brut) ; Petrov et coll. 2023 (NeurIPS : jusqu'à 15 fois, > 4 fois au niveau caractère/octet ; FLORES-200 ; premium FR GPT-2 2,00, DE 2,14) ; Bolukbasi et coll. 2016 (NeurIPS : w2vNEWS 300 d, Google News, 26 377 mots retenus après filtrage des 50 000 plus fréquents ; direction de genre par PCA sur 10 paires ; hard debiasing neutralize + equalize, soft) ; Reimers et Gurevych 2019 (arXiv 1908.10084 : 10 000 phrases, ≈ 50 M d'inférences, ≈ 65 h contre ≈ 5 s) ; Levy, Goldberg et Dagan 2015 (TACL : cds α = 0,75, eig p = 0 ou 0,5, SGNS = PMI − log k) ; Gonen et Goldberg 2019 (NAACL, « Lipstick on a Pig », titre vérifié). Non relus en ligne (cités de mémoire, classiques) : Harris 1954, Firth 1957, Press et Wolf 2017, Schuster et Nakajima 2012, Gage 1994 (cité par Sennrich).
- **Le livre** : les 🕰️ ne citent que ce que les fiches des ch. 22 et 24 ont déjà vérifié (89 caractères, 8 000 mots, word2vec p. 1077).

## 4. Pour le coordinateur

- Termes nouveaux à ajouter au glossaire (ch. B2) : point de code, UTF-8, caractère combinant, normalisation NFC / NFD / NFKC, pré-tokenizer (*pre-tokenizer*), fusion (BPE) et rang, BPE au niveau octet (*byte-level BPE*), token spécial (BOS, EOS, PAD, UNK, CLS, SEP, MASK), offsets (*décalages*), WordPiece, Unigram, SentencePiece, repli sur les octets (*byte fallback*), fertilité, compression (octets par token, ≠ taux de compression du ch. 6), surcoût de tokenisation (*tokenization premium*), masque d'attention, poids liés (*weight tying*), hypothèse distributionnelle, matrice de co-occurrences, PMI, PPMI, lissage des contextes (α = 0,75), SVD tronquée, word2vec, skip-gram, CBOW, échantillonnage négatif (*negative sampling*), sous-échantillonnage des mots fréquents, GloVe, 3CosAdd, embedding statique / contextuel, mean pooling (*moyenne des tokens*), embedding de phrase, siamois (*siamese*), recherche sémantique, cross-encoder.
- Formulaire : les 7 formules du contrat (fertilité et compression, fusion BPE, one-hot × E, PMI/PPMI, loss SGNS, cosinus et 3CosAdd, mean pooling) + score de WordPiece et p_α(c).
- Le contrat cite tokenizers v0.23.2 pour le départage : vérifié sur v0.23.3 (la version figée), même code.
- La ligne « Lecture du parcours rapide » n'existe pas pour B2 (pas de section du livre) : les dix sections sont marquées ⏩, comme en B1.
