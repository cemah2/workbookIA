# B4 · LLM en pratique : Hugging Face, prompting, RAG et LoRA — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels et des exercices ✏️ B4.1 à B4.5 se vérifient aussi dans la partie 0 du notebook, où elles sont recalculées avec ta librairie (version de référence), scikit-learn, transformers et peft ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B4.Q1 — Modèle de base, modèle instruct, modèle de chat : lequel pour quoi ?
a) **B** : prolonger un texte brut, c'est l'objectif même du pré-entraînement ; un modèle de chat (A) attend des tours de conversation, et un modèle instruct continue aussi un texte, mais il a été déplacé vers les consignes (C est faux : « le seul » ne tient pas). b) **Vrai** : des milliers de milliards de tokens (2 000 milliards pour SmolLM2-135M) contre des milliers ou des millions de paires pour le SFT. c) **C** : `SmolLM2-360M` n'a pas de suffixe d'instruction ; `Llama-3.2-1B-Instruct` est instruct, et `Qwen3-0.6B` aussi, puisque Qwen3 marque le modèle de base par `-Base`. d) **C** : le SFT sur des paires consigne-réponse, puis souvent un alignement (RLHF, DPO) ; un pré-entraînement plus long (A) donne un meilleur modèle de base, et un gabarit (B) ne fait que mettre en forme. e) **Faux** : un modèle de base continue le texte ; une question peut être suivie, dans ses données, de dix autres questions d'un questionnaire.
**À retenir** : base = continuer du texte ; instruct = suivre des consignes (SFT, puis alignement) ; chat = instruct entraîné sur des conversations à rôles. Le nom du dépôt le dit souvent, pas toujours.

### B4.Q2 — « Open weights » n'est pas « open source » : lire la licence d'un modèle
a) **Vrai** : Apache 2.0 permet l'usage commercial, la modification et la redistribution, en gardant l'avis de licence et en signalant les modifications. b) **B** : la *Llama 3.2 Community License* demande que le nom d'un modèle dérivé redistribué commence par « Llama », et la mention « Built with Llama » sur le produit ; elle n'impose pas Apache 2.0 (A), et modifier les poids ne libère pas de la licence (C). c) **Faux** : l'*Open Source AI Definition* 1.0 demande aussi une description des données qui permette de reconstruire un système équivalent et le code complet d'entraînement. d) **C** : un compte, l'acceptation des conditions, puis ta clé d'accès personnelle (`token=`) ; un miroir non autorisé (B) contourne les conditions de l'éditeur. e) **A** : les données et le code d'entraînement ; le code d'inférence est dans transformers, la configuration dans `config.json`.
**À retenir** : « open weights » = poids téléchargeables sous la licence de l'éditeur ; lis-la avant la première ligne de code et note la révision, la licence et ses obligations.

### B4.Q3 — dtype, device et mémoire : ce qui décide si un modèle tient sur ton ordinateur
a) **6,0** Go $= 3 \times 10^9 \times 2$ octets. b) **1,5** Go $= 3 \times 10^9 \times 0{,}5$ octet, sans les échelles des groupes. c) **Vrai** : 8 bits d'exposant dans les deux formats, mais 7 bits de mantisse au lieu de 23. d) **C** : $T$, le nombre de tokens, est le seul terme de $2 \times L \times T \times n_{kv} d_h \times$ octets qui grandit pendant la conversation. e) **Faux** : la fiche mesure 7,6 puis 13,9 s en bfloat16 contre 8,4 puis 8,7 s en float32 sur le CPU du workbook, qui n'a pas d'instructions dédiées au bfloat16 : la mémoire est divisée par deux, pas forcément le temps.
**À retenir** : poids $\approx N \times$ octets par paramètre ; s'y ajoute un KV cache qui croît avec la conversation ; un format plus petit économise d'abord la mémoire.

### B4.Q4 — Chat template : pourquoi ne pas écrire soi-même « User: … Assistant: … »
a) **A** : le gabarit Jinja est rangé avec le tokenizer (`tok.chat_template`, dans `tokenizer_config.json` ou un fichier à part). b) **Vrai** : il ajoute `<|im_start|>assistant\n` pour un modèle ChatML ; le modèle sait que c'est à lui d'écrire. c) **C** : le tokenizer ne réécrit rien (A), et le modèle a appris des tokens spéciaux, pas des noms de rôle en clair (B) ; il répond moins bien, ou écrit lui-même la suite du dialogue. d) **Faux** : Llama 3 a ses propres tokens d'en-tête (`<|start_header_id|>`…), Qwen2.5 utilise ChatML. e) **B** : `add_special_tokens=False` ; `skip_special_tokens` (A) sert au décodage.
**À retenir** : le chat template du tokenizer est la seule source fiable du format ; affiche la chaîne produite (`tokenize=False`) avant de lui faire confiance, et ne tokenise jamais deux fois ses tokens spéciaux.

### B4.Q5 — Glouton, température, top-p : lequel pour extraire une date, lequel pour écrire un poème ?
a) **C** : le décodage glouton rend toujours la même sortie pour le même prompt. b) **A** : il faut des sorties différentes, donc un tirage, une température modérée et dix séquences ; le glouton (B) ne donne qu'une réponse, et `temperature=0` (C) lève une `ValueError` dès que le tirage est activé (par `do_sample=True` ou par le `generation_config.json` du modèle) ; sans tirage, les dix appels gloutons rendent dix fois le même poème. c) **Faux** : ce qui n'est pas passé vient de `generation_config.json`, souvent un tirage (`do_sample: true` pour Qwen2.5-0.5B-Instruct). d) **Vrai** : diviser les logits par 0,5 double leurs écarts ; pour deux logits $(1, 0)$, le premier token passe de $0{,}731$ à $0{,}881$. e) **B** : le plus petit ensemble des tokens les plus probables dont les probabilités cumulées atteignent 0,8 ; ce n'est ni une part du vocabulaire (A) ni un seuil par token (C).
**À retenir** : glouton pour la reproductibilité (extraction, évaluation) ; tirage avec température et top-p pour la variété ; affiche toujours `model.generation_config`.

### B4.Q6 — Zéro-shot, few-shot, chain-of-thought : vrai ou faux
a) **Faux** : l'apprentissage en contexte ne touche pas aux poids ; les exemples ne vivent que dans le prompt. b) **Vrai** : Lu et coll. (ACL 2022) montrent qu'un autre ordre des mêmes exemples peut faire passer un modèle de presque l'état de l'art au hasard. c) **Vrai** : chaque exemple devient deux tours, sa question en `user` et sa réponse en `assistant` ; trois exemples font six messages, puis la vraie question en `user`. Tout coller dans un seul message `user` marche souvent moins bien : le modèle n'y reconnaît pas ses propres réponses. d) **Faux** : l'effet n'apparaissait que pour de **grands** modèles (de l'ordre de 100 milliards de paramètres dans l'article de 2022) ; un petit modèle écrit souvent des étapes fausses. e) **Faux** : le raisonnement écrit est une explication plausible, pas forcément le calcul qui a produit la réponse.
**À retenir** : zéro-shot et few-shot changent le prompt, pas le modèle ; le format et l'ordre des exemples comptent ; le chain-of-thought aide surtout les grands modèles, et ne prouve rien.

### B4.Q7 — Pourquoi un LLM invente : hallucination et token le plus probable
a) **A** : une distribution sur le token suivant ; une réponse entière est une boucle de choix ou de tirages, sans base de faits (B) ni vérification (C). b) **Faux** : une température basse rend les réponses moins variées, pas plus vraies ; le glouton peut répéter la même invention à chaque fois. c) **C** : la question repose sur un fait faux (Holmes est un personnage de fiction, sans prix Nobel) ; une année plausible (A, B) est une hallucination. d) **Vrai** : le modèle peut encore mal lire ses passages, ou citer au hasard ; on réduit et on mesure. e) **Faux** : le pré-entraînement récompense la prédiction du texte, presque jamais l'abstention ; c'est le fine-tuning et la consigne qui la demandent.
**À retenir** : un LLM produit le plausible ; contre l'invention, on ancre (RAG), on cite et vérifie, on autorise « je ne sais pas », et l'on mesure sur des questions pièges.

### B4.Q8 — Évaluer un LLM : exact match, juge LLM, contamination des benchmarks
a) **1** : « baker street. » devient « baker street », égal à la référence normalisée. b) **1** : « baker street » apparaît dans « he lives in baker street london » ; l'exact match, lui, vaudrait 0. c) **0,040** $= \sqrt{0{,}8 \times 0{,}2 / 100} = \sqrt{0{,}0016}$ : un intervalle à 95 % d'environ $\pm 8$ points. d) **C** : présenter les deux réponses dans les deux ordres (et ne compter une préférence que si elle tient dans les deux) ; un petit juge (A) est moins fiable, et l'échelle de la note (B) ne change pas le biais de position. e) **Vrai** : c'est l'un des indices de la fiche, avec un modèle qui complète mot pour mot une question du test.
**À retenir** : on fixe la normalisation et la métrique avant de voir les résultats ; sur 100 questions, quelques points d'écart ne départagent rien ; un juge LLM a ses biais (position, longueur, auto-complaisance).

### B4.Q9 — BM25 ou recherche dense : qui retrouve « Watson » et qui retrouve « le docteur » ?
a) **B** : seul le passage 0 contient « watson » ; BM25 ne connaît aucun synonyme, l'idf ne fait que pondérer les mots communs. b) **Vrai** : la somme ne porte que sur les termes de la requête présents dans le document ; un terme absent ($f_{t,d} = 0$) n'apporte rien, quels que soient $k_1$ et $b$ (une implémentation saute ces termes, sinon $k_1 = 0$ donnerait $0/0$). c) **B** : un modèle d'embedding place près l'une de l'autre des expressions employées dans des contextes voisins ; il ne compte pas les lettres (C) et n'utilise pas BM25 (A). d) **A** : un code rare cité tel quel est le point fort de BM25 ; un embedding le connaît mal. e) **0,0317** $= 2/63$ : $1/(60 + 3)$ dans chacun des deux classements.
**À retenir** : BM25 est exact sur les mots, noms propres et codes ; la recherche dense rapproche les paraphrases ; la RRF combine les deux par les rangs.

### B4.Q10 — RAG ou fine-tuning pour ajouter des connaissances ?
a) **B** : une documentation qui change se lit par une recherche, mise à jour sans réentraîner, et chaque réponse cite sa source ; un fine-tuning chaque semaine (A, C) coûte cher et apprend mal des faits précis. b) **B** : un format est un comportement ; on commence par un prompt avec un exemple de format et une validation, puis un fine-tuning si cela ne suffit pas. c) **Vrai** : le filtre se fait dans la recherche, selon les droits de l'utilisateur ; un modèle fine-tuné contient tout ce qu'il a appris pour tout le monde. d) **Faux** : quelques exemples apprennent mal des faits ; le modèle les mélange, ou les invente. e) **A** : réindexer les documents modifiés (découpage, statistiques de BM25, embeddings) ; le générateur ne change pas.
**À retenir** : connaissances qui changent → RAG ; comportement (format, style, tâche) → prompt, puis fine-tuning ; les deux se combinent.

### B4.Q11 — LoRA : ce qui est gelé, ce qui est appris, ce que l'on fusionne
a) **B** : $\mathbf{W}$ est gelée, sans gradient ni état d'Adam ; elle n'est ni entraînée lentement (A) ni remplacée (C). b) **Vrai** : $\mathbf{B}_0 = 0$, donc $\mathbf{B}_0\mathbf{A}_0 = 0$ et $\mathbf{W}' = \mathbf{W}$. c) **4 096** $= 4 \times (512 + 512)$, 1,6 % des 262 144 poids. d) **C** : `merge_and_unload()` calcule $\mathbf{W} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$ dans chaque couche, retire les branches (A) et rend un modèle ordinaire, avec ses poids d'origine modifiés (B est faux). e) **Faux** : $\alpha/r = 8/16 = 0{,}5$ ; la branche est divisée par 2.
**À retenir** : LoRA gèle $\mathbf{W}$, apprend $\mathbf{A}$ et $\mathbf{B}$ ($r(d + k)$ paramètres), part de $\mathbf{B} = 0$ et se fusionne sans surcoût.

### B4.Q12 — Injection de prompt : quand un document récupéré devient une instruction
a) **B** : les instructions arrivent par un document que le système récupère ; l'utilisateur qui écrit lui-même (A) fait une injection directe. b) **Faux** : les parades réduisent le risque sans l'annuler ; une consigne n'est qu'un texte de plus dans le prompt, qu'une instruction mieux placée ou mieux déguisée (le faux message système de la démonstration de la fiche) peut emporter. c) **C** : si le modèle ne peut ni envoyer, ni acheter, ni écrire sans confirmation humaine, une injection réussie fait peu de dégâts ; la température (A) ou la taille du modèle (B) ne l'empêchent pas d'obéir. d) **Vrai** : c'est l'injection indirecte : l'attaquant écrit l'e-mail, l'assistant le lit. e) **A** : LLM01, *Prompt Injection* ; l'agentivité excessive est LLM03, la désinformation LLM07.
**À retenir** : pour un LLM, consignes et données sont le même texte ; on superpose les parades (séparation, filtrage, validation des sorties, droits minimaux) et l'on mesure le taux de réussite des attaques.

<a id="rappels"></a>

## 🔁 Rappels

### B4.R1 — Ch. B3 : ce que le masque causal et le KV cache imposent à la génération
a) **100** : une passe sur les 400 tokens du prompt (le prefill), qui donne le premier token, puis 99 passes d'un seul token chacune ; sans cache, chaque passe relirait tout le texte. b) **Faux** : le token 2 dépend du token 1, qui n'existe pas encore ; seul l'entraînement (teacher forcing) calcule toutes les positions en parallèle. c) **12 288** $= 2 \times 24 \times (2 \times 64) \times 2$ octets : le KV cache de Qwen2.5-0.5B, grâce à ses 2 têtes de clés et valeurs. d) **C** : une passe avant par token généré. e) **Vrai** : le masque causal laisse la position 50 voir les positions 0 à 49, jamais la 51.
**À retenir** : générer, c'est une passe par token ; le KV cache évite de relire le passé, et sa mémoire croît avec le nombre de tokens.

### B4.R2 — Ch. B1 : extraction de features ou fine-tuning : que change le nombre d'exemples ?
a) **B** : avec peu d'exemples, la tête sur un modèle gelé battait souvent le fine-tuning complet, victime d'overfitting et d'oubli. b) **Vrai** : le modèle pré-entraîné est gelé ; seule la tête s'entraîne. c) **A** : LoRA place de petites matrices **dans** les couches, le reste gelé ; il n'entraîne pas que la tête (B), ni tous les poids (C). d) **Faux** : 20 exemples par classe pour des millions de paramètres libres : overfitting et oubli ; l'extraction de features ou LoRA sont plus sûrs. e) **A** : un fine-tuning qui fait perdre au modèle ce qu'il savait faire avant (McCloskey et Cohen, 1989).
**À retenir** : moins on a d'exemples, moins on libère de paramètres ; LoRA est un compromis entre la tête seule et le fine-tuning complet.

### B4.R3 — Ch. 13 : les k plus proches voisins et le choix de la distance
a) **1** : $\mathbf{b} = 2\mathbf{a}$, même direction ; $50 / (5 \times 10) = 1$. b) **5** $= \sqrt{3^2 + 4^2}$ : même direction, mais pas le même point. c) **0,36** $= 2 - 2 \times 0{,}82$. d) **Vrai** : pour des vecteurs de norme 1, $\lVert\mathbf{a} - \mathbf{b}\rVert^2 = 2 - 2\cos$, une fonction strictement décroissante du cosinus : même classement, pour toute requête. e) **38 400 000** $= 50\,000 \times 768$.
**À retenir** : le cosinus ignore la longueur des vecteurs ; une fois normalisés, cosinus et distance euclidienne donnent les mêmes voisins.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B4.1 — Mémoire d'un modèle : paramètres, dtype et KV cache — tient-il sur ton ordinateur ? ✏️
**Réponse** : a) **28 311 552** ; b) **21,0** % ; c) **538,1** Mo ; d) **269,0** Mo ; e) **67,3** Mo ; f) **23 040** octets ; g) **188,7** Mo ; h) **Faux** ; i) **9** ; j) **28** tokens par seconde ; k) **Vrai**.
**Démarche** : a) Une ligne de $d = 576$ nombres par token : $49\,152 \times 576 = 28\,311\,552$, comptés une seule fois puisque la couche de sortie la réutilise. b) $28\,311\,552 / 134\,515\,008 \approx 21{,}05\,\%$, soit 21,0 % : un cinquième d'un petit modèle est sa table d'embedding (B2). c) $4 \times 134\,515\,008 = 538\,060\,032$ octets. d) $2N = 269\,030\,016$ octets. e) $0{,}5N = 67\,257\,504$ octets, plus les échelles des groupes en pratique. f) Par couche, les clés et les valeurs d'un token font chacune $3 \times 64 = 192$ nombres de 2 octets : $2 \times 30 \times 192 \times 2 = 23\,040$ octets. g) $23\,040 \times 8\,192 = 188\,743\,680$ octets. h) Faux : 188,7 Mo contre 269,0 Mo ; c'est la GQA (3 têtes de clés et valeurs pour 9 de requêtes) qui garde le cache sous les poids. i) $(2 \times 10^9 - 269\,030\,016) / 188\,743\,680 \approx 9{,}17$ : 9 conversations entières. j) Qwen2.5 génère $64 / 8{,}4 \approx 7{,}62$ tokens par seconde avec 1 976 Mo de poids et buffers ; SmolLM2 en float32 en a 538 Mo, 3,67 fois moins : $7{,}62 \times 3{,}67 \approx 28$ tokens par seconde. k) Vrai : 538 Mo. Dans ta copie : sans GQA, 9 têtes de clés et valeurs, le cache triple : 566,2 Mo à contexte plein, plus que les poids. Qwen3-0.6B a 114 688 octets par token, 5 fois plus : ses 8 têtes de clés et valeurs de taille 128 font 1 024 nombres par couche, contre 192, ce qui compense largement ses 28 couches contre 30. L'estimation j) suppose une génération limitée par la lecture des poids ; pour un petit modèle, le surcoût fixe de chaque pas (Python, petits noyaux, échantillonnage) pèse davantage, et le CPU, son cache et le dtype changent tout : tu le mesureras en 📦 B4.12.
**Erreurs fréquentes** : compter deux fois la table des poids liés ; confondre Mo ($10^6$) et Mio ($2^{20}$) ; prendre les 9 têtes de requêtes au lieu des 3 têtes de clés et valeurs ; oublier le facteur 2 des clés et des valeurs ; oublier de retirer les poids de la mémoire disponible en i) ; inverser le rapport des mémoires en j).
**Variante** : le même modèle avec un contexte limité à 2 048 tokens par conversation. Réponse : $23\,040 \times 2\,048 \approx 47{,}2$ Mo par conversation, et $(2 \times 10^9 - 269\,030\,016) / 47\,185\,920 \approx 36{,}7$, soit 36 conversations dans les 2 Go : le contexte servi est le premier levier de la mémoire d'un serveur.

### Ex B4.2 — Recall@k et MRR pour cinq requêtes ✏️
**Réponse** : a) **0,100** ; b) **0,467** ; c) **0,633** ; d) **0,433** ; e) **0,800** ; f) **Faux** ; g) **2** ; h) **Vrai**.
**Démarche** : une ligne par question.

| question | recall@1 | recall@3 | recall@5 | rang du premier pertinent | 1 / rang |
|---|---|---|---|---|---|
| 0 | 0 | 1 | 1 | 2 | 1/2 |
| 1 | 0 | 1/2 | 1 | 3 | 1/3 |
| 2 | 1/2 | 1/2 | 1/2 | 1 | 1 |
| 3 | 0 | 0 | 0 | — | 0 |
| 4 | 0 | 1/3 | 2/3 | 3 | 1/3 |

a) $(0 + 0 + \frac{1}{2} + 0 + 0)/5 = 0{,}1$. b) $(1 + \frac{1}{2} + \frac{1}{2} + 0 + \frac{1}{3})/5 = 2{,}333/5 \approx 0{,}467$. c) $(1 + 1 + \frac{1}{2} + 0 + \frac{2}{3})/5 = 3{,}167/5 \approx 0{,}633$. d) $(\frac{1}{2} + \frac{1}{3} + 1 + 0 + \frac{1}{3})/5 = 2{,}167/5 \approx 0{,}433$. e) Les questions 0, 1, 2 et 4 ont un pertinent dans leur top-3 : $4/5 = 0{,}8$. f) Faux : b) et e) diffèrent ; le recall compte la **part** des pertinents trouvés, le taux de succès seulement la présence d'au moins un. g) Les questions 0 et 1. h) Vrai : pour chaque question, la part de ses pertinents au rang 1 vaut 0 ou $1/|\text{pertinents}| \le 1$, et l'inverse du rang de son premier pertinent vaut 1 dans ce second cas, donc $\text{recall@1} \le \text{MRR}$ question par question, et en moyenne. Dans ta copie : non, le document 4 n'est pas dans les cinq résultats de la question 4 ; avec le document 7 en 2ᵉ position pour la question 3, le MRR passerait à $(\frac{1}{2} + \frac{1}{3} + 1 + \frac{1}{2} + \frac{1}{3})/5 \approx 0{,}533$. On divise par le nombre de pertinents pour savoir quelle **part** de l'information utile atteint le modèle ; diviser par $k$ donne la *precision@k*, qui baisse mécaniquement quand $k$ dépasse le nombre de pertinents.
**Erreurs fréquentes** : diviser par $k$ ; confondre recall@k et taux de succès ; prendre le rang du dernier pertinent, ou sommer $1/\text{rang}$ sur tous les pertinents ; compter les rangs à partir de 0 ; moyenner le MRR sur les seules questions réussies.
**Variante** : la *precision@3* des mêmes résultats. Réponse : $(\frac{1}{3} + \frac{1}{3} + \frac{1}{3} + 0 + \frac{1}{3})/5 \approx 0{,}267$ ; elle juge la part de bruit dans ce que lira le modèle, quand le recall juge ce qui lui manque.

### Ex B4.3 — LoRA : paramètres entraînables d'une couche 896×896 de rang 8, puis de tout un modèle ✏️
**Réponse** : a) **14 336** ; b) **1,79** % ; c) **7,14** % ; d) **45 056** ; e) **1 081 344** ; f) **0,2184** % ; g) **4 399 104** ; h) **Vrai** ; i) **4,3** Mo ; j) **448** ; k) **Faux**.
**Démarche** : a) $r(d + k) = 8 \times (896 + 896) = 14\,336$ : $\mathbf{B}$ de forme $(896, 8)$ et $\mathbf{A}$ de forme $(8, 896)$. b) $14\,336 / 802\,816 \approx 1{,}786\,\%$. c) `k_proj` : $8 \times (128 + 896) = 8\,192$ pour $128 \times 896 = 114\,688$ poids, soit 7,14 %. d) $2 \times 14\,336 + 2 \times 8\,192 = 45\,056$ (q et o de $896 \times 896$, k et v de $128 \times 896$). e) $24 \times 45\,056 = 1\,081\,344$. f) $1\,081\,344 / (494\,032\,768 + 1\,081\,344) = 1\,081\,344 / 495\,114\,112 \approx 0{,}2184\,\%$, ce qu'affiche peft. g) Chaque couche du MLP, $896 \to 4\,864$ ou l'inverse, reçoit $8 \times 5\,760 = 46\,080$ : par couche, $45\,056 + 3 \times 46\,080 = 183\,296$, et $24 \times 183\,296 = 4\,399\,104$ (0,88 % du total). h) Vrai : $r(d + k)$ est proportionnel à $r$. i) $4 \times 1\,081\,344 = 4\,325\,376$ octets. j) $r \times 1\,792 \ge 802\,816$ donne $r \ge 448$, la moitié de 896 : au-delà, la factorisation coûte plus cher que la matrice. k) Faux : LoRA prend 7,14 % de `k_proj` contre 1,79 % de `q_proj` ; une matrice étroite a peu de poids, et la factorisation $r(d + k)$ n'y gagne presque rien. Dans ta copie : 4,3 Mo contre 988 Mo pour le modèle en bfloat16, moins de 0,5 % ; on garde un adaptateur par tâche pour presque rien. GPT-3, le modèle de l'article, a une attention multi-têtes classique : $\mathbf{W}_K$ et $\mathbf{W}_V$ y font $d_{\text{model}} \times d_{\text{model}}$ ; la GQA de Qwen2.5 (2 têtes de clés et valeurs pour 14 de requêtes) les réduit à $128 \times 896$.
**Erreurs fréquentes** : compter $rd$ au lieu de $r(d + k)$ ; prendre $896 \times 896$ pour `k_proj` et `v_proj` ; oublier que « all params » compte les adaptateurs ; compter la tête de sortie dans `"all-linear"` ; croire que `lora_alpha` change le nombre de paramètres.
**Variante** : le réglage de l'article, $r = 8$ sur `q_proj` et `v_proj` seulement. Réponse : $24 \times (14\,336 + 8\,192) = 540\,672$ paramètres, $0{,}1093\,\%$ du total d'après peft.

### Ex B4.4 — Un chat template à la main : rôles, tokens spéciaux et prompt de génération ✏️
**Réponse** : a) **5** ; b) **35** ; c) **25** ; d) **21** ; e) **53** ; f) **B** ; g) **Faux** ; h) **Vrai**.
**Démarche** : chaque message devient `<|im_start|>` rôle `\n` contenu `<|im_end|>` `\n`, soit $4 +$ rôle $+$ contenu tokens : $5 +$ contenu pour `system` et `user`, $6 +$ contenu pour `assistant` (2 tokens). Le prompt de génération `<|im_start|>assistant\n` ajoute $1 + 2 + 1 = 4$ tokens. a) Sans message système, la condition `if loop.first and messages[0]['role'] != 'system'` ajoute le tour système par défaut : `<|im_start|>` et `<|im_end|>` pour lui, deux pour U, un pour le prompt de génération, 5 tokens spéciaux. b) Tour par défaut $5 + 16 = 21$, tour de U $5 + 5 = 10$, prompt de génération 4 : 35. c) Avec S, pas de message par défaut : $(5 + 6) + (5 + 5) + 4 = 25$. d) $25 - 4 = 21$. e) $11 + 10 + (6 + 12) + (5 + 5) + 4 = 53$ : tout l'historique repasse à chaque tour. f) B ; A ajoute le message par défaut alors qu'il y a un message système, C oublie le prompt de génération. g) Faux : `User`, `:`, `\n`, `Ass`, `istant`… sont des tokens ordinaires. h) Vrai : la boucle parcourt tous les messages, réponse précédente comprise. Dans ta copie, la chaîne de c) :
```text
<|im_start|>system
Answer in one short sentence.<|im_end|>
<|im_start|>user
Where does Holmes live?<|im_end|>
<|im_start|>assistant

```
Le modèle écrit juste après le dernier retour à la ligne. Avec un tokenizer qui ajoute un token de début (Llama 3 : `<|begin_of_text|>`) et un gabarit qui l'écrit déjà, `tok(chaîne)` en met **deux**, et le modèle lit un début qu'il n'a jamais vu. Le message système par défaut est un piège de comparaison : SmolLM2 reçoit une identité et 21 tokens que tu n'as pas écrits, Qwen2.5 un autre texte ; deux modèles comparés « avec le même prompt » ne lisent pas le même texte, sauf si tu écris toi-même le message système.
**Erreurs fréquentes** : oublier le message système par défaut ; compter `assistant` pour un token ; oublier les retours à la ligne ; ajouter le message par défaut quand un message système existe ; oublier le prompt de génération.
**Variante** : le deuxième tour sans message système, `[U, A, U2]`. Réponse : le message par défaut n'apparaît qu'une fois, en tête (`loop.first`) : $21 + 10 + 18 + 10 + 4 = 63$ tokens, 9 tokens spéciaux.

### Ex B4.5 — BM25 à la main : trois documents, une requête de deux mots ✏️
**Réponse** : a) **5** ; b) **0,1335** ; c) **0,1628** ; d) **0,1599** ; e) **1,2246** ; f) **[0, 1, 2]** ; g) **Faux** ; h) **1,1144** ; i) **Vrai**.
**Démarche** : $N = 3$, longueurs 4, 3 et 8. a) $\overline{|d|} = 15/3 = 5$. b) « nautilus » est dans les trois documents : $\ln(1 + 0{,}5/3{,}5) = \ln(8/7) \approx 0{,}1335$ ; « nemo », dans un seul : $\ln(1 + 2{,}5/1{,}5) = \ln(8/3) \approx 0{,}9808$. Avec $k_1 = 1{,}5$ et $b = 0{,}75$, le dénominateur vaut $f + 1{,}5\,(0{,}25 + 0{,}75\,|d|/5)$ et le numérateur $2{,}5\,f$. c) Doc 1, $|d|/5 = 0{,}6$ : $1 + 1{,}5 \times 0{,}7 = 2{,}05$, et $0{,}1335 \times 2{,}5/2{,}05 \approx 0{,}1628$. d) Doc 2, $|d|/5 = 1{,}6$, $f = 2$ : $2 + 1{,}5 \times 1{,}45 = 4{,}175$, et $0{,}1335 \times 5/4{,}175 \approx 0{,}1599$. e) Doc 0, $|d|/5 = 0{,}8$ : $1 + 1{,}5 \times 0{,}85 = 2{,}275$, facteur $2{,}5/2{,}275 \approx 1{,}0989$ pour chacun des deux termes : $(0{,}9808 + 0{,}1335) \times 1{,}0989 \approx 1{,}2246$. f) $1{,}2246 > 0{,}1628 > 0{,}1599$. g) Faux : avec $b = 0$, le dénominateur devient $f + 1{,}5$ ; le doc 1 vaut $0{,}1335$ et le doc 2 $0{,}1335 \times 5/3{,}5 \approx 0{,}1908$, qui passe devant. h) Avec $k_1 = 0$, chaque terme présent apporte son idf : $0{,}9808 + 0{,}1335 \approx 1{,}1144$. i) Vrai : $\ln(2{,}5/1{,}5) \approx 0{,}5108$ et $\ln(0{,}5/3{,}5) \approx -1{,}9459$ ; le doc 0 vaudrait $1{,}0989 \times (0{,}5108 - 1{,}9459) \approx -1{,}577$, un score négatif, moins que le 0 d'un document sans aucun mot de la requête : un mot présent partout **pénalise** le document qui le contient. (L'énoncé fixe le corpus aux trois documents : ajouter un quatrième document changerait $N$ et tous les idf.) Dans ta copie : la normalisation de la longueur l'emporte sur la seconde occurrence ; le doc 2 est 1,6 fois plus long que la moyenne, et la saturation ($k_1$) fait qu'une deuxième occurrence ne double pas la contribution. Sans « le », « un » et « au », les longueurs deviennent 3, 2 et 5 ($\overline{|d|} \approx 3{,}33$) : le doc 2 a une longueur relative de 1,5 et vaut $0{,}1335 \times 5/4{,}0625 \approx 0{,}1643$, juste devant le doc 1 (inchangé, 0,1628) ; la tokenisation compte souvent plus que les réglages.
**Erreurs fréquentes** : oublier le « 1 + » de l'idf ; compter les occurrences au lieu des documents dans $n_t$ ; oublier le facteur $k_1 + 1$ ; calculer la longueur moyenne sur la requête ; oublier d'additionner les deux termes ; arrondir l'idf avant de multiplier.
**Variante** : $k_1 = 1{,}2$, la valeur par défaut de Lucene, avec $b = 0{,}75$. Réponse : $(1{,}2137 ;\ 0{,}1597 ;\ 0{,}1571)$, même classement : sur un si petit corpus, $k_1$ déplace les scores sans changer l'ordre.

### Ex B4.6 — LoRA : ΔW = BA de rang au plus r, B nulle au départ, gradients de A et de B, facteur α/r ∂
**Réponse** : $\operatorname{rang}(\mathbf{B}\mathbf{A}) \le r$ ; $\frac{\partial L}{\partial \mathbf{B}} = s\,\mathbf{g}(\mathbf{A}\mathbf{x})^\top$, $\frac{\partial L}{\partial \mathbf{A}} = s\,\mathbf{B}^\top\mathbf{g}\mathbf{x}^\top$ ; au premier pas, seul $\mathbf{B}$ bouge et $\Delta\mathbf{W}_1 = -\eta s^2\,\mathbf{g}\mathbf{x}^\top\mathbf{A}_0^\top\mathbf{A}_0$, de rang 1 ; application : $\Delta\mathbf{W}_1\mathbf{x} = (-0{,}4 ;\ 0 ;\ 0{,}8) = -0{,}4\,\mathbf{g}$ et, au deuxième pas, $\partial L/\partial \mathbf{A} = (-4, -2)$.
**Démarche** :
1. La colonne $j$ de $\mathbf{B}\mathbf{A}$ est $\mathbf{B}\,\mathbf{a}^{(j)} = \sum_{i=1}^{r} A_{ij}\,\mathbf{b}_i$, où $\mathbf{a}^{(j)}$ est la colonne $j$ de $\mathbf{A}$ et $\mathbf{b}_i$ la colonne $i$ de $\mathbf{B}$ : toutes les colonnes de $\mathbf{B}\mathbf{A}$ sont des combinaisons de $\mathbf{b}_1, \dots, \mathbf{b}_r$. D'après le premier fait admis du rappel, il y a donc au plus $r$ directions indépendantes : $\operatorname{rang}(\mathbf{B}\mathbf{A}) \le r$, et multiplier par $s \ne 0$ ne change pas le rang. La factorisation coûte moins que $\mathbf{W}$ quand $r(d + k) < dk$, soit $r < \frac{dk}{d + k}$ ; pour $d = k$ : $r < d/2$ (448 pour $d = 896$, ✏️ B4.3 j).
2. $\mathbf{h} = \mathbf{W}\mathbf{x}$ : le modèle adapté est exactement le modèle d'origine. Si $\mathbf{A}_0 = 0$ aussi, les deux gradients de la question 3 sont nuls ($\mathbf{A}\mathbf{x} = 0$ et $\mathbf{B}^\top = 0$) : rien ne bouge jamais, c'est un point d'équilibre. Si $\mathbf{A}$ et $\mathbf{B}$ sont tirées au hasard, $\Delta\mathbf{W} \ne 0$ dès le départ : le modèle est abîmé avant d'avoir appris quoi que ce soit (le piège de la fiche).
3. Pour $\mathbf{B}$ : la branche s'écrit $s\,\mathbf{B}\mathbf{u}$ avec $\mathbf{u} = \mathbf{A}\mathbf{x}$, et le rappel (avec $\mathbf{M} = \mathbf{B}$, l'entrée $s\,\mathbf{u}$) donne $\partial L/\partial \mathbf{B} = \mathbf{g}\,(s\,\mathbf{u})^\top = s\,\mathbf{g}(\mathbf{A}\mathbf{x})^\top$, de forme $(d, 1)(1, r) = (d, r)$. Pour $\mathbf{A}$ : d'abord $\partial L/\partial \mathbf{u} = s\,\mathbf{B}^\top\mathbf{g}$ (taille $r$), puis le rappel avec $\mathbf{M} = \mathbf{A}$ : $\partial L/\partial \mathbf{A} = (s\,\mathbf{B}^\top\mathbf{g})\,\mathbf{x}^\top$, de forme $(r, k)$. Et $\partial L/\partial \mathbf{W} = \mathbf{g}\mathbf{x}^\top$, $d \times k$ nombres que LoRA ne calcule pas et dont il ne garde aucun état d'Adam.
4. Avec $\mathbf{B}_0 = 0$ : $\partial L/\partial \mathbf{A} = s\,\mathbf{0}^\top\mathbf{g}\mathbf{x}^\top = 0$. Seul $\mathbf{B}$ bouge au premier pas : $\mathbf{B}_1 = -\eta s\,\mathbf{g}(\mathbf{A}_0\mathbf{x})^\top$. Alors, avec $(\mathbf{A}_0\mathbf{x})^\top = \mathbf{x}^\top\mathbf{A}_0^\top$ (fait admis du rappel), $\Delta\mathbf{W}_1 = s\,\mathbf{B}_1\mathbf{A}_0 = -\eta s^2\,\mathbf{g}\,(\mathbf{A}_0\mathbf{x})^\top\mathbf{A}_0 = -\eta s^2\,\mathbf{g}\,\mathbf{x}^\top\mathbf{A}_0^\top\mathbf{A}_0$, une colonne $\mathbf{g}$ fois une ligne : rang au plus 1 (exactement 1 si $\mathbf{g} \ne 0$ et $\mathbf{A}_0\mathbf{x} \ne 0$). $\mathbf{A}$ commence à bouger au deuxième pas, quand $\mathbf{B}_1 \ne 0$.
5. $s = 2/1 = 2$, $\mathbf{A}_0\mathbf{x} = 1 \times 2 + (-1) \times 1 = 1$. $\partial L/\partial \mathbf{B} = 2 \times (1, 0, -2)^\top \times 1 = (2, 0, -4)^\top$ ; $\partial L/\partial \mathbf{A} = 0$. $\mathbf{B}_1 = -0{,}1 \times (2, 0, -4)^\top = (-0{,}2 ;\ 0 ;\ 0{,}4)^\top$. $\Delta\mathbf{W}_1 = 2\,\mathbf{B}_1\mathbf{A}_0 = \begin{pmatrix} -0{,}4 & 0{,}4 \\ 0 & 0 \\ 0{,}8 & -0{,}8 \end{pmatrix}$, de rang 1 (ses colonnes sont opposées). $\Delta\mathbf{W}_1\mathbf{x} = (-0{,}8 + 0{,}4 ;\ 0 ;\ 1{,}6 - 0{,}8) = (-0{,}4 ;\ 0 ;\ 0{,}8) = -0{,}4\,\mathbf{g}$ : la sortie a bougé dans la direction opposée au gradient, comme une descente le veut. C'est général : $\Delta\mathbf{W}_1\mathbf{x} = -\eta s^2 \lVert\mathbf{A}_0\mathbf{x}\rVert^2\,\mathbf{g}$, toujours opposé à $\mathbf{g}$ (ici $-0{,}1 \times 4 \times 1 = -0{,}4$). Au deuxième pas : $\mathbf{B}_1^\top\mathbf{g} = -0{,}2 + 0 - 0{,}8 = -1$, et $\partial L/\partial \mathbf{A} = 2 \times (-1) \times (2, 1) = (-4, -2)$.
6. (a) $\Delta\mathbf{W}_1$ est proportionnel à $s^2$ : doubler $\alpha$ le multiplie par 4 au premier pas avec SGD. (b) Avec Adam, $\Delta\mathbf{B}$ a une taille de l'ordre de $\eta$, quelle que soit l'échelle du gradient ; la modification $s\,\Delta\mathbf{B}\,\mathbf{A}$ ne garde qu'un facteur $s$ : doubler $\alpha$ revient à peu près à doubler le learning rate de la branche, d'où la phrase de Hu et coll. et leur choix de ne pas régler $\alpha$. (c) La variance d'une somme de $r$ termes indépendants de même variance $\sigma^2$ vaut $r\sigma^2$ : l'écart-type grandit comme $\sqrt{r}$, et le facteur qui le garde constant est $\propto 1/\sqrt{r}$. Le facteur $\alpha/r$ divise trop quand $r$ grandit (la branche rétrécit comme $1/\sqrt{r}$, et apprend plus lentement aux grands rangs) : c'est l'argument de rsLoRA, qui prend $\alpha/\sqrt{r}$ (Kalajdzievski, 2023).

**Erreurs fréquentes** : écrire $\partial L/\partial \mathbf{B} = \mathbf{g}\mathbf{x}^\top$ (oublier $\mathbf{A}$ et la forme $(d, r)$) ; mettre $\mathbf{B}$ au lieu de $\mathbf{B}^\top$ dans le gradient de $\mathbf{A}$ ; oublier le facteur $s$ dans les gradients ; croire que $\mathbf{A}$ et $\mathbf{B}$ bougent dès le premier pas ; confondre le rang d'une matrice et sa taille.
**Variante** : l'initialisation inverse, $\mathbf{A}_0 = 0$ et $\mathbf{B}_0$ tirée au hasard. Réponse : le modèle part aussi de $\mathbf{W}$ ($\mathbf{B}_0\mathbf{A}_0 = 0$) ; au premier pas, c'est $\partial L/\partial \mathbf{B} = s\,\mathbf{g}(\mathbf{A}_0\mathbf{x})^\top = 0$ qui s'annule, et $\mathbf{A}$ bouge seule, de $-\eta s\,\mathbf{B}_0^\top\mathbf{g}\mathbf{x}^\top$ ; l'entraînement démarre aussi. Ce qui compte, c'est qu'un seul des deux facteurs soit nul au départ.

### Ex B4.7 — Fermi : coût et latence d'un assistant RAG, API facturée au token contre petit modèle local 🧮
**Réponse** (fourchettes acceptées) : 1 540 tokens d'entrée et 120 de sortie par question, ≈ 97 millions et 7,6 millions par mois ; ≈ 13 $ par mois avec l'API économique, ≈ 1 350 $ avec l'API haut de gamme ; ≈ 2,5 s par réponse avec une API, ≈ 25 s en local sur deux cœurs ; 3 machines de deux cœurs ou 1 à 2 machines virtuelles de 8 cœurs (≈ 85 à 600 $ par mois) ; avec $k = 3$, ≈ 24 % de moins sur la facture haut de gamme et ≈ 3,5 s de moins en local.
**Démarche** :
1. Un passage fait $200 \times 1{,}3 = 260$ tokens ; l'entrée vaut $200 + 5 \times 260 + 40 = 1\,540$ tokens, la sortie 120. Par mois, $3\,000 \times 21 = 63\,000$ questions : $97{,}0$ millions de tokens d'entrée et $7{,}56$ millions de sortie.
2. API économique : $97{,}0 \times 0{,}10 + 7{,}56 \times 0{,}40 = 9{,}70 + 3{,}02 \approx 12{,}7$ $ par mois ; haut de gamme : $970 + 378 \approx 1\,350$ $. L'entrée fait 76 % et 72 % de la facture : c'est le prompt, donc les passages, qui coûte, bien que la sortie soit 4 à 5 fois plus chère au token.
3. API : $0{,}5 + 120/60 = 2{,}5$ s. Local : $1\,540/150 \approx 10{,}3$ s de prefill, puis $120/8 = 15$ s de génération, ≈ 25 s ; la génération domine un peu, le prefill reste lourd. Le streaming ne raccourcit pas la réponse, mais l'utilisateur lit dès le premier token, après une dizaine de secondes en local au lieu de 25.
4. Une machine de deux cœurs traite $3\,600/25{,}3 \approx 140$ questions par heure. Il en faut $3\,000/8 = 375$ par heure : 3 machines (2,6 arrondi au-dessus), sans marge pour les pics. Une machine virtuelle de 8 cœurs répond en ≈ 6,3 s, soit ≈ 570 questions par heure : une seule suffit en moyenne, deux pour absorber les pics. Allumée 10 heures par jour ouvré : $0{,}40 \times 10 \times 21 \approx 84$ $ par mois ; 24 heures sur 24 tout le mois : $0{,}40 \times 730 \approx 290$ $ ; doublé avec la seconde machine.
5. Avec $k = 3$ : $200 + 3 \times 260 + 40 = 1\,020$ tokens d'entrée, 64,3 millions par mois ; haut de gamme $643 + 378 \approx 1\,020$ $ (≈ −24 %), économique ≈ 9,5 $ ; en local, $1\,020/150 + 15 \approx 21{,}8$ s. On risque de perdre le bon passage : le recall@3 est au plus le recall@5. La décision se prend sur la courbe du recall@k des questions annotées (204.6).
6. Les données ne quittent pas l'entreprise (RGPD, secrets, aucun transfert hors de l'Union) ; le modèle et sa révision ne changent pas sans toi (reproductibilité, aucune dépendance à un fournisseur ou à sa disponibilité) ; il fonctionne hors ligne.
7. Le temps d'ingénieur (mise en place, surveillance, mises à jour) ; le calcul des embeddings et la réindexation ; les réessais quand un JSON est invalide, les appels d'évaluation et d'un éventuel juge LLM ; les pics et la concurrence ; les remises des fournisseurs sur les prompts répétés ; l'énergie et le matériel ; un GPU, qui traiterait plusieurs questions à la fois.

**Erreurs fréquentes** : oublier les passages dans l'entrée (seulement la question) ; compter des mots au lieu de tokens ; oublier le prefill dans la latence locale ; appliquer le prix de sortie à l'entrée ; additionner les latences de questions qui pourraient être servies en parallèle ; conclure sur un prix sans dire la date du relevé.
**Variante** : les mêmes questions en français, avec une fertilité de 1,6 token par mot. Réponse : 320 tokens par passage, $200 + 5 \times 320 + 40 = 1\,840$ tokens d'entrée, 115,9 millions par mois : ≈ 1 540 $ avec l'API haut de gamme (+14 %), ≈ 27 s en local ; la langue se paie au token (B2).

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B4.8 — Le RAG expliqué en cinq lignes 🗣️
**Réponse modèle** : (1) Un modèle de langue ne sait que ce qu'il a lu pendant son entraînement, il ignore tes documents et ce qui s'est passé depuis, et quand il ne sait pas, il invente une réponse qui sonne juste. (2) L'idée du RAG : avant de répondre, on cherche dans tes documents les quelques passages qui parlent de la question, comme un bibliothécaire qui apporte les bonnes pages. (3) On donne ces pages au modèle avec la consigne : réponds seulement avec elles, et si elles ne disent rien, dis que tu ne sais pas ; c'est un examen à livre ouvert. (4) Il indique la page qu'il a utilisée, et l'on peut vérifier que la page dit bien cela. (5) Pour le tenir à jour, on change les documents, sans réentraîner le modèle, ce qui est rapide et bon marché.
**Ce qui compte** : la limite du modèle seul (connaissances figées, invention) ; la recherche d'abord ; la consigne de s'en tenir aux passages et d'avouer l'ignorance ; des citations vérifiables ; la mise à jour sans réentraînement.
**Erreurs fréquentes** : du jargon non expliqué (« embedding », « vectoriel », « prompt ») ; dire que le modèle « apprend » les documents ; oublier la vérification des citations ; promettre qu'il n'invente plus jamais.
**Variante** : la même idée en une phrase, pour un recruteur. Réponse possible : « Un RAG cherche les passages pertinents de vos documents par une recherche lexicale et dense, les place dans le prompt d'un LLM avec la consigne d'y répondre en citant ses sources, et se met à jour en réindexant les documents, sans réentraîner. »

### Ex B4.9 — Un assistant RH qui répond sur les salaires : données personnelles, hallucinations, injection, transparence ⚖️
**Réponse** :
1. **Données personnelles** : les bulletins de paie, contrats et entretiens annuels sont des données personnelles (toute information sur une personne identifiable, art. 4 du RGPD), souvent sensibles en pratique (salaires, absences, éventuellement des données de santé ou syndicales, catégories particulières de l'art. 9) ; la convention collective et les grilles par poste ne le sont pas, sauf un poste occupé par une seule personne. Base légale : l'exécution du contrat de travail ou les obligations légales pour ce qui touche la paie, l'intérêt légitime pour un outil d'information ; le consentement d'un salarié est fragile, car il n'est pas libre face à son employeur. Minimisation : n'indexer que ce qui sert (pas les entretiens annuels pour répondre sur un salaire de base), ne pas garder les conversations plus longtemps que nécessaire. Sous-traitant : un contrat de l'art. 28 avec le fournisseur de l'API et l'hébergeur. Transferts : une API américaine impose un cadre du chapitre V (décision d'adéquation, clauses contractuelles types) et le dire aux salariés. Une analyse d'impact (art. 35) est très probablement requise : usage innovant d'une technologie, données de personnes vulnérables face au responsable (les salariés), données sensibles en pratique ; dans le doute, on la fait. Le délégué à la protection des données est consulté, les salariés informés (art. 13), et le comité social et économique (CSE) informé et consulté avant l'introduction d'une nouvelle technologie (Code du travail, art. L2312-8, 4°, dans les entreprises d'au moins 50 salariés) ; faute de cette consultation, le tribunal judiciaire de Créteil a suspendu en référé, le 15 juillet 2025, l'usage d'outils d'IA déjà déployés.
2. **Droits d'accès** : une consigne se contourne (injection, reformulation, erreur du modèle) ; si le bulletin de Claire est dans le contexte, il peut sortir. Le filtre se fait **dans la recherche** : chaque chunk porte l'identifiant de son propriétaire et un niveau d'accès, la recherche ne rend à un salarié que les documents communs et les siens, et le modèle ne voit jamais le dossier d'un autre.
3. **Hallucinations** : un montant de salaire ou de prime inventé, une règle de congés ou d'ancienneté fausse, un droit inexistant ; le salarié décide sur une information fausse, l'employeur peut se voir opposer l'écrit de son propre outil. Parades : répondre seulement d'après les passages, avec citation vérifiée ; **calculer les montants par programme** (une requête à la base de paie, dont l'assistant ne fait que présenter le résultat) ; « je ne sais pas » et renvoi vers le service RH ; un jeu de test de questions RH, avec des questions pièges, rejoué à chaque changement.
4. **Injection** : une injection indirecte : le document du salarié est récupéré quand il pose une question, et le modèle peut suivre l'instruction cachée. Parades de la fiche : délimiteurs et consigne de ne jamais suivre le texte des passages, filtrage des formules d'injection, validation des sorties ; mais c'est le filtre d'accès de la question 2 qui rend l'attaque sans objet : le modèle n'a jamais les grilles ni les salaires des autres dans son contexte, et n'a aucun outil pour aller les chercher.
5. **AI Act** : l'article 50 (1) demande que les personnes sachent qu'elles parlent à une IA, sauf si c'est évident ; il s'applique depuis le 2 août 2026, à l'entreprise qui construit l'assistant (elle en est le fournisseur). L'outil informe les salariés, il ne décide ni des conditions de travail, ni d'une promotion, ni d'une évaluation : il ne relève pas du point 4 de l'annexe III. Utilisé par les RH pour proposer les augmentations, il « décide » des conditions de la relation de travail (point 4 b) : haut risque. L'entreprise est alors à la fois fournisseur (gestion des risques, données de qualité, documentation, contrôle humain : art. 9 à 15) et déployeur : l'article 26 lui demande notamment, comme employeur, d'informer les représentants des travailleurs et les salariés concernés **avant** la mise en service (art. 26, § 7) ; depuis le règlement (UE) 2026/1744, ces obligations de l'annexe III s'appliquent à partir du 2 décembre 2027. L'article 4, dans sa version modifiée par ce règlement, demande à l'employeur de prendre des mesures pour soutenir la maîtrise de l'IA de son personnel (former les RH et les utilisateurs à ce que l'outil fait et ne fait pas).
6. **Transparence des salaires** : la directive (UE) 2023/970 devait être transposée au plus tard le 7 juin 2026 (art. 34) ; fin septembre 2026, la plupart des États membres, dont la France, ne l'avaient pas encore transposée (le projet de loi français a été présenté en conseil des ministres le 10 septembre 2026 ; vérifié le 2 novembre 2026 : relis son état avant de t'y fier). L'assistant peut répondre à la demande d'information de l'article 7 avec des **agrégats** calculés par programme (niveau moyen par catégorie et par sexe), jamais par le modèle ; une catégorie de deux personnes révèle le salaire de l'autre : on fixe un effectif minimal, sous lequel la réponse passe par les RH.
7. **Le montage** : un modèle hébergé dans l'entreprise, ou une API dont les serveurs et le contrat restent dans l'Union, parce que des bulletins de paie ne doivent pas partir chez un tiers sans nécessité ; l'assistant ne lit que les documents communs et ceux du salarié, ne calcule aucun montant lui-même, et cite ses sources. « Limites et usage » : « Assistant d'information RH, réservé aux salariés authentifiés ; il ne lit que les accords, les grilles et le dossier de la personne connectée. Les montants viennent de la base de paie, jamais du modèle ; chaque réponse cite sa source et peut être fausse : en cas de doute, le service RH fait foi. Il ne prend aucune décision et n'est pas utilisé pour évaluer ni pour fixer les rémunérations. Conversations conservées 30 jours, pour la sécurité seulement. Testé sur 200 questions, dont 40 sans réponse ; résultats publiés en interne. »

**Erreurs fréquentes** : protéger les salaires par une consigne au modèle ; laisser le modèle calculer des montants ; croire qu'un modèle local dispense du RGPD ; classer tout outil RH en haut risque, ou aucun ; oublier les représentants du personnel ; publier des moyennes sur des groupes minuscules.
**Variante** : les managers utilisent le même assistant pour classer les membres de leur équipe avant l'attribution des primes. Réponse : il évalue la performance et influence une décision sur la rémunération : point 4 b) de l'annexe III, donc haut risque (à partir du 2 décembre 2027), avec en plus le droit des salariés de ne pas faire l'objet d'une décision fondée exclusivement sur un traitement automatisé (art. 22 du RGPD) : il faut une décision humaine réelle et motivée.

### Ex B4.10 — Fine-tuner sur des romans : domaine public, œuvres protégées et droit de retrait (opt-out) ⚖️
**Réponse** :
1. Dans l'Union, les droits patrimoniaux durent 70 ans après la mort de l'auteur, comptés à partir du 1ᵉʳ janvier suivant : Conan Doyle (mort en 1930) est libre depuis le 1ᵉʳ janvier 2001, Verne (mort en 1905) depuis longtemps. Mais une traduction est une œuvre dérivée, protégée pour son traducteur : une traduction de 2015 est protégée jusqu'à 70 ans après la mort de celui-ci. Pour chaque fichier, il faut vérifier l'auteur **et** le traducteur, leur date de mort, la date de l'édition ; le Projet Gutenberg n'est pas une garantie pour l'Union (il juge d'après le droit américain), et ses fichiers portent une licence qui encadre l'usage de sa marque et de son en-tête, pas le texte libre lui-même.
2. L'article 4 de la directive (UE) 2019/790 (art. L122-5-3 du Code de la propriété intellectuelle) permet à tous les reproductions et extractions pour la fouille de textes et de données, à trois conditions : un **accès licite** aux œuvres, l'**absence de réserve** exprimée par le titulaire « de manière appropriée », notamment par des procédés lisibles par machine pour les contenus en ligne (un `robots.txt`, des métadonnées ou un protocole comme TDMRep, des conditions d'utilisation), et une conservation limitée au temps nécessaire. Source (2) : l'accès est licite (livres achetés), mais il faut vérifier que les éditeurs n'ont pas réservé la fouille (beaucoup le font dans les métadonnées et les contrats des livres numériques) et ne pas contourner de protection technique ; sinon, une licence. Source (3) : le titulaire s'est opposé par des moyens lisibles par machine et par ses conditions : l'exception ne s'applique pas ; copier ces romans pour entraîner est une reproduction non autorisée.
3. Le style d'un auteur n'est pas protégé, seule l'expression l'est (et des personnages très caractérisés peuvent l'être) ; un modèle qui recopie de longs passages d'un roman récent produit une reproduction, quel que soit le procédé. Le tribunal régional de Munich (GEMA contre OpenAI, 11 novembre 2025, première instance) a jugé que des paroles de chansons **mémorisées** dans les poids sont reproduites dans le modèle, que cette mémorisation dépasse la fouille de textes et n'est pas couverte par l'exception, et que les sorties qui les recopient sont aussi des contrefaçons, imputables au fournisseur. Mesures : dédupliquer et limiter le nombre de passages sur les données (la mémorisation croît avec les répétitions, B3) ; tester la recopie avant publication (des débuts de passages en prompt, génération gloutonne, comparaison) ; filtrer en sortie les longues suites mot pour mot d'œuvres protégées.
4. Non : les lignes directrices de la Commission (18 juillet 2025) retiennent un critère **indicatif** : celui qui modifie un modèle est considéré comme le fournisseur du modèle modifié quand sa modification utilise plus d'un tiers du calcul d'entraînement du modèle d'origine ; un LoRA de quelques heures sur un petit GPU en est très loin, et les obligations de l'article 53 restent au fournisseur du modèle de base, s'il en a : les mêmes lignes directrices prennent pour indice d'un modèle à usage général un entraînement de plus de $10^{23}$ FLOPs, et la règle $6ND$ (B3) donne à Qwen2.5-0.5B environ $6 \times 0{,}5 \cdot 10^9 \times 18 \cdot 10^{12} \approx 5 \cdot 10^{22}$ FLOPs, si l'on prend les 18 000 milliards de tokens annoncés pour la famille. En revanche, son assistant génère du texte : l'article 50 (2) lui demande, comme fournisseur du système, un marquage lisible par machine des textes produits, depuis le 2 août 2026 (le 2 décembre 2026 pour un système déjà sur le marché avant cette date), et l'information des utilisateurs qui dialoguent avec l'assistant (art. 50 (1)).
5. Apache 2.0 : garder le texte de la licence et l'avis de copyright, signaler les modifications, ne pas utiliser les marques de l'éditeur pour son produit ; l'adaptateur peut être publié sous une licence de son choix, en citant le modèle de base. Avec la *Llama 3.2 Community License* : « Built with Llama » sur le produit, un nom d'adaptateur ou de modèle dérivé qui commence par « Llama », le respect de la politique d'usage et la limite des 700 millions d'utilisateurs actifs mensuels.
6. « Données d'entraînement : les aventures de Holmes (4 romans, 56 nouvelles) et des romans de Verne (Projet Gutenberg, domaine public dans l'Union, vérifié auteur par auteur) et des traductions françaises de Holmes dont le traducteur est mort depuis plus de 70 ans ; 300 romans policiers récents sous licence de leurs éditeurs (contrat du 3 mars 2026) ; exclus : les traductions protégées, le site X (fouille réservée par `robots.txt` et ses conditions). Dédupliqué ; test de recopie : aucune suite de plus de 50 mots reproduite sur 500 essais. »

**Erreurs fréquentes** : croire que tout ce qui est sur le Projet Gutenberg est libre dans l'Union, traductions comprises ; croire que l'achat d'un livre autorise tout ; ignorer un `robots.txt` ou des conditions d'utilisation ; confondre le style (libre) et le texte (protégé) ; croire qu'un LoRA fait de soi un fournisseur de modèle à usage général, ou qu'il dispense de l'article 50.
**Variante** : un laboratoire public de recherche entraîne le même adaptateur sur les 2 000 romans du site, pour étudier la mémorisation. Réponse : l'article 3 de la directive (organismes de recherche, fins de recherche scientifique) ne connaît pas l'opt-out : la fouille est permise si l'accès est licite, avec une conservation sécurisée ; mais la publication d'un adaptateur qui recopie les romans, ou un usage commercial, sort de cette exception.

### Ex B4.11 — Hu et coll. (2021) : « LoRA: Low-Rank Adaptation of Large Language Models » 📄
**Réponse** :
1. Un fine-tuning complet produit, pour chaque tâche, une copie de tous les paramètres : 175 milliards pour GPT-3, à stocker et à servir séparément. Les **adaptateurs** ajoutent des couches traitées l'une après l'autre, donc de la latence ; l'**optimisation du prompt** (prefix tuning) est difficile à optimiser, varie de façon non monotone avec ses paramètres et prend une partie de la longueur de séquence. Tableau 1 (GPT-2 medium, un GPU) : les adaptateurs ajoutent 2 à 3 % de latence avec un batch de 32 et des séquences de 512, mais **+20,7 % et +30,3 %** avec un batch de 1 et des séquences de 128, le cas d'un service en ligne.
2. Sur Li et coll. (2018) et Aghajanyan et coll. (2020), qui montrent que les modèles sur-paramétrés vivent dans un espace de faible dimension « intrinsèque ». L'hypothèse : la **modification** des poids pendant l'adaptation a elle aussi un faible « rang intrinsèque », un petit nombre de directions utiles. Pour GPT-3, un rang de 1 ou 2 suffit, alors que le rang plein vaut 12 288.
3. $\mathbf{h} = \mathbf{W}_0\mathbf{x} + \Delta\mathbf{W}\mathbf{x} = \mathbf{W}_0\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$. $\mathbf{A}$ suit une gaussienne aléatoire, $\mathbf{B}$ est nulle : $\Delta\mathbf{W} = 0$ au départ. $\Delta\mathbf{W}\mathbf{x}$ est multiplié par $\alpha/r$, avec $\alpha$ égal au premier $r$ essayé et jamais réglé ensuite (avec Adam, régler $\alpha$ revient à peu près à régler le learning rate, ∂ B4.6). LoRA « généralise » le fine-tuning complet : avec un rang égal à celui des matrices pré-entraînées, sur toutes les matrices et avec les biais, on retrouve à peu près l'expressivité du fine-tuning complet ; les adaptateurs tendent vers un MLP, et les préfixes vers un modèle qui ne peut plus lire de longues entrées.
4. Seules les matrices de l'attention ($\mathbf{W}_q$, $\mathbf{W}_k$, $\mathbf{W}_v$, $\mathbf{W}_o$, traitées comme $d_{\text{model}} \times d_{\text{model}}$) ; le MLP, les layer norms et les biais restent gelés. Un seul type de matrice avec $r = 8$ : $2 \times 96 \times 12\,288 \times 8 = 18\,874\,368$ paramètres, soit 37,7 Mo en FP16 (36 Mio : les « roughly 35MB » sont un ordre de grandeur) ; deux types avec $r = 4$ donnent le même compte. Le tableau 5 : les auteurs retiennent $\{\mathbf{W}_q, \mathbf{W}_v\}$ avec $r = 4$ (73,7 sur WikiSQL, 91,3 sur MultiNLI), « the best performance overall » ; les quatre matrices avec $r = 2$ font aussi bien (73,7 et 91,7). Mettre tout le budget dans $\mathbf{W}_q$ ou $\mathbf{W}_k$ avec $r = 8$ coûte plus de 3 points sur WikiSQL (70,4 et 70,0), et $\mathbf{W}_v$ ou $\mathbf{W}_o$ seuls restent en dessous (73,0 et 73,2). Leur conclusion : même un rang de 4 capte assez de $\Delta\mathbf{W}$ pour qu'il vaille mieux adapter plus de types de matrices qu'un seul avec un rang plus grand.
5. $175\,255{,}8 \times 10^6 / 18{,}9 \times 10^6 \approx 9\,300$, environ 10 000 fois moins de paramètres entraînables. La mémoire GPU de l'entraînement passe de 1,2 To à 350 Go (÷ 3,4, « 3 times » dans le résumé) : plus d'états d'Adam pour les poids gelés. Un checkpoint passe de 350 Go à 35 Mo ($r = 4$, $\mathbf{W}_q$ et $\mathbf{W}_v$). 100 tâches : $350 + 100 \times 0{,}035 \approx 354$ Go avec LoRA, contre $100 \times 350 = 35$ To. Débits : $43{,}1/32{,}5 \approx 1{,}33$, soit 33 % de tokens en plus par seconde, ou un temps par token réduit de $1 - 32{,}5/43{,}1 \approx 25\,\%$ : c'est le sens des « 25% speedup ».
6. 4,7 M paramètres ($r_q = r_v = 1$, ou $r_v = 2$) et 37,7 M ($r_q = r_v = 8$) : $2 \times 192 \times 12\,288 \times 8 = 37\,748\,736$. L'annexe annonce aussi « $r_q = r_k = r_v = r_o = 2$ » pour 37,7 M, mais la formule de la section 5.1 donne alors $2 \times 384 \times 12\,288 \times 2 \approx 18{,}9$ M, le budget du tableau 5 : une incohérence de l'article ($r = 4$ donnerait 37,7 M), à signaler plutôt qu'à corriger soi-même. L'article se contredit lui-même : son tableau 15 (annexe F) range bien $r_q = r_k = r_v = r_o = 2$ à 18,8 M et $r = 4$ à 37,7 M, et ses lignes à 37,7 M (73,8 et 91,6 pour $r_q = r_v = 8$ ; 74,0 et 91,7 pour les quatre matrices à $r = 4$) suggèrent que la ligne à 37,7 M du tableau 4 mêle les deux réglages (74,0 sur WikiSQL comme le second, 91,6 sur MNLI-m comme le premier). Sur WikiSQL, 73,4 et 74,0 contre 73,8 pour le fine-tuning complet (à ±0,5 près, les fluctuations annoncées) ; sur MNLI-m, 91,7 et 91,6 contre 89,5 : LoRA égale ou dépasse le fine-tuning complet avec 37 000 fois moins de paramètres (pour 4,7 M). Tableau 12 : un learning rate de $2 \times 10^{-4}$ pour LoRA contre $5 \times 10^{-6}$ pour le fine-tuning complet, 40 fois plus grand.
7. Pour $\{\mathbf{W}_q, \mathbf{W}_v\}$, $r = 1$ donne déjà 73,4 et 91,3, aussi bien que $r = 64$ (73,5 et 91,4) ; $\mathbf{W}_q$ seul a besoin d'un rang plus grand (68,8 à $r = 1$, 70,5 à $r = 4$). Note 6 : un petit $r$ ne suffira pas forcément pour toute tâche, par exemple si la tâche est dans une autre langue que celle du pré-entraînement : réentraîner tout le modèle ferait alors mieux.
8. Le tableau 7 projette $\mathbf{W}_q$ sur les $r$ directions principales de $\Delta\mathbf{W}_q$ ($\lVert\mathbf{U}^\top\mathbf{W}_q\mathbf{V}^\top\rVert_F$), et compare aux directions principales de $\mathbf{W}_q$ et à des directions aléatoires. Pour $r = 4$ : $0{,}32$ sur les directions de $\Delta\mathbf{W}_q$, contre $21{,}67$ sur celles de $\mathbf{W}_q$ et $0{,}02$ au hasard ; $\lVert\Delta\mathbf{W}_q\rVert_F = 6{,}91$, d'où le facteur $6{,}91/0{,}32 \approx 21{,}6$ (l'article écrit « 21.5 », avec des normes déjà arrondies à deux décimales). $\Delta\mathbf{W}$ amplifie fortement des directions déjà présentes dans $\mathbf{W}$, mais peu mises en avant par le pré-entraînement : des features apprises que la tâche rend importantes.
9. On cible souvent **toutes** les couches denses (`"all-linear"`, le MLP compris), pas seulement l'attention ; on prend $\alpha = r$ ou $2r$ (peft : $\alpha = 8$ par défaut), ou $\alpha/\sqrt{r}$ (rsLoRA) ; des variantes (DoRA, QLoRA sur un modèle quantifié en 4 bits, d'autres initialisations) ; et la GQA des modèles récents rend $\mathbf{W}_k$ et $\mathbf{W}_v$ bien plus petites que $d_{\text{model}} \times d_{\text{model}}$ (✏️ B4.3).

**Erreurs fréquentes** : attribuer à l'article l'adaptation du MLP ; croire que $\alpha$ se règle comme un hyperparamètre de plus ; lire « 3 times » comme une vitesse ; confondre la taille du checkpoint et la mémoire d'entraînement ; oublier que $\mathbf{W}_k$ et $\mathbf{W}_v$ y sont carrées.
**Variante** : vérifie les deux réglages que l'annexe D.4 donne pour le budget de 4,7 M. Réponse : $r_q = r_v = 1$ donne $2 \times 192 \times 12\,288 \times 1 = 4\,718\,592$, et $r_v = 2$ seul donne $2 \times 96 \times 12\,288 \times 2 = 4\,718\,592$ : le même compte, deux façons de dépenser le même budget.

<a id="entretien"></a>

## 💼 Entretien

### B4.E1 — Comment réduis-tu les hallucinations d'un assistant fondé sur un LLM ?
**Réponse modèle en 60 secondes** : « Un LLM produit la suite la plus plausible, pas la plus vraie ; je ne promets pas de supprimer les hallucinations, je les encadre et je les mesure. D'abord, je regarde d'où viennent les erreurs : dans un assistant documentaire, beaucoup sont des passages manquants, donc je mesure le recall@k de la recherche avant de toucher au prompt. Ensuite, j'ancre les réponses dans des documents par un RAG, avec la consigne de ne répondre qu'avec les passages, de citer le passage utilisé, et de dire "je ne sais pas" sinon ; je vérifie les citations par programme. Tout ce qui peut se calculer ou se valider ne passe pas par le modèle : montants, dates, JSON. Je garde un humain dans la boucle quand une erreur coûte cher. Et je mesure sur un jeu de test à nous, avec des questions sans réponse et des questions à présupposé faux : le taux d'invention, le taux d'abstention juste, avant et après chaque changement. Baisser la température rend les réponses moins variées, pas plus vraies. »
**Ce qui compte** : la cause (le plausible) ; d'abord la recherche ; ancrage, citations vérifiées, abstention ; calcul et validation par programme ; humain dans la boucle ; mesure sur un jeu de test avec des pièges.
**Relances possibles** : « Un modèle plus gros hallucine-t-il moins ? » (souvent un peu moins sur des faits connus, mais il invente avec autant d'aplomb hors de ses connaissances ; la taille ne remplace ni l'ancrage ni la mesure) · « Et si le modèle cite le bon passage mais en tire une mauvaise réponse ? » (c'est un échec de génération : on le compte à part, on raccourcit et on réordonne le contexte, on simplifie la consigne, ou l'on change de modèle)

### B4.E2 — RAG ou fine-tuning : comment choisis-tu ?
**Réponse modèle en 60 secondes** : « Je sépare ce que le système doit savoir et la façon dont il doit répondre. Le catalogue de 50 000 produits, c'est de la connaissance qui change : prix, stocks, nouveautés. Je le mets dans un RAG : on réindexe à chaque mise à jour, chaque réponse cite la fiche produit, et l'on filtre par marché ou par client. Un fine-tuning apprendrait mal 50 000 fiches et serait périmé le lendemain. Le ton de la marque, c'est un comportement : je commence par un prompt système avec quelques exemples de réponses, parce que c'est gratuit et réversible. Si le ton ne tient pas, ou si le prompt devient trop long, je fais un LoRA sur quelques centaines de bonnes réponses de la marque, mises en forme par le chat template, la loss sur la réponse seule. Les deux se combinent : un modèle adapté au ton qui lit les fiches du RAG. Et je décide sur un jeu de test : exactitude des informations, et un jugement du ton, par des humains ou un juge validé. »
**Ce qui compte** : connaissances contre comportement ; RAG pour ce qui change et se cite ; prompt d'abord, fine-tuning ensuite ; la combinaison ; une décision mesurée.
**Relances possibles** : « Et un contexte très long, qui contient tout le catalogue ? » (trop cher par question, lent, et les modèles lisent mal le milieu d'un long contexte ; la recherche reste utile) · « Combien d'exemples pour le LoRA du ton ? » (quelques centaines de bonnes réponses variées suffisent souvent pour un style ; on mesure sur des exemples tenus à l'écart, et l'on surveille l'oubli des capacités générales)

### B4.E3 — Explique LoRA et pourquoi c'est si économique
**Réponse modèle en 60 secondes** : « LoRA gèle les poids pré-entraînés et n'apprend qu'une modification de rang faible : pour une matrice $\mathbf{W}$ de $d \times k$, on apprend deux petites matrices $\mathbf{B}$ ($d \times r$) et $\mathbf{A}$ ($r \times k$), et la couche calcule $\mathbf{W}\mathbf{x} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}\mathbf{x}$. $\mathbf{B}$ part de zéro, donc on part exactement du modèle d'origine. L'économie est triple. Les paramètres : $r(d + k)$ au lieu de $dk$, pour une matrice $4\,096 \times 4\,096$ avec $r = 8$, 65 536 au lieu de 16,8 millions. La mémoire d'entraînement : un paramètre entraîné avec AdamW coûte environ 16 octets (poids, gradient, copie maîtresse, deux moyennes), un poids gelé en bfloat16 en coûte 2 : on peut fine-tuner sur un seul GPU, et même sur un modèle quantifié en 4 bits avec QLoRA. Le stockage : un adaptateur de quelques mégaoctets par tâche au lieu d'une copie du modèle. Après l'entraînement, on fusionne $\frac{\alpha}{r}\mathbf{B}\mathbf{A}$ dans $\mathbf{W}$ : aucune latence en plus. Ce qu'on perd : une capacité limitée pour une tâche très éloignée du pré-entraînement, et des choix à faire, le rang, les modules ciblés, le learning rate. »
**Ce qui compte** : la formule et $\mathbf{B}_0 = 0$ ; l'exemple chiffré ; les trois économies (paramètres, mémoire, stockage) ; la fusion ; les limites.
**Relances possibles** : « Pourquoi $\mathbf{B}$ à zéro et pas $\mathbf{A}$ ? » (il suffit qu'un des deux soit nul pour partir du modèle d'origine ; si les deux l'étaient, les gradients seraient nuls et rien n'apprendrait) · « Quel rang choisir ? » (on commence par 8 ou 16 sur toutes les couches denses ; l'article montre qu'un très petit rang suffit souvent, et l'on vérifie sur la validation en faisant varier $r$)

### B4.E4 — Comment évalues-tu un système RAG, de la recherche à la réponse ?
**Réponse modèle en 60 secondes** : « Je commence par un jeu de test : de vraies questions d'utilisateurs, avec la réponse de référence et l'identifiant du passage qui la contient, et une part de questions dont la réponse n'est pas dans les documents. Puis j'évalue les deux étages séparément. La recherche : recall@k, le bon passage est-il dans ce que je montre au modèle, et MRR, arrive-t-il tôt. Si le recall@k est bas, aucun prompt ne sauvera la réponse : je travaille le découpage, la tokenisation, le modèle d'embedding, l'hybride. La réponse : exactitude normalisée, ou un juge LLM validé sur un échantillon de notes humaines ; la part de citations vérifiées, le passage cité existe et contient la réponse ; et sur les questions sans réponse, la part de "je ne sais pas". Pour chaque échec, je regarde d'abord si le bon passage était dans le top-k : cela dit si c'est la recherche ou la génération. Je rejoue tout à chaque changement de découpage, de modèle ou de prompt, avec la révision et la version du prompt notées, et je n'oublie pas l'incertitude d'un petit jeu de test. »
**Ce qui compte** : un jeu de test annoté, avec des questions sans réponse ; recherche puis réponse, séparément ; citations vérifiées ; diagnostic par le recall de chaque échec ; non-régression et incertitude.
**Relances possibles** : « Un juge LLM suffit-il ? » (non : il a des biais de position, de longueur et d'auto-complaisance ; on le valide sur des notes humaines et on présente les réponses dans les deux ordres) · « Combien de questions faut-il ? » (assez pour que l'écart cherché dépasse l'incertitude : avec 100 questions, l'écart-type d'une proportion de 80 % vaut 4 points, donc on ne conclut pas sur 3 points d'écart)

### B4.E5 — Quels risques de sécurité avec un LLM en production, et quelles parades ?
**Réponse modèle en 60 secondes** : « Le premier risque, ici, c'est l'injection de prompt indirecte : chaque e-mail client est un texte non fiable que le modèle va lire, et quelqu'un peut y écrire des instructions. Comme le modèle peut créer ou modifier des tickets, une injection réussie devient une action : c'est l'agentivité excessive, LLM03 de l'OWASP 2026. Mes parades, en couches : des droits minimaux, le modèle propose, un humain ou une règle valide toute action à conséquence, et il n'a accès qu'aux tickets du client concerné ; aucun secret ni clé dans son prompt ; séparation nette entre consignes et contenu des e-mails, avec des délimiteurs, et filtrage des formules d'injection connues ; validation des sorties avant toute exécution ou affichage, format, liens, champs autorisés. Ensuite, la fuite de données entre clients, si le contexte ou l'index mélange leurs documents, et la consommation sans limite : quotas et plafonds de tokens. Je journalise tout, et je mesure le taux de réussite d'un jeu d'attaques avant et après chaque parade, parce qu'aucune n'est complète. »
**Ce qui compte** : injection indirecte et agentivité ; droits minimaux et confirmation humaine ; secrets, séparation, filtrage, validation des sorties ; fuites et coûts ; journaux et tests d'attaque.
**Relances possibles** : « Un modèle plus récent est-il immunisé ? » (non : les modèles récents résistent mieux aux attaques connues, mais aucune parade n'est complète ; on suppose l'injection possible et on limite ce qu'elle peut faire) · « Comment tester la sécurité avant la mise en production ? » (un jeu d'attaques variées, directes et cachées dans des e-mails, rejoué automatiquement ; on mesure le taux de réussite et l'on bloque la livraison s'il remonte)

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes, exécutées, sont dans `05_solutions.ipynb`. Les chiffres ci-dessous viennent de ce notebook (mode rapide, CPU à deux cœurs) et de `tools/chapters/history_chB4.py` (calibrage du 🔮 B4.13 sur 25 séries de 10 graines et quatre prompts, du 🔮 B4.19 sur 20 perturbations des embeddings, e5 sur Holmes, protocole complet du 🔬 B4.16 avec Qwen2.5, RAG sur les 30 questions, 🐛 B4.22 rejoué sur ses 16 versions, protocoles complets des 🔬 B4.24 et B4.27, défi B4.30 couru sur cinq découpages dans chaque mode). Les modèles sont à des révisions figées et les décodages gloutons sont déterministes : tes réponses sont les mêmes sur la même machine ; sur une autre (Colab), un arrondi différent peut changer un token, puis toute la suite d'une génération, et les tirages aléatoires changent : les chiffres des générations sont alors voisins, pas identiques. Les durées dépendent de la machine et sont données en fourchette.

### Ex B4.12 — Premier contact : AutoTokenizer, AutoModelForCausalLM, generate() et pipeline sur un petit modèle 📦
**Réponse** : a) **22** tokens ; b) **211** modules `nn.Linear` ; c) **3 540 096** paramètres dans le premier bloc. Mesure du notebook de solutions : 64 tokens en 3,55 s, soit **18 tokens par seconde**, environ 0,64 fois les 28 tokens par seconde estimés en ✏️ B4.1 j) (une mesure : 18 à 19 tokens par seconde sur deux cœurs, davantage sur une autre machine).
**Démarche** :
```python
def load_12():
    tok = AutoTokenizer.from_pretrained(SMOL["repo"], revision=SMOL["revision"])
    model = AutoModelForCausalLM.from_pretrained(SMOL["repo"], revision=SMOL["revision"], dtype=torch.float32)
    return tok, model.eval()


def complete_12(tok, model, prompt, max_new_tokens):
    enc = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, enc["input_ids"].shape[1]:]                 # generate() returns the prompt, then the new tokens
    return tok.decode(new, skip_special_tokens=True), int(new.shape[0])


def counts_12(tok, model):
    n_prompt = len(tok(PROMPT_12)["input_ids"])
    n_linear = sum(isinstance(m, nn.Linear) for m in model.modules())
    n_block = sum(p.numel() for p in model.model.layers[0].parameters())
    return n_prompt, n_linear, n_block


def make_pipe_12(tok, model):
    return pipeline("text-generation", model=model, tokenizer=tok)
```
a) Les 19 mots de `PROMPT_12` deviennent 22 tokens (le tokenizer de SmolLM2 n'ajoute aucun token spécial à un texte brut) : un mot courant fait un token, la ponctuation et quelques mots rares en font davantage (B2). b) Chaque bloc a sept couches denses, `q_proj`, `k_proj`, `v_proj`, `o_proj` pour l'attention et `gate_proj`, `up_proj`, `down_proj` pour le MLP : $30 \times 7 = 210$, plus la tête de sortie `lm_head`, dont le poids **est** la table des tokens (poids liés, ✏️ B4.1) mais qui reste un module à part. c) Le bloc : $576^2$ pour `q_proj` et `o_proj`, $576 \times 192$ pour `k_proj` et `v_proj` (3 têtes de clés et valeurs de 64 : le GQA de B3), $3 \times 576 \times 1\,536$ pour le MLP, et deux RMSNorm de 576 poids, soit $663\,552 + 221\,184 + 2\,654\,208 + 1\,152 = 3\,540\,096$ ; $30 \times 3\,540\,096$, plus la table des tokens et la norme finale, redonne les 134 515 008 paramètres de ✏️ B4.1. La vitesse mesurée reste sous l'estimation « par proportion des mémoires » : le calcul suppose qu'un token coûte la lecture de tous les poids, et rien d'autre ; or le petit modèle a une grande table de tokens (21 % de ses paramètres, ✏️ B4.1), lue en entier par la tête de sortie à chaque token, des matrices de 576 colonnes que le CPU traite moins efficacement que de grandes, et chaque token paie aussi le surcoût fixe de `generate()` en Python (le KV cache, les processeurs de logits) sur 30 blocs : un coût fixe qui pèse plus lourd quand le calcul est petit. Le pipeline fait à ta place le chat template, la génération et le décodage : commode pour essayer, mais il cache ce qu'il envoie au modèle, ce qu'on doit pouvoir afficher, figer et versionner pour une expérience.
**Erreurs fréquentes** : oublier `revision=` (une mise à jour du dépôt changerait les résultats sans prévenir) ; ne pas passer `dtype` (le float32 est ici plus rapide, mais un autre modèle se chargerait en bfloat16) ; décoder toute la sortie de `generate()`, prompt compris ; mesurer la première génération, qui paie la mise en route (d'où la passe de chauffe) ; oublier `lm_head` dans le compte des couches denses parce que son poids est partagé ; compter les tokens après le chat template, qui ajoute le message système par défaut et les tokens de rôle.
**Variante** : les mêmes comptes pour Qwen2.5-0.5B-Instruct (sa configuration suffit, sur le device `meta`) donnent 21 tokens pour `PROMPT_12` (un autre vocabulaire, de 151 936 tokens), $24 \times 7 + 1 = 169$ couches denses et 14 912 384 paramètres par bloc : un bloc de Qwen2.5 pèse quatre fois celui de SmolLM2, à cause du MLP (896 → 4 864) et des biais de `q_proj`, `k_proj` et `v_proj`.

### Ex B4.13 — Prédire : dix générations du même prompt à température 0, 0,7 et 1,5 🔮
**Réponse** : a) **False** ; b) **False**. Dans l'expérience : à 0,7, **0** réponse identique à la gloutonne (« The doctor who shares rooms with Sherlock Holmes is Dr. John Watson. ») et **2** sur 10 qui nomment Watson, 10 réponses distinctes ; à 1,5, 10 réponses distinctes, 2 qui nomment Watson, des phrases sans suite.
**Démarche** : à $T = 0$, le décodage glouton est déterministe : dix fois la même réponse. À $T = 0{,}7$, on tire dans la distribution resserrée, mais **sans filtre** : à chaque token, la queue des 49 152 tokens garde une part du tirage, et pour un modèle de 135 M de paramètres, cette distribution est plate. Reproduire la réponse gloutonne demande de tirer le token le plus probable une quinzaine de fois de suite : si chacun l'est avec une probabilité de 0,8 après le passage à 0,7, la réponse entière l'est avec $0{,}8^{15} \approx 0{,}035$, soit en moyenne 0,35 réponse sur 10 ; dès la première bifurcation, la suite diverge. Sur 25 séries de 10 graines et quatre prompts voisins, jamais plus d'une réponse identique à la gloutonne, et jamais plus de 5 réponses sur 10 qui nomment Watson (2 en moyenne avec ce prompt). Le modèle « connaît » Watson au sens où c'est son token le plus probable au bon endroit, mais la masse de probabilité qui va ailleurs (« Dr. James », « Professor Moriarty », une phrase qui s'arrête avant le nom) l'emporte sur un tirage. À 1,5, la distribution est encore aplatie : des phrases sans suite, un vocabulaire décousu.
**Erreurs fréquentes** : penser que 0,7 est « presque glouton » parce que la température est inférieure à 1 ; confondre la probabilité d'un token et celle d'une réponse entière (un produit) ; croire que le modèle « sait » ce que dit sa réponse gloutonne, alors que le glouton ne montre que le mode de la distribution ; oublier que les réglages de Qwen2.5 (top-k 20, top-p 0,8) filtrent la queue et changent beaucoup ce tableau.
**Variante** : avec le filtre de Qwen2.5, `top_k=20, top_p=0.8`, à 0,7, les réponses se resserrent : sur les graines 0 à 9, une réponse sur 10 reproduit la gloutonne, 9 sont distinctes, et toujours 2 nomment Watson : le filtre coupe la queue de la distribution, mais la réponse dérive encore au moment du nom, où le modèle hésite vraiment entre plusieurs candidats.

### Ex B4.14 — Chat template, message système, arrêt et streaming avec un modèle instruct 📦
**Réponse** : tes réponses sont celles de la référence pour la question seule, les deux messages système, le second tour, l'arrêt au retour à la ligne et le streaming. Les sorties du notebook : sans message système, le gabarit ajoute celui de SmolLM2 ; « Answer in one short sentence. » donne « Sherlock Holmes is a fictional character from the Arthur Conan Doyle books. » ; la consigne en français n'est pas suivie (une réponse en anglais, et Phileas Fogg devient un personnage de Robert A. Heinlein : une hallucination d'un modèle de 135 M de paramètres) ; au second tour, le modèle nomme Watson ; la liste s'arrête à « Here are three stories about Sherlock Holmes, one per line: » ; le `TextIteratorStreamer` rend 49 morceaux, souvent un mot et son espace.
**Démarche** :
```python
def chat_14(tok, model, messages, max_new_tokens=48, stop_strings=None, streamer=None):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, stop_strings=stop_strings,
                             tokenizer=tok if stop_strings else None, streamer=streamer)
    reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for stop in stop_strings or ():
        reply = reply.split(stop)[0]                         # cut before the stop string
    return reply


def stream_14(tok, model, messages, max_new_tokens=48):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def run():
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)

    thread = threading.Thread(target=run)
    thread.start()
    pieces = list(streamer)                                  # each iteration waits for the next piece
    thread.join()
    return pieces
```
`apply_chat_template(..., return_dict=True)` rend `input_ids` et `attention_mask`, déjà avec les tokens spéciaux : on ne retokenise pas la chaîne. Le gabarit de SmolLM2 insère « You are a helpful AI assistant named SmolLM, trained by Hugging Face » quand la conversation n'a pas de message système (✏️ B4.4) : un produit fixe toujours le sien. La chaîne d'arrêt arrête la génération au token qui la complète ; ce token peut contenir davantage (ici « `\n\n` » pour la chaîne « `\n` »), d'où la coupure par `split` plutôt que `removesuffix`. Au second tour, toute la conversation repasse dans le prompt : le modèle n'a aucune mémoire entre deux appels, et le prefill grandit avec l'historique. Le streaming ne raccourcit pas la génération ; il fait lire la réponse pendant qu'elle s'écrit. `TextIteratorStreamer` est un itérateur bloquant : la boucle `for` (ici `list`) attend chaque morceau, d'où le fil d'exécution séparé pour `generate()`.
**Erreurs fréquentes** : `add_generation_prompt=False`, et le modèle continue le message de l'utilisateur ; passer `stop_strings` sans `tokenizer=tok` (une `ValueError`) ; retirer la chaîne d'arrêt par `removesuffix` alors que le dernier token en contient plus ; appeler `generate()` dans le fil principal avant de lire le streamer (la boucle ne commence qu'à la fin : un seul gros morceau) ; écrire « User: … Assistant: » à la main au lieu du gabarit.
**Variante** : avec `stop_strings=["."]` sur `ONE_14`, la réponse s'arrête à la première phrase : « Sherlock Holmes is a renowned British detective, renowned for his exceptional skills in solving mysteries and crimes », la première phrase de la réponse gloutonne, sans son point (le glouton produit le même début, avec ou sans chaîne d'arrêt).

### Ex B4.15 — Prompting pour extraire des informations de Holmes : zéro-shot, few-shot, JSON validé, vote majoritaire 📦
**Réponse** : sur les 8 passages, avec SmolLM2 : en zéro-shot, **aucun** JSON valide (du texte autour, une balise Markdown suivie de clés en trop, une année `"2020"` en chaîne) ; en few-shot avec deux exemples, 7 JSON valides sur 8 (0,88) et une précision moyenne des champs de **0,42** (name 0,38, place 0,50, year 0,38) ; le vote sur 5 tirages, 6 valides sur 8 (0,75) et 0,38.
**Démarche** :
```python
def parse_json_15(reply):
    text = reply.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data) != set(KEYS_15):
        return None
    if not all(data[k] is None or isinstance(data[k], str) for k in ("name", "place")):
        return None
    year = data["year"]
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        return None
    return {k: data[k] for k in KEYS_15}


def messages_15(text, examples=(), system=SYSTEM_15):
    messages = [{"role": "system", "content": system}]
    for example_text, answer in examples:
        messages += [{"role": "user", "content": example_text}, {"role": "assistant", "content": json.dumps(answer)}]
    return messages + [{"role": "user", "content": text}]


def vote_15(replies):
    parsed = [p for p in (parse_json_15(r) for r in replies) if p is not None]
    if not parsed:
        return None
    counts = Counter(tuple(p[k] for k in KEYS_15) for p in parsed)   # keeps the order of first sight
    return dict(zip(KEYS_15, counts.most_common(1)[0][0]))
```
Le zéro-shot rate la **forme** : un modèle de 135 M de paramètres annonce son JSON (« Here is the JSON response: »), l'entoure de balises, invente des clés. Deux exemples suffisent à fixer la forme, mais le modèle **recopie** leur contenu : l'année 1883 du second exemple revient quatre fois, « Spence Munro » et « Halifax » deux fois, et « Baker Street » devient un nom de personne. C'est le biais des exemples de la fiche (§204.4) : un petit modèle copie, il n'abstrait pas. Le vote ne compte que les réponses valides : il corrige un format raté par un tirage, pas une erreur que le modèle fait à chaque tirage. Avec 8 passages et 3 champs, un champ juste de plus vaut 4 points sur la moyenne : les écarts du tableau sont du même ordre que le hasard (fiche §204.5).
**Erreurs fréquentes** : `json.loads` sans `try` (une réponse mal formée arrête tout) ; accepter des clés en plus ou en moins, ou `"1888"` comme année ; `isinstance(True, int)` vaut `True` en Python : le booléen passe si on ne l'exclut pas d'abord ; écrire les exemples few-shot dans le message système au lieu de tours `user`/`assistant` ; voter sur les chaînes brutes, où deux JSON identiques à un espace près comptent comme deux réponses ; un vote qui garde le dernier ex aequo.
**Variante** : avec les trois exemples de `EXAMPLES_15` (l'ordre du tableau), SmolLM2 glouton passe à 0,46 (6 JSON valides sur 8) ; Qwen2.5-0.5B-Instruct fait 0,38 en zéro-shot et **0,63** en few-shot avec deux exemples (7 valides sur 8 ; il copie encore « Colonel Spence Munro » sur le dernier passage).

### Ex B4.16 — Sensibilité au prompt : cinq formulations, trois graines, ordre des exemples 🔬
**Réponse** : en mode rapide (SmolLM2, 6 passages, $T = 0{,}7$) : formulations, moyenne 0,163, écart-type entre les cinq formulations **0,036**, écart-type dû aux graines **0,136** ; ordres des exemples, moyenne 0,167, écart-type entre les trois ordres 0,091, dû aux graines 0,087. Protocole complet précalculé (Qwen2.5, 8 passages) : formulations, moyenne 0,550, entre 0,057, graines 0,050 ; ordres, moyenne 0,556, entre 0,063, graines 0,061.
**Démarche** :
```python
def accuracy_16(system, examples, seed):
    predictions = []
    for item in SNIPPETS_16:
        reply = MODEL_16.generate(messages_15(item["text"], examples, system), max_new_tokens=40, do_sample=True,
                                  temperature=0.7, seed=seed, cache=CACHE)
        predictions.append(parse_json_15(reply))
    return score_15(predictions, [item["gold"] for item in SNIPPETS_16])["all"]


def spread_16(table):
    table = np.asarray(table, dtype=float)
    return float(table.mean()), float(table.mean(axis=1).std()), float(table.std(axis=1).mean())
```
Avec SmolLM2 et des tirages à 0,7, la dispersion due aux graines (0,136) écrase celle entre formulations (0,036) : sur 6 passages, un tirage peut faire passer une case de 0 à 0,44. On ne peut donc rien conclure sur les formulations avec ce protocole, et c'est la première leçon : avant de comparer des prompts, on mesure le bruit de la mesure. Avec Qwen2.5, les deux dispersions sont du même ordre (0,057 contre 0,050), comme pour l'ordre des exemples (0,063 contre 0,061) : un écart de 5 à 6 points entre deux formulations, ou entre deux ordres, ne se distingue pas du hasard avec 8 passages et 3 graines (un passage vaut 4 points sur la moyenne). L'ordre des exemples compte aussi : en mode rapide, l'ordre (2, 0, 1) donne 0,278 et l'ordre (1, 2, 0) 0,056, un écart que les graines (0,087) n'expliquent qu'en partie. Pour départager, il faut davantage de passages (l'écart-type d'une moyenne baisse comme $1/\sqrt{n}$) et de graines, ou un décodage glouton, qui supprime la variance des tirages (fiche §204.5). La conclusion « le prompt compte » se chiffre ici ainsi : autant que le hasard d'un tirage, pas davantage.
**Erreurs fréquentes** : comparer deux formulations sur une seule graine ; calculer l'écart-type entre formulations sur toutes les cases du tableau (il mélange alors les deux sources) ; `ddof=1` (le contrat demande `ddof=0`, et avec 3 graines la différence est d'un facteur $\sqrt{3/2}$) ; garder « la meilleure formulation » trouvée sur ces 6 passages, qui ont servi à la choisir (c'est un jeu de validation, pas de test : ch. 8) ; oublier que l'ordre des exemples est une variable du prompt.
**Variante** : en glouton (sans graine), SmolLM2 obtient 0,389 ; 0,5 ; 0,389 ; 0,389 et 0,5 avec les cinq formulations sur les 6 passages : un écart-type de 0,054 entre formulations, qu'aucune graine ne brouille, et une précision bien meilleure qu'en tirant à 0,7 (0,16 en moyenne) ; pour une extraction, le glouton est le bon réglage.

### Ex B4.17 — chunk_text et BM25 : découper un roman et classer ses passages 🔨
**Réponse** : les 62 tests de l'exercice passent ; les douze nouvelles donnent les 1 045 chunks de référence ; les trois premiers passages de « violin music concert » ont les scores 6,55, 5,87 et 5,48 : chacun ne contient qu'**un** des trois mots, une seule fois (« concert » dans « The Red-Headed League », « violin » dans « The Noble Bachelor », « music » dans « The Copper Beeches »), et aucun ne parle vraiment de musique.
**Démarche** :
```python
def chunk_text(text, chunk_size=200, overlap=50, unit="word"):
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"expected chunk_size >= 1 and 0 <= overlap < chunk_size, got {chunk_size}, {overlap}")
    if unit == "word":
        units, sep = text.split(), " "
    elif unit == "char":
        units, sep = text, ""                                 # every character, spaces and newlines included
    else:
        raise ValueError(f"unit must be 'word' or 'char', not {unit!r}")
    chunks, start = [], 0
    while start < len(units):
        chunks.append(sep.join(units[start:start + chunk_size]))
        if start + chunk_size >= len(units):                  # this chunk reaches the end
            break
        start += chunk_size - overlap
    return chunks


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, corpus):
        if self.k1 < 0 or not 0 <= self.b <= 1:
            raise ValueError("expected k1 >= 0 and 0 <= b <= 1")
        docs = [list(doc) for doc in corpus]
        if not any(docs):
            raise ValueError("the corpus is empty")
        self.term_freqs_ = [dict(Counter(doc)) for doc in docs]
        n_t = Counter(term for tf in self.term_freqs_ for term in tf)
        self.n_docs_ = len(docs)
        self.doc_len_ = np.array([len(doc) for doc in docs])
        self.avgdl_ = float(self.doc_len_.mean())
        self.idf_ = {t: math.log(1 + (self.n_docs_ - n + 0.5) / (n + 0.5)) for t, n in n_t.items()}
        return self

    def get_scores(self, query):
        if not hasattr(self, "idf_"):
            raise ValueError("call fit(corpus) before get_scores")
        norm = self.k1 * (1 - self.b + self.b * self.doc_len_ / self.avgdl_)
        scores = np.zeros(self.n_docs_)
        for term in query:
            if term not in self.idf_:
                continue                                      # a term absent from the corpus adds 0
            for d, tf in enumerate(self.term_freqs_):
                f = tf.get(term, 0)
                if f:                                         # f = 0 adds 0: never 0/0
                    scores[d] += self.idf_[term] * f * (self.k1 + 1) / (f + norm[d])
        return scores
```
Une nouvelle de $n$ mots donne $\lceil (n - 20)/100 \rceil$ chunks (fiche §204.6) : la dernière fenêtre, souvent courte, compte pour une, et chaque nouvelle a la sienne, d'où 1 045 chunks pour 104 500 mots, plus que 104 500 / 100. Une fenêtre commence tous les 100 mots et en couvre 120 : tout extrait de 21 mots au plus tient entier dans celle qui commence au dernier multiple de 100 avant lui. Pour BM25, le seul piège est le dénominateur nul : avec `k1 = 0`, ou `b = 1` et un document vide, $f + k_1(1 - b + b|d|/\overline{|d|})$ vaut 0 quand $f = 0$ ; ne calculer que pour $f > 0$ évite le 0/0, et la formule ne change pas, puisque ces termes valent 0. Le premier passage gagne par la rareté : « concert » est le mot le plus rare des trois dans le corpus, donc de plus grand idf, et ses autres mots ne comptent pas. Une requête de trois mots rares ne trouve pas un passage « sur la musique » : elle trouve des passages qui contiennent ces mots.
**Erreurs fréquentes** : `text.split(" ")`, qui garde des chunks vides entre deux espaces et les retours à la ligne ; arrêter la boucle sur `start < len(units)` sans tester la fin, ce qui ajoute un chunk inclus dans le précédent ; normaliser les blancs en mode `"char"` ; ne compter les documents vides ni dans $N$ ni dans $\overline{|d|}$ ; l'idf original sans le « 1 + » (négatif pour un terme présent dans plus de la moitié des documents) ; `n_t` compté en occurrences au lieu de documents ; mettre la requête en minuscules dans `get_scores` (la tokenisation est celle de l'appelant) ; un second `fit` qui ajoute aux statistiques du premier.
**Variante** : avec $k_1 = 1{,}2$ (le défaut de Lucene) au lieu de 1,5, les trois mêmes passages gardent leur rang (6,55 ; 5,83 ; 5,47) : avec une seule occurrence, la contribution d'un terme vaut $\operatorname{idf} \cdot (k_1 + 1)/(1 + k_1(1 - b + b|d|/\overline{|d|}))$, exactement l'idf quand le passage a la longueur moyenne, quel que soit $k_1$ ; $k_1$ ne joue que sur les termes répétés.

### Ex B4.18 — cosine_top_k, fusion de classements (RRF), recall@k et MRR 🔨
**Réponse** : les 67 tests de l'exercice passent ; l'exemple de la fiche redonne [0, 1, 2] et (0,8944 ; 0,8944 ; 0,7071), la fusion [2, 7, 5, 9, 1, 4], et recall@1 = 1/6, recall@3 = 0,5, MRR = 4/9.
**Démarche** :
```python
def cosine_top_k(query_emb, doc_embs, k=5):
    q = np.asarray(query_emb, dtype=float)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    docs = np.asarray(doc_embs, dtype=float)
    if q.shape[1] != docs.shape[1] or not 1 <= k <= docs.shape[0]:
        raise ValueError(f"expected the same dimension and 1 <= k <= {docs.shape[0]}")
    sims = cosine_similarity_matrix(q, docs)                 # B2: a row of zeros has a similarity of 0
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]  # stable: ties keep the smaller index
    values = np.take_along_axis(sims, order, axis=1)
    return (order[0], values[0]) if single else (order, values)


def reciprocal_rank_fusion(rankings, k=60):
    if len(rankings) == 0 or k < 0:
        raise ValueError("expected at least one ranking and k >= 0")
    score = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[int(doc)] = score.get(int(doc), 0.0) + 1 / (k + rank)
    return np.array(sorted(score, key=lambda d: (-score[d], d)), dtype=int)


def recall_at_k(retrieved, relevant, k):
    if len(retrieved) == 0 or len(retrieved) != len(relevant) or k < 1:
        raise ValueError("expected one relevant set per query and k >= 1")
    ratios = []
    for ids, good in zip(retrieved, relevant):
        good = set(good)
        if not good:
            raise ValueError("a relevant set is empty")
        ratios.append(len(set(list(ids)[:k]) & good) / len(good))
    return float(np.mean(ratios))


def mean_reciprocal_rank(retrieved, relevant):
    if len(retrieved) == 0 or len(retrieved) != len(relevant):
        raise ValueError("expected one relevant set per query")
    total = 0.0
    for ids, good in zip(retrieved, relevant):
        good = set(good)
        total += next((1 / r for r, doc in enumerate(ids, start=1) if doc in good), 0.0)
    return total / len(retrieved)
```
Pour des vecteurs de norme 1, $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = 2 - 2\cos(\mathbf{a}, \mathbf{b})$ : trier par cosinus décroissant, c'est trier par distance croissante, ce que fait `NearestNeighbors(metric="cosine")` (avec la distance $1 - \cos$). Avec $k = 0$, la fusion de la fiche donnerait au document 2 le score $1/2 + 1/1 = 1{,}5$ et au document 7 $1 + 1/3 \approx 1{,}33$ : le premier rang pèse alors très lourd ; $k = 60$, fixé par Cormack et ses coauteurs lors d'un essai pilote, aplatit les écarts entre rangs, si bien qu'être présent dans deux listes compte plus qu'être premier dans une. Le recall@k divise par le nombre de pertinents : un ensemble vide le rendrait indéfini ; le MRR ne divise que par le rang, et une requête sans pertinent vaut naturellement 0.
**Erreurs fréquentes** : un cosinus maison sans `eps` (NaN pour un vecteur nul) ; `np.argsort(-sims)` sans `kind="stable"`, qui ne garantit pas le plus petit indice pour les ex aequo ; `argsort(sims)[::-1]`, qui met le **plus grand** indice d'abord ; les rangs à partir de 0 dans la fusion ; `np.array([])` pour une fusion vide (des flottants) ; `if not retrieved` sur un tableau 2-D (une `ValueError` de NumPy) ; diviser le recall par $k$ (c'est la precision@k de ✏️ B4.2) ; un MRR qui ne garde pas le premier pertinent.
**Variante** : la fusion de trois classements, [7, 2, 9, 4], [2, 5, 7, 1] et [9, 7, 3], avec $k = 60$, donne [7, 2, 9, 5, 3, 1, 4] : le document 7, présent dans les trois (rangs 1, 3 et 2), passe devant le document 2 (rangs 2 et 1), le 9 (rangs 3 et 1) devant le 5, et le 3, troisième d'une seule liste ($1/63$), devant le 1 et le 4, quatrièmes ($1/64$).

### Ex B4.19 — Prédire : sur des questions paraphrasées, BM25 ou recherche dense gagne-t-elle ? 🔮
**Réponse** : a) **False** ; b) **False** ; c) **True**. BM25 retrouve **1** des 10 questions paraphrasées dans ses 5 premiers résultats, la recherche dense **2**, l'hybride **3**. Sur 20 perturbations des embeddings (bruit de $10^{-3}$, puis norme 1), les trois comptes ne changent pas.
**Démarche** : sans mot-clé commun, BM25 ne voit que les petits mots de la question (« the », « what », « was »), présents dans presque tous les chunks et de faible idf : il classe d'autres chunks, qui partagent par hasard un mot plein avec la question ; il en retrouve pourtant une (H21, au rang 2). La recherche dense rapproche des paraphrases, mais pas à coup sûr : un chunk de 120 mots mêle dialogue et récit, la question emploie des mots modernes (« headpiece », « guard dog ») face à un anglais de 1892, et il faut battre 1 040 chunks pour entrer dans les 5 premiers. Elle classe le bon chunk premier pour deux questions, puis aux rangs 7, 8, 10 et 10 pour quatre autres, et au-delà de 20 pour les quatre dernières. La fusion fait mieux que les deux : la question que BM25 classe 2ᵉ et la recherche dense 7ᵉ passe 1ʳᵉ, et un chunk premier pour la seule recherche dense ($1/61 \approx 0{,}0164$) reste dans les 5 premiers quand peu de chunks sont bien classés par les deux. Mais elle ne sauve aucune question que les deux méthodes ratent : 3 sur 10 reste très loin des 7 ou 8 que l'on imagine en lisant « la recherche dense rapproche les paraphrases ».
**Erreurs fréquentes** : croire qu'un score BM25 « nul » signifie un rang nul (les petits mots donnent encore des scores, et le bon chunk peut passer par hasard) ; attendre d'un petit modèle d'embedding qu'il comprenne des paraphrases sur un texte ancien comme sur des phrases d'aujourd'hui ; penser que la fusion est la moyenne des deux méthodes (elle peut faire mieux que la meilleure) ; conclure sur 10 questions comme sur 1 000.
**Variante** : avec les 20 premiers résultats au lieu des 5 premiers, BM25 retrouve toujours **1** question paraphrasée, la recherche dense **6** et l'hybride **6** : la recherche dense « voit » plus de paraphrases qu'elle n'en classe en tête, et c'est le rang, pas la présence, qui manque ; un reclassement (un cross-encoder sur les 20 premiers, fiche §204.6) viserait exactement ce cas.

### Ex B4.20 — Recherche dense (MiniLM, e5) contre BM25 contre hybride sur 30 questions annotées 📦
**Réponse** : sur Holmes avec MiniLM, recall@5 de **0,533** pour BM25, **0,517** pour la recherche dense et **0,600** pour l'hybride sur les 30 questions (MRR 0,452, 0,500 et 0,523) ; sur les 20 directes, 0,750, 0,675 et 0,750 ; sur les 10 paraphrasées, 0,100, 0,200 et 0,300. Sur Verne, recall@5 de 0,8 pour BM25, 0,9 pour e5, 0,8 pour l'hybride avec e5, mais 0,5 pour MiniLM. e5 sur Holmes (précalculé) : recall@5 de 0,633 en dense, mais recall@1 de 0,150 seulement.
**Démarche** :
```python
def build_index_20(chunks, sentence_model, subset):
    R = mylearn.retrieval
    return {"chunks": list(chunks), "bm25": R.BM25().fit([tokenize(c) for c in chunks]), "model": sentence_model,
            "embeddings": wb.text.passage_embeddings(sentence_model, list(chunks), subset)}


def search_20(index, query, method="hybrid", k=5):
    R = mylearn.retrieval
    if method not in ("bm25", "dense", "hybrid"):
        raise ValueError(f"unknown method {method!r}")
    depth = max(k, DEPTH_20)
    if method in ("bm25", "hybrid"):
        bm = np.argsort(-np.asarray(index["bm25"].get_scores(tokenize(query))), kind="stable")[:depth]
    if method in ("dense", "hybrid"):
        q = index["model"].embed([query], kind="query")[0]  # « query: » for e5
        dense, _ = R.cosine_top_k(q, index["embeddings"], k=min(depth, len(index["chunks"])))
    if method == "bm25":
        return bm[:k].tolist()
    if method == "dense":
        return np.asarray(dense)[:k].tolist()
    return np.asarray(R.reciprocal_rank_fusion([bm[:DEPTH_20], dense[:DEPTH_20]], k=60))[:k].tolist()


def evaluate_20(index, items, method):
    R = mylearn.retrieval
    retrieved = [search_20(index, item["question"], method, DEPTH_20) for item in items]
    relevant = [wbrag.relevant_ids(index["chunks"], item) for item in items]
    return {"recall@1": float(R.recall_at_k(retrieved, relevant, 1)),
            "recall@5": float(R.recall_at_k(retrieved, relevant, 5)),
            "mrr": float(R.mean_reciprocal_rank(retrieved, relevant))}
```
Sur Holmes, l'hybride est le meilleur compromis : il garde la précision de BM25 sur les questions directes (noms propres, mots rares) et ajoute ce que la recherche dense retrouve sur les paraphrasées ; les trois méthodes restent loin de 1, car un chunk de 120 mots d'un récit à dialogues est un document difficile à résumer en un vecteur, et les questions paraphrasées sont dures par construction. Le recall@1 et le recall@5 ne classent pas toujours les méthodes de la même façon (e5 sur Holmes : le meilleur recall@5 dense, le plus faible recall@1) : le choix de $k$ fait partie du protocole, et c'est le $k$ que verra le modèle qui compte. Sur Verne, MiniLM, entraîné sur de l'anglais, découpe le français en sous-mots qu'il connaît mal : sa recherche dense s'effondre (0,5), alors qu'e5, multilingue, fait 0,9 ; BM25 n'a pas ce problème, il compare des mots. Avec 30 questions, l'écart-type d'un recall voisin de 0,5 est d'environ $\sqrt{0{,}25/30} \approx 0{,}09$ : les écarts de 7 points entre BM25 et l'hybride ne sont pas établis, ceux de 40 points entre e5 et MiniLM sur Verne le sont presque, même sur 10 questions.
**Erreurs fréquentes** : oublier `kind="query"` avec e5 (le préfixe « query: » manque, et les résultats se dégradent sans message : le premier défaut de 🐛 B4.22 en dépend) ; tokeniser la requête autrement que le corpus ; fusionner les 5 premiers au lieu des 20 (un chunk 6ᵉ dans les deux listes disparaît) ; un tri non stable de BM25, qui change l'ordre des ex aequo d'une machine à l'autre ; mesurer le MRR sur les 5 premiers seulement ; comparer deux méthodes à moins de 10 points d'écart sur 30 questions comme si l'écart était sûr.
**Variante** : le recall@10 de Holmes (les 30 questions) monte à 0,633 pour BM25, 0,650 pour la recherche dense et 0,667 pour l'hybride, et le recall@3, celui du RAG de B4.21, vaut 0,500, 0,517 et 0,517 : plus de passages montrés au modèle, c'est plus de chances d'y mettre le bon, au prix de tokens et de distraction (fiche §204.7).

### Ex B4.21 — Un RAG complet sur Holmes : récupération hybride, prompt avec citations, réponse d'un petit modèle 📦
**Réponse** : sur les 10 questions du mode rapide : le passage pertinent est parmi les 3 pour **7** questions, la réponse est juste pour **6**, la citation est vérifiée pour **3**, et **2** réponses seulement sont justes **et** vérifiées ; aucun « I don't know ». Sur les 30 questions (précalculé) : 16 passages trouvés, 15 réponses justes, 14 citations vérifiées, 8 réponses justes et vérifiées.
**Démarche** :
```python
import re


def rag_messages_21(question, passages):
    numbered = "\n".join(f"[{n}] {p}" for n, p in enumerate(passages, start=1))
    return ([{"role": "system", "content": SYSTEM_21}] + [dict(m) for m in EXAMPLE_21]
            + [{"role": "user", "content": f"Passages:\n{numbered}\n\nQuestion: {question}"}])


def parse_reply_21(reply):
    cited = []
    for match in re.finditer(r"\[(\d+)\]", reply):
        if int(match.group(1)) not in cited:
            cited.append(int(match.group(1)))
    return " ".join(re.sub(r"\[\d+\]", " ", reply).split()).strip(" ."), cited


def verify_21(answer, cited, passages):
    words = wbrag.content_words(answer)
    if not cited or not words or any(not 1 <= n <= len(passages) for n in cited):
        return False
    return any(words <= set(tokenize(passages[n - 1])) for n in cited)


def rag_21(question, index, llm, k=3):
    ids = search_20(index, question, "hybrid", k)
    passages = [index["chunks"][i] for i in ids]
    reply = llm.generate(rag_messages_21(question, passages), max_new_tokens=24, cache=CACHE)
    answer, cited = parse_reply_21(reply)
    return {"ids": ids, "passages": passages, "reply": reply, "answer": answer, "cited": cited,
            "verified": verify_21(answer, cited, passages)}
```
Le tableau sépare les étapes du diagnostic (fiche §204.7). Quand le bon passage manque (H05, H25, H28), la réponse est fausse : la recherche est la première cause d'échec. Quand il est là, Qwen2.5 trouve souvent la bonne réponse, mais il **cite mal** : « seventeen [2] » cite le mauvais passage, « Pawnbroker’s business [2] » aussi, « lawyer » et « Sir George Burnwell » ne citent rien ; la vérification les refuse, à raison. À l'inverse, « Fleet Street [3] » (attendu : *The Morning Chronicle*) est **vérifiée** : le passage cité contient bien « Fleet Street », mais il ne répond pas à la question. La règle de `verify_21` ne vérifie que l'appui de la réponse dans le passage : elle refuse une réponse juste mal citée, ou reformulée (« a pawnbroker's shop » quand le passage dit « pawnbroker’s business » : « shop » n'y est pas), et accepte une réponse fausse tirée d'un passage voisin. Une citation non vérifiée ne vaut rien ; une citation vérifiée ne prouve que l'appui, pas la justesse.
**Erreurs fréquentes** : numéroter les passages à partir de 0 alors que le modèle et la vérification comptent à partir de 1 ; mettre la question avant les passages (elle doit être la dernière chose lue, fiche §204.4) ; garder les citations dans la réponse comparée à la référence ; vérifier qu'**un** mot de la réponse est dans le passage cité (« the » suffirait) ; accepter un numéro qui ne désigne aucun passage ; juger une réponse juste sans vérifier sa citation, ou l'inverse : les deux mesures se lisent ensemble.
**Variante** : avec $k = 5$ passages, les mêmes 10 questions donnent 5 réponses justes, 3 citations vérifiées et 2 réponses justes et vérifiées : le passage de *The Morning Chronicle* entre dans le contexte (H05 devient juste et vérifiée), mais le modèle, distrait par deux passages de plus, perd « lawyer » (il répond « hydraulic engineer ») et « Bar of Gold » (« Upper Swandam Lane »). Plus de contexte n'est pas mieux pour un modèle de 0,5 B (fiche §204.7, « Lost in the Middle »).
