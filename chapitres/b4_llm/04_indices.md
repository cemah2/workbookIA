# B4 · LLM en pratique : Hugging Face, prompting, RAG et LoRA — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B4.Q1 — Modèle de base, modèle instruct, modèle de chat : lequel pour quoi ?

<details><summary>Indice 1</summary>

Relis « Trois étapes, trois sortes de modèles » et le paragraphe sur les noms du Hub (§204.1).

</details>
<details><summary>Indice 2</summary>

Le pré-entraînement donne un modèle qui continue n'importe quel texte ; le fine-tuning sur des consignes (SFT), puis l'alignement, en font un modèle qui suit des consignes. Les noms du Hub le disent souvent par un suffixe (`-Instruct`, `-it`, `-Chat`), sauf Qwen3, qui marque au contraire le modèle de base (`-Base`).

</details>
<details><summary>Indice 3</summary>

a) **B** : prolonger un texte brut est exactement ce que fait le pré-entraînement. Pour b), compare des milliers de milliards de tokens et des milliers de paires. Pour c), cherche le dépôt sans suffixe d'instruction, en tenant compte de l'exception de Qwen3. Pour d), quelle étape apprend à suivre une consigne ? Pour e), qu'est-ce qui peut suivre une question dans un texte du Web ?

</details>

### B4.Q2 — « Open weights » n'est pas « open source » : lire la licence d'un modèle

<details><summary>Indice 1</summary>

Relis « Licences : “open weights” n'est pas “open source” » et l'encadré 💼 qui le suit (§204.1).

</details>
<details><summary>Indice 2</summary>

Trois familles : les licences permissives (Apache 2.0, MIT), les licences de poids ouverts rédigées par l'éditeur (Llama, Gemma), avec leurs obligations propres, et les accès contrôlés. L'OSI demande, en plus des poids, une description des données et le code d'entraînement.

</details>
<details><summary>Indice 3</summary>

a) **Vrai** : Apache 2.0 permet l'usage commercial, la modification et la redistribution, avec l'avis de licence. Pour b), relis ce que la licence de Llama 3.2 exige du nom d'un dérivé redistribué. Pour c), compte ce que demande l'*Open Source AI Definition* 1.0. Pour d), relis la troisième puce. Pour e), qu'est-ce qu'une page de modèle ne donne presque jamais ?

</details>

### B4.Q3 — dtype, device et mémoire : ce qui décide si un modèle tient sur ton ordinateur

<details><summary>Indice 1</summary>

Relis « La mémoire des poids », l'encadré 🧮 sur float32, bfloat16 et la quantification, et « Vitesse sur CPU » (§204.2).

</details>
<details><summary>Indice 2</summary>

Mémoire $\approx N \times$ octets par paramètre : 4, 2 ou environ 0,5. Le KV cache vaut $2 \times L \times T \times n_{kv} d_h \times$ octets par valeur : regarde quelles grandeurs changent pendant une conversation. Pour la vitesse, relis les deux mesures de la fiche.

</details>
<details><summary>Indice 3</summary>

a) $3 \times 10^9 \times 2 = 6 \times 10^9$ octets, soit **6** Go. Pour b), refais le calcul avec 0,5 octet par paramètre. Pour c), combien de bits d'exposant ont les deux formats ? Pour d), quelle lettre de la formule du KV cache grandit avec la conversation ? Pour e), compare les temps en float32 et en bfloat16 mesurés par la fiche.

</details>

### B4.Q4 — Chat template : pourquoi ne pas écrire soi-même « User: … Assistant: … »

<details><summary>Indice 1</summary>

Relis « Rôles et chat template » et ses trois choses à retenir (§204.3).

</details>
<details><summary>Indice 2</summary>

Le gabarit Jinja est rangé avec le tokenizer, et chaque famille a le sien. `add_generation_prompt` ajoute le début du tour de l'assistant. Une chaîne obtenue avec `tokenize=False` contient déjà ses tokens spéciaux : le tokenizer ne doit pas en ajouter.

</details>
<details><summary>Indice 3</summary>

a) **A** : le chat template est rangé dans la configuration du tokenizer (`tok.chat_template`). Pour b), relis la première chose à retenir. Pour c), le tokenizer réécrit-il un texte, et le modèle a-t-il vu ce format pendant son entraînement ? Pour d), relis la dernière phrase avant « Une réponse complète ». Pour e), quel argument du tokenizer décide d'ajouter ou non des tokens spéciaux ?

</details>

### B4.Q5 — Glouton, température, top-p : lequel pour extraire une date, lequel pour écrire un poème ?

<details><summary>Indice 1</summary>

Relis « `generate()` et ses réglages » et le piège classique sur `generation_config.json` (§204.3), puis la température au ch. 22 et top-p en B3.

</details>
<details><summary>Indice 2</summary>

Le décodage glouton choisit chaque fois le token le plus probable : même entrée, même sortie. Le tirage donne des sorties variées, d'autant plus que la température est haute. Diviser les logits par une température plus petite que 1 écarte les probabilités. Top-p cumule les probabilités des tokens les plus probables jusqu'au seuil.

</details>
<details><summary>Indice 3</summary>

a) **C** : `do_sample=False` donne toujours la même réponse au même prompt. Pour b), il faut dix sorties **différentes** : quel mode les produit ? Pour c), d'où viennent les réglages qu'on ne passe pas ? Pour d), compare $\operatorname{softmax}(\mathbf{z}/0{,}5)$ et $\operatorname{softmax}(\mathbf{z})$ sur deux logits $(1, 0)$. Pour e), relis la définition de top-p en B3 : on compte des probabilités, pas des tokens.

</details>

### B4.Q6 — Zéro-shot, few-shot, chain-of-thought : vrai ou faux

<details><summary>Indice 1</summary>

Relis « Zéro-shot et few-shot », « Le raisonnement pas à pas » et « Un prompt est fragile » (§204.4).

</details>
<details><summary>Indice 2</summary>

L'apprentissage en contexte ne touche pas aux poids. Les exemples se placent comme des tours de conversation pour un modèle de chat. Le chain-of-thought a deux réserves : la taille du modèle et la fidélité du raisonnement écrit.

</details>
<details><summary>Indice 3</summary>

a) **Faux** : les exemples sont dans le prompt ; les poids ne changent pas (Brown et coll., 2020). Pour b), relis ce que montrent Lu et coll. Pour c), relis la fin de « Zéro-shot et few-shot ». Pour d) et e), relis les deux réserves, une par question.

</details>

### B4.Q7 — Pourquoi un LLM invente : hallucination et token le plus probable

<details><summary>Indice 1</summary>

Relis « Ce que fait vraiment un LLM » (§204.1) et « Les hallucinations » (§204.10).

</details>
<details><summary>Indice 2</summary>

Un LLM rend, à chaque pas, une distribution sur le token suivant ; il continue le texte de la façon la plus plausible. Une question à présupposé faux est une cause classique d'hallucination. Les parades réduisent le risque ; aucune ne le supprime.

</details>
<details><summary>Indice 3</summary>

a) **A** : une distribution sur tout le vocabulaire, puis un choix ou un tirage, un token après l'autre. Pour b), relis la dernière phrase des parades. Pour c), un fait inventé peut-il avoir une « bonne » année ? Pour d), un RAG garantit-il que le modèle lise bien ses passages ? Pour e), qu'a récompensé le pré-entraînement : dire « je ne sais pas » ou prédire le texte ?

</details>

### B4.Q8 — Évaluer un LLM : exact match, juge LLM, contamination des benchmarks

<details><summary>Indice 1</summary>

Relis « Le jeu de test », « Les métriques », « Le juge LLM » et « La contamination » (§204.5).

</details>
<details><summary>Indice 2</summary>

Normalise d'abord les deux textes (minuscules, sans ponctuation ni articles, espaces réduites), puis compare-les en entier (exact match) ou cherche la référence dans la réponse (« contient »). L'écart-type d'une proportion $\hat{p}$ sur $n$ questions vaut $\sqrt{\hat{p}(1 - \hat{p})/n}$.

</details>
<details><summary>Indice 3</summary>

a) « baker street. » devient « baker street », égal à la référence normalisée : **1**. Pour b), « baker street » apparaît-il dans « he lives in baker street london » ? Pour c), calcule $\sqrt{0{,}8 \times 0{,}2 / 100}$. Pour d), quelle parade de la fiche vise le biais de position ? Pour e), relis les indices de contamination.

</details>

### B4.Q9 — BM25 ou recherche dense : qui retrouve « Watson » et qui retrouve « le docteur » ?

<details><summary>Indice 1</summary>

Relis les quatre idées de BM25, « La recherche dense » et « La recherche hybride et la fusion RRF » (§204.6).

</details>
<details><summary>Indice 2</summary>

BM25 ne compte que les mots de la requête présents dans le document. Un embedding place près l'une de l'autre des expressions employées dans des contextes semblables, mais connaît mal un code ou un nom rare. Un document présent dans deux classements additionne deux termes $1/(60 + \text{rang})$.

</details>
<details><summary>Indice 3</summary>

a) **B** : seul le passage 0 contient le mot « watson » ; le passage 1 n'a aucun mot commun avec la requête. Pour b), relis la première des quatre idées. Pour c), d'où vient la proximité de deux embeddings ? Pour d), relis le paragraphe « Les forces s'inversent ». Pour e), calcule $1/(60 + 3)$ pour chacun des deux classements, puis additionne.

</details>

### B4.Q10 — RAG ou fine-tuning pour ajouter des connaissances ?

<details><summary>Indice 1</summary>

Relis « RAG ou fine-tuning ? » (§204.7) et « Les quatre étapes » du RAG.

</details>
<details><summary>Indice 2</summary>

Le RAG ajoute des **connaissances** qui changent, avec leurs sources et leurs droits d'accès ; le fine-tuning change un **comportement** (format, style, tâche répétitive). L'index d'un RAG se construit une fois et se met à jour quand les documents changent.

</details>
<details><summary>Indice 3</summary>

a) **B** : une documentation qui change chaque semaine se lit par une recherche, sans réentraîner. Pour b), un format est-il une connaissance ou un comportement ? Commence par le moins cher. Pour c), où se fait le filtre des droits dans un RAG ? Pour d), relis la phrase sur les faits précis. Pour e), qu'est-ce qui contient encore l'ancienne procédure ?

</details>

### B4.Q11 — LoRA : ce qui est gelé, ce qui est appris, ce que l'on fusionne

<details><summary>Indice 1</summary>

Relis « LoRA », « Ce que dit chaque morceau » et « Fusionner l'adaptateur » (§204.8).

</details>
<details><summary>Indice 2</summary>

$\mathbf{W}' = \mathbf{W} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$, $\mathbf{W}$ gelée, $\mathbf{B}_0 = 0$ ; une matrice $d \times k$ reçoit $r(d + k)$ paramètres entraînables. La fusion calcule $\mathbf{W}'$ une fois pour toutes.

</details>
<details><summary>Indice 3</summary>

a) **B** : $\mathbf{W}$ est gelée ; seuls $\mathbf{A}$ et $\mathbf{B}$ reçoivent des gradients. Pour b), que vaut $\mathbf{B}_0\mathbf{A}_0$ ? Pour c), calcule $4 \times (512 + 512)$. Pour d), que devient la branche après la fusion ? Pour e), calcule $\alpha/r$ avec les valeurs de la question.

</details>

### B4.Q12 — Injection de prompt : quand un document récupéré devient une instruction

<details><summary>Indice 1</summary>

Relis « L'injection de prompt », ses parades et l'encadré 🕰️ sur l'OWASP (§204.10).

</details>
<details><summary>Indice 2</summary>

Directe : l'utilisateur tape l'instruction ; indirecte : elle arrive par un document récupéré. Aucune parade n'est complète ; la plus efficace contre les dégâts est de limiter ce que le modèle peut faire.

</details>
<details><summary>Indice 3</summary>

a) **B** : l'attaquant cache ses instructions dans un document que le système va lire. Pour b), relis la phrase qui ouvre la liste des parades. Pour c), laquelle des trois options empêche une action, même si le modèle obéit ? Pour d), l'auteur de l'e-mail a-t-il besoin de parler à l'assistant ? Pour e), relis LLM01 dans l'encadré.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### B4.R1 — Ch. B3 : ce que le masque causal et le KV cache imposent à la génération

<details><summary>Indice 1</summary>

Relis en B3 le masque causal, le KV cache et le prefill, puis le 🔁 Rappel B3 de §204.3 et la formule du KV cache (§204.2).

</details>
<details><summary>Indice 2</summary>

La première passe lit tout le prompt (le prefill) et donne le premier token ; ensuite, chaque token passe seul, avec le cache. Un token ne peut pas être produit avant le précédent. KV cache par token $= 2 \times L \times n_{kv} d_h \times$ octets par valeur.

</details>
<details><summary>Indice 3</summary>

a) Une passe sur le prompt donne le token 1, puis une passe par token suivant : $1 + 99 =$ **100**. Pour b), le token 2 peut-il être choisi avant que le token 1 soit connu ? Pour c), calcule $2 \times 24 \times (2 \times 64) \times 2$. Pour d), qu'est-ce qui fixe le nombre de passes avant ? Pour e), le masque causal laisse-t-il la position 50 voir la 49, la 51 ?

</details>

### B4.R2 — Ch. B1 : extraction de features ou fine-tuning : que change le nombre d'exemples ?

<details><summary>Indice 1</summary>

Relis en B1 l'extraction de features, le fine-tuning et l'oubli catastrophique, puis le 🔁 Rappel B1 de §204.8.

</details>
<details><summary>Indice 2</summary>

Avec peu d'exemples, un fine-tuning complet a beaucoup de paramètres libres pour peu de données : overfitting et oubli. L'extraction de features n'entraîne qu'une tête ; LoRA place quelques paramètres dans les couches.

</details>
<details><summary>Indice 3</summary>

a) **B** : avec quelques dizaines d'exemples par classe, la tête sur un modèle gelé battait souvent le fine-tuning complet. Pour b), que gèle l'extraction de features ? Pour c), où LoRA ajoute-t-il ses matrices ? Pour d), que risque un fine-tuning complet avec 20 exemples ? Pour e), relis la définition de l'oubli catastrophique du glossaire (B1).

</details>

### B4.R3 — Ch. 13 : les k plus proches voisins et le choix de la distance

<details><summary>Indice 1</summary>

Relis au ch. 13 les k plus proches voisins et leurs distances, puis « La recherche dense » (§204.6) et sa formule $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = 2 - 2\cos(\mathbf{a}, \mathbf{b})$.

</details>
<details><summary>Indice 2</summary>

$\cos(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} / (\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert)$ ; la distance euclidienne est la racine de la somme des carrés des écarts. Un produit scalaire de dimension $d$ demande $d$ multiplications.

</details>
<details><summary>Indice 3</summary>

a) $\mathbf{a} \cdot \mathbf{b} = 18 + 32 = 50$, $\lVert\mathbf{a}\rVert = 5$, $\lVert\mathbf{b}\rVert = 10$ : $\cos = 50/50 =$ **1**. Pour b), calcule $\sqrt{(6 - 3)^2 + (8 - 4)^2}$. Pour c), applique la formule de la fiche. Pour d), la formule fait-elle de la distance une fonction croissante ou décroissante du cosinus ? Pour e), multiplie le nombre de chunks par la dimension.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B4.1 — Mémoire d'un modèle : paramètres, dtype et KV cache — tient-il sur ton ordinateur ? ✏️

<details><summary>Indice 1</summary>

Relis « La mémoire des poids », la formule du KV cache et « Vitesse sur CPU » (§204.2), et la GQA en B3.

</details>
<details><summary>Indice 2</summary>

La table des tokens a $V \times d$ paramètres, comptée une fois avec des poids liés. Mémoire des poids $= N \times$ octets par paramètre, à convertir en Mo de $10^6$ octets. KV cache par token $= 2 \times L \times n_{kv} d_h \times$ octets par valeur, avec les **têtes de clés et valeurs** (3), pas les têtes de requêtes. Pour i), retire d'abord les poids des 2 Go. Pour j), un temps par token proportionnel à la mémoire donne une vitesse inversement proportionnelle.

</details>
<details><summary>Indice 3</summary>

a) $49\,152 \times 576 =$ **28 311 552**. Pour b), divise a) par $N$ et multiplie par 100. Pour c), d) et e), multiplie $N$ par 4, 2 et 0,5, puis divise par $10^6$. Pour f), calcule $2 \times 30 \times (3 \times 64) \times 2$. Pour g), multiplie f) par 8 192 tokens. Pour h), compare g) et d). Pour i), calcule $(2 \times 10^9 - 2N) / (\text{f)} \times 8\,192)$ et garde la partie entière. Pour j), calcule $\frac{64}{8{,}4} \times \frac{1\,976\,131\,328}{4N}$. Pour k), compare c) à 1 000 Mo.

</details>

### Ex B4.2 — Recall@k et MRR pour cinq requêtes ✏️

<details><summary>Indice 1</summary>

Relis « Mesurer une recherche : recall@k et MRR » et son tableau (§204.6).

</details>
<details><summary>Indice 2</summary>

Pour chaque question : recall@$k$ $=$ (pertinents parmi les $k$ premiers) / (nombre de pertinents) ; $1/\text{rang}$ du **premier** pertinent, ou 0 s'il n'y en a aucun. Puis la moyenne sur les cinq questions. Le taux de succès@3 ne regarde que la présence d'**au moins un** pertinent.

</details>
<details><summary>Indice 3</summary>

a) Seule la question 2 a un pertinent au rang 1 (le document 0), sur deux pertinents : recall@1 $= (0 + 0 + \frac{1}{2} + 0 + 0)/5 =$ **0,100**. Pour b) et c), dresse une ligne par question : combien de ses pertinents sont parmi les 3, puis les 5 premiers, divisé par son nombre de pertinents. Pour d), cherche le rang du premier pertinent de chaque question. Pour e), compte les questions qui ont au moins un pertinent dans le top-3. Pour f), compare b) et e). Pour g), compte les recall@5 égaux à 1 dans ta ligne de c). Pour h), compare, pour une question, sa part de pertinents au rang 1 et l'inverse du rang de son premier pertinent.

</details>

### Ex B4.3 — LoRA : paramètres entraînables d'une couche 896×896 de rang 8, puis de tout un modèle ✏️

<details><summary>Indice 1</summary>

Relis « LoRA », son exemple chiffré sur `q_proj` et le compte de « LoRA avec peft » (§204.8).

</details>
<details><summary>Indice 2</summary>

Une matrice $d \times k$ (sorties × entrées) reçoit $r(d + k)$ paramètres. Fais le compte couche par couche, puis multiplie par 24. `print_trainable_parameters` divise par le total, adaptateurs compris. « Au moins autant » se traduit par l'inégalité $r(d + k) \ge dk$.

</details>
<details><summary>Indice 3</summary>

a) $8 \times (896 + 896) =$ **14 336**. Pour b), divise par $896^2$. Pour c), calcule $8 \times (128 + 896)$, puis divise par $128 \times 896$. Pour d), additionne les quatre matrices. Pour e), multiplie par 24. Pour f), calcule $100 \times \text{e)} / (494\,032\,768 + \text{e)})$. Pour g), ajoute à d) trois matrices du MLP, $896 \to 4\,864$ ou l'inverse, puis multiplie par 24. Pour h), regarde comment $r(d + k)$ dépend de $r$. Pour i), multiplie e) par 4 octets. Pour j), résous $r \times 1\,792 \ge 896^2$. Pour k), compare b) et c).

</details>

### Ex B4.4 — Un chat template à la main : rôles, tokens spéciaux et prompt de génération ✏️

<details><summary>Indice 1</summary>

Relis « Rôles et chat template » et ses trois choses à retenir (§204.3), puis lis le gabarit ligne par ligne, en particulier la condition `if loop.first and messages[0]['role'] != 'system'`.

</details>
<details><summary>Indice 2</summary>

Écris d'abord la chaîne, puis compte. Chaque message devient `<|im_start|>` + rôle + `\n` + contenu + `<|im_end|>` + `\n` : 4 tokens de structure, plus le rôle (1 ou 2 tokens) et le contenu. Le prompt de génération ajoute `<|im_start|>assistant\n`. Sans message système, le gabarit en ajoute un en tête.

</details>
<details><summary>Indice 3</summary>

a) Sans message système, le gabarit ajoute le tour système par défaut : 2 tokens spéciaux pour lui, 2 pour U, 1 pour le début du tour de l'assistant, **5** en tout. Pour b), compte le tour système par défaut ($1 + 1 + 1 + 16 + 1 + 1$), le tour de U, puis le prompt de génération ($1 + 2 + 1$). Pour c), le gabarit ajoute-t-il encore le message par défaut quand S est là ? Pour d), retire le prompt de génération de c). Pour e), ajoute les tours de A et de U2 (attention au rôle `assistant`). Pour f), compare chaque option à ta chaîne de c). Pour g), cherche un morceau du tableau marqué « token spécial ». Pour h), relis la boucle `for`.

</details>

### Ex B4.5 — BM25 à la main : trois documents, une requête de deux mots ✏️

<details><summary>Indice 1</summary>

Relis « De TF-IDF à BM25 », ses quatre idées et l'exemple chiffré (§204.6).

</details>
<details><summary>Indice 2</summary>

Calcule une fois pour toutes $N$, $\overline{|d|}$, $n_t$ et l'idf de chaque terme de la requête. Pour chaque document et chaque terme présent, calcule le dénominateur $f + k_1(1 - b + b\,|d|/\overline{|d|})$, puis la contribution $\operatorname{idf} \times f(k_1 + 1)/\text{dénominateur}$ ; additionne les contributions des deux termes.

</details>
<details><summary>Indice 3</summary>

a) Les longueurs sont 4, 3 et 8 : $\overline{|d|} = 15/3 =$ **5**. Pour b), « nautilus » est dans les trois documents : calcule $\ln(1 + 0{,}5/3{,}5)$. Pour c), le doc 1 a $f = 1$ et $|d|/\overline{|d|} = 0{,}6$. Pour d), le doc 2 a $f = 2$ et $|d|/\overline{|d|} = 1{,}6$. Pour e), le doc 0 contient les deux termes, avec $|d|/\overline{|d|} = 0{,}8$ ; « nemo » n'est que dans un document. Pour f), trie c), d) et e). Pour g), refais c) et d) avec un dénominateur $f + k_1$. Pour h), que vaut la fraction quand $k_1 = 0$ et $f > 0$ ? Pour i), calcule l'idf original des deux termes, puis le score du doc 0, et compare-le à 0.

</details>

### Ex B4.6 — LoRA : ΔW = BA de rang au plus r, B nulle au départ, gradients de A et de B, facteur α/r ∂

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur le rang et « LoRA » (§204.8), le rappel de l'énoncé sur le gradient d'un produit matrice-vecteur, et la règle de la chaîne du ch. 18.

</details>
<details><summary>Indice 2</summary>

Pour 1, la colonne $j$ de $\mathbf{B}\mathbf{A}$ vaut $\mathbf{B}$ fois la colonne $j$ de $\mathbf{A}$. Pour 3, la branche est $\mathbf{h}_2 = s\,\mathbf{B}\mathbf{u}$ avec $\mathbf{u} = \mathbf{A}\mathbf{x}$ : applique le rappel à $\mathbf{B}$ (avec $s\,\mathbf{u}$), puis à $\mathbf{u}$, puis à $\mathbf{A}$. Pour 4 et 5, remplace $\mathbf{B}$ par 0 dans tes formules. Pour 6 (c), la variance d'une somme de $r$ termes indépendants est la somme de leurs variances (ch. 2).

</details>
<details><summary>Indice 3</summary>

1. La colonne $j$ de $\mathbf{B}\mathbf{A}$ est $\sum_{i=1}^{r} A_{ij}\,\mathbf{b}_i$, une combinaison des $r$ colonnes $\mathbf{b}_1, \dots, \mathbf{b}_r$ de $\mathbf{B}$ : toutes les colonnes vivent dans un espace engendré par $r$ vecteurs, donc le rang est au plus $r$ ; multiplier par $s$ ne le change pas. Pour la condition, compare $r(d + k)$ et $dk$. Pour 2, remplace $\mathbf{B}$ par 0 dans $\mathbf{h}$, puis dans les gradients de la question 3. Pour 3, pose $\partial L/\partial \mathbf{B} = s\,\mathbf{g}\,\mathbf{u}^\top$ et $\partial L/\partial \mathbf{u} = s\,\mathbf{B}^\top\mathbf{g}$, puis applique le rappel à $\mathbf{u} = \mathbf{A}\mathbf{x}$. Pour 4, écris $\mathbf{B}_1 = \mathbf{B}_0 - \eta\,\partial L/\partial \mathbf{B}$ et multiplie par $s\,\mathbf{A}_0$. Pour 5, commence par $\mathbf{A}_0\mathbf{x}$ et $s$. Pour 6, (a) regarde la puissance de $s$ dans la question 4 ; (b) si $\Delta\mathbf{B}$ ne dépend plus de $s$, combien de facteurs $s$ restent ? (c) l'écart-type d'une somme de $r$ termes de même variance grandit comme une puissance de $r$ : laquelle ?

</details>

### Ex B4.7 — Fermi : coût et latence d'un assistant RAG, API facturée au token contre petit modèle local 🧮

<details><summary>Indice 1</summary>

Relis « Les coûts » (§204.10), « Vitesse sur CPU » (§204.2), les quatre étapes du RAG (§204.7) et le prix d'un token en B2 (§202.6).

</details>
<details><summary>Indice 2</summary>

Tokens par question $=$ consignes $+ k \times$ (mots par passage $\times$ fertilité) $+$ question ; par mois, multiplie par le nombre de questions. Coût $=$ millions de tokens d'entrée $\times$ prix d'entrée $+$ millions de tokens de sortie $\times$ prix de sortie. Latence locale $=$ tokens d'entrée / vitesse de prefill $+$ tokens de sortie / vitesse de génération. Capacité $= 3\,600$ s / latence.

</details>
<details><summary>Indice 3</summary>

1. Un passage fait $200 \times 1{,}3 = 260$ tokens : l'entrée vaut $200 + 5 \times 260 + 40 = 1\,540$ tokens, la sortie 120 ; il y a $3\,000 \times 21 = 63\,000$ questions par mois. Pour 2, applique les deux tarifs aux totaux mensuels. Pour 3, API : $0{,}5 + 120/60$ ; local : $1\,540/150 + 120/8$. Pour 4, divise 3 600 s par la latence locale, puis le débit demandé (questions par heure) par ce résultat ; une VM de 8 cœurs a une latence 4 fois plus petite ; coût $= 0{,}40 \times$ heures. Pour 5, refais l'entrée avec 3 passages. Pour 6 et 7, pense aux données, à la disponibilité et à ce que l'on n'a pas compté (ingénieurs, essais, embeddings, pics).

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B4.8 — Le RAG expliqué en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis « L'idée » et « Les quatre étapes » (§204.7), et le point 7 de « L'essentiel en 10 lignes ».

</details>
<details><summary>Indice 2</summary>

Une image : un examen à livre ouvert. Le modèle est l'élève, qui sait écrire mais ne connaît pas tes documents ; le bibliothécaire lui apporte les bonnes pages ; on lui demande de répondre avec ces pages, de dire d'où vient chaque réponse, et d'avouer quand les pages ne disent rien.

</details>
<details><summary>Indice 3</summary>

Commence par : « Un modèle de langue ne sait que ce qu'il a lu pendant son entraînement, et quand il ne sait pas, il invente une réponse qui sonne juste. » Puis : on cherche d'abord dans tes documents les passages qui parlent de la question ; on les lui donne avec la consigne de ne répondre qu'avec eux ; il cite le passage utilisé, que l'on peut vérifier ; pour mettre ses connaissances à jour, il suffit de changer les documents.

</details>

### Ex B4.9 — Un assistant RH qui répond sur les salaires : données personnelles, hallucinations, injection, transparence ⚖️

<details><summary>Indice 1</summary>

Relis « Les données », « L'injection de prompt » et « L'AI Act » (§204.10), le tableau de diagnostic et les droits d'accès du RAG (§204.7), et les données personnelles du ch. 29.

</details>
<details><summary>Indice 2</summary>

Classe tes réponses en trois couches : les **données** (ce qui entre dans l'index, qui peut le lire, où il part), le **modèle** (ce qu'il peut inventer, ce qu'il ne doit pas calculer), le **cadre** (RGPD, AI Act, directive sur la transparence des rémunérations, représentants du personnel). La parade la plus forte contre une fuite ou une injection n'est jamais une consigne : c'est ce que le modèle ne voit pas et ne peut pas faire.

</details>
<details><summary>Indice 3</summary>

1. Les bulletins, contrats et entretiens sont des données personnelles (art. 4 du RGPD : toute information sur une personne identifiable) ; la convention collective et les grilles par poste ne le sont pas en elles-mêmes. Pour la base légale, pense à l'exécution du contrat de travail et à l'intérêt légitime, et au déséquilibre entre employeur et salarié pour le consentement. Pour 2, le modèle peut-il révéler un passage qu'il n'a jamais reçu ? Pour 3, pense à un montant inventé et à une règle de congés fausse ; et au calcul par une requête à la base de paie. Pour 4, qui a écrit le document, et que lit le modèle ? Pour 5, relis l'article 50 et le point 4 de l'annexe III : l'outil décide-t-il, ou informe-t-il ? Pour 6, cherche la date de transposition de la directive 2023/970, et le risque d'une moyenne sur deux personnes. Pour 7, mets dans la balance les transferts hors de l'Union, la qualité des réponses et l'hébergement.

</details>

### Ex B4.10 — Fine-tuner sur des romans : domaine public, œuvres protégées et droit de retrait (opt-out) ⚖️

<details><summary>Indice 1</summary>

Relis « Licences et droit d'auteur » et « L'AI Act » (§204.10), le droit d'auteur et l'opt-out du ch. 28, et les licences du ch. 29.

</details>
<details><summary>Indice 2</summary>

Sépare trois questions : la **copie d'entraînement** (domaine public, exception de fouille de textes et ses conditions), les **sorties** (une recopie reste une reproduction), le **modèle publié** (licence du modèle de base, obligations de l'AI Act). Pour chaque source, demande-toi : l'œuvre est-elle protégée ? l'accès est-il licite ? le titulaire s'est-il opposé ?

</details>
<details><summary>Indice 3</summary>

1. Dans l'Union, les droits patrimoniaux durent 70 ans après la mort de l'auteur, à compter du 1ᵉʳ janvier suivant : Conan Doyle est libre depuis le 1ᵉʳ janvier 2001. Fais le même calcul pour Verne, puis demande-toi qui est l'auteur d'une traduction. Pour 2, relis l'article 4 (accès licite, réserve des titulaires, conservation limitée) et applique-le aux livres achetés, puis au site qui s'oppose. Pour 3, distingue l'idée ou le style et l'expression ; relis le jugement GEMA contre OpenAI. Pour 4, compare le calcul d'un LoRA à un tiers du pré-entraînement ; pense ensuite au marquage des textes générés. Pour 5, relis la licence Apache 2.0, puis celle de Llama 3.2 (§204.1). Pour 6, une ligne par source, avec son statut.

</details>

### Ex B4.11 — Hu et coll. (2021) : « LoRA: Low-Rank Adaptation of Large Language Models » 📄

<details><summary>Indice 1</summary>

Lis d'abord le résumé et la figure 1 ; la fiche (§204.8) suit l'article pour la formule, l'initialisation et $\alpha$. Les chiffres sont dans les tableaux 1, 4, 5, 6, 7 et 12 et dans les notes 4 et 5.

</details>
<details><summary>Indice 2</summary>

Pour les comptes, reprends ta méthode de ✏️ B4.3 : GPT-3 a 96 couches de $d_{\text{model}} = 12\,288$, et l'article traite chaque matrice d'attention comme $d_{\text{model}} \times d_{\text{model}}$. Un facteur de réduction est un rapport ; une « accélération » de l'entraînement se lit sur le temps par token, l'inverse du débit.

</details>
<details><summary>Indice 3</summary>

1. Chaque tâche demanderait sa copie complète de 175 milliards de paramètres ; les adaptateurs ajoutent des couches traitées l'une après l'autre (de la latence), l'optimisation du prompt est difficile et prend une partie de la séquence. Pour le tableau 1, regarde la colonne où le batch et la séquence sont les plus petits. Pour 2, relis le deuxième paragraphe de l'introduction. Pour 3, relis les deux paragraphes qui suivent l'équation (3). Pour 4, calcule $2 \times 96 \times 12\,288 \times 8$, puis multiplie par 2 octets. Pour 5, divise 175 255,8 millions par le budget de 4 ; calcule le rapport et l'écart relatif des deux débits. Pour 6, relis la fin de l'annexe D.4. Pour 7, relis la phrase sous le tableau 6 et la note 6. Pour 8, divise $\lVert\Delta\mathbf{W}_q\rVert_F$ par $\lVert\mathbf{U}^\top\mathbf{W}_q\mathbf{V}^\top\rVert_F$ pour $r = 4$. Pour 9, compare à la fiche : `target_modules`, `lora_alpha`, rsLoRA, DoRA, QLoRA.

</details>

<a id="entretien"></a>

## 💼 Entretien

### B4.E1 — Comment réduis-tu les hallucinations d'un assistant fondé sur un LLM ?

<details><summary>Indice 1</summary>

Relis « Les hallucinations » (§204.10), « Évaluer et diagnostiquer » (§204.7) et « Le jeu de test » (§204.5).

</details>
<details><summary>Indice 2</summary>

Trois temps : comprendre la cause (le modèle continue le plausible), réduire (ancrage par RAG, citations vérifiées, abstention, vérification par programme, humain dans la boucle), **mesurer** (questions pièges, taux d'invention avant et après).

</details>
<details><summary>Indice 3</summary>

Commence par : « Un LLM produit la suite la plus plausible, pas la plus vraie : je ne cherche pas à le supprimer, je l'encadre et je le mesure. » Puis : d'abord vérifier que les mauvaises réponses ne viennent pas de la recherche ; ancrer dans des documents, exiger et vérifier les citations ; autoriser le « je ne sais pas » ; calculer par programme ce qui peut l'être ; un jeu de test avec des questions sans réponse.

</details>

### B4.E2 — RAG ou fine-tuning : comment choisis-tu ?

<details><summary>Indice 1</summary>

Relis « RAG ou fine-tuning ? » (§204.7) et « Que montrer au modèle pendant l'entraînement ? » (§204.8).

</details>
<details><summary>Indice 2</summary>

Sépare ce que le système doit **savoir** (le catalogue : il change, il se cite, il se filtre) et la façon dont il doit **répondre** (le ton : un comportement). Commence par le moins cher (prompt), mesure, puis ajoute.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je sépare les connaissances et le comportement. » Puis : le catalogue par un RAG (mises à jour sans réentraîner, citations, filtres) ; le ton d'abord par le prompt et quelques exemples, puis un LoRA sur des réponses de la marque si le prompt ne suffit pas ; les deux se combinent ; on décide sur un jeu de test, chiffres à l'appui.

</details>

### B4.E3 — Explique LoRA et pourquoi c'est si économique

<details><summary>Indice 1</summary>

Relis « LoRA », « Ce que dit chaque morceau », « Fusionner l'adaptateur » et « Le coût d'un fine-tuning complet » (§204.8).

</details>
<details><summary>Indice 2</summary>

La formule, puis les trois économies : les paramètres entraînables ($r(d + k)$ au lieu de $dk$), la mémoire d'entraînement (pas d'états d'Adam pour $\mathbf{W}$), le stockage (un petit adaptateur par tâche). Puis la fusion, et ce qu'on perd.

</details>
<details><summary>Indice 3</summary>

Commence par : « On gèle les poids pré-entraînés et on apprend seulement une modification de rang faible, $\Delta\mathbf{W} = \frac{\alpha}{r}\mathbf{B}\mathbf{A}$. » Puis : $\mathbf{B}$ nulle au départ ; un exemple de compte (une matrice $4\,096 \times 4\,096$ avec $r = 8$) ; 16 octets par paramètre entraîné contre 2 pour un paramètre gelé ; un adaptateur de quelques Mo ; la fusion sans latence ; les limites (tâches très éloignées, choix de $r$ et des modules).

</details>

### B4.E4 — Comment évalues-tu un système RAG, de la recherche à la réponse ?

<details><summary>Indice 1</summary>

Relis « Mesurer une recherche » (§204.6), « Évaluer et diagnostiquer » (§204.7) et §204.5.

</details>
<details><summary>Indice 2</summary>

Deux étages, mesurés séparément : la recherche (recall@k, MRR sur des questions annotées avec leur passage) puis la réponse (exactitude, citations vérifiées, « je ne sais pas » sur les questions sans réponse). Puis le diagnostic d'un échec, en commençant par la recherche.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je construis d'abord un jeu de test : des questions réelles, avec la réponse et le passage qui la contient, dont certaines sans réponse. » Puis : recall@k et MRR de la recherche ; exactitude normalisée et part de citations vérifiées ; taux d'abstention juste ; pour chaque échec, le bon passage était-il dans le top-k ? ; un juge LLM seulement validé sur des notes humaines ; incertitude sur un petit jeu.

</details>

### B4.E5 — Quels risques de sécurité avec un LLM en production, et quelles parades ?

<details><summary>Indice 1</summary>

Relis « L'injection de prompt », l'encadré 🕰️ sur l'OWASP Top 10 2026 et « Les données » (§204.10).

</details>
<details><summary>Indice 2</summary>

Ici, deux aggravants : des e-mails venus de l'extérieur (injection indirecte) et un outil qui **agit** (agentivité). Range les parades : ce que le modèle peut faire (droits), ce qu'il voit (données, secrets), ce qu'il produit (validation), et ce qu'on mesure.

</details>
<details><summary>Indice 3</summary>

Commence par : « Chaque e-mail client est une entrée non fiable : quelqu'un peut y écrire des instructions pour le modèle. » Puis : injection indirecte et agentivité excessive (LLM01, LLM03) ; fuite de données et du contexte caché ; sorties non validées ; consommation sans limite ; parades : droits minimaux et confirmation humaine, aucun secret dans le prompt, séparation données et consignes, validation des sorties, quotas et journaux, tests d'attaque réguliers.

</details>

<a id="notebook"></a>

## Exercices du notebook

Les indices des exercices du notebook (B4.12 à B4.30) seront ajoutés à la prochaine session de génération, avec le notebook.
