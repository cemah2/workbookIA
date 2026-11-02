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

a) **Faux** : les exemples sont dans le prompt ; les poids ne changent pas (Brown et coll., 2020). Pour b), relis ce que montrent Lu et coll. Pour c), relis la fin du premier paragraphe de « Zéro-shot et few-shot » : comment place-t-on un exemple pour un modèle de chat ? Pour d) et e), relis les deux réserves, une par question.

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

Relis en B3 le masque causal, le KV cache et le prefill, puis l'encadré 🔗 « Lien avec B3 » de §204.3 et la formule du KV cache (§204.2).

</details>
<details><summary>Indice 2</summary>

La première passe lit tout le prompt (le prefill) et donne le premier token ; ensuite, chaque token passe seul, avec le cache. Un token ne peut pas être produit avant le précédent. KV cache par token $= 2 \times L \times n_{kv} d_h \times$ octets par valeur.

</details>
<details><summary>Indice 3</summary>

a) Une passe sur le prompt donne le token 1, puis une passe par token suivant : $1 + 99 =$ **100**. Pour b), le token 2 peut-il être choisi avant que le token 1 soit connu ? Pour c), calcule $2 \times 24 \times (2 \times 64) \times 2$. Pour d), qu'est-ce qui fixe le nombre de passes avant ? Pour e), le masque causal laisse-t-il la position 50 voir la 49, la 51 ?

</details>

### B4.R2 — Ch. B1 : extraction de features ou fine-tuning : que change le nombre d'exemples ?

<details><summary>Indice 1</summary>

Relis en B1 l'extraction de features, le fine-tuning et l'oubli catastrophique, puis l'encadré 🔗 « Lien avec B1 » de §204.8.

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

1. Chaque tâche demanderait sa copie complète de 175 milliards de paramètres ; les adaptateurs ajoutent des couches traitées l'une après l'autre (de la latence), l'optimisation du prompt est difficile et prend une partie de la séquence. Pour le tableau 1, regarde la colonne où le batch et la séquence sont les plus petits. Pour 2, relis le deuxième paragraphe de l'introduction. Pour 3, relis les deux paragraphes qui suivent l'équation (3). Pour 4, calcule $2 \times 96 \times 12\,288 \times 8$, puis multiplie par 2 octets. Pour 5, divise 175 255,8 millions par le budget de la question 4 ; calcule le rapport et l'écart relatif des deux débits. Pour 6, relis la fin de l'annexe D.4. Pour 7, relis la phrase sous le tableau 6 et la note 6. Pour 8, divise $\lVert\Delta\mathbf{W}_q\rVert_F$ par $\lVert\mathbf{U}^\top\mathbf{W}_q\mathbf{V}^\top\rVert_F$ pour $r = 4$. Pour 9, compare à la fiche : `target_modules`, `lora_alpha`, rsLoRA, DoRA, QLoRA.

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

Les indices des exercices B4.12 à B4.30, dans l'ordre du notebook.

### Ex B4.12 — Premier contact : AutoTokenizer, AutoModelForCausalLM, generate() et pipeline sur un petit modèle 📦

<details><summary>Indice 1</summary>

Relis « Trois objets » et « Vitesse sur CPU » (§204.2), puis « `generate()` et ses réglages » (§204.3). `from_pretrained` prend le nom du dépôt, `revision=` et, pour le modèle, `dtype=`. La sortie de `generate()` contient le prompt **puis** les tokens générés. Pour les comptes, `print(model)` montre la structure : les blocs sont dans `model.model.layers`.

</details>
<details><summary>Indice 2</summary>

- `load_12` : deux `from_pretrained` sur `SMOL["repo"]` avec `revision=SMOL["revision"]`, `dtype=torch.float32` pour le modèle, puis `model.eval()` (déjà fait par `from_pretrained`, mais l'écrire ne coûte rien).
- `complete_12` : `tok(prompt, return_tensors="pt")`, `model.generate(**enc, max_new_tokens=..., do_sample=False)` sous `torch.inference_mode()`, puis la tranche `out[0, n_prompt:]`, décodée avec `skip_special_tokens=True` ; son nombre d'éléments est `n_new`.
- `counts_12` : `len(...)` des identifiants de `tok(PROMPT_12)` ; `isinstance(m, nn.Linear)` sur `model.modules()` ; `p.numel()` sur les paramètres du premier bloc.
- `make_pipe_12` : `pipeline("text-generation", model=model, tokenizer=tok)`.

</details>
<details><summary>Indice 3</summary>

```python
def complete_12(tok, model, prompt, max_new_tokens):
    enc = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, enc["input_ids"].shape[1]:]                 # the generated tokens only
    return ..., int(new.shape[0])


def counts_12(tok, model):
    n_linear = sum(isinstance(m, nn.Linear) for m in model.modules())
    ...                                                      # the tokens of PROMPT_12, the parameters of layers[0]
```

</details>

### Ex B4.13 — Prédire : dix générations du même prompt à température 0, 0,7 et 1,5 🔮

<details><summary>Indice 1</summary>

Relis « `generate()` et ses réglages » (§204.3) et la température du ch. 22 : diviser les logits par $T$ avant la softmax. À $T = 1$, on tire selon la distribution du modèle lui-même ; à $T < 1$, on la resserre vers le token le plus probable, sans la réduire à lui.

</details>
<details><summary>Indice 2</summary>

Pour a), une réponse identique à la gloutonne demande que **chaque** token tiré soit le plus probable, sur une quinzaine de tokens : la probabilité de la séquence entière est le produit des probabilités de ses tokens, une fois la distribution resserrée par $T = 0{,}7$. Pour b), sans filtre top-k ni top-p, toute la queue de la distribution (des dizaines de milliers de tokens peu probables) garde une part du tirage : que pèse-t-elle pour un modèle de 135 M de paramètres, à chacun des tokens qui précèdent le nom ?

</details>
<details><summary>Indice 3</summary>

Pose le calcul de a) : si chacun des 15 tokens de la réponse gloutonne a, après le passage à $T = 0{,}7$, une probabilité $p$, la réponse entière a la probabilité $q = p^{15}$, et le nombre de réponses identiques parmi 10 suit une loi binomiale de paramètres 10 et $q$ (ch. 2) : calcule son espérance pour $p = 0{,}9$, puis pour $p = 0{,}8$, et compare-la à 3. Pour b), demande-toi quelle part des réponses commence par une phrase qui ne mène pas au nom (« The doctor who… », « Sherlock Holmes is… ») et s'arrête à 24 tokens, ou nomme un autre personnage.

</details>

### Ex B4.14 — Chat template, message système, arrêt et streaming avec un modèle instruct 📦

<details><summary>Indice 1</summary>

Relis « Rôles et chat template » (§204.3, la réponse complète de bout en bout), « S'arrêter » et « Le streaming ». Tout part de `apply_chat_template(..., add_generation_prompt=True, return_tensors="pt", return_dict=True)` ; le reste est la tranche des tokens générés, comme en B4.12.

</details>
<details><summary>Indice 2</summary>

- `chat_14` : passe à `generate()` `stop_strings` et `tokenizer=tok` (seulement s'il y a des chaînes d'arrêt) et `streamer` ; après le décodage, coupe le texte avant la première occurrence de chaque chaîne d'arrêt (`texte.split(stop)[0]`) : le dernier token peut contenir plus que la chaîne.
- `stream_14` : crée le `TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)`, lance `model.generate(**enc, ..., streamer=streamer)` dans un `threading.Thread`, puis parcours le streamer dans une boucle (ou `list(streamer)`) : chaque tour rend un morceau, et la boucle s'arrête à la fin de la génération ; enfin `thread.join()`.

</details>
<details><summary>Indice 3</summary>

```python
def stream_14(tok, model, messages, max_new_tokens=48):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(target=model.generate, kwargs=dict(**enc, max_new_tokens=max_new_tokens,
                                                                 do_sample=False, streamer=streamer))
    ...                                                      # start, collect the pieces, join
```

</details>

### Ex B4.15 — Prompting pour extraire des informations de Holmes : zéro-shot, few-shot, JSON validé, vote majoritaire 📦

<details><summary>Indice 1</summary>

Relis « La sortie structurée » (§204.4 : `parse_departure`), « Zéro-shot et few-shot » (les exemples comme tours de conversation) et « Voter : l'auto-cohérence » (`Counter`). La validation se fait en trois temps : nettoyer, lire, contrôler.

</details>
<details><summary>Indice 2</summary>

- `parse_json_15` : `strip()`, puis si le texte commence par trois accents graves, retire `` ```json `` ou `` ``` `` au début et `` ``` `` à la fin ; `json.loads` dans un `try` (`json.JSONDecodeError`) ; `isinstance(data, dict)` et `set(data) == set(KEYS_15)` ; pour les types, `bool` est une sous-classe de `int` en Python : teste-le d'abord.
- `messages_15` : une liste qui commence par le message système, puis deux messages par exemple, `json.dumps(answer)` pour l'assistant, puis le texte.
- `vote_15` : les dictionnaires ne sont pas hachables ; compte des tuples `(name, place, year)` avec un `Counter`, qui garde l'ordre de première apparition : `most_common(1)` rend alors le premier vu en cas d'égalité.

</details>
<details><summary>Indice 3</summary>

```python
def vote_15(replies):
    parsed = [p for p in (parse_json_15(r) for r in replies) if p is not None]
    if not parsed:
        return None
    counts = Counter(tuple(p[k] for k in KEYS_15) for p in parsed)
    ...                                                      # the most common tuple, back to a dict
```

</details>

### Ex B4.16 — Sensibilité au prompt : cinq formulations, trois graines, ordre des exemples 🔬

<details><summary>Indice 1</summary>

Relis « Un prompt est fragile » (§204.4) et « La variance » (§204.5). Deux sources de dispersion : la formulation (une ligne du tableau à une autre) et la graine (une colonne à une autre, dans une même ligne). On ne compare l'une à l'autre qu'avec les mêmes passages, la même température et les mêmes graines.

</details>
<details><summary>Indice 2</summary>

- `accuracy_16` : une boucle sur `SNIPPETS_16` qui construit les messages avec `messages_15(texte, examples, system)`, appelle `MODEL_16.generate(...)` avec les réglages du contrat et la graine reçue, lit la réponse par `parse_json_15`, puis `score_15(prédictions, réponses attendues)["all"]`.
- `spread_16` : `np.asarray(table, dtype=float)` ; la moyenne de tout, `table.mean(axis=1).std()` (entre les lignes) et `table.std(axis=1).mean()` (dans les lignes) ; NumPy prend `ddof=0` par défaut.

</details>
<details><summary>Indice 3</summary>

```python
def accuracy_16(system, examples, seed):
    predictions = []
    for item in SNIPPETS_16:
        reply = MODEL_16.generate(messages_15(item["text"], examples, system), max_new_tokens=40, do_sample=True,
                                  temperature=0.7, seed=seed, cache=CACHE)
        ...                                                  # parse, then score against the expected answers
```

</details>

### Ex B4.17 — chunk_text et BM25 : découper un roman et classer ses passages 🔨

<details><summary>Indice 1</summary>

Relis « Découper » et « De TF-IDF à BM25 » (§204.6) et ✏️ B4.5. Le découpage est une boucle de fenêtres glissantes sur une liste d'unités (des mots ou des caractères) ; BM25 se prépare dans `fit` (fréquences des termes par document, fréquence des documents, longueurs, idf) et se calcule terme par terme dans `get_scores`.

</details>
<details><summary>Indice 2</summary>

- `chunk_text` : vérifie d'abord les arguments ; les unités sont `text.split()` (mot) ou `list(text)` (caractère), et l'on recolle avec `" ".join` ou `"".join` ; la fenêtre commence à 0, puis avance de `chunk_size - overlap` ; on s'arrête dès qu'une fenêtre atteint la fin (`start + chunk_size >= n`), ce qui donne un seul chunk pour un texte d'exactement `chunk_size` unités ; un texte vide donne une liste vide.
- `BM25.fit` : `Counter(doc)` pour chaque document (`term_freqs_`), la fréquence de document $n_t$ (un `Counter` sur les ensembles de termes), `doc_len_` en tableau NumPy, `avgdl_` sa moyenne, l'idf de chaque terme par la formule de la docstring.
- `get_scores` : pour chaque terme de la requête connu du corpus, ajoute $\text{idf} \cdot f (k_1 + 1) / (f + k_1(1 - b + b\,|d|/\overline{|d|}))$ aux documents où $f > 0$ seulement : c'est ce qui évite le 0/0.

</details>
<details><summary>Indice 3</summary>

```python
def get_scores(self, query):
    if not hasattr(self, "idf_"):
        raise ValueError("call fit() before get_scores()")
    scores = np.zeros(self.n_docs_)
    norm = self.k1 * (1 - self.b + self.b * self.doc_len_ / self.avgdl_)
    for term in query:
        ...                                                  # skip unknown terms; add only where f > 0
    return scores
```

</details>

### Ex B4.18 — cosine_top_k, fusion de classements (RRF), recall@k et MRR 🔨

<details><summary>Indice 1</summary>

Relis « La recherche dense », « La recherche hybride et la fusion RRF » et « Mesurer une recherche : recall@k et MRR » (§204.6), et ✏️ B4.2. La recherche dense est un tri de la matrice des cosinus de B2 ; la fusion, un dictionnaire de scores ; les métriques, des moyennes sur les requêtes.

</details>
<details><summary>Indice 2</summary>

- `cosine_top_k` : passe la requête en 2-D (`np.atleast_2d`), vérifie `k` et les dimensions, calcule `cosine_similarity_matrix(Q, D)` (B2), puis un tri **stable** de `-S` sur l'axe des documents (`np.argsort(-S, axis=1, kind="stable")`) garde le plus petit indice des ex aequo ; `np.take_along_axis` lit les similarités ; enlève l'axe des requêtes pour une requête 1-D.
- `reciprocal_rank_fusion` : un dictionnaire `id → score`, `enumerate(ranking, start=1)` ; trie par `(-score, id)` ; `np.array(..., dtype=int)` garde des entiers, même vide.
- `recall_at_k` : `len(set(liste[:k]) & set(pertinents)) / len(pertinents)` par requête, puis la moyenne ; `mean_reciprocal_rank` : le premier rang `r` (à partir de 1) d'un identifiant pertinent, `1 / r`, ou 0.

</details>
<details><summary>Indice 3</summary>

```python
def reciprocal_rank_fusion(rankings, k=60):
    if len(rankings) == 0 or k < 0:
        raise ValueError("expected at least one ranking and k >= 0")
    score = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            ...                                              # add 1 / (k + rank) to the score of int(doc)
    return np.array(sorted(score, key=lambda d: (-score[d], d)), dtype=int)
```

</details>

### Ex B4.19 — Prédire : sur des questions paraphrasées, BM25 ou recherche dense gagne-t-elle ? 🔮

<details><summary>Indice 1</summary>

Relis « La recherche dense » (§204.6 : les forces s'inversent par rapport à BM25) et 🧠 Q9. Pour BM25, demande-toi ce qui reste d'une requête dont aucun mot plein n'est dans le chunk ; pour la recherche dense, ce que l'embedding d'un chunk de 120 mots résume, et ce que celui d'une question courte résume.

</details>
<details><summary>Indice 2</summary>

BM25 ne peut plus compter que sur les petits mots (« the », « what », « was »), présents dans presque tous les chunks, donc d'idf faible : il classe alors d'autres chunks, qui partagent par hasard un mot plein avec la question. La recherche dense compare deux vecteurs : elle rapproche des formulations différentes d'une même idée (fiche §204.6), mais un chunk de 120 mots parle de plusieurs choses à la fois (des dialogues, des descriptions), et la question n'en vise qu'une, avec ses propres mots. Pèse les deux effets ; parmi 1 045 chunks, être dans les 5 premiers demande d'être plus proche que 1 040 autres. Pour c), la fusion additionne $1/(60 + \text{rang})$ : que gagne un chunk bien classé par une seule des deux méthodes ?

</details>
<details><summary>Indice 3</summary>

Pour c), pose le calcul : un chunk premier pour la recherche dense et absent des 20 premiers de BM25 a le score $1/61 \approx 0{,}0164$ ; un chunk 10ᵉ dans les deux a $2/70 \approx 0{,}0286$. Lequel l'hybride met-il devant ? Déduis-en quand l'hybride retrouve une question que la meilleure des deux retrouve seule, et quand il en retrouve une qu'aucune des deux ne met dans ses 5 premiers.

</details>

### Ex B4.20 — Recherche dense (MiniLM, e5) contre BM25 contre hybride sur 30 questions annotées 📦

<details><summary>Indice 1</summary>

Relis « La recherche dense », « La recherche hybride et la fusion RRF » et « Mesurer une recherche » (§204.6). L'expérience de B4.19 fait déjà les trois recherches pour une question : il s'agit de les ranger dans trois fonctions, puis de mesurer avec tes métriques de B4.18.

</details>
<details><summary>Indice 2</summary>

- `build_index_20` : un dictionnaire ; ton `BM25().fit` sur les chunks tokenisés ; `wb.text.passage_embeddings(sentence_model, chunks, subset)`, qui calcule une fois puis relit.
- `search_20` : pour BM25, un tri stable des scores décroissants (`np.argsort(-scores, kind="stable")`) ; pour la recherche dense, `index["model"].embed([query], kind="query")[0]` puis ton `cosine_top_k` ; pour l'hybride, les 20 premiers des deux, fusionnés par ta `reciprocal_rank_fusion`, coupés à `k`. Convertis en liste d'entiers Python (`.tolist()`).
- `evaluate_20` : les listes de 20 résultats et les ensembles pertinents de toutes les questions, puis un appel de `recall_at_k` (à 1 et à 5) et de `mean_reciprocal_rank` ; `float(...)` pour rendre des flottants Python.

</details>
<details><summary>Indice 3</summary>

```python
def search_20(index, query, method="hybrid", k=5):
    R = mylearn.retrieval
    if method not in ("bm25", "dense", "hybrid"):
        raise ValueError(f"unknown method {method!r}")
    bm = np.argsort(-np.asarray(index["bm25"].get_scores(tokenize(query))), kind="stable")[:max(k, DEPTH_20)]
    ...                                                      # the dense ranking, then the method asked for
```

</details>

### Ex B4.21 — Un RAG complet sur Holmes : récupération hybride, prompt avec citations, réponse d'un petit modèle 📦

<details><summary>Indice 1</summary>

Relis « Les quatre étapes » et « Le contexte » (§204.7). Le RAG est une chaîne de quatre fonctions courtes : le prompt, la lecture de la réponse, la vérification, puis l'assemblage avec ta recherche de B4.20 et le modèle.

</details>
<details><summary>Indice 2</summary>

- `rag_messages_21` : `"\n".join(f"[{n}] {p}" for n, p in enumerate(passages, start=1))`, puis le message système, les deux messages de l'exemple (des copies) et le message utilisateur.
- `parse_reply_21` : `re.finditer(r"\[(\d+)\]", reply)` pour les numéros (garde le premier de chaque), `re.sub(r"\[\d+\]", " ", reply)` pour le texte, puis `" ".join(texte.split())` et `.strip(" .")`.
- `verify_21` : `wbrag.content_words(answer)` ; rien à vérifier si la liste est vide, si un numéro sort de 1 à `len(passages)` ou si la réponse n'a aucun mot de contenu ; sinon, `any(...)` sur les passages cités, en comparant des ensembles de tokens (`<=`).
- `rag_21` : `search_20(index, question, "hybrid", k)`, les chunks de ces identifiants, puis les trois fonctions, et un dictionnaire.

</details>
<details><summary>Indice 3</summary>

```python
def verify_21(answer, cited, passages):
    words = wbrag.content_words(answer)
    if not cited or not words or any(not 1 <= n <= len(passages) for n in cited):
        return False
    ...                                                      # one cited passage whose tokens contain every word
```

</details>

### Ex B4.22 — Le RAG qui répond à côté : quatre défauts de découpage, d'encodage et de contexte 🐛

<details><summary>Indice 1</summary>

Lis chaque fonction à côté de sa docstring : chacune fait **autre chose** que ce qu'elle promet, sans erreur. Puis relis le tableau « Évaluer et diagnostiquer » (§204.7) et les pièges classiques de la fiche : quatre de leurs lignes décrivent ces défauts. Corrige dans l'ordre des points : un passage coupé fausse tout ce qui suit.

</details>
<details><summary>Indice 2</summary>

- Point 1 : en quelle unité `chunk_22` coupe-t-il, et que partagent deux chunks voisins ? Un extrait de 21 mots doit tenir entier dans au moins un chunk (fiche §204.6, « Découper »).
- Point 2 : e5 a été entraîné avec un préfixe différent pour une requête et pour un passage (B2) ; `SM_E5_22.embed` prend un argument `kind`.
- Point 3 : que fait `truncation=True, max_length=512` à un prompt de quatre passages, et à quel bout ? `QWEN_22.prompt_ids` rend les tokens du prompt entier.
- Point 4 : `verify_22` doit faire ce que fait `verify_21`.

</details>
<details><summary>Indice 3</summary>

```python
def chunk_22(text):
    words = text.split()
    return [" ".join(words[s:s + 120]) for s in ...]          # starts every 100 words, until the end is covered


def query_22(question):
    return SM_E5_22.embed([question], kind=...)[0]
```
Pour les points 3 et 4 : une ligne chacun, sans troncature d'un côté, et la règle de `verify_21` de l'autre.

</details>

### Ex B4.23 — Évaluer le RAG : recall@k, exactitude des réponses, « je ne sais pas » et analyse des échecs 📈

<details><summary>Indice 1</summary>

Relis « Évaluer et diagnostiquer » (§204.7) : on regarde d'abord la recherche, puis la génération, puis la citation. `category_23` suit cet ordre à l'envers : une réponse juste se classe par sa citation, une réponse fausse par la place du bon passage.

</details>
<details><summary>Indice 2</summary>

- `category_23` : une suite de `if` dans l'ordre de la docstring ; d'abord `answerable`, qui sépare les deux familles ; chaque branche rend une chaîne de `CATEGORIES_23`, recopiée telle quelle (accents compris).
- `recall_curve_23` : les `max(ks)` premiers résultats de `search_20(INDEX_20, question, "hybrid", ...)` pour chaque question, ses chunks pertinents par `wbrag.relevant_ids`, puis ton `recall_at_k` pour chaque $k$ : une seule recherche par question suffit.

</details>
<details><summary>Indice 3</summary>

```python
def category_23(row):
    if not row["answerable"]:
        return "abstention juste" if row["idk"] else "invention"
    ...                                                      # correct → by the citation; else idk, found, retrieval


def recall_curve_23(items, ks):
    retrieved = [search_20(INDEX_20, item["question"], "hybrid", max(ks)) for item in items]
    ...                                                      # the relevant sets, then one recall_at_k per k
```

</details>

### Ex B4.24 — Injection de prompt cachée dans un passage : attaquer, puis atténuer et mesurer 🔬

<details><summary>Indice 1</summary>

Relis « L'injection de prompt » (§204.10) et « La variance » (§204.5). Une expérience d'attaque se mène comme une évaluation : un critère de réussite écrit par programme, plusieurs passages, charges et graines, et une dispersion avant toute conclusion. Ne présume pas qu'une parade aide : c'est ce que tu mesures.

</details>
<details><summary>Indice 2</summary>

- `inject_24` et `attacked_24` : une ligne chacune (`.lower()` pour la casse).
- `delimited_messages_24` : le même gabarit que `rag_messages_21`, avec `GUARD_SYSTEM_24` et `f"[{n}] <passage>{p}</passage>"`.
- `filter_24` : `passage.split("\n")`, garde les lignes pour lesquelles **aucun** motif ne trouve rien (`re.search(motif, ligne, flags=re.IGNORECASE)`), puis `"\n".join(...)`.
- `run_24` : trois boucles imbriquées (graine, puis question et passage avec `zip`, puis charge), deux listes de lignes, `np.array(..., dtype=bool)` à la fin.

</details>
<details><summary>Indice 3</summary>

```python
def run_24(model, defense):
    attacked, correct = [], []
    for seed in SEEDS_24:
        row_a, row_c = [], []
        for item, passage in zip(ITEMS_24, PASSAGES_24):
            for payload in PAYLOADS_24.values():
                text = inject_24(passage, payload)
                ...                                          # filter?, messages (delimited?), one draw, two booleans
        ...
```

</details>

### Ex B4.25 — LoRA from scratch : une couche LoRALinear en PyTorch, vérifiée contre peft 📦

<details><summary>Indice 1</summary>

Relis « LoRA » (§204.8 : la formule, $\mathbf{B}_0 = 0$, $\alpha/r$) et l'exemple NumPy de la fiche, qui calcule déjà la branche et la matrice fusionnée pour des entrées en lignes. En PyTorch, un poids appris est un `nn.Parameter` rangé comme attribut du module ; un poids gelé garde `requires_grad=False`.

</details>
<details><summary>Indice 2</summary>

- `__init__` : garde `base`, passe `requires_grad_(False)` à chacun de ses paramètres, crée `A` (`torch.empty(r, base.in_features)` puis `nn.init.kaiming_uniform_`) et `B` (`torch.zeros(base.out_features, r)`), et `scaling`.
- `forward` : `base(x)` plus la branche étroite, calculée de droite à gauche comme dans la fiche (jamais la matrice $d \times k$).
- `merged` : un `nn.Linear` neuf, puis, sous `torch.no_grad()`, `copy_` du poids fusionné et du biais.
- `add_lora_25` : vérifie d'abord tous les noms dans `dict(model.named_modules())`, gèle tout, puis, pour chaque nom, coupe-le en parent et enfant (`name.rpartition(".")`), et `setattr(parent, enfant, LoRALinear(...))`.

</details>
<details><summary>Indice 3</summary>

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        ...                                                  # its init, B (zeros), scaling

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T) @ self.B.T
```

</details>

### Ex B4.26 — LoRA avec peft : adapter un petit modèle au style de Holmes, perplexité avant et après, fusion 📦

<details><summary>Indice 1</summary>

Relis « LoRA avec peft », le piège des `Conv1D` de GPT-2 et « Que montrer au modèle pendant l'entraînement ? » (§204.8) : on continue l'objectif du pré-entraînement, le token suivant. La perplexité est l'exponentielle de la cross-entropy moyenne (ch. 6) ; la boucle d'entraînement est celle du ch. 20, réduite aux paramètres entraînables.

</details>
<details><summary>Indice 2</summary>

- `load_base_26` : `AutoModelForCausalLM.from_pretrained(SPEC_26["repo"], revision=..., dtype=torch.float32)`, puis `.eval()`.
- `lora_model_26` : un `LoraConfig` avec les six réglages du contrat, puis `get_peft_model`.
- La loss du token suivant : `model(input_ids=w).logits`, les logits des positions 0 à $T - 2$ (`[:, :-1]`) contre les tokens 1 à $T - 1$ (`w[:, 1:]`), aplatis pour `F.cross_entropy`.
- `perplexity_26` : `model.eval()` et `torch.no_grad()`, puis `math.exp(loss.item())`.
- `train_26` : `torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)`, `model.train()`, la boucle, `model.eval()`.

</details>
<details><summary>Indice 3</summary>

```python
def next_token_loss(model, windows):
    logits = model(input_ids=windows).logits
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), windows[:, 1:].flatten())


def train_26(model, steps, lr, generator):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    ...                                                      # steps × (batch, loss, zero_grad, backward, step), eval
```

</details>

### Ex B4.27 — Rang, alpha et modules ciblés : perplexité contre paramètres entraînables 🔬

<details><summary>Indice 1</summary>

Une expérience = tes fonctions de B4.26 enchaînées avec d'autres réglages, sur un modèle **neuf** à chaque fois : un adaptateur entraîné ne doit pas servir de départ au suivant. Relis le rôle de $\alpha/r$ (§204.8, ∂ B4.6) avant de regarder la famille « α = 8 ».

</details>
<details><summary>Indice 2</summary>

`load_base_26()`, `lora_model_26(model, r, alpha, targets)`, `train_26(model, steps, LR_26, torch.Generator().manual_seed(seed))`, puis `trainable(model)` et `perplexity_26(model, HELD_26)` dans un dictionnaire. Pour lire le graphique : à nombre de paramètres égal, compare les points, puis compare les écarts à la barre d'erreur du protocole complet.

</details>
<details><summary>Indice 3</summary>

```python
def experiment_27(r, alpha, targets, steps, seed):
    model = lora_model_26(load_base_26(), r, alpha, targets)
    ...                                                      # train, then {"trainable": ..., "ppl": ...}
```

</details>

### Ex B4.28 — Classer des textes : encodeur fine-tuné par LoRA, LLM en zéro-shot, TF-IDF + régression logistique 📦

<details><summary>Indice 1</summary>

Relis §204.9 : trois approches sur les **mêmes** données, comparées sur l'accuracy, le F1 macro, le temps et les paramètres. La baseline est le `Pipeline` du ch. 15 ; l'encodeur, une tête de classification et LoRA (`task_type="SEQ_CLS"`) ; le LLM ne génère rien : il suffit de lire ses logits à la dernière position.

</details>
<details><summary>Indice 2</summary>

- `baseline_28` : `Pipeline([...]).fit(texts, labels)`.
- `encoder_28` : le tokenizer et `AutoModelForSequenceClassification.from_pretrained(..., num_labels=4)` à la révision de `MINILM_28`, puis `get_peft_model` avec un `LoraConfig` qui a `modules_to_save=["classifier"]`.
- `train_encoder_28` : `torch.randperm(len(texts), generator=generator)` à chaque epoch, des tranches de `batch_size` indices, `tok([...], padding=True, truncation=True, max_length=MAX_LEN_28, return_tensors="pt")`, `F.cross_entropy(model(**enc).logits, y[idx])`.
- `zero_shot_28` : les identifiants des quatre mots (`llm.tokenizer(mot, add_special_tokens=False)["input_ids"][0]`), puis, pour chaque texte, `llm.model(input_ids=torch.tensor([llm.prompt_ids(messages)])).logits[0, -1]` et l'`argmax` sur ces quatre identifiants.

</details>
<details><summary>Indice 3</summary>

```python
def zero_shot_28(llm, texts):
    topic_ids = [llm.tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in TOPICS_28]
    preds = []
    with torch.inference_mode():
        for text in texts:
            messages = [{"role": "system", "content": ZERO_SHOT_SYSTEM_28},
                        {"role": "user", "content": " ".join(text.split()[:100])}]
            ...                                              # one forward pass, the logits of the 4 ids, argmax
    return np.array(preds, dtype=np.int64)
```

</details>

### Ex B4.29 — Rendre un pipeline LLM reproductible : révision du modèle figée, prompts versionnés, cache des réponses 🛠️

<details><summary>Indice 1</summary>

Relis la clé du cache de `wb.llm` (la docstring du module) : tout ce qui décide d'une réponse gloutonne y est. Ta version en garde cinq champs. L'écriture atomique est celle de `wb.llm.write_json_atomic` : un fichier temporaire du même dossier, puis `os.replace`.

</details>
<details><summary>Indice 2</summary>

- `prompt_version_29` : `hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]`.
- `save_prompt_29` : `Path(folder) / "prompts" / f"{version}.txt"`, `parent.mkdir(parents=True, exist_ok=True)`, `write_text(..., encoding="utf-8")`.
- `cache_key_29` : le dictionnaire du contrat, `json.dumps(..., sort_keys=True)`, puis SHA-256.
- `answer_29` : la version, la clé, le chemin ; s'il existe, `json.loads(path.read_text(...))["reply"]` ; sinon la génération, puis `tempfile.mkstemp(dir=path.parent, suffix=".tmp")`, l'écriture du JSON et `os.replace(tmp, path)`.

</details>
<details><summary>Indice 3</summary>

```python
def answer_29(llm, template, question, settings, folder):
    version = save_prompt_29(template, folder)
    key = cache_key_29(llm.repo, llm.revision, version, question, settings)
    path = Path(folder) / "cache" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["reply"]
    ...                                                      # generate, write a temporary file, os.replace, return
```

</details>

### Ex B4.30 — Défi RAG : au moins 40 % de bonnes réponses citées sur 30 questions Holmes, modèle ≤ 0,6 B sur CPU 🏆

<details><summary>Indice 1</summary>

Pars de ton analyse de B4.23 : une réponse se perd à la recherche, à la lecture ou à la citation. Trois leviers, chacun mesurable sur les 30 questions avant la course : une meilleure recherche (B4.20 : quelle combinaison de méthodes retrouve le plus de chunks pertinents dans les $k$ premiers ?), un prompt qui fait **copier** une réponse courte au lieu de la reformuler, et une citation qui ne dépend plus du numéro que choisit le modèle. Garde un œil sur le temps : l'indexation se fait pendant la course.

</details>
<details><summary>Indice 2</summary>

- Indexe une seule fois pour toutes les questions : chunks, BM25 et les embeddings d'un ou deux modèles (MiniLM et e5 se complètent : fusionne leurs classements avec celui de BM25 par la RRF).
- Donne au modèle 3 à 6 passages : au-delà, Qwen2.5-0.5B se laisse distraire (« Lost in the Middle »), en deçà, le bon passage manque plus souvent.
- Demande « la plus courte expression **copiée mot pour mot** du passage » : une réponse recopiée contient les mots du passage, donc sa citation se vérifie.
- La citation : parmi les passages donnés, le premier qui contient tous les mots de contenu de la réponse (`wbrag.supports`), en commençant par celui que le modèle a cité. C'est honnête, puisque le passage rendu contient vraiment la réponse.
- Coupe une réponse trop longue (une phrase entière) avant de la rendre.

</details>
<details><summary>Indice 3</summary>

```python
def method_30(questions, documents):
    chunks = [c for _, body in documents for c in wbrag.chunk_words(body, 120, 20)]
    bm25 = wbrag.BM25Index([wbrag.tokenize(c) for c in chunks])
    ...                                                      # dense embeddings of the chunks, the generator
    out = []
    for question in questions:
        rankings = [wbrag.top_k(bm25.scores(wbrag.tokenize(question)), 20), ...]   # + one per dense model
        passages = [chunks[i] for i in wbrag.rrf(rankings)[:4]]
        ...                                                  # the reply, the answer, the first passage that supports it
    return out
```

</details>
