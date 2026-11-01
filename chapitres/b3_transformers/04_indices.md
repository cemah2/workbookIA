# B3 · Attention et Transformers : un mini-GPT from scratch — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B3.Q1 — Le goulot de l'encodeur-décodeur récurrent : pourquoi un seul vecteur ne suffit pas

<details><summary>Indice 1</summary>

Relis « Traduire avec deux RNN », « Le goulot » et « L'idée de Bahdanau » (§203.1).

</details>
<details><summary>Indice 2</summary>

Sans attention, tout passe par le dernier état de l'encodeur, un vecteur de taille fixe. Avec l'attention, le décodeur calcule, **à chaque pas** de la traduction, un poids pour **chaque** mot source, puis une softmax sur ces mots. Pour c), compte les poids d'un pas, puis le nombre de pas.

</details>
<details><summary>Indice 3</summary>

a) **C** : le décodeur de 2014 ne reçoit que le vecteur de contexte $\mathbf{c}$, le dernier état de l'encodeur. Pour b), relis la phrase « qu'elle compte 5 mots ou 50 ». Pour c), multiplie le nombre de poids d'un pas par le nombre de pas. Pour d), sur quel indice porte la somme de la softmax $\alpha_{ij} = e^{e_{ij}} / \sum_{j'} e^{e_{ij'}}$ ? Pour e), relis d'où viennent les poids dans l'entraînement.

</details>

### B3.Q2 — Requête, clé, valeur : un dictionnaire où l'on cherche « à peu près »

<details><summary>Indice 1</summary>

Relis « Un dictionnaire où l'on cherche à peu près », « D'où viennent les requêtes, les clés et les valeurs » et « La formule » (§203.2).

</details>
<details><summary>Indice 2</summary>

Les trois rôles d'un token viennent de **trois** projections différentes. Un score se divise par $\sqrt{d_k}$ ; les poids sont les exponentielles des scores, divisées par leur somme ; la sortie d'une requête est une somme pondérée de vecteurs valeurs.

</details>
<details><summary>Indice 3</summary>

a) **B** : une moyenne des valeurs pondérée par la ressemblance de la requête à chaque clé ; rien n'est choisi seul. Pour b), combien de matrices $\mathbf{W}$ transforment $\mathbf{x}_i$ ? Pour c), calcule $1 \times 2 + 3 \times 2$, puis divise par $\sqrt{2}$. Pour d), calcule $e^0$, $e^0$ et $e^{\ln 3}$, puis la part de la troisième. Pour e), quelle est la forme de $\sum_j \alpha_j \mathbf{v}_j$ ?

</details>

### B3.Q3 — Pourquoi diviser les scores par √d_k ?

<details><summary>Indice 1</summary>

Relis « Pourquoi diviser par $\sqrt{d_k}$ » et l'encadré 🧮 sur la variance d'un produit (§203.2).

</details>
<details><summary>Indice 2</summary>

Le produit scalaire de deux vecteurs aléatoires de cette sorte a une variance $d_k$ ; l'écart-type est la racine de la variance, et $\operatorname{Var}(aX) = a^2 \operatorname{Var}(X)$. Une softmax de scores très étalés donne presque tout au plus grand ; des scores presque égaux donnent des poids presque égaux.

</details>
<details><summary>Indice 3</summary>

a) La variance vaut $d_k = 256$, donc l'écart-type vaut $\sqrt{256} =$ **16**. Pour b), calcule $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k} / \sqrt{d_k})$ avec la règle $\operatorname{Var}(aX) = a^2 \operatorname{Var}(X)$. Pour c), que devient la Jacobienne $\operatorname{diag}(\mathbf{p}) - \mathbf{p}^\top\mathbf{p}$ quand $\mathbf{p}$ est presque un one-hot ? Pour d), calcule la variance des scores divisés par $d_k$, et demande-toi à quoi ressemble une softmax de scores tous très proches de 0. Pour e), les valeurs entrent-elles dans le calcul des scores ?

</details>

### B3.Q4 — Masque causal : qui a le droit de regarder qui ?

<details><summary>Indice 1</summary>

Relis « Le masque causal » et « Le masque causal quand il y a moins de requêtes que de clés » (§203.3).

</details>
<details><summary>Indice 2</summary>

En booléen, le masque causal est triangulaire inférieur, diagonale comprise : la ligne $i$ autorise les clés $0$ à $i$. Avec $L < S$, `causal_mask` aligne le triangle en bas à droite : la dernière requête voit toutes les clés.

</details>
<details><summary>Indice 3</summary>

a) Les lignes 0 à 4 ont $1, 2, 3, 4, 5$ cases vraies : $1 + 2 + 3 + 4 + 5 =$ **15**. Pour b), écris la ligne 2 du triangle. Pour c), la sortie de la position 1 lit-elle la clé 4 ? Pour d), applique la règle $j \le i + S - L$ avec $L = 1$, $S = 9$, $i = 0$. Pour e), relis la première phrase de « Le masque causal ».

</details>

### B3.Q5 — Auto-attention ou attention croisée : d'où viennent Q, K et V ?

<details><summary>Indice 1</summary>

Relis « Auto-attention et attention croisée » (§203.3) et « L'encodeur-décodeur : T5 et l'attention croisée » avec le tableau des familles (§203.7).

</details>
<details><summary>Indice 2</summary>

Dans l'attention croisée, $\mathbf{Q} = \mathbf{Y}\mathbf{W}_Q$ vient de la séquence qui « demande », $\mathbf{K}$ et $\mathbf{V}$ de celle qu'on regarde. La carte d'attention a une ligne par requête et une colonne par clé. Les arguments de `multi_head_attention` sont, dans l'ordre, `query`, `key`, `value`.

</details>
<details><summary>Indice 3</summary>

a) **A** : les requêtes viennent de la traduction en cours, les clés et les valeurs de la source. Pour b), écris la forme de $\mathbf{Q}\mathbf{K}^\top$ avec 7 requêtes et 12 clés. Pour c), la phrase source est-elle connue en entier avant de traduire ? Relis la ligne « Masque d'auto-attention » du tableau. Pour d), qui fournit `query` dans `(y, x, x)` ? Pour e), d'où viennent les requêtes et les clés d'une auto-attention ?

</details>

### B3.Q6 — Plusieurs têtes pour le prix d'une : que gagne-t-on ?

<details><summary>Indice 1</summary>

Relis « Plusieurs questions à la fois », « Une seule matrice pour toutes les têtes », « Compter les paramètres » et « Moins de têtes pour les clés : MQA et GQA » (§203.4), ainsi que le piège classique du reshape.

</details>
<details><summary>Indice 2</summary>

$d_h = d / H$. Les têtes redécoupent les mêmes matrices $(d, d)$. Le KV cache est proportionnel au nombre de têtes de clés et de valeurs. `repeat_interleave(4)` donne les têtes de clés 0, 0, 0, 0, 1, 1, 1, 1 ; `repeat(4)` donne 0, 1, 0, 1, 0, 1, 0, 1.

</details>
<details><summary>Indice 3</summary>

a) $d_h = 512 / 16 =$ **32**. Pour b), relis la formule $4d^2 + 4d$ : où apparaît $H$ ? Pour c), divise le nombre de têtes de clés et de valeurs de départ par le nouveau. Pour d), relis ce que fait `x.reshape(N, H, T, d_h)` dans le piège classique. Pour e), écris l'ordre des 8 têtes de clés que donne `repeat`, et lis la deuxième position.

</details>

### B3.Q7 — Sans encodage de position, l'attention voit-elle l'ordre des mots ?

<details><summary>Indice 1</summary>

Relis « L'attention ne voit pas l'ordre », « Les positions apprises » et « RoPE : tourner les requêtes et les clés » (§203.5).

</details>
<details><summary>Indice 2</summary>

Sans position, permuter les tokens permute les scores et les poids de la même façon : chaque token garde sa sortie, à sa nouvelle place. Une table de positions apprises a un nombre fixe de lignes. RoPE s'applique dans chaque couche, après les projections. Son angle vaut position $\times\ \theta_i$, avec $\theta_i = \text{base}^{-2i/d_h}$.

</details>
<details><summary>Indice 3</summary>

a) **Vrai** : la sortie de chaque token ne dépend que de l'ensemble des clés et des valeurs, pas de leur ordre ; une permutation des tokens permute donc les sorties. Pour b), les deux phrases ont-elles les mêmes tokens ? Pour c), combien de lignes a la table, et que se passe-t-il pour un indice trop grand ? Pour d), relis le premier des trois points pratiques de RoPE. Pour e), calcule $\theta_{16} = 10000^{-32/64}$, puis multiplie par 250.

</details>

### B3.Q8 — Résiduel, pré-LN ou post-LN : ce qui rend un Transformer profond entraînable

<details><summary>Indice 1</summary>

Relis « Deux sous-couches », « Pré-LN ou post-LN » et « Compter les paramètres d'un bloc » (§203.6), et le rappel R3.

</details>
<details><summary>Indice 2</summary>

Compare les deux formules de §203.6 : en pré-LN, la layer norm est à l'entrée de la branche, et l'entrée $\mathbf{x}$ est ajoutée telle quelle. Le MLP a deux couches denses, $(d, F)$ puis $(F, d)$, chacune avec un biais de la taille de sa sortie.

</details>
<details><summary>Indice 3</summary>

a) La formule pré-LN est $\mathbf{h} = \mathbf{x} + \operatorname{MHA}(\operatorname{LN}_1(\mathbf{x}))$ : c'est **C**. Pour b), suis le chemin qui va de l'entrée à la sortie sans entrer dans les branches. Pour c), relis le paragraphe sur Xiong et coll. Pour d), calcule $dF + F + Fd + d$. Pour e), relis ce que la connexion résiduelle garde pour le gradient.

</details>

### B3.Q9 — BERT, GPT, T5 : quel masque, quel objectif, quel usage ?

<details><summary>Indice 1</summary>

Relis les trois familles et leur tableau (§203.7).

</details>
<details><summary>Indice 2</summary>

BERT choisit 15 % des tokens ; parmi eux, 80 % deviennent `[MASK]`, 10 % un token au hasard, 10 % restent inchangés. Un décodeur fait « tout par le prompt ». Un encodeur rend un vecteur par token ou par texte.

</details>
<details><summary>Indice 3</summary>

a) **A** : retrouver des tokens masqués avec le contexte des deux côtés, le modèle de langue masqué. Pour b), calcule 15 % de 200. Pour c), prends 80 % du résultat de b). Pour d), relis ce que le même modèle sait faire « selon le prompt ». Pour e), relis les lignes « Sortie typique » et « Usages » : quelle famille sert à classer sans générer ?

</details>

### B3.Q10 — Décalage d'un token : ce que chaque position apprend à prédire

<details><summary>Indice 1</summary>

Relis « Teacher forcing et décalage d'un token » (§203.7) et « La loss et la perplexité » (§203.8).

</details>
<details><summary>Indice 2</summary>

La cible de la position $t$ est l'entrée de la position $t + 1$ : $y_t = x_{t+1}$, y compris pour la dernière position (le caractère suivant du texte). À l'initialisation, la distribution est presque uniforme sur les $V$ symboles.

</details>
<details><summary>Indice 3</summary>

a) Les cibles de « W, a, t, s, o » sont « a, t, s, o, n » : **B**. Pour b), chaque position de la fenêtre a-t-elle une cible ? Pour c), relis ce que le modèle reçoit en entrée à l'entraînement. Pour d), avec $y = x$, la cible de la position $t$ est-elle visible pour la position $t$ ? Pour e), calcule $-\log(1/65)$.

</details>

### B3.Q11 — KV cache : ce que l'on garde et pourquoi la génération s'accélère

<details><summary>Indice 1</summary>

Relis « Pourquoi c'est lent sans cache », « Le KV cache » et « Ce que coûte le cache » (§203.9).

</details>
<details><summary>Indice 2</summary>

Avec un masque causal, les clés et les valeurs des anciens tokens ne changent pas quand on en ajoute un ; seul le nouveau token passe dans le modèle. La formule : $2 \times L \times T \times d \times$ octets par valeur, avec $L$ le nombre de couches.

</details>
<details><summary>Indice 3</summary>

a) **B** : les clés et les valeurs, qui resservent à chaque nouveau token ; les anciennes requêtes ne resservent jamais. Pour b), relis l'étape 1 du KV cache. Pour c), relis la vérification de la fiche : `torch.allclose(full, ...)` donne-t-il `True` ? Pour d), quels facteurs de la formule dépendent du nombre de tokens, et à quelle puissance ? Pour e), calcule $2 \times 2 \times 100 \times 64 \times 4$.

</details>

### B3.Q12 — Coût quadratique du contexte et ce que change FlashAttention

<details><summary>Indice 1</summary>

Relis « Le calcul d'une couche », « La mémoire : la matrice $T \times T$ » et « FlashAttention : le même résultat, sans la matrice » (§203.10).

</details>
<details><summary>Indice 2</summary>

La matrice des poids a $T^2$ cases par tête. FlashAttention garde le même calcul, exact, par blocs ; il évite d'écrire la matrice entière. Dans un décodeur causal, l'attention coûte $2Td$ par token contre $24d^2$ pour les couches denses.

</details>
<details><summary>Indice 3</summary>

a) La mémoire varie comme $T^2$ : multiplier $T$ par 8 la multiplie par $8^2 =$ **64**. Pour b), relis la phrase « une attention exacte (pas une approximation) ». Pour c), que deviennent la mémoire et le nombre de FLOPs avec FlashAttention ? Pour d), résous $2Td = 24d^2$ en $T$, avec $d = 768$. Pour e), compare 100 000 au seuil $12d$ pour $d = 4\,096$.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### B3.R1 — Ch. B2 : de quoi dépend le nombre de tokens d'un texte ?

<details><summary>Indice 1</summary>

Relis la fiche de B2 : la fertilité et la compression (§202.6), et ce que fait une fusion BPE (§202.3).

</details>
<details><summary>Indice 2</summary>

Compression $=$ octets / tokens, donc tokens $=$ octets / compression. Chaque fusion BPE remplace des paires de tokens par un seul. Avec GPT-2, le français coûte environ 2 fois plus de tokens que l'anglais (B2).

</details>
<details><summary>Indice 3</summary>

a) **Faux** : chaque tokenizer découpe à sa façon (B2 l'a mesuré sur Holmes et Verne). Pour b), divise 4 000 par 3,2. Pour c), que fait une fusion de plus au nombre de tokens d'un texte où la paire apparaît ? Pour d), relis la fertilité du français avec GPT-2. Pour e), combien de caractères contient en moyenne un token BPE ?

</details>

### B3.R2 — Ch. 22 : pourquoi le gradient s'évanouit dans un RNN déroulé

<details><summary>Indice 1</summary>

Relis la fiche du ch. 22 (rétropropagation dans le temps, gradient évanescent, LSTM, clipping) et le tableau de §203.1.

</details>
<details><summary>Indice 2</summary>

Le gradient qui remonte $k$ pas traverse $k$ jacobiennes : il est multiplié par un produit de $k$ facteurs. Le LSTM ouvre une voie plus directe pour le gradient, sans supprimer tout le problème. Un RNN calcule $\mathbf{h}_t$ à partir de $\mathbf{h}_{t-1}$.

</details>
<details><summary>Indice 3</summary>

a) Un produit de 50 facteurs de norme 0,9 : $0{,}9^{50} \approx$ **0,0052**. Pour b), relis ce que le ch. 22 dit des dépendances très longues avec un LSTM. Pour c), relis la ligne « Chemin entre deux tokens éloignés » du tableau de §203.1. Pour d), relis le remède du ch. 22 contre l'explosion. Pour e), l'état $t$ peut-il se calculer avant l'état $t - 1$ ?

</details>

### B3.R3 — Ch. 20 : layer norm et connexion résiduelle, à quoi sert chacune ?

<details><summary>Indice 1</summary>

Relis la layer norm et les connexions résiduelles du ch. 20, la docstring de ta `layer_norm`, et le 🔗 lien avec le ch. 20 (§203.6).

</details>
<details><summary>Indice 2</summary>

La layer norm centre et réduit chaque exemple (ici, chaque token) sur ses features, avec la variance **biaisée** (division par le nombre de features), puis applique $\gamma$ et $\beta$ ($d$ valeurs chacun). La dérivée d'une somme est la somme des dérivées.

</details>
<details><summary>Indice 3</summary>

a) Moyenne $3$, variance biaisée $(4 + 1 + 0 + 9)/4 = 3{,}5$, écart-type $\sqrt{3{,}5} \approx 1{,}8708$ : $(1 - 3)/1{,}8708 \approx$ **−1,069**. Pour b), sur quel axe ta `layer_norm` calcule-t-elle ses statistiques ? Pour c), dérive $\mathbf{x}$, puis $F(\mathbf{x})$. Pour d), la layer norm d'un token lit-elle d'autres tokens ou d'autres exemples ? Pour e), compte les valeurs de $\gamma$ et de $\beta$.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B3.1 — L'attention à la main : trois tokens, d_k = 2, scores, softmax et sortie ✏️

<details><summary>Indice 1</summary>

Relis « La formule » et son exemple chiffré (§203.2), puis la softmax stable du ch. 17.

</details>
<details><summary>Indice 2</summary>

Calcule d'abord $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}_K$ et $\mathbf{V} = \mathbf{X}\mathbf{W}_V$, ligne par ligne : la ligne $i$ de $\mathbf{X}\mathbf{W}$ est $\mathbf{x}_i \mathbf{W}$. Puis les scores de la requête 2 face aux trois clés, divisés par $\sqrt{2}$ ; leurs exponentielles ; chaque exponentielle divisée par leur somme ; enfin la somme des $\alpha_{2j}\mathbf{v}_j$. Garde toutes les décimales jusqu'au bout.

</details>
<details><summary>Indice 3</summary>

a) $\mathbf{q}_2 = (1, 1)\,\mathbf{W}_Q = (0 + 1 ;\ 1 + 1) =$ **[1, 2]**. Pour b), calcule $\mathbf{k}_1 = (0, 1)\,\mathbf{W}_K$, puis $\mathbf{q}_2 \cdot \mathbf{k}_1 / \sqrt{2}$. Pour c), calcule les trois scores $s_{2,0}$, $s_{2,1}$, $s_{2,2}$, puis $e^{s_{2,1}} / (e^{s_{2,0}} + e^{s_{2,1}} + e^{s_{2,2}})$. Pour d), calcule $\alpha_{2,0}\mathbf{v}_0 + \alpha_{2,1}\mathbf{v}_1 + \alpha_{2,2}\mathbf{v}_2$. Pour e), refais c) pour la clé 2 avec les produits scalaires bruts. Pour f), compare $\mathbf{q}_0 \cdot \mathbf{k}_1$ et $\mathbf{q}_0 \cdot \mathbf{k}_2$. Pour g), calcule les lignes 0 et 1 de la carte comme en c), puis additionne leurs poids de la colonne 0 avec $\alpha_{2,0}$.

</details>

### Ex B3.2 — Le même calcul avec un masque causal ✏️

<details><summary>Indice 1</summary>

Relis « Le masque causal », son exemple chiffré, et « Le masque causal quand il y a moins de requêtes que de clés » (§203.3), ainsi que l'encadré 🕰️ sur `F.scaled_dot_product_attention`.

</details>
<details><summary>Indice 2</summary>

Une case masquée reçoit un score $-\infty$, donc une exponentielle nulle : la softmax se recalcule sur les clés restantes, qui se partagent tout le poids. Pour f) et g), écris les deux masques $(2, 3)$ : `np.tril(np.ones((2, 3)), k=1)` pour l'alignement en bas à droite, `k=0` pour celui de `is_causal=True`.

</details>
<details><summary>Indice 3</summary>

a) Le triangle $3 \times 3$ a $1 + 2 + 3 =$ **6** cases vraies. Pour b), la ligne 1 garde les scores $s_{1,0}$ et $s_{1,1}$ de ✏️ B3.1 et met $-\infty$ en $s_{1,2}$ : recalcule la softmax sur deux scores. Pour c), combine $\mathbf{v}_0$ et $\mathbf{v}_1$ avec ces poids. Pour d), combien de clés la requête 0 voit-elle ? Pour e), la ligne 2 du masque cache-t-elle une clé ? Pour f) et g), compte les `True` de la première ligne de chacun des deux masques $(2, 3)$. Pour h), écris la ligne 2 de `np.triu(np.ones((3, 3)))`.

</details>

### Ex B3.3 — Suivre les formes dans l'attention multi-têtes et compter ses paramètres ✏️

<details><summary>Indice 1</summary>

Relis « Une seule matrice pour toutes les têtes », « Compter les paramètres » et « Moins de têtes pour les clés » (§203.4), ainsi que les docstrings de `split_heads` et `multi_head_attention`.

</details>
<details><summary>Indice 2</summary>

Une couche dense change seulement le dernier axe. `split_heads` : $(N, T, d) \to (N, H, T, d_h)$. Les poids d'attention : $(N, H, L, S)$. Une matrice $(n_{in}, n_{out})$ et son biais ont $n_{in} n_{out} + n_{out}$ paramètres. PyTorch range une matrice de `nn.Linear` en $(n_{out}, n_{in})$.

</details>
<details><summary>Indice 3</summary>

a) La projection change le dernier axe de 384 en 384 : **"(4, 10, 384)"**. Pour b), calcule $d_h = 384 / 6$ et place les têtes juste après $N$. Pour c), une carte $(L, S)$ par séquence et par tête. Pour d), multiplie les quatre dimensions de c). Pour e), compte quatre matrices $(384, 384)$ et quatre biais. Pour f), où apparaît $H$ dans ton calcul de e) ? Pour g), empile trois matrices dans la convention de PyTorch. Pour h), compte $\mathbf{W}_Q$ et $\mathbf{W}_O$ comme en e), puis $\mathbf{W}_K$ et $\mathbf{W}_V$ de forme $(384, 128)$ avec un biais de 128 valeurs chacune. Pour i), $L = 10$ requêtes et $S = 25$ clés.

</details>

### Ex B3.4 — Encodage sinusoïdal : quelques valeurs et l'effet d'un décalage de position ✏️

<details><summary>Indice 1</summary>

Relis « L'encodage sinusoïdal (2017) » et « Pourquoi des sinus et des cosinus ? », avec la matrice du décalage (§203.5).

</details>
<details><summary>Indice 2</summary>

Avec $d = 8$ : $\omega_0 = 1$, $\omega_1 = 0{,}1$, $\omega_2 = 0{,}01$, $\omega_3 = 0{,}001$. La colonne $2i$ est $\sin(p\,\omega_i)$, la colonne $2i + 1$ est $\cos(p\,\omega_i)$. Une horloge de fréquence angulaire $\omega$ a une période $2\pi / \omega$. Un décalage de $k$ positions tourne la paire $i$ d'un angle $k\,\omega_i$, par la matrice de la fiche. $PE_p \cdot PE_{p'} = \sum_i \cos\big((p - p')\,\omega_i\big)$.

</details>
<details><summary>Indice 3</summary>

a) La paire 2 a $\omega_2 = 0{,}01$, donc une période $2\pi / 0{,}01 \approx 628{,}3$, soit **628** positions. Pour b), calcule $\sin(25 \times 1)$ en radians. Pour c) et d), la paire 1 a l'angle $25 \times 0{,}1$. Pour e) et f), l'angle du décalage vaut $10 \times \omega_1$ ; multiplie la matrice $\begin{pmatrix} \cos & \sin \\ -\sin & \cos \end{pmatrix}$ de cet angle par le vecteur $(0{,}6 ;\ 0{,}8)$. Pour g), additionne $\cos(2\,\omega_i)$ pour $i = 0$ à 3. Pour h), de quoi dépend $PE_p \cdot PE_{p'}$ ?

</details>

### Ex B3.5 — Compter les paramètres d'un GPT et retrouver les 124 millions de GPT-2 small ✏️

<details><summary>Indice 1</summary>

Relis « Compter les paramètres d'un bloc » et l'exemple de GPT-2 medium (§203.6), et le compte d'une attention de ✏️ B3.3.

</details>
<details><summary>Indice 2</summary>

Un bloc avec biais : attention $4d^2 + 4d$, MLP $8d^2 + 5d$ (avec $F = 4d$), deux layer norms $4d$. Le modèle : les blocs, la table des tokens $V \times d$, celle des positions $T_{\max} \times d$, la layer norm finale $2d$ ; la sortie liée n'ajoute rien. Un pourcentage d'écart se rapporte à la valeur de référence, ici les 117 millions.

</details>
<details><summary>Indice 3</summary>

a) $4 \times 768^2 + 4 \times 768 = 2\,359\,296 + 3\,072 =$ **2 362 368**. Pour b), calcule $8 \times 768^2 + 5 \times 768$. Pour c), additionne a), b) et $4 \times 768$. Pour d), multiplie c) par 12. Pour e), additionne $50\,257 \times 768$ et $1\,024 \times 768$. Pour f), additionne d), e) et $2 \times 768$. Pour g), retire de f) les deux tables. Pour h), calcule $100 \times (f - 117\,000\,000) / 117\,000\,000$. Pour i), ajoute à f) une matrice de sortie de la forme de la table des tokens. Pour j), refais c), d), e) et f) avec 4 blocs, $d = 128$, $V = 88$ et 128 positions.

</details>

### Ex B3.6 — Pourquoi √d_k : variance du produit scalaire de deux vecteurs aléatoires ∂

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la variance du produit de deux variables indépendantes centrées (§203.2), la variance d'une somme du ch. 2 et la softmax du ch. 17.

</details>
<details><summary>Indice 2</summary>

Pour 1, utilise $\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y]$ et $\mathbb{E}[X^2 Y^2] = \mathbb{E}[X^2]\,\mathbb{E}[Y^2]$ pour des variables indépendantes. Pour 4, $\mathbb{E}[X^2] = \operatorname{Var}(X) + \mathbb{E}[X]^2 = 1 + \mu^2$. Pour 5, divise numérateur et dénominateur de $e^s / (e^s + e^0)$ par $e^s$, puis dérive la sigmoïde (ch. 17).

</details>
<details><summary>Indice 3</summary>

1. $\mathbb{E}[q_c k_c] = \mathbb{E}[q_c]\,\mathbb{E}[k_c] = 0 \times 0 = 0$ ; puis $\operatorname{Var}(q_c k_c) = \mathbb{E}[q_c^2 k_c^2] - 0^2 = \mathbb{E}[q_c^2]\,\mathbb{E}[k_c^2] = 1 \times 1 = 1$. Pour 2, additionne les espérances, puis les variances des $d_k$ termes indépendants, et applique $\operatorname{Var}(aX) = a^2\operatorname{Var}(X)$ avec $a = 1/\sqrt{d_k}$. Pour 3, refais 1 avec $\mathbb{E}[q_c^2] = \sigma_q^2$. Pour 4, calcule $\mathbb{E}[q_c^2 k_c^2] - \mathbb{E}[q_c k_c]^2$ avec $\mathbb{E}[q_c^2] = 1 + \mu^2$, puis demande-toi ce que fait $\operatorname{softmax}(\mathbf{s} + c)$. Pour 5, calcule $\sigma(8)$, $\sigma(1)$ et les deux produits $p(1 - p)$. Pour 6, l'écart-type de $\mathbf{q} \cdot \mathbf{k} / d_k$ vaut $\sqrt{d_k}/d_k$. Pour 7, pense à l'initialisation et aux premiers pas de l'entraînement.

</details>

### Ex B3.7 — RoPE : la rotation rend q_m · k_n fonction de m − n seulement ∂

<details><summary>Indice 1</summary>

Relis « RoPE : tourner les requêtes et les clés » et l'encadré 🧮 sur la rotation d'un vecteur du plan (§203.5), puis le rappel de l'énoncé sur la transposée d'un produit.

</details>
<details><summary>Indice 2</summary>

Pour 1 : $\langle R_{m\theta} \mathbf{q}, R_{n\theta} \mathbf{k} \rangle = (R_{m\theta}\mathbf{q})^\top (R_{n\theta} \mathbf{k})$, puis $R_{m\theta}^\top = R_{-m\theta}$ et $R_{-m\theta} R_{n\theta} = R_{(n-m)\theta}$ (des angles qui s'additionnent). Pour 2, écris $R_\varphi \mathbf{k}$ composante par composante. Pour 4, un produit scalaire en dimension $d_h$ est la somme des produits scalaires de ses paires. Pour 6, développe comme $(a + b)(c + d)$.

</details>
<details><summary>Indice 3</summary>

1. $(R_{m\theta}\mathbf{q})^\top (R_{n\theta}\mathbf{k}) = \mathbf{q}^\top R_{m\theta}^\top R_{n\theta} \mathbf{k} = \mathbf{q}^\top R_{-m\theta} R_{n\theta} \mathbf{k} = \mathbf{q}^\top R_{(n-m)\theta}\mathbf{k} = \langle \mathbf{q}, R_{(n-m)\theta}\mathbf{k}\rangle$. Pour 2, $R_\varphi\mathbf{k} = (k_1\cos\varphi - k_2\sin\varphi ;\ k_1\sin\varphi + k_2\cos\varphi)$ : fais le produit scalaire avec $\mathbf{q}$ et regroupe les termes en $\cos\varphi$ et en $\sin\varphi$. Pour 3, calcule $\mathbf{q} \cdot \mathbf{k}$ et $q_2 k_1 - q_1 k_2$, puis l'angle $\varphi = (n - m)\,\pi/6$ de chaque couple. Pour 4, applique 1 à chaque bloc et additionne. Pour 5, utilise $\lVert \mathbf{x} \rVert^2 = \mathbf{x}^\top\mathbf{x}$ et $R^\top R = \mathbf{I}$, bloc par bloc. Pour 6, écris les quatre produits et regarde de quelles positions dépend chacun. Pour 7, de quoi dépend la clé rangée $R_{n\theta} \mathbf{k}$ ?

</details>

### Ex B3.8 — Fermi : FLOPs d'entraînement (≈ 6 N D), ton mini-GPT sur CPU, GPT-3, et la mémoire du KV cache 🧮

<details><summary>Indice 1</summary>

Relis « Le calcul d'une couche » et « Les lois d'échelle » (§203.10), puis « Ce que coûte le cache » (§203.9).

</details>
<details><summary>Indice 2</summary>

Tokens vus $=$ itérations $\times$ séquences par batch $\times$ longueur. FLOPs $\approx 6ND$, temps $=$ FLOPs / débit. KV cache $= 2 \times L \times T \times n_{kv} d_h \times$ octets par valeur, $L$ étant le nombre de couches. Chinchilla : $D \approx 20N$.

</details>
<details><summary>Indice 3</summary>

1. Mode rapide : $600 \times 32 \times 128 = 2\,457\,600$ tokens, puis $6 \times N \times 2{,}46 \times 10^6$ avec le $N$ de ✏️ B3.5 j), et divise par $10^{11}$ FLOP/s ; refais avec 5 000 itérations. Pour 2, calcule $128 / (12 \times 128)$. Pour 3, calcule $6 \times 1{,}75 \times 10^{11} \times 3 \times 10^{11}$, puis divise par $3 \times 10^{13}$ FLOP/s, et par 1 000 GPU ; convertis en jours ou en années. Pour 4, $D = 20 \times 175 \times 10^9$. Pour 5, applique la formule avec $L = 4$, $T = 128$, $d = 128$ et 4 octets. Pour 6, $n_{kv} d_h = 8 \times 128$, 2 octets, 36 couches ; divise la mémoire libre par la mémoire d'un token. Pour 7, pense à ce qui se passe en dehors des produits matriciels.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B3.9 — L'attention expliquée en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis « L'essentiel en 10 lignes » (points 1 et 2) et « Un dictionnaire où l'on cherche à peu près » (§203.2).

</details>
<details><summary>Indice 2</summary>

Une image : une bibliothèque où chaque livre a un titre sur la tranche (la clé) et un contenu (la valeur) ; tu arrives avec une question (la requête). Au lieu de prendre un seul livre, tu lis un peu de chacun, beaucoup de ceux dont le titre ressemble à ta question.

</details>
<details><summary>Indice 3</summary>

Commence par : « Pour comprendre un mot, il faut souvent regarder d'autres mots de la phrase, parfois loin. » Puis : chaque mot pose une question, montre un titre et propose un contenu ; on compare la question à tous les titres ; les ressemblances deviennent des parts qui font 100 % ; le mot repart avec un mélange des contenus, dosé par ces parts ; tous les mots le font en même temps, sans relire la phrase mot après mot.

</details>

### Ex B3.10 — Mémorisation : quand un modèle de langue recrache ses données d'entraînement ⚖️

<details><summary>Indice 1</summary>

Relis « Ce que le modèle retient » (§203.8), la data card et les données personnelles du ch. 29, et l'encadré sur le droit d'auteur du ch. 28.

</details>
<details><summary>Indice 2</summary>

Pour mesurer : un prompt pris dans l'entraînement, une génération déterministe (le plus probable à chaque pas), une comparaison caractère par caractère avec la vraie suite ; le témoin, ce sont des passages de validation, jamais vus. Pour les protections, classe-les en trois temps : les données (avant), l'entraînement (pendant), l'accès et les sorties (après).

</details>
<details><summary>Indice 3</summary>

1. Prends 200 passages de l'entraînement : leurs 50 premiers caractères en prompt, 50 caractères générés au plus probable, et compte ceux qui reproduisent la suite exacte ; fais de même sur 200 passages de validation. Pour 2 et 3, relis les résumés des deux articles de Carlini et coll. (arXiv:2012.07805 et 2202.07646). Pour 4, demande-toi si l'on peut retirer une conversation d'un modèle sans le réentraîner. Pour 5, une protection par ligne : la déduplication (Lee et coll., arXiv:2107.06499), le retrait ou le remplacement des données personnelles, la confidentialité différentielle (DP-SGD), les canaris (Carlini et coll., arXiv:1802.08232), un filtre de sortie, l'accès limité. Pour 6, relis l'article 53 du règlement 2024/1689. Pour 7, une ligne par risque : données personnelles, passages recopiés, mesure faite, limites de la mesure.

</details>

### Ex B3.11 — Vaswani et coll. (2017) : « Attention Is All You Need » 📄

<details><summary>Indice 1</summary>

Lis d'abord le résumé et la figure 1 ; la fiche (§203.1 à §203.7) suit l'article de près pour l'attention, les têtes, les positions et le bloc post-LN.

</details>
<details><summary>Indice 2</summary>

Les chiffres sont aux sections 3.1 (couches), 3.2.2 (têtes), 3.3 (MLP), 5.1 à 5.4 (données, matériel, optimiseur, régularisation), dans les tableaux 1 à 3 et dans la note 5 (le débit des GPU). Pour le coût d'entraînement, multiplie le temps, le nombre de GPU et leur débit. Pour le compte des paramètres, reprends ta méthode de ✏️ B3.5, avec une attention croisée en plus dans chaque couche du décodeur, et le MLP de largeur $d_{ff}$.

</details>
<details><summary>Indice 3</summary>

1. Un réseau récurrent calcule $\mathbf{h}_t$ à partir de $\mathbf{h}_{t-1}$ : les positions d'un exemple se calculent l'une après l'autre ; pour le matériel et la durée, relis la dernière phrase de la section 1. Pour 2, relis la section 3.1 (« LayerNorm(x + Sublayer(x)) »). Pour 3, relis les deux paragraphes qui suivent l'équation (1). Pour 4, calcule $4 \times 512^2 + 4 \times 512$. Pour 5, la condition est écrite dans la section 4. Pour 6, relis la fin de la section 3.4 et la section 3.5. Pour 7, l'équation (3) atteint son maximum quand ses deux termes sont égaux, à `step_num = warmup_steps`. Pour 8, calcule $9{,}5 \times 10^{12} \times 8 \times 12 \times 3\,600$ ; pour le compte, une couche d'encodeur a une attention, un MLP et deux layer norms, une couche de décodeur deux attentions, un MLP et trois layer norms. Pour 9, compare au tableau de l'encadré 🕰️ du §203.6.

</details>

<a id="entretien"></a>

## 💼 Entretien

### B3.E1 — Explique le mécanisme d'attention, de la requête à la sortie

<details><summary>Indice 1</summary>

Relis « L'essentiel en 10 lignes » (points 2, 3 et 5), « La formule » (§203.2) et « Une seule matrice pour toutes les têtes » (§203.4).

</details>
<details><summary>Indice 2</summary>

Suis le chemin : trois projections, les scores, la mise à l'échelle, le masque, la softmax sur les clés, la somme pondérée des valeurs ; puis les têtes, leur concaténation et la projection de sortie. Termine par le coût.

</details>
<details><summary>Indice 3</summary>

Commence par : « Chaque token produit trois vecteurs par des projections apprises : une requête, une clé, une valeur. » Puis : scores $\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}$ (pourquoi la racine), masque à $-\infty$, softmax sur les clés, sortie $= $ poids $\times\ \mathbf{V}$ ; $H$ têtes de taille $d/H$, concaténées, mélangées par $\mathbf{W}_O$ ; $4d^2$ paramètres, un coût en $T^2$.

</details>

### B3.E2 — Quel est le coût de l'auto-attention en fonction de la longueur, et comment traite-t-on les contextes longs ?

<details><summary>Indice 1</summary>

Relis §203.10 en entier, et « Allonger le contexte » (§203.5).

</details>
<details><summary>Indice 2</summary>

Sépare le calcul ($T^2 d$ dans l'attention contre $T d^2$ dans les couches denses, seuil $T = 12d$) et la mémoire (la matrice $T \times T$ à l'entraînement, le KV cache à l'inférence). Puis les remèdes, par famille : noyaux exacts, partage des clés, attention restreinte, positions étirées, hybrides.

</details>
<details><summary>Indice 3</summary>

Commence par : « L'auto-attention compare chaque token à tous les autres : le calcul et la mémoire des scores croissent comme le carré de la longueur. » Puis : le seuil $T = 12d$ ; FlashAttention (exact, mémoire linéaire, mêmes FLOPs) ; GQA pour le cache ; fenêtre glissante ; RoPE étiré (YaRN) ; hybrides avec des couches récurrentes.

</details>

### B3.E3 — Encodeur, décodeur ou encodeur-décodeur : lequel pour quelle tâche ?

<details><summary>Indice 1</summary>

Relis le tableau des familles et ses trois paragraphes (§203.7).

</details>
<details><summary>Indice 2</summary>

Pour chaque projet, demande-toi ce que l'on veut en sortie (un vecteur ou une classe, un texte libre, un texte qui dépend d'une entrée bien distincte), le coût d'inférence et la quantité de données annotées.

</details>
<details><summary>Indice 3</summary>

Commence par : « Tout dépend de ce que le modèle doit produire. » Puis : classer, un encodeur fine-tuné (petit, rapide, une tête de classification) ; résumer, un encodeur-décodeur ou un décodeur ; l'assistant, un décodeur ; et la nuance de 2026 : un grand décodeur sait tout faire par le prompt, au prix de son coût.

</details>

### B3.E4 — À quoi sert le KV cache, et combien de mémoire consomme-t-il ?

<details><summary>Indice 1</summary>

Relis §203.9 (le KV cache et ce qu'il coûte) et « Moins de têtes pour les clés » (§203.4).

</details>
<details><summary>Indice 2</summary>

Explique pourquoi les clés et les valeurs anciennes ne changent pas, ce que l'on gagne, puis la formule. Pour le chiffre, prends une architecture de 8 milliards de paramètres connue (32 couches, 8 têtes de clés de taille 128, bfloat16). Puis les réductions : GQA, quantification, pages, fenêtre.

</details>
<details><summary>Indice 3</summary>

Commence par : « À la génération, un décodeur causal recalculerait à chaque token les clés et les valeurs de tout le texte, qui ne changent pas. » Puis : on les garde, un token passe à la fois ; la mémoire $2 \times L \times T \times n_{kv} d_h \times$ octets ; calcule-la à 32 000 tokens ; les réductions et ce qu'elles coûtent.

</details>

### B3.E5 — Pourquoi un Transformer a-t-il besoin d'encodages de position, et qu'apporte RoPE ?

<details><summary>Indice 1</summary>

Relis §203.5 en entier.

</details>
<details><summary>Indice 2</summary>

Trois temps : l'équivariance par permutation (pourquoi l'attention ne voit pas l'ordre), les deux solutions additives (sinusoïdal, appris) et leurs limites, puis RoPE (la rotation, la propriété relative, aucun paramètre) et l'allongement du contexte.

</details>
<details><summary>Indice 3</summary>

Commence par : « Sans information de position, l'auto-attention traite une phrase comme un sac de mots. » Puis : sinusoïdes ajoutées (2017) ; table apprise (GPT-2, BERT), limitée à sa longueur ; RoPE tourne $\mathbf{q}$ et $\mathbf{k}$ d'un angle proportionnel à la position, le score ne dépend que de l'écart ; s'applique à chaque couche, sans paramètre, se marie avec le KV cache, et ses fréquences s'étirent (YaRN) pour allonger le contexte.

</details>

<a id="notebook"></a>

## Exercices du notebook

Les indices des exercices du notebook (B3.12 à B3.30) seront ajoutés à la prochaine session de génération, avec les parties A et B du notebook.
