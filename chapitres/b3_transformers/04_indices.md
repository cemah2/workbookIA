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

### Ex B3.12 — causal_mask, padding_mask et scaled_dot_product_attention 🔨

<details><summary>Indice 1</summary>

Relis « La formule » (§203.2), puis « Le masque causal », « Le masque de padding » et « Le masque causal quand il y a moins de requêtes que de clés » (§203.3). Chaque masque est **une seule comparaison** entre une colonne et une ligne d'indices (la diffusion, *broadcasting*) ; l'attention se fait en quatre temps : les scores, le masquage, la softmax sur l'axe des clés, la somme pondérée des valeurs.

</details>
<details><summary>Indice 2</summary>

- `causal_mask` : compare la ligne d'indices de clés `np.arange(S)[None, :]` à la colonne d'indices de requêtes `np.arange(L)[:, None]` décalée de $S - L$ ; les deux refus d'abord.
- `padding_mask` : `np.arange(S)[None, :] < lengths[:, None]`, après avoir vérifié les longueurs.
- `scaled_dot_product_attention` : convertis les trois entrées en tableaux de flottants (des listes peuvent arriver) ; `np.swapaxes(k, -1, -2)` transpose les deux derniers axes, quel que soit le nombre d'axes de tête ; `np.broadcast_to(mask, scores.shape)` étend le masque (et lève une erreur s'il ne se diffuse pas) ; `mask.any(axis=-1)` repère une ligne sans aucune clé autorisée ; `np.where(mask, scores, -np.inf)` masque avant la softmax ; ta `softmax(..., axis=-1)` du ch. 17 fait le reste. Pour l'échelle, teste `scale is None`.

</details>
<details><summary>Indice 3</summary>

```python
def scaled_dot_product_attention(q, k, v, mask=None, scale=None):
    q, k, v = (np.asarray(a, dtype=float) for a in (q, k, v))
    ...                                                   # d_k of q and k, S of k and v: ValueError otherwise
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    scores = (q @ np.swapaxes(k, -1, -2)) * scale
    if mask is not None:
        ...                                               # broadcast to scores.shape, refuse an empty row, then -inf
    weights = softmax(scores, axis=-1)
    return ..., weights
```

</details>

### Ex B3.13 — Prédire la carte d'attention quand une clé copie la requête, puis quand on retire la division par √d_k 🔮

<details><summary>Indice 1</summary>

Relis ∂ B3.6 : la variance du produit scalaire de deux vecteurs aléatoires indépendants, et ce que devient le produit scalaire d'un vecteur avec lui-même. Vois la softmax comme une compétition : le poids de la clé copiée vaut $e^{s_0} / \big(e^{s_0} + \sum_{j \ge 1} e^{s_j}\big)$.

</details>
<details><summary>Indice 2</summary>

Avec la division, le score de la clé copiée est $\lVert\mathbf{q}\rVert^2 / \sqrt{d_k}$, avec $\mathbb{E}\lVert\mathbf{q}\rVert^2 = d_k$ ; les scores des autres clés sont centrés, de variance 1 (∂ B3.6). Pour un score gaussien $Z$ de variance $\sigma^2$, $\mathbb{E}[e^{Z}] = e^{\sigma^2 / 2}$ (la moyenne d'une loi log-normale) : la somme des $S - 1$ autres exponentielles vaut environ $(S - 1)\,\mathbb{E}[e^{Z}]$. La clé copiée garde plus de la moitié du poids tant que son exponentielle dépasse cette somme. Sans la division, refais le raisonnement avec des scores $\sqrt{d_k}$ fois plus grands : le score copié et l'écart-type des autres changent tous les deux.

</details>
<details><summary>Indice 3</summary>

Pose l'égalité $e^{s_0} = (S - 1)\,\mathbb{E}[e^{Z}]$ avec les valeurs typiques de l'indice 2, et résous-la en $S$, une fois avec la division ($s_0 \approx d_k/\sqrt{d_k}$, $\sigma^2 = 1$), une fois sans ($s_0 \approx d_k$, $\sigma^2 = d_k$). Compare le premier $S$ aux puissances de 10 de la question a), le second à 100 000 pour b). La médiane sur 200 requêtes suit le cas typique, $\lVert\mathbf{q}\rVert^2 \approx d_k$.

</details>

### Ex B3.14 — split_heads, merge_heads et multi_head_attention 🔨

<details><summary>Indice 1</summary>

Relis « Une seule matrice pour toutes les têtes » et le piège du reshape direct (§203.4), et tes formes de ✏️ B3.3 : une seule projection pour toutes les têtes, puis un découpage de l'axe des features, que l'on fait passer avant l'axe du temps.

</details>
<details><summary>Indice 2</summary>

- `split_heads` : `reshape(N, T, H, d_h)`, puis `transpose(0, 2, 1, 3)` ; `merge_heads` : les mêmes opérations dans l'ordre inverse.
- `multi_head_attention` : vérifie d'abord que les huit paramètres sont dans `params` (une `ValueError` qui nomme ceux qui manquent, avant toute lecture `params["W_q"]`, qui lèverait `KeyError`), puis les formes ; les trois projections `x @ W + b`, `split_heads` de chacune, ta `scaled_dot_product_attention` de B3.12 (elle diffuse le masque sur les têtes : `(L, S)` ou `(N, 1, 1, S)` conviennent tels quels), `merge_heads`, la projection de sortie.

</details>
<details><summary>Indice 3</summary>

```python
def multi_head_attention(query, key, value, params, n_heads, mask=None):
    missing = [name for name in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o") if name not in params]
    if missing:
        raise ValueError(f"missing parameter(s): {missing}")
    ...                                                   # float arrays, 3-D, same N and D, H divides D
    Q = split_heads(np.asarray(query, dtype=float) @ params["W_q"] + params["b_q"], n_heads)
    ...                                                   # K from key, V from value
    heads, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    return ..., weights                                   # merge_heads, then W_o and b_o
```

</details>

### Ex B3.15 — positional_encoding et apply_rope 🔨

<details><summary>Indice 1</summary>

Relis « L'encodage sinusoïdal (2017) » et « RoPE : tourner les requêtes et les clés » (§203.5), et ✏️ B3.4 : une matrice d'angles positions × fréquences, puis des sinus et des cosinus. Pour RoPE, la propriété de ∂ B3.7 vaut pour chaque paire, quelle que soit la façon de former les paires.

</details>
<details><summary>Indice 2</summary>

- `positional_encoding` : `angles = positions[:, None] / base ** (np.arange(0, D, 2) / D)`, de forme `(T, D/2)`, puis `pe[:, 0::2] = np.sin(angles)` et `pe[:, 1::2] = np.cos(angles)`.
- `apply_rope` : les angles `(T, d/2)`, position × `base ** (-2 i / d)` ; les deux moitiés `x1 = x[..., :d // 2]` et `x2 = x[..., d // 2:]` ; recolle `x1 cos − x2 sin` et `x2 cos + x1 sin` par `np.concatenate(..., axis=-1)`. La diffusion s'occupe des axes de tête. Vérifie toi-même que `positions` a exactement $T$ valeurs : NumPy diffuserait une seule position sans rien dire.

</details>
<details><summary>Indice 3</summary>

```python
def apply_rope(x, positions=None, base=10000.0):
    x = np.asarray(x, dtype=float)
    T, d = x.shape[-2:]
    ...                                                   # d even; pos = arange(T), or exactly T given positions
    theta = pos[:, None] * base ** (-2.0 * np.arange(d // 2) / d)      # (T, d/2)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([..., ...], axis=-1)
```

</details>

### Ex B3.16 — transformer_block en NumPy, validé contre nn.TransformerEncoderLayer 🔨

<details><summary>Indice 1</summary>

Relis « Pré-LN ou post-LN » et « Le bloc en équations, et sa vérification » (§203.6) : deux sous-couches, chacune précédée de **sa** layer norm et ajoutée au chemin résiduel. Tu as déjà presque tout : ta `multi_head_attention` (B3.14), ta `layer_norm` (20.17) et ta `gelu` (17.14).

</details>
<details><summary>Indice 2</summary>

Vérifie d'abord les seize paramètres (une `ValueError` pour un manquant) et lis $F$ dans `W_1.shape[1]`. Puis, dans l'ordre : `a = layer_norm(x, ln1_gamma, ln1_beta, eps=eps)` ; `h = x + multi_head_attention(a, a, a, params, n_heads, mask)[0]` ; `m = layer_norm(h, ln2_gamma, ln2_beta, eps=eps)` ; la sortie `h + gelu(m @ W_1 + b_1) @ W_2 + b_2`. Si tu recodes la layer norm : sur le dernier axe, variance biaisée (`np.var`, comme `nn.LayerNorm`), `eps` sous la racine ; la GELU exacte : $\tfrac{z}{2}\big(1 + \operatorname{erf}(z / \sqrt{2})\big)$, par exemple avec `np.vectorize(math.erf)`.

</details>
<details><summary>Indice 3</summary>

```python
def transformer_block(x, params, n_heads, mask=None, eps=1e-5):
    ...                                                   # the 16 parameters, x 3-D, the shapes
    a = layer_norm(x, params["ln1_gamma"], params["ln1_beta"], eps=eps)
    h = x + multi_head_attention(a, a, a, params, n_heads, mask=mask)[0]
    ...                                                   # LN2 of h, then the MLP, added to h
```

</details>

### Ex B3.17 — Trois masques qui mentent : convention inversée, masque transposé, padding compté dans la loss 🐛

<details><summary>Indice 1</summary>

Les trois erreurs sont dans le tableau des pièges de la fiche et dans ses paragraphes (§203.3 et §203.8) : une convention de masque, un masque, une loss. Lis les diagnostics dans l'ordre : chacun ne parle que de son erreur, et le premier peut cacher les deux autres.

</details>
<details><summary>Indice 2</summary>

Point 1 : une position dont toutes les clés sont masquées rend des `nan` dans `nn.MultiheadAttention` ; dans quelle convention ce module lit-il un masque booléen (encadré 🕰️ sur `nn.MultiheadAttention`, §203.3), et dans laquelle `causal_mask_17` est-il écrit ? Point 2 : écris la ligne 2 de `causal_mask_17(4)` (comme en ✏️ B3.2 h) : quelles clés autorise-t-elle ? Point 3 : quelles cibles entrent dans la moyenne de `F.cross_entropy`, et quel argument en exclut (§203.8, « Deux remarques ») ?

</details>
<details><summary>Indice 3</summary>

Le correctif du point 1 tient en un caractère, dans `forward` ; celui du point 2 en un mot, dans `causal_mask_17` : compare `torch.ones(3, 3, dtype=torch.bool).triu()` et `.tril()` à la définition du masque causal (la requête $i$ voit les clés $j \le i$) ; celui du point 3 en un argument de `F.cross_entropy`, qui reçoit l'identifiant du padding des cibles.

</details>

### Ex B3.18 — De NumPy à PyTorch : un module CausalSelfAttention avec F.scaled_dot_product_attention 📦

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ sur `F.scaled_dot_product_attention` (§203.3) et « Une seule matrice pour toutes les têtes » (§203.4) : un seul produit pour les trois projections, puis le même découpage en têtes que ton `split_heads`, écrit en PyTorch.

</details>
<details><summary>Indice 2</summary>

Dans `__init__` : `self.qkv = nn.Linear(d_model, 3 * d_model)`, `self.proj = nn.Linear(d_model, d_model)`, `self.resid_drop = nn.Dropout(dropout)`, et garde `n_head` et `dropout`. Dans `forward` : `self.qkv(x).split(d_model, dim=2)` rend les requêtes, les clés et les valeurs ; chacune passe par `view(N, T, H, d_model // H).transpose(1, 2)` ; puis SDPA avec `is_causal=True` et `dropout_p=self.dropout if self.training else 0.0` ; enfin `transpose(1, 2).reshape(N, T, d_model)` (pas `view` : après un `transpose`, le tenseur n'est plus contigu), `proj` et le dropout.

</details>
<details><summary>Indice 3</summary>

```python
def forward(self, x):
    N, T, D = x.shape
    q, k, v = self.qkv(x).split(D, dim=2)
    q, k, v = (t.view(N, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
    y = F.scaled_dot_product_attention(...)               # causal; dropout on the weights in training only
    ...                                                   # merge the heads, proj, resid_drop
```

</details>

### Ex B3.19 — Mini-GPT caractère par caractère : embeddings de tokens et de positions, blocs pré-LN, tête liée 📦

<details><summary>Indice 1</summary>

Relis « Le modèle » et « La recette d'entraînement » (§203.8), et ta liste de ✏️ B3.5 j) : les tenseurs du modèle sont ceux que tu as comptés. Le contrat fixe leurs noms : `wbgpt.expected_state_shapes()` les affiche, et `{n: tuple(t.shape) for n, t in modele.state_dict().items()}` montre les tiens.

</details>
<details><summary>Indice 2</summary>

`Block_19` : quatre attributs et deux lignes de `forward`. `MiniGPT_19` : les deux tables, le dropout, un `nn.ModuleList` de blocs, `ln_f` et `head` sans biais ; puis la liaison `self.head.weight = self.tok_emb.weight`, qui affecte le même `nn.Parameter` (copier ses valeurs ferait deux tenseurs). L'initialisation : une boucle sur `self.modules()` (`nn.init.normal_(m.weight, std=0.02)` pour les `nn.Linear` et les `nn.Embedding`, `nn.init.zeros_(m.bias)` pour les biais), puis, pour chaque bloc, `attn.proj` et `mlp[2]` à l'écart-type `0.02 / math.sqrt(2 * n_layer)`. Dans `forward`, `self.pos_emb(torch.arange(T))` se diffuse sur le batch.

</details>
<details><summary>Indice 3</summary>

```python
class MiniGPT_19(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, d_model=128, dropout=0.0):
        super().__init__()
        ...                                               # block_size, tok_emb, pos_emb, drop, blocks, ln_f, head
        self.head.weight = self.tok_emb.weight            # tied: the same parameter
        ...                                               # nanoGPT initialisation, then the two residual projections

    def forward(self, idx):
        ...                                               # ValueError if T > block_size
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1], device=idx.device)))
        ...                                               # the blocks, ln_f, head
```

</details>

### Ex B3.20 — Entraîner le mini-GPT sur Holmes : AdamW, warmup + cosinus, clipping, perplexité, face au LSTM du ch. 22 📦

<details><summary>Indice 1</summary>

Relis « La recette d'entraînement » (§203.8), ta boucle du ch. 20, et le ch. 19 pour AdamW, le warmup + cosinus (19.19) et le clipping. Le contrat fixe l'ordre des opérations de chaque itération : la cellule de vérification compare tes poids, après 6 itérations, à ceux de la recette.

</details>
<details><summary>Indice 2</summary>

- `configure_optimizer_20` : parcours `model.parameters()`, qui ne liste la table liée qu'une fois, et range chaque paramètre selon `p.dim() >= 2` ; puis `torch.optim.AdamW([{"params": ..., "weight_decay": weight_decay}, {"params": ..., "weight_decay": 0.0}], lr=lr, betas=betas)`.
- `estimate_loss_20` : `model.eval()`, puis, sous `torch.no_grad()`, `n_batches` appels à `get_batch` et la moyenne des `loss.item()` ; enfin `model.train()`.
- `train_20` : une fonction interne `evaluate(done)` qui ajoute `done` à `val_iter` et la loss estimée avec un générateur neuf `torch.Generator().manual_seed(settings["eval_seed"])` ; appelle-la avant la boucle. Dans la boucle : le learning rate de chaque groupe (`group["lr"] = ...`, avec ton `cosine_lr` ou sa formule), le batch, la loss, `optimizer.zero_grad()`, `loss.backward()`, `nn.utils.clip_grad_norm_(model.parameters(), settings["clip"])`, `optimizer.step()`, l'historique, puis l'évaluation quand `(it + 1) % eval_interval == 0` ou après la dernière itération.

</details>
<details><summary>Indice 3</summary>

```python
for it in range(s["iters"]):
    lr = ...                                              # cosine_lr(it, s["iters"], s["lr"], s["warmup"], s["min_lr"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    x, y = get_batch(train_ids, s["batch_size"], block_size, generator)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    ...                                                   # gradients, clipping, step, history, evaluation
```

</details>

### Ex B3.21 — Lire les courbes d'entraînement du mini-GPT et poser un diagnostic 📈

<details><summary>Indice 1</summary>

Relis « La loss et la perplexité » (§203.8, le contrôle de départ) et, au ch. 20, la lecture des courbes d'entraînement et de validation : une validation qui baisse encore, une validation qui remonte pendant que l'entraînement continue de baisser, un dropout qui ne joue qu'à l'entraînement.

</details>
<details><summary>Indice 2</summary>

a) À l'initialisation, la distribution prédite est presque uniforme sur les $V$ caractères possibles : la loss vaut alors $\log V$ ; combien de caractères distincts le modèle prédit-il ? b) Regarde les dernières évaluations de ta courbe de validation. c) et d) Le graphique 3 met les deux validations sur la même échelle : cherche le point le plus bas de la courbe sans dropout, puis compare-le au dernier point de la courbe avec dropout. e) Le graphique 2 : mesure, à 5 000 itérations, l'écart entre la courbe d'entraînement et la courbe de validation du même entraînement.

</details>
<details><summary>Indice 3</summary>

a) La loss de départ vaut environ 4,5 sur la courbe : $V \approx e^{4{,}5}$, que tu arrondis au nombre de caractères distincts affiché par la cellule des outils de la partie A. Pour les autres : b) la dernière évaluation est-elle sous l'avant-dernière ? c) le minimum de la courbe rouge, parmi les quatre itérations proposées ; d) « plus bas » se lit sur l'axe vertical du graphique 3 ; e) un écart de 0,1 nat, à l'échelle du graphique 2, fait environ un vingtième de sa hauteur.

</details>

### Ex B3.22 — Prédire : le mini-GPT de Holmes amorcé avec une phrase de Verne 🔮

<details><summary>Indice 1</summary>

Relis « La boucle de génération » (§203.9) : à chaque pas, le modèle ne voit que son contexte, les 128 derniers caractères, et tire le suivant selon ce qu'il a appris. Et « La loss et la perplexité » (§203.8) : une prédiction uniforme sur 88 caractères coûte $\log 88$ nats par caractère.

</details>
<details><summary>Indice 2</summary>

a) Le prompt sort de la fenêtre au fil de la génération : ce que le modèle a appris l'emporte-t-il sur ce qu'il lit ? Pense aussi à ce que le modèle peut écrire : a-t-il jamais vu les mots de Verne ? b) Une prédiction uniforme ne sait rien ; ton modèle sait que les espaces sont fréquents, qu'une majuscule suit un point, quelles lettres reviennent souvent. Le français et l'anglais partagent-ils ces régularités ?

</details>
<details><summary>Indice 3</summary>

a) Pour qu'une continuation compte comme française, il faut que les mots que seul Verne contient y soient plus nombreux que ceux que seul Holmes contient : estime ce que le modèle produit après quelques dizaines de caractères, quand son contexte contient de plus en plus de ses propres mots. b) Compare $\log 88$ à ce que coûterait un modèle qui ne connaîtrait que la fréquence des lettres, communes en partie aux deux langues : un tel modèle est-il plus près d'une prédiction uniforme ou de 1,6 nat par caractère ?

</details>

### Ex B3.23 — Générer vite : le KV cache, et le gain de vitesse mesuré selon la longueur 📦

<details><summary>Indice 1</summary>

Relis « Le KV cache » (§203.9) et son extrait sur une couche : la sortie d'une position ne dépend que des clés et des valeurs des positions précédentes, qui ne changent plus. Il faut le faire dans les quatre blocs, avec les positions apprises et les connexions résiduelles.

</details>
<details><summary>Indice 2</summary>

- `generate_23` : une boucle de `n_new` pas ; à chaque pas, `model(idx[:, -model.block_size:])[:, -1]`, l'argmax, puis `torch.cat` ;
- `generate_cached_23` : une fonction interne qui fait passer `T` nouveaux tokens à partir de la position `start` : les embeddings de tokens et de positions (`torch.arange(start, start + T)`), puis, dans chaque bloc, `ln1`, `attn.qkv`, le découpage en têtes comme ton `forward` de B3.18, l'ajout des clés et des valeurs au cache du bloc (`torch.cat` sur l'axe des positions), SDPA, le recollage, `attn.proj`, l'ajout résiduel, puis `ln2`, `mlp` et le second ajout ; enfin `ln_f` et `head` sur la dernière position ;
- pour le prefill, les `T0` requêtes ont besoin du masque causal ; pour un seul nouveau token, sa requête voit tout le cache : le masque causal aligné en bas à droite de `causal_mask` couvre les deux cas.

</details>
<details><summary>Indice 3</summary>

```python
def forward_new(tokens, start):
    T = tokens.shape[1]
    x = model.tok_emb(tokens) + model.pos_emb(torch.arange(start, start + T))
    for i, block in enumerate(model.blocks):
        q, k, v = ...                                     # block.attn.qkv(block.ln1(x)), split into heads
        if cache[i] is not None:
            k, v = torch.cat([cache[i][0], k], dim=2), torch.cat([cache[i][1], v], dim=2)
        cache[i] = (k, v)
        ...                                               # SDPA with the bottom-right causal mask, proj, residual, mlp
    return model.head(model.ln_f(x[:, -1]))
```

</details>

### Ex B3.24 — Les cartes d'attention de chaque tête sur une phrase de Holmes 📈

<details><summary>Indice 1</summary>

Une ligne de carte est la distribution des poids d'**une** requête sur ses clés, et la somme de la ligne vaut 1. Relis le masque causal (§203.3) : rien au-dessus de la diagonale. Une tête qui regarde toujours la même position **relative** dessine une droite parallèle à la diagonale.

</details>
<details><summary>Indice 2</summary>

a) Le caractère précédent de la requête $i$ est la clé $i - 1$ : cherche la carte où la ligne juste sous la diagonale est la plus jaune. b) Une tête qui regarde les espaces dessine des **colonnes** claires sous les espaces (␣) de la phrase, quelle que soit la requête. c) Une attention étalée se voit à une carte sans case jaune vive, faite de nuances uniformes sur chaque ligne ; une attention concentrée, à des cases isolées très claires.

</details>
<details><summary>Indice 3</summary>

Pour a), parcours les 16 cartes et note, pour chacune, si la sous-diagonale est plus claire que le reste de la ligne : la tête retenue est celle où elle l'est le plus nettement. Fais de même pour les colonnes des espaces (b) et pour les lignes les plus grises (c), puis écris les trois couples `(bloc, tête)`. Si une affirmation est refusée, la mesure affichée te dit à quel point ta tête la vérifie : compare-la aux autres cartes.

</details>

### Ex B3.25 — Ablations : sans position, sans résiduel, post-LN, une seule tête 🔬

<details><summary>Indice 1</summary>

Relis les deux formules du bloc (§203.6, pré-LN et post-LN) et le rôle de la connexion résiduelle (ch. 20). Chaque variante ne change qu'une chose au modèle de B3.19 : écris une classe de bloc à options et une classe de modèle à options, ou des sous-classes des tiennes.

</details>
<details><summary>Indice 2</summary>

- Un bloc à deux options (`residual`, `pre_ln`) qui hérite de `Block_19` : `super().__init__(d_model, n_head)` crée `ln1`, `attn`, `ln2` et `mlp`, et `forward` choisit l'une des trois formules du contrat.
- Un modèle qui reprend `MiniGPT_19` avec ces blocs, une option `positions` (sans positions : `self.pos_emb = None`, qui ne crée aucun paramètre, et `forward` n'ajoute rien) et la même initialisation, faite **après** la construction des blocs.
- `"one_head"` : le même modèle avec `n_head = 1`.
- L'écart à la référence se calcule **graine par graine** (la même graine donne la même initialisation des parties communes et les mêmes batches), puis sa moyenne et son écart-type.

</details>
<details><summary>Indice 3</summary>

```python
class AblatedBlock_25(Block_19):
    def __init__(self, d_model, n_head, residual=True, pre_ln=True):
        super().__init__(d_model, n_head)
        self.residual, self.pre_ln = residual, pre_ln

    def forward(self, x):
        if not self.pre_ln:
            x = self.ln1(x + self.attn(x))
            return ...                                    # the second sub-layer, then ln2
        ...                                               # without residual; else super().forward(x)
```

Pour c), compare dans le tableau, ligne par ligne, l'écart moyen du protocole complet au double de sa dispersion : la première ligne d'ablation se décide ainsi, fais de même pour les trois autres.

</details>

### Ex B3.26 — Tokens BPE du B2 contre caractères : perplexité par caractère à budget égal 🔬

<details><summary>Indice 1</summary>

Relis la fin de « La loss et la perplexité » (§203.8) : une loss totale ne change pas d'unité quand on la répartit sur des caractères au lieu de tokens ; seul le nombre par lequel on la divise change. Et la compression d'un tokenizer (B2, 🔁 R1).

</details>
<details><summary>Indice 2</summary>

- `chars_per_token_26` : `len(text) / len(tok.encode(text))` ;
- `budget_block_26` : `round` du rapport, qui rend déjà un `int` ;
- `nats_per_char_26` : la loss moyenne par token sur les cibles `ids[block_size:]`, multipliée par leur nombre, donne la loss **totale** en nats ; divise-la par le nombre de caractères de `tok.decode(ids[block_size:].tolist())`.

</details>
<details><summary>Indice 3</summary>

```python
def nats_per_char_26(model, ids, tok, block_size):
    per_token = wbgpt.char_loss(model, ids, block_size=block_size, stride=block_size // 4)
    targets = ids[block_size:]
    ...                                                   # total nats, then divided by the characters of the targets
```

Pour b), compare l'écart des deux moyennes du protocole complet au double de la plus grande des deux dispersions.

</details>

### Ex B3.27 — Temps et mémoire de l'attention selon la longueur du contexte (64 à 2 048) 🔬

<details><summary>Indice 1</summary>

Relis « La mémoire : la matrice $T \times T$ » (§203.10) et, au ch. 23 ou en B1, la mesure d'un temps sur CPU : un premier appel plus lent (allocations), des essais perturbés par le reste de la machine, d'où le **meilleur** de plusieurs essais.

</details>
<details><summary>Indice 2</summary>

- `attention_time_27` : crée `q`, `k`, `v` par `torch.randn(1, n_head, T, d_head)`, ouvre `with sdpa_kernel(backend):`, fais un appel non mesuré, puis une boucle de `repeats` appels encadrés par `time.perf_counter()`, et rends `1000 * min(...)` ;
- `weights_bytes_27` : une carte $T \times T$ par tête, chaque poids sur `bytes_per_value` octets.

</details>
<details><summary>Indice 3</summary>

```python
with sdpa_kernel(backend):
    F.scaled_dot_product_attention(q, k, v, is_causal=True)          # not timed
    for _ in range(repeats):
        start = time.perf_counter()
        ...                                                          # one call, then keep the best time
```

</details>

### Ex B3.28 — Un Vision Transformer minuscule : Fashion-MNIST découpé en patchs 📦

<details><summary>Indice 1</summary>

Relis « Le Vision Transformer » (§203.10) et son extrait : la convolution de noyau et de stride $P$ découpe et projette les patchs d'un coup. Le bloc est celui de B3.19, l'encodeur de §203.7 : la seule différence est le masque.

</details>
<details><summary>Indice 2</summary>

- `EncoderBlock_28` : les attributs de `Block_19` et de `CausalSelfAttention_18` réunis (`ln1`, `qkv`, `proj`, `ln2`, `mlp`), et SDPA **sans** `is_causal` ;
- `ViT_28` : `patch_embed(x)` rend `(N, d, 4, 4)` ; `flatten(2).transpose(1, 2)` en fait `(N, 16, d)` ; `self.cls_token.expand(N, -1, -1)` répète le `[CLS]` pour le batch, `torch.cat` le place en tête, puis on ajoute `pos_emb` (qui se diffuse sur le batch) ; après les blocs et `ln_f`, `head` ne lit que la position 0.

</details>
<details><summary>Indice 3</summary>

```python
def forward(self, x):
    tokens = self.patch_embed(x).flatten(2).transpose(1, 2)              # (N, 16, d)
    tokens = torch.cat([self.cls_token.expand(len(x), -1, -1), tokens], dim=1) + self.pos_emb
    ...                                                                  # the blocks, ln_f, head on the [CLS]
```

</details>

### Ex B3.29 — Un test pytest qui prouve la causalité : modifier le futur ne change pas les logits passés 🛠️

<details><summary>Indice 1</summary>

Relis la vérification de causalité de la cellule de B3.19 et le 💼 « En entreprise » de §203.8. Un test qui ne change qu'**une** position ne voit qu'une partie des fuites possibles.

</details>
<details><summary>Indice 2</summary>

Tire un batch d'ids avec un générateur à graine (`torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE), generator=...)`), calcule les logits une fois ; puis, pour **chaque** position $t$ de 1 à `BLOCK_SIZE - 1`, change le token $t$, recalcule, et compare les logits des positions $0$ à $t - 1$. Le message de l'`assert` dit la position et l'écart.

</details>
<details><summary>Indice 3</summary>

```python
def test_causality_29():
    idx = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE), generator=torch.Generator().manual_seed(0))
    logits = MODEL(idx)
    for t in range(1, BLOCK_SIZE):
        changed = idx.clone()
        changed[:, t] = (changed[:, t] + 1) % VOCAB_SIZE
        ...                                               # compare the logits before t, assert with a message
```

</details>

### Ex B3.30 — Défi : la meilleure perplexité caractère sur Holmes avec au plus 1 M de paramètres et 10 min de CPU 🏆

<details><summary>Indice 1</summary>

Relis les courbes de B3.21 : en 600 itérations, ton mini-GPT de B3.20 apprenait encore. Le budget de 10 minutes est d'abord un nombre d'itérations : mesure ce que coûte une itération sur ta machine.

</details>
<details><summary>Indice 2</summary>

Une méthode simple : le mini-GPT de B3.20, entraîné aussi longtemps que le budget le permet, avec un warmup puis un cosinus qui finit **à la fin du budget**. Mesure la durée des premières itérations, déduis-en le nombre total d'itérations, et garde une marge (le contrôle a le quart du temps). Ensuite, B3.21 te dit si le dropout sert à ce nombre d'itérations, et B3.25 ce que coûte chaque pièce du bloc ; tu peux aussi changer $d$, le nombre de blocs, le contexte, le learning rate ou la taille des batches, en restant sous 1 000 000 de valeurs.

</details>
<details><summary>Indice 3</summary>

```python
def method_30(train_ids, vocab_size, seconds):
    start = time.perf_counter()
    model = MiniGPT_19(vocab_size)                        # 820 992 parameters
    optimizer = configure_optimizer_20(model, 5e-3, 0.1, (0.9, 0.99))
    ...                                                   # after 20 iterations: total = 0.92 * seconds / (time per step)
    ...                                                   # warmup + cosine down to `total`, clipping; stop in time
    return model.eval()
```

</details>
