# B3 · Attention et Transformers : un mini-GPT from scratch — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels et des exercices ✏️ B3.1 à B3.5 se vérifient aussi dans la partie 0 du notebook, où elles sont recalculées avec ta librairie (version de référence), PyTorch et transformers ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B3.Q1 — Le goulot de l'encodeur-décodeur récurrent : pourquoi un seul vecteur ne suffit pas
a) **C** : le décodeur de Cho et coll. et de Sutskever et coll. part du seul vecteur de contexte $\mathbf{c}$, le dernier état de l'encodeur ; la moyenne pondérée de tous les états (A) est justement l'apport de Bahdanau, et les mots source passent toujours par l'encodeur (B). b) **Faux** : $\mathbf{c}$ a une taille fixe, celle de l'état du RNN, qu'elle compte 5 mots ou 50 : c'est le goulot. c) **108** $= 12 \times 9$ : à chacun des 9 pas, un poids par mot source. d) **A** : la softmax porte sur les mots source $j$ ; un alignement souple n'est pas un choix d'un seul mot (C). e) **Vrai** : les poids sont dérivables et s'apprennent avec le reste du réseau, sans alignement annoté.
**À retenir** : un seul vecteur de taille fixe pour toute la phrase, c'est le goulot ; l'attention calcule un contexte propre à chaque pas, une moyenne des états source pondérée par une softmax.

### B3.Q2 — Requête, clé, valeur : un dictionnaire où l'on cherche « à peu près »
a) **B** : une moyenne des valeurs, pondérée par la ressemblance de la requête à chaque clé ; on ne choisit jamais une seule valeur (A), et l'on rend des valeurs, pas des clés (C). b) **Faux** : trois projections apprises différentes, $\mathbf{W}_Q$, $\mathbf{W}_K$ et $\mathbf{W}_V$, tirent trois vecteurs différents du même $\mathbf{x}_i$ : ce qu'un token cherche n'est pas ce qu'il montre. c) **5,66** $= (1 \times 2 + 3 \times 2) / \sqrt{2} = 8 / 1{,}4142$. d) **0,6** : les exponentielles valent $1$, $1$ et $3$, et la troisième prend $3/5$. e) **C** : la sortie d'une requête est une seule moyenne des valeurs, de forme $(d_v,)$ ; $(S,)$ est la forme de ses poids (A).
**À retenir** : requête, clé et valeur sont trois projections du même token ; la softmax des scores donne des poids de somme 1, et la sortie est une moyenne de valeurs.

### B3.Q3 — Pourquoi diviser les scores par √d_k ?
a) **16** : $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k = 256$, et l'écart-type est sa racine. b) **Vrai** : $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}/\sqrt{d_k}) = d_k / d_k = 1$ quel que soit $d_k$, donc un écart-type de 1 dans les deux cas. c) **A** : le plus grand score prend presque tout, et la Jacobienne $\operatorname{diag}(\mathbf{p}) - \mathbf{p}^\top\mathbf{p}$ tend vers 0 quand $\mathbf{p}$ tend vers un one-hot : le gradient ne passe plus. Une softmax ne donne jamais de poids négatif (C). d) **C** : la variance tomberait à $1/d_k$ ; des scores tous proches de 0 donnent des poids presque égaux, et chaque requête ferait presque la moyenne de toutes les valeurs. L'échelle compte avant l'exponentielle (A), et des scores plus petits saturent moins (B). e) **Faux** : les scores ne font intervenir que $\mathbf{q}$ et $\mathbf{k}$ ; la division compense leur dimension $d_k$, pas les valeurs.
**À retenir** : le produit scalaire de deux vecteurs aléatoires a une variance $d_k$ ; diviser par $\sqrt{d_k}$ la ramène à 1, assez pour que la softmax ne sature pas dès l'initialisation, sans écraser les écarts.

### B3.Q4 — Masque causal : qui a le droit de regarder qui ?
a) **15** $= 1 + 2 + 3 + 4 + 5$ : le triangle inférieur, diagonale comprise. b) **B** : les clés 0, 1 et 2 ; chaque requête se voit elle-même. c) **Faux** : la position 1 ne lit que les tokens 0 et 1 ; modifier un token futur ne change aucune sortie passée (l'invariant de 🛠️ B3.29). d) **9** : avec l'alignement en bas à droite, la dernière requête voit toutes les clés, $j \le 0 + 9 - 1$ ; `is_causal=True` n'en laisserait qu'une. e) **A** : cacher l'avenir, à l'entraînement (sinon la cible est visible) comme à la génération (où il n'existe pas encore). Le padding (B) a son propre masque, et ta fonction calcule tous les scores, masque ou non (C).
**À retenir** : la requête $i$ voit les clés $j \le i + S - L$ ; pour $L = S$, c'est le triangle inférieur, diagonale comprise.

### B3.Q5 — Auto-attention ou attention croisée : d'où viennent Q, K et V ?
a) **A** : les requêtes viennent de la traduction en cours, $\mathbf{Q} = \mathbf{Y}\mathbf{W}_Q$ ; les clés et les valeurs viennent de la sortie de l'encodeur. b) **"(7, 12)"** : une ligne par requête, une colonne par clé. c) **Faux** : la phrase source est connue en entier avant de traduire ; l'attention croisée n'a pas de masque causal (seulement un masque de padding de la source si besoin). Le masque causal est dans l'**auto**-attention du décodeur. d) **C** : le premier argument, `y`, fournit les requêtes : `y` regarde `x`. e) **Vrai** : en auto-attention, requêtes et clés viennent de la même séquence.
**À retenir** : la séquence qui demande donne $\mathbf{Q}$, celle que l'on regarde donne $\mathbf{K}$ et $\mathbf{V}$ ; la carte a la forme $(L, S)$.

### B3.Q6 — Plusieurs têtes pour le prix d'une : que gagne-t-on ?
a) **32** $= 512 / 16$. b) **Faux** : $4d^2 + 4d$ quel que soit $H$ ; plus de têtes, ce sont des têtes plus petites qui redécoupent les mêmes matrices. c) **8** $= 32 / 4$ : le cache est proportionnel au nombre de têtes de clés et de valeurs. d) **A** : la forme est bonne et rien ne plante, mais le reshape coupe la mémoire dans l'ordre C : une « tête » reçoit des morceaux de tokens entiers au lieu de $d_h$ features de chaque token. e) **Vrai** : `repeat` donne l'ordre 0, 1, 0, 1, 0, 1, 0, 1 : la tête de requêtes 1 tombe sur la tête de clés 1, alors que `repeat_interleave` (0, 0, 0, 0, 1, 1, 1, 1) lui donnerait la tête 0. C'est le piège quand on porte des poids d'un code à l'autre.
**À retenir** : $H$ têtes de taille $d/H$ coûtent les paramètres et le calcul d'une seule tête de taille $d$ ; c'est le nombre de têtes de **clés** qui fixe la taille du KV cache.

### B3.Q7 — Sans encodage de position, l'attention voit-elle l'ordre des mots ?
a) **Vrai** : l'auto-attention sans masque est équivariante par permutation : chaque token garde la même sortie, à sa nouvelle place. b) **A** : les deux phrases ont les mêmes cinq tokens ; sans position, la sortie de « chien » ne dépend que de cet ensemble. C'est pourquoi un Transformer a besoin des positions. c) **B** : la table n'a pas de ligne 1 500 (en PyTorch, `nn.Embedding` lève une `IndexError`) ; c'est la limite des positions apprises. d) **C** : RoPE tourne les requêtes et les clés, dans chaque couche, après les projections ; ni les valeurs ni les embeddings. e) **2,5** : $\theta_{16} = 10000^{-32/64} = 10^{-2}$, et $250 \times 0{,}01 = 2{,}5$ radians.
**À retenir** : sans position, une phrase est un sac de mots pour l'attention ; on ajoute des positions (sinusoïdales ou apprises) aux embeddings, ou l'on tourne $\mathbf{q}$ et $\mathbf{k}$ (RoPE).

### B3.Q8 — Résiduel, pré-LN ou post-LN : ce qui rend un Transformer profond entraînable
a) **C** : la layer norm est à l'entrée de la branche, et $\mathbf{x}$ est ajouté tel quel ; (A) est le post-LN de 2017, et (B) oublie la connexion résiduelle. b) **Vrai** : en pré-LN, le chemin résiduel ne traverse que des additions (la layer norm finale vient après le dernier bloc) ; chaque bloc ajoute sa contribution au flux. c) **A** : Xiong et coll. trouvent de grands gradients près de la sortie en post-LN, d'où un warmup long et soigneux ; c'est le pré-LN qui se contente d'un warmup court (C). d) **2 099 712** $= 2 \times 512 \times 2\,048 + 2\,048 + 512$. e) **Faux** : sans connexion résiduelle, le gradient doit traverser toutes les couches l'une après l'autre ; les layer norms gardent l'échelle des activations, pas la voie directe du gradient.
**À retenir** : la connexion résiduelle garde une voie directe pour le gradient, la layer norm garde l'échelle ; en pré-LN, la layer norm est dans la branche, et le chemin résiduel reste libre.

### B3.Q9 — BERT, GPT, T5 : quel masque, quel objectif, quel usage ?
a) **A** : le modèle de langue masqué ; prédire le token suivant (B) est l'objectif de GPT, réécrire des segments corrompus (C) celui de T5. b) **30** $= 15\,\%$ de 200. c) **24** : 80 % des 30 positions choisies ; les autres deviennent un token au hasard (10 %) ou restent inchangées (10 %). d) **Vrai** : un décodeur fait tout par le prompt, classer compris (« Ce ticket parle-t-il de facturation ? ») ; c'est plus coûteux qu'un petit encodeur. e) **B** : un encodeur fine-tuné rend un vecteur par texte, sur lequel on pose une tête de classification ; un décodeur (A) et un encodeur-décodeur (C) servent d'abord à produire du texte.
**À retenir** : encodeur (BERT, tokens masqués, comprendre), décodeur (GPT, token suivant, générer), encodeur-décodeur (T5, attention croisée, transformer un texte en un autre).

### B3.Q10 — Décalage d'un token : ce que chaque position apprend à prédire
a) **B** : $y_t = x_{t+1}$, donc les cibles de « W a t s o » sont « a t s o n ». b) **256** : une prédiction par position, la dernière comprise (sa cible est le token qui suit la fenêtre). c) **Faux** : le teacher forcing donne la séquence **vraie** en entrée ; les propres prédictions du modèle ne servent qu'à la génération, d'où le biais d'exposition. d) **C** : avec un masque causal, la position $t$ voit $x_t$, qui est sa propre cible : le modèle apprend à recopier, et la loss tombe près de 0 sans rien apprendre d'utile. e) **4,17** $= \log 65$ : une distribution uniforme donne $-\log(1/65)$ à chaque position.
**À retenir** : les cibles sont les entrées décalées d'un token ; une fenêtre de $T$ tokens donne $T$ exemples en parallèle, et la loss de départ vaut $\log V$.

### B3.Q11 — KV cache : ce que l'on garde et pourquoi la génération s'accélère
a) **B** : les clés et les valeurs, qui resservent à chaque nouveau token ; une ancienne requête ne resservira jamais (A), et les logits passés n'apportent rien au pas suivant (C). b) **1** : seul le nouveau token passe dans le modèle, avec une requête, une clé et une valeur par tête et par couche. c) **Faux** : le cache fait le même calcul et donne les mêmes logits, aux arrondis de calcul près (la fiche le vérifie avec `torch.allclose`) ; seul le temps change. d) **A** : une clé et une valeur par token et par couche. e) **102 400** $= 2 \times 2 \times 100 \times 64 \times 4$ octets.
**À retenir** : avec un masque causal, les clés et les valeurs des tokens passés ne changent plus ; les garder rend chaque pas proportionnel à $t$ dans l'attention et constant dans les couches denses, au prix d'une mémoire qui grandit avec le texte.

### B3.Q12 — Coût quadratique du contexte et ce que change FlashAttention
a) **64** $= 8^2$ : la matrice a $T^2$ cases par tête. b) **Faux** : FlashAttention est exact ; il calcule par blocs, avec une softmax en ligne, sans écrire la matrice entière. c) **C** : la mémoire devient linéaire en $T$ et les lectures-écritures baissent ; le nombre de FLOPs reste en $T^2$ (B), et le modèle ne change pas (A). d) **9 216** $= 12 \times 768$ : par token, l'attention causale coûte $2Td$ et les couches denses $24d^2$. e) **Vrai** : le seuil vaut $12 \times 4\,096 = 49\,152$ tokens, et 100 000 le dépasse.
**À retenir** : l'attention coûte $O(T^2 d)$ en calcul et, sans précaution, $O(T^2)$ en mémoire ; elle ne domine qu'au-delà de $T = 12d$ ; FlashAttention supprime le goulot de la mémoire, pas la complexité.

<a id="rappels"></a>

## 🔁 Rappels

### B3.R1 — Ch. B2 : de quoi dépend le nombre de tokens d'un texte ?
a) **Faux** : chaque tokenizer découpe à sa façon ; B2 l'a mesuré sur Holmes et Verne avec plusieurs tokenizers. b) **1 250** $= 4\,000 / 3{,}2$. c) **C** : chaque fusion remplace des paires de tokens par un seul, donc les textes raccourcissent. d) **Vrai** : environ 2 fois plus de tokens en français qu'en anglais avec GPT-2 (B2, §202.6). e) **B** : un token BPE couvre en moyenne plusieurs caractères (de 3 à 5 environ pour un texte anglais, selon le tokenizer et la taille de son vocabulaire, d'après les mesures de B2) : 128 tokens couvrent bien plus de texte que 128 caractères, d'où le 🔬 B3.26.
**À retenir** : le nombre de tokens dépend du tokenizer, de sa taille de vocabulaire et de la langue ; la longueur du contexte se compte en tokens, pas en caractères.

### B3.R2 — Ch. 22 : pourquoi le gradient s'évanouit dans un RNN déroulé
a) **0,0052** $\approx 0{,}9^{50}$ : un produit de 50 facteurs plus petits que 1 décroît exponentiellement. b) **Faux** : le LSTM atténue le problème grâce à sa cellule et à ses portes, sans le supprimer pour des dépendances très longues. c) **B** : une couche d'attention relie directement deux positions, quelle que soit leur distance : un chemin de longueur $O(1)$. d) **A** : le gradient clipping. e) **Faux** : l'état $t$ attend l'état $t - 1$ ; c'est la raison de la parallélisation du Transformer (§203.1).
**À retenir** : un RNN multiplie les jacobiennes de tous les pas, ce qui éteint le gradient sur les longues distances ; l'attention relie deux tokens en un pas, et toutes les positions se calculent en parallèle.

### B3.R3 — Ch. 20 : layer norm et connexion résiduelle, à quoi sert chacune ?
a) **−1,069** : moyenne 3, variance biaisée $(4 + 1 + 0 + 9)/4 = 3{,}5$, $(1 - 3)/\sqrt{3{,}5}$. b) **Faux** : la layer norm calcule ses statistiques sur les features de chaque exemple (ici chaque token), la batchnorm sur le batch. c) **B** : la dérivée de $\mathbf{x}$ est l'identité, celle de $F(\mathbf{x})$ s'y ajoute : même si $\partial F/\partial \mathbf{x}$ est petit, le gradient passe par $\mathbf{I}$. d) **Vrai** : elle ne lit que les $d$ features du token. e) **3 200** $= 2 \times 1\,600$ ($\gamma$ et $\beta$).
**À retenir** : la layer norm garde l'échelle de chaque token, indépendamment du batch et de la longueur ; la connexion résiduelle ajoute l'identité à la jacobienne, une voie directe pour le gradient.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B3.1 — L'attention à la main : trois tokens, d_k = 2, scores, softmax et sortie ✏️
**Réponse** : a) **[1, 2]** ; b) **3,536** ; c) **0,324** ; d) **[0,676 ; 1,286]** ; e) **0,727** ; f) **Faux** ; g) **0,202**.
**Démarche** : chaque ligne de $\mathbf{X}\mathbf{W}$ est $\mathbf{x}_i\mathbf{W}$ : $\mathbf{x}_0 = (1, 0)$ lit la première ligne de $\mathbf{W}$, $\mathbf{x}_1 = (0, 1)$ la seconde, $\mathbf{x}_2 = (1, 1)$ leur somme. D'où
$$\mathbf{Q} = \begin{pmatrix} 0 & 1 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}, \quad \mathbf{K} = \begin{pmatrix} 1 & 0 \\ 1 & 2 \\ 2 & 2 \end{pmatrix}, \quad \mathbf{V} = \begin{pmatrix} 1 & -1 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}, \quad \mathbf{Q}\mathbf{K}^\top = \begin{pmatrix} 0 & 2 & 2 \\ 1 & 3 & 4 \\ 1 & 5 & 6 \end{pmatrix}.$$
a) $\mathbf{q}_2 = (1, 2)$. b) $\mathbf{q}_2 \cdot \mathbf{k}_1 = 1 \times 1 + 2 \times 2 = 5$, et $5/\sqrt{2} \approx 3{,}5355$. c) Les scores de la ligne 2 sont $1/\sqrt{2} \approx 0{,}7071$, $3{,}5355$ et $6/\sqrt{2} \approx 4{,}2426$ ; leurs exponentielles $2{,}0281$, $34{,}3133$ et $69{,}5914$, de somme $105{,}9328$ ; d'où $\alpha_{2,\cdot} \approx (0{,}0191 ;\ 0{,}3239 ;\ 0{,}6569)$. d) $\mathbf{o}_2 = 0{,}0191\,(1, -1) + 0{,}3239\,(0, 2) + 0{,}6569\,(1, 1) \approx (0{,}6761 ;\ 1{,}2856)$. e) Sans la division, $e^1$, $e^5$, $e^6$ ($2{,}718$, $148{,}41$, $403{,}43$, somme $554{,}56$) : $\alpha_{2,2} \approx 0{,}7275$, plus concentré que $0{,}6569$. f) Faux : $\mathbf{q}_0 \cdot \mathbf{k}_1 = \mathbf{q}_0 \cdot \mathbf{k}_2 = 2$, donc mêmes scores et mêmes poids, $\alpha_{0,1} = \alpha_{0,2}$ ($0{,}4458$ chacun, et $0{,}1084$ pour la clé 0). g) La carte entière vaut
$$\begin{pmatrix} 0{,}1084 & 0{,}4458 & 0{,}4458 \\ 0{,}0743 & 0{,}3057 & 0{,}6200 \\ 0{,}0191 & 0{,}3239 & 0{,}6569 \end{pmatrix},$$
chaque ligne fait 1, et la colonne 0 fait $0{,}1084 + 0{,}0743 + 0{,}0191 \approx 0{,}2018$ : les colonnes ne sont pas des distributions (les deux autres font 1,075 et 1,723). Dans ta copie : avec $10\,\mathbf{W}_Q$, tous les scores sont multipliés par 10, et la ligne 2 devient $(0{,}0000 ;\ 0{,}0008 ;\ 0{,}9992)$ : la softmax sature, comme une division oubliée sur des vecteurs de grande dimension.
**Erreurs fréquentes** : multiplier $\mathbf{W}\mathbf{x}$ au lieu de $\mathbf{x}\mathbf{W}$ (ici $\mathbf{W}_Q$ est symétrique et le cache, mais pas $\mathbf{W}_K$) ; numéroter les tokens à partir de 1 ; diviser par $d_k$ au lieu de $\sqrt{d_k}$ ; faire la softmax sur une colonne ; moyenner les clés au lieu des valeurs ; arrondir les exponentielles avant de diviser.
**Variante** : les mêmes projections, avec $\mathbf{x}_2 = (2, 0)$ au lieu de $(1, 1)$. Réponse : $\mathbf{q}_2 = (0, 2)$, $\mathbf{k}_2 = (2, 0)$, $\mathbf{v}_2 = (2, -2)$ ; les produits scalaires de $\mathbf{q}_2$ avec les trois clés valent $0$, $4$ et $0$ : la clé 1 prend $\alpha_{2,1} \approx 0{,}8943$ (score $2{,}8284$ contre deux scores nuls, $0{,}0529$ chacun), et $\mathbf{o}_2 \approx (0{,}1586 ;\ 1{,}6300)$.

### Ex B3.2 — Le même calcul avec un masque causal ✏️
**Réponse** : a) **6** ; b) **[0,196 ; 0,804 ; 0]** ; c) **[0,196 ; 1,413]** ; d) **[1 ; −1]** ; e) **Vrai** ; f) **2** ; g) **1** ; h) **Faux**.
**Démarche** : a) Le triangle $3 \times 3$ a $1 + 2 + 3 = 6$ cases vraies. b) La ligne 1 garde les scores $0{,}7071$ et $2{,}1213$ et met $-\infty$ en clé 2 ; exponentielles $2{,}0281$ et $8{,}3421$, de somme $10{,}3702$ : $(0{,}1956 ;\ 0{,}8044 ;\ 0)$. Le poids masqué ne disparaît pas seulement : les deux autres se partagent tout. c) $0{,}1956\,(1, -1) + 0{,}8044\,(0, 2) \approx (0{,}1956 ;\ 1{,}4133)$. d) La requête 0 ne voit que la clé 0, de poids 1 : $\mathbf{o}_0 = \mathbf{v}_0 = (1, -1)$. e) La ligne 2 du masque est entièrement vraie : rien ne change, $\mathbf{o}_2 \approx (0{,}6761 ;\ 1{,}2856)$. f) `causal_mask(2, 3)` vaut `[[True, True, False], [True, True, True]]` : la requête 0 de ce calcul est le token $0 + 3 - 2 = 1$, qui voit les clés 0 et 1. g) Aligné en haut à gauche, `is_causal=True` donne `[[True, False, False], [True, True, False]]` : la requête du token 1 ne voit que la clé 0, et le dernier token ne se voit pas lui-même. h) La ligne 2 de `np.triu` n'autorise que la clé 2 : $\mathbf{o}_2 = \mathbf{v}_2$, qui ne dépend que du token 2 ; c'est la ligne 0 qui voit alors les trois tokens, et l'avenir avec. Dans ta copie : les sorties du calcul de f) sont exactement $\mathbf{o}_1$ et $\mathbf{o}_2$ du calcul masqué complet ; avec g), elles changent, ce qui ferait générer un modèle avec un cache autrement qu'à l'entraînement, sans aucun message d'erreur.
**Erreurs fréquentes** : mettre à 0 le poids masqué sans renormaliser (on obtient $(0{,}0743 ;\ 0{,}3057 ;\ 0)$, de somme 0,38) ; exclure la diagonale ; aligner le masque $(2, 3)$ en haut à gauche ; transposer le triangle.
**Variante** : le masque de padding `padding_mask([2], 3)` (le token 2 est du padding), sans masque causal. Réponse : chaque requête ne voit que les clés 0 et 1 ; la ligne 1 donne les poids $(0{,}1956 ;\ 0{,}8044 ;\ 0)$, comme avec le masque causal ; la ligne 0 voit maintenant deux clés, avec les **mêmes** poids, car ses deux scores ($0$ et $1{,}4142$) ont le même écart, $\sqrt{2}$, que ceux de la ligne 1 : $\mathbf{o}_0 = \mathbf{o}_1 \approx (0{,}1956 ;\ 1{,}4133)$. La requête de padding, la ligne 2, a une sortie calculée qu'il faudra exclure de la loss.

### Ex B3.3 — Suivre les formes dans l'attention multi-têtes et compter ses paramètres ✏️
**Réponse** : a) **"(4, 10, 384)"** ; b) **"(4, 6, 10, 64)"** ; c) **"(4, 6, 10, 10)"** ; d) **2 400** ; e) **591 360** ; f) **Vrai** ; g) **"(1152, 384)"** ; h) **394 240** ; i) **"(4, 6, 10, 25)"**.
**Démarche** : a) Une couche dense ne change que le dernier axe : $(4, 10, 384)$, les requêtes de toutes les têtes d'un coup. b) $d_h = 384 / 6 = 64$ ; le reshape donne $(4, 10, 6, 64)$, l'échange des axes 1 et 2 donne $(4, 6, 10, 64)$. c) Une carte $(L, S) = (10, 10)$ par séquence et par tête. d) $4 \times 6 \times 10 \times 10 = 2\,400$. e) Quatre matrices $(384, 384)$ et quatre biais : $4 \times 147\,456 + 4 \times 384 = 591\,360$. f) Vrai : 12 têtes de 32 redécoupent les mêmes matrices. g) PyTorch calcule $\mathbf{x}\mathbf{W}^\top$ et range chaque matrice en $(n_{out}, n_{in})$ ; les trois matrices d'entrée empilées font $(3 \times 384, 384)$. h) $\mathbf{W}_Q$ et $\mathbf{W}_O$ : $2 \times (147\,456 + 384) = 295\,680$ ; $\mathbf{W}_K$ et $\mathbf{W}_V$, de forme $(384, 128)$ : $2 \times (49\,152 + 128) = 98\,560$ ; total $394\,240$, soit un tiers de moins. i) $L = 10$ requêtes, $S = 25$ clés : $(4, 6, 10, 25)$. Dans ta copie : avec $H = 5$, $384 / 5$ n'est pas entier, `split_heads` lève une `ValueError` ; en GQA, $\mathbf{K}$ a la forme $(4, 2, 10, 64)$ après le découpage, chaque tête de clés servant 3 têtes de requêtes, et le cache est 3 fois plus petit.
**Erreurs fréquentes** : oublier l'axe du batch ; mettre les têtes après le temps ; donner aux poids la forme de la sortie ; oublier les biais ou $\mathbf{W}_O$ ; croire que le nombre de têtes change le nombre de paramètres ; oublier que PyTorch transpose ses matrices.
**Variante** : la même attention en MQA (une seule tête de clés et de valeurs, de taille 64). Réponse : $295\,680 + 2 \times (384 \times 64 + 64) = 344\,960$ paramètres, et un cache 6 fois plus petit qu'avec 6 têtes de clés.

### Ex B3.4 — Encodage sinusoïdal : quelques valeurs et l'effet d'un décalage de position ✏️
**Réponse** : a) **628** ; b) **−0,132** ; c) **0,5985** ; d) **−0,8011** ; e) **0,9974** ; f) **−0,0726** ; g) **2,5637** ; h) **Vrai**.
**Démarche** : $\omega_i = 10000^{-2i/8} = 10^{-i}$, soit $1$, $0{,}1$, $0{,}01$ et $0{,}001$. a) Période $2\pi/\omega_2 = 200\pi \approx 628{,}3$. b) $\sin 25 \approx -0{,}1324$ (25 radians, soit presque 4 tours). c) et d) La paire 1 a l'angle $25 \times 0{,}1 = 2{,}5$ : $\sin 2{,}5 \approx 0{,}5985$, $\cos 2{,}5 \approx -0{,}8011$. e) et f) Un décalage de 10 positions tourne la paire 1 de $10 \times 0{,}1 = 1$ radian :
$$\begin{pmatrix} \cos 1 & \sin 1 \\ -\sin 1 & \cos 1 \end{pmatrix}\begin{pmatrix} 0{,}6 \\ 0{,}8 \end{pmatrix} = \begin{pmatrix} 0{,}6 \times 0{,}5403 + 0{,}8 \times 0{,}8415 \\ -0{,}6 \times 0{,}8415 + 0{,}8 \times 0{,}5403 \end{pmatrix} \approx \begin{pmatrix} 0{,}9974 \\ -0{,}0726 \end{pmatrix}.$$
g) $PE_0$ vaut 0 dans les colonnes paires et 1 dans les impaires : $PE_0 \cdot PE_2 = \sum_i \cos(2\,\omega_i) = \cos 2 + \cos 0{,}2 + \cos 0{,}02 + \cos 0{,}002 \approx -0{,}4161 + 0{,}9801 + 0{,}9998 + 1{,}0000 \approx 2{,}5637$. h) Vrai : $PE_p \cdot PE_{p'} = \sum_i \big(\sin p\omega_i \sin p'\omega_i + \cos p\omega_i \cos p'\omega_i\big) = \sum_i \cos\big((p - p')\,\omega_i\big)$ ne dépend que de l'écart. Dans ta copie : $0{,}9974^2 + 0{,}0726^2 \approx 1$, car une rotation garde la longueur, et chaque paire $(\sin a, \cos a)$ est de norme 1 ; la plus petite valeur positive est $p = \arcsin(0{,}6)/0{,}1 \approx 6{,}435$ (les autres s'en déduisent par des tours complets, $p + 20\pi k$) ; les colonnes 6 et 7 tournent de $0{,}001$ radian par position : de 0 à 100, l'angle ne dépasse pas $0{,}1$ radian, et le cosinus reste au-dessus de 0,995.
**Erreurs fréquentes** : calculer en degrés ; confondre l'indice de la paire $i$ et celui de la colonne $2i$ ; mettre le cosinus dans les colonnes paires ; tourner dans le mauvais sens (le signe des $\sin$ de la matrice, qui est écrite pour le vecteur (sin, cos)) ; décaler à la fréquence $\omega_0$ au lieu de $\omega_1$.
**Variante** : un décalage de $-5$ positions sur la même paire $(0{,}6 ;\ 0{,}8)$. Réponse : l'angle vaut $-0{,}5$ radian, et la paire devient $(0{,}6\cos 0{,}5 - 0{,}8\sin 0{,}5 ;\ 0{,}6\sin 0{,}5 + 0{,}8\cos 0{,}5) \approx (0{,}1430 ;\ 0{,}9897)$, soit $(\sin 0{,}1435 ;\ \cos 0{,}1435)$.

### Ex B3.5 — Compter les paramètres d'un GPT et retrouver les 124 millions de GPT-2 small ✏️
**Réponse** : a) **2 362 368** ; b) **4 722 432** ; c) **7 087 872** ; d) **85 054 464** ; e) **39 383 808** ; f) **124 439 808** ; g) **85 056 000** ; h) **6,4** % ; i) **Vrai** ; j) **820 992**.
**Démarche** : a) $4 \times 768^2 + 4 \times 768 = 2\,359\,296 + 3\,072$. b) $768 \times 3\,072 + 3\,072 + 3\,072 \times 768 + 768 = 8 \times 768^2 + 5 \times 768 = 4\,718\,592 + 3\,840$. c) Avec les deux layer norms ($4 \times 768 = 3\,072$) : $12 \times 768^2 + 13 \times 768 = 7\,077\,888 + 9\,984$. d) $12 \times 7\,087\,872$. e) Tokens $50\,257 \times 768 = 38\,597\,376$, positions $1\,024 \times 768 = 786\,432$. f) $85\,054\,464 + 39\,383\,808 + 1\,536$ (layer norm finale) $= 124\,439\,808$, le nombre de `sum(p.numel() for p in model.parameters())`. g) $85\,054\,464 + 1\,536$ : les blocs font 68 % du modèle, les tables 32 %. h) $(124\,439\,808 - 117\,000\,000)/117\,000\,000 \approx 6{,}36\,\%$ : un écart que le dépôt `openai/gpt-2` attribue à une erreur de comptage de l'article (« our original parameter counts were wrong due to an error »), pas à une convention. i) Une matrice de sortie à part ajoute $38\,597\,376$ : $163\,037\,184$, plus de 160 millions. j) Un bloc : $12 \times 128^2 + 13 \times 128 = 198\,272$ ; 4 blocs $793\,088$ ; tokens $88 \times 128 = 11\,264$ ; positions $128 \times 128 = 16\,384$ ; layer norm finale 256 : $820\,992$, les « environ 0,8 million » de la fiche. Dans ta copie : le MLP fait $4\,722\,432 / 7\,087\,872 \approx 67\,\%$ d'un bloc, deux fois l'attention ; pour le mini-GPT, $4 \times 12 \times 128^2 = 786\,432$ contre $793\,088$ : les termes en $13d$ (biais et layer norms) font 0,8 % du compte des blocs, parce que $d = 128$ est petit (avec $d = 768$, 0,14 %).
**Erreurs fréquentes** : oublier les biais ou les layer norms ; compter deux fois la table des tokens alors que la sortie est liée ; oublier la table des positions ou la layer norm finale ; prendre $F = d$ ; rapporter l'écart en pourcentage au compte exact au lieu des 117 millions ; croire l'article contre le compte.
**Variante** : BERT-base, 12 blocs post-LN de même taille ($d = 768$, $F = 3\,072$), $V = 30\,522$, 512 positions, une table de 2 types de segment, une layer norm sur les embeddings et un *pooler* (une couche dense $768 \to 768$), sans layer norm finale. Réponse : $85\,054\,464 + 23\,440\,896 + 393\,216 + 1\,536 + 1\,536 + 590\,592 = 109\,482\,240$, les « environ 110 millions » de la fiche.

### Ex B3.6 — Pourquoi √d_k : variance du produit scalaire de deux vecteurs aléatoires ∂
**Réponse** : $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$, donc $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}/\sqrt{d_k}) = 1$ ; sans la division, pour $d_k = 64$, la dérivée $\partial p/\partial s$ de la softmax est environ 586 fois plus petite pour un écart de scores typique.
**Démarche** :
1. Par indépendance, $\mathbb{E}[q_c k_c] = \mathbb{E}[q_c]\,\mathbb{E}[k_c] = 0$. Puis $\operatorname{Var}(q_c k_c) = \mathbb{E}[q_c^2 k_c^2] - \mathbb{E}[q_c k_c]^2 = \mathbb{E}[q_c^2]\,\mathbb{E}[k_c^2] - 0 = 1 \times 1 = 1$, car $\mathbb{E}[q_c^2] = \operatorname{Var}(q_c) + \mathbb{E}[q_c]^2 = 1$.
2. L'espérance d'une somme est la somme des espérances : $\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = 0$. Les $d_k$ termes étant indépendants, les variances s'additionnent : $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k \times 1 = d_k$. Enfin $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}/\sqrt{d_k}) = d_k / d_k = 1$.
3. $\operatorname{Var}(q_c k_c) = \sigma_q^2 \sigma_k^2$, donc $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k\,\sigma_q^2\sigma_k^2$ ; il faudrait diviser par $\sqrt{d_k}\,\sigma_q\sigma_k$. C'est un argument d'**initialisation** : la layer norm avant l'attention et des poids initialisés avec soin gardent les composantes à une échelle voisine de 1.
4. $\mathbb{E}[q_c k_c] = \mu^2$, donc $\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = d_k\mu^2$ ; $\mathbb{E}[q_c^2 k_c^2] = (1 + \mu^2)^2$, d'où $\operatorname{Var}(q_c k_c) = (1 + \mu^2)^2 - \mu^4 = 1 + 2\mu^2$. La softmax est invariante par l'ajout d'une même constante à tous les scores d'une ligne ($e^{s_j + c} = e^c e^{s_j}$, et $e^c$ se simplifie) : un décalage commun ne change rien ; ce qui compte pour la saturation, ce sont les **écarts** entre les scores d'une même ligne. Dans une ligne, la requête est fixée : en écrivant $\mathbf{k}_j = \mu\mathbf{1} + \boldsymbol{\varepsilon}_j$, le score vaut $\mu\sum_c q_c + \mathbf{q} \cdot \boldsymbol{\varepsilon}_j$, dont le premier terme est commun à toutes les clés et disparaît dans la softmax ; les écarts viennent du second, de variance $\lVert\mathbf{q}\rVert^2$ d'une clé à l'autre, soit $d_k(1 + \mu^2)$ en moyenne. La variance totale $d_k(1 + 2\mu^2)$ compte en plus le décalage commun, qui change d'une requête à l'autre sans rien changer aux poids. Dans les deux cas, la dispersion utile grandit comme $d_k$ : la division par $\sqrt{d_k}$ reste nécessaire.
5. $p = e^s / (e^s + e^0) = 1/(1 + e^{-s}) = \sigma(s)$, et $\sigma' = \sigma(1 - \sigma)$ (ch. 17). Pour $s = 8$ : $p \approx 0{,}99966$, $p(1 - p) \approx 3{,}35 \times 10^{-4}$. Pour $s = 1$ : $p \approx 0{,}7311$, $p(1 - p) \approx 0{,}1966$. De $s = 8$ à $s = 1$, la dérivée $\partial p/\partial s$ est multipliée par environ $586$ ($0{,}1966 / 3{,}35 \times 10^{-4}$) : sans la division, presque rien ne remonte vers $\mathbf{W}_Q$ et $\mathbf{W}_K$. Si l'on dérive par rapport au produit scalaire brut $\mathbf{q} \cdot \mathbf{k}$, la division apporte en plus son facteur $1/\sqrt{64} = 1/8$ : le gain net est d'environ $586 / 8 \approx 73$, encore très grand.
6. L'écart-type de $\mathbf{q} \cdot \mathbf{k}/d_k$ vaut $\sqrt{d_k}/d_k = 1/8 = 0{,}125$ ; pour un écart de scores de $0{,}125$, $p = \sigma(0{,}125) \approx 0{,}531$ : presque l'égalité. L'attention ferait la moyenne de presque toutes les valeurs, sans rien sélectionner, et devrait apprendre de grands poids pour s'en sortir.
7. Parce que le modèle part de là : à l'initialisation, une softmax saturée bloque les gradients avant que $\mathbf{W}_Q$ ait pu apprendre à réduire l'échelle ; et une échelle fixe rend les learning rates et les initialisations transférables d'une taille de modèle à l'autre. Le paramètre `scale` permet une autre valeur, choisie en connaissance de cause.

**Erreurs fréquentes** : additionner les écarts-types au lieu des variances ; oublier l'indépendance, qui justifie $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ ; écrire $\operatorname{Var}(XY) = \operatorname{Var}X\operatorname{Var}Y$ sans le centrage (question 4) ; croire que diviser davantage est toujours mieux (question 6).
**Variante** : $\mathbf{q}$ a des composantes indépendantes, centrées, de variance 1, et $\mathbf{k}$ est un vecteur **fixé**. Réponse : $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = \sum_c k_c^2 \operatorname{Var}(q_c) = \lVert\mathbf{k}\rVert^2$ : une clé de grande norme donne des scores dispersés, et c'est l'une des raisons de normaliser requêtes et clés (QK-Norm de Qwen3, §203.6).

### Ex B3.7 — RoPE : la rotation rend q_m · k_n fonction de m − n seulement ∂
**Réponse** : $\langle R_{m\theta} \mathbf{q}, R_{n\theta} \mathbf{k} \rangle = \mathbf{q}^\top R_{-m\theta} R_{n\theta} \mathbf{k} = \langle \mathbf{q}, R_{(n-m)\theta}\mathbf{k} \rangle$ ; en dimension 2, $\langle \mathbf{q}, R_\varphi\mathbf{k}\rangle = (\mathbf{q} \cdot \mathbf{k})\cos\varphi + (q_2 k_1 - q_1 k_2)\sin\varphi$ ; application : $3$, $1$, $-1$, $-1$.
**Démarche** :
1. $\langle R_{m\theta} \mathbf{q}, R_{n\theta} \mathbf{k} \rangle = (R_{m\theta}\mathbf{q})^\top(R_{n\theta}\mathbf{k}) = \mathbf{q}^\top R_{m\theta}^\top R_{n\theta} \mathbf{k}$ (transposée d'un produit). Avec $R_{m\theta}^\top = R_{-m\theta}$ et $R_{-m\theta}R_{n\theta} = R_{(n-m)\theta}$ : $\mathbf{q}^\top R_{(n-m)\theta}\mathbf{k} = \langle \mathbf{q}, R_{(n-m)\theta}\mathbf{k}\rangle$. Seul l'écart $n - m$ reste.
2. $R_\varphi\mathbf{k} = (k_1\cos\varphi - k_2\sin\varphi ;\ k_1\sin\varphi + k_2\cos\varphi)$, d'où $\langle \mathbf{q}, R_\varphi\mathbf{k}\rangle = (q_1k_1 + q_2k_2)\cos\varphi + (q_2k_1 - q_1k_2)\sin\varphi$. Exemple de la fiche : $\mathbf{q} \cdot \mathbf{k} = 1$ et $q_2k_1 - q_1k_2 = 7$ ; pour l'écart $+3$, $\varphi = 1{,}5$ : $\cos 1{,}5 + 7\sin 1{,}5 \approx 0{,}0707 + 6{,}9825 = 7{,}0532$ ; pour l'écart $-3$ : $0{,}0707 - 6{,}9825 \approx -6{,}9117$. Le terme en sinus change de signe avec l'écart : RoPE distingue « avant » et « après ».
3. $\mathbf{q} \cdot \mathbf{k} = 3$ et $q_2k_1 - q_1k_2 = 1 - 2 = -1$. $(0, 0)$ : $\varphi = 0$, score $3$. $(5, 2)$ : $\varphi = -3 \times \pi/6 = -\pi/2$, score $3 \times 0 + (-1)(-1) = 1$. $(1, 4)$ et $(7, 10)$ : $\varphi = \pi/2$, score $-1$ deux fois : même écart, même score, quelle que soit la position absolue.
4. La matrice de la position $m$ est $\operatorname{diag}(R_{m\theta_0}, \dots, R_{m\theta_{d_h/2-1}})$. Le produit scalaire en dimension $d_h$ est la somme des produits scalaires des paires, et chaque paire vérifie la question 1 avec sa fréquence : $\langle R_{m\theta}\mathbf{q}, R_{n\theta}\mathbf{k}\rangle = \sum_i \langle \mathbf{q}^{(i)}, R_{(n-m)\theta_i}\mathbf{k}^{(i)}\rangle$, qui ne dépend que de $n - m$. (Avec la convention « rotate half », les paires sont $(i, i + d_h/2)$ au lieu de $(2i, 2i + 1)$ : une permutation des coordonnées, qui ne change pas l'argument.)
5. $\lVert R_{m\theta}\mathbf{q}\rVert^2 = \mathbf{q}^\top R_{m\theta}^\top R_{m\theta} \mathbf{q} = \mathbf{q}^\top \mathbf{q}$, bloc par bloc. RoPE ne change pas les normes, donc ni la variance des scores de ∂ B3.6 : la division par $\sqrt{d_k}$ reste la bonne.
6. $(\mathbf{x}_m + \mathbf{p}_m)\cdot(\mathbf{x}_n + \mathbf{p}_n) = \mathbf{x}_m\cdot\mathbf{x}_n + \mathbf{x}_m\cdot\mathbf{p}_n + \mathbf{p}_m\cdot\mathbf{x}_n + \mathbf{p}_m\cdot\mathbf{p}_n$. Seul le dernier ne dépend que de $n - m$ (✏️ B3.4 h) ; les deux termes croisés mêlent le contenu d'un token et la position **absolue** de l'autre. RoPE n'a pas ces termes : la position n'entre que par l'écart.
7. La clé rangée, $R_{n\theta}\mathbf{k}_n$, ne dépend que de la position $n$ de son token et de son contenu. Toute requête future, en position $m$, calcule $\langle R_{m\theta}\mathbf{q}_m, R_{n\theta}\mathbf{k}_n\rangle = \langle \mathbf{q}_m, R_{(n-m)\theta}\mathbf{k}_n\rangle$ : la bonne valeur, sans retoucher le cache.

**Erreurs fréquentes** : écrire $(R\mathbf{q})^\top = R\,\mathbf{q}^\top$ au lieu de $\mathbf{q}^\top R^\top$ ; obtenir $R_{(m-n)\theta}$ au lieu de $R_{(n-m)\theta}$ (un signe qui change le terme en sinus) ; croire que le score ne dépend que de $|n - m|$ ; oublier que chaque paire a sa propre fréquence ; tourner aussi les valeurs.
**Variante** : $\theta = \pi/4$, $\mathbf{q} = (1, 0)$, $\mathbf{k} = (0, 1)$, $(m, n) = (0, 2)$. Réponse : $\varphi = \pi/2$, $\mathbf{q} \cdot \mathbf{k} = 0$ et $q_2k_1 - q_1k_2 = -1$ : le score vaut $-1$, alors que $\mathbf{q}$ et $\mathbf{k}$ sont orthogonaux sans rotation.

### Ex B3.8 — Fermi : FLOPs d'entraînement (≈ 6 N D), ton mini-GPT sur CPU, GPT-3, et la mémoire du KV cache 🧮
**Réponse** (fourchettes acceptées) : mode rapide ≈ 2,5 millions de tokens, ≈ $1{,}2 \times 10^{13}$ FLOPs, ≈ 2 min (de 1 à 5 min selon le débit réel) ; mode complet ≈ $10^{14}$ FLOPs, ≈ 15 à 20 min ; GPT-3 ≈ $3 \times 10^{23}$ FLOPs, ≈ 300 ans sur un GPU, ≈ 4 mois sur 1 000 ; KV cache du mini-GPT 512 Kio par séquence ; Qwen3-8B 144 Kio par token, 4,5 Gio à 32 768 tokens, ≈ 50 000 tokens de cache sur 24 Go.
**Démarche** :
1. Mode rapide : $600 \times 32 \times 128 = 2\,457\,600 \approx 2{,}5 \times 10^6$ tokens. Avec $N = 820\,992$ : $6ND \approx 6 \times 8{,}2 \times 10^5 \times 2{,}46 \times 10^6 \approx 1{,}2 \times 10^{13}$ FLOPs, soit $\approx 120$ s à $10^{11}$ FLOP/s. Mode complet : $2{,}05 \times 10^7$ tokens, $\approx 1{,}0 \times 10^{14}$ FLOPs, $\approx 1\,000$ s, environ 17 min : au-delà des 15 minutes de CPU à partir desquelles le workbook marque une cellule 🚀 (GPU Colab conseillé). Sans les embeddings (convention de Kaplan, $N = 793\,344$), 3 % de moins : sans importance à cette précision.
2. $T/(12d) = 128/1\,536 \approx 8\,\%$ : à peine plus que l'incertitude sur le débit du processeur ; la réponse 1 ne change pas d'ordre de grandeur.
3. $6 \times 1{,}75 \times 10^{11} \times 3 \times 10^{11} = 3{,}15 \times 10^{23}$ FLOPs (le tableau D.1 de Brown et coll. donne $3{,}14 \times 10^{23}$). À $3 \times 10^{13}$ FLOP/s : $1{,}05 \times 10^{10}$ s, environ 330 ans ; sur 1 000 GPU, $1{,}05 \times 10^{7}$ s, environ 120 jours. C'est $2{,}6 \times 10^{10}$ fois le calcul de ton mini-GPT en mode rapide.
4. $20 \times 1{,}75 \times 10^{11} = 3{,}5 \times 10^{12}$ tokens, près de 12 fois les 300 milliards utilisés : GPT-3 était sous-entraîné au sens de Chinchilla. Le calcul monterait à $6 \times 1{,}75 \times 10^{11} \times 3{,}5 \times 10^{12} \approx 3{,}7 \times 10^{24}$ FLOPs, ou bien, à budget égal, on aurait pris un modèle plus petit entraîné plus longtemps.
5. $2 \times 4 \times 128 \times 128 \times 4 = 524\,288$ octets $= 512$ Kio par séquence ; pour 32 séquences, 16 Mio. Rien, à côté des 3,3 Mo de poids en float32.
6. Par token : $2 \times 36 \times (8 \times 128) \times 2 = 147\,456$ octets $= 144$ Kio. À 32 768 tokens : $4{,}5$ Gio. Il reste environ $24 - 16 = 8$ Go, soit $8 \times 10^9 / 147\,456 \approx 54\,000$ tokens de cache (avec 8 Gio, $2^{33}$ octets, on trouverait 58 000 tokens et 14 conversations : d'où la convention de l'énoncé) : une seule conversation de 32 768 tokens, ou environ 13 conversations de 4 096 tokens. Sans GQA, 4 fois plus par token (576 Kio) : environ 13 500 tokens, 3 conversations de 4 096.
7. Le calcul hors produits matriciels (softmax, layer norms, GELU, optimiseur), le temps de chargement des données, les FLOPs « perdus » (un débit réel bien sous le débit annoncé, souvent 30 à 50 % sur GPU), l'évaluation, les essais ratés et les réglages des hyperparamètres, la mémoire des activations à l'entraînement, et, pour le cache, la fragmentation de la mémoire (que PagedAttention réduit) et les tampons du moteur d'inférence.

**Erreurs fréquentes** : compter les itérations au lieu des tokens ; prendre 2 au lieu de 6 (la passe avant seule) ; oublier le facteur 2 des clés et des valeurs, ou le nombre de couches ; prendre $d$ au lieu de $n_{kv} d_h$ pour un modèle à GQA ; mélanger Go ($10^9$) et Gio ($2^{30}$) sans le dire.
**Variante** : Llama 3 8B, entraîné sur environ 15 000 milliards de tokens. Réponse : $6 \times 8 \times 10^9 \times 1{,}5 \times 10^{13} \approx 7{,}2 \times 10^{23}$ FLOPs, plus que GPT-3 pour un modèle 22 fois plus petit : environ 1 900 tokens par paramètre, bien au-delà des 20 de Chinchilla, pour un modèle bon marché à utiliser.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B3.9 — L'attention expliquée en cinq lignes 🗣️
**Réponse modèle** : (1) Pour comprendre un mot, il faut souvent regarder d'autres mots de la phrase, parfois loin : dans « le chat que Marie a adopté dort », « dort » a besoin de « chat ». (2) Chaque mot prépare trois choses : une question (la requête : « qui fait l'action ? »), un titre qui dit ce qu'il contient (la clé) et un contenu à transmettre (la valeur). (3) On compare la question d'un mot aux titres de tous les mots : plus ils se ressemblent, plus le poids est grand, et les poids font 100 % en tout. (4) Le mot repart avec un mélange des contenus de tous les mots, dosé par ces poids : surtout celui de « chat ». (5) Tous les mots le font en même temps, sans relire la phrase mot après mot : c'est rapide, et deux mots éloignés se parlent directement.
**Ce qui compte** : un besoin d'information ailleurs dans la phrase ; trois rôles distincts ; des poids de ressemblance qui font 100 % ; un mélange des contenus ; le parallélisme et la distance qui ne compte plus.
**Erreurs fréquentes** : du jargon non expliqué (« softmax », « produit scalaire », « embedding ») ; dire que le modèle « choisit » un seul mot ; confondre la clé et la valeur ; oublier ce que l'on gagne par rapport à une lecture mot après mot.
**Variante** : la même idée en une phrase, pour un recruteur. Réponse possible : « Chaque token compare sa requête aux clés de tous les autres, transforme les ressemblances en poids par une softmax et récupère la moyenne pondérée de leurs valeurs, en parallèle sur toute la séquence. »

### Ex B3.10 — Mémorisation : quand un modèle de langue recrache ses données d'entraînement ⚖️
**Réponse** :
1. **Mémoriser**, c'est reproduire un passage précis des données ; **généraliser**, c'est produire un texte plausible qui n'y est pas. Protocole : 200 passages tirés de la partie entraînement de Holmes, leurs 50 premiers caractères en prompt, 50 caractères générés en prenant chaque fois le plus probable (une génération déterministe, que le tirage ne masque pas), et la part des passages dont la suite est reproduite exactement (ou la longueur moyenne du préfixe commun). Le témoin : 200 passages de la validation, que le modèle n'a jamais vus ; certaines suites sont faciles à deviner (« Sherlock » après « Mr. »), et le témoin mesure ce hasard. Une part bien plus haute sur l'entraînement que sur la validation, surtout pour des passages rares, signale de la mémorisation ; elle doit croître avec le nombre de passages sur les données (mode complet).
2. Carlini et coll. (2021) ont extrait de GPT-2 des centaines de séquences recopiées mot pour mot de ses données, dont des noms, des numéros de téléphone, des adresses électroniques, des conversations IRC, du code et des identifiants aléatoires ; certaines n'apparaissaient que dans **un seul** document des données ; et les grands modèles sont **plus** vulnérables que les petits.
3. Carlini et coll. (2022) trouvent trois relations log-linéaires : la mémorisation croît avec la **capacité** du modèle, le nombre de **répétitions** d'un exemple dans les données et la **longueur du contexte** du prompt. Pour l'assureur : des formules de politesse, des numéros de contrat dans des conversations qui se répètent, beaucoup de passages sur les données : tout pousse à la mémorisation, en particulier des passages répétés.
4. Pour les clients : un conseiller, ou un client si l'outil s'ouvre, peut voir ressortir le nom, le numéro ou le sinistre d'un autre client ; une fuite de données personnelles. Pour l'assureur : une violation du RGPD (minimisation : n'utiliser que les données nécessaires ; sécurité), et le droit à l'effacement (art. 17) devient difficile : effacer une conversation de la base ne la retire pas des poids ; il faut réentraîner sans elle (ou des méthodes de « désapprentissage » encore imparfaites), donc prévoir des versions et des données traçables.
5. **Avant** : dédupliquer les données (Lee et coll., 2022 : dix fois moins de texte mémorisé recopié) ; retirer ou remplacer les données personnelles par des marqueurs (`<NOM>`, `<TEL>`), au prix d'un modèle qui écrit moins naturellement les noms ; ne garder que le nécessaire. **Pendant** : un entraînement à confidentialité différentielle (DP-SGD : gradients rognés et bruités par exemple), qui borne l'influence de chaque conversation, au prix de la qualité et du calcul ; moins de passages sur les données. **Après** : un test de mémorisation avant la mise en service, dont des « canaris » (Carlini et coll., 2019 : insérer des chaînes secrètes aléatoires et mesurer si le modèle les recrache) ; un filtre de sortie qui bloque les numéros et les noms, au prix de faux positifs ; un accès réservé aux conseillers, journalisé et limité.
6. Un corpus protégé ajoute le risque de recopier des passages d'œuvres, donc une reproduction sans autorisation, et la question de la licéité de la copie d'entraînement (la fouille de textes de la directive 2019/790, sauf opposition des titulaires, ch. 28). L'article 53 du règlement 2024/1689 impose aux fournisseurs de modèles d'IA à usage général, depuis le 2 août 2025, une politique de respect du droit d'auteur de l'Union (qui respecte ces oppositions) et la publication d'un résumé suffisamment détaillé des contenus d'entraînement.
7. « Limites et risques » : « Entraîné sur des conversations clients pseudonymisées et dédupliquées ; des données personnelles résiduelles peuvent ressortir. Test de mémorisation (200 prompts, canaris) : X % de suites recopiées, contre Y % sur des conversations jamais vues. Usage interne par les conseillers seulement, avec un filtre de numéros et de noms en sortie. Le test ne couvre pas les attaques ciblées : ne pas exposer le modèle aux clients. »

**Erreurs fréquentes** : mesurer avec un tirage aléatoire, qui cache la mémorisation ; oublier le groupe témoin ; croire qu'une donnée vue une fois ne peut pas ressortir ; croire que retirer une ligne de la base l'efface du modèle ; ne proposer que des filtres de sortie.
**Variante** : pourquoi un modèle de caractères de Holmes est-il un cas facile ? Réponse : Holmes est dans le domaine public et ne contient aucune donnée personnelle réelle ; on peut donc mesurer la mémorisation sans risque, et la faire varier (taille du modèle, nombre de passages, longueur du prompt) pour retrouver, en petit, les trois tendances de Carlini et coll. (2022).

### Ex B3.11 — Vaswani et coll. (2017) : « Attention Is All You Need » 📄
**Réponse** :
1. Un réseau récurrent calcule l'état $\mathbf{h}_t$ à partir de $\mathbf{h}_{t-1}$ : cette nature séquentielle empêche de paralléliser à l'intérieur d'un exemple, ce qui coûte d'autant plus que les séquences s'allongent et que la mémoire limite la taille des batches. Le Transformer atteint un nouvel état de l'art en traduction après **douze heures** d'entraînement sur **huit GPU P100**.
2. $N = 6$ couches identiques dans l'encodeur et dans le décodeur. Encodeur : une auto-attention multi-têtes, puis un MLP appliqué à chaque position. Décodeur : la même chose, avec entre les deux une attention multi-têtes sur la sortie de l'encodeur (l'attention croisée). Chaque sous-couche est entourée d'une connexion résiduelle suivie d'une layer norm : la sortie est `LayerNorm(x + Sublayer(x))`, un bloc **post-LN**. Deux précautions : le masque de l'auto-attention du décodeur (une position ne voit pas les suivantes) et le décalage d'une position des embeddings de sortie (le teacher forcing) ; ensemble, la prédiction de la position $i$ ne dépend que des positions avant $i$.
3. $\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\,\mathbf{V}$. L'attention additive calcule la compatibilité par un petit réseau à une couche cachée ; leur complexité théorique est voisine, mais le produit scalaire s'écrit en produits matriciels très optimisés, donc il est bien plus rapide et économe. Sans mise à l'échelle, l'attention additive fait mieux pour de grands $d_k$ ; les auteurs soupçonnent des produits scalaires trop grands qui poussent la softmax vers des gradients très petits. La note 4 : pour des composantes indépendantes de moyenne 0 et de variance 1, $\mathbf{q} \cdot \mathbf{k}$ a une moyenne 0 et une variance $d_k$ (∂ B3.6 le démontre).
4. $h = 8$, $d_k = d_v = d_{\text{model}}/h = 64$ ; $\mathbf{W}^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$, soit $(512, 512)$. Une attention multi-têtes avec biais : $4 \times 512^2 + 4 \times 512 = 1\,050\,624$ paramètres. Trois usages : l'attention encodeur-décodeur (requêtes du décodeur, clés et valeurs de l'encodeur), l'auto-attention de l'encodeur, l'auto-attention masquée du décodeur.
5. Auto-attention : $O(n^2 d)$ par couche, $O(1)$ opérations séquentielles, chemin maximal $O(1)$ ; récurrente : $O(n d^2)$, $O(n)$, $O(n)$. L'auto-attention est moins coûteuse quand $n < d$, le cas courant des phrases en BPE (pour $d = 512$, des phrases de moins de 512 tokens). Pour les très longues séquences, les auteurs proposent une attention restreinte à un voisinage de taille $r$, en $O(r n d)$, au prix d'un chemin maximal qui remonte à $O(n/r)$.
6. La même matrice de poids sert aux deux couches d'embedding (source et cible) et à la transformation linéaire avant la softmax : des poids liés ; dans les embeddings, les poids sont multipliés par $\sqrt{d_{\text{model}}}$. Les longueurs d'onde vont de $2\pi$ à $10000 \cdot 2\pi$, en progression géométrique. Le choix des sinusoïdes : pour tout décalage fixe $k$, $PE_{pos+k}$ est une fonction linéaire de $PE_{pos}$ (la rotation de ✏️ B3.4), ce qui devrait aider le modèle à regarder par positions relatives, et peut-être à extrapoler au-delà des longueurs vues. La ligne (E) du tableau 3 : des positions apprises donnent des résultats presque identiques (BLEU 25,7 contre 25,8, perplexité 4,92).
7. WMT 2014 anglais-allemand (environ 4,5 millions de paires de phrases, un BPE partagé d'environ 37 000 tokens) et anglais-français (36 millions de phrases, 32 000 word-pieces) ; des batches d'environ 25 000 tokens source et 25 000 tokens cible. Modèle de base : 0,4 s par pas, 100 000 pas, 12 heures ; grand modèle : 1,0 s par pas, 300 000 pas, 3,5 jours. Adam ($\beta_1 = 0{,}9$, $\beta_2 = 0{,}98$, $\epsilon = 10^{-9}$), avec $\text{lrate} = d_{\text{model}}^{-0{,}5} \cdot \min(\text{step}^{-0{,}5}, \text{step} \cdot \text{warmup}^{-1{,}5})$ : une montée linéaire pendant 4 000 pas, puis une décroissance en $1/\sqrt{\text{step}}$ ; au sommet, $512^{-0{,}5} \times 4\,000^{-0{,}5} \approx 0{,}0442 \times 0{,}0158 \approx 7{,}0 \times 10^{-4}$. Régularisation : dropout de 0,1 sur la sortie de chaque sous-couche et sur la somme des embeddings et des positions, et un label smoothing de 0,1, qui **dégrade** la perplexité (le modèle apprend à être moins sûr) mais améliore l'accuracy et le BLEU.
8. BLEU : 28,4 en anglais-allemand et 41,8 en anglais-français pour le grand modèle (27,3 et 38,1 pour le modèle de base). Le résumé et le tableau 2 donnent 41,8 pour l'anglais-français, mais la section 6.1 dit 41,0 : une incohérence de l'article lui-même. Les versions arXiv de juin 2017 (v1 à v4) donnent 41,0 dans le tableau 2 ; la version 5, de décembre 2017, a porté le résumé et le tableau à 41,8 sans corriger la section 6.1, restée telle quelle jusqu'à la version 7 que tu lis. L'article ne dit pas d'où vient le nouveau chiffre : cite le tableau avec sa version, et signale l'écart. Le coût du modèle de base : $9{,}5 \times 10^{12}$ FLOP/s (note 5, P100) $\times 8$ GPU $\times 12 \times 3\,600$ s $\approx 3{,}3 \times 10^{18}$ FLOPs, le chiffre du tableau 2 ; le grand modèle, $9{,}5 \times 10^{12} \times 8 \times 3{,}5 \times 86\,400 \approx 2{,}3 \times 10^{19}$. Lignes (A) : à calcul constant, une seule tête perd 0,9 BLEU (24,9 contre 25,8), et trop de têtes (32 de taille 16) dégradent aussi la qualité (25,4). Le compte : une couche d'encodeur $= 1\,050\,624 + 2\,099\,712 + 2\,048 = 3\,152\,384$ ; une couche de décodeur $= 2 \times 1\,050\,624 + 2\,099\,712 + 3\,072 = 4\,204\,032$ ; les 12 couches $= 44\,138\,496$ ; une matrice d'embedding partagée $37\,000 \times 512 = 18\,944\,000$ ; total $63\,082\,496 \approx 63{,}1$ millions, à 3 % des 65 millions du tableau 3. L'article ne donne pas son compte, et l'écart (environ 1,9 million) reste inexpliqué : des layer norms finales n'ajouteraient que $2 \times 1\,024$ paramètres ; il faudrait environ 3 800 lignes d'embedding de plus (un vocabulaire d'environ 41 000 tokens au lieu des « environ 37 000 » de la section 5.1), ou des paramètres que l'article ne décrit pas. L'article seul ne permet pas de trancher : on retient que l'architecture décrite donne 63,1 millions, et on le dit.
9. Les LLM de 2026 sont des **décodeurs seuls** (sans encodeur ni attention croisée) ; **pré-norm**, avec RMSNorm au lieu du post-LN ; un MLP **SwiGLU** au lieu de ReLU ; des positions **RoPE** au lieu de sinusoïdes ajoutées ; la **GQA** ; pas de biais ; des contextes de 128 000 tokens et plus ; souvent des mélanges d'experts.

**Erreurs fréquentes** : prendre le bloc de l'article pour un pré-LN ; attribuer à l'article RoPE ou GELU ; confondre $d_{ff} = 2\,048$ et $d_{\text{model}} = 512$ ; lire 41,0 dans la section 6.1 sans voir l'écart avec le résumé ; recalculer le coût d'entraînement avec $6ND$ sans dire que le tableau 2 multiplie un temps par un débit.
**Variante** : estime aussi le coût du modèle de base par $6ND$, avec $N = 65$ millions et environ 25 000 tokens cible par pas pendant 100 000 pas. Réponse : $6 \times 6{,}5 \times 10^7 \times 2{,}5 \times 10^9 \approx 10^{18}$ FLOPs. On compte $D$ une seule fois : chaque paramètre ne voit qu'un flux, l'encodeur les tokens source, le décodeur et la couche de sortie les tokens cible, environ 25 000 de chaque par pas. C'est 3 fois moins que les $3{,}3 \times 10^{18}$ du tableau 2, qui suppose les huit GPU occupés à leur pleine capacité pendant les 12 heures (note 5) : une borne haute du calcul utile, qui compte aussi le temps perdu hors des produits matriciels.

<a id="entretien"></a>

## 💼 Entretien

### B3.E1 — Explique le mécanisme d'attention, de la requête à la sortie
**Réponse modèle en 60 secondes** : « Chaque token arrive avec un vecteur de dimension $d$, son embedding plus sa position. Trois projections apprises en tirent une requête, une clé et une valeur. Pour chaque requête, on calcule son produit scalaire avec toutes les clés : ce sont les scores, que l'on divise par $\sqrt{d_k}$, parce que le produit scalaire de vecteurs aléatoires a une variance $d_k$ et qu'une softmax de scores trop étalés sature. On ajoute un masque, $-\infty$ là où c'est interdit (le futur pour un GPT, le padding), puis une softmax sur les clés donne des poids de somme 1. La sortie est la moyenne des valeurs pondérée par ces poids. En multi-têtes, on fait cela $H$ fois en parallèle sur des sous-espaces de taille $d/H$, avec un découpage reshape puis transpose, on recolle les têtes et une projection de sortie les mélange. Le tout coûte $4d^2$ paramètres, quel que soit $H$, et un calcul en $T^2 d$ dans la longueur. »
**Ce qui compte** : les trois projections ; les scores et la raison de $\sqrt{d_k}$ ; le masque et la softmax sur les clés ; la moyenne des valeurs ; les têtes, leur découpage et $\mathbf{W}_O$ ; le coût.
**Relances possibles** : « Pourquoi plusieurs têtes plutôt qu'une grande ? » (chaque tête a sa carte d'attention et peut suivre une relation différente pour le même coût ; une seule tête ne fait qu'une moyenne par token) · « Que se passe-t-il si une ligne est entièrement masquée ? » (la softmax divise 0 par 0 : `nan` dans `nn.MultiheadAttention`, une ligne de zéros dans `F.scaled_dot_product_attention` ; c'est presque toujours un bug de masque, qu'il vaut mieux détecter)

### B3.E2 — Quel est le coût de l'auto-attention en fonction de la longueur, et comment traite-t-on les contextes longs ?
**Réponse modèle en 60 secondes** : « Chaque token est comparé à tous les autres : le calcul de l'attention croît comme $T^2 d$ et, dans une implémentation directe, la mémoire des scores comme $T^2$ par tête et par couche. Les couches denses, elles, coûtent $T d^2$ : dans un décodeur causal, l'attention ne domine qu'au-delà d'environ $12d$ tokens, 49 000 pour $d = 4\,096$. À l'entraînement, FlashAttention calcule l'attention exacte par blocs, avec une softmax en ligne, sans jamais écrire la matrice : mémoire linéaire, mêmes FLOPs, beaucoup moins de lectures. À l'inférence, le problème devient le KV cache, qui grandit avec le contexte : on partage les têtes de clés (GQA), on le quantifie, on le range par pages (vLLM). Pour aller à 128 000 tokens, on étire les fréquences de RoPE (YaRN), on mélange des couches à fenêtre glissante et des couches complètes, et certains modèles hybrides remplacent une partie de l'attention par des couches récurrentes à espace d'états, comme Mamba. »
**Ce qui compte** : $T^2$ en calcul et en mémoire ; le seuil $12d$ ; FlashAttention exact ; le KV cache et la GQA ; les fenêtres, l'étirement des positions et les hybrides.
**Relances possibles** : « FlashAttention réduit-il la complexité ? » (non : les FLOPs restent en $T^2$ ; il supprime le goulot des lectures et écritures en mémoire et rend la mémoire linéaire) · « Une fenêtre glissante perd-elle l'information lointaine ? » (dans une couche, oui ; en empilant les couches, l'information remonte de $w$ positions par couche, et les modèles alternent souvent fenêtre et attention complète)

### B3.E3 — Encodeur, décodeur ou encodeur-décodeur : lequel pour quelle tâche ?
**Réponse modèle en 60 secondes** : « Tout dépend de ce que le modèle doit produire. Pour classer des e-mails, un encodeur comme BERT ou ModernBERT, fine-tuné avec une tête de classification : il lit le texte dans les deux sens, il est petit, rapide et bon marché, et quelques milliers d'exemples annotés suffisent. Pour résumer des contrats, l'entrée et la sortie sont deux textes distincts : un encodeur-décodeur comme T5 ou BART est le choix classique, avec son attention croisée ; mais un décodeur instruit le fait aussi très bien, avec un long contexte. Pour l'assistant, un décodeur : il génère, suit des instructions, tient une conversation ; on le prend pré-entraîné et instruit, et on l'adapte par le prompt, du RAG ou un fine-tuning léger. La nuance de 2026 : un grand décodeur sait tout faire par le prompt, classification comprise ; on garde l'encodeur quand le volume, la latence et le coût comptent. »
**Ce qui compte** : la sortie attendue décide de la famille ; les masques et les objectifs de chaque famille ; le coût et la latence ; la polyvalence des décodeurs.
**Relances possibles** : « Et pour une recherche par le sens dans les e-mails ? » (un encodeur d'embeddings de phrases, comme en B2 : un vecteur par texte, comparé par cosinus) · « Pourquoi BERT ne génère-t-il pas de texte ? » (il est entraîné à deviner des tokens masqués avec le contexte des deux côtés, pas à prédire le suivant ; il n'a pas de masque causal, donc pas de génération de gauche à droite)

### B3.E4 — À quoi sert le KV cache, et combien de mémoire consomme-t-il ?
**Réponse modèle en 60 secondes** : « Un décodeur génère un token à la fois. Avec un masque causal, les clés et les valeurs des tokens déjà vus ne changent pas quand on en ajoute un : on les garde dans un cache, couche par couche, et seul le nouveau token passe dans le modèle, avec une requête qui regarde tout le cache. On évite ainsi de refaire un travail quadratique à chaque pas. Le prix, c'est la mémoire : deux tenseurs, clés et valeurs, par couche, par token et par feature de clé, soit $2 \times L \times T \times n_{kv} d_h \times$ octets. Pour un modèle de 8 milliards comme Llama 3.1 8B (32 couches, 8 têtes de clés de taille 128, bfloat16), c'est 128 Kio par token, environ 4 Gio à 32 000 tokens, pour une seule conversation, et quatre fois plus sans la GQA. On le réduit en partageant les têtes de clés (GQA, MQA), en le quantifiant en 8 ou 4 bits, en le rangeant par pages pour éviter la fragmentation (vLLM), ou avec une fenêtre glissante sur certaines couches. »
**Ce qui compte** : pourquoi les clés et les valeurs ne changent pas ; ce que l'on gagne ; la formule et un ordre de grandeur juste ; les réductions et leurs prix.
**Relances possibles** : « Pourquoi ne garde-t-on pas les requêtes ? » (une ancienne requête ne resservira jamais : seule la nouvelle position calcule sa sortie) · « Le décodage est-il limité par le calcul ? » (plutôt par la mémoire : à chaque token, on relit tous les poids et tout le cache pour peu d'opérations ; d'où le batching de plusieurs conversations et le décodage spéculatif)

### B3.E5 — Pourquoi un Transformer a-t-il besoin d'encodages de position, et qu'apporte RoPE ?
**Réponse modèle en 60 secondes** : « Sans information de position, l'auto-attention est équivariante par permutation : elle traite une phrase comme un sac de mots, et "le chien mord l'homme" ressemble à "l'homme mord le chien". Le Transformer de 2017 ajoute aux embeddings des sinusoïdes de fréquences géométriques, où un décalage de position est une rotation de chaque paire de colonnes ; GPT-2 et BERT apprennent une table de positions, simple mais limitée à sa longueur. RoPE, de RoFormer, ne touche pas aux embeddings : il tourne les requêtes et les clés, paire de coordonnées par paire, d'un angle proportionnel à la position. Comme $\langle R_{m\theta} \mathbf{q}, R_{n\theta} \mathbf{k}\rangle = \langle \mathbf{q}, R_{(n-m)\theta}\mathbf{k}\rangle$, le score ne dépend que de l'écart : une position relative obtenue par une opération absolue, sans paramètre, appliquée dans chaque couche, compatible avec le KV cache puisqu'on range les clés déjà tournées. Et ses fréquences s'étirent (interpolation, YaRN) pour allonger le contexte après l'entraînement : c'est pourquoi Llama, Qwen et Mistral l'utilisent. »
**Ce qui compte** : l'équivariance par permutation ; sinusoïdal et appris, avec leurs limites ; la rotation et la propriété relative ; aucun paramètre, chaque couche, le cache ; l'allongement du contexte.
**Relances possibles** : « Un décodeur causal sans position apprend-il quand même l'ordre ? » (en partie : le masque causal casse la symétrie, chaque position voit un nombre différent de clés, et Haviv et coll. (2022) montrent qu'il en tire une information de position ; on ne compte pas dessus) · « Deux implémentations de RoPE donnent des scores différents avec les mêmes poids : pourquoi ? » (deux conventions de paires, voisines ou moitiés « rotate half » ; elles sont équivalentes à une permutation des colonnes de $\mathbf{W}_Q$ et $\mathbf{W}_K$ près, à appliquer en convertissant les poids)

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes, exécutées, sont dans `05_solutions.ipynb`. Les chiffres ci-dessous viennent de ce notebook (mode rapide, CPU à deux cœurs) et de `tools/chapters/history_chB3.py` (calibrage du 🔮 B3.13 sur onze graines, rejeu du 🐛 B3.17 sur ses huit versions, mini-GPT sur trois graines et en mode complet, LSTM du ch. 22 réentraîné sur le découpage de B3) ; le découpage, le vocabulaire et les graines sont figés, donc tes chiffres sont les mêmes à l'arrondi près, sauf les durées, qui dépendent de la machine et sont données en fourchette.

### Ex B3.12 — causal_mask, padding_mask et scaled_dot_product_attention 🔨
**Réponse** : les 55 tests de l'exercice passent ; l'attention à la main de la fiche redonne les poids (0,1411 ; 0,1411 ; 0,6323 ; 0,0856) ; contre `F.scaled_dot_product_attention`, l'écart est de $3 \times 10^{-16}$ en float64.
**Démarche** :
```python
def causal_mask(q_len, kv_len=None):
    L = q_len
    S = L if kv_len is None else kv_len
    if L < 1 or S < L:
        raise ValueError(f"expected 1 <= q_len <= kv_len, got q_len = {L}, kv_len = {S}")
    return np.arange(S)[None, :] <= np.arange(L)[:, None] + (S - L)      # (L, S) bool, bottom-right


def padding_mask(lengths, max_len):
    lens = np.asarray(lengths)
    if lens.ndim != 1 or lens.size == 0 or lens.min() < 1 or lens.max() > max_len:
        raise ValueError(f"every length must be in [1, max_len = {max_len}], got {lens.tolist()}")
    return np.arange(max_len)[None, :] < lens[:, None]                   # (N, S) bool


def scaled_dot_product_attention(q, k, v, mask=None, scale=None):
    q, k, v = (np.asarray(a, dtype=float) for a in (q, k, v))
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError(f"incompatible shapes q {q.shape}, k {k.shape}, v {v.shape}")
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)    # scale=0.0 is a scale
    scores = (q @ np.swapaxes(k, -1, -2)) * scale                              # (..., L, S)
    if mask is not None:
        allowed = np.broadcast_to(np.asarray(mask, dtype=bool), scores.shape)  # ValueError if it does not broadcast
        if not allowed.any(axis=-1).all():
            raise ValueError("a query row has no allowed key (its mask row is all False)")
        scores = np.where(allowed, scores, -np.inf)                            # before the softmax
    weights = softmax(scores, axis=-1)                                         # your softmax of ch. 17
    return weights @ v, weights
```
Le masque causal est une comparaison entre la colonne des requêtes et la ligne des clés : la requête $i$ est le token de la position $i + S - L$, et elle voit les clés jusqu'à cette position. Avec `causal_mask(2, 5)`, la requête 0 est le token 3 et voit quatre clés, la requête 1 (le token 4) les cinq ; `is_causal=True`, aligné en haut à gauche, n'en laisserait qu'une et deux, comme si les trois clés en cache venaient après. Masquer **avant** la softmax met $e^{-\infty} = 0$ au numérateur et au dénominateur : la ligne somme encore à 1 parmi les clés autorisées, alors que mettre des poids à zéro après la softmax laisserait une somme inférieure à 1. Ta `softmax` retranche le maximum de la ligne, qui est fini dès qu'une clé est autorisée ; une ligne vide donnerait $-\infty - (-\infty)$, un `nan`. SDPA rend une ligne de zéros, `nn.MultiheadAttention` des `nan` : deux comportements différents pour une situation qui est presque toujours une erreur en amont (une séquence de longueur 0, un masque inversé), d'où la `ValueError`.
**Erreurs fréquentes** : `scale or 1 / math.sqrt(d_k)`, qui remplace `scale=0.0` par l'échelle par défaut ; le masque aligné en haut à gauche (`j <= i`), juste quand $L = S$ seulement ; `np.where(mask, scores, -np.inf)` sans `broadcast_to`, qui ne voit pas une ligne vide d'un masque `(N, 1, 1, S)` ; la softmax sur l'axe des requêtes ; `q.T` au lieu de `np.swapaxes(k, -1, -2)` (`.T` inverse **tous** les axes d'un tableau à 4 axes) ; un masque en 0/1 entiers, que `np.where` accepte mais pas le contrat.
**Variante** : un masque de fenêtre glissante (Mistral, §203.10), où chaque requête ne voit que ses $w$ dernières clés, s'écrit `(j <= i + S - L) & (j > i + S - L - w)`. Pour $L = S = 6$ et $w = 3$, il garde 15 cases sur 36 (1, 2, puis 3 par ligne), contre 21 pour le masque causal.

### Ex B3.13 — Prédire la carte d'attention quand une clé copie la requête, puis quand on retire la division par √d_k 🔮
**Réponse** : a) **3** : avec la division, le poids médian de la clé copiée vaut encore 0,619 avec 1 000 clés, puis 0,355 avec 3 000 et 0,143 avec 10 000 ; la bascule se fait vers 2 000 clés. b) **True** : sans la division, le poids médian vaut encore 1,000000 avec 100 000 clés. Sur onze graines de 200 requêtes, le poids médian à 1 000 clés va de 0,59 à 0,67, et de 0,13 à 0,17 à 10 000 : la réponse ne dépend pas du tirage.
**Démarche** : avec la division, le score de la clé copiée vaut $\lVert\mathbf{q}\rVert^2/\sqrt{d_k} \approx d_k/\sqrt{d_k} = 8$, et les scores des autres clés sont centrés, de variance 1 (∂ B3.6). La somme de leurs exponentielles vaut environ $(S - 1)\,\mathbb{E}[e^{Z}] = (S - 1)\,e^{1/2}$ (la moyenne d'une loi log-normale), d'où la bascule pour $e^{8} \approx (S - 1)\,e^{0{,}5}$, soit $S \approx 1 + e^{7{,}5} \approx 1\,800$ : entre $10^3$ et $10^4$. Sans la division, le score copié vaut $\lVert\mathbf{q}\rVert^2 \approx 64$, et les autres ont une variance de 64 : $(S - 1)\,e^{32} = e^{64}$ donne $S \approx e^{32} \approx 8 \times 10^{13}$ clés, hors de portée de toute expérience. La division par $\sqrt{d_k}$ ramène les scores **aléatoires** à une variance 1 ; elle ne ramène pas le score d'une clé alignée sur la requête, qui reste de l'ordre de $\sqrt{d_k}$ : une clé vraiment pertinente garde un poids net, même parmi des centaines de clés. Les cartes le montrent à petite échelle : avec 32 clés, la clé copiée prend de 0,90 à 0,99 du poids de sa ligne avec la division, et 1,000 sans.
**Erreurs fréquentes** : croire que la division « aplatit » la carte au point qu'une clé copiée se perd parmi quelques dizaines de clés ; prendre $\mathbb{E}[e^{Z}] = e^{\mathbb{E}Z} = 1$ (l'exponentielle est convexe : la moyenne des exponentielles dépasse l'exponentielle de la moyenne), ce qui donne $e^8 \approx 3\,000$ clés, la même réponse ici mais un raisonnement faux ; penser que des clés plus nombreuses finissent toujours par diluer la clé copiée sans la division : il en faudrait $10^{13}$.
**Variante** : avec $d_k = 16$, le score copié vaut $16/4 = 4$ et la bascule tombe vers $1 + e^{3{,}5} \approx 34$ clés (mesuré : 0,501 à 30 clés, 0,224 à 100) ; avec $d_k = 256$, vers $e^{15{,}5} \approx 5 \times 10^6$ (mesuré : encore 0,849 à un million de clés). Plus les têtes sont larges, plus une clé alignée l'emporte nettement, même avec la division.

### Ex B3.14 — split_heads, merge_heads et multi_head_attention 🔨
**Réponse** : les 48 tests de l'exercice passent ; ton attention multi-têtes rend la sortie et les poids par tête de `nn.MultiheadAttention` aux mêmes poids, en auto-attention causale comme en attention croisée avec padding.
**Démarche** :
```python
def split_heads(x, n_heads):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[-1] % n_heads:
        raise ValueError(f"expected x of shape (N, T, D) with n_heads = {n_heads} dividing D, got {x.shape}")
    N, T, D = x.shape
    return x.reshape(N, T, n_heads, D // n_heads).transpose(0, 2, 1, 3)    # (N, H, T, d_h)


def merge_heads(x):
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"expected x of shape (N, H, T, d_h), got {x.shape}")
    N, H, T, d_h = x.shape
    return x.transpose(0, 2, 1, 3).reshape(N, T, H * d_h)


def multi_head_attention(query, key, value, params, n_heads, mask=None):
    missing = [n for n in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o") if n not in params]
    if missing:
        raise ValueError(f"missing parameter(s): {', '.join(missing)}")
    q_in, k_in, v_in = (np.asarray(a, dtype=float) for a in (query, key, value))
    if q_in.ndim != 3 or k_in.ndim != 3 or v_in.shape != k_in.shape or k_in.shape[0::2] != q_in.shape[0::2]:
        raise ValueError(f"incompatible shapes query {q_in.shape}, key {k_in.shape}, value {v_in.shape}")
    P = {name: np.asarray(params[name], dtype=float) for name in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o")}
    Q = split_heads(q_in @ P["W_q"] + P["b_q"], n_heads)
    K = split_heads(k_in @ P["W_k"] + P["b_k"], n_heads)
    V = split_heads(v_in @ P["W_v"] + P["b_v"], n_heads)
    heads, weights = scaled_dot_product_attention(Q, K, V, mask=mask)     # scale 1/sqrt(d_h), the mask broadcast
    return merge_heads(heads) @ P["W_o"] + P["b_o"], weights
```
Le `reshape(N, T, H, d_h)` découpe l'axe des features en $H$ blocs consécutifs ; le `transpose` fait passer l'axe des têtes avant celui du temps, pour que les deux derniers axes soient $(T, d_h)$, la forme qu'attend l'attention. Un reshape direct vers `(N, H, T, d_h)` a la bonne forme mais mélange les tokens : la tête 0 reçoit les features de plusieurs tokens. La convention de `mylearn` ($z = xW + b$, $W$ de forme `(n_in, n_out)`) est la transposée de celle de PyTorch (`nn.Linear` calcule $x W^\top + b$ avec un poids `(n_out, n_in)`) : `params_from_mha` transpose donc chaque bloc de `in_proj_weight`. L'échelle par défaut de ta SDPA, $1/\sqrt{d_h}$ ici, est la bonne : chaque tête fait un produit scalaire en dimension $d_h$, de variance $d_h$ (∂ B3.6). Avec $1/\sqrt{D}$, les scores seraient $\sqrt{H}$ fois trop petits, des cartes plus floues que celles de PyTorch.
**Erreurs fréquentes** : le reshape direct ; lire `params["W_q"]` avant de vérifier les clés (un `KeyError` au lieu de la `ValueError` du contrat) ; recopier la convention de PyTorch (`x @ W.T`) ; calculer $\mathbf{K}$ et $\mathbf{V}$ à partir de `query` (juste en auto-attention, faux en attention croisée) ; reconstruire le masque en `(1, 1, L, S)` d'office, ce qui casse un masque `(N, 1, 1, S)`.
**Variante** : un masque de padding `(N, S)` passé tel quel, sans le remodeler en `(N, 1, 1, S)`, se diffuse sur les axes `(L, S)` des scores, pas sur `(N, S)`. Avec $N = 2$ séquences et $L = 5$ requêtes, NumPy lève une erreur (2 contre 5) ; mais avec $N = L$, aucune erreur, et chaque requête $i$ reçoit le masque de la séquence $i$ : un masque faux sans message. Le remodelage `mask[:, None, None, :]` est donc une précaution, pas un détail.

### Ex B3.15 — positional_encoding et apply_rope 🔨
**Réponse** : les 40 tests de l'exercice passent ; décaler de 7 positions applique bien la même rotation à chaque paire (sin, cos), et RoPE coïncide avec `LlamaRotaryEmbedding` de transformers à $10^{-5}$ près (écart mesuré $7 \times 10^{-8}$, celui du float32).
**Démarche** :
```python
def positional_encoding(seq_len, d_model, base=10000.0):
    if seq_len < 1 or d_model < 1 or d_model % 2 or base < 1:
        raise ValueError(f"expected seq_len >= 1, an even d_model >= 1 and base >= 1, got {seq_len}, {d_model}, {base}")
    angles = np.arange(seq_len)[:, None] / base ** (np.arange(0, d_model, 2) / d_model)   # (T, D/2)
    pe = np.empty((seq_len, d_model))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angles), np.cos(angles)
    return pe


def apply_rope(x, positions=None, base=10000.0):
    x = np.asarray(x, dtype=float)
    T, d = x.shape[-2:]
    if d % 2:
        raise ValueError(f"the last axis d must be even, got {d}")
    pos = np.arange(T, dtype=float) if positions is None else np.asarray(positions, dtype=float)
    if pos.shape != (T,):
        raise ValueError(f"positions must have shape ({T},), got {pos.shape}")
    theta = pos[:, None] * base ** (-2.0 * np.arange(d // 2) / d)        # (T, d/2)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]                             # dimension i with i + d/2
    return np.concatenate([x1 * np.cos(theta) - x2 * np.sin(theta), x2 * np.cos(theta) + x1 * np.sin(theta)], axis=-1)
```
Les deux fonctions partent de la même matrice d'angles $p\,\omega_i$. L'encodage sinusoïdal les **ajoute** aux embeddings, une fois, à l'entrée ; RoPE **tourne** les requêtes et les clés de chaque couche. Le produit scalaire d'une requête à la position $m$ et d'une clé à la position $n$ ne dépend alors que de $m - n$ (∂ B3.7) : décaler toutes les positions de 1 000 ne change aucun score (à $2 \times 10^{-14}$ près), ce que la vérification mesure sur trois couples. Les valeurs ne sont pas tournées : elles ne servent pas à comparer des positions, elles sont le contenu rendu. La convention « rotate half » apparie $i$ et $i + d/2$ plutôt que $2i$ et $2i + 1$ : la propriété relative est la même (elle vaut paire par paire), mais les deux conventions ne donnent pas les mêmes nombres, et des poids appris dans l'une sont faux dans l'autre (une erreur classique en convertissant un modèle).
**Erreurs fréquentes** : apparier $2i$ et $2i + 1$ (la convention de l'article, pas celle de Hugging Face) ; une fréquence en $\text{base}^{-i/d}$ ou $\text{base}^{-2i/(d/2)}$ ; échanger sinus et cosinus ; tourner dans l'autre sens ; laisser NumPy diffuser une seule position sur $T$ vecteurs ; calculer les angles en float32 pour de grandes positions (une erreur d'environ $10^{-5}$ sur les cosinus à la position 1 000, déjà au-delà de la tolérance des tests).
**Variante** : $\mathbf{q} = (1, 0, 0, 0)$ à la position 1, avec $d = 4$ : la paire $(q_0, q_2) = (1, 0)$ tourne de 1 radian, la paire $(q_1, q_3) = (0, 0)$ de 0,01 radian ; RoPE rend $(\cos 1, 0, \sin 1, 0) \approx (0{,}5403 ;\ 0 ;\ 0{,}8415 ;\ 0)$.

### Ex B3.16 — transformer_block en NumPy, validé contre nn.TransformerEncoderLayer 🔨
**Réponse** : les 25 tests de l'exercice passent ; ton bloc rend la sortie de `nn.TransformerEncoderLayer(norm_first=True)` aux mêmes poids, avec et sans masque causal, et avec `eps = 0.1`, à $10^{-6}$ près (l'écart mesuré est de $4 \times 10^{-16}$ en float64).
**Démarche** :
```python
BLOCK_KEYS = ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "W_1", "b_1", "W_2", "b_2")


def transformer_block(x, params, n_heads, mask=None, eps=1e-5):
    missing = [n for n in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o", *BLOCK_KEYS) if n not in params]
    if missing:
        raise ValueError(f"missing parameter(s): {', '.join(missing)}")
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or np.shape(params["W_1"])[0] != x.shape[-1]:
        raise ValueError(f"expected x (N, T, D) and W_1 (D, F), got {x.shape} and {np.shape(params['W_1'])}")
    a = layer_norm(x, params["ln1_gamma"], params["ln1_beta"], eps=eps)            # your layer_norm (20.17)
    h = x + multi_head_attention(a, a, a, params, n_heads, mask=mask)[0]           # first residual sub-layer
    m = layer_norm(h, params["ln2_gamma"], params["ln2_beta"], eps=eps)
    return h + gelu(m @ params["W_1"] + params["b_1"]) @ params["W_2"] + params["b_2"]   # exact GELU (17.14)
```
Chaque sous-couche lit une version normalisée du chemin résiduel et **ajoute** son résultat ; le chemin lui-même n'est jamais normalisé, ce qui garde un gradient direct de la sortie à l'entrée (§203.6). Avec tous les poids à zéro, les deux branches rendent 0 et le bloc rend $\mathbf{x}$ : un bloc qui ne sait encore rien laisse passer l'information, au lieu de la détruire. PyTorch range ses couches comme `nn.Linear` ($x W^\top + b$), d'où le `weight.T` de `block_params_from_torch`, et il attend son masque dans la convention de `nn.MultiheadAttention`, d'où le `~mask`. Un post-LN normaliserait la somme : $\mathbf{h} = \operatorname{LN}_1(\mathbf{x} + \operatorname{MHA}(\mathbf{x}))$, puis $\operatorname{LN}_2(\mathbf{h} + \operatorname{MLP}(\mathbf{h}))$. Remplacer la GELU exacte par l'approximation tanh déplace la sortie de $9{,}5 \times 10^{-5}$ au plus ici : bien au-delà de la tolérance de $10^{-6}$, donc un test l'aurait vu.
**Erreurs fréquentes** : le post-LN ; un `eps` passé à une seule des deux layer norms ; la GELU tanh ; normaliser $\mathbf{x}$ au lieu de $\mathbf{h}$ avant le MLP ; oublier de transmettre `mask` à l'attention ; une layer norm recodée avec la variance non biaisée (`ddof=1`) ; modifier `x` en place (`x += …`).
**Variante** : avec tous les poids à zéro sauf $\mathbf{b}_2 = \mathbf{1}$, le bloc rend $\mathbf{x} + 1$ : la branche du MLP ajoute exactement son biais de sortie au chemin résiduel. Empiler quatre de ces blocs ajoute 4.

### Ex B3.17 — Trois masques qui mentent : convention inversée, masque transposé, padding compté dans la loss 🐛
**Réponse** : les trois erreurs, et le correctif de chacune :
1. `nn.MultiheadAttention` lit un masque booléen à l'envers de SDPA (`True` = masqué) : il faut lui passer `~causal_mask_17(T)` ;
2. `triu()` garde le triangle **supérieur** : chaque requête voit le futur ; le masque causal est `tril()` ;
3. la cross-entropy compte les positions de padding : `F.cross_entropy(..., ignore_index=PAD_17)`.

Les symptômes de la cellule, version par version (rejoués sur les huit versions par `history_chB3.py`) : avec les trois erreurs, la loss vaut `nan`, avec des logits `nan` à la position 0 de chaque séquence ; avec la seule erreur 1, à la dernière position (62) ; avec l'erreur 2 (après correction de la 1), changer le caractère 40 change les logits des positions 0 à 39, et la loss de validation tombe à 0,86 nat par caractère après 200 itérations (1,42 si l'erreur 3 est encore là), contre 2,76 pour la version réparée ; avec la seule erreur 3, la loss d'un même batch de 16 lignes vaut 2,30 sur 39 positions et 1,50 avec 21 positions de padding en plus.
**Démarche** : le `nan` vient d'une ligne de masque **entièrement** masquée. Passé tel quel à `nn.MultiheadAttention`, le triangle supérieur (`triu`) masque, pour la requête 0, toutes les clés $j \ge 0$ : sa ligne est vide. Une fois l'erreur 2 corrigée, le triangle inférieur (`tril`) passé tel quel masque la diagonale et le passé : c'est la dernière requête qui n'a plus aucune clé. Le `nan` se limite à une position, car l'attention mélange des **valeurs**, toutes finies ; seule la ligne vide a $0/0$ dans sa softmax. Il cache les deux autres erreurs, puisque la loss entière vaut `nan`. Ensuite, un masque transposé laisse chaque position regarder le caractère qu'elle doit prédire : le modèle n'a qu'à le recopier, et sa loss descend très bas, en entraînement comme en validation, sans rien apprendre de la langue ; en génération, ce futur n'existe pas, et le modèle produit n'importe quoi. Enfin, sans `ignore_index`, chaque position de padding demande de prédire `PAD_17` après `PAD_17`, ce que le modèle apprend vite : la loss d'entraînement baisse avec la proportion de padding, sans que le modèle prédise mieux le texte. La loss d'un même batch ne doit pas dépendre du padding ajouté : c'est ce que mesure le point 3.
**Erreurs fréquentes** : « réparer » le `nan` en remplaçant `-inf` par `-1e9` ou par `torch.nan_to_num` (le symptôme disparaît, le masque reste inversé) ; inverser le masque **et** le transposer (`~triu`), ce qui donne un modèle qui voit le futur sans `nan` ; corriger la loss en divisant par le nombre total de positions ; croire qu'une loss très basse après 200 itérations est une bonne nouvelle.
**Variante** : la ligne « I had seen little of Holmes lately. » (35 caractères) dans un batch de largeur 63 : 34 positions d'entrée et 34 cibles réelles, puis 29 positions de padding ; sans `ignore_index`, ces 29 positions pèsent 46 % de sa part de la loss.

### Ex B3.18 — De NumPy à PyTorch : un module CausalSelfAttention avec F.scaled_dot_product_attention 📦
**Réponse** : ton module rend la sortie de `nn.MultiheadAttention` aux mêmes poids (écart maximal @@GAPT@@) et celle de ta `multi_head_attention` NumPy de B3.14 (écart maximal @@GAPN@@, sous $10^{-5}$) ; il est causal, et son dropout n'agit qu'en mode train.
**Démarche** :
```python
class CausalSelfAttention_18(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        if d_model % n_head:
            raise ValueError(f"n_head = {n_head} must divide d_model = {d_model}")
        self.n_head, self.dropout = n_head, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)       # q, k and v in one product
        self.proj = nn.Linear(d_model, d_model)          # W_O
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        N, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(N, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))   # split_heads
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(N, T, D)))                       # merge_heads
```
Le module refait en PyTorch ce que ta librairie fait en NumPy : `qkv` regroupe les trois projections en un seul produit (`in_proj_weight` de `nn.MultiheadAttention` a la même disposition, d'où la comparaison directe), `view` puis `transpose` est ton `split_heads`, `transpose` puis `reshape` ton `merge_heads`. `reshape` et non `view` : après `transpose`, les éléments d'une ligne ne sont plus contigus en mémoire, et `view` refuse de les réinterpréter (le `RuntimeError` du §203.4) ; `reshape` copie alors. L'écart entre tes deux versions vient du float32 du module : un arrondi relatif de l'ordre de $6 \times 10^{-8}$ par opération, que les sommes de 32 termes accumulent ; la version NumPy calcule en float64 sur les mêmes poids, convertis par `W = weight.T`. `dropout_p` s'applique même en `eval()` (§203.3) : sans le `if self.training`, le modèle évalué serait bruité. Passer un masque et `is_causal=True` ensemble est refusé par le noyau de référence et combiné sans prévenir par le noyau du CPU (§203.3) : on choisit l'un ou l'autre.
**Erreurs fréquentes** : découper les têtes par `view(N, H, T, d_h)` directement ; oublier `is_causal=True` (la sortie de la ligne 0 change avec le futur) ; déballer `k, q, v` dans le mauvais ordre ; `view` après `transpose` ; `dropout_p=self.dropout` sans tenir compte du mode ; des couches sans biais (le contrat en demande).
**Variante** : le même module avec `enable_gqa=True` et 2 têtes de clés et de valeurs pour 4 têtes de requêtes : `qkv` devient `nn.Linear(d, d + 2 · 2 · d_h)`, soit 64 sorties au lieu de 96 pour $d = 32$ ($d_h = 8$), et SDPA répète chaque tête de clés pour deux têtes de requêtes (`repeat_interleave`, §203.4).

### Ex B3.19 — Mini-GPT caractère par caractère : embeddings de tokens et de positions, blocs pré-LN, tête liée 📦
**Réponse** : ton modèle suit le contrat : son `state_dict` a les noms et les formes attendus, sa tête est liée, il a le nombre de paramètres de ✏️ B3.5 j), ses blocs sont pré-LN, son initialisation est celle de nanoGPT, ses positions servent, et il est causal. À l'initialisation, la loss d'un batch est celle d'une prédiction presque uniforme (le contrôle de départ de la fiche).
**Démarche** :
```python
class Block_19(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention_18(d_model, n_head, dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT_19(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, d_model=128, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block_19(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                      # tied
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(block.mlp[2].weight, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, idx):
        T = idx.shape[1]
        if T > self.block_size:
            raise ValueError(f"{T} tokens, more than block_size = {self.block_size}")
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
```
L'affectation `self.head.weight = self.tok_emb.weight` fait de la sortie et de la table d'entrée **un seul** `nn.Parameter` : `model.parameters()` ne le liste qu'une fois, l'optimiseur ne le met à jour qu'une fois, et ses gradients s'additionnent depuis les deux usages. L'initialisation compte doublement ici : avec les tables de PyTorch par défaut, $\mathcal{N}(0;\ 1)$, la tête liée calcule des logits $\mathbf{h}_t \mathbf{E}^\top$ dont l'écart-type est de l'ordre de $\sqrt{d} \approx 11$ (15 mesuré sur un batch) ; la softmax est saturée dès le départ, et la loss initiale atteint 88 nats au lieu d'environ $\log V$. Les deux projections résiduelles de chaque bloc **ajoutent** au chemin : avec $2 n_{\text{layer}}$ contributions indépendantes, la variance du chemin croît comme $2 n_{\text{layer}}$ fois celle d'une contribution, d'où l'écart-type divisé par $\sqrt{2 n_{\text{layer}}}$ (GPT-2, §203.8). Les positions apprises sont la seule source d'ordre : sans elles, un caractère répété donne les mêmes logits partout, ce que vérifie la cellule.
**Erreurs fréquentes** : copier les valeurs de la table dans la tête (`nn.Parameter(tok_emb.weight.clone())`), qui donne deux tenseurs et 11 264 paramètres de plus ; garder l'initialisation par défaut ; un bloc post-LN ; oublier d'ajouter les positions ; une layer norm finale absente ; `pos_emb(torch.arange(T))` sans `device=` (une erreur dès que le modèle est sur GPU).
**Variante** : le même modèle avec des positions sinusoïdales fixes (ta `positional_encoding` de B3.15, enregistrée par `register_buffer`) au lieu de `pos_emb` : $128 \times 128 = 16\,384$ paramètres de moins, et un `state_dict` qui ne contient plus `pos_emb.weight` (le tampon n'est pas un paramètre, mais `register_buffer` l'y ajoute sous son nom, sauf avec `persistent=False`).

### Ex B3.20 — Entraîner le mini-GPT sur Holmes : AdamW, warmup + cosinus, clipping, perplexité, face au LSTM du ch. 22 📦
**Réponse** : tes trois fonctions passent les contrôles (les deux groupes d'AdamW, paramètre par paramètre ; l'estimation en mode eval sans gradient ; 6 itérations aux poids identiques à ceux de la recette, clipping compris). En mode rapide (600 itérations, ≈ 2 min 15 sur deux cœurs), ton mini-GPT atteint **1,590** nat par caractère sur toute la validation (perplexité 4,90 ; 2,29 bits par caractère), sous le seuil de @@TFAST@@ ; sur trois graines, de 1,590 à 1,606. Le LSTM du ch. 22, sur les mêmes caractères de validation : 1,750 dans son mode rapide (2 epochs, ≈ 1,4 min d'entraînement), @@LSTMFULL@@ dans son mode complet (@@LSTMFULLB@@). Le mini-GPT en mode complet (5 000 itérations, @@GPTFULLT@@) descend à @@GPTFULL@@.
**Démarche** :
```python
def configure_optimizer_20(model, lr, weight_decay, betas=(0.9, 0.99)):
    params = list(model.parameters())                                  # the tied table once
    return torch.optim.AdamW([{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
                              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}], lr=lr, betas=betas)


@torch.no_grad()
def estimate_loss_20(model, data, n_batches, batch_size, block_size, generator):
    model.eval()
    losses = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
              for x, y in (get_batch(data, batch_size, block_size, generator) for _ in range(n_batches))]
    model.train()
    return sum(losses) / n_batches


def train_20(model, train_ids, val_ids, settings, generator, block_size=128):
    s = settings
    optimizer = configure_optimizer_20(model, s["lr"], s["weight_decay"], (0.9, s["beta2"]))
    history = {"train_loss": [], "lr": [], "val_iter": [], "val_loss": []}

    def evaluate(done):
        history["val_iter"].append(done)
        history["val_loss"].append(estimate_loss_20(model, val_ids, s["eval_batches"], s["batch_size"], block_size,
                                                    torch.Generator().manual_seed(s["eval_seed"])))

    model.train()
    evaluate(0)
    for it in range(s["iters"]):
        lr = mylearn.optim.cosine_lr(it, s["iters"], s["lr"], s["warmup"], s["min_lr"])   # or the formula
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_ids, s["batch_size"], block_size, generator)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), s["clip"])
        optimizer.step()
        history["train_loss"].append(loss.item())
        history["lr"].append(lr)
        if (it + 1) % s["eval_interval"] == 0 or it + 1 == s["iters"]:
            evaluate(it + 1)
    return history
```
Les matrices et les tables portent l'essentiel de la capacité du modèle : le weight decay les tire vers 0 et limite l'overfitting ; les biais et les $\gamma$, $\beta$ des layer norms sont des décalages et des échelles, qu'il n'y a aucune raison de pousser vers 0 (pour $\gamma$, ce serait éteindre la couche). Le warmup protège les premières itérations, où les moments d'Adam sont mal estimés et les pas trop grands ; le cosinus réduit ensuite le learning rate jusqu'à un dixième, pour finir dans un creux plutôt que de rebondir autour. Le clipping coupe les rares gradients énormes du début. Chaque évaluation reprend **les mêmes** batches de validation (un générateur neuf, `eval_seed`) : la courbe de validation ne bouge que parce que le modèle change, pas parce que les batches changent, et le générateur de l'entraînement suit la même suite de batches qu'il y ait des évaluations ou non. La loss estimée (8 batches) et `char_loss` diffèrent un peu : l'estimation tire des fenêtres au hasard et compte toutes leurs positions, dont les premières n'ont presque aucun contexte (64 caractères en moyenne), alors que `char_loss` prédit chaque caractère de la validation avec au moins 97 caractères de contexte. Le mini-GPT bat le LSTM à temps voisin : il lit 128 caractères au lieu de 40, il a 3,2 fois plus de paramètres (820 992 contre 255 064), et surtout il apprend de **toutes** les positions de chaque fenêtre (600 × 32 × 128 ≈ 2,5 millions de prédictions), alors que le LSTM de 22.25, *many-to-one*, n'apprend que du dernier caractère de chacune de ses fenêtres (2 × 80 000 = 160 000 prédictions en mode rapide). La comparaison reste inégale dans les deux sens : à paramètres égaux, ou à nombre de prédictions égal, les écarts changeraient.
**Erreurs fréquentes** : un seul groupe de paramètres, qui applique le weight decay aux biais et aux layer norms ; parcourir `named_parameters(remove_duplicate=False)`, qui donne deux fois la table liée ; oublier `model.eval()` ou `torch.no_grad()` dans l'estimation (un dropout actif, de la mémoire gaspillée), ou oublier de revenir en mode train ; le clipping après `optimizer.step()`, qui n'agit plus sur rien ; tirer les batches de validation avec le générateur de l'entraînement, ce qui change la suite des batches d'entraînement à chaque évaluation ; comparer des loss par token d'un modèle de caractères et d'un modèle de tokens BPE (🔬 B3.26).
**Variante** : en bits, 1,590 nat par caractère font $1{,}590 / \ln 2 \approx 2{,}29$ bits ; un codeur arithmétique guidé par ce modèle écrirait la validation de Holmes en 2,29 bits par caractère environ, contre 8 en ASCII : une compression d'un facteur 3,5 (mesuré sur la même validation : `zlib` au niveau 9 donne 3,27 bits par caractère, `lzma` 3,06 et `bz2` 2,77 ; des compresseurs généralistes, qui n'apprennent que sur le texte qu'ils compressent).

Les solutions des exercices B3.21 à B3.30 seront ajoutées à la prochaine étape de génération, avec la suite de la partie B du notebook.
