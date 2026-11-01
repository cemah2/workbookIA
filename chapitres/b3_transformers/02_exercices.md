# B3 · Attention et Transformers : un mini-GPT from scratch — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/b3_transformers/06_mes_reponses.md` (créée par `python tools/start_chapter.py B3`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ B3.1 à B3.5 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les dérivations ∂ B3.6 et B3.7, l'estimation 🧮 B3.8, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), une forme de tableau entre guillemets (`"(2, 3)"`), un vecteur ou une ligne de matrice en liste (`[0.5, -1, 3]`).
> Conventions du chapitre (fiche, encadré 📐) : un batch contient $N$ séquences de $T$ tokens ; dans une attention, $L$ est le nombre de requêtes et $S$ le nombre de clés ; $d$ est la dimension du modèle, $H$ le nombre de têtes, $d_h = d/H$, $d_k$ et $d_v$ les tailles des clés et des valeurs, $F$ la largeur du MLP, $V$ la taille du vocabulaire. Les couches denses suivent la convention de `mylearn` : $\mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b}$, une ligne par token. Un masque booléen vaut `True` là où la requête **a le droit** de regarder la clé (convention de `F.scaled_dot_product_attention` et de ta librairie). Les tokens, les positions et les têtes sont numérotés à partir de 0 ; $\log$ est le logarithme naturel.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### B3.Q1 — Le goulot de l'encodeur-décodeur récurrent : pourquoi un seul vecteur ne suffit pas 🧠 ⏱️ 3 min
*Fiche §203.1 · parcours R*

a) Dans l'encodeur-décodeur récurrent de 2014 (sans attention), que reçoit le décodeur de la phrase source ? (A) la moyenne de tous les états de l'encodeur, recalculée à chaque pas ; (B) les embeddings des mots source, sans passer par l'encodeur ; (C) un seul vecteur, le dernier état de l'encodeur.
b) Vrai ou faux : la taille du vecteur de contexte $\mathbf{c}$ grandit avec la longueur de la phrase source.
c) Avec l'attention de Bahdanau, on traduit une phrase source de 12 mots par une phrase de 9 mots. Combien de poids $\alpha_{ij}$ le décodeur calcule-t-il en tout ?
d) À un pas $i$ fixé, les poids $\alpha_{ij}$ : (A) font 1 quand on les additionne sur les mots source $j$ ; (B) font 1 quand on les additionne sur les pas $i$ de la traduction ; (C) valent 0 ou 1, le décodeur choisissant un seul mot source.
e) Vrai ou faux : Bahdanau et ses coauteurs apprennent l'alignement sans aucun alignement annoté, par rétropropagation.

### B3.Q2 — Requête, clé, valeur : un dictionnaire où l'on cherche « à peu près » 🧠 ⏱️ 3 min
*Fiche §203.2 · parcours R*

a) Un dictionnaire Python rend la valeur de la clé égale à la requête. L'attention rend : (A) la valeur de la clé la plus proche de la requête, et elle seule, comme un dictionnaire qui tolérerait les fautes de frappe ; (B) une moyenne des valeurs, pondérée par la ressemblance de la requête à chaque clé ; (C) la clé la plus proche de la requête.
b) Vrai ou faux : dans une auto-attention, la requête, la clé et la valeur d'un même token sont égales, puisqu'elles viennent toutes du même vecteur $\mathbf{x}_i$.
c) Avec $\mathbf{q} = (1, 3)$, $\mathbf{k} = (2, 2)$ et $d_k = 2$, que vaut le score mis à l'échelle $\mathbf{q} \cdot \mathbf{k} / \sqrt{d_k}$, à 2 décimales ?
d) Une requête donne les scores $(0 ;\ 0 ;\ \ln 3)$ à trois clés. Que vaut le poids de la troisième clé, à 1 décimale ?
e) Pour **une** requête, la sortie de l'attention a la forme : (A) $(S,)$, un poids par clé ; (B) $(S, d_v)$, une valeur par clé, multipliée par son poids ; (C) $(d_v,)$, une seule moyenne de valeurs.

### B3.Q3 — Pourquoi diviser les scores par √d_k ? 🧠 ★★ ⏱️ 3 min
*Fiche §203.2 · parcours R, M*

a) Les composantes de $\mathbf{q}$ et de $\mathbf{k}$ sont indépendantes, centrées, de variance 1, et $d_k = 256$. Quel est l'écart-type du produit scalaire $\mathbf{q} \cdot \mathbf{k}$ ?
b) Vrai ou faux : avec les mêmes hypothèses, après la division par $\sqrt{d_k}$, la variance des scores vaut 1 quelle que soit $d_k$.
c) Une softmax **saturée** : (A) donne presque tout le poids au plus grand score, et ses dérivées sont presque nulles ; (B) donne des poids presque égaux à toutes les clés, quelle que soit la requête, et ses dérivées explosent ; (C) donne des poids négatifs aux plus petits scores.
d) Si l'on divisait les scores par $d_k$ au lieu de $\sqrt{d_k}$, avec $d_k$ grand : (A) la softmax resterait la même, à un facteur près ; (B) la softmax saturerait encore plus vite ; (C) les poids seraient presque uniformes.
e) Vrai ou faux : la division par $\sqrt{d_k}$ compense la taille des valeurs $\mathbf{v}_j$, qui grandit avec $d_v$.

### B3.Q4 — Masque causal : qui a le droit de regarder qui ? 🧠 ⏱️ 3 min
*Fiche §203.3 · parcours R*

a) Combien de cases valent `True` dans le masque causal booléen d'une séquence de 5 tokens ?
b) La ligne 2 de ce masque autorise : (A) la seule clé 2 ; (B) les clés 0, 1 et 2 ; (C) les clés 2, 3 et 4, celles qui la suivent.
c) Vrai ou faux : avec un masque causal, modifier le token 4 change la sortie de la position 1.
d) Pendant une génération avec KV cache, une seule nouvelle requête fait face à $S = 9$ clés. Combien de clés voit-elle avec `causal_mask(1, 9)` de ta librairie ?
e) Le masque causal sert à : (A) empêcher chaque position de regarder les tokens qui la suivent ; (B) ignorer le padding en fin de séquence ; (C) sauter la moitié des scores pour aller deux fois plus vite.

### B3.Q5 — Auto-attention ou attention croisée : d'où viennent Q, K et V ? 🧠 ⏱️ 3 min
*Fiche §203.3, §203.7 · parcours R*

a) Dans le décodeur d'un traducteur, l'attention croisée prend ses **requêtes** : (A) dans la traduction en cours ; (B) dans la phrase source, à la sortie de l'encodeur ; (C) dans les deux, mises bout à bout.
b) Les requêtes viennent de 7 tokens de la traduction, les clés de 12 tokens de la source. Quelle est la forme de la carte d'attention d'une tête, pour une séquence ?
c) Vrai ou faux : dans l'attention croisée d'un traducteur, il faut un masque causal entre la traduction et la source.
d) `multi_head_attention(y, x, x, params, n_heads)` calcule : (A) une auto-attention de `x` ; (B) une attention croisée où `x` regarde `y` ; (C) une attention croisée où `y` regarde `x`.
e) Vrai ou faux : en auto-attention, $L = S$.

### B3.Q6 — Plusieurs têtes pour le prix d'une : que gagne-t-on ? 🧠 ⏱️ 3 min
*Fiche §203.4 · parcours R*

a) Avec $d = 512$ et $H = 16$ têtes, que vaut $d_h$ ?
b) Vrai ou faux : à $d$ fixé, passer de 8 à 16 têtes double le nombre de paramètres de l'attention.
c) Un modèle a 32 têtes de requêtes et 4 têtes de clés et de valeurs (GQA). Par combien son KV cache est-il divisé, face au même modèle avec 32 têtes de clés et de valeurs ?
d) Écrire directement `x.reshape(N, H, T, d_h)` au lieu de `split_heads` : (A) donne la bonne forme, mais une tête reçoit des morceaux de tokens différents ; (B) lève une erreur de forme, puisque les axes ne sont pas dans le bon ordre ; (C) donne exactement le même tenseur que `split_heads`, en une seule opération au lieu de deux.
e) Vrai ou faux : en GQA avec 8 têtes de requêtes et 2 de clés, si l'on répète les têtes de clés avec `repeat` (et non `repeat_interleave`), la tête de requêtes 1 utilise la tête de clés 1.

### B3.Q7 — Sans encodage de position, l'attention voit-elle l'ordre des mots ? 🧠 ★★ ⏱️ 3 min
*Fiche §203.5 · parcours R*

a) Vrai ou faux : sans encodage de position ni masque, permuter les tokens d'une séquence permute les sorties de l'auto-attention de la même façon, sans les changer.
b) Sans position ni masque, la sortie du token « chien » dans « le chien mord l'homme » et dans « l'homme mord le chien » (un token par mot) : (A) est la même dans les deux phrases ; (B) diffère, car « chien » n'a pas les mêmes voisins ; (C) est nulle.
c) Un modèle à positions apprises (une table de 1 024 lignes) reçoit un token en position 1 500 : (A) il extrapole sans difficulté, comme pour les positions déjà vues à l'entraînement ; (B) aucune ligne de la table ne correspond à cette position ; (C) la position est ramenée à 1 023 automatiquement.
d) RoPE tourne : (A) les embeddings des tokens, à l'entrée du modèle ; (B) les valeurs, dans chaque couche ; (C) les requêtes et les clés, dans chaque couche.
e) RoPE, base 10 000, $d_h = 64$ : quel est l'angle de rotation de la paire $i = 16$ à la position 250, en radians, à 1 décimale ?

### B3.Q8 — Résiduel, pré-LN ou post-LN : ce qui rend un Transformer profond entraînable 🧠 ★★ ⏱️ 3 min
*Fiche §203.6*

a) Un bloc **pré-LN** écrit sa sous-couche d'attention : (A) $\mathbf{h} = \operatorname{LN}\big(\mathbf{x} + \operatorname{MHA}(\mathbf{x})\big)$ ; (B) $\mathbf{h} = \operatorname{MHA}\big(\operatorname{LN}(\mathbf{x})\big)$ ; (C) $\mathbf{h} = \mathbf{x} + \operatorname{MHA}\big(\operatorname{LN}(\mathbf{x})\big)$.
b) Vrai ou faux : dans un Transformer pré-LN, le chemin résiduel de l'entrée du premier bloc à la sortie du dernier ne traverse aucune layer norm.
c) D'après Xiong et coll. (2020), à l'initialisation d'un Transformer post-LN : (A) les gradients des paramètres proches de la sortie sont grands, et il faut un warmup long ; (B) les gradients des paramètres proches de l'entrée explosent dès les premiers pas, et un gradient clipping à 1 suffit à stabiliser l'entraînement ; (C) les gradients sont bien répartis, et aucun warmup n'est nécessaire.
d) Combien de paramètres a le MLP d'un bloc avec $d = 512$, $F = 2\,048$ et des biais ?
e) Vrai ou faux : sans connexions résiduelles, un Transformer de 12 blocs s'entraîne aussi bien, grâce à ses layer norms.

### B3.Q9 — BERT, GPT, T5 : quel masque, quel objectif, quel usage ? 🧠 ⏱️ 3 min
*Fiche §203.7 · parcours R*

a) L'objectif de pré-entraînement de BERT : (A) retrouver des tokens masqués, avec le contexte des deux côtés ; (B) prédire le token suivant ; (C) réécrire des segments corrompus du texte d'entrée, remplacés par des marqueurs.
b) BERT choisit 15 % des 200 tokens d'un texte. Combien de positions comptent dans la loss ?
c) Parmi ces positions choisies, combien sont remplacées par `[MASK]`, en moyenne ?
d) Vrai ou faux : un décodeur comme GPT peut aussi classer un texte, si on le lui demande dans le prompt.
e) Pour classer des tickets de support en dix catégories avec un petit modèle fine-tuné, sans générer de texte, la famille la plus naturelle est : (A) un décodeur ; (B) un encodeur ; (C) un encodeur-décodeur.

### B3.Q10 — Décalage d'un token : ce que chaque position apprend à prédire 🧠 ⏱️ 3 min
*Fiche §203.7, §203.8 · parcours R*

a) Le texte est « Watson », découpé en caractères. Pour la fenêtre d'entrée $x$ = « Watso » ($T = 5$), les cibles $y$ sont : (A) « Watso » ; (B) « atson » ; (C) « Wats ».
b) Une fenêtre de $T = 256$ tokens, avec ses cibles décalées : combien de prédictions entrent dans la loss de cette fenêtre ?
c) Vrai ou faux : le teacher forcing donne au modèle, à l'entraînement, ses propres prédictions comme entrées.
d) Un bug met $y = x$ (des cibles non décalées) dans un modèle à masque causal. La loss d'entraînement : (A) reste près de $\log V$ ; (B) diverge ; (C) tombe près de 0.
e) Quelle loss attends-tu au tout premier pas pour un modèle de caractères à $V = 65$ symboles, à 2 décimales ?

### B3.Q11 — KV cache : ce que l'on garde et pourquoi la génération s'accélère 🧠 ⏱️ 3 min
*Fiche §203.9 · parcours R*

a) Le KV cache garde, pour chaque couche : (A) les requêtes et les clés des tokens déjà vus ; (B) les clés et les valeurs des tokens déjà vus ; (C) les logits des tokens déjà générés.
b) Avec un KV cache, combien de nouvelles requêtes calcule-t-on, par tête et par couche, pour chaque token généré ?
c) Vrai ou faux : à graine égale, générer avec un KV cache donne d'autres tokens que sans cache.
d) La mémoire du cache d'une séquence est : (A) proportionnelle au nombre de tokens déjà vus ; (B) constante ; (C) proportionnelle au carré du nombre de tokens, comme la matrice des poids d'attention.
e) Combien d'octets occupe le cache d'un modèle de 2 couches, $d = 64$, après 100 tokens, en float32 ?

### B3.Q12 — Coût quadratique du contexte et ce que change FlashAttention 🧠 ★★ ⏱️ 3 min
*Fiche §203.10*

a) Multiplier par 8 la longueur du contexte multiplie la mémoire de la matrice des poids d'attention (implémentation directe) par combien ?
b) Vrai ou faux : FlashAttention calcule une approximation de l'attention, au prix d'une petite erreur.
c) FlashAttention réduit : (A) le nombre de paramètres ; (B) le nombre de FLOPs, qui devient linéaire en $T$ ; (C) la mémoire et les lectures-écritures, pas les FLOPs.
d) Dans un décodeur causal de dimension $d = 768$, à partir de combien de tokens de contexte l'attention d'une couche coûte-t-elle autant que ses couches denses ?
e) Vrai ou faux : avec $d = 4\,096$ et un contexte de 100 000 tokens, l'attention fait la plus grande part du calcul d'une couche.

<a id="rappels"></a>

## 🔁 Rappels

### B3.R1 — Ch. B2 : de quoi dépend le nombre de tokens d'un texte ? 🔁 ★ ⏱️ 5 min
*Ch. B2 (tokenizers, compression, fertilité) · fiche §203.8 · parcours R*

a) Vrai ou faux : un même texte compte le même nombre de tokens, quel que soit le tokenizer.
b) Un texte de 4 000 octets, avec un tokenizer qui compresse à 3,2 octets par token : combien de tokens ?
c) Pour un même texte, augmenter la taille du vocabulaire d'un BPE (plus de fusions) : (A) augmente le nombre de tokens ; (B) ne le change pas ; (C) le diminue.
d) Vrai ou faux : avec le tokenizer de GPT-2, un texte français donne en général plus de tokens que sa traduction anglaise.
e) Un mini-GPT a un contexte de 128 positions. Quelle version couvre le plus de texte ? (A) 128 caractères ; (B) 128 tokens BPE ; (C) les deux couvrent autant.

### B3.R2 — Ch. 22 : pourquoi le gradient s'évanouit dans un RNN déroulé 🔁 ★ ⏱️ 5 min
*Ch. 22 (rétropropagation dans le temps, gradient évanescent, LSTM) · fiche §203.1 · parcours R*

a) Si chaque jacobienne d'un pas a une norme d'environ 0,9, par combien environ le gradient qui remonte 50 pas d'un RNN est-il multiplié, à 4 décimales ?
b) Vrai ou faux : le LSTM supprime complètement le gradient évanescent.
c) Dans une couche d'auto-attention, deux tokens éloignés de 50 positions sont reliés en : (A) 50 pas ; (B) 1 pas ; (C) environ $\log_2 50$ pas.
d) Contre l'explosion du gradient d'un RNN, le ch. 22 utilise : (A) le gradient clipping ; (B) le dropout ; (C) la température.
e) Vrai ou faux : à l'entraînement, un RNN calcule les états de toutes les positions d'une séquence en parallèle, comme un Transformer.

### B3.R3 — Ch. 20 : layer norm et connexion résiduelle, à quoi sert chacune ? 🔁 ★ ⏱️ 5 min
*Ch. 20 (`layer_norm` de ta librairie, connexions résiduelles) · fiche §203.6 · parcours R*

a) `layer_norm` de $\mathbf{x} = (1, 2, 3, 6)$ avec $\gamma = 1$, $\beta = 0$ et $\epsilon$ négligé : que vaut la première composante, à 3 décimales ?
b) Vrai ou faux : la layer norm calcule sa moyenne et sa variance sur le batch, comme la batchnorm.
c) Pour $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$, la jacobienne $\partial \mathbf{y} / \partial \mathbf{x}$ vaut : (A) $\partial F / \partial \mathbf{x}$ seul, puisque $\mathbf{x}$ ne dépend d'aucun poids ; (B) $\mathbf{I} + \partial F / \partial \mathbf{x}$ ; (C) $\mathbf{I}$.
d) Vrai ou faux : une layer norm donne le même résultat pour un token, quels que soient la taille du batch et le reste de la séquence.
e) Combien de paramètres a une layer norm de dimension $d = 1\,600$ (celle de GPT-2 XL) ?

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex B3.1 — L'attention à la main : trois tokens, d_k = 2, scores, softmax et sortie ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer à la main une auto-attention complète sur trois tokens, des projections à la sortie.
**Prérequis :** fiche §203.2 · ch. 17 (softmax stable) · 0B (produit matriciel) · **Parcours :** R, M

Trois tokens ont pour vecteurs d'entrée $\mathbf{x}_0 = (1, 0)$, $\mathbf{x}_1 = (0, 1)$ et $\mathbf{x}_2 = (1, 1)$ : ce sont les lignes 0, 1 et 2 de $\mathbf{X}$, de forme $(3, 2)$. Les projections, sans biais, avec la convention $\mathbf{q}_i = \mathbf{x}_i \mathbf{W}_Q$ :

$$\mathbf{W}_Q = \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}, \qquad \mathbf{W}_K = \begin{pmatrix} 1 & 0 \\ 1 & 2 \end{pmatrix}, \qquad \mathbf{W}_V = \begin{pmatrix} 1 & -1 \\ 0 & 2 \end{pmatrix}.$$

Ainsi $d_k = d_v = 2$. Il n'y a pas de masque : chaque requête regarde les trois clés. On note $s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k}$ le score de la requête $i$ pour la clé $j$, $\alpha_{ij}$ son poids après la softmax sur les clés, et $\mathbf{o}_i = \sum_j \alpha_{ij} \mathbf{v}_j$ la sortie du token $i$. Arrondis à 3 décimales.

a) La requête $\mathbf{q}_2$, en liste.
b) Le score $s_{2,1}$.
c) Le poids $\alpha_{2,1}$.
d) La sortie $\mathbf{o}_2$, en liste.
e) Le poids $\alpha_{2,2}$ que l'on obtiendrait **sans** la division par $\sqrt{d_k}$.
f) Vrai ou faux : $\alpha_{0,1} = \alpha_{0,2}$.
g) La somme $\alpha_{0,0} + \alpha_{1,0} + \alpha_{2,0}$ de la colonne 0 de la carte d'attention.

Dans ta copie : écris $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$, la matrice des scores $3 \times 3$ et la carte d'attention entière, et vérifie que chaque ligne fait 1. Que devient la ligne 2 de la carte si l'on multiplie $\mathbf{W}_Q$ par 10 ?

### Ex B3.2 — Le même calcul avec un masque causal ✏️ ★ ⏱️ 10 min
**Objectif :** appliquer un masque causal à la main, et voir la différence entre l'alignement en bas à droite (KV cache) et celui de `is_causal=True`.
**Prérequis :** Ex B3.1 · fiche §203.3 · **Parcours :** R, M

Reprends $\mathbf{Q}$, $\mathbf{K}$ et $\mathbf{V}$ de ✏️ B3.1, avec cette fois le masque causal `causal_mask(3)`. Arrondis à 3 décimales.

a) Le nombre de cases `True` de `causal_mask(3)`.
b) La ligne 1 de la carte d'attention masquée (les trois poids $\alpha_{1,0}$, $\alpha_{1,1}$, $\alpha_{1,2}$), en liste.
c) La sortie masquée $\mathbf{o}_1$, en liste.
d) La sortie masquée $\mathbf{o}_0$, en liste.
e) Vrai ou faux : la sortie $\mathbf{o}_2$ est la même avec et sans le masque.
f) Avec un KV cache, les tokens 1 et 2 arrivent comme $L = 2$ nouvelles requêtes face aux $S = 3$ clés des tokens 0, 1 et 2. Avec `causal_mask(2, 3)` de ta librairie, combien de clés la requête 0 de ce calcul (le token 1) a-t-elle le droit de voir ?
g) Et combien en voit-elle avec `F.scaled_dot_product_attention(..., is_causal=True)`, qui aligne le triangle en haut à gauche quand $L \ne S$ ?
h) Vrai ou faux : avec le masque transposé (`np.triu` au lieu de `np.tril`, le bug du 🐛 B3.17), la sortie du token 2 dépend des trois tokens.

Dans ta copie : écris `causal_mask(2, 3)`. Vérifie que les deux sorties du calcul de f) sont exactement $\mathbf{o}_1$ et $\mathbf{o}_2$ de ce calcul masqué. Pourquoi le résultat de g) est-il un bug pour un KV cache ?

### Ex B3.3 — Suivre les formes dans l'attention multi-têtes et compter ses paramètres ✏️ ★★ ⏱️ 15 min
**Objectif :** suivre les formes d'une attention multi-têtes, de l'entrée aux poids de chaque tête, et compter ses paramètres, avec et sans GQA.
**Prérequis :** Ex B3.1 · fiche §203.4 · ch. 16 (paramètres d'une couche dense) · **Parcours :** R, M

Une attention multi-têtes de dimension $d = 384$ à $H = 6$ têtes, avec des biais partout (quatre matrices $(d, d)$ et quatre biais $(d,)$, comme `multi_head_attention` de ta librairie), reçoit un batch de $N = 4$ séquences de $T = 10$ tokens : $\mathbf{X}$ a la forme $(4, 10, 384)$. Donne les formes entre guillemets.

a) La forme de $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q + \mathbf{b}_Q$.
b) La forme de $\mathbf{Q}$ après `split_heads`.
c) La forme des poids d'attention rendus par `multi_head_attention` (le second résultat).
d) Le nombre total de scores $\mathbf{q} \cdot \mathbf{k}$ calculés pour ce batch, toutes têtes comprises.
e) Le nombre de paramètres de cette attention multi-têtes.
f) Vrai ou faux : avec $H = 12$ têtes au lieu de 6, le nombre de paramètres est le même.
g) La forme de `in_proj_weight` dans `nn.MultiheadAttention(384, 6)` de PyTorch (les trois matrices d'entrée empilées, dans la convention de PyTorch).
h) Une variante en GQA garde 6 têtes de requêtes de taille 64, mais seulement 2 têtes de clés et 2 têtes de valeurs, de taille 64 aussi : $\mathbf{W}_K$ et $\mathbf{W}_V$ ont la forme $(384, 128)$, avec leurs biais. Combien de paramètres a cette attention, $\mathbf{W}_Q$ et $\mathbf{W}_O$ compris ?
i) Une attention croisée : les requêtes viennent de ces 10 tokens, les clés et les valeurs des 25 tokens de sortie d'un encodeur. Quelle est la forme des poids d'attention ?

Dans ta copie : que se passe-t-il avec $H = 5$ ? Quelles sont les formes de $\mathbf{K}$ après le découpage dans la variante GQA de h), et par combien son KV cache est-il plus petit ?

### Ex B3.4 — Encodage sinusoïdal : quelques valeurs et l'effet d'un décalage de position ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer des valeurs de l'encodage sinusoïdal et vérifier qu'un décalage de position agit comme une rotation de chaque paire de colonnes.
**Prérequis :** fiche §203.5 (encodage sinusoïdal, matrice du décalage) · 0B (sinus, cosinus, radians) · **Parcours :** M

On prend $d = 8$ et la base 10 000 : $PE_{p, 2i} = \sin(p\,\omega_i)$ et $PE_{p, 2i+1} = \cos(p\,\omega_i)$, avec $\omega_i = 10000^{-2i/d}$ et $i = 0, 1, 2, 3$. Les angles sont en **radians** ; arrondis à 4 décimales, sauf en a) (à l'entier) et en b) (à 3 décimales).

a) La période de la paire $i = 2$ (colonnes 4 et 5), en positions, arrondie à l'entier.
b) $PE_{25, 0}$.
c) $PE_{25, 2}$.
d) $PE_{25, 3}$.

La formule a un sens pour toute position réelle $p$. On prend une position $p$ pour laquelle la paire 1 vaut exactement $PE_{p, 2} = 0{,}6$ et $PE_{p, 3} = 0{,}8$. **Sans calculer $p$**, avec la matrice du décalage de la fiche (203.5) :

e) $PE_{p+10, 2}$.
f) $PE_{p+10, 3}$.
g) Le produit scalaire $PE_0 \cdot PE_2$ des encodages des positions 0 et 2 (les 8 colonnes).
h) Vrai ou faux : $PE_{100} \cdot PE_{102} = PE_0 \cdot PE_2$.

Dans ta copie : vérifie que $e^2 + f^2 = 1$ et dis pourquoi c'était attendu. Que vaut $p$ ? Pourquoi les colonnes 6 et 7 ne distinguent-elles presque pas les positions 0 à 100 ?

### Ex B3.5 — Compter les paramètres d'un GPT et retrouver les 124 millions de GPT-2 small ✏️ ★★ ⏱️ 25 min
**Objectif :** compter les paramètres d'un GPT bloc par bloc, retrouver le compte exact de GPT-2 small et celui du mini-GPT du notebook.
**Prérequis :** Ex B3.3 · ch. 16 (paramètres d'une couche dense) · fiche §203.6 (compter les paramètres d'un bloc), §203.7 · **Parcours :** R, M

GPT-2 small, d'après son `config.json` : 12 blocs pré-LN, $d = 768$, 12 têtes, $V = 50\,257$ tokens, 1 024 positions apprises, un MLP de largeur $F = 4d$, des biais dans toutes les couches denses, deux layer norms par bloc ($\gamma$ et $\beta$) et une layer norm finale ; la couche de sortie réutilise la table des tokens (poids liés) et n'a pas de biais.

**Attention aux « 117M ».** Le tableau 2 de l'article de GPT-2 (Radford et coll., 2019) annonce « 117M » pour ce modèle (et « 345M » pour medium). Ce nombre est faux : le dépôt officiel `openai/gpt-2` reconnaît une erreur de comptage. Le modèle publié a exactement le nombre de paramètres que rend `sum(p.numel() for p in model.parameters())`, et c'est ce nombre que tu dois retrouver.

a) Le nombre de paramètres de l'attention d'un bloc.
b) Celui du MLP d'un bloc.
c) Celui d'un bloc entier, ses deux layer norms comprises.
d) Celui des 12 blocs.
e) Celui des deux tables d'embedding ensemble (tokens et positions).
f) Le nombre total de paramètres de GPT-2 small.
g) Le nombre de paramètres **hors embeddings** (la convention de Kaplan et coll. : sans la table des tokens ni celle des positions, la layer norm finale comprise).
h) De combien de pour cent le compte exact f) dépasse-t-il les 117 millions de l'article, à 1 décimale ?
i) Vrai ou faux : sans poids liés (une matrice de sortie à part, sans biais), GPT-2 small aurait plus de 160 millions de paramètres.
j) Le mini-GPT du notebook a la même architecture, avec 4 blocs, $d = 128$, 4 têtes, $V = 88$ caractères et 128 positions. Combien de paramètres a-t-il en tout ?

Dans ta copie : quelle part d'un bloc de GPT-2 est dans le MLP ? Compare le compte exact des blocs de j) à l'approximation $12 d^2$ par bloc, et dis ce qui fait l'écart.

### Ex B3.6 — Pourquoi √d_k : variance du produit scalaire de deux vecteurs aléatoires ∂ ★★ ⏱️ 25 min
**Objectif :** démontrer que le produit scalaire de deux vecteurs aléatoires a une variance $d_k$, puis chiffrer ce que la division par $\sqrt{d_k}$ change aux gradients de la softmax.
**Prérequis :** ch. 2 (espérance, variance d'une somme de variables indépendantes) · Ex B3.1 · fiche §203.2 (encadré 🧮 sur la variance d'un produit) · **Parcours :** M

Hypothèses : $\mathbf{q} = (q_1, \dots, q_{d_k})$ et $\mathbf{k} = (k_1, \dots, k_{d_k})$ ont $2 d_k$ composantes **indépendantes**, de moyenne 0 et de variance 1. Rappels : $\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ ; $\operatorname{Var}(aX) = a^2 \operatorname{Var}(X)$ ; la variance d'une somme de variables **indépendantes** est la somme de leurs variances (ch. 2). Réponds dans ta copie.

1. Montre que $\mathbb{E}[q_c k_c] = 0$ et que $\operatorname{Var}(q_c k_c) = 1$ pour chaque $c$.
2. Les $d_k$ produits $q_c k_c$ sont indépendants entre eux (on l'admet : ce sont des fonctions de couples indépendants). Montre que $\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = 0$ et $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$, puis que $\mathbf{q} \cdot \mathbf{k} / \sqrt{d_k}$ a une variance 1.
3. Les composantes de $\mathbf{q}$ ont maintenant une variance $\sigma_q^2$ et celles de $\mathbf{k}$ une variance $\sigma_k^2$ (toujours centrées). Que vaut $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k})$ ? Par quoi faudrait-il diviser pour revenir à une variance 1 ?
4. Les composantes sont maintenant de moyenne $\mu \ne 0$ et de variance 1. Montre que $\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = d_k \mu^2$ et que $\operatorname{Var}(q_c k_c) = 1 + 2\mu^2$. Un même décalage ajouté à tous les scores d'une ligne change-t-il les poids de la softmax (ch. 17) ? Qu'est-ce qui compte alors pour la saturation ?
5. **La saturation, chiffrée.** Une requête et deux clés, de scores $(s, 0)$. Montre que le poids de la première clé vaut $p = \sigma(s) = 1/(1 + e^{-s})$, et que $\partial p / \partial s = p(1 - p)$. Calcule $p$ et $p(1 - p)$ pour $s = 8$ (un écart-type de score pour $d_k = 64$, sans la division) et pour $s = 1$ (le même écart après la division par $\sqrt{64}$). Par combien le gradient est-il multiplié ?
6. Si l'on divisait par $d_k$ au lieu de $\sqrt{d_k}$, que vaudrait l'écart-type des scores pour $d_k = 64$ ? Calcule $p$ pour un écart de scores égal à cet écart-type, et dis ce que deviendrait l'attention.
7. Un modèle qui apprend $\mathbf{W}_Q$ pourrait réduire lui-même l'échelle de ses scores. Pourquoi fixer quand même $1/\sqrt{d_k}$ dès le départ ?

### Ex B3.7 — RoPE : la rotation rend q_m · k_n fonction de m − n seulement ∂ ★★★ ⏱️ 35 min
**Objectif :** démontrer la propriété de RoPE en dimension 2, puis en dimension $d_h$, et la comparer à l'encodage additif.
**Prérequis :** Ex B3.4 · fiche §203.5 (RoPE, encadré 🧮 sur la rotation d'un vecteur du plan) · **Parcours :** M

On note $R_\varphi$ la matrice de rotation d'angle $\varphi$ de la fiche, et, pour RoPE en dimension 2 avec la fréquence $\theta$, $R_m = R_{m\theta}$ la rotation de la position $m$. Les vecteurs s'écrivent en colonne.

> 🧮 **Rappel maths — produit scalaire et transposée d'un produit** — Pour deux vecteurs colonnes, $\langle \mathbf{a}, \mathbf{b} \rangle = \mathbf{a}^\top \mathbf{b}$. La transposée d'un produit inverse l'ordre : $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top$, donc $(\mathbf{A}\mathbf{x})^\top = \mathbf{x}^\top \mathbf{A}^\top$. Exemple : $\mathbf{A} = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$, $\mathbf{x} = (1, 1)^\top$ : $\mathbf{A}\mathbf{x} = (3, 1)^\top$, et $\mathbf{x}^\top \mathbf{A}^\top = (1, 1) \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix} = (3, 1)$.

Réponds dans ta copie.

1. Montre que $\langle R_m \mathbf{q}, R_n \mathbf{k} \rangle = \mathbf{q}^\top R_m^\top R_n \mathbf{k}$, puis, avec les propriétés de l'encadré de la fiche, que $\langle R_m \mathbf{q}, R_n \mathbf{k} \rangle = \langle \mathbf{q}, R_{n-m} \mathbf{k} \rangle$.
2. Développe : montre que $\langle \mathbf{q}, R_\varphi \mathbf{k} \rangle = (\mathbf{q} \cdot \mathbf{k}) \cos\varphi + (q_2 k_1 - q_1 k_2) \sin\varphi$. Vérifie ta formule sur l'exemple de la fiche ($\mathbf{q} = (1, 2)$, $\mathbf{k} = (3, -1)$, $\theta = 0{,}5$ : écarts $+3$ et $-3$).
3. **Application.** $\theta = \pi/6$, $\mathbf{q} = (2, 1)$, $\mathbf{k} = (1, 1)$. Calcule le score $\langle R_m \mathbf{q}, R_n \mathbf{k} \rangle$ pour $(m, n) = (0, 0)$, $(5, 2)$, $(1, 4)$ et $(7, 10)$. Que remarques-tu ?
4. En dimension $d_h$, RoPE tourne chaque paire $i$ d'un angle $m\,\theta_i$ : la matrice de la position $m$ est **diagonale par blocs**, un bloc $R_{m\theta_i}$ de taille $2 \times 2$ par paire. Montre que la propriété de la question 1 reste vraie, chaque paire ayant sa propre fréquence.
5. Montre que $\lVert R_m \mathbf{q} \rVert = \lVert \mathbf{q} \rVert$ pour tout $d_h$. Pourquoi est-ce une bonne nouvelle pour la mise à l'échelle de ∂ B3.6 ?
6. **Comparaison.** Avec l'encodage additif, la requête et la clé viennent de $\mathbf{x}_m + \mathbf{p}_m$ et $\mathbf{x}_n + \mathbf{p}_n$ ($\mathbf{p}$ sinusoïdal, sans projection pour simplifier). Développe $(\mathbf{x}_m + \mathbf{p}_m) \cdot (\mathbf{x}_n + \mathbf{p}_n)$ en quatre termes. Lesquels dépendent de $m$ et $n$ séparément, et lequel seulement de $n - m$ ?
7. **KV cache.** Avec RoPE, on tourne chaque clé selon sa position **avant** de la ranger dans le cache. Montre, avec la question 1, que cette clé tournée reste juste pour toutes les requêtes qui viendront ensuite, quelle que soit leur position.

### Ex B3.8 — Fermi : FLOPs d'entraînement (≈ 6 N D), ton mini-GPT sur CPU, GPT-3, et la mémoire du KV cache 🧮 ★★ ⏱️ 20 min
**Objectif :** chiffrer en ordres de grandeur le calcul d'un entraînement et la mémoire d'un KV cache, de ton mini-GPT à un grand modèle.
**Prérequis :** Ex B3.5 · fiche §203.9 (ce que coûte le cache), §203.10 (calcul d'une couche, $6ND$) · **Parcours :** R, M

Hypothèses de calcul (des ordres de grandeur, pas des mesures) :
- **ton mini-GPT** : le nombre de paramètres $N$ de ✏️ B3.5 j) ; 4 blocs, $d = 128$, contexte $T = 128$ ; en mode rapide, 600 itérations de 32 séquences de 128 tokens ; en mode complet, 5 000 itérations ; ton processeur soutient environ $10^{11}$ FLOP/s en float32 ;
- **GPT-3** (Brown et coll., 2020, tableau 2.1) : 175 milliards de paramètres, 96 couches, $d = 12\,288$, contexte de 2 048 tokens, entraîné sur 300 milliards de tokens ; on suppose des GPU qui soutiennent chacun $3 \times 10^{13}$ FLOP/s ;
- **Qwen3-8B** (`config.json` du Hub) : 36 couches, 8 têtes de clés et de valeurs de taille 128 (GQA), valeurs en bfloat16 (2 octets) ; ses poids occupent environ 16 Go ;
- on néglige le terme de l'attention dans $6ND$, sauf à la question 2.

Réponds dans ta copie, avec des puissances de 10.

1. Combien de tokens ton mini-GPT voit-il en mode rapide ? Combien de FLOPs d'entraînement, d'après $6ND$, et combien de temps sur ton processeur ? Même question en mode complet. Le mode complet mérite-t-il le 🚀 ?
2. Le terme de l'attention ajoute environ $T/(12d)$ au calcul des couches denses d'un décodeur causal (203.10). Que vaut-il pour ton mini-GPT ? Change-t-il ta réponse 1 ?
3. Combien de FLOPs pour entraîner GPT-3 ? Combien de temps sur un seul GPU, puis sur 1 000 GPU ? Combien de fois le calcul de ton mini-GPT en mode rapide ?
4. Avec la règle de Chinchilla (environ 20 tokens par paramètre), sur combien de tokens aurait-il fallu entraîner GPT-3 ? Combien de FLOPs cela ferait-il ?
5. La mémoire du KV cache de ton mini-GPT pour une séquence de 128 tokens en float32 (formule de la fiche, $L$ = nombre de couches), puis pour un batch de 32 séquences.
6. Le KV cache de Qwen3-8B, par token, puis pour une conversation de 32 768 tokens. Sur un GPU de 24 Go, combien de tokens de cache tiennent à côté des poids ? Combien de conversations de 4 096 tokens servirait-on en même temps ? Et sans GQA (32 têtes de clés et de valeurs) ?
7. Cite trois choses que ces estimations ne comptent pas.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B3.9 — L'attention expliquée en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer le mécanisme d'attention à un débutant, sans formule.
**Prérequis :** fiche §203.2 · **Parcours :** R

Explique en cinq lignes au plus, à quelqu'un qui n'a jamais programmé, ce que fait l'attention dans un modèle de langue. Ton explication doit faire apparaître (1) le problème : un mot a besoin d'informations qui sont ailleurs dans la phrase ; (2) ce que sont la requête, la clé et la valeur ; (3) comment se calculent les poids ; (4) ce qu'est la sortie ; (5) ce que l'on gagne par rapport à une lecture mot après mot. Les mots « requête », « clé », « valeur » et « poids » sont permis s'ils sont expliqués ; une image de la vie courante est la bienvenue.

### Ex B3.10 — Mémorisation : quand un modèle de langue recrache ses données d'entraînement ⚖️ ★★ ⏱️ 20 min
**Objectif :** mesurer la mémorisation d'un modèle de langue, en peser les risques et proposer des protections.
**Prérequis :** ch. 29 (data cards, données personnelles) · fiche §203.8 (ce que le modèle retient) · **Parcours :** —

Ton mini-GPT voit Holmes environ 5 fois en mode rapide (2,5 millions de tokens pour environ 500 000 caractères d'entraînement), et une quarantaine de fois en mode complet. Une compagnie d'assurance veut, elle, entraîner un petit modèle de langue sur 200 000 conversations de son service client (noms, numéros de téléphone, numéros de contrat, détails de sinistres), pour suggérer des réponses à ses conseillers. Réponds dans ta copie.

1. **Mesurer.** Distingue mémoriser et généraliser. Décris un protocole pour mesurer la mémorisation de ton mini-GPT sur Holmes : quel prompt, quelle génération (tirage ou choix du plus probable), quelle comparaison, quel groupe témoin ? Quel résultat t'inquiéterait ?
2. Qu'ont montré N. Carlini et ses coauteurs sur GPT-2 (« Extracting Training Data from Large Language Models », USENIX Security 2021) ? Une donnée vue une seule fois peut-elle ressortir ? Les grands modèles sont-ils plus ou moins exposés ?
3. Carlini et coll. (« Quantifying Memorization Across Neural Language Models », 2022) relient la mémorisation à trois facteurs. Lesquels ? Que prédisent-ils pour le modèle de l'assureur, entraîné de nombreuses fois sur des conversations qui se répètent ?
4. Quels risques pour les clients et pour l'assureur ? Pense au RGPD : minimisation des données, droit à l'effacement (art. 17). Que veut dire « effacer » une donnée d'un modèle déjà entraîné ?
5. Propose au moins quatre protections, avant, pendant et après l'entraînement (pense à la déduplication de K. Lee et coll., 2022, aux données personnelles, à l'entraînement à confidentialité différentielle, aux tests par « canaris » de Carlini et coll., 2019, aux filtres de sortie). Pour chacune, ce qu'elle coûte.
6. Holmes est dans le domaine public. Que changerait un corpus protégé par le droit d'auteur ? Que demande l'article 53 du règlement européen sur l'IA aux fournisseurs de modèles d'IA à usage général ?
7. Rédige les quatre lignes « Limites et risques » de la fiche modèle de l'assureur.

### Ex B3.11 — Vaswani et coll. (2017) : « Attention Is All You Need » 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a introduit le Transformer : son architecture, ses choix, ses tableaux, et ce qui a changé depuis.
**Prérequis :** Ex B3.3 · fiche §203.2 à §203.7 · **Parcours :** M

Lis A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. Gomez, Ł. Kaiser et I. Polosukhin, « Attention Is All You Need » ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762), NeurIPS 2017) : le résumé, les sections 1 et 3, la figure 1, la section 4 et son tableau 1, la section 5, puis les tableaux 2 et 3 et la section 6. Réponds dans ta copie.

1. **Le problème** (section 1). Qu'est-ce qui empêche un réseau récurrent de paralléliser l'entraînement à l'intérieur d'un exemple ? Combien de temps et quel matériel faut-il au Transformer pour atteindre un nouvel état de l'art ?
2. **L'architecture** (figure 1, section 3.1). Combien de couches dans l'encodeur et dans le décodeur ? Quelles sous-couches dans chacun ? Écris la formule de la sortie d'une sous-couche : est-ce un bloc pré-LN ou post-LN ? Quelles deux précautions garantissent que la prédiction de la position $i$ ne dépend que des positions précédentes ?
3. **L'attention** (section 3.2.1 et note 4). Recopie l'équation (1). Quelle est la différence avec l'attention additive, et pourquoi le produit scalaire est-il plus rapide ? Quel argument la note 4 donne-t-elle pour la division par $\sqrt{d_k}$ (∂ B3.6) ?
4. **Les têtes** (sections 3.2.2 et 3.2.3). Que valent $h$, $d_k$ et $d_v$ dans le modèle de base ? Quelle est la forme de $\mathbf{W}^O$ ? Combien de paramètres a une attention multi-têtes de ce modèle, biais compris (comme dans ta librairie) ? Cite les trois usages de l'attention dans le modèle.
5. **Le tableau 1** (section 4). Compare la complexité par couche, les opérations séquentielles et la longueur maximale des chemins de l'auto-attention et d'une couche récurrente. À quelle condition l'auto-attention est-elle moins coûteuse ? Que proposent les auteurs pour les séquences très longues, et à quel prix ?
6. **Embeddings et positions** (sections 3.4 et 3.5). Que partagent les auteurs entre les couches d'embedding et la couche de sortie, et par quoi multiplient-ils les poids d'embedding ? Entre quelles valeurs vont les longueurs d'onde des sinusoïdes ? Pourquoi les auteurs ont-ils choisi des sinusoïdes (relie leur argument à ✏️ B3.4), et que montre la ligne (E) du tableau 3 ?
7. **L'entraînement** (section 5). Sur quelles données, avec quel vocabulaire ? Combien de temps par pas et combien de pas pour le modèle de base, et pour le grand modèle ? Écris l'équation (3) du learning rate et calcule sa valeur maximale pour $d_{\text{model}} = 512$ et 4 000 pas de warmup. Quelles régularisations, et que fait le label smoothing à la perplexité ?
8. **Les résultats** (tableaux 2 et 3, section 6). Quels BLEU sur l'anglais-allemand et sur l'anglais-français ? Compare le BLEU anglais-français du résumé et celui de la section 6.1 : que remarques-tu ? Retrouve le coût d'entraînement du modèle de base du tableau 2 à partir de la note 5 et de la section 5.2. Que montrent les lignes (A) du tableau 3 sur le nombre de têtes ? Compte enfin les paramètres du modèle de base (6 + 6 couches, $d_{\text{model}} = 512$, $d_{ff} = 2\,048$, un vocabulaire partagé de 37 000 tokens, une seule matrice d'embedding) et compare aux 65 millions du tableau 3.
9. **Et aujourd'hui** (fiche §203.6 et §203.7). Donne quatre différences entre ce Transformer de 2017 et le bloc d'un LLM de 2026.

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### B3.E1 — Explique le mécanisme d'attention, de la requête à la sortie 💼 ★★ ⏱️ 10 min
*Fiche §203.2, §203.4 · parcours R*

« Expliquez-moi l'attention d'un Transformer, depuis les vecteurs des tokens jusqu'à la sortie d'une couche multi-têtes. »

### B3.E2 — Quel est le coût de l'auto-attention en fonction de la longueur, et comment traite-t-on les contextes longs ? 💼 ★★ ⏱️ 10 min
*Fiche §203.10 · parcours R*

« Que coûte l'auto-attention quand le contexte s'allonge, en calcul et en mémoire ? Comment les modèles actuels tiennent-ils 128 000 tokens ? »

### B3.E3 — Encodeur, décodeur ou encodeur-décodeur : lequel pour quelle tâche ? 💼 ★★ ⏱️ 10 min
*Fiche §203.7 · parcours R*

« Nous avons trois projets : classer des e-mails, résumer des contrats, et un assistant qui répond aux clients. Quelle famille de Transformer pour chacun, et pourquoi ? »

### B3.E4 — À quoi sert le KV cache, et combien de mémoire consomme-t-il ? 💼 ★★ ⏱️ 10 min
*Fiche §203.9 · parcours R*

« Expliquez le KV cache. Combien de mémoire prend-il pour un modèle de 8 milliards de paramètres à 32 000 tokens de contexte, et comment la réduit-on ? »

### B3.E5 — Pourquoi un Transformer a-t-il besoin d'encodages de position, et qu'apporte RoPE ? 💼 ★★ ⏱️ 10 min
*Fiche §203.5 · parcours R*

« Pourquoi faut-il donner les positions à un Transformer, et pourquoi presque tous les LLM récents utilisent-ils RoPE ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`, en deux phases : **ta librairie NumPy** (B3.12 à B3.17 : masques et attention, têtes, positions, bloc complet, chasse aux masques), puis **le mini-GPT en PyTorch** (B3.18 à B3.30 : module d'attention, modèle, entraînement, génération, cartes d'attention, expériences, ViT, test de causalité, défi).

| ID | Titre | Type | ★ | ⏱️ | 🚀 |
|---|---|---|---|---|---|
| B3.12 | causal_mask, padding_mask et scaled_dot_product_attention | 🔨 | ★★ | 30 | |
| B3.13 | Prédire la carte d'attention quand une clé copie la requête, puis quand on retire la division par √d_k | 🔮 | ★ | 10 | |
| B3.14 | split_heads, merge_heads et multi_head_attention | 🔨 | ★★★ | 45 | |
| B3.15 | positional_encoding et apply_rope | 🔨 | ★★ | 30 | |
| B3.16 | transformer_block en NumPy, validé contre nn.TransformerEncoderLayer | 🔨 | ★★★★ | 100 | |
| B3.17 | Trois masques qui mentent : convention inversée, masque transposé, padding compté dans la loss | 🐛 | ★★ | 25 | |
| B3.18 | De NumPy à PyTorch : un module CausalSelfAttention avec F.scaled_dot_product_attention | 📦 | ★★ | 30 | |
| B3.19 | Mini-GPT caractère par caractère : embeddings de tokens et de positions, blocs pré-LN, tête liée | 📦 | ★★★ | 60 | |
| B3.20 | Entraîner le mini-GPT sur Holmes : AdamW, warmup + cosinus, clipping, perplexité, face au LSTM du ch. 22 | 📦 | ★★★ | 60 | 🚀 |
| B3.21 | Lire les courbes d'entraînement du mini-GPT et poser un diagnostic | 📈 | ★★ | 15 | |
| B3.22 | Prédire : le mini-GPT de Holmes amorcé avec une phrase de Verne | 🔮 | ★ | 10 | |
| B3.23 | Générer vite : le KV cache, et le gain de vitesse mesuré selon la longueur | 📦 | ★★★ | 45 | |
| B3.24 | Les cartes d'attention de chaque tête sur une phrase de Holmes | 📈 | ★★ | 20 | |
| B3.25 | Ablations : sans position, sans résiduel, post-LN, une seule tête | 🔬 | ★★★ | 60 | 🚀 |
| B3.26 | Tokens BPE du B2 contre caractères : perplexité par caractère à budget égal | 🔬 | ★★★ | 45 | 🚀 |
| B3.27 | Temps et mémoire de l'attention selon la longueur du contexte (64 à 2 048) | 🔬 | ★★ | 25 | |
| B3.28 | Un Vision Transformer minuscule : Fashion-MNIST découpé en patchs | 📦 | ★★★ | 45 | 🚀 |
| B3.29 | Un test pytest qui prouve la causalité : modifier le futur ne change pas les logits passés | 🛠️ | ★★ | 20 | |
| B3.30 | Défi : la meilleure perplexité caractère sur Holmes avec au plus 1 M de paramètres et 10 min de CPU | 🏆 | ★★★★ | 120 | |

Le notebook ne télécharge aucun modèle pré-entraîné. Le mini-GPT s'entraîne sur CPU en mode rapide (quelques minutes), est sauvegardé une fois, puis rechargé par les exercices qui s'en servent ; les modes complets marqués 🚀 tournent mieux sur un GPU de Colab, et le défi impose lui-même 10 minutes de CPU.
