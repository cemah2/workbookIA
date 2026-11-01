# B2 · Tokenisation et embeddings — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B2.Q1 — Caractères, mots ou sous-mots : taille du vocabulaire contre longueur des séquences

<details><summary>Indice 1</summary>

Relis « Trois granularités » et son tableau (§202.1).

</details>
<details><summary>Indice 2</summary>

Pour chaque granularité, demande-toi combien de symboles distincts il faut, et combien il en faut pour écrire un même texte : les deux vont en sens inverse. Pour c), « comme le carré » veut dire : longueur multipliée par $k$, coût multiplié par $k^2$.

</details>
<details><summary>Indice 3</summary>

a) **A** : un caractère par identifiant, la séquence la plus longue (≈ 562 000 identifiants pour Holmes, contre ≈ 106 000 en mots). Pour b), relis ce que devient un mot qui n'est pas parmi les $V$ plus fréquents. Pour c), calcule $3^2$. Pour d), relis la dernière phrase du paragraphe sur les sous-mots. Pour e), compare les nombres de symboles distincts du tableau.

</details>

### B2.Q2 — Octets, points de code, caractères affichés : combien « é » compte-t-il ?

<details><summary>Indice 1</summary>

Relis « Points de code et octets », son tableau, et « Un caractère affiché, plusieurs points de code » (§202.2).

</details>
<details><summary>Indice 2</summary>

Situe chaque point de code dans une ligne du tableau d'UTF-8 : sa plage donne son nombre d'octets. En NFD, compte les octets de **chacun** des points de code, puis additionne. `len` d'un `str` compte des points de code, `len` d'un `bytes` des octets.

</details>
<details><summary>Indice 3</summary>

a) U+00E9 est entre U+0080 et U+07FF : **2** octets. Pour b), combien de points de code font « e » suivi de U+0301 ? Pour c), additionne les octets de « e » (ASCII) et de U+0301 (quelle plage ?). Pour d), que compte `len` sur un `str` ? Pour e), dans quelle plage tombe U+2019 ?

</details>

### B2.Q3 — NFC ou NFD : deux chaînes identiques à l'écran, différentes en mémoire

<details><summary>Indice 1</summary>

Relis « Un caractère affiché, plusieurs points de code », l'encadré sur `unicodedata`, « Casse et accents » et le piège classique du §202.2.

</details>
<details><summary>Indice 2</summary>

`==` compare des points de code, pas des pixels. NFD décompose (lettre de base, puis combinant), NFC recompose, NFKC remplace en plus des variantes de présentation. Pour d), un BPE fusionne des identifiants de caractères : « é » composé et « e » + U+0301 n'ont pas les mêmes.

</details>
<details><summary>Indice 3</summary>

a) **Faux** : « café » en NFC a 4 points de code, sa forme NFD en a 5 ; `==` les trouve différentes. Pour b), quelle méthode garde la lettre de base et retire seulement l'accent ? Pour c), peut-on retrouver « ﬁ » à partir de « fi » ? Pour d), les fusions apprises sur « e » + U+0301 voient-elles U+00E9 ? Pour e), que donne NFC appliqué à la décomposition d'une chaîne déjà composée ?

</details>

### B2.Q4 — Fusions BPE : pourquoi la liste ordonnée des fusions est le modèle

<details><summary>Indice 1</summary>

Relis « L'entraînement », l'encadré 🕰️ sur le départage et « Encoder un mot nouveau : les fusions par rang » (§202.3).

</details>
<details><summary>Indice 2</summary>

Ce qui sert à encoder, c'est ce qui permet de rejouer les fusions dans le même ordre. Pour c), une paire se compte une fois par occurrence dans le mot, puis on multiplie par la fréquence du mot. Pour d), une fusion ajoute un token ; les deux tokens fusionnés servent encore ailleurs.

</details>
<details><summary>Indice 3</summary>

a) **B** : l'alphabet et la liste ordonnée des fusions ; deux tokenizers peuvent avoir les mêmes tokens et découper autrement. Pour b), relis la phrase sur la frontière des mots. Pour c), écris les paires de « b a n a n e » et compte « a n », puis multiplie par 3. Pour d), qu'ajoute une fusion au vocabulaire, et que retire-t-elle ? Pour e), relis l'encadré 🕰️ : comment `BpeTrainer` compare-t-il deux paires à égalité ?

</details>

### B2.Q5 — BPE au niveau octet : jamais de token inconnu, vrai ou faux ?

<details><summary>Indice 1</summary>

Relis « Plus jamais de token inconnu » (§202.4) et l'encadré ⚠️ « un BPE qui jette des caractères » (§202.5).

</details>
<details><summary>Indice 2</summary>

Au niveau octet, l'alphabet contient les 256 octets : tout texte UTF-8 s'écrit avec. Les fusions ne font que regrouper ; sans fusion, chaque octet est un token. Pour e), relis ce que donne le décodage d'un seul octet d'un caractère de deux octets.

</details>
<details><summary>Indice 3</summary>

a) **Vrai** : le birman s'écrit en UTF-8, donc avec les 256 octets de l'alphabet. Pour b), combien de tokens donnent 3 octets sans fusion ? Pour c), relis ce que fait `models.BPE()` sans `unk_token`. Pour d), relis la dernière phrase du premier paragraphe du §202.4. Pour e), que doit faire un programme qui affiche les tokens un à un quand il reçoit le premier octet de « Ç » ?

</details>

### B2.Q6 — Tokens spéciaux et espace initial : « Ġthe » n'est pas « the »

<details><summary>Indice 1</summary>

Relis « Pourquoi « Ġ » ? », « La pré-tokenisation » et « Les tokens spéciaux » (§202.4).

</details>
<details><summary>Indice 2</summary>

Chez GPT-2, l'espace reste collée au mot qui la suit, et elle fait partie du token. Un token spécial est un signal : il a son identifiant et n'est jamais découpé. Pour e), relis la phrase qui suit le tableau des tokens spéciaux.

</details>
<details><summary>Indice 3</summary>

a) **Faux** : « the » et « Ġthe » sont deux tokens différents, donc deux identifiants. Pour b), que devient l'octet 0x20 dans la table octet → caractère ? Pour c), relis la définition d'un token spécial. Pour d), le motif de GPT-2 jette-t-il un seul caractère ? Pour e), combien de tokens spéciaux le paragraphe sous le tableau attribue-t-il à GPT-2 ?

</details>

### B2.Q7 — WordPiece, Unigram, SentencePiece : quel critère pour quelle méthode ?

<details><summary>Indice 1</summary>

Relis les trois premières sous-sections du §202.5 : WordPiece, Unigram, SentencePiece.

</details>
<details><summary>Indice 2</summary>

BPE et WordPiece partent des caractères et fusionnent ; Unigram part de trop et élague. BPE choisit la paire au plus grand **compte** ; WordPiece, au plus grand **score** $\frac{\operatorname{count}(a, b)}{\operatorname{count}(a)\operatorname{count}(b)}$. SentencePiece est une bibliothèque, pas un critère.

</details>
<details><summary>Indice 3</summary>

a) **C** : Unigram part d'un grand vocabulaire et l'élague. Pour b), calcule $\frac{20}{40 \times 100}$ et $\frac{30}{300 \times 200}$, puis compare. Pour c), compare seulement 20 et 30. Pour d), relis ce que SentencePiece fait des espaces. Pour e), relis l'exemple de « unbelievably ».

</details>

### B2.Q8 — Pourquoi un LLM compte mal les lettres d'un mot et calcule mal avec les chiffres

<details><summary>Indice 1</summary>

Relis « Les chiffres et les lettres » (§202.6) et la fin de « La pré-tokenisation » (§202.4).

</details>
<details><summary>Indice 2</summary>

Le modèle ne voit que des identifiants. Pour les nombres, applique chaque règle telle qu'elle est écrite : Qwen3, un token par chiffre ; Llama 3, des groupes d'au plus 3 chiffres pris depuis la gauche.

</details>
<details><summary>Indice 3</summary>

a) **C** : quelques identifiants ; il doit avoir appris de quelles lettres chaque token est fait. Pour b), compare les découpages de « 1234567 » et « 1234656 » par GPT-2. Pour c), combien de chiffres compte « 2026 » ? Pour d), découpe « 12345678 » en groupes de 3 depuis la gauche et compte-les. Pour e), relis la dernière phrase de « Les nombres ».

</details>

### B2.Q9 — nn.Embedding : une table de consultation entraînable, pas une couche magique

<details><summary>Indice 1</summary>

Relis « Une couche dense sur un one-hot, c'est une lecture de table » et son exemple PyTorch (§202.7).

</details>
<details><summary>Indice 2</summary>

Chaque identifiant lit une ligne de $d$ nombres : la forme de sortie ajoute l'axe $d$ à celle des identifiants. La table a $V \times d$ paramètres, sans biais. Pour d) et e), la règle de la chaîne : chaque lecture ajoute sa contribution à la ligne lue, et une ligne jamais lue ne reçoit rien.

</details>
<details><summary>Indice 3</summary>

a) **`"(8, 20, 64)"`** : on ajoute l'axe de l'embedding à la forme $(8, 20)$ des identifiants. Pour b), calcule $5\,000 \times 32$. Pour c), relis la formule $\mathbf{x}_{\text{one-hot}}^\top \mathbf{E}$. Pour d), que dit le paragraphe sous l'exemple d'une ligne lue deux fois ? Pour e), sans momentum ni weight decay, qu'est-ce qui pourrait changer une ligne jamais lue ?

</details>

### B2.Q10 — L'hypothèse distributionnelle : un mot se définit par ses voisins

<details><summary>Indice 1</summary>

Relis « L'hypothèse distributionnelle », l'encadré 🧮 sur la PMI et la partie sur word2vec (§202.8).

</details>
<details><summary>Indice 2</summary>

« Contextes semblables » ne veut pas dire « voisins l'un de l'autre » : « thé » et « café » partagent leurs voisins (« boire », « tasse »), sans forcément se suivre. Pour c), $p(w, c) = \#(w, c)/N$, $p(w) = \#(w)/N$, $p(c) = \#(c)/N$.

</details>
<details><summary>Indice 3</summary>

a) **B** : des contextes semblables. Pour b), relis le sens du signe de la PMI. Pour c), calcule $\frac{6/1\,000}{(30/1\,000)(200/1\,000)}$, puis son logarithme. Pour d), relis la définition de CBOW à la fin de la partie word2vec. Pour e), calcule $1 - \sqrt{t/f}$ pour un mot fréquent et pour un mot rare : lequel est jeté le plus souvent ?

</details>

### B2.Q11 — Statique ou contextuel : l'« avocat » du tribunal et celui du marché

<details><summary>Indice 1</summary>

Relis « Un mot, plusieurs sens » et « D'une suite de vecteurs à un vecteur de phrase » (§202.10).

</details>
<details><summary>Indice 2</summary>

Un embedding statique est une ligne de table : un mot, un vecteur. Un Transformer part de cette ligne et la transforme selon toute la phrase. Pour e), applique la formule du mean pooling avec le masque : seuls les vrais tokens comptent, au numérateur et au dénominateur.

</details>
<details><summary>Indice 3</summary>

a) **A** : le même vecteur, la ligne de la table. Pour b), d'où sort le vecteur de chaque token à la fin du dernier bloc ? Pour c), un seul vecteur doit servir aux deux sens : lequel pèse le plus dans l'entraînement ? Pour d), relis la formule de $\mathbf{s}$ et ce que fait $m_t$. Pour e), calcule $\frac{1 + 3 + 2 + 0 \times 0}{1 + 1 + 1 + 0}$.

</details>

### B2.Q12 — Cosinus ou distance euclidienne pour des embeddings normalisés ?

<details><summary>Indice 1</summary>

Relis « Le cosinus » (§202.9) et la relation entre distance au carré et cosinus.

</details>
<details><summary>Indice 2</summary>

Pour deux vecteurs de norme 1, $\lVert\mathbf{a} - \mathbf{b}\rVert^2 = 2 - 2\cos(\mathbf{a}, \mathbf{b})$ : la distance est une fonction décroissante du cosinus. Sans normalisation, le produit scalaire mélange direction et longueur.

</details>
<details><summary>Indice 3</summary>

a) $2 - 2 \times 0{,}8 =$ **0,4**. Pour b), que devient l'ordre quand on applique une fonction décroissante ? Pour c), calcule $\sqrt{2 - 2 \times 0}$. Pour d), imagine un vecteur de même direction mais dix fois plus long. Pour e), compare la direction de $(-6, -8)$ à celle de $(3, 4)$.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### B2.R1 — Ch. B1 : à quoi sert l'embedding d'image extrait d'un réseau pré-entraîné ?

<details><summary>Indice 1</summary>

Relis la fiche de B1 : §201.3 (embeddings figés, linear probe, k plus proches voisins) et le cosinus.

</details>
<details><summary>Indice 2</summary>

Un embedding figé est la sortie d'un backbone gelé : il ne change jamais, on le calcule une fois. Pour b), la tête d'un ResNet-18 est `nn.Linear(512, 1000)`. Pour e), cherche dans B1 la brique qui transforme une entrée en vecteur réutilisable.

</details>
<details><summary>Indice 3</summary>

a) **Vrai** : calculé une fois et sauvegardé, il sert à tous les classifieurs posés dessus. Pour b), lis la taille d'entrée de `nn.Linear(512, 1000)`. Pour c), quel classifieur B1 entraîne-t-il sur les embeddings figés ? Pour d), relis pourquoi B1 compare les embeddings par le cosinus. Pour e), quelle brique de B1 donne un vecteur par image, réutilisable pour plusieurs tâches ?

</details>

### B2.R2 — Ch. 22 : build_vocab, encode, decode et le token inconnu

<details><summary>Indice 1</summary>

Relis ta librairie `sequences.py` (ch. 22, 🔨 22.12) et la docstring de `build_vocab`.

</details>
<details><summary>Indice 2</summary>

Compte les lettres de « banana », range-les par fréquence décroissante : l'identifiant suit ce rang. Avec `unk_token`, il prend l'identifiant 0 et décale tous les autres. Un caractère absent du vocabulaire devient le token inconnu.

</details>
<details><summary>Indice 3</summary>

a) « banana » : a × 3, n × 2, b × 1, d'où a → 0, n → 1, b → **2**. Pour b), décale ces identifiants de 1 (`<unk>` prend 0), puis encode lettre par lettre ; « d » n'est pas dans le vocabulaire. Pour c), que rend `decode` pour l'identifiant de `<unk>` ? Pour d), relis le rôle de `unk_token`. Pour e), avec `max_size=2` et le token inconnu compris, quels mots restent dans le vocabulaire ?

</details>

### B2.R3 — Ch. 6 : entropie, Huffman et ce que l'on compresse vraiment

<details><summary>Indice 1</summary>

Relis la fiche du ch. 6 : entropie, code de Huffman, borne de Shannon, taux de compression ; puis le lien avec le ch. 6 au début du §202.3.

</details>
<details><summary>Indice 2</summary>

$H = -\sum_i p_i \log_2 p_i$. Avec des probabilités puissances de 2, Huffman donne à chaque symbole un mot de code de $-\log_2 p_i$ bits. Le taux de compression du ch. 6 divise les bits envoyés par les bits du code de longueur fixe.

</details>
<details><summary>Indice 3</summary>

a) $H = \frac{1}{2} \times 1 + \frac{1}{4} \times 2 + 2 \times \frac{1}{8} \times 3 =$ **1,75** bit. Pour b), calcule $-\log_2 \frac{1}{2}$. Pour c), relis la phrase du §202.3 qui compare Huffman et BPE. Pour d), l'entropie est-elle un plancher ou un plafond de la longueur moyenne ? Pour e), calcule $\frac{4\,000}{1\,000 \times 8}$.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B2.1 — « Noël à Paris » en UTF-8 : octets, points de code, longueur ✏️

<details><summary>Indice 1</summary>

Relis le tableau d'UTF-8, le calcul de « ç » et l'encadré 🧮 « Rappel outil — `str`, `bytes` et `encode` » (§202.2).

</details>
<details><summary>Indice 2</summary>

Compte les points de code un par un, espaces compris ; puis donne à chacun son nombre d'octets selon sa plage. Pour écrire un point de code de U+0080 à U+07FF : écris-le sur 11 bits, mets les 5 premiers après `110` et les 6 derniers après `10`. Les indices d'un `bytes` numérotent les octets, à partir de 0.

</details>
<details><summary>Indice 3</summary>

a) N, o, ë, l, espace, à, espace, P, a, r, i, s : **12** points de code. Pour b), 10 caractères ASCII et 2 lettres accentuées : additionne leurs octets. Pour c), écris 0xEB sur 11 bits et répartis-les dans `110xxxxx 10xxxxxx`. Pour d), écris les octets de « No » puis ceux de « ë » : lequel a l'indice 3 ? Pour e), en NFD, compte les octets de « e » + U+0308 et de « a » + U+0300, puis ceux des 10 autres caractères. Pour f), même méthode que c) avec 0x153. Pour g), relis ce que fait `decode` sans `errors` sur un octet de début isolé.

</details>

### Ex B2.2 — Analogie à la main : roi − homme + femme sur des vecteurs de dimension 3 ✏️

<details><summary>Indice 1</summary>

Relis « Les analogies » et son exemple, ainsi que « Le cosinus » (§202.9).

</details>
<details><summary>Indice 2</summary>

Calcule d'abord $\mathbf{q}$ composante par composante, et sa norme. Puis, pour chaque mot, son produit scalaire avec $\mathbf{q}$ et sa norme : le cosinus est leur rapport divisé par $\lVert\mathbf{q}\rVert$. Classe les sept mots, avec ou sans les trois mots de la question.

</details>
<details><summary>Indice 3</summary>

a) $\mathbf{q} = (0{,}6 ;\ 1 ;\ 0{,}9) - (0{,}1 ;\ 1 ;\ 0{,}9) + (0{,}1 ;\ -1 ;\ 0{,}9) = (0{,}6 ;\ -1 ;\ 0{,}9)$, et $\mathbf{q} \cdot \mathbf{v}_{\text{reine}} = 0{,}48 + 0{,}5 + 0{,}81 =$ **1,79**. Pour b), divise par $\lVert\mathbf{q}\rVert \times \lVert\mathbf{v}_{\text{reine}}\rVert$. Pour c), même calcul avec femme. Pour d) et e), calcule les sept cosinus et prends le plus grand, sans puis avec l'exclusion. Pour f), compare les produits scalaires bruts des mots restants. Pour g), refais d) et e) avec $\mathbf{v} / \lVert\mathbf{v}\rVert$ pour chaque mot.

</details>

### Ex B2.3 — Trois fusions BPE à la main sur un mini-corpus ✏️

<details><summary>Indice 1</summary>

Relis le pseudo-code de l'entraînement, l'exemple du corpus « chat » et l'encadré 🕰️ sur le départage (§202.3).

</details>
<details><summary>Indice 2</summary>

Trie les caractères par point de code : les lettres sans accent (U+0061 à U+007A) viennent avant les lettres accentuées (U+00E0 et au-delà). Écris chaque mot en identifiants, puis, à chaque étape : compte toutes les paires voisines (pondérées par la fréquence), prends la plus fréquente, et à égalité la plus petite paire $(a, b)$ ; remplace-la dans tous les mots par le nouvel identifiant.

</details>
<details><summary>Indice 3</summary>

a) Les caractères distincts sont é, p, r, s, t : $n_{\text{base}} =$ **5**. Pour b), range-les par point de code et donne son rang à « é ». Pour c) et d), compte les paires de « été » (× 4) et « étés » (× 3), puis départage l'égalité par la plus petite paire d'identifiants. Pour e), réécris les mots avec le token 5, puis recompte. Pour f), recompte après la deuxième fusion : trois paires sont à égalité, compare-les. Pour g), compte les tokens de chaque mot après les trois fusions, multiplie par sa fréquence, additionne. Pour h), compare le compte de la troisième fusion à 3.

</details>

### Ex B2.4 — Encoder un mot jamais vu avec des fusions déjà apprises ✏️

<details><summary>Indice 1</summary>

Relis « Encoder un mot nouveau : les fusions par rang », l'exemple de « ratons » et celui de « abc » (§202.3).

</details>
<details><summary>Indice 2</summary>

À chaque tour, liste les paires voisines du mot, garde celles qui sont dans la table, fusionne celle de plus petit rang, et recommence ; arrête-toi quand aucune paire voisine n'est dans la table. Pour c), au lieu du rang, fusionne à chaque tour la première paire connue en lisant de gauche à droite.

</details>
<details><summary>Indice 3</summary>

a) c h a n t o n s : rang 0 → c h a nt o n s ; rang 1 → ch a nt o n s ; rang 2 → ch ant o n s ; rang 3 → ch ant on s ; rang 4 → chant on s ; rang 6 → chant ons ; plus rien : **[13, 15]**. Pour b), les paires de « o n t » sont (o, n) de rang 3 et (n, t) de rang 0 : laquelle d'abord, et que reste-t-il ? Pour c), fusionne d'abord la paire la plus à gauche. Pour d), suis la méthode de a) sur « chanteront » ; après la fusion de rang 0, regarde quelles lettres restent voisines. Pour e), « z » n'est pas dans l'alphabet. Pour f), compte les octets UTF-8 de « été ». Pour g), que rend le décodage de l'identifiant 17 ?

</details>

### Ex B2.5 — One-hot fois matrice : nn.Embedding à la main, et quelles lignes reçoivent un gradient ✏️

<details><summary>Indice 1</summary>

Relis « Une couche dense sur un one-hot, c'est une lecture de table » et le gradient de l'exemple PyTorch (§202.7) ; pour le gradient, la règle de la chaîne (ch. 18).

</details>
<details><summary>Indice 2</summary>

$\mathbf{X}$ a une ligne par position et une colonne par token ; la ligne $t$ de $\mathbf{X}\mathbf{E}$ est la ligne de $\mathbf{E}$ du token lu en $t$. Le gradient de $\mathbf{w} \cdot \mathbf{e}_t$ par rapport à $\mathbf{e}_t$ est $\mathbf{w}$, et il s'ajoute à la ligne lue ; une ligne lue deux fois reçoit la somme. Un pas de SGD : $\mathbf{E} \leftarrow \mathbf{E} - \eta\,\partial L/\partial \mathbf{E}$.

</details>
<details><summary>Indice 3</summary>

a) 4 positions, 5 tokens : **`"(4, 5)"`**. Pour b), quel token est lu en $t = 1$, et quelle est sa ligne de $\mathbf{E}$ ? Pour c), calcule $\mathbf{w} \cdot \mathbf{E}[i, :]$ pour chaque token lu, puis additionne sur les quatre positions. Pour d), combien de fois le token 3 est-il lu ? Pour e), quels tokens sont lus au moins une fois ? Pour f), calcule $\mathbf{E}[3, :] - 0{,}1 \times$ (ta réponse de d). Pour g), relis la phrase sur `padding_idx`. Pour h), le weight decay agit-il sur le gradient de toute la table ou seulement sur les lignes lues ?

</details>

### Ex B2.6 — Co-occurrences et PPMI à la main (fenêtre de 1, quatre mots) ✏️

<details><summary>Indice 1</summary>

Relis « La matrice de co-occurrences », l'encadré 🧮 sur la PMI et son exemple, et la formule de $p_\alpha$ (§202.8).

</details>
<details><summary>Indice 2</summary>

Écris les 7 couples voisins de la séquence, puis remplis $\mathbf{C}$ : chaque couple ajoute 1 dans les deux sens. $p(w)$ est la somme de la ligne de $w$ divisée par $N$, $p(c)$ celle de la colonne de $c$. Pour le lissage, élève les sommes des colonnes à la puissance 0,75, puis divise par leur total ; $p_\alpha$ remplace $p(c)$ dans la PMI, pas $p(w)$.

</details>
<details><summary>Indice 3</summary>

a) Les couples (le, chat) aux positions 0-1 et 3-4, et (chat, le) aux positions 4-5 : $\mathbf{C}[0, 2] =$ **3**. Pour b), compte les couples voisins et double. Pour c), calcule $\log \frac{\#(\text{chat}, \text{le}) \times N}{\#(\text{chat}) \times \#(\text{le})}$ avec les sommes de la ligne et de la colonne. Pour d), même calcul pour (court, le). Pour e), élève chacune des quatre sommes de colonnes à la puissance 0,75, puis divise celle de « le » par leur total. Pour f), remplace $p(\text{le})$ par $p_\alpha(\text{le})$ dans la PMI de c). Pour g), refais f) dans l'autre sens, PPMI(le, chat). Pour h), refais d) avec $p_\alpha(\text{le})$.

</details>

### Ex B2.7 — Skip-gram avec échantillonnage négatif : gradients par rapport au mot et au contexte ∂

<details><summary>Indice 1</summary>

Relis la loss du skip-gram et son exemple chiffré (§202.8), la sigmoïde (ch. 17) et la règle de la chaîne (ch. 18).

</details>
<details><summary>Indice 2</summary>

Pour 1, dérive $\sigma(s) = (1 + e^{-s})^{-1}$, puis écris $1 - \sigma(s)$ avec le même dénominateur. Pour 2, chaque terme dépend d'un vecteur par un seul score : $\frac{\partial L}{\partial \mathbf{u}} = \frac{dL}{ds}\,\frac{\partial s}{\partial \mathbf{u}}$, avec $\frac{\partial (\mathbf{u}^\top \mathbf{v})}{\partial \mathbf{u}} = \mathbf{v}$ et $\frac{\partial (\mathbf{u}^\top \mathbf{v})}{\partial \mathbf{v}} = \mathbf{u}$. Pour 6, $\frac{d}{dx}\log \sigma(x) = 1 - \sigma(x)$ et $\frac{d}{dx}\log \sigma(-x) = -\sigma(x)$, puis $\frac{\sigma(x)}{1 - \sigma(x)} = e^x$.

</details>
<details><summary>Indice 3</summary>

1. $\sigma'(s) = \frac{e^{-s}}{(1 + e^{-s})^2} = \sigma(s)\,\frac{e^{-s}}{1 + e^{-s}} = \sigma(s)(1 - \sigma(s))$, et $1 - \sigma(s) = \frac{e^{-s}}{1 + e^{-s}} = \frac{1}{e^{s} + 1} = \sigma(-s)$ ; dérive ensuite $-\log \sigma(s)$ en $-\frac{\sigma'(s)}{\sigma(s)}$. Pour 2, multiplie chacune de ces dérivées par $\mathbf{v}_w$ (pour les contextes) ou par le vecteur de contexte (pour $\mathbf{v}_w$), et additionne pour $\mathbf{v}_w$. Pour 4, calcule les scores $\mathbf{u}^\top \mathbf{v}_w$, puis $\sigma$ de chacun. Pour 6, écris $\ell'(x) = -\#(w, c)(1 - \sigma(x)) + \frac{K\,\#(w)\,\#(c)}{N}\,\sigma(x)$, annule-le, et isole $e^x$.

</details>

### Ex B2.8 — Fermi : fertilité française et anglaise, pages dans 8 192 tokens, prix d'un roman traduit par API 🧮

<details><summary>Indice 1</summary>

Relis « Deux rapports », « Le même texte, un autre prix selon la langue » et « Le prix d'un token » (§202.6).

</details>
<details><summary>Indice 2</summary>

Tokens = mots × fertilité ; pages = mots / 250. Pour un appel, écris l'inéquation : instructions + passage + sa traduction ≤ 8 192, la traduction faisant (1,15 × 1,6 / 1,3) fois les tokens du passage. Le prix d'une partie = ses tokens × prix par million / $10^6$.

</details>
<details><summary>Indice 3</summary>

1. En anglais : $8\,192 / 1{,}3 \approx 6\,300$ mots, soit environ 25 pages ; refais le calcul avec 1,6 pour le français. Pour 2, multiplie 90 000 mots par 1,3, puis $90\,000 \times 1{,}15$ mots par 1,6. Pour 3, résous $300 + x + 1{,}415\,x \le 8\,192$, puis divise les tokens d'entrée par $x$ et arrondis au-dessus. Pour 4, sépare l'entrée (roman et instructions) et la sortie, au prix de chaque modèle. Pour 5, échange les rôles des deux langues. Pour 6, refais 2 et 4 avec 2,2, puis divise les tokens de sortie par 40.

</details>

### Ex B2.9 — Fermi : poids de la table d'embedding de GPT-2 et d'un LLM à 128 000 tokens 🧮

<details><summary>Indice 1</summary>

Relis « Le poids de la table » et « Les poids liés » (§202.7).

</details>
<details><summary>Indice 2</summary>

Une table a $V \times d$ paramètres. Avec des poids liés, la matrice de sortie est la même : on ne la compte qu'une fois ; sans poids liés, il y a deux matrices de forme $(V, d)$. Mémoire = paramètres × octets par nombre (4 en `float32`, 2 en `bfloat16`).

</details>
<details><summary>Indice 3</summary>

a) $50\,257 \times 768 =$ **38 597 376**. Pour b), divise par 124 439 808 et multiplie par 100. Pour c), multiplie a) par 4 octets, divise par $10^6$. Pour d), calcule $128\,256 \times 4\,096$. Pour e), double d), puis rapporte-le à 8 030 261 248. Pour f), calcule $128\,256 \times 2\,048$ et rapporte-le à 1 235 814 400. Pour g), multiplie d) par 2 octets et compare à $10^9$.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B2.10 — BPE en cinq lignes, sans jargon 🗣️

<details><summary>Indice 1</summary>

Relis « L'essentiel en 10 lignes » (points 2, 4 et 6) et l'exemple du corpus « chat » (§202.3).

</details>
<details><summary>Indice 2</summary>

Une ligne par point demandé, plus une image. Pour l'image, pense à des abréviations qu'on invente pour les groupes de lettres qu'on écrit le plus souvent, ou à des pièces de jeu de construction assemblées en blocs.

</details>
<details><summary>Indice 3</summary>

Commence par : « Un ordinateur ne lit que des nombres : il faut donc couper le texte en morceaux et donner un numéro à chacun. » Puis : les lettres seules font des textes trop longs, les mots entiers un dictionnaire sans fin ; on part des lettres et l'on colle, encore et encore, les deux morceaux voisins qui reviennent le plus souvent ; les morceaux appris viennent donc du texte lui-même ; un mot jamais vu se découpe en morceaux connus, jusqu'aux lettres ou aux octets.

</details>

### Ex B2.11 — Une taxe cachée sur certaines langues, des stéréotypes dans les vecteurs : tokenisation et biais ⚖️

<details><summary>Indice 1</summary>

Relis « Le même texte, un autre prix selon la langue » (§202.6) et « Les biais des embeddings » (§202.9) ; ton 🧮 B2.8 pour les ordres de grandeur.

</details>
<details><summary>Indice 2</summary>

Pour la langue : un surcoût de $r$ multiplie le prix, la latence et l'occupation du contexte par $r$ ; il vient du corpus du tokenizer. Pour les vecteurs : les cosinus héritent des co-occurrences du corpus, et un test contrefactuel ne change qu'un indice du genre à la fois. Pour 7, cherche les obligations d'un système à haut risque (données, documentation, contrôle humain).

</details>
<details><summary>Indice 3</summary>

1. Pour la même demande, le client tamoulophone paie environ 19 fois plus de tokens que l'anglophone, attend plus longtemps et remplit plus vite la fenêtre de contexte. Pour 2, relis la phrase « Personne n'a décidé… ». Pour 3, pense à mesurer la fertilité sur des textes réels de chaque langue, à facturer autrement qu'au token, et à suivre la qualité par langue. Pour 4 à 6, relis Bolukbasi et coll. puis Gonen et Goldberg. Pour 7, l'annexe III et le calendrier repoussé par le règlement 2026/1744.

</details>

### Ex B2.12 — Sennrich, Haddow et Birch (2016) : les sous-mots pour traduire les mots rares 📄

<details><summary>Indice 1</summary>

Lis d'abord le résumé et l'introduction, puis la section 3.2 avec l'algorithme 1 sous les yeux ; garde les tableaux pour la fin.

</details>
<details><summary>Indice 2</summary>

Pour 3, déroule l'algorithme 1 à la main : `get_stats` compte les paires dans l'ordre où il les rencontre, et `max` rend la **première** clé de compte maximal dans l'ordre du dictionnaire ; compare ce que cet ordre était en 2015 et ce qu'il est depuis Python 3.7. Pour 5, la fertilité se calcule avec le nombre de tokens du texte sans découpage. Pour 6, lis les colonnes « ens-8 » et repère le système de chaque ligne qui fait le mieux que WDict.

</details>
<details><summary>Indice 3</summary>

1. La traduction est « à vocabulaire ouvert » : on doit traduire et produire des mots jamais vus (noms, composés, mots empruntés), alors qu'un modèle neuronal garde 30 000 à 50 000 mots. Pour 2, relis le paragraphe qui suit les exemples de la section 3. Pour 3, compte les paires du dictionnaire (chacune vaut 1 ou 2) et note celles à égalité. Pour 4, cherche « joint BPE » et la note sur ISO-9. Pour 5, divise 112 millions par 100 millions, puis 550 millions par 112 millions. Pour 6, compare les lignes WDict, C2-50k, BPE-60k et BPE-J90k. Pour 7, relis les tableaux 4 et 5.

</details>

<a id="entretien"></a>

## 💼 Entretien

### B2.E1 — Pourquoi les LLM découpent-ils le texte en sous-mots plutôt qu'en mots ou en caractères ?

<details><summary>Indice 1</summary>

Relis « Trois granularités » (§202.1) et « Les chiffres et les lettres » (§202.6).

</details>
<details><summary>Indice 2</summary>

Trois temps : ce que coûtent les caractères (longueur, attention en carré), ce que coûtent les mots (vocabulaire, mots inconnus), ce que font les sous-mots ; puis leur prix : fertilité selon la langue, chiffres, lettres.

</details>
<details><summary>Indice 3</summary>

Commence par : « C'est un compromis entre la taille du vocabulaire et la longueur des séquences. » Puis : les caractères, des séquences 4 à 5 fois plus longues et une attention qui coûte le carré ; les mots, un vocabulaire sans fin et un token inconnu ; les sous-mots, appris par compression, au niveau octet, sans inconnu ; leur coût : la fertilité dépend de la langue, et les chiffres et les lettres se lisent mal.

</details>

### B2.E2 — Explique BPE, puis ce qui le distingue de WordPiece

<details><summary>Indice 1</summary>

Relis le §202.3 (entraînement, encodage par rang) et « WordPiece : un autre critère de fusion » (§202.5).

</details>
<details><summary>Indice 2</summary>

L'entraînement en une phrase, l'encodage en une phrase, le départage, puis les deux différences de WordPiece : le critère de fusion et l'encodage.

</details>
<details><summary>Indice 3</summary>

Commence par : « BPE part des caractères, ou des 256 octets, et fusionne à chaque étape la paire voisine la plus fréquente du corpus. » Puis : la liste ordonnée des fusions est le modèle ; on encode en appliquant les fusions par rang dans chaque pré-token ; WordPiece choisit la paire au meilleur score (compte de la paire divisé par le produit des comptes), marque les suites de mot par `##`, encode de façon gloutonne et rend `[UNK]` pour un mot entier.

</details>

### B2.E3 — Que se passe-t-il si l'on change de tokenizer sans ré-entraîner le modèle ?

<details><summary>Indice 1</summary>

Relis « Un modèle ne lit que des identifiants » (§202.1) et le piège « changer de tokenizer sans changer la table » (§202.7).

</details>
<details><summary>Indice 2</summary>

Le modèle a appris une ligne de table par identifiant : que deviennent ces lignes quand les identifiants changent de sens ? Puis les cas intermédiaires (ajouter quelques tokens) et les vraies options.

</details>
<details><summary>Indice 3</summary>

Commence par : « C'est donner au modèle un texte chiffré avec une autre clé : chaque identifiant pointe vers la ligne apprise pour un autre morceau de texte. » Puis : les sorties perdent leur sens, sans message d'erreur ; ajouter des tokens demande `resize_token_embeddings` et un fine-tuning ; les vraies options : garder le tokenizer, choisir un modèle entraîné avec le tokenizer voulu, ou réentraîner.

</details>

### B2.E4 — Qu'est-ce qu'un embedding, et comment mesurer la similarité de deux textes ?

<details><summary>Indice 1</summary>

Relis le §202.7 (la table), « Le cosinus » (§202.9) et « D'une suite de vecteurs à un vecteur de phrase » (§202.10).

</details>
<details><summary>Indice 2</summary>

Une définition, d'où viennent les embeddings (appris, distributionnels, contextuels), comment passer d'un texte à un vecteur, la mesure (cosinus), puis trois précautions.

</details>
<details><summary>Indice 3</summary>

Commence par : « Un embedding est un vecteur dense qui représente un objet, ici un token ou un texte, de sorte que des objets proches par le sens aient des vecteurs proches. » Puis : la table d'embedding et l'hypothèse distributionnelle ; pour un texte, un modèle d'embedding de phrases et un mean pooling avec masque ; le cosinus sur des vecteurs normalisés ; les précautions : un seul modèle, ses préfixes et sa longueur maximale, un seuil qui ne se transpose pas.

</details>

### B2.E5 — Comment construirais-tu une recherche sémantique sur des documents internes ?

<details><summary>Indice 1</summary>

Relis « Deux modèles d'embedding de phrases », « Une recherche sémantique en vingt lignes » et le piège sur la longueur maximale (§202.10).

</details>
<details><summary>Indice 2</summary>

Les étapes : découper, choisir le modèle (multilingue ici), indexer, interroger, reclasser ; puis l'évaluation (des requêtes annotées, une métrique de classement) et la production (mise à jour de l'index, droits d'accès).

</details>
<details><summary>Indice 3</summary>

Commence par : « Je commence par un jeu d'évaluation : une centaine de vraies requêtes avec les documents qui y répondent. » Puis : découper en passages plus courts que la longueur maximale ; un modèle multilingue (e5 avec ses préfixes) ; les embeddings normalisés, calculés une fois, dans une base vectorielle ; la requête, ses $k$ plus proches, puis un cross-encoder pour reclasser ; le recall@k mesuré contre une recherche par mots-clés ; les droits d'accès de chaque document.

</details>

<a id="notebook"></a>

## Exercices du notebook

Les indices des exercices du notebook (B2.13 à B2.31) seront ajoutés à la prochaine session de génération, avec les parties A à E du notebook.
