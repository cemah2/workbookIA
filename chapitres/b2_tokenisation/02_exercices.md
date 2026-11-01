# B2 · Tokenisation et embeddings — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/b2_tokenisation/06_mes_reponses.md` (créée par `python tools/start_chapter.py B2`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ B2.1 à B2.6 et de l'estimation 🧮 B2.9 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la dérivation ∂ B2.7, l'estimation 🧮 B2.8, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"B"`), une forme de tenseur entre guillemets (`"(2, 3)"`), un mot entre guillemets (`"mot"`), des octets en liste, en hexadécimal ou en décimal (`[0x4E, 0x6F]` ou `[78, 111]`), des identifiants, une paire d'identifiants ou un vecteur en liste (`[4, 0, 7]`).
> Conventions du chapitre (fiche) : l'**identifiant** d'un token est un entier de $0$ à $V - 1$ ; $d$ est la dimension des embeddings et $\mathbf{E}$, de forme $(V, d)$, la table d'embedding, dont $\mathbf{E}[i, :]$ est la ligne $i$ ; $w$ est un mot, $c$ un mot de son contexte, $\#(w, c)$ un compte ; $\log$ est le logarithme naturel. Les octets s'écrivent en hexadécimal (`C3`, ou `0xC3` en Python). Toutes les numérotations (identifiants, positions, lignes) commencent à 0, comme en Python.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### B2.Q1 — Caractères, mots ou sous-mots : taille du vocabulaire contre longueur des séquences 🧠 ⏱️ 3 min
*Fiche §202.1 · parcours R*

a) Quelle granularité donne les séquences les plus longues pour un même texte ? (A) les caractères ; (B) les mots ; (C) les sous-mots.
b) Vrai ou faux : avec un vocabulaire limité aux $V$ mots les plus fréquents, tous les mots rares reçoivent le même identifiant, et le modèle ne peut en produire aucun dans sa réponse.
c) Le coût de l'attention d'un Transformer croît comme le carré de la longueur de la séquence. Une séquence 3 fois plus longue coûte combien de fois plus d'attention ?
d) Le découpage en sous-mots d'un tokenizer moderne : (A) un linguiste choisit les racines, les préfixes et les suffixes, une langue après l'autre ; (B) on l'apprend sur un corpus, en regroupant les suites fréquentes ; (C) chaque mot est coupé en morceaux de trois lettres.
e) Vrai ou faux : au niveau des caractères, Holmes a plus de symboles distincts qu'au niveau des mots.

### B2.Q2 — Octets, points de code, caractères affichés : combien « é » compte-t-il ? 🧠 ⏱️ 3 min
*Fiche §202.2 · parcours R*

a) En NFC, « é » est le seul point de code U+00E9. Combien d'octets prend-il en UTF-8 ?
b) En NFD, « é » devient « e » suivi de l'accent aigu combinant U+0301. Combien de points de code ?
c) Et combien d'octets UTF-8, en NFD ?
d) Que vaut `len("é")` en Python, la chaîne étant écrite en NFC ?
e) Vrai ou faux : l'apostrophe typographique « ’ » (U+2019) prend 2 octets en UTF-8, comme les lettres accentuées.

### B2.Q3 — NFC ou NFD : deux chaînes identiques à l'écran, différentes en mémoire 🧠 ★★ ⏱️ 3 min
*Fiche §202.2*

a) Vrai ou faux : `s == unicodedata.normalize("NFD", s)` vaut `True` pour `s = "café"` écrite en NFC.
b) Pour retirer les accents d'un texte : (A) `casefold()` ; (B) le passer en NFC, puis supprimer les caractères dont le point de code dépasse 127 ; (C) le passer en NFD, puis supprimer les caractères combinants.
c) Vrai ou faux : NFKC est sans perte : on peut toujours retrouver le texte d'origine à partir du texte normalisé.
d) Un tokenizer BPE **sans normaliseur** est entraîné sur un corpus en NFD, puis utilisé sur des textes en NFC : (A) les fusions apprises sur les lettres décomposées ne s'appliquent pas aux lettres composées ; (B) rien ne change, car un BPE compare les caractères tels qu'ils s'affichent à l'écran, quelle que soit leur écriture en mémoire ; (C) le tokenizer lève une erreur dès le premier accent.
e) Vrai ou faux : pour une chaîne `s` écrite en NFC, `unicodedata.normalize("NFC", unicodedata.normalize("NFD", s)) == s`.

### B2.Q4 — Fusions BPE : pourquoi la liste ordonnée des fusions est le modèle 🧠 ⏱️ 3 min
*Fiche §202.3 · parcours R*

a) Pour encoder un texte nouveau exactement comme l'aurait fait le tokenizer entraîné, il faut garder : (A) la liste des tokens du vocabulaire, triée par fréquence ; (B) l'alphabet et la liste ordonnée des fusions ; (C) le corpus d'entraînement.
b) Vrai ou faux : une fusion peut réunir la fin d'un mot et le début du mot suivant si la paire est très fréquente.
c) Avant toute fusion, un corpus contient le mot « banane » 3 fois (et rien d'autre). Que vaut le compte de la paire « a n » ?
d) À chaque fusion apprise, la taille du vocabulaire : (A) augmente de 1 ; (B) augmente de 2 ; (C) diminue de 1.
e) Vrai ou faux : à égalité de compte, `BpeTrainer` choisit la paire au plus petit premier identifiant, puis, à premier identifiant égal, au plus petit second.

### B2.Q5 — BPE au niveau octet : jamais de token inconnu, vrai ou faux ? 🧠 ⏱️ 3 min
*Fiche §202.4 · parcours R*

a) Vrai ou faux : un BPE au niveau octet encode un texte en birman, sans token inconnu, même s'il n'a vu aucun texte birman à l'entraînement.
b) Un caractère chinois s'écrit sur 3 octets en UTF-8. Combien de tokens donne-t-il avec un BPE au niveau octet qui n'a appris aucune fusion sur ces octets ?
c) Vrai ou faux : avec `tokenizers`, un BPE au niveau caractère sans `unk_token` signale par une erreur les caractères qu'il n'a jamais vus.
d) Le revers du BPE au niveau octet : (A) son vocabulaire doit compter plus d'un million de tokens pour couvrir toutes les écritures ; (B) le décodage perd les accents ; (C) une lettre d'une écriture rare se paie en plusieurs tokens sans sens.
e) Vrai ou faux : un programme qui affiche le texte au fil de la génération ne peut pas toujours décoder chaque token dès qu'il le reçoit.

### B2.Q6 — Tokens spéciaux et espace initial : « Ġthe » n'est pas « the » 🧠 ⏱️ 3 min
*Fiche §202.4 · parcours R*

a) Vrai ou faux : pour GPT-2, le mot « the » a le même identifiant au début d'un texte et après une espace.
b) Dans un token de GPT-2, « Ġ » représente : (A) un token spécial de début de mot ; (B) l'octet de l'espace, remplacé par un caractère imprimable ; (C) une lettre grecque ajoutée par le normaliseur.
c) À l'encodage, un token spécial comme `<|endoftext|>` : (A) se découpe en sous-mots comme le reste du texte ; (B) est supprimé du texte avant l'encodage ; (C) reste entier, avec son propre identifiant.
d) Vrai ou faux : avec le pré-tokenizer de GPT-2 et un BPE au niveau octet, sans normaliseur, `decode(encode(s)) == s` pour toute chaîne `s`.
e) Combien de tokens spéciaux le tokenizer de GPT-2 compte-t-il ?

### B2.Q7 — WordPiece, Unigram, SentencePiece : quel critère pour quelle méthode ? 🧠 ★★ ⏱️ 3 min
*Fiche §202.5 · parcours R*

a) Quelle méthode part d'un grand vocabulaire et l'élague ? (A) BPE ; (B) WordPiece ; (C) Unigram.

Pour b) et c), deux paires candidates : (A) la paire « a b », avec count(a, b) = 20, count(a) = 40 et count(b) = 100 ; (B) la paire « c d », avec count(c, d) = 30, count(c) = 300 et count(d) = 200.

b) Laquelle WordPiece fusionne-t-il en premier, avec le score de la fiche ?
c) Laquelle BPE fusionne-t-il en premier ?
d) Vrai ou faux : SentencePiece a besoin d'un texte déjà découpé en mots par ses espaces.
e) Le préfixe `##` de WordPiece signale : (A) un morceau qui continue un mot ; (B) un token spécial, comme le début de phrase ; (C) un nombre.

### B2.Q8 — Pourquoi un LLM compte mal les lettres d'un mot et calcule mal avec les chiffres 🧠 ⏱️ 3 min
*Fiche §202.6 · parcours R*

a) Pour compter les « r » de « strawberry », un modèle de langue voit : (A) les dix lettres, une à une, chacune avec son identifiant ; (B) les octets UTF-8 de chaque lettre, puisque son tokenizer travaille au niveau octet ; (C) quelques identifiants de tokens, dont il a dû apprendre les lettres.
b) Vrai ou faux : avec GPT-2, deux nombres de 7 chiffres sont toujours découpés aux mêmes positions.
c) Avec Qwen3, qui isole chaque chiffre, combien de tokens pour « 2026 », écrit sans espace devant ?
d) Avec la règle de Llama 3 (des groupes de 1 à 3 chiffres, depuis la gauche), combien de tokens pour « 12345678 » ?
e) Vrai ou faux : découpés chiffre par chiffre, comme chez Qwen3, le « 7 » de « 1789 » et celui de « 2027 » ont le même identifiant, au prix de plus de tokens par nombre.

### B2.Q9 — nn.Embedding : une table de consultation entraînable, pas une couche magique 🧠 ⏱️ 3 min
*Fiche §202.7 · parcours R*

a) `nn.Embedding(1000, 64)` reçoit des identifiants de forme $(8, 20)$. Quelle est la forme de sa sortie ?
b) Combien de paramètres a `nn.Embedding(5000, 32)` ?
c) Vrai ou faux : `emb(ids)` donne le même résultat que `F.one_hot(ids, V).float() @ emb.weight`.
d) Dans un batch, le token 7 apparaît 3 fois. La ligne 7 de la table reçoit : (A) la somme des trois gradients ; (B) leur moyenne ; (C) le gradient de la dernière occurrence seulement.
e) Vrai ou faux : dans une table d'entrée sans poids liés, entraînée par SGD sans momentum ni weight decay, un token qui n'apparaît dans aucun batch garde sa ligne de départ, aléatoire.

### B2.Q10 — L'hypothèse distributionnelle : un mot se définit par ses voisins 🧠 ⏱️ 3 min
*Fiche §202.8 · parcours R*

a) D'après l'hypothèse distributionnelle, deux mots ont des sens proches quand : (A) ils s'écrivent avec les mêmes lettres ; (B) ils apparaissent dans des contextes semblables ; (C) l'un apparaît souvent juste à côté de l'autre, comme « boire » et « café ».
b) Vrai ou faux : une PMI négative signifie que $w$ et $c$ apparaissent ensemble moins souvent que s'ils étaient indépendants.
c) Sur $N = 1\,000$ paires (mot, contexte), $\#(w, c) = 6$, le mot $w$ a 30 paires et le contexte $c$ en a 200. Que vaut $\operatorname{PMI}(w, c)$, à 2 décimales ?
d) CBOW, l'autre version de word2vec, prédit : (A) les mots du contexte à partir du mot central ; (B) un mot tiré au hasard ; (C) le mot central à partir de la moyenne de ses contextes.
e) Vrai ou faux : le sous-échantillonnage de word2vec jette plus souvent les occurrences des mots rares.

### B2.Q11 — Statique ou contextuel : l'« avocat » du tribunal et celui du marché 🧠 ⏱️ 3 min
*Fiche §202.9, §202.10 · parcours R*

a) Avec des embeddings word2vec, « avocat » dans « l'avocat plaide » et dans « l'avocat est mûr » a : (A) le même vecteur ; (B) deux vecteurs différents ; (C) un vecteur nul.
b) À la sortie du dernier bloc d'un encodeur comme BERT, les deux « avocat » ont : (A) le même vecteur, celui de la ligne de la table ; (B) deux vecteurs différents, qui dépendent de la phrase ; (C) aucun vecteur.
c) Vrai ou faux : appris sur un corpus juridique, le vecteur statique d'« avocat » est plus proche de celui de « tribunal » que de celui de « mangue ».
d) L'embedding d'une phrase par mean pooling est : (A) le vecteur du premier token ; (B) la moyenne des lignes de la table d'embedding de ses tokens, avant tout bloc du Transformer ; (C) la moyenne des vecteurs de sortie de ses vrais tokens, sans le padding.
e) Une phrase de trois vrais tokens, complétée par un `[PAD]` : $\mathbf{h}_0 = (1, 2)$, $\mathbf{h}_1 = (3, 0)$, $\mathbf{h}_2 = (2, 4)$, $\mathbf{h}_3 = (0, 0)$ pour le `[PAD]`, masque $(1, 1, 1, 0)$. Que vaut la première composante de $\mathbf{s}$, à 1 décimale ?

### B2.Q12 — Cosinus ou distance euclidienne pour des embeddings normalisés ? 🧠 ★★ ⏱️ 3 min
*Fiche §202.9 · parcours M*

a) Deux vecteurs de norme 1 ont un cosinus de 0,8. Que vaut leur distance euclidienne **au carré**, à 1 décimale ?
b) Vrai ou faux : pour des vecteurs normalisés, classer les voisins par distance euclidienne croissante ou par cosinus décroissant donne le même ordre.
c) Que vaut la distance euclidienne (pas au carré) entre deux vecteurs de norme 1 orthogonaux, à 3 décimales ?
d) Vrai ou faux : pour des vecteurs non normalisés, le produit scalaire et le cosinus classent toujours les voisins dans le même ordre.
e) Que vaut $\cos\big((3, 4), (-6, -8)\big)$, à 1 décimale ?

<a id="rappels"></a>

## 🔁 Rappels

### B2.R1 — Ch. B1 : à quoi sert l'embedding d'image extrait d'un réseau pré-entraîné ? 🔁 ★ ⏱️ 5 min
*Ch. B1 (embeddings figés, linear probe, cosinus) · fiche §202.7, §202.10 · parcours R*

a) Vrai ou faux : l'embedding figé d'une image se calcule une fois, puis sert à entraîner plusieurs classifieurs.
b) Quelle est la dimension $d$ de l'embedding d'un ResNet-18 de torchvision, à l'entrée de sa couche `fc` ?
c) Le linear probe de B1, c'est : (A) une régression logistique entraînée sur les embeddings figés ; (B) un fine-tuning complet du réseau, avec un petit learning rate et peu d'epochs ; (C) un k plus proches voisins.
d) Vrai ou faux : pour comparer deux embeddings d'images, la distance euclidienne entre vecteurs bruts suffit, car leur norme ne porte aucune information étrangère à la classe.
e) L'embedding d'une phrase par MiniLM, dans ce chapitre, joue le même rôle que, dans B1 : (A) les pixels bruts de l'image ; (B) l'embedding $\phi(\mathbf{x})$ d'un backbone gelé ; (C) la tête de classification, entraînée sur la tâche source.

### B2.R2 — Ch. 22 : build_vocab, encode, decode et le token inconnu 🔁 ★ ⏱️ 5 min
*Ch. 22 (`build_vocab`, `encode`, `decode` de ta librairie `sequences.py`) · fiche §202.1 · parcours R*

Rappel du ch. 22 : `build_vocab` numérote les tokens par fréquence décroissante (à égalité, dans l'ordre de `sorted`), et `unk_token`, quand il est donné, prend l'identifiant 0 ; `max_size` compte le token inconnu.

a) Avec `stoi, itos = build_vocab("banana")`, que vaut `stoi["b"]` ?
b) Avec `stoi, itos = build_vocab("banana", unk_token="<unk>")`, que rend `encode("bandana", stoi, unk_token="<unk>")` ? Donne la liste.
c) Vrai ou faux : avec ce vocabulaire, `decode(encode("bandana", stoi, unk_token="<unk>"), itos)` redonne « bandana ».
d) Au niveau des mots, avec `max_size=2000`, un mot qui n'est pas dans le vocabulaire : (A) lève toujours une erreur ; (B) est découpé en sous-mots connus, comme le ferait un BPE ; (C) devient le token inconnu, si l'on en a déclaré un.
e) Avec `stoi, itos = build_vocab(["le", "chat", "le", "rat"], max_size=2, unk_token="<unk>")`, `encode(["chat"], stoi, unk_token="<unk>")` rend un seul identifiant : lequel ?

### B2.R3 — Ch. 6 : entropie, Huffman et ce que l'on compresse vraiment 🔁 ★ ⏱️ 5 min
*Ch. 6 (entropie, code de Huffman, taux de compression) · fiche §202.3, §202.6 · parcours R*

a) Une source émet quatre symboles avec les probabilités $(1/2, 1/4, 1/8, 1/8)$. Que vaut son entropie en bits, à 2 décimales ?
b) Combien de bits a le mot de code du symbole le plus probable, dans un code de Huffman de cette source ?
c) Vrai ou faux : Huffman donne un code court aux symboles fréquents, BPE donne un seul token aux suites de symboles fréquentes.
d) Vrai ou faux : un code sans perte peut descendre, en moyenne, sous l'entropie de la source (en bits par symbole).
e) Un texte de 1 000 octets, écrit avec un code de 8 bits par octet, est compressé en 4 000 bits. Que vaut le taux de compression du ch. 6 (bits envoyés divisés par les bits du code de longueur fixe), à 1 décimale ?

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex B2.1 — « Noël à Paris » en UTF-8 : octets, points de code, longueur ✏️ ★ ⏱️ 10 min
**Objectif :** compter les points de code et les octets d'un texte accentué, et écrire un caractère en UTF-8 à partir de son point de code.
**Prérequis :** fiche §202.2 · 0A (chaînes, `str` et `bytes`) · **Parcours :** R, M

Le texte `s = "Noël à Paris"` est écrit en NFC, comme le donne un clavier français : « ë » est U+00EB et « à » est U+00E0. Rappel : de U+0080 à U+07FF, UTF-8 écrit un point de code sur deux octets, `110xxxxx 10xxxxxx`, les 5 bits de poids fort dans le premier, les 6 de poids faible dans le second.

a) Le nombre de points de code de `s` (ce que rend `len(s)`).
b) Le nombre d'octets de `s.encode("utf-8")`.
c) Les deux octets UTF-8 de « ë », dans l'ordre.
d) La valeur de `s.encode("utf-8")[3]`, en décimal.
e) Le nombre d'octets de `unicodedata.normalize("NFD", s).encode("utf-8")`, sachant qu'en NFD « ë » devient « e » + U+0308 et « à » devient « a » + U+0300.
f) Les deux octets UTF-8 de « œ » (U+0153), dans l'ordre.
g) Vrai ou faux : `s.encode("utf-8")[:3].decode("utf-8")`, sans l'argument `errors`, rend « No » suivi du caractère de remplacement « � ».

Dans ta copie : combien de points de code compte `unicodedata.normalize("NFD", s)` ? Pourquoi deux chaînes qui s'affichent « Noël à Paris » peuvent-elles avoir des longueurs différentes, et que faut-il faire avant de les comparer ou de les tokeniser ?

### Ex B2.2 — Analogie à la main : roi − homme + femme sur des vecteurs de dimension 3 ✏️ ★ ⏱️ 10 min
**Objectif :** résoudre une analogie par 3CosAdd, et voir ce que changent l'exclusion des mots de la question et la division par les normes.
**Prérequis :** fiche §202.9 · 0B (produit scalaire, norme, similarité cosinus) · **Parcours :** M

Sept mots ont pour embeddings des vecteurs jouets de dimension 3, dont les axes seraient « royauté », « masculin (+) ou féminin (−) » et « âge adulte » :

| Mot | Embedding | Mot | Embedding |
|---|---|---|---|
| roi | $(0{,}6 ;\ 1 ;\ 0{,}9)$ | prince | $(0{,}7 ;\ 0{,}8 ;\ 0{,}2)$ |
| reine | $(0{,}8 ;\ -0{,}5 ;\ 0{,}9)$ | princesse | $(0{,}7 ;\ -0{,}6 ;\ 0{,}2)$ |
| homme | $(0{,}1 ;\ 1 ;\ 0{,}9)$ | trône | $(3 ;\ 0 ;\ 0{,}5)$ |
| femme | $(0{,}1 ;\ -1 ;\ 0{,}9)$ | | |

On cherche « homme est à roi ce que femme est à ? » : la requête est $\mathbf{q} = \mathbf{v}_{\text{roi}} - \mathbf{v}_{\text{homme}} + \mathbf{v}_{\text{femme}}$.

a) Le produit scalaire $\mathbf{q} \cdot \mathbf{v}_{\text{reine}}$, à 2 décimales.
b) $\cos(\mathbf{q}, \mathbf{v}_{\text{reine}})$, à 3 décimales.
c) $\cos(\mathbf{q}, \mathbf{v}_{\text{femme}})$, à 3 décimales.
d) Le mot de plus grand cosinus avec $\mathbf{q}$ parmi les sept, **sans** rien exclure.
e) La réponse de 3CosAdd : le mot de plus grand cosinus avec $\mathbf{q}$, roi, homme et femme exclus.
f) Le mot de plus grand **produit scalaire** avec $\mathbf{q}$ (sans diviser par les normes), roi, homme et femme exclus.
g) Vrai ou faux : si l'on ramène d'abord les sept vecteurs à la norme 1, puis que l'on recalcule $\mathbf{q}$ et les cosinus, la réponse de e) ne change pas.

Dans ta copie : pourquoi le mot $c$ de la question (ici, femme) est-il si souvent le plus proche de $\mathbf{b} - \mathbf{a} + \mathbf{c}$ ? Que devient la requête pour « femme est à homme ce que roi est à ? » (calcule-la), et pourquoi une analogie spectaculaire ne prouve-t-elle pas grand-chose sans le détail des exclusions ?

### Ex B2.3 — Trois fusions BPE à la main sur un mini-corpus ✏️ ★★ ⏱️ 20 min
**Objectif :** dérouler trois fusions de l'entraînement BPE au niveau caractère, en numérotant l'alphabet et en départageant les égalités comme `tokenizers`.
**Prérequis :** Ex B2.1 · fiche §202.3 (pseudo-code, encadré 🕰️ sur le départage) · **Parcours :** R, M

Le corpus, déjà pré-tokenisé, contient quatre mots distincts, avec leurs fréquences : « été » × 4, « étés » × 3, « pré » × 2, « prés » × 1. On suit la fiche (et `BpeTrainer`) :
- l'**alphabet** est l'ensemble des caractères du corpus, **trié par point de code** (comme `sorted` en Python), numéroté à partir de 0 ;
- à chaque étape, on fusionne la paire voisine la plus fréquente (comptes pondérés par la fréquence des mots, sans franchir la frontière d'un mot) ; **à égalité, la plus petite paire d'identifiants** $(a, b)$ l'emporte : le plus petit $a$, puis le plus petit $b$ ;
- le token fusionné reçoit l'identifiant suivant : $n_{\text{base}}$ pour la première fusion, $n_{\text{base}} + 1$ pour la deuxième, etc.

a) La taille $n_{\text{base}}$ de l'alphabet.
b) L'identifiant de « é ».
c) La première fusion, en paire d'identifiants $(a, b)$.
d) Le compte de cette première paire.
e) La deuxième fusion, en paire d'identifiants.
f) La troisième fusion, en paire d'identifiants.
g) Le nombre total de tokens du corpus après les trois fusions, en comptant chaque mot autant de fois que sa fréquence.
h) Vrai ou faux : avec `min_frequency=3` (l'entraînement s'arrête quand la meilleure paire apparaît moins de 3 fois), la troisième fusion a quand même lieu.

Dans ta copie : écris les quatre mots découpés après chaque fusion, et la chaîne de chaque nouveau token. Quelle aurait été la première fusion si l'on avait numéroté l'alphabet dans l'ordre alphabétique du français (« é » rangé avec « e ») ? Où `tokenizers` placerait-il un token spécial, et ta librairie ?

### Ex B2.4 — Encoder un mot jamais vu avec des fusions déjà apprises ✏️ ★ ⏱️ 10 min
**Objectif :** encoder des mots nouveaux en appliquant des fusions apprises par rang, et voir ce que donnent un ordre de lecture fautif et un caractère inconnu.
**Prérequis :** Ex B2.3 · fiche §202.3 (encoder un mot nouveau), §202.4 · **Parcours :** M

Un BPE au niveau caractère, entraîné sur des formes du verbe « chanter », a l'alphabet a, c, e, h, n, o, r, s, t (identifiants 0 à 8) et a appris huit fusions :

| Rang | Paire | Nouveau token | Identifiant |
|---|---|---|---|
| 0 | n + t | nt | 9 |
| 1 | c + h | ch | 10 |
| 2 | a + nt | ant | 11 |
| 3 | o + n | on | 12 |
| 4 | ch + ant | chant | 13 |
| 5 | e + r | er | 14 |
| 6 | on + s | ons | 15 |
| 7 | chant + er | chanter | 16 |

Le token spécial `<unk>` vient après les fusions, avec l'identifiant 17 ; chaque caractère hors de l'alphabet devient `<unk>`. Pour encoder un mot, on part de ses caractères et, tant qu'une paire voisine est une fusion de la table, on fusionne celle de **plus petit rang**. Au décodage, on recolle les chaînes des tokens, et `<unk>` redonne la chaîne « <unk> ». Donne les identifiants.

a) L'encodage de « chantons ».
b) L'encodage de « ont ».
c) L'encodage de « ont » si l'on fusionnait, par erreur, la paire connue **la plus à gauche** au lieu de celle de plus petit rang.
d) L'encodage de « chanteront ».
e) L'encodage de « chantez ».
f) Un BPE au niveau octet qui n'a appris aucune fusion sur les octets de « é » : combien de tokens pour « été » ?
g) Vrai ou faux : avec le tokenizer de la table, décoder l'encodage de « chantez » redonne « chantez ».

Dans ta copie : pourquoi l'ordre de lecture fautif de c) produit-il un découpage que le modèle n'a jamais vu à l'entraînement ? Encode aussi « enchanter ».

### Ex B2.5 — One-hot fois matrice : nn.Embedding à la main, et quelles lignes reçoivent un gradient ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer une lecture de table d'embedding comme un produit one-hot × matrice, puis le gradient de la table et un pas de SGD.
**Prérequis :** fiche §202.7 · 0B (produit matriciel) · ch. 18 (règle de la chaîne) · **Parcours :** R, M

Une table d'embedding de $V = 5$ tokens en dimension $d = 3$ :

$$\mathbf{E} = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \\ 2 & -1 & 0 \\ -1 & 3 & 1 \\ 0 & 2 & 2 \end{pmatrix} \quad \text{(ligne } i \text{ = embedding du token } i\text{)}.$$

Une séquence de 4 tokens a les identifiants $(3, 0, 3, 1)$, aux positions $t = 0, 1, 2, 3$. On note $\mathbf{X}$ la matrice des one-hot de la séquence (une ligne par position) et $\mathbf{e}_t$ la ligne $t$ de $\mathbf{X}\mathbf{E}$. La loss est $L = \sum_{t=0}^{3} \mathbf{w} \cdot \mathbf{e}_t$, avec $\mathbf{w} = (1 ;\ -2 ;\ 0{,}5)$.

a) La forme de $\mathbf{X}$.
b) La ligne de $\mathbf{X}\mathbf{E}$ à la position $t = 1$.
c) La valeur de $L$, à 1 décimale.
d) La ligne 3 de $\partial L / \partial \mathbf{E}$.
e) Le nombre de lignes de $\partial L / \partial \mathbf{E}$ qui ne sont pas nulles.
f) La ligne 3 de $\mathbf{E}$ après un pas de SGD de learning rate $\eta = 0{,}1$, à 2 décimales.
g) Vrai ou faux : avec `nn.Embedding(5, 3, padding_idx=0)`, la ligne 0 reçoit un gradient non nul pour cette séquence.
h) Vrai ou faux : avec `torch.optim.SGD(..., weight_decay=0.01)`, la ligne 4, qu'aucune position ne lit, change au premier pas.

Dans ta copie : écris $\mathbf{X}$ en entier. Pourquoi `nn.Embedding` ne construit-il jamais cette matrice ? Pour une table de 50 000 tokens et un batch de 512 tokens, combien de lignes au plus reçoivent un gradient ?

### Ex B2.6 — Co-occurrences et PPMI à la main (fenêtre de 1, quatre mots) ✏️ ★★ ⏱️ 25 min
**Objectif :** construire une matrice de co-occurrences avec une fenêtre de 1, puis calculer des PMI et une PPMI lissée.
**Prérequis :** fiche §202.8 (encadré 🧮 sur la PMI, lissage des contextes) · ch. 6 (logarithme, probabilités) · **Parcours :** M

La séquence de 8 tokens « le chat court le chat le rat court » a quatre mots distincts, numérotés dans l'ordre alphabétique : chat 0, court 1, le 2, rat 3. On compte les co-occurrences avec une fenêtre de 1, symétrique et uniforme : chaque couple de tokens voisins $(x, y)$ ajoute 1 à $\#(x, y)$ **et** 1 à $\#(y, x)$. On note $\mathbf{C}$ la matrice $4 \times 4$ des comptes (lignes : mots, colonnes : contextes), $N$ la somme de ses cases, et $\log$ le logarithme naturel.

a) $\#(\text{chat}, \text{le})$, la case $\mathbf{C}[0, 2]$.
b) $N$.
c) $\operatorname{PMI}(\text{chat}, \text{le})$, à 3 décimales.
d) $\operatorname{PMI}(\text{court}, \text{le})$, à 3 décimales.
e) La probabilité lissée $p_\alpha(\text{le})$ avec $\alpha = 0{,}75$, à 3 décimales.
f) $\operatorname{PPMI}(\text{chat}, \text{le})$ avec ce lissage des contextes, à 3 décimales.
g) Vrai ou faux : avec $\alpha = 0{,}75$, la matrice PPMI reste symétrique.
h) Vrai ou faux : avec $\alpha = 0{,}75$, $\operatorname{PPMI}(\text{court}, \text{le})$ devient strictement positive.

Dans ta copie : écris $\mathbf{C}$ en entier, et vérifie que $N = 2(n - 1)$ pour $n = 8$. Que vaut $\operatorname{PPMI}(\text{court}, \text{le})$ sans lissage, bien que les deux mots soient voisins ? Pourquoi « le », voisin de tous les mots, a-t-il des PMI faibles ?

### Ex B2.7 — Skip-gram avec échantillonnage négatif : gradients par rapport au mot et au contexte ∂ ★★★ ⏱️ 35 min
**Objectif :** dériver les gradients de la loss du skip-gram à échantillonnage négatif, les interpréter, puis montrer qu'à l'optimum le score d'une paire est sa PMI décalée de $\log K$.
**Prérequis :** Ex B2.5 · fiche §202.8 (loss du skip-gram, Levy et Goldberg) · ch. 17 (sigmoïde) · ch. 18 (règle de la chaîne) · **Parcours :** M

Pour une paire (mot $w$, vrai contexte $c$) et $K$ négatifs $n_1, \dots, n_K$, la loss est
$$L = -\log \sigma(s_c) - \sum_{k=1}^{K} \log \sigma(-s_k), \qquad s_c = \mathbf{u}_c^\top \mathbf{v}_w, \quad s_k = \mathbf{u}_{n_k}^\top \mathbf{v}_w,$$
avec $\sigma(s) = 1/(1 + e^{-s})$. Réponds dans ta copie.

1. Montre que $\sigma'(s) = \sigma(s)\,(1 - \sigma(s))$ et que $\sigma(-s) = 1 - \sigma(s)$. En déduire $\frac{d}{ds}\big[-\log \sigma(s)\big] = \sigma(s) - 1$ et $\frac{d}{ds}\big[-\log \sigma(-s)\big] = \sigma(s)$.
2. Calcule $\partial L / \partial \mathbf{u}_c$, $\partial L / \partial \mathbf{u}_{n_k}$ et $\partial L / \partial \mathbf{v}_w$ (pense au gradient de $\mathbf{u}^\top \mathbf{v}$ par rapport à chacun des deux vecteurs).
3. Interprète : que vaut le gradient de $\mathbf{u}_c$ quand $s_c$ est déjà très grand ? Celui d'un négatif de score très négatif ? Compte, sur tes gradients de la question 2, les lignes des tables $\mathbf{E}$ et $\mathbf{E}'$ qu'un pas de SGD modifie.
4. Application : $d = 2$, $K = 2$, $\mathbf{v}_w = (2 ;\ -1)$, $\mathbf{u}_c = (1 ;\ 1)$, $\mathbf{u}_{n_1} = (0{,}5 ;\ 0{,}5)$, $\mathbf{u}_{n_2} = (-1 ;\ 2)$. Calcule les trois scores, $L$ à 4 décimales, puis les quatre gradients. Après un pas de SGD de learning rate $\eta = 0{,}5$ sur $\mathbf{u}_{n_1}$ seul, que devient son score $s_1$ ?
5. Un négatif tiré au hasard tombe sur le vrai contexte : $n_1 = c$. Écris le gradient total reçu par $\mathbf{u}_c$. Pour quelle valeur de $s_c$ s'annule-t-il ? Pourquoi word2vec accepte-t-il ces collisions ?
6. L'optimum (Levy et Goldberg, 2014). Sur tout le corpus, la paire $(w, c)$ apparaît $\#(w, c)$ fois comme vraie paire, et, si les négatifs sont tirés selon la loi des contextes $p(c) = \#(c)/N$, $K\,\#(w)\,\#(c)/N$ fois en moyenne comme paire négative. En supposant que le score $x = \mathbf{u}_c^\top \mathbf{v}_w$ de chaque paire peut être choisi librement, la loss totale de la paire est $\ell(x) = -\#(w, c)\log \sigma(x) - \frac{K\,\#(w)\,\#(c)}{N}\log \sigma(-x)$. Annule $\ell'(x)$, puis montre que $x^* = \operatorname{PMI}(w, c) - \log K$.
7. Le lien avec la régression logistique (ch. 13) : pour un mot $w$ fixé, quelles sont les « entrées », les « labels » et les « poids » de la régression que résout chaque terme de $L$ ?

### Ex B2.8 — Fermi : fertilité française et anglaise, pages dans 8 192 tokens, prix d'un roman traduit par API 🧮 ★★ ⏱️ 15 min
**Objectif :** chiffrer en ordres de grandeur ce que coûte la traduction d'un roman par une API facturée au token, et ce que la langue change à la fenêtre de contexte et à la facture.
**Prérequis :** Ex B2.1 · fiche §202.6 (fertilité, contexte, prix) · **Parcours :** R, M

Hypothèses de calcul (des ordres de grandeur, pas des mesures) : un roman anglais de 90 000 mots ; une page de roman fait 250 mots ; la traduction française compte 15 % de mots de plus que l'original ; le tokenizer du modèle a une fertilité de 1,3 token par mot en anglais et de 1,6 en français ; la fenêtre de contexte du modèle est de 8 192 tokens, entrée et sortie comprises ; chaque appel envoie 300 tokens d'instructions en plus du passage à traduire. Prix (pages officielles relevées le 1ᵉʳ novembre 2026, fiche §202.6) : un modèle économique à 0,10 $ par million de tokens d'entrée et 0,40 $ par million de tokens de sortie ; un modèle haut de gamme à 10 $ et 50 $. Réponds dans ta copie.

1. Combien de pages de roman tiennent dans 8 192 tokens, en anglais, puis en français ?
2. Combien de tokens d'entrée (le roman anglais) et de sortie (la traduction française) demande la traduction complète, sans les instructions ?
3. Un appel doit contenir les instructions, le passage anglais et sa traduction. Quelle longueur de passage anglais, en tokens puis en pages, tient au plus dans un appel ? Combien d'appels faut-il, et combien de tokens d'instructions ajoutent-ils en tout ?
4. Le prix de la traduction, instructions comprises, avec chacun des deux modèles. Quelle part vient de la sortie ?
5. Le même roman, écrit en français (103 500 mots), traduit vers l'anglais (90 000 mots) avec le modèle haut de gamme : combien coûte-t-il ? Pourquoi le sens de la traduction change-t-il la facture ?
6. Avec un tokenizer plus ancien, appris surtout sur de l'anglais, la fertilité française passerait à 2,2. De combien augmentent les tokens de sortie et le prix du modèle haut de gamme (recalcule le nombre d'appels) ? Et le temps de génération, à 40 tokens par seconde ?
7. Cite trois coûts ou effets que ton estimation ne compte pas.

### Ex B2.9 — Fermi : poids de la table d'embedding de GPT-2 et d'un LLM à 128 000 tokens 🧮 ★ ⏱️ 10 min
**Objectif :** compter les paramètres et la mémoire d'une table d'embedding, et voir quelle part du modèle elle représente selon sa taille et le partage des poids.
**Prérequis :** Ex B2.5 · fiche §202.7 (poids de la table, poids liés) · **Parcours :** M

Données (lues dans les fichiers `config.json` et les en-têtes des poids sur le Hugging Face Hub, le 1ᵉʳ novembre 2026) :
- **GPT-2** (*small*) : $V = 50\,257$ tokens, $d = 768$, poids liés ; 124 439 808 paramètres en tout ;
- **Llama 3.1 8B** : $V = 128\,256$ tokens, $d = 4\,096$, poids **non** liés (une table d'entrée et une matrice de sortie de même forme) ; 8 030 261 248 paramètres ;
- **Llama 3.2 1B** : $V = 128\,256$ tokens, $d = 2\,048$, poids liés ; 1 235 814 400 paramètres.

a) Le nombre de paramètres de la table d'embedding de GPT-2.
b) Sa part des paramètres de GPT-2, en pourcentage, à 1 décimale.
c) Sa taille en `float32`, en Mo ($10^6$ octets), à 1 décimale.
d) Le nombre de paramètres de la table d'entrée de Llama 3.1 8B.
e) La part de la table d'entrée **et** de la matrice de sortie dans les paramètres de Llama 3.1 8B, en pourcentage, à 1 décimale.
f) La part de la table de Llama 3.2 1B dans ses paramètres, en pourcentage arrondi à l'entier.
g) Vrai ou faux : en `bfloat16` (2 octets par nombre), la table d'entrée de Llama 3.1 8B pèse plus de 1 Go ($10^9$ octets).

Dans ta copie : pourquoi la part de la table baisse-t-elle quand le modèle grandit à vocabulaire égal ? Un texte français de 10 000 tokens lit au plus 10 000 lignes distinctes de la table d'entrée de Llama 3.1 8B (poids non liés) : quelle part au plus de ses 128 256 lignes reçoit un gradient sur ce texte ? Et pour Llama 3.2 1B, dont les poids sont liés ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B2.10 — BPE en cinq lignes, sans jargon 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant comment un tokenizer BPE apprend ses morceaux de mots, et pourquoi il ne rencontre jamais de mot inconnu.
**Prérequis :** fiche §202.3, §202.4 · **Parcours :** R

Explique en cinq lignes au plus, à quelqu'un qui n'a jamais programmé, comment un tokenizer BPE apprend à découper un texte. Ton explication doit faire apparaître (1) pourquoi on ne prend ni les lettres ni les mots entiers, (2) ce que fait une fusion, (3) d'où viennent les morceaux appris, (4) ce qui arrive à un mot jamais vu. Les mots « token », « paire », « fusion » et « vocabulaire » sont permis s'ils sont expliqués ; une image de la vie courante est la bienvenue.

### Ex B2.11 — Une taxe cachée sur certaines langues, des stéréotypes dans les vecteurs : tokenisation et biais ⚖️ ★★ ⏱️ 25 min
**Objectif :** repérer deux biais de la chaîne texte → tokens → vecteurs, l'un dans le coût selon la langue, l'autre dans les embeddings, et proposer comment les mesurer et les réduire.
**Prérequis :** Ex B2.8 · fiche §202.6 (surcoût de tokenisation), §202.9 (biais des embeddings) · **Parcours :** R

Une entreprise propose, dans dix pays, un assistant de support facturé à ses clients **au token**, avec un tokenizer appris surtout sur de l'anglais. Son équipe RH veut aussi classer les CV reçus par le cosinus entre leur embedding et celui de l'offre d'emploi. Réponds dans ta copie.

1. **La langue.** Avec les surcoûts de GPT-2 de la figure `fertilite.png` (environ 2 pour le français, 9 pour le hindi, 19 pour le tamoul), que paie un client tamoulophone pour la même demande qu'un client anglophone ? Quels autres coûts, en dehors de la facture, supporte-t-il ?
2. Qui a décidé de ce surcoût ? D'où vient-il, et que recommandent Petrov et coll. (2023) ?
3. Propose trois mesures pour l'entreprise : sur le choix du tokenizer, sur la tarification, sur l'évaluation (quoi mesurer, sur quels textes).
4. **Les vecteurs.** Comment des stéréotypes de genre peuvent-ils entrer dans le classement des CV par cosinus ? Appuie-toi sur Bolukbasi et coll. (2016).
5. Décris un test simple pour mesurer ce biais avant le déploiement (des CV identiques où seuls changent des indices du genre, prénom compris), et la métrique que tu suivrais.
6. La correction de Bolukbasi et coll. suffit-elle ? Que montrent Gonen et Goldberg (2019) ?
7. Dans l'Union européenne, un système d'IA qui trie des candidatures est classé à haut risque (règlement 2024/1689, annexe III, point 4). Qu'est-ce que cela impose à l'entreprise, et à partir de quand ? Rédige ensuite les quatre lignes « Biais connus et limites » de la fiche modèle du système de tri.

### Ex B2.12 — Sennrich, Haddow et Birch (2016) : les sous-mots pour traduire les mots rares 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a fait de BPE un tokenizer : le problème des mots rares, l'algorithme, et ce que mesurent ses tableaux.
**Prérequis :** Ex B2.3 · fiche §202.3 · **Parcours :** M

Lis R. Sennrich, B. Haddow et A. Birch, « Neural Machine Translation of Rare Words with Subword Units » ([arXiv:1508.07909](https://arxiv.org/abs/1508.07909), ACL 2016) : le résumé, les sections 1, 3 et 4.1, le tableau 1, puis les tableaux 2 et 3 et la section 5.2. Réponds dans ta copie.

1. **Le problème.** Pourquoi la traduction est-elle un problème « à vocabulaire ouvert » ? Quelle taille de vocabulaire ont les modèles neuronaux de l'époque, et que fait la référence WDict des mots qu'elle ne connaît pas ? Pourquoi cela marche-t-il mal entre l'anglais et le russe ?
2. Les auteurs analysent 100 tokens rares de leur corpus allemand (section 3) : quelles catégories trouvent-ils, combien de chacune, et quelle hypothèse en tirent-ils ?
3. **L'algorithme** (section 3.2, algorithme 1, figure 1). Quel symbole ajoutent-ils à la fin de chaque mot, et à quoi sert-il ? Quel est le seul hyperparamètre ? Rejoue l'algorithme 1 sur le dictionnaire de la figure 1 (« low », « lowest », « newer », « wider », une fois chacun) : quelle règle de départage suit `max(pairs, key=pairs.get)`, et obtiens-tu les fusions de la figure ? Est-ce la règle de `BpeTrainer` ?
4. BPE indépendant ou joint : quelle différence, quel avantage pour chacun ? Pourquoi les auteurs translittèrent-ils le russe en alphabet latin pour apprendre le BPE joint ?
5. **Le tableau 1.** Compare le nombre de tokens, de types (formes distinctes) et de tokens inconnus du corpus allemand sans découpage, en caractères et avec BPE. Calcule la fertilité du BPE (tokens par mot) et le rapport entre la longueur en caractères et la longueur en BPE. Combien de fusions BPE les auteurs apprennent-ils ?
6. **Les tableaux 2 et 3.** Les gains de 1,1 et 1,3 BLEU du résumé : quel système les obtient, dans chaque sens de traduction, et face à quoi ? Que montre le F1 sur les mots rares ? Pourquoi, de l'anglais vers l'allemand, la référence reste-t-elle bonne sur les mots absents de l'entraînement (OOV) ?
7. **Section 5.2 et aujourd'hui.** Que montre l'exemple de « rakfisk », et comment le BPE joint le corrige-t-il ? Donne trois différences entre ce BPE de 2016 et celui de GPT-2 ou de Llama 3 (fiche §202.4 et §202.5).

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### B2.E1 — Pourquoi les LLM découpent-ils le texte en sous-mots plutôt qu'en mots ou en caractères ? 💼 ★★ ⏱️ 10 min
*Fiche §202.1, §202.3 · parcours R*

« Les LLM ne lisent ni des mots ni des lettres, mais des sous-mots. Pourquoi ce choix, et qu'est-ce qu'il coûte ? »

### B2.E2 — Explique BPE, puis ce qui le distingue de WordPiece 💼 ★★ ⏱️ 10 min
*Fiche §202.3, §202.5 · parcours R*

« Expliquez-moi comment BPE construit son vocabulaire et encode un mot nouveau. En quoi WordPiece est-il différent ? »

### B2.E3 — Que se passe-t-il si l'on change de tokenizer sans ré-entraîner le modèle ? 💼 ★★ ⏱️ 10 min
*Fiche §202.4, §202.7 · parcours R*

« Un collègue veut remplacer le tokenizer d'un modèle déjà entraîné par un tokenizer plus récent, plus économe en français. Qu'en pensez-vous ? »

### B2.E4 — Qu'est-ce qu'un embedding, et comment mesurer la similarité de deux textes ? 💼 ★★ ⏱️ 10 min
*Fiche §202.9, §202.10 · parcours R*

« Qu'est-ce qu'un embedding ? Comment mesureriez-vous la similarité de deux textes, et à quoi faut-il faire attention ? »

### B2.E5 — Comment construirais-tu une recherche sémantique sur des documents internes ? 💼 ★★ ⏱️ 10 min
*Fiche §202.10 · parcours R*

« Nous avons 200 000 documents internes, en français et en anglais. Comment construiriez-vous une recherche par le sens, et comment sauriez-vous qu'elle marche ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`, en deux phases : **tokenisation** (B2.13 à B2.21 : tokenizers réels, ton BPE, expériences), puis **embeddings** (B2.22 à B2.29 : co-occurrences, word2vec, phrases), suivies de la livraison (B2.30) et du défi (B2.31).

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| B2.13 | Tokenizers réels : GPT-2, BERT et un tokenizer multilingue sur Holmes et Verne | 📦 | ★★ | 20 |
| B2.14 | Prédire : quel texte coûte le plus de tokens, Holmes, Verne, des nombres ou des emoji ? | 🔮 | ★ | 10 |
| B2.15 | get_pair_counts et merge_pair | 🔨 | ★★ | 20 |
| B2.16 | BPETokenizer.fit : apprendre les fusions, au niveau caractère puis octet | 🔨 | ★★★ | 50 |
| B2.17 | encode et decode : fusions appliquées par rang, aller-retour sans perte | 🔨 | ★★★ | 40 |
| B2.18 | Le tokenizer qui abîme les accents : trois bugs d'Unicode et de fusions | 🐛 | ★★ | 25 |
| B2.19 | Ton BPE contre la bibliothèque tokenizers : mêmes fusions, autre vitesse | 📦 | ★★ | 25 |
| B2.20 | Taille du vocabulaire : compression et entropie par token, de 300 à 8 000 tokens | 🔬 | ★★★ | 40 |
| B2.21 | Tokenizer entraîné sur Holmes, appliqué à Verne (et l'inverse) : le coût de la langue | 🔬 | ★★ | 25 |
| B2.22 | cooccurrence_matrix et ppmi | 🔨 | ★★ | 30 |
| B2.23 | cosine_similarity_matrix et most_similar | 🔨 | ★★ | 20 |
| B2.24 | Embeddings PPMI + SVD sur Holmes : les voisins de « Holmes », « Watson » et « letter » | 📦 | ★★ | 30 |
| B2.25 | word2vec (skip-gram, échantillonnage négatif) en PyTorch sur Verne | 📦 | ★★★ | 60 |
| B2.26 | Prédire les voisins de « Fogg » et de « Passepartout » avant de les calculer | 🔮 | ★ | 10 |
| B2.27 | Lire une carte PCA/UMAP d'embeddings : groupes, axes et effet de la fréquence | 📈 | ★★ | 20 |
| B2.28 | Embeddings de phrases (MiniLM, mean pooling) et recherche sémantique dans Holmes | 📦 | ★★ | 30 |
| B2.29 | Statique contre contextuel : le même mot dans deux phrases | 🔬 | ★★ | 25 |
| B2.30 | Sauvegarder un tokenizer en JSON et écrire son test de non-régression | 🛠️ | ★★ | 20 |
| B2.31 | Défi : la meilleure compression sans perte de Verne avec 2 000 tokens, en moins de 2 min de CPU | 🏆 | ★★★★ | 90 |

Le notebook télécharge les fichiers de quelques tokenizers (quelques Mo chacun, sans les poids), puis all-MiniLM-L6-v2 (≈ 90 Mo) pour B2.28 et B2.29 ; multilingual-e5-small (≈ 470 Mo) est en option. Tout tourne sur CPU ; ton BPE, en Python pur, reste dans le budget du mode rapide.
