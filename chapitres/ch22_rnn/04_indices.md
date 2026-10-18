# 22 · Réseaux récurrents (RNN, LSTM, GRU) — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 22.Q1 — Pourquoi un réseau dense ignore l'ordre

<details><summary>Indice 1</summary>

Relis le §22.2 de la fiche, paragraphe « Le défaut de la couche dense ».

</details>
<details><summary>Indice 2</summary>

Dans une couche dense, chaque entrée a son propre poids. Demande-toi si le réseau peut « défaire » une permutation qui est **toujours la même**, puis une permutation qui change d'un exemple à l'autre. Pour e), écris les positions couvertes par deux fenêtres consécutives.

</details>
<details><summary>Indice 3</summary>

a) Vrai : il suffit que le neurone qui recevait la 1ʳᵉ valeur avec le poids $w_1$ la reçoive maintenant sur l'entrée 2, avec ce même poids. Pour b), cherche la propriété qui rend cette permutation des poids possible ; pour c), demande-toi si une même entrée porte encore toujours la même position ; pour d), cherche ce que les trois exemples ont en commun ; pour e), compare les positions 0 à 3 et 1 à 4.

</details>

### 22.Q2 — État ou poids : qu'est-ce qui change en production ?

<details><summary>Indice 1</summary>

Relis les §22.3 et §22.3.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Les poids ne bougent que pendant l'entraînement, à chaque pas de l'optimiseur. L'état, lui, est recalculé à chaque entrée. Pour c) et d), demande-toi de quoi dépend la sortie : des poids, de l'entrée, **et** de l'état de départ.

</details>
<details><summary>Indice 3</summary>

a) Faux : sans optimiseur, aucun poids ne change. Pour b), demande-toi ce que la cellule calcule à chaque entrée ; pour c), compare les deux calculs (mêmes poids, mêmes entrées, même état de départ ?) ; pour d), l'état de départ est-il le même dans les deux cas ? Pour e), cherche ce qui transporte l'information d'une entrée à la suivante.

</details>

### 22.Q3 — Délai, déroulement et pas de temps

<details><summary>Indice 1</summary>

Relis le §22.4 de la fiche, paragraphe « Enroulé et déroulé ».

</details>
<details><summary>Indice 2</summary>

Le schéma déroulé dessine une copie par pas de temps, mais ces copies sont **la même cellule**. Pour e), écris les deux produits matriciels de l'équation de la cellule.

</details>
<details><summary>Indice 3</summary>

a) B : le délai retient l'état jusqu'à l'arrivée de l'entrée suivante. Pour b), demande-toi si les copies ont des poids différents ; pour c), compte une application par pas ; pour d), relis la phrase sur le partage des poids ; pour e), compte les matrices $W_{ih}$ et $W_{hh}$.

</details>

### 22.Q4 — Entrée one-hot, taille de l'état et sens de ses valeurs

<details><summary>Indice 1</summary>

Relis les §22.4.1 et §22.4.2 de la fiche, et le one-hot du ch. 12.

</details>
<details><summary>Indice 2</summary>

Un vecteur one-hot a une case par élément du vocabulaire, toutes nulles sauf une. La matrice $W_{ih}$ a la forme $(D, H)$.

</details>
<details><summary>Indice 3</summary>

a) 3 000 : une case par mot. Pour b), compte les 1 ; pour c), demande-toi si $D$ et $H$ apparaissent ensemble dans une contrainte de forme ; pour d), multiplie $D$ par $H$ ; pour e), demande-toi qui a choisi ce que contient chaque case de l'état.

</details>

### 22.Q5 — (échantillons, pas de temps, features) : la bonne organisation

<details><summary>Indice 1</summary>

Relis le §22.5 de la fiche et l'encadré 🕰️ sur `batch_first`.

</details>
<details><summary>Indice 2</summary>

Un échantillon est une séquence ; ses pas de temps sont des mesures successives ; ses features sont les grandeurs mesurées à chaque pas. Les nuits ne sont pas mesurées : deux journées ne se suivent pas sans trou. Pour c), un `reshape` relit les mêmes nombres dans le même ordre, sans les déplacer.

</details>
<details><summary>Indice 3</summary>

a) « (10, 6, 2) » : 10 journées, 6 mesures, 2 grandeurs. Pour b), mets les pas de temps en premier ; pour c), demande-toi si `reshape` échange des axes ou seulement le découpage des nombres (essaie sur un petit tableau) ; pour d), demande-toi ce qu'est une séquence ici, sachant que les nuits manquent ; pour e), lis le deuxième nombre de la forme.

</details>

### 22.Q6 — BPTT, gradients évanescents et explosifs, dépendances longues

<details><summary>Indice 1</summary>

Relis le §22.6 de la fiche, et l'encadré « Rappel maths » sur les produits de nombreux facteurs.

</details>
<details><summary>Indice 2</summary>

$0{,}9^{50} = e^{50 \ln 0{,}9}$, avec $\ln 0{,}9 \approx -0{,}105$ ; $1{,}1^{50} = e^{50 \ln 1{,}1}$, avec $\ln 1{,}1 \approx 0{,}095$. Pour d), demande-toi ce que fait le clipping à un gradient **petit**.

</details>
<details><summary>Indice 3</summary>

a) B : la BPTT est la rétropropagation ordinaire, appliquée au réseau déroulé. Pour b), $e^{-5{,}3}$ ; pour c), $e^{4{,}8}$ ; pour d), le clipping ne touche que les gradients dont la norme dépasse un seuil ; pour e), relis le paragraphe « Les dépendances à long terme », et l'encadré « Le livre, corrigé » qui le suit.

</details>

### 22.Q7 — Une porte, c'est une multiplication

<details><summary>Indice 1</summary>

Relis le §22.7.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une porte multiplie une valeur par un nombre compris entre 0 et 1, calculé par une sigmoïde.

</details>
<details><summary>Indice 3</summary>

a) 20 : $0{,}25 \times 80$. Pour b), cherche la fonction dont les valeurs sont entre 0 et 1 ; pour c) et d), demande-toi ce que fait une multiplication par un nombre de $[0, 1]$ au signe et à la valeur absolue ; pour e), relis l'équation d'une porte.

</details>

### 22.Q8 — Oublier, se souvenir, sélectionner

<details><summary>Indice 1</summary>

Relis le §22.7.2 de la fiche, son tableau de correspondance entre les noms du livre et ceux de PyTorch, et le paragraphe « Ce que permettent les portes ».

</details>
<details><summary>Indice 2</summary>

La porte d'entrée dose ce qu'on ajoute à la mémoire ; la porte de sortie dose ce qu'on en montre. Pour d), écris $c_t = f_t\,c_{t-1} + i_t\,g_t$ et cherche les valeurs qui donnent $c_t = c_{t-1}$.

</details>
<details><summary>Indice 3</summary>

a) B : « se souvenir », c'est faire entrer du nouveau dans la mémoire. Pour b), cherche la porte qui choisit la partie de la mémoire que voit le reste du réseau ; pour c), relis l'histoire des portes (1997, 2000) ; pour d), remplace $f$ et $i$ dans l'équation de $c_t$ ; pour e), récite l'ordre $i, f, g, o$ à partir de 0.

</details>

### 22.Q9 — LSTM ou GRU ?

<details><summary>Indice 1</summary>

Relis le §22.7.2 de la fiche, paragraphes « Le GRU » et « LSTM ou GRU ? ».

</details>
<details><summary>Indice 2</summary>

Compte les blocs $[\,r \mid z \mid n\,]$ du GRU, et compare la formule du nombre de paramètres (le facteur $k$) des deux cellules.

</details>
<details><summary>Indice 3</summary>

a) 3 : un bloc par lettre $r$, $z$, $n$. Pour b), cherche une variable $c_t$ dans les équations du GRU ; pour c), compare les facteurs 3 et 4 ; pour d), relis ce que disent les comparaisons empiriques.

</details>

### 22.Q10 — One-to-many, many-to-one, many-to-many

<details><summary>Indice 1</summary>

Relis le §22.8.1 de la fiche et sa figure des structures.

</details>
<details><summary>Indice 2</summary>

Pour chaque tâche, compte les entrées et les sorties : une ou plusieurs ? Et si les deux sont multiples, chaque sortie correspond-elle à une entrée au même instant, ou faut-il avoir tout lu avant de répondre ?

</details>
<details><summary>Indice 3</summary>

a) B : une critique entière (plusieurs mots) donne un seul verdict. Pour b), compte les entrées et les sorties ; pour c), demande-toi si l'on peut traduire le premier mot avant d'avoir lu la fin de la phrase ; pour d), un label par image ; pour e), compte ce que lit et ce que rend le générateur pour **une** prédiction.

</details>

### 22.Q11 — RNN profond et bidirectionnel : quand et à quel prix ?

<details><summary>Indice 1</summary>

Relis les §22.8.2 à §22.8.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un RNN bidirectionnel lit aussi la séquence à l'envers : il lui faut toute la séquence. Pour c), regarde la forme de `output` d'une couche bidirectionnelle : $2H$ valeurs par pas.

</details>
<details><summary>Indice 3</summary>

a) Faux : en temps réel, on n'a pas encore les mots à venir quand il faut répondre. Pour b), demande-toi ce que lit une couche RNN : une valeur ou une séquence ? Pour c), d'où vient le $2H$ ? Pour d), compte les paramètres d'une couche bidirectionnelle ; pour e), demande-toi ce que coûte chaque couche et chaque sens ajoutés.

</details>

### 22.Q12 — 🕰️ Transformers, RNN légers et modèles à espace d'états

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §22.1 et la section « Après les RNN : Transformers et modèles à espace d'états ».

</details>
<details><summary>Indice 2</summary>

Un RNN ne peut calculer $h_t$ qu'après $h_{t-1}$ ; un Transformer compare toutes les positions entre elles d'un seul coup. Pour c) et d), relis où les RNN restent utiles et ce qu'ont repris les modèles à espace d'états.

</details>
<details><summary>Indice 3</summary>

a) 2017 (A. Vaswani et coll.). Pour b), demande-toi lequel des deux a une dépendance pas à pas pendant l'entraînement ; pour c), cherche le cas où mémoire et calcul sont rares et où les données arrivent une à une ; pour d), relis la phrase sur Mamba ; pour e), relis la première règle de la section sur les séries temporelles.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 22.R1 — Ch. 21 : Une convolution 1D peut-elle lire une séquence ?

<details><summary>Indice 1</summary>

Relis la convolution 1D (§21.5) et le champ récepteur (§21.3.4) de la fiche du ch. 21.

</details>
<details><summary>Indice 2</summary>

Sans padding, une convolution de noyau $k$ sur $n$ valeurs donne $n - k + 1$ sorties. Chaque convolution de noyau 3 (stride 1) agrandit le champ récepteur de 2.

</details>
<details><summary>Indice 3</summary>

a) 17 : $20 - 4 + 1$. Pour b), relis le partage des poids ; pour c), pars de 3 et ajoute 2 ; pour d), écris le champ récepteur de $n$ couches, $2n + 1$, et résous $2n + 1 \ge 1\,001$ ; pour e), compare la longueur des chemins entre deux positions éloignées, et ce que fait une dilatation au champ récepteur.

</details>

### 22.R2 — Ch. 19 : Gradient clipping et optimiseurs adaptatifs

<details><summary>Indice 1</summary>

Relis le gradient clipping et RMSprop, Adam dans la fiche du ch. 19.

</details>
<details><summary>Indice 2</summary>

Le clipping par la norme multiplie tout le vecteur par `max_norm / norme` quand la norme dépasse `max_norm`. Pour e), regarde la formule du premier pas d'Adam : le gradient apparaît en haut et en bas.

</details>
<details><summary>Indice 3</summary>

a) 5 : $\sqrt{3^2 + 4^2}$. Pour b), divise la composante par la norme ; pour c), demande-toi si une multiplication par un nombre positif change la direction ; pour d), relis la définition de RMSprop ; pour e), écris le premier pas $\eta\,\hat m / (\sqrt{\hat v} + \varepsilon)$ avec $\hat m = g$ et $\hat v = g^2$.

</details>

### 22.R3 — Ch. 15 : Ajuster le scaler sur le train seulement

<details><summary>Indice 1</summary>

Relis la fuite de données (ch. 15) et le min-max scaler (ch. 12), puis la section « Au-delà du livre : prévoir une série temporelle » de la fiche.

</details>
<details><summary>Indice 2</summary>

Avec une série temporelle, le test doit venir **après** l'entraînement, comme en production. Le min-max scaler calcule $(x - \min)/(\max - \min)$ avec le min et le max qu'il a vus pendant le `fit`.

</details>
<details><summary>Indice 3</summary>

a) Faux : mélanger mettrait des mois futurs dans l'entraînement. Pour b), demande-toi quelles valeurs on a le droit de regarder avant de tester ; pour c), applique la formule avec 0 et 200 ; pour d), demande-toi ce qui se passerait en production avec une valeur jamais vue ; pour e), cherche la méthode qui applique un scaler déjà ajusté.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 22.1 — Combien de fenêtres ? Températures, taches solaires et Holmes

<details><summary>Indice 1</summary>

Relis le paragraphe « Combien de fenêtres ? » du §22.2 de la fiche et sa figure.

</details>
<details><summary>Indice 2</summary>

Une fenêtre qui démarre en position $p$ utilise les positions $p$ à $p + w - 1$, et sa cible est en $p + w - 1 + h$. Cherche la plus grande position de départ possible, puis compte les départs espacés de $s$. Pour c), compte les années, puis les mois.

</details>
<details><summary>Indice 3</summary>

a) La cible doit exister : $p + 4 \le 23$, donc $p$ va de 0 à 19, soit 20 fenêtres. Pour b) à f), applique $N = \lfloor (T - w - h)/s \rfloor + 1$ avec le bon $T$ (en e), la série d'entraînement n'a que $T - 600$ mois) ; pour g), inverse la formule : $T \approx N\,s$.

</details>

### Ex 22.2 — Dérouler à la main une cellule RNN scalaire sur trois pas

<details><summary>Indice 1</summary>

Relis le §22.4 de la fiche et son mini-exemple de cellule scalaire.

</details>
<details><summary>Indice 2</summary>

Calcule les pas dans l'ordre : chaque $h_t$ a besoin de $h_{t-1}$. Garde les valeurs non arrondies d'un pas à l'autre. Pour f), sans la tanh, développe $h_3 = x_3 + w\,h_2$, puis remplace $h_2$, puis $h_1$.

</details>
<details><summary>Indice 3</summary>

a) $h_1 = \tanh(1 \times 2 + 0{,}5 \times 0) = \tanh 2$. Pour b) et c), applique la même formule avec $x_2$ puis $x_3$ et l'état précédent ; pour d), multiplie par $v$ ; pour e), recommence avec $(-1, 0, 2)$ ; pour f), le coefficient de $x_1$ est un produit de puissances de $w$ et de $u$.

</details>

### Ex 22.3 — Compter les poids : la cellule à 69 poids du livre, puis nn.RNN et nn.LSTM

<details><summary>Indice 1</summary>

Relis le §22.4.1 de la fiche, paragraphe « Compter les paramètres », et le paragraphe « Combien de paramètres ? » du §22.7.2.

</details>
<details><summary>Indice 2</summary>

Un neurone sans biais qui lit $n$ valeurs a $n$ poids ; une couche de $m$ tels neurones en a $n\,m$. PyTorch compte $k\,(D\,H + H^2 + 2H)$ avec $k = 1$ (RNN), 4 (LSTM) ou 3 (GRU). Pour h), la deuxième couche LSTM reçoit les 128 valeurs d'état de la première.

</details>
<details><summary>Indice 3</summary>

a) Chaque valeur d'état est calculée par un neurone A qui lit les 10 entrées : $8 \times 10 = 80$. Pour b), un neurone B lit les 8 valeurs d'état ; pour c), 3 neurones qui lisent 8 valeurs, plus leurs biais ; pour d), additionne en n'oubliant pas les 8 biais de l'état ; pour e) à h), applique la formule de PyTorch couche par couche (la deuxième couche du générateur a $D = 128$) ; pour i), compte les vecteurs de biais en trop.

</details>

### Ex 22.4 — Un pas de LSTM à la main

<details><summary>Indice 1</summary>

Relis les équations du LSTM au §22.7.2 de la fiche, et son mini-exemple.

</details>
<details><summary>Indice 2</summary>

Calcule d'abord les quatre pré-activations $a_k = W_{ih,k}\,x_t + W_{hh,k}\,h_{t-1} + b_k$, dans l'ordre $i, f, g, o$. Puis sigmoïde pour $i$, $f$, $o$ et tanh pour $g$. Enfin $c_t = f\,c_{t-1} + i\,g$ et $h_t = o \tanh c_t$.

</details>
<details><summary>Indice 3</summary>

a) $a_i = -0{,}5 \times 1 + 1 \times 0{,}5 + 0 = 0$, donc $i_t = \sigma(0)$. Pour b) à d), fais de même avec les blocs 1, 2 et 3 (le biais 1 n'est que dans le bloc $f$) ; pour e) et f), applique les deux dernières équations ; pour g), recalcule $f$ et $i$ avec les biais modifiés (que valent $\sigma(21)$ et $\sigma(-20)$ ?) ; pour h), dérive $f_t\,c_{t-1}$ ; pour i), PyTorch empile les 4 blocs dans les lignes de `weight_ih_l0`.

</details>

### Ex 22.5 — Softmax avec température à la main

<details><summary>Indice 1</summary>

Relis le paragraphe « Toujours le plus probable ? » du §22.9 de la fiche et son mini-exemple.

</details>
<details><summary>Indice 2</summary>

$p_k = e^{z_k/T} / \sum_j e^{z_j/T}$. Pour simplifier, divise en haut et en bas par $e^{z_1/T}$ : $p_1 = 1/(1 + e^{(z_2 - z_1)/T} + e^{(z_3 - z_1)/T})$. Pour les limites, regarde ce que deviennent les exposants négatifs.

</details>
<details><summary>Indice 3</summary>

a) $p(\text{k}) = 1/(1 + e^{-2} + e^{-2{,}5})$. Pour b) et c), divise les écarts $-2$ et $-2{,}5$ par $T$ ; pour d), les exposants tendent vers $-\infty$ ; pour e), ils tendent vers 0 ; pour f), le rapport vaut $e^{(z_1 - z_2)/T}$ ; pour g), demande-toi si diviser tous les logits par un même $T > 0$ change leur ordre.

</details>

### Ex 22.6 — Pourquoi le gradient s'évanouit : un produit de dérivées de tanh

<details><summary>Indice 1</summary>

Relis le §22.6 de la fiche, paragraphe « Le gradient qui s'évanouit », et la règle de la chaîne du ch. 18.

</details>
<details><summary>Indice 2</summary>

1. $\tanh'(a) = 1 - \tanh^2(a)$. 2. La règle de la chaîne s'enchaîne d'un pas au suivant : $\partial h_T/\partial h_t = \prod_k \partial h_k/\partial h_{k-1}$ ; $x_t$ n'agit sur $L$ qu'à travers $h_t$. 3. $0 < 1 - h_k^2 \le 1$. 4. Si $h_k \approx 0$, $1 - h_k^2 \approx 1$ ; une grande pré-activation sature la tanh. 5. Applique l'inégalité à chaque facteur. 6. $w$ intervient à chaque pas : additionne les contributions. 7. Dérive la composition en 0 par la règle de la chaîne.

</details>
<details><summary>Indice 3</summary>

1. $h_k = \tanh(a_k)$ et $\partial a_k/\partial h_{k-1} = w$, donc $\partial h_k/\partial h_{k-1} = (1 - h_k^2)\,w$. Pour 2., multiplie ces facteurs de $k = t + 1$ à $T$, puis ajoute $\partial h_t/\partial x_t = u\,(1 - h_t^2)$ ; pour 3., majore chaque $|1 - h_k^2|$ par 1 et calcule $0{,}9^{50}$ ; pour 4., calcule $1{,}5^{20}$ et demande-toi ce que fait la tanh à des pré-activations qui grandissent ; pour 5., majore la norme d'un produit par le produit des normes ; pour 6., écris $\partial L/\partial w = \sum_k \delta_k\,(1 - h_k^2)\,h_{k-1}$ et relie $\delta_k$ à $\delta_T$ ; pour 7., la dérivée en 0 de la composition est le produit des $\tanh'(0)$.

</details>

### Ex 22.7 — Le chemin additif de la cellule LSTM

<details><summary>Indice 1</summary>

Relis le §22.7.2 de la fiche, paragraphe « Le chemin additif », et l'encadré « Le livre, corrigé » qui le suit.

</details>
<details><summary>Indice 2</summary>

1. Les opérations sont élément par élément : chaque $c_{t,j}$ ne dépend que de $c_{t-1,j}$. 2. Enchaîne les jacobiennes diagonales. 3. $\sigma(0) = 0{,}5$, $\sigma(1) \approx 0{,}731$. 4. $h_{t-1}$ entre dans les pré-activations des portes par $W_{hh}$. 5. Cherche une fonction linéaire dans l'équation de $h_t$. 6. Même calcul qu'en 1.

</details>
<details><summary>Indice 3</summary>

1. $c_{t,j} = f_{t,j}\,c_{t-1,j} + i_{t,j}\,g_{t,j}$, donc $\partial c_{t,j}/\partial c_{t-1,l} = f_{t,j}$ si $l = j$, 0 sinon. Pour 2., multiplie des matrices diagonales et calcule $0{,}99^{100}$ ; pour 3., compare $0{,}5^{20}$ et $0{,}731^{20}$ ; pour 4., suis $c_{t-1} \to \tanh \to h_{t-1} \to W_{hh} \to$ les portes $\to c_t$ ; pour 5., regarde quelle fonction s'applique à $c_t$ avant la sortie ; pour 6., dérive $z_t \odot h_{t-1}$.

</details>

### Ex 22.8 — Fermi : les 17 millions de poids du CNN-LSTM et le prix d'une epoch

<details><summary>Indice 1</summary>

Relis le paragraphe « Le CNN-LSTM » du §22.8.1 de la fiche, la taille de sortie d'une convolution (ch. 21) et la formule des paramètres d'un LSTM.

</details>
<details><summary>Indice 2</summary>

Taille de sortie : $\lfloor (128 + 2 \times 2 - 5)/2 \rfloor + 1$. Le LSTM reçoit à chaque pas toute la carte aplatie : $D$ est grand, et $D \times 4H$ domine. Le produit $x_t W_{ih}$ coûte $D \times 4H$ multiplications-additions par pas.

</details>
<details><summary>Indice 3</summary>

a) $64 \times 64$ pixels sur 8 canaux. Pour b), $8 \times (5 \times 5 \times 1 + 1)$ ; pour c), $4\,(D H + H^2 + 2H)$ avec le $D$ de a) ; pour d), $128 \times 9 + 9$ ; pour e), additionne ; pour f), refais c) avec $H = 32$ ; pour g), $D \times 4 \times 128$ ; pour h), multiplie g) par le nombre d'images de l'epoch et prends la partie entière de $\log_{10}$.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 22.9 — Expliquer l'état caché d'un RNN en cinq lignes

<details><summary>Indice 1</summary>

Relis les §22.3 et §22.3.1 de la fiche, sans regarder les formules.

</details>
<details><summary>Indice 2</summary>

Cherche une situation où quelqu'un lit ou écoute dans l'ordre et garde un résumé en tête, de taille limitée, qu'il met à jour à chaque nouvel élément. Distingue le résumé (qui change à chaque fois) de la façon de résumer (apprise une fois).

</details>
<details><summary>Indice 3</summary>

Commence par : « Imagine quelqu'un qui écoute une histoire mot après mot et qui ne peut garder en tête qu'une petite fiche… » Puis dis ce qu'il écrit sur la fiche à chaque mot, ce qui ne change pas (sa manière de résumer, apprise par l'entraînement), et ce qu'il ne peut pas faire (une fiche de taille fixe ne garde pas tout).

</details>

### Ex 22.10 — Écrire « à la manière de », imiter une voix : usages et abus

<details><summary>Indice 1</summary>

Relis les §22.1, §22.8.1 et §22.9 de la fiche. Pour 1., cherche la durée des droits d'auteur en Europe et l'année de mort des deux auteurs.

</details>
<details><summary>Indice 2</summary>

1. En France et dans l'Union européenne, les droits patrimoniaux durent 70 ans après la mort de l'auteur ; le pastiche est une exception au droit d'auteur (art. L122-5 du Code de la propriété intellectuelle). 2. Cherche les procès intentés à des entreprises d'IA par des auteurs ou des journaux. 3. Pense aux personnes atteintes de SLA. 4. Cherche l'article 50 du règlement (UE) 2024/1689.

</details>
<details><summary>Indice 3</summary>

Pour 1., Conan Doyle est mort en 1930 et Verne en 1905 : calcule l'année où leurs œuvres sont entrées dans le domaine public, puis compare avec un auteur vivant. Le statut des derniers récits de Holmes aux États-Unis a fait l'objet d'un procès (Klinger contre la succession Conan Doyle, 2014) : vérifie ce qu'il a tranché. Pour 2., présente les deux camps (la copie des œuvres pour l'entraînement d'un côté, l'usage « transformatif » ou l'exception de fouille de textes et de données de l'autre). Pour 3. à 5., pense au consentement, à la vérification de l'identité, au marquage et à l'honnêteté envers les lecteurs.

</details>

### Ex 22.11 — Lire Hochreiter et Schmidhuber (1997) : Long Short-Term Memory

<details><summary>Indice 1</summary>

La version publiée a le DOI [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) ; une copie PDF libre se trouve sur le site du cours de deep learning de Carnegie Mellon : [deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf). Lis le résumé et l'introduction en entier, puis le début de chaque section indiquée.

</details>
<details><summary>Indice 2</summary>

1. Le résumé donne un nombre de pas. 2. Cherche l'analyse du « flux d'erreur » (*error flow*) et le facteur multiplié à chaque pas. 3. Le CEC est une unité reliée à elle-même par un poids fixe ; quelle valeur de ce poids et quelle activation donnent un facteur égal à 1 ? 4. Cherche les mots *input gate* et *output gate*. 5. Le problème de l'addition (*adding problem*) est un bon choix. 6. Lis l'annexe A.1.2 (*Approximate Derivatives for Truncated Backpropagation*).

</details>
<details><summary>Indice 3</summary>

Pour 3., écris le facteur d'un pas d'une unité reliée à elle-même : $f'(a)\,w$ ; le flux est constant si ce produit vaut 1 : avec une activation identité, $w = 1$ ; c'est le terme $c_{t-1}$ de $c_t = c_{t-1} + i_t\,g_t$, ta mémoire avec $f_t = 1$. Pour 4., compare avec les quatre blocs $i, f, g, o$ de PyTorch et relis l'historique de la fiche. Pour 6., compare avec `loss.backward()` sur une séquence déroulée.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 22.E1 — Pourquoi les Transformers ont-ils remplacé les RNN pour le texte ? Quand garderais-tu un LSTM ?

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §22.1 et la section « Après les RNN ».

</details>
<details><summary>Indice 2</summary>

Trois arguments : le calcul en parallèle pendant l'entraînement, des chemins courts entre positions éloignées, le passage à l'échelle. Puis les cas où un RNN reste pertinent : la mémoire et le coût par pas constants.

</details>
<details><summary>Indice 3</summary>

Commence par : « Un RNN doit calculer ses états l'un après l'autre, alors qu'un Transformer traite toute la séquence d'un coup… » Puis parle des dépendances lointaines, puis des cas où tu garderais un LSTM (embarqué, flux continu, petites séries), et termine par les modèles à espace d'états.

</details>

### 22.E2 — Explique le gradient évanescent et deux façons d'y remédier

<details><summary>Indice 1</summary>

Relis le §22.6 et le paragraphe « Le chemin additif » du §22.7.2.

</details>
<details><summary>Indice 2</summary>

Un produit de nombreux facteurs plus petits que 1 : d'où viennent ces facteurs ? Puis les remèdes : cellules à portes, initialisation, clipping (pour l'autre problème), séquences plus courtes.

</details>
<details><summary>Indice 3</summary>

Commence par : « Dans un RNN déroulé, le gradient qui remonte de $k$ pas est multiplié $k$ fois par la jacobienne d'un pas… » Donne un ordre de grandeur, la conséquence (les dépendances lointaines ne s'apprennent pas), puis le LSTM et son chemin additif, et une seconde solution.

</details>

### 22.E3 — LSTM ou GRU : différences et critères de choix

<details><summary>Indice 1</summary>

Relis le §22.7.2 de la fiche en entier.

</details>
<details><summary>Indice 2</summary>

Compare : le nombre de portes, la mémoire séparée ou non, le nombre de paramètres (4 blocs contre 3), les résultats empiriques.

</details>
<details><summary>Indice 3</summary>

Commence par : « Les deux ajoutent des portes à un RNN pour garder l'information longtemps ; le LSTM a une mémoire séparée et trois portes, le GRU… » Puis donne le rapport des paramètres, ce que disent les comparaisons, et comment tu choisirais en pratique.

</details>

### 22.E4 — Comment valider un modèle de série temporelle sans fuite de données ?

<details><summary>Indice 1</summary>

Relis la section « Au-delà du livre : prévoir une série temporelle » de la fiche, et 22.R3.

</details>
<details><summary>Indice 2</summary>

Trois sources de fuite : le découpage, la normalisation, les fenêtres à cheval sur la coupure. Puis la comparaison : une baseline naïve, une métrique adaptée.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je coupe dans le temps, jamais au hasard : l'entraînement d'abord, puis la validation, puis le test, comme en production… » Puis le scaler, les fenêtres, la validation glissante (`TimeSeriesSplit`), la persistance et la persistance saisonnière (même semaine l'an dernier), et ce que tu surveillerais après le déploiement.

</details>

### 22.E5 — À quoi sert la température quand on génère du texte ?

<details><summary>Indice 1</summary>

Relis le paragraphe « Toujours le plus probable ? » du §22.9 et l'encadré 🕰️ qui le suit.

</details>
<details><summary>Indice 2</summary>

La température divise les logits avant la softmax : petite, elle concentre la distribution ; grande, elle l'aplatit. Pour un contrat, on veut une réponse stable et fidèle ; pour un slogan, de la variété.

</details>
<details><summary>Indice 3</summary>

Commence par : « Le modèle donne un score à chaque token possible ; la température règle à quel point on suit ces scores au moment de tirer le suivant… » Puis les deux extrêmes, tes deux choix argumentés, et les réglages voisins (top-k, top-p).

</details>

<a id="notebook"></a>

## Notebook

Les indices des exercices 22.12 à 22.29 seront écrits à la session suivante, avec les parties A à D du notebook.
