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

a) B : la BPTT est la rétropropagation ordinaire, appliquée au réseau déroulé. Pour b), $e^{-5{,}3}$ ; pour c), $e^{4{,}8}$ ; pour d), le clipping ne touche que les gradients dont la norme dépasse un seuil ; pour e), relis le paragraphe « Les dépendances à long terme » : stocker le lien, ou l'apprendre ?

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

Relis la fin du §22.7.1 de la fiche (les noms du livre et ceux de PyTorch), puis le paragraphe « Ce que permettent les portes » du §22.7.2.

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

1. $h_k = \tanh(a_k)$ et $\partial a_k/\partial h_{k-1} = w$, donc $\partial h_k/\partial h_{k-1} = (1 - h_k^2)\,w$. Pour 2., multiplie ces facteurs de $k = t + 1$ à $T$, puis ajoute $\partial h_t/\partial x_t = u\,(1 - h_t^2)$ ; pour 3., majore chaque $|1 - h_k^2|$ par 1 et calcule $0{,}9^{50}$ ; pour 4., calcule $1{,}5^{20}$ et demande-toi ce que fait la tanh à des pré-activations qui grandissent ; pour 5., majore la norme d'un produit par le produit des normes ; pour 6., écris la contribution du pas $k$ comme $\delta_k$ fois la dérivée de $h_k$ par rapport à $w$ « à $h_{k-1}$ fixé », puis relie $\delta_k$ à $\delta_T$ par la question 2 ; pour 7., la dérivée en 0 de la composition est le produit des $\tanh'(0)$.

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

## Exercices du notebook

### Ex 22.12 — build_vocab, encode et decode sur Holmes et Verne 🔨

<details><summary>Indice 1</summary>

Un vocabulaire, c'est deux tables inverses l'une de l'autre : `itos` (la liste, l'id est la position) et `stoi` (le dictionnaire). Tout part d'un comptage des tokens, que `collections.Counter` fait en une ligne, sur une `str` comme sur une liste de mots.

</details>
<details><summary>Indice 2</summary>

`build_vocab` : vérifie `max_size` et `min_freq` (des entiers ≥ 1, pas des booléens) ; compte ; refuse un comptage vide et un token vide ; garde les tokens assez fréquents, sauf `unk_token` ; trie-les avec une clé `(-effectif, token)`, qui range par effectif décroissant puis dans l'ordre de `sorted` ; place `unk_token` devant, puis coupe à `max_size` ; construis `stoi` à partir de `itos`. `encode` : parcours les tokens et remplace un inconnu par l'id de `unk_token`, ou lève `ValueError`. `decode` : vérifie que `ids` est 1-D, entier (et pas booléen) et dans $[0, V)$ **avant** d'indexer.

</details>
<details><summary>Indice 3</summary>

```python
def build_vocab(tokens, max_size=None, min_freq=1, unk_token=None):
    # checks of max_size and min_freq (int >= 1, not a bool)
    counts = Counter(tokens)                     # a str counts its characters
    # refuse an empty count and an empty token
    ranked = sorted((tok for tok, cnt in counts.items() if cnt >= min_freq and tok != unk_token),
                    key=lambda tok: (-counts[tok], tok))
    # itos = [unk_token] + ranked (if given), cut to max_size; stoi from itos
```
Pour `decode`, le test sur le type : `arr.dtype == bool or not np.issubdtype(arr.dtype, np.integer)`.

</details>

### Ex 22.13 — make_windows et make_sequence_pairs 🔨

<details><summary>Indice 1</summary>

Relis ✏️ 22.1 : le nombre d'exemples et la position de la cible. Les deux fonctions se résument à une table d'indices : la ligne $i$ contient les positions lues par l'exemple $i$.

</details>
<details><summary>Indice 2</summary>

Vérifie les entiers et le nombre d'axes de la série (1 ou 2), calcule $N$ et refuse $N < 1$. Pour `make_windows`, mets la série en `(T, D)` (`reshape(T, -1)`), construis les départs `np.arange(N) * s` et la table `departs[:, None] + np.arange(w)[None, :]` de forme `(N, w)` : l'indexation par cette table rend directement `(N, w, D)`, et c'est une **copie**. La cible de l'exemple $i$ est à `départ + w - 1 + h`. Pour `make_sequence_pairs`, la même table avec `seq_len`, et `Y` lit la table décalée de 1 ; il faut donc $T - 1$ valeurs utilisables.

</details>
<details><summary>Indice 3</summary>

```python
def make_windows(series, window, horizon=1, stride=1):
    # checks: window, horizon, stride ints >= 1; series 1-D or 2-D
    n = (t - w - h) // s + 1 if t - w - h >= 0 else 0      # refuse n < 1
    starts = np.arange(n) * s
    X = arr.reshape(t, -1)[starts[:, None] + np.arange(w)[None, :]]   # (N, w, D), a copy
    # y = the value h steps after the last value of each window (a copy)
```
`make_sequence_pairs` : $N = \lfloor (T - 1 - L)/s \rfloor + 1$, puis `arr[idx]` et `arr[idx + 1]`.

</details>

### Ex 22.14 — Un MLP sur des fenêtres : l'ordre compte-t-il ? 🔮

<details><summary>Indice 1</summary>

Un sinus de période 24 : dans une fenêtre de 8 valeurs, qu'est-ce qui permet de prédire la suivante ? La hauteur actuelle, et le sens dans lequel la courbe va.

</details>
<details><summary>Indice 2</summary>

Pour a) et c), demande-toi ce qu'un **ensemble** de 8 valeurs sans ordre dit encore de la série, et ce qu'il ne dit plus. Pour b), la persistance répète la dernière valeur : que lui manque-t-il que deux valeurs consécutives donnent ?

</details>
<details><summary>Indice 3</summary>

Raisonne sur des ordres de grandeur. La persistance se trompe d'environ la variation de la série en un pas (plus le bruit) ; la constante, de l'amplitude du sinus. Situe l'erreur de la variante 3 entre les deux selon ce que l'ensemble des valeurs permet de savoir, et compare à chaque seuil de l'énoncé.

</details>

### Ex 22.15 — rnn_cell_forward et rnn_forward 🔨

<details><summary>Indice 1</summary>

Une cellule d'Elman est une couche dense qui reçoit deux entrées, $x_t$ et $h_{t-1}$ ; `rnn_forward` est une boucle `for` sur les pas, qui garde l'état d'un tour à l'autre. Avec la disposition de `mylearn` (`W_ih` de forme `(D, H)`), c'est `x @ W_ih`, sans transposée.

</details>
<details><summary>Indice 2</summary>

Commence par de petits outils que les cinq fonctions partageront : un qui convertit en **nouveau** tableau `float64` et vérifie le nombre d'axes ; un qui vérifie les poids (`W_ih` `(D, kH)`, `W_hh` `(H, kH)`, chaque biais `(kH,)` exactement, `None` remplacé par des zéros) ; un qui vérifie `x_t` et `h_prev` (2-D, même $N$). Dans `rnn_forward`, la taille $H$ vient de `W_hh.shape[0]` pour créer `h0` à zéro ; empile les états avec `np.stack(..., axis=1)` et rends `h_T` par `.copy()`.

</details>
<details><summary>Indice 3</summary>

```python
def rnn_cell_forward(x_t, h_prev, W_ih, W_hh, b_ih=None, b_hh=None, nonlinearity="tanh"):
    # nonlinearity in ("tanh", "relu"); x, h = checked 2-D float64 copies; weights and biases checked
    z = x @ w_ih + bi + h @ w_hh + bh
    return np.tanh(z) if nonlinearity == "tanh" else np.maximum(z, 0.0)
```
Dans `rnn_forward` : `h = rnn_cell_forward(seq[:, step], h, ...)` à chaque pas, puis `stacked = np.stack(outputs, axis=1)`.

</details>

### Ex 22.16 — lstm_cell_forward et lstm_forward (portes i, f, g, o) 🔨

<details><summary>Indice 1</summary>

Un seul produit matriciel donne les quatre portes à la fois : $a = x_t W_{ih} + b_{ih} + h_{t-1} W_{hh} + b_{hh}$, de forme `(N, 4H)`, que l'on découpe en quatre blocs de $H$ colonnes. Réutilise les outils de vérification de 22.15 avec $k = 4$.

</details>
<details><summary>Indice 2</summary>

Les blocs dans l'ordre de PyTorch : `a[:, :H]` pour $i$, `a[:, H:2H]` pour $f$, puis $g$, puis $o$. Une sigmoïde stable : $\sigma(z) = 1/(1 + e^{-z})$ pour $z \geq 0$, $e^{z}/(1 + e^{z})$ sinon ; avec `e = np.exp(-np.abs(z))`, aucune exponentielle ne déborde. Vérifie que `c_prev` a la forme de `h_prev`. Dans `lstm_forward`, l'état est le couple $(h, c)$, et la paire finale se rend avec des parenthèses : `(stacked, (h_T, c_T))`.

</details>
<details><summary>Indice 3</summary>

```python
def _sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def lstm_cell_forward(x_t, h_prev, c_prev, W_ih, W_hh, b_ih=None, b_hh=None):
    # checks (k = 4), c_prev of the shape of h_prev
    a = x @ w_ih + bi + h @ w_hh + bh                 # (N, 4H): blocks [i | f | g | o]
    # i, f, o = sigmoids of their blocks; g = tanh of its block
    c_t = f * c + i * g
```

</details>

### Ex 22.17 — gru_cell_forward (portes r, z, n) 🔨

<details><summary>Indice 1</summary>

Contrairement au LSTM, on ne peut pas additionner les deux produits avant de découper : le bloc $n$ a besoin de la part qui vient de $h_{t-1}$ **seule**, pour la multiplier par $r$.

</details>
<details><summary>Indice 2</summary>

Calcule séparément `gi = x @ W_ih + b_ih` et `gh = h @ W_hh + b_hh`, tous deux `(N, 3H)`. $r$ et $z$ : la sigmoïde de la somme des blocs correspondants ; $n$ : la tanh du bloc $n$ de `gi` plus $r$ fois le bloc $n$ de `gh` (qui contient déjà $b_{hn}$). Le nouvel état mélange $n$ et $h_{t-1}$ avec $z$.

</details>
<details><summary>Indice 3</summary>

```python
def gru_cell_forward(x_t, h_prev, W_ih, W_hh, b_ih=None, b_hh=None):
    # checks (k = 3)
    gi = x @ w_ih + bi                            # (N, 3H): blocks [r | z | n]
    gh = h @ w_hh + bh
    r = _sigmoid(gi[:, :hid] + gh[:, :hid])
    # z from the second blocks; n = tanh(gi_n + r * gh_n); h_t = (1 - z) * n + z * h
```

</details>

### Ex 22.18 — Benchmark : ta cellule NumPy contre nn.LSTM 🛠️

<details><summary>Indice 1</summary>

`time.perf_counter()` rend une heure en secondes ; une durée, c'est la différence de deux lectures, une avant et une après l'appel.

</details>
<details><summary>Indice 2</summary>

Un appel de chauffe sans mesure, puis une boucle de `repeat` tours qui mesure chaque appel et range sa durée dans une liste ; on rend le minimum, converti en `float`. Les arguments se transmettent tels quels avec `fn(*args)`.

</details>
<details><summary>Indice 3</summary>

```python
def best_time_18(fn, *args, repeat=5):
    fn(*args)                                   # warm-up, not timed
    durations = []
    # repeat times: time one call of fn(*args) with time.perf_counter() and keep its duration
    # return the shortest duration, as a Python float
```

</details>

### Ex 22.19 — Mesurer la disparition du gradient : RNN tanh contre LSTM 🔬

<details><summary>Indice 1</summary>

Pour obtenir un gradient par rapport aux **entrées**, il faut que l'entrée soit une feuille du graphe qui demande son gradient (ch. 20). Le contrat interdit de toucher à `x` : travaille sur une copie.

</details>
<details><summary>Indice 2</summary>

Une copie détachée qui demande son gradient (`x.detach().clone().requires_grad_(True)`), un appel du modèle, la somme des sorties du dernier pas, `backward()`, puis le `.grad` de la copie, de forme `(N, T, D)` : sa norme sur les axes 0 et 2 donne une valeur par pas. Les modules récurrents rendent `(output, état)` : seul `output` sert.

</details>
<details><summary>Indice 3</summary>

```python
def input_grad_norms_19(model, x):
    inputs = x.detach().clone().requires_grad_(True)    # a leaf of its own: x stays untouched
    # forward pass, then backward of L = the sum of the outputs of the last step
    # inputs.grad has the shape (N, T, D): one norm per step, over the axes 0 and 2
```

</details>

### Ex 22.20 — temperature_softmax et sample_from_logits 🔨

<details><summary>Indice 1</summary>

Relis ✏️ 22.5 et la softmax stable du ch. 17 : on retire le maximum de chaque ligne avant l'exponentielle, ce qui ne change pas le résultat. Pour tirer un id selon des probabilités, la méthode la plus courte passe par la somme cumulée (la CDF) et un nombre uniforme.

</details>
<details><summary>Indice 2</summary>

`temperature_softmax` : refuse $T \leq 0$, divise par $T$, retire `z.max(axis=-1, keepdims=True)`, exponentielle, normalise sur le dernier axe. `sample_from_logits` : refuse $T < 0$ ; vérifie que les logits sont `(V,)` ou `(N, V)` ; avec `top_k`, vérifie $1 \leq k \leq V$, trouve le $k$-ième plus grand logit de chaque ligne (`np.sort(z, axis=-1)[..., V - k]`) et remplace par $-\infty$ les logits **strictement** plus petits (les ex aequo restent) ; avec $T = 0$, `np.argmax` (qui rend le premier maximum) ; sinon les probabilités, leur somme cumulée, un uniforme par ligne tiré avec `rng`, et l'id est le nombre de valeurs cumulées que l'uniforme dépasse. À la fin, `int(...)` pour une seule ligne, un tableau `int64` sinon.

</details>
<details><summary>Indice 3</summary>

```python
def sample_from_logits(logits, temperature=1.0, top_k=None, rng=None):
    # checks: temperature >= 0, logits (V,) or (N, V), top_k int in [1, V]
    kth = np.sort(z, axis=-1)[..., v - k]
    z = np.where(z < kth[..., None], -np.inf, z)          # ties with the k-th logit are kept
    # T == 0: ids = np.argmax(z, axis=-1); else:
    cum = np.cumsum(temperature_softmax(z, temperature), axis=-1)
    # u = one uniform per row from np.random.default_rng(rng); ids = how many cum values u * cum[..., -1:] reaches
```

</details>

### Ex 22.21 — Série temporelle piégée : fuite, mélange et batch_first 🐛

<details><summary>Indice 1</summary>

Lis le symptôme : une persistance d'erreur **nulle** est impossible sur des données réelles. Puis suis les diagnostics dans l'ordre : chacun regarde une rampe 0, 1, 2…, où la valeur dit sa position.

</details>
<details><summary>Indice 2</summary>

Une ligne de `blocks` contient `window + horizon` valeurs : les entrées, le trou, puis la cible ; regarde quelles colonnes prend `X`. Pour le découpage, demande-toi si une fenêtre d'entraînement peut se trouver **après** une fenêtre de test. Pour le scaler, sur quelles valeurs sont pris le min et le max ? Pour le modèle, relis la forme que `nn.LSTM` attend par défaut (fiche §22.5, encadré 🕰️).

</details>
<details><summary>Indice 3</summary>

Le premier bug se corrige ainsi :
```python
X = blocks[:, :window]          # the first `window` values: the target stays outside the inputs
```
Pour les trois autres, le contrat dit ce qu'il faut : un découpage qui coupe à `len(y) - n_test` sans mélanger, un min-max calculé sur `series[:-n_test]`, et un LSTM qui lit `(N, T, 1)`.

</details>

### Ex 22.22 — RNN profonds et bidirectionnels : formes de output et h_n 📦

<details><summary>Indice 1</summary>

Relis la documentation de `nn.LSTM` (section *Outputs*) : `output` contient les états de la **dernière** couche à chaque pas, `h_n` l'état final de **chaque** couche et de chaque sens ; `batch_first` ne change que `output`.

</details>
<details><summary>Indice 2</summary>

a) et b) : à chaque pas, les deux sens de la dernière couche sont concaténés ; `h_n` empile `couches × 2` états de forme `(N, H)`. d) : `h_n` se lit comme une vue `(couches, 2, N, H)` : l'indice de la couche $\ell$ et du sens $s$ est $2\ell + s$. e) : le sens arrière commence au dernier pas ; où finit-il ? c) et `lstm_params_22` : $4(DH + H^2 + 2H)$ par couche et par sens, où $D$ vaut `input_size` pour la première couche et $2H$ (deux sens) ou $H$ pour les suivantes. `final_state_22` : `h_n.view(num_layers, 2, N, H)[-1]`, puis `torch.cat` des deux sens sur l'axe 1.

</details>
<details><summary>Indice 3</summary>

a) `"(4, 7, 40)"` ; pour b) à e), applique les règles de l'indice 2.
```python
def lstm_params_22(input_size, hidden_size, num_layers=1, bidirectional=False):
    directions = 2 if bidirectional else 1
    # for each layer: d = input_size for the first one, directions * hidden_size after
    total += directions * 4 * (d * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
```

</details>

### Ex 22.23 — Prévoir les taches solaires avec nn.LSTM, face à la persistance 📦

<details><summary>Indice 1</summary>

Trois étapes indépendantes : les données (le scaler, puis les fenêtres d'entraînement et de validation), le modèle (un LSTM et une couche dense), la boucle d'entraînement (celle du ch. 20, avec le clipping). Pour les fenêtres de validation, pars de la cible : la première est `SPOTS[I86]`, et sa fenêtre commence 65 mois plus tôt.

</details>
<details><summary>Indice 2</summary>

`make_windows(SPOTS[:I86] / scale_23, 60, horizon=6)` donne l'entraînement ; la validation vient du même appel sur `SPOTS[I86 - 65:I06] / scale_23`. La persistance à 6 mois compare `SPOTS[I86 - 6:I06 - 6]` à `SPOTS[I86:I06]`. Le modèle : `self.lstm(x)` rend `(out, (h, c))`, la couche dense lit `out[:, -1]` et `.squeeze(-1)` rend `(N,)`. La boucle : à chaque epoch, `torch.randperm(len(X), generator=generator)`, puis des tranches de `batch_size`, et dans chaque tranche `zero_grad`, la loss, `backward`, `clip_grad_norm_`, `step` ; la moyenne de l'epoch pondère chaque loss par la taille de son mini-batch.

</details>
<details><summary>Indice 3</summary>

```python
X_val_23, y_val_23 = mylearn.sequences.make_windows(SPOTS[I86 - 65:I06] / scale_23, 60, horizon=6)


def fit_23(model, X, y, epochs=20, lr=3e-3, batch_size=64, seed=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)      # created once, before the epochs
    # each epoch: order = torch.randperm(len(X), generator=generator), then the mini-batches
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # between backward() and step()
```

</details>

### Ex 22.24 — Température : prédire l'allure du texte généré 🔮

<details><summary>Indice 1</summary>

À température 0, `sample_from_logits` prend toujours l'argmax : le caractère suivant ne dépend que du caractère actuel. Que se passe-t-il dès qu'un caractère revient ?

</details>
<details><summary>Indice 2</summary>

Pour a), suis la chaîne des caractères les plus probables à partir de « T » : elle finit forcément par boucler, et la question est la longueur de la boucle. Pour b), une température de 2 divise les écarts de logits par 2 : calcule ce que devient le rapport des probabilités de deux caractères, puis estime combien de fois sort un caractère rare en 2 000 tirages. Pour c), un modèle qui ne voit qu'un caractère produit-il plutôt de longs mots inventés ou de petits mots fréquents ?

</details>
<details><summary>Indice 3</summary>

a) : le caractère le plus fréquent après une majuscule de début de phrase, puis après lui, et ainsi de suite, jusqu'à un caractère déjà vu ; compte les caractères distincts de la boucle et classe ce nombre dans les tranches. Pour b) et c), estime de la même façon ce qui sort en 2 000 tirages, puis place ton estimation dans les tranches de l'énoncé.

</details>

### Ex 22.25 — Générer du Holmes caractère par caractère (deux LSTM, many-to-one) 📦

<details><summary>Indice 1</summary>

Trois morceaux : le modèle (one-hot, LSTM, couche dense sur le dernier pas), la boucle d'entraînement (celle de 22.23, avec la cross-entropy et RMSprop), et la génération : une boucle qui ajoute un id à la liste à chaque tour et redonne au modèle les 40 derniers.

</details>
<details><summary>Indice 2</summary>

`F.one_hot(ids, V).float()` transforme `(N, T)` en `(N, T, V)` ; garde `vocab_size` en attribut pour `forward`. `train_25` : l'appareil du modèle s'obtient par `next(model.parameters()).device`, chaque mini-batch y va avec `.to(device)`. `generate_25` : encode l'amorce en liste d'ids, passe en mode évaluation et sous `torch.no_grad()`, puis à chaque tour un tenseur `[ids[-40:]]` de forme `(1, ≤ 40)`, les logits du seul exemple ramenés en NumPy `float64`, un tirage par `sample_from_logits`, et l'id ajouté ; à la fin, `decode` des seuls nouveaux ids.

</details>
<details><summary>Indice 3</summary>

```python
def generate_25(model, seed_text, n_chars, stoi, itos, temperature=1.0, rng=None):
    ids = list(mylearn.sequences.encode(seed_text, stoi))
    new = []
    model.eval()
    # with torch.no_grad(), n_chars times:
    logits = model(torch.tensor([ids[-40:]], device=device))[0].cpu().numpy().astype(np.float64)
    # nxt = sample_from_logits(logits, temperature=temperature, rng=rng); append it to ids and to new
```

</details>

### Ex 22.26 — Lire la courbe de loss et la perplexité du générateur 📈

<details><summary>Indice 1</summary>

La loss est une cross-entropy en nats (logarithme naturel) : la perplexité est $e^{\text{loss}}$, les bits par caractère $\text{loss}/\ln 2$. Le modèle sans contexte se calcule comme l'entropie du ch. 6.

</details>
<details><summary>Indice 2</summary>

a) `np.argmin` rend un indice qui commence à 0. b) et c) se calculent à cette epoch, pas à la dernière. d) Les fréquences des caractères de `HOLMES` (`Counter`), leur entropie en nats $-\sum p \ln p$, puis son exponentielle : c'est la perplexité d'un modèle qui donne à chaque caractère sa fréquence, quel que soit le contexte. e) Compare la pente des deux courbes sur les dernières epochs.

</details>
<details><summary>Indice 3</summary>

```python
best_epoch_26 = int(np.argmin(VAL_LOSS_26)) + 1
```
b) et c) appliquent les deux conversions de l'indice 1 à `min(VAL_LOSS_26)` ; d) applique l'exponentielle à l'entropie en nats ; e) : laquelle des deux courbes baisse encore sur les cinq dernières epochs ?

</details>

### Ex 22.27 — Holmes contre Verne : même réseau, deux langues 🔬

<details><summary>Indice 1</summary>

L'entropie d'une distribution de fréquences, en bits : $-\sum_c p_c \log_2 p_c$ (ch. 6). La cross-entropy de PyTorch est en nats : divise par $\ln 2$.

</details>
<details><summary>Indice 2</summary>

`unigram_bits_27` : les effectifs de `Counter(text)`, divisés par la longueur du texte, puis la formule. `bits_per_char_27` : une somme des cross-entropies de chaque mini-batch, prises avec `reduction="sum"`, divisée par le nombre **total** de fenêtres, puis par $\ln 2$. Une moyenne des moyennes de mini-batches donnerait trop de poids au dernier, plus petit.

</details>
<details><summary>Indice 3</summary>

```python
def bits_per_char_27(model, X, y, batch_size=2000):
    device = next(model.parameters()).device
    # model.eval(); with torch.no_grad(): for each slice X[i:i + batch_size] (to the device), add
    F.cross_entropy(logits, targets, reduction="sum").item()     # the sum over the slice, not its mean
    # the mean over ALL windows, in bits
```

</details>

### Ex 22.28 — CNN-LSTM : la balle monte-t-elle ou descend-elle ? 📦

<details><summary>Indice 1</summary>

Compte les paramètres couche par couche avant d'écrire la classe : la convolution (ch. 21), le LSTM (✏️ 22.3, avec deux biais par porte dans PyTorch), la couche dense. Puis le schéma de `forward` : toutes les images à plat, la convolution, retour aux vidéos, le LSTM, le dernier pas.

</details>
<details><summary>Indice 2</summary>

Convolution : $8 \times 1 \times 5 \times 5$ poids et 8 biais. LSTM : $4(DH + H^2 + 2H)$ avec $D = 128$ et $H = 32$. Couche dense : $32 \times 3 + 3$. Dans `forward`, `videos.reshape(N * T, 1, 32, 32)` donne un mini-batch d'images ; après la convolution, la ReLU, le pooling et un `nn.Flatten()`, on obtient `(N T, 128)`, à remettre en `(N, T, 128)` par `reshape(N, T, -1)` (l'ordre des images de chaque vidéo est conservé).

</details>
<details><summary>Indice 3</summary>

```python
def forward(self, videos):
    n, t = videos.shape[:2]
    frames = videos.reshape(n * t, *videos.shape[2:])       # (N T, 1, 32, 32)
    features = self.cnn(frames).reshape(n, t, -1)           # (N, T, 128)
    # the LSTM on features, then the head on the output of the last step
```

</details>

### Ex 22.29 — Défi taches solaires : battre la persistance de 20 % à six mois 🏆

<details><summary>Indice 1</summary>

Pars de ce que tu as : le LSTM de 22.23 réglé sur la période publique ne tient pas forcément sur la période cachée. Garde la dernière partie de `train` comme validation pour tes essais, et méfie-toi de ce qui ne marche que sur 1986-2005.

</details>
<details><summary>Indice 2</summary>

Deux pistes qui aident souvent : transformer la série pour calmer les pics du cycle (une racine carrée), et donner au modèle des **résumés** du passé plutôt que les valeurs brutes (des moyennes sur 1, 3, 6, 12, 24 mois, la tendance sur un an, le niveau d'il y a environ 11 ans, puisque le cycle dure à peu près autant) ; un modèle simple (une régression ridge) sur ces résumés est une bonne base, et la moyenne de deux modèles différents est souvent plus sûre que chacun.

</details>
<details><summary>Indice 3</summary>

```python
def make_forecaster_29(train):
    s = np.sqrt(np.asarray(train, dtype=float))
    # X: features of s[:t - 5] for each target t that has enough past; y: s[t]
    # fit one or two models on (X, y)

    def forecast(past):
        f = features(np.sqrt(np.asarray(past, dtype=float)))[None]
        return float(max(model.predict(f)[0], 0.0) ** 2)    # back from the square-root scale

    return forecast
```

</details>
