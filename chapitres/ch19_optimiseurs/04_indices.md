# 19 · Optimiseurs — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 🛠️ Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 19.Q1 — Qu'appelle-t-on un optimiseur ?

<details><summary>Indice 1</summary>

Relis le chapeau de la fiche et son §19.1 : qui calcule le gradient, qui s'en sert ?

</details>
<details><summary>Indice 2</summary>

Sépare les deux étapes d'un pas d'entraînement : la passe arrière (ch. 18) produit des nombres ; l'optimiseur les transforme en changement de poids. Pour d), pense à l'ordre des appels dans une boucle PyTorch (encadré « En entreprise » du §19.1).

</details>
<details><summary>Indice 3</summary>

a) Faux : la rétropropagation reste là, elle fournit les gradients que l'optimiseur reçoit. Pour b), demande-toi de quoi une règle comme $w \leftarrow w - \eta\,g$ a besoin ; pour c), si le gradient dépend de l'optimiseur ou seulement du réseau, des données et des poids ; pour d), lequel des quatre appels modifie les poids.

</details>

### 19.Q2 — Minimum, maximum, plateau, selle : où le gradient s'annule-t-il ?

<details><summary>Indice 1</summary>

Relis le §19.2.1 de la fiche et son encadré sur la courbure : la nature du point dépend des courbures dans chaque direction.

</details>
<details><summary>Indice 2</summary>

Pour une fonction $ax^2 + by^2$, regarde les signes de $a$ et de $b$ : les deux positifs, ça monte partout ; les deux négatifs, ça descend partout ; de signes contraires, ça monte d'un côté et descend de l'autre. Une fonction constante n'a de courbure dans aucune direction.

</details>
<details><summary>Indice 3</summary>

a) $x^2 + 3y^2$ : deux coefficients positifs, ça monte dans toutes les directions : (A), un minimum. Pour b) et c), compare les signes des deux coefficients ; pour d), demande-toi ce que vaut le gradient sur tout le carré. Pour e), écris un pas $w \leftarrow w - \eta\,g$ avec $g = 0$.

</details>

### 19.Q3 — Minima locaux ou points selles : le vrai piège en grande dimension

<details><summary>Indice 1</summary>

Relis la fin du §19.2.1 de la fiche (Dauphin et coll.), et la raison simple qui y est donnée.

</details>
<details><summary>Indice 2</summary>

Pour b), un minimum demande que **toutes** les directions montent ; avec $d$ directions indépendantes, chacune avec une chance sur deux, multiplie les probabilités. Pour c), regarde comment cette probabilité évolue avec $d$. Pour d), pense à ce que fait une petite poussée au hasard autour d'une selle.

</details>
<details><summary>Indice 3</summary>

a) (B) : le livre et Dauphin et coll. insistent sur les points selles entourés de régions presque plates. b) La probabilité d'un minimum est $\left(\frac12\right)^d$ ; calcule-la pour $d = 10$ et écris-la sous la forme « 1 sur … ». c) Que devient $\left(\frac12\right)^d$ quand $d$ grandit, et que reste-t-il pour les autres cas (hors maximum, encore plus rare) ? d) Une selle a au moins une direction qui descend : un pas au hasard a-t-il une chance de la trouver ?

</details>

### 19.Q4 — Pourquoi lisser une erreur de comptage ?

<details><summary>Indice 1</summary>

Relis le §19.2.2 de la fiche, premier paragraphe.

</details>
<details><summary>Indice 2</summary>

Déplace le seuil d'un tout petit peu : le nombre d'erreurs change-t-il, sauf quand le seuil franchit un point ? Ce que vaut alors la pente dit si un gradient peut guider le seuil.

</details>
<details><summary>Indice 3</summary>

a) Vrai : le nombre d'erreurs ne change que quand le seuil passe sur un point, il est constant entre deux points. Pour b), quelle est la pente d'une fonction constante ? Pour c), relie ta réponse b) à la descente de gradient. Pour d), cherche, parmi les quatre, la seule qui varie de façon continue avec les sorties du modèle (ch. 13 et 18).

</details>

### 19.Q5 — Pas constant : pourquoi rebondit-on au fond de la vallée ?

<details><summary>Indice 1</summary>

Relis la fin du §19.3.1 de la fiche et l'encadré du ch. 18 sur la descente sur une parabole.

</details>
<details><summary>Indice 2</summary>

Pour $L(w) = 5w^2$, $L'(w) = 10w$, donc $w \leftarrow w - 10\eta\,w = (1 - 10\eta)\,w$. Classe chaque facteur : entre 0 et 1, entre −1 et 0, égal à −1, inférieur à −1.

</details>
<details><summary>Indice 3</summary>

a) Le facteur vaut $1 - 10 \times 0{,}15 = -0{,}5$. b) Le signe de ce facteur dit si $w$ change de signe. c) et d) Calcule $1 - 10\eta$ pour $\eta = 0{,}25$ et $0{,}2$, puis compare sa valeur absolue à 1. e) Pour quels $\eta$ la valeur absolue du facteur est-elle strictement inférieure à 1 ?

</details>

### 19.Q6 — Décroissance exponentielle : que vaut η après 50 pas ?

<details><summary>Indice 1</summary>

Relis le §19.3.2 de la fiche : après $k$ multiplications, $\eta_k = \eta_0\,\gamma^k$.

</details>
<details><summary>Indice 2</summary>

Pour b), résous $\gamma^k < \frac12$ avec un logarithme ($k > \ln 0{,}5 / \ln \gamma$, attention au sens de l'inégalité quand on divise par un nombre négatif), puis prends le premier entier. Pour d) et e), c'est une série géométrique (0B).

</details>
<details><summary>Indice 3</summary>

a) $\eta_{50} = 0{,}1 \times 0{,}99^{50} \approx 0{,}0605$. b) Calcule $\ln 0{,}5 / \ln 0{,}99$ et prends l'entier juste au-dessus. c) Calcule $0{,}8^{10}$, puis son inverse. d) et e) $\sum_{k \ge 0} \eta_0\,\gamma^k = \eta_0 / (1 - \gamma)$ pour $0 \le \gamma < 1$ ; applique-le.

</details>

### 19.Q7 — Plannings : exponentiel, retardé, par paliers, sur l'erreur, bold driver

<details><summary>Indice 1</summary>

Relis la liste des plannings du §19.3.3 de la fiche, et l'encadré 🕰️ qui la suit.

</details>
<details><summary>Indice 2</summary>

Repère pour chaque règle le déclencheur : le temps seul (chaque epoch, ou toutes les $s$ epochs, avec ou sans délai au départ) ou la loss (qui ne baisse plus, ou qui baisse et monte).

</details>
<details><summary>Indice 3</summary>

a) Une baisse toutes les 10 epochs, déclenchée par le temps seul : (C), par paliers. Pour b), regarde ce qui se passe pendant les 5 premières epochs ; pour c) et d), la loss décide-t-elle seulement de baisser, ou aussi d'augmenter ? Pour e), cherche le nom anglais de « quand la loss ne bouge plus ». Pour f), compare les unités des plannings du livre et de l'encadré 🕰️.

</details>

### 19.Q8 — Batch, stochastique, mini-batch : compter les mises à jour

<details><summary>Indice 1</summary>

Relis le §19.4.2 et le §19.4.3 de la fiche, et la façon dont le livre compte 120 000 et 50 000 mises à jour.

</details>
<details><summary>Indice 2</summary>

Une mise à jour par groupe d'exemples ; le dernier groupe incomplet compte comme une mise à jour. Pour f), pense à ce que coûte une epoch dans chaque stratégie.

</details>
<details><summary>Indice 3</summary>

a) Le batch, c'est tout le dataset : une seule mise à jour par epoch. Pour b), compte les groupes d'un exemple ; pour c), calcule $\lceil 1\,200 / 64 \rceil$ ; pour d), multiplie par le nombre d'epochs. Pour e), lequel n'a pas besoin de garder les exemples ? Pour f), deux epochs coûtent-elles toujours la même chose ?

</details>

### 19.Q9 — Momentum : que se passe-t-il si γ = 0 ? si γ = 1 ?

<details><summary>Indice 1</summary>

Relis le §19.5.1 de la fiche et son exemple avec $\mu = 0{,}5$.

</details>
<details><summary>Indice 2</summary>

Déroule $b_t = \mu\,b_{t-1} + 1$ sur quelques pas : $b_1 = 1$, $b_2 = 1 + \mu$, $b_3 = 1 + \mu + \mu^2$… C'est une série géométrique. Avec $\mu = 1$, que devient chaque terme ?

</details>
<details><summary>Indice 3</summary>

a) Avec $\mu = 0$, $b_t = g_t$ et le pas vaut $\eta\,g_t$ : (A), la descente ordinaire. Pour b), la limite de $1 + \mu + \mu^2 + \dots$ vaut $1/(1 - \mu)$ ; calcule-la. Pour c), écris $b_t$ avec $\mu = 1$. Pour d), écris $b_t = b_{t-1} + 0$ : la vitesse diminue-t-elle ? Pour e), relis l'histoire de la bille sur le plateau.

</details>

### 19.Q10 — Nesterov : regarder là où l'on va

<details><summary>Indice 1</summary>

Relis le §19.5.2 de la fiche, les deux formules et l'encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

Repère, dans la formule du livre, l'endroit où l'on évalue $\nabla L$ ; puis compte les coefficients que Nesterov demande en plus de $\eta$ et $\gamma$ ; enfin, relis ce que font les bibliothèques et pourquoi.

</details>
<details><summary>Indice 3</summary>

a) (B) : le gradient est pris en $w_t + \gamma\,\Delta_{t-1}$, là où l'élan va mener. Pour b), compte les hyperparamètres de la formule ; pour c), relis la phrase sur le changement de variable ; pour d), si le point anticipé est de l'autre côté du fond, dans quel sens pointe son gradient ?

</details>

### 19.Q11 — Adagrad, Adadelta, RMSprop : un learning rate par poids

<details><summary>Indice 1</summary>

Relis le §19.5.3 et le §19.5.4 de la fiche, avec leurs formules.

</details>
<details><summary>Indice 2</summary>

Pour b), regarde ce qu'on ajoute à $s$ à chaque pas, et son signe. Pour c), lis le coefficient de $g_t^2$ dans la formule de $v_t$. Pour d) et e), cherche le nom et la valeur par défaut dans le paragraphe de RMSprop. Pour f), relis la phrase sur l'algorithme d'origine d'Adadelta.

</details>
<details><summary>Indice 3</summary>

a) Vrai : chaque poids a son propre $s$, donc son propre facteur $1/\sqrt{s}$. Pour b), $g_t^2$ est toujours positif ; que fait alors $s$ ? Pour c), le coefficient du gradient le plus récent est $1 - \alpha$ ; calcule-le. Pour d) et e), relis la fiche. Pour f), quel paramètre Zeiler remplace-t-il par une moyenne des pas ?

</details>

### 19.Q12 — Adam, No Free Lunch et choix d'un optimiseur

<details><summary>Indice 1</summary>

Relis le §19.5.5 et le §19.6 de la fiche, avec l'encadré sur le No Free Lunch.

</details>
<details><summary>Indice 2</summary>

Pour a), repère les deux moyennes d'Adam et le rôle de chacune. Pour d), relis ce que dit vraiment le théorème : en moyenne sur quoi ? Pour e), relis le bilan du livre sur sa figure 19.49.

</details>
<details><summary>Indice 3</summary>

a) (A) : la moyenne des gradients est un momentum, la moyenne des carrés une échelle comme dans RMSprop. b) et c) Relis les valeurs proposées par les auteurs. d) Le théorème parle de **toutes** les fonctions possibles : les losses des réseaux en sont-elles un échantillon quelconque ? e) Relis la première phrase du bilan du livre au §19.6.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 19.R1 — Ch. 18 : du delta au gradient d'un poids

<details><summary>Indice 1</summary>

Relis, au ch. 18, la règle « ce qui entre fois le delta de ce qui reçoit » et la forme matricielle (§18.12.10).

</details>
<details><summary>Indice 2</summary>

$\partial L / \partial w_{ij} = a_i\,\delta_j$, et la descente fait $w \leftarrow w - \eta\,\partial L / \partial w$. Pour c), la transposée change $(32, 64)$ en $(64, 32)$, puis le produit matriciel garde les dimensions extérieures.

</details>
<details><summary>Indice 3</summary>

a) $0{,}4 \times (-2{,}5) = -1$. Pour b), le changement vaut $-\eta$ fois ce gradient ; attention au signe. Pour c), écris les formes de $\mathbf{A}_{\text{prev}}^\top$ et de $\boldsymbol{\Delta}$ côte à côte. Pour d), l'optimiseur reçoit-il le gradient ou le calcule-t-il ?

</details>

### 19.R2 — Ch. 16 : pourquoi l'initialisation He garde la variance stable

<details><summary>Indice 1</summary>

Relis, au ch. 16, la section sur les initialisations de Glorot et de He.

</details>
<details><summary>Indice 2</summary>

He : variance $2/n_{in}$, écart-type sa racine. Glorot (normale) : variance $2/(n_{in} + n_{out})$. Pour d), que fait ReLU à une entrée négative ?

</details>
<details><summary>Indice 3</summary>

a) $2/200 = 0{,}01$. Pour b), prends la racine carrée de a). Pour c), remplace $n_{in}$ et $n_{out}$ dans la formule de Glorot. Pour d), quelle part de ses entrées ReLU laisse-t-elle passer, en moyenne, si elles sont centrées ?

</details>

### 19.R3 — Ch. 12 : standardiser les features, et l'effet sur la descente

<details><summary>Indice 1</summary>

Relis l'encadré sur la courbure, la hessienne et le conditionnement (fiche §19.2.1).

</details>
<details><summary>Indice 2</summary>

La hessienne est diagonale : ses valeurs propres sont ses deux coefficients. Le conditionnement est le rapport du plus grand au plus petit ; la limite de convergence est $2/\lambda_{\max}$. Pour c), que vaut la moyenne des carrés d'une feature standardisée ?

</details>
<details><summary>Indice 3</summary>

a) $\kappa = 10^6 / 0{,}25 = 4\,000\,000$. Pour b), calcule $2/10^6$ et exprime-le en millionièmes. Pour c), remplace les deux moyennes des carrés par leur valeur après standardisation. Pour d), que devient le pas d'une feature dont les gradients sont mille fois plus grands, une fois divisé par leur taille habituelle ?

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 19.1 — Plannings exponentiel et par paliers à la main ✏️

<details><summary>Indice 1</summary>

Relis les formules des §19.3.2 et §19.3.3 de la fiche ; $\lfloor x \rfloor$ est la partie entière.

</details>
<details><summary>Indice 2</summary>

Exponentiel : $0{,}2 \times 0{,}9^k$. Paliers : calcule d'abord $\lfloor k/4 \rfloor$, puis $0{,}2 \times 0{,}5^{\lfloor k/4 \rfloor}$. Pour f), on veut $\gamma^4 = 0{,}5$. Pour g), l'exposant n'est plus $k$ mais $k - 4$.

</details>
<details><summary>Indice 3</summary>

a) $0{,}2 \times 0{,}9^3 = 0{,}2 \times 0{,}729 = 0{,}1458$. Pour b), même formule avec $k = 10$. Pour c) à e), calcule $\lfloor 3/4 \rfloor$, $\lfloor 4/4 \rfloor$ et $\lfloor 11/4 \rfloor$, puis la puissance de 0,5. Pour f), $\gamma = 0{,}5^{1/4}$. Pour g), l'epoch 8 est 4 epochs après la fin du palier.

</details>

### Ex 19.2 — Trois pas de momentum, deux conventions ✏️

<details><summary>Indice 1</summary>

Relis le §19.5.1 de la fiche : les deux écritures du momentum, et la phrase sur le planning.

</details>
<details><summary>Indice 2</summary>

À chaque pas, dans l'ordre : le gradient au point courant ($g = 2w$), puis la vitesse, puis la position. Écris un petit tableau $t$, $w_{t-1}$, $g_t$, $b_t$, $w_t$. Pour g) et h), refais seulement les deux premiers pas, en changeant $\eta$ au deuxième.

</details>
<details><summary>Indice 3</summary>

a) $g_1 = 2 \times 1 = 2$ et $b_1 = g_1 = 2$. b) $w_1 = 1 - 0{,}1 \times b_1$. Pour c) et d), $g_2 = 2\,w_1$, $b_2 = 0{,}9\,b_1 + g_2$, $w_2 = w_1 - 0{,}1\,b_2$ ; pour e), un pas de plus. Pour f), déroule $\Delta_1 = -\eta\,g_1$ et $\Delta_2 = 0{,}9\,\Delta_1 - \eta\,g_2$. Pour g), $w_2 = w_1 - 0{,}05\,b_2$ ; pour h), $\Delta_2 = 0{,}9\,\Delta_1 - 0{,}05\,g_2$, où $\Delta_1$ garde le $\eta$ du premier pas.

</details>

### Ex 19.3 — Nesterov : gradient anticipé et reformulation de PyTorch ✏️

<details><summary>Indice 1</summary>

Relis le §19.5.2 de la fiche, ses deux formules et la phrase sur le changement de variable.

</details>
<details><summary>Indice 2</summary>

Version du livre : à chaque pas, calcule d'abord le point anticipé $w_{t-1} + \mu\,v_{t-1}$, puis le gradient en ce point, puis $v_t$, puis $w_t$. Au premier pas, $v_0 = 0$ : le point anticipé est $w_0$. Version de PyTorch : gradient au point courant $p_{t-1}$, puis $b_t$, puis le pas $\eta\,(g_t + \mu\,b_t)$.

</details>
<details><summary>Indice 3</summary>

Premier pas du livre : point anticipé 1, gradient 2, $v_1 = -0{,}2$, $w_1 = 0{,}8$. a) Le point anticipé suivant vaut $w_1 + 0{,}9\,v_1 = 0{,}8 - 0{,}18 = 0{,}62$ ; b) le gradient y vaut deux fois ce point ; c) $v_2 = 0{,}9\,v_1 - 0{,}1 \times$ ce gradient, puis $w_2 = w_1 + v_2$ ; d) un pas de plus. Pour e), $g_1 = 2$, $b_1 = 2$, et le pas vaut $0{,}1\,(g_1 + 0{,}9\,b_1)$ ; pour f), recommence avec $g_2 = 2\,p_1$. Pour g), compare $p_2$ aux nombres de la version du livre.

</details>

### Ex 19.4 — Adagrad et RMSprop sur une suite de gradients ✏️

<details><summary>Indice 1</summary>

Relis les formules d'Adagrad (§19.5.3) et de RMSprop (§19.5.4) dans la fiche.

</details>
<details><summary>Indice 2</summary>

Fais un tableau par algorithme : $t$, $g_t$, accumulateur, pas $-\eta\,g_t/\sqrt{\cdot}$, position. Les gradients ne dépendent pas de $w$ : tu peux calculer les accumulateurs d'abord. Pour h), compare les deux premiers dénominateurs ; pour i), écris ce que contient la somme $s$ au pas $k$ quand le gradient ne change pas.

</details>
<details><summary>Indice 3</summary>

Adagrad : $s_1 = 9$, premier pas $-0{,}1 \times 3/3 = -0{,}1$, $w_1 = -0{,}1$ ; a) $s_2 = 9 + (-1)^2 = 10$. Pour b) et c), fais les pas avec $\sqrt{s_2}$, puis $s_3 = s_2 + g_3^2$. RMSprop : $v_t = 0{,}9\,v_{t-1} + 0{,}1\,g_t^2$ ; d) et e) au premier pas, f) et g) en continuant. Pour h), le rapport vaut $\sqrt{s_1}/\sqrt{v_1}$. Pour i), avec $g$ constant, $s_k = k\,g^2$ : écris le pas au pas $k$, puis le rapport demandé.

</details>

### Ex 19.5 — Un pas d'Adam et la correction de biais ✏️

<details><summary>Indice 1</summary>

Relis la formule d'Adam (fiche §19.5.5) et l'encadré sur la moyenne mobile exponentielle (§19.5.4).

</details>
<details><summary>Indice 2</summary>

Au pas $t$, dans l'ordre : $m_t$, $v_t$, puis les corrections avec $1 - \beta_1^t$ et $1 - \beta_2^t$ (avec le $t$ du pas en cours, 1 au premier), puis le pas $\eta\,\hat m_t/\sqrt{\hat v_t}$. Pour e), garde $m_1$ et $v_1$ sans les corriger. Pour j), multiplie tous les $g$ par $c$ et regarde comment $m$, $v$ et leur rapport changent.

</details>
<details><summary>Indice 3</summary>

a) $m_1 = 0{,}9 \times 0 + 0{,}1 \times 0{,}5 = 0{,}05$. Pour b), $v_1 = 0{,}001 \times 0{,}5^2$. Pour c) et d), divise $m_1$ par $1 - 0{,}9$ et $v_1$ par $1 - 0{,}999$, puis fais le pas. Pour e), calcule $0{,}01 \times m_1/\sqrt{v_1}$. Pour f) à h), recommence au deuxième pas avec $1 - 0{,}9^2$ et $1 - 0{,}999^2$. Pour i), regarde le signe de $\hat m_2$. Pour j), si $g$ devient $c\,g$, $m$ devient $c\,m$ et $\sqrt{v}$ devient $|c|\sqrt{v}$.

</details>

### Ex 19.6 — Adam + L2 ou AdamW : la même pénalité ? Convertir alpha, lambda et weight_decay d'une convention à l'autre ✏️

<details><summary>Indice 1</summary>

Relis la section « AdamW, le weight decay découplé » de la fiche, et les conventions écrites en tête de l'exercice.

</details>
<details><summary>Indice 2</summary>

Pour a) et b), dérive la pénalité : le terme ajouté au gradient est le `weight_decay` fois $w$. Pour c) à e), un gradient nul ne laisse que la décroissance. Pour f), le gradient vu par Adam est $0 + \text{wd}\cdot w$ : applique le premier pas d'Adam, que la fiche décrit. Pour h), égale les deux décroissances par pas. Pour i), égale le terme ajouté au gradient.

</details>
<details><summary>Indice 3</summary>

a) La dérivée de $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ est $\lambda\,\mathbf{w}$ : le `weight_decay` vaut $\lambda = 0{,}01$. Pour b), dérive $\lambda\,\lVert\mathbf{w}\rVert^2$. Pour c), $w \leftarrow w - 0{,}1 \times 0{,}01 \times w$. Pour d) et e), le facteur est $1 - \eta\cdot\text{wd}$, puis sa puissance 10 000. Pour f), quelle est la taille du premier pas d'Adam, quel que soit le gradient ? Pour g), $w\,(1 - \eta\cdot\text{wd})$, puis un pas d'Adam sur un gradient nul. Pour h), $\eta \cdot \text{wd} = \lambda$ avec $\eta = \alpha$. Pour i), $\text{wd} = \alpha / n_b$. Pour j), déroule deux pas de chaque version : que contient la vitesse $b_2$ dans la version couplée ?

</details>

### Ex 19.7 — Sur une parabole, η doit rester sous 2/a ∂

<details><summary>Indice 1</summary>

Relis l'encadré du ch. 18 sur la descente sur une parabole, puis celui du §19.2.1 sur la hessienne.

</details>
<details><summary>Indice 2</summary>

1. Remplace $L'(w_t) = a\,(w_t - w^\star)$ et soustrais $w^\star$ des deux côtés ; classe le facteur selon $\eta a$. 2. Résous $|1 - \eta a|^t = 0{,}1$. 3. Le gradient a deux composantes indépendantes. 4. Trace $|1 - \eta a_1|$ et $|1 - \eta a_2|$ en fonction de $\eta$ : le maximum est le plus petit là où les deux courbes se croisent. 5. Change de variable $\mathbf{u} = \mathbf{Q}^\top(\mathbf{w} - \mathbf{w}^\star)$, où $\mathbf{Q}$ a pour colonnes les vecteurs propres. 6. Écris $\Delta_t = e_t - e_{t-1}$.

</details>
<details><summary>Indice 3</summary>

1. $w_{t+1} - w^\star = (w_t - w^\star) - \eta\,a\,(w_t - w^\star)$ ; donc $e_t = (1 - \eta a)^t e_0$ : converge si et seulement si $|1 - \eta a| < 1$. Classe ensuite $0 < \eta a < 1$, $\eta a = 1$, $1 < \eta a < 2$, $\eta a = 2$ et $\eta a > 2$. 2. $t = \ln 0{,}1 / \ln|1 - \eta a|$. 3. Chaque coordonnée suit la question 1 avec sa propre courbure ; la condition doit tenir pour les deux. 4. Au croisement, $\eta a_1 - 1 = 1 - \eta a_2$. 5. Dans la base propre, $\mathbf{H}$ est diagonale. 6. Remplace $\Delta_t$ et $\Delta_{t-1}$ par des différences d'écarts dans la formule du livre, puis regroupe les termes en $e_t$ et $e_{t-1}$.

</details>

### Ex 19.8 — Moyenne mobile exponentielle : mémoire effective et biais initial ∂

<details><summary>Indice 1</summary>

Relis l'encadré sur la moyenne mobile exponentielle (fiche §19.5.4) et la formule d'Adam (§19.5.5).

</details>
<details><summary>Indice 2</summary>

1. Récurrence : suppose la formule vraie pour $t$, et injecte-la dans $m_{t+1} = \beta\,m_t + (1 - \beta)\,x_{t+1}$. 2. Série géométrique finie : $\sum_{k=0}^{t-1}\beta^k = (1 - \beta^t)/(1 - \beta)$. 3. Linéarité de l'espérance. 4. Dérive $\sum_k \beta^k = 1/(1 - \beta)$ par rapport à $\beta$. 5. Résous $\beta^h = \frac12$. 6. Calcule $m_1$, $v_1$, puis leurs versions corrigées. 7. Avec 2, $m_t$ et $v_t$ ont une forme simple quand tous les gradients sont égaux.

</details>
<details><summary>Indice 3</summary>

1. Pour $t = 1$, $m_1 = (1 - \beta)\,x_1$ : c'est la formule. Hérédité : $\beta\,(1 - \beta)\sum_{k=0}^{t-1}\beta^k x_{t-k} + (1 - \beta)\,x_{t+1}$, puis décale l'indice. 2. Multiplie la série par $1 - \beta$. 3. $\mathbb{E}[\hat m_t] = \frac{1 - \beta}{1 - \beta^t}\sum_k \beta^k\,\mathbb{E}[x_{t-k}]$. 4. $\sum_k k\,\beta^{k-1} = 1/(1 - \beta)^2$. 5. $h = \ln 2 / (-\ln\beta)$ ; pour la seconde question, résous $1 - 0{,}999^t > \frac12$. 6. $\hat m_1 = g_1$ et $\hat v_1 = g_1^2$ ; sans correction, $m_1 = (1 - \beta_1)\,g_1$ et $\sqrt{v_1} = \sqrt{1 - \beta_2}\,|g_1|$. 7. $m_t = (1 - \beta_1^t)\,g$ et $v_t = (1 - \beta_2^t)\,g^2$.

</details>

### Ex 19.9 — Fermi : la mémoire des optimiseurs, du MLP au LLM 🧮

<details><summary>Indice 1</summary>

Revois, au ch. 16, le nombre de paramètres d'une couche dense ($n_{in} \times n_{out} + n_{out}$), et relis ce que chaque optimiseur garde (fiche §19.5).

</details>
<details><summary>Indice 2</summary>

Additionne les paramètres des trois couches, puis compte les octets : 4 par nombre `float32`. Les états d'Adam : deux nombres par paramètre. La mémoire totale : paramètres, gradients (un nombre par paramètre) et états. Pour e) et f), additionne les octets par paramètre, puis multiplie par $7 \times 10^9$.

</details>
<details><summary>Indice 3</summary>

a) Première couche : $3\,072 \times 512 + 512 = 1\,573\,376$ ; avec $512 \times 256 + 256$ et $256 \times 10 + 10$, $P = 1\,707\,274$. Pour b), $4P$ octets, divisés par $10^6$. Pour c), deux fois b). Pour d), paramètres, gradients et deux états : combien de fois b) ? Pour e), $2 + 2 + 4 + 4 + 4$ octets par paramètre ; pour f), retire un tampon de 4 octets. Pour g), divise e) par 80 et arrondis au-dessus.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 🛠️ Réflexion

### Ex 19.10 — Expliquer le momentum avec une bille qui roule 🗣️

<details><summary>Indice 1</summary>

Relis le §19.5.1 de la fiche, et la figure de la vallée étroite.

</details>
<details><summary>Indice 2</summary>

Quatre idées, une phrase chacune : l'élan garde la direction des pas précédents ; sur le plat, il continue d'avancer ; dans une vallée étroite, les allers-retours s'annulent et l'avancée le long du fond s'additionne ; trop d'élan fait dépasser, voire sortir.

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Imagine une bille qui roule sur un terrain bosselé et cherche le point le plus bas. Sans élan, elle s'arrête dès que le sol est plat… » Continue avec le plateau, la vallée étroite, puis le risque.

</details>

### Ex 19.11 — Le coût énergétique d'une recherche d'hyperparamètres ⚖️

<details><summary>Indice 1</summary>

Les questions 2 à 4 se lisent dans les deux articles ; la question 1 est un calcul ; les questions 5 et 6 font appel au ch. 15 et à ton jugement.

</details>
<details><summary>Indice 2</summary>

1. Multiplie les nombres de choix, puis l'énergie d'un entraînement (puissance × durée). 2. Le tableau 3 de Strubell et coll. et leur tableau 1. 3. Le §4.1 et l'annexe D de Patterson et coll. 4. La section « Conclusions » de Strubell et coll. 5. Pense aux méthodes du ch. 15 qui évaluent moins de configurations, ou les arrêtent tôt.

</details>
<details><summary>Indice 3</summary>

1. $4 \times 5 \times 3 \times 3$ entraînements, chacun de $0{,}3\ \text{kW} \times 2\ \text{h}$. 2. Cherche la ligne « NAS » du tableau 3 (en livres de CO₂) et compare-la à la vie d'une voiture. 3. Patterson et coll. donnent un facteur entre l'estimation et la mesure, et expliquent quelle partie de la recherche l'estimation a mal reproduite. Pour 4 à 6, rédige.

</details>

### Ex 19.12 — Kingma & Ba (2014) : l'algorithme Adam ligne par ligne 📄

<details><summary>Indice 1</summary>

L'algorithme 1 est en page 2 ; le §2.1 juste après ; le §3 sur la correction de biais ; le §6 pour les expériences ; le §7.1 pour AdaMax.

</details>
<details><summary>Indice 2</summary>

2. Regroupe les deux corrections en un seul facteur devant $m_t / \sqrt{v_t}$. 3. Le pas effectif est $\alpha\,\hat m_t/\sqrt{\hat v_t}$ ; les auteurs le bornent dans deux cas selon la comparaison de $1 - \beta_1$ et $\sqrt{1 - \beta_2}$. 4. Que vaut $\hat m_t$ quand les gradients changent de signe d'un pas à l'autre ? 8. Lis la ligne `amsgrad` du pseudo-code de `torch.optim.Adam`.

</details>
<details><summary>Indice 3</summary>

2. $\alpha_t = \alpha\,\sqrt{1 - \beta_2^t}/(1 - \beta_1^t)$, puis $\theta_t = \theta_{t-1} - \alpha_t\,m_t/(\sqrt{v_t} + \hat\epsilon)$ ; compare $\hat\epsilon$ et $\epsilon$. 3. Calcule $1 - \beta_1$ et $\sqrt{1 - \beta_2}$ pour les valeurs par défaut, puis choisis le cas. 4. Quand le signe est incertain, $\hat m$ est petit devant $\sqrt{\hat v}$. 5. Relie le facteur $1 - \beta_2^t$ de leur calcul à la somme des poids de ∂ 19.8. 7. AdaMax garde un maximum, pas une moyenne : est-il biaisé vers 0 au départ ?

</details>

### Ex 19.13 — Lire la doc de torch.optim : correspondance livre ↔ PyTorch 🛠️

<details><summary>Indice 1</summary>

Chaque page de PyTorch commence par un pseudo-code : `input` (les hyperparamètres), `initialize` (les tampons), puis la boucle.

</details>
<details><summary>Indice 2</summary>

Pour la place d'$\epsilon$, cherche dans la boucle s'il est ajouté sous la racine ou après. Pour le weight decay, cherche la ligne $g_t \leftarrow g_t + \lambda\theta_{t-1}$ (couplé) ou $\theta_t \leftarrow \theta_{t-1} - \gamma\lambda\theta_{t-1}$ (découplé). Attention : PyTorch appelle $\gamma$ le learning rate, et $\lambda$ le weight decay.

</details>
<details><summary>Indice 3</summary>

Première ligne du tableau, pour exemple : `SGD` avec momentum, le momentum du livre, son $\gamma$, `momentum` (0 par défaut), pas d'$\epsilon$, weight decay ajouté au gradient avant le momentum. Pour 2, écris la mise à jour de la vitesse avec $\tau$ et compare-la à celle de ta classe. Pour 3, la note s'appelle « The implementation of SGD with Momentum/Nesterov subtly differs… ». Pour 7, demande-toi si l'option change les nombres calculés ou seulement leur façon d'être calculés.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 19.E1 — Adam ou SGD avec momentum : lequel choisir, et pourquoi ?

<details><summary>Indice 1</summary>

Relis le §19.6 de la fiche et ses deux encadrés 🕰️, puis l'encadré « En entreprise » de la fin.

</details>
<details><summary>Indice 2</summary>

Structure ta réponse : ce qu'apporte chacun (réglage plus facile, robustesse ; généralisation, coût mémoire), les domaines où l'un domine, et la méthode pour trancher (une comparaison à réglage égal).

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Par défaut je pars d'AdamW, parce qu'il est peu sensible au réglage et que c'est le standard pour les Transformers ; mais… » Cite la vision, la mémoire, et la façon honnête de comparer.

</details>

### 19.E2 — Adam avec weight_decay contre AdamW

<details><summary>Indice 1</summary>

Relis la section « AdamW, le weight decay découplé » de la fiche, et ✏️ 19.6.

</details>
<details><summary>Indice 2</summary>

Deux phrases à placer : où passe le terme de décroissance dans chacun ; ce que la division par $\sqrt{\hat v}$ fait à ce terme.

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Dans `Adam(weight_decay=...)`, la décroissance est ajoutée au gradient, donc elle passe dans les moyennes et est divisée par $\sqrt{\hat v}$ ; dans AdamW… » Termine par la conséquence pratique et la convention de PyTorch.

</details>

### 19.E3 — À quoi servent le warmup et la décroissance en cosinus ?

<details><summary>Indice 1</summary>

Relis la section « warmup et décroissance en cosinus » de la fiche et l'encadré 🕰️ du §19.3.3.

</details>
<details><summary>Indice 2</summary>

Un rôle chacun : le début de l'entraînement (poids aléatoires, moyennes d'Adam pas encore fiables, grands mini-batches) ; la fin (se poser au fond). Puis la forme du cosinus et ce qu'elle apporte.

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Le warmup fait monter le learning rate de 0 à sa valeur maximale pendant les premiers pas, parce qu'au départ… » Continue avec la décroissance, puis un exemple chiffré (LLaMA) et les alternatives.

</details>

### 19.E4 — Qu'est-ce que le gradient clipping, et quand l'utiliser ?

<details><summary>Indice 1</summary>

Relis la section « le gradient clipping » de la fiche.

</details>
<details><summary>Indice 2</summary>

Trois points : la définition (norme globale, un seul facteur), la différence avec le clipping par valeur, les cas où il est indispensable (RNN, Transformers, losses à falaises) et le choix du seuil (regarder la norme pendant quelques centaines de pas).

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Le gradient clipping par la norme calcule la norme de tous les gradients mis bout à bout ; si elle dépasse un seuil, on les multiplie tous par le même facteur… » Continue par la valeur, puis les cas d'usage.

</details>

### 19.E5 — Taille du mini-batch : quels compromis ?

<details><summary>Indice 1</summary>

Relis les §19.4.1 à §19.4.3 de la fiche et le quiz Q8.

</details>
<details><summary>Indice 2</summary>

Pèse : le bruit du gradient, le nombre de mises à jour par epoch, le parallélisme du matériel, la mémoire, et ce qu'il faut changer au learning rate quand la taille change (Goyal et coll., fiche section warmup).

</details>
<details><summary>Indice 3</summary>

Une ouverture possible : « Un mini-batch plus grand donne un gradient moins bruité et mieux parallélisé, mais moins de mises à jour par epoch… » Pour la question ×8, cite la règle d'échelle linéaire du learning rate et le warmup.

</details>

<a id="notebook"></a>

## Exercices du notebook

### Ex 19.14 — Pas constant sur une vallée : combien de pas pour le fond ? 🔮

<details><summary>Indice 1</summary>

Près du fond, la vallée ressemble à une parabole $\frac{a}{2}w^2$ : la règle de la fiche (§19.3.1) et de ∂ 19.7 dit ce que fait un pas constant selon la place de $\eta$ par rapport à $1/a$ et $2/a$. Loin du fond, elle ne ressemble plus du tout à une parabole.

</details>
<details><summary>Indice 2</summary>

Calcule $a = L''(0)$ : développe $e^{-8w^2} \approx 1 - 8w^2$ près de 0, ou dérive deux fois. Place ensuite chacun des quatre $\eta$ de l'énoncé par rapport à $1/a$ et $2/a$, et calcule le facteur $1 - \eta a$ qui multiplie l'écart au fond. Pour c) et d), demande-toi ce que fait la pente $L'(w)$ quand $|w|$ grandit : elle n'augmente pas sans fin, comme celle d'une parabole.

</details>
<details><summary>Indice 3</summary>

Pour a) : avec $a = L''(0)$, le facteur vaut $1 - a/8$ ; un facteur de valeur absolue proche de 1 rapproche à peine la bille du fond à chaque pas. Pour b) : le signe de $1 - 0{,}1\,a$ dit si la bille change de côté. Pour c) et d) : compare la taille d'un pas $\eta\,|L'(w)|$ à la largeur du creux (environ $|w| < 0{,}5$), et pense à ce qui se passe quand la bille arrive là où $L'(w) \approx 0$.

</details>

### Ex 19.15 — Plannings : exponential_decay_lr, step_decay_lr, bold_driver_lr 🔨

<details><summary>Indice 1</summary>

Trois fonctions pures, sans état : chacune vérifie ses arguments, calcule une formule de la fiche (§19.3.2, §19.3.3), et rend un `float`.

</details>
<details><summary>Indice 2</summary>

Commence chaque fonction par ses vérifications (`if step < 0 or base_lr < 0 or not 0 < gamma <= 1: raise ValueError(...)`). Pour les paliers, l'exposant est `step // step_size`. Pour le bold driver, teste les trois cas dans l'ordre de la docstring : baisse, hausse au-delà de la tolérance, sinon rien. Termine par `return float(...)` pour qu'un calcul sur des entiers rende quand même un `float`.

</details>
<details><summary>Indice 3</summary>

```python
def step_decay_lr(step, base_lr, step_size, gamma=0.1):
    # 1) the checks of the docstring (ValueError)
    # 2) the exponent: the number of FULL intervals already done
    return float(base_lr * gamma ** ...)

def bold_driver_lr(lr, loss, prev_loss, increase=1.05, decrease=0.5, tolerance=0.0):
    # check increase, decrease and tolerance first
    if loss < prev_loss:
        return float(lr * increase)
    # ... the rise beyond prev_loss * (1 + tolerance), then the unchanged case
```

</details>

### Ex 19.16 — SGD, Momentum et Nesterov à la manière de torch.optim 🔨

<details><summary>Indice 1</summary>

Les trois `step` ont la même structure : vérifier les gradients, incrémenter `t`, puis, pour chaque paramètre, calculer le gradient effectif (avec le weight decay), mettre à jour son tampon et modifier le paramètre en place. Seule la formule finale change (fiche §19.5.1 et §19.5.2, ✏️ 19.2 et 19.3).

</details>
<details><summary>Indice 2</summary>

Écris une fonction privée qui vérifie `len(grads) == len(self.params)` puis chaque forme (`np.shape(g) != p.shape`), et lève `ValueError` avant toute modification ; appelle-la au début de chaque `step`. Le gradient effectif : `g = np.asarray(grad, dtype=float)`, puis `g = np.asarray(g + self.weight_decay * p)` s'il y a un weight decay (une copie, et un vrai tableau même pour un 0-D). Le tampon : `state = self.state[i]` ; au premier pas, `state["momentum_buffer"] = g.copy()` ; ensuite, mise à jour en place du tampon. Puis `p -= self.lr * ...`.

</details>
<details><summary>Indice 3</summary>

```python
def step(self, grads):                          # Momentum
    grads = self._checked(grads)               # your helper: count, shapes, float arrays
    self.t += 1
    for p, grad, state in zip(self.params, grads, self.state):
        g = grad if self.weight_decay == 0 else np.asarray(grad + self.weight_decay * p)
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = g.copy()
        else:
            ...                                 # the buffer update of the docstring, in place
        p -= ...                                # in place, with the buffer of the docstring formula
```
Pour Nesterov, seule la dernière ligne change : elle combine `g` et la vitesse **déjà** mise à jour, comme le dit la docstring.

</details>

### Ex 19.17 — Adagrad, RMSprop et Adadelta 🔨

<details><summary>Indice 1</summary>

Même structure qu'en 19.16, avec un ou deux tampons par paramètre, des moyennes de carrés de gradients ; seule la place de `eps` et la formule du pas changent (fiche §19.5.3, §19.5.4, ✏️ 19.4).

</details>
<details><summary>Indice 2</summary>

Adagrad : au premier pas, `state["sum"] = np.full_like(p, self.initial_accumulator_value)` ; puis `s += g * g` et `p -= self.lr * g / (np.sqrt(s) + self.eps)`. RMSprop : `v` part de `np.zeros_like(p)`, puis `v *= alpha` et `v += (1 - alpha) * g * g`. Adadelta : deux tampons `v` et `u` à zéro ; calcule le pas `d` avec le `u` d'avant, puis mets `u` à jour avec `d`, et seulement ensuite `p -= self.lr * d`.

</details>
<details><summary>Indice 3</summary>

```python
for p, g, state in zip(self.params, grads, self.state):   # Adadelta (g with the weight decay)
    if not state:
        state["square_avg"], state["acc_delta"] = np.zeros_like(p), np.zeros_like(p)
    v, u = state["square_avg"], state["acc_delta"]
    ...                                                      # update v in place first
    d = np.sqrt(u + self.eps) / np.sqrt(v + self.eps) * g   # the step uses the OLD u
    ...                                                      # then update u in place, with d
    p -= ...                                                 # in place, with the docstring formula
```

</details>

### Ex 19.18 — Adam et AdamW 🔨

<details><summary>Indice 1</summary>

Deux moyennes mobiles par paramètre, $m$ et $v$, et une correction de biais qui utilise le numéro du pas **après** l'incrément ; AdamW ne diffère que par l'endroit où agit le weight decay (fiche §19.5.5 et section sur AdamW, ✏️ 19.5 et 19.6).

</details>
<details><summary>Indice 2</summary>

Après `self.t += 1`, calcule `c1 = 1 - beta1 ** self.t` et `c2 = 1 - beta2 ** self.t`. Pour chaque paramètre : `m` et `v` créés à zéro au premier pas ; `m *= beta1; m += (1 - beta1) * g` ; `v *= beta2; v += (1 - beta2) * g * g` ; puis le pas $\eta\,(m/c_1)/(\sqrt{v/c_2} + \epsilon)$. Pour Adam, `g` contient le weight decay (couplé) ; pour AdamW, `g` est le gradient brut, et `p *= 1 - self.lr * self.weight_decay` vient **avant** le pas.

</details>
<details><summary>Indice 3</summary>

```python
self.t += 1
beta1, beta2 = self.betas
c1, c2 = 1 - beta1 ** self.t, 1 - beta2 ** self.t
for p, g, state in zip(self.params, grads, self.state):     # AdamW: g is the raw gradient
    p *= 1 - self.lr * self.weight_decay                     # the decoupled decay, in place, first
    if not state:
        state["exp_avg"], state["exp_avg_sq"] = np.zeros_like(p), np.zeros_like(p)
    ...                                                      # update m and v in place
    p -= ...                                                 # in place: m and v corrected by c1 and c2, eps outside the root
```

</details>

### Ex 19.19 — Warmup + cosinus et gradient clipping 🔨

<details><summary>Indice 1</summary>

`cosine_lr` est une fonction par morceaux, à écrire dans l'ordre de la docstring ; `clip_grad_norm` calcule une seule norme pour tous les tableaux, puis, au besoin, un seul facteur (fiche, sections sur le warmup et sur le clipping).

</details>
<details><summary>Indice 2</summary>

`cosine_lr` : les vérifications d'abord, puis `if step < warmup_steps:`, `if step >= total_steps:`, et sinon `progress = (step - warmup_steps) / (total_steps - warmup_steps)` et la demi-arche `0.5 * (1 + math.cos(math.pi * progress))`. `clip_grad_norm` : la somme des carrés de **tous** les tableaux (`np.sum(np.square(g))` pour chacun, additionnés), sa racine, puis, si elle dépasse `max_norm`, `g *= max_norm / (norm + eps)` pour chaque tableau.

</details>
<details><summary>Indice 3</summary>

```python
def clip_grad_norm(grads, max_norm, eps=1e-6):
    if max_norm <= 0:
        raise ValueError(...)
    norm = math.sqrt(sum(float(np.sum(np.square(g))) for g in grads))
    if norm > max_norm:
        # one factor for all the arrays, applied in place
        ...
    return norm
```
Pour `cosine_lr`, écris les trois branches dans l'ordre de la docstring, chacune avec son `float(...)`.

</details>

### Ex 19.20 — Trop de momentum : la bille quitte-t-elle la vallée ? 🔮

<details><summary>Indice 1</summary>

Deux idées de la fiche : le pas effectif du momentum vaut environ $\eta/(1 - \mu)$ (tableau des pièges), et ∂ 19.7 (question 6) donne la condition de convergence sur une parabole. La vallée de 19.14, elle, n'est pas une parabole loin du fond.

</details>
<details><summary>Indice 2</summary>

Calcule $\eta/(1 - \mu)$ pour chaque $\mu$, et compare-le au seuil $2/a$ de 19.14 et à la largeur du creux. Pour b), calcule $\eta a$ et $2(1 + \mu)$ avec $a = 16$. Pour c), imagine la bille arrivée sur la partie plate, où $g \approx 0$ : que devient sa vitesse $b$ à chaque pas, et quelle distance totale fait-elle encore (une série géométrique, 0B) ?

</details>
<details><summary>Indice 3</summary>

Pour a), le premier $\mu$ : le pas effectif $\eta/(1 - 0{,}5) = 0{,}2$ dépasse $2/a$, mais la condition de ∂ 19.7 porte sur $\eta$ lui-même : $\eta a = 1{,}6 < 2(1 + 0{,}5)$, la parabole converge ; reste à savoir si, sur la vallée, la bille garde assez d'élan pour franchir le flanc. Fais le même calcul pour les quatre autres valeurs. Pour b), compare $\eta a$ à $2(1 + \mu)$. Pour c), si la vitesse est multipliée par $\mu$ à chaque pas, la distance restante vaut le dernier pas multiplié par $\mu/(1 - \mu)$.

</details>

### Ex 19.21 — Reproduire les figures du chapitre : pas constant, décroissant, momentum, Nesterov 🎨

<details><summary>Indice 1</summary>

`trajectory_21` est une boucle d'entraînement minuscule : un optimiseur sur un seul paramètre, le gradient de la vallée, et un planning qui change `opt.lr` avant chaque pas. `draw_21` ne fait que tracer.

</details>
<details><summary>Indice 2</summary>

Crée `w = np.array([w0])`, puis `opt = make_optimizer([w])`, et garde `base_lr = opt.lr` avant le premier pas. À chaque pas `k` : si `gamma` est donné, `opt.lr = mylearn.optim.exponential_decay_lr(k, base_lr, gamma)` ; puis `opt.step([valley_grad(w)])`, et note `w[0]`. Pour `draw_21` : `ax_curve.plot` pour la vallée et le trait fin, `ax_curve.scatter(..., c=np.arange(len(ws)), cmap="viridis")` pour les points, `ax_error.bar` (ou `plot`) pour les erreurs `valley(ws)`.

</details>
<details><summary>Indice 3</summary>

```python
def trajectory_21(make_optimizer, steps, gamma=None, w0=-0.6):
    w = np.array([w0], dtype=float)
    opt = make_optimizer([w])
    base_lr, positions = opt.lr, [w[0]]
    for k in range(steps):
        # the schedule (if gamma), then one step with valley_grad(w)
        positions.append(w[0])
    return np.array(positions)
```

</details>

### Ex 19.22 — Chasse au bug : un Adam qui diverge 🐛

<details><summary>Indice 1</summary>

Compare le code ligne à ligne avec la formule d'Adam de la fiche (§19.5.5) : le numéro du pas, chaque moyenne avec **son** coefficient, la racine. ∂ 19.8 (question 6) dit ce que vaut exactement le premier pas.

</details>
<details><summary>Indice 2</summary>

Le premier pas d'Adam doit valoir $\eta$ dans chaque coordonnée (∂ 19.8, question 6). Calcule-le à la main, ligne par ligne du code du collègue, pour le $t$ du premier pas : où le calcul s'écarte-t-il de $\eta$ ? Corrige une erreur à la fois et relance la vérification : chaque point qui passe en débloque un autre.

</details>
<details><summary>Indice 3</summary>

Les trois lignes à regarder : la boucle `for t in range(steps)` (pense à `range(1, steps + 1)`), la mise à jour de `m`, et la ligne qui modifie `w`, qui doit diviser par `np.sqrt(v_hat) + eps`.

</details>

### Ex 19.23 — Diagnostiquer des courbes de loss d'optimiseurs 📈

<details><summary>Indice 1</summary>

Chaque réglage laisse une signature sur la courbe : sa forme au début, à la fin, son bruit, ses ruptures (fiche §19.3.3, §19.4.2, §19.5.3, sections sur le warmup et le cosinus ; 18.22 pour la loss d'un réseau qui répond au hasard).

</details>
<details><summary>Indice 2</summary>

Pour chaque réglage, prédis la courbe avant de regarder : que fait la loss tant que le learning rate est presque nul (début d'un warmup, fin d'un cosinus) ? Que fait-elle quand $\eta$ est divisé d'un coup ? Que fait un pas qui ne cesse de rétrécir ? Que devient la loss d'un réseau ruiné qui répond pareil pour toutes les images ? Que vaut le bruit d'une loss calculée sur un seul exemple ?

</details>
<details><summary>Indice 3</summary>

Pour a) : cherche la courbe qui démarre plus lentement que les autres puis dont le bruit s'éteint à la fin, quand le cosinus ramène $\eta$ vers 0. Pour b) et c), cherche des ruptures brusques et compte-les. Pour d), une chute rapide puis une pente presque nulle. Pour e), une valeur finale qui ne bouge plus du tout. Pour f), l'amplitude du bruit.

</details>

### Ex 19.24 — Les solveurs de MLPClassifier : sgd, momentum, adaptive, adam 📦

<details><summary>Indice 1</summary>

Un dictionnaire de réglages par nom, et un seul `MLPClassifier(**communs, **réglages[name])` : ne passe que ce que le tableau demande (ch. 15 : `get_params()` montre ce que tu as passé).

</details>
<details><summary>Indice 2</summary>

Les réglages communs : `hidden_layer_sizes=(64,)`, `batch_size=64`, `max_iter=EPOCHS_24`, `random_state=0`. Puis, pour chaque ligne du tableau, `solver`, `learning_rate_init` et les réglages en plus ; « Nesterov est le défaut » veut dire qu'il ne faut rien passer pour lui. Rends un modèle **neuf** à chaque appel.

</details>
<details><summary>Indice 3</summary>

```python
def make_24(name):
    common = dict(hidden_layer_sizes=(64,), batch_size=64, max_iter=EPOCHS_24, random_state=0)
    settings = {"sgd": dict(solver="sgd", learning_rate_init=0.01, momentum=0.0),
                "momentum": dict(solver="sgd", learning_rate_init=0.01, momentum=0.9, nesterovs_momentum=False),
                # ... the four other rows of the table
                }
    return MLPClassifier(**common, **settings[name])
```

</details>

### Ex 19.25 — Batch, stochastique, mini-batch sur les lunes 🔬

<details><summary>Indice 1</summary>

C'est la boucle de 18.19, avec un réseau à trois couches cachées, la cross-entropy softmax, et **ton** `SGD` à la place de la mise à jour écrite à la main ; seule la taille des mini-batches change d'une stratégie à l'autre.

</details>
<details><summary>Indice 2</summary>

Importe `init_mlp` et `mlp_forward` (ch. 16), `softmax_cross_entropy` et `mlp_backward` (ch. 18), `relu` et `relu_derivative` (ch. 17), `iterate_minibatches` (`mylearn.utils`). L'optimiseur se crée **une fois**, sur `flat(params)`, avant les epochs ; à chaque mini-batch, `opt.step(flat(grads))` modifie les tableaux de `params` en place. Compte les appels à `step`, et mesure le temps avec `time.perf_counter()`. Après chaque epoch, compare `mlp_forward(X_25, params, relu).argmax(axis=1)` à `y_25`.

</details>
<details><summary>Indice 3</summary>

```python
started = time.perf_counter()
rng = np.random.default_rng(seed)
params = init_mlp(SIZES_25, "he_normal", rng)
opt = mylearn.optim.SGD(flat(params), lr=lr)
errors, updates = [], 0
for _ in range(max_epochs):
    for idx in iterate_minibatches(len(X_25), batch_size, rng=rng):
        logits, cache = mlp_forward(X_25[idx], params, relu, None, return_cache=True)
        # the loss and dout, the grads, opt.step(flat(grads)), updates += 1
    # count the errors on X_25; stop at 0
```

</details>

### Ex 19.26 — Course d'optimiseurs sur la fonction de Rosenbrock 🔬

<details><summary>Indice 1</summary>

Une boucle d'optimisation sur un seul paramètre de forme `(2,)`, comme `trajectory_21`, mais avec la classe choisie par son nom et le gradient de Rosenbrock (ch. 5).

</details>
<details><summary>Indice 2</summary>

`cls = getattr(mylearn.optim, name)` donne la classe ; `opt = cls([w], lr=lr)` garde les autres hyperparamètres par défaut. Note `w.copy()` après chaque pas (sans `copy`, toutes les positions seraient le même tableau, celui de la fin). Mets la boucle dans `with np.errstate(all="ignore"):` pour qu'une course qui explose ne remplisse pas l'écran d'avertissements.

</details>
<details><summary>Indice 3</summary>

```python
def race_26(name, lr, steps=STEPS_26):
    w = np.array(ROSEN_START, dtype=float)
    opt = getattr(mylearn.optim, name)([w], lr=lr)
    path = [w.copy()]
    with np.errstate(all="ignore"):
        for _ in range(steps):
            # one step with rosen_grad(w), then keep a copy of w
            ...
    return np.array(path)
```

</details>

### Ex 19.27 — Huit optimiseurs sur MNIST, avec warmup, cosinus et clipping 🔬

<details><summary>Indice 1</summary>

La boucle de 19.25 sur MNIST, avec trois ajouts dans chaque mise à jour : le planning avant la passe avant, la norme (et le clipping) entre la passe arrière et `step`, et l'historique.

</details>
<details><summary>Indice 2</summary>

Calcule `total = epochs * int(np.ceil(len(X_27) / BATCH_27))` avant la boucle, et un compteur `k` de mises à jour. Dans chaque mise à jour : `opt.lr = schedule(k, total)` si un planning est donné (avant le pas) ; la loss rendue par `softmax_cross_entropy` va dans `"loss"` ; la norme vient de `clip_grad_norm(grads, clip)` (qui rend la norme d'**avant**) si `clip` est donné, sinon de `math.sqrt(sum(np.sum(g * g) for g in grads))` ; puis `opt.step(grads)` et `opt.lr` dans `"lr"`.

</details>
<details><summary>Indice 3</summary>

```python
for _ in range(epochs):
    for idx in mylearn.utils.iterate_minibatches(len(X_27), BATCH_27, rng=rng):
        if schedule is not None:
            opt.lr = schedule(k, total)
        logits, cache = L.mlp_forward(X_27[idx], params, A.relu, None, return_cache=True)
        loss, dout = B.softmax_cross_entropy(logits, Y_27[idx])
        grads = flat(B.mlp_backward(dout, cache, params, activation_derivative=A.relu_derivative))
        # the norm (with clip_grad_norm if clip), opt.step(grads), the history, k += 1
    # the validation accuracy of the epoch
```

</details>

### Ex 19.28 — Défi : 98 % sur MNIST en cinq epochs 🏆

<details><summary>Indice 1</summary>

19.27 t'a montré ce qui gagne en peu d'epochs : un optimiseur adaptatif ou le momentum, **et** un planning. En cinq epochs, chaque mise à jour compte : le planning doit finir à un learning rate presque nul.

</details>
<details><summary>Indice 2</summary>

Pars de la boucle de 19.27 sur les 60 000 images, avec un réseau un peu plus large (deux couches cachées de quelques centaines de neurones), des mini-batches de 64 à 256, et un planning warmup + cosinus sur les 5 epochs. Règle le learning rate maximal sur une validation (les 10 000 dernières images d'entraînement), jamais sur le test caché. Vérifie le temps d'une epoch avant de lancer les cinq.

</details>
<details><summary>Indice 3</summary>

Une piste qui marche : Adam (ou AdamW) avec un learning rate maximal entre $10^{-3}$ et $3 \times 10^{-3}$, un warmup de quelques centaines de mises à jour, puis `cosine_lr` jusqu'à 0 sur `total = 5 * int(np.ceil(len(X) / batch))` mises à jour ; dans la boucle :

```python
opt.lr = O.cosine_lr(k, total, lr_max, warmup_steps=200)
logits, cache = L.mlp_forward(X[idx], params, A.relu, None, return_cache=True)
_, dout = B.softmax_cross_entropy(logits, y[idx])
opt.step(flat(B.mlp_backward(dout, cache, params, activation_derivative=A.relu_derivative)))
```

</details>
