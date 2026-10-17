# 17 · Fonctions d'activation — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 17.Q1 — À quoi sert la fonction d'activation ?

<details><summary>Indice 1</summary>

Relis le §17.2 de la fiche et son encadré sur la composition d'applications affines.

</details>
<details><summary>Indice 2</summary>

Sans activation, chaque couche est affine, et la composée d'affines est affine : le réseau entier vaut **une** couche, des 4 entrées vers les 2 sorties. Pour b), compose $f(z) = 3z + 1$ avec la couche suivante. Pour d), demande-toi ce qui se passe entre les trois couches cachées.

</details>
<details><summary>Indice 3</summary>

a) La couche unique relie directement 4 entrées à 2 neurones : $4 \times 2$ poids, plus un biais par neurone de sortie ; fais la somme. b) Une activation affine multiplie et ajoute : se fond-elle dans la couche suivante ? c) Dans quel ordre un neurone fait-il ses trois gestes (§17.1) ? d) Entre deux couches cachées sans non-linéarité, la composée reste-t-elle affine ?

</details>

### 17.Q2 — Un nombre en entrée, un nombre en sortie : la forme d'une activation

<details><summary>Indice 1</summary>

Relis le §17.2.1 de la fiche : un nombre en entrée, un nombre en sortie, appliqué case par case.

</details>
<details><summary>Indice 2</summary>

« Élément par élément » veut dire que chaque case du résultat ne dépend que de la case correspondante de l'entrée. Pour c), le livre donne un synonyme dès la première phrase du §17.2.1.

</details>
<details><summary>Indice 3</summary>

a) `np.tanh` d'un tableau `(32, 10)` rend un tableau de la même forme : l'activation ne réduit aucun axe ; écris cette forme entre guillemets. b) Le livre dit ce qu'on fait « en pratique » pour une couche. c) Cherche le nom que le livre met entre parenthèses. d) Compare avec la softmax du §17.8, qui, elle, regarde toute la ligne.

</details>

### 17.Q3 — Linéaire ou non linéaire ? Classer six courbes

<details><summary>Indice 1</summary>

Relis le §17.3.1 de la fiche : au sens du livre, « linéaire » veut dire « une seule droite », qu'elle passe ou non par l'origine.

</details>
<details><summary>Indice 2</summary>

Pour chaque fonction, demande-toi si sa courbe a un coude, une courbure, un saut, ou si c'est une seule droite. Une fonction écrite en deux morceaux peut être une seule droite si les deux morceaux suivent la même formule.

</details>
<details><summary>Indice 3</summary>

1. $-2z$ est une droite qui passe par l'origine : `L`. Pour les cinq autres, le critère : 2. une droite d'ordonnée à l'origine 4 compte-t-elle comme « une seule droite » ? 3. un escalier a-t-il des sauts ? 4. $|z|$ a-t-elle un coude en 0 ? 5. et $\max(0, z)$ ? 6. compare les deux formules : sont-elles différentes ?

</details>

### 17.Q4 — Pourquoi une activation doit-elle donner une seule valeur par entrée ?

<details><summary>Indice 1</summary>

Relis le §17.3.1 de la fiche (« Une seule valeur par entrée ») et le §17.3.2 (l'escalier).

</details>
<details><summary>Indice 2</summary>

Le test de la verticale : une courbe est une fonction de $z$ si toute verticale la coupe au plus une fois. La continuité n'est pas demandée.

</details>
<details><summary>Indice 3</summary>

a) La verticale $z = 0$ coupe le cercle $z^2 + a^2 = 1$ en $a = 1$ et en $a = -1$ : deux valeurs, donc pas une fonction. b) Un escalier a-t-il une seule valeur en chaque point (pense aux points pleins et vides) ? c) Fais le test de la verticale pour chacune des cinq courbes ; la parabole couchée $z = a^2$ donne deux $a$ pour chaque $z > 0$.

</details>

### 17.Q5 — Marche unité, Heaviside, signe : les distinguer

<details><summary>Indice 1</summary>

Relis le §17.4 de la fiche et ses trois noms à connaître.

</details>
<details><summary>Indice 2</summary>

La marche unité vaut 0 à gauche du seuil et 1 à droite ; au seuil, le livre et ta librairie prennent la valeur de **droite**. Heaviside est la marche unité de seuil 0. La fonction signe a deux variantes en 0 : regarde laquelle suit NumPy, et laquelle suit `step(z, low=-1)`.

</details>
<details><summary>Indice 3</summary>

a) $-0{,}5$ est à gauche du seuil 0 : la marche de Heaviside y vaut 0. b) $z = 1{,}5$ est exactement au seuil : quelle valeur la convention retient-elle ? c) Essaie `np.sign(0.0)` dans une cellule, ou relis la variante « des mathématiciens ». d) `step` rend `high` dès que $z \ge$ `threshold` : que vaut `high` ? e) Quelle est la pente d'un palier ?

</details>

### 17.Q6 — ReLU, leaky, paramétrique, décalée : qui fait quoi ?

<details><summary>Indice 1</summary>

Relis le §17.5 de la fiche, son encadré « Le livre, corrigé » sur la PReLU, et l'encadré sur la dérivée en un point anguleux.

</details>
<details><summary>Indice 2</summary>

Leaky : $\alpha z$ pour $z < 0$. Décalée : $\max(z, -c)$. Une PReLU est une leaky ReLU dont la pente est un **paramètre** du modèle. Une pente 1 des deux côtés supprime le coude.

</details>
<details><summary>Indice 3</summary>

a) $z = -3 < 0$ : la leaky ReLU de pente $0{,}1$ rend $0{,}1 \times (-3) = -0{,}3$. b) Comment l'article de He et coll. traite-t-il la pente : choisie, tirée, ou apprise ? c) $\max(-3, -2)$ : le plus grand des deux. d) Avec une pente 1 à gauche et à droite, la fonction est-elle encore autre chose qu'une droite ? e) PyTorch prend la pente de quel côté pour ReLU en 0 ?

</details>

### 17.Q7 — Maxout et ReLU bruitée

<details><summary>Indice 1</summary>

Relis le paragraphe sur le maxout (§17.5 de la fiche) : en chaque $z$, la sortie est la plus haute des droites.

</details>
<details><summary>Indice 2</summary>

Calcule les trois droites au point demandé, puis garde la plus grande. Pour c), ReLU a deux morceaux de droite. Pour d), une ReLU bruitée tire un nouveau bruit à chaque passe.

</details>
<details><summary>Indice 3</summary>

a) En $z = -3$ : $0{,}5 \times (-3) = -1{,}5$, $2 \times (-3) - 1 = -7$ et $3 - 2 = 1$ ; le maximum est 1. b) Refais les trois calculs en $z = 2$. c) Combien de droites faut-il pour un coude ? d) Le bruit change-t-il d'une passe à l'autre ?

</details>

### 17.Q8 — Saturation : quelles fonctions, et de quel côté ?

<details><summary>Indice 1</summary>

Relis la saturation au §17.6 de la fiche, et la colonne « Sature » du tableau de la galerie (§17.7).

</details>
<details><summary>Indice 2</summary>

Pour chaque fonction, regarde sa dérivée quand $z \to -\infty$ et quand $z \to +\infty$ : tend-elle vers 0 (ou vaut-elle 0) ? Pour b) et c), $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ et $\tanh'(z) = 1 - \tanh^2(z)$.

</details>
<details><summary>Indice 3</summary>

a) La sigmoïde devient plate des deux côtés : son code est `2`. Pour les cinq autres, regarde la pente quand $z \to -\infty$ et quand $z \to +\infty$ : ReLU (pente 0 ou 1), leaky (pente $\alpha$ ou 1), ELU (pente $\alpha e^{z}$ ou 1), identité (pente 1). b) $\sigma(6) = 1/(1 + e^{-6})$, puis $\sigma(6)(1 - \sigma(6))$. c) $\tanh(3)$, puis $1 - \tanh^2(3)$. d) Où $\sigma'$ est-elle maximale, et que vaut-elle là ?

</details>

### 17.Q9 — Swish, SiLU, GELU : des ReLU lissées

<details><summary>Indice 1</summary>

Relis le §17.6 de la fiche : SiLU $= z\,\sigma(z)$, GELU $= z\,\Phi(z)$.

</details>
<details><summary>Indice 2</summary>

Pour a), calcule $\sigma(-0{,}5)$ puis multiplie par $-0{,}5$. Pour b), regarde la figure des fonctions lisses : SiLU descend-elle sous 0 avant de remonter ? Pour d) et e), regarde ce que vaut $\sigma(\beta z)$ quand $\beta$ est très grand, ou nul.

</details>
<details><summary>Indice 3</summary>

a) $\sigma(-0{,}5) = 1/(1 + e^{0{,}5}) \approx 0{,}3775$, et $\mathrm{SiLU}(-0{,}5) = -0{,}5 \times 0{,}3775 \approx -0{,}1888$. b) Une fonction qui descend puis remonte est-elle croissante partout ? c) $1 \times \Phi(1)$. d) Pour $\beta$ très grand, $\sigma(\beta z)$ vaut presque 0 si $z < 0$ et presque 1 si $z > 0$ : que vaut alors $z\,\sigma(\beta z)$ de chaque côté ? e) Que vaut $\sigma(0)$ ?

</details>

### 17.Q10 — Reconnaître les courbes de la figure 17.22

<details><summary>Indice 1</summary>

Compare la figure avec la galerie de la fiche (§17.7) : regarde surtout ce que fait chaque courbe pour $z$ négatif.

</details>
<details><summary>Indice 2</summary>

Deux courbes en S : la sigmoïde reste entre 0 et 1, tanh entre $-1$ et 1. Parmi les courbes qui ressemblent à ReLU : la leaky ReLU a un coude net ; ELU tend vers $-1$ ; SiLU descend un peu sous 0 puis remonte vers 0 ; softplus reste positive.

</details>
<details><summary>Indice 3</summary>

La courbe 1 est une courbe en S qui descend jusqu'à $-1$ : c'est tanh, lettre B. Pour les cinq autres, les critères : une courbe en S entre 0 et 1 ; un coude net en 0 avec une pente faible à gauche ; une courbe qui tend vers $-1$ sans coude ; une courbe qui descend sous 0 puis remonte vers 0 ; une courbe toujours positive.

</details>

### 17.Q11 — Softmax : somme à 1, ordre conservé, mais pas une activation

<details><summary>Indice 1</summary>

Relis les trois propriétés du §17.8 de la fiche.

</details>
<details><summary>Indice 2</summary>

L'exponentielle est croissante ; ajouter une constante à tous les scores ne change pas leurs écarts, les multiplier par 2 double leurs écarts ; et une activation, au sens du chapitre, agit sur un nombre à la fois.

</details>
<details><summary>Indice 3</summary>

a) Si $z_i > z_j$, alors $e^{z_i} > e^{z_j}$, et les deux sont divisés par la même somme : l'ordre est conservé, donc vrai. b) Les deux vecteurs ont-ils les mêmes écarts ? c) Et ceux-là ? d) De quoi dépend chaque sortie d'une activation, et chaque sortie de la softmax ?

</details>

### 17.Q12 — Vrai ou faux : des scores 0,6 / 0,3 / 0,1 se lisent comme des probabilités

<details><summary>Indice 1</summary>

Relis le §17.8 de la fiche, « Des scores aux probabilités », et le premier encadré « Le livre, corrigé » de cette section.

</details>
<details><summary>Indice 2</summary>

Après la softmax, $p_A / p_B = e^{z_A - z_B}$ ; $p_A = e^{0{,}6} / (e^{0{,}6} + e^{0{,}3} + e^{0{,}1})$. Pour d), multiplier les scores par 10 change leurs écarts.

</details>
<details><summary>Indice 3</summary>

a) Des scores bruts ne sont pas des probabilités : la phrase est fausse. b) Calcule $e^{0{,}6 - 0{,}3}$. c) Calcule les trois exponentielles, leur somme, puis le quotient. d) Les écarts de $6, 3, 1$ sont-ils ceux de $0{,}6, 0{,}3, 0{,}1$ ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 17.R1 — Ch. 16 : quelle initialisation va avec ReLU, et pourquoi ?

<details><summary>Indice 1</summary>

Relis le §16.4.1 de la fiche du ch. 16 (les trois règles).

</details>
<details><summary>Indice 2</summary>

He : $\mathrm{Var}(w) = 2/n_{in}$ ; l'écart-type est sa racine. Le facteur 2 compense ce que ReLU fait à la moitié des valeurs.

</details>
<details><summary>Indice 3</summary>

a) $2/50 = 0{,}04$. b) Prends la racine de a). c) Que fait ReLU aux valeurs négatives, et que vaut $\mathbb{E}[\mathrm{ReLU}(z)^2]$ pour $z$ symétrique ? d) tanh met-elle la moitié de ses valeurs à 0 ?

</details>

### 17.R2 — Ch. 14 : d'où viennent les probabilités d'une forêt aléatoire ?

<details><summary>Indice 1</summary>

Relis, au ch. 14, la différence entre vote dur et vote souple, et ce que fait le `predict_proba` d'une forêt de scikit-learn.

</details>
<details><summary>Indice 2</summary>

scikit-learn moyenne les distributions de probabilités de ses arbres. Une moyenne de distributions qui somment à 1 somme à 1.

</details>
<details><summary>Indice 3</summary>

a) Le `predict_proba` d'une forêt moyenne ceux de ses arbres : lettre A. b) Fais la moyenne des trois probabilités données. c) Une moyenne de distributions qui somment chacune à 1 : que vaut sa somme ?

</details>

### 17.R3 — Ch. 10 : pourquoi un perceptron à seuil ne s'entraîne pas par gradient ?

<details><summary>Indice 1</summary>

Relis le §17.4 de la fiche, « Pourquoi le perceptron ne s'entraîne pas par gradient ».

</details>
<details><summary>Indice 2</summary>

Une marche est plate de chaque côté du seuil. La règle du perceptron (ch. 10) corrige les poids quand un exemple est mal classé. « Presque partout » : partout sauf en quelques points isolés.

</details>
<details><summary>Indice 3</summary>

a) $z = 0{,}3$ est sur le palier de droite : la pente y vaut 0. b) De quoi se sert la règle du perceptron : d'une dérivée, ou de l'erreur commise ? c) Pour chaque fonction, la pente est-elle non nulle sauf en des points isolés ? ReLU est plate sur toute une demi-droite.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 17.1 — Effondrer un réseau linéaire en un seul neurone ✏️

<details><summary>Indice 1</summary>

Applique l'encadré de la fiche (§17.2) : $\mathbf{W}_{\text{eq}} = \mathbf{W}_1\mathbf{W}_2\mathbf{W}_3$, et chaque biais multiplié par les matrices qui le suivent.

</details>
<details><summary>Indice 2</summary>

Calcule d'abord $\mathbf{W}_1\mathbf{W}_2$ (forme $(3, 2)$), puis multiplie par $\mathbf{W}_3$ (forme $(3, 1)$). Le biais : $\mathbf{b}_1\mathbf{W}_2\mathbf{W}_3 + \mathbf{b}_2\mathbf{W}_3 + b_3$. Pour d), fais passer $\mathbf{x}$ couche par couche et applique ReLU après chaque couche cachée.

</details>
<details><summary>Indice 3</summary>

a) La première ligne de $\mathbf{W}_1\mathbf{W}_2$ vaut $(1 \times 2 + (-2) \times (-1) ;\ 1 \times 1 + (-2) \times 3) = (4 ;\ -5)$, et son produit par $\mathbf{W}_3$ donne $4 - 2 \times (-5) = 14$ : c'est le premier poids. Fais de même pour les deux autres lignes. b) Calcule $\mathbf{b}_1\mathbf{W}_2$, ajoute $\mathbf{b}_2$, multiplie par $\mathbf{W}_3$, ajoute $b_3$. c) $\mathbf{x}\cdot\mathbf{w}_{\text{eq}} + b_{\text{eq}}$. d) Les pré-activations de la première couche cachée pour ce $\mathbf{x}$ sont-elles positives ? e) et f) Compte les poids et les biais de chaque couche.

</details>

### Ex 17.2 — Tableau de valeurs de neuf activations ✏️

<details><summary>Indice 1</summary>

Écris la formule de chaque fonction (fiche §17.5 et §17.6) avant de calculer.

</details>
<details><summary>Indice 2</summary>

Pour $z < 0$ : ReLU rend 0, la leaky $0{,}1z$, la décalée $\max(z, -0{,}5)$, ELU $e^{z} - 1$. Softplus : $\ln(1 + e^{z})$ ; sigmoïde $1/(1 + e^{-z})$ ; SiLU $z\,\sigma(z)$ ; GELU $z\,\Phi(z)$, avec les valeurs de $\Phi$ données. Garde les valeurs exactes jusqu'à la fin.

</details>
<details><summary>Indice 3</summary>

a) En $z = -1$ : ReLU vaut 0, la leaky $-0{,}1$, la décalée $\max(-1, -0{,}5) = -0{,}5$. Pour les six autres, calcule $\ln(1 + e^{-1})$, $e^{-1} - 1$, $1/(1 + e^{1})$, $\tanh(-1)$, $-\sigma(-1)$ et $-\Phi(-1)$. b) En $z = 1{,}5$, les quatre fonctions linéaires pour $z > 0$ rendent $z$ ; calcule les cinq autres. c) Compte les valeurs strictement négatives de a).

</details>

### Ex 17.3 — Une composition d'affines reste affine ; une ReLU casse tout ∂

<details><summary>Indice 1</summary>

Pour 1 et 2, développe comme dans l'encadré de la fiche (§17.2). Pour 3, une fonction affine $f(z) = cz + d$ vérifie $f(u + v) = c(u + v) + d$.

</details>
<details><summary>Indice 2</summary>

2. Récurrence sur $L$ : si les $L - 1$ premières couches forment une application affine, la composée avec la couche $L$ l'est encore (question 1). 3. Dans le sens direct, remplace $f$ par $cz + d$ ; dans l'autre sens, pose $c = f(1) - f(0)$ et $d = f(0)$, puis utilise la seconde relation avec $u = 1$. Pour ReLU, essaie un $u$ positif et un $v$ négatif. 5. Chaque neurone $\mathrm{ReLU}(w_k x + b_k)$ a un coude en $x = -b_k / w_k$ (si $w_k \ne 0$).

</details>
<details><summary>Indice 3</summary>

1. $(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2 = \mathbf{x}(\mathbf{W}_1\mathbf{W}_2) + (\mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2)$, de matrice $(n_0, n_2)$ et de vecteur de taille $n_2$. Pour les autres questions, les critères : 2. que devient le biais accumulé quand on ajoute la couche $L$ ? 3. pour ReLU, compare $f(u + v)$ et $f(u) + f(v) - f(0)$ avec $u = 1$ et $v = -1$. 4. Pour quelles pentes la leaky ReLU est-elle une seule droite ? 5. Entre deux coudes consécutifs, chaque neurone est-il affine ? Combien de coudes au plus ? 6. Une composée de fonctions affines par morceaux est-elle affine sur chaque morceau commun ?

</details>

### Ex 17.4 — Dériver la sigmoïde, tanh et softplus ∂

<details><summary>Indice 1</summary>

Écris $\sigma(z) = (1 + e^{-z})^{-1}$ et dérive avec la règle de la chaîne (0B).

</details>
<details><summary>Indice 2</summary>

1. $\sigma'(z) = e^{-z}/(1 + e^{-z})^2$ : écris-le comme un produit de deux fractions. 2. Mets $\tanh(z)$ au même dénominateur que $2\sigma(2z) - 1$ ; puis dérive $2\sigma(2z) - 1$ avec la chaîne. 3. Dérive $\ln(1 + e^{z})$. 4. Dérive $\sigma(1 - \sigma)$ avec la règle du produit. 5. Règle du produit sur $z\,\sigma(z)$. Pour les vérifications, utilise $\sigma(\ln 3) = 3/4$, $\tanh(\ln 2) = 3/5$, $\sigma(\ln 4) = 4/5$.

</details>
<details><summary>Indice 3</summary>

1. $\sigma'(z) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}$, et le second facteur vaut $1 - \sigma(z)$. Pour les vérifications : a) $\sigma(\ln 3)(1 - \sigma(\ln 3)) = \frac34 \cdot \frac14 = 0{,}1875$. Pour les autres, le calcul posé : b) $1 - \tanh^2(\ln 2)$ ; c) $\sigma(\ln 4)$ ; d) $\sigma'(\ln 3)(1 - 2\sigma(\ln 3))$ ; e) $\tanh'(0) / \sigma'(0)$ ; f) $\sigma(\ln 3)\,(1 + \ln 3\,(1 - \sigma(\ln 3)))$.

</details>

### Ex 17.5 — Maxout et ReLU : construire |x|, une cuvette et une bosse ✏️

<details><summary>Indice 1</summary>

Trace les trois droites de a) sur une même figure, entre $z = -2$ et $z = 3$ : la cuvette est leur enveloppe du dessus.

</details>
<details><summary>Indice 2</summary>

Les coudes sont aux intersections de deux droites **voisines sur l'enveloppe** : calcule les trois intersections deux à deux, puis garde celles où les deux droites sont au-dessus de la troisième. Pour c), un $\mathrm{ReLU}(z - t)$ change la pente de $+1$ en $z = t$ : le chapeau a une pente 0, puis $+1$, puis $-1$, puis 0.

</details>
<details><summary>Indice 3</summary>

a) $-2z - 1 = 0{,}5z - 0{,}5$ donne $z = -0{,}2$, où les deux valent $-0{,}6$, et la troisième droite y vaut $-4{,}6$ : c'est un coude. Fais de même pour les deux autres paires, et vérifie chaque fois la troisième droite. b) Où le maximum est-il le plus bas : à un coude, ou loin des coudes ? c) Le coefficient d'un $\mathrm{ReLU}(z - k)$ est le saut de pente en $k$ : lis les pentes successives du chapeau, de gauche à droite, et fais les différences. d) Un maximum de fonctions affines est-il convexe ?

</details>

### Ex 17.6 — Softmax à la main : translation, température, rapports ✏️

<details><summary>Indice 1</summary>

Relis les trois propriétés et la température au §17.8 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un rapport ne dépend que d'un écart : $p_C / p_B = e^{z_C - z_B}$. Pour $p_C$, calcule les six exponentielles et leur somme. La température divise **chaque** score par $T$ avant la softmax.

</details>
<details><summary>Indice 3</summary>

a) $p_C / p_B = e^{0{,}8 - 0{,}1} = e^{0{,}7} \approx 2{,}01$. Pour les autres, le calcul posé : b) $e^{0{,}8} / \sum_j e^{z_j}$ ; c) que fait un décalage commun à la softmax ? d) $e^{8} / \sum_j e^{z_j / 0{,}1}$ ; e) quand $T$ grandit, tous les $z_j / T$ tendent vers 0 : vers quelle distribution ? f) $e^{2 \times 0{,}7}$.

</details>

### Ex 17.7 — La Jacobienne de la softmax ∂

<details><summary>Indice 1</summary>

Écris $p_i = e^{z_i} / S$ avec $S = \sum_j e^{z_j}$, et note que $\partial S / \partial z_j = e^{z_j}$.

</details>
<details><summary>Indice 2</summary>

1. Règle du quotient sur $e^{z_i} / S$ ; distingue $j = i$ (le numérateur dépend de $z_j$) et $j \ne i$ (il n'en dépend pas). 2. Somme une ligne : $\sum_j p_i(\delta_{ij} - p_j)$. Pour l'inversibilité, regarde $\mathbf{J}$ appliquée au vecteur $(1, \dots, 1)$. 4. Règle de la chaîne : $\partial \mathcal{L} / \partial z_j = \sum_i (\partial \mathcal{L} / \partial p_i)\,J_{ij}$, avec $\partial \mathcal{L} / \partial p_i = -\mathbb{1}[i = y] / p_y$. 6. $\mathbf{g}\mathbf{J} = \mathbf{g}\,\mathrm{diag}(\mathbf{p}) - (\mathbf{g}\,\mathbf{p}^\top)\,\mathbf{p}$.

</details>
<details><summary>Indice 3</summary>

1. $\partial p_i / \partial z_i = \frac{e^{z_i} S - e^{z_i} e^{z_i}}{S^2} = p_i - p_i^2 = p_i(1 - p_i)$. Pour la suite, le chemin : $j \ne i$, même quotient avec un numérateur constant ; 2. une ligne somme à $p_i - p_i \sum_j p_j$ ; 3. divise numérateur et dénominateur de $p_1$ par $e^{z_1}$ ; 4. ne garde que le terme $i = y$ de la somme ; 5. $e^{\mathbf{z}} = (1, 2, 3)$, $S = 6$ ; 6. combien d'opérations pour $\mathbf{g}\cdot\mathbf{p}$, puis pour le produit case par case ?

</details>

### Ex 17.8 — Fermi : une activation coûte-t-elle cher face au produit matriciel ? 🧮

<details><summary>Indice 1</summary>

Une passe avant coûte une multiplication par poids et par exemple (ch. 16) ; l'activation, une évaluation par neurone et par exemple.

</details>
<details><summary>Indice 2</summary>

Multiplications : $n \times n_{in} \times n_{out}$. Valeurs d'activation : $n \times n_{out}$. Opérations : 2 par multiplication-addition, 30 par activation. Octets : 4 par nombre, lus puis écrits. Temps : octets / débit, opérations / vitesse de calcul.

</details>
<details><summary>Indice 3</summary>

a) $256 \times 1\,024 \times 1\,024 \approx 268{,}4$ millions, arrondi à 268. Pour les autres, le calcul posé : b) $256 \times 1\,024$ en milliers ; c) a) divisé par b), en nombres exacts ; d) $30 \times$ b) divisé par $(2 \times$ a) $+ 30 \times$ b)) ; e) $2 \times 4 \times$ b) octets, en Mo ; f) e) divisé par $2 \times 10^{12}$, en microsecondes ; g) $2 \times$ a) divisé par $5 \times 10^{13}$, en microsecondes.

</details>

<a id="reflexion"></a>

## 🗣️ 📄 Réflexion

### Ex 17.9 — Expliquer la non-linéarité avec un train articulé 🗣️

<details><summary>Indice 1</summary>

Relis le début du §17.2 de la fiche : le train, les attelages, l'effondrement.

</details>
<details><summary>Indice 2</summary>

Wagons = neurones (ou couches) ; attelage articulé = activation ; attelages soudés = pas d'activation ; un train rigide ne suit pas les courbes, un réseau sans activation ne trace que des droites.

</details>
<details><summary>Indice 3</summary>

Une trame : « Un réseau, c'est comme un train de wagons… Les attelages articulés laissent chaque wagon tourner… Sans articulation, … Les activations sont ces articulations : … Mais l'image a une limite : … ». Pour la limite, pense à ce qui est **appris** dans un réseau (les poids) et qui n'a pas d'équivalent dans un train, ou à la forme de l'articulation (un coude de ReLU, une courbure de tanh).

</details>

### Ex 17.10 — Hendrycks & Gimpel (2016) : les GELU 📄

<details><summary>Indice 1</summary>

Lis d'abord le résumé et la section 2 ; la section 3 se survole (les tableaux et les figures de résultats suffisent), l'annexe se lit en entier.

</details>
<details><summary>Indice 2</summary>

1. Cherche le masque $m \sim \mathrm{Bernoulli}(\Phi(x))$ et son espérance. 2. Les deux approximations sont dans la section 2. 3. Liste les tâches (vision, langage, parole) et le résumé des résultats ; le conseil sur l'optimiseur est dans la discussion. 4. L'annexe raconte l'histoire de la SiLU, d'Elfwing et coll. et de swish.

</details>
<details><summary>Indice 3</summary>

Pour 2 : $\mathrm{GELU}(1) = 1 \times \Phi(1)$ pour la formule exacte ; pour l'approximation, calcule $u = \sqrt{2/\pi}\,(1 + 0{,}044715)$, puis $\frac12(1 + \tanh u)$, et fais la différence (de l'ordre de $10^{-4}$). Pour 5, demande-toi ce qu'un succès de GELU sur MNIST, TIMIT ou CIFAR dit d'un Transformer de plusieurs milliards de paramètres, et relis l'encadré 🕰️ de la fiche.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 17.E1 — Pourquoi ReLU plutôt que sigmoïde dans les couches cachées ?

<details><summary>Indice 1</summary>

Pense à ce que la rétropropagation fait des dérivées : elle les multiplie couche après couche (fiche §17.6, la saturation).

</details>
<details><summary>Indice 2</summary>

Trois arguments : la saturation ($\sigma' \le 1/4$, presque 0 loin de 0), le coût (une comparaison contre une exponentielle), les sorties creuses. Puis où la sigmoïde reste utile.

</details>
<details><summary>Indice 3</summary>

Une trame en trois temps : « Le problème de la sigmoïde, c'est… (un chiffre : dix couches) ; ReLU règle cela parce que… ; mais ReLU a son propre défaut… ; la sigmoïde reste utile pour… ». Appuie-toi sur ton résultat de 🔬 17.23.

</details>

### 17.E2 — Qu'est-ce qu'une ReLU morte, et comment l'éviter ?

<details><summary>Indice 1</summary>

Relis « Les neurones morts » au §17.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Définition (pré-activation négative pour tous les exemples), conséquence (gradient nul, poids figés), causes (learning rate, initialisation, biais, données), détection (la part de neurones toujours nuls sur un batch), remèdes.

</details>
<details><summary>Indice 3</summary>

Une trame : « Un neurone ReLU est mort quand… ; il ne reçoit alors… ; cela arrive surtout quand… ; je le détecte en mesurant… ; pour l'éviter : … ». Cite au moins trois remèdes, dont un qui change l'activation.

</details>

### 17.E3 — Softmax ou sigmoïde en sortie : multi-classe contre multi-étiquette

<details><summary>Indice 1</summary>

Relis « Deux classes : la sigmoïde » au §17.8 et le tableau de la couche de sortie dans la dernière section de la fiche.

</details>
<details><summary>Indice 2</summary>

La softmax force les probabilités à se partager une somme de 1 : une seule classe vraie. Une sigmoïde par classe donne des probabilités indépendantes. Pour PyTorch, relis l'encadré 🕰️ sur la softmax en sortie.

</details>
<details><summary>Indice 3</summary>

Une trame : « Pour les sujets d'articles, plusieurs peuvent être vrais à la fois, donc… ; pour un chiffre manuscrit, un seul est vrai, donc… ; dans PyTorch, le modèle rend… et la loss… ; je n'applique la softmax que pour… ».

</details>

### 17.E4 — Pourquoi GELU et SiLU dans les Transformers ?

<details><summary>Indice 1</summary>

Relis les paragraphes sur SiLU et GELU (§17.6) et l'encadré 🕰️ « GELU et SiLU ».

</details>
<details><summary>Indice 2</summary>

Ce qui les distingue de ReLU : lisses, non monotones, une pente non nulle pour $z$ un peu négatif. Pourquoi les préférer : des gains empiriques mesurés à grande échelle, plus qu'une théorie ; le bloc SwiGLU ajoute une porte multiplicative.

</details>
<details><summary>Indice 3</summary>

Une trame : « GELU, c'est z Φ(z), une ReLU lissée… ; SiLU, c'est… ; leur avantage : … ; pourquoi les Transformers : les articles de BERT et de GPT ont… et LLaMA utilise SwiGLU, qui… ; honnêtement, l'explication est surtout empirique : … ».

</details>

<a id="notebook"></a>

## Exercices du notebook

### Ex 17.11 — Un MLP sans activation sur les lunes : quelle frontière ? 🔮

<details><summary>Indice 1</summary>

Applique l'argument de l'effondrement (fiche §17.2) à un `MLPClassifier` à `activation="identity"` : que calcule-t-il avant sa sigmoïde de sortie ?

</details>
<details><summary>Indice 2</summary>

Un MLP sans activation est une application affine de ses deux entrées suivie d'une sigmoïde : c'est le même **modèle** qu'une régression logistique. Ses milliers de paramètres changent-ils la forme de sa frontière ? Pour c), pense à ce qu'une ligne brisée peut suivre.

</details>
<details><summary>Indice 3</summary>

Les deux modèles ont-ils le même ensemble de frontières possibles ? Qu'est-ce qui pourrait quand même les différencier (la loss, la régularisation, l'optimiseur) ? Pour le troisième modèle, demande-toi quelle forme de frontière ses 100 neurones ReLU lui permettent de tracer.

</details>

### Ex 17.12 — identity, step, relu, leaky_relu et leurs dérivées 🔨

<details><summary>Indice 1</summary>

Chaque fonction tient en deux lignes : convertir `z` en un **nouveau** tableau `float64`, puis un `np.where` (ou une comparaison convertie en `float`).

</details>
<details><summary>Indice 2</summary>

`np.array(z, dtype=float)` copie toujours (contrairement à `np.asarray`). `np.where(z >= threshold, high, low)` pour la marche ; `np.where(z > 0, z, 0.0)` pour ReLU ; la dérivée de ReLU est `(z > 0)` converti en `float` ; celle de la leaky vaut 1 où `z > 0` et `negative_slope` ailleurs. Les dérivées constantes : `np.ones_like` et `np.zeros_like` du tableau converti.

</details>
<details><summary>Indice 3</summary>

```python
def leaky_relu_derivative(z, negative_slope=0.01):
    z = np.array(z, dtype=float)                      # a new float64 array
    return np.where(z > 0, ..., ...)                  # strict: what is the slope at z = 0?
```

Les sept autres suivent le même squelette : la conversion, puis une seule expression.

</details>

### Ex 17.13 — Sigmoïde, tanh et softplus sans débordement 🔨

<details><summary>Indice 1</summary>

Relis les principes 1 et 2 de la section « Calculer sans débordement » de la fiche : l'exponentielle d'un nombre négatif ou nul, et `np.where` qui calcule ses deux branches.

</details>
<details><summary>Indice 2</summary>

Sigmoïde : avec `e = np.exp(-np.abs(z))`, `1 / (1 + e)` si `z >= 0`, `e / (1 + e)` sinon. Tanh : `np.tanh` est déjà stable. Softplus : $\max(u, 0) + \log(1 + e^{-|u|})$ avec $u = \beta z$, divisé par $\beta$, sauf là où $u >$ `threshold` (on rend `z`) ; protège l'argument de la branche écartée. Les dérivées réutilisent la sigmoïde stable.

</details>
<details><summary>Indice 3</summary>

```python
def softplus(z, beta=1.0, threshold=20.0):
    z = np.array(z, dtype=float)
    bz = beta * z
    linear = bz > threshold                        # strictly above: PyTorch's rule
    safe = np.where(linear, 0.0, bz)               # never exp of a large number
    ...                                            # the smooth formula on `safe`, divided by beta
    return np.where(linear, z, smooth)
```

`sigmoid` et `softplus_derivative` suivent la même idée ; `tanh_derivative` vaut `1 - t * t` avec `t = np.tanh(z)`.

</details>

### Ex 17.14 — ELU, SiLU, GELU et le registre get_activation 🔨

<details><summary>Indice 1</summary>

ELU : la même protection de branche que pour softplus. SiLU : $z\,\sigma(z)$, avec ta sigmoïde stable. GELU : $z\,\Phi(z)$, avec `math.erf` vectorisé. Le registre : un dictionnaire nom → couple.

</details>
<details><summary>Indice 2</summary>

`np.expm1(np.minimum(z, 0))` pour la branche négative de l'ELU ; `_erf = np.vectorize(math.erf, otypes=[float])` une fois au niveau du module ; la dérivée exacte de GELU, $\Phi(z) + z\,\varphi(z)$ ; celle de l'approximation, par la règle de la chaîne sur $\tanh(u(z))$, avec $1 - \tanh^2$. Vérifie `approximate` au début des deux fonctions. Pour le registre, mets le dictionnaire **après** la définition de toutes les fonctions, et un alias `"swish"`.

</details>
<details><summary>Indice 3</summary>

```python
_REGISTRY = {"identity": (identity, identity_derivative), ..., "gelu": (gelu, gelu_derivative)}
_ALIASES = {"swish": "silu"}

def get_activation(name):
    key = name.lower()
    ...                                            # ValueError listing the ten names if key is unknown
    return _REGISTRY[_ALIASES.get(key, key)]
```

</details>

### Ex 17.15 — Softmax et log_softmax stables (astuce log-sum-exp) 🔨

<details><summary>Indice 1</summary>

Relis les principes 3 et 4 de la section « Calculer sans débordement » de la fiche, et ton `logsumexp` du ch. 13.

</details>
<details><summary>Indice 2</summary>

Convertis en un nouveau tableau `float64`, refuse `z.ndim == 0`, soustrais `np.max(z, axis=axis, keepdims=True)`, puis normalise avec `np.sum(..., axis=axis, keepdims=True)`. Pour `log_softmax`, garde les écarts au maximum et soustrais le logarithme de la somme de leurs exponentielles.

</details>
<details><summary>Indice 3</summary>

```python
def log_softmax(z, axis=-1):
    z = np.array(z, dtype=float)
    ...                                            # ValueError for a single number
    shifted = z - np.max(z, axis=axis, keepdims=True)
    return shifted - ...                           # the log of the sum of exp(shifted) along axis, keepdims
```

</details>

### Ex 17.16 — La Jacobienne de la softmax en NumPy 🔨

<details><summary>Indice 1</summary>

Écris d'abord la Jacobienne d'un seul exemple, $\mathrm{diag}(\mathbf{p}) - \mathbf{p}^\top\mathbf{p}$ (fiche §17.8), puis généralise au batch sans boucle.

</details>
<details><summary>Indice 2</summary>

Calcule `p` avec la softmax stable le long du dernier axe. Pour un batch `(n, K)`, `p[..., :, None]` a la forme `(n, K, 1)` et `p[..., None, :]` la forme `(n, 1, K)` : leur produit donne les `(n, K, K)` produits $p_i p_j$. `np.eye(K)` se diffuse sur le batch. Les points de suspension `...` font marcher la même expression pour `(K,)`.

</details>
<details><summary>Indice 3</summary>

```python
def softmax_derivative(z):
    z = np.array(z, dtype=float)
    ...                                            # ValueError unless z.ndim is 1 or 2
    ...                                            # p: the stable softmax along the last axis
    return p[..., :, None] * np.eye(z.shape[-1]) - ...   # minus the outer product of p with itself, batched
```

</details>

### Ex 17.17 — Chasse au bug : nan, inf et mauvais axe 🐛

<details><summary>Indice 1</summary>

Lis la fonction ligne par ligne avec les quatre points de la vérification en tête : quelles lignes peuvent produire `inf`, `nan`, une mauvaise normalisation, ou une dérivée prise au mauvais endroit ?

</details>
<details><summary>Indice 2</summary>

Les quatre lignes suspectes : l'exponentielle des logits, l'axe de la somme, le logarithme des probabilités, et la pente de la couche cachée (la convention de la fiche : une dérivée prend la **pré-activation**).

</details>
<details><summary>Indice 3</summary>

Le point 1 vient de `np.exp(logits)` : soustrais d'abord le maximum de chaque ligne, `logits - logits.max(axis=1, keepdims=True)`. Pour les trois autres points, les critères : sur un batch `(n, K)`, quel axe porte les classes ? un logarithme d'une probabilité qui s'arrondit à 0, que vaut-il, et comment le calculer sans passer par la probabilité ? la pente de tanh doit-elle être calculée à partir de `z1` ou de `h` ?

</details>

### Ex 17.18 — Reproduire la galerie de la figure 17.22, dérivées comprises 🎨

<details><summary>Indice 1</summary>

`plt.subplots(2, 5)` et une boucle sur `NAMES_18` avec `get_activation` ; pour les valeurs, une grille fine et `np.argmin`, `np.min`, `np.max`, `np.argmax`.

</details>
<details><summary>Indice 2</summary>

`fine = np.linspace(-4, 4, 800_001)` a un pas de $10^{-5}$. `fine[np.argmin(silu(fine))]` donne l'abscisse du minimum ; pour d), ne garde que `fine[fine > 0]` avant `np.argmax`. Ne mélange pas abscisse et valeur.

</details>
<details><summary>Indice 3</summary>

```python
fig, axes = plt.subplots(2, 5, figsize=(14, 5.4), sharex=True)
for ax, name in zip(axes.ravel(), NAMES_18):
    f, df = mylearn.nn.activations.get_activation(name)
    ...                                            # plot f solid and df dashed on z, set the title and ylim
fine = np.linspace(-4, 4, 800_001)
zmin_silu_18 = float(fine[np.argmin(mylearn.nn.activations.silu(fine))])
```

Les trois autres valeurs se lisent de la même façon.

</details>

### Ex 17.19 — Softmax et température : prédire la distribution 🔮

<details><summary>Indice 1</summary>

Diviser les scores par $T$ multiplie leurs écarts par $1/T$ ; relis la température au §17.8 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pour a) et b), seuls les écarts au plus grand score comptent : $p_i \propto e^{(z_i - z_{\max})/T}$. Pour c), une distribution très concentrée a une entropie proche de 0, une distribution presque uniforme une entropie proche de $\log_2 5$. Pour d), que devient l'ordre des scores divisés par un nombre négatif ?

</details>
<details><summary>Indice 3</summary>

Pour a), à $T = 0{,}5$, les écarts au premier score deviennent $0, -3, -4, -6, -8$ : compare $e^{0}$ à la somme des quatre autres exponentielles. Pour b), écris les écarts au premier score divisés par 5, et estime la plus petite probabilité. Pour c), estime l'entropie aux températures où la distribution cesse d'être dominée par une classe. Pour d), applique la règle à $T = -1$. Ne tranche qu'après ces estimations.

</details>

### Ex 17.20 — Quatre activations de scikit-learn sur les lunes et les spirales 📦

<details><summary>Indice 1</summary>

Une boucle sur les cinq graines, un `MLPClassifier` neuf à chaque tour, `fit_quietly`, puis `score` sur le test.

</details>
<details><summary>Indice 2</summary>

`Xtr, Xte, ytr, yte = DATA_20[dataset]` ; les réglages sont exactement ceux de l'énoncé (`(16,)`, `solver="lbfgs"`, `max_iter=3000`, `random_state=seed`) : la vérification compare des moyennes à 3 décimales. Rends un `float` (la moyenne de NumPy convertie).

</details>
<details><summary>Indice 3</summary>

```python
def mean_accuracy_20(dataset, activation):
    Xtr, Xte, ytr, yte = DATA_20[dataset]
    scores = []
    for seed in range(5):
        model = ...                                # MLPClassifier: 16 hidden neurons, lbfgs, max_iter=3000, random_state=seed
        ...                                        # fit with fit_quietly, append the test accuracy
    return float(np.mean(scores))
```

</details>

### Ex 17.21 — Diagnostiquer la saturation sur des histogrammes de pré-activations 📈

<details><summary>Indice 1</summary>

Pour a) et b), regarde où est **centrée** chaque distribution et quelle **largeur** elle a, par rapport à la courbe de $\sigma'$ (figure de la saturation, §17.6).

</details>
<details><summary>Indice 2</summary>

Une couche est saturée quand la plupart de ses $z$ tombent là où $\sigma'$ est presque nulle (loin de 0). Une couche est « presque linéaire » quand ses $z$ restent dans une plage étroite, où la courbe de $\sigma$ ressemble à une droite. `saturation_21` : la moyenne de `sigmoid_derivative(z)` et la moyenne du test `< 0.01`.

</details>
<details><summary>Indice 3</summary>

Pour a), compare les quatre distributions : celle qui est centrée loin de 0, ou celle qui est large mais centrée en 0 ? Une distribution large garde beaucoup de valeurs près de 0. Pour b), sur quel intervalle de $z$ la sigmoïde est-elle presque une droite ? Quelle distribution garde presque toutes ses valeurs dans cet intervalle ? Pour le code :

```python
def saturation_21(z):
    slope = np.asarray(mylearn.nn.activations.sigmoid_derivative(z), dtype=float)
    return float(slope.mean()), ...                # the fraction of slopes below 0.01
```

</details>

### Ex 17.22 — Une batterie de tests paramétrés pour toutes les activations 🛠️

<details><summary>Indice 1</summary>

Lis les cinq versions buggées : pour chacune, quelle propriété du contrat ne tient plus ? Un test par propriété, paramétré sur les quatre noms.

</details>
<details><summary>Indice 2</summary>

Cinq propriétés : la dérivée égale la pente numérique (loin du coude de ReLU) ; chaque fonction est croissante ; les valeurs restent finies pour $\pm 1\,000$ sans avertissement (`np.errstate(over="raise", ...)`) ; l'entrée n'est pas modifiée ; $\mathrm{ReLU}'(0) = 0$. Deux décorateurs `parametrize` empilés testent à la fois `f` et `f′`.

</details>
<details><summary>Indice 3</summary>

```python
@pytest.mark.parametrize("name", ["relu", "sigmoid", "tanh", "softplus"])
def test_derivative_matches_central_differences(name):
    f, df = act(name)
    z = ...                                        # a grid away from 0
    h = 1e-5
    assert np.allclose(df(z), (f(z + h) - f(z - h)) / (2 * h), rtol=1e-6, atol=1e-7)
```

Les quatre autres tests suivent le même modèle ; range-les tous dans `my_tests_22`.

</details>

### Ex 17.23 — Produit des dérivées sur 10 couches : sigmoïde, tanh, ReLU, GELU 🔬

<details><summary>Indice 1</summary>

C'est la passe avant de ton `mlp_forward`, sans biais, en gardant à chaque couche $f'(\mathbf{z})$ au lieu du cache.

</details>
<details><summary>Indice 2</summary>

`f, df = mylearn.nn.activations.get_activation(name)`, puis, pour chaque `W` de `weights_23(name)` : `z = a @ W`, garder `df(z)`, puis `a = f(z)`. Rends la liste des dix tableaux.

</details>
<details><summary>Indice 3</summary>

```python
def slopes_23(name):
    f, df = mylearn.nn.activations.get_activation(name)
    a, slopes = X_23, []
    for W in weights_23(name):
        ...                                        # z, then keep df(z), then a = f(z)
    return slopes
```

</details>

### Ex 17.24 — Défi spirales : au moins 97 % avec au plus 2 000 paramètres 🏆

<details><summary>Indice 1</summary>

Pars de ce qu'a montré 📦 17.20 sur les spirales, puis cherche par validation croisée sur `X_24` seulement : une activation, une taille de réseau sous la limite, un solveur adapté à un petit dataset.

</details>
<details><summary>Indice 2</summary>

`lbfgs` converge bien mieux que `adam` sur 800 points. Un réseau qui atteint une accuracy de 1 sur l'entraînement **colle** au bruit : essaie `alpha` (la pénalité L2) sur une grille de $10^{-4}$ à 1. Les rayons des spirales sont inférieurs à 1 : une standardisation dans un `Pipeline` peut aider.

</details>
<details><summary>Indice 3</summary>

```python
def build_24():
    return make_pipeline(StandardScaler(),
                         MLPClassifier((..., ...), activation=..., solver="lbfgs", alpha=..., max_iter=3000,
                                       random_state=0))
```

Choisis les trois réglages par `cross_val_score` (5 folds) sur `X_24`, et compte les paramètres : $2n_1 + n_1 + n_1 n_2 + n_2 + n_2 + 1$ pour deux couches cachées et une sortie binaire.

</details>
