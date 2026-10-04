# 7 · Classification — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 🛠️ ⚖️ Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 7.Q1 — Label, prédiction, vérité terrain : le vocabulaire

<details><summary>Indice 1</summary>

Relis le début du §7.1 de la fiche : la liste des noms du label, puis celui de la classe choisie par le modèle.

</details>
<details><summary>Indice 2</summary>

Pour b), demande-toi qui mire les œufs, et si un humain peut se tromper. Pour c), relis le but de l'entraînement : sur quelles données compte-t-on les bonnes réponses ?

</details>
<details><summary>Indice 3</summary>

a) Pour chaque expression, demande-toi d'où vient la classe : de l'expert, ou du modèle ? Le label, la vérité terrain et la valeur réelle nomment tous la classe que l'expert a fixée ; la valeur prédite, elle, sort du modèle : la réponse est **C**. Pour b), le mirage est fait par un humain : un humain peut-il se tromper sur un œuf ambigu, ou en fin de journée ? Pour c), sur quels œufs juge-t-on un classifieur : ceux qu'il a déjà vus, ou des œufs nouveaux (ch. 1) ? Pour d), pose la question de a) au filtre anti-spam : qui fixe la classe d'un e-mail, et qui la prédit ?

</details>

### 7.Q2 — Binaire, multi-classe ou multi-étiquette ? Cinq situations

<details><summary>Indice 1</summary>

Relis les trois définitions du §7.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pose deux questions à chaque tâche : combien de classes possibles ? Un même exemple peut-il en recevoir plusieurs à la fois ?

</details>
<details><summary>Indice 3</summary>

a) Un e-mail est un spam ou ne l'est pas : deux classes possibles, une seule par e-mail, c'est un problème **binaire** (`"B"`). Pour b) à e), pose les deux questions de l'indice 2 : combien de classes possibles, et un même exemple peut-il en recevoir plusieurs à la fois ? Le critère du multi-étiquette est le nombre de classes **par exemple**, pas le nombre de classes en tout. Pour f), relis au §7.3 de la fiche ce que désignent *yolker* et *quitter* : les deux peuvent-ils être vrais du même œuf ?

</details>

### 7.Q3 — Régions et frontières de décision

<details><summary>Indice 1</summary>

Relis le §7.2.1 de la fiche, et regarde la figure des trois situations.

</details>
<details><summary>Indice 2</summary>

Pour b), une classe que le classifieur peut prédire peut-elle n'occuper aucun morceau du plan ? Pour d), pense à ce que produit `predict` pour chaque point. Pour e), essaie de faire légèrement tourner une droite qui sépare deux nuages bien écartés.

</details>
<details><summary>Indice 3</summary>

a) Regarde le panneau (b) de la figure des trois situations de la fiche (ou la figure 7.3 du livre) : la frontière y est une courbe, et un seul contre-exemple suffit à contredire « toujours une droite » : **faux**. Pour b), une classe que le classifieur peut prédire occupe au moins un morceau du plan : compte les classes. Pour c), imagine deux paquets d'œufs viables séparés par une bande de *quitters*. Pour d), combien de classes `predict` rend-il pour un point, même sur la frontière ? Pour e), fais tourner très légèrement une droite qui sépare deux nuages bien écartés : sépare-t-elle encore les deux classes ? Pour f), demande-toi ce qu'une frontière très tortueuse apprend des exemples d'entraînement, y compris de leurs erreurs.

</details>

### 7.Q4 — Classes qui se recouvrent : probabilités et politique de seuil

<details><summary>Indice 1</summary>

Relis le seuil et l'encadré 🧮 « le seuil qui minimise le coût moyen » au §7.2.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

La région « fécondé » est l'ensemble des points où $P \ge t$. Que devient cet ensemble quand $t$ passe de 0,5 à 0,2 ? Suis un œuf fécondé déjà déclaré positif, puis un œuf non fécondé déjà déclaré positif.

</details>
<details><summary>Indice 3</summary>

a) Un œuf fécondé déclaré positif au seuil 0,5 a $P \ge 0{,}5$, donc aussi $P \ge 0{,}2$ : il reste déclaré positif. Les vrais positifs ne peuvent que rester ou s'ajouter, et le nombre d'œufs fécondés ne change pas : le recall ne peut pas baisser, **vrai**. Pour b), suis de la même façon un œuf **non** fécondé déjà déclaré positif. Pour c), compare les deux ensembles $\{P \ge 0{,}5\}$ et $\{P \ge 0{,}2\}$ : l'un contient-il l'autre ? Pour d), $t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ avec $C_{FP} = 1$ et $C_{FN} = 7$. Pour e), relis la fin du §7.2.1 de la fiche : de quoi dépend la place de la frontière, et qui connaît ces coûts ?

</details>

### 7.Q5 — Cinq mesures par œuf : ce que change (ou non) la dimension

<details><summary>Indice 1</summary>

Relis le §7.3 de la fiche : la dimension, la dimension d'une frontière, et les deux nuances.

</details>
<details><summary>Indice 2</summary>

Une dimension par feature. En dimension $d$, une frontière a une dimension de moins.

</details>
<details><summary>Indice 3</summary>

a) Une dimension par feature, et la classe n'est pas une feature : cinq mesures par œuf, donc un espace de dimension **5**. Pour b), écris la distance entre deux œufs avec leurs cinq coordonnées : la formule change-t-elle de forme, ou seulement de nombre de termes ? Pour c), chaque feature ajoute un terme à chaque distance et à chaque moyenne : qu'en déduis-tu pour le temps de calcul et la mémoire ? Pour d), une frontière a une dimension de moins que l'espace qu'elle découpe (une droite dans le plan) : applique cette règle à a). Pour e), pense aux nuages de points des features prises deux à deux, et aux projections en 2D (PCA, ch. 12).

</details>

### 7.Q6 — Un-contre-tous : combien de modèles, quelle décision ?

<details><summary>Indice 1</summary>

Relis le §7.4.1 de la fiche et le panneau (b) de la figure des quatre classes.

</details>
<details><summary>Indice 2</summary>

Pour a), à quelle question répond chacun des classifieurs, et combien de questions de ce genre faut-il poser ? Pour b) et d), la règle de décision regarde-t-elle le signe des scores ? Pour c), additionne les probabilités de b).

</details>
<details><summary>Indice 3</summary>

a) Un-contre-tous : un classifieur par classe, qui répond à « cette classe, ou une autre ? » ; 7 classes, donc **7** modèles. Pour b), la règle prend la classe du plus grand score : range les quatre probabilités, en regardant bien les deux plus grandes, qui sont proches. Pour c), additionne les quatre probabilités de b) : obtiens-tu 1 ? Pour d), même règle qu'en b), même si tous les scores sont négatifs : le plus grand de quatre nombres négatifs est le plus proche de 0. Pour e), relis l'encadré ⚠️ sur *binary relevance* : en multi-étiquette, combien de labels garde-t-on pour un exemple ?

</details>

### 7.Q7 — Un-contre-un : duels, votes et coût

<details><summary>Indice 1</summary>

Relis le §7.4.2 de la fiche : la formule du nombre de duels, le mini-exemple et l'encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

$\binom{K}{2} = \frac{K(K-1)}{2}$. Pour e), compare $\frac{K(K-1)}{2}$ et $\frac{K^2}{2}$ : développe le premier.

</details>
<details><summary>Indice 3</summary>

a) Un duel par paire de classes : $\binom{6}{2} = \frac{6 \times 5}{2} = 15$ duels. Pour b), relis le §7.4.2 de la fiche : sur quels échantillons le duel A contre B s'entraîne-t-il ? Pour c), calcule $\frac{3 \times 2}{2}$ et compare-le au nombre de modèles de l'un-contre-tous, un par classe. Pour d), repère les classes à égalité en tête, puis applique la règle de `mylearn` : la classe de plus petit indice. Pour e), développe $\frac{K(K-1)}{2}$ et compare-le à $\frac{K^2}{2}$ : lequel est le plus grand ? Pour f), compte les voix de la vraie classe quand ses duels sont justes, puis le nombre maximal de voix de chacune des autres, qui ont toutes perdu contre elle.

</details>

### 7.Q8 — Ce que k-means ne peut pas deviner tout seul

<details><summary>Indice 1</summary>

Relis « Sans labels : k-means » et « Combien de clusters ? » au §7.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Apprentissage supervisé ou non ? Qui choisit $k$, et quand ? Que fait l'inertie quand $k$ augmente ?

</details>
<details><summary>Indice 3</summary>

a) k-means est non supervisé : il ne voit que les points, jamais leurs labels. L'affirmation est donc **fausse**. Pour b), qui fixe $k$, et quand : l'algorithme pendant l'entraînement, ou toi avant ? Relis la différence entre paramètre et hyperparamètre (ch. 1). Pour c), combien de clusters k-means rend-il, quel que soit le nombre de groupes des données ? Pour d), relis « minimum local » dans l'encadré 🧮 sur l'algorithme de Lloyd : le point d'arrivée dépend-il du point de départ ? Pour e), que vaut l'inertie quand chaque point est seul dans son cluster ? Pour f), relis « Combien de clusters ? » au §7.5 de la fiche.

</details>

### 7.Q9 — Densité d'échantillons quand les features s'accumulent

<details><summary>Indice 1</summary>

Relis la formule de la densité au §7.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

$\rho = \frac{n}{b^d}$, ici avec $n = 20$ et $b = 4$.

</details>
<details><summary>Indice 3</summary>

a) En dimension 1, l'axe a 4 cases : $\rho = \frac{20}{4} = 5$ échantillons par case. Pour b) et c), compte d'abord les cases, $4^d$, puis divise 20 par ce nombre (4 décimales en c). Pour d) et e), regarde ce que touche chaque changement dans $\rho = \frac{n}{b^d}$ : le numérateur ou le dénominateur, et de quel facteur ? Pour f), imagine une case vide : de quels exemples le classifieur dispose-t-il pour décider de la classe des points qui y tombent ?

</details>

### 7.Q10 — Bénédiction de la structure : vrai ou faux justifié

<details><summary>Indice 1</summary>

Relis la fin du §7.6 de la fiche (la bénédiction de la non-uniformité) et la liste des parades.

</details>
<details><summary>Indice 2</summary>

Pour 2, combien d'images en noir et blanc de 784 pixels existe-t-il ? Combien ressemblent à un chiffre ? Pour 3, le livre emploie un mot précis pour la malédiction et la bénédiction.

</details>
<details><summary>Indice 3</summary>

1 : regarde où sont les chiffres manuscrits, les visages ou les manchots dans l'espace de leurs features : partout à la fois, ou dans de petites régions, près de surfaces de faible dimension (fiche §7.6) ? Ils se concentrent : l'affirmation est **fausse**. Pour 2, compte les images possibles ($2^{784}$ en noir et blanc), puis demande-toi quelle part d'entre elles ressemble à un chiffre. Pour 3, un théorème se démontre : le livre démontre-t-il la bénédiction de la structure, ou la constate-t-il sur des données réelles ? Pour 4, que devient la densité quand on retire une feature qui n'apporte que du bruit (7.30) ? Pour 5, regarde la figure 7.23 du livre : où la densité compte-t-elle ?

</details>

### 7.Q11 — Géométrie déroutante en grande dimension

<details><summary>Indice 1</summary>

Relis le §7.6.1 de la fiche : la peau de l'orange, l'hyper-orange et l'encadré ⚠️ sur la sphère hérissée.

</details>
<details><summary>Indice 2</summary>

Pour a), applique $1 - (1 - \varepsilon)^d$. Pour c), le coin du cube de côté 1 centré en 0 a toutes ses coordonnées égales à $\pm \frac{1}{2}$. Pour d), relis la concentration des distances.

</details>
<details><summary>Indice 3</summary>

a) Le volume d'une boule est proportionnel à $r^d$ : la boule intérieure, de rayon 0,97, garde $0{,}97^{100} \approx 0{,}048$ du volume, et la peau en contient $1 - 0{,}048 \approx 0{,}952$, environ 95 %. C'est moins de 99 % : **faux**. Pour b), calcule $r(10) = \sqrt{10} - 1$ et compare-le à 2 (∂ 7.5). Pour c), le coin du cube de côté 1 centré en 0 a toutes ses coordonnées égales à $\pm\frac{1}{2}$ : écris sa distance au centre en fonction de $d$. Pour d), relis la concentration des distances au §7.6.1 de la fiche : que devient l'écart entre la distance au plus proche voisin et la distance moyenne quand $d$ grandit ? Pour e), suis, quand $d$ grandit, la distance du centre au milieu d'une face de la boîte, puis celle du centre à un coin : laquelle change ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 7.R1 — Ch. 6 : entropie d'un cluster pur et d'un cluster mélangé

<details><summary>Indice 1</summary>

Relis la définition de l'entropie au ch. 6 (§6.7 de sa fiche) : $H(p) = -\sum_i p_i \log_2 p_i$, avec $0 \log_2 0 = 0$.

</details>
<details><summary>Indice 2</summary>

Transforme les comptes en proportions. Un cluster pur a une seule proportion égale à 1 ; l'entropie maximale correspond à la distribution uniforme.

</details>
<details><summary>Indice 3</summary>

a) Une seule espèce, de proportion 1 : $H = -1 \times \log_2 1 = 0$ bit. b) Les proportions sont $[\frac{1}{2}, \frac{1}{2}]$. c) $\frac{1}{2} \times 1 + 2 \times \frac{1}{4} \times 2$. d) $-(0{,}7 \log_2 0{,}7 + 0{,}2 \log_2 0{,}2 + 0{,}1 \log_2 0{,}1)$. e) $\log_2 3$. Pour f), c'est l'entropie conditionnelle du ch. 6.

</details>

### 7.R2 — Ch. 4 : probabilité a posteriori « fécondé » par la règle de Bayes

<details><summary>Indice 1</summary>

Relis la règle de Bayes du ch. 4 (§4.4.1 et §4.4.2 de sa fiche), ou sa forme au §7.2.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

$P = \frac{\pi f_1}{\pi f_1 + (1 - \pi) f_0}$, avec $\pi = 0{,}3$. Pour d), écris $P = 0{,}5$, c'est-à-dire $\pi f_1 = (1 - \pi) f_0$.

</details>
<details><summary>Indice 3</summary>

a) $\frac{0{,}3 \times 0{,}08}{0{,}3 \times 0{,}08 + 0{,}7 \times 0{,}04} = \frac{0{,}024}{0{,}052} \approx 0{,}462$. b) Compare a) au seuil de 0,5. c) Même calcul qu'en a), avec 0,5 et 0,5. d) $\frac{f_1}{f_0} = \frac{1 - \pi}{\pi}$.

</details>

### 7.R3 — 0B : développer ‖a − b‖² avec le produit scalaire

<details><summary>Indice 1</summary>

$\lVert \mathbf{v} \rVert^2 = \mathbf{v} \cdot \mathbf{v}$, la somme des carrés des coordonnées (0B).

</details>
<details><summary>Indice 2</summary>

$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + a_3 b_3$. Pour d), calcule d'abord $\mathbf{a} - \mathbf{b}$.

</details>
<details><summary>Indice 3</summary>

a) $\lVert \mathbf{a} \rVert^2 = 3^2 + (-1)^2 + 2^2 = 14$. b) $a_1 b_1 + a_2 b_2 + a_3 b_3$, attention au signe de $a_2$. c) Même méthode qu'en a), avec les coordonnées de $\mathbf{b}$. d) $\mathbf{a} - \mathbf{b} = (2, -3, 2)$, puis la somme des carrés de ses coordonnées. Pour e), développe $(\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b})$ ; pour des matrices $\mathbf{A}$ et $\mathbf{B}$ (un point par ligne), les produits scalaires de toutes les paires forment $\mathbf{A} \mathbf{B}^\top$ (encadré 🧮 du §7.5 de la fiche).

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 7.1 — Compter les classifieurs OvR et OvO ✏️

<details><summary>Indice 1</summary>

Relis le §7.4.2 de la fiche : $N_{\text{OvR}} = K$ et $N_{\text{OvO}} = \frac{K(K-1)}{2}$.

</details>
<details><summary>Indice 2</summary>

Pour e), cherche $K$ tel que $K(K - 1) > 20\,000$ : $\sqrt{20\,000} \approx 141$, puis essaie les entiers autour. Pour g), combien d'images voit un duel ? Combien y a-t-il de duels ?

</details>
<details><summary>Indice 3</summary>

a) $N_{\text{OvR}} = 3$ et $N_{\text{OvO}} = \frac{3 \times 2}{2} = 3$ : $[3, 3]$. b) à d) Même calcul avec $K = 10$, 26 et 100. e) Compare $141 \times 140$ et $142 \times 141$ à 20 000. f) $10 \times 50\,000$. g) Un duel voit $2 \times 5\,000$ images : multiplie par le nombre de duels de b) (autre lecture : chaque image participe aux $K - 1$ duels de sa classe). h) Un-contre-tous : $10 \times 50\,000^2$ ; un-contre-un : le nombre de duels $\times\, 10\,000^2$ ; fais le rapport.

</details>

### Ex 7.2 — Dépouiller les votes d'un un-contre-un à quatre classes ✏️

<details><summary>Indice 1</summary>

Chaque duel donne une voix à son vainqueur, et une seule.

</details>
<details><summary>Indice 2</summary>

Parcours le tableau colonne par colonne et ajoute 1 à la classe gagnante. Pour g), combien de voix a une classe qui gagne tous ses duels ? Combien au plus pour chacune des autres, qui ont toutes perdu contre elle ?

</details>
<details><summary>Indice 3</summary>

a) Parcours le premier tableau colonne par colonne : A gagne A-C ; B gagne A-B, B-C et B-D ; C ne gagne rien ; D gagne A-D et C-D, d'où $[1, 3, 0, 2]$. b) La classe qui a le plus de voix en a). c) Additionne les voix de a), ou compte les duels. d) Même dépouillement pour le second tableau. e) Repère les classes à égalité en tête, puis garde celle de plus petit indice. f) Pour les classes à égalité, regarde qui a gagné le duel qui les a opposées. g) Une classe qui gagne ses $K - 1$ duels a $K - 1$ voix ; chacune des autres a perdu au moins son duel contre elle, donc a au plus $K - 2$ voix : l'une d'elles peut-elle passer devant ? h) 6 voix pour 4 classes : combien chacune en aurait-elle en cas d'égalité parfaite ?

</details>

### Ex 7.3 — Une itération de k-means à la main ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur l'algorithme de Lloyd et son mini-exemple en dimension 1 (§7.5 de la fiche).

</details>
<details><summary>Indice 2</summary>

Fais un tableau : une ligne par point, deux colonnes pour les distances au carré aux deux centres. L'inertie est la somme, sur les points, de la distance au carré à **son** centre.

</details>
<details><summary>Indice 3</summary>

a) Distances au carré à $c_1 = (1, 1)$ et à $c_2 = (1, 3)$ : $P_1$ 0 et 4, $P_2$ 1 et 5, $P_3$ 4 et 0, $P_4$ 25 et 17, $P_5$ 41 et 29, $P_6$ 45 et 37. Chaque point rejoint le centre le plus proche : $[1, 1, 2, 2, 2, 2]$. b) Additionne, pour chaque point, sa distance au carré à **son** centre, la plus petite de ses deux colonnes. c) Chaque nouveau centre est la moyenne des points de son cluster, coordonnée par coordonnée. d) La somme de b), avec les centres de c) mais l'affectation de a). e) Refais le tableau de a) avec les centres de c) : quel point est maintenant plus près de l'autre centre ? f) et g) Refais c), puis b), avec la nouvelle affectation. h) Refais le tableau avec les centres de f). i) Range les inerties dans l'ordre des étapes : b), d), celle de la deuxième affectation (encore avec les centres de c)), puis g).

</details>

### Ex 7.4 — Densité d'échantillons et nombre d'œufs nécessaires ✏️

<details><summary>Indice 1</summary>

Relis la formule $\rho = \frac{n}{b^d}$ et l'encadré ⚠️ « Une densité n'est pas une probabilité » (§7.6 de la fiche).

</details>
<details><summary>Indice 2</summary>

Pour b) à d), $n = \rho\, b^d$. Pour f), un œuf tombe hors d'un petit cube donné avec la probabilité $\frac{124}{125}$, et les 10 œufs sont indépendants. Pour g), additionne, sur les 125 cubes, la probabilité que chacun soit occupé.

</details>
<details><summary>Indice 3</summary>

a) En dimension 1 : $\frac{360}{6} = 60$ œufs par case ; continue avec $6^2$, $6^3$ et $6^4$ cases. b) à d) $n = \rho\, b^d$, avec la densité, le nombre de cases par axe et la dimension de chaque question. e) Compare $6^7$ et $6^8$ à un million. f) $\left(\frac{124}{125}\right)^{10}$. g) $125 \times (1 - \text{f})$ : l'espérance d'une somme est la somme des espérances.

</details>

### Ex 7.5 — Le rayon de l'hyper-orange : r(d) = √d − 1 ∂

<details><summary>Indice 1</summary>

La distance de l'origine à un point de coordonnées $(x_1, \ldots, x_d)$ est $\sqrt{x_1^2 + \cdots + x_d^2}$.

</details>
<details><summary>Indice 2</summary>

Toutes les coordonnées du centre d'un ballon valent $\pm 1$. Deux boules qui se touchent : la distance entre leurs centres est la somme de leurs rayons.

</details>
<details><summary>Indice 3</summary>

1. Centre du ballon à $\sqrt{d \times 1^2} = \sqrt{d}$ de l'origine, et la distance entre les centres de deux boules qui se touchent est la somme de leurs rayons : $r + 1 = \sqrt{d}$. a) $r(2) = \sqrt{2} - 1$, à arrondir à 3 décimales. b) Même formule avec $d = 3$. c) Un ballon par coin : un coin choisit $+1$ ou $-1$ pour chacune de ses 12 coordonnées ; compte les choix possibles. d) $r \ge 1{,}5$ s'écrit $\sqrt{d} \ge 2{,}5$ : élève au carré, puis prends le plus petit **entier** qui convient. e) Même méthode avec $\sqrt{d} > 3{,}5$. f) $\sqrt{50} - 1$, avec $\sqrt{50} = 5\sqrt{2}$. g) Le centre du ballon le plus proche du milieu de la face diffère de lui de 1 sur la première coordonnée et de 1 sur chacune des $d - 1$ autres.

</details>

### Ex 7.6 — Boule dans un cube : rapport des volumes par récurrence ∂

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur le volume de la boule (§7.6.1 de la fiche) : il n'y a qu'à appliquer la récurrence, deux dimensions à la fois.

</details>
<details><summary>Indice 2</summary>

$V_3 = \frac{4\pi}{3}$, puis $V_4 = \frac{2\pi}{4} V_2$ et $V_5 = \frac{2\pi}{5} V_3$. Garde $\pi$ en symbole jusqu'au bout. Pour f), voir le §7.6.1 de la fiche (la peau).

</details>
<details><summary>Indice 3</summary>

a) $V_4 = \frac{2\pi}{4} V_2 = \frac{2\pi}{4} \times \pi = \frac{\pi^2}{2} \approx 4{,}935$. b) Même méthode : $V_5 = \frac{2\pi}{5} V_3$, avec $V_3 = \frac{2\pi}{3} V_1$. c) $q_4 = \frac{V_4}{2^4}$. d) Monte de deux en deux jusqu'à $V_{10}$ ($V_6$, $V_8$, puis $V_{10}$), divise par $2^{10} = 1\,024$, puis multiplie par 100. e) Calcule aussi les dimensions impaires, $V_3$, $V_5$ et $V_7$, puis compare les huit valeurs. f) La peau est ce qui reste quand on retire la boule de rayon 0,95 : $1 - 0{,}95^{100}$. g) Écris $2^d = 4 \times 2^{d-2}$. h) $\frac{2\pi}{d} < 1$ dès que $d > 2\pi$ ; les facteurs suivants sont de plus en plus petits.

</details>

### Ex 7.7 — La frontière du centroïde le plus proche est une droite ∂

<details><summary>Indice 1</summary>

Développe les deux membres avec l'identité de 7.R3 : les termes $\lVert \mathbf{x} \rVert^2$ se simplifient.

</details>
<details><summary>Indice 2</summary>

Une équation $\mathbf{w} \cdot \mathbf{x} = c$ avec $\mathbf{w} \ne \mathbf{0}$ décrit une droite (un hyperplan) perpendiculaire à $\mathbf{w}$. Vérifie que le milieu $\frac{\boldsymbol{\mu}_0 + \boldsymbol{\mu}_1}{2}$ satisfait l'équation. Pour 4, un point équidistant de $\boldsymbol{\mu}_0$ et $\boldsymbol{\mu}_1$, et de $\boldsymbol{\mu}_1$ et $\boldsymbol{\mu}_2$, l'est aussi de…

</details>
<details><summary>Indice 3</summary>

1. $\lVert \mathbf{x} - \boldsymbol{\mu}_0 \rVert^2 = \lVert \mathbf{x} \rVert^2 - 2\,\boldsymbol{\mu}_0 \cdot \mathbf{x} + \lVert \boldsymbol{\mu}_0 \rVert^2$ ; écris de même la distance à $\boldsymbol{\mu}_1$, égale les deux, et rassemble tout d'un même côté. 2. Lis $\mathbf{w}$ dans l'équation obtenue, puis vérifie le milieu comme le propose l'indice 2. 3. $2(4, 2) \cdot (x, y) + 5 - 41 = 0$, à simplifier ; pour $(4, 0)$, regarde de quel côté de la frontière il tombe, celui de $\boldsymbol{\mu}_0$ ou celui de $\boldsymbol{\mu}_1$, ou compare directement ses deux distances au carré. 4. Le point commun aux médiatrices est à la même distance des trois centroïdes : que peux-tu tracer, centré sur lui, qui passe par les trois ? 5. Fais la liste de ce que le classifieur retient des données d'entraînement : qu'est-ce qui distingue ici les deux classes, et qui manque à cette liste ? Où faudrait-il déplacer la frontière pour classer correctement plus de points de la classe étalée ?

</details>

<a id="reflexion"></a>

## 🗣️ 🛠️ ⚖️ Réflexion et outils

### Ex 7.8 — La malédiction de la dimension en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis le §7.6 de la fiche : la densité, la figure 7.17 du livre et les parades.

</details>
<details><summary>Indice 2</summary>

Une image possible : des invités répartis dans un immeuble dont on multiplie les étages et les pièces. Quelle parade proposer à un agronome ?

</details>
<details><summary>Indice 3</summary>

Plan en cinq phrases : ce qu'il espère (plus de mesures, de meilleures prévisions) ; la densité (150 exemples perdus dans un espace immense) ; ce qui se passe (le modèle devine, ou apprend le bruit) ; la structure qui sauve parfois ; une parade (choisir les mesures utiles, résumer, plus de parcelles, régulariser).

</details>

### Ex 7.9 — Lire la documentation officielle de KMeans (scikit-learn) 🛠️

<details><summary>Indice 1</summary>

La page d'une classe de scikit-learn a toujours les mêmes rubriques : *Parameters*, *Attributes*, *See also*, *Notes*, *Examples*, puis les méthodes. Dans la description d'un paramètre, des mentions « Added in version » et « Changed in version » datent ses évolutions.

</details>
<details><summary>Indice 2</summary>

1 est dans *Notes* ; 2 dans la description du paramètre `n_init` ; 3 compare la description de `tol` et la docstring de ton `mylearn/cluster.py` ; 4 dans la méthode `transform`, puis pense à l'indexation avancée de NumPy (0A) ; 5 à la fin de la description de `n_init`.

</details>
<details><summary>Indice 3</summary>

1, en modèle : dans *Notes*, la phrase « If the algorithm stops before fully converging » dit que les centres ne sont alors pas les moyennes des points de leurs clusters (`labels_` et `cluster_centers_` ne sont pas « cohérents »), puis que `labels_` est quand même réaffecté après la dernière itération, pour coïncider avec `predict` sur les données d'entraînement. Pour 2, la description de `n_init` contient une phrase qui commence par « When `n_init='auto'` » : quelles valeurs de `init` met-elle ensemble, et combien de départs pour chaque groupe ? Pour 3, relis dans ta docstring la formule exacte de l'arrêt : par quelle quantité, calculée sur `X`, `tol` est-il multiplié, et que deviendrait le seuil si tu changeais d'unité ? Pour 4, la description de `transform` donne la forme du tableau ; dans la ligne $i$, quelle colonne correspond au cluster du point $i$ ? L'indexation avancée de 0A prend un élément par ligne avec deux tableaux d'indices de même longueur. Pour 5, les deux mentions sont à la fin de la description de `n_init` ; pour le « pourquoi », imagine un utilisateur dont le code ne fixe pas `n_init` : que verrait-il si la valeur par défaut changeait sans prévenir ? Pour 6, compare ta docstring et la page point par point : le nombre de départs par défaut, la façon de tirer chaque centre de k-means++ (combien de candidats à chaque étape ?), le sort d'un cluster vide, et les paramètres qui n'existent que d'un côté.

</details>

### Ex 7.10 — Qui fixe le seuil ? Œufs, dépistage et coût des erreurs ⚖️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur le seuil qui minimise le coût moyen (§7.2.1 de la fiche).

</details>
<details><summary>Indice 2</summary>

$t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$. Qui est un faux positif ici : un œuf non fécondé déclaré fécondé, ou l'inverse ? Pour 2 à 4, liste les personnes concernées par chaque erreur.

</details>
<details><summary>Indice 3</summary>

1, en modèle : un faux positif est un œuf non fécondé mis en incubation (0,30 €), un faux négatif un œuf fécondé vendu (50 €) ; $t^* = \frac{0{,}30}{0{,}30 + 50} \approx 0{,}006$, moins de 1 % : presque tout œuf un peu douteux part en incubation. Pour en juger, demande-toi ce que ce seuil suppose des probabilités proches de 0 (ch. 3). Pour 2 et 3, fais pour chaque erreur la liste de ceux qui la subissent (l'entreprise, le client, l'animal ; le patient, sa famille, l'hôpital) et de ceux qui ont l'information pour en chiffrer le coût : sont-ce les mêmes personnes ? Pour 4, un même seuil n'a le même effet sur deux groupes que si leurs probabilités veulent dire la même chose : quelles mesures, calculées groupe par groupe, le montreraient (ch. 3) ? Pour 5, demande-toi ce qu'un lecteur de la fiche modèle doit savoir pour ne pas mal utiliser ce seuil.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 7.E1 — OvR ou OvO : lequel choisir, et pourquoi ?

<details><summary>Indice 1</summary>

Relis le §7.4 de la fiche et l'encadré 🕰️ sur les stratégies multi-classes.

</details>
<details><summary>Indice 2</summary>

Compte les modèles pour 50 classes, puis demande-toi comment le coût d'entraînement d'un SVM dépend du nombre d'exemples.

</details>
<details><summary>Indice 3</summary>

50 modèles sur toutes les données, ou 1 225 duels sur deux classes chacun. SVM linéaire : un-contre-tous, simple et rapide. SVM à noyau : un-contre-un (c'est ce que fait `SVC`). Ajoute : un modèle multi-classe natif, si on peut.

</details>

### 7.E2 — Expliquer k-means, ses hypothèses et ses échecs

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur l'algorithme de Lloyd et la section « Quand k-means échoue » (§7.5 de la fiche).

</details>
<details><summary>Indice 2</summary>

Structure ta réponse : ce que fait l'algorithme, ce qu'il minimise, ses hypothèses, un échec, un remède.

</details>
<details><summary>Indice 3</summary>

Affectation, puis mise à jour ; l'inertie ; des clusters ronds, d'étalements proches, des features à la même échelle ; un minimum local ; des groupes allongés ; k-means++ et plusieurs départs ; DBSCAN ou HDBSCAN. Classification : des labels ; clustering : aucun.

</details>

### 7.E3 — Choisir le nombre de clusters sans labels

<details><summary>Indice 1</summary>

Relis « Combien de clusters ? » et l'encadré 🕰️ sur le choix de $k$ (§7.5 de la fiche).

</details>
<details><summary>Indice 2</summary>

Pourquoi l'inertie seule ne suffit-elle pas ? Quels critères pénalisent les découpages inutiles ? Que dit le métier ?

</details>
<details><summary>Indice 3</summary>

Coude de l'inertie, silhouette (ou Davies-Bouldin), stabilité d'un lancer à l'autre, visualisation (PCA), besoin métier ; et vérifier que les clusters ont un sens (profils interprétables).

</details>

### 7.E4 — Malédiction de la dimension : symptômes et parades

<details><summary>Indice 1</summary>

Relis le §7.6 et le §7.6.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Symptômes : que voit-on sur les scores d'entraînement et de validation quand on ajoute des features ? Et sur les distances ?

</details>
<details><summary>Indice 3</summary>

Densité qui s'effondre, phénomène de Hughes (le score de validation baisse quand on ajoute des features), écart entre entraînement et validation, distances qui se ressemblent toutes. Parades : plus de données, sélection, PCA, régularisation, modèles qui exploitent la structure, savoir du domaine.

</details>

<a id="notebook"></a>

## Notebook, parties A à D

Les exercices du notebook (`03_notebook.ipynb`). Pour ceux qui complètent `mylearn/cluster.py` ou `mylearn/multiclass.py`, la docstring de chaque fonction décrit déjà ce qu'elle doit faire, cas d'erreur compris : relis-la avant d'ouvrir un indice.

### Ex 7.11 — Des œufs en 2D : données, régions et frontière de décision 📦

<details><summary>Indice 1</summary>

Un modèle de scikit-learn se crée, s'entraîne avec `fit`, se mesure avec `score` et prédit avec `predict`. Les labels valent 0 ou 1 : compter les 1, c'est les additionner.

</details>
<details><summary>Indice 2</summary>

`int(y_11.sum())` ; `SklearnNearestCentroid().fit(X_11, y_11)` renvoie le modèle entraîné ; `model_11.score(X, y)` donne l'accuracy sur les œufs que tu lui passes ; `model_11.predict(EGGS_11)` renvoie un tableau NumPy, que `.tolist()` transforme en liste.

</details>
<details><summary>Indice 3</summary>

Squelette, avec les deux lignes clés :

```python
n_fertile_11 = int(y_11.sum())                        # the label 1 marks a fertilised egg
model_11 = SklearnNearestCentroid().fit(X_11, y_11)   # fit returns the fitted model
train_acc_11 = ...   # b) model_11.score(...) on the 200 training eggs
new_acc_11 = ...     # c) the same method, on X_new_11 and y_new_11
pred_11 = ...        # d) model_11.predict(...) on EGGS_11, then .tolist()
```

Ne mélange pas b) et c) : b) mesure le modèle sur les œufs qu'il a vus, c) sur des œufs nouveaux.

</details>

### Ex 7.12 — k-means sur deux lunes : où tombera la coupure ? 🔮

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur l'algorithme de Lloyd : quand il s'arrête, à quel centre chaque point du plan est-il rattaché ?

</details>
<details><summary>Indice 2</summary>

Deux centres, et chaque point rejoint le plus proche : quelle est la forme de l'ensemble des points à égale distance de deux points (∂ 7.7) ? Une telle frontière peut-elle suivre deux lunes qui s'emboîtent ?

</details>
<details><summary>Indice 3</summary>

Écris la condition « $\mathbf{x}$ est à égale distance des deux centres $\mathbf{c}_0$ et $\mathbf{c}_1$ », et développe-la comme en ∂ 7.7 : $2(\mathbf{c}_1 - \mathbf{c}_0) \cdot \mathbf{x} = \lVert \mathbf{c}_1 \rVert^2 - \lVert \mathbf{c}_0 \rVert^2$. Quelle forme décrit une équation de ce type, et peut-elle se plier pour suivre une lune ? Pour b), dessine deux lunes emboîtées, trace la meilleure frontière de cette forme, puis estime à l'œil la part de chaque lune qui tombe du mauvais côté.

</details>

### Ex 7.13 — Distances au carré vectorisées : pairwise_sq_distances 🔨

<details><summary>Indice 1</summary>

Commence par les contrôles (conversion en flottants, deux dimensions, même nombre de colonnes), puis choisis une des deux méthodes de l'énoncé : le broadcasting, ou l'identité du rappel 7.R3 (encadré 🧮 « toutes les distances d'un coup » de la fiche). Garde de préférence l'identité.

</details>
<details><summary>Indice 2</summary>

`A.ndim != 2` ou `A.shape[1] != B.shape[1]` → `ValueError`. Avec l'identité : `(A ** 2).sum(axis=1)` est un vecteur de longueur $n_a$ ; `[:, None]` en fait une colonne de forme $(n_a, 1)$, et `(B ** 2).sum(axis=1)[None, :]` une ligne de forme $(1, n_b)$. Leur somme, moins `2 * A @ B.T`, a la forme $(n_a, n_b)$ par broadcasting.

</details>
<details><summary>Indice 3</summary>

```python
def pairwise_sq_distances(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # ValueError if A or B is not 2-D (ndim), or if their numbers of columns (shape[1]) differ
    sq = (A ** 2).sum(axis=1)[:, None] - 2 * A @ B.T + (B ** 2).sum(axis=1)[None, :]   # (n_a, n_b)
    # clip the tiny negative values to 0 (np.maximum), then return
```

La ligne `sq = ...` assemble une colonne $(n_a, 1)$, le produit $(n_a, n_b)$ et une ligne $(1, n_b)$ : le broadcasting fait le reste. Écris les deux contrôles avant elle : sans eux, un `B` à trois dimensions passerait sans erreur, avec un résultat de forme absurde.

</details>

### Ex 7.14 — Le classifieur du centroïde le plus proche 🔨

<details><summary>Indice 1</summary>

`fit` ne fait que deux choses : trouver les classes et calculer une moyenne par classe. Les trois autres méthodes réutilisent `pairwise_sq_distances`.

</details>
<details><summary>Indice 2</summary>

`np.unique(y)` donne les classes triées ; `X[y == c].mean(axis=0)` le centroïde de la classe `c` ; `np.array([...])` empile les centroïdes. Dans `decision_function`, `D = pairwise_sq_distances(X, self.centroids_)` : avec deux classes, `D[:, 0] - D[:, 1]`, sinon `-D`. `predict` : `self.classes_[D.argmin(axis=1)]` (`argmin` garde le premier minimum). Dans la vérification, `list(model_14.classes_).index("mort")` trouve la ligne de la classe « mort ».

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    # ValueError: X not 2-D, len(X) != len(y), fewer than 2 classes
    self.classes_ = np.unique(y)                                                   # sorted labels
    self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])    # same order
    # return self, so that NearestCentroid().fit(X, y).predict(...) works

def decision_function(self, X):
    D = pairwise_sq_distances(X, self.centroids_)      # shape (n, K)
    # 2 classes: one score per row, positive when classes_[1] is the closer one; K > 2: one column per class
```

`predict` prend, dans chaque ligne de `D`, l'indice de la plus **petite** distance, puis le label de `classes_` qui lui correspond ; `score` renvoie la moyenne des bonnes prédictions, convertie en `float`.

</details>

### Ex 7.15 — Carte de probabilité et politique de seuil pour les œufs 📈

<details><summary>Indice 1</summary>

La règle de Bayes (rappel 7.R2) : $P(\text{fécondé} \mid \mathbf{x})$ est le produit « a priori × densité » des œufs fécondés, divisé par la somme des deux produits. Le seuil $t^*$ est dans l'encadré 🧮 du §7.2.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

`scipy.stats.multivariate_normal(MEAN_1_15, COV_1_15).pdf(X)` donne la densité des œufs fécondés en chaque ligne de `X` ; même chose pour les autres, avec l'a priori $1 - 0{,}35$. $t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$. Pour c), un œuf change de décision s'il est entre les deux lignes de niveau. Pour d), déclare « fécondé » quand `p >= t` : un faux négatif est un œuf fécondé déclaré non fécondé (il est vendu), un faux positif l'inverse.

</details>
<details><summary>Indice 3</summary>

```python
def posterior_15(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))                         # one egg per row
    f_1 = scipy.stats.multivariate_normal(MEAN_1_15, COV_1_15).pdf(X)     # density of the fertilised eggs
    # f_0: the same with MEAN_0_15 and COV_0_15
    # Bayes: PRIOR_15 * f_1, divided by the sum of the two products (the prior of f_0 is 1 - PRIOR_15)
```

b) La formule de l'encadré 🧮, avec les constantes `COST_FP_15` et `COST_FN_15`. c) Se lit sur la carte : un œuf change de décision s'il est **entre** les deux lignes de niveau. Pour d) et e), une boucle sur les deux seuils : `declared = posterior_15(eggs_15) >= threshold`, puis `np.sum(~declared & (truth_15 == 1))` (FN) et `np.sum(declared & (truth_15 == 0))` (FP) ; le coût moyen est $\frac{6\,FN + FP}{2\,000}$.

</details>

### Ex 7.16 — Un-contre-tous avec des centroïdes : quelle classe sera sacrifiée ? 🔮

<details><summary>Indice 1</summary>

Pour chaque classe $k$, le classifieur « $k$ contre le reste » n'a que deux centroïdes : celui de $k$ et celui de tout le reste. Place les quatre centres de `LAYOUT_16` sur une feuille, puis, pour chaque classe, le centroïde de « tout sauf elle ».

</details>
<details><summary>Indice 2</summary>

Les quatre classes ont autant de points : le centroïde de « tout sauf A » est la moyenne des centres de B, C et D. Calcule-le de même pour chacune des quatre classes : pour laquelle tombe-t-il presque sur son propre centre ? Que vaut alors, à peu près partout, le score de son classifieur ?

</details>
<details><summary>Indice 3</summary>

Le centroïde de « tout sauf A » est la moyenne des centres de B, C et D : $\left(\frac{4 + 0 + 0}{3} ; \frac{-2 + 5 + 0{,}33}{3}\right) \approx (1{,}33 ; 1{,}11)$, loin du centre de A, $(-4, -2)$ : le classifieur de A compare deux points bien distincts, et son score est franc. Fais le même calcul pour B, C et D, et regarde chaque fois la distance entre les deux centroïdes du classifieur. Si, pour une classe, ces deux centroïdes sont presque confondus, que vaut son score à peu près partout ? Où cette classe peut-elle encore gagner l'argmax des quatre scores, et quelle part de ses points y tombe ?

</details>

### Ex 7.17 — Manchots sans labels : k-means face aux espèces 📦

<details><summary>Indice 1</summary>

La pureté se lit dans le tableau croisé clusters × espèces : pour chaque **ligne** (un cluster), garde le plus grand effectif, puis additionne et divise par le nombre de points.

</details>
<details><summary>Indice 2</summary>

`pd.crosstab(labels, truth)` a une ligne par cluster : `.max(axis=1)` donne le plus grand effectif de chaque ligne (attention, `.max()` seul prend celui de chaque colonne). Standardiser : `(X_peng - X_peng.mean(axis=0)) / X_peng.std(axis=0)`. Pour e), la part des manchots de chaque espèce (une colonne) qui sont dans son cluster principal : `table.max(axis=0) / table.sum(axis=0)`, puis l'espèce de la plus petite part.

</details>
<details><summary>Indice 3</summary>

```python
def purity_17(labels, truth):
    table = pd.crosstab(np.asarray(labels), np.asarray(truth))   # one row per cluster
    # the largest count of each ROW (axis=1), summed, divided by the number of points: a float
```

b) `SklearnKMeans(n_clusters=3, n_init=10, random_state=0).fit(X_peng).labels_`, puis ta `purity_17`. c) Le z-score de chaque colonne, avec `X_peng.mean(axis=0)` et `X_peng.std(axis=0)`, puis les mêmes appels qu'en b) sur `Z_17`. d) Une ligne, avec les labels de c). e) Dans le tableau croisé de c), `table.max(axis=0) / table.sum(axis=0)` donne la part de chaque espèce dans son cluster principal : garde l'espèce de la plus petite part (`.idxmin()`).

</details>

### Ex 7.18 — Formes arbitraires et bruit : DBSCAN et HDBSCAN 📦

<details><summary>Indice 1</summary>

`summary_18` calcule trois nombres, dans l'ordre : le nombre de clusters (sans compter −1), le nombre de points de bruit (−1), et l'ARI sur les points qui ne sont **pas** du bruit dans la vérité `group_18`.

</details>
<details><summary>Indice 2</summary>

`set(labels.tolist()) - {-1}` donne les clusters ; `np.sum(labels == -1)` le bruit ; `real = group_18 != -1` sélectionne les points à comparer : `adjusted_rand_score(group_18[real], labels[real])`. DBSCAN et HDBSCAN s'utilisent avec `.fit_predict(X_18)`. Le balayage tient en un dictionnaire en compréhension sur `EPS_18`.

</details>
<details><summary>Indice 3</summary>

```python
def summary_18(labels):
    labels = np.asarray(labels)
    real = group_18 != -1                  # the points that are not noise in the truth
    # return (number of distinct labels other than -1, number of -1,
    #         the ARI of group_18[real] and labels[real], as a float)
```

b) à e) Crée chaque algorithme avec les réglages de l'énoncé, puis `.fit(X_18).labels_` pour k-means, `.fit_predict(X_18)` pour DBSCAN et HDBSCAN. `scan_18` est un dictionnaire en compréhension, `{eps: summary_18(...) for eps in EPS_18}`, avec un nouveau `DBSCAN(eps=eps, min_samples=5)` à chaque tour.

</details>

### Ex 7.19 — Distance au plus proche voisin quand la dimension grimpe 🔮

<details><summary>Indice 1</summary>

Commence par la distance entre deux points quelconques : chaque coordonnée ajoute sa part au carré de la distance. Puis demande-toi comment 500 points « remplissent » un cube quand la dimension grandit (✏️ 7.4).

</details>
<details><summary>Indice 2</summary>

En dimension $d$, le carré de la distance entre deux points uniformes est une somme de $d$ termes indépendants, chacun de moyenne $\frac{1}{6}$ : que devient sa moyenne quand $d$ grandit ? Et le plus proche voisin ? Avec 500 points et 2 cases par axe, combien de cases y a-t-il en dimension 10, et combien de points par case ?

</details>
<details><summary>Indice 3</summary>

Raisonne sur deux dimensions extrêmes. En dimension 1, 500 points sur un segment de longueur 1 : deux points consécutifs sont à environ $\frac{1}{500}$ l'un de l'autre, le plus proche voisin à environ $\frac{1}{1\,000}$ (le plus proche des deux côtés), et deux points quelconques à $\frac{1}{3}$ en moyenne. En dimension 100, le carré de la distance entre deux points est une somme de 100 termes de moyenne $\frac{1}{6}$ : écris la distance moyenne. Pour le plus proche voisin, compte les cases ($2^{100}$, à 2 par axe) pour 500 points : chaque point a-t-il encore un voisin dans sa case ? Pour c), demande-toi si le plus proche des 499 autres points peut être beaucoup plus près qu'un point quelconque, quand chacune des 100 coordonnées ajoute sa part au carré de la distance.

</details>

### Ex 7.20 — Densité, plus proche voisin et concentration des distances 🔬

<details><summary>Indice 1</summary>

Quatre calculs courts. a) Transforme chaque coordonnée en numéro de case, puis compte les lignes distinctes. b) C'est la formule de ✏️ 7.4 g). c) et d) partent de la même matrice de distances, dont il faut écarter la diagonale.

</details>
<details><summary>Indice 2</summary>

a) `np.minimum(np.floor(X * bins).astype(int), bins - 1)`, puis `len(np.unique(cells, axis=0))`. b) $m\,(1 - (1 - \frac{1}{m})^n)$ avec $m = 4^5$ cases et $n = 1\,000$. c) `D = np.sqrt(mylearn.cluster.pairwise_sq_distances(X, X))` puis `np.fill_diagonal(D, 0.0)` ; le masque `off = ~np.eye(len(X), dtype=bool)` écarte la diagonale : `np.where(off, D, np.inf).min(axis=1)` donne la distance de chaque point à son plus proche voisin, `D[off].mean()` la distance moyenne. d) Même chose avec le maximum, en mettant `-np.inf` sur la diagonale.

</details>
<details><summary>Indice 3</summary>

```python
def occupied_cells_20(X, bins):
    cells = np.minimum(np.floor(np.asarray(X) * bins).astype(int), bins - 1)   # one row of cell numbers per point
    # return the number of distinct rows of cells


def nn_and_mean_20(X):
    D = np.sqrt(mylearn.cluster.pairwise_sq_distances(X, X))
    np.fill_diagonal(D, 0.0)
    off = ~np.eye(len(X), dtype=bool)          # True everywhere except on the diagonal
    # nearest neighbour: np.where(off, D, np.inf), the min of each row, then the mean of these minimums
    # mean distance: the mean of D[off]; return the two numbers as floats
```

`contrast_20` repart des trois mêmes premières lignes : le minimum de chaque ligne comme ci-dessus, le maximum avec `-np.inf` sur la diagonale, puis la moyenne de $\frac{d_{\max} - d_{\min}}{d_{\min}}$. b) La formule de ✏️ 7.4 g), avec $m = 4^5$ cases et $n = 1\,000$ points.

</details>

### Ex 7.21 — Reproduire les figures 7.27 et 7.30 (boule/cube, hyper-orange) 🎨

<details><summary>Indice 1</summary>

a) Une boucle sur $d$ qui applique la récurrence de ∂ 7.6, à partir de $V_1 = 2$ et $V_2 = \pi$. b) Une ligne : le résultat de ∂ 7.5.

</details>
<details><summary>Indice 2</summary>

Range les volumes dans un dictionnaire (ou une liste) indexé par $d$ : `V[d] = 2 * np.pi / d * V[d - 2]`, et le rapport vaut `V[d] / 2 ** d`. Pour b), `np.sqrt(d) - 1` fonctionne aussi quand `d` est un tableau.

</details>
<details><summary>Indice 3</summary>

```python
def ball_ratio_21(d_max):
    volumes = {1: 2.0, 2: np.pi}           # two starting values: the recurrence jumps by 2
    # for d from 3 to d_max: volumes[d] from volumes[d - 2] (∂ 7.6)
    # return the np.array of volumes[d] / 2 ** d, for d = 1, ..., d_max
```

`orange_radius_21` tient en une ligne : la formule de ∂ 7.5, écrite avec `np.sqrt` pour qu'elle marche aussi sur un tableau de dimensions.

</details>

### Ex 7.22 — Un-contre-tous générique : OneVsRestClassifier 🔨

<details><summary>Indice 1</summary>

`fit` est une boucle sur les classes ; à chaque tour, une copie neuve de l'estimateur et un `fit` sur des labels 0/1. `decision_function` assemble une colonne de scores par modèle, et `predict` prend la plus grande.

</details>
<details><summary>Indice 2</summary>

`copy.deepcopy(self.estimator)` crée un modèle neuf ; `(y == c).astype(int)` donne les labels 0/1 ; `hasattr(self.estimator, "decision_function")` teste si une méthode existe. La colonne d'un modèle : `model.decision_function(X)` s'il l'a, sinon `model.predict_proba(X)[:, 1]` ; `np.column_stack` assemble les colonnes. `predict` : `self.classes_[np.argmax(scores, axis=1)]`.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y):
    y = np.asarray(y)
    classes = np.unique(y)
    # ValueError: fewer than 3 classes; an estimator with neither decision_function nor predict_proba (hasattr)
    self.classes_ = classes
    self.estimators_ = []
    # for each class c: copy.deepcopy(self.estimator), fitted on X and (y == c).astype(int), then appended
    return self

def decision_function(self, X):
    columns = []
    # for each model of estimators_: model.decision_function(X) if it has one,
    #     else the column 1 of model.predict_proba(X); append it (np.ravel) to columns
    return np.column_stack(columns)        # one column per model, in the order of classes_
```

`predict` : `self.classes_[np.argmax(self.decision_function(X), axis=1)]` ; `score` comme en 7.14. Copie l'estimateur **avant** chaque `fit`, jamais après : `self.estimator` lui-même ne doit pas être entraîné.

</details>

### Ex 7.23 — Un-contre-un générique : OneVsOneClassifier 🔨

<details><summary>Indice 1</summary>

Comme en 7.22, mais la boucle porte sur les couples $(i, j)$ avec $i < j$, et chaque modèle ne voit que deux classes. `votes` donne une voix par modèle et par point.

</details>
<details><summary>Indice 2</summary>

Les couples : deux boucles imbriquées, `for i in range(K): for j in range(i + 1, K)` (ou `itertools.combinations(range(K), 2)`). Le masque `keep = (y == classes[i]) | (y == classes[j])` garde les deux classes ; les labels sont `(y[keep] == classes[j]).astype(int)`. Dans `votes`, pars de `np.zeros((len(X), K), dtype=int)` ; pour chaque modèle, `for_j = np.asarray(model.predict(X)) == 1`, puis ajoute `for_j` à la colonne `j` et `~for_j` à la colonne `i`.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y):
    X, y = np.asarray(X), np.asarray(y)
    self.classes_ = np.unique(y)
    K = len(self.classes_)
    # ValueError if K < 3
    self.pairs_ = [(i, j) for i in range(K) for j in range(i + 1, K)]
    self.estimators_ = []
    # for each pair (i, j): keep = the samples of classes_[i] or classes_[j];
    #     a deepcopy of the estimator, fitted on X[keep] with the labels 1 for classes_[j] and 0 for classes_[i]
    return self

def votes(self, X):
    votes = np.zeros((len(X), len(self.classes_)), dtype=int)
    # for each pair (i, j) and its model: for_j = (its predict(X) == 1);
    #     one vote in the column j where for_j is True, one vote in the column i elsewhere
    return votes
```

</details>

### Ex 7.24 — OvR, OvO ou multi-classe natif : accuracy, nombre de modèles, temps 🔬

<details><summary>Indice 1</summary>

a) La moyenne et l'écart-type viennent des seules lignes `TRAIN_PENG`, et servent aux deux tableaux. b) Une boucle sur les trois modèles, et pour chacun quatre mesures.

</details>
<details><summary>Indice 2</summary>

`mean = X_peng[TRAIN_PENG].mean(axis=0)` et `std = X_peng[TRAIN_PENG].std(axis=0)`. Pour chronométrer : `start = time.perf_counter()` avant `fit`, puis la différence juste après. Le nombre de modèles : `len(model.estimators_)` pour les deux méta-estimateurs, 1 pour le modèle natif (`getattr(model, "estimators_", [model])` traite les deux cas). Un dictionnaire `{nom de la stratégie: {colonne: valeur}}` donne le `DataFrame` attendu avec `pd.DataFrame.from_dict(rows, orient="index")` : une ligne par stratégie.

</details>
<details><summary>Indice 3</summary>

a) Calcule `mean` et `std` sur `X_peng[TRAIN_PENG]` seulement (indice 2), puis applique ces deux mêmes vecteurs aux manchots d'entraînement et aux manchots de test.

```python
def compare_24(X_train, y_train, X_test, y_test):
    rows = {}
    for name, model in [("native", mylearn.cluster.NearestCentroid()),
                        ("OvR", ...), ("OvO", ...)]:       # the two wrappers, each around a NEW NearestCentroid()
        start = time.perf_counter()
        # fit on the training set, then time.perf_counter(); predict on the test set, then time.perf_counter()
        rows[name] = {"n_models": len(getattr(model, "estimators_", [model])),
                      ...}                                  # accuracy, fit_s, predict_s
    return pd.DataFrame.from_dict(rows, orient="index")     # one row per strategy
```

</details>

### Ex 7.25 — Initialisation k-means++ 🔨

<details><summary>Indice 1</summary>

Deux temps : un premier centre tiré au hasard, puis une boucle qui tire chaque centre suivant avec des probabilités proportionnelles aux $D(\mathbf{x})^2$. Tiens à jour le tableau des $D^2$ au lieu de tout recalculer à chaque tour.

</details>
<details><summary>Indice 2</summary>

`rng.integers(len(X))` tire le premier indice. `d2 = ((X - X[first]) ** 2).sum(axis=1)` donne les $D^2$ au premier centre ; après chaque nouveau centre `c`, `d2 = np.minimum(d2, ((X - X[c]) ** 2).sum(axis=1))`. Le tirage : `rng.choice(len(X), p=d2 / d2.sum())`. Le nombre de lignes distinctes : `len(np.unique(X, axis=0))`.

</details>
<details><summary>Indice 3</summary>

```python
def kmeans_plusplus(X, n_clusters, rng=None):
    X = np.asarray(X, dtype=float)
    # ValueError: X not 2-D; n_clusters not between 1 and len(np.unique(X, axis=0))
    rng = np.random.default_rng() if rng is None else rng
    chosen = [int(rng.integers(len(X)))]                 # the first centre: a row drawn uniformly
    d2 = ((X - X[chosen[0]]) ** 2).sum(axis=1)          # D(x)^2 to the only centre chosen so far
    # n_clusters - 1 times: draw the next index with rng.choice and p = d2 / d2.sum(), append it,
    #     then keep in d2 the smaller of d2 and the squared distances to this new centre
    return X[chosen]
```

</details>

### Ex 7.26 — k-means de Lloyd : la classe KMeans 🔨

<details><summary>Indice 1</summary>

Découpe `fit` en morceaux : la validation, les centres initiaux (selon `init`), **une exécution** de Lloyd, que tu écris dans une fonction d'aide qui renvoie `(centres, labels, inertie, n_iter)`, puis la boucle sur les exécutions, qui garde la meilleure.

</details>
<details><summary>Indice 2</summary>

Une exécution : initialise `previous = np.full(len(X), -1)` (la première itération ne peut pas s'arrêter sur « affectation inchangée »). À chaque itération `i` : `labels` = `argmin` des distances aux centres ; les nouveaux centres, une moyenne par cluster (un cluster vide garde son centre) ; `shift = ((new - centres) ** 2).sum()`. Puis : si `np.array_equal(labels, previous)`, arrête-toi (les labels sont déjà ceux des centres finaux) ; sinon, si `shift <= tol`, arrête-toi aussi, mais recalcule les labels avec les nouveaux centres. `n_iter = i + 1`. Si la boucle va jusqu'à `max_iter`, recalcule aussi les labels.

</details>
<details><summary>Indice 3</summary>

Une exécution de Lloyd, avec l'ordre des deux tests d'arrêt :

```python
def _lloyd(X, centres, max_iter, tol):
    previous, converged = np.full(len(X), -1), False
    for i in range(max_iter):
        labels = pairwise_sq_distances(X, centres).argmin(axis=1)      # (1) assignment
        # (2) new = centres.copy(); each cluster j that has points: new[j] = the mean of its points (axis=0)
        shift = float(((new - centres) ** 2).sum())
        centres = new
        # stop 1: labels equal to previous -> converged = True, break (this iteration counts)
        # stop 2: shift <= tol -> break
        previous = labels
    # if not converged: the labels of the final centres
    # return centres, labels, the inertia (a float), i + 1
```

Dans `fit`, après la validation : `rng = np.random.default_rng(self.random_state)`, créé une seule fois pour toutes les exécutions, `tol = self.tol * np.mean(np.var(X, axis=0))`, puis `n_init` exécutions (une seule si `init` est un tableau), chacune avec ses centres initiaux (`kmeans_plusplus(X, k, rng=rng)`, `X[rng.choice(n, size=k, replace=False)]` ou `np.array(self.init, dtype=float)`). Garde l'exécution dont l'inertie est **strictement** plus petite que la meilleure déjà vue.

</details>

### Ex 7.27 — k-means piégé : quatre bugs à débusquer 🐛

<details><summary>Indice 1</summary>

Isole les bugs un par un, comme le fera la vérification : regarde si `init_27` a changé après l'appel ; lance la fonction avec `max_iter=1` et compare ses centres aux moyennes que tu calcules toi-même ; regarde `n_iter_c27` ; recalcule l'inertie avec les centres et les labels renvoyés.

</details>
<details><summary>Indice 2</summary>

Quatre lignes sont suspectes : `centres = init`, la mise à jour `X[labels == j].mean()`, les deux lignes autour de la comparaison des labels, et le calcul de l'inertie. Pour chacune, pose-toi une question : copie ou même tableau ? moyenne de quoi, sur quel axe ? dans quel ordre ? des carrés ou pas ?

</details>
<details><summary>Indice 3</summary>

Bug 1, en modèle : `centres = init` ne copie rien, si bien que `centres[j] = ...` écrit dans le tableau de l'appelant. Correction : `centres = np.array(init, dtype=float)`, une copie en flottants. Bug 2 : sur le tableau de forme $(m, 4)$ des points d'un cluster, que renvoie `.mean()` sans argument ? Il faut une moyenne par feature : choisis l'axe. Bug 3 : au moment de la comparaison, que vaut `old_labels`, juste après `old_labels = labels` ? Dans quel ordre faut-il donc placer ces deux lignes ? Bug 4 : l'inertie est une somme de **carrés** de distances : quelle opération de la ligne est en trop ?

</details>

### Ex 7.28 — Coefficient de silhouette 🔨

<details><summary>Indice 1</summary>

Une matrice de distances (pas au carré !), puis, pour chaque cluster, un masque booléen : la somme des distances de chaque point aux points de ce cluster. $a(i)$ et $b(i)$ se lisent ensuite dans ce tableau de sommes.

</details>
<details><summary>Indice 2</summary>

`D = np.sqrt(pairwise_sq_distances(X, X))`, puis `np.fill_diagonal(D, 0.0)`. `codes = np.unique(labels, return_inverse=True)[1]` numérote les clusters de 0 à $K - 1$, et `np.bincount(codes)` donne leurs tailles. `sums[:, c] = D[:, codes == c].sum(axis=1)`. $a(i)$ = `sums[i, codes[i]]` divisé par la taille de son cluster moins 1. Pour $b(i)$, divise chaque colonne de `sums` par la taille de son cluster, mets `np.inf` dans la colonne du cluster de $i$, puis prends le minimum de la ligne.

</details>
<details><summary>Indice 3</summary>

```python
def silhouette_samples(X, labels):
    X = np.asarray(X, dtype=float)
    _, codes = np.unique(np.asarray(labels), return_inverse=True)   # clusters numbered 0 to K - 1
    n, K = len(X), codes.max() + 1
    # ValueError unless 2 <= K <= n - 1
    D = np.sqrt(pairwise_sq_distances(X, X))
    np.fill_diagonal(D, 0.0)
    sums = np.column_stack([D[:, codes == c].sum(axis=1) for c in range(K)])   # (n, K)
    # a: sums[i, codes[i]] / max(size of the cluster of i - 1, 1), the sizes coming from np.bincount(codes)
    # b: sums / sizes, np.inf in the column of the cluster of i, then the min of each row
    # s = (b - a) / max(a, b); 0 for a point alone in its cluster, and 0 instead of nan (np.nan_to_num)
```

`silhouette_score` : la moyenne de `silhouette_samples`, en `float`. La ligne `sums` est la clé : une seule matrice de distances, puis une somme par cluster grâce au masque `codes == c`.

</details>

### Ex 7.29 — Choisir k : coude de l'inertie et silhouette, de k = 2 à 7 🔬

<details><summary>Indice 1</summary>

`scan_29` : une boucle sur `ks`, un modèle par $k$, deux valeurs par modèle, rangées dans deux dictionnaires. Pour d) et e), lis le tableau et les courbes affichés par la vérification.

</details>
<details><summary>Indice 2</summary>

`model = mylearn.cluster.KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)`, puis `model.inertia_` et `mylearn.cluster.silhouette_score(X, model.labels_)`. Pour le coude, calcule les baisses successives de l'inertie, `inertias_29[k - 1] - inertias_29[k]` : le coude est le dernier $k$ atteint par une grande baisse, celui après lequel les baisses deviennent petites.

</details>
<details><summary>Indice 3</summary>

```python
def scan_29(X, ks):
    inertias, silhouettes = {}, {}
    for k in ks:
        model = mylearn.cluster.KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        # inertias[k]: its inertia_; silhouettes[k]: the silhouette score of X with its labels_
    return inertias, silhouettes
```

d) `max(silhouettes_29, key=silhouettes_29.get)` donne la clé de la plus grande valeur. e) Range les baisses dans un dictionnaire, `{k: inertias_29[k - 1] - inertias_29[k] for k in range(3, 8)}`, puis compare chaque baisse à la suivante : le coude est le $k$ après lequel les baisses deviennent petites.

</details>

### Ex 7.30 — Phénomène de Hughes : des features de bruit qui font chuter l'accuracy 🔬

<details><summary>Indice 1</summary>

Le 1-NN, c'est l'`argmin` de chaque ligne de la matrice des distances au carré entre les points de test et les points d'entraînement : la plus petite distance au carré désigne aussi le point le plus proche. `hughes_30` construit les features, puis entraîne et mesure les deux classifieurs.

</details>
<details><summary>Indice 2</summary>

`D = mylearn.cluster.pairwise_sq_distances(X_test, X_train)` a une ligne par point de test ; `D.argmin(axis=1)` donne l'indice du voisin, et `np.asarray(y_train)[...]` son label. Dans `hughes_30` : `F = features_30(n_real, n_noise)`, puis `F[TRAIN_PENG]` et `F[TEST_PENG]`, avec `species[TRAIN_PENG]` et `species[TEST_PENG]` pour les labels.

</details>
<details><summary>Indice 3</summary>

```python
def nn1_predict_30(X_train, y_train, X_test):
    nearest = mylearn.cluster.pairwise_sq_distances(X_test, X_train).argmin(axis=1)   # one index per test point
    # return the labels of these training points (np.asarray(y_train), indexed by nearest)


def hughes_30(n_real, n_noise):
    F = features_30(n_real, n_noise)
    # nearest centroid: NearestCentroid fitted on the TRAIN_PENG rows of F and of species, scored on the TEST_PENG rows
    # 1-NN: the mean of (nn1_predict_30(...) == the test species), as a float
    # return [accuracy of the nearest centroid, accuracy of the 1-NN]
```

Garde le même découpage pour les deux classifieurs et toutes les configurations : seules les colonnes de `F` changent.

</details>

### Ex 7.31 — Défi : retrouver les espèces de manchots sans labels 🏆

<details><summary>Indice 1</summary>

Regarde la matrice de corrélation des quatre mesures (`np.corrcoef(X_peng.T)`, ch. 2) : deux d'entre elles disent presque la même chose. Une fois les mesures standardisées, combien pèsent-elles ensemble dans les distances ?

</details>
<details><summary>Indice 2</summary>

Après standardisation, chaque feature pèse autant dans la distance. Deux features corrélées à 0,87 font donc presque voter deux fois la même information : la **taille** du manchot. Or les Adélie et les Chinstrap ont à peu près la même taille ; c'est surtout la longueur du bec qui les distingue. Comment rendre à la taille une seule voix, sans regarder les espèces ?

</details>
<details><summary>Indice 3</summary>

Deux pistes qui marchent : retirer la masse (ou la nageoire) avant de standardiser, ou remplacer ces deux colonnes, standardisées, par leur moyenne, une seule feature « taille ». Dans les deux cas, calcule la moyenne et l'écart-type sur le `X` que reçoit ta fonction, puis lance ton `KMeans` avec `n_clusters=3`.

</details>
