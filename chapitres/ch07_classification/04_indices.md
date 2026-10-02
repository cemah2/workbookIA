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

Trois des expressions désignent la même chose, celle que l'expert a fixée ; la quatrième sort du modèle. La vérité terrain est ce qu'on **tient pour** juste (ch. 3). Un modèle peut apprendre ses exemples par cœur sans rien savoir des œufs suivants (ch. 1, et l'overfitting du ch. 9).

</details>

### 7.Q2 — Binaire, multi-classe ou multi-étiquette ? Cinq situations

<details><summary>Indice 1</summary>

Relis les trois définitions du §7.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pose deux questions à chaque tâche : combien de classes possibles ? Un même exemple peut-il en recevoir plusieurs à la fois ?

</details>
<details><summary>Indice 3</summary>

Deux classes : binaire. Plus de deux classes, une seule par exemple : multi-classe. Plusieurs labels possibles à la fois : multi-étiquette. Une photo de vacances peut montrer à la fois une plage et un chien ; un chiffre manuscrit n'est qu'un seul chiffre.

</details>

### 7.Q3 — Régions et frontières de décision

<details><summary>Indice 1</summary>

Relis le §7.2.1 de la fiche, et regarde la figure des trois situations.

</details>
<details><summary>Indice 2</summary>

Pour b), une classe que le classifieur peut prédire peut-elle n'occuper aucun morceau du plan ? Pour d), pense à ce que produit `predict` pour chaque point. Pour e), essaie de faire légèrement tourner une droite qui sépare deux nuages bien écartés.

</details>
<details><summary>Indice 3</summary>

La figure montre une courbe. Trois classes prédictibles, c'est au moins trois régions ; rien n'oblige une classe à n'occuper qu'un seul morceau. Un classifieur rend **une** classe par point, même sur la frontière (une convention tranche). Entre deux nuages bien séparés, une infinité de droites passent : c'est pour cela qu'il faut un critère pour en choisir une.

</details>

### 7.Q4 — Classes qui se recouvrent : probabilités et politique de seuil

<details><summary>Indice 1</summary>

Relis le seuil et l'encadré 🧮 « le seuil qui minimise le coût moyen » au §7.2.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

La région « fécondé » est l'ensemble des points où $P \ge t$. Que devient cet ensemble quand $t$ passe de 0,5 à 0,2 ? Suis un œuf fécondé déjà déclaré positif, puis un œuf non fécondé déjà déclaré positif.

</details>
<details><summary>Indice 3</summary>

Tout point où $P \ge 0{,}5$ vérifie aussi $P \ge 0{,}2$ : la région ne peut que grandir, donc on garde tous les positifs déjà déclarés (vrais et faux), et l'on en ajoute. Pour d), $t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ avec $C_{FP} = 1$ et $C_{FN} = 7$.

</details>

### 7.Q5 — Cinq mesures par œuf : ce que change (ou non) la dimension

<details><summary>Indice 1</summary>

Relis le §7.3 de la fiche : la dimension, la dimension d'une frontière, et les deux nuances.

</details>
<details><summary>Indice 2</summary>

Une dimension par feature. En dimension $d$, une frontière a une dimension de moins.

</details>
<details><summary>Indice 3</summary>

Cinq features : un point de $\mathbb{R}^5$, et une frontière de dimension $5 - 1$. Les formules ne changent pas, mais le coût, si. Pour e), pense aux projections deux à deux et à la PCA (ch. 12).

</details>

### 7.Q6 — Un-contre-tous : combien de modèles, quelle décision ?

<details><summary>Indice 1</summary>

Relis le §7.4.1 de la fiche et le panneau (b) de la figure des quatre classes.

</details>
<details><summary>Indice 2</summary>

Pour a), à quelle question répond chacun des classifieurs, et combien de questions de ce genre faut-il poser ? Pour b) et d), la règle de décision regarde-t-elle le signe des scores ? Pour c), additionne les probabilités de b).

</details>
<details><summary>Indice 3</summary>

7 classes, 7 modèles. En b), compare 0,55 et 0,52 ; en d), le plus grand de quatre nombres négatifs est le plus proche de 0. Les modèles sont entraînés séparément : leurs probabilités n'ont aucune raison de sommer à 1 (en b), elles font 1,47). Pour e), relis l'encadré ⚠️ sur *binary relevance*.

</details>

### 7.Q7 — Un-contre-un : duels, votes et coût

<details><summary>Indice 1</summary>

Relis le §7.4.2 de la fiche : la formule du nombre de duels, le mini-exemple et l'encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

$\binom{K}{2} = \frac{K(K-1)}{2}$. Pour e), compare $\frac{K(K-1)}{2}$ et $\frac{K^2}{2}$ : développe le premier.

</details>
<details><summary>Indice 3</summary>

$\frac{6 \times 5}{2}$ duels. Un duel ne voit que ses deux classes. Avec 3 classes : $\frac{3 \times 2}{2}$ duels contre 3 modèles. En d), A et B sont à égalité : la règle de `mylearn` garde le plus petit indice. En e), $\frac{K(K-1)}{2} = \frac{K^2}{2} - \frac{K}{2}$. Pour f), compte les voix que peut recevoir la vraie classe, et celles des autres.

</details>

### 7.Q8 — Ce que k-means ne peut pas deviner tout seul

<details><summary>Indice 1</summary>

Relis « Sans labels : k-means » et « Combien de clusters ? » au §7.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Apprentissage supervisé ou non ? Qui choisit $k$, et quand ? Que fait l'inertie quand $k$ augmente ?

</details>
<details><summary>Indice 3</summary>

k-means travaille sans labels, et rend exactement $k$ clusters. $k$ est fixé avant l'entraînement. Le résultat dépend du départ (minimum local, d'où `n_init`). La meilleure inertie possible baisse toujours quand $k$ grandit, jusqu'à 0 quand chaque point est seul.

</details>

### 7.Q9 — Densité d'échantillons quand les features s'accumulent

<details><summary>Indice 1</summary>

Relis la formule de la densité au §7.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

$\rho = \frac{n}{b^d}$, ici avec $n = 20$ et $b = 4$.

</details>
<details><summary>Indice 3</summary>

$\frac{20}{4}$, $\frac{20}{16}$, $\frac{20}{64}$. Doubler $n$ double le numérateur ; ajouter une feature multiplie le dénominateur par $b$.

</details>

### 7.Q10 — Bénédiction de la structure : vrai ou faux justifié

<details><summary>Indice 1</summary>

Relis la fin du §7.6 de la fiche (la bénédiction de la non-uniformité) et la liste des parades.

</details>
<details><summary>Indice 2</summary>

Pour 2, combien d'images en noir et blanc de 784 pixels existe-t-il ? Combien ressemblent à un chiffre ? Pour 3, le livre emploie un mot précis pour la malédiction et la bénédiction.

</details>
<details><summary>Indice 3</summary>

Les données réelles se concentrent près de surfaces de faible dimension ; $2^{784}$ images possibles, dont presque toutes sont de la neige. Le livre parle de constats empiriques. Retirer des features de bruit remonte la densité (phénomène de Hughes, 7.30). La figure 7.23 du livre illustre 5.

</details>

### 7.Q11 — Géométrie déroutante en grande dimension

<details><summary>Indice 1</summary>

Relis le §7.6.1 de la fiche : la peau de l'orange, l'hyper-orange et l'encadré ⚠️ sur la sphère hérissée.

</details>
<details><summary>Indice 2</summary>

Pour a), applique $1 - (1 - \varepsilon)^d$. Pour c), le coin du cube de côté 1 centré en 0 a toutes ses coordonnées égales à $\pm \frac{1}{2}$. Pour d), relis la concentration des distances.

</details>
<details><summary>Indice 3</summary>

$1 - 0{,}97^{100}$ : calcule-le, puis compare à 0,99. Le livre donne lui-même le seuil de l'hyper-orange. Distance du centre au coin : $\sqrt{d \times \frac{1}{4}} = \frac{\sqrt{d}}{2}$. Le plus proche voisin n'est « guère plus proche que les autres points » en grande dimension.

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

a) $-1 \times \log_2 1$. b) $[\frac{1}{2}, \frac{1}{2}]$. c) $\frac{1}{2} \times 1 + 2 \times \frac{1}{4} \times 2$. d) $-(0{,}7 \log_2 0{,}7 + 0{,}2 \log_2 0{,}2 + 0{,}1 \log_2 0{,}1)$. e) $\log_2 3$. Pour f), c'est l'entropie conditionnelle du ch. 6.

</details>

### 7.R2 — Ch. 4 : probabilité a posteriori « fécondé » par la règle de Bayes

<details><summary>Indice 1</summary>

Relis la règle de Bayes du ch. 4 (§4.4.1 et §4.4.2 de sa fiche), ou sa forme au §7.2.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

$P = \frac{\pi f_1}{\pi f_1 + (1 - \pi) f_0}$, avec $\pi = 0{,}3$. Pour d), écris $P = 0{,}5$, c'est-à-dire $\pi f_1 = (1 - \pi) f_0$.

</details>
<details><summary>Indice 3</summary>

a) $\frac{0{,}3 \times 0{,}08}{0{,}3 \times 0{,}08 + 0{,}7 \times 0{,}04} = \frac{0{,}024}{0{,}052}$. c) Même calcul avec 0,5 et 0,5. d) $\frac{f_1}{f_0} = \frac{1 - \pi}{\pi}$.

</details>

### 7.R3 — 0B : développer ‖a − b‖² avec le produit scalaire

<details><summary>Indice 1</summary>

$\lVert \mathbf{v} \rVert^2 = \mathbf{v} \cdot \mathbf{v}$, la somme des carrés des coordonnées (0B).

</details>
<details><summary>Indice 2</summary>

$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + a_3 b_3$. Pour d), calcule d'abord $\mathbf{a} - \mathbf{b}$.

</details>
<details><summary>Indice 3</summary>

$\mathbf{a} - \mathbf{b} = (2, -3, 2)$. Pour e), développe $(\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b})$ ; pour des matrices $\mathbf{A}$ et $\mathbf{B}$ (un point par ligne), les produits scalaires de toutes les paires forment $\mathbf{A} \mathbf{B}^\top$ (encadré 🧮 du §7.5 de la fiche).

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 7.1 — Compter les classifieurs OvR et OvO

<details><summary>Indice 1</summary>

Relis le §7.4.2 de la fiche : $N_{\text{OvR}} = K$ et $N_{\text{OvO}} = \frac{K(K-1)}{2}$.

</details>
<details><summary>Indice 2</summary>

Pour e), cherche $K$ tel que $K(K - 1) > 20\,000$ : $\sqrt{20\,000} \approx 141$, puis essaie les entiers autour. Pour g), combien d'images voit un duel ? Combien y a-t-il de duels ?

</details>
<details><summary>Indice 3</summary>

e) Compare $141 \times 140$ et $142 \times 141$ à 20 000. f) $10 \times 50\,000$. g) Un duel voit $2 \times 5\,000$ images, et il y a 45 duels (autre lecture : chaque image participe aux $K - 1$ duels de sa classe). h) Un-contre-tous : $10 \times 50\,000^2$ ; un-contre-un : $45 \times 10\,000^2$ ; fais le rapport.

</details>

### Ex 7.2 — Dépouiller les votes d'un un-contre-un à quatre classes

<details><summary>Indice 1</summary>

Chaque duel donne une voix à son vainqueur, et une seule.

</details>
<details><summary>Indice 2</summary>

Parcours le tableau colonne par colonne et ajoute 1 à la classe gagnante. Pour g), combien de voix a une classe qui gagne tous ses duels ? Combien au plus pour chacune des autres, qui ont toutes perdu contre elle ?

</details>
<details><summary>Indice 3</summary>

Premier point : B gagne A-B, B-C et B-D ; D gagne A-D et C-D ; A gagne A-C. Second point : A et C ont deux voix chacune ; regarde qui a gagné le duel A-C. g) $K - 1$ voix contre au plus $K - 2$. h) 6 voix pour 4 classes : combien chacune en cas d'égalité parfaite ?

</details>

### Ex 7.3 — Une itération de k-means à la main

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur l'algorithme de Lloyd et son mini-exemple en dimension 1 (§7.5 de la fiche).

</details>
<details><summary>Indice 2</summary>

Fais un tableau : une ligne par point, deux colonnes pour les distances au carré aux deux centres. L'inertie est la somme, sur les points, de la distance au carré à **son** centre.

</details>
<details><summary>Indice 3</summary>

Distances au carré à $c_1 = (1, 1)$ et $c_2 = (1, 3)$ : $P_4$ donne 25 et 17, $P_5$ 41 et 29, $P_6$ 45 et 37. Le nouveau $c_2$ est la moyenne de $P_3$ à $P_6$ : $\left(\frac{19}{4}, \frac{16}{4}\right)$. Pour e), recalcule les distances de $P_3$ aux nouveaux centres. Pour f), un cluster contient $P_1$, $P_2$, $P_3$, l'autre $P_4$, $P_5$, $P_6$.

</details>

### Ex 7.4 — Densité d'échantillons et nombre d'œufs nécessaires

<details><summary>Indice 1</summary>

Relis la formule $\rho = \frac{n}{b^d}$ et l'encadré ⚠️ « Une densité n'est pas une probabilité » (§7.6 de la fiche).

</details>
<details><summary>Indice 2</summary>

Pour b) à d), $n = \rho\, b^d$. Pour f), un œuf tombe hors d'un petit cube donné avec la probabilité $\frac{124}{125}$, et les 10 œufs sont indépendants. Pour g), additionne, sur les 125 cubes, la probabilité que chacun soit occupé.

</details>
<details><summary>Indice 3</summary>

a) $\frac{360}{6}$, $\frac{360}{36}$, $\frac{360}{216}$, $\frac{360}{1296}$. e) Compare $6^7$ et $6^8$ à un million. f) $\left(\frac{124}{125}\right)^{10}$. g) $125 \times (1 - \text{f})$ : l'espérance d'une somme est la somme des espérances.

</details>

### Ex 7.5 — Le rayon de l'hyper-orange : r(d) = √d − 1

<details><summary>Indice 1</summary>

La distance de l'origine à un point de coordonnées $(x_1, \ldots, x_d)$ est $\sqrt{x_1^2 + \cdots + x_d^2}$.

</details>
<details><summary>Indice 2</summary>

Toutes les coordonnées du centre d'un ballon valent $\pm 1$. Deux boules qui se touchent : la distance entre leurs centres est la somme de leurs rayons.

</details>
<details><summary>Indice 3</summary>

Centre du ballon à $\sqrt{d \times 1}$ de l'origine, donc $r + 1 = \sqrt{d}$. a) $\sqrt{2} - 1$. c) Un ballon par coin, et un cube de dimension $d$ a $2^d$ coins. d) $\sqrt{d} \ge 2{,}5$, donc $d \ge 6{,}25$. e) $\sqrt{d} > 3{,}5$. f) $\sqrt{50} - 1$, avec $\sqrt{50} = 5\sqrt{2}$. g) Le centre du ballon le plus proche du milieu de la face diffère de lui de 1 sur la première coordonnée et de 1 sur chacune des $d - 1$ autres.

</details>

### Ex 7.6 — Boule dans un cube : rapport des volumes par récurrence

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur le volume de la boule (§7.6.1 de la fiche) : il n'y a qu'à appliquer la récurrence, deux dimensions à la fois.

</details>
<details><summary>Indice 2</summary>

$V_3 = \frac{4\pi}{3}$, puis $V_4 = \frac{2\pi}{4} V_2$ et $V_5 = \frac{2\pi}{5} V_3$. Garde $\pi$ en symbole jusqu'au bout. Pour f), voir le §7.6.1 de la fiche (la peau).

</details>
<details><summary>Indice 3</summary>

$V_4 = \frac{\pi^2}{2}$, $V_5 = \frac{8\pi^2}{15}$, $V_{10} = \frac{\pi^5}{120}$ ; $q_4 = \frac{V_4}{16}$, $q_{10} = \frac{V_{10}}{1024}$, puis multiplie par 100. e) Calcule $V_6 = \frac{2\pi}{6} V_4$ et compare. g) Écris $2^d = 4 \times 2^{d-2}$. h) $\frac{2\pi}{d} < 1$ dès que $d > 2\pi$ ; les facteurs suivants sont de plus en plus petits.

</details>

### Ex 7.7 — La frontière du centroïde le plus proche est une droite

<details><summary>Indice 1</summary>

Développe les deux membres avec l'identité de 7.R3 : les termes $\lVert \mathbf{x} \rVert^2$ se simplifient.

</details>
<details><summary>Indice 2</summary>

Une équation $\mathbf{w} \cdot \mathbf{x} = c$ avec $\mathbf{w} \ne \mathbf{0}$ décrit une droite (un hyperplan) perpendiculaire à $\mathbf{w}$. Vérifie que le milieu $\frac{\boldsymbol{\mu}_0 + \boldsymbol{\mu}_1}{2}$ satisfait l'équation. Pour 4, un point équidistant de $\boldsymbol{\mu}_0$ et $\boldsymbol{\mu}_1$, et de $\boldsymbol{\mu}_1$ et $\boldsymbol{\mu}_2$, l'est aussi de…

</details>
<details><summary>Indice 3</summary>

3. $2(4, 2) \cdot (x, y) + 5 - 41 = 0$, à simplifier. 4. Le centre du cercle qui passe par les trois centroïdes. 5. Les points de la classe étalée débordent loin de leur centroïde : où devrait passer la frontière pour en classer davantage correctement ? Le classifieur ne connaît que deux moyennes.

</details>

<a id="reflexion"></a>

## 🗣️ 🛠️ ⚖️ Réflexion

### Ex 7.8 — La malédiction de la dimension en cinq lignes

<details><summary>Indice 1</summary>

Relis le §7.6 de la fiche : la densité, la figure 7.17 du livre et les parades.

</details>
<details><summary>Indice 2</summary>

Une image possible : des invités répartis dans un immeuble dont on multiplie les étages et les pièces. Quelle parade proposer à un agronome ?

</details>
<details><summary>Indice 3</summary>

Plan en cinq phrases : ce qu'il espère (plus de mesures, de meilleures prévisions) ; la densité (150 exemples perdus dans un espace immense) ; ce qui se passe (le modèle devine, ou apprend le bruit) ; la structure qui sauve parfois ; une parade (choisir les mesures utiles, résumer, plus de parcelles, régulariser).

</details>

### Ex 7.9 — Lire la documentation officielle de KMeans (scikit-learn)

<details><summary>Indice 1</summary>

La page d'une classe de scikit-learn a toujours les mêmes rubriques : *Parameters*, *Attributes*, *See also*, *Notes*, *Examples*, puis les méthodes. Dans la description d'un paramètre, des mentions « Added in version » et « Changed in version » datent ses évolutions.

</details>
<details><summary>Indice 2</summary>

1 est dans *Notes* ; 2 dans la description du paramètre `n_init` ; 3 compare la description de `tol` et la docstring de ton `mylearn/cluster.py` ; 4 dans la méthode `transform`, puis pense à l'indexation avancée de NumPy (0A) ; 5 à la fin de la description de `n_init`.

</details>
<details><summary>Indice 3</summary>

4 : une ligne par point, une colonne par centre ; dans la ligne $i$, prends la colonne `labels_[i]`, qui est aussi celle du minimum de la ligne. 5 : entre les deux versions, un avertissement (`FutureWarning`) prévient ceux dont le code compte sur l'ancienne valeur : c'est le cycle de dépréciation de scikit-learn. 6 : pense au nombre de départs par défaut, à la variante de k-means++, au sort d'un cluster vide, et aux options que `mylearn` n'a pas (`algorithm`, `sample_weight`, `init` en fonction).

</details>

### Ex 7.10 — Qui fixe le seuil ? Œufs, dépistage et coût des erreurs

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur le seuil qui minimise le coût moyen (§7.2.1 de la fiche).

</details>
<details><summary>Indice 2</summary>

$t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$. Qui est un faux positif ici : un œuf non fécondé déclaré fécondé, ou l'inverse ? Pour 2 à 4, liste les personnes concernées par chaque erreur.

</details>
<details><summary>Indice 3</summary>

$t^* = \frac{0{,}30}{0{,}30 + 50}$, moins de 1 % : presque tout œuf un peu douteux part en incubation. Les coûts traduisent des choix (économiques, éthiques, réglementaires) qui ne relèvent pas de la seule équipe technique. Pour 4, compare la calibration et les taux d'erreur groupe par groupe (bonus B6).

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

## Notebook

Les indices des exercices du notebook (7.11 à 7.31) seront ajoutés à la prochaine session de génération, avec le notebook complet.
