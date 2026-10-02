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

```python
n_fertile_11 = int(y_11.sum())
model_11 = SklearnNearestCentroid().fit(X_11, y_11)
train_acc_11 = model_11.score(X_11, y_11)
new_acc_11 = model_11.score(X_new_11, y_new_11)
pred_11 = model_11.predict(EGGS_11).tolist()
```

</details>

### Ex 7.12 — k-means sur deux lunes : où tombera la coupure ? 🔮

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur l'algorithme de Lloyd : quand il s'arrête, à quel centre chaque point du plan est-il rattaché ?

</details>
<details><summary>Indice 2</summary>

Deux centres, et chaque point rejoint le plus proche : quelle est la forme de l'ensemble des points à égale distance de deux points (∂ 7.7) ? Une telle frontière peut-elle suivre deux lunes qui s'emboîtent ?

</details>
<details><summary>Indice 3</summary>

La frontière est la médiatrice des deux centres. Elle ne peut pas épouser les lunes : elle coupe chacune d'elles. Imagine la coupure qui sépare le mieux possible deux lunes emboîtées, et estime la part de chaque lune qui passe du mauvais côté.

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
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2-D arrays, got shapes {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must have the same number of columns, got {A.shape[1]} and {B.shape[1]}")
    sq = (A ** 2).sum(axis=1)[:, None] - 2 * A @ B.T + (B ** 2).sum(axis=1)[None, :]
    return np.maximum(sq, 0.0)
```

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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"X must be 2-D with one row per label, got shapes {X.shape} and {y.shape}")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"at least 2 classes are needed, got {len(classes)}")
    self.classes_ = classes
    self.centroids_ = np.array([X[y == c].mean(axis=0) for c in classes])
    return self

def decision_function(self, X):
    D = pairwise_sq_distances(X, self.centroids_)
    return D[:, 0] - D[:, 1] if len(self.classes_) == 2 else -D

def predict(self, X):
    return self.classes_[pairwise_sq_distances(X, self.centroids_).argmin(axis=1)]

def score(self, X, y):
    return float(np.mean(self.predict(X) == np.asarray(y)))
```

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
    X = np.atleast_2d(np.asarray(X, dtype=float))
    f_1 = scipy.stats.multivariate_normal(MEAN_1_15, COV_1_15).pdf(X)
    f_0 = scipy.stats.multivariate_normal(MEAN_0_15, COV_0_15).pdf(X)
    return np.atleast_1d(PRIOR_15 * f_1 / (PRIOR_15 * f_1 + (1 - PRIOR_15) * f_0))


t_star_15 = COST_FP_15 / (COST_FP_15 + COST_FN_15)
```

Pour d) et e), une boucle sur les deux seuils : `declared = posterior_15(eggs_15) >= threshold`, puis `np.sum(~declared & (truth_15 == 1))` (FN) et `np.sum(declared & (truth_15 == 0))` (FP) ; le coût moyen est $\frac{6\,FN + FP}{2\,000}$.

</details>

### Ex 7.16 — Un-contre-tous avec des centroïdes : quelle classe sera sacrifiée ? 🔮

<details><summary>Indice 1</summary>

Pour chaque classe $k$, le classifieur « $k$ contre le reste » n'a que deux centroïdes : celui de $k$ et celui de tout le reste. Place les quatre centres de `LAYOUT_16` sur une feuille, puis, pour chaque classe, le centroïde de « tout sauf elle ».

</details>
<details><summary>Indice 2</summary>

Les quatre classes ont autant de points : le centroïde de « tout sauf A » est la moyenne des centres de B, C et D. Calcule-le de même pour chacune des quatre classes : pour laquelle tombe-t-il presque sur son propre centre ? Que vaut alors, à peu près partout, le score de son classifieur ?

</details>
<details><summary>Indice 3</summary>

Le centroïde de « tout sauf D » est la moyenne de $(-4, -2)$, $(4, -2)$ et $(0, 5)$ : $(0 ; 0{,}33)$, presque exactement le centre de D. Le classifieur de D compare deux centroïdes presque confondus : son score reste proche de 0, quand ceux des autres classes sont francs. D ne gagne que là où les trois autres scores sont négatifs : une zone étroite. Une partie de ses points y tombe, pas tous.

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
    table = pd.crosstab(np.asarray(labels), np.asarray(truth))
    return float(table.max(axis=1).sum() / len(labels))


raw_labels_17 = SklearnKMeans(n_clusters=3, n_init=10, random_state=0).fit(X_peng).labels_
raw_purity_17 = purity_17(raw_labels_17, species)
Z_17 = (X_peng - X_peng.mean(axis=0)) / X_peng.std(axis=0)
std_labels_17 = SklearnKMeans(n_clusters=3, n_init=10, random_state=0).fit(Z_17).labels_
std_purity_17 = purity_17(std_labels_17, species)
ari_17 = adjusted_rand_score(species, std_labels_17)
```

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
    real = group_18 != -1
    return (len(set(labels.tolist()) - {-1}), int(np.sum(labels == -1)),
            float(adjusted_rand_score(group_18[real], labels[real])))


km_labels_18 = SklearnKMeans(n_clusters=4, n_init=10, random_state=0).fit(X_18).labels_
db_labels_18 = DBSCAN(eps=0.2, min_samples=5).fit_predict(X_18)
scan_18 = {eps: summary_18(DBSCAN(eps=eps, min_samples=5).fit_predict(X_18)) for eps in EPS_18}
hdb_labels_18 = HDBSCAN(min_cluster_size=10).fit_predict(X_18)
```

</details>

### Ex 7.19 — Distance au plus proche voisin quand la dimension grimpe 🔮

<details><summary>Indice 1</summary>

Commence par la distance entre deux points quelconques : chaque coordonnée ajoute sa part au carré de la distance. Puis demande-toi comment 500 points « remplissent » un cube quand la dimension grandit (✏️ 7.4).

</details>
<details><summary>Indice 2</summary>

En dimension $d$, le carré de la distance entre deux points uniformes est une somme de $d$ termes indépendants, chacun de moyenne $\frac{1}{6}$ : que devient sa moyenne quand $d$ grandit ? Et le plus proche voisin ? Avec 500 points et 2 cases par axe, combien de cases y a-t-il en dimension 10, et combien de points par case ?

</details>
<details><summary>Indice 3</summary>

Les deux distances grandissent. En dimension 1, 500 points sur un segment : le voisin est tout près, et le rapport est minuscule. En dimension 10, il y a déjà 1 024 cases pour 500 points : la plupart des points sont seuls dans leur case, et leur voisin est loin. En proportion, le plus proche voisin rattrape la distance moyenne : le rapport monte vers 1.

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
    cells = np.minimum(np.floor(np.asarray(X) * bins).astype(int), bins - 1)
    return len(np.unique(cells, axis=0))


def _distances_20(X):
    D = np.sqrt(mylearn.cluster.pairwise_sq_distances(X, X))
    np.fill_diagonal(D, 0.0)
    return D


def nn_and_mean_20(X):
    D, off = _distances_20(X), ~np.eye(len(X), dtype=bool)
    return float(np.where(off, D, np.inf).min(axis=1).mean()), float(D[off].mean())


def contrast_20(X):
    D, off = _distances_20(X), ~np.eye(len(X), dtype=bool)
    d_min = np.where(off, D, np.inf).min(axis=1)
    d_max = np.where(off, D, -np.inf).max(axis=1)
    return float(np.mean((d_max - d_min) / d_min))


expected_20 = 4 ** 5 * (1 - (1 - 1 / 4 ** 5) ** 1000)
```

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
    volumes = {1: 2.0, 2: np.pi}
    for d in range(3, d_max + 1):
        volumes[d] = 2 * np.pi / d * volumes[d - 2]
    return np.array([volumes[d] / 2 ** d for d in range(1, d_max + 1)])


def orange_radius_21(d):
    return np.sqrt(d) - 1
```

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
    if len(classes) < 3:
        raise ValueError(f"at least 3 classes are needed, got {len(classes)}: use the binary classifier directly")
    if not (hasattr(self.estimator, "decision_function") or hasattr(self.estimator, "predict_proba")):
        raise ValueError("the estimator needs decision_function or predict_proba")
    self.classes_ = classes
    self.estimators_ = [copy.deepcopy(self.estimator).fit(X, (y == c).astype(int)) for c in classes]
    return self

def decision_function(self, X):
    columns = []
    for model in self.estimators_:
        if hasattr(model, "decision_function"):
            columns.append(np.ravel(model.decision_function(X)))
        else:
            columns.append(np.asarray(model.predict_proba(X))[:, 1])
    return np.column_stack(columns)
```

`predict` et `score` tiennent chacune en une ligne. (La liste en compréhension suppose que `fit` renvoie le modèle, comme dans scikit-learn.)

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
    if K < 3:
        raise ValueError(f"at least 3 classes are needed, got {K}")
    self.pairs_ = [(i, j) for i in range(K) for j in range(i + 1, K)]
    self.estimators_ = []
    for i, j in self.pairs_:
        keep = (y == self.classes_[i]) | (y == self.classes_[j])
        model = copy.deepcopy(self.estimator)
        model.fit(X[keep], (y[keep] == self.classes_[j]).astype(int))
        self.estimators_.append(model)
    return self

def votes(self, X):
    votes = np.zeros((len(X), len(self.classes_)), dtype=int)
    for (i, j), model in zip(self.pairs_, self.estimators_):
        for_j = np.asarray(model.predict(X)) == 1
        votes[:, j] += for_j
        votes[:, i] += ~for_j
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

```python
def compare_24(X_train, y_train, X_test, y_test):
    rows = {}
    for name, model in [("native", mylearn.cluster.NearestCentroid()),
                        ("OvR", mylearn.multiclass.OneVsRestClassifier(mylearn.cluster.NearestCentroid())),
                        ("OvO", mylearn.multiclass.OneVsOneClassifier(mylearn.cluster.NearestCentroid()))]:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fitted = time.perf_counter()
        predicted = model.predict(X_test)
        done = time.perf_counter()
        rows[name] = {"n_models": len(getattr(model, "estimators_", [model])),
                      "accuracy": float(np.mean(predicted == np.asarray(y_test))),
                      "fit_s": fitted - start, "predict_s": done - fitted}
    return pd.DataFrame.from_dict(rows, orient="index")
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
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D array, got shape {X.shape}")
    n_distinct = len(np.unique(X, axis=0))
    if not 1 <= n_clusters <= n_distinct:
        raise ValueError(f"n_clusters must be between 1 and {n_distinct}, got {n_clusters}")
    rng = np.random.default_rng() if rng is None else rng
    chosen = [int(rng.integers(len(X)))]
    d2 = ((X - X[chosen[0]]) ** 2).sum(axis=1)
    for _ in range(1, n_clusters):
        chosen.append(int(rng.choice(len(X), p=d2 / d2.sum())))
        d2 = np.minimum(d2, ((X - X[chosen[-1]]) ** 2).sum(axis=1))
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

```python
def _lloyd(X, centres, max_iter, tol):
    previous, converged = np.full(len(X), -1), False
    for i in range(max_iter):
        labels = pairwise_sq_distances(X, centres).argmin(axis=1)
        new = centres.copy()
        for j in range(len(centres)):
            if np.any(labels == j):
                new[j] = X[labels == j].mean(axis=0)
        shift = float(((new - centres) ** 2).sum())
        centres = new
        if np.array_equal(labels, previous):
            converged = True
            break
        if shift <= tol:
            break
        previous = labels
    if not converged:
        labels = pairwise_sq_distances(X, centres).argmin(axis=1)
    return centres, labels, float(((X - centres[labels]) ** 2).sum()), i + 1
```

Dans `fit`, après la validation : `rng = np.random.default_rng(self.random_state)`, `tol = self.tol * np.mean(np.var(X, axis=0))`, puis `n_init` exécutions (une seule si `init` est un tableau), chacune avec ses centres initiaux (`kmeans_plusplus(X, k, rng=rng)`, `X[rng.choice(n, size=k, replace=False)]` ou `np.array(self.init, dtype=float)`). Garde l'exécution dont l'inertie est **strictement** plus petite que la meilleure déjà vue.

</details>

### Ex 7.27 — k-means piégé : quatre bugs à débusquer 🐛

<details><summary>Indice 1</summary>

Isole les bugs un par un, comme le fera la vérification : regarde si `init_27` a changé après l'appel ; lance la fonction avec `max_iter=1` et compare ses centres aux moyennes que tu calcules toi-même ; regarde `n_iter_c27` ; recalcule l'inertie avec les centres et les labels renvoyés.

</details>
<details><summary>Indice 2</summary>

Quatre lignes sont suspectes : `centres = init`, la mise à jour `X[labels == j].mean()`, les deux lignes autour de la comparaison des labels, et le calcul de l'inertie. Pour chacune, pose-toi une question : copie ou même tableau ? moyenne de quoi, sur quel axe ? dans quel ordre ? des carrés ou pas ?

</details>
<details><summary>Indice 3</summary>

Bug 1 : `centres = np.array(init, dtype=float)`. Bug 2 : `X[labels == j].mean(axis=0)`. Bug 3 : compare `labels` à `old_labels` **avant** d'écrire `old_labels = labels`. Bug 4 : `inertia = ((X - centres[labels]) ** 2).sum()`.

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
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    n, K = len(X), codes.max() + 1
    if not 2 <= K <= n - 1:
        raise ValueError(f"the number of distinct labels must be between 2 and {n - 1}, got {K}")
    D = np.sqrt(pairwise_sq_distances(X, X))
    np.fill_diagonal(D, 0.0)
    sizes = np.bincount(codes)
    sums = np.column_stack([D[:, codes == c].sum(axis=1) for c in range(K)])
    rows = np.arange(n)
    own = sizes[codes]
    a = sums[rows, codes] / np.maximum(own - 1, 1)
    means = sums / sizes
    means[rows, codes] = np.inf
    b = means.min(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (b - a) / np.maximum(a, b)
    s[own == 1] = 0.0
    return np.nan_to_num(s)
```

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
        inertias[k] = model.inertia_
        silhouettes[k] = mylearn.cluster.silhouette_score(X, model.labels_)
    return inertias, silhouettes
```

d) `max(silhouettes_29, key=silhouettes_29.get)`. e) Compare la baisse de 2 à 3 à celle de 3 à 4 : l'une est plus de dix fois plus grande que l'autre.

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
    return np.asarray(y_train)[mylearn.cluster.pairwise_sq_distances(X_test, X_train).argmin(axis=1)]


def hughes_30(n_real, n_noise):
    F = features_30(n_real, n_noise)
    y_train, y_test = species[TRAIN_PENG], species[TEST_PENG]
    nc = mylearn.cluster.NearestCentroid().fit(F[TRAIN_PENG], y_train).score(F[TEST_PENG], y_test)
    nn = float(np.mean(nn1_predict_30(F[TRAIN_PENG], y_train, F[TEST_PENG]) == y_test))
    return [nc, nn]
```

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
