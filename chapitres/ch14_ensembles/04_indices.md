# 14 · Ensembles — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 14.Q1 — Pourquoi un groupe peut battre chacun de ses membres

<details><summary>Indice 1</summary>

Relis le §14.2 de la fiche : quelles sont les **deux** conditions pour qu'un vote fasse mieux que ses membres ?

</details>
<details><summary>Indice 2</summary>

Pour a), l'encadré sur la loi binomiale donne la probabilité qu'une majorité de votants indépendants ait raison : compare-la à $q$ quand $q > 0{,}5$. Pour c), demande-toi si dix copies identiques peuvent jamais être en désaccord.

</details>
<details><summary>Indice 3</summary>

a) Avec $q = 0{,}7 > 0{,}5$ et des votants indépendants, la majorité a raison plus souvent que chaque votant, et d'autant plus qu'ils sont nombreux : décide si l'affirmation le dit. b) Élimine les options qui ne créent aucune diversité ou qui créent des égalités. c) Que prédit la majorité de dix votes toujours identiques ? d) Quand une copie tombe en panne, combien en reste-t-il de justes, et que décide la majorité ?

</details>

### 14.Q2 — Un ensemble corrige-t-il un biais des données ?

<details><summary>Indice 1</summary>

Un bootstrap tire dans les **mêmes** données : que deviennent les proportions de chaque groupe d'un bootstrap à l'autre ?

</details>
<details><summary>Indice 2</summary>

Sépare ce qui **diffère** d'un membre à l'autre (ce que le vote peut moyenner) de ce que **tous** partagent (ce que le vote garde). Pour b) et c), relis le §14.2 de la fiche, qui résume les deux exemples du livre.

</details>
<details><summary>Indice 3</summary>

a) Chaque bootstrap reproduit à peu près les proportions des données d'origine : le groupe y reste sous-représenté, dans tous les arbres ; décide si le vote peut le corriger. b) Relis la proposition du livre au §14.2 de la fiche : que diversifie-t-il ? c) Que change-t-on d'un membre à l'autre en prévision d'ensemble ? d) Une erreur que tous font est-elle de la variance ?

</details>

### 14.Q3 — Pluralité, majorité et vote pondéré

<details><summary>Indice 1</summary>

Compte les voix de chaque classe, puis compare au nombre total de votants.

</details>
<details><summary>Indice 2</summary>

Pour c) et d), chaque classe part de 0 et reçoit le poids de **chacun** de ses votants, signe compris. Repère les votants de C : ce sont les numéros 2 et 6.

</details>
<details><summary>Indice 3</summary>

a) A reçoit les voix des votants 0, 3 et 5 : 3 voix ; compte celles de B et de C. b) Plus de la moitié de 7 voix, c'est au moins combien ? c) Additionne les poids des votants 2 et 6. d) Fais de même pour A (votants 0, 3, 5) et B (votants 1, 4), puis compare les trois totaux.

</details>

### 14.Q4 — Tirer avec remise

<details><summary>Indice 1</summary>

« Avec remise » : après chaque tirage, l'exemple retourne dans l'urne.

</details>
<details><summary>Indice 2</summary>

Pour b) et c), cherche les deux cas extrêmes : quatre tirages tous identiques, ou quatre tous différents. Pour d), relis l'encadré sur la limite de $(1 - 1/n)^n$.

</details>
<details><summary>Indice 3</summary>

a) Un exemple remis dans l'urne peut ressortir au tirage suivant : décide. b) Si les quatre tirages donnent le même exemple, combien en a-t-on de distincts ? c) Au mieux, combien d'exemples distincts quatre tirages peuvent-ils toucher, et combien en reste-t-il parmi 8 ? d) La part d'exemples jamais tirés tend vers $e^{-1}$ ; la part des distincts en est le complément. e) Sans remise, chaque exemple peut-il sortir deux fois ?

</details>

### 14.Q5 — Ce que le bagging réduit

<details><summary>Indice 1</summary>

Relis le paragraphe « Pourquoi le bagging marche » du §14.4 : moyenner des modèles touche-t-il au biais qu'ils partagent ?

</details>
<details><summary>Indice 2</summary>

Pour b), le bagging aide les modèles dont les prédictions changent beaucoup d'un échantillon à l'autre : lequel des quatre est le plus instable ? Pour c), pense aux exemples out-of-bag.

</details>
<details><summary>Indice 3</summary>

a) Une moyenne de modèles qui varient à cause des données réduit cette variation : quel nom porte-t-elle au ch. 9 ? b) Pour chaque modèle, demande-toi si un autre tirage des données changerait beaucoup ses prédictions. c) Un arbre voit-il les exemples que son bootstrap n'a pas tirés ? d) Avec quels arbres prédit-on un exemple pour son score out-of-bag ?

</details>

### 14.Q6 — Le hasard d'une forêt aléatoire

<details><summary>Indice 1</summary>

`"sqrt"` veut dire la partie entière de la racine carrée du nombre de features.

</details>
<details><summary>Indice 2</summary>

Relis le pseudo-code du §14.5 : à quelle ligne le tirage des features a-t-il lieu ? Pour d), sépare ce qu'ajoute la forêt au bagging de ce qu'ajoute ExtraTrees à la forêt.

</details>
<details><summary>Indice 3</summary>

a) $\lfloor\sqrt{16}\rfloor$. b) Dans quelle fonction du pseudo-code le tirage a-t-il lieu, et combien de fois est-elle appelée ? c) Examiner deux fois la même feature aurait-il un sens ? d) Fais la liste de ce qui est tiré au hasard dans le bagging, puis de ce que la forêt y ajoute ; garde seulement ces deux-là. e) Sans tirage de features, que reste-t-il d'aléatoire ?

</details>

### 14.Q7 — ExtraTrees : encore plus de hasard

<details><summary>Indice 1</summary>

Relis le §14.6 de la fiche : qu'est-ce qui est tiré au hasard, et dans quel intervalle ?

</details>
<details><summary>Indice 2</summary>

Pour c), compare un seuil tiré au hasard et le meilleur seuil d'un même nœud : lequel fait le plus baisser l'impureté ? Pour d), relis la ligne « coût d'un nœud » du tableau.

</details>
<details><summary>Indice 3</summary>

a) Un seuil par feature candidate, tiré uniformément entre deux bornes : lesquelles ? b) Relis le défaut `bootstrap` d'ExtraTrees dans scikit-learn. c) Par définition, le meilleur seuil donne le plus grand gain au nœud ; un seuil tiré peut-il faire mieux ? d) Relis la ligne « coût d'un nœud » du tableau du §14.6.

</details>

### 14.Q8 — Apprenant faible, souche et apprenant fort

<details><summary>Indice 1</summary>

Une souche est un arbre réduit à une seule question.

</details>
<details><summary>Indice 2</summary>

Pour c), regarde ce que devient une erreur quand on échange les deux réponses. Pour e), relis le §14.7 de la fiche.

</details>
<details><summary>Indice 3</summary>

a) Une question, deux réponses : combien de feuilles ? b) Combien d'arêtes entre la racine et une feuille ? c) Si un classifieur se trompe dans 60 % des cas, que devient ce taux après l'échange ? d) Que vaut l'accuracy du hasard à deux classes équilibrées, et un apprenant faible doit-il faire plus ou moins ? e) Relis la dernière phrase du paragraphe sur les apprenants faibles et forts (§14.7).

</details>

### 14.Q9 — Ce que repondère AdaBoost

<details><summary>Indice 1</summary>

Relis la formule de $\alpha_m$ et celle de la mise à jour des $w_i$ au §14.7.

</details>
<details><summary>Indice 2</summary>

Pour b), calcule $\ln\frac{1 - 0{,}5}{0{,}5}$ et $\ln(K - 1)$ avec $K = 2$. Pour e), relis le pseudo-code : quelle condition fait rejeter un membre ?

</details>
<details><summary>Indice 3</summary>

a) Les mal classés sont multipliés par $e^{\alpha_m}$, avec $\alpha_m > 0$ : plus ou moins de 1 ? b) Calcule les deux logarithmes de la formule et additionne-les. c) Regarde comment varie $\frac{1 - \varepsilon}{\varepsilon}$ quand $\varepsilon$ baisse. d) De quoi le membre $m$ a-t-il besoin avant d'être entraîné ? e) Le seuil vaut $1 - 1/K$ : remplace $K$.

</details>

### 14.Q10 — Bagging ou boosting ?

<details><summary>Indice 1</summary>

Une seule question par phrase : les membres sont-ils entraînés **indépendamment** (bagging) ou **l'un après l'autre** (boosting) ?

</details>
<details><summary>Indice 2</summary>

Relis le tableau « Bilan » de la fiche, ligne par ligne : membres, modèle de base, ce qui baisse, bruit des labels. Pour c), relis le paragraphe sur le nombre d'arbres au §14.4.

</details>
<details><summary>Indice 3</summary>

a) Phrase 1 : des membres entraînés indépendamment, c'est le bagging : **B**. Applique la même question aux cinq autres, avec la ligne « ce qui baisse » pour 3 et 6 et la ligne « bruit des labels » pour 5. b) Une souche qui underfit a un problème de biais ou de variance ? c) Que dit la fiche sur une forêt à laquelle on ajoute des arbres ?

</details>

### 14.Q11 — Gradient boosting : corriger les résidus

<details><summary>Indice 1</summary>

Relis la formule $F_0 = \bar y$, $r_i = y_i - F_{m-1}(\mathbf{x}_i)$, $F_m = F_{m-1} + \eta\, h_m$.

</details>
<details><summary>Indice 2</summary>

Pour c), l'arbre prédit le résidu, et l'on n'en ajoute qu'une fraction $\eta$. Pour d) et e), relis le paragraphe sur le learning rate et le nombre d'arbres.

</details>
<details><summary>Indice 3</summary>

a) Quelle constante minimise la somme des carrés des écarts ? b) Le résidu est la cible moins la prédiction. c) Ajoute $0{,}1 \times$ le résidu à la prédiction actuelle. d) Si chaque arbre ne corrige qu'un dixième de l'erreur, faut-il plus ou moins d'arbres ? e) Que dit la fiche d'un gradient boosting auquel on ajoute des arbres sans contrôle ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 14.R1 — Ch. 13 : L'impureté de Gini d'un nœud

<details><summary>Indice 1</summary>

$G = 1 - \sum_c p_c^2$, avec les proportions $p_c$ des classes dans le nœud (fiche du ch. 13, §13.5.2).

</details>
<details><summary>Indice 2</summary>

Pour c), le gain pondère chaque enfant par sa part des exemples. Pour d), avec des poids, $p_c$ est la somme des poids de la classe $c$ divisée par la somme de tous les poids.

</details>
<details><summary>Indice 3</summary>

a) Proportions 0,6, 0,3 et 0,1 : $G = 1 - (0{,}36 + 0{,}09 + 0{,}01)$. b) L'enfant droit a 3 exemples de la classe 1 et 1 de la classe 2 : applique la même formule. c) L'enfant gauche est pur ; pondère l'enfant droit par 4/10. d) Les poids des classes valent 6, 3 et 4 : calcule les proportions sur 13. e) Les proportions changent-elles si tous les poids doublent ?

</details>

### 14.R2 — Ch. 11 : Induction et généralisation

<details><summary>Indice 1</summary>

Relis les deux façons de raisonner du ch. 11 : laquelle donne une conclusion seulement probable ?

</details>
<details><summary>Indice 2</summary>

Pour b), demande-toi ce qu'un bootstrap peut contenir qui ne soit pas déjà dans l'échantillon. Pour c), relis le tableau des sophismes inductifs du ch. 11 : lequel est un défaut de **collecte** ?

</details>
<details><summary>Indice 3</summary>

a) On tire une règle d'exemples observés pour l'appliquer à d'autres : de quel côté du tableau déduction/induction est-on ? b) Un bootstrap ne tire que dans l'échantillon : le biais peut-il disparaître ? c) Pour chaque sophisme proposé, demande-toi si l'erreur vient du raisonnement ou de la façon dont les données ont été réunies. d) Un ensemble reste-t-il entraîné sur des données passées ?

</details>

### 14.R3 — Ch. 7 : Un-contre-tous et un-contre-un

<details><summary>Indice 1</summary>

Un-contre-tous : un classifieur par classe ; un-contre-un : un classifieur par **paire** de classes.

</details>
<details><summary>Indice 2</summary>

Le nombre de paires parmi $K$ classes est $\binom{K}{2} = K(K - 1)/2$.

</details>
<details><summary>Indice 3</summary>

a) Une classe contre toutes les autres, pour chacune des 5 classes : 5 classifieurs. Pour b) et c), applique $K(K - 1)/2$. d) Le classifieur « classe 1 contre classe 3 » a-t-il besoin des exemples de la classe 2 ? e) Les positifs d'un classifieur un-contre-tous sont les exemples d'une seule classe : quelle part du total, avec 4 classes de même taille ?

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 14.1 — Dépouiller un vote pondéré

<details><summary>Indice 1</summary>

Fais un tableau : une ligne par exemple, une colonne par classe ; dans chaque case, la somme des poids des votants qui ont prédit cette classe pour cet exemple.

</details>
<details><summary>Indice 2</summary>

N'oublie pas le signe du votant 3 : son poids $-1$ **retire** 1 à la classe qu'il désigne. En cas d'égalité, la plus petite classe gagne. Pour f), écris le score de la classe 1 de l'exemple 1 en fonction du poids $w_3$.

</details>
<details><summary>Indice 3</summary>

a) Exemple 0 : la classe 1 est prédite par les votants 1 (poids 1) et 2 (poids 1,5), d'où 2,5. Remplis de même les scores des classes 0 et 2 de chaque exemple et prends le maximum (b) à d)). e) Refais le tableau avec un poids 1 pour chacun. f) La classe 1 de l'exemple 1 reçoit $2 + 1 + w_3$, la classe 0 reçoit 2 : quelle condition stricte sur $w_3$ ? g) et h) Que deviennent les scores quand chaque poids est multiplié par 3, ou quand chaque **voix** gagne 10 ?

</details>

### Ex 14.2 — Additionner les voix de trois droites, région par région

<details><summary>Indice 1</summary>

Une région est définie par les apprenants qui la mettent de leur côté positif ; son score est la somme de leurs poids.

</details>
<details><summary>Indice 2</summary>

Pour situer un point, teste les trois conditions $x_1 \ge 1$, $x_2 \ge 1$ et $x_1 + x_2 \le 3$. Pour h), écris $\mathbb{1}[\text{côté positif}]$ en fonction de $s_m = \pm 1$ : $\mathbb{1} = (s_m + 1)/2$.

</details>
<details><summary>Indice 3</summary>

a) La région « C » ne reçoit que le poids de C : $-3$. Calcule de même les six autres régions (ABC, AB, AC, A, BC, B) et compte celles de score strictement positif pour d). b) et c) Situe chaque point par les trois tests, puis lis le score de sa région. e) Pour $P_2$, calcule $x_1 + x_2$ avant de conclure. f) Refais les sept scores avec $w_C = -1$. g) Compare les prédictions de A seul aux vraies classes. h) Remplace chaque indicatrice par $(s_m + 1)/2$ et sépare ce qui dépend de $s_m$ de ce qui n'en dépend pas.

</details>

### Ex 14.3 — Probabilité d'être oublié par un bootstrap : vers 36,8 %

<details><summary>Indice 1</summary>

Pour un tirage, l'indice donné a une chance sur $n$ d'être choisi ; les tirages sont indépendants.

</details>
<details><summary>Indice 2</summary>

Pour 2, pose $I_k = 1$ si l'indice $k$ est tiré au moins une fois : $\mathbb{E}[I_k]$ est le complément du résultat de 1, et la linéarité de l'espérance ne demande pas l'indépendance des $I_k$. Pour 3, pose $u = -1/n$. Pour 4, calcule d'abord la probabilité que l'exemple soit **dans** les $M$ bootstraps.

</details>
<details><summary>Indice 3</summary>

1. La probabilité de ne pas être choisi à un tirage vaut $1 - 1/n$ ; multiplie pour $d$ tirages indépendants. 2. $D = \sum_k I_k$ et $\mathbb{E}[I_k] = 1 - (1 - 1/n)^d$ : additionne les $n$ termes. 3. $n \ln(1 - 1/n) = -\ln(1 + u)/u$ avec $u = -1/n \to 0$. 4. Le complément de « dans tous les bootstraps ».

Vérifications : a) $(7/8)^4$ ; b) $8 \times (1 - (7/8)^4)$ ; applique de même les formules aux autres valeurs ; pour h), utilise la question 4 avec $e^{-1}$ à la place de $(1 - 1/n)^n$.

</details>

### Ex 14.4 — Ajuster les résidus, c'est descendre le gradient de la MSE

<details><summary>Indice 1</summary>

Traite chaque $F_i$ comme une variable ordinaire : seul le terme $i$ de la somme en dépend.

</details>
<details><summary>Indice 2</summary>

Pour 3 et 4, annule la dérivée par rapport à la constante. Pour 5, la dérivée de $\lvert t \rvert$ vaut $\operatorname{signe}(t)$ pour $t \ne 0$ ; pour la constante, regarde ce que devient $\sum_i \lvert y_i - c \rvert$ quand $c$ passe un $y_i$. Pour 7, réutilise $d\ell/dz = \hat p - y$ de ∂ 13.8.

</details>
<details><summary>Indice 3</summary>

1. $\frac{\partial}{\partial F_i} \frac12 (y_i - F_i)^2 = -(y_i - F_i)$ par la règle de la chaîne. 2. Écris $F_i \leftarrow F_i - \eta\, \partial L/\partial F_i$, puis demande-toi quelle valeur ce pas donne à un exemple qui n'est pas dans la somme. 3. $\frac{d}{dc}\sum_i (y_i - c)^2 = -2\sum_i (y_i - c)$. 4. Pose $r_i = y_i - F_i$ et reconnais le problème 3. 5. Quand $c$ augmente d'un petit $\delta$, chaque point au-dessus de $c$ rapproche de $\delta$, chaque point en dessous éloigne de $\delta$ : quand l'équilibre est-il atteint ? 6. Une feuille par exemple prédit exactement son résidu : que vaut $F_1$ ? 7. Écris $\ell$ comme une fonction de $F$ par $\hat p = \sigma(F)$, et réutilise la dérivée de ∂ 13.8.

</details>

### Ex 14.5 — Quand la majorité de votants indépendants se trompe

<details><summary>Indice 1</summary>

La majorité a raison quand plus de la moitié des votants ont raison : additionne les probabilités de « $j$ votants justes » pour tous les $j > M/2$.

</details>
<details><summary>Indice 2</summary>

Chaque terme vaut $\binom{M}{j} q^j (1 - q)^{M - j}$ (encadré sur la loi binomiale). Pour f), demande-toi si les trois votes peuvent être différents. Pour g), calcule la probabilité pour $M = 3, 5, 7, \dots$ jusqu'à dépasser 0,9 (`scipy.stats.binom.sf(M // 2, M, q)` le fait pour toi).

</details>
<details><summary>Indice 3</summary>

a) $M = 3$ : il faut 2 ou 3 justes : $0{,}7^3 + 3 \times 0{,}7^2 \times 0{,}3$. Pour b) à e), h) et i), applique la même somme avec $j$ de $\lceil (M + 1)/2 \rceil$ à $M$ (un seul terme pour e)). f) Trois votes identiques : la majorité vaut quel votant ? g) Calcule la somme pour les $M$ impairs successifs et arrête-toi au premier qui atteint 0,9.

</details>

### Ex 14.6 — Une itération d'AdaBoost à la main

<details><summary>Indice 1</summary>

À chaque tour : l'erreur pondérée est la somme des poids des exemples mal classés ; $\alpha = \ln\frac{1 - \varepsilon}{\varepsilon}$ à deux classes ; les mal classés sont multipliés par $e^{\alpha}$, puis on normalise.

</details>
<details><summary>Indice 2</summary>

Au tour 1, tous les poids sont égaux : la meilleure souche est celle qui fait le moins d'erreurs. Au tour 2, compare le poids de l'exemple mal classé au tour 1 à celui des autres avant de chercher la meilleure souche. Garde des fractions : $e^{\ln a} = a$.

</details>
<details><summary>Indice 3</summary>

a) La souche « $x \le 4{,}5$ → 0, sinon 1 » ne se trompe que sur $x = 2$ : $\varepsilon_1 = 1/8$. b) $\ln\frac{7/8}{1/8}$. c) et d) Multiplie le poids de $x = 2$ par $e^{\alpha_1}$, additionne les huit poids, divise. e) et f) Parmi les souches qui classent bien $x = 2$, cherche celle dont les erreurs pèsent le moins avec les poids de c) et d). h) Là où les deux souches se contredisent, compare $\alpha_1$ et $\alpha_2$. i) à k) Mets d'abord à jour les poids du tour 2 (les erreurs de la souche 2, multipliées par $e^{\alpha_2}$, puis normalisation), puis suis le même schéma. l) Ajoute $\ln(K - 1)$. m) Compare 0,55 à $1 - 1/K$.

</details>

### Ex 14.7 — Deux itérations de gradient boosting sur cinq points

<details><summary>Indice 1</summary>

Une souche de régression teste les quatre seuils 1,5 ; 2,5 ; 3,5 ; 4,5 ; pour chacun, chaque feuille prédit la moyenne de ses **résidus**, et l'on garde le seuil de plus petite somme des carrés des écarts.

</details>
<details><summary>Indice 2</summary>

Les résidus de la première étape sont $y - F_0$ ; ceux de la seconde, $y - F_1$, où $F_1 = F_0 + 0{,}5 \times$ (la valeur de la feuille de chaque exemple). Pour un nouvel exemple, cherche de quel côté de chaque seuil il tombe.

</details>
<details><summary>Indice 3</summary>

a) La moyenne des cinq cibles. b) Retire-la à chaque cible. c) Pour le seuil 1,5 : la feuille gauche contient $r = -4$ (somme des carrés 0), la droite $(-3, 1, 2, 4)$, de moyenne 1 (somme $16 + 0 + 1 + 9 = 26$) ; calcule les trois autres seuils de la même façon et garde le plus petit total. d) Les moyennes des deux feuilles du seuil retenu. e) $F_0 + 0{,}5 \times$ la feuille gauche. f) Calcule les cinq nouveaux résidus et la moyenne de leurs carrés. g) Refais c) sur ces résidus. h) à j) Ajoute $0{,}5 \times$ la feuille de la seconde souche à $F_1$. k) $F_0$ plus la feuille gauche entière.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 14.8 — Bagging et boosting expliqués en 5 lignes

<details><summary>Indice 1</summary>

Cherche deux images : un groupe d'experts qui donnent leur avis **en même temps**, et une équipe qui travaille **à la chaîne**.

</details>
<details><summary>Indice 2</summary>

Pour chaque méthode, une phrase sur ce qui change d'un membre à l'autre (les données ? la cible ?), une sur ce qu'elle réduit (biais ou variance), une sur le réglage qu'elle demande.

</details>
<details><summary>Indice 3</summary>

Plan possible en cinq lignes : (1) l'image de la forêt, et ce qu'elle moyenne ; (2) l'image du boosting, et ce que chaque membre corrige ; (3) ce qui varie d'un membre à l'autre dans chacun ; (4) variance contre biais ; (5) quand choisir l'un ou l'autre, avec le risque d'overfitting quand on ajoute des membres.

</details>

### Ex 14.9 — Un jury de modèles biaisés reste biaisé

<details><summary>Indice 1</summary>

Relis le paragraphe « Un ensemble ne corrige pas un biais commun » du §14.2 et l'encadré sur la variance d'une moyenne de modèles corrélés.

</details>
<details><summary>Indice 2</summary>

Pour 2, pense aux informations d'un CV qui sont corrélées au genre sans le nommer. Pour 4, distingue une mesure qui ne demande que les décisions (un taux de sélection par groupe) d'une mesure qui demande une « vérité » sur la qualification.

</details>
<details><summary>Indice 3</summary>

1. Le biais est-il de la variance (ce qui diffère d'un modèle à l'autre) ou une erreur commune ? Que fait un vote d'une erreur commune ? 2. Activités, filières, interruptions de carrière, temps partiel… 3. D'autres décideurs, des évaluations à l'aveugle, une autre cible ; quel biais pourraient-elles partager ? 4. Taux de sélection, erreurs à qualification comparable, calibration, par groupe ; il faut l'attribut sensible pour l'audit, protégé. 5. Le système trie-t-il seul ? Relis l'encadré 🕰️ du §13.7 (RGPD, article 22 ; AI Act, systèmes à haut risque).

</details>

### Ex 14.10 — Pourquoi les arbres battent encore le deep learning sur le tabulaire

<details><summary>Indice 1</summary>

Commence par le résumé et la liste des contributions de l'introduction : la question et les réponses y sont annoncées.

</details>
<details><summary>Indice 2</summary>

La description du benchmark donne les critères de sélection et la taille des jeux ; la partie sur le réglage des hyperparamètres dit le budget et comment les résultats sont présentés. La section sur les biais inductifs suit trois expériences, chacune avec sa figure.

</details>
<details><summary>Indice 3</summary>

1. Cherche ce qui rendait les comparaisons antérieures peu fiables (des jeux et des budgets différents). 2. Liste les critères (types de colonnes, dimension, taille, difficulté…), la taille maximale d'entraînement et le nombre d'itérations de la recherche aléatoire. 3. Liste les modèles à base d'arbres et les réseaux comparés, puis regarde les figures de résultats par tâche (régression, classification, avec ou sans features catégorielles). 4. Pour chaque expérience : ce qu'on modifie (la cible, les features, l'orientation des données), et quelle famille en souffre le plus. 5. Relis la taille des jeux et la date ; compare avec l'encadré 🕰️ sur le gradient boosting.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 14.E1 — Bagging contre boosting : différences et cas d'usage

<details><summary>Indice 1</summary>

Deux axes : comment les membres sont entraînés (indépendamment ou l'un après l'autre) et ce que chaque méthode réduit.

</details>
<details><summary>Indice 2</summary>

Relis le tableau « Bilan » : modèle de base, combinaison, effet du nombre de membres, réglages, bruit. Termine par un critère de choix concret.

</details>
<details><summary>Indice 3</summary>

Structure : définition du bagging (bootstraps, moyenne, forêt), ce qu'il réduit et ses atouts (parallèle, out-of-bag, pas d'overfitting avec plus d'arbres) ; définition du boosting (séquentiel, repondération ou résidus), ce qu'il réduit et ses risques (réglages, early stopping, bruit) ; quand choisir l'un ou l'autre.

</details>

### 14.E2 — Ajouter des arbres à une forêt fait-il surapprendre ?

<details><summary>Indice 1</summary>

Que devient la moyenne de tirages indépendants d'un même procédé quand on en ajoute ?

</details>
<details><summary>Indice 2</summary>

Relis l'encadré sur la variance d'une moyenne corrélée (le plancher $\rho\sigma^2$), puis le paragraphe du gradient boosting sur le nombre d'arbres.

</details>
<details><summary>Indice 3</summary>

Réponds en deux temps : pour la forêt, la variance baisse vers un plancher et rien ne se dégrade (d'où vient alors l'overfitting d'une forêt, et comment le régler ?) ; pour le gradient boosting, chaque arbre corrige les résidus et finit par apprendre le bruit (quel outil fixe le nombre d'arbres ?).

</details>

### 14.E3 — XGBoost ou réseau de neurones sur un tableau de données ?

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ « le gradient boosting, référence des données tabulaires ».

</details>
<details><summary>Indice 2</summary>

Donne une raison empirique (les études comparatives), trois raisons techniques (biais inductifs, préparation des données, coût) et un cas où tu essaierais quand même autre chose.

</details>
<details><summary>Indice 3</summary>

Structure : ton choix ; la raison empirique (Grinsztajn et coll., 2022) ; ce que les arbres boostés font bien sur ces données (features inutiles, colonnes au sens propre, fonctions irrégulières, valeurs manquantes et catégorielles) ; le protocole (baseline, validation, early stopping, fuites) ; quand tu essaierais un réseau ou un modèle de fondation tabulaire.

</details>

### 14.E4 — Peut-on se fier à feature_importances_ ?

<details><summary>Indice 1</summary>

Sur quelles données l'importance par l'impureté est-elle calculée, et quelles features offrent le plus de seuils ?

</details>
<details><summary>Indice 2</summary>

Relis le paragraphe sur l'importance des features au §14.5 et l'encadré 🕰️ qui le suit : deux biais, une alternative, un piège (les features corrélées).

</details>
<details><summary>Indice 3</summary>

Structure : ce que mesure `feature_importances_` ; ses deux biais (forte cardinalité, calcul sur l'entraînement) ; tes premiers soupçons devant une colonne absurde en tête (fuite, identifiant) ; l'importance par permutation sur la validation et son piège ; les valeurs de Shapley pour expliquer une prédiction.

</details>

### 14.E5 — Comment régler un gradient boosting ?

<details><summary>Indice 1</summary>

Le learning rate et le nombre d'arbres se règlent ensemble : lequel fixes-tu, et comment choisis-tu l'autre ?

</details>
<details><summary>Indice 2</summary>

Relis le paragraphe sur le learning rate et le nombre d'arbres : early stopping, complexité des arbres, hasard (sous-échantillonnage), régularisation.

</details>
<details><summary>Indice 3</summary>

Ordre possible : (1) un petit learning rate et l'early stopping pour le nombre d'arbres ; (2) la complexité de chaque arbre (profondeur ou nombre de feuilles, taille minimale des feuilles) ; (3) le sous-échantillonnage des lignes et des colonnes, la régularisation ; (4) la méthode de recherche et le rôle du test ; (5) la reproductibilité (graines, versions).

</details>

<a id="notebook"></a>

## Notebook

Les indices des exercices du notebook (14.11 à 14.28) arriveront avec eux, à la prochaine session de génération.
