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

### 14.R1 — Ch. 13 : l'impureté de Gini d'un nœud

<details><summary>Indice 1</summary>

$G = 1 - \sum_c p_c^2$, avec les proportions $p_c$ des classes dans le nœud (fiche du ch. 13, §13.5.2).

</details>
<details><summary>Indice 2</summary>

Pour c), le gain pondère chaque enfant par sa part des exemples. Pour d), avec des poids, $p_c$ est la somme des poids de la classe $c$ divisée par la somme de tous les poids.

</details>
<details><summary>Indice 3</summary>

a) Proportions 0,6, 0,3 et 0,1 : $G = 1 - (0{,}36 + 0{,}09 + 0{,}01)$. b) L'enfant droit a 3 exemples de la classe 1 et 1 de la classe 2 : applique la même formule. c) L'enfant gauche est pur ; pondère l'enfant droit par 4/10. d) Les poids des classes valent 6, 3 et 4 : calcule les proportions sur 13. e) Les proportions changent-elles si tous les poids doublent ?

</details>

### 14.R2 — Ch. 11 : induction et généralisation

<details><summary>Indice 1</summary>

Relis les deux façons de raisonner du ch. 11 : laquelle donne une conclusion seulement probable ?

</details>
<details><summary>Indice 2</summary>

Pour b), demande-toi ce qu'un bootstrap peut contenir qui ne soit pas déjà dans l'échantillon. Pour c), relis le tableau des sophismes inductifs du ch. 11 : lequel est un défaut de **collecte** ?

</details>
<details><summary>Indice 3</summary>

a) On tire une règle d'exemples observés pour l'appliquer à d'autres : de quel côté du tableau déduction/induction est-on ? b) Un bootstrap ne tire que dans l'échantillon : le biais peut-il disparaître ? c) Pour chaque sophisme proposé, demande-toi si l'erreur vient du raisonnement ou de la façon dont les données ont été réunies. d) Un ensemble reste-t-il entraîné sur des données passées ?

</details>

### 14.R3 — Ch. 7 : un-contre-tous et un-contre-un

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

### Ex 14.1 — Dépouiller un vote pondéré ✏️

<details><summary>Indice 1</summary>

Fais un tableau : une ligne par exemple, une colonne par classe ; dans chaque case, la somme des poids des votants qui ont prédit cette classe pour cet exemple.

</details>
<details><summary>Indice 2</summary>

N'oublie pas le signe du votant 3 : son poids $-1$ **retire** 1 à la classe qu'il désigne. En cas d'égalité, la plus petite classe gagne. Pour f), écris le score de la classe 1 de l'exemple 1 en fonction du poids $w_3$.

</details>
<details><summary>Indice 3</summary>

a) Exemple 0 : la classe 1 est prédite par les votants 1 (poids 1) et 2 (poids 1,5), d'où 2,5. Remplis de même les scores des classes 0 et 2 de chaque exemple et prends le maximum (b) à d)). e) Refais le tableau avec un poids 1 pour chacun. f) La classe 1 de l'exemple 1 reçoit $2 + 1 + w_3$, la classe 0 reçoit 2 : quelle condition stricte sur $w_3$ ? g) et h) Que deviennent les scores quand chaque poids est multiplié par 3, ou quand chaque **voix** gagne 10 ?

</details>

### Ex 14.2 — Additionner les voix de trois droites, région par région ✏️

<details><summary>Indice 1</summary>

Une région est définie par les apprenants qui la mettent de leur côté positif ; son score est la somme de leurs poids.

</details>
<details><summary>Indice 2</summary>

Pour situer un point, teste les trois conditions $x_1 \ge 1$, $x_2 \ge 1$ et $x_1 + x_2 \le 3$. Pour h), écris $\mathbb{1}[\text{côté positif}]$ en fonction de $s_m = \pm 1$ : $\mathbb{1} = (s_m + 1)/2$.

</details>
<details><summary>Indice 3</summary>

a) La région « C » ne reçoit que le poids de C : $-3$. Calcule de même les six autres régions (ABC, AB, AC, A, BC, B) et compte celles de score strictement positif pour d). b) et c) Situe chaque point par les trois tests, puis lis le score de sa région. e) Pour $P_2$, calcule $x_1 + x_2$ avant de conclure. f) Refais les sept scores avec $w_C = -1$. g) Compare les prédictions de A seul aux vraies classes. h) Remplace chaque indicatrice par $(s_m + 1)/2$ et sépare ce qui dépend de $s_m$ de ce qui n'en dépend pas.

</details>

### Ex 14.3 — Probabilité d'être oublié par un bootstrap : vers 36,8 % ∂

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

### Ex 14.4 — Ajuster les résidus, c'est descendre le gradient de la MSE ∂

<details><summary>Indice 1</summary>

Traite chaque $F_i$ comme une variable ordinaire : seul le terme $i$ de la somme en dépend.

</details>
<details><summary>Indice 2</summary>

Pour 3 et 4, annule la dérivée par rapport à la constante. Pour 5, la dérivée de $\lvert t \rvert$ vaut $\operatorname{signe}(t)$ pour $t \ne 0$ ; pour la constante, regarde ce que devient $\sum_i \lvert y_i - c \rvert$ quand $c$ passe un $y_i$. Pour 7, réutilise $d\ell/dz = \hat p - y$ de ∂ 13.8.

</details>
<details><summary>Indice 3</summary>

1. $\frac{\partial}{\partial F_i} \frac12 (y_i - F_i)^2 = -(y_i - F_i)$ par la règle de la chaîne. 2. Écris $F_i \leftarrow F_i - \eta\, \partial L/\partial F_i$, puis demande-toi quelle valeur ce pas donne à un exemple qui n'est pas dans la somme. 3. $\frac{d}{dc}\sum_i (y_i - c)^2 = -2\sum_i (y_i - c)$. 4. Pose $r_i = y_i - F_i$ et reconnais le problème 3. 5. Quand $c$ augmente d'un petit $\delta$, chaque point au-dessus de $c$ rapproche de $\delta$, chaque point en dessous éloigne de $\delta$ : quand l'équilibre est-il atteint ? 6. Une feuille par exemple prédit exactement son résidu : que vaut $F_1$ ? 7. Écris $\ell$ comme une fonction de $F$ par $\hat p = \sigma(F)$, et réutilise la dérivée de ∂ 13.8.

</details>

### Ex 14.5 — Quand la majorité de votants indépendants se trompe ✏️

<details><summary>Indice 1</summary>

La majorité a raison quand plus de la moitié des votants ont raison : additionne les probabilités de « $j$ votants justes » pour tous les $j > M/2$.

</details>
<details><summary>Indice 2</summary>

Chaque terme vaut $\binom{M}{j} q^j (1 - q)^{M - j}$ (encadré sur la loi binomiale). Pour f), demande-toi si les trois votes peuvent être différents. Pour g), calcule la probabilité pour $M = 3, 5, 7, \dots$ jusqu'à dépasser 0,9 (`scipy.stats.binom.sf(M // 2, M, q)` le fait pour toi).

</details>
<details><summary>Indice 3</summary>

a) $M = 3$ : il faut 2 ou 3 justes : $0{,}7^3 + 3 \times 0{,}7^2 \times 0{,}3$. Pour b) à e), h) et i), applique la même somme avec $j$ de $\lceil (M + 1)/2 \rceil$ à $M$ (un seul terme pour e)). f) Trois votes identiques : la majorité vaut quel votant ? g) Calcule la somme pour les $M$ impairs successifs et arrête-toi au premier qui atteint 0,9.

</details>

### Ex 14.6 — Une itération d'AdaBoost à la main ✏️

<details><summary>Indice 1</summary>

À chaque tour : l'erreur pondérée est la somme des poids des exemples mal classés ; $\alpha = \ln\frac{1 - \varepsilon}{\varepsilon}$ à deux classes ; les mal classés sont multipliés par $e^{\alpha}$, puis on normalise.

</details>
<details><summary>Indice 2</summary>

Au tour 1, tous les poids sont égaux : la meilleure souche est celle qui fait le moins d'erreurs. Au tour 2, compare le poids de l'exemple mal classé au tour 1 à celui des autres avant de chercher la meilleure souche. Garde des fractions : $e^{\ln a} = a$.

</details>
<details><summary>Indice 3</summary>

a) La souche « $x \le 4{,}5$ → 0, sinon 1 » ne se trompe que sur $x = 2$ : $\varepsilon_1 = 1/8$. b) $\ln\frac{7/8}{1/8}$. c) et d) Multiplie le poids de $x = 2$ par $e^{\alpha_1}$, additionne les huit poids, divise. e) et f) Parmi les souches qui classent bien $x = 2$, cherche celle dont les erreurs pèsent le moins avec les poids de c) et d). h) Là où les deux souches se contredisent, compare $\alpha_1$ et $\alpha_2$. i) à k) Mets d'abord à jour les poids du tour 2 (les erreurs de la souche 2, multipliées par $e^{\alpha_2}$, puis normalisation), puis suis le même schéma. l) Ajoute $\ln(K - 1)$. m) Compare 0,55 à $1 - 1/K$.

</details>

### Ex 14.7 — Deux itérations de gradient boosting sur cinq points ✏️

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

### Ex 14.8 — Bagging et boosting expliqués en 5 lignes 🗣️

<details><summary>Indice 1</summary>

Cherche deux images : un groupe d'experts qui donnent leur avis **en même temps**, et une équipe qui travaille **à la chaîne**.

</details>
<details><summary>Indice 2</summary>

Pour chaque méthode, une phrase sur ce qui change d'un membre à l'autre (les données ? la cible ?), une sur ce qu'elle réduit (biais ou variance), une sur le réglage qu'elle demande.

</details>
<details><summary>Indice 3</summary>

Plan possible en cinq lignes : (1) l'image de la forêt, et ce qu'elle moyenne ; (2) l'image du boosting, et ce que chaque membre corrige ; (3) ce qui varie d'un membre à l'autre dans chacun ; (4) variance contre biais ; (5) quand choisir l'un ou l'autre, avec le risque d'overfitting quand on ajoute des membres.

</details>

### Ex 14.9 — Un jury de modèles biaisés reste biaisé ⚖️

<details><summary>Indice 1</summary>

Relis le paragraphe « Un ensemble ne corrige pas un biais commun » du §14.2 et l'encadré sur la variance d'une moyenne de modèles corrélés.

</details>
<details><summary>Indice 2</summary>

Pour 2, pense aux informations d'un CV qui sont corrélées au genre sans le nommer. Pour 4, distingue une mesure qui ne demande que les décisions (un taux de sélection par groupe) d'une mesure qui demande une « vérité » sur la qualification.

</details>
<details><summary>Indice 3</summary>

1. Le biais est-il de la variance (ce qui diffère d'un modèle à l'autre) ou une erreur commune ? Que fait un vote d'une erreur commune ? 2. Activités, filières, interruptions de carrière, temps partiel… 3. D'autres décideurs, des évaluations à l'aveugle, une autre cible ; quel biais pourraient-elles partager ? 4. Taux de sélection, erreurs à qualification comparable, calibration, par groupe ; il faut l'attribut sensible pour l'audit, protégé. 5. Le système trie-t-il seul ? Relis l'encadré 🕰️ du §13.7 (RGPD, article 22 ; AI Act, systèmes à haut risque).

</details>

### Ex 14.10 — Pourquoi les arbres battent encore le deep learning sur le tabulaire 📄

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

### Ex 14.11 — Trois classifieurs du ch. 13 votent : mieux que le meilleur ? 🔮

<details><summary>Indice 1</summary>

Reprends les deux conditions du §14.2 : chaque membre doit faire mieux que le hasard (ici, tous font environ 80 %, bien au-dessus du tiers que donnerait un tirage au hasard), et leurs **erreurs** doivent être peu corrélées. La question est donc : trois modèles différents, qui voient les **mêmes** deux mesures, se trompent-ils sur des manchots différents ?

</details>
<details><summary>Indice 2</summary>

Pense aux manchots plutôt qu'aux modèles. Sur l'épaisseur du bec et la longueur de la nageoire, quelles espèces se ressemblent (rappelle-toi les nuages de points des ch. 12 et 13, ou trace-les avant l'expérience) ? Que peut faire un modèle, quelle que soit la forme de sa frontière, pour un manchot qui se trouve au milieu des mesures d'une autre espèce ? Pour a), un vote à la pluralité ne gagne un manchot que si le meilleur membre le rate et que les deux autres le réussissent ensemble.

</details>
<details><summary>Indice 3</summary>

a) Pose le bilan : (accuracy du vote) − (accuracy du meilleur) = (nombre de manchots que le meilleur rate et que les deux autres réussissent − nombre de manchots qu'il réussit et que les deux autres ratent ensemble) / 333 ; estime ces deux nombres à partir de ta réponse à b), puis convertis en points. b) Le critère : quelle part des erreurs du meilleur vient de la zone où deux espèces ont les mêmes mesures, que les trois modèles partagent ? c) Compare les mesures moyennes des trois espèces sur ces deux variables : laquelle se détache des deux autres, et que reste-t-il alors à confondre ? N'exécute l'expérience qu'après avoir écrit tes trois prédictions.

</details>

### Ex 14.12 — Coder le vote pondéré et le tirage bootstrap 🔨

<details><summary>Indice 1</summary>

Pour le vote, raisonne classe par classe : pour une classe $k$, le tableau booléen `predictions == k`, de forme `(n_voters, n_samples)`, dit quels votants la désignent ; pondère ses lignes par les poids et additionne sur les votants : tu obtiens le total de la classe $k$ pour chaque exemple. La classe gagnante est celle du plus grand total. Pour le bootstrap, le générateur de NumPy sait tirer des entiers uniformes (`rng.integers`).

</details>
<details><summary>Indice 2</summary>

Vote : convertis avec `np.asarray`, vérifie `ndim == 2`, fabrique les poids (`np.ones(n_voters)` si `None`, sinon un tableau `float` de forme `(n_voters,)`, sans `NaN`), puis `labels = np.unique(predictions)` (triées). Le broadcasting fait toutes les classes d'un coup : `predictions[None, :, :] == labels[:, None, None]` a la forme `(K, n_voters, n_samples)` ; multiplie par les poids placés sur l'axe des votants, additionne sur cet axe, et `np.argmax(..., axis=0)` donne, pour chaque exemple, l'indice de la classe gagnante (le premier maximum, donc la plus petite classe à égalité). Bootstrap : vérifie les tailles, remplace `n_draws=None` par `n_samples`, et convertis `rng` avec `np.random.default_rng(rng)` (qui accepte `None`, un entier ou un générateur, et rend le générateur tel quel).

</details>
<details><summary>Indice 3</summary>

```python
def plurality_vote(predictions, weights=None):
    predictions = np.asarray(predictions)
    ...                                            # 2-D? weights: one per voter, no NaN -> w of shape (n_voters,)
    labels = np.unique(predictions)                # sorted: argmax keeps the smallest label on ties
    totals = ((predictions[None, :, :] == labels[:, None, None]) * w[None, :, None]).sum(axis=1)   # (K, n_samples)
    return ...                                     # the label of the largest total, for every sample


def bootstrap_indices(n_samples, n_draws=None, rng=None):
    n_draws = ...                                  # n_samples when None
    ...                                            # ValueError for n_samples < 1 or n_draws < 1
    rng = np.random.default_rng(rng)
    return ...                                     # n_draws integers drawn with rng.integers (upper bound excluded)
```

</details>

### Ex 14.13 — Jury de votants : indépendants puis corrélés 🔬

<details><summary>Indice 1</summary>

a) Le nombre de votants justes suit une loi binomiale $\mathcal{B}(M, q)$ ; la majorité a raison quand ce nombre dépasse $M/2$, soit au moins $(M + 1)/2$ pour $M$ impair. `stats.binom.sf(k, M, q)` rend $P(X > k)$. c) et d) Une ligne du tableau de `jury_13` est un vote ; la majorité a raison quand plus de la moitié de ses `True` sont là. f) Sépare les deux cas du simulateur : les votes où tout le monde recopie, et les autres.

</details>
<details><summary>Indice 2</summary>

c) et d) Écris une petite fonction qui reçoit le tableau booléen et rend `np.mean(right.sum(axis=1) > right.shape[1] / 2)`. e) Les deux colonnes sont des tableaux de booléens : `np.corrcoef` les accepte et rend une matrice $2 \times 2$, dont tu prends le terme hors diagonale. f) Par la formule des probabilités totales : $P(\text{majorité juste}) = c \times P(\text{juste} \mid \text{tous recopient}) + (1 - c) \times P(\text{juste} \mid \text{chacun seul})$. Que vaut chacun des deux termes conditionnels quand $M$ devient très grand ? g) Écris cette limite comme une fonction de $q$, puis résous l'équation.

</details>
<details><summary>Indice 3</summary>

a) `exact_13 = [float(stats.binom.sf(M // 2, M, Q_13)) for M in MS_13]` : pour $M$ impair, $P(X > \lfloor M/2 \rfloor) = P(X \ge (M + 1)/2)$. Pour les autres : b) parcours `MS_13` et `exact_13` ensemble (`zip`) et garde le premier $M$ dont la valeur atteint 0,9 ; c) et d) applique ta fonction de l'indice 2 à `jury_13(M, Q_13, c)` pour chaque $M$ ; e) `np.corrcoef(right[:, 0], right[:, 1])[0, 1]` sur `right = jury_13(11, Q_13, 0.3)` ; f) quand tous recopient, la majorité suit l'avis commun, juste avec la probabilité $q$ ; quand chacun décide seul, le théorème du jury (fiche §14.2) dit vers quoi tend la probabilité pour $q > 0{,}5$ : pose $c \times (\dots) + (1 - c) \times (\dots)$ et calcule ; g) résous $0{,}3\,q + 0{,}7 \times (\dots) = 0{,}9$.

</details>

### Ex 14.14 — Voting et stacking avec les classifieurs du ch. 13 📦

<details><summary>Indice 1</summary>

`VotingClassifier` et `StackingClassifier` reçoivent la liste des paires `(nom, modèle)` telle quelle : `MEMBERS_14` est déjà au bon format. Un ensemble se mesure comme n'importe quel modèle de scikit-learn : `cross_val_score(modèle, X_14, Y_14, cv=CV_14)`, dont tu prends la moyenne.

</details>
<details><summary>Indice 2</summary>

Le vote dur et le vote souple ne diffèrent que par `voting="hard"` ou `voting="soft"`. Pour le stacking, donne le méta-modèle par `final_estimator=LogisticRegression(max_iter=1000)` : le reste (la validation croisée interne à 5 folds, les `predict_proba` des membres comme features du méta-modèle) est le comportement par défaut. Le modèle seul est un pipeline : la standardisation puis la régression logistique, sur les 4 colonnes.

</details>
<details><summary>Indice 3</summary>

a) `hard_14 = cross_val_score(VotingClassifier(MEMBERS_14, voting="hard"), X_14, Y_14, cv=CV_14).mean()`. Pour les autres, le même appel avec : b) `voting="soft"` ; c) `StackingClassifier(MEMBERS_14, final_estimator=...)` ; d) `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`, sans `MEMBERS_14`.

</details>

### Ex 14.15 — Forêts scikit-learn sur California : OOB et parallélisme 📦

<details><summary>Indice 1</summary>

Avec `oob_score=True`, la forêt calcule pendant `fit` le score de chaque district par les seuls arbres qui ne l'ont pas vu, et le range dans `oob_score_` ; en régression, ce score est un $R^2$, comme celui que rend `score`. Pour c) et d), il faut entraîner une forêt par nombre d'arbres.

</details>
<details><summary>Indice 2</summary>

Une boucle sur `NS_15` qui entraîne `RandomForestRegressor(n_estimators=n, oob_score=True, random_state=0, n_jobs=-1)`, et range **deux** listes : les `oob_score_` (c) et les $R^2$ de test (pour d). Puis d) : parcours les trois listes ensemble (`zip`) et garde le premier $n$ dont `abs(oob - test) < 0.01`. Pour masquer l'avertissement des petites forêts : `with warnings.catch_warnings(): warnings.simplefilter("ignore", UserWarning)` autour de la boucle.

</details>
<details><summary>Indice 3</summary>

a) `oob_15 = forest_15.oob_score_`, après `forest_15 = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0, n_jobs=-1).fit(X_TRAIN_15, Y_TRAIN_15)`. Pour les autres : b) `forest_15.score` sur le test ; c) la liste de la boucle de l'indice 2 ; d) le critère `abs(oob - test) < 0.01`, en parcourant `NS_15` dans l'ordre.

</details>

### Ex 14.16 — Combien d'arbres ? Bagging, forêt et ExtraTrees face au bruit 🔬

<details><summary>Indice 1</summary>

`curves_16` est une double boucle : sur les ensembles de `ENSEMBLES_16` (`name, make`), puis sur les nombres d'arbres de `NS_16` ; `make(n)` construit l'ensemble de `n` arbres, que tu entraînes et notes sur les données de `moons_16(n_noise)` (appelle-la **une** fois, avant les boucles).

</details>
<details><summary>Indice 2</summary>

Une compréhension de dictionnaire suffit : `{name: [... for n in NS_16] for name, make in ENSEMBLES_16.items()}`. Ensuite, l'indice d'un nombre d'arbres dans les listes est `NS_16.index(400)` (ou `-1` pour le dernier). b) Compare la dernière valeur de chaque courbe de `noisy_16`. d) Pour chaque courbe des deux dictionnaires, calcule (valeur à 100 arbres) − (valeur à 400 arbres), et cherche si l'une dépasse 0,005 (`any`).

</details>
<details><summary>Indice 3</summary>

```python
def curves_16(n_noise):
    X_train, y_train, X_test, y_test = moons_16(n_noise)
    return {name: [...   # make(n), fitted on the training points, then its test accuracy
                   for n in NS_16]
            for name, make in ENSEMBLES_16.items()}
```
Pour les autres : b) l'ensemble dont `noisy_16[name][-1]` est le plus petit, traduit en lettre ; c) `noisy_16["extra"][NS_16.index(400)] - noisy_16["extra"][NS_16.index(25)]` ; d) le critère de l'indice 2 sur les six courbes.

</details>

### Ex 14.17 — Coder BaggingClassifier avec score out-of-bag 🔨

<details><summary>Indice 1</summary>

Le pseudo-code de la fiche (§14.4) est la méthode : un générateur créé au début de `fit`, puis, pour chaque membre, ses indices, un clone du gabarit, sa graine, son entraînement sur les lignes tirées. `predict_proba` moyenne les probabilités des membres, `predict` prend la classe de plus grande moyenne. Pour le score out-of-bag, chaque membre ne prédit que les exemples qu'il n'a pas vus.

</details>
<details><summary>Indice 2</summary>

Dans `fit` : vérifie les hyperparamètres et les données (un `X` 2-D sans `NaN`, `len(X) == len(y)`), calcule le nombre de tirages à partir de `max_samples`, `self.classes_ = np.unique(y)`, puis la boucle. Pour aligner un membre sur `classes_` : `proba = np.zeros((len(X), K))`, puis `proba[:, np.searchsorted(self.classes_, member.classes_)] = member.predict_proba(X)`. Pour le score out-of-bag : deux tableaux, `total` de forme `(n, K)` et `counts` de forme `(n,)` ; pour chaque membre, le masque `out = np.ones(n, dtype=bool)` puis `out[indices] = False` ; ajoute ses probabilités alignées sur `X[out]` à `total[out]` et 1 à `counts[out]` ; garde enfin les exemples dont `counts > 0`.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y):
    ...                                                   # checks, n, n_draws from max_samples
    rng = np.random.default_rng(self.random_state)
    template = DecisionTreeClassifier() if self.estimator is None else self.estimator
    self.classes_ = np.unique(y)
    self.estimators_, self.estimators_samples_ = [], []
    for _ in range(self.n_estimators):
        indices = bootstrap_indices(n, n_draws, rng) if self.bootstrap else rng.choice(n, size=n_draws, replace=False)
        member = clone(template)
        if hasattr(member, "random_state"):
            member.random_state = int(rng.integers(np.iinfo(np.int32).max))
        ...                                               # fit the member on X[indices], y[indices]; keep it
    ...                                                   # oob_score_ if asked (indice 2)
    return self
```

</details>

### Ex 14.18 — Coder RandomForest et ExtraTrees 🔨

<details><summary>Indice 1</summary>

Une forêt est un bagging d'arbres configurés : construis le gabarit `DecisionTreeClassifier(...)` avec les hyperparamètres de la forêt, confie-le à ton `BaggingClassifier`, et délègue-lui les prédictions. ExtraTrees ne diffère que par le `splitter` du gabarit et le `bootstrap` par défaut. Reste l'importance des features, à calculer à partir des arbres.

</details>
<details><summary>Indice 2</summary>

`self._bagging = BaggingClassifier(template, n_estimators=self.n_estimators, max_samples=1.0, bootstrap=self.bootstrap, oob_score=..., random_state=self.random_state)`, entraîné sur `X`, `y` ; puis `estimators_` et `classes_` viennent du bagging, et `predict_proba` lui est confié (avec une `RuntimeError` si `self._bagging` n'existe pas encore). Pour les importances : empile les `feature_importances_` des arbres, garde les lignes dont la somme est positive (les arbres qui coupent), fais leur moyenne, puis divise par la somme de cette moyenne ; s'il ne reste aucun arbre, rends des zéros de longueur $p$.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y):
    template = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
    self._bagging = BaggingClassifier(template, ...).fit(X, y)     # max_samples=1.0, bootstrap, oob_score, random_state
    self.estimators_, self.classes_ = self._bagging.estimators_, self._bagging.classes_
    importances = np.array([tree.feature_importances_ for tree in self.estimators_])
    splitting = importances[importances.sum(axis=1) > 0]
    ...                                                            # the mean of `splitting`, renormalised (or zeros)
    return self
```

</details>

### Ex 14.19 — Importance des features : impureté contre permutation 📦

<details><summary>Indice 1</summary>

`forest_19.feature_importances_` donne l'importance MDI, dans l'ordre des colonnes ; `permutation_importance` rend un objet dont l'attribut `importances_mean` donne l'importance par permutation, dans le même ordre. Une `pandas.Series` indexée par `FEATURES_19` et triée (`sort_values(ascending=False)`) rend les rangs et les noms faciles à lire.

</details>
<details><summary>Indice 2</summary>

Le rang d'une feature dans une Series triée : `list(mdi_19.index).index("row_id") + 1` ; une importance par son nom : `mdi_19["row_id"]` ; la deuxième feature : `mdi_19.index[1]`. e) Le même appel à `permutation_importance` que pour `perm_19`, sur `X_TRAIN_19`, `Y_TRAIN_19`. f) Une boucle sur `r` de 0 à 4 : une copie de `X_VAL_19`, dont les deux colonnes reçoivent les lignes permutées (`X_VAL_19[["Latitude", "Longitude"]].to_numpy()[order]`) ; la baisse est le $R^2$ de validation d'origine moins celui de la copie.

</details>
<details><summary>Indice 3</summary>

a) `rank_id_19 = list(mdi_19.index).index("row_id") + 1`. Pour les autres : b) le rapport de deux valeurs de `mdi_19` ; c) et d) l'élément d'indice 1 de chaque index ; e) `importances_mean[FEATURES_19.index("row_id")]` du calcul sur l'entraînement ; f) le squelette :
```python
base = forest_19.score(X_VAL_19, Y_VAL_19)
drops = []
for r in range(5):
    order = np.random.default_rng(r).permutation(len(X_VAL_19))
    shuffled = X_VAL_19.copy()
    ...                     # both columns of `shuffled` get the rows of X_VAL_19 in the order `order`
    drops.append(base - forest_19.score(shuffled, Y_VAL_19))
```

</details>

### Ex 14.20 — Lire les courbes train/validation d'un boosting 📈

<details><summary>Indice 1</summary>

Trois diagnostics du ch. 9, appliqués au nombre d'arbres : en **overfitting**, l'erreur d'entraînement devient très petite pendant que l'erreur de validation reste nettement au-dessus ; en **underfitting**, les deux erreurs restent proches et assez hautes ; entre les deux, la courbe de validation passe par un minimum, puis remonte. Plus les arbres sont profonds, plus chacun corrige à lui seul.

</details>
<details><summary>Indice 2</summary>

Dans chaque panneau, regarde où finissent les deux courbes à 400 arbres et l'écart entre elles, puis la forme de la courbe de validation : un minimum net, une courbe encore descendante, une courbe plate. Pour d), chaque élément de `CURVES_20` est le couple `(entraînement, validation)` ; `np.argmin` rend une **position**, qui commence à 0. Pour e), compare la dernière valeur de la courbe de validation à sa plus petite.

</details>
<details><summary>Indice 3</summary>

a) `"B"` : sa RMSE d'entraînement tombe presque à 0 (0,02 après une centaine d'arbres : des arbres très profonds isolent chaque district) alors que sa validation reste vers 0,575. Pour les autres : b) cherche le panneau dont les deux courbes descendent encore ensemble à 400 arbres ; c) lis l'abscisse du point le plus bas de la validation du panneau restant (ni celui de a), ni celui de b)) ; d) `[int(np.argmin(CURVES_20[panel][1])) + 1 for panel in "ABC"]` ; e) `validation[-1] - validation.min()` pour le panneau de c).

</details>

### Ex 14.21 — Learning rate et nombre d'arbres : prédire les courbes 🔮

<details><summary>Indice 1</summary>

Chaque arbre ajoute $\eta \times$ (sa prédiction des résidus). Pour corriger une même quantité d'erreur, des pas $\eta$ fois plus petits demandent à peu près combien de fois plus d'arbres ? Pour b), pense à ce que fait un grand learning rate : chaque arbre colle fortement aux résidus de l'entraînement, bruit compris.

</details>
<details><summary>Indice 2</summary>

Si le nombre d'arbres au minimum variait comme $1/\eta$, le produit $\eta \times (\text{arbres au minimum})$ resterait à peu près constant ; s'il variait comme $1/\eta^2$, c'est $\eta^2 \times (\text{arbres})$ qui le resterait. 📈 14.20 te donne un ordre de grandeur de ce produit pour des arbres de profondeur 3 : son learning rate et le nombre d'arbres au minimum du panneau correspondant. Pour b), compare l'ordre de grandeur des RMSE de validation de 14.20 (vers 0,55) à la taille des écarts proposés. Pour c), estime avec ce produit combien d'arbres 0,1 et 0,02 demandent pour arriver à leur minimum.

</details>
<details><summary>Indice 3</summary>

a) Le critère : le nombre d'arbres utile varie-t-il plutôt comme $1/\eta$, comme $1/\eta^2$, ou à peine ? Choisis l'hypothèse que tu crois la plus juste, en pensant à une somme de petits pas, et déduis-en le facteur. b) Le critère : ce que le learning rate change, c'est la façon d'avancer, pas les données ni la forme des arbres ; un gain de plusieurs dixièmes de RMSE est-il plausible, ou quelques centièmes au plus ? c) Le critère : avec l'hypothèse de a), à quelle distance de son minimum 0,02 est-il après 1 000 arbres, comparé à 0,1 ? N'exécute l'expérience qu'après avoir écrit tes trois prédictions.

</details>

### Ex 14.22 — Frontières d'AdaBoost après 1, 3, 10 et 50 souches 🎨

<details><summary>Indice 1</summary>

`model.staged_predict(X)` est un générateur : il donne les prédictions après 1, puis 2, puis 3… membres. Il suffit de le parcourir en comptant, et de garder les étapes de `rounds`. Pour la figure, une prédiction sur la grille `GRID_22` se remet à la forme `GRID1_22.shape` pour `ax.contourf`.

</details>
<details><summary>Indice 2</summary>

`for m, predicted in enumerate(model.staged_predict(X), start=1)` compte les membres à partir de 1 ; garde `predicted` quand `m` est dans `rounds`. Dans `draw_22` : `fig, axes = plt.subplots(1, len(rounds), ...)`, puis, pour chaque panneau, `ax.contourf(GRID1_22, GRID2_22, grille.reshape(GRID1_22.shape), alpha=0.3)`, les points avec `ax.scatter` (une couleur par classe), et un titre qui donne $m$ et l'accuracy d'entraînement après $m$ membres (les prédictions de `stage_predictions_22` sur `X`, comparées à `y`).

</details>
<details><summary>Indice 3</summary>

```python
def stage_predictions_22(model, X, rounds):
    return {m: ... for m, predicted in enumerate(model.staged_predict(X), start=1) if ...}   # keep the m of rounds


def draw_22(model, X, y, rounds):
    on_grid = stage_predictions_22(model, GRID_22, rounds)
    on_train = stage_predictions_22(model, X, rounds)
    fig, axes = plt.subplots(1, len(rounds), figsize=(4.5 * len(rounds), 3.8))
    for ax, m in zip(axes, rounds):
        ...        # contourf of on_grid[m] reshaped like GRID1_22, the points, a title with np.mean(on_train[m] == y)
    return fig
```

</details>

### Ex 14.23 — L'AdaBoost qui n'apprend rien 🐛

<details><summary>Indice 1</summary>

Compare `adaboost_23` au pseudo-code de la fiche (§14.7), ligne par ligne : quels exemples la mise à jour alourdit-elle ? Que fait le pseudo-code au **début** de chaque tour ? Avec quoi entraîne-t-il le membre ? Les trois erreurs sont trois écarts à ces trois lignes.

</details>
<details><summary>Indice 2</summary>

Corrige dans l'ordre des points de la vérification, et relance-la après chaque correction. Point 1 : après la mise à jour, qui doit peser plus lourd, les exemples que la souche a mal classés ou les autres ? (Si les poids deviennent NaN, c'est qu'une erreur pondérée est tombée à 0 : la mise à jour pousse dans le mauvais sens.) Point 2 : des poids qui ne somment plus à 1 font de `err` autre chose qu'une proportion. Point 3 : des souches toutes identiques ne voient pas ce qui change d'un tour à l'autre.

</details>
<details><summary>Indice 3</summary>

Première erreur : la mise à jour multiplie par $e^\alpha$ les exemples **bien** classés ; il faut `w = w * np.exp(alpha * wrong)`. Pour les deux autres : au début de chaque tour, que doit valoir la somme de `w`, et quelle ligne du pseudo-code l'assure ? Et quel argument de `stump.fit` lui transmet les poids du tour ?

</details>

### Ex 14.24 — Coder AdaBoost (SAMME) avec des souches pondérées 🔨

<details><summary>Indice 1</summary>

Le pseudo-code de la fiche (§14.7) est la méthode, avec $K$ classes : normaliser les poids, entraîner un clone avec `sample_weight`, calculer l'erreur pondérée, traiter les deux cas d'arrêt, puis le poids $\alpha$ et la mise à jour. La vérification de la signature se fait avec `inspect.signature(template.fit).parameters`, avant la boucle.

</details>
<details><summary>Indice 2</summary>

Range les membres, leurs $\alpha$ et leurs erreurs dans trois listes, converties en tableaux NumPy à la fin. Pour `decision_function`, un tableau `votes` de forme `(n, K)` : pour chaque membre, `votes[np.arange(n), np.searchsorted(self.classes_, member.predict(X))] += alpha`, puis divise par la somme des $\alpha$. `staged_predict` fait la même boucle et `yield`, après chaque membre, la classe de plus grand vote (`self.classes_[np.argmax(votes, axis=1)]`) ; `predict` prend l'argmax de `decision_function`.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y):
    ...                                                     # checks; signature of template.fit FIRST
    rng = np.random.default_rng(self.random_state)
    w = np.full(len(X), 1.0 / len(X))
    for _ in range(self.n_estimators):
        w = w / w.sum()
        member = clone(template)
        ...                                                 # its random_state, then member.fit(X, y, sample_weight=w)
        wrong = member.predict(X) != y
        err = float(np.sum(w[wrong]))
        ...                                                 # err <= 0: keep (alpha 1, error 0) and stop; err >= 1 - 1/K: stop
        alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(K - 1))
        ...                                                 # keep member, alpha, err; w[wrong] *= exp(alpha)
    return self
```

</details>

### Ex 14.25 — Coder un gradient boosting de régression 🔨

<details><summary>Indice 1</summary>

Pour l'arbre de régression, reprends ce que tu as fait au 13.25 pour le classifieur : des poids dans la valeur (la moyenne pondérée) et dans l'impureté (la variance pondérée), un générateur créé dans `fit` pour `max_features` et `splitter="random"`, et la somme des baisses d'impureté pondérées pour les importances. Pour le boosting, le pseudo-code de la fiche (section sur le gradient boosting) est la méthode : un tableau $F$ des prédictions courantes, mis à jour après chaque arbre.

</details>
<details><summary>Indice 2</summary>

Variance pondérée d'un nœud : $\sum_i w_i (y_i - \bar y_w)^2 / \sum_i w_i$, avec $\bar y_w$ la moyenne pondérée. Dans le `fit` du boosting : `self.init_ = float(np.mean(y))`, `F = np.full(n, self.init_)`, le gabarit `DecisionTreeRegressor(max_depth=..., min_samples_leaf=...)`, puis, à chaque étape, les lignes (toutes, ou `rng.choice(n, size=max(1, int(self.subsample * n)), replace=False)`), le clone ajusté sur `X[rows]` et `(y - F)[rows]`, la mise à jour de $F$ sur **tous** les exemples, et la MSE sur `rows` dans `train_score_`. `staged_predict` part de `init_` et ajoute $\eta \times$ `tree.predict(X)` arbre après arbre, avec `yield` (rends une copie de $F$).

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y):
    ...                                                     # checks
    rng = np.random.default_rng(self.random_state)
    template = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
    self.init_ = float(np.mean(y))
    F = np.full(n, self.init_)
    self.estimators_, scores = [], []
    for _ in range(self.n_estimators):
        rows = ...                                          # all the rows, or a subsample without replacement
        tree = clone(template)
        tree.fit(X[rows], (y - F)[rows])
        F = F + self.learning_rate * tree.predict(X)
        ...                                                 # keep the tree; the MSE on `rows`
    ...                                                     # self.train_score_ as an array
    return self
```

</details>

### Ex 14.26 — HistGradientBoosting, XGBoost et LightGBM sur California 📦

<details><summary>Indice 1</summary>

L'early stopping de `best_iteration_26` parcourt la courbe en retenant la meilleure valeur et sa position ; il s'arrête quand l'écart entre la position courante et celle de la meilleure atteint `patience`. Les deux bibliothèques font la même chose elles-mêmes avec `eval_set` ; leurs attributs `best_iteration` (XGBoost, compté à partir de 0) et `best_iteration_` (LightGBM, à partir de 1) disent où elles se sont arrêtées.

</details>
<details><summary>Indice 2</summary>

`best_iteration_26` : parcours la courbe en gardant la meilleure valeur vue jusqu'ici et sa position ; une valeur strictement plus petite devient la nouvelle meilleure ; sinon, regarde combien d'arbres se sont écoulés depuis la meilleure, et sors de la boucle quand ce nombre atteint `patience` ; rends la position de la meilleure, comptée à partir de 1. b) Parcours `hgb_26.staged_predict(X_CAL_TEST)` en comptant à partir de 1, et calcule la RMSE quand le compte vaut `hgb_trees_26`. Les durées : `time.perf_counter()` avant et après chaque `fit`.

</details>
<details><summary>Indice 3</summary>

a) `hgb_trees_26 = best_iteration_26(hgb_curve_26)`, après `hgb_curve_26 = [rmse(Y_VAL_26, p) for p in hgb_26.staged_predict(X_VAL_26)]`. Pour les autres : b) le parcours de l'indice 2 sur le test ; c) `xgb_26.best_iteration`, corrigé pour compter à partir de 1 ; d) et f) `rmse(Y_CAL_TEST, modèle.predict(X_CAL_TEST))` ; e) `lgbm_26.best_iteration_`, tel quel.

</details>

### Ex 14.27 — Un benchmark reproductible : graines, versions et tableau de résultats 🛠️

<details><summary>Indice 1</summary>

Les versions sont dans `platform.python_version()` et dans l'attribut `__version__` de chaque module (`np`, `sklearn`, `xgboost`, `lightgbm`) ; le nombre de cœurs dans `os.cpu_count()`. Le benchmark est une double boucle (modèles, puis graines) qui remplit une liste de dictionnaires, transformée à la fin en `DataFrame`.

</details>
<details><summary>Indice 2</summary>

Dans la boucle : `model = make(seed)` (un modèle neuf), `start = time.perf_counter()`, `model.fit(...)`, la durée, puis `rmse(y_test, model.predict(X_test))` ; un dictionnaire par ligne, `pd.DataFrame(rows, columns=[...])`. Le résumé : `results.groupby("model")`, puis la moyenne et l'écart-type (`std(ddof=1)`) de `"rmse"`, la moyenne de `"fit_seconds"`, rangés dans un `DataFrame` trié par `sort_values("rmse_mean")`. Le Markdown : une liste de lignes, `"| model | ... |"`, `"|---|---|---|---|"`, puis une f-string par ligne de `summary.iterrows()`, jointes par `"\n"`.

</details>
<details><summary>Indice 3</summary>

```python
def benchmark_27(models, X_train, y_train, X_test, y_test, seeds):
    rows = []
    for name, make in models.items():
        for seed in seeds:
            model = make(seed)
            ...                                        # fit, timed with time.perf_counter(); the test RMSE
            rows.append({"model": name, "seed": seed, "rmse": ..., "fit_seconds": ...})
    return pd.DataFrame(rows, columns=["model", "seed", "rmse", "fit_seconds"])
```
Pour `summary_27` et `markdown_27`, suis l'indice 2 ; les formats `f"{x:.4f}"` et `f"{x:.2f}"` donnent les décimales demandées.

</details>

### Ex 14.28 — Défi : RMSE ≤ 0,45 sur California 🏆

<details><summary>Indice 1</summary>

Pars du HistGradientBoosting de 14.26 et compare tes idées par **validation croisée** sur les données d'entraînement (`cross_val_score(..., scoring="neg_root_mean_squared_error", cv=5)`), jamais sur un test. Le défi demande aussi d'être bon sur trois autres découpages : une idée qui ne gagne que sur un découpage est de la chance.

</details>
<details><summary>Indice 2</summary>

Deux leviers. Les réglages : un learning rate plus petit avec plus d'itérations et un early stopping patient (`max_iter`, `learning_rate`, `early_stopping=True`, `n_iter_no_change`), des arbres plus ou moins grands (`max_leaf_nodes`, `min_samples_leaf`), une régularisation (`l2_regularization`). Les features : un arbre ne coupe qu'**un** axe à la fois, alors que les prix de la Californie suivent des lignes obliques (la côte, les vallées) ; des combinaisons de la latitude et de la longitude, ou des rapports entre les features existantes, peuvent l'aider. Un `FunctionTransformer` les calcule dans un pipeline, pour que le modèle reçoive toujours les 8 features brutes.

</details>
<details><summary>Indice 3</summary>

```python
from sklearn.preprocessing import FunctionTransformer


def add_features_28(X):
    X = np.asarray(X, dtype=float)
    latitude, longitude = X[:, 6], X[:, 7]
    ...                                   # new columns: combinations of latitude and longitude, ratios...
    return np.column_stack([X, ...])


def build_model_28():
    return make_pipeline(FunctionTransformer(add_features_28),
                         HistGradientBoostingRegressor(...))   # settings chosen by cross-validation on the training data
```
Compare chaque essai à HistGradientBoosting par défaut, sur les mêmes 5 folds, et ne garde que ce qui gagne nettement.

</details>
