# 13 · Classifieurs — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 13.Q1 — Pourquoi essayer des classifieurs classiques d'abord ?

<details><summary>Indice 1</summary>

Relis le §13.1 de la fiche : à quoi sert un modèle simple et rapide quand on découvre un problème, et à quoi le compare-t-on ensuite ?

</details>
<details><summary>Indice 2</summary>

Pour b), pèse le coût de chaque option et ce qu'elle t'apprend sur 2 000 lignes. Pour c), demande-toi si quelque chose **garantit** qu'un modèle plus complexe fera mieux sur ces données-là.

</details>
<details><summary>Indice 3</summary>

a) Une baseline fixe le score à battre : si un réseau coûteux ne la dépasse pas nettement, son coût n'est pas justifié ; décide si l'affirmation le dit bien. b) Élimine les options qui coûtent beaucoup sans rien mesurer d'utile, ou qui évaluent mal. c) Cherche un cas où un réseau ne ferait pas mieux (peu de données, données tabulaires…). d) Que révèle, par exemple, un kNN qui bat nettement une régression logistique ?

</details>

### 13.Q2 — Paramétrique ou non paramétrique ?

<details><summary>Indice 1</summary>

Pour chaque modèle, une seule question : le nombre de nombres qu'il garde après l'entraînement est-il **fixé d'avance**, ou **grandit-il** avec le nombre d'exemples ?

</details>
<details><summary>Indice 2</summary>

Fais la liste de ce que chaque modèle garde : des poids, des moyennes et des variances, des exemples, des nœuds, des vecteurs de support. Pour le SVM à noyau, relis la formule de prédiction du §13.4 : de quoi est faite la somme ?

</details>
<details><summary>Indice 3</summary>

a) Élément 1 : une régression logistique garde $p + 1$ nombres, quel que soit $n$ : **P**. Applique la question de l'indice 1 aux cinq autres ; pour l'arbre sans limite, demande-toi ce que devient le nombre de feuilles quand on double les données bruitées. b) Lequel ne fait rien à l'entraînement, sauf mémoriser ? c) Relis la définition d'un modèle paramétrique. d) Relis la dernière phrase de la définition d'un classifieur non paramétrique (§13.2).

</details>

### 13.Q3 — kNN n'est pas k-means

<details><summary>Indice 1</summary>

L'un est supervisé, l'autre non (ch. 7 et fiche §13.3) : lequel a besoin des labels pour fonctionner ?

</details>
<details><summary>Indice 2</summary>

Pour b), relis le pseudo-code du kNN : à quoi sert $k$ dans le vote ? Pour d), un algorithme non supervisé connaît-il le nom des classes ?

</details>
<details><summary>Indice 3</summary>

a) Le kNN fait voter les **classes** de ses voisins : il lui faut les labels. Décide pour le k-means de la même façon. b) Que trie-t-on, puis que garde-t-on, dans le pseudo-code ? c) Relis ce que fait `fit` dans le pseudo-code. d) Un numéro de groupe produit par k-means a-t-il un sens connu d'avance ?

</details>

### 13.Q4 — Petit k, grand k

<details><summary>Indice 1</summary>

Avec $k = 1$, quel est le plus proche voisin d'un point d'entraînement, quand on le cherche dans l'entraînement lui-même ?

</details>
<details><summary>Indice 2</summary>

Pour b), compare $k$ au nombre de points du petit groupe : quand la majorité des voisins d'un point du groupe n'est plus dans le groupe, que prédit le vote ? Pour c), qui vote quand $k = n$ ?

</details>
<details><summary>Indice 3</summary>

a) Chaque point est à distance 0 de lui-même, et il n'y a pas de doublon : son plus proche voisin est donc lui-même, avec sa propre classe. Déduis l'accuracy. b) Un groupe de 6 points ne peut gagner un vote que si **plus** de la moitié des $k$ voisins en font partie (une égalité va à la classe 0) : pour quels $k$ est-ce impossible ? c) Quand tous les exemples votent, le résultat dépend-il encore du point ? d) Quelle valeur de $k$ maximise toujours l'accuracy d'entraînement ? e) Avec deux classes et un nombre impair de votes, une égalité est-elle possible ?

</details>

### 13.Q5 — Marge, vecteurs de support et C

<details><summary>Indice 1</summary>

Relis la définition des vecteurs de support (§13.4) : quels points fixent la solution ?

</details>
<details><summary>Indice 2</summary>

Pour b), relis la loss $\frac12 \lVert \mathbf{w} \rVert^2 + C \sum_i \xi_i$ : que devient le compromis quand chaque violation coûte plus cher ? Pour d), qu'est-ce qu'un noyau RBF calcule à partir des coordonnées ?

</details>
<details><summary>Indice 3</summary>

a) Un point strictement hors de la bande vérifie sa contrainte avec de la place en trop : le retirer ne change pas la solution (Vrai). b) Plus C est grand, plus il est rentable de réduire $\sum_i \xi_i$, quitte à payer en $\lVert \mathbf{w} \rVert$ : qu'arrive-t-il à $2/\lVert \mathbf{w} \rVert$ ? c) Relis la phrase sur les vecteurs de support d'une marge souple. d) Le noyau RBF dépend de $\lVert \mathbf{x} - \mathbf{z} \rVert$ : une feature en grands nombres dominerait-elle cette distance ?

</details>

### 13.Q6 — Ce que l'astuce du noyau évite de calculer

<details><summary>Indice 1</summary>

L'astuce du noyau remplace un calcul long par un calcul court : lequel par lequel (encadré 🧮 sur l'astuce du noyau, §13.4) ?

</details>
<details><summary>Indice 2</summary>

Pour b), calcule d'abord le produit scalaire, puis élève-le au carré. Pour c), quelle est la distance d'un point à lui-même ? Pour d), compare la portée des « bosses » du noyau pour un grand et un petit $\gamma$. Pour e), relis l'encadré ⚠️ sur l'astuce du noyau.

</details>
<details><summary>Indice 3</summary>

a) Le noyau donne $K(\mathbf{x}, \mathbf{z})$ sans construire $\varphi$ : relis les quatre options avec cette phrase. b) $\mathbf{x} \cdot \mathbf{z} = 1 \times 2 + 2 \times 2$ ; termine. c) Remplace $\lVert \mathbf{x} - \mathbf{x} \rVert$ par sa valeur dans la formule. d) Un grand $\gamma$ ne fait ressembler un point qu'à ses voisins immédiats : la frontière peut-elle alors entourer chaque exemple ? e) Qui choisit le noyau, C et $\gamma$ ?

</details>

### 13.Q7 — Racine, feuilles, profondeur : le vocabulaire des arbres

<details><summary>Indice 1</summary>

Dessine l'arbre : une racine, deux branches ; l'une s'arrête tout de suite, l'autre mène à une seconde question.

</details>
<details><summary>Indice 2</summary>

Une feuille n'a pas d'enfant ; un nœud interne en a. La profondeur compte les **arêtes** entre la racine et la feuille la plus basse. Pour d), chaque niveau double au plus le nombre de nœuds.

</details>
<details><summary>Indice 3</summary>

a) La feuille de la branche « non » et les deux feuilles sous la seconde question : 3. b) Combien d'arêtes faut-il descendre pour atteindre les feuilles les plus basses ? c) Compte les nœuds qui posent une question. Vérifie la règle « une feuille de plus que de nœuds internes ». d) Un niveau : 2 feuilles ; deux niveaux : 4 ; continue. e) Combien d'arêtes y a-t-il entre la racine et elle-même ?

</details>

### 13.Q8 — Pourquoi un arbre surapprend-il ?

<details><summary>Indice 1</summary>

Un arbre sans limite continue de couper jusqu'à quand ? Relis le pseudo-code du §13.5.

</details>
<details><summary>Indice 2</summary>

Pour b), une question « $x_j \le s$ ? » dépend-elle de l'unité de $x_j$, ou seulement de l'ordre des valeurs ? Pour c), relis l'expérience du livre sur deux tirages de 70 %. Pour d), dans quelle direction est orientée la frontière d'une question qui ne regarde qu'une feature ?

</details>
<details><summary>Indice 3</summary>

a) Il coupe jusqu'à des feuilles pures, donc chaque exemple d'entraînement est bien classé : accuracy 1. b) Multiplier une feature par 1 000 change-t-il les partitions possibles ? c) Une petite modification des données peut changer la première question : que devient alors le reste de l'arbre ? d) Dessine la frontière de « $x_1 \le 3$ ? » dans le plan, puis celle de « $x_2 \le 5$ ? ».

</details>

### 13.Q9 — Gini ou gain d'information ?

<details><summary>Indice 1</summary>

Applique les formules de l'encadré 🧮 du §13.5.2 : $G = 1 - \sum_c p_c^2$ et $H = -\sum_c p_c \log_2 p_c$.

</details>
<details><summary>Indice 2</summary>

Pour a) et b), $p_0 = p_1 = 0{,}5$. Pour c), une seule classe a $p_c = 1$. Pour d), relis la formule du gain : chaque enfant y est multiplié par une fraction.

</details>
<details><summary>Indice 3</summary>

a) $G = 1 - 0{,}5^2 - 0{,}5^2$ : termine. b) $\log_2 0{,}5 = -1$ : calcule $H$ de la même façon. c) Remplace dans la formule de $G$. d) Quelle fraction multiplie $I(\text{gauche})$ dans la formule du gain ? e) Relis le paragraphe « Gini ou entropie ? » et la figure des impuretés.

</details>

### 13.Q10 — Limiter la profondeur ou élaguer ?

<details><summary>Indice 1</summary>

Pour chaque réglage, demande-toi **quand** il agit : pendant que l'arbre pousse, ou après qu'il a fini de pousser ?

</details>
<details><summary>Indice 2</summary>

Relis les deux paragraphes du §13.5.3 : « arrêter plus tôt » et « faire pousser puis couper ». `ccp_alpha` vient de la formule $R_\alpha(T) = R(T) + \alpha\,\lvert T \rvert$.

</details>
<details><summary>Indice 3</summary>

a) Élément 1 : `max_depth` interdit de couper au-delà d'une profondeur, pendant la croissance : **P**. Pose la même question pour les quatre autres. b) Dans $R_\alpha(T)$, chaque feuille coûte $\alpha$ : quand $\alpha$ grandit, un arbre à beaucoup de feuilles devient-il plus ou moins intéressant ? c) Quelle règle du ch. 8 interdit d'utiliser le test pour choisir un hyperparamètre, et pourquoi l'accuracy d'entraînement est-elle un mauvais juge ?

</details>

### 13.Q11 — Ce qui est vraiment « naïf » dans Naïve Bayes

<details><summary>Indice 1</summary>

Relis l'encadré ⚠️ sur ce qui est « naïf » (§13.6) : quelle hypothèse permet d'écrire la vraisemblance comme un produit ?

</details>
<details><summary>Indice 2</summary>

Pour c), compte ce qu'il faut pour une loi normale (deux nombres), puis combien de lois il y a (une par classe et par feature). Pour d), relis le paragraphe « Pourquoi ça marche quand même ». Pour e), que vaut le logarithme de 0 ?

</details>
<details><summary>Indice 3</summary>

a) Le produit $\prod_j P(x_j \mid c)$ n'est valable que si les features sont indépendantes sachant la classe : **B**. b) D'où viennent $\mu_{cj}$ et $\sigma_{cj}^2$ ? c) Multiplie le nombre de lois par le nombre de nombres de chaque loi. d) Des features corrélées comptées plusieurs fois poussent-elles les probabilités vers le milieu ou vers les extrêmes ? e) Un seul facteur nul dans un produit : que vaut le produit ?

</details>

### 13.Q12 — Quel classifieur pour quelle situation ?

<details><summary>Indice 1</summary>

Pour chaque situation, repère la contrainte qui décide : l'explication, la vitesse sur des textes, l'ajout d'exemples sans réentraînement, des probabilités et des poids, une frontière non linéaire.

</details>
<details><summary>Indice 2</summary>

Relis le tableau du §13.7, lignes « entraînement », « explicabilité », « probabilités » et « frontière ». Lequel n'a pas besoin de réentraînement pour intégrer un nouvel exemple ?

</details>
<details><summary>Indice 3</summary>

a) Situation 1 : des règles « si … et … alors … », c'est exactement ce qui se lit sur un petit arbre : **A**. Pour les quatre autres, cherche le classifieur dont la colonne du tableau répond à la contrainte de l'indice 1, chaque lettre servant une fois. b) Relis le paragraphe « Comment choisir » : comment rend-on une comparaison équitable ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 13.R1 — Ch. 12 : pourquoi standardiser avant de mesurer des distances

<details><summary>Indice 1</summary>

Écris les différences feature par feature entre C et chaque manchot, puis applique $\sqrt{\Delta_1^2 + \Delta_2^2}$.

</details>
<details><summary>Indice 2</summary>

C − A vaut $(2, 100)$ et C − B $(-18, 50)$, en mm et en g. Après standardisation, chaque différence est divisée par l'écart-type de sa feature (les moyennes s'annulent dans une différence).

</details>
<details><summary>Indice 3</summary>

a) $\sqrt{2^2 + 100^2} = \sqrt{10\,004} \approx 100{,}02$. Calcule b) de la même façon et compare pour c). d) Divise : $2/14$ et $100/800$, puis la racine de la somme des carrés. e) Fais le même calcul pour C − B avec $18/14$ et $50/800$, et compare. f) Relis la règle d'or du ch. 12 : où s'apprennent les paramètres d'un scaler ?

</details>

### 13.R2 — Ch. 10 : l'équation de la frontière d'un perceptron

<details><summary>Indice 1</summary>

Calcule $\mathbf{w} \cdot \mathbf{x} + b = x_1 + x_2 - 1{,}5$ pour chaque entrée, et regarde son signe.

</details>
<details><summary>Indice 2</summary>

Pour c), remplis la table de vérité sur les quatre entrées $(0, 0)$, $(0, 1)$, $(1, 0)$, $(1, 1)$. Pour d) et e), isole $x_2$ dans $x_1 + x_2 - 1{,}5 = 0$. Pour f), la formule est $\lvert \mathbf{w} \cdot \mathbf{x} + b \rvert / \lVert \mathbf{w} \rVert$.

</details>
<details><summary>Indice 3</summary>

a) $1 + 1 - 1{,}5 = 0{,}5 > 0$ : la sortie est 1. Fais de même pour b). c) Quelles entrées donnent 1 ? Reconnais la porte. d) et e) $x_2 = -x_1 + 1{,}5$ : lis la pente et l'ordonnée à l'origine. f) $\lVert \mathbf{w} \rVert = \sqrt 2$ et $\lvert f(1, 1) \rvert = 0{,}5$ : divise. g) Pour quelle valeur de $z$ a-t-on $\sigma(z) = 0{,}5$ ?

</details>

### 13.R3 — Ch. 6 : entropie de deux distributions

<details><summary>Indice 1</summary>

$H(p) = -\sum_i p_i \log_2 p_i$ ; les logarithmes en base 2 de $0{,}5$ et de $0{,}25$ sont des entiers.

</details>
<details><summary>Indice 2</summary>

Pour b), $\log_2 x = \ln x / \ln 2$. Pour c), l'entropie est maximale pour la distribution uniforme. Pour d), seuls les $\log_2 q_i$ changent par rapport à une entropie, les poids restent les $p_i$.

</details>
<details><summary>Indice 3</summary>

a) $-0{,}5 \times (-1) - 2 \times 0{,}25 \times (-2)$ : termine. b) $-0{,}8 \log_2 0{,}8 - 0{,}2 \log_2 0{,}2$, avec $\log_2 0{,}8 \approx -0{,}322$ et $\log_2 0{,}2 \approx -2{,}322$. c) $\log_2$ du nombre de valeurs. d) $-0{,}5 \log_2 0{,}8 - 0{,}5 \log_2 0{,}2$. e) Que vaut $H(p, q) - H(p)$, et quel est son signe (ch. 6) ?

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 13.1 — Vote des k plus proches voisins à la main ✏️

<details><summary>Indice 1</summary>

Commence par un tableau : pour chaque exemple, sa distance euclidienne à $(0, 0)$, sa distance de Manhattan et sa classe. Puis trie par distance.

</details>
<details><summary>Indice 2</summary>

Avec $\mathbf{q} = (0, 0)$, la distance euclidienne d'un exemple est $\sqrt{x_1^2 + x_2^2}$, et celle de Manhattan $\lvert x_1 \rvert + \lvert x_2 \rvert$. Pour le vote pondéré, chaque voisin compte $1/d$, et la probabilité d'une classe est la somme de ses poids divisée par la somme de tous les poids.

</details>
<details><summary>Indice 3</summary>

Distances euclidiennes, de l'exemple 0 à l'exemple 6 : 1,414 ; 1,5 ; 2 ; 2,5 ; 2,828 ; 3 ; 3,5. a) et b) Lis le plus petit, et sa classe. c) et d) Prends les trois premiers de la liste triée et compte leurs classes. e) Ajoute les deux suivants, compte, et, en cas d'égalité, applique la règle de l'énoncé. f) Tout le monde vote : compte les classes. g) Poids des trois plus proches : $1/1{,}414$, $1/1{,}5$ et $1/2$ ; additionne ceux de la classe 1, divise par le total. h) Même calcul avec les sept poids, classe par classe. i) Refais le tri avec les distances de Manhattan.

</details>

### Ex 13.2 — Gini et entropie d'une feuille ✏️

<details><summary>Indice 1</summary>

Les proportions sont $p_A = 0{,}1$ et $p_B = 0{,}9$ ; applique les formules de l'encadré 🧮 du §13.5.2.

</details>
<details><summary>Indice 2</summary>

Pour c) et d), une règle qui répond toujours la même classe se trompe sur tous les exemples de l'autre classe. Pour e), on se trompe quand la classe tirée diffère de la vraie : additionne les deux façons de se tromper. Pour h) et i), le maximum est atteint quand les classes sont à égalité.

</details>
<details><summary>Indice 3</summary>

a) $G = 1 - 0{,}1^2 - 0{,}9^2 = 1 - 0{,}01 - 0{,}81$ : termine. b) $H = -0{,}1 \log_2 0{,}1 - 0{,}9 \log_2 0{,}9$. c) Quelle part des exemples n'est pas de la classe B ? d) Même question pour A. e) $P(\text{vrai} = A) \times P(\text{tiré} = B) + P(\text{vrai} = B) \times P(\text{tiré} = A)$ : additionne les deux termes. f) $1 - 0{,}25 - 0{,}09 - 0{,}04$. g) Trois termes $-p \log_2 p$. h) $1 - 3 \times (1/3)^2$. i) $\log_2 3$.

</details>

### Ex 13.3 — Choisir le meilleur seuil sur la longueur des nageoires ✏️

<details><summary>Indice 1</summary>

Trie les dix nageoires en notant l'espèce de chacune, puis écris les seuils au milieu de deux valeurs consécutives.

</details>
<details><summary>Indice 2</summary>

Ordre trié : 181 A, 186 A, 190 A, 193 A, 197 A, 205 P, 210 A, 215 P, 217 P, 228 P. Pour chaque seuil, compte les Adélie et les Papou de chaque côté, calcule les deux Gini, puis le gain $G(\text{parent}) - \frac{n_G}{10} G(\text{gauche}) - \frac{n_D}{10} G(\text{droite})$.

</details>
<details><summary>Indice 3</summary>

a) Dix valeurs distinctes, neuf intervalles : 9 seuils. b) 6 Adélie et 4 Papou : $1 - 0{,}6^2 - 0{,}4^2$. c) $s = 201$ envoie à gauche les cinq Adélie de 181 à 197 (enfant pur), à droite 1 Adélie et 4 Papou ($G = 1 - 0{,}2^2 - 0{,}8^2$) : gain $= G(\text{parent}) - 0{,}5 \times 0 - 0{,}5 \times G$. d) et e) Même méthode : compte qui passe à gauche de 212,5 et de 207,5. f) Calcule le gain des neuf seuils (ceux des extrémités sont faibles) et garde le plus grand. g) Reprends le découpage du seuil trouvé en f), avec $H$ en bits. h) Lis la classe majoritaire de chaque feuille et compte ceux qui ne l'ont pas. i) 203 est-il à gauche ou à droite du seuil ? j) Pour chaque seuil, compte les manchots de chaque côté.

</details>

### Ex 13.4 — Distance à une droite et marge d'un SVM ✏️

<details><summary>Indice 1</summary>

Calcule $f(\mathbf{x}_i) = 0{,}5\,(x_1 + x_2) - 2$ pour les six points : tout le reste s'en déduit.

</details>
<details><summary>Indice 2</summary>

La distance à la frontière est $\lvert f \rvert / \lVert \mathbf{w} \rVert$. Un vecteur de support d'un SVM à marge dure vérifie $y_i f(\mathbf{x}_i) = 1$. Pour i), relis l'encadré 🧮 sur la distance : que devient le rapport $\lvert f \rvert / \lVert \mathbf{w} \rVert$ quand on multiplie tout par 2 ?

</details>
<details><summary>Indice 3</summary>

a) $\lVert \mathbf{w} \rVert = \sqrt{0{,}25 + 0{,}25}$. b) $0{,}5 \times 9 - 2$. c) Divise b) par a). d) Le $y_i f(\mathbf{x}_i)$ du point 0 vaut $0{,}5 \times 6 - 2 = 1$ ; calcule les cinq autres (attention au signe de $y_i$ pour les points 3 à 5) et prends le minimum. e) Garde les points où $y_i f = 1$. f) $2/\lVert \mathbf{w} \rVert$. g) Le signe de $f(2, 3)$. h) Compare $\lvert f(2, 3) \rvert$ à 1. i) Le numérateur et le dénominateur sont multipliés par le même nombre. j) Le point 1 est-il un vecteur de support ?

</details>

### Ex 13.5 — L'astuce du noyau vérifiée à la main ✏️

<details><summary>Indice 1</summary>

Calcule d'abord $\mathbf{x} \cdot \mathbf{z}$ et $\varphi(\mathbf{x})$, $\varphi(\mathbf{z})$, puis compare les deux façons d'obtenir le noyau.

</details>
<details><summary>Indice 2</summary>

Écris $\varphi(\mathbf{x})$ et $\varphi(\mathbf{z})$ coordonnée par coordonnée. Pour e), utilise $\lVert \varphi(\mathbf{x}) - \varphi(\mathbf{z}) \rVert^2 = K(\mathbf{x}, \mathbf{x}) - 2K(\mathbf{x}, \mathbf{z}) + K(\mathbf{z}, \mathbf{z})$. Pour h), compte les carrés, puis les paires $j < l$.

</details>
<details><summary>Indice 3</summary>

a) $1 \times 3 + 2 \times 1$. b) Le carré de a). c) $\sqrt 2 \times 1 \times 2$. d) $\varphi(\mathbf{z}) = (9,\ 3\sqrt 2,\ 1)$ : additionne les trois produits ($\sqrt 2 \times \sqrt 2 = 2$). e) $K(\mathbf{x}, \mathbf{x}) = (1 + 4)^2$ ; calcule $K(\mathbf{z}, \mathbf{z})$ de même, puis la formule. f) $\lVert \mathbf{x} - \mathbf{z} \rVert^2 = 2^2 + 1^2$, puis $e^{-0{,}5 \times 5}$. g) $e^{-1{,}5 \times 5}$, multiplié par $10^5$. h) $p$ carrés et $p(p - 1)/2$ paires.

</details>

### Ex 13.6 — Anglais ou français ? Naïve Bayes multinomial sur des lettres ✏️

<details><summary>Indice 1</summary>

Avec le lissage de Laplace, chaque compte gagne 1 et chaque total gagne $\alpha\,p = 4$ : $\hat\theta = (N + 1)/54$.

</details>
<details><summary>Indice 2</summary>

Écris les huit probabilités lissées $(N + 1)/54$, quatre par langue. Pour un mot, la log-probabilité jointe additionne $\log P(\text{classe})$ et, pour chaque lettre, son compte fois $\log \hat\theta$. Le posterior se déduit des deux log-probabilités jointes $\ell_{\text{EN}}$ et $\ell_{\text{FR}}$ : $P(\text{EN} \mid \text{mot}) = 1/(1 + e^{\ell_{\text{FR}} - \ell_{\text{EN}}})$.

</details>
<details><summary>Indice 3</summary>

a) Le compte de w en français, divisé par le total français. b) $8/54$. c) et d) Même formule avec les comptes français. e) $\log 0{,}6 + 2 \log\frac{31}{54} + \log\frac{13}{54} + \log\frac{8}{54}$ : évalue. f) La même chose avec les probabilités françaises et $\log 0{,}4$. g) Applique la formule de l'indice 2. h) Un facteur nul dans un produit… i) « que » : un q et un e ; même méthode qu'en e) à g).

</details>

### Ex 13.7 — Posterior d'un Naïve Bayes gaussien ✏️

<details><summary>Indice 1</summary>

Pour a) et b), la moyenne et la variance en divisant par $n = 4$. Pour la suite, la formule de l'encadré 🧮 sur la log-densité gaussienne (§13.6) : $\log \mathcal{N}(x;\ \mu, \sigma^2) = -\frac12 \log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}$.

</details>
<details><summary>Indice 2</summary>

La log-probabilité jointe d'une espèce additionne le log de son prior et les deux log-densités. Le posterior se déduit des deux log-probabilités jointes : $P(\text{Papou} \mid \mathbf{x}) = 1/(1 + e^{\ell_A - \ell_P})$. Pour j), ne garde que la nageoire dans les deux sommes.

</details>
<details><summary>Indice 3</summary>

a) La moyenne de 210, 215, 220, 225. b) Les écarts valent $-7{,}5$, $-2{,}5$, $2{,}5$, $7{,}5$ : moyenne de leurs carrés. c) $-\frac12 \log(2\pi \times 42{,}25) - \frac{15^2}{2 \times 42{,}25}$ : évalue. d) Même chose avec un écart de 12. e) et f) Mêmes calculs avec $\sigma^2 = 1{,}44$ et $1$. g) $\log 0{,}55 + $ c) $+$ e). h) $\log 0{,}45 + $ d) $+$ f). i) Formule de l'indice 2. j) Recommence g) à i) sans les termes du bec. k) Relis ce que suppose le produit des densités.

</details>

### Ex 13.8 — Gradient de la log-loss de la régression logistique ∂

<details><summary>Indice 1</summary>

Tout repose sur 1. et 2. : dérive $\sigma(z) = (1 + e^{-z})^{-1}$ comme une puissance $-1$, puis dérive $\ell$ par rapport à $z$ en utilisant $d \log \hat p / dz = \hat p' / \hat p$.

</details>
<details><summary>Indice 2</summary>

1. $\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2}$ ; écris $\frac{e^{-z}}{1 + e^{-z}} = 1 - \sigma(z)$. 2. $\frac{d}{dz}\log \hat p = 1 - \hat p$ et $\frac{d}{dz}\log(1 - \hat p) = -\hat p$. 3. $\partial z / \partial w_j = x_j$ et $\partial z / \partial b = 1$. 4. Écris la somme sur $i$ de $(\hat p_i - y_i)\,x_{ij}$ comme un produit matriciel. 7. Pour un exemple bien classé par $(\mathbf{w}_0, b_0)$, que devient $z_i$ quand on multiplie tout par $t$, et vers quoi tend $\hat p_i$ ?

</details>
<details><summary>Indice 3</summary>

1. $\sigma'(z) = \frac{1}{1 + e^{-z}} \times \frac{e^{-z}}{1 + e^{-z}}$ : reconnais les deux facteurs ; pour $1 - \sigma(z) = \sigma(-z)$, multiplie numérateur et dénominateur de $\frac{e^{-z}}{1 + e^{-z}}$ par $e^{z}$. 2. $\frac{d\ell}{dz} = -[y(1 - \hat p) - (1 - y)\hat p]$ : développe. 3. Règle de la chaîne : $\frac{\partial \ell}{\partial w_j} = \frac{d\ell}{dz}\,\frac{\partial z}{\partial w_j}$. 4. La $j$-ième coordonnée de $\mathbf{X}^\top \mathbf{v}$ est $\sum_i x_{ij} v_i$. 5. Dérive $\frac{\alpha}{2}\sum_j w_j^2$ par rapport à $w_j$. 6. Dérive $\hat p - y$ par rapport à $z$, avec 1. 7. Pour $y_i = 1$, $z_i \to +\infty$ et $\ell_i \to 0$ ; pour $y_i = 0$ ? Conclus sur l'infimum et le minimum. 8. Pour le perceptron, $\hat y \in \{0, 1\}$ : que vaut $y - \hat y$ pour un exemple bien classé ? Et $y - \hat p$ ?

</details>

### Ex 13.9 — Mémoire et temps d'un kNN sur MNIST complet 🧮

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la mémoire et le temps d'un kNN (§13.3) : $n \times p$ valeurs, 8 octets en `float64`, 1 en `uint8`, et $3np$ opérations par requête.

</details>
<details><summary>Indice 2</summary>

$n = 60\,000$ et $p = 784$. Pour e), multiplie le nombre d'opérations d'une requête par 10 000, puis divise par $10^{10}$. Pour f) et g), une matrice $a \times b$ de `float64` occupe $8ab$ octets. Pour h), $10 \times 784$ poids et 10 biais.

</details>
<details><summary>Indice 3</summary>

a) $60\,000 \times 784$. b) Multiplie a) par 8, puis divise par $10^6$. c) Même chose avec 1 octet. d) $3 \times$ a). e) $10\,000 \times$ d) $/ 10^{10}$. f) $10\,000 \times 60\,000 \times 8$ octets, en Go. g) $500 \times 60\,000 \times 8$ octets, en Mo. h) $10 \times 785$. i) b) en octets divisé par $8 \times$ h). j) $60\,000 \times 50 \times 8$ octets.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ Réflexion

### Ex 13.10 — La marge d'un SVM expliquée en 5 lignes 🗣️

<details><summary>Indice 1</summary>

Une image : une route entre deux villages, ou une ligne de démarcation dans une cour de récréation. Pourquoi la tracer au milieu, le plus loin possible des deux côtés ?

</details>
<details><summary>Indice 2</summary>

Construis les cinq lignes ainsi : l'image ; la marge (la largeur de la zone vide autour de la frontière) et pourquoi la maximiser (un nouveau point un peu décalé reste du bon côté) ; les vecteurs de support (les points au bord, qui « tiennent » la frontière) ; C quand les classes se chevauchent (le prix de chaque point toléré dans la zone) ; une conclusion.

</details>
<details><summary>Indice 3</summary>

Première ligne possible : « Imagine qu'on doit tracer une route entre deux villages : on la met le plus loin possible des maisons des deux côtés, pour avoir de la marge si un village s'étend un peu. » Écris les quatre autres en suivant le plan de l'indice 2, sans formule. Pour la question sur la figure, cherche ce qu'on peut **compter** ou **mesurer** sur la figure de la marge.

</details>

### Ex 13.11 — Refus de crédit : arbre lisible ou SVM opaque ? ⚖️

<details><summary>Indice 1</summary>

Pour chaque modèle, pose-toi la question du client : « Pourquoi moi ? Que devrais-je changer pour être accepté ? »

</details>
<details><summary>Indice 2</summary>

Pour 1, un chemin dans l'arbre est une suite de conditions qu'on peut écrire en français. Pour 2, une feature peut porter une information qu'on n'a pas donnée explicitement : c'est une **variable de substitution** (*proxy*). Pour 3, relis l'encadré 🕰️ du §13.7 (RGPD article 22, AI Act article 86). Pour 4, une AUC ne dit rien des coûts : il faut un seuil et le prix de chaque type d'erreur. Pour 5, pense aux explications post hoc et aux modèles lisibles par construction.

</details>
<details><summary>Indice 3</summary>

1. Avec l'arbre : « Votre demande a été refusée parce que le montant demandé dépasse … et que … » ; complète avec des conditions plausibles. Avec le SVM, que peut-on dire sans outil supplémentaire ? 2. Que peut révéler un code postal sur une personne, et l'âge est-il un critère qu'on peut librement utiliser ? Dans un arbre, une question sur le code postal se **voit** ; dans un SVM ? 3. Qui intervient, qui explique, qui peut changer la décision ? 4. À volume de demandes et coûts donnés, que représentent trois points d'AUC en crédits mal accordés ou mal refusés ? 5. Pour chaque solution, une phrase sur ce qu'elle garde et une sur ce qu'elle perd.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 13.E1 — Forces et faiblesses du kNN ; pourquoi standardiser

<details><summary>Indice 1</summary>

Trois forces (simplicité, aucune hypothèse sur la forme de la frontière, ajout d'exemples sans réentraînement), trois faiblesses (prédiction lente, mémoire, sensibilité à l'échelle et à la dimension).

</details>
<details><summary>Indice 2</summary>

Pour la standardisation : une distance additionne des écarts exprimés dans des unités différentes. Donne un exemple chiffré simple (des grammes contre des millimètres) et la règle d'or du ch. 12.

</details>
<details><summary>Indice 3</summary>

Structure : « Le kNN classe un point par le vote de ses $k$ plus proches voisins ; son entraînement se réduit à mémoriser les données. Forces : … Faiblesses : … Standardiser, parce que … ; et je règle $k$ par validation croisée, avec le scaler dans un `Pipeline`. » Termine par une ouverture : les recherches approximatives (HNSW, Faiss) pour passer à l'échelle.

</details>

### 13.E2 — Comment un arbre de décision choisit-il ses splits ?

<details><summary>Indice 1</summary>

Une impureté, un gain pondéré, une recherche gloutonne sur toutes les features et tous les seuils.

</details>
<details><summary>Indice 2</summary>

Définis Gini et l'entropie en une phrase chacune, donne le gain avec sa pondération, puis dis comment on énumère les seuils (au milieu de deux valeurs triées) et ce que « glouton » implique.

</details>
<details><summary>Indice 3</summary>

Structure : « À chaque nœud, l'arbre essaie toutes les questions “feature ≤ seuil”, et garde celle qui fait le plus baisser l'impureté, chaque enfant pesant selon sa part des exemples. L'impureté est… (Gini) ou… (entropie). En pratique… Limite : c'est glouton, donc… » Ajoute la variante en régression (la variance) et le coût (trier une fois par feature).

</details>

### 13.E3 — Comment limiter l'overfitting d'un arbre ?

<details><summary>Indice 1</summary>

100 % contre 78 % : nomme le diagnostic, puis donne les deux familles de remèdes.

</details>
<details><summary>Indice 2</summary>

Pré-élagage : `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`. Post-élagage : `ccp_alpha`. Choix par validation croisée. Et l'étape suivante : les ensembles d'arbres.

</details>
<details><summary>Indice 3</summary>

Structure : « C'est de l'overfitting : l'arbre a appris l'entraînement par cœur… Je limite d'abord sa croissance… ou je l'élague avec… Je choisis ces réglages par validation croisée, en traçant… Si un seul arbre reste trop instable, je passe à une forêt aléatoire ou au gradient boosting (ch. 14). » Pense aussi à vérifier les données (fuite, doublons).

</details>

### 13.E4 — L'hypothèse de Naïve Bayes, et pourquoi il marche quand même

<details><summary>Indice 1</summary>

L'hypothèse : indépendance des features **sachant la classe**. La raison du succès : classer demande seulement que la bonne classe ait le plus grand score.

</details>
<details><summary>Indice 2</summary>

Donne la formule du posterior en produit, un exemple d'hypothèse fausse (deux mots qui vont ensemble), puis les trois raisons du succès sur le spam : l'argmax résiste aux erreurs de probabilité, peu de paramètres par feature, beaucoup de features faibles qui votent ensemble. Termine par les limites (probabilités mal calibrées, lissage nécessaire).

</details>
<details><summary>Indice 3</summary>

Structure : « Naïve Bayes applique la règle de Bayes en supposant que, sachant la classe, les features sont indépendantes : $P(c \mid \mathbf{x}) \propto P(c) \prod_j P(x_j \mid c)$. C'est faux en général : par exemple… Il marche quand même parce que… Pour le spam… Attention à… »

</details>

### 13.E5 — Quelle baseline : régression logistique, SVM ou arbre ?

<details><summary>Indice 1</summary>

Une baseline doit être rapide, robuste, facile à interpréter et difficile à mal régler.

</details>
<details><summary>Indice 2</summary>

Dis lequel tu entraînes en premier et pourquoi (coefficients lisibles, probabilités, convexité), avec quel prétraitement (standardisation, one-hot, dans un `Pipeline`), puis quand tu passerais à un arbre (interactions, pas de mise à l'échelle, règles lisibles) ou à un SVM (frontière non linéaire, jeu de taille moyenne). Termine par les ensembles d'arbres.

</details>
<details><summary>Indice 3</summary>

Structure : « En premier, une régression logistique régularisée, dans un pipeline avec standardisation et one-hot, parce que… Je compare à un arbre peu profond, qui… Si les deux plafonnent et que les données sont de taille moyenne, j'essaie un SVM RBF… Mais sur du tabulaire, mon candidat sérieux est souvent le gradient boosting (ch. 14). Je compare tout en validation croisée sur les mêmes folds. »

</details>

<a id="notebook"></a>

## Notebook

### Ex 13.12 — Prédire la frontière de cinq classifieurs avant de l'afficher 🔮

<details><summary>Indice 1</summary>

Pour chaque classifieur, demande-toi ce qu'il **peut** représenter, pas seulement la forme de sa frontière. Une droite peut-elle séparer (0, 0) et (1, 1) de (0, 1) et (1, 0) ? Un modèle qui résume chaque classe par une moyenne et une variance par coordonnée voit-il une différence entre les deux classes d'un « ou exclusif » ?

</details>
<details><summary>Indice 2</summary>

Calcule, pour chaque classe, la moyenne de $x_1$ : la classe 1 réunit les coins (0, 1) et (1, 0), la classe 0 les coins (0, 0) et (1, 1). Pour l'arbre, compte les rectangles qu'il faut au minimum pour les quatre coins, puis pense aux points que le bruit (écart-type 0,25) envoie de l'autre côté de 0,5 : un arbre sans limite isole-t-il chaque point, ou seulement ceux qui sont mal placés ? Pour c), compare un vote de 15 voisins et un arbre qui a appris le bruit par cœur.

</details>
<details><summary>Indice 3</summary>

a) La moyenne de $x_1$ vaut 0,5 dans les deux classes (et de même pour $x_2$). Calcule aussi les variances, puis décide pour chacun des cinq si une frontière qui ne voit que ces résumés, ou une seule droite, peut faire mieux que le hasard. b) Compte : 4 rectangles pour les coins, plus quelques-uns par point mal placé près des droites $x = 0{,}5$ et $y = 0{,}5$ ; estime combien de points sur 420 le bruit y envoie. c) Pèse l'overfitting de l'arbre contre la frontière lissée du vote. N'exécute l'expérience qu'après.

</details>

### Ex 13.13 — Choisir k : de la frontière en dents de scie à l'érosion 🔬

<details><summary>Indice 1</summary>

`KNeighborsClassifier(n_neighbors=k).fit(X, y)`, puis `.score(X, y)` pour a). Pour b), `cross_val_score(model, X_13, Y_13, cv=CV_13)` rend les 5 accuracies d'un $k$ : prends leur moyenne. Pour d), entraîne sur tous les points et prédis le seul point `ISLAND_13` (une requête est un tableau 2-D : `[ISLAND_13]`).

</details>
<details><summary>Indice 2</summary>

Une liste en compréhension sur `KS_13` donne b) ; `KS_13[int(np.argmax(cv_13))]` donne c). Pour d), parcours `KS_13` dans l'ordre et arrête-toi au premier $k$ dont la prédiction en `ISLAND_13` vaut 0 (`next(k for k in KS_13 if ...)`).

</details>
<details><summary>Indice 3</summary>

```python
train_k1_13 = KNeighborsClassifier(n_neighbors=1).fit(X_13, Y_13).score(X_13, Y_13)
cv_13 = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_13, Y_13, cv=CV_13).mean() for k in KS_13]
best_k_13 = ...    # the k of the largest value of cv_13
erased_13 = ...    # the first k whose prediction at [ISLAND_13] is 0
```

</details>

### Ex 13.14 — Coder pairwise_distances et KNeighborsClassifier 🔨

<details><summary>Indice 1</summary>

`pairwise_distances` : convertis en tableaux de flottants, vérifie les dimensions (`ndim`, `shape[1]`) et l'absence de `NaN` (`np.isnan(...).any()`), puis `diff = A[:, None, :] - B[None, :, :]` (forme `(n_a, n_b, d)`) et une réduction sur le dernier axe. Calcule bien la **différence** de chaque paire : l'identité $\lVert\mathbf{a}\rVert^2 - 2\,\mathbf{a}\cdot\mathbf{b} + \lVert\mathbf{b}\rVert^2$, plus rapide (🛠️ 13.34 la mesure), perd en précision, et un point n'y est plus exactement à distance 0 de lui-même. `fit` ne calcule rien : il vérifie et range des copies (`np.array(X, dtype=float)` copie). `kneighbors` trie les distances de chaque ligne.

</details>
<details><summary>Indice 2</summary>

`kneighbors` : la matrice des distances aux données rangées, puis `np.argsort(D, axis=1, kind="stable")[:, :k]` et `np.take_along_axis(D, ind, axis=1)` pour les distances correspondantes. `predict_proba` : les labels des voisins (codés 0 à $K - 1$ avec `np.unique(y, return_inverse=True)` dans `fit`), leurs poids (1, ou $1/d$ ; si une ligne a des distances nulles, poids 1 pour elles et 0 pour les autres), puis une somme pondérée par classe divisée par la somme des poids. `predict` : `classes_[np.argmax(proba, axis=1)]` (`argmax` prend la première classe à égalité).

</details>
<details><summary>Indice 3</summary>

```python
def predict_proba(self, X):
    dist, ind = self.kneighbors(X)
    codes = self._codes_[ind]                                # (n, k) class codes of the neighbours
    if self.weights == "uniform":
        w = np.ones_like(dist)
    else:
        ...                                                  # 1/d, or the zero distances share all the weight
    proba = np.zeros((len(ind), len(self.classes_)))
    ...                                                      # column c: the sum of the weights of the class-c neighbours
    return proba / proba.sum(axis=1, keepdims=True)
```

</details>

### Ex 13.15 — kNN et malédiction de la dimension 🔬

<details><summary>Indice 1</summary>

`noisy_accuracy_15` : un générateur `rng = np.random.default_rng(seed)`, deux tirages `rng.normal(size=(lignes, n_noise))` (l'entraînement d'abord), `np.hstack` pour coller le bruit aux mesures, puis ton `KNeighborsClassifier(n_neighbors=5)` : `fit`, puis `score`. `distance_ratio_15` : ton `pairwise_distances(points[:50], points)` donne une matrice `(50, 500)` dont la ligne $i$ contient la distance du point $i$ à lui-même.

</details>
<details><summary>Indice 2</summary>

Avec `n_noise = 0`, `rng.normal(size=(233, 0))` est un tableau vide de forme `(233, 0)`, et `np.hstack` marche encore. Pour exclure le point lui-même du plus proche voisin, remplace les distances `D[i, i]` par `np.inf` (avec `np.arange(50)` pour les deux indices) avant le minimum ; pour le plus lointain, le maximum ne le choisit jamais (il vaut 0). b) : le premier élément de `NOISE_15` dont la moyenne est sous 0,9.

</details>
<details><summary>Indice 3</summary>

```python
def distance_ratio_15(p, seed=1315):
    points = np.random.default_rng(seed).uniform(size=(500, p))
    D = mylearn.neighbors.pairwise_distances(points[:50], points)
    farthest = D.max(axis=1)
    D[np.arange(50), np.arange(50)] = np.inf                 # a point is not its own neighbour
    ...                                                      # nearest, then the mean of nearest / farthest
```

</details>

### Ex 13.16 — SVM linéaire : vecteurs de support et paramètre C 📦

<details><summary>Indice 1</summary>

Un `SVC(kernel="linear")` entraîné expose `n_support_` (le nombre de vecteurs de support **par classe**), `support_` (leurs indices), `coef_` ($\mathbf{w}$, de forme `(1, 2)`), `intercept_` ($b$) et `decision_function(X)` ($f(\mathbf{x})$). La bande est l'ensemble des points où $\lvert f(\mathbf{x}) \rvert < 1$ : ses bords sont deux droites parallèles à la frontière.

</details>
<details><summary>Indice 2</summary>

La distance d'un point à la droite $f(\mathbf{x}) = 0$ vaut $\lvert f(\mathbf{x}) \rvert / \lVert\mathbf{w}\rVert$ (✏️ 13.4) : un bord est à $1/\lVert\mathbf{w}\rVert$ de la frontière. Pour c), `np.where(Y_16 == 1, 1, -1)` donne les $y_i$ ; multiplie-les par la fonction de décision et compte les valeurs sous 0,999.

</details>
<details><summary>Indice 3</summary>

```python
models_16 = {C: SVC(kernel="linear", C=C).fit(X_16, Y_16) for C in CS_16}
n_support_16 = [int(models_16[C].n_support_.sum()) for C in CS_16]
w_16 = models_16[1].coef_[0]
...    # b) from w_16 ; c) count y_i * decision_function(X_16) < 0.999
```

</details>

### Ex 13.17 — Relever les cercles en 3D pour les séparer par un plan 🎨

<details><summary>Indice 1</summary>

`lift_17` : `np.column_stack` des deux colonnes et de la somme des carrés (`(X ** 2).sum(axis=1)`). Les accuracies : un `SVC(kernel="linear", C=1)` entraîné sur `X_TRAIN_17`, puis sur `lift_17(X_TRAIN_17)`, scoré sur la partie de test transformée **de la même façon**. Pour la figure, `ax.scatter(x, y, z, c=y_classes)` sur un axe créé avec `projection="3d"`.

</details>
<details><summary>Indice 2</summary>

Le plan s'écrit $w_1 z_1 + w_2 z_2 + w_3 z_3 + b = 0$ ; résolu en $z_3$ : $z_3 = -(w_1 z_1 + w_2 z_2 + b)/w_3$, à évaluer sur une grille `np.meshgrid` de $(z_1, z_2)$, puis `ax.plot_surface(g1, g2, z3, alpha=0.3)`. Pour c), remplace $z_3$ par $x_1^2 + x_2^2$ dans l'équation du plan : tu obtiens l'équation d'un cercle, dont le centre est presque à l'origine si $w_1$ et $w_2$ sont petits.

</details>
<details><summary>Indice 3</summary>

$w_1 x_1 + w_2 x_2 + w_3 (x_1^2 + x_2^2) + b = 0$ : divise par $w_3$ et complète les carrés, $\left(x_1 + \frac{w_1}{2w_3}\right)^2 + \left(x_2 + \frac{w_2}{2w_3}\right)^2 = \frac{w_1^2 + w_2^2}{4w_3^2} - \frac{b}{w_3}$. Le rayon est la racine du membre de droite.

```python
model_3d_17 = SVC(kernel="linear", C=1).fit(lift_17(X_TRAIN_17), Y_TRAIN_17)
w_17, b_17 = model_3d_17.coef_[0], model_3d_17.intercept_[0]
radius_17 = ...    # the square root of the right-hand side above
```

</details>

### Ex 13.18 — Noyau RBF : régler C et gamma par validation croisée 📦

<details><summary>Indice 1</summary>

La recherche reçoit trois choses : le modèle à régler, la grille (`GRID_18`) et le découpage (`cv=CV_18`) ; un seul `fit`, sur la partie d'entraînement, fait tout (le réentraînement de la meilleure combinaison est le comportement par défaut, `refit=True`). Après `fit`, en plus de `best_params_` : `best_score_` (la moyenne de validation croisée de la meilleure combinaison), `score(X, y)` (celui du modèle réentraîné) et `cv_results_` (toutes les moyennes, pour la carte).

</details>
<details><summary>Indice 2</summary>

a) `[search_18.best_params_["C"], search_18.best_params_["gamma"]]`. c) `search_18.score(X_TEST_18, Y_TEST_18)`. d) un `SVC(C=1000, gamma=1000)` à part, entraîné sur la partie d'entraînement, scoré deux fois (entraînement, puis test).

</details>
<details><summary>Indice 3</summary>

```python
search_18 = GridSearchCV(SVC(kernel="rbf"), GRID_18, cv=CV_18).fit(X_TRAIN_18, Y_TRAIN_18)
best_18 = [search_18.best_params_["C"], search_18.best_params_["gamma"]]
best_cv_18 = ...      # the mean cross-validation accuracy of the best combination
test_18 = ...
overfit_18 = ...      # [training accuracy, test accuracy] of SVC(C=1000, gamma=1000)
```

</details>

### Ex 13.19 — Instabilité et profondeur d'un arbre 🔬

<details><summary>Indice 1</summary>

`trees_19` : une liste en compréhension sur `DRAWS_19`, chaque arbre entraîné sur `X_19[draw]`, `Y_19[draw]`. `disagreement_19` : les prédictions de chaque arbre sur `GRID_19`, puis, pour chaque paire $(i, j)$ avec $i < j$, la part des points où elles diffèrent. Le seuil de la racine d'un arbre scikit-learn : `tree.tree_.threshold[0]` ; sa feature : `tree.tree_.feature[0]`.

</details>
<details><summary>Indice 2</summary>

Range les prédictions dans un tableau `(20, 30 000)` ; `np.mean(P[i] != P[j])` donne la part de désaccord d'une paire ; il y a 190 paires. Pour d), `cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), X_19, Y_19, cv=CV_19).mean()` pour chaque `d` de `DEPTHS_19`, puis l'indice du maximum.

</details>
<details><summary>Indice 3</summary>

```python
def disagreement_19(trees):
    P = np.array([tree.predict(GRID_19) for tree in trees])     # (20, 30000)
    ...                                                          # the mean, over the pairs i < j, of np.mean(P[i] != P[j])

full_trees_19 = trees_19(None)
...    # a) the mean of get_n_leaves() ; b) two calls of disagreement_19 ; c) max - min of tree_.threshold[0] ; d) argmax
```

</details>

### Ex 13.20 — Coder les impuretés et le gain d'un split 🔨

<details><summary>Indice 1</summary>

Les trois impuretés commencent de la même façon : convertir `y` (et les poids, `np.ones(len(y))` si `None`), vérifier (vide, poids négatifs, formes, somme nulle), puis calculer des **proportions pondérées** de classes. Une fonction privée qui rend ces proportions sert deux fois. `split_gain` choisit l'impureté selon le critère (un dictionnaire de fonctions), puis applique la formule avec les poids totaux de chaque enfant.

</details>
<details><summary>Indice 2</summary>

Proportions pondérées : `classes, codes = np.unique(y, return_inverse=True)`, puis `np.bincount(codes, weights=w) / w.sum()`. Gini : `1 - np.sum(p ** 2)`. Entropie : garde seulement `p > 0` avant `np.log2` (une classe de poids nul ne doit pas donner `0 * log 0 = nan`). Variance : `np.average(y, weights=w)`, puis la moyenne pondérée des carrés des écarts. `split_gain` : `left = np.asarray(left_mask, dtype=bool)` ; si `left.all()` ou `not left.any()`, le gain vaut `0.0`.

</details>
<details><summary>Indice 3</summary>

```python
def split_gain(y, left_mask, criterion="gini", sample_weight=None):
    impurity = {"gini": gini_impurity, "entropy": entropy_impurity, "squared_error": variance_impurity}[criterion]  # KeyError -> ValueError
    y, left, w = ...                                         # arrays of the same length, checked
    if left.all() or not left.any():
        return 0.0
    w_left, w_right = w[left].sum(), w[~left].sum()
    ...    # parent impurity minus the two weighted children; return float(...)
```

</details>

### Ex 13.21 — L'arbre parfait qui triche : une colonne identifiant 🐛

<details><summary>Indice 1</summary>

Un modèle parfait sur des données réelles mérite une enquête : regarde `FEATURES_21` et `TREE_21.feature_importances_`, et affiche les règles de l'arbre (`export_text(TREE_21, feature_names=FEATURES_21)`). D'où peut venir une colonne que tu n'attendais pas ?

</details>
<details><summary>Indice 2</summary>

a) le nom de `FEATURES_21` à l'indice `np.argmax(TREE_21.feature_importances_)`. b) `TREE_21.score(NEXT_21[FEATURES_21], NEXT_21["species"])`. c) un nouvel arbre sur `TRAIN_21[MEASURES]` (les quatre mesures, sans la colonne fautive), scoré sur `TEST_21[MEASURES]`. d) son `get_n_leaves()`.

</details>
<details><summary>Indice 3</summary>

a) `top_feature_21 = FEATURES_21[int(np.argmax(TREE_21.feature_importances_))]` : c'est la colonne que `read_csv` a créée à partir de l'index écrit par `to_csv`. Pour b), regarde aussi ce que `TREE_21.predict(NEXT_21[FEATURES_21])` prédit (`pd.Series(...).value_counts()`) et les valeurs de cette colonne dans `NEXT_21`. Pour c) et d), refais l'arbre avec `MEASURES` seulement.

</details>

### Ex 13.22 — Coder la recherche du meilleur split 🔨

<details><summary>Indice 1</summary>

Deux boucles : sur les features candidates, puis sur leurs seuils. Pour une feature, les valeurs distinctes triées (`np.unique(X[:, j])`) donnent les seuils candidats, les milieux `(v[:-1] + v[1:]) / 2`. Pour chaque seuil : le masque `X[:, j] <= s`, le test de `min_samples_leaf` sur les deux enfants (comptés sans les poids), puis `split_gain` avec les poids. Garde le meilleur.

</details>
<details><summary>Indice 2</summary>

Rassemble d'abord les candidats `(feature, seuil)` (avec `splitter="random"`, tire les seuils dans l'ordre de `feature_indices`, sans le trier), puis examine-les triés par feature croissante et seuil croissant, et ne remplace le meilleur que si `gain > best_gain + 1e-12` : les égalités vont alors d'elles-mêmes à la plus petite feature, puis au plus petit seuil. Pars de `best_gain = 1e-12` (ou d'une petite tolérance) pour rendre `None` quand aucun gain n'est strictement positif. `splitter="random"` : pour chaque feature, dans l'ordre de `feature_indices`, saute-la si `min == max`, sinon un seul seuil `rng.uniform(min, max)`.

</details>
<details><summary>Indice 3</summary>

```python
def best_split(X, y, criterion="gini", sample_weight=None, feature_indices=None, min_samples_leaf=1,
               splitter="best", rng=None):
    ...                                                       # checks, w, features, rng = np.random.default_rng(rng)
    candidates = []                                           # (feature, threshold)
    for j in features:
        values = np.unique(X[:, j])
        if splitter == "best":
            candidates += [(j, s) for s in (values[:-1] + values[1:]) / 2]
        elif len(values) > 1:
            candidates.append((j, rng.uniform(values[0], values[-1])))
    best, best_gain = None, 1e-12
    for j, s in sorted(candidates):                           # lowest feature, then lowest threshold first
        ...                                                   # mask, min_samples_leaf, gain; keep if gain > best_gain + 1e-12
```

</details>

### Ex 13.23 — Arbres scikit-learn : plot_tree, export_text et élagage ccp_alpha 📦

<details><summary>Indice 1</summary>

`full_23.get_depth()` et `get_n_leaves()` pour a). `full_23.cost_complexity_pruning_path(X, y)` rend un objet dont l'attribut `ccp_alphas` liste, du plus petit au plus grand, les valeurs où l'élagage retire un sous-arbre ; la dernière ne laisse que la racine. Pour chaque valeur, un arbre `DecisionTreeClassifier(ccp_alpha=alpha, random_state=0)` évalué en validation croisée.

</details>
<details><summary>Indice 2</summary>

Range les moyennes de validation croisée dans un tableau `cv_23`, puis garde les valeurs `alphas_23[np.isclose(cv_23, cv_23.max())]` et prends leur maximum : à égalité, l'arbre le plus simple. `pruned_23` se réentraîne sur toute la partie d'entraînement avec cette valeur.

</details>
<details><summary>Indice 3</summary>

```python
full_23 = DecisionTreeClassifier(random_state=0).fit(X_TRAIN_23, Y_TRAIN_23)
size_23 = [full_23.get_depth(), full_23.get_n_leaves()]
alphas_23 = full_23.cost_complexity_pruning_path(X_TRAIN_23, Y_TRAIN_23).ccp_alphas[:-1]
cv_23 = np.array([cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=0), X_TRAIN_23, Y_TRAIN_23,
                                  cv=CV_23).mean() for a in alphas_23])
...    # b) the largest tied alpha ; pruned_23, c) and d)
```

</details>

### Ex 13.24 — Coder un arbre de décision récursif 🔨

<details><summary>Indice 1</summary>

Le pseudo-code de la fiche (§13.5) est la méthode : une fonction `build(lignes, profondeur)` qui crée le nœud (sa `value`, son `n_samples`, son `impurity`), décide s'il est une feuille (pur, profondeur atteinte, moins de `min_samples_split` exemples, ou `best_split` rend `None`), et sinon le coupe et construit ses deux enfants. Range dans `fit` ce qui ne change pas d'un nœud à l'autre : `X` en flottants, les labels codés 0 à $K - 1$, le critère.

</details>
<details><summary>Indice 2</summary>

Code les labels avec `self.classes_, codes = np.unique(y, return_inverse=True)` : la `value` d'un nœud est alors `np.bincount(codes[rows], minlength=K) / len(rows)` (avec les poids en 13.25). Pour prédire, descends depuis `root_` : `node.left` si `x[node.feature] <= node.threshold`, sinon `node.right`, jusqu'à une feuille. `get_depth` et `get_n_leaves` se calculent par récursion sur les enfants (une feuille : profondeur 0, une feuille). Une pile `[(nœud, lignes, profondeur)]` remplace la récursion si tu crains la limite de Python.

</details>
<details><summary>Indice 3</summary>

```python
def _build(self, X, codes, rows, depth):
    node = Node(feature=None, threshold=None, left=None, right=None,
                value=np.bincount(codes[rows], minlength=len(self.classes_)) / len(rows),
                n_samples=len(rows), impurity=...)              # the impurity of codes[rows] for the criterion
    if node.impurity == 0 or (self.max_depth is not None and depth >= self.max_depth) \
            or len(rows) < self.min_samples_split:
        return node
    split = best_split(X[rows], codes[rows], self.criterion, min_samples_leaf=self.min_samples_leaf)
    if split is None:
        return node
    ...    # node.feature, node.threshold, then the two children with depth + 1
```

</details>

### Ex 13.25 — Préparer l'arbre pour les ensembles : poids, features tirées, importances 🔨

<details><summary>Indice 1</summary>

Trois ajouts dans ta construction : les poids passent partout où l'on calcule des proportions, des impuretés ou des gains (`value`, `impurity`, `best_split(..., sample_weight=w[rows])`) ; un générateur créé une fois dans `fit` sert à tirer les features à chaque nœud qu'on tente de couper ; chaque split ajoute sa contribution à un tableau d'importances, normalisé à la fin.

</details>
<details><summary>Indice 2</summary>

Le nombre de features tirées : une fonction privée qui traduit `max_features` en $k$ (ou `None`), avec les `ValueError` du contrat ; à chaque nœud, `features = rng.choice(p, size=k, replace=False)` puis `best_split(..., feature_indices=features, splitter=self.splitter, rng=rng)`. L'importance d'un split : `w[rows].sum() / w.sum() * gain`, ajoutée à `importances[feature]` ; à la fin, divise par la somme si elle est positive. La `value` pondérée : `np.bincount(codes[rows], weights=w[rows], minlength=K) / w[rows].sum()`.

</details>
<details><summary>Indice 3</summary>

```python
def _n_features(self, p):
    m = self.max_features
    if m is None:
        return None
    if m == "sqrt":
        return max(1, int(np.sqrt(p)))
    ...    # "log2", an int in 1..p, a float in (0, 1]; ValueError otherwise (bool is not an int here)

# in fit:  rng = np.random.default_rng(self.random_state) ; k = self._n_features(p) ; importances = np.zeros(p)
# at a node:  features = None if k is None else rng.choice(p, size=k, replace=False)
```

</details>

### Ex 13.26 — Arbre de régression : même squelette, autre critère 🔨

<details><summary>Indice 1</summary>

Tout ce qui distingue le régresseur du classifieur tient en trois points : le critère (`"squared_error"`, donc `variance_impurity`), la `value` d'un nœud (la moyenne pondérée des cibles, dans un tableau de forme `(1,)`) et la sortie de `predict` (cette moyenne, pas une classe). Partage la construction entre les deux classes plutôt que de la recopier : isole-la dans une fonction privée qui reçoit ces différences en paramètres (ou dans une classe de base).

</details>
<details><summary>Indice 2</summary>

Une fonction privée `_grow(tree, X, y, w, criterion, value_of)` (`tree` est l'objet qui porte les hyperparamètres) où `value_of(y_rows, w_rows)` rend la `value` d'un nœud : les proportions pour le classifieur, `np.array([np.average(y_rows, weights=w_rows)])` pour le régresseur. Un nœud est « pur » quand son impureté vaut 0 (des cibles toutes égales). `score` : $1 - \sum (y - \hat y)^2 / \sum (y - \bar y)^2$.

</details>
<details><summary>Indice 3</summary>

```python
class DecisionTreeRegressor:
    def fit(self, X, y, sample_weight=None):
        ...                                                   # checks: criterion == "squared_error", X, y as floats, w
        self.root_, self.feature_importances_ = _grow(
            self, X, y, w, "squared_error",
            value_of=lambda y_rows, w_rows: np.array([np.average(y_rows, weights=w_rows)]))
        self.n_features_in_ = X.shape[1]
        return self
```

</details>

### Ex 13.27 — Naïve Bayes sur des nuages gaussiens puis sur des lunes 🔮

<details><summary>Indice 1</summary>

Naïve Bayes gaussien attribue un point à la classe dont la densité (multipliée par le prior) est la plus grande. Pour a), compare comment les deux densités décroissent quand on s'éloigne : laquelle est encore la plus grande très loin des deux centres, dans n'importe quelle direction ?

</details>
<details><summary>Indice 2</summary>

Pour b), chaque lune est résumée par une gaussienne aux axes parallèles aux coordonnées : regarde la figure des lunes de 13.13 ou 13.19 et imagine la meilleure frontière quadratique. Pour c), demande-toi d'où vient l'erreur : d'une estimation imprécise des 8 paramètres (variance), ou de la forme du modèle (biais) ? Des données en plus corrigent l'une, pas l'autre.

</details>
<details><summary>Indice 3</summary>

a) Écris les deux log-densités le long de l'axe $x_2$, en $x_1 = 0$ : $-\frac{x_2^2}{2 \times 0{,}25}$ pour la classe 0 et $-\frac{x_2^2}{2 \times 2{,}25}$ (plus une constante) pour la classe 1 ; laquelle l'emporte quand $\lvert x_2 \rvert$ grandit ? Fais de même le long de $x_1$, des deux côtés. b) et c) : raisonne sur la forme de la frontière avant de choisir. N'exécute l'expérience qu'après.

</details>

### Ex 13.28 — Coder GaussianNB avec log-densités et logsumexp 🔨

<details><summary>Indice 1</summary>

`log_gaussian_pdf` : la formule en log, écrite directement avec des tableaux NumPy (le broadcasting fait le reste), sans calculer la densité puis son logarithme (loin de la moyenne, l'exponentielle vaudrait 0, et son log $-\infty$), après avoir vérifié `np.all(var > 0)`. `logsumexp` : retire le maximum de l'axe avant l'exponentielle, puis rajoute-le après le logarithme. `fit` : une boucle sur les classes, avec le masque `y == c`, pour les comptes, les moyennes et les variances.

</details>
<details><summary>Indice 2</summary>

`logsumexp` : le maximum de l'axe, gardé avec `keepdims=True` pour pouvoir le retrancher par broadcasting, puis la somme des exponentielles des écarts, son logarithme, et le maximum rajouté (sans sa dimension) ; si une ligne est toute à $-\infty$, l'écart donne `nan` (non testé). `epsilon_` : `self.var_smoothing * X.var(axis=0).max()`. `predict_joint_log_proba` : `log_gaussian_pdf(X[:, None, :], self.theta_[None], self.var_[None])` a la forme `(n, K, d)` ; somme sur le dernier axe, plus `np.log(self.class_prior_)`. `predict_log_proba` : `joint - logsumexp(joint, axis=1)[:, None]`.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y):
    ...                                                       # checks (2-D, NaN, lengths)
    self.classes_ = np.unique(y)
    masks = [y == c for c in self.classes_]
    self.class_count_ = np.array([m.sum() for m in masks], dtype=float)
    self.theta_ = np.array([X[m].mean(axis=0) for m in masks])
    self.epsilon_ = float(self.var_smoothing * X.var(axis=0).max())
    ...                                                       # var_ (+ epsilon_), class_prior_ (priors checked)
    return self
```

</details>

### Ex 13.29 — Coder MultinomialNB avec lissage 🔨

<details><summary>Indice 1</summary>

`fit` compte : pour chaque classe, la somme des lignes de cette classe (`X[y == c].sum(axis=0)`) donne une ligne de `feature_count_`. Le lissage ajoute $\alpha$ à chaque compte et $\alpha d$ au total de la classe. Les méthodes de prédiction sont celles de `GaussianNB`, avec une log-probabilité jointe qui n'est qu'un produit matriciel.

</details>
<details><summary>Indice 2</summary>

`N_c = feature_count_.sum(axis=1, keepdims=True)` ; `feature_log_prob_ = np.log(feature_count_ + alpha) - np.log(N_c + alpha * d)`. `class_log_prior_` : `np.log(class_count_ / class_count_.sum())`, ou `np.full(K, -np.log(K))` si `fit_prior=False`. Avec `alpha=0`, `np.log(0)` vaut $-\infty$ avec un avertissement : `np.errstate(divide="ignore")` le fait taire.

</details>
<details><summary>Indice 3</summary>

```python
def predict_joint_log_proba(self, X):
    if not hasattr(self, "feature_log_prob_"):
        raise RuntimeError("call fit first")
    X = ...                                                   # 2-D floats, the right number of columns
    return X @ self.feature_log_prob_.T + self.class_log_prior_
```

</details>

### Ex 13.30 — Dialogue ou narration ? Classer des phrases de Holmes : sac de mots, TF-IDF et Naïve Bayes 📦

<details><summary>Indice 1</summary>

`make_pipeline(CountVectorizer(), MultinomialNB())` est un seul modèle : `fit(S_TRAIN_30, Y_TRAIN_30)` transforme les phrases en comptes de mots, puis entraîne Naïve Bayes ; `score(S_TEST_30, Y_TEST_30)` fait de même sur le test. La baseline ne demande aucun modèle : la classe la plus fréquente de `Y_TRAIN_30`, comparée à `Y_TEST_30`.

</details>
<details><summary>Indice 2</summary>

Les étapes d'un pipeline s'atteignent par leur position : `pipe[0]` est le vectoriseur (`get_feature_names_out()` donne le vocabulaire, dans l'ordre des colonnes), `pipe[1]` le classifieur (`feature_log_prob_` a une ligne par classe, la ligne 1 pour le dialogue). L'écart de f) est un tableau d'un nombre par mot ; `np.argsort(...)[::-1][:5]` donne les indices des 5 plus grands.

</details>
<details><summary>Indice 3</summary>

```python
dialogue_share_30 = float(LABELS_30.mean())
majority_30 = int(Y_TRAIN_30.mean() > 0.5)
baseline_30 = float(np.mean(Y_TEST_30 == majority_30))
count_pipe_30 = make_pipeline(CountVectorizer(), MultinomialNB()).fit(S_TRAIN_30, Y_TRAIN_30)
...    # c) to e) ; f) from count_pipe_30[0].get_feature_names_out() and count_pipe_30[1].feature_log_prob_
```

</details>

### Ex 13.31 — Régression logistique scikit-learn : coefficients et probabilités 📦

<details><summary>Indice 1</summary>

Après `fit`, `coef_` a la forme `(1, 4)` (une ligne pour le problème binaire) et `classes_` dit quelle classe est la classe 1 (l'ordre alphabétique). Le modèle calcule $\log \frac{P(\text{classe 1})}{P(\text{classe 0})} = \mathbf{w} \cdot \mathbf{z} + b$ sur des mesures **standardisées** : un nouveau manchot passe d'abord par `SCALER_31`.

</details>
<details><summary>Indice 2</summary>

Si $z_j$ augmente de 1 (un écart-type), le log des odds augmente de $w_j$ : les odds sont multipliées par $e^{w_j}$. La probabilité du Chinstrap est la colonne de `predict_proba` qui correspond à `"Chinstrap"` dans `classes_`. Pour d), deux modèles de plus, et `np.linalg.norm` de leurs coefficients.

</details>
<details><summary>Indice 3</summary>

```python
model_31 = LogisticRegression().fit(Z_TRAIN_31, Y_TRAIN_31)
coef_31 = model_31.coef_[0]
odds_31 = float(np.exp(coef_31[0]))                          # MEASURES[0] is the bill length
proba_31 = ...      # predict_proba of SCALER_31.transform(NEW_31), the "Chinstrap" column
norms_31 = ...      # [norm for C = 1, norm for C = 100]
```

</details>

### Ex 13.32 — Coder la régression logistique par descente de gradient 🔨

<details><summary>Indice 1</summary>

C'est la descente de gradient du ch. 5 sur l'objectif de ∂ 13.8 : à chaque pas, les probabilités $\mathbf{p} = \sigma(\mathbf{X}\mathbf{w} + b)$, les deux gradients, puis le pas. `fit` commence par coder les labels en 0 et 1 avec `classes_` (`np.unique(y, return_inverse=True)` donne les deux à la fois) et par calculer l'objectif au départ, avant tout pas.

</details>
<details><summary>Indice 2</summary>

Une méthode privée `_objective(X, y01, w, b)` évite d'écrire deux fois `log_loss(y01, sigmoid(X @ w + b)) + 0.5 * self.alpha * w @ w`. Dans la boucle : `grad_w = X.T @ (p - y01) / n + self.alpha * w`, `grad_b = np.mean(p - y01)` (ou 0 si `fit_intercept=False`), puis les deux mises à jour, puis le nouvel objectif ajouté à l'historique ; arrête-toi dès que `abs(new - old) < self.tol`, après avoir compté le pas.

</details>
<details><summary>Indice 3</summary>

```python
w, b = np.zeros(X.shape[1]), 0.0
history = [self._objective(X, y01, w, b)]
for step in range(1, self.max_iter + 1):
    p = sigmoid(X @ w + b)
    grad_w = X.T @ (p - y01) / len(X) + self.alpha * w
    ...                                                       # grad_b, both updates, history.append(...)
    if abs(history[-1] - history[-2]) < self.tol:
        break
self.n_iter_, self.loss_history_ = step, history
```

</details>

### Ex 13.33 — Associer quatre frontières à leurs classifieurs 📈

<details><summary>Indice 1</summary>

Relis le tableau du §13.7, ligne « frontière » : une droite, une courbe lisse (quadratique), un escalier de segments parallèles aux axes, une frontière quelconque qui suit les points. Cherche d'abord le panneau le plus facile à reconnaître, puis élimine.

</details>
<details><summary>Indice 2</summary>

Un plus proche voisin ($k = 1$) donne une région à chaque point d'entraînement, même isolé au milieu de l'autre classe : cherche les petits îlots. Un arbre de profondeur 6 n'a que des segments horizontaux et verticaux. Entre une droite et une courbe, regarde si la frontière s'infléchit.

</details>
<details><summary>Indice 3</summary>

Panneau 1 : une courbe lisse, qui s'infléchit, sans îlot ni marche d'escalier : la frontière quadratique de Naïve Bayes gaussien, donc C. Pour les panneaux 2 à 4, applique les critères de l'indice 2 (des îlots autour de points isolés ? des segments parallèles aux axes ? une droite d'un bout à l'autre ?). Pour b), demande-toi lequel des quatre modèles fait de chaque point d'entraînement son propre voisin.

</details>

### Ex 13.34 — Mesurer et vectoriser : accélérer son kNN 🛠️

<details><summary>Indice 1</summary>

Les trois versions rendent la même matrice `(len(A), len(B))`. La boucle remplit un tableau `np.empty` case par case ; le broadcasting crée un tableau `(len(A), len(B), d)` de différences ; l'identité ne demande que les normes au carré des lignes (`(A ** 2).sum(axis=1)`) et un produit `A @ B.T`. `timed_34` appelle la fonction `repeat` fois, chaque appel entre deux `time.perf_counter()`.

</details>
<details><summary>Indice 2</summary>

L'identité : `squared = (A ** 2).sum(axis=1)[:, None] - 2 * A @ B.T + (B ** 2).sum(axis=1)[None, :]` (formes `(n_a, 1)`, `(n_a, n_b)` et `(1, n_b)`). L'arrondi peut rendre un carré légèrement négatif là où la distance vaut 0 : `np.maximum(squared, 0.0)` avant `np.sqrt`. Pour `timed_34`, garde le **minimum** des durées.

</details>
<details><summary>Indice 3</summary>

```python
def timed_34(function, *args, repeat=3):
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        function(*args)
        ...                                                   # append the duration of this call
    return ...                                                # the smallest one


def distances_dot_34(A, B):
    squared = ...                                             # the identity, one matrix product
    return np.sqrt(np.maximum(squared, 0.0))
```

</details>

### Ex 13.35 — Comparatif mylearn contre scikit-learn : accuracy et temps 🔬

<details><summary>Indice 1</summary>

Trois mesures dans une seule fonction : un `time.perf_counter()` avant `fit`, un entre `fit` et `score`, un après `score`. La fonction ne fait rien d'autre : la vérification lui passe les six modèles.

</details>
<details><summary>Indice 2</summary>

Rends un tuple de trois nombres, dans l'ordre `(accuracy, durée du fit, durée du score)` ; convertis l'accuracy en `float` (le `score` de scikit-learn rend un `np.float64`, celui de ta librairie un `float`). N'appelle pas `fit(...).score(...)` en chaîne : si ton `fit` oublie `return self`, l'erreur cacherait la mesure.

</details>
<details><summary>Indice 3</summary>

```python
def timed_fit_score_35(model, X_train, y_train, X_test, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    middle = time.perf_counter()
    ...                                                       # the accuracy, then the tuple of three numbers
```

</details>

### Ex 13.36 — Défi : 96 % sur MNIST avec un classifieur classique 🏆

<details><summary>Indice 1</summary>

784 pixels, c'est beaucoup pour un kNN ou un SVM, et beaucoup de ces pixels sont presque toujours noirs : réduis d'abord la dimension (PCA, ch. 12, 12.24). Parmi les classifieurs du chapitre, ceux qui savent tracer une frontière très souple sur des données de taille moyenne sont le kNN et le SVM à noyau.

</details>
<details><summary>Indice 2</summary>

`make_pipeline(PCA(n_components=...), SVC(C=...))` ou `make_pipeline(PCA(...), KNeighborsClassifier(...))` : la PCA s'apprend dans chaque fold, sans fuite. Règle le nombre de composantes et C (ou $k$) par validation croisée sur 5 000 à 10 000 images d'entraînement (`GridSearchCV`, grille logarithmique pour C), puis entraîne le modèle retenu sur 20 000 images.

</details>
<details><summary>Indice 3</summary>

```python
N_TRAIN_36 = 20_000


def build_model_36():
    return make_pipeline(PCA(n_components=50, random_state=0), SVC(C=...))   # C from your cross-validation
```

Une grille de départ : `{"pca__n_components": [30, 50, 80], "svc__C": [1, 5, 10]}` sur 5 000 images d'entraînement, avec `GridSearchCV(..., cv=3)`.

</details>
