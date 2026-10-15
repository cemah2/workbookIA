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

### 13.R1 — Ch. 12 : Pourquoi standardiser avant de mesurer des distances

<details><summary>Indice 1</summary>

Écris les différences feature par feature entre C et chaque manchot, puis applique $\sqrt{\Delta_1^2 + \Delta_2^2}$.

</details>
<details><summary>Indice 2</summary>

C − A vaut $(2, 100)$ et C − B $(-18, 50)$, en mm et en g. Après standardisation, chaque différence est divisée par l'écart-type de sa feature (les moyennes s'annulent dans une différence).

</details>
<details><summary>Indice 3</summary>

a) $\sqrt{2^2 + 100^2} = \sqrt{10\,004} \approx 100{,}02$. Calcule b) de la même façon et compare pour c). d) Divise : $2/14$ et $100/800$, puis la racine de la somme des carrés. e) Fais le même calcul pour C − B avec $18/14$ et $50/800$, et compare. f) Relis la règle d'or du ch. 12 : où s'apprennent les paramètres d'un scaler ?

</details>

### 13.R2 — Ch. 10 : L'équation de la frontière d'un perceptron

<details><summary>Indice 1</summary>

Calcule $\mathbf{w} \cdot \mathbf{x} + b = x_1 + x_2 - 1{,}5$ pour chaque entrée, et regarde son signe.

</details>
<details><summary>Indice 2</summary>

Pour c), remplis la table de vérité sur les quatre entrées $(0, 0)$, $(0, 1)$, $(1, 0)$, $(1, 1)$. Pour d) et e), isole $x_2$ dans $x_1 + x_2 - 1{,}5 = 0$. Pour f), la formule est $\lvert \mathbf{w} \cdot \mathbf{x} + b \rvert / \lVert \mathbf{w} \rVert$.

</details>
<details><summary>Indice 3</summary>

a) $1 + 1 - 1{,}5 = 0{,}5 > 0$ : la sortie est 1. Fais de même pour b). c) Quelles entrées donnent 1 ? Reconnais la porte. d) et e) $x_2 = -x_1 + 1{,}5$ : lis la pente et l'ordonnée à l'origine. f) $\lVert \mathbf{w} \rVert = \sqrt 2$ et $\lvert f(1, 1) \rvert = 0{,}5$ : divise. g) Pour quelle valeur de $z$ a-t-on $\sigma(z) = 0{,}5$ ?

</details>

### 13.R3 — Ch. 6 : Entropie de deux distributions

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

a) Dix valeurs distinctes, neuf intervalles : 9 seuils. b) 6 Adélie et 4 Papou : $1 - 0{,}6^2 - 0{,}4^2$. c) $s = 201$ envoie à gauche les cinq Adélie de 181 à 197 (enfant pur), à droite 1 Adélie et 4 Papou ($G = 1 - 0{,}2^2 - 0{,}8^2 = 0{,}32$) : gain $= G(\text{parent}) - 0{,}5 \times 0 - 0{,}5 \times 0{,}32$. d) et e) Même méthode : compte qui passe à gauche de 212,5 et de 207,5. f) Calcule le gain des neuf seuils (ceux des extrémités sont faibles) et garde le plus grand. g) Reprends le découpage du seuil trouvé en f), avec $H$ en bits. h) Lis la classe majoritaire de chaque feuille et compte ceux qui ne l'ont pas. i) 203 est-il à gauche ou à droite du seuil ? j) Pour chaque seuil, compte les manchots de chaque côté.

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

Pour 1, un chemin dans l'arbre est une suite de conditions qu'on peut écrire en français. Pour 2, une feature peut porter une information qu'on n'a pas donnée explicitement (une **variable de substitution**, *proxy*). Pour 3, relis l'encadré 🕰️ du §13.7 (RGPD article 22, AI Act article 86). Pour 4, une AUC ne dit rien des coûts : il faut un seuil et le prix de chaque type d'erreur. Pour 5, pense aux explications post hoc et aux modèles lisibles par construction.

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

Structure : « À chaque nœud, l'arbre essaie toutes les questions "feature ≤ seuil", et garde celle qui fait le plus baisser l'impureté, chaque enfant pesant selon sa part des exemples. L'impureté est… (Gini) ou… (entropie). En pratique… Limite : c'est glouton, donc… » Ajoute la variante en régression (la variance) et le coût (trier une fois par feature).

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

Les indices des exercices du notebook (13.12 à 13.36) arriveront avec eux, à la prochaine session de génération.
