# 2 · Hasard et statistiques de base — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🧮 🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à D](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 2.Q1 — Moyenne, médiane, mode : laquelle résiste aux valeurs extrêmes ?

<details><summary>Indice 1</summary>

Relis les définitions de la fiche §2.2 : laquelle des trois utilise **toutes** les valeurs dans son calcul, laquelle ne regarde que la position du milieu, laquelle ne regarde que les répétitions ?

</details>
<details><summary>Indice 2</summary>

Pour la médiane, trie d'abord la liste (elle l'est déjà ici) et prends la valeur du milieu. Pour la question 2, demande-toi si le 40 intervient dans le calcul de chaque statistique ou seulement par sa **place** dans la liste triée.

</details>
<details><summary>Indice 3</summary>

Question 1 : la somme des sept valeurs vaut 76, donc la moyenne vaut $\frac{76}{7} \approx 10{,}86$ ; la médiane est la 4ᵉ des sept valeurs triées, 6 ; le mode, la seule valeur répétée, est 5. Pour les autres, un critère chacune. 2 : chaque statistique utilise-t-elle la valeur 40 elle-même, ou seulement sa place dans la liste triée ? 3 : compare chacun des trois résultats de la question 1 aux valeurs de la liste. 4 : avec des noms d'îles, peux-tu additionner ? trier ? compter ? 5 : combien de sommets a la courbe en cloche, et quelle symétrie ? 6 : demande-toi ce qu'une distribution très asymétrique ou des valeurs extrêmes changent à ta façon de résumer et de préparer les données.

</details>

### 2.Q2 — Quand a-t-on le droit de parler de probabilités ?

<details><summary>Indice 1</summary>

Deux conditions pour une distribution discrète : des nombres entre 0 et 1, et une somme égale à 1. Pour une loi continue, ce sont les **aires** qui jouent ce rôle.

</details>
<details><summary>Indice 2</summary>

Additionne les nombres des questions 1 et 2. Pour les questions 3 et 4, relis l'encadré 🧮 « densité » de la fiche §2.2 et la figure de droite de la §2.3.2.

</details>
<details><summary>Indice 3</summary>

Affirmation 1 : $12 + 30 + 8 = 50$, pas 1 ; des comptages ne deviennent des probabilités qu'une fois divisés par leur total, et l'affirmation est donc fausse. Pour les autres, un critère chacune. 2 : fais la même addition. 3 : quelle est l'aire sous une courbe au-dessus d'un intervalle de longueur 0 ? 4 : une densité est une hauteur, pas une aire ; quelle hauteur faut-il à une loi uniforme sur $[0 ; 0{,}5]$ pour que son aire vaille 1 ? 5 et 6 : compare mot pour mot avec les définitions de la fiche §2.2 (la liste « discrète / continue » et l'encadré 🧮 sur la densité).

</details>

### 2.Q3 — Graine et pseudo-aléatoire : vrai ou faux

<details><summary>Indice 1</summary>

Un générateur pseudo-aléatoire est un programme : mêmes entrées, même sortie. Quelle est son « entrée » ?

</details>
<details><summary>Indice 2</summary>

Pour chaque affirmation, demande-toi ce qui se passe quand on relance le programme, avec ou sans graine. Pour la 6, relis la fin du premier paragraphe de la fiche §2.2.1.

</details>
<details><summary>Indice 3</summary>

Une graine fixe le point de départ d'un calcul déterministe : elle rend la suite reproductible, rien de plus. L'affirmation 1 est donc vraie ; juge les cinq autres avec cette seule phrase (pour la 6, demande-toi ce que ferait un attaquant d'un calcul dont il devinerait le point de départ).

</details>

### 2.Q4 — Loi uniforme sur [0, 1] : questions pièges

<details><summary>Indice 1</summary>

Pour une loi uniforme, une probabilité est une **longueur** d'intervalle divisée par la longueur totale (fiche §2.3.1).

</details>
<details><summary>Indice 2</summary>

Une valeur exacte est un intervalle de longueur 0. L'aire sous la densité doit valoir 1 : quelle hauteur faut-il sur un intervalle de longueur 1 ? de longueur 2 ? Pour `rng.random()` et `rng.integers`, relis la fiche : quelles bornes sont exclues ?

</details>
<details><summary>Indice 3</summary>

Question 1 : $P(X \le 0{,}3)$ est la longueur de $[0 ; 0{,}3]$ divisée par la longueur totale, 1, soit 0,3. 2 et 3 : le même calcul (pour 3, quelle est la longueur de l'intervalle $[0{,}5 ; 0{,}5]$ ?). 4 : hauteur × longueur = 1, donc hauteur = $\frac{1}{\text{longueur}}$, pour une longueur de 1, puis de 2. 5 : la moyenne d'une loi uniforme sur $[a, b]$ est $\frac{a + b}{2}$ (fiche §2.3.1), ici avec $a = 0$ et $b = 1$. 6 : la notation $[0, 1)$ veut dire « 0 compris, 1 exclu » ; et `rng.integers(low, high)` suit la convention de `range(low, high)` : `list(range(1, 7))` contient-il 7 ?

</details>

### 2.Q5 — La règle 68-95-99,7

<details><summary>Indice 1</summary>

Exprime chaque borne en nombre d'écarts-types au-dessus ou au-dessous de 170 cm.

</details>
<details><summary>Indice 2</summary>

163 et 177, c'est 170 ± 7 ; 156 et 184, c'est 170 ± 2 × 7 ; 149, c'est 170 − 3 × 7. Pour « plus de » ou « moins de », pense à la symétrie : ce qui reste hors de l'intervalle se partage à parts égales des deux côtés.

</details>
<details><summary>Indice 3</summary>

Question 1 : 163 et 177 cm, c'est $170 \pm 7$, soit ± 1 écart-type, donc environ 68 % des adultes. 2 : combien d'écarts-types de part et d'autre de 170 ? 3 : ce qui sort de l'intervalle de la question 2 se partage à parts égales des deux côtés, soit $\frac{100\,\% - \text{(réponse 2)}}{2}$. 4 : le même calcul avec l'intervalle à ± 3 écarts-types. 5 : la fiche présente la règle comme une propriété de quelle loi (§2.3.2 et son encadré ⚠️) ? 6 : l'écart-type est la racine d'une moyenne d'écarts **au carré** ; quelle unité pour chacun des deux ?

</details>

### 2.Q6 — Bernoulli ou multinoulli ?

<details><summary>Indice 1</summary>

Compte les issues possibles : deux (Bernoulli), un nombre fini supérieur à deux (catégorielle), ou une infinité de valeurs réelles (ni l'une ni l'autre).

</details>
<details><summary>Indice 2</summary>

Pour chaque grandeur, écris la liste de ses valeurs possibles : est-elle finie ? Pour la question 7, souviens-toi que les probabilités doivent avoir une somme égale à 1.

</details>
<details><summary>Indice 3</summary>

Question 1 : pile ou face n'a que deux issues ; c'est une loi de Bernoulli. 2 à 6 : fais la liste des valeurs possibles et compte-les ; deux, un nombre fini plus grand que deux, ou une infinité de valeurs réelles ? (Pour 5 : une masse peut-elle valoir 4 012,37 g ?) 7 : écris les $K$ probabilités et la contrainte qui les relie ; combien sont vraiment libres ?

</details>

### 2.Q7 — Une espérance qu'on ne tire jamais

<details><summary>Indice 1</summary>

L'espérance est une moyenne pondérée par les probabilités (fiche §2.3.5, 0B 101.7.3). Une moyenne tombe souvent « entre » les valeurs.

</details>
<details><summary>Indice 2</summary>

Pour le dé à 4 faces, les quatre probabilités valent $\frac{1}{4}$. Pour la question 4, utilise la linéarité de l'espérance vue en 0B : $\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b$.

</details>
<details><summary>Indice 3</summary>

Question 1 : $\mathbb{E}[X] = \frac{2 + 4 + 6 + 8}{4} = 5$, qui n'est pas une face : on ne peut pas la tirer. 2 : le même calcul, $0 \times 0{,}7 + 1 \times 0{,}3$, puis compare aux valeurs que la variable peut prendre. 3 : relis tes réponses 1 et 2 avant de trancher. 4 : par linéarité, $2\,\mathbb{E}[X] + 1$, avec $\mathbb{E}[X] = 5$. 5 : relis 0B (101.7.5) ; vers quel nombre tend la moyenne de tirages de plus en plus nombreux, et quel nom porte ce résultat ?

</details>

### 2.Q8 — Dépendant, indépendant, i.i.d.

<details><summary>Indice 1</summary>

Pour chaque situation, demande-toi : connaître la première valeur me renseigne-t-il sur la seconde ?

</details>
<details><summary>Indice 2</summary>

Pour les situations 2 à 4, cherche si quelque chose relie les deux grandeurs : le climat ? la taille de l'animal ? l'espèce ? Pour la dernière situation, relis la fiche §2.4.1 : qu'est-ce qui peut casser l'hypothèse i.i.d. entre entraînement et test ?

</details>
<details><summary>Indice 3</summary>

Situation 1 : le premier lancer ne dit rien du second (indépendants), et c'est le même dé (même loi) ; les lancers successifs sont i.i.d. Pour les autres, pose la même question : connaître la première valeur change-t-il ce que tu attends de la seconde ? Pense, pour 2, au temps qu'il fait d'un jour au suivant ; pour 3, à la taille de l'oiseau ; pour 4, à ce que la fiche §2.4 dit des nageoires selon l'espèce. Pour 2, 5 et 6, ajoute le critère i.i.d. : chaque valeur est-elle tirée dans la même loi, sans dépendre des autres ? En 6, demande-toi ce que le jeu de test « sait » déjà du jeu d'entraînement.

</details>

### 2.Q9 — Avec ou sans remise ?

<details><summary>Indice 1</summary>

Pour chaque situation : un élément déjà tiré peut-il ressortir ?

</details>
<details><summary>Indice 2</summary>

Relis le tableau de la fiche §2.5. Pour le découpage, demande-toi si un même exemple peut se retrouver à la fois dans le jeu d'entraînement et dans le jeu de test. Pour `rng.choice`, relis l'exemple de code de la fiche : quelle est la valeur par défaut de `replace` ?

</details>
<details><summary>Indice 3</summary>

Situation 1 : une boule de loterie tirée ne retourne pas dans l'urne ; le tirage est donc **sans** remise. Pour les autres, pose la même question : un élément déjà tiré peut-il ressortir ? 2 : relis la première étape du bootstrap (fiche §2.6). 3 : un exemple peut-il être à la fois dans le jeu d'entraînement et dans le jeu de test ? 4 : combien de fois chaque exemple passe-t-il dans une epoch ? 5 : une boisson commandée disparaît-elle de la carte ? 6 : `rng.choice(a, size=5)` prend la valeur par défaut de `replace` ; lis-la dans la documentation (`help(rng.choice)`). 7 : sans remise, que reste-t-il dans l'urne après $n$ tirages parmi $n$ ?

</details>

### 2.Q10 — Ce que le bootstrap estime, et ce qu'il n'invente pas

<details><summary>Indice 1</summary>

Le bootstrap ne fait que retirer au hasard des éléments de l'échantillon qu'on a déjà. Peut-il savoir ce qui n'y est pas ?

</details>
<details><summary>Indice 2</summary>

Relis la fiche §2.6 (« Ce que le bootstrap ne fait pas ») et l'encadré 🕰️ : que change le **nombre** de rééchantillons, et que change leur **taille** ? Pour la question 6, relis la fin de la §2.5.3.

</details>
<details><summary>Indice 3</summary>

Affirmation 1 : le bootstrap ne fait que retirer des éléments déjà présents dans l'échantillon ; il ne crée aucune donnée, l'affirmation est donc fausse. Pour les autres, un critère chacune. 2 : que mesure la dispersion des valeurs calculées sur les rééchantillons (fiche §2.6, le paragraphe sur l'erreur type) ? 3 : relis l'encadré 🕰️ de la fiche §2.6 sur la **taille** des rééchantillons. 4 : un rééchantillon peut-il contenir quelqu'un qui n'était pas dans l'échantillon ? 5 : l'intervalle est fait de deux percentiles de la distribution bootstrap ; plus de rééchantillons dessinent cette distribution plus finement, mais la rendent-ils moins étalée ? Son étalement dépend de la dispersion de la statistique d'un échantillon à l'autre, donc de quoi ? 6 : calcule $1 - \left(1 - \frac{1}{n}\right)^n$ pour un grand $n$ (fiche §2.5.3).

</details>

### 2.Q11 — Une image est un point dans un espace à 784 dimensions

<details><summary>Indice 1</summary>

Une coordonnée par nombre qui décrit l'image : combien de pixels, combien de nombres par pixel ?

</details>
<details><summary>Indice 2</summary>

Pour la question 2, compte trois nombres par pixel. Pour la question 3, demande-toi combien de coordonnées diffèrent entre les deux images. Pour la question 6, relis la fin de la fiche §2.7.

</details>
<details><summary>Indice 3</summary>

Question 1 : une coordonnée par pixel, $28 \times 28 = 784$. 2 : le même calcul avec trois nombres par pixel, $64 \times 64 \times 3$. 3 : dans $\sqrt{\sum_j (a_j - b_j)^2}$, combien de termes sont non nuls, et sont-ils grands ? 4 : combien d'axes perpendiculaires peux-tu tracer sur une feuille, puis dans l'espace ? 5 : cette formule dépend-elle du nombre de termes ? 6 : avec seulement 10 valeurs possibles par axe, combien de cases faut-il pour quadriller $d$ dimensions ? Compare avec le nombre d'exemples d'un dataset.

</details>

### 2.Q12 — Covariance, corrélation et quartet d'Anscombe

<details><summary>Indice 1</summary>

La covariance est une moyenne de **produits** d'écarts ; la corrélation divise la covariance par deux écarts-types.

</details>
<details><summary>Indice 2</summary>

L'unité d'un produit est le produit des unités ; dans un quotient, des unités identiques se simplifient. Pour la question 4, relis l'encadré ⚠️ « Deux contresens à éviter » de la fiche §2.8.2.

</details>
<details><summary>Indice 3</summary>

Question 1 : la covariance est une moyenne de produits (écart en mm) × (écart en g) ; son unité est le mm·g. 2 : divise cette unité par celles des deux écarts-types (des mm et des g) ; que reste-t-il ? 3 : relis la fiche §2.8.2 ; quelle valeur prend $r$ quand tous les points sont sur une droite, selon le sens de la pente ? 4 : une corrélation nulle exclut quelle **forme** de lien seulement (encadré ⚠️ de la §2.8.2) ? 5 : cherche une troisième variable qui ferait monter les deux grandeurs en même temps. 6 : quatre jeux, mêmes résumés chiffrés ; que révèlent leurs quatre nuages ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 2.R1 — Ch. 1 : supervisé ou non supervisé, quatre tâches sur Penguins

<details><summary>Indice 1</summary>

Supervisé : on a la bonne réponse (le label) pour chaque exemple d'entraînement. Non supervisé : on n'utilise aucun label.

</details>
<details><summary>Indice 2</summary>

Une catégorie à prédire : classification ; une quantité : régression. Sans label : regrouper (clustering) ou résumer avec moins de nombres (réduction de dimension). Relis la fiche du ch. 1, §1.3 et §1.4.

</details>
<details><summary>Indice 3</summary>

Tâche 1 : pour chaque manchot d'entraînement, on connaît son espèce (le label), et l'espèce est une catégorie ; c'est donc de l'apprentissage supervisé, une classification. Pour les trois autres tâches, pose les mêmes questions : y a-t-il un label à prédire ? Si oui, est-ce une catégorie ou une quantité ? Sinon, cherche-t-on des groupes, ou moins de nombres par manchot ?

</details>

### 2.R2 — 0A : NumPy, moyenne par colonne avec `axis`

<details><summary>Indice 1</summary>

`axis=0` fait « disparaître » l'axe des lignes : on moyenne **verticalement**, une valeur par colonne.

</details>
<details><summary>Indice 2</summary>

La forme de `X` est (nombre de lignes, nombre de colonnes). `axis=0` donne un résultat de la longueur d'une ligne, `axis=1` un résultat de la longueur d'une colonne ; sans `axis`, toutes les valeurs sont moyennées ensemble.

</details>
<details><summary>Indice 3</summary>

`X.shape` vaut `(3, 2)` ; `X.mean(axis=0)` donne 2 valeurs (la moyenne de 1, 2, 6, puis celle de 10, 20, 60) ; `X.mean(axis=1)` donne 3 valeurs ; `X.mean()` divise la somme 99 par 6.

</details>

### 2.R3 — 0B : espérance et variance d'un dé équilibré

<details><summary>Indice 1</summary>

Chaque face a la probabilité $\frac{1}{8}$. Espérance : la moyenne des faces pondérée par leurs probabilités.

</details>
<details><summary>Indice 2</summary>

$\mathbb{E}[X^2]$ est la moyenne des **carrés** des faces : $\frac{1 + 4 + 9 + \ldots + 64}{8}$. Puis $\mathrm{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ (0B, 101.7.4). Pour deux dés : $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$.

</details>
<details><summary>Indice 3</summary>

Question 1 : $\mathbb{E}[X] = \frac{1 + 2 + \ldots + 8}{8} = \frac{36}{8} = 4{,}5$. 2 : $\mathbb{E}[X^2] = \frac{1^2 + 2^2 + \ldots + 8^2}{8}$ ; additionne les huit carrés, puis divise. 3 : $\mathrm{Var}(X) = \mathbb{E}[X^2] - 4{,}5^2$, et l'écart-type est sa racine. 4 : $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$, avec deux dés identiques.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 2.1 — Moyenne, médiane et mode d'une liste de salaires ✏️

<details><summary>Indice 1</summary>

Commence par **trier** les neuf salaires du plus petit au plus grand : la médiane et le mode se lisent alors directement.

</details>
<details><summary>Indice 2</summary>

Moyenne : somme des 9 salaires divisée par 9. Médiane : la 5ᵉ valeur de la liste triée (4 en dessous, 4 au-dessus). Mode : la seule valeur qui apparaît deux fois. Pour e et f, seule la plus grande valeur change : qu'est-ce que ça change à la somme ? à la valeur du milieu ?

</details>
<details><summary>Indice 3</summary>

a) La somme des neuf salaires vaut 31 700 €, donc la moyenne vaut $\frac{31\,700}{9} \approx 3\,522{,}2$, soit 3 522 €. Pour les autres : b) et c) trie d'abord les neuf salaires ; la médiane est la 5ᵉ valeur de la liste triée, le mode la seule valeur qui y figure deux fois ; d) compare chaque salaire à la moyenne de a ; e) avec 30 000 € au lieu de 12 000 €, la somme augmente de 18 000 € : divise la nouvelle somme par 9 ; f) dans la liste triée, le salaire de la directrice change-t-il de place ?

</details>

### Ex 2.2 — De la casse de voitures à la distribution de probabilité ✏️

<details><summary>Indice 1</summary>

Une probabilité, ici, c'est un comptage divisé par le nombre total de voitures (fiche §2.2).

</details>
<details><summary>Indice 2</summary>

Un tour de roue fait 360° : la part d'un type est sa probabilité × 360°. « Ne pas tirer un SUV » est l'événement contraire de « tirer un SUV ». Sur 50 tirages avec remise, on attend 50 × la probabilité. Les sommes cumulées s'obtiennent en additionnant les probabilités au fur et à mesure ; le type choisi par $u$ est le premier dont la somme cumulée est strictement plus grande que $u$ (fiche §2.3.4).

</details>
<details><summary>Indice 3</summary>

a) $\frac{224}{800} = 0{,}28$. Pour les autres : b) la même division pour chaque type, dans l'ordre de l'énoncé : $\frac{176}{800}$, $\frac{144}{800}$… ; c) la probabilité de a × 360° ; d) $1 - P(\text{SUV})$ ; e) $50 \times P(\text{pick-up})$ ; f) additionne les probabilités de b au fur et à mesure : $p_1$, puis $p_1 + p_2$, puis $p_1 + p_2 + p_3$… ; g) cherche le **premier** type dont la somme cumulée de f est strictement plus grande que 0,71.

</details>

### Ex 2.3 — La règle 68-95-99,7 sur les nageoires des manchots ✏️

<details><summary>Indice 1</summary>

68 % : à moins d'**un** écart-type de la moyenne ; 95 % : à moins de deux (fiche §2.3.2).

</details>
<details><summary>Indice 2</summary>

Un intervalle à $k$ écarts-types va de $\mu - k\sigma$ à $\mu + k\sigma$. Exprime 203 mm en nombre d'écarts-types au-dessus de la moyenne, puis demande-toi où se trouvent les manchots qui sortent de l'intervalle correspondant. Le z-score est l'écart à la moyenne divisé par l'écart-type.

</details>
<details><summary>Indice 3</summary>

$190 \pm 6{,}5$ et $190 \pm 13$ ; la part au-dessus de 203 mm est la moitié de 5 % ; pour d, multiplie cette part par 151 et arrondis ; $z = \frac{210 - 190}{6{,}5}$.

</details>

### Ex 2.4 — Variance : diviser par N ou par N − 1 ? ✏️

<details><summary>Indice 1</summary>

Fais un tableau à trois colonnes : la valeur, son écart à la moyenne, le carré de cet écart (fiche §2.3.2).

</details>
<details><summary>Indice 2</summary>

La somme des carrés se divise par $n = 5$ (ddof = 0) ou par $n - 1 = 4$ (ddof = 1) ; l'écart-type est la racine de la variance. Pour h et i, relis la règle de 0B : $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$, donc l'écart-type est multiplié par $|a|$.

</details>
<details><summary>Indice 3</summary>

a) $\frac{4 + 7 + 6 + 3 + 10}{5} = 6$. Pour les autres : b) les écarts à 6 sont −2, 1, 0, −3, 4 : additionne leurs **carrés** ; c) et d) divise la somme de b par 5, puis par 4 ; e) et f) prends la racine de c, puis de d ; g) $\frac{16 + 49 + 36 + 9 + 100}{5}$, puis retranche $6^2$ pour retrouver c ; h) ajouter 100 à chaque mesure change-t-il les écarts à la moyenne ? i) multiplie par 1 000 l'écart-type **exact** de e (la racine de c, pas son arrondi) ; j) cherches-tu la dispersion de ces cinq mesures, ou celle de tous les chargements (fiche §2.3.2, 🧮 pourquoi $n - 1$) ?

</details>

### Ex 2.5 — Espérances : Bernoulli, multinoulli et jeu de hasard ✏️

<details><summary>Indice 1</summary>

Toutes les questions se ramènent à $\mathbb{E}[X] = \sum_k x_k\,p_k$ (fiche §2.3.5), sauf b (la variance d'une loi de Bernoulli, fiche §2.3.3) et f (un codage one-hot, fiche §2.3.4).

</details>
<details><summary>Indice 2</summary>

Pour c, le nombre moyen de faces sur 200 lancers vaut 200 × l'espérance d'un lancer. Pour g, regarde une seule composante du vecteur one-hot : c'est une variable de Bernoulli ; quelle est son espérance ? Pour h, écris le tableau des gains nets possibles (en tenant compte de la mise) et de leurs probabilités.

</details>
<details><summary>Indice 3</summary>

Pour h, les gains nets sont 18 € (probabilité $\frac{1}{20}$), 3 € (probabilité $\frac{3}{20}$) et −2 € (probabilité $\frac{16}{20}$). Pour g, la composante n° $k$ vaut 1 avec la probabilité de la classe $k$.

</details>

### Ex 2.6 — Compter les tirages avec et sans remise ✏️

<details><summary>Indice 1</summary>

Avec cinq exemples, tu peux encore **écrire toutes les possibilités** à la main ; fais-le pour a, b et c avant de chercher une formule.

</details>
<details><summary>Indice 2</summary>

Suites avec remise : $n^k$. Sous-ensembles sans remise : $\binom{n}{k}$ (0B, 101.1.7). Pour f, un exemple donné n'est pas tiré à un tirage avec probabilité $1 - \frac{1}{n}$, et les tirages sont indépendants (fiche §2.5.3).

</details>
<details><summary>Indice 3</summary>

a : AB, AC, AD, AE, BC, … (continue la liste sans compter BA, déjà là sous la forme AB). b : ajoute les paires répétées AA, BB… f : $\left(\frac{4}{5}\right)^5$ ; g : $0{,}999^{1000}$ (pas $e^{-1}$, qui n'est que la limite) ; h : le complément de g.

</details>

### Ex 2.7 — Covariance et corrélation de cinq points à la main ✏️

<details><summary>Indice 1</summary>

Même méthode qu'en 2.4, avec deux colonnes d'écarts et une colonne de **produits** d'écarts.

</details>
<details><summary>Indice 2</summary>

Covariance : somme des produits des écarts divisée par $n$ (ddof = 0). Écart-type : racine de la somme des carrés des écarts divisée par $n$. $r$ : covariance divisée par le produit des deux écarts-types, avec le même ddof partout. Pour g et h, utilise ce que dit la fiche §2.8 sur le changement d'unité (ou démontre-le avec 2.8).

</details>
<details><summary>Indice 3</summary>

a) $\bar{x} = \frac{1 + 2 + 3 + 4 + 5}{5} = 3$ et $\bar{y} = \frac{2 + 3 + 5 + 4 + 6}{5} = 4$, soit [3 ; 4]. Pour les autres : b) les écarts de $x$ sont −2, −1, 0, 1, 2 et ceux de $y$ −2, −1, 1, 0, 2 : multiplie-les deux à deux, puis additionne ; c) et d) divise la somme de b par 5, puis par 4 ; e) $\sigma_x = \sqrt{\frac{1}{5}\sum_i (x_i - \bar{x})^2}$, et de même pour $y$ ; f) $r = \frac{\mathrm{Cov}(x, y)}{\sigma_x\,\sigma_y}$, avec la covariance de c et les écarts-types de e (le même ddof partout) ; g) et h) applique la règle de la fiche §2.8 (ou de 2.8) : $x$ est multiplié par 60, $y$ ne change pas.

</details>

### Ex 2.8 — Changer d'unité : la covariance bouge, pas la corrélation ∂

<details><summary>Indice 1</summary>

Tout repose sur une propriété de Σ (0B, 101.1.3) : $\sum_i (a\,x_i + b) = a\sum_i x_i + n\,b$.

</details>
<details><summary>Indice 2</summary>

Divise l'égalité précédente par $n$ pour la question 1. Remplace ensuite $u_i$ et $\bar{u}$ par leurs expressions : le $b$ disparaît. Dans la covariance, sors les constantes $a$ et $c$ de la somme. Pour la question 4, applique la question 3 avec $v = u$, puis prends la racine carrée : attention au signe de $a$.

</details>
<details><summary>Indice 3</summary>

Question 5 : $r(u, v) = \frac{a\,c\,\mathrm{Cov}(x, y)}{|a|\,\sigma_x\,|c|\,\sigma_y}$, et $\frac{a\,c}{|a|\,|c|}$ vaut +1 ou −1 selon les signes. Question 7 : le $\frac{1}{n}$ du numérateur et ceux des deux écarts-types se simplifient ; il reste un produit scalaire divisé par deux normes.

</details>

<a id="reflexion"></a>

## 🧮 🗣️ ⚖️ 📄 Réflexion

### Ex 2.9 — Le bootstrap en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Commence par la **question** à laquelle répond le bootstrap (« à quel point ce chiffre est-il fiable ? »), pas par la méthode.

</details>
<details><summary>Indice 2</summary>

Choisis un exemple de la vie courante : la note moyenne d'un restaurant sur 30 avis, le temps moyen d'attente à un guichet sur 25 clients… Puis déroule les étapes : retirer au hasard, avec remise, autant d'éléments que l'échantillon ; recalculer ; recommencer ; regarder l'étendue des résultats.

</details>
<details><summary>Indice 3</summary>

Une structure possible : (1) la question ; (2) on n'a qu'un échantillon ; (3) on fabrique des « échantillons de rechange » en retirant avec remise ; (4) on recalcule la moyenne sur chacun ; (5) les 95 % centraux de ces moyennes forment l'intervalle.

</details>

### Ex 2.10 — Corrélation, causalité et échantillon biaisé ⚖️

<details><summary>Indice 1</summary>

Pour chaque situation, cherche une **troisième variable** qui pourrait expliquer le lien, ou une raison pour laquelle l'échantillon ne ressemble pas à la population visée.

</details>
<details><summary>Indice 2</summary>

A : qu'est-ce qui fait qu'un quartier a beaucoup d'incendies **et** beaucoup d'interventions ? Et dans quel sens va la causalité ? B : qui utilise une application de sport, et qui porte une montre la nuit ? Relis « Ce que le bootstrap ne fait pas » (fiche §2.6). C : qui pratiquait le rugby parmi les candidats embauchés il y a dix ans ? Que peut « représenter » cette ligne du CV ?

</details>
<details><summary>Indice 3</summary>

Question 1 (A) : une variable de confusion, la taille ou la densité du quartier (plus d'habitants et de bâtiments, donc plus d'incendies **et** plus d'interventions) ; et une causalité inversée, puisque ce sont les incendies qui font venir les pompiers. 2 : l'intervalle bootstrap mesure le hasard du tirage de **quel** échantillon, dans **quelle** population ? Les utilisateurs d'une application de sport, montre au poignet la nuit, ressemblent-ils aux Français ? Avec $n = 50\,000$, que vaut à peu près $\frac{1}{\sqrt{n}}$ ? Que peut-on alors affirmer honnêtement, et sur qui ? 3 : à partir de quoi le modèle a-t-il appris ce qu'est un « bon » CV ? Qui pratiquait le rugby en club parmi les embauchés d'il y a dix ans, et qui en était le plus souvent absent ? 4 : pense à ce que tu mesurerais sur le modèle, à ce que tu retirerais des données, à qui décide en dernier et à ce que tu écrirais ; puis cherche dans quelle catégorie de risque l'AI Act range le recrutement.

</details>

### Ex 2.11 — Fermi : la taille de l'espace des images 🧮

<details><summary>Indice 1</summary>

Le nombre d'images possibles, c'est le nombre de valeurs possibles d'un pixel, élevé à la puissance du nombre de pixels (0B, principe multiplicatif).

</details>
<details><summary>Indice 2</summary>

Pour passer d'une puissance de 2 à une puissance de 10 : $2^{10} \approx 10^3$, donc $2^{k} \approx 10^{0{,}3\,k}$. Pour la mémoire : nombre de valeurs × octets par valeur ; 1 Mo ≈ $10^6$ octets.

</details>
<details><summary>Indice 3</summary>

Question 1 : une image de MNIST a $28 \times 28 = 784$ dimensions, la photo $4\,000 \times 3\,000 \times 3 = 3{,}6 \times 10^7$. 2 : $2^{784}$ ; écris $784 = 78 \times 10 + 4$ pour utiliser $2^{10} \approx 10^3$, ou $2^k = 10^{k \log_{10} 2}$ avec $\log_{10} 2 \approx 0{,}301$. 3 : $256^{784} = (2^8)^{784} = 2^{8 \times 784}$, puis la même méthode. 4 : compare l'exposant de 3 à 80, puis divise $7 \times 10^4$ par le résultat de 3 en soustrayant les exposants. 5 : 784 nombres × 4 octets pour une image, puis × 70 000 pour le dataset (1 Mo ≈ $10^6$ octets), et de même avec 1 octet par nombre. 6 : pense à ce qu'est une image « au hasard », chaque pixel tiré séparément (du bruit), comparée à un vrai chiffre écrit à la main.

</details>

### Ex 2.12 — Anscombe (1973) : regarder avant de calculer 📄

<details><summary>Indice 1</summary>

L'article s'ouvre sur les idées reçues qu'il veut réfuter ; lis surtout son premier paragraphe et regarde ses quatre graphiques.

</details>
<details><summary>Indice 2</summary>

Pour la question 2, fais le tableau des écarts à la moyenne comme en 2.4. Pour la question 4, que vaut l'écart-type de $x$ si toutes les valeurs de $x$ sont égales ? Et une division par 0 ?

</details>
<details><summary>Indice 3</summary>

Question 1 : Anscombe s'attaque à l'idée reçue selon laquelle les calculs sont exacts et les graphiques approximatifs, si bien qu'un bon statisticien calculerait plutôt qu'il ne regarderait. 2 : $\bar{x} = \frac{10 + 8 + 13 + 9 + 11 + 14 + 6 + 4 + 12 + 7 + 5}{11}$, puis le tableau des écarts à cette moyenne et de leurs carrés ; divise la somme des carrés par 11 (ddof = 0), puis par 10 (ddof = 1), et compare avec 11 et avec $3{,}16^2$. 3 : décris d'abord la forme générale des points, puis ce qui s'en écarte ; une droite résume bien un nuage quand les écarts des points à la droite ne dessinent aucune forme. 4 : sans ce point, les dix valeurs de $x$ valent 8 ; calcule $\sigma_x$, puis regarde le dénominateur de $r = \frac{\mathrm{Cov}(x, y)}{\sigma_x\,\sigma_y}$. 5 : relis la fin de la fiche §2.9 et son encadré 🕰️.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 2.E1 — Moyenne ou médiane pour résumer des salaires ?

<details><summary>Indice 1</summary>

Pense à la forme de la distribution des salaires : symétrique ou penchée d'un côté ?

</details>
<details><summary>Indice 2</summary>

Reprends le résultat de 2.1 : une seule valeur très élevée tire la moyenne. Quand la moyenne reste-t-elle utile (masse salariale totale, budget) ?

</details>
<details><summary>Indice 3</summary>

Plan de réponse : la médiane pour « le salaire typique » (robuste aux valeurs extrêmes) ; la moyenne pour les totaux ; idéalement les deux, plus des quartiles ou un histogramme.

</details>

### 2.E2 — i.i.d. : définition et pourquoi le ML en a besoin

<details><summary>Indice 1</summary>

Décompose le sigle : *independent* (indépendantes) and *identically distributed* (de même loi).

</details>
<details><summary>Indice 2</summary>

Relie-le au jeu de test du ch. 1 : pourquoi un score de test prédit-il les performances futures ? Pour les contre-exemples, relis la liste de la fiche §2.4.1.

</details>
<details><summary>Indice 3</summary>

Plan : définition (deux mots) ; pourquoi (le test doit ressembler aux données futures, les exemples ne doivent pas se « copier ») ; un contre-exemple concret (série temporelle, patients vus plusieurs fois) et sa parade (découpage dans le temps, par groupe).

</details>

### 2.E3 — Expliquer un intervalle de confiance bootstrap

<details><summary>Indice 1</summary>

Les 400 exemples de test sont un échantillon : avec 400 autres exemples, l'accuracy aurait été un peu différente. De combien ?

</details>
<details><summary>Indice 2</summary>

Rééchantillonne les 400 prédictions (justes ou fausses) avec remise, recalcule l'accuracy, recommence 1 000 fois, garde les 95 % centraux. Pour le chef de projet : une phrase sans jargon et un ordre de grandeur.

</details>
<details><summary>Indice 3</summary>

Pour l'ordre de grandeur, l'écart-type d'une proportion vaut à peu près $\sqrt{p(1 - p)/n}$ : calcule-le avec $p = 0{,}87$ et $n = 400$, puis compte environ deux de ces écarts-types de chaque côté de 87 % pour un intervalle à 95 %. Mentionne aussi ce que l'intervalle ne couvre pas (un jeu de test non représentatif).

</details>

### 2.E4 — Corrélation nulle veut-elle dire indépendance ?

<details><summary>Indice 1</summary>

La corrélation de Pearson ne mesure qu'une forme de lien. Laquelle ?

</details>
<details><summary>Indice 2</summary>

Pense à un nuage en U ou en cercle (fiche §2.8.2), puis à la variable de confusion (glaces et noyades).

</details>
<details><summary>Indice 3</summary>

Plan : indépendance ⇒ corrélation nulle, mais pas l'inverse (exemple de la parabole) ; forte corrélation ⇏ causalité (confusion, causalité inversée, coïncidence) ; une expérience contrôlée est le moyen le plus sûr d'établir une cause ; toujours tracer le nuage.

</details>

### 2.E5 — Pourquoi fixer la graine aléatoire d'une expérience ?

<details><summary>Indice 1</summary>

Qu'est-ce qui est tiré au hasard dans un entraînement de réseau de neurones ? Fais la liste.

</details>
<details><summary>Indice 2</summary>

La graine sert à reproduire (déboguer, comparer deux réglages à hasard égal). Mais un résultat obtenu avec une seule graine peut être un coup de chance : que faut-il faire pour annoncer une amélioration ?

</details>
<details><summary>Indice 3</summary>

Plan : ce qui est aléatoire (initialisation, mélange, découpage, dropout, augmentation) ; la graine rend l'expérience reproductible (NumPy, `random`, PyTorch ; attention au GPU, pas toujours déterministe) ; mais on répète sur plusieurs graines et on donne une moyenne et un écart-type (ch. 8).

</details>

<a id="notebook"></a>

## Notebook, parties A à D

Les exercices du notebook (`03_notebook.ipynb`). Pour ceux qui complètent `mylearn/stats.py`, la docstring de chaque fonction décrit déjà l'algorithme : relis-la avant d'ouvrir un indice.

### Ex 2.13 — Tendances centrales : `mean`, `median`, `mode` 🔨

<details><summary>Indice 1</summary>

Trois fonctions courtes. Commence par une petite fonction d'aide qui convertit l'entrée (`np.asarray(x, dtype=float)`) et refuse un tableau vide ou qui contient des NaN (`np.isnan(arr).any()`) : `mean`, `median` et bientôt `variance` l'appelleront. `mode`, elle, doit accepter des chaînes de caractères : pas de conversion en `float` dans `mode`.

</details>
<details><summary>Indice 2</summary>

`mean` : `arr.sum(axis=axis)` divisé par le nombre de valeurs réduites (`arr.size` si `axis` vaut `None`, sinon `arr.shape[axis]`) ; renvoie un `float` Python quand `axis` vaut `None`. `median` : trie, puis prends l'élément n° `n // 2` ; si `n` est pair, la moyenne des éléments n° `n // 2 - 1` et `n // 2`. `mode` : `values, counts = np.unique(arr, return_counts=True)` donne les valeurs triées et leurs comptes ; un masque booléen garde celles qui atteignent le maximum.

</details>
<details><summary>Indice 3</summary>

`mode` : la signature, le squelette, puis les deux lignes clés.
```python
def mode(x):
    # 1. arr = x as an array, WITHOUT dtype=float (mode also takes strings)
    # 2. refuse an empty array, or one that is not 1-D (ValueError)
    values, counts = np.unique(arr, return_counts=True)    # sorted values, and how often each one occurs
    return values[counts == counts.max()]                  # every value that reaches the top count (ties too)
```
Pour `median` le long d'un axe : `ordered = np.sort(arr, axis=axis)`, puis `np.take(ordered, n // 2, axis=axis)` (et la même chose avec `n // 2 - 1` si `n` est pair). Pour f : `with_typo = np.append(mass, 57_000)`, puis les différences des moyennes et des médianes.

</details>

### Ex 2.14 — Graine fixée ou graine libre ? 🔮

<details><summary>Indice 1</summary>

Un générateur est un calcul : son état de départ (la graine) fixe toute la suite des nombres qu'il produira, et chaque appel **avance** dans cette suite.

</details>
<details><summary>Indice 2</summary>

Pour chaque cas, demande-toi si les deux lignes partent du même état. a) Deux générateurs neufs, même graine. b) Un seul générateur, deux appels. c) et d) Deux générateurs sans graine : que règle `np.random.seed`, et `default_rng()` s'en sert-il ? e) La même graine, avec deux appels (3 puis 5 nombres) d'un côté et un seul appel (8 nombres) de l'autre. f) La même graine, mais la ligne ajoutée a-t-elle déjà pris des nombres dans la suite ?

</details>
<details><summary>Indice 3</summary>

Pour chaque cas, compare l'état de départ de chaque ligne et les nombres déjà consommés avant elle. a) Même graine, donc même état de départ : que produisent deux calculs identiques ? b) Le second appel repart de là où le premier s'est arrêté, pas de l'état initial. c) Sans graine, chaque création de générateur prend un nouvel état dans l'entropie du système (fiche §2.2.1). d) Relis l'encadré 🕰️ de la fiche §2.2.1 : l'ancienne interface (`np.random.seed`, `np.random.rand`) a son propre état global ; `default_rng()`, appelé sans argument, va-t-il le lire ? e) Aligne les nombres : de quel endroit de la suite viennent les 3 premiers de chaque ligne ? Et les 5 suivants ? f) Combien de nombres la ligne ajoutée a-t-elle consommés avant la permutation ?

</details>

### Ex 2.15 — Dispersion : `variance`, `std`, `percentile`, `zscore` 🔨

<details><summary>Indice 1</summary>

La variance réutilise la moyenne : écarts, carrés, somme, division par $n - \mathrm{ddof}$. L'écart-type est sa racine ; le z-score réutilise `mean` et `std`. Le percentile suit l'encadré 🧮 de la fiche : trier, calculer une position, interpoler.

</details>
<details><summary>Indice 2</summary>

`variance` : `dev = arr - arr.sum(axis=axis, keepdims=True) / n` (avec `keepdims=True`, la dimension réduite reste là, et la soustraction se fait ligne par ligne ou colonne par colonne), puis `(dev ** 2).sum(axis=axis) / (n - ddof)`. `percentile` : `position = q / 100 * (n - 1)`, `below = np.floor(position).astype(int)`, `above = np.minimum(below + 1, n - 1)`, puis `values[below] + (position - below) * (values[above] - values[below])`. `zscore` : avec `axis`, remets la dimension réduite (`np.expand_dims(m, axis)`) avant de soustraire et de diviser.

</details>
<details><summary>Indice 3</summary>

`percentile` avec un axe : trie le long de l'axe puis ramène-le en premier, `values = np.moveaxis(np.sort(arr, axis=axis), axis, 0)` ; `values[below]` a alors la forme `q.shape + (le reste)`, et `position - below` doit prendre la forme `q.shape + (1,) * (values.ndim - 1)` (`reshape`) avant la multiplication. d) `z = zscore(mass)`, puis `z[np.argmax(mass)]`. e) `np.sum(np.abs(z) > 2)`. f) `np.abs(zscore(X_measures, axis=0)).max()`. Pour la question des notes : avec `axis=None`, il n'y a qu'**une** moyenne et **un** écart-type pour tout le tableau, calculés sur des grammes et des millimètres mélangés.

</details>

### Ex 2.16 — Un histogramme fait maison 🔨

<details><summary>Indice 1</summary>

Trois étapes : les bords (`np.linspace(low, high, bins + 1)`), le numéro d'intervalle de chaque valeur, puis le comptage (`np.bincount`). Seul le dernier intervalle garde son bord droit.

</details>
<details><summary>Indice 2</summary>

Garde d'abord les valeurs de `[low, high]` (un masque booléen). `np.searchsorted(edges, inside, side="right") - 1` donne, pour chaque valeur, le numéro du dernier bord inférieur ou égal à elle ; une valeur égale à `high` obtient `bins` : ramène-la à `bins - 1`. `np.bincount(index, minlength=bins)` compte chaque intervalle, même vide. Densité : `counts / (counts.sum() * np.diff(edges))`.

</details>
<details><summary>Indice 3</summary>

`histogram` : la signature, le squelette, puis les deux lignes clés.
```python
def histogram(x, bins=10, bin_range=None, density=False):
    # 1. convert and check (bins >= 1, low < high); without bin_range, low and high are the min and max of x
    # 2. edges = np.linspace(low, high, bins + 1), and inside = the values of x within [low, high] (kept with a boolean mask)
    index = np.searchsorted(edges, inside, side="right") - 1
    index[inside == high] = bins - 1          # the last bin keeps its right edge
    # 3. count with np.bincount(index, minlength=bins); with density=True, divide as in hint 2
```
Pour la question 📝 : quelle largeur fait un intervalle quand il y en a 4, puis 100 ? Les nageoires sont mesurées au millimètre près : combien de valeurs possibles tombent dans un intervalle de 0,6 mm ?

</details>

### Ex 2.17 — Galerie des lois usuelles 🎨

<details><summary>Indice 1</summary>

Deux fonctions d'une ligne, puis quatre panneaux indépendants : écris-les un par un, et exécute après chacun. Avec `fig, axes = plt.subplots(2, 2)`, `axes[0, 1]` est le panneau en haut à droite.

</details>
<details><summary>Indice 2</summary>

`uniform_pdf` : `np.where((x >= a) & (x <= b), 1 / (b - a), 0.0)`. `normal_pdf` : traduis la formule de la fiche morceau par morceau (`np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))`, puis la division). Points du panneau 2 : `draws = rng.normal(mu, sigma, size=30)`, et des hauteurs négatives un peu aléatoires, une rangée par loi ; `line, = ax.plot(...)` puis `color=line.get_color()` donne la même couleur à la courbe et à ses points. Bernoulli : `rng.random(1000) < p` vaut `True` avec la probabilité `p`.

</details>
<details><summary>Indice 3</summary>

Des bâtons côte à côte : `ax.bar(positions - 0.2, probabilities, width=0.4)` puis `ax.bar(positions + 0.2, frequencies, width=0.4)` ; `ax.set_xticks(positions, CAR_TYPES)` écrit les noms sous les bâtons. Fréquences de 1 000 tirages catégoriels : `np.bincount(draws, minlength=5) / 1000`. Hauteurs des points : `-0.1 - 0.12 * k + rng.uniform(-0.04, 0.04, size=30)` pour la loi n° `k`.

</details>

### Ex 2.18 — 68-95-99,7 : la théorie face aux tirages et aux manchots 🔬

<details><summary>Indice 1</summary>

`share_within` tient en une ligne : un masque booléen « à moins de `k` écarts-types de la moyenne », puis la moyenne de ce masque (la proportion de `True`).

</details>
<details><summary>Indice 2</summary>

Un masque booléen « $|x_i - \bar{x}| < k\,\sigma$ » (écart-type de ddof = 0), puis la proportion de `True` : c'est sa moyenne. a) Les tirages : `np.random.default_rng(0).standard_normal(100_000)`, puis la liste des trois proportions. b) et d) : la même chose sur `adelie_flipper` et sur `flipper`. c) Un masque « strictement plus de 203 mm » et sa somme.

</details>
<details><summary>Indice 3</summary>

Le masque : `np.abs(x - x.mean()) < k * x.std()` (`x.std()` a ddof = 0 par défaut) ; la proportion de `True` est la moyenne de ce masque, à convertir en `float`. Pour a, b et d, une liste en compréhension sur `k` = 1, 2, 3 ; pour c, le masque de l'indice 2 et sa somme. Pour interpréter d, regarde l'histogramme de droite : où tombe la moyenne ? Que contient la bande de ± 1 écart-type ? Et celle de ± 3 ?

</details>

### Ex 2.19 — La roue de la fortune : tirer dans une distribution discrète 🔨

<details><summary>Indice 1</summary>

C'est la roue de 2.2 f et g : les sommes cumulées découpent $[0, 1)$ en segments, un par catégorie, et un nombre uniforme tombe dans l'un d'eux.

</details>
<details><summary>Indice 2</summary>

Vérifie `p` (une dimension, aucune valeur négative, `abs(p.sum() - 1) <= 1e-8`), crée le générateur s'il manque, puis `u = rng.random(size)` et `np.searchsorted(np.cumsum(p), u, side="right")`. Avec `side="right"`, un `u` égal à une somme cumulée va dans le segment suivant, comme en 2.2.

</details>
<details><summary>Indice 3</summary>

Après les contrôles de `p` et le `rng` par défaut (étapes 1 et 2 de l'énoncé), les deux lignes clés :
```python
k = np.minimum(np.searchsorted(np.cumsum(p), rng.random(size), side="right"), len(p) - 1)
return int(k) if size is None else k
```

</details>

### Ex 2.20 — Le pelage des animaux : une variable qui dépend d'une autre 🔮

<details><summary>Indice 1</summary>

a) est une espérance : la moyenne de chaque animal compte selon la probabilité de tirer cet animal (fiche §2.3.5, comme en 2.5 e).

</details>
<details><summary>Indice 2</summary>

b) Pour chaque animal, écris l'intervalle moyenne ± 3 écarts-types : s'ils ne se chevauchent pas, chaque animal fait sa propre bosse. c) La loi normale est symétrique autour de sa moyenne. Quels animaux peuvent dépasser 7 cm ? Avec quelle probabilité tire-t-on cet animal, et avec quelle probabilité son pelage dépasse-t-il sa moyenne ?

</details>
<details><summary>Indice 3</summary>

c) (probabilité de tirer un chien) × (probabilité qu'un pelage de chien dépasse sa moyenne). d) L'animal est connu : lequel des deux facteurs de c reste-t-il ? e) Après le mélange, le pelage attribué à un « hamster » est celui d'un animal tiré au hasard parmi **tous** : quelle est la moyenne d'un tel pelage ?

</details>

### Ex 2.21 — Tirer avec ou sans remise 🔨

<details><summary>Indice 1</summary>

On ne tire pas les éléments eux-mêmes, mais leurs **indices** (de 0 à $n - 1$), puis on les lit dans la population : `population[indices]`.

</details>
<details><summary>Indice 2</summary>

Avec remise : `rng.integers(0, n, size)` ; sans remise : `rng.permutation(n)[:size]`. Fais d'abord les contrôles : taille négative, population vide, `size > n` sans remise. `epoch_minibatches` : un ordre complet des indices, tiré sans remise, puis les tranches `order[start:start + batch_size]` pour `start` allant de 0 à `n_examples` par pas de `batch_size`.

</details>
<details><summary>Indice 3</summary>

`mean_share_distinct` : pour chaque rééchantillon, `len(np.unique(sample(np.arange(n), n, rng=rng))) / n`, puis la moyenne des `n_resamples` parts. Le **même** `rng` sert à tous les rééchantillons : il avance de l'un à l'autre, et c'est ce qui les rend différents.

</details>

### Ex 2.22 — Bootstrap : distribution et intervalle de confiance 🔨

<details><summary>Indice 1</summary>

`bootstrap_distribution` est une boucle de `n_boot` tours : des indices tirés avec remise, la statistique du rééchantillon, rangée dans un array. `bootstrap_ci` coupe les deux queues de cette distribution avec ta fonction `percentile`.

</details>
<details><summary>Indice 2</summary>

`size = n if sample_size is None else sample_size` ; `values = np.empty(n_boot)` ; dans la boucle, `idx = rng.integers(0, n, size=size)` et `values[b] = statistic(x[idx])`. Pour un intervalle à 95 %, les percentiles 2,5 et 97,5 ; à 80 %, 10 et 90 : c'est $50\,(1 - c)$ et $50\,(1 + c)$.

</details>
<details><summary>Indice 3</summary>

`bootstrap_ci` : la signature, le squelette, puis les deux lignes clés.
```python
def bootstrap_ci(x, statistic=np.mean, *, confidence=0.95, n_boot=1000, sample_size=None, rng=None):
    # 1. refuse a confidence outside ]0, 1[ (ValueError)
    values = bootstrap_distribution(x, statistic, n_boot=n_boot, sample_size=sample_size, rng=rng)   # same arguments
    low, high = percentile(values, [50 * (1 - confidence), 50 * (1 + confidence)])
    # 2. return the two bounds as Python floats
```
e) l'écart-type (ta fonction `std`) de `bootstrap_distribution(chinstrap_mass, rng=np.random.default_rng(0))`.

</details>

### Ex 2.23 — Bootstraps de 20 (livre) ou de n (aujourd'hui) ? 🔬

<details><summary>Indice 1</summary>

`ci_width` n'est qu'un appel à ta `bootstrap_ci` (à 80 %, avec `sample_size` et `n_boot`), suivi d'une soustraction. `coverage` répète une enquête complète : un nouvel échantillon, puis son intervalle, et compte les réussites.

</details>
<details><summary>Indice 2</summary>

`bootstrap_ci(sample_23, confidence=0.8, n_boot=n_boot, sample_size=sample_size, rng=rng)`. Dans `coverage`, à chaque tour : `new = sample(population, 500, replace=False, rng=rng)`, puis `low, high = bootstrap_ci(new, confidence=0.8, n_boot=500, sample_size=sample_size, rng=rng)`, et compte les tours où `low <= population.mean() <= high`.

</details>
<details><summary>Indice 3</summary>

`widths_by_size` : `[ci_width(size, np.random.default_rng(1)) for size in SIZES_23]`. Pour lire la pente du graphique logarithmique : de 5 à 20, ou de 50 à 200 (la taille multipliée par 4), par combien la largeur est-elle divisée ?

</details>

### Ex 2.24 — Comparer avec `scipy.stats.bootstrap` 📦

<details><summary>Indice 1</summary>

Lis la section *Parameters* de la documentation : `data`, `statistic`, `n_resamples`, `confidence_level`, `method`, `rng`. Le résultat a deux attributs utiles : `confidence_interval` (avec `.low` et `.high`) et `standard_error`.

</details>
<details><summary>Indice 2</summary>

Le premier argument est un tuple d'un échantillon ; précise `method`, `n_resamples`, `confidence_level` et `rng` (un générateur neuf, de graine 0, pour chaque appel) ; lis ensuite `confidence_interval.low` et `.high`.

</details>
<details><summary>Indice 3</summary>

a) `res = scipy_stats.bootstrap((chinstrap_mass,), np.mean, method="percentile", ...)`, en complétant avec les autres arguments de l'énoncé (`confidence_level`, `n_resamples` et un générateur neuf de graine 0 pour `rng`) ; les bornes sont `res.confidence_interval.low` et `.high`, à ranger dans une liste. b) Le même appel sans `method` (ou avec `method="BCa"`), avec un **nouveau** générateur de graine 0. Pour la question des notes : combien de rééchantillons de chaque côté ? Compare aussi `res.bootstrap_distribution[:1000]` avec ta distribution de 2.22.

</details>

### Ex 2.25 — Distances entre chiffres dans l'espace à 784 dimensions 📦

<details><summary>Indice 1</summary>

Chaque image est une ligne de `X_digits`. `X_digits - X_digits[i]` soustrait l'image `i` de toutes les lignes (broadcasting, 0A) ; `np.linalg.norm(..., axis=1)` donne alors les 2 000 distances d'un coup.

</details>
<details><summary>Indice 2</summary>

b) et c) : dans ce vecteur de distances, mets `np.inf` à la place n° `i` (l'image elle-même), puis `np.argmin` donne l'indice de la plus proche. d) Une matrice 500 × 500 de distances (une ligne par image), `np.triu_indices(500, k=1)` pour ne garder chaque paire qu'une fois, et le masque « même chiffre » `y500[:, None] == y500[None, :]`. e) et f) : les distances du point 0 aux autres (`points[1:] - points[0]`), leur minimum et leur maximum.

</details>
<details><summary>Indice 3</summary>

```python
d = np.linalg.norm(points[1:] - points[0], axis=1)
contrast = (d.max() - d.min()) / d.min()
```
Pour c : `np.mean([y_digits[np.argmin(dist(i))] == y_digits[i] for i in range(500)])`, où `dist(i)` renvoie les distances de l'image `i`, la sienne remplacée par `np.inf`.

</details>

### Ex 2.26 — Covariance et corrélation 🔨

<details><summary>Indice 1</summary>

La covariance ressemble à la variance, avec deux variables : le produit des deux écarts remplace le carré de l'écart. La corrélation la divise par les deux écarts-types.

</details>
<details><summary>Indice 2</summary>

Vérifie que `x` et `y` sont à une dimension et de même longueur ; `dx = x - mean(x)`, `dy = y - mean(y)`, puis `(dx * dy).sum() / (n - ddof)`. `correlation` : refuse une variable constante (`np.ptp(x) == 0`), puis `covariance(x, y) / (std(x) * std(y))`, avec ddof = 0 partout, et `np.clip(r, -1, 1)`.

</details>
<details><summary>Indice 3</summary>

c) `covariance(flipper / 10, mass / 1000)`. d) `correlation(bill_length, bill_depth)` : le signe est-il celui que tu attendais pour des oiseaux plus ou moins grands ?

</details>

### Ex 2.27 — Deviner la corrélation d'un nuage de points 📈

<details><summary>Indice 1</summary>

Commence par les nuages faciles : des points serrés le long d'une droite ont un $|r|$ proche de 1, et le signe suit le sens de la pente.

</details>
<details><summary>Indice 2</summary>

Un nuage large mais penché a un $|r|$ moyen. Méfie-toi de deux formes : une courbe qui monte des deux côtés du centre (le lien n'est pas en ligne droite), et un paquet sans forme avec un seul point très loin.

</details>
<details><summary>Indice 3</summary>

b) Repère le nuage au point isolé, puis `far = np.argmax(np.hypot(x - x.mean(), y - y.mean()))` et `np.corrcoef(np.delete(x, far), np.delete(y, far))[0, 1]`.

</details>

### Ex 2.28 — Matrices de covariance et de corrélation des manchots 🔨

<details><summary>Indice 1</summary>

La matrice de covariance range les covariances de toutes les paires de colonnes (🧮 de la fiche §2.8) ; commence par centrer chaque colonne.

</details>
<details><summary>Indice 2</summary>

`Xc = X - mean(X, axis=0)` (ta fonction de 2.13), puis `Xc.T @ Xc / (n - ddof)` : $(p, n) \times (n, p)$ donne $(p, p)$. Corrélation : `s = np.sqrt(np.diag(C))`, puis `C / np.outer(s, s)` divise la case $(j, k)$ par $s_j\,s_k$ ; mets des 1 sur la diagonale (`np.fill_diagonal`).

</details>
<details><summary>Indice 3</summary>

b) le plus petit coefficient hors de la diagonale. d) `measured.groupby("species")` donne les espèces dans l'ordre alphabétique (Adelie, Chinstrap, Gentoo) ; pour chaque groupe, `correlation(group["bill_length_mm"], group["bill_depth_mm"])`.

</details>

### Ex 2.29 — Le piège de ddof : NumPy, pandas et toi 🐛

<details><summary>Indice 1</summary>

Exécute chaque fonction du collègue et regarde ce qu'elle renvoie : une valeur, une forme, un écart-type. Puis, dans chaque ligne, cherche quel diviseur ($n$ ou $n - 1$) utilise chaque fonction appelée (fiche, 🕰️ de la §2.8).

</details>
<details><summary>Indice 2</summary>

a) Quel diviseur `np.cov` utilise-t-il par défaut, et `np.std` ? Écris le rapport de ces deux diviseurs en fonction de $n$. b) `np.cov` considère chaque **ligne** comme une variable. c) La méthode `.std()` de pandas divise par $n - 1$ ; mesure ensuite avec `np.std`, qui divise par $n$.

</details>
<details><summary>Indice 3</summary>

Chaque correction tient en une ligne et porte sur un argument. `correlation_fixed` : le même ddof en haut et en bas, par exemple `np.cov(x, y, ddof=0)[0, 1]` divisé par le produit des deux `np.std`. `cov_matrix_fixed` : `np.cov` attend une variable par **ligne** ; son argument `rowvar` change cette convention (n'oublie pas `ddof=0`). `standardize_fixed` : garde la formule du collègue, mais donne à `.std()` de pandas le ddof de NumPy.

</details>

### Ex 2.30 — Le quartet d'Anscombe 🎨

<details><summary>Indice 1</summary>

`fit_line` : les deux formules de l'énoncé, avec tes fonctions `covariance`, `variance` et `mean` (le même ddof pour la covariance et la variance : il se simplifie). `draw_quartet` : `plt.subplots(2, 2, sharex=True, sharey=True)`, puis une boucle sur `ANSCOMBE.items()` et `axes.ravel()`.

</details>
<details><summary>Indice 2</summary>

Pour tracer une droite, deux points suffisent : `ends = np.array([2, 20])` et `ax.plot(ends, intercept + slope * ends)`. a) `np.median(y)` pour chaque jeu. b) les écarts `np.abs(y - (intercept + slope * x))` et leur maximum.

</details>
<details><summary>Indice 3</summary>

c) `worst = np.argmax(np.abs(y - (intercept + slope * x)))` pour le jeu III, puis `fit_line(np.delete(x, worst), np.delete(y, worst))` ; et la corrélation, `np.corrcoef` des mêmes données.

</details>

### Ex 2.31 — Docstring et test pytest pour `zscore` 🛠️

<details><summary>Indice 1</summary>

La docstring suit le modèle de 0A.61 : un résumé, puis des sections soulignées de tirets. Les tests suivent 0A.62 : une fonction `test_...` par idée, et au moins un test qui fait échouer chaque version buggée.

</details>
<details><summary>Indice 2</summary>

`flag_outliers` : `z = mylearn.stats.zscore(x)`, puis `np.flatnonzero(np.abs(z) > threshold)`. Pour les exemples `>>>`, exécute-les d'abord et recopie exactement ce que Python affiche, erreur comprise (avec la ligne `Traceback (most recent call last):`). Pour le test de l'axe, prends un tableau 2-D dont les colonnes ont des moyennes (ou des écarts-types) différentes : si toutes avaient la même moyenne et le même écart-type, une version qui ignore `axis` passerait.

</details>
<details><summary>Indice 3</summary>

Les lignes clés de chaque test. Le cas limite : dans un bloc `with pytest.raises(ValueError):`, appelle `zscore` sur des données constantes. La propriété : `z = zscore(np.array([...]))` sur des valeurs quelconques, puis `assert z.mean() == pytest.approx(0, abs=1e-12)`, et un `assert` du même genre pour `z.std()`. L'axe : la même idée avec `Z = zscore(X, axis=0)`, en comparant `Z.mean(axis=0)` et `Z.std(axis=0)` à des listes. Enfin, `my_zscore_tests` est la liste de tes trois fonctions de test (leurs noms, sans parenthèses).

</details>

### Ex 2.32 — Mêmes statistiques, autre dessin : fabrique ton quartet 🏆

<details><summary>Indice 1</summary>

Suis les quatre étapes de l'énoncé une par une, en affichant après chacune la moyenne, l'écart-type et la corrélation de ce que tu viens de calculer.

</details>
<details><summary>Indice 2</summary>

1. `zx = (hx - hx.mean()) / hx.std()`. 2. `c = np.mean((hy - hy.mean()) * zx)` (presque 0 pour ce cœur symétrique), `e = hy - hy.mean() - c * zx`, `ze = e / e.std()`. 3. `zy = r * zx + np.sqrt(1 - r ** 2) * ze`.

</details>
<details><summary>Indice 3</summary>

4. Avec ddof = 1, la variance vaut 11 quand l'écart-type **avec ddof = 0** vaut $\sqrt{11\,(n - 1) / n}$ : `x = 9 + np.sqrt(11 * (n - 1) / n) * zx`, et de même pour `y`, avec 7,5 et 4,125.

</details>
