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

La somme des sept valeurs vaut 76. Remplacer 40 par 400 ne change ni la valeur du milieu ni la valeur la plus fréquente. Pour une catégorie comme une île, on ne peut ni additionner ni trier : on peut seulement compter.

</details>

### 2.Q2 — Quand a-t-on le droit de parler de probabilités ?

<details><summary>Indice 1</summary>

Deux conditions pour une distribution discrète : des nombres entre 0 et 1, et une somme égale à 1. Pour une loi continue, ce sont les **aires** qui jouent ce rôle.

</details>
<details><summary>Indice 2</summary>

Additionne les nombres des questions 1 et 2. Pour les questions 3 et 4, relis l'encadré 🧮 « densité » de la fiche §2.2 et la figure de droite de la §2.3.2.

</details>
<details><summary>Indice 3</summary>

Trois affirmations sont vraies : la 3, la 5 et la 6.

</details>

### 2.Q3 — Graine et pseudo-aléatoire : vrai ou faux

<details><summary>Indice 1</summary>

Un générateur pseudo-aléatoire est un programme : mêmes entrées, même sortie. Quelle est son « entrée » ?

</details>
<details><summary>Indice 2</summary>

Pour chaque affirmation, demande-toi ce qui se passe quand on relance le programme, avec ou sans graine. Pour la 6, relis la fin du premier paragraphe de la fiche §2.2.1.

</details>
<details><summary>Indice 3</summary>

Deux affirmations sont vraies : la 1 et la 2.

</details>

### 2.Q4 — Loi uniforme sur [0, 1] : questions pièges

<details><summary>Indice 1</summary>

Pour une loi uniforme, une probabilité est une **longueur** d'intervalle divisée par la longueur totale (fiche §2.3.1).

</details>
<details><summary>Indice 2</summary>

Une valeur exacte est un intervalle de longueur 0. L'aire sous la densité doit valoir 1 : quelle hauteur faut-il sur un intervalle de longueur 1 ? de longueur 2 ? Pour `rng.random()` et `rng.integers`, relis la fiche : quelles bornes sont exclues ?

</details>
<details><summary>Indice 3</summary>

Les réponses 1 et 2 sont des longueurs d'intervalles (0,3 et 0,5) ; la 3 vaut 0 ; les hauteurs sont 1 et 0,5 ; l'espérance est le milieu de l'intervalle ; les deux fonctions de NumPy excluent leur borne haute.

</details>

### 2.Q5 — La règle 68-95-99,7

<details><summary>Indice 1</summary>

Exprime chaque borne en nombre d'écarts-types au-dessus ou au-dessous de 170 cm.

</details>
<details><summary>Indice 2</summary>

163 et 177, c'est 170 ± 7 ; 156 et 184, c'est 170 ± 2 × 7 ; 149, c'est 170 − 3 × 7. Pour « plus de » ou « moins de », pense à la symétrie : ce qui reste hors de l'intervalle se partage à parts égales des deux côtés.

</details>
<details><summary>Indice 3</summary>

1 et 2 : 68 % et 95 %. 3 : la moitié de 5 %. 4 : la moitié de 0,3 %. La règle n'est vraie que pour la loi normale ; l'écart-type a l'unité des données, la variance cette unité au carré.

</details>

### 2.Q6 — Bernoulli ou multinoulli ?

<details><summary>Indice 1</summary>

Compte les issues possibles : deux (Bernoulli), un nombre fini supérieur à deux (catégorielle), ou une infinité de valeurs réelles (ni l'une ni l'autre).

</details>
<details><summary>Indice 2</summary>

Pour chaque grandeur, écris la liste de ses valeurs possibles : est-elle finie ? Pour la question 7, souviens-toi que les probabilités doivent avoir une somme égale à 1.

</details>
<details><summary>Indice 3</summary>

Deux grandeurs suivent une loi de Bernoulli (1 et 3), trois une loi catégorielle (2, 4, 6), une aucune des deux (5). Pour $K$ classes, il faut $K$ probabilités, mais la dernière se déduit des autres.

</details>

### 2.Q7 — Une espérance qu'on ne tire jamais

<details><summary>Indice 1</summary>

L'espérance est une moyenne pondérée par les probabilités (fiche §2.3.5, 0B 101.7.3). Une moyenne tombe souvent « entre » les valeurs.

</details>
<details><summary>Indice 2</summary>

Pour le dé à 4 faces, les quatre probabilités valent $\frac{1}{4}$. Pour la question 4, utilise la linéarité de l'espérance vue en 0B : $\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b$.

</details>
<details><summary>Indice 3</summary>

Le dé donne une espérance de 5, qui n'est pas une face ; la Bernoulli donne 0,3, qui n'est ni 0 ni 1 ; $\mathbb{E}[2X + 1] = 11$ ; la moyenne des tirages se rapproche de l'espérance : c'est la loi des grands nombres (0B, 101.7.5).

</details>

### 2.Q8 — Dépendant, indépendant, i.i.d.

<details><summary>Indice 1</summary>

Pour chaque situation, demande-toi : connaître la première valeur me renseigne-t-il sur la seconde ?

</details>
<details><summary>Indice 2</summary>

Pour les situations 2 à 4, cherche ce qui relie les deux grandeurs (le climat, la taille de l'animal, l'espèce). Pour la dernière situation, relis la fiche §2.4.1 : ce qui casse l'hypothèse i.i.d. entre entraînement et test.

</details>
<details><summary>Indice 3</summary>

Seules les situations 1 et 5 sont i.i.d. Dans la situation 6, les exemples de test ne sont pas indépendants de ceux d'entraînement : c'est une fuite de données.

</details>

### 2.Q9 — Avec ou sans remise ?

<details><summary>Indice 1</summary>

Pour chaque situation : un élément déjà tiré peut-il ressortir ?

</details>
<details><summary>Indice 2</summary>

Relis le tableau de la fiche §2.5. Un même exemple ne doit jamais être à la fois dans le jeu d'entraînement et dans le jeu de test. Pour `rng.choice`, relis l'exemple de code de la fiche : quelle est la valeur par défaut de `replace` ?

</details>
<details><summary>Indice 3</summary>

Sont **avec** remise : le rééchantillon bootstrap, les commandes du café et `rng.choice` par défaut. Les autres sont sans remise, et l'affirmation 7 est fausse.

</details>

### 2.Q10 — Ce que le bootstrap estime, et ce qu'il n'invente pas

<details><summary>Indice 1</summary>

Le bootstrap ne fait que retirer au hasard des éléments de l'échantillon qu'on a déjà. Peut-il savoir ce qui n'y est pas ?

</details>
<details><summary>Indice 2</summary>

Relis la fiche §2.6 (« Ce que le bootstrap ne fait pas ») et l'encadré 🕰️ : que change le **nombre** de rééchantillons, et que change leur **taille** ? Pour la question 6, relis la fin de la §2.5.3.

</details>
<details><summary>Indice 3</summary>

Trois affirmations sont vraies : la 2, la 3 et la 6.

</details>

### 2.Q11 — Une image est un point dans un espace à 784 dimensions

<details><summary>Indice 1</summary>

Une coordonnée par nombre qui décrit l'image : combien de pixels, combien de nombres par pixel ?

</details>
<details><summary>Indice 2</summary>

Pour la question 2, compte trois nombres par pixel. Pour la question 3, demande-toi combien de coordonnées diffèrent entre les deux images. Pour la question 6, relis la fin de la fiche §2.7.

</details>
<details><summary>Indice 3</summary>

784 et 12 288 coordonnées ; deux images presque identiques ont presque tous leurs pixels égaux, donc une petite distance ; on ne sait pas dessiner plus de trois axes, mais les formules marchent en toute dimension ; le nombre de points qu'il faudrait pour « remplir » l'espace explose avec la dimension.

</details>

### 2.Q12 — Covariance, corrélation et quartet d'Anscombe

<details><summary>Indice 1</summary>

La covariance est une moyenne de **produits** d'écarts ; la corrélation divise la covariance par deux écarts-types.

</details>
<details><summary>Indice 2</summary>

L'unité d'un produit est le produit des unités ; dans un quotient, des unités identiques se simplifient. Pour la question 4, relis l'encadré ⚠️ « Deux contresens à éviter » de la fiche §2.8.2.

</details>
<details><summary>Indice 3</summary>

Des mm × g, puis aucune unité ; $r = -1$ ; en U, $y$ dépend fortement de $x$ mais pas en ligne droite ; corrélation n'est pas causalité ; Anscombe : mêmes statistiques, nuages différents, donc il faut tracer les données.

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

Les tâches 1 et 2 sont supervisées (une classification, une régression) ; les tâches 3 et 4 sont non supervisées (un clustering, une réduction de dimension).

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

$\mathbb{E}[X] = \frac{36}{8} = 4{,}5$ ; la somme des carrés de 1 à 8 vaut 204 ; la variance vaut donc $25{,}5 - 4{,}5^2$.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 2.1 — Moyenne, médiane et mode d'une liste de salaires

<details><summary>Indice 1</summary>

Commence par **trier** les neuf salaires du plus petit au plus grand : la médiane et le mode se lisent alors directement.

</details>
<details><summary>Indice 2</summary>

Moyenne : somme des 9 salaires divisée par 9. Médiane : la 5ᵉ valeur de la liste triée (4 en dessous, 4 au-dessus). Mode : la seule valeur qui apparaît deux fois. Pour e et f, seule la plus grande valeur change : qu'est-ce que ça change à la somme ? à la valeur du milieu ?

</details>
<details><summary>Indice 3</summary>

Triés : 1 900 ; 2 100 ; 2 100 ; 2 300 ; 2 400 ; 2 600 ; 2 800 ; 3 500 ; 12 000. La somme vaut 31 700 € ; avec 30 000 € au lieu de 12 000 €, elle augmente de 18 000 €.

</details>

### Ex 2.2 — De la casse de voitures à la distribution de probabilité

<details><summary>Indice 1</summary>

Une probabilité, ici, c'est un comptage divisé par le nombre total de voitures (fiche §2.2).

</details>
<details><summary>Indice 2</summary>

Un tour de roue fait 360° : la part d'un type est sa probabilité × 360°. « Ne pas tirer un SUV » est l'événement contraire de « tirer un SUV ». Sur 50 tirages avec remise, on attend 50 × la probabilité. Les sommes cumulées s'obtiennent en additionnant les probabilités au fur et à mesure ; le type choisi par $u$ est le premier dont la somme cumulée est strictement plus grande que $u$ (fiche §2.3.4).

</details>
<details><summary>Indice 3</summary>

Les probabilités sont 176/800, 144/800, 224/800, 96/800 et 160/800. Les sommes cumulées commencent par 0,22 ; 0,40 ; … Cherche entre quelles sommes cumulées se trouve 0,71.

</details>

### Ex 2.3 — La règle 68-95-99,7 sur les nageoires des manchots

<details><summary>Indice 1</summary>

68 % : à moins d'**un** écart-type de la moyenne ; 95 % : à moins de deux (fiche §2.3.2).

</details>
<details><summary>Indice 2</summary>

Un intervalle à $k$ écarts-types va de $\mu - k\sigma$ à $\mu + k\sigma$. Exprime 203 mm en nombre d'écarts-types au-dessus de la moyenne, puis demande-toi où se trouvent les manchots qui sortent de l'intervalle correspondant. Le z-score est l'écart à la moyenne divisé par l'écart-type.

</details>
<details><summary>Indice 3</summary>

$190 \pm 6{,}5$ et $190 \pm 13$ ; la part au-dessus de 203 mm est la moitié de 5 % ; pour d, multiplie cette part par 151 et arrondis ; $z = \frac{210 - 190}{6{,}5}$.

</details>

### Ex 2.4 — Variance : diviser par N ou par N − 1 ?

<details><summary>Indice 1</summary>

Fais un tableau à trois colonnes : la valeur, son écart à la moyenne, le carré de cet écart (fiche §2.3.2).

</details>
<details><summary>Indice 2</summary>

La somme des carrés se divise par $n = 5$ (ddof = 0) ou par $n - 1 = 4$ (ddof = 1) ; l'écart-type est la racine de la variance. Pour h et i, relis la règle de 0B : $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$, donc l'écart-type est multiplié par $|a|$.

</details>
<details><summary>Indice 3</summary>

Les écarts à la moyenne 6 sont −2, 1, 0, −3, 4 ; la somme de leurs carrés vaut 30. La moyenne des carrés des valeurs est $\frac{16 + 49 + 36 + 9 + 100}{5}$.

</details>

### Ex 2.5 — Espérances : Bernoulli, multinoulli et jeu de hasard

<details><summary>Indice 1</summary>

Toutes les questions se ramènent à $\mathbb{E}[X] = \sum_k x_k\,p_k$ (fiche §2.3.5), sauf b (la variance d'une loi de Bernoulli, fiche §2.3.3) et f (un codage one-hot, fiche §2.3.4).

</details>
<details><summary>Indice 2</summary>

Pour c, le nombre moyen de faces sur 200 lancers vaut 200 × l'espérance d'un lancer. Pour g, regarde une seule composante du vecteur one-hot : c'est une variable de Bernoulli ; quelle est son espérance ? Pour h, écris le tableau des gains nets possibles (en tenant compte de la mise) et de leurs probabilités.

</details>
<details><summary>Indice 3</summary>

Pour h, les gains nets sont 18 € (probabilité $\frac{1}{20}$), 3 € (probabilité $\frac{3}{20}$) et −2 € (probabilité $\frac{16}{20}$). Pour g, la composante n° $k$ vaut 1 avec la probabilité de la classe $k$.

</details>

### Ex 2.6 — Compter les tirages avec et sans remise

<details><summary>Indice 1</summary>

Avec cinq exemples, tu peux encore **écrire toutes les possibilités** à la main ; fais-le pour a, b et c avant de chercher une formule.

</details>
<details><summary>Indice 2</summary>

Suites avec remise : $n^k$. Sous-ensembles sans remise : $\binom{n}{k}$ (0B, 101.1.7). Pour f, un exemple donné n'est pas tiré à un tirage avec probabilité $1 - \frac{1}{n}$, et les tirages sont indépendants (fiche §2.5.3).

</details>
<details><summary>Indice 3</summary>

a : AB, AC, AD, AE, BC, … (continue la liste sans compter BA, déjà là sous la forme AB). b : ajoute les paires répétées AA, BB… f : $\left(\frac{4}{5}\right)^5$ ; g : $0{,}999^{1000}$ (pas $e^{-1}$, qui n'est que la limite) ; h : le complément de g.

</details>

### Ex 2.7 — Covariance et corrélation de cinq points à la main

<details><summary>Indice 1</summary>

Même méthode qu'en 2.4, avec deux colonnes d'écarts et une colonne de **produits** d'écarts.

</details>
<details><summary>Indice 2</summary>

Covariance : somme des produits des écarts divisée par $n$ (ddof = 0). Écart-type : racine de la somme des carrés des écarts divisée par $n$. $r$ : covariance divisée par le produit des deux écarts-types, avec le même ddof partout. Pour g et h, utilise ce que dit la fiche §2.8 sur le changement d'unité (ou démontre-le avec 2.8).

</details>
<details><summary>Indice 3</summary>

Les écarts de $x$ sont −2, −1, 0, 1, 2 ; ceux de $y$ sont −2, −1, 1, 0, 2. La somme de leurs produits vaut 9, et la somme des carrés des écarts vaut 10 pour $x$ comme pour $y$.

</details>

### Ex 2.8 — Changer d'unité : la covariance bouge, pas la corrélation

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

### Ex 2.9 — Le bootstrap en cinq lignes

<details><summary>Indice 1</summary>

Commence par la **question** à laquelle répond le bootstrap (« à quel point ce chiffre est-il fiable ? »), pas par la méthode.

</details>
<details><summary>Indice 2</summary>

Choisis un exemple de la vie courante : la note moyenne d'un restaurant sur 30 avis, le temps moyen d'attente à un guichet sur 25 clients… Puis déroule les étapes : retirer au hasard, avec remise, autant d'éléments que l'échantillon ; recalculer ; recommencer ; regarder l'étendue des résultats.

</details>
<details><summary>Indice 3</summary>

Une structure possible : (1) la question ; (2) on n'a qu'un échantillon ; (3) on fabrique des « échantillons de rechange » en retirant avec remise ; (4) on recalcule la moyenne sur chacun ; (5) les 95 % centraux de ces moyennes forment l'intervalle.

</details>

### Ex 2.10 — Corrélation, causalité et échantillon biaisé

<details><summary>Indice 1</summary>

Pour chaque situation, cherche une **troisième variable** qui pourrait expliquer le lien, ou une raison pour laquelle l'échantillon ne ressemble pas à la population visée.

</details>
<details><summary>Indice 2</summary>

A : qu'est-ce qui fait qu'un quartier a beaucoup d'incendies **et** beaucoup d'interventions ? Et dans quel sens va la causalité ? B : qui utilise une application de sport, et qui porte une montre la nuit ? Relis « Ce que le bootstrap ne fait pas » (fiche §2.6). C : qui pratiquait le rugby parmi les candidats embauchés il y a dix ans ? Que peut « représenter » cette ligne du CV ?

</details>
<details><summary>Indice 3</summary>

A : taille ou densité du quartier, causalité inversée (on envoie les pompiers là où il y a des incendies). B : l'intervalle mesure le hasard de l'échantillonnage parmi les utilisateurs, pas la différence entre eux et les Français. C : la ligne sert de variable de substitution (*proxy*) pour le genre ; vérifications : performances par groupe, retrait des variables proxy, contrôle humain, documentation ; le recrutement est un usage « à haut risque » de l'AI Act.

</details>

### Ex 2.11 — Fermi : la taille de l'espace des images

<details><summary>Indice 1</summary>

Le nombre d'images possibles, c'est le nombre de valeurs possibles d'un pixel, élevé à la puissance du nombre de pixels (0B, principe multiplicatif).

</details>
<details><summary>Indice 2</summary>

Pour passer d'une puissance de 2 à une puissance de 10 : $2^{10} \approx 10^3$, donc $2^{k} \approx 10^{0{,}3\,k}$. Pour la mémoire : nombre de valeurs × octets par valeur ; 1 Mo ≈ $10^6$ octets.

</details>
<details><summary>Indice 3</summary>

$2^{784} \approx 10^{236}$ ; $256^{784} = 2^{6\,272} \approx 10^{1\,888}$. Une image en `float32` : $784 \times 4$ octets. Pour la question 6, pense à ce qu'est une image « au hasard » (du bruit) comparée à un vrai chiffre.

</details>

### Ex 2.12 — Anscombe (1973) : regarder avant de calculer

<details><summary>Indice 1</summary>

L'article s'ouvre sur les idées reçues qu'il veut réfuter ; lis surtout son premier paragraphe et regarde ses quatre graphiques.

</details>
<details><summary>Indice 2</summary>

Pour la question 2, fais le tableau des écarts à la moyenne comme en 2.4. Pour la question 4, que vaut l'écart-type de $x$ si toutes les valeurs de $x$ sont égales ? Et une division par 0 ?

</details>
<details><summary>Indice 3</summary>

La moyenne des $x$ vaut 9 et la somme des carrés des écarts 110 ; divise par 11 puis par 10. Sans le point à $x = 19$, la variance de $x$ est nulle.

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

Avec $p = 0{,}87$ et $n = 400$, l'écart-type d'une proportion vaut à peu près $\sqrt{p(1 - p)/n} \approx 0{,}017$ ; l'intervalle à 95 % fait donc environ ± 3,3 points. Mentionne aussi ce que l'intervalle ne couvre pas (un jeu de test non représentatif).

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

Les indices des exercices 2.13 à 2.32 arriveront avec le notebook, à la prochaine session de génération.
