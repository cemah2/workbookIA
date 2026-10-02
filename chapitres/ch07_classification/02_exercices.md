# 7 · Classification — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch07_classification/06_mes_reponses.md` (créée par `python tools/start_chapter.py 7`), jamais dans ce fichier : il est mis à jour par Claude.
> Nouveauté de la partie II : les réponses courtes des **quiz** (sauf Q10), des **rappels** et des exercices ✏️ et ∂ 7.5 et 7.6 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », le quiz Q10, ∂ 7.7, la réflexion (🗣️ 🛠️ ⚖️) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"D"`), une liste pour plusieurs nombres (`[2, 5]`).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 🛠️ ⚖️ Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 7.Q1 — Label, prédiction, vérité terrain : le vocabulaire 🧠 ⏱️ 3 min
*Fiche §7.1 · livre §7.1 · parcours R*

a) Parmi ces expressions, laquelle ne désigne **pas** la classe fixée par l'expert ? (A) le label (*étiquette*) ; (B) la vérité terrain ; (C) la valeur prédite ; (D) la valeur réelle.
b) Vrai ou faux : les labels obtenus en mirant les œufs peuvent eux-mêmes contenir des erreurs.
c) Vrai ou faux : un modèle qui retrouve sans erreur les labels de tous ses exemples d'entraînement est forcément un bon classifieur.
d) Pour un filtre anti-spam, qui fournit le label d'un e-mail, et qu'est-ce que la valeur prédite ? (dans ta copie)

### 7.Q2 — Binaire, multi-classe ou multi-étiquette ? Cinq situations 🧠 ⏱️ 3 min
*Fiche §7.1, §7.3 · livre §7.1 à §7.3 · parcours R*

Pour chaque tâche, réponds `"B"` (binaire), `"C"` (multi-classe) ou `"E"` (multi-étiquette).

a) Décider si un e-mail est un spam.
b) Reconnaître l'espèce d'un manchot : Adélie, Chinstrap ou Gentoo.
c) Choisir, pour une photo de vacances, ses mots-clés dans une liste de 200 (« plage », « chien », « coucher de soleil »…).
d) Reconnaître le chiffre manuscrit d'une image de MNIST.
e) Décider si un œuf est fécondé.
f) Un œuf peut-il être à la fois *yolker* et *quitter* ? Quel type de problème est celui des trois classes d'œufs du §7.3 ? (dans ta copie)

### 7.Q3 — Régions et frontières de décision 🧠 ⏱️ 3 min
*Fiche §7.2.1, §7.3 · livre §7.2.1 · parcours R*

a) Vrai ou faux : une frontière de décision est toujours une droite.
b) Un classifieur à 3 classes, capable de prédire chacune d'elles, découpe le plan en au moins combien de régions de décision ?
c) Vrai ou faux : une même classe peut occuper plusieurs régions séparées les unes des autres.
d) Vrai ou faux : un classifieur attribue les deux classes à la fois à un point situé exactement sur la frontière.
e) Vrai ou faux : quand une droite sépare parfaitement deux classes, cette droite est unique.
f) Pourquoi le livre cherche-t-il la frontière **la plus simple** qui sépare bien les groupes ? (dans ta copie)

### 7.Q4 — Classes qui se recouvrent : probabilités et politique de seuil 🧠 ⏱️ 3 min
*Fiche §7.2.1 · livre §7.2.1 · parcours R*

On dispose d'une carte de $P(\text{fécondé} \mid \mathbf{x})$, et l'on déclarait « fécondé » dès que cette probabilité atteignait 0,5. On baisse le seuil à 0,2, sans toucher au modèle.

a) Vrai ou faux : le recall des œufs fécondés ne peut pas baisser.
b) Vrai ou faux : le nombre de faux positifs ne peut pas baisser.
c) La région « fécondé » : (A) grandit ou reste la même ; (B) rétrécit ou reste la même ; (C) peut grandir à un endroit et rétrécir ailleurs.
d) Jeter un œuf fécondé (un faux négatif) coûte 7 fois plus cher que garder un œuf non fécondé (un faux positif). Quel seuil minimise le coût moyen, si les probabilités sont calibrées ? (3 décimales)
e) D'après le livre, la place de la frontière ne dépend pas seulement des données et de l'algorithme. De quoi d'autre, et qui en décide ? (dans ta copie)

### 7.Q5 — Cinq mesures par œuf : ce que change (ou non) la dimension 🧠 ⏱️ 3 min
*Fiche §7.3 · livre §7.3 · parcours R*

On mesure cinq choses sur chaque œuf : poids, longueur, couleur, circonférence moyenne et heure de ponte.

a) Quelle est la dimension de l'espace des échantillons ?
b) Vrai ou faux : la distance euclidienne entre deux œufs se calcule avec la même formule qu'en 2D, avec cinq termes au lieu de deux.
c) Vrai ou faux : le temps de calcul et la mémoire d'un algorithme ne dépendent pas du nombre de features.
d) Quelle est la dimension d'une frontière de décision dans cet espace ?
e) On ne peut plus dessiner les données. Comment les « regarder » quand même ? (dans ta copie)

### 7.Q6 — Un-contre-tous : combien de modèles, quelle décision ? 🧠 ⏱️ 3 min
*Fiche §7.4.1 · livre §7.4.1 · parcours R*

a) Combien de classifieurs binaires un un-contre-tous entraîne-t-il pour 7 classes ?
b) Avec quatre classes A, B, C et D, les quatre classifieurs donnent, pour un point, les probabilités 0,30 (A), 0,55 (B), 0,52 (C) et 0,10 (D). Quelle classe est prédite ?
c) Vrai ou faux : dans un un-contre-tous, les scores des $K$ classifieurs somment toujours à 1.
d) Pour un autre point, les scores de décision (positifs pour « oui ») valent −1,2 (A), −0,4 (B), −2,0 (C) et −0,9 (D) : aucun classifieur ne dit « oui ». Quelle classe est prédite ?
e) Le livre range *binary relevance* parmi les noms de l'un-contre-tous. Qu'est-ce qui change dans la décision quand on l'utilise en multi-étiquette ? (dans ta copie)

### 7.Q7 — Un-contre-un : duels, votes et coût 🧠 ⏱️ 3 min
*Fiche §7.4.2 · livre §7.4.2 · parcours R*

a) Combien de duels un un-contre-un organise-t-il pour 6 classes ?
b) Vrai ou faux : le duel A contre B est entraîné sur tous les échantillons, ceux des autres classes formant une troisième catégorie.
c) Vrai ou faux : avec 3 classes, l'un-contre-un et l'un-contre-tous entraînent le même nombre de classifieurs.
d) Avec quatre classes A, B, C, D, un point reçoit 2 voix pour A, 2 pour B, 1 pour C et 1 pour D. Quelle classe la règle de `mylearn` prédit-elle ?
e) Vrai ou faux : pour tout $K \ge 2$, le nombre de duels dépasse la moitié de $K \times K$.
f) Un duel vote même pour un point qui n'appartient à aucune de ses deux classes. Pourquoi ces voix « hors sujet » ne faussent-elles pas, en général, le résultat ? (dans ta copie)

### 7.Q8 — Ce que k-means ne peut pas deviner tout seul 🧠 ⏱️ 3 min
*Fiche §7.5 · livre §7.5 · parcours R*

a) Vrai ou faux : k-means a besoin des labels des exemples.
b) Le nombre $k$, choisi avant l'entraînement, est : (A) un paramètre appris ; (B) un hyperparamètre ; (C) un label.
c) Vrai ou faux : avec $k = 3$, sur des données qui forment 5 groupes bien séparés, k-means trouve les 5 groupes.
d) Vrai ou faux : deux lancers de k-means sur les mêmes données, depuis des centres de départ différents, donnent toujours le même résultat.
e) Vrai ou faux : le $k$ qui donne l'inertie la plus basse est le meilleur.
f) Sans labels, sur quoi peux-tu t'appuyer pour choisir $k$ ? Cite trois pistes. (dans ta copie)

### 7.Q9 — Densité d'échantillons quand les features s'accumulent 🧠 ⏱️ 3 min
*Fiche §7.6 · livre §7.6 · parcours R, M*

On dispose de 20 échantillons. Chaque feature est ramenée à $[0, 1]$, puis chaque axe est découpé en 4 cases de même largeur.

a) Quelle est la densité moyenne (échantillons par case) en dimension 1 ?
b) En dimension 2 ? (2 décimales)
c) En dimension 3 ? (4 décimales)
d) Vrai ou faux : à dimension fixée, doubler le nombre d'échantillons double la densité.
e) Vrai ou faux : ici, chaque feature ajoutée divise la densité par 4.
f) Pourquoi une densité très inférieure à 1 oblige-t-elle le classifieur à « deviner » ? (dans ta copie)

### 7.Q10 — Bénédiction de la structure : vrai ou faux justifié 🧠 ⏱️ 4 min
*Fiche §7.6 · livre §7.6 · parcours R*

Pour chaque affirmation, réponds vrai ou faux **et justifie en une phrase** (dans ta copie ; correction avec `05_solutions.md`).

1. Les données réelles sont en général réparties à peu près uniformément dans l'espace des features.
2. Les images de MNIST, de 784 pixels, n'occupent qu'une toute petite partie de l'espace de toutes les images possibles.
3. La bénédiction de la structure est un théorème : elle garantit qu'un modèle entraîné en grande dimension généralisera.
4. Retirer des features peut améliorer un classifieur.
5. Même si presque tout l'espace est vide, la densité peut être élevée là où se trouvent les données.

### 7.Q11 — Géométrie déroutante en grande dimension 🧠 ⏱️ 3 min
*Fiche §7.6.1 · livre §7.6.1 · parcours R, M*

a) Vrai ou faux : en dimension 100, la « peau » de 3 % d'une boule (la couche comprise entre 0,97 et 1 fois le rayon) contient plus de 99 % de son volume.
b) Vrai ou faux : d'après le livre, le rayon de l'hyper-orange dépasse 2 à partir de la dimension 10.
c) Vrai ou faux : dans un cube de côté 1, la distance du centre à un coin grandit avec la dimension.
d) On répartit uniformément un nombre fixé de points, et la dimension grandit. La distance au plus proche voisin : (A) reste petite devant la distance moyenne entre deux points ; (B) grandit et se rapproche de la distance moyenne entre deux points ; (C) diminue.
e) Dans le paradoxe de l'hyper-orange, qu'est-ce qui défie vraiment l'intuition : l'orange ou la boîte ? (dans ta copie)

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 7.R1 — Ch. 6 : entropie d'un cluster pur et d'un cluster mélangé 🔁 ★ ⏱️ 5 min
*Ch. 6 (§6.7) · parcours R, M*

On compte les espèces de manchots présentes dans un cluster. L'entropie de leur répartition, en bits, mesure à quel point le cluster est mélangé.

a) Un cluster de 30 Gentoo, et rien d'autre.
b) Un cluster de 20 Adélie et 20 Chinstrap.
c) 50 % d'Adélie, 25 % de Chinstrap et 25 % de Gentoo.
d) 70 %, 20 % et 10 % (3 décimales).
e) La plus grande entropie possible avec trois espèces (3 décimales).
f) Pour juger tout un clustering, on fait la moyenne des entropies des clusters, pondérée par leurs tailles. Pourquoi pondérer ? Que vaut cette moyenne quand chaque cluster est pur ? (dans ta copie)

### 7.R2 — Ch. 4 : probabilité a posteriori « fécondé » par la règle de Bayes 🔁 ★ ⏱️ 5 min
*Ch. 4 (§4.4.1, §4.4.2) · parcours R, M*

Dans un élevage, 30 % des œufs sont fécondés. Pour un œuf de poids et de longueur $\mathbf{x}$ donnés, la densité des mesures vaut $f_1(\mathbf{x}) = 0{,}08$ chez les œufs fécondés et $f_0(\mathbf{x}) = 0{,}04$ chez les autres.

a) $P(\text{fécondé} \mid \mathbf{x})$ (3 décimales).
b) Avec le seuil de 0,5, cet œuf est-il déclaré fécondé ? (`True` ou `False`)
c) La même probabilité si la moitié des œufs étaient fécondés (3 décimales).
d) Avec l'a priori de 30 %, quelle valeur du rapport $\frac{f_1(\mathbf{x})}{f_0(\mathbf{x})}$ donne exactement $P(\text{fécondé} \mid \mathbf{x}) = 0{,}5$ ? (3 décimales)
e) Pourquoi l'a priori change-t-il la frontière, alors que les deux lois de mesures restent les mêmes ? (dans ta copie)

### 7.R3 — 0B : développer ‖a − b‖² avec le produit scalaire 🔁 ★ ⏱️ 5 min
*0B (produit scalaire et norme) · parcours R, M*

Soit $\mathbf{a} = (3, -1, 2)$ et $\mathbf{b} = (1, 2, 0)$.

a) $\lVert \mathbf{a} \rVert^2$.
b) $\mathbf{a} \cdot \mathbf{b}$.
c) $\lVert \mathbf{b} \rVert^2$.
d) $\lVert \mathbf{a} - \mathbf{b} \rVert^2$, calculé directement.
e) Écris l'identité qui relie d) à a), b) et c), vérifie-la ici, et explique pourquoi elle permet de calculer **toutes** les distances entre deux nuages de points avec un seul produit matriciel. (dans ta copie)

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 7.1 — Compter les classifieurs OvR et OvO ✏️ ★ ⏱️ 10 min
**Objectif :** compter les modèles d'un un-contre-tous et d'un un-contre-un, et les exemples que chacun voit.
**Prérequis :** 0B (coefficient binomial) · fiche §7.4 · **Parcours :** R, M

Donne chaque fois la liste $[N_{\text{OvR}}, N_{\text{OvO}}]$ des nombres de classifieurs binaires.

a) $K = 3$ classes.
b) $K = 10$ (les chiffres de MNIST).
c) $K = 26$ (les lettres de l'alphabet).
d) $K = 100$.
e) À partir de combien de classes l'un-contre-un demande-t-il plus de 10 000 duels ?

On classe les 10 chiffres de MNIST avec 50 000 images d'entraînement, 5 000 par chiffre.

f) Chaque classifieur de l'un-contre-tous voit toutes les images. Combien d'exemples d'entraînement tous les classifieurs voient-ils **au total** ?
g) Même question pour l'un-contre-un, où chaque duel ne voit que les images de ses deux chiffres.
h) Entraîner un modèle sur $m$ exemples coûte $m^2$ opérations (c'est l'ordre de grandeur de certaines SVM à noyau, ch. 13). Combien de fois l'un-contre-un est-il moins cher que l'un-contre-tous ici ? (1 décimale)
i) Et pour **prédire** une image : quelle stratégie interroge le moins de classifieurs ? Qu'en conclus-tu sur le choix entre les deux ? (dans ta copie)

### Ex 7.2 — Dépouiller les votes d'un un-contre-un à quatre classes ✏️ ★★ ⏱️ 15 min
**Objectif :** compter les votes d'un un-contre-un, et voir ce que change la règle d'égalité.
**Prérequis :** Ex 7.1 · fiche §7.4.2 · **Parcours :** M

Quatre classes A, B, C et D, rangées dans cet ordre (indices 0 à 3). Pour un premier point, les six duels donnent :

| A-B | A-C | A-D | B-C | B-D | C-D |
|---|---|---|---|---|---|
| B | A | D | B | B | D |

a) Les votes, dans l'ordre $[A, B, C, D]$.
b) La classe prédite.
c) Le nombre total de voix distribuées.

Pour un second point :

| A-B | A-C | A-D | B-C | B-D | C-D |
|---|---|---|---|---|---|
| A | C | A | B | D | C |

d) Les votes $[A, B, C, D]$.
e) La classe prédite par la règle de `mylearn` : en cas d'égalité, la classe de plus petit indice.
f) La classe prédite par une autre règle : quand deux classes sont à égalité, le duel qui les a opposées les départage.
g) Vrai ou faux : une classe qui gagne tous ses duels peut ne pas être prédite.
h) Vrai ou faux : avec 4 classes, une égalité parfaite (les quatre classes avec le même nombre de voix) est possible.
i) `OneVsOneClassifier`, dans scikit-learn, départage les égalités en additionnant les « confiances » des duels. Pourquoi ne peux-tu pas appliquer cette règle ici ? Que faudrait-il connaître ? (dans ta copie)

### Ex 7.3 — Une itération de k-means à la main ✏️ ★★ ⏱️ 20 min
**Objectif :** dérouler l'algorithme de Lloyd à la main et voir l'inertie baisser.
**Prérequis :** ch. 2 (moyenne) · 0B (distance) · fiche §7.5 · **Parcours :** R, M

Six points du plan : $P_1 (1, 1)$, $P_2 (2, 1)$, $P_3 (1, 3)$, $P_4 (5, 4)$, $P_5 (6, 5)$, $P_6 (7, 4)$. On cherche $k = 2$ clusters, avec les centres de départ $c_1 = P_1$ et $c_2 = P_3$. Compare des distances **au carré** : pas besoin de racines.

a) Après la première affectation, le cluster (1 ou 2) de chaque point, dans l'ordre $P_1$ à $P_6$.
b) L'inertie de cette affectation, calculée avec les centres de départ.
c) Les deux nouveaux centres, sous la forme $[[x_1, y_1], [x_2, y_2]]$ (2 décimales).
d) L'inertie avec ces nouveaux centres, sans changer l'affectation (2 décimales).
e) Lors de la deuxième affectation, un seul point change de cluster. Lequel ? (son numéro, de 1 à 6)
f) Les centres après la deuxième mise à jour (3 décimales).
g) L'inertie finale (2 décimales).
h) Vrai ou faux : une troisième affectation change encore au moins un point de cluster.
i) Vrai ou faux : sur ces données, l'inertie n'a jamais augmenté d'une étape à l'autre.
j) Recommence avec les centres de départ $c_1 = P_1$ et $c_2 = P_2$. Arrives-tu aux mêmes clusters ? (dans ta copie)

### Ex 7.4 — Densité d'échantillons et nombre d'œufs nécessaires ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer une densité d'échantillons, le nombre d'exemples qu'elle exige, et ne pas la confondre avec une probabilité.
**Prérequis :** 0B (puissances) · fiche §7.6 · **Parcours :** M

Un éleveur dispose de 360 œufs mesurés. Chaque mesure est ramenée à $[0, 1]$, et chaque axe est découpé en 6 cases.

a) La densité en dimension 1, 2, 3 et 4, sous forme de liste (3 décimales).
b) Le nombre d'œufs nécessaires pour une densité de 2 en dimension 3.
c) La même chose en dimension 6.
d) Toujours en dimension 6 et pour une densité de 2, mais avec 10 cases par axe.
e) Un million d'œufs, 6 cases par axe : à partir de quelle dimension la densité passe-t-elle sous 1 ?

Reviens à l'exemple du livre : 10 œufs placés au hasard, uniformément et indépendamment, dans un cube découpé en $5 \times 5 \times 5 = 125$ petits cubes.

f) La probabilité qu'un petit cube donné ne contienne **aucun** œuf (3 décimales).
g) Le nombre moyen de petits cubes qui contiennent au moins un œuf (2 décimales).
h) Le livre traduit la densité de 0,08 par « 8 % de chances qu'un petit cube contienne un échantillon ». Compare avec f), et explique pourquoi une densité et une probabilité sont deux choses différentes, même quand elles sont proches. (dans ta copie)

### Ex 7.5 — Le rayon de l'hyper-orange : r(d) = √d − 1 ∂ ★★ ⏱️ 25 min
**Objectif :** démontrer la formule du rayon de l'hyper-orange et en tirer les surprises du livre.
**Prérequis :** 0B (distance en dimension $d$) · fiche §7.6.1 · **Parcours :** M

On place le centre de la boîte à l'origine. La boîte, de côté 4, est l'ensemble des points dont toutes les coordonnées sont comprises entre −2 et 2. Les ballons, de rayon 1, sont centrés aux points dont **toutes** les coordonnées valent +1 ou −1. L'orange est centrée à l'origine et touche les ballons.

1. Montre que le centre de chaque ballon est à la distance $\sqrt{d}$ de l'origine, et déduis-en que le rayon de l'orange vaut $r(d) = \sqrt{d} - 1$. (dans ta copie)

Puis, dans la partie 0 du notebook :

a) $r(2)$ (3 décimales).
b) $r(3)$ (3 décimales).
c) Le nombre de ballons en dimension 12.
d) La plus petite dimension où $r \ge 1{,}5$.
e) La plus petite dimension où $r > 2{,}5$.
f) $r(50)$ (3 décimales).
g) Le milieu de la face « première coordonnée = 2 » est à la distance 2 de l'origine. À quelle distance est-il du centre du ballon le plus proche ? Pourquoi l'orange peut-elle sortir de la boîte par ce point sans jamais traverser un ballon ? (dans ta copie)

### Ex 7.6 — Boule dans un cube : rapport des volumes par récurrence ∂ ★★★ ⏱️ 35 min
**Objectif :** calculer le volume de la boule en dimension $d$ par récurrence, et voir le rapport boule/cube s'effondrer.
**Prérequis :** Ex 7.5 · fiche §7.6.1 (encadré 🧮 sur le volume de la boule) · **Parcours :** M

On note $V_d$ le volume de la boule de rayon 1 en dimension $d$, avec $V_d = \frac{2\pi}{d} V_{d-2}$, $V_1 = 2$ et $V_2 = \pi$. Le rapport de la boule au cube de côté 2 qui la contient est $q_d = \frac{V_d}{2^d}$.

a) $V_4$ (3 décimales).
b) $V_5$ (3 décimales).
c) $q_4$ (3 décimales).
d) $q_{10}$, **en pourcentage** (3 décimales).
e) La dimension $d$ pour laquelle $V_d$ est le plus grand (calcule $V_1$ à $V_8$).
f) La part du volume d'une boule de dimension 100 située dans sa peau de 5 %, c'est-à-dire entre 0,95 et 1 fois le rayon (3 décimales).
g) Montre que $q_d = q_{d-2} \times \frac{\pi}{2d}$, puis que $q_d < q_{d-2}$ dès que $d \ge 2$. (dans ta copie)
h) Montre que $V_d < V_{d-2}$ dès que $d \ge 7$, puis explique pourquoi $V_d$ tend vers 0. Pourquoi est-ce troublant, alors que le volume du cube de côté 2 explose ? (dans ta copie)

### Ex 7.7 — La frontière du centroïde le plus proche est une droite ∂ ★★★ ⏱️ 30 min
**Objectif :** démontrer que les frontières du centroïde le plus proche sont des médiatrices, et en voir les limites.
**Prérequis :** 0B · rappel 7.R3 · fiche §7.5 (encadré 🧮 sur le centroïde le plus proche) · **Parcours :** M

Deux classes ont pour centroïdes $\boldsymbol{\mu}_0$ et $\boldsymbol{\mu}_1$, distincts. On cherche l'ensemble des points $\mathbf{x}$ à égale distance des deux.

1. Avec l'identité du rappel 7.R3, montre que $\lVert \mathbf{x} - \boldsymbol{\mu}_0 \rVert^2 = \lVert \mathbf{x} - \boldsymbol{\mu}_1 \rVert^2$ équivaut à $2(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) \cdot \mathbf{x} + \lVert \boldsymbol{\mu}_0 \rVert^2 - \lVert \boldsymbol{\mu}_1 \rVert^2 = 0$.
2. Explique pourquoi cet ensemble est une droite en 2D (un hyperplan en dimension $d$), perpendiculaire au segment $[\boldsymbol{\mu}_0, \boldsymbol{\mu}_1]$ et passant par son milieu.
3. Application : $\boldsymbol{\mu}_0 = (1, 2)$ et $\boldsymbol{\mu}_1 = (5, 4)$. Écris l'équation de la frontière sous la forme $ax + by = c$ avec des entiers, vérifie que le milieu du segment la satisfait, et dis dans quelle classe tombe le point $(4, 0)$.
4. Avec trois centroïdes non alignés, montre que les trois médiatrices se coupent en un même point. Que représente ce point ?
5. Imagine deux classes de mêmes centroïdes que dans 3, mais la classe 0 très étalée et la classe 1 très serrée. Où le centroïde le plus proche place-t-il la frontière, et où faudrait-il la placer ? Qu'est-ce que ce classifieur ignore ?

<a id="reflexion"></a>

## 🗣️ 🛠️ ⚖️ Réflexion

### Ex 7.8 — La malédiction de la dimension en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer la malédiction de la dimension à quelqu'un qui ne fait pas de machine learning.
**Prérequis :** fiche §7.6 · **Parcours :** R

Un ami agronome veut ajouter à son modèle de prévision des récoltes « toutes les mesures possibles » de ses capteurs : 200 variables pour 150 parcelles. Explique-lui pourquoi ce n'est pas forcément une bonne idée, en **cinq lignes au plus**. Contraintes :
- les mots « densité », « exemples » et « structure » ;
- une image de la vie courante ;
- au moins une parade concrète ;
- aucune formule.

Relis-toi à voix haute, puis compare avec la réponse modèle de `05_solutions.md`.

### Ex 7.9 — Lire la documentation officielle de KMeans (scikit-learn) 🛠️ ★ ⏱️ 15 min
**Objectif :** trouver vite une information précise dans la documentation d'une classe de scikit-learn, y compris l'historique de ses paramètres.
**Prérequis :** fiche §7.5 (encadré 🕰️ sur `KMeans`) · **Parcours :** C

Ouvre la [page de `sklearn.cluster.KMeans` pour la version 1.6](https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.KMeans.html), la version figée par le workbook, et réponds dans ta copie, en citant chaque fois la rubrique de la page où tu as trouvé la réponse.
1. Que dit la rubrique *Notes* de `labels_` et de `cluster_centers_` quand l'algorithme s'arrête avant d'avoir convergé (à cause de `tol` ou de `max_iter`) ?
2. Avec `n_init="auto"`, combien de départs sont faits quand `init` est une fonction (*callable*) ? Et quand c'est un tableau ?
3. La documentation présente `tol` comme une tolérance « relative ». Compare avec la règle exacte écrite dans la docstring de `mylearn.cluster.KMeans` : relative à quoi ?
4. Que renvoie `transform(X)` (forme et contenu) ? Comment en tirer, en une ligne de NumPy, la distance de chaque point à son propre centre ?
5. La fiche dit que `"auto"` est la valeur par défaut de `n_init` depuis la version 1.4. À la fin de la description du paramètre, retrouve la mention qui le confirme (« Changed in version… ») et celle qui date l'apparition de l'option (« Added in version… ») : depuis quelle version peut-on écrire `n_init="auto"` ? Pourquoi une bibliothèque change-t-elle une valeur par défaut en deux temps, plutôt que d'un coup ?
6. Cite trois différences entre le `KMeans` de scikit-learn et celui que tu écris dans `mylearn`.

### Ex 7.10 — Qui fixe le seuil ? Œufs, dépistage et coût des erreurs ⚖️ ★★ ⏱️ 20 min
**Objectif :** relier le seuil de décision aux coûts des erreurs, et se demander qui a le droit de les fixer.
**Prérequis :** ch. 3 (seuil, precision, recall) · fiche §7.2.1 · **Parcours :** R

Une couveuse industrielle trie ses œufs avec un modèle qui donne $P(\text{fécondé} \mid \mathbf{x})$. Un œuf déclaré fécondé part en incubation, les autres sont vendus pour la consommation. Garder un œuf non fécondé en incubation coûte 0,30 € (une place pendant trois semaines) ; vendre un œuf fécondé fait perdre un poussin, estimé à 50 €.
1. Quel seuil minimise le coût moyen, si les probabilités sont calibrées ? Que penses-tu de ce seuil ?
2. Ces deux coûts sont des choix. Qui les fixe ? Quelles personnes ou quels intérêts ne sont pas représentés par ces deux nombres (le client qui achète un œuf fécondé, le bien-être animal…) ?
3. Même question pour un dépistage du cancer par l'image : un faux négatif est un cancer manqué, un faux positif une biopsie inutile et des semaines d'angoisse. Qui devrait décider du seuil : l'équipe qui a entraîné le modèle, les médecins, les patients, les autorités de santé ?
4. Le modèle est moins bien calibré pour une race de poules que pour les autres (ou, pour le dépistage, pour une partie de la population). Le même seuil est-il juste pour tous ? Que faut-il vérifier ?
5. Rédige les trois phrases que la documentation du modèle (sa « fiche modèle ») devrait contenir au sujet du seuil.

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 7.E1 — OvR ou OvO : lequel choisir, et pourquoi ? 💼 ★★ ⏱️ 10 min
*Fiche §7.4 · prérequis 7.23 · parcours R*

« Vous devez classer des produits en 50 catégories avec un classifieur binaire, par exemple un SVM. Un-contre-tous ou un-contre-un ? Qu'est-ce qui vous ferait changer d'avis ? »

### 7.E2 — Expliquer k-means, ses hypothèses et ses échecs 💼 ★★ ⏱️ 10 min
*Fiche §7.5 · prérequis 7.26 · parcours R*

« Expliquez-moi k-means : ce que fait l'algorithme, ce qu'il optimise, les hypothèses qu'il fait sur les données, et un cas où il échoue. En quoi est-ce différent d'une classification ? »

### 7.E3 — Choisir le nombre de clusters sans labels 💼 ★★ ⏱️ 10 min
*Fiche §7.5 · prérequis 7.29 · parcours R*

« Vous segmentez des clients avec k-means et vous n'avez aucun label. Comment choisissez-vous le nombre de clusters ? »

### 7.E4 — Malédiction de la dimension : symptômes et parades 💼 ★★ ⏱️ 10 min
*Fiche §7.6, §7.6.1 · prérequis 7.20 · parcours R*

« Qu'est-ce que la malédiction de la dimension ? À quels symptômes la reconnaissez-vous dans un projet, et que faites-vous ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch07_classification/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie `mylearn/cluster.py` ou `mylearn/multiclass.py`. La partie 0 du notebook vérifie tes réponses courtes aux quiz, aux rappels et aux exercices ✏️ et ∂ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 7.11 | Des œufs en 2D : données, régions et frontière de décision | 📦 | ★ | 15 |
| 7.12 | k-means sur deux lunes : où tombera la coupure ? | 🔮 | ★ | 10 |
| 7.13 | Distances au carré vectorisées : pairwise_sq_distances | 🔨 | ★★ | 20 |
| 7.14 | Le classifieur du centroïde le plus proche | 🔨 | ★★ | 30 |
| 7.15 | Carte de probabilité et politique de seuil pour les œufs | 📈 | ★★ | 25 |
| 7.16 | Un-contre-tous avec des centroïdes : quelle classe sera sacrifiée ? | 🔮 | ★★ | 15 |
| 7.17 | Manchots sans labels : k-means face aux espèces | 📦 | ★★ | 25 |
| 7.18 | Formes arbitraires et bruit : DBSCAN et HDBSCAN | 📦 | ★★ | 25 |
| 7.19 | Distance au plus proche voisin quand la dimension grimpe | 🔮 | ★★ | 15 |
| 7.20 | Densité, plus proche voisin et concentration des distances | 🔬 | ★★ | 30 |
| 7.21 | Reproduire les figures 7.27 et 7.30 (boule/cube, hyper-orange) | 🎨 | ★★ | 25 |
| 7.22 | Un-contre-tous générique : OneVsRestClassifier | 🔨 | ★★★ | 40 |
| 7.23 | Un-contre-un générique : OneVsOneClassifier | 🔨 | ★★★ | 45 |
| 7.24 | OvR, OvO ou multi-classe natif : accuracy, nombre de modèles, temps | 🔬 | ★★★ | 35 |
| 7.25 | Initialisation k-means++ | 🔨 | ★★★ | 35 |
| 7.26 | k-means de Lloyd : la classe KMeans | 🔨 | ★★★ | 60 |
| 7.27 | k-means piégé : quatre bugs à débusquer | 🐛 | ★★★ | 30 |
| 7.28 | Coefficient de silhouette | 🔨 | ★★★ | 40 |
| 7.29 | Choisir k : coude de l'inertie et silhouette, de k = 2 à 7 | 🔬 | ★★★ | 35 |
| 7.30 | Phénomène de Hughes : des features de bruit qui font chuter l'accuracy | 🔬 | ★★★ | 40 |
| 7.31 | Défi : retrouver les espèces de manchots sans labels | 🏆 | ★★★ | 60 |
