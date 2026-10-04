# 4 · Règle de Bayes — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch04_bayes/06_mes_reponses.md` (créée par `python tools/start_chapter.py 4`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses des exercices ✏️ se vérifient dans la **partie 0** du notebook (`wb.check`) ; les quiz, les rappels, les exercices ∂ 🗣️ ⚖️ 📄 et les questions d'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, puis vérifie dans `05_solutions.md`.

### 4.Q1 — Deux écoles pour une même probabilité 🧠 ⏱️ 3 min
*Fiche §4.1, §4.2 · livre §4.1, §4.2 · parcours R*

1. D'après la fiche (§4.1), quel atout de l'approche bayésienne la rend utile en machine learning ? Illustre-le par un exemple de ton choix, qui n'est pas dans le livre.
2. Donne, en une phrase, la définition fréquentiste d'une probabilité.
3. Donne, en une phrase, la définition bayésienne.
4. « La probabilité que le biais de cette pièce dépasse 0,6 vaut 0,2. » Laquelle des deux écoles peut écrire cette phrase ? Pourquoi l'autre la refuse-t-elle ?
5. « Dans une ville, il a plu en moyenne 3 jours de novembre sur 10 depuis trente ans, donc la probabilité qu'il pleuve un jour de novembre est 0,3. » Quelle définition utilise ce raisonnement ?
6. Vrai ou faux : le machine learning d'aujourd'hui n'utilise que des méthodes bayésiennes.

### 4.Q2 — Le fréquentiste et la hauteur de la montagne 🧠 ⏱️ 3 min
*Fiche §4.2 · livre §4.2.1*

1. Un géomètre mesure dix fois la hauteur d'une montagne et obtient des valeurs un peu différentes. Comment le fréquentiste du livre interprète-t-il ces écarts ?
2. Qu'espère-t-il obtenir en combinant ses dix mesures ?
3. D'où vient vraiment le nom « fréquentiste » ? Quelle explication le livre en donne-t-il ?
4. Pour estimer la hauteur à partir de ces dix mesures bruitées, quelle statistique un fréquentiste calcule-t-il le plus souvent ?
5. Vrai ou faux : pour un fréquentiste strict, « la probabilité que la montagne mesure plus de 4 800 m » n'a pas de sens.
6. Comment l'erreur typique de cette estimation évolue-t-elle quand on multiplie le nombre de mesures par 100 (ch. 2) ?

### 4.Q3 — Le bayésien et la longueur du crayon 🧠 ⏱️ 3 min
*Fiche §4.2 · livre §4.2.2, §4.2.3 · parcours R*

1. Le livre justifie le doute de son bayésien par la pointe émoussée et la gomme usée du crayon. Ce flou empêcherait-il aussi un fréquentiste de mesurer le crayon ? Que dit cela du portrait ?
2. Quelle correction la fiche apporte-t-elle à ce portrait ?
3. Que représente, pour un bayésien, une distribution de probabilité sur la longueur du crayon ?
4. Avant la première mesure, que doit choisir le bayésien ?
5. Vrai ou faux : en accumulant les mesures, la distribution du bayésien se resserre.
6. Vrai ou faux : avec la boucle du §4.6 et un modèle où la grandeur est fixe, l'approche bayésienne suit sans effort une grandeur qui change avec le temps.

### 4.Q4 — Biais d'une pièce : le vocabulaire 🧠 ⏱️ 3 min
*Fiche §4.3 · livre §4.3*

1. Qu'appelle-t-on le biais d'une pièce ?
2. Une pièce a un biais de 0,15. Combien de faces attends-tu en moyenne sur 200 lancers ?
3. Que vaut le biais d'une pièce équilibrée ?
4. On lance une pièce 50 fois et l'on obtient 31 faces. Quelle est l'estimation fréquentiste de son biais ?
5. Pourquoi cette estimation varie-t-elle beaucoup pendant les premiers lancers ?
6. Quelle loi du ch. 2 décrit un seul lancer d'une pièce de biais $\theta$ ?

### 4.Q5 — Une seule face change déjà le verdict 🧠 ⏱️ 3 min
*Fiche §4.4 · livre §4.4 · parcours R*

Deux pièces d'aspect identique : l'une équilibrée, l'autre de biais 0,8. Tu en prends une au hasard (une chance sur deux) et tu la lances une fois : face.
1. Sans calcul : après cette face, laquelle des deux pièces est la plus probable ?
2. Sur le mur peint de la fiche, quelles zones restent possibles après l'observation ?
3. Calcule $P(\text{équilibrée} \mid \text{face})$ (3 décimales).
4. Et $P(\text{équilibrée} \mid \text{pile})$, si le lancer avait donné pile (3 décimales) ?
5. Pourquoi ces deux mises à jour ne s'éloignent-elles pas de 0,5 de la même quantité ?
6. Vrai ou faux : après une seule face, on est sûr d'avoir la pièce truquée.

### 4.Q6 — Prior, vraisemblance, évidence, posterior : qui est qui ? 🧠 ⏱️ 3 min
*Fiche §4.4.1, §4.4.2 · livre §4.4.1, §4.4.2 · parcours R, M*

Un détecteur de fumée sonne. On note $F$ l'hypothèse « il y a un feu » et $S$ l'observation « le détecteur sonne ».
1. Que représente $P(F)$ ? Quel est son nom dans la règle de Bayes ?
2. Que représente $P(S \mid F)$ ? Son nom ?
3. Que représente $P(S)$ ? Son nom ? Comment le calcule-t-on ?
4. Que représente $P(F \mid S)$ ? Son nom ?
5. Lequel de ces nombres le fabricant peut-il mesurer en laboratoire ? Lequel intéresse les habitants de la maison ?
6. Pourquoi le mot « évidence » est-il un faux ami ?

### 4.Q7 — La vraisemblance n'a pas à sommer à 1 🧠 ⏱️ 3 min
*Fiche §4.4.2, §4.7 · livre §4.4.2, §4.7 · parcours M*

Trois hypothèses sur le biais d'une pièce : 0,2 ; 0,5 et 0,9.
1. Écris les trois vraisemblances de l'observation « face ». Que vaut leur somme ?
2. Écris les trois vraisemblances de « pile ». Que vaut leur somme ?
3. Pour une hypothèse fixée, que vaut $P(\text{face} \mid H) + P(\text{pile} \mid H)$ ?
4. Vrai ou faux : après l'observation, les posteriors des trois hypothèses somment à 1.
5. Vrai ou faux : multiplier les trois vraisemblances par 10 ne change pas les posteriors.
6. Quel terme de la règle de Bayes « répare » le fait que les vraisemblances ne somment pas à 1 ?

### 4.Q8 — Sonde spatiale : ne pas inverser la condition 🧠 ⏱️ 3 min
*Fiche §4.5 · livre §4.5 · parcours R*

1. La sonde du livre détecte la vie sur 100 des 101 planètes habitées de son test. Quelle probabilité conditionnelle cela mesure-t-il ? Comment s'appelle-t-elle au ch. 3 ?
2. Le capitaine veut connaître $P(\text{vie} \mid \text{détecté})$. Comment s'appelle-t-elle au ch. 3 ?
3. Quel ingrédient faut-il en plus pour passer de l'une à l'autre ?
4. Le livre trouve $P(\text{vie} \mid \text{rien détecté}) \approx 0{,}001$. Est-ce le taux de faux négatifs (FNR) de la sonde ? Que vaut ce FNR, et quel nom porte $P(\text{vie} \mid \text{rien détecté})$ au ch. 3 ?
5. Pourquoi $P(\text{vie} \mid \text{détecté})$, environ 0,77, est-elle bien plus loin de 1 que $P(\text{pas de vie} \mid \text{rien détecté})$ ?
6. Que deviendrait $P(\text{vie} \mid \text{détecté})$ dans une région où presque aucune planète n'est habitée ?

### 4.Q9 — La boucle posterior → prior 🧠 ⏱️ 3 min
*Fiche §4.6, §4.6.1 · livre §4.6, §4.6.1 · parcours R*

1. Pourquoi le posterior d'une observation peut-il servir de prior pour la suivante ?
2. Quelle hypothèse sur les observations faut-il pour que la boucle soit correcte ?
3. Vrai ou faux : traiter les lancers dans l'ordre face, pile, face ou dans l'ordre pile, face, face donne le même posterior final.
4. Vrai ou faux : deux lancers d'une pièce **dont on ne connaît pas le biais** sont indépendants.
5. À quel moment de chaque tour de boucle divise-t-on par l'évidence ? Pourquoi ne pas attendre la fin ?
6. Pour 1 000 lancers d'une pièce, faut-il faire 1 000 tours de boucle ? Que suffit-il de connaître ?

### 4.Q10 — Plus de lancers, plus de certitude ? 🧠 ⏱️ 3 min
*Fiche §4.6.2, §4.7 · livre §4.6.2, §4.7 · parcours R*

1. En général, que devient le posterior quand les lancers s'accumulent ?
2. Vrai ou faux : avec 30 lancers, le posterior désigne toujours la bonne pièce.
3. On compare une pièce équilibrée et une pièce de biais 0,05. La pièce lancée donne 95 % de faces sur 1 000 lancers. Vers quelle hypothèse le posterior se dirige-t-il ? Est-ce rassurant ?
4. Que devient une hypothèse dont le prior vaut exactement 0 ?
5. Un prior trompeur (une bosse centrée loin du vrai biais, mais jamais nulle) empêche-t-il de trouver le bon biais ?
6. Avec 500 hypothèses sur le biais, quelle forme prend le posterior après beaucoup de lancers ? Comment évolue sa largeur ?

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 4.R1 — Ch. 3 : precision = P(malade | test positif) 🔁 ★ ⏱️ 5 min
*Ch. 3 (§3.7, §3.8) · parcours R, M*

On teste 1 000 personnes, dont 50 malades. Le test est positif pour 45 malades et pour 95 personnes saines.
1. Écris la matrice de confusion : TP, FN, FP, TN.
2. Calcule la precision (3 décimales).
3. Écris la precision comme une probabilité conditionnelle.
4. Calcule la sensibilité, le taux de faux positifs (FPR) et la prévalence.
5. Retrouve la precision avec la règle de Bayes à partir de ces trois nombres (fiche §4.4.1).
6. Quel terme de la règle de Bayes vaut le total de la colonne « test positif » divisé par 1 000 ?

### 4.R2 — Ch. 1 : un filtre anti-spam apprend-il avec des labels ? 🔁 ★ ⏱️ 5 min
*Ch. 1 (§1.2 à §1.4) · parcours R*

1. Pour entraîner un filtre anti-spam, on lui fournit des e-mails marqués « spam » ou « normal ». Quel type d'apprentissage est-ce ?
2. Comment appelle-t-on ces marques ?
3. Sur quelles données mesure-t-on la qualité du filtre, et pourquoi pas sur celles de l'entraînement ?
4. Un algorithme qui range des e-mails en groupes d'e-mails semblables, sans aucun label : quel type d'apprentissage ?
5. Au ch. 13, le classifieur Naive Bayes applique la règle de ce chapitre à chaque mot. À ton avis, quelles probabilités apprend-il à partir des e-mails étiquetés ?

### 4.R3 — 0B : trois faces de suite avec une pièce truquée 🔁 ★ ⏱️ 5 min
*0B (probabilités, $\prod$, logarithmes) · parcours R, M*

Une pièce a un biais de 0,7.
1. Quelle est la probabilité d'obtenir face, face, face ?
2. Quelle est la probabilité d'obtenir au moins une pile en trois lancers ?
3. Écris $\ln(0{,}7^3)$ comme une somme de logarithmes, puis calcule-le (3 décimales).
4. Quelle est la probabilité de face, pile, face, dans cet ordre ? Et d'obtenir exactement deux faces en trois lancers, dans n'importe quel ordre ?
5. Écris $0{,}7 \times 0{,}7 \times 0{,}3$ avec la notation $\prod$, pour la suite de lancers $x_1 = 1$, $x_2 = 1$, $x_3 = 0$ (1 = face), en utilisant $\theta^{x_i}(1 - \theta)^{1 - x_i}$.

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats ✏️ dans la partie 0 du notebook. Garde les valeurs exactes (des fractions) pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 4.1 — Une face : la pièce est-elle équilibrée ? ✏️ ★ ⏱️ 10 min
**Objectif :** calculer un posterior à deux hypothèses après une observation, avec la règle de Bayes puis avec des effectifs.
**Prérequis :** ch. 3 (règle du produit, probabilités totales) · fiche §4.4, §4.4.1 · **Fil rouge :** synthétique · **Parcours :** R, M

Un prestidigitateur te confie deux pièces d'aspect identique. L'une est équilibrée ; l'autre est truquée et tombe sur face 3 fois sur 4 (biais 0,75). Tu en prends une au hasard, chacune avec la même probabilité, et tu la lances une fois : **face**.

a) Le prior $P(\text{équilibrée})$.
b) Les vraisemblances de « face », sous la forme d'une liste $[P(\text{face} \mid \text{équilibrée}), P(\text{face} \mid \text{truquée})]$.
c) L'évidence $P(\text{face})$ (3 décimales).
d) Le posterior $P(\text{équilibrée} \mid \text{face})$ (2 décimales).
e) Le posterior $P(\text{truquée} \mid \text{face})$ (2 décimales).
f) On répète 800 fois l'expérience entière (choisir une pièce au hasard, la lancer une fois). Combien de faces attend-on au total ?
g) Parmi ces faces, combien viennent de la pièce équilibrée ?
h) Explique avec le mur peint pourquoi une seule face suffit à faire pencher la balance. (réponds dans ta copie)

### Ex 4.2 — Une pile : le verdict s'inverse ✏️ ★ ⏱️ 10 min
**Objectif :** comparer l'effet d'une pile et celui d'une face sur le posterior.
**Prérequis :** Ex 4.1 · fiche §4.4 · **Fil rouge :** synthétique · **Parcours :** M

Mêmes pièces et même prior qu'en 4.1, mais le lancer donne **pile**.

a) La vraisemblance $P(\text{pile} \mid \text{truquée})$ (2 décimales).
b) L'évidence $P(\text{pile})$ (3 décimales).
c) Le posterior $P(\text{équilibrée} \mid \text{pile})$ (3 décimales).
d) L'écart $P(\text{équilibrée} \mid \text{pile}) - 0{,}5$ entre le posterior et le prior (3 décimales).
e) Calcule $P(\text{équilibrée} \mid \text{face}) \times P(\text{face}) + P(\text{équilibrée} \mid \text{pile}) \times P(\text{pile})$ (3 décimales).
f) Que retrouves-tu en e), et pourquoi ? Compare d) à la baisse due à une face en 4.1 : laquelle des deux observations est l'indice le plus fort, et pourquoi ? Compare les rapports de vraisemblance (fiche, 🧮 cotes). (réponds dans ta copie)

### Ex 4.3 — Retrouver la règle de Bayes en trois lignes ∂ ★★ ⏱️ 15 min
**Objectif :** démontrer la règle de Bayes à partir de la règle du produit, et en tirer ses formes utiles.
**Prérequis :** Ex 3.4 · fiche §4.4.1, §4.4.2 (🧮 cotes) · **Parcours :** M

1. Écris la règle du produit dans ses deux versions (∂ 3.4) et démontre que $P(H \mid O) = \frac{P(O \mid H)\,P(H)}{P(O)}$. Quelle condition faut-il sur $P(O)$ ?
2. Les hypothèses $H_1, \ldots, H_K$ s'excluent deux à deux, couvrent tous les cas et ont chacune un prior non nul. Écris $P(O)$ avec la formule des probabilités totales, puis la règle de Bayes pour $H_i$ avec ce dénominateur.
3. Montre que les posteriors $P(H_1 \mid O), \ldots, P(H_K \mid O)$ somment à 1, et que $P(H_i \mid O)$ est proportionnel à $P(O \mid H_i)\,P(H_i)$, avec un coefficient qui ne dépend pas de $i$.
4. Démontre la forme « cotes » (quand $P(O \mid H_2)\,P(H_2) > 0$) : $\frac{P(H_1 \mid O)}{P(H_2 \mid O)} = \frac{P(O \mid H_1)}{P(O \mid H_2)} \times \frac{P(H_1)}{P(H_2)}$. Applique-la à l'exercice 4.1 : retrouve $P(\text{truquée} \mid \text{face})$ sans calculer l'évidence.
5. Montre que si $P(O \mid H_i)$ a la même valeur $q > 0$ pour toutes les hypothèses, le posterior est égal au prior. Interprète ce résultat.

### Ex 4.4 — Vie extraterrestre : lire la sonde avec Bayes ✏️ ★★ ⏱️ 20 min
**Objectif :** passer des taux d'une sonde, mesurés sur un test, aux probabilités qui servent à décider, grâce au prior.
**Prérequis :** Ex 4.1, Ex 3.2 · fiche §4.5 · **Parcours :** R, M

Une nouvelle sonde, Argos-2, a été testée sur 2 000 planètes dont on savait si elles étaient habitées :

| | la sonde détecte la vie | la sonde ne détecte rien | total |
|---|---|---|---|
| **planète habitée** | 240 | 10 | 250 |
| **planète stérile** | 105 | 1 645 | 1 750 |

Dans la région où tu vas l'utiliser, l'expérience montre que 5 % des planètes sont habitées.

a) La sensibilité (recall) $P(\text{détecté} \mid \text{habitée})$ (2 décimales).
b) La spécificité $P(\text{rien} \mid \text{stérile})$ (2 décimales).
c) Sur une nouvelle planète de la région, la sonde ne détecte rien : $P(\text{habitée} \mid \text{rien})$, exprimée en millièmes, c'est-à-dire $P \times 1\,000$ (2 décimales).
d) Sur une autre planète de la région, elle détecte la vie : $P(\text{habitée} \mid \text{détecté})$ (3 décimales).
e) La precision de la sonde, mesurée sur les 2 000 planètes du test (3 décimales).
f) Sur 10 000 planètes de la région, combien de fausses alertes (planètes stériles où la sonde détecte la vie) attend-on ?
g) Pourquoi d) et e) diffèrent-elles, alors que c'est la même sonde ? Que faudrait-il pour qu'elles soient égales ? (réponds dans ta copie)

### Ex 4.5 — Deux faces : une mise à jour double ou deux simples ? ✏️ ★★ ⏱️ 20 min
**Objectif :** vérifier que deux mises à jour successives donnent le même posterior qu'une seule mise à jour sur les deux lancers.
**Prérequis :** Ex 4.1 · fiche §4.6, §4.6.1 · **Fil rouge :** synthétique · **Parcours :** M

Mêmes pièces qu'en 4.1 : équilibrée, ou truquée de biais 0,75, avec un prior de 0,5 chacune. On lance la pièce deux fois : face, puis face.

a) **En deux temps** : le posterior de 4.1 devient le prior du second lancer. Calcule $P(\text{équilibrée} \mid \text{face, face})$ (3 décimales).
b) **D'un coup** : la vraisemblance des deux faces sous l'hypothèse « truquée », $P(\text{face, face} \mid \text{truquée})$ (4 décimales).
c) L'évidence $P(\text{face, face})$ (3 décimales).
d) $P(\text{équilibrée} \mid \text{face, face})$ calculé d'un coup (3 décimales).
e) Un troisième lancer donne pile : $P(\text{équilibrée} \mid \text{face, face, pile})$ (3 décimales).
f) Même question si les trois lancers étaient arrivés dans l'ordre pile, face, face (3 décimales).
g) Pourquoi les deux méthodes donnent-elles le même résultat, et pourquoi l'ordre ne compte-t-il pas ? Quelle hypothèse sur les lancers utilises-tu ? (réponds dans ta copie)

### Ex 4.6 — Cinq hypothèses de biais après face, pile, face ✏️ ★★ ⏱️ 20 min
**Objectif :** mettre à jour une distribution sur cinq hypothèses, lancer après lancer.
**Prérequis :** Ex 4.5 · fiche §4.7 · **Fil rouge :** synthétique · **Parcours :** M

L'archéologue du livre pense que les pièces très truquées sont rares. Pour une pièce de son coffre, elle retient cinq hypothèses sur le biais, $\theta \in \{0 ;\ 0{,}25 ;\ 0{,}5 ;\ 0{,}75 ;\ 1\}$, avec le prior $[0{,}1 ;\ 0{,}2 ;\ 0{,}4 ;\ 0{,}2 ;\ 0{,}1]$. On lance la pièce : face, puis pile, puis face. Donne les listes dans l'ordre des $\theta$ croissants.

a) Le posterior après la première face (liste, 2 décimales).
b) Le posterior après face puis pile (liste, 3 décimales).
c) Le posterior après face, pile, face (liste, 3 décimales).
d) L'évidence de la suite entière, $P(\text{face, pile, face})$, sous ce prior (4 décimales).
e) L'hypothèse MAP après les trois lancers : la valeur de $\theta$ de plus grand posterior.
f) Quelles hypothèses sont éliminées pour toujours, et à quel lancer ? Refais le calcul avec un prior uniforme : l'hypothèse MAP change-t-elle ? Qu'en conclus-tu sur le poids du prior quand on a peu de données ? (réponds dans ta copie)

### Ex 4.7 — Le posterior reste une distribution, un prior nul reste nul ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer les propriétés qui rendent la boucle posterior → prior correcte et stable.
**Prérequis :** Ex 4.3 · fiche §4.6.1, §4.7, au-delà du livre (1) · **Parcours :** M

1. Les priors sont positifs ou nuls et somment à 1, et $P(O) > 0$ (les vraisemblances $P(O \mid H_i)$ sont fixées par le modèle, même quand $P(H_i) = 0$). Montre que les posteriors sont positifs ou nuls et somment à 1.
2. Montre qu'une hypothèse de prior nul garde un posterior nul après n'importe quel nombre d'observations. Qu'en déduis-tu pour le choix d'un prior ?
3. Montre qu'une hypothèse sous laquelle l'observation est impossible (vraisemblance nulle) reçoit un posterior nul, et le garde ensuite. Donne un exemple avec les cinq hypothèses du §4.7.
4. Les observations $o_1, \ldots, o_n$ sont indépendantes sachant chaque hypothèse. Démontre par récurrence sur $n$ que la boucle donne $P(H_i \mid o_1, \ldots, o_n) = \frac{P(H_i)\prod_{k=1}^{n} P(o_k \mid H_i)}{\sum_j P(H_j)\prod_{k=1}^{n} P(o_k \mid H_j)}$. Déduis-en que l'ordre des observations ne compte pas.
5. Écris le logarithme du numérateur de la question 4. Pourquoi un ordinateur préfère-t-il cette forme quand $n$ vaut plusieurs milliers ?
6. On ajoute la même constante $c$ à tous les logarithmes de la question 5 avant de revenir aux probabilités et de normaliser (astuce log-sum-exp). Montre que les posteriors obtenus ne changent pas.

### Ex 4.8 — Combien de sondes pour descendre sous un sur un million ? ✏️ ★★ ⏱️ 25 min
**Objectif :** enchaîner des mises à jour avec la forme « cotes », puis discuter l'hypothèse d'indépendance.
**Prérequis :** Ex 4.4, Ex 4.5 · fiche §4.4.2 (🧮 cotes), §4.6.1 · **Parcours :** M

On reprend la sonde Argos-2 de l'exercice 4.4 (sa sensibilité et sa spécificité sont tes réponses à 4.4 a et b), dans la région où 5 % des planètes sont habitées. Le règlement n'autorise l'exploitation d'une planète que si $P(\text{habitée} \mid \text{résultats des sondes}) < 10^{-6}$ (un sur un million). On suppose les sondes **indépendantes sachant l'état de la planète** : chaque sonde se trompe ou non indépendamment des autres.

a) La cote a priori $\frac{P(\text{habitée})}{P(\text{stérile})}$ (4 décimales).
b) Par combien une sonde qui ne détecte rien **divise**-t-elle cette cote (1 décimale) ?
c) $P(\text{habitée} \mid \text{2 sondes négatives})$, exprimée en millionièmes, c'est-à-dire $P \times 10^6$ (3 décimales ; garde la cote de a) et le facteur de b) en valeurs exactes, des fractions, sans les arrondir).
d) Le nombre minimal de sondes négatives pour passer sous un sur un million.
e) Deux sondes détectent la vie : $P(\text{habitée} \mid \text{2 sondes positives})$ (3 décimales).
f) Une sonde détecte la vie, puis une autre ne détecte rien : $P(\text{habitée} \mid \text{positive, puis négative})$ (3 décimales).
g) Pourquoi le résultat de f) n'est-il pas le prior de 0,05 ? Deux sondes peuvent-elles se tromper « pour la même raison » sur une même planète ? Donne un exemple. Dans ce cas, ton calcul de d) est-il trop optimiste ou trop pessimiste ? (réponds dans ta copie)

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 4.9 — La règle de Bayes sans formule, en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer la règle de Bayes simplement, avec des effectifs.
**Prérequis :** fiche §4.4.1, §4.5 · **Parcours :** R

Un ami n'a jamais fait de probabilités. Il ne comprend pas qu'un test de dépistage « fiable à 95 % » puisse se tromper la plupart du temps quand il annonce une maladie rare. Explique-le-lui en **cinq lignes au plus**. Contraintes :
- les mots « avant », « indice » et « mise à jour » ;
- un exemple chiffré avec des **effectifs** (des nombres de personnes, pas des pourcentages de pourcentages) ;
- aucune formule.

Relis-toi à voix haute, puis compare avec la réponse modèle de `05_solutions.md`.

### Ex 4.10 — Le prior est un choix : erreur du procureur et priors partiaux ⚖️ ★★ ⏱️ 20 min
**Objectif :** reconnaître l'inversion de la condition dans un raisonnement judiciaire, et discuter qui choisit un prior et comment le rendre transparent.
**Prérequis :** Ex 4.4 · fiche §4.4.2, pièges · **Parcours :** R

1. Un expert déclare : « la probabilité qu'un innocent ait, par hasard, un ADN compatible avec la trace trouvée sur les lieux est d'une sur un million. » Le procureur conclut : « il n'y a donc qu'une chance sur un million que l'accusé soit innocent. » Écris les deux probabilités conditionnelles en jeu. Laquelle l'expert a-t-il donnée ? Laquelle le procureur affirme-t-il ?
2. L'accusé a été retrouvé en comparant la trace à un fichier de 3 millions de personnes, et rien d'autre ne l'accuse. On admet que le coupable figure dans le fichier. Combien de personnes compatibles attend-on dans le fichier, coupable compris ? Que vaut alors, à peu près, $P(\text{innocent} \mid \text{compatible})$ ?
3. L'affaire Sally Clark (Royaume-Uni). En 1999, une mère est condamnée pour le meurtre de ses deux bébés. Un pédiatre avait estimé à 1 sur 8 543 la probabilité d'une mort subite du nourrisson dans une famille comme la sienne, et en avait déduit 1 sur 73 millions pour deux morts, en élevant ce nombre au carré. En 2001, la Royal Statistical Society a dénoncé deux erreurs dans ce raisonnement. Lesquelles, à ton avis ? (Indices : que suppose l'élévation au carré ? À quelle question le nombre de 1 sur 73 millions répond-il ?) La condamnation a été annulée en 2003.
4. Une banque évalue le risque de défaut de paiement d'un client avec la règle de Bayes, en prenant comme prior le taux de défaut de son quartier. Quels problèmes cela pose-t-il ?
5. Qui devrait choisir le prior d'un modèle utilisé pour prendre des décisions sur des personnes ? Comment le documenter, et comment vérifier que la décision ne dépend pas trop de ce choix ?
6. Bayésianisme « subjectif » ou « automatique » (livre §4.4.2) : lequel te semble le plus facile à défendre devant un juge ou un régulateur ? Pourquoi ?

### Ex 4.11 — VanderPlas (2014) : fréquentisme et bayésianisme 📄 ★★ ⏱️ 30 min
**Objectif :** lire un article de synthèse sur les deux écoles et en retenir les différences pratiques.
**Prérequis :** Ex 4.3 · fiche §4.2, au-delà du livre (3) · **Parcours :** complet seulement (lecture conseillée à tous)

L'article : J. VanderPlas, « Frequentism and Bayesianism: A Python-driven Primer », *Proceedings of the 13th Python in Science Conference* (SciPy 2014), p. 85-93, en accès libre : [arXiv:1411.5018](https://arxiv.org/abs/1411.5018). Lis au moins le début, jusqu'à la section sur le billard de Bayes (« Nuisance Parameters: Bayes' Billiards Game ») comprise, et la section « Confidence vs. Credibility », puis parcours la fin.

1. Comment l'article définit-il la probabilité pour un fréquentiste ? Pour un bayésien ?
2. Dans l'exemple du flux de photons, que donnent les deux approches ? Pourquoi leurs résultats sont-ils si proches ?
3. Qu'est-ce qu'un paramètre de nuisance (*nuisance parameter*) ? Dans le jeu de billard de Bayes, que donne l'approche fréquentiste naïve, et comment l'approche bayésienne traite-t-elle ce paramètre ?
4. Quelle phrase un fréquentiste peut-il dire à propos d'un intervalle de confiance à 95 % ? Et un bayésien, à propos d'une région de crédibilité à 95 % ? Que montre l'exemple de l'exponentielle tronquée de Jaynes ?
5. Quels outils Python l'article utilise-t-il pour l'approche fréquentiste, et pour l'approche bayésienne par MCMC ?
6. D'après sa conclusion, dans quelles situations chaque approche est-elle bien adaptée ?
7. Dans ce chapitre, les calculs de `mylearn.bayes` relèvent-ils de l'une ou de l'autre école ? Et le `bootstrap_ci` du ch. 2 ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 4.E1 — Expliquer la règle de Bayes avec un test médical 💼 ★★ ⏱️ 10 min
*Fiche §4.4.1, §4.5 · prérequis 4.4 · parcours R*

« Expliquez-moi la règle de Bayes avec l'exemple d'un test médical. »

### 4.E2 — Fréquentiste ou bayésien : quelle différence en pratique ? 💼 ★★ ⏱️ 10 min
*Fiche §4.2, au-delà du livre (3) · prérequis Q3 · parcours R*

« Quelle est la différence entre l'approche fréquentiste et l'approche bayésienne ? Qu'est-ce que ça change dans la façon de lire un intervalle ? »

### 4.E3 — Qu'est-ce qu'un prior et comment le choisir ? 💼 ★★ ⏱️ 10 min
*Fiche §4.4.2, §4.7 · prérequis 4.10 · parcours R*

« Qu'est-ce qu'un prior ? Comment le choisissez-vous, et que se passe-t-il s'il est mauvais ? »

### 4.E4 — Où utiliser des méthodes bayésiennes en data science ? 💼 ★★ ⏱️ 10 min
*Fiche §4.2, « Bayes dans le ML d'aujourd'hui » · parcours R*

« Concrètement, dans votre travail de data scientist, où utiliseriez-vous des méthodes bayésiennes ? Donnez deux exemples, et un cas où vous préféreriez une méthode fréquentiste. »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch04_bayes/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie `mylearn/bayes.py`. La partie 0 du notebook vérifie tes réponses aux exercices ✏️ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 4.12 | Combien de lancers pour démasquer la pièce truquée ? | 🔮 | ★ | 10 |
| 4.13 | L'estimation fréquentiste : la moyenne courante des faces | 🔬 | ★ | 15 |
| 4.14 | evidence et bayes_posterior | 🔨 | ★★ | 20 |
| 4.15 | Bayes chez les manchots : l'espèce sachant l'île | 📦 | ★★ | 20 |
| 4.16 | La boucle posterior-prior : update_discrete | 🔨 | ★★ | 30 |
| 4.17 | Reproduire les trente lancers de la figure 4.24 | 🎨 | ★★ | 25 |
| 4.18 | Le posterior qui s'évanouit : underflow | 🐛 | ★★ | 20 |
| 4.19 | La grille biais × proportion de faces | 🔬 | ★★ | 30 |
| 4.20 | Envoyer des sondes jusqu'à la décision | 🔬 | ★★ | 25 |
| 4.21 | Un prior trompeur centré sur 0,8 | 🔮 | ★★ | 20 |
| 4.22 | Le posterior continu : vérifier avec scipy.stats.beta | 📦 | ★★ | 20 |
| 4.23 | Refactoriser : du copier-coller à une fonction testée | 🛠️ | ★★ | 20 |
| 4.24 | coin_bias_posterior : 500 hypothèses en log-probabilités | 🔨 | ★★★ | 35 |
| 4.25 | Intervalle de crédibilité contre intervalle bootstrap | 🔨 | ★★★ | 35 |
| 4.26 | Le détective de pièces : vingt pièces, le moins de lancers possible | 🏆 | ★★★ | 60 |
