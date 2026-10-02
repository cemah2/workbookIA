# 8 · Entraînement et test — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 📈 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 8.Q1 — La boucle d'entraînement : prédire, comparer, corriger

<details><summary>Indice 1</summary>

Relis le §8.2 de la fiche : le pseudo-code de la boucle et le rôle de l'optimiseur, puis les deux encadrés ⚠️ qui suivent.

</details>
<details><summary>Indice 2</summary>

Pour a), que fait la boucle simplifiée du livre quand la prédiction et le label coïncident ? Pour b), de quoi a-t-on besoin pour savoir dans quel sens corriger ? Pour c), regarde le flux de test sur la figure du §8.3 de la fiche. Pour d), calcule la log loss du ch. 6 pour un e-mail bien classé avec une confiance de 0,6.

</details>
<details><summary>Indice 3</summary>

a) La boucle passe à l'exemple suivant. b) Trois ingrédients : ce que le modèle a répondu, ce qu'il aurait dû répondre, et son état actuel. c) Pendant le test, l'erreur ne remonte pas vers l'optimiseur : elle ne sert qu'à compter. d) $-\log 0{,}6 > 0$ : la loss n'est pas nulle, son gradient non plus.

</details>

### 8.Q2 — Epoch, ordre des exemples et fréquence des mises à jour

<details><summary>Indice 1</summary>

Relis la définition de l'epoch et le mini-exemple sur les mini-batches (§8.2 de la fiche).

</details>
<details><summary>Indice 2</summary>

a) Une mise à jour par exemple, et chaque epoch présente tous les exemples. b) Combien de mini-batches pleins de 64 tiennent dans 1 200 ? Que devient le reste ? d) Que pourrait apprendre un modèle qui reçoit toujours les exemples dans le même ordre, par exemple triés par classe ?

</details>
<details><summary>Indice 3</summary>

a) $1\,200 \times 5$. b) $1\,200 = 18 \times 64 + 48$ : 18 mini-batches pleins, puis un dernier de 48 exemples, qui compte aussi pour une mise à jour. c) Une epoch est un passage complet sur le jeu d'entraînement. d) Mélanger ne crée aucun exemple : cela empêche le modèle de suivre l'ordre des exemples.

</details>

### 8.Q3 — 99 % sur l'entraînement : que peut-on vraiment conclure ?

<details><summary>Indice 1</summary>

Relis le §8.2.1 de la fiche : les deux raisons pour lesquelles un score d'entraînement très haut ne garantit rien, et sa dernière phrase.

</details>
<details><summary>Indice 2</summary>

Sur quelles photos le score de 99 % a-t-il été mesuré : des photos que le modèle connaît, ou des photos nouvelles ? Un raccourci est-il certain, ou seulement possible ?

</details>
<details><summary>Indice 3</summary>

a) Ce chiffre ne dit presque rien des photos nouvelles ; « forcément » est trop fort pour B. b) Aucune formule ne lit la performance future dans les paramètres. c) Il faut la mesurer, sur des données que le modèle n'a jamais vues.

</details>

### 8.Q4 — Le renard et la neige : repérer un raccourci appris

<details><summary>Indice 1</summary>

Relis le §8.2.1 de la fiche, en particulier le mini-exemple des vélos et des trottinettes.

</details>
<details><summary>Indice 2</summary>

Quel détail accompagne **toujours** une classe dans ces photos, sans rien dire de l'animal ? Un test tiré du même lot contient-il une seule photo où ce détail et la classe ne vont pas ensemble ?

</details>
<details><summary>Indice 3</summary>

Le décor. Dans le même lot, il accompagne toujours la même classe : le raccourci réussit aussi sur le test. Pour le démasquer, il faut une photo où le décor et la classe se contredisent. d) Pense à la variété des décors pour **chaque** classe.

</details>

### 8.Q5 — La règle d'or du jeu de test

<details><summary>Indice 1</summary>

Relis le §8.3 de la fiche : les deux qualités du jeu de test, puis la liste des formes de fuites.

</details>
<details><summary>Indice 2</summary>

a) Choisir un learning rate, est-ce une décision qui construit le modèle ? c) Une moyenne et un écart-type calculés sur les données servent-ils à transformer les données d'entraînement ? d) Qu'estime le score de test ?

</details>
<details><summary>Indice 3</summary>

a) Choisir en regardant le test, c'est s'en servir pour construire le modèle (§8.4). b) Une seule fois, à la fin. c) C'est la première forme de la liste : un prétraitement calculé sur toutes les données. d) Le test estime la performance en déploiement : il doit donc ressembler aux données de déploiement.

</details>

### 8.Q6 — Fuite de données : ses formes courantes

<details><summary>Indice 1</summary>

Relis la liste des formes de fuites (§8.3 de la fiche).

</details>
<details><summary>Indice 2</summary>

Pour chaque pratique, demande-toi si une information venue du test (ou de l'avenir) influence le modèle. Deux des cinq pratiques sont au contraire de bonnes pratiques.

</details>
<details><summary>Indice 3</summary>

A empêche une fuite (des doublons des deux côtés du découpage) ; C est la règle à suivre. B : la sélection a regardé les labels de toutes les lignes. D : un même patient peut se retrouver des deux côtés. E : la date de sortie n'est pas connue au moment de l'admission.

</details>

### 8.Q7 — Pourquoi un jeu de validation en plus du test ?

<details><summary>Indice 1</summary>

Relis le §8.4 de la fiche et son mini-exemple sur 1 000 exemples.

</details>
<details><summary>Indice 2</summary>

a) Que règle-t-on avec la validation, et pourquoi pas avec le test ? b) et c) Dans la boucle de recherche, sur quel jeu chaque réglage est-il entraîné, et sur quel jeu est-il noté ?

</details>
<details><summary>Indice 3</summary>

a) Les hyperparamètres, sans toucher au test. b) 60 % de 500. c) La validation ne fait que noter les réglages.

</details>

### 8.Q8 — Le score de validation du modèle retenu est-il honnête ?

<details><summary>Indice 1</summary>

Relis, au §8.4 de la fiche, « Pourquoi ne pas annoncer le score de validation du gagnant ? », puis l'encadré 🧮 sur la loi du maximum.

</details>
<details><summary>Indice 2</summary>

Le réglage a été retenu **parce que** son score de validation était le plus haut des 40. Si les 40 réglages se valaient, que vaudrait le plus haut de leurs 40 scores ?

</details>
<details><summary>Indice 3</summary>

Trois phrases : le choix a utilisé la validation, donc le score du gagnant contient une part de chance (le maximum de 40 scores bruités) ; la validation n'est donc plus une mesure indépendante du modèle retenu ; il faut un jeu de test jamais consulté, utilisé une fois, et annoncer ce score-là.

</details>

### 8.Q9 — Validation croisée : ce qu'on moyenne, et pourquoi

<details><summary>Indice 1</summary>

Relis le §8.5 de la fiche : le pseudo-code de la validation croisée et le paragraphe qui le suit.

</details>
<details><summary>Indice 2</summary>

a) Qu'obtient-on à la fin de chaque tour : un modèle, des prédictions, un score ? b) Le modèle d'un tour a-t-il le droit d'avoir vu le fold de validation de ce tour ? c) Qu'a-t-on mis de côté **avant** de commencer ?

</details>
<details><summary>Indice 3</summary>

a) Un score par tour, mesuré sur le fold de validation ; on moyenne ces scores. b) Un modèle neuf à chaque tour (c'est le rôle de `clone`). c) La validation croisée remplace la validation ; le test reste à part.

</details>

### 8.Q10 — k-fold : combien d'entraînements, quelle taille de fold ?

<details><summary>Indice 1</summary>

Relis le §8.5.1 de la fiche, et le paragraphe « Quand $n$ n'est pas un multiple de $k$ ».

</details>
<details><summary>Indice 2</summary>

Écris $1\,003 = 5 \times q + r$. Les $r$ premiers folds ont $q + 1$ exemples, les autres $q$. Le fold 5 est-il parmi les premiers ?

</details>
<details><summary>Indice 3</summary>

$1\,003 = 5 \times 200 + 3$ : trois folds de 201, deux de 200. Au dernier tour, la validation est le fold 5, de 200 exemples, et l'on entraîne sur tout le reste. Le *leave-one-out* fait un tour par exemple.

</details>

### 8.Q11 — Les deux usages des résultats de test

<details><summary>Indice 1</summary>

Relis le §8.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un usage a lieu **avant le déploiement**, l'autre **pendant l'entraînement**. Pour b), relis l'encadré ⚠️ « Si le test déçoit, on réentraîne et on reteste » (§8.3).

</details>
<details><summary>Indice 3</summary>

Estimer la performance avant le déploiement (et la chiffrer), et choisir les hyperparamètres pendant l'entraînement. b) Le nouveau réglage a été choisi après avoir vu le score de test : chaque retour fait un peu plus du test un jeu de validation.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 8.R1 — Ch. 7 : pourquoi l'inertie seule ne permet pas de choisir k

<details><summary>Indice 1</summary>

Relis, au ch. 7 (fiche §7.5), « Combien de clusters ? ».

</details>
<details><summary>Indice 2</summary>

Avec un cluster par point, où est le centre de chaque cluster ? Avec un cluster de plus, peut-on toujours faire au moins aussi bien qu'avant ?

</details>
<details><summary>Indice 3</summary>

a) Chaque point est son propre centre. b) Garder les anciens centres et en ajouter un ne peut pas augmenter l'inertie. c) La silhouette compare la cohésion d'un cluster et sa séparation des autres : couper un vrai groupe en deux la fait baisser. d) Pense au mémoriseur du ch. 1, parfait sur ce qu'il a vu.

</details>

### 8.R2 — Ch. 5 : un pas de descente de gradient à la main

<details><summary>Indice 1</summary>

Relis, au ch. 5, la règle de mise à jour de la descente de gradient : on avance à contre-pente.

</details>
<details><summary>Indice 2</summary>

Calcule d'abord l'erreur $w_0 x - y$, puis la loss (son carré), puis la dérivée $2x(w_0 x - y)$. Garde les signes.

</details>
<details><summary>Indice 3</summary>

$w_0 x - y = 1 - 3 = -2$, d'où la loss et la dérivée $2 \times 2 \times (-2)$. Puis $w_1 = 0{,}5 - 0{,}05 \times L'(w_0)$ : retrancher un nombre négatif fait **monter** $w$. Enfin, l'erreur $w_1 x - y$ et son carré. e) Ce calcul est l'étape « mise à jour » de la boucle du §8.2.

</details>

### 8.R3 — Ch. 1 : généralisation, définition et exemple

<details><summary>Indice 1</summary>

Relis la définition de la généralisation au ch. 1 (§1.2 de la fiche) et l'exercice 1.14 (le mémoriseur).

</details>
<details><summary>Indice 2</summary>

Pense à deux modèles des ch. 1 et 7 sur les manchots : l'un apprend une règle (des centroïdes), l'autre retient les manchots qu'il a vus.

</details>
<details><summary>Indice 3</summary>

Généraliser, c'est bien prédire sur des exemples **nouveaux**, venus de la même source que ceux de l'entraînement. Exemple : le centroïde le plus proche sur les mesures du bec, contre le mémoriseur. La différence se mesure en comparant l'accuracy d'entraînement à celle d'un jeu de test mis de côté.

</details>

<a id="papier"></a>

## ✏️ ∂ 📈 Papier-crayon

### Ex 8.1 — Découper 344 manchots : hold-out, validation et folds

<details><summary>Indice 1</summary>

Relis les deux règles d'arrondi de la fiche : $\lceil t \cdot n \rceil$ pour le test (§8.3), et les tailles des folds (§8.5.1). Pour i), le mini-exemple du plus fort reste (§8.3).

</details>
<details><summary>Indice 2</summary>

c) puis d) : le test d'abord, sur les 344 manchots ; la validation ensuite, sur ceux qui restent. f) Combien de manchots restent une fois le test de c) retiré ? g) $344 = 10 \times 34 + 4$. i) Les parts exactes valent $n_c \times 69 / 344$ ; prends leurs parties entières, puis compte les manchots qui manquent.

</details>
<details><summary>Indice 3</summary>

a) $0{,}25 \times 344 = 86$ tout rond. c) $\lceil 68{,}8 \rceil$. d) $\lceil 0{,}25 \times 275 \rceil = \lceil 68{,}75 \rceil$. f) $275 = 5 \times 55$. h) Au premier tour, la validation est le fold 1, qui compte 35 manchots. i) Parts exactes : Adélie 30,49 ; Chinstrap 13,64 ; Gentoo 24,87. Parties entières : $30 + 13 + 24 = 67$ ; il manque 2 manchots, qui vont aux deux plus grandes parties décimales.

</details>

### Ex 8.2 — Compter les entraînements d'une recherche d'hyperparamètres

<details><summary>Indice 1</summary>

Relis la boucle de recherche d'hyperparamètres (§8.4 de la fiche) et l'encadré 🧮 sur la validation croisée imbriquée (§8.5).

</details>
<details><summary>Indice 2</summary>

a) Chaque valeur d'un hyperparamètre se combine avec chaque valeur des autres. c) Un entraînement par réglage et par fold. f) Que fait-on dans **un** tour extérieur ? Compte-le, puis multiplie par le nombre de tours. g) Retire d'abord du budget le réentraînement final.

</details>
<details><summary>Indice 3</summary>

a) $3 \times 4 \times 2$. c) $24 \times 5$. d) Un de plus. e) $121 \times 4$ minutes, à convertir en heures. f) Chaque tour extérieur refait tout le protocole de d) sur sa partie d'entraînement. g) Avec $r$ réglages : $5r + 1 \le 200$.

</details>

### Ex 8.3 — Fuite ou pas ? Six protocoles à auditer

<details><summary>Indice 1</summary>

Relis la liste des formes de fuites (§8.3 de la fiche) et le paragraphe « Données dépendantes » du §8.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pour chaque protocole, pose quatre questions : un calcul a-t-il utilisé des lignes de test ? Un choix a-t-il été fait en regardant le test ? Une feature sera-t-elle connue au moment de prédire ? Des exemples dépendants ont-ils été séparés au hasard ?

</details>
<details><summary>Indice 3</summary>

a) La médiane a-t-elle vu les annonces de test ? b) Sur quelles images la standardisation est-elle ajustée ? c) Sur quel jeu le réglage a-t-il été choisi ? d) En production, connaîtra-t-on les mois qui suivent celui qu'on prédit ? e) Un locuteur peut-il se retrouver des deux côtés ? f) Le nombre d'appels du mois prochain existe-t-il au moment de prédire ? Il y a quatre fuites.

</details>

### Ex 8.4 — Quelle confiance accorder à une accuracy de test ? Erreur type et taille du test

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur l'erreur type d'une accuracy (§8.3 de la fiche).

</details>
<details><summary>Indice 2</summary>

$\mathrm{SE} = \sqrt{0{,}92 \times 0{,}08 / 250}$, et la demi-largeur vaut $1{,}96\,\mathrm{SE}$. d) Écris $1{,}96 \sqrt{0{,}92 \times 0{,}08 / n} \le 0{,}01$ et isole $n$. e) $n$ est sous une racine.

</details>
<details><summary>Indice 3</summary>

a) $\sqrt{0{,}0002944}$. d) $n \ge (1{,}96 / 0{,}01)^2 \times 0{,}0736 \approx 2\,827{,}4$ : arrondis vers le haut. e) Pour diviser la demi-largeur par 3, il faut $3^2$ fois plus d'exemples. f) Compare l'écart de 0,02 à la demi-largeur trouvée en b). g) A et B sont-ils notés sur les mêmes exemples ? Lesquels réussit l'un et rate l'autre ?

</details>

### Ex 8.5 — Moyenne et écart-type de scores de validation croisée

<details><summary>Indice 1</summary>

Relis les formules de la moyenne et de l'écart-type des scores (§8.5.1 de la fiche) : on divise par $k$.

</details>
<details><summary>Indice 2</summary>

Calcule les écarts à la moyenne, leurs carrés, la moyenne de ces carrés, puis sa racine. Pour f) et g), écris d'abord la liste des différences B − A, fold par fold.

</details>
<details><summary>Indice 3</summary>

Écarts de A à sa moyenne (0,84) : −0,02 ; 0,04 ; −0,05 ; 0,01 ; 0,02. Différences B − A : 0,03 ; −0,04 ; 0,07 ; 0 ; −0,03. h) Compare la moyenne des différences à leur écart-type, et compte les folds gagnés par chacun.

</details>

### Ex 8.6 — Le biais d'optimisme du meilleur de K modèles

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la loi du maximum (§8.4 de la fiche).

</details>
<details><summary>Indice 2</summary>

1. Le maximum est inférieur à $s$ si et seulement si **chaque** $S_j$ l'est. 2. Compare, tirage par tirage, le maximum de $K + 1$ nombres et celui des $K$ premiers. b) et c) $1 - (1 - 0{,}0668)^K$. d) Résous $1 - 0{,}9332^K \ge 0{,}9$ avec des logarithmes.

</details>
<details><summary>Indice 3</summary>

2. Pour chaque tirage, $\max(S_1, \dots, S_{K+1}) \ge \max(S_1, \dots, S_K)$, et l'espérance respecte l'ordre. d) $0{,}9332^K \le 0{,}1$ donne $K \ge \ln 0{,}1 / \ln 0{,}9332 \approx 33{,}3$ (diviser par un logarithme négatif renverse l'inégalité). e) Le réglage retenu a la même accuracy réelle que les autres. 3. Pense aux exemples communs du jeu de validation, et à des réglages voisins.

</details>

### Ex 8.8 — Comparer des modèles à partir de boîtes à moustaches de scores

<details><summary>Indice 1</summary>

Relis, au ch. 2, la lecture d'une boîte à moustaches (médiane, quartiles, moustaches, points isolés), puis le §8.6 de la fiche sur la comparaison exemple par exemple.

</details>
<details><summary>Indice 2</summary>

a) et b) Compare les traits au milieu des boîtes, puis la hauteur des boîtes et l'écart entre les moustaches. d) Dans le panneau (b), compte les points sous la ligne zéro. f) Que se passe-t-il pour C dans ses plus mauvais folds ?

</details>
<details><summary>Indice 3</summary>

Le panneau (b) est **apparié** : il compare A et B sur le même fold, ce qui retire la difficulté propre à chaque fold, commune aux deux modèles. Pour le point isolé de A (environ 0,77) : dans le panneau (b), B ne dépasse jamais A de plus de 0,03 ; quel score B a-t-il pu obtenir sur ce fold-là, et lequel de ses scores est aussi bas ?

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 8.7 — Pourquoi le jeu de test reste sous clé : l'analogie de l'examen

<details><summary>Indice 1</summary>

Relis l'analogie de l'examen dans le livre (§8.3), puis le §8.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Plan en cinq lignes : à quoi sert un examen ; ce qui arrive si l'élève a vu le sujet ; pourquoi on ne regarde le test qu'une fois ; un exemple de la vie courante où « choisir sur ce qui sert à noter » flatte le résultat ; la solution (un autre jeu pour choisir).

</details>
<details><summary>Indice 3</summary>

Un examen mesure ce que l'élève sait faire sur des questions **nouvelles** ; s'il a vu le sujet, il peut **mémoriser** les réponses sans rien comprendre. Pour l'exemple courant, pense à une recette choisie parmi dix en les faisant goûter aux mêmes amis, puis à la note de ces amis annoncée comme celle que donneront des inconnus.

</details>

### Ex 8.9 — Raccourcis appris : radiographies, chars d'assaut et responsabilité

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §8.2.1 et la liste des formes de fuites (§8.3). Dans l'article de Zech et coll., regarde la proportion de pneumonies dans chaque système hospitalier.

</details>
<details><summary>Indice 2</summary>

1. Si un hôpital a beaucoup plus de pneumonies qu'un autre, que gagne un modèle qui reconnaît l'hôpital ? 2. D'où venaient les radiographies de test ? 3. Pense à `GroupKFold`, avec un groupe par hôpital. 5. Liste les personnes qui subiront une erreur, puis celles qui peuvent la prévenir.

</details>
<details><summary>Indice 3</summary>

1. Dans un des systèmes, environ un tiers des radiographies montrent une pneumonie, contre environ 1 % dans un autre : reconnaître l'hôpital (marqueurs, type d'appareil, inscriptions) donne déjà une bonne partie de la réponse. 2. Le même raccourci fonctionne sur un test tiré des mêmes hôpitaux ; c'est un test qui ne vient pas de la distribution qui intéresse vraiment (la troisième famille de Kapoor et Narayanan, 📄 8.10). 3. Une validation externe : entraîner sur certains hôpitaux, tester sur un autre, jamais vu. 5. Cherche « fiche modèle » (*model card*, Mitchell et coll., 2019).

</details>

### Ex 8.10 — Kapoor & Narayanan (2023) : une taxonomie des fuites

<details><summary>Indice 1</summary>

Prends la version publiée dans *Patterns* (en accès libre, par le lien DOI de l'énoncé) : ses chiffres diffèrent de ceux de la première version d'arXiv. La taxonomie y est résumée dans un tableau, puis détaillée type par type ; les *model info sheets* ont leur propre section, et leur modèle complet est fourni avec l'article.

</details>
<details><summary>Indice 2</summary>

1. Les trois familles portent sur la séparation entre entraînement et test, sur la légitimité des features, et sur la distribution du jeu de test. 2. Pour chaque protocole fautif de 8.3, demande-toi lequel de ces trois points est violé. 4. Cherche la section sur la prédiction des guerres civiles : combien d'articles ont été examinés, lesquels contenaient une erreur, et de quels types.

</details>
<details><summary>Indice 3</summary>

2. a) une imputation sur toutes les données ; c) un choix fait sur le test (la taxonomie n'a pas de case dédiée : rapproche-le de la première famille) ; d) le temps ; f) une feature. 3. Compte les questions du modèle de fiche, et regarde les trois arguments qu'elles demandent. 4. Quatre articles sur douze contenaient une erreur : que prétendaient-ils, et que devient l'avantage des modèles complexes sur la régression logistique une fois les fuites corrigées ?

</details>

<a id="entretien"></a>

## 💼 Entretien

### 8.E1 — Pourquoi trois jeux : entraînement, validation et test ?

<details><summary>Indice 1</summary>

Relis le §8.4 de la fiche et son encadré 💼.

</details>
<details><summary>Indice 2</summary>

Un rôle par jeu, puis ce qui arrive si l'on règle les hyperparamètres sur le test (∂ 8.6, 🔮 8.17).

</details>
<details><summary>Indice 3</summary>

Entraînement : les paramètres. Validation : les hyperparamètres et l'arrêt de l'entraînement. Test : une estimation honnête, une seule fois. Régler sur le test rend son score optimiste (le meilleur de $K$) et ne laisse plus rien pour mesurer. Ajoute la validation croisée, le réentraînement final et la validation croisée imbriquée.

</details>

### 8.E2 — Qu'est-ce qu'une p-valeur ? Comment savoir si le modèle B bat vraiment le modèle A ?

<details><summary>Indice 1</summary>

Relis le §8.6 de la fiche, l'encadré 🧮 sur le test par permutation, puis 🔬 8.26.

</details>
<details><summary>Indice 2</summary>

0,7 point sur 2 000 exemples, c'est 14 réussites de plus. Qu'est-ce qui décide vraiment : le nombre total de réussites, ou les exemples sur lesquels A et B ne sont pas d'accord ?

</details>
<details><summary>Indice 3</summary>

La définition de la p-valeur (la probabilité, si A et B se valent, d'un écart au moins aussi grand) ; un test apparié sur les désaccords (McNemar, ou la permutation des réponses de A et de B exemple par exemple) ; un intervalle par bootstrap apparié ; puis la significativité pratique (0,7 point vaut-il un changement de modèle ?) et la règle « le test ne sert qu'une fois ».

</details>

### 8.E3 — Validation croisée ou simple hold-out : quand choisir quoi ?

<details><summary>Indice 1</summary>

Relis le §8.5 de la fiche et ses deux encadrés 🕰️ (le deep learning, les découpeurs de scikit-learn).

</details>
<details><summary>Indice 2</summary>

Trois critères : la taille des données, le coût d'un entraînement, la structure des données (groupes, temps).

</details>
<details><summary>Indice 3</summary>

Petit dataset tabulaire et modèle rapide : k-fold (5 ou 10), stratifiée, éventuellement répétée. Gros dataset ou entraînement long (deep learning) : un seul jeu de validation. Données groupées ou temporelles : `GroupKFold`, `TimeSeriesSplit`. Un score à annoncer sans test séparé : la validation croisée imbriquée. Dans tous les cas, un test final.

</details>

### 8.E4 — 99 % en test, échec en production : vos hypothèses

<details><summary>Indice 1</summary>

Relis le §8.2.1 (raccourcis), le §8.3 (fuites et représentativité du test) et le §8.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

Deux grandes familles d'hypothèses : le score de test était faux (une fuite, un test qui ne ressemble pas à la production), ou les données de production ont changé. Ajoute les bugs de la mise en production.

</details>
<details><summary>Indice 3</summary>

Dans l'ordre, du moins coûteux au plus coûteux à vérifier : mesure-t-on la même chose (métrique, seuil, prétraitement identiques en production) ? Y avait-il une fuite (doublons, prétraitement sur tout, feature indisponible au moment de prédire, temps) ? Les distributions des features et des classes ont-elles changé ? Le modèle s'appuie-t-il sur un raccourci (analyse par sous-groupe, exemples mal classés) ? Puis corriger, et surveiller.

</details>

### 8.E5 — Stratifier un découpage : quand et pourquoi ?

<details><summary>Indice 1</summary>

Relis « Stratifier » (§8.3) et « Stratifier les folds » (§8.5.1) dans la fiche, puis 8.11 et 8.21.

</details>
<details><summary>Indice 2</summary>

Que devient, sans stratification, une classe à 2 % dans un petit jeu de test ? La stratification protège-t-elle contre des exemples dépendants, ou contre un changement des données en production ?

</details>
<details><summary>Indice 3</summary>

La définition (chaque partie garde les proportions des classes) ; indispensable avec des classes rares, de petits datasets ou beaucoup de folds ; insuffisante face aux groupes (`StratifiedGroupKFold`), au temps (`TimeSeriesSplit`) et aux changements de distribution. En régression, on peut stratifier sur des tranches de la cible.

</details>

<a id="notebook"></a>

## Notebook, parties A à D

Les exercices du notebook (`03_notebook.ipynb`). Pour ceux qui complètent `mylearn/model_selection.py`, la docstring de chaque fonction décrit déjà ce qu'elle doit faire, cas d'erreur compris : relis-la avant d'ouvrir un indice.

### Ex 8.11 — train_test_split de scikit-learn : tailles, stratify, random_state 📦

<details><summary>Indice 1</summary>

Tout se fait avec `sklearn_train_test_split` et des comptages. Repère d'abord, parmi les tableaux qu'elle renvoie, celui qui contient les espèces des manchots de test.

</details>
<details><summary>Indice 2</summary>

La fonction renvoie `X_train, X_test, y_train, y_test`. Pour compter les Chinstrap d'un tableau de labels, compare-le à `"Chinstrap"` et additionne. c) Ajoute `stratify=species`. d) `shuffle=False` (sans `random_state`), puis `np.unique` sur les labels de test. e) Une compréhension de liste sur `range(100)`, puis `min` et `max`. Convertis les comptes en `int`.

</details>
<details><summary>Indice 3</summary>

```python
X_tr_11, X_te_11, y_tr_11, y_te_11 = sklearn_train_test_split(X_peng, species, test_size=0.2, random_state=42)
n_test_11a = len(X_te_11)
chinstrap_11b = int(np.sum(y_te_11 == "Chinstrap"))
y_strat_11 = sklearn_train_test_split(X_peng, species, test_size=0.2, random_state=42, stratify=species)[3]
chinstrap_11c = int(np.sum(y_strat_11 == "Chinstrap"))
y_ordered_11 = sklearn_train_test_split(X_peng, species, test_size=0.2, shuffle=False)[3]
n_species_11d = len(np.unique(y_ordered_11))
counts_11 = [int(np.sum(sklearn_train_test_split(X_peng, species, test_size=0.2, random_state=seed)[3] == "Chinstrap"))
             for seed in range(100)]
range_11e = [min(counts_11), max(counts_11)]
```

</details>

### Ex 8.12 — Le modèle qui apprend par cœur : accuracy d'entraînement et de test 🔮

<details><summary>Indice 1</summary>

Relis comment `OneNN` prédit (ch. 7), puis demande-toi ce que « apprendre par cœur » change pour des exemples déjà vus et pour des exemples nouveaux.

</details>
<details><summary>Indice 2</summary>

Pour un manchot d'entraînement, quel est le manchot d'entraînement le plus proche ? Son label, même tiré au hasard, est-il dans la mémoire du modèle ? Sur le test, une fois les labels mélangés, quel lien reste-t-il entre les mesures d'un manchot et son label ?

</details>
<details><summary>Indice 3</summary>

Sur l'entraînement, le plus proche voisin de chaque manchot est lui-même : la réponse ne dépend pas de la vérité des labels. Sur le test, avec les vraies espèces, souviens-toi des accuracies des ch. 1 et 7 avec les quatre mesures. Avec des labels mélangés, copier le label d'un voisin revient à tirer un label au hasard : avec des espèces qui représentent 44 %, 36 % et 20 % des manchots, on tombe juste avec une probabilité d'environ $0{,}44^2 + 0{,}36^2 + 0{,}20^2 \approx 0{,}36$ (l'expérience la calcule avec les manchots d'entraînement : 0,37).

</details>

### Ex 8.13 — train_test_split from scratch 🔨

<details><summary>Indice 1</summary>

Organise ta fonction en trois temps : les contrôles et le calcul de $n_{\text{test}}$ ; le choix d'**un** tableau d'indices de test et d'**un** tableau d'indices d'entraînement (trois cas : stratifié, mélangé, dernières lignes) ; une boucle qui indexe chaque tableau avec ces deux tableaux d'indices.

</details>
<details><summary>Indice 2</summary>

Un `float` : vérifie $0 < t < 1$, puis `math.ceil(test_size * n)` ; un `int` : `n_test = test_size` ; dans les deux cas, il faut `0 < n_test < n`. Attention, `True` est aussi un `int` en Python. Mélangé : `perm = rng.permutation(n)`, le test est `perm[:n_test]`. Stratifié : `_, codes, counts = np.unique(stratify, return_inverse=True, return_counts=True)` ; les quotas par classe viennent de `np.floor(counts * n_test / n)`, plus un pour les plus grandes parties décimales (`np.argsort(-(exact - quotas), kind="stable")[:missing]`) ; puis, pour chaque classe `c`, `rng.permutation(np.flatnonzero(codes == c))[:quota]`. Tout le reste va à l'entraînement.

</details>
<details><summary>Indice 3</summary>

```python
def train_test_split(*arrays, test_size=0.25, shuffle=True, stratify=None, rng=None):
    if not arrays:
        raise ValueError("at least one array is needed")
    arrays = [np.asarray(a) for a in arrays]
    if any(a.ndim == 0 for a in arrays):
        raise ValueError("every array needs one row per sample")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError(f"all arrays must have the same length, got {[len(a) for a in arrays]}")
    if isinstance(test_size, (bool, np.bool_)):
        raise ValueError("test_size must be a float or an int, not a bool")
    if isinstance(test_size, (int, np.integer)):
        n_test = int(test_size)
    elif isinstance(test_size, (float, np.floating)) and 0 < test_size < 1:
        n_test = math.ceil(test_size * n)
    else:
        raise ValueError(f"invalid test_size: {test_size!r}")
    if not 0 < n_test < n:
        raise ValueError(f"{n_test} test samples out of {n}: both parts must be non-empty")
    if rng is None:
        rng = np.random.default_rng()
    if stratify is not None:
        if not shuffle:
            raise ValueError("stratify requires shuffle=True")
        labels = np.asarray(stratify)
        if len(labels) != n:
            raise ValueError("stratify must have one label per row")
        _, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if counts.min() < 2:
            raise ValueError("every class of stratify needs at least 2 members")
        exact = counts * n_test / n
        quotas = np.floor(exact).astype(int)
        missing = n_test - quotas.sum()
        quotas[np.argsort(-(exact - quotas), kind="stable")[:missing]] += 1
        test_idx = np.concatenate([rng.permutation(np.flatnonzero(codes == c))[:quota]
                                   for c, quota in enumerate(quotas)])
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        test_idx, train_idx = rng.permutation(test_idx), rng.permutation(train_idx)
    elif shuffle:
        perm = rng.permutation(n)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
    else:
        train_idx, test_idx = np.arange(n - n_test), np.arange(n - n_test, n)
    result = []
    for a in arrays:
        result += [a[train_idx], a[test_idx]]
    return result
```

</details>

### Ex 8.14 — Les indices de la k-fold : kfold_indices 🔨

<details><summary>Indice 1</summary>

Commence par les contrôles, puis calcule la taille de chaque fold ; ensuite, une boucle qui avance d'un fold à l'autre.

</details>
<details><summary>Indice 2</summary>

`sizes = np.full(n_splits, n_samples // n_splits)`, puis `sizes[: n_samples % n_splits] += 1`. Avec `indices = np.arange(n_samples)` (ou `rng.permutation(n_samples)`), le fold courant est `indices[start:stop]` ; l'entraînement, la concaténation de ce qui est avant et de ce qui est après. Trie chaque partie avec `np.sort`. `np.cumsum(sizes)` donne les bornes des folds.

</details>
<details><summary>Indice 3</summary>

```python
def kfold_indices(n_samples, n_splits=5, shuffle=False, rng=None):
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError(f"n_splits must be between 2 and n_samples={n_samples}, got {n_splits}")
    if shuffle:
        indices = (rng if rng is not None else np.random.default_rng()).permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    sizes = np.full(n_splits, n_samples // n_splits)
    sizes[: n_samples % n_splits] += 1
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    splits = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        val_idx = np.sort(indices[start:stop])
        train_idx = np.sort(np.concatenate([indices[:start], indices[stop:]]))
        splits.append((train_idx, val_idx))
    return splits
```

</details>

### Ex 8.15 — Reproduire la figure 8.13 : la rotation des folds 🎨

<details><summary>Indice 1</summary>

Une ligne par tour, une colonne par échantillon : que doit valoir la case de la ligne `i` et de la colonne `j` ?

</details>
<details><summary>Indice 2</summary>

Pars d'une matrice de zéros de forme `(len(splits), n)`, puis, pour chaque tour, mets des 1 aux indices de validation : `matrix = np.zeros((len(splits), n), dtype=int)`, puis `matrix[i, val_idx] = 1` dans une boucle `for i, (_, val_idx) in enumerate(splits)` (indexation par un tableau d'indices, 0A). Pour le dessin : `ax.imshow(matrix, cmap=ListedColormap(["tab:blue", "tab:orange"]), aspect="auto", vmin=0, vmax=1)`, puis les graduations et le titre.

</details>
<details><summary>Indice 3</summary>

```python
def fold_matrix_15(splits, n):
    matrix = np.zeros((len(splits), n), dtype=int)
    for i, (_, val_idx) in enumerate(splits):
        matrix[i, val_idx] = 1
    return matrix


def draw_folds_15(ax, splits, n, title):
    from matplotlib.colors import ListedColormap
    ax.imshow(fold_matrix_15(splits, n), cmap=ListedColormap(["tab:blue", "tab:orange"]),
              aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(splits)), [f"round {i + 1}" for i in range(len(splits))])
    ax.set_xticks(range(0, n, 5))
    ax.set_xlabel("sample")
    ax.set_title(title, fontsize=10)
```

</details>

### Ex 8.16 — Un estimateur maison à la scikit-learn : PolyFit(degree) 🔨

<details><summary>Indice 1</summary>

Quatre méthodes courtes. `__init__` tient en une ligne ; `fit` en deux, plus `return self` ; `score` applique la formule du $R^2$ de la fiche (§8.4).

</details>
<details><summary>Indice 2</summary>

`np.polyfit(x, y, degree)` renvoie les coefficients, du plus haut degré au plus bas ; `np.polyval(coef, x)` évalue le polynôme en chaque `x`. Dans `score` : $SS_{\text{res}} = \sum (y - \hat{y})^2$, $SS_{\text{tot}} = \sum (y - \bar{y})^2$, et l'on renvoie `float(1 - SS_res / SS_tot)`.

</details>
<details><summary>Indice 3</summary>

```python
class PolyFit:
    """Polynomial regression of y on the single feature X[:, 0], fitted by least squares (np.polyfit)."""

    def __init__(self, degree=1):
        self.degree = degree

    def fit(self, X, y):
        x = np.asarray(X, dtype=float)[:, 0]
        self.coef_ = np.polyfit(x, np.asarray(y, dtype=float), self.degree)
        return self

    def predict(self, X):
        return np.polyval(self.coef_, np.asarray(X, dtype=float)[:, 0])

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot)
```

</details>

### Ex 8.17 — Validation ou test : lequel sera le plus optimiste ? 🔮

<details><summary>Indice 1</summary>

Relis ∂ 8.6 : que vaut, en moyenne, le meilleur de douze scores bruités ?

</details>
<details><summary>Indice 2</summary>

a) Le degré retenu a-t-il été choisi indépendamment de son score de validation ? b) Pour un degré fixé d'avance, la validation et le test sont deux mesures du même modèle sur des districts différents : l'une a-t-elle une raison d'être plus haute que l'autre ? c) Sur 20 districts, un $R^2$ est très bruité : le degré qui gagne est-il le meilleur, ou le plus chanceux ?

</details>
<details><summary>Indice 3</summary>

a) Le maximum de douze scores bruités contient une part de chance, qui disparaît sur le test. b) Sans sélection, chaque mesure est honnête : l'une dépasse l'autre à peu près une fois sur deux. c) Avec si peu de validation, un degré élevé qui a eu de la chance est souvent retenu, alors qu'il extrapole mal sur les 300 districts de test.

</details>

### Ex 8.18 — Boucle de sélection sur un jeu de validation : le degré du polynôme 🔨

<details><summary>Indice 1</summary>

Suis les cinq étapes de l'énoncé, dans l'ordre : chacune tient en une ou deux lignes.

</details>
<details><summary>Indice 2</summary>

Deux appels à `mylearn.model_selection.train_test_split(..., shuffle=False)`, avec `test_size=0.2` puis `0.25` ; une boucle sur `degrees` qui crée un `PolyFit(degree=d)` **neuf**, l'entraîne sur l'entraînement et ajoute son $R^2$ de validation à une liste ; `degrees[int(np.argmax(val_scores))]` ; un dernier `PolyFit` entraîné sur `X_rest, y_rest`, noté une fois sur le test.

</details>
<details><summary>Indice 3</summary>

```python
def select_degree_18(X, y, degrees):
    split = mylearn.model_selection.train_test_split
    X_rest, X_test, y_rest, y_test = split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = split(X_rest, y_rest, test_size=0.25, shuffle=False)
    val_scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # high degrees: np.polyfit warns
        for degree in degrees:
            model = PolyFit(degree=degree)
            model.fit(X_train, y_train)
            val_scores.append(model.score(X_val, y_val))
        best_degree = degrees[int(np.argmax(val_scores))]
        final = PolyFit(degree=best_degree)
        final.fit(X_rest, y_rest)                    # training + validation
    return val_scores, best_degree, final.score(X_test, y_test)
```

</details>

### Ex 8.19 — Données dépendantes : GroupKFold et TimeSeriesSplit 📦

<details><summary>Indice 1</summary>

Les trois découpeurs sont déjà importés : `KFold`, `GroupKFold`, `TimeSeriesSplit`. Tu ne fais que les **créer**, avec leurs paramètres ; c'est `cross_val_score` qui s'en sert.

</details>
<details><summary>Indice 2</summary>

`KFold` prend `n_splits`, `shuffle` et `random_state`. La décennie d'une année s'obtient par une division entière par 10, puis une multiplication par 10 ; `years_19` est un tableau NumPy, l'opération s'applique à tous ses éléments d'un coup.

</details>
<details><summary>Indice 3</summary>

```python
cv_shuffle_19 = KFold(n_splits=5, shuffle=True, random_state=0)
groups_19 = years_19 // 10 * 10
cv_groups_19 = GroupKFold(n_splits=5)
cv_time_19 = TimeSeriesSplit(n_splits=5)
```

</details>

### Ex 8.20 — Écrire tes propres tests pytest pour train_test_split 🛠️

<details><summary>Indice 1</summary>

Un test est une fonction `test_...()` sans argument, qui appelle `train_test_split(...)` (le nom que la vérification donne à chaque version testée) et vérifie **une** propriété avec `assert`. Écris un test par propriété de l'énoncé.

</details>
<details><summary>Indice 2</summary>

Taille : avec $n = 10$ et $t = 0{,}25$, il faut 3 lignes de test, pas 2. Disjoint et complet : `sorted(np.concatenate([train, test]).tolist())` doit valoir `list(range(n))`. Alignement : construis `y` à partir de `X` (par exemple `y = X[:, 0] * 10`) et vérifie que la relation tient dans chaque partie. Même graine : deux appels avec `rng=np.random.default_rng(3)`. Sans mélange : les dernières lignes en test.

</details>
<details><summary>Indice 3</summary>

```python
def test_sizes():
    for n, test_size, n_test in [(10, 0.25, 3), (8, 0.25, 2), (10, 3, 3)]:
        train, test = train_test_split(np.arange(n), test_size=test_size, rng=np.random.default_rng(0))
        assert len(test) == n_test and len(train) == n - n_test


def test_disjoint_and_complete():
    train, test = train_test_split(np.arange(50), test_size=0.3, rng=np.random.default_rng(1))
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(50))


def test_rows_stay_aligned():
    X = np.arange(40).reshape(20, 2)
    y = X[:, 0] * 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, rng=np.random.default_rng(2))
    assert np.array_equal(X_train[:, 0] * 10, y_train) and np.array_equal(X_test[:, 0] * 10, y_test)


def test_same_seed_same_split():
    first = train_test_split(np.arange(30), rng=np.random.default_rng(3))
    second = train_test_split(np.arange(30), rng=np.random.default_rng(3))
    assert np.array_equal(first[1], second[1])


def test_last_rows_without_shuffle():
    train, test = train_test_split(np.arange(10), test_size=0.3, shuffle=False)
    assert test.tolist() == [7, 8, 9] and train.tolist() == list(range(7))


TESTS_20 = [test_sizes, test_disjoint_and_complete, test_rows_stay_aligned, test_same_seed_same_split,
            test_last_rows_without_shuffle]
```

</details>

### Ex 8.21 — k-fold stratifiée : stratified_kfold_indices 🔨

<details><summary>Indice 1</summary>

Trois étapes : l'ordre des indices trié par classe, la distribution « une carte par fold », puis, pour chaque fold, l'entraînement formé de tous les autres indices. Refais d'abord à la main le mini-exemple de la fiche (§8.5.1).

</details>
<details><summary>Indice 2</summary>

`_, codes, counts = np.unique(y, return_inverse=True, return_counts=True)`. Sans mélange : `order = np.argsort(codes, kind="stable")`. Avec mélange : la concaténation, classe par classe, de `rng.permutation(np.flatnonzero(codes == c))`. Le fold `i` : `np.sort(order[i::n_splits])` ; l'entraînement : `np.setdiff1d(np.arange(n), val_idx)`, déjà trié. L'avertissement : `warnings.warn("...", UserWarning)` si `counts.min() < n_splits`.

</details>
<details><summary>Indice 3</summary>

```python
def stratified_kfold_indices(y, n_splits=5, shuffle=False, rng=None):
    y = np.asarray(y)
    n = len(y)
    if n_splits < 2 or n_splits > n:
        raise ValueError(f"n_splits must be between 2 and the number of samples ({n}), got {n_splits}")
    _, codes, counts = np.unique(y, return_inverse=True, return_counts=True)
    if counts.min() < n_splits:
        warnings.warn(f"a class has only {counts.min()} members, fewer than n_splits={n_splits}", UserWarning)
    if shuffle:
        rng = rng if rng is not None else np.random.default_rng()
        order = np.concatenate([rng.permutation(np.flatnonzero(codes == c)) for c in range(len(counts))])
    else:
        order = np.argsort(codes, kind="stable")
    splits = []
    for i in range(n_splits):
        val_idx = np.sort(order[i::n_splits])
        splits.append((np.setdiff1d(np.arange(n), val_idx), val_idx))
    return splits
```

</details>

### Ex 8.22 — clone et cross_val_score 🔨

<details><summary>Indice 1</summary>

`clone` tient en deux lignes (fiche §8.5.1, « Cloner un modèle »). `cross_val_score` : préparer la liste des paires d'indices, puis une boucle « cloner, entraîner, noter ».

</details>
<details><summary>Indice 2</summary>

`vars(estimator).items()`, filtré avec `name.startswith("_")` et `name.endswith("_")`. Pour `cv` : un entier (mais pas un booléen) donne `kfold_indices(len(X), cv)`, sinon `list(cv)`. Dans la boucle : `model = clone(estimator)`, `model.fit(X[train_idx], y[train_idx])`, puis `scoring(model, X[val_idx], y[val_idx])` ou `model.score(...)`. À la fin, `np.asarray(scores, dtype=float)`. Les deux `ValueError` se lèvent **avant** la boucle.

</details>
<details><summary>Indice 3</summary>

```python
def clone(estimator):
    params = {name: copy.deepcopy(value) for name, value in vars(estimator).items()
              if not name.startswith("_") and not name.endswith("_")}
    return type(estimator)(**params)


def cross_val_score(estimator, X, y, cv=5, scoring=None):
    X, y = np.asarray(X), np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} values")
    if scoring is None and not callable(getattr(estimator, "score", None)):
        raise ValueError("the estimator has no score method: pass a scoring function")
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        splits = kfold_indices(len(X), int(cv))
    else:
        splits = list(cv)
    scores = []
    for train_idx, val_idx in splits:
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        if scoring is None:
            scores.append(model.score(X[val_idx], y[val_idx]))
        else:
            scores.append(scoring(model, X[val_idx], y[val_idx]))
    return np.asarray(scores, dtype=float)
```

</details>

### Ex 8.23 — Variabilité de l'évaluation : hold-out répétés contre k-fold 🔬

<details><summary>Indice 1</summary>

Deux boucles sur les graines, de 0 à `n_repeats - 1` ; dans chacune, un découpage, puis un score.

</details>
<details><summary>Indice 2</summary>

`mylearn.model_selection.train_test_split(X, y, test_size=0.25, rng=np.random.default_rng(seed))` renvoie quatre tableaux ; `NearestCentroid().fit(X_train, y_train).score(X_test, y_test)` donne l'accuracy. Pour la 5-fold : `kfold_indices(len(X), 5, shuffle=True, rng=np.random.default_rng(seed))`, puis la moyenne de `cross_val_score(NearestCentroid(), X, y, cv=folds)`.

</details>
<details><summary>Indice 3</summary>

```python
def holdout_scores_23(X, y, n_repeats):
    scores = []
    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = mylearn.model_selection.train_test_split(
            X, y, test_size=0.25, rng=np.random.default_rng(seed))
        scores.append(NearestCentroid().fit(X_train, y_train).score(X_test, y_test))
    return np.array(scores)


def cv_means_23(X, y, n_repeats):
    means = []
    for seed in range(n_repeats):
        folds = mylearn.model_selection.kfold_indices(len(X), 5, shuffle=True, rng=np.random.default_rng(seed))
        means.append(np.mean(mylearn.model_selection.cross_val_score(NearestCentroid(), X, y, cv=folds)))
    return np.array(means)
```

</details>

### Ex 8.24 — Un notebook trop beau pour être vrai : quatre fuites à corriger 🐛

<details><summary>Indice 1</summary>

Pour chaque étape, demande-toi de quelle information elle profite, et si le modèle l'aurait dans l'usage réel. Puis suis, pour `honest_24`, les quatre points du protocole de l'énoncé.

</details>
<details><summary>Indice 2</summary>

Pour chacune des six étapes, pose deux questions : sur quelles lignes travaille-t-elle (toutes, ou l'entraînement seulement) ? Et que se passe-t-il pour un manchot dont des copies se retrouvent des deux côtés du découpage ? Regarde aussi sur quoi le modèle final est choisi. Pour `honest_24` : `X_train, y_train = X_peng[TRAIN_CH1], island[TRAIN_CH1]` ; `folds = mylearn.model_selection.stratified_kfold_indices(y_train, 5)` ; la moyenne de `cross_val_score(IslandPipeline(name), X_train, y_train, cv=folds)` pour chaque nom ; `max(means, key=means.get)` garde le premier nom en cas d'égalité, si `"1-NN"` est le premier.

</details>
<details><summary>Indice 3</summary>

Quatre étapes fuient : seules A (lire les données) et E (découper) sont saines en elles-mêmes, mais E arrive trop tard.

```python
leaky_steps_24 = ...   # your letters


def honest_24():
    X_train, y_train = X_peng[TRAIN_CH1], island[TRAIN_CH1]
    folds = mylearn.model_selection.stratified_kfold_indices(y_train, n_splits=5)
    means = {}
    for name in ("1-NN", "centroid"):
        scores = mylearn.model_selection.cross_val_score(IslandPipeline(name), X_train, y_train, cv=folds)
        means[name] = float(np.mean(scores))
    best = max(means, key=means.get)
    final = IslandPipeline(best)
    final.fit(X_train, y_train)
    return best, final.score(X_peng[TEST_CH1], island[TEST_CH1])
```

</details>

### Ex 8.25 — Sélectionner des features avant la validation croisée : 90 % sur du bruit 🔬

<details><summary>Indice 1</summary>

La corrélation de Pearson du ch. 2 se calcule pour toutes les colonnes à la fois, avec des données centrées. Ensuite, les deux fonctions ne diffèrent que par l'endroit où `top_k_25` est appelée : avant la boucle, sur toutes les lignes, ou dans la boucle, sur les lignes d'entraînement.

</details>
<details><summary>Indice 2</summary>

Centre `X` colonne par colonne, et `y` ; sur ces données centrées, $r_j = \frac{\sum_i x_{ij} y_i}{\sqrt{\sum_i x_{ij}^2 \, \sum_i y_i^2}}$, puis `np.argsort(-np.abs(r))[:k]`. `selected_before_25` : `top_k_25` sur tout, puis `cross_val_score(NearestCentroid(), X[:, columns], y, cv=5)`. `selected_inside_25` : dans la boucle sur `kfold_indices(len(X), 5)`, `top_k_25(X[train_idx], y[train_idx], k)`.

</details>
<details><summary>Indice 3</summary>

```python
def top_k_25(X, y, k):
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    r = Xc.T @ yc / np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    return np.argsort(-np.abs(r))[:k]


def selected_before_25(X, y, k):
    columns = top_k_25(X, y, k)
    return float(np.mean(mylearn.model_selection.cross_val_score(NearestCentroid(), X[:, columns], y, cv=5)))


def selected_inside_25(X, y, k):
    scores = []
    for train_idx, val_idx in mylearn.model_selection.kfold_indices(len(X), 5):
        columns = top_k_25(X[train_idx], y[train_idx], k)
        model = NearestCentroid().fit(X[train_idx][:, columns], y[train_idx])
        scores.append(model.score(X[val_idx][:, columns], y[val_idx]))
    return float(np.mean(scores))
```

</details>

### Ex 8.26 — Comparer deux modèles honnêtement : test par permutation, p-valeur et bootstrap apparié 🔬

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §8.6 de la fiche (test par permutation) et le bootstrap du ch. 2 : l'énoncé donne la recette exacte, à suivre pas à pas, avec **un seul** tirage aléatoire par fonction.

</details>
<details><summary>Indice 2</summary>

a) Compte les manchots où `correct_A_26` vaut 1 et `correct_B_26` vaut 0, puis l'inverse. Ensuite, `d = np.asarray(correct_b, dtype=int) - np.asarray(correct_a, dtype=int)` ; `flips = rng.random((n_perm, len(d))) < 0.5` ; `np.where(flips, -d, d).sum(axis=1)` donne les `n_perm` sommes ; compte celles dont la valeur absolue atteint `abs(d.sum())`. Bootstrap : `rows = rng.integers(0, n, size=(n_boot, n))`, puis `d[rows].mean(axis=1)` (la moyenne de $d$ est l'écart d'accuracy B − A) et `np.percentile(..., [2.5, 97.5])`.

</details>
<details><summary>Indice 3</summary>

```python
discordant_26 = [int(np.sum((correct_A_26 == 1) & (correct_B_26 == 0))),
                 int(np.sum((correct_A_26 == 0) & (correct_B_26 == 1)))]


def permutation_pvalue_26(correct_a, correct_b, n_perm=10_000, seed=826):
    d = np.asarray(correct_b, dtype=int) - np.asarray(correct_a, dtype=int)
    rng = np.random.default_rng(seed)
    flips = rng.random((n_perm, len(d))) < 0.5
    sums = np.where(flips, -d, d).sum(axis=1)
    count = int(np.sum(np.abs(sums) >= abs(int(d.sum()))))
    return (count + 1) / (n_perm + 1)


def paired_bootstrap_26(correct_a, correct_b, n_boot=10_000, seed=8260):
    d = np.asarray(correct_b, dtype=float) - np.asarray(correct_a, dtype=float)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(d), size=(n_boot, len(d)))
    return np.percentile(d[rows].mean(axis=1), [2.5, 97.5])
```

Pour e), compare la p-valeur à 0,05 et regarde si l'intervalle contient 0.

</details>

### Ex 8.27 — Défi : la meilleure paire de features, choisie sans toucher au test 🏆

<details><summary>Indice 1</summary>

Le point de départ choisit sur l'accuracy **d'entraînement** : relis 8.12. Par quoi la remplacer pour noter chaque candidat sur des manchots qu'il n'a pas vus, sans toucher au test ?

</details>
<details><summary>Indice 2</summary>

Une validation croisée sur les seuls manchots reçus par `choose_27` : des folds stratifiés sur le sexe (mélangés, avec une graine fixe), puis, pour chaque candidat, la moyenne de `mylearn.model_selection.cross_val_score(PairModel(*candidate), X_train, y_train, cv=folds)`. Garde le meilleur. Mets `READY_27 = True` seulement quand ton choix est arrêté.

</details>
<details><summary>Indice 3</summary>

```python
def choose_27(X_train, y_train):
    candidates = [(columns, standardize, model) for columns in itertools.combinations(range(4), 2)
                  for standardize in (False, True) for model in ("centroid", "1-NN")]
    folds = mylearn.model_selection.stratified_kfold_indices(y_train, n_splits=5, shuffle=True,
                                                             rng=np.random.default_rng(27))
    scores = [np.mean(mylearn.model_selection.cross_val_score(PairModel(*candidate), X_train, y_train, cv=folds))
              for candidate in candidates]
    return candidates[int(np.argmax(scores))]
```

</details>
