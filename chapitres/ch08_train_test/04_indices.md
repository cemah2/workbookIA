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

a) Dans la boucle simplifiée du livre, seule une erreur déclenche une correction : quand la prédiction est juste, on passe à l'e-mail suivant sans rien changer. C'est B. b) Relis la phrase de la fiche (§8.2) sur ce dont l'optimiseur a besoin : combien d'ingrédients cite-t-elle ? Garde la proposition qui les contient tous. c) Suis le flux de test sur la figure du §8.3 de la fiche : l'erreur remonte-t-elle vers l'optimiseur, ou sert-elle seulement à compter ? d) Calcule la log loss $-\ln 0{,}6$ d'un e-mail bien classé avec une confiance de 0,6 : est-elle nulle ? Tant qu'une loss n'a pas atteint son minimum, que fait la descente de gradient (ch. 5) ?

</details>

### 8.Q2 — Epoch, ordre des exemples et fréquence des mises à jour

<details><summary>Indice 1</summary>

Relis la définition de l'epoch et le mini-exemple sur les mini-batches (§8.2 de la fiche).

</details>
<details><summary>Indice 2</summary>

a) Une mise à jour par exemple, et chaque epoch présente tous les exemples. b) Combien de mini-batches pleins de 64 tiennent dans 1 200 ? Que devient le reste ? d) Que pourrait apprendre un modèle qui reçoit toujours les exemples dans le même ordre, par exemple triés par classe ?

</details>
<details><summary>Indice 3</summary>

a) Une mise à jour par exemple, 1 200 exemples par epoch, 5 epochs : $1\,200 \times 5 = 6\,000$ mises à jour. b) Fais la division euclidienne de 1 200 par 64 : combien de mini-batches pleins, et combien d'exemples restent ? Ce reste forme un dernier mini-batch, plus petit : déclenche-t-il lui aussi une mise à jour ? c) Compare l'affirmation, mot à mot, à la définition de l'epoch dans la fiche (§8.2). d) Pour chaque proposition, demande-toi si le mélange y change vraiment quelque chose : le nombre d'exemples, le jeu de test, le fonctionnement de l'algorithme de mise à jour, ou ce que le modèle peut apprendre de l'ordre des exemples.

</details>

### 8.Q3 — 99 % sur l'entraînement : que peut-on vraiment conclure ?

<details><summary>Indice 1</summary>

Relis le §8.2.1 de la fiche : les deux raisons pour lesquelles un score d'entraînement très haut ne garantit rien, et sa dernière phrase.

</details>
<details><summary>Indice 2</summary>

Sur quelles photos le score de 99 % a-t-il été mesuré : des photos que le modèle connaît, ou des photos nouvelles ? Un raccourci est-il certain, ou seulement possible ?

</details>
<details><summary>Indice 3</summary>

a) Le 99 % est mesuré sur les 1 000 photos que le modèle a vues : il dit très peu de chose des photos nouvelles. A est le piège du chapitre, B est trop fort (« forcément » : un raccourci est possible, pas certain) et D ne mesure rien de nouveau : c'est C. b) Relis la dernière phrase du §8.2.1 de la fiche : dit-elle qu'on peut **calculer** la performance future à partir du modèle, ou qu'il faut la **mesurer** ? c) Pour chaque proposition, demande-toi si elle observe le modèle au travail sur des photos qu'il n'a jamais vues, ou seulement le modèle lui-même.

</details>

### 8.Q4 — Le renard et la neige : repérer un raccourci appris

<details><summary>Indice 1</summary>

Relis le §8.2.1 de la fiche, en particulier le mini-exemple des vélos et des trottinettes.

</details>
<details><summary>Indice 2</summary>

Quel détail accompagne **toujours** une classe dans ces photos, sans rien dire de l'animal ? Un test tiré du même lot contient-il une seule photo où ce détail et la classe ne vont pas ensemble ?

</details>
<details><summary>Indice 3</summary>

a) Le décor (neige ou intérieur) accompagne toujours la même classe et ne dit rien de l'animal : c'est le raccourci, B. Le pelage et les oreilles sont de vraies différences entre renards et chats. b) Dans un test tiré du même lot, le décor et la classe vont-ils encore toujours ensemble ? Si oui, que donne le raccourci sur ces photos ? c) Pour chaque photo, demande-toi ce que répondrait un modèle qui ne regarde que le décor, et si cette réponse serait juste : seule une photo où le décor et la classe se contredisent peut le démasquer. d) Pense à la variété des décors pour **chaque** classe : quels détails sans rapport avec la tâche faudrait-il répartir sur les deux classes ?

</details>

### 8.Q5 — La règle d'or du jeu de test

<details><summary>Indice 1</summary>

Relis le §8.3 de la fiche : les deux qualités du jeu de test, puis la liste des formes de fuites.

</details>
<details><summary>Indice 2</summary>

a) Choisir un learning rate, est-ce une décision qui construit le modèle ? c) Une moyenne et un écart-type calculés sur les données servent-ils à transformer les données d'entraînement ? d) Qu'estime le score de test ?

</details>
<details><summary>Indice 3</summary>

a) Choisir le learning rate en regardant le test, c'est déjà se servir du test pour construire le modèle, même sans entraîner dessus : a) est fausse (§8.4). b) Relis la qualité « intouchable » du jeu de test (§8.3 de la fiche) : combien de fois le consulte-t-on, et à quel moment ? c) La moyenne et l'écart-type d'une standardisation servent-ils à transformer les données d'entraînement ? Si on les calcule avec les lignes de test, que leur apportent ces lignes ? Confronte ta réponse à la liste des formes de fuites du §8.3 de la fiche. d) Relis la qualité « représentatif » : que doit estimer le score de test, et sur quelles données ?

</details>

### 8.Q6 — Fuite de données : ses formes courantes

<details><summary>Indice 1</summary>

Relis la liste des formes de fuites (§8.3 de la fiche).

</details>
<details><summary>Indice 2</summary>

Pour chaque pratique, demande-toi si une information venue du test (ou de l'avenir) influence le modèle. Deux des cinq pratiques sont au contraire de bonnes pratiques.

</details>
<details><summary>Indice 3</summary>

a) A : retirer les doublons **avant** de découper empêche qu'un même exemple se retrouve des deux côtés du découpage. C'est une bonne pratique : A n'est pas dans ta réponse. Passe B, C, D et E au même crible, avec deux questions : un calcul ou un choix a-t-il utilisé des lignes qui serviront à évaluer ? Le modèle profite-t-il d'une information qu'il n'aurait pas en usage réel ? Relis la liste du §8.3 de la fiche jusqu'à ses deux dernières formes : une fuite ne vient pas toujours d'un calcul. b) Pour chaque pratique que tu as retenue, nomme l'information qui passe des lignes d'évaluation, ou de l'avenir, vers le modèle.

</details>

### 8.Q7 — Pourquoi un jeu de validation en plus du test ?

<details><summary>Indice 1</summary>

Relis le §8.4 de la fiche et son mini-exemple sur 1 000 exemples.

</details>
<details><summary>Indice 2</summary>

a) Que règle-t-on avec la validation, et pourquoi pas avec le test ? b) et c) Dans la boucle de recherche, sur quel jeu chaque réglage est-il entraîné, et sur quel jeu est-il noté ?

</details>
<details><summary>Indice 3</summary>

a) La validation sert à comparer des réglages (les hyperparamètres) sur des données qu'aucun n'a vues, sans toucher au test, gardé pour l'évaluation finale : c'est D. b) Pendant la recherche, sur quel jeu chaque réglage apprend-il ses paramètres (étape 1 de la boucle du §8.4 de la fiche) ? Prends la part de ce jeu dans les 500 exemples. Attention au piège : le réentraînement sur entraînement + validation n'a lieu qu'**après** la recherche. c) Dans la boucle de recherche (§8.4 de la fiche), que fait-on des exemples de validation : apprend-on dessus, ou ne s'en sert-on que pour noter chaque réglage ?

</details>

### 8.Q8 — Le score de validation du modèle retenu est-il honnête ?

<details><summary>Indice 1</summary>

Relis, au §8.4 de la fiche, « Pourquoi ne pas annoncer le score de validation du gagnant ? », puis l'encadré 🧮 sur la loi du maximum.

</details>
<details><summary>Indice 2</summary>

Le réglage a été retenu **parce que** son score de validation était le plus haut des 40. Si les 40 réglages se valaient, que vaudrait le plus haut de leurs 40 scores ?

</details>
<details><summary>Indice 3</summary>

Une phrase par question. 1. Pourquoi ce réglage a-t-il été retenu, et que contient le plus haut de 40 scores de validation bruités, si les réglages se valent à peu près (loi du maximum) ? 2. La validation a servi à **choisir** : peut-elle encore mesurer honnêtement le modèle qu'elle a désigné ? 3. Quel jeu, jamais consulté pendant le choix, peut fournir le chiffre à annoncer ? Combien de fois s'en sert-on, et que faut-il annoncer avec ce chiffre ?

</details>

### 8.Q9 — Validation croisée : ce qu'on moyenne, et pourquoi

<details><summary>Indice 1</summary>

Relis le §8.5 de la fiche : le pseudo-code de la validation croisée et le paragraphe qui le suit.

</details>
<details><summary>Indice 2</summary>

a) Qu'obtient-on à la fin de chaque tour : un modèle, des prédictions, un score ? b) Le modèle d'un tour a-t-il le droit d'avoir vu le fold de validation de ce tour ? c) Qu'a-t-on mis de côté **avant** de commencer ?

</details>
<details><summary>Indice 3</summary>

a) Chaque tour produit **un score**, mesuré sur son fold de validation, et l'on moyenne ces $k$ scores : c'est A. Moyenner des prédictions ou des paramètres construirait un autre modèle (un ensemble), et un score d'entraînement ne dit rien de la généralisation. b) Si le modèle d'un tour gardait ce qu'il a appris aux tours précédents, aurait-il déjà vu le fold qui le note ? Relis à quoi sert `clone` (fiche §8.5.1). c) Relis la première ligne du pseudo-code du §8.5 de la fiche : qu'a-t-on mis de côté **avant** la boucle ? Et la boucle fait le travail de quel jeu de la §8.4 ?

</details>

### 8.Q10 — k-fold : combien d'entraînements, quelle taille de fold ?

<details><summary>Indice 1</summary>

Relis le §8.5.1 de la fiche, et le paragraphe « Quand $n$ n'est pas un multiple de $k$ ».

</details>
<details><summary>Indice 2</summary>

Écris $1\,003 = 5 \times q + r$. Les $r$ premiers folds ont $q + 1$ exemples, les autres $q$. Le fold 5 est-il parmi les premiers ?

</details>
<details><summary>Indice 3</summary>

a) Chaque fold sert une fois de validation, et chaque tour demande un entraînement : 5 folds, donc 5 entraînements pour un réglage. b) et c) Fais la division euclidienne de 1 003 par 5 (quotient $q$, reste $r$) : la règle donne $q + 1$ exemples aux $r$ premiers folds, et $q$ aux autres. Le plus grand fold compte donc $q + 1$ exemples, et c) demande combien de folds ont cette taille-là, pas la petite. d) Le fold 5 fait-il partie des $r$ premiers ? L'entraînement du dernier tour compte 1 003 exemples, moins la taille du fold 5. e) Le *leave-one-out* met un seul exemple par fold : combien de folds pour 1 003 exemples, donc combien de tours ?

</details>

### 8.Q11 — Les deux usages des résultats de test

<details><summary>Indice 1</summary>

Relis le §8.6 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un usage a lieu **avant le déploiement**, l'autre **pendant l'entraînement**. Pour b), relis l'encadré ⚠️ « Si le test déçoit, on réentraîne et on reteste » (§8.3).

</details>
<details><summary>Indice 3</summary>

a) Le premier usage a lieu **avant le déploiement** : estimer la performance sur des données nouvelles, et la chiffrer pour la communiquer. C'est A. Cherche le second **pendant** l'entraînement, parmi B, C, D et E : relis le premier paragraphe du §8.6 de la fiche, et écarte ce qui ferait apprendre le modèle sur le test. b) Le nouveau réglage a été choisi **après** avoir vu le score de test : à quoi le test a-t-il alors servi ? Relis ce que la §8.4 de la fiche dit du score d'un jeu qui a servi à choisir.

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

a) Avec un cluster par point, chaque point est son propre centre : sa distance à son centre est nulle, et l'inertie, la somme de ces distances au carré, vaut 0. b) Pars de la meilleure solution à $k$ centres et ajoute un centre posé sur un point : cette nouvelle solution peut-elle avoir une inertie plus grande ? Et la **meilleure** solution à $k + 1$ centres, comparée à celle-ci ? c) Pour chaque critère, demande-toi s'il mesure seulement la distance des points à leur centre (et baisse alors avec $k$), ou s'il compare aussi chaque cluster à ses voisins ; méfie-toi d'un autre nom de l'inertie parmi les propositions. d) Sur quels points l'inertie est-elle mesurée : ceux qui ont placé les centres, ou d'autres ? Compare avec le score d'entraînement d'un modèle de plus en plus flexible, et pense au mémoriseur du ch. 1.

</details>

### 8.R2 — Ch. 5 : un pas de descente de gradient à la main

<details><summary>Indice 1</summary>

Relis, au ch. 5, la règle de mise à jour de la descente de gradient : on avance à contre-pente.

</details>
<details><summary>Indice 2</summary>

Calcule d'abord l'erreur $w_0 x - y$, puis la loss (son carré), puis la dérivée $2x(w_0 x - y)$. Garde les signes.

</details>
<details><summary>Indice 3</summary>

a) $w_0 x - y = 0{,}5 \times 2 - 3 = -2$, donc $L(w_0) = (-2)^2 = 4$. b) Reporte cette erreur dans la dérivée, $L'(w_0) = 2 \times 2 \times (w_0 x - y)$, en gardant son signe. c) $w_1 = 0{,}5 - 0{,}05 \times L'(w_0)$ : retrancher un nombre négatif fait **monter** $w$. d) Calcule la nouvelle erreur $w_1 x - y$ avec ton $w_1$, puis son carré. e) Parmi les trois étapes de la boucle du §8.2 (prédire, comparer, corriger), laquelle ce calcul réalise-t-il, et avec quels ingrédients ?

</details>

### 8.R3 — Ch. 1 : généralisation, définition et exemple

<details><summary>Indice 1</summary>

Relis la définition de la généralisation au ch. 1 (§1.2 de la fiche) et l'exercice 1.14 (le mémoriseur).

</details>
<details><summary>Indice 2</summary>

Pense à deux modèles des ch. 1 et 7 sur les manchots : l'un apprend une règle (des centroïdes), l'autre retient les manchots qu'il a vus.

</details>
<details><summary>Indice 3</summary>

Ta définition doit dire sur quels exemples on juge le modèle (ceux qu'il a vus, ou d'autres ?) et d'où viennent ces exemples. Pour l'exemple, compare le mémoriseur du ch. 1 (1.14) à un modèle qui résume les manchots en quelques nombres ou en quelques règles (le centroïde le plus proche du ch. 7, l'arbre de décision du ch. 1) : que fait chacun sur ses manchots d'entraînement, puis sur un manchot qu'il n'a jamais vu ? Pour la mesure, il te faut deux accuracies, calculées sur deux jeux différents : lesquels, et que regardes-tu entre elles ?

</details>

<a id="papier"></a>

## ✏️ ∂ 📈 Papier-crayon

### Ex 8.1 — Découper 344 manchots : hold-out, validation et folds ✏️

<details><summary>Indice 1</summary>

Relis les deux règles d'arrondi de la fiche : $\lceil t \cdot n \rceil$ pour le test (§8.3), et les tailles des folds (§8.5.1). Pour i), le mini-exemple du plus fort reste (§8.3).

</details>
<details><summary>Indice 2</summary>

c) puis d) : le test d'abord, sur les 344 manchots ; la validation ensuite, sur ceux qui restent. f) Combien de manchots restent une fois le test de c) retiré ? g) Écris $344 = 10 \times q + r$. i) Les parts exactes valent $n_c \times n_{\text{test}} / 344$, avec le $n_{\text{test}}$ de c) ; prends leurs parties entières, puis compte les manchots qui manquent.

</details>
<details><summary>Indice 3</summary>

a) $n_{\text{test}} = \lceil 0{,}25 \times 344 \rceil = \lceil 86 \rceil = 86$ : le produit tombe juste, l'arrondi ne change rien. b) L'entraînement reçoit tout le reste : $n - n_{\text{test}}$. c) Même règle avec $t = 0{,}2$ : calcule $0{,}2 \times 344$, puis arrondis **vers le haut**. d) Retire d'abord des 344 manchots le test de c), puis calcule $\lceil 0{,}25 \times \text{reste} \rceil$. e) $344 - n_{\text{test}} - n_{\text{val}}$, avec tes réponses c) et d). f) Le même reste qu'en d), coupé en 5 folds : fais la division euclidienne par 5. g) Applique la règle des folds à la division de l'indice 2 : combien de folds reçoivent un manchot de plus ? h) Au premier tour, la validation est le fold 1 : quelle est sa taille, d'après g) ? Retire-la des 344. i) Pour chaque espèce, calcule la part exacte $n_c \times n_{\text{test}} / 344$ et sa partie entière ; compte les manchots qui manquent pour atteindre $n_{\text{test}}$, puis donne-les un par un aux plus grandes **parties décimales** (pas aux plus grandes espèces).

</details>

### Ex 8.2 — Compter les entraînements d'une recherche d'hyperparamètres ✏️

<details><summary>Indice 1</summary>

Relis la boucle de recherche d'hyperparamètres (§8.4 de la fiche) et l'encadré 🧮 sur la validation croisée imbriquée (§8.5).

</details>
<details><summary>Indice 2</summary>

a) Chaque valeur d'un hyperparamètre se combine avec chaque valeur des autres. c) Un entraînement par réglage et par fold. f) Que fait-on dans **un** tour extérieur ? Compte-le, puis multiplie par le nombre de tours. g) Retire d'abord du budget le réentraînement final.

</details>
<details><summary>Indice 3</summary>

a) Chaque valeur d'un hyperparamètre se combine avec chaque valeur des deux autres : $3 \times 4 \times 2 = 24$ réglages. b) Avec un jeu de validation fixe, combien d'entraînements faut-il par réglage ? c) Avec 5 folds : un entraînement par réglage **et** par fold. d) Ajoute à c) le réentraînement final, celui du seul réglage retenu. e) Le nombre de d), multiplié par 4 minutes, puis divisé par 60. f) Chaque tour extérieur refait tout le protocole de d) sur sa partie d'entraînement (sa note sur le fold extérieur ne coûte aucun entraînement) : multiplie par le nombre de tours extérieurs. g) Avec $r$ réglages, le protocole de d) coûte $5r + 1$ entraînements : résous $5r + 1 \le 200$, et garde le plus grand entier $r$.

</details>

### Ex 8.3 — Fuite ou pas ? Six protocoles à auditer ✏️

<details><summary>Indice 1</summary>

Relis la liste des formes de fuites (§8.3 de la fiche) et le paragraphe « Données dépendantes » du §8.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pour chaque protocole, pose quatre questions : un calcul a-t-il utilisé des lignes de test ? Un choix a-t-il été fait en regardant le test ? Une feature sera-t-elle connue au moment de prédire ? Des exemples dépendants ont-ils été séparés au hasard ?

</details>
<details><summary>Indice 3</summary>

a) La médiane a été calculée sur toutes les annonces, test compris : une information des annonces de test est entrée dans les données d'entraînement. C'est une fuite par prétraitement : `True`. Pose une question aussi précise aux cinq autres protocoles. b) Sur quelles images la standardisation est-elle ajustée, et d'où vient le test ? c) Sur quel jeu le réglage a-t-il été choisi, et quel score est publié ? d) En production, connaîtra-t-on les mois qui suivent celui qu'on prédit ? e) Avec `GroupKFold`, un même locuteur peut-il se retrouver des deux côtés ? f) Le nombre d'appels du mois prochain existe-t-il au moment de prédire ? Il y a quatre fuites en tout.

</details>

### Ex 8.4 — Quelle confiance accorder à une accuracy de test ? Erreur type et taille du test ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur l'erreur type d'une accuracy (§8.3 de la fiche).

</details>
<details><summary>Indice 2</summary>

$\mathrm{SE} = \sqrt{0{,}92 \times 0{,}08 / 250}$, et la demi-largeur vaut $1{,}96\,\mathrm{SE}$. d) Écris $1{,}96 \sqrt{0{,}92 \times 0{,}08 / n} \le 0{,}01$ et isole $n$. e) $n$ est sous une racine.

</details>
<details><summary>Indice 3</summary>

a) $\mathrm{SE} = \sqrt{0{,}92 \times 0{,}08 / 250} = \sqrt{0{,}0002944} \approx 0{,}0172$. b) Multiplie cette erreur type par 1,96, sans l'arrondir d'abord. c) $0{,}92$ moins la demi-largeur de b). d) Isole $n$ : $n \ge (1{,}96 / 0{,}01)^2 \times 0{,}92 \times 0{,}08$. Le résultat n'est pas entier : dans quel sens arrondir pour que la demi-largeur ne dépasse pas 0,01 ? e) La demi-largeur est proportionnelle à $1 / \sqrt{n}$ : pour la diviser par 3, par combien faut-il multiplier $\sqrt{n}$, donc $n$ ? f) Compare l'écart $0{,}92 - 0{,}90$ à ta demi-largeur de b). g) A et B sont-ils notés sur les mêmes exemples ? Lesquels l'un réussit-il, quand l'autre les rate ?

</details>

### Ex 8.5 — Moyenne et écart-type de scores de validation croisée ✏️

<details><summary>Indice 1</summary>

Relis les formules de la moyenne et de l'écart-type des scores (§8.5.1 de la fiche) : on divise par $k$.

</details>
<details><summary>Indice 2</summary>

Calcule les écarts à la moyenne, leurs carrés, la moyenne de ces carrés, puis sa racine. Pour f) et g), écris d'abord la liste des différences B − A, fold par fold.

</details>
<details><summary>Indice 3</summary>

a) $(0{,}82 + 0{,}88 + 0{,}79 + 0{,}85 + 0{,}86) / 5 = 4{,}20 / 5 = 0{,}840$. b) Écarts de A à 0,84 : −0,02 ; 0,04 ; −0,05 ; 0,01 ; 0,02 ; élève-les au carré, fais leur moyenne (division par 5), puis prends la racine. c) et d) Même méthode pour B. e) Compare A et B fold par fold ; une égalité ne compte pas. f) et g) Différences B − A : 0,03 ; −0,04 ; 0,07 ; 0 ; −0,03 ; calcule leur moyenne, puis leur écart-type (division par 5). h) Compare la moyenne des différences à leur écart-type, et compte les folds gagnés par chacun.

</details>

### Ex 8.6 — Le biais d'optimisme du meilleur de K modèles ∂

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur la loi du maximum (§8.4 de la fiche).

</details>
<details><summary>Indice 2</summary>

1. Le maximum est inférieur à $s$ si et seulement si **chaque** $S_j$ l'est. 2. Compare, tirage par tirage, le maximum de $K + 1$ nombres et celui des $K$ premiers. b) et c) $1 - (1 - 0{,}0668)^K$. d) Résous $1 - 0{,}9332^K \ge 0{,}9$ avec des logarithmes.

</details>
<details><summary>Indice 3</summary>

1. $\max_j S_j < s$ équivaut à « $S_1 < s$ et $S_2 < s$ et … et $S_K < s$ ». Les $S_j$ étant indépendants, la probabilité de cette conjonction est le produit des $K$ probabilités, toutes égales à $P(S_1 < s)$ puisque les $S_j$ ont la même loi : d'où $P(S_1 < s)^K$. La seconde formule vient du complément, appliqué deux fois. 2. Compare, tirage par tirage, $\max(S_1, \dots, S_{K+1})$ et $\max(S_1, \dots, S_K)$ : l'un est-il toujours au moins égal à l'autre ? Que devient une inégalité toujours vraie quand on prend l'espérance des deux côtés ? a) $\sqrt{p(1-p)/n}$ avec $p = 0{,}8$ et $n = 100$. b) et c) $1 - (1 - 0{,}0668)^K$, avec $K = 10$, puis $K = 50$. d) Prends le logarithme des deux côtés de $0{,}9332^K \le 0{,}1$ : $\ln 0{,}9332$ est négatif, et diviser par un nombre négatif renverse l'inégalité. $K$ est entier : dans quel sens arrondir pour atteindre **au moins** 0,9 ? e) Sur un test neuf, le score du réglage retenu profite-t-il encore de la chance qu'il a eue en validation ? Quelle est son accuracy réelle ? 3. Pense aux exemples communs du jeu de validation, et à des réglages voisins.

</details>

### Ex 8.8 — Comparer des modèles à partir de boîtes à moustaches de scores 📈

<details><summary>Indice 1</summary>

Relis, au ch. 2, les quartiles (🧮 « percentiles et quantiles »). Dans une boîte à moustaches, le trait de la boîte est la médiane, la boîte va du 1ᵉʳ au 3ᵉ quartile, la moustache s'arrête au dernier score situé à moins de 1,5 écart interquartile de la boîte, et chaque score au-delà est un point isolé. Puis relis le §8.6 de la fiche sur la comparaison exemple par exemple.

</details>
<details><summary>Indice 2</summary>

a) et b) Compare les traits au milieu des boîtes, puis la hauteur des boîtes et l'écart entre les moustaches. d) Dans le panneau (b), compte les points sous la ligne zéro. f) Que se passe-t-il pour C dans ses plus mauvais folds ?

</details>
<details><summary>Indice 3</summary>

a) Le trait de la boîte de C, vers 0,89, est au-dessus de ceux de B et de A : c'est C. b) Compare la hauteur des trois boîtes (l'écart interquartile), puis l'écart entre le plus haut et le plus bas score de chaque série, moustaches et points isolés compris. c) Ne compte que les points dessinés à part, au-delà de la moustache de A, sans ceux des autres boîtes. d) Dans le panneau (b), un point sous la ligne zéro est un fold où B − A < 0 : compte-les. e) Vérifie séparément les deux constats : les boîtes de A et de B se recouvrent-elles largement (panneau a) ? Les points du panneau (b) sont-ils surtout au-dessus de zéro ? f) Regarde le bas de C (sa moustache et ses points isolés) et compare-le au pire score de A. g) Le panneau (b) compare A et B **sur le même fold** : que retire-t-il, que le panneau (a) mélange ? Pour le point isolé de A (environ 0,77), lis dans le panneau (b) de combien B dépasse A au plus : déduis-en le score maximal de B sur ce fold-là, puis cherche lequel des scores de B est aussi bas.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 8.7 — Pourquoi le jeu de test reste sous clé : l'analogie de l'examen 🗣️

<details><summary>Indice 1</summary>

Relis l'analogie de l'examen dans le livre (§8.3), puis le §8.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Plan en cinq lignes : à quoi sert un examen ; ce qui arrive si l'élève a vu le sujet ; pourquoi on ne regarde le test qu'une fois ; un exemple de la vie courante où « choisir sur ce qui sert à noter » flatte le résultat ; la solution (un autre jeu pour choisir).

</details>
<details><summary>Indice 3</summary>

Un examen mesure ce que l'élève sait faire sur des questions **nouvelles** ; s'il a vu le sujet, il peut **mémoriser** les réponses sans rien comprendre. Pour l'exemple courant, pense à une recette choisie parmi dix en les faisant goûter aux mêmes amis, puis à la note de ces amis annoncée comme celle que donneront des inconnus.

</details>

### Ex 8.9 — Raccourcis appris : radiographies, chars d'assaut et responsabilité ⚖️

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §8.2.1 et la liste des formes de fuites (§8.3). Dans l'article de Zech et coll., regarde la proportion de pneumonies dans chaque système hospitalier.

</details>
<details><summary>Indice 2</summary>

1. Si un hôpital a beaucoup plus de pneumonies qu'un autre, que gagne un modèle qui reconnaît l'hôpital ? 2. D'où venaient les radiographies de test ? 3. Pense à `GroupKFold`, avec un groupe par hôpital. 5. Liste les personnes qui subiront une erreur, puis celles qui peuvent la prévenir.

</details>
<details><summary>Indice 3</summary>

1. Dans un des systèmes, environ un tiers des radiographies montrent une pneumonie, contre environ 1 % dans un autre : reconnaître l'hôpital (marqueurs, type d'appareil, inscriptions) donne déjà une bonne partie de la réponse. 2. Sur un test tiré des mêmes hôpitaux, le lien entre l'hôpital et la maladie est-il le même qu'à l'entraînement ? Que fait alors le raccourci ? Pour le type de fuite, relis les qualités du jeu de test (§8.3 de la fiche) et les trois familles de l'encadré 🕰️ sur la taxonomie : quelle condition ce test ne remplit-il pas ? 3. Le test doit contenir des radiographies où le raccourci ne marche plus : de quels hôpitaux doivent-elles venir ? Écris le protocole en une phrase (sur quels hôpitaux entraîner, sur lequel tester), puis ce qu'il faudrait encore faire avant un usage clinique. 4. Pèse ce qu'apporte une histoire frappante, et ce que coûte une anecdote invérifiable dans un rapport. 5. Cherche « fiche modèle » (*model card*, Mitchell et coll., 2019).

</details>

### Ex 8.10 — Kapoor & Narayanan (2023) : une taxonomie des fuites 📄

<details><summary>Indice 1</summary>

Prends la version publiée dans *Patterns* (en accès libre, par le lien DOI de l'énoncé) : ses chiffres diffèrent de ceux de la première version d'arXiv. La taxonomie y est résumée dans un tableau, puis détaillée type par type ; les *model info sheets* ont leur propre section, et leur modèle complet est fourni avec l'article.

</details>
<details><summary>Indice 2</summary>

1. Les trois familles portent sur la séparation entre entraînement et test, sur la légitimité des features, et sur la distribution du jeu de test. 2. Pour chaque protocole fautif de 8.3, demande-toi lequel de ces trois points est violé. 4. Cherche la section sur la prédiction des guerres civiles : combien d'articles ont été examinés, lesquels contenaient une erreur, et de quels types.

</details>
<details><summary>Indice 3</summary>

1. La première famille, une séparation imparfaite entre entraînement et test, regroupe quatre types : pas de jeu de test, prétraitement ou sélection de features sur l'entraînement et le test, doublons. La sélection des 20 features de Q6 B, faite sur toutes les lignes, en est un exemple. Cherche de même un exemple du chapitre pour chacune des deux autres familles. 2. Pour a), c), d) et f), demande-toi chaque fois quelle condition est violée : la séparation entre entraînement et test (et alors, lequel des quatre types ?), la légitimité d'une feature, ou la distribution du jeu de test. Pour d), cherche dans le tableau de l'article la famille où sont rangées les fuites liées au temps. Pour c), la taxonomie n'a pas de case dédiée à un choix fait sur le test : cherche la famille la plus proche, en te demandant à quoi le test a servi. 3. Compte les questions du modèle de fiche fourni avec l'article, puis lis les trois arguments autour desquels elles s'organisent. 4. Dans la section sur les guerres civiles, relève trois choses : combien d'articles ont été réexaminés, et combien contenaient une erreur ; ce que ces articles affirmaient sur les modèles complexes ; ce que devient leur avantage sur la régression logistique une fois les fuites corrigées.

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

Les deux premières réponses, en modèle ; les trois autres suivent le même schéma :

```python
X_tr_11, X_te_11, y_tr_11, y_te_11 = sklearn_train_test_split(X_peng, species, test_size=0.2, random_state=42)
n_test_11a = len(X_te_11)
chinstrap_11b = int(np.sum(y_te_11 == "Chinstrap"))      # the species of the test part: the 4th array
# c) the same split with stratify=species; [3] picks the species of its test part; count them as in b)
# d) shuffle=False (and no random_state); len(np.unique(...)) of the species of the test part
# e) a list comprehension over seed in range(100): the count of b) with random_state=seed; then [min, max]
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

a) et c) Pour un manchot d'entraînement, quel manchot d'entraînement est à une distance nulle de lui ? Un autre manchot a-t-il exactement les mêmes quatre mesures ? Quel label le modèle recopie-t-il alors, et ce label dépend-il de la vérité des espèces, ou seulement de ce qui est rangé dans sa mémoire ? b) Souviens-toi des accuracies obtenues aux ch. 1 et 7 avec les quatre mesures, et demande-toi si un manchot nouveau, près de la frontière entre deux espèces, peut recevoir le label d'un voisin d'une autre espèce. d) Une fois les espèces mélangées, les mesures d'un manchot de test ne disent plus rien de son label : copier celui d'un voisin revient à tirer un label au hasard, avec les proportions des espèces. Tomber juste a alors une probabilité $\sum_k p_k^2$, la somme des carrés des parts des trois espèces : calcule-la avec leurs effectifs (146 Adélie, 68 Chinstrap et 119 Gentoo sur 333), puis place ce nombre dans l'un des quatre intervalles.

</details>

### Ex 8.13 — train_test_split from scratch 🔨

<details><summary>Indice 1</summary>

Organise ta fonction en trois temps : les contrôles et le calcul de $n_{\text{test}}$ ; le choix d'**un** tableau d'indices de test et d'**un** tableau d'indices d'entraînement (trois cas : stratifié, mélangé, dernières lignes) ; une boucle qui indexe chaque tableau avec ces deux tableaux d'indices.

</details>
<details><summary>Indice 2</summary>

Un `float` : vérifie $0 < t < 1$, puis `math.ceil(test_size * n)` ; un `int` : `n_test = test_size` ; dans les deux cas, il faut `0 < n_test < n`. Attention, `True` est aussi un `int` en Python. Mélangé : `perm = rng.permutation(n)`, le test est `perm[:n_test]`. Stratifié : `_, codes, counts = np.unique(stratify, return_inverse=True, return_counts=True)` ; les quotas par classe viennent de `np.floor(counts * n_test / n)`, plus un pour les plus grandes parties décimales (`np.argsort(-(exact - quotas), kind="stable")[:missing]`) ; puis, pour chaque classe `c`, `rng.permutation(np.flatnonzero(codes == c))[:quota]`. Tout le reste va à l'entraînement.

</details>
<details><summary>Indice 3</summary>

Le squelette, avec les lignes clés du cas stratifié (les quotas du plus fort reste) :

```python
def train_test_split(*arrays, test_size=0.25, shuffle=True, stratify=None, rng=None):
    # 1. checks: at least one array; np.asarray on each; the same number of rows n (ValueError otherwise)
    # 2. n_test: refuse a bool FIRST (True is an int); an int is n_test itself; a float in (0, 1) gives
    #    math.ceil(test_size * n) (the stub does not import math: add `import math` at the top of the file);
    #    anything else: ValueError; then both parts must be non-empty, 0 < n_test < n
    # 3. rng = np.random.default_rng() if rng is None
    # 4. ONE pair of index arrays (train_idx, test_idx), in one of three ways:
    #    - stratify: ValueError if shuffle is False or if a class has a single member;
    #      _, codes, counts = np.unique(np.asarray(stratify), return_inverse=True, return_counts=True), then:
    exact = counts * n_test / n
    quotas = np.floor(exact).astype(int)
    quotas[np.argsort(-(exact - quotas), kind="stable")[:n_test - quotas.sum()]] += 1   # largest remainders
    #      test_idx: for each class c, the first quotas[c] indices of rng.permutation(np.flatnonzero(codes == c));
    #      train_idx: all the other indices
    #    - shuffle only: perm = rng.permutation(n); the test part is perm[:n_test], the training part the rest
    #    - no shuffle: the LAST n_test rows form the test part, in their order
    # 5. index EVERY array with the same pair: [a1_train, a1_test, a2_train, a2_test, ...]
```
La ligne clé de l'étape 5, dans une boucle sur les tableaux : `result += [a[train_idx], a[test_idx]]`. Ne mélange jamais les tableaux eux-mêmes : seuls les indices le sont.

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
    # 1. ValueError if n_splits < 2 or n_splits > n_samples
    # 2. indices: np.arange(n_samples), or a permutation of it drawn with rng if shuffle (default_rng() if rng is None)
    sizes = np.full(n_splits, n_samples // n_splits)
    sizes[: n_samples % n_splits] += 1                     # the FIRST folds get one more sample
    bounds = np.concatenate([[0], np.cumsum(sizes)])       # fold i: indices[bounds[i]:bounds[i + 1]]
    # 3. for each fold: val_idx = its slice, sorted; train_idx = what comes before it and after it,
    #    concatenated, then sorted
    # 4. return a LIST of the n_splits pairs (train_idx, val_idx), not a generator
```

</details>

### Ex 8.15 — Reproduire la figure 8.13 : la rotation des folds 🎨

<details><summary>Indice 1</summary>

Une ligne par tour, une colonne par échantillon : que doit valoir la case de la ligne `i` et de la colonne `j` ?

</details>
<details><summary>Indice 2</summary>

Pars d'une matrice de zéros de forme `(len(splits), n)` (`np.zeros`, avec `dtype=int`), puis, pour chaque tour `i`, mets des 1 dans la ligne `i`, aux indices de validation de ce tour, en une seule affectation (indexation par un tableau d'indices, 0A) ; `enumerate(splits)` donne à la fois le numéro du tour et sa paire d'indices. Pour le dessin : `ax.imshow` de cette matrice, avec la `ListedColormap` de l'énoncé, `aspect="auto"`, `vmin=0` et `vmax=1`, puis les graduations et le titre.

</details>
<details><summary>Indice 3</summary>

```python
def fold_matrix_15(splits, n):
    matrix = np.zeros((len(splits), n), dtype=int)         # one row per round, one column per sample
    # for each round i: put 1 in row i at the validation indices of that round (indexing with an array,
    # no inner loop over the samples); then return the matrix


def draw_folds_15(ax, splits, n, title):
    # ax.imshow of fold_matrix_15(splits, n): the two-colour ListedColormap, aspect="auto", vmin=0, vmax=1
    ax.set_yticks(range(len(splits)), [f"round {i + 1}" for i in range(len(splits))])
    # x ticks every 5 samples, an x label, and the title
```
Garde `vmin=0, vmax=1` : sans eux, `imshow` cale ses deux couleurs sur le plus petit et le plus grand nombre de la matrice, et une matrice qui ne contiendrait que des 1 serait dessinée en bleu, la couleur de l'entraînement.

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
    def __init__(self, degree=1):
        self.degree = degree               # and nothing else: clone (8.22) rebuilds PolyFit(degree=...)

    def fit(self, X, y):
        # the single feature X[:, 0] as a float vector; self.coef_ from np.polyfit(x, y, self.degree); return self

    def predict(self, X):
        # np.polyval(self.coef_, ...) on the same column X[:, 0]: one prediction per row of X

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - self.predict(X)) ** 2)
        # ss_tot: the same sum around y.mean(), the mean of the SCORED targets; return a Python float
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

a) Le degré retenu est celui dont le $R^2$ de validation est le plus haut des douze : relis la question 2 et la question e) de ∂ 8.6. Ce score contient-il une part de chance propre à ces 20 districts ? Cette chance suit-elle le degré sur les 300 districts de test, qu'il n'a jamais vus ? b) Pour le degré 1, fixé d'avance, la validation et le test sont deux mesures du même modèle, sur des districts différents, sans aucune sélection : l'une a-t-elle une raison de battre l'autre plus souvent ? Et si cette part valait 50 % en moyenne, pourrait-elle, sur 300 répétitions, s'en écarter de plus de 10 points (son erreur type serait $\sqrt{0{,}5 \times 0{,}5 / 300}$) ? c) Sur 20 districts, l'écart de $R^2$ entre deux degrés est-il grand devant le bruit de la mesure ? Quels degrés peuvent alors gagner par chance, et que fait un polynôme de haut degré sur des districts situés hors de la zone de ses 40 districts d'entraînement (souviens-toi du degré 8 en 8.16) ?

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
    X_rest, X_test, y_rest, y_test = split(X, y, test_size=0.2, shuffle=False)    # the test, set aside first
    # the validation part, taken in X_rest, y_rest: test_size=0.25, shuffle=False -> X_train, X_val, y_train, y_val
    # inside `with warnings.catch_warnings():` and warnings.simplefilter("ignore") (high degrees: np.polyfit warns):
    #     for each degree: a NEW PolyFit(degree=degree), fitted on the training part; append its validation R²
    best_degree = degrees[int(np.argmax(val_scores))]                              # the first one on ties
    #     the final PolyFit(best_degree), refitted on X_rest, y_rest (training + validation)
    # return val_scores, best_degree, and the test R² of the final model (the test, once)
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
cv_shuffle_19 = KFold(n_splits=5, shuffle=True, random_state=0)     # a), as a model
# b) groups_19: the decade of every sample (1754 -> 1750), from years_19: an integer division by 10,
#    then a multiplication by 10, on the whole array at once
# c) and d): the same pattern as a), with GroupKFold and TimeSeriesSplit; n_splits=5 is all they need
#    (GroupKFold does not take the groups when it is created: cross_val_score passes them, groups=groups_19)
```
Le piège de b) : des groupes par **année** au lieu de par décennie. Le découpage resterait valide, mais des années voisines, très semblables, se retrouveraient de part et d'autre.

</details>

### Ex 8.20 — Écrire tes propres tests pytest pour train_test_split 🛠️

<details><summary>Indice 1</summary>

Un test est une fonction `test_...()` sans argument, qui appelle `train_test_split(...)` (le nom que la vérification donne à chaque version testée) et vérifie **une** propriété avec `assert`. Écris un test par propriété de l'énoncé.

</details>
<details><summary>Indice 2</summary>

Taille : avec $n = 10$ et $t = 0{,}25$, il faut 3 lignes de test, pas 2. Disjoint et complet : `sorted(np.concatenate([train, test]).tolist())` doit valoir `list(range(n))`. Alignement : construis `y` à partir de `X` (par exemple `y = X[:, 0] * 10`) et vérifie que la relation tient dans chaque partie. Même graine : deux appels avec `rng=np.random.default_rng(3)`. Sans mélange : les dernières lignes en test.

</details>
<details><summary>Indice 3</summary>

Un premier test, en modèle ; les autres suivent le même schéma (un appel, une propriété, un `assert`) :

```python
def test_sizes():
    for n, test_size, n_test in [(10, 0.25, 3), (8, 0.25, 2), (10, 3, 3)]:    # 10 × 0.25 = 2.5: rounded UP
        train, test = train_test_split(np.arange(n), test_size=test_size, rng=np.random.default_rng(0))
        # assert: n_test rows in the test part, and all the others in the training part


# test_disjoint_and_complete: the sorted concatenation of train and test equals list(range(n))
# test_rows_stay_aligned: y built from X (y = X[:, 0] * 10), and that relation checked in BOTH parts
# test_same_seed_same_split: two calls, each with rng=np.random.default_rng(3), give the same test part
# test_last_rows_without_shuffle: np.arange(10), test_size=0.3, shuffle=False -> the test part is [7, 8, 9]

TESTS_20 = [test_sizes]   # then your other tests: the functions themselves, not their names
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
    # 1. y = np.asarray(y), n = len(y); ValueError if n_splits < 2 or n_splits > n
    _, codes, counts = np.unique(y, return_inverse=True, return_counts=True)
    # 2. warnings.warn(..., UserWarning) if counts.min() < n_splits
    # 3. order: np.argsort(codes, kind="stable") without shuffle; with shuffle, the concatenation, class by
    #    class (c = 0, 1, ...), of rng.permutation(np.flatnonzero(codes == c))
    # 4. deal the cards: for each fold i,
    val_idx = np.sort(order[i::n_splits])
    #    and train_idx = all the other indices, np.setdiff1d(np.arange(n), val_idx), already sorted
    # 5. return the LIST of the n_splits pairs (train_idx, val_idx)
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
    # return a new object of the same class, built with these hyperparameters (fiche §8.5.1)


def cross_val_score(estimator, X, y, cv=5, scoring=None):
    # 1. X, y = np.asarray(X), np.asarray(y); the two ValueError, BEFORE the loop
    # 2. splits: an int cv that is not a bool (True is an int) -> kfold_indices(len(X), int(cv)); otherwise list(cv)
    # 3. for each pair (train_idx, val_idx):
    model = clone(estimator)                     # a NEW model every round: `estimator` itself is never fitted
    #        model.fit(...) on the training rows, as its own statement (no chained .fit(...).score(...)),
    #        then scoring(model, X_val, y_val) if scoring is given, else model.score(X_val, y_val)
    # 4. return np.asarray(scores, dtype=float)
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
            X, y, test_size=0.25, rng=np.random.default_rng(seed))      # a NEW generator for each seed
        # append the accuracy of NearestCentroid fitted on the training part and scored on the test part
    # return the n_repeats accuracies as a NumPy array


def cv_means_23(X, y, n_repeats):
    # for each seed: folds = kfold_indices(len(X), 5, shuffle=True, rng=np.random.default_rng(seed)),
    #     then the MEAN of the 5 scores of cross_val_score(NearestCentroid(), X, y, cv=folds)
    # return the n_repeats means as a NumPy array
```
Le piège : la même graine à chaque répétition, ou une k-fold sans `shuffle=True` ; tous les découpages seraient alors identiques, et la dispersion mesurée nulle.

</details>

### Ex 8.24 — Un notebook trop beau pour être vrai : quatre fuites à corriger 🐛

<details><summary>Indice 1</summary>

Pour chaque étape, demande-toi de quelle information elle profite, et si le modèle l'aurait dans l'usage réel. Puis suis, pour `honest_24`, les quatre points du protocole de l'énoncé.

</details>
<details><summary>Indice 2</summary>

Pour chacune des six étapes, pose deux questions : sur quelles lignes travaille-t-elle (toutes, ou l'entraînement seulement) ? Et que se passe-t-il pour un manchot dont des copies se retrouvent des deux côtés du découpage ? Regarde aussi sur quoi le modèle final est choisi. Pour `honest_24` : `X_train, y_train = X_peng[TRAIN_CH1], island[TRAIN_CH1]` ; `folds = mylearn.model_selection.stratified_kfold_indices(y_train, 5)` ; la moyenne de `cross_val_score(IslandPipeline(name), X_train, y_train, cv=folds)` pour chaque nom ; `max(means, key=means.get)` garde le premier nom en cas d'égalité, si `"1-NN"` est le premier.

</details>
<details><summary>Indice 3</summary>

a) L'étape A ne fait que lire les quatre mesures et l'île des 333 manchots : aucune information ne passe des futures lignes de test vers le modèle, elle est saine. Pour B à F, applique la règle d'or de la fiche, « rien de ce qui sert à évaluer ne sert à construire » : au moment où chaque étape s'exécute, où sont les futures lignes de test, et l'étape s'en sert-elle ? Une étape saine en elle-même peut aussi arriver trop tard : juge-la, elle aussi, avec la règle d'or.

Pour `honest_24`, le squelette et les lignes clés :

```python
def honest_24():
    X_train, y_train = X_peng[TRAIN_CH1], island[TRAIN_CH1]          # the split of ch. 1, FIRST
    folds = mylearn.model_selection.stratified_kfold_indices(y_train, n_splits=5)
    # means: for name in ("1-NN", "centroid"), the mean of cross_val_score(IslandPipeline(name), X_train, y_train, cv=folds)
    best = max(means, key=means.get)                                 # "1-NN" first in the dict: kept on ties
    # refit IslandPipeline(best) on the 233 training penguins, then ONE score on X_peng[TEST_CH1], island[TEST_CH1]
    # return best and that test accuracy
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
    Xc, yc = X - X.mean(axis=0), y - y.mean()                              # centred data
    r = Xc.T @ yc / np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())       # Pearson r of every column at once
    # return the indices of the k largest |r|: np.argsort of -np.abs(r), the first k


def selected_before_25(X, y, k):
    # columns = top_k_25 on ALL the rows; then the mean of
    # mylearn.model_selection.cross_val_score(NearestCentroid(), X[:, columns], y, cv=5), as a float


def selected_inside_25(X, y, k):
    # for train_idx, val_idx in mylearn.model_selection.kfold_indices(len(X), 5):
    #     columns = top_k_25(X[train_idx], y[train_idx], k)      <- the selection sees the training part only
    #     NearestCentroid fitted on X[train_idx][:, columns], scored on X[val_idx][:, columns]
    # return the mean of the 5 scores, as a float
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
discordant_26 = [int(np.sum((correct_A_26 == 1) & (correct_B_26 == 0))),   # only A is right
                 ...]                                                         # only B is right: the other way round


def permutation_pvalue_26(correct_a, correct_b, n_perm=10_000, seed=826):
    d = np.asarray(correct_b, dtype=int) - np.asarray(correct_a, dtype=int)
    rng = np.random.default_rng(seed)
    flips = rng.random((n_perm, len(d))) < 0.5                  # ONE draw: True = swap A and B on that penguin
    # the n_perm sums: np.where(flips, -d, d).sum(axis=1), integers; count those with |sum| >= abs(d.sum())
    # return (count + 1) / (n_perm + 1)


def paired_bootstrap_26(correct_a, correct_b, n_boot=10_000, seed=8260):
    # d as above, a generator from seed, then ONE draw: rows = rng.integers(0, len(d), size=(n_boot, len(d)))
    # the n_boot accuracy differences B - A: d[rows].mean(axis=1); return their percentiles 2.5 and 97.5
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
    # candidates: the 24 triplets, built as in the starting point
    folds = mylearn.model_selection.stratified_kfold_indices(y_train, n_splits=5, shuffle=True,
                                                             rng=np.random.default_rng(27))   # a fixed seed
    # for each candidate: the MEAN of cross_val_score(PairModel(*candidate), X_train, y_train, cv=folds)
    #     (the same folds for every candidate: they are compared on the same penguins)
    # return the candidate with the best mean (np.argmax)
```
Attention : ne travaille qu'avec `X_train` et `y_train`, jamais avec `X_peng` ni `TRAIN_CH1` : `grade_27` relance ta fonction sur 20 autres découpages, où les manchots de test ne sont plus les mêmes.

</details>
