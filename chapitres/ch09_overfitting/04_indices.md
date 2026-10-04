# 9 · Overfitting et underfitting — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 📈 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 9.Q1 — Overfitting ou underfitting ? Définitions et symptômes

<details><summary>Indice 1</summary>

Deux questions, dans cet ordre : le modèle réussit-il ses propres exemples ? Puis : fait-il presque aussi bien sur des exemples nouveaux ?

</details>
<details><summary>Indice 2</summary>

Relis le tableau des symptômes de la fiche §9.2. Le **niveau** de l'erreur d'entraînement se juge par rapport à une référence (ici, l'humain) ; l'**écart** entre entraînement et validation dit autre chose.

</details>
<details><summary>Indice 3</summary>

a) Modèle 1 : 0,5 % d'erreur sur ses propres exemples, mais 18 % sur des exemples nouveaux. L'écart est énorme : le modèle colle à ses exemples, c'est B. b) et c) Même méthode, dans l'ordre de l'indice 1 : compare d'abord le **niveau** de l'erreur d'entraînement à la référence humaine (2 %), puis l'**écart** avec la validation. Deux erreurs proches mais très au-dessus de la référence signalent un modèle trop rigide ; deux erreurs basses et proches, le bon réglage. d) Teste la phrase sur les trois modèles de a) à c) : lesquels ont un petit écart, et quel diagnostic as-tu posé pour chacun d'eux ?

</details>

### 9.Q2 — Walter et sa moustache : qu'est-ce qui a été mal appris ?

<details><summary>Indice 1</summary>

Sépare deux moments : pendant le mariage, avec les invités qu'on venait de rencontrer, et plus tard, avec d'autres personnes. À quoi correspond chacun en machine learning ?

</details>
<details><summary>Indice 2</summary>

Distingue ce qu'on a retenu (des prénoms) et la façon dont on les a reliés à l'apparence. Le §9.5 du livre revient sur ce que l'on aurait pu remarquer d'autre chez Walter.

</details>
<details><summary>Indice 3</summary>

a) Les prénoms ont bien été retenus (on les ressortait sans hésiter), et l'apparence des invités aussi : ce qui a trompé, c'est le **lien** entre chaque prénom et un seul détail, qu'un autre invité pouvait partager. C'est C. b) Pendant le mariage, les invités déjà rencontrés jouent le rôle des exemples d'entraînement : les reconnaissait-on ? c) Relis ce que le §9.5 du livre dit de ce qu'on aurait pu remarquer d'autre chez Walter : un inconnu partage-t-il aussi facilement plusieurs détails à la fois qu'un seul ? d) Dans l'histoire, la moustache est ce que l'on regardait pour retrouver un prénom : à quoi correspond-elle chez un modèle, et à quoi correspond le prénom ? Le lien entre les deux tenait-il hors du mariage ? Cherche la proposition qui décrit ce genre de lien (pense au décor des photos du ch. 8).

</details>

### 9.Q3 — Underfitting : les vrais remèdes

<details><summary>Indice 1</summary>

Les deux $R^2$ sont presque égaux et bas, alors qu'un autre modèle fait beaucoup mieux sur la même validation. Que dit cette comparaison ?

</details>
<details><summary>Indice 2</summary>

Pour chaque remède, demande-toi s'il rend le modèle plus **souple** (plus de capacité) ou s'il réduit sa **variance**. Le problème de ce modèle-là est-il la variance ?

</details>
<details><summary>Indice 3</summary>

a) Les deux $R^2$ sont presque égaux (0,45 et 0,44) : pas d'écart, donc pas le symptôme de l'overfitting. Mais ils sont bas, alors qu'un modèle plus souple atteint 0,80 sur la même validation : la régression linéaire est trop rigide pour ces données, c'est D. b) Range chaque remède selon son effet (plus de capacité, moins de variance, ou moins de capacité), puis garde ceux qui soignent le diagnostic de a). c) Plus d'exemples du même genre referme un écart entre entraînement et validation : regarde celui de l'énoncé, et le panneau (a) de la figure des courbes d'apprentissage de la fiche. d) L'encadré ⚠️ de la fiche §9.2.2 corrige le livre sur ce point, et dit dans quel cas plus de données aide vraiment.

</details>

### 9.Q4 — Courbes d'erreur : où commence l'overfitting ?

<details><summary>Indice 1</summary>

Lis séparément les deux lignes du tableau : comment chacune évolue-t-elle au fil des epochs ?

</details>
<details><summary>Indice 2</summary>

Repère le minimum de la ligne « validation ». Après lui, que fait la ligne « entraînement » ? Pour d), relis la fiche §9.2 : quelles sont les trois erreurs, et laquelle ne connaît-on jamais exactement ?

</details>
<details><summary>Indice 3</summary>

a) La plus petite valeur de la ligne « validation » est 0,46 : c'est l'epoch 20. b) Regarde la ligne « entraînement » après l'epoch 20 : que fait-elle, et qu'est-ce que cela dit de ce que le modèle tire encore de ses exemples ? c) À l'epoch 35, compare le sens des deux lignes : une erreur d'entraînement qui baisse pendant que la validation remonte, c'est le signe de la figure 9.1 du livre. A-t-on besoin du test pour le voir ? d) L'erreur de validation est une mesure sur un échantillon de données nouvelles : l'encadré ⚠️ de la fiche §9.3 dit ce qu'elle est par rapport à l'erreur de généralisation.

</details>

### 9.Q5 — Un point isolé : frontière tordue ou frontière simple ?

<details><summary>Indice 1</summary>

Relis la figure 9.6 du livre et le paragraphe « Le point isolé » de la fiche §9.3.

</details>
<details><summary>Indice 2</summary>

Pour b), compte les points d'entraînement mal classés par chaque frontière. Pour c), regarde de quel côté de la frontière simple tombe le voisinage du point isolé.

</details>
<details><summary>Indice 3</summary>

a) Un point d'une classe, isolé au milieu de l'autre classe, est un point **aberrant** : c'est D. Un centroïde est le centre d'un groupe, une fuite fait passer une information du test vers le modèle, et les vecteurs de support viendront avec les SVM (ch. 13). b) Compte les points d'entraînement mal classés par chaque frontière : le rond isolé est-il du bon côté de la frontière simple ? Et du détour ? c) Regarde de quel côté de la frontière simple tombe tout le voisinage du rond isolé, et quelle classe y domine. d) Le mot « toujours » doit tenir dans tous les cas : un point aberrant est-il forcément une erreur de mesure ? Demande-toi d'où il vient avant de décider (⚖️ 9.10).

</details>

### 9.Q6 — Early stopping : quand s'arrêter, et pourquoi c'est délicat

<details><summary>Indice 1</summary>

Relis la fiche §9.4, et son pseudo-code.

</details>
<details><summary>Indice 2</summary>

Quelle erreur estime la généralisation sans toucher au test ? Pour c), suis l'attente epoch par epoch après la meilleure, jusqu'à ce qu'elle atteigne la patience.

</details>
<details><summary>Indice 3</summary>

a) On décide sur l'erreur de validation : des quatre propositions, c'est la seule qui estime la généralisation sans toucher au test. C'est D. b) Une hausse d'une seule epoch, sur une courbe mesurée avec des mini-batches et un jeu de validation de taille finie, annonce-t-elle forcément l'overfitting ? Écarte aussi les propositions qui reposent sur un fait faux ou sur le chiffre d'une seule figure. c) Écris une ligne par epoch après la meilleure, avec l'attente : elle augmente de 1 à chaque epoch sans amélioration, et la règle s'arrête à la fin de l'epoch où elle atteint la patience. Compte ces lignes (la meilleure epoch elle-même n'en fait pas partie). d) Lis la dernière ligne du pseudo-code de la fiche §9.4, et méfie-toi de la valeur par défaut de Keras (encadré 🕰️ de la §9.4).

</details>

### 9.Q7 — Régularisation : ce que change λ

<details><summary>Indice 1</summary>

Écris la loss régularisée : loss sur les données + $\lambda$ × pénalité. Que se passe-t-il quand $\lambda$ pèse plus lourd ?

</details>
<details><summary>Indice 2</summary>

Pour b), la loss sur les données est la plus basse possible quand $\lambda = 0$ : que lui arrive-t-il quand la pénalité fait bouger la solution ? Pour d), relis l'encadré 💼 de la fiche §9.5.

</details>
<details><summary>Indice 3</summary>

a) Quand $\lambda$ pèse plus lourd dans la loss régularisée, la minimisation a intérêt à rendre la pénalité plus petite : la taille des poids, mesurée par la pénalité elle-même, diminue. C'est C (la pénalité ne rend pas pour autant les poids égaux). b) Prends $\lambda_1 < \lambda_2$ et leurs solutions $\mathbf{w}_1$ et $\mathbf{w}_2$. Écris que chacune est la meilleure pour son propre $\lambda$ (deux inégalités) et additionne-les : tu obtiens le sens de variation de la pénalité. Reporte-le dans l'une des deux inégalités pour comparer les losses sur les données. c) Si l'on cherchait $\lambda$ en minimisant la loss d'entraînement, quelle valeur trouverait-on toujours ? Un tel réglage servirait-il à quelque chose ? d) Relis l'encadré 💼 de la fiche §9.5 : `C` joue le rôle de quelle fonction de la force de régularisation ? e) Avec des poids écrasés vers 0, que reste-t-il de la prédiction, sachant que l'ordonnée à l'origine n'est pas pénalisée ?

</details>

### 9.Q8 — Pénalité sur les poids, dropout, batchnorm : même objectif ?

<details><summary>Indice 1</summary>

Relis la fin de la fiche §9.5 : la régularisation, le dropout, la batchnorm et l'encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

Pour a), demande-toi, pour chaque technique, si son but premier est de mieux généraliser ou d'aller plus vite. Pour d), compare les deux panneaux de la figure `l1_l2.png`.

</details>
<details><summary>Indice 3</summary>

a) Une pénalité L2 sur les poids (A) n'a qu'un but : empêcher les poids de grandir, pour mieux généraliser. A fait partie de ta réponse. Pose la même question à B, C, D et E : leur but premier est-il de mieux généraliser, ou d'aller plus vite ? Une régularisation peut aussi agir sur la façon d'entraîner, sans aucune pénalité. b) Relis l'encadré ⚠️ de la fiche §9.5 : que cherchaient d'abord S. Ioffe et C. Szegedy, et qu'ont-ils constaté ensuite ? c) Le dropout touche-t-il les exemples ou les neurones ? Une fois pour toutes, ou à chaque pas ? d) Compare les deux panneaux de la figure `l1_l2.png` : laquelle des deux pénalités met des poids exactement à zéro ? Un modèle qui ne garde que quelques features répartit-il l'importance entre toutes ?

</details>

### 9.Q9 — Biais et variance : des propriétés d'une famille de courbes

<details><summary>Indice 1</summary>

Relis les deux encadrés 🧮 du début de la fiche §9.6.

</details>
<details><summary>Indice 2</summary>

Le biais compare le modèle **moyen** à une référence ; la variance compare les modèles **entre eux**. Pour b), demande-toi ce que l'on cherche à retrouver en entraînant tous ces modèles.

</details>
<details><summary>Indice 3</summary>

a) Le biais compare le modèle **moyen** à la courbe idéale, et la variance mesure la **dispersion** des modèles autour de ce modèle moyen. Avec une seule courbe, il n'y a ni moyenne de plusieurs modèles ni dispersion : on ne peut mesurer ni l'un ni l'autre. Faux. b) Le biais est l'erreur **systématique** de la famille, celle qui reste quand on moyenne les modèles de tous les datasets possibles : à quoi faut-il comparer ce modèle moyen pour savoir s'il a appris la vraie relation, et non le hasard d'un dataset ? c) La variance compare les modèles **entre eux** : imagine une famille dont toutes les courbes sont identiques. Laquelle des quatre quantités vaudrait alors forcément 0 ? d) Écris la décomposition biais² + variance + bruit, et remplace les deux premiers termes par 0.

</details>

### 9.Q10 — Courbes raides ou souples : qui a quel biais, quelle variance ?

<details><summary>Indice 1</summary>

Regarde la figure `biais_variance.png` de la fiche, panneaux (a) et (b), et les figures 9.13 et 9.15 du livre.

</details>
<details><summary>Indice 2</summary>

Une droite peut-elle suivre une courbe ondulée ? Un polynôme de degré 15 sur 30 points bruités dépend-il beaucoup des points tirés ?

</details>
<details><summary>Indice 3</summary>

a) Une droite ne peut pas suivre une courbe ondulée : les 50 droites ratent la forme de la même façon, et leur modèle moyen aussi. Le biais le plus élevé est celui des droites : A. b) La variance mesure combien les courbes changent d'un jeu à l'autre : sans pénalité, un polynôme à 16 coefficients ajusté sur 30 points bruités dépend-il plus ou moins des points tirés qu'une droite ? c) Où les points qui retiennent un polynôme sont-ils les plus rares : au milieu de l'intervalle, ou à ses bords ? Pense à la figure 9.15 du livre. d) Avec plus de points par jeu, chaque polynôme dépend-il autant des points tirés ? Relis le panneau (b) des courbes d'apprentissage de la fiche §9.3.

</details>

### 9.Q11 — Droites a posteriori : a-t-on le droit de parler de variance ?

<details><summary>Indice 1</summary>

Combien de datasets y a-t-il dans l'approche bayésienne du §9.7 ? Et dans l'expérience du §9.6 ?

</details>
<details><summary>Indice 2</summary>

Les droites tirées dans le posterior sont des **hypothèses** pondérées par leur probabilité, pas des modèles entraînés sur des jeux différents. Que mesure leur dispersion ?

</details>
<details><summary>Indice 3</summary>

La dispersion des droites tirées mesure l'incertitude qui reste sur la bonne droite, compte tenu du prior et des points observés (ch. 4 : c'est elle que résume un intervalle de crédibilité). Pour la deuxième question, compte les datasets et les algorithmes de chaque côté : la variance de la fiche §9.6 compare les modèles d'**un** algorithme entraîné sur **plusieurs** datasets ; ici, que fait-on varier ? Relis le dernier paragraphe de la fiche §9.7 avant l'encadré 🕰️. Pour la troisième, demande-toi ce que devient cette incertitude avec plus de points.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 9.R1 — Ch. 8 : pourquoi le score de validation du modèle retenu est optimiste

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « la loi du maximum » de la fiche du ch. 8 (§8.4).

</details>
<details><summary>Indice 2</summary>

Le degré retenu a été choisi **parce que** son erreur de validation était la plus basse. Une erreur mesurée sur un échantillon fini contient une part de chance.

</details>
<details><summary>Indice 3</summary>

Première question : on garde le degré dont l'erreur de validation est la plus basse, chance comprise. Cette erreur est donc le minimum de 25 mesures bruitées, et le minimum de plusieurs mesures bruitées est en moyenne plus bas que la vraie erreur du degré retenu (∂ 8.6). Deuxième question : le choix du degré est lui aussi un apprentissage. Sur quelles données s'est-il fait, et à quoi s'est-il donc adapté ? Fais le parallèle avec un modèle et ses exemples d'entraînement. Troisième question : pour une estimation honnête, de quelles données as-tu besoin, et combien de fois as-tu le droit de t'en servir (ch. 8) ?

</details>

### 9.R2 — Ch. 6 : ce que mesure une cross-entropy utilisée comme loss

<details><summary>Indice 1</summary>

La cross-entropy d'un exemple est la surprise de la bonne classe : $-\ln p$ (ch. 6).

</details>
<details><summary>Indice 2</summary>

Calcule $-\ln p$ pour chacune des quatre probabilités, puis fais la moyenne. Pour passer des nats aux bits, souviens-toi que $\log_2 x = \ln x / \ln 2$.

</details>
<details><summary>Indice 3</summary>

a) $-\frac{1}{4}(\ln 0{,}9 + \ln 0{,}6 + \ln 0{,}25 + \ln 0{,}8) = -\frac{1}{4}\ln(0{,}9 \times 0{,}6 \times 0{,}25 \times 0{,}8) = -\frac{1}{4}\ln 0{,}108 \approx \frac{2{,}2256}{4} \approx 0{,}556$ (en nats). b) Passe ce résultat en bits avec la relation de l'indice 2 : une loss en nats se divise par $\ln 2$. c) Chaque exemple ajoute $-\ln p$ à la somme de a) : comment ce terme varie-t-il quand $p$ diminue ? Déduis-en l'exemple qui pèse le plus. d) Un exemple bien classé avec une probabilité de 0,9 a-t-il une loss nulle ? Que fait la descente de gradient tant que la loss n'est pas nulle ?

</details>

### 9.R3 — Ch. 2 : biais et variance d'un estimateur, et le bootstrap

<details><summary>Indice 1</summary>

Calcule d'abord la moyenne, puis la somme des carrés des écarts à la moyenne.

</details>
<details><summary>Indice 2</summary>

Le biais d'un estimateur est $\mathbb{E}[\hat{\theta}] - \theta$ : remplace $\mathbb{E}[\hat{\theta}]$ par l'expression donnée et $\theta$ par $\sigma^2$.

</details>
<details><summary>Indice 3</summary>

a) La moyenne vaut $(3 + 5 + 6 + 10)/4 = 6$ ; les écarts −3, −1, 0 et 4 ont des carrés qui somment à $9 + 1 + 0 + 16 = 26$ ; divisé par $n = 4$ : $26/4 = 6{,}5$. b) La même somme des carrés, divisée par $n - 1$. c) Le biais vaut $\frac{n-1}{n}\sigma^2 - \sigma^2$ : mets $\sigma^2$ en facteur, puis prends $n = 4$. d) Relis « Ce que le bootstrap ne fait pas » dans la fiche du ch. 2 (§2.6) : rééchantillonner un échantillon biaisé crée-t-il les personnes qu'il ne contient pas ? e) Mets les deux définitions côte à côte : une estimation moyennée sur tous les échantillons possibles, comparée à la vraie valeur. Pour une famille de modèles, qu'est-ce qui joue le rôle de l'estimation, et celui de la vraie valeur ?

</details>

<a id="papier"></a>

## ✏️ ∂ 📈 Papier-crayon

### Ex 9.1 — MSE et R² à la main sur cinq points ✏️

<details><summary>Indice 1</summary>

Commence par écrire les cinq résidus $y_i - \hat{y}_i$ dans une ligne du tableau : tout le reste en découle.

</details>
<details><summary>Indice 2</summary>

MSE : moyenne des carrés des résidus ; MAE : moyenne de leurs valeurs absolues ; $R^2 = 1 - SS_{\text{res}}/SS_{\text{tot}}$, avec $SS_{\text{tot}}$ calculé autour de la moyenne des **cibles**. Pour e), les résidus changent, pas $SS_{\text{tot}}$.

</details>
<details><summary>Indice 3</summary>

a) Les résidus valent 1 ; −0,5 ; −1 ; 0,5 ; 1, leurs carrés 1 ; 0,25 ; 1 ; 0,25 ; 1, de somme 3,5 : MSE $= 3{,}5/5 = 0{,}7$. b) La racine de a). c) La moyenne des valeurs absolues des mêmes cinq résidus. d) $\bar{y} = 30/5 = 6$, $SS_{\text{tot}} = \sum_i (y_i - 6)^2$, puis $R^2 = 1 - 3{,}5/SS_{\text{tot}}$. e) Les résidus deviennent $y_i - 8$ ; $SS_{\text{tot}}$ ne change pas. f) et g) Un seul résidu change (le 5ᵉ, qui passe de 1 à 11) : corrige la somme des carrés et la somme des valeurs absolues, puis divise par 5. h) Divise chaque nouvelle mesure par l'ancienne, puis demande-toi laquelle se laisse dominer par un seul résidu.

</details>

### Ex 9.2 — Moindres carrés : la meilleure droite par dérivées partielles ∂

<details><summary>Indice 1</summary>

$L$ est une somme de carrés : dérive chaque terme $(y_i - a x_i - b)^2$ par rapport à $b$, puis à $a$, avec la règle de la chaîne (0B).

</details>
<details><summary>Indice 2</summary>

La condition sur $b$ donne $\sum_i y_i = a\sum_i x_i + n\,b$. Dans la condition sur $a$, remplace $b$ et utilise le fait que $\sum_i (y_i - \bar{y}) = 0$ et $\sum_i (x_i - \bar{x}) = 0$ pour faire apparaître des écarts à la moyenne.

</details>
<details><summary>Indice 3</summary>

1. $\frac{\partial L}{\partial b} = -2\sum_i (y_i - a\,x_i - b)$ ; l'annuler donne $\sum_i y_i - a\sum_i x_i - n\,b = 0$, et en divisant par $n$ : $b = \bar{y} - a\,\bar{x}$. 2. Même méthode pour $a$, avec les deux sommes nulles de l'indice 2. 3. La condition sur $b$ dit déjà quelque chose de la somme des résidus ; pour le point moyen, calcule $a\,\bar{x} + b$ avec l'expression de $b$. Application, sans arrondir : a) $\bar{x} = 2$ et $\bar{y} = 3{,}6$ ; écarts en $x$ : −2, −1, 0, 1, 2 ; en $y$ : −2,6 ; −0,6 ; −1,6 ; 1,4 ; 3,4. Multiplie-les deux à deux et additionne, puis divise par $\sum_i (x_i - \bar{x})^2$. b) $b = \bar{y} - a\,\bar{x}$, avec ton $a$. c) et d) Calcule les cinq prédictions $a\,x_i + b$ et les résidus, puis leur somme (compare-la à la démonstration 3) et la somme de leurs carrés. e) $a \times 6 + b$. f) $\det\begin{pmatrix} p & q \\ q & r \end{pmatrix} = pr - q^2$, avec $p = n$, $q = \sum_i x_i$ et $r = \sum_i x_i^2$. 4. Compare ce déterminant à $n \sum_i (x_i - \bar{x})^2$.

</details>

### Ex 9.3 — Ridge en dimension 1 : w* = Σxy / (Σx² + λ) ∂

<details><summary>Indice 1</summary>

$L(w)$ est une fonction d'une seule variable : annule sa dérivée.

</details>
<details><summary>Indice 2</summary>

$L'(w) = -2\sum_i x_i (y_i - w x_i) + 2\lambda w$ ; regroupe les termes en $w$. Pour les applications, calcule une fois pour toutes $\sum x_i y_i$ et $\sum x_i^2$.

</details>
<details><summary>Indice 3</summary>

1. $L'(w) = -2\sum_i x_i y_i + 2w\big(\sum_i x_i^2 + \lambda\big)$ s'annule en $w^* = \frac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda}$, et $L''(w) = 2\big(\sum_i x_i^2 + \lambda\big) > 0$ : c'est un minimum. 2. Dans cette fraction, qu'est-ce qui dépend de $\lambda$ : le numérateur, ou le dénominateur ? Que devient-elle quand $\lambda \to \infty$, et peut-elle valoir 0 pour un $\lambda$ fini ? Applications : calcule une fois pour toutes $\sum x_i y_i$ et $\sum x_i^2$, puis la fraction pour $\lambda = 0$ ; 2,5 et 5 (a, b et c). d) Écris l'équation « $w^*(\lambda) = w^*(0)/2$ » et résous-la en $\lambda$. e) Calcule les cinq résidus $y_i - w^* x_i$ avec le $w^*$ de cette valeur de $\lambda$, puis la somme de leurs carrés, sans ajouter la pénalité. f) Regarde la somme des carrés comme une parabole en $w$ : où est son minimum, et dans quel sens $w^*$ s'en éloigne-t-il quand $\lambda$ augmente ?

</details>

### Ex 9.4 — Biais² et variance à partir d'un tableau de prédictions ✏️

<details><summary>Indice 1</summary>

Le modèle moyen est la moyenne de chaque **colonne** du tableau (un point à la fois, sur les quatre modèles).

</details>
<details><summary>Indice 2</summary>

Biais² : en chaque point, (modèle moyen − $f$)², puis moyenne sur les trois points. Variance : en chaque point, la moyenne des carrés des écarts des quatre modèles au modèle moyen (divise par 4), puis moyenne sur les trois points.

</details>
<details><summary>Indice 3</summary>

a) Au point $x_1$ : $(1{,}5 + 0{,}5 + 1 + 1)/4 = 1$ ; même calcul pour les colonnes $x_2$ et $x_3$. b) En chaque point, (modèle moyen − $f$)², puis la moyenne des trois carrés : élève au carré **avant** de moyenner. c) En chaque point, la moyenne des carrés des écarts des quatre modèles au modèle moyen (divise par 4), puis la moyenne sur les trois points. d) Calcule les douze erreurs au carré face à $f$ et divise leur somme par 12 ; compare ensuite ton résultat à b) + c). e) Ajoute le bruit (encadré de la décomposition). f) L'erreur du modèle moyen face à $f$ est un des termes que tu as déjà calculés : lequel ? Compare-le à d). g) Regarde lequel des trois carrés de b) est le plus grand, et si les quatre modèles se trompent dans le même sens en ce point.

</details>

### Ex 9.5 — Early stopping avec patience sur une courbe de loss ✏️

<details><summary>Indice 1</summary>

Fais un tableau à trois lignes sous celui de l'énoncé : la meilleure loss jusqu'ici, l'attente, l'epoch des poids gardés.

</details>
<details><summary>Indice 2</summary>

À chaque epoch : si la loss est strictement sous (meilleure − `min_delta`), elle devient la meilleure et l'attente revient à 0 ; sinon, l'attente augmente de 1, et l'on s'arrête quand elle atteint la patience.

</details>
<details><summary>Indice 3</summary>

a) Patience 2 : les epochs 1 à 5 améliorent toutes (meilleure loss 0,53 à l'epoch 5, attente 0) ; l'epoch 6 (0,54) n'améliore pas (attente 1) ; l'epoch 7 (0,52) améliore (meilleure loss 0,52, attente 0) ; les epochs 8 (0,53) et 9 (0,55) n'améliorent pas (attente 1, puis 2 = patience) : arrêt à la fin de l'epoch 9. b) Relis la dernière ligne du pseudo-code de la fiche : quels poids recharge-t-on, et où sont-ils dans ta trace ? c) et d) Patience 1 : refais la trace depuis l'epoch 1 ; une seule epoch sans amélioration suffit à arrêter. e) Patience 4 : reprends la trace de a) après l'epoch 7, et continue jusqu'à quatre epochs de suite sans amélioration. f) et g) Avec `min_delta = 0,02`, une epoch améliore seulement si sa loss est **strictement** sous (meilleure − 0,02) : refais toute la trace depuis l'epoch 1 en écrivant ce seuil sous chaque epoch. Une baisse égale à `min_delta` ne compte pas, et seule une amélioration met à jour la meilleure loss et les poids gardés. h) Compare c-d) à a-b) : qu'a changé la patience ? Pour une patience trop grande, demande-toi ce qu'elle coûte après le vrai minimum, selon les poids qu'on garde.

</details>

### Ex 9.6 — Lasso en dimension 1 : le seuillage doux et les zéros exacts ∂

<details><summary>Indice 1</summary>

Sur $w > 0$, $|w| = w$ ; sur $w < 0$, $|w| = -w$ : sur chaque demi-droite, $g$ est une parabole dérivable.

</details>
<details><summary>Indice 2</summary>

Pour 2, minore $-wz$ par $-|w|\,|z|$. Pour 3, développe la somme des carrés et fais apparaître $\frac{z}{2}\,(w - \rho/z)^2$ plus une constante (mise sous forme canonique).

</details>
<details><summary>Indice 3</summary>

a) $|2{,}5| > 1$ : on rapproche 2,5 de 0 de $\gamma = 1$, et $S(2{,}5 ;\ 1) = 1{,}5$. b) et c) Même règle : compare d'abord $|z|$ à $\gamma$ ; si $|z| \leq \gamma$, le résultat vaut exactement 0, sinon $z$ se rapproche de 0 de $\gamma$, en gardant son signe. d) $\rho = \frac{1}{n}\sum_i x_i y_i$, avec les données de 9.3 ($n = 5$). e) et f) $w^* = S(\rho, \alpha)/z$, avec $z = \frac{1}{n}\sum_i x_i^2$ : seuille d'abord $\rho$, puis divise par $z$. g) $w^* = 0$ exactement quand $|\rho| \leq \alpha$ : quelle est la plus petite valeur de $\alpha$ qui vérifie cette condition ? h) Relis la démonstration 2 de 9.3 : le numérateur de $w^*$ change-t-il avec $\lambda$ ? 4. Pense à ce qu'une feature doit apporter pour « payer » sa pénalité, puis à deux colonnes presque identiques, dont chacune coûte la même pénalité.

</details>

### Ex 9.7 — Mise à jour bayésienne d'une droite sur une grille 3 × 3 ✏️

<details><summary>Indice 1</summary>

Fais un tableau de neuf lignes, une par droite $(a, b)$ : le poids du prior, l'écart au premier point, sa vraisemblance, le produit.

</details>
<details><summary>Indice 2</summary>

Pour $P_1 = (1, 1)$, l'écart vaut $r_1 = 1 - (a \times 1 + b)$ ; pour $P_2 = (-1, 0)$, $r_2 = 0 - (a \times (-1) + b)$. Pour b), normalise les produits de la première étape (divise par leur somme).

</details>
<details><summary>Indice 3</summary>

a) Les poids du prior somment à $4 + 4 \times 2 + 4 \times 1 = 16$ : la droite $(0, 0)$ a la probabilité $4/16 = 0{,}25$. b) Pour chaque droite, écris $r_1 = 1 - a - b$, sa vraisemblance $L_1$, puis le produit prior × $L_1$ ; la probabilité a posteriori de $(0, 1)$ est son produit divisé par la somme des neuf produits. c) Compte les droites dont la vraisemblance $L_1$ est nulle ($|r_1| \geq 3$). d) et e) Multiplie les produits de la première étape par la vraisemblance de $P_2$ (avec $r_2 = a - b$), puis normalise par la **nouvelle** somme (normaliser aussi entre les deux étapes ne change rien : seule la normalisation finale compte). f) Le posterior final est proportionnel à prior × $L_1$ × $L_2$ : l'ordre des facteurs compte-t-il ? g) Cherche le plus grand des neuf produits prior × $L_1$ × $L_2$. h) Regarde la liste des neuf droites de la grille : la droite exacte en fait-elle partie ?

</details>

### Ex 9.9 — Diagnostiquer quatre paires de courbes d'entraînement et de validation 📈

<details><summary>Indice 1</summary>

Pour chaque panneau, regarde d'abord le **niveau** final des deux courbes, puis leur **écart**, puis leur **ordre** (laquelle est au-dessus).

</details>
<details><summary>Indice 2</summary>

Relis le tableau des symptômes de la fiche §9.2. Une validation qui remonte pendant que l'entraînement baisse signale une chose ; deux courbes hautes et collées, une autre.

</details>
<details><summary>Indice 3</summary>

a) Panneau 4 : les deux losses plafonnent vite et haut (vers 0,6, alors que les panneaux 1 et 3 descendent bien plus bas sur la même tâche), et restent collées : c'est l'underfitting. b) Cherche le panneau où la loss d'entraînement continue de descendre pendant que celle de validation remonte après un minimum. c) Dans ce panneau, lis la courbe de validation aux quatre epochs proposées, et garde la plus basse. d) Pour chaque proposition, demande-toi si elle peut faire passer la loss de validation **sous** celle d'entraînement : une technique qui n'agit que pendant l'entraînement ; un modèle qui sous-apprend (regarde où est sa validation sur le panneau 4) ; un jeu de validation qui ne ressemble pas à celui d'entraînement ; un learning rate trop petit, qui ralentit les deux courbes. e) Relis ton diagnostic de a), puis le quiz Q3 : plus de données soigne-t-il ce diagnostic ? f) Pour chaque panneau, pars de ton diagnostic et cherche le remède qui le soigne (le tableau de la fiche §9.2.2).

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 9.8 — Le compromis biais-variance raconté avec le tempo de la boutique 🗣️

<details><summary>Indice 1</summary>

Imagine que la propriétaire refasse la même journée dix fois, avec d'autres morceaux : que deviendrait chaque courbe ?

</details>
<details><summary>Indice 2</summary>

La première courbe change beaucoup d'une journée à l'autre (variance) ; la seconde change peu, mais rate toujours les mêmes choses (biais).

</details>
<details><summary>Indice 3</summary>

Cherche un exemple où l'on peut réagir trop à un seul jour (un parapluie, une tenue, un itinéraire) et ignorer une tendance réelle. Termine par ce que fait la bonne courbe.

</details>

### Ex 9.10 — Écarter un point aberrant : nettoyage ou manipulation ? ⚖️

<details><summary>Indice 1</summary>

Sépare trois décisions dans le cas 1 : retirer des points, les retirer aussi du test, et ne pas le dire.

</details>
<details><summary>Indice 2</summary>

Une valeur peut être **aberrante** (rare) sans être **erronée** (fausse). Sur quelle population le modèle du cas 1 sera-t-il utilisé ?

</details>
<details><summary>Indice 3</summary>

Pense à la MAE et à la loss de Huber (fiche §9.2), aux résultats publiés avec et sans les points retirés, à des règles fixées avant de regarder le test, et à qui décide (le métier, pas seulement le data scientist).

</details>

### Ex 9.11 — Belkin et coll. (2019) : la double descente 📄

<details><summary>Indice 1</summary>

Commence par la figure 1 de l'article, puis le résumé : ils disent l'essentiel.

</details>
<details><summary>Indice 2</summary>

Dans la section sur les features de Fourier aléatoires, cherche la taille du jeu d'entraînement, la règle de choix du prédicteur quand $N > n$, et la figure qui montre l'erreur de test et la norme en fonction de $N$.

</details>
<details><summary>Indice 3</summary>

La conclusion de l'article (*Concluding thoughts*) explique pourquoi le pic a été peu vu : cherche le rôle de la régularisation. Pour 5, relis l'encadré 🕰️ de la fiche §9.6.4.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 9.E1 — Expliquer le compromis biais-variance

<details><summary>Indice 1</summary>

Définis d'abord les deux mots avec l'image de beaucoup de datasets.

</details>
<details><summary>Indice 2</summary>

Cite la décomposition (biais² + variance + bruit) et ce que fait la capacité à chaque terme.

</details>
<details><summary>Indice 3</summary>

Termine par le concret : comment tu le diagnostiques (erreur d'entraînement, écart avec la validation) et les leviers sur chaque terme (régularisation, données, features, ensembles).

</details>

### 9.E2 — MSE ou MAE : laquelle choisir, et pourquoi ?

<details><summary>Indice 1</summary>

Qu'est-ce qui coûte cher au métier : une grosse erreur rare, ou beaucoup de petites ?

</details>
<details><summary>Indice 2</summary>

Une erreur de 10 pèse 10 fois plus qu'une erreur de 1 pour la MAE, et 100 fois plus pour la MSE. Quelle prédiction constante minimise chacune ?

</details>
<details><summary>Indice 3</summary>

MSE : moyenne, sensible aux valeurs aberrantes ; MAE : médiane, robuste. Mentionne la RMSE (unité), la loss de Huber, et les coûts asymétriques (loss quantile).

</details>

### 9.E3 — Régularisation L1 ou L2 : différences et usages

<details><summary>Indice 1</summary>

Écris les deux pénalités, puis dis ce que chacune fait aux poids.

</details>
<details><summary>Indice 2</summary>

L'une rétrécit, l'autre seuille (figure `l1_l2.png`) : laquelle sélectionne des features ? Laquelle répartit le poids entre deux features corrélées ?

</details>
<details><summary>Indice 3</summary>

N'oublie pas Elastic Net, la standardisation préalable, le choix de $\lambda$ par validation croisée, et le weight decay des réseaux.

</details>

### 9.E4 — Détecter l'overfitting avant la mise en production

<details><summary>Indice 1</summary>

Avant tout diagnostic, il faut un protocole : de quoi as-tu besoin pour mesurer honnêtement (ch. 8) ?

</details>
<details><summary>Indice 2</summary>

Cite au moins trois signaux : l'écart entre entraînement et validation, les courbes au fil des epochs, la courbe d'apprentissage, la stabilité entre folds.

</details>
<details><summary>Indice 3</summary>

Donne un ordre d'action : early stopping et régularisation, simplifier, puis plus de données ; et la vérification finale sur un test qui ressemble à la production.

</details>

### 9.E5 — La double descente contredit-elle le compromis biais-variance ?

<details><summary>Indice 1</summary>

Rappelle la courbe en U, puis ce qui se passe au seuil d'interpolation.

</details>
<details><summary>Indice 2</summary>

Parmi toutes les solutions qui passent par les points, laquelle choisit-on, et pourquoi généralise-t-elle ?

</details>
<details><summary>Indice 3</summary>

Ce qui tombe : « zéro erreur d'entraînement = overfitting ». Ce qui reste : le compromis pour les modèles de taille raisonnable, et la validation pour décider. Ajoute la nuance de Curth et coll. (2023).

</details>

<a id="notebook"></a>

## Notebook, parties A à D

Les exercices du notebook (`03_notebook.ipynb`). Pour ceux qui complètent `mylearn/linear.py`, la docstring de chaque fonction décrit déjà ce qu'elle doit faire, cas d'erreur compris : relis-la avant d'ouvrir un indice.

### Ex 9.12 — Le tempo de la boutique : polynômes de degré 1, 4 et 15 📦

<details><summary>Indice 1</summary>

Les trois modèles se construisent et s'entraînent de la même façon : une compréhension de dictionnaire sur `(1, 4, 15)` suffit. scikit-learn attend un tableau à deux dimensions : une ligne par réglage, une colonne pour l'heure.

</details>
<details><summary>Indice 2</summary>

`make_pipeline(PolynomialFeatures(d, include_bias=False), LinearRegression())`, puis `.fit(x_day1.reshape(-1, 1), tempo_day1)`, qui renvoie le pipeline lui-même. Une MSE : `mean_squared_error(tempo_day1, models_12[d].predict(x_day1.reshape(-1, 1)))` ; pour le lendemain, la même chose avec `x_day2` et `tempo_day2`. Garde l'ordre des degrés 1, 4, 15.

</details>
<details><summary>Indice 3</summary>

La clé : `.fit(...)` renvoie le pipeline lui-même, si bien qu'une seule compréhension de dictionnaire construit **et** entraîne les trois modèles.

```python
models_12 = {degree: make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
             .fit(x_day1.reshape(-1, 1), tempo_day1) for degree in (1, 4, 15)}
# a) train_mse_12: a list comprehension over (1, 4, 15), the MSE of models_12[d] on day 1
# b) day2_mse_12: the same comprehension, with x_day2 and tempo_day2
```

L'expression d'une MSE est dans l'indice 2. Les deux listes suivent l'ordre des degrés 1, 4, 15 ; n'arrondis rien, la vérification s'en charge.

</details>

### Ex 9.13 — Erreurs d'entraînement et de test selon le degré : ta courbe d'abord 🔮

<details><summary>Indice 1</summary>

Relis le point 8 de « L'essentiel » : en fonction de la capacité, l'erreur sur des données nouvelles dessine un U. Et l'erreur d'entraînement : plus de capacité peut-il jamais faire moins bien sur les points d'entraînement ? Tes trois prédictions portent sur la largeur du fond du U et sur la hauteur de sa branche droite ; 9.12 t'en a déjà donné trois points (degrés 1, 4 et 15).

</details>
<details><summary>Indice 2</summary>

a) Le degré 1 ne voit qu'une tendance. Dès que le polynôme suit la forme de la journée, il fait mieux que lui, jusqu'au moment où il suit aussi les hésitations du premier jour : combien de degrés tiennent dans ce fond de vallée ? b) Où la forme est-elle suivie sans plus ? c) La constante fait, le lendemain, à peu près la variance des réglages du lendemain : pour faire pire, un polynôme doit s'écarter franchement de la journée quelque part. Où, et à partir de quel degré ?

</details>
<details><summary>Indice 3</summary>

a) Le degré 1 ne suit que la tendance de la journée (9.12). Demande-toi jusqu'à quel degré le polynôme a encore trop peu de coefficients pour suivre chaque hésitation des 16 réglages : tant que c'est le cas, la forme qu'il capte compte plus que le bruit qu'il apprend, et il fait mieux que la droite. Ta réponse est le nombre de degrés de cette vallée, sans compter le degré 1 lui-même. b) Le minimum est là où le polynôme suit la forme de la journée sans ses hésitations : relis les trois courbes de 9.12, et demande-toi à partir de quel degré cette forme est suivie. c) Le degré 15 passe par les 16 réglages ; un degré un peu plus petit a-t-il beaucoup plus de liberté ? Pense aux bords de la journée, où aucun réglage du premier jour ne retient le polynôme, et au réglage de 16 h 15 du lendemain.

</details>

### Ex 9.14 — mean_squared_error, mean_absolute_error et r2_score 🔨

<details><summary>Indice 1</summary>

Les trois fonctions commencent de la même façon : convertir les deux arguments en tableaux de `float`, vérifier qu'ils ont la même longueur et assez d'éléments. Écris ce contrôle une seule fois, dans une petite fonction auxiliaire dont le nom commence par `_` ; chaque mesure tient ensuite en une ou deux lignes.

</details>
<details><summary>Indice 2</summary>

`np.asarray(y, dtype=float).ravel()` ; compare les deux `len(...)` **avant** tout calcul, sinon NumPy diffuse un tableau d'un seul élément sur l'autre sans rien dire. MSE : `np.mean((y_true - y_pred) ** 2)` ; MAE : `np.mean(np.abs(y_true - y_pred))` ; $R^2$ : `1 - ss_res / ss_tot`, avec `ss_tot = np.sum((y_true - y_true.mean()) ** 2)` (la moyenne de `y_true`, pas celle de `y_pred`), et le cas `ss_tot == 0` traité à part. Renvoie `float(...)`.

</details>
<details><summary>Indice 3</summary>

Le contrôle, écrit une seule fois, sert aux trois mesures :

```python
def _check_targets(y_true, y_pred, min_samples):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    # compare the two lengths BEFORE any computation (1 against n would broadcast): ValueError
    # fewer than min_samples values: ValueError
    # return both arrays
```

Chaque mesure appelle `_check_targets` (`min_samples=1`, ou 2 pour `r2_score`), puis tient en une ligne, convertie avec `float(...)`. Les deux lignes clés de `r2_score` : `ss_res = np.sum((y_true - y_pred) ** 2)` et `ss_tot = np.sum((y_true - y_true.mean()) ** 2)`, avec la moyenne de `y_true`. Si `ss_tot == 0`, rends 1.0 quand `ss_res` vaut aussi 0, et 0.0 sinon ; sinon, `1 - ss_res / ss_tot`, sans ramener un résultat négatif à 0.

</details>

### Ex 9.15 — polynomial_features, interactions comprises 🔨

<details><summary>Indice 1</summary>

a) Relis la formule de l'encadré de la fiche : elle compte les monômes de degré 1 à $d$ en $p$ variables, sans la constante (c'est le « − 1 »). Pour la fonction : une boucle sur les degrés, puis une boucle sur les combinaisons d'indices de colonnes ; chaque combinaison donne une colonne.

</details>
<details><summary>Indice 2</summary>

a) $\binom{p + d}{d} - 1$ (`math.comb`). La fonction : `X = np.asarray(X, dtype=float)`, puis `X.reshape(-1, 1)` si `X.ndim == 1` ; `for d in range(1, degree + 1)`, puis `for combo in combinations_with_replacement(range(X.shape[1]), d)` ; la colonne vaut `np.prod(X[:, list(combo)], axis=1)`. Rassemble les colonnes avec `np.column_stack`, après une colonne de 1 si `include_bias`. Vérifie `degree` et `X.ndim` en premier.

</details>
<details><summary>Indice 3</summary>

a) `math.comb(4 + 3, 3) - 1` : les monômes de degré 0 à 3 en 4 variables, moins la constante. La fonction, dans `mylearn/linear.py` :

```python
def polynomial_features(X, degree=2, include_bias=False):
    # 1. X as a float array; a 1-D array is ONE feature: X.reshape(-1, 1)
    # 2. ValueError if X has more than 2 dimensions, or if degree < 1
    # 3. columns: a column of ones first if include_bias, then degree by degree:
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(X.shape[1]), d):
            ...                                 # one column: the product of the columns listed in combo
    # 4. np.column_stack(columns)
```

Le piège : `combinations_with_replacement` garde les carrés ($x_0 x_0$) et ne donne $x_0 x_1$ qu'une fois ; `itertools.product` donnerait aussi $x_1 x_0$, et `combinations` oublierait les carrés.

</details>

### Ex 9.16 — LinearRegression par moindres carrés 🔨

<details><summary>Indice 1</summary>

Trois temps dans `fit` : contrôler `X` et `y`, centrer, résoudre. L'ordonnée à l'origine ne se résout pas : elle se retrouve après coup avec les moyennes (∂ 9.2). Les contrôles resserviront dans `Ridge` et `Lasso` : mets-les dans une fonction auxiliaire.

</details>
<details><summary>Indice 2</summary>

`x_mean, y_mean = X.mean(axis=0), y.mean()` ; `w = np.linalg.lstsq(X - x_mean, y - y_mean, rcond=None)[0]` ; `b = float(y_mean - x_mean @ w)`. Sans ordonnée à l'origine, `lstsq` directement sur `X` et `y`, et `b = 0.0`. Range-les dans `self.coef_` et `self.intercept_`, rien d'autre de public, puis `return self`. `predict` : `X @ self.coef_ + self.intercept_` ; `score` : ton `r2_score(y, self.predict(X))`.

</details>
<details><summary>Indice 3</summary>

Les contrôles d'abord, dans une fonction auxiliaire `_check_X_y(X, y)` qui resservira : `X` en `float` et en 2 dimensions, `y` aplati, autant de lignes que de valeurs, sinon une `ValueError` qui dit ce qui ne va pas. Puis `fit` :

```python
    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        # with an intercept: centre, solve on the centred data, recover b from the means
        x_mean, y_mean = X.mean(axis=0), y.mean()
        w = np.linalg.lstsq(X - x_mean, y - y_mean, rcond=None)[0]
        b = float(y_mean - x_mean @ w)
        # without an intercept: lstsq directly on X and y, and b = 0.0
        # store coef_ and intercept_ (no other public attribute), then return self
```

N'ajoute pas de colonne de 1 à `X` : quand des colonnes sont redondantes, la solution de norme minimale ne serait plus celle des tests.

</details>

### Ex 9.17 — Ridge en forme fermée, intercept non pénalisé 🔨

<details><summary>Indice 1</summary>

C'est ta `LinearRegression` avec deux différences : la vérification de `alpha`, et la résolution, qui utilise le système de l'encadré 🧮 « Ridge en forme fermée » au lieu de `lstsq`. Le centrage fait le reste : sur des données centrées, l'ordonnée à l'origine sort du problème, donc de la pénalité.

</details>
<details><summary>Indice 2</summary>

Vérifie `alpha` au début de `fit` (pas dans `__init__`). Puis `Xc, yc = X - x_mean, y - y_mean` (sans ordonnée à l'origine : `Xc, yc = X, y`) ; `A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])` ; `w = np.linalg.solve(A, Xc.T @ yc)` ; `b = y_mean - x_mean @ w`. Avec `alpha = 0`, tu dois retrouver les moindres carrés.

</details>
<details><summary>Indice 3</summary>

Même squelette que ta `LinearRegression`, avec deux changements :

```python
    def fit(self, X, y):                        # in the class Ridge; predict and score as in LinearRegression
        # 1. alpha < 0: ValueError, here in fit (not in __init__)
        # 2. X, y = _check_X_y(X, y); x_mean, y_mean: the means (zeros and 0.0 without an intercept)
        Xc, yc = X - x_mean, y - y_mean
        w = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ yc)   # never an inverse
        # 3. coef_ = w; intercept_ = float(y_mean - x_mean @ w), 0.0 without an intercept; return self
```

L'identité a la taille $p$ (`X.shape[1]`, le nombre de features), pas $n$ ; et c'est le centrage qui sort l'ordonnée à l'origine de la pénalité.

</details>

### Ex 9.18 — Courbes de validation : le degré, puis λ 🔬

<details><summary>Indice 1</summary>

Deux boucles imbriquées : sur les valeurs de `values`, puis sur les folds. Chaque fold demande un modèle **neuf**, entraîné sur sa partie d'entraînement, puis deux MSE. Ce sont ces MSE que tu moyennes sur les folds, une paire de moyennes par valeur.

</details>
<details><summary>Indice 2</summary>

`for train_idx, val_idx in folds:` ; `model = make_model(value)` ; `model.fit(x[train_idx], y[train_idx])` ; puis `mse(y[train_idx], model.predict(x[train_idx]))` et `mse(y[val_idx], model.predict(x[val_idx]))` (la fonction `mse` est fournie). Ajoute à chacune des deux listes la moyenne des cinq MSE (`float(np.mean(...))`), puis renvoie le couple de listes. Ne réutilise pas un modèle d'un fold à l'autre.

</details>
<details><summary>Indice 3</summary>

Le squelette, avec ses deux boucles :

```python
def validation_curve_18(make_model, values, x, y, folds):
    train_mse, val_mse = [], []
    for value in values:
        # two lists, for the MSE of the 5 folds
        for train_idx, val_idx in folds:
            model = make_model(value)                       # a NEW model for each fold
            model.fit(x[train_idx], y[train_idx])
            # mse(y[train_idx], ...) and mse(y[val_idx], ...), with model.predict
        # append to train_mse and to val_mse the mean over the folds, as a float
    return train_mse, val_mse
```

On moyenne les **MSE** des folds (pas leurs racines), et l'on rend une valeur par élément de `values`, dans leur ordre.

</details>

### Ex 9.19 — Que deviennent les coefficients quand λ grandit ? 🔮

<details><summary>Indice 1</summary>

Relis ∂ 9.3 : en dimension 1, Ridge divise le poids par un facteur qui grandit avec $\lambda$. Mais ici, les six colonnes $x, x^2, \dots, x^6$ se ressemblent beaucoup : demande-toi ce que fait la pénalité quand plusieurs colonnes peuvent faire le même travail.

</details>
<details><summary>Indice 2</summary>

a) et b) Sans pénalité, des colonnes semblables se partagent le travail, souvent avec des coefficients de signes opposés qui se compensent. Quand la pénalité grandit, ce partage se refait : demande-toi si la pénalité porte sur chaque coefficient, ou sur leur ensemble. c) Le dernier coefficient non nul est celui de la colonne qui explique le plus à elle seule : relis le seuil $\alpha_{\max}$ de la fiche. d) Sur un chemin du Lasso, quand un coefficient s'annule, les autres se réajustent.

</details>
<details><summary>Indice 3</summary>

a) La fiche parle de la **norme** des poids. Une norme qui baisse oblige-t-elle chacune de ses coordonnées à baisser ? Pense à deux colonnes presque identiques, dont les coefficients se compensent sans pénalité : que devient leur partage quand la pénalité les force à rétrécir ensemble ? b) Un coefficient qui, sans pénalité, sert surtout à corriger ses voisins (avec un signe opposé au leur) peut changer de signe quand ceux-ci rétrécissent. Estime combien des six coefficients jouent ce rôle de correcteurs dans un polynôme qui suit une journée lisse. c) Relis le seuil $\alpha_{\max} = \max_j |\mathbf{x}_j^\top \mathbf{y}_c| / n$ de la fiche : sur des colonnes standardisées, c'est la colonne la plus corrélée à la cible qui résiste le plus longtemps. Laquelle des six colonnes ressemble le plus à la forme de la journée (regarde les courbes de 9.12) ? d) Quand un coefficient du Lasso s'annule, les autres se réajustent pour reprendre son travail : ce réajustement peut-il redonner un rôle, le temps que la pénalité grandisse encore, à un coefficient déjà nul ?

</details>

### Ex 9.20 — Early stopping d'une descente de gradient sur un polynôme de degré 12 🔨

<details><summary>Indice 1</summary>

Recopie le pseudo-code de la fiche (§9.4) en Python. Les deux pièges : la numérotation des epochs, qui commence à 1, et la copie des poids. `train_one_epoch_20` modifie `theta` sur place : un simple nom de plus ne garde rien.

</details>
<details><summary>Indice 2</summary>

Avant la boucle : `rng = np.random.default_rng(seed)`, `theta = np.zeros(X_tr.shape[1])`, `best = math.inf`, `wait = 0`, `losses = []`. Dans `for epoch in range(1, max_epochs + 1)` : une epoch, puis `loss = val_mse_20(theta, X_val, y_val)`, ajoutée à `losses`. Si `loss < best - min_delta` : `best`, `kept_epoch`, `kept_theta = theta.copy()`, et `wait = 0` ; sinon `wait += 1`, et si `wait == patience`, renvoie le quadruplet. Après la boucle, renvoie `max_epochs` comme epoch d'arrêt.

</details>
<details><summary>Indice 3</summary>

Le pseudo-code de la fiche, traduit ; les deux lignes clés sont l'amélioration stricte et la **copie** des poids :

```python
def early_stopping_20(X_tr, y_tr, X_val, y_val, patience, min_delta=0.0, max_epochs=1000, lr=0.02, seed=921):
    # before the loop: rng = np.random.default_rng(seed) (ONCE), theta = np.zeros(X_tr.shape[1]),
    # best = math.inf, wait = 0, losses = [], and kept_epoch, kept_theta in case nothing ever improves
    for epoch in range(1, max_epochs + 1):                  # epochs numbered from 1
        # one epoch (train_one_epoch_20 updates theta in place), then loss = val_mse_20(...), appended to losses
        if loss < best - min_delta:                         # a real improvement
            best, kept_epoch, kept_theta, wait = loss, epoch, theta.copy(), 0
        # otherwise: wait += 1, and when wait == patience: return epoch, kept_epoch, kept_theta, losses
    # the patience never ran out: return max_epochs, kept_epoch, kept_theta, losses
```

Sans `.copy()`, `kept_theta` ne serait qu'un autre nom du tableau `theta`, qui continue de bouger jusqu'à l'arrêt.

</details>

### Ex 9.21 — Courbes d'apprentissage sur California avec learning_curve 📦

<details><summary>Indice 1</summary>

Une boucle sur les trois modèles ; pour chacun, un appel à `learning_curve` avec les paramètres imposés par l'énoncé. Il reste à transformer ses scores en MSE moyennes : attention au signe.

</details>
<details><summary>Indice 2</summary>

`sizes, train_scores, val_scores = learning_curve(model, X_cal, y_cal, train_sizes=SIZES_21, cv=KFold(5, shuffle=True, random_state=921), scoring="neg_mean_squared_error", shuffle=True, random_state=921)`. Les scores ont une ligne par taille et une colonne par fold : `-train_scores.mean(axis=1)` donne les MSE moyennes. a) le dernier élément de chaque courbe de validation ; b) `np.argmax` d'un tableau de booléens donne la position du premier `True`.

</details>
<details><summary>Indice 3</summary>

Range les trois modèles de l'énoncé dans un dictionnaire, dans l'ordre `"linear"`, `"degree 2"`, `"degree 3"`, puis fais une boucle sur ses éléments. Les deux lignes clés, dans la boucle :

```python
    sizes, train_scores, val_scores = learning_curve(
        model, X_cal, y_cal, train_sizes=SIZES_21, cv=KFold(5, shuffle=True, random_state=921),
        scoring="neg_mean_squared_error", shuffle=True, random_state=921)
    curves_21[name] = (sizes, -train_scores.mean(axis=1), -val_scores.mean(axis=1))   # minus: back to the MSE
```

a) Le dernier élément de chaque courbe de validation, converti en `float`. b) `sizes[np.argmax(val_degree_3 < val_linear)]`, converti en `int` : `np.argmax` d'un tableau de booléens rend la position du premier `True`.

</details>

### Ex 9.22 — Ridge contre Lasso sur California : chemins de régularisation 📦

<details><summary>Indice 1</summary>

Trois étapes : standardiser les 8 features avec les statistiques des districts d'entraînement ; une compréhension de liste par chemin, une ligne de `coef_` par valeur de `alpha` ; puis les trois questions, qui se lisent sur le chemin du Lasso ou se calculent avec la formule de la fiche.

</details>
<details><summary>Indice 2</summary>

`Z = (X - X.mean(axis=0)) / X.std(axis=0)` ; `np.array([Ridge(alpha=a).fit(Z_train_22, y_train_22).coef_ for a in RIDGE_ALPHAS_22])`, et de même pour `Lasso`. a) Les colonnes de `Z_train_22` sont déjà centrées : il suffit de centrer `y`, puis de prendre `np.max(np.abs(Z_train_22.T @ y_centred)) / n`. b) `np.sum(coef_ != 0)` pour `Lasso(alpha=0.05)`. c) La dernière feature à garder un poids est celle qui atteint le maximum de a) : `np.argmax`, puis `FEATURES_CAL[...]`.

</details>
<details><summary>Indice 3</summary>

Le z-score avec les statistiques des districts d'entraînement, puis un chemin par compréhension de liste :

```python
X_train_22 = X_cal[TRAIN_CAL]
Z_train_22 = (X_train_22 - X_train_22.mean(axis=0)) / X_train_22.std(axis=0)
ridge_path_22 = np.array([Ridge(alpha=alpha).fit(Z_train_22, y_train_22).coef_ for alpha in RIDGE_ALPHAS_22])
# lasso_path_22: the same line with Lasso and LASSO_ALPHAS_22
```

a) Les colonnes de `Z_train_22` sont déjà centrées : centre `y_train_22`, prends le maximum de `np.abs(Z_train_22.T @ y_centred)`, et n'oublie pas de le **diviser par $n$**, le nombre de districts d'entraînement. b) `np.sum(coef != 0)`, avec les poids de `Lasso(alpha=0.05)`. c) La dernière feature à garder un poids est celle qui atteint le maximum de a) : sa position par `np.argmax`, puis son nom dans `FEATURES_CAL`.

</details>

### Ex 9.23 — Lasso par descente de coordonnées et soft_threshold 🔨

<details><summary>Indice 1</summary>

`soft_threshold` tient en une ligne de NumPy. Pour `Lasso.fit`, suis le pseudo-code de l'encadré 🧮 de la fiche (§9.5) ligne à ligne : une boucle sur les passes, une boucle sur les features, et le résidu `r = y - X w` mis à jour à chaque changement d'un poids.

</details>
<details><summary>Indice 2</summary>

`np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)`, après avoir refusé `gamma < 0`. Dans `fit` : `z = np.sum(Xc ** 2, axis=0) / n` une fois pour toutes ; pour la feature `j`, `rho = Xc[:, j] @ residual / n + z[j] * w[j]` (le résidu calculé sans la feature `j`) ; `new = soft_threshold(rho, alpha) / z[j]` ; `residual -= Xc[:, j] * (new - w[j])`. Saute les colonnes où `z[j] == 0`. Retiens le plus grand `abs(new - old)` de la passe, et arrête-toi dès qu'il est `< tol`.

</details>
<details><summary>Indice 3</summary>

`soft_threshold` : refuse `gamma < 0`, puis applique la ligne de l'indice 2 à `np.asarray(z, dtype=float)`. Pour `Lasso.fit`, le squelette du pseudo-code de la fiche et ses lignes clés :

```python
    def fit(self, X, y):                        # in the class Lasso; predict and score as in Ridge
        # 1. alpha <= 0: ValueError; X, y = _check_X_y(X, y); n, p = X.shape; centre X and y (if fit_intercept)
        z = np.sum(Xc ** 2, axis=0) / n                     # z_j = x_jᵀ x_j / n, once for all
        # 2. w = np.zeros(p), residual = yc.copy(); then at most max_iter passes, each one:
        for j in range(p):                                  # skip the columns where z[j] == 0
            rho = Xc[:, j] @ residual / n + z[j] * w[j]     # the residual computed WITHOUT feature j
            new = float(soft_threshold(rho, self.alpha)) / z[j]
            residual -= Xc[:, j] * (new - w[j])             # then w[j] = new
        # 3. stop after the first pass whose largest |new - old| is < tol; n_iter_ = number of passes done
        # 4. coef_ = w; intercept_ = float(y_mean - x_mean @ w), 0.0 without an intercept; return self
```

Le piège : seuiller $\rho_j$ avec $\alpha$, **puis** diviser par $z_j$ (seuiller $\rho_j / z_j$ avec $\alpha$ ne donne le même résultat que sur des colonnes standardisées).

</details>

### Ex 9.24 — Biais et variance mesurés : 50 sous-échantillons de 30 points 🔨

<details><summary>Indice 1</summary>

Deux fonctions indépendantes. `bias_variance_decomposition` applique les deux formules de l'encadré 🧮 de la fiche (§9.6) à un tableau dont chaque ligne est un modèle. `family_24` est une boucle : tirer des jours, entraîner, prédire sur toute l'année, ranger la ligne.

</details>
<details><summary>Indice 2</summary>

Le modèle moyen : `predictions.mean(axis=0)` (la moyenne des lignes). Le biais² : `np.mean((average_model - f_true) ** 2)` ; la variance : `np.mean(predictions.var(axis=0))` (`var` divise par le nombre de modèles par défaut, `ddof=0`). Contrôle les formes avant (2 dimensions, autant de colonnes que `f_true`, au moins 2 lignes). Dans `family_24`, le générateur se crée **une** fois, avant la boucle, et `days = rng.choice(len(x_year), size=n_points, replace=False)` à chaque tour.

</details>
<details><summary>Indice 3</summary>

`bias_variance_decomposition`, dans `mylearn/linear.py` : convertis les deux arguments en tableaux de `float`, contrôle les formes (deux dimensions pour `predictions`, une pour `f_true`, autant de colonnes que de points, au moins 2 lignes), puis les deux lignes clés :

```python
    average_model = predictions.mean(axis=0)               # the mean of the ROWS: one value per point
    return float(np.mean((average_model - f_true) ** 2)), float(np.mean(predictions.var(axis=0)))   # ddof=0
```

`family_24`, dans le notebook :

```python
def family_24(alpha, n_sets=50, n_points=30, seed=924):
    rng = np.random.default_rng(seed)                       # ONCE, before the loop
    predictions = np.empty((n_sets, len(x_year)))
    # for each s: days = rng.choice(len(x_year), size=n_points, replace=False), then
    #             PolyRidge(DEGREE_24, alpha) fitted on (x_year[days], wind_year[days]) predicts row s on x_year
    return predictions
```

</details>

### Ex 9.25 — Reproduire les figures 9.13 et 9.15, puis la courbe en U 🎨

<details><summary>Indice 1</summary>

Deux fonctions de dessin qui reçoivent un `ax` déjà créé : elles ne créent ni figure ni sous-graphique, et n'appellent pas `plt.show()`. La vérification compte les courbes de chaque panneau : une courbe par modèle, plus le modèle moyen et la courbe idéale.

</details>
<details><summary>Indice 2</summary>

`draw_family_25` : `days = np.arange(predictions.shape[1])`, une boucle `for curve in predictions: ax.plot(days, curve, lw=0.7, alpha=0.35)`, puis `ax.plot` du modèle moyen (`predictions.mean(axis=0)`) et de `f_true`, en plus épais ; `ax.set(...)`, et `ax.set_ylim(ylim)` seulement si `ylim` n'est pas `None`. `draw_u_25` : trois `ax.plot(alphas, ...)`, `ax.axhline(noise_var, ...)`, `ax.set(xscale="log", yscale="log")` et `ax.legend()`.

</details>
<details><summary>Indice 3</summary>

Les deux fonctions ne dessinent que sur l'`ax` reçu : ni `plt.figure`, ni `plt.subplots`, ni `plt.show()`. Les lignes clés de `draw_family_25` :

```python
    days = np.arange(np.shape(predictions)[1])
    for curve in predictions:                               # one row = one model
        ax.plot(days, curve, color="tab:blue", lw=0.7, alpha=0.35)
    # then, thicker: the mean model (np.mean(predictions, axis=0)) and f_true, with labels;
    # ax.set(xlabel=..., ylabel=..., title=title), ax.set_ylim(ylim) only if ylim is not None, ax.legend()
```

`draw_u_25` : trois `ax.plot(alphas, ...)`, pour le biais², la variance et leur somme plus `noise_var` (convertis d'abord les listes en tableaux NumPy, sinon `+` les met bout à bout), puis `ax.axhline(noise_var, ...)`, `ax.set(xscale="log", yscale="log", ...)` et `ax.legend()`.

</details>

### Ex 9.26 — Le posterior des droites sur une grille pente-ordonnée 🔨

<details><summary>Indice 1</summary>

Travaille sur toute la grille d'un coup : deux tableaux de forme `(n_b, n_s)` donnent la pente et l'ordonnée de chaque case. Additionne des **logs** (le prior, puis une vraisemblance par point), et ne passe à l'exponentielle qu'à la fin, après avoir retranché le maximum.

</details>
<details><summary>Indice 2</summary>

`S, B = np.meshgrid(slopes, intercepts)` (une ligne par ordonnée, une colonne par pente). `log_post = -(S ** 2 + B ** 2) / (2 * prior_std ** 2)` ; pour chaque point, `log_post -= (yi - S * xi - B) ** 2 / (2 * noise_std ** 2)`. Puis `post = np.exp(log_post - log_post.max())` et `post / post.sum()`. Sans point, la boucle ne fait rien : il reste le prior. Contrôle les longueurs et les écarts-types d'abord.

</details>
<details><summary>Indice 3</summary>

Le squelette, avec ses lignes clés :

```python
def bayes_line_posterior(x, y, slopes, intercepts, noise_std=0.1, prior_std=1.0):
    # 1. x and y as 1-D float arrays; ValueError if len(x) != len(y) or if a standard deviation is <= 0
    S, B = np.meshgrid(slopes, intercepts)                 # (n_b, n_s): one row per intercept, one column per slope
    log_post = -(S ** 2 + B ** 2) / (2 * prior_std ** 2)
    # 2. for each point (xi, yi): subtract from log_post its squared residual divided by 2 noise_std²
    post = np.exp(log_post - log_post.max())               # the largest term becomes 1: no underflow
    # 3. return post normalised to sum 1
```

Sans aucun point, la boucle ne fait rien, et il reste le prior normalisé. Attention à l'ordre des arguments de `np.meshgrid` : `(slopes, intercepts)` ; dans l'autre ordre, la table est transposée.

</details>

### Ex 9.27 — Reproduire la figure 9.20 : prior, vraisemblances, posteriors et droites tirées 🎨

<details><summary>Indice 1</summary>

Pour tirer des droites, aplatis la table : chaque case devient un numéro, tiré avec sa probabilité ; retrouve ensuite sa ligne (l'ordonnée) et sa colonne (la pente). Pour la figure, une boucle sur $k = 0, 1, \dots, n$ : à la ligne $k$, le posterior des $k$ premiers points (ta `bayes_line_posterior`, qui donne le prior quand $k = 0$).

</details>
<details><summary>Indice 2</summary>

`cells = rng.choice(posterior.size, size=n_lines, p=posterior.ravel())`, puis `rows, cols = np.unravel_index(cells, posterior.shape)` et `np.column_stack([slopes[cols], intercepts[rows]])`. La figure : `fig, axes = plt.subplots(len(x) + 1, 4)` ; à la ligne `k`, `posterior = mylearn.linear.bayes_line_posterior(x[:k], y[:k], ...)`. La vraisemblance du `k`-ième point seul : `np.exp(-(y[k-1] - S * x[k-1] - B) ** 2 / (2 * noise_std ** 2))`, sur `S, B = np.meshgrid(slopes, intercepts)`. Les images : `ax.imshow(table, origin="lower", extent=[...])` et `ax.grid(False)`. Une droite tirée se dessine par ses deux bouts, en $x = -1$ et $x = 1$.

</details>
<details><summary>Indice 3</summary>

`sample_lines_27` : aplatis la table, tire des numéros de cases avec leurs probabilités, puis retrouve la ligne et la colonne de chacune.

```python
    probs = np.asarray(posterior, dtype=float)
    cells = rng.choice(probs.size, size=n_lines, p=probs.ravel() / probs.sum())
    rows, cols = np.unravel_index(cells, probs.shape)        # row = intercept, column = slope
    # return the (n_lines, 2) array whose columns are the slopes of cols and the intercepts of rows
```

`bayes_figure_27`, le squelette :

```python
    fig, axes = plt.subplots(len(x) + 1, 4, figsize=(13, 3.0 * (len(x) + 1)))
    for k, (ax_data, ax_like, ax_post, ax_lines) in enumerate(axes):
        posterior = mylearn.linear.bayes_line_posterior(x[:k], y[:k], slopes, intercepts, noise_std, prior_std)
        # column 1: the k first points, the k-th in red
        # column 2: nothing when k == 0 (ax_like.axis("off")); otherwise the likelihood of the k-th point ALONE,
        #           np.exp(-(y[k-1] - S * x[k-1] - B) ** 2 / (2 * noise_std ** 2)),
        #           with S, B = np.meshgrid(slopes, intercepts), drawn with imshow
        # column 3: imshow(posterior, origin="lower", extent=[...]), then grid(False)
        # column 4: the n_lines lines of sample_lines_27, each drawn through its two ends, x = -1 and x = 1
    return fig
```

</details>

### Ex 9.28 — Régularisation piégée : quatre erreurs qui faussent Ridge 🐛

<details><summary>Indice 1</summary>

Pour chaque étape, pose trois questions : sur quelles lignes calcule-t-elle quelque chose (toutes, ou l'entraînement seulement) ? Que met-elle à des échelles différentes, alors que Ridge pénalise tous les poids de la même façon ? Qu'est-ce qui finit par être pénalisé, et qu'est-ce qui choisit `alpha` ?

</details>
<details><summary>Indice 2</summary>

Relis les pièges de la régularisation dans la fiche (§9.5) : des features standardisées **avec l'entraînement** ; une ordonnée à l'origine non pénalisée ; une pénalité choisie par validation, jamais sur le test (ch. 8). Pour `honest_28`, écris une petite fonction interne `fit_predict(rows_fit, rows_eval, alpha)` qui fait les étapes 2 et 3 du protocole sur les lignes reçues ; elle sert dans chaque tour de la validation croisée, puis pour le modèle final.

</details>
<details><summary>Indice 3</summary>

a) Passe chaque étape au crible des trois questions de l'indice 1. Exemple, B : le z-score est calculé avec `X.mean(axis=0)` et `X.std(axis=0)` sur **toutes** les lignes, test compris, avant le découpage. C'est une fuite : B fait partie de ta réponse. Fais de même pour A, C, D, E et F, sans oublier qu'une étape saine en elle-même peut arriver trop tard : c'est alors une étape antérieure qui fausse l'étude.

Pour `honest_28`, la fonction interne fait les étapes 2 et 3 du protocole sur les lignes qu'on lui donne :

```python
    poly = PolynomialFeatures(3, include_bias=False)

    def fit_predict(rows_fit, rows_eval, alpha):
        P_fit, P_eval = poly.fit_transform(X_cal[rows_fit]), poly.fit_transform(X_cal[rows_eval])
        mean, std = P_fit.mean(axis=0), P_fit.std(axis=0)          # statistics of the fitted rows only
        # your Ridge(alpha=alpha) fitted on (P_fit - mean) / std and y_cal[rows_fit];
        # return its predictions on (P_eval - mean) / std
```

Puis : pour chaque `alpha`, la MSE de validation moyenne sur les 5 folds (ils donnent des positions dans `TRAIN_28` : `fit_predict(TRAIN_28[tr], TRAIN_28[val], alpha)`) ; le meilleur `alpha` ; et une seule évaluation sur le test, la RMSE de `fit_predict(TRAIN_28, TEST_28, best)`.

</details>

### Ex 9.29 — Refactoriser l'expérience biais-variance en fonction testée 🛠️

<details><summary>Indice 1</summary>

Compare les trois blocs de `script_29` ligne à ligne : seule la valeur de `alpha` devrait changer. Tout ce qui varie d'autre est une incohérence. Ta fonction, elle, reçoit en paramètres tout ce que le script lisait dans le notebook (les données, la courbe idéale, le modèle), si bien qu'un test peut lui donner des données minuscules et un `fit_predict` trivial.

</details>
<details><summary>Indice 2</summary>

La fonction : pour chaque `p`, un nouveau `rng = np.random.default_rng(seed)`, un tableau `(n_sets, len(x))` de prédictions, puis les deux formules de 9.24. Des propriétés à tester, chacune avec un `fit_predict` défini dans le test : un nombre par valeur de `params` ; des modèles constants n'ont aucune variance, et leur biais² se calcule à la main ; biais² + variance = erreur quadratique moyenne des modèles (un `fit_predict` qui range ses prédictions dans une liste permet de la recalculer) ; des sous-échantillons sans doublon ; même graine, même résultat ; toutes les valeurs de `params` voient les mêmes sous-échantillons.

</details>
<details><summary>Indice 3</summary>

La fonction, en squelette :

```python
def bias_variance_study(x, y, f_true, fit_predict, params, n_sets=50, n_points=30, seed=0):
    """One summary line; then how the subsamples are drawn (seed, without replacement), what each parameter is
    (fit_predict(x_train, y_train, x_eval, p) included), and what the function returns."""
    bias2, variance = [], []
    for p in params:
        rng = np.random.default_rng(seed)                   # a NEW generator for each p: the same subsamples for all
        # an array (n_sets, len(x)); row s = fit_predict(x[idx], y[idx], x, p),
        # with idx = rng.choice(len(x), size=n_points, replace=False)
        # append the bias² and the variance of this family (the formulas of 9.24, ddof=0)
    return np.array(bias2), np.array(variance)
```

Un test, en modèle : un `fit_predict` qui ignore ses données et rend toujours 2.

```python
def test_constant_models_have_no_variance():
    x, f = np.arange(10.0), np.linspace(-1, 1, 10)
    constant = lambda x_train, y_train, x_eval, p: np.full(len(x_eval), 2.0)
    bias2, variance = bias_variance_study(x, x, f, constant, [0], n_sets=4, n_points=3)
    # assert: no variance at all (pytest.approx), and a bias² you can compute by hand from f
```

Écris de même un test par propriété de l'indice 2, puis `TESTS_29 = [test_..., ...]` : les fonctions elles-mêmes, ni leurs noms entre guillemets, ni leurs résultats.

</details>

### Ex 9.30 — Double descente avec des features aléatoires 🔬

<details><summary>Indice 1</summary>

Les deux premières fonctions tiennent en une ligne chacune (NumPy fait tout). La troisième est une double boucle : sur les tirages, puis sur les valeurs de $p$. Tire `V` et `c` **une** fois par tirage, avec `max(p_grid)` colonnes : chaque $p$ prend les $p$ premières.

</details>
<details><summary>Indice 2</summary>

`np.maximum(0.0, X @ V + c)` (`c` se diffuse sur les lignes) ; `np.linalg.pinv(F) @ y`. Dans `double_descent_30` : `rng = np.random.default_rng(seed)`, puis, à chaque tirage, `V = rng.normal(0, 1 / np.sqrt(5), (5, p_max))` **puis** `c = rng.normal(0, 1, p_max)` (dans cet ordre), les features des 40 points d'entraînement et des 2 000 points de test, et pour chaque `p` : `w = min_norm_fit_30(F_tr[:, :p], y_tr_30)`, les deux MSE avec `F[:, :p] @ w`.

</details>
<details><summary>Indice 3</summary>

Les deux premières fonctions sont les deux lignes de l'indice 2. Pour `double_descent_30`, le squelette :

```python
def double_descent_30(p_grid, n_draws=20, seed=9300):
    rng = np.random.default_rng(seed)
    p_max = max(p_grid)
    # two arrays (n_draws, len(p_grid)), for the test MSE and the training MSE
    for d in range(n_draws):
        V = rng.normal(0, 1 / np.sqrt(5), (5, p_max))      # V first, then c: the order of the draws matters
        c = rng.normal(0, 1, p_max)
        F_tr, F_te = relu_features_30(X_tr_30, V, c), relu_features_30(X_te_30, V, c)   # the SAME V and c
        # for each p of p_grid: w = min_norm_fit_30(F_tr[:, :p], y_tr_30), then the two MSE with F[:, :p] @ w
    return test_mse, train_mse
```

`V` et `c` se tirent **une fois par tirage**, pas pour chaque $p$ : chaque $p$ prend les $p$ premières colonnes des mêmes features aléatoires, et l'on n'apprend que les poids `w`.

</details>

### Ex 9.31 — Défi California : le meilleur modèle linéaire régularisé 🏆

<details><summary>Indice 1</summary>

Le point de départ est un modèle linéaire sur les 8 mesures brutes : il sous-apprend (9.21). Donne-lui plus de capacité avec des features polynomiales, puis tiens-la en laisse avec une pénalité choisie par validation croisée **à l'intérieur** de `build_31`, sur les seuls districts qu'elle reçoit.

</details>
<details><summary>Indice 2</summary>

Une petite classe avec `fit` et `predict` : les features polynomiales des 8 mesures (compare les degrés 2 et 3), le z-score de chaque colonne avec les statistiques des lignes reçues par `fit`, puis ta `Ridge`. Dans `build_31` : une validation croisée à 5 folds sur `X_train` seul, la MSE de validation moyenne pour quelques valeurs de `alpha` (de 0,01 à 100), puis le meilleur modèle réentraîné sur tout `X_train`. Pour le palier 🌟 : la valeur d'un logement dépend de la distance aux grandes villes.

</details>
<details><summary>Indice 3</summary>

Une petite classe qui calcule tout avec les seules lignes reçues par `fit`, puis le choix d'`alpha` **dans** `build_31` :

```python
class PolyRidge31:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        P = PolynomialFeatures(3, include_bias=False).fit_transform(X)
        self.mean_, self.std_ = P.mean(axis=0), P.std(axis=0)        # statistics of the rows received
        # self.ridge_: your Ridge(alpha=self.alpha) fitted on (P - self.mean_) / self.std_ and y; return self

    # predict(X): the same features, z-scored with self.mean_ and self.std_, then self.ridge_.predict


def build_31(X_train, y_train):
    # folds = list(KFold(5, shuffle=True, random_state=0).split(X_train))
    # for each alpha of a short grid (0.01 to 100): the mean validation MSE of PolyRidge31(alpha) over the folds
    # return PolyRidge31(best alpha).fit(X_train, y_train)
```

Pour 🌟, ajoute avant le degré 3 deux colonnes calculées ligne par ligne : la distance (en degrés de latitude et de longitude) de chaque district à Los Angeles (34,05 ; −118,24) et à San Francisco (37,77 ; −122,42).

</details>
