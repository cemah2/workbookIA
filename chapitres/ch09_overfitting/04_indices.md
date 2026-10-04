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

Un grand écart signale un modèle qui colle à ses exemples ; deux erreurs proches mais très au-dessus de la référence, un modèle trop rigide ; deux erreurs basses et proches, le bon réglage. Pour d), cherche parmi les trois modèles un contre-exemple à la phrase.

</details>

### 9.Q2 — Walter et sa moustache : qu'est-ce qui a été mal appris ?

<details><summary>Indice 1</summary>

Sépare deux moments : pendant le mariage, avec les invités qu'on venait de rencontrer, et plus tard, avec d'autres personnes. À quoi correspond chacun en machine learning ?

</details>
<details><summary>Indice 2</summary>

Distingue ce qu'on a retenu (des prénoms) et la façon dont on les a reliés à l'apparence. Le §9.5 du livre revient sur ce que l'on aurait pu remarquer d'autre chez Walter.

</details>
<details><summary>Indice 3</summary>

Une règle qui marche parfaitement sur les exemples connus et échoue sur les nouveaux : c'est la définition de l'overfitting, avec une erreur d'entraînement faible. Son équivalent en machine learning est une feature qui sépare les exemples d'entraînement par coïncidence (pense au décor des photos du ch. 8).

</details>

### 9.Q3 — Underfitting : les vrais remèdes

<details><summary>Indice 1</summary>

Les deux $R^2$ sont presque égaux et bas, alors qu'un autre modèle fait beaucoup mieux sur la même validation. Que dit cette comparaison ?

</details>
<details><summary>Indice 2</summary>

Pour chaque remède, demande-toi s'il rend le modèle plus **souple** (plus de capacité) ou s'il réduit sa **variance**. Le problème de ce modèle-là est-il la variance ?

</details>
<details><summary>Indice 3</summary>

Contre l'underfitting, il faut de la capacité : de meilleures features, moins de pénalité, un modèle plus riche. Plus d'exemples du même genre referme un écart entre entraînement et validation… qui est déjà presque nul ici (regarde le panneau (a) de la figure des courbes d'apprentissage de la fiche). L'encadré ⚠️ de la fiche §9.2.2 répond à d).

</details>

### 9.Q4 — Courbes d'erreur : où commence l'overfitting ?

<details><summary>Indice 1</summary>

Lis séparément les deux lignes du tableau : comment chacune évolue-t-elle au fil des epochs ?

</details>
<details><summary>Indice 2</summary>

Repère le minimum de la ligne « validation ». Après lui, que fait la ligne « entraînement » ? Pour d), relis la fiche §9.2 : quelles sont les trois erreurs, et laquelle ne connaît-on jamais exactement ?

</details>
<details><summary>Indice 3</summary>

Une erreur d'entraînement qui baisse pendant que la validation remonte : c'est le signe de la figure 9.1 du livre. L'erreur de validation n'est qu'une mesure sur un échantillon de données nouvelles : l'encadré ⚠️ de la fiche §9.3 dit ce qu'elle est.

</details>

### 9.Q5 — Un point isolé : frontière tordue ou frontière simple ?

<details><summary>Indice 1</summary>

Relis la figure 9.6 du livre et le paragraphe « Le point isolé » de la fiche §9.3.

</details>
<details><summary>Indice 2</summary>

Pour b), compte les points d'entraînement mal classés par chaque frontière. Pour c), regarde de quel côté de la frontière simple tombe le voisinage du point isolé.

</details>
<details><summary>Indice 3</summary>

Le détour gagne un point d'entraînement, mais la région autour du point isolé est pleine de carrés. Pour d), un point aberrant n'est pas forcément une erreur : demande-toi d'où il vient avant de décider (⚖️ 9.10).

</details>

### 9.Q6 — Early stopping : quand s'arrêter, et pourquoi c'est délicat

<details><summary>Indice 1</summary>

Relis la fiche §9.4, et son pseudo-code.

</details>
<details><summary>Indice 2</summary>

Quelle erreur estime la généralisation sans toucher au test ? Pour c), suis l'attente epoch par epoch après la meilleure, jusqu'à ce qu'elle atteigne la patience.

</details>
<details><summary>Indice 3</summary>

On décide sur la validation, on attend une hausse durable (la patience absorbe le bruit), et la dernière ligne du pseudo-code dit quels poids on recharge. Méfie-toi de la valeur par défaut de Keras (encadré 🕰️ de la §9.4).

</details>

### 9.Q7 — Régularisation : ce que change λ

<details><summary>Indice 1</summary>

Écris la loss régularisée : loss sur les données + $\lambda$ × pénalité. Que se passe-t-il quand $\lambda$ pèse plus lourd ?

</details>
<details><summary>Indice 2</summary>

Pour b), la loss sur les données est la plus basse possible quand $\lambda = 0$ ; un $\lambda$ plus grand l'éloigne de ce minimum. Pour d), relis l'encadré 💼 de la fiche §9.5.

</details>
<details><summary>Indice 3</summary>

Plus $\lambda$ est grand, plus la norme des poids baisse et moins la courbe colle aux données d'entraînement. Un hyperparamètre ne s'apprend pas sur la loss d'entraînement (que vaudrait alors $\lambda$ ?). Pour e), avec des poids écrasés à zéro, que reste-t-il de la prédiction, sachant que l'ordonnée à l'origine n'est pas pénalisée ?

</details>

### 9.Q8 — Pénalité sur les poids, dropout, batchnorm : même objectif ?

<details><summary>Indice 1</summary>

Relis la fin de la fiche §9.5 : la régularisation, le dropout, la batchnorm et l'encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

Pour a), demande-toi, pour chaque technique, si son but premier est de mieux généraliser ou d'aller plus vite. Pour d), compare les deux panneaux de la figure `l1_l2.png`.

</details>
<details><summary>Indice 3</summary>

L'early stopping et l'augmentation de données sont des régularisations, même sans pénalité. La batchnorm a été proposée pour l'entraînement lui-même. Le Lasso met des poids exactement à zéro : répartit-il l'importance ?

</details>

### 9.Q9 — Biais et variance : des propriétés d'une famille de courbes

<details><summary>Indice 1</summary>

Relis les deux encadrés 🧮 du début de la fiche §9.6.

</details>
<details><summary>Indice 2</summary>

Le biais compare le modèle **moyen** à une référence ; la variance compare les modèles **entre eux**. Quelle référence n'a pas de bruit ?

</details>
<details><summary>Indice 3</summary>

Avec une seule courbe, il n'y a ni moyenne ni dispersion. Dans la décomposition biais² + variance + bruit, que reste-t-il quand les deux premiers termes sont nuls ?

</details>

### 9.Q10 — Courbes raides ou souples : qui a quel biais, quelle variance ?

<details><summary>Indice 1</summary>

Regarde la figure `biais_variance.png` de la fiche, panneaux (a) et (b), et les figures 9.13 et 9.15 du livre.

</details>
<details><summary>Indice 2</summary>

Une droite peut-elle suivre une courbe ondulée ? Un polynôme de degré 15 sur 30 points bruités dépend-il beaucoup des points tirés ?

</details>
<details><summary>Indice 3</summary>

Les familles rigides se trompent toutes de la même façon, les familles souples se trompent chacune à sa façon. Aux bords, peu de points retiennent un polynôme. Pour d), relis le panneau (b) des courbes d'apprentissage de la fiche §9.3.

</details>

### 9.Q11 — Droites a posteriori : a-t-on le droit de parler de variance ?

<details><summary>Indice 1</summary>

Combien de datasets y a-t-il dans l'approche bayésienne du §9.7 ? Et dans l'expérience du §9.6 ?

</details>
<details><summary>Indice 2</summary>

Les droites tirées dans le posterior sont des **hypothèses** pondérées par leur probabilité, pas des modèles entraînés sur des jeux différents. Que mesure leur dispersion ?

</details>
<details><summary>Indice 3</summary>

C'est l'incertitude qui reste sur la droite (ch. 4 : l'intervalle de crédibilité). Relis le dernier paragraphe de la fiche §9.7 avant l'encadré 🕰️, et demande-toi ce que devient cette incertitude avec plus de points.

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

Le minimum de 25 mesures bruitées est en moyenne plus bas que la vraie erreur. Le choix s'est adapté au jeu de validation comme un modèle s'adapte à ses exemples. La solution : un jeu jamais consulté pendant le choix.

</details>

### 9.R2 — Ch. 6 : ce que mesure une cross-entropy utilisée comme loss

<details><summary>Indice 1</summary>

La cross-entropy d'un exemple est la surprise de la bonne classe : $-\ln p$ (ch. 6).

</details>
<details><summary>Indice 2</summary>

Calcule $-\ln p$ pour chacune des quatre probabilités, puis fais la moyenne. Pour passer des nats aux bits, souviens-toi que $\log_2 x = \ln x / \ln 2$.

</details>
<details><summary>Indice 3</summary>

La plus petite probabilité donne la plus grande surprise. Pour d), un exemple bien classé avec une probabilité de 0,9 a-t-il une loss nulle ? Que fait la descente de gradient tant que la loss n'est pas nulle ?

</details>

### 9.R3 — Ch. 2 : biais et variance d'un estimateur, et le bootstrap

<details><summary>Indice 1</summary>

Calcule d'abord la moyenne, puis la somme des carrés des écarts à la moyenne.

</details>
<details><summary>Indice 2</summary>

Le biais d'un estimateur est $\mathbb{E}[\hat{\theta}] - \theta$ : remplace $\mathbb{E}[\hat{\theta}]$ par l'expression donnée et $\theta$ par $\sigma^2$.

</details>
<details><summary>Indice 3</summary>

$\frac{n-1}{n}\sigma^2 - \sigma^2 = -\frac{1}{n}\sigma^2$. Pour d), relis « Ce que le bootstrap ne fait pas » dans la fiche du ch. 2 (§2.6) : rééchantillonner un échantillon biaisé crée-t-il les personnes qu'il ne contient pas ?

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

Pour f) et g), un seul résidu change (le 5ᵉ, qui passe de 1 à 11) : corrige la somme des carrés et la somme des valeurs absolues, puis divise par 5. Compare les deux facteurs pour h).

</details>

### Ex 9.2 — Moindres carrés : la meilleure droite par dérivées partielles ∂

<details><summary>Indice 1</summary>

$L$ est une somme de carrés : dérive chaque terme $(y_i - a x_i - b)^2$ par rapport à $b$, puis à $a$, avec la règle de la chaîne (0B).

</details>
<details><summary>Indice 2</summary>

La condition sur $b$ donne $\sum_i y_i = a\sum_i x_i + n\,b$. Dans la condition sur $a$, remplace $b$ et utilise le fait que $\sum_i (y_i - \bar{y}) = 0$ et $\sum_i (x_i - \bar{x}) = 0$ pour faire apparaître des écarts à la moyenne.

</details>
<details><summary>Indice 3</summary>

Pour l'application : calcule $\bar{x}$ et $\bar{y}$, puis les deux sommes $\sum (x_i - \bar{x})(y_i - \bar{y})$ et $\sum (x_i - \bar{x})^2$, sans arrondir. Pour f), $\det\begin{pmatrix} p & q \\ q & r \end{pmatrix} = pr - q^2$, et pour 4, compare-le à $n \sum_i (x_i - \bar{x})^2$.

</details>

### Ex 9.3 — Ridge en dimension 1 : w* = Σxy / (Σx² + λ) ∂

<details><summary>Indice 1</summary>

$L(w)$ est une fonction d'une seule variable : annule sa dérivée.

</details>
<details><summary>Indice 2</summary>

$L'(w) = -2\sum_i x_i (y_i - w x_i) + 2\lambda w$ ; regroupe les termes en $w$. Pour les applications, calcule une fois pour toutes $\sum x_i y_i$ et $\sum x_i^2$.

</details>
<details><summary>Indice 3</summary>

Pour d), écris l'équation « $w^*(\lambda) = w^*(0)/2$ » et résous-la en $\lambda$. Pour e), calcule les cinq résidus $y_i - w^* x_i$ avec le $w^*$ de cette valeur de $\lambda$, puis la somme de leurs carrés, sans ajouter la pénalité. Pour f), regarde la somme des carrés comme une parabole en $w$.

</details>

### Ex 9.4 — Biais² et variance à partir d'un tableau de prédictions ✏️

<details><summary>Indice 1</summary>

Le modèle moyen est la moyenne de chaque **colonne** du tableau (un point à la fois, sur les quatre modèles).

</details>
<details><summary>Indice 2</summary>

Biais² : en chaque point, (modèle moyen − $f$)², puis moyenne sur les trois points. Variance : en chaque point, la moyenne des carrés des écarts des quatre modèles au modèle moyen (divise par 4), puis moyenne sur les trois points.

</details>
<details><summary>Indice 3</summary>

Pour d), calcule les douze erreurs au carré face à $f$ et divise leur somme par 12 ; tu dois retrouver biais² + variance. Pour e), ajoute le bruit (encadré de la décomposition). Pour f), l'erreur du modèle moyen face à $f$ est un des termes que tu as déjà calculés.

</details>

### Ex 9.5 — Early stopping avec patience sur une courbe de loss ✏️

<details><summary>Indice 1</summary>

Fais un tableau à trois lignes sous celui de l'énoncé : la meilleure loss jusqu'ici, l'attente, l'epoch des poids gardés.

</details>
<details><summary>Indice 2</summary>

À chaque epoch : si la loss est strictement sous (meilleure − `min_delta`), elle devient la meilleure et l'attente revient à 0 ; sinon, l'attente augmente de 1, et l'on s'arrête quand elle atteint la patience.

</details>
<details><summary>Indice 3</summary>

Attention à l'epoch 7, qui bat de peu l'epoch 5 : avec une patience trop courte, on ne l'atteint jamais. Avec `min_delta = 0,02`, applique la comparaison **stricte** à chaque epoch, surtout quand la loss baisse d'exactement 0,02 : une baisse égale à `min_delta` ne compte pas.

</details>

### Ex 9.6 — Lasso en dimension 1 : le seuillage doux et les zéros exacts ∂

<details><summary>Indice 1</summary>

Sur $w > 0$, $|w| = w$ ; sur $w < 0$, $|w| = -w$ : sur chaque demi-droite, $g$ est une parabole dérivable.

</details>
<details><summary>Indice 2</summary>

Pour 2, minore $-wz$ par $-|w|\,|z|$. Pour 3, développe la somme des carrés et fais apparaître $\frac{z}{2}\,(w - \rho/z)^2$ plus une constante (mise sous forme canonique).

</details>
<details><summary>Indice 3</summary>

$S(z, \gamma)$ : zéro si $|z| \leq \gamma$, sinon $z$ rapproché de 0 de $\gamma$. Avec les données de 9.3 : $\rho = \frac{1}{n}\sum x_i y_i$ et $z = \frac{1}{n}\sum x_i^2$, puis $w^* = S(\rho, \alpha)/z$. Le poids s'annule dès que $\alpha$ atteint $|\rho|$.

</details>

### Ex 9.7 — Mise à jour bayésienne d'une droite sur une grille 3 × 3 ✏️

<details><summary>Indice 1</summary>

Fais un tableau de neuf lignes, une par droite $(a, b)$ : le poids du prior, l'écart au premier point, sa vraisemblance, le produit.

</details>
<details><summary>Indice 2</summary>

Pour $P_1 = (1, 1)$, l'écart vaut $r_1 = 1 - (a \times 1 + b)$ ; pour $P_2 = (-1, 0)$, $r_2 = 0 - (a \times (-1) + b)$. Pour b), normalise les produits de la première étape (divise par leur somme).

</details>
<details><summary>Indice 3</summary>

Pour le second point, multiplie les produits de la première étape par la vraisemblance de $P_2$, puis normalise par la nouvelle somme (normaliser aussi entre les deux étapes ne change rien : seule la normalisation finale compte). Pour f), le posterior final est proportionnel à prior × $L_1$ × $L_2$.

</details>

### Ex 9.9 — Diagnostiquer quatre paires de courbes d'entraînement et de validation 📈

<details><summary>Indice 1</summary>

Pour chaque panneau, regarde d'abord le **niveau** final des deux courbes, puis leur **écart**, puis leur **ordre** (laquelle est au-dessus).

</details>
<details><summary>Indice 2</summary>

Relis le tableau des symptômes de la fiche §9.2. Une validation qui remonte pendant que l'entraînement baisse signale une chose ; deux courbes hautes et collées, une autre.

</details>
<details><summary>Indice 3</summary>

Pour d), demande-toi ce qui rend la tâche plus difficile pendant l'entraînement que pendant la validation (des neurones éteints, des exemples déformés), et ce qui rendrait la validation plus facile que l'entraînement (des cas plus nets, des labels plus sûrs).

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

```python
models_12 = {degree: make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
             .fit(x_day1.reshape(-1, 1), tempo_day1) for degree in (1, 4, 15)}
train_mse_12 = [mean_squared_error(tempo_day1, models_12[d].predict(x_day1.reshape(-1, 1))) for d in (1, 4, 15)]
day2_mse_12 = [mean_squared_error(tempo_day2, models_12[d].predict(x_day2.reshape(-1, 1))) for d in (1, 4, 15)]
```

</details>

### Ex 9.13 — Erreurs d'entraînement et de test selon le degré : ta courbe d'abord 🔮

<details><summary>Indice 1</summary>

Relis le point 8 de « L'essentiel » : en fonction de la capacité, l'erreur sur des données nouvelles dessine un U. Et l'erreur d'entraînement : plus de capacité peut-il jamais faire moins bien sur les points d'entraînement ? Tes trois prédictions portent sur la largeur du fond du U et sur la hauteur de sa branche droite ; 9.12 t'en a déjà donné trois points (degrés 1, 4 et 15).

</details>
<details><summary>Indice 2</summary>

a) Le degré 1 ne voit qu'une tendance. Dès que le polynôme suit la forme de la journée, il fait mieux que lui, jusqu'au moment où il suit aussi les hésitations du premier jour : combien de degrés tiennent dans ce fond de vallée ? b) Où la forme est-elle suivie sans plus ? c) La constante fait, le lendemain, à peu près la variance des réglages du lendemain : pour faire pire, un polynôme doit s'écarter franchement de la journée quelque part. Où, et à partir de quel degré ?

</details>
<details><summary>Indice 3</summary>

a) 16 réglages laissent de la marge : tant que le degré reste loin de 15, le polynôme n'a pas assez de coefficients pour suivre chaque hésitation, et la forme qu'il capte l'emporte sur le bruit qu'il apprend ; compte large. b) Les petits degrés se tiennent de près ; le minimum est là où la forme de la journée est suivie, sans plus : 9.12 t'a montré un degré qui y arrive. c) Avec 16 réglages, le degré 15 passe par tous les points ; le degré juste en dessous est presque aussi contraint, et oscille fort aux bords, là où aucun réglage du premier jour ne le retient (16 h 15, par exemple).

</details>

### Ex 9.14 — mean_squared_error, mean_absolute_error et r2_score 🔨

<details><summary>Indice 1</summary>

Les trois fonctions commencent de la même façon : convertir les deux arguments en tableaux de `float`, vérifier qu'ils ont la même longueur et assez d'éléments. Écris ce contrôle une seule fois, dans une petite fonction auxiliaire dont le nom commence par `_` ; chaque mesure tient ensuite en une ou deux lignes.

</details>
<details><summary>Indice 2</summary>

`np.asarray(y, dtype=float).ravel()` ; compare les deux `len(...)` **avant** tout calcul, sinon NumPy diffuse un tableau d'un seul élément sur l'autre sans rien dire. MSE : `np.mean((y_true - y_pred) ** 2)` ; MAE : `np.mean(np.abs(y_true - y_pred))` ; $R^2$ : `1 - ss_res / ss_tot`, avec `ss_tot = np.sum((y_true - y_true.mean()) ** 2)` (la moyenne de `y_true`, pas celle de `y_pred`), et le cas `ss_tot == 0` traité à part. Renvoie `float(...)`.

</details>
<details><summary>Indice 3</summary>

```python
def _check_targets(y_true, y_pred, min_samples):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} values but y_pred has {len(y_pred)}")
    if len(y_true) < min_samples:
        raise ValueError(f"at least {min_samples} sample(s) needed, got {len(y_true)}")
    return y_true, y_pred


def mean_squared_error(y_true, y_pred):
    y_true, y_pred = _check_targets(y_true, y_pred, min_samples=1)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = _check_targets(y_true, y_pred, min_samples=1)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    y_true, y_pred = _check_targets(y_true, y_pred, min_samples=2)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:                              # constant target
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot
```

</details>

### Ex 9.15 — polynomial_features, interactions comprises 🔨

<details><summary>Indice 1</summary>

a) Relis la formule de l'encadré de la fiche : elle compte les monômes de degré 1 à $d$ en $p$ variables, sans la constante (c'est le « − 1 »). Pour la fonction : une boucle sur les degrés, puis une boucle sur les combinaisons d'indices de colonnes ; chaque combinaison donne une colonne.

</details>
<details><summary>Indice 2</summary>

a) $\binom{p + d}{d} - 1$ (`math.comb`). La fonction : `X = np.asarray(X, dtype=float)`, puis `X.reshape(-1, 1)` si `X.ndim == 1` ; `for d in range(1, degree + 1)`, puis `for combo in combinations_with_replacement(range(X.shape[1]), d)` ; la colonne vaut `np.prod(X[:, list(combo)], axis=1)`. Rassemble les colonnes avec `np.column_stack`, après une colonne de 1 si `include_bias`. Vérifie `degree` et `X.ndim` en premier.

</details>
<details><summary>Indice 3</summary>

```python
n_columns_15 = math.comb(4 + 3, 3) - 1          # in the notebook


def polynomial_features(X, degree=2, include_bias=False):     # in mylearn/linear.py
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)                    # a 1-D array is ONE feature
    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got an array with {X.ndim} dimensions")
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError(f"degree must be an integer >= 1, got {degree!r}")
    columns = [np.ones(len(X))] if include_bias else []
    for d in range(1, int(degree) + 1):
        for combo in combinations_with_replacement(range(X.shape[1]), d):
            columns.append(np.prod(X[:, list(combo)], axis=1))
    return np.column_stack(columns)
```

</details>

### Ex 9.16 — LinearRegression par moindres carrés 🔨

<details><summary>Indice 1</summary>

Trois temps dans `fit` : contrôler `X` et `y`, centrer, résoudre. L'ordonnée à l'origine ne se résout pas : elle se retrouve après coup avec les moyennes (∂ 9.2). Les contrôles resserviront dans `Ridge` et `Lasso` : mets-les dans une fonction auxiliaire.

</details>
<details><summary>Indice 2</summary>

`x_mean, y_mean = X.mean(axis=0), y.mean()` ; `w = np.linalg.lstsq(X - x_mean, y - y_mean, rcond=None)[0]` ; `b = float(y_mean - x_mean @ w)`. Sans ordonnée à l'origine, `lstsq` directement sur `X` et `y`, et `b = 0.0`. Range-les dans `self.coef_` et `self.intercept_`, rien d'autre de public, puis `return self`. `predict` : `X @ self.coef_ + self.intercept_` ; `score` : ton `r2_score(y, self.predict(X))`.

</details>
<details><summary>Indice 3</summary>

```python
def _check_X_y(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.ndim} dimension(s): use X.reshape(-1, 1) for one feature")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} values")
    return X, y


class LinearRegression:
    ...                                         # __init__ as in the stub

    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), y.mean()
            w = np.linalg.lstsq(X - x_mean, y - y_mean, rcond=None)[0]
            b = float(y_mean - x_mean @ w)
        else:
            w, b = np.linalg.lstsq(X, y, rcond=None)[0], 0.0
        self.coef_, self.intercept_ = w, b
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X, y):
        return r2_score(y, self.predict(X))
```

</details>

### Ex 9.17 — Ridge en forme fermée, intercept non pénalisé 🔨

<details><summary>Indice 1</summary>

C'est ta `LinearRegression` avec deux différences : la vérification de `alpha`, et la résolution, qui utilise le système de l'encadré 🧮 « Ridge en forme fermée » au lieu de `lstsq`. Le centrage fait le reste : sur des données centrées, l'ordonnée à l'origine sort du problème, donc de la pénalité.

</details>
<details><summary>Indice 2</summary>

Vérifie `alpha` au début de `fit` (pas dans `__init__`). Puis `Xc, yc = X - x_mean, y - y_mean` (sans ordonnée à l'origine : `Xc, yc = X, y`) ; `A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])` ; `w = np.linalg.solve(A, Xc.T @ yc)` ; `b = y_mean - x_mean @ w`. Avec `alpha = 0`, tu dois retrouver les moindres carrés.

</details>
<details><summary>Indice 3</summary>

```python
    def fit(self, X, y):                        # in the class Ridge; predict and score as in LinearRegression
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        X, y = _check_X_y(X, y)
        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), y.mean()
        else:
            x_mean, y_mean = np.zeros(X.shape[1]), 0.0
        Xc, yc = X - x_mean, y - y_mean
        w = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ yc)   # never an inverse
        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w) if self.fit_intercept else 0.0
        return self
```

</details>

### Ex 9.18 — Courbes de validation : le degré, puis λ 🔬

<details><summary>Indice 1</summary>

Deux boucles imbriquées : sur les valeurs de `values`, puis sur les folds. Chaque fold demande un modèle **neuf**, entraîné sur sa partie d'entraînement, puis deux MSE. Ce sont ces MSE que tu moyennes sur les folds, une paire de moyennes par valeur.

</details>
<details><summary>Indice 2</summary>

`for train_idx, val_idx in folds:` ; `model = make_model(value)` ; `model.fit(x[train_idx], y[train_idx])` ; puis `mse(y[train_idx], model.predict(x[train_idx]))` et `mse(y[val_idx], model.predict(x[val_idx]))` (la fonction `mse` est fournie). Ajoute à chacune des deux listes la moyenne des cinq MSE (`float(np.mean(...))`), puis renvoie le couple de listes. Ne réutilise pas un modèle d'un fold à l'autre.

</details>
<details><summary>Indice 3</summary>

```python
def validation_curve_18(make_model, values, x, y, folds):
    train_mse, val_mse = [], []
    for value in values:
        train_scores, val_scores = [], []
        for train_idx, val_idx in folds:
            model = make_model(value)                       # a new model for each fold
            model.fit(x[train_idx], y[train_idx])
            train_scores.append(mse(y[train_idx], model.predict(x[train_idx])))
            val_scores.append(mse(y[val_idx], model.predict(x[val_idx])))
        train_mse.append(float(np.mean(train_scores)))
        val_mse.append(float(np.mean(val_scores)))
    return train_mse, val_mse
```

</details>

### Ex 9.19 — Que deviennent les coefficients quand λ grandit ? 🔮

<details><summary>Indice 1</summary>

Relis ∂ 9.3 : en dimension 1, Ridge divise le poids par un facteur qui grandit avec $\lambda$. Mais ici, les six colonnes $x, x^2, \dots, x^6$ se ressemblent beaucoup : demande-toi ce que fait la pénalité quand plusieurs colonnes peuvent faire le même travail.

</details>
<details><summary>Indice 2</summary>

a) et b) Sans pénalité, des colonnes semblables se partagent le travail, souvent avec des coefficients de signes opposés qui se compensent. Quand la pénalité grandit, ce partage se refait : demande-toi si la pénalité porte sur chaque coefficient, ou sur leur ensemble. c) Le dernier coefficient non nul est celui de la colonne qui explique le plus à elle seule : relis le seuil $\alpha_{\max}$ de la fiche. d) Sur un chemin du Lasso, quand un coefficient s'annule, les autres se réajustent.

</details>
<details><summary>Indice 3</summary>

a) La fiche parle de la **norme** des poids ; une norme peut baisser pendant qu'une de ses coordonnées grandit. b) Un coefficient qui, sans pénalité, sert surtout à corriger ses voisins (avec un signe opposé au leur) peut changer de signe quand ceux-ci rétrécissent ; ici, la moitié des six jouent ce rôle. c) $\alpha_{\max} = \max_j |\mathbf{x}_j^\top \mathbf{y}_c| / n$ : sur des colonnes standardisées, c'est la colonne la plus corrélée à la cible qui résiste le plus longtemps ; sur $[-1, 1]$, la journée de la boutique monte presque tout du long. d) Quand un coefficient s'annule, un autre peut se retrouver utile à nouveau, le temps que la pénalité grandisse encore.

</details>

### Ex 9.20 — Early stopping d'une descente de gradient sur un polynôme de degré 12 🔨

<details><summary>Indice 1</summary>

Recopie le pseudo-code de la fiche (§9.4) en Python. Les deux pièges : la numérotation des epochs, qui commence à 1, et la copie des poids. `train_one_epoch_20` modifie `theta` sur place : un simple nom de plus ne garde rien.

</details>
<details><summary>Indice 2</summary>

Avant la boucle : `rng = np.random.default_rng(seed)`, `theta = np.zeros(X_tr.shape[1])`, `best = math.inf`, `wait = 0`, `losses = []`. Dans `for epoch in range(1, max_epochs + 1)` : une epoch, puis `loss = val_mse_20(theta, X_val, y_val)`, ajoutée à `losses`. Si `loss < best - min_delta` : `best`, `kept_epoch`, `kept_theta = theta.copy()`, et `wait = 0` ; sinon `wait += 1`, et si `wait == patience`, renvoie le quadruplet. Après la boucle, renvoie `max_epochs` comme epoch d'arrêt.

</details>
<details><summary>Indice 3</summary>

```python
def early_stopping_20(X_tr, y_tr, X_val, y_val, patience, min_delta=0.0, max_epochs=1000, lr=0.02, seed=921):
    rng = np.random.default_rng(seed)
    theta = np.zeros(X_tr.shape[1])
    best, kept_epoch, kept_theta, wait, losses = math.inf, 0, theta.copy(), 0, []
    for epoch in range(1, max_epochs + 1):
        train_one_epoch_20(theta, X_tr, y_tr, lr, rng)
        loss = val_mse_20(theta, X_val, y_val)
        losses.append(loss)
        if loss < best - min_delta:                     # a real improvement
            best, kept_epoch, kept_theta, wait = loss, epoch, theta.copy(), 0
        else:
            wait += 1
            if wait == patience:
                return epoch, kept_epoch, kept_theta, losses
    return max_epochs, kept_epoch, kept_theta, losses
```

</details>

### Ex 9.21 — Courbes d'apprentissage sur California avec learning_curve 📦

<details><summary>Indice 1</summary>

Une boucle sur les trois modèles ; pour chacun, un appel à `learning_curve` avec les paramètres imposés par l'énoncé. Il reste à transformer ses scores en MSE moyennes : attention au signe.

</details>
<details><summary>Indice 2</summary>

`sizes, train_scores, val_scores = learning_curve(model, X_cal, y_cal, train_sizes=SIZES_21, cv=KFold(5, shuffle=True, random_state=921), scoring="neg_mean_squared_error", shuffle=True, random_state=921)`. Les scores ont une ligne par taille et une colonne par fold : `-train_scores.mean(axis=1)` donne les MSE moyennes. a) le dernier élément de chaque courbe de validation ; b) `np.argmax` d'un tableau de booléens donne la position du premier `True`.

</details>
<details><summary>Indice 3</summary>

```python
models_21 = {"linear": LinearRegression(),
             "degree 2": make_pipeline(ZScore(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
             "degree 3": make_pipeline(ZScore(), PolynomialFeatures(3, include_bias=False), LinearRegression())}
curves_21 = {}
for name_21, model_21 in models_21.items():
    sizes_21, train_scores_21, val_scores_21 = learning_curve(
        model_21, X_cal, y_cal, train_sizes=SIZES_21, cv=KFold(5, shuffle=True, random_state=921),
        scoring="neg_mean_squared_error", shuffle=True, random_state=921)
    curves_21[name_21] = (sizes_21, -train_scores_21.mean(axis=1), -val_scores_21.mean(axis=1))
final_val_21 = [float(curves_21[name][2][-1]) for name in curves_21]
sizes_21 = curves_21["linear"][0]
crossing_21 = int(sizes_21[np.argmax(curves_21["degree 3"][2] < curves_21["linear"][2])])
```

</details>

### Ex 9.22 — Ridge contre Lasso sur California : chemins de régularisation 📦

<details><summary>Indice 1</summary>

Trois étapes : standardiser les 8 features avec les statistiques des districts d'entraînement ; une compréhension de liste par chemin, une ligne de `coef_` par valeur de `alpha` ; puis les trois questions, qui se lisent sur le chemin du Lasso ou se calculent avec la formule de la fiche.

</details>
<details><summary>Indice 2</summary>

`Z = (X - X.mean(axis=0)) / X.std(axis=0)` ; `np.array([Ridge(alpha=a).fit(Z_train_22, y_train_22).coef_ for a in RIDGE_ALPHAS_22])`, et de même pour `Lasso`. a) Les colonnes de `Z_train_22` sont déjà centrées : il suffit de centrer `y`, puis de prendre `np.max(np.abs(Z_train_22.T @ y_centred)) / n`. b) `np.sum(coef_ != 0)` pour `Lasso(alpha=0.05)`. c) La dernière feature à garder un poids est celle qui atteint le maximum de a) : `np.argmax`, puis `FEATURES_CAL[...]`.

</details>
<details><summary>Indice 3</summary>

```python
X_train_22 = X_cal[TRAIN_CAL]
Z_train_22 = (X_train_22 - X_train_22.mean(axis=0)) / X_train_22.std(axis=0)
ridge_path_22 = np.array([Ridge(alpha=alpha).fit(Z_train_22, y_train_22).coef_ for alpha in RIDGE_ALPHAS_22])
lasso_path_22 = np.array([Lasso(alpha=alpha).fit(Z_train_22, y_train_22).coef_ for alpha in LASSO_ALPHAS_22])
covariances_22 = np.abs(Z_train_22.T @ (y_train_22 - y_train_22.mean())) / len(y_train_22)
alpha_max_22 = float(covariances_22.max())
n_nonzero_22 = int(np.sum(Lasso(alpha=0.05).fit(Z_train_22, y_train_22).coef_ != 0))
last_feature_22 = FEATURES_CAL[int(np.argmax(covariances_22))]
```

</details>

### Ex 9.23 — Lasso par descente de coordonnées et soft_threshold 🔨

<details><summary>Indice 1</summary>

`soft_threshold` tient en une ligne de NumPy. Pour `Lasso.fit`, suis le pseudo-code de l'encadré 🧮 de la fiche (§9.5) ligne à ligne : une boucle sur les passes, une boucle sur les features, et le résidu `r = y - X w` mis à jour à chaque changement d'un poids.

</details>
<details><summary>Indice 2</summary>

`np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)`, après avoir refusé `gamma < 0`. Dans `fit` : `z = np.sum(Xc ** 2, axis=0) / n` une fois pour toutes ; pour la feature `j`, `rho = Xc[:, j] @ residual / n + z[j] * w[j]` (le résidu calculé sans la feature `j`) ; `new = soft_threshold(rho, alpha) / z[j]` ; `residual -= Xc[:, j] * (new - w[j])`. Saute les colonnes où `z[j] == 0`. Retiens le plus grand `abs(new - old)` de la passe, et arrête-toi dès qu'il est `< tol`.

</details>
<details><summary>Indice 3</summary>

```python
def soft_threshold(z, gamma):
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


    def fit(self, X, y):                        # in the class Lasso; predict and score as in Ridge
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        X, y = _check_X_y(X, y)
        n, p = X.shape
        x_mean, y_mean = (X.mean(axis=0), y.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
        Xc, yc = X - x_mean, y - y_mean
        z = np.sum(Xc ** 2, axis=0) / n
        w, residual, n_iter = np.zeros(p), yc.copy(), 0
        for _ in range(self.max_iter):
            n_iter += 1
            largest_change = 0.0
            for j in range(p):
                if z[j] == 0.0:                         # a column of zeros: its weight stays 0
                    continue
                old = w[j]
                rho = Xc[:, j] @ residual / n + z[j] * old
                new = float(soft_threshold(rho, self.alpha)) / z[j]
                if new != old:
                    residual -= Xc[:, j] * (new - old)
                    w[j] = new
                    largest_change = max(largest_change, abs(new - old))
            if largest_change < self.tol:
                break
        self.coef_, self.n_iter_ = w, n_iter
        self.intercept_ = float(y_mean - x_mean @ w) if self.fit_intercept else 0.0
        return self
```

</details>

### Ex 9.24 — Biais et variance mesurés : 50 sous-échantillons de 30 points 🔨

<details><summary>Indice 1</summary>

Deux fonctions indépendantes. `bias_variance_decomposition` applique les deux formules de l'encadré 🧮 de la fiche (§9.6) à un tableau dont chaque ligne est un modèle. `family_24` est une boucle : tirer des jours, entraîner, prédire sur toute l'année, ranger la ligne.

</details>
<details><summary>Indice 2</summary>

Le modèle moyen : `predictions.mean(axis=0)` (la moyenne des lignes). Le biais² : `np.mean((average_model - f_true) ** 2)` ; la variance : `np.mean(predictions.var(axis=0))` (`var` divise par le nombre de modèles par défaut, `ddof=0`). Contrôle les formes avant (2 dimensions, autant de colonnes que `f_true`, au moins 2 lignes). Dans `family_24`, le générateur se crée **une** fois, avant la boucle, et `days = rng.choice(len(x_year), size=n_points, replace=False)` à chaque tour.

</details>
<details><summary>Indice 3</summary>

```python
def bias_variance_decomposition(predictions, f_true):         # in mylearn/linear.py
    predictions, f_true = np.asarray(predictions, dtype=float), np.asarray(f_true, dtype=float)
    if predictions.ndim != 2 or f_true.ndim != 1 or predictions.shape[1] != f_true.shape[0]:
        raise ValueError(f"shapes {predictions.shape} and {f_true.shape} do not match")
    if predictions.shape[0] < 2:
        raise ValueError("at least 2 models are needed to measure a variance")
    average_model = predictions.mean(axis=0)
    return float(np.mean((average_model - f_true) ** 2)), float(np.mean(predictions.var(axis=0)))


def family_24(alpha, n_sets=50, n_points=30, seed=924):        # in the notebook
    rng = np.random.default_rng(seed)
    predictions = np.empty((n_sets, len(x_year)))
    for s in range(n_sets):
        days = rng.choice(len(x_year), size=n_points, replace=False)
        predictions[s] = PolyRidge(DEGREE_24, alpha).fit(x_year[days], wind_year[days]).predict(x_year)
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

```python
def draw_family_25(ax, predictions, f_true, title, ylim=None):
    days = np.arange(np.shape(predictions)[1])
    for curve in predictions:
        ax.plot(days, curve, color="tab:blue", lw=0.7, alpha=0.35)
    ax.plot(days, np.mean(predictions, axis=0), color="tab:orange", lw=2.2, label="mean model")
    ax.plot(days, f_true, color="black", lw=1.5, ls="--", label="ideal curve")
    ax.set(xlabel="day", ylabel="wind (m/s)", title=title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8)


def draw_u_25(ax, alphas, bias2, variance, noise_var):
    bias2, variance = np.asarray(bias2, dtype=float), np.asarray(variance, dtype=float)
    ax.plot(alphas, bias2, "o-", label="bias²")
    ax.plot(alphas, variance, "o-", label="variance")
    ax.plot(alphas, bias2 + variance + noise_var, "o-", color="black", label="bias² + variance + noise")
    ax.axhline(noise_var, color="gray", ls="--", label=f"noise: {noise_var:g}")
    ax.set(xscale="log", yscale="log", xlabel="alpha (log scale)", ylabel="expected squared error")
    ax.legend(fontsize=8)
```

</details>

### Ex 9.26 — Le posterior des droites sur une grille pente-ordonnée 🔨

<details><summary>Indice 1</summary>

Travaille sur toute la grille d'un coup : deux tableaux de forme `(n_b, n_s)` donnent la pente et l'ordonnée de chaque case. Additionne des **logs** (le prior, puis une vraisemblance par point), et ne passe à l'exponentielle qu'à la fin, après avoir retranché le maximum.

</details>
<details><summary>Indice 2</summary>

`S, B = np.meshgrid(slopes, intercepts)` (une ligne par ordonnée, une colonne par pente). `log_post = -(S ** 2 + B ** 2) / (2 * prior_std ** 2)` ; pour chaque point, `log_post -= (yi - S * xi - B) ** 2 / (2 * noise_std ** 2)`. Puis `post = np.exp(log_post - log_post.max())` et `post / post.sum()`. Sans point, la boucle ne fait rien : il reste le prior. Contrôle les longueurs et les écarts-types d'abord.

</details>
<details><summary>Indice 3</summary>

```python
def bayes_line_posterior(x, y, slopes, intercepts, noise_std=0.1, prior_std=1.0):
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError(f"x has {len(x)} values but y has {len(y)}")
    if noise_std <= 0 or prior_std <= 0:
        raise ValueError(f"noise_std and prior_std must be > 0, got {noise_std} and {prior_std}")
    S, B = np.meshgrid(np.asarray(slopes, dtype=float), np.asarray(intercepts, dtype=float))   # (n_b, n_s)
    log_post = -(S ** 2 + B ** 2) / (2.0 * prior_std ** 2)
    for xi, yi in zip(x, y):
        log_post -= (yi - S * xi - B) ** 2 / (2.0 * noise_std ** 2)
    post = np.exp(log_post - log_post.max())       # the largest term becomes 1: no underflow
    return post / post.sum()
```

</details>

### Ex 9.27 — Reproduire la figure 9.20 : prior, vraisemblances, posteriors et droites tirées 🎨

<details><summary>Indice 1</summary>

Pour tirer des droites, aplatis la table : chaque case devient un numéro, tiré avec sa probabilité ; retrouve ensuite sa ligne (l'ordonnée) et sa colonne (la pente). Pour la figure, une boucle sur $k = 0, 1, \dots, n$ : à la ligne $k$, le posterior des $k$ premiers points (ta `bayes_line_posterior`, qui donne le prior quand $k = 0$).

</details>
<details><summary>Indice 2</summary>

`cells = rng.choice(posterior.size, size=n_lines, p=posterior.ravel())`, puis `rows, cols = np.unravel_index(cells, posterior.shape)` et `np.column_stack([slopes[cols], intercepts[rows]])`. La figure : `fig, axes = plt.subplots(len(x) + 1, 4)` ; à la ligne `k`, `posterior = mylearn.linear.bayes_line_posterior(x[:k], y[:k], ...)`. La vraisemblance du `k`-ième point seul : `np.exp(-(y[k-1] - S * x[k-1] - B) ** 2 / (2 * noise_std ** 2))`, sur `S, B = np.meshgrid(slopes, intercepts)`. Les images : `ax.imshow(table, origin="lower", extent=[...])` et `ax.grid(False)`. Une droite tirée se dessine par ses deux bouts, en $x = -1$ et $x = 1$.

</details>
<details><summary>Indice 3</summary>

```python
def sample_lines_27(posterior, slopes, intercepts, n_lines, rng):
    probs = np.asarray(posterior, dtype=float)
    cells = rng.choice(probs.size, size=n_lines, p=probs.ravel() / probs.sum())
    rows, cols = np.unravel_index(cells, probs.shape)
    return np.column_stack([np.asarray(slopes, dtype=float)[cols], np.asarray(intercepts, dtype=float)[rows]])


def bayes_figure_27(x, y, slopes, intercepts, noise_std, prior_std, n_lines, rng):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    S, B = np.meshgrid(slopes, intercepts)
    extent, ends = [slopes[0], slopes[-1], intercepts[0], intercepts[-1]], np.array([-1.0, 1.0])
    fig, axes = plt.subplots(len(x) + 1, 4, figsize=(13, 3.0 * (len(x) + 1)))
    for k, (ax_data, ax_like, ax_post, ax_lines) in enumerate(axes):
        posterior = mylearn.linear.bayes_line_posterior(x[:k], y[:k], slopes, intercepts, noise_std, prior_std)
        ax_data.scatter(x[:k], y[:k], color="tab:blue")
        if k == 0:
            ax_like.axis("off")
        else:
            ax_data.scatter(x[k - 1], y[k - 1], color="red", zorder=3)
            likelihood = np.exp(-(y[k - 1] - S * x[k - 1] - B) ** 2 / (2 * noise_std ** 2))
            ax_like.imshow(likelihood, origin="lower", extent=extent, cmap="gray")
            ax_like.grid(False)
        ax_data.set(xlim=(-1, 1), ylim=(-2, 2))
        ax_post.imshow(posterior, origin="lower", extent=extent, cmap="gray")
        ax_post.grid(False)
        for slope, intercept in sample_lines_27(posterior, slopes, intercepts, n_lines, rng):
            ax_lines.plot(ends, slope * ends + intercept, color="tab:blue", lw=0.8, alpha=0.6)
        ax_lines.set(xlim=(-1, 1), ylim=(-2, 2))
    fig.tight_layout()
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

Quatre étapes faussent l'étude : A (lire les données) et E (découper) sont saines en elles-mêmes, mais E arrive après une étape qui a déjà regardé le test.

```python
def honest_28():
    poly = PolynomialFeatures(3, include_bias=False)

    def fit_predict(rows_fit, rows_eval, alpha):
        P_fit, P_eval = poly.fit_transform(X_cal[rows_fit]), poly.fit_transform(X_cal[rows_eval])
        mean, std = P_fit.mean(axis=0), P_fit.std(axis=0)          # statistics of the fitted rows only
        model = mylearn.linear.Ridge(alpha=alpha)                   # the intercept is not penalised
        model.fit((P_fit - mean) / std, y_cal[rows_fit])
        return model.predict((P_eval - mean) / std)

    folds = list(KFold(5, shuffle=True, random_state=928).split(TRAIN_28))
    cv_mse = [np.mean([mse(y_cal[TRAIN_28[val]], fit_predict(TRAIN_28[tr], TRAIN_28[val], alpha))
                       for tr, val in folds]) for alpha in ALPHAS_28]
    best = ALPHAS_28[int(np.argmin(cv_mse))]
    return best, math.sqrt(mse(y_cal[TEST_28], fit_predict(TRAIN_28, TEST_28, best)))      # the test, once
```

</details>

### Ex 9.29 — Refactoriser l'expérience biais-variance en fonction testée 🛠️

<details><summary>Indice 1</summary>

Compare les trois blocs de `script_29` ligne à ligne : seule la valeur de `alpha` devrait changer. Tout ce qui varie d'autre est une incohérence. Ta fonction, elle, reçoit en paramètres tout ce que le script lisait dans le notebook (les données, la courbe idéale, le modèle), si bien qu'un test peut lui donner des données minuscules et un `fit_predict` trivial.

</details>
<details><summary>Indice 2</summary>

La fonction : pour chaque `p`, un nouveau `rng = np.random.default_rng(seed)`, un tableau `(n_sets, len(x))` de prédictions, puis les deux formules de 9.24. Des propriétés à tester, chacune avec un `fit_predict` défini dans le test : un nombre par valeur de `params` ; des modèles constants n'ont aucune variance, et leur biais² se calcule à la main ; biais² + variance = erreur quadratique moyenne des modèles (un `fit_predict` qui range ses prédictions dans une liste permet de la recalculer) ; des sous-échantillons sans doublon ; même graine, même résultat ; toutes les valeurs de `params` voient les mêmes sous-échantillons.

</details>
<details><summary>Indice 3</summary>

```python
def bias_variance_study(x, y, f_true, fit_predict, params, n_sets=50, n_points=30, seed=0):
    """Squared bias and variance of a family of models, for each value p of `params`.

    For each p, a generator np.random.default_rng(seed) draws n_sets subsamples of n_points indices without
    replacement (so every p sees the same subsamples); fit_predict(x[idx], y[idx], x, p) gives the predictions, at
    every x, of the model fitted on one subsample. Returns two arrays (bias2, variance), one value per p:
    bias2 = mean over x of (mean model - f_true)², variance = mean over x of the variance of the models (ddof=0).
    """
    x, y, f_true = np.asarray(x), np.asarray(y, dtype=float), np.asarray(f_true, dtype=float)
    bias2, variance = [], []
    for p in params:
        rng = np.random.default_rng(seed)                  # the same subsamples for every value of p
        predictions = np.empty((n_sets, len(x)))
        for s in range(n_sets):
            idx = rng.choice(len(x), size=n_points, replace=False)
            predictions[s] = fit_predict(x[idx], y[idx], x, p)
        bias2.append(np.mean((predictions.mean(axis=0) - f_true) ** 2))
        variance.append(np.mean(predictions.var(axis=0)))
    return np.array(bias2), np.array(variance)


def test_constant_models_have_no_variance():
    x, f = np.arange(10.0), np.linspace(-1, 1, 10)
    bias2, variance = bias_variance_study(x, x, f, lambda xt, yt, xe, p: np.full(len(xe), 2.0), [0],
                                          n_sets=4, n_points=3)
    assert variance[0] == pytest.approx(0.0)
    assert bias2[0] == pytest.approx(np.mean((2.0 - f) ** 2))


def test_every_param_sees_the_same_subsamples():
    x = np.arange(20.0)
    mean_of_y = lambda xt, yt, xe, p: np.full(len(xe), yt.mean())     # p is ignored
    bias2, variance = bias_variance_study(x, x ** 2, x, mean_of_y, [1, 2, 3], n_sets=5, n_points=5, seed=2)
    assert np.allclose(bias2, bias2[0]) and np.allclose(variance, variance[0])

# ... and the other properties of indice 2, then TESTS_29 = [test_..., test_..., ...]
```

</details>

### Ex 9.30 — Double descente avec des features aléatoires 🔬

<details><summary>Indice 1</summary>

Les deux premières fonctions tiennent en une ligne chacune (NumPy fait tout). La troisième est une double boucle : sur les tirages, puis sur les valeurs de $p$. Tire `V` et `c` **une** fois par tirage, avec `max(p_grid)` colonnes : chaque $p$ prend les $p$ premières.

</details>
<details><summary>Indice 2</summary>

`np.maximum(0.0, X @ V + c)` (`c` se diffuse sur les lignes) ; `np.linalg.pinv(F) @ y`. Dans `double_descent_30` : `rng = np.random.default_rng(seed)`, puis, à chaque tirage, `V = rng.normal(0, 1 / np.sqrt(5), (5, p_max))` **puis** `c = rng.normal(0, 1, p_max)` (dans cet ordre), les features des 40 points d'entraînement et des 2 000 points de test, et pour chaque `p` : `w = min_norm_fit_30(F_tr[:, :p], y_tr_30)`, les deux MSE avec `F[:, :p] @ w`.

</details>
<details><summary>Indice 3</summary>

```python
def relu_features_30(X, V, c):
    return np.maximum(0.0, X @ V + c)


def min_norm_fit_30(F, y):
    return np.linalg.pinv(F) @ y


def double_descent_30(p_grid, n_draws=20, seed=9300):
    rng = np.random.default_rng(seed)
    p_max = max(p_grid)
    test_mse, train_mse = np.empty((n_draws, len(p_grid))), np.empty((n_draws, len(p_grid)))
    for d in range(n_draws):
        V, c = rng.normal(0, 1 / np.sqrt(5), (5, p_max)), rng.normal(0, 1, p_max)
        F_tr, F_te = relu_features_30(X_tr_30, V, c), relu_features_30(X_te_30, V, c)
        for i, p in enumerate(p_grid):
            w = min_norm_fit_30(F_tr[:, :p], y_tr_30)
            test_mse[d, i] = mse(y_te_30, F_te[:, :p] @ w)
            train_mse[d, i] = mse(y_tr_30, F_tr[:, :p] @ w)
    return test_mse, train_mse
```

</details>

### Ex 9.31 — Défi California : le meilleur modèle linéaire régularisé 🏆

<details><summary>Indice 1</summary>

Le point de départ est un modèle linéaire sur les 8 mesures brutes : il sous-apprend (9.21). Donne-lui plus de capacité avec des features polynomiales, puis tiens-la en laisse avec une pénalité choisie par validation croisée **à l'intérieur** de `build_31`, sur les seuls districts qu'elle reçoit.

</details>
<details><summary>Indice 2</summary>

Une petite classe avec `fit` et `predict` : les features polynomiales des 8 mesures (compare les degrés 2 et 3), le z-score de chaque colonne avec les statistiques des lignes reçues par `fit`, puis ta `Ridge`. Dans `build_31` : une validation croisée à 5 folds sur `X_train` seul, la MSE de validation moyenne pour quelques valeurs de `alpha` (de 0,01 à 100), puis le meilleur modèle réentraîné sur tout `X_train`. Pour le palier 🌟 : la valeur d'un logement dépend de la distance aux grandes villes.

</details>
<details><summary>Indice 3</summary>

```python
class PolyRidge31:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        P = PolynomialFeatures(3, include_bias=False).fit_transform(X)
        self.mean_, self.std_ = P.mean(axis=0), P.std(axis=0)
        self.ridge_ = mylearn.linear.Ridge(alpha=self.alpha).fit((P - self.mean_) / self.std_, y)
        return self

    def predict(self, X):
        return self.ridge_.predict((PolynomialFeatures(3, include_bias=False).fit_transform(X) - self.mean_) / self.std_)


def build_31(X_train, y_train):
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    folds = list(KFold(5, shuffle=True, random_state=0).split(X_train))
    cv_mse = [np.mean([mse(y_train[va], PolyRidge31(alpha).fit(X_train[tr], y_train[tr]).predict(X_train[va]))
                       for tr, va in folds]) for alpha in alphas]
    return PolyRidge31(alphas[int(np.argmin(cv_mse))]).fit(X_train, y_train)
```

Pour 🌟, ajoute avant le degré 3 deux colonnes calculées ligne par ligne : la distance (en degrés de latitude et de longitude) de chaque district à Los Angeles (34,05 ; −118,24) et à San Francisco (37,77 ; −122,42).

</details>
