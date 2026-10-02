# 9 · Surapprentissage et sous-apprentissage — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 📈 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 9.Q1 — Sur- ou sous-apprentissage ? Définitions et symptômes

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

### 9.Q3 — Sous-apprentissage : les vrais remèdes

<details><summary>Indice 1</summary>

Les deux $R^2$ sont presque égaux et bas, alors qu'un autre modèle fait beaucoup mieux sur la même validation. Que dit cette comparaison ?

</details>
<details><summary>Indice 2</summary>

Pour chaque remède, demande-toi s'il rend le modèle plus **souple** (plus de capacité) ou s'il réduit sa **variance**. Le problème de ce modèle-là est-il la variance ?

</details>
<details><summary>Indice 3</summary>

Contre l'underfitting, il faut de la capacité : de meilleures features, moins de pénalité, un modèle plus riche. Plus d'exemples du même genre referme un écart entre entraînement et validation… qui est déjà presque nul ici (regarde le panneau (a) de la figure des courbes d'apprentissage de la fiche). L'encadré ⚠️ de la fiche §9.2.2 répond à d).

</details>

### 9.Q4 — Courbes d'erreur : où commence le surapprentissage ?

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

Combien de jeux de données y a-t-il dans l'approche bayésienne du §9.7 ? Et dans l'expérience du §9.6 ?

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

### Ex 9.1 — MSE et R² à la main sur cinq points

<details><summary>Indice 1</summary>

Commence par écrire les cinq résidus $y_i - \hat{y}_i$ dans une ligne du tableau : tout le reste en découle.

</details>
<details><summary>Indice 2</summary>

MSE : moyenne des carrés des résidus ; MAE : moyenne de leurs valeurs absolues ; $R^2 = 1 - SS_{\text{res}}/SS_{\text{tot}}$, avec $SS_{\text{tot}}$ calculé autour de la moyenne des **cibles**. Pour e), les résidus changent, pas $SS_{\text{tot}}$.

</details>
<details><summary>Indice 3</summary>

Pour f) et g), un seul résidu change (le 5ᵉ, qui passe de 1 à 11) : corrige la somme des carrés et la somme des valeurs absolues, puis divise par 5. Compare les deux facteurs pour h).

</details>

### Ex 9.2 — Moindres carrés : la meilleure droite par dérivées partielles

<details><summary>Indice 1</summary>

$L$ est une somme de carrés : dérive chaque terme $(y_i - a x_i - b)^2$ par rapport à $b$, puis à $a$, avec la règle de la chaîne (0B).

</details>
<details><summary>Indice 2</summary>

La condition sur $b$ donne $\sum_i y_i = a\sum_i x_i + n\,b$. Dans la condition sur $a$, remplace $b$ et utilise le fait que $\sum_i (y_i - \bar{y}) = 0$ et $\sum_i (x_i - \bar{x}) = 0$ pour faire apparaître des écarts à la moyenne.

</details>
<details><summary>Indice 3</summary>

Pour l'application : calcule $\bar{x}$ et $\bar{y}$, puis les deux sommes $\sum (x_i - \bar{x})(y_i - \bar{y})$ et $\sum (x_i - \bar{x})^2$, sans arrondir. Pour f), $\det\begin{pmatrix} p & q \\ q & r \end{pmatrix} = pr - q^2$, et pour 4, compare-le à $n \sum_i (x_i - \bar{x})^2$.

</details>

### Ex 9.3 — Ridge en dimension 1 : w* = Σxy / (Σx² + λ)

<details><summary>Indice 1</summary>

$L(w)$ est une fonction d'une seule variable : annule sa dérivée.

</details>
<details><summary>Indice 2</summary>

$L'(w) = -2\sum_i x_i (y_i - w x_i) + 2\lambda w$ ; regroupe les termes en $w$. Pour les applications, calcule une fois pour toutes $\sum x_i y_i$ et $\sum x_i^2$.

</details>
<details><summary>Indice 3</summary>

Pour d), écris l'équation « $w^*(\lambda) = w^*(0)/2$ » et résous-la en $\lambda$. Pour e), calcule les cinq résidus $y_i - w^* x_i$ avec le $w^*$ de cette valeur de $\lambda$, puis la somme de leurs carrés, sans ajouter la pénalité. Pour f), regarde la somme des carrés comme une parabole en $w$.

</details>

### Ex 9.4 — Biais² et variance à partir d'un tableau de prédictions

<details><summary>Indice 1</summary>

Le modèle moyen est la moyenne de chaque **colonne** du tableau (un point à la fois, sur les quatre modèles).

</details>
<details><summary>Indice 2</summary>

Biais² : en chaque point, (modèle moyen − $f$)², puis moyenne sur les trois points. Variance : en chaque point, la moyenne des carrés des écarts des quatre modèles au modèle moyen (divise par 4), puis moyenne sur les trois points.

</details>
<details><summary>Indice 3</summary>

Pour d), calcule les douze erreurs au carré face à $f$ et divise leur somme par 12 ; tu dois retrouver biais² + variance. Pour e), ajoute le bruit (encadré de la décomposition). Pour f), l'erreur du modèle moyen face à $f$ est un des termes que tu as déjà calculés.

</details>

### Ex 9.5 — Early stopping avec patience sur une courbe de loss

<details><summary>Indice 1</summary>

Fais un tableau à trois lignes sous celui de l'énoncé : la meilleure loss jusqu'ici, l'attente, l'epoch des poids gardés.

</details>
<details><summary>Indice 2</summary>

À chaque epoch : si la loss est strictement sous (meilleure − `min_delta`), elle devient la meilleure et l'attente revient à 0 ; sinon, l'attente augmente de 1, et l'on s'arrête quand elle atteint la patience.

</details>
<details><summary>Indice 3</summary>

Attention à l'epoch 7, qui bat de peu l'epoch 5 : avec une patience trop courte, on ne l'atteint jamais. Avec `min_delta = 0,02`, applique la comparaison **stricte** à chaque epoch, surtout quand la loss baisse d'exactement 0,02 : une baisse égale à `min_delta` ne compte pas.

</details>

### Ex 9.6 — Lasso en dimension 1 : le seuillage doux et les zéros exacts

<details><summary>Indice 1</summary>

Sur $w > 0$, $|w| = w$ ; sur $w < 0$, $|w| = -w$ : sur chaque demi-droite, $g$ est une parabole dérivable.

</details>
<details><summary>Indice 2</summary>

Pour 2, minore $-wz$ par $-|w|\,|z|$. Pour 3, développe la somme des carrés et fais apparaître $\frac{z}{2}\,(w - \rho/z)^2$ plus une constante (mise sous forme canonique).

</details>
<details><summary>Indice 3</summary>

$S(z, \gamma)$ : zéro si $|z| \leq \gamma$, sinon $z$ rapproché de 0 de $\gamma$. Avec les données de 9.3 : $\rho = \frac{1}{n}\sum x_i y_i$ et $z = \frac{1}{n}\sum x_i^2$, puis $w^* = S(\rho, \alpha)/z$. Le poids s'annule dès que $\alpha$ atteint $|\rho|$.

</details>

### Ex 9.7 — Mise à jour bayésienne d'une droite sur une grille 3 × 3

<details><summary>Indice 1</summary>

Fais un tableau de neuf lignes, une par droite $(a, b)$ : le poids du prior, l'écart au premier point, sa vraisemblance, le produit.

</details>
<details><summary>Indice 2</summary>

Pour $P_1 = (1, 1)$, l'écart vaut $r_1 = 1 - (a \times 1 + b)$ ; pour $P_2 = (-1, 0)$, $r_2 = 0 - (a \times (-1) + b)$. Pour b), normalise les produits de la première étape (divise par leur somme).

</details>
<details><summary>Indice 3</summary>

Pour le second point, multiplie les produits de la première étape par la vraisemblance de $P_2$, puis normalise par la nouvelle somme (normaliser aussi entre les deux étapes ne change rien : seule la normalisation finale compte). Pour f), le posterior final est proportionnel à prior × $L_1$ × $L_2$.

</details>

### Ex 9.9 — Diagnostiquer quatre paires de courbes d'entraînement et de validation

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

### Ex 9.8 — Le compromis biais-variance raconté avec le tempo de la boutique

<details><summary>Indice 1</summary>

Imagine que la propriétaire refasse la même journée dix fois, avec d'autres morceaux : que deviendrait chaque courbe ?

</details>
<details><summary>Indice 2</summary>

La première courbe change beaucoup d'une journée à l'autre (variance) ; la seconde change peu, mais rate toujours les mêmes choses (biais).

</details>
<details><summary>Indice 3</summary>

Cherche un exemple où l'on peut réagir trop à un seul jour (un parapluie, une tenue, un itinéraire) et ignorer une tendance réelle. Termine par ce que fait la bonne courbe.

</details>

### Ex 9.10 — Écarter un point aberrant : nettoyage ou manipulation ?

<details><summary>Indice 1</summary>

Sépare trois décisions dans le cas 1 : retirer des points, les retirer aussi du test, et ne pas le dire.

</details>
<details><summary>Indice 2</summary>

Une valeur peut être **aberrante** (rare) sans être **erronée** (fausse). Sur quelle population le modèle du cas 1 sera-t-il utilisé ?

</details>
<details><summary>Indice 3</summary>

Pense à la MAE et à la loss de Huber (fiche §9.2), aux résultats publiés avec et sans les points retirés, à des règles fixées avant de regarder le test, et à qui décide (le métier, pas seulement le data scientist).

</details>

### Ex 9.11 — Belkin et al. (2019) : la double descente

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

Définis d'abord les deux mots avec l'image de beaucoup de jeux de données.

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

### 9.E4 — Détecter le surapprentissage avant la mise en production

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

## Notebook

Les indices des exercices du notebook (9.12 à 9.31) seront ajoutés à la prochaine session de génération, avec le notebook complet.
