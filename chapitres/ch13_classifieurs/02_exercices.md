# 13 · Classifieurs — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch13_classifieurs/06_mes_reponses.md` (créée par `python tools/start_chapter.py 13`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ 13.1 à 13.7 et 🧮 13.9 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la preuve ∂ 13.8, la réflexion (🗣️ ⚖️) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), des lettres collées pour plusieurs choix (`"AE"`), une suite de codes, une lettre par élément et dans l'ordre (`"XYZ"`), éventuellement séparées par des virgules (`"X, Y, Z"`), une liste pour plusieurs nombres (`[2, 5]`).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 ou 4 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 13.Q1 — Pourquoi essayer des classifieurs classiques d'abord ? 🧠 ⏱️ 3 min
*Fiche §13.1 · livre §13.1 · parcours R*

a) Vrai ou faux : un classifieur classique entraîné en quelques secondes fournit une référence qu'un réseau de neurones doit battre nettement pour justifier son coût.
b) Un tableau de 2 000 lignes et 15 features, à classer en deux classes. Que fais-tu en premier ? (A) un réseau de neurones de 50 couches ; (B) une régression logistique et un arbre peu profond, évalués en validation croisée ; (C) rien : on collecte d'abord un million de lignes ; (D) un SVM avec C = 100 000, sans validation.
c) Vrai ou faux : si un kNN atteint 97 % en validation croisée, un réseau de neurones fera forcément mieux.
d) Vrai ou faux : essayer plusieurs classifieurs rapides renseigne sur la structure des données (classes séparables par une droite, régions morcelées, features utiles).

### 13.Q2 — Paramétrique ou non paramétrique ? 🧠 ⏱️ 3 min
*Fiche §13.2 · livre §13.2 · parcours R*

a) Classe chaque classifieur : P (paramétrique) ou N (non paramétrique). Six lettres, dans l'ordre.
1. une régression logistique ;
2. un kNN ;
3. un Naïve Bayes gaussien ;
4. un arbre de décision sans limite de profondeur ;
5. un SVM linéaire ;
6. un SVM à noyau RBF.

b) Lequel garde, dans sa forme de base, **tout** le jeu d'entraînement en mémoire ? (A) la régression logistique ; (B) le kNN ; (C) Naïve Bayes ; (D) le SVM linéaire.
c) Vrai ou faux : la mémoire qu'occupe un modèle paramétrique entraîné ne grandit pas avec le nombre d'exemples d'entraînement.
d) Vrai ou faux : un classifieur non paramétrique n'a aucun paramètre.

### 13.Q3 — kNN n'est pas k-means 🧠 ⏱️ 3 min
*Fiche §13.3 · livre §13.3 · parcours R*

a) Lequel utilise les labels ? (A) le kNN ; (B) le k-means ; (C) les deux ; (D) aucun des deux.
b) Que désigne le $k$ du kNN ? (A) le nombre de clusters ; (B) le nombre de voisins qui votent ; (C) le nombre de features ; (D) le nombre de classes.
c) Vrai ou faux : le `fit` d'un kNN ne fait presque rien d'autre que garder une copie des données.
d) Vrai ou faux : appliqué à un nouveau point, le k-means renvoie un numéro de groupe, qui n'est pas une classe connue d'avance.

### 13.Q4 — Petit k, grand k 🧠 ⏱️ 3 min
*Fiche §13.3 · livre §13.3 · parcours R*

a) Un 1-NN est évalué sur son propre jeu d'entraînement, qui ne contient aucun point en double. Quelle accuracy obtient-il ?
b) Dans un jeu de 400 points, un petit groupe isolé de 6 points de la classe 1 est entouré de points de la classe 0. Avec quel $k$ risque-t-il de disparaître des prédictions ? (A) $k = 1$ ; (B) $k = 3$ ; (C) $k = 50$.
c) Avec $k = n$, le nombre d'exemples d'entraînement, que prédit un kNN à votes uniformes, pour n'importe quel point ? (A) la classe du plus proche voisin ; (B) la classe majoritaire de l'entraînement ; (C) une classe au hasard ; (D) rien : c'est interdit.
d) Vrai ou faux : on choisit $k$ d'après l'accuracy d'entraînement.
e) Vrai ou faux : avec deux classes et des votes uniformes, un $k$ impair évite les égalités de votes.

### 13.Q5 — Marge, vecteurs de support et C 🧠 ⏱️ 3 min
*Fiche §13.4 · livre §13.4 · parcours R, M*

a) Vrai ou faux : on retire un exemple qui n'est pas un vecteur de support, puis on réentraîne un SVM à marge dure : la frontière ne change pas.
b) Dans un SVM à marge souple, augmenter C : (A) élargit la bande et tolère plus de points dedans ; (B) rend chaque violation plus chère, d'où d'ordinaire une bande plus étroite et moins de violations ; (C) ne change rien.
c) Avec une marge souple, quels exemples sont des vecteurs de support ? (A) seulement ceux posés exactement sur les bords de la bande ; (B) ceux sur les bords, dans la bande, et ceux du mauvais côté ; (C) tous les exemples.
d) Vrai ou faux : comme le kNN, un SVM à noyau RBF demande des features standardisées.

### 13.Q6 — Ce que l'astuce du noyau évite de calculer 🧠 ⏱️ 4 min
*Fiche §13.4 · livre §13.4 · parcours R, M*

a) L'astuce du noyau calcule : (A) la transformation $\varphi(\mathbf{x})$ de chaque exemple ; (B) le produit scalaire $\varphi(\mathbf{x}) \cdot \varphi(\mathbf{z})$, sans calculer $\varphi$ ; (C) seulement les distances dans l'espace de départ ; (D) la meilleure transformation parmi toutes les possibles.
b) La valeur du noyau polynomial $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z})^2$ pour $\mathbf{x} = (1, 2)$ et $\mathbf{z} = (2, 2)$.
c) La valeur du noyau RBF $K(\mathbf{x}, \mathbf{x})$, pour n'importe quel $\mathbf{x}$ et n'importe quel $\gamma$.
d) Vrai ou faux : augmenter $\gamma$ rend la frontière d'un SVM RBF plus tortueuse et augmente le risque d'overfitting.
e) Vrai ou faux : une bibliothèque applique d'elle-même le bon noyau ; il n'y a rien à choisir ni à régler.

### 13.Q7 — Racine, feuilles, profondeur : le vocabulaire des arbres 🧠 ⏱️ 3 min
*Fiche §13.5 · livre §13.5 · parcours R*

Un arbre binaire : la racine pose une question ; sa branche « non » mène à une feuille ; sa branche « oui » mène à un nœud qui pose une seconde question, dont les deux branches mènent à des feuilles.

a) Le nombre de feuilles.
b) La profondeur de l'arbre, comptée en arêtes comme `get_depth()`.
c) Le nombre de nœuds internes, racine comprise.
d) Le nombre maximal de feuilles d'un arbre binaire de profondeur 3.
e) Vrai ou faux : un arbre réduit à sa racine (une seule feuille) a une profondeur 0.

### 13.Q8 — Pourquoi un arbre surapprend-il ? 🧠 ⏱️ 3 min
*Fiche §13.5, §13.5.1 · livre §13.5.1, §13.5.3 · parcours R*

a) L'accuracy d'entraînement d'un arbre de scikit-learn sans aucune limite, sur des données sans deux exemples identiques de classes différentes.
b) Vrai ou faux : un arbre de décision a besoin de features standardisées.
c) On réentraîne un arbre sans limite sur un autre tirage de 70 % de données bruitées. On obtient : (A) le même arbre ; (B) souvent un arbre nettement différent ; (C) un arbre de même profondeur, avec les mêmes questions dans un autre ordre.
d) En dimension 2, la frontière d'un arbre de décision est faite : (A) de droites obliques ; (B) de segments parallèles aux axes ; (C) d'arcs de cercle ; (D) d'une seule droite.

### 13.Q9 — Gini ou gain d'information ? 🧠 ⏱️ 3 min
*Fiche §13.5.2 · livre §13.5.2 · parcours R, M*

a) L'impureté de Gini d'un nœud à deux classes à égalité (50 % et 50 %).
b) Son entropie, en bits.
c) L'impureté de Gini d'un nœud pur.
d) Pour calculer le gain d'un split, l'impureté de chaque enfant est : (A) additionnée telle quelle ; (B) pondérée par la part des exemples du parent qui vont dans cet enfant ; (C) remplacée par le maximum des deux enfants ; (D) divisée par le nombre de classes.
e) Vrai ou faux : Gini et entropie choisissent le même split dans la grande majorité des cas.

### 13.Q10 — Limiter la profondeur ou élaguer ? 🧠 ⏱️ 3 min
*Fiche §13.5.3 · livre §13.5.3 · parcours R*

a) Classe chaque réglage : P (on arrête la croissance de l'arbre : pré-élagage) ou E (on coupe après avoir fait pousser l'arbre entier : élagage). Cinq lettres, dans l'ordre.
1. `max_depth=4` ;
2. `min_samples_leaf=10` ;
3. `ccp_alpha=0.01` ;
4. `min_samples_split=20` ;
5. retirer, une fois l'arbre complet construit, les sous-arbres dont l'absence ne dégrade pas l'erreur de validation.

b) Vrai ou faux : plus `ccp_alpha` est grand, moins l'arbre élagué a de feuilles.
c) On choisit `max_depth` : (A) sur le jeu de test ; (B) par validation croisée sur l'entraînement ; (C) toujours le plus grand possible ; (D) d'après l'accuracy d'entraînement.

### 13.Q11 — Ce qui est vraiment « naïf » dans Naïve Bayes 🧠 ⏱️ 4 min
*Fiche §13.6 · livre §13.6 · parcours R, M*

a) Ce qui est « naïf » dans Naïve Bayes : (A) le prior est fixé sans regarder les données ; (B) les features sont supposées indépendantes sachant la classe ; (C) les features sont supposées gaussiennes ; (D) les classes sont supposées équiprobables.
b) Vrai ou faux : un Naïve Bayes gaussien apprend une moyenne et une variance par classe et par feature, à partir des données.
c) Combien de nombres (moyennes et variances, sans les priors) apprend un Naïve Bayes gaussien sur 4 features et 3 classes ?
d) Vrai ou faux : les probabilités `predict_proba` de Naïve Bayes sont en général bien calibrées.
e) Vrai ou faux : sans lissage, un mot absent de tous les messages d'entraînement de la classe « spam » rend impossible la classe « spam » pour tout message qui le contient.

### 13.Q12 — Quel classifieur pour quelle situation ? 🧠 ⏱️ 4 min
*Fiche §13.7 · livre §13.7 · parcours R*

a) Associe à chaque situation le classifieur le mieux adapté comme **premier** essai : A (arbre de décision peu profond), N (Naïve Bayes multinomial), K (kNN), L (régression logistique), S (SVM à noyau RBF). Cinq lettres, dans l'ordre ; chaque lettre sert une fois.
1. Chaque refus doit pouvoir s'expliquer à la personne concernée par trois ou quatre règles simples, du type « si … et … alors … ».
2. Cinquante mille courriels à trier en spam ou non, représentés par leurs comptes de mots ; il faut une baseline en quelques secondes.
3. Quelques centaines d'exemples, de nouveaux exemples étiquetés arrivent chaque jour et doivent servir tout de suite, sans réentraînement.
4. Des données tabulaires ; on veut des probabilités fiables et l'effet de chaque feature, sous la forme d'un poids.
5. Trois mille exemples de 40 features standardisées, une frontière clairement non linéaire, et le temps de régler deux hyperparamètres par validation croisée.

b) Vrai ou faux : pour départager ces candidats sur un même problème, on compare leurs scores de validation croisée calculés sur les mêmes folds.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 13.R1 — Ch. 12 : Pourquoi standardiser avant de mesurer des distances 🔁 ★ ⏱️ 5 min
*Ch. 12 (fiche §12.5, la standardisation) · fil rouge Penguins · parcours R, M, C*

Deux manchots d'entraînement : A (nageoire 190 mm, masse 3 700 g) et B (nageoire 210 mm, masse 3 750 g). Un nouveau manchot C mesure 192 mm et pèse 3 800 g.

a) La distance euclidienne entre C et A, sur les valeurs brutes (2 décimales).
b) La distance euclidienne entre C et B, sur les valeurs brutes (2 décimales).
c) Sur les valeurs brutes, le plus proche voisin de C : A ou B ?
d) On standardise avec les écarts-types de l'entraînement : 14 mm pour la nageoire, 800 g pour la masse (les moyennes s'annulent dans les différences). La distance entre C et A dans l'espace standardisé (3 décimales).
e) Dans l'espace standardisé, le plus proche voisin de C : A ou B ?
f) Vrai ou faux : les écarts-types de la standardisation se calculent sur l'entraînement seul, puis servent tels quels pour C.

### 13.R2 — Ch. 10 : L'équation de la frontière d'un perceptron 🔁 ★ ⏱️ 5 min
*Ch. 10 (fiche, le perceptron et sa frontière) · fil rouge portes logiques · parcours R, M, C*

Un perceptron à deux entrées, de poids $\mathbf{w} = (1, 1)$ et de biais $b = -1{,}5$, sort 1 si $\mathbf{w} \cdot \mathbf{x} + b > 0$, et 0 sinon.

a) Sa sortie pour $(1, 1)$.
b) Sa sortie pour $(0, 1)$.
c) Quelle porte logique réalise-t-il sur les entrées 0 et 1 ? (A) ET ; (B) OU ; (C) OU exclusif ; (D) NON-ET.
d) Sa frontière est la droite $x_2 = a\,x_1 + c$. La pente $a$.
e) L'ordonnée à l'origine $c$.
f) La distance du point $(1, 1)$ à cette frontière, avec la formule de la fiche §13.4 (3 décimales).
g) Vrai ou faux : la frontière d'une régression logistique de mêmes $\mathbf{w}$ et $b$ (là où $\hat p = 0{,}5$) est la même droite.

### 13.R3 — Ch. 6 : Entropie de deux distributions 🔁 ★ ⏱️ 5 min
*Ch. 6 (fiche, entropie et cross-entropy) · parcours R, M, C*

a) L'entropie, en bits, de la distribution $(0{,}5 ;\ 0{,}25 ;\ 0{,}25)$.
b) L'entropie, en bits, de la distribution $(0{,}8 ;\ 0{,}2)$ (3 décimales).
c) L'entropie maximale d'une variable à 4 valeurs, en bits.
d) La cross-entropy $H(p, q) = -\sum_i p_i \log_2 q_i$ de $p = (0{,}5 ;\ 0{,}5)$ par rapport à $q = (0{,}8 ;\ 0{,}2)$, en bits (3 décimales).
e) Vrai ou faux : on a toujours $H(p, q) \ge H(p)$, avec égalité seulement si $q = p$.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 13.1 — Vote des k plus proches voisins à la main ✏️ ★ ⏱️ 15 min
**Objectif :** classer un point à la main avec un kNN, pour plusieurs $k$, deux distances et deux façons de voter.
**Prérequis :** 0B (distance euclidienne) · ch. 7 (classification) · fiche §13.3 · **Fil rouge :** données synthétiques · **Parcours :** R, M

Sept exemples d'entraînement, numérotés de 0 à 6, et leurs classes (0, 1 ou 2) :

| exemple | $x_1$ | $x_2$ | classe |
|---|---|---|---|
| 0 | 1 | 1 | 1 |
| 1 | −1,5 | 0 | 0 |
| 2 | 0 | 2 | 0 |
| 3 | 2 | −1,5 | 1 |
| 4 | −2 | −2 | 2 |
| 5 | 3 | 0 | 2 |
| 6 | 0 | −3,5 | 2 |

On classe le point $\mathbf{q} = (0, 0)$. Les égalités de votes se règlent comme dans scikit-learn : la première classe (la plus petite) l'emporte.

a) Le numéro de l'exemple le plus proche de $\mathbf{q}$, en distance euclidienne.
b) La classe prédite avec $k = 1$, en distance euclidienne.
c) La classe prédite avec $k = 3$, en distance euclidienne et votes uniformes.
d) La probabilité de la classe 0 donnée par `predict_proba` dans le cas c) (3 décimales).
e) La classe prédite avec $k = 5$ (distance euclidienne, votes uniformes).
f) La classe prédite avec $k = 7$ (votes uniformes).
g) Avec $k = 3$ et des votes pondérés par l'inverse de la distance euclidienne, la probabilité de la classe 1 (3 décimales).
h) Avec $k = 7$ et des votes pondérés par l'inverse de la distance euclidienne, la classe prédite.
i) Le numéro de l'exemple le plus proche de $\mathbf{q}$ en distance de **Manhattan**, puis la classe prédite avec $k = 1$ (une liste de deux entiers).

Dans ta copie : comparer e) et f). Que montre le passage de $k = 5$ à $k = 7$ ? Et h) par rapport à f) ?

### Ex 13.2 — Gini et entropie d'une feuille ✏️ ★ ⏱️ 10 min
**Objectif :** calculer les impuretés d'un nœud et interpréter l'impureté de Gini comme une probabilité d'erreur.
**Prérequis :** ch. 6 (entropie en bits) · fiche §13.5.2 · **Parcours :** R, M

Une feuille contient 10 exemples de la classe A et 90 de la classe B (l'exemple du livre, p. 527).

a) Son impureté de Gini.
b) Son entropie, en bits (3 décimales).
c) Le taux d'erreur, sur les exemples de cette feuille, de la règle « toujours répondre B ».
d) Le taux d'erreur de la règle « toujours répondre A ».
e) Le taux d'erreur de la règle « répondre A avec la probabilité 0,1 et B avec la probabilité 0,9, au hasard » (3 décimales).

Une autre feuille a trois classes, dans les proportions 50 %, 30 % et 20 %.

f) Son impureté de Gini.
g) Son entropie, en bits (3 décimales).
h) L'impureté de Gini maximale d'un nœud à trois classes (3 décimales).
i) L'entropie maximale d'un nœud à trois classes, en bits (3 décimales).

### Ex 13.3 — Choisir le meilleur seuil sur la longueur des nageoires ✏️ ★★ ⏱️ 20 min
**Objectif :** dérouler à la main la recherche du meilleur seuil d'un nœud : seuils candidats, impuretés, gain.
**Prérequis :** Ex 13.2 · fiche §13.5.1, §13.5.2 · **Fil rouge :** Penguins · **Parcours :** M

Dix manchots (des valeurs arrondies dans l'esprit de Penguins), de deux espèces : Adélie (classe 0) et Papou (*Gentoo*, classe 1). Longueur de la nageoire, en mm :

| espèce | nageoires (mm) |
|---|---|
| Adélie | 181, 186, 190, 193, 197, 210 |
| Papou | 205, 215, 217, 228 |

Une question « nageoire ≤ $s$ ? » envoie à gauche les manchots dont la nageoire mesure au plus $s$.

a) Le nombre de seuils candidats, au milieu de deux valeurs consécutives triées.
b) L'impureté de Gini du nœud de départ.
c) Le gain de Gini du seuil $s = 201$.
d) Le gain de Gini du seuil $s = 212{,}5$ (3 décimales).
e) Le gain de Gini du seuil $s = 207{,}5$ (3 décimales).
f) Le meilleur seuil parmi tous les candidats.
g) Le gain d'information (entropie, en bits) de ce meilleur seuil (3 décimales).
h) Ce split forme un arbre de profondeur 1 (une souche, *stump*), dont chaque feuille prédit sa classe majoritaire. Combien de manchots d'entraînement classe-t-il mal ?
i) Un nouveau manchot a une nageoire de 203 mm. Quelle classe prédit la souche ?
j) Avec `min_samples_leaf=5` (chaque enfant doit recevoir au moins 5 manchots), combien de seuils candidats restent valides ?

Dans ta copie : les gains de c) et d) sont proches. Lequel des deux seuils choisirait un taux d'erreur à la place de Gini ? Que se passe-t-il ?

### Ex 13.4 — Distance à une droite et marge d'un SVM ✏️ ★★ ⏱️ 20 min
**Objectif :** vérifier à la main une solution de SVM : distances, contraintes, vecteurs de support, largeur de la bande.
**Prérequis :** 0B (produit scalaire, norme) · ch. 10 (frontière $\mathbf{w} \cdot \mathbf{x} + b = 0$) · fiche §13.4 · **Fil rouge :** données synthétiques · **Parcours :** M

Six exemples, numérotés de 0 à 5 : de classe $y = +1$, les points 0 $(3, 3)$, 1 $(5, 4)$ et 2 $(4, 5)$ ; de classe $y = -1$, les points 3 $(1, 1)$, 4 $(0, 1)$ et 5 $(2, 0)$. Un SVM à marge dure donne $\mathbf{w} = (0{,}5 ;\ 0{,}5)$ et $b = -2$, soit $f(\mathbf{x}) = 0{,}5\,x_1 + 0{,}5\,x_2 - 2$.

a) $\lVert \mathbf{w} \rVert$ (4 décimales).
b) $f$ au point 1.
c) La distance du point 1 à la frontière (3 décimales).
d) La plus petite valeur de $y_i\, f(\mathbf{x}_i)$ sur les six exemples.
e) Les numéros des vecteurs de support, dans l'ordre croissant (une liste).
f) La largeur de la bande entre les deux hyperplans $f = \pm 1$, soit $2/\lVert \mathbf{w} \rVert$ (3 décimales).
g) Un nouveau point $(2, 3)$ : la classe prédite ($+1$ ou $-1$).
h) Vrai ou faux : ce point $(2, 3)$ tombe dans la bande ($\lvert f \rvert < 1$).
i) On multiplie $\mathbf{w}$ et $b$ par 2. La distance du point 1 à la frontière change-t-elle ? (Vrai si elle change, Faux sinon.)
j) On retire le point 1 et l'on réentraîne. Vrai ou faux : la solution reste la même.

### Ex 13.5 — L'astuce du noyau vérifiée à la main ✏️ ★★ ⏱️ 20 min
**Objectif :** vérifier sur un exemple qu'un noyau calcule un produit scalaire dans un espace relevé, et calculer un noyau RBF.
**Prérequis :** Ex 13.4 · 0B (produit scalaire, exponentielle) · fiche §13.4 · **Parcours :** M

On prend $\varphi(\mathbf{x}) = (x_1^2,\ \sqrt 2\, x_1 x_2,\ x_2^2)$ et les points $\mathbf{x} = (1, 2)$ et $\mathbf{z} = (3, 1)$.

a) $\mathbf{x} \cdot \mathbf{z}$.
b) Le noyau polynomial $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z})^2$.
c) La deuxième coordonnée de $\varphi(\mathbf{x})$ (3 décimales).
d) $\varphi(\mathbf{x}) \cdot \varphi(\mathbf{z})$, calculé coordonnée par coordonnée.
e) $\lVert \varphi(\mathbf{x}) - \varphi(\mathbf{z}) \rVert^2$, la distance au carré dans l'espace relevé, calculée avec le seul noyau.
f) Le noyau RBF $e^{-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert^2}$ avec $\gamma = 0{,}5$ (4 décimales).
g) Le même noyau avec $\gamma = 1{,}5$, multiplié par $100\,000$ (2 décimales).
h) Pour $p = 1\,000$ features, le nombre de coordonnées de la transformation $\varphi$ du noyau polynomial homogène de degré 2 (les $x_j^2$ et les $\sqrt 2\, x_j x_l$ pour $j < l$).

Dans ta copie : combien de multiplications faut-il pour calculer $K(\mathbf{x}, \mathbf{z})$ en dimension 1 000, et combien pour $\varphi(\mathbf{x}) \cdot \varphi(\mathbf{z})$ une fois $\varphi$ calculée ?

### Ex 13.6 — Anglais ou français ? Naïve Bayes multinomial sur des lettres ✏️ ★★ ⏱️ 25 min
**Objectif :** entraîner à la main un Naïve Bayes multinomial avec lissage de Laplace, puis classer un mot en log-probabilités.
**Prérequis :** ch. 4 (règle de Bayes) · 0B (logarithme) · fiche §13.6 · **Fil rouge :** Holmes/Verne · **Parcours :** M

On ne garde que quatre lettres : e, h, q et w. Dans dix courts extraits d'entraînement, six en anglais (Holmes) et quatre en français (Verne), on a compté ces lettres :

| | e | h | q | w | total |
|---|---|---|---|---|---|
| anglais (EN) | 30 | 12 | 1 | 7 | 50 |
| français (FR) | 38 | 4 | 8 | 0 | 50 |

Les priors sont les fréquences des extraits : $P(\text{EN}) = 0{,}6$ et $P(\text{FR}) = 0{,}4$. Toutes les autres lettres sont ignorées.

a) Sans lissage, $P(\text{w} \mid \text{FR})$.
b) Avec le lissage de Laplace ($\alpha = 1$, quatre lettres), $P(\text{w} \mid \text{EN})$ (3 décimales).
c) Avec le lissage, $P(\text{w} \mid \text{FR})$ (4 décimales).
d) Avec le lissage, $P(\text{e} \mid \text{FR})$ (4 décimales).

On classe le mot « where », qui contient deux e, un h et un w (le r est ignoré).

e) La log-probabilité jointe de la classe EN, $\log P(\text{EN}) + \sum_j x_j \log \hat\theta_{\text{EN}, j}$, en logarithme népérien (3 décimales).
f) La log-probabilité jointe de la classe FR (3 décimales).
g) Le posterior $P(\text{EN} \mid \text{« where »})$ (3 décimales).
h) Sans lissage, le posterior $P(\text{FR} \mid \text{« where »})$.
i) On classe maintenant le mot « que » (un q et un e ; le u est ignoré). Le posterior $P(\text{EN} \mid \text{« que »})$, avec lissage (3 décimales).

Dans ta copie : à quel point l'hypothèse « naïve » est-elle fausse pour des lettres ? Donne un exemple de deux lettres dont les comptes sont liés.

### Ex 13.7 — Posterior d'un Naïve Bayes gaussien ✏️ ★★★ ⏱️ 30 min
**Objectif :** estimer les paramètres d'un Naïve Bayes gaussien, puis calculer un posterior en log-densités.
**Prérequis :** Ex 13.6 · ch. 2 (loi normale, variance) · fiche §13.6 (encadré sur la log-densité) · **Fil rouge :** Penguins · **Parcours :** M

**Estimer.** Quatre manchots Papou d'entraînement ont des nageoires de 210, 215, 220 et 225 mm.

a) La moyenne de leurs nageoires.
b) La variance qu'utilise Naïve Bayes (en divisant par $n$, sans `var_smoothing`).

**Classer.** Un Naïve Bayes gaussien entraîné sur deux espèces a appris (valeurs arrondies) :

| | prior | nageoire : $\mu$ ; $\sigma$ (mm) | profondeur du bec : $\mu$ ; $\sigma$ (mm) |
|---|---|---|---|
| Adélie | 0,55 | 190 ; 6,5 | 18,3 ; 1,2 |
| Papou | 0,45 | 217 ; 6,5 | 15,0 ; 1,0 |

On classe un manchot de nageoire 205 mm et de bec profond de 16,5 mm. Les logarithmes sont népériens.

c) $\log \mathcal{N}(205;\ 190, 6{,}5^2)$ (3 décimales).
d) $\log \mathcal{N}(205;\ 217, 6{,}5^2)$ (3 décimales).
e) $\log \mathcal{N}(16{,}5;\ 18{,}3, 1{,}2^2)$ (3 décimales).
f) $\log \mathcal{N}(16{,}5;\ 15{,}0, 1{,}0^2)$ (3 décimales).
g) La log-probabilité jointe de l'espèce Adélie (3 décimales).
h) La log-probabilité jointe de l'espèce Papou (3 décimales).
i) Le posterior $P(\text{Papou} \mid \mathbf{x})$ (3 décimales).
j) Le même posterior si l'on n'utilise que la nageoire (3 décimales).
k) Vrai ou faux : si, à l'intérieur de chaque espèce, la longueur de la nageoire et la profondeur du bec étaient fortement corrélées, le posterior de i) serait quand même exact.

Dans ta copie : pourquoi la nageoire seule, à 205 mm, à peu près entre les deux moyennes (à 15 mm de 190 et à 12 mm de 217), ne donne-t-elle pas un posterior de 0,5 ? Cite les deux raisons.

### Ex 13.8 — Gradient de la log-loss de la régression logistique ∂ ★★★ ⏱️ 40 min
**Objectif :** démontrer le gradient de la régression logistique, sa convexité, et pourquoi elle a besoin d'une pénalité sur des données séparables.
**Prérequis :** 0B (règle de la chaîne, exponentielle, logarithme) · ch. 5 (descente de gradient) · ch. 6 (log-loss) · fiche, section sur la régression logistique · **Parcours :** M

On note $\sigma(z) = 1/(1 + e^{-z})$ et, pour un exemple $(\mathbf{x}, y)$ avec $y \in \{0, 1\}$, $z = \mathbf{w} \cdot \mathbf{x} + b$, $\hat p = \sigma(z)$ et $\ell = -\big[y \log \hat p + (1 - y) \log(1 - \hat p)\big]$.

1. Montre que $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$, puis que $1 - \sigma(z) = \sigma(-z)$.
2. Montre que $\dfrac{d\ell}{dz} = \hat p - y$. Interprète le signe pour un exemple de label 1 mal prédit.
3. Par la règle de la chaîne, déduis-en $\partial \ell / \partial w_j$ et $\partial \ell / \partial b$.
4. Pour $n$ exemples et $L = \frac1n \sum_i \ell_i$, montre que $\nabla_{\mathbf{w}} L = \frac1n \mathbf{X}^\top (\hat{\mathbf{p}} - \mathbf{y})$, où $\mathbf{X}$ a les exemples en lignes.
5. On ajoute la pénalité $\frac{\alpha}{2} \lVert \mathbf{w} \rVert^2$. Que devient le gradient par rapport à $\mathbf{w}$ ? Et par rapport à $b$ ?
6. Montre que $\dfrac{d^2 \ell}{dz^2} = \hat p\,(1 - \hat p) \ge 0$ : $\ell$ est convexe en $z$. (On admet qu'une fonction convexe de $z$, composée avec une fonction linéaire de $(\mathbf{w}, b)$, est convexe en $(\mathbf{w}, b)$, et qu'une somme de fonctions convexes est convexe.) Pourquoi est-ce une bonne nouvelle pour la descente de gradient ?
7. Les données sont séparables : il existe $(\mathbf{w}_0, b_0)$ avec $y_i = 1 \Leftrightarrow \mathbf{w}_0 \cdot \mathbf{x}_i + b_0 > 0$, et aucun point sur la frontière. Que devient $L(t\,\mathbf{w}_0, t\,b_0)$ quand $t \to +\infty$ ? Conclus : $L$ a-t-elle un minimum ? Que fait la pénalité de la question 5 ?
8. Compare la mise à jour d'un pas de descente de gradient stochastique sur un seul exemple, $\mathbf{w} \leftarrow \mathbf{w} - \eta\,(\hat p - y)\,\mathbf{x}$, à la règle du perceptron du ch. 10. Qu'est-ce qui change pour un exemple déjà bien classé ?

### Ex 13.9 — Mémoire et temps d'un kNN sur MNIST complet 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer la mémoire et le temps de calcul d'un kNN en force brute, et les comparer à un modèle paramétrique.
**Prérequis :** Ex 13.1 · fiche §13.3 (encadré sur la mémoire et le temps) · **Fil rouge :** MNIST · **Parcours :** M

MNIST compte 60 000 images d'entraînement et 10 000 images de test, de $28 \times 28 = 784$ pixels. Un « Mo » vaut $10^6$ octets, un « Go » $10^9$ octets.

a) Le nombre de valeurs que le kNN garde en mémoire.
b) La mémoire occupée en `float64`, en Mo (2 décimales).
c) La même en `uint8` (les pixels bruts, de 0 à 255), en Mo (2 décimales).
d) Le nombre d'opérations d'une prédiction en force brute, à raison de 3 opérations par valeur (soustraction, carré, addition), sans compter le tri.
e) Le temps de prédiction des 10 000 images de test, à $10^{10}$ opérations par seconde, en secondes (2 décimales).
f) La taille, en Go, de la matrice de toutes les distances entre les 10 000 images de test et les 60 000 d'entraînement, en `float64`.
g) La même matrice, calculée par batches de 500 images de test : la taille d'un batch de distances, en Mo.
h) Une régression logistique à 10 classes (un poids par pixel et par classe, plus un biais par classe) : son nombre de paramètres.
i) Le rapport entre la mémoire du kNN en `float64` (question b) et celle de cette régression logistique en `float64` (arrondi à l'entier).
j) Après une PCA à 50 composantes (ch. 12), la mémoire des données d'entraînement réduites en `float64`, en Mo.

<a id="reflexion"></a>

## 🗣️ ⚖️ Réflexion

### Ex 13.10 — La marge d'un SVM expliquée en 5 lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant ce qu'est la marge d'un SVM, ce que sont les vecteurs de support et à quoi sert C.
**Prérequis :** Ex 13.4 · fiche §13.4 · **Parcours :** R

Un ami qui découvre le machine learning te demande pourquoi un SVM ne se contente pas de « n'importe quelle droite qui sépare les classes ». Explique-lui, en **cinq lignes** au plus :
- une image de la vie courante ;
- ce qu'est la marge, et pourquoi la maximiser ;
- ce que sont les vecteurs de support, et pourquoi on les appelle ainsi ;
- ce que fait le paramètre C quand les classes se chevauchent.

Enregistre-toi ou écris ta réponse, puis compare avec la réponse modèle de `05_solutions.md`. Relis ensuite la figure de la marge de la fiche : quelle phrase de ton explication pourrais-tu vérifier sur elle ?

### Ex 13.11 — Refus de crédit : arbre lisible ou SVM opaque ? ⚖️ ★★ ⏱️ 20 min
**Objectif :** peser précision, explicabilité et droits des personnes dans le choix d'un classifieur pour une décision qui les concerne.
**Prérequis :** fiche §13.5, §13.7 (encadré sur l'explicabilité) · **Parcours :** R

Une banque veut automatiser une partie de ses décisions de crédit à la consommation. Sur ses données historiques, deux modèles ont été évalués en validation croisée :
- un **arbre de décision** de profondeur 4 (16 feuilles au plus), qui atteint une AUC de 0,81 ;
- un **SVM à noyau RBF**, réglé par validation croisée, qui atteint une AUC de 0,84.

Les features sont le revenu, l'ancienneté dans l'emploi, les incidents de paiement passés, le montant demandé, l'âge et le code postal. Réponds dans ta copie.

1. Un client se voit refuser son crédit. Avec chacun des deux modèles, que peut-on lui dire, concrètement, sur la raison du refus ? Rédige la réponse qu'il recevrait avec l'arbre.
2. Le code postal et l'âge figurent parmi les features. Quels risques posent-ils, même si le modèle « ne connaît pas » l'origine des clients ? Un arbre rend-il ce risque plus facile ou plus difficile à repérer qu'un SVM ?
3. D'après l'encadré de la fiche sur l'explicabilité, que change le droit européen à ce choix ? Qui, dans la banque, doit pouvoir répondre à une contestation, et avec quelles informations ?
4. Trois points d'AUC séparent les deux modèles. Comment les traduire en conséquences concrètes (crédits mal accordés ou mal refusés) avant de trancher ? De quelles informations aurais-tu besoin ?
5. Propose une solution qui garde l'essentiel des deux mondes (par exemple : un modèle lisible pour décider et un modèle opaque pour alerter, une régression logistique à la place de l'arbre, une explication post hoc du SVM). Pour chacune, une limite.

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 13.E1 — Forces et faiblesses du kNN ; pourquoi standardiser 💼 ★★ ⏱️ 10 min
*Fiche §13.3 · parcours R*

« Quelles sont les forces et les faiblesses des k plus proches voisins ? Pourquoi faut-il standardiser les données avant ? »

### 13.E2 — Comment un arbre de décision choisit-il ses splits ? 💼 ★★ ⏱️ 10 min
*Fiche §13.5.2 · parcours R*

« Expliquez comment un arbre de décision choisit la question qu'il pose à chaque nœud. Quelle différence entre Gini et l'entropie ? »

### 13.E3 — Comment limiter l'overfitting d'un arbre ? 💼 ★★ ⏱️ 10 min
*Fiche §13.5.1, §13.5.3 · parcours R*

« Votre arbre de décision a 100 % d'accuracy sur l'entraînement et 78 % sur la validation. Que faites-vous ? »

### 13.E4 — L'hypothèse de Naïve Bayes, et pourquoi il marche quand même 💼 ★★ ⏱️ 10 min
*Fiche §13.6 · parcours R*

« Quelle hypothèse fait Naïve Bayes ? Elle est presque toujours fausse : pourquoi le modèle marche-t-il quand même, par exemple pour filtrer le spam ? »

### 13.E5 — Quelle baseline : régression logistique, SVM ou arbre ? 💼 ★★ ⏱️ 10 min
*Fiche §13.1, §13.7, section sur la régression logistique · parcours R*

« Vous recevez un nouveau problème de classification sur des données tabulaires. Quelle baseline entraînez-vous en premier, et pourquoi ? Quand passeriez-vous à un SVM ou à un arbre ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch13_classifieurs/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie (`mylearn/neighbors.py`, `tree.py`, `naive_bayes.py` et `logistic.py`). La partie 0 du notebook vérifie tes réponses courtes aux quiz, aux rappels et aux exercices ✏️ et 🧮 ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 13.12 | Prédire la frontière de cinq classifieurs avant de l'afficher | 🔮 | ★ | 15 |
| 13.13 | Choisir k : de la frontière en dents de scie à l'érosion | 🔬 | ★★ | 30 |
| 13.14 | Coder pairwise_distances et KNeighborsClassifier | 🔨 | ★★★ | 45 |
| 13.15 | kNN et malédiction de la dimension | 🔬 | ★★★ | 35 |
| 13.16 | SVM linéaire : vecteurs de support et paramètre C | 📦 | ★★ | 25 |
| 13.17 | Relever les cercles en 3D pour les séparer par un plan | 🎨 | ★★ | 30 |
| 13.18 | Noyau RBF : régler C et gamma par validation croisée | 📦 | ★★★ | 35 |
| 13.19 | Instabilité et profondeur d'un arbre | 🔬 | ★★ | 30 |
| 13.20 | Coder les impuretés et le gain d'un split | 🔨 | ★★ | 25 |
| 13.21 | L'arbre parfait qui triche : une colonne identifiant | 🐛 | ★★ | 20 |
| 13.22 | Coder la recherche du meilleur split | 🔨 | ★★★ | 45 |
| 13.23 | Arbres scikit-learn : plot_tree, export_text et élagage ccp_alpha | 📦 | ★★★ | 35 |
| 13.24 | Coder un arbre de décision récursif | 🔨 | ★★★★ | 100 |
| 13.25 | Préparer l'arbre pour les ensembles : poids, features tirées, importances | 🔨 | ★★★ | 50 |
| 13.26 | Arbre de régression : même squelette, autre critère | 🔨 | ★★ | 30 |
| 13.27 | Naïve Bayes sur des blobs gaussiens puis sur des lunes | 🔮 | ★ | 10 |
| 13.28 | Coder GaussianNB avec log-densités et logsumexp | 🔨 | ★★★ | 40 |
| 13.29 | Coder MultinomialNB avec lissage | 🔨 | ★★ | 30 |
| 13.30 | Dialogue ou narration ? Classer des phrases de Holmes : sac de mots, TF-IDF et Naïve Bayes | 📦 | ★★★ | 40 |
| 13.31 | Régression logistique scikit-learn : coefficients et probabilités | 📦 | ★★ | 25 |
| 13.32 | Coder la régression logistique par descente de gradient | 🔨 | ★★★ | 50 |
| 13.33 | Associer quatre frontières à leurs classifieurs | 📈 | ★ | 15 |
| 13.34 | Mesurer et vectoriser : accélérer son kNN | 🛠️ | ★★ | 25 |
| 13.35 | Comparatif mylearn contre scikit-learn : accuracy et temps | 🔬 | ★★★ | 45 |
| 13.36 | Défi : 96 % sur MNIST avec un classifieur classique | 🏆 | ★★★★ | 100 |
