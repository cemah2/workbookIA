# 14 · Ensembles — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch14_ensembles/06_mes_reponses.md` (créée par `python tools/start_chapter.py 14`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ 14.1, 14.2, 14.5, 14.6 et 14.7 et ∂ 14.3 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la preuve ∂ 14.4, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), des lettres collées pour plusieurs choix (`"AE"`), une suite de codes, une lettre par élément et dans l'ordre (`"XYZ"`), éventuellement séparées par des virgules (`"X, Y, Z"`), une liste pour plusieurs nombres (`[2, 5]`).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 ou 4 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 14.Q1 — Pourquoi un groupe peut battre chacun de ses membres 🧠 ⏱️ 3 min
*Fiche §14.1, §14.2 · livre §14.1, §14.2 · parcours R*

a) Vrai ou faux : quinze classifieurs dont les erreurs sont indépendantes, justes chacun dans 70 % des cas, font mieux en vote à la majorité que chacun d'eux.
b) Pour qu'un vote majoritaire améliore ses membres, il faut avant tout : (A) des membres identiques ; (B) des membres meilleurs que le hasard, dont les erreurs ne coïncident pas ; (C) des membres de plus en plus profonds ; (D) un nombre pair de membres.
c) Vrai ou faux : dix copies du même arbre, entraînées sur les mêmes données par un algorithme sans hasard, ont en votant une meilleure accuracy que l'arbre seul.
d) Vrai ou faux : dans un circuit triplé dont les trois copies votent à la majorité, la panne d'une seule copie est masquée.

### 14.Q2 — Un ensemble corrige-t-il un biais des données ? 🧠 ⏱️ 3 min
*Fiche §14.2 · livre §14.2 · parcours R*

a) Vrai ou faux : si les données d'entraînement sous-représentent un groupe de personnes, entraîner 50 arbres sur des bootstraps de ces données corrige ce biais.
b) Pour limiter un biais des données, le livre propose d'entraîner les membres sur : (A) des bootstraps du même jeu ; (B) des jeux venant de sources différentes et représentatives ; (C) le même jeu, plus longtemps ; (D) des features tirées au hasard.
c) Vrai ou faux : une prévision météo d'ensemble fait tourner le même modèle avec des conditions initiales légèrement différentes.
d) Vrai ou faux : un ensemble réduit l'erreur due à la variance de ses membres, mais pas une erreur que tous ses membres commettent.

### 14.Q3 — Pluralité, majorité et vote pondéré 🧠 ⏱️ 3 min
*Fiche §14.3 · livre §14.3 · parcours R*

Sept votants, numérotés de 0 à 6, prédisent dans l'ordre les classes A, B, C, A, B, A, C.

a) La classe gagnante au vote à la pluralité.
b) Vrai ou faux : cette classe obtient la majorité absolue (plus de la moitié des voix).
c) Les votants reçoivent maintenant les poids $(1 ;\ 2 ;\ 1 ;\ 1 ;\ 2 ;\ 1 ;\ -1)$. Le score de la classe C.
d) Avec ces poids, la classe gagnante.

### 14.Q4 — Tirer avec remise 🧠 ⏱️ 3 min
*Fiche §14.4 · livre §14.4 · parcours R, M*

On tire un bootstrap de 4 exemples parmi les 8 exemples A à H.

a) Vrai ou faux : ce bootstrap peut contenir deux fois l'exemple D.
b) Le plus petit nombre possible d'exemples **distincts** dans ce bootstrap.
c) Le plus petit nombre possible d'exemples out-of-bag (jamais tirés) parmi les 8.
d) Vrai ou faux : un bootstrap de $n$ tirages parmi $n$ exemples contient en moyenne environ 63 % d'exemples distincts, quand $n$ est grand.
e) Vrai ou faux : tirer $n$ exemples **sans** remise parmi $n$ redonne tout le jeu d'entraînement, dans un autre ordre.

### 14.Q5 — Ce que le bagging réduit 🧠 ⏱️ 3 min
*Fiche §14.4 · livre §14.4 · parcours R*

a) Le bagging réduit surtout : (A) le biais ; (B) la variance ; (C) le bruit des labels ; (D) le temps de prédiction.
b) Quel modèle de base profite le plus du bagging ? (A) une régression linéaire ; (B) un arbre profond sans limite ; (C) une souche ; (D) un kNN avec $k = n$.
c) Vrai ou faux : chaque exemple d'entraînement sert à entraîner tous les arbres d'un bagging.
d) Vrai ou faux : le score out-of-bag s'obtient sans mettre de côté un jeu de validation.

### 14.Q6 — Le hasard d'une forêt aléatoire 🧠 ⏱️ 3 min
*Fiche §14.5 · livre §14.5 · parcours R*

a) Avec $p = 16$ features et `max_features="sqrt"`, combien de features examine chaque nœud ?
b) Ce tirage de features a lieu : (A) une fois par forêt ; (B) une fois par arbre ; (C) à chaque nœud qu'on cherche à couper ; (D) une fois par exemple.
c) Vrai ou faux : les features d'un nœud sont tirées avec remise.
d) Quelles sont les deux sources de hasard d'une forêt aléatoire de scikit-learn par défaut ? (A) le bootstrap des exemples ; (B) un seuil tiré au hasard pour chaque feature ; (C) les features tirées à chaque nœud ; (D) l'ordre des exemples. Deux lettres.
e) Vrai ou faux : une forêt avec `max_features=None` et `bootstrap=True` est un bagging d'arbres.

### 14.Q7 — ExtraTrees : encore plus de hasard 🧠 ⏱️ 3 min
*Fiche §14.6 · livre §14.6 · parcours R*

a) Pour chaque feature candidate d'un nœud, un arbre d'ExtraTrees : (A) teste tous les seuils et garde le meilleur ; (B) tire un seul seuil au hasard entre la plus petite et la plus grande valeur de la feature dans le nœud ; (C) prend la médiane ; (D) prend le seuil de l'arbre précédent.
b) Vrai ou faux : par défaut, scikit-learn entraîne chaque arbre d'ExtraTrees sur tout le jeu d'entraînement, sans bootstrap.
c) Vrai ou faux : à profondeur égale, un arbre d'ExtraTrees fait en général baisser l'impureté de l'entraînement un peu moins qu'un arbre aux seuils optimaux.
d) Pourquoi un nœud d'ExtraTrees coûte-t-il moins cher qu'un nœud de forêt aléatoire ? (A) il examine moins de features ; (B) il ne trie pas les valeurs et ne teste pas tous les seuils ; (C) il n'a pas d'enfants ; (D) il ne calcule aucune impureté.

### 14.Q8 — Apprenant faible, souche et apprenant fort 🧠 ⏱️ 3 min
*Fiche §14.7 · livre §14.7 · parcours R*

a) Le nombre de feuilles d'une souche (*decision stump*).
b) Sa profondeur, comptée en arêtes comme `get_depth()`.
c) Vrai ou faux : à deux classes, un classifieur qui se trompe dans 60 % des cas devient juste dans 60 % des cas si l'on échange ses deux réponses.
d) À deux classes équilibrées, un apprenant faible doit avoir une accuracy : (A) d'au moins 90 % ; (B) un peu au-dessus de 50 % ; (C) d'exactement 50 % ; (D) en dessous de 50 %.
e) Vrai ou faux : le boosting n'accepte que des apprenants faibles.

### 14.Q9 — Ce que repondère AdaBoost 🧠 ⏱️ 3 min
*Fiche §14.7 · livre §14.7 · parcours R, M*

a) Après un tour d'AdaBoost, le poids des exemples mal classés : (A) diminue ; (B) augmente ; (C) ne change pas.
b) À deux classes, la valeur que donne la formule de $\alpha$ pour une erreur pondérée $\varepsilon = 0{,}5$ (avec `learning_rate=1`).
c) Plus l'erreur pondérée d'un membre est petite, plus son $\alpha$ est : (A) petit ; (B) grand.
d) Vrai ou faux : AdaBoost entraîne ses membres indépendamment les uns des autres, et peut donc les entraîner en parallèle.
e) Avec $K = 4$ classes, l'erreur pondérée à partir de laquelle SAMME rejette un membre.

### 14.Q10 — Bagging ou boosting ? 🧠 ⏱️ 4 min
*Fiche §14.4, §14.7, bilan · livre §14.4, §14.7 · parcours R*

a) Classe chaque phrase : B (bagging et forêts) ou O (boosting). Six lettres, dans l'ordre.
1. Les membres sont entraînés indépendamment les uns des autres.
2. Chaque membre se concentre sur les erreurs des précédents.
3. Il réduit surtout le biais d'un modèle de base trop simple.
4. Il se parallélise sans effort.
5. Il est plus sensible aux labels faux, qu'il s'acharne à corriger.
6. Il réduit surtout la variance d'un modèle de base instable.

b) Ton modèle de base est une souche, qui underfit nettement. Tu choisis : (A) le bagging ; (B) le boosting.
c) Vrai ou faux : une forêt de 500 arbres surapprend davantage qu'une forêt de 100 arbres aux mêmes réglages.

### 14.Q11 — Gradient boosting : corriger les résidus 🧠 ⏱️ 4 min
*Fiche, section sur le gradient boosting · parcours R, M*

a) En régression avec la MSE, la prédiction de départ $F_0$ d'un gradient boosting est : (A) 0 ; (B) la moyenne des cibles ; (C) leur médiane ; (D) la cible du premier exemple.
b) Un exemple de cible $y = 7$ est prédit $F = 5$. Son résidu.
c) Le learning rate vaut 0,1 et l'arbre suivant prédit exactement ce résidu. La nouvelle prédiction de cet exemple.
d) Vrai ou faux : baisser le learning rate demande en général plus d'arbres pour atteindre la même erreur d'entraînement.
e) Vrai ou faux : contrairement à une forêt, un gradient boosting finit par surapprendre si on lui ajoute des arbres sans contrôle.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 14.R1 — Ch. 13 : L'impureté de Gini d'un nœud 🔁 ★ ⏱️ 5 min
*Ch. 13 (fiche §13.5.2, impuretés et gain) · parcours R, M, C*

Un nœud contient 10 exemples : 6 de la classe 0, 3 de la classe 1 et 1 de la classe 2.

a) Son impureté de Gini.
b) Un split envoie les 6 exemples de la classe 0 à gauche et les 4 autres à droite. L'impureté de Gini de l'enfant droit (3 décimales).
c) Le gain de Gini de ce split.
d) Le seul exemple de la classe 2 reçoit maintenant un poids 4 (`sample_weight`), les autres gardent un poids 1. L'impureté de Gini pondérée du nœud (3 décimales).
e) Vrai ou faux : donner le poids 2 à **tous** les exemples ne change aucune impureté.

### 14.R2 — Ch. 11 : Induction et généralisation 🔁 ★ ⏱️ 5 min
*Ch. 11 (fiche §11.5, l'induction et ses sophismes) · parcours R, M, C*

a) Entraîner un modèle sur des exemples, puis l'appliquer à de nouveaux cas, c'est : (A) une déduction, dont la conclusion est certaine ; (B) une induction, dont la conclusion n'est que probable ; (C) ni l'une ni l'autre.
b) Vrai ou faux : cent arbres entraînés sur des bootstraps d'un échantillon biaisé forment un ensemble sans ce biais.
c) Une équipe évalue un détecteur de fraude sur les seules transactions d'une banque partenaire, puis l'annonce valable pour toutes les banques. Quel sophisme du ch. 11 ? (A) l'argument d'autorité ; (B) l'échantillon biaisé ; (C) l'induction paresseuse ; (D) la sélection des données favorables.
d) Vrai ou faux : un ensemble de modèles dispense de l'hypothèse que les données de service ressemblent aux données d'entraînement.

### 14.R3 — Ch. 7 : Un-contre-tous et un-contre-un 🔁 ★ ⏱️ 5 min
*Ch. 7 (fiche, plusieurs classes avec des classifieurs binaires) · parcours R, M, C*

a) Avec $K = 5$ classes, le nombre de classifieurs binaires de l'approche un-contre-tous.
b) Le nombre de classifieurs binaires de l'approche un-contre-un, pour $K = 5$.
c) Le même nombre pour $K = 10$.
d) Vrai ou faux : en un-contre-un, chaque classifieur binaire n'est entraîné que sur les exemples de ses deux classes.
e) En un-contre-tous, avec $K = 4$ classes de même taille, la part des exemples positifs vus par chaque classifieur binaire.

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 14.1 — Dépouiller un vote pondéré ✏️ ★ ⏱️ 10 min
**Objectif :** appliquer à la main la règle du vote pondéré, avec un poids négatif et la règle d'égalité, et voir quelles transformations des poids changent le résultat.
**Prérequis :** ch. 7 (classification) · fiche §14.3 · **Parcours :** R, M

Cinq votants, numérotés de 0 à 4, de poids $(2 ;\ 1 ;\ 1{,}5 ;\ -1 ;\ 0{,}5)$, prédisent les classes (0, 1 ou 2) de quatre exemples, numérotés de 0 à 3 :

| votant | poids | exemple 0 | exemple 1 | exemple 2 | exemple 3 |
|---|---|---|---|---|---|
| 0 | 2 | 0 | 1 | 2 | 0 |
| 1 | 1 | 1 | 1 | 0 | 2 |
| 2 | 1,5 | 1 | 0 | 2 | 2 |
| 3 | −1 | 0 | 1 | 1 | 0 |
| 4 | 0,5 | 2 | 0 | 0 | 1 |

Chaque classe part d'un score nul ; une égalité donne la plus petite classe.

a) Le score de la classe 1 pour l'exemple 0.
b) La classe gagnante de l'exemple 0.
c) La classe gagnante de l'exemple 1.
d) Les classes gagnantes des quatre exemples, dans l'ordre (une liste).
e) Les classes gagnantes au vote **non pondéré** (une voix par votant), dans l'ordre (une liste).
f) On ne change que le poids du votant 3. La plus petite valeur **entière** de ce poids pour laquelle l'exemple 1 est classé 1.
g) Vrai ou faux : multiplier tous les poids par 3 ne change aucune classe gagnante.
h) Vrai ou faux : ajouter 10 à tous les poids ne change aucune classe gagnante.

Dans ta copie : pourquoi g) et h) n'ont-ils pas la même réponse ? Que fait, concrètement, un votant de poids négatif ?

### Ex 14.2 — Additionner les voix de trois droites, région par région ✏️ ★★ ⏱️ 15 min
**Objectif :** combiner des classifieurs linéaires faibles par un vote pondéré, comme le livre aux figures 14.8 à 14.11, et relier cette convention au vote d'AdaBoost.
**Prérequis :** Ex 14.1 · ch. 10 (frontière d'un classifieur linéaire) · fiche §14.7 · **Fil rouge :** données synthétiques · **Parcours :** M

Trois apprenants faibles, sur le plan $(x_1, x_2)$. Chacun renvoie son poids pour un point de son côté « positif », et 0 sinon :
- A : côté positif $x_1 \ge 1$, poids $w_A = 2$ ;
- B : côté positif $x_2 \ge 1$, poids $w_B = 2$ ;
- C : côté positif $x_1 + x_2 \le 3$, poids $w_C = -3$.

Le score d'un point est la somme des valeurs renvoyées ; le point est classé **1** si son score est strictement positif, **0** sinon. Les trois droites découpent le plan en sept régions, que l'on nomme par les apprenants qui les mettent du côté positif (« AC » : du côté positif de A et de C, du côté négatif de B).

a) Le score de la région « C » (du côté positif de C seulement).
b) Le score du point $(1{,}5 ;\ 1{,}2)$.
c) La classe du point $(0{,}5 ;\ 0{,}5)$.
d) Le nombre de régions classées 1, parmi les sept.
e) Les classes des cinq points $P_1 = (3 ;\ 0{,}5)$, $P_2 = (0{,}5 ;\ 2{,}8)$, $P_3 = (2 ;\ 0{,}5)$, $P_4 = (0{,}2 ;\ 1{,}5)$ et $P_5 = (2{,}5 ;\ 2{,}5)$, dans l'ordre (une liste).
f) On remplace $w_C = -3$ par $w_C = -1$. Le nombre de régions classées 1.
g) Les vraies classes de $P_1$ à $P_5$ sont $(1, 1, 0, 0, 1)$. L'accuracy de l'apprenant A **seul** sur ces cinq points (il prédit 1 de son côté positif, 0 sinon).
h) On code chaque apprenant par $s_m(\mathbf{x}) = +1$ de son côté positif et $-1$ de l'autre. Le score s'écrit alors $\frac12 \sum_m w_m\, s_m(\mathbf{x}) + c$, avec une constante $c$. La valeur de $c$, avec les poids de départ.

Dans ta copie : C a un poids négatif. Par quel apprenant de poids **positif** pourrait-on le remplacer sans changer aucune frontière, et que faudrait-il changer au seuil ? Pourquoi est-ce la convention d'AdaBoost ?

### Ex 14.3 — Probabilité d'être oublié par un bootstrap : vers 36,8 % ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer la probabilité qu'un exemple échappe à un bootstrap, sa limite et le nombre moyen d'exemples distincts, puis les vérifier numériquement.
**Prérequis :** ch. 2 (bootstrap) · ch. 3 (événements indépendants) · 0B (logarithme, exponentielle) · fiche §14.4 · **Parcours :** M

On tire $d$ indices uniformément et avec remise parmi $\{0, 1, \dots, n - 1\}$.

1. Montre qu'un indice donné n'est **jamais** tiré avec la probabilité $(1 - 1/n)^d$. Quelle propriété des tirages utilises-tu ?
2. On note $D$ le nombre d'indices distincts tirés. En écrivant $D$ comme une somme d'indicatrices, une par indice, montre que $\mathbb{E}[D] = n\,\big(1 - (1 - 1/n)^d\big)$.
3. Pour $d = n$, montre que $\ln\big((1 - 1/n)^n\big) = n \ln(1 - 1/n)$ et, en admettant que $\ln(1 + u) / u \to 1$ quand $u \to 0$, que $(1 - 1/n)^n \to e^{-1}$.
4. Un ensemble a $M$ arbres, chacun sur son bootstrap indépendant de $n$ tirages parmi $n$. Montre qu'un exemple donné est out-of-bag pour **au moins un** arbre avec la probabilité $1 - \big(1 - (1 - 1/n)^n\big)^M$.

Vérifications numériques :

a) $n = 8$ exemples et $d = 4$ tirages (la figure 14.1 du livre) : la probabilité qu'un exemple donné ne soit jamais tiré (3 décimales).
b) Toujours $n = 8$ et $d = 4$ : le nombre moyen d'exemples distincts (2 décimales).
c) $d = n = 10$ : la probabilité qu'un exemple donné ne soit jamais tiré (4 décimales).
d) $d = n = 1\,000$ : la même probabilité (4 décimales).
e) $e^{-1}$ (4 décimales).
f) $100 \ln(1 - 1/100)$ (4 décimales).
g) $d = n = 1\,000$ : le nombre moyen d'exemples distincts (1 décimale).
h) $M = 10$ arbres, $n$ grand (prends la limite $e^{-1}$) : la probabilité qu'un exemple soit out-of-bag pour au moins un arbre (3 décimales).

### Ex 14.4 — Ajuster les résidus, c'est descendre le gradient de la MSE ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer que le gradient boosting de régression fait une descente de gradient sur les prédictions, et voir ce que change une autre loss.
**Prérequis :** ch. 5 (descente de gradient) · ch. 9 (MSE) · ch. 13 (arbre de régression, ∂ 13.8) · fiche, section sur le gradient boosting · **Parcours :** M

On a $n$ exemples de cibles $y_1, \dots, y_n$ et des prédictions courantes $F_1, \dots, F_n$, vues comme $n$ nombres libres. On pose $L(F_1, \dots, F_n) = \frac12 \sum_i (y_i - F_i)^2$.

1. Calcule $\partial L / \partial F_i$ et montre que $-\partial L / \partial F_i = y_i - F_i$, le résidu.
2. Écris un pas de descente de gradient de pas $\eta$ sur les $F_i$. Pourquoi ce pas ne suffit-il pas pour prédire un **nouvel** exemple, et que fait l'arbre du gradient boosting à sa place ?
3. Montre que la constante $c$ qui minimise $\sum_i (y_i - c)^2$ est la moyenne $\bar y$ : c'est le $F_0$ du gradient boosting.
4. Une feuille d'un arbre de régression contient les exemples d'un ensemble $S$ et prédit la moyenne de leurs résidus. Montre que c'est la valeur $v$ qui minimise $\sum_{i \in S} \big(y_i - (F_i + v)\big)^2$ : avec la MSE, chaque feuille fait le meilleur pas constant possible pour ses exemples.
5. On remplace la loss par $\sum_i \lvert y_i - F_i \rvert$. Quels sont les pseudo-résidus $-\partial \ell / \partial F_i$ (là où ils existent) ? Quelle constante minimise $\sum_i \lvert y_i - c \rvert$ ? Pourquoi cette loss résiste-t-elle mieux à une cible aberrante ?
6. Avec $\eta = 1$ et des arbres assez profonds pour mettre chaque exemple dans sa propre feuille, que vaut l'erreur d'entraînement après **un** arbre ? Qu'en conclus-tu sur le rôle de $\eta$ et de la profondeur ?
7. En classification binaire, $F$ est un log-odds et $\hat p = \sigma(F)$, avec la log-loss $\ell = -[y \ln \hat p + (1 - y) \ln(1 - \hat p)]$. En reprenant le calcul de ∂ 13.8, donne les pseudo-résidus.

### Ex 14.5 — Quand la majorité de votants indépendants se trompe ✏️ ★★ ⏱️ 25 min
**Objectif :** calculer avec la loi binomiale la probabilité qu'une majorité de votants indépendants ait raison, et voir quand le vote aide et quand il nuit.
**Prérequis :** ch. 3 (événements indépendants) · Ex 14.1 · fiche §14.2 (encadré sur la loi binomiale) · **Parcours :** R, M

Deux classes. Chaque votant a raison avec la probabilité $q$, indépendamment des autres ; l'ensemble prédit la classe de la majorité ($M$ impair).

a) $M = 3$ et $q = 0{,}7$ : la probabilité que la majorité ait raison (3 décimales).
b) $M = 5$ et $q = 0{,}7$ (3 décimales).
c) $M = 3$ et $q = 0{,}4$ (3 décimales).
d) $M = 11$ et $q = 0{,}6$ (4 décimales).
e) $M = 5$ et $q = 0{,}7$ : la probabilité qu'**exactement** trois votants aient raison (4 décimales).
f) Trois votants justes à 70 %, mais les deux derniers recopient toujours la réponse du premier. La probabilité que la majorité ait raison.
g) Le plus petit nombre **impair** $M$ de votants justes à 70 % pour que la majorité ait raison avec une probabilité d'au moins 0,9.
h) $M = 101$ et $q = 0{,}51$ (3 décimales).
i) $M = 25$ et $q = 0{,}45$ (3 décimales).

Dans ta copie : compare h) et i) à $q$. Pourquoi les classifieurs réels sont-ils loin d'être indépendants, et que change, qualitativement, une corrélation positive entre leurs erreurs ?

### Ex 14.6 — Une itération d'AdaBoost à la main ✏️ ★★★ ⏱️ 30 min
**Objectif :** dérouler AdaBoost à la main : erreur pondérée, poids du membre, repondération, souche suivante et vote de l'ensemble.
**Prérequis :** Ex 14.2 · ch. 13 (souche, `sample_weight`) · fiche §14.7 · **Fil rouge :** données synthétiques · **Parcours :** M

Huit exemples sur une droite, d'abscisses $x = 1, 2, \dots, 8$, de classes

| $x$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| classe | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 1 |

Les apprenants faibles sont des souches « $x \le s$ → classe $g$, sinon l'autre classe », avec un seuil $s$ au milieu de deux abscisses ($1{,}5$ ; $2{,}5$ ; … ; $7{,}5$) et une classe $g$ (0 ou 1) pour le côté gauche. À chaque tour, on prend la souche d'erreur pondérée la plus petite (elle est unique ici). Deux classes, `learning_rate=1`.

**Tour 1.** Les poids valent tous $1/8$.

a) L'erreur pondérée $\varepsilon_1$ de la meilleure souche.
b) Son poids $\alpha_1$ (3 décimales).
c) Après la mise à jour et la normalisation, le poids de l'exemple mal classé.
d) Le poids de chacun des autres exemples (4 décimales).

**Tour 2.** Avec ces nouveaux poids :

e) La meilleure souche, sous la forme d'une liste $[s, g]$.
f) Son erreur pondérée $\varepsilon_2$ (4 décimales).
g) Son poids $\alpha_2$ (3 décimales).
h) Les prédictions de l'ensemble des deux souches pour $x = 1, \dots, 8$, dans l'ordre (une liste).

**Tour 3.** Après la mise à jour du tour 2, la meilleure souche est « $x \le 2{,}5$ → 1, sinon 0 ».

i) Son erreur pondérée $\varepsilon_3$ (4 décimales).
j) Son poids $\alpha_3$ (3 décimales).
k) Les prédictions de l'ensemble des trois souches pour $x = 1, \dots, 8$, dans l'ordre (une liste).

**Plusieurs classes.**

l) Avec $K = 3$ classes (SAMME), le poids $\alpha$ d'un membre d'erreur pondérée $0{,}3$ (3 décimales).
m) Vrai ou faux : avec $K = 3$ classes, SAMME garde un membre d'erreur pondérée $0{,}55$.

Dans ta copie : la réponse à c) n'est pas un hasard. Montre que, avec `learning_rate=1` et deux classes, les exemples mal classés par un membre pèsent toujours, ensemble, la moitié du poids total après la mise à jour. Que vaut alors l'erreur pondérée de ce même membre au tour suivant ?

### Ex 14.7 — Deux itérations de gradient boosting sur cinq points ✏️ ★★★ ⏱️ 35 min
**Objectif :** dérouler à la main un gradient boosting de régression : prédiction initiale, résidus, souche de régression, learning rate.
**Prérequis :** Ex 14.4 · ch. 13 (arbre de régression, variance d'un nœud) · fiche, section sur le gradient boosting · **Parcours :** R, M

Cinq exemples d'abscisses $x = 1, 2, 3, 4, 5$ et de cibles $y = 1, 2, 6, 7, 9$. Le gradient boosting utilise des **souches de régression** (un seuil au milieu de deux abscisses ; chaque feuille prédit la moyenne des valeurs que la souche apprend, ici les résidus, et le seuil choisi minimise la somme des carrés des écarts dans les deux feuilles) et un learning rate $\eta = 0{,}5$.

a) La prédiction initiale $F_0$.
b) Les résidus $r_i = y_i - F_0$, dans l'ordre (une liste).
c) Le seuil de la première souche, ajustée aux résidus.
d) Les valeurs de ses deux feuilles, gauche puis droite (une liste, 3 décimales).
e) $F_1(1)$, la prédiction de l'exemple $x = 1$ après la première étape (2 décimales).
f) La MSE d'entraînement après la première étape (3 décimales).
g) Le seuil de la deuxième souche, ajustée aux nouveaux résidus.
h) $F_2(3)$ (4 décimales).
i) $F_2(4{,}2)$ pour un nouvel exemple d'abscisse 4,2 (4 décimales).
j) La MSE d'entraînement après la deuxième étape (4 décimales).
k) Avec $\eta = 1$ au lieu de 0,5, $F_1(1)$.

Dans ta copie : les deux souches n'ont pas le même seuil. Pourquoi la deuxième ne refait-elle pas la même question que la première ? Et pourquoi un petit learning rate demande-t-il plus d'étapes ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 14.8 — Bagging et boosting expliqués en 5 lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant ce que font le bagging et le boosting, et ce qui les distingue.
**Prérequis :** Ex 14.1 · fiche §14.4, §14.7 · **Parcours :** R

Une collègue qui débute en machine learning te demande la différence entre une forêt aléatoire et XGBoost. Explique-lui, en **cinq lignes** au plus :
- une image de la vie courante pour chacun ;
- ce que chacun change d'un membre à l'autre (les données, les poids, la cible) ;
- ce que chacun réduit (biais ou variance) ;
- quand préférer l'un ou l'autre.

Enregistre-toi ou écris ta réponse, puis compare avec la réponse modèle de `05_solutions.md`. Relis ensuite le tableau « Bilan » de la fiche : quelle ligne de ton explication manquait ?

### Ex 14.9 — Un jury de modèles biaisés reste biaisé ⚖️ ★★ ⏱️ 20 min
**Objectif :** voir ce qu'un ensemble peut et ne peut pas corriger dans une décision qui touche des personnes, et proposer des vérifications.
**Prérequis :** Ex 14.5 · fiche §14.2 · **Parcours :** R

Une entreprise trie automatiquement les candidatures à ses offres d'emploi techniques. Elle a entraîné un premier modèle sur dix ans de décisions de ses recruteurs (« convoqué en entretien » ou non). Un audit montre qu'à CV comparables, les candidates sont convoquées nettement moins souvent que les candidats. L'équipe propose : « Remplaçons le modèle par un **ensemble** de 25 gradient boostings entraînés sur des bootstraps des mêmes données, avec des graines différentes : le vote va gommer les biais de chaque modèle. » Réponds dans ta copie.

1. Pourquoi cette proposition ne corrige-t-elle pas le biais constaté ? Utilise le vocabulaire de la fiche (variance, biais commun, erreurs corrélées).
2. Le genre ne fait pas partie des features. Comment le modèle peut-il quand même reproduire le biais ? Donne deux exemples de features qui pourraient servir de substitut.
3. Le livre propose de diversifier les **sources** des données d'entraînement. Que pourrait-on utiliser ici ? Quelle limite garde cette idée ?
4. Quelles mesures, calculées par groupe sur un jeu de validation, permettraient de vérifier qu'un nouveau modèle fait mieux ? Que faut-il pour les calculer, et quelles précautions de confidentialité prendre ?
5. Le modèle trie, un humain décide-t-il encore ? Que change l'encadré sur l'explicabilité de la fiche du ch. 13 (RGPD, AI Act) pour ce système de recrutement ?

### Ex 14.10 — Pourquoi les arbres battent encore le deep learning sur le tabulaire 📄 ★★ ⏱️ 30 min
**Objectif :** lire un article de recherche récent et en tirer ce qu'il démontre, comment, et ses limites.
**Prérequis :** Ex 14.7 · fiche, section sur le gradient boosting · **Parcours :** C

Lis l'article de L. Grinsztajn, E. Oyallon et G. Varoquaux, [« Why do tree-based models still outperform deep learning on typical tabular data? »](https://arxiv.org/abs/2207.08815) (NeurIPS 2022, piste *Datasets and Benchmarks* ; la version arXiv s'intitule « … on tabular data? ») : le résumé, l'introduction, la description du benchmark, les figures des résultats et la section sur les biais inductifs. Réponds dans ta copie.

1. Quelle question pose l'article, et pourquoi les comparaisons publiées avant lui étaient-elles difficiles à interpréter ?
2. Comment le benchmark est-il construit : combien de datasets, quels critères pour les retenir ou les écarter, quelle taille de données, quel budget de réglage des hyperparamètres pour chaque modèle ?
3. Quels modèles sont comparés, et lesquels gagnent, en régression et en classification ?
4. Les auteurs proposent trois explications, chacune testée par une expérience qui modifie les données. Pour chacune, décris l'expérience et ce qu'elle montre.
5. Quelles limites l'article reconnaît-il, ou pourrait-on lui opposer (taille des données, types de features, date de l'étude) ? L'encadré 🕰️ de la fiche cite un résultat plus récent : qu'en conclus-tu pour un projet tabulaire de 2026 ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 14.E1 — Bagging contre boosting : différences et cas d'usage 💼 ★★ ⏱️ 10 min
*Fiche §14.4, §14.7, bilan · parcours R*

« Quelle est la différence entre le bagging et le boosting ? Dans quel cas utilisez-vous l'un ou l'autre ? »

### 14.E2 — Ajouter des arbres à une forêt fait-il surapprendre ? 💼 ★★ ⏱️ 10 min
*Fiche §14.4, §14.5 · parcours R*

« Si j'augmente le nombre d'arbres d'une forêt aléatoire, est-ce que je risque l'overfitting ? Et pour un gradient boosting ? »

### 14.E3 — XGBoost ou réseau de neurones sur un tableau de données ? 💼 ★★ ⏱️ 10 min
*Fiche, section sur le gradient boosting · parcours R*

« On vous donne un tableau de 50 000 lignes et 40 colonnes, numériques et catégorielles, pour prédire un défaut de paiement. Partez-vous sur XGBoost ou sur un réseau de neurones ? Pourquoi ? »

### 14.E4 — Peut-on se fier à feature_importances_ ? 💼 ★★ ⏱️ 10 min
*Fiche §14.5 · parcours R*

« Votre forêt aléatoire classe en tête de ses `feature_importances_` une colonne qui ne devrait rien dire. Comment l'expliquez-vous, et comment mesurez-vous l'importance des features de façon plus fiable ? »

### 14.E5 — Comment régler un gradient boosting ? 💼 ★★ ⏱️ 10 min
*Fiche, section sur le gradient boosting · parcours R*

« Vous avez un gradient boosting à régler. Quels hyperparamètres touchez-vous, dans quel ordre, et comment évitez-vous l'overfitting ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch14_ensembles/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie (`mylearn/ensemble.py`, construit sur tes `tree.py` et `model_selection.py`). La partie 0 du notebook vérifie tes réponses courtes aux quiz, aux rappels et aux exercices ✏️ et ∂ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 14.11 | Trois classifieurs du ch. 13 votent : mieux que le meilleur ? | 🔮 | ★ | 10 |
| 14.12 | Coder le vote pondéré et le tirage bootstrap | 🔨 | ★★ | 25 |
| 14.13 | Jury de votants : indépendants puis corrélés | 🔬 | ★★ | 30 |
| 14.14 | Voting et stacking avec les classifieurs du ch. 13 | 📦 | ★★ | 25 |
| 14.15 | Forêts scikit-learn sur California : OOB et parallélisme | 📦 | ★★ | 25 |
| 14.16 | Combien d'arbres ? Bagging, forêt et ExtraTrees face au bruit | 🔬 | ★★ | 30 |
| 14.17 | Coder BaggingClassifier avec score out-of-bag | 🔨 | ★★★ | 50 |
| 14.18 | Coder RandomForest et ExtraTrees | 🔨 | ★★★ | 45 |
| 14.19 | Importance des features : impureté contre permutation | 📦 | ★★★ | 35 |
| 14.20 | Lire les courbes train/validation d'un boosting | 📈 | ★ | 15 |
| 14.21 | Learning rate et nombre d'arbres : prédire les courbes | 🔮 | ★ | 15 |
| 14.22 | Frontières d'AdaBoost après 1, 3, 10 et 50 souches | 🎨 | ★★ | 25 |
| 14.23 | L'AdaBoost qui n'apprend rien | 🐛 | ★★ | 25 |
| 14.24 | Coder AdaBoost (SAMME) avec des souches pondérées | 🔨 | ★★★ | 50 |
| 14.25 | Coder un gradient boosting de régression | 🔨 | ★★★ | 60 |
| 14.26 | HistGradientBoosting, XGBoost et LightGBM sur California | 📦 | ★★★ | 45 |
| 14.27 | Un benchmark reproductible : graines, versions et tableau de résultats | 🛠️ | ★★ | 25 |
| 14.28 | Défi : RMSE ≤ 0,45 sur California | 🏆 | ★★★★ | 100 |
