# 18 · Rétropropagation — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch18_backprop/06_mes_reponses.md` (créée par `python tools/start_chapter.py 18`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ 18.1 à 18.5 et de l'estimation 🧮 18.9 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les preuves ∂ 18.6 à 18.8, la réflexion (🗣️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), des lettres collées pour plusieurs choix (`"BD"`), une suite de lettres dans l'ordre demandé (`"CADB"`), une liste pour plusieurs nombres (`[2, 5]`), `+1` ou `-1` pour un sens (monter ou baisser).
> Convention du chapitre : le **delta** d'un neurone est $\delta = \partial L / \partial z$, la dérivée de la loss par rapport à sa pré-activation (sans activation, c'est aussi la dérivée par rapport à sa sortie), et l'on met à jour un poids par $w \leftarrow w - \eta\,\partial L / \partial w$.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 18.Q1 — Pourquoi apprendre la rétropropagation si PyTorch la fait pour toi ? 🧠 ⏱️ 3 min
*Fiche §18.1, §18.1.1 · livre §18.1, §18.1.1 · parcours R*

a) Vrai ou faux : dans un projet de deep learning, on écrit d'habitude soi-même le code de la rétropropagation de son réseau.
b) Laquelle de ces raisons de connaître la rétropropagation **n'est pas** donnée par le livre ? (A) lire la littérature et en parler avec d'autres ; (B) comprendre pourquoi un réseau apprend mal ; (C) comprendre les idées nouvelles qui s'appuient sur elle ; (D) obtenir des gradients plus précis que ceux de la bibliothèque.
c) Vrai ou faux : comme un arbre de décision, un réseau de neurones contient d'origine une méthode d'apprentissage qui lui est propre.

Dans ta copie : décris une situation où comprendre la rétropropagation t'aiderait à déboguer un entraînement.

### 18.Q2 — Perturber un poids au hasard : pourquoi est-ce si lent ? 🧠 ⏱️ 3 min
*Fiche §18.2, §18.2.1 · livre §18.2, §18.2.1 · parcours R*

a) Un réseau a 1 000 000 de poids. Avec la méthode lente du livre, un essai consiste à changer un seul poids, puis à refaire une passe avant pour mesurer la nouvelle erreur. Combien de passes avant faut-il, au moins, pour essayer **une fois chaque** poids ?
b) Le sens du changement est tiré au hasard. Pour un poids dont l'erreur varie dans un seul sens autour de sa valeur actuelle, quelle part des essais part, en moyenne, dans le mauvais sens ?
c) Vrai ou faux : dans la méthode lente, un changement qui améliore la prédiction d'un exemple peut dégrader celle d'autres exemples.
d) Comment la méthode traite-t-elle le biais des neurones ? (A) par une règle à part ; (B) comme un poids de plus, grâce à l'astuce du biais du ch. 10 ; (C) elle ne le change jamais ; (D) elle le fixe à 1.

### 18.Q3 — Ce que le gradient dit à chaque poids, et le cas du biais 🧠 ⏱️ 3 min
*Fiche §18.2.2, §18.3 · livre §18.2.2, §18.3 · parcours R*

a) La dérivée de l'erreur par rapport à un poids vaut $-2$ au point actuel. Pour faire baisser l'erreur, faut-il augmenter ce poids (`+1`) ou le diminuer (`-1`) ?
b) Même question pour une dérivée de $+0{,}5$.
c) Vrai ou faux : avec le gradient, on peut déplacer tous les poids en même temps, à condition de faire de petits pas.
d) Pourquoi le livre retire-t-il d'abord les fonctions d'activation ? (A) parce qu'elles n'ont pas de dérivée ; (B) pour simplifier l'explication, avant de les remettre ; (C) parce que la rétropropagation ne marche que sans elles ; (D) parce qu'un réseau sans activation est aussi puissant.

### 18.Q4 — Le delta d'un neurone : signe et effet sur l'erreur 🧠 ⏱️ 3 min
*Fiche §18.4, §18.4.1 · livre §18.4, §18.4.1 · parcours R*

a) Le delta d'un neurone et le changement de sa sortie ont le même signe. L'erreur : (A) monte ; (B) baisse ; (C) ne change pas.
b) Un neurone a un delta de $-2$. Pour faire baisser l'erreur, sa sortie doit-elle monter (`+1`) ou baisser (`-1`) ?
c) Vrai ou faux : le delta relie un changement **relatif** (en pourcentage) de la sortie au changement de l'erreur.
d) La sortie d'un neurone passe de 1 000 à 1 000,4, et son delta vaut $-2{,}5$. Le changement de l'erreur prévu par le delta.
e) Vrai ou faux : la rétropropagation, au sens strict, calcule les deltas ; la mise à jour des poids est une étape à part.

### 18.Q5 — Le réseau minuscule : couches, poids et leurs noms 🧠 ⏱️ 3 min
*Fiche §18.5, §18.6 · livre §18.5, §18.6*

a) Le réseau minuscule du livre ($X, Y \to A, B \to C, D$, sans biais) : son nombre de poids.
b) Le poids qui relie le neurone $B$ au neurone $C$ s'appelle : (A) $BC$ ; (B) $CB$ ; (C) $B_\delta$ ; (D) $C_o$.
c) Le nombre de couches de ce réseau, selon la convention du livre (la couche d'entrée ne compte pas).
d) Pour un exemple, les sorties valent $P_1 = 0{,}3$ et $P_2 = 0{,}8$. Quelle classe le réseau prédit-il ? (A) celle de $P_1$ ; (B) celle de $P_2$ ; (C) aucune, car $P_1 + P_2 \ne 1$.
e) Le label one-hot d'un exemple de la classe d'indice 2, pour quatre classes d'indices 0 à 3 (une liste).

### 18.Q6 — Delta de sortie, delta caché, mise à jour d'un poids 🧠 ⏱️ 3 min
*Fiche §18.7, §18.8 · livre §18.7, §18.8 · parcours R*

a) Sans activation, un poids part d'un neurone dont la sortie vaut $a$ et arrive à un neurone de delta $\delta$. Son gradient $\partial L / \partial w$ vaut : (A) $a\,\delta$ ; (B) $a + \delta$ ; (C) $\delta / a$ ; (D) $w\,\delta$.
b) Avec $a = 2$, $\delta = -0{,}5$, $\eta = 0{,}1$ et un poids qui vaut 1, le poids après la mise à jour.
c) Un neurone caché, sans activation, alimente deux neurones par des poids $0{,}5$ et $-1$ ; leurs deltas valent 2 et 3. Son delta.
d) Vrai ou faux : pour calculer les deltas d'une couche cachée, il suffit des deltas de la couche suivante et des poids qui y mènent.

### 18.Q7 — Rétropropager couche par couche, puis avec une activation 🧠 ⏱️ 3 min
*Fiche §18.9, §18.10 · livre §18.9, §18.10 · parcours R*

a) Dans le réseau 2 → 2 → 4 → 3 → 2 du livre, quelle couche reçoit ses deltas en premier ? (A) celle de 2 neurones cachés ; (B) celle de 4 ; (C) celle de 3 ; (D) la couche de sortie.
b) Le nombre de neurones de la couche dont on calcule les deltas juste après.
c) Avec une activation $f$, la somme pondérée des deltas qui arrive à un neurone caché est multipliée par : (A) $f(z)$ ; (B) $f'(z)$ ; (C) $f'(a)$ ; (D) $1/f'(z)$.
d) Ce facteur, pour un neurone sigmoïde de pré-activation $z = 0$.
e) Vrai ou faux : avec l'activation identité, la rétropropagation est exactement celle des §18.4 à §18.9 (sans activation).

### 18.Q8 — Learning rate trop grand ou trop petit : reconnaître les symptômes 🧠 ⏱️ 3 min
*Fiche §18.11, §18.11.1 · livre §18.11, §18.11.1 · parcours R*

Pour a) et b), choisis parmi : (A) learning rate trop grand ; (B) learning rate trop petit ; (C) learning rate bien choisi ; (D) un bug dans les données.

a) La loss oscille sans jamais baisser, et un poids saute d'une valeur à une autre à chaque pas.
b) La loss baisse régulièrement, mais très lentement.
c) Vrai ou faux : une accuracy bloquée à 0,5 sur deux classes équilibrées peut venir d'un réseau qui prédit toujours la même classe.
d) Une loss reste longtemps sur un plateau, puis chute rapidement. Selon le livre, l'explication la plus fréquente : (A) un minimum local ; (B) une région presque plate, comme un point selle ; (C) un bug ; (D) le hasard des exemples.
e) Vrai ou faux : avec un learning rate nul, le réseau n'apprend rien.

### 18.Q9 — Ce que la rétropropagation propage vraiment 🧠 ⏱️ 3 min
*Fiche §18.12, §18.12.1 à §18.12.3 · livre §18.12, §18.12.1 à §18.12.3 · parcours R*

a) Vrai ou faux : la rétropropagation fait remonter l'erreur de la sortie vers l'entrée.
b) Le delta d'un neurone $C$ représente : (A) l'erreur commise par $C$ ; (B) la dérivée de la loss par rapport à la sortie (ou à la pré-activation) de $C$ ; (C) le poids qui arrive à $C$ ; (D) la sortie de $C$.
c) Range ces étapes dans l'ordre d'un pas d'apprentissage, en lettres collées : (A) la mise à jour des poids ; (B) la passe avant ; (C) les deltas des couches cachées ; (D) les deltas de la couche de sortie.
d) Vrai ou faux : la rétropropagation choisit elle-même la taille du pas de la descente de gradient.

### 18.Q10 — Neurones saturés et ReLU mortes 🧠 ⏱️ 3 min
*Fiche §18.12.4 · livre §18.12.4 · parcours R*

a) Vrai ou faux : pour une pré-activation $z = 15$, la dérivée de la sigmoïde est plus petite que $10^{-6}$.
b) Un neurone ReLU a une pré-activation négative pour tous les exemples : la valeur de sa dérivée.
c) Lesquels de ces gestes aident contre la saturation (lettres collées, dans l'ordre alphabétique) ? (A) des poids initiaux plus petits ; (B) un learning rate beaucoup plus grand ; (C) une régularisation qui garde les poids petits ; (D) une activation sans plateau pour les $z$ positifs, comme ReLU ; (E) plus de neurones sigmoïdes dans chaque couche.
d) Vrai ou faux : une ReLU ne sature pas pour les pré-activations positives.

### 18.Q11 — Mini-batches, mises à jour parallèles et coût du backward 🧠 ⏱️ 3 min
*Fiche §18.12.5 à §18.12.7 · livre §18.12.5 à §18.12.7 · parcours R*

a) Avec des mini-batches de 32 exemples, le nombre de mises à jour des poids pour une epoch de 6 400 exemples.
b) Vrai ou faux : avec des mini-batches, on met à jour les poids après chaque exemple.
c) Le coût d'une passe arrière complète (deltas **et** gradients des poids) vaut à peu près : (A) une passe avant ; (B) deux passes avant ; (C) cent passes avant ; (D) le carré du nombre de poids.
d) Vrai ou faux : toutes les mises à jour d'un pas peuvent se faire en même temps, parce que celle d'un poids ne dépend que des valeurs à ses deux bouts.

### 18.Q12 — Aucune garantie, une longue histoire, des équations à apprivoiser 🧠 ⏱️ 3 min
*Fiche §18.12.8 à §18.12.10 · livre §18.12.8 à §18.12.10*

a) Vrai ou faux : comme pour le perceptron du ch. 10, on sait prouver que la rétropropagation finit toujours par trouver de bons poids.
b) Selon le livre, le premier réflexe quand un réseau n'apprend pas : (A) augmenter le learning rate ; (B) le baisser fortement ; (C) ajouter des couches ; (D) changer de données.
c) L'année du mémoire de master de S. Linnainmaa sur le mode inverse.
d) L'année de l'article de Rumelhart, Hinton et Williams dans *Nature*.
e) En forme matricielle, le gradient qui repart vers la couche d'avant est la matrice des deltas multipliée par : (A) $\mathbf{W}$ ; (B) $\mathbf{W}^\top$ ; (C) $\mathbf{W}^{-1}$ ; (D) $\mathbf{W}^2$.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 18.R1 — Ch. 17 : La dérivée de la sigmoïde à partir de sa sortie 🔁 ★ ⏱️ 5 min
*Ch. 17 (fiche, la sigmoïde et ses dérivées) · parcours R, M*

a) Un neurone sigmoïde sort $a = \sigma(z) = 0{,}8$. La valeur de $\sigma'(z)$.
b) Même question pour $a = 0{,}5$.
c) Un neurone tanh sort $\tanh(z) = 0{,}6$. La valeur de $\tanh'(z)$.
d) Vrai ou faux : pendant la passe arrière, connaître la sortie $a = \sigma(z)$ suffit pour obtenir $\sigma'(z)$.
e) La dérivée de ReLU (convention de PyTorch) en $z = -0{,}3$.

### 18.R2 — Ch. 15 : Chercher le meilleur learning rate par recherche en grille 🔁 ★ ⏱️ 5 min
*Ch. 15 (fiche, les recherches d'hyperparamètres) · parcours R*

On lance `GridSearchCV(MLPClassifier(solver="sgd"), {"learning_rate_init": [0.0001, 0.001, 0.01, 0.1]}, cv=5)` (avec `refit=True`, la valeur par défaut).

a) Le nombre total d'entraînements de `MLPClassifier`, réentraînement final compris.
b) Pourquoi une grille logarithmique pour le learning rate ? (A) son effet se juge par ordres de grandeur ; (B) scikit-learn l'impose ; (C) elle teste plus de valeurs ; (D) elle évite la validation croisée.
c) Après `fit`, où lire la valeur retenue ? (A) `best_params_` ; (B) `best_estimator_.coefs_` ; (C) `cv_results_["params"][0]` ; (D) `score`.
d) Vrai ou faux : dans `MLPClassifier`, `learning_rate_init` ne sert qu'avec les solveurs `"sgd"` et `"adam"`.

### 18.R3 — Ch. 11 : Représentation, évaluation, optimisation dans un MLP 🔁 ★ ⏱️ 5 min
*Ch. 11 (fiche, les trois étapes de l'apprentissage) · parcours R*

Selon Domingos (ch. 11), un algorithme d'apprentissage combine une représentation, une évaluation et une optimisation. Pour un MLP entraîné par rétropropagation, choisis pour chaque élément : (A) représentation ; (B) évaluation ; (C) optimisation.

a) La loss (par exemple la cross-entropy).
b) La descente de gradient, avec les gradients de la rétropropagation.
c) L'architecture : le nombre de couches, de neurones, les activations.
d) Vrai ou faux : si la représentation ne peut pas exprimer la bonne frontière, une meilleure optimisation n'y changera rien.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**. Sauf mention contraire, la loss est l'erreur quadratique $L = \frac12 \sum_k (\hat{y}_k - y_k)^2$.

### Ex 18.1 — Le delta, amplificateur de changement ✏️ ★ ⏱️ 10 min
**Objectif :** prévoir le changement de l'erreur à partir du delta d'un neurone, et mesurer ce que vaut cette prévision.
**Prérequis :** ch. 5 (dérivée, tangente) · fiche §18.4.1 (encadré sur la meilleure approximation linéaire) · **Parcours :** R, M

a) Un neurone a un delta de $-3$. Sa sortie passe de $1{,}2$ à $1{,}5$. Le changement de l'erreur prévu par le delta.
b) Un neurone a un delta de $0{,}5$. Sa sortie passe de 4 à $3{,}6$. Le changement prévu.
c) Un neurone a un delta de $-1{,}5$. De combien faut-il changer sa sortie pour faire baisser l'erreur de $0{,}06$ ?
d) Pour une sortie $o$ et une erreur $E(o) = (o - 2)^2$, le delta en $o = 3$.
e) La sortie passe de 3 à $3{,}1$. Le changement de l'erreur prévu par ce delta.
f) Le vrai changement de l'erreur (2 décimales).
g) Vrai ou faux : un delta négatif veut dire que l'erreur est négative.

Dans ta copie : d'où vient l'écart entre e) et f) ? Que deviendrait-il pour un changement de sortie de $0{,}01$ ?

### Ex 18.2 — Deltas de sortie avec l'erreur quadratique : attention au signe ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer les deltas de sortie, et voir sur un pas ce que change le signe de la convention.
**Prérequis :** 0B (dérivée d'un carré) · Ex 18.1 · fiche §18.6 (encadré « le signe du delta de sortie ») · **Parcours :** R, M

Un réseau sans activation de sortie rend $\hat{\mathbf{y}} = (0{,}6 ;\ -0{,}2)$ pour un exemple de label one-hot $\mathbf{y} = (0 ;\ 1)$.

a) La loss $L$.
b) Le delta $\delta_1 = \partial L / \partial \hat{y}_1$.
c) Le delta $\delta_2$.
d) Si $\hat{y}_2$ augmente de $0{,}01$, le changement de $L$ prévu par $\delta_2$.
e) On déplace directement les deux sorties d'un pas $\hat{\mathbf{y}} \leftarrow \hat{\mathbf{y}} - 0{,}5\,\boldsymbol{\delta}$. La nouvelle loss (valeur exacte).
f) Même pas, mais avec la convention de la figure 18.20 du livre, $\boldsymbol{\delta}' = \mathbf{y} - \hat{\mathbf{y}}$ : $\hat{\mathbf{y}} \leftarrow \hat{\mathbf{y}} - 0{,}5\,\boldsymbol{\delta}'$. La nouvelle loss (valeur exacte).
g) Sans le $\frac12$ : pour $E = \sum_k (\hat{y}_k - y_k)^2$, la valeur de $\partial E / \partial \hat{y}_1$.

Dans ta copie : comment le livre peut-il trouver une pente négative sur sa figure 18.16 et un delta « label moins sortie » positif pour la même situation ?

### Ex 18.3 — Descente sur une parabole : quel learning rate ? ✏️ ★★ ⏱️ 15 min
**Objectif :** relier le learning rate à la courbure de la loss, et trouver les seuils de convergence.
**Prérequis :** ch. 5 (descente de gradient) · fiche §18.11 (encadré sur la descente sur une parabole) · **Parcours :** R, M

On minimise $L(w) = 2\,(w - 1)^2$ (c'est $\frac{a}{2}(w - w^\star)^2$ avec $a = 4$ et $w^\star = 1$) par $w_{t+1} = w_t - \eta\,L'(w_t)$, en partant de $w_0 = 3$.

a) Avec $\eta = 0{,}1$, le facteur par lequel l'écart $w_t - w^\star$ est multiplié à chaque pas.
b) La valeur de $w_1$.
c) La valeur de $w_3$ (3 décimales).
d) Le learning rate qui atteint $w^\star$ en un seul pas.
e) La valeur limite du learning rate : au-dessus, la descente diverge (au-dessous, elle converge).
f) Avec $\eta = 0{,}4$, la valeur de $w_2$.
g) Pour une loss beaucoup plus plate, $L(w) = 0{,}05\,(w - 1)^2$ (soit $a = 0{,}1$), la même valeur limite qu'en e).
h) Avec $\eta = 0{,}1$ et la loss de départ, le nombre de pas nécessaires pour que $|w_t - w^\star| < 0{,}01$.

Dans ta copie : que dit g) de l'affirmation du livre selon laquelle le learning rate est « un nombre entre 0 et 1 » ?

### Ex 18.4 — Rétropropagation complète du réseau minuscule, sans activation ✏️ ★★ ⏱️ 30 min
**Objectif :** faire à la main une passe avant, une passe arrière et un pas de mise à jour du réseau minuscule.
**Prérequis :** Ex 18.2 · fiche §18.5 à §18.9 · **Parcours :** R, M

Le réseau du livre, $X, Y \to A, B \to C, D$, sans biais et sans activation, reçoit l'exemple $(X ;\ Y) = (1 ;\ 2)$ de label $(1 ;\ 0)$. Ses poids, rangés comme au ch. 16 (une ligne par neurone de départ) :

$$\mathbf{W}_1 = \begin{pmatrix} XA & XB \\ YA & YB \end{pmatrix} = \begin{pmatrix} 0{,}5 & 1 \\ -0{,}5 & 0{,}25 \end{pmatrix}, \qquad \mathbf{W}_2 = \begin{pmatrix} AC & AD \\ BC & BD \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ 0{,}5 & 0{,}5 \end{pmatrix}.$$

a) Les sorties $[A_o, B_o]$ des neurones cachés (une liste).
b) Les sorties $[C_o, D_o]$ du réseau.
c) La loss (valeur exacte).
d) Les deltas de sortie $[\delta_C, \delta_D]$.
e) Les deltas cachés $[\delta_A, \delta_B]$.
f) Les gradients des poids de la seconde couche, dans l'ordre de $\mathbf{W}_2$ ligne par ligne : $[\partial L/\partial AC,\ \partial L/\partial AD,\ \partial L/\partial BC,\ \partial L/\partial BD]$.
g) Les gradients des poids de la première couche, dans l'ordre de $\mathbf{W}_1$ ligne par ligne : $[\partial L/\partial XA,\ \partial L/\partial XB,\ \partial L/\partial YA,\ \partial L/\partial YB]$.
h) Le changement de loss prévu par le gradient pour un pas $\eta = 0{,}1$ sur les huit poids : $-\eta \sum_w (\partial L/\partial w)^2$.
i) La vraie loss après ce pas (4 décimales).

Dans ta copie : pourquoi la prévision h) est-elle si loin du vrai changement ? Quel poids a le plus grand gradient, et pourquoi ?

### Ex 18.5 — Le même réseau avec une sigmoïde cachée : le facteur σ'(z) ✏️ ★★ ⏱️ 25 min
**Objectif :** refaire la rétropropagation avec une activation, et voir ce que le facteur $f'(z)$ fait aux deltas.
**Prérequis :** Ex 18.4 · ch. 17 (sigmoïde, $\sigma' = \sigma(1 - \sigma)$) · fiche §18.10 · **Parcours :** M

Même réseau, mêmes poids, même exemple, mais les neurones cachés $A$ et $B$ ont maintenant une **sigmoïde** ; les sorties restent sans activation. Réponds avec 4 décimales.

a) La sortie $A_o = \sigma(z_A)$.
b) La sortie $B_o$.
c) La sortie $C_o$.
d) Le delta $\delta_C$.
e) Le facteur $\sigma'(z_A)$.
f) Le delta $\delta_A$.
g) Le delta $\delta_B$.
h) Le gradient $\partial L / \partial YA$.

Dans ta copie : compare $\delta_A$ ici et dans l'exercice 18.4. D'où vient l'écart ? Que se passerait-il si $z_A$ valait 8 ?

### Ex 18.6 — La règle de la chaîne derrière les deltas ∂ ★★ ⏱️ 25 min
**Objectif :** démontrer les formules de la rétropropagation pour un neurone avec la règle de la chaîne.
**Prérequis :** 0B (règle de la chaîne, y compris à plusieurs chemins) · Ex 18.4 · fiche §18.7, §18.8, §18.10 · **Parcours :** M

Un neurone $j$ a la pré-activation $z_j = \sum_i a_i\,w_{ij} + b_j$ et la sortie $a_j = f(z_j)$ ; ses entrées $a_i$ sont les sorties de la couche d'avant, et il alimente les neurones $k$ de la couche suivante. On note $\delta_j = \partial L / \partial z_j$. Réponds dans ta copie.

1. Montre que $\partial L / \partial w_{ij} = a_i\,\delta_j$ et que $\partial L / \partial b_j = \delta_j$.
2. Avec la règle de la chaîne à plusieurs chemins, montre que $\delta_j = f'(z_j) \sum_k w_{jk}\,\delta_k$. D'où vient la somme ?
3. Pour un neurone de sortie et $L = \frac12 \sum_k (a_k - y_k)^2$, montre que $\delta_k = (a_k - y_k)\,f'(z_k)$. Que devient-il sans activation ?
4. Le livre appelle « delta » la dérivée $\partial L / \partial a_j$ par rapport à la sortie. Exprime-la avec $\delta_j$. Pourquoi les deux coïncident-elles sans activation ?
5. Une couche de $m$ neurones reçoit $n$ entrées. Compte les multiplications pour calculer ses $m$ pré-activations à partir des $n$ entrées, puis les $n$ sommes $\sum_k w_{jk}\,\delta_k$ de la couche d'avant à partir des $m$ deltas. Que retrouves-tu (figure 18.31) ?
6. On change un seul poids $w_{ij}$ de $\Delta w$. Montre que $\Delta L \approx a_i\,\delta_j\,\Delta w$, et que le pas $\Delta w = -\eta\,a_i\,\delta_j$ fait baisser $L$ d'environ $\eta\,(a_i\,\delta_j)^2$. Le livre affirme qu'en retirant $a_i\,\delta_j$ au poids, on fait baisser l'erreur de 1 (p. 747) : est-ce vrai ? Quel changement fait baisser l'erreur de 1, au premier ordre ?

### Ex 18.7 — Forme matricielle : XᵀΔ, ΔWᵀ et somme des lignes ∂ ★★★ ⏱️ 35 min
**Objectif :** établir les gradients d'une couche dense pour un mini-batch, et en tirer le coût de la passe arrière.
**Prérequis :** Ex 18.6 · ch. 16 (couche dense $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$, formes) · fiche §18.12.1, §18.12.5, §18.12.10 (encadré sur les gradients matriciels) · **Parcours :** M

Une couche dense reçoit un batch $\mathbf{X}$ de forme $(n, n_{in})$ et calcule $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^\top$ (le biais ajouté à chaque ligne), avec $\mathbf{W}$ de forme $(n_{in}, n_{out})$. La loss $L$ est un nombre, et l'on note $\boldsymbol{\Delta}$ la matrice $(n, n_{out})$ des $\Delta_{ik} = \partial L / \partial Z_{ik}$. Réponds dans ta copie.

1. Montre que $\partial L / \partial W_{jk} = \sum_i X_{ij}\,\Delta_{ik}$, c'est-à-dire $\partial L / \partial \mathbf{W} = \mathbf{X}^\top\boldsymbol{\Delta}$, et vérifie les formes.
2. Montre que $\partial L / \partial \mathbf{X} = \boldsymbol{\Delta}\mathbf{W}^\top$ et que $\partial L / \partial \mathbf{b}$ est la somme des lignes de $\boldsymbol{\Delta}$.
3. La couche est suivie d'une activation $\mathbf{A} = f(\mathbf{Z})$, case par case. On connaît $\mathbf{G} = \partial L / \partial \mathbf{A}$. Montre que $\boldsymbol{\Delta} = \mathbf{G} \odot f'(\mathbf{Z})$.
4. La loss du batch est la moyenne $L = \frac{1}{n}\sum_i \ell_i$ des losses des exemples. Où se trouve le facteur $\frac{1}{n}$ dans ces formules ? Montre que $\partial L / \partial \mathbf{W}$ est la moyenne des gradients de chaque exemple.
5. Le livre propose de moyenner les deltas d'un mini-batch, puis de mettre à jour les poids avec ces moyennes (§18.12.5). Pour un poids $w_{jk}$, compare $\frac{1}{n}\sum_i X_{ij}\,\Delta_{ik}$ et $\big(\frac{1}{n}\sum_i X_{ij}\big)\big(\frac{1}{n}\sum_i \Delta_{ik}\big)$. Donne un contre-exemple avec $n = 2$, et un cas où les deux coïncident.
6. Compte les multiplications de la passe avant ($\mathbf{X}\mathbf{W}$) et de la passe arrière ($\boldsymbol{\Delta}\mathbf{W}^\top$ et $\mathbf{X}^\top\boldsymbol{\Delta}$). Quel rapport ? Que penser de « la passe arrière prend à peu près le temps d'une passe avant » (§18.12.7) ?

### Ex 18.8 — Softmax et cross-entropy : le gradient vaut p − y ∂ ★★★ ⏱️ 35 min
**Objectif :** établir les gradients des cross-entropies par rapport aux logits, et montrer pourquoi on les préfère à l'erreur quadratique pour classer.
**Prérequis :** ch. 17 (softmax, sigmoïde, ∂ 17.7) · ch. 6 (cross-entropy) · Ex 18.6 · fiche, section sur les losses calculées sur les logits · **Parcours :** M

Réponds dans ta copie.

1. Pour un exemple de logits $\mathbf{z}$ et de **probabilités cibles** $\mathbf{q}$ (positives, de somme 1), $\ell = -\sum_k q_k \log p_k$ avec $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$. Montre que $\partial \ell / \partial z_j = p_j - q_j$. Où sers-toi de $\sum_k q_k = 1$ ? (∂ 17.7 l'a fait pour un one-hot.)
2. Pour un batch de $n$ exemples et la loss moyenne, montre que $\partial L / \partial \mathbf{Z} = (\mathbf{P} - \mathbf{Q})/n$, ligne par ligne.
3. Cas binaire : $a = \sigma(z)$ et $\ell = -[y \log a + (1 - y)\log(1 - a)]$. Montre que $\partial \ell / \partial z = \sigma(z) - y$. Retrouve ce résultat à partir de 1. avec deux logits $(z, 0)$.
4. Avec l'erreur quadratique sur la sortie sigmoïde, $\ell = \frac12(\sigma(z) - y)^2$, montre que $\partial \ell / \partial z = (\sigma(z) - y)\,\sigma(z)\,(1 - \sigma(z))$. Calcule les deux gradients, celui de 3. et celui-ci, pour $y = 0$ et $z = 5$ (une sortie sûre d'elle et fausse). Conclusion pour l'apprentissage ?
5. Montre que $-[y \log \sigma(z) + (1 - y)\log(1 - \sigma(z))] = \max(z, 0) - y\,z + \log(1 + e^{-|z|})$. Pourquoi cette forme ne déborde-t-elle jamais ? Sa valeur pour $z = 1000$ et $y = 0$ ?
6. Le ch. 6 mesurait la cross-entropy en bits ; ici, en nats. Quel facteur relie les deux ? Que change-t-il au gradient, et à l'entraînement ?

### Ex 18.9 — Fermi : rétropropagation contre force brute 🧮 ★★ ⏱️ 20 min
**Objectif :** chiffrer ce que coûterait un gradient par différences finies, et ce que fait gagner la rétropropagation.
**Prérequis :** ch. 16 (nombre de paramètres, coût d'une passe avant) · Ex 18.4 · fiche §18.2.1, §18.12.7 · **Parcours :** R, M

Un MLP 784-256-128-10 (biais compris) classe des images de MNIST. On compte une multiplication par poids et par exemple dans une passe avant (on néglige les biais et les activations), on prend une passe arrière complète égale à **deux** passes avant, et une machine qui fait $10^{12}$ multiplications par seconde.

a) Le nombre de paramètres $P$ du réseau.
b) Le nombre de multiplications d'une passe avant pour une image, en milliers (arrondi à l'entier).
c) Le nombre de passes avant nécessaires pour obtenir le gradient d'une image par différences finies **centrées**, un paramètre à la fois.
d) Le rapport entre le coût de ce gradient numérique et celui de la rétropropagation (passe avant et passe arrière), en milliers (arrondi à l'entier).
e) La durée d'une epoch de 60 000 images par rétropropagation, en millisecondes (arrondie à l'entier).
f) La durée de la même epoch avec le gradient par différences finies, en heures (1 décimale).

Dans ta copie : et la méthode « très lente » du livre (un essai = un poids perturbé et une passe avant) ? Pourquoi n'est-elle même pas comparable aux différences finies ?

<a id="reflexion"></a>

## 🗣️ 📄 Réflexion

### Ex 18.10 — Expliquer la rétropropagation sans équation 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer la rétropropagation à quelqu'un qui n'a jamais fait de dérivée.
**Prérequis :** fiche §18.12.1 à §18.12.3 · **Parcours :** R

Dans ta copie, en 5 lignes au plus et sans aucune formule, explique à un ami qui débute : ce que le réseau cherche à faire quand il se trompe, ce que « remonte » la rétropropagation, et pourquoi c'est beaucoup plus rapide que d'essayer de changer les poids un par un. Puis relis-toi : as-tu dit que « l'erreur remonte » ? (§18.12.2.)

### Ex 18.11 — Rumelhart, Hinton & Williams (1986) : l'article qui a relancé les réseaux 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a fait connaître la rétropropagation, et le relier au chapitre.
**Prérequis :** Ex 18.6 · fiche §18.12.9 · ch. 16 (brisure de symétrie) · **Parcours :** M

Lis D. Rumelhart, G. Hinton et R. Williams, « Learning representations by back-propagating errors », *Nature* 323, p. 533-536, 1986 (quatre pages, [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0) ; G. Hinton en donne une version en ligne depuis la [liste de ses publications](https://www.cs.toronto.edu/~hinton/backprop.html)). Réponds dans ta copie.

1. La mesure d'erreur totale $E$ (équation 3) : écris-la. Compare-la à la loss de la fiche.
2. Les poids sont-ils changés après chaque cas, ou après avoir accumulé $\partial E / \partial w$ sur tous les cas ? Laquelle des deux variantes ressemble aux mini-batches d'aujourd'hui ?
3. L'équation 9 ajoute à $-\varepsilon\,\partial E / \partial w$ un terme $\alpha\,\Delta w(t - 1)$. Quel nom porte cette idée aujourd'hui (ch. 19) ? Quelles valeurs de $\varepsilon$ et $\alpha$ les auteurs prennent-ils pour la détection de symétrie ?
4. La détection de symétrie (figure 1) : combien d'entrées, de neurones cachés, de balayages ? Explique pourquoi des poids dans le rapport $1 : 2 : 4$ de chaque côté du milieu permettent de distinguer un vecteur symétrique.
5. Les arbres généalogiques (figures 2 à 4) : combien de personnes, de relations, de triplets ? Combien servent à l'entraînement, combien au test ? Que représentent certains neurones cachés ?
6. Pourquoi les poids de départ sont-ils aléatoires et petits ? Relie cette remarque aux neurones jumeaux du ch. 16.
7. Que disent les auteurs des minima locaux ? Compare avec les plateaux et les points selles du §18.11.1.
8. La figure 5 : à quoi équivaut un réseau itératif qu'on fait tourner plusieurs pas ? Quelles deux difficultés faut-il régler ? (Tu les retrouveras au ch. 22.)
9. Les auteurs jugent-ils leur procédure plausible comme modèle de l'apprentissage dans le cerveau ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 18.E1 — Explique la rétropropagation en une minute 💼 ★★ ⏱️ 10 min
*Fiche §18.12.1 · parcours R*

« Pouvez-vous m'expliquer la rétropropagation en une minute ? Que calcule-t-elle exactement, et à quoi sert ce calcul ? »

### 18.E2 — Pourquoi la rétropropagation bat-elle les différences finies ? 💼 ★★ ⏱️ 10 min
*Fiche §18.2.1, §18.12.7 · parcours R*

« Pourquoi ne pas calculer le gradient par différences finies, poids par poids ? Combien coûte la rétropropagation par rapport à une passe avant ? »

### 18.E3 — Gradient évanescent : causes, symptômes, remèdes 💼 ★★ ⏱️ 10 min
*Fiche §18.12.4 · parcours R*

« Qu'est-ce que le problème du gradient évanescent ? Comment le reconnaître pendant un entraînement, et comment l'éviter ? »

### 18.E4 — Comment vérifies-tu une implémentation de gradient ? 💼 ★★ ⏱️ 10 min
*Fiche §18.12.10, section sur la vérification d'un gradient · parcours R*

« Vous avez écrit la passe arrière d'une couche maison. Comment vérifiez-vous qu'elle est juste ? Quels pièges ? »

### 18.E5 — Pourquoi calculer la loss sur les logits plutôt qu'après la softmax ? 💼 ★★ ⏱️ 10 min
*Fiche §18.6, §18.10, section sur les losses calculées sur les logits · parcours R*

« Dans PyTorch, pourquoi `CrossEntropyLoss` prend-elle les logits et non les probabilités ? Et pourquoi `BCEWithLogitsLoss` plutôt qu'une sigmoïde suivie de `BCELoss` ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch18_backprop/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie (`mylearn/nn/backward.py` et `mylearn/autograd_mini.py`). La partie 0 du notebook vérifie tes réponses courtes aux quiz, aux rappels et aux exercices ✏️ et 🧮 ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 18.12 | mse_loss : la loss et son gradient | 🔨 | ★ | 10 |
| 18.13 | La méthode très lente : combien d'essais pour progresser ? | 🔮 | ★★ | 20 |
| 18.14 | Cross-entropies calculées sur les logits (binaire et softmax) | 🔨 | ★★ | 30 |
| 18.15 | dense_backward : les trois gradients d'une couche | 🔨 | ★★ | 25 |
| 18.16 | gradient_check : vérifier un gradient par différences finies | 🔨 | ★★ | 25 |
| 18.17 | Chasse au bug : trois gradients faux démasqués | 🐛 | ★★ | 25 |
| 18.18 | mlp_backward : rétropropager à travers tout le réseau | 🔨 | ★★★ | 45 |
| 18.19 | Entraîner le réseau 2-4-4-1 à 37 poids sur les lunes | 🔨 | ★★★ | 45 |
| 18.20 | La même expérience avec MLPClassifier(solver='sgd') | 📦 | ★★ | 20 |
| 18.21 | Reproduire les figures 18.49 à 18.57 (η = 0,5 ; 0,05 ; 0,01) | 🎨 | ★★ | 30 |
| 18.22 | Lire des courbes de loss : divergence, lenteur, plateau, bug | 📈 | ★★ | 20 |
| 18.23 | Balayage du learning rate sur cinq graines | 🔬 | ★★★ | 40 |
| 18.24 | Neurones saturés et ReLU mortes pendant l'entraînement | 🔬 | ★★★ | 45 |
| 18.25 | Écrire un test pytest de gradient pour une couche | 🛠️ | ★★ | 25 |
| 18.26 | Micro-autograd : la classe Value | 🔨 | ★★★ | 90 |
| 18.27 | À la main, Value, mlp_backward, différences finies : quatre gradients identiques ? | 🔮 | ★★ | 20 |
| 18.28 | MiniMLP : entraîner un petit réseau fait de Value | 🔨 | ★★★ | 45 |
| 18.29 | Défi MNIST : au moins 97 % avec ton MLP NumPy | 🏆 | ★★★★ | 120 |
