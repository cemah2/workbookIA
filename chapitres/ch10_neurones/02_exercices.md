# 10 · Neurones — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch10_neurones/06_mes_reponses.md` (créée par `python tools/start_chapter.py 10`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz** (sauf Q3), des **rappels** et des exercices ✏️ 10.1, 10.2, 10.5 et 10.6 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », le quiz Q3, l'exercice ✏️ 10.3, les preuves ∂ 10.4 et 10.7, la réflexion (🗣️ 🧮 ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), des lettres collées pour plusieurs choix (`"AC"`) ou pour un ordre (`"DCBA"`), éventuellement séparées par des virgules (`"D, C, B, A"`), une liste pour plusieurs nombres (`[2, 5]`), une liste de listes pour une matrice (`[[1, 2], [3, 4]]`, ligne par ligne).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 🧮 ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 10.Q1 — Neurones artificiels : où sont-ils indispensables, où s'en passe-t-on ? 🧠 ⏱️ 3 min
*Fiche §10.1 · livre §10.1 · parcours R*

a) Parmi ces méthodes, lesquelles n'utilisent **aucun** neurone artificiel ? (A) le k-means (ch. 7) ; (B) la régression linéaire par moindres carrés (ch. 9) ; (C) un réseau convolutif qui classe des photos ; (D) un arbre de décision (ch. 13) ; (E) un grand modèle de langage.
b) Vrai ou faux : une méthode qui obtient sa solution par une formule explicite (une « forme fermée ») a besoin de neurones artificiels.
c) Vrai ou faux : un perceptron seul est un réseau de neurones profond.

### 10.Q2 — Le neurone biologique en quatre étapes 🧠 ⏱️ 3 min
*Fiche §10.2 · livre §10.2 · parcours R*

a) Range dans l'ordre ces quatre événements du traitement de l'information par un neurone : (A) le total est comparé à un seuil ; (B) des neurotransmetteurs se fixent sur des récepteurs du neurone ; (C) les signaux électriques arrivés pendant un court intervalle s'additionnent ; (D) le neurone libère à son tour des neurotransmetteurs vers d'autres neurones.
b) Vrai ou faux : tous les signaux électriques qui arrivent au corps du neurone le poussent vers la décharge.
c) Vrai ou faux : en général, deux neurones « connectés » se touchent physiquement.

### 10.Q3 — Connectome et émulation du cerveau : vrai ou faux justifié 🧠 ⏱️ 4 min
*Fiche §10.2 (encadré 🕰️) · livre §10.2 · parcours R*

Vrai ou faux ? Justifie chaque réponse en une phrase (dans ta copie).

a) Le connectome d'une personne est la carte de toutes les connexions entre ses neurones.
b) Deux personnes en bonne santé ont exactement le même connectome.
c) Simuler sur des puces des milliards de neurones simplifiés a produit, à ce jour, une intelligence générale comparable à la nôtre.
d) Pour certains chercheurs, l'intelligence ne peut pas émerger sans un corps doté de sens.
e) Le connectome complet du cerveau d'une mouche adulte a été publié en 2024.

### 10.Q4 — Neurone, unité, « cerveau électronique » : bien nommer les choses 🧠 ⏱️ 3 min
*Fiche §10.3 · livre §10.3 · parcours R*

a) Pourquoi beaucoup d'auteurs préfèrent-ils parler d'**unité** plutôt que de neurone ? (A) parce qu'un neurone artificiel est une abstraction très simplifiée d'un vrai neurone ; (B) parce qu'une unité calcule plus vite ; (C) parce qu'une unité n'a pas de poids ; (D) parce que le mot « neurone » est réservé aux réseaux impulsionnels.
b) Vrai ou faux : un réseau de neurones est un cerveau électronique en miniature.
c) Que garde le neurone artificiel du vrai neurone ? (A) la chimie des neurotransmetteurs ; (B) l'addition de signaux suivie d'une décision ; (C) l'instant précis de chaque impulsion ; (D) la forme des dendrites.

### 10.Q5 — McCulloch et Pitts (1943) : ce qu'ils ont démontré 🧠 ⏱️ 3 min
*Fiche §10.3.1 · livre §10.3.1 · parcours R*

a) Qu'ont démontré McCulloch et Pitts ? (A) qu'un réseau de neurones peut apprendre n'importe quelle fonction à partir d'exemples ; (B) que, sous certaines conditions, toute expression logique peut être réalisée par un réseau de neurones formels ; (C) que le cerveau est un ordinateur numérique ; (D) qu'un neurone seul suffit pour tout calculer.
b) Vrai ou faux : les neurones formels de 1943 apprenaient leurs poids à partir d'exemples.
c) Un neurone de McCulloch et Pitts a deux entrées binaires et un seuil de 2 : il répond 1 si la somme de ses entrées vaut au moins 2, et 0 sinon. Quelle porte logique calcule-t-il ? (A) OR ; (B) AND ; (C) XOR ; (D) NAND.

### 10.Q6 — Anatomie d'un perceptron 🧠 ⏱️ 3 min
*Fiche §10.3.1 · livre §10.3.1 · parcours R*

a) Dans quel ordre un perceptron calcule-t-il sa sortie ? (A) chaque entrée est multipliée par son poids, les produits sont additionnés, puis la somme est comparée à 0 ; (B) les entrées sont additionnées, puis la somme est multipliée par un poids unique ; (C) chaque entrée est comparée au seuil, puis les entrées votent ; (D) les entrées sont multipliées entre elles.
b) Que sort le perceptron du livre quand sa somme pondérée vaut exactement 0 ?
c) Vrai ou faux : dans certaines versions, le perceptron sort 1 et 0 au lieu de $+1$ et $-1$.

### 10.Q7 — Mark I, Minsky-Papert, renaissance : la chronologie 🧠 ⏱️ 3 min
*Fiche §10.3.2 · livre §10.3.2 · parcours R*

a) Range ces événements dans l'ordre chronologique : (A) le livre *Perceptrons* de Minsky et Papert ; (B) l'article de McCulloch et Pitts ; (C) l'article de Rumelhart, Hinton et Williams sur la rétropropagation ; (D) le premier rapport de Rosenblatt sur le perceptron ; (E) la démonstration publique du Mark I.
b) Combien de cellules photoélectriques avait le Mark I ?
c) Quelles données un perceptron peut-il apprendre à séparer parfaitement ? (A) deux classes qu'un hyperplan sépare (une droite, en deux dimensions) ; (B) n'importe quelles données à deux classes, avec assez d'époques ; (C) seulement des images ; (D) n'importe quelles données sans bruit.
d) Vrai ou faux : Minsky et Papert ont démontré qu'aucun réseau de neurones, même à plusieurs couches, ne peut calculer XOR.

### 10.Q8 — Du perceptron au neurone moderne : les deux changements 🧠 ⏱️ 3 min
*Fiche §10.3.3 · livre §10.3.3 · parcours R*

a) Quels sont les deux changements qui font passer du perceptron au neurone moderne ? (A) un biais ajouté à la somme pondérée ; (B) des poids entiers ; (C) une fonction d'activation à la place du seuil ; (D) une entrée de plus pour chaque couche cachée ; (E) la suppression des poids.
b) Combien de nombres ajustables (poids et biais) a un neurone moderne à 4 entrées ?
c) Vrai ou faux : avec l'astuce du biais, le biais devient le poids d'une entrée constante égale à 1.
d) Vrai ou faux : sans biais, la frontière de décision d'un neurone passe toujours par l'origine.

### 10.Q9 — Lire un schéma de réseau : poids implicites et convention AD/DA 🧠 ⏱️ 3 min
*Fiche §10.3.3 · livre §10.3.3, §10.4 · parcours R*

a) Vrai ou faux : sur un schéma de réseau où aucun poids n'est dessiné, les entrées des neurones ne sont pas pondérées.
b) Trois neurones A, B et C envoient chacun leur sortie à deux neurones D et E. Combien de poids relient les deux couches ?
c) Combien de nombres ajustables en tout pour D et E, biais compris ?
d) Dans la convention du livre, que désigne le poids BE ? (A) le poids qui multiplie la sortie de B avant son utilisation par E ; (B) le poids qui multiplie la sortie de E avant son utilisation par B ; (C) le biais de E ; (D) le produit des sorties de B et de E.
e) Vrai ou faux : dans PyTorch, `nn.Linear` range ses poids dans une matrice de forme `(n_out, n_in)`, une ligne par neurone.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 10.R1 — Ch. 9 : régularisation L2, que deviennent les poids ? 🔁 ★ ⏱️ 5 min
*Ch. 9 (§9.5, régularisation) · parcours R*

Un modèle linéaire à une feature, sans intercept, prédit $\hat{y} = w\,x$. La régression Ridge (ch. 9) choisit $w$ qui minimise $\sum_i (y_i - w x_i)^2 + \lambda w^2$, dont la solution est $w = \dfrac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda}$. Sur des données où $\sum_i x_i^2 = 10$ et $\sum_i x_i y_i = 20$ :

a) $w$ pour $\lambda = 0$.
b) $w$ pour $\lambda = 10$.
c) Vrai ou faux : quand $\lambda$ grandit sans limite, $w$ tend vers 0.
d) Dans un neurone, à quoi applique-t-on d'habitude la pénalité L2 ? (A) aux poids, pas au biais ; (B) au biais seulement ; (C) aux entrées ; (D) à la sortie du neurone.
e) Pourquoi ne pénalise-t-on pas le biais ? (dans ta copie)

### 10.R2 — Ch. 7 : la frontière du centroïde le plus proche, w·x + b = 0 🔁 ★ ⏱️ 5 min
*Ch. 7 (fiche §7.5, « Avec des labels : le centroïde le plus proche ») · parcours R, M*

Deux classes ont pour centroïdes $\boldsymbol{\mu}_+ = (2, 3)$ (classe $+1$) et $\boldsymbol{\mu}_- = (0, 1)$ (classe $-1$). Le centroïde le plus proche range $\mathbf{x}$ dans la classe $+1$ quand $\lVert \mathbf{x} - \boldsymbol{\mu}_+ \rVert^2 < \lVert \mathbf{x} - \boldsymbol{\mu}_- \rVert^2$. En développant les deux carrés, montre que cette condition s'écrit $\mathbf{w}\cdot\mathbf{x} + b > 0$, avec $\mathbf{w} = \boldsymbol{\mu}_+ - \boldsymbol{\mu}_-$ et un biais $b$ que tu exprimeras avec $\lVert \boldsymbol{\mu}_+ \rVert^2$ et $\lVert \boldsymbol{\mu}_- \rVert^2$ (démarche dans ta copie).

a) $w_1$.
b) $w_2$.
c) $b$.
d) La classe prédite ($+1$ ou $-1$) pour $\mathbf{x} = (3, 2)$.
e) Vrai ou faux : le centroïde le plus proche à deux classes prend la même décision qu'un perceptron dont les poids seraient calculés par une formule à partir des centroïdes, au lieu d'être ajustés par la règle du perceptron.

### 10.R3 — Ch. 3 : matrice de confusion et accuracy d'un classifieur binaire 🔁 ★ ⏱️ 5 min
*Ch. 3 (§3.7.2 à §3.7.7) · parcours R*

Un perceptron classe 10 exemples ; la classe positive est $+1$.
- labels vrais : $+1, +1, +1, +1, -1, -1, -1, -1, -1, -1$ ;
- prédictions : $+1, +1, -1, +1, -1, +1, -1, -1, -1, +1$.

a) Le nombre de vrais positifs (TP).
b) Le nombre de faux positifs (FP).
c) Le nombre de faux négatifs (FN).
d) Le nombre de vrais négatifs (TN).
e) L'accuracy.
f) Le recall (*rappel*), avec 2 décimales.
g) La precision (*précision*).

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 10.1 — Sortie d'un perceptron à quatre entrées, avec et sans biais ✏️ ★ ⏱️ 10 min
**Objectif :** calculer à la main la sortie d'un perceptron, avec la convention du livre pour une somme nulle, puis avec un biais.
**Prérequis :** 0B (produit scalaire) · fiche §10.3.1, §10.3.3 · **Parcours :** R, M

Un perceptron à quatre entrées a pour poids $\mathbf{w} = (0{,}5;\ -1;\ 2;\ 0{,}25)$. Il sort $+1$ si $z > 0$, $-1$ sinon.

a) La somme pondérée $z$ pour $\mathbf{x}^{(1)} = (2;\ 1;\ 0{,}5;\ -4)$.
b) La sortie pour $\mathbf{x}^{(1)}$.
c) La somme pondérée pour $\mathbf{x}^{(2)} = (1;\ 3;\ 1;\ 4)$.
d) La sortie pour $\mathbf{x}^{(2)}$.

On ajoute maintenant un biais $b = -0{,}75$ : $z = \mathbf{w}\cdot\mathbf{x} + b$.

e) La sortie pour $\mathbf{x}^{(2)}$.
f) La sortie pour $\mathbf{x}^{(3)} = (-2;\ 0;\ 0{,}75;\ 2)$.
g) Combien des trois entrées $\mathbf{x}^{(1)}$, $\mathbf{x}^{(2)}$, $\mathbf{x}^{(3)}$ donnent $+1$ avec ce biais ?
h) Dans la version 0/1 du perceptron (sortie 1 si $z > 0$, 0 sinon), que sort-il pour $\mathbf{x}^{(1)}$, **sans** biais ?
i) La plus petite valeur **entière** du biais $b$ pour laquelle les trois entrées donnent $+1$.

### Ex 10.2 — L'astuce du biais : même neurone, une entrée de plus ✏️ ★ ⏱️ 10 min
**Objectif :** réécrire un neurone avec l'astuce du biais, pour un exemple puis pour un lot, et voir ce que le biais permet.
**Prérequis :** Ex 10.1 · fiche §10.3.3 · **Parcours :** M

Un neurone à deux entrées a pour poids $\mathbf{w} = (2, -1)$ et pour biais $b = 0{,}5$ ; son activation est le seuil du perceptron ($+1$ si $z > 0$, $-1$ sinon).

a) Le vecteur augmenté $\tilde{\mathbf{w}} = (b, w_1, w_2)$, sous forme de liste.
b) Pour $\mathbf{x} = (1, 3)$, écris $\tilde{\mathbf{x}}$ et calcule $z = \tilde{\mathbf{w}}\cdot\tilde{\mathbf{x}}$ (la partie 0 vérifie $z$).
c) Le lot $\mathbf{X}$ a trois lignes : $(1, 3)$, $(0, 0)$ et $(2, 1)$. Écris $\tilde{\mathbf{X}}$ (une colonne de 1 en tête), puis les trois sommes pondérées $\tilde{\mathbf{X}}\tilde{\mathbf{w}}$, sous forme de liste (la partie 0 vérifie les trois sommes).
d) Les trois sorties du neurone, sous forme de liste.
e) Vrai ou faux : l'astuce du biais change la frontière de décision du neurone.
f) Une couche de 5 neurones reçoit 3 entrées. Avec l'astuce du biais, combien de nombres sa matrice de poids augmentée $\tilde{\mathbf{W}}$ contient-elle ?
g) Vrai ou faux : **sans** biais, un perceptron à deux entrées 0/1, version 0/1 (sortie 1 si $z > 0$), peut calculer AND.
h) Justifie g) (dans ta copie).

### Ex 10.3 — Portes logiques à la main : AND, OR, NOT, NAND ✏️ ★★ ⏱️ 20 min
**Objectif :** trouver des poids et des biais qui réalisent des portes logiques, et comprendre pourquoi les solutions ne sont pas uniques.
**Prérequis :** Ex 10.1 · fiche §10.3.1 (encadré 🧮 sur la séparabilité) · **Parcours :** R, M

Les entrées valent 0 ou 1 ; le neurone sort 1 si $z = \mathbf{w}\cdot\mathbf{x} + b > 0$, et 0 sinon. La fiche donne AND : $\mathbf{w} = (1, 1)$, $b = -1{,}5$. Réponds dans ta copie, en vérifiant chaque porte sur toutes ses entrées dans un tableau.

a) Trouve des poids et un biais pour **OR** (sortie 1 si au moins une entrée vaut 1).
b) **NOT**, à une seule entrée (sortie 1 si $x = 0$).
c) **NAND** (le contraire de AND). Montre qu'on l'obtient à partir de AND en changeant le signe de $\mathbf{w}$ et de $b$. Pourquoi cela marche-t-il ici ? Que se passerait-il si l'une des quatre sommes valait exactement 0 ?
d) **NOR** (le contraire de OR).
e) La porte « **majorité** » à trois entrées : sortie 1 si au moins deux des trois entrées valent 1.
f) Montre que multiplier $\mathbf{w}$ et $b$ par un même nombre $c > 0$ ne change aucune sortie. Que se passe-t-il avec $c < 0$ ?
g) Dessine les quatre points de OR et la droite de ton neurone.

### Ex 10.4 — Pourquoi un seul perceptron ne peut pas calculer XOR ∂ ★★ ⏱️ 25 min
**Objectif :** démontrer, de deux façons, qu'aucun perceptron ne calcule XOR, puis contourner l'obstacle en ajoutant une feature.
**Prérequis :** Ex 10.3 · fiche §10.3.1 (encadré 🧮 sur la séparabilité) · **Parcours :** R, M

Version 0/1 du perceptron : la sortie vaut 1 si $s(\mathbf{x}) = w_1 x_1 + w_2 x_2 + b > 0$, et 0 sinon. XOR vaut 1 pour $(0, 1)$ et $(1, 0)$, et 0 pour $(0, 0)$ et $(1, 1)$. Suppose qu'un tel perceptron calcule XOR. Réponds dans ta copie.

a) Écris les quatre inégalités que doivent vérifier $w_1$, $w_2$ et $b$.
b) Additionne les deux inégalités des entrées de sortie 1, compare avec les deux autres, et conclus.
c) Une preuve géométrique. Exprime $s(0{,}5 ; 0{,}5)$ en fonction de $s(0, 1)$ et de $s(1, 0)$, puis en fonction de $s(0, 0)$ et de $s(1, 1)$. Pourquoi est-ce contradictoire ? Quelle propriété de $s$ as-tu utilisée ?
d) Un perceptron peut-il calculer XNOR (le contraire de XOR) ? Justifie.
e) On ajoute une troisième entrée, le produit $x_3 = x_1 x_2$. Trouve $w_1$, $w_2$, $w_3$ et $b$ qui calculent XOR avec ces trois entrées. Quels chapitres font de ce genre d'ajout une méthode générale ?
f) Combien existe-t-il de fonctions logiques à deux entrées (de tables de vérité différentes) ? Lesquelles un perceptron ne peut-il pas calculer ?

### Ex 10.5 — XOR en deux couches : câbler et nommer les poids ✏️ ★★ ⏱️ 25 min
**Objectif :** calculer un petit réseau couche par couche, nommer ses poids dans la convention du livre et les ranger dans les matrices de mylearn et de PyTorch.
**Prérequis :** Ex 10.4 · fiche §10.3.3 (encadré 🧮 sur les matrices) · **Parcours :** M

Un réseau a deux entrées A et B, deux neurones cachés C et D, et un neurone de sortie E. Chaque neurone sort 1 si sa somme pondérée, biais compris, est $> 0$, et 0 sinon. Les poids suivent la convention du livre (AC : de A vers C) :
- neurone C : $AC = 1$, $BC = 1$, biais $b_C = -0{,}5$ ;
- neurone D : $AD = -1$, $BD = -1$, biais $b_D = 1{,}5$ ;
- neurone E : $CE = 1$, $DE = 1$, biais $b_E = -1{,}5$.

a) Les sorties de C pour les entrées $(0, 0)$, $(0, 1)$, $(1, 0)$, $(1, 1)$, dans cet ordre (une liste).
b) Les sorties de D, dans le même ordre.
c) Les sorties de E, dans le même ordre.
d) Quelles portes logiques calculent C, D et E ? (dans ta copie)
e) La matrice des poids de la couche cachée dans la convention de mylearn : $\mathbf{W}$ de forme $(n_{\text{in}}, n_{\text{out}})$, lignes A et B, colonnes C et D (une liste de listes).
f) La même matrice dans la convention de PyTorch : `weight` de forme $(n_{\text{out}}, n_{\text{in}})$.
g) Le nombre total de nombres ajustables du réseau (poids et biais).
h) Vrai ou faux : si l'on remplace les trois seuils par l'identité ($f(z) = z$), le réseau calcule une fonction affine de ses entrées, de la forme $v_1 x_1 + v_2 x_2 + c$.
i) Dessine le réseau, puis explique pourquoi h) montre qu'un réseau à plusieurs couches a besoin d'une activation non linéaire (dans ta copie).

### Ex 10.6 — Une époque de la règle du perceptron à la main ✏️ ★★ ⏱️ 25 min
**Objectif :** appliquer à la main la règle d'apprentissage du perceptron, exemple par exemple, et voir ce que la règle appelle une erreur.
**Prérequis :** Ex 10.3 · fiche §10.3.1 (encadré 🧮 sur la règle d'apprentissage) · **Parcours :** M

On entraîne un perceptron avec biais sur la porte **OR**, avec la règle de la fiche : la sortie 0 est codée $y = -1$ et la sortie 1, $y = +1$ ; on part de $\mathbf{w} = (0, 0)$ et $b = 0$, avec $\eta = 1$ ; les exemples sont présentés dans l'ordre $(0, 0)$, $(0, 1)$, $(1, 0)$, $(1, 1)$. Un exemple est mal classé quand $y(\mathbf{w}\cdot\mathbf{x} + b) \le 0$ ; il déclenche alors $\mathbf{w} \leftarrow \mathbf{w} + \eta\, y\, \mathbf{x}$ et $b \leftarrow b + \eta\, y$. Tiens un tableau : exemple, $z$, erreur ou non, $\mathbf{w}$ et $b$ après l'exemple.

a) $b$ après le premier exemple.
b) $\mathbf{w}$ après le deuxième exemple (une liste).
c) $\mathbf{w}$ à la fin de la première époque.
d) $b$ à la fin de la première époque.
e) Le nombre de corrections pendant la première époque.
f) $b$ à la fin de la deuxième époque.
g) Vrai ou faux : à la fin de la deuxième époque, la règle de **prédiction** ($+1$ si $z > 0$, $-1$ sinon) classe correctement les quatre entrées.
h) Vrai ou faux : la règle d'**apprentissage** fait pourtant au moins une correction pendant la troisième époque.
i) Explique h), puis continue jusqu'à la convergence : combien d'époques en tout (y compris l'époque sans correction), avec quels poids finals ? (dans ta copie ; tu pourras vérifier avec ta classe `Perceptron` en 10.21)

### Ex 10.7 — Le théorème de convergence du perceptron, guidé pas à pas ∂ ★★★ ⏱️ 45 min
**Objectif :** démontrer que le perceptron converge sur des données séparables avec une marge, et comprendre ce que dit la borne $(R/\gamma)^2$.
**Prérequis :** Ex 10.6 · 0B (produit scalaire, norme, inégalité de Cauchy-Schwarz) · fiche §10.3.2 (encadré 🧮 sur la convergence) · **Parcours :** M

Hypothèses : des exemples $(\mathbf{x}_i, y_i)$ avec $y_i \in \{-1, +1\}$ et $\lVert \mathbf{x}_i \rVert \le R$ ; un vecteur $\mathbf{u}$ de norme 1 tel que $y_i\, \mathbf{u}\cdot\mathbf{x}_i \ge \gamma > 0$ pour tout $i$. On étudie la règle **sans biais**, avec $\eta = 1$, partie de $\mathbf{w}_0 = \mathbf{0}$ ; $\mathbf{w}_k$ désigne les poids après la $k$-ième correction, faite sur un exemple mal classé $(\mathbf{x}, y)$ : $\mathbf{w}_k = \mathbf{w}_{k-1} + y\,\mathbf{x}$, avec $y\, \mathbf{w}_{k-1}\cdot\mathbf{x} \le 0$. Réponds dans ta copie.

a) Que signifie géométriquement l'hypothèse sur $\mathbf{u}$ ? Fais un dessin en deux dimensions, avec $R$ et $\gamma$.
b) Montre que $\mathbf{u}\cdot\mathbf{w}_k \ge \mathbf{u}\cdot\mathbf{w}_{k-1} + \gamma$, puis que $\mathbf{u}\cdot\mathbf{w}_k \ge k\gamma$.
c) Développe $\lVert \mathbf{w}_{k-1} + y\,\mathbf{x} \rVert^2$ et montre que $\lVert \mathbf{w}_k \rVert^2 \le \lVert \mathbf{w}_{k-1} \rVert^2 + R^2$. Où utilises-tu le fait que l'exemple était mal classé ? En déduire $\lVert \mathbf{w}_k \rVert^2 \le k R^2$.
d) Avec l'inégalité de Cauchy-Schwarz, compare $\mathbf{u}\cdot\mathbf{w}_k$ et $\lVert \mathbf{w}_k \rVert$, puis conclus : $k \le (R/\gamma)^2$.
e) Refais la preuve avec un pas $\eta > 0$ quelconque : la borne change-t-elle ? Pourquoi ?
f) Avec un biais : comment appliquer le théorème aux vecteurs $\tilde{\mathbf{x}}_i = (1, \mathbf{x}_i)$ ? Que deviennent $R$ et la marge ?
g) La borne dépend-elle du nombre d'exemples ? De la dimension ? De l'ordre des exemples ?
h) Application : des images de 20 × 20 pixels, chacun entre 0 et 1, séparables **sans biais** avec une marge de 0,5. Majore $R$, puis le nombre de corrections.
i) Que dit le théorème quand les données ne sont pas séparables ?

<a id="reflexion"></a>

## 🗣️ 🧮 ⚖️ 📄 Réflexion

### Ex 10.8 — Pourquoi un neurone artificiel n'est pas un neurone 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un non-spécialiste ce qu'un neurone artificiel garde d'un vrai neurone, et tout ce qu'il laisse de côté.
**Prérequis :** fiche §10.2, §10.3 · **Parcours :** R

Un ami a lu qu'« une IA, c'est un cerveau électronique fait de neurones ». Réponds-lui à voix haute, en **deux minutes au plus**. Contraintes :
- une ressemblance et au moins trois différences entre un neurone biologique et un neurone artificiel ;
- les mots « unité », « seuil » et « métaphore » ;
- aucune formule.

Enregistre-toi ou parle devant quelqu'un, puis compare avec la réponse modèle de `05_solutions.md`.

### Ex 10.9 — Fermi : cerveau humain contre grands modèles 🧮 ★★ ⏱️ 20 min
**Objectif :** estimer des ordres de grandeur, et voir ce qu'une comparaison entre synapses et paramètres dit, et ne dit pas.
**Prérequis :** fiche §10.2 (encadré 🧮 sur les ordres de grandeur) · **Parcours :** M

Données : un cerveau humain compte environ $8{,}6 \times 10^{10}$ neurones et entre $1 \times 10^{14}$ et $5 \times 10^{14}$ synapses, et consomme environ 20 W. Un grand modèle de langage ouvert récent compte environ $10^{12}$ paramètres, dont $3{,}2 \times 10^{10}$ seulement servent à produire chaque token, le morceau de mot qu'il génère à chaque pas (un modèle « à mélange d'experts »). Un GPU de centre de données a 80 Go de mémoire et consomme environ 700 W. Réponds dans ta copie, avec un ou deux chiffres significatifs.

a) Combien de synapses par neurone, en moyenne (une fourchette) ?
b) Combien de synapses par paramètre du modèle (une fourchette) ?
c) La mémoire nécessaire pour stocker les $10^{12}$ paramètres, à 2 octets chacun. Combien de GPU faut-il, au minimum, rien que pour les contenir ?
d) La puissance de ces GPU, comparée à celle d'un cerveau : quel rapport ? Qu'as-tu oublié de compter ?
e) L'énergie consommée par un cerveau en une journée, en kWh.
f) Le modèle n'utilise que $3{,}2 \times 10^{10}$ paramètres par token. Compare ce nombre au nombre de neurones du cortex (environ $1{,}6 \times 10^{10}$).
g) Donne trois raisons pour lesquelles « le modèle a 100 fois moins de paramètres que le cerveau n'a de synapses » ne permet pas de dire qu'il est 100 fois moins intelligent, ni qu'il le deviendra en grandissant.

### Ex 10.10 — « Cerveaux électroniques » : hype, hivers de l'IA et responsabilité ⚖️ ★★ ⏱️ 20 min
**Objectif :** analyser l'écart entre ce qu'une démonstration montre et ce qu'on en annonce, et ses conséquences pour un domaine de recherche.
**Prérequis :** fiche §10.3, §10.3.2 (encadré 🕰️ sur l'histoire) · **Parcours :** aucun (réflexion conseillée à tous)

En juillet 1958, la Marine américaine présente à la presse le perceptron, simulé sur un ordinateur IBM 704 : après une cinquantaine d'essais, il distingue des cartes marquées à gauche de cartes marquées à droite. Le *New York Times* du 8 juillet rapporte que la Marine attend de cet « embryon » d'ordinateur qu'il puisse un jour « walk, talk, see, write, reproduce itself and be conscious of its existence ». Une dizaine d'années plus tard, les crédits des perceptrons se tarissent ; en 1973, au Royaume-Uni, le rapport Lighthill juge sévèrement l'ensemble de l'IA, dont les financements publics sont alors fortement réduits. Réponds dans ta copie.

1. Qu'est-ce qui a été **démontré** en 1958, et qu'est-ce qui a été **annoncé** ? Mesure l'écart.
2. Qui est responsable de cet écart : le chercheur, l'organisme qui le finance, le journaliste ? Peut-on parler de responsabilité partagée ?
3. Explique le mécanisme qui mène d'une promesse excessive à un « hiver de l'IA ».
4. Trouve une annonce récente sur l'IA qui te semble excessive, et une qui te semble mesurée. Quels critères t'ont permis de les distinguer ?
5. Écris trois règles que tu suivras pour présenter les résultats de tes modèles à des non-spécialistes.

### Ex 10.11 — Rosenblatt (1958) : le perceptron dans le texte 📄 ★★ ⏱️ 30 min
**Objectif :** lire l'article fondateur du perceptron et le comparer au perceptron du chapitre.
**Prérequis :** Ex 10.6 · fiche §10.3.1, §10.3.2 · **Parcours :** aucun (lecture conseillée à tous)

Lis l'introduction, la description de l'organisation du perceptron et les conclusions de F. Rosenblatt, « The perceptron: a probabilistic model for information storage and organization in the brain », *Psychological Review* 65 (6), 386-408, 1958 ([lien DOI](https://doi.org/10.1037/h0042519) ; l'article est aussi reproduit dans de nombreux supports de cours). Réponds dans ta copie.

1. Quelles sont les trois questions fondamentales posées dans l'introduction ? Auxquelles l'article s'attaque-t-il ?
2. L'auteur oppose deux positions sur la mémoire. Lesquelles ? Laquelle adopte-t-il, et où l'information est-elle alors stockée ?
3. Décris l'organisation d'un photo-perceptron : les points sensoriels, les unités d'association, les réponses. Quelles connexions sont tirées au hasard ? Comment les réponses deviennent-elles mutuellement exclusives ?
4. L'article compare trois systèmes de renforcement (α, β et γ). Qu'est-ce qui est renforcé, et quand ? Est-ce la règle de correction des erreurs de la fiche ?
5. Quelle limite du perceptron Rosenblatt reconnaît-il lui-même ? Rapproche-la des critiques de Minsky et Papert.
6. Sur quelle machine ses expériences ont-elles été simulées ?
7. Qu'est-ce qui distingue ce texte d'un article de machine learning d'aujourd'hui (méthode, évaluation, vocabulaire) ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 10.E1 — Qu'est-ce qu'un perceptron, et quelle est sa limite fondamentale ? 💼 ★★ ⏱️ 10 min
*Fiche §10.3.1, §10.3.2 · prérequis 10.4 · parcours R*

« Expliquez-moi ce qu'est un perceptron. Quelle est sa limite fondamentale, et comment la dépasse-t-on ? »

### 10.E2 — À quoi sert le biais d'un neurone ? 💼 ★★ ⏱️ 10 min
*Fiche §10.3.3 · prérequis 10.2 · parcours R*

« À quoi sert le biais d'un neurone ? Que se passerait-il sans lui ? »

### 10.E3 — Pourquoi remplacer le seuil par une activation dérivable ? 💼 ★★ ⏱️ 10 min
*Fiche §10.3.3 (encadré 🕰️ sur les activations) · parcours R*

« Pourquoi les réseaux de neurones modernes n'utilisent-ils plus la fonction seuil du perceptron ? Que faut-il à une bonne fonction d'activation ? »

### 10.E4 — Un réseau de neurones ressemble-t-il au cerveau ? 💼 ★★ ⏱️ 10 min
*Fiche §10.2, §10.3 · prérequis 10.8 · parcours R*

« Un réseau de neurones ressemble-t-il au cerveau ? Jusqu'où va l'analogie ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch10_neurones/03_notebook.ipynb`) ; ceux marqués 🔨 et accompagnés de « mylearn » complètent ta librairie `mylearn/perceptron.py`. La partie 0 du notebook vérifie tes réponses courtes aux quiz, aux rappels et aux exercices ✏️ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 10.12 | sign_step et add_bias_column | 🔨 | ★ | 10 |
| 10.13 | AND, OR, XOR : le perceptron va-t-il converger ? | 🔮 | ★ | 10 |
| 10.14 | neuron_forward : un neurone appliqué à tout un lot | 🔨 | ★★ | 15 |
| 10.15 | Des noms de poids (AD, BE…) à la matrice W | 🔨 | ★★ | 20 |
| 10.16 | XOR avec trois neurones câblés à la main | 🔨 | ★★ | 25 |
| 10.17 | Le learning rate change-t-il un perceptron qui part de zéro ? | 🔮 | ★★ | 15 |
| 10.18 | Le perceptron de scikit-learn sur portes logiques et manchots | 📦 | ★★ | 20 |
| 10.19 | Erreurs par époque : séparable ou pas ? | 📈 | ★★ | 20 |
| 10.20 | Docstring NumPy et doctest pour neuron_forward | 🛠️ | ★★ | 20 |
| 10.21 | La classe Perceptron et sa règle d'apprentissage | 🔨 | ★★★ | 45 |
| 10.22 | Perceptron piégé : quatre bugs classiques | 🐛 | ★★★ | 30 |
| 10.23 | Marge et vitesse de convergence : la borne (R/γ)² à l'épreuve | 🔬 | ★★★ | 40 |
| 10.24 | Trois espèces de manchots avec des perceptrons en un-contre-tous | 🔬 | ★★★ | 35 |
| 10.25 | Défi « Mark I » : un perceptron sur des chiffres de 20 × 20 pixels | 🏆 | ★★★ | 60 |
