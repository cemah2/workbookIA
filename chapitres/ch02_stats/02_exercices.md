# 2 · Hasard et statistiques de base — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch02_stats/06_mes_reponses.md` (créée par `python tools/start_chapter.py 2`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses des exercices ✏️ se vérifient dans la **partie 0** du notebook (`wb.check`) ; les quiz, les rappels, les exercices ∂ 🧮 🗣️ ⚖️ 📄 et les questions d'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🧮 🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 ou 4 minutes chacun. Réponds vite, puis vérifie dans `05_solutions.md`.

### 2.Q1 — Moyenne, médiane, mode : laquelle résiste aux valeurs extrêmes ? 🧠 ⏱️ 3 min
*Fiche §2.1, §2.2 · livre §2.1, §2.2 · parcours R*

Voici sept valeurs : 3, 5, 5, 6, 8, 9, 40.
1. Calcule la moyenne (2 décimales), la médiane et le mode.
2. On remplace 40 par 400. Lesquelles des trois changent ?
3. Laquelle des trois peut ne correspondre à aucune valeur de la liste ?
4. Pour la colonne `island` des manchots (Biscoe, Dream, Torgersen), laquelle des trois a un sens ?
5. Vrai ou faux : pour une loi normale, la moyenne, la médiane et le mode sont égaux.
6. Pourquoi regarde-t-on ces statistiques **avant** de choisir un modèle ?

### 2.Q2 — Quand a-t-on le droit de parler de probabilités ? 🧠 ⏱️ 3 min
*Fiche §2.2 · livre §2.2 · parcours R, M*

Vrai ou faux ?
1. Les comptages 12, 30 et 8 forment une distribution de probabilité.
2. Les nombres 0,2 ; 0,5 et 0,4 forment une distribution de probabilité.
3. Pour une variable continue $X$, $P(X = 1{,}5) = 0$.
4. Une densité de probabilité ne peut jamais dépasser 1.
5. On parle de pmf pour une loi discrète et de densité (pdf) pour une loi continue.
6. L'aire totale sous une densité vaut 1.

### 2.Q3 — Graine et pseudo-aléatoire : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §2.2.1 · livre §2.2.1 · parcours R*

1. Un générateur pseudo-aléatoire est déterministe.
2. Avec la même version de NumPy, `np.random.default_rng(7).random(3)` renvoie toujours les trois mêmes nombres.
3. Fixer la graine rend les nombres « plus aléatoires ».
4. `np.random.default_rng()`, sans argument, donne la même suite à chaque exécution du programme.
5. La graine 42 a des propriétés statistiques particulières.
6. Pour générer un mot de passe, `np.random` est l'outil adapté.

### 2.Q4 — Loi uniforme sur [0, 1] : questions pièges 🧠 ⏱️ 3 min
*Fiche §2.3.1 · livre §2.3, §2.3.1 · parcours M*

$X$ suit la loi uniforme sur $[0, 1]$.
1. Que vaut $P(X \le 0{,}3)$ ?
2. Que vaut $P(0{,}25 \le X \le 0{,}75)$ ?
3. Que vaut $P(X = 0{,}5)$ ?
4. Quelle est la hauteur de la densité ? Et celle de la loi uniforme sur $[-1, 1]$ ?
5. Que vaut l'espérance (la moyenne) de $X$ ?
6. `rng.random()` peut-il renvoyer 1 ? `rng.integers(1, 7)` peut-il renvoyer 7 ?

### 2.Q5 — La règle 68-95-99,7 🧠 ⏱️ 3 min
*Fiche §2.3.2 · livre §2.3.2 · parcours R, M*

La taille des adultes d'un pays suit à peu près une loi normale de moyenne 170 cm et d'écart-type 7 cm.
1. Quelle part des adultes mesure entre 163 et 177 cm ?
2. Entre 156 et 184 cm ?
3. Plus de 184 cm ?
4. Moins de 149 cm ?
5. Vrai ou faux : la règle 68-95-99,7 vaut pour n'importe quelle distribution.
6. Dans quelle unité s'exprime l'écart-type ici ? Et la variance ?

### 2.Q6 — Bernoulli ou multinoulli ? 🧠 ⏱️ 3 min
*Fiche §2.3.3, §2.3.4 · livre §2.3.3, §2.3.4 · parcours R*

Pour chaque grandeur, dis si elle suit une loi de **Bernoulli**, une loi **catégorielle** (multinoulli) à plus de deux issues, ou **aucune des deux**.
1. Pile ou face.
2. Le chiffre écrit sur une image de MNIST.
3. « Cet e-mail est-il un spam ? »
4. L'espèce d'un manchot.
5. La masse d'un manchot.
6. Le résultat d'un dé à 6 faces.
7. Combien de nombres faut-il connaître pour décrire complètement une loi catégorielle à $K$ classes ?

### 2.Q7 — Une espérance qu'on ne tire jamais 🧠 ⏱️ 3 min
*Fiche §2.3.5 · livre §2.3.5 · parcours M*

1. Un dé équilibré à 4 faces porte les nombres 2, 4, 6 et 8. Quelle est son espérance ? Peut-on la tirer ?
2. Quelle est l'espérance d'une variable de Bernoulli de paramètre 0,3 ? Peut-on la tirer ?
3. Vrai ou faux : l'espérance d'une variable aléatoire est toujours l'une de ses valeurs possibles.
4. Si $\mathbb{E}[X] = 5$, que vaut $\mathbb{E}[2X + 1]$ ?
5. On tire 10 000 fois la même variable et on fait la moyenne des tirages. De quoi cette moyenne se rapproche-t-elle, et comment s'appelle ce résultat ?

### 2.Q8 — Dépendant, indépendant, i.i.d. 🧠 ⏱️ 3 min
*Fiche §2.4, §2.4.1 · livre §2.4, §2.4.1 · parcours R*

Pour chaque situation, dis si les grandeurs sont indépendantes. Pour les situations 1, 2, 5 et 6, qui décrivent une suite de valeurs de même nature, dis aussi si l'on peut les considérer comme i.i.d.
1. Deux lancers successifs d'un même dé.
2. La température d'aujourd'hui et celle de demain, au même endroit.
3. La longueur du bec et celle de la nageoire d'un même manchot.
4. L'espèce d'un manchot tiré au hasard, puis la longueur de sa nageoire.
5. Des lignes tirées au hasard, avec remise, dans une très grande population.
6. Un jeu de test qui contient d'autres photos des personnes du jeu d'entraînement.

### 2.Q9 — Avec ou sans remise ? 🧠 ⏱️ 3 min
*Fiche §2.5 · livre §2.5 à §2.5.3 · parcours R*

Avec ou sans remise ?
1. Le tirage d'une loterie.
2. Un rééchantillon bootstrap.
3. Le découpage d'un dataset en jeu d'entraînement et jeu de test.
4. Les mini-batches d'une epoch, après mélange des exemples.
5. Les commandes des clients successifs d'un café.
6. `rng.choice(a, size=5)`, sans autre argument.
7. Vrai ou faux : sans remise, on peut tirer plus d'éléments qu'il n'y en a.

### 2.Q10 — Ce que le bootstrap estime, et ce qu'il n'invente pas 🧠 ⏱️ 3 min
*Fiche §2.6 · livre §2.6 · parcours R*

Vrai ou faux ?
1. Le bootstrap crée de nouvelles données.
2. Il estime de combien une statistique varierait d'un échantillon à l'autre.
3. Dans la pratique actuelle, chaque rééchantillon a la même taille que l'échantillon.
4. Si l'échantillon est biaisé, l'intervalle bootstrap corrige le biais.
5. Augmenter le nombre de rééchantillons rend l'intervalle de confiance plus étroit.
6. Un rééchantillon de taille $n$ contient en moyenne environ 63 % d'éléments distincts.

### 2.Q11 — Une image est un point dans un espace à 784 dimensions 🧠 ⏱️ 3 min
*Fiche §2.7 · livre §2.7 · parcours M*

1. Combien de coordonnées décrivent une image de 28 × 28 pixels en niveaux de gris ?
2. Et une image couleur de 64 × 64 pixels (trois nombres par pixel) ?
3. Deux images presque identiques sont-elles des points proches ou éloignés ?
4. Vrai ou faux : on peut dessiner un espace à 784 dimensions.
5. Vrai ou faux : la formule de la distance entre deux points (0B) reste valable en dimension 784.
6. Pourquoi dit-on que des données sont « clairsemées » en grande dimension ?

### 2.Q12 — Covariance, corrélation et quartet d'Anscombe 🧠 ⏱️ 4 min
*Fiche §2.8, §2.9 · livre §2.8 à §2.9 · parcours R*

1. Quelle est l'unité de la covariance entre la longueur d'une nageoire (mm) et une masse (g) ?
2. Et celle de leur corrélation ?
3. Que vaut la corrélation si tous les points sont alignés sur une droite décroissante ?
4. Des points disposés en U ont une corrélation proche de 0. Les deux variables sont-elles indépendantes ?
5. Vrai ou faux : une forte corrélation entre deux variables prouve que l'une cause l'autre.
6. Que montre le quartet d'Anscombe, en une phrase ?

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 2.R1 — Ch. 1 : supervisé ou non supervisé, quatre tâches sur Penguins 🔁 ★ ⏱️ 5 min
*Ch. 1 (§1.3, §1.4) · parcours R*

Pour chaque tâche, dis si elle relève de l'apprentissage supervisé ou non supervisé, et de quel type de tâche il s'agit (classification, régression, clustering, réduction de dimension).
1. Prédire l'espèce d'un manchot à partir de ses mesures.
2. Prédire la masse d'un manchot à partir de ses autres mesures.
3. Former des groupes de manchots sans utiliser leur espèce.
4. Résumer les quatre mesures de chaque manchot par deux nombres, pour pouvoir les dessiner.

### 2.R2 — 0A : NumPy, moyenne par colonne avec `axis` 🔁 ★ ⏱️ 5 min
*0A (NumPy, réductions par axe) · parcours R, C*

Soit `X = np.array([[1, 10], [2, 20], [6, 60]])`. Sans exécuter de code :
1. Que vaut `X.shape` ?
2. Que renvoie `X.mean(axis=0)` ?
3. Que renvoie `X.mean(axis=1)` ?
4. Que renvoie `X.mean()` ?
5. Dans un dataset où chaque ligne est un échantillon, quel `axis` donne la moyenne de chaque feature ?

### 2.R3 — 0B : espérance et variance d'un dé équilibré 🔁 ★ ⏱️ 5 min
*0B (101.7.3, 101.7.4) · parcours R, M*

Un dé équilibré à 8 faces porte les nombres 1 à 8.
1. Calcule son espérance $\mathbb{E}[X]$.
2. Calcule $\mathbb{E}[X^2]$.
3. Déduis-en sa variance et son écart-type (2 décimales).
4. Quelle est l'espérance de la somme de deux dés de ce type ?

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats ✏️ dans la partie 0 du notebook.

### Ex 2.1 — Moyenne, médiane et mode d'une liste de salaires ✏️ ★ ⏱️ 10 min
**Objectif :** calculer les trois tendances centrales et mesurer l'effet d'une valeur extrême.
**Prérequis :** fiche §2.2 · 0B (moyenne, 101.1.4) · **Parcours :** R, M

Une petite entreprise verse ces salaires mensuels nets, en euros, à ses 9 salariés (dans le désordre) : 2 300 ; 12 000 ; 2 100 ; 1 900 ; 2 800 ; 2 400 ; 3 500 ; 2 100 ; 2 600.

a) La moyenne des salaires, arrondie à l'euro.
b) La médiane.
c) Le mode.
d) Combien de salariés gagnent **moins** que la moyenne ?
e) Le salaire de la directrice passe de 12 000 € à 30 000 €. Nouvelle moyenne, arrondie à l'euro ?
f) Nouvelle médiane ?
g) Un candidat demande : « Combien gagne-t-on ici, en général ? » Quel chiffre lui donnes-tu, et pourquoi ? (réponds dans ta copie)

### Ex 2.2 — De la casse de voitures à la distribution de probabilité ✏️ ★ ⏱️ 10 min
**Objectif :** transformer des comptages en distribution de probabilité, puis s'en servir pour tirer au hasard.
**Prérequis :** fiche §2.2 · livre §2.2 (figures 2.2 à 2.4) · **Parcours :** M

Une casse compte 800 voitures : 176 berlines, 144 pick-up, 224 monospaces, 96 SUV et 160 breaks. Une roue choisit une voiture au hasard, chacune avec la même chance ; après la photo, la voiture reprend sa place.

a) La probabilité de tirer un monospace.
b) Les probabilités des cinq types, dans l'ordre de l'énoncé, sous forme de liste.
c) Sur une roue simplifiée à cinq parts (comme la figure 2.4 du livre), l'angle de la part « monospace », en degrés (1 décimale).
d) La probabilité de **ne pas** tirer un SUV.
e) Sur 50 tirages, combien de pick-up obtient-on en moyenne ?
f) Les sommes cumulées des probabilités, dans l'ordre de l'énoncé, sous forme de liste.
g) Un générateur donne $u = 0{,}71$, tiré uniformément dans $[0, 1)$. Avec les segments de la question f, quel type de voiture correspond à $u$ ?

### Ex 2.3 — La règle 68-95-99,7 sur les nageoires des manchots ✏️ ★ ⏱️ 10 min
**Objectif :** utiliser la règle 68-95-99,7 et le z-score sur une loi normale donnée.
**Prérequis :** fiche §2.3.2 · **Fil rouge :** Penguins · **Parcours :** M

On suppose que la longueur de nageoire des manchots Adélie suit une loi normale de moyenne $\mu = 190$ mm et d'écart-type $\sigma = 6{,}5$ mm (est-ce réaliste ? tu le vérifieras en 2.18). Utilise uniquement la règle 68-95-99,7.

a) L'intervalle qui contient environ 68 % des Adélie, sous forme de liste `[début, fin]`, en mm.
b) L'intervalle qui contient environ 95 % des Adélie.
c) La proportion d'Adélie dont la nageoire dépasse 203 mm (une proportion entre 0 et 1, 3 décimales).
d) Le dataset compte 151 Adélie mesurés. Combien, en moyenne, dépasseraient 203 mm (arrondi à l'entier) ?
e) Le z-score d'une nageoire de 210 mm (2 décimales).
f) On trouve un manchot dont la nageoire mesure 210 mm. Est-ce vraisemblablement un Adélie ? Justifie. (réponds dans ta copie)

### Ex 2.4 — Variance : diviser par N ou par N − 1 ? ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer une variance et un écart-type avec les deux diviseurs, et voir comment ils réagissent à un changement d'unité.
**Prérequis :** Ex 2.1 · fiche §2.3.2 (🧮 pourquoi $n - 1$) · 0B (101.7.4) · **Parcours :** R, M

Cinq mesures du temps de chargement d'une page web, en secondes : 4 ; 7 ; 6 ; 3 ; 10.

a) La moyenne.
b) La somme des carrés des écarts à la moyenne.
c) La variance avec ddof = 0.
d) La variance avec ddof = 1.
e) L'écart-type avec ddof = 0 (3 décimales).
f) L'écart-type avec ddof = 1 (3 décimales).
g) Vérifie c) avec la formule de 0B, variance = moyenne des carrés − carré de la moyenne : que vaut la **moyenne des carrés** ?
h) On ajoute 100 à chaque mesure. Que devient la variance (ddof = 0) ?
i) On exprime les mesures en millisecondes (on multiplie chaque mesure par 1 000). Que devient l'écart-type (ddof = 0), à 1 décimale ? Pars de la valeur exacte, pas de l'arrondi de e.
j) Laquelle des deux variances utiliserais-tu pour estimer la dispersion de **tous** les chargements de cette page à partir de ces cinq mesures ? Pourquoi ? (réponds dans ta copie)

### Ex 2.5 — Espérances : Bernoulli, multinoulli et jeu de hasard ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer espérances et variances des lois usuelles, et décider si un jeu est équitable.
**Prérequis :** Rappel 2.R3 · fiche §2.3.3 à §2.3.5 · **Parcours :** M

a) Une pièce truquée tombe sur face 30 % du temps. On note $X = 1$ si elle tombe sur face, 0 sinon. Que vaut $\mathbb{E}[X]$ ?
b) Que vaut $\mathrm{Var}(X)$ (2 décimales) ?
c) Sur 200 lancers, combien de faces obtient-on en moyenne ?
d) Quelle est l'espérance d'un dé équilibré à 20 faces (numérotées de 1 à 20) ?
e) Une loi catégorielle sur quatre classes numérotées 1, 2, 3 et 4 a pour probabilités $(0{,}1 ;\ 0{,}2 ;\ 0{,}3 ;\ 0{,}4)$. Quelle est l'espérance du numéro tiré ?
f) On code chaque tirage par un vecteur one-hot de 4 composantes. Quel vecteur représente la classe 3 (liste) ?
g) On tire un très grand nombre de fois et on fait la moyenne des vecteurs one-hot, composante par composante. De quel vecteur cette moyenne se rapproche-t-elle (liste) ?
h) À une kermesse, tu paies 2 € pour lancer le dé à 20 faces (cette mise n'est jamais rendue) : tu reçois 20 € si tu fais 20, 5 € si tu fais 17, 18 ou 19, rien sinon. Quel est ton gain moyen par partie, mise déduite (2 décimales) ?
i) Combien l'organisateur gagne-t-il en moyenne sur 400 parties ?

### Ex 2.6 — Compter les tirages avec et sans remise ✏️ ★★ ⏱️ 20 min
**Objectif :** dénombrer les résultats possibles d'un tirage et calculer la part d'éléments absents d'un rééchantillon bootstrap.
**Prérequis :** fiche §2.5.3, §2.6 · 0B (dénombrement, 101.1.7) · **Parcours :** M

Un dataset compte cinq exemples A, B, C, D et E (le livre fait le même calcul avec trois, §2.5.3). On en tire deux pour former un nouveau jeu.

a) Sans remise, sans tenir compte de l'ordre : combien de nouveaux jeux sont possibles ?
b) Avec remise, sans tenir compte de l'ordre (AB et BA sont le même jeu ; AA est permis) ?
c) Avec remise, en tenant compte de l'ordre (on compte des suites) ?
d) On passe à 7 exemples et on en tire 3, sans remise, sans tenir compte de l'ordre. Combien de jeux ?
e) Un rééchantillon bootstrap des 5 exemples A à E : 5 tirages avec remise, dans l'ordre. Combien de suites possibles ?
f) La probabilité qu'un exemple donné n'apparaisse dans **aucun** des 5 tirages (4 décimales).
g) La même probabilité pour un rééchantillon de 1 000 tirages parmi 1 000 exemples (4 décimales).
h) En moyenne, quelle proportion des 1 000 exemples apparaît au moins une fois dans ce rééchantillon (4 décimales) ?
i) Pourquoi un « bootstrap » tiré **sans** remise, de même taille que l'échantillon, ne servirait-il à rien ? (réponds dans ta copie)

### Ex 2.7 — Covariance et corrélation de cinq points à la main ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer une covariance et une corrélation pas à pas, et voir laquelle dépend des unités.
**Prérequis :** Ex 2.4 · fiche §2.8.1, §2.8.2 · **Parcours :** R, M

Cinq étudiants ont révisé $x$ heures et obtenu la note $y$ : $(1 ; 2)$, $(2 ; 3)$, $(3 ; 5)$, $(4 ; 4)$, $(5 ; 6)$. Sauf mention contraire, ddof = 0.

a) Les moyennes de $x$ et de $y$, sous forme de liste `[moyenne de x, moyenne de y]`.
b) La somme des produits des écarts, $\sum_i (x_i - \bar{x})(y_i - \bar{y})$.
c) La covariance.
d) La covariance avec ddof = 1 (2 décimales).
e) Les écarts-types de $x$ et de $y$, sous forme de liste (3 décimales).
f) La corrélation $r$ (3 décimales).
g) On note le temps de révision en minutes ($x$ multiplié par 60). Que devient la covariance ?
h) Et la corrélation (3 décimales) ?
i) Peut-on conclure qu'une heure de révision de plus fait gagner des points ? Qu'est-ce qui pourrait expliquer autrement ce lien ? (réponds dans ta copie)

### Ex 2.8 — Changer d'unité : la covariance bouge, pas la corrélation ∂ ★★ ⏱️ 25 min
**Objectif :** démontrer comment covariance et corrélation réagissent à un changement d'unité, et pourquoi $r$ reste entre −1 et 1.
**Prérequis :** Ex 2.7 · fiche §2.8 · 0B (Σ, 101.1.3 ; produit scalaire et cosinus, 101.3.3) · **Parcours :** M

On a $n$ couples $(x_i, y_i)$ et deux changements d'unité : $u_i = a\,x_i + b$ et $v_i = c\,y_i + d$, avec $a$ et $c$ non nuls. Toutes les formules avec ddof = 0.

1. Montre que $\bar{u} = a\,\bar{x} + b$.
2. Déduis-en que $u_i - \bar{u} = a\,(x_i - \bar{x})$, et de même que $v_i - \bar{v} = c\,(y_i - \bar{y})$.
3. Montre que $\mathrm{Cov}(u, v) = a\,c\,\mathrm{Cov}(x, y)$.
4. Montre que $\sigma_u = |a|\,\sigma_x$ (pars de $\mathrm{Var}(u) = \mathrm{Cov}(u, u)$).
5. Déduis-en $r(u, v)$ en fonction de $r(x, y)$. Que se passe-t-il si $a$ et $c$ sont positifs ? Si $a$ est négatif et $c$ positif ?
6. Application : on passe la masse des manchots des grammes aux kilogrammes, et leur nageoire des millimètres aux centimètres. Par quel nombre la covariance (masse, nageoire) est-elle multipliée ? Et la corrélation ? Le livre (§2.8.2) prend l'exemple de mesures sur une guitare : que permet la corrélation que la covariance ne permet pas ?
7. Pour aller plus loin : on note $\mathbf{d}_x$ le vecteur des écarts $(x_1 - \bar{x}, \ldots, x_n - \bar{x})$ et $\mathbf{d}_y$ celui des $(y_i - \bar{y})$. Montre que $r = \frac{\mathbf{d}_x \cdot \mathbf{d}_y}{\lVert \mathbf{d}_x \rVert\,\lVert \mathbf{d}_y \rVert}$, le cosinus de l'angle entre ces deux vecteurs (0B). Pourquoi cela prouve-t-il que $-1 \le r \le 1$ ? (On admet l'inégalité de Cauchy-Schwarz : $|\mathbf{p} \cdot \mathbf{q}| \le \lVert \mathbf{p} \rVert\,\lVert \mathbf{q} \rVert$.)

<a id="reflexion"></a>

## 🧮 🗣️ ⚖️ 📄 Réflexion

### Ex 2.9 — Le bootstrap en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer le bootstrap simplement, sans formule.
**Prérequis :** fiche §2.6 · **Parcours :** R

Explique le bootstrap à un collègue qui n'a jamais fait de statistiques, en **cinq lignes au plus**. Contraintes :
- un exemple concret (pas celui de la fiche) ;
- les mots « échantillon », « avec remise » et « intervalle » ;
- aucune formule.

Relis-toi à voix haute : ton collègue pourrait-il refaire la méthode avec un tableur ? Compare ensuite avec la réponse modèle de `05_solutions.md`.

### Ex 2.10 — Corrélation, causalité et échantillon biaisé ⚖️ ★★ ⏱️ 20 min
**Objectif :** repérer une variable de confusion et un échantillon biaisé, et mesurer leurs conséquences quand un modèle décide pour des personnes.
**Prérequis :** fiche §2.6, §2.8.2 · **Parcours :** R

Trois situations :
- **A.** Dans une ville, les quartiers où interviennent le plus de pompiers sont aussi ceux où l'on compte le plus d'incendies.
- **B.** Une application de sport publie : « Nos 50 000 utilisateurs dorment en moyenne 7 h 42 ; intervalle de confiance bootstrap à 95 % : [7 h 41 ; 7 h 43]. » Un journal titre : « Les Français dorment 7 h 42 par nuit. »
- **C.** Une entreprise entraîne un modèle de tri de CV sur ses embauches des dix dernières années. Le modèle accorde beaucoup de poids à une ligne du CV : « pratique du rugby en club ».

1. Pour A, donne deux explications de la corrélation autres que « les pompiers provoquent les incendies ».
2. Pour B, que mesure l'intervalle de confiance, et que ne mesure-t-il pas ? Pourquoi est-il si étroit ? Que devrait écrire le journal ?
3. Pour C, pourquoi cette corrélation pose-t-elle problème, même si elle est « vraie » dans les données ? Quels groupes de candidats risquent d'être désavantagés ?
4. Tu es l'ingénieur·e ML chargé·e du modèle C. Que fais-tu avant de le mettre en service ? Cite au moins trois vérifications. (Pense aussi au cadre légal : dans l'Union européenne, le recrutement fait partie des usages de l'IA les plus encadrés.)

### Ex 2.11 — Fermi : la taille de l'espace des images 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer par des ordres de grandeur la taille de l'espace des images et la place qu'y occupe un dataset.
**Prérequis :** fiche §2.7 · 0B (puissances et logarithmes) · **Fil rouge :** MNIST · **Parcours :** M

Pas de calcul exact : des puissances de 10 et des hypothèses raisonnables. Rappel utile : $2^{10} = 1\,024 \approx 10^3$.
1. Combien de dimensions pour une image de MNIST ? Et pour une photo couleur de 12 mégapixels (4 000 × 3 000 pixels, trois couleurs par pixel) ?
2. Combien d'images différentes de 28 × 28 pixels existe-t-il si chaque pixel est noir ou blanc ? Donne une puissance de 10.
3. Même question avec 256 niveaux de gris par pixel ($256 = 2^8$).
4. Compare avec le nombre d'atomes de l'univers observable (environ $10^{80}$). Quelle fraction de l'espace des images en niveaux de gris les 70 000 images de MNIST occupent-elles ?
5. Combien d'octets occupe une image de MNIST stockée en `float32` (4 octets par nombre) ? Le dataset entier ? Et en `uint8` (1 octet par nombre) ?
6. Malgré les questions 2 à 4, un modèle entraîné sur 60 000 images reconnaît très bien des chiffres qu'il n'a jamais vus. Propose une explication.

### Ex 2.12 — Anscombe (1973) : regarder avant de calculer 📄 ★★ ⏱️ 25 min
**Objectif :** lire un article court et fondateur, retrouver ses chiffres et en tirer une règle de travail.
**Prérequis :** Ex 2.7 · fiche §2.9 · livre §2.9 · **Parcours :** complet seulement (lecture conseillée à tous)

L'article : F. J. Anscombe, « Graphs in Statistical Analysis », *The American Statistician*, vol. 27, n° 1, 1973, p. 17-21 ([DOI 10.1080/00031305.1973.10478966](https://doi.org/10.1080/00031305.1973.10478966)). Il fait cinq pages ; l'accès est payant chez l'éditeur, mais une bibliothèque universitaire y donne souvent accès. Sans accès, la page Wikipédia en anglais « [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet) » résume son propos et donne les données ; la figure de la fiche (§2.9) montre les quatre nuages.

1. Quelle croyance sur les graphiques et les calculs Anscombe veut-il combattre ?
2. Les valeurs de $x$ des jeux I, II et III sont 10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5. Calcule leur moyenne, la somme des carrés des écarts, puis leur variance avec ddof = 0 et avec ddof = 1. Laquelle correspond au « 11 » des tableaux habituels ? Laquelle à l'écart-type de 3,16 cité par le livre ?
3. Pour chacun des quatre nuages, décris en une phrase ce que tu vois. Pour lequel la droite de régression est-elle un bon résumé ?
4. Dans le jeu IV, toutes les valeurs de $x$ valent 8, sauf une (19). Que deviendrait la corrélation si l'on retirait ce point ? Explique.
5. Quelle règle de travail en tires-tu pour tes propres analyses ? Quel est son équivalent moderne (fiche, 🕰️ de la §2.9) ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 2.E1 — Moyenne ou médiane pour résumer des salaires ? 💼 ★★ ⏱️ 10 min
*Fiche §2.2 · prérequis 2.1 · parcours R*

« Pour résumer les salaires d'une entreprise dans un tableau de bord, vous prenez la moyenne ou la médiane ? Pourquoi ? »

### 2.E2 — i.i.d. : définition et pourquoi le ML en a besoin 💼 ★★ ⏱️ 10 min
*Fiche §2.4.1 · parcours R*

« Que veut dire i.i.d. ? Pourquoi est-ce si important en machine learning ? Donnez-moi un cas où l'hypothèse ne tient pas. »

### 2.E3 — Expliquer un intervalle de confiance bootstrap 💼 ★★ ⏱️ 10 min
*Fiche §2.6 · prérequis 2.22 (notebook) · parcours R*

« Votre modèle obtient 87 % d'accuracy sur un jeu de test de 400 exemples. Comment donneriez-vous une marge d'incertitude sur ce chiffre, sans collecter de nouvelles données ? Et comment l'expliqueriez-vous à un chef de projet ? »

### 2.E4 — Corrélation nulle veut-elle dire indépendance ? 💼 ★★ ⏱️ 10 min
*Fiche §2.8.2, §2.9 · prérequis 2.26 (notebook) · parcours R*

« Deux variables ont une corrélation nulle : sont-elles indépendantes ? Et si leur corrélation vaut 0,9, l'une cause-t-elle l'autre ? »

### 2.E5 — Pourquoi fixer la graine aléatoire d'une expérience ? 💼 ★★ ⏱️ 10 min
*Fiche §2.2.1 · prérequis 2.Q3 · parcours R*

« Pourquoi fixe-t-on la graine aléatoire dans une expérience de machine learning ? Est-ce que cela suffit pour que vos résultats soient fiables ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch02_stats/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie `mylearn/stats.py`. La partie 0 du notebook vérifie tes réponses aux exercices ✏️ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 2.13 | Tendances centrales : mean, median, mode | 🔨 | ★★ | 30 |
| 2.14 | Graine fixée ou graine libre ? | 🔮 | ★ | 10 |
| 2.15 | Dispersion : variance, std, percentile, zscore | 🔨 | ★★★ | 40 |
| 2.16 | Un histogramme fait maison | 🔨 | ★★ | 20 |
| 2.17 | Galerie des lois usuelles | 🎨 | ★★ | 20 |
| 2.18 | 68-95-99,7 : la théorie face aux tirages et aux manchots | 🔬 | ★★ | 20 |
| 2.19 | La roue de la fortune : tirer dans une distribution discrète | 🔨 | ★★ | 25 |
| 2.20 | Le pelage des animaux : une variable qui dépend d'une autre | 🔮 | ★★ | 15 |
| 2.21 | Tirer avec ou sans remise | 🔨 | ★★ | 20 |
| 2.22 | Bootstrap : distribution et intervalle de confiance | 🔨 | ★★ | 30 |
| 2.23 | Bootstraps de 20 (livre) ou de n (aujourd'hui) ? | 🔬 | ★★ | 30 |
| 2.24 | Comparer avec scipy.stats.bootstrap | 📦 | ★★ | 15 |
| 2.25 | Distances entre chiffres dans l'espace à 784 dimensions | 📦 | ★★ | 25 |
| 2.26 | Covariance et corrélation | 🔨 | ★★ | 25 |
| 2.27 | Deviner la corrélation d'un nuage de points | 📈 | ★★ | 20 |
| 2.28 | Matrices de covariance et de corrélation des manchots | 🔨 | ★★ | 25 |
| 2.29 | Le piège de ddof : NumPy, pandas et toi | 🐛 | ★★ | 20 |
| 2.30 | Le quartet d'Anscombe | 🎨 | ★★ | 25 |
| 2.31 | Docstring et test pytest pour zscore | 🛠️ | ★★ | 30 |
| 2.32 | Mêmes statistiques, autre dessin : fabrique ton quartet | 🏆 | ★★★ | 60 |
