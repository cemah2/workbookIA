# 19 · Optimiseurs — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch19_optimiseurs/06_mes_reponses.md` (créée par `python tools/start_chapter.py 19`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ 19.1 à 19.6 et de l'estimation 🧮 19.9 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les preuves ∂ 19.7 et 19.8, la réflexion (🗣️ ⚖️ 📄), la lecture de documentation 🛠️ 19.13 et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), des lettres collées pour plusieurs choix (`"BD"`), une forme entre guillemets (`"(3, 5)"`).
> Conventions du chapitre : $g_t$ est le gradient au pas $t$, compté à partir de 1 ; les formules d'un optimiseur sont celles de `torch.optim` (fiche §19.5), sauf quand une question dit « convention du livre ».

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 🛠️ Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 19.Q1 — Qu'appelle-t-on un optimiseur ? 🧠 ⏱️ 3 min
*Fiche §19.1, §19.2 · livre §19.1, §19.2 · parcours R*

a) Vrai ou faux : un optimiseur remplace la rétropropagation.
b) Qu'est-ce qu'un optimiseur reçoit à chaque pas ? (A) les données brutes ; (B) les paramètres et les gradients de la loss par rapport à eux ; (C) les labels du mini-batch ; (D) l'architecture du réseau.
c) Vrai ou faux : SGD, Adam et RMSprop, appliqués au même réseau au même point, reçoivent le même gradient ; ce qui les distingue, c'est ce qu'ils en font.
d) Dans PyTorch, quel appel applique la mise à jour des poids ? (A) `loss.backward()` ; (B) `optimizer.step()` ; (C) `optimizer.zero_grad()` ; (D) `model.eval()`.

### 19.Q2 — Minimum, maximum, plateau, selle : où le gradient s'annule-t-il ? 🧠 ⏱️ 3 min
*Fiche §19.2.1 · livre §19.2.1 · parcours R*

Pour chaque fonction, le gradient s'annule en $(0, 0)$. Choisis la nature de ce point : (A) un minimum ; (B) un maximum ; (C) un point selle ; (D) un point d'un plateau.

a) $f(x, y) = x^2 + 3y^2$.
b) $f(x, y) = 2x^2 - y^2$.
c) $f(x, y) = -x^2 - y^2$.
d) $f(x, y) = 5$ sur tout le carré $[-1, 1]^2$.
e) Vrai ou faux : partie **exactement** d'un point où le gradient est nul, la descente de gradient (sans bruit) ne bouge plus, même si ce point n'est pas un minimum.

### 19.Q3 — Minima locaux ou points selles : le vrai piège en grande dimension 🧠 ⏱️ 3 min
*Fiche §19.2.1 · livre §19.2.1 · parcours R*

a) D'après Y. Dauphin et ses coauteurs (2014), cités par le livre, quel obstacle domine en grande dimension ? (A) les mauvais minima locaux ; (B) les points selles, souvent entourés de régions presque plates ; (C) les maxima ; (D) les erreurs d'arrondi.
b) Un modèle jouet : en un point où le gradient s'annule, chacune des $d$ directions propres de la hessienne monte ou descend, avec une chance sur deux, indépendamment des autres. Pour $d = 10$, le point est un minimum avec une probabilité de 1 sur combien ?
c) Vrai ou faux : dans ce modèle jouet, plus la dimension est grande, plus un point où le gradient s'annule a de chances d'être un point selle.
d) Vrai ou faux : le bruit de la descente stochastique (§19.4.2) aide à quitter un point selle.

### 19.Q4 — Pourquoi lisser une erreur de comptage ? 🧠 ⏱️ 3 min
*Fiche §19.2.2 · livre §19.2.2*

On classe des points sur une droite par un seuil, et l'on compte les points mal classés en fonction de la position du seuil (figure 19.6 du livre).

a) Vrai ou faux : ce nombre d'erreurs est une fonction en escalier (constante par morceaux) de la position du seuil.
b) Vrai ou faux : là où elle existe, sa dérivée est nulle.
c) Pourquoi le lisser ? (A) pour calculer plus vite ; (B) pour obtenir un gradient non nul qui dise dans quel sens déplacer le seuil ; (C) pour supprimer les minima locaux ; (D) pour que l'erreur soit nulle au minimum.
d) Quelle loss joue ce rôle de version lisse du comptage pour un classifieur binaire ? (A) l'accuracy ; (B) la cross-entropy binaire (log loss) ; (C) le nombre de faux positifs ; (D) le F1-score.

### 19.Q5 — Pas constant : pourquoi rebondit-on au fond de la vallée ? 🧠 ⏱️ 3 min
*Fiche §19.3, §19.3.1 · livre §19.3, §19.3.1 · parcours R*

Au fond d'une vallée, la loss ressemble à $L(w) = 5\,w^2$ (courbure $a = 10$), et l'on fait $w \leftarrow w - \eta\,L'(w)$.

a) Avec $\eta = 0{,}15$, le facteur par lequel $w$ est multiplié à chaque pas.
b) Vrai ou faux : avec ce learning rate, $w$ change de signe à chaque pas.
c) Avec $\eta = 0{,}25$, la descente : (A) converge sans changer de signe ; (B) converge en oscillant ; (C) oscille sans converger ni diverger ; (D) diverge.
d) Même question avec $\eta = 0{,}2$.
e) Vrai ou faux : avec des gradients exacts et un learning rate constant assez petit, la descente converge vers le fond de cette vallée.

### 19.Q6 — Décroissance exponentielle : que vaut η après 50 pas ? 🧠 ⏱️ 3 min
*Fiche §19.3.2 · livre §19.3.2 · parcours M*

On part de $\eta_0 = 0{,}1$ et l'on multiplie $\eta$ par $\gamma = 0{,}99$ après chaque pas.

a) La valeur de $\eta$ après 50 multiplications (4 décimales).
b) Le premier nombre de multiplications après lequel $\eta$ est passé sous la moitié de $\eta_0$.
c) Avec $\gamma = 0{,}8$ (le réglage des figures du livre), après 10 multiplications, $\eta$ a été divisé par environ : (A) 2 ; (B) 10 ; (C) 100 ; (D) 1 000.
d) Vrai ou faux : avec $\gamma < 1$, la somme $\eta_0 + \eta_1 + \eta_2 + \dots$ de tous les learning rates, sur une infinité de pas, est finie.
e) Sa valeur pour $\eta_0 = 0{,}1$ et $\gamma = 0{,}99$.

### 19.Q7 — Plannings : exponentiel, retardé, par paliers, sur l'erreur, bold driver 🧠 ⏱️ 3 min
*Fiche §19.3.3 · livre §19.3.3 · parcours R*

Associe chaque règle à son planning : (A) exponentiel ; (B) exponentiel retardé ; (C) par paliers ; (D) sur l'erreur ; (E) bold driver.

a) $\eta$ est multiplié par 0,5 toutes les 10 epochs.
b) $\eta$ reste constant pendant 5 epochs, puis il est multiplié par 0,95 à chaque epoch.
c) $\eta$ n'est réduit que lorsque la loss de validation ne baisse plus depuis quelques epochs.
d) $\eta$ augmente de 5 % quand la loss de l'epoch baisse, et il est divisé par 2 quand elle monte.
e) La classe de `torch.optim.lr_scheduler` qui réalise (D) : (A) `StepLR` ; (B) `ExponentialLR` ; (C) `ReduceLROnPlateau` ; (D) `CosineAnnealingLR`.
f) Vrai ou faux : le livre compte ses plannings en epochs, alors que le warmup et le cosinus d'aujourd'hui se comptent souvent en pas de mise à jour.

### 19.Q8 — Batch, stochastique, mini-batch : compter les mises à jour 🧠 ⏱️ 3 min
*Fiche §19.4, §19.4.1 à §19.4.3 · livre §19.4 · parcours R*

Un dataset d'entraînement compte 1 200 exemples.

a) Le nombre de mises à jour par epoch en descente par batch.
b) Le même nombre en descente stochastique (un exemple par mise à jour).
c) Le même nombre avec des mini-batches de 64, en gardant le dernier mini-batch incomplet.
d) Le nombre total de mises à jour pour 50 epochs avec ces mini-batches de 64.
e) Lequel est un algorithme « en ligne » (*online*) ? (A) la descente par batch ; (B) la descente stochastique ; (C) les deux ; (D) aucun.
f) Vrai ou faux : comparer deux stratégies à leur nombre d'epochs suffit à comparer leur coût.

### 19.Q9 — Momentum : que se passe-t-il si γ = 0 ? si γ = 1 ? 🧠 ⏱️ 3 min
*Fiche §19.5, §19.5.1 · livre §19.5, §19.5.1 · parcours R*

On utilise la convention de PyTorch : $b_t = \mu\,b_{t-1} + g_t$ ($b_1 = g_1$), $w \leftarrow w - \eta\,b_t$.

a) Avec $\mu = 0$, on obtient : (A) la descente de gradient ordinaire ; (B) aucun mouvement ; (C) Adam ; (D) une divergence certaine.
b) Si le gradient vaut toujours 1 et $\mu = 0{,}9$, la valeur vers laquelle $b_t$ s'approche.
c) Si le gradient vaut toujours 1 et $\mu = 1$, $b_t$ : (A) s'approche d'une limite ; (B) grandit sans limite, proportionnellement à $t$ ; (C) grandit exponentiellement ; (D) reste égal à 1.
d) Avec $\mu = 1$, si le gradient devient nul à partir d'un certain pas, les poids : (A) s'arrêtent ; (B) continuent à la même vitesse, sans jamais ralentir ; (C) repartent en arrière.
e) Vrai ou faux : le momentum aide à traverser un plateau où le gradient est presque nul.

### 19.Q10 — Nesterov : regarder là où l'on va 🧠 ⏱️ 3 min
*Fiche §19.5.2 · livre §19.5.2*

a) Dans la version du livre, Nesterov calcule le gradient : (A) au point courant ; (B) au point anticipé, le point courant plus la part $\gamma$ du changement précédent ; (C) au point précédent ; (D) en deux points dont il fait la moyenne.
b) Vrai ou faux : Nesterov ajoute un hyperparamètre au momentum.
c) Vrai ou faux : avec `nesterov=True`, PyTorch prend le gradient au point courant, grâce à un changement de variable.
d) Quand l'élan s'apprête à faire dépasser le fond d'une vallée, le gradient anticipé : (A) accélère encore ; (B) freine plus tôt ; (C) ne change rien ; (D) arrête net.

### 19.Q11 — Adagrad, Adadelta, RMSprop : un learning rate par poids 🧠 ⏱️ 3 min
*Fiche §19.5.3, §19.5.4 · livre §19.5.3, §19.5.4 · parcours R*

a) Vrai ou faux : Adagrad donne à chaque poids son propre pas effectif.
b) Pourquoi les pas d'Adagrad ne font-ils que rétrécir ? (A) la somme des carrés des gradients ne fait que grandir ; (B) il oublie les vieux gradients ; (C) son learning rate est divisé par 2 à chaque epoch ; (D) le gradient change de signe.
c) RMSprop avec $\alpha = 0{,}9$ : la part du gradient au carré le plus récent dans la nouvelle moyenne $v_t$.
d) Dans `torch.optim.RMSprop`, quel paramètre joue le rôle du $\gamma$ du livre ? (A) `lr` ; (B) `alpha` ; (C) `momentum` ; (D) `eps`.
e) La valeur par défaut de ce paramètre dans PyTorch.
f) Vrai ou faux : l'Adadelta d'origine (M. Zeiler, 2012) n'a pas de learning rate.

### 19.Q12 — Adam, No Free Lunch et choix d'un optimiseur 🧠 ⏱️ 3 min
*Fiche §19.5.5, §19.6 · livre §19.5.5, §19.6 · parcours R*

a) Adam combine : (A) un momentum et une moyenne mobile des carrés des gradients ; (B) Nesterov et Adagrad ; (C) le clipping et le warmup ; (D) le bold driver et les paliers.
b) La valeur de $\beta_1$ proposée par Kingma et Ba.
c) La valeur de $\beta_2$ proposée par Kingma et Ba.
d) Vrai ou faux : le théorème No Free Lunch dit que tous les optimiseurs se valent sur les problèmes réels.
e) Sur le banc d'essai du livre (les lunes, figure 19.49), lequel l'emporte ? (A) Adagrad ; (B) Adadelta ; (C) la descente par mini-batches avec Nesterov ; (D) Adam.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 19.R1 — Ch. 18 : du delta au gradient d'un poids 🔁 ★ ⏱️ 5 min
*Ch. 18 (fiche, §18.7 et §18.12.10) · parcours R, M*

a) Un neurone $i$ sort $a_i = 0{,}4$ et alimente un neurone $j$ de delta $\delta_j = -2{,}5$. La valeur de $\partial L / \partial w_{ij}$.
b) Avec la descente vanille et $\eta = 0{,}05$, le changement de $w_{ij}$ après un pas.
c) Pour un mini-batch, $d\mathbf{W} = \mathbf{A}_{\text{prev}}^\top\boldsymbol{\Delta}$. Si $\mathbf{A}_{\text{prev}}$ a la forme $(32, 64)$ et $\boldsymbol{\Delta}$ la forme $(32, 10)$, la forme de $d\mathbf{W}$ (entre guillemets).
d) Vrai ou faux : remplacer la descente vanille par Adam change la façon dont la rétropropagation calcule $\partial L / \partial w_{ij}$.

### 19.R2 — Ch. 16 : pourquoi l'initialisation He garde la variance stable 🔁 ★ ⏱️ 5 min
*Ch. 16 (fiche, les initialisations) · parcours R*

a) L'initialisation de He tire les poids d'une couche de $n_{in} = 200$ entrées selon une loi normale de variance $2/n_{in}$. Cette variance.
b) L'écart-type correspondant.
c) L'initialisation de Glorot (loi normale) pour $n_{in} = 200$ et $n_{out} = 50$ : sa variance $2/(n_{in} + n_{out})$.
d) D'où vient le facteur 2 de He ? (A) ReLU met à zéro la moitié de ses entrées, en moyenne ; (B) on compte deux couches ; (C) le biais double la variance ; (D) la dérivée de ReLU vaut 2.

### 19.R3 — Ch. 12 : standardiser les features, et l'effet sur la descente 🔁 ★ ⏱️ 5 min
*Ch. 12 (fiche, la standardisation) · fiche §19.2.1 (encadré sur la courbure) · parcours R*

Une régression linéaire à deux features centrées et non corrélées, de loss $L(\mathbf{w}) = \frac{1}{2n}\sum_i (\mathbf{w}\cdot\mathbf{x}_i - y_i)^2$. Sa hessienne est diagonale, de coefficients $\overline{x_1^2}$ et $\overline{x_2^2}$ (les moyennes des carrés de chaque feature). La première feature, une distance en mètres, a $\overline{x_1^2} = 10^6$ ; la seconde, une proportion, a $\overline{x_2^2} = 0{,}25$.

a) Le conditionnement $\kappa = \lambda_{\max} / \lambda_{\min}$ de la loss.
b) Le plus grand learning rate constant pour lequel la descente converge, $2/\lambda_{\max}$, en millionièmes (en multiple de $10^{-6}$).
c) Le conditionnement après standardisation des deux features (ch. 12).
d) Vrai ou faux : en divisant chaque gradient par la racine d'une moyenne de ses carrés, RMSprop et Adam compensent en partie les différences d'échelle entre les features.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**. Dans les exercices sur les optimiseurs, on prend $\epsilon = 0$ sauf mention contraire.

### Ex 19.1 — Plannings exponentiel et par paliers à la main ✏️ ★ ⏱️ 10 min
**Objectif :** calculer les learning rates de trois plannings de décroissance, epoch par epoch.
**Prérequis :** 0B (puissances, suites géométriques) · fiche §19.3.2, §19.3.3 · **Parcours :** R, M

On part de $\eta_0 = 0{,}2$ à l'epoch 0.

a) Planning exponentiel, $\eta_k = \eta_0\,\gamma^k$ avec $\gamma = 0{,}9$ : la valeur de $\eta$ à l'epoch 3.
b) La valeur de $\eta$ à l'epoch 10 (4 décimales).
c) Planning par paliers, $\eta_k = \eta_0\,\gamma^{\lfloor k/s \rfloor}$ avec $s = 4$ et $\gamma = 0{,}5$ : la valeur de $\eta$ à l'epoch 3.
d) La même à l'epoch 4.
e) La même à l'epoch 11.
f) Le facteur $\gamma$ d'un planning exponentiel qui divise lui aussi $\eta$ par 2 toutes les 4 epochs, mais un peu à chaque epoch (4 décimales).
g) Planning exponentiel retardé : $\eta_k = \eta_0$ pour $k \le 4$, puis $\eta_k = \eta_0 \times 0{,}9^{\,k - 4}$. La valeur de $\eta$ à l'epoch 8 (4 décimales).

Dans ta copie : dans quelle situation préférerais-tu des paliers à une décroissance exponentielle ?

### Ex 19.2 — Trois pas de momentum, deux conventions ✏️ ★★ ⏱️ 15 min
**Objectif :** dérouler le momentum à la main et voir quand les conventions du livre et de PyTorch divergent.
**Prérequis :** ch. 5 (dérivée) · fiche §19.5.1 · **Parcours :** M

On minimise $L(w) = w^2$ (donc $g = 2w$) en partant de $w_0 = 1$, avec $\eta = 0{,}1$ et $\mu = 0{,}9$. Convention de PyTorch : $b_1 = g_1$, puis $b_t = \mu\,b_{t-1} + g_t$, et $w \leftarrow w - \eta\,b_t$ ; le gradient $g_t$ se calcule au point $w_{t-1}$.

a) La vitesse $b_1$.
b) La position $w_1$.
c) La vitesse $b_2$.
d) La position $w_2$.
e) La position $w_3$.
f) Convention du livre : $\Delta_t = \gamma\,\Delta_{t-1} - \eta\,g_t$ avec $\Delta_0 = 0$, puis $w_t = w_{t-1} + \Delta_t$ ($\gamma = \mu = 0{,}9$). Vrai ou faux : elle donne le même $w_2$ que PyTorch.
g) On change maintenant le learning rate au deuxième pas : $\eta = 0{,}1$ au premier pas, $\eta = 0{,}05$ au deuxième. La position $w_2$ avec la convention de PyTorch.
h) La position $w_2$ avec la convention du livre.

Dans ta copie : explique l'écart entre g) et h). Laquelle des deux conventions « oublie » plus vite un ancien learning rate ?

### Ex 19.3 — Nesterov : gradient anticipé et reformulation de PyTorch ✏️ ★★ ⏱️ 20 min
**Objectif :** dérouler Nesterov dans les deux écritures, et retrouver le changement de variable qui les relie.
**Prérequis :** Ex 19.2 · fiche §19.5.2 · **Parcours :** M

Même loss $L(w) = w^2$, même départ $w_0 = 1$, $\eta = 0{,}1$, $\mu = 0{,}9$.

**Version du livre** (gradient anticipé) : $v_0 = 0$, puis $v_t = \mu\,v_{t-1} - \eta\,L'(w_{t-1} + \mu\,v_{t-1})$ et $w_t = w_{t-1} + v_t$.

a) Au deuxième pas, le point anticipé $w_1 + \mu\,v_1$.
b) Le gradient en ce point.
c) La position $w_2$.
d) La position $w_3$ (4 décimales).

**Version de PyTorch** (`nesterov=True`) : $p_0 = 1$, $b_1 = g_1$, puis $b_t = \mu\,b_{t-1} + g_t$, et $p_t = p_{t-1} - \eta\,(g_t + \mu\,b_t)$, où $g_t = L'(p_{t-1})$.

e) La position $p_1$.
f) La position $p_2$.
g) À quelle quantité de la version du livre $p_2$ est-il égal ? (A) $w_2$ ; (B) $w_2 + \mu\,v_2$, le point anticipé du pas suivant ; (C) $w_3$ ; (D) $v_2$.

Dans ta copie : pourquoi peut-on garder comme poids du réseau le point anticipé plutôt que $w_t$ ? Que vaut l'écart entre les deux quand l'entraînement a convergé ?

### Ex 19.4 — Adagrad et RMSprop sur une suite de gradients ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer les pas d'Adagrad et de RMSprop, et comparer leur façon d'adapter le pas.
**Prérequis :** 0B (racine carrée) · fiche §19.5.3, §19.5.4 · **Parcours :** R, M

Un poids part de $w_0 = 0$ et reçoit, quoi qu'il arrive, les gradients $g_1 = 3$, $g_2 = -1$, $g_3 = 2$. On prend $\eta = 0{,}1$ et $\epsilon = 0$ ; les accumulateurs partent de 0.

**Adagrad** :
a) L'accumulateur $s_2$.
b) La position $w_2$ (4 décimales).
c) La position $w_3$ (4 décimales).

**RMSprop**, avec $\alpha = 0{,}9$ :
d) La moyenne $v_1$.
e) La position $w_1$ (4 décimales).
f) La moyenne $v_3$.
g) La position $w_3$ (4 décimales).

h) Le rapport entre le premier pas de RMSprop et le premier pas d'Adagrad (en valeur absolue, 4 décimales).
i) Avec un gradient **constant**, le pas d'Adagrad au 100ᵉ pas divisé par son premier pas.

Dans ta copie : pourquoi le premier pas de RMSprop est-il si grand ? Que corrigera Adam ?

### Ex 19.5 — Un pas d'Adam et la correction de biais ✏️ ★★ ⏱️ 20 min
**Objectif :** dérouler deux pas d'Adam à la main et mesurer l'effet de la correction de biais.
**Prérequis :** Ex 19.4 · fiche §19.5.5 · **Parcours :** R, M

Un poids part de $w_0 = 1$ et reçoit les gradients $g_1 = 0{,}5$ puis $g_2 = -0{,}1$. Adam avec $\eta = 0{,}01$, $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\epsilon = 0$.

a) Le premier moment $m_1$.
b) Le second moment $v_1$.
c) Le premier moment corrigé $\hat m_1$.
d) La position $w_1$.
e) Sans correction de biais (le pas $\eta\,m_1/\sqrt{v_1}$), la taille du premier pas (4 décimales).
f) Le premier moment $m_2$.
g) Le premier moment corrigé $\hat m_2$ (4 décimales).
h) La position $w_2$ (5 décimales).
i) Vrai ou faux : au deuxième pas, $w$ continue de baisser alors que $g_2$ est négatif.
j) Vrai ou faux : si l'on multiplie tous les gradients par 1 000 (avec $\epsilon = 0$), les positions $w_1$ et $w_2$ ne changent pas.

Dans ta copie : explique i) avec les mots « momentum » et « moyenne ». Puis dis à quoi sert $\epsilon$ dans le cas de j).

### Ex 19.6 — Adam + L2 ou AdamW : la même pénalité ? Convertir alpha, lambda et weight_decay d'une convention à l'autre ✏️ ★★ ⏱️ 25 min
**Objectif :** distinguer la pénalité L2 couplée et le weight decay découplé, et passer d'une convention de régularisation à l'autre.
**Prérequis :** Ex 19.5 · ch. 9 (régularisation L2, ses conventions) · fiche « AdamW, le weight decay découplé » · **Parcours :** M

**Conventions.** Le `weight_decay` de `torch.optim.SGD` et de `torch.optim.Adam` ajoute $\text{wd}\cdot w$ au gradient (couplé) ; celui d'`AdamW` fait $w \leftarrow w\,(1 - \eta\cdot\text{wd})$ avant le pas d'Adam (découplé). Dans f) et g), $\epsilon$ garde sa valeur par défaut, $10^{-8}$ : un pas d'Adam sur un gradient nul vaut alors 0, et $\epsilon$ ne change rien d'autre aux arrondis demandés.

a) On ajoute $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ à la loss, avec $\lambda = 0{,}01$. Le `weight_decay` de `SGD` qui donne la même mise à jour.
b) On ajoute $\lambda\,\lVert\mathbf{w}\rVert^2$ (sans le $\frac12$), avec $\lambda = 0{,}01$. Le `weight_decay` équivalent.
c) `SGD(lr=0.1, weight_decay=0.01)`, un poids $w = 2$ et un gradient de la loss nul : la valeur de $w$ après un pas.
d) `AdamW(lr=1e-3, weight_decay=1e-2)` (les valeurs par défaut) et un gradient nul : le facteur par lequel $w$ est multiplié à chaque pas (5 décimales).
e) Le facteur total après 10 000 pas de ce genre (4 décimales).
f) `Adam(lr=1e-3, weight_decay=1e-2)` (couplé), $w = 2$, gradient de la loss nul, premier pas : la valeur de $w_1$.
g) `AdamW(lr=1e-3, weight_decay=1e-2)` dans la même situation : la valeur de $w_1$ (5 décimales).
h) Dans l'article de Loshchilov et Hutter, le weight decay s'écrit $w \leftarrow w - \eta_t\,\lambda\,w$, où $\eta_t$ est le multiplicateur du planning (1 ici) et pas le learning rate $\alpha$. Pour $\lambda = 10^{-4}$ et $\alpha = 10^{-3}$, le `weight_decay` de `torch.optim.AdamW` équivalent.
i) Le `MLPClassifier` de scikit-learn ajoute au gradient moyen d'un mini-batch $\frac{\alpha}{n_b}\,\mathbf{W}$, où $n_b$ est la taille du mini-batch. Pour `alpha=1e-4` et `batch_size=200`, le `weight_decay` de `SGD` équivalent, en multiple de $10^{-7}$.
j) Vrai ou faux : pour la descente **avec momentum** (`SGD(momentum=0.9)`), à learning rate constant, weight decay couplé (ajouté au gradient, avant la vitesse) et découplé (le poids rétréci d'un facteur $1 - \eta\cdot\text{wd}$, puis le momentum sur le gradient brut) donnent les mêmes poids après deux pas.

Dans ta copie : comparer f) et g). Pourquoi la pénalité couplée d'Adam donne-t-elle un pas qui ne dépend presque pas de $\lambda$ ? Qu'en déduis-tu pour les poids qui ont de grands gradients ?

### Ex 19.7 — Sur une parabole, η doit rester sous 2/a ∂ ★★★ ⏱️ 30 min
**Objectif :** démontrer la condition de convergence d'un pas constant, puis l'étendre à une vallée en deux dimensions et au momentum.
**Prérequis :** ch. 5 (descente de gradient) · Ex 19.1 · fiche §19.2.1 (encadré sur la courbure), §19.3.1 · **Parcours :** M

Réponds dans ta copie.

1. Pour $L(w) = \frac{a}{2}(w - w^\star)^2$, avec $a > 0$, montre que la descente $w_{t+1} = w_t - \eta\,L'(w_t)$ vérifie $w_{t+1} - w^\star = (1 - \eta a)\,(w_t - w^\star)$. Déduis-en, selon la valeur de $\eta a$, les quatre comportements possibles (convergence sans oscillation, convergence en oscillant, oscillation d'amplitude constante, divergence) et la condition $0 < \eta < 2/a$.
2. Quel $\eta$ atteint le minimum en un pas ? Pour $\eta a = 0{,}1$, combien de pas faut-il, environ, pour diviser l'écart $|w_t - w^\star|$ par 10 ? (Donne la formule, puis la valeur.)
3. En deux dimensions, $L(\mathbf{w}) = \frac12\,(a_1 w_1^2 + a_2 w_2^2)$ avec $a_1 \ge a_2 > 0$. Montre que les deux coordonnées évoluent indépendamment. Quelle condition sur $\eta$ assure la convergence ? Pour $\eta = 1{,}8/a_1$, quelle coordonnée converge le plus lentement ? Montre que la réponse dépend du conditionnement $\kappa = a_1/a_2$, et donne le seuil.
4. On cherche le $\eta$ constant qui minimise le plus lent des deux facteurs, $\max(|1 - \eta a_1|, |1 - \eta a_2|)$. Montre que c'est $\eta^\star = 2/(a_1 + a_2)$, et que le facteur vaut alors $(\kappa - 1)/(\kappa + 1)$, où $\kappa = a_1/a_2$ est le conditionnement. Que devient ce facteur quand $\kappa$ est très grand ?
5. Près d'un minimum, une loss lisse ressemble à $\frac12(\mathbf{w} - \mathbf{w}^\star)^\top\mathbf{H}(\mathbf{w} - \mathbf{w}^\star)$. En te plaçant dans la base des vecteurs propres de $\mathbf{H}$, explique pourquoi les questions 3 et 4 s'appliquent avec les valeurs propres de $\mathbf{H}$ à la place de $a_1$ et $a_2$.
6. **Le momentum.** Avec la convention du livre, $\Delta_t = \gamma\Delta_{t-1} - \eta\,a\,(w_{t-1} - w^\star)$ et $w_t = w_{t-1} + \Delta_t$. Montre que l'écart $e_t = w_t - w^\star$ vérifie $e_{t+1} = (1 + \gamma - \eta a)\,e_t - \gamma\,e_{t-1}$. On admet que la suite converge si et seulement si les deux racines de $z^2 - (1 + \gamma - \eta a)\,z + \gamma = 0$ sont de module strictement inférieur à 1, ce qui équivaut, pour $0 \le \gamma < 1$, à $0 < \eta a < 2\,(1 + \gamma)$. Compare avec la question 1 : le momentum élargit-il ou réduit-il la plage des learning rates qui convergent ?

### Ex 19.8 — Moyenne mobile exponentielle : mémoire effective et biais initial ∂ ★★★ ⏱️ 30 min
**Objectif :** démontrer les propriétés de la moyenne mobile exponentielle qu'utilisent RMSprop et Adam, dont la correction de biais.
**Prérequis :** Ex 19.5 · 0B (séries géométriques) · fiche §19.5.4 (encadré sur la moyenne mobile), §19.5.5 · **Parcours :** M

On note $m_t = \beta\,m_{t-1} + (1 - \beta)\,x_t$ avec $m_0 = 0$ et $0 < \beta < 1$. Réponds dans ta copie.

1. Montre par récurrence que $m_t = (1 - \beta)\sum_{k=0}^{t-1} \beta^k\,x_{t-k}$.
2. Calcule la somme des poids $(1 - \beta)\sum_{k=0}^{t-1}\beta^k$. Déduis-en que, si $x_t = c$ pour tout $t$, alors $m_t = (1 - \beta^t)\,c$, et que $\hat m_t = m_t / (1 - \beta^t)$ vaut exactement $c$.
3. Si les $x_t$ sont des variables aléatoires de même espérance $\mu$, montre que $\mathbb{E}[\hat m_t] = \mu$ pour tout $t \ge 1$.
4. **Mémoire effective.** Dans la somme infinie ($t \to \infty$), le poids d'une valeur d'âge $k$ est $(1 - \beta)\beta^k$. Montre que l'âge moyen pondéré vaut $\sum_k k\,(1-\beta)\beta^k = \beta/(1 - \beta)$. Pourquoi dit-on que la mémoire vaut « environ $1/(1 - \beta)$ » valeurs ? Donne-la pour $\beta = 0{,}9$, $0{,}99$ et $0{,}999$.
5. **Demi-vie.** Montre que le poids d'une valeur est divisé par 2 au bout d'environ $\ln 2 / (1 - \beta)$ pas (utilise $\ln\beta \approx -(1 - \beta)$ pour $\beta$ proche de 1). Pour $\beta_2 = 0{,}999$, à partir de quel pas $t$ la somme des poids $1 - \beta_2^t$ dépasse-t-elle $\frac12$ ?
6. **Le premier pas d'Adam.** Avec $\epsilon = 0$, montre que le premier pas d'Adam vaut exactement $\eta\,\mathrm{signe}(g_1)$, et que, sans correction de biais, il vaudrait $\eta\,\frac{1 - \beta_1}{\sqrt{1 - \beta_2}}\,\mathrm{signe}(g_1)$. Évalue ce facteur pour les valeurs par défaut. Que se passerait-il, sans correction, avec $\beta_1 = 0{,}9$ et $\beta_2 = 0{,}99999$ ?
7. Montre que, si tous les gradients passés sont égaux à $g \ne 0$, alors $\hat m_t / \sqrt{\hat v_t} = \mathrm{signe}(g)$ pour tout $t$ : Adam fait alors des pas de taille $\eta$, quelle que soit la taille de $g$.

### Ex 19.9 — Fermi : la mémoire des optimiseurs, du MLP au LLM 🧮 ★★ ⏱️ 15 min
**Objectif :** chiffrer ce que coûtent en mémoire les paramètres, les gradients et les états des optimiseurs.
**Prérequis :** ch. 16 (nombre de paramètres d'un MLP) · fiche §19.5.1, §19.5.5 · **Parcours :** R, M

On compte 1 Mo $= 10^6$ octets et 1 Go $= 10^9$ octets. Un nombre `float32` occupe 4 octets.

a) Un MLP 3072-512-256-10 (biais compris) classe les images de CIFAR-10. Son nombre de paramètres $P$.
b) La mémoire de ses paramètres en `float32`, en Mo (1 décimale).
c) Adam garde deux nombres par paramètre ($m$ et $v$), SGD avec momentum un seul, SGD sans momentum aucun. La mémoire des états d'Adam pour ce MLP, en Mo (1 décimale).
d) La mémoire totale pendant l'entraînement avec Adam, sans les activations : paramètres, gradients et états, en `float32`, en Mo (1 décimale).
e) Un modèle de langage de 7 milliards de paramètres, entraîné en précision mixte avec Adam. Par paramètre : le poids en `bfloat16` (2 octets), son gradient en `bfloat16` (2 octets), une copie maîtresse du poids en `float32` (4 octets), et $m$ et $v$ en `float32` (4 octets chacun). La mémoire totale de ces états, en Go.
f) Le même modèle avec SGD et momentum (un seul tampon en `float32` au lieu de deux), en Go.
g) Le nombre minimal de GPU de 80 Go qu'il faut pour loger les seuls états de e), sans les activations.

Dans ta copie : qu'est-ce que ce calcul oublie ? Cite deux techniques qui réduisent la mémoire de l'optimiseur.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 🛠️ Réflexion

### Ex 19.10 — Expliquer le momentum avec une bille qui roule 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer le momentum à quelqu'un qui n'a jamais vu de gradient.
**Prérequis :** fiche §19.5.1 · **Parcours :** R

Dans ta copie, en 5 lignes au plus et sans formule, explique à un ami : pourquoi une bille qui garde son élan descend mieux qu'une bille qui s'arrête à chaque pas, ce qui se passe sur un plateau, ce qui se passe au fond d'une vallée étroite, et le risque d'un élan trop fort. Puis relis-toi : as-tu évité de dire que le momentum « calcule un meilleur gradient » ?

### Ex 19.11 — Le coût énergétique d'une recherche d'hyperparamètres ⚖️ ★★ ⏱️ 20 min
**Objectif :** peser le coût, en calcul et en énergie, de la recherche du meilleur optimiseur, et en tirer des pratiques.
**Prérequis :** ch. 15 (recherches d'hyperparamètres) · fiche §19.6 · **Parcours :** R

Le livre conclut qu'on peut trouver le meilleur optimiseur par une recherche automatique (p. 869). Lis l'article d'E. Strubell, A. Ganesh et A. McCallum, [« Energy and Policy Considerations for Deep Learning in NLP »](https://aclanthology.org/P19-1355/) (ACL 2019, cinq pages), puis la réponse de D. Patterson et ses coauteurs, [« Carbon Emissions and Large Neural Network Training »](https://arxiv.org/abs/2104.10350) (2021, en particulier leur §4.1 et l'annexe D). Réponds dans ta copie.

1. Une recherche en grille sur 4 optimiseurs, 5 learning rates, 3 plannings et 3 graines : combien d'entraînements ? Si un entraînement dure 2 h sur un GPU de 300 W, combien de kWh pour toute la recherche ?
2. Quelle estimation Strubell et coll. donnent-ils pour la recherche d'architecture (NAS) qui a produit l'*Evolved Transformer* ? À quoi la comparent-ils (leur tableau 1) ?
3. Que répondent Patterson et coll. sur ce chiffre ? Quelle erreur d'hypothèse expliquent-ils ?
4. Quelles sont les trois recommandations de Strubell et coll. ? Laquelle concerne directement le choix d'un optimiseur ?
5. Propose trois pratiques qui réduisent le coût d'une recherche d'hyperparamètres sans la rendre malhonnête (pense au ch. 15 : recherche au hasard, successive halving, petits modèles, valeurs par défaut).
6. Qui peut se permettre une recherche de plusieurs centaines d'entraînements ? En quoi est-ce une question d'équité entre laboratoires, et pas seulement d'environnement ?

### Ex 19.12 — Kingma et Ba (2014) : l'algorithme Adam ligne par ligne 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a introduit Adam et relier chaque ligne de son algorithme à la fiche et à ta librairie.
**Prérequis :** Ex 19.5 · Ex 19.8 · fiche §19.5.5 · **Parcours :** M

Lis D. Kingma et J. Ba, « Adam: A Method for Stochastic Optimization », ICLR 2015 ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980), version 9 de 2017 ; les sections 1 à 3 et 6 suffisent). Réponds dans ta copie.

1. Recopie l'algorithme 1 avec les notations de la fiche. Quelles sont les valeurs par défaut de $\alpha$, $\beta_1$, $\beta_2$ et $\epsilon$ ? Lesquelles PyTorch a-t-il gardées ?
2. Les auteurs proposent une version « plus efficace » qui change l'ordre des calculs (fin du §2). Écris-la. Est-elle exactement équivalente à l'algorithme 1 ? Que devient $\epsilon$ ?
3. Le §2.1 borne la taille du pas effectif $\Delta_t$ (avec $\epsilon = 0$). Écris les deux cas. Calcule la borne pour les valeurs par défaut. En quoi cette borne rend-elle le choix de $\alpha$ plus facile à interpréter qu'avec la descente vanille ?
4. Pourquoi les auteurs appellent-ils $\hat m_t / \sqrt{\hat v_t}$ un **rapport signal sur bruit** ? Que se passe-t-il quand on approche d'un minimum ?
5. Le §3 justifie la correction de biais. Relie son calcul à ta preuve ∂ 19.8. Quel cas rend la correction indispensable, selon les auteurs (pense à $\beta_2$ proche de 1) ?
6. Quelles expériences le §6 présente-t-il (modèles et datasets) ? Lequel de ces modèles ressemble à ton MLP du ch. 18 ?
7. Le §7 présente AdaMax. Quelle norme remplace la moyenne des carrés ? Pourquoi n'y a-t-il pas besoin de corriger le biais de $u_t$ ?
8. L'analyse de convergence du §4 a été contestée par S. Reddi, S. Kale et S. Kumar (ICLR 2018), qui ont proposé AMSGrad. Que fait l'option `amsgrad=True` de `torch.optim.Adam` (lis son pseudo-code) ?

### Ex 19.13 — Lire la doc de torch.optim : correspondance livre ↔ PyTorch 🛠️ ★★ ⏱️ 25 min
**Objectif :** lire le pseudo-code officiel de `torch.optim` et le relier aux descriptions du livre et à ta librairie.
**Prérequis :** Ex 19.2 · fiche §19.5 · **Parcours :** C

Ouvre la documentation de PyTorch 2.11 : [`torch.optim`](https://docs.pytorch.org/docs/2.11/optim.html), puis les pages de [`SGD`](https://docs.pytorch.org/docs/2.11/generated/torch.optim.SGD.html), [`Adagrad`](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adagrad.html), [`RMSprop`](https://docs.pytorch.org/docs/2.11/generated/torch.optim.RMSprop.html), [`Adadelta`](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adadelta.html), [`Adam`](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html) et [`AdamW`](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html). Chaque page commence par un pseudo-code. Tu n'as rien à exécuter. Dans ta copie :

1. Remplis un tableau à six lignes (une par optimiseur) et cinq colonnes : le nom de l'idée dans le livre, le symbole du livre pour son coefficient de mémoire, le nom du paramètre dans PyTorch et sa valeur par défaut, la place d'$\epsilon$ (dans ou hors de la racine), et la ligne du pseudo-code où le weight decay intervient.
2. Dans le pseudo-code de `SGD`, que fait `dampening` ($\tau$) ? Quelle valeur faut-il pour retrouver ta classe `Momentum` ? Pourquoi `nesterov=True` exige-t-il `dampening=0` ?
3. Retrouve dans la page de `SGD` la note qui compare son momentum à celui de Sutskever et coll. Écris les deux mises à jour qu'elle compare, et dis quand elles coïncident.
4. Que fait le paramètre `lr_decay` d'`Adagrad` ? Avec quel planning du livre le rapprocher ?
5. `RMSprop` a deux options que ta classe n'a pas : `centered` et `momentum`. Décris ce que fait chacune, d'après le pseudo-code.
6. Dans le pseudo-code d'`Adam`, à quelle étape le `weight_decay` intervient-il ? Et dans celui d'`AdamW` ? Que fait l'option `decoupled_weight_decay` d'`Adam` ?
7. Que font les options `maximize`, `foreach` et `fused` ? Laquelle change le résultat, et lesquelles seulement la vitesse ?
8. Dans la page `torch.optim`, combien d'optimiseurs sont listés ? Lesquels ne sont pas dans le chapitre ? Choisis-en un et décris-le en deux lignes d'après sa page.

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 19.E1 — Adam ou SGD avec momentum : lequel choisir, et pourquoi ? 💼 ★★ ⏱️ 10 min
*Fiche §19.5.1, §19.5.5, §19.6 · parcours R*

« Pour entraîner un nouveau modèle, vous partez sur Adam ou sur SGD avec momentum ? Qu'est-ce qui guiderait votre choix ? »

### 19.E2 — Adam avec weight_decay contre AdamW 💼 ★★ ⏱️ 10 min
*Fiche, section AdamW · parcours R*

« Quelle est la différence entre `Adam(weight_decay=...)` et `AdamW` ? Pourquoi est-ce important ? »

### 19.E3 — À quoi servent le warmup et la décroissance en cosinus ? 💼 ★★ ⏱️ 10 min
*Fiche §19.3.3, section warmup et cosinus · parcours R*

« Dans la plupart des recettes d'entraînement de Transformers, on voit un warmup puis une décroissance en cosinus. À quoi sert chacun ? »

### 19.E4 — Qu'est-ce que le gradient clipping, et quand l'utiliser ? 💼 ★★ ⏱️ 10 min
*Fiche, section gradient clipping · parcours R*

« Qu'est-ce que le gradient clipping ? Par la norme ou par la valeur ? Dans quels cas est-il indispensable, et comment choisir le seuil ? »

### 19.E5 — Taille du mini-batch : quels compromis ? 💼 ★★ ⏱️ 10 min
*Fiche §19.4.1 à §19.4.3 · parcours R*

« Comment choisissez-vous la taille du mini-batch ? Que se passe-t-il si vous la multipliez par 8 sans rien changer d'autre ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch19_optimiseurs/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie (`mylearn/optim.py`). La partie 0 du notebook vérifie tes réponses courtes aux quiz, aux rappels et aux exercices ✏️ et 🧮 ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 19.14 | Pas constant sur une vallée : combien de pas pour le fond ? | 🔮 | ★ | 15 |
| 19.15 | Plannings : exponential_decay_lr, step_decay_lr, bold_driver_lr | 🔨 | ★ | 15 |
| 19.16 | SGD, Momentum et Nesterov à la manière de torch.optim | 🔨 | ★★ | 30 |
| 19.17 | Adagrad, RMSprop et Adadelta | 🔨 | ★★ | 30 |
| 19.18 | Adam et AdamW | 🔨 | ★★ | 30 |
| 19.19 | Warmup + cosinus et gradient clipping | 🔨 | ★★ | 25 |
| 19.20 | Trop de momentum : la bille quitte-t-elle la vallée ? | 🔮 | ★★ | 15 |
| 19.21 | Reproduire les figures du chapitre : pas constant, décroissant, momentum, Nesterov | 🎨 | ★★ | 30 |
| 19.22 | Chasse au bug : un Adam qui diverge | 🐛 | ★★ | 25 |
| 19.23 | Diagnostiquer des courbes de loss d'optimiseurs | 📈 | ★★ | 20 |
| 19.24 | Les solveurs de MLPClassifier : sgd, momentum, adaptive, adam | 📦 | ★★ | 25 |
| 19.25 | Batch, stochastique, mini-batch sur les lunes | 🔬 | ★★★ | 45 |
| 19.26 | Course d'optimiseurs sur la fonction de Rosenbrock | 🔬 | ★★★ | 45 |
| 19.27 | Huit optimiseurs sur MNIST, avec warmup, cosinus et clipping | 🔬 | ★★★ | 60 |
| 19.28 | Défi : 98 % sur MNIST en cinq epochs | 🏆 | ★★★ | 90 |
