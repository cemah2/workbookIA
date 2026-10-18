# 16 · Réseaux feed-forward — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels, des exercices ✏️ 16.1 à 16.5, des vérifications de ∂ 16.7 et de l'estimation 🧮 16.8 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 16.Q1 — Qu'est-ce qui rend un réseau « feed-forward » ?
a) **Vrai** : c'est la définition d'un graphe sans boucle (fiche §16.2). b) **B** : deux matrices de poids, (784, 128) et (128, 10), donc deux couches de neurones qui calculent ; la « couche d'entrée » ne fait que tenir les 784 features. c) **Faux** : dans un MLP, aucune arête ne relie deux neurones d'une même couche. d) **Vrai** : chaque couche calcule sur les sorties de la précédente.
**À retenir** : un MLP « 784-128-10 » se lit « 784 features, une couche cachée de 128 neurones, 10 neurones de sortie » ; on compte les couches par leurs matrices de poids.

### 16.Q2 — Vocabulaire des graphes : nœuds, arêtes, parents, descendants
a) **BC** : les arêtes B → E et C → E. b) **CDE** : A → C, puis C → D et C → E. c) **2** : A et B, où n'arrive aucune flèche. d) **Vrai** : le chemin B → C → D mène de B à D ; B est un ancêtre de D sans en être le parent. e) **Vrai** : de gauche à droite, ou de bas en haut.
**À retenir** : un parent est au bout d'une arête, un ancêtre au bout d'un chemin ; les nœuds sans parent sont les entrées du graphe.

### 16.Q3 — Vrai ou faux : la passe arrière crée une boucle de rétroaction
a) **Faux** : la passe arrière parcourt les arêtes du même DAG à l'envers, sans en ajouter ; aucun nœud ne reçoit sa propre sortie. b) **Vrai** : le livre réserve *feedback* aux boucles. c) **B** : le son du haut-parleur revient dans le micro, qui le renvoie au haut-parleur : une boucle, qui s'emballe.
**À retenir** : passe arrière ≠ rétroaction. La première est un parcours du graphe en sens inverse ; la seconde, une boucle du graphe.

### 16.Q4 — Pourquoi des graphes sans boucle, et où est l'exception des RNN ?
a) **Vrai** : deux chemins parallèles ne reviennent pas à leur départ ; ce n'est pas un cycle. b) **B** : on déroule le RNN, une copie par pas de temps. c) **Vrai** : en remontant de parent en parent dans un graphe fini, on finirait par repasser par un nœud déjà vu, ce qui ferait un cycle. d) **6** : dans un ordre topologique 1, 2, 3, 4, chaque nœud peut envoyer une arête à tous ceux qui viennent après lui, soit $3 + 2 + 1 = 6 = \binom{4}{2}$.
**À retenir** : un DAG a toujours une entrée (un nœud sans parent) et au plus $\binom{V}{2}$ arêtes ; un RNN déroulé est un DAG.

### 16.Q5 — Réseau synchrone ou asynchrone : reconnaître les deux
a) **SAAS** : la battue du chef et le gong sont des signaux communs ; les e-mails et le tableur réagissent dès qu'une valeur arrive. b) **Faux** : on règle la période sur le nœud le **plus lent**, pour que tous aient fini avant le battement suivant. c) **Vrai** : les mises à jour y tombent toutes ensemble, au battement.
**À retenir** : synchrone = un signal commun cadence tout ; asynchrone = chacun agit dès qu'il a du nouveau.

### 16.Q6 — Horodatage : quand un nœud doit-il recalculer sa sortie ?
a) **Vrai** : 12, 15 et 11 sont tous postérieurs à 10. b) **Faux** : l'entrée horodatée 9 a été envoyée avant le dernier calcul. c) **Faux** : l'horodatage est écrit par le nœud qui envoie, au moment où il calcule. d) **Vrai** : la règle compare les horodatages, pas les valeurs.
**À retenir** : un nœud qui attend toutes ses entrées recalcule quand toutes portent un horodatage postérieur à son dernier calcul, même si leurs valeurs n'ont pas bougé.

### 16.Q7 — Dans un programme, qui garantit l'ordre d'évaluation ?
a) **B** : l'ordre des instructions. b) **Vrai** : les opérations CUDA sont mises en file et rendent la main tout de suite ; `torch.cuda.synchronize()` attend leur fin. c) **Vrai**.
**À retenir** : dans un programme, écrire les couches dans l'ordre suffit ; sur GPU, l'exécution est asynchrone, et il faut synchroniser pour chronométrer.

### 16.Q8 — Un neurone, trois enfants : combien de poids ?
a) **3** : une arête par enfant, un poids par arête. b) **Faux** : la même valeur est multipliée par trois poids a priori différents (figure 16.4). c) **12** : $4 \times 3$. d) **Vrai**.
**À retenir** : un poids par arête ; entre deux couches denses de $n_{in}$ et $n_{out}$ neurones, $n_{in} n_{out}$ poids.

### 16.Q9 — Pourquoi pas le même poids partout au départ ?
a) **Vrai**. b) **B** : les neurones jumeaux reçoivent les mêmes corrections et le restent ; la couche vaut un seul neurone. c) **Vrai** : des poids entrants différents donnent des sommes pondérées différentes, même avec des biais égaux. d) **0,029** : $0{,}05/\sqrt{3} \approx 0{,}0289$.
**À retenir** : des poids au hasard, des biais nuls ; l'écart-type de $\mathcal{U}(-a, a)$ vaut $a/\sqrt{3}$.

### 16.Q10 — LeCun, Glorot, He : quel fan-in, pour quelle activation ?
a) **0,071** : $\sqrt{2/400}$. b) **0,110** : $\sqrt{6/500} \approx 0{,}1095$. c) **C** : He. d) **B** : Glorot. e) **Vrai** : $2/n_{in}$.
**À retenir** : LeCun $1/n_{in}$, Glorot $2/(n_{in} + n_{out})$, He $2/n_{in}$ ; écart-type = racine de la variance, borne uniforme = $\sqrt{3}$ fois l'écart-type.

<a id="rappels"></a>

## 🔁 Rappels

### 16.R1 — Ch. 15 : que contient un estimateur après fit() ?
a) **Faux** : `coefs_` est un attribut appris, créé par `fit`. b) **A**. c) **2** : la couche cachée et la couche de sortie. d) **(20, 50)**. e) **(50, 3)** : un neurone de sortie par classe.
**À retenir** : `coefs_[l]` a la forme `(n_in, n_out)`, comme dans ta librairie ; `n_layers_` compte l'entrée (ici 3).

### 16.R2 — Ch. 13 : combien de régions pour un arbre de profondeur 2 ?
a) **4**. b) **32** : $2^5$. c) **Vrai** : chaque question coupe par un plan perpendiculaire à un axe. d) **3** : la racine et ses deux enfants.
**Dans ta copie** : oui, un arbre de décision est un DAG (ses arêtes vont de la racine vers les feuilles, sans boucle), et chacun de ses nœuds a **un seul** parent, sauf la racine, qui n'en a pas. Dans un MLP, chaque neurone a autant de parents que la couche précédente a de neurones, et chaque neurone a plusieurs enfants : les chemins se croisent et se rejoignent.
**À retenir** : un arbre binaire complet de profondeur $d$ a $2^d$ feuilles et $2^d - 1$ nœuds internes.

### 16.R3 — Ch. 0B : formes compatibles dans un produit matriciel
a) **(3, 2)**. b) **erreur** : B a 2 colonnes, A a 3 lignes. c) **(3,)** : un vecteur 1-D donne un résultat 1-D. d) **120** : 30 éléments, 4 multiplications chacun. e) **Vrai**.
**À retenir** : $(a, b) \times (b, c) = (a, c)$, pour $a \times b \times c$ multiplications ; c'est le coût d'une couche dense sur un batch.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 16.1 — Compter les poids et les biais d'un MLP ✏️
**Réponses** : a) **80** · b) **146** · c) **17** · d) **163** · e) **669 706** · f) **419** · g) **Vrai**.
**Démarche** : les couches sont (10, 8), (8, 6) et (6, 3) : $80 + 48 + 18 = 146$ poids et $8 + 6 + 3 = 17$ biais. Pour e), $(784 \times 512 + 512) + (512 \times 512 + 512) + (512 \times 10 + 10) = 401\,920 + 262\,656 + 5\,130$. Pour f), $(160 + 16) + (192 + 12) + (36 + 3) = 419$. g) $400 \times 400 = 160\,000 = 4 \times 40\,000$.
**Dans ta copie** : la première couche porte 401 920 paramètres, 60 % du total ; une troisième couche cachée de 512 neurones ajouterait 262 656 paramètres (39 % de plus). Avant d'élargir ou d'approfondir, se demander si le dataset (60 000 images) justifie autant de paramètres, et mesurer le gain en validation.
**Erreurs fréquentes** : compter les biais de l'entrée ; additionner les tailles au lieu de les multiplier ; oublier une couche.
**Variante** : un MLP 20-30-1 sans biais, puis avec. (600 + 30 = **630** poids ; avec les biais, 630 + 31 = **661** paramètres.)

### Ex 16.2 — Passe avant à la main dans un réseau 2-2-1 ✏️
**Réponses** : a) **0,8** · b) **[2 ; −0,5]** · c) **2,5** · d) **1,870** · e) **1** · f) **Faux** · g) **[2,5 ; −2]**.
**Démarche** : a) $0{,}2 - 0{,}3 + 0{,}8 + 0{,}1$. b) $z_1 = 2 \times 0{,}5 + 1 \times 1 + 0 = 2$ et $z_2 = 2 \times (-1) + 1 \times 0{,}5 + 1 = -0{,}5$ : le neurone $k$ prend la colonne $k$ de $\mathbf{W}^{(1)}$. c) ReLU donne $(2 ;\ 0)$, et la sortie vaut $1{,}5 \times 2 - 2 \times 0 - 0{,}5 = 2{,}5$. d) $\tanh 2 \approx 0{,}9640$ et $\tanh(-0{,}5) \approx -0{,}4621$ : $1{,}5 \times 0{,}9640 - 2 \times (-0{,}4621) - 0{,}5 \approx 1{,}870$. e) $W^{(1)}_{21} = 1$ (ligne de $x_2$, colonne de $h_1$). f) $h_2 = 0$ : le poids de l'arête $h_2 \to$ sortie multiplie 0, et ne change rien pour cet exemple. g) Pour $(0 ;\ 1)$ : $\mathbf{z} = (1 ;\ 0{,}5) + (0 ;\ 1) = (1 ;\ 1{,}5)$, ReLU les garde, et la sortie vaut $1{,}5 - 3 - 0{,}5 = -2$.
**Dans ta copie** : f) dépend de l'exemple parce que ReLU met $h_2$ à zéro seulement quand $z_2 < 0$ ; pour $(0 ;\ 1)$, $h_2 = 1{,}5$ et le même poids compte. Sous forme matricielle : $\mathbf{X}$ (2, 2), $\mathbf{Z}^{(1)} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$ (2, 2), $\mathbf{A}^{(1)} = \mathrm{ReLU}(\mathbf{Z}^{(1)})$ (2, 2), $\hat{\mathbf{y}} = \mathbf{A}^{(1)}\mathbf{W}^{(2)} + b^{(2)}$ (2, 1).
**Erreurs fréquentes** : prendre une **ligne** de $\mathbf{W}$ pour un neurone (on trouve $z = (0 ;\ 3{,}5)$) ; oublier un biais ; arrondir $\tanh$ trop tôt.
**Variante** : avec ReLU, l'exemple $(1 ;\ 2)$. ($\mathbf{z} = (0{,}5 + 2 ;\ -1 + 1 + 1) = (2{,}5 ;\ 1)$, sortie $3{,}75 - 2 - 0{,}5 =$ **1,25**.)

### Ex 16.3 — Formes et broadcasting d'un mini-batch à travers les couches ✏️
**Réponses** : a) **(32, 784)** · b) **(784, 256)** · c) **(32, 256)** · d) **(64,)** · e) **(32, 10)** · f) **45 888** · g) **erreur** · h) **erreur** · i) **(32, 256)**.
**Démarche** : le batch suit (32, 784) → (32, 256) → (32, 64) → (32, 10). f) Les entrées : $32 \times (784 + 256 + 64) = 35\,328$ ; les sommes pondérées : $32 \times (256 + 64 + 10) = 10\,560$ ; total 45 888. g) NumPy aligne `(32, 256)` et `(32,)` par la droite : 256 contre 32, incompatibles. h) 784 colonnes contre 256 lignes. i) `W1.T` a la forme (784, 256).
**Dans ta copie** : avec 256 images et un `b1` de forme `(256,)`, `(256, 256) + (256,)` est **compatible** : le biais s'ajoute sans erreur, mais c'est un vecteur faux (un nombre par image, rangé sur l'axe des neurones), et le réseau calcule autre chose sans rien dire. Une erreur qui s'arrête tout de suite coûte une minute ; un bug silencieux peut coûter des jours.
**Erreurs fréquentes** : écrire `W1` en `(256, 784)` par habitude de PyTorch ; oublier le cache de la couche de sortie ; compter le cache par image au lieu du batch.
**Variante** : le même réseau, un batch de 128 images. (Le cache a $128 \times (1\,104 + 330) =$ **183 552** valeurs, quatre fois plus : il est proportionnel à la taille du batch.)

### Ex 16.4 — Ordre d'évaluation d'un graphe et détection de boucle ✏️
**Réponses** : a) **UVABCY** · b) **7** · c) **5** · d) **Vrai** · e) **Faux** · f) **3**.
**Démarche** : comptes de parents : U 0, V 0, A 2, B 1, C 2, Y 2. Prêts {U, V} → U ; A attend V ; prêts {V} → V, qui libère A et B ; prêts {A, B} → A (C attend B, Y attend C) ; prêts {B} → B, qui libère C ; puis C, puis Y. c) Les cinq ordres : UVABCY, UVBACY, VUABCY, VUBACY et VBUACY (B peut passer avant U, puisqu'il ne dépend que de V). d) B → C → Y existait déjà : Y → B ferme le cycle B → C → Y → B. e) Aucun chemin ne mène de Y à U : U → Y saute des couches vers l'avant, sans cycle. f) V → A → C → Y (ou V → B → C → Y) : 3 arêtes.
**Dans ta copie** : le tableau des étapes ci-dessus. Le rangement couche par couche (les entrées U, V ; puis A et B, dont tous les parents sont rangés ; puis C ; puis Y, qui attend C) donne ici le même ordre : sur ce graphe, la règle « le plus petit nom prêt » et la règle « couche par couche » coïncident. Sur un autre graphe, elles divergent (panneau (b) de la figure du §16.2 de la fiche).
**Erreurs fréquentes** : ranger Y juste après A (il attend aussi C) ; ranger A avant V ; oublier, en c), les ordres où B passe avant A ou avant U.
**Variante** : on ajoute l'arête B → Y. Combien d'ordres topologiques ? (Toujours **5** : Y venait déjà après C, donc après B.)

### Ex 16.5 — Simuler un réseau asynchrone avec horodatage ✏️
**Réponses** : a) **7** · b) **10** · c) **2** · d) **6** · e) **5**.
**Démarche** : après $t = 5$, les horodatages valent (2, 3, 5), tous postérieurs à 1 : S calcule $1 + 4 + 2 = 7$ et note 5. Après $t = 6$ et $t = 8$, R est encore horodaté 5 ; à $t = 9$, (6, 8, 9) sont tous postérieurs à 5 : S calcule $7 + 0 + 3 = 10$ et note 9. Le message de $t = 10$ ne renouvelle que P. La règle naïve calcule à chaque **changement** de valeur : $t = 2, 3, 6, 8, 9, 10$ (pas à $t = 5$, où R garde la valeur 2), soit 6 calculs, le dernier donnant $2 + 0 + 3 = 5$.
**Dans ta copie** : la sortie naïve de $t = 9$ (10) est celle de la règle du livre au même instant : à $t = 9$, les deux règles voient le même jeu complet d'entrées. Celle de e) (5) ne correspond à aucune sortie de la règle du livre, qui attend encore de nouvelles valeurs de Q et de R. La règle naïve mélange des valeurs d'« âges » différents (à $t = 2$, elle additionne le nouveau P et les anciens Q et R) et produit des sorties intermédiaires qui ne correspondent à aucun jeu complet d'entrées. Le message de $t = 5$ ($R = 2$, la même valeur) montre que comparer les valeurs ne suffit pas : R a bien été recalculé, et la règle du livre le compte comme nouveau.
**Erreurs fréquentes** : compter le message de $t = 5$ comme « rien de nouveau » dans la règle du livre ; recalculer dès qu'une entrée est nouvelle ; utiliser l'heure de réception.
**Variante** : un message $Q = 4$ arrive en plus à $t = 11$, puis $R = 1$ à $t = 12$. Que calcule S à $t = 12$ ? (Les horodatages (10, 11, 12) dépassent 9 : **7**, soit $2 + 4 + 1$.)

### Ex 16.6 — Symétrie : pourquoi une initialisation constante ne se brise jamais ∂
**Démarche** :
1. $h_1 = f(\mathbf{x}\cdot\mathbf{w}_1 + b_1)$ et $h_2 = f(\mathbf{x}\cdot\mathbf{w}_2 + b_2)$ : avec $\mathbf{w}_1 = \mathbf{w}_2$ et $b_1 = b_2$, les arguments de $f$ sont égaux pour tout $\mathbf{x}$, donc $h_1 = h_2$.
2. Après l'échange, le neurone 1 calcule ce que calculait le neurone 2 et inversement ; dans $\hat y = v_1 h_1 + v_2 h_2 + \sum_{k \ge 3} v_k h_k + c$, les deux premiers termes échangent leurs places, et la somme ne change pas. Toutes les prédictions sont les mêmes, donc la loss aussi : $\mathcal{L}(\tau(\theta)) = \mathcal{L}(\theta)$ pour tous les paramètres $\theta$, où $\tau$ est l'échange.
3. Au point jumeau $\theta$, notons $\theta + \varepsilon\,e_{v_1}$ le point où seul $v_1$ est augmenté de $\varepsilon$. L'échange l'envoie sur $\theta + \varepsilon\,e_{v_2}$ (puisque $\tau(\theta) = \theta$) : par 2, $\mathcal{L}(\theta + \varepsilon e_{v_1}) = \mathcal{L}(\theta + \varepsilon e_{v_2})$. Les taux d'accroissement sont égaux pour tout $\varepsilon$, donc leurs limites aussi : $\partial \mathcal{L}/\partial v_1 = \partial \mathcal{L}/\partial v_2$. Le même argument vaut pour $b_1, b_2$ et pour chaque $w_{j1}, w_{j2}$.
4. Un pas fait $\theta' = \theta - \eta \nabla \mathcal{L}(\theta)$. Les paramètres des deux neurones sont égaux, leurs gradients aussi : après le pas, ils sont encore égaux. Par récurrence, s'ils sont jumeaux après $t$ pas, ils le sont après $t + 1$.
5. Avec une même constante pour les poids entrants, les biais et les poids sortants, les $m$ neurones sont jumeaux deux à deux, et le restent (4). Alors $\hat y = \sum_k v_k h_k + c = m\,v_1 h_1 + c$ : c'est un réseau à un seul neurone caché de poids sortant $m\,v_1$ (et dont les paramètres évoluent comme ceux des jumeaux).
6. Avec des paramètres nuls, $z_k = 0$ et $h_k = \tanh(0) = 0$. $\partial \hat y/\partial v_k = h_k = 0$ et $\partial \hat y/\partial w_{jk} = v_k \tanh'(z_k)\,x_j = 0$ (car $v_k = 0$) ; seule $\partial \hat y/\partial c = 1$ n'est pas nulle. Le premier pas ne change que $c$ ; et comme les $v_k$ et les $w_{jk}$ restent nuls, les mêmes dérivées restent nulles aux pas suivants : seul le biais de sortie apprend, pour toujours.
7. Tirer les poids au hasard (des valeurs différentes pour chaque neurone, à une échelle adaptée, LeCun, Glorot ou He) ; ou rendre les neurones différents pendant l'entraînement, par exemple avec le dropout (ch. 20), qui éteint des neurones au hasard à chaque pas.
**Erreurs fréquentes** : affirmer « les gradients sont égaux parce que les neurones sont égaux » sans l'argument de symétrie de la loss ; oublier que les poids **sortants** doivent aussi être égaux (avec $v_1 \ne v_2$, les gradients des poids entrants diffèrent, et les neurones se séparent) ; croire que le résultat dépend de la rétropropagation (il vaut pour toute méthode qui traite les paramètres de la même façon, différences finies comprises).
**Variante** : avec $\mathbf{w}_1 = \mathbf{w}_2$, $b_1 = b_2$ mais $v_1 = 2v_2$, les neurones restent-ils jumeaux ? (**Non** : $\partial \hat y/\partial w_{jk} = v_k f'(z_k) x_j$ vaut deux fois plus pour le neurone 1, ses poids entrants bougent deux fois plus, et les neurones se séparent dès le premier pas.)

### Ex 16.7 — Conserver la variance : retrouver les règles de LeCun, Glorot et He ∂
**Réponses** : a) **1** · b) **0,0625** · c) **2** · d) **0,0625** · e) **0,122** · f) **57,7** · g) **0,001953**.
**Démarche** :
1. $\mathbb{E}[z] = \sum_j \mathbb{E}[w_j x_j] = \sum_j \mathbb{E}[w_j]\mathbb{E}[x_j] = 0$ (linéarité de l'espérance, puis indépendance des poids et des entrées, et $\mathbb{E}[w] = 0$). Deux produits différents sont décorrélés : $\mathbb{E}[w_j x_j\, w_k x_k] = \mathbb{E}[w_j]\,\mathbb{E}[w_k]\,\mathbb{E}[x_j x_k] = 0$ (les poids sont indépendants entre eux et des entrées, et centrés), même si les entrées sont corrélées ; la variance de la somme est donc la somme des variances. $\mathrm{Var}(w_j x_j) = \mathbb{E}[w_j^2 x_j^2] - \mathbb{E}[w_j x_j]^2 = \mathbb{E}[w_j^2]\,\mathbb{E}[x_j^2] = \mathrm{Var}(w)\,\mathbb{E}[x^2]$ (indépendance encore, et $\mathbb{E}[w^2] = \mathrm{Var}(w)$ pour un poids centré). D'où $\mathrm{Var}(z) = n\,\mathrm{Var}(w)\,\mathbb{E}[x^2]$.
2. $\mathbb{E}[x^2] = \mathrm{Var}(x) = 1$ : il faut $n\,\mathrm{Var}(w) = 1$, soit $\mathrm{Var}(w) = 1/n$.
3. $\mathrm{ReLU}(z)^2$ vaut $z^2$ si $z > 0$ et 0 sinon. Par symétrie, $\mathbb{E}[z^2 \mathbb{1}[z > 0]] = \mathbb{E}[z^2 \mathbb{1}[z < 0]]$, et leur somme vaut $\mathbb{E}[z^2] = \mathrm{Var}(z)$ : chacune vaut $\frac12 \mathrm{Var}(z)$. Pour une couche qui reçoit $x = \mathrm{ReLU}(z^{\text{préc}})$, $\mathrm{Var}(z) = n_{in}\,\mathrm{Var}(w) \times \frac12 \mathrm{Var}(z^{\text{préc}})$ : la variance se conserve si $\mathrm{Var}(w) = 2/n_{in}$, la règle de He.
4. La passe avant demande $n_{in}\mathrm{Var}(w) = 1$, la passe arrière $n_{out}\mathrm{Var}(w) = 1$ : les deux ne peuvent valoir 1 à la fois que si $n_{in} = n_{out}$. Glorot prend la moyenne des deux conditions : $\frac{n_{in} + n_{out}}{2}\mathrm{Var}(w) = 1$, soit $\mathrm{Var}(w) = 2/(n_{in} + n_{out})$.
5. $\mathbb{E}[w^2] = \int_{-a}^{a} \frac{w^2}{2a}\,dw = \frac{1}{2a}\cdot\frac{2a^3}{3} = \frac{a^2}{3}$ et $\mathbb{E}[w] = 0$ : $\mathrm{Var}(w) = a^2/3$, d'où $a = \sqrt{3\,\mathrm{Var}(w)}$ : $\sqrt{3/n_{in}}$ (LeCun), $\sqrt{6/(n_{in} + n_{out})}$ (Glorot), $\sqrt{6/n_{in}}$ (He).
Vérifications : a) $256 \times \frac{1}{256} \times 1 = 1$. b) $1/16$. c) $4/2 = 2$. d) $2/32$. e) $\sqrt{6/400} \approx 0{,}1225$. f) $1{,}5^{10} \approx 57{,}67$. g) Chacune des 9 couches suivantes divise la variance par 2 ($n_{in} \times \frac{1}{n_{in}} \times \frac12$) : $0{,}5^9 \approx 0{,}001953$.
**Erreurs fréquentes** : supposer $\mathbb{E}[x^2] = \mathrm{Var}(x)$ pour des sorties ReLU (elles ne sont pas centrées : c'est justement le facteur $\frac12$ qui compte $\mathbb{E}[x^2]$, pas la variance) ; confondre variance et écart-type ; en g), compter 10 divisions au lieu de 9.
**Variante** : un réseau ReLU de 10 couches initialisé en He mais avec $\mathrm{Var}(w) = 1{,}8/n_{in}$ (10 % trop petit). La variance de la dixième couche ? (Chaque couche multiplie par $0{,}9$ : $0{,}9^9 \approx$ **0,387** ; une petite erreur d'échelle devient grande avec la profondeur.)

### Ex 16.8 — Fermi : taille, mémoire et coût d'un MLP pour MNIST 🧮
**Réponses** : a) **2,7** Mo · b) **0,67** million · c) **40** milliards · d) **4** s · e) **12** s · f) **1,5** Mo.
**Démarche** : a) $669\,706 \times 4 \approx 2{,}68 \times 10^6$ octets. b) $784 \times 512 + 512 \times 512 + 512 \times 10 = 668\,672$. c) $668\,672 \times 60\,000 \approx 4{,}0 \times 10^{10}$. d) $4 \times 10^{10} / 10^{10}$. e) $3 \times 4$. f) $128 \times [(784 + 512) + (512 + 512) + (512 + 10)] = 363\,776$ nombres, $\times 4 \approx 1{,}46$ Mo.
**Dans ta copie** : optimistes. NumPy n'atteint $10^{10}$ multiplications par seconde que sur de gros produits matriciels bien alignés ; les petites opérations (activations, biais, copies, allocations) et le passage en `float64` (le défaut de NumPy) ralentissent tout ; il faut aussi compter plusieurs epochs, et la mémoire des gradients et de l'optimiseur. Un réseau convolutif réutilise chaque poids à chaque position : ses multiplications ne se comptent plus « une par poids », mais une par poids **et par position**, si bien qu'un petit réseau convolutif peut faire autant de calcul qu'un grand MLP, avec bien moins de paramètres.
**Erreurs fréquentes** : 8 octets au lieu de 4 ; compter les paramètres au lieu des multiplications en b) ; oublier le facteur 3 de l'entraînement ; ne garder que les `z` dans le cache.
**Variante** : le même réseau en `bfloat16`. (2 octets par nombre : **1,3** Mo de paramètres ; c'est l'un des formats des grands modèles actuels.)

<a id="reflexion"></a>

## 🗣️ 📄 Réflexion

### Ex 16.9 — Expliquer un réseau feed-forward avec une chaîne d'usine 🗣️
**Réponse modèle** : « Imagine une usine où une image entre d'un côté et une réponse sort de l'autre, par exemple “c'est un 7”. Entre les deux, des rangées de postes de travail : chaque poste reçoit les pièces de tous les postes de la rangée précédente, les mélange selon ses propres réglages, et passe le résultat à la rangée suivante, sans jamais rien renvoyer en arrière. Ces réglages sont les poids : des milliers de petits boutons, un sur chaque tapis qui relie deux postes. Apprendre, c'est tourner tous ces boutons, un peu à la fois, à partir de milliers d'exemples dont on connaît la bonne réponse, jusqu'à ce que l'usine se trompe rarement. La différence avec une vraie usine : personne ne décide de ce que fait chaque poste ; on ne choisit que l'organisation des rangées, et les tâches émergent des réglages appris. »
**Ce qui compte** : le sens unique ; des couches ; des poids sur les connexions ; apprendre = régler les poids à partir d'exemples ; une limite honnête de l'image.
**Erreurs fréquentes** : comparer un neurone à un vrai neurone (le chapitre parle de graphes, pas de biologie) ; dire que les données « reviennent » pour être corrigées (c'est la passe arrière de l'entraînement, pas une circulation des pièces) ; oublier les poids.
**Variante** : explique la même chose en une phrase à un informaticien. (« Une composition de fonctions affines paramétrées, entrecoupées de non-linéarités, évaluée dans l'ordre topologique d'un DAG, dont on ajuste les paramètres par descente de gradient. »)

### Ex 16.10 — Glorot et Bengio (2010) : pourquoi les réseaux profonds apprenaient mal 📄
**Réponse modèle** :
1. La question : pourquoi la descente de gradient depuis une initialisation au hasard marche-t-elle si mal sur les réseaux profonds, alors que des méthodes de pré-entraînement couche par couche y arrivent ? Les auteurs entraînent des réseaux de une à cinq couches cachées de 1 000 neurones, avec la sigmoïde, tanh et la softsign, sur un dataset synthétique d'images de formes (Shapeset-3 × 2), sur MNIST, CIFAR-10 et un petit extrait d'ImageNet.
2. Avec la sigmoïde et l'initialisation $\mathcal{U}(\pm 1/\sqrt{n_{in}})$, la **dernière** couche cachée sature très vite vers 0 et y reste longtemps, avant d'en sortir lentement : l'apprentissage stagne. Leur explication : au début, la couche de sortie (une softmax) apprend d'abord en s'appuyant sur ses biais, et pousse les sorties de la couche du dessous vers 0, où la sigmoïde a une pente faible et ne transmet plus guère de gradient ; une activation centrée en 0, comme tanh, n'a pas ce problème au même degré, mais montre sa propre saturation, couche après couche à partir de la première.
3. L'initialisation normalisée tire $\mathcal{U}\big(\pm\sqrt{6/(n_{in} + n_{out})}\big)$. Le raisonnement : on veut que la variance des activations se conserve en passe avant ($n_{in}\mathrm{Var}(w) = 1$) et celle des gradients en passe arrière ($n_{out}\mathrm{Var}(w) = 1$), comme dans ∂ 16.7 ; on prend le compromis $\mathrm{Var}(w) = 2/(n_{in} + n_{out})$. L'initialisation critiquée, $\mathcal{U}(\pm 1/\sqrt{n_{in}})$, a la variance $1/(3n_{in})$ : dans la zone linéaire, elle divise la variance des activations par 3 à chaque couche, et celle des gradients par rapport aux activations rétrécit en descendant vers l'entrée (l'article note que la variance des gradients des **poids**, elle, reste à peu près constante d'une couche à l'autre au départ).
4. Des histogrammes des activations et des gradients couche par couche, à différents moments de l'entraînement ; la moyenne et l'écart-type des activations au fil de l'entraînement ; les valeurs singulières de la Jacobienne entre couches (plus proches de 1 avec l'initialisation normalisée : environ 0,8 contre 0,5). 📈 16.22 refait les histogrammes, 🔬 16.23 les écarts-types couche par couche.
5. L'article ne traite ni ReLU (étudiée l'année suivante par Glorot, Bordes et Bengio, et pour laquelle He et coll. (2015) ont refait le calcul, avec le facteur $\frac12$), ni les couches de normalisation (2015), ni les connexions résiduelles (2015), qui ont rendu possibles des réseaux de centaines de couches ; ses réseaux sont petits et ses datasets aussi. L'idée centrale, conserver la variance, est restée, et c'est elle qui explique les défauts des bibliothèques (encadré 🕰️ de la fiche). On remarque d'ailleurs que le défaut de `nn.Linear` dans PyTorch, $\mathcal{U}(\pm 1/\sqrt{n_{in}})$, est exactement l'initialisation « standard » que l'article critique.
**Ce qui compte** : distinguer le diagnostic (saturation, variances qui changent avec la profondeur) de la proposition (l'initialisation normalisée) ; relier le raisonnement à ∂ 16.7 ; situer l'article (2010, avant ReLU, la batchnorm et ResNet).
**Erreurs fréquentes** : attribuer à l'article l'initialisation de He ; dire que l'article « résout » l'apprentissage profond ; confondre la borne uniforme et l'écart-type.
**Variante** : quelle variance des poids donnerait l'initialisation « normalisée » pour une couche ReLU, si l'on refaisait le compromis avec le facteur $\frac12$ ? (La passe avant demande $\mathrm{Var}(w) = 2/n_{in}$, la passe arrière $2/n_{out}$ : le compromis donne **$4/(n_{in} + n_{out})$**. He et coll. remarquent qu'il suffit en fait de satisfaire l'une des deux conditions : quand les largeurs des couches restent du même ordre, l'autre signal ne dérive pas exponentiellement.)

<a id="entretien"></a>

## 💼 Entretien

### 16.E1 — Pourquoi ne pas initialiser tous les poids à zéro ?
**Réponse modèle en 60 secondes** : « Parce que les neurones d'une même couche resteraient identiques. S'ils ont les mêmes poids, ils calculent la même chose pour chaque exemple, reçoivent exactement le même gradient, et sont mis à jour de la même façon : ils restent jumeaux pour toujours, et une couche de 512 neurones ne vaut qu'un neurone. C'est vrai de n'importe quelle constante. Zéro est même pire : dans un réseau tanh, toutes les sorties cachées sont nulles, les gradients des poids aussi, et seul le biais de sortie apprend. On tire donc les poids au hasard, à une échelle qui dépend du nombre d'entrées et de l'activation, He pour ReLU, Glorot pour tanh, pour que la variance des activations reste stable d'une couche à l'autre. Les biais, eux, peuvent partir de zéro : la symétrie est déjà brisée par les poids. »
**Relances possibles** : « Et un tout petit bruit autour d'une constante ? » (la symétrie se brise, mais les neurones mettent longtemps à se séparer ; mieux vaut une vraie initialisation) · « Y a-t-il des poids qu'on initialise à zéro ? » (les biais ; et parfois la dernière couche d'un bloc résiduel, pour qu'il commence par ne rien changer) · « Pourquoi pas des poids très grands ? » (saturation ou explosion des activations dans un réseau profond).

### 16.E2 — He ou Glorot : quelle initialisation pour quelle activation ?
**Réponse modèle en 60 secondes** : « He pour ReLU et ses variantes, Glorot pour tanh et la sigmoïde. Les deux viennent du même calcul : une somme pondérée de $n_{in}$ entrées a une variance $n_{in}$ fois la variance des poids fois le carré moyen des entrées ; pour que cette variance ne grandisse ni ne rétrécisse de couche en couche, il faut une variance des poids en $1/n_{in}$. ReLU met à zéro la moitié des valeurs : He double la variance, $2/n_{in}$. Glorot cherche aussi à conserver la variance des gradients en passe arrière, qui dépend de $n_{out}$, et prend le compromis $2/(n_{in} + n_{out})$. Avec la mauvaise règle, un réseau profond voit ses activations s'éteindre ou exploser exponentiellement avec la profondeur. Attention à PyTorch : `nn.Linear` n'utilise pas He par défaut, mais une uniforme de borne $1/\sqrt{n_{in}}$ ; pour He, on appelle `nn.init.kaiming_normal_`. Et je vérifie toujours en mesurant l'écart-type des activations couche par couche avant d'entraîner. »
**Relances possibles** : « Et pour GELU ou SiLU ? » (proches de ReLU : He, ou les défauts des bibliothèques, avec de la normalisation) · « Et dans un Transformer ? » (souvent une constante, comme $\mathcal{N}(0 ;\ 0{,}02^2)$, avec la layer norm et une réduction des couches résiduelles) · « La batchnorm rend-elle l'initialisation inutile ? » (elle la rend moins critique, pas inutile).

### 16.E3 — Calculer de tête le nombre de paramètres d'un MLP
**Réponse modèle en 60 secondes** : « Chaque couche a $n_{in} \times n_{out}$ poids plus $n_{out}$ biais. Pour 1000-500-100-10 : 500 000 plus 500, puis 50 000 plus 100, puis 1 000 plus 10, soit 551 610 paramètres, dont 90 % dans la première couche. En float32, 4 octets chacun : environ 2,2 Mo. Doubler la taille du batch ne change aucun paramètre : ce qui double, c'est la mémoire des activations, celles que la passe avant garde pour la rétropropagation, et le calcul par pas. Pendant l'entraînement, il faut aussi compter les gradients, autant que de paramètres, et l'état de l'optimiseur, deux fois plus avec Adam : environ quatre fois la taille du modèle. »
**Relances possibles** : « Et en float16 ? » (2 octets : 1,1 Mo) · « Quelle couche réduire pour alléger ? » (la première, la plus lourde) · « Combien de multiplications par image ? » (551 000 environ, une par poids).

### 16.E4 — Feed-forward, récurrent, graphe acyclique : expliquer la différence
**Réponse modèle en 60 secondes** : « Un réseau feed-forward est un graphe orienté acyclique : l'information va des entrées vers les sorties, sans jamais revenir en arrière, et la sortie ne dépend que de l'entrée du moment ; on calcule les couches dans l'ordre. Un réseau récurrent a une boucle : à chaque pas de temps, une couche reçoit, en plus de l'entrée, son propre état du pas précédent ; il garde une mémoire, et sa sortie dépend de toute la séquence. Pour l'entraîner, on le déroule dans le temps : une copie du réseau par pas de temps, avec les mêmes poids, et le dessin redevient un DAG. La rétropropagation, elle, ne crée aucune boucle : elle parcourt le même graphe à l'envers pour calculer les gradients. Exemples : un MLP qui classe une image d'un coup ; un RNN ou un LSTM qui lit une phrase mot par mot. »
**Relances possibles** : « Un Transformer est-il récurrent ? » (non : il traite toute la séquence d'un coup, avec de l'attention ; en génération, on le rappelle mot par mot, mais le réseau lui-même est feed-forward) · « Une connexion résiduelle crée-t-elle une boucle ? » (non, elle saute des couches vers l'avant) · « Pourquoi les RNN sont-ils plus durs à entraîner ? » (les gradients traversent de nombreux pas de temps et s'évanouissent ou explosent, ch. 22).

<a id="notebook"></a>

## Notebook

### Ex 16.11 — Prédire les formes des sorties avant d'exécuter 🔮
**Réponses** : a) **(64, 128)** · b) **(128,)** · c) **erreur** · d) **erreur** · e) **(1, 10)** · f) **(128, 128)**.
**Démarche** : a) (64, 784) × (784, 128), plus un biais (128,) aligné sur les colonnes. b) Un vecteur 1-D donne un résultat 1-D. c) `W1_11.T` a la forme (128, 784) : 784 colonnes contre 128 lignes. d) `(64, 128) + (128, 1)` : alignées par la droite, 128 se compare à 1 (diffusable) et 64 à 128 (incompatible). e) `X_11[:1]` garde l'axe des exemples : (1, 784) → (1, 128) → (1, 10). f) `(128, 128) + (128, 1)` est compatible : la colonne se recopie sur les 128 colonnes, et le biais du neurone $i$ s'ajoute à l'**image** $i$.
**Dans tes notes** : d) échoue parce que le batch (64) n'a pas la taille de la couche (128) ; f) réussit parce qu'il l'a. Un test sûr utilise des tailles toutes différentes (batch, entrées, neurones), et compare une image seule à la même image dans un batch.
**Erreurs fréquentes** : prédire `(1, 128)` pour b) ; croire qu'un biais en colonne lève toujours une erreur ; oublier que `X[:1]` garde deux dimensions.
**Variante** : la forme de `(X_11 @ W1_11).sum(axis=0)`. (**(128,)** : une somme par neurone, sur les 64 images.)

### Ex 16.12 — Compter les paramètres par programme : count_parameters 🔨
**Réponses** : les exemples de la docstring donnent `[37, 28, 235146]` ; trois réseaux ont le même compte que `torch.nn.Linear` ; les 14 tests passent.
**Démarche** : la référence (`solutions/mylearn_ref/nn/layers.py`) valide les tailles (au moins 2, des entiers Python ou NumPy, pas des booléens, ≥ 1), puis additionne `n_in * n_out + (n_out if bias else 0)` sur les couches voisines et rend `int(...)`.
**Dans tes notes** : `True` vaut 1 pour Python ; une taille `True` vient presque toujours d'une erreur (un test passé à la place d'une taille), et la refuser la révèle tout de suite.
**Erreurs fréquentes** : rendre un `np.int64` ; accepter `[784]` (une seule taille, pas de couche) ; accepter `2.5` ou `True` ; compter un biais pour l'entrée.
**Variante** : `count_parameters([3, 3, 3, 3], bias=False)`. (**27** : trois couches de $3 \times 3$ poids.)

### Ex 16.13 — La couche dense : dense_forward 🔨
**Réponses** : les exemples de la docstring donnent `[[2.0, 2.0, 4.0]]` et `[2.0, 2.0, 3.0]` ; 50 images et 30 neurones donnent une sortie (50, 30) en `float64`, égale à celle de `torch.nn.functional.linear(x, W.T, b)` ; chaque image donne la même sortie seule ou dans le batch ; les 12 tests passent.
**Démarche** : la référence convertit `x` et `W` en `float64`, vérifie `W.ndim == 2`, `x.ndim in (1, 2)` et `x.shape[-1] == W.shape[0]`, calcule `z = x @ W`, puis, si `b` n'est pas `None`, vérifie `b.shape == (W.shape[1],)` et ajoute `b`. Les messages d'erreur citent les deux formes.
**Dans tes notes** : NumPy n'arrête que les formes **incompatibles** ; un biais `(n_out, 1)` est compatible dès que le batch a $n_{out}$ exemples (ou un seul), et la somme est fausse sans erreur. Vérifier la forme exacte transforme ce bug silencieux en une erreur claire.
**Erreurs fréquentes** : `x @ W.T` ; un biais accepté en colonne ; un résultat `int64` quand `x` et `W` sont entiers ; modifier `x` en place (`x += b`).
**Variante** : `dense_forward([[1, 2], [3, 4]], np.eye(2), np.array([0.5, -0.5]))`. (**[[1,5 ; 1,5], [3,5 ; 3,5]]** : la matrice identité recopie les entrées, puis chaque colonne reçoit son biais.)

### Ex 16.14 — Dix façons d'initialiser : init_weights 🔨
**Réponses** : les exemples de la docstring passent ; pour une couche de 600 entrées et 400 neurones, les six méthodes ont l'écart-type de leur règle à 0,2 % près (par exemple 0,0408 pour LeCun, 0,0447 pour Glorot, 0,0577 pour He), et les uniformes restent dans leur borne ; les 54 tests passent.
**Démarche** : la référence garde un dictionnaire des méthodes (avec les alias `xavier_*`), vérifie la méthode puis les tailles, crée `rng = np.random.default_rng(rng)`, et tire **un** tableau `(n_in, n_out)` : `rng.uniform(-a, a, size)` avec $a = \sqrt{3/n_{in}}$, $\sqrt{6/(n_{in} + n_{out})}$ ou $\sqrt{6/n_{in}}$, ou `rng.normal(0.0, s, size)` avec $s = \sqrt{1/n_{in}}$, $\sqrt{2/(n_{in} + n_{out})}$ ou $\sqrt{2/n_{in}}$.
**Dans tes notes** : l'écart-type mesuré sur $N$ tirages normaux fluctue d'environ $1/\sqrt{2N}$ en valeur relative : 0,07 % pour un million de poids, 7 % pour cent ; un test sur 10 × 10 poids échouerait souvent par malchance. Un générateur passé en argument permet à l'appelant de choisir le hasard et de le partager entre les couches.
**Erreurs fréquentes** : `rng.normal(0, 2 / n_in)` (la variance prise pour un écart-type) ; une borne uniforme égale à l'écart-type (il manque $\sqrt{3}$) ; `np.random.normal` (l'état global, qui ignore `rng`) ; une graine fixée à l'intérieur.
**Variante** : la borne de `"he_uniform"` pour 600 entrées. (**0,1** : $\sqrt{6/600}$ ; la plus grande valeur tirée au tableau, 0,1000, la touche presque.)

### Ex 16.15 — Reproduire les figures 16.5 et 16.6, puis comparer Glorot et He 🎨
**Réponses** : a) **10** · b) **23,9** · c) **1,414** ; la figure montre un plateau de hauteur 10, une cloche de pic 23,9, et deux histogrammes en cloche dont celui de He est $\sqrt{2}$ fois plus large.
**Démarche** : la densité uniforme vaut $1/(2a)$, pour une aire de 1 ; le pic normal $1/(\sigma\sqrt{2\pi})$ avec $\sigma = 0{,}05/3$ ; pour une couche carrée, Glorot donne $\mathrm{Var}(w) = 1/n$ et He $2/n$. Avec `density=True`, l'histogramme se superpose à la densité.
**Dans tes notes** : la probabilité de dépasser $|w| > 0{,}05 = 3\sigma$ vaut environ 0,27 % : petite, pas nulle. Le troisième panneau : He est plus large ($\sqrt{2}$ fois l'écart-type) pour compenser la moitié des valeurs que ReLU met à zéro ; c'est He qui convient à une couche ReLU.
**Erreurs fréquentes** : tracer `np.histogram` sans `density=True` (les comptes, pas une densité) ; prendre $\sigma = 0{,}05$ pour la loi de la figure 16.6 ; lire b) sur l'histogramme au lieu de la formule.
**Variante** : la hauteur de la densité de `"he_uniform"` pour une couche de 600 entrées. (**5** : la borne vaut 0,1, et $1/(2 \times 0{,}1) = 5$.)

### Ex 16.16 — Ordre topologique d'un graphe : topological_order 🔨
**Réponses** : les exemples de la docstring donnent `['x', 'h1', 'h2', 'y']` et `['a', 'b', 'c']` ; sur 30 graphes au hasard, chaque nœud apparaît une fois, après ses parents ; un cycle lève une `ValueError` ; les 11 tests passent.
**Démarche** : la référence rassemble tous les nœuds (clés et enfants), compte les parents, met les nœuds sans parent dans un tas (`heapq`), puis sort le plus petit, l'ajoute à l'ordre et décrémente ses enfants ; s'il reste des nœuds non rangés, le graphe a un cycle.
**Dans tes notes** : $O(V + E)$ opérations, plus $O(\log V)$ par nœud pour le tas. Au ch. 18, la rétropropagation parcourt l'ordre topologique **à l'envers** : un nœud transmet son gradient à ses parents une fois qu'il a reçu ceux de tous ses enfants.
**Erreurs fréquentes** : oublier les nœuds qui ne sont que des enfants ; un parcours en largeur au lieu du « plus petit prêt » (test `smallest_ready_node_not_breadth_first`) ; une boucle infinie sur un graphe avec cycle ; modifier le dictionnaire reçu.
**Variante** : `topological_order({"c": ["a"], "b": ["a"]})`. (**['b', 'c', 'a']** : b et c sont prêts, b est le plus petit.)

### Ex 16.17 — Empiler les couches : init_mlp et mlp_forward 🔨
**Réponses** : l'exemple de `init_mlp` donne les formes `[(2, 4), (4, 4), (4, 1)]` et 37 paramètres ; celui de `mlp_forward` donne `[[0.5]]` et un cache dont la première somme pondérée vaut `[[1.5, -2.0]]` ; le MLP 784-128-64-10 rend des logits (256, 10) égaux à ceux d'un `torch.nn.Sequential` qui a reçu les poids transposés ; l'écart-type de $z$ vaut 1,29, 0,80 et 0,70 dans les trois couches ; les 23 tests passent.
**Démarche** : `init_mlp` crée un seul générateur et appelle `init_weights` couche après couche, avec des biais `np.zeros(n_out)`. `mlp_forward` boucle sur les couches : `z = dense_forward(a, W, b)`, `cache.append((a, z))`, puis l'activation de la couche (cachée ou de sortie, l'identité si `None`).
**Dans tes notes** : la rétropropagation (ch. 18) aura besoin de la dérivée de l'activation en $z$ et de l'entrée `a_prev` de chaque couche : le cache garde exactement ces deux tableaux. PyTorch range ses poids en `(n_out, n_in)`, d'où la transposée avant la copie.
**Erreurs fréquentes** : appliquer l'activation à la couche de sortie ; garder dans le cache l'activation au lieu de $z$ ; un nouveau générateur par couche (toutes les couches carrées identiques) ; un cache qui garde des références à des tableaux modifiés ensuite.
**Variante** : avec l'exemple de la docstring, la sortie sans activation (`activation=None`). (La couche cachée garde $(1{,}5 ;\ -2)$, la sortie vaut $1{,}5 - 2 - 1 =$ **−1,5**.)

### Ex 16.18 — Relire un MLPClassifier de scikit-learn avec ton mlp_forward 📦
**Réponses** : a) **50 890** · b) **(784, 64)** · c) **1,0** · d) **0,0** · e) **0,939** · f) **0,77** (0,7685).
**Démarche** : `params_18 = list(zip(CLF_18.coefs_, CLF_18.intercepts_))`, puis `mlp_forward(X_TEST_18, params_18, activation=relu, output_activation=softmax_18)`. Ta passe avant refait exactement le calcul de scikit-learn : les mêmes prédictions et les mêmes probabilités (ici à l'identique).
**Dans tes notes** : les poids seuls ne suffisent pas : avec tanh au lieu de ReLU, un quart des prédictions changent. Pour refaire les prédictions d'un réseau, il faut son architecture complète (tailles, activations, sortie) : on sauvegarde un modèle avec sa configuration ou son code. La softmax est croissante et garde l'ordre des logits : `argmax` des logits suffit pour `predict`.
**Erreurs fréquentes** : oublier les biais en a) ; transposer `coefs_` ; comparer e) à `CLF_18.predict` au lieu de `Y_TEST_18` ; une softmax non stable (`np.exp` de grands logits), sans conséquence ici mais risquée.
**Variante** : la part des images de test dont la probabilité maximale dépasse 0,9. (Une ligne : `np.mean(proba_18.max(axis=1) > 0.9)` ; elle mesure la confiance du réseau, et se compare à l'accuracy pour juger sa calibration, ch. 15.)

### Ex 16.19 — Initialisation constante : les neurones restent-ils jumeaux ? 🔮
**Réponses** : a) **True** · b) **C** · c) **B**.
**Démarche** : partis de 0,5, les quatre neurones restent jumeaux, même avec des différences finies (∂ 16.6) : en arithmétique exacte, chaque neurone reçoit la même correction et l'écart reste nul. Sur ordinateur, `spread_19` reste sous $10^{-5}$ : 0 si ta couche calcule `x @ W` ou `np.einsum`, environ $2 \times 10^{-6}$ si elle additionne ses produits par diffusion ou dans une boucle. L'arrondi d'une somme en virgule flottante dépend de l'ordre des additions : perturber le poids sortant du neurone 1 ou celui du neurone 3 ne change pas le même terme de la somme, et les deux arrondis peuvent différer du dernier chiffre (de l'ordre de $10^{-16}$) ; comme le point symétrique est instable (voir c)), ces écarts grandissent un peu en 200 pas, sans approcher $10^{-5}$. Le réseau ne vaut donc qu'un neurone caché, et ne peut pas séparer un « ou exclusif » ; sa MSE finale, 0,244, est proche de celle d'une prédiction constante (0,25) et environ 13 fois celle du départ de Glorot (0,019). Avec un bruit de $10^{-4}$, l'écart entre neurones grandit (de 0,0002 à 0,33 en 200 pas) : le point symétrique est instable ; mais la MSE (0,233) reste proche de celle du départ constant.
**Dans tes notes** : quatre jumeaux équivalent à un neurone caché, qui ne trace qu'une frontière ; un « ou exclusif » en demande au moins deux. En c), la séparation vient de la petite asymétrie de départ, que l'entraînement amplifie ; pour apprendre vraiment, il faut des poids différents dès le départ, à l'échelle d'une règle d'initialisation.
**Erreurs fréquentes** : croire que les différences finies traitent deux jumeaux différemment (elles perturbent chaque paramètre de la même façon ; seuls les arrondis, de l'ordre de $10^{-16}$, peuvent les distinguer) ; à l'inverse, exiger un écart exactement nul : selon l'ordre dans lequel ta couche additionne ses produits (`x @ W`, `np.einsum`, une boucle), l'écart final va de 0 à $2 \times 10^{-6}$ environ, et le point symétrique, instable, amplifie ces arrondis ; croire qu'un bruit infime suffit à apprendre vite.
**Variante** : le départ constant 0,5, mais des poids **sortants** différents (0,1 ; 0,2 ; 0,3 ; 0,4). Les neurones restent-ils jumeaux ? (**Non** : leurs poids entrants reçoivent des gradients proportionnels à leurs poids sortants, et se séparent dès le premier pas.)

### Ex 16.20 — Chasse au bug : transposée, biais mal diffusé, variance prise pour un écart-type 🐛
**Réponses** : les trois points passent une fois les trois erreurs corrigées ; l'écart-type des sommes pondérées reste entre 0,64 et 0,86 sur 10 couches ReLU initialisées par `init_20` (contre 0,12 puis 0,02, 0,005… avec la première erreur).
**Démarche** : 1) `rng.normal(0.0, 2 / n_in, ...)` donne l'écart-type $2/n_{in}$ au lieu de $\sqrt{2/n_{in}}$ : `np.sqrt(2 / n_in)`. 2) `b.reshape(-1, 1)` aligne le biais sur les exemples : `+ b`. 3) `a @ W.T` utilise les poids sortants d'une entrée au lieu des poids entrants d'un neurone : `a @ W`.
**Dans tes notes** : le réseau carré (64 entrées, 64 neurones par couche) rend `W.T` aussi compatible que `W` ; le batch de 64 images rend `(64, 64) + (64, 1)` compatible ; et des biais nuls à l'initialisation rendent l'erreur de biais invisible (la vérification tire des biais au hasard). Seule, la première erreur divise l'écart-type par environ 5,7 à chaque couche (évanouissement) ; la deuxième mélange les exemples d'un batch (la prédiction d'une image dépend de sa position) ; la troisième calcule une autre fonction, que l'entraînement pourrait en partie compenser, mais qui ne correspond plus aux poids sauvegardés.
**Erreurs fréquentes** : corriger `W.T` en transposant aussi `W` à l'initialisation (les formes redeviennent fausses ailleurs) ; remplacer `b.reshape(-1, 1)` par `b.reshape(1, -1)`, qui marche, mais complique sans raison.
**Variante** : si la vérification tirait des biais nuls, son point 2 attraperait-il encore l'erreur de biais ? (**Oui, par la forme** : dans le batch, `b.reshape(-1, 1)` n'ajouterait que des zéros et les valeurs seraient justes, mais une image seule, `(1, 64) + (64, 1)`, donnerait un tableau `(64, 64)` au lieu de `(1, 64)`.)

### Ex 16.21 — Rendre l'aléatoire reproductible et le prouver avec pytest 🛠️
**Réponses** : quatre tests passent sur `init_mlp_ok` ; `init_mlp_bug_1` (graine fixée à l'intérieur), `init_mlp_bug_2` (l'état global `np.random`) et `init_mlp_bug_3` (un nouveau générateur de même graine pour chaque couche) sont tous attrapés.
**Démarche** : un test par propriété du contrat. Même graine, mêmes paramètres : attrape le bug 2. Deux graines, deux tirages : attrape le bug 1. Deux couches de même forme, deux tirages : attrape le bug 3. Indépendance vis-à-vis de `np.random.seed` : attrape de nouveau le bug 2 (le code de la solution est dans `05_solutions.ipynb`).
**Dans tes notes** : le bug 1 ferait croire qu'un résultat est robuste au hasard (il n'y a qu'un tirage possible) ; le bug 2 rendrait les expériences impossibles à refaire et dépendantes de l'ordre des cellules ; le bug 3 donnerait les mêmes poids à toutes les couches carrées, une symétrie de plus entre couches.
**Erreurs fréquentes** : un test qui compare deux appels **sans** graine (il échoue sur la version juste) ; un test qui teste des valeurs précises (il dépend de la version de NumPy) ; oublier de vérifier les biais.
**Variante** : un quatrième bug, `rng = np.random.default_rng()` qui ignore l'argument. Quel test l'attrape ? (**Même graine, mêmes paramètres** : deux appels avec la même graine donnent des tirages différents.)

### Ex 16.22 — Lire des histogrammes d'activations couche par couche 📈
**Réponses** : a) **RPQ** · b) **Vrai** · c) **B** · d) **Faux**.
**Démarche** : Q montre des pics en $\pm 1$ à toutes les couches : c'est $\mathcal{N}(0 ;\ 1)$ (87 % des activations de la couche 6 sont saturées). R rétrécit le plus vite (écart-type 0,72 à la couche 1, 0,14 à la couche 6) : c'est l'échelle fixe 0,05, de gain 0,64 par couche. P rétrécit lentement (de 0,68 à 0,29) : c'est Glorot.
**Dans tes notes** : avec $\mathcal{N}(0 ;\ 1)$, une somme de 784 produits a un écart-type d'environ 28, et tanh sature dès la couche 1. Avec l'échelle fixe, l'écart-type est multiplié par $\sqrt{0{,}64} = 0{,}8$ environ par couche. Glorot suppose une activation linéaire près de 0 ; tanh écrase un peu chaque valeur ($|\tanh z| < |z|$), et chaque couche perd un peu de variance.
**Erreurs fréquentes** : prendre la saturation pour une « bonne » répartition (elle est large, mais sans information) ; lire une variance au lieu d'un écart-type.
**Variante** : quel panneau aurait donné LeCun normale ? (Pour des couches carrées, **le même que Glorot**, P : $1/256 = 2/(256 + 256)$ ; seule la première couche, de 784 entrées, diffère un peu.)

### Ex 16.23 — Activations à travers 20 couches selon l'initialisation 🔬
**Réponses** : en ReLU, He garde l'écart-type de 1,41 à 1,68 (rapport 1,19) ; LeCun et Glorot le divisent par environ 600 (rapport 0,0017) ; $\mathcal{N}(0 ;\ 0{,}05^2)$ par environ 40 000. En tanh, LeCun et Glorot descendent de 1 à 0,17 ; He reste vers 0,8 ; $\mathcal{N}(0 ;\ 0{,}05^2)$ tombe à 0,0065. 28,9 % des neurones de la dernière couche ReLU (He) sont morts sur ces 512 images (une valeur qui dépend des tirages : la vérification accepte de 10 % à 50 %).
**Démarche** : `profile_23` assemble les couches avec `init_weights` (pour passer `scale=0.05`), lance `mlp_forward` avec le cache et rend `float(np.std(z))` par couche ; `dead_23` teste `(z <= 0).all(axis=0)` sur la dernière couche.
**Dans tes notes** : en ReLU, le gain par couche vaut $n_{in}\mathrm{Var}(w) \times \frac12$ : 1 pour He, $\frac12$ pour LeCun et Glorot (des couches carrées), 0,32 pour l'échelle fixe ; l'écart-type est multiplié par la racine de ce gain à chaque couche. En tanh, He a un gain de 2, mais tanh sature et borne les sorties à $\pm 1$ : après une baisse de 1,4 à 1 en deux couches, le profil se stabilise vers 0,8, plus haut que LeCun et Glorot, au prix d'activations souvent saturées. La part des neurones morts ne dépend pas de l'échelle à graine égale (biais nuls, ReLU positivement homogène : multiplier tous les poids par une constante positive ne change le signe d'aucune somme pondérée) ; elle vient des corrélations entre images, qui grandissent avec la profondeur.
**Erreurs fréquentes** : un générateur recréé pour chaque méthode avec une autre graine (les profils ne se comparent plus) ; l'activation oubliée sur la dernière couche ; mesurer l'écart-type des activations au lieu de $z$ (la consigne demande $z$).
**Variante** : en ReLU, avec quelle variance des poids (en fonction de $n_{in}$) l'écart-type serait-il divisé par 2 toutes les deux couches ? (Il faut un gain $\frac12$ par couche, soit $n_{in}\mathrm{Var}(w) \times \frac12 = \frac12$ : **$\mathrm{Var}(w) = 1/n_{in}$**, la règle de LeCun ; le profil mesuré, divisé par environ 600 en 19 couches, est proche de $2^{-19/2} \approx 1/724$.)

### Ex 16.24 — Défi : garder vivant un réseau ReLU de 50 couches 🏆
**Réponses** : le corrigé réussit le défi et le palier sur les trois graines (et sur dix graines essayées) : écarts-types de contrôle entre 0,98 et 1,02, aucun neurone mort.
**Démarche** : quatre étapes, mesurées sur les images de contrôle.
1. **Base** : `init_mlp(SIZES_24, "he_normal", rng)` sur des pixels bruts donne des écarts-types de 55 à 255 et 22 % à 27 % de neurones morts : l'échelle des pixels (écart-type 78) fausse tout, et la règle de He suppose des entrées centrées et réduites.
2. **Échelle réglée sur les données** (LSUV) : diviser les poids de chaque couche par l'écart-type de ses sommes pondérées sur `X_sample` ramène tous les écarts-types vers 1, mais laisse 22 % à 27 % de neurones morts : multiplier les poids par une constante ne change le signe d'aucune somme.
3. **Le biais de chaque neurone** : les pixels, puis les sorties de ReLU, sont tous positifs ; la somme pondérée d'un neurone a donc une moyenne loin de 0, et un neurone de moyenne très négative est mort pour toutes les images. Un biais qui **centre** chaque neurone (moyenne 0 sur l'échantillon) supprime les neurones morts, mais les écarts-types de contrôle dépassent alors 1,7 pour certaines graines (de 0,48 à 1,77 sur dix graines) : ReLU coupe la moitié de chaque neurone à chaque couche, et 50 couches amplifient les petites différences entre l'échantillon et les images de contrôle. Avec des poids orthogonaux (`np.linalg.qr` d'une matrice gaussienne), c'est un peu mieux (de 0,43 à 1,45), sans atteindre le palier partout.
4. **Décaler le biais** : placer la moyenne de chaque neurone à $+0{,}5$ fois son écart-type (mesurés sur l'échantillon), puis diviser la couche par l'écart-type de ses sommes pondérées, rend chaque neurone actif sur environ 70 % des images ; le réseau, plus proche d'un réseau linéaire, amplifie moins les écarts : de 0,98 à 1,02 sur dix graines, avec des poids de He comme avec des poids orthogonaux. C'est le corrigé (`05_solutions.ipynb`). Un décalage de $+0{,}25$ suffit déjà au palier.
**Dans tes notes** : régler seulement l'échelle ne touche pas au signe des sommes pondérées, donc pas aux neurones morts ; il faut aussi régler les biais. C'est ce que fera la batchnorm (ch. 20), à chaque pas d'entraînement plutôt qu'une fois au départ.
**Erreurs fréquentes** : standardiser `X_HELD_24` (interdit : la fonction ne reçoit que `X_sample`) ; régler l'échelle sur un seul exemple ; régler le biais avec la moyenne de **toute** la couche au lieu de celle de chaque neurone ; oublier d'appliquer ReLU entre deux réglages ; juger sa méthode sur l'échantillon qui a servi à la régler.
**Variante** : avec la méthode du corrigé, que donnent 100 couches au lieu de 50 ? (Presque rien : avec `[128] * 99`, les écarts-types de contrôle restent entre 0,98 et 1,02 et aucun neurone n'est mort sur les trois graines. Une initialisation réglée sur les données garde un réseau vivant au départ, même très profond ; l'entraîner reste une autre affaire, qu'aideront la normalisation (ch. 20) et les connexions résiduelles (ch. 21).)
