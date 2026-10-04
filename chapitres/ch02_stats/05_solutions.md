# 2 · Hasard et statistiques de base — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses des exercices ✏️ se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🧮 🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à D](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 2.Q1 — Moyenne, médiane, mode : laquelle résiste aux valeurs extrêmes ?
1. Moyenne $\frac{76}{7} \approx$ **10,86** ; médiane **6** (la 4ᵉ des 7 valeurs triées) ; mode **5**. 2. Seule la **moyenne** change (elle passe à $\frac{436}{7} \approx 62{,}29$) : la médiane et le mode ne dépendent pas de la valeur exacte du maximum. 3. La **moyenne** : 10,86 n'est pas dans la liste (avec un nombre impair de valeurs, la médiane en est une ; le mode en est toujours une). 4. Le **mode** : on ne peut ni additionner ni trier des noms d'îles, seulement compter. 5. **Vrai** : la loi normale est symétrique, avec un seul sommet. 6. Parce que la forme des données guide les choix : une distribution très asymétrique appelle la médiane ou un logarithme, des valeurs aberrantes appellent des méthodes robustes ou un nettoyage, des colonnes d'échelles très différentes appellent une standardisation (ch. 12).
**À retenir** : la médiane est **robuste** aux valeurs extrêmes, la moyenne ne l'est pas.

### 2.Q2 — Quand a-t-on le droit de parler de probabilités ?
1. **Faux** : la somme vaut 50 ; divisés par 50, ils deviennent 0,24, 0,6 et 0,16. 2. **Faux** : la somme vaut 1,1. 3. **Vrai** : une valeur exacte est un intervalle de longueur nulle, donc d'aire nulle. 4. **Faux** : une densité peut valoir 2 ou 100 ; c'est l'aire sous la courbe qui vaut 1 (figure de la fiche §2.3.2). 5. **Vrai**. 6. **Vrai**.
**Erreurs fréquentes** : lire la hauteur d'une densité comme une probabilité.

### 2.Q3 — Graine et pseudo-aléatoire : vrai ou faux
1. **Vrai** : c'est un calcul, et son résultat ne dépend que de son état de départ. 2. **Vrai** (NumPy ne garantit pas que les suites restent identiques d'une version à l'autre : c'est une raison de figer les versions, BIBLE §21). 3. **Faux** : la graine rend la suite **reproductible**, pas meilleure. 4. **Faux** : sans graine, NumPy en tire une nouvelle, imprévisible, à chaque exécution. 5. **Faux** : 42 est une convention (une référence au *Guide du voyageur galactique*) ; n'importe quel entier convient, à condition de le choisir **avant** de regarder les résultats. 6. **Faux** : on utilise le module `secrets` de Python, conçu pour être imprévisible même pour quelqu'un qui connaît les sorties précédentes.

### 2.Q4 — Loi uniforme sur [0, 1] : questions pièges
1. **0,3**. 2. **0,5**. 3. **0**. 4. **1** ; pour la loi uniforme sur $[-1, 1]$, **0,5** (l'intervalle est deux fois plus long, la hauteur deux fois plus petite, pour garder une aire de 1). 5. **0,5**, le milieu de l'intervalle. 6. **Non** et **non** : `rng.random()` tire dans $[0, 1)$, et `rng.integers(1, 7)` renvoie un entier de 1 à 6, la borne haute étant exclue (comme `range`).

### 2.Q5 — La règle 68-95-99,7
1. Environ **68 %** (170 ± 1 écart-type). 2. Environ **95 %** (± 2 écarts-types). 3. Environ **2,5 %** : la moitié des 5 % qui sortent de l'intervalle, par symétrie. 4. Environ **0,15 %** : la moitié des 0,3 % qui sortent de l'intervalle à 3 écarts-types. 5. **Faux** : c'est une propriété de la loi normale ; pour une loi uniforme, ± 1 écart-type ne contient que 57,7 % des valeurs. 6. En **cm** ; la variance en **cm²**, ce qui la rend moins parlante.

### 2.Q6 — Bernoulli ou multinoulli ?
1. **Bernoulli**. 2. **Catégorielle** (10 classes). 3. **Bernoulli**. 4. **Catégorielle** (3 espèces). 5. **Aucune des deux** : la masse est une grandeur continue (pour une espèce donnée, une loi normale la décrit assez bien). 6. **Catégorielle**, avec six probabilités égales (loi uniforme discrète). 7. $K$ probabilités, dont seulement $K - 1$ sont libres, puisque leur somme vaut 1 (pour Bernoulli, un seul nombre : $p$).

### 2.Q7 — Une espérance qu'on ne tire jamais
1. $\frac{2 + 4 + 6 + 8}{4} = $ **5**, qui n'est pas une face : on ne la tire jamais. 2. **0,3**, et on ne tire que 0 ou 1. 3. **Faux**. 4. $2 \times 5 + 1 = $ **11** (linéarité de l'espérance, 0B). 5. De l'**espérance** : c'est la **loi des grands nombres** (0B, 101.7.5) ; l'écart diminue environ comme $\frac{1}{\sqrt{n}}$.

### 2.Q8 — Dépendant, indépendant, i.i.d.
1. **Indépendants, i.i.d.** : même dé, même loi, et un lancer ne renseigne pas sur le suivant. 2. **Dépendantes**, donc la suite des températures n'est **pas** i.i.d. : un jour chaud est souvent suivi d'un jour chaud. 3. **Dépendantes** : un grand manchot a souvent un long bec **et** une longue nageoire (la question i.i.d. ne se pose pas : ce sont deux grandeurs de nature différente). 4. **Dépendantes** : la loi de la nageoire change avec l'espèce (fiche §2.4). En revanche, des manchots **successifs** tirés au hasard donnent des couples (espèce, nageoire) i.i.d. 5. **i.i.d.** : c'est la situation idéale que supposent les méthodes du ML. 6. **Pas indépendants**, donc pas i.i.d. : le test « connaît » déjà les personnes vues à l'entraînement ; le score sera trop optimiste (une fuite de données, ch. 1 et 8).

### 2.Q9 — Avec ou sans remise ?
1. **Sans** (une boule tirée ne revient pas). 2. **Avec**. 3. **Sans** : un exemple ne doit jamais être dans les deux jeux. 4. **Sans** : chaque exemple apparaît une fois par epoch. 5. **Avec** : une boisson commandée reste à la carte. 6. **Avec** : c'est la valeur par défaut de `replace`. 7. **Faux**.

### 2.Q10 — Ce que le bootstrap estime, et ce qu'il n'invente pas
1. **Faux** : il ne fait que retirer au hasard des éléments déjà présents. 2. **Vrai**. 3. **Vrai** (le livre utilisait des rééchantillons plus petits, ce qui élargit l'intervalle : 🕰️ de la fiche). 4. **Faux** : il mesure le hasard de l'échantillonnage, pas le biais de l'échantillon. 5. **Faux** : plus de rééchantillons rendent l'intervalle plus **stable** d'une exécution à l'autre, pas plus étroit ; ce qui le rétrécit, c'est un échantillon plus grand. 6. **Vrai** : environ $1 - e^{-1} \approx 63{,}2\,\%$.

### 2.Q11 — Une image est un point dans un espace à 784 dimensions
1. **784**. 2. $64 \times 64 \times 3 = $ **12 288**. 3. **Proches** : presque toutes leurs coordonnées sont égales, leur distance est petite. 4. **Faux**. 5. **Vrai** : $\sqrt{\sum_j (a_j - b_j)^2}$ s'écrit pareil avec 784 termes. 6. Pour couvrir un espace, il faut un nombre de points qui explose avec la dimension (avec seulement 10 valeurs par axe, $10^d$ cases) : n'importe quel dataset laisse presque tout l'espace vide.

### 2.Q12 — Covariance, corrélation et quartet d'Anscombe
1. Des **mm × g**. 2. **Aucune** : c'est un nombre pur, entre −1 et 1. 3. **−1**. 4. **Non** : $y$ dépend de $x$ (il se déduit presque de $x$), mais pas en ligne droite ; une corrélation nulle signifie seulement l'absence de lien **linéaire**. 5. **Faux** (variable de confusion, causalité inversée, coïncidence). 6. Quatre nuages très différents ont les mêmes statistiques résumées : il faut toujours **tracer** les données.

<a id="rappels"></a>

## 🔁 Rappels

### 2.R1 — Ch. 1 : supervisé ou non supervisé, quatre tâches sur Penguins
1. Supervisé, **classification** (l'espèce est une catégorie). 2. Supervisé, **régression** (la masse est une quantité). 3. Non supervisé, **clustering**. 4. Non supervisé, **réduction de dimension**.

### 2.R2 — 0A : NumPy, moyenne par colonne avec `axis`
1. `(3, 2)`. 2. `array([ 3., 30.])` : une moyenne par colonne. 3. `array([ 5.5, 11. , 33. ])` : une moyenne par ligne. 4. `np.float64(16.5)` : $\frac{99}{6}$. 5. `axis=0` : on « écrase » l'axe des échantillons, il reste une valeur par feature.

### 2.R3 — 0B : espérance et variance d'un dé équilibré
1. $\mathbb{E}[X] = \frac{1 + 2 + \ldots + 8}{8} = \frac{36}{8} = $ **4,5**. 2. $\mathbb{E}[X^2] = \frac{1 + 4 + 9 + 16 + 25 + 36 + 49 + 64}{8} = \frac{204}{8} = $ **25,5**. 3. $\mathrm{Var}(X) = 25{,}5 - 4{,}5^2 = $ **5,25** et $\sigma = \sqrt{5{,}25} \approx$ **2,29**. 4. $4{,}5 + 4{,}5 = $ **9** (l'espérance d'une somme est la somme des espérances, 0B).

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 2.1 — Moyenne, médiane et mode d'une liste de salaires ✏️
a) $\frac{31\,700}{9} \approx 3\,522{,}2$, soit **3 522 €** · b) **2 400 €** · c) **2 100 €** · d) **8** · e) $\frac{49\,700}{9} \approx 5\,522{,}2$, soit **5 522 €** · f) **2 400 €**.
**Pourquoi** : triés, les salaires sont 1 900 ; 2 100 ; 2 100 ; 2 300 ; **2 400** ; 2 600 ; 2 800 ; 3 500 ; 12 000. La médiane est la 5ᵉ valeur. Le salaire de la directrice pèse à lui seul 38 % de la somme : il tire la moyenne si haut que 8 salariés sur 9 gagnent moins que « le salaire moyen ». Quand il passe à 30 000 €, la somme augmente de 18 000 €, la moyenne de 2 000 €, mais la médiane ne bouge pas : 30 000 € reste la plus grande valeur, à la même place.
g) Réponse modèle : « En général, on gagne autour de 2 400 € (la médiane) ; la plupart des salaires sont entre 2 100 et 2 800 €. » Donner la moyenne (3 522 €) serait trompeur : elle ne correspond au salaire de presque personne.
**Erreurs fréquentes** : prendre la valeur du milieu de la liste **non triée** (2 800 €) ; diviser par 10 ; oublier de recalculer la moyenne en e.
**À retenir** : pour des salaires, des prix de logements ou des temps de réponse, toujours donner la médiane, et regarder l'histogramme.

### Ex 2.2 — De la casse de voitures à la distribution de probabilité ✏️
a) $\frac{224}{800} = $ **0,28** · b) **[0,22 ; 0,18 ; 0,28 ; 0,12 ; 0,20]** · c) $0{,}28 \times 360° = $ **100,8°** · d) $1 - 0{,}12 = $ **0,88** · e) $50 \times 0{,}18 = $ **9** · f) **[0,22 ; 0,40 ; 0,68 ; 0,80 ; 1,00]** · g) **SUV**.
**Pourquoi** : diviser par le total **normalise** les comptages (somme 1). La roue répartit 360° au prorata des probabilités. d utilise l'événement contraire. e est une espérance : sur 50 tirages avec remise, on attend en moyenne $50 \times p$ pick-up (parfois 7, parfois 12…). Pour g, les segments sont berline $[0 ; 0{,}22)$, pick-up $[0{,}22 ; 0{,}40)$, monospace $[0{,}40 ; 0{,}68)$, SUV $[0{,}68 ; 0{,}80)$ et break $[0{,}80 ; 1)$ : 0,71 tombe dans celui des SUV.
**Erreurs fréquentes** : donner des comptages ou des pourcentages au lieu de probabilités ; en g, prendre le dernier type dont la somme cumulée est inférieure à $u$ (monospace) au lieu du premier dont la somme cumulée **dépasse** $u$.
**Variante** : si $u$ tombe **pile** sur une somme cumulée (0,68), il appartient au segment suivant, car chaque segment inclut sa borne gauche et exclut sa borne droite. C'est ce que fait `np.searchsorted(np.cumsum(p), u, side="right")`, l'algorithme de `mylearn.stats.sample_categorical` (2.19).

### Ex 2.3 — La règle 68-95-99,7 sur les nageoires des manchots ✏️
a) **[183,5 ; 196,5]** · b) **[177 ; 203]** · c) **0,025** · d) $151 \times 0{,}025 = 3{,}775$, soit **4** · e) $\frac{210 - 190}{6{,}5} \approx$ **3,08**.
**Pourquoi** : 203 mm est exactement $\mu + 2\sigma$ ; 5 % des Adélie sortent de l'intervalle à 95 %, moitié au-dessus, moitié au-dessous, par symétrie.
f) Réponse modèle : une nageoire de 210 mm est à plus de 3 écarts-types au-dessus de la moyenne des Adélie. Moins de 0,15 % des Adélie dépassent $\mu + 3\sigma$ : parmi 151 Adélie, on en attendrait environ 0,2. Ce n'est pas impossible, mais c'est très improbable ; c'est bien plus vraisemblablement un Gentoo, dont la nageoire mesure en moyenne près de 30 mm de plus (environ 217 mm contre 190 mm, fiche §2.4). Tu confronteras ce modèle aux vraies données en 2.18.
**Erreurs fréquentes** : répondre 0,05 en c (on oublie que les 5 % se partagent entre les deux côtés) ; tronquer 3,0769 en 3,07.

### Ex 2.4 — Variance : diviser par N ou par N − 1 ? ✏️
a) **6** · b) **30** · c) $\frac{30}{5} = $ **6** · d) $\frac{30}{4} = $ **7,5** · e) $\sqrt{6} \approx$ **2,449** · f) $\sqrt{7{,}5} \approx$ **2,739** · g) $\frac{16 + 49 + 36 + 9 + 100}{5} = $ **42**, et $42 - 6^2 = 6$ : on retrouve c · h) **6** · i) $\sqrt{6} \times 1\,000 \approx 2\,449{,}49$, soit **2 449,5** ms (pars de $\sqrt{6}$ : l'arrondi 2,449 de e donnerait 2 449,0).
**Pourquoi** : les écarts à la moyenne sont −2, 1, 0, −3, 4 (leur somme vaut toujours 0, d'où le carré). Ajouter 100 à chaque valeur ajoute aussi 100 à la moyenne : les écarts, donc la variance, ne changent pas. Multiplier par 1 000 multiplie les écarts par 1 000, la variance par $1\,000^2$ et l'écart-type par 1 000 : $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$ (0B).
j) Réponse modèle : ddof = 1. On estime la dispersion de **tous** les chargements (la population) à partir de 5 mesures (un échantillon) ; les écarts sont mesurés autour de la moyenne de l'échantillon, trop proche de ces 5 valeurs, et diviser par $n$ sous-estime la variance. Avec seulement 5 mesures, l'écart entre les deux vaut 25 % ; avec 1 000 mesures, il serait négligeable.
**Erreurs fréquentes** : oublier le carré (la somme des écarts vaut 0) ; confondre variance et écart-type ; en i, multiplier l'écart-type par $1\,000^2$.

### Ex 2.5 — Espérances : Bernoulli, multinoulli et jeu de hasard ✏️
a) **0,3** · b) $0{,}3 \times 0{,}7 = $ **0,21** · c) $200 \times 0{,}3 = $ **60** · d) $\frac{1 + 20}{2} = $ **10,5** · e) $0{,}1 + 0{,}4 + 0{,}9 + 1{,}6 = $ **3,0** · f) **[0, 0, 1, 0]** · g) **[0,1 ; 0,2 ; 0,3 ; 0,4]** · h) **−0,25 €** · i) **100 €**.
**Pourquoi** : en g, la composante n° $k$ du vecteur one-hot vaut 1 quand on tire la classe $k$ (probabilité $p_k$) et 0 sinon : c'est une variable de Bernoulli d'espérance $p_k$. La moyenne des vecteurs one-hot se rapproche donc du vecteur des probabilités. C'est pour cela qu'un classifieur qui prédit des probabilités s'entraîne avec des labels en one-hot (ch. 6 et 17). En h, les gains nets valent 18 € (probabilité $\frac{1}{20}$), 3 € ($\frac{3}{20}$) et −2 € ($\frac{16}{20}$) : $\frac{18 + 9 - 32}{20} = -0{,}25$ €. En i, l'organisateur gagne ce que perdent les joueurs : $400 \times 0{,}25 = 100$ €.
**Erreurs fréquentes** : en b, répondre $p$ ou $p^2$ ; en e, faire la moyenne simple de 1, 2, 3, 4 (2,5) ; en h, oublier de déduire la mise (+1,75 €), ou croire que la mise est rendue quand on gagne (+0,15 €).
**Variante** : quel gain faudrait-il promettre pour le 20 pour que le jeu soit équitable (espérance nulle) ? Il faut 0,25 € de plus par partie en moyenne, soit 5 € de plus pour un 20 (probabilité $\frac{1}{20}$) : 25 €.

### Ex 2.6 — Compter les tirages avec et sans remise ✏️
a) $\binom{5}{2} = $ **10** (AB, AC, AD, AE, BC, BD, BE, CD, CE, DE) · b) **15** (les 10 paires précédentes, plus AA, BB, CC, DD et EE) · c) $5^2 = $ **25** · d) $\binom{7}{3} = $ **35** · e) $5^5 = $ **3 125** · f) $0{,}8^5 \approx$ **0,3277** · g) $0{,}999^{1\,000} \approx$ **0,3677** · h) $1 - 0{,}999^{1\,000} \approx$ **0,6323**.
**Pourquoi** : en d, il y a $7 \times 6 \times 5 = 210$ suites sans remise, et chaque groupe de 3 correspond à $3! = 6$ suites. En f et g, un exemple donné échappe à un tirage avec probabilité $1 - \frac{1}{n}$, et les $n$ tirages sont indépendants (avec remise). g est déjà très proche de la limite $e^{-1} \approx 0{,}3679$.
i) Réponse modèle : tirer $n$ éléments parmi $n$ **sans** remise redonne toujours exactement l'échantillon (dans un autre ordre). La statistique (une moyenne, par exemple) serait la même à chaque rééchantillon : la « distribution » bootstrap serait réduite à une seule valeur, et ne dirait rien de la variabilité.
**Erreurs fréquentes** : compter les suites quand l'ordre ne compte pas (20 au lieu de 10 en a) ; en f et g, donner la limite $e^{-1}$ au lieu de la valeur exacte ; en h, donner la part des absents, ou la limite $1 - e^{-1} \approx 0{,}6321$.

### Ex 2.7 — Covariance et corrélation de cinq points à la main ✏️
a) **[3 ; 4]** · b) **9** · c) $\frac{9}{5} = $ **1,8** · d) $\frac{9}{4} = $ **2,25** · e) **[1,414 ; 1,414]** · f) $\frac{1{,}8}{\sqrt{2} \times \sqrt{2}} = $ **0,9** · g) $60 \times 1{,}8 = $ **108** · h) **0,9**.
**Pourquoi** : écarts de $x$ : −2, −1, 0, 1, 2 ; écarts de $y$ : −2, −1, 1, 0, 2 ; produits : 4, 1, 0, 0, 4, de somme 9. Les sommes des carrés valent 10 pour $x$ comme pour $y$, d'où des variances de 2 et des écarts-types de $\sqrt{2}$. Passer aux minutes multiplie la covariance par 60 (et l'écart-type de $x$ par 60) : la corrélation ne change pas (démonstration en 2.8).
i) Réponse modèle : non. Ce sont des **observations**, pas une expérience : les étudiants qui révisent plus sont peut-être aussi plus motivés, ou avaient déjà un meilleur niveau (variables de confusion), et 5 points ne suffisent pas à conclure. Pour mesurer l'effet d'une heure de révision, il faudrait une expérience : tirer au sort qui révise plus longtemps.
**Erreurs fréquentes** : mélanger ddof = 1 pour la covariance et ddof = 0 pour les écarts-types (on trouve alors 1,125, impossible pour une corrélation) ; croire que la corrélation change avec l'unité.

### Ex 2.8 — Changer d'unité : la covariance bouge, pas la corrélation ∂

1. $\bar{u} = \frac{1}{n}\sum_{i=1}^{n}(a\,x_i + b) = a\,\frac{1}{n}\sum_{i=1}^{n} x_i + \frac{1}{n} \times n\,b = a\,\bar{x} + b$.

2. $u_i - \bar{u} = (a\,x_i + b) - (a\,\bar{x} + b) = a\,(x_i - \bar{x})$ : le décalage $b$ disparaît. De même, $v_i - \bar{v} = c\,(y_i - \bar{y})$.

3. $\mathrm{Cov}(u, v) = \frac{1}{n}\sum_i a\,(x_i - \bar{x}) \times c\,(y_i - \bar{y}) = a\,c\,\frac{1}{n}\sum_i (x_i - \bar{x})(y_i - \bar{y}) = a\,c\,\mathrm{Cov}(x, y)$.

4. D'après 3 avec $v = u$ : $\mathrm{Var}(u) = \mathrm{Cov}(u, u) = a^2\,\mathrm{Var}(x)$, donc $\sigma_u = \sqrt{a^2}\,\sigma_x = |a|\,\sigma_x$ (une racine carrée est positive).

5. On suppose $\sigma_x$ et $\sigma_y$ non nuls (sinon la corrélation n'est pas définie). $r(u, v) = \frac{\mathrm{Cov}(u, v)}{\sigma_u\,\sigma_v} = \frac{a\,c\,\mathrm{Cov}(x, y)}{|a|\,\sigma_x\,|c|\,\sigma_y} = \frac{a\,c}{|a|\,|c|}\,r(x, y)$. La fraction $\frac{a\,c}{|a|\,|c|}$ vaut **+1** si $a$ et $c$ ont le même signe, **−1** sinon. Si $a, c > 0$ (grammes → kilogrammes, °C → °F), la corrélation ne change pas ; si $a < 0$ et $c > 0$, elle change de signe mais garde sa valeur absolue (par exemple, « distance parcourue » remplacée par « distance restant à parcourir »).

6. La masse joue le rôle de $x$ et la nageoire celui de $y$ : $a = \frac{1}{1\,000}$ et $c = \frac{1}{10}$. La covariance est multipliée par $\frac{1}{10} \times \frac{1}{1\,000} = 10^{-4}$ ; la corrélation ne change pas. Sur la guitare du livre, les covariances de mesures en millimètres, en secondes ou en newtons ne se comparent pas entre elles ; les corrélations, sans unité, si : elles disent quelles paires de mesures sont le plus liées.

7. Avec ddof = 0, $\mathrm{Cov}(x, y) = \frac{1}{n}\,\mathbf{d}_x \cdot \mathbf{d}_y$, $\sigma_x = \sqrt{\frac{1}{n}\,\lVert \mathbf{d}_x \rVert^2} = \frac{1}{\sqrt{n}}\,\lVert \mathbf{d}_x \rVert$ et de même pour $y$. Donc
$$r = \frac{\frac{1}{n}\,\mathbf{d}_x \cdot \mathbf{d}_y}{\frac{1}{\sqrt{n}}\lVert \mathbf{d}_x \rVert \times \frac{1}{\sqrt{n}}\lVert \mathbf{d}_y \rVert} = \frac{\mathbf{d}_x \cdot \mathbf{d}_y}{\lVert \mathbf{d}_x \rVert\,\lVert \mathbf{d}_y \rVert}$$
C'est la similarité cosinus des deux vecteurs d'écarts (0B, 101.3.3). Par Cauchy-Schwarz, $|\mathbf{d}_x \cdot \mathbf{d}_y| \le \lVert \mathbf{d}_x \rVert\,\lVert \mathbf{d}_y \rVert$, donc $|r| \le 1$. L'égalité a lieu quand $\mathbf{d}_y$ est proportionnel à $\mathbf{d}_x$, c'est-à-dire quand les points sont exactement alignés.

**Erreurs fréquentes** : oublier la valeur absolue en 4 (un écart-type est toujours positif) ; conclure en 5 que la corrélation ne change **jamais** (elle change de signe si une seule des deux échelles est retournée).

<a id="reflexion"></a>

## 🧮 🗣️ ⚖️ 📄 Réflexion

### Ex 2.9 — Le bootstrap en cinq lignes 🗣️
« On a chronométré l'attente de 30 clients à un guichet : 12 minutes en moyenne. Avec 30 autres clients, on aurait trouvé un autre chiffre ; de combien pourrait-il changer ? Le bootstrap répond sans refaire d'enquête : on tire au hasard 30 clients parmi nos 30, **avec remise** (certains reviennent, d'autres manquent), et on recalcule la moyenne. On recommence un millier de fois : les moyennes obtenues montrent de combien le chiffre peut varier d'un **échantillon** à l'autre. On garde les 95 % du milieu : c'est l'**intervalle** de confiance, par exemple de 10 à 14 minutes. »
**Ce qui compte** : partir de la question (« ce chiffre est-il fiable ? »), dire que l'on retire **dans ses propres données**, avec remise, et que l'on lit la dispersion des résultats. **Piège** : dire que le bootstrap « crée des données » ou qu'il corrige un échantillon mal choisi.

### Ex 2.10 — Corrélation, causalité et échantillon biaisé ⚖️
1. **A.** Une **variable de confusion** : la taille ou la densité du quartier (plus d'habitants et de bâtiments, donc plus d'incendies **et** plus d'interventions). Une **causalité inversée** : ce sont les incendies qui font venir les pompiers, pas l'inverse. Les pompiers ne sont pas la cause.
2. **B.** L'intervalle mesure seulement le hasard du tirage **parmi les utilisateurs** : avec 50 000 personnes, ce hasard est minuscule, d'où un intervalle étroit (sa largeur diminue comme $\frac{1}{\sqrt{n}}$). Il ne mesure pas la différence entre ces utilisateurs et les Français : les utilisateurs d'une application de sport (plus jeunes, plus sportifs, qui dorment avec une montre connectée) ne sont pas représentatifs, et la montre estime le sommeil plus qu'elle ne le mesure. Le journal devrait écrire : « Les utilisateurs de l'application dorment en moyenne 7 h 42, selon leur montre. » Pour estimer le sommeil des Français, il faut un échantillon tiré au hasard dans toute la population, ou au moins redressé (repondéré par âge, sexe, région), comme le font les instituts de sondage.
3. **C.** La corrélation reflète les **décisions passées** de l'entreprise, pas la qualité d'un candidat : si l'on a surtout embauché des hommes, une ligne pratiquée surtout par des hommes devient un « bon » signe. Le rugby sert alors de **variable de substitution** (*proxy*) du genre, et le modèle reproduit, voire amplifie, une discrimination passée. Les femmes sont les premières désavantagées, et plus largement tous les profils peu présents dans les embauches passées. En France, écarter un candidat en raison de son sexe est interdit (Code du travail, article L1132-1), que la décision soit prise par un humain ou par un algorithme.
4. Vérifications avant la mise en service :
   - comparer les taux de sélection et les erreurs du modèle **par groupe** (genre, âge…) ;
   - retirer la variable et chercher ses autres substituts (sports, écoles, codes postaux…), puis revérifier ;
   - auditer les données d'entraînement : qui a été embauché, et selon quels critères ;
   - garder une décision **humaine** (le RGPD, article 22, encadre les décisions entièrement automatisées) et documenter le modèle (données, limites, résultats par groupe).
   
   Dans l'Union européenne, les systèmes d'IA utilisés pour recruter ou trier des candidatures sont classés **à haut risque** par l'AI Act (règlement (UE) 2024/1689, annexe III, point 4) : gestion des risques, qualité et gouvernance des données, contrôle humain et documentation y seront obligatoires à partir du **2 décembre 2027** (date repoussée en 2026 par l'« omnibus numérique » sur l'IA ; elle était fixée au 2 août 2026). Le texte évolue : consulte sa version consolidée.
   
   *Sources :* [règlement (UE) 2024/1689 (AI Act), EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) ; [« EU Digital Omnibus on AI enters into force », National Law Review, 2026](https://natlawreview.com/article/eu-digital-omnibus-ai-enters-force) ; [RGPD, EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng) ; [Code du travail, article L1132-1, Légifrance](https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000045391841).

### Ex 2.11 — Fermi : la taille de l'espace des images 🧮
1. **784** dimensions pour MNIST ; $4\,000 \times 3\,000 \times 3 = $ **36 millions** pour la photo.
2. $2^{784} = 2^{4} \times (2^{10})^{78}$ ; avec $2^{10} \approx 10^3$, on trouve environ $16 \times 10^{234} \approx 10^{235}$. Plus précisément, $2^{784} = 10^{784 \times \log_{10} 2} \approx 10^{784 \times 0{,}301} \approx$ **$10^{236}$** images noir et blanc (l'approximation $2^{10} \approx 1\,000$ perd un peu, car $1\,024 > 1\,000$) : pour une estimation de Fermi, les deux réponses sont bonnes.
3. $256^{784} = 2^{6\,272} \approx 10^{0{,}301 \times 6\,272} \approx$ **$10^{1\,888}$** images en niveaux de gris (avec $2^{10} \approx 10^3$, on trouve environ $10^{1\,882}$ : les deux conviennent).
4. Ces nombres écrasent les $10^{80}$ atomes de l'univers observable : même les images noir et blanc sont $10^{156}$ fois plus nombreuses. MNIST, avec $7 \times 10^4$ images, occupe une fraction d'environ $7 \times 10^4 / 10^{1\,888} = 7 \times 10^{-1\,884} \approx 10^{-1\,883}$ de l'espace : pratiquement rien.
5. Une image en `float32` : $784 \times 4 = 3\,136$ octets (≈ 3 Ko) ; tout MNIST : $70\,000 \times 3\,136 \approx 2{,}2 \times 10^8$ octets, soit **≈ 220 Mo**. En `uint8` : 784 octets par image, **≈ 55 Mo**. (Le fichier `data/mnist.npz` du workbook ne pèse que 11,5 Mo : il est compressé, et la plupart des pixels valent 0.)
6. Les vraies images de chiffres ne sont pas tirées au hasard dans cet espace : une image au hasard ressemble à de la neige de télévision. Les chiffres occupent une toute petite région très **structurée** (des traits continus, des pixels voisins semblables, peu de formes possibles). Le modèle n'a pas besoin de couvrir tout l'espace, seulement de comprendre la forme de cette région. C'est l'idée qu'on appelle parfois « hypothèse de la variété » (*manifold hypothesis*), à laquelle on reviendra avec la réduction de dimension (ch. 12) et les autoencodeurs (ch. 25).

### Ex 2.12 — Anscombe (1973) : regarder avant de calculer 📄
1. L'idée reçue que **les calculs sont exacts et les graphiques approximatifs**, donc qu'un bon statisticien calcule plutôt qu'il ne regarde. Anscombe soutient qu'il faut faire les deux : les graphiques révèlent ce que les calculs cachent.
2. Moyenne : 9 ; somme des carrés des écarts : 110 ; variance **10** avec ddof = 0 (écart-type $\sqrt{10} \approx 3{,}16$, le chiffre du livre) et **11** avec ddof = 1 (le chiffre des tableaux habituels).
3. I : un nuage allongé ordinaire autour d'une droite : la droite est un bon résumé. II : une courbe (un arc de parabole) parfaitement régulière : une droite est inadaptée. III : des points parfaitement alignés, sauf un point isolé qui tire la droite de régression vers lui. IV : une colonne verticale de points à $x = 8$, plus un seul point à $x = 19$ qui fabrique à lui seul la pente et la corrélation. Seul le jeu I est bien résumé par sa droite.
4. Sans le point à $x = 19$, toutes les valeurs de $x$ valent 8 : l'écart-type de $x$ est nul, et la corrélation n'est même pas définie (division par 0). Toute la corrélation de 0,816 vient d'un seul point : c'est un point **influent**.
5. Toujours tracer les données (histogrammes, nuages de points) avant de les résumer, puis tracer aussi les écarts au modèle ; se méfier des points isolés qui pèsent lourd. L'équivalent moderne : l'analyse exploratoire systématique (EDA), le *pairplot*, et le Datasaurus Dozen comme argument.
**Pour aller plus loin** : on ne sait pas comment Anscombe a fabriqué ses jeux ; Matejka et Fitzmaurice (2017) ont publié une méthode qui en produit autant qu'on veut (fiche, 🕰️). Tu en fabriqueras un en 2.32.

<a id="entretien"></a>

## 💼 Entretien

### 2.E1 — Moyenne ou médiane pour résumer des salaires ?

**Réponse modèle en 60 secondes** : « Pour dire combien gagne un salarié typique, la médiane. Les salaires sont très asymétriques : quelques très hauts salaires tirent la moyenne vers le haut. Dans une équipe de neuf où la directrice gagne 12 000 € et les autres autour de 2 400 €, la moyenne dépasse 3 500 €, alors que huit salariés sur neuf gagnent moins ; la médiane, 2 400 €, décrit bien le salarié du milieu. La moyenne reste la bonne mesure pour les totaux : masse salariale, budget, coût moyen d'un poste. Dans un tableau de bord, je montrerais la médiane et les quartiles, avec un histogramme, et la moyenne seulement quand on en a besoin pour un total. »
**Relances possibles** : « Et pour les temps de réponse d'un serveur ? » (la médiane et les percentiles élevés, p95 ou p99, car c'est la queue de la distribution qui gêne les utilisateurs) · « Comment repérer une valeur aberrante ? » (un $|z| > 3$ si les données sont à peu près normales, l'écart interquartile sinon ; toujours regarder, et vérifier s'il s'agit d'une erreur de saisie ou d'une vraie valeur) · « À quoi sert le mode ? » (aux catégories : la valeur la plus fréquente).

### 2.E2 — i.i.d. : définition et pourquoi le ML en a besoin

**Réponse modèle en 60 secondes** : « i.i.d. veut dire indépendants et identiquement distribués : chaque exemple est tiré indépendamment des autres, et tous suivent la même loi. C'est l'hypothèse qui justifie presque toute l'évaluation en machine learning : on entraîne sur un échantillon, on mesure sur un jeu de test, et on suppose que les données futures suivront la même loi ; sinon, le score de test ne prédit rien. L'indépendance compte aussi : si des exemples quasi identiques se retrouvent à la fois à l'entraînement et au test, le score est trop beau. Un contre-exemple classique : une série temporelle, comme des ventes quotidiennes. Deux jours voisins se ressemblent, et l'avenir ne ressemble pas forcément au passé ; on découpe alors dans le temps, on entraîne sur le passé et on teste sur la suite. De même, avec plusieurs radios par patient, on découpe par patient. »
**Relances possibles** : « Qu'est-ce que la dérive des données (*data drift*) ? » (la loi des données change en production : on surveille les distributions des features et les performances, et on réentraîne) · « Comment découpez-vous un dataset médical ? » (par patient, avec une validation croisée par groupe, ch. 8) · « Les mini-batches sont-ils i.i.d. ? » (on mélange les exemples à chaque epoch pour s'en approcher).

### 2.E3 — Expliquer un intervalle de confiance bootstrap

**Réponse modèle en 60 secondes** : « 87 % sur 400 exemples, c'est une estimation : avec 400 autres exemples, on aurait trouvé un chiffre un peu différent. Pour chiffrer cette incertitude sans nouvelles données, j'utilise le bootstrap : je tire 400 prédictions parmi les 400, avec remise, je recalcule l'accuracy, et je recommence un millier de fois. Les 95 % centraux de ces valeurs forment un intervalle de confiance, ici d'environ 84 à 90 %. Au chef de projet, je dirais : "Le modèle a environ 87 % de bonnes réponses ; vu la taille du test, la vraie valeur est très probablement entre 84 et 90 %. Pour dire qu'un autre modèle est meilleur, il faut un écart plus grand que cette marge." Et je préciserais que cet intervalle ne corrige pas un jeu de test peu représentatif. »
**Relances possibles** : « Pourquoi avec remise ? » (sans remise, on retrouverait toujours exactement le même jeu de test) · « Combien de rééchantillons ? » (1 000 à 10 000, chacun de la taille du jeu de test) · « Comment comparer deux modèles ? » (rééchantillonner les **mêmes** exemples pour les deux modèles et regarder l'intervalle de la différence, ou un test statistique apparié ; ch. 8).

### 2.E4 — Corrélation nulle veut-elle dire indépendance ?

**Réponse modèle en 60 secondes** : « Non, pas forcément. La corrélation de Pearson ne mesure que le lien linéaire. Si $y = x^2$ avec $x$ symétrique autour de 0, la corrélation est nulle alors que $y$ dépend entièrement de $x$. L'indépendance entraîne une corrélation nulle, mais pas l'inverse. Dans l'autre sens, une corrélation de 0,9 ne prouve pas une causalité : il peut y avoir une variable de confusion, comme la chaleur qui fait monter à la fois les ventes de glaces et les noyades, une causalité inversée, ou une coïncidence sur peu de points. Le moyen le plus sûr d'établir une cause est une expérience contrôlée, un test A/B par exemple. Et dans tous les cas, je trace le nuage de points : le quartet d'Anscombe montre que la même corrélation peut cacher des situations très différentes. »
**Relances possibles** : « Comment mesurer un lien non linéaire ? » (la corrélation de Spearman, sur les rangs, pour un lien monotone ; l'information mutuelle) · « Une corrélation de 0,3, c'est beaucoup ? » (cela dépend du domaine et du nombre de points ; regarder le nuage et un intervalle de confiance) · « Qu'est-ce qu'un test A/B ? » (tirer au sort deux groupes et ne changer qu'une chose entre eux).

### 2.E5 — Pourquoi fixer la graine aléatoire d'une expérience ?

**Réponse modèle en 60 secondes** : « Beaucoup d'étapes d'un entraînement sont aléatoires : l'initialisation des poids, le mélange des exemples, le découpage entraînement/test, le dropout, l'augmentation de données. Fixer la graine rend l'expérience reproductible : je peux rejouer exactement la même exécution pour trouver un bug, ou comparer deux réglages à hasard égal. En pratique, je fixe les graines de NumPy, du module random et de PyTorch ; sur GPU, certaines opérations restent non déterministes, et il faut activer les options de déterminisme. Mais cela ne suffit pas pour que les résultats soient fiables : une seule graine peut donner un résultat chanceux. Pour annoncer qu'un modèle est meilleur, je répète l'expérience avec plusieurs graines, cinq par exemple, et je donne la moyenne et l'écart-type des scores. »
**Relances possibles** : « Deux exécutions avec la même graine donnent des résultats différents : pourquoi ? » (opérations non déterministes sur GPU, versions de bibliothèques différentes, parallélisme) · « Quelle graine choisir ? » (n'importe laquelle, mais choisie avant de voir les résultats : essayer plusieurs graines et garder la meilleure, c'est tricher) · « Qu'est-ce qu'un générateur pseudo-aléatoire ? » (un calcul déterministe dont la suite imite le hasard ; même graine, même suite).

<a id="notebook"></a>

## Notebook, parties A à D

Les réponses ci-dessous sont celles de `05_solutions.ipynb` (exécuté). La référence complète de `mylearn.stats` est dans `solutions/mylearn_ref/stats.py` : lis-la **après** avoir réussi les tests. Des méthodes différentes des corrigés sont acceptées tant que les valeurs sont les mêmes ; quand un tirage aléatoire intervient, l'énoncé impose la méthode.

### Ex 2.13 — Tendances centrales : `mean`, `median`, `mode` 🔨
a) **4 201,8 g** · b) **4 050,0 g** · c) **[3 800]** (12 manchots) · d) **[43,9 ; 17,2 ; 200,9 ; 4 201,8]** · e) **`"Adelie"`** · f) **[153,9 ; 0,0]**, puis les 21 tests passent.
**Démarche** :
```python
def _as_numbers(x):
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or np.isnan(arr).any():
        raise ValueError("x is empty or contains NaN: clean the data first (e.g. with dropna)")
    return arr


def mean(x, axis=None):
    arr = _as_numbers(x)
    n = arr.size if axis is None else arr.shape[axis]
    total = arr.sum(axis=axis)
    return float(total / n) if axis is None else total / n


def median(x, axis=None):
    arr = _as_numbers(x)
    ordered = np.sort(arr, axis=axis)                       # axis=None: sorted and flattened
    ax = 0 if axis is None else axis
    n = ordered.shape[ax]
    middle = np.take(ordered, n // 2, axis=ax)
    if n % 2 == 0:
        middle = (np.take(ordered, n // 2 - 1, axis=ax) + middle) / 2
    return float(middle) if axis is None else middle
```
et `mode` avec `np.unique(arr, return_counts=True)` (indice 3).
**Pourquoi** : la moyenne (4 202 g) dépasse la médiane (4 050 g) parce que les Gentoo, bien plus lourds, étirent la distribution vers la droite. Le mode, 3 800 g, n'est la masse que de 12 manchots sur 342 : les masses sont arrondies à 25 g (et le plus souvent à 50 g), si bien que les valeurs « rondes » se répètent ; sur une mesure continue, le mode dépend de l'arrondi plus que des manchots (fiche, ⚠️). f : une seule valeur aberrante fait monter la moyenne de 154 g. La médiane, elle, passe de la moyenne des masses n° 171 et 172 (triées) à la masse n° 172 : toutes deux valent 4 050 g, elle ne bouge donc pas ; en général, une valeur ajoutée, si grande soit-elle, ne la déplace que d'un demi-rang : elle est **robuste**.
**Erreurs fréquentes** : diviser par `len(x)` avec un `axis` (c'est la longueur de la première dimension, pas celle de l'axe réduit) ; oublier de trier dans `median` : on obtient la moyenne des éléments n° 170 et 171 de la liste **d'origine**, qui n'a rien d'une médiane ; `mode` qui renvoie le **compte** (12) au lieu de la valeur, ou une seule valeur en cas d'égalité ; convertir en `float` dans `mode`, qui échoue alors sur les noms d'espèces.
**Variante** : `mode(measured["island"].to_numpy())`, puis compare avec les effectifs de la fiche §2.2 (la fiche compte les 344 manchots ; `measured` n'en garde que 342, deux de moins, dont les mesures manquent).

### Ex 2.14 — Graine fixée ou graine libre ? 🔮
a) **`True`** (identiques) · b) **`False`** (différentes) · c) **`False`** · d) **`False`** · e) **`True`** · f) **`False`**.
**Pourquoi** : a) deux générateurs neufs créés avec la même graine partent du même état et produisent la même suite. b) Un générateur **avance** à chaque appel : le second appel continue la suite. c) Sans graine, chaque générateur part d'un état tiré de l'entropie du système : imprévisible. d) Le piège : `np.random.seed(0)` règle l'état global de l'ancienne interface (`np.random.rand`, `np.random.normal`…), que `default_rng()` n'utilise pas ; sans graine entre ses parenthèses, il reste imprévisible. C'est pour le code ancien que `wb.setup` fixe aussi `np.random.seed`. e) La suite ne dépend pas du découpage des demandes : 3 nombres puis 5, ou 8 d'un coup, parcourent la même suite (c'est vrai ici pour `random` ; ne compte pas dessus pour toutes les méthodes). f) La ligne ajoutée a consommé des nombres : la permutation est tirée plus loin dans la suite, elle change.
**À retenir** : un générateur par expérience, créé avec une graine explicite et passé aux fonctions (`rng=...`) ; un ajout de tirage au début d'un script change tous les tirages suivants. Pour isoler deux usages (le découpage des données et l'initialisation d'un modèle, par exemple), on crée deux générateurs, ou `rng.spawn(2)`.
**Erreurs fréquentes** : croire que `np.random.seed` rend tout le code reproductible (d).
**Variante** : `rng.spawn(2)` crée deux générateurs indépendants à partir d'un seul ; vérifie qu'ajouter un tirage avec le premier ne change plus ceux du second.

### Ex 2.15 — Dispersion : `variance`, `std`, `percentile`, `zscore` 🔨
a) **197,2 mm²** · b) **802,0 g** · c) **[3 550 ; 4 050 ; 4 750]** · d) **2,62** · e) **9** · f) **2,88**, puis les 22 tests passent.
**Démarche** :
```python
def variance(x, ddof=0, axis=None):
    arr = _as_numbers(x)
    n = arr.size if axis is None else arr.shape[axis]
    if n - ddof <= 0:
        raise ValueError(f"n - ddof must be positive (n = {n}, ddof = {ddof})")
    deviations = arr - arr.sum(axis=axis, keepdims=True) / n    # the deviations first
    result = (deviations ** 2).sum(axis=axis) / (n - ddof)
    return float(result) if axis is None else result


def percentile(x, q, axis=None):
    arr = _as_numbers(x)
    q = np.asarray(q, dtype=float)
    if np.any(q < 0) or np.any(q > 100):
        raise ValueError("percentiles must be between 0 and 100")
    values = np.sort(arr, axis=None) if axis is None else np.moveaxis(np.sort(arr, axis=axis), axis, 0)
    n = values.shape[0]
    position = q / 100 * (n - 1)
    below = np.floor(position).astype(int)
    above = np.minimum(below + 1, n - 1)
    fraction = (position - below).reshape(q.shape + (1,) * (values.ndim - 1))
    result = values[below] + fraction * (values[above] - values[below])
    return float(result) if result.ndim == 0 else result
```
`std` renvoie la racine de `variance` ; `zscore` calcule `mean` et `std` le long de l'axe, refuse un écart-type nul, puis remet la dimension réduite (`np.expand_dims`) avant `(arr - center) / spread`.
**Pourquoi** : la variance des nageoires s'exprime en mm², d'où l'écart-type (14,0 mm), dans l'unité des données. Le manchot le plus lourd (6 300 g) est à 2,62 écarts-types de la moyenne : lourd, pas aberrant ; 9 manchots, tous du côté lourd, dépassent 2 écarts-types, et aucune des quatre mesures n'atteint $|z| = 3$ (le plus grand, 2,88, est un bec). Avec `axis=None`, `zscore` calculerait une moyenne et un écart-type pour tout le tableau, grammes et millimètres mélangés : toutes les masses auraient un z positif (en moyenne + 1,7) et toutes les autres mesures un z négatif. Standardiser colonne par colonne, c'est ce que fait le `StandardScaler` (ch. 12).
**Erreurs fréquentes** : ddof = 1 en a (197,7) ou ddof = 0 en b (800,8) ; `percentile(mass, [0.25, 0.5, 0.75])` (des quantiles, pas des percentiles) ; la formule « moyenne des carrés − carré de la moyenne », qui perd des chiffres sur des valeurs grandes et peu dispersées (le test `test_variance_is_accurate_for_large_values` l'attrape) ; oublier `keepdims` avec un axe (soustraction mal alignée, ou erreur de forme).
**Variante** : compare les quartiles de 2.15 c avec la ligne `25%`, `50%`, `75%` de `measured["body_mass_g"].describe()` : pandas utilise la même interpolation linéaire.

### Ex 2.16 — Un histogramme fait maison 🔨
Tes comptages coïncident avec `np.histogram` (20 intervalles, puis 7 intervalles de 170 à 240 mm : **[8, 69, 113, 38, 71, 35, 8]**), l'aire de la densité vaut 1, et les 14 tests passent.
**Démarche** : l'indice 3, précédé des contrôles (`bins >= 1`, `low < high`) et du choix `low, high = arr.min(), arr.max()` quand `bin_range` vaut `None` ; avec `density=True`, `counts / (counts.sum() * np.diff(edges))`.
**Réponse 📝** : avec 4 intervalles de près de 15 mm, les deux bosses de la fiche se fondent en une seule : le creux entre elles (vers 205 mm) tombe à l'intérieur d'un intervalle (de 201,5 à 216,25 mm). Avec 100 intervalles de 0,59 mm, alors que les nageoires sont mesurées au millimètre près (des entiers), beaucoup d'intervalles ne contiennent aucune valeur possible : d'où le peigne. Un bon choix : de 15 à 25 intervalles, ou une largeur multiple de 1 mm. Le nombre d'intervalles change ce qu'on voit : on en essaie plusieurs avant de conclure sur la forme d'une distribution.
**Erreurs fréquentes** : exclure le bord droit du dernier intervalle (la plus grande valeur disparaît) ; calculer $\lfloor (x - \text{low}) / \text{largeur} \rfloor$ sans précaution : une valeur posée pile sur un bord peut tomber dans l'intervalle précédent à cause d'un arrondi ; diviser la densité par le nombre total de valeurs, alors que les valeurs hors de `bin_range` ne comptent pas.
**Variante** : `np.histogram_bin_edges(flipper, bins="auto")` propose un nombre d'intervalles calculé par des règles classiques (Sturges, Freedman-Diaconis) : ici 10 ; compare avec ton choix.

### Ex 2.17 — Galerie des lois usuelles 🎨
La figure attendue est dans `05_solutions.ipynb`.
**Démarche** :
```python
def uniform_pdf(x, a, b):
    x = np.asarray(x, dtype=float)
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0.0)


def normal_pdf(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
```
Pour la figure, voir `draw_gallery` dans `05_solutions.ipynb` : une courbe par loi, 30 points par loi normale sur une rangée à hauteur négative, légèrement décalés au hasard, des bâtons côte à côte pour les probabilités et les fréquences.
**Ce qu'il faut voir** : l'uniforme sur $[-1, 1]$ est deux fois plus basse que sur $[0, 1]$ (même aire, 1) ; les tirages d'une normale se massent sous la bosse, et ceux de $\mathcal{N}(-1, 2^2)$ s'étalent de −5 à 3 environ ; $\mathcal{N}(-1, 0{,}5^2)$ monte jusqu'à 0,8 (une normale d'écart-type inférieur à 0,4 environ dépasserait 1, et celle d'écart-type 0,1 monterait à 4 : une densité peut dépasser 1). Les fréquences sur 1 000 tirages s'écartent des probabilités d'un ou deux points : l'écart typique vaut $\sqrt{p(1 - p)/1\,000}$.
**Erreurs fréquentes** : oublier la racine de $2\pi$ ou le 2 de $2\sigma^2$ ; tracer $\mathcal{N}(-1, 0{,}5^2)$ avec `sigma=0.25` (la notation donne la **variance** entre parenthèses, et `NORMALS_17` l'**écart-type**) ; un générateur neuf dans chaque panneau, au lieu d'un seul `rng` (les tirages ne sont plus ceux du corrigé, ce qui ne change rien au dessin, mais trahit une mauvaise habitude).
**Variante** : vérifie numériquement que l'aire sous chaque densité vaut 1 : `np.sum(normal_pdf(grid, mu, sigma)) * (grid[1] - grid[0])` (une somme de rectangles, fiche §2.2, 🧮).

### Ex 2.18 — 68-95-99,7 : la théorie face aux tirages et aux manchots 🔬
a) **[0,684 ; 0,955 ; 0,997]** · b) **[0,709 ; 0,954 ; 0,993]** · c) **3** · d) **[0,629 ; 0,968 ; 1,000]**.
**Démarche** : `share_within = lambda x, k: np.mean(np.abs(x - x.mean()) < k * x.std())` (en fonction nommée, avec `def`), puis des listes en compréhension ; c) `int(np.sum(adelie_flipper > 203))`.
**Réponse 📝** : les tirages normaux redonnent la règle à l'aléa près. Les nageoires des Adélie aussi, à peu près : leur histogramme a une forme de cloche. Le modèle de 2.3 prévoyait environ 4 nageoires au-dessus de 203 mm : il y en a 3 (205, 208 et 210 mm), plus une à 203 mm pile. Sur tous les manchots, la règle échoue dans les deux sens : 62,9 % seulement à moins d'un écart-type (la moyenne tombe dans le creux entre les deux bosses, ce qui vide la bande centrale), et 100 % à moins de trois (l'écart-type, gonflé par la distance entre les groupes, fait une bande si large qu'elle contient tout). La règle vaut pour une distribution en cloche ; on regarde l'histogramme avant de l'appliquer.
**Erreurs fréquentes** : prendre l'écart-type avec ddof = 1 en b (la deuxième proportion devient 0,960) ; compter les nageoires de 203 mm **et plus** en c (4) ; donner des pourcentages au lieu de proportions.
**Variante** : refais b sur la masse des Gentoo, puis sur la masse de tous les manchots : laquelle des deux ressemble le plus à une cloche ?

### Ex 2.19 — La roue de la fortune : tirer dans une distribution discrète 🔨
Fréquences sur 10 000 tirages : **[0,222 ; 0,178 ; 0,280 ; 0,118 ; 0,202]**, pour des probabilités de [0,22 ; 0,18 ; 0,28 ; 0,12 ; 0,20] ; les 7 tests passent.
**Démarche** :
```python
def sample_categorical(p, size=None, rng=None):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0 or np.isnan(p).any() or np.any(p < 0) or abs(p.sum() - 1) > 1e-8:
        raise ValueError("p must be a 1-D array of non-negative probabilities that sum to 1")
    if rng is None:
        rng = np.random.default_rng()
    k = np.minimum(np.searchsorted(np.cumsum(p), rng.random(size), side="right"), len(p) - 1)
    return int(k) if size is None else k.astype(int)
```
**Pourquoi** : c'est la roue de la fiche : les sommes cumulées découpent $[0, 1)$ en segments de longueurs $p_k$, et un nombre uniforme tombe dans le segment n° $k$ avec la probabilité $p_k$. `np.minimum` protège contre une somme cumulée qui vaudrait 0,9999999999999999 : sans lui, un `u` plus grand donnerait la catégorie n° `len(p)`, qui n'existe pas. Les fréquences s'écartent des probabilités de quelques dixièmes de point (0,2 point au plus ici) : l'écart typique vaut $\sqrt{p(1 - p)/n}$, environ 0,45 point pour $p = 0{,}28$ et $n = 10\,000$.
**Erreurs fréquentes** : `side="left"` : un `u` égal à une somme cumulée tomberait dans le segment précédent ; c'est si rare (probabilité quasi nulle) qu'aucun test ne le voit, mais c'est la convention de la docstring et de 2.2 g ; renvoyer un `np.int64` au lieu d'un `int` pour un tirage unique ; une boucle Python sur chaque tirage (juste, mais lente) ; ne pas refuser des probabilités de somme 0,9 ou négatives.
**Variante** : `rng.choice(5, size=10_000, p=P_CARS)` fait la même chose en NumPy ; `torch.distributions.Categorical(probs).sample()` en PyTorch (fiche, 🕰️).

### Ex 2.20 — Le pelage des animaux : une variable qui dépend d'une autre 🔮
a) **4,6 cm** · b) **3** · c) **0,25** · d) **0,5** · e) **4,6 cm**. La simulation donne 4,58 cm, 0,249 et 0,502, puis 4,57 cm pour les hamsters après le mélange.
**Pourquoi** : a) l'espérance pondère la moyenne de chaque animal par sa probabilité : $0{,}5 \times 7 + 0{,}3 \times 3 + 0{,}2 \times 1 = 4{,}6$. b) À ± 3 écarts-types, les hamsters vont de 0,4 à 1,6 cm, les chats de 1,8 à 4,2 cm, les chiens de 4,6 à 9,4 cm : trois intervalles séparés, trois bosses. c) Seuls les chiens dépassent 7 cm, et la moitié d'entre eux (symétrie de la loi normale) : $0{,}5 \times 0{,}5 = 0{,}25$. d) Sachant que c'est un chien : 0,5. Connaître l'animal change la probabilité : les deux variables sont **dépendantes**. e) Le mélange donne à chaque « hamster » le pelage d'un animal quelconque : sa moyenne devient celle de tous les animaux, et les trois histogrammes se superposent ; c'est ce que veut dire l'**indépendance**.
**À retenir** : les couples (animal, pelage) tirés les uns après les autres sont **i.i.d.** (chaque tirage recommence avec la même loi), alors qu'à l'intérieur d'un couple, les deux variables sont dépendantes. En ML, l'hypothèse i.i.d. porte sur les **exemples** (les lignes), pas sur les features d'un même exemple, qui sont presque toujours liées. Les probabilités « sachant que » sont le sujet des ch. 3 et 4.
**Erreurs fréquentes** : en a, la moyenne simple de 7, 3 et 1 (3,7) ; en c, répondre 0,5 (c'est la réponse pour un chien) ; en e, garder 1 cm (le mélange sépare chaque pelage de son animal).
**Variante** : donne aux chats un écart-type de 1,5 cm, relance l'expérience : les bosses des chats et des chiens se chevauchent-elles encore assez peu pour qu'on en compte trois ?

### Ex 2.21 — Tirer avec ou sans remise 🔨
a) **`"DEGHABGH"`** · b) **6** · c) **`"FABECGDH"`** · d) **`"ValueError"`** · e) **[3, 5, 9, 0]** · f) **0,633**, puis les 8 tests passent.
**Démarche** :
```python
def sample(population, size, replace=True, rng=None):
    pop = np.asarray(population)
    if pop.ndim == 0 or len(pop) == 0:
        raise ValueError("the population is empty")
    n = len(pop)
    if size < 0 or (not replace and size > n):
        raise ValueError(f"cannot draw {size} elements from {n}" + ("" if replace else " without replacement"))
    if rng is None:
        rng = np.random.default_rng()
    index = rng.integers(0, n, size) if replace else rng.permutation(n)[:size]
    return pop[index]
```
et, dans le notebook, `epoch_minibatches` (un ordre tiré sans remise, découpé en tranches) et `mean_share_distinct` (indice 3).
**Pourquoi** : avec remise, les huit tirages ne donnent que 6 lettres différentes (G et H sortent deux fois, C et F sont absentes) ; sans remise, on obtient une **permutation** : chaque lettre une fois. Une epoch est un tirage **sans** remise, découpé en mini-batches ; un rééchantillon bootstrap est un tirage **avec** remise, qui contient en moyenne $1 - (1 - 1/n)^n$ d'éléments distincts, soit 63,2 % pour $n = 1\,000$ (2.6 h : 0,6323) ; ta simulation donne 0,633, à l'aléa près.
**Erreurs fréquentes** : utiliser `rng.choice(pop, size, replace=False)` : correct, mais d'autres tirages que l'algorithme documenté (c donnerait `"CDAFEGBH"`, et le test d'égalité exacte échoue) ; tirer les éléments d'un tableau 2-D au lieu de ses lignes ; laisser `permutation(n)[:size]` renvoyer silencieusement 8 éléments quand on en demande 10 (d : « no error »).
**Variante** : écris `train_test_split(X, test_size, rng)` avec ta fonction `sample` (sans remise) ; tu le retrouveras au ch. 8.

### Ex 2.22 — Bootstrap : distribution et intervalle de confiance 🔨
a) **3 733,1 g** · b) **[3 647,4 ; 3 822,4]** · c) **[3 676,1 ; 3 790,1]** · d) **[3 650,0 ; 3 800,0]** · e) **45,4 g**, puis les 16 tests passent.
**Démarche** :
```python
def bootstrap_distribution(x, statistic=np.mean, *, n_boot=1000, sample_size=None, rng=None):
    data = np.asarray(x)
    if data.ndim == 0 or len(data) == 0 or n_boot < 1:
        raise ValueError("x must not be empty and n_boot must be >= 1")
    n = len(data)
    size = n if sample_size is None else sample_size
    if size < 1:
        raise ValueError("sample_size must be >= 1")
    if rng is None:
        rng = np.random.default_rng()
    values = np.empty(n_boot)
    for b in range(n_boot):
        values[b] = statistic(data[rng.integers(0, n, size=size)])   # one resample, with replacement
    return values
```
et `bootstrap_ci` (indice 3), qui vérifie d'abord `0 < confidence < 1`.
**Pourquoi** : la masse moyenne des Chinstrap est connue à environ ± 90 g près avec 95 % de confiance (et à ± 57 g à 80 % : un intervalle promet moins, il est plus étroit). L'intervalle de la médiane, [3 650 ; 3 800], a des bornes « rondes » parce que la médiane de masses arrondies à 25 g l'est aussi. L'erreur type, 45,4 g, est proche de la formule classique $s / \sqrt{n}$, soit 46,6 g pour les 68 Chinstrap (avec l'écart-type $s$ de ddof = 1), que tu croiseras dans tout cours de statistique. Avec une infinité de rééchantillons, le bootstrap donnerait exactement l'écart-type **à ddof = 0** divisé par $\sqrt{n}$, soit 46,3 g : l'écart restant est le bruit de 1 000 rééchantillons. Le bootstrap retrouve ce résultat sans formule, et marche aussi pour une médiane ou une corrélation, qui n'ont pas de formule aussi simple.
**Erreurs fréquentes** : tirer les rééchantillons **sans** remise (chaque rééchantillon redonne alors l'échantillon, et la distribution se réduit à une seule valeur : 2.6 i) ; couper 5 % de chaque côté pour un intervalle à 95 % (on obtient un intervalle à 90 %) ; créer un nouveau générateur `np.random.default_rng(0)` à chaque rééchantillon (les 1 000 rééchantillons sont alors identiques, et la distribution se réduit à une seule valeur) ; donner l'écart-type des masses (381,5 g) au lieu de celui des moyennes bootstrap en e.
**Remarque** : un seul `rng.integers(0, n, size=(n_boot, n))` donne exactement les mêmes indices que la boucle (la suite d'un générateur ne dépend pas ici du découpage des tirages, 2.14 e) : c'est une version vectorisée tout aussi juste.
**Variante** : l'intervalle de la **corrélation** nageoire–masse, en gardant les lignes appariées : `bootstrap_ci(np.column_stack([flipper, mass]), lambda d: np.corrcoef(d[:, 0], d[:, 1])[0, 1], rng=np.random.default_rng(0))` donne environ [0,847 ; 0,893].

### Ex 2.23 — Bootstraps de 20 (livre) ou de n (aujourd'hui) ? 🔬
Largeurs de l'intervalle à 80 % selon la taille des rééchantillons : **339** (5), **223** (10), **166** (20), **102** (50), **74** (100), **53** (200) et **35** (500) ; rapport 20 contre 500 : **4,74**. Couverture sur 100 enquêtes : **1,00** avec des rééchantillons de 20, **0,84** avec des rééchantillons de 500 (en mode complet, sur 400 enquêtes : **1,00** et **0,80**, en quelques secondes). Largeurs pour 100, 1 000 et 10 000 rééchantillons de 500 : **30,3**, **34,9** et **33,5**.
**Démarche** : `ci_width` appelle `bootstrap_ci(sample_23, confidence=0.8, n_boot=n_boot, sample_size=sample_size, rng=rng)` et renvoie `high - low` ; `coverage` : voir l'indice 2.
**Réponses 📝** :
1. L'intervalle du livre est environ **4,7 fois** trop large, à peu près $\sqrt{500 / 20} = 5$. Sur le graphique logarithmique, les points sont presque sur une droite de pente $-\frac{1}{2}$ : quand la taille est multipliée par 4, la largeur est divisée par 2 ; elle varie comme $\frac{1}{\sqrt{\text{taille}}}$ (largeur × $\sqrt{\text{taille}}$ reste entre 700 et 790 environ). Un rééchantillon de 20 imite un échantillon de 20 : il mesure l'incertitude qu'on aurait avec 20 valeurs.
2. L'intervalle du livre contient la vraie moyenne **à chaque fois** : il promet 80 % et donne 100 %, parce qu'il est bien trop large. Avec des rééchantillons de 500, la couverture est de 84 %, proche des 80 % promis (sur 100 enquêtes, l'aléa est d'environ ± 4 points ; 80 % sur 400 enquêtes). Détail pour les curieux : l'échantillon est tiré sans remise et représente 10 % de la population, ce qui rend la vraie dispersion de la moyenne un peu plus petite que celle que mesure le bootstrap ; un intervalle « à 80 % » couvre alors plutôt 82 % des cas.
3. Non : 100, 1 000 et 10 000 rééchantillons donnent des largeurs voisines. Le **nombre** de rééchantillons rend les bornes plus **stables** (moins de bruit dans les percentiles) ; c'est la **taille** qui fixe la largeur.
**Erreurs fréquentes** : réutiliser `sample_23` dans `coverage` au lieu de tirer un nouvel échantillon à chaque enquête (la couverture ne vaut alors que 0 ou 1) ; tirer les nouveaux échantillons avec remise ; confondre `n_boot` et `sample_size`.
**Pour aller plus loin** : ajoute une troisième recette, des rééchantillons de 500 mais un intervalle « à 80 % » construit avec les percentiles 20 et 80 : la couverture tombe vers 60 %. Une promesse de couverture se vérifie toujours par simulation.

### Ex 2.24 — Comparer avec `scipy.stats.bootstrap` 📦
a) **[3 643 ; 3 824]** · b) **[3 642 ; 3 822]** · c) **46,0 g**.
**Démarche** :
```python
result = scipy_stats.bootstrap((chinstrap_mass,), np.mean, confidence_level=0.95, n_resamples=9999,
                               method="percentile", rng=np.random.default_rng(0))
ci_percentile = [result.confidence_interval.low, result.confidence_interval.high]
standard_error = result.standard_error
```
et le même appel sans `method` pour BCa.
**Pourquoi les deux intervalles percentile diffèrent** (réponse des notes) : même méthode, et mêmes premiers rééchantillons ! SciPy tire tous ses indices d'un coup (`rng.integers(0, n, (9999, n))`), ce qui redonne exactement la suite de tes tirages : les 1 000 premières valeurs de `result.bootstrap_distribution` sont celles de ta distribution de 2.22. La seule différence est le nombre de rééchantillons, 1 000 contre 9 999 : l'écart de quelques grammes est le **bruit de Monte-Carlo** de 1 000 rééchantillons. BCa corrige le biais et l'asymétrie de la distribution bootstrap ; sur des moyennes presque symétriques, il ne diffère ici que d'un ou deux grammes.
**Erreurs fréquentes** : passer `chinstrap_mass` au lieu de `(chinstrap_mass,)` (`AxisError`) ; oublier `method="percentile"` en a (on obtient l'intervalle BCa) ; réutiliser en b le générateur déjà consommé en a (d'autres rééchantillons, donc un autre intervalle) ; `random_state=0` (une habitude de scikit-learn) : un entier crée l'ancien générateur de NumPy, et donne d'autres rééchantillons. Rappel : `result.confidence_interval` est un *namedtuple* ; `ci.low` et `ci.high`, ou `ci[0]` et `ci[1]`, marchent tous les deux.
**Variante** : `bootstrap_ci(chinstrap_mass, n_boot=9999, rng=np.random.default_rng(0))` redonne l'intervalle percentile de SciPy **à l'identique** ; et `scipy_stats.bootstrap((chinstrap_mass,), np.median, ...)` donne l'intervalle de la médiane, à comparer avec 2.22 d.

### Ex 2.25 — Distances entre chiffres dans l'espace à 784 dimensions 📦
a) **11,75** · b) **548** (l'image 0 est un 5 ; sa plus proche voisine, l'image 548, aussi) · c) **0,904** · d) **1,16** · e) **0,1355** · f) **1,85**.
**Démarche** :
```python
def dist(i):
    d = np.linalg.norm(X_digits - X_digits[i], axis=1)
    d[i] = np.inf                                  # an image is not its own neighbour
    return d


nearest_index = int(np.argmin(dist(0)))
nn_accuracy = float(np.mean([y_digits[np.argmin(dist(i))] == y_digits[i] for i in range(500)]))
```
et, pour d, une matrice des distances des 500 premières images, les paires `np.triu_indices(500, k=1)` et le masque « même chiffre » (indice 2).
**Pourquoi** : en moyenne, deux images de chiffres différents ne sont qu'environ 16 % plus éloignées que deux images du même chiffre : en 784 dimensions, les distances se ressemblent. Pour des points tirés au hasard, c'est pire : le plus lointain n'est qu'à 14 % de plus que le plus proche (contraste 0,1355), alors qu'en dimension 2 le rapport dépasse 100 (graphique). C'est un visage de la **malédiction de la dimension** : « le plus proche » perd son sens. Les images de MNIST gardent un contraste d'environ 1,85 : elles ne remplissent pas l'espace au hasard, elles vivent près d'une structure de bien plus petite dimension (2.11, question 6). C'est pourquoi le plus proche voisin donne déjà le bon chiffre 9 fois sur 10.
**Erreurs fréquentes** : oublier d'écarter l'image elle-même (sa distance vaut 0 : c) donnerait 1,0) ; compter les paires d'une image avec elle-même parmi les paires « même chiffre » (d : 1,18) ; construire le tableau 2 000 × 2 000 × 784 par broadcasting (plus de 12 Go en mémoire) au lieu d'une boucle sur les images ; en e, recréer un générateur différent (une autre graine donne un autre contraste, proche mais pas égal).
**Variante** : le même contraste pour 1 000 points gaussiens (`rng.normal(size=(1000, 784))`) : la malédiction de la dimension ne dépend pas de la loi des points, seulement de leur nombre de coordonnées indépendantes.

### Ex 2.26 — Covariance et corrélation 🔨
a) **9 795,7 mm·g** · b) **0,871** · c) **0,980** (cm·kg) · d) **−0,235**, puis les 15 tests passent.
**Démarche** :
```python
def covariance(x, y, ddof=0):
    x, y = _as_numbers(x), _as_numbers(y)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1-D arrays of the same length")
    n = len(x)
    if n - ddof <= 0:
        raise ValueError(f"n - ddof must be positive (n = {n}, ddof = {ddof})")
    return float(((x - x.sum() / n) * (y - y.sum() / n)).sum() / (n - ddof))


def correlation(x, y):
    x, y = _as_numbers(x), _as_numbers(y)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("the correlation needs at least 2 points and non-constant data")
    return float(np.clip(covariance(x, y) / (std(x) * std(y)), -1.0, 1.0))
```
**Pourquoi** : la nageoire et la masse sont fortement liées : on peut prédire l'une par l'autre avec une droite (la régression, ch. 9). La covariance n'est pas interprétable seule : en cm et en kg, elle est divisée par $10 \times 1\,000$, alors que la corrélation ne bouge pas (2.8). La corrélation longueur–épaisseur du bec est faiblement **négative** sur l'ensemble des manchots : un bec plus long serait plus fin ? Le nuage montre plusieurs paquets : la suite en 2.28.
**Erreurs fréquentes** : ddof = 1 pour la covariance et 0 pour les écarts-types (0,874, et jusqu'à 1,125 sur les cinq points de 2.7) ; ne changer qu'une des deux unités en c ; accepter deux tableaux de longueurs différentes (NumPy les multiplie par broadcasting si l'un a une seule valeur, sans erreur).
**Variante** : la corrélation de Spearman (`scipy.stats.spearmanr`) est la corrélation de Pearson calculée sur les **rangs** : elle mesure un lien qui monte toujours (monotone), même courbe. Compare-la avec celle de Pearson sur la nageoire et la masse.

### Ex 2.27 — Deviner la corrélation d'un nuage de points 📈
a) **[0,99 ; 0,31 ; −0,45 ; 0,58 ; −0,91 ; −0,05]** (nuages A à F) · b) **0,01**.
**Pourquoi** : A et E se lisent vite, des points serrés autour d'une droite, montante pour A, descendante pour E. B et C sont des nuages larges mais penchés : $|r|$ moyen. Les deux pièges : F, une parabole, où $y$ dépend fortement de $x$ mais monte des deux côtés du centre, si bien que les produits d'écarts positifs et négatifs se compensent ($r \approx 0$) ; et D, un paquet sans aucune forme, auquel un seul point très éloigné donne une corrélation de 0,58 (sans lui : 0,01).
**Réponse 📝** : le nuage F. La corrélation ne mesure que la part **linéaire** du lien (fiche, ⚠️ « Deux contresens ») ; une relation en U lui échappe.
**Erreurs fréquentes** : lire D comme un nuage sans corrélation (on ne voit que le paquet) ou comme une corrélation très forte (on ne voit que la « droite » vers le point isolé) ; retirer un point du paquet au lieu du point isolé en b.
**À retenir** : toujours tracer le nuage avant de croire un $r$ ; et recalculer $r$ sans les points isolés pour voir s'ils portent tout le lien. Tu retrouves ce point **influent** dans le jeu IV d'Anscombe (2.12, 2.30).
**Variante** : avec la méthode de 2.32 (étapes 1 à 3), fabrique des nuages de corrélation 0,3, 0,6 et 0,9 exactement, et entraîne ton œil à les reconnaître.

### Ex 2.28 — Matrices de covariance et de corrélation des manchots 🔨
a) **la matrice de corrélation** (2 décimales ; ordre : bec longueur, bec épaisseur, nageoire, masse) :
$$\begin{pmatrix} 1 & -0{,}24 & 0{,}66 & 0{,}60 \\ -0{,}24 & 1 & -0{,}58 & -0{,}47 \\ 0{,}66 & -0{,}58 & 1 & 0{,}87 \\ 0{,}60 & -0{,}47 & 0{,}87 & 1 \end{pmatrix}$$
b) **épaisseur du bec et nageoire** (−0,58) · c) **+1** (positive) · d) **[0,39 ; 0,65 ; 0,64]** (Adélie, Chinstrap, Gentoo) ; les 10 tests passent.
**Démarche** :
```python
def covariance_matrix(X, ddof=0):
    A = _as_numbers(X)
    if A.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    n = A.shape[0]
    if n - ddof <= 0:
        raise ValueError(f"n_samples - ddof must be positive (n_samples = {n}, ddof = {ddof})")
    centered = A - A.sum(axis=0) / n
    C = centered.T @ centered / (n - ddof)
    return (C + C.T) / 2                       # exactly symmetric, despite rounding


def correlation_matrix(X):
    C = covariance_matrix(X)
    spread = np.sqrt(np.diag(C))
    if np.any(spread == 0):
        raise ValueError("a column is constant: its correlation is undefined")
    R = np.clip(C / np.outer(spread, spread), -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    return R
```
**Pourquoi** : la diagonale de la matrice de covariance contient les variances ; ses cases vont de −745 à 641 251, parce que les unités diffèrent (mm², mm·g, g²) : elle se lit mal. La matrice de corrélation met toutes les paires sur la même échelle. **Le paradoxe du bec** : dans chaque espèce, la corrélation longueur–épaisseur est **positive** (un manchot plus grand a un bec plus long **et** plus épais) ; mais les espèces diffèrent : les Adélie (151) ont des becs courts et épais, les Gentoo (123) des becs longs et fins. Ces deux gros paquets occupent deux coins opposés du nuage et tirent la corrélation globale vers le négatif ; les Chinstrap, aux becs longs et épais, l'atténuent seulement (sans eux, $r = -0{,}55$). C'est le **paradoxe de Simpson** (*Simpson's paradox*) : un lien peut changer de sens quand on réunit des groupes différents. L'espèce est une **variable de confusion**. La paire la plus négative (épaisseur du bec – nageoire) relève du même effet : les Gentoo ont de longues nageoires et des becs fins.
**Erreurs fréquentes** : `centered @ centered.T` (une matrice 342 × 342, celle des manchots) ; oublier de centrer ; diviser la covariance par les **variances** au lieu des écarts-types ; en b, répondre la paire la plus liée en valeur absolue (nageoire et masse, +0,87).
**Variante** : trace la « matrice de nuages » des quatre mesures colorée par espèce (`pd.plotting.scatter_matrix`, ou `seaborn.pairplot` si tu l'as installé) : c'est le réflexe de l'analyse exploratoire (fiche, 🕰️ de la §2.9).

### Ex 2.29 — Le piège de ddof : NumPy, pandas et toi 🐛
a) **1,0029** · b) **(342, 342)** · c) **0,9985**.
**Diagnostic** :
- `colleague_correlation` : `np.cov` divise par $n - 1$, `np.std` par $n$ ; le résultat est multiplié par $\frac{n}{n - 1}$ : 1,0029 pour 342 manchots (invisible), 1,25 pour les 5 étudiants de 2.7, où la « corrélation » vaut 1,125, ce qui est impossible.
- `colleague_cov_matrix` : sans `rowvar=False`, `np.cov` prend chaque **ligne** pour une variable : 342 × 342 covariances entre manchots, au lieu de 4 × 4 entre mesures (et avec ddof = 1).
- `colleague_standardize` : `.std()` de pandas divise par $n - 1$ ; les colonnes ont un écart-type de $\sqrt{341/342} \approx 0{,}9985$ au sens de NumPy et du `StandardScaler`, et ne coïncident pas avec eux (écart de 0,004 au plus sur ces données).
**Corrections** : l'indice 3.
**Comment les repérer** : afficher les formes (`.shape`) après chaque étape ; tester une fonction sur un exemple dont on connaît la réponse (les cinq points de 2.7 : 0,9) ; vérifier les propriétés attendues (une corrélation reste dans $[-1, 1]$, un tableau standardisé a un écart-type 1) ; et écrire `ddof` explicitement à chaque appel.
**Erreurs fréquentes** : répondre (4, 4) en b sans exécuter le code du collègue ; mesurer c avec `.std()` de pandas (on trouve 1, et le bug disparaît… de la mesure).
**Variante** : le même piège existe en PyTorch : `torch.std` divise par $n - 1$ par défaut (`correction=1`), comme pandas ; vérifie-le sur `torch.tensor([1.0, 2.0, 3.0, 4.0])`.

### Ex 2.30 — Le quartet d'Anscombe 🎨
a) **[7,58 ; 8,14 ; 7,11 ; 7,04]** · b) **[1,92 ; 1,90 ; 3,24 ; 1,84]** · c) **0,345** (et une corrélation de 0,999997).
**Démarche** :
```python
def fit_line(x, y):
    slope = covariance(x, y) / variance(x)       # the same ddof (0) above and below
    return mean(y) - slope * mean(x), slope
```
et la figure : `plt.subplots(2, 2, sharex=True, sharey=True)`, un nuage et sa droite par panneau (voir `05_solutions.ipynb`).
**Pourquoi** : moyennes, variances, corrélation et droite coïncident, mais pas les médianes de $y$ (de 7,04 à 8,14) : les statistiques « identiques » sont celles qu'Anscombe a choisi d'égaliser. Dans le jeu III, un seul point est à 3,24 de sa droite ; sans lui, les dix autres sont alignés aux arrondis près ($r = 0{,}999997$, les valeurs d'Anscombe n'ayant que deux décimales), sur une autre droite, de pente 0,345 au lieu de 0,5. Le jeu IV (2.12) est le cas extrême : un seul point fabrique toute la pente et toute la corrélation.
**Erreurs fréquentes** : mélanger les ddof dans `fit_line` (covariance avec ddof = 1, variance avec ddof = 0 : la pente est fausse de 10 %) ; oublier la valeur absolue en b ; des axes différents d'un panneau à l'autre, qui trompent l'œil.
**Variante** : ajoute la médiane et l'écart interquartile (percentile 75 − percentile 25) de $x$ et de $y$ au tableau des statistiques : ces résumés robustes distinguent déjà les jeux III et IV (dans le jeu IV, la médiane de $x$ vaut 8 et son écart interquartile 0).

### Ex 2.31 — Docstring et test pytest pour `zscore` 🛠️
Tes tests passent sur `zscore_ok` et attrapent les trois versions buggées ; la docstring a ses quatre sections et doctest valide ses exemples.
**Démarche** : voir `05_solutions.ipynb` (la docstring complète de `flag_outliers`, avec trois exemples, et trois tests). L'essentiel :
```python
def test_axis_0_standardizes_every_column():
    X = np.array([[1.0, 100.0], [2.0, 300.0], [4.0, 200.0], [7.0, 900.0]])
    Z = zscore(X, axis=0)
    assert Z.mean(axis=0) == pytest.approx([0, 0], abs=1e-12)
    assert Z.std(axis=0) == pytest.approx([1, 1])
```
**Pourquoi** : pour `[1, …, 1, 10]` (neuf fois 1), la moyenne vaut 1,9, l'écart-type 2,7, et le z-score de 10 vaut **exactement** 3 : avec `threshold=3.0`, le test « strictement plus grand » tomberait pile sur la limite, et le résultat dépendrait du dernier chiffre d'un arrondi. Un exemple de documentation doit être net. Chaque test vise un bug : la propriété attrape la division par la variance (bug 1), le cas limite attrape la version qui renvoie des zéros au lieu de refuser (bug 2), le test de l'axe attrape celle qui ignore `axis` (bug 3), à condition que les colonnes n'aient pas toutes la même moyenne et le même écart-type.
**Erreurs fréquentes** : un test d'axe sur un tableau dont les colonnes sont semblables (le bug 3 passe) ; comparer des flottants avec `==` sans `pytest.approx` ; un exemple `>>>` dont la sortie a été écrite de mémoire au lieu d'être recopiée (doctest compare le texte exact, espaces compris).
**En entreprise** : une docstring avec des exemples exécutables et des tests qui ciblent chacun un cas limite, c'est ce qu'on attend d'une fonction livrée dans une librairie ; `pytest --doctest-modules` vérifie les exemples de tout un package.
**Variante** : ajoute un test pour `ddof=1`, avec `scipy.stats.zscore(x, ddof=1)` comme oracle, et un test paramétré (`@pytest.mark.parametrize`) sur plusieurs tableaux.

### Ex 2.32 — Mêmes statistiques, autre dessin : fabrique ton quartet 🏆
Objectif atteint quand les cinq statistiques sont à 0,01 près de celles d'Anscombe et que le nuage est une image affine du cœur ; le corrigé donne exactement 9, 7,5, 11, 4,125 et 0,816.
**Démarche** :
```python
def make_heart_dataset(n=40):
    hx, hy = heart_points(n)
    zx = (hx - hx.mean()) / hx.std()
    e = hy - hy.mean() - np.mean((hy - hy.mean()) * zx) * zx     # Cov(e, zx) = 0
    ze = e / e.std()
    r = 0.816
    zy = r * zx + np.sqrt(1 - r ** 2) * ze                       # std 1, correlation r with zx
    return 9 + np.sqrt(11 * (n - 1) / n) * zx, 7.5 + np.sqrt(4.125 * (n - 1) / n) * zy
```
**Pourquoi ça marche** : $\mathrm{Var}(z_y) = r^2\,\mathrm{Var}(z_x) + (1 - r^2)\,\mathrm{Var}(z_e) + 2\,r\sqrt{1 - r^2}\,\mathrm{Cov}(z_x, z_e) = r^2 + 1 - r^2 + 0 = 1$, et $\mathrm{Cov}(z_x, z_y) = r\,\mathrm{Var}(z_x) + \sqrt{1 - r^2}\,\mathrm{Cov}(z_x, z_e) = r$. Les étapes 1 et 4 ne changent pas la corrélation (2.8) ; elles règlent les moyennes et les variances. La droite de régression suit : pente $r\,\frac{s_y}{s_x} = 0{,}816 \times \sqrt{4{,}125 / 11} \approx 0{,}500$, ordonnée à l'origine $7{,}5 - 0{,}5 \times 9 = 3$.
**Erreurs fréquentes** : $x = 9 + \sqrt{11}\,z_x$ donne une variance (ddof = 1) de $11 \times \frac{40}{39} \approx 11{,}28$ : hors de l'objectif ; construire $y$ à partir de $h_x$ seulement (le cœur s'aplatit en un segment) ; oublier de standardiser $e$ (l'écart-type de $z_y$ n'est plus 1).
**Pour aller plus loin** : Matejka et Fitzmaurice (2017) déplacent les points un à un, en gardant les statistiques à deux décimales près, ce qui permet d'atteindre n'importe quelle forme, pas seulement une image affine (le Datasaurus, fiche 🕰️) ; essaie avec un cercle, une étoile ou tes initiales.
