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
**Erreur fréquente** : lire la hauteur d'une densité comme une probabilité.

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

### Ex 2.1 — Moyenne, médiane et mode d'une liste de salaires
a) $\frac{31\,700}{9} \approx 3\,522{,}2$, soit **3 522 €** · b) **2 400 €** · c) **2 100 €** · d) **8** · e) $\frac{49\,700}{9} \approx 5\,522{,}2$, soit **5 522 €** · f) **2 400 €**.
**Pourquoi** : triés, les salaires sont 1 900 ; 2 100 ; 2 100 ; 2 300 ; **2 400** ; 2 600 ; 2 800 ; 3 500 ; 12 000. La médiane est la 5ᵉ valeur. Le salaire de la directrice pèse à lui seul 38 % de la somme : il tire la moyenne si haut que 8 salariés sur 9 gagnent moins que « le salaire moyen ». Quand il passe à 30 000 €, la somme augmente de 18 000 €, la moyenne de 2 000 €, mais la médiane ne bouge pas : 30 000 € reste la plus grande valeur, à la même place.
g) Réponse modèle : « En général, on gagne autour de 2 400 € (la médiane) ; la plupart des salaires sont entre 2 100 et 2 800 €. » Donner la moyenne (3 522 €) serait trompeur : elle ne correspond au salaire de presque personne.
**Erreurs fréquentes** : prendre la valeur du milieu de la liste **non triée** (2 800 €) ; diviser par 10 ; oublier de recalculer la moyenne en e.
**À retenir** : pour des salaires, des prix de logements ou des temps de réponse, toujours donner la médiane, et regarder l'histogramme.

### Ex 2.2 — De la casse de voitures à la distribution de probabilité
a) $\frac{224}{800} = $ **0,28** · b) **[0,22 ; 0,18 ; 0,28 ; 0,12 ; 0,20]** · c) $0{,}28 \times 360° = $ **100,8°** · d) $1 - 0{,}12 = $ **0,88** · e) $50 \times 0{,}18 = $ **9** · f) **[0,22 ; 0,40 ; 0,68 ; 0,80 ; 1,00]** · g) **SUV**.
**Pourquoi** : diviser par le total **normalise** les comptages (somme 1). La roue répartit 360° au prorata des probabilités. d utilise l'événement contraire. e est une espérance : sur 50 tirages avec remise, on attend en moyenne $50 \times p$ pick-up (parfois 7, parfois 12…). Pour g, les segments sont berline $[0 ; 0{,}22)$, pick-up $[0{,}22 ; 0{,}40)$, monospace $[0{,}40 ; 0{,}68)$, SUV $[0{,}68 ; 0{,}80)$ et break $[0{,}80 ; 1)$ : 0,71 tombe dans celui des SUV.
**Erreurs fréquentes** : donner des comptages ou des pourcentages au lieu de probabilités ; en g, prendre le dernier type dont la somme cumulée est inférieure à $u$ (monospace) au lieu du premier dont la somme cumulée **dépasse** $u$.
**Variante** : si $u$ tombe **pile** sur une somme cumulée (0,68), il appartient au segment suivant, car chaque segment inclut sa borne gauche et exclut sa borne droite. C'est ce que fait `np.searchsorted(np.cumsum(p), u, side="right")`, l'algorithme de `mylearn.stats.sample_categorical` (2.19).

### Ex 2.3 — La règle 68-95-99,7 sur les nageoires des manchots
a) **[183,5 ; 196,5]** · b) **[177 ; 203]** · c) **0,025** · d) $151 \times 0{,}025 = 3{,}775$, soit **4** · e) $\frac{210 - 190}{6{,}5} \approx$ **3,08**.
**Pourquoi** : 203 mm est exactement $\mu + 2\sigma$ ; 5 % des Adélie sortent de l'intervalle à 95 %, moitié au-dessus, moitié au-dessous, par symétrie.
f) Réponse modèle : une nageoire de 210 mm est à plus de 3 écarts-types au-dessus de la moyenne des Adélie. Moins de 0,15 % des Adélie dépassent $\mu + 3\sigma$ : parmi 151 Adélie, on en attendrait environ 0,2. Ce n'est pas impossible, mais c'est très improbable ; c'est bien plus vraisemblablement un Gentoo, dont la nageoire mesure en moyenne près de 30 mm de plus (environ 217 mm contre 190 mm, fiche §2.4). Tu confronteras ce modèle aux vraies données en 2.18.
**Erreurs fréquentes** : répondre 0,05 en c (on oublie que les 5 % se partagent entre les deux côtés) ; tronquer 3,0769 en 3,07.

### Ex 2.4 — Variance : diviser par N ou par N − 1 ?
a) **6** · b) **30** · c) $\frac{30}{5} = $ **6** · d) $\frac{30}{4} = $ **7,5** · e) $\sqrt{6} \approx$ **2,449** · f) $\sqrt{7{,}5} \approx$ **2,739** · g) $\frac{16 + 49 + 36 + 9 + 100}{5} = $ **42**, et $42 - 6^2 = 6$ : on retrouve c · h) **6** · i) $\sqrt{6} \times 1\,000 \approx 2\,449{,}49$, soit **2 449,5** ms (pars de $\sqrt{6}$ : l'arrondi 2,449 de e donnerait 2 449,0).
**Pourquoi** : les écarts à la moyenne sont −2, 1, 0, −3, 4 (leur somme vaut toujours 0, d'où le carré). Ajouter 100 à chaque valeur ajoute aussi 100 à la moyenne : les écarts, donc la variance, ne changent pas. Multiplier par 1 000 multiplie les écarts par 1 000, la variance par $1\,000^2$ et l'écart-type par 1 000 : $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$ (0B).
j) Réponse modèle : ddof = 1. On estime la dispersion de **tous** les chargements (la population) à partir de 5 mesures (un échantillon) ; les écarts sont mesurés autour de la moyenne de l'échantillon, trop proche de ces 5 valeurs, et diviser par $n$ sous-estime la variance. Avec seulement 5 mesures, l'écart entre les deux vaut 25 % ; avec 1 000 mesures, il serait négligeable.
**Erreurs fréquentes** : oublier le carré (la somme des écarts vaut 0) ; confondre variance et écart-type ; en i, multiplier l'écart-type par $1\,000^2$.

### Ex 2.5 — Espérances : Bernoulli, multinoulli et jeu de hasard
a) **0,3** · b) $0{,}3 \times 0{,}7 = $ **0,21** · c) $200 \times 0{,}3 = $ **60** · d) $\frac{1 + 20}{2} = $ **10,5** · e) $0{,}1 + 0{,}4 + 0{,}9 + 1{,}6 = $ **3,0** · f) **[0, 0, 1, 0]** · g) **[0,1 ; 0,2 ; 0,3 ; 0,4]** · h) **−0,25 €** · i) **100 €**.
**Pourquoi** : en g, la composante n° $k$ du vecteur one-hot vaut 1 quand on tire la classe $k$ (probabilité $p_k$) et 0 sinon : c'est une variable de Bernoulli d'espérance $p_k$. La moyenne des vecteurs one-hot se rapproche donc du vecteur des probabilités. C'est pour cela qu'un classifieur qui prédit des probabilités s'entraîne avec des labels en one-hot (ch. 6 et 17). En h, les gains nets valent 18 € (probabilité $\frac{1}{20}$), 3 € ($\frac{3}{20}$) et −2 € ($\frac{16}{20}$) : $\frac{18 + 9 - 32}{20} = -0{,}25$ €. En i, l'organisateur gagne ce que perdent les joueurs : $400 \times 0{,}25 = 100$ €.
**Erreurs fréquentes** : en b, répondre $p$ ou $p^2$ ; en e, faire la moyenne simple de 1, 2, 3, 4 (2,5) ; en h, oublier de déduire la mise (+1,75 €), ou croire que la mise est rendue quand on gagne (+0,15 €).
**Variante** : quel gain faudrait-il promettre pour le 20 pour que le jeu soit équitable (espérance nulle) ? Il faut 0,25 € de plus par partie en moyenne, soit 5 € de plus pour un 20 (probabilité $\frac{1}{20}$) : 25 €.

### Ex 2.6 — Compter les tirages avec et sans remise
a) $\binom{5}{2} = $ **10** (AB, AC, AD, AE, BC, BD, BE, CD, CE, DE) · b) **15** (les 10 paires précédentes, plus AA, BB, CC, DD et EE) · c) $5^2 = $ **25** · d) $\binom{7}{3} = $ **35** · e) $5^5 = $ **3 125** · f) $0{,}8^5 \approx$ **0,3277** · g) $0{,}999^{1\,000} \approx$ **0,3677** · h) $1 - 0{,}999^{1\,000} \approx$ **0,6323**.
**Pourquoi** : en d, il y a $7 \times 6 \times 5 = 210$ suites sans remise, et chaque groupe de 3 correspond à $3! = 6$ suites. En f et g, un exemple donné échappe à un tirage avec probabilité $1 - \frac{1}{n}$, et les $n$ tirages sont indépendants (avec remise). g est déjà très proche de la limite $e^{-1} \approx 0{,}3679$.
i) Réponse modèle : tirer $n$ éléments parmi $n$ **sans** remise redonne toujours exactement l'échantillon (dans un autre ordre). La statistique (une moyenne, par exemple) serait la même à chaque rééchantillon : la « distribution » bootstrap serait réduite à une seule valeur, et ne dirait rien de la variabilité.
**Erreurs fréquentes** : compter les suites quand l'ordre ne compte pas (20 au lieu de 10 en a) ; en f et g, donner la limite $e^{-1}$ au lieu de la valeur exacte ; en h, donner la part des absents, ou la limite $1 - e^{-1} \approx 0{,}6321$.

### Ex 2.7 — Covariance et corrélation de cinq points à la main
a) **[3 ; 4]** · b) **9** · c) $\frac{9}{5} = $ **1,8** · d) $\frac{9}{4} = $ **2,25** · e) **[1,414 ; 1,414]** · f) $\frac{1{,}8}{\sqrt{2} \times \sqrt{2}} = $ **0,9** · g) $60 \times 1{,}8 = $ **108** · h) **0,9**.
**Pourquoi** : écarts de $x$ : −2, −1, 0, 1, 2 ; écarts de $y$ : −2, −1, 1, 0, 2 ; produits : 4, 1, 0, 0, 4, de somme 9. Les sommes des carrés valent 10 pour $x$ comme pour $y$, d'où des variances de 2 et des écarts-types de $\sqrt{2}$. Passer aux minutes multiplie la covariance par 60 (et l'écart-type de $x$ par 60) : la corrélation ne change pas (démonstration en 2.8).
i) Réponse modèle : non. Ce sont des **observations**, pas une expérience : les étudiants qui révisent plus sont peut-être aussi plus motivés, ou avaient déjà un meilleur niveau (variables de confusion), et 5 points ne suffisent pas à conclure. Pour mesurer l'effet d'une heure de révision, il faudrait une expérience : tirer au sort qui révise plus longtemps.
**Erreurs fréquentes** : mélanger ddof = 1 pour la covariance et ddof = 0 pour les écarts-types (on trouve alors 1,125, impossible pour une corrélation) ; croire que la corrélation change avec l'unité.

### Ex 2.8 — Changer d'unité : la covariance bouge, pas la corrélation

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

### Ex 2.9 — Le bootstrap en cinq lignes (réponse modèle) 🗣️
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
   
   Dans l'Union européenne, les systèmes d'IA utilisés pour recruter ou trier des candidatures sont classés **à haut risque** par l'AI Act (règlement (UE) 2024/1689, annexe III, point 4) : gestion des risques, qualité et gouvernance des données, contrôle humain et documentation y sont obligatoires.
   
   *Sources :* [règlement (UE) 2024/1689 (AI Act), EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) ; [RGPD, EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng) ; [Code du travail, article L1132-1, Légifrance](https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000045391841).

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

Les solutions des exercices 2.13 à 2.32 arriveront avec le notebook, à la prochaine session de génération.
