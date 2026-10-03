# Checkpoint II · Corrigé détaillé et barème de l'examen blanc

> À lire **après** l'examen, et après la partie B du notebook (vérification automatique des réponses chiffrées). Pour chaque question : la réponse, la démarche, le barème, les erreurs fréquentes et les exercices à refaire si tu as eu moins de la moitié des points. Les solutions exécutées des questions de code sont dans `03_examen_solutions.ipynb`.

## Règles de notation

- **Barème par sous-question**, indiqué entre crochets. Une réponse juste sans démarche, quand la question en demande une, rapporte la moitié des points.
- **Méthode juste, erreur de calcul** : la moitié des points de la sous-question. **Erreur reportée** : si une réponse fausse est réutilisée correctement plus loin, la suite n'est pas pénalisée une seconde fois.
- **Arrondis** : une décimale de trop ou de moins coûte au plus 0,05 point par question ; un arrondi fait trop tôt, qui change la dernière décimale demandée, aussi.
- **Lecture de graphique** (CP2.7 c) : la tolérance est indiquée ; toute lecture dans l'intervalle rapporte tous les points.
- **Questions rédigées** (CP2.1, CP2.3 j, CP2.6 1 à 3, l'interprétation de d et e, CP2.7 f, CP2.8 f et g, CP2.10 a et d, CP2.12) : les critères sont listés ; une idée juste formulée autrement compte autant.

---

## CP2.1 — Vrai ou faux justifiés 🧠 · 2,5 points

Chaque affirmation : **[0,1]** pour le verdict ; **[0,2]** pour une justification juste (**[0,25]** pour b et g), la moitié si elle est incomplète.

**a) Faux** (ch. 7). k-means compare des distances euclidiennes : changer l'échelle d'une feature change les distances, donc les affectations et les centres. Chez les manchots, la masse (en grammes, écart-type d'environ 800) écrase la longueur du bec (en millimètres, écart-type d'environ 5) : sans standardisation, la distance entre deux manchots est presque la seule différence de masse, et les clusters se découpent selon la masse. Standardiser (z-scores, avec les statistiques de l'entraînement) redonne à chaque feature un poids comparable.

**b) Faux** (ch. 7). Le cube central a un côté deux fois plus petit, donc un volume $\left(\frac{1}{2}\right)^{10} = \frac{1}{1\,024}$ fois celui du grand cube : environ **0,1 %** des points y tombent, pas plus de la moitié (0,25 pour ce calcul). En grande dimension, presque tout le volume est loin du centre, dans les couches extérieures et vers les coins ; c'est la même idée que l'écorce de l'orange, $1 - (1 - \varepsilon)^d$.

**c) Faux** (ch. 8). Le meilleur des 50 scores a été **choisi parce qu'il était le plus haut** : même si les 50 réglages se valaient, le maximum de 50 estimations bruitées dépasse en moyenne leur vraie valeur. C'est le biais d'optimisme de la sélection ; seul un jeu jamais consulté pendant le choix, le test, mesure honnêtement le réglage retenu.

**d) Faux** (ch. 8). Diviser par $\sqrt{k}$ suppose $k$ scores indépendants. Or les jeux d'entraînement des tours se recouvrent largement (avec $k = 5$, deux tours partagent les trois quarts de leurs exemples) : les scores sont corrélés. Il n'existe d'ailleurs aucun estimateur sans biais de cette variance qui vaille pour toutes les distributions des données (Bengio et Grandvalet, 2004) ; $\sigma_s / \sqrt{k}$ sous-estime en général l'incertitude, et ne sert que d'ordre de grandeur (le mini-projet, que tu feras ensuite, s'en sert ainsi pour sa règle « d'une erreur type »).

**e) Vrai** (ch. 9). Avec $\lambda = 0$, la solution minimise exactement l'erreur d'entraînement ; augmenter $\lambda$ donne plus de poids à la pénalité, et la solution échange de l'erreur d'entraînement contre des poids plus petits. Cette idée rapporte la moitié de la justification (0,1) : elle compare $\lambda = 0$ à $\lambda > 0$, pas deux valeurs quelconques. Toute la justification (0,2) demande l'argument pour deux valeurs $\lambda_1 < \lambda_2$ : si l'optimum de $\lambda_2$ avait une erreur plus basse, il lui faudrait des poids plus grands (sinon il battrait l'optimum de $\lambda_1$ sur l'objectif de $\lambda_1$), et ces poids plus grands lui coûteraient plus avec $\lambda_2$ qu'avec $\lambda_1$ : une contradiction. La preuve complète, pour $\lambda_1 < \lambda_2$ de minima $w_1$ et $w_2$ : $E(w_1) + \lambda_1 \lVert w_1 \rVert^2 \le E(w_2) + \lambda_1 \lVert w_2 \rVert^2$ et $E(w_2) + \lambda_2 \lVert w_2 \rVert^2 \le E(w_1) + \lambda_2 \lVert w_1 \rVert^2$ ; en les additionnant, $(\lambda_2 - \lambda_1)(\lVert w_2 \rVert^2 - \lVert w_1 \rVert^2) \le 0$, donc $\lVert w_2 \rVert \le \lVert w_1 \rVert$, puis la première inégalité donne $E(w_1) \le E(w_2)$. CP2.6 c) le vérifie sur un exemple (1,9 puis 2,8).

**f) Faux** (ch. 9). $R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$ compare l'erreur du modèle à celle de la moyenne $\bar{y}$ du jeu mesuré : il vaut 1 pour un modèle parfait, 0 pour un modèle qui ne fait pas mieux que la moyenne, et il devient **négatif** quand le modèle fait pire. Sur ses données d'entraînement, une régression linéaire avec ordonnée à l'origine ajustée par moindres carrés a un $R^2$ entre 0 et 1 ; sur un jeu de validation, rien ne l'empêche de devenir négatif, par exemple pour un modèle qui surapprend. Contre-exemple : cibles 1, 2, 3 et prédictions 3, 2, 1, soit $\sum (y - \hat{y})^2 = 8$, $\sum (y - \bar{y})^2 = 2$ et $R^2 = 1 - 4 = -3$. Malgré son nom, le $R^2$ n'est pas un carré.

**g) Vrai** (ch. 10). Par récurrence : au départ, $(\mathbf{w}, b) = (0, 0)$ dans les deux cas. Si, avant un exemple, les poids avec $\eta = 10$ valent 10 fois ceux avec $\eta = 1$, alors $z$ est multiplié par 10, son signe ne change pas (une somme nulle reste nulle), et le test $y\,z \le 0$ donne la même réponse ; la correction éventuelle, $\eta\,y\,\mathbf{x}$, est elle aussi multipliée par 10. Mêmes erreurs aux mêmes moments, donc mêmes prédictions (0,25 pour ce raisonnement). Attention : c'est faux avec un départ aléatoire (10.17).

**h) Vrai** (ch. 11). Un syllogisme valide garantit que, si les prémisses sont vraies, la conclusion l'est aussi. Si la conclusion est fausse, les prémisses ne peuvent donc pas être toutes vraies (c'est un *modus tollens*). Exemple : « tous les nombres premiers sont impairs ; 2 est premier ; donc 2 est impair » est valide, sa conclusion est fausse, et sa majeure aussi.

**Erreurs fréquentes** : en b), répondre à l'intuition du dessin en 2D ou en 3D ; en c), confondre « le modèle n'a pas appris sur la validation » et « le score de validation est honnête » ; en f), croire qu'un $R^2$, parce qu'il porte un carré dans son nom, ne peut pas être négatif ; en g), oublier que la règle compte une somme nulle comme une erreur (avec $\eta$, le seuil 0 ne bouge pas).
**Remédiation** : a) fiche du ch. 7, §7.5 (« Quand k-means échoue »), 7.17, 7.E2 ; b) 7.Q11, 7.21 ; c) 8.6, 8.Q8, 8.17 ; d) 8.5, fiche du ch. 8, §8.5.1 ; e) 9.3, 9.Q7 ; f) 9.1, 9.14, fiche du ch. 9, §9.2 (encadré sur la MSE, la RMSE, la MAE et le $R^2$) ; g) 10.17, 10.6 ; h) 11.Q7, 11.3.

## CP2.2 — Un-contre-tous, un-contre-un ✏️ · 1 point

a) **[0,15]** $N_{\text{OvR}} = K = 12$ et $N_{\text{OvO}} = \frac{K(K-1)}{2} = \frac{12 \times 11}{2} = 66$ : **[12, 66]**.
b) **[0,15]** $\frac{14 \times 13}{2} = 91 < 100$ et $\frac{15 \times 14}{2} = 105 \ge 100$ : **15** classes.
c) **[0,15]** A gagne A-B et A-D ; B gagne B-D ; C gagne A-C, B-C et C-E ; D gagne C-D et D-E ; E gagne A-E et B-E : **[2, 1, 3, 2, 2]**. Vérification : 10 voix, une par duel.
d) **[0,1]** **C**, seule classe à 3 voix.
e) **[0,15]** **[1, 3, 2, 3, 1]** (A : A-E ; B : A-B, B-C, B-E ; C : A-C, C-D ; D : A-D, B-D, D-E ; E : C-E).
f) **[0,15]** B et D sont à égalité avec 3 voix ; la règle de `mylearn` garde le plus petit indice : **B** (indice 1, contre 3 pour D).
g) **[0,15]** Le duel B-D a été gagné par D : **D**. Les deux règles donnent deux prédictions différentes : une règle d'égalité se choisit et se documente.

**Erreurs fréquentes** : $K^2$ ou $K(K-1)$ duels (chaque paire n'est jouée qu'une fois, contre une autre classe) ; un décompte faux (le total doit faire 10) ; en f), appliquer la règle du duel au lieu de celle de l'indice.
**Remédiation** : 7.1, 7.2, 7.Q6, 7.Q7, 7.23.

## CP2.3 — Une itération de k-means ✏️ · 2 points

Les distances au carré aux centres de départ $c_1 = P_2 = (3 ; 6)$ et $c_2 = P_3 = (4 ; 6)$ :

| Point | $d^2(\cdot, c_1)$ | $d^2(\cdot, c_2)$ | Cluster |
|---|---|---|---|
| $P_1\,(2 ; 6)$ | 1 | 4 | 1 |
| $P_2\,(3 ; 6)$ | 0 | 1 | 1 |
| $P_3\,(4 ; 6)$ | 1 | 0 | 2 |
| $P_4\,(6 ; 4)$ | 9 + 4 = 13 | 4 + 4 = 8 | 2 |
| $P_5\,(9 ; 5)$ | 36 + 1 = 37 | 25 + 1 = 26 | 2 |
| $P_6\,(9 ; 6)$ | 36 | 25 | 2 |

a) **[0,2]** **[1, 1, 2, 2, 2, 2]**.
b) **[0,2]** $1 + 0 + 0 + 8 + 26 + 25 =$ **60**.
c) **[0,2]** $c_1 = \frac{P_1 + P_2}{2} = (2{,}5 ; 6)$ et $c_2 = \frac{P_3 + P_4 + P_5 + P_6}{4} = \left(\frac{28}{4} ; \frac{21}{4}\right) = (7 ; 5{,}25)$ : **[[2,5 ; 6], [7 ; 5,25]]**.
d) **[0,2]** $0{,}25 + 0{,}25 + (9 + 0{,}5625) + (1 + 1{,}5625) + (4 + 0{,}0625) + (4 + 0{,}5625) =$ **21,25** : la mise à jour a fait baisser l'inertie de 60 à 21,25.
e) **[0,2]** $P_3$ est à $1{,}5^2 = 2{,}25$ de $c_1$ et à $9 + 0{,}5625 = 9{,}5625$ de $c_2$ : il passe dans le cluster 1. $P_4$ reste dans le cluster 2 ($3{,}5^2 + 2^2 = 16{,}25$ contre $2{,}5625$). Le point qui change : **3**.
f) **[0,2]** $c_1 = \frac{P_1 + P_2 + P_3}{3} = (3 ; 6)$ et $c_2 = \frac{P_4 + P_5 + P_6}{3} = (8 ; 5)$ : **[[3 ; 6], [8 ; 5]]**.
g) **[0,2]** $1 + 0 + 1 + (4 + 1) + (1 + 0) + (1 + 1) =$ **10**.
h) **[0,1]** **Faux** : avec ces centres, chaque point est déjà avec le centre le plus proche ($P_4$ : 13 contre 5 ; $P_3$ : 1 contre 17) ; l'algorithme s'arrête.
i) **[0,3]** Les $D^2$ depuis $P_1$ valent 1, 4, 20, 50 et 49 pour $P_2$ à $P_6$ (0 pour $P_1$ lui-même), de somme 124 (0,15). $P(\text{l'un de } P_4, P_5, P_6) = \frac{20 + 50 + 49}{124} = \frac{119}{124} \approx$ **0,960** (0,15).
j) **[0,2]** « Mauvais » parce que les deux centres de départ sont dans le même groupe naturel, celui de gauche ($P_1$, $P_2$, $P_3$) : le groupe de droite n'a pas de centre à lui, et un centre doit d'abord servir à la fois $P_3$ et tout le groupe de droite (0,1). Ici, Lloyd s'en sort avec une itération de plus et trouve quand même le meilleur découpage (inertie 10), mais un tel départ peut piéger l'algorithme dans un minimum local (deux centres dans un groupe, deux groupes fusionnés). k-means++ tire le deuxième centre avec une probabilité proportionnelle à $D^2$ : 96 % de chances qu'il tombe à droite (i) ; on garde en plus le meilleur de plusieurs départs, `n_init` (0,1).

**Erreurs fréquentes** : des distances sans le carré pour l'inertie ; les centres de départ réutilisés au lieu des nouveaux ; en i), un tirage uniforme (3 chances sur 5) ou $P_1$ compté parmi les candidats.
**Remédiation** : 7.3, 7.25, 7.26, 7.Q8 ; fiche du ch. 7, §7.5 (encadrés sur Lloyd et k-means++).

## CP2.4 — Densité et hyper-orange ✏️ · 1 point

a) **[0,2]** $\frac{2\,000}{4^3} =$ **31,25** et $\frac{2\,000}{4^6} = \frac{2\,000}{4\,096} \approx$ **0,488** : en dimension 6, il y a deux fois plus de cases que d'échantillons, donc plus de la moitié des cases sont forcément vides (environ 61 % pour des points tirés uniformément, puisque $(1 - 1/4\,096)^{2\,000} \approx 0{,}61$). Une densité est une moyenne, pas une probabilité.
b) **[0,15]** $n = \rho\,b^d = 2 \times 4^8 = 2 \times 65\,536 =$ **131 072**.
c) **[0,2]** $\frac{2\,000}{4^8} \approx 0{,}031$ et $\frac{2\,000}{4^9} \approx 0{,}0076$ : **9**.
d) **[0,15]** Le centre d'un ballon a ses 9 coordonnées égales à ±2 : il est à $\sqrt{9 \times 2^2} = 2\sqrt{9} = 6$ de l'origine. L'orange s'arrête au bord du ballon : $r = 6 - 1 =$ **5**. En général, $r(d) = 2\sqrt{d} - 1$.
e) **[0,15]** L'orange sort quand son rayon dépasse la demi-largeur de la boîte : $2\sqrt{d} - 1 > 3 \iff \sqrt{d} > 2 \iff d > 4$. En dimension 4, elle touche les faces ($r = 3$) ; elle sort en dimension **5**.
f) **[0,15]** $1 - 0{,}95^{30} \approx 1 - 0{,}215 =$ **0,785** : plus des trois quarts du volume sont dans une peau de 5 % du rayon.

**Erreurs fréquentes** : $b \times d$ cases au lieu de $b^d$ ; la formule $\sqrt{d} - 1$ de la fiche appliquée telle quelle (elle vaut pour des ballons centrés en ±1) ; « sort » confondu avec « touche » en e) ; $0{,}95^{30}$, la part du volume à l'intérieur, donnée pour la peau.
**Remédiation** : 7.4, 7.5, 7.Q9, 7.Q11, 7.21.

## CP2.5 — Plan d'évaluation ✏️ · 1,5 point

a) **[0,15]** $\lceil 0{,}2 \times 333 \rceil = \lceil 66{,}6 \rceil =$ **67** ; il reste 266 manchots.
b) **[0,2]** $266 = 5 \times 53 + 1$ : le premier fold reçoit un manchot de plus, **[54, 53, 53, 53, 53]**.
c) **[0,2]** $3 \times 5 = 15$ réglages, 5 entraînements chacun, plus le réentraînement final : $15 \times 5 + 1 =$ **76**.
d) **[0,2]** Chaque tour extérieur refait toute la recherche de c) : $5 \times 76 =$ **380**, sans la recherche finale qui produirait le modèle livré (elle ajouterait 76 entraînements, 456 en tout). Le prix d'une estimation honnête de toute la procédure.
e) **[0,25]** $\mathrm{SE} = \sqrt{0{,}94 \times 0{,}06 / 67} \approx 0{,}0290$, et $1{,}96 \times 0{,}0290 \approx 0{,}0569$ : **[0,029 ; 0,057]**. L'intervalle va d'environ 0,883 à 0,997 : sur 67 manchots, 0,94 n'est connu qu'à ±6 points près. La formule $\hat{p} \pm 1{,}96\,\mathrm{SE}$ est grossière avec si peu d'erreurs (environ 4) : d'autres intervalles (celui de Wilson, par exemple) sont plus justes, mais l'ordre de grandeur est le bon.
f) **[0,5]** (0,1 par protocole bien classé) Fuites : **1, 2 et 5**.
- **1** : la moyenne des masses connues contient celles des futurs manchots de test ; le prétraitement a vu le test avant le découpage (une petite fuite, mais une fuite).
- **2** : la standardisation a vu les manchots des folds de validation : la validation croisée est (un peu) optimiste. Le test reste propre, puisqu'il est hors de ce calcul.
- **3** : correct, c'est la bonne pratique.
- **4** : correct : choix par validation croisée, test consulté une fois.
- **5** : chaque retour au test est un choix fait en le regardant ; le dernier score est optimiste.

**Erreurs fréquentes** : arrondir la taille du test vers le bas ; faire la validation croisée sur les 333 manchots ; oublier le réentraînement final ; prendre $n = 333$ dans l'erreur type ; juger le protocole 2 « sans fuite puisque le test n'y est pas ».
**Remédiation** : 8.1, 8.2, 8.3, 8.4, 8.Q6, 8.Q10.

## CP2.6 — Ridge en dimension 1 ∂ · 2 points

1. **[0,25]** $\frac{\partial L}{\partial b} = -2 \sum_i (y_i - w\,x_i - b) = 0 \iff \sum_i y_i - w \sum_i x_i - n\,b = 0 \iff b = \bar{y} - w\,\bar{x}$.

2. **[0,35]** Avec $b = \bar{y} - w\,\bar{x}$, chaque résidu s'écrit $y_i - w\,x_i - b = (y_i - \bar{y}) - w\,(x_i - \bar{x})$ (0,1). Donc
$$L(w) = \sum_i \big((y_i - \bar{y}) - w\,(x_i - \bar{x})\big)^2 + \lambda\,w^2 = S_{yy} - 2\,w\,S_{xy} + w^2\,(S_{xx} + \lambda),$$
avec $S_{yy} = \sum_i (y_i - \bar{y})^2$. $L'(w) = -2\,S_{xy} + 2\,w\,(S_{xx} + \lambda) = 0 \iff w^* = \frac{S_{xy}}{S_{xx} + \lambda}$ (0,15). $L''(w) = 2\,(S_{xx} + \lambda) > 0$ dès que les $x_i$ ne sont pas tous égaux (ou que $\lambda > 0$) : une parabole tournée vers le haut, donc un minimum (0,1).

3. **[0,25]** $w^* = \frac{S_{xx}}{S_{xx} + \lambda}\,w_{\text{MC}}$ : la pente des moindres carrés multipliée par un facteur de **rétrécissement** compris entre 0 et 1, égal à 1 pour $\lambda = 0$ (0,1). Quand $\lambda \to +\infty$, $w^* \to 0$ et $b \to \bar{y}$ : le modèle prédit la **moyenne** $\bar{y}$ pour tout $x$ (0,1), pas 0, parce que $b$ n'est pas pénalisé : la pénalité aplatit la droite sans toucher à son niveau (0,05).

Données : $\bar{x} = 3$, $\bar{y} = 4$, écarts de $x$ : −2, −1, 0, 1, 2 ; écarts de $y$ : −2, −1, 1, 0, 2. $S_{xx} = 10$, $S_{xy} = 4 + 1 + 0 + 0 + 4 = 9$, $S_{yy} = 10$.

a) **[0,2]** $w^* = \frac{9}{10} = 0{,}9$ et $b = 4 - 0{,}9 \times 3 = 1{,}3$ : **[0,9 ; 1,3]**.
b) **[0,2]** $w^* = \frac{9}{15} = 0{,}6$ et $b = 4 - 0{,}6 \times 3 = 2{,}2$ : **[0,6 ; 2,2]**.
c) **[0,25]** $\lambda = 0$ : prédictions 2,2 ; 3,1 ; 4,0 ; 4,9 ; 5,8, résidus −0,2 ; −0,1 ; 1 ; −0,9 ; 0,2, somme des carrés **1,9**. $\lambda = 5$ : prédictions 2,8 ; 3,4 ; 4,0 ; 4,6 ; 5,2, résidus −0,8 ; −0,4 ; 1 ; −0,6 ; 0,8, somme **2,8**. Plus court : $S_{yy} - 2\,w\,S_{xy} + w^2 S_{xx}$, soit $10 - 16{,}2 + 8{,}1$ et $10 - 10{,}8 + 3{,}6$. Réponse : **[1,9 ; 2,8]**.
d) **[0,3]** Avec $x' = 10\,x$ : $\bar{x}' = 30$, $S_{x'x'} = 100\,S_{xx} = 1\,000$, $S_{x'y} = 10\,S_{xy} = 90$. $w' = \frac{90}{1\,005} \approx 0{,}08955$ et $b' = 4 - 30\,w' \approx 1{,}3134$ ; pour $x' = 50$ : $1{,}3134 + 50 \times 0{,}08955 \approx$ **5,791** (0,15). Le modèle de b) prédit $2{,}2 + 0{,}6 \times 5 = 5{,}2$ pour ce même point, et la droite des moindres carrés 5,8 : dans la nouvelle unité, $\lambda = 5$ ne rétrécit presque plus rien (facteur $\frac{1\,000}{1\,005} \approx 0{,}995$, contre $\frac{10}{15} \approx 0{,}67$ avant). Le même $\lambda$ n'a pas la même force selon l'unité des features (0,1) : on **standardise** les features avant Ridge, avec la moyenne et l'écart-type de l'entraînement, pour que la pénalité traite toutes les features de la même façon (0,05).
e) **[0,2]** Non. L'erreur d'entraînement est la plus petite pour $\lambda = 0$ (c : 1,9 < 2,8) : minimiser l'erreur d'entraînement choisirait toujours $\lambda = 0$, sans régularisation (0,1). $\lambda$ est un hyperparamètre : on le choisit sur une validation ou par validation croisée, en général sur une grille logarithmique, puis on évalue une fois sur le test (0,1).

**Erreurs fréquentes** : dériver en oubliant le facteur 2 ou le signe ; garder $b = 1{,}3$ en b) alors que $b$ dépend de $w$ ; utiliser $\sum x_i y_i$ et $\sum x_i^2$ (non centrés) avec une ordonnée à l'origine ; ajouter la pénalité à la somme des carrés des résidus en c).
**Remédiation** : 9.2, 9.3, 9.17, 9.Q7 ; fiche du ch. 9, §9.5 (encadré « Ridge en forme fermée »).

## CP2.7 — Trois paires de courbes d'apprentissage 📈 · 1,5 point

a) **[0,2]** **2** : les deux erreurs se rejoignent vite (environ 0,83) et ne baissent plus avec $n$, bien au-dessus du plancher de 0,5. C'est l'underfitting, un biais fort.
b) **[0,2]** **3** : avec 30 exemples, l'erreur d'entraînement vaut environ 0,13 et celle de validation environ 0,89. L'écart se referme à mesure que $n$ grandit (environ 0,48 et 0,52 à 2 000 exemples) : une forte variance quand les exemples sont peu nombreux.
c) **[0,25]** À $n = 100$ : validation ≈ 0,72, entraînement ≈ 0,26, écart ≈ **0,46** (accepte de 0,40 à 0,50).
d) **[0,2]** **B** : un modèle trop rigide reste trop rigide avec dix fois plus d'exemples (A) ; la régularisation (C) et l'early stopping (D) réduisent encore sa capacité.
e) **[0,2]** **Faux** : les courbes sont proches, mais à 0,83, loin du plancher de 0,5. Le niveau signale l'underfitting ; l'écart ne suffit pas à juger un modèle.
f) **[0,45]** (0,15 par panneau)
- **Panneau 1** : un bon compromis. Les deux erreurs approchent du plancher dès quelques centaines d'exemples ; il n'y a presque plus rien à gagner, sauf avec de meilleures features (qui feraient baisser le bruit irréductible). Suite : vérifier une fois sur le test.
- **Panneau 2** : underfitting. Un modèle plus souple (features polynomiales, interactions), de meilleures features, moins de régularisation. Plus de données ne servirait à rien.
- **Panneau 3** : overfitting quand les données sont rares. Plus de données (l'écart se referme), une régularisation plus forte, un modèle plus simple ou l'early stopping. Avec 2 000 exemples, il est presque au niveau du panneau 1 ; avec 100, le modèle du panneau 1 fait bien mieux.

**Erreurs fréquentes** : prendre le panneau 2 pour un bon modèle parce que ses courbes sont proches ; conseiller plus de données contre l'underfitting ; lire l'écart sur une autre taille que $n = 100$.
**Remédiation** : 9.9, 9.21, 9.Q1, 9.Q3 ; fiche du ch. 9, §9.3 (« Courbes de validation et courbes d'apprentissage »).

## CP2.8 — Perceptron : NAND, puis XOR ✏️ · 2 points

Première époque (y = +1 pour les trois premières entrées, −1 pour (1, 1)) :

| exemple | $y$ | $z$ avant | erreur ? | $\mathbf{w}$ après | $b$ après |
|---|---|---|---|---|---|
| (0, 0) | +1 | 0 | oui ($y z = 0$) | (0, 0) | 1 |
| (0, 1) | +1 | 1 | non | (0, 0) | 1 |
| (1, 0) | +1 | 1 | non | (0, 0) | 1 |
| (1, 1) | −1 | 1 | oui ($y z = -1$) | (−1, −1) | 0 |

a) **[0,3]** **[0, 1, 1, 1]**.
b) **[0,25]** **[−1, −1, 0]**.
c) **[0,15]** **2** corrections.
d) **[0,2]** Avec $\mathbf{w} = (-1, -1)$ et $b = 0$ : $z = 0$ pour (0, 0), qui donne −1 (faux) ; $z = -1$ pour (0, 1) et (1, 0), faux ; $z = -2$ pour (1, 1), juste. **1** entrée bien classée sur quatre.
e) **[0,3]** Deuxième époque : (0, 0) donne $z = 0$, erreur, $b = 1$ ; (0, 1) donne $z = -1 + 1 = 0$, erreur, $\mathbf{w} = (-1, 0)$, $b = 2$ ; (1, 0) donne $z = -1 + 2 = 1$, juste ; (1, 1) donne $z = -1 + 0 + 2 = 1$, erreur, $\mathbf{w} = (-2, -1)$, $b = 1$. Réponse : **[−2, −1, 1]** (3 corrections). En continuant, la règle converge à la 9ᵉ époque (la première sans correction), avec $\mathbf{w} = (-3, -2)$ et $b = 4$.
f) **[0,6]** Les quatre inégalités (0,2) : $b \le 0$ pour (0, 0) ; $w_2 + b > 0$ pour (0, 1) ; $w_1 + b > 0$ pour (1, 0) ; $w_1 + w_2 + b \le 0$ pour (1, 1). En additionnant les deux du milieu : $w_1 + w_2 + 2b > 0$ (0,15). En additionnant la première et la dernière : $w_1 + w_2 + 2b \le 0$ (0,15). Contradiction : aucun perceptron ne calcule XOR (0,1). En image : les entrées de sortie 1 sont sur une diagonale du carré, celles de sortie 0 sur l'autre, et aucune droite ne sépare deux diagonales qui se croisent.
g) **[0,2]** Ajouter le produit $x_3 = x_1 x_2$ (0,1). Par exemple $\mathbf{w} = (1, 1, -2)$, $b = -0{,}5$ : $z = -0{,}5$ pour (0, 0), $0{,}5$ pour (0, 1) et (1, 0), $1 + 1 - 2 - 0{,}5 = -0{,}5$ pour (1, 1), soit les sorties 0, 1, 1, 0 (0,1). D'autres réponses sont justes (une troisième entrée qui vaut AND, NAND ou OR des deux, $(x_1 - x_2)^2$…). Un réseau à deux couches apprend lui-même ce genre de feature (ch. 16).

**Erreurs fréquentes** : calculer tous les $z$ avec les poids de départ ; ne pas compter $z = 0$ comme une erreur ; coder la sortie 0 par $y = 0$ (l'exemple (1, 1) ne corrige alors rien) ; oublier de corriger le biais ; en d), prédire +1 pour $z = 0$.
**Remédiation** : 10.3, 10.4, 10.6, 10.21, 11.R1.

## CP2.9 — Syllogismes et sophismes ✏️ · 1 point

a) **[0,45]** (0,075 par lettre) **N S N V N N**.
1. « Tout $P$ est $M$ ; tout $S$ est $M$ ; donc tout $S$ est $P$ » ($M$ : « régularisé ») : le moyen terme n'est jamais distribué. Non valide.
2. Barbara, avec une mineure qui porte sur un individu ; les deux prémisses sont vraies (un estimateur sans biais a, par définition, une erreur systématique nulle, et la moyenne d'un échantillon en est un) : solide.
3. « Si $X$, alors $Y$ ; non $X$ ; donc non $Y$ » : non valide. Un modèle qui ne surapprend pas peut avoir un grand écart pour une autre raison (un jeu de validation d'une autre distribution que l'entraînement, par exemple).
4. Forme valide (*modus ponens*), mais majeure fausse : un modèle qui apprend par cœur atteint une erreur d'entraînement nulle et généralise mal. Valide, pas solide.
5. « Tout $M$ est $P$ ; aucun $S$ n'est $M$ ; donc aucun $S$ n'est $P$ » : non valide. Sa conclusion est d'ailleurs fausse : une régression linéaire sur 10 000 features a 10 000 poids.
6. « Si $X$, alors $Y$ ; $Y$ ; donc $X$ » : non valide. D'autres causes produisent le même effet (la masse présente deux fois parmi les features, une standardisation faite sur le mauvais axe).

b) **[0,4]** (0,1 par lettre) **E B C A** : 1, moyen terme non distribué ; 3, nier l'antécédent ; 5, majeur illicite (le prédicat « avoir beaucoup de paramètres » est distribué dans la conclusion, une proposition E, mais pas dans la majeure, une proposition A) ; 6, affirmer le conséquent.

c) **[0,15]** **Faux** : une induction peut échouer avec des observations exactes et nombreuses (des siècles de cygnes blancs, puis des cygnes noirs en Australie) ; sa conclusion n'est que probable.

**Erreurs fréquentes** : juger la validité d'après la vérité de la conclusion ; confondre nier l'antécédent et affirmer le conséquent ; confondre majeur et mineur illicite (regarde quel terme de la conclusion est distribué sans l'être dans sa prémisse : le prédicat, c'est le majeur).
**Remédiation** : 11.3, 11.4, 11.Q7, 11.Q8 ; fiche du ch. 11, §11.4 et §11.4.1.

## CP2.10 — La fuite cachée d'une validation croisée 🐛 · 1,5 point

a) **[0,4]** Deux étapes apprennent des données avant la boucle (0,1 chacune) :
- **le prix du quartier**, la moyenne de `y` par case, calculée sur **tous** les districts, ceux du fold de validation compris. La feature d'un district de validation contient son propre prix : avec des cases de 0,1°, une case contient en moyenne moins de 4 districts (2 000 pour 519 cases), souvent un seul, et la feature est alors la cible elle-même. C'est cette fuite qui fausse le score : 0,526 au lieu de 0,681 (0,1) ;
- **la standardisation** des neuf colonnes, faite sur les 2 000 districts. C'est une fuite par principe, mais une moyenne et un écart-type calculés sur 2 000 districts au lieu de 1 600 ne changent presque pas : la RMSE est la même à 4 décimales, 0,68108 au lieu de 0,68107 (0,1). Elle compterait davantage sur un petit jeu de données, ou pour un prétraitement plus sensible.

b) **[0,6]** Une solution :

```python
def cv_rmse_fixed(X, y, cells, k=5, alpha=1.0):
    scores = []
    for train, val in kfold(len(y), k):
        means = cell_means(cells[train], y[train])                 # the training districts only
        neighbourhood = lookup(cells, means, y[train].mean())       # for every district
        F = np.column_stack([X, neighbourhood])
        F = (F - F[train].mean(axis=0)) / F[train].std(axis=0)     # statistics of the training rows
        model = ridge_fit(F[train], y[train], alpha)
        scores.append(rmse(y[val], ridge_predict(model, F[val])))
    return float(np.mean(scores))
```

Barème : les moyennes par case calculées sur les seuls districts d'entraînement du tour (0,2) ; appliquées à tous les districts, avec la moyenne d'entraînement du tour par défaut (0,15) ; la standardisation ajustée sur les lignes d'entraînement (0,15) ; Ridge, RMSE de validation, moyenne des $k$ tours, un nombre renvoyé, sans modifier ses arguments (0,1).

c) **[0,2]** **0,681** (la fonction donne 0,6811).

d) **[0,3]** Sans fuite : **0,716** pour 0,5°, **0,681** pour 0,1°, **0,728** pour 0,05° : on garde **0,1°** (0,15). Le collègue obtenait 0,703, 0,526 et 0,400. Plus les cases sont petites, moins elles contiennent de districts, et plus la moyenne de la case se rapproche du prix du district lui-même : la fuite grandit, et sa RMSE baisse vers 0. Sans fuite, des cases trop petites contiennent trop peu de districts d'entraînement : leur moyenne devient bruitée, ou manque (le défaut), et la RMSE remonte. La fuite ne rend pas seulement le score optimiste : elle **change la décision**, et fait choisir la pire grille (0,15).

**Pour aller plus loin** : même sans fuite, la feature d'un district d'entraînement contient encore son propre prix, et le modèle apprend à lui faire trop confiance. Une version plus propre la calcule sans le district lui-même (une validation croisée interne) : c'est l'encodage par la cible (*target encoding*), que `TargetEncoder` de scikit-learn fait de cette façon (ch. 12).

**Erreurs fréquentes** : ne corriger que la standardisation ; calculer les moyennes des districts de validation avec leur propre fold ; prendre la moyenne de tous les prix comme défaut ; oublier le prix du quartier.
**Remédiation** : 8.24, 8.25, 8.3, 8.Q6 ; fiche du ch. 8, §8.5.1 (encadré ⚠️ « Pas de fuite, puisqu'on crée un modèle neuf à chaque tour »).

## CP2.11 — ε-greedy et moyenne incrémentale 🔨 · 1 point

a) **[0,5]**

```python
def epsilon_greedy_action(q_values, epsilon, rng):
    q_values = np.asarray(q_values, dtype=float)
    if rng.random() < epsilon:                          # explore: any arm, the best one included
        return int(rng.integers(len(q_values)))
    best = np.flatnonzero(q_values == q_values.max())   # all the arms of maximal estimate
    return int(rng.choice(best))
```

Barème : le tirage `rng.random() < epsilon` (0,1) ; l'exploration parmi **tous** les bras (0,15) ; l'exploitation avec les ex aequo tirés au hasard (0,15) ; un `int` renvoyé, `q_values` non modifié, le seul `rng` reçu (0,1). Avec des estimations justes, le meilleur bras est joué avec la probabilité $1 - \varepsilon + \varepsilon/K$.

b) **[0,3]**

```python
def incremental_estimates(rewards, step=None):
    q, estimates = 0.0, []
    for n, reward in enumerate(rewards, start=1):
        a = 1 / n if step is None else step
        q = q + a * (reward - q)
        estimates.append(q)
    return estimates
```

Barème : le départ à 0 (0,05) ; le pas $1/n$ ou constant (0,1) ; la mise à jour $Q \leftarrow Q + a\,(R - Q)$ (0,1) ; une estimation par récompense (0,05).

c) **[0,2]** Estimations successives : 0,25 ; 0,1875 ; 0,1406 ; 0,3555 ; 0,5166 ; 0,6375 ; 0,4781 ; **0,6086**. La moyenne exacte vaut 0,625 : avec un pas constant, la dernière récompense pèse 0,25, la première seulement $0{,}25 \times 0{,}75^7 \approx 0{,}033$ (une moyenne à oubli exponentiel).

**Erreurs fréquentes** : `np.argmax`, qui prend toujours le premier des ex aequo ; une exploration qui exclut le meilleur bras ; `rng.random() > epsilon` ; le générateur global `np.random` au lieu de `rng` ; partir de la première récompense au lieu de 0 ; ne renvoyer que la dernière estimation.
**Remédiation** : 11.2, 11.8, 11.20, 11.24.

## CP2.12 — Entretien 💼 · 1 point

**Réponse modèle en 60 secondes** : « Je ne le sais jamais à partir des données d'entraînement : je compare l'erreur d'entraînement à celle de données que le modèle n'a pas vues. Un grand écart signale l'overfitting ; deux erreurs hautes et proches, l'underfitting, que je juge face à une référence comme un modèle simple. Pour ne pas dépendre d'un seul découpage, je fais une validation croisée et je regarde la dispersion des folds ; les courbes de validation et d'apprentissage me montrent si plus de capacité ou plus de données aiderait. Tous les réglages, la régularisation, le nombre d'epochs avec l'early stopping, le choix des features, se font sur la validation ; le jeu de test reste sous clé et ne sert qu'une fois, à la fin. Je vérifie aussi qu'il n'y a pas de fuite : prétraitement ajusté dans chaque fold, pas de doublons, découpage par groupes ou dans le temps si les données sont dépendantes. Enfin, en production, je surveille les performances, parce que les données dérivent. »

**Critères** (0,25 chacun) : 1) comparer l'entraînement à des données jamais vues, avec l'écart **et** le niveau ; 2) la validation croisée et sa dispersion, ou les courbes de validation et d'apprentissage ; 3) les réglages faits sur la validation, le test gardé pour une seule mesure finale ; 4) la vérification des fuites, ou le suivi en production.
**Erreurs fréquentes** : « mon modèle a 99 % sur l'entraînement » ; une réponse limitée à « je régularise », sans dire comment on mesure ; régler sur le test.
**Relances possibles** : « Votre validation croisée donne 0,92 ± 0,01 et le test 0,85 : que se passe-t-il ? » (une fuite dans la validation, un test d'une autre distribution, l'optimisme de la sélection, ou le hasard : comparer à l'erreur type) · « Et avec 200 exemples seulement ? » (validation croisée, imbriquée pour annoncer un score, modèles simples et régularisés).
**Remédiation** : 9.E4, 8.E1, 8.E3, 8.E4, 9.Q1.

## CP2.13 — Parties antérieures : matrice de confusion, Bayes et entropie ✏️ · 2 points

a) **[0,3]** TP = 120 ; FP = 160 − 120 = 40 ; FN = 150 − 120 = 30 ; TN = 1 000 − 120 − 40 − 30 = 810 : **[120, 40, 30, 810]**.
b) **[0,4]** precision $= \frac{120}{160} = 0{,}75$ ; recall $= \frac{120}{150} = 0{,}8$ ; $F_1 = \frac{2 \times 0{,}75 \times 0{,}8}{0{,}75 + 0{,}8} = \frac{1{,}2}{1{,}55} \approx 0{,}774$ : **[0,75 ; 0,8 ; 0,774]**.
c) **[0,1]** $\frac{120 + 810}{1\,000} =$ **0,93**.
d) **[0,5]** Taux de faux positifs : $\frac{40}{850} \approx 0{,}0471$ (0,1). $P(\text{signalé}) = 0{,}8 \times 0{,}4 + 0{,}0471 \times 0{,}6 = 0{,}32 + 0{,}0282 = 0{,}3482$ (0,2). $P(\text{spam} \mid \text{signalé}) = \frac{0{,}32}{0{,}3482} \approx$ **0,919** (0,2). Même filtre, même recall, même taux de faux positifs : la precision passe de 0,75 à 0,919 parce que les spams sont plus fréquents. La precision dépend de la prévalence.
e) **[0,35]** $p = (0{,}75 ; 0{,}25)$ : $H = 0{,}75 \times 0{,}415 + 0{,}25 \times 2 \approx 0{,}311 + 0{,}5 =$ **0,811** bit.
f) **[0,35]** $p = (0{,}5 ; 0{,}25 ; 0{,}25)$ : $H = 0{,}5 \times 1 + 0{,}25 \times 2 + 0{,}25 \times 2 =$ **1,5** bit (0,25). Le premier cluster est le plus pur : son entropie est la plus basse (0,1). Pour trois espèces, l'entropie ne dépasse jamais $\log_2 3 \approx 1{,}585$ bit.

**Erreurs fréquentes** : échanger FP et FN ; le F1 pris comme la moyenne arithmétique (0,775) ; la precision du test (0,75) gardée en d) ; la spécificité à la place du taux de faux positifs ; des entropies en nats.
**Remédiation** : 3.2, 3.5, 3.7, 3.21, 4.4, 6.3, 7.R1.

---

## Barème récapitulatif et remédiation

| Question | Points | Ch. | Si tu as moins de la moitié des points, refais… |
|---|---|---|---|
| CP2.1 | 2,5 | 7 à 11 | les quiz et exercices cités au CP2.1, puis les flashcards des chapitres concernés |
| CP2.2 | 1 | 7 | 7.1, 7.2, 7.Q7 |
| CP2.3 | 2 | 7 | 7.3, 7.25, 7.26 |
| CP2.4 | 1 | 7 | 7.4, 7.5, 7.21 |
| CP2.5 | 1,5 | 8 | 8.1, 8.2, 8.3, 8.4 |
| CP2.6 | 2 | 9 | 9.2, 9.3, 9.17 |
| CP2.7 | 1,5 | 9 | 9.9, 9.21, 9.Q1 |
| CP2.8 | 2 | 10 | 10.3, 10.4, 10.6 |
| CP2.9 | 1 | 11 | 11.3, 11.4, 11.Q8 |
| CP2.10 | 1,5 | 8, 9 | 8.24, 8.25, 9.28 |
| CP2.11 | 1 | 11 | 11.2, 11.8, 11.20 |
| CP2.12 | 1 | 8, 9 | 9.E4, 8.E1, 8.E4 |
| CP2.13 | 2 | 3, 4, 6 | 3.7, 4.4, 6.3, 7.R1 |
| **Total** | **20** | | |

**Lire ta note**
- **16 à 20** : la partie II est solide. Passe à la partie III ; garde les flashcards des chapitres 7 à 11 dans tes révisions.
- **12 à moins de 16** : c'est bien. Fais la remédiation des questions où tu as eu moins de la moitié des points, puis passe à la suite.
- **8 à moins de 12** : avant la partie III, fais la remédiation, relis la synthèse (`05_synthese.md`) et les fiches des chapitres concernés ; le ch. 9 (overfitting, régularisation) et le ch. 8 (protocole d'évaluation) servent dans tous les chapitres suivants.
- **Moins de 8** : reprends les exercices ★★ des chapitres où tu as perdu le plus de points, puis refais l'examen.

Dans tous les cas, **refais dans une semaine** les questions où tu as eu moins de la moitié des points, sans regarder ce corrigé : c'est l'effort de rappel qui fixe les notions. Reporte ta note et tes points faibles dans `mon_travail/suivi/journal.md`.
