# Checkpoint III · Corrigé détaillé et barème de l'examen blanc

> À lire **après** l'examen, et après la partie B du notebook (vérification automatique des réponses chiffrées). Pour chaque question : la réponse, la démarche, le barème, les erreurs fréquentes et les exercices à refaire si tu as eu moins de la moitié des points. Les solutions exécutées des questions de code sont dans `03_examen_solutions.ipynb`.

## Règles de notation

- **Barème par sous-question**, indiqué entre crochets. Une réponse juste sans démarche, quand la question en demande une, rapporte la moitié des points.
- **Méthode juste, erreur de calcul** : la moitié des points de la sous-question. **Erreur reportée** : si une réponse fausse est réutilisée correctement plus loin, la suite n'est pas pénalisée une seconde fois.
- **Arrondis** : une décimale de trop ou de moins coûte au plus 0,05 point par question ; un arrondi fait trop tôt, qui change la dernière décimale demandée, aussi.
- **Lecture de graphique** (CP3.8) : les réponses se choisissent dans des intervalles ou des lettres ; toute lecture qui tombe dans le bon intervalle rapporte tous les points.
- **Questions rédigées** (CP3.2 h, CP3.4 f, CP3.5 k, CP3.6 b et e, la démonstration de CP3.7 d et CP3.7 f, CP3.9, CP3.10, les justifications de CP3.12, CP3.13 a et f) : les critères sont listés ; une idée juste formulée autrement compte autant.

---

## CP3.1 — Prédire la sortie de six extraits scikit-learn 🔮 · 1 point

Réponses : **a) A · b) C · c) B · d) D · e) A · f) D**, sans justification demandée (0,15 pour a à d, 0,2 pour e et f).

a) **(A) `(5,)`**. À deux classes, `decision_function` rend **un** score par exemple, celui de `classes_[1]` : un tableau à une dimension. Il faut `predict_proba` pour avoir deux colonnes.
b) **(C) cinq 1**. Chaque ligne de `predict_proba` est une distribution sur les trois classes : ses probabilités somment à 1. Une forêt a bien un `predict_proba`, la moyenne des proportions de classes des feuilles de ses arbres.
c) **(B) une `ValueError`** (« Found unknown categories ['Anvers'] … »). Par défaut, `OneHotEncoder(handle_unknown="error")` refuse une catégorie absente de l'entraînement ; la ligne de zéros, c'est `handle_unknown="ignore"`.
d) **(D) quatre valeurs proches de 0 sans être nulles** (ici environ −0,15, −0,13, −0,16 et −0,06). `transform` centre chaque colonne sur la moyenne **d'entraînement** : sur d'autres lignes, la moyenne n'est pas exactement 0, et c'est normal. `transform` travaille ligne par ligne : le nombre de lignes peut changer.
e) **(A) `"knn__n_neighbors"`** : le nom de l'étape, deux tirets bas, le nom du paramètre.
f) **(D) un `UserWarning`, puis trois candidats** : « The total space of parameters 3 is smaller than n_iter=10. Running 3 iterations. For exhaustive searches, use GridSearchCV. » Quand toutes les entrées sont des listes, la recherche au hasard tire **sans remise** : elle ne peut pas proposer plus de candidats que la grille n'en contient.

**Erreurs fréquentes** : en a), croire qu'il y a un score par classe (une erreur du livre, signalée par la fiche du ch. 15) ; en c), confondre le réglage par défaut et `handle_unknown="ignore"` ; en d), attendre une moyenne exactement nulle ; en f), croire que la recherche tire avec remise.
**Remédiation** : a) 15.2, 15.Q5 ; b) 15.10 ; c) 12.26, 15.28 ; d) 12.13, 15.14 ; e) 15.22, 15.Q11 ; f) 15.19, 15.Q10.

## CP3.2 — Standardiser et encoder un mini-tableau ✏️ · 1,5 point

a) **[0,3]** $\mu = \frac{38 + 46 + 42 + 50}{4} = 44$ ; écarts $-6, 2, -2, 6$ ; variance $\frac{36 + 4 + 4 + 36}{4} = 20$ ; $\sigma = \sqrt{20} \approx 4{,}4721$ : **[44 ; 4,4721]**.
b) **[0,15]** $\frac{46 - 44}{4{,}4721} \approx$ **0,447**.
c) **[0,2]** $\frac{54 - 44}{4{,}4721} \approx$ **2,236**.
d) **[0,2]** min 38, max 50 : $\frac{54 - 38}{50 - 38} = \frac{16}{12} \approx$ **1,333**.
e) **[0,2]** Catégories apprises, dans l'ordre alphabétique : Biscoe, Dream, Torgersen. Le manchot 3 est de Dream : **[0, 1, 0]**.
f) **[0,15]** Cuverville n'a pas été vue : avec `handle_unknown="ignore"`, toutes ses colonnes valent 0 : **[0, 0, 0]**.
g) **[0,2]** $z = \frac{41 - 44}{4{,}4721} \approx -0{,}671$, puis le one-hot de f) : **[−0,671 ; 0 ; 0 ; 0]**.
h) **[0,1]** Le min-max utilise le minimum et le maximum **de l'entraînement** (38 et 50) ; un bec de 54 mm dépasse ce maximum, et sa valeur dépasse 1 (0,05). Ce n'est pas une erreur : appliquer la transformation apprise, même hors de $[0 ; 1]$, est exactement ce qu'il faut faire ; réapprendre le scaler sur les données nouvelles serait une fuite (0,05). On peut borner (`clip=True`) si le modèle l'exige, au prix de l'information.

**Erreurs fréquentes** : l'écart-type avec $n - 1$ (5,164, la convention de pandas, pas celle de `StandardScaler`) ; inclure $T_1$ dans la moyenne ; ranger les catégories dans l'ordre d'apparition ; ajouter une colonne pour l'île inconnue ; borner le min-max à 1.
**Remédiation** : 12.1, 12.2, 12.4, 12.13, 12.26, 15.28.

## CP3.3 — Gini et meilleur seuil ✏️ · 1,5 point

a) **[0,2]** 3 A et 3 B : $1 - (0{,}5^2 + 0{,}5^2) =$ **0,5**.
b) **[0,1]** Les milieux de cinq paires d'abscisses consécutives : 1,5 ; 2,5 ; 4 ; 5,5 ; 7 : **5** seuils.
c) **[0,3]** $s = 4$ : à gauche {A, A, B}, Gini $1 - (\frac{4}{9} + \frac{1}{9}) = \frac{4}{9}$ ; à droite {B, A, B}, aussi $\frac{4}{9}$. Gini pondéré $\frac{3}{6} \times \frac{4}{9} + \frac{3}{6} \times \frac{4}{9} \approx$ **0,444**.
d) **[0,3]** Les cinq Gini pondérés : 1,5 → 0,4 ; **2,5 → 0,25** ; 4 → 0,444 ; 5,5 → 0,5 ; 7 → 0,4. Le meilleur seuil est **2,5** : la feuille de gauche {A, A} est pure, celle de droite {B, B, A, B} presque.
e) **[0,3]** À droite, $1 - (0{,}75^2 + 0{,}25^2) = 0{,}375$ ; pondéré : $\frac{2}{6} \times 0 + \frac{4}{6} \times 0{,}375 = 0{,}25$ ; gain $0{,}5 - 0{,}25$ : **[0,25 ; 0,25]**.
f) **[0,3]** $p = (0{,}75 ; 0{,}25)$ : $H = -0{,}75 \log_2 0{,}75 - 0{,}25 \log_2 0{,}25 \approx 0{,}311 + 0{,}5 =$ **0,811** bit.

**Erreurs fréquentes** : six seuils (un par abscisse) ; une moyenne simple des deux Gini, qui oublie de peser chaque enfant par son nombre d'exemples (0,1875 au lieu de 0,25 en e) ; prendre le seuil du milieu de l'intervalle (5,5 : Gini pondéré 0,4, loin du meilleur) ; l'entropie en nats (0,562).
**Remédiation** : 13.2, 13.3, 13.20, 13.22.

## CP3.4 — kNN avant et après standardisation ✏️ · 1,5 point

a) **[0,3]** Sur les mesures brutes, la masse écrase le bec : distances à $Q$ d'environ 200 ($P_1$), 100 ($P_2$), 500 ($P_3$), 600 ($P_4$) et 650 ($P_5$), les becs n'ajoutant que quelques centièmes. Les trois plus proches : **[2, 1, 3]**.
b) **[0,2]** Chinstrap, Adélie, Adélie : **Adélie**.
c) **[0,3]** $z_{\text{masse}} = \frac{3\,800 - 4\,210}{220} \approx -1{,}864$ et $z_{\text{bec}} = \frac{46 - 45}{4} = 0{,}25$ : **[−1,864 ; 0,25]**.
d) **[0,3]** Les cinq manchots standardisés : $P_1\,(-0{,}955 ; -0{,}75)$, $P_2\,(-1{,}409 ; 0{,}75)$, $P_3\,(0{,}409 ; -1{,}5)$, $P_4\,(0{,}864 ; 0{,}25)$, $P_5\,(1{,}091 ; 1{,}25)$. Distances à $Q$ : 1,351 ; **0,676** ; 2,868 ; 2,727 ; 3,119. Les trois plus proches : **[2, 1, 4]**.
e) **[0,2]** Chinstrap, Adélie, Chinstrap : **Chinstrap**.
f) **[0,2]** Une distance euclidienne additionne des différences dans des unités différentes : une centaine de grammes pèse des dizaines de fois plus que quelques millimètres, et la masse décide seule (0,1). Après standardisation, chaque mesure compte en écarts-types, et le bec, qui sépare les deux espèces, retrouve son poids. On standardise $Q$ avec la moyenne et l'écart-type **de l'entraînement**, ceux qui ont servi à placer les manchots connus : un exemple seul n'a pas d'écart-type, et réapprendre sur les données nouvelles serait une fuite (0,1).

**Erreurs fréquentes** : standardiser $Q$ sans standardiser les manchots d'entraînement (ou l'inverse) ; un z-score de signe inversé, $(\mu - x)/\sigma$ ; lister les voisins du plus loin au plus proche ; s'arrêter au plus proche voisin.
**Remédiation** : 13.1, 12.2, 13.R1, 13.14, 13.15.

## CP3.5 — AdaBoost et gradient boosting ✏️ · 2 points

a) **[0,15]** $\varepsilon = \frac{3 \times 0{,}1}{1} =$ **0,3**.
b) **[0,2]** $\alpha = \ln\frac{0{,}7}{0{,}3} = \ln 2{,}333\ldots \approx$ **0,847** (avec $K = 2$, $\ln(K - 1) = 0$).
c) **[0,2]** Les trois mal classés sont multipliés par $e^{\alpha} = \frac{0{,}7}{0{,}3}$ : $0{,}1 \times 2{,}333 \approx 0{,}2333$ ; les sept autres gardent 0,1 ; total $3 \times 0{,}2333 + 0{,}7 = 1{,}4$. Après normalisation, un mal classé pèse $\frac{0{,}2333}{1{,}4} \approx$ **0,1667**.
d) **[0,2]** $\frac{0{,}1}{1{,}4} \approx$ **0,0714**. Les trois mal classés pèsent ensemble $3 \times \frac{1}{6} = \frac{1}{2}$ : c'est toujours le cas à deux classes avec `learning_rate=1` (14.6), et la souche du tour 1 aurait une erreur pondérée de 0,5 au tour suivant.
e) **[0,25]** SAMME ajoute $\ln(K - 1) = \ln 3 \approx 1{,}0986$ : $\alpha \approx 0{,}847 + 1{,}099 =$ **1,946**. Avec quatre classes, deviner au hasard se trompe 75 % du temps : une erreur de 30 % est bien meilleure que le hasard.
f) **[0,1]** $F_0 = \frac{2 + 4 + 9 + 11}{4} =$ **6,5** (la constante qui minimise la MSE).
g) **[0,15]** **[−4,5 ; −2,5 ; 2,5 ; 4,5]**.
h) **[0,25]** Trois seuils : 1,5 (somme des carrés 0 + 26 = 26), **2,5** (2 + 2 = 4) et 3,5 (26). La souche coupe en 2,5 ; ses feuilles valent les moyennes des résidus, $-3{,}5$ et $3{,}5$ : **[2,5 ; −3,5 ; 3,5]**.
i) **[0,2]** $F_1(1) = 6{,}5 + 0{,}1 \times (-3{,}5) =$ **6,15**.
j) **[0,2]** $F_1 = (6{,}15 ; 6{,}15 ; 6{,}85 ; 6{,}85)$, nouveaux résidus $(-4{,}15 ; -2{,}15 ; 2{,}15 ; 4{,}15)$ : MSE $= \frac{2 \times (17{,}2225 + 4{,}6225)}{4} =$ **10,9225** (au lieu de 13,25 avec $F_0$).
k) **[0,1]** Avec $\eta = 0{,}1$, chaque arbre ne corrige qu'un dixième de ce qu'il a appris : il en faut bien plus pour descendre aussi bas (0,05). On l'accepte parce que de petits pas, plus nombreux, donnent en général une meilleure erreur de validation et une courbe plus stable : chaque arbre, ajusté sur un jeu fini, est bruité, et un petit pas moyenne ce bruit (une forme de régularisation, avec l'early stopping pour s'arrêter au bon nombre d'arbres) (0,05).

**Erreurs fréquentes** : en b), le facteur ½ des textes à labels $\pm 1$ (0,424), qui ne change pas l'algorithme mais n'est pas la convention de la fiche ; oublier la normalisation (0,2333 et 0,1) ; $\ln K$ au lieu de $\ln(K - 1)$ ; en h), des feuilles qui valent la moyenne des $y$ et non des résidus ; en i), oublier $\eta$ (3,0) ou partir de 0 au lieu de $F_0$.
**Remédiation** : 14.6, 14.7, 14.4, 14.24, 14.25.

## CP3.6 — Trouver trois fuites dans un script d'entraînement 🐛 · 2 points

a) **[0,6]** **A, C et E** (0,2 chacune ; une lettre de trop retire 0,1) :
- **(A)** ajoute la « valeur par pièce », $y / \text{AveRooms}$ : une feature **calculée à partir de la cible**. Multipliée par `AveRooms`, qui est aussi dans `X`, elle redonne $y$ exactement : le kNN lit la réponse. Dans l'usage réel, on ne connaît pas $y$ au moment de prédire : la feature n'existerait pas ;
- **(C)** standardise les neuf colonnes sur **tous** les districts, test compris : la moyenne et l'écart-type viennent en partie du test ;
- **(E)** choisit $k$ d'après son score **sur le test**, puis annonce ce même score : le test a servi à choisir, il ne mesure plus honnêtement le modèle choisi (le maximum de six scores bruités est optimiste).

(B) se calcule district par district, sans rien apprendre des autres : ce n'est pas une fuite. (D) découpe ; (F) rend ce que (E) a calculé.

b) **[0,4]** C'est **(A)** qui fausse le score : le $R^2$ passe de 0,74 (sans fuite) à 0,93 avec elle (0,2). La standardisation (C) apprend deux nombres par colonne sur 4 000 districts au lieu de 3 000 : ils changent à peine, et le $R^2$ aussi (0,739 au lieu de 0,737). Le choix de $k$ sur le test (E) ne change rien **ici**, parce que le $k$ qu'une validation croisée choisit (10) est aussi le meilleur sur le test (0,1). Il faut quand même les corriger : (C) compte davantage sur un petit dataset ou avec des points aberrants, et (E) rend le score optimiste dès que les candidats sont nombreux ou proches ; surtout, un protocole qui fuit ne dit pas **quand** il ment (0,1).

c) **[0,6]** Une solution :

```python
def honest_report(X, y):
    X = X.copy()                                          # do not modify the caller's array
    X[:, 4] = np.log1p(X[:, 4])                           # (B) kept: row by row, nothing learnt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    search = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                          {"kneighborsregressor__n_neighbors": K_GRID}, cv=5).fit(X_train, y_train)
    return search.best_params_["kneighborsregressor__n_neighbors"], search.score(X_test, y_test)
```

Barème : la feature (A) retirée et (B) gardée (0,1) ; le découpage d'abord, sur les huit colonnes, avec le même `train_test_split` (0,1) ; le scaler **dans** le pipeline, appris dans chaque fold (0,15) ; $k$ choisi par la `GridSearchCV` sur l'entraînement, avec le nom `étape__paramètre` (0,15) ; le test mesuré une seule fois, `X` non modifié, un couple renvoyé (0,1).

d) **[0,2]** **[10 ; 0,737]** (la fonction donne 0,7367 ; le score moyen de validation croisée du meilleur candidat, `best_score_`, vaut 0,688, et ce n'est pas ce qui est demandé).

e) **[0,2]** Le 0,93 venait de la fuite : un $R^2$ honnête vaut environ 0,74 (0,1). On annonce ce chiffre, mesuré une fois sur le test, avec le protocole qui l'a produit, et l'on retire du modèle toute feature qu'on ne connaîtra pas au moment de prédire (0,1).

**Erreurs fréquentes** : désigner (B), qui n'apprend rien des autres districts ; ne voir que la standardisation ; garder la feature (A) dans `honest_report` ; standardiser avant le découpage, ou hors du pipeline sur tout l'entraînement ; renvoyer `best_score_` au lieu du score de test ; modifier `X` en place (la ligne (B) écrit dans le tableau reçu).
**Remédiation** : 12.9, 12.31, 15.24, 13.21 (une colonne qui trahit la réponse), 8.24 ; fiche du ch. 12, §12.10, et du ch. 15, section sur les pipelines.

## CP3.7 — Le seuil de décision de coût minimal 🔨 · 2 points

a) et b) **[0,8]** Une solution :

```python
def total_cost(y_true, y_pred, cost_fn, cost_fp):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return int(cost_fn * false_negatives + cost_fp * false_positives)


def min_cost_threshold(proba, y_true, cost_fn, cost_fp, thresholds):
    proba = np.asarray(proba)
    costs = [total_cost(y_true, (proba >= t).astype(int), cost_fn, cost_fp) for t in thresholds]
    best = int(np.argmin(costs))                          # argmin keeps the first minimum
    return float(thresholds[best]), costs[best]
```

Barème : a) les faux négatifs et les faux positifs comptés (0,15), pondérés par leurs coûts, un `int` renvoyé (0,15) ; b) la règle `proba >= t` (0,1), le coût de chaque seuil (0,15), le premier minimum (0,1), un couple renvoyé, les arguments non modifiés (0,15).

c) **[0,3]** **[0,23 ; 173]**.
d) **[0,3]** Pour un exemple de probabilité $p$ (calibrée), décider « 1 » coûte en moyenne $(1 - p)\,c_{FP}$ (une fausse alerte s'il est négatif), décider « 0 » coûte $p\,c_{FN}$ (un oubli s'il est positif) (0,15). On décide « 1 » dès que $p\,c_{FN} \ge (1 - p)\,c_{FP}$, soit $p \ge \frac{c_{FP}}{c_{FP} + c_{FN}} = \frac{1}{6} \approx$ **0,1667** (0,15).
e) **[0,3]** **[163 ; 231]** : au seuil de c), le test coûte 30 % de moins qu'au seuil 0,5.
f) **[0,3]** Le seuil de c) est mesuré sur 600 exemples : la courbe du coût est plate et bruitée près de son minimum (176 à 0,22, 178 à 0,17, 181 à 0,16), et le minimum tombe là où le hasard de ces 600 exemples l'a mis (0,15). Le seuil théorique ne dépend que des coûts, mais suppose des probabilités **calibrées**. Ici, les deux se valent sur le test (165 au seuil 0,17, 163 au seuil 0,23) : on déploie le seuil théorique si la calibration a été vérifiée (une courbe de fiabilité), sinon un seuil choisi par validation croisée, moins bruité qu'un seul découpage (`TunedThresholdClassifierCV`) ; dans tous les cas, loin de 0,5 (0,15).

**Erreurs fréquentes** : `proba > t` au lieu de `>=` (0,22 et 176) ; les deux coûts échangés (0,75) ; le dernier minimum au lieu du premier ; un coût de type `np.int64` ; prendre le rapport des coûts, $\frac{1}{5}$, pour le seuil ; mesurer e) sur la validation.
**Remédiation** : 15.25, 15.26, 3.21 ; fiche du ch. 15, section sur le seuil de décision.

## CP3.8 — Lire une courbe de validation et une courbe de boosting 📈 · 1 point

a) **[0,1]** **2** : la courbe ② descend sans cesse jusqu'à presque 0 à `max_depth=20` (un arbre profond apprend l'entraînement par cœur) ; la courbe ① est celle de la validation.
b) **[0,2]** **(B)**, de 5 à 9 : la validation est la plus basse vers 7 et 8 (0,685 et 0,684), puis remonte.
c) **[0,15]** **(A) underfitting** : à `max_depth=2`, les deux erreurs sont hautes (environ 0,85) et proches.
d) **[0,15]** **(B) overfitting** : à `max_depth=18`, l'entraînement est presque parfait (0,06) et la validation est remontée (0,74) : un écart énorme.
e) **[0,15]** **1** : sa validation descend à environ 0,58 vers 50 arbres, puis remonte doucement jusqu'à 0,64 ; celle du learning rate 0,1 baisse encore à 400 arbres.
f) **[0,15]** **(B)**, de 21 à 100 (le minimum est vers 56 arbres).
g) **[0,1]** **Vrai** : la validation du learning rate 0,1 baisse encore à 400 arbres (0,493), et l'entraînement (0,35) reste à environ 0,14 de la validation, un écart modéré. L'early stopping trouverait le bon nombre d'arbres.

**Erreurs fréquentes** : choisir la profondeur où les deux courbes se croisent ou sont les plus proches ; lire l'overfitting sur le niveau de la courbe d'entraînement seule ; prendre le learning rate 1, qui descend plus vite, pour le meilleur.
**Remédiation** : 13.19, 14.20, 14.21, 9.9.

## CP3.9 — L'astuce du noyau en 5 lignes 🗣️ · 1 point

**Réponse modèle** : « Imagine des points sur une ligne : les rouges au milieu, les bleus aux deux bouts. Aucun point de coupure ne les sépare. Mais si l'on ajoute à chaque point une deuxième coordonnée, son carré, les points se rangent sur une parabole : les rouges en bas, les bleus en haut, et une droite horizontale les sépare. Ajouter des dimensions rend séparable ce qui ne l'était pas. L'astuce : un SVM n'a besoin que de la **ressemblance** entre deux exemples (un produit scalaire), et le noyau calcule directement cette ressemblance dans le grand espace, à partir des coordonnées d'origine, sans jamais construire les nouvelles coordonnées. C'est pour cela qu'un noyau RBF peut travailler dans une infinité de dimensions. »

**Critères** (0,25 chacun) : 1) des données non séparables deviennent séparables quand on ajoute des dimensions (des features calculées), avec une image ou un exemple ; 2) le SVM ne se sert que de ressemblances, des produits scalaires entre paires d'exemples ; 3) le noyau calcule ces ressemblances directement, sans construire les coordonnées du grand espace : c'est ce qu'il évite de calculer ; 4) l'explication tient en cinq lignes, sans formule, compréhensible par un débutant.
**Erreurs fréquentes** : dire que le noyau « transforme les données » sans dire qu'il évite de le faire ; confondre le noyau avec une fonction d'activation ; réciter $K(\mathbf{x}, \mathbf{x}') = e^{-\gamma\lVert\mathbf{x} - \mathbf{x}'\rVert^2}$ sans l'expliquer.
**Remédiation** : 13.5, 13.17, 13.18, 13.10.

## CP3.10 — Entretien 💼 · 1 point

**Réponse modèle en 60 secondes** : « Pas forcément. `predict_proba` donne un score entre 0 et 1, qui n'est une probabilité que si le modèle est **calibré** : parmi tous les clients à qui il donne 0,9, environ 90 % doivent être positifs. Je le vérifie sur des données jamais vues avec une courbe de fiabilité, et le score de Brier ou la log loss. Beaucoup de modèles ne sont pas calibrés : un Naïve Bayes est trop sûr de lui, une forêt tasse ses scores vers le milieu, et un SVC n'a de probabilités que par une calibration interne (Platt). Si la courbe s'écarte de la diagonale, je recalibre sur une validation avec `CalibratedClassifierCV`. Et je choisis le seuil de décision d'après le coût des erreurs, pas 0,5. Pour la production : je sauvegarde **tout le pipeline**, préparation comprise, avec joblib, et les versions de Python et des bibliothèques à côté, pour recharger avec les mêmes ; je vérifie le schéma des données reçues (mêmes colonnes, catégories inconnues traitées, `handle_unknown="ignore"`) ; et je surveille la dérive des données et des performances, pour recalibrer ou réentraîner. »

**Critères** (0,25 chacun) : 1) 0,9 n'est une probabilité que si le modèle est calibré, et comment le vérifier (courbe de fiabilité, Brier, log loss, sur des données jamais vues) ; 2) un remède (`CalibratedClassifierCV`) ou le seuil choisi d'après les coûts ; 3) sauvegarder le pipeline entier avec les versions (et ne jamais charger un fichier d'une source inconnue) ; 4) une autre précaution de production : le schéma et les catégories inconnues, la surveillance de la dérive, ou le réentraînement.
**Erreurs fréquentes** : « oui, c'est la probabilité » ; ne sauvegarder que le modèle, sans sa préparation ; oublier les versions.
**Relances possibles** : « Une forêt aléatoire est-elle calibrée ? » (souvent trop prudente, scores tassés vers le milieu ; à vérifier) · « Votre courbe de fiabilité est sous la diagonale pour les hauts scores : que dites-vous ? » (le modèle est trop sûr de lui : 0,9 annoncé, moins de 90 % de positifs) · « Comment savez-vous qu'il faut réentraîner ? » (suivi des performances sur des données étiquetées récentes, tests de dérive des features).
**Remédiation** : 15.E4, 15.E5, 15.10, 15.30, 3.28 (calibration).

## CP3.11 — Parties I et II ✏️ · 2 points

a) **[0,2]** TP = 45 ; FP = 60 − 45 = 15 ; FN = 75 − 45 = 30 ; TN = 500 − 45 − 15 − 30 = 410 : **[45, 15, 30, 410]**.
b) **[0,25]** precision $= \frac{45}{60} =$ **0,750**.
c) **[0,25]** recall $= \frac{45}{75} =$ **0,600**.
d) **[0,3]** $F_1 = \frac{2 \times 0{,}75 \times 0{,}6}{0{,}75 + 0{,}6} = \frac{0{,}9}{1{,}35} \approx$ **0,667**.
e) **[0,2]** $\frac{45 + 410}{500} =$ **0,910**. Une accuracy flatteuse : un détecteur qui ne signale rien ferait déjà 0,85.
f) **[0,3]** Chaque fold de validation contient un cinquième des exemples, et la stratification y met un cinquième de chaque classe : **[48, 12]**.
g) **[0,2]** Les quatre autres folds : **[192, 48]**.
h) **[0,3]** 5 folds × 3 répétitions = **15** modèles ; `cross_val_score` ne réentraîne pas de modèle final.

**Erreurs fréquentes** : échanger FP et FN ; le F1 pris comme une moyenne arithmétique (0,675) ; compter un réentraînement final en h).
**Remédiation** : 3.2, 3.5, 3.7, 8.1, 8.Q10, 8.21, 8.E5.

## CP3.12 — Vrai ou faux justifiés 🧠 · 1,5 point

Chaque affirmation : **[0,1]** pour le verdict ; **[0,15]** pour une justification juste, la moitié si elle est incomplète.

**a) Faux** (ch. 13). Un nœud d'arbre compare une feature à un seuil. Une standardisation $z = (x - \mu)/\sigma$, avec $\sigma > 0$, est croissante : elle garde l'ordre des valeurs, donc les mêmes partitions possibles des exemples ; les seuils se déplacent avec les valeurs, mais l'arbre coupe les mêmes groupes et rend les mêmes prédictions. Les arbres (et leurs ensembles) n'ont pas besoin de mise à l'échelle.

**b) Vrai** (ch. 12 et 15). `cross_val_score` clone le pipeline à chaque tour et appelle son `fit` sur les seuls folds d'entraînement : le scaler apprend sa moyenne et son écart-type sans voir le fold de validation. C'est la raison d'être d'un pipeline.

**c) Faux** (ch. 14). Chaque arbre d'un gradient boosting est ajusté aux résidus (l'opposé du gradient) de l'ensemble **précédent** : il dépend de tous les arbres d'avant, et ils s'entraînent l'un après l'autre. Une forêt tire des arbres indépendants, qu'on peut entraîner en parallèle (`n_jobs`). Les bibliothèques de boosting parallélisent seulement la construction de chaque arbre.

**d) Vrai** (ch. 13 et 15). Naïve Bayes multiplie les vraisemblances des features comme si elles étaient indépendantes : une feature en deux exemplaires compte deux fois la même preuve. Si une copie multiplie le rapport des vraisemblances des deux classes par 3, les deux copies le multiplient par 9 : avec des classes de même effectif (l'a priori ne tire dans aucun sens), le posterior s'éloigne de 0,5. C'est pourquoi Naïve Bayes est souvent trop sûr de lui quand des features sont corrélées (15.10).

**e) Faux** (ch. 14). Chaque bootstrap laisse de côté environ 36,8 % des exemples ; le score out-of-bag évalue chaque exemple avec les seuls arbres qui ne l'ont **pas** tiré, pour lesquels il est une donnée nouvelle. C'est ce qui en fait une estimation honnête, presque gratuite.

**f) Vrai** (ch. 13). La distance d'un kNN additionne toutes les features à parts égales : 38 features de bruit noient les 2 qui comptent, et les « plus proches voisins » deviennent presque quelconques (la malédiction de la dimension). Un arbre choisit à chaque nœud la feature qui fait gagner le plus d'impureté : avec assez d'exemples, il coupe surtout sur les deux features informatives et ignore les autres.

**Erreurs fréquentes** : en a), croire que tout modèle demande des features standardisées ; en c), confondre bagging et boosting ; en d), croire que des features en plus ne peuvent qu'aider ; en e), croire que le score OOB est un score d'entraînement.
**Remédiation** : a) fiche du ch. 13 (arbres), 13.Q8 b) ; b) 12.31, 15.Q11, 15.6 ; c) 14.8 ; d) 15.10, 13.27 ; e) 14.3, 14.17, 14.15 ; f) 13.15, 13.R1.

## CP3.13 — Variance d'une moyenne de modèles corrélés ∂ · 1,5 point

a) **[0,5]** $\operatorname{Var}\big(\frac{1}{M}\sum_m h_m\big) = \frac{1}{M^2}\operatorname{Var}\big(\sum_m h_m\big)$ (0,1). La double somme $\sum_m \sum_{m'} \operatorname{Cov}(h_m, h_{m'})$ contient $M$ termes avec $m = m'$, qui valent $\operatorname{Var}(h_m) = \sigma^2$, et $M(M - 1)$ termes avec $m \ne m'$, qui valent $\rho\,\sigma^2$ (0,2). Donc
$$\operatorname{Var}\Big(\frac{1}{M}\sum_m h_m\Big) = \frac{M\sigma^2 + M(M - 1)\rho\,\sigma^2}{M^2} = \frac{\sigma^2}{M} + \frac{(M - 1)\rho\,\sigma^2}{M} = \rho\,\sigma^2 + \frac{1 - \rho}{M}\,\sigma^2 \quad (0{,}2).$$

b) **[0,2]** $0{,}25 \times 4 + \frac{0{,}75 \times 4}{10} = 1 + 0{,}3 =$ **1,3**.
c) **[0,2]** Le second terme tend vers 0 : la limite vaut $\rho\,\sigma^2 =$ **1**.
d) **[0,2]** $1 + \frac{3}{M} \le 1{,}08 \iff M \ge 37{,}5$ : **38** arbres ($M = 37$ donne 1,081).
e) **[0,2]** Limite $0{,}1 \times 5 = 0{,}5$ ; pour $M = 30$ : $0{,}5 + \frac{0{,}9 \times 5}{30} = 0{,}5 + 0{,}15 = 0{,}65$ : **[0,5 ; 0,65]**.
f) **[0,2]** Avec 30 arbres, la forêt : 0,65 contre $1 + \frac{3}{30} = 1{,}1$ pour le bagging (0,1). Ajouter des arbres ne fait descendre la variance que jusqu'au plancher $\rho\,\sigma^2$ : pour aller plus bas, il faut des arbres **moins corrélés**, quitte à ce que chacun soit un peu plus variable. C'est ce que fait la forêt en tirant les features à chaque nœud (0,1).

**Erreurs fréquentes** : en a), oublier les termes croisés (on retrouve $\sigma^2/M$, la variance de modèles indépendants) ou en compter $M^2$ ; en b), $\sigma^2/M = 0{,}4$ ; en c), croire que la variance tend vers 0 ; en d), répondre 37 en arrondissant 37,5 au plus proche.
**Remédiation** : fiche du ch. 14, encadré 🧮 sur la variance d'une moyenne de modèles corrélés ; 14.13, 14.16, 14.5.

## CP3.14 — Combien coûte une recherche en grille ? 🧮 · 0,5 point

a) **[0,1]** $4 \times 3 \times 5 = 60$ candidats, $60 \times 5 = 300$ entraînements, plus le réentraînement final : **301**.
b) **[0,15]** $301 \times 2 = 602$ s, soit environ **10** minutes.
c) **[0,15]** Chaque tour extérieur refait les 301 entraînements : $5 \times 301 =$ **1 505** (environ 50 minutes, un peu moins en fait, car chaque entraînement voit 4/5 des données).
d) **[0,1]** $12 \times 5 + 1 =$ **61**.

**Erreurs fréquentes** : additionner les nombres de valeurs au lieu de les multiplier ; oublier le réentraînement final, ou le compter aussi une fois de plus dans la validation imbriquée ; laisser la durée en secondes.
**Remédiation** : 15.1, 15.5, 15.Q10.

---

## Barème récapitulatif et remédiation

| Question | Points | Ch. | Si tu as moins de la moitié des points, refais… |
|---|---|---|---|
| CP3.1 | 1 | 12 à 15 | 15.2, 15.10, 15.19, 12.13 |
| CP3.2 | 1,5 | 12 | 12.1, 12.2, 12.4, 12.13 |
| CP3.3 | 1,5 | 13 | 13.2, 13.3, 13.20 |
| CP3.4 | 1,5 | 12, 13 | 13.1, 12.2, 13.R1 |
| CP3.5 | 2 | 14 | 14.6, 14.7, 14.4 |
| CP3.6 | 2 | 12, 15 | 12.31, 15.24, 13.21 |
| CP3.7 | 2 | 15 | 15.25, 15.26 |
| CP3.8 | 1 | 13, 14 | 13.19, 14.20, 14.21 |
| CP3.9 | 1 | 13 | 13.5, 13.17, 13.10 |
| CP3.10 | 1 | 15 | 15.E4, 15.E5, 15.30 |
| CP3.11 | 2 | 3, 8 | 3.5, 3.7, 8.1, 8.21 |
| CP3.12 | 1,5 | 12 à 15 | les exercices cités au CP3.12, puis les flashcards des chapitres concernés |
| CP3.13 | 1,5 | 14 | 14.13, 14.16 |
| CP3.14 | 0,5 | 15 | 15.1, 15.5 |
| **Total** | **20** | | |

**Lire ta note**
- **16 à 20** : la partie III est solide. Passe à la partie IV ; garde les flashcards des chapitres 12 à 15 dans tes révisions.
- **12 à moins de 16** : c'est bien. Fais la remédiation des questions où tu as eu moins de la moitié des points, puis passe à la suite.
- **8 à moins de 12** : avant la partie IV, fais la remédiation, relis la synthèse (`05_synthese.md`) et les fiches des chapitres concernés ; la préparation sans fuite (ch. 12 et 15) servira dans tous les projets.
- **Moins de 8** : reprends les exercices ★★ des chapitres où tu as perdu le plus de points, puis refais l'examen.

Dans tous les cas, **refais dans une semaine** les questions où tu as eu moins de la moitié des points, sans regarder ce corrigé : c'est l'effort de rappel qui fixe les notions. Reporte ta note et tes points faibles dans la section CP3 de `mon_travail/suivi/auto_evaluation.md`, et ajoute une ligne dans `journal.md`.
