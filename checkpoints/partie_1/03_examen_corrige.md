# Checkpoint I · Corrigé détaillé et barème de l'examen blanc

> À lire **après** l'examen, et après la partie B du notebook (vérification automatique des réponses chiffrées). Pour chaque question : la réponse, la démarche, le barème, les erreurs fréquentes et les exercices à refaire si tu as eu moins de la moitié des points. Les solutions exécutées des questions de code sont dans `03_examen_solutions.ipynb`.

## Règles de notation

- **Barème par sous-question**, indiqué entre crochets. Une réponse juste sans démarche, quand la question en demande une, rapporte la moitié des points.
- **Méthode juste, erreur de calcul** : la moitié des points de la sous-question. **Erreur reportée** : si une réponse fausse est réutilisée correctement plus loin, la suite n'est pas pénalisée une seconde fois.
- **Arrondis** : une décimale de trop ou de moins coûte au plus 0,05 point par question ; un arrondi fait trop tôt, qui change la dernière décimale demandée, aussi.
- **Lectures de graphique** (CP1.7) : les tolérances sont indiquées ; toute lecture dans l'intervalle rapporte tous les points.
- **Questions rédigées** (CP1.1, CP1.12 à CP1.14) : les critères sont listés ; une idée juste formulée autrement compte autant.

---

## CP1.1 — Questions flash sur toute la partie (vrai ou faux, justifié) 🧠 · 1,5 point

Chaque affirmation : **[0,1]** pour le verdict, **[0,15]** pour une justification juste (0,05 à 0,1 si elle est incomplète).

**a) Vrai** (ch. 1). À l'entraînement, le modèle compare ses prédictions aux labels pour corriger ses paramètres : c'est ce qui rend l'apprentissage « supervisé ». Pour prédire, on ne lui donne que les features d'un nouvel exemple : le label est justement ce qu'il doit trouver. Les labels du jeu de test servent à évaluer le modèle, pas à prédire.

**b) Faux** (ch. 2). Une corrélation nulle dit seulement qu'il n'y a pas de tendance **linéaire**. Contre-exemple : $x$ réparti symétriquement autour de 0 (par exemple uniforme sur $[-1 ; 1]$) et $y = x^2$. $y$ dépend entièrement de $x$, et pourtant $r = 0$, car la parabole est symétrique. L'indépendance entraîne une corrélation nulle, pas l'inverse.

**c) Vrai** (ch. 3). L'accuracy ne regarde que les décisions, de part et d'autre du seuil. Le score de Brier, $\frac{1}{n} \sum_i (\hat{p}_i - y_i)^2$, regarde les probabilités elles-mêmes. Rendre une probabilité plus juste sans la faire changer de côté du seuil baisse le Brier sans changer aucune décision : pour un cas positif annoncé à 0,6, passer à 0,9 fait tomber son terme de $(1 - 0{,}6)^2 = 0{,}16$ à $0{,}01$. C'est ce que fait une meilleure calibration (3.28).

**d) Vrai** (ch. 4). Le posterior est proportionnel au produit vraisemblance × prior. Avec un prior constant, le plus grand posterior est donc celui de la plus grande vraisemblance : le MAP coïncide avec le maximum de vraisemblance.

**e) Faux** (ch. 5). En mathématiques, l'erreur de la dérivée centrée diminue avec $h$ (comme $h^2$). Mais un ordinateur ne garde qu'environ 16 chiffres significatifs : $f(x + h) - f(x - h)$ soustrait deux nombres presque égaux, et l'erreur d'arrondi, divisée par $2h$, grandit quand $h$ diminue. Il existe donc un meilleur pas, de l'ordre de $10^{-5}$ pour une fonction ordinaire en `float64` ; en dessous, l'estimation se dégrade (5.12, 5.13).

**f) Faux** (ch. 6). La perplexité se calcule par token : elle dépend du découpage. Un tokenizer qui coupe en petits morceaux produit plus de tokens, chacun plus facile à deviner. On ne compare deux perplexités qu'avec le même tokenizer et les mêmes données de test ; sinon, on ramène la loss à une unité commune, comme les bits par caractère.

**Erreurs fréquentes** : en b), « corrélation nulle, donc aucun lien » ; en c), croire qu'une mesure ne peut s'améliorer que si des décisions changent ; en e), oublier les arrondis de l'ordinateur.
**Remédiation** : a) 1.Q3, 1.Q6 ; b) 2.Q12, 2.12 ; c) 3.28, 3.E5 ; d) 4.6 f ; e) 5.12, 5.13 ; f) 6.Q12, 6.E3.

## CP1.2 — Norme, produit scalaire et log₂ ✏️ · 1 point

a) $\|\mathbf{u}\| = \sqrt{1^2 + (-2)^2 + 2^2} = \sqrt{9} =$ **3**. **[0,2]**
b) $\mathbf{u} \cdot \mathbf{v} = 1 \times 2 + (-2) \times 3 + 2 \times 4 = 2 - 6 + 8 =$ **4**. **[0,2]**
c) $\mathbf{u} \cdot \mathbf{w} = 1 \times 2 + (-2) \times 1 + 2 \times 0 = 0$ : **oui**, ils sont orthogonaux, puisque leur produit scalaire est nul. **[0,2]** (0,1 pour le verdict, 0,1 pour le calcul)
d) $\frac{1}{32} = 2^{-5}$, donc $\log_2 \frac{1}{32} =$ **−5**. **[0,2]**
e) $24 = 8 \times 3 = 2^3 \times 3$, donc $\log_2 24 = 3 + \log_2 3 \approx 3 + 1{,}585 =$ **4,585**. **[0,2]** (0,1 si la valeur est juste mais tirée de la calculatrice sans le calcul demandé)

**Erreurs fréquentes** : 9 en a) (la racine oubliée) ; 16 en b) (le signe de −2 oublié) ; 4,755 en e), soit $3 \times 1{,}585$ : le log d'un produit est la **somme** des logs.
**Remédiation** : 0B.8 (norme), 0B.18 (produit scalaire), 0B.15 et 0B.16 (logarithmes).

## CP1.3 — La moyenne qui oublie `axis` 🐛 · 1 point

a) **[0,3]** `X.mean()` sans `axis` fait la moyenne de **toutes** les valeurs du tableau : 333 × 4 = 1 332 nombres, millimètres et grammes mélangés. Le résultat est **un seul nombre**, de forme `()` (≈ 1 117) ; de même, `X.std()` (≈ 1 830) est l'écart-type de tout le tableau. Le collègue retire donc la même valeur à toutes les colonnes et les divise toutes par le même écart-type, dominé par la masse en grammes. Résultat : les colonnes en millimètres deviennent presque constantes (écarts-types de 0,001 à 0,008), et aucune colonne n'a une moyenne nulle. (0,1 : une seule moyenne pour tout le tableau ; 0,1 : la forme `()`, ou « un seul nombre », « un scalaire » ; 0,1 : la conséquence.)

b) **[0,4]**

```python
def standardize(X):
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)
```

`X.mean(axis=0)` a la forme `(4,)` : une moyenne par colonne. Le *broadcasting* la retire de chaque ligne de `X`, de forme `(333, 4)`. `np.std` utilise ddof = 0 par défaut. (0,3 pour `axis=0` dans la moyenne **et** dans l'écart-type ; 0,1 pour ddof = 0 et `X` non modifié.)

c) **[0,3]** La masse du premier manchot vaut 3 750 g ; la colonne a pour moyenne 4 207,1 g et pour écart-type 804,0 g (ddof = 0). $z = \frac{3\,750 - 4\,207{,}1}{804{,}0} \approx$ **−0,57**.

**Erreurs fréquentes** : `axis=1`, qui calcule la moyenne de chaque manchot (une ligne mélange encore les unités) ; `X -= X.mean(axis=0)`, qui modifie `X` ; le `.std()` de pandas, avec ddof = 1 par défaut (−0,5676 au lieu de −0,5685 : même valeur à 2 décimales, mais la vérification automatique, à 4 décimales, la refuse).
**Remédiation** : 0A.51, 0A.56 ; 2.15 (`zscore` avec `axis`) ; 2.4 (ddof).

## CP1.4 — Statistiques de cinq points ✏️ · 1,5 point

a) $\bar{x} = \frac{2 + 4 + 6 + 8 + 10}{5} =$ **6** ; $\bar{y} = \frac{3 + 7 + 5 + 11 + 9}{5} =$ **7**. **[0,2]**
b) Écarts de $x$ : −4, −2, 0, 2, 4 ; carrés : 16, 4, 0, 4, 16, de somme 40. Variance $\frac{40}{5} =$ **8** ; écart-type $\sqrt{8} \approx$ **2,83**. **[0,3]** (0,2 pour la variance, 0,1 pour l'écart-type)
c) $z = \frac{10 - 6}{2{,}828} \approx$ **1,41** (c'est $\sqrt{2}$). **[0,2]**
d) Écarts de $y$ : −4, 0, −2, 4, 2. Produits des écarts : 16, 0, 0, 8, 8, de somme 32. Covariance $\frac{32}{5} =$ **6,4** avec ddof = 0, $\frac{32}{4} =$ **8** avec ddof = 1. **[0,3]** (0,2 + 0,1)
e) Variance de $y$ : $\frac{16 + 0 + 4 + 16 + 4}{5} = 8$. $r = \frac{6{,}4}{\sqrt{8} \times \sqrt{8}} = \frac{6{,}4}{8} =$ **0,80** (ou directement $\frac{32}{\sqrt{40 \times 40}}$ : ddof s'annule). **[0,3]**
f) Avec 40 : la somme des $y$ passe à 75, la moyenne à **12,5**. Valeurs triées : 3, 5, 7, 9, 11, 40 ; la médiane est la moyenne des deux du milieu, $\frac{7 + 9}{2} =$ **8**. La **médiane** résiste (de 7 à 8) ; la moyenne est tirée vers le point extrême (de 7 à 12,5). **[0,2]**

**Erreurs fréquentes** : diviser par $n - 1$ (10 et 3,16 en b) ; prendre $r^2 = 0{,}64$ pour $r$ ; la troisième valeur triée (7) pour médiane de six valeurs.
**Remédiation** : 2.4, 2.7, 2.Q1, 2.13, 2.15.

## CP1.5 — Intervalle de confiance bootstrap 🔨 · 1,5 point

a) **[0,8]** Une solution (d'autres sont justes, en particulier les tirages faits d'un coup avec `size=(n_boot, n)` ou `rng.choice(x, size=n)`, qui donnent exactement les mêmes nombres) :

```python
def bootstrap_ci_mean(x, confidence=0.95, n_boot=1000, seed=0):
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)          # n indices drawn with replacement
        means[b] = x[idx].mean()
    low, high = np.percentile(means, [50 * (1 - confidence), 50 * (1 + confidence)])
    return float(low), float(high)
```

Barème : le générateur créé dans la fonction avec la graine (0,15) ; $n$ indices tirés **avec remise** (0,25) ; la moyenne de chaque rééchantillon (0,1) ; les percentiles $50\,(1 - c)$ et $50\,(1 + c)$ (0,2) ; un tuple de deux nombres (0,1).

b) **[0,2]** $[5\,008{,}0 ;\ 5\,185{,}1]$ g, autour d'une moyenne d'échantillon de 5 092,4 g.

c) **[0,2]** $[5\,023{,}7 ;\ 5\,166{,}2]$ g : **plus étroit**. On garde 90 % des moyennes bootstrap au lieu de 95 %, en coupant 5 % de chaque côté au lieu de 2,5 % : moins de confiance, intervalle plus court.

d) **[0,3]** La phrase **(2)** est juste : c'est une propriété de la méthode, qui porte sur la **moyenne** (refaite sur beaucoup d'échantillons, elle donnerait un intervalle qui contient la vraie moyenne environ 95 fois sur 100). La phrase (1) est fausse : l'intervalle ne dit rien des individus. Les masses des Gentoo vont de 3 950 à 6 300 g, et 95 % d'entre elles tiennent à peu près entre 4 200 et 6 000 g, bien au-delà de cet intervalle de moins de 200 g.

**Erreurs fréquentes** : un générateur créé hors de la fonction (deux appels donnent deux intervalles) ; des rééchantillons d'une autre taille que $n$ ; les percentiles 5 et 95 pour un intervalle à 95 % ; arrondir les bornes dans la fonction (seule la réponse écrite s'arrondit) ; confondre l'intervalle de la moyenne et la dispersion des individus.
**Remédiation** : 2.21, 2.22, 2.24, 2.Q10.

## CP1.6 — Dépistage ✏️ · 2 points

a) **[0,5]** (0,125 par case) Malades : 4 % × 25 000 = 1 000 ; sains : 24 000.
**TP** = 0,90 × 1 000 = **900** ; **FN** = 1 000 − 900 = **100** ; **FP** = (1 − 0,92) × 24 000 = **1 920** ; **TN** = 24 000 − 1 920 = **22 080**.

b) **[0,3]** precision $= \frac{900}{900 + 1\,920} = \frac{900}{2\,820} \approx$ **0,319**.
c) **[0,2]** NPV $= \frac{22\,080}{22\,080 + 100} = \frac{22\,080}{22\,180} \approx$ **0,9955**.
d) **[0,1]** accuracy $= \frac{900 + 22\,080}{25\,000} = \frac{22\,980}{25\,000} \approx$ **0,919**.

e) **[0,3]** $P(+) = P(+ \mid M)\,P(M) + P(+ \mid S)\,P(S) = 0{,}90 \times 0{,}04 + 0{,}08 \times 0{,}96 = 0{,}036 + 0{,}0768 = 0{,}1128$ (0,15). $P(M \mid +) = \frac{0{,}036}{0{,}1128} \approx 0{,}319$ : on retrouve la precision (0,15). La règle de Bayes et la matrice de confusion font le même calcul, l'une avec des probabilités, l'autre avec des effectifs.

f) **[0,4]** Le posterior après le premier test devient le prior du second (0,2) :
$P(M \mid ++) = \frac{0{,}90 \times 0{,}319}{0{,}90 \times 0{,}319 + 0{,}08 \times 0{,}681} \approx$ **0,841** (0,2).
Avec les cotes, plus rapide : cote de départ $\frac{0{,}04}{0{,}96} = \frac{1}{24}$ ; rapport de vraisemblance d'un positif $\frac{0{,}90}{0{,}08} = 11{,}25$ ; après deux positifs, $\frac{11{,}25^2}{24} \approx 5{,}27$, soit une probabilité $\frac{5{,}27}{6{,}27} \approx 0{,}841$.

g) **[0,2]** La maladie est rare : il y a 24 fois plus de sains que de malades, si bien que 8 % de faux positifs chez les sains (1 920 personnes) dépassent de loin les 900 vrais positifs. La precision dépend de la prévalence (0,1). Sans changer le test (0,1) : le réserver à une population plus à risque (plus forte prévalence : symptômes, antécédents), ou confirmer chaque positif par un second test indépendant, comme en f). « Relever le seuil de décision » ne compte pas : c'est changer le test, puisque sa sensibilité et sa spécificité changent.

**Erreurs fréquentes** : compter les faux positifs parmi les 25 000 personnes (2 000) au lieu des 24 000 sains ; prendre la sensibilité (0,90) pour la precision, ou la spécificité (0,92) pour la NPV ; donner la moyenne de la sensibilité et de la spécificité (0,91, la *balanced accuracy*) pour l'accuracy ; en f), oublier de mettre à jour le prior.
**Remédiation** : 3.7, 3.21, 4.4, 4.8, 3.Q12.

## CP1.7 — Lire une courbe ROC et une courbe PR 📈 · 1,5 point

a) **[0,2]** La courbe ROC est faite de deux taux calculés **séparément** : le recall parmi les fraudes, le taux de faux positifs parmi les transactions normales. Chacun ne dépend que de la loi des scores de son groupe, pas de la proportion de fraudes. Mêmes lois de scores, donc même courbe ROC, aux fluctuations d'échantillonnage près.

b) **[0,2]** Taux de faux positifs ≈ **0,05** au recall 0,8 (accepte de 0,04 à 0,06 ; les valeurs exactes des deux échantillons sont 0,046 et 0,050).

c) **[0,4]** Fraudes : 1 % × 100 000 = 1 000, dont **800** détectées (0,1). Transactions normales : 99 000, dont 0,05 × 99 000 ≈ **4 950** fausses alertes (0,15 ; ta lecture de b) multipliée par 99 000 est juste ; multipliée par 100 000, elle perd 0,05 : les fausses alertes ne viennent que des 99 000 transactions normales). precision $\approx \frac{800}{800 + 4\,950} \approx$ **0,14** (0,15 ; accepte de 0,12 à 0,17).

d) **[0,2]** Au recall 0,8 : ≈ **0,7** avec 10 % de fraudes (valeur exacte 0,66 ; accepte 0,6 ou 0,7) et ≈ **0,1** avec 1 % (valeur exacte 0,14 ; accepte 0,1 ou 0,2). C'est cohérent avec c) : environ 0,14.

e) **[0,2]** **0,01** : un modèle qui répond au hasard a une precision égale à la proportion de positifs, quel que soit son seuil. La courbe PR est donc à comparer à cette ligne de base, pas à 0,5.

f) **[0,3]** La courbe **precision-recall** (ou au moins la precision à côté du recall). À 1 % de fraudes, la courbe ROC reste flatteuse (AUC ≈ 0,96), alors qu'au recall 0,8 environ six alertes sur sept sont fausses. C'est la precision qui dit la charge de l'équipe : environ 5 750 alertes par jour pour 800 fraudes. Idéalement, on traduit chaque seuil en nombres par jour (fraudes trouvées, fausses alertes), et on le choisit selon les coûts.

**Erreurs fréquentes** : croire que la courbe ROC change avec la prévalence ; confondre le taux de faux positifs avec la part de fausses alertes (1 − precision) ; placer la ligne de base PR à 0,5.
**Remédiation** : 3.24, 3.26, 3.27 ; fiche du ch. 3, « Au-delà du livre » (ROC et precision-recall).

## CP1.8 — Trois hypothèses, deux lancers ✏️ · 2 points

a) **[0,4]** Prior × vraisemblance : $0{,}25 \times 0{,}2 = 0{,}05$ ; $0{,}5 \times 0{,}5 = 0{,}25$ ; $0{,}25 \times 0{,}8 = 0{,}2$ (0,2). Évidence : 0,5. Posterior : **[0,1 ; 0,5 ; 0,4]** (0,2).

b) **[0,5]** Le posterior de a) devient le prior : $[0{,}1 \times 0{,}2 ;\ 0{,}5 \times 0{,}5 ;\ 0{,}4 \times 0{,}8] = [0{,}02 ;\ 0{,}25 ;\ 0{,}32]$, d'évidence 0,59. Posterior : $\left[\frac{2}{59} ;\ \frac{25}{59} ;\ \frac{32}{59}\right] \approx$ **[0,034 ; 0,424 ; 0,542]**. On peut aussi tout faire d'un coup : prior × $\theta^2$ = [0,01 ; 0,125 ; 0,16], de somme 0,295.

c) **[0,3]** $P(\text{face, face}) = 0{,}5 \times 0{,}59 = \sum \text{prior} \times \theta^2 =$ **0,295**.

d) **[0,2]** **θ = 0,8** (posterior 0,542), alors que le prior favorisait 0,5 : deux faces ont suffi à changer d'avis.

e) **[0,4]** Avec le posterior de b) comme prior, l'évidence d'une face est la moyenne des $\theta$ pondérée par le posterior (0,2) : $\frac{2}{59} \times 0{,}2 + \frac{25}{59} \times 0{,}5 + \frac{32}{59} \times 0{,}8 = \frac{38{,}5}{59} \approx$ **0,653** (0,2). Ce n'est ni le MAP (0,8), ni la probabilité de face d'avant les lancers (0,5). Avec le posterior arrondi [0,034 ; 0,424 ; 0,542], on trouve 0,6524, soit 0,652 : l'arrondi fait trop tôt change la dernière décimale (−0,05, voir les règles).

f) **[0,2]** **0** : le posterior est proportionnel à prior × vraisemblance, donc un prior nul reste nul quelles que soient les données. Si la pièce avait vraiment un biais de 0,8, on ne pourrait jamais le découvrir : on ne donne jamais une probabilité nulle à une hypothèse qu'on n'a pas exclue.

**Erreurs fréquentes** : oublier de normaliser (le posterior doit sommer à 1) ; oublier le prior ; lire le MAP sur le prior ; répondre 0,8 (le MAP) en e).
**Remédiation** : 4.5, 4.6, 4.7, 4.16.

## CP1.9 — Gradient, point selle et descente ∂ · 2 points

a) **[0,3]** $\frac{\partial f}{\partial x} = 2(x - 1)$ et $\frac{\partial f}{\partial y} = -(y + 2)$, donc $\nabla f(x, y) = \big(2(x - 1),\ -(y + 2)\big)$.

b) **[0,2]** $2(x - 1) = 0$ et $-(y + 2) = 0$ : le point critique est **(1 ; −2)**.

c) **[0,4]** $f(1 + t, -2) = t^2$, de dérivée seconde **2** > 0 : un creux le long de l'axe des $x$. $f(1, -2 + t) = -\frac{t^2}{2}$, de dérivée seconde **−1** < 0 : une bosse le long de l'axe des $y$ (0,2). Le point critique n'est ni un minimum ni un maximum : c'est un **point selle** (0,2).

d) **[0,5]** (0,25 par pas) $\nabla f(2, -1) = (2, -1)$, donc $(2 - 0{,}25 \times 2 ;\ -1 - 0{,}25 \times (-1)) =$ **(1,5 ; −0,75)**.
$\nabla f(1{,}5 ; -0{,}75) = (1 ; -1{,}25)$, donc $(1{,}5 - 0{,}25 ;\ -0{,}75 + 0{,}3125) =$ **(1,25 ; −0,4375)**.

e) **[0,3]** $f(2, -1) = 1 - 0{,}5 =$ **0,5** ; $f(1{,}5 ; -0{,}75) = 0{,}25 - 0{,}781 =$ **−0,531** ; $f(1{,}25 ; -0{,}4375) = 0{,}0625 - 1{,}221 =$ **−1,158** : $f$ baisse à chaque pas (0,1). D'un pas à l'autre, $x - 1$ est divisé par 2 (1 ; 0,5 ; 0,25), tandis que $y + 2$ est multiplié par 1,25 (1 ; 1,25 ; 1,5625) (0,1). **Non**, elle ne converge pas vers le point critique : $y + 2$ grandit à chaque pas, et la descente s'échappe dans la direction de $y$, où $f$ descend sans fin ($f$ n'a pas de minimum) (0,1). Remarque : la distance au point critique passe de 1,414 à 1,346, puis à 1,582 ; elle baisse d'abord, parce que $x - 1$ diminue plus vite que $y + 2$ n'augmente, puis elle remonte. « Elle s'en approche, puis s'en éloigne », justifié, rapporte aussi le point.

f) **[0,3]** Depuis $(2 ; -2)$, $\frac{\partial f}{\partial y} = 0$ à chaque pas : $y$ reste égal à −2, et $x - 1$ est divisé par 2 à chaque pas. La descente converge vers le **point selle** (1 ; −2) et s'y arrête, puisque le gradient y est nul (0,1). C'est un mauvais signe : un gradient nul ressemble à un minimum, alors que ce n'en est pas un (0,1). La moindre perturbation en $y$ (du bruit, un arrondi, le hasard d'un mini-batch) suffit à repartir, car le point est instable dans cette direction (0,1).

**Erreurs fréquentes** : le signe de $\frac{\partial f}{\partial y}$ ; monter au lieu de descendre ($x + \eta \nabla f$) ; conclure « minimum » parce que le gradient est nul.
**Remédiation** : 5.3, 5.5, 5.20, 5.Q8.

## CP1.10 — Prédire l'effet du learning rate 🔮 · 1 point

a) **[0,6]** (0,15 par lettre) **C, A, D, B** (« CADB »). Avec $f'(x) = 6x$ : $x_{t+1} = x_t - 6\eta\, x_t = (1 - 6\eta)\, x_t$.
- $\eta = 0{,}3$ : facteur −0,8 : le signe change à chaque pas, mais $|-0{,}8| < 1$ : elle converge en oscillant (**C**).
- $\eta = 0{,}02$ : facteur 0,88, positif et proche de 1 : la descente converge lentement, sans osciller (**A**).
- $\eta = 0{,}4$ : facteur −1,4 : $|-1{,}4| > 1$ : elle diverge, et $x$ change de signe à chaque pas (**D**).
- $\eta = 0{,}15$ : facteur 0,1 : elle converge très vite (**B**).

b) **[0,2]** $x_{t+1} = (1 - 6\eta)\, x_t$ (0,1) ; la descente converge si et seulement si $|1 - 6\eta| < 1$, soit $0 < \eta <$ **1/3** ≈ 0,333 (0,1). ($\eta = 1/6$ atteint 0 en un seul pas.)

c) **[0,2]** La loss vaut $3x^2$ : elle est multipliée par $(1 - 6\eta)^2$ à chaque pas. Cas (A) : par $0{,}88^2 \approx 0{,}77$ ; elle baisse régulièrement, sans rebond, mais lentement à côté du cas (B), où elle est multipliée par 0,01 à chaque pas (0,1). Cas (D) : par $1{,}4^2 = 1{,}96$ ; la loss monte de plus en plus vite, jusqu'à déborder (`inf`, puis `nan`) (0,1). Remarque : ici, $x$ change de signe à chaque pas, mais pas la loss $3x^2$, qui monte sans osciller. Sur un vrai réseau, la loss grimpe souvent en dents de scie avant d'exploser (fiche du ch. 5) ; ici, « elle oscille, puis explose » ne rapporte que 0,05 : l'explosion est juste, mais seule $x$ oscille.

**Erreurs fréquentes** : dériver $3x^2$ en $3x$ (la borne devient 2/3) ; oublier le facteur 3, $f'(x) = 2x$ (la borne devient 1) ; donner 1/6, le learning rate qui atteint 0 en un pas, au lieu de la borne ; confondre (C) et (D) : le signe du facteur dit s'il y a oscillation, sa valeur absolue s'il y a convergence.
**Remédiation** : 5.3 (questions e à g), 5.19, 1.17.

## CP1.11 — Entropie, cross-entropy, KL et Huffman ✏️ · 2 points

a) **[0,3]** Avec les surprises de l'énoncé : $H(p) = 0{,}4 \times 1{,}3219 + 0{,}3 \times 1{,}7370 + 0{,}2 \times 2{,}3219 + 0{,}1 \times 3{,}3219 = 0{,}52876 + 0{,}5211 + 0{,}46438 + 0{,}33219 = 1{,}84643 \approx$ **1,846** bit (0,15 pour la formule, 0,15 pour la valeur).

b) **[0,3]** Fusions : R (0,1) et M (0,2) donnent un groupe de 0,3 ; ce groupe et F (0,3) donnent 0,6 ; enfin V (0,4) et ce groupe de 0,6. Longueurs : **[1, 2, 3, 3]**, par exemple V = `0`, F = `10`, M = `110`, R = `111`.

c) **[0,2]** $0{,}4 \times 1 + 0{,}3 \times 2 + 0{,}2 \times 3 + 0{,}1 \times 3 =$ **1,9** bit par lettre.

d) **[0,2]** $H(p) = 1{,}846 \le 1{,}9 < H(p) + 1$, et 1,9 bit, c'est moins que les 2 bits d'un code fixe pour quatre groupes. Huffman donne des longueurs **entières** ; il n'atteint $H(p)$ que si chaque surprise $-\log_2 p_i$ est entière (des probabilités en puissances de 1/2), ce qui n'est pas le cas ici (1,3219 n'est pas entier).

e) **[0,2]** $H(p, q) = -\sum_i p_i \log_2 0{,}25 = 2 \sum_i p_i =$ **2** bits : chaque lettre coûte 2 bits avec ce code.

f) **[0,3]** $\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p) = 2 - 1{,}846 \approx$ **0,154** bit : le surcoût du code fait pour une autre source.

g) **[0,3]** $\mathrm{KL}(q \,\|\, p) = H(q, p) - H(q)$, avec $H(q, p) = 0{,}25 \times (1{,}3219 + 1{,}7370 + 2{,}3219 + 3{,}3219) = 0{,}25 \times 8{,}7027 \approx 2{,}176$ et $H(q) = 2$ : $\mathrm{KL}(q \,\|\, p) \approx$ **0,176** bit (ou directement $\sum_i 0{,}25 \log_2 \frac{0{,}25}{p_i} = 0{,}25 \times (-0{,}678 - 0{,}263 + 0{,}322 + 1{,}322)$) (0,2). Les deux KL sont positives mais **différentes** : la KL n'est pas symétrique, ce n'est pas une distance (0,1).

h) **[0,2]** $H(p, q') = +\infty$ : les consonnes rares arrivent (probabilité 0,1) et coûteraient $-\log_2 0 = +\infty$ bits (0,1). On l'évite en **lissant** $q'$ (lissage de Laplace : un pseudo-compte pour chaque groupe), pour qu'aucune issue possible n'ait une probabilité nulle (0,1).

**Erreurs fréquentes** : calculer en nats (1,280 en a) ; prendre la moyenne simple des longueurs (2,25) ; calculer $H(q, p)$ (2,176) au lieu de $H(p, q)$ en e) ; inverser les deux sens de la KL.
**Remédiation** : 6.3, 6.5, 6.6, 6.17, 6.Q10, 6.Q11.

## CP1.12 — Un LLM expliqué en cinq lignes : données, loss, perplexité 🗣️ · 1 point

**Réponse modèle** : « On rassemble des milliards de phrases (livres, sites web, code), découpées en petits morceaux de mots, les tokens. Le modèle lit le début d'un texte et doit deviner le token suivant : il donne une probabilité à chaque token possible. Sa loss mesure sa surprise devant le bon token : elle est faible s'il lui avait donné une forte probabilité, et l'entraînement ajuste ses paramètres, petit à petit, pour la faire baisser en moyenne. Une perplexité de 10 veut dire qu'en moyenne il hésite autant que s'il devait choisir au hasard entre 10 tokens également probables. »

**Critères** (0,2 chacun) : 1) les données : beaucoup de textes, découpés en tokens ; 2) la tâche : prédire le token suivant, avec une probabilité pour chaque token ; 3) la loss : la surprise moyenne devant le bon token (la cross-entropy, sans avoir à la nommer) ; 4) l'apprentissage : on ajuste les paramètres pour la faire baisser (descente de gradient) ; 5) la perplexité : un nombre de choix équiprobables équivalent, le tout en cinq lignes, sans jargon non expliqué. Retire 0,2 au-delà de cinq lignes ou avec une formule.
**Erreurs fréquentes** : dire que le modèle « cherche la réponse dans une base de données » ou qu'il « comprend » ; oublier la loss ; lire une perplexité de 10 comme « 10 % d'erreurs » ou « une erreur sur dix » ; employer « token » ou « gradient » sans les expliquer.
**Remédiation** : 6.20, 6.22, 6.E3, 1.Q11.

## CP1.13 — Corrélation, causalité et échantillon ⚖️ · 1 point

a) **[0,6]** (0,2 par raison, de trois catégories différentes)
- **La causalité** : une corrélation ne prouve pas que la vitesse fait réussir. Une variable de confusion peut créer le lien : les étudiants déjà à l'aise accélèrent les vidéos **et** réussissent mieux. Le lien peut même aller dans l'autre sens : ceux qui comprennent vite accélèrent.
- **L'échantillon** : 35 volontaires, c'est peu, et ce ne sont pas des étudiants tirés au hasard (biais de sélection : les volontaires sont peut-être les plus motivés).
- **L'incertitude** : avec 35 personnes, la corrélation est très imprécise (un intervalle de confiance à 95 % irait environ de 0,1 à 0,66) ; et 0,42 reste un lien modéré ($r^2 \approx 0{,}18$).
Aussi accepté : la façon de mesurer la « vitesse de lecture », un seul chiffre choisi parmi beaucoup d'autres possibles, aucun nuage de points montré (un point aberrant peut créer une corrélation, Anscombe).

b) **[0,2]** Une **expérience contrôlée**, randomisée (un test A/B) : tirer au sort deux groupes d'étudiants, l'un en ×1, l'autre en ×1,5, puis comparer leurs notes avec un intervalle de confiance. Le tirage au sort équilibre les variables de confusion entre les deux groupes.

c) **[0,2]** Un **intervalle de confiance** de la corrélation (par bootstrap sur les 35 étudiants), le mode de recrutement de l'échantillon, et le nuage de points lui-même.

**Erreurs fréquentes** : trois raisons qui sont trois variantes de « corrélation n'est pas causalité » (une seule catégorie : 0,2) ; proposer en b) « plus d'étudiants » sans tirage au sort : un grand échantillon d'observation garde ses variables de confusion ; juger 0,42 « fort, donc prouvé » ou « faible, donc sans intérêt ».
**Remédiation** : 2.10, 2.12, 2.22, 2.Q10, 2.Q12.

## CP1.14 — Entretien express 💼 · 1 point

**Réponse modèle en 60 secondes** : « Pas encore. Avec 2 % de fraudes, un modèle qui répond toujours « pas de fraude » a déjà 98 % d'accuracy : notre chiffre ne prouve donc rien, c'est peut-être exactement ce que fait ce modèle trivial. Je veux d'abord la matrice de confusion : combien de fraudes il attrape, le recall, et quelle part de ses alertes sont de vraies fraudes, la precision, puisque c'est elle qui fixe la charge de l'équipe qui vérifie. Comme les fraudes sont rares, je regarde la courbe precision-recall et l'average precision, pas seulement la courbe ROC. Ensuite, je choisis le seuil sur un jeu de validation, selon le coût d'une fraude manquée et celui d'une vérification inutile. Enfin, je vérifie le protocole : pas de fuite de données, un test postérieur à l'entraînement, puisque la fraude évolue, et un suivi des performances une fois le modèle en production. »

**Critères** : le modèle trivial à 98 % (0,3) ; la matrice de confusion, le recall et la precision (0,3) ; la courbe precision-recall, ou le seuil choisi sur une validation selon les coûts (0,2) ; le protocole : fuite, test dans le temps, suivi en production (0,2).
**Erreurs fréquentes** : répondre « oui, 98 %, c'est excellent » ; remplacer l'accuracy par une autre mesure unique sans parler du seuil ni des coûts ; proposer de rééquilibrer le jeu de **test** (on évalue toujours sur la proportion réelle de fraudes).
**Relances possibles** : « Le recall est de 90 %, ça vous va ? » (ça dépend de la precision et du nombre d'alertes par jour) · « Comment gérer le déséquilibre à l'entraînement ? » (pondérer les classes ou rééchantillonner, mais toujours évaluer sur la distribution réelle).
**Remédiation** : 3.E1, 3.27, 3.29, 3.Q9.

---

## Barème récapitulatif et remédiation

| Question | Points | Ch. | Si tu as moins de la moitié des points, refais… |
|---|---|---|---|
| CP1.1 | 1,5 | 1 à 6 | les quiz et exercices cités au CP1.1, puis les flashcards des chapitres concernés |
| CP1.2 | 1 | 0B | 0B.8, 0B.15, 0B.16, 0B.18 |
| CP1.3 | 1 | 0A | 0A.51, 0A.56, 2.4, 2.15 |
| CP1.4 | 1,5 | 2 | 2.4, 2.7, 2.13, 2.15 |
| CP1.5 | 1,5 | 2 | 2.21, 2.22, 2.24 |
| CP1.6 | 2 | 3, 4 | 3.7, 3.21, 4.4, 4.8 |
| CP1.7 | 1,5 | 3 | 3.24, 3.26, 3.27 |
| CP1.8 | 2 | 4 | 4.5, 4.6, 4.7, 4.16 |
| CP1.9 | 2 | 5 | 5.3, 5.5, 5.20 |
| CP1.10 | 1 | 1, 5 | 5.3, 5.19, 1.17 |
| CP1.11 | 2 | 6 | 6.3, 6.5, 6.6, 6.17 |
| CP1.12 | 1 | 1, 6 | 6.20, 6.22, 6.E3 |
| CP1.13 | 1 | 2 | 2.10, 2.12, 2.22 |
| CP1.14 | 1 | 3 | 3.E1, 3.27, 3.29 |
| **Total** | **20** | | |

**Lire ta note**
- **16 à 20** : les fondations sont solides. Passe à la partie II ; garde les flashcards de la partie I dans tes révisions.
- **12 à moins de 16** : c'est bien. Fais la remédiation des questions où tu as eu moins de la moitié des points, puis passe à la suite.
- **8 à moins de 12** : avant la partie II, fais la remédiation, relis la synthèse (`05_synthese.md`) et les fiches des chapitres concernés.
- **Moins de 8** : reprends les exercices ★★ des chapitres où tu as perdu le plus de points, puis refais l'examen.

Dans tous les cas, **refais dans une semaine** les questions où tu as eu moins de la moitié des points, sans regarder ce corrigé : c'est l'effort de rappel qui fixe les notions. Reporte ta note et tes points faibles dans `mon_travail/suivi/journal.md`.
