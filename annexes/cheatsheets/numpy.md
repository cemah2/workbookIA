# Cheatsheet NumPy

> Aide-mémoire rempli au fil des chapitres (0A, 0B, puis une section par chapitre). Une ligne = une commande utile + ce qu'elle fait.

## Créer des tableaux

| Code | Effet | Ch. |
|---|---|---|
| `np.array([[1, 2], [3, 4]])` | array à partir de listes (un seul `dtype` pour tous les éléments) | 0A |
| `np.zeros((3, 4))`, `np.ones(5)`, `np.full((2, 2), 7)` | array rempli de 0, de 1, ou d'une valeur | 0A |
| `np.arange(0, 10, 2)` | 0, 2, 4, 6, 8 (fin exclue, comme `range`) | 0A |
| `np.linspace(0, 1, 11)` | 11 valeurs régulièrement espacées de 0 à 1 inclus | 0A |
| `a.shape`, `a.ndim`, `a.size`, `a.dtype` | forme, nombre de dimensions, nombre d'éléments, type | 0A |
| `a.astype(np.float32)` | copie convertie dans un autre type | 0A |
| `df[cols].to_numpy()` | d'un DataFrame pandas à un array `(n, p)` | 0A |

## Formes, reshape et axes

| Code | Effet | Ch. |
|---|---|---|
| `a.reshape(4, 6)`, `a.reshape(2, -1, 3)` | nouvelle forme, même nombre d'éléments ; `-1` = « calcule-la » | 0A |
| `images.reshape(len(images), -1)` | aplatit chaque image : `(N, 28, 28)` → `(N, 784)` | 0A |
| `v.reshape(-1, 1)`, `v[:, None]` | vecteur `(n,)` → colonne `(n, 1)` | 0A |
| `a.T` | transposée : `(n, p)` → `(p, n)` | 0A |
| `X.mean(axis=0)` | une valeur **par colonne** (l'axe 0 disparaît) : forme `(p,)` | 0A |
| `X.max(axis=1)` | une valeur **par ligne** : forme `(n,)` | 0A |
| `X.mean(axis=1, keepdims=True)` | garde l'axe réduit avec la taille 1 : `(n, 1)` | 0A |
| `np.concatenate([a, b], axis=0)`, `np.vstack`, `np.hstack`, `np.stack` | coller le long d'un axe existant, ou d'un nouvel axe (`stack`) | 0A |

## Indexation et masques booléens

| Code | Effet | Ch. |
|---|---|---|
| `X[10, 2]`, `X[-1]` | un élément ; la dernière ligne | 0A |
| `X[:, 2]` / `X[:, 2:3]` | une colonne, forme `(n,)` / forme `(n, 1)` | 0A |
| `X[100:200]` | lignes 100 à 199 : une **vue** (modifier la vue modifie `X`) | 0A |
| `X[[0, 5, 9]]` | lignes choisies par une liste d'indices : une **copie** | 0A |
| `X[X[:, 3] > 4500]` | lignes qui vérifient la condition (masque booléen) : une copie | 0A |
| `(cond1) & (cond2)`, `(cond1) \| (cond2)`, `~cond` | et, ou, non entre masques (parenthèses obligatoires ; jamais `and` / `or`) | 0A |
| `mask.sum()`, `mask.mean()` | nombre de `True`, proportion de `True` | 0A |
| `a.copy()`, `np.shares_memory(a, b)` | copie indépendante ; `a` et `b` partagent-ils la mémoire ? | 0A |

## Opérations vectorisées et broadcasting

| Code | Effet | Ch. |
|---|---|---|
| `X[:, 3] / 1000`, `a + b`, `a * b`, `a ** 2` | opérations élément par élément, sans boucle | 0A |
| `np.sqrt`, `np.exp`, `np.log`, `np.abs`, `np.round(a, 2)` | fonctions universelles (*ufuncs*) appliquées à chaque élément | 0A |
| `np.where(cond, x, y)` | `x` là où la condition est vraie, `y` ailleurs | 0A |
| `(X - X.mean(axis=0)) / X.std(axis=0)` | standardisation par broadcasting : `(n, p)` avec `(p,)` | 0A |
| `a.sum()`, `a.mean()`, `a.std()`, `a.min()`, `a.max()` | réductions (tout le tableau, ou `axis=`) | 0A |
| `a.argmax()`, `np.argmax(X, axis=1)` | **position** du maximum (la première en cas d'égalité) | 0A |
| `np.sort(a)`, `np.argsort(a)` | valeurs triées ; indices qui trient | 0A |
| `np.unique(y, return_counts=True)` | valeurs distinctes (triées) et leurs effectifs | 0A |
| `np.isnan(a)`, `np.isclose(a, b)` | valeurs manquantes ; égalité approchée des flottants | 0A |

## Algèbre linéaire

| Code | Effet | Ch. |
|---|---|---|
| `u @ v`, `np.dot(u, v)` | produit scalaire de deux vecteurs `(n,)` : un nombre | 0B |
| `u * v` | produit de Hadamard (élément par élément) : un vecteur, **pas** un produit scalaire | 0B |
| `np.linalg.norm(v)`, `np.linalg.norm(v, ord=1)`, `ord=np.inf` | normes L2, L1 et L∞ | 0B |
| `np.linalg.norm(a - b)`, `math.dist(a, b)` | distance euclidienne | 0B |
| `(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))` | similarité cosinus | 0B |
| `A.T` | transposée : forme `(m, n)` → `(n, m)` ; `.T` ne change rien sur un vecteur `(n,)` | 0B |
| `A @ B`, `np.matmul(A, B)` | produit matriciel : `(m, n) @ (n, p)` → `(m, p)` ; sinon `ValueError: matmul: ... mismatch` | 0B |
| `A @ v` | produit matrice-vecteur : `(m, n) @ (n,)` → `(m,)` | 0B |
| `X @ w + b` | prédictions d'un modèle linéaire pour tous les exemples d'un coup | 0B |
| `np.eye(n)` | matrice identité `(n, n)` | 0B |
| `np.linalg.det(M)`, `np.linalg.inv(M)` | déterminant (un flottant, parfois `1.9999999999999996`) ; inverse | 0B |
| `np.linalg.solve(M, y)` | solution de `M @ x = y` : plus rapide et plus précis que `inv(M) @ y` | 0B |
| `v.reshape(-1, 1)`, `v[:, None]` | vecteur `(n,)` → colonne `(n, 1)` (attention au broadcasting silencieux) | 0B |
| `u[:, None] @ v[None, :]`, `np.outer(u, v)` | produit extérieur : la matrice `(n, m)` des $u_i v_j$ (`u @ v.T` donne un nombre : `.T` ne change pas un vecteur `(n,)`) | 0B |

## Fonctions mathématiques

| Code | Effet | Ch. |
|---|---|---|
| `np.exp(x)`, `np.log(x)`, `np.log2(x)`, `np.log10(x)` | $e^x$, $\ln x$ (**népérien**), $\log_2 x$, $\log_{10} x$ | 0B |
| `np.log(0)` → `-inf`, `np.log(-1)` → `nan`, `np.exp(1000)` → `inf` | avertissements `RuntimeWarning` au lieu d'erreurs : vérifie les entrées | 0B |
| `np.log1p(u)`, `np.sum(np.log(p))` | $\ln(1 + u)$ ; le logarithme d'un produit sans calculer le produit (pas d'underflow) | 0B |
| `with np.errstate(over="ignore"):` | masquer un avertissement attendu, dans ce bloc seulement | 0B |
| `np.floor(x)`, `np.ceil(x)`, `np.sign(x)`, `np.abs(x)` | partie entière par défaut, par excès, signe (−1, 0, 1), valeur absolue | 0B |
| `np.tanh(x)`, `1 / (1 + np.exp(-x))` | tangente hyperbolique ; sigmoïde | 0B |
| `np.cos(x)`, `np.pi`, `np.radians(60)` | cosinus (en radians), $\pi$, degrés → radians | 0B |
| `np.cumsum(a)`, `np.prod(a)` | sommes cumulées ; produit des éléments | 0B |
| `np.average(x, weights=w)` | moyenne pondérée | 0B |
| `np.convolve(x, np.ones(k) / k, mode="valid")` | moyenne mobile d'ordre `k` (`n - k + 1` valeurs) | 0B |
| `x.var()`, `x.std()` | variance et écart-type (division par `n` ; `ddof=1` pour `n - 1`) | 0B |

## Aléatoire reproductible (default_rng)

| Code | Effet | Ch. |
|---|---|---|
| `rng = np.random.default_rng(42)` | générateur avec une graine : même graine, même suite | 0A |
| `rng.random(3)`, `rng.normal(0, 1, size=5)` | uniformes dans [0, 1[ ; loi normale | 0A |
| `rng.integers(1, 7, size=10)` | entiers de 1 à 6 (fin exclue) | 0A |
| `rng.permutation(n)` | les entiers de 0 à n − 1 mélangés (pour mélanger un dataset) | 0A |
| `rng.choice(n, size=k, replace=False)` | k indices distincts tirés parmi n (sans remise) | 0A |
| `rng.choice(a, size=k)` | k éléments tirés **avec** remise (c'est la valeur par défaut de `replace`) | 2 |
| `rng.choice(len(p), size=k, p=p)` | k tirages dans une loi catégorielle de probabilités `p` | 2 |
| `rng.uniform(a, b, size)`, `rng.normal(mu, sigma, size)` | lois uniforme sur [a, b[ et normale de moyenne `mu`, d'écart-type `sigma` | 2 |
| `rng.random(n) < p` | `n` tirages de Bernoulli de paramètre `p` (booléens) | 2 |
| `rng.integers(0, n, size=n)` | les indices d'un rééchantillon bootstrap (n parmi n avec remise) | 2 |
| `child_a, child_b = rng.spawn(2)` | deux générateurs indépendants fabriqués à partir d'un seul (une expérience par générateur) | 2 |
| `np.random.seed(0)` | règle seulement l'ancienne interface (`np.random.rand`…) : **pas** `np.random.default_rng()` | 2 |

## Statistiques descriptives (ch. 2)

| Code | Effet | Ch. |
|---|---|---|
| `np.mean(x)`, `np.median(x)`, `np.mean(X, axis=0)` | moyenne, médiane ; une valeur par colonne avec `axis=0` | 2 |
| `x.var()`, `x.var(ddof=1)`, `x.std(ddof=1)` | variance et écart-type : NumPy divise par `n` par défaut (`ddof=0`) | 2 |
| `np.percentile(x, [25, 50, 75])`, `np.quantile(x, 0.9)` | percentiles (0 à 100), quantiles (0 à 1), interpolation linéaire | 2 |
| `np.percentile(X, [25, 50, 75], axis=0)` | forme `(3, p)` : une ligne par percentile demandé, une colonne par colonne de `X` | 2 |
| `np.take(np.sort(X, axis=1), k, axis=1)` | le k-ième plus petit élément de chaque ligne (médiane le long d'un axe) | 2 |
| `counts, edges = np.histogram(x, bins=10)` | comptages par intervalle et bords des intervalles (`density=True` : aire totale 1) | 2 |
| `np.cov(x, y)[0, 1]` | covariance, divisée par `n - 1` par défaut (`ddof=0` pour diviser par `n`) | 2 |
| `np.cov(X, rowvar=False)`, `np.corrcoef(X, rowvar=False)` | matrices de covariance et de corrélation des **colonnes** (sans `rowvar=False` : des lignes) | 2 |
| `np.cumsum(p)`, `np.searchsorted(c, u, side="right")` | sommes cumulées ; position de `u` dans un tableau trié (tirage catégoriel) | 2 |
| `scipy.stats.zscore(X, axis=0)` | z-scores colonne par colonne | 2 |
| `scipy.stats.bootstrap((x,), np.mean, method="percentile", rng=rng)` | intervalle de confiance bootstrap (9 999 rééchantillons par défaut, méthode BCa par défaut) | 2 |
| `np.linalg.norm(X - X[0], axis=1)` | distance de chaque ligne de `X` à la ligne 0, en une seule opération (broadcasting) | 2 |
| `slope, intercept = np.polyfit(x, y, 1)` | droite des moindres carrés $y \approx \text{slope} \cdot x + \text{intercept}$ (pente d'abord) | 2 |


## Probabilités sur une grille, log-probabilités, loi Beta (ch. 4)

| Code | Effet | Ch. |
|---|---|---|
| `float(np.dot(prior, likelihood))` | évidence $\sum_i P(O \mid H_i)\,P(H_i)$, en `float` Python | 4 |
| `prior * likelihood / evidence` | posterior : un **nouveau** tableau (éviter `prior *= ...`, qui modifie celui de l'appelant) | 4 |
| `abs(p.sum() - 1) <= 1e-8` | « somme à 1 » avec une tolérance (`np.array([0.6, 0.3, 0.1]).sum()` vaut 0,9999999999999999) | 4 |
| `table[:, o]` | colonne de l'issue `o` : les vraisemblances $P(o \mid H_i)$ de toutes les hypothèses | 4 |
| `np.linspace(0, 1, 501)` | grille de 501 hypothèses sur un biais, de 0 à 1 compris (un pas de 0,002) | 4 |
| `np.column_stack([1 - grid, grid])` | tableau hypothèses × issues d'une pièce : colonne 0 = pile, colonne 1 = face | 4 |
| `with np.errstate(divide="ignore"):` | faire taire l'avertissement de `np.log(0)` quand $-\infty$ est voulu | 4 |
| `w = np.exp(l - l.max()); w / w.sum()` | log-probabilités → probabilités normalisées, sans underflow | 4 |
| `scipy.special.logsumexp(l)` | $\log \sum_j e^{\ell_j}$ sans underflow ; `np.exp(l - logsumexp(l))` normalise | 4 |
| `np.finfo(float).tiny`, `np.nextafter(0, 1)` | plus petit `float64` normal ($\approx 2{,}2 \times 10^{-308}$) ; plus petit positif ($\approx 5 \times 10^{-324}$) | 4 |
| `grid[np.argmax(post)]` | MAP : la valeur de la grille de plus grand posterior | 4 |
| `np.searchsorted(np.cumsum(post), level)` | premier indice où la probabilité cumulée atteint `level` (côté `"left"` : `>=`) | 4 |
| `stats.beta(a, b).mean()`, `.std()`, `.pdf(x)` | moyenne, écart-type, densité de la loi Beta (`from scipy import stats`) | 4 |
| `stats.beta(a, b).sf(0.5)` | $P(\theta > 0{,}5)$, c'est-à-dire `1 - cdf(0.5)` | 4 |
| `stats.beta(a, b).interval(0.95)`, `.ppf([0.025, 0.975])` | intervalle à 95 % à queues égales | 4 |

## Dérivées, gradients et descentes (ch. 5)

| Code | Effet | Ch. |
|---|---|---|
| `np.ndim(x) == 0` | `x` est un nombre (Python ou NumPy), pas une liste ni un tableau | 5 |
| `np.array(x, dtype=float)` | une **copie** en flottants (`np.asarray` ne copie pas un tableau NumPy, et garde ses entiers) | 5 |
| `flat = point.reshape(-1)` | une vue à plat d'un tableau de forme quelconque : modifier `flat[i]` modifie `point` | 5 |
| `(f(x + h) - f(x - h)) / (2 * h)` | pente centrée ; marche sur un tableau `x` si `f` est vectorisée (`np.sin`, `np.exp`…) | 5 |
| `10.0 ** -k` | le pas $10^{-k}$ (`10 ** -k` refuse un exposant entier négatif de NumPy) | 5 |
| `np.finfo(float).eps` | epsilon machine, $\approx 2{,}2 \times 10^{-16}$ | 5 |
| `np.linalg.norm(g)` | norme euclidienne de **toutes** les composantes de `g`, quelle que soit sa forme | 5 |
| `np.linalg.norm(path - target, axis=1)` | distance de chaque point d'un chemin `(n, 2)` à `target` | 5 |
| `np.flatnonzero(d < 1e-3)` | indices où la condition est vraie ; `[0]` donne le premier | 5 |
| `np.concatenate([y[max(0, i - k):i], y[i + 1:i + k + 1]])` | les `k` voisins de chaque côté de `y[i]`, tronqués aux bords | 5 |
| `np.eye(n)[i]` | le vecteur $\mathbf{e}_i$ (1 en position `i`, 0 ailleurs) | 5 |
| `np.linalg.eigvalsh(H)` | valeurs propres d'une matrice symétrique (une hessienne) : leurs signes classent un point critique | 5 |
| `X, Y = np.meshgrid(xs, ys)` | grille de points pour tracer $f(x, y)$ : `Z = f(X, Y)` | 5 |
| `ax.contour(X, Y, Z, levels=20)`, `ax.contourf(...)` | lignes de niveau (remplies) ; `plt.colorbar(...)` pour la légende des couleurs | 5 |
| `ax.quiver(X, Y, U, V)` | une flèche $(U, V)$ en chaque point, par exemple le gradient | 5 |
| `ax.set_aspect("equal")` | même échelle sur les deux axes : les angles droits restent droits | 5 |
| `fig.add_subplot(1, 2, 1, projection="3d")`, `ax.plot_surface(X, Y, Z)` | une surface en 3D, à côté d'un panneau ordinaire | 5 |
| `wb.plot.plot_contour(f, xlim, ylim, path=path)` | carte de lignes de niveau avec une trajectoire (Rosenbrock, ch. 5 et 19) | 5 |

## Comptes, entropies et codes (ch. 6)

| Code | Effet | Ch. |
|---|---|---|
| `np.log2(p)`, `np.log(p)` | logarithme en base 2 (bits), népérien (nats) ; `np.log(x) / np.log(b)` pour une base `b` | 6 |
| `p[p > 0]` | les issues possibles seulement : $0 \log 0 = 0$ sans `nan` | 6 |
| `collections.Counter(text)`, `counts.most_common(10)` | compter les caractères (ou les mots) en un passage ; un élément absent compte 0 | 6 |
| `np.frombuffer(text.encode("ascii", "ignore"), dtype=np.uint8)` | les codes des octets d'un texte (97 pour « a ») | 6 |
| `np.bincount(codes, minlength=123)[97:123]` | les comptes des lettres a à z, d'un coup | 6 |
| `np.add.at(counts, (a[:-1], a[1:]), 1)` | compter des paires consécutives (bigrammes) dans une matrice, même avec des indices répétés | 6 |
| `table[idx[:-1], idx[1:]]` | indexation avancée : les probabilités de toutes les paires (lettre précédente, lettre suivante) d'un texte | 6 |
| `np.argsort(-p, kind="stable")` | indices du plus probable au moins probable, ordre d'origine en cas d'égalité | 6 |
| `heapq.heapify(h)`, `heapq.heappop(h)`, `heapq.heappush(h, x)` | file de priorité : le plus petit élément sort en premier (code de Huffman) | 6 |
| `itertools.product("ab", repeat=2)` | tous les blocs de 2 symboles : `aa`, `ab`, `ba`, `bb` (avec `"".join`) | 6 |
| `scipy.stats.entropy(pk, qk=None, base=None)` | entropie, ou KL avec `qk` ; normalise `pk` (des comptes suffisent) ; en nats sans `base=2` | 6 |
| `scipy.spatial.distance.jensenshannon(p, q, base=2)` | la **racine** de la divergence de Jensen-Shannon | 6 |
| `zlib.compress(text.encode("utf-8"), 9)`, `zlib.decompress` | compression sans perte (DEFLATE) ; `bz2`, `lzma` : deux autres compresseurs de la bibliothèque standard | 6 |
| `min(timeit.repeat(lambda: f(x), number=3, repeat=5)) / 3` | durée d'un appel : le minimum de plusieurs mesures | 6 |

## Distances, centroïdes et clustering (ch. 7)

| Code | Effet | Ch. |
|---|---|---|
| `(A ** 2).sum(axis=1)[:, None] - 2 * A @ B.T + (B ** 2).sum(axis=1)[None, :]` | toutes les distances au carré entre les lignes de `A` et celles de `B`, forme `(n_a, n_b)`, sans boucle ; `np.maximum(D, 0.0)` efface les petits négatifs dus aux arrondis | 7 |
| `A[:, None, :] - B[None, :, :]` | toutes les différences, forme `(n_a, n_b, d)` : simple, mais gourmand en mémoire | 7 |
| `D.argmin(axis=1)` | pour chaque point, l'indice du centre le plus proche (le premier en cas d'égalité) | 7 |
| `np.array([X[y == c].mean(axis=0) for c in np.unique(y)])` | un centroïde par classe, dans l'ordre trié des classes | 7 |
| `np.fill_diagonal(D, 0.0)` ; `off = ~np.eye(n, dtype=bool)` | mettre la diagonale à 0 (en place) ; un masque qui l'écarte | 7 |
| `np.where(off, D, np.inf).min(axis=1)` | distance de chaque point à son plus proche voisin, lui-même exclu | 7 |
| `np.unique(cells, axis=0)` | les **lignes** distinctes d'un tableau (des cases occupées, des points sans doublon) | 7 |
| `np.unique(labels, return_inverse=True)[1]` ; `np.bincount(codes)` | numéroter des labels quelconques de 0 à $K - 1$ ; la taille de chaque groupe | 7 |
| `rng.choice(n, p=w / w.sum())` | un indice tiré avec des poids `w` (k-means++ : `w` = les $D^2$) | 7 |
| `np.minimum(d2, d2_new)` | minimum élément par élément de deux tableaux (distance au centre le plus proche, mise à jour) | 7 |
| `copy.deepcopy(model)` | une copie indépendante d'un objet Python (un modèle non entraîné, avant chaque `fit`) | 7 |
| `start = time.perf_counter()` … `time.perf_counter() - start` | durée d'un calcul, en secondes | 7 |

## Découper, rééchantillonner et comparer (ch. 8)

| Code | Effet | Ch. |
|---|---|---|
| `math.ceil(0.2 * n)` | la taille du test de scikit-learn, arrondie vers le haut ($0{,}2 \times 333 = 66{,}6$ donne 67) | 8 |
| `perm = rng.permutation(n)` ; `test_idx, train_idx = perm[:n_test], perm[n_test:]` | un hold-out : **une** permutation, qui sert à indexer tous les tableaux (`X[train_idx]`, `y[train_idx]`) pour les garder alignés | 8 |
| `np.setdiff1d(np.arange(n), val_idx)` | les indices qui ne sont pas dans `val_idx`, triés : la partie d'entraînement | 8 |
| `np.array_split(np.arange(n), k)` | $k$ morceaux consécutifs, les $n \bmod k$ premiers ayant un élément de plus (les folds de `KFold`) | 8 |
| `np.concatenate([[0], np.cumsum(sizes)])` | les bornes de morceaux de tailles `sizes` : le morceau $j$ va de `bounds[j]` à `bounds[j + 1]` | 8 |
| `order = np.argsort(codes, kind="stable")` ; `order[i::k]` | indices triés par classe (ordre d'origine gardé dans chaque classe) ; distribués « une carte par fold » | 8 |
| `_, codes, counts = np.unique(y, return_inverse=True, return_counts=True)` | numéroter les classes (ordre trié) et compter chacune | 8 |
| `np.polyfit(x, y, deg)` ; `np.polyval(coef, x)` | polynôme des moindres carrés (coefficients du plus haut degré au plus bas) ; l'évaluer en chaque `x` | 8 |
| `flips = rng.random((n_perm, n)) < 0.5` ; `np.where(flips, -d, d).sum(axis=1)` | test par permutation apparié : un tirage de signes par ligne, puis la statistique de chaque tirage | 8 |
| `rows = rng.integers(0, n, size=(n_boot, n))` ; `d[rows].mean(axis=1)` | bootstrap vectorisé : un rééchantillon (avec remise) par ligne, puis la moyenne de chacun | 8 |
| `years // 10 * 10` | la décennie de chaque année (des groupes pour `GroupKFold`) | 8 |

## Moindres carrés, pénalités et grilles de droites (ch. 9)

| Code | Effet | Ch. |
|---|---|---|
| `w = np.linalg.lstsq(X, y, rcond=None)[0]` | solution des moindres carrés ; s'il y en a une infinité (features redondantes, $p > n$), la plus courte | 9 |
| `np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)` | Ridge en forme fermée sur des données centrées : on résout le système, sans calculer d'inverse | 9 |
| `np.linalg.pinv(X) @ y` | solution de norme minimale (pseudo-inverse) : la même que `lstsq` | 9 |
| `Xc = X - X.mean(axis=0)` ; `b = y.mean() - X.mean(axis=0) @ w` | centrer les colonnes, puis retrouver l'ordonnée à l'origine | 9 |
| `np.column_stack([x ** k for k in range(1, d + 1)])` | les colonnes $x, x^2, \dots, x^d$ d'une seule feature | 9 |
| `itertools.combinations_with_replacement(range(p), k)` | les monômes de degré $k$, dans l'ordre de `PolynomialFeatures` : `(0, 0), (0, 1), (1, 1)` pour $p = k = 2$ | 9 |
| `np.prod(X[:, list(combo)], axis=1)` | la colonne d'un monôme : `combo = (0, 0, 1)` donne $x_0^2 x_1$ | 9 |
| `np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)` | seuillage doux, élément par élément (renvoie `-0.0` pour un $z$ négatif mis à zéro ; `+ 0.0` l'efface) | 9 |
| `P.mean(axis=0)`, `P.var(axis=0)` | une ligne par modèle, une colonne par point : le modèle moyen et la variance en chaque point (`ddof=0`) | 9 |
| `S, B = np.meshgrid(slopes, intercepts)` | deux grilles de forme `(n_b, n_s)` : `S[i, j]` est la pente de la colonne `j`, `B[i, j]` l'ordonnée de la ligne `i` | 9 |
| `logp -= logp.max()` ; `post = np.exp(logp)` ; `post /= post.sum()` | normaliser des log-probabilités sans underflow (même résultat que `np.exp(logp - logsumexp(logp))`) | 9 |
| `np.unravel_index(post.argmax(), post.shape)` | la ligne et la colonne du maximum d'un tableau 2D (la droite la plus probable de la grille) | 9 |
| `np.where(z > 0, 1.0, -1.0)` | le seuil du perceptron, élément par élément ; `np.sign(z)` donnerait 0 en 0 | 10 |
| `np.hstack([np.ones((len(X), 1)), X])` | l'astuce du biais : une colonne de 1 en tête de `X` (`np.column_stack` marche aussi) | 10 |
| `rng.permutation(n)` puis `X[i]`, `y[i]` | parcourir les exemples dans un ordre aléatoire sans toucher aux tableaux ; `rng.shuffle(X)` mélange `X` sur place, sans `y` | 10 |
| `w_sum += w` après chaque exemple, puis `w_sum / count` | moyenne des poids au fil de l'entraînement (perceptron moyenné) ; `+=` sur un tableau modifie l'objet, attention aux alias | 10 |
| `best = np.flatnonzero(q == q.max())` ; `rng.choice(best)` | l'argmax avec tirage au sort parmi les ex aequo (`np.argmax` prend toujours le premier) | 11 |
| `rng.random() < epsilon` | un événement de probabilité `epsilon` (explorer ou non) | 11 |
| `rng.beta(1 + s, 1 + f)` | un tirage du posterior Beta de chaque bras d'un coup (`s`, `f` : tableaux de succès et d'échecs) | 11 |
| `rng.choice(values, size=n, p=w / w.sum())` | un tirage pondéré (ici proportionnel à `w`) ; avec remise par défaut | 11 |
| `np.cumsum(best_mean - means[actions])` | le pseudo-regret cumulé d'une partie de bandit | 11 |
| `np.rint(q * n)` | les succès d'un bras retrouvés à partir de sa moyenne et de son compteur (arrondis contre les erreurs de virgule flottante) | 11 |
| `np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), -(x**2 + y**2), rcond=None)` | l'ajustement algébrique d'un cercle : $D$, $E$, $F$ de $x^2 + y^2 + Dx + Ey + F = 0$ | 11 |
