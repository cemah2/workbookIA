# Cheatsheet NumPy

> Aide-mémoire rempli au fil des chapitres (0A, 0B, 2). Une ligne = une commande utile + ce qu'elle fait.

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

## Fonctions mathématiques

| Code | Effet | Ch. |
|---|---|---|
| `np.exp(x)`, `np.log(x)`, `np.log2(x)`, `np.log10(x)` | $e^x$, $\ln x$ (**népérien**), $\log_2 x$, $\log_{10} x$ | 0B |
| `np.log(0)` → `-inf`, `np.log(-1)` → `nan`, `np.exp(1000)` → `inf` | avertissements `RuntimeWarning` au lieu d'erreurs : vérifie les entrées | 0B |
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

