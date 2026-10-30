# Formulaire

Toutes les formules du workbook, rangées par chapitre (complété à chaque nouveau chapitre). Notations : BIBLE §6.

## Conventions de notation

| Objet | Notation | Exemple |
|---|---|---|
| scalaire | italique minuscule | $x$, $\eta$ |
| vecteur | gras minuscule | $\mathbf{x}$, $\mathbf{w}$ |
| matrice | gras majuscule | $\mathbf{W}$, $\mathbf{X}$ |
| prédiction / cible | chapeau / sans chapeau | $\hat{y}$, $y$ |
| loss | $L$ | $L(\hat{y}, y)$ |
| learning rate | $\eta$ | $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L$ |
| somme pondérée / sortie activée | $z$ / $a$ | $z = \mathbf{w}\cdot\mathbf{x} + b$, $a = \sigma(z)$ |
| fonction d'activation | $\sigma$ ou $f$ | |
| indices | $i$ exemple, $j, k$ neurones, $\ell$ couche | $a^{(\ell)}_j$ |

## Conventions de `mylearn` et des bibliothèques

Chaque bibliothèque a ses habitudes ; `mylearn` reprend celles de la bibliothèque qui sert d'oracle à ses tests. Ces tableaux évitent les confusions classiques.

### Force de la régularisation L2 (et L1)

$n$ = nombre d'exemples, $r_i = y_i - \hat{y}_i$, « moyenne » = moyenne de la loss sur les exemples.

| Où | Objectif minimisé | Gradient ajouté par la pénalité |
|---|---|---|
| `mylearn.linear.Ridge(alpha)`, `sklearn.linear_model.Ridge` | $\sum_i r_i^2 + \alpha \lVert \mathbf{w} \rVert^2$ (somme, sans ½) | $2\alpha\mathbf{w}$ |
| `mylearn.linear.Lasso(alpha)`, `sklearn.linear_model.Lasso` | $\frac{1}{2n}\sum_i r_i^2 + \alpha \lVert \mathbf{w} \rVert_1$ | $\alpha\,\mathrm{signe}(\mathbf{w})$ |
| `mylearn.logistic.LogisticRegression(alpha)` | moyenne de la log-loss $+ \frac{\alpha}{2}\lVert \mathbf{w} \rVert^2$ | $\alpha\mathbf{w}$ |
| `sklearn.linear_model.LogisticRegression(C)` | $C \sum_i \text{log-loss}_i + \frac12 \lVert \mathbf{w} \rVert^2$ | équivaut à `alpha` $= 1/(C\,n)$ ci-dessus |
| `mylearn.nn.regularization.l2_penalty(weights, lam)` | loss $+ \frac{\lambda}{2}\sum \mathbf{W}^2$ | $\lambda\mathbf{W}$ |
| `torch.optim.SGD(weight_decay=wd)`, `mylearn.optim.SGD`, `Adam` | (pas de terme dans la loss) | $wd\cdot\mathbf{p}$ ajouté au gradient : identique à `l2_penalty` avec $\lambda = wd$ |
| `torch.optim.AdamW(weight_decay=wd)`, `mylearn.optim.AdamW` | (découplé) | $\mathbf{p} \leftarrow \mathbf{p}\,(1 - \eta\, wd)$ **hors** des moments d'Adam : différent d'Adam + L2 |
| `sklearn.neural_network.MLPClassifier(alpha)` | moyenne de la loss $+ \frac{\alpha}{2n}\sum \mathbf{W}^2$, où $n$ est la taille du **mini-batch** (vérifié dans scikit-learn 1.6.1) | $\frac{\alpha}{n}\mathbf{W}$ : `weight_decay` $= \alpha / n$ (✏️ 19.6) |
| AdamW, article de Loshchilov et Hutter (2019) | (découplé) | $\mathbf{w} \leftarrow \mathbf{w} - \eta_t\lambda\mathbf{w}$, $\eta_t$ le multiplicateur du planning : `weight_decay` de PyTorch $= \lambda / \alpha$ |
| Keras `regularizers.l2(l2=λ)` (code du livre) | loss $+ \lambda\sum \mathbf{W}^2$ (sans ½) | $2\lambda\mathbf{W}$ |

### Nom du learning rate

| Livre | `mylearn` | scikit-learn | PyTorch | Keras |
|---|---|---|---|---|
| $\eta$ | `eta0` (`Perceptron`), `learning_rate` (`LogisticRegression`, `AdaBoostClassifier`, `GradientBoostingRegressor`), `lr` (`calculus.gradient_descent`, `optim.*`) | `eta0` (`Perceptron`, `SGDClassifier`), `learning_rate` (boosting), `learning_rate_init` (`MLPClassifier`) | `lr` | `learning_rate` |

### Orientation des poids

| Couche | `mylearn` | PyTorch | scikit-learn / Keras |
|---|---|---|---|
| dense | `W` de forme `(n_in, n_out)`, $\mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b}$ | `nn.Linear.weight` de forme `(n_out, n_in)`, `x @ W.T + b` | `MLPClassifier.coefs_[l]` et `Dense.kernel` : `(n_in, n_out)` |
| RNN, attention | `W_ih (D, H)`, `W_hh (H, H)`, `W_q (D, D)` | `weight_ih (H, D)` = `W_ih.T` ; `in_proj_weight` = `cat(W_q.T, W_k.T, W_v.T)` | — |
| convolution | filtres `(C_out, C_in, kh, kw)`, images `(N, C, H, W)` | idem | Keras : `(kh, kw, C_in, C_out)`, images `(N, H, W, C)` |

### Unités de l'information

`mylearn.info` calcule en **bits** (log en base 2) par défaut (`base=2.0`), comme le livre au ch. 6. Exception : `info.log_loss` est en **nats** (log népérien) par défaut, comme scikit-learn et PyTorch (`base=2` pour des bits). Conversion : 1 nat $= 1/\ln 2 \approx 1{,}443$ bit. La perplexité ne dépend pas de la base.

## Partie 0 : prérequis
### 0A · Python, notebooks et outils

| Notion | Formule | En code |
|---|---|---|
| division euclidienne | $a = b \times q + r$, avec $0 \le r < b$ | `q, r = a // b, a % b` (ou `divmod(a, b)`) |
| division entière d'un négatif | `//` arrondit vers $-\infty$ : $-17 // 5 = -4$ | `-17 // 5` |
| moyenne | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | `sum(x) / len(x)`, `x.mean()` |
| médiane ($n$ pair, liste triée $x_{(1)} \le \dots \le x_{(n)}$) | $\frac{1}{2}\left(x_{(n/2)} + x_{(n/2+1)}\right)$ | `np.median(x)`, `statistics.median(x)` |
| nombre de mini-batches par epoch | $\lceil n / b \rceil$ (dernier batch de taille $n - b\lfloor n/b \rfloor$ s'il n'est pas vide) ; $\lfloor n / b \rfloor$ avec `drop_last` | `math.ceil(n / b)`, `n // b` |
| mises à jour des poids | (batches par epoch) $\times$ (nombre d'epochs) | |
| normalisation min-max | $x' = \dfrac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$ | `(x - x.min()) / (x.max() - x.min())` |
| standardisation (par colonne) | $z = \dfrac{x - \bar{x}}{\sigma}$ | `(X - X.mean(axis=0)) / X.std(axis=0)` |
| proportion | $\frac{1}{n}\sum_i \mathbb{1}[\text{condition}_i]$ | `(condition).mean()` |
| encodage one-hot de $y \in \{0, \dots, K-1\}$ | $\mathbf{e}_y$ : vecteur de taille $K$, 1 en position $y$, 0 ailleurs | `np.eye(K)[y]`, `mylearn.utils.one_hot` |
| règle du broadcasting | formes alignées à droite ; chaque paire de dimensions : égales, ou l'une vaut 1, ou l'une manque | `(333, 4)` et `(4,)` → `(333, 4)` |
| taille d'un array | $\text{size} = \prod_k \text{shape}_k$ | `a.size` |

### 0B · Maths du lycée au ML

**Nombres, notations, dénombrement (101.1)**

| Notion | Formule | En code |
|---|---|---|
| puissances | $a^m a^n = a^{m+n}$ ; $(a^m)^n = a^{mn}$ ; $a^{-n} = \frac{1}{a^n}$ ; $a^{1/n} = \sqrt[n]{a}$ | `a ** m`, `math.sqrt(x)` |
| ordres de grandeur | $2^{10} = 1024 \approx 10^3$ ; $2^{30} \approx 10^9$ | `4.7e7` $= 4{,}7 \times 10^7$ |
| valeur absolue, partie entière, signe | $\lvert x \rvert = \max(x, -x)$ ; $\lfloor x \rfloor$ : plus grand entier $\le x$ ; $\lceil x \rceil$ : plus petit entier $\ge x$ ; $\mathrm{sign}(x) \in \{-1, 0, 1\}$ | `abs(x)`, `math.floor(x)`, `math.ceil(x)`, `np.sign(x)` |
| distance et intervalle | $\lvert x - a \rvert \le r \iff a - r \le x \le a + r$ | |
| somme, produit | $\sum_{i=1}^{n} x_i = x_1 + \dots + x_n$ ; $\prod_{i=1}^{n} x_i = x_1 \times \dots \times x_n$ ; $\sum_{i=m}^{n}$ compte $n - m + 1$ termes | `sum(x)`, `math.prod(x)`, `range(m, n + 1)` |
| règles des sommes | $\sum_i (a x_i + b) = a \sum_i x_i + n\,b$ | |
| moyenne pondérée | $\bar{x}_w = \frac{\sum_i w_i x_i}{\sum_i w_i}$ | `np.average(x, weights=w)` |
| somme pondérée d'un neurone | $z = \sum_i w_i x_i + b = \mathbf{w} \cdot \mathbf{x} + b$ | `w @ x + b` |
| moyenne mobile d'ordre $k$ | $m_t = \frac{1}{k}\sum_{j=0}^{k-1} x_{t-j}$ ($n - k + 1$ valeurs) | `np.convolve(x, np.ones(k) / k, mode="valid")` |
| suite géométrique | $u_n = u_0\,q^n$ ; $\sum_{k=0}^{n-1} q^k = \frac{1 - q^n}{1 - q}$ ($q \neq 1$) ; $\sum_{k=0}^{\infty} q^k = \frac{1}{1 - q}$ si $\lvert q \rvert < 1$ | `q ** n` |
| ensembles | $\lvert A \cup B \rvert = \lvert A \rvert + \lvert B \rvert - \lvert A \cap B \rvert$ ; $\overline{A \cup B} = \bar{A} \cap \bar{B}$ | `A \| B`, `A & B`, `omega - A`, `len(A)` |
| dénombrement | choix successifs : $n_1 \times n_2 \times \dots$ ; $n! = n \times (n-1) \times \dots \times 1$ ; $\binom{n}{k} = \frac{n!}{k!\,(n-k)!}$ ; $\binom{K}{2} = \frac{K(K-1)}{2}$ ; $2^n$ sous-ensembles | `math.factorial(n)`, `math.comb(n, k)`, `math.perm(n, k)` |

**Fonctions usuelles (101.2)**

| Notion | Formule | En code |
|---|---|---|
| droite (fonction affine) | $f(x) = a\,x + b$ ; pente $a = \frac{y_B - y_A}{x_B - x_A}$ ; ordonnée à l'origine $b = f(0)$ | |
| parabole $ax^2 + bx + c$ | $\Delta = b^2 - 4ac$ ; racines $\frac{-b \pm \sqrt{\Delta}}{2a}$ ; sommet en $x = -\frac{b}{2a}$ | `np.roots([a, b, c])` |
| exponentielle | $e^{a+b} = e^a e^b$ ; $e^0 = 1$ ; $e \approx 2{,}718$ | `math.exp(x)`, `np.exp(x)` |
| logarithmes | $\ln(ab) = \ln a + \ln b$ ; $\ln(a^k) = k \ln a$ ; $\ln(e^x) = x$ ; $\log_b x = \frac{\ln x}{\ln b}$ | `np.log` ($\ln$), `np.log2`, `np.log10` |
| produits sous forme de logarithmes | $\ln \prod_i p_i = \sum_i \ln p_i$ ; moyenne géométrique $\left(\prod_i x_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_i \ln x_i\right)$ | `np.log(p).sum()`, `np.exp(np.log(x).mean())`, `np.log1p(u)` $= \ln(1 + u)$ |
| bits et nats | 1 nat $= \frac{1}{\ln 2} \approx 1{,}443$ bit | |
| sigmoïde | $\sigma(x) = \frac{1}{1 + e^{-x}}$ ; $\sigma(0) = \frac{1}{2}$ ; $\sigma(-x) = 1 - \sigma(x)$ ; inverse (logit) : $\ln\frac{p}{1 - p}$ | `1 / (1 + np.exp(-x))`, `scipy.special.expit` |
| tangente hyperbolique | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$ | `np.tanh(x)` |
| cosinus | $\pi$ rad $= 180°$ ; $\cos(-x) = \cos x$ ; $\cos(x + 2\pi) = \cos x$ | `math.cos`, `math.radians` |
| planning en cosinus | $\eta_t = \eta_{\max} \cdot \frac{1}{2}\left(1 + \cos\frac{\pi t}{T}\right)$ | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| composition | $(g \circ f)(x) = g(f(x))$ | `g(f(x))` |

**Vecteurs et matrices (101.3, 101.4)**

| Notion | Formule | En code |
|---|---|---|
| normes | $\lVert \mathbf{x} \rVert_2 = \sqrt{\sum_i x_i^2}$ ; $\lVert \mathbf{x} \rVert_1 = \sum_i \lvert x_i \rvert$ ; $\lVert \mathbf{x} \rVert_\infty = \max_i \lvert x_i \rvert$ | `np.linalg.norm(x, ord=2)` (1, `np.inf`) |
| distance euclidienne | $d(\mathbf{a}, \mathbf{b}) = \lVert \mathbf{a} - \mathbf{b} \rVert$ | `math.dist(a, b)` |
| produit scalaire | $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = \lVert \mathbf{a} \rVert \lVert \mathbf{b} \rVert \cos\theta$ | `a @ b`, `np.dot(a, b)` |
| similarité cosinus | $\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \lVert \mathbf{b} \rVert}$ | `mylearn.linalg_basics.cosine_similarity` |
| distance et produit scalaire | $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = \lVert \mathbf{a} \rVert^2 + \lVert \mathbf{b} \rVert^2 - 2\,\mathbf{a} \cdot \mathbf{b}$ ; vecteurs unitaires : $2 - 2\cos(\mathbf{a}, \mathbf{b})$ | |
| produit de Hadamard | $(\mathbf{a} \odot \mathbf{b})_i = a_i b_i$ ; $\sum_i (\mathbf{a} \odot \mathbf{b})_i = \mathbf{a} \cdot \mathbf{b}$ | `a * b` |
| produit extérieur | $(\mathbf{u}\mathbf{v}^\top)_{ij} = u_i v_j$, forme $(n, m)$ | `u[:, None] @ v[None, :]`, `np.outer(u, v)` |
| transposée | $(\mathbf{A}^\top)_{ij} = A_{ji}$ ; forme $(m, n) \to (n, m)$ | `A.T` |
| produit matriciel | $(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$ ; formes $(m, n) \times (n, p) \to (m, p)$ ; coût $m\,n\,p$ multiplications | `A @ B` |
| propriétés | $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top$ ; $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$ ; en général $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ ; $\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}$ | `np.eye(n)` |
| inverse $2 \times 2$ | $\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$ si $ad - bc \neq 0$ | `np.linalg.inv(M)` ; système : `np.linalg.solve(M, y)` |

**Dérivées et gradient (101.5, 101.6)**

| Notion | Formule | En code |
|---|---|---|
| dérivée | $f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$ ; tangente : $y = f(a) + f'(a)(x - a)$ | `(f(a + h) - f(a - h)) / (2 * h)` |
| dérivées usuelles | $(x^n)' = n x^{n-1}$ ; $(\sqrt{x})' = \frac{1}{2\sqrt{x}}$ ; $\left(\frac{1}{x}\right)' = -\frac{1}{x^2}$ ; $(e^x)' = e^x$ ; $(\ln x)' = \frac{1}{x}$ | |
| règles | $(u + v)' = u' + v'$ ; $(uv)' = u'v + uv'$ ; $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$ | |
| règle de la chaîne | $(g \circ f)'(x) = g'(f(x))\, f'(x)$, soit $\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}$ | |
| gradient | $\nabla f(x, y) = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)$ | |
| pas de descente | $\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$ | |
| somme sur les chemins | $z = f(u, v)$, $u = g(x)$, $v = h(x)$ ⟹ $\frac{dz}{dx} = \frac{\partial z}{\partial u}\frac{du}{dx} + \frac{\partial z}{\partial v}\frac{dv}{dx}$ | |
| la moyenne minimise les carrés | $\arg\min_a \sum_i (y_i - a)^2 = \bar{y}$, et le minimum vaut $n\,\mathrm{Var}(y)$ | |

**Probabilités (101.7)**

| Notion | Formule | En code |
|---|---|---|
| issues équiprobables | $P(A) = \frac{\lvert A \rvert}{\lvert \Omega \rvert}$ ; $P(\bar{A}) = 1 - P(A)$ ; $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | |
| indépendance | $P(A \cap B) = P(A)\,P(B)$ | |
| espérance | $\mathbb{E}[X] = \sum_k x_k\, p_k$ ; $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$ ; $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ | `values @ probs` |
| variance, écart-type | $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ ; $\mathrm{Var}(aX + b) = a^2 \mathrm{Var}(X)$ ; $\sigma = \sqrt{\mathrm{Var}(X)}$ | `x.var()`, `x.std()` |
| loi des grands nombres | la fréquence tend vers la probabilité, la moyenne vers l'espérance ; écart typique en $\frac{1}{\sqrt{n}}$ | `rng.integers(1, 7, size=n).mean()` |

## Partie I : fondations
### Ch. 1 · Introduction au machine learning et au deep learning

| Notion | Formule | En code |
|---|---|---|
| accuracy, taux d'erreur | $\text{accuracy} = \frac{\text{prédictions correctes}}{\text{nombre d'échantillons}}$ ; taux d'erreur $= 1 - \text{accuracy}$ | `np.mean(y_pred == y_true)`, `model.score(X, y)` |
| code de $k$ symboles, erreurs indépendantes | $P(\text{tout juste}) = \text{accuracy}^k$ | `acc ** k` |
| droite de régression (2 paramètres) | $\hat{y} = w\,x + b$ | `w * x + b` |
| erreur quadratique moyenne (loss) | $L = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2$ | `np.mean((y_hat - y) ** 2)` |
| règle de correction (1.16, justifiée aux ch. 5 et 19) | $w \leftarrow w + \eta\,(y - \hat{y})\,x$ ; $b \leftarrow b + \eta\,(y - \hat{y})$ | `error = y_i - (w * x_i + b)` |
| effet d'une correction sur l'erreur de l'échantillon | $e \leftarrow \left(1 - \eta\,(x^2 + 1)\right) e$ : l'erreur change de signe et grandit si $\eta\,(x^2 + 1) > 2$ | |
| interpolation linéaire au milieu | $\hat{y}\left(\frac{t_1 + t_2}{2}\right) = \frac{y_1 + y_2}{2}$ | |
| régression vers la moyenne (Galton) | $\hat{y} = \bar{y} + \frac{2}{3}\,(x - \bar{y})$ : écart de l'enfant $\approx \frac{2}{3}$ de l'écart mi-parental | |
| connexions entre deux couches denses | $n_{\text{entrée}} \times n_{\text{sortie}}$ poids, plus $n_{\text{sortie}}$ biais | `sum(w.size for w in mlp.coefs_)` |
| moyenne mobile (lissage, 0B) | $\tilde{x}_t = \frac{1}{k}\sum_{j=0}^{k-1} x_{t-j}$ ($k - 1$ valeurs manquantes au début) | `s.rolling(k).mean()` |
| position le long d'un axe (réduction de dimension) | $\mathbf{q} \cdot \mathbf{u}$ avec $\lVert \mathbf{u} \rVert = 1$ ; distance perdue $= \sqrt{\lVert \mathbf{q} \rVert^2 - (\mathbf{q} \cdot \mathbf{u})^2}$ | `q @ u` |
| pureté d'un clustering | $\frac{1}{n}\sum_{\text{groupes}} (\text{effectif de l'espèce majoritaire du groupe})$ | `pd.crosstab(g, y).max(axis=1).sum() / n` |

### Ch. 2 · Hasard et statistiques

| Notion | Formule | En code |
|---|---|---|
| moyenne | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | `np.mean(x)`, `mylearn.stats.mean` |
| médiane | valeur du milieu des données triées (moyenne des deux du milieu si $n$ est pair) | `np.median(x)` |
| variance, écart-type | $\mathrm{Var}(x) = \frac{1}{n - \mathrm{ddof}}\sum_i (x_i - \bar{x})^2$ ; $\sigma = \sqrt{\mathrm{Var}(x)}$ | `x.var(ddof=0)` (NumPy), `s.var()` (pandas : ddof = 1) |
| percentile $q$ (méthode linéaire) | position $\frac{q}{100}(n - 1)$ dans les données triées, interpolation entre les deux voisines | `np.percentile(x, q)` |
| z-score | $z_i = \frac{x_i - \bar{x}}{\sigma}$ | `(x - x.mean()) / x.std()`, `scipy.stats.zscore(x)` |
| densité | $P(a \le X \le b)$ = aire sous la densité entre $a$ et $b$ ; aire totale 1 | `np.histogram(x, density=True)` |
| loi uniforme sur $[a, b]$ | densité $\frac{1}{b - a}$ ; $P(c \le X \le d) = \frac{d - c}{b - a}$ ; moyenne $\frac{a + b}{2}$ | `rng.uniform(a, b)` |
| loi normale | $f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$ ; $P(\lvert X - \mu \rvert \le k\sigma) \approx$ 68 %, 95 %, 99,7 % pour $k = 1, 2, 3$ | `rng.normal(mu, sigma, size)` |
| loi de Bernoulli | $P(X = 1) = p$ ; $\mathbb{E}[X] = p$ ; $\mathrm{Var}(X) = p(1 - p)$ | `rng.random(n) < p` |
| espérance | $\mathbb{E}[X] = \sum_k x_k\,p_k$ | `np.dot(values, probs)` |
| tirage catégoriel (roue) | catégorie $= \min\{j : u < \sum_{i \le j} p_i\}$ avec $u$ uniforme sur $[0, 1)$ | `np.searchsorted(np.cumsum(p), u, side="right")` |
| élément absent d'un tirage de $n$ parmi $n$ avec remise | $\left(1 - \frac{1}{n}\right)^n \to e^{-1} \approx 0{,}368$ ; environ 63,2 % d'éléments distincts | |
| intervalle bootstrap percentile de niveau $c$ | percentiles $50(1 - c)$ et $50(1 + c)$ des statistiques de $B$ rééchantillons de taille $n$ | `scipy.stats.bootstrap(..., method="percentile")` |
| covariance | $\mathrm{Cov}(x, y) = \frac{1}{n - \mathrm{ddof}}\sum_i (x_i - \bar{x})(y_i - \bar{y})$ ; $\mathrm{Cov}(x, x) = \mathrm{Var}(x)$ | `np.cov(x, y, ddof=0)[0, 1]` (défaut de NumPy : $n - 1$) |
| corrélation de Pearson | $r = \frac{\mathrm{Cov}(x, y)}{\sigma_x\,\sigma_y} = \frac{\mathbf{d}_x \cdot \mathbf{d}_y}{\lVert \mathbf{d}_x \rVert\,\lVert \mathbf{d}_y \rVert} \in [-1, 1]$ (vecteurs d'écarts) | `np.corrcoef(x, y)[0, 1]` |
| changement d'unité | $\mathrm{Cov}(ax + b, cy + d) = ac\,\mathrm{Cov}(x, y)$ ; $r$ inchangé si $ac > 0$, de signe opposé si $ac < 0$ | |
| matrices | case $(j, k)$ : covariance (corrélation) des colonnes $j$ et $k$ ; diagonale : variances (des 1) | `np.cov(X, rowvar=False)`, `df.corr(numeric_only=True)` |

### Ch. 3 · Probabilités et mesure de la qualité

| Notion | Formule | En code |
|---|---|---|
| probabilité d'une région (point uniforme) | $P(A) = \frac{\text{aire}(A)}{\text{aire}(\text{mur})}$ ; estimée par la proportion d'impacts | `np.mean(inside)` |
| probabilité conditionnelle | $P(A \mid B) = \frac{P(A, B)}{P(B)}$, pour $P(B) > 0$ | `pd.crosstab(a, b, normalize="columns")` |
| règle du produit | $P(A, B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$ | |
| probabilités totales | $P(A) = \sum_b P(A \mid B = b)\,P(B = b) = \sum_b P(A, B = b)$ | `pd.crosstab(..., margins=True)` |
| indépendance | $P(A, B) = P(A)\,P(B)$, soit $P(A \mid B) = P(A)$ | |
| matrice de confusion (scikit-learn, labels 0/1) | `[[TN, FP], [FN, TP]]` : vérité en lignes, prédiction en colonnes | `confusion_matrix(y_true, y_pred)` |
| accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | `accuracy_score` |
| precision, recall | $\frac{TP}{TP + FP}$ ; $\frac{TP}{TP + FN}$ | `precision_score`, `recall_score` |
| spécificité, NPV | $\frac{TN}{TN + FP}$ ; $\frac{TN}{TN + FN}$ | `recall_score(..., pos_label=0)`, `precision_score(..., pos_label=0)` |
| FPR, FNR, FDR, FOR | $1 - \text{spécificité}$, $1 - \text{recall}$, $1 - \text{precision}$, $1 - \text{NPV}$ | |
| prévalence | $\frac{TP + FN}{n}$ | `np.mean(y_true == pos_label)` |
| F1, F-beta | $F_1 = \frac{2PR}{P + R} = \frac{2\,TP}{2\,TP + FP + FN}$ ; $F_\beta = \frac{(1 + \beta^2)\,TP}{(1 + \beta^2)\,TP + \beta^2 FN + FP}$ | `f1_score`, `fbeta_score(..., beta=2)` |
| moyenne harmonique | $H = \frac{2ab}{a + b}$ ; $\min(a, b) \le H \le \frac{a + b}{2}$ et $H \le 2\min(a, b)$ | |
| balanced accuracy | $\frac{\text{recall} + \text{spécificité}}{2}$ (binaire) ; moyenne des recalls par classe | `balanced_accuracy_score` |
| MCC | $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$ | `matthews_corrcoef` |
| moyennes sur plusieurs classes | macro : moyenne simple ; pondérée : poids = support ; micro : TP, FP, FN additionnés (micro = accuracy si un seul label) | `average="macro"`, `"weighted"`, `"micro"`, `None` |
| precision d'un dépistage | $\frac{\text{sens} \cdot p}{\text{sens} \cdot p + (1 - \text{spéc}) (1 - p)}$, $p$ = prévalence | |
| prévalence où la precision vaut 0,5 | $p = \frac{1 - \text{spéc}}{\text{sens} + 1 - \text{spéc}}$ (autant de vrais que de faux positifs) | |
| courbe ROC, AUC | points $(\text{FPR}(t), \text{TPR}(t))$ pour tous les seuils $t$ (positif si score $\ge t$) ; $\text{AUC} = P(s^+ > s^-)$, ex-æquo : $\frac{1}{2}$ | `roc_curve`, `roc_auc_score` |
| aire par les trapèzes | $\sum_i (x_{i+1} - x_i)\,\frac{y_i + y_{i+1}}{2}$ | `np.trapezoid(y, x)`, `sklearn.metrics.auc(x, y)` |
| average precision | $\text{AP} = \sum_j (R_j - R_{j-1})\,P_j$ (en escalier, sans interpolation ; $j$ parcourt les seuils du plus haut au plus bas, $R_0 = 0$) | `average_precision_score` |
| score de Brier | $\frac{1}{n}\sum_i (p_i - y_i)^2$ ; 0,25 pour une réponse toujours égale à 0,5 | `brier_score_loss` |
| calibration (diagramme de fiabilité) | par intervalle de probabilité : fréquence observée des positifs contre probabilité moyenne annoncée | `sklearn.calibration.calibration_curve` |

### Ch. 4 · Règle de Bayes

$H$ : une hypothèse ; $O$ : une observation ; $\theta$ : le biais d'une pièce ; $h$ faces et $t$ piles en $n = h + t$ lancers.

| Notion | Formule | En code |
|---|---|---|
| règle de Bayes | $P(H \mid O) = \frac{P(O \mid H)\,P(H)}{P(O)}$, pour $P(O) > 0$ ; posterior ∝ vraisemblance × prior | `mylearn.bayes.bayes_posterior(prior, likelihood)` |
| évidence | $P(O) = \sum_j P(O \mid H_j)\,P(H_j)$ | `np.dot(prior, likelihood)`, `mylearn.bayes.evidence` |
| cote, forme « cotes » | cote $= \frac{P(H)}{1 - P(H)}$, $P = \frac{\text{cote}}{1 + \text{cote}}$ ; $\frac{P(H_1 \mid O)}{P(H_2 \mid O)} = \frac{P(O \mid H_1)}{P(O \mid H_2)} \times \frac{P(H_1)}{P(H_2)}$ | |
| plusieurs observations, indépendantes sachant $H$ | $P(H_i \mid o_1, \ldots, o_n) = \frac{P(H_i)\prod_k P(o_k \mid H_i)}{\sum_j P(H_j)\prod_k P(o_k \mid H_j)}$ ; l'ordre ne compte pas | `mylearn.bayes.update_discrete(prior, table, observations)` |
| boucle posterior → prior | le posterior après $o_k$ est le prior de $o_{k+1}$ ; une cote est multipliée par un rapport de vraisemblance à chaque observation | `update_discrete(..., return_history=True)` |
| biais d'une pièce sur une grille | $P(\theta \mid \text{lancers}) \propto P(\theta)\,\theta^h (1 - \theta)^t$ | `mylearn.bayes.coin_bias_posterior(flips, grid)` |
| log-posterior | $\log P(\theta) + h \log\theta + t \log(1 - \theta) + \text{constante}$ (n'ajouter $h \log\theta$ que si $h > 0$, car $\theta^0 = 1$) | `np.log(prior) + h * np.log(grid) + ...` |
| log-sum-exp | $\log\sum_j e^{\ell_j} = m + \log\sum_j e^{\ell_j - m}$, $m = \max_j \ell_j$ ; normaliser : $p_i = \frac{e^{\ell_i - m}}{\sum_j e^{\ell_j - m}}$ | `scipy.special.logsumexp(l)`, `w = np.exp(l - l.max()); w / w.sum()` |
| limites du `float64` | plus petit nombre normal $\approx 2{,}2 \times 10^{-308}$ ; plus petit positif $\approx 4{,}9 \times 10^{-324}$ ; en dessous, le résultat vaut 0 | `np.finfo(float).tiny` |
| loi Beta | densité $\frac{\theta^{a-1}(1 - \theta)^{b-1}}{B(a, b)}$ sur $[0, 1]$ ; moyenne $\frac{a}{a + b}$ ; mode $\frac{a - 1}{a + b - 2}$ si $a, b > 1$ | `scipy.stats.beta(a, b)` : `pdf`, `cdf`, `sf`, `ppf`, `mean`, `std`, `interval` |
| posterior d'une pièce, prior uniforme | $\mathrm{Beta}(h + 1, t + 1)$ ; moyenne $\frac{h + 1}{n + 2}$ (règle de succession de Laplace) ; mode $\frac{h}{n}$ | `stats.beta(h + 1, t + 1)` |
| prior conjugué Beta | prior $\mathrm{Beta}(a, b)$, puis $h$ faces et $t$ piles : posterior $\mathrm{Beta}(a + h, b + t)$ | |
| MAP | $\hat\theta_{\text{MAP}} = \arg\max_\theta P(\theta \mid \text{données})$ ; prior uniforme : $\frac{h}{n}$ | `grid[np.argmax(posterior)]` |
| intervalle de crédibilité à queues égales, niveau $c$ | premières valeurs de la grille où la probabilité cumulée atteint $\frac{1 - c}{2}$, puis $\frac{1 + c}{2}$ | `mylearn.bayes.credible_interval(grid, posterior)`, `stats.beta(a, b).interval(c)` |
| probabilité au-delà d'un seuil | $P(\theta > x \mid \text{données})$ : aire à droite de $x$ sous la densité du posterior | `stats.beta(a, b).sf(x)` |
| moyenne et écart-type d'un posterior sur une grille | $m = \sum_\theta \theta\,P(\theta)$ ; $s = \sqrt{\sum_\theta (\theta - m)^2 P(\theta)}$ ; avec beaucoup de données, $s \approx \sqrt{\hat\theta(1 - \hat\theta)/n}$ | `m = np.sum(grid * post)` |

### Ch. 5 · Courbes et surfaces

$f$ : une fonction d'une variable ($f'$, $f''$) ou de plusieurs, $f(\mathbf{x})$ ; $h$ : le pas d'une différence finie ; $\eta$ : le learning rate ; $\mathbf{e}_i$ : 1 en position $i$, 0 ailleurs ; $\varepsilon \approx 2{,}2 \times 10^{-16}$.

| Notion | Formule | En code |
|---|---|---|
| dérivée (sécante symétrique), si $f$ est dérivable en $x$ | $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x - h)}{2h}$ ; en un point anguleux, cette limite peut exister sans que $f$ soit dérivable (0 pour $\lvert x \rvert$ en 0) | |
| différences finies | avant $\frac{f(x + h) - f(x)}{h}$, arrière $\frac{f(x) - f(x - h)}{h}$ : erreur en $h$ ; centrée $\frac{f(x + h) - f(x - h)}{2h}$ : erreur en $h^2$ | `mylearn.calculus.numerical_derivative(f, x, h=1e-5, method="central")` |
| dérivée seconde | $f''(x) \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}$ ; $f'(x^*) = 0$ et $f''(x^*) > 0$ : minimum local ; $< 0$ : maximum local ; $= 0$ : on ne conclut pas | `mylearn.calculus.second_derivative(f, x, h=1e-4)` |
| erreur totale d'une différence finie | centrée $\approx h^2 + \frac{\varepsilon}{h}$, meilleur pas vers $10^{-5}$ ; avant $\approx h + \frac{\varepsilon}{h}$, meilleur pas vers $10^{-8}$ (constantes omises) | `np.finfo(float).eps` |
| gradient | $\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)$ ; pointe vers la plus grande montée, de pente $\|\nabla f\|$ ; perpendiculaire aux lignes de niveau | |
| gradient numérique | $\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\,\mathbf{e}_i) - f(\mathbf{x} - h\,\mathbf{e}_i)}{2h}$ : $2n$ évaluations de $f$ | `mylearn.calculus.numerical_gradient(f, x)` (sur une copie `np.array(x, dtype=float)`) |
| pente dans la direction $\mathbf{u}$ ($\|\mathbf{u}\| = 1$) | $\nabla f(\mathbf{x}) \cdot \mathbf{u} = \|\nabla f\| \cos\theta$ | `grad @ u` |
| descente (montée) de gradient | $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\,\nabla f(\mathbf{x}_t)$ (montée : $+$) ; arrêt quand $\|\nabla f\| < $ `tol` | `mylearn.calculus.gradient_descent(grad, x0, lr, n_steps, tol, maximize)`, `torch.optim.SGD` |
| parabole de courbure $c$ | $x_{t+1} - x^* = (1 - \eta c)(x_t - x^*)$ : converge si et seulement si $0 < \eta < \frac{2}{c}$ ; en un pas si $\eta = \frac{1}{c}$ | |
| plusieurs courbures (bol, vallée) | le plus grand $\eta$ est fixé par la courbure la plus forte, la vitesse par la plus faible ; deux courbures $c_1 < c_2$ : meilleur $\eta = \frac{2}{c_1 + c_2}$ | |
| Rosenbrock | $f(x, y) = (a - x)^2 + b\,(y - x^2)^2$ ; $\nabla f = \big(-2(a - x) - 4bx(y - x^2),\ 2b(y - x^2)\big)$ ; minimum $(a, a^2)$ ; pour $a = 1$, $b = 100$, courbures au fond $\approx 1\,002$ et $0{,}4$ | `wb.synth.rosenbrock(x, y)`, `wb.synth.rosenbrock_grad(x, y)` |
| nature d'un point critique | dérivées secondes selon les axes $\mathbf{e}_i$ et les diagonales $\mathbf{e}_i \pm \mathbf{e}_j$ : toutes $> 0$ : minimum probable (une selle peut échapper à ces directions, variante de 5.24) ; toutes $< 0$ : maximum probable ; des deux signes : selle ; sinon on ne conclut pas ; seuls les signes des valeurs propres de la hessienne tranchent à coup sûr | `mylearn.calculus.classify_critical_point(f, x)`, `np.linalg.eigvalsh(H)` |
| extrema d'une courbe échantillonnée | $y_i$ strictement plus petit (grand) que ses `order` voisins de chaque côté | `mylearn.calculus.find_local_extrema(y, order)`, `scipy.signal.argrelextrema` |
| différentiation automatique | gradient exact aux arrondis près, pour quelques évaluations, quel que soit le nombre de paramètres | `p = torch.tensor(x, dtype=torch.float64, requires_grad=True)` ; `f(p).backward()` ; `p.grad` |

### Ch. 6 · Théorie de l'information

$p$ : la distribution des données ; $q$ : celle du code ou du modèle ; $n$ : le nombre d'issues ; $\ell_i$ : la longueur du mot de code de l'issue $i$ ; logarithmes en base 2 (bits) sauf mention contraire.

| Notion | Formule | En code |
|---|---|---|
| surprise (information) | $I(x) = -\log_2 P(x)$ ; 0 si $P(x) = 1$ ; $I(x, y) = I(x) + I(y)$ si $x$ et $y$ sont indépendants | `mylearn.info.self_information(p, base=2.0)` |
| unités | bits ($\log_2$), nats ($\ln$) : 1 nat $= \frac{1}{\ln 2} \approx 1{,}443$ bit ; $\log_2 x = \frac{\ln x}{\ln 2}$ | `np.log2`, `np.log`, `math.log2` |
| code de longueur fixe | $\lceil \log_2 N \rceil$ bits par symbole pour $N$ symboles | `math.ceil(math.log2(N))` |
| inégalité de Kraft | code préfixe binaire : $\sum_i 2^{-\ell_i} \le 1$ ; $= 1$ pour un code complet (Huffman) | `sum(2.0 ** -len(w) for w in code.values())` |
| longueur moyenne d'un code | $\bar{L} = \sum_i p_i\,\ell_i$ ; Huffman : $H(p) \le \bar{L} < H(p) + 1$ | `mylearn.info.huffman_code(symbols, probs)`, `huffman_encode`, `huffman_decode` |
| entropie | $H(p) = -\sum_i p_i \log_2 p_i$, avec $0 \log 0 = 0$ ; $0 \le H(p) \le \log_2 n$, maximum pour la loi uniforme | `mylearn.info.entropy(p)`, `scipy.stats.entropy(p, base=2)` |
| entropie d'une pièce | $h(p) = -p \log_2 p - (1 - p) \log_2 (1 - p)$, maximale en $p = \frac{1}{2}$ (1 bit) | |
| cross-entropy | $H(p, q) = -\sum_i p_i \log_2 q_i \ge H(p)$ ; infinie si $q_i = 0$ là où $p_i > 0$ | `mylearn.info.cross_entropy(p, q)` |
| divergence KL | $\mathrm{KL}(p \,\|\, q) = \sum_i p_i \log_2 \frac{p_i}{q_i} = H(p, q) - H(p) \ge 0$, nulle si et seulement si $q = p$ ; pas symétrique | `mylearn.info.kl_divergence(p, q)`, `scipy.stats.entropy(p, q, base=2)`, `scipy.special.rel_entr(p, q).sum()` (nats) |
| divergence de Jensen-Shannon | $\mathrm{JS}(p, q) = \frac{1}{2}\mathrm{KL}(p \,\|\, m) + \frac{1}{2}\mathrm{KL}(q \,\|\, m)$, $m = \frac{p + q}{2}$ ; $0 \le \mathrm{JS} \le 1$ bit | `mylearn.info.js_divergence(p, q)`, `scipy.spatial.distance.jensenshannon(p, q, base=2) ** 2` |
| distribution empirique, lissage de Laplace | $\hat{p}_i = \frac{n_i + \alpha}{n + \alpha V}$ ($V$ éléments, total $n$) | `mylearn.info.token_distribution(tokens, vocabulary, smoothing)`, `char_distribution(text, alphabet, lowercase, smoothing)` |
| log loss | $-\frac{1}{n}\sum_i \ln \hat{p}_{i, y_i}$ ; en binaire, $-\frac{1}{n}\sum_i \left[y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i)\right]$ ; probabilités coupées à $[\varepsilon ; 1 - \varepsilon]$ | `mylearn.info.log_loss(y, p)` (nats), `sklearn.metrics.log_loss`, `torch.nn.functional.cross_entropy(logits, y)` |
| perplexité | $\exp\left(-\frac{1}{n}\sum_t \ln q_t\right) = 2^{\text{cross-entropy en bits}}$ ; modèle uniforme sur $V$ tokens : $V$ | `mylearn.info.perplexity(token_probs)`, `torch.exp(loss)` |
| entropie conditionnelle | $H(X_t \mid X_{t-1}) = -\sum_{a, b} P(a, b) \log_2 P(b \mid a) \le H(X_t)$ | |
| surprise moyenne d'un modèle sur un texte | $\frac{1}{n}\sum_t -\log_2 q(x_t \mid \text{contexte})$ : une cross-entropy, à mesurer sur un texte non vu | `np.mean(mylearn.info.self_information(probs))` |
| compression | bits par caractère $= \frac{8 \times \text{octets comprimés}}{\text{nombre de caractères}}$ | `zlib.compress(text.encode("utf-8"), 9)` |

## Partie II : concepts
### Ch. 7 · Classification

$K$ : le nombre de classes ; $k$ : le nombre de clusters ; $n$ : le nombre d'échantillons ; $p$ : le nombre de features ; $d$ : la dimension de l'espace des features ($d = p$), dans les formules de géométrie ; $b$ : le nombre de cases par axe ; $\boldsymbol{\mu}_k$ : un centroïde ; $c_i$ : le cluster de l'échantillon $i$.

| Notion | Formule | En code |
|---|---|---|
| probabilité d'une classe (deux classes) | $P(1 \mid \mathbf{x}) = \frac{\pi f_1(\mathbf{x})}{\pi f_1(\mathbf{x}) + (1 - \pi) f_0(\mathbf{x})}$, avec $\pi$ l'a priori de la classe 1 et $f_1$, $f_0$ les densités des mesures | `scipy.stats.multivariate_normal(mean, cov).pdf(X)` |
| seuil de coût minimal | probabilités calibrées : positif si $p \ge t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ (à l'égalité, les deux décisions coûtent autant) | `y_pred = (p >= t).astype(int)` |
| nombre de classifieurs | $N_{\text{OvR}} = K$ ; $N_{\text{OvO}} = \binom{K}{2} = \frac{K(K-1)}{2}$ | `math.comb(K, 2)` |
| décision multi-classe | OvR : $\hat{y} = \arg\max_k s_k(\mathbf{x})$ ; OvO : $\hat{y} = \arg\max_k \text{votes}_k(\mathbf{x})$, avec une règle d'égalité | `mylearn.multiclass.OneVsRestClassifier(est)`, `OneVsOneClassifier(est)` ; `sklearn.multiclass` |
| distances au carré entre deux nuages | $\lVert \mathbf{a} - \mathbf{b} \rVert^2 = \lVert \mathbf{a} \rVert^2 - 2\,\mathbf{a} \cdot \mathbf{b} + \lVert \mathbf{b} \rVert^2$ ; négatifs d'arrondi remplacés par 0 | `mylearn.cluster.pairwise_sq_distances(A, B)`, `scipy.spatial.distance.cdist(A, B, "sqeuclidean")` |
| centroïde le plus proche | $\boldsymbol{\mu}_k = \frac{1}{\lvert C_k \rvert} \sum_{i \in C_k} \mathbf{x}_i$ ; $\hat{y}(\mathbf{x}) = \arg\min_k \lVert \mathbf{x} - \boldsymbol{\mu}_k \rVert^2$ | `mylearn.cluster.NearestCentroid()`, `sklearn.neighbors.NearestCentroid()` |
| frontière entre deux centroïdes | la médiatrice : $2(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) \cdot \mathbf{x} + \lVert \boldsymbol{\mu}_0 \rVert^2 - \lVert \boldsymbol{\mu}_1 \rVert^2 = 0$ | |
| inertie | $J = \sum_{i=1}^{n} \lVert \mathbf{x}_i - \boldsymbol{\mu}_{c_i} \rVert^2$ ; n'augmente jamais pendant Lloyd ; sa meilleure valeur possible baisse quand $k$ augmente | `KMeans(...).fit(X).inertia_` ; `score(X)` $= -J$ |
| algorithme de Lloyd | $c_i \leftarrow \arg\min_j \lVert \mathbf{x}_i - \boldsymbol{\mu}_j \rVert^2$, puis $\boldsymbol{\mu}_j \leftarrow$ moyenne des $\mathbf{x}_i$ tels que $c_i = j$ ; arrêt quand les affectations ne changent plus, ou $\sum_j \lVert \Delta \boldsymbol{\mu}_j \rVert^2 \le$ `tol` $\times$ variance moyenne des features | `mylearn.cluster.KMeans(n_clusters, init, n_init, max_iter, tol, random_state)`, `sklearn.cluster.KMeans` |
| k-means++ | 1ᵉʳ centre uniforme ; ensuite $P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}'} D(\mathbf{x}')^2}$ | `mylearn.cluster.kmeans_plusplus(X, k, rng)` ; `rng.choice(n, p=w / w.sum())` |
| silhouette | $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$ ; 0 pour un point seul ; score = moyenne | `mylearn.cluster.silhouette_samples(X, labels)`, `silhouette_score` ; `sklearn.metrics.silhouette_score` |
| pureté | $\frac{1}{n} \sum_{\text{clusters}} \max_{\text{classe}} n_{\text{cluster}, \text{classe}}$ | `pd.crosstab(labels, y).max(axis=1).sum() / len(y)` |
| densité d'échantillons | $\rho = \frac{n}{b^d}$ ; pour une densité $\rho$ : $n = \rho\, b^d$ | `n / b ** d` |
| volume de la boule de rayon 1 | $V_d = \frac{2\pi}{d} V_{d-2}$, $V_1 = 2$, $V_2 = \pi$ ($V_3 = \frac{4\pi}{3}$) ; boule de rayon $r$ : $V_d\, r^d$ | |
| boule dans le cube | $q_d = \frac{V_d}{2^d} = \frac{\pi}{2d}\,q_{d-2}$ : $q_2 = \frac{\pi}{4}$, $q_3 = \frac{\pi}{6}$, puis vers 0 | |
| peau d'une boule | part du volume entre $(1 - \varepsilon)\,r$ et $r$ : $1 - (1 - \varepsilon)^d$ | |
| hyper-orange | boîte de côté 4, ballons de rayon 1 dans les $2^d$ coins : $r(d) = \sqrt{d} - 1$ (1 en dimension 4, 2 en dimension 9) | `np.sqrt(d) - 1` |
| distances en grande dimension | cube unité : du centre à un coin $\frac{\sqrt{d}}{2}$ ; contraste $\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$ sans structure | `scipy.spatial.distance.pdist(X)` |

### Ch. 8 · Entraînement et test

$n$ : le nombre d'exemples ; $t$ : la part du test ; $k$ : le nombre de folds ; $n_c$ : l'effectif de la classe $c$ ; $\hat{p}$ : une accuracy mesurée ; $K$ : le nombre de réglages comparés ; $d_i$ : le désaccord entre deux modèles sur l'exemple $i$.

| Notion | Formule | En code |
|---|---|---|
| taille du test (hold-out) | $n_{\text{test}} = \lceil t \cdot n \rceil$, $n_{\text{train}} = n - n_{\text{test}}$ ; deux temps pour 60 / 20 / 20 : $t = 0{,}2$, puis $t = 0{,}25$ sur le reste | `mylearn.model_selection.train_test_split(X, y, test_size=0.2, rng=rng)` ; `sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)` |
| découpage stratifié | quota de la classe $c$ : $\lfloor n_c\, n_{\text{test}} / n \rfloor$, puis une unité de plus pour les plus grandes parties décimales (plus fort reste) | `train_test_split(..., stratify=y)` |
| tailles des folds | les $n \bmod k$ premiers folds ont $\lfloor n/k \rfloor + 1$ exemples, les autres $\lfloor n/k \rfloor$ ; au tour $j$, on entraîne sur $n - \lvert \text{fold}_j \rvert$ exemples | `mylearn.model_selection.kfold_indices(n, k)` ; `KFold(k)` ; `np.array_split(np.arange(n), k)` |
| k-fold stratifiée (règle de `mylearn`) | indices triés par classe (tri stable), puis le fold $i$ reçoit les positions $i, i + k, i + 2k, \dots$ : chaque classe a $\lfloor n_c / k \rfloor$ ou $\lfloor n_c / k \rfloor + 1$ membres par fold | `mylearn.model_selection.stratified_kfold_indices(y, k)` ; `StratifiedKFold(k)` (pour chaque classe, les mêmes effectifs par fold, mais pas dans les mêmes folds) |
| résumé d'une validation croisée | $\bar{s} = \frac{1}{k}\sum_j s_j$, $\sigma_s = \sqrt{\frac{1}{k}\sum_j (s_j - \bar{s})^2}$ ; $\sigma_s / \sqrt{k}$ sous-estime en général l'incertitude (folds dépendants) | `scores = mylearn.model_selection.cross_val_score(est, X, y, cv=5)` ; `scores.mean()`, `scores.std()` |
| nombre d'entraînements | grille de $G$ réglages : $G$ (validation fixe), $kG$ (k-fold), $kG + 1$ avec le réentraînement final ; imbriquée : $k_{\text{ext}}\,(k_{\text{int}}\,G + 1)$ | |
| erreur type d'une accuracy | $\mathrm{SE} = \sqrt{\hat{p}(1 - \hat{p})/n}$ ; intervalle à 95 % : $\hat{p} \pm 1{,}96\,\mathrm{SE}$ ; taille pour une demi-largeur $h$ : $n \ge (1{,}96 / h)^2\, \hat{p}(1 - \hat{p})$ | `math.sqrt(p * (1 - p) / n)` |
| meilleur de $K$ réglages indépendants | $P(\max_j S_j \ge s) = 1 - \big(1 - P(S \ge s)\big)^K$ ; $\mathbb{E}[\max]$ croît avec $K$ | `1 - (1 - q) ** K` |
| coefficient $R^2$ | $R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$ ($\bar{y}$ : moyenne des cibles notées) ; 1 parfait, 0 comme la moyenne, $< 0$ pire | `model.score(X, y)` (régresseurs) ; `sklearn.metrics.r2_score(y, y_pred)` |
| test par permutation (apparié) | $d_i = 1$ si seul B a raison, $-1$ si seul A, 0 sinon ; $D = \sum_i d_i$ ; p-valeur $= \frac{C + 1}{n_{\text{perm}} + 1}$, $C$ = nombre de tirages de signes avec $\lvert \sum_i \pm d_i \rvert \ge \lvert D \rvert$ | `np.where(rng.random((n_perm, n)) < 0.5, -d, d).sum(axis=1)` |
| test exact de McNemar | $b$ désaccords gagnés par A, $c$ par B : p-valeur bilatérale $= \min\big(1,\ 2\, P(X \le \min(b, c))\big)$, $X \sim \mathcal{B}(b + c, \frac{1}{2})$ | `scipy.stats.binomtest(min(b, c), b + c, 0.5).pvalue` |
| bootstrap apparié | rééchantillonner les exemples du test avec remise, en gardant les réponses de A et de B ; percentiles 2,5 et 97,5 de l'écart d'accuracy | `rows = rng.integers(0, n, (n_boot, n))` ; `np.percentile(d[rows].mean(axis=1), [2.5, 97.5])` |
| clonage d'un estimateur | les attributs qui ne commencent ni ne finissent par `_` sont les hyperparamètres | `type(est)(**{k: copy.deepcopy(v) for k, v in vars(est).items() if not k.startswith("_") and not k.endswith("_")})` ; `sklearn.base.clone(est)` |
| données dépendantes | groupes : un groupe entier par fold ; séries temporelles : entraînement toujours avant la validation | `GroupKFold(k).split(X, y, groups)`, `StratifiedGroupKFold` ; `TimeSeriesSplit(k, gap=...)` |

### Ch. 9 · Overfitting et underfitting

$n$ : le nombre d'exemples ; $p$ : le nombre de features ; $\mathbf{X}_c$, $\mathbf{y}_c$ : les données centrées (moyennes retirées) ; $\mathbf{1}$ : le vecteur de 1 ; $\lambda$ : la force de la régularisation (`alpha` dans scikit-learn) ; $f$ : la courbe idéale, sans bruit ; $\hat{f}_D$ : le modèle entraîné sur le dataset $D$ ($D_1, \dots, D_M$ : $M$ datasets tirés de la même source) ; $\bar{f}$ : le modèle moyen ; $\sigma^2$ : la variance du bruit.

| Notion | Formule | En code |
|---|---|---|
| mesures d'erreur d'une régression | $\mathrm{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$ ; $\mathrm{RMSE} = \sqrt{\mathrm{MSE}}$ (dans l'unité de la cible) ; $\mathrm{MAE} = \frac{1}{n}\sum_i \lvert y_i - \hat{y}_i \rvert$ (moins sensible aux points aberrants) | `mylearn.linear.mean_squared_error(y, y_pred)`, `mean_absolute_error` ; `sklearn.metrics.mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error` |
| coefficient $R^2$ | $R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ (ch. 8) ; si $SS_{\text{tot}} = 0$ : 1 quand $SS_{\text{res}} = 0$, sinon 0 (convention de scikit-learn) | `mylearn.linear.r2_score(y, y_pred)` ; `model.score(X, y)` |
| features polynomiales | tous les monômes de degré total $1$ à $d$ ; leur nombre : $\binom{p + d}{d} - 1$ (9 pour $p = 3$, $d = 2$) ; à standardiser avant une pénalité | `mylearn.linear.polynomial_features(X, degree=d)` ; `PolynomialFeatures(d, include_bias=False).fit_transform(X)` |
| droite des moindres carrés | $a = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2}$, $b = \bar{y} - a\,\bar{x}$ ; la somme des résidus est nulle, la droite passe par $(\bar{x}, \bar{y})$ | `a, b = np.polyfit(x, y, 1)` |
| moindres carrés (cas général) | équations normales $\mathbf{X}_c^\top \mathbf{X}_c\,\hat{\mathbf{w}} = \mathbf{X}_c^\top \mathbf{y}_c$, puis $b = \bar{y} - \bar{\mathbf{x}} \cdot \hat{\mathbf{w}}$ | `np.linalg.lstsq(Xc, yc, rcond=None)[0]` ; `mylearn.linear.LinearRegression()` ; `sklearn.linear_model.LinearRegression()` |
| Ridge (L2) | minimise $\lVert \mathbf{y} - \mathbf{X}\mathbf{w} - b\,\mathbf{1} \rVert^2 + \lambda \lVert \mathbf{w} \rVert^2$ ($b$ non pénalisé) : $\hat{\mathbf{w}} = (\mathbf{X}_c^\top \mathbf{X}_c + \lambda \mathbf{I})^{-1} \mathbf{X}_c^\top \mathbf{y}_c$ ; en dimension 1, sans ordonnée : $w^* = \frac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda}$ (jamais exactement 0) ; avec une ordonnée non pénalisée : $w^* = \frac{S_{xy}}{S_{xx} + \lambda}$, $S_{xx} = \sum_i (x_i - \bar{x})^2$, $S_{xy} = \sum_i (x_i - \bar{x})(y_i - \bar{y})$ (CP2.6) | `np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), Xc.T @ yc)` ; `mylearn.linear.Ridge(alpha=lam)` ; `sklearn.linear_model.Ridge(alpha=lam)` |
| seuillage doux | $S(z, \gamma) = \arg\min_w \tfrac{1}{2}(w - z)^2 + \gamma \lvert w \rvert = \operatorname{signe}(z)\max(\lvert z \rvert - \gamma, 0)$ : exactement 0 si $\lvert z \rvert \le \gamma$ | `np.sign(z) * np.maximum(np.abs(z) - gamma, 0)` ; `mylearn.linear.soft_threshold(z, gamma)` |
| Lasso (L1) | minimise $\frac{1}{2n}\lVert \mathbf{y} - \mathbf{X}\mathbf{w} - b\,\mathbf{1} \rVert^2 + \lambda \lVert \mathbf{w} \rVert_1$ ; descente de coordonnées : $w_j \leftarrow S(\rho_j, \lambda)/z_j$, $\rho_j = \frac{1}{n}\mathbf{x}_j^\top \mathbf{r}_j$ ($\mathbf{r}_j$ : résidu sans la feature $j$), $z_j = \frac{1}{n}\lVert \mathbf{x}_j \rVert^2$ ; tous les poids nuls si $\lambda \ge \max_j \lvert \mathbf{x}_{c,j}^\top \mathbf{y}_c \rvert / n$ | `mylearn.linear.Lasso(alpha=lam, max_iter=1000, tol=1e-4)` ; `sklearn.linear_model.Lasso(alpha=lam)`, `ElasticNet(alpha=lam, l1_ratio=0.5)` |
| early stopping avec patience | amélioration si $L_{\text{val}} < L_{\text{meilleure}} - \delta_{\min}$ (stricte) ; arrêt quand `patience` epochs consécutives n'améliorent pas ; on recharge les poids de la dernière amélioration (la meilleure epoch quand $\delta_{\min} = 0$) | Lightning : `EarlyStopping(monitor="val_loss", patience=P, min_delta=d)` ; scikit-learn : `MLPRegressor(early_stopping=True, n_iter_no_change=P)` |
| biais² et variance d'une famille | $\bar{f}(x) = \frac{1}{M}\sum_m \hat{f}_{D_m}(x)$ ; $\text{biais}^2 = \operatorname{moy}_x \big(\bar{f}(x) - f(x)\big)^2$ ; $\text{variance} = \operatorname{moy}_x \frac{1}{M}\sum_m \big(\hat{f}_{D_m}(x) - \bar{f}(x)\big)^2$ (ddof = 0) | `mylearn.linear.bias_variance_decomposition(P, f)` ; `((P.mean(axis=0) - f) ** 2).mean()`, `P.var(axis=0).mean()` |
| décomposition de l'erreur | $\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \text{biais}^2 + \text{variance} + \sigma^2$ ; sans bruit, l'erreur moyenne des $M$ modèles face à $f$ vaut exactement biais² + variance | `((P - f) ** 2).mean()` |
| solution de norme minimale | pour $p > n$ : $\hat{\mathbf{w}} = \mathbf{X}^{+}\mathbf{y}$, la plus courte des solutions qui passent par tous les points ; pic de l'erreur de test vers $p \approx n$ (double descente) | `np.linalg.pinv(X) @ y` ; `np.linalg.lstsq(X, y, rcond=None)[0]` |
| posterior d'une droite sur une grille | $\log p(a, b \mid \mathcal{D}) = -\frac{a^2 + b^2}{2\tau^2} - \sum_i \frac{(y_i - a x_i - b)^2}{2\sigma^2} + C$ ; retrancher le maximum, exponentielle, normaliser ; point par point = tout d'un coup | `mylearn.linear.bayes_line_posterior(x, y, slopes, intercepts, noise_std, prior_std)` (lignes : ordonnées ; colonnes : pentes) ; `sklearn.linear_model.BayesianRidge` |
| MAP et Ridge | prior $\mathcal{N}(0, \tau^2)$ sur les paramètres, bruit $\mathcal{N}(0, \sigma^2)$ : la droite MAP minimise $\sum_i (y_i - a x_i - b)^2 + \frac{\sigma^2}{\tau^2}(a^2 + b^2)$, une Ridge avec $\lambda = \sigma^2/\tau^2$ (ordonnée pénalisée aussi) | |

### Ch. 10 · Neurones

$\mathbf{x}$ : les entrées d'un neurone ; $\mathbf{w}$ : ses poids ; $b$ : son biais ; $z$ : la somme pondérée ; $f$ : la fonction d'activation ; $y_i \in \{-1, +1\}$ : les labels du perceptron ; $\eta$ : le learning rate (*pas d'apprentissage*, `eta0`) ; $R$ : la plus grande norme des exemples ; $\gamma$ : la marge ; $\mathbf{X}$ : un batch (une ligne par exemple) ; $\mathbf{W}$ : les poids d'une couche.

| Notion | Formule | En code |
|---|---|---|
| perceptron | $z = \sum_j w_j x_j = \mathbf{w}\cdot\mathbf{x}$ ; $\hat{y} = +1$ si $z > 0$, $-1$ sinon ($z = 0$ donne $-1$) ; version 0/1 : 1 si $z > 0$ | `mylearn.perceptron.sign_step(z)` ; `np.where(z > 0, 1.0, -1.0)` (pas `np.sign`, qui donne 0 en 0) |
| neurone moderne | $a = f(\mathbf{w}\cdot\mathbf{x} + b)$ | `mylearn.perceptron.neuron_forward(X, w, b, activation)` ; `torch.nn.functional.linear(X, w[None, :], b)` |
| astuce du biais | $\tilde{\mathbf{x}} = (1, x_1, \dots, x_p)$, $\tilde{\mathbf{w}} = (b, w_1, \dots, w_p)$ : $z = \tilde{\mathbf{w}}\cdot\tilde{\mathbf{x}}$ ; pour un batch, une colonne de 1 en tête | `mylearn.perceptron.add_bias_column(X)` ; `np.hstack([np.ones((n, 1)), X])` |
| frontière de décision | l'hyperplan $\mathbf{w}\cdot\mathbf{x} + b = 0$ ; $\mathbf{w}$ lui est perpendiculaire et pointe vers le côté $+1$ ; sans biais, il passe par l'origine | |
| portes logiques (entrées 0/1, sortie 1 si $z > 0$) | AND : $\mathbf{w} = (1, 1)$, $b = -1{,}5$ ; OR : $b = -0{,}5$ ; NAND, NOR : signes changés ; XOR : aucun perceptron, deux couches (OR et NAND, puis AND) | `wb.synth.logic_gate("xor")` |
| règle d'apprentissage | départ $\mathbf{w} = \mathbf{0}$, $b = 0$ ; si $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) \le 0$ : $\mathbf{w} \leftarrow \mathbf{w} + \eta\,y_i\,\mathbf{x}_i$, $b \leftarrow b + \eta\,y_i$ ; arrêt après une epoch sans correction | `mylearn.perceptron.Perceptron(eta0, max_iter).fit(X, y)` ; `sklearn.linear_model.Perceptron(shuffle=False, tol=None)` |
| théorème de convergence (Novikoff) | si $\lVert \mathbf{x}_i \rVert \le R$ et $y_i\,\mathbf{u}\cdot\mathbf{x}_i \ge \gamma$ ($\lVert \mathbf{u} \rVert = 1$) : la règle sans biais, partie de zéro, fait au plus $(R/\gamma)^2$ corrections (avec un biais : vecteurs $(1, \mathbf{x}_i)$) ; preuve : $\mathbf{u}\cdot\mathbf{w}_k \ge k\gamma$, $\lVert \mathbf{w}_k \rVert^2 \le kR^2$, Cauchy-Schwarz | `sum(model.errors_)` |
| marge d'un séparateur | $\gamma = \min_i y_i\, \mathbf{u}\cdot\mathbf{x}_i$, avec $\mathbf{u}$ de norme 1 | `np.min(y_pm * (X @ (u / np.linalg.norm(u))))` |
| perceptron moyenné | renvoyer $\bar{\mathbf{w}} = \frac{1}{T}\sum_t \mathbf{w}_t$, la moyenne des poids après chacun des $T$ exemples vus | `w_sum += w` après chaque exemple |
| une couche de neurones | $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$, $\mathbf{W}$ de forme $(n_{\text{in}}, n_{\text{out}})$, $W_{jk}$ : de l'entrée $j$ vers le neurone $k$ (la convention AD) ; $\mathbf{A} = f(\mathbf{Z})$ | `X @ W + b` ; PyTorch : `nn.Linear(n_in, n_out)`, `weight` de forme `(n_out, n_in)`, `x @ weight.T + bias` |
| couches sans activation | $\mathbf{W}_2(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2)$ : encore affine | |

### Ch. 11 · Apprentissage et raisonnement

$S$, $M$, $P$ : le sujet, le moyen terme et le prédicat d'un syllogisme ; $K$ : le nombre de bras d'un bandit ; $q_*(a)$ : la vraie valeur (l'espérance de la récompense) du bras $a$, $q_* = \max_a q_*(a)$ ; $A_t$, $R_t$ : le bras joué et la récompense au pas $t$ ; $\alpha$ : le pas constant d'une estimation (sans rapport avec l'`alpha` de Ridge, ch. 9) ; $Q_t(a)$, $N_t(a)$ : l'estimation et le nombre de tirages de $a$ avant le pas $t$ ; $\varepsilon$ : la probabilité d'explorer ; $c$ : la force de l'exploration d'UCB ; $s_a$, $f_a$ : les succès et les échecs d'un bras de Bernoulli.

| Notion | Formule | En code |
|---|---|---|
| compter une représentation | $n$ bits : $2^n$ valeurs (0 à $2^n - 1$ sans signe, $-2^{n-1}$ à $2^{n-1} - 1$ en complément à deux) ; $2^{2^n}$ fonctions booléennes de $n$ entrées ; fonctions à seuil (perceptron) : 14 sur 16 pour 2 entrées | |
| propositions catégoriques | A « tout $S$ est $P$ » (distribue $S$) ; E « aucun $S$ n'est $P$ » (les deux) ; I « quelque $S$ est $P$ » (aucun) ; O « quelque $S$ n'est pas $P$ » (distribue $P$) | |
| règles d'un syllogisme valide | moyen terme distribué au moins une fois ; un terme distribué dans la conclusion l'est dans sa prémisse ; pas deux prémisses négatives ; conclusion négative si et seulement si une prémisse l'est | |
| validité par force brute | 8 régions de Venn, $2^8 = 256$ mondes (régions vides ou non) ; valide si aucun monde ne rend les prémisses vraies et la conclusion fausse ; 15 formes valides sur 256 (24 si $S$, $M$, $P$ ont chacun un membre) | `itertools.product([False, True], repeat=8)` (11.15) |
| raisonnement conditionnel | valides : *modus ponens* ($X \Rightarrow Y$, $X$, donc $Y$), *modus tollens* ($X \Rightarrow Y$, non $Y$, donc non $X$) ; sophismes : affirmer le conséquent, nier l'antécédent | |
| généralisation et prédiction | $\hat{p} = h/n$ ; erreur type $\sqrt{\hat{p}(1-\hat{p})/n}$ ; prédiction de Laplace $(h+1)/(n+2)$ ; $n \ge p(1-p)/\mathrm{SE}^2$ pour une erreur type visée | `math.sqrt(p * (1 - p) / n)` |
| capture proportionnelle à une taille $m$ | espérance de la taille capturée : $\sum_i m_i^2 / \sum_i m_i$ (biais de sélection, qu'aucune taille d'échantillon ne corrige) | `np.sum(m * m / m.sum())` (11.17) |
| moyenne incrémentale | $Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)$ ; pas constant : $Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} R_i$ (oubli exponentiel) | `mylearn.bandit.incremental_update(q, r, step)` |
| ε-greedy | meilleur bras avec la probabilité $1 - \varepsilon + \varepsilon/K$ (estimations justes) ; regret par pas $\approx \varepsilon \cdot \frac{1}{K}\sum_a (q_* - q_*(a))$ : regret linéaire | `mylearn.bandit.epsilon_greedy_action(q, eps, rng)` ; `argmax_random_tie(q, rng)` |
| UCB | bras jamais tiré d'abord, puis $\arg\max_a Q_t(a) + c\sqrt{\ln t / N_t(a)}$ ; UCB1 (récompenses dans $[0, 1]$) : $c = \sqrt{2}$, regret en $O(\ln T)$ | `mylearn.bandit.ucb_action(q, n, t, c)` |
| échantillonnage de Thompson | posterior $\mathrm{Beta}(1 + s_a, 1 + f_a)$ ; tirer $\theta_a$ dans chacun, jouer $\arg\max_a \theta_a$ ; chaque bras est joué avec la probabilité qu'il soit le meilleur | `mylearn.bandit.thompson_action(s, f, rng)` ; `rng.beta(1 + s, 1 + f)` |
| regret | pseudo-regret $\sum_{t=1}^T (q_* - q_*(A_t))$, jamais décroissant ; regret réalisé $T q_* - \sum_t R_t$ ; agent au hasard : $T\,(q_* - \frac{1}{K}\sum_a q_*(a))$ | `np.cumsum(best_mean - means[actions])` ; `mylearn.bandit.run_bandit(...)["regret"]` |

### Checkpoints I et II, mini-projets MP1 et MP2

| Notion | Formule | En code |
|---|---|---|
| posterior avec une température (MP1) | $P(c \mid \text{texte}) \propto P(c)\, e^{\ell_c / T}$, avec $\ell_c = \ln P(\text{texte} \mid c)$ ; on retire d'abord $\max_c \ell_c / T$ (log-sum-exp) ; $T > 1$ adoucit les probabilités, $T < 1$ les durcit, sans changer le classement quand le prior est uniforme | `langid.posterior_from_loglik(log_lik, prior, temperature)` |
| règle de trois (MP1) | 0 erreur sur $n$ essais indépendants : taux d'erreur inférieur à $3/n$ avec 95 % de confiance | |
| règle d'une erreur type (MP2) | garder le modèle le plus simple dont la moyenne ne dépasse pas $\bar{s}_{\text{meilleur}} + \sigma_{\text{meilleur}}/\sqrt{k}$ (RMSE : plus petit est meilleur ; $\sigma/\sqrt{k}$ est optimiste, les folds n'étant pas indépendants) | `pick_simplest(mean_rmse, std_rmse, n_folds)` (notebook de MP2, étape MP2.3) |
| Ridge en dimension 1 avec ordonnée (CP2.6) | $w^* = \frac{S_{xy}}{S_{xx} + \lambda}$, $b^* = \bar{y} - w^* \bar{x}$ ; $S_{xx} = \sum_i (x_i - \bar{x})^2$, $S_{xy} = \sum_i (x_i - \bar{x})(y_i - \bar{y})$ | |

## Partie III : ML classique
### Ch. 12 · Préparation des données

$\mathbf{X}$ : les données, de forme $(n, p)$ ; pour la feature $j$ de l'**entraînement**, $\min_j$, $\max_j$, $\mu_j$ et $\sigma_j$ (ddof = 0) ; $Q_1$, $Q_3$ : les quartiles ; $\mathbf{X}_c = \mathbf{X} - \boldsymbol{\mu}$ : les données centrées ; $\boldsymbol{\Sigma}$ : la matrice de covariance (ddof = 1) ; $\mathbf{u}$ : un vecteur unitaire ; $\lambda_j$ : la variance le long de la composante $j$ ; $\mathbf{W}_k$ : les $k$ premières composantes, une par ligne ; $\mathbf{Z}$ : les coordonnées.

| Notion | Formule | En code |
|---|---|---|
| règle d'or | `fit` sur l'entraînement seul ; `transform` avec les **mêmes** paramètres sur la validation, le test et la production ; en validation croisée, `fit` dans chaque fold | `scaler.fit(X_train)` ; `make_pipeline(StandardScaler(), model)` dans `cross_val_score` |
| normalisation min-max | $x' = a + (b - a)\,\frac{x - \min_j}{\max_j - \min_j}$ ; inverse $x = \min_j + \frac{x' - a}{b - a}(\max_j - \min_j)$ ; étendue nulle : on divise par 1 | `mylearn.preprocessing.MinMaxScaler(feature_range=(a, b), clip=False)` ; `sklearn.preprocessing.MinMaxScaler` |
| standardisation | $z = \frac{x - \mu_j}{\sigma_j}$, $\sigma_j$ avec ddof = 0 ; inverse $x = z\,\sigma_j + \mu_j$ ; écart-type nul : on divise par 1 (colonne constante repérée par $\max_j = \min_j$) | `mylearn.preprocessing.StandardScaler()` ; `sklearn.preprocessing.StandardScaler` ; pandas `s.std()` divise par $n - 1$ |
| mise à l'échelle robuste | $\frac{x - \text{médiane}_j}{Q_{3,j} - Q_{1,j}}$ | `RobustScaler()` |
| règle de Tukey | point aberrant hors de $[Q_1 - 1{,}5\,\mathrm{IQR} ;\ Q_3 + 1{,}5\,\mathrm{IQR}]$, $\mathrm{IQR} = Q_3 - Q_1$ | `q1, q3 = np.percentile(x, [25, 75])` |
| imputation simple | chaque NaN de la colonne $j$ remplacé par une statistique de ses valeurs **observées** d'entraînement (moyenne, médiane, valeur la plus fréquente, constante) | `mylearn.preprocessing.SimpleImputer(strategy="median")` ; `SimpleImputer(add_indicator=True)` |
| one-hot | $K$ catégories → $K$ colonnes 0/1 ($K - 1$ avec `drop="first"`) ; distance entre deux catégories : $\sqrt{2}$ ; catégorie inconnue avec `handle_unknown="ignore"` : un bloc de zéros | `mylearn.preprocessing.OneHotEncoder(drop=None, handle_unknown="error")` ; `OneHotEncoder(sparse_output=False)` |
| encodage ordinal | code = position dans `categories_[j]` (ordre de tri de Python par défaut, ou la liste donnée) | `OrdinalEncoder(categories=[["faible", "moyen", "fort"]])` |
| matrice de covariance | $\boldsymbol{\Sigma} = \frac{1}{n - 1}\,\mathbf{X}_c^\top \mathbf{X}_c$ | `np.cov(X, rowvar=False)` |
| projection sur une droite | coordonnée $t = \mathbf{x}\cdot\mathbf{u}$, projection $t\,\mathbf{u}$ ; pour un nuage centré, $\mathbf{t} = \mathbf{X}_c\,\mathbf{u}$, de variance $\mathbf{u}^\top \boldsymbol{\Sigma}\,\mathbf{u}$ | `Xc @ u` |
| vecteur propre, valeur propre | $\boldsymbol{\Sigma}\,\mathbf{u} = \lambda\,\mathbf{u}$ ; la direction de variance maximale est le vecteur propre de la plus grande valeur propre ; la somme des $\lambda_j$ est la variance totale, $\operatorname{tr}\boldsymbol{\Sigma}$ | `np.linalg.eigh(Sigma)` (valeurs propres croissantes) |
| PCA par SVD | $\mathbf{X}_c = \mathbf{U}\,\mathbf{S}\,\mathbf{V}^\top$ ; composantes : lignes de $\mathbf{V}^\top$ ; $\lambda_j = s_j^2/(n - 1)$ ; signe arbitraire (scikit-learn : la plus grande coordonnée en valeur absolue positive) | `np.linalg.svd(Xc, full_matrices=False)` ; `mylearn.preprocessing.PCA(n_components=k)` ; `sklearn.decomposition.PCA` |
| projeter et reconstruire | $\mathbf{Z} = (\mathbf{X} - \boldsymbol{\mu})\,\mathbf{W}_k^\top$ ; $\hat{\mathbf{X}} = \mathbf{Z}\,\mathbf{W}_k + \boldsymbol{\mu}$ ; erreur de reconstruction moyenne $= \frac{n - 1}{n}\sum_{j > k}\lambda_j$ | `pca.transform(X)`, `pca.inverse_transform(Z)` |
| part de variance expliquée | $\lambda_j / \sum_l \lambda_l$ ; on garde le plus petit $k$ dont la part cumulée dépasse **strictement** un seuil | `pca.explained_variance_ratio_` ; `PCA(n_components=0.9)` |
| whitening | $z_j / \sqrt{\lambda_j}$ : coordonnées non corrélées et de variance 1 | `PCA(whiten=True)` |

### Ch. 13 · Classifieurs

$\mathbf{X}$ : les données d'entraînement, de forme $(n, p)$ ; classes $0, \dots, K - 1$ ; $p_c$ : la proportion de la classe $c$ dans un nœud ; $\sigma(z) = 1/(1 + e^{-z})$ ; pour un SVM, $y = \pm 1$ et $f(\mathbf{x}) = \mathbf{w}\cdot\mathbf{x} + b$.

| Notion | Formule | En code |
|---|---|---|
| distances | euclidienne $\sqrt{\sum_j (x_j - z_j)^2}$ ; Manhattan $\sum_j \lvert x_j - z_j \rvert$ ; par broadcasting `A[:, None, :] - B[None, :, :]` | `mylearn.neighbors.pairwise_distances(A, B, metric="euclidean")` ; `sklearn.metrics.pairwise_distances` ; `scipy.spatial.distance.cdist` |
| kNN | les $k$ plus proches (tri stable) votent ; avec $V$ les $k$ voisins et $V_c$ ceux de la classe $c$, part de la classe $c$ : $\frac{\sum_{i \in V_c} w_i}{\sum_{i \in V} w_i}$, $w_i = 1$ ou $1/d_i$ ; égalité : la première classe | `mylearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform")` ; `sklearn.neighbors.KNeighborsClassifier` |
| mémoire et temps d'un kNN | $n \times p$ nombres (8 octets en `float64`) ; une requête : environ $3np$ opérations, plus le tri | `X.nbytes` ; `time.perf_counter()`, `timeit.repeat` |
| régression logistique | $\hat p = \sigma(\mathbf{w}\cdot\mathbf{x} + b)$ ; $L = -\frac1n \sum_i [y_i \log\hat p_i + (1 - y_i)\log(1 - \hat p_i)] + \frac{\alpha}{2}\lVert\mathbf{w}\rVert^2$ ; $\nabla_{\mathbf{w}} L = \frac1n \mathbf{X}^\top(\hat{\mathbf{p}} - \mathbf{y}) + \alpha\mathbf{w}$, $\partial L/\partial b = \frac1n\sum_i(\hat p_i - y_i)$ ; $\sigma' = \sigma(1 - \sigma)$ | `mylearn.logistic.LogisticRegression(alpha=0.0, learning_rate=0.1)` ; `sklearn.linear_model.LogisticRegression(C=1/(alpha*n))` (pénalité L2 par défaut, `C=1.0`) |
| distance à un hyperplan | $\lvert \mathbf{w}\cdot\mathbf{x} + b \rvert / \lVert \mathbf{w} \rVert$ ; le signe de $f$ donne le côté | `abs(X @ w + b) / np.linalg.norm(w)` |
| SVM à marge dure | $\min \frac12\lVert\mathbf{w}\rVert^2$ sous $y_i f(\mathbf{x}_i) \ge 1$ ; largeur de la bande $2/\lVert\mathbf{w}\rVert$ ; vecteurs de support : $y_i f(\mathbf{x}_i) = 1$ | `SVC(kernel="linear", C=1e6)` ; `svm.support_vectors_`, `svm.coef_`, `svm.intercept_` |
| marge souple | $\min \frac12\lVert\mathbf{w}\rVert^2 + C\sum_i \max(0, 1 - y_i f(\mathbf{x}_i))$ (hinge loss) ; grand C : bande étroite | `SVC(kernel="linear", C=1.0)` ; `LinearSVC` |
| noyaux | $K(\mathbf{x}, \mathbf{z}) = \varphi(\mathbf{x})\cdot\varphi(\mathbf{z})$ ; polynomial $(\mathbf{x}\cdot\mathbf{z})^2$, $\varphi(\mathbf{x}) = (x_1^2, \sqrt2\,x_1x_2, x_2^2)$ ; RBF $e^{-\gamma\lVert\mathbf{x} - \mathbf{z}\rVert^2}$ ; $\lVert\varphi(\mathbf{x}) - \varphi(\mathbf{z})\rVert^2 = K(\mathbf{x}, \mathbf{x}) - 2K(\mathbf{x}, \mathbf{z}) + K(\mathbf{z}, \mathbf{z})$ ; prédiction $\sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$ | `SVC(kernel="rbf", C=1.0, gamma="scale")` ($\gamma = 1/(p\,\mathrm{Var}(\mathbf{X}))$) ; `GridSearchCV` sur C et `gamma` |
| impuretés | Gini $1 - \sum_c p_c^2$ (max $1 - 1/K$) ; entropie $-\sum_c p_c\log_2 p_c$ (max $\log_2 K$) ; régression : variance $\frac1n\sum_i (y_i - \bar y)^2$ | `mylearn.tree.gini_impurity`, `entropy_impurity`, `variance_impurity` ; `tree_.impurity` |
| gain d'un split | $I(\text{parent}) - \frac{n_G}{n} I(G) - \frac{n_D}{n} I(D)$ (poids d'exemples : sommes de poids) ; seuils candidats : milieux de deux valeurs consécutives triées ; $x_j \le s$ à gauche | `mylearn.tree.split_gain`, `best_split` |
| arbres | une feuille de plus que de nœuds internes ; au plus $2^D$ feuilles à la profondeur $D$ ; `predict_proba` = proportions de la feuille | `mylearn.tree.DecisionTreeClassifier(max_depth=None, min_samples_leaf=1)`, `DecisionTreeRegressor` ; `plot_tree`, `export_text` |
| élagage coût-complexité | garder le sous-arbre qui minimise $R_\alpha(T) = R(T) + \alpha\lvert T\rvert$ | `DecisionTreeClassifier(ccp_alpha=0.01)` ; `cost_complexity_pruning_path(X, y)` |
| importance des features (MDI) | $\sum_{\text{nœuds sur } j} \frac{w_{\text{nœud}}}{w_{\text{racine}}}\,\text{gain}$, normalisée à 1 | `tree.feature_importances_` ; préférer `permutation_importance` sur la validation |
| Naïve Bayes | $P(c \mid \mathbf{x}) \propto P(c)\prod_j P(x_j \mid c)$ (indépendance sachant la classe) ; $\ell_c = \log P(c) + \sum_j \log P(x_j \mid c)$ ; $\log P(c \mid \mathbf{x}) = \ell_c - \mathrm{logsumexp}(\ell)$ | `mylearn.naive_bayes.logsumexp` ; `scipy.special.logsumexp` |
| Naïve Bayes gaussien | $\log\mathcal{N}(x; \mu, \sigma^2) = -\frac12\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}$ ; $\mu_{cj}$, $\sigma^2_{cj}$ par classe (ddof = 0), plus $\varepsilon$ = `var_smoothing` × la plus grande variance | `mylearn.naive_bayes.GaussianNB()` ; `sklearn.naive_bayes.GaussianNB` |
| Naïve Bayes multinomial | $\hat\theta_{cj} = \frac{N_{cj} + \alpha}{N_c + \alpha p}$ ; $\ell_c = \log P(c) + \sum_j x_j \log\hat\theta_{cj}$ | `mylearn.naive_bayes.MultinomialNB(alpha=1.0)` ; `MultinomialNB(alpha=1.0)` |
| TF-IDF | $\mathrm{idf}(t) = \ln\frac{1 + n}{1 + \mathrm{df}(t)} + 1$, puis chaque vecteur divisé par sa norme | `TfidfVectorizer()` ; `CountVectorizer()` |

### Ch. 14 · Ensembles

$M$ membres $h_1, \dots, h_M$ ; $K$ classes ; $w_m$ le poids d'un votant, $w_i$ celui d'un exemple ; $q$ la probabilité qu'un votant ait raison ; $\eta$ le learning rate.

| Notion | Formule | En code |
|---|---|---|
| vote pondéré | $\hat y = \arg\max_k \sum_m w_m\,\mathbb{1}[h_m(\mathbf{x}) = k]$ (égalité : la plus petite classe) ; vote souple : $\arg\max_k \frac1M \sum_m \hat p_{m,k}(\mathbf{x})$ | `mylearn.ensemble.plurality_vote(predictions, weights)` ; `VotingClassifier(estimators, voting="soft")` |
| majorité de votants indépendants | $P(\text{majorité juste}) = \sum_{j > M/2} \binom{M}{j} q^j (1 - q)^{M - j}$ ($M$ impair, deux classes) | `scipy.stats.binom.sf(M // 2, M, q)` |
| bootstrap | jamais tiré : $(1 - 1/n)^n \to e^{-1} \approx 0{,}368$ ; exemples distincts : $n\,(1 - (1 - 1/n)^n) \approx 0{,}632\,n$ | `mylearn.ensemble.bootstrap_indices(n, rng=rng)` ; `rng.integers(0, n, size=n)` |
| variance d'une moyenne corrélée | $\operatorname{Var}\big(\frac1M \sum_m h_m\big) = \rho\sigma^2 + \frac{1 - \rho}{M}\sigma^2$ | |
| bagging, forêts | moyenne des `predict_proba` des membres ; forêt : `max_features` features tirées à chaque nœud ($\lfloor\sqrt p\rfloor$ par défaut) ; ExtraTrees : un seuil uniforme entre le min et le max de la feature dans le nœud | `BaggingClassifier`, `RandomForestClassifier(n_estimators=100, oob_score=True)`, `ExtraTreesClassifier` ; `oob_score_`, `feature_importances_` |
| AdaBoost (SAMME) | $\varepsilon_m = \frac{\sum_i w_i \mathbb{1}[h_m(\mathbf{x}_i) \ne y_i]}{\sum_i w_i}$ ; $\alpha_m = \eta\,(\ln\frac{1 - \varepsilon_m}{\varepsilon_m} + \ln(K - 1))$ ; $w_i \leftarrow w_i e^{\alpha_m}$ pour les mal classés, puis normalisation ; arrêt si $\varepsilon_m = 0$ ou $\varepsilon_m \ge 1 - 1/K$ | `mylearn.ensemble.AdaBoostClassifier` ; `AdaBoostClassifier(n_estimators=50, learning_rate=1.0)` ; `estimator_weights_`, `estimator_errors_`, `staged_predict` |
| gradient boosting (MSE) | $F_0 = \bar y$ ; $r_i = y_i - F_{m-1}(\mathbf{x}_i) = -\frac{\partial}{\partial F}\frac12 (y_i - F)^2$ ; $F_m = F_{m-1} + \eta\,h_m$ | `mylearn.ensemble.GradientBoostingRegressor` ; `GradientBoostingRegressor(criterion="squared_error")` ; `train_score_`, `staged_predict` |
| pseudo-résidus | $-\partial \ell / \partial F$ : $y - F$ (quadratique), $\operatorname{signe}(y - F)$ (valeur absolue), $y - \hat p$ (log-loss, $\hat p = \sigma(F)$) | `HistGradientBoostingRegressor(loss=...)`, `XGBRegressor(objective=...)` |
| importance par permutation | baisse du score quand on mélange une colonne de la validation | `sklearn.inspection.permutation_importance(model, X_val, y_val, n_repeats=10)` |

### Ch. 15 · scikit-learn

| Notion | Formule | En code |
|---|---|---|
| du score à la probabilité (régression logistique) | deux classes : $\hat p = \sigma(s) = \frac{1}{1 + e^{-s}}$, $s = \ln\frac{\hat p}{1 - \hat p}$ ; $K$ classes : $\hat p_k = \frac{e^{s_k}}{\sum_j e^{s_j}}$ ; à deux classes, `predict` donne `classes_[1]` si $s > 0$ | `model.decision_function(X)` (forme `(n,)` à deux classes, `(n, K)` sinon), `model.predict_proba(X)` ; `scipy.special.expit`, `softmax` |
| mise à l'échelle min-max et son inverse | $x' = \frac{x - m}{M - m}$, $x = m + (M - m)\,x'$ ($m$, $M$ : minimum et maximum de l'entraînement) ; une droite reste une droite | `MinMaxScaler().fit(X_train)`, `transform`, `inverse_transform` |
| nombre d'entraînements d'une recherche | grille : $\big(\prod_h n_h\big) \times k + 1$ (le $+1$ : `refit=True`) ; au hasard : $n_{\text{iter}} \times k + 1$ ; une liste de grilles additionne les produits | `GridSearchCV(est, param_grid, cv=k)`, `RandomizedSearchCV(est, param_distributions, n_iter=...)` |
| au moins un bon essai au hasard | $P = 1 - (1 - q)^n$ ; $n \ge \frac{\ln(1 - P)}{\ln(1 - q)}$ | |
| successive halving | $N$ candidats, facteur $f$ : chaque tour garde $1/f$ des candidats et multiplie leurs ressources par $f$ ; avec des coûts proportionnels aux ressources, chaque tour coûte autant | `HalvingGridSearchCV(est, grid, factor=3)` (après `from sklearn.experimental import enable_halving_search_cv`) |
| nombre de features polynomiales | $\binom{p + d}{d}$ monômes de degré au plus $d$, constante comprise ; $\binom{p + d - 1}{d}$ de degré exactement $d$ ; sans carré (`interaction_only=True`) : $\sum_{j=0}^{d} \binom{p}{j}$ | `PolynomialFeatures(degree=d, include_bias=True).fit(X).n_output_features_` |
| poids des classes | `class_weight="balanced"` : $w_c = \frac{n}{K\, n_c}$ | `LogisticRegression(class_weight="balanced")` ; `sklearn.utils.class_weight.compute_class_weight` |
| taille du test | $\lceil t \cdot n \rceil$ exemples pour `test_size` $= t$ (0,25 par défaut) | `train_test_split(X, y, test_size=t, stratify=y, random_state=0)` |


## Partie IV : réseaux

### Ch. 16 · Réseaux feed-forward

$n$ exemples ; une couche de $n_{in}$ entrées (fan-in) et $n_{out}$ neurones (fan-out) ; $L$ couches de tailles $n_0, \dots, n_L$ ; $f$ la fonction d'activation.

| Notion | Formule | En code |
|---|---|---|
| couche dense sur un mini-batch | $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$, $\mathbf{X}$ $(n, n_{in})$, $\mathbf{W}$ $(n_{in}, n_{out})$, $\mathbf{b}$ $(n_{out},)$ ajouté à chaque ligne | `mylearn.nn.layers.dense_forward(X, W, b)` ; `torch.nn.functional.linear(x, W.T, b)` (PyTorch range `weight` en $(n_{out}, n_{in})$) |
| passe avant d'un MLP | $\mathbf{A}^{(0)} = \mathbf{X}$, $\mathbf{A}^{(\ell)} = f(\mathbf{A}^{(\ell-1)}\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)})$ ; cache : $(\mathbf{A}^{(\ell-1)}, \mathbf{Z}^{(\ell)})$ par couche | `mylearn.nn.layers.mlp_forward(X, params, activation, output_activation, return_cache=True)` |
| nombre de paramètres | $P = \sum_{\ell} (n_{\ell-1} n_\ell + n_\ell)$ ; mémoire en `float32` : $4P$ octets ; passe avant : $n \sum_\ell n_{\ell-1} n_\ell$ multiplications (entraînement ≈ 3 fois plus) | `mylearn.nn.layers.count_parameters(sizes)` ; `sum(p.numel() for p in model.parameters())` |
| variance d'une somme pondérée | $\mathrm{Var}(z) = n_{in}\,\mathrm{Var}(w)\,\mathbb{E}[x^2]$ (poids centrés, indépendants des entrées) ; ReLU : $\mathbb{E}[\mathrm{ReLU}(z)^2] = \frac12 \mathrm{Var}(z)$ ; $L$ couches de gain $g$ : variance $\times g^L$ | |
| loi uniforme | $\mathcal{U}(-a, a)$ : variance $a^2/3$, écart-type $a/\sqrt3$, densité $1/(2a)$ ; borne d'une règle : $a = \sqrt{3\,\mathrm{Var}(w)}$ | `rng.uniform(-a, a, size=(n_in, n_out))` |
| LeCun, Glorot, He | $\mathrm{Var}(w) = 1/n_{in}$ ; $2/(n_{in} + n_{out})$ ; $2/n_{in}$ ; bornes uniformes $\sqrt{3/n_{in}}$, $\sqrt{6/(n_{in} + n_{out})}$, $\sqrt{6/n_{in}}$ | `mylearn.nn.layers.init_weights(n_in, n_out, "he_normal", rng)` ; `torch.nn.init.kaiming_normal_(w, nonlinearity="relu")`, `xavier_uniform_` |
| ordre topologique (Kahn) | ranger un nœud dont tous les parents sont rangés (le plus petit nom d'abord) ; s'il en reste, le graphe a un cycle | `mylearn.nn.layers.topological_order(graph)` ; `graphlib.TopologicalSorter` |

### Ch. 17 · Fonctions d'activation

$z$ la somme pondérée d'un neurone ; $\sigma$ la sigmoïde ; $\mathbf{z} \in \mathbb{R}^K$ des logits, $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$.

| Notion | Formule | En code |
|---|---|---|
| effondrement | $L$ couches sans activation : $\mathbf{W} = \mathbf{W}_1 \mathbf{W}_2 \cdots \mathbf{W}_L$, biais $\sum_\ell \mathbf{b}_\ell \mathbf{W}_{\ell+1} \cdots \mathbf{W}_L$ | `MLPClassifier(activation="identity")` est une régression logistique |
| marche unité | $\mathrm{high}$ si $z \ge$ seuil, $\mathrm{low}$ sinon ; dérivée 0 | `mylearn.nn.activations.step(z, threshold, low, high)` ; `np.sign(0.0)` vaut 0 |
| ReLU et variantes | $\max(0, z)$, pente 0 en 0 ; leaky : $z$ si $z > 0$, $\alpha z$ sinon, pente $\alpha$ en 0 ; maxout : $\max_k (w_k z + b_k)$ | `relu`, `leaky_relu(z, negative_slope=0.01)` ; `torch.nn.functional.leaky_relu` |
| sigmoïde stable | $\frac{1}{1 + e^{-z}}$ si $z \ge 0$, $\frac{e^{z}}{1 + e^{z}}$ sinon ; $\sigma' = \sigma(1 - \sigma) \le \frac14$ | `sigmoid`, `sigmoid_derivative` ; `scipy.special.expit` |
| tanh | $\tanh z = 2\sigma(2z) - 1$ ; $\tanh' = 1 - \tanh^2 = 4\sigma'(2z) \le 1$ | `np.tanh`, `tanh_derivative` |
| softplus | $\frac{1}{\beta}\ln(1 + e^{\beta z})$, $z$ si $\beta z >$ seuil ; dérivée $\sigma(\beta z)$ | `softplus(z, beta=1.0, threshold=20.0)` ; `np.logaddexp(0, b * z) / b` |
| ELU | $z$ si $z > 0$, $\alpha(e^{z} - 1)$ sinon ; dérivée 1 ou $\alpha e^{z}$ | `elu(z, alpha=1.0)` ; `np.expm1` |
| SiLU | $z\,\sigma(z)$ ; dérivée $\sigma(z)\,(1 + z(1 - \sigma(z)))$, jusqu'à 1,10 | `silu`, `silu_derivative` |
| GELU | $z\,\Phi(z)$, dérivée $\Phi(z) + z\,\varphi(z)$ ; approximation $\frac{z}{2}\bigl(1 + \tanh(\sqrt{2/\pi}\,(z + 0{,}044715\,z^3))\bigr)$ | `gelu(z, approximate="none")` ; `scipy.special.ndtr` ; `torch.nn.functional.gelu` |
| softmax et température | $p_k = \frac{e^{(z_k - m)/T}}{\sum_j e^{(z_j - m)/T}}$, $m = \max_j z_j$ ; $p_i / p_j = e^{(z_i - z_j)/T}$ ; invariante par décalage | `softmax(z, axis=-1)` ; `scipy.special.softmax(z / T, axis=-1)` |
| log-softmax | $z_k - m - \log\sum_j e^{z_j - m}$ ; cross-entropy d'un exemple de classe $y$ : $-\log p_y$, de gradient $\mathbf{p} - \mathbf{e}_y$ | `log_softmax(z, axis=-1)` ; `scipy.special.log_softmax` |
| Jacobienne de la softmax | $J_{ij} = p_i(\mathbb{1}[i = j] - p_j)$, $\mathbf{J} = \mathrm{diag}(\mathbf{p}) - \mathbf{p}^\top\mathbf{p}$ ; $\mathbf{g}\mathbf{J} = \mathbf{p} \odot (\mathbf{g} - \mathbf{g}\cdot\mathbf{p})$ en $O(K)$ | `softmax_derivative(z)` (forme `(..., K, K)`) ; `torch.autograd.functional.jacobian` |
| saturation d'une couche | pente moyenne $\overline{f'(z)}$ et part des pentes sous 0,01 sur un batch de pré-activations | `sigmoid_derivative(Z).mean()`, `(sigmoid_derivative(Z) < 0.01).mean()` |

### Ch. 18 · Rétropropagation

$\delta_j = \partial L / \partial z_j$ le delta d'un neurone ; une couche $\mathbf{Z} = \mathbf{A}_{\text{prev}}\mathbf{W} + \mathbf{b}$, $\mathbf{A} = f(\mathbf{Z})$ sur un batch de $n$ exemples ; $\boldsymbol{\Delta} = \partial L / \partial \mathbf{Z}$.

| Notion | Formule | En code |
|---|---|---|
| delta et changement de la loss | $\Delta L \approx \delta\,\Delta z$ (premier ordre) | |
| delta de sortie (loss quadratique) | $L = \frac12\sum_k(\hat{y}_k - y_k)^2$ : $\delta_k = (\hat{y}_k - y_k)\,f'(z_k)$ ; sans activation, $\hat{y}_k - y_k$ | |
| delta caché | $\delta_j = f'(z_j) \sum_k w_{jk}\,\delta_k$ | |
| gradient d'un poids, d'un biais | $\partial L / \partial w_{ij} = a_i\,\delta_j$, $\partial L / \partial b_j = \delta_j$ | |
| mise à jour | $w \leftarrow w - \eta\,\partial L / \partial w$ ; au premier ordre, $\Delta L \approx -\eta\,(\partial L / \partial w)^2$ | |
| forme matricielle | $\boldsymbol{\Delta} = \mathbf{G} \odot f'(\mathbf{Z})$, $d\mathbf{W} = \mathbf{A}_{\text{prev}}^\top\boldsymbol{\Delta}$, $d\mathbf{b} = \mathbf{1}^\top\boldsymbol{\Delta}$, $\mathbf{G}_{\text{prev}} = \boldsymbol{\Delta}\mathbf{W}^\top$ | `mylearn.nn.backward.dense_backward(dout, x, W)`, `mlp_backward(dout, cache, params, activation_derivative)` |
| coût | passe avant $n\,n_{in}\,n_{out}$ multiplications par couche, passe arrière $2\,n\,n_{in}\,n_{out}$ ; différences centrées : $2P$ passes avant pour $P$ paramètres | |
| MSE | $\frac{1}{m}\sum(\hat{y} - y)^2$ sur les $m$ cases, gradient $\frac{2}{m}(\hat{\mathbf{y}} - \mathbf{y})$ | `mse_loss(y_pred, y_true)` ; `torch.nn.functional.mse_loss` |
| cross-entropy binaire sur le logit | $\ell = \max(z, 0) - yz + \log(1 + e^{-\lvert z\rvert})$, $\partial \ell / \partial z = \sigma(z) - y$ (moyenne : $/n$) | `binary_cross_entropy_with_logits(logits, y)` ; `torch.nn.BCEWithLogitsLoss` |
| cross-entropy softmax | $\ell = -\sum_k y_k \log p_k$, $\partial \ell / \partial \mathbf{z} = \mathbf{p} - \mathbf{y}$ (batch : $(\mathbf{P} - \mathbf{Y})/n$) | `softmax_cross_entropy(logits, y)` ; `torch.nn.CrossEntropyLoss` (logits, indices ou probabilités) |
| erreur quadratique après une sigmoïde | $\partial \ell / \partial z = (\sigma(z) - y)\,\sigma(z)(1 - \sigma(z))$ : s'éteint quand la sortie sature | |
| descente sur une parabole | $L = \frac a2(w - w^\star)^2$ : $w_{t+1} - w^\star = (1 - \eta a)(w_t - w^\star)$ ; converge si $0 < \eta < 2/a$, en un pas si $\eta = 1/a$ | |
| vérification d'un gradient | $g_n = \frac{L(w + \varepsilon) - L(w - \varepsilon)}{2\varepsilon}$, erreur relative $\frac{\lvert g_a - g_n\rvert}{\max(\lvert g_a\rvert + \lvert g_n\rvert, 10^{-12})}$ ; float64, $\varepsilon \approx 10^{-6}$ | `gradient_check(f, param, analytic_grad)` ; `torch.autograd.gradcheck` |
| micro-autograd | `c = a * b` : `a.grad += b.data * c.grad`, `b.grad += a.data * c.grad` ; `backward()` dans l'ordre topologique inverse | `mylearn.autograd_mini.Value`, `MiniMLP` ; `loss.backward()` |

### Ch. 19 · Optimiseurs

$g_t$ le gradient au pas $t$ (compté à partir de 1), $\eta$ le learning rate ; opérations case par case. Conventions de `torch.optim`.

| Notion | Formule | En code |
|---|---|---|
| descente vanille | $w_{t+1} = w_t - \eta\,g_t$ ; weight decay couplé : $g \leftarrow g + \lambda w$ | `SGD(params, lr, weight_decay)` ; `torch.optim.SGD` |
| courbure et pas | $L = \frac a2 w^2$ : $w \leftarrow (1 - \eta a)\,w$, converge si $0 < \eta < 2/a$ ; hessienne : $\eta < 2/\lambda_{\max}$, conditionnement $\kappa = \lambda_{\max}/\lambda_{\min}$, meilleur facteur $(\kappa - 1)/(\kappa + 1)$ | |
| plannings | exponentiel $\eta_k = \eta_0\gamma^k$ ; par paliers $\eta_0\gamma^{\lfloor k/s \rfloor}$ ; somme bornée $\sum_k \eta_0\gamma^k = \eta_0/(1 - \gamma)$ | `exponential_decay_lr(step, base_lr, gamma)`, `step_decay_lr(...)`, `bold_driver_lr(...)` ; `ExponentialLR`, `StepLR`, `ReduceLROnPlateau` |
| warmup et cosinus | $\eta_t = \eta_{\max} t/T_w$ ($t < T_w$), puis $\eta_{\min} + \frac12(\eta_{\max} - \eta_{\min})\bigl(1 + \cos\frac{\pi(t - T_w)}{T - T_w}\bigr)$, et $\eta_{\min}$ après $T$ | `cosine_lr(step, total_steps, base_lr, warmup_steps, min_lr)` ; `get_cosine_schedule_with_warmup`, `CosineAnnealingLR` |
| momentum | $b_t = \mu b_{t-1} + g_t$ ($b_1 = g_1$), $w \leftarrow w - \eta\,b_t$ ; livre : $\Delta_t = \gamma\Delta_{t-1} - \eta g_t$ (égaux à $\eta$ constant) ; gradient constant : $b \to g/(1 - \mu)$ | `Momentum(params, lr, momentum)` ; `SGD(momentum=0.9)` |
| Nesterov | livre : gradient en $w + \gamma\Delta$ ; PyTorch : $w \leftarrow w - \eta\,(g_t + \mu b_t)$, paramètres = points anticipés | `Nesterov(...)` ; `SGD(momentum=0.9, nesterov=True)` |
| Adagrad | $s_t = s_{t-1} + g_t^2$, $w \leftarrow w - \eta\,g_t/(\sqrt{s_t} + \epsilon)$ | `Adagrad(params, lr=1e-2, eps=1e-10)` |
| RMSprop | $v_t = \alpha v_{t-1} + (1 - \alpha)g_t^2$, $w \leftarrow w - \eta\,g_t/(\sqrt{v_t} + \epsilon)$ | `RMSprop(params, lr=1e-2, alpha=0.99)` |
| Adadelta | $v_t = \rho v + (1 - \rho)g_t^2$, $\Delta w = -\frac{\sqrt{u_{t-1} + \epsilon}}{\sqrt{v_t + \epsilon}}\,g_t$, $u_t = \rho u + (1 - \rho)\Delta w^2$, $w \leftarrow w + \eta\,\Delta w$ | `Adadelta(params, lr=1.0, rho=0.9, eps=1e-6)` |
| moyenne mobile exponentielle | $m_t = (1 - \beta)\sum_{k<t}\beta^k x_{t-k}$, somme des poids $1 - \beta^t$, mémoire $\approx 1/(1 - \beta)$, demi-vie $\approx 0{,}69/(1 - \beta)$ | |
| Adam | $m_t = \beta_1 m + (1 - \beta_1)g$, $v_t = \beta_2 v + (1 - \beta_2)g^2$, $\hat m = m/(1 - \beta_1^t)$, $\hat v = v/(1 - \beta_2^t)$, $w \leftarrow w - \eta\,\hat m/(\sqrt{\hat v} + \epsilon)$ ; premier pas : $\eta\,\mathrm{signe}(g_1)$ | `Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)` |
| AdamW | $w \leftarrow w\,(1 - \eta\lambda)$, puis le pas d'Adam sur $g$ brut ; article : $w \leftarrow w - \eta_t\lambda w$ ($\text{wd}_{\text{torch}} = \lambda/\alpha$) | `AdamW(params, weight_decay=1e-2)` ; `Adam(decoupled_weight_decay=True)` |
| gradient clipping | $\mathbf{g} \leftarrow \mathbf{g}\,\min\bigl(1, c/(\lVert\mathbf{g}\rVert_2 + 10^{-6})\bigr)$, norme globale de tous les gradients | `clip_grad_norm(grads, max_norm)` ; `torch.nn.utils.clip_grad_norm_` |
| mémoire d'un optimiseur | par paramètre : SGD 0, momentum 1, Adam 2 nombres ; précision mixte avec Adam ≈ 16 octets par paramètre | |

### Ch. 20 · Deep learning et premiers pas en PyTorch

Conventions de PyTorch ; une feature $j$, un mini-batch de $m$ valeurs par feature.

| Notion | Formule | En code |
|---|---|---|
| couche dense | $\mathbf{z} = \mathbf{x}\mathbf{W}^\top + \mathbf{b}$ (PyTorch, `weight` de forme $(n_{out}, n_{in})$) ; $n_{in}n_{out} + n_{out}$ paramètres | `nn.Linear(n_in, n_out)` |
| convolution, pooling | $(k^2 C_{in} + 1)\,C_{out}$ paramètres ; pooling $k \times k$ disjoint : $H \times W \to \lfloor H/k \rfloor \times \lfloor W/k \rfloor$ ; padding $k$ : $+2k$ | `nn.Conv2d`, `nn.MaxPool2d`, `nn.ZeroPad2d` |
| dropout inversé | $\tilde a = a\,m/(1 - p)$, $m \sim \mathrm{Bernoulli}(1 - p)$ ; $\mathbb{E}[\tilde a] = a$, $\mathrm{Var}[\tilde a] = a^2 p/(1 - p)$ ; passe arrière : même masque ; identité en évaluation | `dropout_forward(x, p, training, rng)`, `dropout_backward(dout, mask)` ; `nn.Dropout(p)` |
| batchnorm (entraînement) | $\hat x = (x - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$, $\sigma_B^2$ biaisée ; $y = \gamma\hat x + \beta$ | `batchnorm_forward(x, gamma, beta, running_mean, running_var)` ; `nn.BatchNorm1d`, `nn.BatchNorm2d` |
| statistiques glissantes | $\mu^{\text{run}} \leftarrow (1 - \rho)\,\mu^{\text{run}} + \rho\,\mu_B$, $\sigma^{2,\text{run}} \leftarrow (1 - \rho)\,\sigma^{2,\text{run}} + \rho\,\frac{m}{m-1}\sigma_B^2$ ; $\rho = 0{,}1$ (Keras : $1 - \rho$) ; départ $0$ et $1$ | `momentum=0.1` |
| batchnorm (passe arrière) | $\partial L/\partial\beta = \sum_i g_i$, $\partial L/\partial\gamma = \sum_i g_i\hat x_i$, $\partial L/\partial x_j = \frac{\gamma s}{m}\bigl(m g_j - \sum_i g_i - \hat x_j\sum_i g_i\hat x_i\bigr)$, $s = (\sigma_B^2 + \epsilon)^{-1/2}$ ; évaluation : $\gamma g_j/\sqrt{\sigma^{2,\text{run}} + \epsilon}$ | `batchnorm_backward(dout, cache)` (∂ 20.8) |
| layer norm | mêmes formules, $\mu_i$ et $\sigma_i^2$ (biaisée) sur les $D$ features de l'exemple $i$ ; pas de statistiques glissantes | `layer_norm(x, gamma, beta, eps)` ; `nn.LayerNorm(D)` |
| pénalité L2 | $L + \frac{\lambda}{2}\sum\lVert\mathbf{W}\rVert^2$, gradient $+\lambda\mathbf{W}$ (= `weight_decay` de SGD) | `l2_penalty(weights, lam)` |
| early stopping | amélioration si $v < v^\star - \delta$ (mode `min`) ; arrêt après `patience` vérifications sans amélioration ; restaurer les meilleurs poids | `EarlyStopping(patience, min_delta, mode)` |
| connexion résiduelle | $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$, $\partial\mathbf{y}/\partial\mathbf{x} = I + \partial F/\partial\mathbf{x}$ ; sur $L$ blocs, $\prod(1 + F'_\ell)$ au lieu de $\prod f'_\ell$ | `x = x + block(x)` dans `forward` |
| une epoch PyTorch | `model.train()` ; par mini-batch : passe avant, loss sur les logits, `zero_grad()`, `backward()`, `step()` ; puis `model.eval()` et `torch.no_grad()` | |
| coût d'entraînement | passe avant + arrière $\approx 3 \times$ la passe avant ; mémoire Adam en `float32` : 16 octets par paramètre, sans les activations | (🧮 20.9) |

## Partie V : architectures
### Ch. 21 · Réseaux convolutifs

Conventions de PyTorch : format `(N, C, H, W)`, poids `(C_out, C_in/groups, k_h, k_w)`, corrélation croisée ; formules par direction.

| Notion | Formule | En code |
|---|---|---|
| corrélation croisée 2D | $y_{o,i,j} = b_o + \sum_{c,u,v} W_{o,c,u,v}\,x_{c,\,s i + d u,\,s j + d v}$ (entrée complétée) ; la convolution retourne le noyau | `conv2d(x, weight, bias, stride, padding, dilation, groups)` ; `F.conv2d` |
| taille de sortie | $H_{out} = \lfloor (H + 2p - d(k - 1) - 1)/s \rfloor + 1$ ; « same » ($s = 1$) : padding total $d(k - 1)$, le reste en bas et à droite | `conv_output_size(size, k, stride, padding, dilation)` |
| paramètres | $C_{out}\,(C_{in}\,k_h k_w/g + 1)$ ; fan-in $= C_{in}\,k_h k_w/g$ ; défaut PyTorch $\mathcal{U}(\pm 1/\sqrt{\text{fan-in}})$ | `nn.Conv2d(C_in, C_out, k, groups=g)` |
| calcul | $H_{out}W_{out}\,C_{out}\,C_{in}\,k^2/g$ multiplications-additions par image | (🧮 21.8) |
| im2col | matrice $(N, C_{in}k_hk_w, H_{out}W_{out})$, lignes dans l'ordre $c\,k_hk_w + u\,k_w + v$ ; convolution = poids aplatis × colonnes | `im2col(x, kernel_size, stride, padding, dilation)` ; `F.unfold` |
| champ récepteur | $r_\ell = r_{\ell-1} + (k_\ell - 1)\prod_{i<\ell} s_i$, $r_0 = 1$ ; pooling compris | (✏️ 21.5) |
| pooling | même taille de sortie ; max : padding $= -\infty$ ; moyenne : divise par $k_hk_w$ (`count_include_pad=True`) ; padding $\le \lfloor k/2 \rfloor$ | `max_pool2d`, `avg_pool2d` ; `nn.MaxPool2d`, `nn.AvgPool2d` |
| convolution transposée | $H_{out} = (H - 1)s - 2p + k$ (+ `output_padding`) ; recette : $s - 1$ zéros entre les pixels, $k - 1 - p$ de bordure, noyau retourné et canaux échangés | `conv_transpose2d(x, weight, bias, stride, padding)` ; poids `(C_in, C_out, k, k)` |
| rétropropagation | $\partial L/\partial W_{o,c,u,v} = \sum_{n,i,j} g_{n,o,i,j}\,x_{n,c,\,si+u,\,sj+v}$ ; $\partial L/\partial b_o = \sum_{n,i,j} g_{n,o,i,j}$ ; $\partial L/\partial \mathbf{x}$ = convolution transposée de $g$ par $W$ | `conv2d_backward(dout, x, weight, stride, padding)` (∂ 21.7) |
| FGSM | $\mathbf{x}_{adv} = \operatorname{clamp}\big(\mathbf{x} + \varepsilon\,\operatorname{sign}(\nabla_{\mathbf{x}} L(\mathbf{x}, y)), 0, 1\big)$ ; modèle linéaire : le score change de $\varepsilon\lVert\mathbf{w}\rVert_1$ | `x.requires_grad_()`, `loss.backward()`, `x.grad.sign()` |

### Ch. 22 · Réseaux récurrents

Conventions de PyTorch et de `mylearn` : vecteurs lignes, $x_t$ de taille $D$, état de taille $H$ ; `mylearn` range `W_ih` en $(D, kH)$ (la transposée de `weight_ih`) ; $\sigma$ la sigmoïde, $\odot$ le produit élément par élément.

| Notion | Formule | En code |
|---|---|---|
| fenêtres | $N = \lfloor (T - w - h)/s \rfloor + 1$ ; la fenêtre $p$ couvre $p..p+w-1$, sa cible est en $p + w - 1 + h$ | `make_windows(series, window, horizon, stride)` |
| cellule RNN | $h_t = \tanh(x_t W_{ih} + b_{ih} + h_{t-1} W_{hh} + b_{hh})$, $h_0 = 0$ | `rnn_cell_forward`, `rnn_forward` ; `nn.RNN` |
| LSTM ($i, f, g, o$) | $[a_i \mid a_f \mid a_g \mid a_o] = x_t W_{ih} + b_{ih} + h_{t-1} W_{hh} + b_{hh}$ ; $i, f, o = \sigma(\cdot)$, $g = \tanh(a_g)$ ; $c_t = f \odot c_{t-1} + i \odot g$ ; $h_t = o \odot \tanh c_t$ | `lstm_cell_forward`, `lstm_forward` ; `nn.LSTM` |
| GRU ($r, z, n$) | $r, z = \sigma(\cdot)$ ; $n = \tanh\big(x_t W_{in} + b_{in} + r \odot (h_{t-1} W_{hn} + b_{hn})\big)$ ; $h_t = (1 - z) \odot n + z \odot h_{t-1}$ | `gru_cell_forward` ; `nn.GRU` |
| paramètres d'une couche | $k\,(DH + H^2 + 2H)$, $k = 1$ (RNN), 3 (GRU), 4 (LSTM) ; Keras : un seul biais, $k\,(DH + H^2 + H)$ ; bidirectionnelle : $\times 2$ | (✏️ 22.3) |
| gradient dans le temps | $\partial h_T/\partial h_t = \prod_{k=t+1}^{T} \operatorname{diag}(1 - h_k^2)\,W_{hh}^\top$ ; $\lVert\cdot\rVert \le \sigma_{\max}(W_{hh})^{T-t}$ | (∂ 22.6) |
| chemin additif | $\partial c_T/\partial c_t = \operatorname{diag}\big(\prod_{k=t+1}^{T} f_k\big)$ (chemin direct) ; GRU : $\operatorname{diag}(z_t)$ par pas | (∂ 22.7) |
| température | $p = \mathrm{softmax}(z/T)$ ; $T \to 0$ : l'argmax ; $T \to \infty$ : l'uniforme ; $p_1/p_2 = e^{(z_1 - z_2)/T}$ | `temperature_softmax`, `sample_from_logits(logits, temperature, top_k, rng)` |
| perplexité | $\exp\big(-\frac{1}{N}\sum_t \ln p(x_t \mid x_{<t})\big)$ = $e^{\text{loss moyenne}}$ (en nats) | `math.exp(loss)` |

### Ch. 23 · PyTorch en pratique 1

Conventions de PyTorch : images `(N, C, H, W)`, rangement en mémoire dans l'ordre C, entrées en `float32`, labels en `int64`.

| Notion | Formule | En code |
|---|---|---|
| indice plat | élément $(i, j, k)$ d'une forme $(D_1, D_2, D_3)$ : position $i\,D_2 D_3 + j\,D_3 + k$ | `np.ravel_multi_index((i, j, k), shape)` ; `reshape` garde l'ordre, `permute` le change |
| normalisation | $x' = (x/255 - \mu_{\text{train}})/\sigma_{\text{train}}$ ($\sigma$ avec le diviseur $n$) | `v2.Normalize((mu,), (sigma,))` ; $\mu$, $\sigma$ sauvés avec le modèle |
| couche dense | $(n_{in} + 1)\,n_{out}$ paramètres ; MLP 784-784-10 : 623 290 | `sum(p.numel() for p in model.parameters())` |
| cross-entropy sur des logits | $L = -z_y + \ln \sum_j e^{z_j}$, moyenne sur le mini-batch ; loss de départ à $K$ classes $\approx \ln K$ | `nn.CrossEntropyLoss()(logits, y)` |
| mini-batches par epoch | $\lceil N/B \rceil$ ; $\lfloor N/B \rfloor$ avec `drop_last=True` ; une mise à jour par mini-batch | `len(loader)` |
| early stopping | amélioration si $v_e < v^\star - \delta$ ; arrêt quand $p$ epochs de suite n'en sont pas ; recharger la dernière amélioration | `EarlyStopping(patience, min_delta)` (ch. 20) |
| `LambdaLR` | $\eta_e = \eta_0 \cdot \lambda(e)$, $e$ = nombre d'appels à `scheduler.step()` ; livre : $\lambda(e) = \max(0{,}1 ;\ 1 - 0{,}01\,e)$ avec $\eta_0 = 1$ | `LambdaLR(opt, lambda e: ...)` |
| `StepLR`, cosinus | $\eta_0\,\gamma^{\lfloor e/s \rfloor}$ ; $\eta_0\,(1 + \cos(\pi e/T))/2$ | `StepLR(opt, s, gamma)`, `CosineAnnealingLR(opt, T)` |
| coût d'une epoch | $\approx 6 P N$ opérations ; poids : $4P$ octets en `float32`, $12P$ avec les deux moments d'Adam | (🧮 23.8) |

### Ch. 24 · PyTorch en pratique 2

Conventions de PyTorch : images `(N, C, H, W)`, séquences `(N, T, D)` avec `batch_first=True`, $H$ valeurs d'état ; MAC = multiplication-addition.

| Notion | Formule | En code |
|---|---|---|
| combinaisons et coût d'une grille | $\prod_j \lvert V_j \rvert$ candidats ; $K \times \prod_j \lvert V_j \rvert$ entraînements, $+1$ pour le réentraînement final | `GridSearchCV(pipe, grid, cv=K)` (cv = 5 par défaut) |
| recherche au hasard | si une fraction $q$ des réglages est bonne, $P(\text{au moins un bon en } n \text{ tirages}) = 1 - (1 - q)^n$ | `RandomizedSearchCV(pipe, distributions, n_iter=n)` |
| bruit d'un mini-batch | écart-type du gradient moyen $\approx \sigma/\sqrt{B}$ ; $\lceil N/B \rceil$ pas par epoch | `DataLoader(ds, batch_size=B)` |
| dropout inversé | $\tilde h = m\,h/(1 - p)$, $m \sim \text{Bernoulli}(1 - p)$, un masque par passe avant ; $\mathbb{E}[\tilde h] = h$, $\text{Var}(\tilde h) = \frac{p}{1-p}h^2$ ; évaluation : $\tilde h = h$ | `nn.Dropout(p)` (∂ 24.6) |
| max-norm | $\mathbf{w} \leftarrow \mathbf{w}\,\min(1, c/\lVert\mathbf{w}\rVert_2)$, la projection sur la boule de rayon $c$ ; une ligne de `weight` par neurone | `W.copy_(torch.renorm(W, 2, 0, c))` sous `no_grad`, après `step()` |
| convolution | $C_{out}(C_{in}k_hk_w + 1)$ paramètres ; $H_{out}W_{out}C_{out}C_{in}k_hk_w$ MAC par image ; « same » de Keras avec un stride $s$ : $\lceil H/s \rceil$ | `nn.Conv2d(C_in, C_out, k, stride, padding=k // 2)` |
| couche dense | $(n_{in} + 1)\,n_{out}$ paramètres, $n_{in}\,n_{out}$ MAC | `nn.Linear(n_in, n_out)` |
| mise à l'échelle des images | $x' = x/127{,}5 - 1$ dans $[-1, 1]$ ; standardisation par canal $(x/255 - \mu_c)/\sigma_c$ (CIFAR-10 : $\mu \approx (0{,}491 ; 0{,}482 ; 0{,}447)$) | `v2.Normalize(mean, std)` |
| MinMaxScaler | $x' = (x - x_{\min})/(x_{\max} - x_{\min})$, $x = x_{\min} + x'(x_{\max} - x_{\min})$ (min et max du train) | `MinMaxScaler().fit(train.reshape(-1, 1))`, `inverse_transform` |
| fenêtres | $T - w$ exemples pour une cible juste après la fenêtre | `make_windows(series, window=w)` |
| RMSE | $\sqrt{\frac1N \sum_i (y_i - \hat y_i)^2}$, en unités d'origine | `math.sqrt(mean_squared_error(y, y_hat))` |
| somme de sinus | $s(n) = \sum_k a_k \sin\big(f_k(\varphi_k + n\,\Delta\theta)\big)$ | `(a * np.sin(f * (ph + (n * d_theta)[:, None]))).sum(axis=1)` |
| sorties de `nn.LSTM` | `output` $(N, T, n_{dir}H)$ ; `h_n`, `c_n` $(L\,n_{dir}, N, H)$ ; un sens : `output[:, -1]` = `h_n[-1]` | `output, (h_n, c_n) = lstm(x)` |
| paramètres d'une couche LSTM | $4H(D_{in} + H) + 8H$ (deux biais) | `sum(p.numel() for p in lstm.parameters())` |
| embedding | $\mathbf{o}_j^\top \mathbf{E} = \mathbf{E}_{j,:}$ : lire la ligne $j$ d'une table $(V, d)$ | `nn.Embedding(V, d)` |
| bloc résiduel | $\mathbf{y} = \text{ReLU}(F(\mathbf{x}) + \mathbf{x})$ ; raccourci $1 \times 1$ si les formes diffèrent | `torch.relu(self.f(x) + self.shortcut(x))` |
| coût d'un entraînement | MAC par image $\times 2 \times 3 \times$ images $\times$ epochs opérations | (🧮 24.7) |

## Partie VI : génératif et RL
### Ch. 25 · Autoencodeurs et VAE

Notations : une entrée $\mathbf{x}$ de $p$ valeurs, un code $\mathbf{z}$ de $d$ valeurs ; $\log$ népérien, KL en nats.

| Notion | Formule | En code |
|---|---|---|
| autoencodeur | $\mathbf{z} = f_\theta(\mathbf{x})$, $\hat{\mathbf{x}} = g_\phi(\mathbf{z})$ ; $L_{\text{rec}} = \frac{1}{n}\sum_i \lVert\mathbf{x}_i - \hat{\mathbf{x}}_i\rVert^2$ | `loss_fn(model(x), x)` |
| cross-entropy binaire (pixels dans $[0, 1]$) | $-\sum_j [x_j \ln \hat{x}_j + (1 - x_j)\ln(1 - \hat{x}_j)]$ | `F.binary_cross_entropy_with_logits(logits, x)` |
| autoencodeur linéaire et ACP | $\min_{\mathbf{W}} \lVert\mathbf{X} - \mathbf{X}\mathbf{W}\mathbf{W}^\top\rVert_F^2$ : le sous-espace des $d$ premières composantes principales (pas ses axes) ; en 2D, $L = \operatorname{tr}\mathbf{C} - \mathbf{w}^\top\mathbf{C}\,\mathbf{w}$ | `PCA(n_components=d)` (∂ 25.8) |
| convolution transposée | $H_{out} = (H_{in} - 1)s - 2p + k + p_{out}$, $p_{out} < s$ ; poids $(C_{in}, C_{out}, k, k)$ | `nn.ConvTranspose2d(C_in, C_out, k, stride, padding, output_padding)` |
| PSNR | $10\log_{10}(1/\text{MSE})$ dB pour des pixels dans $[0, 1]$ | `10 * torch.log10(1 / F.mse_loss(out, clean))` |
| norme d'un tirage gaussien | $\mathbb{E}\lVert\mathbf{z}\rVert^2 = d$ pour $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ ; en 2D, $P(\lVert\mathbf{z}\rVert \le r) = 1 - e^{-r^2/2}$ | `np.linalg.norm(rng.standard_normal((n, d)), axis=1)` |
| interpolation | linéaire $(1 - t)\mathbf{z}_0 + t\mathbf{z}_1$ ; sphérique $\frac{\sin((1-t)\Omega)}{\sin\Omega}\mathbf{z}_0 + \frac{\sin(t\Omega)}{\sin\Omega}\mathbf{z}_1$, $\cos\Omega = \frac{\mathbf{z}_0\cdot\mathbf{z}_1}{\lVert\mathbf{z}_0\rVert\lVert\mathbf{z}_1\rVert}$ | `mylearn.vae.interpolate_latents(z0, z1, n, method)` |
| reparamétrisation | $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\boldsymbol{\sigma} = e^{\frac{1}{2}\log\boldsymbol{\sigma}^2}$ ; $\partial z/\partial\mu = 1$, $\partial z/\partial \log\sigma^2 = \tfrac{1}{2}\sigma\varepsilon$ | `mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)` ; `mylearn.vae.reparameterize`, `reparameterize_backward` (∂ 25.5) |
| KL gaussienne | $\mathrm{KL}(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)) = \tfrac{1}{2}(\mu^2 + \sigma^2 - 1 - \ln\sigma^2)$, sommée sur les latents ; gradients $\mu$ et $\tfrac{1}{2}(\sigma^2 - 1)$ par rapport à $\log\sigma^2$ | `0.5 * (mu**2 + log_var.exp() - 1 - log_var).sum()` ; `mylearn.vae.gaussian_kl`, `gaussian_kl_grad` (∂ 25.7) |
| loss du VAE | $\mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}[-\ln p(\mathbf{x}\mid\mathbf{z})] + \beta\,\mathrm{KL}(q(\mathbf{z}\mid\mathbf{x}) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I}))$, $\beta = 1$ : l'opposé de l'ELBO | reconstruction **sommée** sur les pixels + `beta * kl` |
| espérances d'une gaussienne | $\mathbb{E}[(z - \mu)^2] = \sigma^2$, $\mathbb{E}[z^2] = \mu^2 + \sigma^2$ | |

### Ch. 26 · Apprentissage par renforcement

Notations : état $s$, action $a$, récompense $r$, état suivant $s'$, action suivante $a'$ ; $\gamma$ le facteur d'actualisation, $\alpha$ le learning rate, $\varepsilon$ le taux d'exploration, $\tau$ la température ; $|\mathcal{A}|$ le nombre d'actions.

| Notion | Formule | En code |
|---|---|---|
| retour actualisé | $G_t = \sum_{k \ge 0} \gamma^k r_{t+k} = r_t + \gamma\,G_{t+1}$ ; $\gamma = 1$ : retour total | `mylearn.rl.discounted_return`, `returns_to_go` |
| borne et horizon | $\lvert G_t\rvert \le \frac{r_{\max}}{1-\gamma}$ ; horizon effectif $\frac{1}{1-\gamma}$ ; $\gamma^k < \tfrac{1}{2}$ dès $k > \frac{\ln 2}{-\ln\gamma}$ | (∂ 26.7) |
| Q-learning | $Q(s, a) \leftarrow Q(s, a) + \alpha\,[y - Q(s, a)]$, $y = r + \gamma \max_{a'} Q(s', a')$, $y = r$ si $s'$ terminal | `mylearn.rl.q_learning_update` |
| SARSA | $y = r + \gamma\,Q(s', a')$, $a'$ l'action choisie et jouée ensuite | `mylearn.rl.sarsa_update` |
| ε-greedy | $p(a^*) = 1 - \varepsilon + \frac{\varepsilon}{\lvert\mathcal{A}\rvert}$, $p(a) = \frac{\varepsilon}{\lvert\mathcal{A}\rvert}$ pour $a \ne a^*$ | `mylearn.bandit.epsilon_greedy_action` |
| softmax (Boltzmann) | $p(a) = \frac{e^{Q(s, a)/\tau}}{\sum_b e^{Q(s, b)/\tau}}$, invariante par $Q + c$ | `mylearn.rl.softmax_action` |
| équation de Bellman | $Q^*(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a')]$ ; contraction : $\max\lvert\mathcal{T}Q_1 - \mathcal{T}Q_2\rvert \le \gamma \max\lvert Q_1 - Q_2\rvert$ | `mylearn.rl.value_iteration` (∂ 26.8) |
| convergence du Q-learning | chaque couple visité une infinité de fois ; $\sum \alpha = \infty$, $\sum \alpha^2 < \infty$ (par exemple $\alpha = 1/n(s, a)$) | |
| taille d'une table | lignes $\times$ colonnes $\times$ 4 octets (`float32`) ; Flippers $2^9 \times 9$, morpion $3^9 \times 9$ | (✏️ 26.2, 🧮 26.9) |
| Gymnasium | `state, info = env.reset(seed=…)` ; `state, reward, terminated, truncated, info = env.step(action)` | `wb.datasets.make_env("FrozenLake-v1")` |

### Ch. 27 à 29
*(à compléter)*

## Partie VII : bonus
### B1 à B8
*(à compléter)*
