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
| somme pondérée / sortie activée | $z$ / $a$ | $z = \mathbf{w}^\top \mathbf{x} + b$, $a = \sigma(z)$ |
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
| `sklearn.neural_network.MLPClassifier(alpha)` | moyenne de la loss $+ \frac{\alpha}{2n}\sum \mathbf{W}^2$ | $\frac{\alpha}{n}\mathbf{W}$ |
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
| nombre de mini-lots par epoch | $\lceil n / b \rceil$ (dernier lot de taille $n - b\lfloor n/b \rfloor$ s'il n'est pas vide) ; $\lfloor n / b \rfloor$ avec `drop_last` | `math.ceil(n / b)`, `n // b` |
| mises à jour des poids | (lots par epoch) $\times$ (nombre d'epochs) | |
| normalisation min-max | $x' = \dfrac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$ | `(x - x.min()) / (x.max() - x.min())` |
| standardisation (par colonne) | $z = \dfrac{x - \bar{x}}{s}$ | `(X - X.mean(axis=0)) / X.std(axis=0)` |
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
| droite | $y = m x + p$ ; pente $m = \frac{y_B - y_A}{x_B - x_A}$ | |
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
| règle de correction (1.16, justifiée aux ch. 5 et 18) | $w \leftarrow w + \eta\,(y - \hat{y})\,x$ ; $b \leftarrow b + \eta\,(y - \hat{y})$ | `error = y_i - (w * x_i + b)` |
| effet d'une correction sur l'erreur de l'échantillon | $e \leftarrow \left(1 - \eta\,(x^2 + 1)\right) e$ : l'erreur change de signe et grandit si $\eta\,(x^2 + 1) > 2$ | |
| interpolation linéaire au milieu | $\hat{y}\left(\frac{t_1 + t_2}{2}\right) = \frac{y_1 + y_2}{2}$ | |
| régression vers la moyenne (Galton) | $\hat{y} = \bar{y} + \frac{2}{3}\,(x - \bar{y})$ : écart de l'enfant $\approx \frac{2}{3}$ de l'écart mi-parental | |
| connexions entre deux couches pleines | $n_{\text{entrée}} \times n_{\text{sortie}}$ poids, plus $n_{\text{sortie}}$ biais | `sum(w.size for w in mlp.coefs_)` |
| moyenne mobile (lissage, 0B) | $\tilde{x}_t = \frac{1}{k}\sum_{j=0}^{k-1} x_{t-j}$ ($k - 1$ valeurs manquantes au début) | `s.rolling(k).mean()` |
| position le long d'un axe (réduction de dimension) | $\mathbf{q} \cdot \mathbf{u}$ avec $\lVert \mathbf{u} \rVert = 1$ ; distance perdue $= \sqrt{\lVert \mathbf{q} \rVert^2 - (\mathbf{q} \cdot \mathbf{u})^2}$ | `q @ u` |
| pureté d'un clustering | $\frac{1}{n}\sum_{\text{groupes}} (\text{effectif de l'espèce majoritaire du groupe})$ | `pd.crosstab(g, y).max(axis=1).sum() / n` |

### Ch. 2 · Hasard et statistiques
*(à compléter)*
### Ch. 3 · Probabilités et mesure de la qualité
*(à compléter)*
### Ch. 4 · Règle de Bayes
*(à compléter)*
### Ch. 5 · Courbes et surfaces
*(à compléter)*
### Ch. 6 · Théorie de l'information
*(à compléter)*

## Partie II : concepts
### Ch. 7 à 11
*(à compléter)*

## Partie III : ML classique
### Ch. 12 à 15
*(à compléter)*

## Partie IV : réseaux
### Ch. 16 à 20
*(à compléter)*

## Partie V : architectures
### Ch. 21 à 24
*(à compléter)*

## Partie VI : génératif et RL
### Ch. 25 à 29
*(à compléter)*

## Partie VII : bonus
### B1 à B8
*(à compléter)*
