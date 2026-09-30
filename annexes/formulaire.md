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
*(à compléter)*

## Partie I : fondations
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
