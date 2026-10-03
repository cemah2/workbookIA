# Cheatsheet PyTorch

> Aide-mémoire rempli au fil des chapitres (20-27). Une ligne = une commande utile + ce qu'elle fait.

## Tenseurs et device

| Code | Effet |
|---|---|
| | |

## Autograd

| Code | Effet |
|---|---|
| `p = torch.tensor([0.5, -1.0], dtype=torch.float64, requires_grad=True)` | un tenseur dont PyTorch enregistre les opérations (en `float64` pour comparer à un gradient numérique) (ch. 5) |
| `loss = f(p)` puis `loss.backward()` | applique la règle de la chaîne de `loss` vers `p` ; `loss` doit être un scalaire (ch. 5) |
| `p.grad`, `p.grad.numpy()` | le gradient de `loss` par rapport à `p`, en tenseur ou en tableau NumPy (ch. 5) |
| `(g,) = torch.autograd.grad(loss, p)` | le même gradient, renvoyé sans toucher à `p.grad` (ch. 5) |
| `torch.autograd.gradcheck(f, (p,))` | compare le gradient automatique à des différences finies (entrées en `float64`) (ch. 5) |
| `torch.relu(x)`, `torch.clamp(x, min=0)`, `torch.maximum(x, y)` | même fonction pour ReLU, mais pente en 0 différente : 0, 1 et 0,5 (ch. 5) |
| `torch.optim.SGD([p], lr=0.1, maximize=False)` | descente de gradient : `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` (ch. 5, détails au ch. 19) |

## Définir un modèle (nn.Module)

| Code | Effet |
|---|---|
| `layer = torch.nn.Linear(n_in, n_out)` | une couche pleine : `layer.weight` de forme `(n_out, n_in)`, une ligne par neurone, `layer.bias` de forme `(n_out,)` ; `layer(x)` calcule `x @ weight.T + bias` (ch. 10) |
| `with torch.no_grad(): layer.weight.copy_(torch.tensor(W.T))` | charger des poids rangés à la façon de mylearn (`W` de forme `(n_in, n_out)`) : transposer ; une matrice carrée non transposée passe sans erreur (10.15) |
| `torch.nn.functional.linear(x, w[None, :], b)` | la somme pondérée d'un seul neurone pour un lot : `x @ w + b` (ch. 10) |
| `torch.nn.ReLU()`, `nn.GELU()`, `nn.SiLU()`, `nn.Sigmoid()`, `nn.Tanh()` | les fonctions d'activation courantes ; `nn.Threshold(t, v)` n'est **pas** le seuil du perceptron : il garde $x$ si $x > t$, et met `v` ailleurs (ch. 10, détails au ch. 17) |

## Dataset et DataLoader

| Code | Effet |
|---|---|
| | |

## Boucle d'entraînement

| Code | Effet |
|---|---|
| `loss = torch.nn.functional.cross_entropy(logits, y)` | la loss des classifieurs : la moyenne de $-\ln \mathrm{softmax}(\text{logits})_y$, en **nats** ; elle attend des logits, pas des probabilités (ch. 6, détails au ch. 20) |
| `torch.nn.CrossEntropyLoss(label_smoothing=0.1)` | la même loss avec une cible lissée (ch. 6) |
| `torch.exp(loss)` | la perplexité d'un modèle de langage, si `loss` est la cross-entropy moyenne par token (ch. 6) |

## Sauvegarder et recharger

| Code | Effet |
|---|---|
| | |

