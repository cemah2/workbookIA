# Cheatsheet PyTorch

> Aide-mémoire rempli au fil des chapitres (20-27). Une ligne = une commande utile + ce qu'elle fait.

## Tenseurs et device

| Code | Effet |
|---|---|
| `x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])` | un tenseur ; les réels sont en `torch.float32` par défaut (NumPy : `float64`) ; `torch.tensor([1, 2])` est en `int64` (ch. 20) |
| `x.shape`, `x.ndim`, `x.numel()`, `x.dtype`, `x.device` | forme (`torch.Size`), nombre de dimensions, nombre d'éléments, type, appareil (ch. 20) |
| `torch.zeros(2, 3)`, `torch.ones_like(x)`, `torch.randn(64, 20)`, `torch.arange(10)` | les créations courantes, comme en NumPy (ch. 20) |
| `t = torch.from_numpy(a)` ; `t.numpy()` | conversion **sans copie** : les deux partagent la mémoire, et le `float64` de NumPy est gardé ; `.numpy()` seulement sur CPU et sans gradient (ch. 20) |
| `x.float()`, `x.double()`, `x.long()`, `x.to(torch.float32)` | changer de dtype ; les labels de `CrossEntropyLoss` sont en `int64` (`long`) (ch. 20) |
| `device = "cuda" if torch.cuda.is_available() else "cpu"` ; `x.to(device)`, `model.to(device)` | choisir l'appareil et y envoyer données et modèle ; tous les tenseurs d'un calcul doivent être sur le même (ch. 20) |
| `x.sum(dim=0)`, `x.mean(dim=1, keepdim=True)` | réductions : les paramètres s'appellent `dim` et `keepdim` (PyTorch accepte aussi `axis` et `keepdims`, les noms de NumPy, mais écris `dim` et `keepdim`, comme la documentation) (ch. 20) |
| `x.reshape(32, -1)`, `x.view(32, -1)`, `x.flatten(start_dim=1)` | remodeler ; `view` ne copie jamais (et échoue si la mémoire ne s'y prête pas), `reshape` copie si besoin (ch. 20) |
| `x.unsqueeze(0)`, `x.squeeze()`, `x.permute(0, 2, 3, 1)` | ajouter ou retirer une dimension de taille 1, réordonner les axes (NCHW → NHWC) (ch. 20) |
| `torch.var(x, dim=0)` contre `torch.var(x, dim=0, correction=0)` | variance non biaisée par défaut ($n - 1$), biaisée avec `correction=0` (`np.var` : biaisée par défaut) (ch. 20) |
| `torch.manual_seed(42)` ; `torch.Generator().manual_seed(0)` | fixer le générateur global, ou créer un générateur à part (pour un `DataLoader`) (ch. 20) |

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
| `p.grad.zero_()` ou `optimizer.zero_grad()` | remettre les gradients à zéro : sans cela, deux `backward()` successifs **ajoutent** leurs gradients (ch. 18) |
| `z.backward(dout)` | rétropropager un gradient amont `dout` (de la forme de `z`) à partir d'un résultat non scalaire, comme `mlp_backward(dout, ...)` (ch. 18) |
| `torch.nn.functional.cross_entropy(logits, labels)`, `binary_cross_entropy_with_logits(z, y)` | les losses calculées sur les logits, moyennées (`reduction="mean"`) ; `labels` en `torch.long` ou des probabilités (ch. 18) |
| `torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)` | vérifier une opération maison ; ses tolérances par défaut sont faites pour le `float64` (ch. 18) |

## Définir un modèle (nn.Module)

| Code | Effet |
|---|---|
| `layer = torch.nn.Linear(n_in, n_out)` | une couche dense : `layer.weight` de forme `(n_out, n_in)`, une ligne par neurone, `layer.bias` de forme `(n_out,)` ; `layer(x)` calcule `x @ weight.T + bias` (ch. 10) |
| `with torch.no_grad(): layer.weight.copy_(torch.tensor(W.T))` | charger des poids rangés à la façon de mylearn (`W` de forme `(n_in, n_out)`) : transposer ; une matrice carrée non transposée passe sans erreur (10.15) |
| `torch.nn.functional.linear(x, w[None, :], b)` | la somme pondérée d'un seul neurone pour un batch : `x @ w + b` (ch. 10) |
| `torch.nn.ReLU()`, `nn.GELU()`, `nn.SiLU()`, `nn.Sigmoid()`, `nn.Tanh()` | les fonctions d'activation courantes ; `nn.Threshold(t, v)` n'est **pas** le seuil du perceptron : il garde $x$ si $x > t$, et met `v` ailleurs (ch. 10, détails au ch. 17) |
| `torch.nn.Sequential(nn.Linear(784, 128), nn.Tanh(), nn.Linear(128, 10)).double()` | un MLP ; `model[::2]` donne ses couches `Linear` ; `.double()` passe en `float64` pour comparer à NumPy (ch. 16) |
| `sum(p.numel() for p in model.parameters())` | le nombre de paramètres d'un modèle (ch. 16) |
| `torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")`, `xavier_uniform_(layer.weight)`, `zeros_(layer.bias)` | les initialisations de He et de Glorot ; le défaut de `nn.Linear` est $\mathcal{U}(\pm 1/\sqrt{n_{in}})$, pas He (ch. 16) |
| `torch.cuda.synchronize()` | attendre la fin des calculs GPU (lancés de façon asynchrone) avant de lire l'heure (ch. 16) |
| `torch.nn.functional.relu`, `leaky_relu(x, negative_slope=0.01)`, `elu(x, alpha=1.0)`, `silu`, `gelu(x, approximate="none")` | les activations de mylearn, mêmes paramètres par défaut ; `approximate="tanh"` pour la variante rapide de GELU (ch. 17) |
| `torch.nn.functional.softplus(x, beta=1.0, threshold=20.0)` | softplus, linéaire dès que `beta * x > threshold` (ch. 17) |
| `torch.softmax(z, dim=-1)`, `torch.log_softmax(z, dim=-1)` | softmax et log-softmax stables ; `nn.CrossEntropyLoss` attend les **logits**, pas les probabilités (ch. 17) |
| `torch.func.jacrev(lambda t: torch.softmax(t, dim=-1))(z)`, `torch.autograd.functional.jacobian(f, z)` | une Jacobienne calculée par autograd, pour vérifier `softmax_derivative` (ch. 17) |

## Couches et modes (ch. 20)

| Code | Effet |
|---|---|
| `nn.Dropout(p=0.5)` | dropout inversé : un nouveau masque à chaque passe avant, facteur $1/(1-p)$ ; l'identité après `model.eval()` |
| `nn.BatchNorm1d(64)`, `nn.BatchNorm2d(C)` | batchnorm par feature (ou par canal) ; $\gamma$, $\beta$ appris ; `running_mean`, `running_var` en buffers ; `momentum=0.1` (convention inverse de Keras), `eps=1e-5` |
| `nn.LayerNorm(D)`, `nn.RMSNorm(D)` | normalisation sur les features de chaque exemple : même calcul en entraînement et en évaluation |
| `nn.Flatten()` | met à plat tout sauf la dimension des exemples (`start_dim=1`) ; `torch.flatten(x)` met **tout** à plat |
| `nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)`, `nn.MaxPool2d(2)`, `nn.AvgPool2d(2)` | convolution ($(k^2 C_{in} + 1)\,C_{out}$ paramètres) et pooling (aucun) ; détails au ch. 21 |
| `nn.ZeroPad2d(1)`, `nn.Upsample(scale_factor=2)`, `nn.Unflatten(1, (C, H, W))` | couches utilitaires sans paramètres ; `Upsample` répète les valeurs par défaut (`mode="nearest"`) |
| `x + sigma * torch.randn_like(x) if self.training else x` | une couche de bruit à écrire soi-même dans `forward` (aucune couche toute faite) |
| `model.train()`, `model.eval()`, `model.training` | changer de mode (dropout, batchnorm) ; `eval()` ne coupe pas l'autograd |
| `with torch.no_grad():` ; `torch.inference_mode()` | ne pas enregistrer le graphe : pour mesurer ou prédire |
| `print(model)` ; `model.named_parameters()`, `model.buffers()` | afficher les couches ; parcourir paramètres (avec leur nom) et buffers (statistiques glissantes) |

## Convolutions (ch. 21)

| Code | Effet |
|---|---|
| `nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)` | corrélation croisée sur `(N, C, H, W)` ; poids `(C_out, C_in/groups, k, k)` ; `padding="same"` seulement avec `stride=1` ; sortie $\lfloor (H + 2p - d(k-1) - 1)/s \rfloor + 1$ |
| `nn.Conv2d(C, C, 3, padding=1, groups=C)` puis `nn.Conv2d(C, C_out, 1)` | convolution séparable : par canal, puis 1 × 1 (MobileNet) |
| `nn.Conv1d(C_in, C_out, k)` | convolution 1D sur `(N, C, L)` (séquences, signaux) |
| `nn.MaxPool2d(2)`, `nn.AvgPool2d(3, stride=2, padding=1)` | `stride` vaut `kernel_size` par défaut ; padding au plus `kernel_size // 2` ; aucun paramètre |
| `nn.AdaptiveAvgPool2d(1)` | pooling moyen global : `(N, C, H, W)` → `(N, C, 1, 1)`, quelle que soit la taille de l'image |
| `nn.ConvTranspose2d(C_in, C_out, kernel_size=4, stride=2, padding=1)` | convolution transposée, poids `(C_in, C_out, k, k)` ; sortie $(H - 1)s - 2p + k$ (ici, le côté doublé) ; `output_padding` pour lever l'ambiguïté |
| `F.conv2d(x, w, b, stride, padding)`, `F.unfold(x, kernel_size)` | la version fonctionnelle (des filtres faits main) ; `unfold` est l'im2col de PyTorch, de forme `(N, C·k·k, L)` |
| `torchvision.models.resnet18(weights="DEFAULT")` ; `weights.transforms()` | un CNN pré-entraîné sur ImageNet, et le prétraitement qui va avec |
| `x.requires_grad_()` ; `loss.backward()` ; `x.grad.sign()` | gradient par rapport à l'**entrée** : visualisation de filtres, FGSM |
| `model.layer1.register_forward_hook(f)` | lire les cartes d'activation d'une couche pendant la passe avant |

## Dataset et DataLoader

| Code | Effet |
|---|---|
| `ds = torch.utils.data.TensorDataset(X, y)` | un `Dataset` à partir de deux tenseurs de même première dimension ; `ds[i]` rend `(X[i], y[i])` (ch. 20) |
| `class MyData(Dataset): __len__(self)`, `__getitem__(self, i)` | écrire son propre `Dataset` : sa longueur et l'exemple numéro `i` (ch. 20) |
| `loader = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))` | des mini-batches mélangés à chaque epoch, dans un ordre reproductible (ch. 20) |
| `DataLoader(..., drop_last=True)` | jeter le dernier mini-batch incomplet (une `BatchNorm1d` refuse en entraînement un mini-batch d'un seul exemple) (ch. 20) |
| `for X, y in loader:` ; `X, y = next(iter(loader))` | parcourir les mini-batches ; en prendre un seul pour vérifier les formes (ch. 20) |
| `torch.utils.data.random_split(ds, [50_000, 10_000], generator=g)`, `Subset(ds, idx)` | découper un dataset en entraînement et validation, ou en garder une partie (ch. 20) |

## Boucle d'entraînement

| Code | Effet |
|---|---|
| `loss = torch.nn.functional.cross_entropy(logits, y)` | la loss des classifieurs : la moyenne de $-\ln \mathrm{softmax}(\text{logits})_y$, en **nats** ; elle attend des logits, pas des probabilités (ch. 6, détails au ch. 20) |
| `torch.nn.CrossEntropyLoss(label_smoothing=0.1)` | la même loss avec une cible lissée (ch. 6) |
| `torch.exp(loss)` | la perplexité d'un modèle de langage, si `loss` est la cross-entropy moyenne par token (ch. 6) |
| `model.train()` ; `for X, y in loader:` `X, y = X.to(device), y.to(device)` ; `logits = model(X)` ; `loss = loss_fn(logits, y)` ; `opt.zero_grad()` ; `loss.backward()` ; `opt.step()` | un pas d'entraînement, dans cet ordre (ch. 20) |
| `model.eval()` ; `with torch.no_grad(): logits = model(X)` ; puis `model.train()` | mesurer la validation sans dropout ni mise à jour des statistiques de batchnorm (ch. 20) |
| `(logits.argmax(dim=1) == y).float().mean().item()` | l'accuracy d'un mini-batch ; `.item()` rend un nombre Python (ch. 20) |
| `nn.MSELoss()`, `nn.BCEWithLogitsLoss()`, `nn.CrossEntropyLoss()` | régression ; un logit par sortie binaire ou par label (labels réels 0 ou 1) ; $K$ logits et des labels entiers `(N,)` (ch. 20) |

## Optimiseurs et plannings (ch. 19)

| Code | Effet |
|---|---|
| `opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)` | l'optimiseur par défaut des projets récents ; créé **une fois**, avant la boucle (ses tampons portent la mémoire) |
| `torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)` | SGD avec momentum de Nesterov ; `dampening` doit rester à 0 |
| `torch.optim.Adam(params, weight_decay=wd)` | weight decay **couplé** (ajouté au gradient) ; `decoupled_weight_decay=True` (depuis 2.7) le rend identique à AdamW |
| `torch.optim.Adadelta(params)` | `lr=1.0` par défaut : c'est un multiplicateur, ne pas le baisser à 0,01 |
| `sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)` ; `sched.step()` | un planning ; `sched.step()` après `opt.step()`, à chaque epoch (ou à chaque pas, selon le planning) |
| `ExponentialLR(opt, gamma)`, `CosineAnnealingLR(opt, T_max)`, `LinearLR`, `SequentialLR`, `OneCycleLR`, `ReduceLROnPlateau(opt)` | les plannings courants ; `ReduceLROnPlateau.step(val_loss)` prend la loss de validation |
| `transformers.get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_training_steps)` | warmup linéaire puis cosinus, compté en pas (Hugging Face) |
| `OneCycleLR(opt, max_lr=3e-3, epochs=E, steps_per_epoch=len(loader))` ; `sched.step()` à chaque **mini-batch** | monter puis descendre le learning rate sur tout l'entraînement ; compté en pas, pas en epochs (ch. 20) |
| `opt.param_groups[0]["lr"]` | lire (ou changer à la main) le learning rate courant |
| `norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` | clipping par la norme globale, **entre** `loss.backward()` et `opt.step()` ; rend la norme d'avant |

## Sauvegarder et recharger

| Code | Effet |
|---|---|
| `torch.save(model.state_dict(), "model.pt")` | sauvegarder les paramètres **et** les buffers (statistiques glissantes) (ch. 20) |
| `model.load_state_dict(torch.load("model.pt"))` | recharger dans un modèle de même architecture ; `torch.load` ne lit que des tenseurs et des types simples par défaut (`weights_only=True`, depuis 2.6) (ch. 20) |
| `torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, "ckpt.pt")` | un checkpoint pour reprendre l'entraînement : sans l'état de l'optimiseur, Adam repart de moments nuls (ch. 20) |
| `copy.deepcopy(model.state_dict())` | garder en mémoire les meilleurs poids (early stopping) ; sans copie, le dictionnaire suit les poids qui changent (ch. 20) |
| `torch.load(path, map_location="cpu")` | recharger sur CPU un fichier sauvegardé depuis un GPU (ch. 20) |
| `torchvision.models.resnet18(weights="DEFAULT")`, `vgg16(weights=None)` | un modèle pré-entraîné (téléchargé une fois) ou la seule architecture, poids aléatoires (ch. 20) |
| `weights = MobileNet_V3_Small_Weights.DEFAULT` ; `mobilenet_v3_small(weights=weights)` ; `weights.transforms()` ; `weights.meta["categories"]` | un modèle pré-entraîné avec **son** prétraitement (redimensionnement, normalisation ImageNet) et les noms de ses 1 000 classes (ch. 20) |
| `with torch.device("meta"): model = vgg16(weights=None)` | construire une architecture sans allouer ses poids (compter les paramètres, lire les formes) (ch. 20) |
| `{"generator": g.get_state(), "torch_rng": torch.get_rng_state()}` ; `g.set_state(...)`, `torch.set_rng_state(...)` | sauvegarder aussi les générateurs aléatoires (mélange du `DataLoader`, dropout) pour une reprise identique au bit près (ch. 20) |
| `torch.use_deterministic_algorithms(True)` | exiger des opérations déterministes (ou une erreur), pour la reproductibilité sur GPU (ch. 20) |

