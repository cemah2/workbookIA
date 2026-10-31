# Cheatsheet PyTorch

> Aide-mémoire rempli au fil des chapitres (5 et 10 à 29). Une ligne = une commande utile + ce qu'elle fait.

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

## Réseaux récurrents (ch. 22)

| Code | Effet |
|---|---|
| `nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=False)` | entrée `(T, N, D)`, ou `(N, T, D)` avec `batch_first=True` ; rend `output, (h_n, c_n)` ; `nn.RNN` et `nn.GRU` rendent `output, h_n` |
| `output` ; `h_n`, `c_n` | `output` : tous les pas de la dernière couche, `(N, T, n_dir·H)` avec `n_dir = 2` si bidirectionnel (1 sinon) ; `h_n` : le dernier pas de chaque couche et de chaque sens, toujours `(couches·n_dir, N, H)`, indice `couche·n_dir + sens` ($D$ reste le nombre de features par pas) |
| `out, (h, c) = lstm(x)` puis `out, (h, c) = lstm(x_next, (h, c))` | passer l'état d'un morceau de séquence au suivant ; `(h.detach(), c.detach())` pour ne pas rétropropager au-delà (truncated BPTT) |
| `lstm.weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0` ; `_reverse`, `_l1` | poids `(4H, D)` et `(4H, H)`, blocs de lignes dans l'ordre `i, f, g, o` (GRU : `r, z, n`) ; sens arrière, deuxième couche |
| `with torch.no_grad(): lstm.bias_ih_l0[H:2 * H].fill_(1.0); lstm.bias_hh_l0[H:2 * H].zero_()` | biais total de la porte d'oubli à 1 : la porte voit `bias_ih + bias_hh` (PyTorch ne le fait pas ; à répéter pour `_l1`, `_l2`…) |
| `x.permute(1, 0, 2)` | passer de `(N, T, D)` à `(T, N, D)` ; jamais `reshape` |
| `nn.Embedding(V, E)` | table de vecteurs indexée par des ids entiers : `(N, T)` → `(N, T, E)` ; initialisation $\mathcal{N}(0, 1)$ |
| `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` | entre `loss.backward()` et `optimizer.step()` ; rend la norme **avant** clipping |
| `logits[:, -1] / T` ; `torch.multinomial(torch.softmax(z, -1), 1)` | échantillonner le token suivant avec une température ; `torch.topk(z, k)` pour un top-k |
| `nn.CrossEntropyLoss()(logits.reshape(-1, V), y.reshape(-1))` | loss many-to-many : aplatir `(N, T, V)` et `(N, T)` (ou `logits.transpose(1, 2)`, `(N, V, T)`) |

## Du dataset au modèle sauvegardé (ch. 23)

| Code | Effet |
|---|---|
| `"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"` | choisir le device : GPU NVIDIA (ou AMD), GPU Apple, sinon CPU |
| `t.reshape(4, -1)` ; `t.view(4, 3)` ; `x.permute(0, 3, 1, 2)` ; `.contiguous()` | regrouper autrement (même ordre) ; idem sans copie, refusé si la mémoire ne s'y prête pas (aplatir après `permute`) ; NHWC → NCHW (change l'ordre) ; recopier dans le nouvel ordre |
| `train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)` | une validation stratifiée prise dans le train |
| `torch.from_numpy(((X / 255.0 - mu) / sigma).astype(np.float32)).unsqueeze(1)` | des images `uint8` `(N, 28, 28)` en `float32` `(N, 1, 28, 28)`, normalisées avec $\mu$ et $\sigma$ du train |
| `F.one_hot(y, num_classes=10)` ; `p.argmax(dim=1)` | un one-hot (toujours avec `num_classes`) ; revenir à l'indice de classe |
| `nn.Sequential(nn.Flatten(), nn.Linear(784, 784), nn.ReLU(), nn.Linear(784, 10))` ; `nn.LazyLinear(784)` | le MLP du livre, qui sort des logits ; une couche dense qui découvre sa taille d'entrée au premier passage |
| `model.eval()` ; `with torch.inference_mode(): probs = model(x).softmax(dim=1)` | prédire : mode évaluation, pas de graphe, puis des probabilités |
| `LambdaLR(opt, lambda e: max(0.1, 1 - 0.01 * e))` | un planning écrit à la main : la fonction rend un **facteur** du learning rate de départ |
| `for p in backbone.parameters(): p.requires_grad_(False)` ; `backbone.fc = nn.Linear(backbone.fc.in_features, 10)` | geler un modèle pré-entraîné et lui donner une nouvelle tête, seule entraînée |
| `safetensors.torch.save_file(model.state_dict(), "m.safetensors")` ; `load_file(path)` | sauver et charger des poids sans pickle (des tenseurs contigus seulement) |
| `json.dump({"mean": mu, "std": sigma, "classes": CLASSES, "n_hidden": 784}, f)` | sauver à côté des poids le prétraitement et les arguments de l'architecture |
| `breakpoint()` | arrêter l'exécution et ouvrir le débogueur `pdb` (`p x.shape`, `c` pour continuer) |

## Améliorer, chercher, CNN et RNN (ch. 24)

| Code | Effet |
|---|---|
| `nn.Dropout(0.2)` après l'activation ; `nn.Dropout2d(p)` | mettre à zéro 20 % des valeurs (un masque neuf à chaque passe avant) ; éteindre des canaux entiers |
| `with torch.no_grad(): W.copy_(torch.renorm(W, 2, 0, 3.0))` après `optimizer.step()` | max-norm : chaque ligne de `W` (un neurone) ramenée à une norme d'au plus 3 |
| `class Net(ClassifierMixin, BaseEstimator)` : `__init__` range ses arguments, `fit` rend `self` | un réseau PyTorch utilisable par `cross_val_score` et `GridSearchCV` (`clone`, `get_params`) |
| `Pipeline([("scale", MinMaxScaler()), ("net", Net())])` ; `{"net__lr": [1e-3, 3e-3]}` | le prétraitement réappris dans chaque fold ; une grille sur un paramètre d'étape (`étape__paramètre`) |
| `RandomizedSearchCV(pipe, {"net__lr": loguniform(1e-4, 1e-1)}, n_iter=20)` | une recherche au hasard, learning rate sur une échelle logarithmique (`scipy.stats.loguniform`) |
| `nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)`, `nn.BatchNorm2d(16)`, `nn.ReLU()` | une convolution de stride 2 (taille divisée par 2), sans biais avant la batchnorm, puis l'activation |
| `nn.AdaptiveAvgPool2d(1)`, `nn.Flatten()`, `nn.Linear(C, 10)` | la tête d'un CNN moderne : une valeur par canal, puis une couche linéaire |
| `v2.Compose([v2.ToImage(), v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])` | l'augmentation d'un `Dataset` de train CIFAR-10 (la validation garde ToImage, ToDtype, Normalize) |
| `class Stream(IterableDataset): def __iter__(self): ... yield x, y` | un flux d'exemples fabriqués à la volée ; c'est le code qui fixe la longueur d'une epoch |
| `scaler = MinMaxScaler().fit(train.reshape(-1, 1))` ; `scaler.inverse_transform(pred.reshape(-1, 1)).ravel()` | mettre une série à l'échelle (train seul) ; revenir aux unités d'origine avant la RMSE |
| `output, (h_n, c_n) = lstm(x)` ; `output[:, -1]` ; `h_n[-1]` | toute la séquence de la dernière couche ; son dernier pas ; l'état final de la dernière couche (le même, à un seul sens) |
| `state = tuple(s.detach() for s in state)` ; `output, state = lstm(xb, state)` | porter l'état d'un batch au suivant sans les gradients (BPTT tronquée) |
| `nn.Linear(H, k)(output)` | la même couche dense à chaque pas : `(N, T, H)` → `(N, T, k)` (le `TimeDistributed` de Keras) |
| `nn.Embedding(V, d)(ids)` | des indices `(N, T)` → des vecteurs `(N, T, d)` : une lecture de table |
| `torch.relu(self.f(x) + self.shortcut(x))` | un bloc résiduel dans `forward()` (raccourci `nn.Identity()` ou convolution 1 × 1) |
| `layer.requires_grad_(False)` ; `layer.eval()` | geler des poids ; figer aussi les statistiques d'une batchnorm |
| `with torch.autocast(device_type="cpu", dtype=torch.bfloat16):` ; `torch.amp.GradScaler("cuda")` | précision mixte (bfloat16 sur CPU ; float16 avec mise à l'échelle de la loss sur GPU) |
| `model = torch.compile(model)` | compiler le modèle à son premier appel (lent la première fois ; compilateur C++ requis sur CPU) |

## Autoencodeurs et VAE (ch. 25)

| Code | Effet |
|---|---|
| `self.encoder = nn.Sequential(...)` ; `self.decoder = nn.Sequential(...)` ; `forward: return self.decoder(self.encoder(x))` | un autoencodeur dont on peut appeler chaque moitié seule |
| `loss = F.binary_cross_entropy_with_logits(model(x), x)` | la cible est l'entrée ; des logits en sortie, `torch.sigmoid(model(x))` pour voir l'image |
| `noisy = (x + sigma * torch.randn_like(x)).clamp(0, 1)` ; `loss_fn(model(noisy), x)` | un débruiteur : bruit neuf à chaque mini-batch, cible propre |
| `nn.Upsample(scale_factor=2)` ; `nn.ConvTranspose2d(C, C2, 3, stride=2, padding=1, output_padding=1)` | doubler la hauteur et la largeur, sans ou avec paramètres |
| `mu, log_var = self.to_mu(h), self.to_log_var(h)` ; `z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)` | l'encodeur d'un VAE et la reparamétrisation |
| `kl = 0.5 * (mu**2 + log_var.exp() - 1 - log_var).sum() / len(x)` | la KL vers $\mathcal{N}(\mathbf{0}, \mathbf{I})$ par exemple, à ajouter à une reconstruction sommée sur les pixels |
| `with torch.no_grad(): images = torch.sigmoid(model.decoder(torch.randn(64, d)))` | générer : des codes tirés du prior, le décodeur seul |
| `torch.save(model.decoder.state_dict(), "decoder.pt")` | exporter le décodeur comme générateur autonome |
| `10 * torch.log10(1 / F.mse_loss(out, clean))` | le PSNR d'un débruitage (pixels dans $[0, 1]$) |

## GAN (ch. 27)

| Code | Effet |
|---|---|
| `opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))` ; idem pour `G` | deux optimiseurs, les réglages de Radford |
| `fake = G(z)` ; `loss_d = bce(D(real), ones) + bce(D(fake.detach()), zeros)` ; `opt_d.zero_grad(); loss_d.backward(); opt_d.step()` | l'étape du discriminateur : les faux détachés de $G$ |
| `loss_g = bce(D(fake), ones)` ; `opt_g.zero_grad(); loss_g.backward(); opt_g.step()` | l'étape du générateur : label 1 (loss non saturante), seul `opt_g` avance |
| `D.requires_grad_(False)` … `D.requires_grad_(True)` | geler $D$ explicitement pendant l'étape de $G$ (le `trainable = False` de Keras) |
| `bce = nn.BCEWithLogitsLoss()` et un `nn.Linear(..., 1)` en sortie de $D$ | des logits, pas de sigmoïde : stable numériquement |
| `nn.ConvTranspose2d(c, c2, 4, stride=2, padding=1)` ; `nn.Conv2d(c, c2, 4, stride=2, padding=1)` | doubler ou diviser par deux la taille (DCGAN) |
| `G.apply(init)` avec `nn.init.normal_(m.weight, 0.0, 0.02)` | l'initialisation de Radford |
| `G.eval()` ; `with torch.no_grad(): images = G(torch.randn(64, nz))` | générer, sans dépendre des autres $\mathbf{z}$ du mini-batch |
| `torch.autograd.grad(loss, list(G.parameters()))` | des gradients sans remplir les `.grad` (mesurer une norme) |
| `torch.save({"G": G.state_dict(), "D": ..., "opt_g": ..., "opt_d": ..., "gen": gen.get_state()}, path)` ; `torch.load(path, weights_only=True)` | le checkpoint complet d'un GAN |

## Optimiser une image, hooks et transfert de style (ch. 28)

| Code | Effet |
|---|---|
| `net, mean, std = wb.datasets.load_pretrained("squeezenet1_1")` ; ou `torchvision.models.vgg16(weights="DEFAULT").features.eval()` puis `p.requires_grad_(False)` | un réseau pré-entraîné gelé, en mode `eval()` |
| `h = net[i].register_forward_hook(lambda m, inp, out, name=name: acts.__setitem__(name, out))` ; … ; `h.remove()` | lire l'activation d'une couche pendant la passe (retirer le hook dans un `finally`) |
| `x = img.clone().requires_grad_(True)` ; `opt = torch.optim.Adam([x], lr=0.05)` | l'image devient le seul paramètre |
| `loss = -(net[:i + 1]((x - mean) / std)[:, c] ** 2).mean()` | monter l'activation du filtre `c` : on minimise l'opposé, sur une image normalisée |
| `with torch.no_grad(): x.clamp_(0, 1)` | garder les pixels dans la plage valide après chaque pas |
| `(g,) = torch.autograd.grad(loss, x)` ; `x += lr * g / (g.abs().mean() + 1e-8)` | un pas de deep dream à gradient normalisé |
| `flat = A.reshape(*A.shape[:-2], -1)` ; `G = flat @ flat.transpose(-1, -2) / (C * H * W)` | la matrice de Gram normalisée, avec ou sans axe de batch |
| `((x[..., 1:, :] - x[..., :-1, :]) ** 2).sum() + ((x[..., 1:] - x[..., :-1]) ** 2).sum()` | la variation totale |
| `torch.roll(x, shifts=(dy, dx), dims=(2, 3))` ; `F.interpolate(x, size=s, mode="bilinear", align_corners=False)` | décalages aléatoires et octaves d'une visualisation |

## Environnements de RL (ch. 26)

| Code | Effet |
|---|---|
| `env = gym.make("FrozenLake-v1")` ; `env.observation_space.n`, `env.action_space.n` | un environnement de Gymnasium et ses nombres d'états et d'actions (espaces discrets) |
| `state, info = env.reset(seed=int(rng.integers(2**31)))` | commencer un épisode ; la graine doit être un `int` Python (un `np.int64` est refusé) |
| `state, reward, terminated, truncated, info = env.step(action)` | jouer une action ; `terminated` : état terminal (pas d'amorçage), `truncated` : épisode coupé (on amorce quand même) |
| `env.unwrapped.P[s][a]` | le modèle de FrozenLake : une liste de `(probabilité, état suivant, récompense, terminated)` |
| `np.flatnonzero(info["action_mask"])` | les actions légales du morpion de `wb.envs` |

## Données publiques (ch. 29)

| Code | Effet |
|---|---|
| `datasets.load_dataset(nom, split="test", revision=sha)` | un dataset du Hub de Hugging Face, à une révision figée |
| `ds.with_format("numpy")[:]["image"]` | les colonnes en tableaux NumPy |
| `torchvision.datasets.FashionMNIST(root, train=True, download=True)` | un dataset de torchvision, téléchargé une fois dans `root` |
| `h = hashlib.sha256()` ; `for bloc in iter(lambda: f.read(1 << 20), b""): h.update(bloc)` ; `h.hexdigest()` | la somme SHA-256 d'un fichier lu par morceaux, à noter dans la data card |

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

