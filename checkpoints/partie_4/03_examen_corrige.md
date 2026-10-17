# Checkpoint IV · Corrigé détaillé et barème de l'examen blanc

> À lire **après** l'examen, et après la partie B du notebook (vérification automatique des réponses chiffrées). Pour chaque question : la réponse, la démarche, le barème, les erreurs fréquentes et les exercices à refaire si tu as eu moins de la moitié des points. La solution exécutée de la question de code est dans `03_examen_solutions.ipynb`.

## Règles de notation

- **Barème par sous-question**, indiqué entre crochets. Une réponse juste sans démarche, quand la question en demande une, rapporte la moitié des points.
- **Méthode juste, erreur de calcul** : la moitié des points de la sous-question. **Erreur reportée** : si une réponse fausse est réutilisée correctement plus loin (en CP4.2 et CP4.3 surtout), la suite n'est pas pénalisée une seconde fois.
- **Arrondis** : une décimale de trop ou de moins coûte au plus 0,05 point par question ; un arrondi fait trop tôt, qui change la dernière décimale demandée, aussi.
- **Estimations** (CP4.6) : toute valeur qui s'arrondit à la réponse attendue rapporte tous les points ; un ordre de grandeur juste mais une valeur fausse, la moitié.
- **Questions rédigées** (CP4.2 i, CP4.3 h, CP4.4 c (le pourquoi) et h, CP4.5 a, b et d, CP4.7 b, CP4.10, CP4.11, CP4.12 f, CP4.14) : les critères sont listés ; une idée juste formulée autrement compte autant.

---

## CP4.1 — Questions flash 🧠 · 1,5 point

Six items à 0,25 point, sans justification demandée.

a) **3 523**. Chaque couche dense a $n_{\text{in}} \times n_{\text{out}}$ poids et $n_{\text{out}}$ biais : $(20 \times 64 + 64) + (64 \times 32 + 32) + (32 \times 3 + 3) = 1\,344 + 2\,080 + 99$.
b) **« (16, 32) »** : une ligne par exemple, une colonne par neurone de la couche.
c) **[0 ; 0,5 ; 1 ; −0,2]** : ReLU(−1,5) = max(0 ; −1,5) = 0 ; $\sigma(0) = \frac{1}{1 + e^0} = \frac12$ ; $\tanh'(0) = 1 - \tanh^2(0) = 1$ ; la leaky ReLU de pente 0,1 rend $0{,}1 \times (-2)$.
d) **0,787** : $\frac{e^2}{e^2 + e^0 + e^0} = \frac{7{,}389}{9{,}389}$.
e) **128** : $\gamma$ et $\beta$, un de chaque par feature ($2 \times 64$). `running_mean` et `running_var` sont des **buffers** : ils sont dans le `state_dict`, mais pas dans `parameters()`, et l'optimiseur n'y touche pas.
f) **224** : 8 filtres de $3 \times 3 \times 3$ poids (le filtre couvre les 3 canaux d'entrée), plus un biais par filtre : $8 \times 27 + 8$.

**Erreurs fréquentes** : oublier les biais (3 424 en a), 216 en f) ; inverser la forme en b) ; $\tanh'(0) = 0$ ; compter les statistiques glissantes comme des paramètres (256).
**Remédiation** : 16.1, 16.3, 16.12, 17.2, 17.6, 20.1, 20.Q6, 20.20.

## CP4.2 — Passe avant, rétropropagation et un pas de SGD ✏️ · 3 points

a) **[0,3]** $z_1 = 2 \times 1 + (-1) \times 0{,}5 - 0{,}5 = 1$ et $z_2 = 2 \times (-0{,}5) + (-1) \times 1 + 0 = -2$ : **[1 ; −2]**. La colonne $k$ de $\mathbf{W}^{(1)}$ porte les poids du neurone $k$.
b) **[0,4]** ReLU : $\mathbf{h} = (1 ;\ 0)$. $\hat y = 1 \times 0{,}75 + 0 \times (-1) + 0{,}5 = 1{,}25$ ; $L = \frac12 (1{,}25 - 2)^2 = 0{,}28125$ : **[1,25 ; 0,28125]**.
c) **[0,3]** $\delta = \hat y - y =$ **−0,75** (la sortie est trop basse : le gradient est négatif, et la descente fera monter $\hat y$).
d) **[0,4]** $\partial L / \partial W^{(2)}_k = \delta \times h_k$ et $\partial L / \partial b^{(2)} = \delta$ : **[−0,75 ; 0 ; −0,75]**. Le neurone 2 sort 0 : son poids de sortie ne reçoit aucun gradient.
e) **[0,4]** $\partial L / \partial \mathbf{h} = \delta\,\mathbf{W}^{(2)} = (-0{,}5625 ;\ 0{,}75)$, puis le facteur ReLU$'(z)$ : 1 pour $z_1 = 1 > 0$, 0 pour $z_2 = -2$ : **[−0,5625 ; 0]**.
f) **[0,4]** $\partial L / \partial \mathbf{W}^{(1)} = \mathbf{x}^\top \boldsymbol{\delta}^{(1)}$ (un produit extérieur, rangé comme $\mathbf{W}^{(1)}$) : **[[−1,125 ; 0] ; [0,5625 ; 0]]**.
g) **[0,3]** $\mathbf{W}^{(1)} - 0{,}1 \times \partial L/\partial \mathbf{W}^{(1)}$ : **[[1,1125 ; −0,5] ; [0,44375 ; 1]]**.
h) **[0,3]** Les autres paramètres font aussi leur pas : $b^{(1)}_1 = -0{,}44375$, $W^{(2)}_1 = 0{,}825$, $b^{(2)} = 0{,}575$ (les autres ne bougent pas). Alors $z_1 = 2 \times 1{,}1125 - 0{,}44375 - 0{,}44375 = 1{,}3375$, $\hat y = 1{,}3375 \times 0{,}825 + 0{,}575 \approx 1{,}6784$ et $L = \frac12 (1{,}6784 - 2)^2 \approx$ **0,05170** : la loss a été divisée par plus de 5.
i) **[0,2]** Le neurone 2 a une somme négative pour cet exemple : ReLU$'(z_2) = 0$, son delta est nul, et **tous** les gradients qui le traversent aussi : ses poids d'entrée, son biais et son poids de sortie (0,1). Si c'était le cas pour tous les exemples du dataset, le neurone ne recevrait plus jamais de gradient : un **neurone mort**, qui ne reviendra pas à la vie, d'où les leaky ReLU, une bonne initialisation et un learning rate raisonnable (0,1).

**Erreurs fréquentes** : lire $\mathbf{W}^{(1)}$ par lignes au lieu de colonnes ; oublier le ½ de la loss (δ = −1,5) ; prendre $z$ au lieu de $h$ dans le gradient de $\mathbf{W}^{(2)}$ ; oublier le facteur ReLU$'(z)$ ; ranger $\partial L / \partial \mathbf{W}^{(1)}$ transposé ; monter le gradient au lieu de le descendre ; ne mettre à jour que $\mathbf{W}^{(1)}$ avant de recalculer la loss.
**Remédiation** : 16.2, 18.4, 18.5, 18.7, 18.15, 18.18, 18.Q10.

## CP4.3 — Momentum, Adam et AdamW ✏️ · 2 points

a) **[0,2]** $b_1 = g_1 = 2$, puis $b_2 = 0{,}9 \times 2 + 1 =$ **2,8**.
b) **[0,2]** $w_1 = 1 - 0{,}1 \times 2 = 0{,}8$, puis $w_2 = 0{,}8 - 0{,}1 \times 2{,}8 =$ **0,52**.
c) **[0,3]** $m_1 = 0{,}1 \times 2 = 0{,}2$ et $v_1 = 0{,}001 \times 4 = 0{,}004$ ; corrigés : $\hat m_1 = 0{,}2 / 0{,}1 = 2$ et $\hat v_1 = 0{,}004 / 0{,}001 = 4$. Le pas vaut $0{,}01 \times 2 / \sqrt4 = 0{,}01$ : **0,99000**. Au premier pas, Adam avance de $\eta$ dans le sens opposé au signe du gradient, quelle que soit sa taille.
d) **[0,3]** $m_2 = 0{,}9 \times 0{,}2 + 0{,}1 \times 1 = 0{,}28$ et $v_2 = 0{,}999 \times 0{,}004 + 0{,}001 \times 1 = 0{,}004996$ ; corrigés par $1 - 0{,}9^2 = 0{,}19$ et $1 - 0{,}999^2 = 0{,}001999$ : **[1,474 ; 2,499]**.
e) **[0,3]** $w_2 = 0{,}99 - 0{,}01 \times \frac{1{,}47368}{\sqrt{2{,}49925}} = 0{,}99 - 0{,}00932 \approx$ **0,98068**.
f) **[0,3]** AdamW : $w \leftarrow 1 \times (1 - 0{,}01 \times 0{,}1) = 0{,}999$, puis le même pas d'Adam qu'en c), $0{,}01$ : **0,98900**.
g) **[0,2]** Adam couplé : le gradient devient $g_1 + \lambda w_0 = 2{,}1$ ; au premier pas, Adam divise $\hat m_1 = 2{,}1$ par $\sqrt{\hat v_1} = 2{,}1$ : le pas vaut encore $\eta$, et **0,99000**, exactement comme sans weight decay.
h) **[0,2]** Couplée, la pénalité passe dans le gradient, que la normalisation d'Adam divise par sa propre taille : au premier pas, elle disparaît complètement ; ensuite, un poids qui a de grands gradients n'est presque pas pénalisé (0,1). AdamW applique le weight decay à part, avec la même force pour tous les poids : c'est le régularisateur qu'on croit appliquer, d'où AdamW par défaut dans les projets récents (0,1).

**Erreurs fréquentes** : un facteur $(1 - \mu)$ dans la vitesse (une autre convention, pas celle de PyTorch) ; oublier la correction de biais (le premier pas devient 0,0316) ; corriger au deuxième pas par $1 - \beta$ au lieu de $1 - \beta^2$ ; appliquer le weight decay d'AdamW après le pas (0,98901 au lieu de 0,98900) ou sans le learning rate.
**Remédiation** : 19.2, 19.5, 19.6, 19.16, 19.18, 19.E2.

## CP4.4 — Dropout et batchnorm ✏️ · 1,5 point

a) **[0,25]** La première valeur ($u = 0{,}1 < 0{,}25$) est mise à zéro ; les trois autres sont gardées et divisées par $1 - p = 0{,}75$ : **[0 ; 5,333 ; 8 ; 10,667]**.
b) **[0,1]** En évaluation, le dropout inversé ne fait **rien** : **[2 ; 4 ; 6 ; 8]**.
c) **[0,15]** **6** (0,05), contre 5 pour $\mathbf{x}$. Le dropout inversé conserve chaque valeur **en espérance** : sur tous les masques possibles, $\mathbb{E}[x_i m_i / (1-p)] = x_i$. Un seul tirage ne la conserve pas : ici, c'est la plus petite valeur qui a été coupée (0,1).
d) **[0,2]** $\mu = \frac{2 + 6 + 8 + 12}{4} = 7$ ; écarts $-5, -1, 1, 5$ ; $\sigma^2 = \frac{25 + 1 + 1 + 25}{4} = 13$ : **[7 ; 13]**.
e) **[0,25]** $0{,}5 \times \frac{12 - 7}{\sqrt{13}} - 1 = 0{,}5 \times 1{,}3868 - 1 \approx$ **−0,3066**.
f) **[0,25]** $\mu^{\text{run}} = 0{,}9 \times 0 + 0{,}1 \times 7 = 0{,}7$ ; variance non biaisée $52 / 3 \approx 17{,}333$, d'où $\sigma^{2,\text{run}} = 0{,}9 \times 1 + 0{,}1 \times 17{,}333 \approx 2{,}6333$ : **[0,7 ; 2,6333]**.
g) **[0,2]** $0{,}5 \times \frac{7 - 0{,}7}{\sqrt{2{,}6333}} - 1 = 0{,}5 \times 3{,}8823 - 1 \approx$ **0,9411**.
h) **[0,1]** En évaluation, la couche normalise avec ses statistiques **glissantes**, qui n'ont vu qu'un seul mini-batch : elles sont encore très près de leur départ (0 et 1), loin de 7 et 13. Il faut de nombreux mini-batches d'entraînement (des dizaines à des centaines, avec $\rho = 0{,}1$) avant qu'elles se rapprochent des statistiques des données.

**Erreurs fréquentes** : garder les valeurs avec $u < p$ ; oublier la division par $1 - p$ (ou diviser par $p$) ; multiplier par $1 - p$ en évaluation (le dropout « du livre ») ; la variance non biaisée pour normaliser, ou la biaisée pour la statistique glissante ; inverser $\rho$ et $1 - \rho$ ; normaliser en évaluation avec les statistiques du mini-batch.
**Remédiation** : 20.3, 20.4, 20.16, 20.22, 20.E1, 20.Q5.

## CP4.5 — Le gradient à travers des sigmoïdes empilées ∂ · 1 point

a) **[0,3]** $\sigma(z) = (1 + e^{-z})^{-1}$, donc $\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \times \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)\,(1 - \sigma(z))$, car $\frac{e^{-z}}{1 + e^{-z}} = 1 - \sigma(z)$ (0,15). Avec $s = \sigma(z) \in\ ]0, 1[$ : $s(1 - s) = \frac14 - \left(s - \frac12\right)^2 \le \frac14$, avec égalité en $s = \frac12$, c'est-à-dire $z = 0$ (0,15).
b) **[0,2]** Pour chaque couche, $\frac{\partial a_k}{\partial a_{k-1}} = \sigma'(z_k)\,w_k$ ; la règle de la chaîne multiplie ces facteurs : $\dfrac{\partial L}{\partial a_0} = \dfrac{\partial L}{\partial a_L}\displaystyle\prod_{k=1}^{L} w_k\,\sigma'(z_k)$.
c) **[0,2]** Chaque facteur vérifie $|w_k\,\sigma'(z_k)| \le 1 \times \frac14$, donc le rapport vaut au plus $\left(\frac14\right)^L$ : $\left(\frac14\right)^9 \approx 3{,}8 \times 10^{-6}$ et $\left(\frac14\right)^{10} \approx 9{,}5 \times 10^{-7}$ : **L = 10**. Une dizaine de couches suffit pour que les premières ne reçoivent presque plus rien, et c'est une borne optimiste : dès qu'un neurone sature, son $\sigma'$ est bien plus petit que $\frac14$.
d) **[0,3]** Avec une ReLU, le facteur vaut **1** pour un neurone actif (0 s'il est éteint) : le produit ne se rétrécit plus à chaque couche (0,1). Les **connexions résiduelles** (ch. 20) : chaque bloc calcule $a_k = a_{k-1} + F(a_{k-1})$, dont la dérivée vaut $1 + F'$ : le produit de b) gagne un terme « 1 » par couche, une voie directe par laquelle le gradient remonte sans s'éteindre (0,2). (Accepté aussi : une initialisation qui conserve la variance, ou une normalisation, à condition d'expliquer son effet sur les facteurs.)

**Erreurs fréquentes** : borner $\sigma'$ par $\frac12$ (L = 20) ; oublier les $w_k$ dans le produit ; affirmer que la ReLU « supprime » le problème (un neurone éteint donne 0, d'où les ReLU mortes).
**Remédiation** : 17.4, 17.23, 18.E3, 20.30, 20.E5.

## CP4.6 — Fermi : paramètres et mémoire 🧮 · 1 point

a) **1,9** million : $784 \times 1\,024 + 1\,024 \times 1\,024 + 1\,024 \times 10 \approx 0{,}80 + 1{,}05 + 0{,}01$ million, plus 2 058 biais (1 863 690 en tout).
b) **30** Mo : quatre tableaux de la taille des paramètres (les paramètres, leurs gradients, $m$ et $v$) de 4 octets : $1{,}86 \times 10^6 \times 16 \approx 29{,}8 \times 10^6$ octets.
c) **2** : quatre tableaux contre deux.
d) **2 350** : $\lceil 60\,000 / 256 \rceil = 235$ mises à jour par epoch (234 mini-batches complets et un de 96 images), fois 10.

**Erreurs fréquentes** : compter les paramètres seuls (7 Mo) ; un seul moment pour Adam ; le mégaoctet de $2^{20}$ octets (28) ; oublier le dernier mini-batch (2 340). Les activations gardées pour la passe arrière s'ajoutent à b) et grandissent avec la taille du mini-batch : pour 256 exemples, environ 1 Mo par tableau gardé (1 024 × 256 × 4 octets), soit 2 Mo par couche cachée qui garde $\mathbf{z}$ et $\mathbf{h}$, peu de chose ici, mais l'essentiel de la mémoire d'un grand réseau convolutif.
**Remédiation** : 16.8, 19.9, 19.Q8.

## CP4.7 — Quatre bugs dans une boucle d'entraînement 🐛 · 2 points

a) **[1]** **D, I, K et M** (0,25 par bonne lettre, −0,25 par lettre en trop, sans descendre sous 0). (J) à la place de (K) désigne le même bug, l'ordre de `backward()` et de `zero_grad()` : tous les points si b) remet `zero_grad()` avant `backward()`.
b) **[1]** 0,25 par ligne : la correction (0,1) et le symptôme (0,15).

| Ligne | L'erreur | La correction | Le symptôme |
|---|---|---|---|
| (D) | l'optimiseur est recréé à chaque epoch | le créer **une fois**, avant la boucle des epochs | AdamW perd ses moments et son compteur $t$ à chaque epoch : ses premiers pas, corrigés du biais, valent à peu près $\eta$ par poids, et la loss saute au début de chaque epoch ; un planning du learning rate serait remis à zéro de même |
| (I) | une softmax avant `CrossEntropyLoss`, qui applique déjà `log_softmax` | `loss = loss_fn(model(X), y)`, sur les logits | les probabilités sont prises pour des logits : la loss ne peut pas descendre sous $\ln(1 + 9/e) \approx 1{,}46$ (dix classes), les gradients sont écrasés, l'apprentissage est très lent |
| (K) | `zero_grad()` entre `backward()` et `step()` | `optimizer.zero_grad()` **avant** `loss.backward()` (ou après `step()`) | les gradients sont effacés (mis à `None`) juste avant d'être utilisés : `step()` ignore les paramètres sans gradient, les poids ne bougent plus, la loss reste au niveau du hasard |
| (M) | `total += loss` additionne des tenseurs qui gardent leur graphe | `total += loss.item() * len(y)` | `total` devient un tenseur qui garde l'historique de l'autograd de chaque mini-batch (ses nœuds `grad_fn`) : une fuite de mémoire lente, quelques kilo-octets par pas ici, car `backward()` a déjà libéré les activations ; elle devient grave quand on additionne la loss **avant** `backward()` ou avec `retain_graph=True` : chaque mini-batch garde alors tout son graphe et ses activations, jusqu'à l'erreur « out of memory » sur GPU |

Les autres lignes sont justes : (E) remet le mode entraînement à chaque epoch, après le `model.eval()` de la fin de la précédente ; (O) et (P) évaluent sans gradient, avec une moyenne par exemple ; (Q) affiche un `float`.

**Erreurs fréquentes** : désigner (N) ou (E) (les modes sont bien placés) ; corriger (I) en appliquant `log` à la softmax sans changer de loss (`NLLLoss` le permettrait, mais `CrossEntropyLoss` attend des logits) ; placer `zero_grad()` après `backward()`.
**Remédiation** : 20.23, 20.24, 20.E3, 18.E5.

## CP4.8 — Quatre paires de courbes 📈 · 1 point

0,25 par lettre.

a) **A**, un learning rate beaucoup trop grand : la loss d'entraînement ne baisse pas, elle saute autour de 2 et revient près de $\ln 10 \approx 2{,}3$, et la validation fait des pics (3,2 à l'epoch 5). (Ici, SGD avec $\eta = 1{,}5$.)
b) **C**, de l'overfitting : la loss d'entraînement tombe presque à 0, la validation atteint son minimum vers l'epoch 8 (environ 0,58), puis remonte doucement jusqu'à 0,69. (Adam sur 300 images seulement.)
c) **B**, un learning rate beaucoup trop petit : les deux losses partent de $\ln 10 \approx 2{,}30$ et n'ont perdu que 0,06 en 30 epochs, sans un seul saut. (SGD avec $\eta = 2 \times 10^{-4}$.)
d) **D**, un entraînement sain, régularisé : la validation suit l'entraînement, reste même en dessous au début (la loss d'entraînement est mesurée avec le dropout et pendant l'epoch), et baisse encore à l'epoch 30 (0,205). (Adam, dropout 0,5, 6 000 images.)

(E) et (F) ne servent pas : une fuite donnerait des losses de validation presque nulles et collées à l'entraînement ; `model.eval()` oublié laisserait l'entraînement progresser normalement, avec une validation bruitée au niveau de l'entraînement.

**Erreurs fréquentes** : prendre ① pour `eval()` oublié (mais ici l'entraînement lui-même n'avance pas) ; prendre ④ pour de l'overfitting (sa validation ne remonte pas) ; prendre ③ pour un entraînement sain qui débute (lire l'échelle : 2,3, c'est le hasard sur 10 classes).
**Remédiation** : 18.22, 19.23, 20.25.

## CP4.9 — Une epoch d'entraînement avec AdamW et clipping 🔨 · 2 points

La solution complète, exécutée et comparée à `torch.optim.AdamW` et `clip_grad_norm_` (écart de l'ordre de $10^{-16}$), est dans `03_examen_solutions.ipynb`. L'essentiel :

```python
def clip_by_global_norm(grads, max_norm):
    norm = float(np.sqrt(sum(np.sum(g ** 2) for g in grads)))
    coef = min(1.0, max_norm / (norm + 1e-6))
    return [g * coef for g in grads], norm          # new arrays: grads stay intact


def adamw_step(params, grads, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    beta1, beta2 = betas
    if not state:
        state["t"], state["m"], state["v"] = 0, [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
    state["t"] += 1
    t = state["t"]
    for p, g, m, v in zip(params, grads, state["m"], state["v"]):
        p *= 1 - lr * weight_decay                  # decoupled weight decay, in place
        m *= beta1
        m += (1 - beta1) * g
        v *= beta2
        v += (1 - beta2) * g * g
        p -= lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)


def train_epoch(params, X, y, state, lr, batch_size, max_norm, rng):
    order = rng.permutation(len(X))
    total = 0.0
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        loss, grads = loss_and_grads(params, X[idx], y[idx])
        grads, _ = clip_by_global_norm(grads, max_norm)
        adamw_step(params, grads, state, lr)
        total += loss * len(idx)
    return total / len(X)
```

a) **[0,5]** La norme **globale** sur tous les tableaux (0,2), le coefficient $\min(1, \cdot)$ (0,15), de nouveaux tableaux sans toucher à `grads` (0,15).
b) **[0,7]** Le compteur $t$ incrémenté avant la correction de biais (0,1) ; les moments rangés dans `state` et réutilisés d'un appel à l'autre (0,2) ; le weight decay **découplé**, avant le pas (0,15) ; les deux moyennes mobiles et la correction de biais (0,15) ; des mises à jour **sur place** (`p -=`, `p *=` : une affectation `p = p - ...` dans la boucle ne change pas les tableaux de `params`) (0,1).
c) **[0,5]** L'ordre `rng.permutation` et des mini-batches consécutifs (0,15) ; clipping puis pas à chaque mini-batch, avec un seul `state` pour l'epoch (0,2) ; la moyenne pesée par le nombre d'exemples (le dernier mini-batch n'en a que 48) (0,15).
d) **[0,3]** **[1,0946 ; 0,688]** (21 mises à jour). Avec un code faux, recopier fidèlement les valeurs de son code rapporte 0,1.

**Erreurs fréquentes** : une norme par tableau au lieu de la norme globale ; `g *= coef`, qui modifie les gradients reçus ; `p = p - ...` (rien ne change dans `params`) ; le weight decay ajouté au gradient (Adam couplé) ; recréer les moments à chaque appel ; la moyenne simple des losses de mini-batch.
**Remédiation** : 19.18, 19.19, 20.23, 19.13.

## CP4.10 — Expliquer la rétropropagation comme en entretien 💼 · 1 point

**Une réponse modèle (en 60 secondes).** « Un réseau est une composition de fonctions simples, et on cherche, pour chaque poids, de combien la loss change quand on le bouge un peu : la dérivée partielle. Plutôt que de bouger chaque poids un par un, ce qui demanderait une passe avant par poids, soit des millions, la rétropropagation applique la règle de la chaîne de la sortie vers l'entrée : elle calcule l'erreur à la sortie, puis la renvoie couche par couche en la multipliant par la dérivée locale de chaque opération. Une seule passe arrière, qui coûte à peu près deux passes avant, donne tous les gradients d'un coup ; c'est de la différentiation automatique en mode inverse. Avec PyTorch, la passe avant enregistre un graphe des opérations ; `loss.backward()` le parcourt à l'envers, applique la dérivée de chaque opération et **ajoute** les résultats dans `.grad` (d'où le `zero_grad()`). L'optimiseur se sert ensuite de ces gradients. »

**Critères** : la règle de la chaîne appliquée de la sortie vers l'entrée (0,3) ; le coût : une passe arrière pour tous les gradients, contre une passe par poids en perturbant (0,3) ; le graphe enregistré pendant la passe avant et parcouru à l'envers par `backward()` (0,2) ; un détail juste qui montre la pratique : les gradients s'accumulent dans `.grad`, la vérification par différences finies, ou ce que fait l'optimiseur ensuite (0,2).
**Relances possibles du recruteur** : « Pourquoi a-t-on besoin de garder les activations de la passe avant ? » (les dérivées locales en dépendent : c'est la mémoire d'entraînement) ; « Comment vérifieriez-vous un gradient écrit à la main ? » (différences centrées sur quelques poids, erreur relative) ; « Que se passe-t-il avec 50 couches de sigmoïdes ? » (CP4.5).
**Remédiation** : 18.10, 18.E1, 18.E2, 18.9.

## CP4.11 — Refus de crédit par un réseau ⚖️ · 0,5 point

**Critères** (0,5 en tout) :
- une explication **complète** donne tous les calculs, neurone par neurone ; elle peut être exacte et inutile. Une explication **satisfaisante** nomme les facteurs décisifs, dit pourquoi chacun est légitime, et montre ce qui changerait la décision (0,2) ;
- une explication contrefactuelle, par exemple : « avec un endettement mensuel inférieur de 150 €, ou deux ans d'ancienneté dans votre emploi au lieu d'un, le crédit aurait été accordé » ; elle porte sur des facteurs que le client peut comprendre et, si possible, changer, jamais sur un substitut d'une caractéristique protégée comme le code postal (0,15) ;
- le droit européen : le **RGPD (art. 22)** donne le droit de ne pas faire l'objet d'une décision **fondée exclusivement** sur un traitement automatisé qui produit des effets significatifs, comme un refus de crédit (l'arrêt SCHUFA de la CJUE, 2023, l'applique à un score de crédit) : le client peut demander une intervention humaine, exprimer son point de vue et contester ; l'**AI Act** classe l'évaluation de la solvabilité parmi les systèmes à haut risque (annexe III), avec un droit à des explications (art. 86) (0,15 pour l'un des deux textes, correctement cité).

**Erreurs fréquentes** : croire qu'envoyer les poids du réseau suffit ; une explication contrefactuelle sur l'âge ou le quartier ; invoquer un « droit à l'explication » général sans dire d'où il vient.
**Remédiation** : 20.11, 20.Q12, fiche du ch. 20, §20.8.1.

## CP4.12 — Parties I à III ✏️ · 2 points

a) **[0,3]** La cross-entropy ne regarde que la probabilité de la **vraie** classe : $-\ln 0{,}7 \approx$ **0,357** nat.
b) **[0,3]** $-\log_2 0{,}7 = 0{,}357 / \ln 2 \approx$ **0,515** bit.
c) **[0,3]** $\frac{-\ln 0{,}7 - \ln 0{,}5 - \ln 0{,}9}{3} = \frac{0{,}3567 + 0{,}6931 + 0{,}1054}{3} \approx$ **0,3851**.
d) **[0,5]** En annulant la dérivée, $w = \frac{\sum x_i y_i}{\sum x_i^2 + \alpha}$ ; $\sum x_i y_i = 2 + 6 + 21 = 29$ et $\sum x_i^2 = 14$ : $w = \frac{29}{14} \approx 2{,}071$, puis $\frac{29}{21} \approx 1{,}381$ : **[2,071 ; 1,381]**. La pénalité rapproche le poids de 0.
e) **[0,4]** Moyenne 195 ; écarts $-13, -4, 2, 15$ ; variance $\frac{169 + 16 + 4 + 225}{4} = 103{,}5$ ; $\sigma \approx 10{,}1735$ ; $z = \frac{216 - 195}{10{,}1735} \approx$ **2,064**.
f) **[0,2]** Les statistiques s'apprennent sur l'entraînement seul et s'appliquent telles quelles aux données nouvelles : les calculer avec le manchot de test, c'est une fuite (le test influence la préparation, et l'évaluation devient optimiste) ; en production, on ne connaîtrait d'ailleurs pas les exemples à venir.

**Erreurs fréquentes** : l'entropie de toute la distribution prédite (0,802) au lieu de $-\ln p_{\text{vraie}}$ ; log₁₀ ; $-\ln$ de la moyenne des probabilités (0,3567) ; le rapport des moyennes en d) ; l'écart-type avec $n - 1$ (1,788) ou le manchot de test dans les statistiques (1,357).
**Remédiation** : 6.5, 6.16, 6.E1, 9.3, 9.17, 12.6, 12.16, 12.31.

## CP4.13 — Prédire ce que font cinq extraits PyTorch 🔮 · 1 point

Réponses : **a) B · b) C · c) C · d) A · e) D** (0,2 chacune).

a) **(B) `tensor([4., 4., 4.])`** : chaque tour construit un graphe neuf, et `backward()` **ajoute** son gradient (2 par composante) à ce que contient déjà `x.grad`. C'est pourquoi on appelle `zero_grad()` à chaque pas.
b) **(C) des 1** : en mode évaluation, le dropout de PyTorch est l'identité ; il est **inversé**, il a déjà divisé par $1 - p$ en entraînement.
c) **(C) une `ValueError`** (« Expected more than 1 value per channel when training ») : une couche neuve est en mode entraînement, et une variance de mini-batch sur un seul exemple n'a pas de sens. En mode évaluation (`.eval()`), elle utiliserait ses statistiques glissantes, et accepterait une seule image.
d) **(A) `torch.Size([3, 5])`** : PyTorch range le poids de `nn.Linear(n_in, n_out)` en $(n_{\text{out}}, n_{\text{in}})$ et calcule $x W^\top + b$, la transposée de la convention $(n_{\text{in}}, n_{\text{out}})$ du workbook.
e) **(D) $\ln 4 \approx 1{,}3863$** : des logits nuls donnent une softmax uniforme, $\frac14$ pour chaque classe ; la loss vaut $-\ln\frac14$ pour chaque exemple, en moyenne aussi.

**Remédiation** : 20.13, 20.15, 20.14, 20.22, 20.Q5, 18.14.

## CP4.14 — Expliquer pourquoi on n'initialise pas les poids à zéro 🗣️ · 0,5 point

**Une réponse possible.** « Imagine une équipe où tout le monde reçoit les mêmes consignes et les mêmes informations : chacun fera exactement le même travail, et l'équipe ne vaut pas mieux qu'une seule personne. Des poids tous égaux, c'est pareil : les neurones d'une couche calculent la même chose, reçoivent la même correction, et restent jumeaux pour toujours. Le hasard leur donne des points de départ différents, donc des rôles différents. Mais l'échelle compte : des poids trop grands font exploser les valeurs couche après couche (ou saturer les activations), des poids trop petits les font s'éteindre ; on choisit la taille du hasard d'après le nombre d'entrées de chaque neurone (LeCun, Glorot, He) pour que les signaux gardent la même ampleur d'une couche à l'autre. »

**Critères** : la symétrie, des neurones qui restent identiques (0,25) ; l'échelle, des signaux qui explosent ou s'éteignent avec la profondeur, et la règle qui dépend du nombre d'entrées (0,15) ; une image ou un exemple, sans formule, en cinq lignes au plus (0,1).
**Remédiation** : 16.6, 16.7, 16.19, 16.E1, 16.23.

---

## Après la correction

| Ta note | Ce que tu fais ensuite |
|---|---|
| 16 à 20 | Le mini-projet MP4. Garde les questions ratées pour une révision dans une semaine. |
| 12 à 16 | Les remédiations des questions où tu as eu moins de la moitié des points, puis le mini-projet. |
| moins de 12 | Refais les 🔨 des ch. 18 et 19 (`dense_backward`, `mlp_backward`, Adam et AdamW) et le 📦 20.23 (la boucle PyTorch), relis la synthèse, puis repasse les questions ratées sans le corrigé avant le mini-projet. |

Reporte ta note, le temps réellement passé sur chaque question et tes points faibles dans la section CP4 de `mon_travail/suivi/auto_evaluation.md`.
