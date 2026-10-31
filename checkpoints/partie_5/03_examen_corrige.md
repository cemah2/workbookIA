# Checkpoint V · Corrigé détaillé et barème de l'examen blanc

> À lire **après** l'examen, et après la partie B du notebook (vérification automatique du code et des réponses chiffrées). Pour chaque question : la réponse, la démarche, le barème, les erreurs fréquentes et les exercices à refaire si tu as eu moins de la moitié des points. Les solutions exécutées des deux questions de code sont dans `03_examen_solutions.ipynb`.

## Règles de notation

- **Barème par sous-question**, indiqué entre crochets. Une réponse juste sans démarche, quand la question en demande une, rapporte la moitié des points.
- **Méthode juste, erreur de calcul** : la moitié des points de la sous-question. **Erreur reportée** : si une réponse fausse est réutilisée correctement plus loin (en CP5.2, CP5.3 et CP5.12 surtout), la suite n'est pas pénalisée une seconde fois.
- **Arrondis** : une décimale de trop ou de moins coûte au plus 0,05 point par question ; un arrondi fait trop tôt, qui change la dernière décimale demandée, aussi.
- **Estimations** (CP5.8) : toute valeur qui s'arrondit à la réponse attendue rapporte tous les points ; un ordre de grandeur juste mais une valeur fausse, la moitié.
- **Code** (CP5.5, CP5.6) : les points vont aux propriétés du contrat (listées ci-dessous), que la partie B vérifie une à une ; un code qui lève une erreur garde les points des parties justes que tu peux montrer en le relisant.
- **Questions rédigées** (CP5.1 g, CP5.2 g, CP5.3 e, CP5.4 b, CP5.9, CP5.12 b, CP5.13, CP5.14) : les critères sont listés ; une idée juste formulée autrement compte autant.

---

## CP5.1 — Forme de sortie ou erreur ? 🔮 · 2 points

a) à f) : 0,25 point chacune.

a) **(8, 16, 28, 28)** : sans padding, un noyau 5 × 5 tient dans $32 - 5 + 1 = 28$ positions par dimension ; 16 filtres, 16 canaux de sortie ; le batch de 8 ne bouge pas.
b) **Erreur** (`RuntimeError` : « expected input[8, 32, 32, 3] to have 3 channels, but got 32 channels instead ») : PyTorch lit l'axe 1 comme celui des canaux, et l'image est rangée « canaux en dernier » (NHWC).
c) **(1, 4, 3, 3)** : `MaxPool2d(2)` a un stride de 2 et arrondit vers le bas, $\lfloor (7 - 2)/2 \rfloor + 1 = 3$ ; la dernière ligne et la dernière colonne ne sont lues par aucune fenêtre.
d) **(2, 7, 20)** : sans `batch_first`, l'entrée `(4, 7, 10)` se lit (temps = 4, batch = 7, features = 10). `[1]` est le couple `(h_n, c_n)`, et `h_n` a la forme `(num_layers, batch, hidden)`.
e) **(4, 7, 40)** : avec `batch_first=True`, la sortie est `(batch, temps, 2 × hidden)` : chaque pas concatène l'état du sens direct et celui du sens inverse.
f) **(2, 3, 5)** : `nn.Linear` agit sur le dernier axe et garde tous les autres.
g) **[0,5]** Seul b) lève une erreur (0,1) : l'image est en `(N, H, W, C)` et la couche attend `(N, C, H, W)` (0,15) ; la correction : `x.permute(0, 3, 1, 2)` (ou `torch.zeros(8, 3, 32, 32)`), qui donne **(8, 16, 32, 32)**, car `padding=1` garde la taille avec un noyau 3 × 3 (0,1). En d) : 2 couches empilées, 7 séquences dans le batch, 20 neurones d'état (0,15).

**Erreurs fréquentes** : n − k au lieu de n − k + 1 (27 en a) ; croire que b) passe, en lisant la forme comme une image NHWC à la manière de Keras ; arrondir 3,5 vers le haut en c) ; prendre l'axe 0 pour le batch en d) sans `batch_first`, ou confondre `out` et `h_n` ; oublier le facteur 2 du bidirectionnel en e) ; croire que `nn.Linear` exige une entrée à deux axes en f).
**Remédiation** : 21.3, 21.20, 21.24, 22.22, 24.4, 23.Q6.

## CP5.2 — Formes, paramètres et champ récepteur d'un petit CNN ✏️ · 2,5 points

a) **[0,2]** `padding=1` avec un noyau 3 × 3 garde 32 × 32, puis le pooling divise par 2 : **(1, 16, 16, 16)**.
b) **[0,2]** De même : **(1, 32, 8, 8)**.
c) **[0,2]** $\lfloor (8 + 2 \times 1 - 3) / 2 \rfloor + 1 = 4$ : **(1, 64, 4, 4)**.
d) **[0,6]** Une convolution a $C_{\text{out}} \times C_{\text{in}} \times k \times k$ poids et $C_{\text{out}}$ biais : $16 \times 3 \times 9 + 16 = 448$ ; $32 \times 16 \times 9 + 32 = 4\,640$ ; $64 \times 32 \times 9 + 64 = 18\,496$ ; la couche dense, $64 \times 10 + 10 = 650$ : **[448 ; 4 640 ; 18 496 ; 650 ; 24 234]** (0,15 par couche ; le total compris dans la dernière).
e) **[0,4]** Une MAC par poids et par position de sortie : $64 \times (4 \times 4) \times (32 \times 3 \times 3) = 1\,024 \times 288 =$ **294 912**. Sans le stride de 2, la carte ferait 8 × 8 et coûterait 4 fois plus.
f) **[0,6]** Chaque couche ajoute $(k - 1) \times j$ pixels, où $j$, l'écart entre deux positions voisines mesuré en pixels de l'image, est le produit des strides **précédents** : conv A, $+2 \times 1$ (3) ; pool A, $+1 \times 1$ (4) ; conv B, $+2 \times 2$ (8) ; pool B, $+1 \times 2$ (10) ; conv C, $+2 \times 4$ : **18 pixels** (0,3 pour la règle, 0,3 pour la valeur). Le stride de la convolution C n'agrandit pas son propre champ, seulement celui des couches suivantes.
g) **[0,3]** Après `Flatten()`, la couche dense recevrait $64 \times 4 \times 4 = 1\,024$ valeurs : $1\,024 \times 10 + 10 = 10\,250$ paramètres au lieu de 650 (0,15). Autre conséquence, au choix (0,15) : le réseau n'accepterait plus que des images 32 × 32 (la moyenne globale accepte n'importe quelle taille) ; la tête saurait **où** est chaque motif, alors qu'une classe d'objet n'en dépend presque pas (la moyenne donne une invariance à la position) ; plus de paramètres, plus de risque d'overfitting.

**Erreurs fréquentes** : oublier un pooling ; ignorer le stride de la convolution C (8 au lieu de 4), ou oublier le padding (3 au lieu de 4) ; oublier $C_{\text{in}}$ dans les poids d'un filtre ; oublier les biais (24 112) ; compter les MAC sur une carte 8 × 8 ; additionner les noyaux sans tenir compte des strides (7, ou 9) ; oublier l'apport des poolings (15) ; compter le stride de la convolution C dans son propre champ (26), ou multiplier l'écart avant d'ajouter l'apport de chaque couche (29).
**Remédiation** : 21.3, 21.4, 21.5, 21.8, 24.2, 24.3.

## CP5.3 — Un pas de LSTM à la main ✏️ · 2 points

a) **[0,6]** Les sommes pondérées valent $0{,}5 \times 2 - 1 = 0$, $1 \times 2 + 0 = 2$, $0{,}25 \times 2 = 0{,}5$ et $-0{,}5 \times 2 + 1 = 0$ (0,2). Puis $i = \sigma(0) = 0{,}5$, $f = \sigma(2) \approx 0{,}8808$, $g = \tanh(0{,}5) \approx 0{,}4621$, $o = \sigma(0) = 0{,}5$ : **[0,5 ; 0,8808 ; 0,4621 ; 0,5]** (0,1 par porte).
b) **[0,4]** $c_1 = 0{,}8808 \times 0{,}5 + 0{,}5 \times 0{,}4621 = 0{,}4404 + 0{,}2311 \approx$ **0,6715**.
c) **[0,4]** $h_1 = 0{,}5 \times \tanh(0{,}6715) = 0{,}5 \times 0{,}5859 \approx$ **0,2930**.
d) **[0,3]** Quatre blocs (i, f, g, o), chacun avec $20 \times 10$ poids d'entrée, $20 \times 20$ poids récurrents et deux vecteurs de biais de 20 : $4 \times (200 + 400 + 40) =$ **2 560**.
e) **[0,3]** Le gradient de $c_t$ par rapport à $c_{t-1}$ vaut $f_t$ (le long de ce chemin) : remonter $k$ pas multiplie par $f_t f_{t-1} \cdots$, des nombres que le réseau peut garder près de 1 quand il doit se souvenir (0,2). Dans un RNN simple, chaque pas multiplie par $W_h$ et par la dérivée de tanh, plus petite que 1 : le produit s'écrase (ou explose, selon $W_h$) exponentiellement (0,1).

**Erreurs fréquentes** : appliquer la sigmoïde à $g$ (0,6225) ; oublier $f\,c_0$ dans la cellule (0,2311) ou faire entrer $g$ sans la porte $i$ ; oublier le tanh de $h_1$ (0,3357) ou la porte de sortie (0,5859) ; un seul biais par bloc (2 480), ou le compte d'un GRU (1 920).
**Remédiation** : 22.4, 22.3, 22.7, 22.16, 22.Q8.

## CP5.4 — Formes et `batch_first` : quatre bugs 🐛 · 2 points

a) **[1]** **B, D, F et I** (0,25 par bonne lettre, −0,25 par lettre en trop, sans descendre sous 0). (E) à la place de (B) ou de (D) désigne l'endroit où le bug se voit, pas la ligne à corriger : 0,1 pour cette lettre, sans la pénalité d'une lettre en trop. (H) à la place de (F) vaut tous les points si b) corrige l'appel en transposant l'entrée (`lstm(images.transpose(0, 1))`) **et** prend en (I) la sortie du dernier pas (`out[-1]`) : c'est une correction valable, même si régler `batch_first=True` à la construction de la couche est plus sûr (l'ordre des axes se décide alors une fois pour toutes).
b) **[1]** 0,25 par ligne : la correction (0,1) et ce qui se passe sans elle (0,15).

| Ligne | L'erreur | La correction | Sans correction |
|---|---|---|---|
| (B) | `unsqueeze(-1)` ajoute l'axe des canaux en dernier : `(64, 28, 28, 1)` | `images.unsqueeze(1)`, qui donne `(64, 1, 28, 28)` | **erreur** en (E), au premier appel de `conv` : « expected input[64, 28, 28, 1] to have 1 channels, but got 28 channels » |
| (D) | après `MaxPool2d(2)`, la carte fait 14 × 14 : 8 × 14 × 14 = 1 568 valeurs | `nn.Linear(8 * 14 * 14, 10)` | **erreur** en (E), une fois (B) corrigée : « mat1 and mat2 shapes cannot be multiplied (64x1568 and 6272x10) » |
| (F) | sans `batch_first=True`, le LSTM lit `(temps, batch, features)` | `nn.LSTM(input_size=28, hidden_size=64, batch_first=True)` | **résultat faux sans message** : PyTorch ne vérifie que le dernier axe, qui vaut bien `input_size = 28` ; le LSTM croit lire 28 séquences de 64 pas, chaque « pas » étant une image différente ; `h_n` a la forme `(1, 28, 64)` |
| (I) | `out[:, 0]` est la sortie du **premier** pas, qui n'a vu que la première ligne (presque noire) | `out[:, -1]` (ou `h_n[-1]`), le dernier pas, qui a lu toute l'image | **résultat faux sans message** : la forme `(64, 64)` est la bonne ; le classifieur décide d'après une seule ligne de pixels ; avec (F) non corrigée, `out[:, 0]` prend même la séquence 0 de chaque « pas » |

Les autres lignes sont justes : (A) convertit en flottants dans $[0, 1]$ ; (C) garde 28 × 28 avant le pooling (`padding=1`) ; (E) aplatit en gardant l'axe du batch ; (G) va de 64 à 10 ; (J) donne des labels entiers ; (K) applique la cross-entropy aux logits.

**Erreurs fréquentes** : désigner (E), où les deux erreurs se déclarent, plutôt que leur cause ; désigner (H) plutôt que (F) (l'appel est juste, c'est la couche qui lit mal l'ordre des axes) ; manquer (F) et (I), justement parce qu'ils ne lèvent aucune erreur : vérifier les formes à la main (ou un `assert` sur `h_n.shape`) est le seul moyen de les voir.
**Remédiation** : 21.24, 22.21, 22.22, 24.4, 24.19, 23.Q6.

## CP5.5 — `rnn_forward` à partir d'une cellule fournie 🔨 · 1,5 point

La solution exécutée et comparée à `nn.RNN` (écart de l'ordre de $10^{-16}$) est dans `03_examen_solutions.ipynb` :

```python
def rnn_forward(X, h0, Wx, Wh, b):
    h, states = h0, []
    for t in range(X.shape[1]):                  # the time axis is axis 1 (batch_first)
        h = rnn_cell(X[:, t], h, Wx, Wh, b)      # the same weights at every step
        states.append(h)
    return np.stack(states, axis=1), h           # (N, T, H): h0 is not one of the states
```

a) **[1,2]** La boucle sur l'axe du **temps**, l'axe 1, avec `X[:, t]` (0,3) ; l'état rendu par un pas passé au suivant, en partant de `h0` (0,3) ; `H_all` de forme `(N, T, H)`, empilé sur l'axe 1 et sans `h0` (0,3) ; `h_last`, l'état du dernier pas, et `X` laissé intact (0,15) ; les tailles lues sur les arguments, sans 5 pas ni 4 neurones écrits en dur (0,15).
b) **[0,3]** **−0,4653**. Avec un code faux, recopier fidèlement la valeur de son code rapporte 0,1.

**Erreurs fréquentes** : boucler sur l'axe 0 (le batch) ; passer `h0` à chaque pas au lieu de l'état précédent (la valeur devient −0,8120, sans récurrence) ; empiler sur l'axe 0, qui donne `(T, N, H)` ; garder `h0` dans `H_all` (`(N, T + 1, H)`) ; rendre une liste d'états au lieu d'un tableau.
**Remédiation** : 22.2, 22.15, 22.16, 23.R1.

## CP5.6 — Entraîner avec early stopping et garder le meilleur modèle 🔨 · 2 points

La solution exécutée est dans `03_examen_solutions.ipynb` :

```python
def fit_early_stopping(model, train_loader, val_loader, max_epochs, patience, path, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)          # one optimizer: its moments live all along
    history = {"train_loss": [], "val_loss": [], "best_epoch": None}
    best, waited = float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()                                                # dropout on
        total, count = 0.0, 0
        for X, y in train_loader:
            loss = nn.functional.cross_entropy(model(X), y)          # logits: the softmax is inside
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        history["train_loss"].append(total / count)
        model.eval()                                                 # dropout off
        with torch.no_grad():
            total, count = 0.0, 0
            for X, y in val_loader:
                total += nn.functional.cross_entropy(model(X), y, reduction="sum").item()
                count += len(y)
        val = total / count                                          # per example: the last mini-batch has 124
        history["val_loss"].append(val)
        if val < best:                                               # strictly better: save
            best, waited, history["best_epoch"] = val, 0, epoch
            torch.save(model.state_dict(), path)
        else:
            waited += 1
            if waited == patience:
                break
    model.load_state_dict(torch.load(path, weights_only=True))       # back to the best weights
    return history
```

a) **[1,6]** Un seul optimiseur, créé avant la boucle, avec le `lr` reçu (0,2) ; `model.train()` puis la boucle d'entraînement dans le bon ordre, la loss d'entraînement moyenne par exemple (0,3) ; `model.eval()` et `torch.no_grad()` (ou `inference_mode`) pour la validation (0,3) ; la loss de validation moyenne **par exemple** sur tout `val_loader` (0,2) ; la sauvegarde du `state_dict` au seul progrès strict et le compteur remis à zéro (0,2) ; l'arrêt dès `patience` epochs de suite sans progrès, et `max_epochs` respecté (0,2) ; les poids rechargés à la fin et le dictionnaire du contrat, des `float`, `best_epoch` compté à partir de 1 (0,2).
b) **[0,4]** **[9 ; 0,4819]** : la loss de validation descend jusqu'à 0,4819 à l'epoch 9, puis vaut 0,5105, 0,4886 et 0,4827 ; l'entraînement s'arrête après l'epoch 12. Avec un code faux, recopier fidèlement les valeurs de son code rapporte 0,1.

**Erreurs fréquentes** : oublier `model.eval()` (le dropout reste actif pendant la mesure : [12 ; 0,5124]) ; faire la moyenne des moyennes des mini-batches de validation, qui donne au dernier (124 images) le poids d'un mini-batch de 300 ([9 ; 0,4709]) ; s'arrêter une epoch trop tard (`waited > patience` : [23 ; 0,4351]) ; un progrès au sens large (`<=`), qui sauvegarde aussi les égalités ; recréer l'optimiseur à chaque epoch (Adam perd ses moments) ; oublier de recharger les meilleurs poids (le modèle rendu est celui de la dernière epoch) ; rendre des tenseurs au lieu de `loss.item()` ; compter les epochs à partir de 0.
**Remédiation** : 23.6, 23.18, 23.25, 23.27, 23.23, 24.13.

## CP5.7 — Diagnostiquer quatre paires de courbes 📈 · 1 point

0,25 par lettre.

a) **C**, le dropout, actif pendant l'entraînement seulement : la loss de validation reste **sous** celle d'entraînement pendant presque tout l'entraînement (0,53 contre 0,97 à l'epoch 1, 0,348 contre 0,348 à l'epoch 30), et les deux baissent régulièrement. La loss d'entraînement est mesurée avec des neurones éteints, la validation avec tout le réseau. Au début, une partie de l'écart vient aussi de la mesure **pendant** l'epoch, quand le réseau était moins bon qu'à la fin (📈 24.14 sépare les deux effets).
b) **A**, un learning rate beaucoup trop grand : la loss d'entraînement dépasse 11 à la première epoch, puis les deux courbes restent plates à $\ln 10 \approx 2{,}30$, le niveau d'un réseau qui ne prédit plus que des probabilités presque uniformes (Adam avec $\eta = 0{,}2$ : les premiers pas, énormes, ont envoyé les poids dans une région d'où il ne repart pas).
c) **B**, de l'overfitting : la loss d'entraînement descend jusque vers 0,06, la validation atteint son minimum aux epochs 4 et 5 (0,90), puis remonte en dents de scie jusqu'à 1,36. Un early stopping aurait gardé les poids de l'epoch 4.
d) **E**, un learning rate beaucoup trop petit : les deux losses partent de $\ln 10 \approx 2{,}30$ et n'ont perdu que 0,16 en 30 epochs, ensemble, sans un saut (Adam avec $\eta = 3 \times 10^{-6}$).

(D) et (F) ne servent pas : avec des labels au hasard, la loss d'entraînement baisserait lentement (le réseau mémorise) pendant que la validation resterait vers 2,3 ou remonterait ; `model.eval()` oublié laisserait le dropout ou la batchnorm en mode entraînement pendant la validation, ce qui la rapprocherait de l'entraînement, sans ces formes-là.

**Erreurs fréquentes** : prendre ① pour une fuite ou pour `eval()` oublié (mais `eval()` oublié **remonterait** la validation au niveau de l'entraînement) ; prendre ② pour des labels au hasard (lire la première epoch : 11, ce n'est pas un réseau qui mémorise) ; prendre ③ pour `eval()` oublié à cause des dents de scie (c'est l'écart qui se creuse qui compte) ; prendre ④ pour un entraînement sain qui débute (lire l'échelle : 2,3, c'est le hasard sur dix classes).
**Remédiation** : 24.14, 24.13, 23.19, 21.23, 24.21.

## CP5.8 — Fermi : un générateur de texte LSTM 🧮 · 1 point

0,25 par item.

a) **9,3 millions** : l'embedding, $10\,000 \times 256 = 2\,560\,000$ ; le LSTM, $4 \times (512 \times 256 + 512 \times 512 + 2 \times 512) = 1\,576\,960$ ; la sortie, $512 \times 10\,000 + 10\,000 = 5\,130\,000$ ; total $9\,266\,960$.
b) **55 %** ($5{,}13 / 9{,}27$) : avec un grand vocabulaire, la couche de sortie (et l'embedding, 28 %) pèsent plus que le LSTM lui-même (17 %).
c) **5,6** : $6 \times 9{,}27 \times 10^6 \times 10^6 \approx 5{,}6 \times 10^{13}$ FLOP.
d) **6 secondes** : $5{,}6 \times 10^{13} / 10^{13}$. Une surestimation : l'embedding ne calcule rien, il lit une ligne de sa table par token (le vrai coût est plus proche de 4 secondes) ; et la récurrence empêche de paralléliser sur le temps, si bien qu'une machine atteint rarement ses FLOP de crête sur un LSTM.

**Erreurs fréquentes** : un seul bloc de poids pour le LSTM (8,1 M) ; oublier l'embedding (6,7 M) ou la couche de sortie (4,1 M) ; 2 FLOP par paramètre (la passe avant seule) ; une erreur de puissance de 10 en d).
**Remédiation** : 22.3, 22.8, 23.8, 21.8.

## CP5.9 — Choisir et justifier une architecture pour trois projets 🗣️ · 1 point

**Réponse modèle**
1. **Les rayures** : un CNN **pré-entraîné** (un ResNet-18 de torchvision, par exemple), dont on remplace la tête avant un fine-tuning ; 2 000 photos ne suffisent pas à apprendre de bons filtres depuis zéro, mais les premiers filtres d'un réseau pré-entraîné détectent déjà bords et textures. Pièges : les classes déséquilibrées (peu de défauts : regarder le recall, pas l'accuracy), des augmentations qui effacent la rayure (un recadrage trop serré) ou qui n'arrivent jamais en production, et des photos de la même pièce des deux côtés du split.
2. **La consommation** : d'abord les **baselines** (la même heure la veille, la même heure la semaine d'avant), puis un LSTM, un GRU ou une convolution 1D sur des fenêtres de quelques jours, avec l'heure et le jour de la semaine en features. Piège : la validation doit être **temporelle** (entraîner sur le passé, valider sur la suite), avec un scaler ajusté sur l'entraînement seulement ; un mélange aléatoire des fenêtres laisse fuir le futur.
3. **Les messages** : un **Transformer pré-entraîné** (un petit modèle de type BERT), avec un fine-tuning sur les 8 000 messages ; il connaît déjà la langue, ce qu'un LSTM entraîné sur 8 000 messages ne peut pas apprendre. Une baseline TF-IDF et régression logistique d'abord, pour savoir ce que le gros modèle apporte. Piège : des messages en double (ou du même client) des deux côtés du split.

**Critères** : un choix défendable et une raison juste par projet (0,2 chacun) ; au moins deux pièges justes en tout (0,4).
**Erreurs fréquentes** : un CNN entraîné depuis zéro sur 2 000 photos ; une validation croisée mélangée sur une série temporelle ; un LSTM depuis zéro pour du texte, sans baseline ; oublier les baselines.
**Remédiation** : 21.E5, 22.E1, 22.E4, 23.24, 24.E2.

## CP5.10 — Retour sur les parties III et IV ✏️ · 2 points

a) **[0,3]** `KFold(3)` sans mélange découpe dans l'ordre : le premier fold **teste** (1, 2) et s'entraîne sur (3, 4, 5, 9). Dans le `Pipeline`, le scaler n'apprend que sur ces quatre valeurs : $\mu = 21/4 = 5{,}25$ ; écarts $-2{,}25 ;\ -1{,}25 ;\ -0{,}25 ;\ 3{,}75$ ; variance $20{,}75 / 4 = 5{,}1875$ ; **[5,25 ; 2,2776]**.
b) **[0,3]** $\frac{2 - 5{,}25}{2{,}2776} \approx$ **−1,427**. Avec un scaler ajusté sur les six valeurs (la fuite qu'évite le `Pipeline`), on trouverait −0,775 : le test aurait influencé la préparation.
c) **[0,2]** **13** : 4 valeurs de `C` × 3 folds = 12 ajustements pendant la recherche, chacun sur l'entraînement de son fold, plus 1 pour le réentraînement du meilleur pipeline sur tout le dataset (`refit=True`).
d) **[0,4]** $\partial L / \partial \mathbf{W} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Z}}$, de la forme de $\mathbf{W}$, $(2, 3)$ : **[[0,5 ; −1 ; 0] ; [0 ; −2 ; −2]]** (la ligne 2 vaut $2 \times (0{,}5 ;\ -1 ;\ 0) - 1 \times (1 ;\ 0 ;\ 2)$).
e) **[0,3]** La somme des lignes de $\frac{\partial L}{\partial \mathbf{Z}}$ (le biais sert à chaque exemple) : **[1,5 ; −1 ; 2]**.
f) **[0,5]** $\partial L / \partial \mathbf{X} = \frac{\partial L}{\partial \mathbf{Z}}\,\mathbf{W}^\top$, de la forme de $\mathbf{X}$, $(2, 2)$ : **[[0,5 ; 0] ; [−1 ; 2]]**.

**Erreurs fréquentes** : ajuster le scaler sur les six valeurs ; prendre pour premier fold d'entraînement (1, 2, 3, 4) ; l'écart-type avec $n - 1$ (2,6300, d'où −1,236) ; oublier le réentraînement final (12), ou ajouter les trois ajustements du `cross_val_score` d'avant (16) ; $\mathbf{X}\,\partial L/\partial \mathbf{Z}$ au lieu de $\mathbf{X}^\top \partial L/\partial \mathbf{Z}$ ; la moyenne au lieu de la somme pour le biais ; $\partial L / \partial \mathbf{X}$ transposé.
**Remédiation** : 15.17, 15.18, 15.1, 24.17, 18.7, 18.15.

## CP5.11 — Questions flash 🧠 · 1 point

0,25 par item.

a) **168** : 8 filtres de $4 \times 5$ poids (un filtre 1D couvre tous les canaux d'entrée), plus 8 biais.
b) **1 920** : trois blocs (r, z, n), chacun avec $20 \times 10 + 20 \times 20$ poids et deux biais de 20 : $3 \times 640$.
c) **32** : $(16 - 1) \times 2 - 2 \times 1 + 4 = 32$ ; avec ces réglages (noyau 4, stride 2, padding 1), une convolution transposée double exactement la taille.
d) **320 000** : une ligne de 64 valeurs par token, sans biais.

**Erreurs fréquentes** : oublier les canaux d'entrée (48) ; le compte d'un LSTM (2 560) ou un seul biais par bloc (1 860) ; appliquer la formule d'une convolution ordinaire (8) ; oublier de retirer le padding (34).
**Remédiation** : 21.4, 21.6, 21.20, 22.3, 24.30 ; fiche du ch. 24, §24.5.11 (encadré « un embedding est une lecture de table »).

## CP5.12 — Rétropropager dans une convolution 1D ∂ · 1 point

a) **[0,2]** $y_1 = 1 - 2 = -1$, $y_2 = 2 - (-1) = 3$, $y_3 = -1 - 3 = -4$ : **[−1 ; 3 ; −4]**. Le noyau n'est pas retourné : c'est une corrélation croisée, comme `nn.Conv1d`.
b) **[0,3]** $\frac{\partial L}{\partial y_t} = y_t$, et $\frac{\partial y_t}{\partial w_1} = x_t$ ; par la règle de la chaîne, $\frac{\partial L}{\partial w_1} = \sum_t \frac{\partial L}{\partial y_t}\frac{\partial y_t}{\partial w_1} = \sum_t y_t\,x_t$ (0,15). De même $\frac{\partial L}{\partial w_2} = \sum_t y_t\,x_{t+1}$ et $\frac{\partial L}{\partial b} = \sum_t y_t$ (0,05). La somme vient du **partage des poids** : le même $w_1$ sert aux trois positions, et chaque position lui renvoie sa part du gradient ; les contributions s'additionnent, comme celles des exemples d'un mini-batch (0,1).
c) **[0,3]** $\frac{\partial L}{\partial w_1} = (-1)(1) + 3 \times 2 + (-4)(-1) = 9$ ; $\frac{\partial L}{\partial w_2} = (-1)(2) + 3 \times (-1) + (-4)(3) = -17$ ; $\frac{\partial L}{\partial b} = -1 + 3 - 4 = -2$ : **[9 ; −17 ; −2]**.
d) **[0,2]** Chaque $x_k$ sert aux positions qui le lisent : $\frac{\partial L}{\partial x_k} = \sum_t y_t \frac{\partial y_t}{\partial x_k}$. $x_1$ : $w_1 y_1 = -1$ ; $x_2$ : $w_2 y_1 + w_1 y_2 = 1 + 3 = 4$ ; $x_3$ : $w_2 y_2 + w_1 y_3 = -3 - 4 = -7$ ; $x_4$ : $w_2 y_3 = 4$ : **[−1 ; 4 ; −7 ; 4]**. C'est une convolution « pleine » des $y_t$ par le noyau retourné.

**Erreurs fréquentes** : retourner le noyau en a) ; diviser par 3 (la loss est une somme) ; échanger les fenêtres de $w_1$ et $w_2$ ; oublier en d) que $x_2$ et $x_3$ reçoivent deux contributions.
**Remédiation** : 21.7, 21.R2, 21.31, 18.6, 18.7.

## CP5.13 — `model.eval()` et `torch.inference_mode()` 💼 · 0,5 point

**Réponse modèle en 60 secondes** : « Ce sont deux choses indépendantes. `model.eval()` change le **comportement** de certaines couches : le dropout ne coupe plus rien, la batchnorm normalise avec ses statistiques glissantes au lieu de celles du mini-batch (et ne les met plus à jour). `torch.inference_mode()` (comme `no_grad`, en plus strict) change le **calcul** : PyTorch n'enregistre plus le graphe, donc ni mémoire pour les activations ni temps perdu, et les tenseurs créés ne peuvent pas servir à un `backward()`. Oublier `eval()` donne des prédictions fausses ou bruitées, qui dépendent du mini-batch, et modifie même les statistiques de la batchnorm ; oublier `inference_mode` donne les bonnes prédictions, mais consomme plus de mémoire, au point de saturer un GPU sur un gros jeu d'évaluation. »

**Critères** : `eval()` agit sur dropout et batchnorm, avec ce qui change pour chacune (0,2) ; `inference_mode` coupe l'enregistrement du graphe, sans changer les résultats (0,15) ; ce que coûte l'oubli de chacun (0,15).
**Relances possibles** : « Quelle différence entre `no_grad` et `inference_mode` ? » (les tenseurs créés sous `inference_mode` ne pourront jamais entrer dans un graphe, ce qui permet plus d'optimisations) ; « Et pour un modèle sans dropout ni batchnorm ? » (`eval()` ne change alors rien aux prédictions, mais on l'appelle quand même : le modèle pourrait en gagner plus tard) ; « Que faire avant de reprendre l'entraînement ? » (`model.train()`).
**Erreurs fréquentes** : croire que `eval()` désactive les gradients, ou que `no_grad` désactive le dropout ; oublier que la batchnorm met à jour ses statistiques glissantes en mode entraînement, même sous `no_grad` ou `inference_mode` : seul `eval()` les fige.
**Remédiation** : 23.E2, 23.17, 23.22, 24.E5.

## CP5.14 — Une seule graine pour 0,8 point ⚖️ · 0,5 point

**Critères** (0,5 en tout) :
- un seul entraînement par architecture ne dit pas si l'écart dépasse la **variabilité** due au hasard (initialisation, ordre des mini-batches, dropout) : d'une graine à l'autre, l'accuracy d'un même réseau bouge souvent de plusieurs dixièmes de point, et le jeu de test, de taille finie, ajoute sa propre incertitude (0,2) ;
- à la place : entraîner chaque architecture sur plusieurs graines (5 au moins), rapporter la moyenne et l'écart-type (ou un intervalle), sur le même split, avec le jeu de test ouvert une seule fois, après tous les choix ; ou écrire honnêtement « sur un seul entraînement, B a fait 0,8 point de mieux ; l'écart n'est pas établi » (0,15) ;
- l'honnêteté : le client (un hôpital) prendra des décisions d'après le rapport ; présenter comme établi un écart qui peut venir du hasard, ou choisir la graine qui arrange, l'induit en erreur, même sans intention, et une affirmation de performance sur un dispositif médical engage l'équipe (0,15).

**Erreurs fréquentes** : répondre seulement « il faut plus de données » ; proposer de garder la meilleure graine de chaque modèle (c'est sélectionner le hasard) ; oublier que choisir l'architecture sur le jeu de test le rend lui-même optimiste.
**Remédiation** : 24.9, 24.11, 24.15, 23.29.

---

## Barème récapitulatif et remédiation

| Question | Points | Ch. | Si tu as moins de la moitié des points, refais… |
|---|---|---|---|
| CP5.1 | 2 | 21 à 24 | 21.3, 21.20, 21.24, 22.22, 24.4, 23.Q6 |
| CP5.2 | 2,5 | 21, 24 | 21.3, 21.4, 21.5, 21.8, 24.2, 24.3 |
| CP5.3 | 2 | 22 | 22.4, 22.3, 22.7, 22.16, 22.Q8 |
| CP5.4 | 2 | 21 à 24 | 21.24, 22.21, 22.22, 24.4, 24.19, 23.Q6 |
| CP5.5 | 1,5 | 22 | 22.2, 22.15, 22.16, 23.R1 |
| CP5.6 | 2 | 23, 24 | 23.6, 23.18, 23.25, 23.27, 23.23, 24.13 |
| CP5.7 | 1 | 21, 23, 24 | 24.14, 24.13, 23.19, 21.23, 24.21 |
| CP5.8 | 1 | 21 à 23 | 22.3, 22.8, 23.8, 21.8 |
| CP5.9 | 1 | 21 à 24 | 21.E5, 22.E1, 22.E4, 23.24, 24.E2 |
| CP5.10 | 2 | 15, 18 | 15.17, 15.18, 15.1, 24.17, 18.7, 18.15 |
| CP5.11 | 1 | 21, 22, 24 | 21.4, 21.6, 21.20, 22.3, 24.30, fiche du ch. 24 (§24.5.11) |
| CP5.12 | 1 | 21 | 21.7, 21.R2, 21.31, 18.6, 18.7 |
| CP5.13 | 0,5 | 23, 24 | 23.E2, 23.17, 23.22, 24.E5 |
| CP5.14 | 0,5 | 24 | 24.9, 24.11, 24.15, 23.29 |
| **Total** | **20** | | |

**Lire ta note**
- **16 à 20** : la partie V est solide. Passe au mini-projet MP5 ; garde les flashcards des chapitres 21 à 24 dans tes révisions.
- **12 à moins de 16** : c'est bien. Fais la remédiation des questions où tu as eu moins de la moitié des points, puis le mini-projet.
- **8 à moins de 12** : avant le mini-projet, refais (ou fais, s'il n'est pas dans ton parcours) le 🔨 22.15 (`rnn_forward`), les 📦 23.18 et 23.25 (la boucle `fit()` et ses callbacks), et les ✏️ 21.3 à 21.5 (formes, paramètres, champ récepteur) ; relis la synthèse (`05_synthese.md`), puis repasse les questions ratées sans le corrigé.
- **Moins de 8** : fais tout ce qui précède, reprends en plus les exercices ★★ des chapitres où tu as perdu le plus de points, puis refais l'examen avant le mini-projet.

Dans tous les cas, **refais dans une semaine** les questions où tu as eu moins de la moitié des points, sans regarder ce corrigé : c'est l'effort de rappel qui fixe les notions. Reporte ta note, le temps réellement passé sur chaque question et tes points faibles dans la section CP5 de `mon_travail/suivi/auto_evaluation.md`.
