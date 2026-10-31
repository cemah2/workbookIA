# Audit P5 — A5 : exactitude des ch. 23 et 24 (commit b961aa3)

**Périmètre :** `chapitres/ch23_pytorch_1`, `chapitres/ch24_pytorch_2` (fiches, exercices, indices, solutions, notebooks, flashcards, figures), `annexes/cheatsheets/pytorch.md`, lignes 23-24 de `annexes/formulaire.md`, `glossaire.md`, `erreurs_frequentes.md`, data cards `fashion_mnist.md`, `cifar10.md`, `pretrained.md`, renvois des autres chapitres vers 23-24. Scripts de vérification : `scripts/` du dossier de travail (liste en fin de rapport).

## Constats

### Vérifications faites sans constat (pour mémoire)

- **Partie 0, re-résolution à l'aveugle** : 130 réponses du ch. 23 (Q1 à Q12, R1 à R3, ✏️ 23.1 à 23.7, 🧮 23.8) et 125 du ch. 24 (Q1 à Q12, R1 à R3, ✏️ 24.1 à 24.5, 🧮 24.7), calculées sans lire les solutions, puis passées à `wb.check` : **255 sur 255 acceptées** (`scripts/check23.py`, `scripts/check24.py`).
- **Nombres des solutions recalculés** (plus de 90) : variantes de 23.1 à 23.8, 23.12 à 23.29 (paramètres de `layer4` de ResNet-18 = 8 393 728, tête 5 130, gelés 11 176 512, 39 505, 101 770, 0,240, 422/112, 0,0599, 73,3, `top_k`, `C + C.T`) ; ch. 24 : 24.1 à 24.7 et leurs variantes (299 tirages, 16 198, 2 304 000/576 000, GRU 228, 1 995/63, 440 min, ∂ 24.6 q. 4 et 6), 24.9 (moyennes et écarts-types à n − 1 des deux réglages, et sur 3 graines), 24.10 (6n + 2, 0,27 M, 6,43 %, 19,4 M), 24.11 à 24.31 (tableaux de paramètres de 24.12, 24.19 = 25 578, 24.20 a-c et 9,9 M MAC, 24.21, 24.22 et sa bande de 240 pixels, 24.25-24.26 : epochs 572/667/64, moyennes des effets profondeur/fenêtre/données, 24.27, 24.28 a-c et variante 19 954, 24.29, 24.30). Tous conformes aux sorties exécutées des deux `05_solutions.ipynb` et à `history_ch23.json` et `history_ch24.json` (24.14, 24.21, 24.23 recalculés depuis le JSON), et aux statistiques de Fashion-MNIST recalculées (moyennes par classe 34,9 à 98,3, pixels allumés 32 à 65 %, 0,2860/0,3530 sur 60 000 images), sauf le constat 6.
- **Affirmations PyTorch des fiches exécutées sous torch 2.11.0+cpu / torchvision 0.26.0+cpu** (`scripts/v23_fiche.py`, `v24_fiche.py`, `v_cheat.py`) : messages d'erreur cités (`mat1 and mat2…`, `shape '[7, 2]' is invalid…`, `padding='same' is not supported for strided convolutions`, `Trying to backward through the graph a second time`, `Expected more than 1 value per channel…`), avertissements (`lr_scheduler.step()` avant `optimizer.step()`, `MSELoss` à formes différentes), `torch.load(weights_only=None → True)`, `torch.cuda.amp.GradScaler` en `FutureWarning`, `torch.amp.GradScaler("cpu")`, `autocast` CPU en `bfloat16` par défaut et loss en `float32`, plus petit `float16` ≈ 5,96 × 10⁻⁸, `renorm`, formes de `nn.LSTM` (dont bidirectionnel et `output[:, -1] == h_n[-1]`), `LambdaLR`, `v2.Compose(ToImage, ToDtype(scale=True), Normalize)`, `ResNet18_Weights.DEFAULT.transforms()` (256/224, moyennes d'ImageNet) et `meta`, dérivées en 0 de `relu`/`clamp`/`maximum` (0, 1, 0,5), `Adam(decoupled_weight_decay=…)`, `nn.Embedding` en 𝒩(0, 1), `ConvTranspose2d` et `unfold`. Tout est exact, sauf le constat 1. Les 25 lignes des ch. 23 et 24 de `annexes/erreurs_frequentes.md` ont leurs messages reproduits mot pour mot sous torch 2.11 et scikit-learn 1.6.1 (`scripts/v_errors.py`), y compris l'ordre des classes mères (`BaseEstimator` d'abord : `is_classifier` faux sans avertissement, `check_cv` rend un `KFold`).
- **Le livre** : encadrés « Le livre, corrigé » et chiffres cités contrôlés sur `book_ch23.txt` et `book_ch24.txt` (p. 1097, 1106, 1111, 1139-1140, 1183-1184, 1197-1204, 1224-1229, 1250-1252, 1282, 1303-1304, 1342 et les durées des CNN) : tous justes, pages comprises, sauf l'oubli du constat 4. Aucun passage recopié ou paraphrasé de près dans les sondages (p. 1098, 1129-1135, 1250-1252).
- **Flashcards** : les 22 du ch. 23 et les 25 du ch. 24 relues une à une ; une seule erreur (constat 1).
- **Figures** : les 13 images relues (formes, nombres, légendes) ; cohérentes avec la fiche (meilleure epoch 18 + patience 8 = arrêt 26 ; 64 et 16 positions ; sorties de `nn.LSTM`), une suggestion (constat 11).
- **Annexes** : lignes 23-24 du formulaire et 31 termes du glossaire relus ; cheatsheet relue en entier (≈ 150 lignes, dont une trentaine exécutées) ; data cards Fashion-MNIST et CIFAR-10 cohérentes avec les notebooks.

### 1. MINEUR · `chapitres/ch23_pytorch_1/flashcards.csv:7`, `annexes/cheatsheets/pytorch.md:104`, `annexes/glossaire.md:774` (« `view` refuse un tenseur non contigu »)
**Extrait :** flashcard « view rend toujours une vue et refuse un tenseur non contigu » ; cheatsheet « idem sans copie, refusé si non contigu » ; glossaire « `permute` et `.T` rendent des vues non contiguës, que `view` refuse ».
**Problème :** inexact. `view` refuse seulement une forme **incompatible avec les pas** (*strides*) du tenseur, pas tout tenseur non contigu. Sous torch 2.11, `x = torch.zeros(2, 5, 6, 3).permute(0, 3, 1, 2)` donne `x.is_contiguous() == False`, et pourtant `x.view(2, 3, 30)` réussit (H et W restent fusionnables) ; seul `x.view(2, -1)` lève l'erreur. De même, `torch.arange(12).reshape(3, 4).t().view(2, 2, 3)` réussit. La fiche dit juste (§23.5.1 : « lève une erreur si la forme demandée ne correspond pas au rangement en mémoire », « `view` refuse de l'**aplatir** ») ; la carte Anki, apprise par cœur, installe la règle fausse.
**Correction :** flashcard : « (view rend toujours une vue, et refuse une forme que le rangement en mémoire ne permet pas, typiquement aplatir un tenseur permuté ; reshape copie si besoin) » ; cheatsheet : « idem sans copie, refusé si la mémoire ne s'y prête pas (aplatir après `permute`) » ; glossaire : « …rendent des vues non contiguës, que `view` ne peut pas aplatir et que `.contiguous()` recopie ».

### 2. MINEUR · `annexes/cheatsheets/pytorch.md:3` (en-tête)
**Extrait :** « Aide-mémoire rempli au fil des chapitres (20-27). »
**Problème :** la cheatsheet a des lignes des ch. 5, 10, 16, 17, 18 et 19 (section « Autograd », « Définir un modèle », « Optimiseurs ») et une section entière du ch. 28 (« Optimiser une image, hooks et transfert de style »).
**Correction :** « Aide-mémoire rempli au fil des chapitres (5 et 10 à 28). »

### 3. MINEUR · `chapitres/ch24_pytorch_2/04_indices.md:321` (Ex 24.2, indice 3, g)
**Extrait :** « pour g), additionne les six couches qui ont des paramètres »
**Problème :** le CNN à pooling n'a que **cinq** couches à paramètres (deux `Conv2d`, trois `Linear`) ; la solution additionne bien cinq termes (780 + 4 336 + 100 480 + 8 256 + 650). L'apprenant cherche une sixième couche (il pourrait compter le dropout ou le pooling, que l'indice 2 dit justement sans paramètres).
**Correction :** « pour g), additionne les cinq couches qui ont des paramètres (deux convolutions, trois couches denses) ».

### 4. MINEUR · `chapitres/ch23_pytorch_1/01_fiche.md:136-151` (§23.3.3-23.3.4 ; contre `docs/BIBLE.md:685`)
**Extrait :** la ligne du §22 (2026-10-29, « Erreurs du livre signalées par la fiche du ch. 23 ») annonce « channels_last et « channels first » confondus (p. 1106) » ; la fiche ne cite p. 1104-1106 que pour définir les deux ordres (l. 151).
**Problème :** la confusion du livre n'est signalée nulle part dans le ch. 23 : p. 1106, le livre règle `image_data_format` à `channels_last` en disant que les données sont « structured with the channels first » (« … » (citation du livre retirée de la version versionnée, §3)). L'apprenant qui lit le livre ⏩ §23.3.3 n'a pas la correction, alors que le ch. 24 corrige la même confusion (p. 1264-1266, 1276). Le §22 décrit un état que la fiche n'a plus (ou n'a jamais eu).
**Correction :** ajouter à la fin du §23.3.4 : « > ⚠️ **Le livre, corrigé — channels_last n'est pas « channels first »** — En présentant `keras.json` (p. 1106), le livre règle `image_data_format` sur `channels_last` mais écrit que les canaux viennent en premier. `channels_last` range bien les canaux **en dernier**, `(H, W, C)` ; c'est `channels_first` qui donne `(C, H, W)`, l'ordre de PyTorch. » (ou, à défaut, retirer cette erreur de la ligne du §22).

### 5. MINEUR · `chapitres/ch24_pytorch_2/01_fiche.md:233, 247, 266` (§24.3.2 à §24.3.4, `X_train`)
**Extrait :** `scores = cross_val_score(TorchMLPClassifier(epochs=3), X_train, y_train, cv=cv)` ; `make_pipeline(MinMaxScaler(), TorchMLPClassifier())` ; `search.fit(X_train, y_train)`.
**Problème :** la fiche ne dit jamais ce qu'est `X_train` dans ces extraits ; le seul `X_train` défini (ch. 23, §23.5.2, et `wb.datasets.load_fashion_mnist`) a la forme `(60000, 28, 28)` en `uint8`. Recopié tel quel, `MinMaxScaler().fit` lève `ValueError: Found array with dim 3. MinMaxScaler expected <= 2.` (vérifié sous scikit-learn 1.6.1), et l'estimateur de la fiche, qui lit `X.shape[1]`, construirait une couche `Linear(28, …)`. Le même nom `X_train` sert ensuite aux fenêtres `(N, w, 1)` (l. 490).
**Correction :** en tête du §24.3.2, une ligne : `X_train = X_train.reshape(len(X_train), -1).astype(np.float32)   # (60000, 784): scikit-learn wants one row per example` (sans division par 255 au §24.3.3, puisque c'est le scaler qui met à l'échelle), et dire en une phrase que les estimateurs scikit-learn reçoivent des tableaux 2D.

### 6. MINEUR · `chapitres/ch24_pytorch_2/05_solutions.md:549` (24.20, variante)
**Extrait :** « `make_deep_cnn_20(widths=(8, 16, 32))` a **23 314** paramètres, environ quatre fois moins : diviser les largeurs par 2 divise par 4 les poids des convolutions. »
**Problème :** 82 330 / 23 314 = 3,53 : le modèle a trois fois et demie moins de paramètres, pas quatre ; seules les convolutions sont divisées par environ 4 (72 080 / 18 184 = 3,96), la couche dense seulement par 2 (10 250 contre 5 130).
**Correction :** « …a **23 314** paramètres, trois fois et demie moins : diviser les largeurs par 2 divise par 4 les poids des convolutions (72 080 → 18 184), mais seulement par 2 ceux de la couche dense. » (à corriger dans `tools/chapters/build_ch24.py` si la variante y est aussi).

### 7. MINEUR · `chapitres/ch22_rnn/01_fiche.md:220` (renvoi vers le livre ch. 23)
**Extrait :** « (`return_sequences=True` dans Keras, que le livre retrouvera aux ch. 23 et 24) »
**Problème :** le ch. 23 du livre ne parle pas de RNN : `return_sequences` y apparaît 0 fois (25 fois au ch. 24, §24.5.8).
**Correction :** « …que le livre retrouvera au ch. 24 (§24.5.8) ».

### 8. MINEUR · `chapitres/ch29_datasets/01_fiche.md:9` (prérequis)
**Extrait :** « ch. 23 (classifieur PyTorch sur MNIST) »
**Problème :** le ch. 23 du workbook entraîne son classifieur sur **Fashion-MNIST** (fiche §23.4.1 et tout le notebook) ; MNIST est le dataset du livre.
**Correction :** « ch. 23 (classifieur PyTorch sur Fashion-MNIST) ».

### 9. MINEUR · `chapitres/ch24_pytorch_2/01_fiche.md:400, 416, 448` (`num_workers` sous Windows et macOS)
**Extrait :** « `DataLoader(num_workers=...)` prépare les batches suivants en parallèle pendant que le modèle calcule » ; « avec `num_workers > 1`, chaque worker exécute `__iter__` de son côté ».
**Problème :** la fiche présente `num_workers` sans sa limite principale pour l'apprenant, qui travaille aussi en local sur un PC : sous Windows et macOS, les workers sont lancés par *spawn*, et un `Dataset` défini dans une cellule de notebook ne leur parvient pas (erreur `Can't get attribute 'CifarImages_23' on <module '__main__'>`). Le notebook le sait (24.24 ne lance le test à deux workers que sous Linux, avec un message qui l'explique), la fiche et la cheatsheet non.
**Correction :** ajouter après la phrase de la l. 400 : « Sur Colab (Linux), `num_workers=2` marche directement ; sous Windows et macOS, les workers sont lancés par *spawn* et ne voient pas une classe définie dans une cellule de notebook : il faut la mettre dans un fichier `.py` importé, ou garder `num_workers=0`. »

### 10. SUGGESTION · `data/cards/pretrained.md` (portée de la data card)
**Extrait :** « # Réseaux pré-entraînés du ch. 28 (poids torchvision) » : SqueezeNet 1.1 et VGG16 seulement.
**Problème :** les poids `ResNet18_Weights.IMAGENET1K_V1` (ch. 21, 21.27 ; ch. 23, §23.9.4 et 📦 23.24 ; MP5) et `MobileNet_V3_Small_Weights` (ch. 20, 🔮 20.26) n'ont de fiche nulle part (`grep -ril resnet data/cards` ne trouve que des mentions dans `cifar10.md` et `fashion_mnist.md`), alors que ⚖️ 23.10 fait justement réfléchir à la licence et à la provenance de ces poids. BIBLE §9 : chaque dataset (ici, des poids appris sur ImageNet) a sa fiche.
**Correction :** renommer la carte « Réseaux pré-entraînés (poids torchvision) » et ajouter une ligne par jeu de poids : `ResNet18_Weights.IMAGENET1K_V1` (11,7 millions de paramètres, 44,7 Mo, prétraitement 256 → 224 et moyennes d'ImageNet, `weights.meta`) pour les ch. 21, 23 et MP5, `MobileNet_V3_Small_Weights.IMAGENET1K_V1` pour le ch. 20, avec la même ligne de licence ; et une section « Utilisation » par chapitre.

### 11. SUGGESTION · `chapitres/ch24_pytorch_2/figures/sinus.png` (`tools/chapters/figures_ch24.py`) et `01_fiche.md:464-474`
**Extrait :** le texte décrit la série du livre (deux ondes, fréquences et amplitudes $(1, 2)$, $\Delta\theta = 0{,}057$, test à partir du point 400) ; la figure placée juste après trace $s(n) = \sin(n\Delta\theta) + 0{,}5\sin(3n\Delta\theta)$, $\Delta\theta = 0{,}06$, test aux points 300 à 499.
**Problème :** rien de faux (la légende dit « une somme de deux sinus »), mais l'apprenant qui compare au livre (fig. 24.47 et suivantes) ne retrouve ni la courbe ni l'emplacement du test.
**Correction :** tracer la série « calme » du livre (`freqs=(1, 2)`, `amps=(1, 2)`, `phases=(0, 0)`, `d_theta=0.057`, test à partir de 400), ou ajouter à la légende « (une série d'exemple, pas celle du livre) ».

### 12. SUGGESTION · `chapitres/ch24_pytorch_2/02_exercices.md:148` (🔁 24.R1 a)
**Extrait :** « a) Ce que garde l'historique d'une fonction `fit()` honnête : (A) la loss d'entraînement… ; (B) la loss et l'accuracy de validation… ; (C) une copie des poids… ; (D) le learning rate de chaque epoch ; (E) la loss de test… » ; réponse « ABD ».
**Problème :** l'historique que le ch. 23 a fait écrire (fiche §23.7 ; `fit_18`, quatre listes) ne contient pas le learning rate : « AB » est une lecture défendable de « honnête », que seule la piste de `wb.check` (« le learning rate de chaque epoch aide à lire les courbes ») redresse. Changer l'énoncé touchant au contrat, la correction est facultative.
**Correction :** au choix, préciser D : « (D) le learning rate de chaque epoch (utile dès qu'un planning le fait varier) », sans changer la réponse.

## Affirmations 🕰️ revérifiées sur le web (10)

| Affirmation (fiche) | Source consultée (2026-10-31) | Verdict |
|---|---|---|
| PyTorch/XLA remplacé par TorchTPU, dépôt archivé en novembre 2026 (§23.3.5) | PR pytorch/xla #9775, fusionnée le 2026-10-19 : « this repo will be archived on November 15, 2026 » | exact |
| Adaptateurs scikit-learn de Keras dépréciés en 2021, retirés dans Keras 2.13 (§24.3.1) | notes de version de TensorFlow 2.13.0 : « Removed the Keras scikit-learn API wrappers » (dépréciés en août 2021), SciKeras recommandé | exact |
| SciKeras, dernière version en 2024 (§24.3.1) | piwheels : 0.13.0 du 2024-04-11, aucune depuis | exact |
| `keras.wrappers.SKLearnClassifier` avec `model=` au lieu de `build_fn=` (§24.3.1) | keras.io, *Scikit-Learn API wrappers* (sources 3.15.1) : `SKLearnClassifier(model, warm_start=False, model_kwargs=None, fit_kwargs=None)` | exact (la version 3.8 n'a pas pu être confirmée par une source accessible) |
| `to_yaml` retiré pour raison de sécurité (§23.9.2) | notes de TensorFlow 2.6.0 : `Model.to_yaml()` et `model_from_yaml` lèvent `RuntimeError` | exact |
| `predict_classes` et `predict_proba` retirés dans TensorFlow 2.6 (§23.8.2) | documentation du paquet R `keras` 2.13.0 : « These functions were removed in Tensorflow version 2.6 » | exact |
| `torch.cuda.amp.GradScaler` déprécié depuis 2.4 (au-delà du livre, ch. 24) | notes de version de torch 2.4.0 : « Deprecate device-specific GradScaler autocast API », `torch.amp.GradScaler("cuda", …)` ; confirmé par un `FutureWarning` sous 2.11 | exact |
| Faille de `weights_only=True` jusqu'à 2.5.1, corrigée en 2.6 (05_solutions Q11 c) | OSV, CVE-2025-32434 : « In version 2.5.1 and prior… patched in version 2.6.0 » | exact |
| CNTK arrêté à sa version 2.7 (2019) (§23.3.3) | Microsoft Learn, notes de CNTK 2.7 (2019-04) : « Today's 2.7 release will be the last main release of CNTK » | exact |
| `pretrained=` déprécié depuis torchvision 0.13 ; licences propres des poids (§23.9.4, ⚖️ 23.10) | documentation de torchvision (*Models and pre-trained weights*) : API multi-poids « as of v0.13 », « may have their own licenses or terms and conditions » | exact |
| `Adam(decoupled_weight_decay=True)` depuis 2.7 (cheatsheet l. 211) | code source de `torch.optim.adamw` dans la documentation 2.7 : `class AdamW(Adam)` avec `decoupled_weight_decay=True` | exact |

## Bilan

**Ce qui est solide.**
- L'exactitude des deux chapitres est très bonne. Les 255 réponses courtes, re-résolues à l'aveugle, passent toutes `wb.check`. Chaque nombre recalculé des solutions et des variantes concorde avec les sorties exécutées et les historiques précalculés, à une approximation près (constat 6).
- Les affirmations sur PyTorch 2.11 et torchvision 0.26 sont justes quand on les exécute : messages d'erreur, défauts de `torch.load`, `autocast` sur CPU, `GradScaler`, formes de `nn.LSTM`, `renorm`, `transforms.v2`, etc.
- Les 10 affirmations 🕰️ revérifiées tiennent, dont la date d'archivage de PyTorch/XLA (15 novembre 2026).
- Les « Le livre, corrigé » sont justes et bien sourcés.
- Les annexes (formulaire, glossaire, erreurs fréquentes) sont exactes.

**Les 3 risques principaux.**
1. **Une règle fausse sur `view` apprise par cœur** (constat 1). La flashcard, la cheatsheet et le glossaire disent que `view` refuse tout tenseur non contigu, alors que la fiche dit juste.
2. **Des extraits de la fiche du ch. 24 qui ne tournent pas tels quels** (constat 5). `X_train` y est implicitement aplati, alors que le seul `X_train` défini est `(N, 28, 28)`. Et la limite de `num_workers` hors de Colab n'est pas dite (constat 9), alors que l'apprenant travaille aussi en local.
3. **Un écart entre le journal (§22) et la fiche du ch. 23** (constat 4). Une erreur du livre que le §22 dit signalée (p. 1106) ne l'est pas. Ce type de dérive passe inaperçu à la relecture chapitre par chapitre.

| Sévérité | Nombre |
|---|---|
| MAJEUR | 0 |
| MINEUR | 9 |
| SUGGESTION | 3 |

**Scripts de vérification** (dans `scripts/` du dossier de travail) :

| Script | Rôle |
|---|---|
| `check23.py`, `check24.py` | les 255 réponses à l'aveugle passées à `wb.check` |
| `v23_fiche.py`, `v24_fiche.py` | les affirmations des fiches exécutées |
| `v_cheat.py` | la cheatsheet |
| `v_errors.py` | les messages de `erreurs_frequentes.md` |
| `nbdump.py` | extraction des cellules et des sorties des notebooks |
