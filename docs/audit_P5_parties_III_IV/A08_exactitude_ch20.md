# A8 — Exactitude du ch. 20 (et cheatsheet PyTorch, annexes, data cards)

Périmètre relu : `chapitres/ch20_deep_learning/` (fiche, exercices, indices, solutions .md et .ipynb exécuté, notebook d'exercices, flashcards, 4 figures), `annexes/cheatsheets/pytorch.md`, lignes ch. 20 de `annexes/formulaire.md`, `glossaire.md`, `erreurs_frequentes.md`, data cards MNIST, Fashion-MNIST, CIFAR-10 (et synthetic), référence `solutions/mylearn_ref/nn/regularization.py`, texte du livre (vol2.txt, l. 1197-3011).

Méthode : toutes les réponses courtes recalculées depuis les énoncés (`scripts/num.py`), comparées à `05_solutions.md` et aux cellules `answer` du notebook exécuté ; ~70 affirmations de comportement PyTorch 2.11 / torchvision 0.26 rejouées (`scripts/claims1.py`, `scripts/claims2.py`) ; expérience 🔮 20.26 rejouée (`scripts/exp26.py`, poids MobileNet en cache) ; ∂ 20.5 et ∂ 20.8 relues ligne à ligne ; encadrés « Le livre » confrontés au texte du livre ; 🕰️ revérifiés sur le web.

## Constats

1. **MAJEUR · `01_fiche.md:341` (tableau des pièges) · repris dans `05_solutions.md:89` et `:221`**
   - Extrait (fiche) : « | De mauvais labels pour la loss | des labels one-hot `float` pour `CrossEntropyLoss`, ou des entiers pour `BCEWithLogitsLoss` | entiers `int64` de forme `(N,)` pour l'une… »
   - Extrait (20.2, Dans ta copie) : « des entiers de 0 à 25, de forme `(64,)` et de dtype `torch.int64` (`torch.long`), pas des vecteurs one-hot » ; (20.13, erreurs fréquentes) : « laisser les labels en `uint8` (`CrossEntropyLoss` veut des `int64`) ».
   - Problème : fait faux pour PyTorch 2.11. Depuis la 1.10, `CrossEntropyLoss` accepte comme cible des **probabilités par classe** (float de forme `(N, K)`) : des one-hot `float` donnent exactement la même loss que les indices. Rejoué : `F.cross_entropy(logits, yi)` = `F.cross_entropy(logits, F.one_hot(yi, 3).float())` = 1,68847. Docstring 2.11 : « The `target` that this criterion expects should contain either: Class indices… or Probabilities for each class ». De même, des labels `uint8` passent sur CPU (même loss), alors que `int32` échoue (`RuntimeError: expected scalar type Long but found Int`), comme des réels de forme `(N,)` (`…but found Float`). La cheatsheet dit d'ailleurs le contraire de la fiche : `pytorch.md:34` « `labels` en `torch.long` ou des probabilités (ch. 18) ». Deux documents se contredisent.
   - Correction : fiche, colonne Exemple : « des labels réels de forme `(N,)` (ou `int32`) pour `CrossEntropyLoss`, ou des entiers pour `BCEWithLogitsLoss` » ; colonne « Ce qu'il faut faire » : « des indices `int64` de forme `(N,)` (ou des probabilités `float` de forme `(N, K)`, utiles pour le lissage des labels) pour l'une… ». 20.2 : « …de dtype `torch.int64`, la forme la plus simple (des vecteurs one-hot `float` de forme `(64, 26)` marcheraient aussi : `CrossEntropyLoss` accepte des probabilités, mais c'est plus lourd) ». 20.13 : remplacer l'erreur par « des labels `int32` (fréquents après un `np.array` sous Windows avec NumPy 1.x, ou un `astype(np.int32)`) : `CrossEntropyLoss` veut des `int64` ».

2. **MINEUR · `05_solutions.md:103` et `:104` (✏️ 20.4)**
   - Extraits : « avec $m = 0{,}1$, la mémoire est d'environ 10 mini-batches (ch. 19) » ; « inverser $m$ et $1 - m$ (la convention de Keras) ».
   - Problème : le §22 (session 40) fixe « Le momentum de la batchnorm s'écrit $\rho$ dans tout le chapitre ($m$ = taille du mini-batch) » ; ici $m$ désigne le momentum, alors que la même solution (et ∂ 20.8) prend $m$ pour la taille du mini-batch.
   - Correction : « avec $\rho = 0{,}1$… » et « inverser $\rho$ et $1 - \rho$ (la convention de Keras) ».

3. **MINEUR · `01_fiche.md:225` (§20.5) — contradiction avec le ch. 17**
   - Extrait : « seules les couches de **calcul** (dense, convolution, récurrente) et les **normalisations** ont des paramètres appris ; le dropout, le pooling, les activations et les couches utilitaires n'en ont pas ».
   - Problème : le ch. 17 corrige justement le livre sur ce point (`ch17_activations/01_fiche.md:136`, « Le livre, corrigé — la ReLU paramétrique » : la pente de la PReLU est « un **paramètre appris** ») ; `nn.PReLU` a un paramètre (le défi 20.31 l'autorise même, `ALLOWED_31`).
   - Correction : « …le dropout, le pooling, les activations (sauf la PReLU, dont la pente est apprise, ch. 17) et les couches utilitaires n'en ont pas ».

4. **MINEUR · `annexes/cheatsheets/pytorch.md:15`**
   - Extrait : « `x.sum(dim=0)`, `x.mean(dim=1, keepdim=True)` | réductions : le paramètre s'appelle `dim`, pas `axis` (ch. 20) ».
   - Problème : faux tel quel, et contraire à la fiche (`01_fiche.md:85` : « PyTorch accepte aussi `axis`, mais la documentation et les exemples écrivent `dim` »). Rejoué : `torch.randn(4, 3).sum(axis=0).shape` → `torch.Size([3])`, `mean(axis=1, keepdim=True)` → `(4, 1)`.
   - Correction : « le paramètre s'appelle `dim` (PyTorch accepte aussi `axis`, mais écris `dim`, comme la documentation) ; `keepdim`, sans s, au lieu du `keepdims` de NumPy ».

5. **MINEUR · `05_solutions.md:221` (20.13, erreurs fréquentes)**
   - Extrait : « `x.view(64, 1, 28, 28)` sur un tenseur non contigu ».
   - Problème : cette erreur ne peut pas se produire : passer de `(64, 28, 28)` à `(64, 1, 28, 28)` n'insère qu'un axe de taille 1, toujours compatible avec les pas mémoire. Rejoué : `torch.zeros(64, 28, 28).transpose(1, 2).view(64, 1, 28, 28)` réussit. En revanche, aplatir échoue : `torch.zeros(64, 1, 28, 28).transpose(2, 3).view(64, -1)` → `RuntimeError: view size is not compatible with input tensor's size and stride`.
   - Correction : « `x.view(64, -1)` sur un tenseur non contigu (par exemple après un `transpose`) : `RuntimeError: view size is not compatible…` ; `reshape` copie si besoin ».

6. **MINEUR · `04_indices.md:938` et `:943` (🔮 20.26) · `data/cards/cifar10.md:25`**
   - Extraits : « des photos de 32 × 32 pixels agrandies à 224 × 224 » ; « une photo de 32 × 32 pixels, agrandie à 224 × 224 » ; card : « les 200 premières photos de test, agrandies en 224 × 224 ».
   - Problème : l'énoncé (`build_ch20.py`, cellule [99]) et la solution disent « agrandies à 256 × 256, puis recadrées au centre en 224 × 224 » (correction du §22, session 41) ; rejoué : `MobileNet_V3_Small_Weights.DEFAULT.transforms()` → `resize_size=[256]`, `crop_size=[224]`. Les bords perdus comptent dans l'explication de la solution.
   - Correction : « agrandies à 256 × 256 puis recadrées au centre en 224 × 224 » aux trois endroits.

7. **MINEUR · `data/cards/fashion_mnist.md:21`**
   - Extrait : « 20 (early stopping d'un petit réseau convolutif, validé sur les 10 000 dernières images d'entraînement, 20.28 ; … »
   - Problème : 20.28 entraîne un **MLP** 784-256-256-10 (Linear, BatchNorm1d, ReLU, Dropout ; 270 346 paramètres, vérifié par la cellule de 20.28) ; aucune convolution.
   - Correction : « early stopping d'un MLP 784-256-256-10 avec batchnorm et dropout, validé sur… ».

8. **MINEUR · `data/cards/mnist.md:29`**
   - Extrait : « une première boucle d'entraînement PyTorch sur 10 000 puis 60 000 images, 20.23 ».
   - Problème : le notebook entraîne sur 12 000 images en mode rapide et 50 000 en mode complet (`N_TRAIN_23 = wb.by_mode(fast=12_000, full=50_000)` ; sortie : « training: 12000 images · validation: 10000 images ») ; `05_solutions.md:392` dit aussi 12 000 et 50 000.
   - Correction : « sur 12 000 images (50 000 en mode complet), validée sur les 10 000 dernières, 20.23 ».

9. **MINEUR · `02_exercices.md:381` et `05_solutions.md:3` — textes d'avant la session 41**
   - Extraits : « Ces exercices sont dans `03_notebook.ipynb` ; ils arriveront avec la prochaine session de génération du chapitre. » ; « les exercices du notebook seront résolus et exécutés dans `05_solutions.ipynb` ».
   - Problème : le chapitre est terminé (le notebook et ses solutions existent) ; aucun autre chapitre n'a gardé cette phrase (`grep` sur `chapitres/ch*/02_exercices.md`).
   - Correction : « Ces exercices sont dans `03_notebook.ipynb`. » ; « …les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas) ».

10. **MINEUR · `annexes/formulaire.md` (section Ch. 20, en-tête et ligne « statistiques glissantes »)**
    - Extraits : « Conventions de PyTorch ; une feature $j$, un mini-batch de $m$ valeurs par feature. » puis « $\sigma^{2,\text{run}} \leftarrow (1 - \rho)\,\sigma^{2,\text{run}} + \rho\,\frac{n}{n-1}\sigma_B^2$ ».
    - Problème : $n$ n'est pas défini dans le formulaire (il y est appelé $m$), alors que la fiche (`01_fiche.md:161-163`) prend $m$ = exemples et $n$ = valeurs par feature.
    - Correction : écrire $\frac{m}{m-1}$ dans le formulaire (cohérent avec son en-tête et avec ∂ 20.8 Q8, où $m = N \times H \times W$).

11. **MINEUR · `04_indices.md:74` (Q4, indice 2) et `:263` (R2, indice 3) — réponses données hors du premier item (BIBLE §12)**
    - Extraits : « Une activation appliquée dans la couche ou juste après donne le même nombre. » ; « Pour b), $\log(e^{1000} + e^{0}) \approx 1000$, donc la composante vaut environ $0 - 1000$ ».
    - Problème : la première phrase énonce le verdict de Q4 c) (« Vrai : … calcule la même chose ») dès l'indice 2, et « donne le même nombre » est ambigu (nombre de quoi ?) ; la seconde donne la valeur de R2 b) (−1 000) alors que l'indice 3 ne résout que le premier item.
    - Correction : Q4, indice 2 : « Pour c), compare ce que calculent les deux écritures, étape par étape. » ; R2, indice 3 : « Pour b), écris $\mathrm{log\_softmax}(\mathbf z)_2 = 0 - \log(e^{1000} + e^{0})$ et majore le logarithme par $1000 + \log 2$ ».

12. **MINEUR · `05_solutions.md:265` (20.16, erreurs fréquentes) · SUGGESTION sur `figures/dropout.png`**
    - Extrait : « un masque par exemple (`rng.random(x.shape[1:])`) partagé par tout le mini-batch ».
    - Problème : contradictoire : `x.shape[1:]` tire **un seul** masque pour tout le mini-batch, ce qui est justement l'erreur (le contrat veut un masque tiré valeur par valeur). Dans la figure, les panneaux « mini-batch 1 (entraînement) » et « mini-batch 2 » dessinent un seul réseau coupé par mini-batch, ce qui suggère la même erreur, alors que la fiche dit « indépendamment pour chaque exemple du mini-batch » (l. 145).
    - Correction : « un seul masque (`rng.random(x.shape[1:])`) partagé par tous les exemples du mini-batch » ; figure (`tools/chapters/figures_ch20.py`, `fig_dropout`) : titres « passe avant 1, un exemple » et « passe avant 2, un exemple », ou sous-titre « (chaque exemple du mini-batch a son propre tirage) ».

13. **MINEUR · §3, paraphrases proches du livre · `01_fiche.md:95` (§20.3.1) et `01_fiche.md:198-199` (§20.4.8)**
    - Fiche l. 95 : « La couche d'entrée ne contient pas de neurones : c'est seulement la mémoire où l'on place les données avant de les envoyer à la première couche de calcul […] Les bibliothèques la créent pour nous […] On s'en soucie quand on veut transformer les données avant la première couche […] Attention au vocabulaire : certains auteurs appellent « couche d'entrée » la première couche de neurones. »
    - Livre p. 879-880 : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session).
    - Fiche l. 198-199 : « un zero-padding : une bordure de zéros, d'une épaisseur choisie, entoure l'image (ou le volume) ; un suréchantillonnage : le tenseur grandit, souvent par simple répétition de ses valeurs ; c'est l'opération qui défait, en taille, un pooling. »
    - Livre p. 897 : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session)
    - Problème : mêmes idées, dans le même ordre, phrase pour phrase.
    - Correction (l. 95) : « En PyTorch, la couche d'entrée n'est pas un objet : c'est le tenseur passé à `model(x)` (p. 879-880). Elle n'a pas de neurones ; on n'y pense que pour préparer les données avant la première couche (standardiser, recadrer). Méfie-toi du mot : certains textes l'emploient pour la première couche cachée. » (l. 198-199) : « zero-padding (`nn.ZeroPad2d(k)`) : $H \times W$ devient $(H + 2k) \times (W + 2k)$ ; suréchantillonnage (`nn.Upsample`) : chaque valeur répétée $s \times s$ fois, $H \times W$ devient $sH \times sW$, la réciproque, en taille seulement, d'un pooling. »

14. **SUGGESTION · `05_solutions.md:165` (⚖️ 20.11, Q5)**
    - Extrait : « les obligations des systèmes à haut risque de l'annexe III (et, par ricochet, ce droit de l'art. 86) s'appliqueront à partir du 2 décembre 2027 (règlement omnibus 2026/1744) ».
    - Problème : la date du 2 décembre 2027 pour l'annexe III est confirmée (Cuatrecasas : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session) ; euaiact.com, FPF), mais aucune des trois sources consultées ne dit que l'art. 86 (chapitre IX) est décalé : « par ricochet » est une inférence non sourcée.
    - Correction : « …s'appliqueront à partir du 2 décembre 2027 (règlement omnibus 2026/1744) ; l'art. 86 vise les décisions fondées sur un tel système : vérifie sa date d'application dans le texte consolidé (EUR-Lex) avant de l'invoquer ».

15. **SUGGESTION · `01_fiche.md:157`**
    - Extrait : « (avec $\gamma_j = \sigma_j$ et $\beta_j = \mu_j$, il peut même annuler la normalisation) ».
    - Problème : exact seulement pour $\epsilon = 0$ ; la sortie vaut $\hat x\,\gamma + \beta$ avec $\hat x = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$. La solution de 📄 20.12 (Q2) écrit $\gamma = \sqrt{\mathrm{Var}[x]}$, comme l'article.
    - Correction : « avec $\gamma_j = \sqrt{\sigma_j^2 + \epsilon}$ et $\beta_j = \mu_j$ ».

16. **MINEUR · `annexes/erreurs_frequentes.md:239`**
    - Extrait : « *(complété à partir du ch. 20)* », placé **après** les 13 lignes du ch. 20 de la section PyTorch.
    - Problème : note de chantier périmée.
    - Correction : la supprimer.

17. **MINEUR · `05_solutions.md:396` (20.23, variante)**
    - Extrait : « la même expérience (mode rapide, 3 epochs, même modèle de départ) avec `torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)` au lieu d'Adam donne **95,3 %** en validation, contre 93,3 % pour Adam et 91,6 % pour SGD sans momentum ».
    - Problème : ce n'est pas « la même expérience » que celle du notebook, qui affiche 93,5 % pour Adam (sortie : `3 … 0.9355`, et la réponse de 20.23 dit « environ **93,5 %** »). Rejoué (`scripts/exp23.py`, `exp23b.py`) : sans évaluation entre les epochs, Adam 93,29 %, SGD + momentum 95,30 %, SGD 91,60 % (les chiffres de la variante) ; avec l'évaluation après chaque epoch, comme dans la boucle du notebook, Adam 93,55 %, SGD + momentum 95,11 %, SGD 91,68 %. Cause : chaque `iter()` d'un `DataLoader` **sans** générateur (ici `VAL_LOADER_23`) tire sa graine de base dans le générateur global, ce qui change les masques de dropout des epochs suivantes. L'apprenant qui refait la variante avec sa boucle de 20.23 ne retrouve donc pas 93,3 %.
    - Correction : « …donne **95,1 %** en validation, contre 93,5 % pour Adam (le résultat ci-dessus) et 91,7 % pour SGD sans momentum » (wb.check n'est pas concerné). Ajout possible à l'encadré « reproductibilité » de la fiche (l. 286) : « même un `DataLoader` sans `shuffle` consomme le générateur global à chaque parcours : une évaluation de plus change les masques de dropout qui suivent ».

18. **MINEUR · `01_fiche.md:237` (🕰️ VGG16 et ses successeurs) — unités contraires à ✏️ 20.6 h**
    - Extrait : « ses poids pèsent 528 Mo et il fait 71,6 % top-1 sur ImageNet (torchvision), quand **ResNet-18** fait 69,8 % avec 45 Mo, **MobileNet V3 Small** 67,7 % avec 10 Mo ».
    - Problème : ✏️ 20.6 g et h enseignent que les « MB » de torchvision sont des Mio : 527,8 Mio = 553,4 Mo (réponse B de 20.6 h ; le notebook de 20.21 dit d'ailleurs « VGG16 occupe 553 Mo »). Valeurs rejouées (`weights.meta["_file_size"]`) : 527,796 ; 44,661 ; 9,829 (en Mio).
    - Correction : « ses poids pèsent 528 Mio (553 Mo) … avec 45 Mio … avec 10 Mio » (ou « 553 Mo, 47 Mo, 10 Mo »).

19. **MINEUR · `04_indices.md:393` (✏️ 20.6, indice 3)**
    - Extrait : « pour e), ajoute les deux autres couches denses ; pour f) et g), divise ou multiplie par 4 octets ; ».
    - Problème : f) demande la **part** des paramètres des trois couches denses, en % : elle ne se calcule pas avec 4 octets ; l'indice oriente vers une fausse piste.
    - Correction : « pour f), divise les paramètres des trois couches denses par le total, puis multiplie par 100 ; pour g), multiplie le total par 4 octets et divise par $10^6$ ; ».

20. **MINEUR · `annexes/cheatsheets/pytorch.md:76` et `annexes/erreurs_frequentes.md` (ligne `ValueError: Expected more than 1 value per channel when training`)**
    - Extraits : « jeter le dernier mini-batch incomplet (une batchnorm refuse un mini-batch d'un seul exemple) » ; « une batchnorm en mode entraînement reçoit un mini-batch d'un seul exemple (souvent le dernier) ».
    - Problème : vrai pour `BatchNorm1d` sur `(N, D)`, pas pour `BatchNorm2d` : rejoué, `nn.BatchNorm2d(3)(torch.randn(1, 3, 2, 2))` passe ; seul `(1, 3, 1, 1)` lève l'erreur (la fiche, l. 163, est juste : « `nn.BatchNorm2d` l'accepte tant qu'une image a plus d'un pixel »).
    - Correction : « (une `BatchNorm1d` refuse un mini-batch d'un seul exemple) » ; erreurs fréquentes : « …reçoit une seule valeur par feature (un mini-batch d'un seul exemple pour `BatchNorm1d`, souvent le dernier) ».

## Ce qui a été vérifié sans écart

- **Réponses courtes** (117) : toutes recalculées depuis les énoncés et identiques à `05_solutions.md` et aux cellules `answer` (20.1 à 20.9, quiz, rappels) ; les messages `mistakes` des parties 0 associent bien une phrase à une mauvaise valeur cohérente (ex. 20.4 j : 0,1 + 0,9 × 20/3 pour l'inversion de ρ ; 20.6 h D : 58,9 Mo de convolutions). Exemples : 20.4 k = 5,752 (PyTorch, `eps=1e-12` : 5,75234) ; 20.6 : 14 714 688 / 138 357 544 / 89,4 % / 553,4 Mo / 527,8 Mio ; 20.7 g = 44 (0,9⁴³ = 0,0108, 0,9⁴⁴ = 0,0097) ; 20.9 : 4,40 × 10¹⁸, 5,1 j, 510 j, 2,21 Go ; ResNet-18 17,4 h ; 2¹⁰⁰ = 1,27 × 10³⁰ ; $\ln(1 + 9/e) = 1{,}4612$ (rejoué : 1,46115).
- **∂ 20.5** (espérance, variance $a^2 p/(1-p)$, $2^n$ sous-réseaux, règle du test, learning rate effectif $\eta/(1-p)^2$, Jensen : $\mathbb E[h] = 1/4$ contre $h(\mathbb E z) = 0$) et **∂ 20.8** (chemins par μ et σ², $\partial\hat x_i/\partial x_j = s(\delta_{ij} - \frac1m - \frac1m\hat x_i\hat x_j)$, formule en $O(m)$, $\sum_j \hat x_j\,\partial L/\partial x_j = \gamma s\,\frac{\epsilon}{\sigma^2+\epsilon}\sum g_i\hat x_i$, cas $m = 1$ et $m = 2$) : justes ligne à ligne.
- **Variantes chiffrées** recalculées : 20.1 (64, 150528), 20.3 (somme 12), 20.4 (9 ; 1 ; 0,5 ; 2,1), 20.6 VGG11 (9 220 480 / 132 863 336), 20.7 (1,3416), 20.15 (0,1906), 20.17 RMSNorm (1,6971 ; `nn.RMSNorm` donne 1,69706), 20.18 (203 530), 20.20 (1 216 ; LSTM 7 936 ; RNN 1 984), 20.25 (patience 3 → epoch 10), 20.26 (rejoué : 7 % de chats domestiques, **11 %** de félins, **77 %** des chiens CIFAR-10 classés parmi les 118 races).
- **Sorties de `05_solutions.ipynb`** et chiffres de `05_solutions.md` concordent (20.13 : 0,1291 ; 20.23 : 1,9155 / 0,7755 / 93,5 % ; 20.24 : 87,8 % ; 20.26 : 7 %, 0,130, 0,380, 68 % « prayer rug », 75 % ; 20.28 : epoch 10, arrêt à 13, 0,389, 86,3 % ; tableaux de 20.29 et 20.30 ; 20.31 : 90,45 %, 270 346 paramètres) ; nombres de tests (27, 43, 32) conformes ; données de `wb.synth.training_curves` cohérentes avec 20.25 (B : minimum à l'epoch 7 ; E : 99,5 % ; C : 0,80-0,83 et 78 % ; D : validation sous l'entraînement à toutes les epochs).
- **Comportements PyTorch 2.11 / torchvision 0.26 rejoués** (tous conformes, sauf constats 1, 4, 5, 20) : dtype par défaut, `from_numpy` (partage, `float64`), `torch.var` ($n - 1$), init de `nn.Linear` (borne $1/\sqrt{n_{in}}$, variance $1/(3n_{in})$), `nn.Dropout` (valeurs 0 et 1,25, masques par valeur, identité en `eval`), batchnorm (`ValueError` pour une seule valeur par canal, `BatchNorm2d` sur une image de 2 × 2 accepté, `eps=0` refusé : « batch_norm eps must be positive », `layer_norm` l'accepte ; statistiques 20.4 : 0,4 / 1,0 / 1,5667 ; `affine=False` sans paramètres ; buffers et `num_batches_tracked`), formes de 20.14 et variantes, `Upsample` en `nearest`, messages d'erreur cités (`mat1 and mat2 shapes cannot be multiplied (1x50176 and 25088x4096)`, `cannot assign module before Module.__init__() call`, `optimizer got an empty parameter list`, `Trying to backward through the graph a second time`, `Weights only load failed`, `result type Float can't be cast…`), `torch.load` sûr sur un `state_dict` et sur les états de générateurs, liste ordinaire → 0 paramètre, `AdamW` `weight_decay=0.01`, métadonnées torchvision (VGG16 71,592 / 90,382 / 527,796 Mio / 15,47 GFLOPS / 138 357 544 ; ResNet-18 69,758 / 44,661 / 1,814 ; MobileNet V3 Small 67,668 / 9,829 / 2 542 856 ; prétraitement 256 → 224 ; indices 281-293 des félins, 151-268 des chiens, 741 « prayer rug »), VGG16 construit sur `meta`.
- **Encadrés « Le livre : » et « Le livre, corrigé »** : tous fidèles au texte (dropout tiré « at the start of each epoch », p. 885 ; batchnorm « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session), p. 887, et justifiée comme régularisation ; division par 2, p. 887-888 ; « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session), p. 902 ; renvoi au « Chapter 8 » pour le learning rate et évaluation « on the test data », p. 911 ; pooling « falling out of favor », p. 893 ; aucune connexion résiduelle dans le livre ; « explainability is still difficult… we need to stay vigilant », p. 922) ; numéros de pages et de figures exacts ; le livre ne mentionne nulle part ResNet ni la layer norm.
- **Flashcards** (28) : toutes justes ; une idée par carte.
- **Figures** : conformes à leur texte alternatif et à la fiche (voir constat 12 pour `dropout.png`).
- **Cohérence avec les ch. 16 à 19** : init de `nn.Linear` (ch. 16), dérivées et softmax (ch. 17), réseau minuscule de 18.4 (gradients et 0,0531 retrouvés), AdamW de `torch.optim` (ch. 19), early stopping (même sémantique qu'au ch. 9), conventions de régularisation du formulaire : pas de contradiction, sauf la PReLU (constat 3).

## Affirmations 🕰️ (et assimilées) revérifiées sur le web

| Affirmation (fiche ou solutions) | Verdict | Source |
|---|---|---|
| `nn.Dropout` : éléments « chosen independently for each forward call », facteur $1/(1-p)$, identité en évaluation | confirmé | docstring de `torch.nn.Dropout` 2.11 installée ; [docs PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) |
| batchnorm : variance biaisée pour normaliser, non biaisée pour la moyenne glissante, `momentum=0.1`, formule $(1-\rho)\hat x + \rho x_t$ | confirmé | docstring de `torch.nn.BatchNorm1d` 2.11 ; [docs PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) |
| PaLM entraîné sans dropout (§5) | confirmé : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session) | [Chowdhery et coll., arXiv:2204.02311](https://arxiv.org/pdf/2204.02311) |
| Liu et coll. (2025), « Drop Dropout on Single-Epoch Language Model Pretraining » | confirmé (H. Liu, J. Bauer, C. Manning ; « we advocate to drop dropout during single-epoch pretraining ») | [arXiv:2505.24788](https://arxiv.org/abs/2505.24788) |
| Santurkar et coll. : la stabilité des distributions « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session), paysage plus lisse, gradients plus prédictifs | confirmé | [arXiv:1805.11604](https://arxiv.org/abs/1805.11604) |
| VGG : 7,3 % top-5 = soumission à 7 réseaux (tableau 7), 2ᵉ place en classification ; arrêt après 370 000 itérations (74 epochs) ; 2-3 semaines par réseau sur 4 Titan Black | confirmé | [Simonyan et Zisserman, arXiv:1409.1556](https://arxiv.org/pdf/1409.1556) |
| torchvision : VGG16 71,6 % top-1, 90,4 % top-5, 527,8 Mio ; ResNet-18 69,8 % ; MobileNet V3 Small 67,7 %, 9,8 Mio | confirmé (métadonnées des poids de torchvision 0,26, identiques aux pages de modèles) | [torchvision.models](https://docs.pytorch.org/vision/stable/models.html) |
| `torch.load` en `weights_only=True` par défaut depuis 2.6 | confirmé : « Starting in version 2.6, `torch.load` will use `weights_only=True` if the `pickle_module` argument is not passed » | [notes de sérialisation 2.6](https://docs.pytorch.org/docs/2.6/_sources/notes/serialization.rst.txt), [dev-discuss](https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573) |
| Omnibus numérique 2026/1744 : obligations de l'annexe III au 2 décembre 2027 | confirmé (publié au JO le 24 juillet 2026) ; **rien** sur l'art. 86 (constat 14) | [Cuatrecasas](https://www.cuatrecasas.com/en/spain/intellectual-property/art/digital-omnibus-ai-has-been-published), [euaiact.com](https://www.euaiact.com/digital-omnibus-ai), [FPF, chronologie](https://fpf.org/wp-content/uploads/2026/07/FPF-EU-AI-Act-Timeline-2026-R2.pdf) |
| CJUE SCHUFA (C-634/21), 7 décembre 2023 : un score de crédit peut être une décision automatisée (art. 22 RGPD) | confirmé | [A&O Shearman](https://www.aoshearman.com/en/insights/ao-shearman-on-data/cjeu-rules-that-a-credit-score-constitutes-automated-decision-making-under-the-gdpr), [Bird & Bird](https://twobirds.com/en/insights/2023/global/key-takeaways-from-the-schufa-case-of-the-cjeu) |
| Directive 2023/2225 (crédit aux consommateurs) : application au 20 novembre 2026 ; intervention humaine, explication, point de vue, réexamen en cas d'évaluation automatisée | confirmé (transposition au 20 novembre 2025, application au 20 novembre 2026) | [Schoenherr](https://schoenherr.eu/content/consumer-credit-directive-ii-consumer-credit-legislation-for-the-digital-age) |
| `CrossEntropyLoss` n'accepte que des indices `int64` (sous-entendu du tableau des pièges) | **infirmé** : la cible peut aussi être des probabilités par classe (constat 1) | docstring de `torch.nn.CrossEntropyLoss` 2.11 ; [docs PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |

Pas rejouée (machine chargée, replay interrompu) : la variante « mode complet » de 20.28 (epoch 7, 88,6 %, 0,305).

Non revérifiées sur le web (vérifiées localement ou connues, sans écart constaté) : dropout 0,1 de BERT, GPT-2, T5 ; RMSNorm dans LLaMA et `nn.RMSNorm` (présent dans torch 2.11) ; convention inverse du `momentum` de Keras ; GoogLeNet 6,67 % ; 93 Mo d'activations par image (CS231n).

## Bilan

**Solide.** Les calculs du chapitre sont justes partout où je les ai refaits : les 117 réponses vérifiées, les deux preuves ∂, les variantes chiffrées et les sorties du notebook exécuté concordent entre fiche, énoncés, indices, solutions et figures. Les formules de PyTorch (dropout inversé, batchnorm biaisée et non biaisée, ρ, `eps`, layer norm, L2 et `weight_decay`, early stopping, connexion résiduelle) sont exactes et conformes à la référence `regularization.py`. Les encadrés « Le livre » sont fidèles et justes envers le livre, et les affirmations datées (VGG, torchvision, PaLM, Santurkar, omnibus, SCHUFA, CCD II) tiennent.

**Les trois risques principaux.**
1. Des affirmations PyTorch trop absolues dans les pièges et les erreurs fréquentes (one-hot « interdits », `uint8` « refusés », `axis` « pas accepté », `view` qui « échoue » là où il ne peut pas échouer, une batchnorm qui « refuse » une seule image) : un apprenant qui essaie verra le contraire (constats 1, 4, 5, 20).
2. Des chiffres du chapitre recopiés dans les annexes et les data cards qui ont divergé de la version finale (12 000 et 50 000 images, MLP et pas CNN, 256 puis 224, unités Mo et Mio, $n$ contre $m$, ρ écrit $m$) (constats 2, 6, 7, 8, 10, 18).
3. Des résultats d'expériences qui dépendent de l'état du générateur global sans le dire : la variante de 20.23 ne se reproduit pas avec la boucle de l'apprenant, parce que chaque parcours d'un `DataLoader` sans générateur consomme ce générateur (constat 17). C'est aussi une leçon de reproductibilité qui mériterait une ligne dans la fiche.
