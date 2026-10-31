# Rapport A7 — exactitude des ch. 27 (GAN) et 28 (applications créatives)

**Périmètre** : `chapitres/ch27_gan`, `chapitres/ch28_creatif` (fiches, exercices, indices, solutions, notebooks exécutés, flashcards, figures), leurs lignes de `annexes/formulaire.md`, `glossaire.md`, `erreurs_frequentes.md`, les data cards `images.md`, `pretrained.md`, `mnist.md`, `fashion_mnist.md`, les renvois des autres chapitres vers 27 et 28 ; commit `b961aa3`. Scripts de vérification : `scripts/nbdump.py` (extraction des cellules), `scripts/check27.py`, `scripts/check28.py` (recalculs), `scripts/snip27.py` (boucle de la fiche et dérivées sous torch 2.11), `scripts/rt27.py` (oubli de `detach`), `scripts/unpack27.py` (données cachées de 27.13 et 27.14), dans ce dossier ; code de l'auteur lu dans une copie temporaire de `bssrdf/DeepLearningBookCode-Volume2` (supprimée).

## Constats

### 1. MINEUR · `chapitres/ch27_gan/05_solutions.md:255` (27.15, erreurs fréquentes) ; `01_fiche.md:190`
**Extrait :** « un bruit normal au lieu d'uniforme (la traînée de départ et la trajectoire de la fig. 27.20 viennent d'un bruit uniforme sur $[0, 1]$) »
**Problème :** le texte du livre (p. 1586) annonce bien un bruit uniforme sur [0, 1], mais le code qui a produit les fig. 27.20 et 27.21 (`Chapter27-GANs/GANs-Notebook-2.ipynb` de la copie `bssrdf/DeepLearningBookCode-Volume2`, licence MIT) tire un bruit **normal** : `def get_batch_of_normal_noise(mean=0, std=1, ...): return np.random.normal(mean, std, shape)`, utilisé par `train_generator`, le pas de D sur les faux et le tracé. L'affirmation causale sur l'origine de la fig. 27.20 est donc fausse. Ce même code fait aussi un pré-entraînement de D d'une epoch, et le panneau « Epoch 0 » est tracé **après** la première epoch (`for epoch in range(21): train...; if epoch in plot_epochs: plot`).
**Correction :** remplacer par « un bruit normal au lieu d'uniforme : l'énoncé suit le texte du livre (bruit uniforme sur [0, 1]) ; un bruit centré déplace le point de départ du nuage généré ». En option, une demi-phrase dans la fiche §27.5 : « (le code de l'auteur tire en fait un bruit gaussien) ».

### 2. MINEUR · `chapitres/ch27_gan/flashcards.csv:12` (carte « Somme nulle, minimax, équilibre de Nash »)
**Extrait :** « Minimax : choisir le coup dont le pire résultat est le meilleur. »
**Problème :** c'est la définition que la fiche (§27.4.4, encadré 🧮) et l'Ex 27.2 a) donnent au **maximin** (« Ligne joue prudemment le coup dont le pire gain est le meilleur (le maximin, $\max_i \min_j g_{ij}$) »), le minimax étant $\min_j \max_i g_{ij}$, le coup de Colonne. L'apprenant qui révise avec la carte confondra les deux valeurs de l'Ex 27.2 (0 et 3).
**Correction :** « Minimax : choisir le coup qui laisse le plus petit des meilleurs gains à l'adversaire, $\min_j \max_i g_{ij}$ (son pendant pour l'autre joueur, le coup dont le pire gain est le meilleur, est le maximin). »

### 3. MINEUR · `chapitres/ch27_gan/figures/jeu_gan.png` (produite par `tools/chapters/figures_ch27.py:91`)
**Extrait :** `ax.text(7.4, 0.75, "étape de $G$ : faux → label 1, $D$ gelé (seul l'optimiseur de $G$ avance)", ha="center", ...)`
**Problème :** la légende de l'étape de G, centrée en x = 7,4 et longue, passe **sous** la boîte « générateur G » : on lit « …e de G : faux → label 1… », les premières lettres sont cachées, et la fin touche le bord droit de l'image.
**Correction :** couper la ligne en deux (`"étape de $G$ : faux → label 1\n$D$ gelé (seul l'optimiseur de $G$ avance)"`) et la centrer à droite de la boîte (x ≈ 8,0), ou la placer sous la flèche $G(\mathbf{z}) \to D$ avec `zorder` supérieur à celui des boîtes.

### 4. MINEUR · `chapitres/ch27_gan/01_fiche.md:273` (🕰️ « évaluer un générateur »)
**Extrait :** « il dépend du nombre d'échantillons (biaisé vers le haut pour de petits mini-batchs, d'où des comparaisons toujours au même nombre, souvent 50 000) »
**Problème :** le biais du FID dépend du nombre d'**images** sur lesquelles on estime moyenne et covariance, pas d'un mini-batch (notion d'entraînement, §5). L'Ex 27.23 le montre d'ailleurs avec « 200, 500, 1 000 et 2 000 images ».
**Correction :** « (biaisé vers le haut quand on l'estime sur peu d'images, d'où des comparaisons toujours au même nombre d'images, souvent 50 000) ».

### 5. MINEUR · `chapitres/ch27_gan/01_fiche.md:64` contre `:119`
**Extrait :** l. 64 « $\mathbf{z} \sim p_z$ le bruit d'entrée (souvent une gaussienne $\mathcal{N}(\mathbf{0}, \mathbf{I})$ de dimension 64 à 128) » ; l. 119 « aujourd'hui, le plus souvent, une gaussienne de dimension 64 à 512 ».
**Problème :** deux fourchettes différentes pour la même chose dans la même fiche (le DCGAN utilise 100, StyleGAN 512).
**Correction :** écrire « de dimension 64 à 512 » aux deux endroits.

### 6. MINEUR · `chapitres/ch27_gan/01_fiche.md:251` (⚠️ « modal collapse et ce qui se passe ensuite »)
**Extrait :** « Le nom de « scénario Helvetica » vient de l'article de Goodfellow et coll. lui-même. »
**Problème :** présenté comme une correction du livre, qui cite pour ce nom le sketch [Popper12] de la BBC (p. 1598). Or le nom vient bien de ce sketch : Goodfellow et coll. (2014, §6) l'empruntent à la série comique *Look Around You* (épisode « Calcium »), et c'est leur article qui l'a fait entrer en apprentissage automatique. La phrase corrige donc à tort le livre sur ce point. Source : billet « The Helvetica scenario », blog.oppedahl.com (« the terminology was coined in the pilot episode » de *Look Around You*).
**Correction :** « Le nom de « scénario Helvetica » est un clin d'œil de Goodfellow et coll. (2014) à la série comique *Look Around You* que cite le livre : c'est leur article qui l'a fait entrer en apprentissage automatique. »

### 7. MINEUR · `chapitres/ch28_creatif/01_fiche.md:99` (🧮 « dériver par rapport à une image en PyTorch »)
**Extrait :** « Le livre normalise aussi le gradient par sa valeur absolue moyenne, une astuce qui rend le pas indépendant de l'échelle de l'activation (Adam le fait de lui-même, à sa façon). »
**Problème :** le texte du ch. 28 du livre ne parle d'aucune normalisation du gradient (§28.2.2, p. 1609-1610 : il laisse de côté les « détails d'implémentation »). Le code de l'auteur pour la visualisation des filtres (`Chapter21-CNNs/CNN-Notebook-4.ipynb`, repris de Chollet) divise par la **moyenne quadratique** : `grads /= (K_backend.sqrt(K_backend.mean(K_backend.square(grads))) + 1e-5)` ; son code du deep dream (`Creative-Applications-Notebook-1.ipynb`) passe le gradient à `scipy.optimize` sans le normaliser. La division par la valeur absolue moyenne est celle du deep dream de Google, que la fiche cite justement au §28.3 (« un gradient normalisé à chaque pas ») et que fait 📦 28.16.
**Correction :** « Le code de visualisation qu'adapte le livre (celui de Chollet) divise aussi le gradient par sa moyenne quadratique, et le deep dream de Google par sa valeur absolue moyenne : deux astuces qui rendent le pas indépendant de l'échelle de l'activation (Adam le fait de lui-même, à sa façon). »

### 8. MINEUR · `chapitres/ch28_creatif/01_fiche.md:121` et `:214` (renvois du §28.5 du livre)
**Extrait :** l. 121 « la génération de texte mot à mot n'est pas au « ch. 22 », qui travaille par caractères, mais au §24.5.11 » ; l. 214 « ce que le livre lui-même faisait au §24.5.11, où il montrait que le mot le plus probable donne beaucoup de répétitions ».
**Problème :** (1) le §22.9 du livre (p. 1076-1077) décrit bien la génération mot à mot (un numéro par mot, le mot suivant prédit, « une approche tout à fait raisonnable ») avant de choisir les caractères pour aller plus vite : le renvoi du livre n'est pas faux, il est seulement incomplet. Le §24.5.11 commente surtout le code **lettre par lettre** et ne fait que nommer le notebook mot à mot. (2) Ce que le §24.5.11 dit des répétitions porte sur le **caractère** le plus probable (p. 1359 : « … » (citation du livre retirée de la version versionnée, §3)), pas sur le mot le plus probable.
**Correction :** l. 121 : « la génération de texte mot à mot n'est qu'esquissée au ch. 22 (§22.9), qui choisit de travailler par caractères ; le code mot à mot est au §24.5.11 (son second notebook) » ; l. 214 : « …au §24.5.11, où il notait que choisir toujours le caractère le plus probable donne beaucoup de mots répétés ».

### 9. MINEUR · `chapitres/ch28_creatif/figures/variation_totale.png` (`tools/chapters/figures_ch28.py:176-178`) et `01_fiche.md:185`
**Extrait :** titre du 3ᵉ panneau « bruit lissé (flou) » ; légende de la fiche « La variation totale d'une photo, de la même photo bruitée, et du bruit lissé ».
**Problème :** le 3ᵉ panneau n'est pas un bruit lissé mais la **photo bruitée, floutée** (`smooth = gaussian_filter(noisy, 1.2)`). Le titre laisse croire que la figure lisse un bruit pur ; or l'intérêt du panneau est justement qu'un flou ramène la variation totale de la photo bruitée (637) sous celle de la photo d'origine (227 → 31), au prix des détails.
**Correction :** titre « photo bruitée, floutée » ; légende « La variation totale d'une photo, de la même photo bruitée, puis de la photo bruitée floutée ».

### 10. MINEUR · `chapitres/ch28_creatif/05_solutions.md:120` (🧮 28.6, point 4)
**Extrait :** « $14{,}7 / 138{,}3 \approx 11\,\%$ »
**Problème :** l'énoncé (`02_exercices.md:229`) donne « 138,4 millions de paramètres » (torchvision : 138 357 544). Le résultat (10,6 %, arrondi à 11 %) ne change pas, mais l'apprenant ne retrouve pas le nombre de l'énoncé.
**Correction :** « $14{,}7 / 138{,}4 \approx 11\,\%$ ».

### 11. SUGGESTION · `chapitres/ch27_gan/01_fiche.md:119`, `:239` ; `chapitres/ch28_creatif/01_fiche.md:125` (règle du §3)
**Extrait :** l. 119 « Le générateur est **déterministe** : un même $\mathbf{z}$ donne toujours le même échantillon, et les composantes de $\mathbf{z}$ jouent le rôle de **variables latentes**. » ; l. 239 « le discriminateur repère trop facilement les faux et les gradients qu'il renvoie poussent le générateur dans des directions presque aléatoires » ; ch. 28 l. 125 « Un réseau entraîné sur des fournitures de bureau ferait surgir des agrafeuses. »
**Problème :** trois phrases suivent de près, dans le même ordre, le texte du livre (p. 1577 « … » (citation du livre retirée de la version versionnée, §3) ; p. 1597 « … » (citation du livre retirée de la version versionnée, §3) ; p. 1619 « … » (citation du livre retirée de la version versionnée, §3)). Elles sont courtes et factuelles, d'où une simple suggestion.
**Correction :** l. 119 : « Comme tout réseau, le générateur est une fonction : à bruit égal, échantillon égal ; le bruit sert de code, de **variables latentes**. » ; l. 239 : « …le discriminateur sépare vrais et faux sans effort, et ses gradients, qui portent sur des millions de pixels à la fois, n'indiquent plus de direction utile au générateur » ; ch. 28 : créditer l'idée, « (l'exemple des agrafeuses est du livre) ».

### 12. SUGGESTION · `chapitres/ch27_gan/01_fiche.md:99` et `:198` (encadrés sur le round et la loss « idéale »)
**Extrait :** « Ce n'est pas l'algorithme qu'on utilise, ni celui du code que cite le livre. » ; « c'est exactement le palier des courbes du livre, qui sont donc à l'équilibre ».
**Problème :** le code de l'auteur (`GANs-Notebook-2.ipynb`, `train_gan_one_epoch`) confirme et précise ces deux corrections : il fait bien D (vrais), G, D (faux), G, mais **sur tous les échantillons**, sans la règle « n'apprendre que de ses erreurs » (c'est la variante « tous les échantillons » de 🔬 27.19) ; et ses courbes tracent, pour D, la seule BCE du demi-batch de faux du dernier mini-batch : un terme, donc $\ln 2 \approx 0{,}69$ à l'équilibre (la figure de sortie du notebook montre les deux courbes vers 0,69).
**Correction :** ajouter à l'encadré du §27.2.3 : « Le notebook de l'auteur lui-même enchaîne D, G, D, G sur tous les échantillons, sans cette sélection. » ; et au §27.5 : « (chacune de ses courbes trace un seul terme de cross-entropy) ».

### 13. MINEUR · `chapitres/ch27_gan/01_fiche.md:156` (🧮 `.detach()`) et `:298` (tableau des pièges)
**Extrait :** l. 156 « Sans lui, `loss_d.backward()` déposerait des gradients dans $G$ (inutiles, et ajoutés à ceux de l'étape suivante si l'on oublie `opt_g.zero_grad()`). » ; l. 298 « `D(fake)` dans `loss_d` : son `backward()` remplit les gradients de $G$, qui s'ajoutent à ceux de l'étape suivante ».
**Problème :** avec la boucle de la fiche (l. 139-153), où le même `fake` sert aux deux étapes, l'oubli de `detach` ne laisse pas seulement des gradients en trop : `loss_d.backward()` libère le graphe de $G$, et `loss_g.backward()` lève `RuntimeError: Trying to backward through the graph a second time` (vérifié avec torch 2.11, `scripts/rt27.py`). C'est ce que disent le corrigé de CP6.9 a) et `annexes/erreurs_frequentes.md:292` ; la fiche ne décrit que le cas où `G(z)` est recalculé (🐛 27.16). L'apprenant qui rencontre l'erreur ne la retrouve pas dans le tableau des pièges de son chapitre.
**Correction :** l. 156, ajouter : « Avec la boucle ci-dessus, où `fake` sert aux deux étapes, l'oubli se voit même tout de suite : le premier `backward()` libère le graphe de $G$, et `loss_g.backward()` lève « Trying to backward through the graph a second time ». » ; l. 298, colonne Exemple : « `D(fake)` dans `loss_d` : erreur « backward through the graph a second time » si `fake` resert à l'étape de $G$ ; sinon, des gradients en trop dans $G$ ».

## Bilan

**Ce qui est solide.** Aucune réponse vérifiée fausse : j'ai recalculé 52 valeurs du ch. 27 (`scripts/check27.py` : quiz chiffrés, ✏️ 27.1 à 27.4, ∂ 27.5 contrôlé sur des lois discrètes, ∂ 27.6 par autograd, 🧮 27.7, exemples de la fiche, variantes) et 60 du ch. 28 (`scripts/check28.py` : ✏️ 28.1, 28.2, 28.4, ∂ 28.3 et 28.5 par autograd, 🧮 28.6 et 28.7, top-p, tailles de SqueezeNet et de VGG16), toutes identiques aux réponses de `05_solutions.md`, aux `wb.record` et aux sorties exécutées ; les tableaux des solutions des notebooks (27.13 à 27.28, 28.11 à 28.29) correspondent aux sorties, aux arrondis près. Les formules (discriminateur optimal, $-\ln 4 + 2\,\mathrm{JS}$, dérivées $-D$ et $-(1-D)$, distance de Fréchet, Gram normalisée et facteur 4 de Gatys, gradient $4(FF^\top - S)F$, variation totale) sont justes ; la boucle de la fiche tourne avec torch 2.11 ; `pytest --impl=ref tests/test_ch27_gen_metrics.py` : 102 tests verts. Les encadrés « Le livre, corrigé » sont fondés sur le texte du livre (pages et sections vérifiées : p. 1558-1602 et 1603-1647, « 23.3 », ch. 24 pour le VAE, ch. 8 pour les mini-batchs, demi-million d'images, 10,90, bloc 1 conv 2, 0,025 / 1 / 0,0001, 250 itérations), et le code de l'auteur confirme le palier à 0,69. Huit affirmations 🕰️ revérifiées sur le web, toutes exactes : règlement (UE) 2026/1744 et délai au 2 décembre 2026, LPIPS (`vgg` pour optimiser), VGG16 dans torchvision (138 357 544 paramètres, 15,47 GFLOPS, 527,8 Mo), ADD en 1 à 4 pas, Stein et coll. (DINOv2 à la place d'Inception), ProGAN (2 à 6 fois, fondu en $\alpha$), statut discuté de *Nighthawks*, art. 53 et 111. Annexes (formulaire, glossaire, erreurs fréquentes), data cards et renvois des ch. 1, 6, 21, 22, 25, 29, de CP6 et de MP6 : cohérents avec les deux chapitres.

**Trois risques principaux.**
1. Des « Le livre, corrigé » qui corrigent à tort ou attribuent au livre ce qu'il ne dit pas (constats 6, 7, 8) : l'apprenant apprend à se méfier du livre sur des points où il avait raison.
2. Des affirmations sur l'origine des figures du livre qui contredisent le code de l'auteur (constat 1 : bruit normal, et non uniforme, dans la fig. 27.20).
3. Une fiche du ch. 27 qui n'annonce pas l'erreur la plus probable de sa propre boucle (constat 13), alors que CP6.9 la teste.

| Sévérité | Nombre |
|---|---|
| MAJEUR | 0 |
| MINEUR | 11 |
| SUGGESTION | 2 |
