# 27 · Réseaux antagonistes génératifs (GAN) — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels et des exercices ✏️ 27.1 à 27.4 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 27.Q1 — À quoi bon générer des données ?
a) **« AB »** : des données d'entraînement en plus, et des idées à proposer (jardins, meubles) ; compresser (C) ou retrouver une photo existante (D) n'inventent rien. b) **Faux** : un générateur qui recopierait ses données n'aurait rien appris ; on veut des échantillons **nouveaux**, de la même distribution. c) **B** : le discriminateur ne sert qu'à entraîner, on garde le générateur. d) **Faux** : un GAN ne contient que ce qu'il a appris de ses données ; un classifieur entraîné seulement sur des images générées fait nettement moins bien (Ravuri et Vinyals, 2019). e) **B** : *Adversarial* ; *Antagonistic* est le mot du livre.
**À retenir** : on garde le générateur ; ses échantillons complètent les données, ils ne les remplacent pas.

### 27.Q2 — Le faussaire, l'inspectrice et les quatre verdicts
a) **B** : un vrai déclaré faux est un faux négatif. b) **C** : un faux qui passe est un faux positif. c) **D** : selon le livre, le générateur apprend quand son faux est repéré, un vrai négatif. d) **1** : seul le vrai positif ne fait apprendre personne. e) **Vrai** : $-\ln 0{,}9 \approx 0{,}105 > 0$ ; tant que $D(\mathbf{x}) < 1$, la cross-entropy d'un vrai bien reconnu a un gradient, qui pousse $D(\mathbf{x})$ vers 1.
**À retenir** : en pratique, aucun seuil ne décide qui apprend : tous les échantillons d'un mini-batch donnent un gradient.

### 27.Q3 — Pourquoi le générateur ne voit-il jamais de vrais billets ?
a) **B** : des nombres tirés au hasard. b) **B** : ce que $D$ a appris des vrais revient au générateur par le gradient qui traverse $D$. c) **Faux** : le générateur doit être dérivable, sinon aucun gradient ne l'atteint ; un procédé chimique ne s'entraîne pas ainsi. d) **A** : les composantes du bruit sont des variables latentes. e) **Faux** : un GAN n'a pas d'encodeur ; aucune donnée réelle n'a de code.
**À retenir** : le générateur ne voit pas les données, mais il en reçoit l'information par le gradient du discriminateur.

### 27.Q4 — Le round d'apprentissage du livre en quatre temps
a) **« BCAC »** : un vrai pour $D$ (B), un faux pour $G$ (C), un faux pour $D$ (A), de nouveau un faux pour $G$ (C). b) **2**. c) **B** : $D$ est gelé, mais le gradient le traverse jusqu'à $G$. d) **Vrai** : c'est l'algorithme 1 de Goodfellow et coll., avec $k = 1$ pas de $D$ par pas de $G$. e) **B** : `detach` coupe le graphe à la sortie de $G$.
**À retenir** : le round du livre n'est pas la boucle utilisée ; 🔬 27.19 montre qu'il fait échouer l'entraînement.

### 27.Q5 — Coopératifs ou antagonistes : somme nulle, minimax et équilibre de Nash
a) **Vrai** : $\min_G \max_D V(D, G)$. b) **A** : les gains s'additionnent à zéro, quelle que soit l'issue ; la définition B est celle du livre. c) **Vrai** : chaque jeton gagné par l'un est perdu par un autre. d) **A** : personne ne gagne à changer seul de stratégie. e) **0,5** : là où $p_g = p_{\text{data}}$, $D^* = \frac{1}{2}$.
**À retenir** : un GAN est un jeu ; son équilibre de Nash est $p_g = p_{\text{data}}$, $D = \frac{1}{2}$.

### 27.Q6 — Trois modèles, des couches partagées et un discriminateur gelé
a) **3** : le générateur, le discriminateur et le modèle combiné. b) **Faux** : ce sont les mêmes couches, partagées ; rien à resynchroniser. c) **B** : deux optimiseurs, et `D(G(z))` calculé à la volée. d) **Vrai** : `backward()` remplit le `.grad` de tous les poids traversés, ceux de $D$ compris. e) **B** : seul `opt_g.step()` modifie des poids, et `opt_d.zero_grad()` efface les gradients de $D$ avant son pas.
**À retenir** : en PyTorch, « geler » un réseau pendant une étape, c'est surtout ne pas appeler son optimiseur.

### 27.Q7 — Pourquoi juger des mini-batchs plutôt qu'un point isolé
a) **313** : $\lceil 10\,000 / 32 \rceil$, 312 mini-batchs pleins et un de 16 (le générateur fait autant de pas). b) **Faux** : il reçoit **un** point (deux nombres) et rend une probabilité ; le mini-batch ne fait que moyenner les gradients. c) **A**. d) **B** : $1 - e^{-1/2} \approx 39{,}3\,\%$ ; 68 % est la règle d'une dimension. e) **114** : $(4 + 1) \times 16 + (16 + 1) \times 2 = 80 + 34$.
**À retenir** : un discriminateur point par point ne voit pas qu'un lot manque de variété ; il faut lui donner une statistique du lot.

### 27.Q8 — Les règles de Radford : vrai ou faux
a) **Vrai**. b) **Faux** : $2 \cdot 10^{-4}$ pour les deux, avec $\beta_1 = 0{,}5$. c) **Faux** : les Leaky ReLU (pente 0,2) vont dans le **discriminateur** ; le générateur a des ReLU. d) **Vrai**. e) **Vrai** : une tanh rend des valeurs dans $[-1, 1]$ ; des vrais pixels dans $[0, 1]$ trahiraient les faux.
**À retenir** : Adam $2 \cdot 10^{-4}$, $\beta_1 = 0{,}5$, Leaky ReLU dans $D$, tanh en sortie de $G$.

### 27.Q9 — Grandes images et croissance progressive
a) **3 000 000**. b) **A**. c) **B**. d) **Faux** : elles arrivent en fondu, avec un poids $\alpha$ qui passe de 0 à 1. e) **9** : 4, 8, 16, 32, 64, 128, 256, 512 et 1 024.
**À retenir** : on apprend du grossier vers le fin, et chaque nouvelle couche entre en douceur.

### 27.Q10 — Effondrement de mode total ou partiel
a) **B** : un pic de la distribution. b) **B** : partiel, il reste quelques 7. c) **Vrai** : tant que ce chiffre trompe $D$, $-\ln D(G(\mathbf{z}))$ est petit ; la loss ne voit pas le manque de variété. d) **B**. e) **« ABD »** : entraîner plus longtemps sans rien changer ne donne au discriminateur aucun moyen de voir le manque de variété.
**À retenir** : une loss basse ne dit rien de la variété ; on compte les modes ou les classes.

<a id="rappels"></a>

## 🔁 Rappels

### 27.R1 — Deux agents qui jouent l'un contre l'autre (self-play)
a) **A**. b) **Faux** : deux réseaux différents, aux rôles opposés ; en auto-apprentissage, une seule table joue les deux camps. c) **B** : le morpion parfait est nul. d) **Vrai** : dans les deux cas, l'adversaire évolue, la cible n'est pas fixe. e) **A** : un gradient, qui dit dans quelle direction modifier chaque sortie ; un agent de RL ne reçoit qu'un nombre.
**À retenir** : GAN et auto-apprentissage sont deux jeux à cible mouvante ; le GAN a l'avantage d'un signal dérivable.

### 27.R2 — Precision et recall pour juger un générateur (fidélité contre diversité)
a) **0,727** : $\frac{80}{80 + 30}$. b) **0,80** : $\frac{80}{80 + 20}$. c) **A**. d) **A** : chaque échantillon est fidèle, mais le générateur ne couvre presque rien. e) **Vrai** : un manque de diversité change la covariance et fait monter le FID, comme un manque de fidélité.
**À retenir** : precision générative = fidélité, recall génératif = diversité ; le FID mêle les deux.

### 27.R3 — Batchnorm, où la placer, train() ou eval()
a) **A**. b) **B**. c) **A**. d) **Vrai** : en `eval()`, la normalisation ne dépend plus du lot. e) **128** : $\gamma$ et $\beta$ pour chacun des 64 canaux ; la moyenne et la variance glissantes sont des *buffers*, pas des paramètres appris.
**À retenir** : on génère en `eval()` ; sinon l'image d'un $\mathbf{z}$ dépend des autres $\mathbf{z}$ du lot.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 27.1 — Dimensions d'un DCGAN pour MNIST : 6 272 neurones, tailles 7 → 14 → 28 et retour ✏️
**Réponse** : a) **6 272** ; b) **6 428 800** ; c) **14** ; d) **7** ; e) **14** ; f) **131 072** ; g) **204 864**.
**Démarche** : a) $7 \times 7 \times 128$. b) $1\,024 \times 6\,272 = 6\,422\,528$ poids et $6\,272$ biais. c) $\lfloor (28 + 2 - 4)/2 \rfloor + 1 = 13 + 1 = 14$ ; d) de même $\lfloor (14 + 2 - 4)/2 \rfloor + 1 = 7$. e) $(7 - 1) \times 2 - 2 + 4 = 14$. f) $64 \times 128 \times 16 = 131\,072$. g) $64 \times 128 \times 25 + 64 = 204\,864$. Dans ta copie : on part de 7 parce que deux doublements doivent donner 28 ($6 \to 24$, $8 \to 32$) ; la convolution de g) coûte plus cher (noyau 5 × 5, et elle travaille après l'agrandissement) ; la convolution transposée de f) est la plus exposée au damier : quand le noyau n'est pas un multiple du pas, certains pixels de sortie reçoivent plus de contributions que leurs voisins ; avec $k = 4$ et $s = 2$, chaque pixel en reçoit autant, ce qui supprime ce chevauchement inégal (un damier peut encore être appris, mais il n'est plus imposé par la géométrie).
**Erreurs fréquentes** : oublier le $+1$ ou le $2p$ de la formule ; partir de $H$ au lieu de $H - 1$ pour la convolution transposée ; compter un biais par entrée ; oublier les biais (ou en compter quand l'énoncé dit `bias=False`).
**Variante** : le discriminateur de 27.20 avec $w = 32$ : la couche dense finale a combien de paramètres ? Réponse : $2 \times 32 \times 7 \times 7 + 1 = 3\,137$.

### Ex 27.2 — Valeur minimax d'une petite matrice de gains ✏️
**Réponse** : a) **B** ; b) **0** ; c) **3** ; d) **Faux** ; e) **Faux**.
**Démarche** : minimums des lignes : A → −1, B → 0, C → −3 ; Ligne joue B et garantit 0. Maximums des colonnes : 3, 5, 4 ; Colonne joue 1 et ne laisse au plus que 3. Maximin $0 <$ minimax $3$ : pas de point selle ; en stratégies pures, chacun peut regretter son coup, et l'équilibre de Nash passe par des stratégies tirées au hasard. e) L'ordre compte : en optimisant $G$ contre le $D$ du moment, la descente alternée se rapproche d'un $\max_D \min_G$, où le générateur a intérêt à tout miser sur un seul échantillon (l'explication de Goodfellow de l'effondrement de mode). Dans ta copie : on choisit le coup qui ne laisse que 5 à l'adversaire, en supposant qu'il répond au mieux : on minimise son maximum.
**Erreurs fréquentes** : prendre le maximum des lignes (le coup le plus tentant, pas le plus prudent) ; confondre les rôles des lignes et des colonnes pour Colonne ; croire qu'il y a toujours un point selle.
**Variante** : remplace le 1 de la case (B, 1) par 3. Réponse : maximin toujours 0 (la ligne B a toujours un 0), minimax $\min(3, 5, 4) = 3$ : rien ne change.

### Ex 27.3 — Losses BCE du discriminateur et du générateur à partir de sorties données ✏️
**Réponse** : a) **0,367** ; b) **0,231** ; c) **0,598** ; d) **1,753** ; e) **−0,231** ; f) **1,00**.
**Démarche** : a) $\frac{1}{2}(0{,}2231 + 0{,}5108)$. b) $\frac{1}{2}(-\ln 0{,}7 - \ln 0{,}9) = \frac{1}{2}(0{,}3567 + 0{,}1054)$. c) La somme. d) $\frac{1}{2}(1{,}2040 + 2{,}3026)$. e) L'opposé de b). f) Les deux vrais sont au-dessus de 0,5, les deux faux en dessous : 4 sur 4. Dans ta copie : dans d), c'est le faux le plus repéré ($D = 0{,}1$) qui pèse le plus (2,30 contre 1,20) ; dans e), il pèse le **moins** ($\ln 0{,}9 = -0{,}105$ contre $\ln 0{,}7 = -0{,}357$). C'est pourtant lui qui a le plus à apprendre : c'est la saturation (∂ 27.6).
**Erreurs fréquentes** : oublier de diviser par 2 ; mettre le label 1 aux faux dans $L_D$ ; donner la loss minimax positive (elle est négative : $G$ minimise un logarithme de nombres plus petits que 1).
**Variante** : avec $D(G(\mathbf{z}_1)) = D(G(\mathbf{z}_2)) = 0{,}5$, la loss non saturante vaut $\ln 2 \approx 0{,}693$ et la quantité minimax $-0{,}693$.

### Ex 27.4 — À l'équilibre, la loss vaut ln 2 et non 0,5 ✏️
**Réponse** : a) **1,386** ; b) **0,693** ; c) **0,607** ; d) **−1,386** ; e) **Vrai**.
**Démarche** : a) $2 \times (-\ln \frac{1}{2}) = 2\ln 2$. b) $\ln 2$. c) $-\ln p = 0{,}5 \Rightarrow p = e^{-0{,}5} \approx 0{,}607$. d) $V = \ln \frac{1}{2} + \ln \frac{1}{2} = -\ln 4$, la valeur du théorème 1 de Goodfellow et coll. e) Le palier de 0,69 est exactement $\ln 2$. Dans ta copie : un discriminateur qui répond toujours $0{,}607$ aurait $L_D = -\ln 0{,}607 - \ln 0{,}393 = 0{,}5 + 0{,}934 = 1{,}434 > 1{,}386$ : **pire**. Quand vrais et faux sont indiscernables, la meilleure réponse est $\frac{1}{2}$, et la meilleure loss $2\ln 2$.
**Erreurs fréquentes** : confondre la sortie de $D$ (0,5) et sa loss ; prendre le logarithme décimal ; oublier que $L_D$ additionne deux termes.
**Variante** : avec un $\ln$ en base 2 (en bits), la loss de $D$ à l'équilibre vaudrait exactement **2** bits, et celle de $G$ **1** bit.

### Ex 27.5 — Le discriminateur optimal D* = p_data / (p_data + p_g) ∂
**Réponse** : $D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$ et $V(D^*, G) = -\ln 4 + 2\,\mathrm{JS}(p_{\text{data}} \,\|\, p_g)$, minimal ($-\ln 4$) pour $p_g = p_{\text{data}}$.
**Démarche** : 1. $\mathbb{E}_{\mathbf{z}}[\ln(1 - D(G(\mathbf{z})))] = \int p_g(\mathbf{x}) \ln(1 - D(\mathbf{x}))\,d\mathbf{x}$, car $G(\mathbf{z})$ suit $p_g$ ; d'où $V = \int [p_{\text{data}} \ln D + p_g \ln(1 - D)]\,d\mathbf{x}$. 2. $f'(y) = \frac{a}{y} - \frac{b}{1-y} = 0 \iff a(1 - y) = by \iff y = \frac{a}{a+b}$ ; $f''(y) = -\frac{a}{y^2} - \frac{b}{(1-y)^2} < 0$ : c'est un maximum. Comme $D$ est libre en chaque point, maximiser l'intégrale revient à maximiser l'intégrande partout. 3. $D^* = \frac{1}{2}$ là où $p_g = p_{\text{data}}$ ; quand $p_g(\mathbf{x}) \to 0$, $D^* \to 1$ (le générateur n'y produit presque rien, tout ce qu'on y voit est vrai). 4. $D^* = \frac{p_{\text{data}}}{2m}$ et $1 - D^* = \frac{p_g}{2m}$, donc $V = \int p_{\text{data}} \ln \frac{p_{\text{data}}}{m} + \int p_g \ln \frac{p_g}{m} + (\ln \frac{1}{2})(\int p_{\text{data}} + \int p_g) = \mathrm{KL}(p_{\text{data}} \| m) + \mathrm{KL}(p_g \| m) - 2\ln 2$. 5. La somme des KL est positive, nulle seulement si $p_{\text{data}} = m = p_g$ : le minimum, $-\ln 4$, est atteint pour $p_g = p_{\text{data}}$. 6. C'est la valeur de l'Ex 27.4 d), et $L_D = -V = 2 \ln 2$ de l'Ex 27.4 a).
**Erreurs fréquentes** : dériver par rapport à $\mathbf{x}$ au lieu de $y = D(\mathbf{x})$ ; oublier de vérifier que l'extremum est un maximum ; oublier les deux $\ln \frac{1}{2}$ en faisant apparaître $m$ (on trouve alors un minimum de 0 au lieu de $-\ln 4$).
**Variante** : $p_{\text{data}} = \mathcal{U}[0, 1]$ et $p_g = \mathcal{U}[0, 2]$. Réponse : $D^* = \frac{1}{1 + 1/2} = \frac{2}{3}$ sur $[0, 1]$ et $0$ sur $]1, 2]$.

### Ex 27.6 — Pourquoi log(1 − D(G(z))) sature et pas −log D(G(z)) ∂
**Réponse** : $\frac{d}{da}\ln(1 - \sigma(a)) = -D$ et $\frac{d}{da}[-\ln \sigma(a)] = -(1 - D)$ ; pour $D = 0{,}01$, $-0{,}01$ contre $-0{,}99$ : **99** fois plus.
**Démarche** : 1. $\sigma'(a) = \sigma(a)(1 - \sigma(a))$. 2. $\frac{-\sigma'(a)}{1 - \sigma(a)} = -\sigma(a) = -D$. 3. $-\frac{\sigma'(a)}{\sigma(a)} = -(1 - \sigma(a)) = -(1 - D)$. 4. $|{-0{,}99}| / |{-0{,}01}| = 99$. 5. Les deux dérivées sont négatives : en descendant le gradient, le générateur augmente $a$, donc $D(G(\mathbf{z}))$, dans les deux cas : même but (que $D$ prenne les faux pour des vrais) et, dans le jeu, même équilibre ; aucune des deux n'a de point stationnaire pour un $a$ fini. Même objectif (tromper $D$), mais un gradient fort là où le générateur en a besoin (ses faux sont repérés) au lieu d'un gradient fort quand il trompe déjà $D$. 6. Avec le label 1 : $-\ln \sigma(a) = \ln(1 + e^{-a})$, la loss **non saturante** ; PyTorch la calcule à partir du logit sans former $\sigma(a)$, ce qui évite $\ln 0$ quand $\sigma(a)$ s'arrondit à 0 ou 1 (et c'est plus précis).
**Erreurs fréquentes** : oublier le facteur $\sigma'(a)$ de la dérivée composée ; croire que les deux losses ont le même gradient parce qu'elles ont le même minimum ; dériver par rapport à $D$ au lieu du logit (on trouve $-\frac{1}{1-D}$ et $-\frac{1}{D}$, et la conclusion paraît inverse : c'est le facteur $\sigma'$ qui écrase tout près de 0).
**Variante** : pour $D(G(\mathbf{z})) = 0{,}9$ (le générateur trompe bien), les dérivées valent $-0{,}9$ et $-0{,}1$ : la loss minimax pousse alors 9 fois plus fort.

### Ex 27.7 — Croissance progressive : où passe le temps de calcul ? 🧮
**Réponse** : la dernière étape prend **75 %** du calcul, les quatre premières environ 0,1 % ; la croissance progressive coûte environ **6,75 fois** moins qu'un entraînement à pleine résolution sur le même nombre d'images.
**Démarche** : 1. $16, 64, 256, 1\,024, 4\,096, 16\,384, 65\,536, 262\,144, 1\,048\,576$ ; somme $= 16 \frac{4^9 - 1}{3} = 1\,398\,096$. 2. $\frac{1\,048\,576}{1\,398\,096} \approx 0{,}75$ (chaque étape coûte trois fois la somme de toutes les précédentes, plus un) ; les quatre premières : $\frac{1\,360}{1\,398\,096} \approx 0{,}1\,\%$. 3. $\frac{9 \times 1\,048\,576}{1\,398\,096} \approx 6{,}75$ ; Karras et coll. annoncent 2 à 6 fois, parce que l'hypothèse est grossière (point 4) et qu'on ne passe pas le même temps à chaque étape. 4. Les couches à basse résolution ont souvent **plus** de canaux, les couches d'entrée et de sortie et la mémoire ne suivent pas exactement les pixels, et les étapes de fondu font travailler deux branches. 5. $16 \times 3 \times 1\,024^2 \times 4 \approx 201$ Mo (192 Mio) pour la seule entrée ; chaque couche garde ses activations pour la rétropropagation, sur des dizaines ou centaines de canaux : plusieurs gigaoctets.
**Erreurs fréquentes** : compter les côtés au lieu des pixels (la suite double au lieu de quadrupler) ; oublier l'étape 4 × 4 ou 1 024 × 1 024 ; oublier les trois canaux de couleur pour la mémoire.
**Variante** : si l'on s'arrête à 256 × 256 (7 étapes), la dernière étape prend encore $\frac{65\,536}{87\,376} \approx 75\,\%$ : la proportion ne dépend presque pas du nombre d'étapes.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 27.8 — Un GAN en cinq lignes, sans faux billets 🗣️
**Réponse modèle** : (1) Un apprenti parfumeur veut recréer un parfum célèbre qu'il n'a jamais senti. (2) Un « nez », lui, connaît l'original : on lui présente tantôt le vrai, tantôt les essais de l'apprenti, et il dit lequel est lequel. (3) L'apprenti ne reçoit que ce verdict, et une indication : « plus boisé », « moins sucré » ; il corrige sa formule. (4) Le nez s'affine aussi, à force de comparer, et repère des défauts de plus en plus fins. (5) Si le nez devient trop fort d'un coup, il ne dit plus que « non » sans indication utile ; s'il est trop faible, l'apprenti s'arrête sur un parfum médiocre : il faut qu'ils progressent ensemble.
**Ce qui compte** : le producteur ne voit jamais l'original ; il apprend d'un **sens** de correction, pas seulement d'un oui ou d'un non (le gradient) ; le juge apprend aussi ; l'équilibre entre les deux. Une image où le producteur copie l'original rate l'essentiel.

### Ex 27.9 — Visages qui n'existent pas, deepfakes et obligation de transparence ⚖️
**Réponse modèle** : 1. Légitimes : des personnages de jeux vidéo ou de publicités sans négocier de droit à l'image, des données d'entraînement ou de test sans données personnelles. Malveillants : des faux profils pour l'escroquerie sentimentale ou la désinformation (un visage généré ne se retrouve pas par une recherche d'image inversée), des faux témoignages de clients. 2. L'élue (réputation), les électeurs (vote faussé), la confiance dans toutes les vidéos (le « dividende du menteur » : un vrai enregistrement peut alors être nié). Le démenti arrive trop tard parce qu'une vidéo choquante se partage en quelques heures, la veille d'un scrutin, et qu'une correction circule moins que l'original. 3. Les deux : le **fournisseur** d'un système qui génère des images, du son, de la vidéo ou du texte doit marquer ses sorties de façon détectable par une machine ; le **déployeur** qui diffuse un deepfake dans un cadre professionnel (l'usage purement personnel est exclu) doit indiquer qu'il est généré ou manipulé. Ces obligations s'appliquent depuis le 2 août 2026 (jusqu'au 2 décembre 2026 pour le marquage des systèmes déjà sur le marché). Pour une œuvre manifestement artistique, satirique ou fictive, l'obligation se limite à signaler l'existence du trucage sans gêner l'œuvre. 4. Filigrane invisible : se lit automatiquement, mais peut être affaibli par une compression, un recadrage ou une régénération. Métadonnées C2PA signées : prouvent l'origine et les modifications quand elles sont là, mais disparaissent avec une capture d'écran, et leur absence ne prouve rien. Détecteurs : ne demandent rien au générateur, mais se trompent et prennent du retard sur chaque nouveau modèle. 5. Les personnes ont-elles consenti ? Les photos sont des données personnelles (RGPD : base légale, information, droit d'opposition) ; le jeu est-il représentatif (âges, couleurs de peau), sinon le générateur reproduira ses biais ; qui pourra utiliser le modèle, et pour quoi ?
**Ce qui compte** : distinguer les acteurs (qui génère, qui publie, qui est imité, qui est trompé) ; savoir que l'obligation existe en Europe et sur qui elle pèse ; voir que les solutions techniques sont complémentaires, aucune n'étant suffisante seule.

### Ex 27.10 — Lire Goodfellow et coll. (2014), « Generative Adversarial Nets » 📄
**Réponse modèle** : 1. $\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$ ; $D$ maximise, $G$ minimise. 2. $k$ pas de $D$ (chacun sur $m$ vrais et $m$ faux) puis un pas de $G$ ; les auteurs prennent $k = 1$, le moins coûteux. Le livre fait D, G, D, G, et n'apprend que des erreurs : ni l'un ni l'autre ne sont dans l'article. 3. Au début, $G$ est mauvais, $D$ rejette ses faux avec confiance et $\log(1 - D(G(\mathbf{z})))$ sature : maximiser $\log D(G(\mathbf{z}))$ garde le même point fixe et donne des gradients bien plus forts (∂ 27.6). 4. La proposition 1 donne $D^*$ ; le théorème 1 montre que le minimum de $C(G) = \max_D V$ est $-\log 4$, atteint seulement pour $p_g = p_{\text{data}}$ (via la divergence de Jensen-Shannon) : c'est l'Ex 27.4 d). 5. Noir pointillé : $p_{\text{data}}$ ; vert : $p_g$ ; bleu pointillé : $D$. Les images successives montrent $D$ qui s'ajuste, puis $p_g$ qui se rapproche de $p_{\text{data}}$, jusqu'à $p_g = p_{\text{data}}$ et $D = \frac{1}{2}$ partout. 6. Le générateur envoie trop de $\mathbf{z}$ vers la même valeur de $\mathbf{x}$ quand on l'entraîne trop sans mettre à jour $D$ : c'est l'**effondrement de mode**. 7. L'estimation par fenêtres de Parzen a une grande variance et marche mal en grande dimension ; elle a été remplacée par le FID (Heusel et coll., 2017), puis complétée par la precision et le recall génératifs.
**Ce qui compte** : relier l'article aux exercices (∂ 27.5 et 27.6, ✏️ 27.4) ; voir que la loss non saturante, la valeur à l'équilibre et l'effondrement de mode sont déjà dans l'article fondateur, et que le round du livre n'y est pas.

<a id="entretien"></a>

## 💼 Entretien

### 27.E1 — Expliquer l'entraînement d'un GAN et sa loss 💼
**Réponse modèle en 60 secondes** : « Un GAN, ce sont deux réseaux. Le générateur transforme un bruit aléatoire en un échantillon ; le discriminateur est un classifieur binaire qui dit si un échantillon est vrai ou généré. On alterne deux étapes sur chaque mini-batch. D'abord le discriminateur : cross-entropy binaire avec le label 1 pour les vrais et 0 pour les faux, les faux détachés du générateur. Ensuite le générateur : on repasse ses faux dans le discriminateur avec le label 1, et seul l'optimiseur du générateur fait un pas. Cette loss, $-\ln D(G(z))$, est la version non saturante : la version minimax d'origine, $\ln(1 - D(G(z)))$, a un gradient presque nul au début, quand les faux sont faciles à repérer. À l'équilibre, le discriminateur répond 0,5 partout et chaque terme vaut ln 2. »
**Ce qui compte** : les deux étapes dans l'ordre, `detach`, le label 1 du générateur, la raison de la loss non saturante, et le fait que la valeur des losses ne mesure pas la qualité.
**Relances possibles** :
- « Pourquoi deux optimiseurs ? » → Pour que chaque étape ne modifie qu'un réseau : à l'étape du générateur, le gradient traverse le discriminateur et dépose des `.grad` dans ses poids, mais on n'appelle que l'optimiseur du générateur.
- « Que valent les losses à l'équilibre ? » → $2\ln 2 \approx 1{,}39$ pour le discriminateur (vrais + faux), $\ln 2 \approx 0{,}69$ pour le générateur ; mais des courbes à l'équilibre peuvent cacher de mauvais échantillons si le discriminateur est trop faible.
- « Combien de pas de discriminateur par pas de générateur ? » → Un dans le DCGAN et l'article d'origine ; cinq dans le WGAN, dont le critique doit approcher une distance.

### 27.E2 — Qu'est-ce que l'effondrement de mode et comment le combattre ? 💼
**Réponse modèle en 60 secondes** : « Le générateur ne produit plus qu'une partie de la variété des données : un seul visage, ou presque que des 1 sur MNIST. Il a trouvé des sorties qui trompent le discriminateur du moment ; comme il s'optimise contre ce discriminateur précis, il a intérêt à tout miser dessus. Le discriminateur finit par rejeter ce mode, le générateur saute vers un autre, et ça peut tourner en rond. La loss ne le voit pas, donc je le mesure : histogramme des classes prédites par un classifieur, modes couverts sur un jeu jouet, recall génératif ou FID. Remèdes : donner au discriminateur une statistique du mini-batch (couche d'écart-type par mini-batch, discrimination par mini-batch), une loss de Wasserstein avec pénalité de gradient, équilibrer les learning rates. Sur un anneau de huit gaussiennes, j'ai vu un générateur tomber à trois modes et remonter à huit avec la couche d'écart-type. »
**Ce qui compte** : définition (total, partiel), cause (optimisation contre un discriminateur fixe), détection chiffrée, remèdes concrets, et une expérience vécue.
**Relances possibles** :
- « Pourquoi un discriminateur classique ne le voit-il pas ? » → Il juge chaque échantillon seul : un lot de cent chiffres identiques mais réalistes lui paraît cent fois vrai. Il faut lui montrer une statistique du lot.
- « Les modèles de diffusion ont-ils ce problème ? » → Beaucoup moins : leur loss de débruitage couvre toute la distribution des données, c'est une des raisons de leur succès ; leur faiblesse est plutôt la lenteur de génération.

### 27.E3 — GAN, VAE ou diffusion : lequel choisir et pourquoi ? 💼
**Réponse modèle en 60 secondes** : « Pour générer des images de qualité aujourd'hui, je pars d'un modèle de diffusion : meilleure diversité que les GAN, entraînement stable, conditionnement par le texte naturel ; depuis 2021, ils battent les GAN sur ImageNet. Je prends un GAN quand la vitesse compte, parce qu'il génère en un seul passage : super-résolution, temps réel, ou comme loss adversariale d'appoint, et pour distiller un modèle de diffusion en un à quatre pas. Je prends un VAE quand j'ai besoin d'un encodeur et d'un espace latent régulier : compression, détection d'anomalies, et justement l'espace latent dans lequel travaillent les modèles de diffusion latente. »
**Ce qui compte** : des critères (qualité, diversité, vitesse, stabilité, besoin d'un encodeur) plutôt qu'un classement absolu ; savoir que les trois se combinent dans les systèmes actuels.
**Relances possibles** :
- « Pourquoi les images d'un VAE sont-elles floues ? » → Sa loss de reconstruction (souvent une erreur quadratique) moyenne les possibles : face à une incertitude, la moyenne de plusieurs images nettes est floue. La loss adversariale d'un GAN pénalise justement le flou.
- « Qu'est-ce qui rend la diffusion lente, et comment l'accélère-t-on ? » → Des dizaines d'étapes de débruitage, une passe de réseau chacune ; on réduit le nombre de pas (meilleurs solveurs, distillation, dont la distillation adversariale).

### 27.E4 — Comment évaluer un modèle génératif (FID et ses limites) ? 💼
**Réponse modèle en 60 secondes** : « Je regarde toujours des grilles d'échantillons pour un bruit fixé, mais je mesure aussi. Le FID passe des vraies images et des images générées dans un réseau entraîné, Inception v3, ajuste une gaussienne sur chaque nuage de features et calcule leur distance de Fréchet : la distance des moyennes plus un terme de trace qui compare les covariances. Plus il est bas, mieux c'est. Ses limites : il dépend du nombre d'échantillons, il faut toujours comparer au même nombre, souvent 50 000 ; il dépend de l'extracteur, entraîné sur ImageNet ; et il mêle fidélité et diversité. Je le complète par une precision et un recall génératifs, et pour des images loin d'ImageNet, par des features plus générales, comme celles de DINOv2. »
**Ce qui compte** : savoir ce que compare la formule ; les pièges de comparaison (nombre d'échantillons, extracteur) ; une mesure complémentaire.
**Relances possibles** :
- « Pourquoi pas une distance entre pixels ? » → Deux images très proches pour un humain peuvent être loin en pixels (un décalage d'un pixel) ; les features d'un réseau entraîné sont plus proches de la perception.
- « Un FID de 0 est-il possible ? » → Seulement à l'aléa d'échantillonnage près pour deux échantillons de la même distribution ; un générateur qui recopie ses données d'entraînement aurait un FID excellent, d'où des contrôles de mémorisation à part.

### 27.E5 — Pourquoi l'entraînement des GAN est-il instable ? 💼
**Réponse modèle en 60 secondes** : « Parce qu'on ne minimise pas une fonction : on cherche l'équilibre d'un jeu à deux joueurs. Chaque pas du générateur change le problème du discriminateur, et inversement ; la descente de gradient alternée peut tourner autour de l'équilibre sans y arriver, et rien ne garantit la convergence pour de vrais réseaux. S'y ajoutent la saturation, quand le discriminateur gagne trop facilement, et l'effondrement de mode. Pour stabiliser : loss non saturante, Adam avec un β1 de 0,5, des architectures éprouvées comme le DCGAN, une régularisation du discriminateur, normalisation spectrale, pénalité de gradient R1 ou WGAN-GP, des learning rates différents par réseau (TTUR), et je suis le FID plutôt que les losses. »
**Ce qui compte** : la cause (jeu, cible mouvante), les symptômes (oscillations, saturation, effondrement), et des remèdes précis, pas une liste vague.
**Relances possibles** :
- « Un discriminateur trop fort, c'est toujours mauvais ? » → Pas forcément : avec la loss non saturante et Adam, un discriminateur plus rapide que le générateur peut même stabiliser (dans le notebook, l'écart-type généré oscillait moins) ; le danger vient de la saturation, avec la loss minimax ou un optimiseur sans normalisation du gradient.
- « Qu'apporte la pénalité R1 ? » → Elle pénalise la norme du gradient du discriminateur sur les vraies données : le discriminateur ne peut plus devenir trop raide autour d'elles, et l'entraînement converge localement (Mescheder et coll., 2018).

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes sont dans `05_solutions.ipynb` (exécuté) ; voici les démarches et les résultats du mode rapide (CPU, 2 cœurs). Les chiffres d'un entraînement peuvent bouger d'une machine à l'autre ; les vérifications du notebook ont des marges.

### Ex 27.11 — Distance de Wasserstein 1D entre deux échantillons 🔨
**Réponse** : la fonction de référence (dans `solutions/mylearn_ref/gen_metrics.py`) :
```python
def wasserstein_1d(u, v):
    u_sorted = np.sort(_sample_1d(u, "u"))
    v_sorted = np.sort(_sample_1d(v, "v"))
    if u_sorted.size == v_sorted.size:
        return float(np.mean(np.abs(u_sorted - v_sorted)))
    merged = np.sort(np.concatenate([u_sorted, v_sorted]))
    widths = np.diff(merged)
    cdf_u = np.searchsorted(u_sorted, merged[:-1], side="right") / u_sorted.size
    cdf_v = np.searchsorted(v_sorted, merged[:-1], side="right") / v_sorted.size
    return float(np.sum(np.abs(cdf_u - cdf_v) * widths))
```
où `_sample_1d` convertit en tableau `float64` et refuse un échantillon vide, non 1-D ou non fini. Les quatre échantillons comparés aux données : même loi **0,043**, décalée de 1 **1,012**, deux fois plus étalée **0,969**, effondrée sur 4 **1,001**.
**Démarche** : en 1D, le transport optimal envoie le $k$-ième plus petit point sur le $k$-ième plus petit : d'où le tri. Pour des tailles différentes, les fonctions de répartition sont des escaliers, constants entre deux valeurs de l'échantillon fusionné : l'aire entre eux est une somme de rectangles. Un décalage de 1 coûte exactement 1 par point. Le générateur effondré sur 4 a la bonne moyenne, mais il faut déplacer chaque point de sa distance à 4 : $\mathbb{E}|X - 4| = 1{,}25\sqrt{2/\pi} \approx 1{,}0$. La moyenne seule ne dit rien de la forme ; $W_1$, si.
**Erreurs fréquentes** : trier sur place (`u.sort()`) et modifier les données de l'appelant ; `side="left"` dans `searchsorted` (les fonctions de répartition sont continues à droite) ; diviser les deux répartitions par la même taille ; rendre un `np.float64` au lieu d'un `float`.
**Variante** : $W_1$ entre $\mathcal{N}(0, 1)$ et un générateur effondré sur 0 vaut $\mathbb{E}|X| = \sqrt{2/\pi} \approx 0{,}80$ (en moyenne, pour de grands échantillons).

### Ex 27.12 — Un GAN minuscule pour une gaussienne 1D 📦
**Réponse** :
```python
def gan_step_12(G, D, opt_g, opt_d, real, z):
    fake = G(z)
    logits_real, logits_fake = D(real), D(fake.detach())
    loss_d = (F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
              + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake)))
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()
    logits = D(fake)
    loss_g = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    return {"loss_d": loss_d.item(), "loss_g": loss_g.item(),
            "d_real": torch.sigmoid(logits_real).mean().item(), "d_fake": torch.sigmoid(logits_fake).mean().item()}
```
`train_gan_12` crée les deux Adam, le `torch.Generator`, et enchaîne `randperm`, les mini-batchs, `sample_noise` et `step` en remplissant l'historique (`history.setdefault(key, []).append(value)`). Après 10 epochs (3 s) : moyenne générée **4,03**, écart-type **0,98** (cible 4 et 1,25), $W_1 \approx$ **0,23** ; `loss_d` finit vers $2\ln 2$, `loss_g` vers $\ln 2$, `d_real` et `d_fake` vers 0,5.
**Démarche** : `fake.detach()` empêche la loss de $D$ de remonter dans $G$ ; à l'étape de $G$, on veut au contraire ce chemin, d'où `D(fake)`. `loss_g.backward()` remplit bien les `.grad` de $D$, mais seul `opt_g.step()` est appelé, et `opt_d.zero_grad()` les efface à l'itération suivante. Réutiliser `fake` (calculé avant le pas de $D$) est permis : ses valeurs ne dépendent que de $G$, qui n'a pas bougé. L'écart-type reste un peu bas : à l'équilibre, $D$ répond $\frac{1}{2}$ et ne corrige plus finement la forme (🔮 27.13).
**Erreurs fréquentes** : un seul optimiseur sur les deux réseaux ; `opt_d.step()` à l'étape de $G$ ; `zero_grad` après `backward` ; `d_real` calculé après le pas de $D$ ; une sigmoïde en sortie de $D$ avec `BCEWithLogitsLoss` ; créer le `torch.Generator` dans la boucle (même mini-batch à chaque epoch).
**Variante** : avec des mini-batchs de 256 au lieu de 64, il y a 4 fois moins de pas par epoch : après 10 epochs, la moyenne générée est encore loin de 4 si l'on garde le même learning rate (il faut plus d'epochs, ou un learning rate plus grand).

### Ex 27.13 — Prédire la moyenne et l'écart-type générés au fil des epochs 🔮
**Réponse** : a) **C** ; b) **Faux** ; c) **C**.

| graine | moyenne après l'epoch 1 | écart-type, epochs 2 à 10 | loss de D, epoch 10 |
|---|---|---|---|
| 0 | 5,61 | 0,86 à 1,62 | 1,387 |
| 1 | 5,42 | 0,91 à 1,75 | 1,386 |
| 2 | 5,60 | 0,85 à 1,68 | 1,386 |
| 3 | 4,97 | 0,85 à 1,66 | 1,386 |
| 4 | 5,64 | 0,88 à 2,05 | 1,387 |

**Démarche** : a) Pendant la première epoch, tous les faux sont près de 0 : le discriminateur apprend « petit = faux », sans rien savoir de ce qui se passe au-delà de 4. Le générateur, poussé vers « plus grand », dépasse la cible avant que $D$ n'apprenne à rejeter les grandes valeurs ; puis il revient. C'est le dépassement que le livre décrit pour son nuage 2D. b) Une fois $D$ à $\frac{1}{2}$ partout, plus rien ne fixe l'écart-type : il oscille entre 0,85 et 2 selon les graines. c) $2\ln 2 \approx 1{,}386$ dès l'epoch 2, alors que l'écart-type bouge encore : la loss ne mesure pas la qualité.
**Erreurs fréquentes** : imaginer une convergence monotone, comme une descente de gradient sur une loss fixe ; répondre 0,5 pour la loss (la sortie de $D$) ; répondre 0,69 (un seul terme).
**Variante** : avec un learning rate dix fois plus petit pour les deux réseaux, la moyenne n'atteint pas 4 en 10 epochs (l'entraînement A de 27.14 : 4,50 après 10 epochs, pour une autre graine), et le dépassement arrive plus tard.

### Ex 27.14 — Lire les courbes d'un GAN : losses, D(x) et D(G(z)) 📈
**Réponse** : a) **D** ; b) **1,39** ; c) **0,3** ; d) **Faux**.

| entraînement | learning rates (G, D) | moyenne | écart-type | $W_1$ |
|---|---|---|---|---|
| A | $10^{-4}$, $10^{-4}$ (lent) | 4,50 | 0,95 | 0,535 |
| B | $10^{-2}$, $10^{-4}$ (G rapide, D lent) | 5,72 | 0,33 | 1,760 |
| C | $10^{-3}$, $10^{-3}$ (standard) | 3,94 | 0,73 | 0,426 |
| D | $10^{-4}$, $10^{-2}$ (D domine) | 2,21 | 0,90 | 1,755 |

**Démarche** : a) Dans D, `d_real` reste vers 0,73 et `d_fake` vers 0,27, la loss de $D$ sous 1, celle de $G$ vers 1,6 : $D$ gagne, et le générateur avance lentement (sa moyenne n'est qu'à 2,2). b) Le palier de C : $2\ln 2$. c) Lu sur la courbe `d_fake` de D. d) B a des courbes parfaites ($2\ln 2$, $\ln 2$, $\frac{1}{2}$, $\frac{1}{2}$) et les pires échantillons : un discriminateur trop lent ne voit pas les défauts des faux (moyenne 5,7, écart-type 0,33), et « je ne sais pas » ne veut pas dire « c'est parfait ». La loss de $G$ la plus basse n'est pas non plus un critère : B et C finissent presque à égalité (0,700 et 0,697).
**Erreurs fréquentes** : croire que l'équilibre des courbes garantit la qualité ; lire la loss de $G$ comme une mesure de qualité ; confondre « D domine » (courbes écartées) et « l'entraînement est lent » (courbes qui convergent tard).
**Variante** : relance B avec 50 epochs. Le discriminateur finit-il par voir le défaut ? Réponse attendue : il progresse si lentement que le générateur reste mal placé ; il faut rééquilibrer les learning rates.

### Ex 27.15 — Reproduire l'apprentissage du nuage centré en (5, 5) (fig. 27.20-27.21) 🎨
**Réponse** : `train_book_15` garde les nuages par `on_epoch` ; `plot_clouds_15` dessine 12 panneaux et rend leurs moyennes et écarts-types ; `plot_losses_15` trace les deux losses moyennes par epoch (code complet dans le notebook). 13 epochs en 8 s.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| centre généré | (−0,1 ; 0,4) | (7,0 ; 7,0) | (3,7 ; 5,6) | (5,5 ; 4,8) | (4,2 ; 4,9) | (5,0 ; 5,6) | (4,7 ; 4,9) | (5,0 ; 5,1) | (5,2 ; 5,1) | (4,9 ; 5,0) | (4,9 ; 5,1) |

**Démarche** : on retrouve le scénario du livre, en plus rapide : une traînée près de l'origine (epoch 0, écarts-types 0,06 et 0,11), un fort dépassement dès l'epoch 1, puis des oscillations autour de $(5, 5)$ qui s'amortissent ; les écarts-types finissent vers 1 (1,15 et 0,94 à l'epoch 13). Les losses moyennes : celle de $D$ part de 1,0 à l'epoch 1 et se stabilise vers 1,38-1,39 dès l'epoch 5, soit $2\ln 2$ ; celle de $G$ vers $\ln 2$. Le « 0,5 idéal » du livre n'est pas une valeur de loss.
**Erreurs fréquentes** : un bruit normal au lieu d'uniforme (le générateur du livre n'a pas de biais à décaler) ; des axes différents d'un panneau à l'autre, qui cachent le mouvement ; tracer les losses pas par pas, illisibles ; oublier l'epoch 0.
**Variante** : avec des mini-batchs de 128, il y a 4 fois moins de pas par epoch : le dépassement arrive plus tard (vers l'epoch 3 ou 4), plus près de ce que montre le livre.

### Ex 27.16 — Le GAN qui s'entraînait à l'envers 🐛
**Réponse** : **« FHM »**.
```python
            opt_d.zero_grad()                                       # (F) reset D's gradients too
            opt_g.zero_grad()
            fake = G(z)
            logits_real, logits_fake = D(real), D(fake.detach())    # (H) the loss of D must not reach G
            ...
            loss_g = bce(logits, torch.ones_like(logits))           # (M) label 1 for G
```
**Démarche** : (H) sans `detach`, `loss_d.backward()` remonte dans $G$ et y dépose le gradient qui **fait repérer** ses faux (20 passes arrière au lieu de 10) ; comme (F) ne remet à zéro que $G$ en début d'itération, ce gradient s'ajoute à celui de `loss_g`. (F) les gradients de $D$ ne sont jamais remis à zéro : chaque pas de $D$ utilise la somme de tous les gradients passés, ceux de l'étape de $G$ compris. (M) avec le label 0, le générateur minimise $-\ln(1 - D(G(\mathbf{z})))$ : il apprend à être repéré et fuit les données. Cette erreur seule suffit à tout casser ; les deux autres sont plus sournoises, parce qu'un GAN peut sembler apprendre malgré elles. Corrigée, la boucle donne exactement le générateur de `train_gan_12` (mêmes tirages, même graine), et la moyenne générée atteint 4,02 en 3 epochs.
**Erreurs fréquentes** : citer (L) (recalculer `G(z)` est correct, juste plus coûteux) ; corriger (H) en détachant aussi à l'étape de $G$ (plus aucun gradient pour $G$) ; ajouter `opt_d.zero_grad()` après `loss_d.backward()`.
**Variante** : supprime seulement la correction de (F) (garde (H) et (M) corrigées). Le diagnostic 2 échoue ; l'entraînement apprend quand même, plus lentement et plus bruyamment : Adam normalise une somme de gradients qui grandit.

### Ex 27.17 — Prédire l'effet d'un discriminateur trop fort 🔮
**Réponse** : a) **B** ; b) **Vrai** ; c) **Vrai**.

| epoch | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| 27.12 : moyenne / écart-type | 5,61 / 1,61 | 3,99 / 0,97 | 4,01 / 1,31 | 4,00 / 1,44 | 3,83 / 1,47 | 4,03 / 0,98 |
| D fort : moyenne / écart-type | 3,65 / 1,01 | 4,00 / 1,22 | 4,00 / 1,42 | 4,00 / 1,29 | 3,91 / 1,23 | 4,04 / 1,21 |

**Démarche** : a) $D(G(\mathbf{z}))$ vaut 0,28 en moyenne à l'epoch 1 (0,21 à 0,28 sur huit graines) : $D$ repère mieux les faux, sans les rejeter tous. b) Avec la loss non saturante, un faux repéré donne au générateur un gradient fort (∂ 27.6), et Adam en normalise l'échelle : le générateur ne s'arrête pas. c) Un discriminateur qui suit de près le générateur reste proche de son optimum $D^*$, et corrige vite tout écart de forme : l'écart-type varie de 0,11 entre les epochs 6 et 10, contre 0,49 dans 27.12 (huit graines sur huit). C'est l'idée du WGAN (plusieurs pas du critique par pas du générateur) et de TTUR. L'avertissement du livre vaut pour la loss minimax et un optimiseur qui ne normalise pas le gradient (🔬 27.18).
**Erreurs fréquentes** : prédire que $D(G(\mathbf{z}))$ tombe à 0 ; appliquer sans recul la règle « si l'un progresse trop vite, l'autre ne rattrape jamais » ; confondre un $D$ qui apprend vite et un $G$ qui apprend lentement (l'entraînement D de 27.14 combine les deux).
**Variante** : garde $10^{-2}$ pour $D$ et passe $G$ à $10^{-4}$ : c'est l'entraînement D de 27.14, où le générateur n'atteint que 2,2 en 10 epochs.

### Ex 27.18 — Loss saturante ou non saturante : la dynamique du début 🔬
**Réponse** : `saturating_step_18` change une ligne (`loss_g = -F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))`) ; `generator_grad_norm_18` utilise `torch.autograd.grad`, qui ne remplit aucun `.grad`.

| graine | $D(G(\mathbf{z}))$ | norme saturante | norme non saturante | rapport |
|---|---|---|---|---|
| 0 | 0,0066 | 0,049 | 7,07 | 144 |
| 1 | 0,0088 | 0,053 | 5,74 | 108 |
| 2 | 0,0112 | 0,059 | 4,87 | 83 |

Moyenne générée après 50, 250 et 600 pas (graine 0) : SGD saturante −0,16, −0,03, **4,67** ; SGD non saturante 2,41, 7,50, 3,56 ; Adam saturante 0,99, 5,04, 3,96 ; Adam non saturante 0,56, 4,80, 3,67.
**Démarche** : le rapport des normes est de l'ordre de $\frac{1 - D}{D} \approx 100$, comme le prédit ∂ 27.6 (pas exactement, car chaque faux a son propre $D$). Avec **SGD**, le pas est proportionnel au gradient : la loss saturante perd 500 pas sur trois graines (la graine 1 n'atteint même que 0,82 en 600 pas). Avec **Adam**, chaque paramètre est divisé par la racine de la moyenne de ses gradients au carré : l'échelle disparaît, et la loss saturante avance aussi vite, voire plus. On garde quand même la loss non saturante : elle ne dépend pas de l'optimiseur, et quand le gradient devient vraiment minuscule (proche de la précision des flottants), Adam ne peut plus rien.
**Erreurs fréquentes** : appeler `loss.backward()` pour la norme et laisser des `.grad` (ils s'ajouteraient au pas suivant) ; prendre la norme du gradient d'une seule couche ; oublier le signe moins de la loss minimax (le générateur fuirait les données).
**Variante** : avec SGD et un learning rate de 0,05 pour $G$, la loss saturante finit par démarrer, mais d'un coup, avec des moyennes de plusieurs milliers sur une graine : un grand learning rate n'est pas un remède.

### Ex 27.19 — Le round en quatre temps du livre contre la boucle standard 🔬
**Réponse** : `book_round_19` (code dans le notebook) utilise une fonction interne `learn(logits, target, selected, optimizer)` ; sur les deux discriminateurs extrêmes, il retient (0, 0, 32, 0) puis (32, 32, 0, 32) échantillons. Après 10 epochs : boucle standard $W_1 = $ **0,23** ; round du livre (erreurs seules) **7,03** ; même round sur tous les échantillons **1,11**.
**Démarche** : le round « erreurs seules » échoue : le générateur se fige en un point loin des données (écart-type nul). Dès que $D$ dit « vrai » à un faux, ce faux ne donne plus rien au générateur, et le discriminateur ne corrige que la frontière ; le générateur est poussé vers la frontière, puis plus rien ne le fait bouger. Le même ordre D, G, D, G sur tous les échantillons apprend (plus lentement que la boucle standard, à cause de pas séparés pour les vrais et les faux) : c'est la règle « n'apprendre que de ses erreurs » qui casse tout, pas l'ordre des étapes. Un échantillon bien classé porte une information : $-\ln 0{,}6$ dit encore « pas assez sûr », et son gradient en tient compte.
**Erreurs fréquentes** : faire un pas d'optimiseur même quand aucun échantillon n'est retenu (Adam bougerait avec ses moments) ; diviser par le nombre d'échantillons retenus au lieu de la taille du mini-batch (le pas serait aussi grand pour une erreur que pour trente-deux) ; décider des masques avec un $D$ figé en début de round.
**Variante** : divise la somme par le nombre d'échantillons retenus. Réponse : l'échec est le même, ou pire, sur le nuage 2D : le défaut est la sélection, pas la normalisation.

### Ex 27.20 — Un DCGAN sur MNIST selon les règles de Radford 📦
**Réponse** :
```python
def make_generator_20(nz=NZ, width=32):
    return nn.Sequential(
        nn.Linear(nz, 2 * width * 7 * 7, bias=False), nn.BatchNorm1d(2 * width * 7 * 7), nn.ReLU(),
        nn.Unflatten(1, (2 * width, 7, 7)),
        nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
        nn.ConvTranspose2d(width, 1, 4, stride=2, padding=1), nn.Tanh())

def make_discriminator_20(width=32, in_channels=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        nn.Conv2d(width, 2 * width, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(2 * width), nn.LeakyReLU(0.2),
        nn.Flatten(), nn.Linear(2 * width * 7 * 7, 1))
```
`init_weights_20` : `isinstance`, puis `nn.init.normal_(module.weight, 0.0, 0.02)` ou `(1.0, 0.02)` et `nn.init.zeros_(module.bias)` pour les batchnorms. 3 epochs, **8 s par epoch** ; losses moyennes de $D$ 0,12, 0,22, 0,44 et de $G$ 4,26, 3,48, 2,37 ; classes des 1 000 images générées : 0:137, 1:47, 2:137, 3:161, 4:62, 5:24, 6:26, 7:44, 8:213, 9:149, confiance moyenne 69 %.
**Démarche** : la couche dense projette le bruit sur 64 canaux de 7 × 7, les convolutions transposées doublent deux fois. Après une epoch, des taches ; après trois, des chiffres reconnaissables mais imparfaits, avec trop de 8 et de 3 et trop peu de 5, 6 et 7 : un début d'effondrement partiel. Les losses sont loin de l'équilibre : $D$ domine encore (le générateur a tout à apprendre). `G.eval()` change les images : en `train()`, les batchnorms normalisent avec les statistiques du lot, et l'image d'un $\mathbf{z}$ dépend des autres $\mathbf{z}$.
**Erreurs fréquentes** : une batchnorm après la première convolution de $D$ ou avant la tanh ; `nn.LeakyReLU()` sans la pente (0,01 par défaut) ; une sigmoïde en sortie de $D$ ; des images dans $[0, 1]$ ; `G.apply(init_weights_20())` (on passe la fonction, sans l'appeler).
**Variante** : avec `width=64`, le générateur passe à 950 657 paramètres et chaque epoch dure environ trois fois plus ; les chiffres sont plus nets à epochs égales, pas à temps égal.

### Ex 27.21 — Règles empiriques à l'épreuve : une architecture de 2016 contre celle de Radford 🔬
**Réponse** : `make_gildenblat_21` (code dans le notebook). Le modèle du livre a **851 138** paramètres (contre 276 898) et prend **14,6 s** par epoch (contre 8,0) ; après 3 epochs, ses classes sont mieux réparties (0:21, 1:101, …, 9:95) et sa distance de Fréchet est plus basse (91 contre 119, 27.23).
**Démarche** : à epochs égales, le modèle du livre fait mieux, mais il coûte 1,8 fois plus par epoch et a 3 fois plus de paramètres : ses couches denses cachées et ses convolutions 5 × 5 à pleine résolution coûtent cher. À **temps** égal, le DCGAN de Radford aurait fait environ 5 epochs ; la comparaison honnête se fait donc à budget égal, et sur plusieurs graines. Ici, contrairement au livre, les deux modèles sont entraînés de la même façon (Adam $2 \cdot 10^{-4}$, $\beta_1 = 0{,}5$) : on compare des architectures, pas une architecture réglée contre une autre appliquée à la hâte. Les règles de Radford sont un bon point de départ, pas une loi.
**Erreurs fréquentes** : conclure sur une seule graine et un seul budget ; oublier `nn.BatchNorm1d` après la seconde couche dense ; un padding de 1 avec un noyau 5 (les tailles ne tombent plus juste).
**Variante** : entraîne le DCGAN de Radford 5 epochs (environ le temps de 3 epochs du modèle du livre) et compare les distances de Fréchet. Réponse attendue : l'écart se réduit ou s'inverse ; mesure-le.

### Ex 27.22 — Distance de Fréchet entre deux gaussiennes (la formule du FID) 🔨
**Réponse** : la référence, après validation des formes :
```python
    diff = mu1 - mu2
    eigenvalues = np.linalg.eigvals(sigma1 @ sigma2)
    trace_sqrt = np.sum(np.sqrt(np.clip(eigenvalues.real, 0.0, None)))
    d2 = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * trace_sqrt
    return float(max(d2, 0.0))
```
et `frechet_distance_from_features` ajuste `arr.mean(axis=0)` et `np.cov(arr, rowvar=False).reshape(d, d)` sur chaque ensemble avant d'appeler `frechet_distance`.
**Démarche** : $\boldsymbol{\Sigma}_1\boldsymbol{\Sigma}_2$ a les mêmes valeurs propres que $\boldsymbol{\Sigma}_1^{1/2}\boldsymbol{\Sigma}_2\boldsymbol{\Sigma}_1^{1/2}$, symétrique et positive : en théorie, elles sont réelles et positives. En pratique, le produit n'est pas symétrique et `eigvals` rend de petites parties imaginaires ou de petites valeurs négatives (covariances singulières, plus de features que d'échantillons) : on garde la partie réelle et on coupe à 0. `scipy.linalg.sqrtm` calcule toute la matrice racine, plus cher et parfois instable (avertissements sur des matrices singulières) ; on n'a besoin que de sa trace. Le reshape traite le cas d'une seule feature, où `np.cov` rend un nombre.
**Erreurs fréquentes** : la racine de la trace au lieu de la trace de la racine ; la racine de la diagonale du produit ; oublier le facteur 2 ; `np.cov` sans `rowvar=False` (covariance entre échantillons) ; laisser NumPy diffuser une moyenne de longueur 1.
**Variante** : deux gaussiennes de même covariance, moyennes $\mathbf{0}$ et $(1, 1, 1)$ : $d_F^2 = 3$, quel que soit $\boldsymbol{\Sigma}$ (le terme de trace s'annule).

### Ex 27.23 — Évaluer un générateur : distance de Fréchet sur les features d'un classifieur 📦
**Réponse** : `fid_23` enchaîne `sample_images`, `features` et `frechet_distance_from_features`.

| ensemble mesuré | distance de Fréchet |
|---|---|
| 2 000 vraies images d'entraînement | 3,4 |
| un réseau non entraîné (du bruit) | 745 |
| DCGAN de 27.20 après 1 epoch | 638 |
| DCGAN de 27.20 après 3 epochs | 119 |
| modèle du livre (27.21) après 3 epochs | 91 |

Le DCGAN final mesuré avec 200, 500, 1 000 et 2 000 images : 123,2, 121,8, 120,8, 119,3.
**Démarche** : le classement suit le jugement visuel. Les vraies images ne sont pas à 0 : deux échantillons finis de la même loi diffèrent un peu, et le jeu de test n'est pas tiré comme le jeu d'entraînement. La distance **baisse** quand on génère plus d'images : l'estimation de la covariance sur peu d'images est bruitée, et ce bruit gonfle la trace (biais vers le haut). On ne compare donc deux FID qu'au même nombre d'images, avec le même extracteur et la même référence. Ces valeurs ne se comparent pas aux FID publiés : autres features (64 valeurs d'un petit CNN MNIST contre 2 048 d'Inception), autres images, autres tailles d'échantillon.
**Erreurs fréquentes** : générer en mode `train()` ; comparer des distances mesurées sur des nombres d'images différents ; prendre les logits du classifieur au lieu des features ; inverser les rôles (le résultat est symétrique, mais la référence doit rester la même).
**Variante** : mesure 2 000 images de test contre les 2 000 autres images de test. Réponse : environ 1,7, plus bas que les images d'entraînement (3,4) : les deux moitiés du test viennent du même tirage.

### Ex 27.24 — Provoquer puis soigner un effondrement de mode (anneau de 8 gaussiennes) 🔬
**Réponse** : `ring_24` et `modes_covered_24` (code dans le notebook ; la matrice des distances par diffusion NumPy, puis `np.bincount`).

| entraînement (2 000 pas) | modes couverts tous les 500 pas | bons points à la fin |
|---|---|---|
| équilibré, $10^{-3}$ et $10^{-3}$ | 6, 8, 8, 8 | 69 % |
| D 30 fois plus rapide, $10^{-4}$ et $3 \cdot 10^{-3}$ | 4, 4, 3, 3 | 66 % |
| le même + écart-type par mini-batch | 0, 8, 8, 8 | 38 % |

**Démarche** : avec un générateur lent face à un discriminateur rapide, le générateur se réfugie sur 3 modes, qu'il dessine bien (66 % de bons points) : un effondrement partiel, propre et précis. La couche d'écart-type par mini-batch donne à $D$ l'écart-type moyen du lot : un lot concentré sur trois modes a un écart-type anormal, que $D$ apprend à repérer ; le générateur doit alors s'étaler sur les huit modes. Le prix : des points moins précis à 2 000 pas (38 % de bons) ; la diversité d'abord, la précision ensuite. Sur d'autres graines, le réglage déséquilibré couvre 2 à 7 modes (une graine sur quatre s'en sort) ; la couche d'écart-type en couvre 7 à 8.
**Erreurs fréquentes** : compter un mode dès qu'un point en est proche (un point égaré n'est pas un mode couvert) ; oublier les points loin de tout centre dans la part de bons points ; tirer le bruit avant les composantes (l'ordre de tirage de la docstring) ; conclure sur une seule graine.
**Variante** : avec `min_share=0.05`, l'entraînement équilibré couvre toujours 8 modes à la fin (chaque mode reçoit environ 1/8 des bons points).

### Ex 27.25 — Un GAN conditionnel : choisir le chiffre généré 📦
**Réponse** : les deux classes et `cgan_step_25` (code dans le notebook) ; après 3 epochs (28 s), le classifieur reconnaît **65 %** des chiffres demandés (10 % au hasard) : 0 94 %, 1 34 %, 2 78 %, 3 79 %, 4 74 %, 5 37 %, 6 2 %, 7 90 %, 8 76 %, 9 83 %. Variété à l'intérieur d'un chiffre : 0,13 à 0,26 pour les images générées, 0,14 à 0,32 pour les vraies.
**Démarche** : le label commande déjà la plupart des chiffres ; les 6 ne sont pas encore appris (souvent dessinés comme des 0 ou des 5), comme dans le DCGAN non conditionnel, qui en produisait peu. La variété est un peu plus faible que celle des vraies images : le générateur utilise son bruit, sans couvrir toute la variété d'écriture. Un générateur conditionnel qui **ignore** son bruit dessinerait une seule image par label : un effondrement de mode à l'intérieur de chaque classe (la graine 25 le fait, au bout de 3 epochs). Sans le label dans le discriminateur, rien n'obligerait le générateur à respecter le label : $D$ jugerait seulement « est-ce un chiffre ? ». Le conditionnement par le texte des modèles d'images remplace l'embedding d'un label par celui d'une phrase.
**Erreurs fréquentes** : réinitialiser les embeddings avec $\mathcal{N}(0 ;\, 0{,}02^2)$ (le label devient trop faible et le générateur l'ignore) ; donner aux faux les labels des vrais du mini-batch dans $D$ mais d'autres dans $G$ ; concaténer le label sur la mauvaise dimension.
**Variante** : avec des embeddings dix fois plus petits dans le générateur, la variété monte (0,22 à 0,36) mais le classifieur ne reconnaît plus que 24 % à 35 % des chiffres demandés : le label et le bruit se disputent l'entrée du générateur.

### Ex 27.26 — Se promener dans l'espace latent du générateur 🔬
**Réponse** : `latent_path_26` appelle `mylearn.vae.interpolate_latents` puis `generate` ; `path_report_26` calcule `predict` et la racine de la moyenne des carrés de `np.diff`. Trois chemins de dix images :

| chemin | classes | plus grand saut | norme de $\mathbf{z}$ |
|---|---|---|---|
| 1, ligne droite | 2 2 2 2 0 0 0 5 0 8 | 0,142 | de 8,3 à 5,9 |
| 1, grand cercle | 2 2 2 2 2 2 0 0 8 8 | 0,150 | de 8,3 à 7,7 |
| 2, ligne droite | 9 9 9 9 4 4 4 4 4 4 | 0,164 | de 8,0 à 5,6 |
| 3, ligne droite | 0 0 0 0 0 0 0 0 0 0 | 0,115 | de 8,6 à 5,5 |

**Démarche** : les images changent par petits pas (des sauts de 0,12 à 0,17, sur des pixels entre −1 et 1) ; le changement de classe se fait en une ou deux images, avec parfois une image ambiguë (un 5 au milieu d'un chemin de 2 à 8). En dimension 64, un bruit gaussien a presque toujours une norme proche de $\sqrt{64} = 8$ ; le milieu d'une ligne droite entre deux tirages indépendants a une norme d'environ $8/\sqrt{2} \approx 5{,}7$, une zone que le générateur n'a presque jamais vue pendant l'entraînement. Le grand cercle garde une norme proche de 8 : ses images du milieu ressemblent davantage à des chiffres.
**Erreurs fréquentes** : interpoler les images au lieu des codes (on obtient une superposition, pas un chemin dans l'espace des chiffres) ; générer en mode `train()` ; oublier de convertir les tenseurs en tableaux NumPy de flottants pour `interpolate_latents`.
**Variante** : l'arithmétique de Radford : la moyenne des $\mathbf{z}$ de trois « 7 barrés » moins celle de trois « 7 simples », plus celle de trois « 1 » ; à essayer en choisissant les $\mathbf{z}$ à la main, sans garantie avec un générateur de 3 epochs.

### Ex 27.27 — Checkpoint d'un GAN : deux réseaux, deux optimiseurs, états aléatoires, reprise à l'identique 🛠️
**Réponse** :
```python
def save_gan_checkpoint_27(path, G, D, opt_g, opt_d, epoch, gen):
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "epoch": int(epoch), "generator_state": gen.get_state(), "torch_rng_state": torch.get_rng_state()}, path)

def load_gan_checkpoint_27(path, G, D, opt_g, opt_d, gen):
    state = torch.load(path, weights_only=True)
    G.load_state_dict(state["G"])
    D.load_state_dict(state["D"])
    opt_g.load_state_dict(state["opt_g"])
    opt_d.load_state_dict(state["opt_d"])
    gen.set_state(state["generator_state"])
    torch.set_rng_state(state["torch_rng_state"])
    return state["epoch"]
```
La reprise est exacte, au bit près ; avec les poids seuls, l'écart atteint **0,022** après une epoch ; le checkpoint pèse **1,5 Mo**.
**Démarche** : sans l'état du générateur aléatoire, la seconde epoch voit d'autres mini-batchs et d'autres bruits ; sans les optimiseurs, Adam repart avec des moments nuls (ses premiers pas sont plus grands) ; les statistiques des batchnorms sont dans les `state_dict` des réseaux. `weights_only=True` interdit au fichier d'exécuter du code au chargement : un checkpoint téléchargé n'est plus un risque. En pratique, on garde le dernier checkpoint (pour reprendre) et le meilleur selon le FID (pour livrer), avec l'epoch et la configuration dans le nom ou à côté.
**Erreurs fréquentes** : sauver `G.parameters()` (sans les batchnorms) ; recréer les optimiseurs après le chargement (on perd leur état) ; sauver le générateur aléatoire lui-même au lieu de son état ; un fichier qui contient un objet Python quelconque, que `weights_only=True` refuse.
**Variante** : retire seulement `"generator_state"` du checkpoint. Réponse : la reprise n'est plus exacte (la seconde epoch tire d'autres mini-batchs), même avec des poids et des optimiseurs parfaits.

### Ex 27.28 — Défi DCGAN Fashion-MNIST : passer sous un seuil de distance de Fréchet 🏆
**Réponse** : la référence garde le DCGAN de 27.20 ($w = 32$, Adam $2 \cdot 10^{-4}$, $\beta_1 = 0{,}5$, mini-batchs de 64) et l'entraîne **8 epochs** sur 10 000 vêtements (66 s) : distance publique **21,0**, cachée **20,3**, pour un objectif de **33,6** (la pire de trois graines, plus 10 %). Le DCGAN de 27.20 tel quel (3 epochs, 25 s) reste entre 71 et 88.
**Démarche** : en 27.20 et 27.23, la distance baissait encore fortement d'une epoch à l'autre : le premier levier est donc le temps d'entraînement. Une epoch coûte environ 8 s : 8 epochs tiennent largement dans les 150 s. Sur trois graines, la référence donne 21 à 31 : l'écart entre graines est grand, d'où une marge dans l'objectif. D'autres pistes valent d'être mesurées une par une : plus d'images (les 60 000 en mode complet), des mini-batchs de 32 (plus de pas par epoch), des réseaux un peu plus larges (plus lents par epoch).
**Erreurs fréquentes** : changer plusieurs choses à la fois ; régler sur le jeu caché ; oublier de chronométrer (un entraînement de 160 s est refusé) ; juger à l'œil seulement ; entraîner sur les images de test.
**Variante** : vérifie avec `digit_report(G_28, CLF_FASHION)` que les dix classes de vêtements sont produites. Réponse attendue : toutes le sont, avec des déséquilibres (souvent plus de pantalons et de sacs, plus faciles).
