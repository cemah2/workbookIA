# B6 · Explicabilité, équité et éthique — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels et des exercices ✏️ B6.1, B6.3, B6.4 et B6.5 se vérifient aussi dans la partie 0 du notebook, où elles sont recalculées avec ta librairie (version de référence de `explain.py` et de `metrics.py`), shap (`explainers.Exact`), scikit-learn et torchvision ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B6.Q1 — Interprétable par construction ou expliqué après coup ?
a) **B** : six coefficients de features standardisées se lisent directement, le modèle **est** son explication ; la forêt (A) et le réseau (C) restent des boîtes noires, que TreeSHAP ou la saillance expliquent après coup. b) **Faux** : une explication post hoc **approche** le comportement du modèle, sans en être la règle ; elle peut même lui être infidèle (206.7). c) **A** : Grad-CAM lit les cartes d'activation d'une couche et leurs gradients, il ne marche que sur un réseau convolutif. d) **Vrai** : elle ne fait que mélanger une colonne et redemander des prédictions ; c'est une méthode agnostique. e) **16** : un arbre binaire de profondeur 4 a au plus $2^4 = 16$ feuilles, une règle par chemin de la racine à une feuille (15 nœuds internes).
**À retenir** : interprétable par construction = on lit le modèle ; post hoc = on lit une approximation, agnostique (par les prédictions) ou spécifique (par l'intérieur d'une famille).

### B6.Q2 — Explication globale ou locale : quel outil répond à quelle question ?
a) **C** : « pourquoi **ce** dossier » est une question locale. b) **A** : l'importance par permutation, sur des données que le modèle n'a pas vues, décrit le modèle entier ; un waterfall ou une courbe ICE ne parlent que d'un client. c) **Faux** : mélanger une colonne qui vaut presque toujours 0 ne change presque rien **en moyenne**, d'où une importance minuscule ; mais pour les 0,5 % de clients qui ont un prêt en cours, elle peut décider de tout. d) **B** : c'est l'importance globale du beeswarm, construite à partir des explications locales ; elle ne mélange aucune colonne (C) et ne dit plus rien d'un client (A). e) **0,19** : la moyenne des cinq scores mélangés vaut $3{,}20 / 5 = 0{,}64$, et $0{,}83 - 0{,}64 = 0{,}19$.
**À retenir** : global et local répondent à des questions différentes, et l'un ne se déduit pas de l'autre.

### B6.Q3 — Dépendance partielle et courbes ICE : ce que montre la moyenne, ce qu'elle cache
a) **C** : $\operatorname{PD}_1(v) = \frac{1}{n}\sum_i v\,x_{i,2} = v \times \bar x_2 = 0$ : la moyenne efface l'effet. b) **2** : les exemples où $x_2 = 1$ ont la courbe ICE $v \mapsto v$ (pente 1), ceux où $x_2 = -1$ la courbe $v \mapsto -v$ (pente $-1$) ; la PD plate cache deux pentes opposées. c) **7,67** : $\operatorname{PD}_1(2) = 3 \times 2 + \frac{1}{3}(0 + 1 + 4) = 6 + \frac{5}{3}$. d) **Vrai** : `centered=True` retire à chaque courbe sa valeur au premier point de la grille ; deux courbes parallèles ne diffèrent que d'une constante, qui disparaît. e) **A** : la formule impose $v$ à **tous** les exemples sans toucher aux autres features ; d'où des logements de 200 m² et 2 pièces, que le modèle n'a jamais vus.
**À retenir** : la dépendance partielle est une moyenne de courbes ICE ; elle ment quand les courbes se croisent (interaction) et quand on l'évalue hors des données.

### B6.Q4 — Les quatre propriétés de Shapley : efficacité, symétrie, joueur nul, additivité
a) **0,22** : l'efficacité donne $\phi_1 + \phi_2 + \phi_3 = 0{,}72 - 0{,}40 = 0{,}32$, d'où $\phi_3 = 0{,}32 - 0{,}15 + 0{,}05$. b) **A** : le joueur nul. c) **0,2** : l'additivité ajoute les parts des deux jeux, $0{,}3 - 0{,}1$. d) **Faux** : la symétrie demande que les deux joueurs apportent la même chose **à toute coalition**, ce qui dépend des valeurs du client ; avec $f = x_1 + x_2$, un fond $(0, 0)$ et $\mathbf{x} = (1, 5)$, $\phi_1 = 1$ et $\phi_2 = 5$. e) **16** : chacune des 4 autres features est dans la coalition ou non, $2^4$.
**À retenir** : efficacité (tout l'écart est réparti), symétrie (même apport, même part), joueur nul (aucun apport, part nulle), additivité (les parts d'une somme s'ajoutent).

### B6.Q5 — Changer le fond (background) change les valeurs SHAP : vrai ou faux
a) **Faux** : la somme vaut $f(\mathbf{x}) - \mathbb{E}[f(\mathbf{b})]$, et la prédiction moyenne sur le fond change avec le fond. b) **C** : la valeur de base est la moyenne de $f$ sur le fond, ici une seule ligne. c) **2** $= 2 \times (4 - 3)$. d) **−2** $= 2 \times (4 - 5)$ : même modèle, même client, et le signe s'inverse ; le client est au-dessus de la moyenne du fond A, en dessous de celle du fond B. e) **Faux** : sans fond, `TreeExplainer` suit les branches au prorata de l'entraînement (« chemins de l'arbre ») ; avec un fond, il calcule le Shapley interventionnel ; ce ne sont pas les mêmes valeurs. f) **100** : `max_samples` vaut 100 par défaut, et shap sous-échantillonne le fond ; passe `max_samples=len(bg)` pour garder toutes les lignes.
**À retenir** : une valeur SHAP répond à « pourquoi ce score plutôt que le score moyen **de ce fond** » ; écris toujours lequel.

### B6.Q6 — Features corrélées : ce que SHAP répartit et ce qu'il ne dit pas
a) **2** $= 2 \times (1{,}5 - 0{,}5)$, et de même $\phi_2 = 2$. b) **0** : le modèle B ne lit pas $x_2$, et le Shapley interventionnel est fidèle au modèle. c) **Vrai** : 4 dans les deux cas ($2 + 2$, puis $4 + 0$). d) **C** : en tirant $x_1$ sachant $x_2$, il « devine » $x_1$ à partir de sa copie, et crédite $x_2$ d'une part de l'effet, alors que B ne la lit pas ; c'est la critique « fidèle aux données » contre « fidèle au modèle ». e) **A** : la somme des valeurs d'un groupe de features corrélées est stable d'un modèle à l'autre, leur partage dépend d'un détail d'entraînement.
**À retenir** : entre features corrélées, SHAP répartit selon ce que le modèle a fait, pas selon le monde ; on rapporte la somme du groupe.

### B6.Q7 — Gradient, gradient × entrée, gradients intégrés : pourquoi une ligne de référence ?
a) **0** : en $x = 3$, $f$ est plate. b) **1** : le chemin de 0 à 3 a une pente de 1 sur son premier tiers, 0 ensuite ; $\operatorname{IG} = 3 \times \frac{1}{3} = 1 = f(3) - f(0)$, la complétude. c) **0** : de 2 à 3, $f$ est plate ; et $f(3) - f(2) = 0$. La même entrée reçoit 1 ou 0 selon la référence : la référence fait partie de la question. d) **Vrai** : le gradient d'un modèle linéaire est $\mathbf{w}$ partout, donc $\operatorname{IG}_i = (x_i - 0)\,w_i$, exactement gradient × entrée. e) **A** : un pixel du fond est égal à sa référence ; le facteur $x_i - x'_i$ y est nul, quelle que soit l'importance du fond pour le modèle (une référence noire ferait de même avec un objet noir).
**À retenir** : un gradient est local et s'annule quand le modèle sature ; les gradients intégrés expliquent un écart à une référence, qu'il faut choisir et écrire.

### B6.Q8 — Grad-CAM : pourquoi la dernière couche convolutive ?
a) **5** : `layer4` divise par 32, $160 / 32 = 5$. b) **B** : le pooling moyen global résume chaque carte en un nombre et efface la position ; c'est pourquoi on s'arrête juste avant. c) **56** $= 224 / 4$ : une carte fine, qui suit les bords et les textures plus que la classe. d) **Vrai** : un poids $\alpha_c$ négatif (un canal qui « vote contre ») peut rendre la somme négative ; le ReLU ne garde que ce qui augmente le score. e) **C** : le logit, avant la softmax.
**À retenir** : la dernière couche convolutive est le meilleur compromis entre le sens (des canaux liés aux classes) et la localisation (avant le pooling global).

### B6.Q9 — Une carte de saillance qui ne dépend pas du modèle : que prouve le test de randomisation ?
a) **B** : une carte qui ne bouge pas quand le réseau ne sait plus rien ne dépend pas de ce qu'il a appris ; c'est un détecteur de bords, pas une explication. b) **Faux** : réussir le test est **nécessaire** (la carte dépend du modèle), pas suffisant (elle peut en dépendre et rester infidèle). c) **C** ; (A) est la randomisation indépendante, (B) la randomisation des données. d) **Faux** : une corrélation de rangs de 1 dit que les pixels sont rangés dans le même ordre ; $(1, 2, 3)$ et $(10, 20, 30)$ ont une corrélation de rangs de 1 et des valeurs différentes. e) **0,1** : appris sur des labels au hasard, le réseau ne fait pas mieux que le hasard sur un test aux vrais labels, une chance sur dix.
**À retenir** : un test de santé raté disqualifie une méthode pour ce modèle ; un test réussi ne la qualifie pas à lui seul.

### B6.Q10 — Sources de biais : données, labels, variables de substitution, boucles de rétroaction
a) **A** : un groupe sous-représenté dans l'entraînement. b) **B** : la cible elle-même (le jugement des managers) porte leurs préjugés. c) **C** : le modèle choisit les contrôles, les contrôles fabriquent les labels, les labels confirment le modèle. d) **Vrai** : certains sports sont pratiqués surtout par des hommes ou surtout par des femmes ; la liste porte donc une part de l'information sur le sexe, même une fois la colonne `sexe` retirée. e) **A** : l'instrument mesure moins bien pour certains patients.
**À retenir** : demande où est le défaut : dans qui est représenté, dans la cible, dans un instrument, dans une colonne qui en remplace une autre, ou dans des décisions qui fabriquent les données.

### B6.Q11 — Parité démographique, égalité des chances, calibration : peut-on tout avoir ?
a) **0,35** : un classifieur parfait sélectionne exactement les positifs ; ses taux de sélection valent les taux de base, $0{,}6 - 0{,}25$. b) **Vrai** : TPR = 1 et FPR = 0 dans les deux groupes. c) **Vrai** : precision = 1 dans les deux groupes ; c'est l'une des deux issues du théorème d'impossibilité, le classifieur parfait. d) **0,20** $= \frac{0{,}5}{0{,}5} \times \frac{0{,}2}{0{,}8} \times 0{,}8$. e) **A** : l'erreur coûteuse est un faux négatif (refuser qui rembourserait) ; égaliser le TPR, c'est donner à chaque groupe la même chance d'être accepté quand on aurait remboursé.
**À retenir** : sans classifieur parfait ni taux de base égaux, il faut choisir son critère selon l'erreur qui coûte, et à qui.

### B6.Q12 — AI Act : dans quelle catégorie de risque tombe un scoring de crédit ?
a) **B** : annexe III, point 5 b), « l'évaluation de la solvabilité des personnes physiques ». b) **A** : le point 5 b) excepte les systèmes de détection de la fraude financière (le projet de lignes directrices de 2026 limite l'exception aux systèmes dont c'est l'usage principal). c) **C** : la notation sociale est une pratique interdite (art. 5). d) **Faux** : le règlement 2026/1744 a repoussé ces obligations au 2 décembre 2027. e) **B** : l'article 27 vise les déployeurs des systèmes des points 5 b) et c). f) **Faux** : le texte vise les « personnes physiques », pas les entreprises.
**À retenir** : un scoring de crédit de particuliers est à haut risque ; le fournisseur conçoit et documente, le déployeur supervise, analyse l'impact et explique.

<a id="rappels"></a>

## 🔁 Rappels

### B6.R1 — Ch. 21 : visualiser ce qu'a appris un CNN, et l'attaque FGSM
a) **[0.37, 1.0, 0.1]** : $0{,}40 - 0{,}03$ ; $0{,}98 + 0{,}03 = 1{,}01$, ramené à 1 ; le signe de 0 vaut 0, le troisième pixel ne bouge pas. b) **Vrai** : les deux font la même passe arrière jusqu'aux **pixels**, les poids restant fixes ; FGSM dérive la loss et n'en garde que le signe, la saillance dérive le score de la classe et en garde la valeur absolue (puis le maximum sur les trois canaux). c) **B** : les poids sont gelés ; on modifie l'image de départ par montée de gradient. d) **0,31** : $\varepsilon\sum_j \lvert w_j \rvert = 0{,}01 \times 3\,072 \times 0{,}01 = 0{,}3072$ ; chaque pixel bouge d'un centième, le score de 0,3. e) **Faux** : la montée de gradient cherche un **maximum** de l'activation.
**À retenir** : le gradient par rapport à l'entrée sert à attaquer (FGSM), à visualiser (montée de gradient) et à expliquer (saillance) ; en grande dimension, de petites retouches s'additionnent.

### B6.R2 — Ch. 14 : feature_importances_ contre importance par permutation
a) **A** : un identifiant offre un seuil par ligne ; un arbre profond s'en sert pour séparer les exemples d'entraînement, et la MDI le récompense. b) **Vrai** : la MDI additionne des baisses d'impureté de l'entraînement, y compris celles qui ne servent qu'à apprendre par cœur. c) **1** : les importances MDI sont normalisées pour faire 1. d) **B** : sur la validation, l'identifiant ne dit rien de la cible ; le mélanger ne coûte rien, aux fluctuations près. e) **8** $= 0{,}02 \times 400$.
**À retenir** : la MDI est gratuite mais calculée sur l'entraînement et biaisée vers les features à beaucoup de valeurs ; la permutation sur la validation la corrige.

### B6.R3 — Ch. 3 : TPR, FPR et precision à partir d'une matrice de confusion
a) **0,80** $= \frac{40}{40 + 10}$. b) **0,12** $= \frac{12}{12 + 88}$. c) **0,769** $= \frac{40}{40 + 12}$. d) **0,347** $= \frac{40 + 12}{150}$. e) **A** : ligne « vérité 0 », colonne « prédit 1 » : les faux positifs.
**À retenir** : `[[TN, FP], [FN, TP]]` ; le TPR et le FPR se lisent sur les lignes (la vérité), la precision et le taux de sélection sur la colonne « prédit 1 ».

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B6.1 — Valeurs de Shapley exactes à la main : trois features, huit coalitions ✏️
**Réponse** : a) **6** ; b) **8** ; c) **12** ; d) **0,17** ; e) **3,5** ; f) **−1** ; g) **2,5** ; h) **Faux** ; i) **16** ; j) **4,0** ; k) **B**.
**Démarche** : avec $f(\mathbf{x}) = 2x_1 + x_1x_3 + x_2$, les huit coalitions (masque $(s_1, s_2, s_3)$, lignes construites, moyenne) :

| $S$ | lignes | $v(S)$ |
|---|---|---|
| $\varnothing$ | $(0, 2, 0)$, $(2, 2, 2)$ | $(2 + 10)/2 = 6$ |
| $\{1\}$ | $(2, 2, 0)$, $(2, 2, 2)$ | $(6 + 10)/2 = 8$ |
| $\{2\}$ | $(0, 1, 0)$, $(2, 1, 2)$ | $(1 + 9)/2 = 5$ |
| $\{3\}$ | $(0, 2, 3)$, $(2, 2, 3)$ | $(2 + 12)/2 = 7$ |
| $\{1, 2\}$ | $(2, 1, 0)$, $(2, 1, 2)$ | $(5 + 9)/2 = 7$ |
| $\{1, 3\}$ | $(2, 2, 3)$ deux fois | $12$ |
| $\{2, 3\}$ | $(0, 1, 3)$, $(2, 1, 3)$ | $(1 + 11)/2 = 6$ |
| $\{1, 2, 3\}$ | $(2, 1, 3)$ deux fois | $11$ |

Avec $M = 3$, les poids valent $\frac{0!\,2!}{3!} = \frac13$ (taille 0 ou 2) et $\frac{1!\,1!}{3!} = \frac16$ (taille 1), d'où d). Feature 1 : contributions $8 - 6 = 2$ (depuis $\varnothing$), $7 - 5 = 2$ (depuis $\{2\}$), $12 - 7 = 5$ (depuis $\{3\}$), $11 - 6 = 5$ (depuis $\{2, 3\}$), donc $\phi_1 = \frac13 \cdot 2 + \frac16 \cdot 2 + \frac16 \cdot 5 + \frac13 \cdot 5 = 3{,}5$. Feature 2 : $5 - 6$, $7 - 8$, $6 - 7$, $11 - 12$, toutes égales à $-1$, donc $\phi_2 = -1$ : $x_2$ entre seule et linéairement, son apport ne dépend pas de la coalition ($x_2 - \bar b_2 = 1 - 2$). Feature 3 : $7 - 6 = 1$, $12 - 8 = 4$, $6 - 5 = 1$, $11 - 7 = 4$, donc $\phi_3 = \frac13 + \frac46 + \frac16 + \frac43 = 2{,}5$. Efficacité : $3{,}5 - 1 + 2{,}5 = 5 = 11 - 6$. h) Le terme $2x_1$ apporte $2(x_1 - \bar b_1) = 2$ à $\phi_1$ ; l'interaction $x_1x_3$ vaut 6 chez le client et $(0 + 4)/2 = 2$ en moyenne sur le fond ; son écart de 4 se partage en $1{,}5$ (feature 1) et $2{,}5$ (feature 3), pas à parts égales. Ligne par ligne (∂ B6.2, question 8) : la ligne $(0, 2, 0)$ partage son écart de 6 en 3 et 3 ; sur la ligne $(2, 2, 2)$, $x_1 = b_1$ : la feature 1 n'y apporte rien, et tout l'écart de 2 va à la feature 3 ; la moyenne des deux lignes donne 1,5 et 2,5. Le partage suit l'écart de chaque feature à ses valeurs du fond. i) $2^3 \times 2 = 16$ évaluations, chaque $v(S)$ servant aux trois features ; la coalition pleine compte aussi ses deux lignes (deux fois $f(\mathbf{x})$). j) Avec la seule ligne $(1, 2, 1)$ : $\phi_1 = 2(2 - 1) + \frac12(2 - 1)(3 + 1) = 4$ (∂ B6.2, question 8), $\phi_2 = -1$, $\phi_3 = \frac12(3 - 1)(2 + 1) = 3$, valeur de base $f(1, 2, 1) = 5$ : remplacer un fond par sa ligne moyenne change les valeurs, parce que $f$ n'est pas linéaire. k) `phi[2]` est la colonne 2, la feature 3. Dans ta copie : la valeur de base (6) n'est pas $f(1, 2, 1) = 5$ : la moyenne des prédictions n'est pas la prédiction de la moyenne quand $f$ a un produit.
**Erreurs fréquentes** : calculer $f$ sur la ligne moyenne au lieu de la moyenne de $f$ sur les lignes ; inverser le masque (prendre du client les features **absentes**) ; des poids uniformes ($\frac14$ ou $\frac18$) au lieu des poids de Shapley ; ne garder qu'une contribution marginale ; recalculer les $v(S)$ pour chaque feature ; confondre `phi[2]` et $\phi_2$.
**Variante** : même modèle, même fond, client $\mathbf{x} = (2, 2, 0)$. Réponse : $f(\mathbf{x}) = 6$, égal à la valeur de base, et pourtant $\boldsymbol{\phi} = (2 ;\ 0 ;\ -2)$ : $x_2$ vaut 2 comme dans tout le fond (joueur nul pour ce client), et les apports de $x_1$ et de $x_3$ se compensent. Une prédiction égale à la moyenne ne veut pas dire « rien à expliquer ».

### Ex B6.2 — Modèle linéaire et fond indépendant : φ_i = w_i (x_i − E[x_i]) ∂
**Réponse** : $\phi_i = w_i(x_i - \bar b_i)$ et $v(\varnothing) = f(\bar{\mathbf{b}})$ ; application : $\boldsymbol{\phi} = (1 ;\ 0 ;\ 0)$, valeur de base 0, $f(\mathbf{x}) = 1$ ; pour le produit avec une ligne de fond, $\phi_1 = \frac12(x_1 - b_1)(x_2 + b_2)$ et $\phi_2 = \frac12(x_2 - b_2)(x_1 + b_1)$.
**Démarche** :
1. Pour une ligne $\mathbf{b}$, $f(\mathbf{x}_S, \mathbf{b}_{\bar S}) = w_0 + \sum_{j \in S} w_j x_j + \sum_{j \notin S} w_j b_j$. Les deux premiers termes ne dépendent pas de la ligne ; par (i), la moyenne sur le fond donne $v(S) = w_0 + \sum_{j \in S} w_j x_j + \sum_{j \notin S} w_j \bar b_j$.
2. Pour $i \notin S$, passer de $S$ à $S \cup \{i\}$ remplace le terme $w_i \bar b_i$ par $w_i x_i$, et rien d'autre : $v(S \cup \{i\}) - v(S) = w_i(x_i - \bar b_i)$. La contribution ne dépend **pas** de $S$ : c'est la signature d'un modèle sans interaction.
3. $\phi_i = \sum_{S \not\ni i} \text{poids}(S) \times w_i(x_i - \bar b_i) = w_i(x_i - \bar b_i) \times \sum_{S \not\ni i}\text{poids}(S) = w_i(x_i - \bar b_i)$ par (ii).
4. $v(\varnothing) = w_0 + \sum_j w_j \bar b_j = f(\bar{\mathbf{b}})$ ; $\sum_i \phi_i = \sum_i w_i x_i - \sum_i w_i \bar b_i = f(\mathbf{x}) - f(\bar{\mathbf{b}})$. En ✏️ B6.1, $f$ contient un produit : la moyenne des $b_1 b_3$ (2) n'est pas le produit des moyennes (1), et $v(\varnothing) = 6 \neq f(1, 2, 1) = 5$.
5. Moyennes du fond : $(2, 1, 3)$. $\phi_1 = 0{,}5 \times (4 - 2) = 1$ ; $\phi_2 = -2 \times (1 - 1) = 0$ ; $\phi_3 = 0 \times (7 - 3) = 0$. Valeur de base $1 + 0{,}5 \times 2 - 2 \times 1 + 0 = 0$, $f(\mathbf{x}) = 1 + 2 - 2 + 0 = 1$ : l'efficacité tient. $x_2$ reçoit 0 parce qu'il est **à la moyenne du fond** ; $x_3$ parce que le modèle **ne le lit pas** (joueur nul), bien qu'il soit loin de sa moyenne.
6. Non : la formule ne dépend du fond que par ses moyennes de colonnes. Le Shapley interventionnel remplace chaque feature absente indépendamment des autres ; il ignore les corrélations du fond (c'est l'hypothèse d'indépendance de Lundberg et Lee).
7. Sur le **logit** $w_0 + \mathbf{w}^\top\mathbf{x}$, qui est linéaire. La probabilité $\sigma(\cdot)$ ne l'est pas : la contribution de $x_i$ y dépend des autres features (la pente de la sigmoïde change avec le logit), et la formule ne vaut plus ; c'est pourquoi le choix de la sortie expliquée change les valeurs (206.4).
8. Avec une ligne $\mathbf{b}$ : $v(\varnothing) = b_1b_2$, $v(\{1\}) = x_1b_2$, $v(\{2\}) = b_1x_2$, $v(\{1, 2\}) = x_1x_2$. $\phi_1 = \frac12\big[(x_1b_2 - b_1b_2) + (x_1x_2 - b_1x_2)\big] = \frac12(x_1 - b_1)(b_2 + x_2)$, et de même pour $\phi_2$. Somme : $\frac12\big[(x_1 - b_1)(x_2 + b_2) + (x_2 - b_2)(x_1 + b_1)\big] = x_1x_2 - b_1b_2 = f(\mathbf{x}) - f(\mathbf{b})$. Docstring : $\frac12(1 - 0)(1 + 0) = \frac12$ chacun. ✏️ B6.1 j) : la part de l'interaction dans $\phi_1$ vaut $\frac12(2 - 1)(3 + 1) = 2$, celle de $\phi_3$ vaut $\frac12(3 - 1)(2 + 1) = 3$.

**Erreurs fréquentes** : oublier que la moyenne porte sur les lignes du fond (et écrire $v(S)$ avec les valeurs d'une seule ligne) ; croire que la valeur de base est $w_0$ (c'est $f(\bar{\mathbf{b}})$) ; appliquer la formule à la probabilité d'une régression logistique ; conclure que deux features corrélées se partagent l'effet (le Shapley interventionnel ne voit que les coefficients).
**Variante** : $f(\mathbf{x}) = x_1 + 2x_2 + x_1x_2$, client $(2, 1)$, fond d'une ligne $(0, 1)$. Réponse : $\phi_1 = 1 \times 2 + \frac12(2 - 0)(1 + 1) = 4$, $\phi_2 = 2 \times 0 + \frac12(1 - 1)(2 + 0) = 0$, et $4 + 0 = f(2, 1) - f(0, 1) = 6 - 2$ : une feature égale à sa valeur de fond reçoit 0, même quand elle interagit.

### Ex B6.3 — Gradients intégrés à la main sur f(x) = x₁² + 3x₂ et vérification de la complétude ✏️
**Réponse** : a) **[4, 3]** ; b) **[8, −3]** ; c) **1** ; d) **[4, −3]** ; e) **3,2** ; f) **4,8** ; g) **4,0** ; h) **134** ; i) **3** ; j) **0** ; k) **3,33** ; l) **C**.
**Démarche** : $\nabla f = (2x_1, 3)$. a) En $(2, -1)$ : $(4, 3)$. b) $(4 \times 2,\ 3 \times (-1))$, de somme 5, alors que c) l'écart à expliquer vaut $f(2, -1) - f(0, 0) = 4 - 3 = 1$ : gradient × entrée ne vérifie pas la complétude dès que $f$ n'est pas linéaire. d) Au point $\alpha\mathbf{x}$ du chemin, $\partial_1 f = 2\alpha x_1$ ; $\operatorname{IG}_1 = x_1\int_0^1 2\alpha x_1\,d\alpha = x_1^2 = 4$ ; $\operatorname{IG}_2 = x_2 \times 3 = -3$ ; la somme vaut 1 : la complétude. e) Points $0 ;\ 0{,}2 ;\ 0{,}4 ;\ 0{,}6 ;\ 0{,}8$, gradients $4\alpha$ : moyenne $4 \times 0{,}4 = 1{,}6$, et $\operatorname{IG}_1 = 2 \times 1{,}6 = 3{,}2$. f) Points $0{,}2$ à $1$ : moyenne $4 \times 0{,}6 = 2{,}4$, $\operatorname{IG}_1 = 4{,}8$. g) $\frac15\big[\frac{0}{2} + 0{,}8 + 1{,}6 + 2{,}4 + 3{,}2 + \frac{4}{2}\big] = 2$, $\operatorname{IG}_1 = 4{,}0$ : exact. h) La règle de gauche donne $\operatorname{IG}_1 = x_1^2\,\frac{n - 1}{n}$, d'où une erreur $\frac{4}{n}$ ($\operatorname{IG}_2$ est exact) ; $\frac{4}{n} \le 0{,}03$ pour $n \ge 133{,}3$, soit $n = 134$ (avec 133 intervalles, l'erreur vaut 0,03008). i) Le chemin est $(1 + \alpha,\ -\alpha)$ ; $\operatorname{IG}_1 = (2 - 1)\int_0^1 2(1 + \alpha)\,d\alpha = 3 = x_1^2 - x_1'^2$. j) $\operatorname{IG}_2 = (-1 - 0) \times 3 = -3$, et $3 - 3 = 0 = f(2, -1) - f(1, 0) = 1 - 1$. k) Six points $0, 0{,}2, \dots, 1$, poids $\frac16$ divisés par deux aux extrémités : $\frac16\big[0 + 0{,}8 + 1{,}6 + 2{,}4 + 3{,}2 + 2\big] = \frac{10}{6}$, $\operatorname{IG}_1 = \frac{20}{6} \approx 3{,}33 = 4 \times \frac56$ ; captum perd un sixième de l'attribution, et sa complétude est fausse d'autant. l) **C** : `n_steps` compte les intervalles. Dans ta copie : la dérivée le long du chemin est **linéaire** en $\alpha$, et la règle des trapèzes intègre exactement une fonction affine (l'aire d'un trapèze) ; 'left' prend les petites valeurs d'une fonction croissante et sous-estime, 'right' surestime d'autant. $\operatorname{IG}_2$ intègre une constante, exacte avec toute règle. En j), une somme nulle dit seulement que $f(\mathbf{x}) = f(\mathbf{x}')$ : les deux features ont des effets opposés de 3, qui se compensent.
**Erreurs fréquentes** : oublier le facteur $x_i - x'_i$ ; confondre le nombre de points et le nombre d'intervalles (captum contre ta docstring) ; ne pas diviser par deux les extrémités du trapèze ; partir de $\alpha\mathbf{x}$ au lieu de $\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}')$ quand la référence n'est pas nulle ; lire une somme nulle comme une absence d'effet.
**Variante** : $f(x) = x^3$ en une dimension, $x = 2$, référence 0, règle des trapèzes avec $n = 2$. Réponse : le gradient le long du chemin vaut $3(\alpha \cdot 2)^2 = 12\alpha^2$ ; trapèze $\frac12\big[0 + 3 + \frac{12}{2}\big] = 4{,}5$, $\operatorname{IG} = 2 \times 4{,}5 = 9$ au lieu de l'exact $8$ : sur une dérivée courbée, même le trapèze se trompe, et il faut augmenter $n$.

### Ex B6.4 — Grad-CAM à la main : deux cartes 3×3 et leurs gradients ✏️
**Réponse** : a) **1,0** ; b) **−0,5** ; c) **−1,0** ; d) **3** ; e) **0,33** ; f) **B** ; g) **−0,33** ; h) **C** ; i) **Faux** ; j) **12**.
**Démarche** : a) Les gradients du canal 1 somment à 9, sur 9 cases : $\alpha_1 = 1$. b) Ceux du canal 2 somment à $-3 - 1{,}5 = -4{,}5$ : $\alpha_2 = -0{,}5$. La somme pondérée $A^1 - 0{,}5\,A^2$ vaut $\begin{pmatrix} 1 & -0{,}5 & -1 \\ 2 & 1 & -0{,}5 \\ 3 & 2 & 1 \end{pmatrix}$. c) En haut à droite, $1 \times 0 - 0{,}5 \times 2 = -1$. d) Trois cases négatives, mises à 0 par le ReLU. e) Le maximum après ReLU vaut 3 (en bas à gauche, d'où f) ; le centre vaut $1/3 \approx 0{,}33$. g) Sans ReLU, la plus grande valeur absolue est encore 3, et $-1/3 \approx -0{,}33$. h) $-0{,}5\,A^1 + A^2 = \begin{pmatrix} -0{,}5 & 1 & 2 \\ -1 & -0{,}5 & 1 \\ -1{,}5 & -1 & -0{,}5 \end{pmatrix}$ : le maximum est en haut à droite ; une autre classe éclaire une autre région de la même couche. i) Les $\alpha_c$ ne lisent que les gradients : ils ne changent pas, et la somme pondérée, donc toute la carte, est multipliée par 10 (le ReLU garde les mêmes cases) ; la division par le maximum l'annule. Il en irait de même en multipliant les gradients par 10. j) Une case couvre $72 / 3 = 24$ pixels de l'image ; un demi-pixel de la couche en vaut 12. Dans ta copie : le canal 2 a un poids négatif, il « vote contre » la classe $k$ et éteint le coin en haut à droite. Pour ta `grad_cam`, `A` a la forme `(1, 2, 3, 3)` : une image, deux canaux, $3 \times 3$.
**Erreurs fréquentes** : prendre la somme des gradients au lieu de leur moyenne (sans effet après normalisation, mais faux sans elle) ; multiplier gradients et activations case par case avant de moyenner ; appliquer le ReLU canal par canal avant la somme ; normaliser par la somme au lieu du maximum ; oublier le signe des poids.
**Variante** : la carte normalisée de la classe $k'$ (après ReLU). Réponse : $\begin{pmatrix} 0 & 0{,}5 & 1 \\ 0 & 0 & 0{,}5 \\ 0 & 0 & 0 \end{pmatrix}$ : trois cases non nulles, le long du bord en haut à droite, là où la classe $k$ n'a rien.

### Ex B6.5 — Audit à la main : deux groupes, deux matrices de confusion, cinq indicateurs ✏️
**Réponse** : a) **0,24** ; b) **0,182** ; c) **0,60** ; d) **0,125** ; e) **0,791** ; f) **0,28** ; g) **0,20** ; h) **0,39** ; i) **Faux** ; j) **Vrai** ; k) **24**.
**Démarche** : femmes, $n = 500$ : taux de base $\frac{120}{500} = 0{,}24$ ; sélection $\frac{91}{500} = 0{,}182$ ; TPR $\frac{72}{120} = 0{,}60$ ; FPR $\frac{19}{380} = 0{,}05$ ; precision $\frac{72}{91} \approx 0{,}791$ ; accuracy $\frac{433}{500} = 0{,}866$. Hommes, $n = 640$ : taux de base $0{,}5$ ; sélection $\frac{296}{640} = 0{,}4625$ ; TPR $0{,}80$ ; FPR $\frac{40}{320} = 0{,}125$ ; precision $\frac{256}{296} \approx 0{,}865$ ; accuracy $\frac{536}{640} \approx 0{,}838$. f) $0{,}4625 - 0{,}182 = 0{,}2805 \approx 0{,}28$. g) Écart de TPR $0{,}20$, écart de FPR $0{,}075$ : $\Delta_{\text{EO}} = 0{,}20$. h) $\frac{0{,}182}{0{,}4625} \approx 0{,}39$. i) $0{,}39 < 0{,}8$ : un fort indice d'effet défavorable au sens de la règle des quatre cinquièmes. j) $0{,}866 - 0{,}838 \approx 0{,}03$ : moins de 5 points, et l'accuracy des femmes est même la plus haute. k) $0{,}8 \times 120 = 96$ qualifiées sélectionnées, soit 24 de plus. Dans ta copie : pour une présélection d'entretiens, l'erreur coûteuse est d'écarter une candidate qualifiée, ce qui plaide pour l'égalité des chances (même TPR) ; la parité démographique ignorerait la différence de taux de base qu'a mesurée le jury (0,24 contre 0,5), qu'il faut d'ailleurs interroger (le vivier, ou le jury ?). Baisser le seuil sélectionne aussi des non-qualifiées : le FPR des femmes montera. Avec 120 qualifiées, l'intervalle bootstrap du TPR des femmes est large : rapporte-le.
**Erreurs fréquentes** : diviser le TPR par tout le groupe au lieu des positifs réels ; prendre l'écart des taux de base pour la parité démographique ; additionner les écarts de TPR et de FPR ; juger l'équité à l'accuracy (ici, presque égale) ; inverser le rapport des quatre cinquièmes.
**Variante** : le seuil des femmes est baissé comme en k), et les 24 qualifiées de plus s'accompagnent de 19 faux positifs de plus (TP 96, FN 24, FP 38, TN 342). Réponse : sélection 0,268, TPR 0,80, FPR 0,10, precision 0,716 ; $\Delta_{\text{EO}}$ tombe à $0{,}025$ (l'écart de FPR), mais $\Delta_{\text{DP}}$ ne descend qu'à $0{,}19$ et le rapport des quatre cinquièmes ne monte qu'à $0{,}58$ : l'égalité des chances est atteinte, pas la parité démographique, et la precision des femmes baisse.

### Ex B6.6 — Impossibilité : taux de base différents, calibration et égalité des erreurs incompatibles ∂
**Réponse** : $\operatorname{FPR} = \frac{p}{1 - p}\cdot\frac{1 - \operatorname{PPV}}{\operatorname{PPV}}\cdot(1 - \operatorname{FNR})$ ; à PPV et FNR égaux, les FPR sont dans le rapport des cotes $\frac{p}{1 - p}$ (0,055 contre 0,175 dans l'application) ; à FPR et FNR égaux, la PPV croît avec $p$ (0,716 contre 0,889) ; avec des taux de base différents, les trois égalités ne tiennent ensemble que si aucun groupe n'a de faux positif (PPV = 1, FPR = 0), dont le classifieur parfait est le cas extrême.
**Démarche** :
1. $pn$ positifs réels et $(1 - p)n$ négatifs. $TP = (1 - \operatorname{FNR})\,pn$, $FN = \operatorname{FNR}\,pn$.
2. $\operatorname{PPV}(TP + FP) = TP$, donc $FP = TP\,\frac{1 - \operatorname{PPV}}{\operatorname{PPV}}$.
3. $\operatorname{FPR} = \frac{FP}{(1 - p)n} = \frac{(1 - \operatorname{FNR})\,pn}{(1 - p)n}\cdot\frac{1 - \operatorname{PPV}}{\operatorname{PPV}}$, la relation annoncée.
4. Bayes : $\operatorname{PPV} = \frac{P(\hat{y} = 1 \mid y = 1)\,P(y = 1)}{P(\hat{y} = 1)} = \frac{\operatorname{TPR}\,p}{\operatorname{TPR}\,p + \operatorname{FPR}\,(1 - p)}$. Donc $\frac{1}{\operatorname{PPV}} - 1 = \frac{\operatorname{FPR}\,(1 - p)}{\operatorname{TPR}\,p}$, et l'on isole le FPR : la même relation, avec $\operatorname{TPR} = 1 - \operatorname{FNR}$.
5. Femmes : $\frac{0{,}24}{0{,}76} \times \frac{19/91}{72/91} \times 0{,}6 = 0{,}3158 \times 0{,}2639 \times 0{,}6 = 0{,}05$ ; hommes : $1 \times \frac{40}{256} \times 0{,}8 = 0{,}125$ : les FPR de ✏️ B6.5. La relation est une identité, vraie pour tout classifieur.
6. Les facteurs $\frac{1 - \operatorname{PPV}}{\operatorname{PPV}}$ et $1 - \operatorname{FNR}$ sont communs ; s'ils ne sont pas nuls (PPV < 1, FNR < 1), ils se simplifient : $\frac{\operatorname{FPR}_A}{\operatorname{FPR}_B} = \frac{p_A/(1 - p_A)}{p_B/(1 - p_B)}$. S'ils sont nuls, les deux FPR valent 0. Application : $\frac{0{,}24}{0{,}76} \times 0{,}25 \times 0{,}7 \approx 0{,}055$ et $1 \times 0{,}25 \times 0{,}7 = 0{,}175$, un rapport de 3,2 : le groupe au taux de base le plus haut subit plus de faux positifs.
7. $\operatorname{PPV} = \frac{p(1 - \operatorname{FNR})}{p(1 - \operatorname{FNR}) + (1 - p)\operatorname{FPR}} = \frac{1 - \operatorname{FNR}}{(1 - \operatorname{FNR}) + \frac{1 - p}{p}\operatorname{FPR}}$ ; quand $p$ croît, $\frac{1 - p}{p}$ décroît et la PPV croît (strictement si FPR > 0 ; si FPR = 0, elle vaut 1 pour tout $p$). Application : $\frac{0{,}192}{0{,}192 + 0{,}076} \approx 0{,}716$ et $\frac{0{,}4}{0{,}4 + 0{,}05} \approx 0{,}889$ : à erreurs égales, une décision positive est moins fiable dans le groupe au taux de base bas.
8. Si $p_A = p_B$, rien ne s'oppose aux trois égalités. Si $p_A \neq p_B$, les cotes diffèrent : à PPV et FNR égaux, la question 6 dit que les deux FPR ne peuvent être égaux que s'ils sont nuls, et la relation de 3 (avec $TP > 0$, donc FNR < 1) impose alors PPV = 1 : aucun faux positif dans aucun groupe, quel que soit le FNR commun. Le classifieur parfait ($\operatorname{FNR} = \operatorname{FPR} = 0$, $\operatorname{PPV} = 1$) en est le cas extrême : la relation y donne $0 = 0$ pour tout $p$. Hors de ces cas, il faut renoncer à l'une des trois égalités.
9. L'éditeur de COMPAS montrait la **calibration** (à score égal, même taux de récidive : une parité prédictive) ; ProPublica montrait des **FPR** inégaux. Les taux de base de récidive mesurés différant entre les groupes, la question 6 dit qu'à PPV et FNR égaux, les FPR **devaient** différer : chacun avait raison selon son critère. Kleinberg et coll. montrent la même incompatibilité pour des scores (calibration contre équilibre des scores moyens parmi positifs et négatifs), sauf score parfait ou taux de base égaux.

**Erreurs fréquentes** : diviser $FP$ par tout le groupe au lieu des négatifs réels ; confondre FNR et FPR ; écrire la PPV comme $P(\hat{y} = 1 \mid y = 1)$ (c'est le TPR) ; conclure que le classifieur est « mauvais » alors que la relation vaut pour tout classifieur imparfait.
**Variante** : PPV = 0,75 et FNR = 0,25 imposés ; quel taux de base donne un FPR de 0,1 ? Réponse : $\frac{p}{1 - p} \times \frac13 \times 0{,}75 = 0{,}1$, donc $\frac{p}{1 - p} = 0{,}4$ et $p = \frac{0{,}4}{1{,}4} \approx 0{,}286$.

### Ex B6.7 — Fermi : coût de Shapley exact, de KernelSHAP et de TreeSHAP pour 10, 20 et 50 features 🧮
**Réponse** (fourchettes acceptées) : exact, par ligne : 51 200 prédictions et ≈ 0,3 s ($M = 10$), ≈ $5 \times 10^7$ et ≈ 6 min ($M = 20$), ≈ $6 \times 10^{16}$ et ≈ 10 000 ans ($M = 50$) ; seuil d'une minute à $M = 18$ ; KernelSHAP ≈ 0,3 à 0,7 s par ligne quel que soit $M$, mais il ne voit que 0,2 % des coalitions à $M = 20$ et $2 \times 10^{-12}$ à $M = 50$ ; TreeSHAP ≈ 1 à 3 ms par ligne ; les $12!$ ordres demandent ≈ 70 Go.
**Démarche** :
1. $2^{10} \times 50 = 51\,200$ prédictions, $51\,200 / 150\,000 \approx 0{,}34$ s ; 1 000 lignes ≈ 6 min. $2^{20} \times 50 \approx 5{,}2 \times 10^7$, ≈ 350 s ≈ 6 min par ligne, ≈ 4 jours pour 1 000 lignes. $2^{50} \times 50 \approx 5{,}6 \times 10^{16}$, ≈ $3{,}8 \times 10^{11}$ s ≈ 12 000 ans par ligne.
2. $2^M \times 50 / 150\,000 > 60$ demande $2^M > 180\,000$ : $M = 18$ ($2^{18} = 262\,144$, ≈ 87 s) ; à $M = 17$, ≈ 44 s.
3. $M = 10$ : $2M + 2\,048 = 2\,068$ dépasse $2^{10} - 2 = 1\,022$ ; shap prend alors **toutes** les coalitions (≈ 51 000 prédictions, ≈ 0,34 s), et la régression retombe sur les valeurs exactes : le même coût que le calcul exact, à 0,2 % près (il lui faut aussi $v(\varnothing)$ et $f(\mathbf{x})$). Autant prendre le calcul exact, plus simple et sans régression. $M = 20$ : 2 088 coalitions, ≈ 104 000 prédictions, ≈ 0,7 s, soit 0,2 % des $2^{20}$ coalitions. $M = 50$ : 2 148 coalitions, ≈ 107 000 prédictions, ≈ 0,7 s, une part de $2 \times 10^{-12}$. Le coût ne croît presque plus avec $M$, mais l'estimation est aléatoire : avec si peu de coalitions, les valeurs varient d'un tirage à l'autre ; on augmente `nsamples` et l'on vérifie la stabilité sur deux graines.
4. $100 \times 31 \times 50 = 155\,000$ pas, ≈ 1,5 ms à 10 ns le pas (la fiche a mesuré ≈ 3 ms) ; 1 000 lignes en 2 à 3 s. $M$ n'apparaît pas : le coût dépend de la taille des arbres et du fond.
5. $10! = 3\,628\,800$ ordres (≈ 0,5 Go à 140 octets) ; $12! = 479\,001\,600$ ordres, ≈ 67 Go : bien plus que les 12 Go d'une session Colab, la mémoire sature avant le moindre calcul. Énumérer les $2^{12} = 4\,096$ coalitions suffit, avec les poids de Shapley.
6. $20 \times 2^8 \times 50 = 256\,000$ prédictions, ≈ 2 s.
7. Calcul exact : un petit nombre de features (au plus une douzaine), pour vérifier ou quand il faut des valeurs exactes de n'importe quel modèle. KernelSHAP : n'importe quel modèle avec beaucoup de features, en acceptant une estimation bruitée dont on contrôle la stabilité. TreeSHAP : tout modèle d'arbres, exact et rapide, avec un fond pour le Shapley interventionnel.

**Erreurs fréquentes** : oublier le facteur $B$ ; compter $2^{M-1}$ au lieu de $2^M$ coalitions ; croire que KernelSHAP est toujours moins cher que le calcul exact (pas à $M = 10$) ; croire qu'il est exact quand il ne voit qu'une infime part des coalitions ; prendre la complexité de TreeSHAP pour une fonction de $M$.
**Variante** : KernelSHAP avec `nsamples=10_000` pour $M = 20$ et un fond de 100 lignes. Réponse : $10^4 \times 100 = 10^6$ prédictions, ≈ 7 s par ligne, pour ≈ 1 % des coalitions ; doubler le fond double le coût sans réduire la part de coalitions vues.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B6.8 — Expliquer une valeur SHAP à un client dont le prêt est refusé, en cinq lignes 🗣️
**Réponse modèle** : (1) Madame, votre demande a été refusée : notre modèle a estimé à 41 % le risque que le prêt ne soit pas remboursé, quand nous acceptons jusqu'à 30 %, et pour un dossier moyen il l'estime à 12 %. (2) Ce qui a le plus pesé, c'est votre taux d'endettement de 42 %, puis votre ancienneté de 4 mois dans votre emploi, et dans une moindre mesure les deux incidents de paiement de 2025 ; votre revenu a un peu joué en votre faveur. (3) Avec un endettement nettement plus bas, ou après quelques mois d'ancienneté de plus, le risque estimé aurait été plus faible ; nous ne pouvons pas vous promettre l'accord, mais ce sont les deux éléments qui comptent le plus. (4) Ces éléments décrivent le calcul d'un modèle à partir de dossiers passés, pas un jugement sur vous. (5) Vous pouvez demander qu'un conseiller réexamine votre dossier, présenter vos observations et contester la décision.
**Ce qui compte** : la décision et le score en mots simples, avec le seuil ; les deux ou trois contributions les plus fortes, dans le bon sens, sans chiffres SHAP bruts ; un contrefactuel prudent, qui ne promet rien ; la distinction entre le modèle et la personne ; les droits (réexamen humain, observations, contestation : RGPD art. 22 et AI Act art. 86).
**Erreurs fréquentes** : parler de « valeur de base » ou de « contribution SHAP » ; présenter une contribution comme une cause (« vous êtes refusée **parce que** vous êtes endettée », comme si le modèle avait raison) ; promettre l'accord à un seuil d'endettement qu'on n'a pas calculé ; oublier le droit au réexamen.
**Variante** : la même explication pour le conseiller qui reçoit Mme D. Réponse possible : « Score de 0,41 pour un seuil de 0,30 ; endettement +0,17 et ancienneté +0,09 dominent, les incidents ajoutent 0,06 ; le fond de référence est la clientèle de l'an dernier ; vérifie que le taux d'endettement est bien calculé sur les charges actuelles avant le réexamen. »

### Ex B6.9 — Scoring de crédit : le code postal est-il une variable de substitution ? Droits du client, obligations de l'entreprise ⚖️
**Réponse** :
1. **Variable de substitution, très probablement.** Le département porte des informations légitimes (le niveau des loyers, le marché de l'emploi), mais aussi une origine probable : l'écart de 9 points qui **reste** à profil financier comparable le suggère. On le mesure sans collecter l'origine : par des données agrégées publiques (la part de population immigrée par département, de l'Insee) croisées avec les taux d'acceptation ; en mesurant ce que le département ajoute à profil financier égal ; en testant si les autres features prédisent le département (et donc s'il pourrait être reconstitué). L'article 4 bis créé par le règlement 2026/1744 permet, à titre exceptionnel, de traiter des données sensibles quand c'est **strictement nécessaire** pour détecter et corriger des biais (art. 10, paragraphe 2, f) et g)), si des données synthétiques ou anonymisées ne suffisent pas, avec des limites de réutilisation, une sécurité à l'état de l'art et l'effacement une fois le biais corrigé ; il ne crée aucune obligation.
2. **Le droit de la non-discrimination.** L'article 225-1 du code pénal range le « lieu de résidence » parmi les critères de discrimination, et l'article 225-2 punit le refus de fournir un bien ou un service fondé sur l'un d'eux (trois ans d'emprisonnement et 45 000 € d'amende pour une personne physique). Ce délit suppose une intention : refuser sciemment des demandeurs à cause de leur département y expose la fintech. Sans intention, le droit civil s'applique aussi : la loi n° 2008-496 du 27 mai 2008 (art. 1 et 2) interdit, pour l'accès aux biens et services, la discrimination **indirecte**, un critère neutre en apparence (le département) qui désavantage particulièrement des personnes en raison de leur origine ou de leur lieu de résidence, sauf s'il est objectivement justifié par un but légitime et que les moyens sont nécessaires et appropriés ; l'écart de 9 points à profil comparable est exactement ce qu'il faudrait alors justifier. La CNIL (recommandation sur l'évaluation de la solvabilité dans le cadre de l'octroi de crédit, mai 2026) écrit que le département de résidence « peut être utilisé au détriment des personnes concernées » et ne peut être traité que si le responsable démontre qu'il est nécessaire à « une stratégie géographique particulière, propre à répondre aux besoins ou spécificités de certains territoires et tenant compte du type de crédit concerné », stratégie et raisons étant documentées.
3. **Le RGPD.** Un refus décidé par le modèle seul, qui prive la personne d'un crédit, relève de l'article 22. La fintech peut invoquer la nécessité pour conclure le contrat (art. 22.2.a), à démontrer, ou, pour le crédit à la consommation, l'article L. 312-16, VIII du code de la consommation à partir du 20 novembre 2026 (art. 22.2.b, l'exemple de la recommandation de la CNIL) ; elle doit garantir au moins le droit d'obtenir une intervention humaine, d'exprimer son point de vue et de contester (art. 22.3). Si elle achetait le score à une agence et s'en servait de manière déterminante, le calcul du score serait lui-même une décision automatisée (arrêt SCHUFA). À une demande d'accès (art. 15.1.h), elle doit expliquer de façon concise et compréhensible la procédure et les principes concrètement appliqués à **ses** données, ce qui n'est ni l'algorithme ni le détail de chaque étape ; la Cour admet que dire dans quelle mesure une variation des données aurait changé le résultat puisse suffire (arrêt Dun & Bradstreet, point 62). Si elle invoque le secret des affaires, elle transmet les informations à la CNIL ou au juge, qui arbitre : pas de refus en bloc.
4. **L'AI Act.** Un système à haut risque (annexe III, point 5 b)). La fintech est **à la fois** fournisseur (elle l'a développé) et déployeur (elle l'utilise). Obligations, à partir du 2 décembre 2027 : gestion des risques (art. 9), gouvernance des données qui examine les biais (art. 10), documentation technique (art. 11), journaux (art. 12), transparence envers les utilisateurs (art. 13), supervision humaine (art. 14), accuracy et robustesse (art. 15), évaluation de conformité ; côté déployeur, supervision et suivi (art. 26) et analyse d'impact sur les droits fondamentaux (art. 27). L'article 86 donne à la personne refusée le droit d'obtenir du déployeur des « explications claires et pertinentes sur le rôle du système d'IA dans la procédure décisionnelle et sur les principaux éléments de la décision prise ».
5. **Le plan technique.** Taux d'acceptation, TPR (parmi ceux qui remboursent) et FPR par groupe de départements, par âge et par sexe, et leurs croisements, avec effectifs et intervalles bootstrap ; puis retrait du département, réentraînement et réaudit. Les variables qui pourraient prendre le relais (situation familiale, type de contrat, banque d'origine) se testent en essayant de prédire le département ou le groupe de départements à partir d'elles. Une baisse de 0,007 d'AUC, à mettre en regard d'un écart de 9 points à profil comparable et d'un risque pénal, n'est pas un argument suffisant pour garder la variable ; il faut chiffrer aussi l'effet sur le taux de défaut et le documenter.
6. **Le message.** « Votre demande n'a pas été acceptée : notre outil a estimé un risque de non-remboursement supérieur à notre limite, surtout à cause de votre taux d'endettement et de votre ancienneté dans l'emploi. Avec un endettement plus bas, le résultat aurait pu être différent. Vous pouvez demander qu'un conseiller réexamine votre dossier et présenter vos observations : [lien]. »
7. **Fiche modèle, « Données et équité ».** Données : demandes 2023-2025, revenus et charges déclarés, ancienneté, incidents, âge, situation familiale ; le département a été retiré en [date] après un audit (écart de 9 points à profil égal entre groupes de départements). Équité : critère retenu, l'égalité des TPR entre groupes de départements, d'âge et de sexe, avec effectifs et intervalles en annexe. Limites : variables de substitution résiduelles possibles (testées : …) ; labels limités aux prêts accordés. Supervision : tout refus peut être réexaminé par un conseiller ; taux de désaccord suivi chaque mois. Réaudit trimestriel.

**Erreurs fréquentes** : conclure « pas de discrimination » parce que l'origine n'est pas une feature, ou parce que personne n'avait l'intention de discriminer (la discrimination indirecte n'en demande pas) ; croire que le RGPD interdit toute décision automatisée (il l'encadre) ; donner l'algorithme ou les poids du modèle comme « explication » ; oublier que la fintech est à la fois fournisseur et déployeur ; garder une variable risquée pour un gain minuscule d'AUC ; prendre ce résumé pour un avis juridique.
**Variante** : la fintech remplace le département par la distance du domicile à l'agence la plus proche. Réponse : c'est encore une information de lieu de résidence, probablement corrélée aux mêmes territoires ; elle doit passer le même test (ce qu'elle ajoute à profil financier égal, ce qu'elle prédit des groupes), et sa justification métier écrite ; un nouveau nom ne change pas la nature de la variable.

### Ex B6.10 — Reconnaissance faciale : erreurs par sous-groupe et décision de déploiement ⚖️
**Réponse** :
1. $0{,}213 \times 20{,}8 + 0{,}250 \times 6{,}0 + 0{,}233 \times 1{,}7 + 0{,}303 \times 0{,}0 \approx 6{,}3\,\%$, l'erreur totale du tableau 4. Avec 80 % de LM et 6,7 % de chacun des trois autres : $0{,}067 \times (20{,}8 + 6{,}0 + 1{,}7) \approx 1{,}9\,\%$, un système « presque parfait ». Avec 50 % de DF (les autres dans leurs proportions) : $0{,}5 \times 20{,}8 + 0{,}5 \times 2{,}4 \approx 11{,}6\,\%$. Les mêmes taux par sous-groupe donnent une moyenne de 2 à 12 % selon la population : l'erreur totale n'est pas une propriété du système seul.
2. Par sexe, 10,7 % contre 2,6 % ; par couleur de peau, 12,9 % contre 0,7 % : les écarts sont visibles, mais l'erreur la plus haute, 20,8 % pour les DF, n'apparaît qu'au croisement, presque deux fois plus que le pire chiffre d'une seule dimension. Le prix : des sous-groupes plus petits (environ 270 DF sur 1 270), donc des intervalles plus larges, et plus de comparaisons, donc plus d'écarts dus au hasard ; on choisit les croisements d'avance et l'on rapporte les effectifs.
3. $200\,000 \times 0{,}10 \times 4\,\% = 800$ clientes DF rejetées à tort par an, un taux 8 fois celui des LM. Un faux rejet coûte au **client** légitime (un parcours plus long, un sentiment d'exclusion, parfois l'abandon) ; une fausse acceptation coûte à la **banque** et à la personne dont l'identité est usurpée (fraude, blanchiment).
4. (a) Baisser le seuil pour tout le monde réduit les faux rejets de tous, mais augmente les fausses acceptations, sans corriger l'écart relatif. (b) Un seuil par sous-groupe peut égaliser les faux rejets, mais suppose d'inférer la couleur de peau et le sexe de chaque client, ce qui est une catégorisation biométrique et pose un problème juridique et éthique (traitement différent selon un attribut sensible). (c) Un rejet envoyé vers un conseiller sous 24 h ne supprime pas l'écart, mais en limite le coût ; c'est la supervision humaine, à condition que le conseiller ait le temps et l'autorité. (d) Exiger un modèle réentraîné et un rapport par sous-groupe, comme les éditeurs l'ont fait après Gender Shades (Raji et Buolamwini, 2019), traite la cause. (e) Une voie sans biométrie (vidéo avec un conseiller, envoi de pièces) est souvent nécessaire pour que le consentement à la biométrie soit libre.
5. La vérification 1:1, « dont la seule finalité est de confirmer qu'une personne physique spécifique est la personne qu'elle prétend être », est exclue du point 1 a) de l'annexe III : elle n'est pas à haut risque à ce titre (un autre point de l'annexe ou d'autres règles peuvent s'appliquer). Elle traite des données biométriques « aux fins d'identifier une personne physique de manière unique », une catégorie particulière de l'article 9 du RGPD : interdites sauf exception, ici en pratique le consentement explicite, qui suppose une alternative réelle ; une analyse d'impact sur la protection des données s'impose. Déduire du visage la « race » d'une personne (le mot du texte, avec les opinions politiques, l'affiliation syndicale, les convictions religieuses ou philosophiques, la vie et l'orientation sexuelles) est une pratique interdite (art. 5, paragraphe 1, g)). Le genre n'est pas dans cette liste : classer les clients par genre à partir du visage est une catégorisation biométrique selon un attribut protégé, à haut risque (annexe III, point 1 b)).
6. Recommandation (le NIST, en 2019, a trouvé des écarts entre groupes démographiques dans la majorité des algorithmes évalués, d'ampleur très variable : seuls des chiffres mesurés sur ce système et sur une population proche de la clientèle comptent) : déployer la vérification avec une voie alternative sans biométrie et un réexamen humain de tout rejet sous 24 h ; exiger du fournisseur, avant la signature, des taux de faux rejets par sous-groupe mesurés sur une population proche de la clientèle, avec intervalles, et un plafond d'écart contractuel ; suivre chaque mois les faux rejets par sous-groupe (estimés par les réexamens) et la part des clients qui abandonnent ; revoir la décision si l'écart dépasse le plafond.

**Erreurs fréquentes** : juger le système à son erreur totale ; croire qu'un écart n'existe que s'il apparaît dans une dimension seule ; proposer un seuil par sous-groupe sans voir qu'il faut alors inférer la couleur de peau ; confondre vérification 1:1 et identification 1:N ; croire que le consentement suffit s'il n'existe aucune alternative.
**Variante** : la police veut identifier en temps réel des personnes recherchées, dans une gare, à partir des caméras (une identification 1:N). Réponse : l'identification biométrique à distance « en temps réel » dans des espaces accessibles au public à des fins répressives est en principe interdite (art. 5, paragraphe 1, h)), sauf si elle est strictement nécessaire à l'un de trois objectifs : la recherche ciblée de victimes (enlèvement, traite, exploitation sexuelle) ou de personnes disparues ; la prévention d'une menace spécifique, substantielle et imminente pour la vie, ou d'une attaque terroriste ; la localisation d'un suspect d'une infraction grave de l'annexe II (punie d'au moins quatre ans). Il faut en outre une loi nationale qui le permette (art. 5, paragraphe 5) et, pour chaque utilisation, l'autorisation préalable d'un juge ou d'une autorité administrative indépendante, demandée au plus tard dans les 24 heures en cas d'urgence (art. 5, paragraphe 3) ; et les erreurs par sous-groupe y deviennent des contrôles injustifiés, concentrés sur certains groupes.

### Ex B6.11 — Lundberg et Lee (2017) : « A Unified Approach to Interpreting Model Predictions » 📄
**Réponse** :
1. **Le problème.** Les modèles les plus précis sur de grands datasets (ensembles, réseaux profonds) sont les plus difficiles à interpréter, et les méthodes d'explication proposées ne disent pas comment elles se relient ni laquelle préférer. Les deux composantes nouvelles : une **classe** de mesures d'importance, les explications additives ; et des résultats théoriques qui montrent qu'une **seule** solution de cette classe a les propriétés souhaitées.
2. **La définition.** $z' \in \{0, 1\}^M$ est une entrée simplifiée : 1 si la feature est présente (sa valeur vient de $\mathbf{x}$), 0 si elle est absente ; $h_x$ ramène $z'$ dans l'espace des entrées originales. Une coalition est un $z'$ : $\{1, 3\}$ s'écrit $z' = (1, 0, 1)$. $g(0, 0, 0) = \phi_0$, la valeur de base, 6 en ✏️ B6.1.
3. **Les six méthodes.** LIME (un super-pixel absent est remplacé par la moyenne des pixels voisins ; un mot absent compte 0), DeepLIFT (une valeur de référence choisie par l'utilisateur), la propagation de pertinence par couches (DeepLIFT avec une référence nulle), les valeurs de régression de Shapley (une feature absente est **retirée du modèle** : on réentraîne un modèle par sous-ensemble, soit $2^{10} = 1\,024$ modèles pour $M = 10$), les valeurs d'échantillonnage de Shapley et Quantitative Input Influence (l'absence est intégrée sur des exemples d'entraînement, par échantillonnage).
4. **Les trois propriétés.** Précision locale : la somme des attributions et de $\phi_0$ redonne la prédiction, c'est l'efficacité de la fiche. Absence : une feature absente de l'entrée simplifiée reçoit 0. Cohérence : si le modèle change de sorte que l'apport d'une feature augmente ou reste le même pour toute coalition, son attribution ne baisse pas. Le théorème 1 dit qu'une seule explication additive les vérifie toutes les trois : les valeurs de Shapley. Young (1985) a caractérisé les valeurs de Shapley par trois axiomes, voisins des propriétés 1 et 3 plus un troisième, redondant ici ; l'absence sert à adapter ces preuves aux explications additives, où une feature peut être « absente » de l'entrée simplifiée.
5. **Les valeurs SHAP.** De 10 à 11 : l'hypothèse d'**indépendance** des features (l'espérance des features absentes ne dépend plus de celles de $S$) ; de 11 à 12 : l'hypothèse de **linéarité** du modèle ($\mathbb{E}[f(z)] \approx f(\mathbb{E}[z])$). L'équation 11 est le $v(S)$ de la fiche et de ta `shapley_values`, avec le fond comme loi des features. En ✏️ B6.1, 11 donne la valeur de base 6, 12 donne $f(1, 2, 1) = 5$ ; pour un modèle linéaire (∂ B6.2, question 4), $v(\varnothing) = f(\bar{\mathbf{b}})$ : les deux coïncident.
6. **Kernel SHAP.** $\pi = \frac{M - 1}{\binom{M}{|z'|}\,|z'|\,(M - |z'|)}$ : pour $M = 4$, $\frac{3}{4 \times 1 \times 3} = 0{,}25$ ($|z'| = 1$), $\frac{3}{6 \times 2 \times 2} = 0{,}125$ ($|z'| = 2$), $0{,}25$ ($|z'| = 3$). Il est infini pour $|z'| = 0$ et $|z'| = M$ : ces deux coalitions imposent $\phi_0 = f_x(\varnothing)$ et la précision locale ; les auteurs éliminent analytiquement deux variables avec ces contraintes. La figure 2A range les $2^{15}$ vecteurs d'un exemple à $M = 15$ par cardinalité : le noyau est symétrique et grand aux extrémités.
7. **Linear SHAP.** $\phi_i(f, \mathbf{x}) = w_j(x_j - \mathbb{E}[x_j])$, la formule de ∂ B6.2 ; l'indice $i$ à gauche et $j$ à droite est une coquille. On peut en relever une seconde, sur le fond cette fois : avec $\phi_0 = b$, la somme $\phi_0 + \sum_i \phi_i$ vaut $f(\mathbf{x}) - \sum_j w_j\,\mathbb{E}[x_j]$ ; la précision locale n'est vérifiée que si $\sum_j w_j\,\mathbb{E}[x_j] = 0$ (par exemple des features centrées). En général, $\phi_0$ doit valoir $f_x(\varnothing) = b + \sum_j w_j\,\mathbb{E}[x_j] = f(\mathbb{E}[\mathbf{x}])$, la valeur de base de ∂ B6.2.
8. **Les jeux de la figure 4.** (A) $v(\varnothing) = 0$, $v(\{\text{fièvre}\}) = v(\{\text{toux}\}) = 5$, $v(\{\text{fièvre}, \text{toux}\}) = 2$ : $\phi_{\text{fièvre}} = \frac12\big[(5 - 0) + (2 - 5)\big] = 1$, et de même 1 pour la toux ; la congestion, absente du modèle, reçoit 0. (B) $v(\{1\}) = 5$, $v(\{2\}) = 4$, $v(\{1, 2\}) = 5$, et le troisième homme n'apporte jamais rien (joueur nul) : $\phi_1 = \frac12\big[5 + (5 - 4)\big] = 3$, $\phi_2 = \frac12\big[4 + 0\big] = 2$, $\phi_3 = 0$. La figure montre que la réponse la plus fréquente des participants (30 en A, 52 en B) coïncide avec SHAP ; LIME donne des attributions négatives à la fièvre et à la toux, et DeepLIFT dans sa version d'origine attribue les 5 $ au premier homme seul.
9. **Les expériences.** La figure 3 compare, sur 200 répétitions, Kernel SHAP, les valeurs d'échantillonnage de Shapley et LIME en fonction du nombre d'évaluations du modèle : Kernel SHAP s'approche plus vite de la vraie valeur de Shapley, et LIME converge vers une autre valeur. La figure 5 utilise 50 000 échantillons pour Kernel SHAP (et LIME), et masque 20 % des pixels pour faire passer la prédiction de 8 à 3.
10. **Et aujourd'hui.** Le choix entre l'espérance conditionnelle (équation 10) et le fond indépendant (11) est un débat ouvert (Chen et coll., 2020 ; Janzing et coll., 2019) : l'usage courant est le Shapley interventionnel, avec un fond qu'on écrit ; TreeSHAP (2018, 2020) calcule exactement les valeurs pour les arbres, ce que l'article ne propose pas encore ; la bibliothèque shap, communautaire, a changé d'API (`shap.Explainer`, `Explanation`, `shap.plots`) depuis le code de l'article.

**Erreurs fréquentes** : croire que SHAP est une méthode nouvelle de calcul plutôt qu'une **définition** unifiée, approchée de plusieurs façons ; confondre l'absence d'une feature (une coalition) et une valeur nulle de la feature ; lire l'équation 12 comme la définition de la valeur de base ; croire que le noyau de LIME redonne Shapley.
**Variante** : le jeu de la figure 4B si les deux premiers hommes avaient tous deux répondu juste à 5 questions. Réponse : $v(\{1\}) = v(\{2\}) = v(\{1, 2\}) = 5$, d'où $\phi_1 = \phi_2 = \frac12(5 + 0) = 2{,}5$ et $\phi_3 = 0$ : la symétrie partage le gain à parts égales.

<a id="entretien"></a>

## 💼 Entretien

### B6.E1 — Comment expliquer les prédictions d'un gradient boosting à une équipe métier ?
**Réponse modèle en 60 secondes** : « Je leur montrerais d'abord ce que le modèle utilise en général : une importance par permutation sur des données qu'il n'a pas vues, ou le beeswarm des valeurs SHAP, qui dit aussi dans quel sens joue chaque feature. Puis comment la prédiction varie avec deux ou trois features clés, par une dépendance partielle accompagnée des courbes ICE, pour repérer les interactions. Ensuite, des cas concrets qu'ils connaissent : un waterfall par client, en précisant le fond de référence. Pour la confiance, je comparerais à un modèle simple, une régression logistique, en montrant ce que le gradient boosting apporte, et je montrerais aussi des erreurs. Enfin, je dirais ce que l'explication ne dit pas : elle décrit le modèle, pas les causes du churn. »
**Ce qui compte** : global puis local ; les bons outils à chaque niveau ; le fond écrit ; la comparaison à un modèle simple ; montrer les erreurs ; la limite causale.
**Relances possibles** : « Et si deux features sont très corrélées ? » (leur partage dépend du modèle, je rapporte la somme du groupe et je le dis) · « Pourquoi pas simplement `feature_importances_` ? » (calculée sur l'entraînement, biaisée vers les features à beaucoup de valeurs ; la permutation sur la validation la corrige) · « Ils veulent savoir quoi faire pour retenir un client. » (une explication ne dit pas l'effet d'une action ; il faut une expérience, un test A/B)

### B6.E2 — Qu'est-ce qu'une valeur SHAP, et quelles sont ses limites ?
**Réponse modèle en 60 secondes** : « Une valeur SHAP dit de combien une feature a déplacé la prédiction d'un exemple par rapport à la prédiction moyenne d'un fond de référence. Elle vient de la théorie des jeux : on fait la moyenne de l'apport de la feature sur tous les ordres d'arrivée des features, et c'est la seule répartition qui vérifie l'efficacité, la symétrie, le joueur nul et l'additivité ; la somme des valeurs redonne la prédiction moins la valeur de base. En pratique, TreeSHAP les calcule exactement pour les arbres, KernelSHAP les estime pour n'importe quel modèle. Je m'en méfie dans trois cas : quand le fond n'est pas dit, puisqu'il change les valeurs ; avec des features corrélées, où le partage dépend du modèle et où l'on évalue des combinaisons irréalistes ; et quand on les lit comme des causes. »
**Ce qui compte** : la définition par rapport à un fond ; l'origine et l'unicité ; l'efficacité ; les algorithmes ; trois limites concrètes.
**Relances possibles** : « Quelle différence entre interventionnel et observationnel ? » (on impose les valeurs sans regarder leur plausibilité, fidèle au modèle ; ou on les tire sachant les autres, fidèle aux données) · « Combien ça coûte ? » ($2^M \times B$ prédictions en exact ; TreeSHAP est linéaire dans la taille des arbres) · « Une valeur SHAP peut-elle être manipulée ? » (oui, Slack et coll. : un modèle qui détecte les points artificiels et y répond de façon anodine)

### B6.E3 — Comment audites-tu un modèle pour détecter des biais ?
**Réponse modèle en 60 secondes** : « Avant de calculer quoi que ce soit, je regarde qui est concerné, qui a produit les labels et quelles erreurs coûtent le plus, et à qui : écarter un bon candidat n'a pas le même coût que convoquer un candidat moyen. Je choisis les attributs à auditer et leurs croisements, puis je calcule par groupe le taux de sélection, le TPR, le FPR et la precision, avec les effectifs et des intervalles bootstrap, et les écarts entre groupes avec leurs propres intervalles. Je cherche les variables de substitution en essayant de prédire l'attribut sensible à partir des autres features. Je choisis un critère et je le justifie par écrit, en sachant qu'on ne peut pas tout égaliser quand les taux de base diffèrent. Puis je documente tout dans la fiche modèle et je prévois de réauditer en production. »
**Ce qui compte** : le cadrage avant les chiffres ; les taux par groupe et par croisement ; les intervalles ; les variables de substitution ; le choix argumenté du critère ; la documentation et le suivi.
**Relances possibles** : « Et si vous n'avez pas l'attribut sensible ? » (données agrégées, enquête volontaire, ou l'exception de l'article 4 bis de l'AI Act pour détecter les biais, encadrée) · « L'accuracy est la même dans tous les groupes, c'est bon ? » (non : à accuracy égale, les TPR et les taux de sélection peuvent être très différents) · « Que faites-vous si vous trouvez un écart ? » (comprendre sa source, corriger avant, pendant ou après l'entraînement, chiffrer le coût, réauditer tous les groupes)

### B6.E4 — Grad-CAM : principe, usage et limites
**Réponse modèle en 60 secondes** : « Je lui proposerais une carte Grad-CAM : on prend les cartes d'activation de la dernière couche convolutive, on pondère chaque canal par la moyenne des gradients du score de la classe, on garde le positif et on agrandit la carte à la taille de l'image. Elle montre les régions qui ont augmenté le score. Ce qu'on cherche d'abord, c'est une fuite : un modèle qui regarde un marqueur d'hôpital ou un bord de cliché plutôt que la lésion. Mais je préviendrais des limites : la carte est grossière, elle dépend de la couche et de la classe choisies, et une carte convaincante n'est pas une preuve. Avant de m'y fier, je ferais un test de randomisation des poids, je la comparerais à une autre méthode, et je ferais relire des cas variés, y compris des erreurs, par des radiologues. »
**Ce qui compte** : le calcul en une phrase ; l'usage pour détecter une fuite ; la résolution, la couche et la classe ; le test de santé ; la validation par des experts.
**Relances possibles** : « Pourquoi pas la première couche, plus fine ? » (elle suit les bords et les textures, pas la classe ; la dernière couche convolutive est le compromis entre sens et localisation) · « Comment l'implémentez-vous ? » (deux hooks PyTorch, l'un garde la sortie de la couche, l'autre son gradient ; puis moyenne, somme pondérée, ReLU, interpolation) · « Et pour un ViT ? » (il n'y a pas de cartes convolutives ; on utilise les gradients intégrés, ou des variantes adaptées aux tokens d'image)

### B6.E5 — Retirer la variable sensible suffit-il à rendre un modèle équitable ?
**Réponse modèle en 60 secondes** : « Non, pour trois raisons. D'abord, d'autres colonnes portent souvent la même information : un code postal, un prénom, un type de contrat ; le modèle la retrouve. Ensuite, une partie des écarts vient des labels et des taux de base, que le retrait ne change pas. Enfin, sans l'attribut, on ne peut plus ni mesurer ni corriger. Ce que je ferais : garder le sexe et l'âge hors du modèle mais dans les données d'audit, mesurer si les autres features les prédisent, comparer les taux par groupe avant et après le retrait, choisir un critère et, si besoin, corriger par repondération ou par seuil, en chiffrant le coût. »
**Ce qui compte** : les variables de substitution ; les écarts qui viennent des données ; l'impossibilité d'auditer sans l'attribut ; une démarche de mesure.
**Relances possibles** : « Comment savoir si une feature est une variable de substitution ? » (essayer de prédire l'attribut à partir d'elle, et mesurer ce qu'elle ajoute à profil égal) · « Utiliser l'attribut pour corriger, c'est légal ? » (l'utiliser dans la décision peut être une discrimination directe ; l'utiliser pour mesurer et corriger demande un cadre, comme l'article 4 bis de l'AI Act pour les données sensibles)

<a id="notebook"></a>

## Exercices du notebook

### Ex B6.12 — Explications globales sur California : coefficients, permutation, dépendance partielle et ICE 📦
**Réponse** : coefficients standardisés : `MedInc` 0,86, `Latitude` −0,87, `Longitude` −0,84, `AveBedrms` 0,28, `AveRooms` −0,26, `HouseAge` 0,12, `AveOccup` −0,04, `Population` −0,01 (R² de validation de la régression : 0,584) ; importances par permutation du gradient boosting (R² 0,825) sur la validation : `Latitude` 0,727, `Longitude` 0,624, `MedInc` 0,538, `AveOccup` 0,157, `AveRooms` 0,049, `HouseAge` 0,039, `Population` et `AveBedrms` 0,006. La dépendance partielle de `AveOccup` descend de 2,67 à 1,74 entre 1,9 et 4,4 occupants par logement, presque toute la baisse avant 3 occupants (1,94 à 3) ; celle de `HouseAge` ne bouge que de 0,32 sur toute sa grille.
**Démarche** :
```python
def linear_12(X, y):
    return make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)


def coefs_12(pipeline):
    return pd.Series(pipeline[-1].coef_, index=pipeline.feature_names_in_)


def boosting_12(X, y):
    return HistGradientBoostingRegressor(random_state=0).fit(X, y)


def permutation_12(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=5, random_state=0)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def pd_ice_12(model, X, feature):
    result = partial_dependence(model, X, [feature], kind="both", grid_resolution=50)
    return {"grid": result["grid_values"][0], "average": result["average"][0], "individual": result["individual"][0]}
```
Les trois explications ne répondent pas à la même question. Les coefficients décrivent la **régression**, un modèle qui n'explique que 58 % de la variance : ils disent ce que vaut un écart-type de chaque feature, les autres fixées, dans un monde linéaire. `AveBedrms` et `AveRooms` y ont des coefficients de signes opposés, deux features corrélées que la régression oppose ; le gradient boosting, lui, s'en sert à peine. La permutation décrit le **gradient boosting**, sur des districts qu'il n'a pas vus : la position (latitude et longitude, qui ne servent qu'ensemble) passe devant le revenu, parce que les arbres découpent la carte en zones, ce qu'une droite ne sait pas faire. Les courbes ICE de `AveOccup` ont presque toutes la même forme (une baisse, puis un plateau), à des hauteurs différentes : la dépendance partielle les résume bien pour la forme, pas pour le niveau. Au-delà du 95ᵉ centile (4,4 occupants), peu de districts : la grille s'arrête là, et une courbe prolongée n'y dirait que ce que les arbres ont appris ailleurs.
**Erreurs fréquentes** : comparer des coefficients de features brutes (celui de `Population` vaut $-10^{-5}$ par habitant, celui de `AveBedrms` 0,82 par chambre : l'unité décide) ; calculer la permutation sur l'entraînement ; oublier `random_state=0` dans `permutation_importance`, qui change alors à chaque exécution ; lire une dépendance partielle comme un effet causal (« ajouter un occupant fait baisser le prix »).
**Variante** : la même permutation sur les districts d'**entraînement** donne `Latitude` 0,770, `Longitude` 0,636, `MedInc` 0,585, `AveOccup` 0,170, `AveRooms` 0,062, `HouseAge` 0,051, `Population` 0,016, `AveBedrms` 0,014 : toutes plus grandes, parce que le modèle connaît ces districts (il y perd plus quand on les brouille), et `Population` et `AveBedrms` changent de rang : la différence entre entraînement et validation mesure ce que le modèle a appris par cœur.

### Ex B6.13 — Prédire la forme des courbes ICE du revenu médian et de la latitude 🔮
**Réponse** : a) **`"C"`** : la dépendance partielle de `MedInc` monte de 1,94 (de 1,38 à 3,32) ; b) **`"C"`** : celle de `Latitude` descend de 1,82, de 3,20 au sud à 1,38 au nord ; c) **`"A"`** : aucune des 500 courbes ICE de la latitude ne finit plus haut qu'elle ne commence ; d) **`True`** : l'écart-type de la variation des courbes ICE vaut 0,60 pour la latitude, 0,45 pour le revenu. Sur cinq découpages de California et deux graines du modèle (`tools/chapters/history_chB6.py`), les quatre réponses ne changent jamais : la montée du revenu va de 1,80 à 2,03, la descente de la latitude de 1,54 à 1,82, aucune courbe ICE de la latitude ne monte, et le rapport des deux écarts-types va de 1,3 à 2,0.
**Démarche** : toutes les courbes ICE du revenu montent, à peu près parallèles : un district plus riche vaut plus cher, quel que soit l'endroit. La latitude surprend : on attend souvent une bosse vers la baie de San Francisco (37,5° à 38°), et la courbe moyenne y est presque plate, puis descend encore vers le nord. C'est l'effet de la longitude **fixée** : déplacer un district de Los Angeles (34°, −118°) vers le nord sans changer sa longitude le fait entrer dans les terres (la vallée centrale, la Sierra), où les logements sont moins chers ; la baie est plus à l'ouest. Les courbes ICE de la latitude descendent toutes, mais pas de la même hauteur ni au même endroit : elles portent l'interaction avec la longitude, que la dépendance partielle moyenne efface. Une courbe ICE reste une question posée au modèle (« que prédirais-tu si ce district était plus au nord ? »), souvent sur des combinaisons qui n'existent pas, et non l'effet d'un déménagement.
**Erreurs fréquentes** : lire la latitude comme une variable indépendante de la longitude (la carte n'est pas un axe) ; attendre une bosse pour San Francisco sans fixer la longitude ; confondre la variation moyenne (la dépendance partielle) et sa dispersion d'un district à l'autre (les ICE).
**Variante** : pour la longitude (d'ouest en est), la dépendance partielle descend de 1,55 ; aucune courbe ICE ne monte ; l'écart-type de leur variation vaut 0,49 : la même lecture, la côte contre l'intérieur.

### Ex B6.14 — shapley_values : Shapley exact par énumération des coalitions 🔨
**Réponse** : les 33 tests de l'exercice passent ; sur California, 8 features et un fond de 50 lignes, la fonction rend les valeurs de `shap.explainers.Exact` en 0,06 s, avec un seul appel à `f` sur les 12 800 lignes.
**Démarche** :
```python
def shapley_values(f, x, background, max_features=12):
    x = np.array(x, dtype=float)
    bg = np.array(background, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if bg.ndim != 2 or bg.shape[1] != len(x):
        raise ValueError(f"background must be 2-D with {len(x)} columns, got shape {bg.shape}")
    M, B = len(x), len(bg)
    if M > max_features:
        raise ValueError(f"{M} features > max_features = {max_features}")
    masks = list(itertools.product((False, True), repeat=M))
    blocks = []
    for mask in masks:
        rows = bg.copy()
        rows[:, list(mask)] = x[list(mask)]
        blocks.append(rows)
    out = np.asarray(f(np.concatenate(blocks)), dtype=float).reshape(len(masks), B)
    values = dict(zip(masks, out.mean(axis=1)))              # v(S): f first, then the mean over the background
    phi = np.zeros(M)
    for mask, value in values.items():
        size = sum(mask)
        for i in range(M):
            if not mask[i]:
                weight = math.factorial(size) * math.factorial(M - size - 1) / math.factorial(M)
                phi[i] += weight * (values[mask[:i] + (True,) + mask[i + 1:]] - value)
    return phi, float(values[(False,) * M])
```
Chaque coalition est un masque booléen ; son $v(S)$ remplace les colonnes de $S$ dans une copie du fond et fait la moyenne de $f$ sur les $B$ lignes, **une** fois. Les $2^M$ blocs de $B$ lignes sont ici concaténés pour un seul appel à `f` : un `predict` de 12 800 lignes coûte à peine plus qu'un `predict` de 50, et bien moins que 256 appels. Le poids $\frac{|S|!\,(M - |S| - 1)!}{M!}$ compte les ordres d'arrivée où $i$ suit exactement $S$ ; leur somme sur les coalitions sans $i$ vaut 1. Le modèle linéaire à 5 features vérifie $\phi_i = w_i(x_i - \bar b_i)$ (∂ B6.2), l'efficacité tient à $10^{-12}$ près, et `shap.explainers.Exact`, avec `max_samples` égal à la taille du fond, rend les mêmes nombres.
**Erreurs fréquentes** : moyenner le fond **avant** `f` (on explique alors $f$ en une ligne moyenne, et non la moyenne de $f$ : faux dès que le modèle n'est pas linéaire) ; prendre `f` sur la ligne moyenne pour valeur de base ; énumérer les $M!$ ordres sans garder les $v(S)$ (les tests l'arrêtent) ; appeler `f` sur une ligne 1-D ; oublier la coalition vide ou la pleine ; modifier le fond en place (`bg[:, mask] = ...` sans copie).
**Variante** : la définition par les ordres, $\phi_i = \frac{1}{M!}\sum_{\text{ordres}} [v(S_{\prec i} \cup \{i\}) - v(S_{\prec i})]$, avec un dictionnaire des $v(S)$ : les mêmes valeurs à $10^{-13}$ près, en 0,5 s au lieu de 0,06 s pour $M = 8$ (40 320 ordres de 8 contributions chacun) ; à $M = 12$, 479 millions d'ordres : la formule par coalitions n'en lit que 4 096 valeurs.

### Ex B6.15 — SHAP sur un gradient boosting California : TreeExplainer, waterfall, beeswarm, dépendance 📦
**Réponse** : a) valeur de base **1,9641** (la prédiction moyenne du modèle sur les 50 lignes du fond) ; b) valeur SHAP de `MedInc` pour le premier district **−0,0743** ; c) moyenne des $\lvert\phi_{\text{MedInc}}\rvert$ sur les 1 000 districts **0,5050** ; d) **`True`** : ta `shapley_values` redonne TreeSHAP à $6 \cdot 10^{-9}$ près sur le premier district, et l'efficacité tient à $3 \cdot 10^{-8}$ près sur les 1 000. TreeSHAP explique les 1 000 districts en 0,7 s environ ; ta fonction, à 0,045 s par district, en prendrait 45.
**Démarche** :
```python
def fit_15(X, y):
    return HistGradientBoostingRegressor(max_depth=3, random_state=0).fit(X, y)


def tree_shap_15(model, background, rows):
    return shap.TreeExplainer(model, data=background, feature_perturbation="interventional")(rows)
```
Avec un fond, TreeSHAP interventionnel calcule exactement le $v(S)$ de 206.3 en parcourant les arbres, pour un coût en $O(TLR)$ ($T$ arbres de $L$ feuilles, $R$ lignes de fond) au lieu de $2^M \times B$ prédictions. La valeur de base est la prédiction moyenne **sur le fond** : ni la moyenne des cibles (2,0659), ni la prédiction moyenne des districts expliqués (2,0735). Le premier district (Berkeley : 37,85°, −122,26°, des logements de 52 ans, un revenu médian de 3,27) est prédit 2,70 : son occupation moyenne basse (+0,66), sa longitude (+0,48) et l'âge du bâti (+0,18) le font monter, sa latitude (−0,48) le fait descendre, et un revenu un peu inférieur à celui du fond (3,27 contre 3,50 en moyenne) retire 0,07. Sur le beeswarm, les features se rangent par moyenne des $\lvert\phi\rvert$ : `MedInc` 0,505, `Latitude` 0,486, `Longitude` 0,402, `AveOccup` 0,219, puis moins de 0,07 ; un grand revenu (rouge) pousse vers le haut. Le graphique de dépendance de `MedInc` monte presque en ligne droite jusqu'à un revenu de 8 environ, puis plafonne (la cible l'est, à 5) ; à revenu égal, la dispersion verticale est faible, et shap colore les points par `AveOccup`, la feature qui interagit le plus avec le revenu : les courbes ICE de B6.13, presque parallèles, le disaient autrement.
**Erreurs fréquentes** : appeler `TreeExplainer(model)` sans fond (le mode « chemins de l'arbre » rend une valeur de base de 2,0664 et $\phi_{\text{MedInc}} = -0{,}172$ pour le premier district : une autre question) ; garder le modèle par défaut au lieu de `max_depth=3` ; rendre `explainer.shap_values(rows)`, un tableau sans valeur de base ; oublier que shap ne garde que 100 lignes d'un fond plus grand.
**Variante** : sur les données d'origine en float64 (sans l'arrondi de `wbx.california`), le même calcul rend des valeurs SHAP dont la somme s'écarte de la prédiction pour presque tous les districts (jusqu'à 0,19 sur les 200 premiers et 0,53 sur les 1 000 de l'exercice avec `max_depth=3` ; 0,23 et 0,49 avec le gradient boosting par défaut) et shap lève une erreur d'additivité : TreeSHAP parcourt les arbres en float32, et des seuils posés exactement sur des valeurs répétées de la latitude et de la longitude changent de côté.

### Ex B6.16 — Le fond change tout : trois fonds différents, puis une feature dupliquée 🔬
**Réponse** : le district expliqué (revenu 4,78, près de Long Beach : 33,74°, −118,10°) est prédit 3,80. Fond « population » : valeur de base 1,82, $\phi_{\text{MedInc}} = +0{,}64$ ; fond « riches » : 3,92 et **−0,67** ; fond « ligne type » : 2,16 et +0,56. Avec cinq tirages des fonds, $\phi_{\text{MedInc}}$ reste entre +0,49 et +0,64 contre la population, entre −0,70 et −0,66 contre les riches. La copie : exacte (bruit 0), elle reçoit 0 sur toutes les graines ; avec un bruit de 0,01 écart-type, la part de `MedInc` dans la somme va de 0,23 à 0,74 selon la graine (dix graines), mais la somme des deux reste entre 0,363 et 0,388, contre 0,390 sans copie ; avec un bruit de 0,1, la copie, moins fiable, ne reçoit que 6 à 19 %.
**Démarche** :
```python
def explain_16(model, x, background):
    explanation = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")(x.to_frame().T)
    return explanation.values[0], float(np.ravel(explanation.base_values)[0])
```
(et `copy_experiment_16` comme son contrat : le même `rng`, le modèle sur `A`, TreeSHAP sur `V`, les moyennes de valeurs absolues). Un revenu de 4,78 est au-dessus de la moyenne de la population, au-dessous de celle des districts riches : la même feature, du même district, pour le même modèle, pousse vers le haut ou vers le bas selon la question posée. La valeur de base suit le fond, et l'efficacité répartit toujours $f(\mathbf{x})$ moins elle : contre les riches, il y a moins à expliquer, et le revenu en « retire ». La ligne type répond à « pourquoi ce district plutôt qu'un district médian ? », une question plus simple à dire, mais d'une seule ligne : les interactions y sont évaluées en un seul point. Pour la copie, les arbres de `HistGradientBoosting` départagent deux colonnes au gain égal par leur ordre : la copie exacte, placée après `MedInc`, n'est jamais choisie, et le Shapley interventionnel, fidèle au modèle, lui donne 0. Avec un bruit minuscule, chaque nœud choisit l'une ou l'autre selon le hasard des valeurs : le partage change d'une graine à l'autre, la somme non (🧠 Q6). Avec un bruit plus grand, la copie est moins informative et le modèle revient à `MedInc`.
**Erreurs fréquentes** : comparer des valeurs SHAP calculées avec des fonds différents sans le dire ; conclure d'un $\phi = 0$ qu'une feature est inutile dans les données (la copie exacte porte la même information) ; rapporter l'importance de l'une des deux copies au lieu de leur somme ; croire que le partage est une propriété des données, alors qu'il dépend de l'entraînement.
**Variante** : avec pour fond 50 districts du sud de la Californie (latitude inférieure à 34,5°, tirés avec `random_state=0`), la valeur de base vaut 2,17, $\phi_{\text{MedInc}} = +0{,}46$ et $\phi_{\text{Latitude}}$ tombe de +0,55 (population) à +0,12 : contre des districts du sud, la latitude de Long Beach n'a presque plus rien de particulier.

### Ex B6.17 — Penguins : SHAP par classe pour une régression logistique et une forêt aléatoire 📦
**Réponse** : le premier Gentoo du test (bec de 45,1 mm de long et 14,5 mm de haut, nageoire de 207 mm, 5 050 g) est classé Gentoo à 0,990 par la régression logistique et à 0,80 par la forêt (accuracies de test 98,8 % et 97,6 %). Logit de la classe Gentoo : valeur de base −0,42 ; hauteur du bec **+2,25**, masse +1,16, nageoire +0,47, longueur du bec +0,08. Probabilité Gentoo (régression logistique) : base 0,357 ; hauteur du bec +0,332, masse +0,151, longueur du bec +0,078, nageoire +0,071. Forêt : base 0,357 ; +0,239, +0,109, +0,069, +0,027. Les sommes sur les trois classes valent 0 pour chaque mesure, en probabilité comme en logit.
**Démarche** :
```python
def class_output_17(model, k, kind="proba"):
    if kind == "proba":
        return lambda X: model.predict_proba(X)[:, k]
    if kind == "logit":
        return lambda X: model.decision_function(X)[:, k]
    raise ValueError(f"kind must be 'proba' or 'logit', got {kind!r}")


def shap_by_class_17(model, x, background, kind="proba"):
    results = [mylearn.explain.shapley_values(class_output_17(model, k, kind), x, background)
               for k in range(len(model.classes_))]
    return np.array([phi for phi, _ in results]), np.array([base for _, base in results])
```
(`fit_logistic_17` et `fit_forest_17` sont un `make_pipeline(...).fit(X, y)` et un `RandomForestClassifier(...).fit(X, y)`.) Le logit d'une régression logistique est linéaire dans les mesures standardisées : ses valeurs de Shapley sont $w_{ki}(x_i - \bar b_i)$ (∂ B6.2), et sa valeur de base, l'ordonnée à l'origine de la classe (−0,42), parce que le fond est l'entraînement sur lequel le scaler a appris (ses mesures standardisées sont de moyenne nulle). La probabilité passe par la softmax : ses valeurs ne sont plus proportionnelles à celles du logit, et l'ordre change (la longueur du bec passe devant la nageoire). La somme des trois probabilités vaut 1 pour tout manchot, une fonction constante dont chaque feature est un joueur nul : par additivité, les trois explications d'une même mesure s'annulent, ce qu'elle apporte à Gentoo est retiré à Adelie et à Chinstrap. Pour les logits, la somme est nulle aussi, mais pour une autre raison : la pénalité L2 de scikit-learn centre les coefficients sur les classes ($\sum_k w_{ki} = 0$). La forêt s'appuie sur les mêmes mesures, dans le même ordre, avec des parts plus modestes : elle est moins sûre d'elle (0,80).
**Erreurs fréquentes** : expliquer `predict` (une classe, pas un nombre) ; mélanger les classes (l'ordre est celui de `model.classes_`) ; comparer des valeurs de logit et de probabilité comme si elles avaient la même unité ; laisser `TreeExplainer` sous-échantillonner le fond à 100 lignes (un écart de 0,05 avec ta fonction).
**Variante** : la sortie Adelie (logit) du même manchot : longueur du bec −0,51, hauteur −1,71, nageoire −0,21, masse +0,30 : son bec bas l'éloigne d'Adelie ; sa masse, à autres mesures fixées, l'en rapproche un peu (la même masse retire 1,46 au logit de Chinstrap), et la valeur de base vaut l'ordonnée à l'origine d'Adelie, +0,49.

### Ex B6.18 — integrated_gradients et grad_cam 🔨
**Réponse** : les 111 tests de l'exercice passent (99, et 12 ignorés sans captum) ; sur $x_0x_1 + x_2^3$ en $(1, 2, -1)$ avec 6 intervalles, les quatre règles donnent $(0{,}833 ; 0{,}833 ; -0{,}764)$ à gauche, $(1{,}167 ; 1{,}167 ; -1{,}264)$ à droite, $(1 ; 1 ; -0{,}993)$ au milieu et $(1 ; 1 ; -1{,}014)$ en trapèzes, pour des valeurs exactes $(1, 1, -1)$ ; la complétude sur la sigmoïde de 206.5 tient à $10^{-5}$ près avec 200 intervalles.
**Démarche** :
```python
METHODS = ("left", "right", "midpoint", "trapezoid")


def integrated_gradients(grad_fn, x, baseline=None, n_steps=50, method="trapezoid"):
    x = np.asarray(x, dtype=float)
    ref = np.zeros_like(x) if baseline is None else np.asarray(baseline, dtype=float)
    if ref.shape != x.shape:
        raise ValueError(f"baseline must have the shape of x {x.shape}, got {ref.shape}")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps!r}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    n = int(n_steps)
    if method == "trapezoid":
        alphas = np.arange(n + 1) / n
        weights = np.full(n + 1, 1.0 / n)
        weights[[0, -1]] /= 2
    else:
        k = np.arange(n)
        alphas = {"left": k / n, "right": (k + 1) / n, "midpoint": (k + 0.5) / n}[method]
        weights = np.full(n, 1.0 / n)
    points = ref[None] + alphas.reshape((-1,) + (1,) * x.ndim) * (x - ref)[None]
    grads = np.asarray(grad_fn(points), dtype=float)
    return (x - ref) * np.tensordot(weights, grads, axes=1)


def grad_cam(activations, gradients, relu=True, normalize=True):
    A, G = np.asarray(activations, dtype=float), np.asarray(gradients, dtype=float)
    if A.ndim != 4 or A.shape != G.shape:
        raise ValueError(f"activations and gradients must be 4-D with the same shape, got {A.shape} and {G.shape}")
    cam = np.einsum("nc,nchw->nhw", G.mean(axis=(2, 3)), A)
    if relu:
        cam = np.maximum(cam, 0.0)
    if normalize:
        peak = np.abs(cam).max(axis=(1, 2), keepdims=True)
        cam = np.divide(cam, peak, out=np.zeros_like(cam), where=peak > 0)
    return cam
```
Le chemin, les poids et **un** appel à `grad_fn` sur tout le batch de points : la règle n'est qu'un choix d'$\alpha$ et de poids. Sur $x_0x_1 + x_2^3$, le gradient le long du chemin vaut $(\alpha x_1, \alpha x_0, 3\alpha^2 x_2^2)$ : linéaire en $\alpha$ pour les deux premières features, que le milieu et les trapèzes intègrent exactement ; quadratique pour la troisième, où le trapèze surestime de $1/(6n^2)$ et le milieu sous-estime de $1/(12n^2)$, d'où −1,014 et −0,993. La complétude est un test d'implémentation : une erreur d'indice, un $\mathbf{x} - \mathbf{x}'$ oublié ou des poids qui ne somment pas à 1 la cassent. Grad-CAM : des poids par canal (la moyenne spatiale), une somme pondérée, une seule ReLU, puis chaque carte divisée par son propre maximum ; la fonction ne voit que des tableaux, ce qui la rend testable sans réseau, et laisse l'extraction (les hooks) à PyTorch.
**Erreurs fréquentes** : appeler `grad_fn` sur un point 1-D ; compter `n_steps` en points ; oublier de diviser les extrémités du trapèze par deux ; diffuser une ligne de référence de mauvaise forme ; sommer au lieu de moyenner les gradients ; appliquer la ReLU par canal avant la somme ; normaliser par le maximum de tout le batch ; diviser par 0 une carte nulle (NaN).
**Variante** : le `riemann_trapezoid` de captum, à 7 points (son `n_steps` compte les points), rend sur le même exemple $(0{,}857 ; 0{,}857 ; -0{,}869)$, ta règle des trapèzes multipliée par $6/7$ : ses poids somment à $6/7$ (fiche, 🕰️ de 206.6).

### Ex B6.19 — Saillance, gradient × entrée et gradients intégrés sur un CNN CIFAR-10 📦
**Réponse** : sur les 6 premières images (le CNN de référence, 65,6 % d'accuracy de test, en prédit 4 justes), la somme des gradients intégrés depuis l'image noire, avec 50 intervalles, colle à $f(\mathbf{x}) - f(\mathbf{x}')$ à 4,3 % près au pire (0,970 contre 1,014), à 1,3 % près ailleurs ; depuis l'image floutée, à 1,8 % près. La somme de gradient × entrée n'a rien à voir avec l'écart (−0,42 contre 3,96 pour la première grenouille, −3,13 contre 0,96 pour la deuxième). Avec 200 intervalles, l'écart de la complétude tombe à 0,7 % au pire sur 16 images.
**Démarche** :
```python
def class_gradient_19(model, images, target):
    x = torch.as_tensor(np.asarray(images), dtype=torch.float32).clone().requires_grad_(True)
    t = torch.as_tensor(np.asarray(target)).long().reshape(-1)
    model(x).gather(1, t[:, None]).sum().backward()      # each logit depends on its own image only
    return x.grad.numpy()


def saliency_19(gradients):
    return np.abs(np.asarray(gradients)).max(axis=1)


def grad_times_input_19(gradients, images):
    return (np.asarray(gradients) * np.asarray(images)).sum(axis=1)


def integrated_19(model, image, target, baseline, n_steps=50):
    grad_fn = lambda points: class_gradient_19(model, points, [target] * len(points))
    return mylearn.explain.integrated_gradients(grad_fn, np.asarray(image, dtype=float),
                                                np.asarray(baseline, dtype=float), n_steps=n_steps)
```
Le gradient se prend par rapport à l'entrée, sur le **logit** de la classe demandée : la probabilité sature (son gradient s'éteint quand le réseau est sûr de lui) et mélange les classes. La saillance est la plus bruitée : elle dit où une petite retouche changerait le score, pas ce qui a compté, et s'allume aussi sur le fond. Gradient × entrée et les gradients intégrés depuis le noir s'éteignent là où l'image est sombre (le facteur $x_i - x'_i$ y est petit) et se concentrent davantage sur l'objet. Un CNN à ReLU est linéaire par morceaux : le long du chemin, ses régions changent, et 50 intervalles suffisent ici pour approcher l'intégrale à quelques % près ; gradient × entrée, une seule pente en $\mathbf{x}$, ne vérifie aucune complétude. La référence floutée garde les couleurs et les grandes formes : $f(\mathbf{x}')$ y est souvent déjà élevé (le chien : 1,32 seulement à expliquer, contre 3,74 depuis le noir), et seuls les détails fins reçoivent une attribution. Pour un objet noir sur fond clair, l'image noire attribuerait 0 à l'objet : une référence floutée ou moyenne convient mieux.
**Erreurs fréquentes** : dériver la probabilité au lieu du logit ; prendre le gradient de la classe prédite (ou de la somme des 10 logits) quand on demande une autre classe ; remettre le modèle en `train()` (ses batchnorms changent alors le calcul) ; réutiliser un tenseur dont le `.grad` s'accumule d'un appel à l'autre ; faire la moyenne des canaux pour la saillance (le maximum est la convention de Simonyan et coll.) ; oublier le facteur $\mathbf{x} - \mathbf{x}'$.
**Variante** : avec 10 intervalles seulement, l'écart de la complétude depuis l'image noire vaut 4 % en médiane et 21 % au pire sur 16 images ; avec 200, 0,16 % en médiane et 0,7 % au pire : les auteurs conseillent 20 à 300 pas, et de vérifier la complétude.

### Ex B6.20 — Grad-CAM par hooks sur le ResNet fine-tuné du B1 : bonnes prédictions et erreurs 📦
**Réponse** : avec le ResNet-18 de référence (recette de B1.17 à 96 pixels, 79,6 % sur 500 images de test, 75,0 % sur les 200 du réservoir), les cartes de `layer4` (3 × 3, agrandies à 96 × 96) tombent sur l'objet pour la plupart des bonnes prédictions. Les 8 erreurs retenues : grenouille → cheval, avion → cheval, automobile → avion, cheval → cerf, chien → cerf, chat → chien, chien → chat, oiseau → chat ; la corrélation entre la carte de la classe prédite et celle de la vraie classe vaut 0,96 et 0,97 pour les confusions chat/chien, 0,59 et 0,44 pour cheval → cerf et chien → cerf, −0,23 et −0,78 pour l'avion et l'automobile ; pour la grenouille, la carte de la vraie classe est nulle (la ReLU a tout éteint).
**Démarche** :
```python
def activations_and_gradients_20(model, layer, images, target):
    store = {}

    def save(module, inputs, output):                     # returns None: the output is left unchanged
        store["A"] = output.detach()
        output.register_hook(lambda grad: store.__setitem__("G", grad.detach()))

    x = torch.as_tensor(images, dtype=torch.float32).clone().requires_grad_(True)   # frozen model: the input needs grad
    t = torch.as_tensor(np.asarray(target)).long().reshape(-1)
    handle = layer.register_forward_hook(save)
    try:
        model(x).gather(1, t[:, None]).sum().backward()
    finally:
        handle.remove()
    return store["A"].numpy(), store["G"].numpy()


def cam_20(model, layer, images, target, size):
    A, G = activations_and_gradients_20(model, layer, images, target)
    cams = torch.as_tensor(mylearn.explain.grad_cam(A, G), dtype=torch.float32)
    return F.interpolate(cams[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()
```
Le modèle est gelé : sans une entrée qui demande son gradient, la sortie de `layer4` n'est reliée à aucun tenseur qui en demande un, et `register_hook` lève une erreur. Le hook de passe avant ne rend rien (sinon il remplacerait la sortie) et se retire dans un `finally`. Sur une confusion entre deux classes voisines (chat et chien), les deux cartes regardent au même endroit : le réseau a vu l'animal, il s'est trompé d'espèce. Pour l'automobile prise pour un avion, la carte d'« avion » s'allume sur le ciel, celle d'« automobile » sur la voiture et la route : une erreur de **contexte** ; pour l'avion pris pour un cheval, les deux cartes regardent aussi des régions différentes. Une carte nulle pour la vraie classe veut dire qu'aucune région n'augmente son score : le réseau n'y a rien vu de cette classe. À 3 × 3, une carte ne dit guère plus que « en haut à gauche » ou « au centre » ; `layer3` (6 × 6) localiserait plus finement, mais avec des notions moins liées aux classes (🐛 B6.21). Et une carte qui tombe sur l'objet ne prouve pas que le modèle a raison pour de bonnes raisons : il faut encore vérifier qu'elle dépend du modèle (🔬 B6.22).
**Erreurs fréquentes** : oublier `requires_grad_` sur l'entrée d'un modèle gelé ; laisser un hook posé (chaque passe suivante le déclenche encore) ; prendre le gradient de la classe prédite quand on étudie la vraie ; agrandir la carte avec une autre convention que l'image (`align_corners=True`, ou `nearest`) ; enlever la ReLU de Grad-CAM ; viser `layer4` à 32 pixels (une carte de 1 × 1).
**Variante** : sur les 8 bonnes prédictions, la carte de la **deuxième** classe la plus probable ressemble souvent à celle de la prédite : corrélations 0,85 (grenouille et oiseau), 0,98 (chien et chat), −0,15 (camion et avion), 0,48, 0,72, −0,06, 0,30 et 0,66 : quand deux classes partagent l'objet (deux animaux), leurs cartes se confondent ; Grad-CAM dit où se trouve ce qui fait monter un score, pas ce qui distingue deux classes.

### Ex B6.21 — Grad-CAM à l'envers : mauvaise couche, mauvaise classe, ReLU oubliée, carte mal alignée 🐛
**Réponse** : quatre lignes à corriger ; tel quel, le Grad-CAM du collègue s'écarte de la référence de 1,64 au pire, avec des valeurs de −0,99 à 1,00 :
```python
LAYER_21 = RESNET.layer4                                            # 1. the last block (3 x 3), not layer2 (12 x 12)


def score_21(logits, target):
    return logits.gather(1, target.reshape(-1, 1)).sum()            # 2. the class studied, not logits.argmax(1)


def combine_21(A, G):
    weights = G.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * A).sum(dim=1))                      # 3. the ReLU, once, on the weighted sum
    peak = cam.abs().amax(dim=(1, 2), keepdim=True)
    return torch.where(peak > 0, cam / torch.where(peak > 0, peak, 1.0), 0.0)


def upsample_21(cam, size):
    return F.interpolate(cam[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]   # 4.
```
**Démarche** : les diagnostics suivent le chemin du calcul, et chacun ne regarde que son morceau. 1. La résolution se lit sur la forme des activations : `layer2` sort 12 × 12 à 96 pixels, une case pour 8 × 8 pixels, des canaux qui répondent à des bords et des textures ; Grad-CAM vise la dernière couche convolutive, où les canaux portent des notions liées aux classes (🧠 Q8). 2. Le score prenait le logit de la classe **prédite** (`argmax`) : la carte ne dépendait plus de `target`, ce qui se voit en comparant les gradients de deux classes (identiques), et que la carte d'une bonne prédiction ne trahit pas. 3. Sans ReLU, les régions qui font **baisser** le score restent, avec un signe négatif ; une fois chaque carte divisée par sa plus grande valeur absolue, une carte presque toute négative se lit comme une carte « inversée ». 4. Les images ont été agrandies avec `align_corners=False` : les centres des 3 cases tombent aux pixels 16, 48 et 80 ; avec `align_corners=True`, les centres des cases du bord sont collés aux coins, et la carte glisse vers le bord, de 10 pixels au centre de la première case (16 pixels tout au bord). Une rampe 0, 1, 2 le mesure sans dépendre du modèle, là où la carte d'une vraie image ne dit pas où elle « devrait » tomber. Le point 5 compare enfin le tout à `wbx.grad_cam` : une couche de la bonne taille mais qui n'est pas la sortie d'un bloc entier (`layer4[1].conv2`) n'y passerait pas. De ces défauts, l'alignement est le plus discret : à 3 × 3, un décalage d'un tiers de case ne change pas « l'histoire » que raconte la carte.
**Erreurs fréquentes** : corriger la classe en `logits[:, target].sum()` (une matrice $N \times N$ : chaque image reçoit les logits des classes de toutes les autres) ; mettre la ReLU sur les activations (elles sont déjà positives à la sortie d'un bloc) au lieu de la somme pondérée ; remplacer la ReLU par une valeur absolue ; agrandir en `nearest` (une carte en marches) ; dériver la probabilité au lieu du logit.
**Variante** : avec `LAYER_21 = RESNET.layer3` (6 × 6), une fois les trois autres défauts corrigés, la corrélation de chaque carte agrandie avec celle de `layer4` vaut 0,14, 0,56, 0,68, −0,17 et −0,13 sur cinq des six images (médiane 0,14 ; la quatrième a une carte de `layer4` nulle, toute éteinte par la ReLU) ; avec `layer2`, médiane 0,01 : plus la couche est précoce, moins sa carte ressemble à celle que Grad-CAM recommande.

### Ex B6.22 — Test de santé d'Adebayo : randomiser les poids couche par couche, la carte change-t-elle ? 🔬
**Réponse** : sur les 16 images du CNN (médiane sur 16 images × 3 graines, entre crochets les quartiles), en cascade :

| carte | `fc` seule | jusqu'à `features.17` | jusqu'à `features.10` | jusqu'à `features.0` |
|---|---|---|---|---|
| Grad-CAM (`features.19`) | 0,20 [−0,08 ; 0,43] | −0,00 [−0,15 ; 0,24] | −0,14 [−0,37 ; 0,24] | −0,12 [−0,32 ; 0,26] |
| gradients intégrés (abs.) | 0,39 [0,31 ; 0,48] | 0,36 [0,29 ; 0,46] | 0,25 [0,19 ; 0,35] | 0,16 [0,12 ; 0,23] |
| gradient (saillance) | 0,52 [0,45 ; 0,57] | 0,42 [0,38 ; 0,46] | 0,30 [0,25 ; 0,34] | 0,23 [0,16 ; 0,28] |
| bords de l'image | 1,00 | 1,00 | 1,00 | 1,00 |

Grad-CAM change dès que la sortie est randomisée et ne garde plus rien de sa carte ensuite (jusqu'à 9 cartes sur 48 deviennent constantes, toutes éteintes par la ReLU) ; les gradients et les gradients intégrés changent aussi, mais gardent une corrélation positive qui ne descend qu'à 0,16-0,23 ; le contrôle reste à 1 par construction. Une partie à la fois (`fc`, `features.17`, `features.10`, `features.0`), la saillance reste à 0,35-0,52 et les gradients intégrés à 0,25-0,39 ; Grad-CAM y garde des médianes de 0,15 à 0,46, mais sur peu de cartes : jusqu'à 15 sur 48 deviennent constantes. Sur le ResNet, la carte de Grad-CAM n'a plus que 0,35 de corrélation médiane après la seule tête (22 cartes constantes sur 48), puis −0,06 et −0,16 en descendant dans `layer4`.
**Démarche** :
```python
def randomize_22(model, names, seed=0):
    out = copy.deepcopy(model)
    with torch.random.fork_rng():                     # the draws of the following cells do not move
        torch.manual_seed(seed)
        for name in names:
            for module in out.get_submodule(name).modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    module.reset_parameters()
    return out.eval().requires_grad_(False)


def sanity_22(model, explain, names, mode="cascade", seed=0, absolute=False):
    flat = lambda maps: np.abs(np.asarray(maps, dtype=float)) if absolute else np.asarray(maps, dtype=float)
    before = flat(explain(model))
    rows = []
    for k in range(len(names)):
        parts = names[:k + 1] if mode == "cascade" else [names[k]]
        after = flat(explain(randomize_22(model, parts, seed)))
        rows.append([np.nan if np.ptp(u) == 0 or np.ptp(v) == 0 else float(spearmanr(u, v).statistic)
                     for u, v in zip(before.reshape(len(before), -1), after.reshape(len(after), -1))])
    return np.array(rows)
```
La copie protège le modèle des cellules suivantes, `fork_rng` leurs tirages. Le contrôle sert d'étalon : une carte qui ne regarde pas le modèle a une corrélation de 1 quoi qu'on fasse aux poids ; une méthode qui s'en approche « ressemble » à un détecteur de bords. Grad-CAM passe nettement le test : sa carte dépend des poids de la tête et des dernières convolutions. Les gradients et les gradients intégrés passent aussi (leur corrélation baisse quand on randomise davantage), mais leur carte garde une part de structure qui ne vient pas de ce qu'a appris le réseau : pour les gradients intégrés en valeurs absolues, le facteur $\lvert x_i - x'_i \rvert$, l'intensité des pixels (la variante le montre) ; pour le gradient, sans doute l'architecture elle-même (les convolutions et les ReLU d'un réseau aux poids aléatoires donnent déjà des gradients structurés par l'image). Une carte constante veut dire qu'après randomisation aucune région ne fait monter le score de la classe : la carte a changé du tout au tout, mais la corrélation n'est pas définie, d'où un compte à part. La dispersion compte : d'une graine à l'autre, la médiane de Grad-CAM après `fc` va de 0,04 à 0,38, celle des gradients intégrés de 0,36 à 0,42 ; d'une image à l'autre, les quartiles de Grad-CAM couvrent souvent de −0,3 à 0,4 : la conclusion se tire sur la tendance, pas sur une image. Dans l'article, gradients et Grad-CAM réussissent le test, Guided Backprop et Guided Grad-CAM échouent ; ici, sur un petit CNN, on retrouve les deux premiers résultats. Un test réussi prouve seulement que la carte dépend des poids (🧠 Q9) ; il ne prouve pas qu'elle montre ce que le modèle utilise.
**Erreurs fréquentes** : randomiser le modèle lui-même (sans copie : toutes les cellules suivantes utilisent un CNN détruit) ; tirer les poids sans `fork_rng` (les tirages suivants du notebook changent) ; remettre aussi les batchnorms à zéro (le contrat les garde) ; en cascade, ne randomiser que la partie $k$ ; comparer les cartes par une corrélation de Pearson, sensible à quelques pixels extrêmes ; conclure sur une image ou une graine.
**Variante** : les gradients intégrés **signés** (`absolute=False`), en cascade : médianes 0,10, −0,14, 0,08, 0,05, 0,00, 0,03 et 0,03 (de `fc` à `features.0`, une graine) : sans la valeur absolue, la ressemblance disparaît dès la tête ; c'est bien $\lvert x_i - x'_i \rvert$ qui la portait. Gradient × entrée en valeurs absolues, la même cascade : 0,36, 0,32, 0,28, 0,21, 0,18, 0,14 et 0,15.

### Ex B6.23 — group_rates, demographic_parity_difference et equalized_odds_difference 🔨
**Réponse** : les 42 tests du groupe passent (39, et 3 ignorés sans fairlearn).
**Démarche** :
```python
def _ratio(num, den):
    return float(num / den) if den else float("nan")


def _binary(name, a):
    a = np.asarray(a)
    if not np.isin(a, (0, 1)).all():                  # NaN, 0.5, 2 or -1: refused; True/False and 0.0/1.0: accepted
        raise ValueError(f"{name} must contain only 0 and 1")
    return a.astype(int)


def group_rates(y_true, y_pred, groups):
    yt, yp, g = _binary("y_true", y_true), _binary("y_pred", y_pred), np.asarray(groups)
    if not len(yt) == len(yp) == len(g):
        raise ValueError("y_true, y_pred and groups must have the same length")
    rates = {}
    for key in np.unique(g):
        m = g == key
        (tn, fp), (fn, tp) = np.asarray(mylearn.metrics.confusion_matrix(yt[m], yp[m], labels=[0, 1])).tolist()
        n = tn + fp + fn + tp
        rates[key] = {"n": int(n), "base_rate": _ratio(tp + fn, n), "selection_rate": _ratio(tp + fp, n),
                      "tpr": _ratio(tp, tp + fn), "fpr": _ratio(fp, fp + tn), "precision": _ratio(tp, tp + fp),
                      "accuracy": _ratio(tp + tn, n)}
    return rates


def demographic_parity_difference(y_pred, groups):
    rates = group_rates(np.zeros(len(np.asarray(y_pred)), dtype=int), y_pred, groups)
    if len(rates) < 2:
        raise ValueError("at least two groups are needed")
    s = [r["selection_rate"] for r in rates.values()]
    return float(max(s) - min(s))


def equalized_odds_difference(y_true, y_pred, groups):
    rates = group_rates(y_true, y_pred, groups)
    if len(rates) < 2:
        raise ValueError("at least two groups are needed")
    if any(math.isnan(r["tpr"]) or math.isnan(r["fpr"]) for r in rates.values()):
        raise ValueError("a group has no positive or no negative example")
    tpr, fpr = [r["tpr"] for r in rates.values()], [r["fpr"] for r in rates.values()]
    return float(max(max(tpr) - min(tpr), max(fpr) - min(fpr)))
```
Chaque taux est le rapport de deux cases de la matrice **du groupe**, jamais de la matrice totale. Avec `labels=[0, 1]`, la matrice a toujours ses quatre cases, même pour un groupe sans positif ou sans sélectionné, et `C.ravel()` se lit `TN, FP, FN, TP` (lignes = vérité) : le livre et beaucoup de cours mettent TP en haut à gauche, scikit-learn TN. Un dénominateur nul donne `nan`, qui dit « ce taux n'existe pas pour ce groupe » ; l'equalized odds refuse alors de rendre un nombre, parce qu'un écart calculé en ignorant ce groupe (ce que fait fairlearn) cacherait qu'on ne sait rien de ses erreurs. Avec trois groupes ou plus, l'écart max − min dit l'ampleur, pas **entre qui** : on garde `group_rates` à côté pour le dire.
**Erreurs fréquentes** : `confusion_matrix` sans `labels` (matrice 1 × 1 pour un groupe à un seul label, et `ValueError: not enough values to unpack`) ; FP et FN échangés ; diviser par la taille du groupe au lieu des positifs ou des négatifs ; `0` au lieu de `nan` pour un dénominateur nul ; prendre l'écart entre les deux premiers groupes seulement ; faire la moyenne des deux écarts de l'equalized odds au lieu du plus grand.
**Variante** : sur la validation d'*Adult*, par catégorie raciale (5 groupes, de 98 à 8 345 personnes), au seuil 0,5 : $\Delta_{\text{DP}} = 0{,}179$ et $\Delta_{\text{EO}} = 0{,}260$, l'écart de TPR allant de 0,40 (`Amer-Indian-Eskimo`, 99 personnes) à 0,66 (`White`) : l'écart le plus grand porte sur le plus petit groupe, ce qui demande son intervalle avant toute conclusion.

### Ex B6.24 — Audit d'équité sur Adult : indicateurs par sexe et par âge, intervalles de confiance bootstrap 📦
**Réponse** : sur la validation, au seuil 0,5 (500 tirages) : les femmes, 3 165 personnes dont 350 positives, ont un TPR de 0,569 [0,512 ; 0,619] et un FPR de 0,015 [0,011 ; 0,020] ; les hommes, 6 604 dont 1 987 positifs, 0,663 [0,642 ; 0,684] et 0,083 [0,076 ; 0,091] ; taux de sélection 0,076 et 0,258 pour des taux de base de 0,111 et 0,301. Écarts par sexe : $\Delta_{\text{DP}} = 0{,}181$ [0,166 ; 0,194], écart de TPR 0,095 [0,042 ; 0,150], écart de FPR 0,068 [0,058 ; 0,077], donc $\Delta_{\text{EO}} = 0{,}095$ [0,064 ; 0,150]. Par âge, les moins de 25 ans n'ont que **12** positifs sur 1 647 personnes : TPR 0,417 [0,133 ; 0,727], aucun faux positif ; l'écart de TPR entre tranches vaut 0,299 [0,163 ; 0,578] et $\Delta_{\text{DP}}$ par âge 0,344.
**Démarche** :
```python
def _rate_24(y, p, rate):
    y, p = np.asarray(y), np.asarray(p)
    if rate == "selection_rate":
        return float(p.mean()) if len(p) else float("nan")
    keep = y == (1 if rate == "tpr" else 0)
    return float(p[keep].mean()) if keep.any() else float("nan")


def rate_ci_24(y_true, y_pred, rate, n_boot=500, rng=None):
    pairs = np.column_stack([np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)])   # rows stay paired
    return mylearn.stats.bootstrap_ci(pairs, lambda s: _rate_24(s[:, 0], s[:, 1], rate), n_boot=n_boot, rng=rng)


def audit_24(y_true, y_pred, groups, n_boot=500, seed=0):
    y, p, g = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(groups)
    rng, rows = np.random.default_rng(seed), {}
    for key, r in mylearn.explain.group_rates(y, p, g).items():
        m = g == key
        row = {"n": r["n"], "positives": int(y[m].sum()),
               **{k: r[k] for k in ("base_rate", "selection_rate", "tpr", "fpr", "precision", "accuracy")}}
        for rate in ("selection_rate", "tpr", "fpr"):
            row[f"{rate}_low"], row[f"{rate}_high"] = rate_ci_24(y[m], p[m], rate, n_boot=n_boot, rng=rng)
        rows[key] = row
    table = pd.DataFrame.from_dict(rows, orient="index")[COLUMNS_24]
    table.index.name = "group"
    return table


def gap_ci_24(y_true, y_pred, groups, kind, n_boot=500, seed=0):
    y, p, g = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(groups)
    E = mylearn.explain

    def gap(yy, pp, gg):
        if kind == "dp":
            return float(E.demographic_parity_difference(pp, gg))
        if kind == "eo":
            return float(E.equalized_odds_difference(yy, pp, gg))
        values = [r[kind] for r in E.group_rates(yy, pp, gg).values()]
        return float(max(values) - min(values))

    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(g == key) for key in np.unique(g)]
    draws = [gap(y[idx], p[idx], g[idx]) for idx in
             (np.concatenate([m[rng.integers(0, len(m), size=len(m))] for m in members]) for _ in range(n_boot))]
    low, high = np.percentile(draws, [2.5, 97.5])
    return gap(y, p, g), float(low), float(high)
```
Les paires `(y, p)` se tirent ensemble, sinon on casse le lien entre un label et sa décision. Le TPR d'un groupe repose sur ses **positifs** : 12 chez les moins de 25 ans, d'où un intervalle de 0,13 à 0,73 qui ne dit presque rien ; 350 chez les femmes, 1 987 chez les hommes. Tirer dans chaque groupe garde les effectifs fixes (c'est ce qu'on veut dire par « l'incertitude de ce taux pour ce groupe ») ; tirer dans toute la validation ferait aussi varier la taille des groupes, avec des intervalles voisins pour les grands groupes. Par sexe, les trois écarts excluent 0 : le modèle sélectionne moins les femmes, en manque davantage parmi les qualifiées (TPR) et se trompe moins sur les autres (FPR). Par âge, les taux de sélection suivent de près les taux de base (de 0,007 à 0,387) : l'écart de sélection vient surtout des données ; l'écart de TPR, lui, est porté par les moins de 25 ans, au TPR très incertain. Avec 24 sous-groupes croisés, beaucoup n'auraient qu'une poignée de positifs, et un écart sur vingt sortirait de son intervalle à 95 % par hasard : on choisit les comparaisons d'avance (206.9).
**Erreurs fréquentes** : tirer les labels et les décisions séparément ; calculer l'intervalle d'un groupe sur toute la validation (les intervalles de tous les groupes deviennent pareils) ; rendre `std` ± 1,96 au lieu des centiles ; un intervalle à 80 % (le défaut d'un exemple du livre au ch. 2) ; un écart signé (femmes − hommes) au lieu de max − min ; un nouveau `rng` réinitialisé dans chaque groupe (les groupes reçoivent des tirages corrélés).
**Variante** : au seuil du MP3 (0,3607, le meilleur F1 sur ses folds), l'accuracy de validation tombe à 0,862 et l'écart change de nature : écart de TPR 0,108 [0,058 ; 0,161], écart de FPR 0,132 [0,118 ; 0,143], soit $\Delta_{\text{EO}} = 0{,}132$ : en sélectionnant plus, le modèle fait surtout plus de faux positifs chez les hommes.

### Ex B6.25 — Lire un tableau d'audit : où est l'écart, est-il significatif, que conclure ? 📈
**Réponse** : a) `"A"` (les moins de 25 ans, TPR 0,417) ; b) 12 ; c) `"F"` (65 ans et plus : de 0,448 à 0,651, une largeur de 0,203, sur 89 positifs) ; d) `True` (de 0,042 à 0,153) ; e) `False` (la borne haute des femmes, 0,619, est sous la borne basse des hommes, 0,642) ; f) `"C"`.
**Démarche** : un TPR se lit avec son nombre de **positifs**, pas avec l'effectif du groupe : les moins de 25 ans sont 1 647, mais 12 seulement gagnent plus de 50 000 $ ; leur intervalle, de 0,125 à 0,714, couvre presque tout. Parmi les autres tranches, la plus large est celle des 65 ans et plus (89 positifs), devant les 55-64 ans (314 positifs, largeur 0,105). Un écart est significatif quand l'intervalle **de l'écart** exclut 0 : c'est le cas de l'écart de TPR par sexe ; que les intervalles des deux taux soient disjoints (e) est une preuve plus forte encore, mais l'inverse ne vaut pas : deux intervalles qui se chevauchent n'excluent pas un écart réel. (A) est faux : des intervalles qui se chevauchent ne prouvent pas l'égalité, et ceux des 25-34 et des 45-54 ans ne se chevauchent même pas. (B) confond un constat et une prescription : la sélection suit les taux de base (0,003 contre 0,007 chez les moins de 25 ans, 0,347 contre 0,387 chez les 45-54 ans), et rien dans le tableau ne dit que la parité démographique est le bon critère. (D) est faux : l'écart de TPR par sexe exclut 0. (C) tient les trois morceaux : le TPR le plus bas repose sur 12 positifs, l'écart de TPR par âge a un intervalle qui exclut 0 mais va de 0,16 à 0,58, et la sélection suit les taux de base. À un métier, on rapporterait d'abord l'écart de TPR par sexe, avec son intervalle : « parmi les personnes qui dépassent 50 000 $, le modèle en repère 57 % chez les femmes et 66 % chez les hommes ; l'écart, de 4 à 15 points, est très probablement réel. » Six tranches font quinze comparaisons deux à deux : choisis-les d'avance.
**Erreurs fréquentes** : lire l'effectif du groupe au lieu de ses positifs ; conclure « pas d'écart » d'un intervalle large ; juger la significativité d'un écart au chevauchement des intervalles des deux taux ; prendre ΔDP pour un défaut du modèle quand il suit les taux de base.
**Variante** : si les moins de 25 ans avaient 120 positifs au lieu de 12, à TPR égal, l'intervalle (approximation normale) ferait $\pm 1{,}96\sqrt{0{,}417 \times 0{,}583 / 120} \approx \pm 0{,}088$, une largeur d'environ 0,18 au lieu de 0,59 : dix fois plus de positifs, un intervalle environ trois fois plus étroit ($\sqrt{10} \approx 3{,}2$).

### Ex B6.26 — Prédire : retirer la variable « sexe » rend-il le modèle équitable ? 🔮
**Réponse** : a) `"D"` (ROC-AUC 0,933) ; b) `"C"` (0,815) ; c) `"B"` (ΔEO 0,109 sans `sex` contre 0,095 avec, un rapport de 1,15) ; d) `"A"` (0,300). Calibrage (`tools/chapters/history_chB6.py`, 8 découpages d'*Adult* × 2 graines du modèle) : AUC de 0,931 à 0,937, puis de 0,807 à 0,818 ; rapport des ΔEO de 0,74 à 1,15 ; rapport des taux de sélection de 0,29 à 0,35 ; accuracy changée de −0,2 à +0,1 point.
**Démarche** : `relationship` vaut `Husband` ou `Wife` pour 45 % des personnes, ce qui dit presque leur sexe ; `marital-status` complète. Sans elles, le métier, les heures travaillées et les revenus du capital laissent encore deviner le sexe bien mieux qu'au hasard (0,81). Le modèle sans `sex` retrouve donc l'information par ses variables de substitution, et ses écarts ne baissent pas : sur cette validation, ΔEO **monte** un peu (le TPR des femmes passe de 0,569 à 0,554), et la sélection reste au même rapport, 0,30, très loin des quatre cinquièmes. Une part des écarts ne vient d'ailleurs d'aucune colonne : les taux de base (0,111 et 0,301) sont différents, et un modèle qui sélectionne selon ses estimations de revenu sélectionne à ces taux-là. Et sans `sex` dans les données, on ne pourrait plus ni mesurer ces écarts ni les corriger (206.8, 💼 B6.E5).
**Erreurs fréquentes** : croire qu'un modèle qui ne voit pas l'attribut ne peut pas en dépendre ; juger une variable de substitution à sa corrélation avec la cible au lieu de sa capacité à prédire l'attribut ; prendre un rapport de ΔEO de 1,15 pour une aggravation significative (il va de 0,74 à 1,15 d'un découpage à l'autre).
**Variante** : pour la catégorie raciale (`White` ou non), le même gradient boosting la prédit à partir des 11 autres features du pipeline avec une ROC-AUC de 0,761, et de 0,683 sans `native-country` (10 features) : dans *Adult*, la race a des variables de substitution plus faibles que le sexe, mais pas absentes.

### Ex B6.27 — Atténuer : retrait des variables de substitution, repondération, seuils par groupe, et ce que cela coûte 🔬
**Réponse** : sur la moitié d'évaluation de la validation (4 885 personnes), seuils par sexe choisis sur l'autre moitié pour un TPR de 0,638 (0,356 pour les femmes, 0,524 pour les hommes) :

| stratégie | accuracy | ΔDP sexe | ΔEO sexe | ΔDP âge | ΔEO âge | TPR F / M | FPR F / M |
|---|---|---|---|---|---|---|---|
| référence (seuil 0,5) | 0,8741 | 0,173 [0,155 ; 0,193] | 0,076 [0,058 ; 0,147] | 0,353 | 0,407 | 0,60 / 0,67 | 0,016 / 0,084 |
| sans `sex` | 0,8747 | 0,175 [0,156 ; 0,194] | 0,105 [0,058 ; 0,177] | 0,347 | 0,401 | 0,57 / 0,68 | 0,017 / 0,082 |
| sans `sex`, `relationship`, `marital-status` | 0,8487 | 0,113 [0,092 ; 0,133] | 0,081 [0,019 ; 0,158] | 0,313 | 0,454 | 0,45 / 0,53 | 0,034 / 0,055 |
| repondération (sexe × label) | 0,8731 | 0,106 [0,085 ; 0,124] | 0,118 [0,061 ; 0,178] | 0,318 | 0,359 | 0,73 / 0,61 | 0,035 / 0,060 |
| seuils par sexe (même TPR visé sur `CAL_27`) | 0,8757 | 0,135 [0,115 ; 0,155] | 0,068 [0,038 ; 0,130] | 0,347 | 0,401 | 0,72 / 0,65 | 0,030 / 0,075 |

Aucune stratégie ne divise ΔEO par sexe de façon nette : leurs intervalles se recouvrent tous.
**Démarche** :
```python
def reweigh_27(groups, y):
    g, y = np.asarray(groups), np.asarray(y)
    w = np.empty(len(y), dtype=float)
    for key in np.unique(g):
        for label in np.unique(y):
            cell = (g == key) & (y == label)
            if cell.any():
                w[cell] = np.mean(g == key) * np.mean(y == label) / np.mean(cell)
    return w


def tpr_thresholds_27(scores, y, groups, target_tpr):
    s, y, g = np.asarray(scores, dtype=float), np.asarray(y), np.asarray(groups)
    return {key: float(np.quantile(s[(g == key) & (y == 1)], 1 - target_tpr)) for key in np.unique(g)}


def decide_27(scores, groups, thresholds):
    s, g = np.asarray(scores, dtype=float), np.asarray(groups)
    return (s >= np.array([thresholds[key] for key in g])).astype(int)
```
Le retrait de `sex` seul ne change rien (B6.26). Retirer aussi ses deux variables de substitution coûte 2,5 points d'accuracy : le modèle perd une information utile pour **tout le monde**, et ses TPR baissent dans les deux groupes ; ΔDP diminue, ΔEO non, et l'écart par âge augmente (0,454). La repondération rend sexe et label indépendants dans l'entraînement : les femmes positives pèsent plus, et le modèle les sélectionne davantage, au point d'inverser l'écart de TPR (0,73 contre 0,61) : ΔDP baisse, ΔEO monte. Les seuils par sexe rapprochent les TPR sur la moitié de calibrage, mais sur d'autres personnes le TPR des femmes dépasse sa cible (0,72) : avec environ 175 femmes positives par moitié, un quantile se déplace de plusieurs points ; le FPR des femmes double, l'écart de FPR se réduit, et ΔEO baisse un peu, pour une accuracy qui ne bouge pas. Ces seuils utilisent le sexe **dans la décision**, la repondération seulement dans l'entraînement : la fiche dit le risque juridique du premier (206.10). Pour la statistique publique du MP3, où il faut des erreurs semblables dans les deux groupes sans changer la décision de qui que ce soit, ni l'une ni l'autre ne suffit telle quelle : on chercherait à égaliser **les deux** taux d'erreur, et à le vérifier sur des personnes neuves.
**Erreurs fréquentes** : les poids inversés ($P(g, y) / (P(g)P(y))$, qui aggravent le déséquilibre) ; le quantile `target_tpr` au lieu de `1 - target_tpr` ; le quantile de tous les scores du groupe au lieu de ceux de ses positifs ; choisir les seuils et les mesurer sur les mêmes personnes ; ne réauditer que le sexe.
**Variante** : choisis **et** mesurés sur la même moitié d'évaluation, les seuils par sexe (0,440 et 0,556) donnent des TPR de 0,639 et 0,638, et ΔEO 0,039 (le seul écart de FPR) : c'est ce que promettent les seuils sur les personnes qui ont servi à les choisir ; sur d'autres personnes, la moitié d'évaluation en donne 0,068.

### Ex B6.28 — Fiche modèle avec sections explicabilité et équité : indicateurs par groupe, limites, usage prévu 🛠️
**Réponse** : la fiche du notebook de solutions (`CARD_28`, plus de 600 mots) passe la vérification ; ses chiffres viennent de `FACTS_28` (accuracy de validation 0,8722, ROC-AUC 0,9293, n, TPR et FPR des huit groupes, importance par permutation : `capital-gain` 0,0538, `education-num` 0,0257, `marital-status` 0,0170, `occupation` 0,0168) et des mesures de B6.24 à B6.27.
**Démarche** : la fiche répond à trois lecteurs. Le métier lit l'usage prévu et les mises en garde : ici, un exercice, à ne jamais utiliser pour décider de quelqu'un, sur des données de 1994. L'auditeur lit les analyses quantitatives et l'équité : un tableau par groupe avec les effectifs, le critère retenu **et sa raison** (l'equalized odds, parce qu'une statistique publique doit avoir des erreurs semblables dans chaque groupe), les écarts avec leurs intervalles, les corrections essayées et leur coût. L'autorité cherche la base légale de l'usage de données sensibles et la supervision humaine (206.11). L'explicabilité dit la méthode **et** ses données (la validation ; le fond pour une valeur SHAP), et ses limites (des features corrélées se partagent l'importance). Une vérification de structure ne juge ni la justesse d'une raison, ni l'honnêteté des limites, ni ce qui manque sans qu'on l'ait demandé (les sous-groupes croisés, par exemple) : ce sont elles qui font une bonne fiche.
**Erreurs fréquentes** : recopier les chiffres à la main (ils divergent du modèle au premier réentraînement : une f-string) ; des taux sans effectifs ; un critère sans sa raison ; une rubrique « limites » générique (« le modèle peut se tromper ») ; oublier les usages hors périmètre ; une explicabilité sans ses données.
**Variante** : une analyse croisée sexe × âge, à ajouter aux analyses quantitatives : les femmes de 45 à 54 ans (489 personnes, 91 positives) ont un TPR de 0,58 et un FPR de 0,018, les hommes du même âge (1 219, 570 positifs) 0,74 et 0,174 ; les femmes de moins de 25 ans n'ont que 3 positives sur 736 : leur TPR (0,33) ne se rapporte pas sans dire qu'il ne veut rien dire.

### Ex B6.29 — Défi : écart d'equalized odds sous 0,07 sur Adult, en perdant moins de 2 points d'accuracy 🏆
**Réponse** : la référence passe : sur le test du MP3, ΔEO 0,016 (écart de TPR 0,014, de FPR 0,016), accuracy 0,8638 contre 0,8707 sans correction (0,70 point de moins) ; sur le découpage de contrôle, ΔEO 0,022 et 0,89 point de moins ; ajustée sur des labels mélangés, balanced accuracy 0,500. Sur 11 découpages (le test du MP3 avec 5 validations, puis 6 autres tests), son ΔEO va de 0,012 à 0,058 et sa baisse d'accuracy de 0,53 à 0,93 point ; sans correction, ΔEO va de 0,053 à 0,118.
**Démarche** :
```python
TARGET_EO_29 = 0.02     # the gap aimed at on the validation part: below the goal, for a margin on new people


def method_29(X_train, y_train, X_val, y_val):
    model = adult_pipeline().fit(X_train, y_train)                 # the reference recipe, without correction
    scores, sex = model.predict_proba(X_val)[:, 1], X_val["sex"].to_numpy()
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    curves = {}
    for group in ("Female", "Male"):                               # TPR, FPR and right decisions at each threshold
        pos, neg = scores[(sex == group) & (y_val == 1)], scores[(sex == group) & (y_val == 0)]
        tpr = (pos[None, :] >= grid[:, None]).mean(axis=1)
        fpr = (neg[None, :] >= grid[:, None]).mean(axis=1)
        curves[group] = (tpr, fpr, tpr * len(pos) + (1 - fpr) * len(neg))
    (tpr_f, fpr_f, right_f), (tpr_m, fpr_m, right_m) = curves["Female"], curves["Male"]
    eo = np.maximum(np.abs(tpr_f[:, None] - tpr_m[None, :]), np.abs(fpr_f[:, None] - fpr_m[None, :]))
    acc = (right_f[:, None] + right_m[None, :]) / len(y_val)
    if (eo <= TARGET_EO_29).any():                                 # the best accuracy among the fair enough pairs
        i, j = np.unravel_index(np.argmax(np.where(eo <= TARGET_EO_29, acc, -1.0)), acc.shape)
    else:
        i, j = np.unravel_index(np.argmin(eo), eo.shape)
    thresholds = {"Female": float(grid[i]), "Male": float(grid[j])}

    def decide(X):
        t = np.array([thresholds.get(value, 0.5) for value in X["sex"]])
        return (model.predict_proba(X)[:, 1] >= t).astype(int)

    return decide
```
Au seuil 0,5, les hommes ont à la fois plus de vrais positifs et plus de faux positifs : un seul seuil par sexe ne peut pas rapprocher les deux taux. Deux seuils choisis **ensemble** le peuvent : sur le test du MP3, la référence relève les deux (0,60 pour les femmes, 0,70 pour les hommes), ce qui ramène le FPR des hommes vers celui des femmes et les TPR l'un vers l'autre, pour moins d'un point d'accuracy. Le couple se choisit sur la validation, avec une marge (0,02 visé pour 0,07 exigé), parce que le TPR des femmes y repose sur environ 350 positives : un écart-type d'environ 0,026, autant sur le test. C'est aussi pourquoi le titre du contrat est passé de 0,05 à **0,07** : sur 11 découpages, la référence atteignait 0,058 au pire, et aucune variante essayée ne restait partout sous 0,05 avec moins de 2 points d'accuracy perdus : viser 0,01 sur la validation donne jusqu'à 0,083 ; choisir les seuils sur des scores hors fold de tout l'entraînement, jusqu'à 0,060 en visant 0,02, ou 0,040 en visant 0,01 mais jusqu'à 2,5 points perdus. Si `sex` était interdit dans la décision, il resterait à l'utiliser **pendant** l'entraînement seulement : une repondération ou les réductions d'Agarwal et coll. (`ExponentiatedGradient`, 206.10), avec un modèle qui ne voit pas `sex` ; B6.27 montre que la repondération seule ne suffit pas.
**Erreurs fréquentes** : choisir les seuils sur le test, ou les régler à la main après avoir regardé le test (la règle d'honneur) ; viser exactement 0,07 sur la validation ; ne régler que les TPR (l'écart de FPR reste) ; régler les seuils sur les features du test dans `decide` (refusé : chaque personne se décide seule) ; réutiliser `HGB_C` au lieu d'ajuster un modèle neuf.
**Variante** : en visant 0,01 au lieu de 0,02 sur la validation, ΔEO vaut 0,011 sur le test du MP3 mais 0,040 sur le contrôle, et l'accuracy baisse de 1,72 et 2,53 points : le contrôle échoue (plus de 2 points). Viser plus bas fait payer de l'accuracy sans gagner en robustesse : la marge utile se mesure en écarts-types de l'échantillon, pas en centièmes sur la validation.
