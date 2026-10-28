# 22 · Réseaux récurrents (RNN, LSTM, GRU) — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels, des exercices ✏️ 22.1 à 22.5 et de l'estimation 🧮 22.8 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 22.Q1 — Pourquoi un réseau dense ignore l'ordre
a) **Vrai** : une permutation fixe des entrées se compense par la même permutation des poids de la première couche ; le réseau permuté calcule exactement les mêmes sorties. b) **B** : chaque entrée a ses propres poids, et rien ne relie l'entrée 2 à l'entrée 1. c) **Faux** : si l'ordre change d'une fenêtre à l'autre, une même entrée ne porte plus toujours la même position, et l'information sur l'ordre est perdue. d) **B** : c'est le lien entre un élément et ses voisins qui fait une séquence. e) **Vrai** : les fenêtres des positions 0 à 3 et 1 à 4 partagent les positions 1, 2 et 3.
**À retenir** : une couche dense voit des positions, pas un ordre : elle ne sait pas que l'entrée 2 suit l'entrée 1, ni qu'un motif peut apparaître n'importe où dans la fenêtre.

### 22.Q2 — État ou poids : qu'est-ce qui change en production ?
a) **Faux** : sans optimiseur, les poids sont figés. b) **Vrai** : l'état est recalculé à chaque entrée, c'est la mémoire du réseau. c) **Vrai** : mêmes poids, mêmes entrées, même état de départ, donc les mêmes calculs. d) **Faux** : après A, l'état de départ de B n'est plus nul, et les sorties peuvent changer (c'est voulu quand A et B se suivent dans un même texte). e) **B** : l'état.
**À retenir** : les poids sont ce que le réseau a appris ; l'état est ce qu'il a lu. Le premier ne change qu'à l'entraînement, le second à chaque entrée.

### 22.Q3 — Délai, déroulement et pas de temps
a) **B** : le délai garde l'état calculé jusqu'à l'entrée suivante. b) **1** : les 12 copies du schéma déroulé sont la même cellule. c) **12** : une application par pas de temps. d) **Faux** : elles partagent leurs poids, c'est ce qui permet de lire une séquence de n'importe quelle longueur. e) **2** : `weight_ih` et `weight_hh`.
**À retenir** : déroulé, un RNN est un réseau profond de $T$ couches qui partagent toutes leurs poids.

### 22.Q4 — Entrée one-hot, taille de l'état et sens de ses valeurs
a) **3 000** : une case par mot. b) **1** : toutes les cases sont nulles sauf celle du mot. c) **Faux** : $W_{ih}$ a la forme $(D, H)$, $D$ et $H$ sont indépendants ; la taille de l'état est un hyperparamètre. d) **1 920** : $30 \times 64$. e) **B** : c'est l'entraînement qui fixe ce que contient chaque case de l'état ; on n'en interprète que rarement quelques-unes.
**À retenir** : la taille de l'état se choisit comme le nombre de neurones d'une couche cachée, et son contenu s'apprend.

### 22.Q5 — (échantillons, pas de temps, features) : la bonne organisation
a) **« (10, 6, 2) »** : 10 journées, 6 mesures, 2 grandeurs. b) **« (6, 10, 2) »** : `(T, N, D)`, le temps d'abord. c) **Faux** : `reshape` relit les 120 nombres dans le même ordre et les redécoupe ; il mélange les journées et les heures. Il faut `x.permute(1, 0, 2)` (ou `transpose(0, 1)`). d) **B** : une journée est une séquence de 6 pas ; les nuits ne sont pas mesurées, donc deux journées ne se suivent pas sans trou, et les mettre bout à bout ferait croire au réseau que 18 h précède directement 8 h. e) **1** : le deuxième nombre de `(N, T, D)` ; le RNN voit un seul pas à 6 features, sans ordre entre elles.
**À retenir** : le réseau tourne avec un tenseur mal rangé, mais il n'apprend pas ce qu'on croit ; chaque axe a un sens, et l'on vérifie la forme avant d'entraîner.

### 22.Q6 — BPTT, gradients évanescents et explosifs, dépendances longues
a) **B**. b) **C** : $0{,}9^{50} \approx 0{,}0052$. c) **C** : $1{,}1^{50} \approx 117$. d) **Faux** : le clipping réduit les gradients trop grands, il n'agrandit pas ceux qui disparaissent ; contre la disparition, il faut changer la cellule (LSTM, GRU) ou l'initialisation. e) **B** : la difficulté est d'**apprendre** le lien, pas seulement de le stocker.
**À retenir** : le gradient qui remonte $k$ pas est multiplié $k$ fois par un facteur ; à 0,9 près ou à 1,1 près, il disparaît ou explose.

### 22.Q7 — Une porte, c'est une multiplication
a) **20** : $0{,}25 \times 80$. b) **B** : la sigmoïde, à valeurs dans $]0, 1[$. c) **Faux** : un facteur positif garde le signe. d) **Faux** : un facteur inférieur à 1 ne peut que réduire la valeur absolue. e) **A** : l'entrée courante et l'état caché précédent, à travers des poids appris.
**À retenir** : une porte est un vecteur de nombres entre 0 et 1, recalculé à chaque pas, qui dose ce qui passe, élément par élément.

### 22.Q8 — Oublier, se souvenir, sélectionner
a) **B** : la porte d'entrée $i$. b) **C** : la porte de sortie $o$. c) **Faux** : elle a été ajoutée par Gers, Schmidhuber et Cummins (1999-2000). d) **B** : $c_t = 1 \times c_{t-1} + 0 \times g_t = c_{t-1}$. e) **C** : les blocs 0, 1, 2, 3 sont $i, f, g, o$.
**À retenir** : oubli, entrée, candidat, sortie : $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ et $h_t = o_t \odot \tanh(c_t)$, dans l'ordre $i, f, g, o$ de PyTorch.

### 22.Q9 — LSTM ou GRU ?
a) **3** : $[\,r \mid z \mid n\,]$. b) **Faux** : le GRU n'a qu'un état $h_t$. c) **B** : 3 blocs au lieu de 4, soit trois quarts des paramètres du LSTM. d) **Faux** : les comparaisons ne désignent pas de vainqueur général ; on essaie les deux.
**À retenir** : le GRU fusionne oubli et entrée dans une seule porte de mise à jour, et coûte 25 % de paramètres de moins.

### 22.Q10 — One-to-many, many-to-one, many-to-many
a) **B**. b) **A**. c) **D** : l'ordre des mots change d'une langue à l'autre, il faut avoir lu la phrase avant de la traduire. d) **C** : un label par image, au même pas. e) **B** : 40 caractères lus, une seule sortie, le caractère suivant.
**À retenir** : la structure se lit en comptant les entrées et les sorties, et en se demandant si une sortie peut être produite avant d'avoir tout lu.

### 22.Q11 — RNN profond et bidirectionnel : quand et à quel prix ?
a) **Faux** : le sens arrière a besoin de la fin de la séquence, qu'on n'a pas encore en temps réel. b) **B** : toute sa séquence d'états, un par pas. c) **C** : les deux états sont concaténés, d'où $2H$ valeurs par pas dans `output`. d) **Vrai** : PyTorch les nomme `weight_ih_l0_reverse`, etc. ; une couche bidirectionnelle a deux fois plus de paramètres. e) **B** : chaque couche et chaque sens ajoutent des poids, qu'il faut assez de données pour apprendre.
**À retenir** : bidirectionnel = toute la séquence disponible avant de répondre ; profond = chaque couche rend toute sa séquence à la suivante.

### 22.Q12 — 🕰️ Transformers, RNN légers et modèles à espace d'états
a) **2017**. b) **Vrai** : un RNN ne peut calculer $h_t$ qu'après $h_{t-1}$ ; un Transformer calcule toutes les positions d'une séquence en même temps pendant l'entraînement. c) **B** : mémoire de taille fixe et coût constant par pas. d) **B**. e) **Vrai** : sans comparaison à une baseline simple, on ne sait pas si le LSTM apporte quelque chose.
**À retenir** : les Transformers dominent le texte ; les RNN restent utiles en flux continu et à petit budget, et leurs idées reviennent dans les modèles à espace d'états.

<a id="rappels"></a>

## 🔁 Rappels

### 22.R1 — Ch. 21 : Une convolution 1D peut-elle lire une séquence ?
a) **17** : $20 - 4 + 1$. b) **Vrai** : c'est le partage de poids. c) **5** : $3 + 2$. d) **500** couches : $n$ couches de noyau 3 donnent un champ récepteur de $2n + 1$, et $2n + 1 \ge 1\,001$ demande $n \ge 500$. e) Dans ta copie : une pile de convolutions calcule toutes les positions en parallèle (pas d'attente d'un pas à l'autre), et le chemin entre deux positions traverse autant de couches qu'il en faut pour les relier, pas autant que de pas ; avec une dilatation qui double à chaque couche (1, 2, 4, 8…), le champ récepteur double aussi : $2^{n+1} - 1$ valeurs après $n$ couches de noyau 3, soit 9 couches pour dépasser 1 001 au lieu de 500. Le prix : le champ récepteur est fixé par l'architecture, alors qu'un RNN a en principe une mémoire illimitée.
**À retenir** : une convolution 1D est une alternative sérieuse au RNN pour les séquences (les réseaux WaveNet ou TCN), surtout avec des dilatations.

### 22.R2 — Ch. 19 : Gradient clipping et optimiseurs adaptatifs
a) **5** : $\sqrt{9 + 16}$. b) **0,6** : $3/5$ (la norme devient 1). c) **Faux** : on multiplie tout le vecteur par $1/5$, la direction ne change pas. d) **A**. e) **Vrai** : le premier pas d'Adam vaut environ $\eta\,g/\sqrt{g^2} = \eta \operatorname{signe}(g)$, quelle que soit l'échelle de $g$.
**À retenir** : `clip_grad_norm_` borne la norme totale sans changer la direction ; Adam et RMSprop normalisent le pas par la taille récente des gradients.

### 22.R3 — Ch. 15 : Ajuster le scaler sur le train seulement
a) **Faux** : le test doit venir après l'entraînement, comme en production. b) **B**. c) **1,25** : $(250 - 0)/(200 - 0)$. d) **Faux** : en production, des valeurs jamais vues arriveront aussi ; on garde le scaler ajusté sur l'entraînement, et une valeur hors de $[0, 1]$ est normale. e) **« transform »**.
**À retenir** : `fit` sur l'entraînement, `transform` sur tout le reste ; avec une série temporelle, l'entraînement est le passé.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 22.1 — Combien de fenêtres ? Températures, taches solaires et Holmes ✏️
**Réponses** : a) **20** ; b) **10** ; c) **3 312** ; d) **3 187** ; e) **2 587** ; f) **191 654** ; g) **C**.
**Démarche** : la fenêtre qui démarre en $p$ a sa cible en $p + w - 1 + h$, qui doit être au plus $T - 1$ : $p \le T - w - h$. Les départs $0, s, 2s, \dots$ jusqu'à $T - w - h$ sont au nombre de $N = \lfloor (T - w - h)/s \rfloor + 1$. a) $24 - 4 - 1 = 19$, d'où 20. b) $\lfloor 19/2 \rfloor + 1 = 10$. c) $276$ années de 1749 à 2024 incluses, $276 \times 12 = 3\,312$. d) $3\,312 - 126 = 3\,186$, d'où 3 187. e) $T = 3\,312 - 600 = 2\,712$ ; $2\,712 - 126 + 1 = 2\,587$. f) $\lfloor 574\,959/3 \rfloor + 1 = 191\,653 + 1$. g) $N \approx T/s$, donc $T \approx 500\,000 \times 3 = 1{,}5$ million de caractères (le livre parle d'environ 304 000 mots).
**Dans ta copie** : avec un pas de 3, on a trois fois moins de fenêtres, donc trois fois moins de calcul par epoch et de mémoire ; les fenêtres consécutives se recouvraient de 39 caractères sur 40 et apportaient peu de neuf. On perd deux tiers des cibles : chaque caractère n'est cible qu'une fois sur trois, mais le texte reste couvert presque entièrement par les entrées.
**Erreurs fréquentes** : oublier l'horizon ($N = T - w + 1$) ; compter 1749 à 2024 comme 275 ans ; prendre en e) les fenêtres dont seule l'entrée est avant le test (leur cible y déborde : une fuite).
**Variante** : $T = 100$, $w = 10$, $h = 3$, $s = 5$ : $\lfloor 87/5 \rfloor + 1 =$ **18** fenêtres.

### Ex 22.2 — Dérouler à la main une cellule RNN scalaire sur trois pas ✏️
**Réponses** : a) **0,9640** ; b) **0,4479** ; c) **−0,6504** ; d) **−1,3009** ; e) **0,9487** ; f) **0,25**.
**Démarche** : a) $\tanh 2 = 0{,}96403$. b) $\tanh(0 + 0{,}5 \times 0{,}96403) = \tanh 0{,}48201 = 0{,}44785$. c) $\tanh(-1 + 0{,}22393) = \tanh(-0{,}77607) = -0{,}65045$. d) $2 \times (-0{,}65045)$. e) $h_1 = \tanh(-1) = -0{,}76159$, $h_2 = \tanh(-0{,}38080) = -0{,}36340$, $h_3 = \tanh(2 - 0{,}18170) = \tanh 1{,}81830 = 0{,}94867$. f) Sans tanh, $h_3 = u x_3 + w(u x_2 + w\,u x_1) = u x_3 + w u x_2 + w^2 u x_1$ : $\alpha_1 = w^2 u = 0{,}25$.
**Dans ta copie** : les mêmes $u$, $w$, $b$ à chaque pas, c'est le partage des poids : la cellule traite chaque pas de la même façon, le nombre de paramètres ne dépend pas de la longueur, et un motif est reconnu où qu'il soit. Pour 20 pas, $\alpha_1 = w^{19} u = 0{,}5^{19} \approx 1{,}9 \times 10^{-6}$ : la première entrée ne compte presque plus. La dérivée de $h_{20}$ par rapport à $x_1$ est aussi de cet ordre : c'est l'annonce du gradient évanescent (∂ 22.6).
**Erreurs fréquentes** : arrondir $h_1$ à 0,96 avant de calculer $h_2$ (on trouve 0,4462 au lieu de 0,4479) ; calculer en degrés ; oublier que $h_0 = 0$ ; croire que l'ordre inverse donne le même $h_3$.
**Variante** : avec $x = (1, 1, 1)$, $h_1 = 0{,}7616$, $h_2 = 0{,}8811$, $h_3 =$ **0,8938** : avec une entrée constante, l'état converge vers un point fixe, solution de $h = \tanh(1 + 0{,}5\,h)$.

### Ex 22.3 — Compter les poids : la cellule à 69 poids du livre, puis nn.RNN et nn.LSTM ✏️
**Réponses** : a) **80** ; b) **64** ; c) **27** ; d) **179** ; e) **160** ; f) **8** ; g) **640** ; h) **255 705** ; i) **1 024**.
**Démarche** : a) 8 neurones A qui lisent 10 entrées. b) 8 neurones B qui lisent 8 valeurs d'état. c) $3 \times 8 + 3$. d) $80 + 64 + 8 + 27 = 179$. e) $1 \times (10 \times 8 + 8^2 + 2 \times 8) = 80 + 64 + 16$. f) $160 + (8 \times 3 + 3) = 187$, soit $187 - 179 = 8$ : le second vecteur de biais de PyTorch. g) $4 \times 160 = 640$. h) Couche 1 : $4\,(89 \times 128 + 128^2 + 256) = 112\,128$ ; couche 2 : $4\,(128 \times 128 + 128^2 + 256) = 132\,096$ ; dense : $128 \times 89 + 89 = 11\,481$ ; total $255\,705$. i) Keras n'a qu'un vecteur de $4 \times 128$ biais par couche : $2 \times 512 = 1\,024$ de moins, soit 254 681, « un peu plus de 250 000 ».
**Dans ta copie** : la cellule est la même à chaque pas, avec les mêmes poids : une séquence plus longue l'applique plus souvent, sans ajouter de paramètres. Le **calcul** et la mémoire des activations, eux, croissent avec $T$.
**Erreurs fréquentes** : oublier un des deux biais de PyTorch ; compter $W_{hh}$ comme $(D, H)$ ; pour la deuxième couche du générateur, prendre $D = 89$ ; multiplier par $T$.
**Variante** : `nn.GRU(10, 8)` a $3 \times 160 =$ **480** paramètres.

### Ex 22.4 — Un pas de LSTM à la main ✏️
**Réponses** : a) **0,5** ; b) **0,7311** ; c) **0,7616** ; d) **0,6225** ; e) **1,8429** ; f) **0,5920** ; g) **2** ; h) **0,7311** ; i) **« (4, 1) »**.
**Démarche** : les pré-activations $a = W_{ih}\,x_t + W_{hh}\,h_{t-1} + b$ valent, bloc par bloc : $i$ : $-0{,}5 + 0{,}5 + 0 = 0$ ; $f$ : $1 - 1 + 1 = 1$ ; $g$ : $1 + 0 + 0 = 1$ ; $o$ : $-1 + 1{,}5 + 0 = 0{,}5$. a) $\sigma(0)$. b) $\sigma(1)$. c) $\tanh 1$. d) $\sigma(0{,}5)$. e) $0{,}73106 \times 2 + 0{,}5 \times 0{,}76159 = 1{,}46212 + 0{,}38080$. f) $0{,}62246 \times \tanh 1{,}84292 = 0{,}62246 \times 0{,}95106$. g) $f = \sigma(21) \approx 1$ et $i = \sigma(-20) \approx 2 \times 10^{-9}$ : $c_t \approx 2{,}0000$, la mémoire est recopiée. h) $\partial (f_t\,c_{t-1})/\partial c_{t-1} = f_t$. i) PyTorch empile les 4 blocs de $H = 1$ ligne : $(4H, D) = (4, 1)$, la transposée de la convention de `mylearn`.
**Dans ta copie** : avec le réglage de g), la mémoire reste à 2 après 30 pas ($2 \times \sigma(21)^{30} \approx 2$) ; avec celui de a) à f), le seul terme $f\,c$ donne $2 \times 0{,}7311^{30} \approx 1{,}7 \times 10^{-4}$ : l'information de départ a disparu. Seule une porte d'oubli proche de 1 transporte une information loin, et le même facteur règle le gradient qui remonte par ce chemin.
**Erreurs fréquentes** : ranger les portes dans l'ordre $f, i, o, g$ ; appliquer une sigmoïde au candidat $g$ (c'est une tanh) ; oublier la tanh de $h_t = o \tanh c_t$ ; répondre « (1, 4) » en i).
**Variante** : avec $c_{t-1} = -2$ et le reste inchangé, $c_t = -1{,}0813$ et $h_t =$ **−0,4940**.

### Ex 22.5 — Softmax avec température à la main ✏️
**Réponses** : a) **0,8214** ; b) **0,9756** ; c) **0,6045** ; d) **1** ; e) **0,3333** ; f) **54,60** ; g) **Vrai**.
**Démarche** : $p(\text{k}) = 1/(1 + e^{-2/T} + e^{-2{,}5/T})$. a) $1/(1 + 0{,}13534 + 0{,}08208)$. b) $1/(1 + e^{-4} + e^{-5}) = 1/(1 + 0{,}01832 + 0{,}00674)$. c) $1/(1 + e^{-1} + e^{-1{,}25}) = 1/(1 + 0{,}36788 + 0{,}28650)$. d) Les deux exponentielles tendent vers 0. e) Elles tendent vers 1 : $1/3$. f) $e^{(3 - 1)/0{,}5} = e^4 = 54{,}598$. g) Diviser tous les logits par $T > 0$ garde leur ordre : la température change les probabilités, jamais le caractère le plus probable.
**Dans ta copie** : en prenant toujours le plus probable, le générateur est déterministe : dès qu'une suite de caractères revient, ce qui suit revient aussi, et le texte tourne en boucle. Pour un texte lisible, une température un peu inférieure à 1 (0,5 à 0,8) ; pour des noms propres inventés, une température proche de 1 ou au-dessus, qui laisse passer des suites rares.
**Erreurs fréquentes** : multiplier les logits par $T$ au lieu de les diviser ; calculer la softmax des logits sans soustraire, puis élever les probabilités à la puissance $T$ ; croire qu'une température élevée change le plus probable.
**Variante** : avec des logits $(2 ;\ 2 ;\ 0)$ à $T = 0{,}5$, chacun des deux premiers caractères a la probabilité $e^4/(2e^4 + 1) =$ **0,4955** : une température basse ne départage pas deux ex aequo.

### Ex 22.6 — Pourquoi le gradient s'évanouit : un produit de dérivées de tanh ∂
1. $h_k = \tanh(a_k)$ avec $\tanh' = 1 - \tanh^2$, et $\partial a_k / \partial h_{k-1} = w$ : $\dfrac{\partial h_k}{\partial h_{k-1}} = (1 - h_k^2)\,w$.
2. $h_T$ dépend de $h_t$ à travers $h_{t+1}, \dots, h_{T-1}$ : la règle de la chaîne multiplie les facteurs, $\dfrac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} w\,(1 - h_k^2)$. L'entrée $x_t$ n'agit que par $a_t$, avec $\partial h_t/\partial x_t = u\,(1 - h_t^2)$ : $\dfrac{\partial L}{\partial x_t} = \dfrac{\partial L}{\partial h_T}\,\Big(\prod_{k=t+1}^{T} w\,(1 - h_k^2)\Big)\,u\,(1 - h_t^2)$.
3. Comme $h_k \in\, ]-1, 1[$, $0 < 1 - h_k^2 \le 1$, donc chaque facteur vérifie $|w\,(1 - h_k^2)| \le |w|$, et $\left|\dfrac{\partial h_T}{\partial h_t}\right| \le |w|^{T - t}$. Si $|w| < 1$, cette borne décroît exponentiellement avec la distance : le gradient venant des entrées lointaines **s'évanouit**, quelles que soient les valeurs des états. Pour $w = 0{,}9$ et 50 pas : $0{,}9^{50} \approx 5{,}2 \times 10^{-3}$.
4. Si les $h_k$ restent proches de 0, $1 - h_k^2 \approx 1$ et le produit vaut environ $w^{T - t}$ : $1{,}5^{20} \approx 3\,325$, une **explosion**. Mais un gradient énorme donne un pas énorme : $w$ grandit, les pré-activations aussi, la tanh **sature** ($h_k \approx \pm 1$), les facteurs $1 - h_k^2$ tombent presque à 0, et le gradient s'effondre : la loss fait un bond, puis l'apprentissage stagne. D'où le clipping, qui évite le pas énorme.
5. $\dfrac{\partial h_T}{\partial h_t} = J_T \cdots J_{t+1}$ (dans l'ordre que fixe la convention), et $\lVert J_k \rVert \le \lVert \operatorname{diag}(1 - h_k^2) \rVert\,\lVert W_{hh}^\top \rVert \le \sigma_{\max}(W_{hh})$ (une matrice et sa transposée ont les mêmes valeurs singulières). Par l'inégalité sur les produits : $\lVert \partial h_T/\partial h_t \rVert \le \sigma_{\max}(W_{hh})^{T - t}$. Si $\sigma_{\max}(W_{hh}) < 1$, le gradient s'évanouit sûrement : c'est la condition suffisante de Pascanu, Mikolov et Bengio (2013). L'inverse n'est pas vrai : $\sigma_{\max} > 1$ est nécessaire pour une explosion, pas suffisant, car la tanh peut compenser.
6. $w$ intervient dans chaque $a_k$ : $\dfrac{\partial L}{\partial w} = \sum_{k=1}^{T} \delta_k\,(1 - h_k^2)\,h_{k-1}$, avec $\delta_k = \delta_T\,\dfrac{\partial h_T}{\partial h_k}$. Le terme d'indice $k$ porte le facteur de la question 2 sur $T - k$ pas : quand le gradient s'évanouit, les termes de $k$ proche de $T$ dominent la somme. La mise à jour de $w$ se règle donc presque uniquement sur ce que les dernières entrées expliquent ; le lien entre une sortie et une entrée lointaine contribue un terme minuscule, noyé dans les autres : la descente de gradient ne le découvre pas. Le gradient total ne disparaît pas, c'est sa part « à long terme » qui disparaît.
7. Notons $\tanh^{\circ n}$ la composition. Comme $\tanh 0 = 0$, tous les intermédiaires valent 0 en 0, et la règle de la chaîne donne $(\tanh^{\circ n})'(0) = \prod_{j=1}^{n} \tanh'(0) = 1^n = 1$. La composition de tanh ne fait donc pas disparaître le gradient près de 0 : elle aplatit la courbe loin de 0 (en $x = 1$, la pente vaut 0,42 pour $n = 1$, 0,25 pour $n = 2$, 0,096 pour $n = 5$, 0,012 pour $n = 25$), mais la cause principale est le facteur $w$ (ou $W_{hh}$) de chaque pas, et la saturation quand les états s'éloignent de 0. Cette explication ne dit d'ailleurs rien de l'explosion, impossible avec la seule tanh.
**Erreurs fréquentes** : oublier le facteur $w$ dans la jacobienne ; écrire une somme au lieu d'un produit en 2. et un produit au lieu d'une somme en 6. ; croire que le gradient total devient nul ; confondre valeur propre et valeur singulière (pour une matrice non symétrique, la plus grande valeur propre en module ne borne pas la norme du produit d'un nombre fini de facteurs).
**Variante** : avec $w = 0{,}5$, la borne sur 10 pas vaut $0{,}5^{10} \approx$ **0,00098** : un poids récurrent de 0,5 limite la mémoire utile du gradient à quelques pas.

### Ex 22.7 — Le chemin additif de la cellule LSTM ∂
1. Toutes les opérations sont élément par élément : $c_{t,j} = f_{t,j}\,c_{t-1,j} + i_{t,j}\,g_{t,j}$ ne dépend que de $c_{t-1,j}$ parmi les composantes de $c_{t-1}$, avec la dérivée $f_{t,j}$ (les portes et le candidat étant fixés). La jacobienne est diagonale : $\operatorname{diag}(f_t)$.
2. Le long du chemin direct, $\dfrac{\partial c_T}{\partial c_t} = \operatorname{diag}(f_T) \cdots \operatorname{diag}(f_{t+1}) = \operatorname{diag}\Big(\prod_{k=t+1}^{T} f_k\Big)$. Avec des portes à 0,99 sur 100 pas : $0{,}99^{100} \approx 0{,}37$ ; le RNN aux facteurs de 0,9 donne $0{,}9^{100} \approx 2{,}7 \times 10^{-5}$, plus de 10 000 fois moins. Aucun poids ni aucune dérivée de tanh n'apparaît sur ce chemin : seules les portes d'oubli, que le réseau apprend à ouvrir.
3. Biais 0 : $0{,}5^{20} \approx 9{,}5 \times 10^{-7}$ ; biais 1 : $0{,}731^{20} \approx 1{,}9 \times 10^{-3}$, environ 2 000 fois plus. Au début de l'entraînement, les portes ne savent encore rien ; avec $f \approx 0{,}5$, le chemin additif perd la moitié du gradient à chaque pas, et le réseau ne peut pas découvrir qu'il lui faudrait garder une information. Un biais positif ouvre la porte par défaut : le gradient porte loin dès le départ, et le réseau apprend ensuite à fermer la porte là où il faut oublier. Jozefowicz et coll. le recommandent pour toute implémentation (1 ou 2), et constatent que cela comble l'écart avec le GRU.
4. $c_{t-1} \to \tanh \to$ multiplié par $o_{t-1}$ $\to h_{t-1} \to$ multiplié par $W_{hh}$ (les blocs de $i$, $f$, $g$) $\to$ dérivées de la sigmoïde ou de la tanh $\to i_t, f_t, g_t \to c_t$ (multiplié par $c_{t-1}$, $g_t$ ou $i_t$). Ce chemin traverse une matrice de poids et des dérivées de fonctions qui saturent, à chaque pas, comme celui d'un RNN simple : il peut s'évanouir ou exploser. Le gradient total est la somme des deux chemins ; le chemin direct suffit à transporter l'information loin.
5. $h_t = o_t \odot \tanh(c_t)$ : ce que la cellule « présente au monde » passe par une tanh et une porte, deux fonctions non linéaires. Ce qui est linéaire, c'est la **mise à jour de la mémoire**, $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ : une somme, de dérivée $f_t$ par rapport à $c_{t-1}$. En 1997, sans porte d'oubli, ce facteur valait exactement 1 : le « carrousel à erreur constante ». La bonne explication est donc le chemin additif dans la mémoire, pas la sortie.
6. $\dfrac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}(z_t)$ le long du chemin direct : la porte de mise à jour $z_t$ joue le rôle de $f_t$ (et $1 - z_t$ celui de $i_t$).
**Erreurs fréquentes** : écrire la jacobienne comme une matrice pleine ; oublier l'autre chemin (le gradient d'un LSTM peut lui aussi exploser, d'où le clipping) ; croire que le LSTM supprime toute disparition (si les portes d'oubli apprennent à valoir 0,5, le chemin s'évanouit aussi) ; confondre $c_t$ et $h_t$.
**Variante** : un GRU dont la porte $z$ vaut 0,95 transmet, le long du chemin direct, un facteur $0{,}95^{50} \approx$ **0,077** sur 50 pas.

### Ex 22.8 — Fermi : les 17 millions de poids du CNN-LSTM et le prix d'une epoch 🧮
**Réponses** : a) **32 768** ; b) **208** ; c) **16 843 776** ; d) **1 161** ; e) **16 845 145** ; f) **4 198 656** ; g) **16 777 216** ; h) **11**.
**Démarche** : a) $\lfloor (128 + 4 - 5)/2 \rfloor + 1 = 64$, donc $64 \times 64 \times 8$. b) $8 \times (25 + 1)$. c) $4\,(32\,768 \times 128 + 128^2 + 2 \times 128) = 4 \times 4\,210\,944$. d) $128 \times 9 + 9$. e) $208 + 16\,843\,776 + 1\,161$, « un peu moins de 17 millions », dont 99,99 % dans le LSTM. f) $4\,(32\,768 \times 32 + 32^2 + 64) = 4 \times 1\,049\,664$. g) $32\,768 \times 4 \times 128$. h) $16\,777\,216 \times 16\,000 \approx 2{,}7 \times 10^{11}$, puissance 11.
**Dans ta copie** : $2{,}7 \times 10^{11} / 10^{10} \approx 27$ secondes pour cette seule passe avant ; la passe arrière en coûte environ deux fois plus, et la convolution ($64 \times 64 \times 8 \times 25 \approx 0{,}8$ million de multiplications-additions par image) est négligeable. Le gain est dans l'entrée du LSTM : un pooling global moyen de chaque carte réduit l'entrée à 8 valeurs, et le LSTM à $4\,(8 \times 128 + 128^2 + 256) = 70\,656$ paramètres, environ 240 fois moins (mais l'information de position disparaît, ce qu'il faudrait vérifier : ici, la position de la balle est justement ce qui compte ; un pooling 8 × 8, qui garde $8 \times 8 \times 8 = 512$ valeurs, donnerait 328 704 paramètres et garderait une position grossière).
**Erreurs fréquentes** : oublier le stride (sortie de 128 × 128) ; multiplier le nombre de paramètres par le nombre d'images ; oublier le $4\times$ des portes ; prendre l'ordre de grandeur de g) au lieu de celui de l'epoch.
**Variante** : un max pooling 2 × 2 après la convolution réduit l'entrée à $32 \times 32 \times 8 = 8\,192$ valeurs, et le LSTM à **4 260 864** paramètres.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 22.9 — Expliquer l'état caché d'un RNN en cinq lignes 🗣️
**Réponse modèle** : « Imagine quelqu'un qui écoute une histoire mot après mot et qui ne peut garder qu'une petite fiche de notes. À chaque mot, il relit sa fiche, l'efface en partie et la réécrit en tenant compte du mot entendu : cette fiche, c'est l'état caché. Elle change à chaque mot, même quand il connaît déjà son métier ; ce qui ne change plus, c'est sa manière de prendre des notes, qu'il a apprise en écoutant des milliers d'histoires : ce sont les poids. Sa fiche a une taille fixe, il doit donc choisir quoi garder, et il oublie peu à peu les détails anciens. »
**Ce qui compte** : un résumé de tout ce qui a été lu, mis à jour à chaque entrée ; la différence entre l'état (qui change à chaque entrée, même après l'entraînement) et les poids (appris une fois) ; une taille fixe, donc un choix de ce qu'on garde ; ce choix est appris, pas programmé. La limite de l'image : le réseau ne « choisit » pas consciemment, et ses notes sont des nombres qu'on sait rarement lire.
**Erreurs fréquentes** : confondre l'état et les poids ; dire que l'état garde toute l'histoire mot pour mot ; utiliser une formule ou le mot « vecteur » sans l'expliquer ; reprendre la boîte de pièces du livre.
**Variante** : explique en une phrase ce qu'ajoute un LSTM : « il a en plus un tiroir où il peut ranger une note intacte aussi longtemps qu'il veut, avec des interrupteurs pour décider quand y mettre, garder ou effacer ».

### Ex 22.10 — Écrire « à la manière de », imiter une voix : usages et abus ⚖️
1. Arthur Conan Doyle est mort en 1930, Jules Verne en 1905 : en France et dans l'Union européenne, les droits patrimoniaux s'éteignent 70 ans après la mort de l'auteur, et leurs œuvres sont dans le domaine public depuis 2001 et 1976. On peut donc entraîner un modèle sur ces textes, les citer et les imiter librement (le droit moral, le respect de l'œuvre et du nom de l'auteur, reste perpétuel en France). Aux États-Unis, les dernières nouvelles de Holmes, publiées dans les années 1920, sont restées protégées plus longtemps : dans l'affaire *Klinger contre la succession Conan Doyle* (2014), la cour d'appel du 7ᵉ circuit a jugé que les personnages et les éléments des récits tombés dans le domaine public étaient libres, et que seuls les éléments ajoutés dans les récits encore protégés le restaient. Pour un auteur vivant, ses textes sont protégés : les recopier ou en reproduire des passages (ce que fait un modèle qui a mémorisé) est une contrefaçon. Le **style**, lui, n'est pas protégé, et le **pastiche** est une exception au droit d'auteur en France (art. L122-5 du Code de la propriété intellectuelle, « compte tenu des lois du genre ») : il imite une manière d'écrire, se donne pour une imitation et ne reprend pas le texte. La copie reprend l'expression elle-même. La question nouvelle est celle de l'entraînement sur l'œuvre d'un auteur vivant sans son accord, et de la concurrence qu'un tel modèle peut faire à ses propres livres.
2. Les auteurs et les éditeurs : les œuvres ont été copiées sans autorisation ni rémunération, parfois depuis des bibliothèques pirates ; les modèles peuvent recracher des passages ; leurs sorties concurrencent les œuvres sur le même marché. Les entreprises : l'entraînement est un usage « transformatif » (le *fair use* américain), le modèle apprend des régularités statistiques et non des copies ; en Europe, la directive de 2019 sur le droit d'auteur prévoit une exception de fouille de textes et de données, dont les ayants droit peuvent se retirer. Affaires : *New York Times contre OpenAI et Microsoft* (déposée en décembre 2023) ; *Bartz contre Anthropic* (2025), où le juge a estimé que l'entraînement sur des livres achetés relevait du *fair use*, mais pas la constitution d'une bibliothèque de copies piratées, avant un accord de 1,5 milliard de dollars ; en Allemagne, *GEMA contre OpenAI* (tribunal de Munich, novembre 2025), qui a jugé que la mémorisation de paroles de chansons par un modèle portait atteinte au droit d'auteur. Ces affaires évoluent : vérifie leur état au moment où tu lis.
3. Légitime : une personne atteinte de la maladie de Charcot (SLA) enregistre sa voix avant de la perdre, et un synthétiseur parlera ensuite avec sa voix ; le doublage d'un film avec l'accord de l'acteur ; un livre audio lu par la voix de son auteur. Abus : l'arnaque au « faux proche » ou au faux dirigeant, qui imite une voix au téléphone pour obtenir un virement ; un faux message d'un responsable politique (en janvier 2024, des appels automatiques imitaient la voix de Joe Biden pour dissuader des électeurs du New Hampshire de voter). Précautions : un consentement vérifié du titulaire de la voix (par exemple un texte de consentement qu'il lit lui-même), une vérification d'identité, le refus des voix de personnalités publiques, un marquage des sons produits (filigrane), la conservation des traces et une procédure de signalement.
4. L'article 50 du règlement (UE) 2024/1689 : les fournisseurs de systèmes qui génèrent du son, des images, des vidéos ou du texte doivent **marquer** leurs sorties de façon lisible par une machine et détectable ; ceux qui diffusent un hypertrucage (*deep fake*) doivent **indiquer** que le contenu a été généré ou manipulé ; un texte généré publié pour informer le public sur des questions d'intérêt public doit aussi le dire, sauf s'il a été relu par un humain qui en assume la responsabilité éditoriale. Ces obligations s'appliquent depuis le **2 août 2026** ; le paquet « omnibus » a seulement laissé aux fournisseurs de systèmes déjà sur le marché jusqu'au 2 décembre 2026 pour le marquage.
5. De ne pas appeler ces textes des « inédits » : c'est faire croire qu'ils sont de Conan Doyle. Il doit dire clairement que ce sont des pastiches générés par un modèle entraîné sur les récits de Conan Doyle, indiquer comment ils ont été produits et relus, et ne pas présenter le nom de l'auteur comme une signature. Les lecteurs ont le droit de savoir ce qu'ils lisent ; un pastiche annoncé comme tel est une tradition littéraire, un faux annoncé comme authentique est une tromperie.
**Erreurs fréquentes** : croire que le domaine public autorise à attribuer un texte à l'auteur ; confondre style (libre) et texte (protégé) ; présenter le *fair use* américain comme une règle européenne ; affirmer qu'une affaire en cours est tranchée ; oublier le consentement pour une voix.
**Variante** : une agence veut faire dire une publicité par une voix clonée d'un acteur vivant connu. Sans son accord, c'est une atteinte à ses droits de la personnalité (en France, le droit sur sa voix, comme sur son image) et souvent une pratique commerciale trompeuse ; avec son accord, il faut un contrat qui fixe l'usage, la durée et la rémunération, et le son doit être marqué comme généré.

### Ex 22.11 — Lire Hochreiter et Schmidhuber (1997) : Long Short-Term Memory 📄
1. Apprendre à garder une information sur de longs intervalles par la rétropropagation récurrente prend un temps énorme, parce que l'erreur qui remonte dans le temps est insuffisante et décroît. Les auteurs visent des délais minimaux de **plus de 1 000 pas** entre une entrée et la sortie qui en dépend.
2. À chaque pas en arrière, l'erreur est multipliée par la dérivée de l'activation et par un poids ($f'(net)\,w$) ; sur $q$ pas, c'est un produit de $q$ facteurs : si ces facteurs dépassent 1 en valeur absolue, l'erreur explose (et les poids oscillent) ; s'ils sont plus petits que 1, elle décroît exponentiellement. C'est exactement le produit $\prod w\,(1 - h_k^2)$ de ∂ 22.6, avec la sigmoïde de l'époque au lieu de la tanh (dont la dérivée vaut au plus 1/4).
3. Pour qu'une unité reliée à elle-même transmette une erreur constante, il faut que son facteur d'un pas vaille 1 : $f'_j(net_j)\,w_{jj} = 1$. Les auteurs prennent une activation **identité** et un poids **fixe** $w_{jj} = 1$ : c'est le carrousel à erreur constante, l'unité centrale de la cellule. C'est la mémoire $c_t = c_{t-1} + i_t\,g_t$, le chemin additif de ∂ 22.7 avec $f_t = 1$.
4. Une **porte d'entrée**, qui protège la mémoire des entrées non pertinentes, et une **porte de sortie**, qui protège le reste du réseau d'un contenu de mémoire pas encore utile. La porte d'oubli manque (le mot *forget* n'apparaît pas dans l'article) : elle a été ajoutée par Gers, Schmidhuber et Cummins (1999, puis *Neural Computation* en 2000), pour que la cellule puisse se remettre à zéro.
5. Par exemple l'expérience 4, le problème de l'addition : chaque pas reçoit un nombre tiré dans $[-1, 1]$ et un marqueur ; deux pas seulement sont marqués, et le réseau doit donner à la fin de la séquence une version mise à l'échelle de la somme des deux nombres marqués. Séquences d'au moins $T = 100$, 500 ou 1 000 pas, soit des délais minimaux de 50 à 500 pas ; le LSTM ne se trompe que sur une séquence de test sur 2 560 environ, après 74 000 à 853 000 séquences d'entraînement. Il n'y a pas de comparaison : les auteurs écrivent qu'aucun autre algorithme récurrent n'avait résolu ce type de tâche. Pour une comparaison, l'expérience 2a met le LSTM face à RTRL, BPTT et au compresseur de séquences de Schmidhuber : il réussit à tous les essais, et plus vite. Conclusion : sur des tâches artificielles construites pour n'avoir que des dépendances longues, le LSTM réussit là où les méthodes de l'époque échouent.
6. L'erreur n'est rétropropagée que le long des connexions vers les sorties et des auto-connexions des cellules : elle est coupée dès qu'elle « voudrait » quitter une cellule ou une porte vers le pas précédent. Ce sont les chemins par $h_{t-1}$ (question 4 de ∂ 22.7) qui sont coupés ; le calcul est local dans l'espace et le temps, et coûte $O(1)$ par pas et par poids. Aujourd'hui, `loss.backward()` calcule le gradient **exact** sur la séquence déroulée (BPTT complète) ; pour de très longues séquences, on coupe la séquence en morceaux (*truncated BPTT*), ce qui coupe dans le temps, pas à l'intérieur d'un pas.
7. Tel quel : la mémoire additive, les portes multiplicatives à sigmoïde, l'idée d'un flux d'erreur protégé. Ce qui a changé : la porte d'oubli (et un biais d'oubli positif à l'initialisation), des cellules aux portes calculées pour chaque valeur de la mémoire au lieu de blocs qui partagent leurs portes, la tanh au lieu des sigmoïdes mises à l'échelle de 1997, le gradient exact au lieu de l'approximation, et les connexions *peephole* de Gers, ajoutées puis abandonnées par PyTorch.
**Erreurs fréquentes** : attribuer la porte d'oubli à l'article de 1997 ; dire que le CEC a un poids appris (il est fixé à 1) ; confondre la troncature de 1997 (dans un pas) et la *truncated BPTT* (dans le temps) ; citer une comparaison pour l'expérience 4, qui n'en a pas.
**Variante** : sans porte d'oubli, une cellule qui reçoit $i\,g = 0{,}1$ à chaque pas voit sa mémoire grandir de 0,1 par pas : **10** après 100 pas, et sans limite ensuite. C'est la « dérive de l'état interne » de la section 4.8, à laquelle la porte d'oubli a remédié.

<a id="entretien"></a>

## 💼 Entretien

### 22.E1 — Pourquoi les Transformers ont-ils remplacé les RNN pour le texte ? Quand garderais-tu un LSTM ?
**Réponse modèle en 60 secondes** : « Un RNN calcule ses états l'un après l'autre : pour avoir l'état du pas 1 000, il faut les 999 précédents. Pendant l'entraînement, on ne peut donc pas paralléliser le long de la séquence, ce qui limite la taille des modèles et des données qu'on peut utiliser. Le Transformer, avec l'attention, compare toutes les positions entre elles d'un seul coup : l'entraînement se parallélise sur GPU, et deux mots éloignés sont reliés par un chemin court au lieu de mille pas où le gradient s'affaiblit. C'est ce qui a permis les grands modèles de langage. Je garderais un LSTM ou un GRU quand les données arrivent en flux et qu'il faut répondre à chaque pas avec une mémoire et un coût constants : un capteur embarqué, du débruitage audio en temps réel, une petite série temporelle où un modèle léger suffit. Et je regarderais aussi les modèles à espace d'états comme Mamba, qui gardent ce coût constant par pas tout en s'entraînant en parallèle. »
**Ce qui compte** : le calcul séquentiel du RNN contre le parallélisme du Transformer pendant l'entraînement ; la longueur des chemins entre positions éloignées ; le passage à l'échelle ; le coût de l'attention, quadratique en la longueur ; les cas où un RNN reste pertinent (flux, embarqué, petites séries) ; une mention des modèles à espace d'états.
**Relances possibles** : « Et à l'inférence, le Transformer est-il aussi plus rapide ? » (pas forcément : il génère aussi token par token, et doit garder en mémoire les clés et valeurs de tous les tokens précédents, une mémoire qui croît avec le texte ; un RNN n'a qu'un état de taille fixe) · « Quel est le coût de l'attention ? » (quadratique en la longueur de la séquence, en calcul et en mémoire, contre linéaire pour un RNN ; d'où les variantes à attention efficace et les modèles à espace d'états) · « Un Transformer sait-il l'ordre des mots ? » (pas de lui-même : l'attention est invariante par permutation, on ajoute un encodage de position).

### 22.E2 — Explique le gradient évanescent et deux façons d'y remédier
**Réponse modèle en 60 secondes** : « Dans un RNN déroulé, le gradient qui remonte d'un pas au précédent est multiplié par la jacobienne d'un pas : la matrice de poids récurrente, combinée à la dérivée de l'activation, qui vaut au plus 1 pour une tanh. Sur $k$ pas, c'est un produit de $k$ facteurs : à 0,9 par pas, il reste 0,5 % du gradient après 50 pas. Le gradient total ne s'annule pas, mais la contribution des entrées lointaines devient négligeable, et le réseau n'apprend pas qu'une sortie dépend de quelque chose de lu longtemps avant. Le remède principal est une cellule à portes, LSTM ou GRU : la mémoire du LSTM se met à jour par une somme, et le long de ce chemin le gradient n'est multiplié que par la porte d'oubli, que le réseau peut garder proche de 1 ; on initialise d'ailleurs son biais à 1. Deuxième remède : une bonne initialisation de la matrice récurrente, orthogonale par exemple, et des séquences pas trop longues ou des connexions plus courtes. Le problème symétrique, l'explosion, se traite par le gradient clipping. »
**Ce qui compte** : le produit de facteurs et son ordre de grandeur ; la distinction entre le gradient total et sa part à long terme ; la conséquence sur les dépendances longues ; le chemin additif du LSTM ou du GRU ; une seconde solution (initialisation, architecture, séquences plus courtes) ; le clipping réservé à l'explosion.
**Relances possibles** : « Le clipping aide-t-il ? » (contre l'explosion seulement : il réduit les gradients trop grands, il n'agrandit pas ceux qui disparaissent) · « Un LSTM est-il immunisé ? » (non : le chemin par l'état caché passe par les poids et peut s'évanouir ou exploser, et si les portes d'oubli apprennent à valoir 0,5, le chemin additif s'évanouit aussi ; on garde le clipping) · « Comment le mesurer ? » (calculer la norme du gradient de la loss par rapport à l'entrée de chaque pas, ou à chaque état, et la tracer en fonction de la distance à la sortie).

### 22.E3 — LSTM ou GRU : différences et critères de choix
**Réponse modèle en 60 secondes** : « Les deux ajoutent des portes à un RNN pour garder l'information longtemps. Le LSTM a une mémoire séparée de son état et trois portes : l'oubli, l'entrée et la sortie, plus un candidat ; la mémoire se met à jour par une somme. Le GRU n'a qu'un état et deux portes : une porte de mise à jour, qui dose à la fois ce qu'on garde et ce qu'on ajoute, et une porte de réinitialisation. Il a trois blocs de poids au lieu de quatre, donc 25 % de paramètres en moins, et il est un peu plus rapide. Les grandes comparaisons ne désignent pas de vainqueur : les résultats sont souvent proches, le LSTM s'en sort parfois mieux sur de très longues dépendances, et un LSTM avec un biais d'oubli de 1 rattrape le GRU là où il était en retard. En pratique, je commence par celui que la bibliothèque et l'équipe connaissent, souvent le LSTM, j'essaie l'autre avec le même budget de réglage, et je choisis sur la validation et sur le coût. »
**Ce qui compte** : la mémoire séparée et le nombre de portes ; le rapport des paramètres (4 blocs contre 3) ; l'absence de vainqueur général ; le biais d'oubli ; une méthode de choix fondée sur la validation et les contraintes.
**Relances possibles** : « Combien de paramètres a `nn.LSTM(100, 64)` ? » ($4 \times (100 \times 64 + 64^2 + 2 \times 64) = 42\,496$ ; le GRU en a les trois quarts, 31 872) · « Pourquoi deux vecteurs de biais dans PyTorch ? » (pour la compatibilité avec cuDNN ; un seul suffirait pour le LSTM, mais dans le GRU la porte de réinitialisation multiplie le second, $b_{hn}$) · « Que fait la porte de réinitialisation ? » (elle dose la part de l'état précédent qui entre dans le candidat : à 0, le candidat ne dépend plus que de l'entrée, comme si l'on recommençait une séquence).

### 22.E4 — Comment valider un modèle de série temporelle sans fuite de données ?
**Réponse modèle en 60 secondes** : « Je coupe dans le temps, jamais au hasard : les semaines les plus anciennes pour l'entraînement, puis une période de validation, puis une période de test, comme en production où l'on prévoit le futur à partir du passé. J'ajuste la normalisation sur l'entraînement seulement, et je construis les fenêtres de chaque partie sans qu'aucune entrée ni aucune cible ne déborde sur la période suivante. Pour régler les hyperparamètres, j'utilise une validation glissante, `TimeSeriesSplit` : plusieurs coupures successives, l'entraînement s'agrandissant à chaque fois. Les variables explicatives aussi doivent être connues au moment de la prévision, pas calculées après coup. Je compare le LSTM à des baselines simples, la persistance (la dernière valeur) et surtout la persistance saisonnière (la même semaine l'an dernier), puis un modèle de gradient boosting sur des valeurs retardées : dans la compétition M5, seule une équipe sur trois a battu la persistance saisonnière. Je mesure une erreur adaptée, la MAE ou une erreur relative à la baseline, à l'horizon dont l'entreprise a besoin, et je surveille la dérive après le déploiement. »
**Ce qui compte** : la coupure temporelle ; le scaler ajusté sur l'entraînement ; les fenêtres à cheval sur la coupure ; la validation glissante ; les variables connues au moment de prévoir ; les baselines naïves et saisonnières ; une métrique à l'horizon utile ; le suivi en production.
**Relances possibles** : « Pourquoi ne pas faire une validation croisée classique ? » (les plis mélangeraient passé et futur : le modèle verrait des semaines postérieures à celles qu'il prévoit, et les fenêtres voisines, presque identiques, se retrouveraient des deux côtés de la coupure ; le score serait trop optimiste) · « Que faire si le LSTM ne bat pas la baseline ? » (le dire : garder la baseline ou un modèle plus simple, vérifier la quantité de données, essayer des variables explicatives, un horizon différent ou un modèle de gradient boosting ; un modèle complexe qui ne bat pas la persistance n'a pas sa place en production) · « Comment prévoir 12 semaines à l'avance ? » (soit un modèle qui sort directement les 12 valeurs, soit une prévision pas à pas qui réinjecte ses propres prédictions, dont l'erreur s'accumule ; on évalue chaque horizon séparément).

### 22.E5 — À quoi sert la température quand on génère du texte ?
**Réponse modèle en 60 secondes** : « Le modèle donne un score, un logit, à chaque token possible ; on les transforme en probabilités par une softmax, puis on tire le token suivant. La température divise les logits avant la softmax : en dessous de 1, la distribution se concentre sur les tokens les plus probables, et à la limite on prend toujours le plus probable ; au-dessus de 1, elle s'aplatit, et des tokens rares passent plus souvent. Elle ne change jamais l'ordre des tokens, seulement l'écart entre leurs probabilités. Pour extraire des informations d'un contrat, je veux une réponse fidèle et reproductible : une température nulle ou très basse. Pour un slogan, je veux de la variété : une température autour de 1, et plusieurs propositions parmi lesquelles choisir. On la combine souvent avec un top-k ou un top-p, qui coupent les tokens trop improbables, pour garder de la variété sans incohérences. »
**Ce qui compte** : la division des logits avant la softmax ; l'effet des deux extrêmes ; l'ordre conservé ; deux choix argumentés ; top-k et top-p ; le prix d'une température basse (répétitions) et d'une température haute (erreurs, incohérences).
**Relances possibles** : « Avec une température nulle, la sortie est-elle toujours identique ? » (en principe oui, on prend l'argmax ; en pratique, de petites différences numériques entre exécutions sur GPU, ou des ex aequo, peuvent la faire varier) · « Quelle différence entre top-k et top-p ? » (top-k garde les $k$ tokens les plus probables ; top-p garde le plus petit ensemble dont la probabilité cumulée atteint $p$, qui s'adapte : peu de tokens quand le modèle est sûr, beaucoup quand il hésite) · « Pourquoi le décodage glouton se répète-t-il ? » (il est déterministe : dès qu'une suite revient, ce qui suit revient aussi ; et une répétition rend la suivante plus probable).

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes, exécutées, sont dans `05_solutions.ipynb` ; voici, pour chaque exercice, la réponse, la démarche, les erreurs fréquentes et une variante. Les chiffres sont ceux du mode rapide, sur CPU. Les fonctions de ta librairie sont données en entier. Celles de `sequences.py` partagent deux petits outils :

```python
import numbers
from collections import Counter

import numpy as np


def _positive_int(value, name):
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _series(values, name):
    arr = np.asarray(values)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must have shape (T,) or (T, D), got shape {arr.shape}")
    return arr
```

et celles de `rnn.py`, cinq autres :

```python
def _array(value, name, ndim):
    arr = np.array(value, dtype=np.float64, copy=True)          # a new float64 array: the input is never modified
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    return arr


def _weights(W_ih, W_hh, b_ih, b_hh, d, h, gates):
    w_ih, w_hh = _array(W_ih, "W_ih", 2), _array(W_hh, "W_hh", 2)
    if w_ih.shape != (d, gates * h):
        raise ValueError(f"W_ih must have shape (D, {gates}H) = {(d, gates * h)}, got {w_ih.shape}")
    if w_hh.shape != (h, gates * h):
        raise ValueError(f"W_hh must have shape (H, {gates}H) = {(h, gates * h)}, got {w_hh.shape}")
    biases = []
    for name, b in (("b_ih", b_ih), ("b_hh", b_hh)):
        arr = np.zeros(gates * h) if b is None else np.array(b, dtype=np.float64, copy=True)
        if arr.shape != (gates * h,):
            raise ValueError(f"{name} must have shape ({gates * h},), got {arr.shape}")
        biases.append(arr)
    return w_ih, w_hh, biases[0], biases[1]


def _inputs(x_t, h_prev):
    x, h = _array(x_t, "x_t", 2), _array(h_prev, "h_prev", 2)
    if x.shape[0] != h.shape[0]:
        raise ValueError(f"x_t and h_prev must have the same number of rows, got {x.shape[0]} and {h.shape[0]}")
    return x, h


def _sigmoid(z):
    e = np.exp(-np.abs(z))                                      # never an overflow: the exponent is <= 0
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _sequence(x, h0, name="h0"):
    seq = _array(x, "x", 3)
    if seq.shape[1] < 1:
        raise ValueError("x must have at least one time step (T >= 1)")
    state = None if h0 is None else _array(h0, name, 2)
    if state is not None and state.shape[0] != seq.shape[0]:
        raise ValueError(f"{name} must have N = {seq.shape[0]} rows, got {state.shape[0]}")
    return seq, state
```

Les vérifications de forme de `_weights` sont ce qui refuse la disposition de PyTorch et les poids qu'on aurait diffusés sans erreur : NumPy ne le fait pas pour toi.

### Ex 22.12 — build_vocab, encode et decode sur Holmes et Verne 🔨
**Réponses** : a) **88** caractères distincts ; b) l'id de « e » vaut **1** (l'espace a l'id 0) ; c) « Watson » s'encode **`[33, 3, 2, 7, 4, 5]`** ; d) **0,1348 (13,5 %)** des mots de Verne deviennent `<unk>` ; e) le mot d'id 1 est **« de »**. Les 32 tests de l'exercice passent.
**Démarche** :
```python
def build_vocab(tokens, max_size=None, min_freq=1, unk_token=None):
    if max_size is not None:
        max_size = _positive_int(max_size, "max_size")
    min_freq = _positive_int(min_freq, "min_freq")
    counts = Counter(tokens)                                    # a str counts its characters
    if not counts:
        raise ValueError("tokens is empty: nothing to build a vocabulary from")
    if "" in counts:
        raise ValueError("tokens contains an empty token ''")
    ranked = sorted((tok for tok, cnt in counts.items() if cnt >= min_freq and tok != unk_token),
                    key=lambda tok: (-counts[tok], tok))       # most frequent first, then the order of sorted
    itos = ([unk_token] if unk_token is not None else []) + ranked
    if max_size is not None:
        itos = itos[:max_size]                                  # unk_token counts in max_size
    return {tok: i for i, tok in enumerate(itos)}, itos


def encode(tokens, stoi, unk_token=None):
    if unk_token is not None and unk_token not in stoi:
        raise ValueError(f"unk_token {unk_token!r} is not a key of stoi")
    ids = []
    for tok in tokens:
        if tok in stoi:
            ids.append(stoi[tok])
        elif unk_token is not None:
            ids.append(stoi[unk_token])
        else:
            raise ValueError(f"token {tok!r} is not in the vocabulary (pass unk_token to map it)")
    return np.array(ids, dtype=np.int64)


def decode(ids, itos, sep=""):
    arr = np.asarray(ids)
    if arr.ndim != 1:
        raise ValueError(f"ids must be 1-D, got shape {arr.shape}")
    if arr.size and (arr.dtype == bool or not np.issubdtype(arr.dtype, np.integer)):
        raise ValueError(f"ids must be integers, got dtype {arr.dtype}")
    if arr.size and ((arr < 0).any() or (arr >= len(itos)).any()):
        raise ValueError(f"ids must be in [0, {len(itos)}), got min {arr.min()} and max {arr.max()}")
    return sep.join(itos[int(i)] for i in arr)
```
La clé de tri `(-effectif, token)` fait tout le classement en un seul `sorted`. `np.array([], dtype=np.int64)` a bien la forme `(0,)` pour un texte vide.
**Dans tes notes** : avec 2 000 mots, environ un mot sur sept du roman devient inconnu, des noms propres rares et toutes les formes conjuguées peu fréquentes ; avec les caractères, rien n'est inconnu, mais les séquences sont près de six fois plus longues. Les modèles actuels découpent en **sous-mots** (BPE, WordPiece) : les mots fréquents restent entiers, les rares se coupent en morceaux connus (ch. 24 et 25).
**Erreurs fréquentes** : rendre `(itos, stoi)` au lieu de `(stoi, itos)` ; trier par effectif seulement (l'ordre des ex aequo dépend alors de l'ordre d'apparition) ; ajouter `<unk>` **après** avoir coupé à `max_size` (2 001 entrées) ; laisser `stoi[tok]` lever une `KeyError` ; `decode` qui accepte `-1` (Python lit le dernier token) ou `1.7` (que `int()` tronquerait).
**Variante** : `build_vocab("abracadabra")` donne `itos = ['a', 'b', 'r', 'c', 'd']` (5 fois « a », 2 fois « b » et « r » dans l'ordre de `sorted`, puis « c » et « d ») ; avec `max_size=3, unk_token="?"`, **`['?', 'a', 'b']`**, et `encode("bar", stoi, "?")` vaut **[2, 1, 0]**.

### Ex 22.13 — make_windows et make_sequence_pairs 🔨
**Réponses** : les fenêtres des taches solaires sont celles de `sliding_window_view`, les cibles à pas de 12 et les 25 morceaux de `np.arange(1001)` sont justes, les trois refus passent ; les 39 tests de l'exercice passent.
**Démarche** :
```python
def make_windows(series, window, horizon=1, stride=1):
    w = _positive_int(window, "window")
    h = _positive_int(horizon, "horizon")
    s = _positive_int(stride, "stride")
    arr = _series(series, "series")
    t = arr.shape[0]
    n = (t - w - h) // s + 1 if t - w - h >= 0 else 0
    if n < 1:
        raise ValueError(f"a series of {t} steps is too short for one window of {w} steps and a horizon of {h}")
    starts = np.arange(n) * s
    X = arr.reshape(t, -1)[starts[:, None] + np.arange(w)[None, :]]     # fancy indexing: a new array (N, w, D)
    y = arr[starts + w - 1 + h].copy()
    return X, y


def make_sequence_pairs(seq, seq_len, stride=None):
    length = _positive_int(seq_len, "seq_len")
    s = length if stride is None else _positive_int(stride, "stride")
    arr = _series(seq, "seq")
    t = arr.shape[0]
    n = (t - 1 - length) // s + 1 if t - 1 - length >= 0 else 0
    if n < 1:
        raise ValueError(f"a sequence of {t} steps is too short for one chunk of {length} steps and its target")
    idx = np.arange(n)[:, None] * s + np.arange(length)[None, :]        # (N, L)
    return arr[idx], arr[idx + 1]
```
Une table d'indices `(N, w)` remplace toute boucle ; l'indexation par un tableau d'entiers rend toujours une **copie**, du dtype de la série. Une série trop courte donne $N \leq 0$ (`//` arrondit vers le bas), que refuse le test `n < 1` ; la condition `t - w - h >= 0` ne fait que le dire plus clairement.
**Dans tes notes** : des vues partageraient la mémoire de la série, et des fenêtres qui se recouvrent partagent les mêmes cases : normaliser `X` sur place modifierait la série et plusieurs fois les mêmes valeurs. `make_sequence_pairs` donne 40 cibles par morceau (une à chaque pas), `make_windows` une seule par fenêtre.
**Erreurs fréquentes** : la cible à `start + w + h` (un pas trop loin) ; `X` de forme `(N, w)` pour une série 1-D (le contrat veut `(N, w, 1)`) ; `np.lib.stride_tricks.sliding_window_view` rendu tel quel (une vue en lecture seule) ; `stride=None` traité comme 1 ; oublier qu'il faut $L + 1$ valeurs pour un morceau et sa suite.
**Variante** : 3 327 mois, des fenêtres de 132 mois (11 ans), un horizon de 12 et un pas de 6 : $\lfloor (3\,327 - 132 - 12)/6 \rfloor + 1 =$ **531** exemples ; `make_sequence_pairs(np.arange(10), 3, stride=2)` donne **4** morceaux, dont le dernier est `[6, 7, 8]` avec la suite `[7, 8, 9]`.

### Ex 22.14 — Un MLP sur des fenêtres : l'ordre compte-t-il ? 🔮
**Réponses** : a) **C** ; b) **True** ; c) **False**. Les erreurs de test : persistance 0,0528, constante 0,5109 ; variante 1 0,0190, 2 0,0167, 3 0,4484, 4 0,0385.
**Démarche** : l'expérience, exécutée, donne le tableau. Les variantes 1 et 2 font presque la même erreur : une permutation fixe des entrées ne change rien pour une couche dense, qui apprend les poids dans le nouvel ordre (fiche §22.2). La variante 3 est environ 8,5 fois moins bonne que la persistance et à peine meilleure que la constante : un ensemble de huit valeurs sans ordre donne la hauteur de la série, pas son sens ; deux tronçons symétriques du sinus ont le même ensemble de valeurs et deux suites opposées, et le réseau ne peut viser que leur moyenne. La variante 4, avec deux valeurs ordonnées, connaît la pente et bat la persistance.
**Erreurs fréquentes** : confondre les variantes 2 et 3 (une même permutation pour toutes les fenêtres ne détruit rien) ; penser que plus de valeurs aident toujours (huit en vrac font pire que deux dans l'ordre) ; juger la variante 3 à la persistance seulement : face à la constante, elle n'a presque rien appris.
**Variante** : avec une fenêtre de 24 (une période entière), l'ensemble des valeurs d'une fenêtre mélangée est presque le même partout : la variante 3 ne peut plus rien dire de la position dans le cycle : elle fait même **pire que la constante** (environ deux fois son erreur lors de nos essais, le réseau ajustant le bruit de ses 24 entrées), alors que la variante 1 reste bien meilleure que la persistance.

### Ex 22.15 — rnn_cell_forward et rnn_forward 🔨
**Réponses** : les sorties de `nn.RNN` et de `nn.RNNCell` (ReLU) sont retrouvées au milliardième près, les deux refus passent ; les 48 tests de l'exercice passent.
**Démarche** :
```python
def rnn_cell_forward(x_t, h_prev, W_ih, W_hh, b_ih=None, b_hh=None, nonlinearity="tanh"):
    if nonlinearity not in ("tanh", "relu"):
        raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
    x, h = _inputs(x_t, h_prev)
    w_ih, w_hh, bi, bh = _weights(W_ih, W_hh, b_ih, b_hh, x.shape[1], h.shape[1], 1)
    z = x @ w_ih + bi + h @ w_hh + bh
    return np.tanh(z) if nonlinearity == "tanh" else np.maximum(z, 0.0)


def rnn_forward(x, W_ih, W_hh, b_ih=None, b_hh=None, h0=None, nonlinearity="tanh"):
    seq, h = _sequence(x, h0)
    n, t, _ = seq.shape
    if h is None:
        h = np.zeros((n, np.shape(W_hh)[0] if np.ndim(W_hh) == 2 else 0))   # a bad W_hh is refused by the cell
    outputs = []
    for step in range(t):
        h = rnn_cell_forward(seq[:, step], h, W_ih, W_hh, b_ih, b_hh, nonlinearity)
        outputs.append(h)
    stacked = np.stack(outputs, axis=1)                         # (N, T, H)
    return stacked, stacked[:, -1].copy()                      # h_T: an array of its own
```
La boucle ne fait qu'appeler la cellule, qui vérifie tout ; `np.stack(..., axis=1)` range les états au format batch-first.
**Erreurs fréquentes** : `W_ih.T @ x` (la disposition de PyTorch) ; additionner un seul biais (le contrat en a deux, comme PyTorch, dont la somme compte) ; rendre `stacked[:, -1]`, une vue de `outputs` ; `np.stack(outputs)` sans `axis=1`, qui donne `(T, N, H)` ; accepter `nonlinearity="sigmoid"` en passant par un dictionnaire de fonctions sans vérifier la clé.
**Variante** : une cellule scalaire avec $W_{ih} = 1$, $W_{hh} = 0{,}5$, sans biais et en ReLU, sur les entrées $1, -3, 2$ en partant de 0 : $h_1 = 1$, $h_2 = \max(0, -3 + 0{,}5) = 0$, $h_3 = \max(0, 2 + 0) =$ **2**.

### Ex 22.16 — lstm_cell_forward et lstm_forward (portes i, f, g, o) 🔨
**Réponses** : les sorties, $h_T$ et $c_T$ de `nn.LSTM` et le pas de `nn.LSTMCell` sont retrouvés, le refus passe ; les 49 tests de l'exercice passent.
**Démarche** :
```python
def lstm_cell_forward(x_t, h_prev, c_prev, W_ih, W_hh, b_ih=None, b_hh=None):
    x, h = _inputs(x_t, h_prev)
    c = _array(c_prev, "c_prev", 2)
    if c.shape != h.shape:
        raise ValueError(f"c_prev must have the shape of h_prev {h.shape}, got {c.shape}")
    hid = h.shape[1]
    w_ih, w_hh, bi, bh = _weights(W_ih, W_hh, b_ih, b_hh, x.shape[1], hid, 4)
    a = x @ w_ih + bi + h @ w_hh + bh                           # (N, 4H): blocks [i | f | g | o]
    i = _sigmoid(a[:, :hid])
    f = _sigmoid(a[:, hid:2 * hid])
    g = np.tanh(a[:, 2 * hid:3 * hid])
    o = _sigmoid(a[:, 3 * hid:])
    c_t = f * c + i * g
    return o * np.tanh(c_t), c_t


def lstm_forward(x, W_ih, W_hh, b_ih=None, b_hh=None, h0=None, c0=None):
    seq, h = _sequence(x, h0)
    n, t, _ = seq.shape
    hid = np.shape(W_hh)[0] if np.ndim(W_hh) == 2 else 0
    h = np.zeros((n, hid)) if h is None else h
    c = np.zeros((n, hid)) if c0 is None else _array(c0, "c0", 2)
    outputs = []
    for step in range(t):
        h, c = lstm_cell_forward(seq[:, step], h, c, W_ih, W_hh, b_ih, b_hh)
        outputs.append(h)
    stacked = np.stack(outputs, axis=1)
    return stacked, (stacked[:, -1].copy(), c)                  # a tuple, as nn.LSTM
```
Un seul produit par poids donne les quatre portes ; les tranches `a[:, k*H:(k+1)*H]` suivent l'ordre $i, f, g, o$ de PyTorch. `c` est déjà un nouveau tableau à chaque pas : il peut être rendu tel quel.
**Dans tes notes** : `outputs[:, -1]` vaut `h_T` (et $h_T = o_T \odot \tanh(c_T)$) : pour une seule couche, `h_n[0]` est redondant avec la sortie du dernier pas. Sans la tanh de $h_t$, la sortie ne serait plus bornée : $c_t$ peut croître sur des centaines de pas, et $h_t$ avec lui.
**Erreurs fréquentes** : l'ordre $i, f, o, g$ de certains manuels (ou $i, g, f, o$) au lieu de celui de PyTorch ; une sigmoïde `1 / (1 + np.exp(-z))` qui déborde pour $z = -1\,000$ ; rendre `(c_t, h_t)` ; rendre `[h_T, c_T]` (une liste) ; partir de `c0 = h0` quand seul `h0` est donné.
**Variante** : $H = 1$, toutes les pré-activations nulles, $c_{t-1} = 1$ : $i = f = o = 0{,}5$, $g = 0$, donc $c_t = 0{,}5$ et $h_t = 0{,}5 \tanh(0{,}5) \approx$ **0,231**.

### Ex 22.17 — gru_cell_forward (portes r, z, n) 🔨
**Réponses** : la cellule retrouve `nn.GRUCell`, et le cas où seul $b_{hn}$ vaut 3 donne $0{,}5 \tanh(1{,}5) + 0{,}5 \approx 0{,}953$ ; le refus passe ; les 24 tests de l'exercice passent.
**Démarche** :
```python
def gru_cell_forward(x_t, h_prev, W_ih, W_hh, b_ih=None, b_hh=None):
    x, h = _inputs(x_t, h_prev)
    hid = h.shape[1]
    w_ih, w_hh, bi, bh = _weights(W_ih, W_hh, b_ih, b_hh, x.shape[1], hid, 3)
    gi = x @ w_ih + bi                                          # (N, 3H): blocks [r | z | n]
    gh = h @ w_hh + bh
    r = _sigmoid(gi[:, :hid] + gh[:, :hid])
    z = _sigmoid(gi[:, hid:2 * hid] + gh[:, hid:2 * hid])
    n = np.tanh(gi[:, 2 * hid:] + r * gh[:, 2 * hid:])          # r multiplies h_prev @ W_hn + b_hn
    return (1.0 - z) * n + z * h
```
Le cas de vérification : $x = 0$, $h = 1$, poids nuls, $b_{hn} = 3$ ; donc $r = z = 0{,}5$, $n = \tanh(0{,}5 \times 3)$, et $h_t = 0{,}5\,n + 0{,}5 \times 1$.
**Erreurs fréquentes** : additionner `gi + gh` avant de découper (tout le terme $h_{t-1} W_{hn} + b_{hn}$ perd le facteur $r$, et `nn.GRUCell` n'est pas retrouvée) ; fusionner les deux biais en un seul $b_{ih} + b_{hh}$ ajouté avant la porte (seul $b_{hn}$ perd le facteur $r$ : une erreur invisible quand les biais sont nuls, que le second cas de vérification attrape) ; $h_t = z\,n + (1 - z)\,h$ (la convention inverse, celle de certains articles) ; les blocs dans l'ordre $z, r, n$.
**Variante** : avec $z = 1$ partout (un biais de $z$ très grand), le GRU **recopie** son état : $h_t = h_{t-1}$, quelle que soit l'entrée. C'est son chemin direct, l'équivalent d'une porte d'oubli ouverte et d'une porte d'entrée fermée.

### Ex 22.18 — Benchmark : ta cellule NumPy contre nn.LSTM 🛠️
**Réponse** : sur la machine de génération (deux cœurs), ta `lstm_forward` met environ 20 ms et `nn.LSTM` 7 ms, un rapport de 2,7 à 2,8 selon l'exécution ; le rapport varie de quelques dizaines de pour cent d'une exécution à l'autre.
**Démarche** :
```python
def best_time_18(fn, *args, repeat=5):
    fn(*args)                                       # warm-up: caches, memory allocation, lazy initialisation
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args)
        durations.append(time.perf_counter() - start)
    return float(min(durations))
```
Le faux calcul dort 0,03 s à la chauffe, puis 0,02, 0,01, 0,025, 0,015 et 0,02 s : la plus courte mesure, environ 0,010 s, est la seule réponse juste (la chauffe, plus longue, n'est pas comptée ; la moyenne vaudrait 0,018 s). Par pas, ta boucle fait deux produits matriciels, quatre tranches, trois sigmoïdes, deux tanh, et alloue une dizaine de tableaux ; l'interpréteur Python exécute tout cela cent fois. `nn.LSTM` fait le même calcul dans une boucle compilée.
**Erreurs fréquentes** : mesurer aussi la chauffe ; rendre la moyenne (elle inclut les interruptions) ; `time.time()`, moins précis ; appeler `fn(args)` au lieu de `fn(*args)` ; mesurer `nn.LSTM` avec l'autograd actif (le graphe coûte du temps et de la mémoire : `torch.no_grad()`).
**Variante** : `best_time_18(sum, range(10**6), repeat=3)` appelle `sum` **4** fois ; pour comparer à `nn.LSTM` en `float32` (son format habituel), il faut aussi convertir les entrées : le rapport grandit encore, NumPy calculant toujours en `float64` ici.

### Ex 22.19 — Mesurer la disparition du gradient : RNN tanh contre LSTM 🔬
**Réponses** : ta fonction retrouve la jacobienne de PyTorch et laisse `x` intact. Les rapports en $\log_{10}$ à 10, 40 et 79 pas : RNN par défaut −2,1, −10,3 et −19,5 ; RNN à $W_{hh}$ orthogonale +0,2, −1,4 et −3,1 ; LSTM par défaut −3,0, −8,2 et −15,6 ; LSTM au biais d'oubli de 1 −1,7, −3,4 et −6,3 (graine 2219).
**Démarche** :
```python
def input_grad_norms_19(model, x):
    inputs = x.detach().clone().requires_grad_(True)      # a leaf of its own: x keeps no gradient
    output, _ = model(inputs)
    output[:, -1].sum().backward()
    return inputs.grad.norm(dim=(0, 2)).numpy()           # one norm per time step, over the batch and the features
```
Le gradient de $L$ par rapport à $x_t$ passe par $h_t, h_{t+1}, \dots, h_T$ : un produit de $T - t$ jacobiennes (∂ 22.6). Pour le RNN par défaut, chaque facteur est en moyenne bien plus petit que 1 (les poids de PyTorch sont tirés dans $[-1/\sqrt{H}, 1/\sqrt{H}]$, et la tanh réduit encore) : le gradient perd un facteur 10 environ tous les 4 pas. Le LSTM par défaut a des portes d'oubli proches de $\sigma(0) = 0{,}5$, et son chemin additif (∂ 22.7) divise donc lui aussi le gradient par environ 2 à chaque pas. Un biais d'oubli de 1 porte $f$ vers 0,73 et adoucit nettement la pente. $W_{hh}$ orthogonale garde la norme de chaque facteur près de 1 (seule la tanh la réduit) : c'est la courbe la plus plate, au départ.
**Dans tes notes** : le gradient divisé par $10^6$ en moins de 25 pas pour le RNN par défaut ; le LSTM par défaut fait à peine mieux, pour la raison donnée plus haut ; le biais d'oubli est le réglage qui change tout pour le LSTM. L'orthogonalité ne dure pas : l'entraînement déforme $W_{hh}$ et la tanh sature, alors que les portes du LSTM **apprennent** à rester ouvertes quand une information doit traverser le temps.
**Erreurs fréquentes** : `x.requires_grad_(True)` sur l'entrée elle-même (le contrat l'interdit, et un second appel accumule les gradients) ; une norme sur tout le tenseur (un seul nombre) ; oublier `[0]` (les modules récurrents rendent un couple) ; prendre `output.sum()` (la loss dépendrait alors de chaque pas, et le gradient d'un pas ancien serait dominé par sa propre sortie).
**Variante** : avec $T = 20$ au lieu de 80, la courbe du RNN par défaut s'arrête vers $10^{-4}$ ou $10^{-5}$ : un RNN simple apprend sans peine des dépendances de quelques pas, pas de quelques dizaines.

### Ex 22.20 — temperature_softmax et sample_from_logits 🔨
**Réponses** : la softmax retrouve celle de SciPy sans avertissement, le cas glouton, les fréquences, le top-k avec ex aequo, la reproductibilité et les trois refus passent ; les 44 tests de l'exercice passent.
**Démarche** :
```python
def temperature_softmax(logits, temperature=1.0):
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    z = np.array(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)                      # no overflow: the largest exponent is 0
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sample_from_logits(logits, temperature=1.0, top_k=None, rng=None):
    if not temperature >= 0:
        raise ValueError(f"temperature must be >= 0, got {temperature!r}")
    z = np.array(logits, dtype=np.float64)
    if z.ndim not in (1, 2) or z.shape[-1] < 1:
        raise ValueError(f"logits must have shape (V,) or (N, V), got {z.shape}")
    v = z.shape[-1]
    if top_k is not None:
        k = _positive_int(top_k, "top_k")
        if k > v:
            raise ValueError(f"top_k must be <= V = {v}, got {k}")
        kth = np.sort(z, axis=-1)[..., v - k]                    # the k-th largest logit of each row
        z = np.where(z < kth[..., None], -np.inf, z)            # ties with it are kept
    if temperature == 0:
        ids = np.argmax(z, axis=-1)                             # greedy decoding: the first maximum
    else:
        gen = np.random.default_rng(rng)                        # a Generator is returned as it is
        cum = np.cumsum(temperature_softmax(z, temperature), axis=-1)
        u = gen.random(z.shape[:-1] + (1,))                     # one uniform per row
        ids = np.minimum((u * cum[..., -1:] >= cum).sum(axis=-1), v - 1)   # inverse of the cumulative sum
    return int(ids) if z.ndim == 1 else ids.astype(np.int64)
```
`np.random.default_rng(rng)` rend `rng` lui-même quand c'est déjà un `Generator`, et un nouveau générateur sans graine quand c'est `None`. L'id tiré est le nombre de valeurs cumulées que l'uniforme atteint : avec $p = (0{,}2 ; 0{,}5 ; 0{,}3)$, un uniforme de 0,6 dépasse 0,2 mais pas 0,7, d'où l'id 1. Multiplier `u` par le dernier cumul protège des arrondis, et `np.minimum` du cas extrême.
**Erreurs fréquentes** : retirer le maximum global au lieu de celui de chaque ligne (les lignes de ±1 000) ; `rng.choice(V, p=p)` ligne par ligne sur 20 000 lignes (juste, mais lent) ; `np.argsort(z)[-k:]` pour le top-k (il perd les ex aequo du $k$-ième) ; `temperature_softmax(z, 0)` pour le cas glouton (division par zéro) ; rendre un `numpy.int64` au lieu d'un `int`.
**Variante** : logits $(2, 1, 0)$ et `top_k=2` à la température 1 : le dernier est retiré, et l'id 0 sort avec la probabilité $e^2/(e^2 + e) = 1/(1 + e^{-1}) \approx$ **0,731**.

### Ex 22.21 — Série temporelle piégée : fuite, mélange et batch_first 🐛
**Réponse** : quatre corrections.
```python
def prepare_21(series, window=24, horizon=6, n_test=240):
    lo, hi = series[:-n_test].min(), series[:-n_test].max()     # 3. the scaler sees the training period only
    scaled = (series - lo) / (hi - lo)
    blocks = np.lib.stride_tricks.sliding_window_view(scaled, window + horizon)
    X = blocks[:, :window]                                       # 1. the target stays outside the inputs
    y = blocks[:, -1]
    cut = len(y) - n_test                                        # 2. a cut in time, no shuffling
    X_train, X_test, y_train, y_test = X[:cut], X[cut:], y[:cut], y[cut:]
    ...


class Forecaster21(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)    # 4. the windows arrive as (N, T, 1)
        ...
```
**Démarche** : le pipeline fautif annonce une RMSE de test de quelques taches et une persistance **nulle** ; puis les diagnostics, dans l'ordre. 1) Sur la rampe, la cible d'une fenêtre est exactement sa dernière valeur : `blocks[:, -window:]` prend les **dernières** colonnes, cible comprise. 2) Les cibles de test ne sont plus les 40 dernières : `train_test_split` mélange par défaut, et chaque fenêtre de test est entourée de fenêtres d'entraînement presque identiques. 3) Le scaler vaut `(0, 399)` : il a vu le test. 4) Changer la fenêtre 0 change toutes les prédictions : sans `batch_first`, le LSTM lit `(N, T, 1)` comme `(T, N, 1)`, c'est-à-dire l'axe des exemples comme le temps. Une fois tout corrigé : le petit LSTM fait une RMSE de test de 23,0 taches, contre 26,1 pour la persistance, un gain d'une dizaine de pour cent.
**Dans tes notes** : le bug 1 seul rend la persistance exacte (la dernière entrée **est** la cible) ; aucun des quatre ne lève d'erreur, puisque les formes passent toujours ; le scaler ne change presque rien aux scores affichés et ne se voit que dans le diagnostic 3, alors que le bug 4 empire le modèle sans rien signaler ; corrigé, le petit LSTM ne bat la persistance que d'une dizaine de pour cent : 24 mois et 15 epochs ne suffisent pas pour mieux (22.23 fait mieux avec 60 mois).
**Erreurs fréquentes** : corriger le découpage avec `shuffle=False` dans `train_test_split` sans recalculer le scaler ; prendre le scaler sur `X_train` (des fenêtres qui se recouvrent : le résultat est le même ici, mais le principe est de le prendre sur la période) ; ajouter `x.transpose(0, 1)` dans `forward` au lieu de `batch_first=True` (l'entrée est alors bien lue, mais la sortie est elle aussi `(T, N, H)` : il faut lire `out[-1]`, pas `out[:, -1]`) ; croire le premier score parce qu'il est bon.
**Variante** : avec `window=24, horizon=6` sur une série de 400 valeurs, il y a $400 - 30 + 1 = 371$ fenêtres, dont **331** d'entraînement pour `n_test=40` ; la première fenêtre de test commence au pas 331 et lit les pas 331 à 354, tous avant sa cible 360 : elle lit la fin de la période d'entraînement, ce que le contrat autorise.

### Ex 22.22 — RNN profonds et bidirectionnels : formes de output et h_n 📦
**Réponses** : a) **(4, 7, 40)** ; b) **(6, 4, 20)** ; c) **24 960** paramètres ; d) **4** ; e) **0**. `final_state_22` et `lstm_params_22` retrouvent PyTorch.
**Démarche** :
```python
def final_state_22(h_n, num_layers, bidirectional):
    if not bidirectional:
        return h_n[-1]
    last = h_n.view(num_layers, 2, *h_n.shape[1:])[-1]     # (2, N, H): forward, then backward
    return torch.cat([last[0], last[1]], dim=1)


def lstm_params_22(input_size, hidden_size, num_layers=1, bidirectional=False):
    directions = 2 if bidirectional else 1
    total = 0
    for layer in range(num_layers):
        d = input_size if layer == 0 else directions * hidden_size      # a layer reads the outputs of the one below
        total += directions * 4 * (d * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
    return total
```
`output` ne garde que la dernière couche, ses deux sens concaténés à chaque pas : $(N, T, 2H)$. `h_n` garde l'état final de chaque couche et de chaque sens, rangés couche par couche puis sens par sens : $(L \cdot 2, N, H)$, sans `batch_first`. L'état final du sens avant de la couche 2 (la troisième) est à l'indice $2 \times 2 + 0 = 4$, celui du sens arrière à 5 ; le sens arrière lit la séquence à l'envers et finit au pas 0. Paramètres : $2 \times 4\,(10 \cdot 20 + 400 + 40) = 5\,120$ pour la première couche, $2 \times 4\,(40 \cdot 20 + 400 + 40) = 9\,920$ pour chacune des deux autres.
**Dans tes notes** : `output[:, -1]` colle l'état final du sens avant et le **premier** état du sens arrière, qui n'a lu qu'un pas ; un réseau à un seul sens aurait $4\,(200 + 400 + 40) + 2 \times 4\,(400 + 400 + 40) =$ **9 280** paramètres.
**Erreurs fréquentes** : `(4, 6, 20)` pour `h_n` (`batch_first` ne s'applique pas aux états) ; `D = H` pour les couches 2 et 3 d'un réseau bidirectionnel ; un seul biais par porte ; `h_n[-1]` pour un réseau bidirectionnel (le sens arrière seul) ; `h_n[-2:].reshape(N, 2H)` (mélange les exemples : il faut concaténer sur l'axe des features).
**Variante** : `nn.GRU(8, 16, num_layers=2, bidirectional=True)` a $2 \times 3\,(8 \cdot 16 + 256 + 32) + 2 \times 3\,(32 \cdot 16 + 256 + 32) = 2\,496 + 4\,800 =$ **7 296** paramètres, et son `h_n` a la forme **(4, N, 16)**.

### Ex 22.23 — Prévoir les taches solaires avec nn.LSTM, face à la persistance 📦
**Réponses** : a) **2 779** fenêtres d'entraînement ; b) une persistance de **41,1** taches. Le LSTM : une RMSE de validation de 31,6 taches, soit 0,77 de la persistance.
**Démarche** :
```python
scale_23 = SPOTS[:I86].max()
X_train_23, y_train_23 = mylearn.sequences.make_windows(SPOTS[:I86] / scale_23, 60, horizon=6)
X_val_23, y_val_23 = mylearn.sequences.make_windows(SPOTS[I86 - 65:I06] / scale_23, 60, horizon=6)
n_train_23 = len(X_train_23)
persistence_23 = float(np.sqrt(np.mean((SPOTS[I86 - 6:I06 - 6] - SPOTS[I86:I06]) ** 2)))


class SunspotLSTM_23(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)                # many-to-one: the last step only


def fit_23(model, X, y, epochs=20, lr=3e-3, batch_size=64, seed=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        total = 0.0
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X[batch]), y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(X))
    return history
```
$N = \lfloor (2\,844 - 60 - 6)/1 \rfloor + 1 = 2\,779$ fenêtres : `I86` vaut 2 844 (janvier 1986 est le 2 845ᵉ mois depuis janvier 1749). La première cible de validation, `SPOTS[I86]`, a sa fenêtre qui finit 6 mois plus tôt et commence 65 mois plus tôt : d'où la tranche `SPOTS[I86 - 65:I06]`, qui donne exactement 240 fenêtres. Le scaler, calculé avant 1986, ne voit rien de la validation.
**Dans tes notes** : la figure le montre, le LSTM gagne surtout dans les montées et les descentes du cycle, où la valeur d'il y a 6 mois est la plus fausse ; au sommet, le bruit mensuel domine et les deux se valent. Choisir `hidden_size` ou les epochs en regardant cette validation, puis annoncer sa RMSE, c'est la même fuite que choisir un modèle sur le test : le score annoncé serait optimiste. D'où le test caché du défi.
**Erreurs fréquentes** : des fenêtres de validation prises dans `SPOTS[I86:I06]` (il en manque 65, et les premières cibles tombent en 1991) ; `scale_23` calculé sur toute la série ; la persistance à 1 mois (`SPOTS[I86 - 1:I06 - 1]`), bien meilleure, qui rend le LSTM mauvais en comparaison ; un `torch.Generator` recréé à chaque epoch (le même ordre à chaque fois) ; `history.append(loss.item())` (la loss du dernier mini-batch seulement) ; oublier `.squeeze(-1)` (une MSE entre `(N, 1)` et `(N,)` diffuse en `(N, N)`, avec un simple avertissement).
**Variante** : avec une fenêtre de 132 mois (un cycle entier) et le même horizon, il y a $2\,844 - 132 - 6 + 1 =$ **2 707** fenêtres d'entraînement, et la tranche de validation devient `SPOTS[I86 - 137:I06]`.

### Ex 22.24 — Température : prédire l'allure du texte généré 🔮
**Réponses** : a) **A** ; b) **D** ; c) **B**. L'expérience donne : 5 caractères distincts à la température 0 (« The the the… »), 23 à 0,3 avec 79 % de mots qui existent, 65 à 1 et les 88 à 2.
**Démarche** : à la température 0, le caractère suivant est une fonction du caractère actuel : après « T » vient « h », puis « e », puis l'espace, puis « t », puis « h »… La suite entre dans la boucle « the the the » au bout de quelques caractères : une poignée de caractères distincts. À 2, les logits divisés par 2 : un caractère dont la probabilité était $10^{-4}$ de celle du plus probable en devient $10^{-2}$ ; sur 2 000 tirages, presque tous les caractères sortent. À 0,3, les écarts sont multipliés par 3,3 : le modèle suit surtout les paires fréquentes et produit beaucoup de petits mots courants (« the », « he », « a », « an »), mais aussi des fragments qui n'existent pas : il ne voit qu'un caractère.
**Dans tes notes** : un texte qui ressemble à de l'anglais demande une température basse sans être nulle, ici autour de 0,3 à 0,5 ; mais un modèle de paires ne peut pas écrire de mots longs : il lui faut plus de contexte, ce que donne la fenêtre de 40 caractères du LSTM (22.25).
**Erreurs fréquentes** : penser qu'à température 0 le texte reprend les mots les plus fréquents du livre dans l'ordre (il ne reprend que la chaîne des paires les plus fréquentes) ; croire qu'à 2 les caractères rares restent invisibles (sur 2 000 tirages, une probabilité de $10^{-3}$ sort deux fois en moyenne).
**Variante** : deux caractères dont les logits diffèrent de 1 (leurs probabilités sont dans un rapport $e \approx 2{,}72$) : à la température 0,5, le rapport devient $e^{2} \approx$ **7,39** ; à la température 2, $e^{0{,}5} \approx$ **1,65**. La température élève le rapport des probabilités à la puissance $1/T$.

### Ex 22.25 — Générer du Holmes caractère par caractère (deux LSTM, many-to-one) 📦
**Réponses** : 79 987 fenêtres d'entraînement et 18 727 de validation ; après 2 epochs (environ 75 s sur deux cœurs), une loss de validation de 1,754 nat par caractère, sous la cible de 2,1 ; à la température 0, le texte tourne en boucle (« that that that… »), à 0,5 il a l'allure de l'anglais.
**Démarche** :
```python
class CharLSTM_25(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, ids):
        out, _ = self.lstm(F.one_hot(ids, self.vocab_size).float())   # one-hot: (N, T) -> (N, T, V)
        return self.head(out[:, -1])                                   # many-to-one: the last step only


def train_25(model, X, y, epochs=2, batch_size=128, seed=0):
    device = next(model.parameters()).device
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(X), generator=generator).split(batch_size):
            xb, yb = X[batch].to(device), y[batch].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(X))
    return history


def generate_25(model, seed_text, n_chars, stoi, itos, temperature=1.0, rng=None):
    device = next(model.parameters()).device
    rng = np.random.default_rng() if rng is None else rng
    ids = list(mylearn.sequences.encode(seed_text, stoi))
    new = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_chars):
            window = torch.tensor([ids[-40:]], device=device)
            logits = model(window)[0].cpu().numpy().astype(np.float64)
            nxt = mylearn.sequences.sample_from_logits(logits, temperature=temperature, rng=rng)
            ids.append(nxt)
            new.append(nxt)
    return mylearn.sequences.decode(np.array(new, dtype=np.int64), itos)
```
Le modèle a $4\,(88 \cdot 128 + 128^2 + 256) + 4\,(128 \cdot 128 + 128^2 + 256) + 128 \cdot 88 + 88 = 111\,616 + 132\,096 + 11\,352 =$ 255 064 paramètres. Le one-hot remplace l'entrée par un vecteur de 88 valeurs dont une seule vaut 1 : le produit $x_t W_{ih}$ revient à choisir une ligne de $W_{ih}$, comme une couche `nn.Embedding` (ch. 24). Le réglage `alpha=0.9` est celui du code Keras du livre (`rho=0.9`) : avec la valeur par défaut de PyTorch (0,99) et ce taux de 0,01, la loss de validation remonte dès la deuxième epoch (de 2,2 à 3,2 nats lors de nos essais).
**Dans tes notes** : à la température 0, le texte tourne vite en boucle (une même suite de caractères ramène la même suite) ; vers 0,5, il a l'allure de Holmes, des mots et la ponctuation à leur place, sans suite logique ; à 1, il invente des mots. Une fenêtre tous les 3 caractères divise le temps d'une epoch par 3 : des fenêtres voisines partagent 37 caractères sur 40.
**Erreurs fréquentes** : appliquer une softmax dans `forward` (la cross-entropy en applique une seconde) ; lire `h_n` au lieu de `out[:, -1]` (juste aussi pour la dernière couche, mais `h_n[0]` serait la première) ; `torch.tensor(ids[-40:])` sans les crochets (une forme `(40,)`, sans axe des exemples) ; oublier `model.eval()` et `torch.no_grad()` dans la génération ; tirer avec des logits `float32` (les probabilités ne somment pas exactement à 1 pour certaines fonctions de tirage) ; rendre l'amorce avec le texte.
**Variante** : avec une fenêtre à chaque caractère (`stride=1`), il y a trois fois plus de fenêtres, **239 960** en mode rapide, et l'epoch dure trois fois plus longtemps pour un gain faible.

### Ex 22.26 — Lire la courbe de loss et la perplexité du générateur 📈
**Réponses** : a) **9** ; b) **4,71** ; c) **2,235** bits par caractère ; d) **22,41** ; e) **B**.
**Démarche** :
```python
best_epoch_26 = int(np.argmin(VAL_LOSS_26)) + 1               # the epochs count from 1
perplexity_26 = float(np.exp(min(VAL_LOSS_26)))                # the loss is in nats: e to the loss
bits_26 = min(VAL_LOSS_26) / np.log(2)
freqs_26 = np.array(list(Counter(HOLMES).values())) / len(HOLMES)
unigram_26 = float(np.exp(-(freqs_26 * np.log(freqs_26)).sum()))   # e to the entropy (nats), as for the network
```
La loss de validation la plus basse, 1,549 nat, est celle de l'epoch 9 ; elle oscille ensuite et remonte (1,63 à la 20ᵉ epoch, 1,66 à la 19ᵉ), alors que la loss d'entraînement baisse jusqu'au bout (de 1,325 à l'epoch 9 à 1,234) : c'est le cas B, le début de l'overfitting. $e^{1{,}549} \approx 4{,}71$ et $1{,}549/\ln 2 \approx 2{,}235$. Le modèle sans contexte a une entropie de 3,109 nats (4,49 bits) : $e^{3{,}109} \approx 22{,}4$.
**Dans tes notes** : un modèle uniforme sur les 88 caractères aurait une perplexité de 88 ; le modèle sans contexte, 22,41 ; le LSTM, 4,71 : le contexte de 40 caractères ramène le choix moyen d'une vingtaine de caractères à moins de cinq. On garderait les poids de l'epoch 9, ce que fait un early stopping sur la loss de validation (ch. 19). Les textes générés continuent parfois de « paraître » meilleurs après : le réseau recopie plus de morceaux du livre, ce qui n'est pas mieux généraliser.
**Erreurs fréquentes** : $2^{\text{loss}}$ pour la perplexité d'une loss en nats ; prendre la dernière epoch ; l'entropie en bits passée à l'exponentielle $e$ (ou l'inverse) ; un modèle sans contexte uniforme (88) au lieu des fréquences ; lire « B » comme un overfitting sans conséquence : c'est le moment où il faut s'arrêter.
**Variante** : une loss de validation de 1,0 nat par caractère donnerait une perplexité de $e \approx$ **2,72** et **1,443** bit par caractère.

### Ex 22.27 — Holmes contre Verne : même réseau, deux langues 🔬
**Réponses** : tes deux fonctions retrouvent les entropies exactes et la cross-entropy en bits. Le tableau : Holmes ($V = 88$) : 4,486 bits sans contexte, 3,389 puis 3,066 bits après les deux epochs, un gain de 1,42 bit ; Verne ($V = 103$) : 4,638 bits sans contexte, 3,200 puis 2,847, un gain de 1,79 bit.
**Démarche** :
```python
def unigram_bits_27(text):
    p = np.array(list(Counter(text).values()), dtype=float) / len(text)
    return float(-(p * np.log2(p)).sum())


def bits_per_char_27(model, X, y, batch_size=2000):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        total = sum(F.cross_entropy(model(X[i:i + batch_size].to(device)), y[i:i + batch_size].to(device),
                                    reduction="sum").item() for i in range(0, len(X), batch_size))
    return total / len(X) / np.log(2)                     # nats -> bits
```
Verne part de plus haut (son vocabulaire a 15 caractères de plus, les lettres accentuées, et ses fréquences sont un peu plus étalées) et arrive plus bas : le même réseau tire davantage du contexte en français, sur ce livre. Le roman de Verne suit une seule histoire, avec des noms qui reviennent à chaque page, et le français a des mots plus longs et des terminaisons très prévisibles (« -ment », « -tion », les accords) : une fois le début d'un mot lu, la suite coûte peu. Les textes générés à 0,5 restent pauvres dans les deux langues après deux epochs.
**Dans tes notes** : la comparaison n'est juste qu'à moitié : mêmes caractères d'entraînement, même réseau, même budget, mais pas les mêmes livres. Le roman de Verne suit une seule histoire, avec quelques noms qui reviennent sans cesse (Phileas Fogg, Passepartout, Fix) ; Holmes est un recueil de douze nouvelles, aux personnages et aux lieux qui changent. Pour une comparaison plus juste : plusieurs textes par langue, la même traduction d'un même livre (Verne en anglais, par exemple), plusieurs graines, et un modèle entraîné jusqu'au bout.
**Erreurs fréquentes** : une moyenne des moyennes de mini-batches dans `bits_per_char_27` (le dernier mini-batch, plus petit, pèse autant que les autres) ; `np.log` au lieu de `np.log2` dans l'entropie ; comparer la loss d'entraînement d'une langue à la loss de validation de l'autre ; conclure qu'une langue est « plus simple » à partir d'un seul livre.
**Variante** : un texte où chaque caractère d'un alphabet de 32 caractères serait tiré au hasard, uniformément, aurait une entropie sans contexte de **5** bits, et aucun réseau ne pourrait descendre en dessous : il n'y a rien à apprendre du contexte.

### Ex 22.28 — CNN-LSTM : la balle monte-t-elle ou descend-elle ? 📦
**Réponses** : **21 043** paramètres ; une accuracy de test de 0,98 ; 0,50 quand on mélange les images et 0,25 quand on les passe à l'envers.
**Démarche** :
```python
params_28 = (8 * 1 * 5 * 5 + 8) + 4 * (128 * 32 + 32 * 32 + 2 * 32) + (32 * 3 + 3)   # 208 + 20 736 + 99


class BallCNNLSTM_28(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(), nn.AdaptiveMaxPool2d(4), nn.Flatten())
        self.lstm = nn.LSTM(128, 32, batch_first=True)
        self.head = nn.Linear(32, 3)

    def forward(self, videos):
        n, t = videos.shape[:2]
        frames = videos.reshape(n * t, *videos.shape[2:])       # every frame of every video: (N T, 1, 32, 32)
        features = self.cnn(frames).reshape(n, t, -1)           # back to one sequence per video: (N, T, 128)
        out, _ = self.lstm(features)
        return self.head(out[:, -1])
```
Le `reshape` à l'aller et au retour garde l'ordre : les $T$ images d'une même vidéo restent consécutives. La convolution de stride 2 réduit 32 × 32 à 16 × 16, le max pooling adaptatif à 4 × 4 : chaque image devient un vecteur de 128 valeurs, et presque tous les poids sont dans le LSTM, comme dans le CNN-LSTM du livre (🧮 22.8).
**Dans tes notes** : mélanger les images garde chaque image intacte mais détruit le mouvement ; le réseau voit une suite de positions sans ordre et ne reconnaît plus guère que le vol à plat. À l'envers, une balle qui monte devient une balle qui descend, et elle va de droite à gauche, ce que le réseau n'a jamais vu : il se trompe sur presque tout, d'où un quart environ, moins que le hasard. Avec la carte entière, $8 \times 16 \times 16 = 2\,048$ valeurs par image, le LSTM aurait $4\,(2\,048 \times 32 + 32^2 + 64) = 266\,496$ poids, treize fois plus.
**Erreurs fréquentes** : `features.reshape(t, n, -1)` au retour (aucune erreur de forme, mais les images de vidéos différentes sont mélangées : c'est ce que vérifie l'indépendance des vidéos) ; oublier `nn.Flatten()` ; un LSTM sans `batch_first=True` ; compter un seul biais par porte (20 915).
**Variante** : avec 16 filtres au lieu de 8, chaque image donne 256 valeurs et le réseau a $(16 \times 25 + 16) + 4\,(256 \times 32 + 1\,024 + 64) + 99 = 416 + 37\,120 + 99 =$ **37 635** paramètres.

### Ex 22.29 — Défi taches solaires : battre la persistance de 20 % à six mois 🏆
**Réponse** (une solution parmi d'autres) : 0,740 de la persistance sur la période publique et 0,786 sur le test caché.
```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features_29(v):
    return np.array([v[-1], v[-2:].mean(), v[-3:].mean(), v[-6:].mean(), v[-12:].mean(), v[-24:].mean(), v[-48:].mean(),
                     v[-1] - v[-13], v[-120:-108].mean(), v[-132:-120].mean()])


def make_forecaster_29(train):
    s = np.sqrt(np.asarray(train, dtype=float))                 # the square root tames the peaks of the cycle
    first = 132 + HORIZON_29 - 1                                # the first target with 132 months of past
    X = np.array([features_29(s[:t - HORIZON_29 + 1]) for t in range(first, len(s))])
    y = s[first:]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    trees = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0).fit(X, y)

    def forecast(past):
        f = features_29(np.sqrt(np.asarray(past, dtype=float)))[None]
        return float(max((ridge.predict(f)[0] + trees.predict(f)[0]) / 2, 0.0) ** 2)

    return forecast
```
**Démarche** : la série a deux échelles, le bruit d'un mois à l'autre et un cycle d'environ 11 ans. Les moyennes sur 1 à 48 mois lissent le bruit à plusieurs échelles, la différence sur un an donne la tendance, et les niveaux d'il y a 9 à 11 ans disent où l'on était au cycle précédent. La racine carrée réduit les pics, où les erreurs absolues sont les plus fortes et la variance la plus grande. Deux modèles différents, l'un linéaire, l'autre à arbres, se trompent différemment : leur moyenne est plus sûre que chacun. Sur la période publique, la ridge seule fait environ 0,78 et le gradient boosting seul 0,78 ; leur moyenne, 0,74. Un LSTM sur les 60 derniers mois (22.23), utilisé comme prévision, fait presque aussi bien sur la période publique (0,77), mais près de 0,9 sur la période cachée : réglé sur des cycles forts, il ne tient pas sur le cycle 24, le plus faible depuis un siècle.
**Erreurs fréquentes** : choisir ses réglages sur la période publique jusqu'à y gagner quelques millièmes (le score caché n'en profite pas, et c'est le seul qui compte) ; utiliser `SPOTS` dans `forecast` (les mois à venir sont cachés : des NaN) ; réajuster le modèle dans `forecast` ; prédire la valeur suivante au lieu de celle de 6 mois plus tard ; oublier de revenir de la racine carrée.
**Variante** : avec le palier 🌟, il faut gagner encore un peu sur le test caché : d'autres résumés du cycle (le niveau d'il y a exactement 11 ans, aligné sur la cible, la variabilité récente), une régularisation plus forte de la ridge, une validation prise dans `train` pour régler sans regarder 1986-2005. Par exemple, quatre résumés de plus, placés devant les dix autres dans `features_29` : `v[-127:-124].mean()` (11 ans avant la cible), `v[-139:-136].mean()` (12 ans avant), `v[-1] - v[-7]` et `v[-12:].std()`, la première cible à `140 + HORIZON_29 - 1` et `Ridge(alpha=10.0)` : la même moyenne atteint alors **0,72** sur la période publique et **0,77** sur le test caché, d'autres variantes voisines entre 0,77 et 0,79.
