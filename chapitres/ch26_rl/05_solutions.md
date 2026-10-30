# 26 · Apprentissage par renforcement — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels et des exercices ✏️ 26.1 à 26.6 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 26.Q1 — Apprendre avec des récompenses : ni supervisé, ni non supervisé
a) **B** : on ne dit jamais à l'agent quelle action était la bonne, seulement si le résultat est meilleur ou pire ; le RL utilise souvent des réseaux (A est faux) et apprend de ses propres essais, qui sont des données (C). b) **« ACE »** : régler les ascenseurs, faire marcher un robot et jouer au backgammon demandent d'agir et de juger le résultat ; reconnaître des chiffres étiquetés est supervisé (B), regrouper des clients sans label est non supervisé (D). c) **Faux** : la récompense qui compte, gagner ou perdre, peut n'arriver qu'à la fin de la partie, bien après le coup qui l'a décidée. d) **Faux** : le RL est une approche générale, une famille de méthodes (L-learning, Q-learning, SARSA, DQN…), pas un algorithme. e) **B** : souvent, aucune réponse n'est objectivement juste ; on cherche la meilleure pour le critère qu'on s'est donné.
**À retenir** : supervisé = la bonne réponse ; non supervisé = rien ; RL = une note sur ce qu'on a fait, parfois tardive.

### 26.Q2 — Les quatre informations données à l'agent du morpion
a) **B** : l'amie apprend, c'est l'agent ; nous jouons les ronds et donnons les récompenses, nous sommes l'environnement. b) **« ABDF »** : le plateau, la liste des coups possibles, une récompense et la consigne de chercher les grandes récompenses ; ni la règle (C), qu'elle découvre, ni même le nom du jeu (E). c) **Faux** : sa façon de choisir et ce qu'elle retient sont ses informations **privées**. d) **B** : avec des coups interdits notés 0, l'agent apprend seul à les éviter. e) **7** : les cases vides.
**À retenir** : l'agent ne connaît ni les règles ni le but ; il ne voit qu'un état, des actions possibles et des récompenses.

### 26.Q3 — État, actions, politique et informations privées
a) **« CAB »** : choisir, l'environnement répond, se mettre à jour. b) **B** : un épisode ; une epoch est un passage sur un dataset. c) **B** : la politique associe une action à chaque état. d) **Faux** : c'est l'environnement qui exécute l'action, parce que le monde peut réagir pendant qu'elle se déroule. e) **Vrai** : l'état contient tous les nombres qui décrivent l'environnement, l'argent ou les cartes cachées compris.
**À retenir** : état → politique → action → l'environnement exécute et répond → mise à jour ; une partie entière est un épisode.

### 26.Q4 — Réponse de l'environnement, épisode et agents libres ou déclenchés
a) **B** : un nouvel état et une récompense ; la politique (A) et les valeurs Q (D) appartiennent à l'agent. b) **« BA »** : le cuisinier attend la sonnerie (déclenché), le minuteur tourne de lui-même (libre). c) **B** : chacun attend son tour. d) **Vrai** : quand chacun attend l'autre, la simulation n'a aucun problème de temps réel. e) **C** : `(state, reward, terminated, truncated, info)` ; `reset` rend `(state, info)`, et l'ancienne API de Gym rendait un seul `done`.
**À retenir** : Gymnasium sépare la vraie fin (`terminated`) de la coupure (`truncated`), et la différence compte pour la cible (§26.6.1).

### 26.Q5 — MDP, observabilité, attribution du crédit, exploration ou exploitation
a) **B** : l'état présent résume tout ce qui compte du passé. b) **B** : partiellement observable, l'agent ne voit pas tout l'état. c) **B** : l'attribution du crédit. d) **Faux** : une politique purement gloutonne rejoue toujours la même action ; une action sous-estimée par malchance n'est plus jamais essayée, et son erreur n'est jamais corrigée. e) **B** : le bandit à bras multiples, un problème de RL à un seul état.
**À retenir** : les deux difficultés du RL sont l'attribution du crédit (qui a fait gagner ?) et l'exploration (essayer pour savoir).

### 26.Q6 — Retour total, retour actualisé et rôle de γ
a) **3** $= 2 + 0 + 1$. b) **2,25** $= 2 + 0{,}5 \times 0 + 0{,}5^2 \times 1$. c) **Vrai** : $\gamma^0 = 1$ et $\gamma^k = 0$ pour $k \ge 1$. d) **B** : $\gamma$ fait partie de l'objectif ; A est l'interprétation du livre, C le rôle de $\alpha$. e) **20** $= 1/(1 - 0{,}95)$.
**À retenir** : $\gamma$ dit combien on préfère le proche au lointain ; l'horizon effectif vaut environ $1/(1-\gamma)$ pas.

### 26.Q7 — Flippers et L-learning : pourquoi « lousy » ?
a) **B** : exactement trois points sur une ligne ou une colonne complète, les autres tuiles vides ; les diagonales ne comptent pas. b) **6** : 3 lignes et 3 colonnes. c) **B** : le TFR de la dernière partie **remplace** l'ancienne valeur. d) **Vrai** : sans hasard, chaque coup rapporte toujours la même chose, et après 6 000 parties l'agent joue en deux coups les plateaux qui en demandent deux. e) **B** : une partie allongée par le camion écrase la bonne valeur d'un coup par une mauvaise, et l'agent « oublie » ce coup.
**À retenir** : remplacer marche dans un monde prévisible ; dès qu'il y a du hasard, il faut faire la moyenne.

### 26.Q8 — La mise à jour du Q-learning : α, γ et la valeur de l'état suivant
a) **B** : la ligne de l'état suivant $s'$, celui où mène l'action. b) **B** : avec $\alpha = 0$, la nouvelle valeur est l'ancienne. c) **B** : avec $\alpha = 1$, la cible remplace la valeur, comme au L-learning. d) **Faux** : le Q-learning met à jour après **chaque** pas. e) **0,6** : cible $0 + 1 \times 0{,}8 = 0{,}8$, puis $0{,}4 + 0{,}5 \times (0{,}8 - 0{,}4) = 0{,}6$.
**À retenir** : $Q \leftarrow (1 - \alpha)\,Q + \alpha\,(r + \gamma \max_{a'} Q(s', a'))$ ; $\alpha$ est le poids du nouveau.

### 26.Q9 — ε-greedy ou softmax : vrai ou faux
a) **Vrai** : en exploration, l'action est tirée parmi toutes, la meilleure comprise ; le livre la tire parmi les autres. b) **0,92** $= 1 - 0{,}1 + 0{,}1/5$. c) **Faux** : une grande température rend les probabilités presque égales ; c'est une petite température qui choisit presque toujours la meilleure. d) **Vrai** : la probabilité d'une action croît avec sa valeur ; ε-greedy, lui, explore toutes les autres actions de la même façon, même les pires. e) **B** : la somme d'une ligne de zéros vaut 0, et $0/0$ n'est pas défini.
**À retenir** : ε-greedy explore au hasard ; softmax explore en priorité ce qui semble presque aussi bon.

### 26.Q10 — Table initialisée à zéro : comment la valeur remonte et converge
a) **A** : une cible faite de zéros laisse une case nulle à 0. b) **B** : seule la dernière action voit la récompense de la victoire. c) **Vrai** : chaque cible regarde l'état suivant ; la valeur avance d'un état chaque fois qu'on passe par l'état précédent (✏️ 26.6). d) **A** : chaque case visitée une infinité de fois, et $\alpha$ qui décroît assez lentement. e) **C** : après 6 000 parties, la plus longue partie dure 18 coups (contre 36 après 300 et 20 après 3 000).
**À retenir** : rien ne bouge avant la première récompense ; ensuite, l'information remonte la chaîne des états.

### 26.Q11 — SARSA contre Q-learning : l'action suivante réellement jouée
a) **A** : $s, a, r, s', a'$. b) **B** : l'action choisie par la politique dans $s'$, qui sera jouée au pas suivant. c) **A** : le Q-learning apprend la valeur de la politique gloutonne, pas de celle qu'il suit. d) **Vrai** : si $a'$ est la gloutonne, $Q(s', a') = \max_{a'} Q(s', a')$. e) **Faux** : ils apprennent deux choses différentes ; chacun a ses cas.
**À retenir** : Q-learning : $\max$, hors politique ; SARSA : $Q(s', a')$ de l'action jouée, sur politique.

### 26.Q12 — Tables géantes, rejeu d'expérience, AlphaGo Zero et lancer de rayons
a) **16** : une colonne par case du plateau ; le livre en compte 9 par erreur. b) **B** : un réseau qui reçoit l'état et rend une valeur par action. c) **B** : des transitions gardées, tirées en mini-batches au hasard. d) **Faux** : « Zero » veut dire aucune partie humaine, seulement les règles et le jeu contre soi-même. e) **B** : quand un joueur joue, l'autre fait partie de son environnement.
**À retenir** : la table ne passe pas à l'échelle ; un réseau la remplace, et le rejeu stabilise son apprentissage.

<a id="rappels"></a>

## 🔁 Rappels

### 26.R1 — Un réseau comme fonction qui remplace une table (le décodeur d'un VAE)
a) **C** : 20 nombres réels peuvent prendre une infinité de valeurs. b) **Vrai** : un réseau généralise aux entrées voisines de ses données (25.Q11 montre aussi la limite : loin des données, il ressort ce qu'il connaît). c) **4** : une valeur Q par action, en un seul passage.
**À retenir** : un réseau est une table infinie et compressée, qui généralise près de ses données.

### 26.R2 — Que sauvegarder pour reprendre un entraînement ?
a) **« ABCD »** : les poids, l'état de l'optimiseur (les moyennes d'Adam), la position dans la boucle et les générateurs aléatoires ; le code de PyTorch n'est pas un état. b) **A** : la table Q, ce qui est appris. c) **Faux** : il faut aussi savoir où l'on en était dans le calendrier de $\varepsilon$ (le numéro d'épisode) et l'état du générateur aléatoire, sinon la suite n'est pas la même.
**À retenir** : un checkpoint contient tout ce qui change pendant l'entraînement, hasard compris.

### 26.R3 — Bandit à bras multiples, ε-greedy et UCB
a) **A** : un seul état, des bras qui sont les actions. b) **3,5** $= 3 + 0{,}25 \times (5 - 3)$, la même forme que la mise à jour du Q-learning, avec la cible à la place du gain. c) **B** : le gain moyen plus un bonus qui grandit quand le bras a été peu essayé.
**À retenir** : le Q-learning est un bandit dont la cible regarde l'état suivant ; `rl.py` réutilise `epsilon_greedy_action` et `incremental_update` du ch. 11.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 26.1 — Retours total et actualisé d'une partie de six coups ✏️
**Réponses** : a) **7** ; b) **6** ; c) **1,375** ; d) **4,820** ; e) **5,24** ; f) **1** ; g) **Vrai**.
**Démarche** : la récompense du coup $t + k$ pèse $\gamma^k$ dans le retour du coup $t$. a) $1 + 0 + 0 + 2 + 0 + 4 = 7$. b) $0 + 2 + 0 + 4 = 6$. c) $1 + 2 \times 0{,}5^3 + 4 \times 0{,}5^5 = 1 + 0{,}25 + 0{,}125 = 1{,}375$. d) $1 + 2 \times 0{,}729 + 4 \times 0{,}59049 = 4{,}81996$. e) $2 + 0 + 4 \times 0{,}81 = 5{,}24$. f) seul $\gamma^0 = 1$ reste : la récompense du coup 1. g) Avec $\gamma = 1$, $G_t = r_t + G_{t+1}$, donc $G_{t+1} = G_t - r_t \le G_t$ quand $r_t \ge 0$.
**Dans ta copie** : à rebours avec $\gamma = 0{,}9$ : $G_6 = 4$, $G_5 = 0 + 0{,}9 \times 4 = 3{,}6$, $G_4 = 2 + 3{,}24 = 5{,}24$, $G_3 = 4{,}716$, $G_2 = 4{,}2444$, $G_1 = 1 + 3{,}81996 = 4{,}81996$. La récurrence ne demande aucune puissance, et donne au passage le retour de chaque coup.
**Erreurs fréquentes** : actualiser la récompense immédiate (poids $\gamma$ au lieu de 1) ; commencer le retour du coup 4 à la récompense du coup 5 ; arrondir pendant le calcul.
**Variante** : récompenses $(0, 3, 0, 1)$ et $\gamma = 0{,}5$ : $G_1 = 0 + 1{,}5 + 0 + 0{,}125 =$ **1,625**.

### Ex 26.2 — Taille des tables : de Flippers au Go ✏️
**Réponses** : a) **512** ; b) **4 608** ; c) **1 048 576** ; d) **8** ; e) **19 683** ; f) **688 747 536** ; g) **173**.
**Démarche** : a) $2^9$. b) $512 \times 9$. c) $2^{16} \times 16 = 65\,536 \times 16$. d) 4 lignes et 4 colonnes. e) $3^9$. f) $3^{16} \times 16 = 43\,046\,721 \times 16$. g) $\log_{10} 3^{361} = 361 \times 0{,}47712 = 172{,}24$, donc 173 chiffres.
**Dans ta copie** : avec 9 colonnes, le livre compte $3^{16} \times 9 = 387\,420\,489$ cases, soit $301\,327\,047$ de moins qu'il n'en faut ; sa table ne pourrait pas noter les coups sur les 7 dernières cases. La plupart des codages de e) sont impossibles : il y a autant de croix que de ronds, ou une de plus ; la partie s'arrête au premier alignement, donc deux alignements de joueurs différents ne coexistent pas ; il reste 5 478 positions atteignables.
**Erreurs fréquentes** : $9^2$ au lieu de $2^9$ ; oublier de multiplier par le nombre d'actions ; garder 9 colonnes pour le 4 × 4 ; compter les diagonales en d).
**Variante** : un Flippers 5 × 5 a $2^{25} \times 25 =$ **838 860 800** cases : déjà près d'un milliard pour un jeu de 25 tuiles.

### Ex 26.3 — Une mise à jour de Q-learning à la main ✏️
**Réponses** : a) **0,74** ; b) **0,24** ; c) **0,62** ; d) **0,35** ; e) **0,74** ; f) **Faux**.
**Démarche** : a) $\max(0{,}1 ; 0{,}6 ; 0{,}3 ; 0{,}6) = 0{,}6$, $y = 0{,}2 + 0{,}9 \times 0{,}6 = 0{,}74$. b) $\delta = 0{,}74 - 0{,}5 = 0{,}24$. c) $0{,}5 + 0{,}5 \times 0{,}24 = 0{,}62$. d) Un état terminal n'a pas d'avenir : $y = r = 0{,}2$, et $0{,}5 + 0{,}5 \times (0{,}2 - 0{,}5) = 0{,}35$. e) $\alpha = 1$ : la nouvelle valeur est la cible, 0,74. f) La formule ne modifie que $Q(s, a)$ ; la ligne de $s'$ n'est que lue.
**Dans ta copie** : la cible n'a besoin que de la **valeur** du max, 0,6, quelle que soit l'action qui l'atteint. Une politique gloutonne, elle, doit **choisir** entre les actions 1 et 3 : `np.argmax` prendrait toujours la première, et l'action 3 ne serait jamais essayée ; on départage au hasard (`argmax_random_tie` du ch. 11).
**Erreurs fréquentes** : prendre le max de la ligne de $s$ ; inverser $\delta$ ; garder $\gamma \max$ pour un état terminal ; donner la cible comme nouvelle valeur quand $\alpha < 1$.
**Variante** : avec $\gamma = 0{,}5$, la cible vaut $0{,}2 + 0{,}3 = 0{,}5$ et la valeur ne change pas : **0,5**, $\delta = 0$.

### Ex 26.4 — La même transition vue par SARSA ✏️
**Réponses** : a) **0,47** ; b) **0,485** ; c) **0,135** ; d) **Vrai** ; e) **B**.
**Démarche** : a) $Q(s', 2) = 0{,}3$ (la troisième valeur), $y = 0{,}2 + 0{,}9 \times 0{,}3 = 0{,}47$. b) $0{,}5 + 0{,}5 \times (0{,}47 - 0{,}5) = 0{,}485$. c) $0{,}62 - 0{,}485 = 0{,}135$. d) $Q(s', 1) = 0{,}6$, le max : la même cible que le Q-learning. e) SARSA apprend la valeur de la politique qu'il suit, exploration comprise.
**Erreurs fréquentes** : numéroter les actions à partir de 1 (et prendre 0,6) ; prendre le max par réflexe ; inverser la différence de c).
**Variante** : si la politique avait choisi l'action 0 (valeur 0,1), $y = 0{,}29$ et SARSA donnerait $0{,}5 + 0{,}5 \times (0{,}29 - 0{,}5) =$ **0,395** : une exploration malheureuse dans $s'$ fait baisser la valeur de $(s, a)$.

### Ex 26.5 — Probabilités d'action : ε-greedy (standard et version du livre) et softmax ✏️
**Réponses** : a) **0,85** ; b) **0,05** ; c) **0,80** ; d) **0,067** ; e) **0,831** ; f) **0,579** ; g) **Faux** ; h) **0,571**.
**Démarche** : a) $1 - 0{,}2 + 0{,}2/4 = 0{,}85$. b) $0{,}2/4 = 0{,}05$. c) le livre : $1 - \varepsilon = 0{,}80$. d) les trois autres se partagent $0{,}2$ : $0{,}0667$. e) $e^4 / (e + e^2 + e^4 + 1) = 54{,}60 / 65{,}71 = 0{,}8310$. f) Valeurs divisées par 2 : $e^2 / (e^{0,5} + e + e^2 + 1) = 7{,}389 / 12{,}756 = 0{,}5793$. g) $e^{Q_i + 10} = e^{10} e^{Q_i}$ : le facteur $e^{10}$ s'annule entre le numérateur et le dénominateur. h) $4 / (1 + 2 + 4 + 0) = 0{,}5714$.
**Dans ta copie** : la version du livre explore davantage chaque mauvaise action : $0{,}067$ contre $0{,}05$, un tiers de plus ($\varepsilon/3$ contre $\varepsilon/4$). Avec une valeur négative, la division par la somme peut donner une « probabilité » négative, ou une somme nulle : la « softmax » du livre n'est pas une loi de probabilité. Une vraie softmax reste positive et ne dépend que des **écarts** entre valeurs (g).
**Erreurs fréquentes** : oublier que l'exploration standard peut tirer la gloutonne ; multiplier par $\tau$ au lieu de diviser ; confondre la softmax du livre et la vraie.
**Variante** : avec $\tau = 0{,}5$, l'action 2 a la probabilité $e^8 / (e^2 + e^4 + e^8 + 1) =$ **0,979** : une température basse rend la softmax presque gloutonne.

### Ex 26.6 — Regarder la valeur remonter une chaîne d'états ✏️
**Réponses** : a) **1** ; b) **0,90** ; c) **0,81** ; d) **3** ; e) **0,225** ; f) **0,875** ; g) **0,81** ; h) **10**.
**Démarche** : avec $\alpha = 1$, chaque mise à jour remplace la valeur par sa cible. Épisode 1 : $Q(0) = 0{,}9 \times 0 = 0$, $Q(1) = 0$, $Q(2) = 1$. Épisode 2 : $Q(0) = 0$, $Q(1) = 0{,}9$, $Q(2) = 1$. Épisode 3 : $Q(0) = 0{,}81$. La valeur avance d'un état par épisode, parce que l'état 0 est mis à jour **avant** que l'état 1 ne change. Avec $\alpha = 0{,}5$ : épisode 1, $Q(2) = 0{,}5$ ; épisode 2, $Q(1) = 0{,}5 \times 0{,}9 \times 0{,}5 = 0{,}225$ (la valeur de l'état 2 d'avant sa mise à jour), puis $Q(2) = 0{,}75$ ; épisode 3, $Q(2) = 0{,}875$. g) Le point fixe : $Q(2) = 1$, $Q(1) = 0{,}9$, $Q(0) = 0{,}81$ ; $\alpha$ change la vitesse, pas la limite. h) L'état 9 reçoit la valeur à l'épisode 1, l'état $9 - k$ à l'épisode $k + 1$, l'état 0 à l'épisode 10.
**Dans ta copie** : des mises à jour à rebours (2, puis 1, puis 0) font remonter la valeur sur toute la chaîne en un seul épisode : avec $\alpha = 1$, $Q(0) = 0{,}81$ dès l'épisode 1. Le L-learning le fait déjà, puisqu'il attend la fin de la partie et écrit le TFR de tous les coups. C'est aussi l'idée des méthodes de Monte-Carlo et des traces d'éligibilité.
**Erreurs fréquentes** : faire les mises à jour à rebours alors que l'énoncé demande l'ordre des pas ; utiliser la valeur déjà mise à jour de l'état suivant ; croire que $\alpha$ change la limite.
**Variante** : avec $\gamma = 0{,}5$ et $\alpha = 1$, $Q(0)$ vaut **0,25** à la fin de l'épisode 3, sa limite.

### Ex 26.7 — Somme géométrique et horizon effectif de γ ∂
**Réponse** :
1. $S_n - \gamma S_n = (1 + \gamma + \dots + \gamma^{n-1}) - (\gamma + \dots + \gamma^n) = 1 - \gamma^n$, d'où $S_n = \frac{1 - \gamma^n}{1 - \gamma}$. Pour $0 \le \gamma < 1$, $\gamma^n \to 0$, et $S_n \to \frac{1}{1-\gamma}$.
2. $\sum_k \gamma^k r_{\max} = r_{\max} \sum_k \gamma^k = \frac{r_{\max}}{1-\gamma}$. Si $r_k \le r_{\max}$ pour tout $k$, alors $\gamma^k r_k \le \gamma^k r_{\max}$ terme à terme, et la somme ne peut pas dépasser $\frac{r_{\max}}{1-\gamma}$.
3. $\gamma = 0{,}5$ : 2 pas ; $\gamma = 0{,}9$ : 10 pas ; $\gamma = 0{,}99$ : 100 pas.
4. $\gamma^k < \tfrac{1}{2} \iff k \ln\gamma < -\ln 2 \iff k > \frac{\ln 2}{-\ln\gamma}$ (on divise par $\ln\gamma < 0$, l'inégalité se retourne). $\gamma = 0{,}9$ : $0{,}6931 / 0{,}1054 = 6{,}58$, donc à partir de $k = 7$ ; $\gamma = 0{,}99$ : $0{,}6931 / 0{,}01005 = 68{,}97$, donc $k = 69$. Ces « demi-vies » valent environ $0{,}69$ fois l'horizon effectif.
5. $0{,}2^4 = 0{,}0016$ de sa valeur ; à 6 pas, $0{,}2^6 = 0{,}000064$. En valeur relative, la victoire proche pèse 25 fois plus : l'agent la préfère. Mais les deux valeurs sont minuscules : un écart de 0,0015 se noie dans le bruit des mises à jour avec $\alpha = 0{,}95$ et le camion, et dans les récompenses immédiates éventuelles. En pratique, l'agent voit à peine plus loin que deux ou trois coups.
**Erreurs fréquentes** : oublier de retourner l'inégalité en 4 ; prendre $\ln$ de base 10 et de base $e$ dans le même calcul (le rapport ne dépend pas de la base, à condition de garder la même) ; confondre l'horizon $\frac{1}{1-\gamma}$ et la demi-vie.
**Variante** : pour $\gamma = 0{,}95$, l'horizon effectif vaut **20** pas et la demi-vie $\ln 2 / (-\ln 0{,}95) = 13{,}5$, soit **14** pas.

### Ex 26.8 — Équation de Bellman : le point fixe que vise le Q-learning ∂
**Réponse** :
1. Avec $V^*(s) = \max_a Q^*(s, a)$ et $V^*(T) = 0$ :
   $Q^*(A, \text{rester}) = 0 + 0{,}9\,V^*(A)$ ; $Q^*(A, \text{aller}) = 0 + 0{,}9\,V^*(B)$ ; $Q^*(B, \text{finir}) = 1 + 0{,}9 \times 0 = 1$ ; $Q^*(B, \text{jouer}) = 2 + 0{,}9\,\big(\tfrac{1}{2} \times 0 + \tfrac{1}{2}\,V^*(A)\big) = 2 + 0{,}45\,V^*(A)$.
2. Avec « aller » et « jouer » : $V(A) = 0{,}9\,V(B)$ et $V(B) = 2 + 0{,}45 \times 0{,}9\,V(B) = 2 + 0{,}405\,V(B)$, d'où $V(B) = 2 / 0{,}595 = 3{,}361$ et $V(A) = 3{,}025$. Les quatre valeurs : $Q^*(A, \text{rester}) = 0{,}9 \times 3{,}025 = 2{,}723$, $Q^*(A, \text{aller}) = 3{,}025$, $Q^*(B, \text{finir}) = 1$, $Q^*(B, \text{jouer}) = 3{,}361$ (exactement $\tfrac{324}{119}$, $\tfrac{360}{119}$, 1, $\tfrac{400}{119}$). La politique est gloutonne : $3{,}025 > 2{,}723$ dans $A$, $3{,}361 > 1$ dans $B$ ; elle est donc optimale.
3. Pour une case, $(\mathcal{T}Q)(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q(s', a')]$. Les récompenses s'annulent dans la différence : $|\mathcal{T}Q_1(s, a) - \mathcal{T}Q_2(s, a)| = \gamma\,\big|\mathbb{E}[\max_{a'} Q_1(s', a') - \max_{a'} Q_2(s', a')]\big| \le \gamma\,\mathbb{E}\big[\max_{a'} |Q_1(s', a') - Q_2(s', a')|\big] \le \gamma \max |Q_1 - Q_2|$. Avec $\gamma < 1$, $\mathcal{T}$ est une **contraction** : chaque application multiplie la distance entre deux tables par au plus $\gamma$. Deux points fixes $Q$ et $Q'$ vérifieraient $\max |Q - Q'| \le \gamma \max |Q - Q'|$, donc seraient égaux ; et la suite $\mathcal{T}^n Q_0$ se rapproche de $Q^*$ d'un facteur $\gamma$ à chaque pas, quelle que soit $Q_0$ (théorème du point fixe de Banach). C'est l'itération sur les valeurs (🔨 26.26).
4. Le Q-learning ne connaît pas les probabilités, mais l'environnement tire $s'$ avec ces probabilités : la cible $r + \gamma \max_{a'} Q(s', a')$ vaut, en moyenne sur les tirages, le second membre de Bellman. La mise à jour $Q \leftarrow Q + \alpha\,(y - Q)$ est une moyenne mobile de ces échantillons. Dans « jouer », la cible vaut tantôt 2 (fin en $T$), tantôt $2 + 0{,}9\,V(A) \approx 4{,}72$. Avec $\alpha$ proche de 1, $Q(B, \text{jouer})$ saute entre ces deux valeurs au lieu de se poser sur leur moyenne, 3,36 : il ne fait que copier le dernier tirage, exactement le défaut du L-learning.
**Erreurs fréquentes** : donner la récompense 2 seulement quand on atteint $T$ ; oublier le facteur $\tfrac{1}{2}$ ; mettre une valeur non nulle à l'état terminal ; supposer la politique sans vérifier qu'elle est gloutonne.
**Variante** : si « jouer » ne rapportait que 0,5, la politique optimale jouerait « finir » dans $B$ : $Q^*(B, \text{finir}) = 1$, $Q^*(A, \text{aller}) = 0{,}9$, $Q^*(A, \text{rester}) = 0{,}81$, $Q^*(B, \text{jouer}) = 0{,}5 + 0{,}45 \times 0{,}9 =$ **0,905** < 1.

### Ex 26.9 — Mémoire d'une table Q : morpion 4×4, 5×5 et Go 🧮
**Réponse** :
1. $19\,683 \times 9 \times 4 \approx 7{,}1 \times 10^5$ octets, 0,7 Mo : rien.
2. 4 × 4 : $4{,}3 \times 10^7 \times 16 \times 4 \approx 2{,}8 \times 10^9$ octets, 2,8 Go, qui tiennent dans un portable de 16 Go. 5 × 5 : $8{,}5 \times 10^{11} \times 25 \times 4 \approx 8{,}5 \times 10^{13}$ octets, 85 To : environ 5 000 portables, mais seulement un dix-millième d'un exaoctet ; c'est démesuré, pas impossible. Go : $3^{361} \approx 1{,}7 \times 10^{172}$ codages, soit environ $2{,}5 \times 10^{175}$ octets, $10^{157}$ fois un grand centre de données : impossible, et même sans rapport avec le nombre d'atomes de l'univers (environ $10^{80}$).
3. $5\,478 \times 9 \times 4 \approx 2 \times 10^5$ octets ; la table est $19\,683 / 5\,478 \approx 3{,}6$ fois plus petite (et encore plus si l'on ne garde que les coups permis, ou si l'on identifie les plateaux symétriques).
4. $3^{25} \times 25 \approx 2{,}1 \times 10^{13}$ cases ; à $10^9$ par seconde, $2{,}1 \times 10^4$ s, environ 6 heures pour **une** visite. Mais les visites viennent de parties jouées, beaucoup de cases ne sont jamais atteintes, et chacune doit être visitée de nombreuses fois pour que sa moyenne se stabilise : en pratique, on n'y arrive pas.
5. Le réseau reçoit l'état (25 nombres) et rend 25 valeurs Q ; des plateaux voisins partagent ce qu'il a appris. Paramètres : $26 \times 256 + 257 \times 256 + 257 \times 25 = 6\,656 + 65\,792 + 6\,425 = 78\,873$, environ 315 Ko en `float32` : $10^8$ fois moins que la table.
**Erreurs fréquentes** : oublier les 4 octets ; garder 9 colonnes pour les grands plateaux ; confondre octets et bits ; conclure qu'une table de 85 To est « impossible » (elle est démesurée ; c'est la remplir qui l'est).
**Variante** : un puissance 4 (6 × 7 cases, 7 colonnes où jouer) se code en au plus $3^{42} \approx 1{,}1 \times 10^{20}$ plateaux, soit une table d'environ $3 \times 10^{21}$ octets : **trois mille exaoctets**, déjà hors de portée.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 26.10 — Attribution du crédit et exploration expliquées en cinq lignes 🗣️
**Réponse modèle** : « Quand tu réussis un gâteau, tu ne sais pas si c'est grâce au beurre, à la cuisson ou au four : le résultat n'arrive qu'à la fin. Un agent qui joue a le même problème avec les coups d'une partie gagnée. Le Q-learning s'en sort en notant chaque coup d'après la note de la situation où il mène : la valeur de la victoire remonte, partie après partie, du dernier coup vers les premiers. Et si tu vas toujours dans le même restaurant parce qu'il est bon, tu ne sauras jamais s'il en existe un meilleur : l'agent doit parfois essayer un coup qu'il croit moins bon, pour vérifier qu'il ne se trompe pas. »
**Ce qui compte** : le résultat tardif ; le mérite partagé entre des étapes ; la valeur qui remonte d'une étape à la précédente ; le prix de l'exploration (on joue parfois moins bien) et son gain (on apprend).
**Erreurs fréquentes** : employer « récompense actualisée » ou « table Q » sans les expliquer ; présenter l'exploration comme une erreur à éliminer.
**Variante** : en une phrase pour un chef de projet : « Un agent de RL apprend en notant chaque décision d'après ce qu'elle a permis ensuite, et en essayant de temps en temps autre chose pour ne pas rester bloqué sur une habitude. »

### Ex 26.11 — Quand l'agent optimise la mauvaise récompense ⚖️
**Réponse** :
1. Pour l'agent, c'est la meilleure stratégie : les bonus rapportent, réapparaissent, et rien dans la récompense ne parle de finir ; tourner en rond donne plus de points que la course entière. Une récompense de **résultat** (le classement à l'arrivée, ou une pénalité par seconde jusqu'à l'arrivée) l'aurait évité. Son prix : une récompense rare et tardive, donc une attribution du crédit et une exploration bien plus difficiles ; on garde souvent quelques récompenses intermédiaires, mais conçues pour ne pas créer de boucle (en termes techniques, des récompenses de forme « différence de potentiel », qui ne changent pas la politique optimale).
2. Le temps passé n'est pas la satisfaction. Pour les utilisateurs : des recommandations qui accrochent sans apporter (contenus de plus en plus extrêmes ou anxiogènes, défilement compulsif, sommeil réduit). Pour la société : polarisation, désinformation qui retient l'attention, créateurs poussés vers ce qui retient plutôt que ce qui vaut. Mesures à surveiller : satisfaction déclarée (sondages), retours explicites (« pas intéressé »), désabonnements, diversité des contenus consommés, part de signalements, effets par groupe d'âge ; et des expériences contrôlées qui suivent ces mesures, pas seulement le temps.
3. Le modèle de récompense apprend ce que **préfèrent** des évaluateurs, qui jugent vite : une réponse assurée, flatteuse, longue et bien présentée plaît souvent davantage qu'une réponse juste qui doute ou contredit. En maximisant la note, le modèle exploite ces biais et les erreurs du modèle de récompense loin de ses données (piratage de la récompense). La pénalité KL le garde proche du modèle de départ : il ne peut pas s'en éloigner assez pour trouver les failles extrêmes du modèle de récompense, et garde ses capacités d'origine.
4. L'agent : il « choisit », mais il n'a ni intention ni compréhension du but ; on ne peut pas lui imputer une faute. L'ingénieur : il a écrit la récompense et pouvait tester des cas de détournement. L'entreprise : elle choisit l'objectif (souvent commercial), le déploiement, les mesures surveillées, et elle en tire le bénéfice. Position défendable : la responsabilité est humaine et partagée, d'abord celle de l'organisation qui déploie, qui doit surveiller les effets réels, pas seulement la récompense, et corriger ; l'ingénieur a un devoir d'alerte.
**Erreurs fréquentes** : dire que l'agent « triche » (il fait exactement ce qu'on lui demande) ; ne proposer qu'une meilleure récompense, sans surveillance ; oublier le prix d'une récompense plus juste mais plus rare.
**Variante** : lis la liste d'exemples de *specification gaming* tenue par V. Krakovna et coll. (DeepMind, 2020) et classe trois exemples selon la cause : récompense incomplète, faille de l'environnement, ou modèle de récompense trompé.

### Ex 26.12 — Lire Mnih et coll. (2015), « Human-level control through deep reinforcement learning » 📄
**Réponse** :
1. L'entrée : les 4 dernières images du jeu, passées en gris et réduites à $84 \times 84$, empilées (une image seule ne donne pas la vitesse). Trois couches de convolution, une couche dense de 512 unités, puis **une sortie par action** : 18 sorties pour un jeu à 18 actions. Un seul passage donne toutes les valeurs Q de l'état, donc le max de la cible et le choix glouton ; un réseau qui prendrait le couple (état, action) en entrée demanderait 18 passages.
2. La loss est l'erreur quadratique entre $Q(s, a ; \theta)$ et la cible $y = r + \gamma \max_{a'} Q(s', a' ; \theta^-)$ ($y = r$ si l'épisode se termine). C'est la cible du Q-learning, mais calculée avec $\theta^-$, les poids d'un **réseau cible**, copie du réseau recopiée tous les $C$ pas (10 000 mises à jour) : sans elle, chaque mise à jour déplace aussi la cible, et l'apprentissage oscille ou diverge.
3. La mémoire de rejeu garde le dernier million de transitions $(s, a, r, s')$ ; chaque mise à jour utilise un mini-batch de 32 transitions tirées uniformément. Deux raisons : chaque transition sert à plusieurs mises à jour (moins de données gaspillées), et le tirage au hasard casse les corrélations entre transitions consécutives, ce qui réduit la variance des mises à jour et évite des boucles où la politique du moment fausse les données qu'elle produit.
4. $\varepsilon$ décroît linéairement de 1 à 0,1 pendant le premier million d'images, puis reste à 0,1 ; l'évaluation utilise $\varepsilon = 0{,}05$. Le livre conseille un $\varepsilon$ fixe de 0,01 ou moins (et l'utilise à 0,1) : DQN explore beaucoup au début, quand la table, ici le réseau, ne sait encore rien.
5. Les scores des jeux ont des échelles très différentes ; écrêter les récompenses à $[-1, 1]$ permet de garder le même learning rate et les mêmes hyperparamètres pour tous les jeux. On y perd la différence entre une petite et une grosse récompense : l'agent ne sait plus qu'un objet vaut 10 fois plus qu'un autre.
6. DQN reste faible sur les jeux où la récompense demande une longue suite d'actions précises avant d'arriver, comme Montezuma's Revenge (0 % du score humain) : l'exploration ε-greedy n'y tombe presque jamais par hasard, et l'attribution du crédit sur des centaines de pas est trop lente. Il brille au contraire sur les jeux à récompenses fréquentes et à réaction rapide (Breakout, Pong).
**Erreurs fréquentes** : croire que le réseau prend l'action en entrée ; confondre le réseau cible et le réseau entraîné ; oublier la pile de 4 images ; dire que le rejeu « rejoue des parties ».
**Variante** : lis ensuite H. van Hasselt et coll., « Deep Reinforcement Learning with Double Q-learning » (AAAI 2016, [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)) : le max de la cible surestime les valeurs, et une petite modification de la cible (choisir l'action avec un réseau, l'évaluer avec l'autre) corrige ce biais.

<a id="entretien"></a>

## 💼 Entretien

### 26.E1 — Q-learning ou SARSA : on-policy, off-policy, lequel choisir ?
**Réponse modèle en 60 secondes** : « Les deux mettent à jour Q(s, a) vers r plus γ fois une valeur de l'état suivant ; seule cette valeur diffère. Le Q-learning prend le maximum sur les actions de s' : il apprend la valeur de la politique gloutonne optimale, quelle que soit l'exploration qui produit les données. Il est hors politique. SARSA prend la valeur de l'action a' réellement choisie, exploration comprise : il apprend la valeur de la politique qu'il suit. Il est sur politique. L'exemple classique est la falaise : avec ε-greedy, le Q-learning apprend le chemin le plus court, au bord du vide, et y tombe parfois pendant l'apprentissage ; SARSA, qui sait qu'il lui arrive d'explorer, apprend un chemin plus prudent. Je choisis SARSA quand les erreurs pendant l'apprentissage coûtent cher, comme un robot réel, et le Q-learning quand je veux la meilleure politique finale, ou réutiliser des données produites par une autre politique, comme avec une mémoire de rejeu. »
**Ce qui compte** : la seule différence de formule ; on-policy contre off-policy ; l'exemple de la falaise ; le critère de choix (coût des erreurs pendant l'apprentissage, réutilisation des données).
**Relances possibles** : « Les deux convergent-ils vers Q* ? » (le Q-learning oui, sous les conditions de visite et de pas ; SARSA aussi si l'exploration décroît vers zéro, sinon vers la valeur de la politique ε-greedy) · « Et Expected SARSA ? » (la cible prend l'espérance de Q(s', ·) sous la politique au lieu d'un tirage : moins de variance, et il retrouve le Q-learning quand la politique est gloutonne) · « Pourquoi DQN est-il un Q-learning ? » (la mémoire de rejeu contient des transitions produites par d'anciennes politiques : il faut une méthode hors politique).

### 26.E2 — À quoi sert le facteur d'actualisation γ ?
**Réponse modèle en 60 secondes** : « Le retour actualisé pèse une récompense reçue dans k pas par γ puissance k. γ a deux rôles. Il dit combien on préfère une récompense proche à une récompense lointaine. Et, pour une tâche sans fin, il garde la somme finie : avec des récompenses bornées par r max, le retour ne dépasse pas r max sur un moins γ. Ce nombre, un sur un moins γ, donne aussi l'horizon effectif : avec γ égal à 0,99, l'agent regarde environ 100 pas devant lui. Je choisis γ d'après l'horizon de la tâche : assez grand pour que l'agent voie les conséquences de ses actes, mais pas trop proche de 1, parce que les cibles deviennent plus bruitées et l'apprentissage plus lent ; 0,99 est un défaut courant. γ fait partie de l'objectif : ce n'est ni la vitesse d'apprentissage, qui est α, ni une mesure du hasard de l'environnement. »
**Ce qui compte** : la définition ; la préférence pour le proche ; la somme finie ; l'horizon $1/(1-\gamma)$ ; le choix selon la tâche ; la distinction avec $\alpha$.
**Relances possibles** : « Que se passe-t-il avec γ = 0 ? » (un agent myope, qui ne maximise que la récompense immédiate : un bandit contextuel) · « Peut-on prendre γ = 1 ? » (pour des épisodes qui finissent toujours, oui ; pour une tâche sans fin, la somme peut diverger) · « γ change-t-il la politique optimale ? » (oui : un petit γ peut préférer un petit gain rapide à un gros gain lointain).

### 26.E3 — Comment gérer le compromis exploration/exploitation ?
**Réponse modèle en 60 secondes** : « Exploiter, c'est jouer ce qui semble le meilleur ; explorer, c'est essayer autre chose pour savoir si on se trompe. Sans exploration, une action sous-estimée par malchance n'est plus jamais essayée. La méthode la plus simple est ε-greedy : une action au hasard avec la probabilité ε, la meilleure sinon, souvent avec un ε qui décroît, de 1 vers 0,05. La softmax, ou politique de Boltzmann, tire les actions avec une probabilité qui croît avec leur valeur : elle explore surtout ce qui semble presque aussi bon. UCB, issu des bandits, ajoute à chaque valeur un bonus d'autant plus grand que l'action a été peu essayée : on explore là où l'incertitude est grande. On peut aussi initialiser la table de façon optimiste, pour que chaque action soit essayée au moins une fois. Et pour les problèmes aux récompenses très rares, on ajoute des bonus de curiosité ou de nouveauté. »
**Ce qui compte** : le dilemme ; ε-greedy et sa décroissance ; softmax ; UCB ; l'initialisation optimiste ; les bonus de curiosité pour les récompenses rares.
**Relances possibles** : « Pourquoi faire décroître ε ? » (au début on ne sait rien, l'exploration rapporte beaucoup ; à la fin, elle coûte sans rien apprendre) · « Quel ε pendant l'évaluation ? » (0, ou très petit : on mesure la politique apprise, pas l'exploration) · « Pourquoi ε-greedy marche-t-il mal sur Montezuma's Revenge ? » (une suite précise de centaines d'actions n'arrive jamais par hasard).

### 26.E4 — Pourquoi DQN utilise-t-il un replay buffer et un réseau cible ?
**Réponse modèle en 60 secondes** : « Le Q-learning avec un réseau de neurones est instable, pour deux raisons. D'abord, les transitions arrivent dans l'ordre du jeu : des transitions consécutives se ressemblent beaucoup, alors que la descente de gradient stochastique suppose des exemples à peu près indépendants. La mémoire de rejeu garde les dernières transitions, par exemple un million, et chaque mise à jour tire un mini-batch au hasard dans cette mémoire : les corrélations sont cassées, et chaque transition sert plusieurs fois. Ensuite, la cible r plus γ max Q(s', a') dépend du réseau qu'on entraîne : chaque mise à jour déplace la cible qu'on poursuit, et l'apprentissage peut osciller ou diverger. Le réseau cible est une copie figée du réseau, recopiée tous les quelques milliers de pas, qui calcule les cibles : on poursuit une cible fixe pendant un moment, comme en apprentissage supervisé. La mémoire de rejeu impose une méthode hors politique, ce qui tombe bien : DQN est un Q-learning. »
**Ce qui compte** : les deux causes d'instabilité (corrélations, cible mouvante) ; le remède à chacune ; la réutilisation des données ; le lien avec off-policy.
**Relances possibles** : « Qu'est-ce que le rejeu prioritaire ? » (tirer plus souvent les transitions à grande erreur δ, en corrigeant le biais par des poids d'importance) · « Et la mise à jour douce du réseau cible ? » (au lieu de recopier, on fait $\theta^- \leftarrow \tau\theta + (1 - \tau)\theta^-$ à chaque pas, comme dans DDPG ou SAC) · « Quels autres défauts de DQN ? » (la surestimation due au max, corrigée par le Double DQN ; l'exploration ε-greedy, faible sur les récompenses rares).

### 26.E5 — Qu'est-ce que le RLHF et pourquoi utilise-t-on le RL avec les LLM ?
**Réponse modèle en 60 secondes** : « Un modèle de langage pré-entraîné apprend à prolonger du texte, pas à être utile, honnête ou inoffensif. Le RLHF, apprentissage par renforcement à partir de retours humains, se fait en trois étapes. On ajuste d'abord le modèle sur des réponses écrites par des personnes. Puis on leur fait classer plusieurs réponses à une même question, et on entraîne sur ces classements un modèle de récompense qui note une réponse. Enfin on ajuste le modèle de langage par RL, souvent avec PPO, pour maximiser cette note, avec une pénalité KL qui le garde proche du modèle de départ. Pourquoi du RL ? Parce que la note porte sur une réponse entière, produite token par token par des tirages : elle ne se dérive pas comme une loss supervisée. Dans le vocabulaire du RL, l'état est le texte déjà écrit, l'action le token suivant, la récompense arrive à la fin. Depuis, DPO obtient un effet voisin directement à partir des paires de préférences, sans boucle de RL, et les modèles qui raisonnent sont entraînés par RL sur des récompenses vérifiables, comme la justesse d'une réponse mathématique, avec des méthodes comme GRPO. »
**Ce qui compte** : la limite du pré-entraînement ; les trois étapes ; le modèle de récompense ; la pénalité KL ; pourquoi le RL (récompense non dérivable sur une séquence) ; DPO et les récompenses vérifiables.
**Relances possibles** : « Pourquoi la pénalité KL ? » (sans elle, le modèle exploite les failles du modèle de récompense et perd ses capacités d'origine : piratage de la récompense) · « Quel avantage pour DPO ? » (plus simple et plus stable : un seul entraînement supervisé, pas de modèle de récompense ni d'échantillonnage pendant l'entraînement) · « Pourquoi des récompenses vérifiables ? » (elles ne se laissent pas flatter : on ne peut pas tromper un test unitaire ou une réponse exacte, sauf si la vérification elle-même a une faille).

<a id="notebook"></a>

## Exercices du notebook

Les solutions des exercices du notebook (26.13 à 26.38) seront ajoutées à la prochaine session de génération, avec les parties A à F du notebook.
