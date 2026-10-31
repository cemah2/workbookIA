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
a) **B** : exactement trois points sur une ligne ou une colonne complète, les autres tuiles vides ; les diagonales ne comptent pas. b) **6** : 3 lignes et 3 colonnes. c) **B** : le TFR de la dernière partie **remplace** l'ancienne valeur. d) **Vrai** : c'est ce que montrent les fig. 26.25 et 26.26 du livre, sur deux plateaux choisis ; 🔬 26.18 vérifie si cela vaut pour les 506 plateaux. e) **B** : une partie allongée par le camion écrase la bonne valeur d'un coup par une mauvaise, et l'agent « oublie » ce coup.
**À retenir** : remplacer, c'est oublier tout ce qu'on savait dès qu'une partie tourne mal ; dès qu'il y a du hasard, il faut faire la moyenne.

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
**À retenir** : Q-learning : $\max$, off-policy (*hors politique*) ; SARSA : $Q(s', a')$ de l'action jouée, on-policy (*sur politique*).

### 26.Q12 — Tables géantes, rejeu d'expérience, AlphaGo Zero et lancer de rayons
a) **16** : une colonne par case du plateau ; le livre en compte 9 par erreur. b) **B** : un réseau qui reçoit l'état et rend une valeur par action. c) **B** : des transitions gardées, tirées en mini-batches au hasard. d) **Faux** : « Zero » veut dire aucune partie humaine, seulement les règles et le jeu contre soi-même. e) **B** : quand un joueur joue, l'autre fait partie de son environnement.
**À retenir** : la table ne passe pas à l'échelle ; un réseau la remplace, et le rejeu stabilise son apprentissage.

<a id="rappels"></a>

## 🔁 Rappels

### 26.R1 — Ch. 25 : un réseau comme fonction qui remplace une table (le décodeur d'un VAE)
a) **C** : 20 nombres réels peuvent prendre une infinité de valeurs. b) **Vrai** : un réseau généralise aux entrées voisines de ses données (25.Q11 montre aussi la limite : loin des données, il ressort ce qu'il connaît). c) **4** : une valeur Q par action, en un seul passage.
**À retenir** : un réseau est une table infinie et compressée, qui généralise près de ses données.

### 26.R2 — Ch. 23 : que sauvegarder pour reprendre un entraînement ?
a) **« ABCD »** : les poids, l'état de l'optimiseur (les moyennes d'Adam), la position dans la boucle et les générateurs aléatoires ; le code de PyTorch n'est pas un état. b) **A** : la table Q, ce qui est appris. c) **Faux** : il faut aussi savoir où l'on en était dans le calendrier de $\varepsilon$ (le numéro d'épisode) et l'état du générateur aléatoire, sinon la suite n'est pas la même.
**À retenir** : un checkpoint contient tout ce qui change pendant l'entraînement, hasard compris.

### 26.R3 — Ch. 11 : bandit à bras multiples, ε-greedy et UCB
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
**Réponse modèle en 60 secondes** : « Les deux mettent à jour Q(s, a) vers r plus γ fois une valeur de l'état suivant ; seule cette valeur diffère. Le Q-learning prend le maximum sur les actions de s' : il apprend la valeur de la politique gloutonne optimale, quelle que soit l'exploration qui produit les données. Il est off-policy. SARSA prend la valeur de l'action a' réellement choisie, exploration comprise : il apprend la valeur de la politique qu'il suit. Il est on-policy. L'exemple classique est la falaise : avec ε-greedy, le Q-learning apprend le chemin le plus court, au bord du vide, et y tombe parfois pendant l'apprentissage ; SARSA, qui sait qu'il lui arrive d'explorer, apprend un chemin plus prudent. Je choisis SARSA quand les erreurs pendant l'apprentissage coûtent cher, comme un robot réel, et le Q-learning quand je veux la meilleure politique finale, ou réutiliser des données produites par une autre politique, comme avec une mémoire de rejeu. »
**Ce qui compte** : la seule différence de formule ; on-policy contre off-policy ; l'exemple de la falaise ; le critère de choix (coût des erreurs pendant l'apprentissage, réutilisation des données).
**Relances possibles** : « Les deux convergent-ils vers Q* ? » (le Q-learning oui, sous les conditions de visite et de pas ; SARSA aussi si l'exploration décroît vers zéro, sinon vers la valeur de la politique ε-greedy) · « Et Expected SARSA ? » (la cible prend l'espérance de Q(s', ·) sous la politique au lieu d'un tirage : moins de variance, et il retrouve le Q-learning quand la politique est gloutonne) · « Pourquoi DQN est-il un Q-learning ? » (la mémoire de rejeu contient des transitions produites par d'anciennes politiques : il faut une méthode off-policy).

### 26.E2 — À quoi sert le facteur d'actualisation γ ?
**Réponse modèle en 60 secondes** : « Le retour actualisé pèse une récompense reçue dans k pas par γ puissance k. γ a deux rôles. Il dit combien on préfère une récompense proche à une récompense lointaine. Et, pour une tâche sans fin, il garde la somme finie : avec des récompenses bornées par r max, le retour ne dépasse pas r max sur un moins γ. Ce nombre, un sur un moins γ, donne aussi l'horizon effectif : avec γ égal à 0,99, l'agent regarde environ 100 pas devant lui. Je choisis γ d'après l'horizon de la tâche : assez grand pour que l'agent voie les conséquences de ses actes, mais pas trop proche de 1, parce que les cibles deviennent plus bruitées et l'apprentissage plus lent ; 0,99 est un défaut courant. γ fait partie de l'objectif : ce n'est ni la vitesse d'apprentissage, qui est α, ni une mesure du hasard de l'environnement. »
**Ce qui compte** : la définition ; la préférence pour le proche ; la somme finie ; l'horizon $1/(1-\gamma)$ ; le choix selon la tâche ; la distinction avec $\alpha$.
**Relances possibles** : « Que se passe-t-il avec γ = 0 ? » (un agent myope, qui ne maximise que la récompense immédiate : un bandit contextuel) · « Peut-on prendre γ = 1 ? » (pour des épisodes qui finissent toujours, oui ; pour une tâche sans fin, la somme peut diverger) · « γ change-t-il la politique optimale ? » (oui : un petit γ peut préférer un petit gain rapide à un gros gain lointain).

### 26.E3 — Comment gérer le compromis exploration/exploitation ?
**Réponse modèle en 60 secondes** : « Exploiter, c'est jouer ce qui semble le meilleur ; explorer, c'est essayer autre chose pour savoir si on se trompe. Sans exploration, une action sous-estimée par malchance n'est plus jamais essayée. La méthode la plus simple est ε-greedy : une action au hasard avec la probabilité ε, la meilleure sinon, souvent avec un ε qui décroît, de 1 vers 0,05. La softmax, ou politique de Boltzmann, tire les actions avec une probabilité qui croît avec leur valeur : elle explore surtout ce qui semble presque aussi bon. UCB, issu des bandits, ajoute à chaque valeur un bonus d'autant plus grand que l'action a été peu essayée : on explore là où l'incertitude est grande. On peut aussi initialiser la table de façon optimiste, pour que chaque action soit essayée au moins une fois. Et pour les problèmes aux récompenses très rares, on ajoute des bonus de curiosité ou de nouveauté. »
**Ce qui compte** : le dilemme ; ε-greedy et sa décroissance ; softmax ; UCB ; l'initialisation optimiste ; les bonus de curiosité pour les récompenses rares.
**Relances possibles** : « Pourquoi faire décroître ε ? » (au début on ne sait rien, l'exploration rapporte beaucoup ; à la fin, elle coûte sans rien apprendre) · « Quel ε pendant l'évaluation ? » (0, ou très petit : on mesure la politique apprise, pas l'exploration) · « Pourquoi ε-greedy marche-t-il mal sur Montezuma's Revenge ? » (une suite précise de centaines d'actions n'arrive jamais par hasard).

### 26.E4 — Pourquoi DQN utilise-t-il une mémoire de rejeu et un réseau cible ?
**Réponse modèle en 60 secondes** : « Le Q-learning avec un réseau de neurones est instable, pour deux raisons. D'abord, les transitions arrivent dans l'ordre du jeu : des transitions consécutives se ressemblent beaucoup, alors que la descente de gradient stochastique suppose des exemples à peu près indépendants. La mémoire de rejeu garde les dernières transitions, par exemple un million, et chaque mise à jour tire un mini-batch au hasard dans cette mémoire : les corrélations sont cassées, et chaque transition sert plusieurs fois. Ensuite, la cible r plus γ max Q(s', a') dépend du réseau qu'on entraîne : chaque mise à jour déplace la cible qu'on poursuit, et l'apprentissage peut osciller ou diverger. Le réseau cible est une copie figée du réseau, recopiée tous les quelques milliers de pas, qui calcule les cibles : on poursuit une cible fixe pendant un moment, comme en apprentissage supervisé. La mémoire de rejeu impose une méthode off-policy, ce qui tombe bien : DQN est un Q-learning. »
**Ce qui compte** : les deux causes d'instabilité (corrélations, cible mouvante) ; le remède à chacune ; la réutilisation des données ; le lien avec off-policy.
**Relances possibles** : « Qu'est-ce que le rejeu prioritaire ? » (tirer plus souvent les transitions à grande erreur δ, en corrigeant le biais par des poids d'importance) · « Et la mise à jour douce du réseau cible ? » (au lieu de recopier, on fait $\theta^- \leftarrow \tau\theta + (1 - \tau)\theta^-$ à chaque pas, comme dans DDPG ou SAC) · « Quels autres défauts de DQN ? » (la surestimation due au max, corrigée par le Double DQN ; l'exploration ε-greedy, faible sur les récompenses rares).

### 26.E5 — Qu'est-ce que le RLHF et pourquoi utilise-t-on le RL avec les LLM ?
**Réponse modèle en 60 secondes** : « Un modèle de langage pré-entraîné apprend à prolonger du texte, pas à être utile, honnête ou inoffensif. Le RLHF, apprentissage par renforcement à partir de retours humains, se fait en trois étapes. On ajuste d'abord le modèle sur des réponses écrites par des personnes. Puis on leur fait classer plusieurs réponses à une même question, et on entraîne sur ces classements un modèle de récompense qui note une réponse. Enfin on ajuste le modèle de langage par RL, souvent avec PPO, pour maximiser cette note, avec une pénalité KL qui le garde proche du modèle de départ. Pourquoi du RL ? Parce que la note porte sur une réponse entière, produite token par token par des tirages : elle ne se dérive pas comme une loss supervisée. Dans le vocabulaire du RL, l'état est le texte déjà écrit, l'action le token suivant, la récompense arrive à la fin. Depuis, DPO obtient un effet voisin directement à partir des paires de préférences, sans boucle de RL, et les modèles qui raisonnent sont entraînés par RL sur des récompenses vérifiables, comme la justesse d'une réponse mathématique, avec des méthodes comme GRPO. »
**Ce qui compte** : la limite du pré-entraînement ; les trois étapes ; le modèle de récompense ; la pénalité KL ; pourquoi le RL (récompense non dérivable sur une séquence) ; DPO et les récompenses vérifiables.
**Relances possibles** : « Pourquoi la pénalité KL ? » (sans elle, le modèle exploite les failles du modèle de récompense et perd ses capacités d'origine : piratage de la récompense) · « Quel avantage pour DPO ? » (plus simple et plus stable : un seul entraînement supervisé, pas de modèle de récompense ni d'échantillonnage pendant l'entraînement) · « Pourquoi des récompenses vérifiables ? » (elles ne se laissent pas flatter : on ne peut pas tromper un test unitaire ou une réponse exacte, sauf si la vérification elle-même a une faille).

<a id="notebook"></a>

## Exercices du notebook

Les solutions complètes, exécutées, sont dans `05_solutions.ipynb` ; voici, pour chaque exercice, la réponse, la démarche, les erreurs fréquentes et une variante. Les chiffres sont ceux du mode rapide, sur CPU (2 cœurs) : les tables viennent de tirages aléatoires fixés par des graines, et une autre implémentation de `rl.py`, qui consomme les tirages dans un autre ordre, donne d'autres chiffres de détail. Les résultats sur plusieurs graines de 🔮 26.16, 📈 26.23, 🔮 26.28 et 🏆 26.38 viennent de `tools/chapters/history_ch26.py` (`tools/chapters/history_ch26.json`).

### Ex 26.13 — Prendre en main gymnasium avec FrozenLake 📦
**Réponse** : a) **16** ; b) **4** ; c) **0,019** (1,9 % des épisodes au hasard atteignent le but) ; d) **0,001** (la marche fixe réussit une fois sur 1 000 sur la glace glissante).
```python
def play_actions_13(env, actions, seed=None):
    env.reset(seed=seed)
    total, steps, terminated = 0.0, 0, False
    for action in actions:
        _, reward, terminated, truncated, _ = env.step(int(action))
        total += reward
        steps += 1
        if terminated or truncated:
            break
    return total, steps, terminated


def random_success_13(env, n_episodes, rng):
    n_actions = env.action_space.n
    successes = 0
    for _ in range(n_episodes):
        env.reset()
        total, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(n_actions)))
            total += reward
        successes += total > 0
    return successes / n_episodes
```
**Démarche** : un état par case (16), une action par direction (4). Sans glissade, ↓ ↓ → → ↓ → passe par les cases 4, 8, 9, 10, 14 et atteint G en 6 pas ; ↓ → tombe dans le trou de la case 5 au 2ᵉ pas, et `terminated` vaut `True` dans les deux cas. Au hasard, un épisode tombe presque toujours dans un trou avant d'atteindre G (1,9 %). Sur la glace glissante, chaque pas part de travers deux fois sur trois : une suite fixée d'actions ne réussit presque jamais. Il faut une **politique**, qui choisit l'action d'après l'état atteint. `terminated` signale une vraie fin (un trou, le but), `truncated` la coupure à 100 pas : après la première, l'état suivant n'a pas d'avenir ; après la seconde, il en a un (§26.6.1).
**Erreurs fréquentes** : déballer `step` en quatre valeurs (l'ancienne API de Gym) ; continuer à jouer après la fin d'un épisode ; oublier `int(...)` autour de l'action ; compter un épisode coupé comme une réussite.
**Variante** : les mêmes 1 000 épisodes au hasard sur la glace glissante réussissent **1,6 %** du temps, à peine moins que sans glissade : le hasard des actions domine déjà celui de la glace.

### Ex 26.14 — Retours actualisés et retours à venir 🔨
**Réponse** (le cœur de la référence, sans les validations) :
```python
def returns_to_go(rewards, gamma=1.0):
    r = _rewards(rewards)                  # 1-D float64 copy, no NaN; gamma checked in [0, 1]
    out = np.empty_like(r)
    G = 0.0
    for t in range(len(r) - 1, -1, -1):
        G = r[t] + gamma * G
        out[t] = G
    return out


def discounted_return(rewards, gamma):
    returns = returns_to_go(rewards, gamma)
    return float(returns[0]) if returns.size else 0.0
```
**Démarche** : la relation $G_t = r_t + \gamma\,G_{t+1}$ (fiche §26.3.7) donne tous les retours en un passage de la fin vers le début : $T$ multiplications et additions, au lieu de $T(T+1)/2$ pour une somme directe à chaque pas. Les tests comparent `discounted_return` à `polyval` (un polynôme en $\gamma$ dont les coefficients sont les récompenses) et `returns_to_go` à un filtre de `scipy.signal`.
**Erreurs fréquentes** : actualiser la récompense immédiate ; rendre un `np.float64` (la docstring affiche `1.75`, pas `np.float64(1.75)`) ; renvoyer une vue des récompenses ; accepter un tableau 2-D (deux épisodes collés) ; un épisode vide qui lève une `IndexError`.
**Variante** : `returns_to_go([1, 1, 1, 1], gamma=0.5)` vaut **[1,875 ; 1,75 ; 1,5 ; 1]** : chaque retour est la moitié du suivant, plus 1.

### Ex 26.15 — Reproduire les courbes d'actualisation (fig. 26.13-26.14) 🎨
**Réponse** :
```python
def plot_discounting_15(rewards, gammas):
    rewards = np.asarray(rewards, dtype=float)
    k = np.arange(len(rewards))
    returns = np.array([mylearn.rl.discounted_return(rewards, g) for g in gammas])
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    for g in gammas:
        left.plot(k + 1, g ** k * rewards, "o-", ms=3.5, label=f"γ = {fr(g, 1)}")
    left.set_xlabel("coup")
    left.set_ylabel("récompense actualisée $\\gamma^k r_k$")
    left.legend(fontsize=7, ncol=2)
    right.plot(gammas, returns, "o-")
    right.set_xlabel("γ")
    right.set_ylabel("retour actualisé du premier coup")
    plt.tight_layout()
    plt.show()
    return returns
```
**Démarche** : les retours du premier coup valent 0,80 ($\gamma = 0$), 1,00 (0,2), 1,32 (0,4), 1,96 (0,6), 3,55 (0,8), 5,22 (0,9) et 8,06 ($\gamma = 1$, la somme des dix récompenses). Avec $\gamma = 0{,}2$, le 3ᵉ coup ne pèse déjà plus que $0{,}04$ fois sa récompense et le 4ᵉ $0{,}008$ fois : au-delà, plus rien ne compte. La courbe de droite est un polynôme en $\gamma$ à coefficients positifs : sa pente $\sum_k k\,\gamma^{k-1} r_k$ grandit avec $\gamma$, et ce sont les coups lointains, qui ont les plus grands $k$, qui la font monter près de 1.
**Erreurs fréquentes** : numéroter les coups à partir de 1 dans la puissance ($\gamma^1$ pour le premier coup) ; tracer $\gamma^k$ sans les récompenses ; oublier `plt.show()` ou le `return`.
**Variante** : avec $\gamma = 0{,}95$, le retour du premier coup vaut **6,45**.

### Ex 26.16 — Prédire l'effet du camion sur la table L 🔮
**Réponse** : a) **Faux** ; b) **Faux** ; c) **C**.

| | sans camion | avec camion |
|---|---|---|
| plateaux gagnés (5 graines) | 100 % | 83 % à 94 % |
| joués au minimum de coups | 2 % à 14 % | 14 % à 20 % |
| longueur moyenne | 9,6 à 20,0 | 12,1 à 22,9 |
| cases de la table encore à 0 | 88 % | 88 % |

**Démarche** : a) le L-learning n'explore qu'au hasard des ex æquo. Dès qu'un coup a rapporté une valeur positive depuis un plateau, il bat les cases nulles, et l'agent le rejoue à chaque passage : les autres coups ne sont jamais essayés. Sans hasard, les parties se répètent ; seules changent les valeurs, parce que le TFR d'un coup dépend aussi du nombre de coups joués **avant** lui (la récompense vaut $1/\sqrt{k}$ pour la partie entière). L'agent garde donc ses premiers chemins, souvent longs. b) Avec le camion, une partie coupée à 100 coups écrit des 0 dans toutes ses cases ; ailleurs, des valeurs basses écrasent les bonnes. L'agent glouton peut alors tourner en rond entre deux plateaux sans jamais gagner. c) Une partie touche une dizaine de cases, souvent les mêmes : environ 530 cases sur 4 608 reçoivent une valeur.
**Erreurs fréquentes** : croire, d'après le livre, que 6 000 parties suffisent à jouer juste (il montre deux plateaux, pas les 506) ; penser que le camion ne fait que du mal (il force aussi l'agent à visiter d'autres plateaux : plus de parties optimales qu'avec une table sans camion) ; confondre « gagner » et « gagner vite ».
**Variante** : fais jouer les tables entraînées avec le camion **avec** le camion. Réponse : elles gagnent **99,6 % à 100 %** des plateaux (5 graines) : le camion qui a écrit les boucles les casse aussi, en retournant une tuile au hasard.

### Ex 26.17 — Flippers : la longueur optimale de chaque partie par recherche exhaustive 🔬
**Réponse** : la longueur la plus fréquente est **3** coups (187 plateaux) ; la moyenne sur les 506 plateaux non gagnants vaut **2,70** ; le plateau le plus difficile est le **511**, toutes les tuiles pleines (6 coups).
```python
def min_moves_17():
    dist = np.full(512, -1, dtype=int)
    frontier = sorted(Flippers().winning)
    dist[frontier] = 0
    while frontier:
        nxt = []
        for board in frontier:
            for tile in range(9):
                neighbour = board ^ (1 << tile)
                if dist[neighbour] < 0:
                    dist[neighbour] = dist[board] + 1
                    nxt.append(neighbour)
        frontier = nxt
    return dist
```
**Démarche** : la recherche en largeur visite chaque plateau une seule fois, par couches : 6 plateaux à 0 coup, 54 à 1, 162 à 2, 187 à 3, 87 à 4, 15 à 5 et 1 à 6. Le livre dit que « la plupart » se gagnent en 3 ou 4 coups : c'est vrai de justesse (274 plateaux sur 512), mais 2 coups sont plus fréquents que 4. Le plateau plein demande 6 coups : un plateau gagnant garde 3 points, il faut en effacer 6, et chaque coup n'en efface qu'un. La recherche visite 512 plateaux et 4 608 voisins : au go, le nombre de positions (✏️ 26.2) rend impossible toute énumération.
**Erreurs fréquentes** : une recherche en profondeur, qui trouve un chemin mais pas le plus court ; donner une distance à un plateau déjà atteint (il garderait la plus grande) ; inclure les 6 plateaux gagnants dans la moyenne (2,67) ; confondre la longueur la plus fréquente et la plus longue.
**Variante** : sur un Flippers de 2 × 2 (`Flippers(size=2)`), les 16 plateaux se répartissent en **4** gagnants, **8** à un coup et **4** à deux coups.

### Ex 26.18 — L-learning sur Flippers, sans puis avec camion 🔬
**Réponse** :
```python
def l_learning_18(env, n_games, rng, max_steps=100):
    L = np.zeros((env.observation_space.n, env.action_space.n))
    lengths = []
    for _ in range(n_games):
        state, _ = env.reset(seed=int(rng.integers(2**31)))
        moves, rewards = [], []
        for _ in range(max_steps):
            action = tie_argmax(L[state], rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            moves.append((state, action))
            rewards.append(reward)
            state = next_state
            if terminated or truncated:
                break
        for (s, a), tfr in zip(moves, mylearn.rl.returns_to_go(rewards)):
            L[s, a] = tfr
        lengths.append(len(moves))
    return L, np.array(lengths)
```
**Démarche** : sur les 506 plateaux joués sans camion, l'agent entraîné sans camion gagne toujours, mais en 15,6 coups en moyenne (l'optimum est 2,70), et il ne joue que 7,7 % des plateaux au minimum, **autant après 3 000 qu'après 6 000 parties** : sans exploration, la table ne change presque plus. L'amélioration montrée par le livre entre 3 000 et 6 000 parties ne se retrouve pas sur l'ensemble des plateaux. Avec le camion, l'agent ne gagne que 86 % des plateaux après 3 000 parties et 97 % après 6 000 (une autre graine que celles de 26.16 ; d'une graine à l'autre, cette part varie beaucoup) : des cases écrasées par une partie coupée à 0 créent des boucles. Le camion l'oblige pourtant à visiter d'autres plateaux : il joue 19 % des plateaux au minimum. Le Q-learning répond à chaque défaut : la **moyenne** ($\alpha < 1$) au lieu du remplacement, l'**exploration** ε-greedy, et une cible qui ne dépend que de l'état suivant, pas de toute la partie.
**Erreurs fréquentes** : mettre à jour les cases pendant la partie (ce n'est plus le L-learning) ; faire la moyenne au lieu de remplacer ; écrire les TFR dans l'ordre inverse (un couple joué deux fois garderait la valeur de son **premier** passage) ; oublier que `max_steps` coupe aussi la partie.
**Variante** : ajoute une exploration ε-greedy ($\varepsilon = 0{,}1$) au L-learning sans camion (6 000 parties). Réponse : il joue **14 %** des plateaux au minimum, mais ses parties durent **35,4** coups en moyenne : chaque coup d'exploration allonge la partie et écrase, par remplacement, la valeur de tous les coups joués avant. Explorer ne suffit pas : il faut aussi faire la moyenne.

### Ex 26.19 — Politiques gloutonne et softmax (ε-greedy importée du ch. 11) 🔨
**Réponse** (le cœur de la référence) :
```python
def greedy_policy(Q):
    return np.argmax(_q_values_2d(Q), axis=1)      # 2-D, no NaN; ties go to the lowest index


def softmax_action(q_values, temperature=1.0, rng=None):
    q, T = _q_vector(q_values), _temperature(temperature)   # 1-D, not empty, no NaN; T > 0
    rng = np.random.default_rng() if rng is None else rng
    probs = np.exp((q - q.max()) / T)
    probs /= probs.sum()
    return int(rng.choice(q.size, p=probs))
```
**Démarche** : soustraire le maximum ne change pas les probabilités (✏️ 26.5 g) et évite le débordement : $e^{10\,000}$ vaut l'infini en `float64`. Sur la ligne $(1 ; 2 ; 4 ; 0{,}5)$, ε-greedy ($\varepsilon = 0{,}2$) donne 0,85 à la meilleure action et 0,05 à chacune des autres ; la softmax ($T = 0{,}8$) donne 0,89, 0,07, 0,02 et 0,01 : elle explore surtout l'action presque aussi bonne. La politique d'**évaluation** doit être reproductible : les ex æquo vont au plus petit indice. Pendant l'**entraînement**, au contraire, des ex æquo toujours tranchés de la même façon empêchent d'essayer les autres actions d'une table nulle (26.24).
**Erreurs fréquentes** : diviser par la température **après** l'exponentielle ; normaliser par la somme des valeurs (la « softmax » du livre) ; tirer avec `rng.random()` et une boucle, ce qui consomme les tirages autrement ; rendre un `np.int64`.
**Variante** : pour que la softmax donne, comme ε-greedy, la probabilité 0,85 à la meilleure action de cette ligne, il faut $T \approx$ **0,92**. Les autres actions n'ont pas pour autant les mêmes chances : 0,10, 0,03 et 0,02.

### Ex 26.20 — Mises à jour de Q-learning et de SARSA, états terminaux compris 🔨
**Réponse** (le cœur de la référence) :
```python
def q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma, next_mask=None):
    # checks: Q float ndarray, indices, reward, alpha in (0, 1], gamma in [0, 1], mask (length, not all False)
    if terminated:
        target = reward
    else:
        row = Q[next_state] if next_mask is None else Q[next_state][np.asarray(next_mask).astype(bool)]
        target = reward + gamma * row.max()
    delta = target - Q[state, action]
    Q[state, action] = incremental_update(Q[state, action], target, alpha)
    return float(delta)
```
`sarsa_update` est la même, avec `Q[next_state, next_action]` à la place du max.
**Démarche** : sur la table de la vérification, la cible vaut $0{,}5 + 0{,}8 \times 1 = 1{,}3$, l'erreur $0{,}9$, la case passe de 0,4 à 0,85 ; si $s'$ est terminal, la cible vaut 0,5 et la case 0,45 ; avec le masque, le max ne voit que 0,6 et la case vaut 0,69. $\delta$ se calcule avant la mise à jour : c'est l'erreur de la prédiction qu'on corrige, et une transition vers le même état doit lire l'ancienne valeur. Les deux mises à jour coïncident quand l'action suivante est la gloutonne (✏️ 26.4 d).
**Erreurs fréquentes** : rendre $\delta$ après la mise à jour (il vaut alors $(1 - \alpha)\,\delta$) ; amorcer sur un état terminal ; `int8` utilisé comme indices au lieu d'un masque (`Q[s][mask]` avec un masque `int8` choisit les colonnes 0 et 1, pas les colonnes permises) ; accepter une table d'entiers, où `Q[s, a] += 0.45` est tronqué.
**Variante** : *Expected SARSA* prend l'espérance de $Q(s', \cdot)$ sous la politique au lieu d'un tirage. Avec ε-greedy ($\varepsilon = 0{,}2$) sur la ligne $(1 ; 0{,}6)$, l'espérance vaut $0{,}9 \times 1 + 0{,}1 \times 0{,}6 = 0{,}96$, la cible $0{,}5 + 0{,}8 \times 0{,}96 = 1{,}268$ et l'erreur **0,868**.

### Ex 26.21 — La boucle d'entraînement tabulaire et l'évaluation d'une politique 🔨
**Réponse** (le squelette de la référence ; `_explore` applique ε-greedy à `Q[state, légales]` puis réindexe) :
```python
for episode in range(n_episodes):
    eps = epsilon(episode) if callable(epsilon) else epsilon
    state, info = env.reset(seed=int(rng.integers(2**31)))
    action = _explore(Q, state, info, eps, rng)
    total, steps = 0.0, 0
    for step in range(max_steps):
        next_state, reward, terminated, truncated, next_info = env.step(action)
        total += reward
        steps += 1
        if algorithm == "q_learning":
            q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma,
                              next_mask=_mask_of(next_info))
            if terminated or truncated:
                break
            action = _explore(Q, next_state, next_info, eps, rng)
        else:                                      # SARSA: choose a' first, use it, then play it
            if terminated:
                sarsa_update(Q, state, action, reward, next_state, None, True, alpha, gamma)
                break
            next_action = _explore(Q, next_state, next_info, eps, rng)
            sarsa_update(Q, state, action, reward, next_state, next_action, False, alpha, gamma)
            if truncated:
                break
            action = next_action
        state, info = next_state, next_info
    returns.append(total)
    lengths.append(steps)
```
**Démarche** : les graines de `reset` viennent de `rng` pour que tout l'entraînement se rejoue à l'identique avec la même graine, environnement compris (sur FrozenLake glissant, `reset(seed=...)` fixe aussi le hasard de la glace). Sans glissade, 300 épisodes suffisent : la politique gloutonne atteint le but en 6 pas. Traiter la coupure de `max_steps` comme une fin écrirait une cible $r$ seule pour un état qui avait un avenir : ses valeurs seraient trop basses.
**Erreurs fréquentes** : choisir l'action suivante de SARSA **après** la mise à jour ; utiliser le masque de l'état courant pour le max ; faire un pas de plus après `terminated` ; `env.reset()` sans graine (l'historique ne se rejoue plus) ; un `np.int64` passé comme graine (Gymnasium le refuse) ; un état aléatoire global (`np.random.rand`).
**Variante** : le même entraînement avec `algorithm="sarsa"` atteint aussi le but, en **6** pas : sans glissade ni falaise, l'exploration ne coûte rien de grave, et les deux algorithmes trouvent le même chemin.

### Ex 26.22 — Q-learning sur FrozenLake, glissant ou non 📦
**Réponse** :
```python
def epsilon_schedule_22(n_episodes, start=1.0, end=0.05, fraction=0.8):
    def epsilon(i):
        return max(end, start - (start - end) * i / (fraction * n_episodes))
    return epsilon


def train_lake_22(slippery, n_episodes, seed=0):
    env = make_env("FrozenLake-v1", seed=0, is_slippery=slippery)
    Q, history = mylearn.rl.train_tabular(env, n_episodes, alpha=0.1, gamma=0.99,
                                          epsilon=epsilon_schedule_22(n_episodes), max_steps=100,
                                          rng=np.random.default_rng(seed))
    return env, Q, history
```
**Démarche** : sans glissade, la politique gloutonne réussit 100 % des épisodes, en 6 pas. Sur la glace glissante, elle réussit 74,9 % des épisodes, en 44 pas en moyenne. Elle « va contre le mur » près des trous : pousser vers un mur ne fait pas bouger, et la glissade ne peut alors partir que dans les deux directions perpendiculaires, loin du trou. La courbe d'apprentissage finit vers 0,5 : elle mesure la politique qui explore encore ($\varepsilon = 0{,}05$), et dont les détours coûtent cher sur la glace.
**Erreurs fréquentes** : un calendrier qui atteint `end` à la fin de l'entraînement au lieu de `fraction * n_episodes` ; un `min` au lieu d'un `max` (ε resterait à 0,05 dès le début) ; juger la politique sur la courbe d'entraînement.
**Variante** : avec $\alpha = 0{,}05$, la politique gloutonne du lac glissant réussit **74 %** des épisodes : un learning rate plus petit fait mieux la moyenne des glissades, mais apprend plus lentement, et 4 000 épisodes suffisent encore.

### Ex 26.23 — Lire des courbes d'apprentissage RL bruitées 📈
**Réponse** : a) **C** ; b) **B** ; c) **2** ; d) **Faux**.
**Démarche** : C est la seule courbe qui monte : à mesure que $\varepsilon$ baisse, l'agent joue sa politique, qui s'améliore (vers 0,53 à la fin ; sa politique gloutonne réussit 71 %). B reste basse vers 0,05 pendant tout l'entraînement, mais sa barre monte à 0,60 : avec $\varepsilon = 0{,}5$, l'agent joue au hasard un pas sur deux, ce qui le fait presque toujours tomber, alors que ses valeurs, apprises off-policy, sont bonnes. A et D restent basses toutes les deux : A ($\varepsilon = 0$) n'explore qu'au hasard des ex æquo, trouve parfois le but, et sa courbe égale sa barre (0,10 et 0,09) ; D ($\alpha = 1$) copie à chaque pas la dernière glissade : ses valeurs ne font jamais la moyenne, et sa politique ne réussit jamais. Deux barres dépassent 0,5 (B et C). Pour B, la courbe sous-estime la politique d'environ 0,55 : une courbe d'entraînement mesure la politique qui explore, pas la politique apprise. Pour suivre celle-ci, on évalue de temps en temps la politique gloutonne, sans exploration, sur des épisodes à graines fixées.
**Erreurs fréquentes** : prendre la courbe la plus haute pour la meilleure politique ; prendre B pour un échec ; confondre A et D sans regarder les barres.
**Variante** : quelles courbes estiment bien leur politique ? Réponse : **A et D**, les deux qui n'apprennent presque rien : quand l'exploration est nulle ou rare et que la politique ne change plus, la courbe et la barre mesurent la même chose.

### Ex 26.24 — L'agent qui n'apprenait rien 🐛
**Réponse** : **« EGHI »**.
```python
def q_learning_fixed_24(env, n_episodes, alpha=0.1, gamma=0.99, epsilon=0.1, max_steps=100, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(n_episodes):
        state, _ = env.reset(seed=int(rng.integers(2**31)))
        for step in range(max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = tie_argmax(Q[state], rng)                       # (E) ties at random
            next_state, reward, terminated, truncated, _ = env.step(action)
            bootstrap = gamma * np.max(Q[next_state])                    # (G) the NEXT state
            target = reward if terminated else reward + bootstrap       # (H) no future after a real end
            Q[state, action] += alpha * (target - Q[state, action])      # (I) removed: a truncation keeps it
            state = next_state
            if terminated or truncated:
                break
    return Q
```
**Démarche** : (E) `np.argmax` d'une ligne nulle rend toujours 0 : avec une table nulle et $\varepsilon = 0$, l'agent joue toujours la même action (point 1). (G) la cible lit la ligne de l'état **courant** : la valeur d'un état ne remonte jamais vers celui qui le précède (point 2). (H) la cible amorce même après `terminated` : invisible quand l'état terminal a une ligne nulle, comme sur FrozenLake, mais faux dès que l'observation finale est aussi un état ordinaire, comme une case de CartPole découpé (point 3). (I) une coupure est traitée comme une fin : les états proches de la limite de temps sont sous-estimés (point 4). (G) suffit à lui seul à tout bloquer : le code fautif ne réussit jamais sur le lac glissant (0 %), ta version 75,1 %.
**Erreurs fréquentes** : désigner (L) (`terminated or truncated` est juste pour **arrêter** la boucle) ; corriger (H) en traitant aussi `truncated` comme une fin ; changer les tirages de `rng` (les diagnostics ne seraient plus comparables).
**Variante** : ne corrige que (E). Réponse : la réussite reste à **0 %** : sans la bonne ligne dans la cible, rien ne remonte du but vers le départ, quel que soit le départage des ex æquo.

### Ex 26.25 — Q-learning sur Flippers avec camion : 300, 3 000 puis 6 000 parties 🔬
**Réponse** :
```python
def train_in_stages_25(stages, seed=0, truck_prob=0.1, **kwargs):
    env = Flippers(truck_prob=truck_prob)
    tables, Q, done = {}, None, 0
    for k, n in enumerate(stages):
        Q, _ = mylearn.rl.train_tabular(env, n - done, max_steps=100, Q_init=Q, rng=np.random.default_rng(seed + k),
                                        **kwargs)
        done = n
        tables[n] = Q.copy()
    return tables
```
**Démarche** : avec les réglages du livre, la longueur moyenne passe de 6,06 coups (300 parties) à 4,08 (3 000) puis 3,47 (6 000), pour un optimum de 2,70 sans camion ; la part des parties jouées au minimum passe de 29 % à 62 %, les cases non nulles de 8 % à 45 %. La plus longue partie vaut 24, puis 35, puis 10 coups : un maximum dépend d'un seul plateau malchanceux et monte ou descend sans rien dire de la tendance. Les valeurs se groupent en paquets : avec $\gamma = 0{,}2$ et une récompense finale, une case vaut à peu près $\gamma^{d-1}/\sqrt{k}$, et chaque distance $d$ au but donne sa bande. Avec le camion, même une politique parfaite perd parfois un coup ; et 6 000 parties ne visitent pas encore la moitié de la table.
**Erreurs fréquentes** : repartir d'une table nulle à chaque étape ; garder la même référence au lieu d'une copie (toutes les étapes montreraient la table finale) ; réutiliser le même générateur à chaque étape (les tirages se répètent).
**Variante** : avec $\alpha = 0{,}5$ au lieu de 0,95 (6 000 parties), la longueur moyenne vaut **3,58** coups.

### Ex 26.26 — Itération sur les valeurs : la référence quand le modèle est connu 🔨
**Réponse** (le cœur de la référence) :
```python
def backup(V):
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = sum(p * (r + gamma * V[s2] * (1 - done)) for p, s2, r, done in P[s][a])
    return Q


V = np.zeros(n_states)
for _ in range(max_iter):
    V_new = backup(V).max(axis=1)
    change = np.abs(V_new - V).max()
    V = V_new
    if change < theta:
        break
return V, backup(V)
```
**Démarche** : sur la chaîne, $V^* = (0{,}5 ; 1 ; 0)$ avec $\gamma = 0{,}5$, et un seul balayage ne donne une valeur qu'à l'état qui précède la récompense : la valeur remonte d'un état par balayage, comme dans ✏️ 26.6. Sur le lac glissant, $V^*(0) = 0{,}54$ avec $\gamma = 0{,}99$, et la politique optimale réussit 73,2 % des épisodes de 100 pas. Elle va « contre le mur » à côté des trous, pour ne glisser que vers des cases sûres. La table de 26.22 choisit la même action dans les 16 états ; ses valeurs s'écartent de $Q^*$ de 0,18 au plus, surtout dans les cases peu visitées, que la politique gloutonne évite et dont les valeurs ont reçu peu de mises à jour.
**Erreurs fréquentes** : mettre à jour $V$ sur place pendant le balayage (c'est Gauss-Seidel : ça converge aussi, mais `max_iter=k` ne rend plus le $k$-ième itéré) ; oublier le facteur `(1 - done)` ; s'arrêter quand la variation vaut **exactement** `theta` ; rendre le `Q` du balayage d'avant.
**Variante** : avec $\gamma = 0{,}9$, $V^*(0)$ ne vaut plus que **0,069**, et la politique optimale change dans un état sur seize. Elle réussit à peine moins souvent : sur 20 000 épisodes (graines 0 à 19 999), **72,7 %** contre 73,6 % avec $\gamma = 0{,}99$, à ± 0,3 point près ; sur 2 000 épisodes, l'écart serait noyé dans le bruit. Un but lointain compte moins : le long détour sûr coûte trop cher, et l'agent prend un chemin plus court et plus risqué.

### Ex 26.27 — α, γ, ε : balayage d'hyperparamètres sur Flippers bruité 🔬
**Réponse** :
```python
def sweep_27(settings, n_games, seeds, epsilon=0.1):
    results = {}
    for alpha, gamma, truck in settings:
        lengths = []
        for seed in seeds:
            Q, _ = mylearn.rl.train_tabular(Flippers(truck_prob=truck, reward="win"), n_games, alpha=alpha, gamma=gamma,
                                            epsilon=epsilon, max_steps=100, rng=np.random.default_rng(seed))
            lengths.append(float(evaluate_flippers(Q, truck_prob=truck, seed=seed)["length"].mean()))
        results[(alpha, gamma, truck)] = lengths
    return results
```
**Démarche** : avec le camion rare (0,1), les trois $\alpha$ font jeu égal (3,8 à 4,0 coups). Avec un camion une fois sur deux, $\alpha = 0{,}95$ fait un peu moins bien (5,4 et 5,9 contre 4,6 à 5,5). Avec un camion à chaque coup, il décroche nettement (14,1 contre 8,8 à 11,5). Le hasard de Flippers ne change qu'une tuile après certains coups, et la cible du Q-learning ne dépend que de l'état suivant : une seule mise à jour bruitée ne fausse qu'une case, et la suivante la répare. Sur le lac glissant (26.23), au contraire, deux pas sur trois partent de travers, et seul un petit $\alpha$ fait la moyenne. $\gamma = 0{,}9$ fait moins bien que $\gamma = 0{,}2$ partout, avec ce budget de 3 000 parties : ses valeurs convergent plus lentement, et les écarts entre deux distances au but sont plus petits que le bruit d'apprentissage. Deux graines suffisent quand l'écart est grand (camion 1), pas quand il est de quelques dixièmes de coup.
**Erreurs fréquentes** : réutiliser la même table d'un réglage à l'autre ; évaluer sans le camion de l'entraînement ; conclure sur une seule graine ; comparer des réglages évalués avec des graines différentes.
**Variante** : avec $\varepsilon = 0{,}3$ (au lieu de 0,1), $\alpha = 0{,}3$, $\gamma = 0{,}2$ et le camion à 0,5, la longueur moyenne tombe à **3,9 et 4,1** coups : avec peu de parties, explorer davantage aide plus que régler $\alpha$.

### Ex 26.28 — Prédire le chemin appris par SARSA et par Q-learning au bord de la falaise 🔮
**Réponse** : a) **C** ; b) **B** ; c) **C**.
**Démarche** : SARSA apprend un chemin de 17 pas par la rangée du haut ; le Q-learning longe la falaise en 13 pas. Pendant l'entraînement, le Q-learning tombe 191 fois en 500 épisodes, SARSA 48 fois (4 fois moins), et le retour moyen des 100 derniers épisodes vaut −48,7 contre −25,4. SARSA apprend la valeur de la politique qu'il suit, exploration comprise : près du bord, un pas d'exploration vers le bas coûte −100, et deux explorations de suite peuvent faire tomber depuis la 2ᵉ rangée ; la rangée du haut est la plus sûre. Sur huit graines, SARSA fait 17 ou 19 pas (et une fois tourne en rond), le rapport des chutes va de 2,2 à 4,4, et le Q-learning tombe 142 à 204 fois. Avec un $\varepsilon$ qui décroît vers 0, SARSA finirait par apprendre le chemin le plus court, mais lentement.
**Erreurs fréquentes** : croire que SARSA s'écarte d'une seule rangée ; confondre le chemin glouton (sans exploration) et les épisodes d'entraînement ; oublier que le Q-learning tombe aussi pendant les premiers épisodes, avant d'avoir appris.
**Variante** : avec un $\varepsilon$ qui décroît linéairement de 0,1 à 0 en 400 épisodes, SARSA garde un chemin de **17** pas après 500 épisodes : ses valeurs, apprises sous exploration, ne se corrigent pas en 100 épisodes sans exploration ni chute.

### Ex 26.29 — Q-learning contre SARSA sur Flippers et CliffWalking 🔬
**Réponse** :
```python
def compare_29(make, n_episodes, seeds, score, **train_kwargs):
    results = {}
    for algorithm in ("q_learning", "sarsa"):
        online, greedy = [], []
        for seed in seeds:
            env = make()
            Q, history = mylearn.rl.train_tabular(env, n_episodes, algorithm=algorithm,
                                                  rng=np.random.default_rng(seed), **train_kwargs)
            online.append(float(np.mean(history["returns"][-100:])))
            greedy.append(float(score(env, Q)))
        results[algorithm] = {"online": online, "greedy": greedy}
    return results
```
**Démarche** : sur CliffWalking, SARSA gagne pendant l'entraînement (−25,5 contre −54,5) et perd après (−17 contre −13, sur les 3 graines) : chacun est le meilleur sur l'objectif qu'il apprend. Sur Flippers avec camion, une erreur d'exploration ne coûte qu'un coup de plus : les retours d'entraînement sont presque égaux (0,53 et 0,51), et la politique gloutonne du Q-learning joue plus court (3,91 coups contre 4,38), avec un écart plus grand que celui des graines (3,74 à 4,01 contre 4,30 à 4,52). Avec 3 000 parties, le livre aurait conclu l'inverse de ce qu'il écrit ; pour conclure, il lui aurait fallu plusieurs graines, une même évaluation pour les deux, et préciser ce qu'il mesure (avec ou sans exploration).
**Erreurs fréquentes** : réutiliser l'environnement d'un entraînement à l'autre sans `reset` (`make()` à chaque fois) ; comparer des « online » qui n'ont pas la même exploration ; inverser le signe du score de Flippers.
**Variante** : avec $\varepsilon = 0{,}01$ sur CliffWalking, les retours d'entraînement se rapprochent (−15,6 et −16,1 en moyenne), et SARSA apprend un chemin de **15** pas : moins d'exploration, moins de prudence.

### Ex 26.30 — Reproduire les comparaisons au fil de l'entraînement (fig. 26.58-26.60) 🎨
**Réponse** :
```python
def measure_30(tables, truck_prob=0.1):
    curves = {"games": [], "length": [], "mean_q": [], "nonzero": []}
    for n, Q in tables.items():
        curves["games"].append(n)
        curves["length"].append(float(evaluate_flippers(Q, truck_prob=truck_prob, seed=30)["length"].mean()))
        curves["mean_q"].append(float(Q.mean()))
        curves["nonzero"].append(float(np.mean(Q != 0)))
    return curves
```
`plot_comparison_30` trace les trois mesures avec `ax.set_xscale("log")` (voir le notebook de solutions).
**Démarche** : à 6 000 parties, SARSA joue un peu plus long que le Q-learning (3,92 contre 3,46 coups), mais les courbes se croisent (à 2 000 parties, c'est SARSA qui est devant : 4,40 contre 4,56) ; leurs valeurs moyennes restent indiscernables (vers 0,009) et leurs parts de cases non nulles égales (44 %). On ne retrouve pas les trois écarts du livre, qui allaient tous en faveur de SARSA. En principe, la cible de SARSA prend l'action tirée, parfois mauvaise, au lieu du maximum, ce qui tire ses valeurs vers le bas ; avec $\varepsilon = 0{,}1$ et $\gamma = 0{,}2$, cet effet est trop petit pour se voir ici. Une seule graine ne suffit pas pour trancher un écart de quelques dixièmes (26.29).
**Erreurs fréquentes** : la moyenne des seules cases non nulles (une autre mesure que celle du livre) ; une échelle linéaire, qui écrase les premières étapes ; des tables entraînées séparément pour chaque étape (une autre expérience).
**Variante** : avec $\alpha = 0{,}3$, les deux algorithmes jouent en **3,40** (Q-learning) et **3,48** coups (SARSA) après 6 000 parties : l'écart fond.

### Ex 26.31 — Une mémoire de rejeu 🔨
**Réponse** (le cœur de la référence) :
```python
def push(self, state, action, reward, next_state, terminated):
    # checks: shapes equal to obs_shape
    self._states[self._pos] = state
    self._actions[self._pos] = action
    self._rewards[self._pos] = reward
    self._next_states[self._pos] = next_state
    self._terminated[self._pos] = terminated
    self._pos = (self._pos + 1) % self.capacity
    self._size = min(self._size + 1, self.capacity)


def sample(self, batch_size):
    # checks: an int, not a bool, 1 <= batch_size <= len(self)
    idx = self._rng.choice(self._size, size=batch_size, replace=False)
    return {"states": self._states[idx], "actions": self._actions[idx], "rewards": self._rewards[idx],
            "next_states": self._next_states[idx], "terminated": self._terminated[idx]}


def __len__(self):
    return self._size
```
**Démarche** : la mémoire est une file circulaire : la 4ᵉ transition écrase la 1ʳᵉ, la 5ᵉ la 2ᵉ. L'indexation par un tableau d'indices crée une copie : le mini-batch peut être modifié sans toucher la mémoire. Une vue serait un piège : un entraînement qui normalise ses états sur place modifierait la mémoire elle-même. Sans remise, un mini-batch ne compte jamais deux fois la même transition dans une même moyenne de gradient.
**Erreurs fréquentes** : tirer parmi `range(self.capacity)` avant que la mémoire soit pleine (des transitions vides) ; oublier de borner `_size` ; `np.random.choice` au lieu de `self._rng` (pas reproductible) ; rendre des tranches `self._states[:n]` (des vues).
**Variante** : avec remise, un mini-batch de 32 tiré dans 1 000 transitions contient au moins un doublon avec la probabilité $1 - \prod_{i < 32}(1 - i/1000) \approx$ **0,39** (le paradoxe des anniversaires).

### Ex 26.32 — Concevoir la récompense : finale, pénalité par coup ou façonnée 🔬
**Réponse** :
```python
class DotBonus_32(gym.Wrapper):
    def __init__(self, env, bonus=DOT_BONUS_32):
        super().__init__(env)
        self.bonus = bonus

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self._dots = bin(int(state)).count("1")
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        dots = bin(int(state)).count("1")
        if dots > self._dots:
            reward += self.bonus
        self._dots = dots
        return state, reward, terminated, truncated, info
```
**Démarche** (3 000 parties, 2 graines) : la récompense finale $1/\sqrt{k}$ du livre ne gagne que 87 % des plateaux, en 17 coups. Elle dépend du nombre de coups déjà joués, que l'état ne contient pas : la même case reçoit des cibles différentes selon la partie, et la table ne peut pas les représenter. « Victoire 1 » gagne tout, en 4,60 coups ; avec un coût par coup, 3,71 ; façonnée, 3,48, la plus proche de l'optimum (2,70) : le bonus donne à chaque pas l'information « tu t'es rapproché », que l'agent devrait sinon découvrir par la remontée lente des valeurs. L'agent payé pour les points ne gagne que 4,5 % des plateaux, en 95,5 coups : ajouter un point, l'enlever, le remettre lui rapporte 0,2 tous les deux coups, soit environ $0{,}2/(1 - 0{,}81) \approx 1{,}05$ avec $\gamma = 0{,}9$ : plus qu'une victoire à plusieurs coups de distance, que l'actualisation réduit. Pour lui, c'est la bonne stratégie : c'est la récompense qui est mauvaise (⚖️ 26.11).
**Erreurs fréquentes** : compter les points avant le coup dans `step` sans les avoir mémorisés dans `reset` ; donner le bonus quand un point disparaît ; évaluer sur le jeu à bonus (on mesurerait la récompense, pas la tâche).
**Variante** : avec un bonus de 0,05, l'agent gagne tout, mais en **7,6** coups au lieu de 4,6 : un bonus trop petit pour qu'on le farme en boucle suffit encore à allonger les parties.

### Ex 26.33 — Rejouer l'expérience pour apprendre avec moins d'épisodes 🔬
**Réponse** :
```python
def train_with_replay_33(env, n_episodes, replay_updates, *, alpha=0.1, gamma=0.99, epsilon=0.1,
                         capacity=10_000, max_steps=100, seed=0):
    R = mylearn.rl
    rng = np.random.default_rng(seed)
    buffer = R.ReplayBuffer(capacity, random_state=seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = []
    for episode in range(n_episodes):
        eps = epsilon(episode) if callable(epsilon) else epsilon
        state, _ = env.reset(seed=int(rng.integers(2**31)))
        total = 0.0
        for _ in range(max_steps):
            action = mylearn.bandit.epsilon_greedy_action(Q[state], eps, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            R.q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma)
            buffer.push(state, action, reward, next_state, terminated)
            if replay_updates > 0 and len(buffer) >= replay_updates:
                batch = buffer.sample(replay_updates)
                for i in range(replay_updates):
                    R.q_learning_update(Q, int(batch["states"][i]), int(batch["actions"][i]), float(batch["rewards"][i]),
                                        int(batch["next_states"][i]), bool(batch["terminated"][i]), alpha, gamma)
            state = next_state
            if terminated or truncated:
                break
        returns.append(total)
    return Q, np.array(returns)
```
**Démarche** : en 500 épisodes, la politique gloutonne réussit 47 %, 4 % et 25 % du temps sans rejeu (25 % en moyenne), 68 %, 50 % et 70 % avec 8 mises à jour rejouées par pas (63 %), près de l'optimum de 73 %. La version avec rejeu fait 9 fois plus de mises à jour : une partie du gain vient de ce nombre. Mais 9 fois plus d'**épisodes** coûterait 9 fois plus d'interactions avec l'environnement, la ressource chère d'un robot ou d'un jeu lent : le rejeu tire plus d'apprentissage de chaque expérience vécue. Il suppose une méthode off-policy : les transitions rejouées ont été produites par d'anciennes politiques, et une méthode off-policy, comme le Q-learning, les utilise correctement. DQN y ajoute ce qu'une table ne demande pas : casser les corrélations entre transitions successives, qui déstabilisent la descente de gradient d'un réseau.
**Erreurs fréquentes** : rejouer avant que la mémoire contienne assez de transitions (`sample` lève `ValueError`) ; utiliser `rng` pour la mémoire (les tirages ne suivent plus `train_tabular`) ; oublier la mise à jour sur la transition vécue.
**Variante** : avec 4 mises à jour rejouées par pas, la réussite vaut **68 %, 61 % et 60 %** : l'essentiel du gain vient dès les premières mises à jour rejouées.

### Ex 26.34 — Rapporter un résultat RL honnête : plusieurs graines, configuration, intervalle 🛠️
**Réponse** :
```python
def run_34(config, seed):
    R = mylearn.rl
    env = make_env(config["env"], seed=0, is_slippery=config["is_slippery"])
    schedule = epsilon_schedule_22(config["n_episodes"], config["epsilon_start"], config["epsilon_end"],
                                   config["epsilon_fraction"])
    Q, _ = R.train_tabular(env, config["n_episodes"], algorithm=config["algorithm"], alpha=config["alpha"],
                           gamma=config["gamma"], epsilon=schedule, max_steps=config["max_steps"],
                           rng=np.random.default_rng(seed))
    score = R.evaluate_policy(env, R.greedy_policy(Q), config["eval_episodes"], max_steps=config["max_steps"],
                              rng=np.random.default_rng(config["eval_seed"]))
    return float(score["mean_return"])


def report_34(config, seeds, path):
    results = [run_34(config, seed) for seed in seeds]
    low, high = mylearn.stats.bootstrap_ci(results, confidence=0.95, n_boot=2000, rng=np.random.default_rng(0))
    report = {"config": config, "seeds": list(seeds), "results": results, "mean": float(np.mean(results)),
              "ci95": [float(low), float(high)]}
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
```
**Démarche** : les cinq graines donnent 57 %, 35 %, 74 %, 74 % et 38 % : une seule graine publiée aurait pu annoncer 35 % comme 74 %. La moyenne vaut 55,6 %, avec un intervalle bootstrap à 95 % de [41,2 % ; 70,6 %]. Avec 5 valeurs, le bootstrap percentile est grossier et trop étroit : il ne peut pas sortir de l'intervalle des valeurs observées (ch. 2). Les graines d'évaluation sont les mêmes pour toutes les méthodes comparées : l'écart mesuré vient alors des politiques, pas de la chance de l'évaluation. Pour refaire l'expérience dans un an, le fichier devrait aussi contenir les versions de Gymnasium et de NumPy, et le commit du code (`git rev-parse HEAD`).
**Erreurs fréquentes** : une graine d'évaluation qui change avec la graine d'entraînement ; garder la meilleure graine (*cherry-picking*) ; écrire la configuration à la main dans le rapport au lieu de l'objet utilisé ; un `np.float64` qui ne passe pas dans `json.dumps`.
**Variante** : avec 10 graines, la moyenne vaut **56,5 %** et l'intervalle **[46,9 % ; 65,5 %]** : plus étroit, et toujours aussi loin de la meilleure graine (74 %).

### Ex 26.35 — Morpion contre un adversaire aléatoire : masque d'actions ou pénalité 📦
**Réponse** :
```python
def play_ttt_35(env, Q, n_games, seed=0):
    rng = np.random.default_rng(seed)
    with_mask = masked_greedy(Q)
    counts = {"win": 0, "draw": 0, "loss": 0, "illegal": 0}
    for _ in range(n_games):
        state, info = env.reset(seed=int(rng.integers(2**31)))
        terminated = truncated = False
        while not (terminated or truncated):
            action = with_mask(state, info) if "action_mask" in info else int(np.argmax(Q[state]))
            state, _, terminated, truncated, info = env.step(action)
        counts[info["result"]] += 1
    return counts
```
**Démarche** : après 20 000 parties, l'agent à masque gagne 98,2 % des parties contre l'adversaire aléatoire (1,8 % de nuls) ; l'agent à pénalité 95,5 %, et 2,5 % de ses parties finissent encore sur un coup interdit : il a dû apprendre les règles en plus du jeu, case par case, et une case jamais visitée ne sait pas qu'un coup est interdit. Contre minimax, l'agent à masque fait 100 % de nuls : il commence, et les positions où minimax le mène sont simples. L'agent à pénalité perd 23,5 % des parties, et 24 % finissent sur un coup interdit : un joueur parfait l'amène dans des positions qu'un adversaire aléatoire ne lui montrait presque jamais. Les deux ont appris à battre **cet** adversaire ; 26.37 apprend le jeu en jouant contre soi-même.
**Erreurs fréquentes** : prendre l'argmax de toute la ligne avec un masque (des coups interdits, qui lèvent `ValueError`) ; oublier la clé `"illegal"` à 0 ; une graine de `reset` fixe pour toutes les parties (l'adversaire jouerait toujours la même chose).
**Variante** : entraîné directement contre minimax (20 000 parties), l'agent à masque fait 100 % de nuls contre lui, mais ne gagne que **80,4 %** des parties contre l'adversaire aléatoire et en perd **11,8 %** : il n'a jamais vu les fautes d'un débutant.

### Ex 26.36 — CartPole : discrétiser un état continu pour un Q-learning tabulaire 📦
**Réponse** :
```python
class Discretize_36(gym.ObservationWrapper):
    def __init__(self, env, edges):
        super().__init__(env)
        self.edges = [np.asarray(e, dtype=float) for e in edges]
        self.dims = tuple(len(e) + 1 for e in self.edges)
        self.observation_space = gym.spaces.Discrete(int(np.prod(self.dims)))

    def observation(self, observation):
        indices = [int(np.digitize(value, e)) for value, e in zip(observation, self.edges)]
        return int(np.ravel_multi_index(indices, self.dims))
```
**Démarche** : 800 épisodes (une quinzaine de secondes) suffisent pour une politique gloutonne qui tient 199,7 pas en moyenne sur 100 épisodes coupés à 200 ; 60 % des 144 cases ont une valeur. L'état terminal (le bâton tombé) tombe dans la case d'un angle extrême, que visitent aussi des états non terminaux : amorcer sur lui serait le bug (H) de 26.24. L'angle et sa vitesse décident de la chute ; la position compte peu sur 200 pas, et chaque intervalle en plus multiplie le nombre de cases.
**Erreurs fréquentes** : passer aussi les bornes extrêmes (−0,2 et 0,2 pour l'angle) : un intervalle de plus de chaque côté, où ne tombent que les observations où le bâton est déjà tombé ; un `np.int64` comme état ; oublier de changer `observation_space` (`train_tabular` lit son `n`) ; inverser l'ordre des dimensions entre `digitize` et `ravel_multi_index`.
**Variante** : avec 20 intervalles pour l'angle et pour sa vitesse (1 200 cases), les mêmes 800 épisodes ne tiennent plus que **178** pas en moyenne : 40 % des cases seulement ont une valeur, et chacune reçoit moins de mises à jour.

### Ex 26.37 — Auto-apprentissage au morpion : l'agent joue les deux camps 🔬
**Réponse** :
```python
def self_play_37(n_games, alpha=0.5, gamma=0.9, epsilon=0.2, seed=0):
    legal_table, winner, swap = TABLES_37["legal"], TABLES_37["winner"], TABLES_37["swap"]
    rng = np.random.default_rng(seed)
    Q = np.zeros((3**9, 9))
    for _ in range(n_games):
        c = 0
        while True:
            legal = np.flatnonzero(legal_table[c])
            if rng.random() < epsilon:
                a = int(rng.choice(legal))
            else:
                a = int(legal[tie_argmax(Q[c, legal], rng)])
            after = c + 3**a
            c2 = int(swap[after])
            if winner[after] == 1:
                target, done = 1.0, True
            elif not legal_table[after].any():
                target, done = 0.0, True
            else:
                target, done = -gamma * Q[c2, legal_table[c2] == 1].max(), False
            Q[c, a] += alpha * (target - Q[c, a])
            if done:
                break
            c = c2
    return Q
```
`vs_minimax_37` joue avec `legal[np.argmax(Q[canonical_37(state), legal])]` (voir le notebook de solutions).
**Démarche** : après 50 000 parties (10 s), l'agent fait 100 nuls sur 100 contre minimax, en X comme en O : aucune défaite. Le signe moins traduit la somme nulle : ce que vaut pour l'adversaire sa meilleure réponse, tu le perds. Les ronds sont plus difficiles : ils jouent en second, et une seule erreur contre un joueur parfait suffit à perdre ; avec 20 000 parties, l'agent perd encore 9 parties sur 100 en O. Un quart seulement des cases légales ont reçu une valeur : la plupart des 19 683 codes ne sont pas atteignables, ou pas par un jeu raisonnable. Au go, la table est impossible (✏️ 26.2) : AlphaGo Zero remplace la table par un réseau qui reçoit la position, et guide une recherche dans l'arbre des coups (fiche §26.10).
**Erreurs fréquentes** : oublier d'échanger les marques (les deux joueurs écrivent dans les mêmes cases, chacun avec son propre point de vue) ; un signe plus dans la cible (l'agent aide l'adversaire) ; mettre à jour des coups interdits ; prendre le max sur toute la ligne de `c2`.
**Variante** : avec $\varepsilon = 0{,}05$ au lieu de 0,2 (50 000 parties), l'agent ne perd toujours pas (**100 nuls** dans chaque camp), mais n'a donné une valeur qu'à **21 %** des cases légales (25 % avec $\varepsilon = 0{,}2$) : au morpion, explorer peu suffit, parce que minimax ne mène que dans des positions que le jeu contre soi-même visite déjà.

### Ex 26.38 — Défi CartPole tabulaire : tenir 475 pas en moyenne sur 500 🏆
**Réponse** (une solution parmi d'autres) :
```python
def make_cartpole_38():
    edges = [[], [], [-0.05, 0.0, 0.05], [-0.5, 0.5]]   # the angle (4 intervals) and its speed (3) only: 12 states
    return Discretize_36(gym.make("CartPole-v1"), edges)


def train_38(env):
    n = 800
    Q, _ = mylearn.rl.train_tabular(env, n, alpha=0.2, gamma=0.99, epsilon=epsilon_schedule_22(n, 1.0, 0.01, 0.5),
                                    max_steps=200, rng=np.random.default_rng(2638))
    return Q
```
**Démarche** : la table de 26.36 (144 cases) tient 200 pas, mais seulement 220 à 360 pas en moyenne sur des épisodes de 500 (selon la graine) : beaucoup de ses cases sont visitées trop peu pour que leurs valeurs soient justes, et le chariot finit par dériver dans une case mal apprise. Avec 12 cases seulement (l'angle en 4 intervalles, sa vitesse en 3, le chariot ignoré), chaque case reçoit des milliers de mises à jour : en 800 épisodes (une quinzaine de secondes), la politique tient 500 pas sur les graines publiques comme sur les graines cachées, et la même recette réussit avec trois graines d'entraînement (`history_ch26.json`). Une table plus fine pourrait mieux faire, mais demanderait beaucoup plus d'épisodes : c'est le compromis entre la finesse d'une représentation et la quantité de données, que les réseaux de neurones (DQN, B8) déplacent en généralisant d'une case à ses voisines.
**Erreurs fréquentes** : affiner encore la grille de 26.36 (des cases jamais visitées) ; des épisodes d'entraînement de 500 pas (plus lents, sans gain) ; régler sur les graines publiques jusqu'à les réussir par chance ; évaluer avec exploration.
**Variante** : avec la grille de 26.36 (144 cases) mais 1 500 épisodes, la politique tient aussi **500** pas en moyenne : avec assez de données, la grille plus fine rattrape la grille grossière, en deux fois plus de temps.
