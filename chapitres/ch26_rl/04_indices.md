# 26 · Apprentissage par renforcement — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 26.Q1 — Apprendre avec des récompenses : ni supervisé, ni non supervisé

<details><summary>Indice 1</summary>

Relis les §26.1 et §26.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Demande-toi ce que reçoit l'apprenant dans chaque cadre : la bonne réponse (supervisé), rien (non supervisé) ou une note sur ce qu'il a fait (RL). Un problème relève du RL quand on **agit**, qu'on ne connaît pas la meilleure action et qu'on peut juger le résultat.

</details>
<details><summary>Indice 3</summary>

a) Le RL ne reçoit jamais la bonne action : seulement un signal « mieux » ou « moins bien » (B). Applique le même test aux cinq problèmes de b) : lesquels demandent d'agir, sans label ? Pour c), pense à une partie gagnée au dernier coup ; pour d), relis la première phrase du §26.1 ; pour e), relis ce que le §26.2 dit de la « bonne » réponse.

</details>

### 26.Q2 — Les quatre informations données à l'agent du morpion

<details><summary>Indice 1</summary>

Relis le §26.2.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Dans l'exemple, les rôles sont inversés par rapport à l'intuition : celui qui **apprend** est l'agent, celui qui répond et note est l'environnement. Pour b), l'agent ne sait même pas qu'il joue : on ne lui dit que ce qui lui permet de choisir et d'être noté.

</details>
<details><summary>Indice 3</summary>

a) L'amie apprend : c'est l'agent ; nous jouons les ronds et donnons les récompenses : c'est l'environnement (B). Pour b), garde le plateau, les coups possibles, la récompense et la consigne de viser les grandes récompenses. Pour c), relis ce que le livre appelle les informations **privées** ; pour d), à quoi servent des coups interdits notés 0 ? Pour e), compte les cases vides.

</details>

### 26.Q3 — État, actions, politique et informations privées

<details><summary>Indice 1</summary>

Relis les §26.3 et §26.3.1 de la fiche, et la figure de la boucle.

</details>
<details><summary>Indice 2</summary>

La boucle a trois temps : choisir, répondre, se mettre à jour. La politique est une **règle** : un état en entrée, une action en sortie. L'environnement exécute l'action, parce que le monde peut changer pendant qu'elle se déroule.

</details>
<details><summary>Indice 3</summary>

a) L'agent choisit d'abord (C), puis l'environnement répond (A), puis l'agent se met à jour (B) : `"CAB"`. Pour b), une partie complète porte un nom propre au RL, différent de l'epoch ; pour c), la règle état → action ; pour d), qui pose la croix au morpion ? Pour e), relis l'exemple du jeu de société au §26.3.1.

</details>

### 26.Q4 — Réponse de l'environnement, épisode et agents libres ou déclenchés

<details><summary>Indice 1</summary>

Relis les §26.3.2 à §26.3.4 de la fiche, et l'encadré 🕰️ sur Gymnasium.

</details>
<details><summary>Indice 2</summary>

« Libre » : agit de lui-même, en continu. « Déclenché » : attend qu'on le sollicite. Le cas le plus simple à simuler est celui où personne n'agit sans que l'autre ait fini.

</details>
<details><summary>Indice 3</summary>

a) L'environnement renvoie ce qui permet à l'agent de continuer et d'être noté : le nouvel état et une récompense (B). Pour b), le cuisinier attend la sonnerie, le minuteur tourne seul ; pour c), chacun attend son tour ; pour d), relis la fin du §26.3.4 ; pour e), Gymnasium distingue deux façons de finir un épisode et ajoute un dictionnaire `info`.

</details>

### 26.Q5 — MDP, observabilité, attribution du crédit, exploration ou exploitation

<details><summary>Indice 1</summary>

Relis le §26.3.5 de la fiche et son encadré ⚠️ sur les MDP.

</details>
<details><summary>Indice 2</summary>

La propriété de Markov parle de **mémoire** : le passé n'ajoute rien à l'état présent. L'observabilité dit si l'agent voit tout l'état. Les deux difficultés du RL : à qui revient le mérite d'une victoire, et quand essayer du nouveau.

</details>
<details><summary>Indice 3</summary>

a) Le présent suffit, le passé n'apporte rien de plus (B). Pour b), l'agent voit-il les cartes des autres ? Pour c), il s'agit de répartir le mérite entre les 40 coups ; pour d), demande-toi si une politique gloutonne essaie un jour une action qu'elle croit mauvaise ; pour e), quel problème du ch. 11 n'a qu'un seul état ?

</details>

### 26.Q6 — Retour total, retour actualisé et rôle de γ

<details><summary>Indice 1</summary>

Relis le §26.3.7 de la fiche, son encadré 🧮 et l'encadré ⚠️ sur γ.

</details>
<details><summary>Indice 2</summary>

Le retour actualisé pèse la récompense reçue $k$ pas plus tard par $\gamma^k$, en commençant à $k = 0$ pour la récompense immédiate.

</details>
<details><summary>Indice 3</summary>

a) $2 + 0 + 1 = 3$. Pour b), $2 + 0{,}5 \times 0 + 0{,}5^2 \times 1$ ; pour c), que valent les poids $\gamma^k$ pour $k \ge 1$ quand $\gamma = 0$ ? Pour d), relis l'encadré qui corrige le livre ; pour e), $1/(1 - 0{,}95)$.

</details>

### 26.Q7 — Flippers et L-learning : pourquoi « lousy » ?

<details><summary>Indice 1</summary>

Relis les §26.4, §26.5 et §26.5.1 de la fiche, et la figure de Flippers.

</details>
<details><summary>Indice 2</summary>

Un plateau gagnant n'a que trois points, sur une ligne ou une colonne entière. Le L-learning **remplace** la valeur d'une case par le TFR de la dernière partie : une seule partie suffit à effacer ce qu'on savait.

</details>
<details><summary>Indice 3</summary>

a) Trois points sur une ligne ou une colonne complète, les autres tuiles vides (B). Pour b), compte les lignes, puis les colonnes ; pour c), relis la règle de mise à jour ; pour d), relis ce que le livre montre après 3 000 puis 6 000 parties ; pour e), relis l'exemple où le camion allonge une partie.

</details>

### 26.Q8 — La mise à jour du Q-learning : α, γ et la valeur de l'état suivant

<details><summary>Indice 1</summary>

Relis les §26.6.1 et §26.6.3 de la fiche, et la figure de la mise à jour.

</details>
<details><summary>Indice 2</summary>

Écris la mise à jour sous la forme $Q \leftarrow (1 - \alpha)\,Q + \alpha\,y$ : $\alpha$ est le poids de la cible $y$, $1 - \alpha$ celui de l'ancienne valeur.

</details>
<details><summary>Indice 3</summary>

a) La cible regarde l'état **atteint** par l'action : la ligne de $s'$ (B). Pour b) et c), remplace $\alpha$ par 0, puis par 1, dans la forme de l'indice 2 ; pour d), relis le deuxième principe du Q-learning au §26.6 ; pour e), la cible vaut $0 + 1 \times 0{,}8$, puis $0{,}4 + 0{,}5 \times (\text{cible} - 0{,}4)$.

</details>

### 26.Q9 — ε-greedy ou softmax : vrai ou faux

<details><summary>Indice 1</summary>

Relis le §26.6.2 de la fiche et son encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

ε-greedy standard : avec la probabilité $\varepsilon$, une action tirée uniformément parmi **toutes** ; sinon, la gloutonne. La température d'une softmax joue le rôle inverse d'un zoom : grande, elle aplatit ; petite, elle accentue.

</details>
<details><summary>Indice 3</summary>

a) Vrai : c'est la définition standard, la meilleure peut être tirée en exploration. Pour b), $1 - \varepsilon + \varepsilon/5$ ; pour c), que devient $e^{Q/\tau}$ quand $\tau$ est très grand ? Pour d), la probabilité d'une action dépend-elle de sa valeur dans ε-greedy ? et dans softmax ? Pour e), calcule la somme d'une ligne de zéros.

</details>

### 26.Q10 — Table initialisée à zéro : comment la valeur remonte et converge

<details><summary>Indice 1</summary>

Relis les §26.6.4 et §26.6.5 de la fiche, et l'encadré ⚠️ sur la convergence.

</details>
<details><summary>Indice 2</summary>

Une cible faite de zéros (récompense nulle, ligne suivante nulle) ne change rien à une case nulle. La première valeur non nulle naît là où une récompense non nulle est reçue.

</details>
<details><summary>Indice 3</summary>

a) Tant que $r = 0$ et que la ligne de $s'$ est nulle, la cible vaut 0 et la case reste à 0 (A). Pour b), quelle mise à jour voit la récompense de la victoire ? Pour c), qu'est-ce qui fait remonter la valeur d'un état à celui qui le précède ? Pour d), relis l'encadré sur la convergence ; pour e), relis les résultats après 300, 3 000 et 6 000 parties.

</details>

### 26.Q11 — SARSA contre Q-learning : l'action suivante réellement jouée

<details><summary>Indice 1</summary>

Relis les §26.7 à §26.7.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Le Q-learning prend le **max** de la ligne de $s'$ ; SARSA prend la valeur de l'action $a'$ que la politique a choisie, et qu'elle jouera. Un algorithme est *on-policy* quand il apprend la valeur de la politique qu'il suit.

</details>
<details><summary>Indice 3</summary>

a) Les cinq lettres sont les cinq éléments d'une mise à jour : $s, a, r, s', a'$ (A). Pour b), relis la formule de SARSA ; pour c), lequel apprend la valeur d'une autre politique (la gloutonne) que celle qu'il suit ? Pour d), compare $Q(s', a')$ et $\max_{a'} Q(s', a')$ quand $a'$ est gloutonne ; pour e), relis l'encadré « un défaut du Q-learning ? ».

</details>

### 26.Q12 — Tables géantes, rejeu d'expérience, AlphaGo Zero et lancer de rayons

<details><summary>Indice 1</summary>

Relis les §26.8 à §26.10 de la fiche et l'encadré ⚠️ sur la taille des tables.

</details>
<details><summary>Indice 2</summary>

Une colonne par action : au morpion, une action est une case où jouer. La mémoire de rejeu de DQN garde des **transitions**, pas des parties. Pendant l'auto-apprentissage, chacun voit l'autre comme une partie du monde qui lui répond.

</details>
<details><summary>Indice 3</summary>

a) Un plateau 4 × 4 a 16 cases, donc 16 actions et 16 colonnes. Pour b), relis le §26.8 : qu'est-ce qui reçoit l'état et rend une valeur par action ? Pour c), relis l'encadré 🕰️ du §26.9 ; pour d), relis ce que veut dire « Zero » ; pour e), qui répond aux coups d'un joueur ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 26.R1 — Ch. 25 : un réseau comme fonction qui remplace une table (le décodeur d'un VAE)

<details><summary>Indice 1</summary>

Relis la fiche du ch. 25 (le décodeur) et le §26.8 de cette fiche.

</details>
<details><summary>Indice 2</summary>

Une table a une case par entrée possible ; un réseau calcule une sortie pour toute entrée, même jamais vue. Un DQN reçoit un état et rend une valeur Q par action.

</details>
<details><summary>Indice 3</summary>

a) 20 nombres **réels** peuvent prendre une infinité de valeurs : aucune table ne les liste (C). Pour b), repense à ce que fait un autoencodeur d'une entrée proche de ses données (25.Q11) ; pour c), une sortie par action.

</details>

### 26.R2 — Ch. 23 : que sauvegarder pour reprendre un entraînement ?

<details><summary>Indice 1</summary>

Relis la fiche du ch. 23 (les checkpoints) et le §26.3.6 de cette fiche.

</details>
<details><summary>Indice 2</summary>

Pour reprendre **exactement**, il faut tout ce qui change pendant l'entraînement : les poids, l'état de l'optimiseur, la position dans la boucle, et le hasard.

</details>
<details><summary>Indice 3</summary>

a) Le `state_dict` du modèle (A) est nécessaire ; applique le même critère (« change-t-il pendant l'entraînement ? ») à B, C, D et E. Pour b), qu'est-ce qui est **appris** dans un Q-learning tabulaire ? Pour c), qu'est-ce qui change au fil des épisodes, à part la table ?

</details>

### 26.R3 — Ch. 11 : bandit à bras multiples, ε-greedy et UCB

<details><summary>Indice 1</summary>

Relis la fiche du ch. 11 (le bandit) et l'encadré ⚠️ du §26.6.2 de cette fiche.

</details>
<details><summary>Indice 2</summary>

Un bandit n'a pas d'état qui change : on rejoue toujours la même situation, avec plusieurs bras. La mise à jour incrémentale : estimation + pas × (gain − estimation).

</details>
<details><summary>Indice 3</summary>

a) Un seul état (A). Pour b), $3 + 0{,}25 \times (5 - 3)$ ; pour c), relis la formule d'UCB : que devient le bonus quand un bras a été peu essayé ?

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 26.1 — Retours total et actualisé d'une partie de six coups ✏️

<details><summary>Indice 1</summary>

Relis le §26.3.7 de la fiche et son encadré 🧮.

</details>
<details><summary>Indice 2</summary>

Le retour du coup $t$ commence par **sa propre** récompense, de poids $\gamma^0 = 1$ ; la récompense du coup $t + k$ a le poids $\gamma^k$. La récurrence $G_t = r_t + \gamma\,G_{t+1}$, à partir du dernier coup, évite les puissances.

</details>
<details><summary>Indice 3</summary>

a) $1 + 0 + 0 + 2 + 0 + 4 = 7$. Pour b), somme les récompenses des coups 3 à 6 ; pour c) et d), $1 + 2\gamma^3 + 4\gamma^5$ ; pour e), le retour du coup 4 commence à sa récompense 2 : $2 + 0 \cdot \gamma + 4\gamma^2$ ; pour f), seul $\gamma^0$ ne s'annule pas ; pour g), que vaut $G_t - G_{t+1}$ avec $\gamma = 1$ ?

</details>

### Ex 26.2 — Taille des tables : de Flippers au Go ✏️

<details><summary>Indice 1</summary>

Relis les §26.4 et §26.8 de la fiche, et l'encadré ⚠️ sur la taille des tables.

</details>
<details><summary>Indice 2</summary>

$n$ cases à deux états donnent $2^n$ plateaux ; à trois états, $3^n$. Une table Q a une ligne par état et une colonne par action. Le nombre de chiffres d'un entier $N \ge 1$ vaut $\lfloor \log_{10} N \rfloor + 1$.

</details>
<details><summary>Indice 3</summary>

a) $2^9 = 512$. Pour b), multiplie par 9 actions ; pour c), $2^{16}$ plateaux et 16 actions ; pour d), compte les lignes et les colonnes d'un plateau 4 × 4 ; pour e), $3^9$ ; pour f), $3^{16}$ codages et 16 colonnes ; pour g), $361 \times 0{,}47712$, puis la partie entière plus 1.

</details>

### Ex 26.3 — Une mise à jour de Q-learning à la main ✏️

<details><summary>Indice 1</summary>

Relis les §26.6.1 et §26.6.3 de la fiche, et la figure de la mise à jour.

</details>
<details><summary>Indice 2</summary>

Trois étapes : la cible $y$ (le max de la ligne de $s'$, ou rien si $s'$ est terminal), l'erreur $\delta = y - Q(s, a)$, puis $Q(s, a) + \alpha\,\delta$.

</details>
<details><summary>Indice 3</summary>

a) Le max de la ligne de $s'$ vaut 0,6 : $y = 0{,}2 + 0{,}9 \times 0{,}6 = 0{,}74$. Pour b), soustrais 0,5 ; pour c), $0{,}5 + 0{,}5\,\delta$ ; pour d), la cible n'a plus de terme $\gamma \max$ : $y = r$ ; pour e), $\alpha = 1$ ; pour f), quelle case la formule modifie-t-elle ?

</details>

### Ex 26.4 — La même transition vue par SARSA ✏️

<details><summary>Indice 1</summary>

Relis le §26.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

SARSA remplace $\max_{a'} Q(s', a')$ par $Q(s', a')$ de l'action **choisie**. Les actions sont numérotées à partir de 0.

</details>
<details><summary>Indice 3</summary>

a) L'action 2 est la troisième valeur de la ligne, 0,3 : $y = 0{,}2 + 0{,}9 \times 0{,}3 = 0{,}47$. Pour b), $0{,}5 + 0{,}5\,(y - 0{,}5)$ ; pour c), reprends le c) de l'Ex 26.3 ; pour d), que vaut $Q(s', 1)$ par rapport au max ? Pour e), relis la définition de *on-policy*.

</details>

### Ex 26.5 — Probabilités d'action : ε-greedy (standard et version du livre) et softmax ✏️

<details><summary>Indice 1</summary>

Relis le §26.6.2 de la fiche, son encadré ⚠️ et la figure des politiques.

</details>
<details><summary>Indice 2</summary>

ε-greedy standard : chaque action reçoit $\varepsilon/|\mathcal{A}|$, la gloutonne reçoit en plus $1 - \varepsilon$. Version du livre : la gloutonne a $1 - \varepsilon$, les $|\mathcal{A}| - 1$ autres se partagent $\varepsilon$. Softmax : $p_i = e^{Q_i/\tau} / \sum_j e^{Q_j/\tau}$.

</details>
<details><summary>Indice 3</summary>

a) $1 - 0{,}2 + 0{,}2/4 = 0{,}85$. Pour b), $\varepsilon/4$ ; pour c) et d), la formule du livre de l'indice 2 ; pour e), $e^4 / (e^1 + e^2 + e^4 + e^0)$ ; pour f), divise d'abord les valeurs par 2 ; pour g), factorise $e^{10}$ au numérateur et au dénominateur ; pour h), $4 / (1 + 2 + 4 + 0)$.

</details>

### Ex 26.6 — Regarder la valeur remonter une chaîne d'états ✏️

<details><summary>Indice 1</summary>

Relis le §26.6.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Fais un tableau : une ligne par épisode, une colonne par état, et remplis-le **dans l'ordre des pas** (état 0, puis 1, puis 2). La cible de l'état 2 vaut 1 (l'état 3 est terminal) ; celle de l'état $s < 2$ vaut $0{,}9\,Q(s + 1)$, avec la valeur de $Q(s + 1)$ **au moment** de la mise à jour.

</details>
<details><summary>Indice 3</summary>

Épisode 1 avec $\alpha = 1$ : $Q(0) = 0{,}9 \times 0 = 0$, puis $Q(1) = 0{,}9 \times 0 = 0$, puis $Q(2) = 1$ : a) vaut 1. Continue le tableau pour les épisodes 2 et 3 (b, c, d). Avec $\alpha = 0{,}5$, chaque mise à jour fait $Q \leftarrow Q + 0{,}5\,(y - Q)$ : refais le tableau (e, f). Pour g), la limite est le point fixe, où $Q(2) = 1$ et $Q(s) = 0{,}9\,Q(s + 1)$ ; pour h), la valeur avance d'un état par épisode.

</details>

### Ex 26.7 — Somme géométrique et horizon effectif de γ ∂

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §26.3.7 et les Ex 0B.13 et 0B.14.

</details>
<details><summary>Indice 2</summary>

1. Dans $S_n - \gamma S_n$, presque tous les termes s'annulent. 4. $\gamma^k < \tfrac{1}{2}$ équivaut à $k \ln\gamma < -\ln 2$ ; attention, $\ln\gamma < 0$ change le sens de l'inégalité quand on divise.

</details>
<details><summary>Indice 3</summary>

1. $S_n - \gamma S_n = 1 - \gamma^n$ ; divise par $1 - \gamma$, et $\gamma^n \to 0$. 2. Factorise $r_{\max}$, puis compare terme à terme. 3. $1/(1 - \gamma)$ pour chaque valeur. 4. $k > \ln 2 / (-\ln\gamma)$ ; calcule-le pour 0,9 et 0,99 et prends l'entier suivant. 5. $0{,}2^4$ ; compare-le à $0{,}2^6$, en valeur relative et en valeur absolue.

</details>

### Ex 26.8 — Équation de Bellman : le point fixe que vise le Q-learning ∂

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §26.6.4 (l'équation de Bellman) et l'Ex 26.3.

</details>
<details><summary>Indice 2</summary>

1. Chaque $Q^*(s, a)$ vaut l'espérance de $r + \gamma \max_{a'} Q^*(s', a')$, avec $\max = 0$ pour l'état terminal. 2. Avec la politique supposée, $V(A) = Q^*(A, \text{aller})$ et $V(B) = Q^*(B, \text{jouer})$ : deux équations, deux inconnues. 3. Écris $|\mathcal{T}Q_1 - \mathcal{T}Q_2|$ pour une case : les récompenses s'annulent.

</details>
<details><summary>Indice 3</summary>

1. Par exemple, $Q^*(B, \text{jouer}) = 2 + 0{,}9 \times \big(\tfrac{1}{2} \times 0 + \tfrac{1}{2} \max_{a'} Q^*(A, a')\big)$ ; écris les trois autres de la même façon. 2. $V(A) = 0{,}9\,V(B)$ et $V(B) = 2 + 0{,}45\,V(A)$ : substitue. 3. Pour une case, $|\mathcal{T}Q_1 - \mathcal{T}Q_2| \le \gamma\,\mathbb{E}\big[|\max Q_1(s', \cdot) - \max Q_2(s', \cdot)|\big]$, puis l'inégalité donnée ; une contraction a un seul point fixe (pense à la distance qui est multipliée par $\gamma$ à chaque application). 4. La cible tire $s'$ selon les vraies probabilités ; que fait $\alpha \approx 1$ de la cible précédente ?

</details>

### Ex 26.9 — Mémoire d'une table Q : morpion 4×4, 5×5 et Go 🧮

<details><summary>Indice 1</summary>

Relis le §26.8 de la fiche et reprends l'Ex 26.2.

</details>
<details><summary>Indice 2</summary>

Mémoire = nombre de lignes × nombre de colonnes × 4 octets. Garde les puissances de 10 : $3^{9} \approx 2 \times 10^4$, $3^{16} \approx 4{,}3 \times 10^7$, $3^{25} \approx 8{,}5 \times 10^{11}$.

</details>
<details><summary>Indice 3</summary>

1. $19\,683 \times 9 \times 4 \approx 7 \times 10^5$ octets. 2. Même calcul avec $3^{16}$ et 16 colonnes, puis $3^{25}$ et 25 colonnes ; divise par $16 \times 10^9$ et par $10^{18}$. 3. Le rapport $3^9 / 5\,478$. 4. Le nombre de cases divisé par $10^9$ donne des secondes. 5. Une couche dense de $n_{in}$ vers $n_{out}$ a $(n_{in} + 1)\,n_{out}$ paramètres.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 26.10 — Attribution du crédit et exploration expliquées en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis les §26.3.5 et §26.6.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pour (1), une image où l'on ne sait qu'à la fin si tout s'est bien passé (une recette, un trajet) ; puis l'idée que chaque étape apprend de la note de l'étape suivante. Pour (2), une image où l'on retourne toujours au même endroit (un restaurant, un chemin).

</details>
<details><summary>Indice 3</summary>

Commence par « Quand on gagne une partie, on ne sait pas quels coups ont compté », puis : le Q-learning note chaque coup d'après la note du coup suivant, si bien que la valeur de la victoire remonte peu à peu ; et sans jamais essayer un autre restaurant, on ne saura pas s'il en existe un meilleur.

</details>

### Ex 26.11 — Quand l'agent optimise la mauvaise récompense ⚖️

<details><summary>Indice 1</summary>

Relis l'encadré 💼 du §26.2, le §26.3.7 et la partie « Au-delà du livre » de la fiche.

</details>
<details><summary>Indice 2</summary>

Pour chaque cas, sépare ce qu'on **voulait**, ce qu'on a **récompensé** et la façon la moins chère de maximiser la récompense. Les garde-fous : récompenser le but plutôt qu'un indicateur, surveiller d'autres mesures, limiter l'écart à un comportement de référence.

</details>
<details><summary>Indice 3</summary>

1. Les bonus rapportent plus que l'arrivée, et tourner en rond en ramasse sans fin : la récompense ne parle pas de finir. Une récompense d'arrivée (ou une pénalité par seconde) l'évite, mais elle arrive tard : l'attribution du crédit devient plus dure. Raisonne de même pour 2 (le temps passé n'est pas la satisfaction) et 3 (le modèle de récompense n'est qu'une approximation des préférences). Pour 4, donne un argument par acteur, puis tranche.

</details>

### Ex 26.12 — Lire Mnih et coll. (2015), « Human-level control through deep reinforcement learning » 📄

<details><summary>Indice 1</summary>

Lis d'abord le résumé, la figure 1 et le paragraphe *Methods* sur l'algorithme d'entraînement ; garde la fiche (§26.6, §26.8, §26.9) ouverte à côté.

</details>
<details><summary>Indice 2</summary>

Cherche dans l'article : *preprocessing* (l'entrée), *model architecture* (la sortie), *target network* (la cible), *replay memory* (la mémoire), *ε annealed* (l'exploration), *clip* (les récompenses). Pour 6, regarde la figure 3 : les jeux en bas de la liste.

</details>
<details><summary>Indice 3</summary>

1. L'entrée est une pile des dernières images prétraitées ; une sortie par action, pour obtenir toutes les valeurs Q d'un état en un seul passage. 2. La cible est celle du Q-learning, avec les poids figés du réseau cible ; elle évite de poursuivre une cible qui bouge à chaque mise à jour. Continue de la même façon pour 3 (la taille et les deux raisons), 4 (de quelle valeur à quelle valeur, sur combien d'images) et 5 (des échelles de score très différentes d'un jeu à l'autre). Pour 6, pense aux jeux où la première récompense demande une longue suite d'actions précises.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 26.E1 — Q-learning ou SARSA : on-policy, off-policy, lequel choisir ?

<details><summary>Indice 1</summary>

Relis les §26.7 à §26.7.2 de la fiche et l'encadré ⚠️ « un défaut du Q-learning ? ».

</details>
<details><summary>Indice 2</summary>

Structure : la seule différence de formule, ce qu'apprend chacun (on-policy, off-policy), un exemple où cela compte (la falaise), puis le critère de choix.

</details>
<details><summary>Indice 3</summary>

Commence par « les deux mettent à jour Q(s, a) vers r + γ Q(s', ·), seule la valeur de s' diffère », puis : le Q-learning apprend la politique gloutonne quelle que soit l'exploration, SARSA la politique qu'il suit vraiment ; au bord d'une falaise, SARSA apprend un chemin prudent ; on choisit selon que les erreurs pendant l'apprentissage coûtent ou non.

</details>

### 26.E2 — À quoi sert le facteur d'actualisation γ ?

<details><summary>Indice 1</summary>

Relis le §26.3.7 de la fiche, son encadré 🧮 et l'encadré ⚠️ sur γ.

</details>
<details><summary>Indice 2</summary>

Structure : la définition, les deux rôles (préférence pour le proche, somme finie), l'horizon effectif, puis comment le choisir.

</details>
<details><summary>Indice 3</summary>

Commence par « γ pèse une récompense reçue dans k pas par γ^k », puis : il rend la somme finie pour une tâche sans fin, il fixe un horizon d'environ 1/(1 − γ) pas ; on le choisit d'après l'horizon de la tâche (souvent 0,99), et c'est une partie de l'objectif, pas un réglage de la vitesse d'apprentissage.

</details>

### 26.E3 — Comment gérer le compromis exploration/exploitation ?

<details><summary>Indice 1</summary>

Relis les §26.3.5 et §26.6.2 de la fiche, et le ch. 11 (UCB).

</details>
<details><summary>Indice 2</summary>

Structure : le dilemme, puis trois ou quatre méthodes, de la plus simple à la plus fine, avec ce qui les distingue.

</details>
<details><summary>Indice 3</summary>

Commence par « exploiter rapporte maintenant, explorer rapporte de l'information », puis : ε-greedy (souvent avec un ε qui décroît), softmax (explore surtout les actions proches de la meilleure), UCB (un bonus aux actions peu essayées), l'initialisation optimiste ; et, pour les problèmes difficiles, des bonus de curiosité.

</details>

### 26.E4 — Pourquoi DQN utilise-t-il un replay buffer et un réseau cible ?

<details><summary>Indice 1</summary>

Relis les §26.8 et §26.9 de la fiche et l'encadré 🕰️ sur la mémoire de rejeu.

</details>
<details><summary>Indice 2</summary>

Structure : ce qui rend le Q-learning avec un réseau instable (deux causes), puis le remède à chacune.

</details>
<details><summary>Indice 3</summary>

Commence par « la descente de gradient suppose des exemples à peu près indépendants, et une cible fixe », puis : des transitions consécutives sont très corrélées (la mémoire de rejeu les mélange et les réutilise), et la cible dépend du réseau qu'on entraîne (le réseau cible la fige pendant quelques milliers de pas).

</details>

### 26.E5 — Qu'est-ce que le RLHF et pourquoi utilise-t-on le RL avec les LLM ?

<details><summary>Indice 1</summary>

Relis la partie « Au-delà du livre » de la fiche.

</details>
<details><summary>Indice 2</summary>

Structure : pourquoi le pré-entraînement ne suffit pas, les trois étapes du RLHF, pourquoi du RL (une récompense sur une réponse entière, non dérivable), puis les alternatives récentes.

</details>
<details><summary>Indice 3</summary>

Commence par « un modèle pré-entraîné prolonge du texte, il ne cherche pas à être utile », puis : ajustement supervisé, modèle de récompense appris sur des classements humains, RL (PPO) avec une pénalité KL ; une note sur une réponse entière ne se dérive pas mot à mot, d'où le RL ; puis DPO et les récompenses vérifiables.

</details>

<a id="notebook"></a>

## Exercices du notebook

### Ex 26.13 — Prendre en main gymnasium avec FrozenLake 📦

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ sur Gymnasium (fiche §26.3) : `reset` rend `(state, info)`, `step` rend cinq valeurs.

</details>
<details><summary>Indice 2</summary>

Les tailles se lisent sur `env.observation_space.n` et `env.action_space.n`. Une boucle d'épisode : `reset`, puis `step` tant que ni `terminated` ni `truncated` ne valent `True` ; additionne les récompenses en route. Pour d), une liste de 1 000 réussites (`total > 0`), puis sa moyenne.

</details>
<details><summary>Indice 3</summary>

```python
def play_actions_13(env, actions, seed=None):
    env.reset(seed=seed)
    total, steps, terminated = 0.0, 0, False
    for action in actions:
        _, reward, terminated, truncated, _ = env.step(int(action))
        # add the reward, count the step, stop if the episode is over
    return total, steps, terminated
```
`random_success_13` a la même boucle, avec `int(rng.integers(n_actions))` comme action et un `while not (terminated or truncated)`.

</details>

### Ex 26.14 — Retours actualisés et retours à venir 🔨

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §26.3.7 et ton ✏️ 26.1 (le calcul à rebours).

</details>
<details><summary>Indice 2</summary>

`discounted_return` : les poids $\gamma^k$ forment un tableau `gamma ** np.arange(T)`. `returns_to_go` : on parcourt l'épisode à l'envers en gardant $G$, avec $G_t = r_t + \gamma\,G_{t+1}$. Valide d'abord : $\gamma$ dans $[0, 1]$, des récompenses 1-D sans NaN.

</details>
<details><summary>Indice 3</summary>

```python
def returns_to_go(rewards, gamma=1.0):
    r = _validated_rewards(rewards)        # your checks: 1-D float64, no NaN; gamma in [0, 1]
    out = np.empty_like(r)
    G = 0.0
    for t in range(len(r) - 1, -1, -1):
        ...   # the return from step t, from the one of step t + 1 (✏️ 26.1), stored at position t
    return out
```
`discounted_return` : les récompenses pondérées par les poids de l'indice 2, sommées, rendues en `float` Python (relis le contrat pour l'épisode vide).

</details>

### Ex 26.15 — Reproduire les courbes d'actualisation (fig. 26.13-26.14) 🎨

<details><summary>Indice 1</summary>

Relis la figure `poids_gamma.png` de la fiche : la même idée, appliquée aux récompenses d'une partie.

</details>
<details><summary>Indice 2</summary>

`k = np.arange(len(rewards))` ; pour chaque $\gamma$, `gamma ** k * rewards` est une courbe du panneau de gauche. `plt.subplots(1, 2)` donne les deux panneaux ; le panneau de droite trace les retours selon `gammas`.

</details>
<details><summary>Indice 3</summary>

```python
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
for g in gammas:
    left.plot(k + 1, g ** k * rewards, "o-", label=f"γ = {g}")
returns = np.array([mylearn.rl.discounted_return(rewards, g) for g in gammas])
# right.plot(gammas, returns, "o-"), labels, legend, plt.show(), return returns
```

</details>

### Ex 26.16 — Prédire l'effet du camion sur la table L 🔮

<details><summary>Indice 1</summary>

Relis les §26.5 et §26.5.1 de la fiche : comment le L-learning choisit-il ses coups, et quand une case change-t-elle ?

</details>
<details><summary>Indice 2</summary>

Pour a), demande-toi qui essaie les autres coups : le L-learning n'explore qu'au hasard des ex æquo, et une case positive bat toujours une case nulle. Pour b), pense à ce que vaut une case après une partie coupée à 100 coups, sans victoire. Pour c), compte les cases qu'une partie touche, et combien de parties passent par les mêmes plateaux.

</details>
<details><summary>Indice 3</summary>

Pour a), demande-toi quand l'agent glouton essaie un coup dont la case vaut 0, alors qu'une autre case du même plateau est positive. Pour b), demande-toi ce que l'agent fait ensuite depuis des plateaux dont une partie coupée sans victoire a remis les cases à 0. Pour c), compare le nombre de cases de la table (4 608) au nombre de cases qu'une partie peut toucher, et demande-toi si les parties passent par des plateaux différents.

</details>

### Ex 26.17 — Flippers : la longueur optimale de chaque partie par recherche exhaustive 🔬

<details><summary>Indice 1</summary>

Une recherche en largeur traite les plateaux par « couches » : ceux à 0 coup, puis ceux à 1 coup, etc.

</details>
<details><summary>Indice 2</summary>

Un tableau `dist` de 512 valeurs à −1 (« pas encore atteint ») ; la couche 0 : les 6 plateaux de `Flippers().winning`. Pour chaque plateau de la couche courante, ses 9 voisins s'obtiennent par `board ^ (1 << tile)` ; un voisin encore à −1 reçoit la distance du plateau plus 1 et va dans la couche suivante. Ensuite : `np.bincount`, une moyenne sur `dist > 0`, `argmax`.

</details>
<details><summary>Indice 3</summary>

```python
dist = np.full(512, -1, dtype=int)
frontier = sorted(Flippers().winning)
dist[frontier] = 0
while frontier:
    nxt = []
    for board in frontier:
        for tile in range(9):
            neighbour = board ^ (1 << tile)
            # if not reached yet: its distance, and it joins the next layer
    # the next layer becomes the frontier
```

</details>

### Ex 26.18 — L-learning sur Flippers, sans puis avec camion 🔬

<details><summary>Indice 1</summary>

Relis le §26.5 de la fiche : jouer toute la partie, puis écrire les TFR.

</details>
<details><summary>Indice 2</summary>

Pendant la partie, garde deux listes : les couples `(state, action)` joués et les récompenses. À la fin, `mylearn.rl.returns_to_go(rewards)` donne le TFR de chaque coup ; une boucle dans l'ordre des coups écrit chaque TFR dans sa case (le dernier passage gagne). Le tirage de la graine de `reset` et `tie_argmax(L[state], rng)` utilisent le même `rng`.

</details>
<details><summary>Indice 3</summary>

```python
L = np.zeros((env.observation_space.n, env.action_space.n))
for _ in range(n_games):
    state, _ = env.reset(seed=int(rng.integers(2**31)))
    moves, rewards = [], []
    # play: action = tie_argmax(L[state], rng), step, append, stop at the end or after max_steps moves
    # then: the TFR of each move (returns_to_go), written into its cell, in the order of the moves
```

</details>

### Ex 26.19 — Politiques gloutonne et softmax (ε-greedy importée du ch. 11) 🔨

<details><summary>Indice 1</summary>

Relis le §26.6.2 de la fiche et ton ✏️ 26.5 e) et f).

</details>
<details><summary>Indice 2</summary>

`greedy_policy` : `np.argmax(Q, axis=1)` sur un tableau validé (2-D, sans NaN). `softmax_action` : valide la température et les valeurs, calcule `np.exp((q - q.max()) / T)`, normalise, puis un seul `rng.choice`.

</details>
<details><summary>Indice 3</summary>

```python
q = np.asarray(q_values, dtype=float)      # 1-D, not empty, no NaN; T > 0 and not NaN
probs = np.exp((q - q.max()) / temperature)
# normalise, create rng if it is None, then ONE draw with rng.choice, returned as a Python int
```

</details>

### Ex 26.20 — Mises à jour de Q-learning et de SARSA, états terminaux compris 🔨

<details><summary>Indice 1</summary>

Relis le §26.6.1 de la fiche, son encadré 🧮 sur les états terminaux, et tes ✏️ 26.3 et 26.4.

</details>
<details><summary>Indice 2</summary>

Valide tout d'abord (table de flottants, indices entiers dans les bornes, $\alpha$, $\gamma$, récompense, masque). Puis : la cible (`reward` si `terminated`, sinon `reward + gamma *` la valeur de $s'$), $\delta$, la mise à jour. Pour le masque : `mask = np.asarray(next_mask).astype(bool)`, puis le max de `Q[next_state][mask]`.

</details>
<details><summary>Indice 3</summary>

```python
if terminated:
    target = reward
else:
    row = Q[next_state] if next_mask is None else Q[next_state][mask]
    # target = ...
# delta: computed BEFORE the update; then the update of Q[state, action] (incremental_update works)
```

</details>

### Ex 26.21 — La boucle d'entraînement tabulaire et l'évaluation d'une politique 🔨

<details><summary>Indice 1</summary>

Relis le §26.6.3 de la fiche (tout assembler) et ta fonction `play_actions_13` : la boucle d'épisode est la même.

</details>
<details><summary>Indice 2</summary>

Deux boucles : les épisodes, puis les pas. Garde l'`info` de l'état courant (son masque sert à choisir l'action) et celle de l'état suivant (son masque sert au max). Une petite fonction `choose(state, info)` : sans masque, `epsilon_greedy_action(Q[state], eps, rng)` ; avec masque, la même sur `Q[state, legal]`, puis `legal[...]`. Pour SARSA, choisis `next_action` avant la mise à jour, puis joue-la au pas suivant. `max_steps` : la boucle s'arrête, et la dernière mise à jour garde l'amorçage.

</details>
<details><summary>Indice 3</summary>

```python
for episode in range(n_episodes):
    eps = epsilon(episode) if callable(epsilon) else epsilon
    state, info = env.reset(seed=int(rng.integers(2**31)))
    action = choose(state, info, eps)
    for step in range(max_steps):
        next_state, reward, terminated, truncated, next_info = env.step(action)
        # Q-learning: q_learning_update(..., terminated, ..., next_mask=mask of next_info)
        # SARSA: next_action = choose(next_state, next_info, eps) unless TERMINATED (a truncation still bootstraps), then sarsa_update
        # count the step and the reward, stop if terminated or truncated, else move on
```
`evaluate_policy` : la même boucle, sans exploration ni mise à jour, avec `policy[state]` ou `policy(state, info)`.

</details>

### Ex 26.22 — Q-learning sur FrozenLake, glissant ou non 📦

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §26.6.2 de la fiche : un $\varepsilon$ qui décroît.

</details>
<details><summary>Indice 2</summary>

Le calendrier est une fonction définie **dans** `epsilon_schedule_22`, qui garde `start`, `end`, `fraction` et `n_episodes` ; le `max` avec `end` la rend constante après la baisse. `train_lake_22` n'a que deux lignes utiles : créer le lac, appeler `train_tabular`.

</details>
<details><summary>Indice 3</summary>

```python
def epsilon_schedule_22(n_episodes, start=1.0, end=0.05, fraction=0.8):
    def epsilon(i):
        ...                                # start - (start - end) * i / (fraction * n_episodes), never below end
    return epsilon
```
`train_lake_22` crée le lac avec `make_env`, puis rend `(env, *mylearn.rl.train_tabular(...))`.

</details>

### Ex 26.23 — Lire des courbes d'apprentissage RL bruitées 📈

<details><summary>Indice 1</summary>

Relis le §26.6.2 de la fiche : que fait un $\varepsilon$ élevé à la réussite **pendant** l'entraînement ?

</details>
<details><summary>Indice 2</summary>

Une courbe mesure la politique qui explore, la barre la politique gloutonne. Une exploration qui décroît fait monter la courbe vers la barre ; une exploration constante et forte laisse la courbe basse même quand la politique gloutonne est bonne ; un $\alpha = 1$ sur un lac glissant n'apprend pas à faire la moyenne.

</details>
<details><summary>Indice 3</summary>

a) **C** : c'est la seule courbe qui monte nettement, comme une exploration qui décroît le fait attendre. Pour b), cherche une courbe basse dont la barre est haute. Pour c), compte les barres au-dessus de 0,5. Pour d), compare, pour chaque lettre, la fin de la courbe à la hauteur de la barre.

</details>

### Ex 26.24 — L'agent qui n'apprenait rien 🐛

<details><summary>Indice 1</summary>

Lis les diagnostics dans l'ordre : chacun vise un défaut précis de la fiche (§26.6.1, encadré 🧮, et §26.6.4).

</details>
<details><summary>Indice 2</summary>

Pour chaque ligne, demande-toi ce qu'elle doit faire : départager des ex æquo (que fait `np.argmax` de `[0, 0, 0]` ?), lire la ligne de quel état pour la cible, garder ou non l'amorçage après `terminated`, après `truncated`. Corrige un défaut à la fois, et relance `diagnose_24` sur ta version.

</details>
<details><summary>Indice 3</summary>

Pour le premier point : la ligne (E) rend toujours 0 pour une ligne de zéros ; la version corrigée choisit `action = tie_argmax(Q[state], rng)`. Pour les trois autres, demande-toi quelle ligne lit la ligne de l'état **courant** au lieu de celle de l'état suivant, quelle ligne garde l'amorçage après une vraie fin, et quelle ligne le coupe après une simple coupure. La cible corrigée a la forme :
```python
target = reward if terminated else reward + gamma * ...   # the row of which state?
```

</details>

### Ex 26.25 — Q-learning sur Flippers avec camion : 300, 3 000 puis 6 000 parties 🔬

<details><summary>Indice 1</summary>

Relis la docstring de `train_tabular` : `Q_init` permet de reprendre un entraînement.

</details>
<details><summary>Indice 2</summary>

Garde la table courante et le nombre de parties déjà jouées ; à chaque étape, joue la différence, à partir de la table courante, avec `np.random.default_rng(seed + k)` ; garde une **copie** de la table dans le dictionnaire.

</details>
<details><summary>Indice 3</summary>

```python
env = Flippers(truck_prob=truck_prob)
tables, Q, done = {}, None, 0
for k, n in enumerate(stages):
    # train n - done more games from Q (Q_init), with np.random.default_rng(seed + k) and **kwargs
    # remember n, and a copy of Q
```

</details>

### Ex 26.26 — Itération sur les valeurs : la référence quand le modèle est connu 🔨

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 du §26.6.4 (l'équation de Bellman) et ton ∂ 26.8.

</details>
<details><summary>Indice 2</summary>

Valide `P` (états `0 .. n - 1`, mêmes actions partout, probabilités de somme 1, états suivants dans les bornes) et $\gamma \in [0, 1[$. Une fonction `backup(V)` calcule toute la table `Q` à partir d'un `V` : pour chaque issue `(p, s2, r, done)`, `p * (r + gamma * V[s2] * (1 - done))`. On répète `V_new = backup(V).max(axis=1)` jusqu'à ce que la plus grande variation soit sous `theta`.

</details>
<details><summary>Indice 3</summary>

```python
V = np.zeros(n_states)
for _ in range(max_iter):
    V_new = backup(V).max(axis=1)          # synchronous: computed from the old V only
    # stop when the largest change is strictly below theta
# Q: one more backup of the final V
```

</details>

### Ex 26.27 — α, γ, ε : balayage d'hyperparamètres sur Flippers bruité 🔬

<details><summary>Indice 1</summary>

Relis l'encadré ⚠️ sur $\alpha$ du §26.6.1 et ton 26.25 : la même boucle, en faisant varier les réglages.

</details>
<details><summary>Indice 2</summary>

Deux boucles : les réglages, puis les graines. Pour chaque couple, un entraînement neuf (pas de `Q_init`), puis une évaluation avec le **même** camion et `seed=seed`. La clé du dictionnaire est le tuple du réglage.

</details>
<details><summary>Indice 3</summary>

```python
for alpha, gamma, truck in settings:
    lengths = []
    for seed in seeds:
        # a NEW training on Flippers(truck_prob=truck, reward="win"), then
        # the mean length of evaluate_flippers(Q, truck_prob=truck, seed=seed)
```

</details>

### Ex 26.28 — Prédire le chemin appris par SARSA et par Q-learning au bord de la falaise 🔮

<details><summary>Indice 1</summary>

Relis l'encadré ⚠️ du §26.7 de la fiche : ce que chacun apprend quand l'exploration ne s'arrête pas.

</details>
<details><summary>Indice 2</summary>

Avec $\varepsilon = 0{,}1$ et 4 actions, l'agent joue une action au hasard une fois sur dix environ. Pour a), demande-toi à quelle distance du bord ce hasard ne fait plus tomber, sachant qu'il peut frapper plusieurs fois de suite. Pour b) et c), un épisode du Q-learning longe le bord sur 11 pas : estime ses chances de tomber.

</details>
<details><summary>Indice 3</summary>

Le long du bord, chaque pas d'exploration tire « bas » une fois sur quatre : estime la probabilité de tomber en 11 pas, puis multiplie par 500 épisodes, sans oublier que les premiers épisodes, avant que le chemin soit appris, tombent aussi. Pour SARSA, compare ce que vaut, sous ε-greedy, une case à une rangée du bord et une case à deux rangées : de combien d'explorations malheureuses de suite faut-il pour tomber de chacune ?

</details>

### Ex 26.29 — Q-learning contre SARSA sur Flippers et CliffWalking 🔬

<details><summary>Indice 1</summary>

Relis les §26.7.1 et §26.7.2 de la fiche et ton 26.28.

</details>
<details><summary>Indice 2</summary>

Deux boucles, les algorithmes puis les graines ; un environnement neuf (`make()`) à chaque entraînement. La mesure « online » est la moyenne des 100 derniers retours de l'historique ; « greedy » est `score(env, Q)`.

</details>
<details><summary>Indice 3</summary>

```python
for algorithm in ("q_learning", "sarsa"):
    online, greedy = [], []
    for seed in seeds:
        env = make()
        # Q, history = train_tabular(env, ..., algorithm=algorithm, rng=np.random.default_rng(seed), **train_kwargs)
        # the two measures
```

</details>

### Ex 26.30 — Reproduire les comparaisons au fil de l'entraînement (fig. 26.58-26.60) 🎨

<details><summary>Indice 1</summary>

Le livre trace trois mesures en fonction du nombre de parties ; `train_in_stages_25` donne déjà les tables à chaque étape.

</details>
<details><summary>Indice 2</summary>

`measure_30` : une boucle sur `tables.items()`, quatre listes remplies en parallèle. `plot_comparison_30` : `plt.subplots(1, 3)`, puis, pour chaque mesure et chaque courbe, `ax.plot(curve["games"], curve[key], "o-", label=...)` et `ax.set_xscale("log")`.

</details>
<details><summary>Indice 3</summary>

```python
for n, Q in tables.items():
    curves["games"].append(n)
    curves["length"].append(float(evaluate_flippers(Q, truck_prob=truck_prob, seed=30)["length"].mean()))
    # the mean of ALL the cells, and the share of nonzero cells
```

</details>

### Ex 26.31 — Une mémoire de rejeu (replay buffer) 🔨

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ du §26.9 de la fiche, puis la docstring de `__init__` : le stockage est déjà prêt.

</details>
<details><summary>Indice 2</summary>

Une file circulaire : `push` écrit à `self._pos`, puis `self._pos = (self._pos + 1) % self.capacity` et `self._size = min(self._size + 1, self.capacity)`. `sample` tire des indices distincts parmi `range(self._size)` avec `self._rng.choice(..., replace=False)`, puis indexe chaque tableau (l'indexation par un tableau d'indices rend une copie).

</details>
<details><summary>Indice 3</summary>

```python
def sample(self, batch_size):
    # checks: an int (not a bool), 1 <= batch_size <= len(self)
    idx = self._rng.choice(self._size, size=batch_size, replace=False)
    return {"states": self._states[idx], ...}      # the five fields, indexed by idx (a copy)
```

</details>

### Ex 26.32 — Concevoir la récompense : finale, pénalité par coup ou façonnée 🔬

<details><summary>Indice 1</summary>

Relis l'encadré 💼 du §26.2 et le piratage de la récompense (« Au-delà du livre »).

</details>
<details><summary>Indice 2</summary>

Le wrapper doit se souvenir du nombre de points du plateau d'avant : compte-le dans `reset`, puis à chaque `step`, compare au nouveau compte (`bin(state).count("1")`) et mets la mémoire à jour. `train_eval_32` : un entraînement, puis une évaluation sur le jeu ordinaire.

</details>
<details><summary>Indice 3</summary>

```python
def step(self, action):
    state, reward, terminated, truncated, info = self.env.step(action)
    dots = bin(int(state)).count("1")
    # add self.bonus if dots > self._dots, then remember dots
    return state, reward, terminated, truncated, info
```

</details>

### Ex 26.33 — Rejouer l'expérience pour apprendre avec moins d'épisodes 🔬

<details><summary>Indice 1</summary>

Relis le §26.9 de la fiche et ta `train_tabular` : la même boucle, avec une mémoire en plus.

</details>
<details><summary>Indice 2</summary>

Pour retrouver exactement `train_tabular` sans rejeu, les tirages de `rng` doivent venir dans le même ordre : la graine de `reset` d'abord, puis une action par pas avec `epsilon_greedy_action`. La mémoire a son propre générateur (`random_state=seed`) : ses tirages ne touchent pas `rng`.

</details>
<details><summary>Indice 3</summary>

```python
R.q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma)
buffer.push(state, action, reward, next_state, terminated)
if replay_updates > 0 and len(buffer) >= replay_updates:
    batch = buffer.sample(replay_updates)
    # one q_learning_update per row i of the batch (int(...), float(...), bool(...) of each field)
```

</details>

### Ex 26.34 — Rapporter un résultat RL honnête : plusieurs graines, configuration, intervalle 🛠️

<details><summary>Indice 1</summary>

Relis « Rapporter un résultat de RL » dans la fiche, et le bootstrap du ch. 2.

</details>
<details><summary>Indice 2</summary>

`run_34` lit tout dans `config` : rien ne doit être écrit en dur. L'évaluation tire ses graines d'un générateur **neuf**, `np.random.default_rng(config["eval_seed"])`, à chaque appel. `report_34` : une liste de résultats, `bootstrap_ci`, puis `json.dumps(report, indent=2)` dans le fichier.

</details>
<details><summary>Indice 3</summary>

```python
results = [run_34(config, seed) for seed in seeds]
low, high = mylearn.stats.bootstrap_ci(results, confidence=0.95, n_boot=2000, rng=np.random.default_rng(0))
# the dict of the five keys (Python floats and lists), written as JSON, then returned
```

</details>

### Ex 26.35 — Morpion contre un adversaire aléatoire : masque d'actions ou pénalité 📦

<details><summary>Indice 1</summary>

Relis le §26.2.1 de la fiche (les coups interdits) et la docstring de `TicTacToe`.

</details>
<details><summary>Indice 2</summary>

`masked_greedy(Q)` rend une fonction `(state, info) -> action` qui ne choisit que des coups permis ; sans masque, `np.argmax(Q[state])`. La partie finit sur `terminated` ; `info["result"]` dit comment.

</details>
<details><summary>Indice 3</summary>

```python
counts = {"win": 0, "draw": 0, "loss": 0, "illegal": 0}
for _ in range(n_games):
    state, info = env.reset(seed=int(rng.integers(2**31)))
    # play until terminated, with with_mask(state, info) if "action_mask" in info else int(np.argmax(Q[state]))
    counts[info["result"]] += 1
```

</details>

### Ex 26.36 — CartPole : discrétiser un état continu pour un Q-learning tabulaire 📦

<details><summary>Indice 1</summary>

Relis le §26.8 de la fiche et 26.R1 : une table a besoin d'états qu'on peut compter.

</details>
<details><summary>Indice 2</summary>

`np.digitize(valeur, bornes)` rend l'indice de l'intervalle (0 avant la première borne, `len(bornes)` après la dernière ; une liste vide donne toujours 0). `np.ravel_multi_index(indices, dims)` numérote les cases d'une grille, comme les indices d'un tableau à plusieurs dimensions.

</details>
<details><summary>Indice 3</summary>

```python
# __init__: keep the edges, dims = tuple(len(e) + 1 for e in edges),
#           self.observation_space = gym.spaces.Discrete(int(np.prod(dims)))
def observation(self, observation):
    indices = [int(np.digitize(value, e)) for value, e in zip(observation, self.edges)]
    # the number of the cell of these indices in the grid of shape self.dims, as a Python int
```

</details>

### Ex 26.37 — Auto-apprentissage au morpion : l'agent joue les deux camps 🔬

<details><summary>Indice 1</summary>

Relis le §26.10 de la fiche, puis ton ∂ 26.8 : une cible qui regarde la position suivante, vue par l'adversaire.

</details>
<details><summary>Indice 2</summary>

Avec la vue canonique, le joueur au trait a toujours les marques 1 : poser sa marque en `a`, c'est `after = c + 3**a`. La position suivante, vue par l'adversaire, est `TABLES_37["swap"][after]`. Ce que l'adversaire gagne avec sa meilleure réponse, tu le perds : d'où le signe moins. Les coups interdits ne sont jamais choisis ni mis à jour.

</details>
<details><summary>Indice 3</summary>

```python
after = c + 3**a
c2 = int(swap[after])
if winner[after] == 1:
    target, done = 1.0, True
# a full board: 0; otherwise minus gamma times the best LEGAL value of Q[c2]
```

</details>

### Ex 26.38 — Défi CartPole tabulaire : tenir 475 pas en moyenne sur 500 🏆

<details><summary>Indice 1</summary>

Mesure d'abord ta politique de 26.36 sur des épisodes de 500 pas : où en est-elle ? Puis demande-toi ce qui limite une table : sa taille, ou le nombre de fois qu'on visite chaque case ?

</details>
<details><summary>Indice 2</summary>

Chaque case de la table doit être visitée souvent pour que sa valeur soit juste : avec 144 cases et quelques centaines d'épisodes, beaucoup restent mal estimées. Les leviers : **moins** de cases (ne garder que les variables qui décident de la chute, avec peu d'intervalles), un $arepsilon$ qui finit très bas, et des épisodes d'entraînement coupés à 200 pas (moins chers, et une politique qui tient 200 pas en tient souvent 500).

</details>
<details><summary>Indice 3</summary>

```python
def make_cartpole_38():
    edges = [[], [], [...], [...]]         # ignore the cart; a few bounds for the angle and its speed only
    return Discretize_36(gym.make("CartPole-v1"), edges)
```
Garde l'entraînement de 26.36, change un levier à la fois, et mesure ensuite sur plusieurs graines d'entraînement : une recette qui ne réussit qu'avec une graine est fragile.

</details>
