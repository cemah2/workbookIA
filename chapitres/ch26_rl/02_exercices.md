# 26 · Apprentissage par renforcement — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch26_rl/06_mes_reponses.md` (créée par `python tools/start_chapter.py 26`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ 26.1 à 26.6 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les dérivations ∂ 26.7 et 26.8, l'estimation 🧮 26.9, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"F"`), plusieurs lettres entre guillemets pour un choix multiple (`"AF"`).
> Conventions du chapitre (fiche) : état $s$, action $a$, récompense $r$, état suivant $s'$, facteur d'actualisation $\gamma$, learning rate $\alpha$, taux d'exploration $\varepsilon$ ; ε-greedy et softmax ont leur définition **standard** (fiche §26.6.2), sauf quand un exercice demande la version du livre.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 26.Q1 — Apprendre avec des récompenses : ni supervisé, ni non supervisé 🧠 ⏱️ 3 min
*Fiche §26.1, §26.2 · livre §26.1, §26.2*

a) Ce qui distingue le RL de l'apprentissage supervisé : (A) il n'utilise pas de réseau de neurones ; (B) on ne donne pas la bonne action, seulement un signal qui dit si le résultat est bon ou mauvais ; (C) il n'a pas besoin de données ; (D) il ne marche que pour les jeux.
b) Parmi ces problèmes, lesquels se posent naturellement en RL ? (A) régler la position d'attente des ascenseurs d'une tour au fil de la journée ; (B) reconnaître un chiffre manuscrit à partir de 60 000 images étiquetées ; (C) apprendre à un robot à marcher sans lui donner la suite des commandes ; (D) regrouper des clients en segments sans aucun label ; (E) jouer au backgammon contre un adversaire. Écris leurs lettres.
c) Vrai ou faux : en RL, la récompense qui juge vraiment une action (gagner ou perdre la partie) arrive toujours juste après cette action.
d) Vrai ou faux : le RL est un algorithme précis, comme les k plus proches voisins.
e) Pour le livre, le RL cherche : (A) la réponse objectivement juste ; (B) la meilleure réponse qu'on puisse trouver pour le critère qu'on s'est donné ; (C) la réponse la plus rapide à calculer.

### 26.Q2 — Les quatre informations données à l'agent du morpion 🧠 ⏱️ 3 min
*Fiche §26.2.1 · livre §26.2.1*

a) Dans l'exemple du morpion du livre, qui joue le rôle de l'environnement ? (A) l'amie qui joue les croix ; (B) nous, qui jouons les ronds et donnons les récompenses ; (C) le plateau seul.
b) Parmi ces informations, lesquelles l'agent reçoit-il à chaque tour ? (A) le plateau ; (B) la liste des coups possibles ; (C) la règle du jeu ; (D) une récompense ; (E) le nom du jeu ; (F) la consigne de chercher les plus grandes récompenses. Écris leurs lettres.
c) Vrai ou faux : l'agent doit dire à l'environnement comment il choisit ses coups.
d) Le livre propose une variante où l'agent peut proposer les neuf cases, même occupées. Son intérêt : (A) aller plus vite ; (B) que l'agent apprenne seul, par les récompenses, à éviter les coups interdits ; (C) rendre le jeu plus difficile pour l'environnement.
e) Sur un plateau de morpion où deux cases sont occupées, combien de coups sont permis ?

### 26.Q3 — État, actions, politique et informations privées 🧠 ⏱️ 3 min
*Fiche §26.3, §26.3.1 · livre §26.3, §26.3.1*

a) Remets dans l'ordre les trois temps de la boucle : (A) l'environnement répond ; (B) l'agent se met à jour ; (C) l'agent choisit une action. Écris les lettres dans l'ordre.
b) Une partie complète, du premier état au dernier, s'appelle : (A) une epoch ; (B) un épisode ; (C) une politique ; (D) une transition.
c) La politique est : (A) la récompense moyenne ; (B) la règle qui associe une action à chaque état ; (C) la liste des états possibles ; (D) le plateau de départ.
d) Vrai ou faux : dans le schéma du livre, c'est l'agent qui exécute l'action dans le monde.
e) Vrai ou faux : l'état d'un jeu de société peut contenir autre chose que la position des pièces (l'argent de chacun, des cartes cachées).

### 26.Q4 — Réponse de l'environnement, épisode et agents libres ou déclenchés 🧠 ⏱️ 3 min
*Fiche §26.3.2 à §26.3.4 · livre §26.3.2 à §26.3.4*

a) Après une action, l'environnement renvoie : (A) une nouvelle politique ; (B) un nouvel état et une récompense ; (C) la meilleure action ; (D) la valeur Q de l'action.
b) Un cuisinier attend la sonnerie de son minuteur, qui tourne de lui-même. Le cuisinier est (A) libre ou (B) déclenché ? Et le minuteur ? Écris les deux lettres, le cuisinier d'abord.
c) Au backgammon, joué chacun son tour, agent et environnement sont : (A) tous deux libres ; (B) tous deux déclenchés ; (C) l'un libre, l'autre déclenché.
d) Vrai ou faux : le cas où agent et environnement sont tous deux déclenchés est le plus simple à simuler.
e) Dans Gymnasium, que renvoie `env.step(action)` ? (A) `(state, reward)` ; (B) `(state, info)` ; (C) `(state, reward, terminated, truncated, info)` ; (D) `(reward, done)`.

### 26.Q5 — MDP, observabilité, attribution du crédit, exploration ou exploitation 🧠 ⏱️ 3 min
*Fiche §26.3.5 · livre §26.3.5*

a) La propriété de Markov dit que : (A) toutes les actions ont la même probabilité ; (B) l'état présent suffit pour prédire la suite, le passé n'apporte rien de plus ; (C) les récompenses sont toujours positives ; (D) l'environnement est déterministe.
b) Une partie de poker où l'agent ne voit pas les cartes de ses adversaires est un environnement : (A) totalement observable ; (B) partiellement observable.
c) Une partie est gagnée au 40ᵉ coup. Le problème de savoir lesquels des 40 coups ont mérité la victoire s'appelle : (A) le dilemme exploration-exploitation ; (B) l'attribution du crédit ; (C) l'overfitting d'une politique ; (D) l'actualisation.
d) Vrai ou faux : une politique qui joue toujours l'action la mieux notée explore suffisamment, puisqu'elle finit par tout essayer.
e) Le dilemme exploration-exploitation est le même que celui du ch. 11 : (A) la régression logistique ; (B) le bandit à bras multiples ; (C) le k-means.

### 26.Q6 — Retour total, retour actualisé et rôle de γ 🧠 ⏱️ 3 min
*Fiche §26.3.6, §26.3.7 · livre §26.3.6, §26.3.7*

a) Une partie de trois coups reçoit les récompenses $(2, 0, 1)$. Son retour total.
b) Le retour actualisé du premier coup, avec $\gamma = 0{,}5$.
c) Vrai ou faux : avec $\gamma = 0$, le retour d'un coup se réduit à sa récompense immédiate.
d) Selon la fiche, $\gamma$ sert surtout à : (A) mesurer combien l'environnement est prévisible ; (B) dire combien on préfère une récompense proche à une récompense lointaine, et garder une somme finie pour une tâche sans fin ; (C) régler la vitesse d'apprentissage.
e) L'horizon effectif $1/(1-\gamma)$ pour $\gamma = 0{,}95$.

### 26.Q7 — Flippers et L-learning : pourquoi « lousy » ? 🧠 ⏱️ 3 min
*Fiche §26.4, §26.5, §26.5.1 · livre §26.4, §26.5, §26.5.1*

a) À Flippers, un plateau est gagnant quand : (A) toutes les tuiles montrent un point ; (B) exactement trois points forment une ligne ou une colonne complète, toutes les autres tuiles vides ; (C) trois points sont alignés, quelle que soit la direction.
b) Combien de plateaux sont gagnants ?
c) Le L-learning met à jour une case en : (A) faisant la moyenne de l'ancienne valeur et du TFR ; (B) remplaçant l'ancienne valeur par le TFR du coup ; (C) ajoutant la récompense immédiate.
d) Vrai ou faux : selon le livre (fig. 26.23 à 26.26), sans le camion, le L-learning qui jouait en six coups deux plateaux faciles les joue en deux après 6 000 parties.
e) Pourquoi le camion fait-il échouer le L-learning ? (A) il rend le jeu impossible à gagner ; (B) une seule partie malchanceuse efface la bonne valeur d'un coup ; (C) il change le nombre de tuiles.

### 26.Q8 — La mise à jour du Q-learning : α, γ et la valeur de l'état suivant 🧠 ⏱️ 3 min
*Fiche §26.6, §26.6.1, §26.6.3 · livre §26.6, §26.6.1, §26.6.3*

a) Dans la cible $r + \gamma \max_{a'} Q(s', a')$, le maximum est pris sur la ligne de : (A) l'état courant $s$ ; (B) l'état suivant $s'$ ; (C) l'état initial.
b) Avec $\alpha = 0$, une mise à jour : (A) remplace la valeur par la cible ; (B) ne change rien ; (C) met la valeur à 0.
c) Avec $\alpha = 1$, la mise à jour revient à : (A) faire la moyenne ; (B) remplacer la valeur par la cible, comme le L-learning ; (C) ne rien faire.
d) Vrai ou faux : le Q-learning attend la fin de l'épisode pour mettre la table à jour.
e) $Q(s, a) = 0{,}4$, $r = 0$, $\gamma = 1$, $\max_{a'} Q(s', a') = 0{,}8$, $\alpha = 0{,}5$. La nouvelle valeur de $Q(s, a)$.

### 26.Q9 — ε-greedy ou softmax : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §26.6.2 · livre §26.6.2*

a) Vrai ou faux : dans la définition standard d'ε-greedy, l'exploration tire une action parmi **toutes** les actions, la meilleure comprise.
b) Avec ε-greedy standard, $\varepsilon = 0{,}1$ et 5 actions, la probabilité de jouer l'action gloutonne.
c) Vrai ou faux : une softmax de grande température choisit presque toujours la meilleure action.
d) Vrai ou faux : contrairement à ε-greedy, softmax explore surtout les actions dont la valeur est proche de la meilleure.
e) La « softmax » du livre divise les valeurs par leur somme. Que donne-t-elle pour une ligne de zéros, celle de toute table au début ? (A) des probabilités égales ; (B) une division par zéro : elle n'est pas définie ; (C) toujours l'action 0.

### 26.Q10 — Table initialisée à zéro : comment la valeur remonte et converge 🧠 ⏱️ 3 min
*Fiche §26.6.4, §26.6.5 · livre §26.6.4, §26.6.5*

a) Au début de l'entraînement, la table Q vaut 0 partout et toutes les récompenses intermédiaires sont nulles. Les premières mises à jour : (A) laissent la table à 0 jusqu'à la première récompense non nulle ; (B) donnent des valeurs au hasard ; (C) font diverger la table.
b) Après une première victoire, quelle case reçoit une valeur non nulle ? (A) toutes les cases de la partie ; (B) celle de la dernière action, qui a mené à la victoire ; (C) celle de la première action.
c) Vrai ou faux : la valeur d'une victoire remonte la chaîne des états d'un pas par épisode environ, parce que chaque cible regarde l'état suivant.
d) Les preuves de convergence du Q-learning demandent : (A) que chaque case soit visitée une infinité de fois, et un $\alpha$ qui décroît ; (B) un $\alpha$ égal à 1 ; (C) une politique purement gloutonne.
e) Dans l'expérience du livre, combien de parties d'entraînement donnent les parties les plus courtes ? (A) 300 ; (B) 3 000 ; (C) 6 000.

### 26.Q11 — SARSA contre Q-learning : l'action suivante réellement jouée 🧠 ⏱️ 3 min
*Fiche §26.7 à §26.7.2 · livre §26.7 à §26.7.2*

a) SARSA tient son nom de : (A) *State, Action, Reward, State, Action* ; (B) *Stochastic Approximation with Reward and Stable Actions* ; (C) son inventeur.
b) Dans la cible de SARSA, $Q(s', a')$ est la valeur de : (A) la meilleure action de $s'$ ; (B) l'action $a'$ que la politique a choisie dans $s'$, et qui sera jouée ; (C) l'action jouée au pas précédent.
c) Lequel est **hors politique** (*off-policy*) ? (A) le Q-learning ; (B) SARSA.
d) Vrai ou faux : quand l'action suivante choisie est l'action gloutonne, SARSA et le Q-learning font exactement la même mise à jour.
e) Vrai ou faux : selon la fiche, SARSA est meilleur que le Q-learning dans tous les cas.

### 26.Q12 — Tables géantes, rejeu d'expérience, AlphaGo Zero et lancer de rayons 🧠 ⏱️ 4 min
*Fiche §26.8 à §26.10 · livre §26.8 à §26.10*

a) Combien de colonnes a la table Q d'un morpion 4 × 4 (une par action possible) ?
b) Le deep RL remplace la table par : (A) un arbre de décision ; (B) un réseau de neurones qui reçoit l'état et rend une valeur par action ; (C) une table plus petite.
c) La mémoire de rejeu de DQN : (A) rejoue des parties entières dans l'ordre ; (B) garde des transitions et en tire des mini-batches au hasard ; (C) enregistre les meilleures parties seulement.
d) Vrai ou faux : AlphaGo Zero a appris le go à partir de parties de joueurs humains.
e) Pendant l'auto-apprentissage (*self-play*), l'adversaire d'un joueur fait partie : (A) de sa politique ; (B) de son environnement ; (C) de sa récompense.

<a id="rappels"></a>

## 🔁 Rappels

### 26.R1 — Un réseau comme fonction qui remplace une table (le décodeur d'un VAE) 🔁 ⏱️ 5 min
*ch. 25 · fiche §26.8*

a) Le décodeur d'un VAE à 20 latents reçoit un code de 20 nombres réels. Combien de codes différents pourrait-il recevoir ? (A) 20 ; (B) $2^{20}$ ; (C) une infinité : aucune table ne pourrait les lister.
b) Vrai ou faux : un réseau rend une sortie raisonnable pour une entrée qu'il n'a jamais vue si elle ressemble à ses données d'entraînement, ce qu'une table ne sait pas faire.
c) Un DQN pour un jeu à 4 actions rend, pour chaque état, combien de nombres ?

### 26.R2 — Que sauvegarder pour reprendre un entraînement ? 🔁 ⏱️ 5 min
*ch. 23*

a) Pour reprendre un entraînement PyTorch exactement où il s'était arrêté, que faut-il sauvegarder ? (A) le `state_dict` du modèle ; (B) le `state_dict` de l'optimiseur ; (C) le numéro de l'epoch ; (D) l'état des générateurs aléatoires ; (E) le code source de PyTorch. Écris les lettres nécessaires.
b) Pour un agent de Q-learning tabulaire, l'équivalent du `state_dict` est : (A) la table Q ; (B) la liste des récompenses ; (C) l'environnement.
c) Vrai ou faux : pour reprendre un Q-learning à $\varepsilon$ décroissant, il suffit de sauvegarder la table Q.

### 26.R3 — Bandit à bras multiples, ε-greedy et UCB 🔁 ⏱️ 5 min
*ch. 11 · fiche §26.6.2*

a) Un bandit à bras multiples est un problème de RL avec : (A) un seul état ; (B) deux états ; (C) un état par bras.
b) La mise à jour incrémentale du ch. 11 : estimation 3, nouveau gain 5, pas 0,25. La nouvelle estimation.
c) UCB choisit le bras qui maximise : (A) son gain moyen seul ; (B) son gain moyen plus un bonus qui grandit quand le bras a été peu essayé ; (C) un tirage au hasard.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 26.1 — Retours total et actualisé d'une partie de six coups ✏️ ★ ⏱️ 10 min
**Objectif :** calculer à la main un retour total, un retour d'un coup et des retours actualisés pour plusieurs γ.
**Prérequis :** fiche §26.3.7 · 0B (puissances) · **Parcours :** R, M

Une partie de six coups, numérotés de 1 à 6, reçoit les récompenses $(1, 0, 0, 2, 0, 4)$ (la récompense du coup 6 est la dernière).

a) La récompense totale de la partie.
b) Le TFR du coup 3 : sa récompense et toutes celles qui suivent ($\gamma = 1$).
c) Le retour actualisé du coup 1 avec $\gamma = 0{,}5$, à 3 décimales.
d) Le retour actualisé du coup 1 avec $\gamma = 0{,}9$, à 3 décimales.
e) Le retour actualisé du coup 4 avec $\gamma = 0{,}9$, à 2 décimales.
f) Le retour actualisé du coup 1 avec $\gamma = 0$.
g) Vrai ou faux : avec $\gamma = 1$ et des récompenses toutes positives ou nulles, le retour ne peut que diminuer (ou rester égal) d'un coup au suivant.

Dans ta copie : vérifie d) en calculant les retours à rebours, $G_t = r_t + \gamma\,G_{t+1}$, du coup 6 au coup 1.

### Ex 26.2 — Taille des tables : de Flippers au Go ✏️ ★ ⏱️ 10 min
**Objectif :** compter les états et les cases d'une table Q, et voir quand une table devient impossible.
**Prérequis :** fiche §26.4, §26.8 · 0B (puissances, logarithmes) · **Parcours :** R, M

a) Le nombre de plateaux de Flippers 3 × 3 (chaque tuile a un point ou non).
b) Le nombre de cases de sa table Q (une ligne par plateau, une colonne par tuile à retourner).
c) Le nombre de cases de la table Q d'un Flippers 4 × 4 (16 tuiles).
d) Le nombre de plateaux gagnants d'un Flippers 4 × 4, si l'on gagne avec exactement quatre points qui forment une ligne ou une colonne complète, toutes les autres tuiles vides.
e) Le nombre de codages d'un plateau de morpion 3 × 3, si chaque case est vide, une croix ou un rond (on compte aussi les plateaux impossibles).
f) Le nombre de cases de la table Q d'un morpion 4 × 4 (un codage par ligne, une colonne par case du plateau).
g) Le nombre de chiffres de $3^{361}$, le nombre de codages d'un plateau de go 19 × 19 (prends $\log_{10} 3 \approx 0{,}47712$).

Dans ta copie : le livre compte 9 colonnes pour le morpion 4 × 4. Combien de cases de trop ou de moins cela fait-il ? Pourquoi la plupart des codages de e) ne servent-ils à rien ?

### Ex 26.3 — Une mise à jour de Q-learning à la main ✏️ ★ ⏱️ 10 min
**Objectif :** appliquer pas à pas la mise à jour du Q-learning, avec et sans état terminal.
**Prérequis :** fiche §26.6.1, §26.6.3 · **Parcours :** R, M

Dans l'état $s$, l'agent joue l'action $a$ et reçoit $r = 0{,}2$. Avant la mise à jour, $Q(s, a) = 0{,}5$, et la ligne de l'état suivant $s'$ vaut $(0{,}1 ; 0{,}6 ; 0{,}3 ; 0{,}6)$. On prend $\gamma = 0{,}9$ et $\alpha = 0{,}5$.

a) La cible $y = r + \gamma \max_{a'} Q(s', a')$, à 2 décimales.
b) L'erreur $\delta = y - Q(s, a)$, à 2 décimales.
c) La nouvelle valeur de $Q(s, a)$, à 2 décimales.
d) La même transition, mais $s'$ est **terminal** : la nouvelle valeur de $Q(s, a)$, à 2 décimales.
e) Le cas de c) avec $\alpha = 1$ : la nouvelle valeur, à 2 décimales.
f) Vrai ou faux : pendant ce pas, le Q-learning modifie aussi des valeurs de la ligne de $s'$.

Dans ta copie : pourquoi les deux 0,6 de la ligne de $s'$ ne posent-ils aucun problème à la cible ? Que poseraient-ils à une politique gloutonne qui doit **choisir** une action ?

### Ex 26.4 — La même transition vue par SARSA ✏️ ★ ⏱️ 10 min
**Objectif :** appliquer la mise à jour de SARSA et la comparer à celle du Q-learning.
**Prérequis :** Ex 26.3 · fiche §26.7 · **Parcours :** R, M

On reprend la transition de l'Ex 26.3 ($Q(s, a) = 0{,}5$, $r = 0{,}2$, ligne de $s'$ : $(0{,}1 ; 0{,}6 ; 0{,}3 ; 0{,}6)$, $\gamma = 0{,}9$, $\alpha = 0{,}5$). La politique ε-greedy a choisi, dans $s'$, l'action **2** (les actions sont numérotées à partir de 0).

a) La cible de SARSA, à 2 décimales.
b) La nouvelle valeur de $Q(s, a)$ selon SARSA, à 3 décimales.
c) La valeur du Q-learning (Ex 26.3 c) moins celle de SARSA, à 3 décimales.
d) Vrai ou faux : si la politique avait choisi l'action 1, SARSA aurait donné la même valeur que le Q-learning.
e) Lequel apprend la valeur de la politique qu'il suit, exploration comprise ? (A) le Q-learning ; (B) SARSA.

### Ex 26.5 — Probabilités d'action : ε-greedy (standard et version du livre) et softmax ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer les probabilités de chaque action selon ε-greedy, standard ou du livre, et selon softmax.
**Prérequis :** fiche §26.6.2 · ch. 17 (softmax) · ch. 11 (ε-greedy) · **Parcours :** M

Une ligne de la table compte quatre actions, de valeurs $Q = (1, 2, 4, 0)$. On prend $\varepsilon = 0{,}2$.

a) ε-greedy **standard** : la probabilité de jouer l'action 2 (la gloutonne), à 2 décimales.
b) ε-greedy standard : la probabilité de jouer l'action 0, à 2 décimales.
c) La version du **livre** (l'exploration tire parmi les **autres** actions) : la probabilité de jouer l'action 2, à 2 décimales.
d) La version du livre : la probabilité de jouer l'action 0, à 3 décimales.
e) Softmax de température $\tau = 1$ : la probabilité de jouer l'action 2, à 3 décimales.
f) Softmax de température $\tau = 2$ : la probabilité de jouer l'action 2, à 3 décimales.
g) Vrai ou faux : ajouter 10 à toutes les valeurs de la ligne change les probabilités de la softmax.
h) La « softmax » du livre (les valeurs divisées par leur somme) : la probabilité de jouer l'action 2, à 3 décimales.

Dans ta copie : laquelle des deux ε-greedy explore le plus les mauvaises actions, et de combien ? Que donnerait h) si l'une des valeurs était négative ?

### Ex 26.6 — Regarder la valeur remonter une chaîne d'états ✏️ ★★ ⏱️ 20 min
**Objectif :** suivre, épisode par épisode, la remontée de la valeur d'une victoire dans une table initialisée à zéro.
**Prérequis :** Ex 26.3 · fiche §26.6.4 · **Parcours :** M

Une chaîne d'états $0 \to 1 \to 2 \to 3$, avec une seule action, « avancer ». L'état 3 est terminal. Le pas de 2 à 3 rapporte $r = 1$, les autres 0. La table part de 0, $\gamma = 0{,}9$. Chaque épisode part de l'état 0 et va jusqu'à 3 ; les mises à jour du Q-learning se font **dans l'ordre des pas** (d'abord l'état 0, puis 1, puis 2).

Avec $\alpha = 1$ :
a) $Q(2)$ à la fin de l'épisode 1.
b) $Q(1)$ à la fin de l'épisode 2, à 2 décimales.
c) $Q(0)$ à la fin de l'épisode 3, à 2 décimales.
d) Le numéro du premier épisode à la fin duquel $Q(0)$ n'est plus nul.

Avec $\alpha = 0{,}5$ :
e) $Q(1)$ à la fin de l'épisode 2, à 3 décimales.
f) $Q(2)$ à la fin de l'épisode 3, à 3 décimales.
g) La limite de $Q(0)$ quand le nombre d'épisodes tend vers l'infini, à 2 décimales.

h) Avec $\alpha = 1$ et une chaîne de 10 états non terminaux avant le but ($0 \to 1 \to \dots \to 9 \to 10$, l'état 10 terminal), le numéro du premier épisode à la fin duquel $Q(0)$ n'est plus nul.

Dans ta copie : que changerait le fait de faire les mises à jour à rebours, de l'état 2 à l'état 0, à la fin de chaque épisode ? Quel algorithme du livre le fait déjà ?

### Ex 26.7 — Somme géométrique et horizon effectif de γ ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer la formule de la somme géométrique et en tirer l'horizon effectif d'un facteur d'actualisation.
**Prérequis :** fiche §26.3.7 (encadré 🧮) · 0B (Ex 0B.13 et 0B.14, logarithmes) · **Parcours :** M

Réponds dans ta copie.

1. Pour $0 \le \gamma < 1$, on note $S_n = 1 + \gamma + \dots + \gamma^{n-1}$. Calcule $S_n - \gamma S_n$, et déduis-en $S_n = \frac{1 - \gamma^n}{1 - \gamma}$, puis la limite $\frac{1}{1-\gamma}$.
2. Des récompenses toutes égales à $r_{\max} > 0$, pour toujours. Montre que le retour actualisé vaut $\frac{r_{\max}}{1-\gamma}$. Pourquoi aucun retour actualisé de récompenses bornées par $r_{\max}$ ne peut-il le dépasser ?
3. Calcule l'horizon effectif $\frac{1}{1-\gamma}$ pour $\gamma = 0{,}5$, $0{,}9$ et $0{,}99$.
4. À partir de combien de pas $k$ le poids $\gamma^k$ passe-t-il sous $\frac{1}{2}$ ? Exprime $k$ avec des logarithmes, puis calcule-le pour $\gamma = 0{,}9$ et $\gamma = 0{,}99$.
5. Avec le $\gamma = 0{,}2$ du livre, quelle part de sa valeur garde une récompense de victoire reçue 4 pas après le coup présent (son poids est $\gamma^4$) ? Et 6 pas après ? Que dit cela de la capacité de l'agent à préférer la victoire la plus proche ?

### Ex 26.8 — Équation de Bellman : le point fixe que vise le Q-learning ∂ ★★★ ⏱️ 35 min
**Objectif :** montrer que les valeurs optimales sont le point fixe de la mise à jour de Bellman, et le calculer sur un petit MDP.
**Prérequis :** Ex 26.3 · fiche §26.6.4 (encadré 🧮) · 0B (espérance) · **Parcours :** M

Un MDP à deux états non terminaux, $A$ et $B$, et un état terminal $T$. Dans $A$, l'action « rester » rapporte 0 et laisse en $A$ ; l'action « aller » rapporte 0 et mène en $B$. Dans $B$, l'action « finir » rapporte 1 et mène en $T$ ; l'action « jouer » rapporte 2 (quelle que soit l'issue) et mène en $T$ avec la probabilité $\frac{1}{2}$, en $A$ avec la probabilité $\frac{1}{2}$. On prend $\gamma = 0{,}9$. Réponds dans ta copie.

1. Écris l'équation d'optimalité de Bellman pour chacune des quatre valeurs $Q^*(A, \text{rester})$, $Q^*(A, \text{aller})$, $Q^*(B, \text{finir})$, $Q^*(B, \text{jouer})$.
2. On suppose que la politique optimale joue « aller » dans $A$ et « jouer » dans $B$. Résous le système et calcule les quatre valeurs. Vérifie que cette politique est bien gloutonne pour ces valeurs.
3. On écrit $\mathcal{T}$ la mise à jour de Bellman, qui transforme une table $Q$ en la table des seconds membres. Montre que, pour deux tables $Q_1$ et $Q_2$, $\max |\mathcal{T}Q_1 - \mathcal{T}Q_2| \le \gamma \max |Q_1 - Q_2|$ (utilise $|\max_a x_a - \max_a y_a| \le \max_a |x_a - y_a|$). Pourquoi cela garantit-il qu'en répétant $\mathcal{T}$ à partir de n'importe quelle table, on converge vers une seule table, $Q^*$ ?
4. Le Q-learning ne connaît pas les probabilités de transition. Explique comment sa cible $r + \gamma \max_{a'} Q(s', a')$ est un **échantillon** du second membre, et pourquoi un $\alpha$ constant proche de 1 empêche de faire la moyenne de ces échantillons dans l'action « jouer ».

### Ex 26.9 — Mémoire d'une table Q : morpion 4×4, 5×5 et Go 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer la mémoire d'une table Q, et voir quand une table devient démesurée, puis impossible.
**Prérequis :** Ex 26.2 · fiche §26.8 · **Parcours :** R, M

Une valeur Q est un `float32` (4 octets). Réponds dans ta copie, avec des ordres de grandeur.

1. La mémoire de la table Q d'un morpion 3 × 3, codée comme à l'Ex 26.2 e) avec 9 colonnes.
2. La même chose pour le morpion 4 × 4 (16 colonnes), puis 5 × 5 (25 colonnes), puis le go 19 × 19 ($3^{361}$ codages, 361 colonnes). Compare à la mémoire d'un ordinateur portable (16 Go) et d'un grand centre de données (environ un exaoctet, $10^{18}$ octets).
3. Si l'on ne garde que les positions réellement atteignables (5 478 au morpion 3 × 3), combien la table gagne-t-elle ?
4. Pour remplir une table, chaque case doit être visitée plusieurs fois. À un milliard de mises à jour par seconde, combien de temps faudrait-il pour visiter **une** fois chaque case de la table du 5 × 5 ?
5. Que fait un réseau à la place de la table, et combien de paramètres aurait, par exemple, un réseau dense 25 → 256 → 256 → 25 ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 26.10 — Attribution du crédit et exploration expliquées en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant, sans formule, les deux difficultés du RL.
**Prérequis :** fiche §26.3.5, §26.6.4 · **Parcours :** R

Explique en cinq lignes au plus, à quelqu'un qui n'a jamais entendu parler de RL, (1) pourquoi il est difficile de savoir quels coups d'une partie gagnée étaient bons, et comment le Q-learning s'en sort, puis (2) pourquoi un agent doit parfois jouer un coup qu'il croit moins bon. Une image du quotidien par idée.

### Ex 26.11 — Quand l'agent optimise la mauvaise récompense ⚖️ ★★ ⏱️ 20 min
**Objectif :** analyser des récompenses mal conçues et leurs conséquences, et proposer des garde-fous.
**Prérequis :** fiche §26.2 (encadré 💼), §26.3.7, « Au-delà du livre » · **Parcours :** R

Réponds dans ta copie.

1. Un bateau de course virtuel est récompensé pour chaque bonus ramassé. Il apprend à tourner en rond dans une lagune où les bonus réapparaissent, sans jamais finir la course. Pourquoi est-ce, pour l'agent, une **bonne** stratégie ? Quelle récompense aurait évité cela, et à quel prix ?
2. Une plateforme vidéo récompense son système de recommandation par le temps passé sur le site. Décris deux effets indésirables possibles, pour les utilisateurs et pour la société. Quelles mesures surveillerais-tu en plus de la récompense ?
3. Un assistant conversationnel est ajusté par RLHF sur un modèle de récompense appris à partir de préférences humaines. Pourquoi peut-il apprendre à flatter l'utilisateur ou à paraître sûr de lui plutôt qu'à avoir raison ? À quoi sert la pénalité KL qui le garde proche de sa version de départ ?
4. Qui est responsable quand un agent fait ce qu'on a récompensé, mais pas ce qu'on voulait : l'agent, l'ingénieur qui a écrit la récompense, l'entreprise qui le déploie ? Donne un argument pour chacun, puis ta position.

### Ex 26.12 — Lire Mnih et coll. (2015), « Human-level control through deep reinforcement learning » 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article du DQN et relier chacune de ses idées au chapitre.
**Prérequis :** fiche §26.6, §26.8, §26.9 · ch. 21 (convolutions) · **Parcours :** M

Lis V. Mnih et coll., « Human-level control through deep reinforcement learning », *Nature* 518, 529-533, 2015 ([doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)) : le résumé, le texte principal, la figure 1 et la section *Methods* (« Training algorithm for deep Q-networks »). Si tu n'as pas accès à *Nature*, la version d'atelier de 2013 est libre : [arXiv:1312.5602](https://arxiv.org/abs/1312.5602). Réponds dans ta copie.

1. Que reçoit le réseau en entrée, et que rend-il en sortie ? Combien de sorties pour un jeu à 18 actions, et pourquoi ce choix plutôt qu'une sortie unique par couple (état, action) ?
2. Écris la cible de la loss du DQN avec les notations de la fiche. Quelle différence avec la cible du Q-learning tabulaire ? Qu'est-ce que le **réseau cible**, et quel problème règle-t-il ?
3. Décris la mémoire de rejeu de l'article : sa taille, ce qu'elle contient, comment on y tire. Donne les deux raisons qu'avancent les auteurs pour l'utiliser.
4. Comment $\varepsilon$ évolue-t-il pendant l'entraînement ? Comparez-le au conseil du livre.
5. Pourquoi les récompenses sont-elles **écrêtées** entre −1 et 1 ? Qu'y perd-on ?
6. Sur quels jeux le DQN reste-t-il faible, et pourquoi (pense à l'attribution du crédit et à l'exploration) ?

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### 26.E1 — Q-learning ou SARSA : on-policy, off-policy, lequel choisir ? 💼 ★★ ⏱️ 10 min
« Quelle est la différence entre le Q-learning et SARSA ? Dans quel cas prendriez-vous l'un plutôt que l'autre ? »

### 26.E2 — À quoi sert le facteur d'actualisation γ ? 💼 ★★ ⏱️ 10 min
« À quoi sert le facteur d'actualisation, et comment le choisissez-vous ? »

### 26.E3 — Comment gérer le compromis exploration/exploitation ? 💼 ★★ ⏱️ 10 min
« Comment un agent de RL gère-t-il le compromis entre exploration et exploitation ? Citez plusieurs méthodes. »

### 26.E4 — Pourquoi DQN utilise-t-il un replay buffer et un réseau cible ? 💼 ★★ ⏱️ 10 min
« Pourquoi le DQN a-t-il besoin d'une mémoire de rejeu et d'un réseau cible ? »

### 26.E5 — Qu'est-ce que le RLHF et pourquoi utilise-t-on le RL avec les LLM ? 💼 ★★ ⏱️ 10 min
« Qu'est-ce que le RLHF, et pourquoi utilise-t-on l'apprentissage par renforcement pour entraîner les modèles de langage ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 26.13 | Prendre en main gymnasium avec FrozenLake | 📦 | ★ | 15 |
| 26.14 | Retours actualisés et retours à venir | 🔨 | ★ | 15 |
| 26.15 | Reproduire les courbes d'actualisation (fig. 26.13-26.14) | 🎨 | ★ | 15 |
| 26.16 | Prédire l'effet du camion sur la table L | 🔮 | ★ | 10 |
| 26.17 | Flippers : la longueur optimale de chaque partie par recherche exhaustive | 🔬 | ★★ | 25 |
| 26.18 | L-learning sur Flippers, sans puis avec camion | 🔬 | ★★★ | 40 |
| 26.19 | Politiques gloutonne et softmax (ε-greedy importée du ch. 11) | 🔨 | ★★ | 20 |
| 26.20 | Mises à jour de Q-learning et de SARSA, états terminaux compris | 🔨 | ★★ | 25 |
| 26.21 | La boucle d'entraînement tabulaire et l'évaluation d'une politique | 🔨 | ★★★ | 45 |
| 26.22 | Q-learning sur FrozenLake, glissant ou non | 📦 | ★★ | 25 |
| 26.23 | Lire des courbes d'apprentissage RL bruitées | 📈 | ★★ | 15 |
| 26.24 | L'agent qui n'apprenait rien | 🐛 | ★★ | 25 |
| 26.25 | Q-learning sur Flippers avec camion : 300, 3 000 puis 6 000 parties | 🔬 | ★★★ | 40 |
| 26.26 | Itération sur les valeurs : la référence quand le modèle est connu | 🔨 | ★★★ | 40 |
| 26.27 | α, γ, ε : balayage d'hyperparamètres sur Flippers bruité | 🔬 | ★★★ | 45 |
| 26.28 | Prédire le chemin appris par SARSA et par Q-learning au bord de la falaise | 🔮 | ★ | 10 |
| 26.29 | Q-learning contre SARSA sur Flippers et CliffWalking | 🔬 | ★★★ | 45 |
| 26.30 | Reproduire les comparaisons au fil de l'entraînement (fig. 26.58-26.60) | 🎨 | ★★★ | 35 |
| 26.31 | Une mémoire de rejeu (replay buffer) | 🔨 | ★★ | 20 |
| 26.32 | Concevoir la récompense : finale, pénalité par coup ou façonnée | 🔬 | ★★ | 30 |
| 26.33 | Rejouer l'expérience pour apprendre avec moins d'épisodes | 🔬 | ★★★ | 35 |
| 26.34 | Rapporter un résultat RL honnête : plusieurs graines, configuration, intervalle | 🛠️ | ★★ | 25 |
| 26.35 | Morpion contre un adversaire aléatoire : masque d'actions ou pénalité | 📦 | ★★★ | 45 |
| 26.36 | CartPole : discrétiser un état continu pour un Q-learning tabulaire | 📦 | ★★★ | 45 |
| 26.37 | Auto-apprentissage au morpion : l'agent joue les deux camps | 🔬 | ★★★★ | 90 |
| 26.38 | Défi CartPole tabulaire : tenir 195 pas en moyenne | 🏆 | ★★★★ | 120 |

Aucun exercice ne demande de GPU : les tables tiennent sur un CPU, et les boucles d'environnement dominent le temps de calcul.
