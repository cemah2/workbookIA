# B8 · RL moderne : DQN, gradient de politique, PPO et RLHF — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/b8_rl_moderne/06_mes_reponses.md` (créée par `python tools/start_chapter.py B8`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ B8.1 à B8.6 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la dérivation ∂ B8.7, l'estimation 🧮 B8.8, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), les lettres ensemble pour plusieurs choix, dans n'importe quel ordre (`"EF"`), une forme entre guillemets (`"(3, 5)"`).
> Conventions du chapitre (fiche, encadré 📐) : $s$, $a$, $r$, $s'$ l'état, l'action, la récompense et l'état suivant ; $\gamma$ le facteur d'actualisation ; $d = 1$ quand $s'$ termine **vraiment** l'épisode (`terminated`) ; $Q_\theta$ le réseau en ligne, $Q_{\bar\theta}$ le réseau cible ; $\pi_\theta$ la politique, $V$ le critique, $\delta_t$ l'erreur TD, $\hat A_t$ l'avantage, $\rho$ (ou `ratio`) le ratio de probabilités de PPO, $\varepsilon$ la largeur du clip (0,2 sauf mention contraire), $\sigma$ la sigmoïde, $\ln$ le logarithme naturel. Les actions et les pas sont numérotés à partir de 0, comme en Python.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### B8.Q1 — Pourquoi une table Q ne suffit plus pour CartPole 🧠 ⏱️ 3 min
*Fiche §208.1 · parcours R*

a) Tu découpes chacune des quatre variables de l'état de CartPole en 6 intervalles. Combien de cases contient la table Q, une par état discrétisé et par action ?
b) Avec un réseau Q construit comme en 208.1, combien de passes avant faut-il pour choisir l'action gloutonne dans un état ?
c) Vrai ou faux : avec un réseau Q, un pas de gradient sur la valeur d'un état change en général aussi les valeurs des états voisins.
d) Vrai ou faux : si l'on laisse le gradient traverser aussi la cible $y$, on obtient la même mise à jour qu'avec `y.detach()`, seulement plus lente.
e) `states` est un tenseur de 32 états de CartPole, de forme `(32, 4)`. Quelle est la forme de `q_net(states)` ?

### B8.Q2 — La triade mortelle : approximation, amorçage et apprentissage hors politique 🧠 ★★ ⏱️ 3 min
*Fiche §208.1, §208.2 · parcours —*

a) Lesquels sont les trois ingrédients de la triade mortelle ? (A) l'approximation de fonction ; (B) un learning rate trop grand ; (C) l'amorçage ; (D) l'apprentissage off-policy ; (E) la loss de Huber. (une ou plusieurs lettres)

Pour b) à d), compte les ingrédients de la triade présents dans chaque méthode (un nombre de 0 à 3).

b) Le Q-learning tabulaire du ch. 26, avec ε-greedy.
c) Un critique $V_w(s)$, un réseau, entraîné par régression sur les retours $G_t$ **observés** jusqu'à la fin des épisodes que joue la politique actuelle.
d) SARSA avec un réseau Q à la place de la table, sans mémoire de rejeu, sur les transitions de sa propre politique ε-greedy.
e) Vrai ou faux : le réseau cible de DQN retire l'un des trois ingrédients de la triade.
f) Vrai ou faux : une mémoire de rejeu, qui garde des transitions jouées par d'anciennes versions de la politique, rend l'apprentissage off-policy, même si l'agent n'explore plus.

### B8.Q3 — Réseau cible et mémoire de rejeu : quel problème chacun résout-il ? Vrai ou faux 🧠 ⏱️ 3 min
*Fiche §208.2 · parcours R*

a) Une mémoire de rejeu de 10 000 transitions est pleine ; à chaque pas, une transition nouvelle remplace la plus ancienne, et une mise à jour tire un mini-batch de 64 transitions distinctes, uniformément. Pendant son séjour dans la mémoire, combien de fois une transition est-elle tirée, en moyenne ?
b) Vrai ou faux : avec une mémoire de 1 000 transitions au lieu de 10 000, et les mêmes réglages, chaque transition serait tirée moins souvent pendant son séjour.
c) Vrai ou faux : entre deux copies du réseau cible, deux calculs de la cible d'une même transition donnent la même valeur, même si le réseau en ligne a fait 300 pas d'Adam entre-temps.
d) Avec la mise à jour douce $\bar\theta \leftarrow \tau\theta + (1 - \tau)\bar\theta$, appliquée à chaque pas avec $\tau = 0{,}005$ : quelle part des poids qu'avait le réseau cible il y a 200 pas compte encore dans ses poids actuels, à 2 décimales ?
e) Vrai ou faux : une mémoire de rejeu permet d'apprendre plusieurs fois d'une transition rare (la chute qui suit un long épisode) au lieu de l'oublier aussitôt.

### B8.Q4 — Double DQN : d'où vient la surestimation des valeurs Q ? 🧠 ★★ ⏱️ 3 min
*Fiche §208.2 · parcours R*

a) Toutes les actions de $s'$ valent vraiment 0 ; leurs estimations sont justes en moyenne, mais bruitées. En moyenne, la cible $r + \gamma \max_{a'} Q(s', a')$ est : (A) juste ; (B) trop haute ; (C) trop basse.
b) Trois actions valent 0 ; chacune est estimée $+1$ ou $-1$ avec la même probabilité, indépendamment des autres. Que vaut l'espérance du maximum des trois estimations, à 2 décimales ?
c) Dans la situation de b), on choisit l'action avec un premier jeu d'estimations, puis on lit sa valeur dans un second jeu, tiré de la même façon mais indépendamment du premier. Que vaut l'espérance de la valeur lue ?
d) Dans Double DQN : (A) le réseau en ligne choisit l'action de $s'$, et le réseau cible donne sa valeur ; (B) le réseau cible choisit l'action de $s'$, et le réseau en ligne donne sa valeur ; (C) deux réseaux en ligne, entraînés chacun sur la moitié des transitions de la mémoire, se répartissent les deux rôles.
e) Vrai ou faux : Double DQN double le nombre de poids qu'Adam entraîne.
f) Vrai ou faux : si toutes les valeurs Q d'un état sont surestimées de la même quantité ($+5$ pour chaque action), la politique gloutonne ne change pas dans cet état.

### B8.Q5 — Méthodes par valeur ou par politique : actions continues et politiques aléatoires 🧠 ⏱️ 3 min
*Fiche §208.3 · parcours R*

a) Un bras robotique règle le couple de 6 moteurs, des nombres réels. Quelle approche convient directement ? (A) une table Q où chaque moteur prend 10 valeurs possibles, sans aucun réseau, dont on lit l'action gloutonne par un `argmax` sur chaque ligne ; (B) un DQN ; (C) une politique qui rend une moyenne et un écart-type par moteur.
b) Dans l'approche (A) de a), combien d'actions discrètes la table compte-t-elle par état ?
c) Une politique softmax à trois actions a les logits $(\ln 2 ;\ \ln 6 ;\ 0)$. Quelle est la probabilité de l'action 1, à 2 décimales ?
d) Lesquels sont des avantages d'un gradient de politique sur DQN ? (A) il traite directement des actions continues ; (B) il peut représenter une politique aléatoire, quand c'est elle la meilleure ; (C) il réutilise chaque transition de nombreuses fois ; (D) l'exploration est intégrée à la politique, sans ε. (une ou plusieurs lettres)
e) Vrai ou faux : au jeu de « pair ou impair » (chaque joueur montre un ou deux doigts ; l'un gagne si la somme est paire, l'autre sinon), contre un adversaire qui exploite toute régularité, la meilleure politique est déterministe.

### B8.Q6 — REINFORCE : pourquoi soustraire une baseline ne biaise pas le gradient 🧠 ★★ ⏱️ 3 min
*Fiche §208.3 · parcours R, M*

Une pièce truquée tombe sur pile avec la probabilité $p_\theta(\text{pile}) = \theta = 0{,}3$.

a) Que vaut la fonction score $\frac{d}{d\theta} \ln p_\theta(\text{face})$, à 2 décimales ?
b) Que vaut l'espérance de la fonction score, sur les deux issues, à 2 décimales ?
c) Laquelle de ces baselines **biaise** le gradient de REINFORCE ? (A) la moyenne des retours des 20 derniers épisodes ; (B) un critique $V_w(s_t)$ ; (C) le retour $G_t$ de l'épisode en cours, au même pas.
d) Vrai ou faux : sur CartPole, une baseline constante $b = 1\,000$, très loin des retours observés, laisse l'espérance du gradient inchangée mais en augmente la variance.
e) Vrai ou faux : la baseline corrige aussi un biais de l'estimation de REINFORCE.

### B8.Q7 — Avantage, erreur TD et GAE : ce que règle λ 🧠 ★★ ⏱️ 3 min
*Fiche §208.4 · parcours R*

a) $r_t = 1$, $\gamma = 0{,}9$, $V(s_t) = 3{,}0$, $V(s_{t+1}) = 2{,}5$, et l'épisode continue. Que vaut l'erreur TD $\delta_t$, à 2 décimales ?
b) Avec $\lambda = 0$, l'avantage GAE $\hat A_t$ vaut : (A) l'erreur TD $\delta_t$ ; (B) $G_t - V(s_t)$ ; (C) 0.
c) Lequel de ces réglages donne l'estimation de plus forte variance ? (A) $\lambda = 0{,}95$ ; (B) $\lambda = 0$ ; (C) $\lambda = 1$.
d) Avec $\gamma = 0{,}99$ et $\lambda = 0{,}95$, quel poids porte l'erreur TD $\delta_{t+10}$ dans $\hat A_t$, à 2 décimales (on suppose que l'épisode continue) ?
e) Vrai ou faux : avec $\lambda = 1$, l'avantage GAE ne dépend plus du tout du critique.
f) Vrai ou faux : pour un pas tronqué, $\delta_t$ amorce sur la valeur de la vraie dernière observation, mais $\hat A_t$ n'ajoute aucune erreur TD des pas suivants du rollout.

### B8.Q8 — PPO : à quoi sert le clip, et pourquoi réutiliser plusieurs fois le même lot ? 🧠 ⏱️ 3 min
*Fiche §208.5 · parcours R*

a) PPO fait plusieurs epochs sur le même rollout parce que : (A) le clip rend l'estimation exacte quelle que soit la distance entre $\pi_\theta$ et $\pi_{\text{old}}$ ; (B) chaque transition coûte une interaction, et le ratio corrige l'écart entre les deux politiques tant qu'elles restent proches ; (C) un gradient de politique est off-policy.
b) Vrai ou faux : pour un avantage négatif, le clip empêche aussi de faire redescendre un ratio monté à 1,5.
c) Un rollout de $N = 8$ environnements et $T = 256$ pas, 4 epochs, des mini-batches de 256 transitions : combien de pas d'Adam par itération ?
d) $\log \pi_\theta(a \mid s) = -0{,}9$ et $\log \pi_{\text{old}}(a \mid s) = -1{,}2$. Que vaut le ratio, à 2 décimales ?
e) Vrai ou faux : TRPO impose sa contrainte de KL par un simple terme ajouté à la loss, qu'Adam optimise directement.
f) Vrai ou faux : une transition d'avantage nul ne contribue pas au gradient de l'objectif clippé, quel que soit son ratio.

### B8.Q9 — On-policy ou off-policy : DQN, REINFORCE et PPO 🧠 ⏱️ 3 min
*Fiche §208.2, §208.3, §208.5 · parcours R*

a) Laquelle de ces méthodes apprend, sans aucune correction, sur des transitions jouées par d'anciennes versions de sa politique ? (A) REINFORCE ; (B) DQN ; (C) PPO.
b) Vrai ou faux : PPO, qui réutilise un rollout pendant plusieurs epochs, est une méthode off-policy, comme DQN.
c) Vrai ou faux : REINFORCE n'utilise plus un épisode une fois qu'il a servi à un pas de gradient.
d) Pourquoi DQN peut-il apprendre la valeur de la politique gloutonne à partir de transitions jouées en ε-greedy ? (A) sa cible prend le max sur $a'$, quelle que soit l'action que jouera ensuite la politique qui explore ; (B) le réseau cible corrige chaque transition par un ratio de probabilités ; (C) ε finit par décroître jusqu'à 0.
e) Sur un robot réel, chaque épisode prend une heure. Quel choix de départ est le plus raisonnable ? (A) PPO, robuste et simple à régler ; (B) REINFORCE, dont l'estimation est sans biais ; (C) une méthode off-policy, qui réutilise chaque transition de nombreuses fois.

### B8.Q10 — Évaluer un agent : graines, épisodes, politique déterministe 🧠 ⏱️ 3 min
*Fiche §208.6 · parcours R*

a) Une politique parfaite pour CartPole-v1 est évaluée par `evaluate_policy(env, policy, n_episodes=100, max_steps=300, rng=rng)`. Quel retour moyen affiche-t-elle ?
b) 100 épisodes d'évaluation donnent un retour moyen de 412 et un écart-type de 40. Que vaut l'erreur type de la moyenne ?
c) Vrai ou faux : après `env.reset(seed=3)`, un second `env.reset()` sans graine redonne le même état initial.
d) Pour comparer DQN et Double DQN sur CartPole : (A) une graine chacun, mais 1 000 épisodes d'évaluation ; (B) au moins trois graines chacun, et les mêmes graines d'évaluation pour les deux ; (C) la meilleure de cinq graines pour chacun.
e) Vrai ou faux : pour qu'une politique aléatoire qui tire ses actions par `env.action_space.sample()` soit reproductible, `env.reset(seed=...)` ne suffit pas.
f) Pour rapporter ce que vaut un agent PPO, on l'évalue avec : (A) l'action de plus grande probabilité ; (B) une action tirée avec une température de 2 ; (C) l'action qui maximise la valeur $V$ du critique.

### B8.Q11 — RLHF, DPO, GRPO : qui a besoin d'un modèle de récompense ou d'un critique ? 🧠 ⏱️ 3 min
*Fiche §208.7 · parcours R*

a) Lesquels entraînent un modèle de récompense séparé ? (A) le RLHF d'InstructGPT ; (B) DPO ; (C) GRPO avec des récompenses vérifiables (réponse juste ou fausse). (une ou plusieurs lettres)
b) Lesquels tirent de nouvelles réponses de la politique pendant l'entraînement ? (A) le RLHF avec PPO ; (B) DPO ; (C) GRPO. (une ou plusieurs lettres)
c) GRPO, un groupe de 4 réponses de récompenses $(1 ;\ 0 ;\ 0 ;\ 0)$ : quel est l'avantage de la première réponse, avec l'écart-type de population (divisé par le nombre de réponses du groupe), à 2 décimales ?
d) Vrai ou faux : dans DPO, $\pi_{\text{ref}}$ reste figée pendant tout l'entraînement.
e) GRPO se passe de critique parce que : (A) les récompenses vérifiables n'ont aucune variance ; (B) un critique serait un second modèle de la taille de la politique ; (C) PPO interdit un critique quand les actions sont des tokens.
f) Vrai ou faux : si les $G$ réponses d'un groupe GRPO ont toutes la même récompense, cette question ne fait pas bouger la politique.

### B8.Q12 — Piratage de la récompense (reward hacking) : quand l'agent optimise la lettre et pas l'esprit 🧠 ⏱️ 3 min
*Fiche §208.7, §208.8 · parcours R*

a) Lesquels sont des piratages de la récompense ? (A) un agent qui atteint le seuil de 475 sur CartPole ; (B) un assistant noté par un modèle de récompense qui allonge toutes ses réponses, parce que ce modèle préfère les réponses longues ; (C) un agent qui joue presque au hasard au début de son entraînement ; (D) un robot de ménage récompensé quand sa caméra ne voit plus de désordre, qui pousse le désordre hors du champ. (une ou plusieurs lettres)
b) Gao et coll. (2022) optimisent une politique contre un modèle de récompense appris, en mesurant aussi une récompense « vraie ». À mesure que la KL entre la politique et la référence grandit, la vraie récompense : (A) monte toujours, moins vite que la récompense apprise ; (B) monte, plafonne, puis redescend ; (C) baisse dès le début.
c) Vrai ou faux : augmenter $\beta$, la force de la pénalité KL du RLHF, laisse la politique s'éloigner davantage de la référence.
d) Quelle parade attaque le mieux la cause du piratage ? (A) entraîner plus longtemps contre le même modèle de récompense ; (B) retirer la pénalité KL, pour que la politique explore davantage ; (C) surveiller une mesure qu'on ne récompense pas, par exemple des évaluations humaines sur un échantillon de réponses.
e) Vrai ou faux : si les annotateurs préfèrent, une fois sur cinq, une réponse convaincante mais fausse à une réponse correcte, le modèle de récompense peut apprendre à favoriser les réponses convaincantes.

<a id="rappels"></a>

## 🔁 Rappels

### B8.R1 — Ch. 26 : la mise à jour du Q-learning et le rôle de la mémoire de rejeu 🔁 ★ ⏱️ 5 min
*Ch. 26 (§26.6.1, §26.9, 🔨 26.31) · fiche §208.2 · parcours R*

Une case de la table vaut $Q(s, a) = 0{,}5$. On joue $a$, on reçoit $r = 0{,}3$, et la ligne de $s'$ a pour maximum 0,6. $\gamma = 0{,}8$, $\alpha = 0{,}2$.

a) Que vaut la cible $y$, à 2 décimales ?
b) Que devient $Q(s, a)$ après la mise à jour, à 3 décimales ?
c) Même transition, mais $s'$ est terminal : que devient $Q(s, a)$, à 2 décimales ?
d) La classe `ReplayBuffer` de `rl.py` garde : (A) des épisodes entiers, qu'on rejoue depuis leur état de départ ; (B) les dernières transitions $(s, a, r, s', \text{terminated})$, dans lesquelles on tire des mini-batches au hasard ; (C) les valeurs Q des états visités, pour ne pas les recalculer.
e) Vrai ou faux : une fois pleine, la mémoire remplace la transition la plus ancienne par la nouvelle.
f) Vrai ou faux : SARSA peut, sans rien changer, apprendre sur des mini-batches tirés d'une mémoire remplie par d'anciennes versions de sa politique.

### B8.R2 — Ch. 19 : Adam et le gradient clipping 🔁 ★ ⏱️ 5 min
*Ch. 19 (§19.5.5, « le gradient clipping ») · fiche §208.5 · parcours R*

a) Un poids vaut $w = 1{,}0$ ; son premier gradient vaut $g = -40$. Adam ($\eta = 0{,}001$, $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\epsilon = 10^{-8}$, correction de biais comprise) : que vaut $w$ après le premier pas, à 4 décimales ?
b) Les gradients de deux tenseurs de paramètres valent $(3 ;\ 4)$ et $(12)$. Après `clip_grad_norm_(..., max_norm=0.5)` (la norme globale, avec le $10^{-6}$ de PyTorch), que vaut le premier élément, à 3 décimales ?
c) Vrai ou faux : `clip_grad_norm_` s'appelle entre `optimizer.step()` et `optimizer.zero_grad()`.
d) CleanRL règle `eps=1e-5` dans l'Adam de PPO. Cet $\epsilon$ sert à : (A) éviter un pas démesuré quand $\sqrt{\hat v}$ est minuscule ; (B) fixer le plus petit learning rate permis ; (C) borner la norme du gradient.
e) Vrai ou faux : le clipping par la norme globale garde la direction du gradient de l'ensemble des paramètres.

### B8.R3 — Ch. 11 : exploration, softmax et bandits 🔁 ★ ⏱️ 5 min
*Ch. 11 (« Au-delà du livre : le bandit manchot ») · ch. 26 (§26.6.2) · fiche §208.3 · parcours R*

a) Une politique ε-greedy avec $\varepsilon = 0{,}2$, sur les deux actions de CartPole : avec quelle probabilité joue-t-elle l'action gloutonne, à 2 décimales ?
b) Une politique softmax de température $\tau = 0{,}5$ sur deux actions de valeurs $Q = (1{,}0 ;\ 1{,}5)$ : probabilité de l'action 1, à 3 décimales ?
c) Un bras a reçu les récompenses 2, 4 et 6 ($Q = 4$) ; il reçoit 8. Que devient $Q$ avec la moyenne incrémentale ?
d) Vrai ou faux : une politique softmax de très grande température se rapproche de la politique gloutonne.
e) Le « gradient bandit » de Sutton et Barto (§2.8), qui monte la préférence du bras tiré quand sa récompense dépasse la moyenne des récompenses, est : (A) ε-greedy avec un ε qui décroît ; (B) REINFORCE sur un seul état, avec une baseline ; (C) UCB.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde au moins 4 décimales pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex B8.1 — Cible DQN à la main : état terminal, puis cible Double DQN ✏️ ★ ⏱️ 15 min
**Objectif :** calculer les cibles de DQN et de Double DQN sur un mini-batch, en traitant une vraie fin d'épisode et une limite de temps.
**Prérequis :** ch. 26 (cible du Q-learning, `terminated` et `truncated`) · fiche §208.2 (réseau cible, « la limite de temps n'est pas une fin », Double DQN) · **Parcours :** R, M

Un mini-batch de trois transitions de CartPole, $\gamma = 0{,}99$, une récompense $r = 1$ pour chacune. Les deux actions sont numérotées 0 et 1.

| Transition | Fin | $Q_{\bar\theta}(s', \cdot)$ (cible) | $Q_\theta(s', \cdot)$ (en ligne) | $Q_\theta(s, a)$ de l'action jouée |
|---|---|---|---|---|
| 1 | aucune | $(12{,}0 ;\ 13{,}7)$ | $(12{,}6 ;\ 12{,}2)$ | 15,0 |
| 2 | `terminated` (le mât tombe) | $(4{,}0 ;\ 3{,}1)$ | $(2{,}5 ;\ 3{,}8)$ | 2,9 |
| 3 | `truncated` (500 pas atteints) | $(40{,}2 ;\ 41{,}6)$ | $(42{,}5 ;\ 41{,}7)$ | 45,3 |

a) La cible DQN de la transition 1, à 2 décimales.
b) La cible DQN de la transition 2, à 2 décimales.
c) La cible DQN de la transition 3, à 2 décimales.
d) La cible Double DQN de la transition 1, à 2 décimales.
e) La cible Double DQN de la transition 3, à 2 décimales.
f) De combien la cible de DQN dépasse-t-elle celle de Double DQN, en moyenne sur les trois transitions, à 2 décimales ?
g) Vrai ou faux : pour la transition 1, le réseau en ligne et le réseau cible s'accordent sur la meilleure action de $s'$.
h) L'écart $x_3 = Q_\theta(s_3, a_3) - y_3$ entre la prédiction et la cible DQN de la transition 3, à 2 décimales.

Dans ta copie : que deviendrait $y_3$ si la mémoire avait rangé `done = terminated or truncated` à la place de `terminated`, et que l'agent en apprendrait-il sur les états proches de la limite de temps ? Pourquoi la cible de la transition 2 ne dépend-elle d'aucun des deux réseaux ? Écris la ligne PyTorch qui calcule les trois cibles Double DQN d'un coup (`argmax`, `gather`), sous `torch.no_grad()`.

### Ex B8.2 — Loss de Huber et son gradient pour trois erreurs TD ✏️ ★ ⏱️ 10 min
**Objectif :** calculer la loss de Huber et son gradient, et voir ce qui les distingue de l'erreur quadratique et de `smooth_l1_loss`.
**Prérequis :** ch. 5 (dérivées par morceaux) · ch. 9 (loss de Huber) · fiche §208.2 (rappel maths « la loss de Huber ») · **Parcours :** M

Trois écarts $x = Q_\theta(s, a) - y$ d'un mini-batch : $x = (0{,}6 ;\ -2{,}4 ;\ 3{,}2)$. Seuil $\delta = 1{,}5$. « La loss du mini-batch » est la moyenne des trois termes, comme `F.huber_loss(pred, y, delta=1.5)` avec sa réduction par défaut.

a) $\ell_{1{,}5}(0{,}6)$, à 2 décimales.
b) $\ell_{1{,}5}(-2{,}4)$, à 3 décimales.
c) La loss du mini-batch, à 2 décimales.
d) La dérivée $\ell'_{1{,}5}(-2{,}4)$, à 1 décimale.
e) Le gradient de la loss du mini-batch par rapport à $Q_\theta(s_3, a_3)$ (dont l'écart vaut $x_3 = 3{,}2$), à 2 décimales.
f) Le même gradient si la loss du mini-batch était la moyenne des $\frac{1}{2}x^2$, à 3 décimales.
g) La valeur de `F.smooth_l1_loss(pred, y, beta=1.5)` sur le même mini-batch, à 3 décimales.
h) Vrai ou faux : pour l'écart $0{,}6$, la loss de Huber et $\frac{1}{2}x^2$ ont le même gradient.

Dans ta copie : pourquoi un gradient borné protège-t-il l'apprentissage quand une cible amorcée est très fausse ? Comparer e) et f) : de combien un seul écart aberrant pèse-t-il dans chacun des deux gradients si $x_3$ passe de 3,2 à 32 ? Quelle valeur de `beta` donne à `smooth_l1_loss` exactement la loss de Huber de seuil $\delta$, et pour quel seul $\delta$ ?

### Ex B8.3 — Gradient du log d'une politique softmax à trois actions ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer à la main le gradient du log d'une politique softmax, faire un pas de REINFORCE et voir son effet sur la politique.
**Prérequis :** ch. 17 (softmax, log-softmax) · ch. 18 (gradient de la cross-entropy par rapport aux logits) · ch. 6 (entropie) · fiche §208.3 · **Parcours :** M

Une politique softmax à trois actions a les logits $\mathbf{z} = (0{,}5 ;\ 1{,}5 ;\ -0{,}5)$ dans l'état $s$. L'agent a joué l'action 2 et l'avantage estimé de ce choix vaut $\hat A = -2$. On fait un pas de montée de gradient sur $\hat A \log \pi(2)$, directement sur les logits, avec un pas de 0,1 : $\mathbf{z} \leftarrow \mathbf{z} + 0{,}1\,\hat A\,\nabla_{\mathbf{z}} \log \pi(2)$.

a) $\pi(2)$, à 4 décimales.
b) $\pi(1)$, à 4 décimales.
c) La composante $\partial \log \pi(2) / \partial z_0$, à 4 décimales.
d) La composante $\partial \log \pi(2) / \partial z_2$, à 4 décimales.
e) Vrai ou faux : la somme des trois composantes de $\nabla_{\mathbf{z}} \log \pi(2)$ vaut 0.
f) Le logit $z_2$ après le pas, à 3 décimales.
g) $\pi(2)$ après le pas, à 4 décimales.
h) L'entropie de la politique **avant** le pas, en nats, à 3 décimales.

Dans ta copie : pourquoi la somme de e) vaut-elle ce qu'elle vaut, quelle que soit l'action jouée ? Comment ce gradient se relie-t-il à celui de la cross-entropy du ch. 18 ? Les logits $z_0$ et $z_1$ montent-ils ou baissent-ils, et lequel bouge le plus ? Si l'on ajoutait 10 aux trois logits, que deviendraient la politique et le gradient ?

### Ex B8.4 — GAE à la main sur quatre pas, pour λ = 0, 0,95 et 1 ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer à rebours les avantages GAE d'un rollout qui contient une troncature, pour trois valeurs de $\lambda$, et chiffrer l'erreur qui traite la limite de temps comme une fin.
**Prérequis :** ch. 26 (erreur TD, retours actualisés) · fiche §208.4 (GAE, « fin d'épisode ou limite de temps ») · **Parcours :** R, M

Un rollout de $T = 4$ pas dans un seul environnement, $\gamma = 0{,}9$. Les pas 0 et 1 sont les deux derniers d'un épisode **tronqué** au pas 1 (la limite de temps) ; les pas 2 et 3 sont les deux premiers de l'épisode suivant, que le rollout coupe après le pas 3. `next_values[t]` est la valeur de l'observation qui **suit** le pas $t$ : au pas 1, celle de la vraie dernière observation de l'épisode tronqué. Aucun pas n'est `terminated`.

| Pas $t$ | $r_t$ | $V(s_t)$ | `next_values[t]` | `truncated` |
|---|---|---|---|---|
| 0 | 1 | 4,0 | 3,8 | non |
| 1 | 1 | 3,8 | 3,6 | **oui** |
| 2 | 1 | 2,0 | 2,4 | non |
| 3 | 1 | 2,4 | 2,6 | non |

a) L'erreur TD $\delta_1$, à 2 décimales.
b) L'erreur TD $\delta_3$, à 2 décimales.
c) $\lambda = 0$ : l'avantage $\hat A_2$, à 2 décimales.
d) $\lambda = 0{,}95$ : l'avantage $\hat A_2$, à 4 décimales.
e) $\lambda = 0{,}95$ : l'avantage $\hat A_0$, à 4 décimales.
f) $\lambda = 0{,}95$ : la cible du critique au pas 0 (la deuxième sortie de `compute_gae`), à 4 décimales.
g) $\lambda = 1$ : l'avantage $\hat A_0$, à 3 décimales.
h) $\lambda = 1$ : l'avantage $\hat A_2$, à 3 décimales.
i) $\lambda = 0{,}95$, si l'on traitait la troncature du pas 1 comme une vraie fin (aucun amorçage) : l'avantage $\hat A_0$, à 3 décimales.
j) Vrai ou faux : avec $\lambda = 1$, $\hat A_0 + V(s_0)$ vaut $r_0 + \gamma r_1 + \gamma^2 \times 3{,}6$.

Dans ta copie : pourquoi $\hat A_1$ est-il le même pour les trois valeurs de $\lambda$ ? Pourquoi $\hat A_3$ amorce-t-il sur 2,6 alors que le rollout s'arrête ? Comment varient $\hat A_2$ quand $\lambda$ passe de 0 à 0,95 à 1, et pourquoi ? Que fait l'erreur de i) à la politique, quand les meilleurs épisodes sont justement ceux qui atteignent la limite de temps ?

### Ex B8.5 — Objectif clippé de PPO : quatre cas, valeur et gradient nul ou non ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer l'objectif clippé de PPO et son gradient par rapport aux ratios dans les quatre cas (signe de l'avantage, ratio sous ou au-dessus de l'intervalle), et distinguer fraction clippée et gradient nul.
**Prérequis :** Ex B8.3 · fiche §208.5 (« l'objectif clippé ») · figure 1 de Schulman et coll. (2017), reprise par la fiche · **Parcours :** R, M

Quatre transitions, $\varepsilon = 0{,}2$ : ratios $\rho = (1{,}4 ;\ 0{,}6 ;\ 0{,}75 ;\ 1{,}35)$, avantages $\hat A = (2{,}0 ;\ 1{,}5 ;\ -1{,}2 ;\ -0{,}8)$. La loss est $-\frac{1}{4}\sum_i \min\big(\rho_i \hat A_i,\ \operatorname{clip}(\rho_i, 1 - \varepsilon, 1 + \varepsilon)\,\hat A_i\big)$, comme `clipped_surrogate` ; son gradient par rapport à $\rho_i$ vaut $-\hat A_i / 4$ quand le terme non clippé est le minimum, 0 quand le terme clippé est strictement plus petit.

a) Le terme $\min(\cdot)$ de la transition 0, à 2 décimales.
b) Le terme $\min(\cdot)$ de la transition 2, à 2 décimales.
c) Le terme $\min(\cdot)$ de la transition 3, à 2 décimales.
d) La loss, à 3 décimales.
e) Le gradient de la loss par rapport à $\rho_1$, à 3 décimales.
f) Le gradient de la loss par rapport à $\rho_3$, à 2 décimales.
g) Combien des quatre transitions ont un gradient nul ?
h) La fraction clippée (la part des transitions où $|\rho - 1| > \varepsilon$), à 2 décimales.
i) Avec $\varepsilon = 0{,}3$ et les mêmes nombres, combien des quatre transitions ont un gradient nul ?

Dans ta copie : place les quatre transitions sur les deux panneaux de la figure 1 de l'article. Pour chacune, dis si un pas de descente sur la loss augmente ou diminue son ratio, ou le laisse. Pourquoi la fraction clippée de h) ne dit-elle pas combien de transitions ont cessé de pousser ? Que deviendrait le gradient de la transition 3 si son ratio valait 1,15 ?

### Ex B8.6 — Préférences : loss de Bradley-Terry puis loss DPO sur une paire de réponses ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer la loss d'un modèle de récompense de Bradley-Terry et son gradient, puis la loss DPO de la même paire, et voir le rôle de $\beta$.
**Prérequis :** ch. 17 (sigmoïde) · ch. 6 (cross-entropy, divergence KL) · fiche §208.7 (Bradley-Terry, DPO) · **Parcours :** M

Un annotateur préfère la réponse $y_w$ à la réponse $y_l$ pour le même prompt $x$.

*Modèle de récompense.* Il donne $r_\phi(y_w) = 2{,}3$ et $r_\phi(y_l) = 1{,}1$.

a) La probabilité que le modèle donne au choix observé, $\sigma\big(r_\phi(y_w) - r_\phi(y_l)\big)$, à 3 décimales.
b) La loss $L_{\text{RM}}$ de cette paire, à 3 décimales.
c) Sa dérivée par rapport à $r_\phi(y_w)$, à 3 décimales.
d) La loss si l'annotateur avait préféré $y_l$, avec les mêmes récompenses, à 3 décimales.
e) Vrai ou faux : si le modèle ajoutait 10 aux deux récompenses, la loss de b) ne changerait pas.

*DPO.* Les log-probabilités des deux réponses entières valent $-35{,}2$ ($y_w$) et $-38{,}9$ ($y_l$) sous la politique, $-36{,}0$ et $-37{,}5$ sous la référence figée.

f) La récompense implicite de $y_w$, $\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}$, avec $\beta = 0{,}1$, à 2 décimales.
g) La marge de DPO avec $\beta = 0{,}1$, à 2 décimales.
h) La loss DPO avec $\beta = 0{,}1$, à 3 décimales.
i) La loss DPO avec $\beta = 0{,}5$, à 3 décimales.
j) La dérivée de la loss DPO ($\beta = 0{,}1$) par rapport à $\log \pi_\theta(y_w \mid x)$, à 4 décimales.

Dans ta copie : écris la loss de b) sous la forme qu'on calcule sans risque de $\log 0$, et dis ce que donnerait le calcul naïf pour un écart de $-1\,000$. Pourquoi d) dépasse-t-elle $\ln 2$, et b) non ? Que vaudrait la loss DPO si la politique était égale à la référence ? Comment j) change-t-il quand la politique s'éloigne encore de la référence dans le bon sens ?

### Ex B8.7 — Le gradient de politique en une étape : l'astuce de la log-dérivée, puis la baseline sans biais ∂ ★★★ ⏱️ 40 min
**Objectif :** démontrer le théorème du gradient de politique sous la forme de REINFORCE, puis que soustraire une baseline qui ne dépend pas de l'action ne change pas son espérance.
**Prérequis :** 0B (dérivée d'un logarithme, espérance) · ch. 11 (bandit, politique softmax) · fiche §208.3 (rappel maths « l'astuce de la log-dérivée ») · **Parcours :** M

On note $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_{T})$ un épisode de longueur $T$, de loi
$$p_\theta(\tau) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\,P(s_{t+1} \mid s_t, a_t),$$
où $\mu$ est la loi de l'état initial et $P$ la dynamique de l'environnement, qui ne dépendent pas de $\theta$. La récompense $r_t$ est une fonction de $(s_t, a_t, s_{t+1})$. $J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$, avec $R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t$. Les sommes sont finies (actions et états discrets) : on peut dériver sous le signe somme. On admet que $T$ est fixé.

1. **L'astuce.** Pour une loi discrète $p_\theta(x)$ et une fonction $f(x)$ qui ne dépend pas de $\theta$, montre que $\nabla_\theta \mathbb{E}_{p_\theta}[f] = \mathbb{E}_{p_\theta}[f \,\nabla_\theta \ln p_\theta]$. Où as-tu supposé $p_\theta(x) > 0$ ?
2. **L'environnement disparaît.** Montre que $\nabla_\theta \ln p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t \mid s_t)$. Déduis-en une première forme : $\nabla_\theta J = \mathbb{E}\big[R(\tau) \sum_t \nabla_\theta \ln \pi_\theta(a_t \mid s_t)\big]$.
3. **La fonction score est d'espérance nulle.** Montre que, pour tout état $s$, $\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}[\nabla_\theta \ln \pi_\theta(a \mid s)] = 0$. Vérifie-le pour une politique softmax à deux actions de logits $(z_0, z_1)$, en calculant les deux composantes.
4. **La causalité.** Soit $t' < t$. Montre que $\mathbb{E}\big[\gamma^{t'} r_{t'}\,\nabla_\theta \ln \pi_\theta(a_t \mid s_t)\big] = 0$ (conditionne sur le début de l'épisode jusqu'à $s_t$, puis utilise 3.). Déduis-en la forme de REINFORCE : $\nabla_\theta J = \mathbb{E}\big[\sum_t \nabla_\theta \ln \pi_\theta(a_t \mid s_t)\,\gamma^t G_t\big]$, avec $G_t = \sum_{k \ge t} \gamma^{k - t} r_k$. (La forme qu'on implémente, celle de la fiche, laisse tomber le facteur $\gamma^t$ : on le dit, et c'est un biais connu et accepté.)
5. **La baseline.** Soit $b(s)$ une fonction de l'état seul. Montre que $\mathbb{E}\big[\nabla_\theta \ln \pi_\theta(a_t \mid s_t)\,b(s_t)\big] = 0$, et conclus que remplacer $G_t$ par $G_t - b(s_t)$ ne change pas l'espérance du gradient. Pourquoi l'argument tombe-t-il si $b$ dépend de $a_t$ ?
6. **La variance, sur un bandit.** Un seul état, deux actions, politique softmax de logits $(0, z)$ en $z = 0$, donc $\pi = (\frac12 ;\ \frac12)$. L'action 0 rapporte toujours 2, l'action 1 toujours 4. On estime $\frac{\partial J}{\partial z}$ par un seul tirage : $g = (R - b)\,\frac{\partial}{\partial z}\ln \pi(a)$. Calcule la vraie valeur de $\frac{\partial J}{\partial z}$, puis l'espérance et la variance de $g$ en fonction de $b$. Pour quelle valeur de $b$ la variance est-elle minimale, et combien vaut-elle alors ? Et pour $b = 0$ ?

### Ex B8.8 — Fermi : combien d'interactions ? CartPole, Atari (DQN 2015), AlphaGo Zero et le temps CPU correspondant 🧮 ★★ ⏱️ 15 min
**Objectif :** chiffrer à la louche le nombre d'interactions et de passes de réseau qu'ont demandées trois succès du RL, et ce qu'elles coûteraient sur ton CPU.
**Prérequis :** ch. 26 (le DQN et AlphaGo Zero en quelques lignes) · fiche §208.6 (« ce que coûte un pas sur CPU », rappel maths « une estimation de Fermi en RL »), §208.8 · **Parcours :** R, M

Hypothèses (des ordres de grandeur, à prendre telles quelles) :
- **ton CPU** (mesures de la fiche) : un pas de CartPole vectorisé ≈ 20 µs par copie ; une passe avant d'un petit MLP sur un batch d'observations ≈ 50 µs ; un tirage par `Categorical` ≈ 200 µs ; un pas d'Adam de PPO sur un mini-batch de 128 ≈ 3 ms ; une mise à jour de DQN (mini-batch de 128) ≈ 1,7 ms ;
- **PPO de CleanRL sur CartPole** : 500 000 pas d'environnement, 4 copies, des rollouts de 128 pas, 4 epochs de 4 mini-batches par itération ;
- **DQN 2015** (V. Mnih et coll., *Nature*) : 49 jeux ; par jeu, « 50 millions d'images, environ 38 jours de jeu » ; l'agent ne voit qu'une image sur 4 de l'émulateur et répète son action sur les autres ; une mise à jour tous les 4 choix d'action, sur un mini-batch de 32 ; un jeu Atari tourne à 60 images par seconde ;
- **AlphaGo Zero** (D. Silver et coll., *Nature*, 2017, version de 3 jours) : 4,9 millions de parties contre lui-même, 1 600 simulations par coup (chacune évalue une fois le réseau), environ 0,4 s par coup ; on suppose 200 coups par partie ; sur ton CPU, une passe du réseau de 20 blocs résiduels sur une position prendrait environ 0,2 s ;
- un joueur humain très actif : 2 000 parties de go par an pendant 30 ans ; un humain joue 2 heures avant d'être testé sur un jeu Atari (le testeur de DQN 2015).

Réponds dans ta copie, avec des ordres de grandeur ; des fourchettes sont bienvenues.

1. **CartPole.** Combien d'itérations, de rollouts vectorisés (appels de `step` sur les 4 copies) et de pas d'Adam fait le PPO de CleanRL ? Estime le temps de collecte, le temps des mises à jour, puis le total. Combien de temps prendrait le budget de 100 000 pas du défi B8.28 ?
2. **Atari.** Combien d'images de l'émulateur font 38 jours à 60 images par seconde ? Qu'en déduis-tu sur ce que comptent les « 50 millions d'images » de l'article ? Combien de mises à jour et combien de transitions tirées dans la mémoire, par jeu, puis pour les 49 jeux ? Compare aux 2 heures du testeur humain.
3. **AlphaGo Zero.** Combien de positions jouées, et combien de passes du réseau pour les 4,9 millions de parties ? À 0,4 s par coup, combien de parties devaient tourner en même temps pour tenir en 3 jours ? Combien de temps prendraient toutes ces passes sur ton CPU ? Compare au nombre de parties d'une vie de joueur.
4. En deux phrases : qu'est-ce que ces trois chiffres disent de l'efficacité en échantillons du RL, et pourquoi ce n'est un problème que hors d'un simulateur ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B8.9 — PPO expliqué en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer PPO à quelqu'un qui connaît l'apprentissage supervisé, mais pas le RL.
**Prérequis :** fiche §208.3 à §208.5 · **Parcours :** R

Un collègue data scientist, qui entraîne des classifieurs tous les jours mais n'a jamais fait de RL, te demande ce qu'est « ce PPO dont tout le monde parle pour les modèles de langue ». Explique-le en cinq lignes au plus, sans formule et sans les mots « ratio », « surrogate » ni « trust region ». Ton explication doit faire apparaître (1) ce que l'agent apprend, et à partir de quoi ; (2) pourquoi on ne peut pas simplement faire de la descente de gradient sur des labels ; (3) l'idée du « petit pas » et ce que fait le clip, en une image ; (4) pourquoi on réutilise plusieurs fois les mêmes données ; (5) une limite.

### Ex B8.10 — RLHF : qui choisit les préférences ? Annotateurs, valeurs et flagornerie des modèles ⚖️ ★★ ⏱️ 25 min
**Objectif :** analyser d'où viennent les préférences qui alignent un modèle de langue, ce qu'elles y mettent, et comment le documenter.
**Prérequis :** B4 (modèles instruct, fine-tuning) · fiche §208.7 (RLHF, InstructGPT, piratage de la récompense, flagornerie) · **Parcours :** R

Ouyang et coll. (2022, [arXiv:2203.02155](https://arxiv.org/abs/2203.02155), section 5.2 « Who are we aligning to? » et annexe B) décrivent les préférences qui ont servi à InstructGPT : environ 40 prestataires recrutés par Upwork et Scale AI, surtout anglophones, vivant aux États-Unis ou en Asie du Sud-Est, choisis sur un test de sélection ; un accord entre annotateurs d'environ 73 % ; des consignes écrites par les chercheurs, qui demandaient de privilégier l'utilité pendant l'entraînement, et la véracité et l'innocuité lors des évaluations finales ; des prompts venus des clients de l'API, eux-mêmes sélectionnés sur une liste d'attente. Ton entreprise veut aligner un assistant interne, en français, pour répondre aux questions des salariés sur les ressources humaines (congés, mutuelle, procédures disciplinaires). (Cas inventé pour l'exercice.) Réponds dans ta copie.

1. **Les sources de préférences.** Cite les quatre groupes dont InstructGPT, selon ses auteurs, reflète les préférences. Pour ton assistant, qui seraient ces quatre groupes, et lequel manquerait si tu suivais la même méthode ?
2. **Le désaccord.** Que veut dire un accord de 73 % ? Que devient, dans un modèle de récompense de Bradley-Terry, une paire sur laquelle deux annotateurs se contredisent ? Donne deux sujets RH sur lesquels des salariés et la direction pourraient préférer des réponses différentes.
3. **La flagornerie.** Explique, avec la chaîne préférences → modèle de récompense → PPO, comment un assistant peut apprendre à dire aux salariés ce qu'ils veulent entendre (« oui, vous avez droit à ce congé »). Pourquoi la pénalité KL ne suffit-elle pas à l'empêcher ?
4. **La mesure.** Propose une mesure **non récompensée** à surveiller pour détecter ce piratage, et la façon de la collecter sans biais.
5. **Les annotateurs.** Quelles conditions de travail, de rémunération et de consignes faudrait-il documenter pour les personnes qui classeront les réponses, et pourquoi cela change-t-il les préférences recueillies ?
6. **DPO ou RLHF ?** L'équipe propose DPO « pour aller plus vite ». Qu'est-ce que ce choix change, et ne change pas, aux questions 1 à 4 ?
7. Rédige les cinq lignes « Données de préférences » de la fiche modèle de l'assistant : qui, combien, quelles consignes, quel accord, quelles limites connues.

### Ex B8.11 — Schulman et coll. (2017) : « Proximal Policy Optimization Algorithms » 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a introduit PPO, comprendre ses deux variantes et ses expériences, et relier chaque choix aux implémentations d'aujourd'hui.
**Prérequis :** Ex B8.5 · fiche §208.4, §208.5 · **Parcours :** M

Lis J. Schulman, F. Wolski, P. Dhariwal, A. Radford et O. Klimov, « Proximal Policy Optimization Algorithms » ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347), version 2 du 28 août 2017, douze pages) : le résumé, l'introduction, les sections 2 à 5 avec les figures 1 et 2, puis la section 6 et les tableaux de l'annexe A. Réponds dans ta copie.

1. **Le problème** (résumé, section 1). Quels défauts les auteurs reprochent-ils au Q-learning avec approximation, aux gradients de politique « vanilla » et à TRPO ? Que veut dire « multiple epochs of minibatch updates », et pourquoi est-ce nouveau ?
2. **La section 2.** Écris l'équation 2 ($L^{PG}$) et l'équation 3 (l'objectif de TRPO). Que disent les auteurs de plusieurs pas d'optimisation sur $L^{PG}$ avec la même trajectoire ? Pourquoi TRPO utilise-t-il une contrainte plutôt qu'une pénalité (équation 5) ?
3. **L'objectif clippé** (section 3). Écris l'équation 7. Que vaut $\varepsilon$ dans l'article ? Pourquoi les auteurs parlent-ils d'une borne « pessimiste » ? Explique la phrase « we only ignore the change in probability ratio when it would make the objective improve ». Que valent $L^{\text{CLIP}}$ et $L^{CPI}$ « au premier ordre » autour de $\theta_{\text{old}}$ ?
4. **Les figures 1 et 2.** Que trace la figure 1, et que montre le point rouge ? Place-y les quatre transitions de ✏️ B8.5. Que trace la figure 2, sur quel environnement, et quelle KL a la politique mise à jour au maximum de $L^{\text{CLIP}}$ ?
5. **La pénalité KL adaptative** (section 4). Donne la règle de mise à jour de $\beta$ (les facteurs 1,5 et 2). Que concluent les auteurs de cette variante ?
6. **L'algorithme** (section 5). Écris l'équation 9 et dis ce que sont $c_1$, $c_2$, $L^{VF}$ et $S$. Que devient l'équation 11 quand $\lambda = 1$ ? Décris l'algorithme 1 avec $N$, $T$, $K$ et $M$. Compare aux réglages de CartPole de la fiche (208.6).
7. **Le tableau 1** (section 6.1). Sur quels environnements, avec combien de pas et de graines ? Comment les scores sont-ils normalisés ? Quelle variante gagne, avec quel score, et pourquoi le score sans clip ni pénalité est-il négatif ? Les écarts entre $\varepsilon = 0{,}1$, 0,2 et 0,3 sont-ils solides, avec 21 runs (relis 208.6) ?
8. **Atari** (section 6.4 et tableau 5). Quels réglages changent par rapport à MuJoCo (horizon, epochs, mini-batch, acteurs, $\varepsilon$, coefficients) ? Que montre le tableau 2, et pourquoi les deux mesures ne donnent-elles pas le même vainqueur ? Combien de pas d'environnement font « 40M game frames » ?
9. **Et aujourd'hui.** Cite trois détails des implémentations actuelles (CleanRL, 208.5 et 208.6) que l'article ne décrit pas, et un réglage de l'article qu'elles ont gardé tel quel. Pourquoi PPO est-il revenu avec les modèles de langue (208.7) ?

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### B8.E1 — Méthodes par valeur ou par gradient de politique : avantages et inconvénients 💼 ★★ ⏱️ 10 min
*Fiche §208.2, §208.3 · parcours R*

« Quand choisiriez-vous une méthode par valeur, comme DQN, plutôt qu'une méthode par gradient de politique, comme PPO ? Donnez les avantages et les inconvénients de chacune. »

### B8.E2 — Explique PPO et le rôle du clipping 💼 ★★ ⏱️ 10 min
*Fiche §208.5 · parcours R*

« Expliquez-moi PPO. Que fait exactement le clipping, et que se passerait-il sans lui ? »

### B8.E3 — Ton agent RL n'apprend pas : comment le débogues-tu ? 💼 ★★ ⏱️ 10 min
*Fiche §208.6 · parcours R*

« Votre agent de RL tourne depuis une heure et son retour ne monte pas. Aucune erreur. Que faites-vous, dans quel ordre ? »

### B8.E4 — PPO avec un modèle de récompense ou DPO pour aligner un modèle de langue : que choisis-tu ? 💼 ★★ ⏱️ 10 min
*Fiche §208.7 · parcours R*

« Vous devez aligner un modèle de langue de 7 milliards de paramètres sur les préférences de vos utilisateurs. RLHF avec PPO et un modèle de récompense, ou DPO ? Qu'est-ce qui guide votre choix ? »

### B8.E5 — Pourquoi faut-il plusieurs graines pour comparer deux algorithmes de RL ? 💼 ★★ ⏱️ 10 min
*Fiche §208.6 · parcours R*

« Un collègue annonce que sa variante de PPO bat la version standard : sa courbe est au-dessus. Que lui demandez-vous avant d'y croire, et comment rapporteriez-vous la comparaison ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`, en trois phases, de difficulté croissante dans chacune : **la valeur** (B8.12 à B8.15 : CartPole vu par un réseau, ton DQN, ses quatre bugs et ses ablations), **la politique** (B8.16 à B8.24 : REINFORCE, GAE, l'objectif clippé, ton PPO sur des environnements vectorisés, ses journaux, ses hyperparamètres et sa robustesse), **les préférences** (B8.25 à B8.27 : les losses de préférences, un RLHF miniature sur CartPole et une revue de code face à CleanRL), puis le défi B8.28. Les agents s'écrivent en PyTorch dans le notebook ; GAE, l'objectif clippé et les losses de préférences vont dans ta librairie (`mylearn/deep_rl.py`), et `rl.py` du ch. 26 sert tel quel.

| ID | Titre | Type | ★ | ⏱️ | 🚀 |
|---|---|---|---|---|---|
| B8.12 | CartPole pour le deep RL : espaces, politique aléatoire, evaluate_policy et la référence tabulaire du ch. 26 | 📦 | ★ | 15 | |
| B8.13 | DQN sur CartPole : réseau Q, ReplayBuffer de rl.py, réseau cible, Huber, ε décroissant | 📦 | ★★★ | 60 | |
| B8.14 | Le DQN qui diverge : quatre erreurs de cible, de fin d'épisode et d'exploration | 🐛 | ★★ | 30 | |
| B8.15 | Ablations du DQN : sans réseau cible, sans rejeu, avec Double DQN — trois graines | 🔬 | ★★★ | 50 | 🚀 |
| B8.16 | Prédire : REINFORCE sans baseline, avec baseline, avec retours normalisés — lequel apprend le plus vite ? | 🔮 | ★ | 10 | |
| B8.17 | REINFORCE avec et sans baseline : variance du gradient et vitesse d'apprentissage | 📦 | ★★★ | 45 | |
| B8.18 | compute_gae et explained_variance | 🔨 | ★★ | 30 | |
| B8.19 | clipped_surrogate : valeur et gradient par rapport au ratio | 🔨 | ★★ | 20 | |
| B8.20 | PPO from scratch sur CartPole : environnements vectorisés, acteur-critique, GAE, clip, entropie | 📦 | ★★★★ | 150 | |
| B8.21 | Lire les journaux de PPO : KL approchée, fraction clippée, entropie, variance expliquée | 📈 | ★★ | 20 | |
| B8.22 | Hyperparamètres de PPO : ε du clip, λ, nombre d'epochs — robustesse sur trois graines | 🔬 | ★★★ | 45 | 🚀 |
| B8.23 | Prédire : l'agent PPO face à une gravité doublée ou à un mât deux fois plus long | 🔮 | ★ | 10 | |
| B8.24 | Robustesse : modifier la physique de CartPole et mesurer la chute du retour | 🔬 | ★★ | 25 | |
| B8.25 | bradley_terry_loss et dpo_loss | 🔨 | ★★ | 20 | |
| B8.26 | RLHF miniature : un modèle de récompense appris sur des préférences simulées, puis PPO — et le piratage de la récompense | 📦 | ★★★★ | 90 | |
| B8.27 | Revue de code : comparer ton PPO à une implémentation de référence et documenter chaque écart | 🛠️ | ★★ | 30 | |
| B8.28 | Défi : CartPole-v1 résolu (≥ 475 sur 100 épisodes) en au plus 100 000 pas, trois graines sur trois | 🏆 | ★★★★ | 120 | |

Tout tourne sur CPU. Les ablations sur trois graines (🔬 B8.15 et B8.22) passent par des résultats précalculés en mode rapide, et se relancent en entier en mode complet, plutôt sur Colab (🚀).
