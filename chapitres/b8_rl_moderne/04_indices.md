# B8 · RL moderne : DQN, gradient de politique, PPO et RLHF — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B8.Q1 — Pourquoi une table Q ne suffit plus pour CartPole

<details><summary>Indice 1</summary>

Relis « Pourquoi la table ne suffit plus » et « Apprendre une valeur par régression » (§208.1).

</details>
<details><summary>Indice 2</summary>

Une table a une ligne par combinaison d'intervalles et une colonne par action. Un réseau Q reçoit l'état et rend un vecteur, une valeur par action ; ses poids sont partagés par tous les états. La cible se traite comme une constante pendant le calcul du gradient : c'est le semi-gradient. Une passe sur un batch garde l'axe des exemples en tête de la forme.

</details>
<details><summary>Indice 3</summary>

a) $6^4 = 1\,296$ états discrétisés, deux actions chacun : **2 592** cases. Pour b), que rend une passe avant : une valeur, ou toutes les valeurs de l'état ? Pour c), un pas de gradient modifie des poids : qui d'autre s'en sert ? Pour d), écris le gradient de $(Q_\theta(s, a) - y_\theta)^2$ quand $y$ dépend de $\theta$ : a-t-il un terme de plus ? Pour e), 32 exemples en entrée, combien de nombres en sortie par exemple ?

</details>

### B8.Q2 — La triade mortelle : approximation, amorçage et apprentissage off-policy

<details><summary>Indice 1</summary>

Relis « Instabilité : la triade mortelle » (§208.1), puis le ⚠️ du ch. 26 sur SARSA (on-policy) et le Q-learning (off-policy).

</details>
<details><summary>Indice 2</summary>

Pour chaque méthode, pose trois questions : la valeur est-elle donnée par une fonction paramétrée (un réseau) plutôt qu'une table ? La cible contient-elle une estimation (amorçage) ou seulement des récompenses observées ? Les transitions viennent-elles d'une autre politique que celle dont on apprend la valeur ? Un réseau cible et une mémoire de rejeu changent la façon d'apprendre, pas la liste des ingrédients.

</details>
<details><summary>Indice 3</summary>

a) Les trois ingrédients nommés par Sutton et Barto : l'approximation (A), l'amorçage (C) et l'off-policy (D) ; vérifie si une autre lettre en fait partie. Pour b), une table est-elle une approximation, la cible du Q-learning amorce-t-elle, et de quelle politique apprend-il la valeur ? Pour c), la cible $G_t$ contient-elle une estimation ? Pour d), de quelle politique SARSA apprend-il la valeur ? Pour e), DQN abandonne-t-il l'un des trois ? Pour f), quelles politiques ont joué les transitions d'une mémoire ?

</details>

### B8.Q3 — Réseau cible et mémoire de rejeu : quel problème chacun résout-il ? Vrai ou faux

<details><summary>Indice 1</summary>

Relis « Les deux stabilisateurs » (§208.2).

</details>
<details><summary>Indice 2</summary>

Une transition reste dans une mémoire pleine autant de pas que sa capacité ; à chaque pas, elle a la probabilité « taille du mini-batch / capacité » d'être tirée. Entre deux copies, les poids du réseau cible ne bougent pas. Avec la mise à jour douce, la part des anciens poids est multipliée par $1 - \tau$ à chaque pas.

</details>
<details><summary>Indice 3</summary>

a) $10\,000$ pas de séjour, et à chaque pas une probabilité $64/10\,000$ d'être tirée : **64** tirages en moyenne. Pour b), refais ce calcul avec une capacité de 1 000. Pour c), quels poids calculent la cible ? Pour d), calcule $(1 - 0{,}005)^{200}$. Pour e), une transition rare reste-t-elle dans la mémoire après avoir servi une fois ?

</details>

### B8.Q4 — Double DQN : d'où vient la surestimation des valeurs Q ?

<details><summary>Indice 1</summary>

Relis « La surestimation, et Double DQN » (§208.2) et sa figure.

</details>
<details><summary>Indice 2</summary>

Le max de plusieurs estimations bruitées choisit celle dont le bruit est le plus haut : il est biaisé vers le haut. Avec trois estimations $\pm 1$, énumère les $2^3 = 8$ cas équiprobables. Si l'on lit la valeur dans un second jeu indépendant, le bruit qui a fait gagner l'action n'est plus celui qu'on lit. La politique gloutonne ne dépend que de l'ordre des valeurs.

</details>
<details><summary>Indice 3</summary>

a) Le max d'estimations justes en moyenne mais bruitées penche vers le haut : **B**. Pour b), dans combien des 8 cas les trois estimations valent-elles $-1$ ? Que vaut le max dans les autres ? Pour c), quelle est l'espérance d'une estimation du second jeu, quelle que soit l'action choisie par le premier ? Pour d), relis la formule de $y^{\text{Double}}$ : où est l'argmax, où est la lecture ? Pour e), Double DQN crée-t-il un réseau nouveau ? Pour f), qu'arrive-t-il à l'argmax quand on ajoute la même constante à toutes les valeurs ?

</details>

### B8.Q5 — Méthodes par valeur ou par politique : actions continues et politiques aléatoires

<details><summary>Indice 1</summary>

Relis « Apprendre la politique elle-même » (§208.3).

</details>
<details><summary>Indice 2</summary>

Un `argmax` porte sur une liste finie d'actions ; discrétiser plusieurs moteurs multiplie les combinaisons. Une politique softmax donne à chaque action $e^{z_a} / \sum_b e^{z_b}$. Une politique paramétrée tire ses actions : elle explore sans ε et peut représenter une loi mixte ; la réutilisation des transitions est l'avantage des méthodes off-policy.

</details>
<details><summary>Indice 3</summary>

a) Une politique qui rend une moyenne et un écart-type par moteur tire directement des couples réels : **C**. Pour b), 10 valeurs par moteur, pour 6 moteurs qui se combinent : quelle puissance ? Pour c), $e^{\ln 2} = 2$ : calcule les deux autres exponentielles, puis la part de la deuxième. Pour d), reprends les trois raisons de 208.3, et demande-toi laquelle des quatre lettres décrit DQN. Pour e), si ta politique est déterministe, que fait un adversaire qui l'a devinée ?

</details>

### B8.Q6 — REINFORCE : pourquoi soustraire une baseline ne biaise pas le gradient

<details><summary>Indice 1</summary>

Relis le rappel maths « l'astuce de la log-dérivée » et « La baseline et la variance » (§208.3).

</details>
<details><summary>Indice 2</summary>

$p_\theta(\text{face}) = 1 - \theta$, donc $\ln p_\theta(\text{face}) = \ln(1 - \theta)$. L'espérance de la fonction score pondère le score de chaque issue par sa probabilité. Une baseline ne biaise pas tant qu'elle ne dépend pas de l'action jouée ; elle change la variance, en bien ou en mal, selon sa distance aux retours.

</details>
<details><summary>Indice 3</summary>

a) $\frac{d}{d\theta}\ln(1 - \theta) = \frac{-1}{1 - \theta} = \frac{-1}{0{,}7} \approx$ **−1,43**. Pour b), calcule $0{,}3 \times \frac{1}{0{,}3} + 0{,}7 \times (\text{a})$. Pour c), laquelle des trois baselines dépend de l'action jouée au pas $t$ ? Pour d), une constante dépend-elle de l'action ? Et que deviennent les $(G_t - b)^2$ quand $b$ est loin des retours ? Pour e), REINFORCE a-t-il un biais au départ ?

</details>

### B8.Q7 — Avantage, erreur TD et GAE : ce que règle λ

<details><summary>Indice 1</summary>

Relis « L'erreur TD, première estimation de l'avantage », « GAE : une moyenne de toutes les longueurs » et « Fin d'épisode ou limite de temps » (§208.4).

</details>
<details><summary>Indice 2</summary>

$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ quand l'épisode continue. $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ : le poids de l'erreur TD $l$ pas plus loin est $(\gamma\lambda)^l$. $\lambda = 0$ garde un seul pas de hasard ; $\lambda = 1$ redonne $G_t - V(s_t)$. Un pas tronqué amorce, mais la récursion s'arrête.

</details>
<details><summary>Indice 3</summary>

a) $1 + 0{,}9 \times 2{,}5 - 3{,}0 =$ **0,25**. Pour b), que reste-t-il de la somme quand $\gamma\lambda = 0$ ? Pour c), quel réglage porte le hasard de tout le reste de l'épisode ? Pour d), calcule $(0{,}99 \times 0{,}95)^{10}$. Pour e), que reste-t-il du critique dans $G_t - V(s_t)$ ? Pour f), que vaut $1 - \text{done}_t$ pour un pas tronqué ?

</details>

### B8.Q8 — PPO : à quoi sert le clip, et pourquoi réutiliser plusieurs fois le même lot ?

<details><summary>Indice 1</summary>

Relis « Réutiliser un rollout sans tout casser » et « L'objectif clippé » (§208.5).

</details>
<details><summary>Indice 2</summary>

Une interaction coûte ; le ratio pondère chaque transition pour l'adapter à la nouvelle politique, ce qui ne vaut que tant qu'elle reste proche de l'ancienne. Le clip n'arrête le gradient que dans la direction qui améliorerait l'objectif. Une itération : $N \times T$ transitions, découpées en mini-batches à chaque epoch. Le ratio est l'exponentielle de la différence des log-probabilités.

</details>
<details><summary>Indice 3</summary>

a) Le ratio corrige l'écart entre politiques tant qu'elles restent proches, et chaque transition coûte une interaction : **B**. Pour b), avec un avantage négatif et un ratio de 1,5, lequel des deux termes est le minimum ? Pour c), combien de mini-batches de 256 dans $8 \times 256$ transitions, et combien d'epochs ? Pour d), calcule $e^{-0{,}9 - (-1{,}2)}$. Pour e), comment TRPO résout-il son problème d'optimisation sous contrainte ? Pour f), que vaut le gradient $-\hat A_i / n$ quand $\hat A_i = 0$ ?

</details>

### B8.Q9 — On-policy ou off-policy : DQN, REINFORCE et PPO

<details><summary>Indice 1</summary>

Relis la fin de « REINFORCE » (§208.3), la cible de DQN (§208.2) et l'encadré 💼 de 208.5.

</details>
<details><summary>Indice 2</summary>

Une méthode on-policy apprend sur des données de sa politique actuelle et les jette ensuite (ou les corrige, comme PPO, tant que la politique a peu bougé) ; une méthode off-policy apprend la valeur d'une politique à partir de données jouées par d'autres. La cible de DQN prend un max, indépendant de la politique qui a joué.

</details>
<details><summary>Indice 3</summary>

a) DQN apprend sur une mémoire de transitions anciennes, sans aucun ratio : **B**. Pour b), PPO apprend-il sur des données d'anciennes politiques lointaines ? Pour c), pour quelle politique vaut le gradient de REINFORCE ? Pour d), quel terme de la cible rend l'action suivante de la politique qui explore sans importance ? Pour e), quelle famille réutilise chaque transition de nombreuses fois ?

</details>

### B8.Q10 — Évaluer un agent : graines, épisodes, politique déterministe

<details><summary>Indice 1</summary>

Relis « Les graines », « Évaluer un agent » et l'encadré ⚠️ « un plafond caché dans l'évaluation » (§208.6).

</details>
<details><summary>Indice 2</summary>

`max_steps` coupe chaque épisode d'évaluation. L'erreur type d'une moyenne de $n$ valeurs vaut $s / \sqrt n$. Un environnement se graine une fois, au premier `reset` ; l'espace d'actions a son propre générateur. Une comparaison se fait sur plusieurs graines d'entraînement et les mêmes graines d'évaluation ; on évalue la politique sans son hasard.

</details>
<details><summary>Indice 3</summary>

a) Une politique parfaite ne tombe jamais : chaque épisode est coupé à `max_steps=300`, d'où **300**. Pour b), divise 40 par $\sqrt{100}$. Pour c), que fait un `reset()` sans graine après un premier `reset(seed=3)` ? Pour d), quelle option tient compte du hasard de l'entraînement, et sans choisir le tirage ? Pour e), `reset(seed=...)` touche-t-il au générateur de l'espace d'actions ? Pour f), quelle action ne dépend d'aucun tirage ?

</details>

### B8.Q11 — RLHF, DPO, GRPO : qui a besoin d'un modèle de récompense ou d'un critique ?

<details><summary>Indice 1</summary>

Relis 208.7 et sa figure « RLHF, DPO et GRPO ».

</details>
<details><summary>Indice 2</summary>

Le RLHF d'InstructGPT : modèle de récompense appris, puis PPO (avec son critique). DPO : une loss supervisée sur des paires, deux passes avant, aucun échantillonnage. GRPO : une récompense calculée par une règle, un groupe de réponses tirées par question, l'avantage normalisé dans le groupe. Si les récompenses d'un groupe sont toutes égales, que vaut chaque avantage ?

</details>
<details><summary>Indice 3</summary>

a) Seul le RLHF d'InstructGPT entraîne un modèle de récompense à part ; vérifie d'où viennent les récompenses de DPO (implicites) et de GRPO (vérifiées). Pour b), lesquelles sont des méthodes de RL, qui jouent pour apprendre ? Pour c), moyenne $0{,}25$, écart-type de population $\sqrt{0{,}25 \times 0{,}75}$ : calcule $(1 - 0{,}25)$ divisé par cet écart-type. Pour d), relis le dernier piège du tableau de la fiche. Pour e), que coûte le critique de PPO pour un modèle de langue ? Pour f), calcule les avantages d'un groupe de récompenses égales.

</details>

### B8.Q12 — Piratage de la récompense (reward hacking) : quand l'agent optimise la lettre et pas l'esprit

<details><summary>Indice 1</summary>

Relis « Le piratage de la récompense » (§208.7) et « Sécurité » (§208.8).

</details>
<details><summary>Indice 2</summary>

Un piratage : la récompense monte sans que la tâche voulue soit faite. Gao et coll. mesurent une récompense substitut et une récompense vraie en fonction de la KL. La pénalité KL est une laisse : plus $\beta$ est grand, plus la politique reste près de la référence. La parade de fond : une mesure qu'on n'optimise pas.

</details>
<details><summary>Indice 3</summary>

a) Pour (A), atteindre le seuil, c'est faire la tâche voulue : ce n'est pas un piratage ; applique le même critère à B, C et D. Pour b), relis ce que fait la vraie récompense quand la KL grandit. Pour c), dans quel sens $\beta$ tire-t-il la politique ? Pour d), quelle option ajoute une information que la politique ne peut pas optimiser ? Pour e), un modèle de récompense apprend-il les préférences des annotateurs, ou la vérité ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### B8.R1 — Ch. 26 : la mise à jour du Q-learning et le rôle de la mémoire de rejeu

<details><summary>Indice 1</summary>

Relis §26.6.1 du ch. 26 (la cible, l'erreur TD, le rappel maths « état terminal ou épisode coupé »), §26.9 et la docstring de `ReplayBuffer` (🔨 26.31).

</details>
<details><summary>Indice 2</summary>

$y = r + \gamma \max_{a'} Q(s', a')$, ou $y = r$ si $s'$ est terminal ; $Q(s, a) \leftarrow Q(s, a) + \alpha\,(y - Q(s, a))$. La mémoire range des transitions, une capacité fixe, la plus ancienne remplacée. SARSA apprend la valeur de la politique qu'il suit.

</details>
<details><summary>Indice 3</summary>

a) $y = 0{,}3 + 0{,}8 \times 0{,}6 =$ **0,78**. Pour b), calcule $0{,}5 + 0{,}2 \times (y - 0{,}5)$. Pour c), refais b) avec $y = r$. Pour d), que range `push` ? Pour e), que dit la docstring de `push` quand la mémoire est pleine ? Pour f), d'où vient l'action $a'$ de la cible de SARSA, et pour quelle politique cette cible vaut-elle ?

</details>

### B8.R2 — Ch. 19 : Adam et le gradient clipping

<details><summary>Indice 1</summary>

Relis §19.5.5 (Adam) et « Au-delà du livre : le gradient clipping » du ch. 19.

</details>
<details><summary>Indice 2</summary>

Au premier pas, $m_1 = (1 - \beta_1) g$ et $v_1 = (1 - \beta_2) g^2$ ; la correction de biais divise par $1 - \beta_1$ et $1 - \beta_2$ ; le pas vaut $\eta\,\hat m / (\sqrt{\hat v} + \epsilon)$. Le clipping par la norme calcule une seule norme sur tous les gradients mis bout à bout, puis un seul facteur $\min(1, c / (\lVert g \rVert + 10^{-6}))$. Il se place entre `backward` et `step`.

</details>
<details><summary>Indice 3</summary>

a) $\hat m_1 = g = -40$, $\hat v_1 = g^2 = 1\,600$, pas $\approx 0{,}001 \times (-40) / 40 = -0{,}001$, d'où $w = 1 - (-0{,}001) =$ **1,0010**. Pour b), calcule la norme de $(3 ;\ 4 ;\ 12)$, puis multiplie 3 par $0{,}5 / (\text{norme} + 10^{-6})$. Pour c), à quel moment les gradients existent-ils, et quand l'optimiseur s'en sert-il ? Pour d), où $\epsilon$ apparaît-il dans la formule d'Adam ? Pour e), les composantes sont-elles multipliées par le même facteur ?

</details>

### B8.R3 — Ch. 11 : exploration, softmax et bandits

<details><summary>Indice 1</summary>

Relis le rappel maths sur les bandits manchots (ch. 11, « Au-delà du livre : le bandit manchot »), puis §26.6.2 du ch. 26 (ε-greedy et softmax).

</details>
<details><summary>Indice 2</summary>

ε-greedy joue l'action gloutonne avec la probabilité $1 - \varepsilon + \varepsilon/K$. Softmax à température : $\pi(a) = e^{Q(a)/\tau} / \sum_b e^{Q(b)/\tau}$. Moyenne incrémentale : $Q \leftarrow Q + \frac{1}{n}(R - Q)$, avec $n$ le nombre de récompenses, la nouvelle comprise. L'encadré 🔗 de 208.3 relie le « gradient bandit » à ce chapitre.

</details>
<details><summary>Indice 3</summary>

a) $1 - 0{,}2 + 0{,}2/2 =$ **0,90**. Pour b), divise les deux valeurs par 0,5, puis prends la part de la seconde exponentielle. Pour c), $n = 4$ : calcule $4 + \frac{1}{4}(8 - 4)$. Pour d), que deviennent les $Q/\tau$ quand $\tau$ est très grand ? Pour e), relis l'encadré 🔗 de 208.3.

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B8.1 — Cible DQN à la main : état terminal, puis cible Double DQN ✏️

<details><summary>Indice 1</summary>

Relis « Les deux stabilisateurs », l'encadré ⚠️ « la limite de temps n'est pas une fin » et « La surestimation, et Double DQN » (§208.2).

</details>
<details><summary>Indice 2</summary>

DQN : $y = r + \gamma (1 - d) \max_{a'} Q_{\bar\theta}(s', a')$, avec $d = 1$ seulement si l'épisode est `terminated`. Double DQN : on prend l'argmax du réseau **en ligne**, puis on lit la valeur de cette action dans le réseau **cible**. Une troncature n'est pas une fin : $d = 0$.

</details>
<details><summary>Indice 3</summary>

a) $\max(12{,}0 ;\ 13{,}7) = 13{,}7$, et $y_1 = 1 + 0{,}99 \times 13{,}7 = 14{,}563 \approx$ **14,56**. Pour b), que vaut $1 - d$ pour une transition `terminated` ? Pour c), $d$ vaut-il 1 pour une transition `truncated` ? Pour d) et e), prends l'argmax de la colonne « en ligne », puis la valeur de cette action dans la colonne « cible ». Pour f), fais la moyenne des trois écarts DQN − Double DQN. Pour g), compare les deux argmax de la transition 1. Pour h), retranche c) de 45,3.

</details>

### Ex B8.2 — Loss de Huber et son gradient pour trois erreurs TD ✏️

<details><summary>Indice 1</summary>

Relis le rappel maths « la loss de Huber » (§208.2).

</details>
<details><summary>Indice 2</summary>

Pour chaque écart, compare $|x|$ au seuil $\delta = 1{,}5$ : en dessous, $\frac12 x^2$ et la dérivée $x$ ; au-dessus, $\delta(|x| - \delta/2)$ et la dérivée $\delta\,\operatorname{signe}(x)$. La loss du mini-batch est une moyenne : son gradient par rapport à une prédiction divise la dérivée de son terme par 3. `smooth_l1_loss(beta=β)` vaut $\ell_\beta / \beta$.

</details>
<details><summary>Indice 3</summary>

a) $|0{,}6| \le 1{,}5$ : $\frac12 \times 0{,}36 =$ **0,18**. Pour b), $|-2{,}4| > 1{,}5$ : calcule $1{,}5 \times (2{,}4 - 0{,}75)$. Pour c), calcule le troisième terme de la même façon, puis la moyenne des trois. Pour d), quelle branche s'applique, et quel signe ? Pour e), divise la dérivée du troisième terme par 3. Pour f), refais e) avec la dérivée de $\frac12 x^2$. Pour g), divise c) par $\beta$. Pour h), compare les deux dérivées sous le seuil.

</details>

### Ex B8.3 — Gradient du log d'une politique softmax à trois actions ✏️

<details><summary>Indice 1</summary>

Relis « Le théorème du gradient de politique » (§208.3), l'exemple à deux actions et son code.

</details>
<details><summary>Indice 2</summary>

Calcule les trois exponentielles et leur somme pour obtenir $\pi$. $\nabla_{\mathbf z} \log \pi(a) = \mathbf e_a - \pi$ : $-\pi_k$ pour $k \ne a$, $1 - \pi_a$ pour l'action jouée. Le pas ajoute $0{,}1 \times \hat A \times$ ce gradient à chaque logit, puis on refait la softmax. L'entropie vaut $-\sum_k \pi_k \ln \pi_k$.

</details>
<details><summary>Indice 3</summary>

a) $e^{0{,}5} = 1{,}6487$, $e^{1{,}5} = 4{,}4817$, $e^{-0{,}5} = 0{,}6065$, somme $6{,}7369$ : $\pi(2) = 0{,}6065 / 6{,}7369 =$ **0,0900**. Pour b), même somme, autre numérateur. Pour c), $z_0$ n'est pas l'action jouée : quelle formule ? Pour d), $z_2$ est l'action jouée. Pour e), additionne $\mathbf e_a$ et retranche la somme des $\pi_k$. Pour f), $z_2 + 0{,}1 \times (-2) \times$ d). Pour g), calcule les trois nouveaux logits, puis la softmax. Pour h), $-\sum_k \pi_k \ln \pi_k$ avec les probabilités d'avant le pas.

</details>

### Ex B8.4 — GAE à la main sur quatre pas, pour λ = 0, 0,95 et 1 ✏️

<details><summary>Indice 1</summary>

Relis « GAE : une moyenne de toutes les longueurs », son tableau, et « Fin d'épisode ou limite de temps » (§208.4).

</details>
<details><summary>Indice 2</summary>

D'abord les quatre erreurs TD, $\delta_t = r_t + \gamma\,(1 - \text{terminated}_t)\,\texttt{next\_values}[t] - V(s_t)$ : ici aucun pas n'est `terminated`, tous amorcent. Puis la récursion, à rebours, en partant de $\hat A_4 = 0$ : $\hat A_t = \delta_t + \gamma\lambda\,(1 - \text{done}_t)\,\hat A_{t+1}$, avec $\text{done}_1$ vrai (troncature). La cible du critique vaut $\hat A_t + V(s_t)$. $\gamma\lambda = 0{,}855$ pour $\lambda = 0{,}95$, $0{,}9$ pour $\lambda = 1$.

</details>
<details><summary>Indice 3</summary>

a) Le pas 1 est tronqué, il amorce sur la vraie dernière observation : $\delta_1 = 1 + 0{,}9 \times 3{,}6 - 3{,}8 =$ **0,44**. Pour b), même formule au pas 3, avec `next_values[3]`. Pour c), que vaut $\hat A_t$ quand $\lambda = 0$ ? Pour d), $\hat A_3 = \delta_3$ (le rollout s'arrête après), puis $\hat A_2 = \delta_2 + 0{,}855\,\hat A_3$. Pour e), $\hat A_1 = \delta_1$ (la récursion s'arrête au pas tronqué), puis $\hat A_0 = \delta_0 + 0{,}855\,\hat A_1$. Pour f), ajoute $V(s_0)$ à e). Pour g) et h), refais e) et d) avec $\gamma\lambda = 0{,}9$. Pour i), remplace $\delta_1$ par $1 - 3{,}8$, puis refais e). Pour j), écris $\hat A_0 + V(s_0)$ avec $\lambda = 1$ et simplifie.

</details>

### Ex B8.5 — Objectif clippé de PPO : quatre cas, valeur et gradient nul ou non ✏️

<details><summary>Indice 1</summary>

Relis « L'objectif clippé » (§208.5), sa figure, et l'exemple vérifié avec l'autograd.

</details>
<details><summary>Indice 2</summary>

Pour chaque transition : le terme non clippé $\rho\hat A$, le terme clippé $\operatorname{clip}(\rho, 0{,}8, 1{,}2)\,\hat A$, le minimum des deux. La loss est l'opposé de la moyenne des minimums. Gradient : $-\hat A / 4$ si le terme non clippé est le minimum (égalité comprise), 0 si le terme clippé est strictement plus petit. La fraction clippée regarde seulement $|\rho - 1| > \varepsilon$.

</details>
<details><summary>Indice 3</summary>

a) Non clippé $1{,}4 \times 2{,}0 = 2{,}8$, clippé $1{,}2 \times 2{,}0 = 2{,}4$ : le minimum vaut **2,40**. Pour b) et c), calcule les deux termes avec l'avantage négatif, et prends le plus petit (le plus négatif). Pour d), calcule aussi le terme de la transition 1, puis $-\frac14$ de la somme des quatre. Pour e) et f), quel terme est le minimum pour les transitions 1 et 3 ? Pour g), compte les transitions où le terme clippé est strictement le plus petit. Pour h), compare chaque $|\rho - 1|$ à 0,2. Pour i), refais la comparaison des deux termes avec l'intervalle $[0{,}7 ;\ 1{,}3]$.

</details>

### Ex B8.6 — Préférences : loss de Bradley-Terry puis loss DPO sur une paire de réponses ✏️

<details><summary>Indice 1</summary>

Relis « Le modèle de récompense de Bradley-Terry » et « DPO : la récompense est cachée dans la politique » (§208.7).

</details>
<details><summary>Indice 2</summary>

$P = \sigma(d)$ avec $d = r_\phi(y_w) - r_\phi(y_l)$ ; $L = -\ln \sigma(d) = \ln(1 + e^{-d})$ ; sa dérivée par rapport à $r_\phi(y_w)$ vaut $-(1 - \sigma(d))$. DPO : la récompense implicite d'une réponse vaut $\beta\,(\log\pi_\theta - \log\pi_{\text{ref}})$, la marge est la différence des deux récompenses implicites, et la loss vaut $\ln(1 + e^{-\text{marge}})$ ; sa dérivée par rapport à $\log\pi_\theta(y_w \mid x)$ vaut $-\beta\,\sigma(-\text{marge})$.

</details>
<details><summary>Indice 3</summary>

a) $d = 2{,}3 - 1{,}1 = 1{,}2$, $\sigma(1{,}2) = 1/(1 + e^{-1{,}2}) = 1/1{,}3012 \approx$ **0,769**. Pour b), calcule $\ln(1 + e^{-1{,}2})$. Pour c), $-(1 - \text{a})$. Pour d), refais b) avec $d = -1{,}2$. Pour e), que devient $d$ ? Pour f), $0{,}1 \times (-35{,}2 - (-36{,}0))$. Pour g), fais la même chose pour $y_l$, et retranche. Pour h) et i), $\ln(1 + e^{-\text{marge}})$ avec la marge de chaque $\beta$. Pour j), $-0{,}1 \times \sigma(-\text{g})$.

</details>

### Ex B8.7 — Le gradient de politique en une étape : l'astuce de la log-dérivée, puis la baseline sans biais ∂

<details><summary>Indice 1</summary>

Relis le rappel maths « l'astuce de la log-dérivée » et « Le théorème du gradient de politique » (§208.3). Tout repose sur deux faits : $\nabla p = p\,\nabla \ln p$, et $\sum_x p_\theta(x) = 1$ pour tout $\theta$.

</details>
<details><summary>Indice 2</summary>

1. Écris $\mathbb{E}[f] = \sum_x p_\theta(x) f(x)$ et dérive terme à terme. 2. Le logarithme transforme le produit en somme ; seuls les facteurs $\pi_\theta$ dépendent de $\theta$. 3. $\sum_a \pi_\theta(a \mid s) = 1$ : dérive cette égalité. 4. Pour $t' < t$, $r_{t'}$ ne dépend que de ce qui s'est passé jusqu'à $s_t$ ; conditionne sur ce début, et l'espérance de la fonction score au pas $t$ apparaît. 5. Même argument, avec $b(s_t)$ fixé une fois $s_t$ connu. 6. Écris $g$ dans les deux cas (action 0, action 1), chacun de probabilité $\frac12$.

</details>
<details><summary>Indice 3</summary>

1. $\nabla \sum_x p_\theta(x) f(x) = \sum_x f(x)\,\nabla p_\theta(x) = \sum_x f(x)\,p_\theta(x)\,\nabla \ln p_\theta(x)$ : c'est $\mathbb{E}[f\,\nabla \ln p_\theta]$ ; le $\ln$ exige $p_\theta(x) > 0$ là où $f(x) \ne 0$. Pour 2, écris $\ln p_\theta(\tau) = \ln\mu(s_0) + \sum_t [\ln \pi_\theta(a_t \mid s_t) + \ln P(s_{t+1} \mid s_t, a_t)]$ et dérive, puis applique 1. avec $f = R$. Pour 3, $\sum_a \nabla \pi_\theta(a \mid s) = \nabla 1$. Pour 4, écris $\mathbb{E}[\,\cdot\,] = \mathbb{E}_{\text{début}}\big[\gamma^{t'} r_{t'}\,\mathbb{E}_{a_t}[\nabla \ln \pi_\theta(a_t \mid s_t) \mid s_t]\big]$. Pour 5, sors $b(s_t)$ de l'espérance sur $a_t$. Pour 6, $\frac{\partial}{\partial z}\ln\pi(1) = 1 - \pi(1)$ et $\frac{\partial}{\partial z}\ln\pi(0) = -\pi(1)$ : écris $g$ dans les deux cas, puis $\mathbb{E}[g]$ et $\mathbb{E}[g^2] - \mathbb{E}[g]^2$ comme des fonctions de $b$.

</details>

### Ex B8.8 — Fermi : combien d'interactions ? CartPole, Atari (DQN 2015), AlphaGo Zero et le temps CPU correspondant 🧮

<details><summary>Indice 1</summary>

Relis « Ce que coûte un pas sur CPU » et le rappel maths « une estimation de Fermi en RL » (§208.6), puis « L'appétit d'interactions » (§208.8).

</details>
<details><summary>Indice 2</summary>

CartPole : itérations $= 500\,000 / (4 \times 128)$ ; chaque itération fait 128 appels de `step` vectorisés (4 copies, une passe avant, un tirage) et $4 \times 4$ pas d'Adam. Atari : jours $\times$ 86 400 s $\times$ 60 images ; une mise à jour tous les 4 choix d'action, 32 transitions chacune. AlphaGo Zero : parties $\times$ coups $\times$ simulations ; durée d'une partie $= 200 \times 0{,}4$ s. Compare toujours à un ordre de grandeur humain.

</details>
<details><summary>Indice 3</summary>

1. $500\,000 / 512 \approx 976$ itérations ; $976 \times 128 \approx 125\,000$ appels vectorisés à $4 \times 20 + 50 + 200 \approx 330$ µs, soit environ 40 s de collecte ; $976 \times 16 \approx 15\,600$ pas d'Adam à 3 ms, soit environ 47 s : **un peu moins de 2 minutes** en tout, et environ cinq fois moins pour les 100 000 pas du défi. Pour 2, calcule $38 \times 86\,400 \times 60$ et compare à 50 millions ; puis $50 \times 10^6 / 4$ mises à jour, $\times 32$, $\times 49$ ; et 38 jours contre 2 heures. Pour 3, $4{,}9 \times 10^6 \times 200$ positions, $\times 1\,600$ passes ; $4{,}9 \times 10^6 \times 80$ s divisés par $3 \times 86\,400$ s ; les passes $\times 0{,}2$ s, en années. Pour 4, mets ces nombres face à ceux d'un humain.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B8.9 — PPO expliqué en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis « Apprendre la politique elle-même » (§208.3), « Réutiliser un rollout sans tout casser » et « L'objectif clippé » (§208.5).

</details>
<details><summary>Indice 2</summary>

Une image pour l'apprentissage par essais : un joueur qui ne reçoit qu'un score à la fin. Pourquoi pas de labels : personne ne dit quelle action était la bonne. Le petit pas : on ne change la façon de jouer que d'un peu, et une fois qu'une action a assez gagné en probabilité, on cesse de la pousser. La réutilisation : jouer coûte, apprendre plusieurs fois sur les mêmes parties tant qu'on a peu changé. Une limite : beaucoup de parties, ou une récompense mal écrite.

</details>
<details><summary>Indice 3</summary>

Commence par : « PPO apprend une façon de jouer, une probabilité pour chaque action, à partir des seules récompenses que l'agent reçoit en jouant. » Puis : pas de bonne réponse à imiter, seulement un score ; on augmente les probabilités des actions qui ont fait mieux que prévu ; l'image du petit pas et du frein ; plusieurs passages sur les mêmes parties ; la limite.

</details>

### Ex B8.10 — RLHF : qui choisit les préférences ? Annotateurs, valeurs et flagornerie des modèles ⚖️

<details><summary>Indice 1</summary>

Relis « Le RLHF : PPO sous une laisse KL », les faits d'InstructGPT et « Le piratage de la récompense » (§208.7), puis la section 5.2 d'Ouyang et coll. (2022).

</details>
<details><summary>Indice 2</summary>

Les auteurs d'InstructGPT citent quatre groupes : leurs annotateurs, eux-mêmes (les consignes), les clients de l'API (les prompts) et, par défaut, les absents (les autres utilisateurs et les personnes touchées). Bradley-Terry ne sait représenter qu'une moyenne : deux avis contraires sur une même paire deviennent une probabilité proche de ½. La flagornerie naît quand une réponse agréable gagne plus souvent que la réponse exacte. Une mesure non récompensée se collecte à part, par d'autres personnes ou contre une vérité de référence.

</details>
<details><summary>Indice 3</summary>

Pour 1, construis un tableau : le groupe d'InstructGPT, son équivalent pour l'assistant RH (des annotateurs internes ? la DRH qui écrit les consignes ? les salariés qui posent les questions ?), et qui manque (les représentants du personnel, les juristes). Pour 2, pense aux congés, aux sanctions, au télétravail. Pour 3, suis la chaîne : une réponse « oui » plaît, elle gagne la paire, le modèle de récompense la note haut, PPO la reproduit ; que voit la pénalité KL ? Pour 4, un échantillon de réponses vérifié par un juriste contre le texte de la convention collective. Pour 5, la rémunération, le temps par paire, les consignes, la possibilité de s'abstenir. Pour 6, DPO change l'algorithme, pas l'origine des paires. Pour 7, réponds à « qui, combien, quelles consignes, quel accord, quelles limites ».

</details>

### Ex B8.11 — Schulman et coll. (2017) : « Proximal Policy Optimization Algorithms » 📄

<details><summary>Indice 1</summary>

Lis l'article dans l'ordre des questions : les sections 1 à 5 tiennent en cinq pages ; la section 6 et l'annexe A donnent les chiffres. La fiche (208.4 et 208.5) en reprend les équations 7 et 9 et la figure 1.

</details>
<details><summary>Indice 2</summary>

Pour les équations, recopie-les avec les notations de l'article ($r_t(\theta)$ est le ratio, $\epsilon$ la largeur du clip). Le tableau 1 donne des scores normalisés : 0 pour la politique aléatoire, 1 pour le meilleur résultat, moyennés sur 21 runs (7 environnements × 3 graines). Le tableau 5 donne les réglages Atari ; une image sur quatre est jouée par l'agent.

</details>
<details><summary>Indice 3</summary>

1. Q-learning : « fails on many simple problems and is poorly understood » ; vanilla PG : faible efficacité en données et robustesse ; TRPO : compliqué, incompatible avec le bruit (dropout) et le partage de paramètres. Pour 2, cherche la phrase « not well-justified » et les raisons du choix d'une contrainte. Pour 3, lis le paragraphe qui suit l'équation 7. Pour 4, les légendes des deux figures disent tout. Pour 5, la section 4 tient en une demi-page. Pour 6, l'équation 10 est le cas $\lambda = 1$. Pour 7, compare les lignes « Clipping » du tableau 1, puis relis 208.6 sur les écarts mesurés sur peu de runs. Pour 8, le tableau 5 et la légende du tableau 6. Pour 9, relis le paragraphe « Les implémentations ajoutent des détails qui comptent » (§208.5) et le tableau de 208.6.

</details>

<a id="entretien"></a>

## 💼 Entretien

### B8.E1 — Méthodes par valeur ou par gradient de politique : avantages et inconvénients

<details><summary>Indice 1</summary>

Relis 208.2, « Apprendre la politique elle-même » (§208.3) et l'encadré 💼 de 208.5.

</details>
<details><summary>Indice 2</summary>

Trois axes : le type d'actions (discrètes ou continues), le coût des données (off-policy contre on-policy), la stabilité et le réglage. Valeur : réutilise ses données, actions discrètes, instable sans stabilisateurs, surestime. Politique : actions continues, politiques aléatoires, changements progressifs, mais beaucoup d'interactions et une forte variance.

</details>
<details><summary>Indice 3</summary>

Commence par : « Tout dépend de deux questions : mes actions sont-elles discrètes, et combien me coûte une interaction ? » Puis : DQN et ses stabilisateurs, PPO et sa consommation de données, l'exemple d'un robot réel, SAC pour les actions continues quand les données coûtent cher.

</details>

### B8.E2 — Explique PPO et le rôle du clipping

<details><summary>Indice 1</summary>

Relis 208.5 en entier, et ✏️ B8.5.

</details>
<details><summary>Indice 2</summary>

Le problème : un gradient de politique est on-policy ; plusieurs pas sur les mêmes données font dériver l'estimation. La solution : le ratio de probabilités, et un min entre le terme normal et le terme clippé, qui annule le gradient d'une transition dont le ratio est déjà sorti de l'intervalle dans la bonne direction. Sans clip : des mises à jour destructrices, une politique qui s'effondre.

</details>
<details><summary>Indice 3</summary>

Commence par : « PPO est un acteur-critique qui réutilise chaque rollout pendant quelques epochs, en empêchant la politique de trop s'éloigner de celle qui a joué. » Puis : le ratio, le min, les deux cas de signe de l'avantage, ce qui se passe sans clip (le tableau 1 de l'article), les journaux qui le surveillent (KL approchée, fraction clippée).

</details>

### B8.E3 — Ton agent RL n'apprend pas : comment le débogues-tu ?

<details><summary>Indice 1</summary>

Relis « Déboguer un agent qui n'apprend pas » (§208.6) et le tableau des pièges de la fiche.

</details>
<details><summary>Indice 2</summary>

Du moins coûteux au plus coûteux : les calculs isolés testés, les formes et les frontières d'épisodes, un environnement où la bonne réponse est connue, les journaux (retour, entropie, KL, fraction clippée, variance expliquée, norme du gradient), puis la comparaison à une implémentation de référence. Plusieurs graines avant toute conclusion.

</details>
<details><summary>Indice 3</summary>

Commence par : « Un agent qui n'apprend pas ne lève aucune erreur : je cherche d'abord les erreurs silencieuses les plus fréquentes. » Puis : `terminated` contre `truncated`, la cible sans gradient, un signe de loss inversé, les formes qui diffusent ; un test sur un problème plus simple ; les journaux ; la référence ; les graines.

</details>

### B8.E4 — PPO avec un modèle de récompense ou DPO pour aligner un modèle de langue : que choisis-tu ?

<details><summary>Indice 1</summary>

Relis 208.7 (RLHF, DPO, GRPO) et la figure « RLHF, DPO et GRPO ».

</details>
<details><summary>Indice 2</summary>

DPO : simple, stable, une loss supervisée sur un dataset de paires figé, deux passes avant ; mais hors ligne, il n'explore pas et ne voit que les réponses du dataset. RLHF avec PPO : un modèle de récompense réutilisable, des réponses tirées de la politique actuelle, mais quatre modèles en mémoire, un réglage délicat, le piratage de la récompense. Si la récompense se vérifie, GRPO.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je pars de DPO, sauf si trois conditions me poussent vers le RL. » Puis : le budget de calcul et de mémoire, la nature de la récompense (préférences ou vérifiable), le besoin d'apprendre sur ses propres réponses, les mesures pour détecter le piratage dans les deux cas.

</details>

### B8.E5 — Pourquoi faut-il plusieurs graines pour comparer deux algorithmes de RL ?

<details><summary>Indice 1</summary>

Relis « Les graines » et « Rapporter plusieurs graines » (§208.6), et l'encadré 🕰️ sur rliable.

</details>
<details><summary>Indice 2</summary>

Deux graines d'un même algorithme peuvent donner des courbes très différentes. Une comparaison demande plusieurs graines par méthode, une évaluation à part (politique déterministe, mêmes graines d'évaluation, `max_steps` correct), un budget d'interactions égal, et un résumé honnête : médiane et intervalle, IQM et bootstrap stratifié quand il y a beaucoup de runs.

</details>
<details><summary>Indice 3</summary>

Commence par : « En RL, la variance entre graines est souvent plus grande que l'écart entre deux méthodes : une courbe ne prouve rien. » Puis : ce que je demande (combien de graines, quel budget, quelle évaluation, quels réglages pour la référence), comment je rapporte, et ce qu'a montré Agarwal et coll.

</details>

<a id="notebook"></a>

## Exercices du notebook

Les indices des exercices du notebook (B8.12 à B8.28) seront ajoutés à la prochaine session de génération, avec le notebook.
