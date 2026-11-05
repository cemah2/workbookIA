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

### B8.Q3 — Réseau cible et mémoire de rejeu : quel problème chacun résout-il ?

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

a) $1 + 0{,}9 \times 2{,}5 - 3{,}0 =$ **0,25**. Pour b), que reste-t-il de la somme quand $\gamma\lambda = 0$ ? Pour c), quel réglage porte le hasard de tout le reste de l'épisode ? Pour d), calcule $(0{,}99 \times 0{,}95)^{10}$. Pour e), que reste-t-il du critique dans $G_t - V(s_t)$ ? Pour f), sur quelle valeur $\delta_5$ amorce-t-il, et que vaut $1 - \text{done}_5$ ?

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

### B8.R2 — Ch. 19 : l'optimiseur Adam et le gradient clipping

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

1. $\nabla \sum_x p_\theta(x) f(x) = \sum_x f(x)\,\nabla p_\theta(x) = \sum_x f(x)\,p_\theta(x)\,\nabla \ln p_\theta(x)$ : c'est $\mathbb{E}[f\,\nabla \ln p_\theta]$ ; le $\ln$ exige $p_\theta(x) > 0$ : on somme sur les issues possibles, un ensemble qui ne dépend pas de $\theta$. Pour 2, écris $\ln p_\theta(\tau) = \ln\mu(s_0) + \sum_t [\ln \pi_\theta(a_t \mid s_t) + \ln P(s_{t+1} \mid s_t, a_t)]$ et dérive, puis applique 1. avec $f = R$. Pour 3, $\sum_a \nabla \pi_\theta(a \mid s) = \nabla 1$. Pour 4, écris $\mathbb{E}[\,\cdot\,] = \mathbb{E}_{\text{début}}\big[\gamma^{t'} r_{t'}\,\mathbb{E}_{a_t}[\nabla \ln \pi_\theta(a_t \mid s_t) \mid s_t]\big]$. Pour 5, sors $b(s_t)$ de l'espérance sur $a_t$. Pour 6, $\frac{\partial}{\partial z}\ln\pi(1) = 1 - \pi(1)$ et $\frac{\partial}{\partial z}\ln\pi(0) = -\pi(1)$ : écris $g$ dans les deux cas, puis $\mathbb{E}[g]$ et $\mathbb{E}[g^2] - \mathbb{E}[g]^2$ comme des fonctions de $b$.

</details>

### Ex B8.8 — Fermi : combien d'interactions ? CartPole, Atari (DQN 2015), AlphaGo Zero et le temps CPU correspondant 🧮

<details><summary>Indice 1</summary>

Relis « Ce que coûte un pas sur CPU » et le rappel maths « une estimation de Fermi en RL » (§208.6), puis « L'appétit d'interactions » (§208.8).

</details>
<details><summary>Indice 2</summary>

CartPole : itérations $= 500\,000 / (4 \times 128)$ ; chaque itération fait 128 appels de `step` vectorisés (4 copies, deux passes avant, un tirage) et $4 \times 4$ pas d'Adam. Atari : jours $\times$ 86 400 s $\times$ 60 images ; une mise à jour tous les 4 choix d'action, 32 transitions chacune. AlphaGo Zero : parties $\times$ coups $\times$ simulations ; durée d'une partie $= 200 \times 0{,}4$ s. Compare toujours à un ordre de grandeur humain.

</details>
<details><summary>Indice 3</summary>

1. $500\,000 / 512 \approx 976$ itérations ; $976 \times 128 \approx 125\,000$ appels vectorisés à $4 \times 20 + 2 \times 50 + 200 \approx 380$ µs (l'acteur et le critique), soit environ 48 s de collecte ; $976 \times 16 \approx 15\,600$ pas d'Adam à 3 ms, soit environ 47 s : **un peu plus d'une minute et demie** en tout, et environ cinq fois moins pour les 100 000 pas du défi. Pour 2, calcule $38 \times 86\,400 \times 60$ et compare à 50 millions ; puis $50 \times 10^6 / 4$ mises à jour, $\times 32$, $\times 49$ ; et 38 jours contre 2 heures. Pour 3, $4{,}9 \times 10^6 \times 200$ positions, $\times 1\,600$ passes ; $4{,}9 \times 10^6 \times 80$ s divisés par $3 \times 86\,400$ s ; les passes $\times 0{,}2$ s, en années. Pour 4, mets ces nombres face à ceux d'un humain.

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

### Ex B8.12 — CartPole pour le deep RL : espaces, politique aléatoire, evaluate_policy et la référence tabulaire du ch. 26 📦

<details><summary>Indice 1</summary>

Tout ce que demande `describe_12` est dans l'objet `env` : son espace d'observation, son espace d'actions et sa fiche d'enregistrement `env.spec` (relis le premier bloc de code de 208.1). Pour la politique aléatoire, le générateur doit **survivre** d'un appel à l'autre : il se crée dans `random_policy_12`, et la fonction rendue s'en sert. Pour `evaluate_12`, relis le piège « un plafond caché dans l'évaluation » (208.6).

</details>
<details><summary>Indice 2</summary>

- `describe_12` : `env.observation_space.shape` (déjà un tuple), `env.action_space.n`, `env.spec.max_episode_steps`, `env.spec.reward_threshold` ; convertis en `tuple`, `int` et `float`.
- `random_policy_12` : une fermeture (*closure*, 0A) : `rng = np.random.default_rng(seed)`, puis une fonction interne `policy(state, info)` qui rend `int(rng.integers(2))`, et `return policy`.
- `evaluate_12` : un seul appel à `mylearn.rl.evaluate_policy`, avec `max_steps=env.spec.max_episode_steps` et `rng=np.random.default_rng(seed)`.

</details>
<details><summary>Indice 3</summary>

```python
def random_policy_12(seed):
    rng = np.random.default_rng(seed)       # created ONCE, outside the policy
    def policy(state, info):
        ...                                  # one draw of rng per call, as a Python int
    return policy


def evaluate_12(env, policy, seed, n_episodes=100):
    return mylearn.rl.evaluate_policy(env, policy, n_episodes, max_steps=..., rng=...)
```

</details>

### Ex B8.13 — DQN sur CartPole : réseau Q, ReplayBuffer de rl.py, réseau cible, Huber, ε décroissant 📦

<details><summary>Indice 1</summary>

Relis l'algorithme en cinq étapes de 208.2 (« Les deux stabilisateurs ») et le premier bloc de code de 208.2 : chaque pas d'environnement fait une action, range une transition, et parfois une mise à jour ; le réseau cible n'est qu'une **copie** du réseau en ligne, réécrite de temps en temps. La cible se calcule sous `torch.no_grad()` ; la prédiction se lit avec `gather` sur l'action jouée. Écris d'abord les trois petites fonctions, vérifie-les, puis la boucle.

</details>
<details><summary>Indice 2</summary>

- `make_q_net_13` : `nn.Sequential` de trois `nn.Linear` séparés par deux `nn.ReLU`.
- `dqn_targets_13` : `q_target(next_states).max(dim=1).values` donne une valeur par ligne ; `terminated.float()` transforme les booléens en 0 et 1 (on ne soustrait pas un booléen d'un nombre en PyTorch).
- `epsilon_13` : la progression `min(1, step / (eps_fraction × total_steps))`, puis une interpolation entre `eps_start` et `eps_end`.
- `train_dqn_13`, dans l'ordre : les graines et le réseau, `copy.deepcopy` pour le réseau cible, Adam, le générateur de l'exploration, la `ReplayBuffer` ; puis la boucle `for step in range(total_steps)` : l'action (`rng.random() < ε` → `int(rng.integers(2))`, sinon `argmax` sans gradient), `env.step`, `buffer.push(..., terminated)`, la fin d'épisode (les deux listes de `history`, `env.reset()`), la mise à jour (`buffer.sample`, les tenseurs, la cible par `dqn_targets_13`, appelée par son nom, `gather`, `F.huber_loss`, `zero_grad`, `backward`, `clip_grad_norm_`, `step`), la copie du réseau cible quand `(step + 1) % target_every == 0` ; `env.reset()` sans graine après le premier épisode.
- Les tableaux de `buffer.sample` sont en NumPy : `torch.as_tensor(...)` ; les récompenses sont en float64, à passer en float32 (`dtype=torch.float32`) pour les mêler aux sorties du réseau.

</details>
<details><summary>Indice 3</summary>

```python
def train_dqn_13(env, seed, total_steps, settings=SETTINGS_13):
    s = settings
    torch.manual_seed(seed)
    q_net = make_q_net_13(s["hidden"])
    q_target = copy.deepcopy(q_net)
    # Adam, rng = np.random.default_rng(seed), the ReplayBuffer of mylearn.rl (random_state=seed), history
    state, _ = env.reset(seed=seed)
    for step in range(total_steps):
        # 1. epsilon-greedy action with epsilon_13(step, total_steps, s)
        next_state, reward, terminated, truncated, _ = env.step(action)
        buffer.push(state, action, reward, next_state, terminated)       # terminated only
        # 2. end of an episode (terminated or truncated): history, env.reset()
        # 3. an update when step + 1 >= learning_starts (and every train_freq steps)
        q_sa = q_net(states).gather(1, actions[:, None]).squeeze(1)
        # 4. the copy into q_target every target_every steps
    return q_net, buffer, history
```

</details>

### Ex B8.14 — Le DQN qui diverge : quatre erreurs de cible, de fin d'épisode et d'exploration 🐛

<details><summary>Indice 1</summary>

Lance d'abord l'expérience telle quelle : regarde les trois courbes (les retours, ε, les valeurs Q) et demande-toi ce qu'un DQN sain devrait y montrer. Puis lis les diagnostics dans l'ordre : chacun regarde un seul morceau, et le premier en échec dit lequel. Relis ensuite les pièges de 208.1 et 208.2, et 🧠 Q3.

</details>
<details><summary>Indice 2</summary>

- Point 1 : une cible doit être une **constante** pour le gradient, et venir du réseau qui ne bouge pas entre deux copies. Lequel des deux réseaux reçus `bellman_targets` utilise-t-il, et sous quel contexte ?
- Point 2 : `remember` a deux rôles, ranger la transition et dire si l'épisode est fini. Ces deux rôles demandent-ils le même drapeau (208.2, ⚠️ « la limite de temps n'est pas une fin ») ?
- Point 3 : calcule à la main `exploration_rate(0)` et `exploration_rate(EPS_STEPS)` : la formule passe-t-elle vraiment d'une borne à l'autre ?
- Point 4 : combien d'épisodes de 50 pas tiennent dans la mémoire, et que tire alors un mini-batch de 64 ? Compare aux capacités du tableau de 208.2.

</details>
<details><summary>Indice 3</summary>

Les quatre lignes fautives : la ligne `follow = …` de `bellman_targets`, l'appel `memory.push(...)` de `remember`, le `return` d'`exploration_rate` et la constante `MEMORY_SLOTS`. Pour la première, par exemple :

```python
    with torch.no_grad():                                         # a constant for the gradient
        follow = frozen(torch.as_tensor(batch["next_states"])).max(dim=1).values
```

</details>

### Ex B8.15 — Ablations du DQN : sans réseau cible, sans rejeu, avec Double DQN — trois graines 🔬

<details><summary>Indice 1</summary>

Les quatre variantes ne diffèrent que par **deux** choses : le réseau qui donne la valeur de $s'$ dans la cible (et, pour Double DQN, celui qui choisit l'action), et la taille de la mémoire. Relis la formule de Double DQN (208.2) et ta copie de ✏️ B8.1 d) et e).

</details>
<details><summary>Indice 2</summary>

- `targets_15` : la même cible que `dqn_targets_13`, sous `torch.no_grad()` ; pour `"no_target"`, le réseau en ligne remplace `frozen` ; pour `"double"`, `online(next_states).argmax(dim=1, keepdim=True)` donne l'action, et `frozen(next_states).gather(1, ...)` sa valeur.
- `memory_size_15` : un test sur `variant`, deux clés de `settings`.
- Pour lire la figure : regarde la médiane **et** la largeur de l'intervalle de chaque variante ; un écart de médianes plus petit que les intervalles ne permet pas de conclure avec trois graines.

</details>
<details><summary>Indice 3</summary>

```python
def targets_15(online, frozen, next_states, rewards, terminated, gamma, variant):
    with torch.no_grad():
        if variant == "double":
            best = online(next_states).argmax(dim=1, keepdim=True)
            following = ...                                  # the value of `best` read in frozen, one per row
        else:
            following = ...                                  # max over the actions of the right network
        return rewards + gamma * (1.0 - terminated.float()) * following
```

</details>

### Ex B8.16 — Prédire : REINFORCE sans baseline, avec baseline, avec retours normalisés — lequel apprend le plus vite ? 🔮

<details><summary>Indice 1</summary>

Relis « La baseline et la variance » (208.3) et ta copie de ∂ B8.7 : une baseline ne change pas l'espérance du gradient, mais sa **dispersion**. Pense à ce que vaut $G_t$ sur CartPole (tous les retours sont positifs, et d'autant plus grands que l'épisode dure) et à ce que fait chaque variante d'un pas d'Adam au learning rate fixé.

</details>
<details><summary>Indice 2</summary>

Pour a), demande-toi quelle variante a le gradient le moins bruité **et** des pas d'une taille raisonnable ; la normalisation centre comme une baseline, mais change aussi l'échelle de chaque épisode. Pour b), qu'est-ce qui permettrait, ou empêcherait, un gradient sans biais mais bruité d'atteindre 200 en 300 mises à jour ? Pour c), une prédiction « sur chacune des six graines » est forte : la dispersion entre graines peut-elle faire passer une graine sans baseline devant une graine avec ?

</details>
<details><summary>Indice 3</summary>

Pour a), range les trois variantes par la dispersion attendue de leur gradient, puis demande-toi si une échelle de pas différente peut inverser l'ordre. Pour b), pars de la moyenne aléatoire (≈ 22) : combien de graines atteindraient 200 avant l'épisode 300 si la progression varie beaucoup d'une graine à l'autre ? Une moyenne glissante qui touche 200 une seule fois suffit. Pour c), compare l'écart moyen que tu attends entre les deux variantes à la dispersion entre graines.

</details>

### Ex B8.17 — REINFORCE avec et sans baseline : variance du gradient et vitesse d'apprentissage 📦

<details><summary>Indice 1</summary>

Relis « REINFORCE » (208.3) et son bloc de code : une loss par épisode, $-\operatorname{mean}_t \log\pi(a_t \mid s_t)(G_t - b)$, dont le gradient se calcule par l'autograd. Pour un gradient **par épisode**, on calcule cette loss épisode par épisode et on demande le gradient sans toucher aux `.grad` du réseau (`torch.autograd.grad`). La dispersion se calcule ensuite sur ces lignes, sans PyTorch.

</details>
<details><summary>Indice 2</summary>

- `episode_gradients_17` : les retours `mylearn.rl.returns_to_go(e["rewards"], gamma)` de chaque épisode ; $b$ = la moyenne de **tous** ces retours mis bout à bout (`np.concatenate`) pour `"mean"` ; pour chaque épisode, `torch.log_softmax(policy_net(states), dim=1)`, `gather` sur les actions, la loss, puis `torch.autograd.grad(loss, list(policy_net.parameters()))` et `torch.cat([g.reshape(-1) for g in grads])` ; une ligne par épisode, `np.stack`.
- `train_reinforce_17` : un épisode entier (une boucle `while`), avec `torch.distributions.Categorical(logits=...)`, `.sample()` et `.log_prob(...)` à chaque pas ; puis les retours, $b$ (les 20 épisodes précédents, 0 au premier), la loss, `zero_grad`, `backward`, `step` ; garde les retours des épisodes dans une liste glissante de 20.

</details>
<details><summary>Indice 3</summary>

```python
def episode_gradients_17(policy_net, episodes, baseline, gamma=0.99):
    returns = [mylearn.rl.returns_to_go(e["rewards"], gamma) for e in episodes]
    b = ...                                          # 0, or the mean of np.concatenate(returns)
    params = list(policy_net.parameters())
    rows = []
    for e, G in zip(episodes, returns):
        log_pi_a = torch.log_softmax(policy_net(torch.as_tensor(e["states"])), dim=1).gather(...).squeeze(1)
        loss = -(log_pi_a * torch.as_tensor(G - b, dtype=torch.float32)).mean()
        grads = torch.autograd.grad(loss, params)
        # one flat float64 row per episode
    return np.stack(rows)
```

</details>

### Ex B8.18 — compute_gae et explained_variance 🔨

<details><summary>Indice 1</summary>

Relis le tableau de 208.4 (l'exemple à rebours) et « Fin d'épisode ou limite de temps » : deux drapeaux, deux rôles. `terminated` coupe l'amorçage dans $\delta_t$ ; `terminated` **ou** `truncated` coupe la récursion. La forme `(T, n_envs)` ne demande rien de plus : écris chaque calcul pour une ligne entière (un pas de toutes les copies à la fois).

</details>
<details><summary>Indice 2</summary>

- Valide d'abord : `np.asarray(..., dtype=float)` pour les trois tableaux de valeurs, `dtype=bool` pour les drapeaux, des formes **identiques** (`a.shape == b.shape`, sans diffusion), $\gamma$ et $\lambda$ dans $[0, 1]$.
- Les erreurs TD d'un coup, en NumPy : `delta = r + gamma * np.where(term, 0.0, next_values) - values` (`np.where` plutôt que `(1 - term) *`, qui donne `nan` avec une valeur infinie).
- Puis une boucle `for t in reversed(range(T))` : `last = delta[t] + gamma * lam * np.where(done[t], 0.0, last)`, en partant de `last = 0` ; `advantages[t] = last`.
- `explained_variance` : `np.var(y_true)` ; si elle vaut 0, `float("nan")` ; sinon `1 - np.var(y_true - y_pred) / var`.

</details>
<details><summary>Indice 3</summary>

```python
def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95):
    # 1. arrays, same shapes (ValueError otherwise), gamma and lam in [0, 1]
    done = term | trunc
    delta = r + gamma * np.where(term, 0.0, next_values) - values
    advantages = np.zeros_like(delta)
    last = np.zeros_like(delta[0])                       # A_T = 0, one value per column
    for t in reversed(range(len(delta))):
        ...                                              # the recursion, stopped where done[t]
    return advantages, advantages + values
```

</details>

### Ex B8.19 — clipped_surrogate : valeur et gradient par rapport au ratio 🔨

<details><summary>Indice 1</summary>

Relis « L'objectif clippé » (208.5) et sa vérification en trois transitions, puis ta copie de ✏️ B8.5. Le minimum de deux termes a pour gradient celui du terme **qui est le minimum** ; le terme clippé ne dépend plus du ratio hors de l'intervalle, donc son gradient y est nul.

</details>
<details><summary>Indice 2</summary>

- Valide : deux tableaux 1-D de même forme, aucun ratio négatif, `clip_eps > 0` (un `ValueError` sinon).
- `unclipped = ratio * adv`, `clipped = np.clip(ratio, 1 - eps, 1 + eps) * adv`, `loss = -np.mean(np.minimum(unclipped, clipped))`.
- Le gradient : `-adv / n` là où le terme non clippé est le minimum, **égalité comprise** (`unclipped <= clipped`), 0 ailleurs : un `np.where`.

</details>
<details><summary>Indice 3</summary>

```python
def clipped_surrogate(ratio, advantages, clip_eps=0.2):
    # 1. arrays, the same shape (n,), no negative ratio, clip_eps > 0
    unclipped = r * a
    clipped = np.clip(r, 1 - clip_eps, 1 + clip_eps) * a
    loss = -float(np.mean(np.minimum(unclipped, clipped)))
    grad = ...                                           # -a / n where unclipped <= clipped, else 0
    return loss, grad
```

</details>

### Ex B8.20 — PPO from scratch sur CartPole : environnements vectorisés, acteur-critique, GAE, clip, entropie 📦

<details><summary>Indice 1</summary>

Relis « La loss complète et la boucle » (208.5) : quatre étapes, collecter, calculer GAE, mettre à jour pendant plusieurs epochs, recommencer. Puis le rappel outil de l'API vectorisée et son bloc de code : le drapeau `just_reset` se met à jour **après** chaque pas (`terminated | truncated`), et le pas suivant de cette copie n'est pas une transition. Écris et vérifie les morceaux dans l'ordre : le réseau, le rollout, la loss, la boucle.

</details>
<details><summary>Indice 2</summary>

- `ActorCritic20` : deux `nn.Sequential` (`self.actor`, `self.critic`) ; `value` rend `self.critic(obs).squeeze(-1)`. L'initialisation orthogonale de CleanRL (`nn.init.orthogonal_`, gain $\sqrt 2$, 0,01 pour la couche des logits, 1 pour celle de la valeur) aide, sans être exigée.
- `rollout_20` : à chaque pas, sous `torch.no_grad()`, les log-probabilités (`torch.log_softmax`), une action par copie (`torch.multinomial(probs, 1, generator=generator)` ou `Categorical`), $V$ ; `envs.step(actions.numpy())` ; une ligne par champ (copies avec `np.array(..., copy=True)`), `valid = ~just_reset` ; puis `just_reset = terminated | truncated`. À la fin, `np.stack` et `next_values` en une passe du critique sur `next_obs`.
- `ppo_loss_20` : `log_ratio = log π(a|s) − old_log_probs`, `ratio = log_ratio.exp()`, puis la formule ; les `parts` se calculent sous `torch.no_grad()`, en `float`.
- `train_ppo_20` : `rollout_20`, `compute_gae` et `ppo_loss_20` appelées par leur nom (la vérification les observe) ; après chaque rollout, `compute_gae` sur les tableaux `(T, N)` entiers, tels que le rollout les rend (les pas de reset ont des drapeaux à `False` et les pas qui les précèdent arrêtent la récursion) ; puis on ne garde que `batch[k][valid]` pour la mise à jour ; `torch.randperm(n, generator=generator)` pour mélanger ; dans chaque mini-batch, `a = (a - a.mean()) / (a.std() + 1e-8)` ; le retour des épisodes finis se suit par copie, en ajoutant les récompenses des pas valides.

</details>
<details><summary>Indice 3</summary>

```python
def rollout_20(agent, envs, obs, just_reset, n_steps, generator):
    rows = {k: [] for k in ("obs", "actions", "log_probs", "values", "rewards", "terminated", "truncated",
                            "next_obs", "valid")}
    for _ in range(n_steps):
        # under torch.no_grad(): log-probabilities, one action per copy, values
        next_obs, rewards, terminated, truncated, _ = envs.step(actions.numpy())
        # append the nine fields; valid = ~just_reset
        just_reset = terminated | truncated
        obs = next_obs
    batch = {k: np.stack(v) for k, v in rows.items()}
    batch["next_values"] = ...                           # V(next_obs), shape (n_steps, N)
    return batch, obs, just_reset
```

</details>

### Ex B8.21 — Lire les journaux de PPO : KL approchée, fraction clippée, entropie, variance expliquée 📈

<details><summary>Indice 1</summary>

Relis « Les journaux de PPO » (208.5) et « Le critique et sa variance expliquée » (208.4) : chaque réglage changé agit d'abord sur **une** des quatre mesures. Ne commence pas par le retour : deux runs malades peuvent finir à 500, et un run sain peut avoir une mauvaise fin. Lis colonne par colonne, en comparant les cinq runs à la même échelle.

</details>
<details><summary>Indice 2</summary>

Pour chaque réglage changé, demande-toi ce qu'il touche :
- le learning rate fixe la taille de chaque pas d'Adam : combien la politique bouge pendant une itération, et donc combien de ratios le clip doit rogner ;
- sans clipping, la part des ratios rognés ne peut valoir qu'une seule valeur ;
- le bonus d'entropie est le seul terme de la loss qui agit directement sur l'entropie ;
- `vf_coef = 0` retire la loss du critique : que devient sa prédiction des retours ?

Puis cherche, dans chaque colonne, le run qui se distingue des autres dès le début (pas seulement à la fin), et garde pour le run sain celui qu'aucune colonne ne désigne.

</details>
<details><summary>Indice 3</summary>

Le critique d'abord, comme modèle : `vf_coef = 0` laisse le critique à son initialisation ; sa prédiction ne suit pas les retours, et la variance expliquée reste près de 0 **pendant tout le run**, pas seulement à la fin (où celle du run sain peut aussi baisser, quand presque tous les épisodes durent 500 pas). Pour les trois autres, regarde la colonne que l'indice 2 associe à chaque réglage et cherche le run dont la courbe y est la plus haute, la plus basse ou constante, dès les premières itérations ; vérifie ensuite qu'aucune autre colonne ne le désigne pour une autre cause.

</details>

### Ex B8.22 — Hyperparamètres de PPO : ε du clip, λ, nombre d'epochs — robustesse sur trois graines 🔬

<details><summary>Indice 1</summary>

`summary_22` ne demande que deux lectures d'un journal ; le piège est le `nan` d'une itération où aucun épisode ne finit (fréquent quand les épisodes durent 500 pas), qui empoisonne une moyenne et ne vaut jamais « au moins 475 ». Pour l'expérience, relis « Rapporter plusieurs graines » (208.6) : une médiane de trois graines et leur intervalle, puis la question de savoir si les intervalles se recouvrent.

</details>
<details><summary>Indice 2</summary>

- `auc` : `np.asarray(logs["mean_return"], dtype=float)`, puis les valeurs finies (`np.isfinite`) et leur moyenne.
- `first_475` : parcours `logs["step"]` et `logs["mean_return"]` ensemble (`zip`) ; le premier retour `>= 475` donne son pas, en `int` ; `None` sinon. Une comparaison avec `nan` vaut toujours `False`.
- Ne modifie pas le journal (pas de `.pop`, pas d'affectation dans ses listes).
- Pour lire l'expérience : la KL et la fraction clippée de chaque run sont dans `runs_22[f"{réglage}/{graine}"]["approx_kl"]` et `["clipfrac"]` ; compare leurs médianes entre `epochs_4`, `reference` et `epochs_20`.

</details>
<details><summary>Indice 3</summary>

```python
def summary_22(logs):
    returns = np.asarray(logs["mean_return"], dtype=float)
    finite = ...                                         # the values that are not nan
    first = next((int(step) for step, r in zip(logs["step"], returns) if ...), None)
    return {"auc": float(finite.mean()), "first_475": first}
```

</details>

### Ex B8.23 — Prédire : l'agent PPO face à une gravité doublée ou à un mât deux fois plus long 🔮

<details><summary>Indice 1</summary>

Relis « CartPole, vu de près » (208.1) et « Généraliser au-delà de son environnement » (208.8). Deux questions séparées : la physique devient-elle plus dure ou plus facile à tenir (pour n'importe quel contrôleur) ? Et l'agent, qui n'a vu qu'une physique, sait-il réagir à une autre ?

</details>
<details><summary>Indice 2</summary>

Pour la physique : un mât plus long a plus d'inertie et tombe plus lentement (pense à tenir un balai et un crayon en équilibre sur ta main) ; une gravité plus forte le fait tomber plus vite. Pour l'agent : sa politique est une fonction de l'état ; elle pousse d'un côté ou de l'autre selon l'angle et les vitesses. Elle peut rester juste tant que la bonne poussée garde le même signe, et devenir fausse quand la dynamique change assez pour qu'il faille pousser autrement (plus tôt, plus longtemps).

</details>
<details><summary>Indice 3</summary>

Pour a), demande-toi si une gravité doublée change le **signe** de la bonne réaction à un mât qui penche, ou seulement sa **vitesse** ; une politique qui réagit à chaque pas (50 fois par seconde) a-t-elle le temps de corriger ? Pour b) et c), pose la même question avec un mât plus lent : ce qui était une bonne poussée au bon moment le reste-t-il quand le mât réagit deux, puis quatre fois plus lentement ? Les tranches de a) et b) portent sur la médiane de six agents ; c) compte les agents, un par un.

</details>

### Ex B8.24 — Robustesse : modifier la physique de CartPole et mesurer la chute du retour 🔬

<details><summary>Indice 1</summary>

Relis la fin de « Généraliser au-delà de son environnement » (208.8) : `gravity` et `length` sont des attributs de l'environnement nu (`env.unwrapped`), mais deux constantes en sont déduites une fois pour toutes dans `__init__`. Lis `CartPoleEnv.__init__` (`inspect.getsource`) pour les trouver. `robustness_24` ne fait qu'une boucle sur les facteurs.

</details>
<details><summary>Indice 2</summary>

- `set_physics_24` : `core = env.unwrapped` ; change `core.gravity` et `core.length` quand ils ne valent pas `None` ; puis recalcule `core.total_mass` et `core.polemass_length` à partir des masses et de la longueur ; rends `env`.
- `robustness_24` : pour chaque facteur, un `wbr.make_env()` neuf, ta `set_physics_24` avec `{name: DEFAULTS_24[name] * factor}` (un dictionnaire passé par `**`), puis `wbr.evaluate(policy, n_episodes, seed, env=env)["mean_return"]` en `float`.

</details>
<details><summary>Indice 3</summary>

```python
def set_physics_24(env, gravity=None, length=None):
    core = env.unwrapped
    # 1. the constants you are given (None: unchanged)
    core.total_mass = core.masspole + core.masscart
    core.polemass_length = ...                           # the other constant derived by __init__
    return env


def robustness_24(policy, name, factors, n_episodes=EPISODES_24, seed=EVAL_SEED):
    out = []
    for factor in factors:
        env = set_physics_24(wbr.make_env(), **{name: DEFAULTS_24[name] * factor})
        ...                                              # one wbr.evaluate per factor
    return out
```

</details>

### Ex B8.25 — bradley_terry_loss et dpo_loss 🔨

<details><summary>Indice 1</summary>

Relis « Le modèle de récompense de Bradley-Terry » et « DPO » (208.7), et ta copie de ✏️ B8.6. Bradley-Terry est une régression logistique sur la **différence** $d = r_c - r_r$ : la loss d'une paire est $-\log\sigma(d)$, et sa dérivée par rapport à $d$ vaut $-(1 - \sigma(d)) = -\sigma(-d)$. DPO est la même loss, appliquée à une marge calculée à partir de quatre log-probabilités.

</details>
<details><summary>Indice 2</summary>

- Valide d'abord : `np.asarray(..., dtype=float)`, des tableaux 1-D de **même** forme (pas de diffusion), $\beta > 0$ pour DPO (un `ValueError` sinon).
- La loss stable : `np.logaddexp(0, -d)` vaut $\ln(1 + e^{-d}) = -\log\sigma(d)$ sans jamais calculer $e^{1000}$ ; sa moyenne.
- Le gradient : $\sigma(-d)$ = `np.exp(-np.logaddexp(0, d))`, toujours entre 0 et 1 ; `grad_chosen = -σ(-d) / n`, `grad_rejected = -grad_chosen`.
- DPO : `margins = beta * ((pc - rc) - (pr - rr))`, puis la même loss sur les marges.

</details>
<details><summary>Indice 3</summary>

```python
def bradley_terry_loss(r_chosen, r_rejected):
    # 1. float arrays, the same shape (n,) (ValueError otherwise)
    d = rc - rr
    loss = float(np.mean(np.logaddexp(0.0, -d)))
    grad_chosen = ...                                    # -sigma(-d) / n, computed with logaddexp
    return loss, grad_chosen, -grad_chosen
```

</details>

### Ex B8.26 — RLHF miniature : un modèle de récompense appris sur des préférences simulées, puis PPO — et le piratage de la récompense 📦

<details><summary>Indice 1</summary>

Le modèle de récompense note un **état** ; une trajectoire reçoit la somme des notes de ses états, et Bradley-Terry compare deux sommes. Ta `bradley_terry_loss` (NumPy) rend la loss **et** son gradient par rapport aux deux sommes : il reste à pousser ce gradient dans le réseau, ce que fait `torch.autograd.backward(tenseurs, gradients)` (la règle de la chaîne du ch. 18). Pour l'expérience, relis « Le piratage de la récompense » (208.7) : un agent optimise la récompense qu'on lui donne, là où elle a été apprise… et ailleurs.

</details>
<details><summary>Indice 2</summary>

- `make_reward_model_26` : un `nn.Module` qui garde l'échelle (`register_buffer("scale", torch.tensor(wbr.OBS_SCALE))`) et un `nn.Sequential` ; `forward` divise, applique le réseau et rend `.squeeze(-1)`.
- `trajectory_returns_26` : une seule passe sur toutes les observations (`np.concatenate`), puis une somme par trajectoire (`torch.tensor_split` aux longueurs cumulées, ou `index_add`) ; pas de `.detach()`.
- `train_reward_model_26` : `torch.where(gagnant_premier, r1, r2)` choisit le préféré de chaque paire ; `bradley_terry_loss` sur leurs valeurs détachées en float64 ; puis `optimizer.zero_grad()`, `torch.autograd.backward([chosen, rejected], [g_chosen, g_rejected])` (des tenseurs du type des retours), `optimizer.step()`.
- `reward_fn_26` : une fonction interne qui appelle le modèle sous `torch.no_grad()` et rend `.numpy()`.

</details>
<details><summary>Indice 3</summary>

```python
def train_reward_model_26(pairs, seed, epochs=EPOCHS_26, lr=LR_26):
    torch.manual_seed(seed)
    model = make_reward_model_26()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    first_wins = torch.as_tensor(np.asarray(pairs["preferred"]) == 0)
    losses = []
    for _ in range(epochs):
        r1, r2 = trajectory_returns_26(model, pairs["first"]), trajectory_returns_26(model, pairs["second"])
        chosen, rejected = torch.where(first_wins, r1, r2), torch.where(first_wins, r2, r1)
        loss, g_c, g_r = mylearn.deep_rl.bradley_terry_loss(...)       # NumPy, float64
        # zero_grad, torch.autograd.backward([chosen, rejected], [...]), step
        losses.append(float(loss))
    return model, losses
```

</details>

### Ex B8.27 — Revue de code : comparer ton PPO à une implémentation de référence et documenter chaque écart 🛠️

<details><summary>Indice 1</summary>

Lis `ppo.py` dans l'ordre où il s'exécute (de la ligne 129 à la fin), avec ton `train_ppo_20` ouvert à côté, et compare bloc par bloc : la création des environnements, la boucle de collecte, GAE, l'aplatissement du batch, les epochs et mini-batches, la loss, les journaux. Puis parcours les « 37 détails » (les 13 premiers, ceux de base) : pour chacun, `ppo.py` et toi faites-vous la même chose ?

</details>
<details><summary>Indice 2</summary>

Les deux écarts de la fiche se trouvent là où `ppo.py` calcule `next_done` et là où il cherche les retours des épisodes dans `infos`. Pour les autres, cherche dans `ppo.py` : la loss du critique (`clip_vloss`), ce qui est écrit dans les journaux et pour quel mini-batch, `target_kl`, la taille des mini-batches et ce qu'ils contiennent, les réglages par défaut (`Args`), la source du hasard du mélange. Un détail identique chez toi et chez lui (l'init orthogonale, `eps=1e-5`) n'est pas un écart. Pour chaque écart, l'« effet » dit **dans quel cas** il compte (CartPole, un autre environnement, la lecture des journaux).

</details>
<details><summary>Indice 3</summary>

Le premier écart, comme modèle de forme (à toi d'écrire son contenu) :

```python
{"sujet": "fin d'épisode et limite de temps",
 "lignes": ...,                       # the line of ppo.py where next_done is computed
 "extrait": "next_done = ...",        # copied from that line
 "chez_moi": "...",                   # what your compute_gae does with terminated and truncated
 "effet": "...",                      # what a time limit of 500 steps becomes in ppo.py
 "decision": "garder"},
```

Pour les suivants, une ligne de `ppo.py` par écart, l'extrait recopié de cette ligne, et une phrase de chaque côté.

</details>

### Ex B8.28 — Défi : CartPole-v1 résolu (≥ 475 sur 100 épisodes) en au plus 100 000 pas, trois graines sur trois 🏆

<details><summary>Indice 1</summary>

Ton PPO de B8.20 est déjà un bon point de départ : relis tes notes de B8.22 (quel réglage apprend vite, lequel apprend **sûrement**) et les journaux de B8.21. Le budget de 100 000 pas est celui du RL Zoo pour PPO sur CartPole (208.6) ; la difficulté n'est pas d'y arriver une fois, mais sur trois graines.

</details>
<details><summary>Indice 2</summary>

- `train_28` peut appeler `train_ppo_20(seed, budget, réglages)` : le nombre d'itérations, `budget // (n_envs × n_steps)`, ne dépasse jamais le budget.
- Un learning rate qui décroît jusqu'à 0 finit sur une politique stable ; la politique rendue est évaluée de façon déterministe (l'action la plus probable).
- Toute évaluation pendant l'entraînement coûte des pas du budget : si tu en fais, compte-les.

</details>
<details><summary>Indice 3</summary>

```python
SETTINGS_28 = {**SETTINGS_20, ...}            # your choice, from B8.22


def train_28(seed, budget):
    # 1. your PPO of B8.20 with SETTINGS_28, on `budget` steps (its number of iterations never goes beyond)
    # 2. return the agent: a module, evaluated by its most probable action
```

Essaie d'abord tes réglages sur une graine à toi (pas une de `SEEDS_28`), évaluée par `wbr.evaluate(agent, 100, EVAL_SEED)`, pour voir le temps et le retour ; garde les graines d'évaluation du défi pour la cellule de vérification.

</details>
