# B6 · Explicabilité, équité et éthique — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B6.Q1 — Interprétable par construction ou expliqué après coup ?

<details><summary>Indice 1</summary>

Relis « Interprétable par construction, ou expliqué après coup » (§206.1).

</details>
<details><summary>Indice 2</summary>

Un modèle interprétable par construction **est** son explication : on le lit lui-même. Une explication post hoc s'ajoute à un modèle déjà entraîné et l'approche. Une méthode agnostique n'utilise que les prédictions ; une méthode spécifique lit l'intérieur d'une famille de modèles. Un arbre binaire double son nombre de feuilles à chaque niveau.

</details>
<details><summary>Indice 3</summary>

a) **B** : six coefficients de features standardisées se lisent directement ; la forêt et le réseau ont besoin d'une explication ajoutée. Pour b), une explication post hoc **approche**-t-elle ou **reproduit**-elle le modèle ? Pour c), de quoi Grad-CAM a-t-il besoin pour calculer sa carte ? Pour d), que fait-on au modèle pour mesurer une importance par permutation ? Pour e), compte les feuilles niveau par niveau : 2, puis 4, puis…

</details>

### B6.Q2 — Explication globale ou locale : quel outil répond à quelle question ?

<details><summary>Indice 1</summary>

Relis « Global ou local » (§206.1) et son tableau, puis « L'importance par permutation » (§206.2).

</details>
<details><summary>Indice 2</summary>

Une question sur **une** prédiction appelle une explication locale ; une question sur **tout** le modèle, une explication globale. Une importance moyenne peut être faible pour une feature rare qui décide pourtant de quelques cas. L'importance par permutation est le score de départ moins la **moyenne** des scores après mélange.

</details>
<details><summary>Indice 3</summary>

a) **C** : on demande pourquoi **ce** dossier a été refusé, une question locale. Pour b), quel outil du tableau répond à « quelles features, en tout ? ». Pour c), demande-toi ce que devient l'importance moyenne d'une feature qui ne vaut 1 que pour un client sur 200. Pour d), une moyenne de valeurs locales décrit-elle un client ou le modèle ? Pour e), calcule la moyenne des cinq scores mélangés, puis retire-la de 0,83.

</details>

### B6.Q3 — Dépendance partielle et courbes ICE : ce que montre la moyenne, ce qu'elle cache

<details><summary>Indice 1</summary>

Relis « La dépendance partielle », « Les courbes ICE » et le piège classique « une dépendance partielle hors des données » (§206.2).

</details>
<details><summary>Indice 2</summary>

$\operatorname{PD}_j(v) = \frac{1}{n}\sum_i f(v, \mathbf{x}_{i,\bar j})$ : on impose $v$ à **tous** les exemples, les autres features gardent leurs valeurs, puis on fait la moyenne des **prédictions**. Une courbe ICE est la même chose pour un seul exemple. `centered=True` retire à chaque courbe sa valeur au début de la grille.

</details>
<details><summary>Indice 3</summary>

a) **C** : $\operatorname{PD}_1(v) = \frac{1}{n}\sum_i v\,x_{i,2} = v \times (\text{moyenne de } x_2) = v \times 0$. Pour b), écris la courbe ICE d'un exemple où $x_2 = 1$, puis d'un exemple où $x_2 = -1$. Pour c), calcule $3 \times 2 + \frac{1}{3}(0^2 + 1^2 + 2^2)$. Pour d), que deviennent deux droites parallèles quand on retire à chacune sa valeur en un même point ? Pour e), relis la formule : touche-t-on au nombre de pièces de chaque logement ?

</details>

### B6.Q4 — Les quatre propriétés de Shapley : efficacité, symétrie, joueur nul, additivité

<details><summary>Indice 1</summary>

Relis « Quatre propriétés, une seule solution » et l'efficacité avec un fond (§206.3).

</details>
<details><summary>Indice 2</summary>

Efficacité : $\sum_i \phi_i = f(\mathbf{x}) - v(\varnothing)$. Additivité : les parts d'une somme de modèles s'ajoutent. Symétrie : deux joueurs qui apportent **toujours** la même chose à toute coalition ; cela dépend des valeurs du client, pas seulement de la forme de $f$. Une feature est dans une coalition ou non.

</details>
<details><summary>Indice 3</summary>

a) $\phi_3 = 0{,}72 - 0{,}40 - 0{,}15 - (-0{,}05) = 0{,}22$ : attention au signe de $\phi_2$. Pour b), quelle propriété parle d'un joueur qui n'apporte jamais rien ? Pour c), ajoute les deux parts. Pour d), essaie $f = x_1 + x_2$ avec un fond d'une ligne $(0, 0)$ et calcule $\phi_1$ et $\phi_2$. Pour e), chacune des 4 autres features est dans la coalition ou non.

</details>

### B6.Q5 — Changer le fond (background) change les valeurs SHAP : vrai ou faux

<details><summary>Indice 1</summary>

Relis « Les features comme joueurs », « Le cas linéaire » et « Le fond fait partie de la question » (§206.3), puis les deux modes de `TreeExplainer` (§206.4).

</details>
<details><summary>Indice 2</summary>

La somme des $\phi_i$ vaut $f(\mathbf{x})$ moins la prédiction **moyenne sur le fond**. Pour un modèle linéaire, $\phi_i = w_i(x_i - \bar b_i)$, où $\bar b_i$ est la moyenne de la feature sur le fond. `maskers.Independent` a un `max_samples` par défaut.

</details>
<details><summary>Indice 3</summary>

a) **Faux** : la somme vaut $f(\mathbf{x}) - \mathbb{E}[f(\mathbf{b})]$, et le second terme change avec le fond. Pour b), que vaut la moyenne de $f$ sur un fond d'une ligne ? Pour c) et d), calcule $2 \times (4 - \bar b_1)$ avec la moyenne de chaque fond. Pour e), un mode suit les branches au prorata de l'entraînement, l'autre utilise un fond : est-ce la même question ? Pour f), relis la valeur par défaut de `max_samples`.

</details>

### B6.Q6 — Features corrélées : ce que SHAP répartit et ce qu'il ne dit pas

<details><summary>Indice 1</summary>

Relis « Features corrélées : ce que SHAP répartit » (§206.4) et le cas linéaire (§206.3).

</details>
<details><summary>Indice 2</summary>

Le Shapley interventionnel est fidèle au **modèle** : il crédite ce que le modèle lit, avec $\phi_i = w_i(x_i - \bar b_i)$ pour un modèle linéaire. L'observationnel est fidèle aux **données**. Entre copies, la somme est stable, le partage dépend de la façon dont le modèle les a départagées.

</details>
<details><summary>Indice 3</summary>

a) $\phi_1 = 2 \times (1{,}5 - 0{,}5) = 2$. Pour b), le modèle B lit-il $x_2$ ? Pour c), calcule $\phi_1 + \phi_2$ pour chaque modèle avec la même formule. Pour d), le Shapley observationnel tire les features absentes **sachant** celles de la coalition : si $x_1$ est absente mais $x_2$ présente, que devine-t-il de $x_1$ ? Pour e), quelle quantité reste la même d'un modèle à l'autre ?

</details>

### B6.Q7 — Gradient, gradient × entrée, gradients intégrés : pourquoi une ligne de référence ?

<details><summary>Indice 1</summary>

Relis « Le problème de la saturation », « Les gradients intégrés » et « Le choix de la ligne de référence » (§206.5).

</details>
<details><summary>Indice 2</summary>

$f(x) = \min(x, 1)$ a une pente de 1 sous 1 et de 0 au-delà. $\operatorname{IG}(x) = (x - x')\int_0^1 f'(x' + \alpha(x - x'))\,d\alpha$ : la moyenne de la pente le long du chemin, multipliée par le chemin parcouru. Par la complétude, la somme des attributions vaut $f(x) - f(x')$.

</details>
<details><summary>Indice 3</summary>

a) $x = 3$ est au-delà de 1, où $f$ est plate : $f'(3) = 0$. Pour b), le chemin de 0 à 3 passe un tiers de son parcours sous 1 : multiplie 3 par la pente moyenne, ou utilise la complétude. Pour c), sur quelle partie de $f$ se trouve le chemin de 2 à 3 ? Pour d), le gradient d'un modèle linéaire est-il le même en tout point du chemin ? Pour e), que vaut $x_i - x'_i$ pour un pixel du fond ?

</details>

### B6.Q8 — Grad-CAM : pourquoi la dernière couche convolutive ?

<details><summary>Indice 1</summary>

Relis « L'idée », « Pourquoi la dernière couche convolutive ? » et les « Quatre détails » (§206.6).

</details>
<details><summary>Indice 2</summary>

Sur un ResNet de torchvision, `layer4` divise la taille de l'image par 32. Après la dernière couche convolutive, le réseau résume chaque carte en un nombre. Les poids $\alpha_c$ peuvent être négatifs, et c'est le ReLU qui ne garde que le positif. Le gradient se prend sur un score avant la softmax.

</details>
<details><summary>Indice 3</summary>

a) $160 / 32 = 5$ : des cartes de $5 \times 5$. Pour b), qu'est-ce qui efface la position dans un ResNet ? Pour c), divise 224 par 4. Pour d), une somme pondérée avec des poids négatifs peut-elle être négative ? Pour e), relis la phrase qui suit la formule de Grad-CAM : « $y^k$ est le score de la classe $k$… ».

</details>

### B6.Q9 — Une carte de saillance qui ne dépend pas du modèle : que prouve le test de randomisation ?

<details><summary>Indice 1</summary>

Relis « Les tests de santé » et le paragraphe sur ce que prouve un test réussi (§206.7).

</details>
<details><summary>Indice 2</summary>

L'exigence : une explication censée décrire le modèle doit **changer quand le modèle change**. Réussir le test est nécessaire, pas suffisant. La randomisation en cascade et la randomisation indépendante ne randomisent pas les mêmes couches. Une corrélation de rangs ne compare que l'**ordre** des valeurs. Un réseau entraîné sur des labels mélangés n'apprend rien de transférable au test.

</details>
<details><summary>Indice 3</summary>

a) **B** : si la carte ne bouge pas quand le réseau ne sait plus rien, elle ne dépend pas de ce qu'il a appris. Pour b), relis ce qu'un test **réussi** prouve, et ce qu'il ne prouve pas. Pour c), laquelle garde randomisées les couches déjà traitées ? Pour d), deux cartes $(1, 2, 3)$ et $(10, 20, 30)$ ont-elles le même ordre ? les mêmes valeurs ? Pour e), avec dix classes, quelle accuracy obtient-on en répondant au hasard ?

</details>

### B6.Q10 — Sources de biais : données, labels, variables de substitution, boucles de rétroaction

<details><summary>Indice 1</summary>

Relis « D'où vient un biais » (§206.8).

</details>
<details><summary>Indice 2</summary>

Pour chaque cas, demande-toi **où** est le défaut : dans la composition des données (qui est représenté), dans la cible (qui l'a produite), dans une mesure (quel instrument), dans une colonne qui porte un attribut à sa place, ou dans des décisions qui fabriquent les données futures.

</details>
<details><summary>Indice 3</summary>

a) **A** : 90 % de voix d'hommes, c'est un groupe mal représenté dans les données d'entraînement. Pour b), la note des managers est-elle une feature ou la cible ? Pour c), qui choisit les dossiers qui entrent dans l'entraînement ? Pour d), une colonne qui n'est pas le sexe peut-elle en porter l'information ? Pour e), qu'est-ce qui est faux : l'instrument, la population, ou les décisions ?

</details>

### B6.Q11 — Parité démographique, égalité des chances, calibration : peut-on tout avoir ?

<details><summary>Indice 1</summary>

Relis « Les critères » et « L'impossibilité » (§206.9).

</details>
<details><summary>Indice 2</summary>

Un classifieur parfait sélectionne exactement les positifs : son taux de sélection vaut le taux de base, son TPR 1, son FPR 0, sa precision 1. La relation de Chouldechova : $\operatorname{FPR} = \frac{p}{1-p}\cdot\frac{1 - \operatorname{PPV}}{\operatorname{PPV}}\cdot(1 - \operatorname{FNR})$. Le choix d'un critère dépend de l'erreur qui coûte le plus, et à qui.

</details>
<details><summary>Indice 3</summary>

a) Les taux de sélection valent les taux de base, 0,6 et 0,25 : $\Delta_{\text{DP}} = 0{,}6 - 0{,}25 = 0{,}35$. Pour b) et c), écris le TPR, le FPR et la precision du classifieur parfait dans chaque groupe. Pour d), calcule $\frac{0{,}5}{0{,}5} \times \frac{0{,}2}{0{,}8} \times 0{,}8$. Pour e), l'erreur coûteuse touche des positifs (ils rembourseraient) : quel taux compte les positifs reconnus ?

</details>

### B6.Q12 — AI Act : dans quelle catégorie de risque tombe un scoring de crédit ?

<details><summary>Indice 1</summary>

Relis « L'AI Act » et l'encadré 🕰️ sur la régulation (§206.11).

</details>
<details><summary>Indice 2</summary>

Quatre niveaux : pratiques interdites (art. 5), haut risque (annexe III, dont le point 5 b) et son exception), transparence (art. 50), risque minimal. Le fournisseur développe le système ; le déployeur l'utilise et réalise l'analyse d'impact de l'art. 27. Le règlement 2026/1744 a déplacé des dates.

</details>
<details><summary>Indice 3</summary>

a) **B** : le point 5 b) vise l'évaluation de « la solvabilité des personnes physiques ». Pour b), relis l'exception que contient ce même point. Pour c), la notation sociale est-elle dans la liste de l'art. 5 ? Pour d), quelle date l'encadré 🕰️ donne-t-il pour l'annexe III ? Pour e), relis qui l'art. 27 désigne. Pour f), relis les mots exacts du point 5 b) : de quelles personnes évalue-t-il la solvabilité ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### B6.R1 — Ch. 21 : visualiser ce qu'a appris un CNN, et l'attaque FGSM

<details><summary>Indice 1</summary>

Relis, au ch. 21, le rappel maths « le gradient par rapport à l'entrée » et la section sur l'attaque FGSM.

</details>
<details><summary>Indice 2</summary>

FGSM : $\mathbf{x}_{\text{adv}} = \mathbf{x} + \varepsilon\,\operatorname{sign}(\nabla_{\mathbf{x}} L)$, puis on ramène chaque pixel dans sa plage ; $\operatorname{sign}$ vaut $+1$, $0$ ou $-1$. Pour un score linéaire, la perturbation $\varepsilon\,\operatorname{sign}(\mathbf{w})$ change le score de $\varepsilon\sum_j \lvert w_j \rvert$. La maximisation d'activation est une **montée** de gradient sur l'image.

</details>
<details><summary>Indice 3</summary>

a) $0{,}40 - 0{,}03 = 0{,}37$ ; $0{,}98 + 0{,}03 = 1{,}01$, ramené à $1{,}00$ ; $\operatorname{sign}(0) = 0$, donc $0{,}10$ ne bouge pas : **[0.37, 1.0, 0.1]**. Pour b), compare les deux calculs : quelle quantité dérive-t-on, par rapport à quoi, et que garde-t-on du gradient ? Pour c), qu'est-ce qui reste figé pendant la visualisation ? Pour d), calcule $0{,}01 \times 3\,072 \times 0{,}01$. Pour e), une **montée** cherche-t-elle un minimum ?

</details>

### B6.R2 — Ch. 14 : feature_importances_ contre importance par permutation

<details><summary>Indice 1</summary>

Relis, au ch. 14, le paragraphe « L'importance des features » et l'encadré 🕰️ « peut-on se fier à `feature_importances_` ? ».

</details>
<details><summary>Indice 2</summary>

L'importance MDI additionne les baisses d'impureté de l'**entraînement**, normalisées ; elle favorise les features à forte cardinalité. L'importance par permutation mesure la baisse du score sur la **validation** quand on mélange la colonne.

</details>
<details><summary>Indice 3</summary>

a) **A** : un identifiant offre un seuil différent à chaque ligne, et un arbre profond s'en sert pour séparer les exemples d'entraînement. Pour b), sur quelles données la MDI est-elle calculée ? Pour c), relis la normalisation de `feature_importances_`. Pour d), mélanger un identifiant change-t-il les prédictions sur des exemples jamais vus d'une façon qui compte ? Pour e), multiplie la baisse d'accuracy par 400.

</details>

### B6.R3 — Ch. 3 : TPR, FPR et precision à partir d'une matrice de confusion

<details><summary>Indice 1</summary>

Relis, au ch. 3, la matrice de confusion et le tableau des mesures (TPR, FPR, precision), avec la disposition de `confusion_matrix`.

</details>
<details><summary>Indice 2</summary>

`[[TN, FP], [FN, TP]]` : la ligne est la vérité, la colonne la prédiction. TPR $= \frac{TP}{TP + FN}$ (la ligne 1), FPR $= \frac{FP}{FP + TN}$ (la ligne 0), precision $= \frac{TP}{TP + FP}$ (la colonne 1), taux de sélection $= \frac{TP + FP}{n}$.

</details>
<details><summary>Indice 3</summary>

a) TPR $= \frac{40}{40 + 10} = 0{,}80$. Pour b), divise les faux positifs par les négatifs réels de la ligne 0. Pour c), divise TP par la colonne « prédit 1 ». Pour d), divise la colonne « prédit 1 » par les 150 exemples. Pour e), la case `[0, 1]` est la ligne « vérité 0 », colonne « prédit 1 ».

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B6.1 — Valeurs de Shapley exactes à la main : trois features, huit coalitions ✏️

<details><summary>Indice 1</summary>

Relis « Les features comme joueurs », l'exemple à deux features et l'encadré 🧮 sur les poids de Shapley (§206.3).

</details>
<details><summary>Indice 2</summary>

Pour chaque masque $S$ (huit, de $\varnothing$ à $\{1, 2, 3\}$), construis les deux lignes : les features de $S$ prennent les valeurs du client, les autres celles de la ligne du fond ; calcule $f$ sur chacune, puis la **moyenne**. Avec $M = 3$, une coalition de taille 0 ou 2 pèse $\frac{1}{3}$, une coalition de taille 1 pèse $\frac{1}{6}$. $\phi_i$ est la somme pondérée des quatre contributions marginales $v(S \cup \{i\}) - v(S)$, pour les quatre $S$ sans $i$.

</details>
<details><summary>Indice 3</summary>

a) $v(\varnothing) = \frac{1}{2}\big[f(0, 2, 0) + f(2, 2, 2)\big] = \frac{1}{2}(2 + 10) = 6$. Pour b), les deux lignes sont $(2, 2, 0)$ et $(2, 2, 2)$. Pour c), $(2, 2, 3)$ deux fois. Pour d), applique $\frac{|S|!\,(M - |S| - 1)!}{M!}$ avec $|S| = 1$. Pour e) à g), pose pour chaque feature ses quatre contributions et leurs poids ($\frac13, \frac16, \frac16, \frac13$) ; contrôle par l'efficacité. Pour h), retire de $\phi_1$ la part du terme $2x_1$ (soit $2(x_1 - \bar b_1)$) et compare ce qui reste à $\phi_3$. Pour i), multiplie le nombre de coalitions par le nombre de lignes du fond. Pour j), refais les huit $v(S)$ avec la seule ligne $(1, 2, 1)$ (ou la formule de ∂ B6.2, question 8). Pour k), relis l'encadré 📐 de la fiche.

</details>

### Ex B6.2 — Modèle linéaire et fond indépendant : φ_i = w_i (x_i − E[x_i]) ∂

<details><summary>Indice 1</summary>

Relis « Les features comme joueurs » et « Le cas linéaire » (§206.3). Le calcul tient en trois lignes une fois $v(S)$ écrit.

</details>
<details><summary>Indice 2</summary>

Pour une ligne $\mathbf{b}$ : $f(\mathbf{x}_S, \mathbf{b}_{\bar S}) = w_0 + \sum_{j \in S} w_j x_j + \sum_{j \notin S} w_j b_j$. La moyenne sur le fond ne touche que le dernier terme. La contribution marginale de $i$ remplace un terme $w_i \bar b_i$ par $w_i x_i$. Pour l'interaction, écris les quatre $v(S)$ avec une seule ligne et applique les poids $\frac12$.

</details>
<details><summary>Indice 3</summary>

1. $v(S) = w_0 + \sum_{j \in S} w_j x_j + \sum_{j \notin S} w_j \bar b_j$. Pour 2, fais la différence avec $S \cup \{i\}$ : il ne reste qu'un terme, qui ne dépend pas de $S$. Pour 3, sors ce terme de la somme et applique (ii). Pour 4, prends $S = \varnothing$, puis additionne les $\phi_i$. Pour 5, calcule les moyennes des colonnes $(2, 1, 3)$ et applique la formule. Pour 6, de quoi la formule dépend-elle dans le fond ? Pour 7, sur quelle sortie $f$ est-il de la forme $w_0 + \mathbf{w}^\top\mathbf{x}$ ? Pour 8, $v(\varnothing) = b_1b_2$, $v(\{1\}) = x_1b_2$, $v(\{2\}) = b_1x_2$, $v(\{1, 2\}) = x_1x_2$ ; écris $\phi_1 = \frac12[(v(\{1\}) - v(\varnothing)) + (v(\{1, 2\}) - v(\{2\}))]$ et factorise.

</details>

### Ex B6.3 — Gradients intégrés à la main sur f(x) = x₁² + 3x₂ et vérification de la complétude ✏️

<details><summary>Indice 1</summary>

Relis « Les gradients intégrés », l'encadré 🧮 « approcher une intégrale par une somme » et le paragraphe sur les conventions de ta `integrated_gradients` (§206.5), puis l'encadré 🕰️ sur captum (§206.6).

</details>
<details><summary>Indice 2</summary>

$\nabla f = (2x_1, 3)$. Le long du chemin depuis $\mathbf{0}$, au point $\alpha\mathbf{x}$, la première dérivée vaut $2\alpha x_1$ : c'est une fonction **linéaire** de $\alpha$, et $\operatorname{IG}_1 = x_1 \times \int_0^1 2\alpha x_1\,d\alpha$. La seconde dérivée est constante. Pour une règle, $\operatorname{IG}_1 = x_1 \times \sum_k (\text{poids}_k \times 2\alpha_k x_1)$. Depuis une autre référence, le point du chemin est $\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}')$. captum prend $N$ points et des poids $1/N$, divisés par deux aux extrémités.

</details>
<details><summary>Indice 3</summary>

a) $\nabla f(2, -1) = (2 \times 2,\ 3) =$ **[4, 3]**. Pour b), multiplie chaque composante par $x_i$. Pour c), calcule $f(2, -1) - f(0, 0)$. Pour d), intègre $2\alpha x_1$ de 0 à 1, multiplie par $x_1$ ; la seconde vaut $3 \times x_2$. Pour e), les points $\alpha = 0 ;\ 0{,}2 ;\ 0{,}4 ;\ 0{,}6 ;\ 0{,}8$, la moyenne de $4\alpha$, multipliée par 2. Pour f), les points $0{,}2$ à $1$. Pour g), la moyenne de e) et de f) est-elle la règle des trapèzes ? Vérifie-le avec les poids. Pour h), écris l'erreur de la règle de gauche en fonction de $n$ et résous « erreur $\le 0{,}03$ ». Pour i), intègre $2(1 + \alpha)$ de 0 à 1, multiplie par $x_1 - x'_1$. Pour j), la complétude donne la somme. Pour k), six points $0, 0{,}2, \dots, 1$, poids $\frac16$ divisés par deux aux extrémités, puis multiplie par $x_1$. Pour l), `n_steps` compte des intervalles.

</details>

### Ex B6.4 — Grad-CAM à la main : deux cartes 3×3 et leurs gradients ✏️

<details><summary>Indice 1</summary>

Relis « L'idée », « Un exemple à la main » et le paragraphe sur ta `grad_cam` (§206.6), puis le détail « L'alignement ».

</details>
<details><summary>Indice 2</summary>

$\alpha_c$ est la **moyenne** des neuf gradients du canal $c$. La carte est $\alpha_1 A^1 + \alpha_2 A^2$, case par case ; le ReLU met à 0 les cases négatives ; la normalisation divise par la plus grande valeur absolue de la carte. Un demi-pixel de la couche vaut la moitié de la taille d'une case une fois la carte agrandie.

</details>
<details><summary>Indice 3</summary>

a) La somme des neuf gradients du canal 1 vaut $0 + 1 + 2 + 1 + 1 + 1 + 2 + 1 + 0 = 9$, donc $\alpha_1 = 9/9 = 1{,}0$. Pour b), fais de même avec le canal 2 (attention aux signes). Pour c), calcule $\alpha_1 \times 0 + \alpha_2 \times 2$. Pour d), écris la carte entière et compte ses cases négatives. Pour e), divise la case du centre par le maximum de la carte après ReLU. Pour f), compare les trois coins. Pour g), même calcul que c), divisé par la plus grande valeur absolue de la carte **sans** ReLU. Pour h), calcule $-0{,}5\,A^1 + A^2$ dans les trois coins. Pour i), les $\alpha_c$ ne lisent que les gradients : que deviennent-ils, puis la carte et son maximum ? Pour j), divise 72 par 3, puis par 2.

</details>

### Ex B6.5 — Audit à la main : deux groupes, deux matrices de confusion, cinq indicateurs ✏️

<details><summary>Indice 1</summary>

Relis « Des taux par groupe », « Les critères » et « Un audit à la main » (§206.9), puis « Après l'entraînement » (§206.10).

</details>
<details><summary>Indice 2</summary>

Dans chaque groupe : $n = TP + FN + FP + TN$ ; taux de base $\frac{TP + FN}{n}$ ; sélection $\frac{TP + FP}{n}$ ; TPR $\frac{TP}{TP + FN}$ ; FPR $\frac{FP}{FP + TN}$ ; precision $\frac{TP}{TP + FP}$ ; accuracy $\frac{TP + TN}{n}$. Avec deux groupes, un écart est la différence des deux taux ; $\Delta_{\text{EO}}$ est le plus grand des écarts de TPR et de FPR. Règle des quatre cinquièmes : le rapport des taux de sélection, comparé à 0,8.

</details>
<details><summary>Indice 3</summary>

a) Les femmes : $n = 72 + 48 + 19 + 361 = 500$, taux de base $\frac{72 + 48}{500} = 0{,}24$. Pour b), $\frac{72 + 19}{500}$. Pour c) et d), applique les formules du TPR (femmes) et du FPR (hommes, $n = 640$). Pour e), $\frac{72}{72 + 19}$. Pour f), calcule le taux de sélection des hommes et fais la différence. Pour g), calcule les deux écarts et garde le plus grand. Pour h), divise b) par le taux de sélection des hommes. Pour i), compare h) à 0,8. Pour j), calcule les deux accuracies. Pour k), multiplie le TPR des hommes par le nombre de candidates qualifiées, puis retire les 72 déjà sélectionnées.

</details>

### Ex B6.6 — Impossibilité : taux de base différents, calibration et égalité des erreurs incompatibles ∂

<details><summary>Indice 1</summary>

Relis « L'impossibilité » (§206.9) et la règle de Bayes du ch. 4. Tout part des définitions : il suffit d'exprimer $TP$, $FN$ et $FP$ avec $p$, $n$, le FNR et la PPV.

</details>
<details><summary>Indice 2</summary>

Positifs réels : $pn$ ; négatifs réels : $(1 - p)n$. $TP = (1 - \operatorname{FNR})\,pn$. De $\operatorname{PPV} = \frac{TP}{TP + FP}$ on tire $FP = TP\,\frac{1 - \operatorname{PPV}}{\operatorname{PPV}}$. Par Bayes, $\operatorname{PPV} = \frac{\operatorname{TPR}\,p}{\operatorname{TPR}\,p + \operatorname{FPR}\,(1 - p)}$. Pour la question 7, écris la PPV avec le FPR et le FNR, et regarde-la comme une fonction de $p$.

</details>
<details><summary>Indice 3</summary>

1. $pn$ positifs et $(1 - p)n$ négatifs ; $TP = (1 - \operatorname{FNR})\,pn$ et $FN = \operatorname{FNR}\,pn$. Pour 2, isole $FP$ dans la définition de la PPV. Pour 3, $\operatorname{FPR} = \frac{FP}{(1 - p)n}$ : remplace $FP$, puis $TP$. Pour 4, inverse l'expression de Bayes : $\frac{1}{\operatorname{PPV}} - 1 = \dots$ Pour 5, prends $p$, PPV et FNR dans tes réponses de ✏️ B6.5. Pour 6, écris le rapport des deux FPR : quels facteurs se simplifient ? Pour 7, divise numérateur et dénominateur de la PPV par $p$ : comment varie $\frac{1 - p}{p}$ ? Pour 8, examine le cas $p_A = p_B$, puis, si les cotes diffèrent, ce que la question 6 impose à deux FPR égaux, et ce que la relation de 3 en déduit pour la PPV. Pour 9, quel critère chacun mesurait-il, et que dit la question 6 quand les taux de base diffèrent ?

</details>

### Ex B6.7 — Fermi : coût de Shapley exact, de KernelSHAP et de TreeSHAP pour 10, 20 et 50 features 🧮

<details><summary>Indice 1</summary>

Relis « Combien ça coûte » (§206.3) et la présentation de KernelSHAP et de TreeSHAP (§206.4).

</details>
<details><summary>Indice 2</summary>

Calcul exact : $2^M \times B$ prédictions par ligne. KernelSHAP : $\min(2M + 2\,048,\ 2^M - 2)$ coalitions, chacune sur $B$ lignes. TreeSHAP interventionnel : $T \times L \times B$ pas par ligne. Temps = nombre de prédictions / 150 000 par seconde. Une année compte environ $3{,}2 \times 10^7$ s. Pour les permutations, $M!$ ordres de 140 octets.

</details>
<details><summary>Indice 3</summary>

1. $M = 10$ : $2^{10} \times 50 = 51\,200$ prédictions, soit environ 0,34 s ; refais le calcul pour 20 et 50, puis multiplie par 1 000 lignes. Pour 2, résous $2^M \times 50 / 150\,000 > 60$. Pour 3, calcule $2M + 2\,048$ et compare à $2^M - 2$ ; la part vue est le nombre de coalitions tirées divisé par $2^M$. Pour 4, calcule $100 \times 31 \times 50$ pas, multiplie par $10^{-8}$ s. Pour 5, $10!$ et $12!$, puis $12! \times 140$ octets. Pour 6, $20 \times 2^8 \times 50$ prédictions. Pour 7, pense au nombre de features, au type de modèle et au besoin d'exactitude.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B6.8 — Expliquer une valeur SHAP à un client dont le prêt est refusé, en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis le waterfall (§206.4), « Deux autres explications locales : substitut et contrefactuel » (§206.4) et « Le RGPD, article 22 » (§206.11).

</details>
<details><summary>Indice 2</summary>

Traduis chaque élément : la valeur de base devient « un dossier moyen », chaque barre devient « ce qui a fait monter ou baisser le risque estimé ». Garde les deux ou trois plus grandes barres. Un contrefactuel prudent : « avec un endettement plus bas, le risque estimé aurait été plus bas », sans promettre un seuil précis si tu ne l'as pas calculé.

</details>
<details><summary>Indice 3</summary>

Commence par : « Votre demande a été refusée parce que notre modèle a estimé à 41 % le risque que le prêt ne soit pas remboursé, quand nous acceptons jusqu'à 30 %. » Puis : les deux éléments qui ont le plus pesé (endettement, ancienneté) ; ce qui aurait pu changer la décision ; que c'est le raisonnement d'un modèle sur des données, pas un jugement sur la personne ; le réexamen par un conseiller.

</details>

### Ex B6.9 — Scoring de crédit : le code postal est-il une variable de substitution ? Droits du client, obligations de l'entreprise ⚖️

<details><summary>Indice 1</summary>

Relis 206.8 (variables de substitution, équité par aveuglement), 206.9 (audit avec intervalles) et 206.11 (RGPD, arrêts SCHUFA et Dun & Bradstreet, AI Act et son encadré 🕰️), puis le ch. 29 sur les données personnelles.

</details>
<details><summary>Indice 2</summary>

Sépare trois plans : **technique** (le département porte-t-il une autre information, et laquelle ; comment le mesurer sans l'attribut), **droit de la non-discrimination** (le code pénal punit un refus intentionnel fondé sur le lieu de résidence ; la loi du 27 mai 2008 vise aussi la discrimination indirecte, sans intention ; la CNIL demande une justification), **droit des données et de l'IA** (une décision entièrement automatisée, des informations utiles sur la logique, un système à haut risque avec ses obligations, réparties entre fournisseur et déployeur). La fintech est à la fois fournisseur et déployeur.

</details>
<details><summary>Indice 3</summary>

Pour 1 : le département est une variable de substitution probable d'une origine, et aussi d'un niveau de revenu local ; on le mesure par des données agrégées publiques, en tentant de prédire l'attribut, ou par l'écart qui reste à profil financier égal (les 9 points). Pour 2 : relis la liste de l'art. 225-1 ; un délit suppose une intention, alors que la discrimination indirecte de la loi du 27 mai 2008 (art. 1 et 2) n'en suppose pas : un critère neutre en apparence qui désavantage des personnes en raison de leur origine, sauf justification ; puis l'encadré de la recommandation de la CNIL sur le « département de résidence » (une stratégie géographique justifiée et documentée). Pour 3 : une décision sans intervention humaine, qui affecte la personne, relève de l'art. 22 ; cherche l'exception (contrat ou loi) et les droits de l'art. 22.3 ; pour la demande d'accès, relis ce que l'arrêt Dun & Bradstreet exige et ce qu'il refuse. Pour 4 : annexe III, point 5 b), puis les articles 9 à 15, 26, 27 et 86, et la date de l'encadré 🕰️. Pour 5 : audit avant et après retrait, avec intervalles, et un test des variables qui pourraient prendre le relais. Pour 6 : une raison compréhensible, ce qui aurait changé la décision, le droit à un réexamen humain.

</details>

### Ex B6.10 — Reconnaissance faciale : erreurs par sous-groupe et décision de déploiement ⚖️

<details><summary>Indice 1</summary>

Relis le cas Gender Shades (§206.8), « Des écarts avec leurs intervalles » (§206.9) et « L'AI Act » (§206.11).

</details>
<details><summary>Indice 2</summary>

Une erreur totale est une moyenne des erreurs des sous-groupes, **pondérée par leur part** dans la population testée : changer la population change la moyenne. Pour la décision, distingue l'erreur qui pénalise le client légitime (faux rejet) de celle qui profite à un fraudeur (fausse acceptation). Côté droit : l'annexe III exclut la vérification 1:1 de son point 1 a) ; l'article 9 du RGPD protège les données biométriques ; l'article 5 interdit certaines catégorisations.

</details>
<details><summary>Indice 3</summary>

Pour 1 : $0{,}213 \times 20{,}8 + 0{,}250 \times 6{,}0 + 0{,}233 \times 1{,}7 + 0{,}303 \times 0{,}0$, en pour cent ; refais le calcul avec les nouvelles parts. Pour 2 : compare l'erreur la plus haute de chaque découpage à une seule dimension à celle des DF. Pour 3 : $200\,000 \times 0{,}10 \times 4\,\%$ faux rejets de DF ; compare les taux. Pour 4 : pour chaque option, qui gagne, qui perd, et est-ce permis. Pour 5 : relis la dernière phrase du point 1 a) de l'annexe III, puis l'article 9 du RGPD (données biométriques « aux fins d'identifier une personne physique de manière unique »). Pour 6 : une décision, une condition, une mesure de suivi par sous-groupe.

</details>

### Ex B6.11 — Lundberg et Lee (2017) : « A Unified Approach to Interpreting Model Predictions » 📄

<details><summary>Indice 1</summary>

Ouvre la version 2 de l'article sur arXiv ; lis le résumé, les sections 1 à 4 et les figures 1, 2 et 4, avec la fiche à côté : $M$, $\phi_i$, $\phi_0$ et « fond » y ont le même sens.

</details>
<details><summary>Indice 2</summary>

Repère dans l'article chaque notion de la fiche : définition 1 (explication additive), équations 5 à 7 (les trois propriétés), théorème 1 (l'unicité), équations 9 à 12 (les approximations de $v(S)$), théorème 2 (le noyau de Shapley), corollaire 1 (Linear SHAP). $z'$ est un vecteur de 0 et de 1 qui dit quelles features sont présentes.

</details>
<details><summary>Indice 3</summary>

1. La tension entre accuracy et interprétabilité : les modèles les plus précis (ensembles, réseaux) sont les plus durs à interpréter ; les deux composantes sont une nouvelle classe de mesures (les explications additives) et un résultat d'unicité. Pour 2, la coalition $\{1, 3\}$ s'écrit $z' = (1, 0, 1)$. Pour 3, cherche le mot « reference » ou « missing » dans chaque sous-section ; pour les modèles à réentraîner, compte les sous-ensembles de 10 features. Pour 6, applique $\pi = \frac{M - 1}{\binom{M}{|z'|}\,|z'|\,(M - |z'|)}$. Pour 8, applique la formule de Shapley à deux puis à trois joueurs, la coalition vide valant 0. Pour 9, lis les légendes des figures 3 et 5.

</details>

<a id="entretien"></a>

## 💼 Entretien

### B6.E1 — Comment expliquer les prédictions d'un gradient boosting à une équipe métier ?

<details><summary>Indice 1</summary>

Relis 206.1 (global ou local), 206.2 (permutation, dépendance partielle, ICE) et les trois graphiques de 206.4.

</details>
<details><summary>Indice 2</summary>

Trois niveaux : le modèle entier (quelles features, et dans quel sens), un client (pourquoi ce score), et la confiance (comparer à un modèle simple, montrer des cas où le modèle se trompe). Dis aussi ce qu'une explication ne dit pas.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je leur montrerais d'abord ce que le modèle utilise en général, puis pourquoi il prédit ce qu'il prédit pour des clients qu'ils connaissent. » Puis : importance par permutation ou beeswarm, dépendance partielle avec ICE, waterfall sur des cas concrets, le fond choisi, la comparaison à un modèle simple, et la limite « corrélation n'est pas causalité ».

</details>

### B6.E2 — Qu'est-ce qu'une valeur SHAP, et quelles sont ses limites ?

<details><summary>Indice 1</summary>

Relis « L'essentiel en 10 lignes », 206.3, 206.4 et 206.7.

</details>
<details><summary>Indice 2</summary>

Une définition (la part d'une feature dans l'écart entre la prédiction et la prédiction moyenne du fond, moyennée sur les coalitions), une propriété (l'efficacité), le calcul en pratique (TreeSHAP, KernelSHAP), puis trois limites : le fond, les features corrélées, la causalité.

</details>
<details><summary>Indice 3</summary>

Commence par : « Une valeur SHAP dit de combien une feature a déplacé la prédiction d'un exemple par rapport à la prédiction moyenne d'un fond de référence, en moyennant son apport sur tous les ordres d'arrivée des features. » Puis : la somme redonne la prédiction ; TreeSHAP est exact pour les arbres ; méfiance avec un fond mal choisi, des features corrélées, et toute lecture causale.

</details>

### B6.E3 — Comment audites-tu un modèle pour détecter des biais ?

<details><summary>Indice 1</summary>

Relis 206.8 (d'où vient un biais, les questions d'un audit), 206.9 (les taux, les critères, les intervalles) et 206.11 (la fiche modèle).

</details>
<details><summary>Indice 2</summary>

Une démarche en étapes : cadrer (qui est concerné, quelle décision, quelles erreurs coûtent), choisir les attributs et les croisements, calculer les taux par groupe avec effectifs et intervalles, chercher les variables de substitution, choisir un critère et le justifier, documenter, surveiller après le déploiement.

</details>
<details><summary>Indice 3</summary>

Commence par : « Avant de calculer quoi que ce soit, je regarde qui est concerné par la décision, qui a produit les labels et quelles erreurs coûtent le plus, et à qui. » Puis : taux de sélection, TPR, FPR et precision par groupe et par croisement, avec intervalles bootstrap ; variables de substitution ; critère choisi par écrit ; correction et réaudit ; fiche modèle et suivi en production.

</details>

### B6.E4 — Grad-CAM : principe, usage et limites

<details><summary>Indice 1</summary>

Relis 206.6 (l'idée, la dernière couche, les quatre détails) et 206.7 (les tests de santé, la causalité).

</details>
<details><summary>Indice 2</summary>

Le principe en une phrase (cartes de la dernière couche convolutive pondérées par la moyenne des gradients du score), l'usage (vérifier que le modèle regarde la lésion et pas un marqueur sur le cliché ou un bord d'image), les limites (résolution grossière, choix de la couche et de la classe, une carte n'est pas une preuve), et un test de santé avant de s'y fier.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je lui proposerais une carte Grad-CAM : elle montre les régions de l'image qui ont augmenté le score de la classe prédite. » Puis : comment elle se calcule, ce qu'on cherche (une fuite, comme un marqueur d'hôpital), ses limites, et les précautions : test de randomisation, comparaison avec une autre méthode, validation par des experts sur des cas variés.

</details>

### B6.E5 — Retirer la variable sensible suffit-il à rendre un modèle équitable ?

<details><summary>Indice 1</summary>

Relis « L'équité par aveuglement » (§206.8), le piège classique correspondant et « Le prix à payer » (§206.10).

</details>
<details><summary>Indice 2</summary>

Trois arguments : d'autres colonnes portent la même information (variables de substitution) ; les écarts viennent aussi des labels et des taux de base ; sans l'attribut, on ne peut plus mesurer ni corriger. Termine par ce que tu ferais à la place.

</details>
<details><summary>Indice 3</summary>

Commence par : « Non : retirer la variable n'empêche pas le modèle de retrouver la même information ailleurs, et ça nous empêche surtout de mesurer. » Puis : un exemple de variable de substitution, la mesure (prédire l'attribut à partir des autres features, comparer les taux par groupe avant et après), et la proposition de garder l'attribut pour l'audit, hors du modèle.

</details>

<a id="notebook"></a>

## Exercices du notebook

Les indices des exercices du notebook (B6.12 à B6.29) seront ajoutés à la prochaine session de génération, avec le notebook.
