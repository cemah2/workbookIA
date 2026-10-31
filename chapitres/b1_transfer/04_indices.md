# B1 · Transfer learning et modèles pré-entraînés — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B1.Q1 — Réutiliser plutôt que repartir de zéro : trois raisons et une limite

<details><summary>Indice 1</summary>

Relis le §201.1 de la fiche, « Des données et du calcul que tu n'as pas » et « Hiérarchie des features », puis l'encadré ⚖️ du §201.7.

</details>
<details><summary>Indice 2</summary>

Pour a), demande-toi pour chaque option si elle parle de ce que le modèle **apporte** (données, calcul, features) ou de ce qu'il **transmet** malgré lui. Pour d), la matrice de décision du §201.7 dit quand le transfert devient difficile.

</details>
<details><summary>Indice 3</summary>

a) **« ABD »** : moins d'images étiquetées, moins de calcul, des features déjà apprises ; l'option C contredit l'encadré « Biais hérités » (§201.7). Pour b), cherche le nombre de paires dans le premier paragraphe du §201.1. Pour c), rapproche 500 images de millions de paramètres (ch. 9). Pour d), quelle ressemblance entre tâche source et tâche cible le transfert suppose-t-il ? Pour e), qui a fait tourner les GPU pendant des jours ?

</details>

### B1.Q2 — Premières couches génériques, dernières couches spécialisées : vrai ou faux

<details><summary>Indice 1</summary>

Relis « Hiérarchie des features : du général au spécifique » (§201.1), puis le premier paragraphe après la matrice de décision (§201.7).

</details>
<details><summary>Indice 2</summary>

Range les couches de la plus générale à la plus spécifique, puis demande-toi, pour d), ce qui arrive aux features les plus spécifiques à ImageNet quand les images cibles n'ont plus rien d'ImageNet.

</details>
<details><summary>Indice 3</summary>

a) **Vrai** : bords et taches de couleur se retrouvent dans presque tous les CNN entraînés sur des photos. Pour b) et c), place la dernière couche du backbone, puis `layer1` et `layer4`, sur l'échelle « général → spécifique ». Pour d), lesquelles deviennent les premières inutiles sur un domaine lointain ? Pour e), relis d'où vient le mot « probe » (§201.3).

</details>

### B1.Q3 — Où trouver un modèle pré-entraîné, et quoi lire avant de s'en servir

<details><summary>Indice 1</summary>

Relis « Trois sources », l'encadré sur l'API transformers et « Lire la fiche modèle avant de télécharger » (§201.2).

</details>
<details><summary>Indice 2</summary>

Sépare les deux sources : torchvision pour les CNN classiques, le Hub pour les modèles de fondation. Pour b), la fiche liste six réponses à chercher dans une fiche modèle : lesquelles parlent du modèle, lesquelles de toi ?

</details>
<details><summary>Indice 3</summary>

a) **C** : DINOv2 vient du Hub, et `AutoModel` reconstruit le modèle à partir du nom de son dépôt. Pour b), garde ce qui décrit le modèle (droits, données, usages). Pour c), relis l'encadré ⚠️ du §201.2. Pour d), relis la ligne `facebook/dinov2-*` du tableau des licences. Pour e), que désigne `revision=` dans un dépôt git ?

</details>

### B1.Q4 — weights.transforms() : pourquoi le prétraitement fait partie du modèle

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « Rappel outil — `weights.transforms()` » (§201.2) et « La normalisation » (§201.5).

</details>
<details><summary>Indice 2</summary>

Une normalisation s'applique à des nombres à virgule, un recadrage à une image déjà à la bonne taille : cela fixe l'ordre. Pour b), demande-toi si la normalisation change la **forme** du tenseur ou seulement ses **valeurs**.

</details>
<details><summary>Indice 3</summary>

a) **C** : redimensionner, recadrer, passer en `float32` dans $[0, 1]$, puis normaliser. Pour b), une erreur de forme n'arrive que si une dimension change. Pour c), lis la fin de l'encadré, qui compare deux jeux de poids. Pour d), relis les statistiques citées pour CLIP au §201.5. Pour e), compare les deux jeux de poids de c).

</details>

### B1.Q5 — Extraction de features, linear probe, fine-tuning : qui s'entraîne ?

<details><summary>Indice 1</summary>

Relis « Trois façons de réutiliser » (§201.1) et la figure `strategies.png`.

</details>
<details><summary>Indice 2</summary>

Pour chaque stratégie, demande-toi ce qu'on laisse bouger : rien du backbone, ses derniers blocs, ou tout. Pour d), la tête d'un linear probe est une couche dense : un poids par couple (entrée, sortie), un biais par sortie.

</details>
<details><summary>Indice 3</summary>

a) **A** : backbone gelé, seul le classifieur posé sur les embeddings apprend. Pour b), situe le fine-tuning partiel entre l'extraction de features et le fine-tuning complet. Pour c), qu'est-ce qu'un k-NN garde en mémoire ? Pour d), calcule $d \times K + K$ avec $d = 768$ et $K = 7$. Pour e), un embedding figé change-t-il d'une epoch à l'autre ?

</details>

### B1.Q6 — Pourquoi un learning rate 10 à 100 fois plus petit pour fine-tuner ?

<details><summary>Indice 1</summary>

Relis « Pourquoi un learning rate beaucoup plus petit » et « Learning rates discriminatifs » (§201.4).

</details>
<details><summary>Indice 2</summary>

Pense à la loss comme à un paysage : où sont les poids pré-entraînés, et que fait un grand pas ? Pour c), les lettres LP et FT désignent deux étapes, dans l'ordre où on les écrit.

</details>
<details><summary>Indice 3</summary>

a) **A** : les poids sont déjà bons, au fond d'une vallée ; de grands pas les en feraient sortir. Pour b), que vaut la loss quand la tête est aléatoire ? Pour c), lis « LP-FT » de gauche à droite. Pour d), dans $\eta_\ell = \eta_{\text{tête}}\,\gamma^{L-\ell}$ avec $\gamma < 1$, quel groupe a le plus grand exposant ? Pour e), à quoi sert un warmup (ch. 19), et pourquoi en a-t-on besoin ici ?

</details>

### B1.Q7 — Batchnorm pendant le fine-tuning sur de tout petits batches

<details><summary>Indice 1</summary>

Relis « La batchnorm pendant le fine-tuning » (§201.4) et le ch. 20 (statistiques glissantes, `train()` et `eval()`).

</details>
<details><summary>Indice 2</summary>

Une batchnorm a deux régimes : en entraînement, elle utilise les statistiques **du mini-batch** et met à jour ses statistiques glissantes ; en évaluation, elle utilise les statistiques glissantes. Ses $\gamma$ et $\beta$ sont des paramètres ordinaires, réglés par `requires_grad`.

</details>
<details><summary>Indice 3</summary>

a) **B** : en mode `train()`, la moyenne et la variance du mini-batch en cours. Pour b), lequel des deux régimes met à jour les statistiques glissantes ? Pour c), relis la phrase sur l'ordre des appels, juste sous `freeze_batchnorm`. Pour d), où apparaissent des statistiques abîmées : pendant l'entraînement ou à l'évaluation ? Pour e), le mode `eval()` touche-t-il à `requires_grad` ?

</details>

### B1.Q8 — Peu ou beaucoup de données, domaine proche ou lointain : la bonne stratégie

<details><summary>Indice 1</summary>

Relis la matrice de décision du §201.7 (`matrice_decision.png`) et le protocole des courbes d'apprentissage (§201.6).

</details>
<details><summary>Indice 2</summary>

Place chaque cas dans la matrice : peu ou beaucoup d'images, domaine proche ou lointain d'ImageNet. Un spectrogramme est une image, mais d'un genre que les photos d'ImageNet ne contiennent pas.

</details>
<details><summary>Indice 3</summary>

a) **A** : peu d'images d'un domaine proche, la case de l'extraction de features. Pour b) et c), place chaque cas dans sa case de la matrice. Pour d), relis la case « beaucoup de données, domaine lointain ». Pour e), relis le troisième point du protocole d'une courbe d'apprentissage.

</details>

### B1.Q9 — Transfert négatif et oubli catastrophique : reconnaître les symptômes

<details><summary>Indice 1</summary>

Relis « Le transfert négatif » et « L'oubli catastrophique » (§201.7).

</details>
<details><summary>Indice 2</summary>

Le transfert négatif compare deux points de départ sur la **nouvelle** tâche ; l'oubli regarde ce que devient l'**ancienne** tâche. Pour e), demande-toi vers quoi chaque mesure tire les poids, ou ce qu'elle empêche de bouger.

</details>
<details><summary>Indice 3</summary>

a) **B** : moins bien que le même modèle entraîné de zéro, à budget égal. Pour b), quelle tâche a été perdue ? Pour c), sans quel point de comparaison ne peut-on pas dire qu'un transfert nuit ? Pour d), relis la liste des parades. Pour e), demande-toi, pour chacune des trois mesures, si elle retient les poids près de $\theta_0$ ou les tire ailleurs.

</details>

### B1.Q10 — Classer sans exemple d'entraînement : le zéro-shot de CLIP

<details><summary>Indice 1</summary>

Relis « Le zéro-shot » et « Les gabarits comptent » (§201.8).

</details>
<details><summary>Indice 2</summary>

Dans le zéro-shot, la tête est fabriquée **à partir de mots** : une phrase par classe. Pour b), la fiche distingue une valeur **de départ** et une valeur **apprise**.

</details>
<details><summary>Indice 3</summary>

a) **A** : des phrases écrites à partir d'un gabarit, une par classe. Pour b), relis la phrase sur `logit_scale_init_value`. Pour c), de combien diffèrent deux logits quand $s = 1$, et que fait une softmax de nombres presque égaux ? Pour d), relis la fin de « Les gabarits comptent ». Pour e), « zéro-shot » veut dire zéro exemple de **la tâche cible** : sur quoi CLIP a-t-il été entraîné ?

</details>

### B1.Q11 — Auto-supervisé (DINOv2) ou supervisé sur ImageNet : quels embeddings ?

<details><summary>Indice 1</summary>

Relis « DINOv2 : apprendre sans labels » (§201.8), l'encadré sur l'API transformers (§201.2) et la fin du §201.9 (Kornblith et coll.).

</details>
<details><summary>Indice 2</summary>

DINOv2 n'a ni labels ni légendes : son signal vient de l'image elle-même, vue de deux façons. Pour e), compte les carrés de 14 pixels le long d'un côté, puis sur toute l'image.

</details>
<details><summary>Indice 3</summary>

a) **B** : un élève et un professeur, sur des vues différentes, sans labels. Pour b), relis comment ont été obtenus les modèles *small*, *base* et *large*. Pour c), quel champ donne **un** vecteur par image ? Pour d), relis la dernière phrase du paragraphe sur Kornblith et coll. Pour e), calcule $(448 / 14)^2$, puis ajoute le token `CLS`.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### B1.R1 — Ch. 29 : licence d'un dataset et décalage de distribution

<details><summary>Indice 1</summary>

Relis la fiche du ch. 29 (licences, décalage de distribution), puis l'encadré 🕰️ sur les licences des poids (§201.2) et le §201.7.

</details>
<details><summary>Indice 2</summary>

Décode le sigle NC. Un décalage de distribution compare les données **d'entraînement** et les données **d'usage** ; pour le mesurer, il faut donc des données d'usage.

</details>
<details><summary>Indice 3</summary>

a) **Faux** : NC interdit l'usage commercial. Pour b), relis la citation des conditions d'ImageNet dans l'encadré 🕰️. Pour c), lequel des trois parle de données d'usage ? Pour d), relis la dernière phrase du deuxième paragraphe du §201.7 : qu'est-ce qui change quand le capteur change ? Pour e), quel test voit des données qui ne viennent pas du même dataset ?

</details>

### B1.R2 — Ch. 23 : geler un réseau et remplacer sa tête en PyTorch

<details><summary>Indice 1</summary>

Relis le ch. 23 (📦 23.24, gel et nouvelle tête ; sauvegarde et `weights_only`) et « Partiel ou complet » (§201.4).

</details>
<details><summary>Indice 2</summary>

Geler, c'est empêcher le calcul et l'usage du gradient d'un paramètre : quel attribut le règle ? `eval()` change le **comportement** de certaines couches (batchnorm, dropout). Pour c), une couche dense de $d$ entrées et $K$ sorties.

</details>
<details><summary>Indice 3</summary>

a) **B** : `p.requires_grad_(False)` pour chaque paramètre. Pour b), sépare ce que fait `eval()` et ce que fait `requires_grad`. Pour c), calcule $1\,280 \times 4 + 4$. Pour d), quelle est la valeur par défaut de `requires_grad` d'un paramètre neuf ? Pour e), lequel des trois appels refuse d'exécuter du code caché dans un fichier ?

</details>

### B1.R3 — Ch. 21 : ce que détectent les filtres des premières et des dernières couches

<details><summary>Indice 1</summary>

Relis le ch. 21 (📦 21.27, les filtres et les cartes d'activation d'un ResNet-18) et « Hiérarchie des features » (§201.1).

</details>
<details><summary>Indice 2</summary>

Un filtre de convolution a autant de poids que (taille du noyau) × (canaux d'entrée) ; la couche en a autant de fois qu'elle a de filtres. Pour d), combien de canaux d'entrée a un filtre de `layer4`, et combien de couleurs a une image ?

</details>
<details><summary>Indice 3</summary>

a) **B** : des bords orientés et des taches de couleur. Pour b), calcule $64 \times 3 \times 7 \times 7$, sans biais. Pour c), combien de strides 2 séparent `conv1` de `layer4` ? Pour d), peut-on afficher en couleur un filtre à 256 canaux d'entrée ? Pour e), sur quels axes porte la moyenne du pooling moyen global (ch. 21) ?

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B1.1 — Nouvelle tête : formes des tenseurs de l'image 32×32 redimensionnée jusqu'aux logits ✏️

<details><summary>Indice 1</summary>

Relis « La résolution » (§201.5), le tableau des formes de ResNet-50 et la formule de la taille de sortie d'une convolution (ch. 21).

</details>
<details><summary>Indice 2</summary>

Suis le tenseur étape par étape. Les canaux viennent du nombre de filtres de chaque bloc ; le côté change à chaque stride 2 : `conv1`, `maxpool`, puis la première convolution de `layer2`, `layer3` et `layer4`. L'axe des exemples (16) ne change jamais. Pour f) et g), applique la formule à chaque étape, sans raccourci.

</details>
<details><summary>Indice 3</summary>

a) `conv1` : $\lfloor (128 + 6 - 7)/2 \rfloor + 1 = 64$ ; `maxpool` : $\lfloor (64 + 2 - 3)/2 \rfloor + 1 = 32$ ; d'où **`"(16, 64, 32, 32)"`**. Pour b) et c), chaque bloc qui commence par un stride 2 applique $\lfloor (n + 2 - 3)/2 \rfloor + 1$ au côté, et prend ses canaux dans l'énoncé. Pour d), que garde le pooling moyen adaptatif de chaque canal ? Pour e), combien de sorties a la nouvelle `fc` ? Pour f) et g), refais les cinq étapes en partant de 32, puis de 100. Pour h), de quoi dépend la taille de $\phi(\mathbf{x})$ ?

</details>

### Ex B1.2 — k plus proches voisins cosinus sur six embeddings donnés ✏️

<details><summary>Indice 1</summary>

Relis « Les k plus proches voisins sur des embeddings » (§201.3) et l'exemple chiffré qui le suit.

</details>
<details><summary>Indice 2</summary>

Calcule une fois pour toutes la norme de $\mathbf{q}$, puis pour chaque $\mathbf{e}_i$ son produit scalaire avec $\mathbf{q}$ et sa norme ; range les six cosinus dans l'ordre décroissant. Pour les votes, compte les classes des $k$ premiers et applique la règle de l'énoncé en cas d'égalité. Pour f), la fiche donne une relation entre distance au carré et cosinus pour des vecteurs de norme 1.

</details>
<details><summary>Indice 3</summary>

a) $\mathbf{q} \cdot \mathbf{e}_5 = 0 + 2 + 2 = 4$, $\lVert\mathbf{q}\rVert = 3$, $\lVert\mathbf{e}_5\rVert = \sqrt{2}$ : $\cos = 4 / (3\sqrt{2}) \approx$ **0,943**. Pour b), calcule de même les cinq autres cosinus et prends le plus grand. Pour c) et d), compte les classes des 2 puis des 3 premiers voisins. Pour e), compare les six produits scalaires bruts. Pour f), applique la relation de la fiche à $\cos(\mathbf{q}, \mathbf{e}_1)$. Pour g), après la division, refais la comparaison de e) : les autres vecteurs ont-ils changé ?

</details>

### Ex B1.3 — Learning rates discriminatifs et warmup : le learning rate de chaque groupe, pas à pas ✏️

<details><summary>Indice 1</summary>

Relis « Learning rates discriminatifs » et « Warmup puis cosinus, groupe par groupe » (§201.4).

</details>
<details><summary>Indice 2</summary>

Deux étapes pour chaque question : le learning rate de pointe du groupe, $100 \times 0{,}3^{L-\ell}$ (en unités de $10^{-5}$), puis le facteur $f(t)$ du pas demandé. Pendant le warmup, $f(t) = t/5$ ; ensuite, la progression part de 0 à la fin du warmup.

</details>
<details><summary>Indice 3</summary>

a) `layer2` est le groupe $\ell = 1$ : $L - \ell = 3$, d'où $100 \times 0{,}3^3 =$ **2,7**. Pour b) et c), multiplie la pointe du groupe par $f(t) = t / 5$. Pour d), le facteur $f(t)$ est le même pour tous les groupes : que reste-t-il dans le rapport ? Pour e) et f), calcule la progression $(t - 5) / 100$, puis $\frac{1}{2}(1 + \cos(\pi \times \text{progression}))$. Pour g), que vaut $f(0)$, et que multiplie le learning rate dans un pas d'AdamW ? Pour h), relis la phrase de la fiche sur `LambdaLR`.

</details>

### Ex B1.4 — Compter les paramètres entraînables : tête seule, dernier bloc, réseau entier ✏️

<details><summary>Indice 1</summary>

Relis « Partiel ou complet » (§201.4) et le comptage des paramètres d'une couche au ch. 16.

</details>
<details><summary>Indice 2</summary>

Une couche dense : $d \times K + K$. Une convolution sans biais : (côté du noyau)² × (canaux d'entrée) × (canaux de sortie). Une batchnorm : $2 \times$ (canaux). Pour le réseau entier, pars du nombre donné, retire l'ancienne tête et ajoute la nouvelle. Pour `layer4`, fais la liste des sept couches à poids et des cinq batchnorms.

</details>
<details><summary>Indice 3</summary>

a) $512 \times 10 + 10 =$ **5 130**. Pour b), $11\,689\,512 - (512 \times 1\,000 + 1\,000) + $ la nouvelle tête. Pour c), $3 \times 3 \times 512 \times 512$. Pour d), additionne bloc 0 (trois convolutions, dont celle du raccourci, et trois batchnorms) et bloc 1 (deux convolutions, deux batchnorms). Pour e), `layer4` plus la tête ; pour f), e) divisé par b), fois 100. Pour g), retire de e) les $\gamma$ et $\beta$ des batchnorms de `layer4`. Pour h), sépare le mode d'une couche et son `requires_grad`.

</details>

### Ex B1.5 — Zéro-shot CLIP à la main : cosinus, logit_scale et softmax ✏️

<details><summary>Indice 1</summary>

Relis « Le zéro-shot » (§201.8), l'exemple à trois classes et le piège « Prendre l'initialisation pour l'échelle apprise ».

</details>
<details><summary>Indice 2</summary>

Calcule d'abord $s = e^{4{,}60517}$. Multiplie chaque cosinus par $s$ pour obtenir les logits, retranche le plus grand (les exponentielles restent petites), prends les exponentielles et divise par leur somme. Pour c) et f), écris le rapport de deux probabilités : il ne reste que l'écart de leurs logits.

</details>
<details><summary>Indice 3</summary>

a) $s \approx 100$ ; logits 24, 30, 28, 19 ; écarts au plus grand : −6, 0, −2, −11 ; $p(\text{oiseau}) = 1 / (e^{-6} + 1 + e^{-2} + e^{-11}) \approx$ **0,879**. Pour b) et d), même dénominateur, avec l'exponentielle de l'écart de la classe au numérateur. Pour c), $e^{s\,(0{,}30 - 0{,}28)}$. Pour e), refais a) avec $s = e^{2{,}6592}$. Pour f), le rapport de c) contient-il le dénominateur de la softmax ?

</details>

### Ex B1.6 — Rester près des poids pré-entraînés : le gradient de la pénalité L2-SP ∂

<details><summary>Indice 1</summary>

Relis « Rester près des poids pré-entraînés : la pénalité L2-SP » (§201.4), puis la régularisation L2 au ch. 9 et le weight decay découplé d'AdamW au ch. 19.

</details>
<details><summary>Indice 2</summary>

Dans la somme qui définit $\Omega$, seul le terme $j$ dépend de $\theta_j$. Pour 2, remplace $\nabla\Omega$ dans $\theta_{t+1} = \theta_t - \eta\,(\nabla L + \nabla\Omega)$, puis retranche $\theta_0$ des deux côtés : on obtient une suite géométrique. Pour 4 et 5, pose $\delta = \theta - \theta_0$, puis cherche le point où le gradient total s'annule.

</details>
<details><summary>Indice 3</summary>

1. $\frac{\partial \Omega}{\partial \theta_j} = \frac{\lambda}{2} \times 2(\theta_j - \theta_{0,j}) = \lambda(\theta_j - \theta_{0,j})$, d'où $\nabla\Omega = \lambda(\theta - \theta_0)$. Pour 2, l'écart est multiplié à chaque pas par le même facteur : quelles conditions sur ce facteur pour qu'il rapetisse sans changer de signe ? Pour 3, calcule ce facteur, élève-le à la puissance 10, et cherche le plus petit $t$ tel que (facteur)$^t \le 1/2$. Pour 5, annule $\lambda_1(\theta - \theta_0) + \lambda_2\,\theta$. Pour 6, annule $\mathbf{g} + \lambda(\theta - \theta_0)$. Pour 7, que fait Adam d'un gradient avant de l'appliquer (ch. 19) ?

</details>

### Ex B1.7 — Fermi : pré-entraîner sur ImageNet ou fine-tuner sur 5 000 images (GPU-heures, énergie, coût) 🧮

<details><summary>Indice 1</summary>

Une seule formule sert à tout : opérations = (images vues) × (opérations par image et par pas). Puis durée = opérations / débit de la machine.

</details>
<details><summary>Indice 2</summary>

Images vues = epochs × images du dataset ; opérations par image d'entraînement = $3 \times 4{,}1 \times 10^9$ ; pour une seule passe avant (le linear probe), $4{,}1 \times 10^9$. Une heure = 3 600 s ; énergie = puissance × durée.

</details>
<details><summary>Indice 3</summary>

1. $90 \times 1\,281\,167 \times 3 \times 4{,}1 \times 10^9 \approx 1{,}42 \times 10^{18}$ opérations ; divise par $10^{13}$ op/s, puis par 3 600 pour des GPU-heures ; l'énergie et le coût suivent. Pour 2, multiplie par 30, et compte les annotations : images × 3 × 0,02 €. Pour 3, même calcul avec 10 epochs de 5 000 images, puis deux débits. Pour 4, $5\,000 \times 4{,}1 \times 10^9$ opérations sur le CPU. Pour 5, fais les rapports ; pour CLIP, $12 \times 24 \times 256$ GPU-heures. Pour 6, pense à la recherche qui a précédé, à la collecte, et au stockage.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B1.8 — Le transfer learning expliqué en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis « L'essentiel en 10 lignes », points 1, 2 et 7.

</details>
<details><summary>Indice 2</summary>

Cherche une image de la vie courante où l'on réutilise un savoir général pour un métier précis : on garde le savoir général, on n'apprend que la partie nouvelle. Une ligne par point de l'énoncé, sans jargon (ni « backbone », ni « embedding »).

</details>
<details><summary>Indice 3</summary>

Ligne 1, sur le modèle : « Un réseau qui a regardé des millions de photos a appris à y voir des bords, des textures, des formes. » Pour les lignes 2 à 5 : ce qu'on garde et ce qu'on remplace (la dernière étape, qui nomme les classes) ; pourquoi quelques centaines d'images suffisent alors ; un cas où ce savoir ne sert plus (des images d'un tout autre genre) ; et ton image de la vie courante.

</details>

### Ex B1.9 — Biais et licences hérités d'un modèle pré-entraîné : un projet d'imagerie médicale ⚖️

<details><summary>Indice 1</summary>

Relis le tableau des licences et les encadrés 🕰️ et ⚠️ du §201.2, l'encadré ⚖️ du §201.7, et la fiche du ch. 29 (données personnelles, data card).

</details>
<details><summary>Indice 2</summary>

Range tes réponses en trois familles : ce que le modèle de base **permet** (licence, usages prévus), ce qu'il **transmet** (biais du pré-entraînement) et ce que tes données **ajoutent** (un hôpital, trois annotateurs, des appareils). Pour la réglementation, pense à la documentation, aux tests et à la surveillance.

</details>
<details><summary>Indice 3</summary>

Pour 1, relis la colonne « À savoir » du tableau des licences pour chacun des trois modèles : lequel exclut **tout** déploiement non testé ? Pour 2, sépare la licence du code, les conditions d'ImageNet et le statut des poids appris. Pour 3, cherche qui manque sur les photos d'ImageNet et dans les 2 000 photos d'un seul hôpital. Pour 4, une métrique par sous-groupe, et la plus grave des erreurs (une lésion maligne manquée). Pour 5, les photos de santé sont des données sensibles au sens du RGPD. Pour 6, ce qu'impose un système à haut risque (documentation technique, évaluation, surveillance), et qui évalue un dispositif médical avant sa mise sur le marché. Pour 7, une ligne par biais ou limite, chacune avec ce qui a été mesuré.

</details>

### Ex B1.10 — Yosinski et coll. (2014) : jusqu'où les features se transfèrent-elles ? 📄

<details><summary>Indice 1</summary>

Lis d'abord la figure 1 de l'article, qui montre les quatre traitements sur un schéma, puis la figure 2 avec son commentaire numéroté (section 4.1).

</details>
<details><summary>Indice 2</summary>

La notation se lit ainsi : « AnB » = les $n$ premières couches viennent d'un réseau entraîné sur A, le reste est entraîné sur B ; le « + » indique que les couches copiées sont fine-tunées au lieu d'être gelées. Le contrôle BnB copie les couches d'un réseau entraîné sur **la même** tâche : ce qui s'y dégrade ne peut pas venir de la spécificité des features.

</details>
<details><summary>Indice 3</summary>

Pour 1, cherche dans la section 2 le nombre de classes de chaque moitié et le nombre de découpages au hasard. Pour 2, demande-toi ce que BnB garde identique à baseB, à part le découpage du réseau. Pour 3, regarde à quelles couches BnB descend sous baseB, et cherche l'expression *fragile co-adaptation*. Pour 4, compare AnB à BnB couche par couche : l'écart entre les deux courbes est la part de la spécificité. Pour 5, cherche le tableau des gains moyens et le rôle de BnB+. Pour 6, lis le début de la section 4.2 (le découpage par la hiérarchie de WordNet), puis la section 4.3. Pour 7, relis la règle du §201.1 de la fiche et compare.

</details>

<a id="entretien"></a>

## 💼 Entretien

### B1.E1 — Extraction de features ou fine-tuning : comment choisis-tu ?

<details><summary>Indice 1</summary>

Relis la matrice de décision (§201.7), l'encadré 💼 du §201.1 et les courbes d'apprentissage (§201.6).

</details>
<details><summary>Indice 2</summary>

Deux critères (le volume de données, la distance de domaine), une première expérience toujours la même (le linear probe), puis une décision mesurée sur la validation, avec ses coûts.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je ne choisis pas a priori : je commence toujours par un linear probe sur un backbone gelé, c'est ma baseline. » Puis les deux critères et leur matrice, le fine-tuning partiel puis complet si le probe plafonne, et ce que tu compares (validation, coût, risque d'oubli).

</details>

### B1.E2 — Tu as 500 images médicales étiquetées : quelle stratégie de transfer learning ?

<details><summary>Indice 1</summary>

Relis la case « peu de données, domaine lointain » de la matrice (§201.7), le protocole du §201.6 et l'encadré ⚖️ du §201.7.

</details>
<details><summary>Indice 2</summary>

Structure : le diagnostic (peu d'images, domaine lointain), les premières expériences (plusieurs backbones gelés, sondés à plusieurs blocs), les baselines, le protocole d'évaluation (validation croisée, plusieurs graines, découpage par patient), puis la suite si cela ne suffit pas.

</details>
<details><summary>Indice 3</summary>

Commence par : « 500 radiographies, c'est peu, et c'est loin d'ImageNet : je commence par mesurer, pas par entraîner. » Puis : linear probe sur deux ou trois backbones (dont un modèle de fondation, et un modèle pré-entraîné sur de l'imagerie médicale s'il en existe sous une licence compatible), sondage des blocs intermédiaires, fine-tuning partiel prudent, baselines, découpage par patient, recall de la classe grave, et la vérification des licences et des données.

</details>

### B1.E3 — Pourquoi un learning rate plus faible pour les couches pré-entraînées ?

<details><summary>Indice 1</summary>

Relis « Pourquoi un learning rate beaucoup plus petit » et « Learning rates discriminatifs » (§201.4), et ton ✏️ B1.3.

</details>
<details><summary>Indice 2</summary>

Deux idées : les poids sont déjà bons (ne pas les détruire), et les premières couches sont les plus générales (les bouger le moins). Puis la mise en œuvre : groupes de paramètres, planning, warmup.

</details>
<details><summary>Indice 3</summary>

Commence par : « Les poids pré-entraînés sont déjà dans une bonne région de la loss : un grand pas les en sortirait. » Puis la tête aléatoire au départ, la hiérarchie des features, la formule $\eta_\ell = \eta_{\text{tête}}\gamma^{L-\ell}$, les `param_groups` d'AdamW et le warmup puis le cosinus de `LambdaLR`.

</details>

### B1.E4 — Qu'est-ce qu'un modèle de fondation, et que change le zéro-shot ?

<details><summary>Indice 1</summary>

Relis « Qu'est-ce qu'un modèle de fondation ? » et « Le zéro-shot » (§201.8).

</details>
<details><summary>Indice 2</summary>

Une définition (entraîné une fois, sur des données vastes et variées, sans tâche précise, réutilisé pour beaucoup), deux familles en vision, puis ce que le zéro-shot change pour un projet : le premier prototype, et ses limites.

</details>
<details><summary>Indice 3</summary>

Commence par : « Un modèle de fondation est un grand modèle entraîné une fois sur des données très vastes, sans viser une tâche précise, puis réutilisé pour beaucoup de tâches. » Puis DINOv2 et CLIP, le zéro-shot de CLIP en une phrase (cosinus avec des phrases, échelle, softmax), ce que cela change (un prototype sans aucune image étiquetée) et ses limites (classes fines, liste de classes, anglais, licence).

</details>

### B1.E5 — Comment vérifier qu'un modèle pré-entraîné peut servir dans un produit commercial ?

<details><summary>Indice 1</summary>

Relis « Lire la fiche modèle avant de télécharger », son tableau et ses encadrés (§201.2), puis « Choisir un backbone pour la production » (§201.9).

</details>
<details><summary>Indice 2</summary>

Passe en revue : la licence des poids, les données du pré-entraînement et leurs conditions, l'usage prévu et les usages exclus de la fiche modèle, les conditions d'accès, puis la traçabilité (révision figée, fiche modèle du modèle dérivé).

</details>
<details><summary>Indice 3</summary>

Commence par : « Téléchargeable ne veut pas dire utilisable : je lis la fiche modèle avant la première ligne de code. » Puis la licence (permissive, non commerciale, propre à l'entreprise), les données d'entraînement (les conditions d'ImageNet), l'usage prévu (l'exemple de CLIP), les obligations (mentions, citation), la révision figée et la fiche modèle du modèle dérivé, et l'avis d'un juriste en cas de doute.

</details>

<a id="notebook"></a>

## Exercices du notebook

### Ex B1.11 — Trois modèles pré-entraînés en main : ResNet-18, MobileNetV3-small, DINOv2-small 📦

<details><summary>Indice 1</summary>

Relis le §201.2 de la fiche (les énumérations de poids de torchvision, `weights.transforms()`, l'API `AutoModel` et `AutoImageProcessor`) et « La nouvelle tête » (§201.5) pour la forme de `classifier` chez MobileNet.

</details>
<details><summary>Indice 2</summary>

Pour chaque modèle de torchvision : prends la valeur `IMAGENET1K_V1` de son énumération de poids, construis le modèle avec `weights=`, passe-le en `eval()`, et range le couple (modèle, `weights.transforms()`). Pour DINOv2 : `from_pretrained` avec le nom du dépôt **et** `revision=`, pour le modèle comme pour le processeur. Pour les valeurs : `print(model.classifier)` montre les couches et leurs tailles ; `model.config` contient `hidden_size` ; une passe avant de deux images préparées par le processeur donne les deux sorties à comparer ; `print(preprocess)` et l'attribut `resample` du processeur (un entier de PIL) disent l'interpolation.

</details>
<details><summary>Indice 3</summary>

```python
def load_backbones_11():
    out = {}
    for name, builder, weights in (("resnet18", resnet18, ResNet18_Weights.IMAGENET1K_V1), ...):
        out[name] = (builder(weights=weights).eval(), weights.transforms())
    ...                      # DINOv2: AutoImageProcessor and AutoModel, both with revision=DINOV2_REVISION
    return out
```
a) lis `in_features` de la première couche de `classifier` ; b) celui de sa dernière couche ; d) `torch.allclose(out.pooler_output, out.last_hidden_state[:, 0])` ; f) PIL numérote ses interpolations : `Image.Resampling(3)` te dit laquelle est la 3.

</details>

### Ex B1.12 — Prédire : un ResNet ImageNet face à des images CIFAR-10 de 32×32 pixels 🔮

<details><summary>Indice 1</summary>

Relis « La résolution » (§201.5) et ta réponse au ✏️ B1.1 f) : à 32 pixels, que reste-t-il de la carte de `layer4` avant le pooling ?

</details>
<details><summary>Indice 2</summary>

Pour a), deux arguments s'opposent : les filtres attendent des motifs à l'échelle d'une image de 224 pixels, mais un linear probe n'a besoin que de 512 nombres qui varient avec la classe, même s'ils ne « voient » plus les objets comme sur ImageNet. Compare à deux repères : le hasard (10 %) et ce que donnent des embeddings à la bonne échelle. Pour b), demande-toi si une image trop petite pousse le réseau vers une poignée de classes « par défaut », ou si de petites différences entre images suffisent à changer sa classe la plus probable parmi 1 000.

</details>
<details><summary>Indice 3</summary>

Écris tes deux raisonnements avant de choisir : a) « une carte 1 × 1 garde ou perd … parce que … » ; b) « parmi 1 000 classes, la plus probable change ou ne change pas d'une image à l'autre parce que … ». Puis place chaque estimation dans les catégories de l'énoncé.

</details>

### Ex B1.13 — Extraire des embeddings figés, puis linear probe et k plus proches voisins 📦

<details><summary>Indice 1</summary>

Relis le §201.3 de la fiche (calculer les embeddings une fois, le linear probe, les k plus proches voisins) et l'encadré 🧮 sur `weights.transforms()` (§201.2) pour les étapes de la préparation.

</details>
<details><summary>Indice 2</summary>

`make_backbone_13` : `model.fc = nn.Identity()`, puis `eval()`. `embed_13` : une boucle sur les mini-batches ; pour chacun, `torch.from_numpy`, `permute(0, 3, 1, 2)` (NHWC → NCHW), `float() / 255`, `F.interpolate(..., size=(size, size), mode="bilinear")`, puis soustraire la moyenne et diviser par l'écart-type, deux tenseurs de forme `(1, 3, 1, 1)` ; le tout sous `torch.inference_mode()`, et `torch.cat` à la fin. `probe_13` : `make_pipeline` du ch. 15 ; `knn_13` : `KNeighborsClassifier` du ch. 13 avec `metric="cosine"` ; `score` rend l'accuracy.

</details>
<details><summary>Indice 3</summary>

```python
def embed_13(backbone, images, size, batch_size=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(images[i:i + batch_size]).permute(0, 3, 1, 2).float() / 255
            ...              # resize to size x size, normalise, then the backbone
    return torch.cat(out).numpy()
```
Une seule ligne par classifieur : `float(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(...).score(...))`.

</details>

### Ex B1.14 — Résolution d'entrée : accuracy du linear probe contre temps de calcul (32 à 224 pixels) 🔬

<details><summary>Indice 1</summary>

Tu as déjà les deux briques : `embed_13` et `probe_13`. Relis « Mesurer une latence correctement » (§201.9) pour le temps.

</details>
<details><summary>Indice 2</summary>

Pour chaque taille : les embeddings de `train` et de `test` à cette taille, puis le probe ; ensuite, `repeats` fois, `time.perf_counter()` avant et après `embed_13` sur `X_test[:64]`, et tu gardes la plus petite durée, divisée par 64 et multipliée par 1 000. Le premier appel d'une taille paie des préparations (allocation de la mémoire, choix des algorithmes) : c'est pour cela qu'on ne garde pas une mesure unique.

</details>
<details><summary>Indice 3</summary>

```python
(X_tr, y_tr), (X_te, y_te) = train, test
rows = []
for size in sizes:
    accuracy = probe_13(embed_13(backbone, X_tr, size), y_tr, embed_13(backbone, X_te, size), y_te)
    ...                      # repeats timings of embed_13 on X_te[:64], keep the smallest
    rows.append({"size": size, "accuracy": accuracy, "ms_per_image": ...})
return rows
```

</details>

### Ex B1.15 — Lire des projections PCA/UMAP : pixels bruts, ResNet et DINOv2 📈

<details><summary>Indice 1</summary>

Relis « Regarder et sonder » (§201.3) et les précautions du ch. 12 sur UMAP : il garde les voisinages, pas les distances entre groupes.

</details>
<details><summary>Indice 2</summary>

Un groupe net, c'est un nuage d'une seule couleur, compact, peu envahi par les autres couleurs. Pour b), UMAP fabrique des groupes à partir de n'importe quelles données : regarde de quelles couleurs sont faits ceux des pixels bruts. Pour c), repère les quatre couleurs des véhicules dans l'UMAP de (B). Pour d), demande-toi ce que perd une projection de 512 dimensions sur 2.

</details>
<details><summary>Indice 3</summary>

a) Dans les panneaux du bas, les pixels bruts (A) ne forment aucun groupe d'une seule couleur ; reste à comparer (B) et (C) : lequel a les nuages les plus compacts et les moins mêlés ? b) Dans l'UMAP de (A), choisis un groupe : combien de couleurs y vois-tu ? c) Les véhicules tombent-ils d'un même côté de la figure ? d) Compare ce que montre la PCA de (B) à l'accuracy de ton linear probe en B1.13.

</details>

### Ex B1.16 — Sonder chaque bloc : linear probe couche par couche, CIFAR-10 contre Fashion-MNIST 🔬

<details><summary>Indice 1</summary>

Relis la fin de « Regarder et sonder » (§201.3 : `create_feature_extractor`), « Les canaux » (§201.5) et « La distance de domaine » (§201.7).

</details>
<details><summary>Indice 2</summary>

`create_feature_extractor(model, return_nodes={...})` rend, en une passe, un dictionnaire des sorties des nœuds demandés : les blocs s'appellent `"layer1"` … `"layer4"`, et la sortie du stem est celle du nœud `"maxpool"`. Le pooling moyen global d'une carte `(N, C, H, W)` est `.mean(dim=(2, 3))`. Prépare chaque mini-batch par `to_input(batch, size)`. `probe_by_block_16` : les embeddings des deux jeux pour tous les blocs, puis un `probe_13` par bloc.

</details>
<details><summary>Indice 3</summary>

```python
nodes = {("maxpool" if b == "stem" else b): b for b in blocks}
extractor = create_feature_extractor(model, return_nodes=nodes)
with torch.inference_mode():
    for i in range(0, len(images), batch_size):
        maps = extractor(to_input(images[i:i + batch_size], size))
        ...                  # for each block: maps[b].mean(dim=(2, 3)), kept in a list
```

</details>

### Ex B1.17 — Fine-tuning partiel : dégeler le dernier bloc avec des learning rates par groupe 📦

<details><summary>Indice 1</summary>

Relis le §201.4 de la fiche : partiel ou complet, learning rates discriminatifs (les groupes de paramètres), la batchnorm pendant le fine-tuning ; et ton ✏️ B1.3.

</details>
<details><summary>Indice 2</summary>

`param_groups_17` : pour le $k$-ième module, la liste de ses paramètres qui demandent un gradient, coupée en deux selon `p.ndim >= 2` ; un dictionnaire par liste non vide. `make_partial_17` : la nouvelle tête d'abord, puis tout geler (`requires_grad_(False)`), puis dégeler `layer4` et `fc`. `freeze_batchnorm_17` : `model.modules()` et `isinstance(m, nn.modules.batchnorm._BatchNorm)`. `train_partial_17` : l'optimiseur une fois, avant la boucle des epochs ; dans chaque epoch, `model.train()` puis `freeze_batchnorm_17`, puis la boucle du ch. 23 sur `batches(...)`, et la moyenne de la loss pondérée par la taille des mini-batches.

</details>
<details><summary>Indice 3</summary>

```python
def param_groups_17(modules, head_lr, gamma, weight_decay=0.01):
    groups = []
    for k, module in enumerate(modules):
        params = [p for p in module.parameters() if p.requires_grad]
        ...                  # two lists by p.ndim, one dict {"params", "lr": head_lr * gamma ** k, "weight_decay"} each
    return groups
```
Dans `train_partial_17` : `optimizer = torch.optim.AdamW(param_groups_17([model.fc, model.layer4], head_lr, gamma))`.

</details>

### Ex B1.18 — Le fine-tuning qui détruit le modèle : quatre erreurs à trouver 🐛

<details><summary>Indice 1</summary>

Lis les messages des diagnostics dans l'ordre : chacun mesure un symptôme et pose une question. Relis le §201.4 (learning rate, batchnorm) et le §201.5 (normalisation, nouvelle tête).

</details>
<details><summary>Indice 2</summary>

Point 1 : que contient l'optimiseur au moment où il est créé ? Point 2 : compare la préparation des images à celle de ton `embed_13`. Point 3 : dans quel mode sont les batchnorms pendant la boucle, et que font-elles de leurs statistiques glissantes en ce mode ? Point 4 : quel pas reçoit chaque paramètre pré-entraîné, comparé à la fourchette de la fiche pour AdamW ?

</details>
<details><summary>Indice 3</summary>

Chaque erreur est sur une ou deux lignes de `finetune_18`, avant la boucle ou dans sa préparation des images. Après chaque correction, relance la cellule de vérification : le point suivant te dit où chercher. Pour le point 4, un squelette :
```python
optimizer = torch.optim.AdamW([{"params": ..., "lr": ...},      # the new head
                               {"params": ..., "lr": ...}])     # the pre-trained layers
```

</details>

### Ex B1.19 — Fine-tuning complet : warmup + cosinus, early stopping, meilleur checkpoint 📦

<details><summary>Indice 1</summary>

Relis « Warmup puis cosinus, groupe par groupe » (§201.4, avec son code `warmup_cosine`), ton ✏️ B1.3, et la fonction `fit()` du ch. 23 (early stopping, meilleur checkpoint).

</details>
<details><summary>Indice 2</summary>

`warmup_cosine_19` : deux cas, le warmup puis le cosinus, la progression comptée à partir de la fin du warmup et bornée à 1. `fine_tune_19` : l'optimiseur sur `param_groups_17` des cinq modules, un `LambdaLR` dont la fonction reçoit le numéro du pas ; dans la boucle des mini-batches, note les learning rates des groupes, puis `optimizer.step()`, puis `scheduler.step()`. À la fin de chaque epoch : `evaluate` sur la validation ; si c'est un record strict, `copy.deepcopy(model.state_dict())` et le compteur d'attente à 0 ; sinon, le compteur augmente, et l'on sort de la boucle quand il atteint `patience`. Après la boucle, `load_state_dict` de la copie.

</details>
<details><summary>Indice 3</summary>

```python
t_total = epochs * math.ceil(len(X) / batch_size)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: warmup_cosine_19(t, t_warmup, t_total))
for epoch in range(epochs):
    ...                      # train() + freeze_batchnorm_17, the mini-batches, history["lr"].append([...]) before step()
    acc = evaluate(model, X_val, y_val)
    if acc > best_acc:
        best_acc, best_state, waited = acc, copy.deepcopy(model.state_dict()), 0
    ...                      # else: one more epoch without progress; stop at patience
```

</details>

### Ex B1.20 — Prédire : où le pré-entraînement rapporte-t-il le plus, 10 ou 1 000 images par classe ? 🔮

<details><summary>Indice 1</summary>

Relis « La courbe d'apprentissage selon la taille du train » (§201.6) et tes résultats de B1.13 et B1.14 : que valait déjà le linear probe avec quelques dizaines d'images par classe ?

</details>
<details><summary>Indice 2</summary>

Estime séparément les quatre accuracies, puis fais les deux soustractions. Pour le linear probe, pense à ce que l'embedding sait déjà avant la première image de CIFAR-10. Pour le CNN from scratch, demande-toi ce que peut apprendre un réseau de 24 000 paramètres avec 10 photos par classe, puis avec 1 000 : que doit-il découvrir seul, que le backbone a appris sur ImageNet ? Le hasard vaut 10 %.

</details>
<details><summary>Indice 3</summary>

Écris tes quatre estimations avant de choisir : « avec 10 images par classe, le probe fait environ … % et le CNN from scratch environ … %, parce que … » ; puis la même phrase pour 1 000 images par classe. Place chaque différence dans les catégories de l'énoncé (en points, pas en pourcentage relatif).

</details>

### Ex B1.21 — Courbes d'apprentissage : from scratch, linear probe et fine-tuning de 10 à 1 000 images par classe 🔬

<details><summary>Indice 1</summary>

Relis le protocole de la fiche (§201.6, quatre points) : chaque fonction du contrat en applique un. Les trois stratégies réutilisent ce que tu as déjà écrit : `probe_13` (B1.13), `make_full_19` et `fine_tune_19` (B1.19), et la fonction donnée `train_scratch`.

</details>
<details><summary>Indice 2</summary>

`probe_21` : deux appels à `reference_embeddings` (le train avec le sous-ensemble `"curve"`, le test avec `"test"` : le cache reconnaît les images de test déjà calculées), puis `probe_13(..., C=0.01)`. `scratch_21` : une ligne, `train_scratch(...).accuracy(...)`. `finetune_21` : le nombre de pas par epoch est `ceil(len(X) / 32)` ; le nombre d'epochs, le plus grand de `FT_EPOCHS` et `ceil(40 / pas par epoch)` ; `fine_tune_19` reçoit `**FT_RECIPE_21`, puis `evaluate` sur le test ; une graine fixée avant `make_full_19` rend la tête reproductible. `learning_curve_21` : deux boucles, `sorted(plan)` puis les graines dans l'ordre ; `summarize_21` : regroupe par taille et applique `np.mean` et `np.std`.

</details>
<details><summary>Indice 3</summary>

```python
def learning_curve_21(fit_eval, plan, test, load=wb.datasets.load_cifar10_subset):
    X_test, y_test = test
    rows = []
    for size in sorted(plan):
        for seed in plan[size]:
            X, y = load(size, "train", seed)
            ...              # one point: {"size", "seed", "accuracy": float(fit_eval(..., seed=seed))}
    return rows
```
Pour `finetune_21` : `fine_tune_19(model, (X, y), (X_VAL, Y_VAL), epochs=..., t_warmup=per_epoch, batch_size=32, seed=seed, **FT_RECIPE_21)`.

</details>

### Ex B1.22 — Domaine lointain : transférer vers Fashion-MNIST, comparé au CNN from scratch 🔬

<details><summary>Indice 1</summary>

Relis « La distance de domaine » et « Le transfert négatif » (§201.7), puis tes courbes de B1.16 : sur Fashion-MNIST, quel bloc donnait le meilleur probe ? Les deux fonctions du contrat ont la forme des `fit_eval` de B1.21, que ton `learning_curve_21` sait déjà appeler.

</details>
<details><summary>Indice 2</summary>

`fm_probe_22(block)` est une fabrique : elle définit une fonction interne `fit_eval(X, y, X_test, y_test, seed=0)`, qui lit `fm_embeddings(X)[block]` et `fm_embeddings(X_test)[block]` et appelle `probe_13` avec `C=0.01`, puis elle **rend** cette fonction (sans l'appeler). `fm_scratch_22` est la même ligne que `scratch_21` : `train_scratch` accepte des images en niveaux de gris `(N, 28, 28)`.

</details>
<details><summary>Indice 3</summary>

```python
def fm_probe_22(block):
    def fit_eval(X, y, X_test, y_test, seed=0):
        ...                  # probe_13 on the embeddings of this block, C = 0.01
    return fit_eval          # the function itself, not fit_eval(...)
```

</details>

### Ex B1.23 — Oubli catastrophique : que reste-t-il des prédictions ImageNet après le fine-tuning ? 🔬

<details><summary>Indice 1</summary>

Relis « L'oubli catastrophique » (§201.7) : la mesure remet l'ancienne tête sur le nouveau backbone. Pense aussi à ce que tu as appris au ch. 23 sur la copie d'un modèle et sur `eval()`.

</details>
<details><summary>Indice 2</summary>

Pour ne pas toucher au modèle reçu, travaille sur une copie (`copy.deepcopy`) : remplace sa tête `fc` par (une copie de) `reference.fc`, passe les deux réseaux en `eval()` et sur le CPU, puis compare, mini-batch par mini-batch, `argmax(1)` des deux sorties sous `torch.inference_mode()` ; l'accord est le nombre d'égalités divisé par le nombre d'images. Pour a) et b), lis les deux lignes de la cellule d'expérience.

</details>
<details><summary>Indice 3</summary>

```python
old = copy.deepcopy(model).cpu().eval()
old.fc = copy.deepcopy(reference.fc)          # the ImageNet head on the fine-tuned backbone
...                                            # loop over mini-batches: x = to_input(images[i:i + batch_size], size)
                                               # count (old(x).argmax(1) == reference(x).argmax(1)).sum()
```
a) Repère la ligne de `MODEL_19` et place son accord dans les catégories. b) Compare les accords de `MODEL_17` et de `MODEL_19`.

</details>

### Ex B1.24 — Zéro-shot avec CLIP sur CIFAR-10 : gabarits de prompts et ensembles de prompts 📦

<details><summary>Indice 1</summary>

Relis « Le zéro-shot » et « Les gabarits comptent » (§201.8), avec l'extrait de code de la fiche, et ton ✏️ B1.5 pour le rôle de $s$.

</details>
<details><summary>Indice 2</summary>

`text_embeddings_24` : pour chaque classe, la liste des phrases `[t.format(nom) for t in templates]`, passée au processeur (`text=`, `return_tensors="pt"`, `padding=True`) puis à `get_text_features` ; avec transformers 5, l'embedding est `.pooler_output`. Normalise chaque ligne (`v / v.norm(dim=-1, keepdim=True)`), fais la moyenne sur les phrases, renormalise, empile les classes et convertis en NumPy `float32`. `zero_shot_24` : normalise les deux matrices ligne par ligne, multiplie `u @ v.T` par `scale`, puis une softmax stable sur l'axe des classes (soustrais le maximum de chaque ligne, ch. 17).

</details>
<details><summary>Indice 3</summary>

```python
def zero_shot_24(image_emb, text_emb, scale):
    u = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
    v = ...                  # the same for the text vectors
    z = scale * (u @ v.T)    # (N, K) logits
    ...                      # softmax over axis=1, stable
```
Dans `text_embeddings_24`, sous `torch.inference_mode()` : `out = model.get_text_features(**processor(text=texts, return_tensors="pt", padding=True))`, puis `v = out.pooler_output`.

</details>

### Ex B1.25 — Few-shot par prototypes de classe avec DINOv2 et CLIP (1 à 16 exemples par classe) 📦

<details><summary>Indice 1</summary>

Relis « Le few-shot et les prototypes de classe » (§201.6) : la fiche donne déjà le calcul des prototypes et de la prédiction en NumPy. `few_shot_25` répète ce calcul pour chaque nombre d'exemples et chaque tirage du support.

</details>
<details><summary>Indice 2</summary>

`prototypes_25` : normalise d'abord chaque embedding, puis la moyenne des lignes de chaque classe, `c` de 0 à `n_classes - 1`. `predict_25` : normalise **aussi** les prototypes (la moyenne de vecteurs de norme 1 n'est plus de norme 1), puis `argmax` de la matrice des cosinus. `few_shot_25` : pour chaque `k`, une liste de `draws` accuracies ; pour chaque tirage `d`, `pos = draw_support(y_support, k, d, seed)`, les prototypes de `E_support[pos]`, puis l'accuracy sur les requêtes ; enfin `np.mean` et `np.std`.

</details>
<details><summary>Indice 3</summary>

```python
def few_shot_25(support, query, shots, draws, seed=0):
    (E_s, y_s), (E_q, y_q) = support, query
    rows = []
    for k in shots:
        accs = []
        for d in range(draws):
            pos = draw_support(y_s, k, d, seed)
            ...              # prototypes of E_s[pos], then the accuracy of predict_25 on the query
        rows.append({"shots": k, "mean": float(np.mean(accs)), "std": float(np.std(accs))})
    return rows
```

</details>

### Ex B1.26 — Taille, latence et accuracy sur CPU : MobileNetV3-small, ResNet-18, DINOv2-small 🔬

<details><summary>Indice 1</summary>

Relis « Quatre mesures à ne pas confondre » et « Mesurer une latence correctement » (§201.9) : la fiche donne une fonction de latence presque complète, et la différence entre latence et débit.

</details>
<details><summary>Indice 2</summary>

`latency_ms_26` : `model.eval()`, puis sous `torch.inference_mode()` d'abord `warmup` appels non chronométrés, puis `reps` appels entourés de `time.perf_counter()` ; rends `1000 * np.median(...)`, en `float`. `size_26` : parcours `model.parameters()` ; `p.numel()` compte les valeurs, `p.element_size()` les octets de chacune (4 en `float32`, 2 en `float16`). `profile_26` : deux appels à ta latence, l'un sur `torch.randn(1, 3, size, size)`, l'autre sur un batch de `batch` images avec `reps=5` ; le débit est `batch` divisé par la durée du batch **en secondes**.

</details>
<details><summary>Indice 3</summary>

```python
def latency_ms_26(model, x, reps=20, warmup=3):
    model.eval()
    times = []
    with torch.inference_mode():
        ...                  # warmup calls, not timed
        for _ in range(reps):
            start = time.perf_counter()
            ...              # one call, then times.append(time.perf_counter() - start)
    return 1000 * float(np.median(times))
```

</details>

### Ex B1.27 — Publier un modèle fine-tuné : safetensors, configuration et fiche modèle qui cite le modèle de base 🛠️

<details><summary>Indice 1</summary>

Relis « Publier un modèle dérivé » (§201.9, avec l'extrait `save_file` / `load_file`), ta fiche modèle du ch. 23 (🛠️ 23.28), et les deux data cards citées par l'énoncé pour les licences et la source de CIFAR-10.

</details>
<details><summary>Indice 2</summary>

`publish_27` : crée le dossier (`mkdir(parents=True, exist_ok=True)`) ; pour les poids, un dictionnaire `{nom: tenseur.detach().cpu().contiguous()}` et `save_file(..., metadata={...})` avec des valeurs **chaînes** (`str(10)`) ; pour la configuration, un dictionnaire Python écrit par `json.dumps(..., indent=2)` (les versions viennent de `torch.__version__` et `torchvision.__version__`) ; pour la fiche, un texte Markdown avec les cinq titres `## ` dans l'ordre, rempli avec les chiffres de ton entraînement (`pct(test_accuracy)` écrit l'accuracy à la française). `load_published_27` : `json.loads` de la configuration, `resnet18(weights=None)`, une tête `nn.Linear(512, config["num_classes"])`, puis `load_state_dict(load_file(...))` et `eval()`.

</details>
<details><summary>Indice 3</summary>

```python
state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
save_file(state, folder / "model.safetensors", metadata={"base_model": BASE_MODEL_27, "num_classes": "10"})
...                          # config.json (json.dumps), then README.md with the "## " sections of CARD_SECTIONS_27
```
Pour recharger : `model.load_state_dict(load_file(folder / "model.safetensors"))`, jamais `torch.load`.

</details>

### Ex B1.28 — Défi : au moins 90 % sur le test CIFAR-10 avec 100 images par classe et 10 minutes de CPU 🏆

<details><summary>Indice 1</summary>

Relis tes propres mesures : B1.21 (fine-tuning et linear probe à 100 images par classe), B1.24 (zéro-shot), B1.25 (prototypes) et B1.26 (accuracy et coût de trois backbones). Laquelle de ces approches est déjà la plus proche de 90 %, et pour quel coût ?

</details>
<details><summary>Indice 2</summary>

Compare trois familles sur la **validation** : un ResNet fine-tuné (B1.19) ; un backbone gelé, avec un linear probe ou des prototypes sur ses embeddings ; le zéro-shot de CLIP, éventuellement combiné aux prototypes. Pour un backbone gelé, les embeddings se calculent une fois (`transfer.cached_embeddings`), et les réglages (`C`, la résolution) se choisissent sur `X_VAL` par une petite grille, **avant** de lancer `evaluate_28`. Chronomètre chaque essai à froid : sur deux cœurs, DINOv2-small coûte ≈ 25 à 30 ms par image à 140 pixels, ≈ 70 à 80 ms à 224 ; CLIP ≈ 50 ms à 224.

</details>
<details><summary>Indice 3</summary>

```python
def method_28(X, y):
    E = ...                  # the embeddings of X by the backbone you chose (cached_embeddings, subset="train")
    clf = ...                # a classifier on E, its settings chosen on X_VAL beforehand

    def predict(images):
        ...                  # the embeddings of `images`, same backbone and same size, then clf
    return predict
```

</details>