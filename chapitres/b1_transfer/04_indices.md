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

a) **Vrai** : bords et taches de couleur se retrouvent dans presque tous les CNN entraînés sur des photos. Pour b) et c), place la dernière couche du backbone sur l'échelle « général → spécifique ». Pour d), lesquelles deviennent les premières inutiles sur un domaine lointain ? Pour e), relis d'où vient le mot « probe » (§201.3).

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

Le transfert négatif compare deux points de départ sur la **nouvelle** tâche ; l'oubli regarde ce que devient l'**ancienne** tâche. Pour e), demande-toi vers quoi chaque parade tire les poids.

</details>
<details><summary>Indice 3</summary>

a) **B** : moins bien que le même modèle entraîné de zéro, à budget égal. Pour b), quelle tâche a été perdue ? Pour c), sans quel point de comparaison ne peut-on pas dire qu'un transfert nuit ? Pour d), relis la liste des parades. Pour e), vers quelle valeur chacune des trois parades tire-t-elle les poids : $\theta_0$, ou autre chose ?

</details>

### B1.Q10 — Classer sans exemple d'entraînement : le zéro-shot de CLIP

<details><summary>Indice 1</summary>

Relis « Le zéro-shot » et « Les gabarits comptent » (§201.8).

</details>
<details><summary>Indice 2</summary>

Dans le zéro-shot, la tête est fabriquée **à partir de mots** : une phrase par classe. Pour b), la fiche distingue une valeur **de départ** et une valeur **apprise**.

</details>
<details><summary>Indice 3</summary>

a) **A** : des phrases écrites à partir d'un gabarit, une par classe. Pour b), relis la phrase sur `logit_scale_init_value`. Pour c), que fait une softmax de trois nombres presque égaux ? Pour d), relis la fin de « Les gabarits comptent ». Pour e), « zéro-shot » veut dire zéro exemple de **la tâche cible** : sur quoi CLIP a-t-il été entraîné ?

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

a) **Faux** : NC interdit l'usage commercial. Pour b), relis la citation des conditions d'ImageNet dans l'encadré 🕰️. Pour c), lequel des trois parle de données d'usage ? Pour d), relis la dernière phrase du deuxième paragraphe du §201.7. Pour e), quel test voit des données qui ne viennent pas du même dataset ?

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

Calcule d'abord $s = e^{4{,}6052}$. Multiplie chaque cosinus par $s$ pour obtenir les logits, retranche le plus grand (les exponentielles restent petites), prends les exponentielles et divise par leur somme. Pour c) et f), écris le rapport de deux probabilités : il ne reste que l'écart de leurs logits.

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

Pour 1, relis la colonne « À savoir » du tableau des licences pour chacun des trois modèles : lequel exclut **tout** déploiement non testé ? Pour 2, sépare la licence du code, les conditions d'ImageNet et le statut des poids appris. Pour 3, cherche qui manque sur les photos d'ImageNet et dans les 2 000 photos d'un seul hôpital. Pour 4, une métrique par sous-groupe, et la plus grave des erreurs (une lésion maligne manquée). Pour 5, les photos de santé sont des données sensibles au sens du RGPD. Pour 6, ce qu'impose un système à haut risque : documentation technique, évaluation, surveillance. Pour 7, une ligne par biais ou limite, chacune avec ce qui a été mesuré.

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

### B1.E1 — Feature extraction ou fine-tuning : comment choisis-tu ?

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

Les indices des exercices du notebook (B1.11 à B1.28) seront ajoutés à la prochaine session de génération, avec les parties A à D du notebook.
