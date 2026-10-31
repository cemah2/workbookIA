# B1 · Transfer learning et modèles pré-entraînés — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/b1_transfer/06_mes_reponses.md` (créée par `python tools/start_chapter.py B1`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ B1.1 à B1.5 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », la dérivation ∂ B1.6, l'estimation 🧮 B1.7, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"B"`), plusieurs lettres entre guillemets pour un choix multiple (`"AC"`), une forme de tenseur entre guillemets (`"(2, 3)"`), un nom de classe entre guillemets (`"chat"`).
> Conventions du chapitre (fiche) : $\phi(\mathbf{x}) \in \mathbb{R}^d$ est l'embedding d'une image $\mathbf{x}$ (la sortie du backbone), $K$ le nombre de classes de la nouvelle tâche, $\theta_0$ les poids pré-entraînés ; les groupes de paramètres sont numérotés $\ell = 1, \dots, L$ de l'entrée vers la tête ; un mini-batch d'images a la forme $(N, C, H, W)$ ; $s$ est l'échelle des logits de CLIP. Les blocs d'un ResNet portent les noms de torchvision (`conv1`, `bn1`, `maxpool`, `layer1` à `layer4`, `avgpool`, `fc`), et les numéros (d'exemples, de vecteurs) commencent à 0, comme en Python.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · 🚀 GPU Colab conseillé en mode complet · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### B1.Q1 — Réutiliser plutôt que repartir de zéro : trois raisons et une limite 🧠 ⏱️ 3 min
*Fiche §201.1 · parcours R*

a) Parmi ces raisons de partir d'un modèle pré-entraîné, lesquelles sont justes ? (A) il faut beaucoup moins d'images étiquetées à la main ; (B) l'entraînement coûte beaucoup moins de calcul ; (C) le modèle n'hérite d'aucun biais de ses données d'origine ; (D) ses premières couches savent déjà détecter des bords et des textures. Écris leurs lettres.
b) CLIP a été entraîné sur combien de millions de paires image-texte ?
c) Vrai ou faux : avec 500 images étiquetées et un ordinateur portable, entraîner un ResNet de zéro est en général le meilleur choix.
d) La principale limite du transfert : (A) un modèle pré-entraîné demande un GPU, aussi bien pour l'entraîner que pour s'en servir ensuite ; (B) il aide moins, voire nuit, quand tes images ressemblent peu à celles du pré-entraînement ; (C) on ne peut pas changer sa tête.
e) Vrai ou faux : le calcul coûteux du pré-entraînement a été payé une fois, par ceux qui ont publié le modèle.

### B1.Q2 — Premières couches génériques, dernières couches spécialisées : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §201.1 · parcours R*

a) Les filtres de la première couche d'un CNN entraîné sur des photos ressemblent à des détecteurs de bords orientés et de taches de couleur, quelle que soit la tâche d'entraînement.
b) Les features de la dernière couche du backbone sont les plus générales du réseau.
c) Plus une couche est profonde, plus ses features sont liées à la tâche source.
d) Pour une tâche cible très éloignée d'ImageNet, les features de la dernière couche du backbone restent en général les plus utiles.
e) Un linear probe entraîné sur la sortie d'un bloc intermédiaire mesure si l'information des classes y est lisible par un classifieur linéaire.

### B1.Q3 — Où trouver un modèle pré-entraîné, et quoi lire avant de s'en servir 🧠 ⏱️ 3 min
*Fiche §201.2 · parcours R*

a) Pour charger DINOv2 depuis le Hugging Face Hub : (A) `torchvision.models.get_model("dinov2", weights="DEFAULT")` ; (B) `torch.load("dinov2.pt", weights_only=True)` ; (C) `AutoModel.from_pretrained("facebook/dinov2-small")`.
b) Parmi ces informations, lesquelles se cherchent dans la model card avant d'utiliser un modèle ? (A) la licence ; (B) les données d'entraînement ; (C) les usages déconseillés ; (D) la vitesse de ta connexion. Écris leurs lettres.
c) Vrai ou faux : un modèle téléchargeable sans compte ni conditions à accepter peut servir dans un produit commercial.
d) La licence des poids `facebook/dinov2-*` aujourd'hui : (A) CC BY-NC, comme à leur sortie ; (B) Apache 2.0 ; (C) aucune, comme CLIP.
e) Vrai ou faux : l'argument `revision=` de `from_pretrained` fixe la version du dépôt que l'on charge.

### B1.Q4 — weights.transforms() : pourquoi le prétraitement fait partie du modèle 🧠 ⏱️ 3 min
*Fiche §201.2, §201.5 · parcours R*

a) Dans quel ordre `weights.transforms()` applique-t-il ses étapes ? (A) normaliser, ramener dans $[0, 1]$, redimensionner, recadrer au centre ; (B) recadrer au centre, ramener dans $[0, 1]$, normaliser, redimensionner ; (C) redimensionner, recadrer au centre, ramener dans $[0, 1]$, normaliser.
b) Vrai ou faux : donner à un ResNet ImageNet des pixels dans $[0, 1]$, sans la normalisation, déclenche une erreur de forme.
c) Pour `ResNet50_Weights.IMAGENET1K_V2`, combien de pixels mesure le petit côté après le redimensionnement (`resize_size`) ?
d) Pour CLIP, on normalise les images avec : (A) la moyenne et l'écart-type que donne le processeur de CLIP ; (B) ceux d'ImageNet, comme pour tous les modèles de torchvision ; (C) ceux de ton propre dataset.
e) Vrai ou faux : deux jeux de poids d'une même architecture peuvent demander deux prétraitements différents.

### B1.Q5 — Extraction de features, linear probe, fine-tuning : qui s'entraîne ? 🧠 ⏱️ 3 min
*Fiche §201.3, §201.4 · parcours R*

a) En extraction de features, qu'est-ce qui s'entraîne ? (A) le classifieur posé sur les embeddings, seul ; (B) le backbone et la nouvelle tête, avec un petit learning rate ; (C) les batchnorms du backbone.
b) En fine-tuning partiel, qu'est-ce qui s'entraîne ? (A) tout le réseau, avec un petit learning rate ; (B) la tête seule, le backbone restant gelé ; (C) les derniers blocs du backbone et la tête.
c) Vrai ou faux : un k plus proches voisins sur des embeddings figés n'apprend aucun poids par descente de gradient.
d) Combien de paramètres a un linear probe sur des embeddings de dimension $d = 768$ pour $K = 7$ classes ?
e) Vrai ou faux : en extraction de features sans augmentation, on recalcule les embeddings à chaque epoch.

### B1.Q6 — Pourquoi un learning rate 10 à 100 fois plus petit pour fine-tuner ? 🧠 ⏱️ 3 min
*Fiche §201.4 · parcours R*

a) La raison principale : (A) les poids pré-entraînés sont au fond d'une vallée de la loss, et de grands pas les en feraient sortir ; (B) AdamW diverge dès que le learning rate dépasse $10^{-4}$, quels que soient le réseau et la loss ; (C) un petit learning rate fait converger plus vite.
b) Vrai ou faux : au début du fine-tuning, la tête neuve, aléatoire, renvoie vers le backbone des gradients grands et désordonnés.
c) LP-FT consiste à : (A) fine-tuner tout le réseau, puis entraîner un linear probe sur le résultat ; (B) entraîner d'abord un linear probe, puis fine-tuner le réseau ; (C) alterner une epoch de linear probe et une epoch de fine-tuning.
d) Vrai ou faux : avec des learning rates discriminatifs, les premières couches reçoivent le plus grand learning rate.
e) Un warmup au début d'un fine-tuning sert surtout à : (A) réduire la mémoire qu'occupent les états de l'optimiseur pendant les premiers pas ; (B) remplacer les batchnorms ; (C) faire de petits pas tant que la tête est encore mauvaise.

### B1.Q7 — Batchnorm pendant le fine-tuning sur de tout petits batches 🧠 ★★ ⏱️ 3 min
*Fiche §201.4 · parcours R*

a) En mode `train()`, une batchnorm normalise avec : (A) les statistiques glissantes apprises sur ImageNet ; (B) la moyenne et la variance du mini-batch en cours ; (C) une moyenne de 0 et un écart-type de 1, fixés.
b) Vrai ou faux : en mode `eval()`, les statistiques glissantes d'une batchnorm ne sont pas mises à jour.
c) Vrai ou faux : `model.train()` laisse en mode `eval()` les batchnorms qu'on y avait mises.
d) Avec des mini-batches de 4 images et des batchnorms en mode `train()`, le symptôme typique est : (A) une erreur de forme au premier pas ; (B) des `nan` dans les poids dès la première epoch, à cause d'une division par zéro ; (C) une loss d'entraînement qui baisse et une accuracy de validation qui s'effondre.
e) Vrai ou faux : une batchnorm en mode `eval()` peut quand même entraîner ses paramètres $\gamma$ et $\beta$.

### B1.Q8 — Peu ou beaucoup de données, domaine proche ou lointain : la bonne stratégie 🧠 ⏱️ 3 min
*Fiche §201.6, §201.7 · parcours R*

a) 300 photos de produits d'épicerie, un domaine proche d'ImageNet : (A) extraction de features et linear probe ; (B) fine-tuning complet avec le learning rate d'un entraînement de zéro ; (C) un ResNet-50 entraîné de zéro.
b) 200 000 photos d'animaux sauvages : (A) un k plus proches voisins sur les pixels ; (B) un fine-tuning complet avec un petit learning rate ; (C) un linear probe, sans jamais toucher au backbone.
c) 150 spectrogrammes de chants d'oiseaux : (A) le cas facile, un linear probe sur la dernière couche suffit ; (B) un ResNet-50 entraîné de zéro ; (C) le cas difficile : sonder les blocs intermédiaires et chercher un modèle pré-entraîné plus proche.
d) Vrai ou faux : avec beaucoup d'images d'un domaine lointain, on compare le fine-tuning complet à un entraînement de zéro.
e) Vrai ou faux : sur une courbe d'apprentissage, une seule graine suffit pour le point à 10 images par classe.

### B1.Q9 — Transfert négatif et oubli catastrophique : reconnaître les symptômes 🧠 ⏱️ 3 min
*Fiche §201.7 · parcours R*

a) On parle de transfert négatif quand : (A) l'accuracy de test baisse d'une epoch à l'autre ; (B) le modèle pré-entraîné fait moins bien que le même modèle entraîné de zéro, à budget égal ; (C) les gradients changent de signe pendant le fine-tuning, et les poids reculent.
b) Après un fine-tuning sur CIFAR-10, on remet l'ancienne tête, et le modèle ne reconnaît plus les classes d'ImageNet : (A) de l'oubli catastrophique ; (B) du transfert négatif ; (C) de l'overfitting.
c) Vrai ou faux : on ne peut affirmer un transfert négatif qu'en comparant à une baseline entraînée de zéro avec le même soin.
d) Vrai ou faux : un grand learning rate et beaucoup d'epochs limitent l'oubli catastrophique.
e) Laquelle de ces parades ne limite **pas** l'oubli ? (A) la pénalité L2-SP ; (B) des learning rates discriminatifs ; (C) un weight decay plus fort.

### B1.Q10 — Classer sans exemple d'entraînement : le zéro-shot de CLIP 🧠 ⏱️ 3 min
*Fiche §201.8 · parcours R*

a) Pour classer en zéro-shot, CLIP compare l'embedding de l'image : (A) aux embeddings de phrases écrites à partir d'un gabarit, une par classe ; (B) aux embeddings de quelques images étiquetées de chaque classe, gardés en mémoire ; (C) aux 1 000 sorties d'une tête entraînée sur ImageNet.
b) Vrai ou faux : `logit_scale_init_value` de `config.json` est l'échelle $s$ qu'utilise le modèle entraîné.
c) Vrai ou faux : sans la multiplication par $s$, la softmax des cosinus serait presque uniforme.
d) Un ensemble de prompts, c'est : (A) une phrase différente pour chaque image ; (B) plusieurs CLIP entraînés séparément, dont on moyenne les probabilités ; (C) pour chaque classe, la moyenne des embeddings normalisés de plusieurs gabarits, renormalisée.
e) Vrai ou faux : un modèle zéro-shot comme CLIP n'a vu aucune image pendant son entraînement.

### B1.Q11 — Auto-supervisé (DINOv2) ou supervisé sur ImageNet : quels embeddings ? 🧠 ★★ ⏱️ 3 min
*Fiche §201.8, §201.9*

a) DINOv2 apprend : (A) à rapprocher chaque image de sa légende ; (B) sans labels, un élève devant produire les mêmes sorties qu'un professeur qui voit d'autres vues de l'image ; (C) à prédire les 1 000 classes d'ImageNet à partir d'images recadrées au hasard.
b) Vrai ou faux : DINOv2-small est une version distillée d'un modèle bien plus grand.
c) Avec `AutoModel`, l'embedding d'une image par DINOv2 se lit dans : (A) `pooler_output`, le token `CLS` après la dernière normalisation ; (B) `pixel_values`, la sortie du processeur ; (C) `last_hidden_state[:, 1:]`, les tokens des morceaux d'image.
d) Vrai ou faux : le backbone qui a le meilleur top-1 sur ImageNet donne toujours le meilleur linear probe sur ta tâche.
e) DINOv2-base découpe une image en carrés de 14 pixels de côté et ajoute un token `CLS`. Combien de tokens rend-il pour une image de 448 × 448 ?

<a id="rappels"></a>

## 🔁 Rappels

### B1.R1 — Ch. 29 : licence d'un dataset et décalage de distribution 🔁 ★ ⏱️ 5 min
*Ch. 29 (licences, décalage de distribution) · fiche §201.2, §201.7 · parcours R*

a) Vrai ou faux : un dataset sous licence CC BY-NC peut entraîner un modèle vendu dans un produit.
b) Les conditions d'ImageNet réservent la base : (A) à la recherche non commerciale et à l'enseignement ; (B) aux entreprises qui ont acheté une licence ; (C) à personne : elle est dans le domaine public.
c) Un décalage de distribution, c'est : (A) un découpage en entraînement et test qui n'a pas été stratifié ; (B) un learning rate qui change pendant l'entraînement ; (C) des données d'usage qui ne suivent pas la distribution des données d'entraînement.
d) Vrai ou faux : la distance de domaine est la version « features » du décalage de distribution.
e) Pour estimer ce que le décalage coûte à ton modèle : (A) l'entraîner plus d'epochs ; (B) l'évaluer sur un test venu d'une autre source, ou de l'usage visé ; (C) l'évaluer sur un plus grand découpage du même dataset.

### B1.R2 — Ch. 23 : geler un réseau et remplacer sa tête en PyTorch 🔁 ★ ⏱️ 5 min
*Ch. 23 (gel des poids, nouvelle tête, sauvegarde) · fiche §201.3, §201.4 · parcours R*

a) Geler les poids d'un réseau en PyTorch : (A) `model.eval()` ; (B) `p.requires_grad_(False)` pour chacun de ses paramètres ; (C) `torch.no_grad()` autour de `optimizer.step()`.
b) Vrai ou faux : `model.eval()` gèle les poids.
c) La tête d'un réseau est `nn.Linear(1280, 1000)` ; on la remplace pour 4 classes. Combien de paramètres a la nouvelle tête ?
d) Vrai ou faux : une couche créée par `nn.Linear` a `requires_grad=True`, même si l'on a gelé le reste du modèle avant de la créer.
e) Pour recharger des poids d'un fichier reçu : (A) `torch.load(f, weights_only=True)`, le défaut depuis PyTorch 2.6 ; (B) `torch.load(f, weights_only=False)`, pour être sûr de tout relire ; (C) `pickle.load(f)`.

### B1.R3 — Ch. 21 : ce que détectent les filtres des premières et des dernières couches 🔁 ★ ⏱️ 5 min
*Ch. 21 (filtres, cartes d'activation, ResNet, 📦 21.27) · fiche §201.1 · parcours R*

a) Les filtres de `conv1` d'un ResNet-18 entraîné sur ImageNet, affichés comme de petites images, montrent : (A) des visages et des roues ; (B) des bords orientés et des taches de couleur ; (C) du bruit sans structure.
b) Combien de poids a `conv1` d'un ResNet-18 : 64 filtres de 7 × 7 sur 3 canaux, sans biais ?
c) Vrai ou faux : la carte d'activation de `layer4` a une plus faible résolution spatiale que celle de `conv1`.
d) Vrai ou faux : on lit facilement ce que détecte un filtre de `layer4` en affichant ses poids comme une petite image.
e) Le pooling moyen global transforme une carte d'activation de forme $(C, H, W)$ en : (A) un vecteur $(C,)$, la moyenne de chaque canal ; (B) une image $(H, W)$, la moyenne des canaux ; (C) une carte $(C, H/2, W/2)$.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex B1.1 — Nouvelle tête : formes des tenseurs de l'image 32×32 redimensionnée jusqu'aux logits ✏️ ★ ⏱️ 15 min
**Objectif :** suivre la forme d'un mini-batch à travers un ResNet pré-entraîné, de l'image agrandie aux logits de la nouvelle tête.
**Prérequis :** fiche §201.5 · ch. 21 (taille de sortie d'une convolution) · ch. 23 (nouvelle tête) · **Fil rouge :** CIFAR-10 · **Parcours :** R, M

Un mini-batch de 16 images de CIFAR-10 (3 canaux, 32 × 32) est agrandi à 128 × 128 (`v2.Resize(128)`), normalisé, puis passe dans un ResNet-18 de torchvision dont on a remplacé la tête par `nn.Linear(model.fc.in_features, 10)`. L'architecture :
- le stem : `conv1` (64 filtres 7 × 7, stride 2, padding 3), `bn1`, une ReLU, puis `maxpool` (3 × 3, stride 2, padding 1) ;
- `layer1` garde la taille, avec 64 canaux ; `layer2`, `layer3` et `layer4` ont 128, 256 et 512 canaux, et la première convolution de chacun (3 × 3, padding 1) a un stride 2 ;
- `avgpool` est un pooling moyen adaptatif vers 1 × 1, suivi d'un aplatissement, puis `fc`.

Taille de sortie d'une convolution ou d'un pooling (ch. 21) : $\left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$. Donne les formes avec l'axe des exemples.

a) La forme du tenseur après le stem.
b) La forme après `layer2`.
c) La forme après `layer4`.
d) La forme de $\phi(\mathbf{x})$, après `avgpool` et l'aplatissement.
e) La forme des logits.
f) Sans agrandir les images (32 × 32 en entrée), le côté de la carte d'activation de `layer4`.
g) Avec des images agrandies à 100 × 100, le côté de la carte d'activation de `layer4`.
h) Vrai ou faux : la même couche `fc` sert, que les images soient agrandies à 100, à 128 ou à 224 pixels.

Dans ta copie : combien de fois plus de pixels traite le réseau à 128 qu'à 32, et donc, à peu près, combien de fois plus de calcul ? Qu'est-ce qui, dans l'architecture, rend la tête indépendante de la taille de l'image ?

### Ex B1.2 — k plus proches voisins cosinus sur six embeddings donnés ✏️ ★ ⏱️ 10 min
**Objectif :** classer un embedding par ses plus proches voisins au sens du cosinus, et voir ce que changent la norme et une égalité de votes.
**Prérequis :** fiche §201.3 · 0B (similarité cosinus) · ch. 13 (k plus proches voisins) · **Parcours :** M

Six images de référence ont pour embeddings (en dimension 3, pour le calcul à la main) :

| Vecteur | Embedding | Classe |
|---|---|---|
| $\mathbf{e}_0$ | $(3, 0, 4)$ | oiseau |
| $\mathbf{e}_1$ | $(0, 3, 4)$ | avion |
| $\mathbf{e}_2$ | $(16, 2, 8)$ | avion |
| $\mathbf{e}_3$ | $(2, 1, 2)$ | oiseau |
| $\mathbf{e}_4$ | $(1, 0, 0)$ | avion |
| $\mathbf{e}_5$ | $(0, 1, 1)$ | oiseau |

On classe l'image $\mathbf{q} = (1, 2, 2)$. Règle de vote, celle de `KNeighborsClassifier` de scikit-learn : chaque voisin vote pour sa classe, et **à égalité de votes, la classe qui vient la première dans l'ordre alphabétique** l'emporte (l'ordre de `classes_`), quelle que soit la distance des voisins.

a) $\cos(\mathbf{q}, \mathbf{e}_5)$, à 3 décimales.
b) Le numéro du plus proche voisin de $\mathbf{q}$ au sens du cosinus.
c) La classe prédite avec $k = 2$.
d) La classe prédite avec $k = 3$.
e) Le numéro du vecteur qui a le plus grand produit scalaire avec $\mathbf{q}$, sans normaliser.
f) La distance euclidienne **au carré** entre $\mathbf{q}$ et $\mathbf{e}_1$, une fois les deux vecteurs ramenés à la norme 1, à 3 décimales.
g) Vrai ou faux : si l'on divise seulement $\mathbf{e}_2$ par sa norme, le vecteur de plus grand produit scalaire avec $\mathbf{q}$ (toujours sans normaliser les autres) devient le plus proche voisin au sens du cosinus.

Dans ta copie : pourquoi prend-on souvent un $k$ impair quand il y a deux classes ? Pourquoi la norme d'un embedding trompe-t-elle un classement par produit scalaire ?

### Ex B1.3 — Learning rates discriminatifs et warmup : le learning rate de chaque groupe, pas à pas ✏️ ★ ⏱️ 15 min
**Objectif :** calculer le learning rate de chaque groupe de paramètres à chaque pas, pendant le warmup puis pendant la décroissance en cosinus.
**Prérequis :** fiche §201.4 · ch. 19 (warmup et cosinus) · **Parcours :** M

On fine-tune un ResNet en quatre groupes de paramètres : la tête ($\ell = 4$), `layer4` ($\ell = 3$), `layer3` ($\ell = 2$) et `layer2` ($\ell = 1$). Le learning rate de pointe du groupe $\ell$ est $\eta_\ell = \eta_{\text{tête}}\,\gamma^{L-\ell}$, avec $L = 4$, $\eta_{\text{tête}} = 10^{-3}$ et $\gamma = 0{,}3$. Le planning, appliqué à chaque groupe par `LambdaLR`, multiplie ce learning rate de pointe par
$$f(t) = \begin{cases} \dfrac{t}{t_{\text{warmup}}} & \text{si } t < t_{\text{warmup}}, \\[2mm] \dfrac{1}{2}\left(1 + \cos\left(\pi\,\dfrac{t - t_{\text{warmup}}}{t_{\text{total}} - t_{\text{warmup}}}\right)\right) & \text{sinon,} \end{cases}$$
avec $t_{\text{warmup}} = 5$ et $t_{\text{total}} = 105$ ; $t = 0, 1, 2, \dots$ est le numéro du pas. **Donne les learning rates en unités de $10^{-5}$** (par exemple, $10^{-3}$ s'écrit 100).

a) Le learning rate de pointe de `layer2`, à 1 décimale.
b) Le learning rate de la tête au pas $t = 2$.
c) Le learning rate de `layer4` au pas $t = 3$.
d) Le rapport entre le learning rate de la tête et celui de `layer2` au pas $t = 3$, à 2 décimales.
e) Le learning rate de `layer3` au pas $t = 55$, à 2 décimales.
f) Le learning rate de la tête au pas $t = 30$, à 2 décimales.
g) Vrai ou faux : au pas $t = 0$, AdamW (avec son weight decay) ne modifie aucun poids.
h) Vrai ou faux : la fonction donnée à `LambdaLR` rend directement le learning rate de chaque groupe.

Dans ta copie : écris la liste `param_groups` de PyTorch qui donne ces learning rates de pointe, et la fonction $f$ en Python.

### Ex B1.4 — Compter les paramètres entraînables : tête seule, dernier bloc, réseau entier ✏️ ★★ ⏱️ 20 min
**Objectif :** compter ce que l'on entraîne vraiment selon la stratégie, couche par couche, à partir de l'architecture.
**Prérequis :** Ex B1.1 · ch. 16 (paramètres d'une couche) · fiche §201.4 · **Fil rouge :** CIFAR-10 · **Parcours :** R, M

Le ResNet-18 de l'Ex B1.1. Avec sa tête ImageNet, `nn.Linear(512, 1000)`, il a 11 689 512 paramètres (`weights.meta["num_params"]`). Ses convolutions n'ont **pas de biais** ; chaque batchnorm a deux paramètres par canal, $\gamma$ et $\beta$ (ses statistiques glissantes sont des buffers, pas des paramètres). `layer4` est fait de deux blocs :
- bloc 0 : une convolution 3 × 3 de 256 vers 512 canaux (stride 2) et sa batchnorm, une convolution 3 × 3 de 512 vers 512 et sa batchnorm, et un raccourci (`downsample`) : une convolution 1 × 1 de 256 vers 512 (stride 2) et sa batchnorm ;
- bloc 1 : deux convolutions 3 × 3 de 512 vers 512, chacune suivie de sa batchnorm.

a) Le nombre de paramètres de la nouvelle tête à $K = 10$ classes.
b) Le nombre de paramètres du réseau entier muni de cette nouvelle tête.
c) Le nombre de poids d'une convolution 3 × 3 de 512 vers 512 canaux.
d) Le nombre de paramètres de `layer4`.
e) Le nombre de paramètres entraînables quand on dégèle `layer4` et la tête, tout le reste étant gelé.
f) La part de e) dans b), en pourcentage, à 1 décimale.
g) Le nombre de paramètres entraînables si, en plus de e), on gèle aussi les $\gamma$ et $\beta$ des batchnorms de `layer4`.
h) Vrai ou faux : mettre les batchnorms en mode `eval()` (`freeze_batchnorm`) réduit le nombre de paramètres entraînables.

Dans ta copie : AdamW garde deux nombres par paramètre entraînable (ch. 19). Combien de Mo ($10^6$ octets) d'états d'optimiseur en `float32` pour e), et pour l'entraînement complet b) ?

### Ex B1.5 — Zéro-shot CLIP à la main : cosinus, logit_scale et softmax ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer les probabilités du zéro-shot de CLIP à partir des cosinus, et voir ce que fait l'échelle $s$.
**Prérequis :** Ex B1.2 · fiche §201.8 · ch. 17 (softmax) · **Parcours :** R, M

Une image de CIFAR-10 et les quatre phrases « a photo of a {classe} » ont des embeddings normalisés dont les cosinus avec l'image valent :

| Classe | (A) avion | (B) oiseau | (C) bateau | (D) camion |
|---|---|---|---|---|
| $\cos(\mathbf{u}_x, \mathbf{v}_c)$ | 0,24 | 0,30 | 0,28 | 0,19 |

Le paramètre entraîné du modèle vaut `logit_scale` $= 4{,}6052$, et `config.json` donne `logit_scale_init_value` $= 2{,}6592$. Rappel : $p(c \mid \mathbf{x}) = \dfrac{\exp(s \cos(\mathbf{u}_x, \mathbf{v}_c))}{\sum_{c'} \exp(s \cos(\mathbf{u}_x, \mathbf{v}_{c'}))}$, avec $s = \exp(\texttt{logit\_scale})$.

a) $p(\text{oiseau} \mid \mathbf{x})$, à 3 décimales.
b) $p(\text{bateau} \mid \mathbf{x})$, à 3 décimales.
c) Le rapport $p(\text{oiseau} \mid \mathbf{x}) / p(\text{bateau} \mid \mathbf{x})$, à 3 décimales.
d) $p(\text{avion} \mid \mathbf{x})$, à 4 décimales.
e) $p(\text{oiseau} \mid \mathbf{x})$ si l'on prend par erreur $s = \exp(\texttt{logit\_scale\_init\_value})$, à 2 décimales.
f) Vrai ou faux : ajouter une cinquième classe, de cosinus 0,10, ne change pas le rapport c).

Dans ta copie : que deviendraient les probabilités si l'on oubliait de normaliser les embeddings de texte, et que la phrase « a photo of a truck » avait une norme deux fois plus grande que les autres ?

### Ex B1.6 — Rester près des poids pré-entraînés : le gradient de la pénalité L2-SP ∂ ★★ ⏱️ 25 min
**Objectif :** dériver le gradient de la pénalité L2-SP, en déduire son effet sur un pas de descente de gradient, et le comparer au weight decay vers 0.
**Prérequis :** fiche §201.4 (L2-SP) · ch. 9 (régularisation L2) · ch. 19 (SGD, AdamW et weight decay découplé) · **Parcours :** M

On note $\theta \in \mathbb{R}^m$ les poids du backbone, $\theta_0$ leurs valeurs pré-entraînées, et $\Omega(\theta) = \frac{\lambda}{2}\lVert\theta - \theta_0\rVert^2 = \frac{\lambda}{2}\sum_{j=1}^{m} (\theta_j - \theta_{0,j})^2$, avec $\lambda > 0$. On minimise $L(\theta) + \Omega(\theta)$ par descente de gradient (SGD), de learning rate $\eta$. Réponds dans ta copie.

1. Calcule $\frac{\partial \Omega}{\partial \theta_j}$, puis écris $\nabla \Omega$ sous forme vectorielle.
2. Écris un pas de SGD sur $L + \Omega$. On suppose, le temps de cette question, que la loss ne pousse plus : $\nabla L(\theta_t) = 0$. Montre que $\theta_{t+1} - \theta_0 = (1 - \eta\lambda)(\theta_t - \theta_0)$, puis donne $\theta_t - \theta_0$ en fonction de $\theta_{\text{départ}} - \theta_0$ après $t$ pas. Pour quelles valeurs de $\eta\lambda$ l'écart diminue-t-il à chaque pas sans changer de signe ?
3. Application : $\eta = 0{,}5$ et $\lambda = 0{,}2$, un poids pré-entraîné $\theta_0 = -0{,}4$ qu'un premier fine-tuning a amené à $0{,}6$. Toujours avec $\nabla L = 0$, que vaut ce poids après 10 pas ? Combien de pas faut-il pour que l'écart à $\theta_0$ soit divisé par 2 au moins ?
4. Le weight decay classique ajoute $\frac{\lambda}{2}\lVert\theta\rVert^2$. Écris son pas de SGD dans les mêmes conditions. Montre que L2-SP est exactement le weight decay appliqué à l'écart $\delta = \theta - \theta_0$. Vers où chacun tire-t-il les poids, et pourquoi 0 n'a-t-il « rien de spécial » pour un réseau pré-entraîné ?
5. On combine les deux, $\frac{\lambda_1}{2}\lVert\theta - \theta_0\rVert^2 + \frac{\lambda_2}{2}\lVert\theta\rVert^2$, toujours avec $\nabla L = 0$. Vers quel point les poids convergent-ils ? Qu'en conclus-tu sur la ligne « weight decay et L2-SP ensemble » du tableau des pièges ?
6. Supposons maintenant que la loss pousse avec un gradient constant $\mathbf{g}$ : $\nabla L(\theta) = \mathbf{g}$ pour tout $\theta$. Où se trouve le point d'équilibre ? Comment $\lambda$ règle-t-il la distance à laquelle les poids peuvent s'éloigner de $\theta_0$ ?
7. Avec AdamW, ajouter $\Omega$ à la loss ne donne plus le facteur $(1 - \eta\lambda)$ de la question 2 : pourquoi (pense à la différence entre pénalité L2 et weight decay découplé, ch. 19) ? Et pourquoi la tête neuve ne reçoit-elle pas de pénalité L2-SP ?

### Ex B1.7 — Fermi : pré-entraîner sur ImageNet ou fine-tuner sur 5 000 images (GPU-heures, énergie, coût) 🧮 ★★ ⏱️ 20 min
**Objectif :** estimer en ordres de grandeur ce que coûte un pré-entraînement et ce que coûte sa réutilisation, pour voir ce que le transfer learning économise.
**Prérequis :** fiche §201.1, §201.9 · ch. 20 (coût d'un entraînement) · **Parcours :** R, M

Hypothèses de calcul (des ordres de grandeur, pas des mesures) : une passe avant de ResNet-50 sur une image de 224 × 224 coûte 4,1 milliards d'opérations (`weights.meta["_ops"]`, fiche §201.9) ; un pas d'entraînement (passe avant et rétropropagation) coûte environ 3 passes avant ; le pré-entraînement dure 90 epochs sur les 1 281 167 images d'ImageNet-1k ; un GPU de centre de calcul tient en pratique $10^{13}$ opérations par seconde, un CPU de portable $10^{11}$ ; un serveur consomme 0,5 kW par GPU ; une heure de GPU louée coûte 2 €. Réponds dans ta copie.

1. Le nombre d'opérations du pré-entraînement, puis sa durée en GPU-heures, son énergie en kWh et son coût de location.
2. Une équipe essaie en moyenne 30 réglages avant le bon (learning rates, augmentations, planning). Que devient le coût en GPU-heures ? Et l'annotation : 1,28 million d'images, trois annotateurs par image, 0,02 € par annotation ?
3. Le fine-tuning complet du même ResNet-50 sur 5 000 images, 10 epochs : le nombre d'opérations, puis la durée sur un GPU et sur le CPU d'un portable.
4. Un linear probe sur ces 5 000 images : une seule passe avant pour calculer les embeddings, puis une régression logistique (quelques secondes). La durée sur le CPU.
5. Combien de fois le pré-entraînement (question 1, sans les essais) coûte-t-il plus de calcul que le fine-tuning de la question 3 ? Et CLIP, avec ses 12 jours sur 256 GPU V100 (fiche §201.1) : combien de GPU-heures, et combien de pré-entraînements de ResNet-50 de la question 1 ?
6. Qui paie le pré-entraînement, combien de fois, et combien de fois est-il réutilisé ? Cite deux coûts que tes calculs ne comptent pas.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B1.8 — Le transfer learning expliqué en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant ce qu'est le transfer learning, pourquoi il marche et quand il échoue.
**Prérequis :** fiche §201.1 · **Parcours :** R

Explique en cinq lignes au plus, à quelqu'un qui ne fait pas de deep learning, ce qu'est le transfer learning. Ton explication doit faire apparaître (1) ce qu'un réseau a appris sur un grand dataset, (2) ce qu'on garde et ce qu'on remplace, (3) pourquoi cela demande peu d'images, et (4) un cas où cela marche mal. Une image de la vie courante est la bienvenue.

### Ex B1.9 — Biais et licences hérités d'un modèle pré-entraîné : un projet d'imagerie médicale ⚖️ ★★ ⏱️ 25 min
**Objectif :** repérer ce qu'un modèle dérivé hérite de son modèle de base (licence, biais, limites) et ce qu'il faut vérifier avant un usage médical.
**Prérequis :** fiche §201.2 (licences, model cards), §201.7 (encadré ⚖️) · ch. 29 (licences, décalage de distribution, data card) · **Parcours :** R

Une start-up veut aider des dermatologues à repérer des lésions suspectes sur des photos de peau. Elle dispose de 2 000 photos étiquetées par trois dermatologues d'un même hôpital, et hésite entre trois backbones : un ResNet-50 de torchvision (poids ImageNet), DINOv2 (`facebook/dinov2-base`) et CLIP (`openai/clip-vit-base-patch32`). Réponds dans ta copie.

1. Pour chacun des trois backbones, que dit sa licence (ou son absence de licence) et sa model card d'un usage dans un produit commercial ? Lequel écarter d'emblée, et pourquoi ?
2. Les poids ImageNet viennent d'une base réservée « à la recherche non commerciale et à l'enseignement ». Qu'est-ce qui est tranché, qu'est-ce qui ne l'est pas, et que fait en pratique une entreprise prudente ?
3. Quels biais le modèle peut-il hériter de son pré-entraînement, et quels biais viennent des 2 000 photos elles-mêmes (couleurs de peau, appareils, hôpital, annotateurs) ? Donne un exemple de chaque.
4. Comment mesurer ces biais avant un déploiement ? Quels sous-groupes, quelle métrique (pense au recall des lésions malignes), et quelles données d'évaluation ?
5. Les photos sont des données de santé. Que faut-il vérifier avant de s'en servir pour un fine-tuning (consentement, base légale, RGPD), et que doit dire la data card ?
6. Dans l'Union européenne, un logiciel d'aide au diagnostic est un dispositif médical, et un système d'IA à haut risque au sens du règlement sur l'IA. Qu'est-ce que cela change pour la documentation, l'évaluation et la surveillance après la mise sur le marché ?
7. Rédige les cinq lignes de la section « Limites et biais hérités » de la model card du modèle dérivé.

### Ex B1.10 — Yosinski et coll. (2014) : jusqu'où les features se transfèrent-elles ? 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a mesuré couche par couche ce qui se transfère d'une tâche à l'autre, et distinguer ses deux causes de dégradation.
**Prérequis :** fiche §201.1, §201.7 · ch. 21 (CNN, AlexNet) · **Parcours :** M

Lis J. Yosinski, J. Clune, Y. Bengio et H. Lipson, « How transferable are features in deep neural networks? » ([arXiv:1411.1792](https://arxiv.org/abs/1411.1792), NIPS 2014) : l'introduction, la section 2, la figure 1, la figure 2 et son commentaire (section 4.1), puis les sections 4.2 et 4.3. Réponds dans ta copie.

1. Le dispositif : comment les auteurs fabriquent-ils deux tâches A et B à partir d'ImageNet (nombre de classes, nombre d'images, nombre de découpages) ? Quel réseau utilisent-ils, et combien de couches peut-on transplanter ?
2. Définis les réseaux baseB, BnB, BnB+, AnB et AnB+. Pourquoi les auteurs ont-ils besoin du réseau « selffer » BnB, alors que la question porte sur le transfert de A vers B ?
3. Figure 2 : décris la courbe de BnB couche par couche. Comment les auteurs expliquent-ils le creux du milieu, et pourquoi la performance remonte-t-elle quand on transplante presque tout le réseau ?
4. Décris la courbe d'AnB. Quelles sont les deux causes de dégradation que l'article sépare, et laquelle domine aux couches du milieu, laquelle aux dernières couches ?
5. AnB+ (transfert puis fine-tuning) fait mieux que baseB. De combien en moyenne ? Pourquoi les auteurs écartent-ils l'explication « il a simplement été entraîné plus longtemps » ?
6. Section 4.2 : comment est construit le découpage « objets fabriqués / êtres naturels », et que devient le transfert entre ces deux moitiés ? Section 4.3 : que donnent des filtres aléatoires gelés, comparés à des features transférées d'une tâche lointaine ?
7. Qu'est-ce que ces résultats disent du choix des couches à geler en extraction de features et en fine-tuning partiel ? Donne deux limites de l'étude vue d'aujourd'hui (architecture, données, nombre d'expériences).

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### B1.E1 — Feature extraction ou fine-tuning : comment choisis-tu ? 💼 ★★ ⏱️ 10 min
*Fiche §201.3, §201.4, §201.6 · parcours R*

« Vous partez d'un modèle pré-entraîné pour une tâche de classification d'images. Comment décidez-vous entre extraire des features et fine-tuner, et que mesurez-vous pour trancher ? »

### B1.E2 — Tu as 500 images médicales étiquetées : quelle stratégie de transfer learning ? 💼 ★★ ⏱️ 10 min
*Fiche §201.6, §201.7 · parcours R*

« Un hôpital vous confie 500 radiographies étiquetées, en deux classes. Quelle stratégie proposez-vous, et comment savez-vous qu'elle marche ? »

### B1.E3 — Pourquoi un learning rate plus faible pour les couches pré-entraînées ? 💼 ★★ ⏱️ 10 min
*Fiche §201.4 · parcours R*

« Pourquoi utilise-t-on un learning rate plus faible pour les couches pré-entraînées que pour la nouvelle tête ? Comment le mettez-vous en place en PyTorch ? »

### B1.E4 — Qu'est-ce qu'un modèle de fondation, et que change le zéro-shot ? 💼 ★★ ⏱️ 10 min
*Fiche §201.8 · parcours R*

« Qu'est-ce qu'un modèle de fondation ? Que veut dire classer en zéro-shot, et qu'est-ce que cela change pour un projet ? »

### B1.E5 — Comment vérifier qu'un modèle pré-entraîné peut servir dans un produit commercial ? 💼 ★★ ⏱️ 10 min
*Fiche §201.2, §201.9 · parcours R*

« Vous avez trouvé sur le Hugging Face Hub le modèle idéal pour votre produit. Que vérifiez-vous avant de vous en servir ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`, en quatre phases : charger et sonder (B1.11 à B1.16), fine-tuner (B1.17 à B1.19), quand le transfert aide ou échoue (B1.20 à B1.23), modèles de fondation, coût et livraison (B1.24 à B1.28).

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| B1.11 | Trois modèles pré-entraînés en main : ResNet-18, MobileNetV3-small, DINOv2-small | 📦 | ★★ | 20 |
| B1.12 | Prédire : un ResNet ImageNet face à des images CIFAR-10 de 32×32 pixels | 🔮 | ★ | 10 |
| B1.13 | Extraire des embeddings figés, puis linear probe et k plus proches voisins | 📦 | ★★ | 30 |
| B1.14 | Résolution d'entrée : accuracy du linear probe contre temps de calcul (32 à 224 pixels) | 🔬 | ★★★ | 40 |
| B1.15 | Lire des projections PCA/UMAP : pixels bruts, ResNet et DINOv2 | 📈 | ★★ | 20 |
| B1.16 | Sonder chaque bloc : linear probe couche par couche, CIFAR-10 contre Fashion-MNIST | 🔬 | ★★★ | 45 |
| B1.17 | Fine-tuning partiel : dégeler le dernier bloc avec des learning rates par groupe | 📦 | ★★★ | 45 |
| B1.18 | Le fine-tuning qui détruit le modèle : quatre erreurs à trouver | 🐛 | ★★ | 25 |
| B1.19 | Fine-tuning complet : warmup + cosinus, early stopping, meilleur checkpoint 🚀 | 📦 | ★★★ | 50 |
| B1.20 | Prédire : où le pré-entraînement rapporte-t-il le plus, 10 ou 1 000 images par classe ? | 🔮 | ★ | 10 |
| B1.21 | Courbes d'apprentissage : from scratch, linear probe et fine-tuning de 10 à 1 000 images par classe 🚀 | 🔬 | ★★★ | 60 |
| B1.22 | Domaine lointain : transférer vers Fashion-MNIST, comparé au CNN from scratch | 🔬 | ★★★ | 45 |
| B1.23 | Oubli catastrophique : que reste-t-il des prédictions ImageNet après le fine-tuning ? | 🔬 | ★★ | 25 |
| B1.24 | Zéro-shot avec CLIP sur CIFAR-10 : gabarits de prompts et ensembles de prompts | 📦 | ★★ | 30 |
| B1.25 | Few-shot par prototypes de classe avec DINOv2 et CLIP (1 à 16 exemples par classe) | 📦 | ★★★ | 40 |
| B1.26 | Taille, latence et accuracy sur CPU : MobileNetV3-small, ResNet-18, DINOv2-small | 🔬 | ★★ | 30 |
| B1.27 | Publier un modèle fine-tuné : safetensors, configuration et model card qui cite le modèle de base | 🛠️ | ★★ | 25 |
| B1.28 | Défi : au moins 90 % sur le test CIFAR-10 avec 100 images par classe et 10 minutes de CPU | 🏆 | ★★★★ | 120 |

Le notebook télécharge ResNet-18 et MobileNetV3-small (torchvision) et DINOv2-small (Hugging Face Hub), environ 150 Mo en tout, puis CLIP ViT-B/32 (environ 600 Mo) pour la dernière phase ; si ce téléchargement échoue, le few-shot se rabat sur les prototypes de DINOv2. En mode rapide, les embeddings sont calculés une fois et mis en cache, sur 2 000 images de 96 à 128 pixels ; les fine-tunings complets à 224 pixels (B1.19, B1.21) sont marqués 🚀 : mode rapide sur CPU, mode complet sur un GPU de Colab.
