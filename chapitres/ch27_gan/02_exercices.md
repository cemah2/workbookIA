# 27 · Réseaux antagonistes génératifs (GAN) — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch27_gan/06_mes_reponses.md` (créée par `python tools/start_chapter.py 27`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ 27.1 à 27.4 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les dérivations ∂ 27.5 et 27.6, l'estimation 🧮 27.7, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"F"`), plusieurs lettres entre guillemets pour un choix multiple (`"AF"`).
> Conventions du chapitre (fiche) : générateur $G$, discriminateur $D$, bruit $\mathbf{z} \sim p_z$, données $\mathbf{x} \sim p_{\text{data}}$, distribution générée $p_g$ ; $D(\cdot) \in [0, 1]$ est la probabilité, selon $D$, que son entrée soit **vraie** ; « positif » veut dire « $D$ répond vrai » ; $\ln$ est le logarithme népérien, celui de la cross-entropy de PyTorch.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 27.Q1 — À quoi bon générer des données ? 🧠 ⏱️ 3 min
*Fiche §27.1 · livre §27.1*

a) Parmi ces usages, lesquels le livre donne-t-il pour un générateur ? (A) fabriquer des données d'entraînement en plus pour d'autres réseaux ; (B) proposer des idées, comme des plans de jardin ou un meuble assorti à un salon ; (C) compresser un fichier sans perte d'information ; (D) retrouver une photo existante dans une base d'images. Écris leurs lettres.
b) Vrai ou faux : un générateur bien entraîné rend, pour chaque bruit, une image de son jeu d'entraînement.
c) Une fois l'entraînement fini, on garde le plus souvent : (A) le discriminateur ; (B) le générateur ; (C) toujours les deux.
d) Vrai ou faux : 10 000 images générées par un GAN entraîné sur un dataset apportent à un classifieur autant d'information que 10 000 nouvelles images réelles.
e) Le « A » de GAN veut dire : (A) *Antagonistic* ; (B) *Adversarial* ; (C) *Autoencoding*.

### 27.Q2 — Le faussaire, l'inspectrice et les quatre verdicts 🧠 ⏱️ 3 min
*Fiche §27.2 · livre §27.2, §27.2.2*

Glenn (le générateur) fabrique de faux billets, Dawn (la discriminatrice) répond « vrai » (positif) ou « faux » (négatif).
a) Un vrai billet que Dawn déclare faux est : (A) un vrai positif ; (B) un faux négatif ; (C) un faux positif ; (D) un vrai négatif.
b) Un faux billet qui passe pour vrai est : (A) un vrai positif ; (B) un faux négatif ; (C) un faux positif ; (D) un vrai négatif.
c) Selon le livre, dans quel cas le **générateur** apprend-il ? (A) vrai positif ; (B) faux négatif ; (C) faux positif ; (D) vrai négatif.
d) Selon le livre, dans combien des quatre cas personne n'apprend-il ?
e) Vrai ou faux : dans la pratique (une cross-entropy binaire sur tout le mini-batch), un vrai billet bien reconnu, avec $D(\mathbf{x}) = 0{,}9$, donne quand même un gradient au discriminateur.

### 27.Q3 — Pourquoi le générateur ne voit-il jamais de vrais billets ? 🧠 ⏱️ 3 min
*Fiche §27.2.1, §27.4.2 · livre §27.2.1, §27.4.2*

a) Le générateur reçoit en entrée : (A) un vrai billet abîmé ; (B) des nombres tirés au hasard ; (C) le label vrai ou faux ; (D) la sortie du discriminateur.
b) D'où le générateur tire-t-il ce qu'il sait des vraies données ? (A) de nulle part : il ne l'apprend jamais ; (B) du gradient qui traverse le discriminateur, lui-même entraîné sur les vraies données ; (C) d'une copie des vraies images qu'on lui donne à la fin de chaque epoch.
c) Vrai ou faux : on peut entraîner comme un GAN un « générateur » qui est un procédé chimique (le vin du livre), dont on ne sait pas calculer de dérivée.
d) Le bruit d'entrée joue le rôle : (A) de variables latentes ; (B) d'un label ; (C) d'une régularisation, comme le dropout.
e) Vrai ou faux : comme dans un VAE (ch. 25), ces variables latentes s'obtiennent en encodant une donnée réelle.

### 27.Q4 — Le round d'apprentissage du livre en quatre temps 🧠 ⏱️ 3 min
*Fiche §27.2.3, §27.4.3 · livre §27.2.3, §27.4.3*

Trois sortes d'étapes : (A) on montre un **faux** à $D$, et $D$ apprend s'il s'est fait duper ; (B) on montre un **vrai** à $D$, et $D$ apprend s'il s'est trompé ; (C) on montre un **faux** à $D$, et $G$ apprend si son faux est repéré.
a) Écris les lettres des quatre étapes d'un round du livre, dans l'ordre (une lettre peut revenir).
b) Combien de fois le générateur a-t-il l'occasion d'apprendre dans un round ?
c) Pendant une étape où $G$ apprend, le discriminateur est : (A) entraîné lui aussi ; (B) gelé, mais le gradient le traverse ; (C) retiré du calcul.
d) Vrai ou faux : dans la boucle standard (Goodfellow et coll. 2014, tutoriel PyTorch), $D$ fait un pas sur un mini-batch de vrais **et** de faux, quel que soit son verdict, puis $G$ fait un pas.
e) Dans cette boucle en PyTorch, on écrit `fake.detach()` à l'étape de $D$ pour que : (A) la mémoire de $G$ soit libérée ; (B) la loss de $D$ ne calcule pas de gradient dans les poids de $G$ ; (C) $D$ soit gelé.

### 27.Q5 — Coopératifs ou antagonistes : somme nulle, minimax et équilibre de Nash 🧠 ⏱️ 3 min
*Fiche §27.3, §27.4.4 · livre §27.3, §27.4.4*

a) Vrai ou faux : dans la formulation de Goodfellow et coll. (2014), $G$ et $D$ jouent sur la même fonction $V(D, G)$ : $D$ cherche à la maximiser, $G$ à la minimiser.
b) Un jeu est à somme nulle quand : (A) la somme des gains des joueurs vaut zéro, quelle que soit l'issue ; (B) les ressources à se partager sont en quantité fixe ; (C) personne ne gagne jamais.
c) Vrai ou faux : une partie de poker entre amis, sans commission prélevée par une maison de jeu, est un jeu à somme nulle.
d) Un équilibre de Nash est une situation où : (A) aucun joueur ne gagne à changer **seul** de stratégie ; (B) les deux joueurs ont la même loss ; (C) la loss du discriminateur vaut zéro.
e) À l'équilibre idéal d'un GAN, quelle valeur rend le discriminateur, pour toute entrée ?

### 27.Q6 — Trois modèles, des couches partagées et un discriminateur gelé 🧠 ⏱️ 3 min
*Fiche §27.4, §27.4.3 · livre §27.4, §27.4.3, §27.5*

a) Combien de modèles le livre construit-il pour entraîner un GAN ?
b) Vrai ou faux : le modèle combiné du livre contient des **copies** du générateur et du discriminateur, qu'il faut resynchroniser après chaque étape.
c) En PyTorch, le modèle combiné est remplacé par : (A) un seul optimiseur sur tous les poids ; (B) deux optimiseurs, un pour $G$ et un pour $D$, et le calcul `D(G(z))` écrit à la volée ; (C) une copie de $D$ dont on recharge les poids.
d) Vrai ou faux : à l'étape de $G$, avec deux optimiseurs, `loss_g.backward()` dépose aussi des gradients dans les poids de $D$.
e) Ce qui empêche cette étape de modifier $D$ : (A) `loss_g.backward()` ne touche pas $D$ ; (B) seul `opt_g.step()` est appelé, et les gradients laissés dans $D$ sont effacés par `opt_d.zero_grad()` avant son pas suivant ; (C) on entoure l'étape de `torch.no_grad()`.

### 27.Q7 — Pourquoi juger des mini-batchs plutôt qu'un point isolé 🧠 ⏱️ 3 min
*Fiche §27.5 · livre §27.5*

a) Le jeu d'entraînement du livre compte 10 000 points, et les mini-batchs 32 points. Combien de pas le discriminateur fait-il par epoch, à raison d'un pas par mini-batch (le dernier, incomplet, compte aussi) ?
b) Vrai ou faux : le discriminateur du livre (deux couches de 16 neurones, une sortie) reçoit le mini-batch entier et juge l'ensemble des points.
c) Quelle technique donne vraiment au discriminateur une information sur tout le mini-batch ? (A) une couche d'écart-type par mini-batch (Karras et coll.) ou la discrimination par mini-batch (Salimans et coll.) ; (B) le dropout ; (C) un learning rate plus petit.
d) Pour une gaussienne 2D de centre $(5, 5)$ et d'écart-type 1 dans chaque direction, la part des points à moins de 1 du centre est d'environ : (A) 68 % ; (B) 39 % ; (C) 95 %.
e) Le générateur du livre : 4 nombres en entrée, une couche dense de 16 neurones, puis une couche dense de 2 neurones. Combien a-t-il de paramètres, biais compris ?

### 27.Q8 — Les règles de Radford : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §27.6.1 · livre §27.6.1*

D'après l'article de Radford et coll. (2016), vrai ou faux ?
a) Dans le discriminateur, on remplace le pooling par des convolutions à pas (*stride*) de 2.
b) L'optimiseur conseillé est Adam, avec un learning rate de $10^{-3}$ pour le générateur et de $10^{-4}$ pour le discriminateur.
c) Le **générateur** utilise des Leaky ReLU dans toutes ses couches.
d) Pas de batchnorm sur la couche de sortie du générateur, ni sur la couche d'entrée du discriminateur.
e) Le générateur finit par une tanh, donc on ramène les pixels des images d'entraînement dans $[-1, 1]$.

### 27.Q9 — Grandes images et croissance progressive 🧠 ⏱️ 3 min
*Fiche §27.7.1 · livre §27.7.1*

a) Combien de nombres (features) contient une photo **couleur** de 1 000 × 1 000 pixels ?
b) Une difficulté des GAN sur de grandes images : (A) le discriminateur distingue trop facilement les faux, et les gradients donnés au générateur deviennent peu utiles ; (B) les images ne tiennent pas dans un fichier PNG ; (C) le générateur n'a pas assez de bruit en entrée.
c) La croissance progressive commence par : (A) la pleine résolution, que l'on réduit ensuite ; (B) des images de 4 × 4, dont on double la taille par étapes.
d) Vrai ou faux : dans l'article de Karras et coll. (2018), les couches de la nouvelle résolution arrivent d'un coup, sans transition.
e) En doublant le côté à chaque étape, de 4 × 4 jusqu'à 1 024 × 1 024, combien de résolutions traverse-t-on, les deux extrêmes compris ?

### 27.Q10 — Effondrement de mode total ou partiel 🧠 ⏱️ 3 min
*Fiche §27.7.2 · livre §27.7.2*

a) Dans « effondrement de mode » (*mode collapse*), le mode désigne : (A) une manière de travailler ; (B) un pic de la distribution des données ; (C) le mode `train()` ou `eval()` d'un réseau.
b) Un DCGAN entraîné sur MNIST ne produit presque que des 1, avec quelques 7 : l'effondrement est (A) total ou (B) partiel ?
c) Vrai ou faux : un générateur qui ne produit plus qu'un seul chiffre, bien dessiné, peut avoir pendant un moment une loss basse.
d) Ce qui se passe en général ensuite : (A) le générateur s'arrête pour de bon ; (B) le discriminateur apprend à rejeter ce mode, le générateur saute vers un autre, et cela peut tourner en rond.
e) Parmi ces moyens, lesquels combattent l'effondrement de mode ? (A) une couche d'écart-type par mini-batch dans le discriminateur ; (B) la discrimination par mini-batch ; (C) entraîner plus longtemps, sans rien changer d'autre ; (D) une loss de Wasserstein avec pénalité de gradient (WGAN-GP). Écris leurs lettres.

<a id="rappels"></a>

## 🔁 Rappels

### 27.R1 — Deux agents qui jouent l'un contre l'autre (self-play) 🔁 ⏱️ 5 min
*ch. 26 · fiche §27.3*

a) En auto-apprentissage (*self-play*, ch. 26), l'agent joue contre : (A) une copie de lui-même ; (B) un humain ; (C) un adversaire qui joue au hasard.
b) Vrai ou faux : dans un GAN, comme au morpion en auto-apprentissage, les deux joueurs partagent les mêmes poids.
c) Au morpion, deux joueurs parfaits aboutissent à : (A) une victoire des croix ; (B) un match nul ; (C) une victoire des ronds.
d) Vrai ou faux : en auto-apprentissage comme dans un GAN, la cible bouge, car l'adversaire change à mesure que l'on apprend.
e) Ce que le générateur reçoit de plus qu'un agent de RL : (A) un gradient, qui dit dans quelle direction changer sa sortie ; (B) une récompense ; (C) rien de plus.

### 27.R2 — Precision et recall pour juger un générateur (fidélité contre diversité) 🔁 ⏱️ 5 min
*ch. 3 · fiche « Au-delà du livre »*

Un discriminateur juge 100 vrais et 100 faux. Il répond « vrai » pour 80 des vrais et pour 30 des faux. La classe positive est « vrai ».
a) La precision du discriminateur, à 3 décimales.
b) Son recall, à 2 décimales.
c) Pour un **générateur**, la precision de Kynkäänniemi et coll. (2019) mesure : (A) la fidélité, la part des échantillons générés qui ressemblent à des données réelles ; (B) la diversité, la part des données réelles que le générateur sait approcher.
d) Un générateur effondré sur un seul chiffre parfait a : (A) une precision haute et un recall bas ; (B) une precision basse et un recall haut ; (C) les deux hauts.
e) Vrai ou faux : le FID mêle fidélité et diversité en un seul nombre.

### 27.R3 — Batchnorm, où la placer, train() ou eval() 🔁 ⏱️ 5 min
*ch. 20 · fiche §27.6.1*

a) Pendant l'entraînement (`train()`), une batchnorm normalise avec : (A) la moyenne et l'écart-type du mini-batch ; (B) des moyennes glissantes calculées pendant l'entraînement.
b) En mode `eval()`, elle normalise avec : (A) la moyenne et l'écart-type du mini-batch ; (B) les moyennes glissantes.
c) Selon le livre (ch. 20, rappelé au §27.6), une batchnorm se place : (A) entre la sortie d'une couche et son activation ; (B) après l'activation ; (C) avant la couche.
d) Vrai ou faux : pour générer des images après l'entraînement, on passe le générateur en `eval()` ; un même $\mathbf{z}$ donne alors la même image, quels que soient les autres $\mathbf{z}$ du batch.
e) Combien de paramètres **appris** compte `nn.BatchNorm2d(64)` ?

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 27.1 — Dimensions d'un DCGAN pour MNIST : 6 272 neurones, tailles 7 → 14 → 28 et retour ✏️ ★ ⏱️ 15 min
**Objectif :** suivre la forme des tenseurs dans un générateur et un discriminateur convolutifs, et compter leurs paramètres.
**Prérequis :** fiche §27.6 · ch. 16 (couche dense) · ch. 21 (taille de sortie d'une convolution) · ch. 25 (convolution transposée) · **Parcours :** R, M

Formules (fiche §27.6) : $H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$ pour une convolution, $H_{\text{out}} = (H_{\text{in}} - 1)s - 2p + k + p_{\text{out}}$ pour une convolution transposée.

a) Le générateur du livre remet en forme la sortie d'une couche dense en un tenseur de 128 canaux de 7 × 7. Combien de neurones doit avoir cette couche ?
b) Cette couche dense reçoit 1 024 valeurs. Combien a-t-elle de paramètres, biais compris ?
c) Une convolution de noyau 4, de pas 2, de padding 1, sur une image de 28 × 28 : la hauteur de sortie.
d) La même convolution appliquée à la sortie de c) : la hauteur de sortie.
e) Une convolution transposée de noyau 4, de pas 2, de padding 1 (sans `output_padding`), sur un tenseur de 7 × 7 : la hauteur de sortie.
f) Le nombre de paramètres de `nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)`.
g) Le générateur du livre fait plutôt un sur-échantillonnage 2 × 2 (qui n'a pas de paramètres) suivi d'une convolution de 128 vers 64 canaux, de noyau 5 × 5, avec biais. Combien de paramètres a cette convolution ?

Dans ta copie : pourquoi 7 × 7 au départ, et pas 6 × 6 ou 8 × 8 ? Laquelle des couches de f) et de g) coûte le plus ? Laquelle des deux façons d'agrandir est la plus exposée au motif en damier (fiche §27.6.1), et pourquoi son noyau de 4 pour un pas de 2 limite-t-il le défaut ?

### Ex 27.2 — Valeur minimax d'une petite matrice de gains ✏️ ★ ⏱️ 10 min
**Objectif :** trouver le coup prudent de chaque joueur dans un jeu à somme nulle, et voir que maximin et minimax peuvent différer.
**Prérequis :** fiche §27.4.4 (encadré 🧮) · 0B (maximum, minimum) · **Parcours :** M

Deux joueurs, Ligne et Colonne, jouent une fois, en même temps. Le tableau donne le gain de Ligne ; Colonne perd ce que Ligne gagne (somme nulle). Ligne choisit A, B ou C ; Colonne choisit 1, 2 ou 3.

| | 1 | 2 | 3 |
|---|---|---|---|
| **A** | 3 | −1 | 2 |
| **B** | 1 | 0 | 4 |
| **C** | −2 | 5 | −3 |

a) Ligne joue prudemment : pour chaque coup, elle regarde le pire gain possible et choisit le coup dont le pire est le meilleur (maximin). Quelle lettre joue-t-elle ?
b) La valeur maximin, le gain garanti à Ligne.
c) Colonne raisonne de même : pour chaque coup, le pire résultat pour elle (le plus grand gain de Ligne) ; elle choisit le coup dont ce pire résultat laisse le moins à Ligne (minimax). La valeur minimax.
d) Vrai ou faux : ce jeu a un point selle, un couple de coups où maximin et minimax coïncident.
e) Vrai ou faux : dans un GAN entraîné par descente de gradient alternée, le résultat est toujours le même que l'on vise $\min_G \max_D$ ou $\max_D \min_G$.

Dans ta copie : l'exemple du livre (chaque coup donne à l'adversaire une meilleure réponse de 5, 10 ou 20 territoires) : quel coup joue-t-on, et pourquoi est-ce un raisonnement minimax ?

### Ex 27.3 — Losses BCE du discriminateur et du générateur à partir de sorties données ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer à la main les losses d'un GAN pour un petit mini-batch, avec la loss minimax et la loss non saturante.
**Prérequis :** fiche §27.4.3 (encadré 🧮) · ch. 6 (cross-entropy binaire) · **Parcours :** R, M

Un mini-batch de deux vrais et de deux faux. Le discriminateur rend $D(\mathbf{x}_1) = 0{,}8$ et $D(\mathbf{x}_2) = 0{,}6$ sur les vrais, $D(G(\mathbf{z}_1)) = 0{,}3$ et $D(G(\mathbf{z}_2)) = 0{,}1$ sur les faux. Chaque terme est une **moyenne** sur son demi-batch, comme le fait `nn.BCELoss` ; la loss du discriminateur est la **somme** du terme des vrais et du terme des faux (comme dans le tutoriel DCGAN de PyTorch). Toutes les réponses à 3 décimales.

a) Le terme des vrais, $-\frac{1}{2}\left[\ln D(\mathbf{x}_1) + \ln D(\mathbf{x}_2)\right]$.
b) Le terme des faux, $-\frac{1}{2}\left[\ln(1 - D(G(\mathbf{z}_1))) + \ln(1 - D(G(\mathbf{z}_2)))\right]$.
c) La loss du discriminateur $L_D$.
d) La loss **non saturante** du générateur, $-\frac{1}{2}\left[\ln D(G(\mathbf{z}_1)) + \ln D(G(\mathbf{z}_2))\right]$.
e) La quantité que le générateur minimise dans la version **minimax**, $\frac{1}{2}\left[\ln(1 - D(G(\mathbf{z}_1))) + \ln(1 - D(G(\mathbf{z}_2)))\right]$.
f) La part des quatre échantillons bien classés par le discriminateur, au seuil 0,5.

Dans ta copie : lequel des deux faux pèse le plus dans d) ? Et dans e) ? Lequel des deux devrait pourtant faire le plus progresser le générateur ?

### Ex 27.4 — À l'équilibre, la loss vaut ln 2 et non 0,5 ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer la valeur des losses quand le discriminateur ne sait plus distinguer, et corriger la valeur « idéale » du livre.
**Prérequis :** Ex 27.3 · fiche §27.5 · **Parcours :** R, M

Le générateur imite parfaitement les données : le discriminateur répond $D = \frac{1}{2}$ pour toute entrée. Mêmes conventions qu'à l'Ex 27.3.

a) La loss du discriminateur $L_D$ (terme des vrais + terme des faux), à 3 décimales.
b) La loss non saturante du générateur, à 3 décimales.
c) Le livre dit que les deux losses devraient idéalement valoir 0,5. Quelle sortie $D(\mathbf{x})$ sur un vrai donnerait un terme $-\ln D(\mathbf{x}) = 0{,}5$ ? À 3 décimales.
d) La valeur du jeu $V(D, G) = \mathbb{E}[\ln D(\mathbf{x})] + \mathbb{E}[\ln(1 - D(G(\mathbf{z})))]$ à cet équilibre, à 3 décimales.
e) Vrai ou faux : les courbes de la fig. 27.21 du livre, qui se stabilisent vers 0,69, sont en fait à la valeur idéale.

Dans ta copie : un discriminateur qui répondrait toujours $0{,}607$ aurait-il une meilleure loss à l'équilibre que celui qui répond $\frac{1}{2}$ ? Calcule $L_D$ dans ce cas.

### Ex 27.5 — Le discriminateur optimal D* = p_data / (p_data + p_g) ∂ ★★★ ⏱️ 30 min
**Objectif :** dériver le discriminateur optimal pour un générateur fixé, puis la valeur du jeu qui en découle.
**Prérequis :** fiche §27.4.4 (encadré 🧮) · 0B (dérivée nulle d'une fonction, logarithme) · ch. 5 (maximiser une fonction) · ch. 6 (divergence KL) · **Parcours :** M

On fixe le générateur, donc la densité $p_g$. Réponds dans ta copie.

1. Écris $V(D, G)$ comme **une seule** intégrale sur $\mathbf{x}$, en remplaçant $\mathbb{E}_{\mathbf{z}}[\ln(1 - D(G(\mathbf{z})))]$ par une espérance sous $p_g$.
2. Pour un $\mathbf{x}$ fixé, on pose $a = p_{\text{data}}(\mathbf{x}) > 0$, $b = p_g(\mathbf{x}) > 0$ et $y = D(\mathbf{x}) \in\, ]0, 1[$. Montre que $f(y) = a \ln y + b \ln(1 - y)$ atteint son maximum en $y^* = \frac{a}{a + b}$ (dérivée nulle, puis signe de la dérivée seconde).
3. Déduis-en $D^*(\mathbf{x})$. Que vaut-il là où $p_g = p_{\text{data}}$ ? Vers quoi tend-il quand $p_g(\mathbf{x})$ tend vers 0 (le générateur ne produit presque rien autour de $\mathbf{x}$) ?
4. Remplace $D$ par $D^*$ dans $V$, et montre que $V(D^*, G) = -\ln 4 + \mathrm{KL}\!\left(p_{\text{data}} \,\middle\|\, m\right) + \mathrm{KL}\!\left(p_g \,\middle\|\, m\right)$, où $m = \frac{p_{\text{data}} + p_g}{2}$ (fais apparaître $\frac{1}{2}$ au dénominateur de chaque logarithme).
5. La somme des deux KL vaut $2\,\mathrm{JS}(p_{\text{data}} \,\|\, p_g)$, la **divergence de Jensen-Shannon** promise au ch. 6. Sachant qu'une KL est positive ou nulle, et nulle seulement entre deux distributions égales, quel est le minimum de $V(D^*, G)$ sur les générateurs, et pour quel $p_g$ ?
6. Relie ce minimum aux valeurs de l'Ex 27.4.

### Ex 27.6 — Pourquoi log(1 − D(G(z))) sature et pas −log D(G(z)) ∂ ★★★ ⏱️ 30 min
**Objectif :** montrer par le calcul du gradient pourquoi la loss minimax du générateur s'éteint au début de l'entraînement, et pourquoi la loss non saturante règle le problème.
**Prérequis :** Ex 27.3 · fiche §27.4.3 · ch. 17 (sigmoïde et sa dérivée) · **Parcours :** M

Le discriminateur rend $D = \sigma(a)$, où $a$ est son logit (sa sortie avant la sigmoïde) et $\sigma(a) = \frac{1}{1 + e^{-a}}$. Réponds dans ta copie.

1. Rappelle $\sigma'(a)$ en fonction de $\sigma(a)$.
2. La loss minimax du générateur, pour un faux de logit $a$, est $\ell_{\text{mm}}(a) = \ln(1 - \sigma(a))$. Calcule $\frac{d\ell_{\text{mm}}}{da}$ et simplifie-la en fonction de $D = \sigma(a)$.
3. La loss non saturante est $\ell_{\text{ns}}(a) = -\ln \sigma(a)$. Calcule $\frac{d\ell_{\text{ns}}}{da}$ et simplifie-la de même.
4. Au début de l'entraînement, le discriminateur repère facilement les faux : $D(G(\mathbf{z})) = 0{,}01$. Calcule les deux dérivées. Combien de fois la seconde est-elle plus grande que la première (en valeur absolue) ?
5. Pour les deux losses, dans quel sens le générateur pousse-t-il $a$ ? Visent-elles le même but (que $D$ prenne les faux pour des vrais) ? Pourquoi dit-on que la seconde « garde le même objectif » mais « change la dynamique » ?
6. Exprime `F.binary_cross_entropy_with_logits(a, torch.ones_like(a))` en fonction de $a$ : laquelle des deux losses calcule-t-elle, et pourquoi PyTorch préfère-t-il les logits à `nn.BCELoss` sur une sortie déjà passée dans la sigmoïde ?

### Ex 27.7 — Croissance progressive : où passe le temps de calcul ? 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer en ordres de grandeur le coût de chaque étape de la croissance progressive, et ce qu'elle fait gagner.
**Prérequis :** fiche §27.7.1 · **Parcours :** R, M

On suppose que le coût d'un pas est proportionnel au nombre de pixels des images, et que chaque étape (4 × 4, 8 × 8, …, 1 024 × 1 024) voit le **même** nombre d'images, $N$. Réponds dans ta copie, avec des ordres de grandeur.

1. Le nombre de pixels (d'un canal) à chaque résolution, et leur total sur les neuf étapes.
2. Quelle part du calcul total passe dans la dernière étape ? Dans les quatre premières réunies ?
3. On entraîne plutôt à 1 024 × 1 024 dès le départ, sur les $9N$ images. Combien de fois plus de calcul cela demande-t-il ? Compare au gain de 2 à 6 fois rapporté par Karras et coll.
4. Pourquoi l'hypothèse « coût proportionnel aux pixels » est-elle grossière (pense aux couches à basse résolution, aux canaux) ?
5. La mémoire d'un mini-batch de 16 images couleur de 1 024 × 1 024 en `float32`, pour l'entrée seule (en Mo, $10^6$ octets). Pourquoi les activations en demandent-elles beaucoup plus ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 27.8 — Un GAN en cinq lignes, sans faux billets 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer un GAN à un débutant avec une image à toi.
**Prérequis :** fiche §27.2, §27.3 · **Parcours :** R

Explique en cinq lignes au plus, à quelqu'un qui ne connaît pas le deep learning, ce que font les deux réseaux d'un GAN et pourquoi leur compétition finit par produire des images crédibles. Interdit : la fausse monnaie. Ton image doit faire apparaître (1) que le générateur ne voit jamais de vraies données, (2) qu'il apprend des réactions de l'autre, (3) ce qui se passe quand l'un des deux devient trop fort.

### Ex 27.9 — Visages qui n'existent pas, deepfakes et obligation de transparence ⚖️ ★★ ⏱️ 20 min
**Objectif :** analyser les usages des images générées, leurs risques, et les règles qui les encadrent.
**Prérequis :** fiche « Au-delà du livre » (encadrés 🕰️) · **Parcours :** R

Réponds dans ta copie.

1. Un site affiche à chaque visite le portrait, généré, d'une personne qui n'existe pas. Donne deux usages légitimes de tels portraits et deux usages malveillants.
2. Une vidéo truquée (*deepfake*) montre une élue tenir des propos qu'elle n'a jamais tenus, la veille d'un scrutin. Qui est lésé ? Pourquoi un démenti arrive-t-il souvent trop tard ?
3. Le règlement européen sur l'IA (AI Act, art. 50) impose de signaler les contenus générés ou manipulés. Qui doit le faire : le fournisseur du modèle, la personne qui publie le contenu, les deux ? Que prévoit-il pour une œuvre manifestement satirique ou artistique ?
4. Filigranes invisibles, métadonnées de provenance signées (C2PA), détecteurs entraînés : pour chaque moyen, une force et une faiblesse.
5. Ton équipe entraîne un GAN sur des photos de visages récupérées sur le web. Quelles questions dois-tu poser avant (consentement, données personnelles, biais du dataset) ?

### Ex 27.10 — Lire Goodfellow et coll. (2014), « Generative Adversarial Nets » 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article fondateur des GAN et relier ses résultats au chapitre.
**Prérequis :** Ex 27.5 · fiche §27.3, §27.4 · **Parcours :** M

Lis I. Goodfellow et coll., « Generative Adversarial Nets », NeurIPS 2014 ([arXiv:1406.2661](https://arxiv.org/abs/1406.2661)) : le résumé, les sections 3 et 4 (avec l'algorithme 1, la proposition 1 et le théorème 1), la section 6 et la figure 1. Réponds dans ta copie.

1. Écris la fonction de valeur $V(D, G)$ de l'article avec les notations de la fiche. Quel joueur la minimise, lequel la maximise ?
2. L'algorithme 1 : combien de pas du discriminateur par pas du générateur ($k$), et quelle valeur les auteurs prennent-ils ? Comment le comparer au round en quatre temps du livre ?
3. Au début de la section 3, les auteurs proposent de faire maximiser $\log D(G(\mathbf{z}))$ au générateur. Pourquoi ? Relie la réponse à l'Ex 27.6.
4. La proposition 1 et le théorème 1 : que démontrent-ils ? Quelle est la valeur $-\log 4$, et à quoi correspond-elle dans l'Ex 27.4 ?
5. La figure 1 : que représentent les courbes noire, verte et bleue, et que montre la dernière image ?
6. Quel est le « scénario Helvetica » de l'article ? Quel nom lui donne-t-on aujourd'hui ?
7. Les auteurs mesurent leurs générateurs par une estimation de densité à fenêtres de Parzen. Quelle limite en donnent-ils ? Quelle mesure l'a remplacée depuis (fiche) ?

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### 27.E1 — Expliquer l'entraînement d'un GAN et sa loss 💼 ★★ ⏱️ 10 min
« Comment entraîne-t-on un GAN ? Quelle loss pour chacun des deux réseaux ? »

### 27.E2 — Qu'est-ce que l'effondrement de mode et comment le combattre ? 💼 ★★ ⏱️ 10 min
« Qu'est-ce que le *mode collapse* ? Comment le détectez-vous, et que faites-vous contre ? »

### 27.E3 — GAN, VAE ou diffusion : lequel choisir et pourquoi ? 💼 ★★ ⏱️ 10 min
« Pour générer des images, choisiriez-vous un GAN, un VAE ou un modèle de diffusion ? Pourquoi ? »

### 27.E4 — Comment évaluer un modèle génératif (FID et ses limites) ? 💼 ★★ ⏱️ 10 min
« Comment mesurez-vous la qualité d'un modèle génératif ? Qu'est-ce que le FID, et quelles sont ses limites ? »

### 27.E5 — Pourquoi l'entraînement des GAN est-il instable ? 💼 ★★ ⏱️ 10 min
« Pourquoi les GAN sont-ils réputés difficiles à entraîner ? Que faites-vous pour les stabiliser ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 27.11 | Distance de Wasserstein 1D entre deux échantillons | 🔨 | ★ | 15 |
| 27.12 | Un GAN minuscule pour une gaussienne 1D | 📦 | ★★ | 30 |
| 27.13 | Prédire la moyenne et l'écart-type générés au fil des epochs | 🔮 | ★★ | 15 |
| 27.14 | Lire les courbes d'un GAN : losses, D(x) et D(G(z)) | 📈 | ★★ | 15 |
| 27.15 | Reproduire l'apprentissage du nuage centré en (5, 5) (fig. 27.20-27.21) | 🎨 | ★★ | 30 |
| 27.16 | Le GAN qui s'entraînait à l'envers | 🐛 | ★★ | 25 |
| 27.17 | Prédire l'effet d'un discriminateur trop fort | 🔮 | ★★ | 15 |
| 27.18 | Loss saturante ou non saturante : la dynamique du début | 🔬 | ★★ | 30 |
| 27.19 | Le round en quatre temps du livre contre la boucle standard | 🔬 | ★★★ | 40 |
| 27.20 | Un DCGAN sur MNIST selon les règles de Radford | 📦 | ★★★ | 45 |
| 27.21 | Règles empiriques à l'épreuve : une architecture de 2016 contre celle de Radford | 🔬 | ★★★ | 45 |
| 27.22 | Distance de Fréchet entre deux gaussiennes (la formule du FID) | 🔨 | ★★★ | 35 |
| 27.23 | Évaluer un générateur : distance de Fréchet sur les features d'un classifieur | 📦 | ★★★ | 40 |
| 27.24 | Provoquer puis soigner un effondrement de mode (anneau de 8 gaussiennes) | 🔬 | ★★★ | 45 |
| 27.25 | Un GAN conditionnel : choisir le chiffre généré | 📦 | ★★★ | 45 |
| 27.26 | Se promener dans l'espace latent du générateur | 🔬 | ★★★ | 30 |
| 27.27 | Checkpoint d'un GAN : deux réseaux, deux optimiseurs, états aléatoires, reprise à l'identique | 🛠️ | ★★★ | 30 |
| 27.28 | Défi DCGAN Fashion-MNIST : passer sous un seuil de distance de Fréchet | 🏆 | ★★★★ | 120 |

Aucun exercice ne demande de GPU : en mode rapide, les GAN jouets s'entraînent en quelques secondes, et les DCGAN sur 10 000 images en moins d'une minute par réseau. Les entraînements complets (60 000 images, plus d'epochs) sont signalés 🚀 dans le notebook.
