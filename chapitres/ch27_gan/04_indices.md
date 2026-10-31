# 27 · Réseaux antagonistes génératifs (GAN) — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 27.Q1 — À quoi bon générer des données ?

<details><summary>Indice 1</summary>

Relis le §27.1 de la fiche et son encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

Un générateur fabrique de **nouveaux** échantillons qui ressemblent aux données, sans les copier. Pour chaque usage proposé, demande-toi : crée-t-on quelque chose de nouveau, ou retrouve-t-on, ou transforme-t-on sans rien inventer ?

</details>
<details><summary>Indice 3</summary>

a) Le livre donne deux usages : des données d'entraînement en plus (A), et des idées à proposer à un humain (B) ; compresser (C) et retrouver une photo (D) n'inventent rien. Pour b), un générateur qui recopierait ses données aurait-il appris quelque chose ? Pour c), relis le §27.1 ; pour d), l'encadré ⚠️ du §27.1 ; pour e), le titre de l'article de 2014.

</details>

### 27.Q2 — Le faussaire, l'inspectrice et les quatre verdicts

<details><summary>Indice 1</summary>

Relis le §27.2 de la fiche et la figure des quatre verdicts.

</details>
<details><summary>Indice 2</summary>

Deux questions pour chaque billet : est-il vraiment vrai ? Dawn a-t-elle répondu « vrai » (positif) ? « Vrai/faux » dans le nom dit si la réponse est juste ; « positif/négatif », ce qu'elle a répondu.

</details>
<details><summary>Indice 3</summary>

a) Le billet est vrai, Dawn répond « faux » : sa réponse est négative et fausse, c'est un faux négatif (B). Pour b), applique les deux mêmes questions à un faux billet qui passe. Pour c) et d), relis qui apprend dans chaque case selon le livre. Pour e), écris la cross-entropy d'un vrai pour lequel $D(\mathbf{x}) = 0{,}9$.

</details>

### 27.Q3 — Pourquoi le générateur ne voit-il jamais de vrais billets ?

<details><summary>Indice 1</summary>

Relis les §27.2.1 et §27.4.2 de la fiche et l'encadré ⚠️ « ce qu'il faut pour être un générateur ».

</details>
<details><summary>Indice 2</summary>

Suis le chemin de l'information : les vraies données entrent dans $D$, $D$ apprend ; le générateur reçoit un gradient qui traverse $D$. Pour qu'un gradient le traverse, que faut-il savoir calculer ?

</details>
<details><summary>Indice 3</summary>

a) Le générateur part de nombres tirés au hasard (B) ; il ne reçoit ni billet, ni label, ni la sortie de $D$ comme entrée. Pour b), demande-toi par où passe ce que $D$ a appris des vrais. Pour c), un procédé chimique a-t-il une dérivée qu'on sait calculer ? Pour d) et e), compare avec le VAE du ch. 25 : qui fabrique le code latent ?

</details>

### 27.Q4 — Le round d'apprentissage du livre en quatre temps

<details><summary>Indice 1</summary>

Relis les §27.2.3 et §27.4.3 de la fiche, et l'encadré ⚠️ sur le round.

</details>
<details><summary>Indice 2</summary>

Le livre alterne les deux réseaux : discriminateur, générateur, discriminateur, générateur. Une étape qui commence par un **vrai** ne peut concerner que $D$ ; une étape où un faux est **repéré** concerne $G$.

</details>
<details><summary>Indice 3</summary>

a) La première étape montre un vrai à $D$ : c'est B. La deuxième fait apprendre $G$ ; la troisième fait apprendre $D$ sur un faux ; la quatrième répète la deuxième. Pour b), compte les étapes « G apprend ». Pour c), si $D$ apprenait pendant l'étape de $G$, vers quoi le pousserait le label « vrai » ? Pour d), relis l'algorithme 1 dans l'encadré ⚠️ ; pour e), l'encadré 🧮 sur `detach`.

</details>

### 27.Q5 — Coopératifs ou antagonistes : somme nulle, minimax et équilibre de Nash

<details><summary>Indice 1</summary>

Relis le §27.4.4 de la fiche, son encadré 🧮 et l'encadré ⚠️ sur le poker.

</details>
<details><summary>Indice 2</summary>

Somme nulle : additionne les gains des joueurs pour chaque issue. Nash : imagine que chaque joueur, seul, essaie de changer sa stratégie ; y gagne-t-il ?

</details>
<details><summary>Indice 3</summary>

a) La formule $\min_G \max_D V(D, G)$ : $D$ cherche le maximum de $V$, $G$ le minimum, sur la même fonction (vrai). Pour b), applique la définition de l'encadré 🧮 ; pour c), suis un euro misé au poker ; pour d), relis la définition de Nash ; pour e), relis le §27.4.4 : là où $p_g = p_{\text{data}}$, que vaut $D^*$ ?

</details>

### 27.Q6 — Trois modèles, des couches partagées et un discriminateur gelé

<details><summary>Indice 1</summary>

Relis le §27.4 de la fiche et l'encadré 🧮 « `.detach()`, `requires_grad_(False)` et les deux optimiseurs ».

</details>
<details><summary>Indice 2</summary>

En PyTorch, `backward()` remplit le `.grad` de **tous** les poids qui ont servi au calcul ; c'est `optimizer.step()` qui modifie les poids, et seulement ceux qu'on lui a confiés.

</details>
<details><summary>Indice 3</summary>

a) Le livre construit le générateur, le discriminateur, et un troisième modèle qui les enchaîne : 3. Pour b), relis le §27.4 : copies ou mêmes couches ? Pour c), qu'est-ce qui remplace le modèle combiné quand on écrit `D(G(z))` ? Pour d), le calcul de `loss_g` passe-t-il par les poids de $D$ ? Pour e), quel appel modifie les poids, et quel appel efface les `.grad` de $D$ ?

</details>

### 27.Q7 — Pourquoi juger des mini-batches plutôt qu'un point isolé

<details><summary>Indice 1</summary>

Relis le §27.5 de la fiche et ses deux premiers encadrés ⚠️.

</details>
<details><summary>Indice 2</summary>

Un pas par mini-batch, le dernier compris ; une division arrondie au-dessus. Pour la gaussienne 2D, la distance au centre ne suit pas la loi normale d'une seule variable : la fiche donne $P(r \le 1)$.

</details>
<details><summary>Indice 3</summary>

a) $10\,000 / 32 = 312{,}5$ : 312 mini-batches pleins et un incomplet. Pour b), compte les entrées du discriminateur du livre ; pour c), relis l'encadré ⚠️ sur le mini-batch ; pour d), calcule $1 - e^{-1/2}$ ; pour e), compte pour chaque couche (entrées + 1) × sorties, puis additionne.

</details>

### 27.Q8 — Les règles de Radford : vrai ou faux

<details><summary>Indice 1</summary>

Relis le §27.6.1 de la fiche et l'encadré ⚠️ « les règles de Radford ».

</details>
<details><summary>Indice 2</summary>

Compare chaque affirmation à la liste de la fiche (§27.6.1) et à son encadré ⚠️, pas au livre.

</details>
<details><summary>Indice 3</summary>

a) Vrai : des convolutions de stride 2 remplacent le pooling dans $D$. Pour b), cherche dans la fiche le learning rate de l'article et son $\beta_1$ ; pour c), dans quel réseau va la Leaky ReLU de pente 0,2 ? Pour d), relis la règle de la batchnorm ; pour e), quelle plage de valeurs rend une tanh ?

</details>

### 27.Q9 — Grandes images et croissance progressive

<details><summary>Indice 1</summary>

Relis le §27.7.1 de la fiche et son encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

Une photo couleur a trois canaux. Pour la croissance, écris la suite des côtés en doublant à chaque fois, de 4 à 1 024.

</details>
<details><summary>Indice 3</summary>

a) $1\,000 \times 1\,000$ pixels, et trois nombres par pixel. Pour b), relis le premier paragraphe du §27.7.1 ; pour c), par où commence-t-on ? Pour d), relis l'encadré ⚠️ : comment les nouvelles couches entrent-elles ? Pour e), compte les termes de la suite $4, 8, 16, \dots, 1\,024$.

</details>

### 27.Q10 — Effondrement de mode total ou partiel

<details><summary>Indice 1</summary>

Relis le §27.7.2 de la fiche et son encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

Un mode, en statistique, est un pic de la distribution. Total : une seule sortie ; partiel : quelques-unes. Pour la suite, rappelle-toi que le discriminateur, lui, continue d'apprendre.

</details>
<details><summary>Indice 3</summary>

a) Le mode est un pic de la distribution des données (B), pas une manière de travailler. Pour b), reste-t-il plusieurs sortes de chiffres ? Pour c), si un seul chiffre trompe $D$ un moment, que vaut $-\ln D(G(\mathbf{z}))$ ? Pour d), que fait $D$ d'un chiffre qu'il voit trop souvent ? Pour e), relis la liste des remèdes du §27.7.2 et de l'encadré 🕰️ du §27.7, puis juge chaque proposition.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 27.R1 — Ch. 26 : deux agents qui jouent l'un contre l'autre (auto-apprentissage)

<details><summary>Indice 1</summary>

Relis l'exercice d'auto-apprentissage du ch. 26 (26.37) et le §27.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

En auto-apprentissage, un seul agent joue les deux camps. Dans un GAN, deux réseaux différents ont des rôles opposés. Dans les deux cas, l'adversaire apprend aussi.

</details>
<details><summary>Indice 3</summary>

a) L'agent joue contre une copie de lui-même (A). Pour b), un GAN a-t-il un seul jeu de poids pour deux rôles ? Pour c), rappelle-toi le résultat de minimax au morpion (ch. 26) ; pour d), l'adversaire reste-t-il le même ? Pour e), compare ce que reçoit un agent de RL (une récompense, un nombre) et ce qui revient au générateur à travers $D$.

</details>

### 27.R2 — Ch. 3 : precision et recall pour juger un générateur (fidélité contre diversité)

<details><summary>Indice 1</summary>

Relis les définitions de la precision et du recall au ch. 3, puis le paragraphe « Precision et recall génératifs » de la fiche.

</details>
<details><summary>Indice 2</summary>

Avec « vrai » pour classe positive : TP = vrais dits vrais, FP = faux dits vrais, FN = vrais dits faux. Precision $= \frac{TP}{TP + FP}$, recall $= \frac{TP}{TP + FN}$.

</details>
<details><summary>Indice 3</summary>

a) $TP = 80$, $FP = 30$ : precision $= \frac{80}{110}$. Pour b), $FN = 100 - 80$. Pour c), relis la définition de la fiche : laquelle des deux mesure regarde les échantillons générés un par un ? Pour d), un chiffre parfait est-il fidèle ? couvre-t-il la variété ? Pour e), relis ce que compare le FID (fiche).

</details>

### 27.R3 — Ch. 20 : batchnorm, où la placer, train() ou eval()

<details><summary>Indice 1</summary>

Relis la batchnorm au ch. 20 et la règle de Radford au §27.6.1.

</details>
<details><summary>Indice 2</summary>

En entraînement, la batchnorm normalise avec les statistiques du mini-batch, et met à jour des moyennes glissantes ; en évaluation, elle utilise ces moyennes. Ses paramètres appris : un facteur $\gamma$ et un décalage $\beta$ par canal.

</details>
<details><summary>Indice 3</summary>

a) En `train()`, la moyenne et l'écart-type du mini-batch (A). Pour b), quelles statistiques ne dépendent pas du batch ? Pour c), relis le §27.6 : avant ou après l'activation ? Pour d), en `eval()`, l'image d'un $\mathbf{z}$ dépend-elle des autres ? Pour e), compte $\gamma$ et $\beta$ pour 64 canaux ; les moyennes glissantes sont-elles apprises par gradient ?

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 27.1 — Dimensions d'un DCGAN pour MNIST : 6 272 neurones, tailles 7 → 14 → 28 et retour ✏️

<details><summary>Indice 1</summary>

Relis le §27.6 de la fiche (les deux formules de taille) et le comptage des paramètres d'une couche dense (ch. 16) et d'une convolution (ch. 21).

</details>
<details><summary>Indice 2</summary>

Une couche dense de $n_{\text{in}}$ vers $n_{\text{out}}$ a $n_{\text{in}} n_{\text{out}}$ poids et $n_{\text{out}}$ biais. Une convolution (ou une convolution transposée) de $c_{\text{in}}$ vers $c_{\text{out}}$ canaux, de noyau $k \times k$, a $c_{\text{out}} c_{\text{in}} k^2$ poids et $c_{\text{out}}$ biais. Remplace dans les formules de taille sans arrondir avant la fin.

</details>
<details><summary>Indice 3</summary>

a) $7 \times 7 \times 128 = 6\,272$ neurones. Pour b), $1\,024 \times 6\,272$ poids plus un biais par neurone de sortie. Pour c), $\lfloor (28 + 2 - 4)/2 \rfloor + 1$ ; d) la même formule appliquée au résultat de c). Pour e), $(7 - 1) \times 2 - 2 + 4$. Pour f), $64 \times 128 \times 4^2$, sans biais ; pour g), $64 \times 128 \times 5^2 + 64$.

</details>

### Ex 27.2 — Valeur minimax d'une petite matrice de gains ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « somme nulle, minimax et équilibre de Nash » du §27.4.4.

</details>
<details><summary>Indice 2</summary>

Maximin : le minimum de chaque **ligne**, puis le plus grand de ces minimums. Minimax : le maximum de chaque **colonne**, puis le plus petit de ces maximums. Point selle : les deux valeurs sont égales.

</details>
<details><summary>Indice 3</summary>

a) Les minimums des lignes : A → −1, B → 0, C → −3 ; le meilleur des trois est celui de B. Pour b), quel gain cette lettre garantit-elle à Ligne ? Pour c), calcule les maximums des colonnes 1, 2 et 3, puis le plus petit. Pour d), compare b) et c). Pour e), relis la dernière phrase de l'encadré 🧮.

</details>

### Ex 27.3 — Losses BCE du discriminateur et du générateur à partir de sorties données ✏️

<details><summary>Indice 1</summary>

Relis le §27.4.3 de la fiche et l'encadré 🧮 sur la cross-entropy binaire.

</details>
<details><summary>Indice 2</summary>

Un vrai a le label 1 : son terme vaut $-\ln D(\mathbf{x})$. Un faux a le label 0 : son terme vaut $-\ln(1 - D(G(\mathbf{z})))$. Chaque terme est une moyenne sur ses deux échantillons ; $L_D$ additionne les deux moyennes.

</details>
<details><summary>Indice 3</summary>

a) $-\frac{1}{2}(\ln 0{,}8 + \ln 0{,}6) = \frac{1}{2}(0{,}2231 + 0{,}5108) \approx 0{,}367$. Pour b), $-\frac{1}{2}(\ln 0{,}7 + \ln 0{,}9)$ ; c), a) + b). Pour d), $-\frac{1}{2}(\ln 0{,}3 + \ln 0{,}1)$ ; pour e), $\frac{1}{2}(\ln 0{,}7 + \ln 0{,}9)$ ; pour f), compare chaque sortie à 0,5.

</details>

### Ex 27.4 — À l'équilibre, la loss vaut ln 2 et non 0,5 ✏️

<details><summary>Indice 1</summary>

Reprends les formules de l'Ex 27.3 avec $D = \frac{1}{2}$ partout, et relis l'encadré ⚠️ « la loss idéale » du §27.5.

</details>
<details><summary>Indice 2</summary>

$-\ln \frac{1}{2} = \ln 2$. Pour c), résous $-\ln p = 0{,}5$ ; pour d), $V$ est la somme de deux espérances de $\ln \frac{1}{2}$.

</details>
<details><summary>Indice 3</summary>

a) Deux termes de $-\ln \frac{1}{2}$ : $2\ln 2 \approx 1{,}386$. Pour b), un seul terme $-\ln \frac{1}{2}$. Pour c), $p = e^{-0{,}5}$. Pour d), $\ln\frac{1}{2} + \ln\frac{1}{2}$. Pour e), compare 0,69 à b).

</details>

### Ex 27.5 — Le discriminateur optimal D* = p_data / (p_data + p_g) ∂

<details><summary>Indice 1</summary>

Relis le §27.4.4 de la fiche (le discriminateur optimal, Jensen-Shannon) et la divergence KL au ch. 6.

</details>
<details><summary>Indice 2</summary>

1. $\mathbb{E}_{\mathbf{z}}[f(G(\mathbf{z}))] = \mathbb{E}_{\mathbf{x} \sim p_g}[f(\mathbf{x})]$ : les deux espérances deviennent des intégrales sur $\mathbf{x}$, qu'on regroupe. 2. Maximiser l'intégrale revient à maximiser l'intégrande en chaque $\mathbf{x}$. 4. Écris $\frac{p_{\text{data}}}{p_{\text{data}} + p_g} = \frac{1}{2} \cdot \frac{p_{\text{data}}}{m}$, et de même pour l'autre terme.

</details>
<details><summary>Indice 3</summary>

1. $V = \int \bigl[p_{\text{data}}(\mathbf{x}) \ln D(\mathbf{x}) + p_g(\mathbf{x}) \ln(1 - D(\mathbf{x}))\bigr] d\mathbf{x}$. 2. $f'(y) = \frac{a}{y} - \frac{b}{1 - y}$ ; annule-la et résous ; calcule $f''(y)$ et regarde son signe. 4. $\ln D^* = \ln \frac{p_{\text{data}}}{m} + \ln \frac{1}{2}$ et $\ln(1 - D^*) = \ln \frac{p_g}{m} + \ln \frac{1}{2}$ : remplace dans $V$, et reconnais les deux KL et les deux $\ln \frac{1}{2}$. 5. Que vaut une somme de deux KL au minimum, et quand ?

</details>

### Ex 27.6 — Pourquoi log(1 − D(G(z))) sature et pas −log D(G(z)) ∂

<details><summary>Indice 1</summary>

Relis la sigmoïde et sa dérivée au ch. 17, et le paragraphe « la loss non saturante » du §27.4.3.

</details>
<details><summary>Indice 2</summary>

$\sigma'(a) = \sigma(a)(1 - \sigma(a))$. Dérive $\ln(u(a))$ en $\frac{u'(a)}{u(a)}$, avec $u = 1 - \sigma$ puis $u = \sigma$. Pour 6, écris la cross-entropy binaire avec un logit (encadré 🧮 du §27.4.3).

</details>
<details><summary>Indice 3</summary>

2. $\frac{d}{da}\ln(1 - \sigma(a)) = \frac{-\sigma'(a)}{1 - \sigma(a)} = -\sigma(a)$. 3. Fais de même avec $-\ln \sigma(a)$ : la simplification donne une expression en $1 - D$. 4. Remplace $D = 0{,}01$ dans les deux formules, puis divise. 5. Les deux dérivées ont-elles le même signe ? Où s'annulent-elles ? 6. Avec le label 1, `binary_cross_entropy_with_logits` calcule $\ln(1 + e^{-a})$ : à laquelle des deux losses est-ce égal ?

</details>

### Ex 27.7 — Croissance progressive : où passe le temps de calcul ? 🧮

<details><summary>Indice 1</summary>

Relis le §27.7.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Les côtés sont $4 \times 2^k$ pour $k = 0, \dots, 8$ ; les nombres de pixels $16 \times 4^k$ forment une suite géométrique de raison 4 (encadré 🧮 du ch. 26). Le coût d'une étape est proportionnel à $N \times$ ses pixels.

</details>
<details><summary>Indice 3</summary>

1. $4^2 = 16$, $8^2 = 64$, …, $1\,024^2 = 1\,048\,576$ ; la somme vaut $16 \frac{4^9 - 1}{4 - 1}$. 2. La part de la dernière étape : $\frac{1\,048\,576}{\text{somme}}$ ; pour les quatre premières, additionne $16 + 64 + 256 + 1\,024$. 3. Entraîner à pleine résolution sur $9N$ images coûte $9N \times 1\,048\,576$ : divise par le coût progressif. 5. $16 \times 3 \times 1\,024^2 \times 4$ octets.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 27.8 — Un GAN en cinq lignes, sans faux billets 🗣️

<details><summary>Indice 1</summary>

Cherche une situation où quelqu'un apprend à imiter quelque chose sans jamais le voir, seulement à partir des réactions d'un juge.

</details>
<details><summary>Indice 2</summary>

Il faut deux personnages : un qui produit sans voir le modèle, un qui compare avec le modèle et réagit. Le juge doit apprendre lui aussi, sinon le producteur trouverait vite un truc qui marche à tous les coups.

</details>
<details><summary>Indice 3</summary>

Un plan possible : (1) qui produit, à partir de quoi ; (2) qui juge, en comparant à quoi ; (3) ce que le producteur apprend de chaque verdict ; (4) pourquoi le juge s'améliore aussi ; (5) ce qui se passe si l'un des deux devient trop fort. Par exemple, un apprenti parfumeur qui n'a jamais senti le parfum original, et un nez qui compare.

</details>

### Ex 27.9 — Visages qui n'existent pas, deepfakes et obligation de transparence ⚖️

<details><summary>Indice 1</summary>

Relis les deux derniers encadrés 🕰️ de la fiche, et pense aux personnes concernées : celle qu'on imite, celle qu'on trompe, celle qui publie.

</details>
<details><summary>Indice 2</summary>

Pour 3, l'article 50 du règlement distingue le **fournisseur** du système (qui doit permettre de détecter que le contenu est généré) et le **déployeur** qui publie un deepfake (qui doit le signaler). Pour 4, demande-toi pour chaque moyen : peut-on l'enlever ? peut-on le contrefaire ? qui peut le vérifier ?

</details>
<details><summary>Indice 3</summary>

1. Légitimes : des personnages de jeux ou de publicités sans droit à l'image à négocier, des données d'entraînement anonymes. Cherche deux usages malveillants du côté des faux profils. 2. Pense à la vitesse de diffusion d'une vidéo face à celle d'un démenti. 3. Relis l'encadré : deux obligations, deux acteurs, une exception. 4. Un filigrane survit-il à une capture d'écran ? Une signature C2PA peut-elle être retirée ? Un détecteur suit-il les nouveaux générateurs ? 5. Consentement, RGPD, représentativité des visages.

</details>

### Ex 27.10 — Lire Goodfellow et coll. (2014), « Generative Adversarial Nets » 📄

<details><summary>Indice 1</summary>

Lis d'abord le résumé et la figure 1, puis la section 4 ; l'Ex 27.5 t'a déjà fait la démonstration de la proposition 1.

</details>
<details><summary>Indice 2</summary>

L'article utilise $\log$ (népérien) et les mêmes notations que la fiche. L'algorithme 1 se lit comme une boucle : $k$ pas de $D$, un pas de $G$. Le « scénario Helvetica » est décrit dans la section 6, avec les avantages et les inconvénients.

</details>
<details><summary>Indice 3</summary>

1. L'équation (1) de l'article. 2. Cherche « k = 1 » dans le texte de l'algorithme. 3. Le paragraphe qui suit l'équation (1). 4. La proposition 1 donne $D^*$ ; le théorème 1 relie $C(G)$ à la divergence de Jensen-Shannon et donne la valeur au minimum. 5. La légende de la figure 1 nomme les trois courbes. 6. Cherche « Helvetica ». 7. La section 5 parle des fenêtres de Parzen et de leur variance.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 27.E1 — Expliquer l'entraînement d'un GAN et sa loss

<details><summary>Indice 1</summary>

Fiche §27.4.3 : les deux étapes d'une itération, et la loss non saturante.

</details>
<details><summary>Indice 2</summary>

Trois temps : les deux réseaux et leurs rôles ; l'itération (pas de $D$ sur vrais et faux, pas de $G$ à travers $D$) ; la loss (BCE pour $D$, $-\ln D(G(\mathbf{z}))$ pour $G$, et pourquoi pas la version minimax).

</details>
<details><summary>Indice 3</summary>

Commence par « Deux réseaux : $G$ transforme un bruit en échantillon, $D$ classe vrai ou faux » ; puis l'itération avec `detach` ; puis les valeurs à l'équilibre ($\ln 2$), et le fait que la loss ne mesure pas la qualité.

</details>

### 27.E2 — Qu'est-ce que l'effondrement de mode et comment le combattre ?

<details><summary>Indice 1</summary>

Fiche §27.7.2, et 🔬 27.24.

</details>
<details><summary>Indice 2</summary>

Définition (total, partiel), cause (le générateur optimise contre le $D$ du moment), détection (classes générées, modes couverts, recall), remèdes (statistique du mini-batch, WGAN-GP, learning rates).

</details>
<details><summary>Indice 3</summary>

Un exemple chiffré marque : « un DCGAN sur MNIST qui ne dessine presque que des 1 » ; puis un remède que tu as essayé toi-même (la couche d'écart-type par mini-batch de 27.24).

</details>

### 27.E3 — GAN, VAE ou diffusion : lequel choisir et pourquoi ?

<details><summary>Indice 1</summary>

Le tableau de la fiche « les GAN aujourd'hui ».

</details>
<details><summary>Indice 2</summary>

Compare sur quatre critères : qualité, diversité, vitesse de génération, stabilité de l'entraînement ; ajoute ce dont on a besoin (un encodeur ? du texte en entrée ?).

</details>
<details><summary>Indice 3</summary>

Une réponse en trois cas : diffusion par défaut (qualité, diversité, texte) ; GAN quand la vitesse compte (temps réel, super-résolution) ou comme loss d'appoint ; VAE quand il faut un espace latent avec encodeur (compression, détection d'anomalies, latent d'un modèle de diffusion).

</details>

### 27.E4 — Comment évaluer un modèle génératif (FID et ses limites) ?

<details><summary>Indice 1</summary>

Fiche « Au-delà du livre : mesurer un générateur », 🔨 27.22 et 📦 27.23.

</details>
<details><summary>Indice 2</summary>

Ce que compare le FID (moyennes et covariances de features), ce qu'il mélange (fidélité et diversité), ce dont il dépend (nombre d'échantillons, extracteur), et ce qu'on y ajoute (precision/recall, jugement humain).

</details>
<details><summary>Indice 3</summary>

Cite ton expérience de 27.23 : la distance qui change avec le nombre d'images générées. Termine par « on compare toujours au même nombre d'échantillons, avec le même extracteur ».

</details>

### 27.E5 — Pourquoi l'entraînement des GAN est-il instable ?

<details><summary>Indice 1</summary>

Fiche §27.7 et ses encadrés.

</details>
<details><summary>Indice 2</summary>

Un jeu à deux joueurs n'est pas une minimisation : la cible bouge ; la descente de gradient alternée peut tourner autour de l'équilibre. Ajoute la saturation, l'effondrement de mode, et les remèdes.

</details>
<details><summary>Indice 3</summary>

Structure : (1) pourquoi (cible mouvante, pas de garantie de convergence, saturation) ; (2) ce qu'on voit (oscillations, effondrement) ; (3) ce qu'on fait (loss non saturante, Adam avec $\beta_1 = 0{,}5$, normalisation spectrale, pénalité R1 ou WGAN-GP, TTUR, suivi du FID).

</details>

<a id="notebook"></a>

## Exercices du notebook

### Ex 27.11 — Distance de Wasserstein 1D entre deux échantillons 🔨

<details><summary>Indice 1</summary>

Relis le paragraphe sur la distance de Wasserstein dans la fiche (« Au-delà du livre ») et sa figure.

</details>
<details><summary>Indice 2</summary>

Valide d'abord les entrées (`np.asarray(..., dtype=float)`, 1-D, non vide, sans NaN). Tailles égales : trie les deux échantillons et moyenne les écarts absolus. Tailles différentes : fusionne et trie toutes les valeurs ; entre deux valeurs consécutives, les deux fonctions de répartition sont constantes et valent (nombre de valeurs $\le x$) / taille ; ajoute $|F_u - F_v| \times$ la largeur de l'intervalle.

</details>
<details><summary>Indice 3</summary>

```python
def wasserstein_1d(u, v):
    u, v = ...                     # validated float arrays, sorted with np.sort (never in place)
    if len(u) == len(v):
        ...                        # mean of the absolute differences, as a float
    merged = np.sort(np.concatenate([u, v]))
    cdf_u = np.searchsorted(u, merged[:-1], side="right") / len(u)
    # same for v, then the sum of |cdf_u - cdf_v| times the widths of the intervals
```

</details>

### Ex 27.12 — Un GAN minuscule pour une gaussienne 1D 📦

<details><summary>Indice 1</summary>

Relis le §27.4.3 de la fiche : la boucle PyTorch y est écrite pour un mini-batch.

</details>
<details><summary>Indice 2</summary>

`gan_step_12` : calcule `fake = G(z)` une fois ; étape de $D$ sur `D(real)` et `D(fake.detach())`, avec `torch.ones_like` et `torch.zeros_like` pour les labels ; `zero_grad`, `backward`, `step` ; étape de $G$ sur `D(fake)` avec le label 1. Garde les logits de l'étape de $D$ pour `d_real` et `d_fake` (`torch.sigmoid(...).mean().item()`). `train_gan_12` : deux `torch.optim.Adam`, un `torch.Generator`, deux boucles, et l'historique dans un dictionnaire de listes.

</details>
<details><summary>Indice 3</summary>

```python
def gan_step_12(G, D, opt_g, opt_d, real, z):
    fake = G(z)
    logits_real, logits_fake = D(real), D(fake.detach())
    loss_d = ...                      # BCE(logits_real, 1) + BCE(logits_fake, 0)
    # zero_grad, backward, step for D
    logits = D(fake)
    loss_g = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    # zero_grad, backward, step for G; return the dict of four floats

def train_gan_12(...):
    # step = gan_step_12 if step is None else step; two Adam with betas=(0.5, 0.999); gen = torch.Generator().manual_seed(seed)
    # for epoch in 1..n_epochs: perm = torch.randperm(len(data), generator=gen); for each mini-batch:
    #     z = sample_noise(len(real), latent_dim(G), gen, noise); append each value of step(...) to history[key]
    #     after the epoch: on_epoch(epoch, G) if given
```

</details>

### Ex 27.13 — Prédire la moyenne et l'écart-type générés au fil des epochs 🔮

<details><summary>Indice 1</summary>

Relis la description de la fig. 27.20 dans la fiche (§27.5) et l'Ex 27.4.

</details>
<details><summary>Indice 2</summary>

a) Pendant la première epoch, le discriminateur n'a vu de faux que près de 0 : qu'apprend-il, et qu'en déduit le générateur ? b) Une fois que $D$ répond $\frac{1}{2}$ partout, qu'est-ce qui, dans les deux losses, dépend encore de l'écart-type des faux ? c) Additionne deux termes à l'équilibre.

</details>
<details><summary>Indice 3</summary>

Pour a), demande-toi ce que $D$ sait, à la fin de l'epoch 1, des valeurs qu'il n'a jamais vues, d'un côté comme de l'autre de la cible. Pour b), écris $L_G$ quand $D(\mathbf{x}) = \frac{1}{2}$ pour tout $\mathbf{x}$ : son gradient par rapport aux poids de $G$ dépend-il de la forme du nuage ? Pour c), reprends l'Ex 27.4 a).

</details>

### Ex 27.14 — Lire les courbes d'un GAN : losses, D(x) et D(G(z)) 📈

<details><summary>Indice 1</summary>

Relis l'Ex 27.4 et l'encadré ⚠️ « la loss idéale » du §27.5.

</details>
<details><summary>Indice 2</summary>

À l'équilibre : loss de $D$ vers $2\ln 2$, loss de $G$ vers $\ln 2$, `d_real` et `d_fake` vers $\frac{1}{2}$. Un discriminateur qui domine : `d_real` haut, `d_fake` bas, loss de $D$ basse, loss de $G$ haute.

</details>
<details><summary>Indice 3</summary>

a) Cherche l'entraînement où l'écart entre `d_real` et `d_fake` reste grand jusqu'au bout. Pour b) et c), lis la dernière valeur des courbes demandées. Pour d), des courbes à l'équilibre disent que $D$ ne distingue plus les faux : cela prouve-t-il que les faux sont bons, ou peut-être que $D$ est trop faible pour voir leurs défauts ?

</details>

### Ex 27.15 — Reproduire l'apprentissage du nuage centré en (5, 5) (fig. 27.20-27.21) 🎨

<details><summary>Indice 1</summary>

Tout passe par ton `train_gan_12` : le paramètre `on_epoch` sert à garder les nuages.

</details>
<details><summary>Indice 2</summary>

`train_book_15` : un dictionnaire `snapshots = {0: generate(G, Z_FIXED_15)}`, une fonction `keep(epoch, G)` qui ajoute `generate(G, Z_FIXED_15)` si `epoch in EPOCHS_15`, puis `train_gan_12(..., batch_size=32, noise="uniform", seed=seed, on_epoch=keep)`. `plot_clouds_15` : `plt.subplots(3, 4)`, un panneau par epoch triée, deux `scatter`, deux `add_ellipse`, les mêmes limites. `plot_losses_15` : `mean_epochs` deux fois, deux courbes, deux `axhline`.

</details>
<details><summary>Indice 3</summary>

```python
def plot_clouds_15(snapshots, data):
    epochs = sorted(snapshots)
    fig, axes = plt.subplots(math.ceil(len(epochs) / 4), 4, figsize=(10, 8), squeeze=False)
    stats = []
    for ax, epoch in zip(axes.flat, epochs):
        fake = np.asarray(snapshots[epoch])
        # scatter 500 real points and the fake ones, add_ellipse for both, limits 0..14, title
        stats.append([*fake.mean(axis=0), *fake.std(axis=0)])
    # plt.show(), then return np.array(stats)
```

</details>

### Ex 27.16 — Le GAN qui s'entraînait à l'envers 🐛

<details><summary>Indice 1</summary>

Lis les diagnostics : chacun vise une erreur. Relis l'encadré 🧮 sur `detach` et le tableau des pièges de la fiche.

</details>
<details><summary>Indice 2</summary>

Point 1 : par où la loss de $D$ peut-elle atteindre $G$ ? Point 2 : quelle ligne remet les gradients à zéro, et de quel optimiseur ? Point 3 : quel label vise le générateur ? Corrige une erreur à la fois et relance les diagnostics.

</details>
<details><summary>Indice 3</summary>

Un exemple : le point 1 compte les passes arrière qui atteignent $G$ ; la loss de $D$ passe par `D(fake)` à la ligne (H), et rien ne coupe le graphe entre `fake` et $G$. Pour les deux autres points, regarde quelles lignes remettent des gradients à zéro (et de quel optimiseur), et quel label vise le générateur. Garde les mêmes tirages dans le même ordre dans ta version corrigée.

</details>

### Ex 27.17 — Prédire l'effet d'un discriminateur trop fort 🔮

<details><summary>Indice 1</summary>

Relis l'Ex 27.6 : avec la loss non saturante, que reçoit le générateur quand $D$ repère bien ses faux ?

</details>
<details><summary>Indice 2</summary>

Deux questions : avec la loss non saturante, quel gradient reçoit le générateur quand $D(G(\mathbf{z}))$ est petit ? Que fait Adam de l'échelle d'un gradient (ch. 19) ? Pense aussi à ce que suppose la théorie (∂ 27.5) sur le discriminateur.

</details>
<details><summary>Indice 3</summary>

Pour a), situe $D(G(\mathbf{z}))$ entre deux extrêmes : 0 (tous les faux repérés à coup sûr) et $\frac{1}{2}$ (l'équilibre). Pour b), demande-toi si le générateur reçoit encore un gradient qui le fait avancer. Pour c), un discriminateur proche de $D^*$ réagit-il plus vite ou plus lentement à un écart de forme du nuage généré ?

</details>

### Ex 27.18 — Loss saturante ou non saturante : la dynamique du début 🔬

<details><summary>Indice 1</summary>

Reprends `gan_step_12` : seule la loss du générateur change. Relis ∂ 27.6.

</details>
<details><summary>Indice 2</summary>

$\ln(1 - \sigma(a)) = -\text{BCE}(a, 0)$ : la loss minimax s'écrit avec `binary_cross_entropy_with_logits` et le label 0, au signe près. Pour la norme : `torch.autograd.grad(loss, list(G.parameters()))` rend les gradients sans toucher aux `.grad` ; la norme est la racine de la somme de tous leurs carrés.

</details>
<details><summary>Indice 3</summary>

```python
def generator_grad_norm_18(G, D, z, saturating):
    logits = D(G(z))
    if saturating:
        loss = -F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    else:
        loss = ...                    # the non-saturating loss
    grads = torch.autograd.grad(loss, list(G.parameters()))
    # return the square root of the sum of (g ** 2).sum() over grads, as a float
```

</details>

### Ex 27.19 — Le round en quatre temps du livre contre la boucle standard 🔬

<details><summary>Indice 1</summary>

Relis le §27.2.3 de la fiche et son encadré ⚠️, puis 27.Q4.

</details>
<details><summary>Indice 2</summary>

Une petite fonction interne `learn(logits, target, selected, optimizer)` fait une étape : si `errors_only` est faux, tout est choisi ; s'il n'y a rien de choisi, elle ne fait rien ; sinon la cross-entropy par échantillon (`reduction="none"`), multipliée par le masque, sommée et divisée par la taille du mini-batch, puis `zero_grad`, `backward`, `step`. Les masques : `(logits < 0).float()` pour « D dit faux », `(logits >= 0).float()` pour « D dit vrai ».

</details>
<details><summary>Indice 3</summary>

```python
    def learn(logits, target, selected, optimizer):
        # all selected when errors_only is False; nothing selected: return 0 without any step
        losses = F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target), reduction="none")
        loss = (losses * selected).sum() / len(logits)
        # zero_grad, backward, step; return the number of selected samples

    # (a) D(real), target 1, selected where the logit < 0, opt_d
    # (b), (d) D(G(z)), target 1, logit < 0, opt_g ; (c) D(G(z_c).detach()), target 0, logit >= 0, opt_d
```

</details>

### Ex 27.20 — Un DCGAN sur MNIST selon les règles de Radford 📦

<details><summary>Indice 1</summary>

Relis le §27.6.1 de la fiche et l'Ex 27.1 : les tailles $7 \to 14 \to 28$ et les nombres de paramètres.

</details>
<details><summary>Indice 2</summary>

Générateur : `nn.Linear(nz, 2 * width * 7 * 7, bias=False)`, `nn.BatchNorm1d`, `nn.ReLU`, `nn.Unflatten(1, (2 * width, 7, 7))`, puis `nn.ConvTranspose2d(..., 4, stride=2, padding=1)` deux fois (la première sans biais, suivie d'une batchnorm et d'une ReLU, la seconde vers 1 canal, suivie d'une tanh). Discriminateur : `nn.Conv2d(..., 4, stride=2, padding=1)` deux fois, `nn.LeakyReLU(0.2)`, une `nn.BatchNorm2d` après la seconde seulement, `nn.Flatten`, `nn.Linear(2 * width * 7 * 7, 1)`. Initialisation : `isinstance`, puis `nn.init.normal_` et `nn.init.zeros_`.

</details>
<details><summary>Indice 3</summary>

```python
def make_generator_20(nz=NZ, width=32):
    return nn.Sequential(
        nn.Linear(nz, 2 * width * 7 * 7, bias=False), nn.BatchNorm1d(2 * width * 7 * 7), nn.ReLU(),
        nn.Unflatten(1, (2 * width, 7, 7)),
        # a ConvTranspose2d 2*width -> width (no bias), BatchNorm2d, ReLU, then a ConvTranspose2d width -> 1, Tanh
    )

def init_weights_20(module):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    # the batchnorms: weight ~ N(1, 0.02²), bias 0
```

</details>

### Ex 27.21 — Règles empiriques à l'épreuve : une architecture de 2016 contre celle de Radford 🔬

<details><summary>Indice 1</summary>

Relis la description du modèle du livre au §27.6 de la fiche, et compare-la règle par règle au §27.6.1.

</details>
<details><summary>Indice 2</summary>

Générateur : deux `nn.Linear` (la seconde suivie d'une `nn.BatchNorm1d`), des `nn.Tanh`, `nn.Unflatten`, puis deux fois `nn.Upsample(scale_factor=2)` suivi de `nn.Conv2d(..., 5, padding=2)` et d'une tanh. Discriminateur : deux fois `nn.Conv2d(..., 5, padding=2)`, `nn.Tanh`, `nn.MaxPool2d(2)`, puis `nn.Flatten` et deux `nn.Linear`. Vérifie tes nombres de paramètres avec `n_params` avant de lancer l'entraînement.

</details>
<details><summary>Indice 3</summary>

```python
    G = nn.Sequential(
        nn.Linear(nz, hidden), nn.Tanh(),
        nn.Linear(hidden, 2 * width * 7 * 7), nn.BatchNorm1d(2 * width * 7 * 7), nn.Tanh(),
        nn.Unflatten(1, (2 * width, 7, 7)),
        # twice: Upsample(x2), Conv2d 5x5 with padding 2 (towards width, then 1 channel), Tanh
    )
    # D: twice (Conv2d 5x5 with padding 2, Tanh, MaxPool2d(2)), then Flatten and two Linear layers
```

</details>

### Ex 27.22 — Distance de Fréchet entre deux gaussiennes (la formule du FID) 🔨

<details><summary>Indice 1</summary>

Relis la formule de la distance de Fréchet et l'encadré 🧮 sur la trace et la racine carrée, dans la fiche (« Au-delà du livre »).

</details>
<details><summary>Indice 2</summary>

`frechet_distance` : valide les formes ; le premier terme est la somme des carrés de `mu1 - mu2` ; la trace de la racine vient de `np.linalg.eigvals(sigma1 @ sigma2)` : garde la partie réelle, ramène les négatives à 0, prends la racine et somme ; finis par `max(d2, 0.0)`. `frechet_distance_from_features` : valide, puis `feats.mean(axis=0)` et `np.cov(feats, rowvar=False)` (remis en forme `(d, d)` quand $d = 1$, où `np.cov` rend un nombre).

</details>
<details><summary>Indice 3</summary>

```python
def frechet_distance(mu1, sigma1, mu2, sigma2):
    # validation: mu1 1-D of length d, mu2 of shape (d,), sigma1 and sigma2 of shape (d, d), finite numbers
    eigenvalues = np.linalg.eigvals(sigma1 @ sigma2)
    trace_sqrt = np.sum(np.sqrt(np.clip(eigenvalues.real, 0.0, None)))
    d2 = ...                          # the formula of the fiche, with trace_sqrt for its last term
    return float(max(d2, 0.0))
```

</details>

### Ex 27.23 — Évaluer un générateur : distance de Fréchet sur les features d'un classifieur 📦

<details><summary>Indice 1</summary>

Les outils font presque tout : `sample_images`, `features`, et ta `frechet_distance_from_features`.

</details>
<details><summary>Indice 2</summary>

Trois appels enchaînés : générer $n$ images avec la graine, calculer leurs features, mesurer la distance aux features de référence.

</details>
<details><summary>Indice 3</summary>

```python
def fid_23(G, clf, real_feats, n=2000, seed=0):
    fake_feats = features(clf, sample_images(G, n, seed))
    # return float(your frechet_distance_from_features(...)) between fake_feats and real_feats
```

</details>

### Ex 27.24 — Provoquer puis soigner un effondrement de mode (anneau de 8 gaussiennes) 🔬

<details><summary>Indice 1</summary>

Relis le §27.7.2 de la fiche, et l'encadré ⚠️ du §27.5 sur le mini-batch.

</details>
<details><summary>Indice 2</summary>

`ring_24` : les composantes d'abord (`rng.integers(k, size=n)`), leurs angles $2\pi j / k$, les centres $r(\cos, \sin)$, puis le bruit (`rng.normal(0, std, size=(n, 2))`), en `float32`. `modes_covered_24` : la matrice des distances de chaque point à chaque centre (diffusion NumPy, `np.linalg.norm(..., axis=2)`), le centre le plus proche, le masque des bons points, `np.bincount` des bons points par centre.

</details>
<details><summary>Indice 3</summary>

```python
def modes_covered_24(points, k=8, radius=2.0, threshold=0.15, min_share=0.01):
    # the k centres on the circle (angles 2*pi*j/k)
    dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)   # (n, k)
    nearest, good = dist.argmin(axis=1), dist.min(axis=1) < threshold
    # count the good points of each centre, compare each count to min_share * n, return (int, float)
```

</details>

### Ex 27.25 — Un GAN conditionnel : choisir le chiffre généré 📦

<details><summary>Indice 1</summary>

Relis le paragraphe « le GAN conditionnel » de la fiche, et l'embedding du ch. 24.

</details>
<details><summary>Indice 2</summary>

Générateur : `self.embed = nn.Embedding(10, emb_dim)` et `self.net = make_generator_20(nz + emb_dim, width)` ; `forward` concatène `z` et l'embedding sur la dimension 1. Discriminateur : `nn.Embedding(10, 28 * 28)`, `.view(-1, 1, 28, 28)`, concaténé après l'image sur la dimension des canaux, puis `make_discriminator_20(width, in_channels=2)`. `cgan_step_25` : `gan_step_12`, en passant les labels à chaque appel de $G$ et de $D$.

</details>
<details><summary>Indice 3</summary>

```python
class CondDiscriminator25(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.embed = nn.Embedding(10, 28 * 28)
        self.net = make_discriminator_20(width, in_channels=2)

    def forward(self, images, labels):
        # the embedding of the labels reshaped to (N, 1, 28, 28), put after the images on the channel axis, then self.net
```

</details>

### Ex 27.26 — Se promener dans l'espace latent du générateur 🔬

<details><summary>Indice 1</summary>

Relis `interpolate_latents` dans ta librairie `vae.py` (ch. 25), et la question de la norme dans l'énoncé.

</details>
<details><summary>Indice 2</summary>

`latent_path_26` : convertis $\mathbf{z}_0$ et $\mathbf{z}_1$ en tableaux NumPy de flottants, appelle `mylearn.vae.interpolate_latents`, puis `generate`. `path_report_26` : `predict`, puis les différences entre images voisines (`np.diff` sur l'axe 0, après avoir aplati chaque image), la moyenne des carrés par image, la racine.

</details>
<details><summary>Indice 3</summary>

```python
def path_report_26(clf, images):
    images = np.asarray(as_numpy(images), dtype=float).reshape(len(images), -1)
    # the differences between consecutive images (np.diff along axis 0), then for each one the root of its mean square
    # return (the classes predicted by clf, the jumps)
```

</details>

### Ex 27.27 — Checkpoint d'un GAN : deux réseaux, deux optimiseurs, états aléatoires, reprise à l'identique 🛠️

<details><summary>Indice 1</summary>

Relis le checkpoint complet du ch. 23, et l'encadré « En entreprise » du §27.6.1.

</details>
<details><summary>Indice 2</summary>

Un seul dictionnaire, sauvé par `torch.save` : quatre `state_dict()` (les batchnorms y sont, avec leurs moyennes glissantes), l'epoch en `int`, `gen.get_state()` et `torch.get_rng_state()` (deux tenseurs). Au chargement : `torch.load(path, weights_only=True)`, quatre `load_state_dict`, `gen.set_state(...)`, `torch.set_rng_state(...)`.

</details>
<details><summary>Indice 3</summary>

```python
def save_gan_checkpoint_27(path, G, D, opt_g, opt_d, epoch, gen):
    torch.save({"G": G.state_dict(), "D": D.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                # the epoch, the state of gen and the global random state of PyTorch
                }, path)
```

</details>

### Ex 27.28 — Défi DCGAN Fashion-MNIST : passer sous un seuil de distance de Fréchet 🏆

<details><summary>Indice 1</summary>

Mesure d'abord le point de départ : le DCGAN de 27.20, entraîné comme en 27.20, et sa distance publique. Puis change **une** chose à la fois.

</details>
<details><summary>Indice 2</summary>

En 27.20, la distance baissait encore d'une epoch à l'autre : où en est-on après 3 epochs, et combien d'epochs tiennent dans le temps permis ? Les autres leviers : la largeur des réseaux (le coût par epoch monte vite), la taille des mini-batches (plus de pas par epoch), les learning rates. Chronomètre une epoch avant de choisir.

</details>
<details><summary>Indice 3</summary>

Un squelette, à régler :

```python
def make_gan_28():
    G, D = make_generator_20(NZ, 32), make_discriminator_20(32)
    G.apply(init_weights_20)
    D.apply(init_weights_20)
    return G, D

def train_28(G, D, images):
    train_gan_12(G, D, images[:10_000], n_epochs=..., batch_size=64, lr_g=2e-4, lr_d=2e-4, seed=...)
```

</details>
