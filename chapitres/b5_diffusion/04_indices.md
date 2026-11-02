# B5 · Modèles de diffusion : un DDPM minimal — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B5.Q1 — Diffusion, VAE, GAN : comment chacun fabrique-t-il une image ?

<details><summary>Indice 1</summary>

Relis « Trois façons de fabriquer une image » et l'encadré 🔗 « Lien avec les ch. 25 et 27 » (§205.1).

</details>
<details><summary>Indice 2</summary>

Pour chaque modèle, demande-toi : combien de passes du réseau pour une image, et contre quoi il apprend (une MSE, une ELBO, un discriminateur). Le processus direct de la diffusion est fixé une fois pour toutes ; ses « codes » sont les images bruitées.

</details>
<details><summary>Indice 3</summary>

a) **A** : le décodeur d'un VAE transforme en une passe un code tiré de la loi a priori. Pour b), relis « Détruire est facile » : qu'est-ce qui est appris ? Pour c), lequel a un adversaire ? Pour d), quelle taille ont $\mathbf{x}_1, \dots, \mathbf{x}_T$ ? Pour e), multiplie le coût d'une passe par le nombre de passes.

</details>

### B5.Q2 — Pourquoi détruire l'image en mille petits pas de bruit plutôt qu'en un seul ?

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « la MSE apprend une moyenne » (§205.1) et « Ce qu'on voudrait » (§205.3).

</details>
<details><summary>Indice 2</summary>

Quand plusieurs cibles sont possibles pour la même entrée, la prédiction qui minimise la MSE est leur moyenne, et la MSE qui reste est leur variance. Un petit pas de bruit a une réponse presque certaine ; à destruction finale égale, $\ln \bar\alpha_T \approx -\sum_t \beta_t$.

</details>
<details><summary>Indice 3</summary>

a) **B** : mille images différentes auraient pu donner ce bruit, et la MSE rend leur moyenne, floue. Pour b), calcule la moyenne de $-1$, $0$ et $2$. Pour c), calcule la moyenne des carrés des écarts à ta réponse b). Pour d), relis la condition « $\beta_t$ petit » de 205.3. Pour e), garde $\sum_t \beta_t$ fixe et double le nombre de termes.

</details>

### B5.Q3 — ᾱ_t : quelle part de l'image reste-t-il au pas t ?

<details><summary>Indice 1</summary>

Relis « La forme fermée » et « Le rapport signal sur bruit » (§205.2).

</details>
<details><summary>Indice 2</summary>

$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}$ : les coefficients sont des **racines**, et ce sont leurs carrés qui font 1. Le SNR est un rapport de variances, $\bar\alpha_t / (1 - \bar\alpha_t)$.

</details>
<details><summary>Indice 3</summary>

a) $\sqrt{0{,}64} =$ **0,80**. Pour b), prends la racine de $1 - 0{,}64$. Pour c), additionne a) et b). Pour d), calcule $0{,}64 / (1 - 0{,}64)$. Pour e), $\bar\alpha_t$ est un produit de nombres $\alpha_s < 1$ : que fait un facteur de plus ?

</details>

### B5.Q4 — Prédire le bruit, l'image ou la vitesse : la même information ?

<details><summary>Indice 1</summary>

Relis « Prédire le bruit » (§205.3), l'encadré 🕰️ sur les paramétrisations, et le paragraphe qui suit l'encadré sur Euler (§205.9).

</details>
<details><summary>Indice 2</summary>

La forme fermée lie $\mathbf{x}_t$, $\mathbf{x}_0$ et $\boldsymbol{\varepsilon}$ : connaissant deux d'entre eux et $t$, on calcule le troisième. Ho et coll. ont comparé trois cibles ; la fiche donne une raison intuitive à leur choix. La « vitesse » du flow matching et la $\mathbf{v}$ de Salimans et Ho portent le même nom.

</details>
<details><summary>Indice 3</summary>

a) **A** : $\mathbf{x}_0 = (\mathbf{x}_t - \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}) / \sqrt{\bar\alpha_t}$, sans approximation. Pour b), calcule $(0{,}5 - \sqrt{0{,}36} \times 0{,}2) / \sqrt{0{,}64}$. Pour c), relis le paragraphe qui suit la moyenne $\boldsymbol{\mu}_\theta$. Pour d), quelle propriété du bruit ne change pas d'un pas à l'autre ? Pour e), écris les deux combinaisons de $\boldsymbol{\varepsilon}$ et de $\mathbf{x}_0$ et compare-les.

</details>

### B5.Q5 — Le pas de temps est une entrée du réseau : pourquoi, et comment l'encoder ?

<details><summary>Indice 1</summary>

Relis « Dire au réseau quel est le pas » et « Un bloc résiduel conditionné par le pas » (§205.4).

</details>
<details><summary>Indice 2</summary>

Un seul réseau sert à tous les pas ; il doit savoir ce qu'on attend de lui à ce pas-là. L'encodage sinusoïdal range un sinus et un cosinus par fréquence ; le bloc résiduel le transforme par une couche dense en un vecteur de la taille du nombre de canaux.

</details>
<details><summary>Indice 3</summary>

a) **C** : le même réseau sert à tous les pas, et le pas lui dit combien de bruit il y a. Pour b), relis ce que la fiche dit d'un nombre de 0 à 999 donné tel quel. Pour c), divise `d_model` par le nombre de colonnes d'un couple. Pour d), lis la ligne `h = h + self.time(temb)[:, :, None, None]`. Pour e), qu'est-ce qu'un MLP voit d'un ordre de colonnes fixé une fois pour toutes ?

</details>

### B5.Q6 — U-Net : à quoi servent les connexions de saut ?

<details><summary>Indice 1</summary>

Relis « Le U-Net et ses connexions de saut », le piège « deux skip connections » et le paragraphe sur la group norm (§205.4).

</details>
<details><summary>Indice 2</summary>

Une connexion de saut du U-Net **concatène** sur l'axe des canaux : le nombre de canaux d'entrée de la convolution suivante est la somme des deux. Une convolution de stride 2 divise largeur et hauteur par 2. La group norm normalise chaque image séparément.

</details>
<details><summary>Indice 3</summary>

a) **A** : elle recopie une carte de l'encodeur et la concatène à la carte du décodeur de même résolution. Pour b), additionne les canaux du décodeur et ceux de l'encodeur. Pour c), relis le paragraphe qui suit la figure du U-Net. Pour d), divise 32 par 2 autant de fois qu'il y a de réductions. Pour e), qu'est-ce qui distingue la group norm de la batchnorm dans la fiche ?

</details>

### B5.Q7 — Échantillonnage DDPM : pourquoi est-il lent et aléatoire ?

<details><summary>Indice 1</summary>

Relis « L'algorithme », « Quelle variance ? » et « Ce que coûte un échantillon » (§205.5).

</details>
<details><summary>Indice 2</summary>

Chaque pas de l'algorithme 2 prend la sortie du pas précédent et tire un $\mathbf{z}$ neuf, sauf au dernier. Le réseau ne prédit que la moyenne du pas ; la variance $\sigma_t^2$ est une constante choisie à part.

</details>
<details><summary>Indice 3</summary>

a) $1\,000 \times 0{,}05 =$ **50** s. Pour b), que tire-t-on à chaque pas ? Pour c), relis la phrase qui suit l'algorithme 2. Pour d), de quoi la passe du pas $t$ a-t-elle besoin ? Pour e), écris le pas sous la forme « moyenne + $\sigma_t\,\mathbf{z}$ » et cherche où entre $\sigma_t^2$.

</details>

### B5.Q8 — DDIM : moins de pas avec le même réseau, vrai ou faux

<details><summary>Indice 1</summary>

Relis toute la section 205.6, surtout « L'idée de DDIM », « Le paramètre η » et « Pourquoi c'est plus rapide ».

</details>
<details><summary>Indice 2</summary>

DDIM ne touche pas à l'entraînement : il n'utilise que la forme fermée. Son pas estime $\hat{\mathbf{x}}_0$ puis le rebruite au niveau d'un indice antérieur quelconque ; $\eta$ règle le hasard. Les trajectoires se suivent mieux avec assez de pas.

</details>
<details><summary>Indice 3</summary>

a) **Faux** : DDIM réutilise le même réseau, entraîné une fois avec ses 1 000 pas. Pour b), y a-t-il un tirage quand $\eta = 0$ ? Pour c), que vaut $\sigma^2$ quand $\eta = 1$ et $t' = t - 1$ ? Pour d), relis l'étape 2 de DDIM et le cas $t' = 0$. Pour e), relis ce que la fiche dit d'une trajectoire suivie avec trop peu de pas.

</details>

### B5.Q9 — Guidance sans classifieur : ce que règle l'échelle w

<details><summary>Indice 1</summary>

Relis « La guidance sans classifieur », le paragraphe qui suit la figure, et le piège classique sur les deux conventions (§205.7).

</details>
<details><summary>Indice 2</summary>

$\tilde\varepsilon = (1 + w)\,\varepsilon_\theta(\mathbf{x}_t, c) - w\,\varepsilon_\theta(\mathbf{x}_t, \varnothing)$, et diffusers écrit `guidance_scale` $= 1 + w$. Pendant l'entraînement, la condition vide remplace le label pour une part $p_{\text{uncond}}$ des exemples.

</details>
<details><summary>Indice 3</summary>

a) $3 \times 0{,}3 - 2 \times 0{,}5 =$ **−0,1**. Pour b), remplace $w$ par 0 dans la formule. Pour c), relis ce que mesurent Ho et Salimans entre petite et forte guidance. Pour d), résous $1 + w = 3$. Pour e), comment un seul réseau apprend-il aussi la version sans condition ?

</details>

### B5.Q10 — Un bon FID peut-il cacher de la copie des données d'entraînement ?

<details><summary>Indice 1</summary>

Relis « La distance de Fréchet », « Fidélité et diversité » et « La mémorisation » (§205.8).

</details>
<details><summary>Indice 2</summary>

La distance de Fréchet ne compare que des moyennes et des covariances de features : elle ignore d'où vient chaque image. La mémorisation se cherche image par image, par les plus proches voisins. Une distance dépend du nombre d'images sur lequel on l'estime.

</details>
<details><summary>Indice 3</summary>

a) **Vrai** : des copies exactes ont exactement les statistiques des vraies images. Pour b), quel outil du ch. 13 compare une image à celles du dataset ? Pour c), relis la dernière puce des résultats de Carlini et coll. Pour d), relis la deuxième des trois règles. Pour e), un seul nombre peut-il séparer fidélité et diversité ?

</details>

### B5.Q11 — Diffusion latente, transformers de diffusion, flow matching : ce qui a changé depuis 2020

<details><summary>Indice 1</summary>

Relis toute la section 205.9 et l'encadré 🧮 sur la méthode d'Euler.

</details>
<details><summary>Indice 2</summary>

Un facteur $f$ divise la largeur **et** la hauteur ; le latent a son propre nombre de canaux. Un DiT découpe le latent comme un ViT découpe une image. La vitesse du flow matching est la dérivée de $\mathbf{x}_s$ par rapport à $s$.

</details>
<details><summary>Indice 3</summary>

a) $(256/4) \times (256/4) \times 3 =$ **12 288**. Pour b), relis la phrase sur les patchs de Peebles et Xie. Pour c), dérive $(1 - s)\,\mathbf{x}_0 + s\,\boldsymbol{\varepsilon}$ par rapport à $s$. Pour d), d'où vient l'erreur d'Euler, d'après l'encadré ? Pour e), relis ce que la fiche dit de l'architecture et de l'espace de travail de SD3 et de FLUX.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### B5.R1 — Ch. B3 : l'encodage sinusoïdal des positions

<details><summary>Indice 1</summary>

Relis l'encodage sinusoïdal en B3 et « Dire au réseau quel est le pas » (§205.4).

</details>
<details><summary>Indice 2</summary>

Les colonnes paires sont des sinus, les impaires des cosinus. La paire $i$ a la fréquence $1/10\,000^{2i/d}$ : une période de $2\pi \cdot 10\,000^{2i/d}$ positions. Les petits $i$ tournent vite, les grands lentement.

</details>
<details><summary>Indice 3</summary>

a) $\operatorname{PE}[0, 1] = \cos(0) =$ **1**. Pour b), la colonne 0 vaut $\sin(p)$ : quelle est sa période ? Pour c), quelles colonnes changent le plus entre deux positions voisines ? Pour d), calcule $2\pi \cdot 10\,000^{126/128}$ et compare à 1 000. Pour e), relis la formule de l'énoncé.

</details>

### B5.R2 — Ch. 27 : ce que mesure la distance de Fréchet et l'effondrement de mode

<details><summary>Indice 1</summary>

Relis la distance de Fréchet et l'effondrement de mode au ch. 27, et « La distance de Fréchet » (§205.8).

</details>
<details><summary>Indice 2</summary>

En dimension 1, avec des variances $v_1$ et $v_2$ : $d_F^2 = (\mu_1 - \mu_2)^2 + v_1 + v_2 - 2\sqrt{v_1 v_2} = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$. Le premier terme compare les moyennes, le second les écarts-types.

</details>
<details><summary>Indice 3</summary>

a) $(0 - 1)^2 + (1 - 2)^2 =$ **2**. Pour b), que deviennent la moyenne et la covariance des features d'un générateur qui ne fait plus que deux classes ? Pour c), quand chacun des deux termes est-il nul ? Pour d), sur quoi le FID calcule-t-il ses statistiques ? Pour e), applique la formule de l'indice 2 avec $\sigma_2 = 0$.

</details>

### B5.R3 — Ch. 25 : l'astuce de reparamétrisation et le terme KL du VAE

<details><summary>Indice 1</summary>

Relis la reparamétrisation et le terme KL au ch. 25, l'encadré 🔗 « Lien avec la reparamétrisation » et « Le lien avec l'ELBO » (§205.3).

</details>
<details><summary>Indice 2</summary>

$z = \mu + \sigma\,\varepsilon$ avec $\sigma = \exp(0{,}5 \ln \sigma^2)$. $\mathrm{KL}\big(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)\big) = \frac{1}{2}\big(\mu^2 + \sigma^2 - 1 - \ln \sigma^2\big)$. Dans un DDPM, le processus direct est fixé.

</details>
<details><summary>Indice 3</summary>

a) $\sigma = 0{,}5$, donc $z = 0{,}5 + 0{,}5 \times (-2) =$ **−0,50**. Pour b), calcule $\frac{1}{2}(0{,}25 + 0{,}25 - 1 - \ln 0{,}25)$. Pour c), qu'est-ce que l'astuce permet de faire avec le gradient ? Pour d), relis l'encadré 🔗 de 205.3. Pour e), ce terme dépend-il d'un paramètre appris ?

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex B5.1 — Guidance sans classifieur : combiner les deux prédictions pour w = 0, 1 et 3 ✏️

<details><summary>Indice 1</summary>

Relis « La guidance sans classifieur », l'exemple chiffré qui la suit et le piège classique sur les deux conventions (§205.7).

</details>
<details><summary>Indice 2</summary>

Applique la formule composante par composante : $\tilde\varepsilon = (1 + w)\,\varepsilon_c - w\,\varepsilon_\varnothing = \varepsilon_c + w\,(\varepsilon_c - \varepsilon_\varnothing)$. La seconde écriture montre que $\tilde{\boldsymbol{\varepsilon}} - \boldsymbol{\varepsilon}_c$ est $w$ fois l'écart des deux prédictions. Chaque pas guidé demande les deux prédictions.

</details>
<details><summary>Indice 3</summary>

a) Avec $w = 0$, $\tilde{\boldsymbol{\varepsilon}} = \boldsymbol{\varepsilon}_c =$ **[0.8, -0.5]**. Pour b) et c), calcule $\boldsymbol{\varepsilon}_c + w\,(0{,}3 ;\ -0{,}6)$. Pour d), résous `guidance_scale` $= 1 + w$. Pour e), multiplie la norme $\sqrt{0{,}3^2 + 0{,}6^2}$ par $w$. Pour f), remplace $w$ par $-1$. Pour g), compare chaque composante de c) à celles de $\boldsymbol{\varepsilon}_\varnothing$ et $\boldsymbol{\varepsilon}_c$. Pour h), multiplie le nombre de pas par le nombre de prédictions par pas.

</details>

### Ex B5.2 — Planning linéaire : β_t, α_t et ᾱ_t pour T = 4, puis ᾱ_T pour T = 1 000 ✏️

<details><summary>Indice 1</summary>

Relis « Un planning jouet, à la main » et « Le planning linéaire de Ho et coll. » (§205.2).

</details>
<details><summary>Indice 2</summary>

Quatre valeurs régulièrement espacées de 0,05 à 0,35 font trois intervalles égaux. $\alpha_t = 1 - \beta_t$, et $\bar\alpha_t$ est le produit des $\alpha$ jusqu'au pas $t$ ; la part de l'image est sa racine. La somme d'une suite arithmétique de $T$ termes vaut $T \times (\text{premier} + \text{dernier}) / 2$, et $\bar\alpha_T \approx e^{-\sum \beta}$.

</details>
<details><summary>Indice 3</summary>

a) L'écart entre deux $\beta$ vaut $0{,}30 / 3 = 0{,}10$, donc $\beta_2 =$ **0,15**. Pour b) et c), multiplie $0{,}95 \times 0{,}85 \times \dots$ jusqu'au bon pas. Pour d), prends la racine de c). Pour e), calcule $e^{-(0{,}05 + 0{,}15 + 0{,}25 + 0{,}35)}$. Pour f), calcule l'écart relatif entre e) et c). Pour g), $1\,000 \times (10^{-4} + 0{,}02)/2$. Pour h), $e^{-g)} \times 10^5$. Pour i), $100 \times \sqrt{h) \times 10^{-5}}$. Pour j) et k), la même somme avec 400 termes, et les bonnes bornes. Pour l), compare g) et k).

</details>

### Ex B5.3 — Rapport signal sur bruit : à quel pas un chiffre devient-il illisible, planning linéaire ou cosinus ? ✏️

<details><summary>Indice 1</summary>

Relis « Le planning cosinus de Nichol et Dhariwal » et « Le rapport signal sur bruit » (§205.2). Tu as besoin de résoudre une équation du second degré et d'un arc cosinus.

</details>
<details><summary>Indice 2</summary>

Traduis d'abord chaque seuil de SNR en seuil de $\bar\alpha$ : $\operatorname{SNR} = r$ donne $\bar\alpha = r / (1 + r)$. Linéaire : $e^{-S(t)} = \bar\alpha$ s'écrit $S(t) = -\ln \bar\alpha$, une équation du second degré en $t$. Cosinus : $\cos^2\theta = \bar\alpha\,f(0)$ avec $\theta = \frac{u + s}{1 + s}\cdot\frac{\pi}{2}$ ; résous en $\theta$, puis en $u$.

</details>
<details><summary>Indice 3</summary>

a) $S(500) = 500 \times 10^{-4} + 0{,}0199 \times \frac{500 \times 499}{2 \times 999} =$ **2,535**. Pour b), calcule $\bar\alpha = e^{-S(500)}$, puis $\bar\alpha / (1 - \bar\alpha)$. Pour c), calcule $f(0{,}5)/f(0)$, puis le SNR. Pour d) et f), résous $S(t) = \ln 2$, puis $S(t) = \ln 11$, et divise $t$ par 1 000. Pour e) et g), résous $\cos^2\theta = \frac{1}{2}f(0)$, puis $\cos^2\theta = \frac{1}{11}f(0)$, et tire $u = \frac{2\theta}{\pi}(1 + s) - s$. Pour h) et i), compare f) et g) à la part demandée. Pour j), compare f) − d) et g) − e).

</details>

### Ex B5.4 — Un pas DDPM à la main, en dimension 1, avec un bruit prédit donné ✏️

<details><summary>Indice 1</summary>

Relis « Un pas à la main » (§205.5) et le posterior (§205.3). Prends les constantes du planning de ✏️ B5.2 au **pas 3** (indice 2).

</details>
<details><summary>Indice 2</summary>

$\mu_\theta = \frac{1}{\sqrt{\alpha_3}}\big(x_3 - \frac{\beta_3}{\sqrt{1 - \bar\alpha_3}}\,\varepsilon_\theta\big)$ ; $\tilde\beta_3 = \frac{1 - \bar\alpha_2}{1 - \bar\alpha_3}\,\beta_3$ ; $x_2 = \mu + \sigma z$, où $\sigma$ est la **racine** de la variance choisie. Avec écrêtage : $\hat{x}_0 = (x_3 - \sqrt{1 - \bar\alpha_3}\,\varepsilon_\theta)/\sqrt{\bar\alpha_3}$, ramené dans $[-1, 1]$, puis la moyenne du posterior avec ses deux coefficients. Au pas 1, $\bar\alpha_0 = 1$.

</details>
<details><summary>Indice 3</summary>

a) $0{,}25 / \sqrt{1 - 0{,}605625} = 0{,}25 / 0{,}62799 =$ **0,3981**. Pour b), calcule $(0{,}8 - a) \times 0{,}5) / \sqrt{0{,}75}$. Pour c), calcule $\frac{1 - 0{,}8075}{1 - 0{,}605625} \times 0{,}25$. Pour d) et e), ajoute $\sigma \times (-1)$ à b), avec $\sigma = \sqrt{c)}$, puis $\sigma = \sqrt{\beta_3}$. Pour f), calcule $\hat{x}_0$, écrête-le, puis $\frac{\sqrt{\bar\alpha_2}\,\beta_3}{1 - \bar\alpha_3}\,\hat{x}_0 + \frac{\sqrt{\alpha_3}(1 - \bar\alpha_2)}{1 - \bar\alpha_3}\,x_3$. Pour g), la formule de b) avec les constantes de l'indice 0, sans $z$. Pour h), relis la docstring de `ddpm_step`. Pour i), où la variance entre-t-elle dans tes calculs ?

</details>

### Ex B5.5 — Un pas DDIM à la main et l'image estimée x̂_0 ✏️

<details><summary>Indice 1</summary>

Relis « L'idée de DDIM » et « Le paramètre η » (§205.6), et la docstring de `ddim_step`. Constantes du planning de ✏️ B5.2.

</details>
<details><summary>Indice 2</summary>

$\hat{x}_0 = (x_4 - \sqrt{1 - \bar\alpha_4}\,\varepsilon_\theta)/\sqrt{\bar\alpha_4}$, puis $x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat{x}_0 + \sqrt{1 - \bar\alpha_{t'} - \sigma^2}\,\varepsilon_\theta + \sigma z$, avec $\sigma = \eta\sqrt{\frac{1 - \bar\alpha_{t'}}{1 - \bar\alpha_t}}\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t'}}}$. Le pas $t$ de l'article est l'indice $t - 1$ ; l'indice $-1$ est l'image propre.

</details>
<details><summary>Indice 3</summary>

a) $(-0{,}5 - 0{,}77868 \times (-1{,}2)) / 0{,}62742 =$ **0,6924**. Pour b), calcule $\sqrt{\bar\alpha_1}\,\hat{x}_0 + \sqrt{1 - \bar\alpha_1}\,\varepsilon_\theta$. Pour c), traduis « pas 4 » et « pas 1 » en indices. Pour d), la même formule avec $\bar\alpha_2$. Pour e), recalcule $\hat{x}_0$ à partir de d) et du même $\varepsilon_\theta$, puis fais le pas. Pour f) et g), applique la formule de $\sigma$ avec $\eta = 1$, $t = 4$ et $t' = 3$, puis $t' = 1$. Pour h), calcule $\sqrt{1 - 0{,}95 - \sigma^2}$ avec le $\sigma$ de g). Pour i), compare g) et h).

</details>

### Ex B5.6 — Forme fermée : deux pas de bruit gaussien n'en font qu'un, d'où q(x_t | x_0) ∂

<details><summary>Indice 1</summary>

Relis « Un pas de bruit », l'encadré 🧮 sur la somme de deux gaussiennes indépendantes et « La forme fermée » (§205.2), et le rappel sur la récurrence de l'énoncé.

</details>
<details><summary>Indice 2</summary>

Deux règles suffisent : $aX \sim \mathcal{N}(0, a^2)$ si $X \sim \mathcal{N}(0, 1)$, et deux gaussiennes **indépendantes** s'additionnent en additionnant leurs variances. Pour l'hérédité, remplace $x_{t-1}$ par l'hypothèse dans $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1 - \alpha_t}\,\varepsilon_t$, puis regroupe le terme en $x_0$ et les deux bruits. L'identité clé : $\alpha_t(1 - \bar\alpha_{t-1}) + 1 - \alpha_t = 1 - \bar\alpha_t$.

</details>
<details><summary>Indice 3</summary>

1. $x_1 = \sqrt{\alpha_1}\,x_0 + \sqrt{1 - \alpha_1}\,\varepsilon_1 \sim \mathcal{N}(\sqrt{\alpha_1}\,x_0,\ 1 - \alpha_1)$, et $\bar\alpha_1 = \alpha_1$ : c'est la forme fermée pour $t = 1$. Pour 2, $x_2 = \sqrt{\alpha_2\alpha_1}\,x_0 + \sqrt{\alpha_2(1 - \alpha_1)}\,\varepsilon_1 + \sqrt{1 - \alpha_2}\,\varepsilon_2$ : additionne les variances des deux derniers termes. Pour 3, $\bar\varepsilon_{t-1}$ est une combinaison de quels bruits ? Pour 4, calcule $0{,}85 \times 0{,}05 + 0{,}15$ et $1 - 0{,}8075$. Pour 5, calcule $\bar\alpha_t \operatorname{Var}(x_0) + 1 - \bar\alpha_t$. Pour 6, additionne les variances $\beta_t$. Pour 7, refais la question 3 en partant de $x_s$, puis cherche $1 - \bar\alpha_t/\bar\alpha_{t'}$ dans la formule de $\sigma$ de DDIM.

</details>

### Ex B5.7 — Moyenne et variance du posterior q(x_{t−1} | x_t, x_0) en dimension 1 ∂

<details><summary>Indice 1</summary>

Relis la règle de Bayes (ch. 4), « Le posterior » et l'encadré 🧮 sur le produit de deux densités gaussiennes (§205.3).

</details>
<details><summary>Indice 2</summary>

Les deux facteurs, vus comme des fonctions de $y$, sont des gaussiennes : $\mathcal{N}(y;\ x_t/\sqrt{\alpha_t},\ \beta_t/\alpha_t)$ et $\mathcal{N}(y;\ \sqrt{\bar\alpha_{t-1}}\,x_0,\ 1 - \bar\alpha_{t-1})$. Les précisions s'additionnent, la moyenne est la moyenne des moyennes pondérée par les précisions. L'identité $\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t = 1 - \bar\alpha_t$ simplifie tout.

</details>
<details><summary>Indice 3</summary>

1. Le pas $t$ ajoute un bruit $\varepsilon_t$ indépendant de tout le passé : sachant $y = x_{t-1}$, $x_0$ n'apporte rien sur $x_t$, d'où $q(x_t \mid y, x_0) = q(x_t \mid y)$ ; le second facteur est la forme fermée $\mathcal{N}(y;\ \sqrt{\bar\alpha_{t-1}}\,x_0,\ 1 - \bar\alpha_{t-1})$. Pour 2, le carré de $-\sqrt{\alpha_t}(\dots)$ fait sortir un facteur $\alpha_t$ que tu passes au dénominateur. Pour 3, écris $\frac{1}{\tilde\beta_t} = \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar\alpha_{t-1}}$ au même dénominateur. Pour 4, écris $\tilde\mu_t = \tilde\beta_t\big(\frac{\alpha_t}{\beta_t}\cdot\frac{x_t}{\sqrt{\alpha_t}} + \frac{\sqrt{\bar\alpha_{t-1}}\,x_0}{1 - \bar\alpha_{t-1}}\big)$ et remplace $\tilde\beta_t$. Pour 5, mets $\bar\alpha_0 = 1$, compare $1 - \bar\alpha_{t-1}$ à $1 - \bar\alpha_t$, puis utilise l'identité de l'indice 2. Pour 6, prends $\bar\alpha_1 = 0{,}95$, $\bar\alpha_2 = 0{,}8075$, $\beta_2 = 0{,}15$. Pour 7, regroupe les termes en $x_t$ et en $\varepsilon$ séparément.

</details>

### Ex B5.8 — Fermi : coût d'échantillonnage, de MNIST sur CPU à une image 1024×1024, et ce que gagnent DDIM et l'espace latent 🧮

<details><summary>Indice 1</summary>

Relis « Ce que coûte un échantillon » (§205.5), l'encadré 💼 de 205.6, « La diffusion latente » et « Les transformers de diffusion » (§205.9), et la règle « 2 FLOP par paramètre et par token » de B3.

</details>
<details><summary>Indice 2</summary>

Un coût d'échantillonnage = nombre de passes du réseau × coût d'une passe ; nombre de passes = images (ou batches) × pas × (2 avec la guidance). Pour le transformer : tokens = (côté du latent / taille du patch)², FLOP d'une passe ≈ $2 \times N \times$ tokens, temps = FLOP / débit.

</details>
<details><summary>Indice 3</summary>

1. $10\,000 / 64 \approx 157$ batches ; en DDPM, $157 \times 1\,000 \times 0{,}13 \approx 20\,000$ s, soit environ 5,7 h. Pour le reste de 1, remplace 1 000 pas par 50, puis double. Pour 2, divise $64 \times 2 \times 10^8$ FLOP par 0,13 s. Pour 3, $1\,024^2 \times 3$ valeurs ; le latent fait $(1\,024/8)^2 \times 16$ ; les tokens, $(128/2)^2$. Pour 4, $2 \times 8 \times 10^9 \times$ tokens, puis $\times 28 \times 2$, divisé par chaque débit. Pour 5, refais 4 avec 2 000 passes, puis avec 4. Pour 6, $(1\,024/8)^2$ tokens, à comparer à 3 ; élève le rapport au carré pour l'attention. Pour 7, pense à ce que tu as négligé dans les hypothèses, et au reste du service.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B5.9 — La diffusion expliquée en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis 205.1 en entier, puis cherche une image de la vie courante pour « défaire un petit pas ».

</details>
<details><summary>Indice 2</summary>

Une ligne par point de l'énoncé, sans aucun mot technique : « bruit » plutôt que « gaussienne », « réseau qui devine » plutôt que « prédiction de ε ». Des images possibles : une photo qu'on fait vieillir grain après grain, un sculpteur qui retire la matière, une goutte d'encre qui se dilue dans l'eau et qu'on rembobinerait.

</details>
<details><summary>Indice 3</summary>

Commence par : « On prend des milliers de photos et on les abîme petit à petit, en ajoutant à chaque fois un peu de “neige”, comme sur une vieille télévision, jusqu'à ce qu'il ne reste que de la neige. » Puis : ce que le programme apprend à chaque étape, comment il fabrique une image à partir de neige pure, pourquoi les petites étapes, ce que cela coûte.

</details>

### Ex B5.10 — Un générateur peut-il recopier ses images d'entraînement ? Mémorisation, droits et traçabilité ⚖️

<details><summary>Indice 1</summary>

Relis « La mémorisation » (§205.8) et « Transparence et provenance des contenus générés » (§205.9), puis les données personnelles et les licences au ch. 29, et ⚖️ B4.10 sur la fouille de textes.

</details>
<details><summary>Indice 2</summary>

Prends les trois sources une par une avec trois questions : qu'est-ce que le modèle risque de recopier (petit volume, répétitions, 40 epochs, duplications) ; quel droit protège ce contenu (données personnelles, droit d'auteur, marques) ; que faut-il dire et marquer à la sortie (AI Act, art. 50 ; C2PA, filigrane). Sur la fouille de textes, la question est la forme de la réserve : un texte en toutes lettres est-il « lisible par machine » ?

</details>
<details><summary>Indice 3</summary>

1. Les portraits et les logos : peu d'images, vues 40 fois, souvent dupliquées (un même logo sur cent photos) ; c'est exactement ce que Carlini et coll. trouvent le plus mémorisé. Pour la suite de 1, pense aux plus proches voisins et au seuil relatif. Pour 2, compare la finalité de l'accord (une brochure) à l'usage (entraîner un modèle commercial). Pour 3, la cour de Hambourg s'est prononcée sur une réserve en langage naturel ; Getty contre Stability porte sur le modèle comme « copie contrefaisante » et sur les filigranes ; GEMA contre OpenAI, sur la mémorisation dans les poids. Pour 4, distingue fournisseur (marquage) et déployeur (hypertrucage). Pour 5, demande-toi ce que garde un fichier recadré ou capturé à l'écran. Pour 6, reprends sources, statut, exclusions et tests de recopie.

</details>

### Ex B5.11 — Ho, Jain et Abbeel (2020) : « Denoising Diffusion Probabilistic Models » 📄

<details><summary>Indice 1</summary>

Ouvre la version 2 de l'article sur arXiv ; lis le résumé, la figure 2, les sections 2 à 4.3 et l'annexe B, avec la fiche à côté : ses notations sont celles de l'article.

</details>
<details><summary>Indice 2</summary>

Repère dans l'article chaque formule de la fiche : (4) forme fermée, (6)-(7) posterior, (11) moyenne paramétrée par $\boldsymbol{\varepsilon}$, (12) le poids de chaque pas, (14) $L_{\text{simple}}$, (15) $\hat{\mathbf{x}}_0$. Pour les chiffres, le tableau 1 (FID, IS, NLL), le tableau 2 (ablation), l'annexe B (réglages et temps).

</details>
<details><summary>Indice 3</summary>

1. FID de 3,17 et Inception Score de 9,46 sans condition, mieux que la plupart des modèles publiés, y compris conditionnels ; mais une log-vraisemblance (≤ 3,75 bits par dimension) moins bonne que celle d'autres modèles à vraisemblance. Pour 4, le poids au pas 1 se calcule avec $\beta_1 = 10^{-4}$, $\alpha_1 = 1 - 10^{-4}$ et $1 - \bar\alpha_1 = 10^{-4}$ ; au pas 1 000, $1 - \bar\alpha_T \approx 1$. Pour 7, divise $800\,000 \times 128$ par 50 000, puis 800 000 pas par 21 pas par seconde, puis 17 s par 1 000 passes. Pour 8, cherche « nearest neighbor » dans le texte et les figures de l'annexe D. Pour les autres questions, une ligne de l'article suffit à chaque fois.

</details>

<a id="entretien"></a>

## 💼 Entretien

### B5.E1 — Explique le principe d'un modèle de diffusion, de l'entraînement à la génération

<details><summary>Indice 1</summary>

Relis « L'essentiel en 10 lignes », puis 205.3 (algorithme 1) et 205.5 (algorithme 2).

</details>
<details><summary>Indice 2</summary>

Trois temps : le processus direct (fixé, forme fermée), l'entraînement (une MSE sur le bruit, à un pas tiré au hasard), la génération (partir du bruit pur et débruiter pas à pas). Ajoute ce qu'on en fait en pratique : DDIM, guidance, espace latent.

</details>
<details><summary>Indice 3</summary>

Commence par : « Un modèle de diffusion apprend à défaire un petit pas de bruit. » Puis : on bruite les images avec un planning connu, et la forme fermée permet de sauter directement à n'importe quel pas ; le réseau, un U-Net qui reçoit aussi le pas, prédit le bruit ajouté ; pour générer, on part d'un bruit pur et on applique le pas inverse ; en production, on utilise moins de pas et on travaille dans un espace latent.

</details>

### B5.E2 — Pourquoi le réseau prédit-il le bruit plutôt que l'image ?

<details><summary>Indice 1</summary>

Relis « Prédire le bruit », « Le lien avec l'ELBO » et l'encadré 🕰️ sur les paramétrisations (§205.3).

</details>
<details><summary>Indice 2</summary>

Trois arguments : l'équivalence (connaissant $\mathbf{x}_t$ et $t$, prédire $\boldsymbol{\varepsilon}$, $\mathbf{x}_0$ ou $\mathbf{v}$ revient au même) ; l'apprentissage (une cible de même échelle à tous les pas) ; l'expérience (Ho et coll. trouvent de meilleurs échantillons). Puis les limites : la division par $\sqrt{\bar\alpha_t}$ aux grands $t$, d'où la $\mathbf{v}$-prédiction.

</details>
<details><summary>Indice 3</summary>

Commence par : « C'est un choix de paramétrisation, pas une obligation : connaissant x_t et t, le bruit et l'image se déduisent l'un de l'autre par la forme fermée. » Puis : pourquoi le bruit est une cible commode, ce qu'a montré l'ablation de l'article, et ce que l'on prédit aussi aujourd'hui.

</details>

### B5.E3 — Comment accélérer l'échantillonnage d'un modèle de diffusion ?

<details><summary>Indice 1</summary>

Relis 205.6 en entier (DDIM, η, distillation, l'encadré 💼) et « La diffusion latente » (§205.9).

</details>
<details><summary>Indice 2</summary>

Le coût = pas × coût d'une passe × images. Les leviers, du moins cher au plus cher : un meilleur échantillonneur sans réentraîner, la guidance en un seul batch, la précision réduite, un modèle distillé publié, une distillation maison ; et pour un nouveau projet, l'espace latent. Le risque : la qualité, à mesurer à chaque réglage.

</details>
<details><summary>Indice 3</summary>

Commence par : « D'abord, je mesure où part le temps : nombre de pas, coût d'une passe, guidance. » Puis : passer de 50 à 20 pas avec un meilleur échantillonneur, en mesurant une distance de Fréchet et un jugement humain sur un jeu fixe de prompts ; puis un modèle distillé en 1 à 4 pas ; et dire ce qu'on perd.

</details>

### B5.E4 — Qu'est-ce que la guidance sans classifieur et quel compromis règle-t-elle ?

<details><summary>Indice 1</summary>

Relis toute la section 205.7, figure et piège classique compris.

</details>
<details><summary>Indice 2</summary>

Quatre points : l'entraînement (la condition remplacée par ∅ une fois sur dix environ) ; l'échantillonnage (deux prédictions, extrapolées) ; l'intuition (un classifieur implicite) ; le compromis (fidélité contre diversité) et son prix (deux passes par pas). Puis la convention de `guidance_scale`.

</details>
<details><summary>Indice 3</summary>

Commence par : « On entraîne un seul réseau à prédire le bruit avec et sans la condition, en masquant la condition une fois sur dix. » Puis : la formule avec $w$, ce que fait une petite et une grande valeur, comment on la choisit (une courbe de qualité et de diversité sur un jeu de prompts), et le piège des deux conventions.

</details>

### B5.E5 — Diffusion ou flow matching : qu'est-ce qui a changé dans les générateurs récents ?

<details><summary>Indice 1</summary>

Relis « Flow matching et rectified flow », l'encadré 🧮 sur Euler et « Du texte à l'image, et au-delà » (§205.9).

</details>
<details><summary>Indice 2</summary>

Ce qui change : le chemin entre la donnée et le bruit (une ligne droite), la cible (une vitesse), l'échantillonneur (Euler sur un champ de vitesses), et l'architecture (un transformer dans un espace latent). Ce qui reste : un réseau qui débruite, une MSE, un bruit de départ, la guidance, la distillation.

</details>
<details><summary>Indice 3</summary>

Commence par : « Le rectified flow est un cousin très proche de la diffusion : on relie chaque image à un bruit par une ligne droite, et le réseau apprend la vitesse le long de cette ligne. » Puis : pourquoi des trajectoires droites demandent moins de pas, ce que SD3 change en plus (tirage de $s$, MM-DiT), et ce qui n'a pas changé.

</details>

<a id="notebook"></a>

## Exercices du notebook

Les indices des exercices du notebook (B5.12 à B5.31) seront ajoutés à la prochaine session de génération, avec le notebook.
