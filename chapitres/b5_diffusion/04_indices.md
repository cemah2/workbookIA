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

a) **Faux** : DDIM réutilise le même réseau, entraîné une fois avec ses 1 000 pas. Pour b), y a-t-il un tirage quand $\eta = 0$ ? Pour c), que vaut $\sigma^2$ quand $\eta = 1$ et $t' = t - 1$ ? Pour d), relis l'étape 2 de DDIM et le cas $t' = 0$. Pour e), relis « Pourquoi c'est plus rapide » : de quoi dépend le nombre de pas qu'il faut pour suivre une trajectoire ?

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

a) $0{,}25 / \sqrt{1 - 0{,}605625} = 0{,}25 / 0{,}62799 =$ **0,3981**. Pour b), calcule $(0{,}8 - a) \times 0{,}5) / \sqrt{0{,}75}$. Pour c), calcule $\frac{1 - 0{,}8075}{1 - 0{,}605625} \times 0{,}25$. Pour d) et e), ajoute $\sigma \times (-1)$ à b), avec $\sigma = \sqrt{c)}$, puis $\sigma = \sqrt{\beta_3}$. Pour f), calcule $\hat{x}_0$, écrête-le, puis $\frac{\sqrt{\bar\alpha_2}\,\beta_3}{1 - \bar\alpha_3}\,\hat{x}_0 + \frac{\sqrt{\alpha_3}(1 - \bar\alpha_2)}{1 - \bar\alpha_3}\,x_3$, avec des coefficients gardés à cinq décimales. Pour g), la formule de b) avec les constantes de l'indice 0, sans $z$. Pour h), relis la docstring de `ddpm_step`. Pour i), où la variance entre-t-elle dans tes calculs ?

</details>

### Ex B5.5 — Un pas DDIM à la main et l'image estimée x̂_0 ✏️

<details><summary>Indice 1</summary>

Relis « L'idée de DDIM » et « Le paramètre η » (§205.6), et la docstring de `ddim_step`. Constantes du planning de ✏️ B5.2.

</details>
<details><summary>Indice 2</summary>

$\hat{x}_0 = (x_4 - \sqrt{1 - \bar\alpha_4}\,\varepsilon_\theta)/\sqrt{\bar\alpha_4}$, puis $x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat{x}_0 + \sqrt{1 - \bar\alpha_{t'} - \sigma^2}\,\varepsilon_\theta + \sigma z$, avec $\sigma = \eta\sqrt{\frac{1 - \bar\alpha_{t'}}{1 - \bar\alpha_t}}\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t'}}}$. Le pas $t$ de l'article est l'indice $t - 1$ ; l'indice $-1$ est l'image propre.

</details>
<details><summary>Indice 3</summary>

a) $(-0{,}5 - 0{,}77868 \times (-1{,}2)) / 0{,}62742 =$ **0,6924**. Pour b), calcule $\sqrt{\bar\alpha_1}\,\hat{x}_0 + \sqrt{1 - \bar\alpha_1}\,\varepsilon_\theta$. Pour c), traduis « pas 4 » et « pas 1 » en indices. Pour d), la même formule avec $\bar\alpha_2$. Pour e), recalcule $\hat{x}_0$ à partir de d) et du même $\varepsilon_\theta$, puis fais le pas. Pour f) et g), applique la formule de $\sigma$ avec $\eta = 1$, $t = 4$ et $t' = 3$, puis $t' = 1$. Pour h), calcule $\sqrt{1 - 0{,}95 - \sigma^2}$ avec le $\sigma^2$ de g) gardé à six décimales : les deux termes sont proches, et la soustraction efface des chiffres. Pour i), compare g) et h).

</details>

### Ex B5.6 — Forme fermée : deux pas de bruit gaussien n'en font qu'un, d'où q(x_t | x_0) ∂

<details><summary>Indice 1</summary>

Relis « Un pas de bruit », l'encadré 🧮 sur la somme de deux gaussiennes indépendantes et « La forme fermée » (§205.2), et le rappel sur la récurrence de l'énoncé.

</details>
<details><summary>Indice 2</summary>

Deux règles suffisent : $aX \sim \mathcal{N}(0, a^2)$ si $X \sim \mathcal{N}(0, 1)$, et deux gaussiennes **indépendantes** s'additionnent en additionnant leurs variances. Pour l'indépendance de la question 3, sers-toi du fait admis dans l'énoncé : de quels bruits $\bar\varepsilon_{t-1}$ est-il fait ? Pour l'hérédité, remplace $x_{t-1}$ par l'hypothèse dans $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1 - \alpha_t}\,\varepsilon_t$, puis regroupe le terme en $x_0$ et les deux bruits. L'identité clé : $\alpha_t(1 - \bar\alpha_{t-1}) + 1 - \alpha_t = 1 - \bar\alpha_t$.

</details>
<details><summary>Indice 3</summary>

1. $x_1 = \sqrt{\alpha_1}\,x_0 + \sqrt{1 - \alpha_1}\,\varepsilon_1 \sim \mathcal{N}(\sqrt{\alpha_1}\,x_0,\ 1 - \alpha_1)$, et $\bar\alpha_1 = \alpha_1$ : c'est la forme fermée pour $t = 1$. Pour 2, $x_2 = \sqrt{\alpha_2\alpha_1}\,x_0 + \sqrt{\alpha_2(1 - \alpha_1)}\,\varepsilon_1 + \sqrt{1 - \alpha_2}\,\varepsilon_2$ : additionne les variances des deux derniers termes. Pour 3, $\bar\varepsilon_{t-1}$ est une combinaison de quels bruits ? Pour 4, calcule $0{,}85 \times 0{,}05 + 0{,}15$ et $1 - 0{,}8075$. Pour 5, calcule $\bar\alpha_t \operatorname{Var}(x_0) + 1 - \bar\alpha_t$. Pour 6, additionne les variances $\beta_t$. Pour 7, refais la question 3 en partant de $x_s$, puis cherche $1 - \bar\alpha_t/\bar\alpha_{t'}$ dans la formule de $\sigma$ de DDIM.

</details>

### Ex B5.7 — Moyenne et variance du posterior q(x_{t−1} | x_t, x_0) en dimension 1 ∂

<details><summary>Indice 1</summary>

Relis la règle de Bayes (ch. 4), « Le posterior » et l'encadré 🧮 sur le produit de deux densités gaussiennes (§205.3).

</details>
<details><summary>Indice 2</summary>

Pour la question 1, la propriété de Markov admise dans l'énoncé suffit : écris $x_t$ en fonction de $y$ et de $\varepsilon_t$. Les deux facteurs, vus comme des fonctions de $y$, sont des gaussiennes : $\mathcal{N}(y;\ x_t/\sqrt{\alpha_t},\ \beta_t/\alpha_t)$ et $\mathcal{N}(y;\ \sqrt{\bar\alpha_{t-1}}\,x_0,\ 1 - \bar\alpha_{t-1})$. Les précisions s'additionnent, la moyenne est la moyenne des moyennes pondérée par les précisions. L'identité $\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t = 1 - \bar\alpha_t$ simplifie tout.

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

1. Les logos et les portraits : un même logo revient sur des dizaines de photos, des portraits pris en séries de poses voisines sont presque des doublons, et 40 epochs multiplient les passages ; c'est exactement ce que Carlini et coll. trouvent le plus mémorisé. Pour la suite de 1, pense aux plus proches voisins et au seuil relatif. Pour 2, compare la finalité de l'accord (une brochure) à l'usage (entraîner un modèle commercial). Pour 3, la cour de Hambourg s'est prononcée sur une réserve en langage naturel ; Getty contre Stability porte sur le modèle comme « copie contrefaisante » et sur les filigranes ; GEMA contre OpenAI, sur la mémorisation dans les poids. Pour 4, distingue fournisseur (marquage) et déployeur (deepfake). Pour 5, demande-toi ce que garde un fichier recadré ou capturé à l'écran. Pour 6, reprends sources, statut, exclusions et tests de recopie.

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

### Ex B5.12 — linear_beta_schedule, cosine_beta_schedule et noise_schedule 🔨

<details><summary>Indice 1</summary>

Relis « Le planning linéaire de Ho et coll. », « Le planning cosinus de Nichol et Dhariwal » et « Bruiter un batch » (§205.2), et tes calculs de ✏️ B5.2 : un planning est un tableau de $T$ valeurs ; toutes les constantes s'en déduisent par des opérations élément par élément et **un** produit cumulé.

</details>
<details><summary>Indice 2</summary>

- `linear_beta_schedule` : valide d'abord, puis `np.linspace(beta_start, beta_end, n_steps, dtype=np.float64)`, qui rend `[beta_start]` pour un seul pas.
- `cosine_beta_schedule` : `t = np.arange(n_steps + 1)`, `f = np.cos((t / n_steps + s) / (1 + s) * np.pi / 2) ** 2`, puis le rapport de deux valeurs consécutives, `f[1:] / f[:-1]`, et `np.minimum(…, max_beta)`.
- `noise_schedule` : `np.asarray(betas, dtype=np.float64)`, la validation, `alphas = 1 - betas`, `alpha_bars = np.cumprod(alphas)`, `alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])`, puis les racines et les trois constantes du posterior d'après la docstring.
- Pour refuser `nan`, écris la condition **qui doit être vraie**, `np.all((b > 0) & (b < 1))`, et lève l'erreur si elle ne l'est pas : `nan > 0` est faux, donc `nan` est refusé. La condition inverse, « un $\beta$ est $\le 0$ ou $\ge 1$ », est fausse elle aussi pour `nan` et le laisse passer. De même pour `beta_start`, `beta_end` et `max_beta`.

</details>
<details><summary>Indice 3</summary>

```python
def noise_schedule(betas):
    b = np.asarray(betas, dtype=np.float64)
    if b.ndim != 1 or not np.all((b > 0) & (b < 1)):
        raise ValueError(...)
    alphas = 1.0 - b
    alpha_bars = np.cumprod(alphas)
    alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])
    ...                                                   # the two square roots and the three posterior constants
    return {"betas": b, ...}                              # the nine keys of the docstring
```

</details>

### Ex B5.13 — Prédire l'allure d'un chiffre bruité aux pas 100, 300 et 600, planning linéaire puis cosinus 🔮

<details><summary>Indice 1</summary>

Calcule d'abord les six SNR, avec la formule de §205.2 et tes $\bar\alpha$ (ou la méthode de ✏️ B5.3) : les pas 100, 300 et 600 de l'article sont les indices 99, 299 et 599. Puis demande-toi ce que le classifieur a vu pendant son entraînement : des chiffres **propres**, rien d'autre.

</details>
<details><summary>Indice 2</summary>

Un SNR de 1 veut dire que la variance du bruit égale celle de l'image ; un SNR de 0,1, que le bruit en a dix fois plus. Ta convention de ✏️ B5.3 dit ce que ton œil, qui a vu des milliers d'images bruitées, devine encore. Un réseau entraîné sans bruit n'a jamais vu de grain : il réagit à des motifs qu'il n'a pas appris. Classe les six cas du SNR le plus grand au plus petit, et place la frontière où, selon toi, un tel réseau lâche.

</details>
<details><summary>Indice 3</summary>

Pour a), range les six cas par SNR décroissant (le premier est le cosinus au pas 100, SNR ≈ 35 ; calcule les cinq autres) et place une seule frontière : au-dessus de quel SNR un réseau qui n'a vu que des chiffres propres en reconnaît-il encore 70 % ? Ta prédiction est le nombre de cas au-dessus. Pour b), compare le SNR du cosinus au pas 600 à 1 (bruit et image de même variance), pas seulement à 0,1.

</details>

### Ex B5.14 — q_sample : bruiter un batch d'images à des pas différents 🔨

<details><summary>Indice 1</summary>

Relis « La forme fermée » et « Bruiter un batch » (§205.2), avec le piège « `alpha_bars[t]` sans remise en forme » : la formule est une ligne ; le travail est de choisir le bon coefficient pour chaque exemple et de lui donner la forme `(N, 1, …, 1)`.

</details>
<details><summary>Indice 2</summary>

- Convertis `x0`, `t` et `alpha_bars` en tableaux (`np.asarray(x0, dtype=float)`, `np.asarray(t)`), sans jamais modifier ceux que tu reçois.
- Vérifie que `t` est 1-D de longueur `x0.shape[0]`, d'entiers, et dans $[0, T)$ : `ValueError` sinon (teste-le toi-même avant d'indexer, sinon `alpha_bars[-1]` passerait sans erreur et `alpha_bars[T]` lèverait `IndexError`).
- Si `noise` est donné, vérifie `noise.shape == x0.shape` ; sinon, `rng = np.random.default_rng() if rng is None else rng`, puis **un** tirage `rng.standard_normal(x0.shape)`.
- La forme du coefficient : `(N,) + (1,) * (x0.ndim - 1)`.

</details>
<details><summary>Indice 3</summary>

```python
def q_sample(x0, t, alpha_bars, noise=None, rng=None):
    x0 = np.asarray(x0, dtype=float)
    t = np.asarray(t)
    ...                                                   # t: 1-D, one integer index per sample, in [0, T)
    ...                                                   # noise: given (same shape as x0) or drawn once from rng
    shape = (len(x0),) + (1,) * (x0.ndim - 1)
    a = np.sqrt(alpha_bars[t]).reshape(shape)
    return a * x0 + ..., noise
```

</details>

### Ex B5.15 — Lire ᾱ_t, le rapport signal sur bruit et la grille de bruitage des deux plannings 📈

<details><summary>Indice 1</summary>

L'axe horizontal est la fraction de la chaîne, pas l'indice : c'est ce qui permet de comparer des plannings de $T$ différents. Sur le graphique de droite, l'axe vertical est un logarithme : une différence de 1 y vaut un facteur $e \approx 2{,}7$ sur le SNR.

</details>
<details><summary>Indice 2</summary>

- a) Suis la courbe L à gauche, puis les deux courbes en pointillés : laquelle se confond avec elle ? Relie ta réponse à ✏️ B5.2 : que vaut $\sum \beta$ pour chacune ?
- b) La fin de chaque courbe, au bord droit : le $\ln \operatorname{SNR}$ du dernier indice dit quelle part de l'image il reste. Le graphique de gauche les écrase toutes près de 0 ; celui de droite les sépare.
- c) Repère où les courbes C et L se croisent, s'il y en a un, en regardant jusqu'au bord droit.

</details>
<details><summary>Indice 3</summary>

a) **S** : sa courbe (en pointillés fins) reste collée à L de bout en bout, alors que D s'en écarte dès le premier tiers ; c'est ✏️ B5.2 : avec les bornes multipliées par $1\,000/T$, la somme des $\beta$ jusqu'à une même fraction de la chaîne est la même, donc $\bar\alpha \approx e^{-\sum\beta}$ aussi. Pour b), compare les quatre valeurs de $\ln \operatorname{SNR}$ au bord droit, la plus haute garde le plus d'image. Pour c), situe le croisement de C et L sur l'axe horizontal et compare cette fraction à 0,5, 0,9 et 0,95.

</details>

### Ex B5.16 — q_posterior_mean_variance et predict_x0_from_eps 🔨

<details><summary>Indice 1</summary>

Relis « Le posterior : un pas en arrière quand on connaît $\mathbf{x}_0$ » (§205.3) et ta dérivation ∂ B5.7 : `noise_schedule` a déjà calculé les deux coefficients de la moyenne et la variance ; il reste à les lire à l'indice de chaque exemple, comme dans `q_sample`. `predict_x0_from_eps` inverse la forme fermée.

</details>
<details><summary>Indice 2</summary>

- Les mêmes validations que `q_sample` (forme de `t`, entiers, intervalle), plus `x_t.shape == x0.shape` (et `eps.shape == x_t.shape`) **exactement**.
- `coef1 = schedule["posterior_mean_coef1"][t].reshape(shape)`, de même pour `coef2` et `posterior_variance`, avec `shape = (N,) + (1,) * (x0.ndim - 1)` : la variance a donc autant d'axes que `x0` (forme `(N,)` en 1-D).
- `predict_x0_from_eps` : `(x_t - sqrt(1 - ab[t]) * eps) / sqrt(ab[t])`, coefficients remis en forme.

</details>
<details><summary>Indice 3</summary>

```python
def q_posterior_mean_variance(x0, x_t, t, schedule):
    x0, x_t, t = np.asarray(x0, dtype=float), np.asarray(x_t, dtype=float), np.asarray(t)
    ...                                                   # validations: t as in q_sample, x_t.shape == x0.shape
    shape = (len(x0),) + (1,) * (x0.ndim - 1)
    mean = schedule["posterior_mean_coef1"][t].reshape(shape) * x0 + ...
    return mean, schedule["posterior_variance"][t].reshape(shape)
```

</details>

### Ex B5.17 — ddpm_step et ddim_step 🔨

<details><summary>Indice 1</summary>

Relis « L'algorithme » et « Un pas à la main » (§205.5) pour DDPM, « L'idée de DDIM » et « Le paramètre η » (§205.6) pour DDIM, et refais de tête ✏️ B5.4 et B5.5 : chaque fonction est une moyenne, plus un bruit dont il faut bien choisir l'écart-type, et deux ou trois cas particuliers (le dernier pas, `t_prev = -1`, `eta = 0`).

</details>
<details><summary>Indice 2</summary>

- `ddpm_step` : valide `t` (un entier de $[0, T)$) et `variance` **au début**, avant tout calcul. Sans écrêtage, la moyenne de l'algorithme 2 ; avec `clip_x0`, $\hat{\mathbf{x}}_0$ (ta `predict_x0_from_eps`, ou la même formule), `np.clip(x0_hat, -clip_x0, clip_x0)`, puis la moyenne du posterior avec les coefficients de `schedule` à l'indice `t`. Si `t > 0` : `rng = np.random.default_rng() if rng is None else rng`, `z = rng.standard_normal(x_t.shape)`, et ajoute `np.sqrt(var) * z`.
- `ddim_step` : valide, puis `ab = alpha_bars[t]` et `ab_prev = 1.0 if t_prev == -1 else alpha_bars[t_prev]` ; $\hat{\mathbf{x}}_0$, puis $\sigma$, puis la formule de la docstring. Ne tire $\mathbf{z}$ que si `eta > 0` (et alors `t_prev = -1` donne $\sigma = 0$ : rien à ajouter). Pour refuser `eta = nan`, la condition qui doit être vraie : `0 <= eta <= 1`.

</details>
<details><summary>Indice 3</summary>

```python
def ddim_step(x_t, t, t_prev, eps_pred, alpha_bars, eta=0.0, rng=None):
    x_t, eps = np.asarray(x_t, dtype=float), np.asarray(eps_pred, dtype=float)
    ...                                                   # -1 <= t_prev < t < T, 0 <= eta <= 1 (nan refused)
    ab, ab_prev = alpha_bars[t], (1.0 if t_prev == -1 else alpha_bars[t_prev])
    x0_hat = (x_t - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
    sigma = eta * np.sqrt((1 - ab_prev) / (1 - ab)) * np.sqrt(1 - ab / ab_prev)
    out = np.sqrt(ab_prev) * x0_hat + np.sqrt(1 - ab_prev - sigma ** 2) * eps
    ...                                                   # + sigma * z, drawn only if eta > 0
```

</details>

### Ex B5.18 — Avec le vrai bruit, DDPM et DDIM retrouvent-ils x_0 ? Contrôle sur des données 2D 🔬

<details><summary>Indice 1</summary>

« Le vrai bruit » d'un point $\mathbf{x}_t$ est celui qui, avec $\mathbf{x}_0$, redonne $\mathbf{x}_t$ par la forme fermée : on l'obtient en l'inversant. Les deux chaînes sont des boucles qui appellent tes pas de B5.17 ; tout le reste est une question de « quel bruit donner à chaque pas ».

</details>
<details><summary>Indice 2</summary>

- `true_noise_18` : la forme fermée résolue en $\boldsymbol{\varepsilon}$, avec $\bar\alpha_t$ = `alpha_bars[t]` (un nombre : pas de remise en forme).
- `ddpm_chain_18` : $T$ = `len(schedule["alpha_bars"])` ; le point de départ à l'indice $T - 1$ ; une boucle `for t in range(T - 1, -1, -1)` qui choisit `eps_pred` selon `oracle`, puis appelle `mylearn.diffusion.ddpm_step(x, t, eps_pred, schedule, rng=rng)`.
- `ddim_chain_18` : la liste des indices de `timesteps`, suivie de −1 ; on parcourt les couples (indice, indice suivant) avec `zip(steps[:-1], steps[1:])`.
- Pour les réponses, compare chaque écart du tableau à $10^{-6}$.

</details>
<details><summary>Indice 3</summary>

```python
def ddim_chain_18(x0, eps, alpha_bars, timesteps, eta=0.0, rng=None):
    steps = [int(t) for t in timesteps] + [-1]
    a0 = float(alpha_bars[steps[0]])
    x = np.sqrt(a0) * np.asarray(x0, dtype=float) + np.sqrt(1 - a0) * np.asarray(eps, dtype=float)
    for t, t_prev in zip(steps[:-1], steps[1:]):
        x = ...                                           # your ddim_step, with eps_pred = eps
    return x
```

a) se lit sur la première ligne du tableau : un écart de plusieurs centaines, bien au-delà de $10^{-6}$, donc `False`. Fais de même pour les lignes 2, 4 et 5 : un écart de l'ordre de $10^{-14}$ donne `True`.

</details>

### Ex B5.19 — Diffusion en 2D : un MLP débruiteur conditionné par le pas de temps, entraîné puis échantillonné 📦

<details><summary>Indice 1</summary>

Relis « Dire au réseau quel est le pas » (§205.4) et « La loss simplifiée » (§205.3) : le réseau reçoit $\mathbf{x}_t$ et l'encodage de $t$, et la loss compare sa sortie au bruit qui a servi à fabriquer $\mathbf{x}_t$. L'échantillonnage est l'algorithme 2 : ta `ddpm_step` fait tout, il suffit de lui donner la prédiction du réseau à chaque pas.

</details>
<details><summary>Indice 2</summary>

- Le modèle : `self.register_buffer("pe", torch.tensor(mylearn.attention.positional_encoding(n_positions, t_dim), dtype=torch.float32), persistent=False)`, le `nn.Sequential` du contrat, et `torch.cat([x, self.pe[t]], dim=1)` dans `forward`.
- L'entraînement : `alpha_bars` converti une fois en tenseur float32 ; à chaque itération les trois tirages **dans l'ordre du contrat**, chacun avec `generator=generator` ; `a = ab[t][:, None]` (forme `(B, 1)`, pour la diffusion sur les deux coordonnées) ; `loss.item()` dans la liste.
- L'échantillonnage : sous `torch.no_grad()`, convertis $\mathbf{x}$ en float32 pour le réseau et sa sortie en float64 NumPy (`.double().numpy()`) pour `ddpm_step` ; `torch.full((n,), t, dtype=torch.long)` donne le même pas à tous les points.

</details>
<details><summary>Indice 3</summary>

```python
def train_19(model, data, alpha_bars, n_iters, batch_size, lr, generator):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ab = torch.tensor(np.asarray(alpha_bars), dtype=torch.float32)
    losses = []
    for _ in range(n_iters):
        x0 = data[torch.randint(len(data), (batch_size,), generator=generator)]
        ...                                               # t, then eps, both from generator
        loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), eps)
        ...                                               # zero_grad, backward, step, losses.append(loss.item())
    return losses
```

</details>

### Ex B5.20 — Flow matching en 2D : même MLP, cible « vitesse » et intégration d'Euler 🔬

<details><summary>Indice 1</summary>

Relis « Flow matching et rectified flow » (§205.9) et l'encadré 🧮 « la méthode d'Euler » : l'entraînement ne change que par l'entrée ($\mathbf{x}_s$ sur la ligne droite) et la cible (la vitesse) ; la génération suit la vitesse prédite à petits pas, de $s = 1$ à $s = 0$, donc **en reculant** : chaque pas retire `v / n_steps`.

</details>
<details><summary>Indice 2</summary>

- `train_fm_20` : la boucle de `train_19`, avec `u = torch.randint(1, 1001, (batch_size,), generator=generator)` à la place des pas, `s = (u.float() / 1000)[:, None]`, l'entrée `(1 - s) * x0 + s * eps`, la cible `eps - x0`, et `model(x_s, u)`.
- `euler_20` : la position du pas `k` est `round(1000 * (1 - k / n_steps))` (1 000 au premier pas, le bruit pur) ; sous `torch.no_grad()`.
- `ddim_20` : `wbd.ddim_timesteps(len(alpha_bars), n_steps)` suivi de −1, et la même boucle que `ddim_chain_18`, avec la prédiction du réseau à la place du vrai bruit.
- Pour a) et b), lis le tableau du protocole complet (les médianes) **et** les bandes du graphique : une réponse doit valoir pour chaque graine.

</details>
<details><summary>Indice 3</summary>

```python
def euler_20(model, n, n_steps, generator):
    x = torch.randn((n, 2), generator=generator)
    with torch.no_grad():
        for k in range(n_steps):
            u = round(1000 * (1 - k / n_steps))
            x = ...                                       # one Euler step backwards: minus v / n_steps
    return x.double().numpy()
```

Pour a), commence par 1 pas : pour chaque graine, compare sa distance à 1 pas au triple de sa distance à 100 pas ; si une seule graine dépasse, 1 pas ne convient pas ; passe alors à 2, puis à 5… Pour b), compare, graine par graine, les deux distances de la colonne des 5 pas.

</details>

### Ex B5.21 — Un U-Net minuscule en PyTorch : blocs convolutifs, connexions de saut, encodage du pas de temps 📦

<details><summary>Indice 1</summary>

Le bloc est écrit dans la fiche (§205.4, « Un bloc résiduel conditionné par le pas ») ; le U-Net suit la figure `unet.png` : deux étages qui descendent, un bloc au milieu, deux étages qui remontent, et une concaténation à chaque résolution. La seule nouveauté par rapport à la fiche : la classe, une table d'embeddings dont on ajoute la ligne à l'encodage du pas.

</details>
<details><summary>Indice 2</summary>

- Les noms du contrat, dans `__init__`, dans l'ordre où le signal passe : `t_mlp`, `class_emb` (seulement si `n_classes` n'est pas `None`), `inp`, `d1`, `down1`, `d2`, `down2`, `mid`, `up2`, `u2`, `up1`, `u1`, `out`.
- Le nombre de canaux à l'entrée de `u2` et `u1` est le double : la sortie de `up2` (`c2` canaux) **plus** `h2` (`c2` canaux), de même pour `u1`.
- La classe vide : `torch.full_like(t, self.n_classes)` quand `y` est `None`.
- Pour b), un *hook* de PyTorch lit l'entrée d'un module pendant un appel : `model.u1.register_forward_pre_hook(lambda mod, inp: print(inp[0].shape))` ; ou raisonne : à 28 × 28, combien de canaux sortent de `up1`, et combien en garde `h1` ?

</details>
<details><summary>Indice 3</summary>

```python
class TinyUNet_21(nn.Module):
    def __init__(self, n_steps, channels=(32, 64, 128), t_dim=128, n_classes=None):
        super().__init__()
        c1, c2, c3 = channels
        ...                                               # self.config, self.n_classes, the buffer pe, t_mlp, class_emb
        self.u2 = ResBlock_21(2 * c2, c2, t_dim)          # up2 output + h2
        ...

    def forward(self, x, t, y=None):
        temb = self.t_mlp(self.pe[t])
        ...                                               # + class_emb(y or the empty class), then the U
        h = self.u2(torch.cat([self.up2(h), h2], dim=1), temb)
```

</details>

### Ex B5.22 — Le DDPM qui ne produit que du gris : quatre bugs d'indices, de racines et d'échelle 🐛

<details><summary>Indice 1</summary>

Lis le message du premier point en échec, et lui seul : les points suivants n'ont de sens qu'une fois celui-ci réparé. Chacun mesure une propriété que la fiche énonce : l'échelle des images que supposent le bruit et l'écrêtage, la forme fermée (§205.2), l'aller-retour avec le vrai bruit (🔬 B5.18). Relis aussi le tableau des pièges de la fiche.

</details>
<details><summary>Indice 2</summary>

- Point 1 : un octet de 0 à 255 doit devenir un nombre de −1 à 1 ; que donne la division seule ?
- Point 2 : `t` est déjà un indice de 0 à $T - 1$ (relis les conventions de la fiche) ; que lit `alpha_bars[t - 1]` pour `t = 0` ?
- Point 3 : dans la forme fermée, les **carrés** des deux coefficients font 1.
- Point 4 : à quel indice l'algorithme 2 n'ajoute-t-il pas de bruit, et pourquoi ?
- Quand les quatre points passent, compare les images à celles du début : le débruiteur parfait ne montrait que le bug d'échelle.

</details>
<details><summary>Indice 3</summary>

Le point 1, réparé :
```python
def prepare_22(raw):
    return torch.as_tensor(np.asarray(raw), dtype=torch.float32)[:, None] / 127.5 - 1
```
Pour les trois autres, une ligne chacun : l'indice lu dans `alpha_bars` (point 2), le coefficient de `x0` (point 3), une condition sur `t` avant d'ajouter le bruit (point 4).

</details>

### Ex B5.23 — Entraîner un DDPM sur MNIST : boucle, moyenne exponentielle des poids (EMA), courbe de loss 📦

<details><summary>Indice 1</summary>

La boucle est l'algorithme 1 de la fiche (§205.3, « La loss simplifiée ») : tirer des pas et du bruit, bruiter par la forme fermée, régresser le bruit. Pour $\rho$ : après $N$ pas, la copie EMA garde $\rho^N$ des poids de départ (des poids aléatoires) et moyenne environ $1/(1 - \rho)$ pas ; cherche le plus grand $\rho$ qui respecte les deux conditions de l'énoncé, puis prends une valeur simple en dessous.

</details>
<details><summary>Indice 2</summary>

- `ema_update_23` : sous `torch.no_grad()`, `p_ema.mul_(rho).add_(p, alpha=1 - rho)` pour chaque couple de `zip(ema.parameters(), model.parameters())`, puis `b_ema.copy_(b)` pour les buffers ; jamais `p_ema = …` (un nouvel objet, que l'optimiseur et le modèle ne voient pas).
- `train_23` : `order = torch.randperm(len(images), generator=generator)` au début de **chaque** epoch, puis `order[i:i + bs]` pour `i` de 0 à `len(images)` par pas de `bs` ; dans chaque batch, `torch.randint` **puis** `torch.randn`, tous deux avec `generator` ; le coefficient `alpha_bars[t]` remis en forme `(b, 1, 1, 1)` (fiche, piège « `alpha_bars[t]` sans remise en forme ») ; l'EMA **après** `opt.step()`.
- La loss d'un pas : `loss.item()`, un float Python.
- Pour $\rho$ : $\rho^{N} \le 0{,}01$ donne $\rho \le 0{,}01^{1/N}$.

</details>
<details><summary>Indice 3</summary>

```python
def train_23(model, ema, images, settings, generator, alpha_bars):
    ab = torch.tensor(np.asarray(alpha_bars), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    losses, bs = [], settings["batch_size"]
    model.train()
    for _ in range(settings["epochs"]):
        order = torch.randperm(len(images), generator=generator)
        for i in range(0, len(images), bs):
            x0 = images[order[i:i + bs]]
            ...                                           # t, eps (in this order), x_t, the loss, one step, the EMA
    ...                                                   # eval mode for both, then the history
```
Avec $N = 1\,252$ pas en mode rapide, $0{,}01^{1/1252} \approx 0{,}9963$ : un $\rho$ juste en dessous respecte la première condition ; vérifie la seconde.

</details>

### Ex B5.24 — Échantillonner : DDPM à 1 000 pas contre DDIM à 50 pas, qualité et temps 📦

<details><summary>Indice 1</summary>

Tes deux pas existent déjà (🔨 B5.17) : il reste à les enchaîner avec la prédiction du réseau. DDPM parcourt tous les indices de $T - 1$ à 0 ; DDIM parcourt la sous-séquence, puis va vers −1. L'écrêtage de $\hat{\mathbf{x}}_0$ (fiche §205.5) évite les valeurs qui divergent quand le réseau se trompe aux grands indices.

</details>
<details><summary>Indice 2</summary>

- Une fonction d'aide qui rend la prédiction du réseau en float64 : `model(torch.as_tensor(x, dtype=torch.float32), torch.full((len(x),), t))` sous `torch.no_grad()`, puis `.double().numpy()`.
- DDPM : `x = mylearn.diffusion.ddpm_step(x, t, eps, schedule, rng=rng, clip_x0=clip_x0)` pour `t` de $T - 1$ à 0.
- DDIM : `steps = list(wbd.ddim_timesteps(T, n_steps)) + [-1]`, puis les couples `zip(steps[:-1], steps[1:])` ; $\hat{\mathbf{x}}_0$ par `predict_x0_from_eps(x, np.full(len(x), t), eps, alpha_bars)`.
- Le tenseur rendu : `torch.as_tensor(x, dtype=torch.float32)`.

</details>
<details><summary>Indice 3</summary>

```python
def sample_ddim_24(model, x_T, alpha_bars, n_steps, clip_x0=1.0):
    D = mylearn.diffusion
    x = x_T.double().numpy()
    steps = [int(t) for t in wbd.ddim_timesteps(len(alpha_bars), n_steps)] + [-1]
    for t, t_prev in zip(steps[:-1], steps[1:]):
        eps = ...                                         # the network's noise, float64
        if clip_x0 is not None:
            x0 = np.clip(D.predict_x0_from_eps(x, np.full(len(x), t), eps, alpha_bars), -clip_x0, clip_x0)
            eps = ...                                     # the noise that goes with the clipped x0
        x = D.ddim_step(x, t, t_prev, eps, alpha_bars)
    return torch.as_tensor(x, dtype=torch.float32)
```

</details>

### Ex B5.25 — Nombre de pas DDIM, planning linéaire ou cosinus : distance de Fréchet contre temps de calcul 🔬

<details><summary>Indice 1</summary>

Une courbe « qualité contre coût » se lit de gauche à droite : chaque point coûte plus cher que le précédent ; on cherche où elle cesse de descendre (fiche §205.8, « La distance de Fréchet »). Pour a) et b), lis le tableau `FULL_25` graine par graine, pas seulement les médianes.

</details>
<details><summary>Indice 2</summary>

- `fd_curve_25` : une boucle sur `steps_list`, `time.perf_counter()` avant et après `sample_ddim_24`, puis `frechet_mnist(images)` **hors** du chronomètre (on mesure le coût de l'échantillonnage, pas celui de la mesure).
- a) Pour chaque graine, divise chaque distance de la ligne L par la dernière (100 pas) ; cherche le premier nombre de pas à partir duquel tous les rapports restent sous le seuil ; la réponse vaut pour la graine la plus lente.
- b) Compare, graine par graine, la colonne des 5 pas des lignes L et C.

</details>
<details><summary>Indice 3</summary>

```python
def fd_curve_25(model, alpha_bars, steps_list, x_T):
    out = []
    for n in steps_list:
        start = time.perf_counter()
        images = sample_ddim_24(model, x_T, alpha_bars, n)
        out.append((frechet_mnist(images), time.perf_counter() - start))
    return out
```
Pour a), la première graine : divise ses cinq distances par la dernière, et regarde à partir de quelle colonne tous les rapports suivants restent sous le seuil ; fais de même pour les autres graines, puis prends le plus grand des nombres de pas trouvés.

</details>

### Ex B5.26 — Prédire l'effet de w = 0, 1, 3 et 7 sur la netteté et la diversité 🔮

<details><summary>Indice 1</summary>

Relis « La guidance sans classifieur » et la figure 2D de la fiche (§205.7) : la guidance extrapole la différence entre la prédiction conditionnée et la prédiction sans condition. Pense séparément à chacune des quatre mesures : la distance de Fréchet compare des nuages de features à ceux des vraies images (fidélité **et** diversité), la classe reconnue ne regarde que la fidélité.

</details>
<details><summary>Indice 2</summary>

- a) Un modèle petit, entraîné peu de temps : ses images sans guidance sont-elles déjà proches des vraies ? Une forte guidance resserre chaque classe : est-ce que cela rapproche ou éloigne le nuage de celui des vraies images ?
- b) La classe demandée n'est qu'une entrée de plus pour le réseau ; avec peu d'entraînement, la suit-il déjà vraiment ?
- c) La diversité mesure l'étalement des features dans chaque classe demandée. Une image ambiguë, entre un pull et un manteau, ou à demi formée, étale-t-elle moins ou plus les features qu'une vraie image de la classe ?

</details>
<details><summary>Indice 3</summary>

Pour a), écris ce que tu attends de la distance à $w = 0$ (sans guidance : des images de toutes les classes mêlées, un peu floues) puis à $w = 7$ (des images très typées de chaque classe, toutes semblables) : laquelle des deux erreurs, manque de fidélité ou manque de diversité, pèse le plus dans une distance entre gaussiennes de features ? Les deux autres prédictions se raisonnent de la même façon, sans calcul.

</details>

### Ex B5.27 — Diffusion conditionnelle et guidance sans classifieur sur Fashion-MNIST 📦

<details><summary>Indice 1</summary>

Rien de neuf dans la boucle : c'est celle de B5.23, avec un troisième tirage qui remplace parfois le label par la classe vide (fiche §205.7, « La guidance sans classifieur ») ; et à l'échantillonnage, ta `sample_ddim_24`, où le bruit du réseau devient le bruit guidé.

</details>
<details><summary>Indice 2</summary>

- `train_cond_27` : après `eps`, `drop = torch.rand(len(x0), generator=generator) < settings["p_uncond"]`, puis `y = torch.where(drop, torch.full_like(y, model.n_classes), y)` ; le modèle reçoit `(x_t, t, y)`.
- `guided_eps_27` : `torch.cat([x, x])`, `torch.cat([tt, tt])` et `torch.cat([y, torch.full_like(y, model.n_classes)])`, un appel, puis `.chunk(2)` : la première moitié est la prédiction conditionnée.
- `sample_cond_27` : copie `sample_ddim_24`, en remplaçant l'appel du réseau par `guided_eps_27(model, torch.as_tensor(x, dtype=torch.float32), t, y, w)`.

</details>
<details><summary>Indice 3</summary>

```python
def guided_eps_27(model, x, t, y, w):
    tt = torch.full((len(x),), t, dtype=torch.long)
    with torch.no_grad():
        if w == 0:
            return model(x, tt, y)
        both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, torch.full_like(y, model.n_classes)]))
    cond, uncond = both.chunk(2)
    return ...                                            # the formula of the fiche, with w
```

</details>

### Ex B5.28 — Mémorisation : plus proches voisins des échantillons, entraînement sur 500 images contre 60 000 🔬

<details><summary>Indice 1</summary>

Un plus proche voisin en pixels, c'est une distance euclidienne entre images aplaties, calculée pour toutes les paires (ch. 13) ; `torch.cdist` le fait en une fois. Pour $r$, il faut un second plus proche voisin : celui de l'image $a$ dans son propre dataset, en excluant $a$ elle-même.

</details>
<details><summary>Indice 2</summary>

- `nearest_28` : `torch.cdist(a, b)` avec `a = samples.reshape(N, -1).double()` et `b = train.reshape(M, -1).double()`, puis `.min(dim=1)` (valeurs et indices ; à égalité, PyTorch rend le premier).
- `copy_ratio_28` : les distances de chaque $a$ (`train[indices]`) à tout `train`, la distance de $a$ à elle-même mise à l'infini (`to_all[torch.arange(N), indices] = float("inf")`), puis le minimum de chaque ligne.
- Pour a), lis `FULL_28["copies"]` ; pour b), `FULL_28["median_r"]` : une valeur par graine.

</details>
<details><summary>Indice 3</summary>

```python
def copy_ratio_28(samples, train):
    d, idx = nearest_28(samples, train)
    flat = torch.as_tensor(train).reshape(len(train), -1).double()
    to_all = torch.cdist(flat[idx], flat)
    ...                                                   # inf on the distance of each a to itself
    return d / ...                                        # the nearest OTHER image of each a
```
Sur les points de la vérification, le premier : $(0 ; 0{,}5)$ a pour voisin $a = (0 ; 0)$ à 0,5, et $a$ a pour voisin $(3 ; 0)$ à 3 : $r = 0{,}5/3$. Fais les deux autres de même.

</details>

### Ex B5.29 — Distance de Fréchet : situer le DDPM face au VAE (ch. 25) et au DCGAN (ch. 27) sur MNIST, avec le code d'évaluation du mini-projet MP6 📦

<details><summary>Indice 1</summary>

Tout est déjà écrit dans `genbench.evaluate` : la fonction n'a qu'à assembler quatre appels, toujours avec le même classifieur et la même référence. Lis les docstrings de `classifier_features`, `class_report` et `nearest_neighbors` (`help(gb_eval.class_report)`).

</details>
<details><summary>Indice 2</summary>

- Les vraies images de référence : `(REAL_MNIST.images + 1) / 2`, dans $[0, 1]$ comme `images01`.
- `gb_eval.classifier_features(CLF01, …)` pour les deux ensembles, puis `gb_eval.frechet_distance(real, fake)`.
- `report = gb_eval.class_report(CLF01, images01)` donne `report["entropy"]` et `report["confidence"]`.
- `distances, _ = gb_eval.nearest_neighbors(images01, train01)`, puis `float(np.median(distances))`.

</details>
<details><summary>Indice 3</summary>

```python
def evaluate_29(images01, train01):
    real = gb_eval.classifier_features(CLF01, (REAL_MNIST.images + 1) / 2)
    fake = ...
    report = gb_eval.class_report(CLF01, images01)
    distances, _ = ...
    return {"fd": ..., "entropy": report["entropy"], "confidence": ..., "nn": float(np.median(distances))}
```

</details>

### Ex B5.30 — Un échantillonneur autonome : sample.py avec --steps, --seed et --guidance, images et paramètres sauvegardés 🛠️

<details><summary>Indice 1</summary>

Un script en ligne de commande : `argparse` pour les options, une fonction `main(argv=None)` qui rend un code de sortie, et `sys.exit(main())` en bas. Tout ce qui fait les images (le checkpoint, la graine, les pas, la guidance) doit pouvoir se relire dans `params.json`.

</details>
<details><summary>Indice 2</summary>

- `argparse.ArgumentParser` : un argument positionnel `checkpoint` (`type=Path`), puis `--steps`, `--seed`, `--guidance`, `--label`, `--n`, `--out` avec leurs types et défauts.
- `payload = torch.load(path, map_location="cpu", weights_only=True)` ; `config = payload["config"]` ; `settings = json.loads(payload["key"])["settings"]`.
- Les refus : `print(message, file=sys.stderr)` puis `return 2`, **avant** de générer.
- L'empreinte : `hashlib.sha256(path.read_bytes()).hexdigest()`.
- Une grille d'images : `matplotlib.use("Agg")` avant `pyplot` (aucun écran dans un terminal), puis `fig.savefig(...)`.

</details>
<details><summary>Indice 3</summary>

```python
def main(argv=None):
    args = parse_args(argv)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings, config = json.loads(payload["key"])["settings"], payload["config"]
    if not 1 <= args.steps <= config["n_steps"]:
        print(f"sample.py: --steps must be in [1, {config['n_steps']}]", file=sys.stderr)
        return 2
    ...                                                   # the other refusals, the model (EMA weights), x_T, y
    images = wbd.sample_ddim(model, x_T, alpha_bars, args.steps, y=y, w=args.guidance).clamp(-1, 1)
    ...                                                   # samples.npy, samples.png, params.json
    return 0
```

</details>

### Ex B5.31 — Défi : la plus petite distance de Fréchet sur Fashion-MNIST en au plus 50 pas d'échantillonnage 🏆

<details><summary>Indice 1</summary>

Pars de ce qui marche déjà : le U-Net conditionné et la guidance de B5.27, l'EMA de B5.23, DDIM. Le budget est un **temps** : une boucle qui s'arrête quand le temps est écoulé plutôt qu'un nombre d'epochs. À 50 passes par image, tu choisis entre 50 pas sans guidance et 25 pas guidés : mesure les deux.

</details>
<details><summary>Indice 2</summary>

- La boucle : `while time.perf_counter() - start < 0.95 * seconds`, un batch tiré au hasard (`torch.randint`) à chaque tour.
- Une EMA qui se « réchauffe » : $\rho_k = \min(0{,}99,\ (1 + k)/(10 + k))$ au pas $k$, pour que la copie ne garde pas les poids de départ quand le nombre de pas est inconnu d'avance.
- `generate` : ta `sample_cond_27` avec autant d'images par classe (`torch.arange(n) % 10`), puis `.clamp(-1, 1)`.
- Essaie plusieurs $w$ sur un même modèle : la meilleure guidance pour la distance n'est ni 0 ni la plus forte (🔮 B5.26).

</details>
<details><summary>Indice 3</summary>

```python
def method_31(images, labels, seconds):
    start = time.perf_counter()
    model = UNET_CLASS(T_STEPS, channels=(16, 32, 64), n_classes=10)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    step = 0
    while time.perf_counter() - start < 0.95 * seconds:
        ...                                               # one step of train_cond_27 on a random batch, then the EMA
        step += 1

    def generate(n, generator):
        ...                                               # x_T from generator, the classes, sample_cond_27 (25 steps)
    return generate
```

</details>
