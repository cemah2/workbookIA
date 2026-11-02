# B5 · Modèles de diffusion : un DDPM minimal — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/b5_diffusion/06_mes_reponses.md` (créée par `python tools/start_chapter.py B5`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ B5.1 à B5.5 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les dérivations ∂ B5.6 et B5.7, l'estimation 🧮 B5.8, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), une liste de nombres pour un vecteur (`[0.5, -1.25]`).
> Conventions du chapitre (fiche, encadré 📐) : dans les formules, les pas vont de $t = 1$ à $t = T$, comme dans l'article de Ho et coll. ; $\bar\alpha_0 = 1$. **Dans le code** (ta librairie, diffusers), les pas sont des **indices de 0 à $T - 1$** : le pas $t$ est l'indice $t - 1$, et l'indice $-1$ de `ddim_step` désigne l'image propre. Chaque énoncé donne les deux (« pas $t = 3$, indice 2 »). $\boldsymbol{\varepsilon}$ est le vrai bruit, $\boldsymbol{\varepsilon}_\theta$ celui que prédit le réseau, $\mathbf{z}$ le bruit tiré pendant l'échantillonnage. La guidance s'écrit dans la convention de Ho et Salimans, $\tilde{\boldsymbol{\varepsilon}} = (1 + w)\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, c) - w\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, \varnothing)$ ; diffusers écrit `guidance_scale` $= 1 + w$. $\ln$ est le logarithme naturel.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### B5.Q1 — Diffusion, VAE, GAN : comment chacun fabrique-t-il une image ? 🧠 ⏱️ 3 min
*Fiche §205.1 · parcours R*

a) Pour générer une image, un VAE déjà entraîné fait : (A) une seule passe de son décodeur sur un code tiré de $\mathcal{N}(\mathbf{0}, \mathbf{I})$ ; (B) un millier de passes de débruitage, chacune partant de la sortie de la précédente ; (C) une passe de son encodeur sur une image du dataset, puis une passe de son décodeur sur le code obtenu.
b) Vrai ou faux : le processus direct d'un modèle de diffusion a des paramètres, appris pendant l'entraînement en même temps que le réseau.
c) Lequel des trois apprend contre un second réseau chargé de juger ses images ? (A) le modèle de diffusion ; (B) le VAE ; (C) le GAN.
d) Vrai ou faux : un modèle de diffusion ressemble à un VAE à $T$ étages dont tous les codes $\mathbf{x}_1, \dots, \mathbf{x}_T$ ont la taille de l'image.
e) Un générateur en une passe met 0,02 s par image. Un DDPM dont le réseau coûte autant par passe échantillonne en 1 000 pas. Combien de secondes lui faut-il pour une image ?

### B5.Q2 — Pourquoi détruire l'image en mille petits pas de bruit plutôt qu'en un seul ? 🧠 ⏱️ 3 min
*Fiche §205.1, §205.2 · parcours R*

a) Un réseau entraîné par MSE à passer **en un seul coup** du bruit pur à l'image rend : (A) une image nette, tirée au hasard parmi toutes celles qui auraient pu donner ce bruit ; (B) la moyenne des images compatibles avec ce bruit, donc une image floue ; (C) l'image du dataset la plus proche du bruit, recopiée.
b) Pour une même entrée, trois cibles sont possibles et équiprobables : $-1$, $0$ et $2$. Quelle prédiction minimise la MSE, à 2 décimales ?
c) Et que vaut alors cette MSE, à 2 décimales ?
d) Vrai ou faux : quand $\beta_t$ est petit, la loi du pas inverse $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ est presque gaussienne.
e) Vrai ou faux : pour détruire autant l'image au bout de la chaîne, une chaîne de 2 000 pas demande des $\beta_t$ plus grands qu'une chaîne de 1 000 pas.

### B5.Q3 — ᾱ_t : quelle part de l'image reste-t-il au pas t ? 🧠 ⏱️ 3 min
*Fiche §205.2 · parcours R, M*

Au pas $t$, $\bar\alpha_t = 0{,}64$.

a) Le coefficient de $\mathbf{x}_0$ dans $\mathbf{x}_t$, à 2 décimales.
b) Le coefficient du bruit $\boldsymbol{\varepsilon}$ dans $\mathbf{x}_t$, à 2 décimales.
c) Vrai ou faux : la somme de ces deux coefficients vaut toujours 1.
d) Le rapport signal sur bruit au pas $t$, à 2 décimales.
e) Quand $t$ augmente : (A) $\bar\alpha_t$ croît, puisque le réseau a vu plus de pas ; (B) $\bar\alpha_t$ reste presque constant, puis tombe à 0 au dernier pas ; (C) $\bar\alpha_t$ décroît strictement.

### B5.Q4 — Prédire le bruit, l'image ou la vitesse : la même information ? 🧠 ★★ ⏱️ 3 min
*Fiche §205.3, §205.9 · parcours —*

a) Connaissant $\mathbf{x}_t$, $t$ et le **vrai** bruit $\boldsymbol{\varepsilon}$, l'image propre $\mathbf{x}_0$ se retrouve : (A) exactement, en inversant la forme fermée ; (B) seulement de façon approchée, d'autant mieux que $t$ est petit ; (C) jamais sans un réseau entraîné.
b) $\bar\alpha_t = 0{,}64$, $x_t = 0{,}5$ et un réseau qui prédit $\varepsilon_\theta = 0{,}2$ : que vaut l'estimation $\hat{x}_0$, à 3 décimales ?
c) Vrai ou faux : Ho et coll. obtenaient de moins bons échantillons en faisant prédire $\mathbf{x}_0$ au réseau qu'en lui faisant prédire le bruit.
d) Le bruit est une cible commode parce que : (A) il ne dépend pas du pas, si bien que le réseau n'a plus besoin de connaître $t$ ; (B) il est plus petit que l'image à tous les pas, donc plus facile à apprendre ; (C) c'est une gaussienne standard, de la même échelle à tous les pas.
e) Vrai ou faux : la $\mathbf{v}$-prédiction de Salimans et Ho et la vitesse $\mathbf{v}$ du flow matching sont la même combinaison de $\boldsymbol{\varepsilon}$ et de $\mathbf{x}_0$.

### B5.Q5 — Le pas de temps est une entrée du réseau : pourquoi, et comment l'encoder ? 🧠 ⏱️ 3 min
*Fiche §205.4 · parcours R*

a) Le réseau reçoit le pas $t$ : (A) pour choisir lequel de ses $T$ jeux de poids utiliser à ce pas ; (B) pour calculer lui-même $\bar\alpha_t$ et l'ajouter à sa sortie ; (C) pour savoir combien de bruit s'attendre à trouver dans $\mathbf{x}_t$.
b) Vrai ou faux : donner $t$ brut, un flottant de 0 à 999 ajouté comme un canal de plus, marche aussi bien qu'un encodage sinusoïdal.
c) Tu prends les lignes `t` de `positional_encoding(1000, 128)` (B3) comme encodage des pas. Combien de couples (sinus, cosinus) contient l'encodage d'un pas ?
d) Dans le bloc résiduel de la fiche, l'encodage du pas $\mathbf{e}_t$ devient : (A) un canal de plus, concaténé à la carte d'activation ; (B) un nombre par canal, ajouté à toute la carte de ce canal ; (C) un facteur qui multiplie les poids de la première convolution.
e) Vrai ou faux : ranger les colonnes de l'encodage en deux moitiés (sinus, puis cosinus) au lieu de les alterner ne change rien pour le MLP qui le suit, si l'on garde la même convention de l'entraînement à l'échantillonnage.

### B5.Q6 — U-Net : à quoi servent les connexions de saut ? 🧠 ⏱️ 3 min
*Fiche §205.4 · parcours R*

a) Une connexion de saut du U-Net : (A) concatène, sur l'axe des canaux, une carte de l'encodeur à la carte du décodeur de même taille ; (B) relie l'entrée du réseau à sa sortie, par une addition ; (C) additionne l'entrée d'un bloc à sa sortie, comme les connexions résiduelles d'un ResNet du ch. 20.
b) Dans un décodeur, une convolution transposée rend 128 canaux en $14 \times 14$ ; la connexion de saut apporte la carte de l'encodeur de même taille, à 64 canaux. Combien de canaux d'entrée (`in_channels`) attend la convolution suivante ?
c) Vrai ou faux : sans ses connexions de saut, le U-Net est un autoencodeur, dont les détails fins doivent passer par le goulot.
d) Une image de $32 \times 32$ traverse trois réductions de taille par des convolutions de stride 2. Quelle est la largeur de la carte du bas du U ?
e) Le U-Net de la fiche normalise par group norm plutôt que par batchnorm : (A) parce que la group norm n'a aucun paramètre appris ; (B) parce qu'elle normalise chaque pixel séparément ; (C) parce qu'elle ne dépend pas des autres images du batch.

### B5.Q7 — Échantillonnage DDPM : pourquoi est-il lent et aléatoire ? 🧠 ⏱️ 3 min
*Fiche §205.5, §205.6 · parcours R*

a) Une passe du réseau sur un batch prend 0,05 s. Combien de secondes pour échantillonner ce batch par DDPM en 1 000 pas ?
b) Vrai ou faux : deux générations DDPM qui partent du même $\mathbf{x}_T$ donnent la même image.
c) Vrai ou faux : au dernier pas ($t = 1$, indice 0), l'algorithme 2 n'ajoute aucun bruit.
d) On ne peut pas faire les 1 000 passes en même temps sur 1 000 cœurs : (A) parce que chaque passe a besoin du $\mathbf{x}_t$ produit par la précédente ; (B) parce que le planning de bruit interdit les batches de plus d'une image ; (C) parce que le réseau ne tient qu'une fois en mémoire.
e) On remplace $\sigma_t^2 = \tilde\beta_t$ par $\sigma_t^2 = \beta_t$ dans l'échantillonneur. Alors : (A) la moyenne de chaque pas change, et le réseau doit être réentraîné ; (B) seul le bruit ajouté à chaque pas change, la moyenne reste la même ; (C) rien ne change, puisque $\tilde\beta_t = \beta_t$ pour tous les pas.

### B5.Q8 — DDIM : moins de pas avec le même réseau, vrai ou faux 🧠 ★★ ⏱️ 3 min
*Fiche §205.6 · parcours R*

a) Vrai ou faux : pour échantillonner en 50 pas par DDIM, il faut réentraîner le réseau avec un planning de 50 pas.
b) Vrai ou faux : avec $\eta = 0$, deux générations DDIM qui partent du même $\mathbf{x}_T$ donnent la même image.
c) Vrai ou faux : avec $\eta = 1$ et $t' = t - 1$, un pas DDIM est un pas DDPM avec la variance $\tilde\beta_t$.
d) Vrai ou faux : un pas DDIM peut aller de l'indice $t$ à n'importe quel indice antérieur, et même directement à l'image propre.
e) Vrai ou faux : avec un réseau bien entraîné, DDIM en 10 pas donne toujours des images aussi bonnes que DDIM en 100 pas.

### B5.Q9 — Guidance sans classifieur : ce que règle l'échelle w 🧠 ⏱️ 3 min
*Fiche §205.7 · parcours R*

a) En un pixel, $\varepsilon_\theta(\mathbf{x}_t, c) = 0{,}3$ et $\varepsilon_\theta(\mathbf{x}_t, \varnothing) = 0{,}5$. Que vaut $\tilde\varepsilon$ avec $w = 2$, à 1 décimale ?
b) Avec $w = 0$, on échantillonne : (A) le modèle sans condition, comme si le label était $\varnothing$ ; (B) le modèle conditionné, sans guidance ; (C) la moyenne des prédictions avec et sans condition.
c) Vrai ou faux : augmenter $w$ rend les échantillons d'une même classe plus variés.
d) Dans diffusers, `guidance_scale=3`. Quelle est la valeur de $w$ dans la convention de ce chapitre ?
e) Ce qui permet à **un seul** réseau de prédire avec et sans condition : (A) un second réseau, entraîné sans condition, dont on combine les sorties ; (B) rien de particulier : la prédiction sans condition se calcule à l'échantillonnage en moyennant les classes ; (C) remplacer le label par $\varnothing$ pour une partie des exemples d'entraînement.

### B5.Q10 — Un bon FID peut-il cacher de la copie des données d'entraînement ? 🧠 ⏱️ 3 min
*Fiche §205.8 · parcours R*

a) Vrai ou faux : un générateur qui recracherait des copies exactes de ses images d'entraînement pourrait avoir une excellente distance de Fréchet.
b) Pour chercher la mémorisation, on : (A) compare chaque échantillon à ses plus proches voisins d'entraînement ; (B) calcule la distance de Fréchet sur deux fois plus d'images, pour réduire son biais ; (C) regarde si la loss de validation remonte en fin d'entraînement.
c) Vrai ou faux : d'après Carlini et coll., les images dupliquées dans le dataset sont celles que les modèles mémorisent le moins.
d) Une distance de Fréchet est calculée sur 64 images générées, une autre sur 10 000, avec les mêmes features : (A) elles se comparent directement, puisque les features et le réseau qui les calcule sont les mêmes ; (B) elles ne se comparent pas : avec peu d'images, l'estimation est biaisée vers le haut ; (C) elles se comparent après avoir multiplié la première par $10\,000/64$.
e) Vrai ou faux : une distance de Fréchet basse suffit à garantir à la fois la fidélité et la diversité des échantillons.

### B5.Q11 — Diffusion latente, transformers de diffusion, flow matching : ce qui a changé depuis 2020 🧠 ⏱️ 3 min
*Fiche §205.9 · parcours R*

a) Un autoencodeur réduit des images de $256 \times 256 \times 3$ d'un facteur $f = 4$ en largeur et en hauteur, avec 3 canaux. Combien de valeurs compte le latent d'une image ?
b) Un transformer de diffusion (DiT) découpe le latent en : (A) patchs, comme une image dans un ViT ; (B) tokens de texte, par le tokenizer de l'encodeur de texte ; (C) pixels, un token par valeur du latent.
c) Dans le flow matching de la fiche, $\mathbf{x}_s = (1 - s)\,\mathbf{x}_0 + s\,\boldsymbol{\varepsilon}$. La vitesse que le réseau apprend : (A) $\mathbf{x}_0 - \boldsymbol{\varepsilon}$ ; (B) $\mathbf{x}_0 + \boldsymbol{\varepsilon}$ ; (C) $\boldsymbol{\varepsilon} - \mathbf{x}_0$.
d) Vrai ou faux : la méthode d'Euler suit exactement, en un seul pas, une trajectoire droite parcourue à vitesse constante.
e) Vrai ou faux : Stable Diffusion 3 et FLUX.1 sont des U-Net de diffusion qui travaillent sur les pixels.

<a id="rappels"></a>

## 🔁 Rappels

### B5.R1 — Ch. B3 : l'encodage sinusoïdal des positions 🔁 ★ ⏱️ 5 min
*Ch. B3 (encodage sinusoïdal, `positional_encoding`) · fiche §205.4 · parcours R*

Ta `positional_encoding(seq_len, d_model)` calcule $\operatorname{PE}[p, 2i] = \sin(p / 10\,000^{2i/d})$ et $\operatorname{PE}[p, 2i + 1] = \cos(p / 10\,000^{2i/d})$. Ici, $d = 128$ et la « position » $p$ est le pas de diffusion.

a) Que vaut $\operatorname{PE}[0, 1]$ ?
b) La colonne 0 est une sinusoïde en $p$. Quelle est sa période, en pas, à 2 décimales ?
c) Vrai ou faux : les encodages des pas 500 et 501 diffèrent surtout dans les dernières colonnes (les grands $i$, les fréquences lentes).
d) Vrai ou faux : sur les 1 000 pas d'un DDPM, la dernière paire de colonnes ($i = 63$) ne fait même pas un tour complet.
e) Dans ta `positional_encoding`, la colonne $2i$ contient : (A) le cosinus ; (B) le sinus ; (C) la position $p$ divisée par $d$.

### B5.R2 — Ch. 27 : ce que mesure la distance de Fréchet et l'effondrement de mode 🔁 ★ ⏱️ 5 min
*Ch. 27 (distance de Fréchet, FID, effondrement de mode, `frechet_distance`) · fiche §205.8 · parcours R*

Ta `frechet_distance` rend le **carré** $d_F^2 = \lVert \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2 \rVert^2 + \operatorname{Tr}\big(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2 - 2(\boldsymbol{\Sigma}_1\boldsymbol{\Sigma}_2)^{1/2}\big)$. Les features sont ici de dimension 1.

a) Features réelles $\mathcal{N}(0, 1^2)$, features générées $\mathcal{N}(1, 2^2)$ : que vaut $d_F^2$ ?
b) Un GAN effondré ne génère plus que des 1 et des 7. Sa distance de Fréchet, par rapport à celle du même GAN avant l'effondrement : (A) augmente : moyenne et covariance de ses features s'écartent de celles des vraies images ; (B) reste la même, puisque chaque image générée reste nette et bien formée ; (C) devient nulle.
c) Vrai ou faux : $d_F^2 = 0$ si et seulement si les deux gaussiennes ajustées ont la même moyenne et la même covariance.
d) Vrai ou faux : le FID compare les images générées et les vraies images pixel par pixel.
e) Features réelles $\mathcal{N}(0, 1^2)$ ; un générateur rend toujours la même image, dont la feature vaut exactement 0. Que vaut $d_F^2$ ?

### B5.R3 — Ch. 25 : l'astuce de reparamétrisation et le terme KL du VAE 🔁 ★ ⏱️ 5 min
*Ch. 25 (VAE, `reparameterize`, `gaussian_kl`, ELBO) · fiche §205.2, §205.3 · parcours R*

Un encodeur de VAE rend, pour une image, $\mu = 0{,}5$ et $\ln \sigma^2 = \ln 0{,}25$ (un latent de dimension 1).

a) Avec $\varepsilon = -2$, que vaut le code $z$ tiré par la reparamétrisation, à 2 décimales ?
b) Que vaut le terme $\mathrm{KL}\big(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)\big)$, en nats, à 4 décimales ?
c) La reparamétrisation sert à : (A) laisser passer le gradient jusqu'à $\mu$ et $\sigma$ ; (B) diviser par deux la mémoire du latent, en ne gardant qu'un tirage par image ; (C) rendre $z$ déterministe pendant l'entraînement, pour que la loss ne varie plus d'un batch à l'autre.
d) Vrai ou faux : dans un modèle de diffusion, le bruit $\boldsymbol{\varepsilon}$ de l'écriture « moyenne + écart-type × bruit » devient la cible du réseau.
e) Vrai ou faux : le premier terme de l'ELBO d'un DDPM, $\mathrm{KL}\big(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\big)$, diminue pendant l'entraînement, comme le terme KL du VAE.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex B5.1 — Guidance sans classifieur : combiner les deux prédictions pour w = 0, 1 et 3 ✏️ ★ ⏱️ 10 min
**Objectif :** appliquer la formule de la guidance sans classifieur, voir qu'elle extrapole, et compter ce qu'elle coûte.
**Prérequis :** 0B (vecteurs, norme) · fiche §205.7 (guidance sans classifieur, conventions de l'échelle) · **Parcours :** R, M

À un pas donné, pour deux pixels, le réseau prédit $\boldsymbol{\varepsilon}_c = \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, c) = (0{,}8 ;\ -0{,}5)$ avec la condition et $\boldsymbol{\varepsilon}_\varnothing = \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, \varnothing) = (0{,}5 ;\ 0{,}1)$ sans condition. Donne les vecteurs sous la forme d'une liste de deux nombres, à 2 décimales.

a) $\tilde{\boldsymbol{\varepsilon}}$ pour $w = 0$.
b) $\tilde{\boldsymbol{\varepsilon}}$ pour $w = 1$.
c) $\tilde{\boldsymbol{\varepsilon}}$ pour $w = 3$.
d) La valeur de `guidance_scale` qui, dans diffusers, donne le même $\tilde{\boldsymbol{\varepsilon}}$ que c).
e) La norme euclidienne de $\tilde{\boldsymbol{\varepsilon}} - \boldsymbol{\varepsilon}_c$ pour $w = 3$, à 2 décimales.
f) Vrai ou faux : avec $w = -1$, $\tilde{\boldsymbol{\varepsilon}}$ est exactement la prédiction sans condition.
g) Vrai ou faux : pour $w = 3$, chaque composante de $\tilde{\boldsymbol{\varepsilon}}$ reste comprise entre celles de $\boldsymbol{\varepsilon}_\varnothing$ et de $\boldsymbol{\varepsilon}_c$.
h) Un échantillonnage DDIM en 50 pas utilise la guidance. Combien d'évaluations du réseau (une image, une condition, un pas) faut-il pour une image ?

Dans ta copie : dessine les trois points $\boldsymbol{\varepsilon}_\varnothing$, $\boldsymbol{\varepsilon}_c$ et $\tilde{\boldsymbol{\varepsilon}}$ ($w = 3$) dans le plan, et la droite qui les porte. Que se passe-t-il en un pixel où les deux prédictions sont égales ? Pourquoi la guidance ne coûte-t-elle en pratique presque pas plus de temps que sans elle sur un GPU, alors qu'elle double le nombre d'évaluations ?

### Ex B5.2 — Planning linéaire : β_t, α_t et ᾱ_t pour T = 4, puis ᾱ_T pour T = 1 000 ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer à la main un planning de bruit et ses produits cumulés, et estimer ce qu'il reste de l'image au bout d'une vraie chaîne.
**Prérequis :** 0B (produits, exponentielle et logarithme, somme d'une suite arithmétique) · fiche §205.2 (le planning linéaire, l'approximation $\bar\alpha_T \approx e^{-\sum \beta}$) · **Parcours :** R, M

Un planning linéaire jouet de $T = 4$ pas va de $\beta_1 = 0{,}05$ à $\beta_4 = 0{,}35$ (c'est `linear_beta_schedule(4, 0.05, 0.35)`). Il resservira dans ✏️ B5.4, B5.5 et les dérivations ∂ B5.6 et B5.7.

a) $\beta_2$ (pas 2, indice 1), à 2 décimales.
b) $\bar\alpha_2$, à 4 décimales.
c) $\bar\alpha_4$, à 4 décimales.
d) La part de l'image qui reste au pas 4, c'est-à-dire le coefficient de $x_0$ dans $x_4$, à 4 décimales.
e) L'approximation $e^{-\sum_t \beta_t}$ de $\bar\alpha_4$, à 4 décimales.
f) Vrai ou faux : sur ce planning, l'approximation e) est à moins de 1 % de la valeur exacte c).

Le planning de Ho et coll. : $T = 1\,000$, $\beta$ linéaire de $10^{-4}$ à $0{,}02$.

g) La somme des 1 000 valeurs de $\beta$, à 2 décimales.
h) L'approximation de $\bar\alpha_T$ par $e^{-\sum \beta}$, **multipliée par $10^5$**, à 2 décimales.
i) La part de l'image qui reste au dernier pas, $\sqrt{\bar\alpha_T}$, **en pourcentage**, à 2 décimales (avec ta valeur de h).
j) Avec les mêmes bornes mais $T = 400$, la somme des $\beta$, à 2 décimales.
k) Avec $T = 400$ et les deux bornes multipliées par $1\,000/T$ (comme dans le code de Nichol et Dhariwal), la somme des $\beta$, à 2 décimales.
l) Vrai ou faux : avec ces bornes multipliées, $\bar\alpha_T$ reste du même ordre de grandeur qu'avec $T = 1\,000$.

Dans ta copie : montre que $\ln(1 - \beta) \le -\beta$ pour $0 \le \beta < 1$ (étudie la fonction $\beta \mapsto \ln(1 - \beta) + \beta$), et déduis-en si l'approximation $e^{-\sum \beta}$ surestime ou sous-estime $\bar\alpha_T$. Pourquoi est-elle si bonne pour le planning de Ho et coll., et si mauvaise pour le planning jouet ?

### Ex B5.3 — Rapport signal sur bruit : à quel pas un chiffre devient-il illisible, planning linéaire ou cosinus ? ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer le rapport signal sur bruit des deux plannings de référence, et trouver où chacun passe deux seuils.
**Prérequis :** Ex B5.2 · 0B (équation du second degré, trigonométrie) · fiche §205.2 (planning cosinus, rapport signal sur bruit) · **Parcours :** M

$T = 1\,000$ ; $t$ est le pas de l'article ($t = 1$ à $1\,000$), et $u = t/T$. On admet, **pour cet exercice**, qu'un chiffre de MNIST reste lisible tant que $\operatorname{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t) \ge 0{,}1$ (une convention commode, pas une loi).

- **Planning linéaire** ($\beta$ de $10^{-4}$ à $0{,}02$) : on prend l'approximation $\bar\alpha_t \approx e^{-S(t)}$, où $S(t) = \sum_{k=1}^{t} \beta_k = t\,\beta_1 + (\beta_T - \beta_1)\,\frac{t(t - 1)}{2(T - 1)}$ est la somme des $t$ premiers $\beta$.
- **Planning cosinus** ($s = 0{,}008$) : $\bar\alpha_t = f(u) / f(0)$, avec $f(u) = \cos^2\!\left(\frac{u + s}{1 + s} \cdot \frac{\pi}{2}\right)$ ; cette formule est exacte (hors des tout derniers pas, où les $\beta$ sont écrêtés).

a) $S(500)$, à 3 décimales.
b) Le SNR du planning linéaire au pas 500, à 2 décimales.
c) Le SNR du planning cosinus au pas 500, à 2 décimales.
d) Planning linéaire : la fraction $t/T$ de la chaîne où le SNR passe sous 1, à 2 décimales.
e) Planning cosinus : la même fraction, à 2 décimales.
f) Planning linéaire : la fraction $t/T$ où le SNR passe sous le seuil de lisibilité 0,1, à 2 décimales.
g) Planning cosinus : la même fraction, à 2 décimales.
h) Vrai ou faux : avec le planning linéaire, le chiffre est illisible pendant plus de la moitié des pas de la chaîne.
i) Vrai ou faux : avec le planning cosinus, le chiffre reste lisible pendant plus de 90 % des pas.
j) Lequel des deux plannings passe la plus grande part de ses pas dans la zone intermédiaire, entre $\operatorname{SNR} = 1$ et $\operatorname{SNR} = 0{,}1$ ? (A) le linéaire ; (B) le cosinus ; (C) aucun, ils y passent autant de pas.

Dans ta copie : pourquoi dit-on que le planning linéaire « gaspille » des pas ? En quoi le planning cosinus répond-il au constat de Nichol et Dhariwal (fiche §205.2) ? Calcule $\ln \operatorname{SNR}$ au dernier pas du planning linéaire avec ta réponse h) de ✏️ B5.2 : pourquoi la figure `plannings.png`, coupée à $-8$, ne le montre-t-elle pas ?

### Ex B5.4 — Un pas DDPM à la main, en dimension 1, avec un bruit prédit donné ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer à la main un pas de l'échantillonnage ancestral, avec ses deux choix de variance, l'écrêtage de $\hat{x}_0$ et le dernier pas.
**Prérequis :** Ex B5.2 · fiche §205.3 (posterior, prédire le bruit), §205.5 (l'algorithme 2, le pas à la main) · **Parcours :** R, M

Planning de ✏️ B5.2 ($T = 4$, $\beta = 0{,}05 ;\ 0{,}15 ;\ 0{,}25 ;\ 0{,}35$). On est au pas $t = 3$ (indice 2), avec $x_3 = 0{,}8$ ; le réseau prédit $\varepsilon_\theta = 0{,}5$, et le tirage de l'échantillonneur donne $z = -1{,}0$. Arrondis à 4 décimales.

a) Le coefficient $\beta_3 / \sqrt{1 - \bar\alpha_3}$.
b) La moyenne $\mu_\theta$ de l'algorithme 2.
c) $\tilde\beta_3$.
d) $x_2$ avec $\sigma_3^2 = \tilde\beta_3$ (`variance="posterior"`).
e) $x_2$ avec $\sigma_3^2 = \beta_3$ (`variance="beta"`), le même $z$.
f) Toujours au pas 3, mais avec $x_3 = 1{,}5$ et $\varepsilon_\theta = -0{,}6$ : la moyenne du pas quand $\hat{x}_0$ est d'abord écrêté dans $[-1, 1]$ (`clip_x0=1.0`), puis mis à la place de $x_0$ dans la moyenne du posterior.
g) Le dernier pas : au pas 1 (indice 0), avec $x_1 = 0{,}5$ et $\varepsilon_\theta = 0{,}2$, la valeur rendue $x_0$.
h) Vrai ou faux : au pas 1, `ddpm_step(..., variance="beta")` ajoute un bruit $\sqrt{\beta_1}\,z$ à la moyenne.
i) Vrai ou faux : sans écrêtage, la moyenne $\mu_\theta$ est la même avec les deux choix de variance.

Dans ta copie : refais b) par le détour de $\hat{x}_0$ et des coefficients du posterior. Pour f), que vaudrait la moyenne sans écrêtage, et pourquoi l'écrêtage est-il utile aux grands $t$ ? Pourquoi le dernier pas n'ajoute-t-il pas de bruit ?

### Ex B5.5 — Un pas DDIM à la main et l'image estimée x̂_0 ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer à la main l'image estimée et un pas DDIM, déterministe ou non, et le traduire en appel de ta librairie.
**Prérequis :** Ex B5.4 · fiche §205.6 (DDIM, le paramètre η) · **Parcours :** M

Planning de ✏️ B5.2. Au pas 4 (indice 3), $x_4 = -0{,}5$ et le réseau prédit $\varepsilon_\theta = -1{,}2$. Arrondis à 4 décimales.

a) L'image estimée $\hat{x}_0$.
b) $x_1$ obtenu en **un** pas DDIM ($\eta = 0$) du pas 4 au pas 1.
c) L'appel de ta librairie qui calcule b) : (A) `ddim_step(x, 4, 1, eps, alpha_bars)` ; (B) `ddim_step(x, 3, 0, eps, alpha_bars)` ; (C) `ddim_step(x, 3, -1, eps, alpha_bars)`.
d) $x_2$ obtenu en un pas DDIM ($\eta = 0$) du pas 4 au pas 2.
e) Vrai ou faux : si le réseau prédit encore $\varepsilon_\theta = -1{,}2$ au pas 2, un second pas DDIM du pas 2 au pas 1 redonne exactement le $x_1$ de b).
f) Avec $\eta = 1$, du pas 4 au pas 3 : $\sigma^2$.
g) Avec $\eta = 1$, du pas 4 au pas 1 : $\sigma$.
h) Dans le cas g), le coefficient de $\varepsilon_\theta$, $\sqrt{1 - \bar\alpha_1 - \sigma^2}$.
i) Vrai ou faux : dans le cas g), la plus grande part du bruit de $x_1$ vient de $\varepsilon_\theta$, et non du tirage $\sigma z$.

Dans ta copie : compare f) à $\tilde\beta_4$ ; que montre cette égalité ? Avec le **vrai** bruit à la place de $\varepsilon_\theta$, que rendrait un pas DDIM jusqu'à l'indice $-1$ ? Pourquoi e) explique-t-il qu'un réseau dont la prédiction varie peu d'un pas à l'autre s'échantillonne bien en peu de pas ?

### Ex B5.6 — Forme fermée : deux pas de bruit gaussien n'en font qu'un, d'où q(x_t | x_0) ∂ ★★★ ⏱️ 35 min
**Objectif :** démontrer par récurrence la forme fermée du processus direct, puis la loi d'un état sachant un état antérieur quelconque.
**Prérequis :** Ex B5.2 · ch. 2 (variance d'une somme de variables indépendantes) · ch. 25 (reparamétrisation) · fiche §205.2 (encadré 🧮 sur la somme de deux gaussiennes, la forme fermée) · **Parcours :** M

On travaille en dimension 1 : chaque pixel suit la même loi, avec les mêmes coefficients. Un pas direct s'écrit $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1 - \alpha_t}\,\varepsilon_t$, où les $\varepsilon_1, \varepsilon_2, \dots$ sont des $\mathcal{N}(0, 1)$ indépendants entre eux et de $x_0$ ; $\bar\alpha_t = \alpha_1 \cdots \alpha_t$.

> 🧮 **Rappel maths — le raisonnement par récurrence** — Pour démontrer qu'une propriété $P(t)$ est vraie pour tout entier $t \ge 1$, on montre deux choses : **l'initialisation**, $P(1)$ est vraie ; **l'hérédité**, si $P(t - 1)$ est vraie, alors $P(t)$ l'est aussi. Comme une rangée de dominos : le premier tombe, et chacun fait tomber le suivant. *Exemple* : $P(t)$ : « $1 + 3 + \dots + (2t - 1) = t^2$ ». $P(1)$ : $1 = 1^2$. Si $P(t - 1)$ est vraie, la somme jusqu'à $2t - 1$ vaut $(t - 1)^2 + 2t - 1 = t^2$ : $P(t)$ est vraie.

Réponds dans ta copie.

1. **Un pas.** Quelle est la loi de $x_1$ sachant $x_0$ ? Vérifie que c'est la forme fermée $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\bar\varepsilon_t$ pour $t = 1$.
2. **Deux pas.** Écris $x_2$ en fonction de $x_0$, $\varepsilon_1$ et $\varepsilon_2$. Donne le coefficient de $x_0$, puis la loi de la somme des deux termes de bruit, avec l'encadré de la fiche. Montre que sa variance vaut $1 - \alpha_1\alpha_2$, et conclus.
3. **La récurrence.** On suppose $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1 - \bar\alpha_{t-1}}\,\bar\varepsilon_{t-1}$, avec $\bar\varepsilon_{t-1} \sim \mathcal{N}(0, 1)$. Pourquoi $\bar\varepsilon_{t-1}$ est-il indépendant de $\varepsilon_t$ ? Montre la propriété au pas $t$, et dis à quel endroit l'indépendance sert.
4. **Application.** Avec le planning de ✏️ B5.2, calcule de deux façons la variance du bruit de $x_2$ : par la question 2 ($\alpha_2(1 - \alpha_1) + (1 - \alpha_2)$), puis par $1 - \bar\alpha_2$.
5. **La variance préservée.** Si $x_0$ a une moyenne 0 et une variance 1 et qu'il est indépendant des bruits, que vaut la variance de $x_t$ pour tout $t$ ? Et si les pixels de l'image ont une variance de $0{,}5$ ? Calcule-la au pas 4 du planning de ✏️ B5.2, et dis vers quoi elle tend quand $\bar\alpha_t \to 0$.
6. **Sans le facteur $\sqrt{1 - \beta_t}$.** Si chaque pas faisait $x_t = x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t$, quelle serait la loi de $x_t$ sachant $x_0$ ? Calcule l'écart-type du bruit final pour le planning linéaire de Ho et coll. (ta réponse g) de ✏️ B5.2). Pourquoi ce choix compliquerait-il l'échantillonnage, qui part d'un bruit tiré au hasard ?
7. **D'un état à un autre.** Pour $s < t$, montre que $x_t = \sqrt{\bar\alpha_t / \bar\alpha_s}\;x_s + \sqrt{1 - \bar\alpha_t / \bar\alpha_s}\;\varepsilon$, avec $\varepsilon \sim \mathcal{N}(0, 1)$ indépendant de $x_s$. Indication : refais la récurrence en partant de $x_s$ ; le produit $\alpha_{s+1} \cdots \alpha_t$ vaut $\bar\alpha_t / \bar\alpha_s$. Application : $s = 2$, $t = 4$, planning de ✏️ B5.2. Où retrouves-tu le terme $1 - \bar\alpha_t / \bar\alpha_s$ dans la fiche ?

### Ex B5.7 — Moyenne et variance du posterior q(x_{t−1} | x_t, x_0) en dimension 1 ∂ ★★★ ⏱️ 45 min
**Objectif :** dériver la moyenne et la variance du posterior d'un pas, puis en déduire la moyenne de l'algorithme 2.
**Prérequis :** Ex B5.6 · ch. 4 (règle de Bayes, prior et posterior) · fiche §205.3 (encadré 🧮 sur le produit de deux densités gaussiennes, le posterior, prédire le bruit) · **Parcours :** M

En dimension 1, on cherche la loi de $y = x_{t-1}$ quand on connaît $x_t$ et $x_0$ (deux nombres). On note $\mathcal{N}(y;\ m, v)$ la densité d'une gaussienne de moyenne $m$ et de variance $v$, vue comme une fonction de $y$. Rappel : $\alpha_t = 1 - \beta_t$ et $\bar\alpha_t = \alpha_t\,\bar\alpha_{t-1}$. Réponds dans ta copie.

1. **Bayes.** Écris $q(y \mid x_t, x_0) \propto q(x_t \mid y, x_0)\; q(y \mid x_0)$. Pourquoi $q(x_t \mid y, x_0) = q(x_t \mid y)$ ? Lequel des deux facteurs donne la forme fermée de ∂ B5.6, et que vaut-il ?
2. **La vraisemblance, vue en $y$.** $q(x_t \mid y) = \mathcal{N}(x_t;\ \sqrt{\alpha_t}\,y,\ \beta_t) \propto \exp\!\left(-\frac{(x_t - \sqrt{\alpha_t}\,y)^2}{2\beta_t}\right)$. En écrivant $x_t - \sqrt{\alpha_t}\,y = -\sqrt{\alpha_t}\,(y - x_t / \sqrt{\alpha_t})$, montre que, comme fonction de $y$, elle est proportionnelle à $\mathcal{N}(y;\ x_t / \sqrt{\alpha_t},\ \beta_t / \alpha_t)$.
3. **Les précisions.** Avec la règle du produit de l'encadré de la fiche, écris $1/\tilde\beta_t$ comme la somme de deux précisions. Montre que $\alpha_t(1 - \bar\alpha_{t-1}) + \beta_t = 1 - \bar\alpha_t$, puis que $\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t$.
4. **La moyenne.** Écris la moyenne $\tilde\mu_t$ donnée par la règle du produit, puis simplifie-la jusqu'à la forme de la fiche : $\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\,x_t$.
5. **Trois contrôles.** (a) Au pas $t = 1$ ($\bar\alpha_0 = 1$), que valent $\tilde\beta_1$ et les deux coefficients ? Interprète. (b) Montre que $\tilde\beta_t \le \beta_t$. (c) Si $x_t$ vaut exactement $\sqrt{\bar\alpha_t}\,x_0$ (un bruit nul), montre que $\tilde\mu_t = \sqrt{\bar\alpha_{t-1}}\,x_0$ : le posterior ne s'éloigne pas de la trajectoire sans bruit.
6. **Application.** Avec le planning de ✏️ B5.2, au pas 2 (indice 1) : calcule les deux coefficients et $\tilde\beta_2$ à 4 décimales, puis $\tilde\mu_2$ pour $x_0 = 0{,}4$ et $x_2 = 0{,}1$.
7. **Du posterior à l'échantillonnage.** Remplace $x_0$ par $\hat{x}_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\varepsilon) / \sqrt{\bar\alpha_t}$ dans $\tilde\mu_t$, et montre que l'on obtient la moyenne de l'algorithme 2, $\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\varepsilon\right)$. Indication : montre d'abord que le coefficient de $x_t$ vaut $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} + \frac{\beta_t}{\sqrt{\alpha_t}\,(1 - \bar\alpha_t)} = \frac{1}{\sqrt{\alpha_t}}$.

### Ex B5.8 — Fermi : coût d'échantillonnage, de MNIST sur CPU à une image 1024×1024, et ce que gagnent DDIM et l'espace latent 🧮 ★★ ⏱️ 20 min
**Objectif :** chiffrer en ordres de grandeur le coût d'un échantillonnage, de ton DDPM sur CPU à un générateur moderne, et ce que rapporte chaque levier.
**Prérequis :** Ex B5.5 · ch. 25 (autoencodeur, espace latent) · B3 (une passe avant coûte environ 2 FLOP par paramètre et par token) · fiche §205.5 (ce que coûte un échantillon), §205.6 (💼 En entreprise), §205.9 (diffusion latente, transformers de diffusion) · **Parcours :** R, M

Hypothèses de calcul (des ordres de grandeur, pas des mesures) :
- **ton DDPM sur MNIST** : le U-Net de la fiche, environ $2 \times 10^8$ FLOP par image et par passe, et 0,13 s par passe sur un batch de 64 images, sur les deux cœurs du workbook ;
- **un générateur texte vers image de 2026**, à $1\,024 \times 1\,024$ pixels en couleurs : un autoencodeur qui réduit l'image d'un facteur $f = 8$ en largeur et en hauteur, avec 16 canaux (comme Stable Diffusion 3) ; un transformer de diffusion de $8 \times 10^9$ paramètres (l'ordre de grandeur de Stable Diffusion 3.5 Large) qui découpe le latent en patchs de $2 \times 2$ ; une passe coûte environ $2 \times N \times$ (nombre de tokens) FLOP ; on néglige le coût de l'attention, de l'encodeur de texte et du décodeur ; 28 pas d'échantillonnage avec guidance ;
- **un GPU récent** qui tient environ $2 \times 10^{14}$ FLOP par seconde **utiles** (une fraction de sa crête) ; le CPU du workbook, environ $10^{11}$.

Réponds dans ta copie, avec des ordres de grandeur ; des fourchettes sont bienvenues.

1. **MNIST sur CPU.** Pour calculer une distance de Fréchet, tu veux 10 000 chiffres générés. Combien de batches de 64 ? Combien de temps par DDPM en 1 000 pas ? Par DDIM en 50 pas ? Et par DDIM en 50 pas avec la guidance d'un modèle conditionné ?
2. **Le débit du CPU.** D'après les hypothèses sur MNIST, combien de FLOP par seconde le CPU tient-il ? Compare à l'hypothèse de $10^{11}$.
3. **Pixels et latent.** Combien de valeurs compte une image de $1\,024 \times 1\,024 \times 3$ ? Et son latent ? Quel facteur de compression des **valeurs** ? Combien de tokens voit le transformer ?
4. **Une image.** Le coût d'une passe, puis d'une image (28 pas, guidance). Le temps sur le GPU, puis sur le CPU du workbook.
5. **Le nombre de pas.** Le même modèle échantillonné par DDPM en 1 000 pas, avec guidance : combien de temps sur le GPU ? Et une version distillée qui génère en 4 pas, sans guidance (la guidance est apprise par l'élève) ? Range les trois réglages (1 000, 28 et 4 pas) par coût.
6. **Le latent.** Un transformer de même taille travaillerait directement sur les pixels, en patchs de $8 \times 8$. Combien de tokens ? Par combien le coût des couches denses (proportionnel au nombre de tokens) est-il multiplié ? Et celui de l'attention (proportionnel au carré du nombre de tokens, B3) ?
7. Cite trois coûts ou effets que ton estimation ne compte pas.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B5.9 — La diffusion expliquée en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant comment un modèle de diffusion génère une image, sans jargon.
**Prérequis :** fiche §205.1 · **Parcours :** R

Explique en cinq lignes au plus, à quelqu'un qui n'a jamais programmé, comment un modèle de diffusion fabrique une image. Ton explication doit faire apparaître (1) la destruction : on abîme des images par petites touches de bruit, jusqu'à ne plus rien reconnaître ; (2) l'apprentissage : un réseau apprend à deviner le bruit ajouté, donc à défaire un petit pas ; (3) la génération : partir de bruit pur et défaire les pas un à un ; (4) pourquoi beaucoup de petits pas plutôt qu'un grand ; (5) ce que cela coûte, ou ce qui guide l'image vers ce qu'on demande. Une image de la vie courante est la bienvenue.

### Ex B5.10 — Un générateur peut-il recopier ses images d'entraînement ? Mémorisation, droits et traçabilité ⚖️ ★★ ⏱️ 20 min
**Objectif :** analyser les risques d'un générateur d'images fine-tuné sur des données sensibles ou protégées, et proposer des mesures techniques et juridiques.
**Prérequis :** ch. 27 (usages des images générées, AI Act) · ch. 29 (données personnelles, licences, data card) · fiche §205.8 (mémorisation), §205.9 (transparence et provenance) · **Parcours :** R

Une jeune entreprise française vend aux marques un service d'« images de campagne à la demande ». Elle adapte un modèle texte vers image à poids ouverts (licence permissive) sur trois sources : (1) 1 200 photos de produits et de logos fournies par ses clients, sous contrat ; (2) 300 portraits des salariés d'un client, pris pour une brochure interne, avec l'accord écrit de chacun « pour la brochure » ; (3) 50 000 photos collectées sur des banques d'images en ligne, dont certaines portent un filigrane visible, et dont les conditions d'utilisation interdisent la fouille de textes et de données, en toutes lettres, sans fichier `robots.txt` ni métadonnées qui le disent. L'adaptation fait 40 epochs sur ces images. Réponds dans ta copie.

1. **La mémorisation.** Laquelle des trois sources est la plus exposée à la recopie, et pourquoi (pense aux résultats de Carlini et coll. et aux 40 epochs) ? Comment le vérifier avant la mise en service ? Pourquoi une distance de Fréchet n'en dit rien ?
2. **Les données personnelles.** L'accord des salariés couvre-t-il l'entraînement ? Quels principes du RGPD sont en jeu (finalité, base légale, droit à l'effacement), et que devient ce dernier si le modèle a mémorisé un visage ? Une image générée qui ressemble à un salarié est-elle une donnée personnelle ?
3. **Le droit d'auteur et les marques.** La fouille de textes et de données (directive (UE) 2019/790, art. 4 ; art. L122-5-3 du Code de la propriété intellectuelle) permet-elle la source (3), sachant que la réserve n'est écrite qu'en toutes lettres dans les conditions d'utilisation ? Que disent sur ce point la cour d'appel de Hambourg (Kneschke contre LAION, 10 décembre 2025) et, sur des questions voisines, la High Court de Londres (Getty Images contre Stability AI, 4 novembre 2025) et le tribunal régional de Munich (GEMA contre OpenAI, 11 novembre 2025) ? Que risque un modèle qui reproduit le filigrane d'une banque d'images ?
4. **La transparence.** Quelles obligations de l'article 50 de l'AI Act s'appliquent, à qui, et depuis quand ? Une image de campagne qui montre un salarié réel dans une scène qu'il n'a jamais vécue est-elle un hypertrucage ?
5. **La provenance.** Comment marquer les images produites (métadonnées C2PA, filigrane invisible) ? Que perd chacun des deux outils quand l'image est recadrée, compressée ou capturée à l'écran ?
6. Rédige les cinq lignes « Données d'entraînement et limites » de la fiche du modèle adapté.

### Ex B5.11 — Ho, Jain et Abbeel (2020) : « Denoising Diffusion Probabilistic Models » 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a lancé les modèles de diffusion modernes : son graphe, ses formules, ses choix et ses chiffres.
**Prérequis :** ∂ B5.7 · fiche §205.2 à §205.5 · **Parcours :** M

Lis J. Ho, A. Jain et P. Abbeel, « Denoising Diffusion Probabilistic Models » ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239v2), version 2 de décembre 2020, NeurIPS 2020) : le résumé, la section 1 et la figure 2, les sections 2 et 3, les algorithmes 1 et 2, la section 4 jusqu'à 4.3 (tableaux 1 et 2, figure 6), puis l'annexe B. Réponds dans ta copie.

1. **Le résultat** (résumé, section 1). Quelle qualité d'images les auteurs obtiennent-ils sur CIFAR-10, et avec quelles mesures ? Sur quel point disent-ils eux-mêmes que leurs modèles ne sont pas compétitifs ?
2. **Le graphe** (figure 2). Recopie la figure : quels états, dans quel sens va $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$, dans quel sens va $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ ? Lequel des deux est appris ? Pourquoi les auteurs disent-ils que $\mathbf{x}_T$ n'a « presque aucune information mutuelle » avec $\mathbf{x}_0$ (section 5) ?
3. **Les formules** (équations 4, 6 et 7). Retrouve la forme fermée (∂ B5.6) et le posterior (∂ B5.7). Quels sont les deux choix de $\sigma_t^2$ de la section 3.2, et dans quels cas extrêmes chacun serait-il optimal ?
4. **La paramétrisation** (équations 8 à 12). Pourquoi l'équation 10 suggère-t-elle de faire prédire $\boldsymbol{\varepsilon}$ ? Avec $\sigma_t^2 = \beta_t$, le poids de l'équation 12 vaut $\frac{\beta_t}{2\alpha_t(1 - \bar\alpha_t)}$ : calcule-le au pas $t = 1$ et au pas $t = 1\,000$ du planning linéaire. Que fait l'équation 14 de ces poids, et quel effet les auteurs en attendent-ils (fin de la section 3.4) ?
5. **Les deux algorithmes.** Recopie-les. Dans l'algorithme 2, que vaut $\mathbf{z}$ au dernier pas ? Traduis la ligne 3 de l'algorithme 1 dans la convention des indices de ta librairie.
6. **L'ablation** (tableau 2). Que donnent la prédiction de $\tilde{\boldsymbol{\mu}}$ et celle de $\boldsymbol{\varepsilon}$, avec la borne variationnelle complète puis avec la MSE simple ? Que se passe-t-il quand la variance est apprise ?
7. **Les réglages** (section 4 et annexe B). Le planning, le réseau (famille, normalisation, attention, encodage du temps, nombre de paramètres pour CIFAR-10), l'optimiseur, le learning rate, la taille de batch, l'EMA, le dropout. Combien d'epochs font 800 000 pas de batches de 128 sur les 50 000 images de CIFAR-10 ? Vérifie la durée de 10,6 h à partir des 21 pas par seconde. Combien de temps prend une passe du réseau quand on échantillonne 256 images en 17 s ?
8. **La génération progressive** (section 4.3, figure 6, équation 15). Que montre la figure 6 sur l'ordre dans lequel apparaissent les détails ? Quelle fonction de ta librairie calcule l'équation 15 ? Les auteurs vérifient-ils que leur modèle ne recopie pas CIFAR-10, et comment (section 4.3 et annexe D) ?
9. **Et aujourd'hui** (fiche §205.2, §205.5 à §205.9). Donne trois choses que l'usage de 2026 fait autrement que l'article.

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### B5.E1 — Explique le principe d'un modèle de diffusion, de l'entraînement à la génération 💼 ★★ ⏱️ 10 min
*Fiche §205.1, §205.3, §205.5 · parcours R*

« Vous avez mis un modèle de diffusion dans votre portfolio. Expliquez-moi comment il fonctionne, de l'entraînement jusqu'à l'image générée. »

### B5.E2 — Pourquoi le réseau prédit-il le bruit plutôt que l'image ? 💼 ★★ ⏱️ 10 min
*Fiche §205.3 · parcours R*

« Dans un DDPM, le réseau prédit le bruit ajouté, pas l'image propre. Pourquoi ce choix ? Est-ce le seul possible ? »

### B5.E3 — Comment accélérer l'échantillonnage d'un modèle de diffusion ? 💼 ★★ ⏱️ 10 min
*Fiche §205.6, §205.9 · parcours R*

« Notre générateur d'images met 20 secondes par image sur GPU, et le produit en demande moins de deux. Que pouvez-vous faire, et que risquez-vous ? »

### B5.E4 — Qu'est-ce que la guidance sans classifieur et quel compromis règle-t-elle ? 💼 ★★ ⏱️ 10 min
*Fiche §205.7 · parcours R*

« Expliquez la guidance sans classifieur. À quoi sert le paramètre `guidance_scale`, et comment choisiriez-vous sa valeur ? »

### B5.E5 — Diffusion ou flow matching : qu'est-ce qui a changé dans les générateurs récents ? 💼 ★★ ⏱️ 10 min
*Fiche §205.9 · parcours R*

« Stable Diffusion 3 et FLUX disent utiliser du "rectified flow" plutôt que de la diffusion. Qu'est-ce que ça change, et qu'est-ce qui reste pareil ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`, en trois phases : **les mathématiques de ta librairie et leurs contrôles** (B5.12 à B5.18, avec `linear_beta_schedule`, `cosine_beta_schedule`, `noise_schedule`, `q_sample`, `q_posterior_mean_variance`, `predict_x0_from_eps`, `ddpm_step` et `ddim_step`), **les jouets 2D** (B5.19 et B5.20), **le U-Net sur MNIST et Fashion-MNIST** (B5.21 à B5.31 : entraînement, échantillonnage, guidance, évaluation, script autonome et défi).

| ID | Titre | Type | ★ | ⏱️ | 🚀 |
|---|---|---|---|---|---|
| B5.12 | linear_beta_schedule, cosine_beta_schedule et noise_schedule | 🔨 | ★★ | 20 | |
| B5.13 | Prédire l'allure d'un chiffre bruité aux pas 100, 300 et 600, planning linéaire puis cosinus | 🔮 | ★ | 10 | |
| B5.14 | q_sample : bruiter un batch d'images à des pas différents | 🔨 | ★★ | 20 | |
| B5.15 | Lire ᾱ_t, le rapport signal sur bruit et la grille de bruitage des deux plannings | 📈 | ★ | 15 | |
| B5.16 | q_posterior_mean_variance et predict_x0_from_eps | 🔨 | ★★ | 30 | |
| B5.17 | ddpm_step et ddim_step | 🔨 | ★★★ | 40 | |
| B5.18 | Avec le vrai bruit, DDPM et DDIM retrouvent-ils x_0 ? Contrôle sur des données 2D | 🔬 | ★★ | 25 | |
| B5.19 | Diffusion en 2D : un MLP débruiteur conditionné par le pas de temps, entraîné puis échantillonné | 📦 | ★★ | 30 | |
| B5.20 | Flow matching en 2D : même MLP, cible « vitesse » et intégration d'Euler | 🔬 | ★★★ | 40 | |
| B5.21 | Un U-Net minuscule en PyTorch : blocs convolutifs, connexions de saut, encodage du pas de temps | 📦 | ★★★ | 50 | |
| B5.22 | Le DDPM qui ne produit que du gris : quatre bugs d'indices, de racines et d'échelle | 🐛 | ★★ | 25 | |
| B5.23 | Entraîner un DDPM sur MNIST : boucle, moyenne exponentielle des poids (EMA), courbe de loss | 📦 | ★★★ | 60 | 🚀 |
| B5.24 | Échantillonner : DDPM à 1 000 pas contre DDIM à 50 pas, qualité et temps | 📦 | ★★ | 30 | |
| B5.25 | Nombre de pas DDIM, planning linéaire ou cosinus : distance de Fréchet contre temps de calcul | 🔬 | ★★★ | 45 | 🚀 |
| B5.26 | Prédire l'effet de w = 0, 1, 3 et 7 sur la netteté et la diversité | 🔮 | ★ | 10 | |
| B5.27 | Diffusion conditionnelle et guidance sans classifieur sur Fashion-MNIST | 📦 | ★★★ | 60 | 🚀 |
| B5.28 | Mémorisation : plus proches voisins des échantillons, entraînement sur 500 images contre 60 000 | 🔬 | ★★★ | 40 | 🚀 |
| B5.29 | Distance de Fréchet : situer le DDPM face au VAE (ch. 25) et au DCGAN (ch. 27) sur MNIST, avec le code d'évaluation du mini-projet MP6 | 📦 | ★★★ | 30 | |
| B5.30 | Un échantillonneur autonome : sample.py avec --steps, --seed et --guidance, images et paramètres sauvegardés | 🛠️ | ★★ | 25 | |
| B5.31 | Défi : la plus petite distance de Fréchet sur Fashion-MNIST en au plus 50 pas d'échantillonnage | 🏆 | ★★★★ | 120 | 🚀 |

Le notebook télécharge une fois MNIST et Fashion-MNIST. Le mode rapide entraîne un petit U-Net sur un sous-ensemble en quelques minutes de CPU ; les modes complets marqués 🚀 tournent mieux sur un GPU de Colab.
