# B5 · Modèles de diffusion : un DDPM minimal — fiche de cours

> Ce chapitre bonus n'a pas d'équivalent dans le livre de Glassner : **cette fiche est le cours**. Le livre de 2018 s'arrête aux GAN (ch. 27) et au transfert de style (ch. 28). Depuis 2020, presque tous les générateurs d'images, de sons et de vidéos reposent sur une autre idée : abîmer les données avec du bruit gaussien, un petit peu à chaque pas, jusqu'à ne plus rien reconnaître, puis entraîner un réseau à **défaire un pas de bruit**. En enchaînant ce débruitage depuis du bruit pur, on obtient une image nouvelle. Tu dérives les deux formules qui rendent cela possible (la forme fermée du bruitage et le posterior d'un pas), tu les codes en NumPy dans ta librairie, puis tu entraînes un petit **U-Net** sur MNIST, tu accélères l'échantillonnage (**DDIM**), tu conditionnes par la classe avec la **guidance sans classifieur** et tu évalues honnêtement le résultat face au VAE du ch. 25 et au GAN du ch. 27. La dernière section situe ce DDPM minimal dans le paysage de 2026 : diffusion latente, transformers de diffusion, flow matching.

| | |
|---|---|
| **Livre** | — (chapitre propre au workbook) |
| **Sections** | 205.1 à 205.9 (numérotation interne du workbook : 205 = bonus B5) |
| **Temps total estimé** | ≈ 21 h : lecture de la fiche ≈ 1,8 h, exercices ≈ 18 h, 26 flashcards ≈ 0,9 h |
| **Prérequis** | 0B (produits, exponentielle et logarithme) · ch. 2 (loi normale, variance d'une somme de variables indépendantes) · ch. 4 (règle de Bayes, prior et posterior, pour ∂ B5.7) · ch. 13 (plus proches voisins) · ch. 19 (AdamW, moyenne mobile exponentielle) · ch. 20 (boucle d'entraînement PyTorch, connexion résiduelle, layer norm) · ch. 21 (convolution, stride, convolution transposée) · ch. 23 (sauvegarde d'un modèle) · ch. 25 (débruiteur, VAE, astuce de reparamétrisation, ELBO, espace latent) · ch. 27 (DCGAN, effondrement de mode, distance de Fréchet) · ch. 29 (licences et données personnelles, pour ⚖️ B5.10) · B3 (encodage sinusoïdal des positions, attention croisée, ViT) |
| **Fichiers du chapitre** | `02_exercices.md` (quiz, rappels, papier, réflexion, entretien) · `03_notebook.ipynb` · `04_indices.md` · `05_solutions.md` et `05_solutions.ipynb` · `06_mes_reponses.md` · `flashcards.csv` |
| **mylearn** | `diffusion.py` : `linear_beta_schedule`, `cosine_beta_schedule` et `noise_schedule` (B5.12), `q_sample` (B5.14), `q_posterior_mean_variance` et `predict_x0_from_eps` (B5.16), `ddpm_step` et `ddim_step` (B5.17). NumPy seulement, sans réseau ; le notebook les branche sur un U-Net PyTorch. Il réutilise ta `positional_encoding` (B3) pour le pas de temps et ta `frechet_distance_from_features` (ch. 27) pour l'évaluation |

## Comment utiliser ce chapitre

La fiche et les exercices papier posent les idées et les calculs ; le notebook les met en œuvre, d'abord en NumPy et en 2D, puis sur MNIST et Fashion-MNIST. Les codes de la fiche ont été exécutés le 2 novembre 2026 avec les versions du workbook (Python 3.13, NumPy 2.1, PyTorch 2.11, diffusers 0.40), sur deux cœurs de CPU, avec **d'autres nombres et d'autres données** que les exercices : les tiens, tu les calcules toi-même.

**Ordre conseillé.**
1. Lis 205.1. Fais le quiz Q1 et l'explication 🗣️ B5.9.
2. Lis 205.2. Fais les quiz Q2 et Q3, le rappel R3, les calculs ✏️ B5.2 et B5.3, puis la dérivation ∂ B5.6.
3. Lis 205.3 et 205.4. Fais les quiz Q4 à Q6, le rappel R1 et la dérivation ∂ B5.7.
4. Lis 205.5 et 205.6. Fais les quiz Q7 et Q8, les calculs ✏️ B5.4 et B5.5 et l'estimation 🧮 B5.8.
5. Lis 205.7 et 205.8. Fais les quiz Q9 et Q10, le rappel R2, le calcul ✏️ B5.1 et le cas ⚖️ B5.10.
6. Lis 205.9. Fais le quiz Q11 et la lecture 📄 B5.11. Vérifie tes réponses courtes dans la partie 0 du notebook.
7. Fais le notebook dans l'ordre des ID, en trois phases : **les mathématiques de ta librairie et leurs contrôles** (B5.12 à B5.18), **les jouets 2D** (B5.19 et B5.20), **le U-Net sur MNIST et Fashion-MNIST** (B5.21 à B5.31 : entraînement, échantillonnage, guidance, évaluation, script, défi). Termine par les cinq questions d'entretien.

| Section | Quiz 🧠 et rappels 🔁 | Papier ✏️ ∂ 🧮 et réflexion | Notebook | Entretien 💼 |
|---|---|---|---|---|
| 205.1 L'idée | Q1 | B5.9 | | E1 |
| 205.2 Le processus direct | Q2, Q3, R3 | B5.2, B5.3, B5.6, B5.11 | B5.12, B5.13, B5.14, B5.15, B5.22 | |
| 205.3 Le processus inverse et la loss | Q4, R3 | B5.7, B5.11 | B5.16, B5.19, B5.23 | E1, E2 |
| 205.4 Le débruiteur | Q5, Q6, R1 | B5.11 | B5.19, B5.21 | |
| 205.5 Échantillonnage DDPM | Q7 | B5.4, B5.7, B5.11 | B5.16, B5.17, B5.18, B5.19, B5.22, B5.24 | E1 |
| 205.6 Accélérer | Q7, Q8 | B5.5, B5.8 | B5.17, B5.18, B5.24, B5.25, B5.30, B5.31 | E3 |
| 205.7 Conditionner | Q9 | B5.1 | B5.26, B5.27 | E4 |
| 205.8 Évaluer | Q10, R2 | B5.10 | B5.25, B5.28, B5.29, B5.31 | |
| 205.9 Paysage actuel | Q11 | B5.8, B5.10 | B5.20 | E3, E5 |

## Objectifs d'apprentissage

À la fin de ce chapitre, tu sais :
1. **Dériver** la forme fermée $q(\mathbf{x}_t \mid \mathbf{x}_0)$ et le posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$, et **calculer** à la main un pas DDPM et un pas DDIM.
2. **Implémenter** en NumPy les plannings de bruit, le bruitage et les pas d'échantillonnage, et les **tester** par leurs propriétés (l'aller-retour exact avec le vrai bruit).
3. **Écrire** et **entraîner** en PyTorch un petit U-Net conditionné par le pas de temps, avec une moyenne exponentielle des poids.
4. **Accélérer** l'échantillonnage par DDIM et **mesurer** le compromis entre qualité et nombre de pas.
5. **Conditionner** la génération par une classe et **régler** la guidance sans classifieur.
6. **Évaluer** un générateur (distance de Fréchet, plus proches voisins) et le **comparer** au VAE et au GAN.
7. **Situer** le DDPM dans le paysage actuel : diffusion latente, transformers de diffusion, flow matching.

## L'essentiel en 10 lignes

1. Un **modèle de diffusion** apprend à **débruiter** : on abîme une donnée par $T$ petits ajouts de bruit gaussien (le **processus direct**, fixé), et un réseau apprend à remonter la chaîne (le **processus inverse**). Partir de bruit pur et remonter donne une donnée nouvelle.
2. Chaque pas fait $\mathbf{x}_t = \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\varepsilon}$ ; le **planning de bruit** $\beta_1, \dots, \beta_T$ dit combien de bruit on ajoute à chaque pas.
3. Les variances de gaussiennes indépendantes s'additionnent, donc les $t$ premiers pas n'en font qu'un : $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}$, avec $\bar\alpha_t = \prod_{s \le t}(1 - \beta_s)$. On bruite n'importe quelle image à n'importe quel pas en un seul calcul.
4. Le **rapport signal sur bruit** $\bar\alpha_t / (1 - \bar\alpha_t)$ décroît de très grand à presque 0 ; le planning **cosinus** le fait décroître plus régulièrement que le planning **linéaire**.
5. Le réseau $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$ apprend à **prédire le bruit** ajouté, par une simple MSE : c'est la **loss simplifiée**, une version repondérée de l'ELBO du VAE.
6. Sachant $\mathbf{x}_0$, le pas en arrière $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ est une gaussienne connue (le **posterior**) ; l'échantillonnage **DDPM** y remplace $\mathbf{x}_0$ par l'estimation qu'en donne le réseau, et ajoute un peu de bruit : $T$ passes du réseau, un tirage à chaque pas.
7. Le débruiteur est un **U-Net** : un encodeur, un décodeur et des **connexions de saut** entre les deux ; le pas $t$ y entre par un **encodage sinusoïdal**, comme une position dans un Transformer.
8. **DDIM** réutilise le même réseau sans réentraînement : il estime $\hat{\mathbf{x}}_0$, puis le rebruite directement au niveau d'un pas bien antérieur, sans tirage. 50 pas au lieu de 1 000, et un même bruit de départ donne toujours la même image.
9. La **guidance sans classifieur** entraîne un seul réseau avec et sans la condition, puis extrapole : $(1 + w)\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, c) - w\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, \varnothing)$ ; $w$ échange de la diversité contre de la fidélité à la condition.
10. Aujourd'hui, la diffusion se fait dans l'**espace latent** d'un autoencodeur, avec un **transformer** à la place du U-Net, et souvent par **flow matching** (une ligne droite entre la donnée et le bruit) ; on l'évalue par des distances de Fréchet et en cherchant ce qu'elle **recopie** de ses données d'entraînement.

> 📐 **Conventions de la fiche** — Dans les formules, comme dans l'article de Ho et coll. (2020), les pas vont de $t = 1$ à $t = T$ ; $\mathbf{x}_0$ est la donnée propre, $\mathbf{x}_T$ le bruit presque pur, et $\bar\alpha_0 = 1$. **Dans le code** (ta librairie, PyTorch, diffusers), les pas sont des **indices de 0 à $T - 1$** : le pas $t$ de l'article est l'indice $t - 1$, `alpha_bars[i]` vaut $\bar\alpha_{i+1}$, et `alpha_bars_prev[0]` vaut 1. Chaque exemple chiffré donne les deux (« pas $t = 4$, indice 3 »). $\boldsymbol{\varepsilon}$ est le vrai bruit, $\boldsymbol{\varepsilon}_\theta$ celui que prédit le réseau, $\mathbf{z}$ le bruit tiré pendant l'échantillonnage ; $\mathcal{N}(\boldsymbol{\mu}, \sigma^2\mathbf{I})$ est une gaussienne de moyenne $\boldsymbol{\mu}$ dont chaque coordonnée a la variance $\sigma^2$. Une donnée a $p$ valeurs (784 pixels pour MNIST, ramenés dans $[-1, 1]$), un batch en contient $N$ (comme dans les docstrings). En 205.9, le temps du flow matching s'écrit $s \in [0, 1]$, pour ne pas le confondre avec le pas $t$ (en 205.2, le $s$ du planning cosinus est un petit décalage, celui des docstrings) ; $q$ et $p_\theta$ sont des lois, $p$ seul un nombre de valeurs. $\ln$ est le logarithme naturel. **Écarts au §6** : $\mathbf{x}_t$ désigne la donnée **au pas $t$** (pas l'exemple numéro $t$), et $\sigma_t$ un écart-type d'échantillonnage (pas une activation).

---

## 205.1 · L'idée : détruire par du bruit, apprendre à reconstruire ⏩

### Détruire est facile

Prends une image de MNIST, ramenée dans $[-1, 1]$. Ajoute à chaque pixel un tout petit bruit gaussien, et recommence mille fois. Après quelques dizaines de pas, l'image a du grain ; après quelques centaines, on la devine à peine ; à la fin, il ne reste qu'un nuage de pixels gaussiens indépendants, le même quelle que soit l'image de départ. Cette destruction ne demande aucun apprentissage : c'est le **processus direct** (*forward process*), fixé une fois pour toutes.

La figure le montre sur des données en 2D, plus faciles à regarder qu'une image : un nuage en spirale se dilue progressivement dans une gaussienne. Le mot « diffusion » vient de là : comme une goutte d'encre dans l'eau, la structure s'étale jusqu'à disparaître.

![Six nuages de points : à gauche, une spirale nette (les données) ; vers la droite, la même spirale de plus en plus floue, jusqu'à une tache gaussienne ronde (le bruit pur). Une flèche en haut, de gauche à droite : le processus direct, fixé, ajoute du bruit ; une flèche en bas, de droite à gauche : le processus inverse, appris, en retire](figures/chaine.png)

### Reconstruire s'apprend

Le **processus inverse** (*reverse process*) va dans l'autre sens : partir du bruit pur et retirer, pas après pas, un peu de bruit, jusqu'à une donnée nette. Si chaque pas direct n'ajoute qu'**un peu** de bruit, chaque pas inverse n'a qu'**un peu** à retirer, et c'est une tâche qu'un réseau sait apprendre : regarder une image légèrement bruitée et dire quel bruit s'y trouve. Le même réseau sert à tous les pas ; il reçoit le numéro du pas $t$ en plus de l'image, pour savoir combien de bruit il doit s'attendre à trouver.

Pour **générer**, on tire un bruit pur $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ et l'on applique le pas inverse $T$ fois. Chaque bruit de départ donne une image différente, qui ne figure pas dans le dataset : c'est ce qu'on attend d'un générateur.

Pourquoi tant de petits pas plutôt qu'un seul grand ? Parce qu'un débruitage en un coup, du bruit pur à l'image, n'a pas de bonne réponse unique : mille images différentes auraient pu donner ce bruit, et un réseau entraîné par MSE rendrait leur **moyenne**, une tache floue (encadré ci-dessous). Un petit pas, lui, a une réponse presque certaine : l'image d'avant ressemble beaucoup à l'image d'après (🧠 Q2).

> 🧮 **Rappel maths — la MSE apprend une moyenne** — Si la même entrée $x$ peut avoir plusieurs cibles $y$, la prédiction $\hat{y}$ qui minimise l'erreur quadratique moyenne $\mathbb{E}[(y - \hat{y})^2]$ est la **moyenne** des cibles possibles, $\mathbb{E}[y \mid x]$ (ch. 9 : c'est pour cela qu'une régression prédit une moyenne). Exemple : pour des cibles $-1$ et $+1$ équiprobables, $\hat{y} = 0$ donne une MSE de 1, et $\hat{y} = 1$ une MSE de 2. Un débruiteur entraîné par MSE prédit donc la moyenne de tout ce qui est compatible avec ce qu'il voit : nette quand il y a peu de bruit, floue quand il y en a beaucoup.

### Trois façons de fabriquer une image

> 🔗 **Lien avec les ch. 25 et 27** — Le **débruiteur** du ch. 25 retirait un bruit **d'un seul niveau** pour rendre une image propre ; le modèle de diffusion apprend tous les niveaux à la fois, du grain léger au bruit pur, et c'est en les enchaînant qu'il génère. Le **VAE** encode l'image en un petit code $\mathbf{z}$, appris, puis décode : une seule passe du décodeur génère. La diffusion ressemble à un VAE à $T$ étages dont l'« encodeur » est fixé (le bruitage) et dont tous les codes $\mathbf{x}_1, \dots, \mathbf{x}_T$ ont la taille de l'image ; sa loss vient de la même ELBO (205.3). Le **GAN** génère aussi en une passe, mais apprend contre un discriminateur, sans loss de reconstruction : rapide à l'usage, instable à entraîner, sujet à l'effondrement de mode. La diffusion s'entraîne par une simple MSE, stable, et couvre bien les modes ; elle paie cela par un échantillonnage en **beaucoup de passes** du réseau (🧠 Q1).

L'idée vient de J. Sohl-Dickstein et ses coauteurs (2015), qui s'inspiraient de la physique hors d'équilibre : détruire lentement la structure d'une distribution par une diffusion, puis apprendre la diffusion inverse. Elle est restée confidentielle jusqu'à J. Ho, A. Jain et P. Abbeel (2020, 📄 B5.11), qui ont simplifié la loss, choisi de **prédire le bruit**, et obtenu sur CIFAR-10 un FID de 3,17, meilleur que les GAN de l'époque. Leur méthode s'appelle **DDPM** (*denoising diffusion probabilistic models*) ; c'est celle de ce chapitre.

> 🕰️ **Mise à jour (2026) — la génération d'images** — **Le livre :** les GAN (ch. 27) et le transfert de style (ch. 28) ; rien sur la diffusion, publiée deux ans plus tard. · **Aujourd'hui :** les générateurs d'images et de vidéos à poids ouverts documentés sont des **transformers de diffusion** entraînés dans l'espace latent d'un autoencodeur, souvent par **flow matching** : Stable Diffusion 3 (2024), FLUX.1 (2024) et FLUX.2 (novembre 2025, 32 milliards de paramètres pour la version à poids ouverts), Qwen-Image-2.0 (mai 2026), Wan pour la vidéo ; des versions **distillées** génèrent en 1 à 4 pas. En tête du classement d'Artificial Analysis consulté le 2 novembre 2026, on trouve des modèles fermés (OpenAI, Google, Microsoft) dont l'architecture n'est pas publiée en détail ; OpenAI a décrit la génération d'images de GPT-4o (mars 2025) comme **autorégressive**, et non comme une diffusion. · **Faut-il quand même l'apprendre ?** Oui : le DDPM reste la porte d'entrée, et ses idées (bruitage, prédiction du bruit, posterior, guidance) se retrouvent dans tous ces modèles (205.9). · *Sources :* [Esser et coll., arXiv:2403.03206](https://arxiv.org/abs/2403.03206v1) ; [Black Forest Labs, « FLUX.2 »](https://bfl.ai/blog/flux-2) ; [Qwen-Image-2.0, arXiv:2605.10730](https://arxiv.org/abs/2605.10730v1) ; [Wan, arXiv:2503.20314](https://arxiv.org/abs/2503.20314) ; [Artificial Analysis, classement texte vers image](https://artificialanalysis.ai/text-to-image/arena/leaderboard-text) ; [OpenAI, fiche système de la génération d'images de GPT-4o](https://openai.com/index/gpt-4o-image-generation-system-card-addendum/).

---

## 205.2 · Le processus direct : planning de bruit, ᾱ_t et forme fermée q(x_t | x_0) ⏩

### Un pas de bruit

À chaque pas, on rétrécit un peu l'image et l'on ajoute un peu de bruit :

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\big(\sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I}\big), \qquad \text{soit} \qquad \mathbf{x}_t = \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).$$

Le nombre $\beta_t \in (0, 1)$ est la **variance du bruit** ajouté au pas $t$, petite (de l'ordre de $10^{-4}$ à $10^{-2}$). La suite $\beta_1, \dots, \beta_T$ s'appelle le **planning de bruit** (*noise schedule*). La seconde écriture est l'astuce de reparamétrisation du ch. 25 : un tirage gaussien s'écrit « moyenne + écart-type × bruit standard ».

**Pourquoi rétrécir par $\sqrt{1 - \beta_t}$ ?** Pour garder la variance stable. Si une coordonnée de $\mathbf{x}_{t-1}$ a la variance 1, celle de $\mathbf{x}_t$ vaut $(1 - \beta_t) \cdot 1 + \beta_t = 1$ (la variance d'une somme de variables indépendantes est la somme des variances, ch. 2, et multiplier par $a$ multiplie la variance par $a^2$). Sans ce facteur, la variance grandirait à chaque pas et le bruit final n'aurait pas d'échelle fixe. On parle de diffusion **à variance préservée**. On note aussi $\alpha_t = 1 - \beta_t$.

> 🧮 **Rappel maths — somme de deux gaussiennes indépendantes** — Si $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ et $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ sont **indépendantes**, alors $X + Y \sim \mathcal{N}(\mu_1 + \mu_2,\ \sigma_1^2 + \sigma_2^2)$ : les moyennes s'ajoutent (toujours vrai), les **variances** s'ajoutent (grâce à l'indépendance, ch. 2), et la somme reste **gaussienne** (propriété propre aux gaussiennes, admise ici). Ce sont les variances qui s'ajoutent, pas les écarts-types : $\mathcal{N}(1, 0{,}3^2) + \mathcal{N}(-2, 0{,}4^2) = \mathcal{N}(-1, 0{,}5^2)$, car $0{,}09 + 0{,}16 = 0{,}25$, et l'écart-type vaut 0,5, pas 0,7. Et $aX \sim \mathcal{N}(a\mu_1, a^2\sigma_1^2)$.

```python
import numpy as np

rng = np.random.default_rng(0)
s = rng.normal(1.0, 0.3, 1_000_000) + rng.normal(-2.0, 0.4, 1_000_000)   # two independent Gaussians
print(s.mean().round(2), s.std().round(2))                                # -1.0 0.5
```

### La forme fermée

Enchaîne deux pas. $\mathbf{x}_{t-1}$ s'écrit $\sqrt{\alpha_{t-1}}\,\mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}}\,\boldsymbol{\varepsilon}_{t-1}$, et $\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\,\boldsymbol{\varepsilon}_t$ : en remplaçant, $\mathbf{x}_t$ est $\sqrt{\alpha_t \alpha_{t-1}}\,\mathbf{x}_{t-2}$ plus **deux** bruits gaussiens indépendants. Par la règle de l'encadré, ces deux bruits n'en font qu'un, et sa variance se calcule. En répétant jusqu'à $\mathbf{x}_0$, on obtient la **forme fermée** (*closed form*) du processus direct :

$$\bar\alpha_t = \prod_{s=1}^{t} \alpha_s, \qquad q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\ (1 - \bar\alpha_t)\,\mathbf{I}\big), \qquad \mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}.$$

Le calcul complet, avec la récurrence, est la dérivation ∂ B5.6. Retiens le résultat et ce qu'il change en pratique : **on n'a jamais besoin de simuler la chaîne**. Pour fabriquer un exemple d'entraînement au pas 700, on tire un seul $\boldsymbol{\varepsilon}$ et l'on calcule $\mathbf{x}_{700}$ directement. Le coefficient $\sqrt{\bar\alpha_t}$ dit **quelle part de l'image** reste, $\sqrt{1 - \bar\alpha_t}$ **quelle part de bruit** il y a, et leurs carrés font 1 : $\bar\alpha_t$ est la part de la variance qui vient de l'image (🧠 Q3).

### Un planning jouet, à la main

Prends $T = 5$ et un planning **linéaire** de $\beta_1 = 0{,}02$ à $\beta_5 = 0{,}10$ (rien à voir avec les vrais réglages, juste pour calculer). Alors $\beta = (0{,}02 ;\ 0{,}04 ;\ 0{,}06 ;\ 0{,}08 ;\ 0{,}10)$, $\alpha = (0{,}98 ;\ 0{,}96 ;\ 0{,}94 ;\ 0{,}92 ;\ 0{,}90)$ et, par produits successifs :

| pas $t$ (indice) | 1 (0) | 2 (1) | 3 (2) | 4 (3) | 5 (4) |
|---|---|---|---|---|---|
| $\bar\alpha_t$ | 0,98 | 0,9408 | 0,8844 | 0,8136 | 0,7322 |
| $\sqrt{\bar\alpha_t}$ (part de l'image) | 0,9899 | 0,9700 | 0,9404 | 0,9020 | 0,8557 |
| $\sqrt{1 - \bar\alpha_t}$ (part du bruit) | 0,1414 | 0,2433 | 0,3401 | 0,4317 | 0,5174 |

Par exemple, au pas 4 (indice 3), une valeur de pixel $x_0 = 0{,}6$ et un bruit $\varepsilon = -0{,}5$ donnent $x_4 = 0{,}9020 \times 0{,}6 + 0{,}4317 \times (-0{,}5) = 0{,}3253$. Avec si peu de pas et des $\beta$ si petits, il reste beaucoup d'image à la fin : ce planning jouet ne détruit rien. Les vrais plannings prennent $T = 1\,000$.

```python
betas = np.linspace(0.02, 0.10, 5)                 # toy linear schedule, T = 5
alpha_bars = np.cumprod(1 - betas)                 # alpha_bar_t, indices 0..4
print(alpha_bars.round(4))                         # [0.98   0.9408 0.8844 0.8136 0.7322]
x4 = np.sqrt(alpha_bars[3]) * 0.6 + np.sqrt(1 - alpha_bars[3]) * (-0.5)
print(round(x4, 4))                                # 0.3253  (step t = 4 is index 3)
```

### Le planning linéaire de Ho et coll.

Ho et ses coauteurs prennent $T = 1\,000$ et des $\beta_t$ qui croissent **linéairement** de $10^{-4}$ à $0{,}02$, choisis « petits devant des données dans $[-1, 1]$ », pour que chaque pas inverse reste presque gaussien, et pour qu'il reste aussi peu d'image que possible dans $\mathbf{x}_T$. C'est `linear_beta_schedule(1000)` de ta librairie (🔨 B5.12), identique à `np.linspace(1e-4, 0.02, 1000)`.

Pour savoir ce qu'il reste au bout, une approximation suffit : pour $\beta$ petit, $\ln(1 - \beta) \approx -\beta$, donc

$$\ln \bar\alpha_T = \sum_{t=1}^{T} \ln(1 - \beta_t) \approx -\sum_{t=1}^{T} \beta_t, \qquad \bar\alpha_T \approx e^{-\sum_t \beta_t}.$$

Attention : ce résultat dépend de $T$. Avec les mêmes bornes mais $T = 200$, la somme des $\beta$ vaut $200 \times 0{,}01005 = 2{,}01$, et $\bar\alpha_{200} \approx e^{-2{,}01} \approx 0{,}134$ (le produit exact donne 0,1322) : il reste $\sqrt{0{,}132} \approx 36\,\%$ de l'image au « bout » de la chaîne, et l'on ne part plus d'un bruit pur. Raccourcir la chaîne demande donc de **grossir les $\beta$** : le code de Nichol et Dhariwal multiplie les deux bornes par $1\,000 / T$. L'exercice ✏️ B5.2 te fait le calcul pour $T = 4$ et $T = 1\,000$.

```python
betas_200 = np.linspace(1e-4, 0.02, 200)           # same bounds, T = 200
print(np.prod(1 - betas_200).round(4), np.exp(-betas_200.sum()).round(4))   # 0.1322 0.134
```

### Le planning cosinus de Nichol et Dhariwal

A. Nichol et P. Dhariwal (2021) ont remarqué que le planning linéaire détruit l'information **plus vite que nécessaire** : une bonne partie de la fin de la chaîne est déjà presque du bruit pur, si bien qu'on pouvait sauter jusqu'à 20 % du processus inverse sans beaucoup abîmer le FID. Ils définissent directement $\bar\alpha$, par une courbe en cosinus qui baisse lentement au début, à peu près linéairement au milieu, et lentement à la fin :

$$f(u) = \cos^2\!\left(\frac{u + s}{1 + s} \cdot \frac{\pi}{2}\right), \qquad \bar\alpha(u) = \frac{f(u)}{f(0)}, \qquad u = \frac{t}{T} \in [0, 1].$$

On en tire les $\beta$ par rapport de deux valeurs consécutives, $\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}$, écrêtés à $0{,}999$ : près de $u = 1$, $\bar\alpha$ tend vers 0 et ce rapport vers 1 (le dernier $\beta$ vaut exactement le plafond). Le petit décalage $s = 0{,}008$ évite des $\beta$ quasi nuls au début : les auteurs l'ont choisi pour que $\sqrt{\beta}$ du premier pas soit un peu plus petit que l'écart entre deux niveaux de gris, $1/127{,}5$ (les entiers de 0 à 255 ramenés dans $[-1, 1]$) ; avec $T = 1\,000$, le premier $\beta$ vaut environ $4{,}1 \times 10^{-5}$. Le cos² est un choix « arbitraire », disent-ils : d'autres courbes de même allure marcheraient aussi.

Dans ta librairie, `cosine_beta_schedule(T)` évalue $f$ aux $T + 1$ points $u = 0, 1/T, \dots, 1$, puis fait les rapports : l'indice $i$ (de 0 à $T - 1$) reçoit $\beta = \min\big(1 - \bar\alpha((i+1)/T) / \bar\alpha(i/T),\ 0{,}999\big)$, et la docstring écrit $\bar\alpha(t)$ pour la valeur de la courbe au point $t/T$. Les deux plannings sont ceux de diffusers (`beta_schedule="linear"` et `"squaredcos_cap_v2"`) : on l'a vérifié à $10^{-7}$ près (diffusers calcule en float32).

![Deux graphiques en fonction de t/T. À gauche, ᾱ_t : la courbe du planning linéaire (T = 1000) tombe vite vers 0 et y reste pendant toute la seconde moitié ; la courbe cosinus descend plus lentement, presque en ligne droite au milieu ; la courbe linéaire à T = 200 avec les mêmes β descend lentement et finit vers 0,13. À droite, le logarithme du rapport signal sur bruit : les trois courbes partent d'environ 9, la linéaire à T = 1000 descend le plus vite, la cosinus plonge brusquement tout à la fin](figures/plannings.png)

### Le rapport signal sur bruit

Pour comparer des plannings, on regarde la part de signal et la part de bruit à chaque pas par le **rapport signal sur bruit** (*signal-to-noise ratio*, SNR) :

$$\operatorname{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}.$$

C'est le rapport de la variance qui vient de l'image ($\bar\alpha_t$, pour une image de variance 1) à celle qui vient du bruit. Il est énorme au début (49 au pas 1 du planning jouet), vaut 1 quand les deux parts sont égales, et tend vers 0 à la fin (2,73 au pas 5 du jouet, qui ne va donc pas assez loin). Comme il couvre plusieurs ordres de grandeur, on trace son **logarithme** (figure, à droite). Le planning cosinus fait décroître $\ln \operatorname{SNR}$ presque linéairement au milieu de la chaîne, alors que le linéaire passe beaucoup de pas dans une zone où il ne reste presque plus rien à voir. Le SNR est la bonne coordonnée pour comparer deux plannings, deux résolutions ou deux méthodes : ✏️ B5.3 l'utilise pour dire à quel pas un chiffre devient illisible, et 📈 B5.15 te fait lire les courbes à côté d'une grille d'images bruitées.

> ⚠️ **Piège classique — un SNR final qui n'est pas nul** — Ni le planning linéaire, ni bien des plannings publiés n'atteignent exactement $\bar\alpha_T = 0$. S. Lin et coll. (2024) ont montré que le planning de Stable Diffusion laissait assez de signal au dernier pas pour que le modèle apprenne à garder la luminosité moyenne de l'image : partant d'un bruit pur à la génération, il ne savait produire que des images de luminosité moyenne, ni très claires ni très sombres. Le remède : un planning dont le SNR final est nul, et commencer l'échantillonnage au **dernier** indice. Avant d'entraîner, affiche $\bar\alpha$ au dernier indice de ton planning.

### Bruiter un batch : `noise_schedule` et `q_sample`

Toutes les constantes dérivées des $\beta$ se calculent une fois. `noise_schedule(betas)` (🔨 B5.12) les range dans un dictionnaire de tableaux de forme `(T,)`, aux clés fixées par la docstring : `betas`, `alphas`, `alpha_bars`, `alpha_bars_prev` ($\bar\alpha_{t-1}$, qui vaut 1 à l'indice 0), `sqrt_alpha_bars`, `sqrt_one_minus_alpha_bars`, et les trois constantes du posterior (205.3) : `posterior_variance`, `posterior_mean_coef1` et `posterior_mean_coef2`. Deux propriétés servent de contrôle : `alpha_bars` décroît strictement en restant dans $(0, 1)$, et `posterior_variance[0]` vaut 0.

`q_sample(x0, t, alpha_bars, noise)` (🔨 B5.14) applique la forme fermée à un batch où **chaque exemple a son propre pas**. La difficulté est une question de formes : `t` a la forme `(N,)`, `alpha_bars[t]` aussi, mais `x0` a la forme `(N, 1, 28, 28)` ; il faut redonner au coefficient la forme `(N, 1, 1, 1)` pour que le broadcasting (0A) l'applique à tous les pixels de **son** image.

```python
x0 = rng.uniform(-1, 1, size=(3, 1, 4, 4))         # 3 tiny "images" in [-1, 1]
t = np.array([0, 2, 4])                            # one step index per image
eps = rng.standard_normal(x0.shape)
a = np.sqrt(alpha_bars[t]).reshape(-1, 1, 1, 1)    # (3,) -> (3, 1, 1, 1)
b = np.sqrt(1 - alpha_bars[t]).reshape(-1, 1, 1, 1)
x_t = a * x0 + b * eps
print(a.shape, x_t.shape)                          # (3, 1, 1, 1) (3, 1, 4, 4)
```

> ⚠️ **Piège classique — `alpha_bars[t]` sans remise en forme** — `np.sqrt(alpha_bars[t]) * x0` avec `t` de forme `(3,)` et `x0` de forme `(3, 1, 4, 4)` aligne le coefficient sur le **dernier** axe (la largeur) : NumPy lève une erreur si les tailles diffèrent, mais si l'image a justement 3 colonnes, le calcul passe sans erreur et donne n'importe quoi. Remets toujours le coefficient en forme `(N, 1, …, 1)`, et teste avec des tailles toutes différentes.

---

## 205.3 · Le processus inverse et la loss simplifiée : prédire le bruit, lien avec l'ELBO ⏩

### Ce qu'on voudrait : $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$

Pour générer, il faudrait savoir tirer $\mathbf{x}_{t-1}$ connaissant $\mathbf{x}_t$. Cette loi dépend de toutes les images possibles, on ne la connaît pas. Mais on sait deux choses. D'abord, quand $\beta_t$ est petit, elle est **presque gaussienne** (un résultat que Sohl-Dickstein et ses coauteurs reprennent de la théorie des processus de diffusion) : on peut donc l'approcher par une gaussienne dont un réseau donne la moyenne,

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\big(\boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\ \sigma_t^2 \mathbf{I}\big).$$

Ensuite, si l'on connaît **aussi** l'image propre $\mathbf{x}_0$, la loi du pas en arrière se calcule exactement.

### Le posterior : un pas en arrière quand on connaît $\mathbf{x}_0$

La règle de Bayes (ch. 4) retourne le pas direct : $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ est proportionnel à $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \times q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)$, la vraisemblance du dernier pas fois le « prior » que donne la forme fermée. Vus comme des fonctions de $\mathbf{x}_{t-1}$, ces deux facteurs sont des densités gaussiennes, et leur produit en est une aussi (encadré). Le résultat, qu'on appelle le **posterior** du pas $t$ (Ho et coll., éq. 6 et 7), est

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\big(\tilde{\boldsymbol{\mu}}_t,\ \tilde\beta_t \mathbf{I}\big), \qquad \tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\,\mathbf{x}_t, \qquad \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t.$$

La moyenne est un **mélange pondéré** de l'image propre et de l'image bruitée ; les deux coefficients sont `posterior_mean_coef1` et `posterior_mean_coef2` de `noise_schedule`, et $\tilde\beta_t$ est `posterior_variance`. Au premier pas ($t = 1$, indice 0), $\bar\alpha_0 = 1$ : le premier coefficient vaut 1, le second 0 et la variance 0 ; connaissant $\mathbf{x}_0$, il n'y a plus aucun doute sur lui. Et $\tilde\beta_t \le \beta_t$ toujours, puisque $1 - \bar\alpha_{t-1} < 1 - \bar\alpha_t$ : savoir où l'on va réduit l'incertitude du pas. La dérivation (en dimension 1, en complétant le carré) est l'exercice ∂ B5.7.

> 🧮 **Rappel maths — produit de deux densités gaussiennes** — En dimension 1, le produit de deux densités gaussiennes $\mathcal{N}(x;\ m_1, v_1)$ et $\mathcal{N}(x;\ m_2, v_2)$, vu comme une fonction de $x$, est proportionnel à la densité d'**une** gaussienne $\mathcal{N}(x;\ m, v)$ avec
> $$\frac{1}{v} = \frac{1}{v_1} + \frac{1}{v_2}, \qquad m = v\left(\frac{m_1}{v_1} + \frac{m_2}{v_2}\right).$$
> Les **inverses des variances** (les « précisions ») s'ajoutent, et la moyenne est la moyenne des deux moyennes pondérée par leurs précisions : la plus sûre des deux tire plus fort. Exemple : $\mathcal{N}(x;\ 2, 1) \times \mathcal{N}(x;\ 5, 0{,}5)$ donne $1/v = 1 + 2 = 3$, $v = 1/3$, et $m = \frac{1}{3}(2 + 10) = 4$, plus près de 5, la plus précise. On le démontre en développant les deux exposants, $-\frac{(x - m_1)^2}{2v_1} - \frac{(x - m_2)^2}{2v_2}$, puis en **complétant le carré** : on regroupe les termes en $x^2$ et en $x$ sous la forme $-\frac{(x - m)^2}{2v} + \text{constante}$. La constante disparaît dans la normalisation, car une densité intègre à 1.

On vérifie la formule du posterior sur le planning jouet, au pas 4 (indice 3), avec $x_0 = 0{,}6$ et $x_4 = 0{,}3253$ : $\tilde\mu_4 = 0{,}4036 \times 0{,}6 + 0{,}5951 \times 0{,}3253 = 0{,}4358$ et $\tilde\beta_4 = 0{,}0496$. Une vérification numérique de la règle de Bayes, sur une grille fine, redonne les mêmes nombres ; c'est l'un des tests de ta `q_posterior_mean_variance` (🔨 B5.16) :

```python
from scipy.stats import norm

ab, ab_prev, beta = alpha_bars[3], alpha_bars[2], betas[3]       # step t = 4 is index 3
x0, xt = 0.6, np.sqrt(ab) * 0.6 + np.sqrt(1 - ab) * (-0.5)
g = np.linspace(-5, 5, 200_001)                                    # candidate values of x_{t-1}
dens = norm.pdf(xt, np.sqrt(1 - beta) * g, np.sqrt(beta)) * norm.pdf(g, np.sqrt(ab_prev) * x0, np.sqrt(1 - ab_prev))
dens /= np.trapezoid(dens, g)                                      # Bayes: normalise the product
m = np.trapezoid(g * dens, g)
print(round(m, 4), round(np.trapezoid((g - m) ** 2 * dens, g), 4))  # 0.4358 0.0496
```

### Prédire le bruit

À la génération, on ne connaît pas $\mathbf{x}_0$ : c'est lui qu'on cherche. Mais la forme fermée le relie à $\mathbf{x}_t$ et au bruit : si un réseau devine le bruit $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$ contenu dans $\mathbf{x}_t$, on en déduit une estimation de l'image propre en inversant la formule,

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar\alpha_t}},$$

c'est `predict_x0_from_eps` (🔨 B5.16), et l'on met $\hat{\mathbf{x}}_0$ dans la moyenne du posterior. En simplifiant, la moyenne du pas inverse ne dépend plus que de $\mathbf{x}_t$ et du bruit prédit :

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)\right).$$

Ho et ses coauteurs ont comparé : faire prédire au réseau $\mathbf{x}_0$ donnait de moins bons échantillons ; prédire $\tilde{\boldsymbol{\mu}}_t$ ou le bruit revenait à peu près au même avec l'ELBO complète (plus bas), mais avec la loss simplifiée, prédire le bruit marchait nettement mieux. Une raison intuitive : le bruit est toujours une gaussienne standard, de la même échelle à tous les pas, alors que la moyenne à prédire change d'échelle avec $t$. Un réseau apprend mieux une cible de taille constante.

> 🔗 **Lien avec la reparamétrisation (ch. 25)** — C'est la même écriture qu'au ch. 25 : $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}$ est un tirage de $\mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0, 1 - \bar\alpha_t)$ écrit « moyenne + écart-type × bruit ». Dans le VAE, ce bruit tiré à part laissait passer le gradient ; ici, il devient la **cible** du réseau (🔁 R3).

### La loss simplifiée

D'où l'entraînement (algorithme 1 de Ho et coll.), répété jusqu'à convergence :

```text
x0  ← une image du dataset (pixels dans [-1, 1])
t   ← un indice uniforme dans {0, …, T-1}         (un par image du batch)
eps ← un bruit N(0, I) de la forme de x0
x_t ← sqrt(alpha_bars[t]) * x0 + sqrt(1 - alpha_bars[t]) * eps      (q_sample)
pas de gradient sur la loss  || eps - eps_theta(x_t, t) ||²
```

$$L_{\text{simple}} = \mathbb{E}_{t,\ \mathbf{x}_0,\ \boldsymbol{\varepsilon}}\left[\big\lVert \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t) \big\rVert^2\right].$$

C'est une **MSE sur le bruit** (`F.mse_loss(model(x_t, t), eps)`) : pas d'adversaire, pas de terme KL à doser, une loss qui baisse de façon régulière. Chaque batch mélange des pas tirés au hasard, si bien qu'un seul réseau apprend tous les niveaux de bruit. La loss n'atteint jamais 0, et sa valeur dépend du pas : aux grands $t$, $\mathbf{x}_t$ est presque tout bruit et $\boldsymbol{\varepsilon} \approx \mathbf{x}_t$ se devine bien ; aux petits $t$, isoler au pixel près le petit bruit d'une image presque propre est délicat. Une courbe de loss de diffusion descend vite puis se stabilise, et c'est la qualité des **échantillons**, pas la valeur de la loss, qui dit quand arrêter (📦 B5.23).

### Le lien avec l'ELBO

D'où vient cette loss ? Comme le VAE (ch. 25), le modèle de diffusion maximise une borne inférieure de $\ln p_\theta(\mathbf{x}_0)$, l'ELBO, où les $\mathbf{x}_1, \dots, \mathbf{x}_T$ jouent le rôle des latents et le processus direct celui de l'encodeur (fixé, sans paramètres). Son opposé se décompose en une somme de termes, un par pas :

$$L_{\text{VLB}} = \underbrace{\mathrm{KL}\big(q(\mathbf{x}_T \mid \mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\big)}_{\text{constante}} + \sum_{t=2}^{T} \underbrace{\mathrm{KL}\big(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big)}_{L_{t-1}} \; - \ln p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1).$$

Chaque terme $L_{t-1}$ compare le **posterior exact** au **pas appris** : deux gaussiennes de même variance, dont la KL se réduit à l'écart de leurs moyennes au carré (ch. 6 et 25). Avec la paramétrisation par le bruit, Ho et ses coauteurs obtiennent (leur éq. 12)

$$L_{t-1} = \mathbb{E}\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1 - \bar\alpha_t)}\,\big\lVert \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t) \big\rVert^2\right] + \text{constante}.$$

$L_{\text{simple}}$ est donc l'ELBO **dont on a retiré les poids** devant chaque pas. Ces poids sont bien plus grands pour les petits $t$ que pour les grands ; les retirer donne plus d'importance aux pas très bruités, où le réseau doit inventer la forme de l'image (les débruitages « plus difficiles », disent les auteurs), et ils ont observé de meilleurs échantillons ainsi, au prix d'une vraisemblance un peu moins bonne. Le premier terme ne dépend d'aucun paramètre : on l'ignore.

> 🕰️ **Mise à jour (2026) — paramétrisations et pondérations** — **Le livre :** absent, comme toute la diffusion. · **Aujourd'hui :** prédire $\boldsymbol{\varepsilon}$ n'est qu'un choix parmi trois, qui contiennent la même information une fois $\mathbf{x}_t$ et $t$ connus (on passe de l'un à l'autre par la forme fermée) : $\boldsymbol{\varepsilon}$, $\mathbf{x}_0$, ou la **vitesse** $\mathbf{v} = \sqrt{\bar\alpha_t}\,\boldsymbol{\varepsilon} - \sqrt{1 - \bar\alpha_t}\,\mathbf{x}_0$ de T. Salimans et J. Ho (2022), plus stable quand on échantillonne en peu de pas ; le calcul de $\hat{\mathbf{x}}_0$ divise par $\sqrt{\bar\alpha_t}$, qui tend vers 0 aux grands $t$, et une erreur sur $\boldsymbol{\varepsilon}_\theta$ y devient énorme. Le choix change la difficulté de la tâche et le poids effectif de chaque pas : la pondération **min-SNR** (T. Hang et coll., 2023) plafonne le poids des pas très peu bruités et accélère la convergence. Les modèles récents décalent aussi leurs plannings vers plus de bruit quand la résolution augmente : une grande image garde plus d'information à bruit égal. · **Faut-il quand même l'apprendre ?** Oui, la prédiction du bruit d'abord : c'est la plus simple, et les autres s'en déduisent. · *Sources :* [Salimans et Ho, arXiv:2202.00512](https://arxiv.org/abs/2202.00512v2) ; [Hang et coll., arXiv:2303.09556](https://arxiv.org/abs/2303.09556v3) ; [Lin et coll., arXiv:2305.08891](https://arxiv.org/abs/2305.08891v4) ; [Esser et coll., arXiv:2403.03206](https://arxiv.org/abs/2403.03206v1) (§5.3.2, décalage selon la résolution).

### En pratique : la moyenne exponentielle des poids

Les poids d'un réseau de diffusion bougent beaucoup d'un pas d'optimisation à l'autre : la loss mélange des pas tirés au hasard et reste bruitée. Pour générer, on n'utilise pas les poids du dernier pas, mais une **moyenne mobile exponentielle des poids** (EMA, ch. 19) : une copie $\theta_{\text{EMA}}$, mise à jour après chaque pas d'optimisation par

$$\theta_{\text{EMA}} \leftarrow \rho\,\theta_{\text{EMA}} + (1 - \rho)\,\theta,$$

qui garde la mémoire d'environ $1/(1 - \rho)$ pas. Ho et ses coauteurs prennent $\rho = 0{,}9999$, soit une moyenne sur environ 10 000 pas, pour un entraînement de 800 000 pas. Pour un entraînement court sur CPU, de quelques milliers de pas, une telle EMA serait encore presque égale aux poids du départ, aléatoires : $\rho$ se choisit en fonction du nombre de pas (📦 B5.23). La copie EMA sert à générer et à évaluer ; c'est le réseau entraîné qui continue l'optimisation.

> ⚠️ **Piège classique — évaluer le mauvais réseau** — Générer avec les poids bruts au lieu de la copie EMA, ou l'inverse en fin d'un entraînement trop court pour la valeur de $\rho$ choisie, change nettement les échantillons. Sauvegarde les deux, et dis lequel tu évalues.

---

## 205.4 · Le débruiteur : U-Net minuscule et encodage du pas de temps ⏩

### Une image en entrée, une image en sortie

Le réseau reçoit une image bruitée de forme `(N, 1, 28, 28)` et le pas `t`, et rend une prédiction du bruit de **la même forme**. Il faut à la fois voir loin (reconnaître qu'un nuage de pixels ressemble à un 3) et rester précis au pixel près (le bruit est différent en chaque pixel). Un CNN de classification (ch. 21) réduit l'image à un vecteur et perd la position ; un autoencodeur (ch. 25) la reconstruit, mais tout passe par le goulot, et les détails fins s'y perdent.

### Le U-Net et ses connexions de saut

Le **U-Net** d'O. Ronneberger, P. Fischer et T. Brox (2015), conçu pour la segmentation d'images médicales (une sortie par pixel), résout ce problème. Il a la forme d'un U :

- un **encodeur** (« chemin contractant ») qui alterne blocs de convolutions et réductions de taille (convolutions de stride 2 ou pooling) : 28 × 28 → 14 × 14 → 7 × 7, avec de plus en plus de canaux ;
- un **décodeur** (« chemin expansif ») qui remonte à la taille de départ par des convolutions transposées ou un suréchantillonnage suivi d'une convolution (ch. 21) ;
- des **connexions de saut** (*skip connections*) : à chaque résolution, la carte d'activation de l'encodeur est **recopiée et concaténée** (sur l'axe des canaux) à celle du décodeur de même taille.

![Schéma d'un petit U-Net : à gauche l'entrée x_t de 1 × 28 × 28, puis deux étages d'encodeur (32 canaux à 28 × 28, 64 canaux à 14 × 14), un bloc du milieu à 128 canaux en 7 × 7, puis deux étages de décodeur qui remontent à 64 × 14 × 14 et 32 × 28 × 28, et une convolution finale qui rend la prédiction du bruit, de 1 × 28 × 28. Deux flèches bleues en arc relient chaque étage de l'encodeur à l'étage du décodeur de même résolution : les connexions de saut, par concaténation. En bas, une boîte verte « pas t → encodage sinusoïdal → MLP » envoie des flèches en pointillés vers chaque bloc résiduel](figures/unet.png)

Le chemin du bas voit l'image entière en petit, et décide « c'est un 3, la boucle est en haut » ; les connexions de saut rendent au décodeur les détails à pleine résolution que la réduction avait écrasés (🧠 Q6). Sans elles, la sortie serait floue, comme celle d'un autoencodeur.

> ⚠️ **Piège classique — deux « skip connections »** — En anglais, *skip connection* désigne aussi la **connexion résiduelle** du ch. 20, $\mathbf{x} + F(\mathbf{x})$, qui **additionne** une entrée à la sortie du même bloc. La connexion de saut du U-Net, elle, **concatène** une carte de l'encodeur à une carte du décodeur, loin dans le réseau : le nombre de canaux s'additionne, et la convolution suivante doit l'attendre (`in_channels` = canaux du décodeur + canaux de l'encodeur). Un U-Net de diffusion contient les deux : des connexions résiduelles dans chaque bloc, des connexions de saut entre les deux branches.

### Dire au réseau quel est le pas : l'encodage du pas de temps

Le même réseau sert à tous les pas, mais il ne doit pas faire la même chose au pas 10 (retirer un léger grain) et au pas 900 (deviner une forme dans du bruit). Le pas $t$ est donc une **entrée** (🧠 Q5). Le donner tel quel, un nombre de 0 à 999, marche mal : une seule valeur, d'une échelle sans rapport avec celle des activations. On utilise l'**encodage sinusoïdal** des positions d'un Transformer (B3, 🔁 R1), où le pas tient le rôle de la position : un vecteur de sinus et de cosinus de fréquences géométriquement espacées, les rapides distinguent deux pas voisins, les lentes deux pas lointains. Un petit MLP transforme ensuite ce vecteur en un **encodage du pas de temps** appris, $\mathbf{e}_t$. Ho et ses coauteurs le disent ainsi : le temps est donné au réseau « par l'encodage de position sinusoïdal du Transformer », ajouté dans chaque bloc résiduel.

Ta `positional_encoding(seq_len, d_model)` (B3) calcule la table des $T$ encodages ; il suffit de prendre ses lignes `t`. Elle range sinus et cosinus en alternance (colonnes paires et impaires), alors que le code de DDPM les met en deux moitiés, sinus puis cosinus, et le `UNet2DModel` de diffusers cosinus puis sinus : ce sont des permutations des colonnes, sans effet pour le MLP qui suit.

### Un bloc résiduel conditionné par le pas

Dans chaque bloc, $\mathbf{e}_t$ passe par une couche dense qui en tire **un nombre par canal**, ajouté à toute la carte d'activation de ce canal : le pas décale les activations, comme un biais qui dépendrait de $t$. Voici le bloc du U-Net de la figure (PyTorch 2.11), avec une **group norm** (*group normalization*, Y. Wu et K. He, 2018), qui normalise les canaux par groupes, pour chaque image séparément : contrairement à la batchnorm (ch. 20), elle ne dépend pas des autres images du batch, ce qui compte pour de petits batches et pour un réseau qui voit des images de bruits très différents dans un même batch.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ResBlock(nn.Module):
    """Two 3x3 convolutions; the step embedding adds one bias per channel."""
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.time = nn.Linear(t_dim, c_out)
        self.norm2, self.conv2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):                            # x: (N, c_in, H, W), temb: (N, t_dim)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]          # (N, c_out) -> (N, c_out, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)                             # residual connection

block = ResBlock(64, 64, t_dim=128)
print(sum(p.numel() for p in block.parameters()))          # 82368
```

Compte-le à la main : deux convolutions $64 \times 64 \times 3 \times 3 + 64 = 36\,928$ chacune, la couche du pas $128 \times 64 + 64 = 8\,256$, deux group norms à $2 \times 64$ paramètres : $73\,856 + 8\,256 + 256 = 82\,368$. Le U-Net complet de la figure (un bloc par étage, encodage de 128 valeurs, MLP du pas à deux couches) a 740 865 paramètres ; celui de Ho et coll. pour CIFAR-10, avec deux blocs par étage, quatre résolutions et de l'attention à 16 × 16, en a 35,7 millions. Tu écriras le tien en 📦 B5.21, et une version en MLP pour des points 2D en 📦 B5.19 : là, il suffit de **concaténer** l'encodage du pas aux deux coordonnées du point.

> 💼 **En entreprise** — Personne ne réécrit un U-Net de diffusion pour un projet : la bibliothèque diffusers fournit `UNet2DModel` (et `UNet2DConditionModel` pour le texte), configurables par le nombre de canaux et de blocs. L'écrire une fois, petit, sert à savoir lire ces configurations, à diagnostiquer une forme qui ne passe pas et à répondre en entretien à « pourquoi un U-Net ? ».

---

## 205.5 · Échantillonnage DDPM : moyenne du posterior, variance, T pas ⏩

### L'algorithme

Pour générer (algorithme 2 de Ho et coll.), on part de $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ et l'on répète, pour $t = T, \dots, 1$ :

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t\,\mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$

avec $\mathbf{z} = \mathbf{0}$ au dernier pas ($t = 1$, indice 0) : on ne rajoute pas de bruit à l'image qu'on rend. Ce tirage pas à pas, où chaque état ne dépend que du précédent, s'appelle l'**échantillonnage ancestral** (*ancestral sampling*). C'est `ddpm_step(x_t, t, eps_pred, schedule, rng)` de ta librairie (🔨 B5.17), appelée $T$ fois par le notebook avec la sortie du U-Net.

**Quelle variance ?** Le réseau ne prédit que la moyenne ; $\sigma_t^2$ est une constante. Deux choix naturels : $\sigma_t^2 = \beta_t$ (la variance du pas direct) ou $\sigma_t^2 = \tilde\beta_t$ (celle du posterior). Ho et ses coauteurs les trouvent « similaires » en qualité ; ce sont les deux extrêmes raisonnables (le premier serait optimal si les données étaient elles-mêmes du bruit gaussien, le second si le dataset n'avait qu'une seule image). Ta librairie prend $\tilde\beta_t$ par défaut (`variance="posterior"`), comme le `variance_type="fixed_small"` de diffusers ; Nichol et Dhariwal (2021) font **apprendre** la variance au réseau, ce qui permet d'échantillonner en bien moins de pas.

### Un pas à la main

Reprends le planning jouet ($T = 5$), au pas $t = 4$ (indice 3), avec $x_4 = 0{,}3253$ (le pixel de 205.2) et un réseau qui prédit $\varepsilon_\theta = -0{,}4$ (le vrai bruit était $-0{,}5$ : le réseau se trompe un peu). Les constantes : $\beta_4 = 0{,}08$, $1/\sqrt{\alpha_4} = 1/\sqrt{0{,}92} = 1{,}0426$, $\beta_4 / \sqrt{1 - \bar\alpha_4} = 0{,}08 / 0{,}4317 = 0{,}1853$.

1. **Moyenne** : $\mu = 1{,}0426 \times (0{,}3253 - 0{,}1853 \times (-0{,}4)) = 1{,}0426 \times 0{,}3994 = 0{,}4165$.
2. **Écart-type** : $\tilde\beta_4 = \frac{1 - 0{,}8844}{1 - 0{,}8136} \times 0{,}08 = 0{,}0496$, donc $\sigma_4 = 0{,}2228$.
3. **Tirage** : avec $z = 1{,}2$, $x_3 = 0{,}4165 + 0{,}2228 \times 1{,}2 = 0{,}6838$.

La même moyenne s'obtient par le détour de $\hat{x}_0$ : $\hat{x}_0 = (0{,}3253 - 0{,}4317 \times (-0{,}4)) / 0{,}9020 = 0{,}5521$, puis $\tilde\mu = 0{,}4036 \times 0{,}5521 + 0{,}5951 \times 0{,}3253 = 0{,}4165$. Avec le vrai bruit, $\hat{x}_0$ redonnerait exactement $0{,}6$ et la moyenne serait celle du posterior exact, $0{,}4358$. Ce chemin par $\hat{\mathbf{x}}_0$ permet d'**écrêter** l'estimation dans $[-1, 1]$ avant de calculer la moyenne (`clip_x0=1.0` de ta librairie, `clip_sample=True` par défaut dans diffusers) : aux grands $t$, une petite erreur sur $\boldsymbol{\varepsilon}_\theta$ donne un $\hat{\mathbf{x}}_0$ hors de la plage des pixels, et l'écrêtage évite qu'elle se propage. Sans écrêtage, les deux chemins donnent la même moyenne ; tes tests le vérifient. ✏️ B5.4 te fait un pas complet avec d'autres nombres.

```python
eps_pred, z = -0.4, 1.2
a, ab, ab_prev = 1 - betas[3], alpha_bars[3], alpha_bars[2]
mean = (xt - betas[3] / np.sqrt(1 - ab) * eps_pred) / np.sqrt(a)            # Algorithm 2
var = betas[3] * (1 - ab_prev) / (1 - ab)                                     # beta tilde
x0_hat = (xt - np.sqrt(1 - ab) * eps_pred) / np.sqrt(ab)
coef1, coef2 = betas[3] * np.sqrt(ab_prev) / (1 - ab), (1 - ab_prev) * np.sqrt(a) / (1 - ab)
print(round(mean, 4), round(coef1 * x0_hat + coef2 * xt, 4), round(mean + np.sqrt(var) * z, 4))
# 0.4165 0.4165 0.6838
```

### Ce que coûte un échantillon

L'échantillonnage DDPM est **lent** : $T$ passes du réseau, l'une après l'autre (chacune dépend de la précédente), soit 1 000 passes là où un GAN ou un VAE n'en font qu'une. Il est **aléatoire** : un $\mathbf{z}$ neuf à chaque pas, si bien que deux générations depuis le même $\mathbf{x}_T$ donnent deux images différentes (🧠 Q7). La figure le montre en 2D, avec le prédicteur de bruit **exact** d'un mélange de trois gaussiennes (calculable sans réseau, en formule fermée) : depuis les mêmes points de départ, deux séries de tirages mènent à des arrivées différentes, parfois dans des gaussiennes différentes.

Ordre de grandeur mesuré sur deux cœurs de CPU : le U-Net de 205.4 coûte environ 0,2 GFLOP par image et par passe (compté par `torch.utils.flop_counter`, deux opérations par multiplication-addition), et une passe sur un batch de 64 images prend environ 0,13 s ; 1 000 pas font donc un peu plus de deux minutes pour 64 chiffres. Ho et coll. génèrent 256 images de CIFAR-10 en 17 s sur un TPU v3-8, avec un réseau 50 fois plus gros.

![Deux panneaux de trajectoires en 2D, avec en gris les cercles de trois gaussiennes. À gauche, DDPM : depuis trois départs marqués d'une croix, des trajectoires très agitées, deux tirages par départ (trait plein et tirets), qui finissent à des endroits différents. À droite, DDIM : depuis les mêmes départs, trois trajectoires lisses et courtes, chacune vers une gaussienne](figures/ddpm_ddim.png)

> ⚠️ **Piège classique — l'échelle et le dernier pas** — Plusieurs erreurs donnent un échantillonneur qui ne produit que du gris ou du bruit, sans jamais planter : des images d'entraînement laissées dans $[0, 1]$ alors que le bruit et l'échantillonnage supposent $[-1, 1]$ ; un décalage d'un cran entre le pas de l'article ($1$ à $T$) et l'indice du code ($0$ à $T - 1$) ; du bruit ajouté au dernier pas. Avant tout entraînement, vérifie l'aller-retour avec le **vrai** bruit : bruite une image avec `q_sample`, puis passe le bruit exact à ton échantillonneur ; il doit retrouver l'image (🔬 B5.18).

---

## 205.6 · Accélérer : DDIM déterministe, moins de pas, distillation ⏩

### L'idée de DDIM

J. Song, C. Meng et S. Ermon (2020) remarquent que l'entraînement n'utilise que la forme fermée $q(\mathbf{x}_t \mid \mathbf{x}_0)$, jamais l'enchaînement des pas. D'autres processus que la chaîne de Markov de DDPM ont les mêmes formes fermées, donc **le même entraînement** et le même réseau ; certains se remontent bien plus vite. Leur méthode, **DDIM** (*denoising diffusion implicit models*), fait à chaque pas deux choses simples :

1. estimer l'image propre, $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\varepsilon}_\theta) / \sqrt{\bar\alpha_t}$ ;
2. la **rebruiter** au niveau d'un pas antérieur $t' < t$, quelconque, en réutilisant le bruit prédit :

$$\mathbf{x}_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat{\mathbf{x}}_0 + \sqrt{1 - \bar\alpha_{t'}}\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t) \qquad (\text{DDIM},\ \eta = 0).$$

Aucun tirage : à partir du même $\mathbf{x}_T$, on obtient toujours la même image. Et $t'$ peut être très loin de $t$ : on remonte la chaîne par une **sous-séquence de pas**, par exemple 50 indices régulièrement espacés parmi 1 000, sans réentraîner (🧠 Q8). Les auteurs génèrent des images de qualité comparable 10 à 50 fois plus vite.

**Avec le même planning jouet**, du pas 4 (indice 3) au pas 2 (indice 1), avec $\hat{x}_0 = 0{,}5521$ et $\varepsilon_\theta = -0{,}4$ comme plus haut : $x_2 = \sqrt{0{,}9408} \times 0{,}5521 + \sqrt{0{,}0592} \times (-0{,}4) = 0{,}9699 \times 0{,}5521 - 0{,}2433 \times 0{,}4 = 0{,}4382$, en un seul pas au lieu de deux. Pour $t' = 0$ (l'indice $-1$ de ta librairie, où $\bar\alpha = 1$), DDIM rend directement $\hat{x}_0$. Si $\boldsymbol{\varepsilon}_\theta$ est le **vrai** bruit, un seul pas DDIM jusqu'à $t' = 0$ retrouve exactement $\mathbf{x}_0$ : c'est le test de ta `ddim_step` (🔨 B5.17). ✏️ B5.5 te fait un pas DDIM avec d'autres nombres.

### Le paramètre η

DDIM a un réglage $\eta$, entre 0 et 1, qui réintroduit du hasard. Le pas général est

$$\mathbf{x}_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat{\mathbf{x}}_0 + \sqrt{1 - \bar\alpha_{t'} - \sigma^2}\,\boldsymbol{\varepsilon}_\theta + \sigma\,\mathbf{z}, \qquad \sigma = \eta\,\sqrt{\frac{1 - \bar\alpha_{t'}}{1 - \bar\alpha_t}}\,\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t'}}}.$$

Avec $\eta = 0$, on retrouve le pas déterministe. Avec $\eta = 1$ et $t' = t - 1$, $\sigma^2$ vaut exactement $\tilde\beta_t$ et le pas est **exactement** un pas DDPM (sur le planning jouet au pas 4 : $\sigma^2 = 0{,}0496 = \tilde\beta_4$). DDPM et DDIM sont donc les deux bouts d'une même famille. Ta `ddim_step` refuse un $\eta$ hors de $[0, 1]$ et un $t'$ qui n'est pas strictement avant $t$.

### Pourquoi c'est plus rapide

DDIM avec $\eta = 0$ revient à suivre une trajectoire **continue et déterministe** du bruit vers l'image (Song et ses coauteurs le relient à une équation différentielle), et chaque pas en est une approximation par une tangente : une trajectoire peu courbée se suit bien avec des pas assez nombreux, mal avec trop peu. DDPM, lui, ajoute à chaque pas un bruit qu'il faut ensuite retirer, et a besoin de petits pas pour que ce va-et-vient se compense. La figure de 205.5 le montre : les trajectoires DDIM sont lisses et courtes, celles de DDPM agitées. 📦 B5.24 compare sur MNIST DDPM à 1 000 pas et DDIM à 50 pas, en qualité et en temps ; 🔬 B5.25 cherche combien de pas DDIM suffisent, avec les deux plannings.

**Un effet secondaire utile** : comme le chemin est déterministe, on peut le parcourir **à l'envers**, de $\mathbf{x}_0$ vers $\mathbf{x}_T$ ; c'est l'**inversion DDIM**, qui retrouve un « bruit » donnant une image réelle donnée. On s'en sert pour modifier une image existante (la rebruiter un peu, puis la débruiter sous une autre condition), et pour interpoler entre deux images en interpolant leurs bruits.

> ⚠️ **Piège classique — une sous-séquence qui ne part pas du dernier indice** — Pour 50 pas parmi 1 000, l'espacement `"leading"` (le défaut de `DDIMScheduler` dans diffusers 0.40) donne les indices 980, 960, …, 0 : le premier pas part de l'indice 980, pas de 999, alors que $\mathbf{x}_T$ a été tiré comme un bruit pur. L'espacement `"trailing"` donne 999, 979, …, 19, et `"linspace"` 999, …, 0. Lin et coll. (2024) recommandent de toujours partir du dernier indice. Avec ta propre boucle, construis la sous-séquence explicitement (par exemple `np.linspace(T - 1, 0, n).round().astype(int)`) et affiche-la.

### La distillation

DDIM accélère sans réentraîner ; pour aller plus loin, on **réentraîne** un modèle élève qui imite en peu de pas ce que le modèle maître fait en beaucoup. C'est une **distillation**, au sens large du ch. 6 (un élève apprend d'un maître), mais ce qu'on distille ici, ce sont les **pas** de l'échantillonneur :

- **distillation progressive** (Salimans et Ho, 2022) : l'élève apprend à faire en un pas ce que le maître fait en deux (DDIM), et l'on recommence en divisant par deux à chaque fois : de 8 192 pas à 4, avec un FID de 3,0 sur CIFAR-10 en 4 pas ;
- **modèles de cohérence** (*consistency models*, Y. Song et coll., 2023) : un réseau apprend à envoyer **n'importe quel** point d'une trajectoire directement à son arrivée, ce qui permet de générer en un seul pas ;
- **distillation adversariale** (A. Sauer et coll., 2023, SDXL Turbo) : on ajoute un discriminateur, comme dans un GAN, pour garder des images nettes en 1 à 4 pas. FLUX.1 [schnell] (12 milliards de paramètres, Apache 2.0) génère ainsi en 1 à 4 pas.

> 💼 **En entreprise** — Le coût d'un générateur d'images en production est presque entièrement celui de l'échantillonnage : nombre de pas × coût d'une passe × nombre d'images. Les leviers, dans l'ordre où on les essaie : un meilleur échantillonneur (DDIM ou un solveur d'ordre supérieur, sans réentraîner), un modèle distillé publié, la précision réduite, puis la distillation maison. Et on mesure la qualité **à chaque** réglage du nombre de pas, sur le même jeu de prompts : un gain de temps qui dégrade les images n'en est pas un. 🧮 B5.8 te fait chiffrer ce coût.

---

## 205.7 · Conditionner : label de classe et guidance sans classifieur ⏩

### Donner la classe au réseau

Pour générer « un 7 » plutôt qu'un chiffre quelconque, on donne au réseau un **label de classe** $c$ en plus de $\mathbf{x}_t$ et $t$ : $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t, c)$. La façon la plus simple : une table `nn.Embedding(n_classes + 1, d)` donne un vecteur par classe, qu'on **ajoute** à l'encodage du pas de temps ; chaque bloc reçoit ainsi « le pas et la classe ». La ligne supplémentaire de la table sert à la classe « vide », notée $\varnothing$ : « pas de condition ». Pour un texte, le conditionnement passe par l'**attention croisée** (B3) : les requêtes viennent des pixels, les clés et les valeurs des tokens du prompt encodés par un encodeur de texte (CLIP, T5 ou un modèle de langue).

Un modèle conditionné génère bien des 7, mais souvent des 7 un peu hésitants, qui pourraient passer pour des 1. Pour **renforcer** la condition, P. Dhariwal et A. Nichol (2021) avaient proposé la **guidance par classifieur** (*classifier guidance*) : un classifieur entraîné sur des images bruitées pousse chaque pas dans la direction qui rend la classe plus probable. Il faut alors entraîner et garder un second réseau.

### La guidance sans classifieur

J. Ho et T. Salimans (2022) se passent du classifieur. Pendant l'entraînement, on remplace le label par $\varnothing$ avec une probabilité $p_{\text{uncond}}$ (0,1 ou 0,2 marchent aussi bien l'un que l'autre ; 0,5 est moins bon) : **un seul** réseau apprend à la fois la version conditionnée et la version sans condition. À l'échantillonnage, on fait **deux** prédictions à chaque pas et on les combine :

$$\tilde{\boldsymbol{\varepsilon}} = (1 + w)\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, c) - w\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, \varnothing) = \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, c) + w\,\big(\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, c) - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, \varnothing)\big),$$

puis on utilise $\tilde{\boldsymbol{\varepsilon}}$ à la place de $\boldsymbol{\varepsilon}_\theta$ dans le pas DDPM ou DDIM. C'est la **guidance sans classifieur** (*classifier-free guidance*, CFG). L'**échelle de guidance** $w \ge 0$ règle la force : $w = 0$ redonne le modèle conditionné seul ; $w > 0$ **extrapole** dans la direction « ce que la condition change », au-delà du modèle conditionné.

Exemple, avec deux valeurs de bruit prédit : $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, c) = (0{,}4 ;\ -1{,}0)$, $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, \varnothing) = (0{,}1 ;\ -0{,}6)$ et $w = 0{,}5$ : $\tilde{\boldsymbol{\varepsilon}} = 1{,}5 \times (0{,}4 ;\ -1{,}0) - 0{,}5 \times (0{,}1 ;\ -0{,}6) = (0{,}55 ;\ -1{,}2)$. L'écart entre les deux prédictions, $(0{,}3 ;\ -0{,}4)$, est ajouté une demi-fois de plus. ✏️ B5.1 te fait la combinaison pour trois valeurs de $w$.

**Pourquoi ça marche ?** Le bruit prédit est lié à la direction dans laquelle la densité des données augmente le plus vite (le **score** $\nabla \ln p(\mathbf{x}_t)$, à un facteur $-\sqrt{1 - \bar\alpha_t}$ près). La différence $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, c) - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, \varnothing)$ correspond alors, au même facteur près, au gradient de $\ln p(c \mid \mathbf{x}_t)$, par la règle de Bayes : un classifieur **implicite**, que le modèle contient sans qu'on l'ait entraîné à part. Guider, c'est suivre ce classifieur implicite plus fort que le modèle ne le ferait seul.

![Trois panneaux de points en 2D sur trois cercles gris (trois gaussiennes, trois classes). À gauche, sans condition : des points dans les trois gaussiennes. Au milieu, la classe 2 avec w = 0 : un nuage centré sur la gaussienne du haut, d'écart-type moyen 0,33. À droite, la classe 2 avec w = 0,5 : un nuage plus serré, d'écart-type moyen 0,27, un peu décalé vers le haut, loin des deux autres classes](figures/guidance.png)

La figure le montre sur le mélange de 205.5, avec des prédicteurs exacts : même à $w = 0{,}5$, les échantillons de la classe se resserrent et s'éloignent des autres classes. Ho et Salimans le mesurent sur ImageNet : le meilleur FID s'obtient avec une **petite** guidance ($w = 0{,}1$ ou $0{,}3$), le meilleur Inception Score avec une guidance forte ($w \ge 4$), et entre les deux, plus de guidance donne des échantillons plus fidèles à leur classe et **moins variés** (🧠 Q9). Ce que cela donne sur Fashion-MNIST, pour quelles valeurs de $w$, tu le prédiras en 🔮 B5.26 avant de le mesurer en 📦 B5.27.

> ⚠️ **Piège classique — deux conventions pour l'échelle de guidance** — Dans diffusers et la plupart des interfaces, la guidance s'écrit $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, \varnothing) + s\,\big(\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, c) - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, \varnothing)\big)$, avec `guidance_scale` $= s = 1 + w$ : $s = 1$ est le modèle conditionné sans guidance, et la guidance n'est activée que pour $s > 1$. Le défaut de Stable Diffusion, $s = 7{,}5$, correspond à $w = 6{,}5$ dans la convention de Ho et Salimans, celle de ce chapitre. Avant de comparer deux chiffres de guidance, vérifie la convention.

La guidance a un prix : **deux** passes du réseau par pas (on les fait en un seul batch, en empilant les entrées avec $c$ et avec $\varnothing$), donc un échantillonnage deux fois plus cher.

---

## 205.8 · Évaluer : distance de Fréchet, diversité, mémorisation ⏩

### La distance de Fréchet

On évalue un modèle de diffusion comme le GAN du ch. 27 (🔁 R2) : on calcule des **features** pour des images réelles et générées par un réseau entraîné (l'avant-dernière couche d'Inception v3 pour le FID ; d'un petit classifieur MNIST dans le notebook), on ajuste une gaussienne à chaque nuage, et l'on mesure leur **distance de Fréchet** :

$$d_F^2 = \lVert \boldsymbol{\mu}_r - \boldsymbol{\mu}_g \rVert^2 + \operatorname{Tr}\big(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g)^{1/2}\big).$$

C'est ta `frechet_distance_from_features` du ch. 27, et le code d'évaluation du mini-projet MP6 resservira tel quel (📦 B5.29). Trois règles pour que deux nombres se comparent : **les mêmes features**, **le même nombre d'images** (la distance estimée est biaisée vers le haut quand on a peu d'images, d'où les 50 000 échantillons du FID publié), **le même jeu de référence**. Une distance sur 64 images générées ne se compare qu'à une autre distance sur 64 images.

Pour un modèle de diffusion, la distance de Fréchet sert surtout à **régler l'échantillonnage** : nombre de pas, planning, $\eta$, échelle de guidance, EMA ou non. Chaque réglage se paie en temps, et la courbe « distance de Fréchet contre temps de calcul » est la façon honnête de choisir (🔬 B5.25 ; le défi 🏆 B5.31 limite l'échantillonnage à 50 pas).

### Fidélité et diversité

Un seul nombre mélange deux qualités. Les **precision et recall génératifs** du ch. 27 les séparent : la part des échantillons qui tombent près de vraies données (fidélité) et la part des vraies données qui ont un échantillon près d'elles (diversité). C'est ainsi que Nichol et Dhariwal ont montré qu'un DDPM couvre mieux la distribution qu'un GAN : à FID voisin, un recall bien plus haut. La guidance, elle, déplace le curseur dans l'autre sens.

### La mémorisation

Un générateur peut aussi être « trop bon » : rendre, à quelques pixels près, une image de son dataset d'entraînement. On appelle cela la **mémorisation** (*memorization*). Elle pose des problèmes de vie privée (une photo de personne), de droit d'auteur (une œuvre protégée) et de marques (un logo), et aucune distance de Fréchet ne la voit : un modèle qui recopierait le dataset aurait une distance excellente (🧠 Q10).

Pour la chercher, on compare chaque échantillon à son **plus proche voisin** dans le dataset d'entraînement (ch. 13), en pixels ou en features. Une distance faible ne suffit pas : sur CIFAR-10, des ciels bleus unis sont proches de beaucoup d'images. N. Carlini et ses coauteurs (2023) déclarent une image « extraite » quand elle est **anormalement** proche de son voisin, comparée aux distances de ce voisin à ses propres voisins. Leurs résultats :

- sur **Stable Diffusion**, en générant 500 images pour chacune des 350 000 légendes les plus dupliquées du dataset (175 millions d'images), ils retrouvent 94 images d'entraînement au sens strict de leur définition, et 109 en ajoutant celles qu'une inspection visuelle juge presque identiques : des photos de personnes, des logos ; Imagen, plus gros, en recrache davantage ;
- sur des modèles de diffusion entraînés sur **CIFAR-10**, ils extraient 1 280 images d'entraînement (2,5 % du dataset) ;
- les modèles de diffusion de leur étude **mémorisent plus que les GAN** de même FID (deux fois plus que StyleGAN-ADA), et la mémorisation augmente quand la qualité augmente ;
- les images **dupliquées** dans le dataset sont de loin les plus mémorisées : dédupliquer le dataset est la première parade, sans être suffisante.

G. Somepalli et ses coauteurs (2023) mesurent aussi la réplication de contenus par recherche d'images, et en trouvent dans Stable Diffusion. Dans 🔬 B5.28, tu cherches les plus proches voisins de tes échantillons, pour deux tailles de dataset d'entraînement.

> 🕰️ **Mise à jour (2026) — évaluer un générateur** — **Le livre :** le jugement visuel (ch. 27). · **Aujourd'hui :** le FID reste le nombre standard, critiqué : ses features d'Inception jugent mal ce qui s'éloigne d'ImageNet, et il ne suit pas toujours le jugement humain. G. Stein et coll. (2023) ont mené la plus grande évaluation humaine de générateurs publiée jusque-là : aucune des 17 mesures qu'ils comparent ne suit bien le jugement humain, aucune ne sait non plus détecter la mémorisation, et ils recommandent une distance de Fréchet sur les features de **DINOv2** (B1) ; on y joint les precision et recall génératifs, et pour le texte vers image le **CLIP score** (la similarité entre l'image et le prompt dans l'espace de CLIP, B1), plus des évaluations humaines. Et une recherche de mémorisation fait partie de toute évaluation sérieuse. · **Faut-il quand même l'apprendre ?** Oui : la distance de Fréchet est le langage commun, à condition de dire sur quelles features et sur combien d'images. · *Sources :* [Stein et coll., arXiv:2306.04675](https://arxiv.org/abs/2306.04675v2) ; [Hessel et coll., CLIPScore, arXiv:2104.08718](https://arxiv.org/abs/2104.08718) ; [Carlini et coll., arXiv:2301.13188](https://arxiv.org/abs/2301.13188v1) ; [Somepalli et coll., arXiv:2212.03860](https://arxiv.org/abs/2212.03860v3).

---

## 205.9 · Paysage actuel : diffusion latente, transformers de diffusion, flow matching, texte vers image ⏩

### La diffusion latente

Un DDPM sur des pixels coûte cher : chaque pas traite toutes les valeurs de l'image, et une image de 512 × 512 en couleurs en a 786 432. R. Rombach et ses coauteurs (2022) font la diffusion dans l'**espace latent** d'un autoencodeur entraîné à part (ch. 25), qui comprime l'image en une grille plus petite en gardant les détails perceptibles :

1. un **encodeur** réduit l'image d'un facteur $f$ en largeur et en hauteur ; pour Stable Diffusion 1.x, $f = 8$ et 4 canaux : 512 × 512 × 3 devient 64 × 64 × 4, soit 16 384 valeurs, 48 fois moins ;
2. le modèle de diffusion apprend et génère dans cette grille, exactement comme dans ce chapitre ;
3. le **décodeur** transforme le latent généré en image, une seule fois, à la fin.

Un pas de diffusion traite ainsi 64 fois moins de positions. Les auteurs trouvent qu'un facteur de 4 à 16 équilibre le coût et la qualité : trop peu de compression laisse le coût élevé, trop en perd les détails. C'est la **diffusion latente** (*latent diffusion*), le principe de Stable Diffusion (2022), qui y ajoute l'attention croisée vers un encodeur de texte (CLIP ViT-L/14 pour la version 1). Les autoencodeurs récents gardent plus de canaux : 16 pour Stable Diffusion 3, ce qui améliore nettement la reconstruction (🧮 B5.8 chiffre ce que gagne l'espace latent).

> 🔗 **Lien avec le ch. 25** — L'autoencodeur d'une diffusion latente est un VAE à faible poids de KL, souvent complété par une loss perceptuelle et un discriminateur : on lui demande une reconstruction fidèle, pas un espace latent où l'on pourrait tirer au hasard ; c'est la diffusion qui génère dans cet espace. Ton VAE du ch. 25 en est la version miniature, et le mini-projet MP6 proposait déjà une diffusion latente en extension.

### Les transformers de diffusion

W. Peebles et S. Xie (2023) remplacent le U-Net par un **transformer de diffusion** (*diffusion transformer*, DiT) : le latent est découpé en **patchs**, comme une image dans un ViT (B3) ; pour une image de 256 × 256, le latent de 32 × 32 × 4 en patchs de 2 × 2 donne 256 tokens. Le pas et la classe ne passent plus par un biais ajouté, mais modulent les layer norms de chaque bloc (*adaLN-Zero*). Leur constat : plus le modèle coûte de calcul (plus profond, plus large, ou plus de tokens), plus le FID baisse ; le plus grand, DiT-XL/2, atteint un FID de 2,27 sur ImageNet 256 × 256. Stable Diffusion 3 (P. Esser et coll., 2024) généralise l'idée au texte vers image avec le **MM-DiT** : les tokens d'image et de texte ont chacun leurs poids, et une attention commune les fait échanger dans les deux sens. FLUX.1 et FLUX.2, Qwen-Image et les modèles vidéo comme Wan suivent ce schéma.

### Flow matching et rectified flow

Le **flow matching** (Y. Lipman et coll., 2022) et le **rectified flow** (X. Liu et coll., 2022), sous des formes voisines, simplifient encore le processus direct : au lieu de la chaîne de DDPM, une **ligne droite** entre la donnée et le bruit, parcourue par un temps $s \in [0, 1]$ :

$$\mathbf{x}_s = (1 - s)\,\mathbf{x}_0 + s\,\boldsymbol{\varepsilon}, \qquad \frac{d\mathbf{x}_s}{ds} = \boldsymbol{\varepsilon} - \mathbf{x}_0 = \mathbf{v}.$$

Le réseau apprend la **vitesse** $\mathbf{v}$ de chaque point, par une MSE comme en 205.3 : $\big\lVert (\boldsymbol{\varepsilon} - \mathbf{x}_0) - \mathbf{v}_\theta(\mathbf{x}_s, s) \big\rVert^2$, avec $s$ tiré au hasard. Pour générer, on part d'un bruit pur ($s = 1$) et l'on suit le **champ de vitesses** (*velocity field*) appris jusqu'à $s = 0$, par la méthode d'Euler (encadré) :

```python
x = rng.standard_normal((n, 2))                    # s = 1: pure noise (2-D points here)
for k in range(n_steps, 0, -1):                    # s = k/n_steps -> (k-1)/n_steps
    x = x - (1 / n_steps) * v_model(x, k / n_steps)    # one Euler step backwards in s
```

Chaque trajectoire « donnée → bruit » de l'entraînement est droite, mais le réseau n'apprend qu'une **moyenne** des vitesses de toutes les trajectoires qui passent par un point (l'encadré de 205.1 vaut aussi ici) : le champ appris n'est pas exactement droit. Rectifier, au sens de Liu et coll., c'est réentraîner sur les couples (bruit, image) que le premier modèle produit, pour redresser les trajectoires et les suivre en moins de pas.

> 🧮 **Rappel maths — la méthode d'Euler** — Pour suivre une trajectoire dont on connaît la vitesse en chaque point, $\frac{dx}{ds} = v(x, s)$, on avance par petits pas en supposant la vitesse constante sur chaque pas : $x(s + h) \approx x(s) + h\,v(x(s), s)$. C'est remplacer la courbe par sa tangente (0B). L'erreur vient de la courbure : sur $\frac{dy}{ds} = y$ avec $y(0) = 1$, la valeur exacte en $s = 1$ est $e \approx 2{,}718$ ; Euler donne $2$ en un pas ($h = 1$), $2{,}25$ en deux, $2{,}441$ en quatre, $2{,}594$ en dix. Si la vitesse est **constante** le long du chemin (une ligne droite parcourue à vitesse constante), un seul pas est exact. D'où l'intérêt de trajectoires droites : moins de pas pour la même précision. Pour remonter le temps (de $s = 1$ vers $s = 0$), on prend $h < 0$, soit $x \leftarrow x - |h|\,v$.

Le lien avec ce chapitre est étroit : la ligne droite du flow matching est un planning de bruit particulier (le signal décroît comme $1 - s$, le bruit croît comme $s$, sans préserver la variance), connaissant $\mathbf{x}_s$ et $s$, prédire $\mathbf{v}$ revient à prédire $\mathbf{x}_0 = \mathbf{x}_s - s\,\mathbf{v}$ ou $\boldsymbol{\varepsilon} = \mathbf{x}_s + (1 - s)\,\mathbf{v}$, et DDIM avec $\eta = 0$ est lui-même une méthode d'Euler, dans d'autres variables. Ne confonds pas cette vitesse avec la « $\mathbf{v}$-prédiction » de Salimans et Ho (205.3), qui est une autre combinaison de $\boldsymbol{\varepsilon}$ et de $\mathbf{x}_0$. Attention aussi au sens du temps : Lipman et coll. vont du bruit ($s = 0$) vers la donnée ($s = 1$) ; Stable Diffusion 3, et ce chapitre, de la donnée ($s = 0$) vers le bruit ($s = 1$). Stable Diffusion 3, FLUX.1 et FLUX.2 sont des transformers de **rectified flow** ; SD3 tire $s$ plus souvent au milieu de $[0, 1]$ qu'aux bords, là où la tâche est la plus difficile. Tu compares les deux méthodes en 2D, avec le même MLP, en 🔬 B5.20 (💼 E5).

### Du texte à l'image, et au-delà

Un modèle texte vers image moderne assemble donc : un **encodeur de texte** (CLIP, T5, ou un modèle de langue entier : FLUX.2 utilise un modèle vision-langage de 24 milliards de paramètres), un **autoencodeur** d'images, un **transformer de diffusion** entraîné par flow matching dans l'espace latent, la **guidance sans classifieur** à l'échantillonnage, souvent une **distillation** pour réduire le nombre de pas, et des fine-tunings sur des préférences humaines. La même recette, avec des latents qui ont aussi un axe de temps, produit de la vidéo. Le DDPM de ce chapitre en contient chaque idée de base, en petit.

> 🕰️ **Mise à jour (2026) — la bibliothèque diffusers** — **Le livre :** Keras 2 seulement ; diffusers n'existait pas (2022). · **Aujourd'hui :** diffusers (Hugging Face) fournit les échantillonneurs (`DDPMScheduler`, `DDIMScheduler`, `FlowMatchEulerDiscreteScheduler` et plus de quarante autres), les réseaux (`UNet2DModel`, des transformers de diffusion) et des pipelines prêts à l'emploi pour des modèles pré-entraînés. Le workbook la fige à 0.40.0 (`requirements.txt`), une version qui s'installe avec torch 2.11 et transformers 5.21 ; elle sert d'oracle aux tests de ta `diffusion.py` (ignorés si elle est absente). Ses conventions sont celles de la fiche : indices de 0 à $T - 1$, variance $\tilde\beta_t$ par défaut (`variance_type="fixed_small"`), mais **écrêtage de $\hat{\mathbf{x}}_0$ activé par défaut** (`clip_sample=True`), espacement `"leading"` par défaut pour DDIM et `guidance_scale` $= 1 + w$. · **Faut-il quand même l'apprendre ?** Oui pour s'en servir en projet ; mais après avoir écrit soi-même les pas, pour savoir ce que cachent ses réglages par défaut. · *Sources :* [documentation de diffusers, `DDPMScheduler`](https://huggingface.co/docs/diffusers/v0.40.0/en/api/schedulers/ddpm) et [`DDIMScheduler`](https://huggingface.co/docs/diffusers/v0.40.0/en/api/schedulers/ddim) ; [diffusers sur PyPI](https://pypi.org/project/diffusers/0.40.0/).

### Transparence et provenance des contenus générés

Une image générée peut tromper : un faux document, une fausse photo de presse, un **hypertrucage** (*deepfake*) d'une personne réelle. Deux outils techniques, complémentaires, permettent de signaler l'origine d'un contenu :

- les **métadonnées de provenance** signées, selon le standard **C2PA** (Coalition for Content Provenance and Authenticity ; les « Content Credentials ») : le fichier transporte un manifeste signé qui dit quel outil l'a produit ou modifié. Fiables tant qu'on les garde, elles disparaissent si l'on fait une capture d'écran ou si un site les retire ;
- un **filigrane** (*watermark*) invisible inscrit dans les pixels eux-mêmes, qui résiste mieux aux copies mais peut être affaibli par des retouches, et que seul un détecteur adapté lit.

> 🕰️ **Mise à jour (2026) — la transparence des contenus générés** — **Le livre :** aucune règle sur les contenus générés (2018). · **Aujourd'hui :** l'article 50 de l'AI Act (règlement (UE) 2024/1689) s'applique depuis le **2 août 2026**, quel que soit le niveau de risque : les fournisseurs de systèmes génératifs marquent leurs images, sons, vidéos et textes de façon **lisible par machine** et détectable (les systèmes déjà sur le marché avant cette date ont jusqu'au 2 décembre 2026), et ceux qui diffusent un hypertrucage doivent dire qu'il est artificiel. Le **code de bonnes pratiques** sur la transparence des contenus générés, publié par la Commission le 10 juin 2026 et signé par environ 190 organisations fin juillet, propose de combiner métadonnées signées et filigrane invisible. OpenAI annonçait en mars 2025 des métadonnées C2PA sur toutes les images de GPT-4o ; la spécification C2PA est révisée plusieurs fois par an (version 2.4 en avril 2026). · **Faut-il quand même l'apprendre ?** Oui : dès qu'un projet génère des images pour d'autres personnes, le marquage fait partie du livrable (⚖️ B5.10). · *Sources :* [règlement (UE) 2024/1689, art. 50](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) ; [Commission européenne, code de bonnes pratiques sur les contenus générés](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content) ; [Baker Botts, « EU AI Act Article 50 Transparency Obligations Go Live » (septembre 2026)](https://www.bakerbotts.com/thought-leadership/publications/2026/september/eu-ai-act-article-50-transparency-obligations-go-live) ; [spécification C2PA 2.4](https://spec.c2pa.org/specifications/specifications/2.4/specs/_attachments/C2PA_Specification.pdf) ; [OpenAI, fiche système (PDF), « C2PA metadata on all assets »](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf).

---

## Les pièges classiques ⚠️ (récapitulatif)

| Piège | Exemple | Ce qu'il faut faire |
|---|---|---|
| Confondre le pas de l'article et l'indice du code | `alpha_bars[t]` avec $t$ de 1 à $T$ : décalage d'un cran, et `alpha_bars[T]` hors du tableau | indices de 0 à $T - 1$ partout dans le code ; `alpha_bars_prev[0] = 1` |
| Mettre $\bar\alpha_t$ là où il faut $\sqrt{\bar\alpha_t}$ | `ab * x0 + (1 - ab) * eps` : la variance n'est plus préservée | coefficients $\sqrt{\bar\alpha_t}$ et $\sqrt{1 - \bar\alpha_t}$, dont les **carrés** font 1 |
| Laisser les images dans $[0, 1]$ | le bruit et l'écrêtage supposent $[-1, 1]$ : échantillons gris | `x * 2 - 1` avant l'entraînement, `(x + 1) / 2` pour afficher |
| Ajouter du bruit au dernier pas | `x = mean + sigma * z` aussi à l'indice 0 : images granuleuses | aucun tirage à l'indice 0 |
| Garder le planning linéaire en raccourcissant la chaîne | `np.linspace(1e-4, 0.02, 200)` : il reste 13 % de la variance de l'image à la fin | multiplier les bornes par $1\,000/T$, ou le planning cosinus ; afficher $\bar\alpha$ au dernier indice |
| Oublier de remettre `alpha_bars[t]` en forme | un coefficient de forme `(N,)` aligné sur la largeur de l'image | `.reshape(-1, 1, 1, 1)` (ou `view` en PyTorch) |
| Concaténer au lieu d'additionner, ou l'inverse | une connexion de saut ajoutée, ou une convolution dont `in_channels` oublie la concaténation | concaténation dans le U-Net, et la convolution suivante attend la somme des canaux |
| Passer le pas brut au réseau | `t` comme un flottant de 0 à 999 | encodage sinusoïdal (ta `positional_encoding`), puis un MLP |
| Juger un entraînement à sa loss | « la loss ne baisse plus depuis 2 epochs » | regarder des échantillons à bruit de départ fixé, et une distance de Fréchet |
| Générer avec les poids bruts | les échantillons varient d'une epoch à l'autre | générer avec la copie EMA, réglée selon le nombre de pas d'entraînement |
| Une sous-séquence DDIM qui ne part pas du dernier indice | espacement `"leading"` : premier pas à l'indice 980 sur 1 000 | `np.linspace(T - 1, 0, n)` ou `"trailing"`, et l'afficher |
| Confondre les deux conventions de guidance | `guidance_scale = 5` lu comme $w = 5$ (c'est $w = 4$) | diffusers : $s = 1 + w$ ; dire laquelle on utilise |
| Comparer des distances de Fréchet incomparables | 64 échantillons contre 10 000, ou deux extracteurs de features | mêmes features, même nombre d'images, même référence |
| Conclure « il génère » sans chercher la copie | une distance de Fréchet excellente sur un petit dataset | plus proches voisins dans le dataset d'entraînement |

## Liens avec les autres chapitres 🔗

- **Ch. 2 et 4** : la loi normale, la variance d'une somme de variables indépendantes (la forme fermée) ; la règle de Bayes, le prior et le posterior (le posterior d'un pas).
- **Ch. 13** : les plus proches voisins, pour chercher la mémorisation.
- **Ch. 19 et 20** : AdamW, la moyenne mobile exponentielle (devenue EMA des poids) ; la boucle d'entraînement PyTorch, les connexions résiduelles, les normalisations (la group norm en est une de plus).
- **Ch. 21** : convolutions de stride 2, convolution transposée et suréchantillonnage, les deux branches du U-Net ; ses artefacts en damier se retrouvent dans un décodeur mal réglé.
- **Ch. 25** : le débruiteur, ancêtre direct du réseau de diffusion ; l'astuce de reparamétrisation, qui écrit la forme fermée ; l'ELBO, d'où vient la loss ; l'espace latent, où se fait la diffusion latente.
- **Ch. 27** : le DCGAN et l'effondrement de mode, à comparer ; ta distance de Fréchet et le code du MP6, qui évaluent les trois générateurs sur MNIST.
- **B1 et B3** : CLIP et DINOv2, pour conditionner par du texte et évaluer ; l'encodage sinusoïdal (ta `positional_encoding`), réutilisé pour le pas de temps ; l'attention croisée et le ViT, à la base des transformers de diffusion.
- **B6** : documenter un modèle (fiche modèle), la supervision humaine et l'AI Act, vus du côté de l'explicabilité et de l'équité.
- **B7** : passer du notebook au projet : ligne de commande avec `argparse`, configuration, graines et suivi des runs ; ton échantillonneur autonome (🛠️ B5.30) en est un premier pas.
- **B8** : le modèle de récompense appris sur des préférences humaines et le RLHF, que le fine-tuning des générateurs d'images récents emprunte aux modèles de langue (Qwen-Image-2.0-RL, 2026).

## Guide de lecture et de travail

Ce chapitre n'a pas de lecture dans le livre : **tous les parcours lisent la fiche en entier** (≈ 1 h 50), d'où ses neuf sections marquées ⏩. Refais à la main ses exemples chiffrés (le planning jouet et son SNR, le pixel bruité au pas 4, le posterior, un pas DDPM et un pas DDIM sur ce pixel, la combinaison de la guidance, le bloc résiduel de 82 368 paramètres), suis l'ordre conseillé, écris tes réponses papier dans `mon_travail/b5_diffusion/06_mes_reponses.md` et vérifie les ✏️ dans la partie 0 du notebook (`wb.check`).

**Le notebook** télécharge MNIST et Fashion-MNIST (une fois). Le mode rapide entraîne un petit U-Net sur un sous-ensemble en quelques minutes de CPU ; les modes complets 🚀 (B5.23, B5.25, B5.27, B5.28 et le défi B5.31) tournent mieux sur un GPU de Colab.

**Parcours rapide** : toute la fiche, les quiz Q1 à Q3 et Q5 à Q11, les trois rappels, les ✏️ B5.1, B5.2 et B5.4, 🧮 B5.8, 🗣️ B5.9 et ⚖️ B5.10 ; dans le notebook B5.12 à B5.14, B5.16, B5.17, B5.19 et B5.21 à B5.24 ; les cinq entretiens ; les corrigés de ✏️ B5.5, ∂ B5.6 et ∂ B5.7 à lire. **Parcours maths** : le quiz Q3, les ✏️ B5.1 à B5.5, ∂ B5.6 et B5.7, 🧮 B5.8, 📄 B5.11, puis dans le notebook B5.12, B5.14 et B5.16 à B5.20 (ta librairie, ses contrôles et les jouets 2D). **Parcours code** : tout le notebook (B5.12 à B5.31), et les corrigés de ✏️ B5.1 à B5.3, ✏️ B5.5, ∂ B5.6 et ∂ B5.7 à lire. Le quiz Q4 n'est dans aucun parcours réduit : fais-le dans le parcours complet. La liste exacte est dans `docs/PARCOURS.md`. **Si tu bloques** : règle des 15 minutes, puis les indices de `04_indices.md`, un niveau à la fois.

## Pour aller plus loin

- J. Ho, A. Jain et P. Abbeel, « [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239v2) », NeurIPS 2020 (📄 B5.11) : la forme fermée, le posterior, la loss simplifiée et les deux algorithmes en une page chacun ; l'annexe B donne tous les réglages.
- L. Weng, « [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) » (2021, mis à jour jusqu'en avril 2024) : toutes les dérivations de ce chapitre, pas à pas, puis DDIM, la guidance, la diffusion latente et la distillation, avec les notations de Ho et coll.
- C. Luo, « [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970v1) », 2022 : du VAE à la diffusion par l'ELBO, et l'équivalence des prédictions de $\boldsymbol{\varepsilon}$, de $\mathbf{x}_0$ et du score.
- J. Song, C. Meng et S. Ermon, « [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502v4) », ICLR 2021, puis J. Ho et T. Salimans, « [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598v1) », 2022 : les deux articles derrière l'échantillonnage rapide et la guidance de ce chapitre.
- Le [cours de Hugging Face sur la diffusion](https://github.com/huggingface/diffusion-models-class/tree/72bf3dd739da641075dcff0ee54ce28e7a38a8ab) : DDPM avec diffusers, fine-tuning, guidance et Stable Diffusion, en notebooks qui tournent sur Colab.
