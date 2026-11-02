# B5 — notes de rédaction de la fiche (session 71)

*Pour le rédacteur des exercices, des indices, des solutions et du notebook. Ce fichier n'est pas lu par l'apprenant ; à supprimer (ou à fondre dans `EN_COURS.md`) à la fin du chapitre.*

Fiche : `01_fiche.md`, ≈ 13 550 mots (≈ 108 min à 125 mots/min ; l'en-tête garde 1,8 h, valeur du contrat pour un chapitre bonus). Figures : `tools/chapters/figures_chB5.py` (≈ 4 s, rien d'entraîné ni de téléchargé) → `figures/chaine.png`, `plannings.png`, `unet.png`, `ddpm_ddim.png`, `guidance.png`. `tests/infra/test_terminology.py` passe.

## 1. Chiffres vérifiés (exécutés le 2 novembre 2026, venv313 : NumPy 2.1.3, torch 2.11.0+cpu ; diffusers 0.40.0)

**Conventions de la référence** : formules de la fiche = article (pas $t = 1..T$, $\bar\alpha_0 = 1$) ; code = indices $0..T-1$ (le pas $t$ est l'indice $t - 1$), comme la docstring figée de `diffusion.py` et diffusers. **Tout énoncé qui dit « pas 100 » doit dire s'il s'agit du pas de l'article ou de l'indice** (B5.3, B5.13, B5.15).

Contre diffusers 0.40.0 (installé hors du venv pour ces essais, en `--target` dans le scratchpad ; il se trouve que diffusers 0.40.0 était déjà dans `/root/venv313`, installé à 20 h 00 par l'agent de la référence, et `requirements.txt` ligne 61 le fige) :
- `DDPMScheduler(beta_schedule="linear")` = `np.linspace(1e-4, 0.02, 1000)` ; `"squaredcos_cap_v2"` = formule cosinus de la docstring (écart max 2,2e-8, diffusers en float32).
- `DDPMScheduler.step` (variance `fixed_small` = $\tilde\beta_t$, défaut) = algorithme 2 avec le même $z$ (écart 1,7e-7) ; `_get_variance(t)` = `posterior_variance[t]`. **`clip_sample=True` par défaut** (écrêtage de $\hat x_0$ à ±1) : à désactiver pour comparer à `ddpm_step(..., clip_x0=None)`.
- `DDIMScheduler` : `set_alpha_to_one=True`, `steps_offset=0` ; espacement par défaut `"leading"` → pour 50 pas sur 1 000 : `[980, 960, …, 20, 0]` ; `"trailing"` → `[999, 979, …, 19]` ; `"linspace"` → `[999, 979, 958, …, 20, 0]`. Un pas DDIM $t \to t-1$ ($\eta = 0$) = notre formule (écart 8e-8).
- Guidance des pipelines : `noise_uncond + guidance_scale * (noise_text - noise_uncond)`, active si `guidance_scale > 1` ⇒ `guidance_scale` $= 1 + w$ (défaut SD : 7,5 ⇔ $w = 6{,}5$).
- `UNet2DModel` : `flip_sin_to_cos=True` (cosinus puis sinus) ; le code DDPM original : sinus puis cosinus ; `positional_encoding` (B3) : alternés.

Plannings (indices de la docstring) :

| | linéaire | cosinus |
|---|---|---|
| $\bar\alpha$ au dernier indice, $T = 1\,000$ | **4,0358e-5** (approx. $e^{-\sum\beta}$ = 4,3186e-5, $\sum\beta = 10{,}05$) | 2,43e-9 (dernier $\beta$ écrêté à 0,999) |
| $T = 400$ | **0,01747** (SNR ≈ 0,018 : il reste $\sqrt{0{,}0175} = 13\,\%$ de l'image) | 1,5e-8 |
| $T = 400$, bornes × 1 000/T (code de Nichol) | 3,64e-5 | — |
| $T = 500$ / $T = 200$ / $T = 50$ | 6,35e-3 / 0,1322 / 0,603 | ≈ 0 |
| $\bar\alpha$ à l'indice $T/2 - 1$ | 0,0786 ($T = 1\,000$) | 0,4938 (indépendant de $T$) |
| premier $\beta$ ($T = 1\,000$) | 1e-4 | 4,128e-5 ($\sqrt{\beta_0} = 0{,}0064 < 1/127{,}5$) |

**⚠️ Pour le notebook (FAST_MODE, T = 400 dans le contrat)** : `linear_beta_schedule(400)` avec les bornes par défaut laisse $\bar\alpha_T = 0{,}0175$ — on n'échantillonne plus depuis un bruit pur (piège « SNR final non nul » de la fiche). Passer `beta_start=2.5e-4, beta_end=0.05` (× 1 000/T, comme improved-diffusion, commit `1bc7bbb`), ou le planning cosinus.

SNR, $T = 1\,000$ (premier **indice** où le SNR passe sous le seuil) — **non donnés dans la fiche** :

| seuil | 10 | 1 | 0,1 | 0,01 |
|---|---|---|---|---|
| linéaire | 93 | 259 | 485 | 674 |
| cosinus | 188 | 496 | 803 | 936 |

$\bar\alpha$ et SNR aux indices 99, 299, 599 (B5.13 si « pas » = indice + 1) : linéaire 0,8970 / 0,3964 / 0,02588 (SNR 8,71 / 0,657 / 0,0266) ; cosinus 0,9721 / 0,7869 / 0,3408 (SNR 34,8 / 3,69 / 0,517). Aux indices 249, 499, 749 : linéaire SNR 1,101 / 0,0853 / 0,00336 ; cosinus 5,54 / 0,976 / 0,169.

Coûts mesurés (2 cœurs, machine partagée, sous `flock`) : U-Net de la fiche (canaux 32-64-128, un bloc résiduel par étage, group norm 8 groupes, encodage 128 + MLP 2 couches, convolutions de stride 2, convolutions transposées 4×4) : **740 865** paramètres, **2,01e8 FLOP** par image et par passe (`FlopCounterMode`, 2 par MAC), batch de 64 : 0,134 s min, 0,138 s médiane ⇒ DDPM 1 000 pas ≈ 135 s, DDIM 50 pas ≈ 7 s pour 64 images. Bloc `ResBlock(64, 64, t_dim=128)` : 82 368 (détail dans la fiche). Script : `scratchpad/s71fiche/unet_cost.py` (non versionné).

Articles relus (PDF ou API arXiv) : DDPM v2 (T = 1 000, β 1e-4 → 0,02, FID 3,17 / IS 9,46, 35,7 M paramètres, EMA 0,9999, Adam 2e-4, batch 128, 800 k pas en 10,6 h sur TPU v3-8, 256 images en 17 s ; $\sigma_t^2 = \beta_t$ ou $\tilde\beta_t$ « similaires » ; prédire $x_0$ moins bon, $\varepsilon$ ≈ $\tilde\mu$ avec la VLB, nettement mieux avec $L_\text{simple}$) ; iDDPM (cos², s = 0,008, β ≤ 0,999, 20 % de la chaîne sautable, recall « much higher for similar FID ») ; DDIM v4 (10× à 50× plus rapide) ; CFG (éq. 6 avec $(1 + w)$ ; $p_\text{uncond}$ 0,1 ≈ 0,2 > 0,5 ; meilleur FID à w = 0,1 ou 0,3, meilleur IS à w ≥ 4, ImageNet 64 et 128) ; LDM (f de 4 à 16 ; 150 à 1 000 jours de V100 pour un DM en pixels) ; carte modèle SD v1-4 (f = 8, 4 canaux, CLIP ViT-L/14) ; SD3 (z_t = (1 − t)x_0 + tε, d = 16 canaux, logit-normal, MM-DiT, décalage selon la résolution §5.3.2) ; DiT v2 (FID 2,27, adaLN-Zero, p = 2, latent 32 × 32 × 4) ; Flow Matching v2 (bruit à t = 0, données à t = 1 : sens inverse de SD3 et du contrat) ; rectified flow ; Salimans & Ho (8 192 → 4 pas, FID 3,0) ; consistency models (un pas, FID 3,55 CIFAR-10) ; ADD (1 à 4 pas) ; FLUX.1 [schnell] (carte modèle : 12 B, rectified flow transformer, 1 à 4 pas, Apache 2.0) ; FLUX.2 (blog BFL, 25 nov. 2025 : [dev] 32 B, VLM Mistral-3 24 B, 4 MP) ; Carlini et coll. v1 (SD : 350 000 légendes × 500 = 175 M images → 94 extraites, 109 avec l'inspection visuelle ; Imagen 23 sur 1 000 ; CIFAR-10 : 1 280 images = 2,5 % ; meilleur DDPM 2× StyleGAN-ADA à FID égal ; plus de mémorisation quand le FID s'améliore ; duplication) ; Somepalli et coll. ; Stein et coll. (17 mesures, plus grande évaluation humaine, DINOv2-ViT-L/14, mémorisation non détectée par les mesures) ; Lin et coll. (SNR final nul, luminosité moyenne) ; Hang et coll. (min-SNR, 3,4× plus rapide) ; Qwen-Image-2.0 (mai 2026, Qwen3-VL + MMDiT) et Qwen-Image-2.0-RL (juin 2026, RLHF à modèles de récompense) ; Wan (DiT vidéo, 14 B) ; fiche système OpenAI (PDF : « 4o image generation is an autoregressive model », « C2PA metadata on all assets », mars 2025).

Web (2 novembre 2026) : AI Act art. 50 applicable le 2 août 2026, délai au 2 décembre 2026 pour le marquage des systèmes déjà sur le marché (règlement omnibus 2026/1744, aligné sur B4) ; code de bonnes pratiques sur la transparence des contenus générés : 1ᵉʳ projet 17 déc. 2025, 2ᵉ 3 mars 2026, version finale 10 juin 2026, ≈ 190 signataires fin juillet (page de la Commission, mise à jour du 19 oct. 2026) ; il propose métadonnées signées + filigrane invisible (sous-mesures 1.1.1 et 1.1.2, d'après l'IPTC), interopérabilité des détecteurs au 2 février 2027 ; spécification C2PA 2.3 (5 janv. 2026) et 2.4 (1ᵉʳ avril 2026) ; classement texte vers image d'Artificial Analysis : en tête GPT Image 2.5 (OpenAI), Nano Banana 2.1 (Google), MAI-Image-2.6 (Microsoft), puis Grok Imagine, FLUX 3 Image, Seedream 5.0, Qwen-Image-3.0…

**Colab** : `pip-freeze.txt` de `googlecolab/backend-info` (branche `main`, lu le 2 nov. 2026) affiche **diffusers 0.42.1**, transformers 5.22.0, numpy 2.4.6, torch 2.11.0 : à revérifier au prochain contrôle des versions (BIBLE §21) ; la fiche dit seulement que le workbook fige 0.40.0.

## 2. Valeurs que la fiche utilise déjà (ne pas les reprendre dans un exercice)

- Planning jouet linéaire $T = 5$, $\beta$ de 0,02 à 0,10 : $\bar\alpha$ = 0,98 ; 0,9408 ; 0,8844 ; 0,8136 ; 0,7322 ; SNR 49 ; 15,89 ; 7,65 ; 4,36 ; 2,73 ; $\sqrt{\bar\alpha}$ et $\sqrt{1-\bar\alpha}$ tabulés.
- Pixel $x_0 = 0{,}6$, $\varepsilon = -0{,}5$ au pas 4 (indice 3) → $x_4 = 0{,}3253$ ; posterior exact 0,4358, $\tilde\beta_4 = 0{,}0496$ (coefficients 0,4036 et 0,5951).
- Pas DDPM : $\varepsilon_\theta = -0{,}4$, $z = 1{,}2$ → $\mu = 0{,}4165$ (aussi par $\hat x_0 = 0{,}5521$), $\sigma = 0{,}2228$, $x_3 = 0{,}6838$.
- Pas DDIM du pas 4 au pas 2 (indice 3 → 1) : 0,4382 ; $\eta = 1$ et $t' = t - 1$ : $\sigma^2 = \tilde\beta_4$.
- Linéaire $T = 200$ (mêmes bornes) : $\bar\alpha_T = 0{,}1322$, approximation 0,134, « 36 % de l'image ».
- Cosinus : $s = 0{,}008$, premier $\beta$ ≈ 4,1e-5 ; dernier $\beta$ = 0,999.
- Gaussiennes : $\mathcal N(1, 0{,}3^2) + \mathcal N(-2, 0{,}4^2) = \mathcal N(-1, 0{,}5^2)$ ; produit $\mathcal N(x; 2, 1)\,\mathcal N(x; 5, 0{,}5) \propto \mathcal N(x; 4, 1/3)$.
- MSE : cibles ±1 → 0 (MSE 1) contre 1 (MSE 2).
- Guidance : $\varepsilon_c = (0{,}4 ; -1{,}0)$, $\varepsilon_\varnothing = (0{,}1 ; -0{,}6)$, $w = 0{,}5$ → $(0{,}55 ; -1{,}2)$ ; figure à $w = 0{,}5$ (écart-type moyen 0,33 → 0,27) ; piège `guidance_scale = 5` ⇔ $w = 4$ ; SD 7,5 ⇔ 6,5.
- Euler sur $y' = y$ : 2 ; 2,25 ; 2,441 ; 2,594 contre $e$.
- U-Net 740 865 paramètres, 0,2 GFLOP/image, 0,13 s par batch de 64 ; `ResBlock(64, 64, 128)` = 82 368 ; DDPM CIFAR-10 35,7 M (« 50 fois plus gros »).
- Diffusion latente : 512 × 512 × 3 = 786 432 → 64 × 64 × 4 = 16 384 (48 fois moins de valeurs, 64 fois moins de positions) ; DiT : 256 px → latent 32 × 32 × 4 → 256 tokens. **Le 🧮 B5.8 (1 024 × 1 024) peut garder 1 024 px mais doit changer de cas** (16 canaux de SD3, f = 8 ou 16, autre réseau), sans reprendre 48× ni 64×.
- Données 2D des figures : spirale (chaîne), mélange de 3 gaussiennes aux centres (−1,6 ; −0,6), (1,4 ; −1,0), (0,2 ; 1,5), écart-type 0,35, poids 0,5/0,3/0,2 (DDPM/DDIM, guidance). Le notebook peut garder l'anneau de gaussiennes du contrat.

## 3. Ce que la fiche ne dit pas (laissé aux exercices)

- Aucune valeur pour $T = 4$ ni $\bar\alpha_T$ du linéaire à $T = 1\,000$ (✏️ B5.2) : seulement la méthode $\ln\bar\alpha_T \approx -\sum\beta$ et l'exemple $T = 200$. La figure `plannings.png` coupe $\ln$ SNR à −8 (la fin du linéaire, ≈ −10, n'y est pas lisible) et ne marque pas la ligne SNR = 1.
- Aucun pas où le SNR passe un seuil (✏️ B5.3), aucune image MNIST bruitée ni aucun numéro de pas sur une figure (🔮 B5.13, 📈 B5.15). La fiche dit en général (Nichol) que la fin de la chaîne linéaire est « presque du bruit pur » et qu'on pouvait en sauter 20 %.
- ∂ B5.6 : la fiche donne le résultat, la règle de la somme de gaussiennes et l'amorce « deux pas = $\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}$ + deux bruits » ; ni la variance combinée ni la récurrence.
- ∂ B5.7 : résultat, Bayes « vraisemblance × forme fermée », règle du produit de densités ; pas la réécriture de $q(x_t \mid x_{t-1})$ comme densité en $x_{t-1}$ (moyenne $x_t/\sqrt{\alpha_t}$, variance $\beta_t/\alpha_t$), ni le calcul des précisions.
- Les effets de $w$ = 0, 1, 3, 7 sur Fashion-MNIST (🔮 B5.26) : la fiche donne le compromis général de Ho et Salimans (ImageNet : meilleur FID à petite guidance, IS à forte guidance, plus de guidance = moins de variété) et la figure 2D à $w = 0{,}5$ ; **le 🔮 doit donc porter sur ce que la fiche ne dit pas** (à partir de quel $w$ les images se dégradent, ordre des distances de Fréchet, saturation des pixels…).
- L'effet de la taille du dataset sur la mémorisation (🔬 B5.28) : rien n'est dit (Somepalli et coll. et Kadkhodaie et coll., ICLR 2024, arXiv:2310.02557v3, l'étudient : sources pour la solution, pas pour la fiche). La fiche cite seulement Carlini (duplication, qualité ↔ mémorisation, diffusion > GAN).
- Le nombre de pas DDIM suffisant, linéaire contre cosinus (🔬 B5.25), la comparaison flow matching / diffusion en 2D (🔬 B5.20), la courbe de loss de B5.23, le temps de B5.24 (seul l'ordre de grandeur sur un autre U-Net est donné), la distance de Fréchet DDPM / VAE / DCGAN (B5.29), le défi.
- Le 🐛 B5.22 : la fiche décrit séparément, comme pièges généraux, $[0, 1]$ contre $[-1, 1]$, le décalage d'un cran, le bruit au dernier pas (encadré de 205.5) et $\bar\alpha$ contre $\sqrt{\bar\alpha}$ (tableau des pièges) ; aucun renvoi au 🐛 ne les regroupe.
- Le contenu exact de l'architecture du U-Net de B5.21 (la fiche a son propre U-Net, 32-64-128) et du MLP 2D de B5.19 (seulement « concaténer l'encodage du pas aux coordonnées »).
- La boucle d'entraînement : seulement l'algorithme 1 en pseudo-code et `F.mse_loss(model(x_t, t), eps)` ; l'EMA par sa formule (ρ = 0,9999 de DDPM et l'avertissement qu'il faut l'adapter au nombre de pas ; aucune valeur conseillée).

## 4. Termes à ajouter au glossaire (chapitre B5)

processus direct (*forward process*) · processus inverse (*reverse process*) · planning de bruit (*noise schedule*) · planning linéaire, planning cosinus **(de bruit)** — à distinguer de l'entrée « planning en cosinus » (learning rate, 0B) · forme fermée (*closed form*) · diffusion à variance préservée (*variance preserving*) · rapport signal sur bruit (*signal-to-noise ratio*, SNR) et SNR final nul (*zero terminal SNR*) · posterior (d'un pas) · prédiction du bruit, loss simplifiée ($L_\text{simple}$) · échantillonnage ancestral (*ancestral sampling*) · échantillonneur (*sampler*) · DDPM · DDIM, η, sous-séquence de pas, inversion DDIM · U-Net · **connexion de saut** (*skip connection*, concaténation) — et corriger l'entrée « connexion résiduelle », dont la colonne anglaise donne aussi *skip connection* · group norm (*group normalization*) · encodage du pas de temps (*timestep embedding*) · moyenne mobile exponentielle des poids (EMA) — préciser l'entrée du ch. 19 · distillation (de l'échantillonneur), distillation progressive, modèle de cohérence (*consistency model*), distillation adversariale · guidance par classifieur (*classifier guidance*), guidance sans classifieur (CFG), échelle de guidance $w$ (et `guidance_scale` = $1 + w$), condition vide $\varnothing$ · score (d'une densité, $\nabla \ln p$) — autre sens que le « score » d'un classifieur (ch. 3) · v-prédiction (Salimans et Ho) · min-SNR · mémorisation (générateur d'images) — l'entrée B3 ne parle que des modèles de langue · diffusion latente (*latent diffusion*) · transformer de diffusion (DiT), MM-DiT, adaLN-Zero · flow matching, rectified flow, champ de vitesses (*velocity field*), méthode d'Euler · CLIP score · filigrane (*watermark*) · métadonnées de provenance, C2PA (*Content Credentials*).

Doublon existant à régler au glossaire : « deepfake » (ch. 27) et « hypertrucage » (B4) sont tous deux en colonne « Retenu » ; la fiche B5 suit B4 (« hypertrucage (*deepfake*) »).
