# B1 — notes pour le rédacteur des exercices (temporaire, à supprimer avant le commit)

> **Mise à jour du relecteur indépendant de la fiche (session 63)** : sections touchées ci-dessous, « Ce que la fiche utilise déjà » (latences, débits et rapport MobileNet/ResNet recalés sur de nouvelles mesures ; nouveaux faits : top-1/top-5 définis, Mio défini, `vit_b_16` n'accepte que 224 × 224, 196 et 100 classes de Stanford Cars et FGVC Aircraft, `layer4` = 64 % de ResNet-50, tête K = 5 = 0,04 % du réseau). La fiche ne renvoie plus à 🐛 B1.18 dans ses paragraphes sur la batchnorm et sur l'optimiseur créé avant la tête (le 🐛 garde ses quatre erreurs, la ligne du tableau section → exercices reste). La rubrique « Auto-évaluation » a été retirée (hors §12). Fiche : ≈ 11 300 mots.

Session 63. Fiche : `01_fiche.md` (≈ 11 300 mots après relecture, soit ≈ 90 min à 125 mots/min ; les bonus gardent la durée du contrat, 90 min). Figures : `python tools/chapters/figures_chB1.py` → `figures/strategies.png`, `matrice_decision.png`, `planning_lr.png`, `clip_zero_shot.png`, `backbones.png` (aucun entraînement, hors ligne, < 5 s).

## ⚠️ À faire par le coordinateur

- `chapitres/b1_transfer/` existe maintenant **sans** `EN_COURS.md` : `syllabus.published()` considère B1 comme publié, et deux tests échouent (`test_generated_documents_are_up_to_date[tableau_de_bord]`, `test_recall_headers_of_the_exercises_show_their_stars_and_minutes`). Avec un `EN_COURS.md`, `test_syllabus.py`, `test_repo_structure.py` et `test_terminology.py` passent (99 tests, vérifié).
- Glossaire : « foundation model » y est le terme **retenu** (colonne 1) avec « modèle de fondation » en français, alors que le contrat (201.8, B1.E4) et la fiche disent « modèle de fondation (*foundation model*) ». À trancher (§5) et, si besoin, ligne au §22.
- Termes définis par la fiche, à ajouter au glossaire : backbone (*tronc*), linear probe (*sonde linéaire*), embedding figé, learning rates discriminatifs (*discriminative fine-tuning*), groupes de paramètres (*param groups*), dégel progressif (*gradual unfreezing*), LP-FT, L2-SP, EWC, distance de domaine (*domain gap*), transfert négatif (*negative transfer*), oubli catastrophique (*catastrophic forgetting*), few-shot (*à quelques exemples*), prototype de classe, support, zéro-shot (*zero-shot*), gabarit (*prompt template*), ensemble de prompts (*prompt ensembling*), apprentissage contrastif, latence (*latency*), débit (*throughput*), modèle dérivé.
- Formulaire : les 7 formules du contrat sont dans la fiche (201.3, 201.4, 201.6, 201.8), à recopier dans `annexes/formulaire.md`.

## Chiffres vérifiés en exécution (torch 2.11, torchvision 0.26, transformers 5.21) — absents de la fiche, réservés aux exercices

**ResNet-18** (`ResNet18_Weights.IMAGENET1K_V1`, = DEFAULT) : `num_params` 11 689 512 ; `fc` 512 → 1000 (513 000 paramètres) ; `conv1` 9 408, `bn1` 128, `layer1` 147 968, `layer2` 525 568, `layer3` 2 099 712, `layer4` 8 393 728 ; 20 batchnorms, 9 600 paramètres de batchnorm ; acc@1 69,758, acc@5 89,078 ; `_ops` 1,814 GFLOPS ; `_file_size` 44,661 Mio ; `min_size` (1, 1) ; `transforms()` : resize 256, crop 224, BILINEAR, moyenne/écart-type ImageNet.
- Tête à 10 classes : 512 × 10 + 10 = **5 130** ; réseau entier avec cette tête : **11 181 642** ; `layer4` + tête : **8 398 858**.
- Formes (N, C, H, W) : à 224 → stem (64, 56, 56), layer1 (64, 56, 56), layer2 (128, 28, 28), layer3 (256, 14, 14), layer4 (512, 7, 7), avgpool (512, 1, 1) ; à 160 → layer4 5 × 5 ; à 96 → 3 × 3 ; à 64 → 2 × 2 ; à 32 (sans agrandir) → stem 8 × 8, layer4 **1 × 1**.

**MobileNetV3-small** (`IMAGENET1K_V1`) : `num_params` 2 542 856 (features 927 008, classifier 1 615 848) ; classifier = Linear(576 → 1024), Hardswish, Dropout(0,2), Linear(1024 → 1000) ; embedding **576** à l'entrée de `classifier`, **1024** entre ses deux couches denses ; acc@1 67,668, acc@5 87,402 ; `_ops` 0,057 ; `_file_size` 9,829 Mio ; transforms : resize 256, crop 224, BILINEAR, ImageNet.

**DINOv2-small** (`facebook/dinov2-small`, révision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, Apache 2.0, `model.safetensors` 88,2 Mo) : 22 056 576 paramètres ; hidden_size **384**, 12 couches, 6 têtes, patch 14 ; processeur `BitImageProcessor` (backend torchvision ; la variante PIL s'appelle `BitImageProcessorPil`) : shortest_edge 256, crop 224 × 224, resample 3 (BICUBIC), moyenne/écart-type ImageNet ; sorties pour 2 images de 224 : `last_hidden_state` (2, 257, 384), `pooler_output` (2, 384) = token CLS (égal à `last_hidden_state[:, 0]`).

**CLIP ViT-B/32** (`openai/clip-vit-base-patch32`, main = `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, aucun champ de licence, model card : tout déploiement hors champ) : 151 277 313 paramètres (vision 87 456 000, texte 63 165 952) ; projection 512 (visual 768 → 512, text 512 → 512, sans biais) ; `config.json` : `logit_scale_init_value` = 2.6592 (valeur **initiale**, e^2,6592 ≈ 14,28) ; poids entraîné `logit_scale` = 4.605170 → `exp()` = **100,00** ; processeur : shortest_edge 224, crop 224, BICUBIC, mean [0.48145466, 0.4578275, 0.40821073], std [0.26862954, 0.26130258, 0.27577711].
- Le dépôt n'a qu'un `pytorch_model.bin` (605 Mo) : transformers 5.21 a téléchargé un `model.safetensors` d'une autre révision (`c237dc49…`, conversion du Hub). Téléchargement ≈ 10 s ici.
- transformers 5 : `get_text_features` / `get_image_features` rendent un `BaseModelOutputWithPooling` ; l'embedding projeté (N, 512) est `.pooler_output`. `torch_dtype=` affiche « `torch_dtype` is deprecated! Use `dtype` instead! ».
- `save_file(clip.state_dict(), …)` échoue (`visual_projection.weight` non contigu) : pour 🛠️ B1.27, rester sur le ResNet fine-tuné, ou `{k: v.contiguous() …}`.

**Divers** : `torch.load(..., weights_only=None)` → True par défaut (depuis 2.6) ; `list_models(module=torchvision.models)` → 80 classifieurs ; `pretrained=` encore accepté avec avertissement (« deprecated since 0.13 ») ; `freeze_batchnorm` après `model.train()` laisse `running_mean` intact (vérifié), un simple `train()` le modifie.

## Ce que la fiche utilise déjà (un exercice doit changer de cas, pas seulement de nombres)

- **Formes et paramètres** : ResNet-50 à **160 px** (table complète des formes), d = 2048, tête K = 5 → 10 245 ; ResNet-50 entier avec cette tête 23 518 277 ; `layer4` + tête 14 974 981 (`layer4` seul 14 964 736) ; une convolution 1 × 1 512 → 2048 = 1 048 576 poids ; EfficientNet-B0 d = 1280, K = 5 → 6 405 ; DINOv2-base d = 768, 86 580 480 paramètres, 257 tokens. → **B1.1 et B1.4 sur ResNet-18 sont libres** (ne pas reprendre 160 px ni K = 5). Ne pas utiliser 512, 5 130, 11 181 642, 8 398 858 comme exemple ailleurs (fiche, flashcards, indices) : ce sont des réponses.
- **k-NN cosinus** (201.3) : 4 vecteurs de dimension 3, q = (2, 1, 1), a₁ = 2q ; k = 1 → « chien », k = 3 → « chat » ; cosinus 0,816 / 1 / 0,913 / 0,577 ; l'euclidienne change le plus proche. → B1.2 : six embeddings, autres vecteurs, autre piège (par ex. une égalité à départager, ou une norme qui trompe autrement).
- **Learning rates discriminatifs + warmup** (201.4) : η_tête = 2 × 10⁻³, γ = 0,5, trois groupes (tête, layer4, layer3), t_warmup = 4 → tête 0 ; 0,5 ; 1 ; 1,5 ; 2 × 10⁻³ aux pas 0 à 4 ; convention min(1, t/t_warmup) (η = 0 au pas 0, comme `cosine_lr` du ch. 19). Figure : warmup 6 pas sur 60, dégel toutes les 2 epochs. ULMFiT : γ = 1/2,6 cité. → B1.3 : autres η_tête, γ, nombre de groupes et t_warmup (éviter 2e-3, 0,5 et 4).
- **L2-SP** : λ = 0,01, θ₀ = 0,8, θ = 1,1 → Ω = 0,00045, ∇ = 0,003. La fiche donne Ω et ∇Ω (contrat) mais **ne dérive pas** l'effet sur un pas de SGD (θ − θ₀ multiplié par 1 − ηλ, rappel géométrique vers θ₀, comparaison au weight decay vers 0) : c'est la matière de ∂ B1.6.
- **Zéro-shot CLIP** : cosinus 0,31 / 0,27 / 0,22, s = 100 → 0,982 / 0,018 / 0,0001 ; avec s = 1 → 0,348 / 0,334 / 0,318 (le rôle de s est donc expliqué). La fiche affirme s = exp(logit_scale) = 100 pour ce modèle. → B1.5 : autres cosinus, idéalement quatre classes, et une question qui ne soit pas « que se passe-t-il avec s = 1 ».
- **Prototypes** : 2D, pomme (0,8 ; 0,2) / poire (0,3 ; 0,8), q = (0,6 ; 0,5) → 0,900 / 0,869.
- **Latence** (2 cœurs, 224 px, fourchettes sur plusieurs exécutions, machine parfois chargée) : MobileNetV3-large ≈ 18-30 ms, EfficientNet-B0 ≈ 28-46, ConvNeXt-T ≈ 85-105, ResNet-50 ≈ 95-140 ms ; débits ≈ 55-70 / 35-45 / 14-16 / 11-12 images/s ; « 19 fois moins de FLOPS, seulement 4 à 6 fois plus rapide ». ResNet-50 : 25 557 032 paramètres, ≈ 102 Mo (97,5 Mio) en float32 ; fichier safetensors à tête K = 5 : 94,3 Mo. Les trois modèles de B1.26 sont exclus de la table et de la figure `backbones.png`.
- **Coût ∝ pixels** : (224/160)² ≈ 2 donné en exemple.

## Ce que la fiche ne dit pas (à garder pour les 🔮, 📄 et 🔬)

- **🔮 B1.12** : la fiche donne seulement la mécanique des formes (÷ 32, pooling adaptatif) et l'argument d'échelle (« les filtres sont réglés pour des motifs de la taille d'une image de 224 ») ; aucune accuracy, aucune prédiction sur 32 × 32. Le sens de l'effet est donc suggéré : faire prédire une **grandeur** (accuracy du probe à 32 contre 224, ou classes ImageNet prédites).
- **🔮 B1.20** : la fiche dit qualitativement que le pré-entraînement compte davantage avec peu de données (cœur du chapitre) et cite He et coll. (2019) : faire prédire l'**écart chiffré** à 10 images par classe, ou le point de croisement, pas le sens.
- **📄 B1.10** : la fiche donne le dispositif (deux moitiés de classes d'ImageNet, couches transplantées) et « général → spécifique ». Elle ne dit **pas** : la co-adaptation fragile (creux du BnB aux couches 3 à 6), le gain de généralisation de AnB+ (+1,6 % en moyenne, +2,1 % à partir de 5 couches), les chutes d'environ 8 % et 25 % (moitiés semblables / naturel contre artificiel), ni que des features d'une tâche lointaine battent des poids aléatoires.
- **🔬 B1.16** : la fiche suggère de sonder les blocs intermédiaires sur un domaine lointain, sans dire lequel gagne.
- **🔬 B1.23** : aucun chiffre d'accord top-1.
- **🐛 B1.18** : les quatre erreurs plantées (lr 1e-1 sur tout le réseau, batchnorm en train avec des batches de 4, normalisation oubliée, optimiseur créé avant la nouvelle tête) sont toutes des lignes du tableau des pièges et des paragraphes de 201.4-201.5 : le 🐛 doit les faire **trouver dans le code** (diagnostics enchaînés), et ses messages ne doivent pas recopier ces lignes.
- **Chiffres d'articles donnés dans la fiche** (ne pas en faire des réponses) : Kornblith r = 0,99 (features figées) et 0,96 (fine-tuning), 16 réseaux, 12 datasets, Stanford Cars 92,7 / 93,3, Aircraft 88,8 / 89,0 ; LP-FT ≈ +2 points ID, −7 OOD par rapport au fine-tuning direct contre le probe ; CLIP +1,3 (gabarit) et +3,5 (80 gabarits), 400 M de paires, 32 768 par mini-batch, 12 jours sur 256 V100 ; ImageNet-1k 1 281 167 images.
- **🧮 B1.7** : la fiche ne chiffre pas le coût d'un pré-entraînement ImageNet (seulement CLIP) ; elle renvoie à 6PN (ch. 23) via « 🧮 B1.7 te fait chiffrer l'écart ». `_ops` de ResNet-50 (4,1 GFLOPS) est donné.
- **🏆 B1.28** : aucune recette chiffrée.

## Licences vérifiées (31 octobre 2026)

dinov2-small/base : Apache 2.0 (CC BY-NC à la sortie, Apache 2.0 depuis le 31 août 2023) · clip-vit-base-patch32 : pas de champ licence ; model card « Any deployed use case … is currently out of scope » · dinov3-* : « dinov3-license », dépôt gated, « Built with DINOv3 », usages militaires interdits · siglip2-* : Apache 2.0 · ImageNet : « non-commercial research and educational purposes » · torchvision : « It is your responsibility to determine whether you have permission ».

## Caches

Téléchargés dans les caches par défaut (rien dans le dépôt) : resnet50 (97,8 Mo), efficientnet_b0 (20,5 Mo) dans `~/.cache/torch/hub/checkpoints` ; `facebook/dinov2-small`, `facebook/dinov2-base`, `openai/clip-vit-base-patch32` dans `~/.cache/huggingface/hub`.
