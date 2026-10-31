# Audit P5 parties V et VI — A8 : exactitude du ch. 29, des annexes et des data cards

**Périmètre** : `chapitres/ch29_datasets` (fiche, exercices, indices, solutions .md et .ipynb, notebook d'exercices, flashcards), `data/cards/` en entier (README, trap, cifar10, fashion_mnist, mnist, holmes, verne, sunspots, images, pretrained, rl_envs, adult, penguins), ce que disent la documentation et les fiches des ajouts V-VI de `src/wb/datasets.py` (Fashion-MNIST, CIFAR-10, `make_trap_dataset`, Hugging Face, OpenML), les entrées ch. 21-29 de `annexes/glossaire.md`, `formulaire.md`, `erreurs_frequentes.md`, `ressources.md`, `metiers.md`, les renvois vers le ch. 29 (CP6, MP6, cheatsheets, suivi), contre `book_ch29.txt` et le web. Commit `b961aa3`.
**Scripts de vérification** (dossier `scripts/` de ce répertoire, exécutés dans `COPIE/`) : `nbdump.py` (extraction des cellules et sorties des notebooks), `trap_check.py` (les huit défauts de `make_trap_dataset`, graines 2916 et 7, crosstab code/fruit), `variants.py` (variantes de 29.8, 29.9 et 29.13 recalculées), `v14.py` (29.14 : fine-tuning `center` et variante `naive`) ; contrôles ponctuels en ligne de commande (message de `load_boston`, README des images de scikit-learn, `with_format("numpy")` de `datasets` hors ligne, tailles des caches et des poids).

## Recalculs effectués (aucun écart)

- ✏️ 29.1 : 54,88 Mo ; 219,52 Mo ; 184,32 Mo ; 0,737 Go ; 318 877 ; Vrai. ✏️ 29.2 : 140 ; 30 ; 0,60 ; 0,90 ; 0,00. 🧮 29.3 : 2,7 To ; 216 000 s = 2,5 j ; 2 160 s = 0,6 h ; 585 To ; 21,2 To et 1,35 To ; variantes 192,9 Go (ImageNet) et 2,4 s (Fashion-MNIST, ≈ 30 Mo, fichiers .gz de Zalando).
- 29.7 : 212/569 = 0,373 ; Wine 48/178 = 0,270. 29.8 : sorties 0,916 / 0,812 / 0,116 ; variante 5 epochs **0,938 / 0,827** (recalculée). 29.9 : 0,198, confiance 0,761. 29.13 : 500, k = 3, 0,728 ; variante `copies=2` **0,936, 190/250, 0,736** (recalculée) ; $0{,}25^4 \times 500 \approx 1{,}95$. 29.14 : 0,198 / 0,576 / 0,661 ; 0,793 / 0,892 / 0,932 ; variante `naive` **0,826** (recalculée). 29.12 : 2 809 / 2 799 / 857, 52 doublons, 0,76, 92 % et 95 %. 29.16 : 13, 17 (18 paires), `acidite`, `pays`, `code_qualite`, 39 valeurs de 58,5 à 100,3, 11 labels, coing 24 (1,95 %), 1 200 lignes ; graine 7 : 13, 17, 38, 12 (tout recalculé par `trap_check.py`).
- Rappels : 2044, 1961, 16, 344, 3. Quiz : les 40 réponses enregistrées concordent avec la fiche et le livre.
- Formulaire ch. 21 à 29 : chaque formule revérifiée (taille de sortie avec dilatation, champ récepteur, convolution transposée, gradients d'une convolution, FGSM, LSTM/GRU dans l'ordre de PyTorch, nombre de paramètres, variance du dropout inversé $\frac{p}{1-p}h^2$, max-norm et `renorm`, KL gaussienne et ses gradients, reparamétrisation, $P(\lVert z\rVert \le r) = 1 - e^{-r^2/2}$, $V(D^*, G) = -\ln 4 + 2\,\mathrm{JS}$, dérivées des deux losses de $G$ par rapport au logit, $C\,G_{ii} = \mu_i^2 + \sigma_i^2$, gradient de la loss de style $\frac{4}{CHW}(G - G_s)F$) : aucune erreur trouvée.

## Vérifications web (13 affirmations, toutes confirmées sauf mention)

1. AI Act, art. 53(1)(d) : modèle de résumé publié le 24 juillet 2025, obligation applicable le 2 août 2025, le 2 août 2027 pour les modèles déjà sur le marché ([WilmerHale](https://wilmerhale.com/en/insights/blogs/wilmerhale-privacy-and-cybersecurity-law/european-commission-releases-mandatory-template-for-public-disclosure-of-ai-training-data)). Fiche §« documenter » et fiche du ch. 28 : justes et cohérentes.
2. Fashion-MNIST : licence MIT, 60 000 + 10 000, noms des classes, Xiao, Rasul, Vollgraf 2017 ([dataset card du Hub](https://huggingface.co/datasets/zalando-datasets/fashion_mnist/blob/c768752bb4d0087d5b71a06869167377c5ef20f5/README.md?code=true)).
3. CIFAR-10 : aucune licence formelle (« unknown » sur le Hub), rapport de Krizhevsky 2009, dérivé de 80 Million Tiny Images ; **téléchargement Hugging Face de 143,6 Mo**, pas 170 ([card `cifar10`](https://huggingface.co/datasets/cifar10/blob/main/README.md?code=true)) → constat 7.
4. SILSO : CC BY-NC 4.0, DOI 10.24414/qnza-ac80 ([ROB](https://www.astro.oma.be/doi/ROB-SIDC-SILSO_SunspotNumberV2.html)).
5. MNIST : CC BY-SA 3.0 d'après la page d'origine (reprise par [aiwiki](https://www.aiwiki.ai/wiki/mnist/raw)) ; la [card du Hub](https://huggingface.co/datasets/mnist/blob/main/README.md) affiche « MIT » → suggestion 24.
6. Papers with Code fermé le 24 juillet 2025, remplacé par « Trending Papers » du Hub ([hyper.ai](https://hyper.ai/en/news/42900)).
7. Keras 3 garde `boston_housing` avec un avertissement éthique sur la variable B, `test_split=0.2` ([keras.io](https://keras.io/api/datasets/boston_housing)).
8. LAION-5B retiré le 19 décembre 2023 (rapport de D. Thiel, Stanford Internet Observatory), Re-LAION-5B publié le 30 août 2024, ≈ 5,8 milliards de liens ([LAION](https://laion.ai/blog/relaion-5b/)).
9. `kagglehub` : « Authenticating is only needed to access public resources requiring user consent or private resources » ([PyPI](https://pypi.org/project/kagglehub/1.0.3/)).
10. Northcutt, Athalye, Mueller : « an average of at least 3.3% errors across the 10 datasets » ([arXiv:2103.14749](https://arxiv.org/abs/2103.14749)).
11. Speech Commands : 64 727 enregistrements et 30 mots (v0.01), 105 829 et 35 mots (v0.02), CC BY 4.0 ([Hub](https://www.huggingface.co/datasets/google/speech_commands)).
12. CelebA : 202 599 images, 10 177 identités, « non-commercial research purposes only » ([CUHK](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)).
13. Gutenberg n° 800 : *Le tour du monde en quatre-vingts jours*, « Produced by ebooksgratuits » ([gutenberg.org](https://www.gutenberg.org/ebooks/800)).

Vérifiés hors web : message de l'`ImportError` de `load_boston` (scikit-learn 1.6.1), README des photos de scikit-learn (CC BY 2.0, danielbuechele, lien vultilion), `HfApi`/`DatasetCardData["license"]`, `with_format("numpy")[:]["image"]` → `(10000, 28, 28)` `uint8`, label du premier exemple d'entraînement de Fashion-MNIST = 9.

## Constats

### 1. MINEUR · `src/wb/datasets.py:758-761` (`make_trap_dataset`, défaut 7) → `chapitres/ch29_datasets/05_solutions.md:206-207`, cellule 73 de `05_solutions.ipynb` (`tools/chapters/ch29_code.py`), `annexes/cheatsheets/pandas.md:112`
**Extrait :** « `variants = ["pome", "Poire", "poire ", "pommes", "POMME", "poir"]` / `df.loc[typos, "fruit"] = [variants[i % len(variants)] for i in range(len(typos))]` » ; solution e) : « **"code_qualite"** (sa lettre, M, R ou C, recopie le fruit) » ; cheatsheet : « une colonne qui recopie le label : chaque valeur n'apparaît que pour une classe ».
**Problème :** les 11 fautes de frappe sont tirées sans regarder le vrai fruit : trois lignes reçoivent la faute d'un **autre** fruit. Le crosstab que la solution elle-même conseille (sortie de la cellule 73) le montre : la lettre M apparaît sous `Poire` (1) et `"poire "` (1), la lettre R sous `pommes` (1). Avec la graine 7 de la variante, c'est pire (« pommes » sur un coing, « POMME » sur une poire, trois `"poire "` sur des pommes). Conséquences : (a) le défaut « labels mal écrits » cache en fait des **labels faux**, et `clean_16` de la solution (`.replace({"pome": "pomme", …})`, `str.lower()`) les « corrige » vers le mauvais fruit, soit trois erreurs de label introduites par le nettoyage de référence ; (b) l'affirmation « sa lettre recopie le fruit » et le critère de la cheatsheet sont contredits par la sortie que l'apprenant a sous les yeux, ce qui peut le faire douter de e) ou le pousser à « réparer » les labels avec la colonne fuyante.
**Correction :** tirer la faute dans la famille du vrai fruit, sans consommer d'aléa (le flux de `rng` ne change pas, donc aucune réponse enregistrée ne change : 13, 17, 11, coing…) :
```python
variants = {"pomme": ["pome", "pommes", "POMME"], "poire": ["Poire", "poire ", "poir"], "coing": ["Coing", "coings"]}
true = df.loc[typos, "fruit"].tolist()
df.loc[typos, "fruit"] = [variants[t][i % len(variants[t])] for i, t in enumerate(true)]
```
puis réexécuter le notebook de solutions (les accuracies 0,991 et 0,963 de la solution peuvent bouger d'un millième) et relancer `build_answers.py --check`. Mettre à jour la variante (graine 7 : le nombre de labels invalides reste 12).

### 2. MINEUR · `chapitres/ch29_datasets/01_fiche.md:9` et `:159` (prérequis, « Liens avec les autres chapitres »)
**Extrait :** « ch. 23 (classifieur PyTorch sur MNIST) » ; « **Ch. 23** : `Dataset`, `DataLoader`, `torchvision.datasets` ; le classifieur MNIST réutilisé en 29.8 et 29.14. »
**Problème :** le ch. 23 du workbook traduit le livre sur **Fashion-MNIST** (sa fiche, l. 3 : « traduit chaque étape en PyTorch moderne, sur **Fashion-MNIST** »), et le réseau de 29.8 et 29.14 n'est pas celui du ch. 23 : c'est `train_mlp` (784-128-10), défini dans la cellule d'outils du notebook. `Dataset` et `DataLoader` sont introduits au ch. 20 (glossaire, l. 713).
**Correction :** l. 9 : « ch. 20 et 23 (`Dataset`, `DataLoader`, un classifieur PyTorch sur Fashion-MNIST) » ; l. 159 : « **Ch. 20 et 23** : `Dataset`, `DataLoader`, `torchvision.datasets` ; une boucle d'entraînement comme celle de `train_mlp`, le petit réseau de 29.8, 29.9 et 29.14. »

### 3. MINEUR · `chapitres/ch29_datasets/01_fiche.md:62-64` (§29.1)
**Extrait :** « Il insiste sur deux précautions : **lire les règles** […] et **inspecter** les données […]. Il ajoute qu'une liste de liens vieillit vite […]. / Les trois précautions ont pris de l'importance depuis 2018. Côté droit : […] Côté histoire : […] Côté qualité : […] »
**Problème :** « deux précautions » puis « les trois précautions » ; et les trois « côtés » du second paragraphe (droit, histoire, qualité) ne sont pas les deux précautions du premier plus la remarque sur les liens. L'indice 1 de Q2 (« le deuxième paragraphe du §29.1 de la fiche (les trois précautions) ») et celui de 29.5 renvoient à ce passage.
**Correction :** « Il insiste sur trois points : **lire les règles** attachées à un dataset (…), **inspecter** les données à la main (…), et se souvenir qu'une liste de liens vieillit vite (…). / Les trois ont pris de l'importance depuis 2018. Côté droit : … »

### 4. MINEUR · `chapitres/ch29_datasets/05_solutions.md:177` (29.11, « Dans tes notes »)
**Extrait :** « seules CC0, CC BY, MIT, Apache ou ODbL (avec ses conditions de partage) permettent un produit »
**Problème :** « seules » est faux : CC BY-SA (et les licences à partage à l'identique en général) permettent aussi un usage commercial, avec l'obligation de partager les dérivés sous la même licence, ce que dit la réponse modèle de 29.E4 (l. 141 : « ShareAlike impose de partager les dérivés sous la même licence ») et ce que montre le glossaire (l. 896). Les deux réponses se contredisent.
**Correction :** « parmi les licences courantes, CC0, CC BY, MIT et Apache 2.0 permettent un produit (en citant pour CC BY) ; CC BY-SA et ODbL aussi, à condition de partager les dérivés sous la même licence ; NC, ND pour une modification, ou « recherche seulement », non ».

### 5. MINEUR · `data/cards/*.md`, sections « Chapitres » / « Utilisation » (renvois faux ou manquants)
**Extrait / Problème** (vérifié par `grep` des loaders dans `tools/chapters/` et les notebooks) :
- `cifar10.md:25` : « 28 (applications créatives) » — le ch. 28 n'utilise pas CIFAR-10 (aucun `load_cifar10`, aucune occurrence de « CIFAR » dans `ch28_*.py` ni la fiche ; il travaille sur les photos de scikit-learn et trois tableaux).
- `fashion_mnist.md:21` : le ch. 24 manque (`build_ch24.py:576-580`, partie A « Améliorer un MLP sur Fashion-MNIST » : 24.11 à 24.18, 6 000 images en mode rapide), et le checkpoint V aussi (`build_cp5.py:61-62` : CP5.6, 3 072 images en float64 ; `figures_cp5.py:67` : MLP 784-32-32-10 de la figure).
- `mnist.md:29` : le ch. 29 manque (🔮 29.8, 29.9, 🔬 29.14) et le checkpoint VI aussi (`build_cp6.py:42` : 16 images de CP6.8 ; `figures_cp6.py:65` : le VAE de la figure, 6 000 images) ; « 19 (…), 17, 20 » : le 17 est rangé après le 19.
- `adult.md:47` : « MP3 (…), B6 » — le ch. 29 manque (📈 29.12, profil d'Adult, 52 doublons, 0,76).
- `penguins.md:88` : le ch. 29 manque (🔁 29.R3, l'OpenML simulé de 29.11, la variante de 29.12, la data card d'exemple de 29.15).
**Correction :** supprimer « 28 (applications créatives) » de `cifar10.md` ; ajouter « 24 (partie A, les MLP du livre traduits : 6 000 images d'entraînement du découpage du ch. 23 en mode rapide, 54 000 sinon, 24.11 à 24.18), checkpoint V (CP5.6 : 3 072 images, graine 56, 2 048 pour entraîner et 1 024 pour valider) » à `fashion_mnist.md` ; « 29 (le petit réseau 784-128-10 de 🔮 29.8 et 29.9, entraîné sur 10 000 images, puis testé sur les chiffres de scikit-learn agrandis, 🔬 29.14), checkpoint VI (CP6.8 : 16 images ; la figure de 📈 CP6.7, un VAE sur 6 000 images) » à `mnist.md`, et remettre 17 avant 18 ; « 29 (📈 29.12 : profil express, 52 doublons, référence de 0,76) » à `adult.md` ; « 29 (🔁 29.R3, OpenML simulé de 📦 29.11, variante de 📈 29.12, data card d'exemple de 🛠️ 29.15) » à `penguins.md`.

### 6. MINEUR · `data/cards/pretrained.md` (titre, tableau) et `data/cards/README.md:19`
**Extrait :** « # Réseaux pré-entraînés du ch. 28 (poids torchvision) » ; README : « Réseaux pré-entraînés (SqueezeNet 1.1, VGG16) ».
**Problème :** d'autres poids pré-entraînés servent dans la partie V sans aucune fiche : `MobileNet_V3_Small_Weights.DEFAULT` (🔮 20.26, `build_ch20.py:2040`), `ResNet18_Weights.DEFAULT` (21.27 `build_ch21.py:1690`, 23.24 `build_ch23.py:1838`, MP5 `build_mp5.py:1161`). Le ch. 29 demande de documenter licence et provenance de tout ce qu'on utilise, et la fiche modèle de 🛠️ 23.28 (ResNet-18 gelé) n'a pas de fiche de poids à citer. La mention de licence d'ImageNet vaut aussi pour eux.
**Correction :** renommer en « Réseaux pré-entraînés (poids torchvision) » et ajouter deux lignes au tableau : `MobileNet_V3_Small_Weights.IMAGENET1K_V1` (2,5 millions de paramètres, ≈ 10 Mo ; ch. 20) et `ResNet18_Weights.IMAGENET1K_V1` (11,7 millions de paramètres, 46,8 Mo, fichier `resnet18-f37072fd.pth` de 46 830 571 octets ; ch. 21, 23, MP5), avec la même ligne de licence ; ajuster le README.

### 7. MINEUR · tailles annoncées : `data/cards/cifar10.md:5`, `data/cards/README.md:11`, `src/wb/datasets.py:434` et `:448` ; `data/cards/pretrained.md:6`, `data/cards/README.md:19`, `src/wb/datasets.py:690`
**Extrait :** « téléchargé (≈ 170 Mo) » ; « ℹ️ Premier chargement de CIFAR-10 (≈ 170 Mo depuis Hugging Face) » ; « `SqueezeNet1_1_Weights.IMAGENET1K_V1` (1,2 million de paramètres, 4,7 Mo …) » ; README : « cache de torch (5 Mo ou 528 Mio) ».
**Problème :** depuis Hugging Face (`uoft-cs/cifar10`, la source essayée en premier), le téléchargement fait 143 646 105 octets ≈ **144 Mo** (card du Hub ; cache local 136,8 Mo) ; 170 Mo est la taille de l'archive de Toronto (et ≈ la taille du cache `cifar10.npz`, 173 Mo). Le fichier de SqueezeNet fait 4 958 839 octets, soit **5,0 Mo** avec la convention du workbook (1 Mo = $10^6$ octets, ch. 29) ou 4,7 **Mio** ; « 4,7 Mo » mélange les deux, alors que la même carte écrit « 528 Mio » pour VGG16 et que le ch. 28 écrit « 553 Mo ».
**Correction :** « téléchargé (≈ 145 Mo depuis Hugging Face, ≈ 170 Mo depuis Toronto), cache `cifar10.npz` de ≈ 173 Mo » ; message : « (≈ 145 Mo depuis Hugging Face) » ; pretrained : « 4,7 Mio (5,0 Mo) » et « 528 Mio (553 Mo) » ; README : « 5 Mo ou 553 Mo ».

### 8. MINEUR · `annexes/glossaire.md:737/874`, `:766/886`, `:708/811` (entrées en double et orthographes concurrentes)
**Extrait :** « maximisation d'activation | … | 21 » et « maximisation d'activation | … | 28 » ; « décodage glouton | … | 22 » et « décodage glouton | … | 28 » ; « suréchantillonnage | upsampling | … | 20 » et « sur-échantillonnage | upsampling | … | 25 ».
**Problème :** deux entrées pour un même terme, avec des définitions un peu différentes ; et deux orthographes du même terme, chacune reprise par des chapitres différents (« suréchantillonnage » au ch. 21, « sur-échantillonnage » aux ch. 25 et 27), ce que le §5 veut éviter (un terme retenu, partout).
**Correction :** garder la première entrée de chaque terme, en y ajoutant le renvoi (« … | 21, 28 » ; « … | 22, 28 ») et, pour la maximisation d'activation, la précision du ch. 28 (« un filtre, une couche ou une classe ») ; retenir « suréchantillonnage » (forme du dictionnaire, déjà au ch. 20), supprimer l'entrée l. 811 et corriger les fiches des ch. 25 et 27 (et ajouter la forme écartée à `tests/infra/test_terminology.py`).

### 9. MINEUR · `annexes/glossaire.md` (termes des parties V et VI absents)
**Problème :** des notions mises en gras ou employées comme termes dans les fiches n'ont pas d'entrée : **gradient de politique** (*policy gradient*, ch. 26, l. 291 de la fiche ; PPO cité l. 299 et 303) ; **WGAN-GP**, **pénalité de gradient** (*gradient penalty*, ch. 27, l. 235 et 253) ; **règlement européen sur l'IA** (*AI Act*), introduit au ch. 20 (fiche l. 330) puis écrit tantôt « AI Act » (ch. 20 exercices, ch. 21 solutions, ch. 25, ch. 27), tantôt « règlement sur l'IA » (ch. 21, 27, 28, 29) ; **RGPD** (*GDPR*, ch. 21, 23, 27, 29).
**Correction :** ajouter : « gradient de politique | policy gradient | méthodes qui ajustent directement les paramètres de la politique dans le sens qui augmente le retour espéré (REINFORCE, PPO) | 26 » ; « pénalité de gradient (WGAN-GP) | gradient penalty | terme qui pousse la norme du gradient du critique vers 1 entre vraies et fausses données, à la place de l'écrêtage des poids du WGAN (Gulrajani et coll., 2017) | 27 » ; « règlement européen sur l'IA | AI Act (règlement (UE) 2024/1689) | règlement qui classe les systèmes d'IA par niveau de risque et impose des obligations (transparence, documentation, résumé des données d'entraînement des modèles à usage général, art. 53) | 20 » ; « RGPD | GDPR | règlement européen sur la protection des données personnelles (2016/679) | 20 ou le 1ᵉʳ chapitre qui l'emploie ». Puis s'en tenir à « règlement sur l'IA » dans la prose (ou à « AI Act », mais un seul).

### 10. MINEUR · `chapitres/ch29_datasets/01_fiche.md:45, 62, 133`, `02_exercices.md:22, 79`, `05_solutions.md:12, 130`, `04_indices.md:24`, `flashcards.csv:2` (termes retenus par le glossaire)
**Extrait :** « (transfer learning, bonus B1) », « le transfer learning du bonus B1 », « c'est du transfer learning » ; « Pour les modèles, le pendant est la *model card* » ; « (B) un *model card* » ; « Quelle est la différence avec une *model card* ? » ; titre et prose « référence « classe majoritaire » », « une référence simple à battre ».
**Problème :** le glossaire retient « apprentissage par transfert » (l. 782, défini au ch. 23, fiche l. 539), « fiche modèle » (l. 409, ch. 8, reprise au ch. 23 et dans le titre de 23.28) et « baseline » (l. 150). Le ch. 29 emploie les formes écartées (et « un *model card* » au masculin, à côté de « une *model card* »). Les ch. 24 (fiche l. 773) et 28 (l. 263) ont la même dérive pour *transfer learning*, ce qui suggère que le glossaire devrait plutôt garder l'anglais (le bonus B1 s'intitule « Transfer learning… », BIBLE §8).
**Options :** (A) garder le glossaire et corriger le ch. 29 : « l'apprentissage par transfert (bonus B1) », « la fiche modèle (*model card*, Mitchell et coll., 2019) », « (B) une fiche modèle » ; laisser le titre publié de 29.2, mais écrire « baseline » dans la prose ; (B) retenir « transfer learning » (usage professionnel, titre de B1) : changer l'entrée l. 782 et la glose du ch. 23 en « **transfer learning** (*apprentissage par transfert*) », puis ajouter la forme écartée au test de terminologie. **Recommandation :** B pour *transfer learning*, A pour « fiche modèle » et « baseline ».

### 11. MINEUR · `annexes/erreurs_frequentes.md` (entrées en double, ch. 20 à 28)
**Extrait :** l. 240 et l. 266 : « `ValueError: padding='same' is not supported for strided convolutions` » (deux fois) ; l. 238 et l. 265 : la même erreur NHWC donnée à `nn.Conv2d` ; l. 252 et l. 304 : « un générateur de texte qui répète la même phrase » / « un texte généré qui répète la même phrase » ; l. 234 et l. 257 : softmax avant `CrossEntropyLoss` ; l. 231 et l. 258 : `expected scalar type Long but found Float`.
**Problème :** la même ligne, avec la même cause et la même solution, revient à chaque chapitre qui la rencontre : la table grossit sans rien apprendre de plus et l'apprenant qui cherche un message trouve deux réponses.
**Correction :** fusionner chaque paire en une ligne et y lister les chapitres (« (fiches 21, §21.3.5, et 24, §24.4.3) », « (fiches 22, §22.9, et 28, §28.5) », etc.).

### 12. MINEUR · `annexes/ressources.md:37-38` (parties V et VI)
**Extrait :** « | V · Architectures | | » ; « | VI · Génératif et RL | | »
**Problème :** l'en-tête promet « les meilleures [ressources des fiches] sont reprises ici » et « complété à chaque chapitre » ; les deux parties auditées n'ont aucune ligne (les parties III et IV non plus), alors que leurs fiches citent des références vérifiées.
**Correction (proposition, liens déjà cités et vérifiés dans les fiches) :** V : CS231n (cours de Stanford sur les CNN) ; K. He et coll., « Deep Residual Learning » ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) ; documentation de `torchvision.models` ; C. Olah, « Understanding LSTM Networks » ; VI : D. Kingma et M. Welling, « Auto-Encoding Variational Bayes » ([arXiv:1312.6114](https://arxiv.org/abs/1312.6114)) ; R. Sutton et A. Barto, *Reinforcement Learning: An Introduction* (2ᵉ éd., en ligne) ; I. Goodfellow et coll., « Generative Adversarial Nets » ([arXiv:1406.2661](https://arxiv.org/abs/1406.2661)) ; T. Gebru et coll., « Datasheets for Datasets » ([arXiv:1803.09010](https://arxiv.org/abs/1803.09010)). Les liens et auteurs existants (Raschka, Kapoor et Narayanan, dépôts `blueberrymusic` et `bssrdf`, *Deep Learning: A Visual Approach*, 2021) sont justes.

### 13. MINEUR · `annexes/metiers.md:3, 24-31, 45` (pas enrichi depuis la partie IV)
**Extrait :** « Il sera enrichi à chaque partie (liens vers les exercices 💼 et 🛠️ et vers les mini-projets de portfolio). » ; « | Préparation des données, fuites de données | 12, 15 | … » ; aucune ligne pour l'apprentissage par renforcement ni pour le travail sur les datasets.
**Problème :** la partie VI ajoute deux compétences d'emploi absentes de la table : le RL (ch. 26, B8) et le choix, l'audit et la documentation d'un dataset (ch. 29 : licences, fuites, data card, téléchargement vérifié) ; les mini-projets MP5 (`cifarkit`) et MP6 (`genbench`) ne sont reliés à aucun métier ; la ligne des fuites oublie les ch. 8 et 29.
**Correction :** ligne « Préparation des données, fuites de données | 8, 12, 15, 29 » ; ajouter « Choisir, auditer et documenter un dataset (licences, data card, téléchargement reproductible) | 29, projet final | ●● | ●●● | ●●● | ●● » et « Apprentissage par renforcement | 26, B8 | | ● | ●● | ●● » ; sous « Construire ton portfolio », citer MP5 (classifieur CIFAR-10 empaqueté : ML engineer) et MP6 (banc d'essai génératif à protocole écrit d'avance : data scientist, ML engineer).

### 14. MINEUR · `chapitres/ch29_datasets/flashcards.csv:5, 12, 13, 14`
**Extrait :** « `df.isna()` ne voit que NaN et None » ; « `load_dataset("nom", split="test", revision=sha)` » ; « (fichier lu par morceaux, `hashlib.sha256`) » ; « télécharger dans un fichier `.part` ».
**Problème :** des backticks Markdown dans un deck Anki (BIBLE §12 : HTML autorisé) : Anki les affiche tels quels. C'est le seul chapitre des parties V et VI à le faire ; les ch. 20, 21, 25 et 26 utilisent `<code>`.
**Correction :** remplacer chaque paire de backticks par `<code>…</code>` (et vérifier avec `tools/export_flashcards.py --check`).

### 15. MINEUR · `suivi/PROGRESS.md:1101` (renvoi vers le ch. 29)
**Extrait :** « | ch. 29 | `wb.datasets.make_trap_dataset` (doublons entre train et test, fuite) | exercice 29.13 | »
**Problème :** le dataset piégé sert au 🏆 29.16, pas au 🐛 29.13 (qui fabrique ses quasi-doublons dans le notebook, `make_near_duplicates_13`) ; et la ligne n'est pas marquée faite, alors que la fonction existe depuis la session 60 (BIBLE §22).
**Correction :** « | ch. 29 | `wb.datasets.make_trap_dataset` et `trap_dataset_answers` (huit défauts plantés) | 🏆 29.16 ; ✅ fait en session 60 | ».

### 16. MINEUR · `chapitres/ch29_datasets/01_fiche.md:92` (encadré 🕰️ « charger des données en PyTorch »)
**Extrait :** « rend des objets `Dataset` à passer à un `DataLoader` (ch. 23) »
**Problème :** `Dataset` et `DataLoader` sont introduits au ch. 20 (20.19, glossaire l. 713, fiche du ch. 20) ; le ch. 23 les réutilise.
**Correction :** « (ch. 20 et 23) ».

### 17. SUGGESTION · `chapitres/ch29_datasets/05_solutions.md:134`, `04_indices.md` (29.E3, indice 2), `flashcards.csv:7`
**Extrait :** « Je cherche ensuite trois familles de fuites. Les mêmes exemples des deux côtés […]. Une variable qui contient la réponse […]. Une étape ajustée sur tout le dataset […] »
**Problème :** le ch. 8 (fiche, encadré 🕰️ « la taxonomie des fuites », Kapoor et Narayanan) a déjà donné « trois familles » : séparation imparfaite entre entraînement et test (prétraitement sur tout, doublons, données dépendantes), features illégitimes, test qui ne vient pas de la distribution visée. Le ch. 29 en donne trois autres sous le même nom ; ce n'est pas faux, mais deux classements différents appelés pareil gênent la révision (et le décalage de distribution, troisième famille du ch. 8, est justement le sujet du ch. 29).
**Correction :** ajouter une phrase à la réponse modèle et à la flashcard : « (c'est la taxonomie du ch. 8 regroupée autrement : mes deux premières familles en sont la séparation imparfaite et les features illégitimes ; la troisième famille du ch. 8, un test qui ne vient pas de la bonne distribution, est le décalage de distribution du ch. 29) ».

### 18. SUGGESTION · `data/cards/*.md` et `src/wb/datasets.py:430-441` (la leçon du ch. 29 appliquée au workbook)
**Problème :** le ch. 29 apprend à figer une révision et à vérifier une somme SHA-256, et donne `data/cards/` comme modèle (fiche l. 133, README des cards l. 3, énoncé de 29.15) ; or seule `images.md` donne des empreintes, aucune fiche n'a de section « Téléchargement », et `_cifar_from_huggingface` charge `uoft-cs/cifar10` **sans** `revision=` (29.10 fixe pourtant celle de Fashion-MNIST, `531be5e2…`). Les fiches ne suivent pas non plus les six sections que vérifie `check_card` (Provenance, Contenu, Licence, Usage prévu, Limites et biais, Téléchargement) : un apprenant qui recopie `cifar10.md`, comme l'y invite l'énoncé, échoue à la vérification (le squelette `CARD_15` du notebook limite le dégât).
**Correction :** dans `_cifar_from_huggingface`, `load_dataset("uoft-cs/cifar10", revision="0b2714987fa478483af9968de7c934580d0bb9a2", …)` (la révision du cache actuel) ; dans `mnist.md`, `penguins.md`, `holmes.md`, `verne.md`, `sunspots.md`, `adult.md`, une ligne « **Empreinte** | SHA-256 du fichier versionné » ; dans l'énoncé de 29.15, « Modèle : les six sections ci-dessous ; les fiches de `data/cards/` montrent le niveau de détail attendu ».

### 19. SUGGESTION · `data/cards/mnist.md:10`
**Extrait :** « **Licence** | Creative Commons Attribution-Share Alike 3.0 ; copyright Yann LeCun et Corinna Cortes »
**Problème :** juste d'après la page d'origine, mais la card `ylecun/mnist` du Hub (que la fiche du ch. 29 cite comme source, l. 72) annonce « MIT ». Un apprenant qui vérifie, comme le ch. 29 le lui demande, tombe sur deux licences.
**Correction :** ajouter : « (la card du Hugging Face Hub, `ylecun/mnist`, affiche MIT : en cas de désaccord, c'est la source d'origine qui fait foi ; un bon exemple de vérification de licence pour 📦 29.11) ».

## Bilan

**Solide.** Toutes les réponses vérifiables du ch. 29 (quiz, rappels, ✏️, 🧮 et les dix exercices du notebook) recalculées sans écart, y compris les variantes des solutions (29.8, 29.13, 29.14) et les deux graines du dataset piégé. Les corrections que la fiche apporte au livre (Digits 1 797, Boston 506, LFW 62 × 47 en gris, Olivetti 400, RCV1 804 414, tailles de Keras, Reuters 11 228, COCO, CelebA, Speech Commands) sont exactes, et ses affirmations datées (Boston retiré en 1.2, retraits de MS-Celeb-1M, de Tiny Images et de LAION-5B, Re-LAION-5B, Papers with Code, `kagglehub`, art. 53 de l'AI Act, Northcutt 3,3 %) se vérifient sur le web. Les encadrés « Le livre : » sont fidèles au texte et aux pages (p. 1648-1657), sans copie. Le formulaire des ch. 21 à 29 est juste, ses notations suivent celles des fiches, et les licences des data cards (Fashion-MNIST MIT, CIFAR-10 sans licence, SILSO CC BY-NC 4.0 avec son DOI, Gutenberg, photos CC BY 2.0, tableaux du domaine public) sont correctes.

**Les trois risques principaux.**
1. Le défaut « labels mal écrits » du dataset piégé contient des labels **faux**, que le nettoyage de référence entérine, et qui contredisent l'affirmation de la solution sur la colonne fuyante (constat 1).
2. Les renvois croisés des annexes et des data cards ont dérivé : chapitres manquants ou faux dans les cards, poids pré-entraînés de la partie V sans fiche, ressources et métiers jamais complétés pour V et VI (constats 5, 6, 12, 13).
3. La terminologie des annexes n'est plus tenue d'un chapitre à l'autre : doublons et orthographes concurrentes dans le glossaire, termes retenus non suivis au ch. 29 (*transfer learning*, *model card*, baseline), entrées manquantes (constats 8, 9, 10, 11).

| Sévérité | Nombre |
|---|---|
| MAJEUR | 0 |
| MINEUR | 16 |
| SUGGESTION | 3 |
| **Total** | **19** |
