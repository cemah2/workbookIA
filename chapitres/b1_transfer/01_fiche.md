# B1 · Transfer learning et modèles pré-entraînés — fiche de cours

> Ce chapitre bonus n'a pas d'équivalent dans le livre de Glassner : **cette fiche est le cours**. Le livre ne consacre au transfer learning que deux pages (§23.9.4), alors que c'est aujourd'hui la façon normale de commencer presque tout projet d'images : on part d'un réseau que d'autres ont entraîné pendant des jours sur des millions d'images, et on l'adapte à ses données. Tu apprends à choisir et charger un tel modèle, à t'en servir sans le modifier (extraction de features), à le modifier avec prudence (fine-tuning), à mesurer quand cela aide et quand cela nuit, puis à classer des images sans exemple d'entraînement, ou presque, avec des modèles de fondation (CLIP, DINOv2).

| | |
|---|---|
| **Livre** | — (chapitre propre au workbook ; le livre évoque l'idée au §23.9.4) |
| **Sections** | 201.1 à 201.9 (numérotation interne du workbook : 201 = bonus B1) |
| **Temps total estimé** | ≈ 19 h : lecture de la fiche ≈ 1,5 h, exercices ≈ 16 h, 24 flashcards ≈ 0,8 h |
| **Prérequis** | 0B (similarité cosinus) · ch. 9 (régularisation L2) · ch. 12 (PCA, UMAP) · ch. 13 (k plus proches voisins, régression logistique) · ch. 15 (`Pipeline`, validation croisée) · ch. 16 (paramètres d'une couche) · ch. 17 (softmax) · ch. 19 (AdamW, warmup et cosinus) · ch. 20 (batchnorm, `train()` et `eval()`) · ch. 21 (CNN, ResNet, cartes d'activation) · ch. 23 (`fit()`, gel des poids, nouvelle tête, `weights_only=True`) · ch. 24 (augmentation, `transforms.v2`) · ch. 29 (licences, décalage de distribution, data card) |
| **Fichiers du chapitre** | `02_exercices.md` (quiz, rappels, papier, réflexion, entretien) · `03_notebook.ipynb` · `04_indices.md` · `05_solutions.md` et `05_solutions.ipynb` · `06_mes_reponses.md` · `flashcards.csv` |
| **mylearn** | aucun module : le cœur du chapitre est l'usage de PyTorch, de torchvision et de Hugging Face ; un module NumPy n'apprendrait rien de plus que les exercices papier (✏️ B1.1 à B1.5) |

## Comment utiliser ce chapitre

La fiche et les exercices papier posent les idées et les calculs (formes des tenseurs, nombres de paramètres, learning rates par groupe, softmax de CLIP) ; le notebook les met en œuvre sur CIFAR-10, avec trois modèles pré-entraînés qui tournent sur un CPU. Les codes de la fiche ont été exécutés avec les versions du workbook (PyTorch 2.11, torchvision 0.26, transformers 5.21, scikit-learn 1.6) ; sauf pour CLIP, ils utilisent **d'autres modèles** que le notebook (ResNet-50, EfficientNet-B0, DINOv2-base) : les tiens, tu les découvres toi-même.

**Ordre conseillé.**
1. Lis 201.1 et 201.2. Fais les quiz Q1 à Q4 et les rappels R1 et R3.
2. Lis 201.3 à 201.5. Fais les quiz Q5 à Q7, le rappel R2, les exercices ✏️ B1.1 à B1.4 et la dérivation ∂ B1.6.
3. Lis 201.6 et 201.7. Fais les quiz Q8 et Q9, le cas ⚖️ B1.9 et la lecture 📄 B1.10.
4. Lis 201.8 et 201.9. Fais les quiz Q10 et Q11, le calcul ✏️ B1.5, l'estimation 🧮 B1.7 et l'explication 🗣️ B1.8. Vérifie tes réponses courtes dans la partie 0 du notebook.
5. Fais le notebook dans l'ordre des ID, en quatre phases : **charger et sonder** (B1.11 à B1.16), **fine-tuner** (B1.17 à B1.19), **quand le transfert aide ou échoue** (B1.20 à B1.23), **modèles de fondation, coût et livraison** (B1.24 à B1.28). Termine par les cinq questions d'entretien.

| Section | Quiz 🧠 et rappels 🔁 | Papier ✏️ ∂ 🧮 et réflexion | Notebook | Entretien 💼 |
|---|---|---|---|---|
| 201.1 Pourquoi réutiliser | Q1, Q2, R1, R3 | B1.7, B1.8, B1.10 | B1.16 | |
| 201.2 Trouver et charger un modèle | Q3, Q4 | B1.9 | B1.11, B1.27 | E5 |
| 201.3 Extraction de features | Q5, R2 | B1.2, B1.4 | B1.13, B1.15 | E1 |
| 201.4 Fine-tuning | Q5, Q6, Q7, R2 | B1.3, B1.4, B1.6 | B1.17, B1.18, B1.19, B1.28 | E1, E3 |
| 201.5 Adapter l'entrée et la sortie | Q4 | B1.1 | B1.12, B1.14, B1.18 | |
| 201.6 Peu de données | Q8 | | B1.20, B1.21, B1.25, B1.28 | E1, E2 |
| 201.7 Quand le transfert échoue | Q8, Q9, R1 | B1.6, B1.9, B1.10 | B1.16, B1.22, B1.23 | E2 |
| 201.8 Modèles de fondation | Q10, Q11 | B1.5 | B1.15, B1.24, B1.25 | E4 |
| 201.9 Coût d'usage | Q11 | B1.7 | B1.14, B1.26, B1.27 | E5 |

## Objectifs d'apprentissage

À la fin de ce chapitre, tu sais :
1. **Expliquer** pourquoi les features d'un réseau pré-entraîné se transfèrent, et **choisir** entre extraction de features, fine-tuning partiel et fine-tuning complet selon le volume de données et la distance de domaine.
2. **Charger** un modèle pré-entraîné avec son prétraitement (torchvision, Hugging Face Hub) et **vérifier** sa fiche modèle et sa licence.
3. **Entraîner** un linear probe et un k plus proches voisins sur des embeddings figés.
4. **Fine-tuner** un réseau en PyTorch : dégel progressif, learning rates par groupe, warmup, batchnorm maîtrisée.
5. **Mesurer** le gain du pré-entraînement par des courbes d'apprentissage, et **diagnostiquer** transfert négatif et oubli catastrophique.
6. **Classer** en zéro-shot et en few-shot avec des modèles de fondation (CLIP, DINOv2).
7. **Comparer** des backbones selon l'accuracy, la taille et la latence sur CPU.

## L'essentiel en 10 lignes

1. Un réseau entraîné sur un grand dataset a appris des **features** utiles bien au-delà de sa tâche : bords et textures dans ses premières couches, parties d'objets plus loin. On les réutilise au lieu de repartir de zéro.
2. On coupe le réseau en deux : le **backbone**, qui transforme une image en un vecteur $\phi(\mathbf{x})$ (l'embedding), et la **tête**, qui transforme ce vecteur en classes. Pour une nouvelle tâche, on remplace la tête.
3. Un modèle pré-entraîné arrive avec **son prétraitement** (taille, recadrage, moyenne et écart-type) : s'en écarter dégrade les résultats sans aucun message d'erreur.
4. **Extraction de features** : backbone gelé, embeddings calculés une fois, puis un classifieur simple (linear probe, k plus proches voisins). Rapide, sûr, souvent le meilleur choix avec peu de données.
5. **Fine-tuning** : on continue d'entraîner tout ou partie du backbone, avec un learning rate 10 à 100 fois plus petit que de zéro, plus petit encore pour les premières couches, et un warmup.
6. Sur de petits mini-batches, on garde les **batchnorms en mode `eval()`** : les statistiques d'ImageNet valent mieux que celles de 8 images.
7. Le pré-entraînement rapporte d'autant plus que les données sont **rares** et le domaine **proche** ; sur un domaine lointain, il peut ne rien rapporter, voire nuire (transfert négatif).
8. Un fine-tuning trop énergique **oublie** ce que le modèle savait (oubli catastrophique) ; un petit learning rate, peu d'epochs ou une pénalité L2-SP le limitent.
9. Les **modèles de fondation** (DINOv2, CLIP) donnent des embeddings génériques ; CLIP classe même **sans exemple** (zéro-shot), en comparant une image à des phrases comme « a photo of a dog ».
10. En production, on choisit un backbone sur sa licence, sa latence et son accuracy sur **ta** validation ; on publie le modèle dérivé en safetensors, avec une fiche modèle qui cite le modèle de base.

> 📐 **Conventions de la fiche** — $\mathbf{x}$ est une image, $\phi(\mathbf{x}) \in \mathbb{R}^d$ son embedding (la sortie du backbone), de dimension $d$ ; $K$ est le nombre de classes de la nouvelle tâche ; $\theta$ désigne les paramètres d'un réseau, $\theta_0$ leurs valeurs pré-entraînées. Les groupes de paramètres d'un fine-tuning sont numérotés $\ell = 1, \dots, L$ de l'entrée vers la tête ($\ell = L$ est la tête). Un mini-batch d'images a la forme $(N, C, H, W)$, comme aux ch. 21 et 23. $s$ est l'échelle des logits de CLIP (`logit_scale`). Les blocs portent les noms de torchvision : le *stem* (les couches d'entrée : `conv1`, `bn1`, ReLU, `maxpool`), `layer1` à `layer4`, `fc` (la tête). **Écarts au §6** : $d$ est la dimension de l'embedding (quand les embeddings servent de features à un classifieur, il y en a $p = d$) ; $L$ compte des groupes, pas la loss (la fiche n'écrit aucune loss) ; $\theta$ s'écrit en maigre, comme dans les articles, bien que ce soit un vecteur.

---

## 201.1 · Pourquoi réutiliser : coût des données et du calcul, hiérarchie des features ⏩

### Des données et du calcul que tu n'as pas

Un CNN moderne a des millions de paramètres. Pour les régler sans overfitting, il lui faut des centaines de milliers d'images étiquetées et des jours de calcul sur des GPU. Le dataset de référence des années 2010, **ImageNet-1k** (le concours ILSVRC 2012), compte 1 281 167 images d'entraînement en 1 000 classes, étiquetées à la main. Les modèles de fondation (201.8) vont bien plus loin : CLIP a été entraîné sur 400 millions de paires image-texte, et sa plus grande version a demandé 12 jours sur 256 GPU V100 (Radford et coll., 2021).

Ton projet, lui, a souvent quelques centaines d'images et un ordinateur portable. Entraîner un ResNet de zéro sur 500 images, c'est la recette de l'overfitting (ch. 9). Le **transfer learning** (*apprentissage par transfert*, ch. 23) contourne le problème : on part d'un réseau entraîné sur une grande tâche **source** (ImageNet) et on le réutilise pour une tâche **cible** (la tienne). Le calcul coûteux a été fait une fois, par d'autres (🧮 B1.7 te fait chiffrer l'écart).

### Backbone et tête

On découpe un classifieur pré-entraîné en deux :
- le **backbone** (*tronc*) : toutes les couches jusqu'à l'avant-dernière, qui transforment une image en un vecteur de features $\phi(\mathbf{x})$, l'**embedding** de l'image ;
- la **tête** (ch. 23) : la ou les dernières couches, qui transforment cet embedding en scores pour les classes de la tâche source.

Pour un ResNet de torchvision, le backbone va de `conv1` au pooling moyen global (`avgpool`, ch. 21), et la tête est une couche dense, `fc`. Le backbone se réutilise ; la tête, qui parle de tanches, de requins et de 118 races de chiens, se **remplace** par une tête neuve à $K$ sorties.

### Hiérarchie des features : du général au spécifique

Pourquoi un backbone entraîné sur des chiens et des voitures servirait-il à reconnaître des fleurs ou des pièces défectueuses ? Parce que ce qu'il a appris est en grande partie **général**. Tu l'as vu au ch. 21 en affichant les filtres et les cartes d'activation d'un ResNet-18 (📦 21.27) :
- les **premières couches** détectent des bords orientés, des taches de couleur, des textures fines ; presque tous les CNN entraînés sur des images naturelles ont des filtres semblables ;
- les **couches intermédiaires** combinent ces bords en motifs, puis en parties d'objets (un œil, une roue, un pétale) ;
- les **dernières couches** deviennent **spécifiques** : leurs features séparent au mieux les 1 000 classes d'ImageNet.

D'où la règle : plus une couche est profonde, plus elle est liée à la tâche source, et moins elle se transfère telle quelle. J. Yosinski, J. Clune, Y. Bengio et H. Lipson l'ont mesuré couche par couche en 2014, en coupant ImageNet en deux moitiés de classes et en transplantant les premières couches d'un réseau entraîné sur l'une dans un réseau entraîné sur l'autre (📄 B1.10). Le 🔬 B1.16 reproduit l'idée en petit, avec un linear probe sur la sortie de chaque bloc.

### Trois façons de réutiliser

Les trois stratégies du chapitre diffèrent par **ce qu'on laisse bouger**.

![Un backbone (stem et quatre blocs) suivi d'une nouvelle tête : en extraction de features, tout le backbone est gelé et seule la tête s'entraîne ; en fine-tuning partiel, le dernier bloc s'entraîne aussi, avec un learning rate plus petit ; en fine-tuning complet, tout s'entraîne, avec des learning rates plus petits vers l'entrée](figures/strategies.png)

- **Extraction de features** (201.3) : tout le backbone est gelé ; on calcule les embeddings une fois, puis on entraîne un classifieur simple dessus.
- **Fine-tuning partiel** (201.4) : on dégèle les derniers blocs, les plus spécifiques, et on les entraîne avec la tête.
- **Fine-tuning complet** (201.4) : tout le réseau s'entraîne, en partant des poids pré-entraînés au lieu de poids aléatoires.

Le choix dépend surtout de deux choses : combien d'images étiquetées tu as, et à quel point elles ressemblent aux données du pré-entraînement (201.6, 201.7).

> 💼 **En entreprise** — Pour une tâche d'images, la première expérience est presque toujours la même : un backbone pré-entraîné gelé, un linear probe (201.3), une mesure sur la validation. Elle prend une heure, donne une baseline sérieuse, et dit si le problème est facile (le linear probe suffit), difficile (il faut fine-tuner) ou mal posé (rien ne marche : regarder les données, ch. 29). Entraîner de zéro devient l'exception, à justifier.

---

## 201.2 · Trouver et charger un modèle pré-entraîné : torchvision, Hugging Face Hub, fiches modèles, licences ⏩

### Trois sources

- **`torchvision.models`** : 80 classifieurs d'images (ResNet, EfficientNet, MobileNet, ConvNeXt, ViT…), presque tous entraînés sur ImageNet-1k ; `get_model("resnet50", weights="DEFAULT")`.
- **Le Hugging Face Hub** : plus d'un million de modèles, dont les modèles de fondation (DINOv2, CLIP, SigLIP 2), chacun avec sa fiche modèle et sa licence ; `AutoModel.from_pretrained("facebook/dinov2-base")`.
- **`timm`** (PyTorch Image Models) : des centaines d'architectures de vision et leurs poids ; préinstallé sur Colab, absent de l'environnement du workbook.

Le workbook utilise les deux premières : torchvision pour les CNN classiques, le Hub pour les modèles de fondation.

### torchvision : des poids qui portent leurs métadonnées

Chaque architecture de torchvision a une **énumération de poids** (`ResNet50_Weights.IMAGENET1K_V1`, `…_V2`), dont `DEFAULT` désigne les meilleurs. Chaque valeur porte l'adresse du fichier, le prétraitement (`weights.transforms()`) et des métadonnées (`weights.meta`).

```python
from torchvision.models import get_model, get_model_weights

weights = get_model_weights("efficientnet_b0").DEFAULT    # EfficientNet_B0_Weights.IMAGENET1K_V1
model = get_model("efficientnet_b0", weights=weights).eval()   # downloaded once into ~/.cache/torch
preprocess = weights.transforms()                          # the preprocessing that goes with these weights
meta = weights.meta
print(meta["num_params"], meta["_metrics"]["ImageNet-1K"]["acc@1"], meta["_ops"], meta["_file_size"])
# 5288548 77.692 0.386 20.451
print(len(meta["categories"]), meta["categories"][:2])     # 1000 ['tench', 'goldfish']
```

`num_params` compte tous les paramètres, tête comprise ; `_metrics` donne l'accuracy **top-1** (la classe la plus probable est la bonne) et **top-5** (la bonne est parmi les cinq plus probables) sur la validation d'ImageNet ; `_ops`, le coût d'une passe avant en GFLOPS (milliards d'opérations par image) ; `_file_size`, la taille du fichier en Mio (mébioctets : 1 Mio = $2^{20}$ octets ≈ 1,05 Mo) ; `categories`, les noms des 1 000 classes dans l'ordre des sorties ; `min_size`, la plus petite image acceptée.

> 🧮 **Rappel outil — `weights.transforms()`** — Ce module reçoit une image PIL ou un tenseur `uint8` et rend un tenseur `float32` en quatre étapes : (1) **redimensionner** pour que le petit côté mesure `resize_size` pixels, (2) **recadrer au centre** un carré de `crop_size`, (3) ramener les valeurs dans $[0, 1]$, (4) **normaliser** chaque canal par la moyenne et l'écart-type des pixels d'ImageNet, `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`. Les tailles et l'interpolation changent d'un jeu de poids à l'autre : `print(preprocess)` affiche `resize_size=[256]`, `crop_size=[224]` et `BICUBIC` pour EfficientNet-B0, mais `resize_size=[232]` et `BILINEAR` pour `ResNet50_Weights.IMAGENET1K_V2`. On ne les devine pas : on les lit (🧠 Q4, 📦 B1.11).

### Hugging Face Hub : `AutoModel` et `AutoImageProcessor`

Sur le Hub, chaque modèle est un dépôt git (`facebook/dinov2-base`) qui contient sa configuration (`config.json`), ses poids (`model.safetensors`), son prétraitement (`preprocessor_config.json`) et sa fiche modèle (`README.md`) : `transformers` reconstruit le modèle et son prétraitement à partir du seul nom du dépôt.

> 🧮 **Rappel outil — l'API transformers pour la vision** — `AutoImageProcessor.from_pretrained(nom)` rend le **processeur d'images** du modèle : appelé sur une image PIL ou une liste d'images, il redimensionne, recadre, normalise et rend un dictionnaire dont `pixel_values` est un tenseur $(N, 3, H, W)$. `AutoModel.from_pretrained(nom)` rend le **backbone**, sans tête de classification ; sa sortie a des champs nommés (`last_hidden_state`, `pooler_output`). Pour CLIP, `CLIPProcessor` et `CLIPModel` traitent images **et** textes (201.8). Deux arguments utiles de `from_pretrained` : `revision=` fixe une version du dépôt (le hash d'un commit, comme pour les datasets au ch. 29) et `dtype=` la précision des poids. Le premier appel télécharge dans `~/.cache/huggingface`, les suivants relisent le disque.

```python
import torch
from transformers import AutoImageProcessor, AutoModel

name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(name)
model = AutoModel.from_pretrained(name).eval()             # 86 580 480 parameters, no classification head
inputs = processor(images=[img_a, img_b], return_tensors="pt")   # two PIL images, any size
print(inputs["pixel_values"].shape)                        # torch.Size([2, 3, 224, 224])
with torch.inference_mode():
    out = model(**inputs)
print(out.last_hidden_state.shape)                         # torch.Size([2, 257, 768]): CLS token + 16 x 16 patches
emb = out.pooler_output                                    # torch.Size([2, 768]): one vector per image
```

Ici $d = 768$. DINOv2-base est un **Vision Transformer** (ViT, bonus B3) : il découpe l'image en $16 \times 16 = 256$ carrés de 14 pixels, traite chaque carré comme un token, ajoute un token spécial qui résume l'image (le token `CLS`), et rend 768 nombres par token ; `pooler_output` est le token `CLS` après la dernière normalisation. Il suffit ici de savoir où lire l'embedding.

> 🕰️ **Mise à jour (2026) — les sources de modèles pré-entraînés** — **Le livre :** charge VGG16 par `keras.applications` (ch. 23 et 28), avec un prétraitement à appliquer soi-même. · **Aujourd'hui :** torchvision attache à chaque jeu de poids son prétraitement et ses métadonnées (`pretrained=True` est déprécié depuis la version 0.13 au profit de `weights=`) ; le Hugging Face Hub est devenu la grande bibliothèque de modèles ; `timm` reste la référence des backbones de vision. · **Faut-il quand même l'apprendre ?** Oui : charger un modèle **avec son prétraitement** est le premier geste de tout projet de transfert. · *Sources :* [documentation des modèles de torchvision 0.26](https://docs.pytorch.org/vision/0.26/models.html) (« Using the correct preprocessing method is critical ») ; [processeurs d'images de transformers 5.21](https://huggingface.co/docs/transformers/v5.21.0/en/main_classes/image_processor).

> 🕰️ **Mise à jour (2026) — transformers 5** — **Le livre :** n'en parle pas (la bibliothèque est née fin 2018, sous un autre nom). · **Aujourd'hui :** la version 5 ne garde que **PyTorch** (TensorFlow et JAX ont été retirés) ; les processeurs d'images s'appuient par défaut sur **torchvision** (`use_fast` cède la place à `backend`) ; `from_pretrained` prend `dtype=` (`torch_dtype=` affiche « `torch_dtype` is deprecated! Use `dtype` instead! ») ; `get_image_features` et `get_text_features` de CLIP rendent un objet dont l'embedding est dans `.pooler_output`, et non plus un tenseur. · **Faut-il quand même l'apprendre ?** Oui, avec la bonne version de la documentation : beaucoup de tutoriels datent de la version 4 et leur code casse. · *Sources :* [guide de migration vers la v5](https://raw.githubusercontent.com/huggingface/transformers/v5.20.0/MIGRATION_GUIDE_V5.md) ; [annonce de transformers v5](https://huggingface.co/blog/transformers-v5).

### Lire la fiche modèle avant de télécharger

Une **fiche modèle** (*fiche modèle* ; ch. 23 ; Mitchell et coll., 2019) est au modèle ce que la data card est au dataset (ch. 29). On y cherche six réponses : la **licence** (usage commercial, citation, conditions d'accès) ; les **données d'entraînement** (provenance, biais) ; l'**usage prévu** et les usages exclus (*out of scope*) ; le **prétraitement** ; les **performances**, et sur quels benchmarks ; les **limites** connues. Pour les modèles de ce chapitre, vérifié sur leurs pages le 31 octobre 2026 :

| Modèle | Licence des poids | À savoir |
|---|---|---|
| poids ImageNet de torchvision | code sous BSD 3-Clause ; pas de licence propre aux poids | la documentation te laisse la responsabilité de vérifier ton droit de t'en servir ; ImageNet a ses propres conditions |
| `facebook/dinov2-*` | Apache 2.0 | publié sous CC BY-NC en avril 2023, passé en Apache 2.0 le 31 août 2023 |
| `openai/clip-vit-base-patch32` | aucun champ de licence sur le Hub | la fiche modèle exclut **tout** déploiement, commercial ou non, et déconseille même un usage interne sans tests poussés dans le domaine visé |
| `facebook/dinov3-*` | *DINOv3 License*, propre à Meta | accès soumis à l'acceptation de la licence ; mention « Built with DINOv3 » en cas de redistribution ; usages militaires interdits |
| `google/siglip2-*` | Apache 2.0 | |

> 🕰️ **Mise à jour (2026) — les licences des poids** — **Le livre :** ne traite pas des licences des poids. · **Aujourd'hui :** les licences sont permissives (Apache 2.0, MIT, BSD), non commerciales (CC BY-NC), ou rédigées par chaque entreprise (licences « open weights »), avec des usages interdits ou des mentions obligatoires (DINOv3, Llama). Les poids ImageNet posent une question à part : les conditions d'ImageNet réservent la base « à la recherche non commerciale et à l'enseignement », et certaines images sont protégées par le droit d'auteur ; le statut des poids appris dessus n'est pas tranché, et beaucoup d'entreprises les utilisent après un avis juridique. · **Faut-il quand même l'apprendre ?** Oui : c'est un réflexe professionnel et une question d'entretien (💼 E5, ⚖️ B1.9). · *Sources :* [conditions d'ImageNet, page `imagenet-1k` du Hub](https://huggingface.co/datasets/imagenet-1k/blob/main/README.md) (« non-commercial research and educational purposes ») ; [documentation de torchvision](https://docs.pytorch.org/vision/0.26/models.html) (« It is your responsibility to determine whether you have permission ») ; [licence de DINOv3](https://ai.meta.com/resources/models-and-libraries/dinov3-license) ; [passage de DINOv2 en Apache 2.0](https://www.maginative.com/article/embracing-open-source-and-equity-meta-announces-commercial-relicensing-of-dinov2-and-introduction-of-facet).

> ⚠️ **Piège classique — public n'est pas libre** — Un modèle téléchargeable sans compte n'est pas pour autant utilisable dans un produit : CLIP n'affiche aucune licence et exclut tout déploiement, une licence CC BY-NC interdit l'usage commercial, un modèle à accès contrôlé (*gated*) t'a fait accepter des conditions. Lis la page avant la première ligne de code, comme pour un dataset (ch. 29).

---

## 201.3 · Extraction de features : embeddings figés, linear probe, k plus proches voisins ⏩

### Calculer les embeddings une fois

En extraction de features, le backbone ne change jamais : chaque image a **toujours le même embedding**, on parle d'**embedding figé**. Inutile de le recalculer à chaque epoch : on fait passer le dataset **une seule fois** dans le backbone, on range les vecteurs dans une matrice $\mathbf{X}$ de forme $(n, d)$ et on la sauvegarde. N'importe quel classifieur des ch. 13 et 15 s'entraîne ensuite en quelques secondes : c'est un problème tabulaire à $p = d$ features.

```python
import numpy as np
import torch
from torch import nn
from torchvision.models import get_model, get_model_weights

weights = get_model_weights("resnet50").DEFAULT
backbone = get_model("resnet50", weights=weights)
backbone.fc = nn.Identity()               # drop the ImageNet head: the output is now phi(x), of size d = 2048
backbone.eval()                           # batchnorm uses its stored statistics, dropout is off

@torch.inference_mode()                   # no gradient: faster, less memory
def extract(backbone, loader):
    feats, labels = [], []
    for xb, yb in loader:                 # xb already went through weights.transforms()
        feats.append(backbone(xb))
        labels.append(yb)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

X_train, y_train = extract(backbone, train_loader)
np.savez_compressed("emb_resnet50_train.npz", X=X_train, y=y_train)   # computed once, reused everywhere
```

`nn.Identity()` rend son entrée telle quelle : remplacer la tête par elle est la façon la plus simple de récupérer $\phi(\mathbf{x})$. Sans `eval()`, chaque batchnorm normaliserait avec les statistiques du mini-batch, et l'embedding d'une image dépendrait de ses voisines.

### Le linear probe

Un **linear probe** (*sonde linéaire*) est une régression logistique multiclasse entraînée sur les embeddings figés :

$$\hat{\mathbf{y}} = \operatorname{softmax}\big(\mathbf{W}\,\phi(\mathbf{x}) + \mathbf{b}\big), \qquad \mathbf{W} \in \mathbb{R}^{K \times d},\ \mathbf{b} \in \mathbb{R}^{K}.$$

C'est une tête linéaire neuve, entraînée seule, backbone gelé. Son nom vient de son autre usage, **sonder** une couche (G. Alain et Y. Bengio, 2016) : si un classifieur linéaire retrouve les classes à partir de ses features, l'information y est, sous une forme facile à lire. Sa taille est celle d'une couche dense (ch. 16) :

$$\text{paramètres de la tête} = d \times K + K.$$

Sur ResNet-50 ($d = 2048$), une tête à $K = 5$ classes a $2048 \times 5 + 5 = 10\,245$ paramètres ; sur EfficientNet-B0 ($d = 1280$), $6\,405$. À côté des 23,5 millions de paramètres du backbone de ResNet-50, on entraîne moins d'un deux-millième du réseau (0,04 %) : l'overfitting reste modéré même avec quelques centaines d'images. Le plus simple est scikit-learn (ch. 15) :

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
probe.fit(X_train, y_train)
print(probe.score(X_val, y_val))          # accuracy of the linear probe on the validation embeddings
```

La standardisation aide le solveur à converger ; `C` règle la régularisation L2 (ch. 9 et 13) et se choisit par validation croisée. Une tête `nn.Linear(d, K)` entraînée en PyTorch sur les mêmes embeddings donne un résultat très proche : c'est le même modèle.

### Les k plus proches voisins sur des embeddings

Un bon embedding place les images semblables **près** les unes des autres : on peut classer une image par ses k plus proches voisins (ch. 13), sans rien entraîner. La mesure est presque toujours la **similarité cosinus** (0B, 101.3.3), $\cos(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} / (\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert)$. Elle compare des **directions** et ignore les longueurs ; or la norme d'un embedding porte surtout des informations étrangères à la classe (contraste, taille de l'objet, quantité de texture).

Exemple. Quatre images de référence ont pour embeddings (en dimension 3) $\mathbf{a}_0 = (1, 0, 0)$ « chat », $\mathbf{a}_1 = (4, 2, 2)$ « chien », $\mathbf{a}_2 = (2, 1, 0)$ « chat » et $\mathbf{a}_3 = (0, 1, 1)$ « chien ». On classe $\mathbf{q} = (2, 1, 1)$, de norme $\sqrt{6} \approx 2{,}449$.
- Produits scalaires avec $\mathbf{q}$ : 2, 12, 5, 2 ; normes : 1 ; 4,899 ; 2,236 ; 1,414.
- Cosinus : $\approx 0{,}816$ ; exactement 1 ($\mathbf{a}_1 = 2\mathbf{q}$, même direction) ; $\approx 0{,}913$ ; $\approx 0{,}577$.
- Avec $k = 1$ : $\mathbf{a}_1$, « chien ». Avec $k = 3$ : $\mathbf{a}_1$, $\mathbf{a}_2$, $\mathbf{a}_0$, deux « chat » contre un « chien » : « chat ».
- En distance euclidienne, le plus proche serait $\mathbf{a}_2$ (distance 1) et $\mathbf{a}_1$ le plus loin (2,449) : la mesure change la réponse.

Avec scikit-learn : `KNeighborsClassifier(n_neighbors=k, metric="cosine")`, ou bien `Normalizer()` (norme 1) puis la distance euclidienne : pour des vecteurs de norme 1, $\lVert\mathbf{a} - \mathbf{b}\rVert^2 = 2 - 2\cos(\mathbf{a}, \mathbf{b})$, les deux classements sont identiques. Le linear probe est en général un peu meilleur ; le k-NN n'entraîne rien, accepte une nouvelle classe dès qu'on ajoute ses exemples, et montre **pourquoi** il décide (ses voisins). Le 📦 B1.13 compare les deux.

### Regarder et sonder

Une PCA ou un UMAP à deux dimensions des embeddings (ch. 12) montre si les classes forment des groupes, avec les précautions du ch. 12 : UMAP ne conserve pas les distances entre groupes, et une projection en 2D perd presque toute l'information ; une image mélangée ne prouve rien (📈 B1.15).

Rien n'oblige à prendre l'embedding à la fin du backbone. `create_feature_extractor` rend un modèle qui sort les activations de couches choisies :

```python
from torchvision.models.feature_extraction import create_feature_extractor

probe_net = create_feature_extractor(backbone, return_nodes={"layer2": "l2", "layer4": "l4"})
with torch.inference_mode():
    out = probe_net(torch.randn(2, 3, 160, 160))
print({k: tuple(v.shape) for k, v in out.items()})      # {'l2': (2, 512, 20, 20), 'l4': (2, 2048, 5, 5)}
```

On résume chaque carte d'activation par un pooling moyen global (`v.mean(dim=(2, 3))`), puis on entraîne un linear probe par couche : c'est le protocole du 🔬 B1.16.

---

## 201.4 · Fine-tuning : partiel, complet, dégel progressif, learning rates discriminatifs, batchnorm ⏩

### Partiel ou complet

Le **fine-tuning** (ch. 23) continue d'entraîner les poids pré-entraînés sur la nouvelle tâche. Il est **partiel** quand on ne dégèle que les derniers blocs, **complet** quand tout le réseau s'entraîne. Geler, c'est `requires_grad_(False)` ; dégeler, `requires_grad_(True)` :

```python
model = get_model("resnet50", weights=get_model_weights("resnet50").DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 5)          # the new head, created BEFORE the optimizer
for p in model.parameters():
    p.requires_grad_(False)                            # freeze everything...
for module in (model.layer4, model.fc):
    for p in module.parameters():
        p.requires_grad_(True)                         # ...then unfreeze the last block and the head

def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)

print(count_params(model), count_params(model, trainable_only=True))     # 23518277 14974981
```

Sur ResNet-50, `layer4` contient à lui seul 14 964 736 paramètres, près des deux tiers du réseau (64 %) : « dégeler le dernier bloc » n'est pas un petit changement. Les blocs profonds ont beaucoup de canaux, et une convolution a autant de poids que (taille du filtre) × (canaux d'entrée) × (canaux de sortie) : la convolution 1 × 1 qui passe de 512 à 2 048 canaux dans un bloc de `layer4` a $1 \times 1 \times 512 \times 2048 = 1\,048\,576$ poids (sans biais : la batchnorm qui suit en tient lieu). Les ✏️ B1.1 et B1.4 te font faire ces comptes sur le modèle du notebook.

### Pourquoi un learning rate beaucoup plus petit

Les poids pré-entraînés sont **déjà bons** : ils sont au fond d'une vallée de la loss d'ImageNet, et celle de ta tâche est, on l'espère, toute proche. De grands pas détruiraient en quelques itérations ce qui a coûté des jours. D'où un learning rate **10 à 100 fois plus petit** que de zéro (de $10^{-4}$ à $10^{-5}$ avec AdamW, au lieu de $10^{-3}$).

Le danger est maximal au début : la tête neuve est **aléatoire**, la loss est grande, et les gradients qu'elle renvoie vers le backbone sont grands et désordonnés. Trois remèdes, qui se combinent :
- un **warmup** (ch. 19), le temps que la tête devienne raisonnable ;
- un **linear probe d'abord**, puis le fine-tuning : c'est **LP-FT** (A. Kumar et coll., 2022). Sur dix datasets à décalage de distribution, un fine-tuning direct, parti d'une tête aléatoire, gagnait en moyenne 2 points sur le linear probe pour des données comme celles du train, mais en perdait 7 hors de cette distribution : il **déforme** les features pré-entraînées. LP-FT faisait mieux que les deux ;
- des **learning rates différents par couche**, plus petits près de l'entrée.

### Learning rates discriminatifs

Les premières couches sont les plus générales (201.1) : on veut les bouger le moins. J. Howard et S. Ruder l'ont systématisé en 2018 (ULMFiT, pour le texte) avec les **learning rates discriminatifs** (*discriminative learning rates*) : chaque groupe de paramètres a son learning rate, qui **décroît géométriquement** de la tête vers l'entrée. Avec $L$ groupes numérotés de l'entrée ($\ell = 1$) à la tête ($\ell = L$) :

$$\eta_\ell = \eta_{\text{tête}}\,\gamma^{L-\ell}, \qquad 0 < \gamma < 1.$$

ULMFiT divisait par 2,6 d'une couche à la précédente ($\gamma \approx 0{,}38$) ; en vision, on voit souvent $\gamma$ entre 0,1 et 0,5. Exemple : trois groupes (`layer3`, `layer4`, la tête), $\eta_{\text{tête}} = 2 \times 10^{-3}$ et $\gamma = 0{,}5$ donnent $2 \times 10^{-3}$ à la tête, $10^{-3}$ à `layer4` et $0{,}5 \times 10^{-3}$ à `layer3`. En PyTorch, un optimiseur accepte une liste de **groupes de paramètres** (*param groups*), chacun avec ses réglages :

```python
head_lr, gamma = 2e-3, 0.5
groups = [model.fc, model.layer4, model.layer3]                       # from the head down
param_groups = [{"params": g.parameters(), "lr": head_lr * gamma ** k} for k, g in enumerate(groups)]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
print([pg["lr"] for pg in optimizer.param_groups])                    # [0.002, 0.001, 0.0005]
```

Un bloc qui n'est dans aucun groupe n'est jamais mis à jour, même si son `requires_grad` est vrai : l'optimiseur ne connaît que les paramètres qu'on lui a donnés. Gèle-le quand même, pour ne pas calculer ses gradients pour rien.

### Warmup puis cosinus, groupe par groupe

Le planning du ch. 19 s'applique **à chaque groupe**. Le warmup linéaire sur $t_{\text{warmup}}$ pas s'écrit

$$\eta_t = \eta_{\max}\,\min\!\left(1, \frac{t}{t_{\text{warmup}}}\right),$$

avec $\eta_{\max}$ le learning rate de pointe du groupe ; une décroissance en cosinus prend ensuite le relais. Avec l'exemple précédent et $t_{\text{warmup}} = 4$, la tête reçoit 0, puis 0,5, 1, 1,5 et $2 \times 10^{-3}$ aux pas 0 à 4, et `layer3` quatre fois moins à chaque pas. `LambdaLR` (ch. 23) s'en charge pour tous les groupes à la fois, car sa fonction rend un **facteur** qui multiplie le learning rate initial de **chaque** groupe :

```python
import math

def warmup_cosine(step, t_warmup=4, t_total=200):
    if step < t_warmup:
        return step / t_warmup                                        # min(1, t / t_warmup)
    progress = (step - t_warmup) / (t_total - t_warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
# in the training loop, after each optimizer.step(): scheduler.step()   (one step per mini-batch)
```

Comme `cosine_lr` au ch. 19, le learning rate vaut 0 au pas 0 (la convention de Hugging Face). Ici le planning avance **à chaque mini-batch**, `t_total` étant le nombre total de pas, alors qu'au ch. 23 il avançait à chaque epoch. Le ✏️ B1.3 te fait suivre chaque groupe pas à pas.

![À gauche : trois groupes à learning rates discriminatifs (tête, bloc 4, bloc 3, rapport 0,5 de l'un à l'autre), un warmup de 6 pas puis une décroissance en cosinus jusqu'à 0 au pas 60 ; à droite : un dégel progressif, ici un groupe de plus toutes les deux epochs (la tête seule, puis le bloc 4, puis le bloc 3)](figures/planning_lr.png)

### Le dégel progressif

ULMFiT proposait aussi le **dégel progressif** (*gradual unfreezing*) : d'abord la tête seule, puis on dégèle le dernier groupe, puis l'avant-dernier, un de plus par epoch. Chaque groupe ne bouge qu'une fois ceux du dessus adaptés. En PyTorch, on crée l'optimiseur avec **tous** les groupes dès le départ, puis on bascule `requires_grad` au bon moment : `step()` ignore un paramètre dont le gradient est `None`. Dégel progressif et LP-FT relèvent de la même prudence : ne laisser bouger les couches générales qu'une fois la tête raisonnable.

### La batchnorm pendant le fine-tuning

Une batchnorm (ch. 20) a deux régimes. En mode `train()`, elle normalise avec la moyenne et la variance **du mini-batch** et met à jour ses statistiques glissantes ; en mode `eval()`, elle normalise avec ces statistiques glissantes, ici celles d'ImageNet. Sur CPU, avec des mini-batches de 8, voire de 4 images, le mode `train()` pose deux problèmes : les statistiques de 4 images sont très bruitées, et les statistiques glissantes, mises à jour à chaque pas, s'éloignent de celles avec lesquelles les poids ont été appris. À l'évaluation, la batchnorm se sert de ces statistiques abîmées : l'accuracy de validation peut s'effondrer alors que la loss d'entraînement baisse.

La parade habituelle : garder les batchnorms en mode `eval()` pendant tout le fine-tuning. Elles conservent les statistiques d'ImageNet ; leurs paramètres appris (l'échelle et le décalage, `weight` et `bias`) peuvent s'entraîner ou rester gelés.

```python
def freeze_batchnorm(model):
    """Put every batchnorm in eval mode: it normalizes with its stored ImageNet statistics."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

model.train()                  # dropout on, batchnorm in training mode...
freeze_batchnorm(model)        # ...then batchnorm back to eval mode; call both at the start of EVERY epoch
```

L'ordre compte : `model.train()` remet **tous** les modules en mode entraînement. Si ta fonction `fit()` du ch. 23 appelle `model.train()` au début de chaque epoch, c'est juste après qu'il faut appeler `freeze_batchnorm`. Avec de grands mini-batches et des données éloignées d'ImageNet, laisser les batchnorms s'adapter peut au contraire aider : c'est une décision à mesurer (🧠 Q7).

### Rester près des poids pré-entraînés : la pénalité L2-SP

Le weight decay d'AdamW tire chaque poids vers **0**. Pour un réseau pré-entraîné, 0 n'a rien de spécial : ce sont les poids **de départ** qui portent le savoir. X. Li, Y. Grandvalet et F. Davoine (2018) tirent donc les poids vers leurs valeurs pré-entraînées $\theta_0$ : c'est la pénalité **L2-SP** (*L2 – starting point*) :

$$\Omega(\theta) = \frac{\lambda}{2}\,\lVert\theta - \theta_0\rVert^2, \qquad \nabla\Omega = \lambda\,(\theta - \theta_0).$$

On l'ajoute à la loss pour les poids du backbone ; la tête neuve, sans valeur de départ, garde une pénalité L2 ordinaire. Le gradient est une force de rappel proportionnelle à l'écart. Exemple : $\lambda = 0{,}01$, un poids parti de $\theta_0 = 0{,}8$ et arrivé à $\theta = 1{,}1$ ; la pénalité vaut $\frac{0{,}01}{2} \times 0{,}3^2 = 0{,}00045$ et ajoute $0{,}01 \times 0{,}3 = 0{,}003$ à son gradient, qui le pousse à redescendre vers 0,8 (∂ B1.6). En code : une copie détachée des poids de départ (`theta0 = [p.detach().clone() for p in params]`), le terme `lam / 2 * sum(((p - p0) ** 2).sum() for p, p0 in zip(params, theta0))` ajouté à la loss, et `weight_decay=0` pour ces groupes, sinon AdamW les tirerait aussi vers 0.

> 🕰️ **Mise à jour (2026) — la recette de fine-tuning** — **Le livre :** conseille de geler les premières couches et d'entraîner les autres avec un très petit learning rate (§23.9.4). · **Aujourd'hui :** **AdamW** avec un petit learning rate, un **warmup** puis un **cosinus**, des **groupes de paramètres** (learning rates discriminatifs, pas de weight decay sur les biais et les normalisations), éventuellement un **dégel progressif** ou LP-FT. Pour les très grands modèles, on ne fine-tune plus tous les poids : les méthodes **PEFT** (*parameter-efficient fine-tuning*) comme **LoRA** n'entraînent que de petites matrices ajoutées au modèle gelé (bonus B4). · **Faut-il quand même l'apprendre ?** Oui : param groups et warmup servent dans tous les fine-tunings, LoRA compris. · *Sources :* [Howard et Ruder, ULMFiT, ACL 2018, arXiv:1801.06146](https://arxiv.org/abs/1801.06146) ; [documentation de `torch.optim` 2.11](https://docs.pytorch.org/docs/2.11/optim.html) (options par paramètre) ; [Kumar et coll., ICLR 2022, arXiv:2202.10054](https://arxiv.org/abs/2202.10054).

---

## 201.5 · Adapter l'entrée et la sortie : résolution, canaux, normalisation, nouvelle tête ⏩

### La résolution

Un ResNet de torchvision divise la taille de l'image par 4 dans son stem (une convolution de stride 2, puis un max pooling de stride 2), puis par 2 au début de `layer2`, `layer3` et `layer4` : par **32** en tout. Le pooling moyen global (ch. 21) réduit ensuite la carte d'activation finale à un vecteur, **quelle que soit sa taille** : un même ResNet accepte des images de 224, de 160 ou de 96 pixels. Pour ResNet-50 sur des images de 160 × 160 :

| Étape | Forme du tenseur |
|---|---|
| entrée | $(N, 3, 160, 160)$ |
| stem (`conv1`, `bn1`, ReLU, `maxpool`) | $(N, 64, 40, 40)$ |
| `layer1` | $(N, 256, 40, 40)$ |
| `layer2` | $(N, 512, 20, 20)$ |
| `layer3` | $(N, 1024, 10, 10)$ |
| `layer4` | $(N, 2048, 5, 5)$ |
| `avgpool` puis aplatissement : $\phi(\mathbf{x})$ | $(N, 2048)$ |
| nouvelle tête `fc` | $(N, K)$ |

Le calcul accepte presque toutes les tailles, mais **ce que le réseau a appris dépend de l'échelle** : il a vu des objets qui occupent une bonne part d'une image de 224 pixels. Que se passe-t-il quand on lui donne les images de 32 × 32 de CIFAR-10 ? C'est le 🔮 B1.12 ; le ✏️ B1.1 te fait suivre les formes, et le 🔬 B1.14 mesure le compromis entre résolution, accuracy et temps. Le coût d'un CNN est à peu près proportionnel au **nombre de pixels** : passer de 160 à 224 pixels de côté multiplie le calcul par $(224/160)^2 \approx 2$.

Les Vision Transformers sont plus rigides : `vit_b_16` de torchvision n'accepte que des images de 224 × 224 (« Wrong image height! » sinon), car il apprend un vecteur de position par morceau d'image. DINOv2 interpole les siens et accepte d'autres tailles, de préférence multiples de 14.

### Les canaux

Les modèles ImageNet attendent **trois** canaux. Pour des images en niveaux de gris (radiographies, Fashion-MNIST) :
- **répéter** le canal : `x.expand(-1, 3, -1, -1)` sur un tenseur $(N, 1, H, W)$, ou `v2.Grayscale(num_output_channels=3)` ;
- ou **sommer** les poids de la première convolution sur l'axe des canaux, pour obtenir une convolution à un canal d'entrée : c'est exactement équivalent (trois canaux identiques multipliés chacun par son filtre, puis additionnés), et trois fois moins coûteux pour cette couche.

Pour plus de trois canaux (13 bandes d'une image satellite), on agrandit la première convolution : les poids pré-entraînés pour les canaux RVB, la moyenne des trois pour les autres, par exemple.

### La normalisation

On normalise avec la moyenne et l'écart-type **du pré-entraînement**, pas avec ceux de ton dataset : les poids ont été appris pour des entrées centrées ainsi. Les modèles n'ont pas tous les mêmes : torchvision et DINOv2 prennent ceux d'ImageNet ; CLIP les siens (`mean=[0.4815, 0.4578, 0.4082]`, `std=[0.2686, 0.2613, 0.2758]`, arrondis) ; SigLIP 2 ramène chaque canal dans $[-1, 1]$. Prends donc le **processeur du modèle**, ne recopie pas des nombres trouvés ailleurs. Oublier la normalisation ne déclenche aucune erreur : les formes sont bonnes, les prédictions simplement mauvaises.

Pour augmenter les données pendant un fine-tuning (ch. 24), la normalisation reste à la fin de la chaîne :

```python
from torchvision.transforms import v2

t = weights.transforms()                      # read the sizes and statistics of the weights
train_tf = v2.Compose([
    v2.RandomResizedCrop(t.crop_size[0], scale=(0.6, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),     # uint8 [0, 255] -> float32 [0, 1]
    v2.Normalize(mean=t.mean, std=t.std),                    # the pretraining statistics
])
eval_tf = t                                    # validation and test: the deterministic preprocessing
```

### La nouvelle tête

On remplace la dernière couche dense par une couche à $K$ sorties, de même nombre d'entrées $d$ :
- ResNet, RegNet : `model.fc = nn.Linear(model.fc.in_features, K)` ;
- EfficientNet, MobileNet, ConvNeXt, VGG : la tête est un `nn.Sequential` nommé `classifier`, dont on remplace le dernier élément, `model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, K)` ;
- un modèle du Hub chargé par `AutoModel` n'a pas de tête : on ajoute `nn.Linear(model.config.hidden_size, K)` sur `pooler_output`.

Chez certains modèles, la tête d'origine compte **plus d'une couche** : celle des MobileNetV3 de torchvision enchaîne une couche dense, une activation, un dropout et une seconde couche dense. L'embedding se lit alors à l'**entrée** de `classifier` ou entre ses deux couches denses, et ce n'est pas le même vecteur (📦 B1.11). Imprime toujours `model` avant de couper.

Dernière règle, souvent oubliée : on crée l'optimiseur **après** avoir remplacé la tête. Il reçoit sa liste de paramètres à sa création ; créé avant, il continue de mettre à jour l'ancienne tête, qui ne sert plus, et jamais la nouvelle. Vérification : `any(p is model.fc.weight for g in optimizer.param_groups for p in g["params"])` doit valoir `True`.

---

## 201.6 · Peu de données : courbes d'apprentissage, augmentation, few-shot ⏩

### La courbe d'apprentissage selon la taille du train

Pour savoir si le pré-entraînement t'aide, et de combien, on trace une **courbe d'apprentissage**, l'accuracy de test en fonction du nombre d'images d'entraînement par classe, pour chaque stratégie (from scratch, linear probe, fine-tuning). L'axe horizontal est **logarithmique** (10, 30, 100, 300, 1 000 images par classe) : c'est l'ordre de grandeur qui compte. Un protocole honnête, celui du 🔬 B1.21 :
- des sous-ensembles **emboîtés** et **stratifiés** du train, tirés avec une graine fixée (les 10 images par classe font partie des 30, qui font partie des 100…) : d'un point à l'autre, seule la quantité change ;
- le **même** jeu de test pour tous les points, jamais utilisé pour régler quoi que ce soit ;
- **plusieurs graines** pour les petits effectifs : avec 10 images par classe, l'accuracy dépend beaucoup des images tirées (ch. 2) ;
- le même budget de réglage pour chaque stratégie.

L'intuition dit que le pré-entraînement compte d'autant plus que les données sont rares. K. He, R. Girshick et P. Dollár (2019) l'ont nuancée : en détection d'objets sur COCO, un réseau entraîné de zéro **assez longtemps** rattrapait le même réseau pré-entraîné, même avec 10 % des images ; le pré-entraînement accélérait surtout la convergence. Où se situe le croisement pour CIFAR-10, et combien vaut l'écart à 10 images par classe ? Prédis-le dans le 🔮 B1.20 avant de le mesurer.

### L'augmentation avec un modèle pré-entraîné

L'augmentation du ch. 24 marche aussi en fine-tuning, et compte d'autant plus que le train est petit. Deux remarques propres au transfert. Elle est **incompatible avec le cache des embeddings** : un recadrage différent à chaque epoch oblige à repasser dans le backbone ; avec un backbone gelé, on peut calculer une fois quelques versions augmentées de chaque image. Et les transformations doivent rester **plausibles** : une rotation de 90° est naturelle pour une image satellite, absurde pour une photo de rue.

### Le few-shot et les prototypes de classe

À l'extrême, on n'a que **quelques exemples par classe** (1, 5, 16) : c'est le **few-shot** (*à quelques exemples*). Avec un bon embedding, la méthode la plus simple est aussi l'une des plus solides : représenter chaque classe $c$ par la **moyenne** des embeddings de ses exemples, son **prototype** :

$$\mathbf{m}_c = \frac{1}{|S_c|}\sum_{\mathbf{x} \in S_c} \phi(\mathbf{x}),$$

où $S_c$ est l'ensemble des exemples étiquetés de la classe $c$ (le *support*) et $|S_c|$ leur nombre. Une nouvelle image va dans la classe du **prototype le plus proche**, au sens du cosinus. C'est le classifieur au centre le plus proche, celui des *Prototypical Networks* (J. Snell, K. Swersky et R. Zemel, 2017) ; normaliser les embeddings avant la moyenne l'améliore souvent.

Exemple en dimension 2. « Pomme » : $(0{,}9 ; 0{,}1)$ et $(0{,}7 ; 0{,}3)$, de prototype $(0{,}8 ; 0{,}2)$. « Poire » : $(0{,}2 ; 0{,}8)$ et $(0{,}4 ; 0{,}8)$, de prototype $(0{,}3 ; 0{,}8)$. Pour $\mathbf{q} = (0{,}6 ; 0{,}5)$ : $\cos(\mathbf{q}, \mathbf{m}_{\text{pomme}}) = 0{,}58 / (0{,}781 \times 0{,}825) \approx 0{,}900$ et $\cos(\mathbf{q}, \mathbf{m}_{\text{poire}}) = 0{,}58 / (0{,}781 \times 0{,}854) \approx 0{,}869$ : « pomme », de peu.

```python
def class_prototypes(emb, labels, n_classes):
    """Mean of the L2-normalised embeddings of each class: shape (n_classes, d)."""
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return np.stack([emb[labels == c].mean(axis=0) for c in range(n_classes)])

def predict_nearest_prototype(emb, protos):
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    return (emb @ protos.T).argmax(axis=1)                 # cosine with every prototype, best class
```

Ajouter une classe, c'est calculer un prototype de plus. Le 📦 B1.25 compare les prototypes de DINOv2 et de CLIP de 1 à 16 exemples par classe.

---

## 201.7 · Quand le transfert échoue : distance de domaine, transfert négatif, oubli catastrophique ⏩

### La distance de domaine

Le transfert marche quand la tâche cible ressemble à la tâche source. On parle de **distance de domaine** (*domain gap*) : un domaine, c'est le genre de données (photos du quotidien, radiographies, images satellites, spectrogrammes, dessins) et la façon dont elles ont été produites (cadrage, capteur, résolution, couleur). Signes d'un domaine lointain pour un modèle ImageNet : pas de couleur, ou des couleurs qui ne veulent rien dire ; des objets toujours centrés sur fond uni ; une échelle très différente (des cellules, des galaxies) ; des détails de quelques pixels qui décident de la classe. **Fashion-MNIST** en est un exemple commode : des vêtements, mais en niveaux de gris, en 28 × 28, centrés sur fond noir (🔬 B1.22).

Même des photos ordinaires réservent des surprises : S. Kornblith, J. Shlens et Q. Le (2019) ont trouvé que, sur deux datasets à classes très fines (*fine-grained* : 196 modèles de voitures pour Stanford Cars, 100 modèles d'avions pour FGVC Aircraft), un Inception v4 entraîné de zéro faisait presque aussi bien que fine-tuné (92,7 % contre 93,3 % sur les voitures, 88,8 % contre 89,0 % sur les avions). La distance de domaine est la version « features » du **décalage de distribution** du ch. 29.

![Matrice deux sur deux : peu de données et domaine proche, extraction de features ; beaucoup de données et domaine proche, fine-tuning complet avec un petit learning rate ; peu de données et domaine lointain, le cas difficile, où l'on sonde les blocs intermédiaires et l'on cherche un modèle plus proche ; beaucoup de données et domaine lointain, fine-tuning complet comparé à un entraînement from scratch](figures/matrice_decision.png)

La matrice donne un premier choix, que les courbes d'apprentissage (201.6) confirment ou non. Sur un domaine lointain, les dernières couches, les plus spécifiques à ImageNet, sont celles qui risquent le plus de ne plus servir : il vaut alors la peine de **sonder** les blocs intermédiaires (201.3), et de chercher un **modèle de fondation** entraîné sur des données plus variées (201.8) ou un modèle pré-entraîné sur un domaine voisin (le Hub en propose beaucoup pour l'imagerie médicale ou satellitaire).

### Le transfert négatif

On parle de **transfert négatif** (*negative transfer*) quand partir du modèle pré-entraîné donne un résultat **pire** que partir de zéro, à budget égal. C'est rare avec un fine-tuning complet bien réglé, plus fréquent avec un backbone gelé sur un domaine lointain. Le seul moyen de le voir est de **mesurer contre des baselines** : un modèle entraîné de zéro avec le même soin (le CNN du ch. 21 pour Fashion-MNIST) et la classe majoritaire. « Le transfert fait 85 % » ne veut rien dire tant qu'on ignore ce que fait un petit CNN from scratch sur les mêmes données.

### L'oubli catastrophique

Après un fine-tuning, le réseau est bon sur la nouvelle tâche ; mais qu'est devenu ce qu'il savait faire ? Souvent, il l'a **oublié** : ses features se sont réorganisées, et l'ancienne tête, remise en place, ne s'y retrouve plus forcément. C'est l'**oubli catastrophique** (*catastrophic forgetting*), décrit dès 1989 par M. McCloskey et N. Cohen. Il compte dès qu'un modèle doit garder ses anciennes compétences : un assistant qu'on spécialise sans qu'il perde sa culture générale, un détecteur qu'on enrichit de nouvelles classes.

Pour le **mesurer**, on remet l'ancienne tête sur le backbone fine-tuné et l'on compare ses prédictions à celles du modèle d'origine, sur les mêmes images : la part d'images où elles coïncident (l'accord top-1) dit ce qui reste (🔬 B1.23). Pour le **limiter** : un petit learning rate et peu d'epochs ; des learning rates discriminatifs et le gel des premiers blocs ; la pénalité L2-SP, ou sa cousine **EWC** (J. Kirkpatrick et coll., 2017), qui pondère le rappel par l'importance de chaque poids pour l'ancienne tâche ; LP-FT ; et, quand c'est possible, **garder deux modèles**, ou un backbone gelé partagé et une tête par tâche.

> ⚖️ **Biais hérités** — Un modèle pré-entraîné transmet ses features, et avec elles les biais de ses données. Les images d'ImageNet viennent surtout d'Amérique du Nord et d'Europe ; celles de CLIP, récoltées sur Internet, représentent davantage « les personnes et les sociétés les plus connectées à Internet », plutôt des pays développés, plutôt jeunes et masculines, dit sa fiche modèle. Un modèle dérivé hérite de ces absences, même entraîné ensuite sur des données soignées : sa fiche modèle doit le dire, et l'évaluation doit le mesurer sur les populations de l'usage visé (⚖️ B1.9, bonus B6).

---

## 201.8 · Modèles de fondation : auto-supervisé (DINOv2), texte-image (CLIP), zéro-shot ⏩

### Qu'est-ce qu'un modèle de fondation ?

Un **modèle de fondation** (*foundation model*, R. Bommasani et coll., 2021) est un grand modèle entraîné une fois, sur des données très vastes et variées, sans viser une tâche précise, puis réutilisé pour beaucoup de tâches. Les LLM en sont (bonus B4). En vision, deux familles servent d'extracteurs de features universels : les modèles **auto-supervisés** (ch. 25), qui apprennent sans aucun label (DINOv2, DINOv3), et les modèles **texte-image**, qui apprennent d'images accompagnées de leur légende (CLIP, SigLIP 2). Ils n'ont pas appris à séparer 1 000 classes choisies d'avance, mais à décrire des images en général : leurs embeddings se transfèrent souvent mieux, même par un simple linear probe ou un k-NN.

### DINOv2 : apprendre sans labels

DINOv2 (M. Oquab et coll., Meta, 2023) est un Vision Transformer entraîné par **auto-distillation** : un réseau « élève » voit des recadrages d'une image et doit produire les mêmes sorties qu'un « professeur » (une moyenne glissante de l'élève) qui en voit d'autres ; une seconde loss lui fait deviner des morceaux d'image masqués. Personne n'a dit ce que contient l'image : pour réussir, le réseau doit apprendre ce qui reste stable d'une vue à l'autre, c'est-à-dire **ce qu'est** l'objet. Les auteurs insistent sur les **données** : 142 millions d'images choisies automatiquement pour leur diversité. Ils ont entraîné un ViT d'environ un milliard de paramètres, puis l'ont **distillé** : des modèles plus petits (*small*, *base*, *large*) ont appris à reproduire ses sorties ; le notebook utilise le plus petit (📦 B1.11). L'embedding est le token `CLS` (`pooler_output`), et le mode d'emploi prévu est le linear probe ou le k-NN, **backbone gelé**.

### CLIP : rapprocher les images de leur légende

CLIP (A. Radford et coll., OpenAI, 2021) entraîne **deux encodeurs**, un pour les images, un pour les textes, qui rendent des vecteurs de même dimension (512 pour `clip-vit-base-patch32`), sur 400 millions de paires (image, légende) récoltées sur Internet. Le principe est **contrastif** : dans un mini-batch de $N$ paires (32 768 dans l'article), l'embedding de chaque image doit être plus proche, au sens du cosinus, de **sa** légende que des $N - 1$ autres, et réciproquement. On calcule la matrice $N \times N$ des cosinus, on la multiplie par une échelle $s$, et l'on applique une cross-entropy sur chaque ligne et sur chaque colonne.

### Le zéro-shot

CLIP classe ensuite des images dans des classes qu'on ne lui a jamais apprises comme telles, sans un seul exemple : c'est le **zéro-shot** (*zero-shot*, sans exemple). On écrit une phrase par classe à partir d'un **gabarit** (*prompt template*), « a photo of a {classe} » ; l'encodeur de texte en tire un vecteur $\mathbf{v}_c$ par classe, l'encodeur d'image tire $\mathbf{u}_x$ de l'image, et la probabilité de chaque classe est une softmax (ch. 17) des cosinus multipliés par $s$ :

$$p(c \mid \mathbf{x}) = \frac{\exp\big(s \cos(\mathbf{u}_x, \mathbf{v}_c)\big)}{\sum_{c'} \exp\big(s \cos(\mathbf{u}_x, \mathbf{v}_{c'})\big)}.$$

Les vecteurs de texte jouent le rôle des lignes de $\mathbf{W}$ d'un linear probe : CLIP fabrique une tête **à partir de mots**.

L'échelle $s$ (`logit_scale`) est **apprise**. CLIP la stocke sous forme de logarithme, initialisée à $\ln(1/0{,}07) \approx 2{,}659$ (le `logit_scale_init_value = 2.6592` de `config.json` est cette valeur **de départ**), et l'empêche de dépasser 100. Pour `openai/clip-vit-base-patch32`, le paramètre entraîné vaut $\ln 100 \approx 4{,}6052$ : `model.logit_scale.exp()` rend **100**. Pourquoi si grand ? Les cosinus d'une image avec les différentes légendes sont très proches les uns des autres ; sans multiplication, la softmax serait presque uniforme.

Exemple : une image dont les cosinus avec « a photo of a cat », « … dog » et « … truck » valent 0,31, 0,27 et 0,22. Avec $s = 100$, les logits sont 31, 27 et 22 ; leurs écarts au plus grand, 0, −4 et −9 ; $e^{0} = 1$, $e^{-4} \approx 0{,}0183$, $e^{-9} \approx 0{,}000123$, de somme $\approx 1{,}0184$ ; d'où les probabilités 0,982, 0,018 et 0,0001. Avec $s = 1$, les mêmes cosinus donneraient 0,348, 0,334 et 0,318 : presque rien ne distinguerait les classes. Le ✏️ B1.5 te fait refaire ce calcul sur d'autres cosinus.

![Zéro-shot de CLIP : l'encodeur d'image rend u, l'encodeur de texte rend un vecteur par phrase « a photo of a cat », « … dog », « … truck » ; les cosinus 0,31, 0,27 et 0,22, multipliés par s = 100 puis passés dans une softmax, donnent les probabilités 0,982, 0,018 et 0,0001](figures/clip_zero_shot.png)

En code, avec transformers 5 :

```python
from transformers import CLIPModel, CLIPProcessor

name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(name).eval()
processor = CLIPProcessor.from_pretrained(name)
texts = [f"a photo of a {c}" for c in ["cat", "dog", "truck"]]
with torch.inference_mode():
    v = model.get_text_features(**processor(text=texts, return_tensors="pt", padding=True)).pooler_output
    u = model.get_image_features(**processor(images=images, return_tensors="pt")).pooler_output
    u, v = u / u.norm(dim=-1, keepdim=True), v / v.norm(dim=-1, keepdim=True)   # (N, 512) and (3, 512)
    probs = (model.logit_scale.exp() * u @ v.T).softmax(dim=-1)                # (N, 3): the formula above
```

`model(**processor(text=texts, images=images, return_tensors="pt", padding=True)).logits_per_image` donne les mêmes logits en une ligne ; le calcul écrit montre qu'il n'y a là qu'un cosinus, une échelle et une softmax.

**Les gabarits comptent.** Le nom seul (« cat ») est ambigu ; une phrase ressemble davantage aux légendes vues à l'entraînement. Dans l'article, le gabarit « A photo of a {label}. » faisait gagner 1,3 point sur ImageNet par rapport au nom seul, et la moyenne des embeddings de **80 gabarits** (« a blurry photo of a {} », « a drawing of a {} »…) 3,5 points de plus : c'est un **ensemble de prompts** (*prompt ensembling*). On normalise chaque vecteur de texte, on fait la moyenne par classe et l'on renormalise (📦 B1.24). Les embeddings d'image de CLIP et de DINOv2 se prêtent aussi aux prototypes de 201.6, et avec CLIP on peut combiner le vecteur de texte d'une classe et le prototype de ses quelques images (📦 B1.25).

**Les limites.** Sa fiche modèle le dit : CLIP peine sur les classifications fines et le comptage d'objets, ses performances dépendent beaucoup de la liste de classes proposée, et il n'a été évalué qu'en anglais. Le zéro-shot est un formidable point de départ, rarement le point d'arrivée.

> 🕰️ **Mise à jour (2026) — les backbones de référence** — **Le livre :** VGG16 (ch. 23 et 28), entraîné sur ImageNet-1k avec des labels, comme ResNet-50, l'autre référence de 2018. · **Aujourd'hui :** côté supervisé, des CNN modernes (ConvNeXt, EfficientNet) et des Vision Transformers (ViT, Swin) ; côté modèles de fondation, des backbones auto-supervisés (**DINOv2**, 2023, Apache 2.0 ; **DINOv3**, 2025, de ViT-S à ViT-7B, 1,7 milliard d'images, licence propre à Meta) et texte-image (**CLIP**, 2021 ; **SigLIP 2**, 2025, de 86 millions à 1 milliard de paramètres, multilingue, Apache 2.0) ; leurs petites versions tournent sur un CPU. · **Faut-il quand même l'apprendre ?** Oui : le principe (un backbone, un embedding, une tête) ne change pas, seul le backbone change. · *Sources :* [Oquab et coll., DINOv2, arXiv:2304.07193](https://arxiv.org/abs/2304.07193) ; [fiche modèle de DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/blob/main/README.md) ; [Tschannen et coll., SigLIP 2, arXiv:2502.14786](https://arxiv.org/abs/2502.14786) ; [Radford et coll., CLIP, arXiv:2103.00020](https://arxiv.org/abs/2103.00020).

---

## 201.9 · Coût d'usage : taille, latence CPU, choix d'un backbone pour la production ⏩

Un modèle s'utilise des millions de fois après avoir été choisi une fois : son **coût d'usage** compte autant que son accuracy.

### Quatre mesures à ne pas confondre

| Mesure | Définition | ResNet-50 |
|---|---|---|
| **taille** | paramètres × octets par paramètre (4 en `float32`, 2 en `float16`, 1 en `int8`) | 25,6 millions de paramètres, ≈ 102 Mo (97,5 Mio) en `float32` |
| **coût de calcul** | opérations d'une passe avant, en GFLOPS (`weights.meta["_ops"]`) | 4,1 GFLOPS par image de 224 × 224 |
| **latence** (*latency*) | temps de réponse à **une** requête (batch de 1) | ≈ 95 à 140 ms sur deux cœurs de CPU |
| **débit** (*throughput*) | images traitées par seconde en regroupant les requêtes | ≈ 11 à 12 images/s en batches de 32, mêmes cœurs |

La taille se vérifie sur le fichier : un ResNet-50 à tête de 5 classes (23 518 277 paramètres) pèse 94,3 Mo en safetensors, pour $23\,518\,277 \times 4 \approx 94{,}1$ Mo de paramètres (le reste : les statistiques des batchnorms et l'en-tête). La latence compte pour une application interactive ; le débit, pour classer un million d'images la nuit. Regrouper augmente le débit, mais chaque image attend que son batch soit plein.

### Mesurer une latence correctement

```python
import time

def latency_ms(model, size=224, batch=1, reps=20):
    model.eval()
    x = torch.randn(batch, 3, size, size)
    with torch.inference_mode():
        for _ in range(3):
            model(x)                                   # warm-up runs, not timed (allocations, caches)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))              # the median resists a background task
```

Les règles : `eval()` et `inference_mode()`, comme en production ; quelques passes de chauffe non chronométrées ; la **médiane** de plusieurs mesures ; une machine au repos ; le nombre de cœurs noté (`torch.get_num_threads()`). Mesures faites ainsi pour cette fiche, sur deux cœurs, en 224 × 224, au fil de plusieurs exécutions (elles varient de 20 à 50 % d'une exécution à l'autre, davantage si la machine travaille à autre chose) :

| Backbone | Paramètres | GFLOPS | Latence, batch de 1 | Débit, batches de 32 | Top-1 ImageNet |
|---|---|---|---|---|---|
| MobileNetV3-large | 5,5 M | 0,22 | ≈ 18-30 ms | ≈ 55-70 images/s | 75,3 % |
| EfficientNet-B0 | 5,3 M | 0,39 | ≈ 28-46 ms | ≈ 35-45 images/s | 77,7 % |
| ConvNeXt-Tiny | 28,6 M | 4,46 | ≈ 85-105 ms | ≈ 14-16 images/s | 82,5 % |
| ResNet-50 | 25,6 M | 4,09 | ≈ 95-140 ms | ≈ 11-12 images/s | 80,9 % |

Deux leçons. **Les FLOPS ne sont pas la latence** : MobileNetV3-large fait près de 19 fois moins d'opérations que ResNet-50, mais ne va que 4 à 6 fois plus vite ; ses convolutions par canal (*depthwise*) calculent peu par octet lu, et c'est la mémoire qui devient le goulot. Et l'accuracy ImageNet ne suit ni la taille ni le coût : à coût voisin, ConvNeXt-Tiny bat ResNet-50 de 1,7 point.

![Accuracy top-1 sur ImageNet en fonction des GFLOPS par image (échelle logarithmique) pour treize backbones de torchvision, l'aire de chaque point étant proportionnelle au nombre de paramètres ; VGG16, le plus gros et l'un des plus coûteux, est aussi le moins précis](figures/backbones.png)

Une meilleure accuracy ImageNet annonce-t-elle un meilleur transfert ? En grande partie : Kornblith et ses coauteurs (2019) ont trouvé, sur 16 réseaux et 12 datasets, une corrélation de 0,99 entre l'accuracy ImageNet et celle d'un classifieur sur les features figées, et de 0,96 après fine-tuning. Mais des régularisations qui font gagner un peu sur ImageNet (label smoothing, dropout) donnent des features d'avant-dernière couche **nettement moins transférables**. Le classement ImageNet est un indice ; ta validation est la preuve.

### Choisir un backbone pour la production

Dans l'ordre où les questions éliminent des candidats :
1. **La licence** permet-elle l'usage prévu (201.2) ?
2. **Le budget de latence et de mémoire** (un téléphone, un serveur sans GPU) fixe une famille de tailles.
3. **L'accuracy sur ta validation**, pas sur ImageNet : un linear probe sur trois ou quatre candidats prend une heure (🔬 B1.26).
4. **Le prétraitement** : quelle résolution exige-t-il, et que coûte-t-elle (🔬 B1.14) ?
5. **La maintenance** : un dépôt officiel, à une révision figée, plutôt qu'un fichier trouvé sur un forum.

`torch.compile`, la quantification en `int8`, l'export ONNX ou la demi-précision sur GPU (ch. 24) font ensuite gagner du temps sans changer de modèle (bonus B7).

### Publier un modèle dérivé

Un modèle fine-tuné est un **modèle dérivé** : il hérite des poids, de la licence et des biais de son modèle de base. Le publier proprement, même dans ton portfolio, demande trois fichiers (🛠️ B1.27) :
- **les poids**, en **safetensors**, un format qui ne contient que des tenseurs et se charge sans exécuter de code (ch. 23) ;
- **une configuration** (`config.json`) : l'architecture et le nom exact des poids de départ, le nombre de classes et leurs noms **dans l'ordre des sorties**, le prétraitement, les versions des bibliothèques ;
- **une fiche modèle** (`README.md`) : le **modèle de base**, sa licence et un lien ; les données du fine-tuning et leur licence ; la recette ; les résultats sur un test jamais vu ; les limites et les biais hérités.

```python
from safetensors.torch import load_file, save_file

save_file(model.state_dict(), "model.safetensors",
          metadata={"base_model": "torchvision resnet50 IMAGENET1K_V2", "num_classes": "5"})  # str -> str only
model.load_state_dict(load_file("model.safetensors"))       # no pickle: nothing is executed while loading
```

> 🕰️ **Mise à jour (2026) — sauvegarder et publier des poids** — **Le livre :** sauvegarde ses modèles Keras en HDF5 (ch. 23) ; côté PyTorch, `torch.save` utilise `pickle`, et charger un fichier reçu pouvait exécuter n'importe quel code. · **Aujourd'hui :** depuis PyTorch 2.6, `torch.load` utilise `weights_only=True` par défaut ; pour **publier** des poids, **safetensors** est la norme du Hub (transformers 5 ne sauvegarde plus que dans ce format) ; pour un dépôt qui n'a qu'un `pytorch_model.bin`, comme `openai/clip-vit-base-patch32`, transformers télécharge à sa place une conversion en safetensors quand le Hub en propose une (une *pull request* du dépôt). · **Faut-il quand même l'apprendre ?** Oui : ne charge jamais un fichier de poids inconnu avec `weights_only=False`, et publie les tiens en safetensors. · *Sources :* [notes de sérialisation de PyTorch 2.11](https://docs.pytorch.org/docs/2.11/notes/serialization.html) (« Starting in version 2.6, torch.load will use weights_only=True ») ; [documentation de safetensors](https://huggingface.co/docs/safetensors/index) ; [guide de migration de transformers v5](https://raw.githubusercontent.com/huggingface/transformers/v5.20.0/MIGRATION_GUIDE_V5.md) (« Drop support for `safe_serialization=False` »).

> 💼 **En entreprise** — Un backbone se défend en réunion avec un petit tableau : trois ou quatre candidats, leur licence, leur latence sur la machine cible, leur accuracy sur la validation de l'équipe, leur taille. Le plus gros est rarement celui qu'on garde : deux fois plus lent pour un point d'accuracy, il coûte cher en serveurs pendant des années.

---

## Les pièges classiques ⚠️ (récapitulatif)

| Piège | Exemple | Ce qu'il faut faire |
|---|---|---|
| Oublier la normalisation du modèle, ou en prendre une autre | des pixels dans $[0, 1]$ donnés à un ResNet ; les statistiques d'ImageNet pour CLIP | le processeur du modèle (`weights.transforms()`, `AutoImageProcessor`) |
| Extraire des features en mode entraînement | embeddings calculés sans `model.eval()` | `eval()` et `inference_mode()` |
| Recalculer des embeddings figés à chaque epoch | un linear probe qui repasse tout le dataset dans le backbone | calculer une fois, sauvegarder en `.npz` |
| Le learning rate d'un entraînement de zéro | `lr=1e-1` sur tout un réseau pré-entraîné | 10 à 100 fois plus petit, plus petit vers l'entrée, avec un warmup |
| Des batchnorms en mode entraînement sur de tout petits mini-batches | mini-batches de 4 : validation qui s'effondre | `freeze_batchnorm(model)` après chaque `model.train()` |
| Créer l'optimiseur avant de remplacer la tête | la nouvelle tête reste aléatoire | remplacer la tête, **puis** créer l'optimiseur |
| Couper la mauvaise couche | toute la tête d'un MobileNetV3 remplacée au lieu de sa dernière couche | imprimer `model` avant de couper |
| Weight decay et L2-SP ensemble | AdamW, `weight_decay=0.1`, plus une pénalité vers $\theta_0$ | `weight_decay=0` sur les groupes pénalisés par L2-SP |
| Conclure sans baseline, sur une seule graine | « le transfert fait 85 % », mesuré une fois | un from scratch réglé avec le même soin, la classe majoritaire, plusieurs graines |
| Oublier ce que le modèle savait | un modèle fine-tuné remis sur l'ancienne tâche | mesurer l'oubli ; petit learning rate, L2-SP, ou deux modèles |
| Le nom de classe seul en zéro-shot | `texts = ["cat", "dog"]` | un gabarit, voire un ensemble de gabarits |
| Des embeddings de CLIP non normalisés | `u @ v.T` sur des vecteurs bruts | diviser chaque vecteur par sa norme |
| Prendre l'initialisation pour l'échelle apprise | $s = e^{2{,}6592} \approx 14{,}3$, lu dans `config.json` | `model.logit_scale.exp()`, lu dans les poids |
| Ignorer la licence | un modèle CC BY-NC dans un produit payant | lire la fiche modèle d'abord ; citer le modèle de base |
| Charger des poids inconnus sans précaution | `torch.load(f, weights_only=False)` sur un fichier téléchargé | `weights_only=True` (le défaut) ou safetensors |

## Liens avec les autres chapitres 🔗

- **0B, ch. 12, 13 et 15** : la similarité cosinus ; PCA et UMAP ; le k-NN et la régression logistique, appliqués ici à des embeddings.
- **Ch. 9, 16, 17 et 19** : la régularisation L2, qui devient L2-SP ; les paramètres d'une couche dense ; la softmax ; AdamW, warmup et cosinus, ici groupe par groupe.
- **Ch. 20 et 21** : la batchnorm et ses statistiques glissantes ; ResNet, le pooling moyen global, les cartes d'activation.
- **Ch. 23** : un ResNet-18 gelé et une nouvelle tête sur Fashion-MNIST (📦 23.24), `fit()`, `weights_only=True`, la fiche modèle ; B1 va jusqu'au fine-tuning complet, à l'oubli et au zéro-shot.
- **Ch. 24 et 29** : l'augmentation avec `transforms.v2` ; licences, décalage de distribution, data card.
- **B2 et B3** : les embeddings de mots et de phrases, comparés eux aussi par cosinus (B2.R1) ; les Vision Transformers, derrière DINOv2 et CLIP.
- **B4** : le fine-tuning des LLM avec PEFT et LoRA, la recherche par embeddings d'un RAG (B4.R2).
- **B6 et B7** : Grad-CAM sur le ResNet fine-tuné de ce chapitre (📦 B6.20) ; latence, quantification et export en production.
- **Projet final** : PF.6 demande un modèle adapté à tes données ; avec des images, ce sera presque toujours un modèle pré-entraîné, choisi et documenté comme ici.

## Guide de lecture et de travail

Ce chapitre n'a pas de lecture dans le livre : la fiche est le cours, et **tous les parcours la lisent en entier** (≈ 1,5 h), d'où ses neuf sections marquées ⏩. Lis chaque section en refaisant à la main ses exemples chiffrés, puis suis l'ordre conseillé en tête de fiche ; écris tes réponses papier dans `mon_travail/b1_transfer/06_mes_reponses.md` et vérifie les ✏️ dans la partie 0 du notebook (`wb.check`).

**Le notebook** télécharge ResNet-18 et MobileNetV3-small (torchvision), DINOv2-small (Hub), ≈ 150 Mo en tout, puis CLIP ViT-B/32 (≈ 600 Mo) pour la dernière phase ; si ce dernier téléchargement échoue, le few-shot se rabat sur les prototypes de DINOv2. En mode rapide, les embeddings sont calculés une fois et mis en cache (2 000 images, 96 à 128 pixels) ; les fine-tunings complets à 224 pixels (B1.19, B1.21) sont marqués 🚀 : mode rapide sur CPU, mode complet sur un GPU de Colab.

**Parcours rapide** : toute la fiche, les quiz Q1 à Q10 et les rappels, les ✏️ B1.1, B1.4 et B1.5, 🧮 B1.7, 🗣️ B1.8 et ⚖️ B1.9 ; dans le notebook B1.11, B1.13, B1.17 à B1.20, B1.24, B1.26 et B1.27 ; les cinq entretiens ; les corrigés de ✏️ B1.2 et B1.3 à lire. **Parcours maths** : les ✏️, ∂ B1.6, 🧮 B1.7 et 📄 B1.10, puis B1.11, B1.13 et B1.16. **Parcours code** : tout le notebook, et les corrigés de ✏️ B1.3, ✏️ B1.5 et 📄 B1.10 à lire. La liste exacte est dans `docs/PARCOURS.md`. **Si tu bloques** : règle des 15 minutes, puis les indices de `04_indices.md`, un niveau à la fois.

## Pour aller plus loin

- Le tutoriel de PyTorch [« Transfer Learning for Computer Vision Tutorial »](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) : extraction de features et fine-tuning d'un ResNet-18, pas à pas.
- Le guide de Hugging Face sur la [classification d'images zéro-shot](https://huggingface.co/docs/transformers/v5.21.0/en/tasks/zero_shot_image_classification) : CLIP et ses successeurs avec `transformers`.
- S. Kornblith, J. Shlens et Q. Le, « [Do Better ImageNet Models Transfer Better?](https://arxiv.org/abs/1805.08974) », CVPR 2019 : ce que le classement ImageNet dit, ou non, du transfert.
- A. Kumar et coll., « [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/abs/2202.10054) », ICLR 2022 : pourquoi un linear probe avant le fine-tuning.
- M. Oquab et coll., « [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) », 2023 : fabriquer un backbone universel sans labels.
