# 29 · Datasets et préparation du projet final — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch29_datasets/06_mes_reponses.md` (créée par `python tools/start_chapter.py 29`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ 29.1 et 29.2 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », l'estimation 🧮 29.3, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"B"`), plusieurs lettres entre guillemets pour un choix multiple (`"AC"`).
> Conventions du chapitre (fiche) : 1 Mo = $10^6$ octets, 1 Go = $10^9$ octets ; un `uint8` occupe 1 octet, un `float32` 4 octets ; $N$ exemples, $n_c$ de la classe $c$.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 29.Q1 — Partir de l'existant : modèle pré-entraîné, dataset public ou collecte ? 🧠 ⏱️ 3 min
*Fiche §29.1 · livre §29.1*

a) L'ordre que conseille le livre : (A) collecter ses données, puis chercher un modèle ; (B) un modèle déjà entraîné, ajusté si besoin, sinon un dataset public, et la collecte en dernier ; (C) toujours entraîner de zéro, pour maîtriser le résultat.
b) Vrai ou faux : ajuster un modèle pré-entraîné en entraînant quelques couches de plus sur ses propres données, c'est du transfer learning.
c) Le principal avantage de partir d'un modèle pré-entraîné : (A) il demande bien moins de calcul et de données ; (B) il est toujours meilleur qu'un modèle entraîné de zéro ; (C) il n'a pas de biais.
d) Vrai ou faux : un dataset public peut toujours servir dans un produit commercial.
e) Avant d'investir du temps dans un dataset trouvé en ligne, le livre conseille : (A) d'entraîner un grand modèle pour voir ; (B) de lire ses règles d'usage et d'inspecter les données à la main ; (C) de le doubler par augmentation.

### 29.Q2 — Licences et restrictions d'usage : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §29.1 · livre §29.1*

a) Un dataset sous licence CC BY peut servir à un usage commercial, à condition de citer sa source.
b) Un dataset sous licence CC BY-NC peut servir dans un produit vendu.
c) Un dataset « réservé à la recherche » peut entraîner le prototype interne d'une entreprise, qui sera ensuite vendu.
d) Une licence libre garantit que toutes les personnes photographiées dans un dataset ont donné leur accord.
e) Les données personnelles d'un dataset publié restent soumises au RGPD.

### 29.Q3 — Les défauts qu'une inspection manuelle révèle 🧠 ⏱️ 3 min
*Fiche §29.1, « Au-delà du livre : auditer un dataset » · livre §29.1*

a) Parmi ces défauts, lesquels le livre cite-t-il ? (A) des champs vides ; (B) des labels qui n'apparaissent qu'une fois ; (C) des fautes d'orthographe ; (D) des poids de réseau mal initialisés. Écris leurs lettres.
b) La valeur −999 dans une colonne de poids de fruits est le plus probablement : (A) un fruit au poids négatif ; (B) une valeur manquante codée par un nombre.
c) Vrai ou faux : `df.isna().sum()` compte les −999 d'une colonne.
d) Pour repérer des diamètres notés tantôt en centimètres, tantôt en millimètres : (A) `df.isna()` ; (B) `df["diametre"].describe()` ou un histogramme ; (C) `df.duplicated()`.
e) Vrai ou faux : on a trouvé des erreurs de labels jusque dans les jeux de test des benchmarks les plus utilisés.

### 29.Q4 — MNIST et Fashion-MNIST : même format, difficulté différente 🧠 ⏱️ 3 min
*Fiche §29.2 · livre §29.2*

a) Combien d'images compte Fashion-MNIST au total ?
b) Le côté d'une image, en pixels.
c) Le nombre de classes.
d) Vrai ou faux : un code qui charge et entraîne un modèle sur MNIST peut passer à Fashion-MNIST en changeant seulement la source des données.
e) Fashion-MNIST est plus difficile que MNIST parce que : (A) ses images sont en couleur ; (B) certaines classes se ressemblent beaucoup (T-shirt, chemise, pull) ; (C) il a moins d'exemples.

### 29.Q5 — Les datasets de scikit-learn : lequel pour quelle tâche, et pourquoi Boston a disparu 🧠 ⏱️ 3 min
*Fiche §29.3.1 · livre §29.3.1*

a) Combien d'images rend `sklearn.datasets.load_digits()` ?
b) `load_boston` a été retiré de scikit-learn dans la version : (A) 0.24 ; (B) 1.2 ; (C) 2.0.
c) La raison : (A) une de ses variables a été construite à partir de la proportion de population noire du quartier ; (B) il était trop petit ; (C) ses prix étaient faux.
d) Pour une régression de prix immobiliers, le workbook utilise : (A) Iris ; (B) California Housing ; (C) Breast Cancer.
e) Vrai ou faux : Diabetes, dans scikit-learn, est un dataset de textes.

### 29.Q6 — Des datasets de Keras à torchvision et Hugging Face 🧠 ⏱️ 3 min
*Fiche §29.3.2 · livre §29.3.2*

a) `keras.datasets.mnist.load_data()` rend : (A) un DataFrame ; (B) deux couples `(x_train, y_train), (x_test, y_test)` de tableaux NumPy ; (C) un `DataLoader`.
b) Combien d'images compte CIFAR-10 au total (entraînement et test) ?
c) Dans Keras, les critiques d'IMDB sont : (A) des textes bruts ; (B) des suites d'entiers, un par mot.
d) L'équivalent PyTorch pour charger CIFAR-10 : (A) `torch.load("cifar10")` ; (B) `torchvision.datasets.CIFAR10(root, train=True, download=True)`.
e) Vrai ou faux : l'argument `revision=` de `load_dataset` fixe la version exacte du dataset que l'on charge.

### 29.Q7 — Collections de datasets : UCI, Kaggle, OpenML, Hugging Face Hub et liens morts 🧠 ⏱️ 3 min
*Fiche §29.4 · livre §29.4*

a) Parmi ces sites, lesquels sont aujourd'hui de grandes sources de datasets ? (A) le Hugging Face Hub ; (B) Kaggle ; (C) OpenML ; (D) deeplearning.net. Écris leurs lettres.
b) Vrai ou faux : on peut télécharger un dataset Kaggle par son API sans compte.
c) `fetch_openml` fait partie de : (A) scikit-learn ; (B) torchvision ; (C) Hugging Face `datasets`.
d) Vrai ou faux : un lien vers un dataset noté dans un livre de 2018 marche encore aujourd'hui, en général.
e) Sur le Hugging Face Hub, la page qui décrit un dataset (contenu, licence, usages, limites) s'appelle : (A) une *dataset card* ; (B) un *model card* ; (C) un *README de code*.

### 29.Q8 — Les « nouveaux » datasets de 2018 : tâches, tailles et retraits 🧠 ⏱️ 4 min
*Fiche §29.5 · livre §29.5*

a) Open Images compte environ combien de millions d'images ?
b) Parmi ces datasets, lesquels ont été retirés par leurs auteurs ? (A) MS-Celeb-1M ; (B) COCO ; (C) 80 Million Tiny Images ; (D) Fashion-MNIST. Écris leurs lettres.
c) LAION-5B, retiré fin 2023, a été republié en 2024 sous le nom : (A) Re-LAION-5B ; (B) LAION-6B ; (C) il n'a pas été republié.
d) CelebA peut servir : (A) à tout usage ; (B) à la recherche non commerciale seulement.
e) Vrai ou faux : un dataset retiré par ses auteurs devient libre d'usage dès qu'on en trouve une copie sur un miroir.

<a id="rappels"></a>

## 🔁 Rappels

### 29.R1 — Qui possède les droits d'une image de style ? 🔁 ⏱️ 5 min
*ch. 28 · fiche §29.1*

a) Picasso est mort en 1973. En quelle année ses œuvres entreront-elles dans le domaine public de l'Union européenne ?
b) Vrai ou faux : dans l'Union européenne, une photo fidèle d'un tableau du domaine public crée de nouveaux droits pour le photographe.
c) Vrai ou faux : un style de peinture est protégé par le droit d'auteur.
d) Les photos livrées avec scikit-learn (`china.jpg`, `flower.jpg`) sont sous licence : (A) CC BY 2.0, avec attribution ; (B) domaine public, sans condition.
e) Van Gogh est mort en 1890. Depuis quelle année ses œuvres sont-elles dans le domaine public de l'Union européenne ?

### 29.R2 — En RL, d'où viennent les données ? 🔁 ⏱️ 5 min
*ch. 26 · fiche §29.1*

a) En apprentissage par renforcement, les données d'entraînement viennent : (A) d'un dataset annoté à l'avance ; (B) des interactions de l'agent avec son environnement.
b) Vrai ou faux : la distribution des données change quand la politique de l'agent change.
c) Une mémoire de rejeu (*replay buffer*) stocke : (A) des transitions $(s, a, r, s', \text{fin})$ ; (B) des images annotées ; (C) les poids du réseau.
d) Combien d'états a FrozenLake 4 × 4 ?
e) Vrai ou faux : on découpe les données d'un agent de RL en entraînement et test une fois pour toutes, comme un dataset supervisé.

### 29.R3 — Biais d'échantillonnage : un dataset représente-t-il la population visée ? 🔁 ⏱️ 5 min
*ch. 2 · fiche §29.1 (encadré ⚠️)*

a) Combien de manchots compte le dataset Palmer Penguins ?
b) Combien d'espèces ?
c) Un biais d'échantillonnage, c'est : (A) un échantillon qui ne représente pas la population visée ; (B) une erreur de mesure ; (C) un modèle trop simple.
d) Vrai ou faux : un modèle entraîné sur les manchots de l'archipel Palmer vaut pour tous les manchots du monde.
e) Appliquer ce modèle à des manchots d'une autre région, c'est un cas de : (A) décalage de distribution ; (B) overfitting ; (C) fuite de données.

<a id="papier"></a>

## ✏️ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 29.1 — Poids en mémoire de MNIST, Fashion-MNIST et CIFAR-10 ✏️ ★ ⏱️ 10 min
**Objectif :** estimer la mémoire d'un dataset d'images avant de le charger, selon son type numérique.
**Prérequis :** fiche (conventions) · 0A (types NumPy) · **Parcours :** M

Mémoire $= N \times H \times W \times C \times$ (octets par valeur).

a) MNIST (70 000 images de 28 × 28, un canal) en `uint8`, en Mo, à 2 décimales.
b) Le même en `float32`, en Mo, à 2 décimales.
c) CIFAR-10 (60 000 images de 32 × 32, trois canaux) en `uint8`, en Mo, à 2 décimales.
d) CIFAR-10 en `float32`, en Go, à 3 décimales.
e) Combien d'images MNIST en `float32` tiennent dans 1 Go ? (un entier : arrondi vers le bas)
f) Vrai ou faux : Fashion-MNIST occupe exactement la même mémoire que MNIST.

Dans ta copie : pourquoi garder les images en `uint8` sur le disque et ne les convertir en `float32` qu'au moment de former un mini-batch ?

### Ex 29.2 — Découpage stratifié et référence « classe majoritaire » ✏️ ★ ⏱️ 10 min
**Objectif :** prévoir les effectifs d'un découpage stratifié et la performance d'un modèle qui ne fait rien.
**Prérequis :** ch. 8 (stratification) · fiche « Au-delà du livre : auditer un dataset » · **Parcours :** M

Un dataset de 2 000 exemples : 1 200 de la classe A, 600 de la classe B, 200 de la classe C. On le découpe en 70 % pour l'entraînement, 15 % pour la validation et 15 % pour le test, de façon **stratifiée**.

a) Le nombre d'exemples de la classe C dans l'entraînement.
b) Le nombre d'exemples de la classe C dans le test.
c) L'accuracy, à 2 décimales, d'un modèle qui prédit toujours la classe majoritaire.
d) L'accuracy maximale, à 2 décimales, d'un modèle qui ne prédit jamais la classe C.
e) Le recall de la classe C, à 2 décimales, pour le modèle de c).
f) Vrai ou faux : dans un découpage stratifié, chaque partie a les mêmes proportions de classes que le dataset entier, à l'arrondi près.

Dans ta copie : pourquoi une accuracy de 0,85 sur ce dataset n'est-elle pas forcément bonne ? Quelle métrique regarderais-tu pour la classe C ?

### Ex 29.3 — Télécharger Open Images ou LAION : stockage et temps 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer en ordres de grandeur le coût d'un très grand dataset avant de s'y lancer.
**Prérequis :** Ex 29.1 · fiche §29.5 · **Parcours :** M

Réponds dans ta copie, avec des ordres de grandeur.

1. Open Images : environ 9 millions d'images JPEG d'en moyenne 300 Ko. Sa taille totale, en To ($10^{12}$ octets) ?
2. Le temps de le télécharger avec une connexion de 100 Mbit/s, en jours (1 octet = 8 bits).
3. Le même temps sur un serveur à 10 Gbit/s, en heures, si le débit était tenu (ce qui n'arrive presque jamais).
4. LAION-5B : environ 5,85 milliards de paires image-texte, distribuées sous forme de **liens**, pas d'images. Si l'on téléchargeait toutes les images à 100 Ko chacune, combien de To ? Pourquoi distribuer des liens plutôt que des images (pense au droit d'auteur et au poids) ? Quel problème pose un lien qui meurt, ou dont le contenu change ?
5. Décodées en `uint8` à 224 × 224 × 3, combien de To prendraient les 9 millions d'images d'Open Images ? Pourquoi garde-t-on les JPEG sur le disque ?
6. Ton projet final doit tourner sur un portable ou sur Colab : quelle taille de dataset te semble raisonnable ? Que fais-tu si le dataset idéal est trop gros ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 29.4 — Présenter son projet final en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** formuler clairement le problème, les données et la mesure de réussite de ton projet final.
**Prérequis :** fiche (toute) · **Parcours :** R

En cinq lignes au plus, présente ton projet final à quelqu'un qui ne fait pas de deep learning : (1) le problème et pour qui il compte ; (2) les données (source, taille, licence) ; (3) ce que le modèle prendra en entrée et rendra en sortie ; (4) comment tu sauras si c'est réussi (une métrique et une référence simple à battre) ; (5) un risque ou une limite connue. Cette présentation est le premier livrable du projet final.

### Ex 29.5 — Retirés, restreints ou biaisés : que vérifier avant d'utiliser un dataset ? ⚖️ ★★ ⏱️ 25 min
**Objectif :** construire et appliquer une liste de vérifications éthiques et juridiques avant d'adopter un dataset.
**Prérequis :** fiche §29.1, §29.3.1, §29.5 (encadrés 🕰️) · **Parcours :** R

Réponds dans ta copie.

1. Écris une liste de six à huit vérifications à faire avant d'utiliser un dataset trouvé en ligne (licence, provenance, consentement, données personnelles, retraits, représentativité…).
2. Applique-la à Boston Housing : qu'est-ce qui pose problème, et que propose scikit-learn à la place ?
3. MS-Celeb-1M et *80 Million Tiny Images* ont été retirés. Pour chacun : pourquoi ? Que faire d'un modèle déjà entraîné dessus ?
4. CelebA est réservé à la recherche non commerciale. Une start-up veut l'utiliser pour un filtre de photos : que lui réponds-tu ?
5. LAION-5B distribuait des liens, pas des images. Cela met-il ses utilisateurs à l'abri des problèmes de contenu et de droit d'auteur ?
6. Pour ton projet final : quelles vérifications de ta liste ne peux-tu pas faire toi-même, et comment le diras-tu dans ta data card ?

### Ex 29.6 — Lire Gebru et coll., « Datasheets for Datasets » 📄 ★★★ ⏱️ 40 min
**Objectif :** lire l'article qui a lancé la documentation systématique des datasets, et l'appliquer à un dataset du workbook.
**Prérequis :** fiche « Au-delà du livre : documenter et télécharger » · **Parcours :** M

Lis T. Gebru, J. Morgenstern, B. Vecchione, J. Wortman Vaughan, H. Wallach, H. Daumé III et K. Crawford, « Datasheets for Datasets » ([arXiv:1803.09010](https://arxiv.org/abs/1803.09010), version *Communications of the ACM*, 2021). Réponds dans ta copie.

1. Quelle analogie avec l'électronique les auteurs proposent-ils ? Que dit-elle du rôle d'une datasheet ?
2. Quelles sont les sept sections de questions ? Donne une question de chacune.
3. À qui la datasheet s'adresse-t-elle : à ceux qui créent le dataset, à ceux qui l'utilisent, aux deux ? Quels objectifs différents servent-elles ?
4. Les auteurs disent que les questions ne se remplissent pas toutes automatiquement. Pourquoi ? Donne un exemple de question qui demande une réflexion plutôt qu'une mesure.
5. Choisis un dataset du workbook (MNIST, Penguins, Verne…) et réponds à trois questions de la section « Composition » à partir de sa fiche dans `data/cards/` ou de tes recherches. Que manque-t-il à la fiche du workbook ?
6. Quelles limites de l'approche les auteurs reconnaissent-ils ?

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### 29.E1 — Comment choisir et valider un dataset pour un nouveau projet ? 💼 ★★ ⏱️ 10 min
« On vous confie un nouveau projet de classification. Comment choisissez-vous les données, et comment vous assurez-vous qu'elles conviennent ? »

### 29.E2 — Qu'est-ce qu'une data card et que doit-elle contenir ? 💼 ★★ ⏱️ 10 min
« Qu'est-ce qu'une data card (ou datasheet) ? Que mettez-vous dedans, et à quoi sert-elle ? »

### 29.E3 — Comment détecter une fuite de données entre entraînement et test ? 💼 ★★ ⏱️ 10 min
« Votre modèle obtient 99,5 % sur le test. Comment vérifiez-vous qu'il n'y a pas de fuite ? »

### 29.E4 — Données publiques et usage commercial : que vérifier ? 💼 ★★ ⏱️ 10 min
« Nous voulons utiliser un dataset public trouvé en ligne pour un produit. Que vérifiez-vous avant ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 29.7 | Les datasets intégrés de scikit-learn, et une affirmation du livre à vérifier | 📦 | ★ | 15 |
| 29.8 | Prédire : le même modèle sur MNIST puis sur Fashion-MNIST | 🔮 | ★ | 10 |
| 29.9 | Prédire la chute d'accuracy en changeant de source de chiffres | 🔮 | ★ | 10 |
| 29.10 | torchvision.datasets et Hugging Face datasets : mêmes données, deux API | 📦 | ★★ | 25 |
| 29.11 | Explorer le Hugging Face Hub et OpenML : chercher, filtrer, lire la licence | 📦 | ★★ | 25 |
| 29.12 | Profil express d'un dataset inconnu | 📈 | ★★ | 20 |
| 29.13 | Un score trop beau pour être vrai : doublons entre entraînement et test | 🐛 | ★★ | 25 |
| 29.14 | Décalage de distribution : un modèle MNIST face aux chiffres de scikit-learn | 🔬 | ★★★ | 35 |
| 29.15 | Data card et script de téléchargement vérifié pour le projet final | 🛠️ | ★★★ | 60 |
| 29.16 | Le dataset piégé : trouver les huit défauts | 🏆 | ★★★ | 60 |

Tout tourne sur CPU en quelques minutes. Les exercices 29.10 et 29.11 interrogent le Hugging Face Hub et OpenML : sans réseau, ils utilisent des résultats mis en cache et le disent.
