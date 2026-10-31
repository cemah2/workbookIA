# 29 · Datasets et préparation du projet final — fiche de cours

> Le dernier chapitre du livre est une liste : des datasets intégrés aux bibliothèques (scikit-learn, Keras), des collections en ligne, et quelques datasets « récents » de 2018. Huit ans plus tard, plusieurs liens sont morts, certains datasets ont été retirés, et les chiffres de la liste sont souvent faux. La fiche corrige la liste, puis fait de ce chapitre la **préparation du projet final** : où chercher un dataset aujourd'hui (Hugging Face Hub, Kaggle, OpenML, UCI), comment vérifier sa licence et son histoire, comment l'**auditer** (valeurs manquantes déguisées, doublons, fuites entre découpages, labels douteux, classes rares), comment mesurer un **décalage de distribution**, et comment le **documenter** (data card) et le **télécharger de façon reproductible** (révision figée, somme de contrôle SHA-256).

| | |
|---|---|
| **Livre** | vol. 2, ch. 29 « Datasets », p. 1648-1657 (§29.1 à §29.5) |
| **Temps total estimé** | ≈ 9,5 h : lecture du livre et de la fiche ≈ 1,1 h, exercices ≈ 7,9 h, 15 flashcards ≈ 0,5 h |
| **Prérequis** | 0A (pandas : `read_csv`, `info`, `describe`, `value_counts`, `isna`) · ch. 2 (échantillon et population) · ch. 8 (découpage, stratification) · ch. 12 (préparation des données, fuite de données) · ch. 15 (scikit-learn) · ch. 20 et 23 (`Dataset`, `DataLoader`, un classifieur PyTorch sur Fashion-MNIST) · ch. 28 (licences, domaine public) |
| **Fichiers du chapitre** | `02_exercices.md` (quiz, rappels, papier, réflexion, entretien) · `03_notebook.ipynb` · `04_indices.md` · `05_solutions.md` et `05_solutions.ipynb` · `06_mes_reponses.md` · `flashcards.csv` |
| **mylearn** | aucun module : les outils d'audit tiennent en quelques lignes de pandas, et le livrable est la data card de ton projet |

## Comment utiliser ce chapitre

Le chapitre du livre se lit en vingt minutes : c'est un annuaire. Lis-le pour l'idée générale (partir de l'existant, vérifier les règles, inspecter les données), puis prends dans la fiche les corrections de la liste et tout ce qui sert au projet final.

**Ordre conseillé.**
1. Lis le livre §29.1 et §29.2, puis les sections correspondantes de la fiche. Fais les quiz Q1 à Q4, le rappel R3 et l'exercice ✏️ 29.1.
2. Lis le livre §29.3 à §29.5, puis la fiche. Fais les quiz Q5 à Q8, les rappels R1 et R2 et l'estimation 🧮 29.3.
3. Lis la fin de la fiche (auditer, décalage de distribution, documenter, télécharger). Fais l'exercice ✏️ 29.2, la réflexion ⚖️ 29.5, la lecture 📄 29.6 et les entretiens.
4. Fais le notebook dans l'ordre des ID : charger, prédire, comparer deux API, chercher sur le Hub, profiler, déboguer une fuite, mesurer un décalage, puis le livrable 🛠️ 29.15 (la data card de **ton** projet) et le défi du dataset piégé. Termine par 🗣️ 29.4 : présenter ton projet en cinq lignes.

| Section | Quiz 🧠 et rappels 🔁 | Papier ✏️ 🧮 et réflexion | Notebook | Entretien 💼 |
|---|---|---|---|---|
| 29.1 Les datasets publics | Q1, Q2, Q3, R1, R2, R3 | 29.2, 29.4, 29.5, 29.6 | 29.12, 29.13, 29.15, 29.16 | E1, E3, E4 |
| 29.2 MNIST et Fashion-MNIST | Q4 | 29.1 | 29.8, 29.9, 29.10, 29.14 | |
| 29.3 et 29.3.1 Les datasets de scikit-learn | Q5 | 29.5 | 29.7 | |
| 29.3.2 Les datasets de Keras | Q6 | 29.1 | 29.10 | |
| 29.4 Les collections | Q7 | 29.6 | 29.11, 29.15 | E2 |
| 29.5 Des datasets plus récents | Q8 | 29.3, 29.5 | 29.9, 29.11, 29.14 | E4 |
| Au-delà : auditer, décalage, documenter, télécharger | | 29.2, 29.6 | 29.12 à 29.16 | E1 à E4 |

## Objectifs d'apprentissage

À la fin de ce chapitre, tu sais :
1. **Charger** des datasets avec scikit-learn, torchvision, Hugging Face `datasets` et OpenML.
2. **Évaluer** un dataset avant de l'utiliser : licence, provenance, retraits, biais.
3. **Auditer** sa qualité : valeurs manquantes (même déguisées), doublons, fuites entre découpages, labels suspects, unités incohérentes, classes rares.
4. **Mesurer** l'effet d'un décalage de distribution sur un modèle.
5. **Rédiger** une data card et un script de téléchargement reproductible (révision figée, SHA-256).
6. **Choisir** et **présenter** le dataset de ton projet final.

## L'essentiel en 10 lignes

1. Avant d'entraîner de zéro, cherche ce qui existe : un modèle pré-entraîné à ajuster (transfer learning, bonus B1), sinon un dataset public, et seulement ensuite une collecte.
2. Un dataset public n'est pas un dataset libre : lis sa **licence** (usage commercial ? attribution ? recherche seulement ?), son histoire (a-t-il été retiré ?) et ce qu'il contient (données personnelles, œuvres protégées).
3. **Inspecte** toujours les données : des champs vides, des codes comme −999 qui jouent les valeurs manquantes, des doublons, des labels mal orthographiés, des unités mélangées, des images manquantes.
4. MNIST est propre et facile ; Fashion-MNIST a exactement le même format et il est plus difficile : c'est le benchmark suivant.
5. Les bibliothèques fournissent des datasets prêts à charger : `sklearn.datasets`, `torchvision.datasets`, Hugging Face `datasets` ; vérifie leurs vraies tailles, la liste du livre se trompe souvent.
6. Les grandes collections d'aujourd'hui : Hugging Face Hub, Kaggle, OpenML, le dépôt UCI ; chaque dataset y a une page (sa *dataset card*) qui donne licence, taille et usage prévu.
7. Une **fuite** entre entraînement et test (doublons, quasi-doublons, une colonne qui contient la réponse) donne un score trop beau pour être vrai ; on la cherche avant de célébrer.
8. Un modèle excellent sur son dataset peut s'effondrer sur des données d'une autre source : c'est le **décalage de distribution** ; on le mesure sur un jeu de test qui vient d'ailleurs.
9. Une **data card** (ou *datasheet*) dit d'où viennent les données, ce qu'elles contiennent, à quoi elles servent et à quoi pas, leurs biais et leur licence.
10. Un téléchargement **reproductible** fixe une révision (un commit, une version) et vérifie une **somme de contrôle** (SHA-256) : le même script donne les mêmes octets, dans un an comme aujourd'hui.

> 📐 **Conventions de la fiche** — Un *dataset* est un ensemble d'exemples ; on le découpe en entraînement, validation et test (ch. 8). $N$ est le nombre d'exemples, $n_c$ celui de la classe $c$. Les tailles en octets suivent le SI : 1 Mo = $10^6$ octets, 1 Go = $10^9$ octets ; un `uint8` occupe 1 octet, un `float32` 4 octets.

---

## 29.1 · Les datasets publics ⏩

Entraîner un grand réseau coûte du temps et du calcul. Le livre conseille donc de partir de l'existant : un système déjà entraîné, éventuellement ajusté sur nos données (le transfer learning du bonus B1), et, quand ce n'est pas possible, un entraînement complet sur un dataset choisi. Il insiste sur trois points : **lire les règles** attachées à un dataset (beaucoup de bases de visages sont réservées à la recherche, sur demande ; d'autres contiennent des données protégées), et **inspecter** les données à la main avant d'investir du temps (des fautes de frappe, des cases laissées vides, des classes qui n'ont qu'un seul exemple, des fichiers ou des trames vidéo qui manquent), et se souvenir qu'une liste de liens vieillit vite : des datasets naissent, d'autres disparaissent.

Les trois ont pris de l'importance depuis 2018. Côté droit : une licence dit ce qu'on peut faire des données (CC BY : tout, avec attribution ; CC BY-NC : pas d'usage commercial ; « recherche seulement » : ni produit ni service) ; les données personnelles relèvent du RGPD ; les œuvres protégées, du droit d'auteur (ch. 28). Côté histoire : plusieurs datasets célèbres ont été **retirés** (encadré 🕰️ du §29.5). Côté qualité : les audits systématiques de grands datasets ont trouvé des erreurs de labels jusque dans les jeux de test de référence (C. Northcutt et coll., « Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks », NeurIPS 2021, [arXiv:2103.14749](https://arxiv.org/abs/2103.14749) : au moins 3,3 % d'erreurs en moyenne sur dix jeux de test).

> ⚠️ **Le livre, corrigé — représentatif de quoi ?** — Le livre parle de vérifier que les données sont « assez propres ». Il faut aussi qu'elles soient **représentatives** de l'usage visé (ch. 2, biais d'échantillonnage) : des photos de visages prises surtout en studio, des textes surtout anglais, des chiffres écrits par des employés américains ne représentent pas tous les utilisateurs. Un modèle hérite des absences de son dataset ; une data card doit le dire.

## 29.2 · MNIST et Fashion-MNIST ⏩

MNIST sert d'exemple dans tout le livre. Ses chiffres sont centrés, nets, sans fond : un modèle à 99 % sur MNIST ne dit presque rien de ce qu'il ferait sur des données du monde réel. **Fashion-MNIST** (H. Xiao, K. Rasul, R. Vollgraf, 2017, Zalando) garde exactement le même format (70 000 images en niveaux de gris de 28 × 28, 10 classes, 60 000 pour l'entraînement et 10 000 pour le test) avec des vêtements, des chaussures et des sacs. Le livre le propose comme marche suivante : on change une ligne de code, la tâche devient plus dure, et l'on peut encore regarder les erreurs une à une (🔮 29.8).

> 🕰️ **Mise à jour (2026) — où trouver MNIST** — **Le livre :** le site de Yann LeCun. · **Aujourd'hui :** le site d'origine est souvent inaccessible ou refuse les téléchargements automatiques ; `torchvision.datasets.MNIST` passe par des miroirs, Hugging Face héberge `ylecun/mnist`, et le workbook garde sa propre copie versionnée (`data/mnist.npz`, `wb.datasets.load_mnist`). Pour Fashion-MNIST, le dépôt GitHub de Zalando reste la source (licence MIT). · **Faut-il quand même l'apprendre ?** Oui : MNIST reste le test minimal d'un code, et la leçon est générale : un projet ne doit pas dépendre d'une seule URL. · *Source :* [code de `torchvision.datasets.MNIST`](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/datasets/mnist.py) ; [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist).

## 29.3 · Les datasets des bibliothèques ⏩

Beaucoup de bibliothèques fournissent des datasets prêts à charger, en une ligne : c'est la façon la plus rapide de tester une idée, avec un format connu et une licence en général claire.

### 29.3.1 · scikit-learn ⏩

`sklearn.datasets` offre trois familles : de petits datasets livrés avec la bibliothèque (`load_iris`, `load_digits`, `load_diabetes`, `load_breast_cancer`, `load_wine`, `load_linnerud`), des datasets téléchargés à la demande et mis en cache (`fetch_california_housing`, `fetch_20newsgroups`, `fetch_covtype`, `fetch_lfw_people`, `fetch_olivetti_faces`, `fetch_rcv1`), et des générateurs (`make_classification`, `make_moons`…). Chacun rend un objet `Bunch` (`.data`, `.target`, `.feature_names`, `.DESCR`), ou un DataFrame avec `as_frame=True`. Et `fetch_openml(name=..., version=...)` donne accès aux milliers de datasets d'OpenML (📦 29.11).

> ⚠️ **Le livre, corrigé — les chiffres de la liste de scikit-learn** — Plusieurs descriptions sont fausses. **Digits** : `load_digits` contient **1 797** images de 8 × 8 (le dataset complet de l'UCI en a 5 620, mais scikit-learn n'en livre qu'une partie ; 📦 29.7). **Boston** : 506 quartiers de recensement, pas 500 maisons (et le dataset a été retiré, encadré 🕰️). **Olivetti** : 10 images de chacune des 40 personnes, soit **400** images. **Labeled Faces in the Wild** : 13 233 images, rendues en **niveaux de gris** par défaut et réduites (62 × 47 avec les réglages par défaut). Le livre décrit aussi comme du « texte » des datasets **numériques** (Diabetes, Iris, Linnerud, Covertype, Breast Cancer) : ce sont des tableaux de nombres ; seuls 20 Newsgroups et RCV1 sont des textes (RCV1 est fourni sous forme de vecteurs TF-IDF, 804 414 documents).

> 🕰️ **Mise à jour (2026) — Boston Housing** — **Le livre :** listé parmi les datasets de scikit-learn et de Keras. · **Aujourd'hui :** `load_boston` a été déprécié dans scikit-learn 1.0, puis **retiré** dans la version 1.2 : une de ses variables (`B`) a été construite à partir de la proportion de population noire d'un quartier, sous l'hypothèse d'un effet sur les prix ; la documentation renvoie vers California Housing ou Ames Housing. Keras le garde, avec un avertissement. Le workbook utilise California Housing depuis les premiers chapitres. · **Faut-il quand même l'apprendre ?** Oui, comme leçon : un dataset classique peut encoder un choix contestable ; lire la description des variables fait partie de l'audit. · *Source :* [notes de version de scikit-learn 1.0](https://scikit-learn.org/1.6/whats_new/v1.0.html) (dépréciation, retrait annoncé pour la 1.2) ; [documentation de `load_boston`, scikit-learn 1.1](https://scikit-learn.org/1.1/modules/generated/sklearn.datasets.load_boston.html).

### 29.3.2 · Keras ⏩

Keras fournit `keras.datasets` : MNIST, Fashion-MNIST, CIFAR-10 et CIFAR-100, IMDB (critiques de films, sentiment), Reuters (dépêches, 46 sujets), Boston Housing et, depuis, California Housing. Chaque fonction `load_data()` rend deux couples `(x_train, y_train), (x_test, y_test)` de tableaux NumPy.

> ⚠️ **Le livre, corrigé — les tailles de la liste de Keras** — Le livre ne compte souvent que le jeu d'**entraînement**. CIFAR-10 et CIFAR-100 ont 60 000 images chacun (50 000 + 10 000 de test) ; MNIST et Fashion-MNIST 70 000 (60 000 + 10 000) ; IMDB 50 000 critiques (25 000 + 25 000) ; Reuters 11 228 dépêches en tout (le « 11 000 » du livre est ce total arrondi ; `load_data` en garde 20 % pour le test). Et l'intérêt d'IMDB et de Reuters dans Keras est qu'ils sont déjà **découpés en mots et encodés** en entiers : pratique, mais on y perd le texte brut.

> 🕰️ **Mise à jour (2026) — charger des données en PyTorch** — **Le livre :** `keras.datasets`. · **Aujourd'hui :** Keras 3 garde `keras.datasets` ; en PyTorch, `torchvision.datasets` (images : MNIST, FashionMNIST, CIFAR10, ImageFolder pour un dossier à soi) rend des objets `Dataset` à passer à un `DataLoader` (ch. 20 et 23), avec des `transforms` ; la bibliothèque Hugging Face `datasets` charge les datasets du Hub (plusieurs centaines de milliers) par `load_dataset("nom", revision=...)`, au format Arrow (mémoire projetée, rapide), avec un mode *streaming* pour les datasets trop gros pour le disque. · **Faut-il quand même l'apprendre ?** Oui : les trois API se ressemblent (un nom, un découpage, des transformations), et passer de l'une à l'autre est un geste quotidien (📦 29.10). · *Source :* [documentation de `torchvision.datasets`](https://docs.pytorch.org/vision/0.26/datasets.html) ; [documentation de Hugging Face `datasets`](https://huggingface.co/docs/datasets).

> 🧮 **Rappel outil — Hugging Face `datasets` et `huggingface_hub`** — `from datasets import load_dataset` ; `ds = load_dataset("zalando-datasets/fashion_mnist", revision="<sha du commit>")` rend un `DatasetDict` (`ds["train"]`, `ds["test"]`) ; `ds["train"][0]` est un dictionnaire (`{"image": <PIL.Image>, "label": 9}`) ; `ds["train"].features` décrit les colonnes (`ClassLabel` donne les noms des classes) ; `ds["train"].with_format("numpy")` ou `.to_pandas()` convertit. `revision=` fixe une version précise du dataset (un commit de son dépôt sur le Hub) : sans elle, on charge la dernière, qui peut changer. Pour **chercher** sans télécharger : `from huggingface_hub import HfApi` ; `HfApi().list_datasets(search="mnist", limit=10)` ; `HfApi().dataset_info(nom)` donne le `sha` de la dernière révision, la licence (`card_data["license"]`) et la liste des fichiers. Le téléchargement va dans un cache (`~/.cache/huggingface`) et n'est pas refait.

## 29.4 · Les collections de datasets ⏩

Le livre liste des sites qui recensent des datasets : des listes sur GitHub (*Awesome Deep Learning*), deeplearning.net, DeepMind, Kaggle (« plus de 1 000 datasets »), deeplearning4j, StatLib, la *Time Series Data Library* de DataMarket, le dépôt UCI (« environ 400 »), une page de Wikipédia.

> 🕰️ **Mise à jour (2026) — les collections** — **Le livre :** neuf collections de 2018, de 7 à 1 000 datasets chacune. · **Aujourd'hui :** plusieurs liens sont morts ou redirigés (deeplearning.net, DataMarket, la page de deeplearning4j, l'ancienne page de DeepMind). Les grandes sources sont le **Hugging Face Hub** (plusieurs centaines de milliers de datasets, chacun avec sa *dataset card*, sa licence et son historique de versions), **Kaggle** (des centaines de milliers aussi ; l'outil en ligne de commande `kaggle` demande un compte et une clé d'API personnelle ; la bibliothèque plus récente `kagglehub` télécharge beaucoup de datasets publics sans compte, mais pas ceux dont il faut accepter les règles), **OpenML** (des milliers de datasets tabulaires avec des tâches et des résultats comparables, chargeables par `sklearn.datasets.fetch_openml`), et le **dépôt UCI**, refait à neuf (archive.ics.uci.edu), avec plus de 600 datasets. *Papers with Code*, qui reliait datasets, tâches et articles, a fermé en 2025. · **Faut-il quand même l'apprendre ?** Oui : savoir **où** chercher, puis comment juger une page de dataset, est la première étape d'un projet (📦 29.11). · *Source :* [huggingface.co/datasets](https://huggingface.co/datasets) ; [kaggle.com/datasets](https://www.kaggle.com/datasets) ; [openml.org](https://www.openml.org) ; [archive.ics.uci.edu](https://archive.ics.uci.edu).

## 29.5 · Des datasets plus récents ⏩

Le livre termine par des datasets « nouveaux » en 2018 : AVA (actions dans des films), Celebrity in Places, COCO (objets en contexte), le Cornell Movie-Dialogs Corpus, Google Speech Commands, Kinetics (actions dans des vidéos), CelebA (visages de célébrités annotés), LSUN (scènes), Matterport3D (maisons en 3D), le Million Song Dataset, MSRA-CFW (visages de célébrités trouvés sur le web), NSynth (notes de musique), Open Images, Quick, Draw!, Quora Question Pairs, des images de Street View, YouTube-BoundingBoxes. Beaucoup sont devenus des références ; certains sont énormes (Open Images : 9 millions d'images ; 🧮 29.3).

> 🕰️ **Mise à jour (2026) — des datasets retirés ou restreints** — **Le livre :** CelebA listé sans réserve, MSRA-CFW avec la seule mention « protégé par le droit d'auteur ». · **Aujourd'hui :** plusieurs grands datasets de personnes ont été retirés : MS-Celeb-1M (Microsoft, 10 millions de photos de 100 000 personnes, sans leur consentement) en 2019 ; *80 Million Tiny Images* (MIT) en 2020, après la découverte de labels racistes et injurieux (A. Birhane et V. Prabhu, « Large image datasets: A pyrrhic win for computer vision? », WACV 2021) ; LAION-5B (des milliards de paires image-texte) retiré fin 2023 après la détection de liens vers des contenus pédocriminels, puis republié nettoyé en 2024 (Re-LAION-5B). CelebA reste disponible, mais pour la recherche non commerciale seulement. Un dataset retiré continue souvent de circuler sur des miroirs : l'utiliser reste un problème. · **Faut-il quand même l'apprendre ?** Oui : vérifier qu'un dataset n'a pas été retiré, et pourquoi, fait partie du choix (⚖️ 29.5). · *Source :* [Birhane et Prabhu, arXiv:2006.16923](https://arxiv.org/abs/2006.16923) ; annonces de LAION (2023-2024) ; [page de CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

> ⚠️ **Le livre, corrigé — quelques chiffres** — COCO compte environ 330 000 images, dont plus de 200 000 annotées (le livre : 200 000) ; CelebA, 202 599 images de 10 177 personnes (le livre : « environ 10 000 célébrités », juste) ; Speech Commands est passé de 65 000 à 105 829 enregistrements de 35 mots dans sa version 2 (2018). Les tailles d'un dataset changent d'une version à l'autre : on cite toujours la version.

## Au-delà du livre : auditer un dataset

Un audit répond à une question : **ces données peuvent-elles entraîner et évaluer honnêtement mon modèle ?** On le fait avant toute modélisation, avec pandas (ch. 0A, ch. 12) :

| Défaut | Comment le voir | Que faire |
|---|---|---|
| valeurs manquantes | `df.isna().sum()` ; mais aussi des **codes** : −999, 0, `"?"`, `"NA"`, une date en 1900 | les recoder en `NaN` (`df.replace(-999, np.nan)`), puis décider : imputer (ch. 12) ou retirer |
| doublons exacts | `df.duplicated().sum()` | les retirer avant le découpage |
| quasi-doublons entre découpages | mêmes valeurs à un détail près (un identifiant, un bruit infime, une image recadrée) ; pour des images, un hachage des pixels ou des features | découper **par groupe** (un objet, une personne, un patient : tous ses exemples du même côté) |
| colonne constante | `df.nunique() == 1` | la retirer : elle n'apporte rien |
| fuite de la cible | une colonne qui contient la réponse (un code, une date postérieure, un identifiant trié) ; une accuracy de 100 % | la retirer ; se demander ce qu'on connaîtra **au moment de prédire** |
| unités incohérentes | `describe()` : des valeurs dix fois trop grandes, une distribution à deux bosses | convertir, en notant la règle |
| labels douteux | `value_counts()` : fautes, majuscules, espaces ; des exemples mal classés | normaliser, corriger, ou écarter |
| classe très rare | `value_counts(normalize=True)` | découpage stratifié (ch. 8), métrique adaptée (recall, F1), et une baseline « classe majoritaire » |

Le 📈 29.12 fait ce profil sur un dataset inconnu, le 🐛 29.13 traque une fuite par doublons, et le 🏆 29.16 te donne un dataset où **huit** défauts ont été plantés.

## Au-delà du livre : le décalage de distribution

Un modèle apprend la distribution de son dataset. Sur des données d'une autre source, même pour la même tâche, ses performances peuvent chuter : c'est le **décalage de distribution** (*distribution shift*). On distingue le décalage des entrées (*covariate shift* : des chiffres écrits plus gros, une autre caméra, un autre hôpital), celui des classes (*label shift* : une maladie plus fréquente ici que là) et celui de la relation entre les deux (*concept shift* : le même mot change de sens). Le seul moyen de le voir est un **jeu de test qui vient d'ailleurs** ; le 🔮 29.9 et le 🔬 29.14 entraînent un modèle sur MNIST et le testent sur les chiffres de scikit-learn, redimensionnés, puis cherchent à réduire l'écart par le prétraitement.

## Au-delà du livre : documenter et télécharger

> 🕰️ **Mise à jour (2026) — documenter un dataset** — **Le livre :** absent. · **Aujourd'hui :** la *datasheet* (T. Gebru et coll., « Datasheets for Datasets », 2018, [arXiv:1803.09010](https://arxiv.org/abs/1803.09010), version *Communications of the ACM* 2021) pose une série de questions : motivation, composition, collecte, prétraitement, usages, distribution, maintenance ; les *dataset cards* du Hugging Face Hub en sont la forme courante ; le format **Croissant** (MLCommons, 2024) décrit les métadonnées d'un dataset de façon lisible par machine. Pour les modèles, le pendant est la fiche modèle (M. Mitchell et coll., 2019 ; ch. 8 et 23). Le règlement européen sur l'IA demande aux fournisseurs de modèles d'IA à usage général un résumé suffisamment détaillé des contenus d'entraînement (art. 53, applicable depuis le 2 août 2025, sur un modèle publié par le Bureau de l'IA en juillet 2025 ; jusqu'au 2 août 2027 pour les modèles mis sur le marché avant). · **Faut-il quand même l'apprendre ?** Oui : la data card de ton projet final est un livrable (🛠️ 29.15), sur le modèle des fiches de `data/cards/` du workbook. · *Source :* Gebru et coll., cité ; [mlcommons.org/croissant](https://mlcommons.org/working-groups/data/croissant/) ; [règlement (UE) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj).

Une data card répond, en une page, à : **d'où** viennent les données (source, date, révision) ; **quoi** (exemples, variables, labels, tailles, découpages) ; **sous quelles conditions** (licence, données personnelles, retraits) ; **pour quoi** (usage prévu, usages déconseillés) ; **avec quelles limites** (biais, défauts connus, ce qui manque) ; et **comment l'obtenir à l'identique** (le script de téléchargement).

> 🧮 **Rappel outil — une somme de contrôle SHA-256** — Une fonction de hachage transforme un fichier, de n'importe quelle taille, en une empreinte de taille fixe ; SHA-256 donne 256 bits, écrits en 64 chiffres hexadécimaux. Changer un seul octet du fichier change complètement l'empreinte, et l'on ne sait pas fabriquer deux fichiers différents de même empreinte. En Python : `import hashlib` ; `hashlib.sha256(Path(f).read_bytes()).hexdigest()` pour un petit fichier ; pour un gros, lis-le par morceaux (`h = hashlib.sha256()` puis `h.update(chunk)` dans une boucle, et `h.hexdigest()`). Un script de téléchargement reproductible fixe l'**adresse** et la **révision** (un tag, un commit, une version), télécharge dans un fichier temporaire, compare l'empreinte à celle qui est écrite dans le script, et ne garde le fichier que si elles coïncident (🛠️ 29.15).

> 💼 **En entreprise** — Le choix des données décide souvent plus du résultat que le choix du modèle. Une équipe sérieuse tient un registre de ses datasets (source, licence, version, empreinte, data card), vérifie les droits avant tout usage commercial avec son service juridique, découpe par groupe pour éviter les fuites, et garde un jeu de test « d'ailleurs » pour surveiller le décalage de distribution après le déploiement.

## Les pièges classiques ⚠️ (récapitulatif)

| Piège | Exemple | Ce qu'il faut faire |
|---|---|---|
| Croire la taille annoncée | « 5 620 images » pour `load_digits`, qui en a 1 797 | vérifier sur les données chargées : `X.shape` |
| Prendre « public » pour « libre » | un dataset « recherche seulement » utilisé dans un produit | lire la licence et les conditions avant de commencer |
| Utiliser un dataset retiré | une copie de MS-Celeb-1M trouvée sur un miroir | chercher l'histoire du dataset ; préférer une version officielle maintenue |
| Rater des manquants déguisés | `-999` dans une colonne de mesures : la moyenne est faussée | `describe()`, `value_counts()` ; recoder en `NaN` |
| Découper après avoir dupliqué | des images augmentées ou recadrées des deux côtés du découpage | dédupliquer, puis découper par groupe ; augmenter seulement l'entraînement |
| Garder une colonne qui contient la réponse | un code de qualité qui encode la classe : 100 % d'accuracy | se demander ce qu'on connaîtra au moment de prédire |
| Évaluer seulement sur la même source | 99 % sur MNIST, beaucoup moins sur d'autres chiffres | un jeu de test d'une autre source |
| Un téléchargement non figé | la nouvelle version du dataset change les résultats | `revision=` ; une somme de contrôle SHA-256 |

## Liens avec les autres chapitres 🔗

- **Ch. 2 et 8** : échantillon et population, biais d'échantillonnage ; découpages stratifiés et jeu de test intouché.
- **Ch. 12** : préparation des données, valeurs manquantes, fuite de données dans un pipeline.
- **Ch. 15** : les datasets de scikit-learn et `fetch_openml`.
- **Ch. 20 et 23** : `Dataset`, `DataLoader`, `torchvision.datasets` ; une boucle d'entraînement comme celle de `train_mlp`, le petit réseau de 29.8, 29.9 et 29.14.
- **Ch. 28** : licences, droit d'auteur, domaine public, crédits ; la data card des images du workbook.
- **Projet final** : 🛠️ 29.15 (data card et script de téléchargement) et 🗣️ 29.4 (présentation) en sont les deux premiers livrables.

## Guide de lecture et de travail

Le chapitre du livre est un annuaire de dix pages, sans figure : lis-le en entier (c'est tout le parcours rapide, avec la fiche, soit environ 1,1 h), en gardant à l'esprit que ses chiffres et ses liens datent de 2018. Le travail est dans le notebook : chaque exercice prépare ton projet final, jusqu'à la data card de 29.15.

## Pour aller plus loin

- T. Gebru et coll., « Datasheets for Datasets », [arXiv:1803.09010](https://arxiv.org/abs/1803.09010) : les questions à se poser sur un dataset (📄 29.6).
- M. Mitchell et coll., « Model Cards for Model Reporting », FAT* 2019, [arXiv:1810.03993](https://arxiv.org/abs/1810.03993) : le pendant pour les modèles.
- C. Northcutt, A. Athalye, J. Mueller, « Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks », NeurIPS 2021 Datasets and Benchmarks, [arXiv:2103.14749](https://arxiv.org/abs/2103.14749) : des erreurs de labels jusque dans les jeux de test de référence.
- La documentation de [Hugging Face `datasets`](https://huggingface.co/docs/datasets) et la page [*Dataset Cards*](https://huggingface.co/docs/hub/datasets-cards) du Hub.
- P. W. Koh et coll., « WILDS: A Benchmark of in-the-Wild Distribution Shifts », ICML 2021, [arXiv:2012.07421](https://arxiv.org/abs/2012.07421) : des décalages de distribution réels, mesurés.
