# 29 · Datasets et préparation du projet final — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 29.Q1 — Partir de l'existant : modèle pré-entraîné, dataset public ou collecte ?

<details><summary>Indice 1</summary>

Relis le §29.1 de la fiche.

</details>
<details><summary>Indice 2</summary>

Classe les trois voies par coût : réutiliser un réseau entraîné, entraîner sur des données déjà publiées, collecter et annoter soi-même. Pour d), demande-toi ce qui décide de l'usage permis d'un dataset : le fait qu'on puisse le télécharger, ou sa licence ?

</details>
<details><summary>Indice 3</summary>

a) **B** : du moins cher au plus cher, la collecte en dernier. Pour b), rapproche la phrase de la définition du transfer learning (bonus B1). Pour c), écarte les options qui affirment un « toujours » ou un « jamais » faux. Pour e), que faut-il faire avant d'investir des heures dans un dataset ?

</details>

### 29.Q2 — Licences et restrictions d'usage : vrai ou faux

<details><summary>Indice 1</summary>

Relis le deuxième paragraphe du §29.1 de la fiche (les trois points du livre, vus d'aujourd'hui).

</details>
<details><summary>Indice 2</summary>

Décode les sigles : BY = citer l'auteur, NC = pas d'usage commercial. Une licence règle ce que **celui qui publie** permet ; elle ne dit rien des droits des personnes qui figurent dans les données.

</details>
<details><summary>Indice 3</summary>

a) **Vrai** : CC BY permet tout usage, à condition de citer. Pour b) et c), demande-toi si un produit vendu, ou un prototype destiné à l'être, est de la recherche non commerciale. Pour d) et e), sépare la licence du dataset et les droits des personnes (consentement, RGPD).

</details>

### 29.Q3 — Les défauts qu'une inspection manuelle révèle

<details><summary>Indice 1</summary>

Relis le §29.1 et le tableau de « Au-delà du livre : auditer un dataset ».

</details>
<details><summary>Indice 2</summary>

Pour a), un défaut de données se voit **dans les données** ; un réseau mal initialisé n'en fait pas partie. Pour c), demande-toi ce que pandas considère comme manquant : une valeur spéciale (`NaN`, `None`) ou n'importe quel nombre bizarre ?

</details>
<details><summary>Indice 3</summary>

a) **ABC** : champs vides, labels isolés, fautes d'orthographe. Pour b), un poids de fruit peut-il être négatif ? Pour d), quelle commande montre le minimum, le maximum et les quartiles d'une colonne ? Pour e), relis la référence à Northcutt et coll. dans le §29.1.

</details>

### 29.Q4 — MNIST et Fashion-MNIST : même format, difficulté différente

<details><summary>Indice 1</summary>

Relis le §29.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Fashion-MNIST reprend **tout** le format de MNIST : nombre d'images, taille, nombre de classes, découpage. Ce qui change, c'est ce que représentent les images.

</details>
<details><summary>Indice 3</summary>

a) **70 000** : 60 000 pour l'entraînement et 10 000 pour le test. Pour b) et c), MNIST : des images de combien de pixels de côté, et combien de chiffres ? Pour e), pense à trois vêtements du haut du corps.

</details>

### 29.Q5 — Les datasets de scikit-learn : lequel pour quelle tâche, et pourquoi Boston a disparu

<details><summary>Indice 1</summary>

Relis le §29.3.1 et ses deux encadrés (⚠️ et 🕰️).

</details>
<details><summary>Indice 2</summary>

Le livre donne un nombre pour Digits, la fiche le corrige d'après ce que rend vraiment `load_digits`. Pour Boston, cherche la version de scikit-learn et la raison du retrait dans l'encadré 🕰️.

</details>
<details><summary>Indice 3</summary>

a) **1 797** (le livre dit 5 620, le dataset complet de l'UCI). Pour b), déprécié dans une version, retiré deux versions plus tard. Pour c), quelle variable de Boston pose problème ? Pour d) et e), demande-toi ce que contient chaque dataset : des fleurs, des maisons, des tumeurs, des mesures médicales ?

</details>

### 29.Q6 — Des datasets de Keras à torchvision et Hugging Face

<details><summary>Indice 1</summary>

Relis le §29.3.2, ses encadrés, et l'encadré 🧮 sur Hugging Face `datasets`.

</details>
<details><summary>Indice 2</summary>

Chaque `load_data()` de Keras rend l'entraînement et le test, sous forme de tableaux NumPy. Pour b), le livre ne compte souvent que l'entraînement : ajoute le test.

</details>
<details><summary>Indice 3</summary>

a) **C** : deux couples `(x_train, y_train), (x_test, y_test)`. Pour b), relis l'encadré ⚠️ du §29.3.2 : le livre compte-t-il le test ? Pour c), relis la fin de l'encadré ⚠️ du §29.3.2. Pour d), quel module de PyTorch contient les datasets d'images ? Pour e), à quoi correspond une révision sur le Hub ?

</details>

### 29.Q7 — Collections de datasets : UCI, Kaggle, OpenML, Hugging Face Hub et liens morts

<details><summary>Indice 1</summary>

Relis le §29.4 et son encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

L'encadré cite des liens morts et les grandes sources d'aujourd'hui. Pour b), l'encadré dit ce que demande l'outil en ligne de commande `kaggle`.

</details>
<details><summary>Indice 3</summary>

a) **ABC** : le Hub, Kaggle et OpenML ; deeplearning.net est mort. Pour b), une clé d'API personnelle est liée à quoi ? Pour c), quelle bibliothèque du workbook a une fonction `fetch_openml` ? Pour d), l'encadré cite-t-il des liens morts ? Pour e), relis ce que l'encadré 🕰️ du §29.4 dit de chaque dataset du Hub.

</details>

### 29.Q8 — Les « nouveaux » datasets de 2018 : tâches, tailles et retraits

<details><summary>Indice 1</summary>

Relis le §29.5 et ses deux encadrés.

</details>
<details><summary>Indice 2</summary>

L'encadré 🕰️ du §29.5 nomme les datasets retirés, l'année et la raison, et ce qu'est devenu LAION-5B. La licence de CelebA y est aussi.

</details>
<details><summary>Indice 3</summary>

a) **9** (millions d'images). Pour b), cherche les deux datasets de l'encadré qui ne sont plus distribués par leurs auteurs. Pour c), relis la fin du paragraphe sur LAION-5B dans l'encadré. Pour e), une copie efface-t-elle les raisons du retrait ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 29.R1 — Ch. 28 : qui possède les droits d'une image de style ?

<details><summary>Indice 1</summary>

Relis la fiche du ch. 28 (encadré sur le domaine public) et `data/cards/images.md`.

</details>
<details><summary>Indice 2</summary>

Dans l'Union européenne, une œuvre entre dans le domaine public le 1ᵉʳ janvier qui suit les 70 ans de la mort de son auteur. Le droit d'auteur protège une œuvre précise, pas une manière de peindre.

</details>
<details><summary>Indice 3</summary>

a) Picasso meurt en 1973 : 1973 + 70 = 2043, et l'œuvre entre dans le domaine public le 1ᵉʳ janvier suivant, soit **2044**. Applique la même règle à e). Pour b), relis ce que dit la directive de 2019 d'une reproduction fidèle. Pour d), la fiche des images du ch. 28 donne la licence des photos de scikit-learn.

</details>

### 29.R2 — Ch. 26 : en RL, d'où viennent les données ?

<details><summary>Indice 1</summary>

Relis la fiche du ch. 26 (la boucle agent-environnement, la mémoire de rejeu).

</details>
<details><summary>Indice 2</summary>

En RL, personne n'annote d'exemples : l'agent agit, l'environnement répond. Ce que l'agent voit dépend donc de ce qu'il fait.

</details>
<details><summary>Indice 3</summary>

a) **B** : des interactions de l'agent avec son environnement. Pour b) et e), si la politique change, les états visités changent-ils ? Pour c), relis la fiche du ch. 26 : que faut-il garder d'un pas de l'agent pour calculer une cible de Q-learning ? Pour d), compte les cases de la grille.

</details>

### 29.R3 — Ch. 2 : biais d'échantillonnage : un dataset représente-t-il la population visée ?

<details><summary>Indice 1</summary>

Relis la fiche de Palmer Penguins (`data/cards/penguins.md`) et l'encadré ⚠️ du §29.1.

</details>
<details><summary>Indice 2</summary>

Un échantillon représente la population **d'où il vient**. Pour e), relis « Au-delà du livre : le décalage de distribution ».

</details>
<details><summary>Indice 3</summary>

a) **344** manchots, lignes incomplètes comprises. Pour b), compte les espèces de la fiche. Pour c), un biais d'échantillonnage porte sur le choix des exemples. Pour d) et e), d'où viennent les manchots du dataset, et que se passe-t-il quand on applique le modèle ailleurs ?

</details>

<a id="papier"></a>

## ✏️ 🧮 Papier-crayon

### Ex 29.1 — Poids en mémoire de MNIST, Fashion-MNIST et CIFAR-10 ✏️

<details><summary>Indice 1</summary>

Relis les conventions de l'énoncé : 1 Mo = $10^6$ octets, `uint8` = 1 octet, `float32` = 4 octets.

</details>
<details><summary>Indice 2</summary>

Mémoire $= N \times H \times W \times C \times$ (octets par valeur). Compte bien $C$ : un canal pour MNIST, trois pour CIFAR-10. N'arrondis qu'à la fin.

</details>
<details><summary>Indice 3</summary>

a) $70\,000 \times 28 \times 28 \times 1 \times 1 = 54\,880\,000$ octets, soit **54,88 Mo**. Pour b), quatre octets par valeur ; pour c) et d), $32 \times 32 \times 3$ valeurs par image (d en Go : divise par $10^9$). Pour e), divise $10^9$ par la taille d'une image en `float32`, et garde la partie entière.

</details>

### Ex 29.2 — Découpage stratifié et référence « classe majoritaire » ✏️

<details><summary>Indice 1</summary>

Relis le tableau des défauts (ligne « classe très rare ») et le ch. 8 sur la stratification.

</details>
<details><summary>Indice 2</summary>

Stratifier, c'est appliquer les pourcentages (70, 15, 15) **à chaque classe** séparément. Un modèle qui prédit toujours la même classe a juste sur tous les exemples de cette classe, et faux sur tous les autres.

</details>
<details><summary>Indice 3</summary>

a) $0{,}70 \times 200 = $ **140** exemples de C dans l'entraînement. Pour b), même calcul avec 15 %. Pour c), quelle part du total représente la classe la plus nombreuse ? Pour d), au mieux, le modèle a juste sur tout ce qui n'est pas C. Pour e), le recall de C compte les C trouvés parmi les vrais C.

</details>

### Ex 29.3 — Télécharger Open Images ou LAION : stockage et temps 🧮

<details><summary>Indice 1</summary>

Écris toutes les tailles en octets et tous les débits en octets par seconde avant de diviser.

</details>
<details><summary>Indice 2</summary>

100 Mbit/s = $10^8 / 8 = 1{,}25 \times 10^7$ octets par seconde. Une journée = 86 400 s. Pour 5, une image décodée de 224 × 224 × 3 en `uint8` occupe $224 \times 224 \times 3$ octets.

</details>
<details><summary>Indice 3</summary>

1. $9 \times 10^6 \times 3 \times 10^5 = 2{,}7 \times 10^{12}$ octets, soit **2,7 To**. Pour 2 et 3, divise cette taille par le débit en octets par seconde, puis convertis en jours ou en heures. Pour 4, même produit avec 5,85 milliards et $10^5$ octets. Pour 5, $9 \times 10^6 \times 1\,024 \times 768 \times 3$ octets, puis le même produit en 224 × 224 × 3 ; compare au total en JPEG, et demande-toi ce qu'on perd à ne garder qu'une version réduite.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 29.4 — Présenter son projet final en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis l'objectif du chapitre (fiche, « Comment utiliser ce chapitre ») et ton ✏️ 29.2 pour la référence à battre.

</details>
<details><summary>Indice 2</summary>

Une ligne par point, sans jargon : qui a le problème, d'où viennent les données, ce qui entre et ce qui sort, comment on mesure, ce qui peut mal tourner.

</details>
<details><summary>Indice 3</summary>

Ligne 1, sur le modèle : « Aider [qui] à [quoi], parce que [pourquoi c'est utile]. » Pour les lignes 2 à 5 : une source précise avec sa taille et sa licence ; une entrée et une sortie concrètes (« une photo », « une classe parmi 10 ») ; une métrique **et** une baseline simple (classe majoritaire, régression logistique) ; un risque nommé (biais, décalage, licence).

</details>

### Ex 29.5 — Retirés, restreints ou biaisés : que vérifier avant d'utiliser un dataset ? ⚖️

<details><summary>Indice 1</summary>

Relis les encadrés 🕰️ des §29.3.1 et §29.5, et les deux premiers paragraphes du §29.1 (les trois points du livre).

</details>
<details><summary>Indice 2</summary>

Range tes vérifications en trois familles : le droit (licence, données personnelles, droit d'auteur), l'histoire (provenance, retraits, versions) et la qualité (représentativité, labels, doublons). Pour chaque cas des questions 2 à 5, demande-toi quelle famille est en cause.

</details>
<details><summary>Indice 3</summary>

Pour 1, une vérification commence par un verbe : « lire la licence », « vérifier que le dataset n'a pas été retiré »… Pour 2, Boston pose un problème de variable, pas de licence. Pour 3, un modèle entraîné garde ce qu'il a appris des données. Pour 4, « non commercial » couvre-t-il un filtre vendu par une start-up ? Pour 5, un lien pointe-t-il vers un contenu que l'on ne contrôle pas ?

</details>

### Ex 29.6 — Lire Gebru et coll., « Datasheets for Datasets » 📄

<details><summary>Indice 1</summary>

Lis d'abord l'introduction et la liste des sections de questions ; garde les exemples de datasheets (en annexe de l'article) pour la fin.

</details>
<details><summary>Indice 2</summary>

L'analogie est dans l'introduction : un composant électronique est toujours vendu avec une fiche qui dit ses caractéristiques et ses conditions d'usage. Les sections suivent la vie d'un dataset, de sa raison d'être à sa maintenance.

</details>
<details><summary>Indice 3</summary>

Pour 2, les sept sections suivent la vie d'un dataset, de sa raison d'être à sa maintenance : retrouve-les dans la table des matières de l'article, et pour chacune, recopie une question dans tes mots. Pour 4, cherche une question dont la réponse est un jugement (« y a-t-il des usages à éviter ? ») plutôt qu'un nombre. Pour 6, relis la section où les auteurs discutent des limites et du coût de l'approche.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 29.E1 — Comment choisir et valider un dataset pour un nouveau projet ?

<details><summary>Indice 1</summary>

Relis l'encadré 💼 de la fiche et le tableau des défauts.

</details>
<details><summary>Indice 2</summary>

Structure la réponse en étapes : définir le besoin (tâche, population visée, métrique), chercher (existant d'abord), vérifier (licence, provenance, retraits), auditer (qualité, doublons, représentativité), figer (version, somme de contrôle, data card).

</details>
<details><summary>Indice 3</summary>

Commence par : « D'abord je définis la population sur laquelle le modèle servira et la métrique qui compte. » Puis une phrase par étape de l'indice 2, chacune avec un exemple concret (une licence NC, des doublons entre entraînement et test, un hôpital sous-représenté).

</details>

### 29.E2 — Qu'est-ce qu'une data card et que doit-elle contenir ?

<details><summary>Indice 1</summary>

Relis « Au-delà du livre : documenter et télécharger » et son encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

Une définition en une phrase, les rubriques (les six sections de 🛠️ 29.15), puis à quoi elle sert : à celui qui crée le dataset, à celui qui l'utilise.

</details>
<details><summary>Indice 3</summary>

Commence par : « C'est la notice d'un dataset : d'où il vient, ce qu'il contient, ce qu'on a le droit et le devoir d'en faire. » Puis cite les sections (provenance, contenu, licence, usage prévu, limites et biais, téléchargement), les références (Gebru et coll., les dataset cards du Hub, Croissant) et un exemple de limite qu'elle aurait évité d'ignorer.

</details>

### 29.E3 — Comment détecter une fuite de données entre entraînement et test ?

<details><summary>Indice 1</summary>

Relis le tableau des défauts (lignes « quasi-doublons » et « fuite de la cible ») et ton 🐛 29.13.

</details>
<details><summary>Indice 2</summary>

Trois familles de fuites : les mêmes exemples (ou presque) des deux côtés, une variable qui contient la réponse, une étape ajustée sur toutes les données (normalisation, choix d'hyperparamètres sur le test). Pour chacune, une façon de la détecter.

</details>
<details><summary>Indice 3</summary>

Commence par : « 99,5 %, c'est d'abord un signal d'alerte : je compare au niveau attendu pour la tâche. » Puis : chercher les doublons et quasi-doublons entre découpages (hachage, distance des features), découper par groupes, regarder l'importance des variables, vérifier que toute transformation est ajustée sur l'entraînement seul, et tester sur des données d'une autre source ou d'une période plus récente.

</details>

### 29.E4 — Données publiques et usage commercial : que vérifier ?

<details><summary>Indice 1</summary>

Relis le §29.1 (les trois points du livre) et l'encadré 🕰️ du §29.5.

</details>
<details><summary>Indice 2</summary>

« Public » veut dire accessible, pas libre. Passe en revue : la licence, les droits sur le contenu (photos, textes d'auteurs), les données personnelles, l'histoire du dataset, et les obligations qui suivent le modèle.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je lis la licence : CC BY ou MIT autorisent un produit, NC ou “recherche seulement” l'interdisent. » Puis : les droits des contenus eux-mêmes, le RGPD s'il y a des personnes, les retraits et les miroirs, et la traçabilité (une data card, une version figée). Termine par l'avis d'un juriste quand le doute persiste.

</details>

<a id="notebook"></a>

## Exercices du notebook

### Ex 29.7 — Les datasets intégrés de scikit-learn, et une affirmation du livre à vérifier 📦

<details><summary>Indice 1</summary>

Relis l'encadré ⚠️ du §29.3.1 de la fiche. Toutes les fonctions `load_*` de scikit-learn acceptent `return_X_y=True`.

</details>
<details><summary>Indice 2</summary>

Pour `inventory_7`, une boucle sur `SKLEARN_LOADERS_7.items()`, une ligne de dictionnaire par dataset, puis `pd.DataFrame(rows)`. Pour `has_loader_7`, `getattr(skd, name)` dans un `try` qui attrape **deux** exceptions.

</details>
<details><summary>Indice 3</summary>

```python
rows = []
for name, loader in SKLEARN_LOADERS_7.items():
    X, y = getattr(skd, loader)(return_X_y=True)
    rows.append({"dataset": name, "exemples": X.shape[0], ...})   # then the keys variables, tache, classes
return pd.DataFrame(rows)
```
Pour `has_loader_7`, `except (AttributeError, ImportError): return False`. Pour b), `np.bincount(y)` sur Breast Cancer donne les deux effectifs.

</details>

### Ex 29.8 — Prédire : le même modèle sur MNIST puis sur Fashion-MNIST 🔮

<details><summary>Indice 1</summary>

Relis le §29.2 de la fiche et ton 🧠 Q4 e).

</details>
<details><summary>Indice 2</summary>

Pense au budget d'entraînement : un petit réseau dense, deux epochs, 10 000 images seulement. Pour b), quelles classes de vêtements un réseau qui ne voit que des pixels confondra-t-il ? Pour c), qu'est-ce qu'un réseau de chiffres « reconnaît » dans une chaussure ?

</details>
<details><summary>Indice 3</summary>

Pour a), demande-toi ce que deux epochs sur 10 000 images changent par rapport à l'entraînement complet que décrit le livre. Pour b), compte combien de classes de Fashion-MNIST se ressemblent, et ce qu'elles coûtent si le réseau se trompe sur une bonne partie de leurs images. Pour c), demande-toi si quelque chose de ce que le réseau a appris sur les chiffres sert à distinguer une chaussure d'un sac.

</details>

### Ex 29.9 — Prédire la chute d'accuracy en changeant de source de chiffres 🔮

<details><summary>Indice 1</summary>

Relis « Au-delà du livre : le décalage de distribution » dans la fiche.

</details>
<details><summary>Indice 2</summary>

Un chiffre de MNIST occupe un carré de 20 × 20 au centre de l'image ; un chiffre de scikit-learn remplit presque toute son image de 8 × 8. Agrandi tel quel, il déborde. Pour b), un softmax rend toujours des probabilités qui somment à 1 : rien ne l'oblige à « douter » sur une image inhabituelle.

</details>
<details><summary>Indice 3</summary>

Imagine un « 1 » de scikit-learn agrandi à 28 × 28 : un trait épais qui va du haut au bas de l'image, bien plus gros que ceux de MNIST. Le réseau compare des pixels à des positions fixes : à quel point ses repères tiennent-ils ? Place ton estimation dans les catégories de a). Pour b), le réseau a été entraîné à être sûr de lui sur ce qu'il voyait : cette habitude disparaît-elle hors de sa distribution ?

</details>

### Ex 29.10 — torchvision.datasets et Hugging Face datasets : mêmes données, deux API 📦

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « Hugging Face `datasets` et `huggingface_hub` » du §29.3.2. Une image PIL se convertit en tableau par `np.asarray(img)`.

</details>
<details><summary>Indice 2</summary>

`split["image"]` rend la liste des images PIL, `split["label"]` la liste des entiers : empile les tableaux avec `np.stack`. Pour `compare_10`, traite d'abord le cas des formes différentes ; puis calcule les écarts **en entiers signés**.

</details>
<details><summary>Indice 3</summary>

```python
X = np.stack([np.asarray(img, dtype=np.uint8) for img in split["image"]])
...                                   # y: the labels, as int64
```
Pour `compare_10` : traite d'abord le cas `Xa.shape != Xb.shape` (rends tout de suite `max_diff = -1` et `same_pixels = False`) ; sinon, l'écart se calcule après `Xa.astype(np.int64)`. Les labels se comparent après avoir vérifié qu'ils ont la même longueur.

</details>

### Ex 29.11 — Explorer le Hugging Face Hub et OpenML : chercher, filtrer, lire la licence 📦

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 sur `huggingface_hub` (§29.3.2) et l'encadré 🕰️ du §29.4 (OpenML et `fetch_openml`).

</details>
<details><summary>Indice 2</summary>

`api.list_datasets(...)` rend un itérateur d'objets ; chacun a `.id`, `.downloads` et `.tags`, une liste de chaînes comme `"license:mit"`. `fetch(...)` rend un `Bunch` : `bunch.frame` (un `DataFrame`) et `bunch.details` (un dictionnaire).

</details>
<details><summary>Indice 3</summary>

```python
if api is None:
    from huggingface_hub import HfApi
    api = HfApi()
for info in api.list_datasets(search=query, limit=limit, sort="downloads"):
    licences = [t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")]
    ...
```
Pour `openml_summary_11`, `rows, columns = bunch.frame.shape`, puis lis les clés de `bunch.details` avec `.get` pour celles qui peuvent manquer.

</details>

### Ex 29.12 — Profil express d'un dataset inconnu 📈

<details><summary>Indice 1</summary>

Relis le tableau de « Au-delà du livre : auditer un dataset » : les colonnes « Comment le voir » donnent les méthodes de pandas.

</details>
<details><summary>Indice 2</summary>

Pour chaque colonne `s = df[col]` : `s.isna().sum()`, `s.isna().mean()`, `s.nunique()`, et `s.value_counts()` (qui ignore les manquants et trie du plus au moins fréquent). Construis un dictionnaire par colonne, puis `pd.DataFrame.from_dict(rows, orient="index")`.

</details>
<details><summary>Indice 3</summary>

```python
counts = s.value_counts()
rows[col] = {"type": str(s.dtype), "manquants": int(s.isna().sum()), ...,
             "mode": counts.index[0], "part_mode": counts.iloc[0] / counts.sum()}
```
Pour a), `idxmax()` sur la colonne `"manquants"` du profil ; pour b), `ADULT_12.duplicated().sum()` ; pour c), `value_counts(normalize=True)` sur `"class"`.

</details>

### Ex 29.13 — Un score trop beau pour être vrai : doublons entre entraînement et test 🐛

<details><summary>Indice 1</summary>

Relis les lignes « quasi-doublons » et « fuite » du tableau des défauts. Une fuite, c'est toute information du test qui influence l'entraînement **ou le choix** du modèle.

</details>
<details><summary>Indice 2</summary>

Lis chaque ligne en te demandant : « cette étape voit-elle des lignes de test ? » Trois familles : le découpage lui-même (qu'ignore-t-il ?), un prétraitement ajusté (sur quelles lignes ?), un choix d'hyperparamètre (sur quel score ?). Pour b), `train_test_split` accepte n'importe quelle liste : découpe `GROUPS_13` avec les réglages de la ligne B.

</details>
<details><summary>Indice 3</summary>

La ligne **B** découpe au hasard : les quatre photos d'un vêtement se répartissent des deux côtés, et le 1-NN retrouve la jumelle. Cherche les deux autres lignes avec la question de l'indice 2. Pour b), refais le découpage de la ligne B sur `GROUPS_13` (mêmes réglages), puis compte les lignes de test dont le groupe est aussi dans l'entraînement (`np.isin`). Pour `evaluate_fixed_13`, le squelette :
```python
train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups))
fit, val = ...                        # a second GroupShuffleSplit on the rows of train (random_state=1)
fit, val = train[fit], train[val]     # its indices are relative to train
...                                   # PCA on X[fit], k chosen on X[val]; then PCA and k-NN on X[train], one score on X[test]
```

</details>

### Ex 29.14 — Décalage de distribution : un modèle MNIST face aux chiffres de scikit-learn 🔬

<details><summary>Indice 1</summary>

Lis `digits_naive` dans la cellule des outils : la méthode `"naive"` en est presque une copie.

</details>
<details><summary>Indice 2</summary>

Un tenseur `(N, 1, 8, 8)` en `float32`, multiplié par 255/16 ; `F.interpolate(..., size=(20, 20), mode="bilinear", align_corners=False)` ; `F.pad(x, (4, 4, 4, 4))` ajoute 4 pixels de chaque côté des deux dernières dimensions. Pour `"center"`, une boucle sur les images, en `float`, avant l'arrondi final.

</details>
<details><summary>Indice 3</summary>

```python
t = torch.as_tensor(np.asarray(images), dtype=torch.float32)[:, None] * 255.0 / 16.0
small = F.interpolate(t, size=(20, 20), mode="bilinear", align_corners=False)
out = F.pad(small, (4, 4, 4, 4))[:, 0].numpy()        # "box"
```
Pour `"naive"`, la même interpolation directement à 28 × 28. Pour `"center"`, une boucle sur les images de `"box"` (encore en `float`) : le centre de masse, puis le décalage qui l'amène en (13,5 ; 13,5). L'arrondi, le `clip` et la conversion en `uint8` viennent tout à la fin, pour les trois méthodes.

</details>

### Ex 29.15 — Data card et script de téléchargement vérifié pour le projet final 🛠️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « une somme de contrôle SHA-256 » de la fiche, et ouvre deux fiches de `data/cards/` pour la data card.

</details>
<details><summary>Indice 2</summary>

`hashlib.sha256()`, puis `update` sur chaque morceau lu par `f.read(chunk_size)` jusqu'à une chaîne vide, puis `hexdigest()`. Pour le téléchargement, trois cas dans l'ordre : déjà là et intact (rendre tout de suite), téléchargement dans le `.part`, puis somme juste (renommer) ou fausse (supprimer, lever).

</details>
<details><summary>Indice 3</summary>

```python
dest = Path(dest)
if ...:                               # dest exists AND its checksum is the right one
    return dest
part = dest.with_name(dest.name + ".part")
...                                   # download into part, then compare sha256_15(part) with sha256
part.replace(dest)                    # only if the checksum is right; otherwise part.unlink() and a ValueError
```
Pour la data card, remplace chaque `...` ; la somme se calcule avec ta fonction une fois le fichier téléchargé.

</details>

### Ex 29.16 — Le dataset piégé : trouver les huit défauts 🏆

<details><summary>Indice 1</summary>

Le tableau de « Au-delà du livre : auditer un dataset » a exactement une ligne par défaut, avec la commande qui le révèle. Passe-les toutes.

</details>
<details><summary>Indice 2</summary>

`describe()` montre les valeurs absurdes (un minimum, un maximum) ; `nunique()` les colonnes sans information ; `value_counts()` les labels mal écrits et la classe rare ; `duplicated()` les doublons ; `pd.crosstab(une_colonne, FRUITS_16["fruit"])` une colonne qui recopie le label. Pour les quasi-doublons, un `merge` des lignes de test avec celles d'entraînement sur toutes les colonnes sauf `id_lot`, `split` et `poids_g`.

</details>
<details><summary>Indice 3</summary>

a) `FRUITS_16.duplicated().sum()`. Pour b), fusionne (`merge`) les lignes de test avec celles d'entraînement sur toutes les colonnes sauf `id_lot`, `split` et `poids_g`, en gardant les deux poids (`suffixes=`), puis garde les paires dont les poids diffèrent d'au plus 1 g : attention, une ligne de test peut avoir deux jumelles, compte des lignes de test **distinctes**. Pour c) à h), une commande de l'indice 2 par question. Pour `clean_16`, commence par `out = df.drop_duplicates().copy()` et traite les défauts un à un.

</details>
