"""Parts A and B of the chapter 29 notebook: statements and cells (used by build_ch29.py)."""

from __future__ import annotations

from chapter_kit import Ex, Part, bug_letters_mistakes, guarded

import ch29_code as C

RECORD_13 = ('wb.record("29.13a", "BCE", mistakes=' + bug_letters_mistakes("BCE", "ADFG", {
    "B": "le découpage au hasard ignore GROUPS_13 : les copies d'une même photo tombent des deux côtés",
    "C": "la PCA est ajustée sur toutes les lignes, test compris",
    "E": "k est choisi sur le score de test, qui sert ensuite à juger le modèle"},
    special={("E", "F"): "la ligne (F) réentraîne avec le k déjà choisi : le choix fait sur le test est à la ligne (E)"}) + ')\n'
    'wb.record("29.13b", 500, mistakes={"compte les lignes de test dont le groupe apparaît aussi dans '
    'l\'entraînement (np.isin), pas l\'inverse": 0, "le test du découpage de la ligne B a 500 lignes : compte-les une '
    'par une, pas par groupe": 125})')

RECORD_16 = r'''answers_r16 = wb.datasets.trap_dataset_answers()
wb.record("29.16a", answers_r16["exact_duplicates"], mistakes={"duplicated() ne compte que les répétitions, pas la première ligne de chaque paire": 26})
wb.record("29.16b", answers_r16["near_duplicates"], mistakes={"compte chaque ligne de test une seule fois, même si elle a deux jumelles dans l'entraînement": 18})
wb.record("29.16c", answers_r16["missing_code_column"], mistakes={"poids_g n'a pas de valeur aberrante négative : regarde les minimums de describe()": "poids_g"})
wb.record("29.16d", answers_r16["constant_column"], mistakes={"id_lot change à chaque ligne : c'est un identifiant, pas une colonne constante": "id_lot"})
wb.record("29.16e", answers_r16["leaking_column"], mistakes={"les mesures prédisent le fruit honnêtement : cherche une colonne qui le recopie": "couleur_r"})
wb.record("29.16f", answers_r16["units_column"], mistakes={"le poids varie de 95 à 386 g, sans saut d'unité : regarde le maximum des diamètres": "poids_g"})
wb.record("29.16g", answers_r16["invalid_labels"], mistakes={"« poire » suivi d'une espace n'est pas « poire » : value_counts() les sépare": 9})
wb.record("29.16h", answers_r16["rare_class"], mistakes={"compte seulement les labels bien écrits : la poire est loin d'être rare": "poire"})'''

PART_A = Part("A", "Charger, chercher, comparer",
              "Les datasets intégrés aux bibliothèques, deux sources d'images au même format, le Hugging Face Hub et "
              "OpenML. Chaque fois, la même discipline : mesurer sur les données chargées plutôt que croire la "
              "description, et noter d'où viennent les données et sous quelle licence.", exercises=[
    Ex("29.7", "📦", 1, 15, "Les datasets intégrés de scikit-learn, et une affirmation du livre à vérifier",
       "inventorier les datasets intégrés de scikit-learn en mesurant les données chargées, et vérifier qu'un loader existe encore.",
       "ch. 15 (scikit-learn) · fiche §29.3.1 (encadrés ⚠️ et 🕰️)", tracks="R, M, C",
       body=r"""Le livre annonce 5 620 images pour Digits et liste Boston parmi les datasets de scikit-learn. Vérifie sur ta version.

**Contrat**
- `inventory_7()` : un `DataFrame`, une ligne par dataset de `SKLEARN_LOADERS_7` (dans cet ordre), colonnes `"dataset"`, `"exemples"`, `"variables"`, `"tache"` (`"classification"` ou `"régression"`, d'après `REGRESSION_7`) et `"classes"` (le nombre de labels distincts, 0 pour une régression). Tout se **mesure** sur `X, y = getattr(skd, loader)(return_X_y=True)`.
- `has_loader_7(name)` : `True` si `sklearn.datasets` fournit encore la fonction `name`, `False` sinon, sans lever d'erreur. Piège : pour un loader retiré, scikit-learn lève une `ImportError` explicative, pas une `AttributeError` ; `hasattr` ne suffit donc pas.

La cellule de vérification compare ta table aux mesures, puis te fait reporter :
a) le nombre d'images de Digits, mesuré par ta table ;
b) la part de la classe minoritaire de Breast Cancer, à 3 décimales (`minority_7`, à calculer) ;
c) ce que rend `has_loader_7("load_boston")`.

Dans tes notes : lis le message de l'`ImportError` de `load_boston` (`try: skd.load_boston() except ImportError as e: print(e)`). Que propose scikit-learn à la place ? Pourquoi une table mesurée vaut-elle mieux qu'une liste recopiée d'un livre ou d'une page web ?""",
       given=C.GIVEN_7, todo=C.TODO_7, check=C.CHECK_7, solution=C.SOLUTION_7 + "\n\n\n" + C.CHECK_7, record=C.RECORD_7),

    Ex("29.8", "🔮", 1, 10, "Prédire : le même modèle sur MNIST puis sur Fashion-MNIST",
       "prédire ce que coûte le passage de MNIST à Fashion-MNIST à un même petit réseau, et ce que fait un modèle de chiffres face à des vêtements.",
       "Ex 29.7 · fiche §29.2", thread="MNIST", tracks="R, M, C",
       body=r"""La cellule suivante entraîne deux fois le même petit réseau dense (784-128-10, deux epochs d'Adam sur 10 000 images, `train_mlp` dans les outils) : `MODEL_M` sur MNIST, `MODEL_F` sur Fashion-MNIST. Même format, même code, seule la source change. Elle n'affiche rien.

**Avant d'exécuter l'expérience**, écris tes trois prédictions :
a) l'accuracy de test de `MODEL_M` sur MNIST : `"A"` moins de 0,90, `"B"` de 0,90 à 0,97, `"C"` plus de 0,97 ;
b) les points d'accuracy perdus par `MODEL_F` sur Fashion-MNIST par rapport à a) : `"A"` moins de 5, `"B"` de 5 à 15, `"C"` plus de 15 ;
c) l'accuracy de `MODEL_M` (entraîné sur des chiffres) sur le test de Fashion-MNIST : `"A"` moins de 0,2, `"B"` de 0,2 à 0,5, `"C"` plus de 0,5.

La cellule d'expérience n'affiche rien tant que tes trois prédictions ne sont pas écrites.""",
       given=C.GIVEN_8, todo=C.TODO_8, solution=C.SOLUTION_8, hypothesis=True, record=C.RECORD_8,
       after=[("code", guarded(C.EXPERIMENT_8, ["mnist_8", "drop_8", "crossed_8"],
                               "⏳ Ex 29.8 : écris d'abord tes trois prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : quelles classes de Fashion-MNIST le réseau reconnaît-il le moins bien (le "
                     "graphique), et pourquoi (Q4 e) ? Que dit c) de la confiance d'un modèle hors de sa "
                     "distribution d'entraînement ? Pourquoi le livre conseille-t-il Fashion-MNIST comme étape après "
                     "MNIST ?")]),

    Ex("29.9", "🔮", 1, 10, "Prédire la chute d'accuracy en changeant de source de chiffres",
       "prédire ce que devient un modèle MNIST face à d'autres chiffres manuscrits, et s'il s'en rend compte.",
       "Ex 29.8 · fiche §29.2, « Au-delà du livre : le décalage de distribution »", thread="MNIST", tracks="R, M, C",
       body=r"""scikit-learn fournit d'autres chiffres manuscrits : 1 797 images de 8 × 8 pixels, valeurs de 0 à 16 (`load_digits`). La cellule suivante les agrandit **telles quelles** à 28 × 28, valeurs de 0 à 255 (`digits_naive`, dans les outils) : même format que MNIST, mêmes dix classes. On les donne à `MODEL_M` (29.8), sans autre entraînement.

**Avant d'exécuter**, écris tes deux prédictions :
a) l'accuracy de `MODEL_M` sur ces chiffres : `"A"` plus de 0,7, `"B"` de 0,4 à 0,7, `"C"` moins de 0,4 ;
b) la **confiance moyenne** du modèle sur ces chiffres (pour chaque image, sa plus grande probabilité softmax ; puis la moyenne) reste-t-elle au-dessus de 0,6 ? `True` ou `False`.

La cellule d'expérience montre aussi quelques images des deux sources côte à côte, et n'affiche rien tant que tes deux prédictions ne sont pas écrites.""",
       given=C.GIVEN_9, todo=C.TODO_9, solution=C.SOLUTION_9, hypothesis=True, record=C.RECORD_9,
       after=[("code", guarded(C.EXPERIMENT_9, ["digits_9", "confident_9"],
                               "⏳ Ex 29.9 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : compare les deux rangées d'images. Qu'est-ce qui diffère, à part la taille "
                     "d'origine (la place du chiffre dans l'image, l'épaisseur du trait, le flou) ? Pourquoi une "
                     "confiance élevée sur des données décalées est-elle dangereuse en production ? 29.14 mesure "
                     "ce que l'on récupère en préparant mieux les images.")]),

    Ex("29.10", "📦", 2, 25, "torchvision.datasets et Hugging Face datasets : mêmes données, deux API",
       "charger un dataset du Hugging Face Hub à une révision fixée, le convertir en tableaux NumPy et le comparer à une autre source.",
       "ch. 20 et 23 (Dataset, DataLoader) · fiche §29.3.2 (encadré 🕰️, encadré 🧮)", thread="Fashion-MNIST",
       tracks="R, M, C",
       body=r"""La cellule suivante charge le split de test de Fashion-MNIST depuis le Hugging Face Hub, à une **révision** fixée (un commit du dépôt du dataset) : `load_dataset(nom, split="test", revision=...)`. Sans réseau, elle fabrique le même format depuis la copie du workbook et le dit. `wb.datasets.load_fashion_mnist`, lui, passe par `torchvision.datasets.FashionMNIST` (ou un miroir) et rend des tableaux NumPy.

**Contrat**
- `hf_to_numpy_10(split)` : depuis un split aux colonnes `"image"` (images PIL) et `"label"`, rend `(X, y)` : `X` en `uint8` de forme `(N, 28, 28)`, `y` en `int64` de forme `(N,)`, dans l'ordre du split ;
- `compare_10(Xa, ya, Xb, yb)` : un dictionnaire `{"same_shape", "same_labels", "same_pixels", "max_diff"}` ; `same_labels` exige autant de labels, égaux dans le même ordre ; `max_diff` est le plus grand écart absolu entre deux pixels (0 si identiques) ; si les formes diffèrent, `same_pixels` vaut `False` et `max_diff` vaut −1. Piège : en `uint8`, `3 - 10` vaut 249.

La cellule de vérification teste `compare_10` sur des cas fabriqués (un pixel modifié, des labels mélangés, des formes différentes), puis compare les deux sources de Fashion-MNIST et affiche les noms des classes de chacune.

Dans tes notes : les deux sources sont-elles identiques, pixel à pixel et dans le même ordre ? Les noms des classes sont-ils écrits pareil ? Pourquoi fixer `revision=` plutôt que prendre la dernière version ? Écris l'équivalent `torchvision.datasets.FashionMNIST(root, train=False, download=True)` et dis ce qu'il rend (des images PIL et des entiers, transformables par `transform=`).""",
       given=C.GIVEN_10, todo=C.TODO_10, check=C.CHECK_10, solution=C.SOLUTION_10 + "\n\n\n" + C.CHECK_10),

    Ex("29.11", "📦", 2, 25, "Explorer le Hugging Face Hub et OpenML : chercher, filtrer, lire la licence",
       "chercher des datasets par programme sur le Hugging Face Hub et OpenML, et en lire la licence, la taille et la source.",
       "Ex 29.10 · fiche §29.4 (encadré 🕰️), §29.5", tracks="R, M, C",
       body=r"""**Contrat**
- `search_hub_11(query, limit=5, api=None)` : les `limit` datasets du Hub **les plus téléchargés** que trouve `query`, du plus au moins téléchargé, dans un `DataFrame` aux colonnes `"id"`, `"downloads"` et `"licence"`. La licence est le `xxx` d'un tag `"license:xxx"`, recopié tel quel (`"inconnue"` s'il n'y a pas de tag). `api` est un `HfApi` (un nouveau si `None`) : `api.list_datasets(search=..., limit=..., sort="downloads")` rend des objets qui ont `.id`, `.downloads` et `.tags`.
- `openml_summary_11(data_id, fetch=None)` : un dictionnaire sur un dataset d'OpenML, lu dans `bunch = fetch(data_id=data_id, as_frame=True)` (`fetch` vaut par défaut `sklearn.datasets.fetch_openml`) : `"name"`, `"version"`, `"licence"` (dans `bunch.details`), `"rows"` et `"columns"` (de `bunch.frame`, cible comprise), `"target"` (`details["default_target_attribute"]`) et `"source"` (`details["original_data_url"]`, `None` s'il manque).

La cellule de vérification ne dépend pas du réseau : elle passe à tes fonctions un Hub et un OpenML **simulés** (`FakeHubApi`, `fake_fetch_openml`, dans les outils). La cellule suivante les lance pour de vrai si le réseau répond : la recherche `"penguins"` sur le Hub, puis Palmer Penguins sur OpenML (`data_id=42585`).

Dans tes notes : parmi les résultats du Hub, lesquels pourrais-tu utiliser dans un produit, d'après leur licence ? Que fais-tu d'un dataset sans licence ? Ouvre la page (la *dataset card*) du premier résultat : que dit-elle de la provenance des données ? Cherche deux datasets candidats pour ton projet final et note leur licence et leur taille.""",
       todo=C.TODO_11, check=C.CHECK_11, solution=C.SOLUTION_11 + "\n\n\n" + C.CHECK_11,
       after=[("code", C.EXPERIMENT_11)]),
])

PART_B = Part("B", "Auditer, documenter, déjouer les pièges",
              "Avant d'entraîner quoi que ce soit : profiler un dataset inconnu, traquer les doublons qui gonflent un "
              "score, mesurer un décalage de distribution, puis écrire la data card et le script de téléchargement de "
              "ton projet final. Le défi final rassemble tout sur un tableau qui cache huit défauts.", exercises=[
    Ex("29.12", "📈", 2, 20, "Profil express d'un dataset inconnu",
       "écrire en quelques lignes le profil de chaque colonne d'un tableau, et y lire ses défauts.",
       "ch. 12 (pandas) · 0A · fiche « Au-delà du livre : auditer un dataset »", tracks="R, M, C",
       body=r"""Adult (le recensement américain de 1994, `data/cards/adult.md`) est chargé dans `ADULT_12`.

**Contrat** de `profile_12(df)` : un `DataFrame` indexé par les noms des colonnes de `df` (dans leur ordre), aux colonnes `"type"` (`str` du dtype), `"manquants"` (le nombre de valeurs manquantes), `"part_manquants"` (leur part, entre 0 et 1), `"distinctes"` (le nombre de valeurs distinctes, sans les manquantes), `"mode"` (la valeur la plus fréquente) et `"part_mode"` (sa part parmi les valeurs présentes).

La cellule de vérification compare ton profil aux mesures de pandas et trace deux graphiques (la part de manquants et la part du mode, colonne par colonne). Reporte :
a) la colonne d'`ADULT_12` qui a le plus de valeurs manquantes ;
b) le nombre de lignes qui répètent une ligne précédente (`duplicated`) ;
c) la part de la classe majoritaire de la colonne `"class"`, à 2 décimales.

Dans tes notes : quelles colonnes sont presque constantes (une part du mode au-dessus de 0,9) ? Sont-elles inutiles pour autant ? Les doublons de b) sont-ils des erreurs, ou deux personnes qui se ressemblent ? Que vaut un modèle qui prédit toujours la classe majoritaire (✏️ 29.2) ?""",
       given=C.GIVEN_12, todo=C.TODO_12, check=C.CHECK_12, solution=C.SOLUTION_12 + "\n\n\n" + C.CHECK_12,
       record=C.RECORD_12),

    Ex("29.13", "🐛", 2, 25, "Un score trop beau pour être vrai : doublons entre entraînement et test",
       "trouver les trois fuites d'une évaluation et la corriger, en découpant par groupes.",
       "ch. 8 (découpage, validation) · fiche « Au-delà du livre : auditer un dataset » · 💼 29.E3", tracks="R, M, C",
       body=r"""Un collègue a photographié 500 vêtements, chacun quatre fois (lumière et bruit un peu différents) : 2 000 images, `X_13`, leurs labels `Y_13`, et `GROUPS_13`, le numéro du vêtement de chaque image. Son évaluation, `evaluate_buggy_13` (cellule suivante), annonce une accuracy de test parfaite. Elle contient **trois** lignes fautives, trois façons de laisser le test influencer l'entraînement ou l'évaluation.

1. Écris dans `faulty_13` les lettres des lignes fautives (dans n'importe quel ordre).
2. Mesure la fuite principale : `shared_13`, le nombre de lignes de test du découpage de la ligne B dont le vêtement (`GROUPS_13`) apparaît aussi dans l'entraînement. Refais ce découpage avec les mêmes réglages : `train_test_split` rend les mêmes indices pour toute liste de même longueur, découpée avec le même `stratify` et le même `random_state`.
3. Écris `evaluate_fixed_13` selon sa docstring : mêmes sorties, sans les fuites.

La cellule de vérification lance aussi ta version sur un piège : un label tiré **au hasard** pour chaque vêtement. Rien n'est apprenable, une évaluation honnête doit tomber au niveau du hasard (environ 0,1). Puis elle espionne `PCA` et `KNeighborsClassifier` pendant un appel de ta version : quelles lignes voit chaque `fit`, chaque `score` ?

Dans tes notes : pour chaque ligne fautive, la fuite qu'elle crée. Laquelle explique presque tout l'écart ? Où trouve-t-on des quasi-doublons dans de vrais datasets (photos en rafale, plusieurs images d'un même patient, phrases traduites deux fois) ?""",
       given=C.GIVEN_13, todo=C.TODO_13, check=C.CHECK_13, solution=C.SOLUTION_13 + "\n\n\n" + C.CHECK_13,
       record=RECORD_13),

    Ex("29.14", "🔬", 3, 35, "Décalage de distribution : un modèle MNIST face aux chiffres de scikit-learn",
       "mesurer ce que récupère une préparation fidèle des images, puis quelques exemples étiquetés, face à un décalage de distribution.",
       "Ex 29.9 · ch. 23 (fine-tuning) · fiche « Au-delà du livre : le décalage de distribution »", thread="MNIST",
       tracks="R, M, C",
       body=r"""En 29.9, les chiffres de scikit-learn, agrandis tels quels, ont dérouté `MODEL_M`. Or les images de MNIST suivent une recette précise : chaque chiffre est réduit pour tenir dans un carré de 20 × 20, puis placé dans l'image de 28 × 28 de sorte que son **centre de masse** soit au centre.

**Contrat** de `to_mnist_14(images, method)` : la docstring de la cellule suivante. Trois méthodes : `"naive"` (comme `digits_naive`), `"box"` (agrandir à 20 × 20, puis 4 pixels noirs de chaque côté) et `"center"` (`"box"`, puis chaque image décalée pour mettre son centre de masse au centre, `(13,5 ; 13,5)`).

La cellule de vérification contrôle les trois méthodes. L'expérience mesure ensuite :
1. l'accuracy de `MODEL_M` avec chaque méthode ;
2. l'accuracy après un fine-tuning du réseau sur $k$ chiffres étiquetés par classe ($k$ = 0, 1, 5, 20), mesurée sur les autres chiffres ;
3. une référence : un modèle simple entraîné sur les chiffres de scikit-learn eux-mêmes.

Dans tes notes : quelle étape de préparation rapporte le plus ? Combien d'exemples étiquetés faut-il pour approcher la référence ? Le décalage restant tient-il aux images (le décalage de covariables de la fiche) ou aux labels ? Que ferais-tu, dans ton projet final, si les données d'utilisation diffèrent de celles d'entraînement ?""",
       given=C.GIVEN_14, todo=C.TODO_14, check=C.CHECK_14, solution=C.SOLUTION_14 + "\n\n\n" + C.CHECK_14,
       after=[("code", C.EXPERIMENT_14)]),

    Ex("29.15", "🛠️", 3, 60, "Data card et script de téléchargement vérifié pour le projet final",
       "écrire un téléchargement reproductible, vérifié par somme de contrôle, et la data card du dataset de ton projet final.",
       "Ex 29.6, 29.11 · fiche « Au-delà du livre : documenter et télécharger » (encadré 🧮)", tracks="R, M, C",
       body=r"""**Contrat**
- `sha256_15(path, chunk_size=1 << 20)` : la somme de contrôle SHA-256 d'un fichier (64 caractères hexadécimaux), lu **par morceaux** (`hashlib.sha256`, `update`, `hexdigest`) : un fichier de plusieurs gigaoctets ne tient pas en mémoire ;
- `download_verified_15(url, dest, sha256)` : si `dest` existe déjà avec la bonne somme, ne rien télécharger ; sinon télécharger dans `dest + ".part"` (par exemple avec `urllib.request.urlopen`), vérifier la somme, puis renommer en `dest` ; si la somme est fausse, supprimer le `.part` et lever `ValueError` (`dest` reste tel quel). Rendre `Path(dest)`.

La cellule de vérification « télécharge » des fichiers locaux (des URL `file://`) : arrivée intacte, lecture par morceaux, pas de nouveau téléchargement si le fichier est déjà là et intact, un nouveau s'il est abîmé, refus d'une somme fausse sans toucher à `dest`.

Puis remplis `CARD_15`, la data card du dataset de **ton projet final** : les six sections (Provenance, Contenu, Licence, Usage prévu, Limites et biais, Téléchargement), l'adresse de la source, et la somme SHA-256 de ton fichier (calculée avec `sha256_15` une fois le fichier téléchargé). Pour un dataset du Hugging Face Hub, note aussi la révision (`HfApi().dataset_info(nom).sha`) et télécharge avec `hf_hub_download(..., repo_type="dataset", revision=...)`. Modèle : les six sections ci-dessus ; les fiches de `data/cards/` montrent le niveau de détail attendu (elles n'ont pas tous ces six titres). La vérification liste ce qui manque.

Dans tes notes : pourquoi télécharger dans un `.part` puis renommer ? Une somme de contrôle protège-t-elle contre un dataset qui change de contenu, contre un fichier corrompu, contre un dataset retiré ? Que doit faire ton script si la source disparaît ?""",
       todo=C.TODO_15, check=C.CHECK_15, solution=C.SOLUTION_15 + "\n\n\n" + C.CHECK_15),

    Ex("29.16", "🏆", 3, 60, "Le dataset piégé : trouver les huit défauts",
       "auditer seul un tableau inconnu, y trouver huit défauts de types différents, puis le nettoyer.",
       "Ex 29.12, 29.13 · fiche « Au-delà du livre : auditer un dataset » (tableau des défauts)", tracks="C",
       body=r"""`FRUITS_16` décrit 1 230 fruits : un identifiant, six mesures, un pays, un code qualité, le label `fruit` (pomme, poire ou coing) et le découpage `split` déjà fait (`"train"` ou `"test"`). Il cache **huit défauts**, un de chaque sorte : doublons exacts, quasi-doublons entre entraînement et test, valeurs manquantes codées par un nombre, colonne sans information, colonne qui trahit le label, unités incohérentes, labels mal écrits, classe très rare.

Explore-le dans autant de cellules que tu veux (`describe`, `value_counts`, `duplicated`, `nunique`, `crosstab`…), puis réponds aux questions a) à h) de la cellule à compléter et écris `clean_16(df)`, qui rend une copie nettoyée.

La cellule de vérification contrôle tes huit réponses, puis ta copie nettoyée. L'expérience entraîne la même forêt aléatoire sur les données brutes et sur les données nettoyées, et compare leurs accuracies de test.

Dans tes notes : pour chaque défaut, la commande qui l'a révélé et ce qu'il aurait fait au modèle. Pourquoi l'accuracy baisse-t-elle après le nettoyage, alors que le modèle est meilleur ? Que mettrais-tu dans la section « Limites et biais » de la data card de ce dataset ?""",
       given=C.GIVEN_16, todo=C.TODO_16, check=C.CHECK_16, solution=C.SOLUTION_16 + "\n\n\n" + C.CHECK_16,
       record=RECORD_16, after=[("code", C.EXPERIMENT_16)]),
])

PARTS = [PART_A, PART_B]
ROCKETS = ()
