# 29 · Datasets et préparation du projet final — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels et des exercices ✏️ 29.1 et 29.2 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb` (démarches plus bas).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 29.Q1 — Partir de l'existant : modèle pré-entraîné, dataset public ou collecte ?
a) **B** : un modèle déjà entraîné (ajusté si besoin), sinon un dataset public, et la collecte en dernier, parce qu'elle coûte le plus. b) **Vrai** : c'est le transfer learning du bonus B1. c) **C** : on économise le calcul et les données de l'entraînement initial ; un modèle pré-entraîné n'est pas toujours meilleur (A) et il hérite des biais de ses données (B). d) **Vrai** : « public » veut dire accessible, pas libre ; la licence décide. e) **B** : lire les règles d'usage, puis regarder les données à la main.
**À retenir** : réutiliser d'abord, collecter en dernier ; et rien ne se réutilise sans lire sa licence.

### 29.Q2 — Licences et restrictions d'usage : vrai ou faux
a) **Vrai** : CC BY permet tout usage, commercial compris, à condition de citer. b) **Faux** : NC interdit l'usage commercial. c) **Faux** : un prototype destiné à un produit vendu n'est pas de la recherche ; « recherche seulement » l'exclut. d) **Faux** : la licence dit ce que **celui qui publie** permet ; elle ne prouve pas que les personnes photographiées ont consenti. e) **Vrai** : des données personnelles publiées restent des données personnelles.
**À retenir** : trois questions distinctes : que permet la licence, qui a des droits sur le contenu, quelles personnes figurent dans les données.

### 29.Q3 — Les défauts qu'une inspection manuelle révèle
a) **« ABC »** : champs vides, labels isolés, fautes d'orthographe ; un réseau mal initialisé (D) n'est pas un défaut des données. b) **B** : −999 est un code pour « inconnu ». c) **Faux** : `isna()` ne voit que `NaN` et `None` ; il faut d'abord `df.replace(-999, np.nan)`. d) **B** : `describe()` montre un maximum dix fois trop grand, un histogramme deux bosses. e) **Vrai** : Northcutt et coll. (2021) trouvent au moins 3,3 % d'erreurs de labels en moyenne dans dix jeux de test de référence.
**À retenir** : un manquant peut se déguiser en nombre ; on regarde les minimums, les maximums et les valeurs rondes absurdes.

### 29.Q4 — MNIST et Fashion-MNIST : même format, difficulté différente
a) **70 000** (60 000 + 10 000). b) **28**. c) **10**. d) **Vrai** : même format, mêmes formes de tableaux ; c'est tout l'intérêt de Fashion-MNIST. e) **B** : T-shirt, chemise, pull et manteau se ressemblent beaucoup à 28 × 28 pixels.
**À retenir** : Fashion-MNIST remplace MNIST sans changer une ligne de code, et montre mieux les limites d'un modèle.

### 29.Q5 — Les datasets de scikit-learn : lequel pour quelle tâche, et pourquoi Boston a disparu
a) **1 797** : `load_digits` ne livre qu'une partie des 5 620 chiffres de l'UCI. b) **B** : déprécié en 1.0, retiré en 1.2. c) **A** : la variable `B` avait été construite à partir de la proportion de population noire du quartier. d) **C** : California Housing. e) **Faux** : Diabetes est un tableau de dix mesures et d'une cible numérique.
**À retenir** : on mesure un dataset sur les données chargées, et on lit la description de ses variables.

### 29.Q6 — Des datasets de Keras à torchvision et Hugging Face
a) **C** : deux couples `(x_train, y_train), (x_test, y_test)` de tableaux NumPy. b) **60 000** (50 000 + 10 000). c) **B** : des suites d'entiers, un par mot (rang de fréquence) ; le texte brut est perdu. d) **B** : `torchvision.datasets.CIFAR10(root, train=True, download=True)`. e) **Vrai** : `revision=` désigne un commit (ou une branche, un tag) du dépôt du dataset.
**À retenir** : trois API, un même geste : un nom, un découpage, une version, des transformations.

### 29.Q7 — Collections de datasets : UCI, Kaggle, OpenML, Hugging Face Hub et liens morts
a) **« ABC »** : le Hub, Kaggle, OpenML ; deeplearning.net est un lien mort. b) **Faux** : l'outil `kaggle` lit une clé d'API personnelle, liée à un compte (la bibliothèque `kagglehub` télécharge, elle, beaucoup de datasets publics sans compte). c) **C** : scikit-learn (`sklearn.datasets.fetch_openml`). d) **Faux** : plusieurs liens du livre sont morts ou redirigés. e) **A** : une *dataset card*.
**À retenir** : on cherche sur le Hub, Kaggle, OpenML ou l'UCI, et on juge un dataset d'abord sur sa page : licence, provenance, versions.

### 29.Q8 — Les « nouveaux » datasets de 2018 : tâches, tailles et retraits
a) **9** millions d'images environ. b) **« AC »** : MS-Celeb-1M (2019) et *80 Million Tiny Images* (2020). c) **A** : Re-LAION-5B. d) **B** : recherche non commerciale seulement. e) **Vrai** : une copie ne lève ni le retrait ni ses raisons (labels injurieux, absence de consentement, contenus illégaux).
**À retenir** : avant d'utiliser un dataset célèbre, vérifier qu'il n'a pas été retiré, et pourquoi.

<a id="rappels"></a>

## 🔁 Rappels

### 29.R1 — Qui possède les droits d'une image de style ?
a) **2044** : le 1ᵉʳ janvier qui suit les 70 ans de la mort (1973 + 70 = 2043). b) **Faux** : la directive 2019/790 (art. 14) exclut la protection d'une reproduction fidèle d'une œuvre d'art visuel du domaine public. c) **Faux** : le droit d'auteur protège des œuvres, pas un style. d) **A** : CC BY 2.0, avec attribution. e) **1961** (1890 + 70 = 1960, puis le 1ᵉʳ janvier suivant).
**À retenir** : les images d'un dataset ont chacune leurs droits ; un dataset d'images « trouvé » en ligne n'est pas libre pour autant.

### 29.R2 — En RL, d'où viennent les données ?
a) **B**. b) **Vrai** : une autre politique visite d'autres états ; la distribution des données suit l'agent. c) **A** : des transitions $(s, a, r, s', \text{fin})$. d) **16**. e) **Faux** : les données dépendent de la politique en cours ; on évalue l'agent en le faisant jouer de nouveaux épisodes, avec d'autres graines.
**À retenir** : en RL, le dataset n'existe pas d'avance : l'agent le fabrique, et il change avec lui.

### 29.R3 — Biais d'échantillonnage : un dataset représente-t-il la population visée ?
a) **344**. b) **3** (Adélie, Chinstrap, Gentoo). c) **A**. d) **Faux** : trois îles d'une seule région, trois saisons, trois espèces sur une vingtaine. e) **A** : un décalage de distribution.
**À retenir** : un dataset représente la population d'où il vient ; ailleurs, il faut un jeu de test qui vient d'ailleurs.

<a id="papier"></a>

## ✏️ 🧮 Papier-crayon

### Ex 29.1 — Poids en mémoire de MNIST, Fashion-MNIST et CIFAR-10 ✏️
**Réponse** : a) **54,88 Mo** ; b) **219,52 Mo** ; c) **184,32 Mo** ; d) **0,737 Go** ; e) **318 877** images ; f) **Vrai**.
**Démarche** : a) $70\,000 \times 784 = 54\,880\,000$ octets. b) Quatre fois plus : $219\,520\,000$ octets. c) $60\,000 \times 32 \times 32 \times 3 = 184\,320\,000$ octets. d) $4 \times 184{,}32 \cdot 10^6 = 737{,}28 \cdot 10^6$ octets, soit 0,737 Go. e) Une image MNIST en `float32` occupe $784 \times 4 = 3\,136$ octets ; $10^9 / 3\,136 \approx 318\,877{,}55$, d'où 318 877 images entières. f) Même nombre d'images, même taille, même type. Dans ta copie : `uint8` divise la place par quatre sur le disque et en mémoire ; on ne convertit (et ne normalise) qu'au moment de former un mini-batch, ce qui ne garde en `float32` que quelques dizaines d'images à la fois.
**Erreurs fréquentes** : oublier les trois canaux de CIFAR-10 ; compter 8 octets par `float32` ; prendre 1 Mo = $2^{20}$ octets alors que l'énoncé fixe $10^6$ ; arrondir e) au plus proche.
**Variante** : le jeu d'entraînement d'ImageNet-1k (1 281 167 images) réduit à 224 × 224 × 3 en `uint8` occupe $1\,281\,167 \times 150\,528 \approx 192{,}9$ Go : il ne tient pas en mémoire, on le lit par morceaux depuis le disque.

### Ex 29.2 — Découpage stratifié et référence « classe majoritaire » ✏️
**Réponse** : a) **140** ; b) **30** ; c) **0,60** ; d) **0,90** ; e) **0,00** ; f) **Vrai**.
**Démarche** : a) et b) on applique 70 % et 15 % aux 200 exemples de C. c) Toujours prédire A : juste sur les 1 200 A, soit $1\,200 / 2\,000$. d) Sans jamais prédire C, on rate au moins les 200 C : au mieux $1\,800 / 2\,000$. e) Le modèle de c) ne trouve aucun C : recall $0 / 200$. f) C'est la définition. Dans ta copie : 0,85 ne bat la référence 0,60 que de 25 points et peut cacher un modèle qui ignore C ; on regarde le recall (et la précision, ou le F1) de C, et la matrice de confusion.
**Erreurs fréquentes** : appliquer les pourcentages au total (1 400 et 300) ; prendre la part de B pour c) ; croire qu'une accuracy élevée suffit sur des classes déséquilibrées.
**Variante** : avec un découpage 80/10/10, C a 160 exemples dans l'entraînement et 20 dans le test ; 20 exemples, c'est peu pour mesurer un recall (un exemple de plus ou de moins le change de 5 points).

### Ex 29.3 — Télécharger Open Images ou LAION : stockage et temps 🧮
**Réponse** : environ **2,7 To** ; **2,5 jours** à 100 Mbit/s ; **0,6 h** (36 minutes) à 10 Gbit/s ; **585 To** pour les images de LAION-5B ; **21 To** décodées à 1 024 × 768 et **1,35 To** réduites à 224 × 224.
**Démarche** :
1. $9 \cdot 10^6 \times 3 \cdot 10^5 = 2{,}7 \cdot 10^{12}$ octets.
2. 100 Mbit/s $= 1{,}25 \cdot 10^7$ octets/s ; $2{,}7 \cdot 10^{12} / 1{,}25 \cdot 10^7 = 216\,000$ s, soit 2,5 jours sans interruption.
3. 10 Gbit/s $= 1{,}25 \cdot 10^9$ octets/s : 2 160 s, soit 0,6 h (36 minutes). En pratique, les serveurs d'origine, le nombre de connexions et le disque limitent bien avant.
4. $5{,}85 \cdot 10^9 \times 10^5 = 5{,}85 \cdot 10^{14}$ octets. Distribuer des liens évite de redistribuer des œuvres protégées (le droit d'auteur reste aux propriétaires des images) et réduit la distribution à quelques centaines de gigaoctets de métadonnées. Mais un lien peut mourir (le dataset rétrécit avec le temps, et deux équipes n'ont plus les mêmes images) ou pointer vers un contenu remplacé, éventuellement illicite : la reproductibilité et le contrôle du contenu en souffrent (le retrait de 2023 en est un exemple).
5. $9 \cdot 10^6 \times 1\,024 \times 768 \times 3 \approx 2{,}1 \cdot 10^{13}$ octets, environ 21 To ; réduites, $9 \cdot 10^6 \times 150\,528 \approx 1{,}35 \cdot 10^{12}$ octets. Le JPEG compresse environ huit fois les images en pleine taille. Une copie réduite est plus petite que les JPEG d'origine, mais elle fige une résolution et un recadrage : on garde les originaux pour pouvoir changer de taille, recadrer au hasard (augmentation) ou passer à un modèle qui voit plus grand.
6. Sur un portable ou Colab : quelques centaines de mégaoctets à quelques gigaoctets, et un entraînement de quelques minutes à une heure. Si le dataset idéal est trop gros : un sous-ensemble stratifié, des images réduites une fois pour toutes, le mode *streaming* de `datasets`, ou des features pré-calculées par un réseau pré-entraîné.
**Erreurs fréquentes** : confondre bits et octets (un facteur 8) ; compter $1\,024^4$ octets par To ; croire qu'un débit annoncé est tenu pendant des jours.
**Variante** : Fashion-MNIST compressé (environ 30 Mo) se télécharge à 100 Mbit/s en $3 \cdot 10^7 / 1{,}25 \cdot 10^7 = 2{,}4$ s.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 29.4 — Présenter son projet final en cinq lignes 🗣️
**Réponse modèle** (un exemple ; ton projet sera différent) : (1) Aider les bénévoles d'une association de sciences participatives à trier automatiquement les photos de pièges photographiques, pour qu'ils ne regardent que celles où un animal apparaît. (2) Données : un sous-ensemble de 20 000 images d'un dataset public de pièges photographiques du Hugging Face Hub, licence CC BY 4.0, version figée par sa révision. (3) Entrée : une photo ; sortie : « vide » ou l'une de 10 espèces. (4) Réussite : un recall d'au moins 0,95 sur « animal présent », mesuré sur des pièges que le modèle n'a jamais vus ; référence à battre : un modèle qui prédit toujours « vide » et une régression logistique sur des features pré-calculées. (5) Risque : les pièges d'une même forêt se ressemblent ; un modèle peut reconnaître le décor plutôt que l'animal, d'où un découpage par piège.
**Ce qui compte** : un utilisateur et un besoin réels ; une source précise avec sa licence et sa taille ; une entrée et une sortie concrètes ; une métrique **et** une référence simple ; un risque nommé, avec la parade. Une présentation qui commence par « je vais utiliser un transformer » part du mauvais bout.

### Ex 29.5 — Retirés, restreints ou biaisés : que vérifier avant d'utiliser un dataset ? ⚖️
**Réponse modèle** :
1. Lire la licence (usage commercial, attribution, partage à l'identique) ; vérifier les droits sur le contenu lui-même (photos, textes d'auteurs) ; chercher des données personnelles (visages, noms, adresses) et la base de leur collecte (consentement, RGPD) ; retracer la provenance (qui a collecté, comment, quand) ; vérifier que le dataset n'a pas été retiré ou corrigé, et prendre la dernière version officielle ; examiner la représentativité (qui est présent, qui manque) ; lire la description des variables (une variable sensible, une variable construite sur un critère contestable) ; auditer la qualité (labels, doublons, fuites).
2. Boston : la licence n'est pas le problème ; la variable `B` est calculée à partir de la proportion de population noire du quartier, sous l'hypothèse d'un effet sur les prix. scikit-learn l'a retiré (1.2) et renvoie vers California Housing ou Ames Housing.
3. MS-Celeb-1M (2019) : 10 millions de photos de 100 000 personnes, collectées sans leur consentement. *80 Million Tiny Images* (2020) : des labels racistes et injurieux, hérités des catégories utilisées pour collecter les images, et des images impossibles à vérifier une à une. Un modèle déjà entraîné garde ce qu'il a appris : on documente son origine, on évalue ses biais, et on le réentraîne sur d'autres données si son usage le justifie ; on ne le diffuse pas comme si de rien n'était.
4. Non, sauf accord explicite des ayants droit : un filtre vendu est un usage commercial, et les photos sont des visages de personnes réelles (données personnelles, droit à l'image). Proposer un dataset sous licence commerciale, ou collecter des photos avec consentement.
5. Non : distribuer des liens évite de redistribuer les fichiers, mais celui qui télécharge les images les copie, et les liens peuvent pointer vers des contenus protégés ou illicites. LAION-5B a dû être retiré fin 2023 pour cette raison, et republié nettoyé en 2024.
6. Le plus souvent : le consentement des personnes, la conformité au RGPD de la collecte d'origine, l'absence d'erreurs systématiques dans les labels. On l'écrit dans la section « Limites et biais » de la data card : ce qui a été vérifié, comment, et ce qui ne l'a pas été.
**Ce qui compte** : distinguer le droit (licence, contenu, personnes), l'histoire (provenance, retraits) et la qualité (représentativité, labels) ; savoir qu'un retrait ne s'efface pas avec une copie ; dire honnêtement ce qu'on n'a pas pu vérifier.

### Ex 29.6 — Lire Gebru et coll., « Datasheets for Datasets » 📄
**Réponse modèle** :
1. Dans l'industrie électronique, chaque composant, même le plus simple, est livré avec une fiche technique (*datasheet*) : caractéristiques, conditions de fonctionnement, résultats de tests, usages recommandés. Un dataset devrait avoir la même chose : on ne monte pas un composant sans savoir dans quelles conditions il marche.
2. Motivation (« Pour quoi le dataset a-t-il été créé ? ») ; composition (« Que représentent les exemples : des documents, des photos, des personnes ? ») ; processus de collecte (« Comment les données de chaque exemple ont-elles été obtenues ? ») ; prétraitement, nettoyage et annotation (« A-t-on fait un prétraitement, et les données brutes ont-elles été gardées ? ») ; usages (« À quelles autres tâches pourrait-il servir ? ») ; distribution (« Sera-t-il partagé hors de l'organisation qui l'a créé ? ») ; maintenance (« Qui l'héberge et l'entretient ? »).
3. Aux deux. Pour ceux qui créent le dataset : réfléchir soigneusement à chaque étape de sa création, de sa distribution et de son entretien, y compris aux hypothèses et aux risques. Pour ceux qui l'utilisent : disposer de l'information nécessaire pour décider en connaissance de cause s'il convient à leur usage.
4. Les auteurs écrivent que remplir une datasheet n'est pas destiné à être automatisé : une génération automatique irait contre l'objectif de réflexion. Exemple de question de jugement : « Y a-t-il des tâches pour lesquelles le dataset ne devrait pas être utilisé ? » ou « La collecte a-t-elle fait l'objet d'une évaluation de son impact sur les personnes concernées ? ».
5. Pour MNIST (`data/cards/mnist.md`) : que représentent les exemples (des chiffres écrits à la main, numérisés et normalisés en 28 × 28) ; combien y en a-t-il (70 000, découpés 60 000 / 10 000) ; les données contiennent-elles des informations sur des personnes (les scripteurs sont des employés du recensement américain et des lycéens, non identifiés). Ce qui manque souvent : la façon dont les scripteurs ont été répartis entre entraînement et test, les erreurs de labels connues, et ce qui a été fait des données brutes.
6. Une datasheet ne suffit pas à éliminer les biais (les repérer demande souvent des informations démographiques indisponibles, parfois pour des raisons de vie privée) ; elle coûte du travail à ceux qui créent le dataset, et les organisations doivent l'intégrer à leurs processus ; les questions ne conviennent pas à tous les domaines ni à tous les contextes (une publication académique, un dataset interne) ; les datasets qui évoluent demandent une datasheet par version ; et documenter des données sur des personnes demande parfois des compétences d'autres disciplines (sociologie, anthropologie).
**Ce qui compte** : retrouver les sept sections et les deux publics ; comprendre que la datasheet est un outil de réflexion, pas un formulaire automatique ; voir que la data card de 🛠️ 29.15 en est une version courte.

<a id="entretien"></a>

## 💼 Entretien

### 29.E1 — Comment choisir et valider un dataset pour un nouveau projet ? 💼
**Réponse modèle en 60 secondes** : « D'abord je définis la population sur laquelle le modèle servira et la métrique qui compte, avec une référence simple à battre. Ensuite je cherche l'existant : un modèle pré-entraîné à ajuster, sinon un dataset public sur le Hugging Face Hub, Kaggle ou OpenML ; je ne collecte qu'en dernier. Pour chaque candidat, je vérifie la licence, la provenance et l'historique, notamment les retraits. Puis j'audite : valeurs manquantes, y compris codées par des nombres, doublons, quasi-doublons entre découpages, colonnes qui trahissent la cible, labels douteux, déséquilibre des classes, et représentativité par rapport à ma population. Je découpe tôt, par groupes s'il le faut, et je garde un jeu de test venu d'ailleurs pour mesurer le décalage. Enfin je fige : une version, une somme de contrôle, une data card. »
**Ce qui compte** : partir du besoin, pas du dataset ; vérifier le droit avant la qualité ; auditer avant d'entraîner ; finir par la reproductibilité.
**Relances possibles** :
- « Et si aucun dataset public ne convient ? » → Collecter peu mais bien : un protocole d'annotation écrit, deux annotateurs sur une partie des exemples pour mesurer leur accord, et un modèle pré-entraîné ajusté sur ces quelques centaines d'exemples.
- « Comment savoir si le dataset est représentatif ? » → Comparer ses distributions (classes, sous-groupes, conditions d'acquisition) à celles des données d'usage, et mesurer la performance par sous-groupe ; un test sur des données d'une autre source révèle le décalage.

### 29.E2 — Qu'est-ce qu'une data card et que doit-elle contenir ? 💼
**Réponse modèle en 60 secondes** : « C'est la notice d'un dataset. Elle dit d'où viennent les données (qui les a collectées, comment, quand), ce qu'elles contiennent (nombre d'exemples, variables, classes, valeurs manquantes), sous quelle licence, pour quel usage elles ont été prévues et pour quels usages elles ne conviennent pas, leurs limites et leurs biais connus, et comment les obtenir à l'identique : une adresse, une version, une somme de contrôle. L'idée vient des *datasheets* de Gebru et coll. ; sur le Hugging Face Hub, c'est la *dataset card*, et Croissant en donne une version lisible par machine. Elle sert à celui qui crée le dataset, qui doit réfléchir à ces questions, et à celui qui l'utilise, qui peut décider en connaissance de cause. »
**Ce qui compte** : les rubriques, les deux publics, les références ; dire qu'une data card se maintient avec les versions du dataset.
**Relances possibles** :
- « Quelle est la différence avec une *model card* ? » → La model card (Mitchell et coll., 2019) décrit un modèle : usage prévu, performances par sous-groupe, limites ; elle renvoie aux data cards de ses données d'entraînement et d'évaluation.
- « Est-ce une obligation ? » → Pour les fournisseurs de modèles d'IA à usage général dans l'Union européenne, le règlement sur l'IA (art. 53) impose de publier un résumé suffisamment détaillé des contenus d'entraînement ; ailleurs, c'est une bonne pratique, souvent exigée par les clients ou les conférences.

### 29.E3 — Comment détecter une fuite de données entre entraînement et test ? 💼
**Réponse modèle en 60 secondes** : « 99,5 %, c'est d'abord un signal d'alerte : je le compare au niveau attendu pour la tâche. Je cherche ensuite trois familles de fuites. Les mêmes exemples des deux côtés : doublons exacts, quasi-doublons (photos en rafale, plusieurs images d'un patient), que je trouve par hachage ou par plus proches voisins entre les découpages ; la parade est un découpage par groupes. Une variable qui contient la réponse : je regarde l'importance des variables et je me demande lesquelles seront connues au moment de prédire. Une étape ajustée sur tout le dataset : normalisation, sélection de variables, choix d'hyperparamètres sur le test ; tout doit être ajusté sur l'entraînement, dans un pipeline. Enfin je teste sur des données d'une autre source ou d'une période plus récente : si le score s'effondre, il y avait une fuite ou un décalage. »
**Ce qui compte** : les trois familles ; une méthode de détection pour chacune ; le réflexe du pipeline et du découpage par groupes.
**Relances possibles** :
- « Un découpage temporel, quand ? » → Dès que le modèle prédit le futur : on entraîne sur le passé et on teste sur la période suivante ; un découpage au hasard laisserait le modèle voir l'avenir.
- « Comment repérer des quasi-doublons d'images ? » → Un hachage perceptuel (qui résiste au redimensionnement et à la compression) ou la distance entre les features d'un réseau pré-entraîné, avec un seuil vérifié à l'œil sur quelques paires.

### 29.E4 — Données publiques et usage commercial : que vérifier ? 💼
**Réponse modèle en 60 secondes** : « Public veut dire accessible, pas libre. Je lis d'abord la licence : CC BY, CC0 ou MIT autorisent un produit, avec attribution pour CC BY ; NC ou "recherche seulement" l'interdisent ; ShareAlike impose de partager les dérivés sous la même licence. Ensuite les droits sur le contenu lui-même : des photos ou des textes peuvent appartenir à d'autres que celui qui a publié le dataset. Puis les personnes : s'il y a des données personnelles, la collecte d'origine doit être conforme au RGPD. Je vérifie l'histoire du dataset : retraits, versions corrigées. Je documente tout dans une data card, avec la version utilisée. Et quand le doute persiste, je demande à un juriste avant d'entraîner, pas après. »
**Ce qui compte** : séparer licence, droits sur le contenu et données personnelles ; ne pas oublier l'histoire du dataset ; garder une trace.
**Relances possibles** :
- « Et un modèle entraîné sur des données NC ? » → La question est discutée et dépend des licences et des pays ; par prudence, on traite le modèle comme soumis aux mêmes restrictions, et on documente l'origine de ses données.
- « Et les données publiques du web, en Europe ? » → L'exception de fouille de textes et de données (directive 2019/790, art. 4) permet de les copier pour l'entraînement, sauf opposition du titulaire exprimée de façon appropriée (lisible par machine pour un contenu en ligne) ; elle ne lève ni le RGPD ni les autres droits.

<a id="notebook"></a>

## Exercices du notebook

### Ex 29.7 — Les datasets intégrés de scikit-learn, et une affirmation du livre à vérifier 📦
**Réponse** : a) **1 797** ; b) **0,373** (212 tumeurs malignes sur 569) ; c) **False**.
**Démarche** : la table se construit en chargeant chaque dataset (`return_X_y=True`) et en mesurant `X.shape` et `np.unique(y)` : Iris (150, 4, 3 classes), Digits (1 797, 64, 10), Wine (178, 13, 3), Breast Cancer (569, 30, 2), Diabetes (442, 10, régression). Pour `load_boston`, scikit-learn ≥ 1.2 n'a plus la fonction, mais son module définit un `__getattr__` qui lève une `ImportError` explicative ; `getattr` dans un `try` qui attrape `AttributeError` **et** `ImportError` répond donc `False`. Dans tes notes : le message renvoie vers California Housing et Ames Housing (`fetch_openml(name="house_prices")`), et explique le problème de la variable `B`.
**Erreurs fréquentes** : utiliser `hasattr` (qui laisse passer l'`ImportError`) ; recopier les tailles de la description (`DESCR`) au lieu de les mesurer ; compter les classes de Diabetes, une régression.
**Variante** : sur Wine, la classe minoritaire (la classe 2, 48 vins sur 178) représente 0,270.

### Ex 29.8 — Prédire : le même modèle sur MNIST puis sur Fashion-MNIST 🔮
**Réponse** : a) **B** (accuracy 0,916) ; b) **B** (0,812 sur Fashion-MNIST, 10,3 points de moins) ; c) **A** (0,116, presque le hasard).
**Démarche** : avec deux epochs sur 10 000 images, un petit réseau dense reste sous les 98 % habituels sur MNIST. Sur Fashion-MNIST, les classes du haut du corps (T-shirt, pull, manteau, chemise) se confondent : le graphique montre les recalls les plus bas pour la chemise et le pull. Le modèle MNIST appliqué à des vêtements répond quand même, avec une confiance moyenne de 0,676 : il ne sait pas dire « je ne connais pas ». Le livre recommande Fashion-MNIST comme marche suivante : même code, tâche plus dure, et des erreurs que l'on peut encore regarder une par une.
**Erreurs fréquentes** : prédire les 98 % de MNIST sans tenir compte du budget d'entraînement ; croire qu'un modèle de chiffres « reconnaîtra » un peu les vêtements.
**Variante** : avec cinq epochs (`train_mlp(..., epochs=5)`), les accuracies passent à 0,938 sur MNIST et 0,827 sur Fashion-MNIST : l'écart reste d'environ 11 points.

### Ex 29.9 — Prédire la chute d'accuracy en changeant de source de chiffres 🔮
**Réponse** : a) **C** (accuracy 0,198) ; b) **True** (confiance moyenne 0,761, contre 0,880 sur le test de MNIST).
**Démarche** : agrandi tel quel, un chiffre de scikit-learn occupe toute l'image de 28 × 28, avec des traits épais et flous ; ceux de MNIST tiennent dans un carré de 20 × 20 centré. Les pixels moyens le disent : 77,8 contre 33,8. Le réseau compare des pixels à des positions fixes, et ses repères ne tiennent plus : l'accuracy tombe à 0,198, à peine le double du hasard. Pourtant sa confiance reste élevée : un softmax rend toujours une distribution, et rien, dans l'entraînement, n'a appris au réseau à douter hors de sa distribution. En production, c'est le pire cas : des erreurs nombreuses et sûres d'elles, qu'aucun seuil de confiance ne filtre.
**Erreurs fréquentes** : croire que « mêmes classes, même format » suffit ; prendre la confiance pour une mesure de justesse.
**Variante** : sur 1 000 images de bruit uniforme, le même modèle a une confiance moyenne de 0,661 : il est presque aussi sûr de lui face à du bruit que face à des chiffres décalés.

### Ex 29.10 — torchvision.datasets et Hugging Face datasets : mêmes données, deux API 📦
**Réponse** : les deux sources sont identiques : mêmes formes, mêmes labels dans le même ordre, mêmes pixels (`max_diff = 0`). Seuls les noms des classes diffèrent dans leur écriture (`"T - shirt / top"` sur le Hub, `"T-shirt/top"` dans le workbook).
**Démarche** : `np.stack` des images PIL converties par `np.asarray`, labels en `int64`. Dans `compare_10`, la soustraction se fait en entiers signés : en `uint8`, `3 - 10` donne 249 et un écart de 7 passerait pour un écart de 249 (ou un pixel plus clair pour un pixel plus sombre). Dans tes notes : fixer `revision=` garantit que le même code chargera les mêmes données dans un an, même si le dépôt du dataset change ; `torchvision.datasets.FashionMNIST(root, train=False, download=True)` rend un `Dataset` dont chaque élément est un couple (image PIL, entier), transformable par `transform=transforms.ToTensor()`.
**Erreurs fréquentes** : soustraire des `uint8` ; comparer des labels de longueurs différentes avec `==` (avertissement ou erreur selon la version de NumPy) ; laisser `X` en `float` ou en liste.
**Variante** : `HF_10.with_format("numpy")[:]["image"]` rend directement un tableau `(10000, 28, 28)` en `uint8`, identique à `hf_to_numpy_10(HF_10)[0]`.

### Ex 29.11 — Explorer le Hugging Face Hub et OpenML : chercher, filtrer, lire la licence 📦
**Réponse** : sur le Hub simulé, les trois plus téléchargés sont `someone/penguins-images` (3 400, cc-by-nc-4.0), `demo/penguins-clean` (950, cc0-1.0) et `lab/penguins-gps` (610, odbl) ; le résumé d'OpenML de Palmer Penguins : version 1, licence CC0, 344 lignes, 7 colonnes, cible `species`.
**Démarche** : `list_datasets(search=..., limit=..., sort="downloads")` trie avant de couper : sans `sort`, on garderait les `limit` premiers trouvés, pas les plus téléchargés. La licence se lit dans les tags `license:xxx`. Côté OpenML, `fetch_openml` rend un `Bunch` dont `details` donne le nom, la version, la licence, la cible par défaut et l'adresse d'origine. En direct (le 30 octobre 2026), la recherche `"penguins"` donnait en tête `SIH/palmer-penguins` (licence cc0-1.0) ; d'autres résultats avaient une licence `other`, `unknown` ou aucune. Dans tes notes : seules CC0, CC BY, MIT, Apache ou ODbL (avec ses conditions de partage) permettent un produit ; un dataset sans licence n'accorde **aucun** droit par défaut : on ne l'utilise pas sans l'accord de son auteur.
**Erreurs fréquentes** : trier soi-même après un `limit` (on trie les mauvais) ; lire la licence dans `card_data`, souvent vide dans les résultats d'une recherche ; confondre « pas de tag » et « libre ».
**Variante** : `HfApi().dataset_info("zalando-datasets/fashion_mnist")` rend la révision courante (`.sha`, `531be5e2…` à cette date) et la licence (`card_data["license"]`, ici la liste `["mit"]`) : c'est ce qu'on note dans une data card.

### Ex 29.12 — Profil express d'un dataset inconnu 📈
**Réponse** : a) **"occupation"** (2 809 manquants) ; b) **52** ; c) **0,76**.
**Démarche** : `value_counts()` ignore les manquants et trie par fréquence, ce qui donne le mode et sa part en une ligne. `workclass` suit de près (2 799 manquants : ces 2 799 personnes sont toutes aussi sans `occupation`), puis `native-country` (857). Les colonnes presque constantes : `capital-loss` (0 pour 95 % des personnes), `capital-gain` (0 pour 92 %) et `native-country` (États-Unis pour 91 % des valeurs présentes) ; elles ne sont pas inutiles pour autant, car leurs rares valeurs non nulles sont très informatives sur le revenu. Les 52 doublons sont probablement des personnes différentes aux mêmes réponses (le recensement n'a pas d'identifiant) : on les garde, mais on veille à ce qu'ils ne se répartissent pas entre entraînement et test. Un modèle qui prédit toujours `<=50K` a une accuracy de 0,76 : c'est la référence à battre.
**Erreurs fréquentes** : compter les manquants avec `== "?"` alors que `load_adult` les a recodés en `NaN` ; inclure les manquants dans la part du mode ; doubler les doublons (`duplicated(keep=False)` marque les deux lignes d'une paire).
**Variante** : sur Palmer Penguins (`wb.datasets.load_penguins()`), la colonne `sex` a le plus de manquants (11), il n'y a aucun doublon, et l'espèce majoritaire (Adélie) représente 0,442 des manchots.

### Ex 29.13 — Un score trop beau pour être vrai : doublons entre entraînement et test 🐛
**Réponse** : a) **« BCE »** ; b) **500** : chacune des 500 lignes de test a au moins une copie de son vêtement dans l'entraînement ; la version corrigée choisit k = 3 et obtient **0,728**, contre 1,000 annoncé.
**Démarche** : B découpe les images au hasard, sans tenir compte de `GROUPS_13` : les quatre photos d'un vêtement se répartissent des deux côtés, et le plus proche voisin d'une image de test est sa jumelle (avec quatre copies, la probabilité que les quatre tombent dans le test est de $0{,}25^4 \approx 0{,}4\,\%$, soit environ 2 vêtements attendus sur 500 ; ici, aucun). C'est la fuite qui explique presque tout l'écart. C ajuste la PCA sur toutes les lignes, test compris : les axes principaux connaissent le test. E choisit k sur le score de test, qui sert ensuite de mesure finale (ligne G) : le score est optimiste. F et G sont correctes en elles-mêmes : réentraîner avec le k choisi, puis mesurer. La correction découpe par groupes (`GroupShuffleSplit`), choisit k sur une validation elle-même découpée par groupes, ajuste la PCA sur les seules lignes qui entraînent, puis réentraîne et mesure **une fois** sur le test. Le piège des labels tirés au hasard le confirme : rien n'est apprenable, et la version honnête tombe à 0,092, au niveau du hasard, quand la version fautive affiche 0,998. Les espions de la vérification contrôlent le reste : aucune ligne de test dans un `fit` de la PCA, un seul score sur le test, une validation sans vêtement commun avec les lignes qui entraînent.
**Erreurs fréquentes** : désigner F (elle réentraîne avec le k déjà choisi : le choix fautif est fait en E) ; désigner seulement B (les fuites par la PCA et par le choix de k sont plus petites, mais ce sont des fuites) ; désigner A (diviser par 255 n'apprend rien des données) ; découper la validation au hasard à l'intérieur de l'entraînement ; compter les groupes partagés au lieu des lignes de test.
**Variante** : avec deux copies par vêtement (`make_near_duplicates_13(copies=2)`), la version fautive annonce encore 0,936 (190 des 250 lignes de test ont leur jumelle dans l'entraînement), la version corrigée 0,736.

### Ex 29.14 — Décalage de distribution : un modèle MNIST face aux chiffres de scikit-learn 🔬
**Réponse** : accuracy de `MODEL_M` : **0,198** (naive), **0,576** (box), **0,661** (center) ; après fine-tuning sur 1, 5 et 20 chiffres étiquetés par classe : **0,793**, **0,892** et **0,932** ; référence entraînée sur les chiffres de scikit-learn : **0,958**.
**Démarche** : reproduire la recette de MNIST rapporte le plus : ramener le chiffre dans un carré de 20 × 20 fait passer de 0,198 à 0,576, le centrer par son centre de masse ajoute 8,5 points. Le reste du décalage tient aux images elles-mêmes (8 × 8 pixels agrandis, donc flous ; d'autres scripteurs, d'autres stylos) : c'est un décalage de covariables, les labels gardent le même sens. Cinq exemples étiquetés par classe, soit 50 images, récupèrent l'essentiel ; vingt approchent la référence. Dans un projet : documenter la source et le prétraitement des données d'entraînement, appliquer **le même** prétraitement en production, et prévoir quelques exemples étiquetés de la cible pour mesurer et ajuster.
**Erreurs fréquentes** : centrer avant d'avoir réduit à 20 × 20 (le chiffre déborde) ; arrondir et convertir en `uint8` avant le décalage (perte de précision) ; mesurer le fine-tuning sur les images qui ont servi à l'ajuster.
**Variante** : sans meilleure préparation (méthode `"naive"`), les mêmes cinq chiffres étiquetés par classe (même tirage que l'expérience) donnent 0,826, contre 0,892 avec `"center"` : bien préparer les données reste rentable même quand on peut ajuster le modèle.

### Ex 29.15 — Data card et script de téléchargement vérifié pour le projet final 🛠️
**Réponse** : la solution exécutée lit le fichier par morceaux de 1 Mio, télécharge dans un `.part`, vérifie, puis renomme ; sa data card d'exemple décrit Palmer Penguins (`data/penguins.csv`), avec sa somme SHA-256.
**Démarche** : une somme de contrôle se calcule en flux (`hashlib.sha256`, `update` par morceau, `hexdigest`), sans charger le fichier. Le `.part` garantit que `dest` n'existe que complet et vérifié : un téléchargement interrompu ou corrompu ne laisse jamais un faux fichier que la prochaine exécution prendrait pour bon. Le test « déjà présent » vérifie la somme, pas seulement l'existence du fichier. Dans tes notes : la somme protège contre la corruption et contre un contenu qui change à la même adresse (le script s'arrête au lieu d'entraîner sur autre chose) ; elle ne protège pas contre un dataset retiré : il faut alors une copie archivée, dont on vérifiera la même somme, et une ligne dans la data card. Pour un dataset du Hub, la révision (un commit) joue le même rôle pour tout le dépôt.
**Erreurs fréquentes** : `f.read()` d'un coup ; écrire directement dans `dest` ; oublier de supprimer le `.part` en cas d'erreur ; lever une `AssertionError` (désactivée par `python -O`) au lieu d'une `ValueError` ; une data card sans limites ni licence.
**Variante** : `huggingface_hub.hf_hub_download(repo_id, filename, repo_type="dataset", revision=sha)` fait la même chose pour un fichier du Hub : il télécharge la version demandée et la garde en cache ; le chemin rendu se passe ensuite à `sha256_15` pour la data card.

### Ex 29.16 — Le dataset piégé : trouver les huit défauts 🏆
**Réponse** : a) **13** doublons exacts ; b) **17** quasi-doublons dans le test ; c) **"acidite"** (−999) ; d) **"pays"** (toujours « France ») ; e) **"code_qualite"** (sa lettre, M, R ou C, recopie le fruit) ; f) **"diametre_cm"** (39 valeurs en millimètres, de 58,5 à 100,3) ; g) **11** labels invalides ; h) **"coing"** (24 lignes, 2 %).
**Démarche** : `duplicated()` pour a) ; un `merge` des lignes de test avec celles d'entraînement sur toutes les colonnes sauf `id_lot`, `split` et `poids_g`, puis les poids à 1 g près, pour b) : 18 paires, mais une ligne de test a deux jumelles (sa source est aussi un doublon exact), d'où 17 lignes ; `describe()` pour c) et f) (un minimum de −999, un maximum de 100,3 cm) ; `nunique()` pour d) ; `pd.crosstab(FRUITS_16["code_qualite"].str[1], FRUITS_16["fruit"])` pour e) ; `value_counts()` pour g) (`pome`, `POMME`, `pommes`, `Poire`, `poir`, `"poire "` avec une espace) et h). Le nettoyage garde 1 200 lignes. La forêt aléatoire obtient 0,991 sur les données brutes et 0,963 sur les données nettoyées : le score brut était gonflé par la fuite du code qualité et par les quasi-doublons, faciles à reconnaître ; le score nettoyé est plus bas mais honnête. Dans la data card : les mesures sont simulées, la classe coing est rare (24 exemples : son recall sera mal estimé), les diamètres ont été convertis selon une règle à écrire.
**Erreurs fréquentes** : compter 26 doublons (les deux lignes de chaque paire) ; compter 18 quasi-doublons (des paires, pas des lignes) ; prendre `id_lot` pour la colonne constante ; compter 9 labels invalides (en oubliant `"poire "` et ses espaces, que `value_counts()` affiche pourtant à part) ; modifier `FRUITS_16` en place.
**Variante** : avec une autre graine (`make_trap_dataset(seed=7)`), les défauts sont les mêmes mais certains nombres changent : 13 doublons, 17 quasi-doublons, 38 diamètres en millimètres et 12 labels invalides (une ligne mal écrite a été recopiée).
