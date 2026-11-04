# B7 · Du notebook à la production — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### B7.Q1 — Les pièges du notebook : état caché et ordre d'exécution

<details><summary>Indice 1</summary>

Relis « Ce que cache un notebook » et la fin de « pyproject.toml et installation éditable » (§207.1).

</details>
<details><summary>Indice 2</summary>

Le noyau garde les variables dans l'ordre où **tu** as exécuté les cellules ; une cellule supprimée a pu laisser ses variables en mémoire, jusqu'au redémarrage. « Redémarrer et tout exécuter » repart d'un noyau vide et suit l'ordre du fichier. Un module déjà importé n'est pas relu par le noyau, même si le fichier change.

</details>
<details><summary>Indice 3</summary>

a) Suis $x$ : $2$, puis $20$ (cellule 2), affichage $21$ (cellule 3), puis $200$ (cellule 2 encore), affichage **201**. Pour b), après le redémarrage, quelle cellule définit encore `scale` ? Pour c), que contient un `.ipynb` à côté du code ? Pour d), qui garde l'ancienne version du module : le lien éditable, ou le noyau ? Pour e), supprimer une cellule exécute-t-il quoi que ce soit dans le noyau ?

</details>

### B7.Q2 — src/, tests/, configs/, scripts/ : où va quoi ?

<details><summary>Indice 1</summary>

Relis « Une structure de dépôt » et « pyproject.toml et installation éditable » (§207.1).

</details>
<details><summary>Indice 2</summary>

`src/cifarprod/` ne contient que du code importé par l'entraînement et le service ; `tests/`, les tests que lance pytest ; `configs/`, les réglages ; `scripts/`, ce qu'on lance à la main. Les données et les sorties des runs restent hors de git. Avec le layout `src/`, le paquet n'est importable qu'une fois installé. `[project.scripts]` relie un nom de commande à une fonction.

</details>
<details><summary>Indice 3</summary>

a) **A** : la fonction est importée par deux programmes, elle appartient au paquet. Pour b), ce fichier est-il importé, ou lancé une fois ? Pour c), est-ce du code ou un réglage ? Pour d), où pytest cherche-t-il ? Pour e), que dit le paragraphe qui suit l'arborescence des données et des runs ? Pour f), depuis la racine, le dossier `cifarprod` est-il sous la main de Python sans installation ? Pour g), à quoi sert la ligne `cifarprod = "cifarprod.cli:main"` ?

</details>

### B7.Q3 — print ou logging : niveaux, destinations et messages utiles

<details><summary>Indice 1</summary>

Relis « Le logging plutôt que print » et « Les codes de sortie » (§207.2).

</details>
<details><summary>Indice 2</summary>

Les niveaux, du plus détaillé au plus grave : `DEBUG`, `INFO` (les étapes normales), `WARNING` (inattendu, on continue), `ERROR`, `CRITICAL`. Un niveau laisse passer les messages de ce niveau et des niveaux plus graves. La configuration se fait une fois, dans le point d'entrée. Un bon message donne des valeurs vérifiables. Un programme qui va au bout sans `sys.exit` rend 0.

</details>
<details><summary>Indice 3</summary>

a) **B** : une fin d'epoch est une étape normale. Pour b), le programme continue, mais quelque chose d'attendu manque : quel niveau ? Pour c), compte les messages de niveau `WARNING` ou plus grave. Pour d), où se configurent les handlers, et qu'arriverait-il si chaque module les configurait ? Pour e), lequel des trois messages contient un fait que l'on peut vérifier ? Pour f), le `except` empêche-t-il le script d'aller au bout ?

</details>

### B7.Q4 — Ce qui va dans la configuration, et ce qui ne doit jamais y aller

<details><summary>Indice 1</summary>

Relis le début de 207.3, le rappel outil sur PyYAML, puis « Variables d'environnement et secrets » (§207.9).

</details>
<details><summary>Indice 2</summary>

La configuration se versionne et se recopie dans les journaux : elle contient les réglages d'un run, jamais un secret. PyYAML suit la version 1.1 du format : une notation scientifique sans point n'est pas un nombre, des guillemets font une chaîne. Une image Docker garde chacune de ses couches ; git garde chaque commit. Une dataclass ne vérifie pas les types : c'est le rôle de `__post_init__`.

</details>
<details><summary>Indice 3</summary>

a) Le learning rate (A) est un réglage : il y va ; pour B, C et D, demande-toi si la valeur peut être lue par toute personne qui clone le dépôt. Pour b), cherche la ligne qui a un point avant l'exposant et pas de guillemets. Pour c), relis le `except (OSError, ValueError)` de `main`. Pour d), que montre `docker history` ? Pour e), un nouveau commit efface-t-il les anciens ? Pour f), que dit le commentaire de `__post_init__` dans `TrainConfig` ?

</details>

### B7.Q5 — Même graine, résultats différents : les sources de non-déterminisme

<details><summary>Indice 1</summary>

Relis « Les graines et le déterminisme » et l'encadré 🕰️ qui le suit (§207.4).

</details>
<details><summary>Indice 2</summary>

Python, NumPy et PyTorch ont chacun leur générateur ; `torch.manual_seed` ne fixe que ceux de PyTorch. Sur GPU, des threads accumulent dans un ordre qui varie, et l'addition des flottants n'est pas associative. `warn_only=True` remplace l'erreur par un avertissement. Sans `generator`, le `DataLoader` tire son ordre dans le générateur global de PyTorch.

</details>
<details><summary>Indice 3</summary>

a) **C** : `np.random.rand` tire dans le générateur de NumPy, que `torch.manual_seed` ne touche pas. Pour b), deux sommes des mêmes nombres dans un autre ordre sont-elles égales au dernier chiffre ? Pour c), que change `warn_only=True` à la `RuntimeError` ? Pour d), calcule les deux expressions dans une console Python. Pour e), où en est le générateur global au moment où le `DataLoader` tire son ordre ?

</details>

### B7.Q6 — requirements.txt, fichier de verrouillage, pyproject.toml : qui fige quoi ?

<details><summary>Indice 1</summary>

Relis « pyproject.toml et installation éditable » (§207.1) et « Les versions figées et l'empreinte des données » (§207.4).

</details>
<details><summary>Indice 2</summary>

Les **bornes** de `pyproject.toml` disent ce que le code supporte ; un **fichier de verrouillage** dit ce qui a marché, version exacte par version exacte, dépendances indirectes comprises. Un projet qu'on livre se verrouille ; une bibliothèque qu'on publie garde des bornes larges. Un commit ne contient que ce qui a été commité.

</details>
<details><summary>Indice 3</summary>

a) **C** : `>=2.5` est une borne inférieure, ce que le code accepte au moins. Pour b), lequel des trois fichiers est écrit par `uv lock` ? Pour c), une bibliothèque qui épinglerait `numpy==2.1.3` pourrait-elle s'installer à côté d'une autre qui demande `numpy>=2.2` ? Pour d), où est le code modifié qui a servi au run ? Pour e), un verrou dit-il « au moins » ou « exactement » ?

</details>

### B7.Q7 — Que tester dans un projet de machine learning ?

<details><summary>Indice 1</summary>

Relis « Quatre familles de tests » et « Une grille : le ML Test Score » (§207.6).

</details>
<details><summary>Indice 2</summary>

Tests unitaires (formes, prétraitement, configuration), tests de données (schéma, équilibre, recouvrement par empreinte), overfitting d'un mini-batch (la boucle sait-elle apprendre ?), entraînement fumée (tout va-t-il au bout ?). Le ML Test Score compte des points par famille, puis résume les quatre sommes par un seul nombre, le plus prudent.

</details>
<details><summary>Indice 3</summary>

a) **B** : sans `zero_grad`, les gradients s'accumulent et le réseau n'arrive plus à apprendre par cœur 16 images. Pour b), quel test regarde le contenu des images de deux jeux ? Pour c), relis la puce « bout en bout » : que juge l'entraînement fumée, et que ne juge-t-il pas ? Pour d), une loss qui ne baisse pas empêche-t-elle un script de rendre 0 ? Pour e), quelle somme est la plus prudente des quatre ?

</details>

### B7.Q8 — Artefact de modèle : les poids seuls ne suffisent pas

<details><summary>Indice 1</summary>

Relis « L'artefact du modèle » (§207.7) et, au ch. 23, §23.9.2 (« Les poids seuls »).

</details>
<details><summary>Indice 2</summary>

L'artefact contient tout ce qu'il faut pour **prédire** sans le notebook : poids, architecture, prétraitement exact (appris sur l'entraînement), classes dans l'ordre des sorties, traçabilité (version, run, empreinte des données). Ce qui sert à **reprendre** un entraînement n'en fait pas partie. `load_state_dict` vérifie les noms et les formes. Les métadonnées de safetensors n'acceptent que des chaînes.

</details>
<details><summary>Indice 3</summary>

a) A (les statistiques de normalisation) y va : le service doit normaliser exactement comme l'entraînement ; pour B, C et D, demande-toi si la prédiction en dépend. Pour b), une couche `Conv2d(3, 16, …)` peut-elle recevoir un poids de forme `(12, 3, 3, 3)` ? Pour c), quel type a la valeur `1` ? Pour d), que devient une empreinte quand un seul pixel change ? Pour e), à quoi sert de savoir, pour une prédiction donnée, quel artefact l'a produite ?

</details>

### B7.Q9 — Latence p50, p95 et débit : ce que ressent l'utilisateur

<details><summary>Indice 1</summary>

Relis « Latence, percentiles et débit » (§207.7), dont la loi de Little.

</details>
<details><summary>Indice 2</summary>

Une moyenne pèse chaque requête ; le p95 est la valeur sous laquelle tombent 95 % des requêtes triées. Débit $= \frac{\text{taille du batch}}{\text{latence du batch}}$ ; loi de Little $L = \lambda \times W$, avec $W$ en secondes si $\lambda$ est par seconde. Un batch amortit les coûts fixes, mais chaque image attend la fin du sien.

</details>
<details><summary>Indice 3</summary>

a) $\frac{900 \times 10 + 100 \times 200}{1\,000} = 29$ ms. Pour b), triées, quelle valeur occupe la position $0{,}95 \times 999$ ? Pour c), imagine un service où toutes les requêtes ont la même durée. Pour d), calcule $60 \times 0{,}150$. Pour e), calcule $64 / 0{,}4$. Pour f), que gagne-t-on et que paie-t-on en groupant ?

</details>

### B7.Q10 — torch.export, ONNX, TorchScript : lequel choisir aujourd'hui ?

<details><summary>Indice 1</summary>

Relis « torch.export et ONNX » et l'encadré 🕰️ « exporter un modèle PyTorch » (§207.8).

</details>
<details><summary>Indice 2</summary>

`torch.export` fige le graphe et les poids dans un `.pt2`, avec des gardes sur les formes ; ONNX se sert sans PyTorch ; TorchScript est déprécié ; `torch.compile` accélère un modèle dans un programme Python, sans fichier à servir. Un graphe figé ne contient pas de `if` Python sur une valeur des données.

</details>
<details><summary>Indice 3</summary>

a) **B** : TorchScript est déprécié dans la 2.11, et un pickle de l'objet demande le code de sa classe au chargement. Pour b), quel format se lit sans PyTorch, depuis d'autres langages ? Pour c), que produit `torch.compile` : un fichier, ou un modèle plus rapide dans le même programme ? Pour d), que dit la fiche d'un export sans `dynamic_shapes` ? Pour e), relis l'encadré 🕰️ : `dynamo=True` par défaut depuis quelle version ? Pour f), la condition dépend-elle de la forme de `x`, ou de ses valeurs ?

</details>

### B7.Q11 — Écart entraînement-production et dérive des données : vrai ou faux

<details><summary>Indice 1</summary>

Relis le paragraphe sur le training-serving skew (§207.7), puis 207.10 jusqu'au test de Kolmogorov-Smirnov.

</details>
<details><summary>Indice 2</summary>

La **dérive** vient du monde qui change ; le **training-serving skew** vient de deux chaînes de préparation qui diffèrent. Une p-valeur mesure la certitude d'un écart, pas sa taille. Les sorties du modèle se surveillent sans labels. Les bornes du PSI se fixent une fois, sur la référence.

</details>
<details><summary>Indice 3</summary>

a) **Vrai** : le modèle ne change pas, mais les images qu'il reçoit, si. Pour b), le skew existe-t-il dès le premier jour du déploiement ? Pour c), que devient une p-valeur quand les échantillons deviennent très grands ? Pour d), faut-il connaître la vraie classe pour compter les classes prédites ? Pour e), qu'est-ce qui a changé : la façon de préparer les images, les images reçues, ou le lien entre une image et sa classe ? Pour f), deux PSI calculés sur des intervalles différents mesurent-ils la même chose ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### B7.R1 — Ch. 24 : ce qu'une ligne de journal d'expériences doit contenir

<details><summary>Indice 1</summary>

Relis, au ch. 24, l'encadré 💼 sur le journal d'expériences, puis 207.5.

</details>
<details><summary>Indice 2</summary>

Une ligne par run, écrite automatiquement : ce qui **définit** le run (code, données, réglages complets, graine, versions), ce qu'il **mesure**, ce qu'il **produit**. Un JSONL s'écrit en ajoutant une ligne à la fin, jamais en réécrivant. Un run raté reste, avec son statut. L'identifiant de la fiche est fait de la date et de quelques caractères aléatoires.

</details>
<details><summary>Indice 3</summary>

a) Le commit (A) est indispensable : sans lui, on ne sait pas quel code a tourné ; pour B, C et D, demande-toi si la valeur change ce que fait le run. Pour b), que deviendrait une moyenne calculée sans les runs ratés ? Pour c), une ligne JSON ancienne a-t-elle besoin du nouveau champ pour se lire ? Pour d), deux runs relancés ajoutent-ils ou remplacent-ils des lignes ? Pour e), relis le format de `new_run_id` : `%Y%m%d-%H%M%S`.

</details>

### B7.R2 — Ch. 23 : checkpoint complet, state_dict et weights_only

<details><summary>Indice 1</summary>

Relis, au ch. 23, §23.9.1 (« Tout dans un fichier ») et §23.9.2 (« Les poids seuls »), avec l'encadré 🕰️ sur les formats.

</details>
<details><summary>Indice 2</summary>

Reprendre un entraînement demande tout ce qui détermine la suite : modèle, optimiseur, epoch, générateurs aléatoires, en dictionnaires de tenseurs. Utiliser un modèle demande les poids et le code qui reconstruit l'architecture. `weights_only=True` ne lit que des tenseurs et des types simples.

</details>
<details><summary>Indice 3</summary>

a) Le `state_dict` du modèle (A) en fait partie ; pour B, C et D, demande-toi si la suite de l'entraînement en dépend, et si l'objet se sauvegarde comme un dictionnaire de tenseurs. Pour b), de quoi la prédiction a-t-elle besoin ? Pour c), une instance de `Trainer` est-elle un tenseur ou un type simple ? Pour d), que faut-il construire **avant** d'appeler `load_state_dict` ? Pour e), où Adam garde-t-il ses moyennes de gradients ?

</details>

### B7.R3 — Ch. 15 : sauvegarder un Pipeline et figer les versions

<details><summary>Indice 1</summary>

Relis, au ch. 15, « Au-delà du livre : sauvegarder un modèle et ses versions » et ses trois règles.

</details>
<details><summary>Indice 2</summary>

On sauvegarde le pipeline entier ; un `joblib` ne se relit de façon fiable qu'avec les mêmes versions, et scikit-learn prévient depuis la 1.3 ; charger un pickle peut exécuter du code ; les métadonnées notent les versions, le score et les données, jamais un secret ; ONNX sert des prédictions sans Python.

</details>
<details><summary>Indice 3</summary>

a) **A** : le prétraitement fait partie du modèle, il voyage avec lui. Pour b), relis la deuxième règle : que garantit la documentation d'une version à l'autre ? Pour c), quel avertissement porte le nom des versions ? Pour d), relis la troisième règle. Pour e), demande-toi pour chaque lettre si elle aide à relire ou à juger le modèle, et si elle peut être lue par n'importe qui. Pour f), lequel des trois formats ne demande pas Python ?

</details>

<a id="papier"></a>

## ✏️ 🧮 Papier-crayon

### Ex B7.1 — Taille d'un artefact : float32, bfloat16, int8, et la mémoire au chargement ✏️

<details><summary>Indice 1</summary>

Relis le rappel maths « précision réduite et taille d'un modèle » et « La quantification dynamique » (§207.8).

</details>
<details><summary>Indice 2</summary>

Taille $= N \times$ octets par paramètre : 4 en float32, 2 en bfloat16, 1 en int8. Mo : divise par $10^6$ ; Mio : par $2^{20} = 1\,048\,576$. La quantification dynamique ne touche que les matrices des couches listées. Au chargement naïf, deux copies des poids coexistent. Un `state_dict` contient les paramètres **et** les buffers, chacun avec la taille de son type.

</details>
<details><summary>Indice 3</summary>

a) $4 \times 28\,589\,128 = 114\,356\,512$ octets, soit **114,4** Mo. Pour b), divise les mêmes octets par $1\,048\,576$. Pour c) et d), refais a) avec 2 puis 1 octet. Pour e), calcule $26\,646\,528 \times 1 + (28\,589\,128 - 26\,646\,528) \times 4$. Pour f), divise a) par e). Pour g), deux copies de b). Pour h), compte $4 \times (72\,954 + 2 \times (16 + 16 + 32 + 32 + 64 + 64))$ puis ajoute $6 \times 8$. Pour i), retire h) de la taille du fichier. Pour j), calcule $100 \times 650 / 72\,954$.

</details>

### Ex B7.2 — Cinq graines, un intervalle : la dispersion d'une accuracy rapportée ✏️

<details><summary>Indice 1</summary>

Relis « Rapporter un score : plusieurs graines et un intervalle » et le rappel maths sur l'intervalle de Student (§207.4).

</details>
<details><summary>Indice 2</summary>

Moyenne, écarts à la moyenne, somme de leurs carrés, divisée par $n - 1 = 4$, racine : $s$. Erreur type $s/\sqrt 5$. Demi-largeur $2{,}776 \times$ l'erreur type. Pour comparer deux modèles sur les mêmes graines, fais le même calcul sur les cinq différences graine par graine.

</details>
<details><summary>Indice 3</summary>

a) $(64{,}2 + 65{,}0 + 63{,}6 + 64{,}8 + 64{,}4)/5 = 322{,}0/5 =$ **64,4**. Pour b), les écarts sont $-0{,}2$ ; $0{,}6$ ; $-0{,}8$ ; $0{,}4$ ; $0$ : calcule $\sqrt{\sum (\text{écart})^2 / 4}$. Pour c), divise b) par $\sqrt 5$. Pour d), multiplie c) par 2,776. Pour e), retire d) de a). Pour f), calcule l'intervalle de B de la même façon et compare sa borne basse à la borne haute de A. Pour g), écris les cinq différences B − A. Pour h), refais b) à d) sur ces différences. Pour i), calcule g) ± h).

</details>

### Ex B7.3 — Latence et débit : p50 et p95 sur vingt mesures, batch 1 contre batch 32 ✏️

<details><summary>Indice 1</summary>

Relis « Latence, percentiles et débit » (§207.7) et la convention des percentiles en tête de `02_exercices.md`.

</details>
<details><summary>Indice 2</summary>

Trie les vingt mesures, numérotées de 0 à 19. Le percentile $q$ est en position $h = \frac{q}{100} \times 19$ : prends la mesure de rang $\lfloor h \rfloor$ et ajoute la fraction $h - \lfloor h \rfloor$ de l'écart avec la suivante. Débit $= \frac{\text{taille du batch}}{\text{latence du batch en secondes}}$. Pour j), passe par l'événement contraire : aucune des huit ne dépasse.

</details>
<details><summary>Indice 3</summary>

a) La somme des vingt mesures vaut 455,8 ms, d'où une moyenne de $455{,}8/20 =$ **22,79** ms. Pour b), $h = 0{,}5 \times 19 = 9{,}5$ : la moitié du chemin entre les mesures triées de rangs 9 et 10. Pour c), $h = 0{,}95 \times 19 = 18{,}05$ : rangs 18 et 19. Pour d), compte les mesures au-dessus de a). Pour e), $1\,000 / \text{p50}$. Pour f), $32 \times 1\,000 / 118$. Pour g), f)/e). Pour h), $118/32$. Pour i), compare le p95 du batch de 32 à 100 ms. Pour j), $1 - 0{,}95^8$.

</details>

### Ex B7.4 — Indice de stabilité de population (PSI) à la main sur quatre intervalles ✏️

<details><summary>Indice 1</summary>

Relis « L'indice de stabilité de population (PSI) », son exemple et « Les seuils » (§207.10).

</details>
<details><summary>Indice 2</summary>

Parts de la production : chaque compte divisé par 200. Contribution de $I_k$ : $(p_k - 0{,}25)\ln\frac{p_k}{0{,}25}$, positive ou nulle. Le PSI est leur somme ; il vaut aussi $D_{\text{KL}}(p\,\|\,q) + D_{\text{KL}}(q\,\|\,p)$. Pour la semaine 3, ajoute $\varepsilon$ à toutes les parts, $p_k$ comme $q_k$, avant de calculer.

</details>
<details><summary>Indice 3</summary>

a) $p_1 = 38/200 = 0{,}19$ : $(0{,}19 - 0{,}25)\ln\frac{0{,}19}{0{,}25} = (-0{,}06) \times (-0{,}2744) =$ **0,0165**. Pour b), ajoute les contributions de $p = 0{,}26$ ; $0{,}27$ ; $0{,}28$. Pour c), refais avec $p = 0{,}48$ ; $0{,}31$ ; $0{,}15$ ; $0{,}06$. Pour d), compare les quatre termes de c). Pour e), calcule $\sum 0{,}25 \ln\frac{0{,}25}{p_k}$. Pour f), place b) et c) par rapport à 0,1 et 0,25. Pour g), que vaut $\ln(0)$, et le produit $(0 - 0{,}25) \times \ln 0$ ? Pour h), le terme de $I_4$ devient $(10^{-4} - 0{,}2501)\ln\frac{10^{-4}}{0{,}2501}$ ; ajoute les trois autres, avec $\varepsilon$ aussi. Pour i), refais h) avec $10^{-3}$.

</details>

### Ex B7.5 — Fermi : servir 100 requêtes par seconde — cœurs CPU, regroupement en batches, coût mensuel 🧮

<details><summary>Indice 1</summary>

Relis « En ligne ou par batches, et une API HTTP » et la loi de Little (§207.7).

</details>
<details><summary>Indice 2</summary>

Cœurs occupés $=$ requêtes par seconde $\times$ temps de CPU par requête (en secondes) ; avec la règle des 60 %, divise par 0,6 et arrondis au-dessus. Requêtes en cours : $L = \lambda W$. En 10 ms, il arrive $\lambda \times 0{,}01$ requêtes. Avec un batch de $b$, le temps de CPU par requête vaut $\frac{2{,}7 + 1{,}33\,b}{b} + 3$ ms. Coût $=$ machines $\times$ prix horaire $\times$ 730.

</details>
<details><summary>Indice 3</summary>

1. $4 + 3 = 7$ ms par requête ; à 100 requêtes par seconde, $100 \times 0{,}007 = 0{,}7$ cœur occupé ; divise par 0,6 et arrondis ; refais au pic. Pour 2, $100 \times 0{,}030$ et $300 \times 0{,}060$. Pour 3, $100 \times 0{,}01$ et $300 \times 0{,}01$ requêtes, puis le temps de CPU par requête avec ce $b$. Pour 4, divise les cœurs nécessaires par 4, ajoute une machine, multiplie par $0{,}15$ à $0{,}25 \times 730$. Pour 5, refais 1 et 3 avec $40 + 3$ ms et $\frac{27 + 13{,}3\,b}{b} + 3$. Pour 6, compare la part fixe de 3 ms à la part du modèle.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B7.6 — « Ça marche dans mon notebook » : convaincre un manager d'industrialiser, en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis « Ce que cache un notebook » (§207.1), l'introduction de 207.6 (« un bug de ML ne plante pas ») et « Agir : alerter, réentraîner, revenir en arrière » (§207.10).

</details>
<details><summary>Indice 2</summary>

Une image parlante : un notebook est une recette notée au fil de la cuisine, que personne d'autre ne peut refaire. Ce que tu fais : ranger le code, le faire vérifier automatiquement, livrer le modèle avec son mode d'emploi, prévoir de revenir à la version précédente. Ce qu'elle gagne : des résultats qu'on retrouve, des changements sans peur, un incident réglé en minutes. Un fait : l'étude de Pimentel et coll. (207.1).

</details>
<details><summary>Indice 3</summary>

Commence par : « Le modèle marche, mais aujourd'hui il ne tourne que sur mon ordinateur, dans l'ordre où j'ai lancé mes essais. » Puis : le fait (une minorité de notebooks publics se rejouent jusqu'au bout) ; les trois jours en mots simples ; ce qu'elle y gagne ; la démo de vendredi, la version fiable trois jours après.

</details>

### Ex B7.7 — Journaliser les prédictions en production : minimisation, durée de conservation, droits des personnes ⚖️

<details><summary>Indice 1</summary>

Relis « Journaliser les prédictions, et le RGPD » et « Pourquoi un modèle se dégrade sans changer, et que surveiller » (§207.10), puis la réponse 422 de 207.7.

</details>
<details><summary>Indice 2</summary>

Une donnée est personnelle dès qu'elle se rattache à une personne, directement ou non (un identifiant, une adresse IP, une photo de chez soi) ; un pseudonyme reste une donnée personnelle tant que quelqu'un peut faire le lien. Pour surveiller, il faut des statistiques d'entrée (luminosité, contraste, éventuellement un embedding), les sorties et la décision finale du vendeur, pas la photo. Une finalité nouvelle se justifie à part. Une durée se fixe par type de donnée et se fait respecter par une suppression automatique.

</details>
<details><summary>Indice 3</summary>

Commence par la liste : identifiant, adresse IP et photo sont des données personnelles ; la classe et la probabilité le deviennent en étant rattachées au compte. Puis : un journal minimal (heure arrondie, pseudonyme, statistiques de l'image, classe, probabilité, catégorie choisie, latence, version) ; le réentraînement comme finalité distincte (compatibilité, information, éventuellement consentement pour un échantillon de photos) ; des durées par type, une purge planifiée ; une table de correspondance pour retrouver les lignes d'un vendeur ; une réponse 422 sans l'entrée et des journaux techniques sans corps de requête.

</details>

### Ex B7.8 — Sculley et coll. (2015) : « Hidden Technical Debt in Machine Learning Systems » 📄

<details><summary>Indice 1</summary>

Lis d'abord le résumé et la figure 1 (p. 4), puis les titres en gras de chaque section : chacun nomme un risque.

</details>
<details><summary>Indice 2</summary>

Pour chaque risque, cherche trois choses dans le texte : la définition, l'exemple, la parade (« mitigation strategy »). Les chiffres à relever : l'année de la métaphore (section 1), la part du code de colle (section 5), Knight Capital (section 5). Les six principes de configuration et les cinq questions finales sont des listes à puces (sections 6 et 9).

</details>
<details><summary>Indice 3</summary>

1. La métaphore vient de Ward Cunningham, en 1992 ; la dette cachée « compounds silently ». Pour 2, la figure a dix boîtes, dont « ML Code » au milieu. Pour 3, CACE est défini dans le paragraphe « Entanglement ». Pour 4, cherche « versioned copy » et les quatre puces de « Underutilized Data Dependencies ». Pour 5, l'exemple des deux modèles boursiers. Pour 6, la phrase « at most … machine learning code ». Pour 7, compare chaque principe à `load_config`. Pour 8, les trois puces de « Monitoring and Testing ». Pour 9, les titres en gras de la section 8. Pour 10, pense au suivi d'expériences, à la validation des données et au ML Test Score.

</details>

<a id="entretien"></a>

## 💼 Entretien

### B7.E1 — Comment passes-tu d'un notebook à du code prêt pour la production ?

<details><summary>Indice 1</summary>

Relis 207.1 à 207.3, puis l'arborescence du dépôt.

</details>
<details><summary>Indice 2</summary>

Quatre étapes : extraire le code en fonctions dans un paquet installable ; une ligne de commande avec logging et codes de sortie ; une configuration typée et validée ; des tests et une intégration continue. Le notebook reste, mais il appelle le paquet.

</details>
<details><summary>Indice 3</summary>

Commence par : « D'abord, je fais tourner le notebook de bout en bout dans un noyau neuf, pour savoir ce qu'il fait vraiment. » Puis : les fonctions dans `src/` (données, modèle, entraînement), `pyproject.toml` et installation éditable, une ligne de commande, une configuration YAML validée, un journal des runs, les tests et la CI, l'artefact.

</details>

### B7.E2 — Comment garantis-tu la reproductibilité d'un entraînement ?

<details><summary>Indice 1</summary>

Relis 207.4 et 207.5.

</details>
<details><summary>Indice 2</summary>

Deux sens : refaire le même run (tout figer : code, données, environnement, graines) et retrouver la même conclusion (plusieurs graines, un intervalle). Pour chacun, dis ce que tu figes et ce que tu notes.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je distingue deux choses : refaire exactement le même run, et retrouver le même résultat avec d'autres graines. » Puis : `seed_everything`, `generator` du `DataLoader`, algorithmes déterministes ; un verrou des versions ; le commit ; l'empreinte des données ; le journal ; cinq graines, moyenne et intervalle ; comparer graine par graine.

</details>

### B7.E3 — Que testes-tu dans un projet de machine learning ?

<details><summary>Indice 1</summary>

Relis 207.6.

</details>
<details><summary>Indice 2</summary>

Les quatre familles de la fiche, plus le service (l'API qui refuse une entrée invalide, un test de non-régression qui compare l'API à l'évaluation hors ligne) ; et quand chaque test tourne (à chaque commit, chaque nuit, avant une mise en production).

</details>
<details><summary>Indice 3</summary>

Commence par : « Un bug de ML ne fait pas planter le programme, il rend le modèle un peu moins bon : je teste donc des propriétés. » Puis : tests unitaires, tests de données, overfitting d'un mini-batch, entraînement fumée, test de l'API, test de non-régression ; la CI ; la grille du ML Test Score.

</details>

### B7.E4 — Comment surveilles-tu un modèle en production, et que fais-tu en cas de dérive ?

<details><summary>Indice 1</summary>

Relis 207.10, de « que surveiller » à « Agir ».

</details>
<details><summary>Indice 2</summary>

Quatre choses à surveiller (qualité quand les labels arrivent, entrées, sorties, service), deux outils (PSI, test KS) sur des fenêtres fixes, des seuils écrits à l'avance, trois décisions (surveiller de plus près, réentraîner, revenir en arrière), et le RGPD pour le journal.

</details>
<details><summary>Indice 3</summary>

Commence par : « Le modèle ne prévient pas : il faut construire la surveillance avant le premier incident. » Puis : référence et fenêtres, PSI et KS par variable, distribution des classes prédites, un échantillon labellisé, latence et erreurs ; une alerte, une décision écrite ; un retour arrière en une minute ; un déploiement progressif du nouveau modèle.

</details>

### B7.E5 — Tu dois exposer ton modèle par une API : quels choix fais-tu (batches, latence, versions) ?

<details><summary>Indice 1</summary>

Relis 207.7 et 207.8.

</details>
<details><summary>Indice 2</summary>

Partir du besoin (en ligne ou par batches, objectif de latence en p95, débit attendu) ; l'artefact chargé une fois ; la validation des entrées ; la mesure ; la version dans chaque réponse ; le regroupement dynamique si le trafic le permet ; l'export et la précision réduite seulement après mesure.

</details>
<details><summary>Indice 3</summary>

Commence par : « Je pars de l'usage : quelqu'un attend-il la réponse, et combien de requêtes par seconde ? » Puis : FastAPI, `/health` et `/predict`, pydantic, le prétraitement partagé, p50 et p95 mesurés côté client, le regroupement, la loi de Little pour le nombre de processus, la version du modèle dans la réponse, un retour arrière possible.

</details>

<a id="notebook"></a>

## Exercices du notebook

Les indices des exercices du notebook (B7.9 à B7.25) seront ajoutés à la prochaine session de génération, avec le notebook.
