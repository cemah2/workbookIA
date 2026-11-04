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

a) **B** : écrire l'artefact à la fin d'un run réussi est une étape normale. Pour b), le programme continue, mais dans des conditions inattendues : quel niveau ? Pour c), compte les messages de niveau `WARNING` ou plus grave. Pour d), où se configurent les handlers, et qu'arriverait-il si chaque module les configurait ? Pour e), lequel des trois messages contient un fait que l'on peut vérifier ? Pour f), le `except` empêche-t-il le script d'aller au bout ?

</details>

### B7.Q4 — Ce qui va dans la configuration, et ce qui ne doit jamais y aller

<details><summary>Indice 1</summary>

Relis le début de 207.3, le rappel outil sur PyYAML, puis « Variables d'environnement et secrets » (§207.9).

</details>
<details><summary>Indice 2</summary>

La configuration se versionne et se recopie dans les journaux : elle contient les réglages d'un run, jamais un secret. PyYAML suit la version 1.1 du format : une notation scientifique sans point n'est pas un nombre, des guillemets font une chaîne. Une image Docker garde chacune de ses couches ; git garde chaque commit. Une dataclass ne vérifie pas les types : c'est le rôle de `__post_init__`.

</details>
<details><summary>Indice 3</summary>

a) Le learning rate (A) est un réglage : il y va ; pour B, C et D, demande-toi si la valeur peut être lue par toute personne qui clone le dépôt. Pour b), applique à chacune des trois lignes les deux règles de PyYAML de l'indice 2. Pour c), relis le `except (OSError, ValueError)` de `main`. Pour d), que montre `docker history` ? Pour e), un nouveau commit efface-t-il les anciens ? Pour f), que dit le commentaire de `__post_init__` dans `TrainConfig` ?

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

a) **B** : sans `zero_grad`, les gradients s'accumulent d'un pas à l'autre, et le réseau a du mal à apprendre par cœur 16 images ; les deux autres tests ne regardent même pas la boucle. Pour b), quel test regarde le contenu des images de deux jeux ? Pour c), relis la puce « bout en bout » : que juge l'entraînement fumée, et que ne juge-t-il pas ? Pour d), une loss qui ne baisse pas empêche-t-elle un script de rendre 0 ? Pour e), quelle somme est la plus prudente des quatre ?

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

a) **B** : TorchScript est déprécié dans la 2.11, et un pickle de l'objet demande le code de sa classe au chargement. Pour b), quel format se lit sans PyTorch, depuis d'autres langages ? Pour c), que produit `torch.compile` : un fichier, ou un modèle plus rapide dans le même programme ? Pour d), que dit la fiche d'un export sans `dynamic_shapes` ? Pour e), relis l'encadré 🕰️ : sur quoi `torch.onnx.export` s'appuie-t-il par défaut, et depuis quelle version ? Pour f), la condition dépend-elle de la forme de `x`, ou de ses valeurs ?

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

a) Le commit (A) est indispensable : sans lui, on ne sait pas quel code a tourné ; pour B, C et D, demande-toi si la valeur change ce que fait le run. Pour b), que deviendrait une moyenne calculée sans les runs ratés ? Pour c), une ligne JSON ancienne a-t-elle besoin du nouveau champ pour se lire, et réécrit-on un journal JSONL ? Pour d), deux runs relancés ajoutent-ils ou remplacent-ils des lignes ? Pour e), relis le format de `new_run_id` : `%Y%m%d-%H%M%S`.

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

a) **A** : le prétraitement fait partie du modèle, il voyage avec lui. Pour b), relis la deuxième règle : que garantit la documentation d'une version à l'autre ? Pour c), quel avertissement porte le nom des versions ? Pour d), relis la troisième règle. Pour e), demande-toi pour chaque lettre si elle aide à relire ou à juger le modèle, et si elle a sa place dans un petit fichier qui voyage avec lui. Pour f), lequel des trois formats ne demande pas Python ?

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

1. $4 + 3 = 7$ ms par requête ; à 100 requêtes par seconde, $100 \times 0{,}007 = 0{,}7$ cœur occupé ; divise par 0,6 et arrondis ; refais au pic. Pour 2, $100 \times 0{,}030$ et $300 \times 0{,}060$. Pour 3, $100 \times 0{,}01$ et $300 \times 0{,}01$ requêtes, puis le temps de CPU par requête avec ce $b$. Pour 4, divise les cœurs nécessaires par 4, ajoute une machine, multiplie par $0{,}15$ à $0{,}25 \times 730$. Pour 5, refais 1 et 3 avec $40 + 3$ ms et $\frac{27 + 13{,}3\,b}{b} + 3$, ajoute une machine de redondance à chaque flotte, puis compare $730 \times$ le prix horaire des deux solutions, et le nombre d'images qu'arrive à réunir un GPU en 10 ms. Pour 6, compare la part fixe de 3 ms à la part du modèle.

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

1. La métaphore vient de Ward Cunningham, en 1992 ; la dette cachée « compounds silently ». Pour 2, la figure a dix boîtes, dont « ML Code » au milieu. Pour 3, CACE est défini dans le paragraphe « Entanglement ». Pour 4, cherche « versioned copy » et les quatre puces de « Underutilized Data Dependencies ». Pour 5, cherche « randomization » pour la boucle directe, et l'exemple des deux modèles boursiers pour la boucle cachée. Pour 6, la phrase « at most … machine learning code ». Pour 7, compare chaque principe à `load_config`. Pour 8, les trois puces de « Monitoring and Testing ». Pour 9, les titres en gras de la section 8. Pour 10, pense au suivi d'expériences, à la validation des données et au ML Test Score.

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

### Ex B7.9 — Du notebook au paquet : data.py, model.py, train.py, pyproject.toml et installation éditable 📦

<details><summary>Indice 1</summary>

Pars de ton `cifarkit` du MP5 et de la fiche : 207.1 donne un `pyproject.toml` presque complet, 207.4 `seed_everything` et l'empreinte, 207.6 le CNN. Un paquet n'est que du code importable : aucune variable globale du notebook, aucun chemin en dur, aucun `print`. Chaque fonction reçoit ce dont elle a besoin et rend un résultat ; c'est ce qui la rend testable.

</details>
<details><summary>Indice 2</summary>

- `pyproject.toml` : la fiche, plus `safetensors` dans `dependencies`, l'extra `serve` avec `pydantic`, et la table `[tool.pytest.ini_options]` (`testpaths`, `markers = ["slow: ..."]`). Avec le layout `src/`, setuptools trouve le paquet tout seul.
- `load_npz` : `np.load(path)` dans un `with`, puis les tableaux `x_<part>` et `y_<part>` ; vérifie `X.dtype == np.uint8`, `X.ndim == 4`, `X.shape[1:] == (32, 32, 3)`, `y.shape == (len(X),)`, `np.issubdtype(y.dtype, np.integer)` et `0 <= y <= 9` avant de rendre `y.astype(np.int64)`.
- `channel_stats` : `x = images / 255.0` en float64, puis `x.mean(axis=(0, 1, 2))` et `x.std(axis=(0, 1, 2))`, convertis en floats Python (`float(v)`).
- `preprocess` : `torch.from_numpy(images).permute(0, 3, 1, 2)` (NHWC vers NCHW), `.to(torch.float32) / 255`, puis la moyenne et l'écart-type mis en forme `(1, 3, 1, 1)` pour diffuser sur chaque canal.
- `make_loader` : un `TensorDataset(preprocess(X, mean, std), torch.as_tensor(y))` et `DataLoader(..., shuffle=shuffle, generator=torch.Generator().manual_seed(seed) if shuffle else None)`.
- `build_model` : une liste de couches construite par une boucle sur `widths`, puis `nn.Sequential(*layers)` ; la taille de la couche dense du `"cnn"` vaut `widths[-1] * (32 // 2**len(widths))**2`.
- `train` : `seed_everything`, `channel_stats(X)`, `build_model`, les deux loaders, `torch.optim.SGD(..., momentum=0.9)` ; dans chaque epoch, la boucle de 207.6 (`zero_grad`, la loss, `backward`, `step`), puis `evaluate` sur la validation et une ligne `log.info`.

</details>
<details><summary>Indice 3</summary>

```python
def preprocess(images, mean, std):
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise ValueError(...)
    if images.ndim == 3:
        images = images[None]                       # one image -> a batch of one
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    ...                                             # (x - m) / s with m, s of shape (1, 3, 1, 1)


def train(X, y, X_val, y_val, *, arch="cnn", widths=(16, 32), epochs=1, lr=0.05, batch_size=128, seed=0,
          deterministic=True, num_workers=0):
    start = time.perf_counter()
    seed_everything(seed, deterministic)            # before build_model: the initialization depends on it
    mean, std = channel_stats(X)                    # the training images only
    ...                                             # model, loaders, SGD; for each epoch: the loop, evaluate, log.info
    model.eval()
    return {"model": model, "mean": mean, "std": std, "history": history, "seconds": time.perf_counter() - start}
```

</details>

### Ex B7.10 — Une interface en ligne de commande : argparse, logging et codes de sortie 📦

<details><summary>Indice 1</summary>

Relis 207.2 : le code de `cli.py` y est presque entier. Ce qui reste : l'option `--out` et `--version`, la lecture des données, l'appel de `train`, l'écriture des poids et le résumé sur stdout. Range chaque erreur dans sa case : une configuration refusée (code 2), des données illisibles ou un entraînement qui plante (code 1).

</details>
<details><summary>Indice 2</summary>

- `--version` : `parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")` ; la sous-commande obligatoire : `add_subparsers(dest="command", required=True)`.
- Le logging : un seul `logging.basicConfig(level=..., format="%(asctime)s %(levelname)s %(name)s: %(message)s")`, dans `main`, après `parse_args` ; il écrit sur stderr par défaut.
- Trois `try` : autour de `load_config` (`except (OSError, ValueError)` → `log.error`, `return 2`), autour de `load_npz` et du contrôle de `n_train` (→ `return 1`), autour du reste (`except Exception` → `log.exception`, `return 1`).
- Les poids : `from safetensors.torch import save_file`, `out.mkdir(parents=True, exist_ok=True)`, `save_file(result["model"].state_dict(), str(out / "model.safetensors"))`.
- Le résumé : `print(json.dumps({...}))`, le seul `print` du paquet, avec `str(weights.resolve())` pour le chemin absolu.
- `__main__.py` : deux lignes, `from .cli import main` puis `raise SystemExit(main())`.

</details>
<details><summary>Indice 3</summary>

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config, args.set)
    except (OSError, ValueError) as err:
        log.error("invalid configuration: %s", err)
        return 2
    ...   # the data (return 1 on OSError/ValueError), train, save_file, log.info, print(json.dumps(...)), return 0
```

</details>

### Ex B7.11 — Configuration : dataclass + YAML, surcharges en ligne de commande, validation 📦

<details><summary>Indice 1</summary>

207.3 donne `TrainConfig` et `load_config` presque entiers. Ce qui manque : les bornes de tous les réglages, le contrôle de `arch` et de `widths`, la conversion d'une liste en tuple, un fichier qui n'est pas un dictionnaire, et `to_dict`. Le principe : tout refuser au chargement, avec un message qui nomme le réglage, plutôt qu'au milieu d'un entraînement.

</details>
<details><summary>Indice 2</summary>

- Les types : `type(f.default)` donne le type attendu de chaque champ (`int`, `float`, `str`, `tuple`, `bool`) ; compare avec `type(value) is kind` (et non `isinstance`, pour qu'un booléen ne passe pas pour un entier) ; deux conversions par `object.__setattr__` (la dataclass est figée).
- Les bornes, après les types : un `if` par règle, ou un seul message qui donne les valeurs reçues ; `widths` : non vide, chaque élément un `int` (pas un booléen) ≥ 1.
- `to_dict` : `dataclasses.asdict(self)`, puis `widths` en liste.
- `load_config` : `yaml.safe_load(...) or {}` pour un fichier vide, `isinstance(loaded, dict)`, puis chaque surcharge par `item.partition("=")` et `yaml.safe_load(value)` ; les clés inconnues : `set(raw) - {f.name for f in fields(TrainConfig)}`.
- Les fichiers YAML : une clé par ligne (`widths: [16, 32]`), un nombre avec un point (`lr: 0.05`, jamais `5e-2` sans point, que PyYAML lit comme une chaîne).

</details>
<details><summary>Indice 3</summary>

```python
def __post_init__(self):
    for f in fields(self):
        value, kind = getattr(self, f.name), type(f.default)
        if kind is float and type(value) is int:
            object.__setattr__(self, f.name, float(value))
        elif kind is tuple and type(value) is list:
            object.__setattr__(self, f.name, tuple(value))
        elif type(value) is not kind:
            raise ValueError(f"{f.name}: expected {kind.__name__}, got {value!r}")
    ...   # widths, arch, then the bounds of seed, epochs, lr, batch_size, n_train, num_workers
```

</details>

### Ex B7.12 — Prédire : même graine, puis num_workers = 4, puis GPU — résultats identiques ? 🔮

<details><summary>Indice 1</summary>

Repère tous les tirages d'un run : l'initialisation des poids (après `torch.manual_seed`), l'ordre des mini-batches (le `generator` du `DataLoader`), et les retournements tirés dans `__getitem__`. Pour chacun, demande-toi dans quel processus il a lieu, et quel générateur il utilise.

</details>
<details><summary>Indice 2</summary>

Avec `num_workers = 0`, `__getitem__` tire dans le générateur global du notebook, juste après l'initialisation. Avec des workers, chaque processus fils a son propre générateur, semé à partir d'une graine de base que le `DataLoader` tire, à chaque epoch, dans son `generator` (207.4, et la page « Reproducibility » de PyTorch). Pour a) et b), la question est : un run rejoué refait-il exactement les mêmes tirages ? Pour c) et d), les deux runs tirent-ils dans les mêmes générateurs, et y a-t-il seulement des tirages à faire ?

</details>
<details><summary>Indice 3</summary>

Pour a) : deux runs à 0 worker partent de `torch.manual_seed(0)`, puis font les mêmes opérations dans le même ordre : compare les deux suites de tirages. Pour b), écris d'où vient la graine de chaque worker et si elle change d'un run à l'autre. Pour c), compare le générateur qui tire un retournement à 0 worker et celui qui le tire à 4 workers. Pour d), demande-toi ce qui reste aléatoire une fois les retournements retirés, et où il est tiré. Pour e), relis 207.4 et le ch. 23 sur le matériel.

</details>

### Ex B7.13 — Mesurer le non-déterminisme : graines fixées ou non, algorithmes déterministes, cinq exécutions 🔬

<details><summary>Indice 1</summary>

`train_run_13` n'entraîne rien elle-même : elle lance ta commande, comme le ferait un script ou une intégration continue, puis relit ce que la commande a écrit (la ligne JSON de stdout, le fichier de poids). Pour comparer deux runs, compare **tous** leurs tenseurs, pas une accuracy.

</details>
<details><summary>Indice 2</summary>

- `wbp.cli(PROJECT, "train", "--config", "configs/fast.yaml", "--set", f"seed={seed}", ...)` rend un `RunResult` : `.ok`, `.returncode`, `.tail(3)` (la fin de stderr), `.last_json()` (la dernière ligne de stdout, lue en JSON) ; un dossier neuf par run : `tempfile.mkdtemp()`.
- Les poids : `safetensors.torch.load_file(summary["weights"])` rend un dictionnaire de tenseurs.
- `same_weights_13` : les mêmes clés (`a.keys() == b.keys()`) et `torch.equal` pour chacune ; `max_gap_13` : `(a[k].double() - b[k].double()).abs().max()` pour chaque clé, le plus grand, en `float`.
- `deterministic` s'écrit `true` ou `false` dans la surcharge (YAML) : `str(deterministic).lower()`.

</details>
<details><summary>Indice 3</summary>

```python
def train_run_13(seed, deterministic=True):
    result = wbp.cli(PROJECT, "train", "--config", "configs/fast.yaml", "--set", f"seed={seed}",
                     "--set", f"deterministic={str(deterministic).lower()}", "--out", tempfile.mkdtemp())
    if not result.ok:
        raise RuntimeError(...)                     # the code and result.tail(3)
    summary = result.last_json()
    ...                                             # {"weights": load_file(...), "val_acc": ..., "seconds": ...}
```

</details>

### Ex B7.14 — Suivi d'expériences maison : identifiant de run, configuration, métriques et artefacts en JSONL 📦

<details><summary>Indice 1</summary>

Relis 207.5 : `new_run_id` et `log_run` y sont entiers. Le travail est dans `cli.py` : décider **quand** une ligne s'écrit (toujours, même si le run échoue, d'où un `finally`) et **quoi** y mettre (ce qui définit le run, ce qu'il mesure, ce qu'il produit). Prépare le dictionnaire au début du run, complète-le en chemin, écris-le à la fin.

</details>
<details><summary>Indice 2</summary>

- `file_sha256` : `hashlib.sha256()` nourri par blocs (`iter(lambda: f.read(1 << 20), b"")`), puis `.hexdigest()` ; `artifact` : `Path(path).stat().st_size`.
- `environment` : `subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)`, dans un `try` (pas de git : `None`) ; `dirty` : `bool(sortie de git status --porcelain)`.
- `log_run` : `path.parent.mkdir(parents=True, exist_ok=True)`, puis `open(path, "a")` (le mode `"a"` ajoute à la fin ; `"w"` effacerait le journal) et `json.dumps(record, sort_keys=True) + "\n"`.
- `cli.py` : `run_id = new_run_id()` après la configuration ; `record = {"run_id": ..., "started": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(), "environment": environment(), "status": "failed"}` ; dans le chemin réussi, `record.update(status="ok", ...)` ; dans le `finally`, `record["seconds"] = ...` puis `log_run(args.runs, record)`.
- Tout ce qui va dans le journal doit passer par `json.dumps` : des floats Python, des listes, pas de tableaux NumPy ni de tenseurs.

</details>
<details><summary>Indice 3</summary>

```python
    record = {"run_id": run_id, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "config": config.to_dict(), "environment": environment(), "status": "failed"}
    start = time.perf_counter()
    try:
        ...                                         # data (record["data"]), train, save the weights
        record.update(status="ok", metrics=..., normalization=..., artifacts=[artifact(weights)])
        return 0
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"
        ...
    finally:
        record["seconds"] = round(time.perf_counter() - start, 3)
        log_run(args.runs, record)
```

</details>

### Ex B7.15 — Tests pytest pour le ML : formes, overfitting d'un mini-batch, prétraitement, schéma des données 📦

<details><summary>Indice 1</summary>

Un bon test de ML est un cas où **tu sais la réponse sans le code** : une image toute blanche prétraitée vaut $(1 - \text{mean}_c)/\text{std}_c$ dans chaque canal ; le fichier de données a 2 000 images `uint8` ; aucune image de validation n'a l'empreinte d'une image d'entraînement ; `epochs=0` doit être refusé. Pour chaque famille de 207.6, écris d'abord la phrase « si le code est juste, alors… », puis l'`assert`.

</details>
<details><summary>Indice 2</summary>

- Prétraitement : `@pytest.mark.parametrize("value", [0, 255])`, une image `np.full((32, 32, 3), value, dtype=np.uint8)`, une moyenne et un écart-type simples (`(0.5, 0.4, 0.3)`, `(0.25, 0.2, 0.1)`), la valeur attendue calculée dans le test, `torch.allclose`. L'ordre des canaux : une image rouge (seul le canal 0 à 255) doit donner 1 dans le canal 0, 0 ailleurs, avec une moyenne nulle et un écart-type de 1.
- Schéma : la fixture `data_file`, `load_npz(data_file, part)` pour `"train"` et `"val"` ; un fichier mal formé écrit dans `tmp_path` par `np.savez`, puis `with pytest.raises(ValueError):`.
- Recouvrement : un ensemble (`set`) des `fingerprint(image)` de l'entraînement, puis compter les images de validation qui y sont.
- Configuration : `@pytest.mark.parametrize("override", [...])` et `pytest.raises(ValueError)` autour de `load_config(None, [override])`.
- Codes de sortie : `main([...])` rend un entier ; passe `--runs str(tmp_path / "runs.jsonl")`.
- Bout en bout : `@pytest.mark.slow`, la fixture `synthetic_file`, `--set data=...`, `--set n_train=96`, `--out` dans `tmp_path`, puis le code, le fichier de poids et le journal.

</details>
<details><summary>Indice 3</summary>

```python
@pytest.mark.parametrize("value", [0, 255])
def test_constant_image_preprocessed(value):
    x = preprocess(np.full((32, 32, 3), value, dtype=np.uint8), MEAN, STD)
    want = ...                                      # (value / 255 - m) / s for each channel, as a (1, 3, 32, 32) tensor
    assert torch.allclose(x, want, atol=1e-6), f"expected {...} per channel, got {...}"


def test_exit_code_of_an_invalid_config(tmp_path):
    code = main(["train", "--set", "epochs=0", "--runs", str(tmp_path / "runs.jsonl")])
    assert code == 2, f"expected the exit code 2 for epochs=0, got {code}"
```

</details>

### Ex B7.16 — Le modèle qui marche dans le notebook et pas en production : cinq écarts à retrouver 🐛

<details><summary>Indice 1</summary>

Suis une image du fichier PNG jusqu'au nom renvoyé, et compare à chaque étape avec ce que faisait le notebook du collègue (l'énoncé le décrit) : le format de l'image lue, son échelle, sa normalisation, son type, le mode du modèle, l'ordre des noms. Les diagnostics s'enchaînent dans cet ordre : corrige celui qui échoue, réexécute, passe au suivant.

</details>
<details><summary>Indice 2</summary>

- Point 1 : lis le type de chaque tableau entre la lecture et le modèle ; NumPy garde le type le plus précis d'une opération, et `torch.from_numpy` garde celui du tableau.
- Point 2 : `wbp.imread_bgr` lit comme OpenCV ; compare sa sortie pour `PATHS_16[0]` à `X_16[0]`, canal par canal, et leurs valeurs extrêmes.
- Point 3 : d'où viennent la moyenne et l'écart-type de `preprocess_16` ? Prétraite une image seule, puis dans un batch : que devrait-il se passer ?
- Point 4 : relis ce que fait un modèle à batchnorm en mode entraînement (ch. 20), et ce que rend `load_service_16`.
- Point 5 : la liste `names` de `predict_16` suit-elle l'ordre des sorties du modèle ? Qui, de l'entraînement ou du service, connaît cet ordre, et comment le second l'apprend-il ?

</details>
<details><summary>Indice 3</summary>

Pour le point 1 : `np.asarray(images, dtype=np.float64)` fait du prétraitement entier un calcul en float64, et `conv2d` refuse un tenseur float64 avec des poids float32 ; choisis le type de l'entraînement. Pour les points 2 à 5, pose le critère sans le corriger : l'image lue doit valoir `X_16[0] / 255` à $10^{-6}$ près ; une image doit avoir le même prétraitement seule et dans un batch ; la même image doit avoir les mêmes scores seule et dans un batch, et le modèle ne doit pas changer en prédisant ; le nom d'une sortie $k$ doit venir de l'artefact, quel que soit le modèle.

</details>

*Les indices des exercices B7.17 à B7.25 (la fin du service et l'exploitation) seront ajoutés à la prochaine session de génération, avec le notebook.*
