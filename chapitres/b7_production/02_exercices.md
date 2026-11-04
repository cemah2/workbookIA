# B7 · Du notebook à la production — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/b7_production/06_mes_reponses.md` (créée par `python tools/start_chapter.py B7`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ B7.1 à B7.4 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », l'estimation 🧮 B7.5, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), les lettres ensemble pour plusieurs choix, dans n'importe quel ordre (`"EF"`).
> Conventions du chapitre (fiche, encadré 📐) : $n$ compte des graines en 207.4 ; 1 Mo $= 10^6$ octets et 1 Mio $= 2^{20} = 1\,048\,576$ octets ; $\ln$ est le logarithme naturel ; un percentile se calcule avec la méthode par défaut de `np.percentile` (`method="linear"`) : on trie les $n$ mesures $x_{(0)} \le \dots \le x_{(n-1)}$, le percentile $q$ est à la position $h = \frac{q}{100}(n - 1)$, et l'on interpole entre ses deux voisines, $x_{(\lfloor h \rfloor)} + (h - \lfloor h \rfloor)\,\big(x_{(\lfloor h \rfloor + 1)} - x_{(\lfloor h \rfloor)}\big)$. Les références juridiques sont un repère de culture professionnelle, pas un conseil juridique.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### B7.Q1 — Les pièges du notebook : état caché et ordre d'exécution 🧠 ⏱️ 3 min
*Fiche §207.1 · parcours R*

Un notebook a trois cellules : (1) `x = 2` ; (2) `x = x * 10` ; (3) `print(x + 1)`.

a) Tu exécutes les cellules 1, 2 et 3, puis encore 2 et 3. Qu'affiche la dernière exécution de la cellule 3 ?
b) Dans ton vrai notebook, une cellule définissait `scale`, dont les cellules suivantes se servent ; tu l'as supprimée hier, et tout marche encore. Après « Redémarrer et tout exécuter » : (A) une `NameError` à la première cellule qui lit `scale` ; (B) les mêmes résultats, car le noyau garde `scale` d'une session à l'autre ; (C) Jupyter recrée la cellule supprimée à partir de son historique.
c) Vrai ou faux : le `git diff` d'un `.ipynb` dont tu as changé une ligne de code se relit aussi facilement que celui d'un `.py`.
d) Ton notebook a fait `import cifarprod.data` ; tu modifies ensuite `src/cifarprod/data.py`, installé en mode éditable. Pour que le notebook voie la modification : (A) rien à faire, l'installation éditable relit le fichier à chaque appel de fonction ; (B) réinstalle le paquet avec `pip install -e .` ; (C) redémarre le noyau, puis réimporte le module.
e) Vrai ou faux : une variable définie par une cellule que tu as ensuite supprimée existe encore dans le noyau, tant qu'il n'a pas redémarré.

### B7.Q2 — src/, tests/, configs/, scripts/ : où va quoi ? 🧠 ⏱️ 3 min
*Fiche §207.1 · parcours R*

Le dépôt de 207.1 a aussi un dossier `scripts/`, pour les programmes qu'on lance à la main. Pour a) à d), choisis la place de chaque fichier : (A) `src/cifarprod/` ; (B) `tests/` ; (C) `configs/` ; (D) `scripts/`.

a) `preprocess.py`, qui normalise une image, appelé par l'entraînement et par l'API.
b) `download_subset.py`, lancé une fois pour télécharger les 2 000 images d'entraînement et vérifier leur empreinte.
c) `smoke.yaml` : une epoch sur 200 images, pour l'intégration continue.
d) `test_preprocess.py`, qui vérifie la sortie de `preprocess` sur une image constante.
e) Vrai ou faux : les 2 000 images téléchargées et le dossier `runs/` des sorties de chaque run se versionnent avec le code, pour qu'un clone neuf ait tout.
f) Sur un clone neuf, sans `pip install -e .` ni réglage particulier de pytest, tu lances `pytest` depuis la racine : (A) les tests importent `src/cifarprod/` directement ; (B) `ModuleNotFoundError: No module named 'cifarprod'` ; (C) pytest installe le paquet avant de lancer les tests.
g) `[project.scripts]` contient `cifarprod = "cifarprod.cli:main"`. Vrai ou faux : après `pip install -e .`, la commande `cifarprod train --config configs/fast.yaml` appelle la fonction `main` du module `cifarprod.cli`.

### B7.Q3 — print ou logging : niveaux, destinations et messages utiles 🧠 ⏱️ 3 min
*Fiche §207.2 · parcours R*

a) « saved artifact to artifacts/1.0.0 (73744 bytes) », à la fin d'un run réussi, s'écrit au niveau : (A) `DEBUG` ; (B) `INFO` ; (C) `WARNING` ; (D) `ERROR`.
b) « no GPU found: training on CPU (about 10 times slower) » s'écrit au niveau : (A) `DEBUG` ; (B) `INFO` ; (C) `WARNING` ; (D) `ERROR`.
c) Le point d'entrée appelle `logging.basicConfig(level=logging.WARNING)`, puis le programme émet un message `log.debug`, trois `log.info`, un `log.warning` et un `log.error`. Combien de messages s'affichent ?
d) Vrai ou faux : `data.py`, un module de ton paquet, doit appeler `logging.basicConfig` pour que ses messages s'affichent.
e) Lequel de ces messages aide le plus quelqu'un qui cherche un bug six mois plus tard ? (A) « étape de chargement des données terminée avec succès, on passe à la suite » ; (B) « ATTENTION : données chargées, à vérifier si les résultats sont mauvais » ; (C) « loaded 2000 train images (sha256 3fa1c2…), labels 0..9 ».
f) Un script attrape toute erreur par `except Exception as err: print(err)`, puis va jusqu'à sa dernière ligne. L'entraînement plante à l'epoch 2. Quel code de sortie lit `subprocess.run(...).returncode` ?

### B7.Q4 — Ce qui va dans la configuration, et ce qui ne doit jamais y aller 🧠 ⏱️ 3 min
*Fiche §207.3, §207.9 · parcours R*

a) Lesquels vont dans `configs/fast.yaml`, versionné avec le code ? (A) le learning rate ; (B) la clé d'API de Hugging Face ; (C) la graine ; (D) le chemin du dossier des données, relatif au dépôt. (une ou plusieurs lettres)
b) Laquelle de ces lignes PyYAML 6.0 lit-il comme un nombre à virgule ? (A) `lr: 5.0e-4` ; (B) `lr: 5e-4` ; (C) `lr: "5.0e-4"`.
c) Avec `load_config` et `main` de la fiche, `cifarprod train --set epochs=0` rend quel code de sortie ?
d) Vrai ou faux : une clé posée par `ENV HF_TOKEN=...` dans un Dockerfile reste lisible par toute personne qui a l'image, même si une ligne plus loin la remplace par une chaîne vide.
e) Tu as commité une clé d'API par erreur, puis tu l'as retirée au commit suivant : (A) c'est réglé, le dernier commit n'a plus la clé ; (B) c'est réglé si le dépôt est privé ; (C) la clé reste dans l'historique : il faut la révoquer.
f) Vrai ou faux : `@dataclass(frozen=True)` refuse à lui seul la valeur `lr="1e-3"` pour un champ annoté `lr: float`.

### B7.Q5 — Même graine, résultats différents : les sources de non-déterminisme 🧠 ★★ ⏱️ 3 min
*Fiche §207.4 · parcours R*

a) Un run appelle seulement `torch.manual_seed(0)`. Lequel de ces tirages n'est pas fixé ? (A) l'initialisation des poids des `nn.Linear` ; (B) le dropout pendant l'entraînement ; (C) une augmentation écrite avec `np.random.rand`.
b) Vrai ou faux : sur GPU, deux exécutions du même code, à la même graine et sur la même machine, peuvent différer à la dernière décimale, puis davantage au fil des epochs.
c) Avec `torch.use_deterministic_algorithms(True, warn_only=True)`, une opération qui n'a pas de version déterministe : (A) lève une `RuntimeError` ; (B) est remplacée par une opération voisine qui en a une ; (C) affiche un avertissement, et le calcul continue.
d) Vrai ou faux : en float64, `(0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3)` vaut `True`.
e) Le `DataLoader(shuffle=True)` d'un run est créé sans `generator`. Tu ajoutes, avant sa création, une ligne qui tire un nombre au hasard avec PyTorch : (A) l'ordre des mini-batches ne change pas ; (B) l'ordre des mini-batches peut changer ; (C) le `DataLoader` lève une erreur.

### B7.Q6 — requirements.txt, fichier de verrouillage, pyproject.toml : qui fige quoi ? 🧠 ⏱️ 3 min
*Fiche §207.1, §207.4 · parcours R*

a) `dependencies = ["torch>=2.5"]`, dans `pyproject.toml`, dit : (A) la version exacte qui a servi au dernier run ; (B) la version à installer sur Colab ; (C) la plus ancienne version que le code accepte.
b) Ton `pyproject.toml` demande `numpy>=2.0`. Quel fichier dit quelle version de NumPy, et de chacune de ses dépendances, le dernier `uv sync` a installée ? (A) `uv.lock` ; (B) `pyproject.toml`, dans sa table `[project]` ; (C) le cache de pip, dans `~/.cache/pip`.
c) Vrai ou faux : une bibliothèque publiée sur PyPI épingle ses dépendances (`numpy==2.1.3`) dans son `pyproject.toml`.
d) Ton run a noté le commit `a1b2c3d`, mais `git status --porcelain` n'était pas vide : (A) une partie du code du run n'est dans aucun commit ; (B) on peut rejouer le run exactement depuis `a1b2c3d` ; (C) le commit enregistré est invalide.
e) Vrai ou faux : deux machines Linux, avec la même version de Python, qui installent le même `uv.lock` obtiennent la même version de NumPy, même si une version plus récente est sortie entre les deux installations.

### B7.Q7 — Que tester dans un projet de machine learning ? 🧠 ⏱️ 3 min
*Fiche §207.6 · parcours R*

a) Ta boucle d'entraînement oublie `opt.zero_grad()`. Quel test le détecte le mieux ? (A) la forme de la sortie du modèle ; (B) l'overfitting d'un mini-batch de 16 images ; (C) le schéma des données d'entrée.
b) 37 images du test sont aussi dans l'entraînement. Quel test le détecte ? (A) l'entraînement fumée, qui fait tourner toute la chaîne sur 200 images ; (B) l'overfitting d'un mini-batch de 16 images de l'entraînement ; (C) un test de données qui compare les empreintes des images.
c) L'entraînement fumée de ta CI finit en 40 s avec une accuracy de 15 %. Il est réussi si : (A) le programme a rendu 0 et laissé un artefact ; (B) son accuracy approche celle du modèle livré, à quelques points près ; (C) l'API, relancée sur son artefact, répond en moins de 100 ms.
d) Vrai ou faux : un entraînement fumée qui rend 0 et écrit son artefact montre que le modèle apprend.
e) Grille du ML Test Score : une équipe totalise 3 points en données, 2,5 en modèle, 4 en infrastructure et 1 en surveillance. Quel est son score, à 1 décimale ?

### B7.Q8 — Artefact de modèle : les poids seuls ne suffisent pas 🧠 ⏱️ 3 min
*Fiche §207.7 · parcours R*

a) Lesquels doivent être dans l'artefact pour que l'API prédise comme l'évaluation hors ligne ? (A) la moyenne et l'écart-type par canal des images d'entraînement ; (B) la liste des classes, dans l'ordre des sorties ; (C) l'état de l'optimiseur ; (D) le learning rate atteint à la fin de l'entraînement. (une ou plusieurs lettres)
b) Le service construit le modèle avec `widths=(16, 32)`, alors que l'artefact a été entraîné avec `widths=(12, 24)`, puis appelle `load_state_dict` : (A) le modèle se charge sans erreur, mais prédit presque au hasard ; (B) `load_state_dict` lève une erreur qui nomme les tenseurs de formes différentes ; (C) PyTorch tronque ou complète par des zéros les poids qui n'ont pas la bonne forme, puis prédit.
c) `save_file(state_dict, "model.safetensors", metadata={"version": 1})` : (A) enregistre la version 1 dans l'en-tête ; (B) écrit le fichier et ignore les métadonnées ; (C) lève une `TypeError`.
d) Vrai ou faux : l'empreinte des données d'entraînement, notée dans l'artefact, permet de vérifier six mois plus tard qu'un réentraînement a utilisé exactement les mêmes images.
e) Le service renvoie la version de l'artefact dans chaque réponse. À quoi sert-elle surtout ? (A) à accélérer le chargement, en gardant en mémoire un modèle par version ; (B) à dire au client dans quel format envoyer les pixels de son image ; (C) à savoir quel modèle a produit chaque prédiction.

### B7.Q9 — Latence p50, p95 et débit : ce que ressent l'utilisateur 🧠 ★★ ⏱️ 3 min
*Fiche §207.7, §207.8 · parcours —*

Sur 1 000 requêtes, 900 prennent 10 ms et 100 prennent 200 ms.

a) Que vaut la latence moyenne, en ms ?
b) Que vaut le p95, en ms ?
c) Vrai ou faux : deux services de même latence moyenne peuvent avoir des p95 très différents.
d) Un autre service reçoit 60 requêtes par seconde, et chacune y reste 150 ms en moyenne. Combien de requêtes y sont en cours, en moyenne ?
e) Un batch de 64 images est traité en 0,4 s. Quel débit, en images par seconde ?
f) Ton service passe d'un appel du modèle par image à un appel par groupe de 32 requêtes. Sur CPU : (A) il sert plus d'images par seconde, mais une requête peut attendre sa réponse plus longtemps ; (B) chaque requête est servie plus vite, puisque le calcul est partagé ; (C) rien ne change, faute de calcul parallèle.

### B7.Q10 — torch.export, ONNX, TorchScript : lequel choisir aujourd'hui ? 🧠 ★★ ⏱️ 3 min
*Fiche §207.8 · parcours —*

a) Un nouveau projet sous PyTorch 2.11 veut un fichier que son service Python recharge sans le code de la classe du modèle : (A) `torch.jit.script`, puis `torch.jit.save` ; (B) `torch.export` et un fichier `.pt2` ; (C) `torch.save(model)`, l'objet entier par pickle.
b) Le service est écrit en C# et ne doit pas dépendre de PyTorch : (A) ONNX, servi par ONNX Runtime ; (B) `torch.compile` ; (C) le `state_dict` en safetensors.
c) Vrai ou faux : `torch.compile` écrit un fichier que l'on peut servir sans le code Python du modèle.
d) Un modèle exporté sur un exemple de batch 8, sans `dynamic_shapes`, reçoit un batch de 3 : (A) il calcule normalement ; (B) une garde refuse l'entrée ; (C) il complète le batch par des zéros.
e) Vrai ou faux : dans PyTorch 2.11, `torch.onnx.export` passe encore par défaut par TorchScript, comme dans la 2.8.
f) Le `forward` d'un modèle contient `if x.mean() > 0:`. `torch.export.export` : (A) échoue, car le graphe dépendrait de la valeur des données ; (B) garde les deux branches et choisit à l'exécution ; (C) réussit, et ne garde que la branche prise par l'exemple.

### B7.Q11 — Écart entraînement-production et dérive des données : vrai ou faux 🧠 ⏱️ 3 min
*Fiche §207.7, §207.10 · parcours R*

a) Vrai ou faux : un modèle dont les poids ne changent pas peut voir son accuracy baisser en production.
b) Vrai ou faux : le training-serving skew vient d'un changement du monde après le déploiement.
c) Vrai ou faux : avec 50 000 requêtes par fenêtre, un test KS peut donner une p-valeur de 0,001 pour un décalage sans importance pratique.
d) Vrai ou faux : la distribution des classes prédites peut donner l'alerte avant que les vrais labels n'arrivent.
e) Une nouvelle marque de téléphone, aux photos plus saturées, devient majoritaire chez les utilisateurs : (A) une dérive des entrées ; (B) un training-serving skew ; (C) un *concept shift*.
f) Vrai ou faux : des intervalles de PSI recalculés chaque semaine sur la production donnent des PSI comparables d'une semaine à l'autre.

<a id="rappels"></a>

## 🔁 Rappels

### B7.R1 — Ch. 24 : ce qu'une ligne de journal d'expériences doit contenir 🔁 ★ ⏱️ 5 min
*Ch. 24 (journal d'expériences, 🛠️ 24.15) · fiche §207.5 · parcours R*

a) Lesquels faut-il noter pour pouvoir rejouer un run ? (A) le commit git du code ; (B) la graine ; (C) tous les hyperparamètres, y compris ceux laissés par défaut ; (D) les versions de Python et de PyTorch. (une ou plusieurs lettres)
b) Un run a divergé (loss `nan` à l'epoch 3) : (A) sa ligne reste, marquée `"status": "diverged"` ; (B) tu effaces sa ligne, qui fausserait les moyennes ; (C) tu le relances à la même graine et tu remplaces sa ligne.
c) Vrai ou faux : pour ajouter un champ `warmup_steps` aux runs à venir, il faut d'abord réécrire les anciennes lignes du journal JSONL avec ce champ, sinon elles ne se liront plus.
d) Un journal JSONL contient 7 runs ; tu en relances 2, avec d'autres graines. Combien de lignes contient-il ensuite ?
e) Dans l'identifiant de run `20261104-161731-e06339`, que représente `161731` ? (A) la graine du run, recopiée de sa configuration ; (B) les six premiers caractères du commit git du code ; (C) l'heure du lancement, 16 h 17 min 31 s (UTC).

### B7.R2 — Ch. 23 : checkpoint complet, state_dict et weights_only 🔁 ★ ⏱️ 5 min
*Ch. 23 (§23.9, sauvegarder et recharger) · fiche §207.7 · parcours R*

a) Pour reprendre un entraînement exactement là où il s'est arrêté, que faut-il sauvegarder, en plus de l'epoch ? (A) le `state_dict` du modèle ; (B) le `state_dict` de l'optimiseur ; (C) l'objet `DataLoader` entier, par pickle ; (D) l'état des générateurs aléatoires. (une ou plusieurs lettres)
b) Pour servir le modèle dans une API, il te faut : (A) le checkpoint complet, optimiseur compris ; (B) le `state_dict` et le code qui reconstruit l'architecture ; (C) l'état de l'optimiseur, avec les moments qu'Adam garde pour chaque poids.
c) Vrai ou faux : `torch.load(path, weights_only=True)` refuse un fichier qui contient un objet Python quelconque, comme une instance de ta classe `Trainer`.
d) Vrai ou faux : `model.load_state_dict(...)` reconstruit l'architecture à partir des noms des tenseurs du fichier.
e) Tu reprends un entraînement avec Adam en rechargeant le modèle, mais pas l'optimiseur : (A) Adam repart de moments nuls : ses premiers pas changent ; (B) rien ne change, car Adam recalcule ses moments à partir des poids rechargés ; (C) `load_state_dict` lève une erreur, faute de l'état de l'optimiseur.

### B7.R3 — Ch. 15 : sauvegarder un Pipeline et figer les versions 🔁 ★ ⏱️ 5 min
*Ch. 15 (« sauvegarder un modèle et ses versions », 🛠️ 15.30) · fiche §207.4, §207.7 · parcours R*

a) Tu livres un classifieur scikit-learn dont les features sont standardisées : (A) tu sauvegardes le `Pipeline` entier, prétraitement compris ; (B) le modèle seul, le service standardisera ; (C) les seuls coefficients du modèle, dans un CSV plus lisible qu'un binaire.
b) Vrai ou faux : un fichier `joblib` écrit avec scikit-learn 1.6 se relit de façon garantie avec scikit-learn 1.9.
c) Charger avec scikit-learn 1.9 un modèle écrit avec la 1.6 affiche : (A) un `FutureWarning` ; (B) un `InconsistentVersionWarning` ; (C) rien.
d) Vrai ou faux : charger un fichier `joblib` d'une source inconnue peut exécuter du code arbitraire.
e) Lesquels vont dans le fichier de métadonnées, à côté du modèle ? (A) `sklearn.__version__` ; (B) `numpy.__version__` ; (C) le score de validation ; (D) une copie des données d'entraînement, pour pouvoir vérifier le modèle plus tard. (une ou plusieurs lettres)
f) Pour servir des prédictions sans Python : (A) `skops.io` ; (B) `joblib` ; (C) ONNX.

<a id="papier"></a>

## ✏️ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde au moins 4 décimales pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex B7.1 — Taille d'un artefact : float32, bfloat16, int8, et la mémoire au chargement ✏️ ★ ⏱️ 10 min
**Objectif :** chiffrer la taille des poids d'un modèle selon leur type, ce que rapporte une quantification partielle, et la mémoire qu'il faut pour le charger.
**Prérequis :** ch. 20 (paramètres, buffers, device `meta`) · fiche §207.8 (rappel maths « précision réduite et taille d'un modèle », quantification dynamique) · **Parcours :** M

ConvNeXt-Tiny (torchvision) a $N = 28\,589\,128$ paramètres et aucun buffer ; ses couches denses (`nn.Linear`) en portent $26\,646\,528$ dans leurs matrices de poids et $34\,120$ dans leurs biais. On néglige l'en-tête des fichiers et les échelles de la quantification.

a) La taille de ses poids en float32, en Mo, à 1 décimale.
b) La même taille en Mio, à 1 décimale.
c) La taille en bfloat16, en Mo, à 1 décimale.
d) La taille si tous les paramètres passent en int8, en Mo, à 1 décimale.
e) La quantification dynamique des seules couches denses range leurs **matrices** en int8 ; leurs biais et tous les autres paramètres restent en float32. Quelle taille, en Mo, à 1 décimale ?
f) Combien de fois e) est-il plus petit que a), à 1 décimale ?
g) Chargement naïf : `build_model()` alloue des poids aléatoires en float32, puis `torch.load` lit une seconde copie complète avant `load_state_dict`. Quel pic de mémoire pour les poids, en Mio, à 1 décimale ?

Le VGG livré par la solution du MP5 a 72 954 paramètres et six batchnorms, de 16, 16, 32, 32, 64 et 64 canaux ; chacune a deux buffers float32 d'une valeur par canal (`running_mean`, `running_var`) et un compteur `num_batches_tracked` en int64. Sa seule couche dense, `Linear(64, 10)`, porte 650 paramètres.

h) Combien d'octets occupent les tenseurs de son `state_dict` (paramètres et buffers) ?
i) Son fichier safetensors pèse 296 464 octets. Combien d'octets d'en-tête ?
j) Quelle part de ses paramètres porte sa couche dense, en %, à 1 décimale ?

Dans ta copie : pourquoi la quantification dynamique divise-t-elle par plus de 3 la taille de ConvNeXt-Tiny, et pourquoi ne réduirait-elle presque pas le fichier du VGG ? Que contient l'en-tête de i) ? Comment la fiche (207.8) abaisse-t-elle le pic g), et pourquoi « la taille du fichier n'est qu'un plancher » ?

### Ex B7.2 — Cinq graines, un intervalle : la dispersion d'une accuracy rapportée ✏️ ★★ ⏱️ 20 min
**Objectif :** rapporter une accuracy avec son intervalle de Student sur cinq graines, puis comparer deux modèles graine par graine.
**Prérequis :** ch. 2 (moyenne, écart-type corrigé, erreur type, bootstrap) · fiche §207.4 (« Rapporter un score : plusieurs graines et un intervalle », rappel maths sur l'intervalle de Student) · **Parcours :** M

Ton classifieur A est entraîné cinq fois, avec les graines 0 à 4 et les mêmes réglages. Ses accuracies de validation, en % : 64,2 ; 65,0 ; 63,6 ; 64,8 ; 64,4. Un modèle B, entraîné avec les mêmes graines et évalué sur la même validation, donne, graine par graine : 64,9 ; 65,3 ; 64,5 ; 65,6 ; 64,7. L'intervalle à 95 % est $\bar a \pm t\,\frac{s}{\sqrt n}$, avec $s$ l'écart-type corrigé (divisé par $n - 1$) et $t = t_{0{,}975;\,4} = 2{,}776$, le quantile de la **loi de Student** à $n - 1 = 4$ degrés de liberté (table de 207.4).

a) La moyenne de A, à 1 décimale.
b) L'écart-type corrigé $s$ de A, à 3 décimales.
c) L'erreur type de A, à 3 décimales.
d) La demi-largeur de l'intervalle de A, à 2 décimales.
e) La borne basse de l'intervalle de A, à 2 décimales.
f) Vrai ou faux : l'intervalle de A et celui de B, calculés chacun de la même façon, se chevauchent.
g) La moyenne des différences B − A, graine par graine, à 1 décimale.
h) La demi-largeur de l'intervalle à 95 % de cette différence (le même calcul sur les cinq différences), à 2 décimales.
i) Vrai ou faux : l'intervalle de la différence contient 0.

Dans ta copie : écris le résultat de A comme dans un rapport. Pourquoi f) et i) ne disent-ils pas la même chose, et lequel croire ? Combien de rééchantillonnages différents (à l'ordre près) un bootstrap peut-il tirer de cinq valeurs, et pourquoi est-ce trop peu ? Que rapporterait quelqu'un qui ne garde que la meilleure graine ?

### Ex B7.3 — Latence et débit : p50 et p95 sur vingt mesures, batch 1 contre batch 32 ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer p50 et p95 avec la convention de NumPy, en tirer un débit, et comparer un service image par image à un service par batches.
**Prérequis :** ch. 2 (percentiles) · fiche §207.7 (« Latence, percentiles et débit ») · **Parcours :** M

Vingt appels du modèle sur une image (batch 1), mesurés après quelques appels à blanc, ont pris, en ms : 21,3 ; 19,8 ; 20,4 ; 22,1 ; 19,6 ; 20,9 ; 20,1 ; 35,7 ; 20,6 ; 19,9 ; 21,0 ; 20,2 ; 23,4 ; 19,7 ; 20,5 ; 47,9 ; 20,8 ; 20,3 ; 21,6 ; 20,0. Vingt appels sur un batch de 32 images donnent un p50 de 118 ms et un p95 de 131 ms. Les percentiles suivent la convention de l'encadré du début (méthode par défaut de `np.percentile`).

a) La latence moyenne du batch 1, à 2 décimales.
b) Son p50, à 2 décimales.
c) Son p95, à 2 décimales.
d) Combien des vingt mesures dépassent la moyenne ?
e) Le débit du batch 1 calculé avec son p50, en images par seconde, arrondi à l'entier.
f) Le débit du batch 32 calculé avec son p50, en images par seconde, arrondi à l'entier.
g) Combien de fois le débit du batch 32 est-il plus grand que celui du batch 1 (débits non arrondis), à 1 décimale ?
h) Le temps de calcul par image dans un batch de 32 (le p50 du batch divisé par 32), en ms, à 2 décimales.
i) Vrai ou faux : l'objectif « 95 % des images servies en moins de 100 ms » est tenu si chaque image est servie dans un batch de 32 (une image attend la fin de son batch, et l'on ne compte pas le temps passé à le remplir).
j) Une page affiche 8 prédictions, demandées l'une après l'autre, de latences indépendantes. Quelle probabilité qu'au moins une dépasse le p95, à 2 décimales ?

Dans ta copie : sur vingt mesures, combien de mesures décident du p95 ? Que donnerait `method="inverted_cdf"` ? Pourquoi la moyenne est-elle mauvaise ici ? Si le serveur doit attendre que 32 requêtes arrivent avant de calculer, qu'ajoute cette attente à la latence de chaque image ? Quel mode choisirais-tu pour une API interactive, et pour un traitement de nuit de 100 000 images ?

### Ex B7.4 — Indice de stabilité de population (PSI) à la main sur quatre intervalles ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer un PSI à la main, le relier aux deux divergences KL, et voir ce qu'un intervalle vide fait au résultat.
**Prérequis :** ch. 6 (divergence KL, logarithme naturel) · fiche §207.10 (« L'indice de stabilité de population », seuils, intervalle vide) · **Parcours :** M

On surveille la luminosité moyenne des images reçues. Les bornes des quatre intervalles $I_1$ (le plus sombre) à $I_4$ (le plus clair) sont les **quartiles** de la référence (1 000 images de validation) : $q_k = 0{,}25$ pour tout $k$. Chaque semaine, on range 200 images de production :

| Semaine | $I_1$ | $I_2$ | $I_3$ | $I_4$ |
|---|---|---|---|---|
| 1 | 38 | 52 | 54 | 56 |
| 2 | 96 | 62 | 30 | 12 |
| 3 | 120 | 56 | 24 | 0 |

$\operatorname{PSI} = \sum_{k=1}^{4} (p_k - q_k)\ln\frac{p_k}{q_k}$, avec $p_k$ la part de la production dans $I_k$ et le logarithme naturel. Pour un intervalle vide, on ajoute $\varepsilon$ à **toutes** les parts, de la production et de la référence, sans renormaliser ; $\varepsilon = 10^{-4}$ sauf mention contraire.

a) La contribution de $I_1$ au PSI de la semaine 1, à 4 décimales.
b) Le PSI de la semaine 1, à 3 décimales.
c) Le PSI de la semaine 2, à 3 décimales.
d) Quel intervalle contribue le plus au PSI de la semaine 2 ? (A) $I_1$ ; (B) $I_2$ ; (C) $I_3$ ; (D) $I_4$.
e) $D_{\text{KL}}(q\,\|\,p) = \sum_k q_k \ln\frac{q_k}{p_k}$ pour la semaine 2, à 3 décimales.
f) Selon les seuils usuels (0,1 et 0,25), les semaines 1 et 2 tombent dans : (A) « pas de changement notable » et « petit changement » ; (B) « pas de changement notable » et « changement important » ; (C) « petit changement » et « changement important ».
g) Vrai ou faux : sans $\varepsilon$, NumPy rend un PSI infini pour la semaine 3.
h) Le PSI de la semaine 3, avec $\varepsilon = 10^{-4}$, à 2 décimales.
i) Le PSI de la semaine 3, avec $\varepsilon = 10^{-3}$, à 3 décimales.

Dans ta copie : vérifie que le PSI de la semaine 2 vaut $D_{\text{KL}}(p\,\|\,q) + D_{\text{KL}}(q\,\|\,p)$. Pourquoi chaque terme est-il positif ou nul ? Que disent h) et i) de la façon de rapporter un PSI avec un intervalle vide ? Propose une autre façon de traiter cet intervalle. Que suggère l'évolution des trois semaines sur les images reçues, et que ferais-tu avant de réentraîner ?

### Ex B7.5 — Fermi : servir 100 requêtes par seconde — cœurs CPU, regroupement en batches, coût mensuel 🧮 ★★ ⏱️ 20 min
**Objectif :** dimensionner à la louche un service de prédiction sur CPU : cœurs nécessaires, requêtes en cours, effet du regroupement en batches et coût mensuel.
**Prérequis :** Ex B7.3 · fiche §207.7 (inférence en ligne ou par batches, débit, loi de Little) · **Parcours :** R, M

Hypothèses (des ordres de grandeur, à prendre telles quelles) :
- le trafic : 100 requêtes par seconde en moyenne, avec un pic de 300 requêtes par seconde deux heures par jour ; chaque requête porte une image ;
- sur un cœur, le modèle traite une image seule en 4 ms et un batch de 16 en 24 ms ; entre les deux, un batch de $b$ images prend à peu près $2{,}7 + 1{,}33\,b$ ms ;
- chaque requête coûte en plus 3 ms de CPU que le regroupement ne partage pas (lire le JSON, valider, prétraiter, répondre) ;
- l'objectif : un p95 sous 100 ms ; pour que les files d'attente restent courtes, on ne charge un cœur qu'à 60 % en moyenne ;
- une machine virtuelle de 4 vCPU coûte de 0,15 à 0,25 € de l'heure (prix indicatif) ; un mois compte 730 heures ; on compte un vCPU comme un cœur ;
- pour la question 5 seulement : une machine de 4 vCPU avec un GPU d'entrée de gamme coûte de 0,50 à 1 € de l'heure, et son GPU traite un batch de 64 images du modèle lourd en environ 15 ms.

Réponds dans ta copie, avec des ordres de grandeur ; des fourchettes sont bienvenues.

1. **Sans regroupement.** Quel temps de CPU par requête ? Combien de cœurs sont occupés en moyenne à 100 requêtes par seconde, et combien en faut-il avec la règle des 60 % ? Et au pic ?
2. **Les requêtes en cours.** Une requête reste en moyenne 30 ms dans le service hors pic et 60 ms au pic (calcul, attente et réseau). Combien de requêtes sont en cours dans chaque cas ? Qu'en déduis-tu sur le nombre de requêtes que le serveur doit accepter en même temps, comparé au nombre de cœurs ?
3. **Le regroupement.** Le serveur attend au plus 10 ms pour remplir un batch, de 16 images au plus. Combien de requêtes arrivent en 10 ms, hors pic et au pic ? Au pic, quel temps de CPU par requête, et quel gain par rapport à 1 ? Quel prix en latence ? Pourquoi un batch de 16 est-il hors de portée, hors pic comme au pic ?
4. **Le coût.** Combien de machines de 4 vCPU faut-il pour tenir le pic, sans et avec regroupement ? Avec une seconde machine pour la redondance, quel coût mensuel ?
5. **Un modèle dix fois plus lourd.** Le modèle prend maintenant 40 ms pour une image et $27 + 13{,}3\,b$ ms pour un batch de $b$. Refais 1 et 3 au pic. Compare le coût mensuel de la flotte CPU, redondance comprise, à celui d'une machine GPU et de sa remplaçante : quand un GPU devient-il intéressant ?
6. En deux phrases : qu'est-ce qui pèse le plus dans le coût de ce service, et quel est le premier levier que tu essaierais ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B7.6 — « Ça marche dans mon notebook » : convaincre un manager d'industrialiser, en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à une personne qui n'est pas du métier pourquoi un modèle de notebook ne se met pas en production tel quel, et ce que l'industrialisation lui rapporte.
**Prérequis :** fiche §207.1 (« Ce que cache un notebook »), §207.6, §207.10 · **Parcours :** R

Ta manager, qui n'est pas développeuse, a vu ton notebook atteindre 85 % d'accuracy sur CIFAR-10 et veut « le mettre en ligne vendredi ». Tu demandes trois jours pour en faire un projet. Explique-lui, en cinq lignes au plus et sans jargon (ni « layout `src/` », ni « CI », ni « pytest »), pourquoi ces trois jours. Ton explication doit faire apparaître (1) le risque concret d'un notebook en production, en une image ; (2) ce que tu vas faire, en mots simples ; (3) ce qu'elle y gagne, elle, dans six mois ; (4) un chiffre ou un fait qui rend le risque crédible ; (5) une proposition qui respecte son délai.

### Ex B7.7 — Journaliser les prédictions en production : minimisation, durée de conservation, droits des personnes ⚖️ ★★ ⏱️ 20 min
**Objectif :** concevoir le journal des prédictions d'un service d'images qui respecte le RGPD sans perdre ce qu'il faut pour surveiller le modèle.
**Prérequis :** ch. 23 (fiche modèle) · ch. 29 (données personnelles) · fiche §207.7 (API, code 422), §207.10 (dérive, « Journaliser les prédictions, et le RGPD ») · **Parcours :** R

Une plateforme française de petites annonces entre particuliers ajoute ton classifieur à son application : quand un vendeur photographie un objet, l'API lui propose une catégorie (dix classes). L'équipe veut tout journaliser « pour surveiller et réentraîner » : pour chaque requête, l'heure, l'identifiant du compte, l'adresse IP, la photo entière (souvent prise chez soi), la classe proposée, sa probabilité, la catégorie finalement choisie par le vendeur, la latence et la version du modèle, le tout gardé sans limite de durée. (Cas inventé pour l'exercice.) Réponds dans ta copie.

1. **Les données personnelles.** Lesquelles de ces informations sont des données personnelles ? Pourquoi la photo peut-elle en être une, même quand elle montre un objet ? Un identifiant de compte remplacé par un pseudonyme (son empreinte calculée avec une clé secrète) en reste-t-il une ?
2. **La minimisation (art. 5, 1 c).** De quoi as-tu vraiment besoin pour surveiller la dérive et la qualité (207.10) ? Propose les champs du journal, et ce qui remplace la photo entière.
3. **La finalité (art. 5, 1 b).** L'équipe veut aussi réentraîner le modèle sur les photos. Est-ce la même finalité que la surveillance ? Que faudrait-il vérifier ou prévoir (art. 6, paragraphe 4 ; l'information des vendeurs) ?
4. **La conservation (art. 5, 1 e).** Propose une durée écrite pour chaque type de donnée de ton journal (lignes détaillées, échantillon éventuel de photos, statistiques agrégées) et la façon de la faire respecter. Pourquoi la fourchette de six mois à un an de la délibération CNIL n° 2021-122 ne se transpose-t-elle pas telle quelle ?
5. **Les droits (art. 15 et 17).** Un vendeur demande ce que le service garde sur lui, puis l'effacement. Que doit permettre ton journal, et comment, s'il est pseudonymisé ?
6. **La sécurité (art. 25 et 32).** Ta réponse 422 renvoie l'entrée refusée (le champ `input` de FastAPI), et un `log.exception` écrit la requête entière dans les journaux techniques. Que changes-tu, dans le code et dans l'accès aux journaux ?
7. Rédige les cinq lignes « Journalisation et données » de la fiche modèle.

### Ex B7.8 — Sculley et coll. (2015) : « Hidden Technical Debt in Machine Learning Systems » 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a nommé la dette technique propre au ML, et relier chacun de ses risques à une pratique de ce chapitre.
**Prérequis :** MP5 (ton projet `cifarkit`) · fiche §207.1, §207.10 (« La dette technique du ML ») · **Parcours :** —

Lis D. Sculley et coll., « Hidden Technical Debt in Machine Learning Systems » (NeurIPS 2015, [PDF officiel des actes](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf), huit pages et la bibliographie) : le résumé, l'introduction et la figure 1, puis les sections 2 à 9. Réponds dans ta copie.

1. **La métaphore** (résumé, section 1). Qui a introduit la dette technique, et en quelle année ? Pourquoi les auteurs disent-ils qu'une dette **cachée** est dangereuse ? Que l'article ne propose-t-il pas ?
2. **La figure 1.** Nomme ses boîtes. Lesquelles ton projet du MP5 a-t-il déjà, lesquelles ce chapitre ajoute-t-il, lesquelles restent absentes ?
3. **Les frontières qui s'effacent** (section 2). Définis le principe CACE avec un exemple tiré de ton projet. Qu'est-ce qu'un consommateur non déclaré, et quelles parades les auteurs proposent-ils ? Quel lien avec ton API (📦 B7.18) ?
4. **Les dépendances de données** (section 3). Quelle parade les auteurs proposent-ils contre une dépendance instable, et à quel prix ? Cite les quatre façons dont une dépendance sous-utilisée s'installe, et la façon de la détecter. Contre lesquels de ces risques l'empreinte des données (207.4) protège-t-elle, et contre lesquels ne fait-elle rien ?
5. **Les boucles de rétroaction** (section 4). Distingue boucle directe et boucle cachée ; donne la parade de l'article contre la première et l'un de ses exemples de la seconde. Imagine une boucle cachée pour la plateforme de ⚖️ B7.7.
6. **Les anti-patterns** (section 5). Quelle part du code d'un système mûr est du code de colle, selon les auteurs, et quelle parade proposent-ils ? Qu'est-ce qu'une jungle de pipelines ? Qu'est-il arrivé à Knight Capital, et à quel anti-pattern les auteurs l'attribuent-ils ? Cite les trois « odeurs » de la fin de la section.
7. **La dette de configuration** (section 6). Recopie les six principes d'un bon système de configuration. Lesquels la configuration de 207.3 (dataclass, YAML, validation, surcharges, fichiers versionnés) respecte-t-elle, et lequel lui manque le plus ?
8. **Le monde extérieur** (section 7). Pourquoi un seuil fixé à la main vieillit-il mal ? Décris les trois points de départ de la surveillance que propose l'article, et ce que chacun donnerait pour ton classifieur.
9. **Mesurer la dette** (sections 8 et 9). Cite les quatre autres dettes de la section 8. Réponds aux cinq questions de la section 9 pour ton projet.
10. **Et aujourd'hui.** Donne deux outils ou pratiques, apparus depuis 2015, qui répondent à des problèmes de l'article, et un problème de l'article qu'aucun outil ne règle.

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### B7.E1 — Comment passes-tu d'un notebook à du code prêt pour la production ? 💼 ★★ ⏱️ 10 min
*Fiche §207.1, §207.2, §207.3 · parcours R*

« Vous avez un notebook d'exploration qui donne de bons résultats. Comment le transformez-vous en code qu'une équipe peut maintenir et déployer ? »

### B7.E2 — Comment garantis-tu la reproductibilité d'un entraînement ? 💼 ★★ ⏱️ 10 min
*Fiche §207.4, §207.5 · parcours R*

« Un collègue n'arrive pas à retrouver votre accuracy de 85 %. Que faites-vous pour qu'un entraînement soit reproductible, et comment rapportez-vous le résultat ? »

### B7.E3 — Que testes-tu dans un projet de machine learning ? 💼 ★★ ⏱️ 10 min
*Fiche §207.6 · parcours R*

« Au-delà des tests unitaires habituels, que testez-vous dans un projet de machine learning, et quand ces tests tournent-ils ? »

### B7.E4 — Comment surveilles-tu un modèle en production, et que fais-tu en cas de dérive ? 💼 ★★ ⏱️ 10 min
*Fiche §207.10 · parcours R*

« Votre modèle est en production depuis six mois. Comment savez-vous qu'il fonctionne encore, et que faites-vous si les données dérivent ? »

### B7.E5 — Tu dois exposer ton modèle par une API : quels choix fais-tu (batches, latence, versions) ? 💼 ★★ ⏱️ 10 min
*Fiche §207.7, §207.8 · parcours R*

« Vous devez exposer votre classifieur d'images par une API. Requêtes une par une ou par batches, latence, versions du modèle : quels choix faites-vous, et comment les vérifiez-vous ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`, en trois phases, de difficulté croissante dans chacune : **le projet** (B7.9 à B7.15 : ton notebook devient un paquet installable, avec sa ligne de commande, sa configuration, des runs reproductibles et journalisés, et ses tests), **le service** (B7.16 à B7.21 : les écarts entre entraînement et production, l'artefact du modèle, l'API, sa latence, l'accélération sur CPU et l'export), **l'exploitation** (B7.22 à B7.25 : la dérive, l'intégration continue, le Dockerfile et le défi). Les cellules du notebook écrivent les fichiers de ton paquet dans ton dossier de travail et les exécutent ; l'API tourne dans un thread ; le Dockerfile se lit et se complète, sans Docker.

| ID | Titre | Type | ★ | ⏱️ | 🚀 |
|---|---|---|---|---|---|
| B7.9 | Du notebook au paquet : data.py, model.py, train.py, pyproject.toml et installation éditable | 📦 | ★★★ | 60 | |
| B7.10 | Une interface en ligne de commande : argparse, logging et codes de sortie | 📦 | ★★ | 30 | |
| B7.11 | Configuration : dataclass + YAML, surcharges en ligne de commande, validation | 📦 | ★★ | 30 | |
| B7.12 | Prédire : même graine, puis num_workers = 4, puis GPU — résultats identiques ? | 🔮 | ★ | 10 | |
| B7.13 | Mesurer le non-déterminisme : graines fixées ou non, algorithmes déterministes, cinq exécutions | 🔬 | ★★ | 30 | |
| B7.14 | Suivi d'expériences maison : identifiant de run, configuration, métriques et artefacts en JSONL | 📦 | ★★ | 30 | |
| B7.15 | Tests pytest pour le ML : formes, overfitting d'un mini-batch, prétraitement, schéma des données | 📦 | ★★★ | 45 | |
| B7.16 | Le modèle qui marche dans le notebook et pas en production : cinq écarts à retrouver | 🐛 | ★★ | 30 | |
| B7.17 | L'artefact du modèle : safetensors, configuration, normalisation, classes, version, empreinte des données | 📦 | ★★ | 30 | |
| B7.18 | Une API de prédiction avec FastAPI (repli : http.server) : /health, /predict, validation des entrées | 📦 | ★★★ | 45 | |
| B7.19 | Prédire la latence p95 de l'API, requête par requête puis par batches de 32 | 🔮 | ★ | 10 | |
| B7.20 | Accélérer l'inférence sur CPU : batches, threads, inference_mode, bfloat16, quantification | 🔬 | ★★ | 30 | |
| B7.21 | Exporter avec torch.export et vérifier l'équivalence numérique (ONNX en option) | 📦 | ★★ | 25 | |
| B7.22 | Surveiller la dérive : PSI et test KS sur des images assombries ou bruitées, quand alerter ? | 📈 | ★★ | 25 | |
| B7.23 | Intégration continue : un workflow GitHub Actions qui installe, teste et lance un entraînement fumée | 🛠️ | ★★ | 25 | |
| B7.24 | Lire et compléter un Dockerfile : image de base, dépendances CPU, utilisateur non root, variables | 📦 | ★★ | 20 | |
| B7.25 | Défi : un dépôt prêt pour la production, de git clone à une prédiction servie en moins de 10 min de CPU | 🏆 | ★★★★ | 180 | |

Le notebook part de ton classifieur CIFAR-10 du MP5 (ou d'un CNN de référence fourni, avec son checkpoint) et de 2 000 images en mode rapide. Tout tourne sur CPU ; aucun exercice n'est marqué 🚀.
