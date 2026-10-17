# Audit P5 des parties III et IV : A3, conformité structurelle (BIBLE §11, §12, §17, §19)

**Périmètre** : ch. 12 à 20 (437 exercices), CP3, CP4, MP3, MP4, au commit `5a17b91`. Dépôt lu sans modification (`git -C /home/claude/workbookia status --short` vide à la fin). Reconstructions et tests dans une copie `tar` (`A03_structure/copy/`).
**Méthode** : un script par contrôle, dans `A03_structure/` : `check1_exercises.py` (contrat ↔ 02, 03, 05.ipynb, tableau de bord, 04, 05, 06), `check1b_06.py`, `check1c_solutions.py`, `cmp_styles.py` (rubriques de 05 de 0A à 20), `check1d` (en-têtes), `check2_fiche.py`, `check4_notebooks.py`, `check5_composition.py`, `check6_hints.py` et `check6b_codecov.py` (indices 3), `check7_scaffold.py` (échafaudage), `check9_checkpoints.py`, `compare_rebuild.py`. Sorties brutes : `raw_*.txt` et `raw_*.json`. Chaque écart retenu a été relu dans le fichier ; les faux positifs ont été écartés à la main : préfixe « Ch. N : » des rappels, titres de 06 en réponse rédigée, `WB_ANSWER` qui contient un « ⚠️ » de message, « données synthétiques » pour `synth`.

## Contrôles globaux

- `python tools/syllabus.py check` : 46 chapitres, 2 024 exercices, 0 problème.
- `python tools/build_answers.py --check` : ✅, 3 348 réponses à jour.
- `python tools/export_flashcards.py --check` : ✅, 548 cartes.
- `pytest tests/infra`, lancé dans la copie : 289 réussis, 4 ignorés (réseau).
- Reconstruction des 26 notebooks du périmètre par leurs 13 scripts (`build_ch12.py` à `build_ch20.py`, `build_cp3.py`, `build_cp4.py`, `build_mp3.py`, `build_mp4.py`) : types, sources et tags **identiques** aux notebooks publiés, cellule par cellule (`raw_rebuild.txt`).
- Durées des notebooks de solutions, lues dans les métadonnées d'exécution publiées (rien n'a été relancé pour mesurer) :
  - chapitres : de 22,2 s (ch. 16) à 164,3 s (ch. 14) ; la cellule la plus longue prend 44,5 s (ch. 18, cellule 98) ;
  - CP3 4,0 s, CP4 2,5 s, MP3 120,7 s (une cellule de 43,3 s), MP4 62,0 s.
  - Le budget FAST_MODE du §4 (moins de 10 min par notebook, aucune cellule au-delà de 3 min) est tenu partout, avec une marge d'au moins 3,6×.
- Notebooks d'exercices (9 chapitres, 2 examens, 2 kits de départ) :
  - badge Colab au bon chemin, une seule cellule `setup` ;
  - ni sortie, ni `!`, ni `%` ;
  - « Objectifs et rappel express », puis la partie 0, puis les parties A à D ou F ;
  - une cellule finale « ✅ Bilan » avec 3 questions d'auto-évaluation et « Pour aller plus loin » (les examens n'en ont pas, comme CP1 et CP2 ; les MP finissent par la grille, comme MP1 et MP2).
- Notebooks de solutions : toutes les cellules sont exécutées, sans erreur, traceback, ❌ ni ⏳. Un seul ⚠️ subsiste (constat 6).

## Tableau récapitulatif

✅ = aucun écart non justifié. Sinon, le numéro du constat est entre parenthèses.

| Ch. | 1 · contrat ↔ 02, 03, tableau de bord, 04, 05, 06 | 2 · 05 : rubriques du §12 | 3 · fiche (10 rubriques, ⏩, 3-5 ressources, temps) | 4 · ordre de 02 | 5 · notebooks 03 / 05 (budget) | 6 · composition §11 | 7 · indices 3 (échantillon manuel) | 8 · échafaudage 🔨/📦 (mots, moyenne) | 9 · flashcards | 10 · annexes, suivi, data cards |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | ✅ | quiz et rappels : « À retenir » seul (S 17) | ✅ (⏩ = SYLLABUS ; 4,9 h) | ✅ | ⚠️ 12.20d (6) ; 92 s | 🔨/📦 13, justifié | ✅ | 229 | 26 ✅ | auto-éval avant CP2 (10) ; PROGRESS « 56 » (11) |
| 13 | ✅ | idem (S 17) | 6 ressources (8) | 🧮 en papier (7) | ✅ 34 s | 🔨/📦 14, justifié | 13.3 c (4) | 280 | 30 ✅ | data card Penguins (12) |
| 14 | ✅ ; 04 sans icônes (5) | idem (S 17) | 7 ressources (8) | ✅ | ✅ 164 s | ✅ | ✅ | 288 | 25 ✅ | ✅ |
| 15 | ✅ ; 04 sans icônes (5) | idem (S 17) | ✅ | ✅ (🧮 en réflexion) | ✅ 55 s | 📦 13, justifié | ✅ | 233 | 25 ✅ | formulaire sans titre (13) |
| 16 | ✅ ; 04 sans icônes (5) | idem (S 17) | ✅ | 🧮 en papier (7) | ✅ 22 s | ✅ (📄, 🧮) | ✅ | 198 | 18 ✅ | ✅ |
| 17 | 17.9 « – » (S 18) ; 04 sans icônes (5) | 💼 sans relances (2) | ✅ | 🧮 en papier (7) | ✅ 34 s | ✅ | 17.12 (3) | 201 | 22 ✅ | ✅ |
| 18 | ✅ | 💼 sans relances (2) | ✅ | 🧮 en papier (7) | ✅ 162 s | ✅ | 18.26 (3) | 261 | 28 ✅ | ✅ |
| 19 | ✅ ; 04 sans icônes (5) | 💼 (2) ; 🗣️ 19.10 (2) | ✅ | 🧮 en papier (7) | ✅ 98 s | ✅ | 19.16-19.18 (3) | 290 | 26 ✅ | ✅ |
| 20 | ✅ ; 04 sans icônes (5) | 💼 (2) ; 🗣️ 20.10 (2) | ✅ | 🧮 en papier (7) | ✅ 87 s | ✅ | 20.R2 b (4) ; 20.23 (1) | 261 | 28 ✅ | glossaire (14) |

Repère pour la colonne 8 : de 194 à 268 mots aux ch. 5 à 11 (moyennes par chapitre, même script).

**Lecture des colonnes**

- **1.** Les 437 exercices concordent partout avec leur contrat (ID, titre, type, ★, durée, parcours) : `02_exercices.md` (en-têtes et tableau du notebook), `03_notebook.ipynb` (tableau et en-têtes), en-têtes de `05_solutions.ipynb`, `suivi/tableau_de_bord.md` (lignes et groupes Q, R, E, avec leurs totaux), `04_indices.md`, `05_solutions.md` et `06_mes_reponses.md`.
  - Aucun ID ne manque ni n'est en trop.
  - Chaque exercice a exactement trois `<details>`, « Indice 1 », « Indice 2 », « Indice 3 ».
  - 06 a une place pour chaque sous-question de 02, pour chaque « Dans ta copie », et une liste « Notes sur le notebook » égale aux ID du notebook.
  - En-têtes : « Objectif » et « Prérequis » partout ; « Fil rouge » partout, sauf 18.17.
- **3.** 10 rubriques dans l'ordre ; 6 à 7 objectifs ; « L'essentiel » en 10 points ; tableau des pièges au format du §14 ; encadrés au format P5, chaque 🕰️ avec sa source.
  - Les ⏩ des titres de section correspondent exactement aux sections du parcours rapide du SYLLABUS.
  - En-tête « Temps total estimé », SYLLABUS (« Temps d'étude ») et calcul de `syllabus.py` (fiche à 125 mots/min) concordent pour les 9 chapitres, de même que le temps du parcours rapide du « Guide de lecture ».
- **6.** 🧠 10-12, 🔁 3, ✏️/∂ 4-8, 🔬 ≥ 1, 🔮 2, 🐛 ≥ 1, 📈 1, 🗣️ 1, 🏆 1, 💼 4-5, 🛠️ 1 partout ; 🧮 et 📄 présents dans les ch. 16 à 20. Les 🔨/📦 au-delà de 10 (ch. 12 : 13 ; ch. 13 : 14 ; ch. 15 : 13) sont justifiés dans les `notes` (S 19).
- **7.** L'indice 3 a été relu à la main pour 78 exercices : 45 à réponse courte (Q4, Q9, R2, x.2 et x.5 de chaque chapitre), les 18 🔮 et 15 🔨. Sur les 437, deux métriques par script :
  - réponses de 05 retrouvées dans l'indice 3 au-delà du premier item : 1 cas (13.3 c, constat 4) ;
  - part des lignes de la référence `mylearn_ref` présentes mot pour mot dans les indices : de 0 à 24 % ; aucun indice ne recopie la référence.
  Les 🔮 ne révèlent aucune issue.
- **9.** 15 à 30 cartes, en-tête `front;back;tags`, pas de doublon ni de série « Qu'est-ce que ». Les tags des parties III et IV sont tous en `_` (S 16 de l'audit précédent, non tranché).

### Checkpoints et mini-projets (§17)

| | Durée | Types (§17, CP3-CP6) | Parties antérieures | Barème sur 20 | Synthèse | Notebooks | Mini-projet |
|---|---|---|---|---|---|---|---|
| CP3 + MP3 | 111 min ✅ | 10 types ; 📄 et ⚖️ absents, justifiés dans les `notes` ✅ | CP3.11, 2 pts (10 %) ✅ | sujet = corrigé = contrat question par question ; sous-barèmes entre crochets qui totalisent les points de chaque question ✅ | Mermaid + fiche d'une page (22 formules) ✅ | sujet sans sortie ; solutions exécutées en 4,0 s ✅ | `mp3_revenus.ipynb` (départ et solution) ; `requires` (3) dans « Ta librairie » ; grille 4+3+3+3+3+2+2 ✅ |
| CP4 + MP4 | 114 min ✅ | 11 types ; 📄 absent, justifié ✅ | CP4.12, 2 pts sur les ch. 6, 9, 12 ✅ | idem ✅ (CP4.6 : « chaque item vaut 0,25 point » dans le sujet) | Mermaid + fiche d'une page (24 formules) ✅ | sujet sans sortie ; solutions exécutées en 2,5 s ✅ | `mp4_mini_framework.ipynb` ; `requires` (16) ; grille 5+4+4+3+4 ✅ |

Étoiles des examens : conformes à la règle du §17 (★★ pour CP3.4, 3.5, 3.6, 3.7, 3.13 ; CP4.2, 4.5, 4.7, 4.9). Écarts de titres : constat 9.

## Constats

### MAJEUR

**1. MAJEUR (à trancher)** · Échafaudage décroissant (§12, point 8 ; §22 Q10) : les énoncés 🔨/📦 des parties III et IV ne sont pas plus courts que ceux des ch. 5 à 11, et les 📦 donnent encore la méthode.

- **Mesure** (`check7_scaffold.py`, `raw_check7.txt` ; mots de l'énoncé hors en-tête et encadré « mylearn ») :
  - moyennes par chapitre : ch. 12 : 229, 13 : 280, 14 : 288, 15 : 233, 16 : 198, 17 : 201, 18 : 261, 19 : 290, 20 : 261 ;
  - repère : de 194 à 268 aux ch. 5 à 11, la plage que la décision Q10 jugeait trop longue ;
  - maxima : 19.16 (487 mots), 12.26 (418), 13.14 (410), 13.24 (400), 20.17 (381).
- **Ce qui a changé** : chaque énoncé 🔨 a un bloc « **Contrat** » (45 sur 45) et ne cite presque plus de fonctions NumPy. Mais le contrat décrit souvent le comportement pas à pas, parce que les tests comparent pas à pas à un oracle, et des conseils de méthode restent dans l'énoncé. Trois exemples :
  - `chapitres/ch19_optimiseurs/03_notebook.ipynb`, cellule 58 (source : `tools/chapters/build_ch19.py:851-854`) : « un paramètre peut être un tableau 0-D, de forme `()` : `p[:] = ...` y échoue, et `g + wd * p` y rend un scalaire NumPy, qu'`np.asarray(...)` remet en tableau » et « un tampon est une **copie** (`g.copy()`), et le weight decay s'ajoute à une copie (`g = g + wd * p`, jamais `g += wd * p`) ». C'est de la méthode (comment l'écrire), pas du contrat (ce que vérifient les tests).
  - `chapitres/ch20_deep_learning/03_notebook.ipynb`, cellule 87 (`build_ch20.py:2139`), 📦 20.23 : « met le modèle en mode entraînement, puis, pour chaque mini-batch de `loader`, envoie `X` et `y` sur `device`, fait **une** passe avant, calcule la loss, puis `optimizer.zero_grad()`, `loss.backward()` et `optimizer.step()` ». La docstring de la cellule TODO 89 (`build_ch20.py:1625`) le répète. Pourtant, l'indice 3 (`04_indices.md:885`) retient justement ces lignes : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session). L'échelle d'aide est inversée : l'énoncé en dit plus que l'indice 3.
  - 📦 20.18 (cellule 69, `build_ch20.py:1500`) et 20.28 (cellule 110) décrivent eux aussi la construction étape par étape. Aucun 📦 des ch. 12 à 14 ni 16 à 20 n'a de bloc « Contrat » ; au ch. 15, les 13 📦 en ont un.
- **Problème.** La décision Q10 se justifiait par la longueur des énoncés et par la méthode qu'ils donnaient. La longueur n'a pas baissé. La méthode a quitté les 🔨, mais pas tous les 📦, et les docstrings des TODO la redonnent (20.23).
- **Correction (à décider, les énoncés sont publiés)** :
  - option A : écrire au §22 que le contrat d'un 🔨 comparé pas à pas à un oracle inclut l'ordre des opérations, si bien que sa longueur ne baisse pas. Garder ce qui est vérifié par un test (copie des tampons, gradients intacts, mise à jour en place) et passer dans l'indice 1 ou 2 les conseils de mise en œuvre (`np.asarray` pour le 0-D, `g.copy()`, `p[:]`) ;
  - option B, pour les 📦 de la partie IV : un bloc « **Contrat** » (signatures, ce qui est vérifié) au lieu de la recette. Pour 20.23 : « `train_one_epoch_23` met le modèle en mode entraînement, fait une passe avant par mini-batch et met à jour les poids ; elle rend la loss moyenne par exemple, en `float`. » Les trois appels passent dans l'indice 2, et la docstring perd sa deuxième phrase ;
  - dans les deux cas, ajouter à `check7_scaffold.py` (ou à un test) un seuil indicatif par partie, pour que la tendance se voie.

### MINEUR

**2. MINEUR** · Rubriques de `05_solutions.md` (§12 : « Les questions 💼 reçoivent une réponse modèle en 60 secondes et les relances possibles du recruteur » ; règle « Toujours valable » de `PROGRESS.md:45`).

- **💼 des ch. 17 à 20 (19 questions)** : « **Relances possibles** » est remplacé par « **Ce qu'on attend** » et une seule « **Variante** ». Exemple, `chapitres/ch17_activations/05_solutions.md:171` : « **Variante** : « Et tanh ? » — Elle est centrée en 0… ». Les ch. 0A à 16 ont tous « Relances possibles » (`raw_cmp_styles.txt` : relances = 4 ou 5 par chapitre jusqu'au ch. 16, 0 ensuite).
  - Occurrences : ch. 17 : 05:169-188 ; ch. 18 : 05:179-203 ; ch. 19 : 05:206-230 ; ch. 20 : 05:186-210.
  - Correction : ajouter à chaque 💼 une ligne « **Relances possibles** : « … » (…) · « … » (…) », avec deux ou trois relances du format des ch. 12 à 16. La « Variante » actuelle peut devenir la première relance, et « Ce qu'on attend » rester.
- **🗣️ 19.10 et 20.10** (`chapitres/ch19_optimiseurs/05_solutions.md:153-156`, `chapitres/ch20_deep_learning/05_solutions.md:155-158`) : pas d'« **Erreurs fréquentes** », alors que les sept autres 🗣️ en ont une (ex. `ch18_backprop/05_solutions.md:159`).
  - Correction : pour 19.10, « **Erreurs fréquentes** : dire que le momentum « trouve un meilleur gradient » (il change le pas, pas le gradient) ; oublier le risque de dépasser le fond ; glisser une formule. » Pour 20.10, de même avec ses pièges (confondre profond et large, promettre une explicabilité intacte).

**3. MINEUR** · Indices 3 qui donnent une fonction ou une méthode complète (§12 : « jamais la fonction complète » ; §22 Q1).

- `chapitres/ch17_activations/04_indices.md:586-589` (17.12) : « def leaky_relu_derivative(z, negative_slope=0.01): / z = np.array(z, dtype=float) / return np.where(z > 0, 1.0, float(negative_slope)) ». C'est une des huit fonctions testées de l'exercice, complète.
  - Correction : `return np.where(z > 0, ..., ...)   # strict: what is the slope at z = 0?`
- `chapitres/ch18_backprop/04_indices.md:953-961` (18.26) : la méthode `__mul__` complète, `_backward` compris (« self.grad += other.data * out.grad / other.grad += self.data * out.grad »). L'indice 2 (ligne 945) en donne déjà la forme.
  - Correction : remplacer le corps de `_backward` par `...  # the local derivative of a product, times out.grad, accumulated`.
- `chapitres/ch19_optimiseurs/04_indices.md:686-697` (19.16), `:713-723` (19.17), `:741-751` (19.18) : le `step` de Momentum, d'Adadelta et d'AdamW est entier, en code ou dans un commentaire. Exemples : « # then: b = mu * b + g, in place (from the second step on) » suivi de « p -= self.lr * state["momentum_buffer"] » ; « # p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps) ». Ce sont 8 à 9 lignes, au-delà des « deux ou trois lignes clés ».
  - Correction : garder le squelette (la boucle, la création paresseuse de `state`) et une seule ligne clé par optimiseur ; remplacer la mise à jour du paramètre par `p -= ...   # in place, with the buffer of the docstring formula`.

**4. MINEUR** · Indices 3 qui donnent la valeur d'un item autre que le premier (§12, §22 Q1).

- `chapitres/ch20_deep_learning/04_indices.md:263` (20.R2) : « Pour b), $\log(e^{1000} + e^{0}) \approx 1000$, donc la composante vaut environ $0 - 1000$ ». La réponse vérifiée de b) est −1 000 (`05_solutions.md`, 20.R2).
  - Correction : « Pour b), écris la composante $z_2 - \log(e^{z_1} + e^{z_2})$ et estime le logarithme quand un terme écrase l'autre. »
- `chapitres/ch13_classifieurs/04_indices.md:339` (13.3 c) : « ($G = 1 - 0{,}2^2 - 0{,}8^2 = 0{,}32$) ». Le Gini de l'enfant droit vaut, par coïncidence, la réponse de c), le gain 0,32 (`05_solutions.md:93`).
  - Correction : « ($G = 1 - 0{,}2^2 - 0{,}8^2$) ».

**5. MINEUR** · `04_indices.md` : les en-têtes des exercices papier n'ont pas l'icône de type, contrairement à la règle P5 (§22 du 2026-10-04 : « en-têtes « Ex N.k — Titre icône » dans 04 et 05 »).

- 63 en-têtes : ch. 14 (14.1 à 14.10), ch. 15 (15.1 à 15.8), ch. 16 (16.1 à 16.10), ch. 17 (17.1 à 17.10), ch. 19 (19.1 à 19.13), ch. 20 (20.1 à 20.12). Exemple, `chapitres/ch14_ensembles/04_indices.md:271` : « ### Ex 14.1 — Dépouiller un vote pondéré ».
- Les ch. 9 à 13 et 18 l'ont (« ### Ex 13.1 — Vote des k plus proches voisins à la main ✏️ »), comme les exercices du notebook dans les mêmes fichiers (« ### Ex 16.12 — … count_parameters 🔨 »).
- Correction : ajouter l'icône du contrat à ces 63 titres (✏️, ∂, 🧮, 🗣️, ⚖️, 📄, 🛠️), et un test dans `tests/infra/` qui l'exige pour tout `### Ex` de 04 et 05.

**6. MINEUR** · `chapitres/ch12_preparation/05_solutions.ipynb`, cellule 92 (source : `tools/chapters/build_ch12.py:1380`, `wb.record("12.20d", right_20.tolist(), decimals=4, …)`) · sortie « ⚠️ Ex 12.20d : un élément est proche d'une limite d'arrondi ; envisage un autre nombre de décimales. »

- **Problème.** Le §22 du 2026-10-04 (« plus aucun ⚠️ d'arrondi dans les corrigés ») et `CLAUDE.md` (« 4 décimales, davantage si `record` signale une limite d'arrondi ») ne sont pas tenus. Calcul (copie, cellules 0 à 92 exécutées) : la matrice contient −1,5582481191. À 4 décimales, la partie fractionnaire de 15 582,481 est à 0,019 de 0,5, sous le seuil de 0,02 de `_near_rounding_boundary` (`src/wb/checker.py:377-380`). À 5 décimales : 0,0016, encore sous le seuil. À 6 décimales : 0,136, au-dessus.
- **Correction.** `decimals=6` dans `build_ch12.py:1380`, reconstruire, réexécuter, puis `build_answers.py --accept-answer-changes` (seul le nombre de décimales change). La cellule de vérification utilise `computed=True` (`build_ch12.py:1142`), donc l'énoncé ne change pas.

**7. MINEUR** · Place du 🧮 dans `02_exercices.md` (§12 : « ✏️/∂ Papier-crayon → 🧮/🗣️/⚖️/📄 Réflexion »).

- Les ch. 13 et 16 à 20 rangent le 🧮 dans le papier-crayon. Exemple, `chapitres/ch16_feedforward/02_exercices.md:140` : « ## ✏️ ∂ 🧮 Papier-crayon », avec 16.8 à la ligne 272.
- Le ch. 15 (`02_exercices.md` : « ## 🧮 🗣️ ⚖️ 📄 Réflexion ») et les ch. 0B, 1, 2, 5, 6 et 10 suivent le §12. Même écart dans les `06_mes_reponses.md`.
- C'est plausible : le 🧮 a des réponses chiffrées vérifiées en partie 0, comme le papier. Mais c'est contraire au §12 et différent d'un chapitre à l'autre.
- Correction : soit une ligne au §22 (« 🧮 va avec le papier-crayon quand ses réponses sont vérifiées en partie 0 ; le §12 est mis à jour »), en alignant le ch. 15 ; soit déplacer 13.9, 16.8, 17.8, 18.9, 19.9 et 20.9 dans « Réflexion ». La première option ne touche à aucun ID.

**8. MINEUR** · « Pour aller plus loin » : plus de 5 ressources (§12, rubrique 10 : « 3–5 ressources »).

- `chapitres/ch13_classifieurs/01_fiche.md:419-426` : 6 puces (Cover et Hart ; Cortes et Vapnik ; Quinlan ; Domingos et Pazzani ; guides scikit-learn ; VanderPlas).
- `chapitres/ch14_ensembles/01_fiche.md:317-325` : 7 puces (Breiman ; Geurts et coll. ; Freund et Schapire ; Friedman ; Chen et Guestrin, puis Ke et coll. ; Hastie, Tibshirani et Friedman ; guide scikit-learn).
- L'audit précédent avait corrigé le même défaut au ch. 11 (constat 12, en regroupant deux puces).
- Correction : regrouper par thème, sans retirer de lien. Au ch. 13 : les deux articles fondateurs du kNN et des SVM dans une puce, Quinlan et Domingos-Pazzani dans une autre, VanderPlas avec le guide scikit-learn. Au ch. 14 : Breiman et Geurts ; Freund-Schapire et Friedman ; Chen-Guestrin et Ke ; ESL ; guide scikit-learn (5 puces).

**9. MINEUR** · Titres des sujets d'examen différents du contrat (§22 P5 : « avec le titre du contrat, et non un titre raccourci »).

- `checkpoints/partie_3/01_examen_sujet.md:39` : « ### CP3.1 — Prédire la sortie de six extraits scikit-learn 🔮 » ; le contrat et le tableau de bord ajoutent « (formes, scores, avertissements) ».
- `checkpoints/partie_3/01_examen_sujet.md:185` : « CP3.8 — Lire une courbe de validation et une courbe de boosting » ; le contrat dit « … de validation (max_depth) et … ».
- `checkpoints/partie_4/01_examen_sujet.md:163` : « CP4.9 — Écrire de mémoire une epoch d'entraînement avec AdamW et clipping, en NumPy » ; le contrat (`cp4.json`) et `suivi/tableau_de_bord.md` ajoutent « (notebook) ».
- Correction : aligner les deux sujets de CP3 sur le contrat (`build_cp3.py`, puis le sujet) ; pour CP4.9, retirer « (notebook) » du titre du contrat (l'information est déjà dans le corps du sujet), puis `syllabus.py build`. Les titres courts des corrigés suivent la convention de CP2 et ne sont pas comptés.

**10. MINEUR** · `suivi/auto_evaluation.md:229-243` · la section « <!-- wb:section 12 --> ## 12 — Préparation des données » précède « <!-- wb:section CP2 --> ## CP2 — Checkpoint de la partie II ».

- **Problème.** L'ordre du workbook (`tools/syllabus.py:32`, `ORDER`) place CP2 entre 11 et 12, et `suivi/tableau_de_bord.md:597-616` le respecte. Un apprenant qui crée sa copie avec `start_chapter.py --init` voit le bilan de la partie II après le ch. 12.
- **Correction.** Déplacer le bloc `<!-- wb:section CP2 --> … <!-- wb:end CP2 -->` avant `<!-- wb:section 12 -->`. Ajouter à `tests/infra/` un test : l'ordre des sections `wb:section` du modèle suit `ORDER`.

**11. MINEUR** · `suivi/PROGRESS.md:142` · « | 12 | Préparation des données | ✅ | 25, 26 | 56 exercices (dont 23 dans le notebook, parties A à D) ».

- **Problème.** `docs/syllabus/data/ch12.json` contient 52 exercices, dont 23 dans le notebook (`syllabus.py` et `check1_exercises.py`) : 11 quiz, 3 rappels, 12.1 à 12.10, 5 questions d'entretien et 12.11 à 12.33.
- **Correction.** « 52 exercices (dont 23 dans le notebook, parties A à D) ».

**12. MINEUR** · `data/cards/penguins.md`, section « Chapitres » (ligne 40 et suivantes) : elle cite CP1 et CP2, mais ni CP3 ni CP4.

- `cp3.json` donne le fil Penguins à CP3.2 et CP3.4, et `cp4.json` à CP4.12 (« standardisation sans fuite du ch. 12, sur Penguins », `study_notes`). C'est le défaut du constat 10 de l'audit précédent, corrigé pour CP1 et CP2.
- Correction : ajouter après « 15 (…) » : « CP3 (CP3.2 : standardiser et encoder un mini-tableau ; CP3.4 : kNN avant et après standardisation), CP4 (CP4.12 : standardisation sans fuite) ».

**13. MINEUR** · `annexes/formulaire.md:462` · « ### Ch. 15 » : le seul titre de chapitre du formulaire sans son nom (ligne 446 : « ### Ch. 14 · Ensembles » ; ligne 478 : « ### Ch. 16 · Réseaux feed-forward »).

- Correction : « ### Ch. 15 · scikit-learn ».

**14. MINEUR** · `annexes/glossaire.md:24-26` · « | pooling | agrégation, sous-échantillonnage | … | | », de même pour « padding » et « stride », avec une colonne « Ch. » vide.

- **Problème.** Le ch. 20 les introduit : fiche §20.4.5 à §20.4.7, « zero-padding » (`01_fiche.md:58`), le pas d'une fenêtre de pooling dans 🔮 20.14. Le §19 demande un glossaire à jour, et la colonne « Ch. » donne le chapitre d'introduction. La ligne 20, « framework », est elle aussi vide, alors que le mot est employé dès les ch. 15 et 16.
- **Correction.** « 20 » dans la colonne « Ch. » de pooling, padding et stride. Pour framework, le premier chapitre qui l'emploie (`grep -n framework chapitres/*/01_fiche.md`). « kernel » (ligne 27) reste vide jusqu'au ch. 21.

**15. MINEUR** · Colonne 🚀 des tableaux d'exercices des notebooks (§12, 03_notebook point 1 : « tableau des exercices (ID, type, ★, ⏱️, 🚀) »).

- `tools/chapters/build_ch12.py:2674` à `build_ch20.py:2715` copient tous `rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", …`, sans colonne 🚀.
- La suite de l'audit précédent avait noté : « un `exercise_table` commun au kit dès la partie III » (`docs/AUDIT_parties_I_II.md:544`). Il n'existe pas : `grep exercise_table tools/` ne trouve rien, et le tableau est recopié dans 9 scripts de plus.
- Aucun exercice des parties III et IV n'a `gpu: true`, mais la partie V (CNN, CIFAR-10) en aura.
- Correction : `exercise_table(chapter, parts)` dans `tools/chapters/chapter_kit.py`, qui lit le contrat et ajoute la colonne 🚀 quand un exercice l'a, avant le ch. 21 ; les scripts publiés n'ont pas à changer.

### SUGGESTION

**16. SUGGESTION** · Valeurs enregistrées à une limite d'arrondi, acceptées sous les deux arrondis (ℹ️ dans les corrigés) : 17.6 b, c et f, 17.8 f, 18.19 d, 20.7 h.

- Exemple : `wb.record("20.7h", 1.01 ** 30, decimals=4, …)`. $1{,}01^{30} = 1{,}347849$ : 1,3478 et 1,3479 passent tous deux, alors que 1,3479 est un arrondi faux. Pour 18.19 d, une valeur calculée par la fonction de l'apprenant (`computed=True`) tombe à une limite.
- Ce n'est pas bloquant : les deux arrondis sont acceptés.
- Correction proposée à la prochaine reconstruction de ces chapitres : un nombre de décimales sans limite pour les valeurs `computed=True` (18.19 d), et une ligne au §22 qui dit que les ℹ️ restent permis pour une réponse papier exacte (17.6) ou dont l'énoncé fixe les décimales.

**17. SUGGESTION** · Rubriques des quiz et des rappels : aux ch. 12 à 20, chaque 🧠 et chaque 🔁 de `05_solutions.md` a une ligne « **À retenir** » et ni « Erreurs fréquentes » ni « Variante » (131 items : 104 quiz et 27 rappels, `raw_cmp_styles.txt`).

- Exemple : `chapitres/ch16_feedforward/05_solutions.md:13`.
- C'est homogène et plus complet que les ch. 1 à 11 (quelques « Erreurs fréquentes », rarement « À retenir »). Mais la règle « Toujours valable » (`suivi/PROGRESS.md:45`) et le §12 réservent « À retenir » aux items courts de la partie 0.
- Correction : une ligne au §22, « 🧠 et 🔁 : la réponse justifiée par item et une ligne « À retenir » ; pas d'erreurs fréquentes ni de variante » ; puis aligner, à l'occasion, les ch. 1 à 11 sur cette forme.

**18. SUGGESTION** · `chapitres/ch17_activations/02_exercices.md:284` (17.9) · « **Parcours :** – ».

- C'est le seul exercice des parties III et IV, hors quiz, qui n'est dans aucun parcours. Les quiz dans le même cas (16.Q5 à Q7, 17.Q2…) n'affichent rien.
- Correction : « **Parcours :** complet seulement », dans le script et dans `02_exercices.md`, pour que l'apprenant comprenne le tiret.

**19. SUGGESTION** · `notes` des contrats dont la première phrase est périmée.

- `docs/syllabus/data/ch12.json` (et `docs/SYLLABUS.md:2254`) : « Composition : 🔨/📦 = 11 (> 10) … 6 📦 = pandas, … ». La suite des mêmes notes dit « 🔨/📦 passe à 13 ».
- `ch15.json` : « Composition : 📦 = 12 », puis « 📦 passe à 13 ».
- `ch13.json` : « 🔨/📦 = 14 (maximum autorisé pour un chapitre dense) ». Aucun maximum de ce genre n'est écrit dans la bible : le §11 dit 6 à 10, avec un écart justifié.
- Correction : réécrire la première phrase (« Composition : 🔨/📦 = 13 (> 10) : … 8 📦 = … ») ; au ch. 13, « 🔨/📦 = 14 (> 10, justifié : quatre modules à écrire…) ».

**20. SUGGESTION** · `annexes/cheatsheets/sklearn.md:95-106` : XGBoost et LightGBM n'apparaissent dans aucun aide-mémoire, alors que 📦 14.26, 🏆 14.28 et MP3 s'en servent. Leurs pièges sont propres à chacun : `early_stopping_rounds` dans le constructeur de XGBoost 3, `best_iteration` compté à partir de 0 dans XGBoost et `best_iteration_` à partir de 1 dans LightGBM, `callbacks=[lightgbm.early_stopping(20)]`.

- Correction : deux lignes après `HistGradientBoostingRegressor`, avec un renvoi à 14.26.

**21. SUGGESTION** · Libellé du fil rouge dans les en-têtes : « **Fil rouge :** données synthétiques » en 13.1, 13.4, 14.2, 14.6 et 15.3, contre « synth » partout ailleurs (contrat compris). 18.17 n'a pas de ligne « Fil rouge » (contrat : `synth`).

- Correction : « **Fil rouge :** synth » ; ajouter la ligne à 18.17 (cellule 62 de `03_notebook.ipynb`, via `build_ch18.py`).

## Bilan

**Ce qui est solide**

- **Le contrat est tenu partout** : les 437 exercices concordent dans les huit supports (contrat, 02, 03, 05.ipynb, tableau de bord, 04, 05, 06), et `syllabus.py check`, `build_answers.py --check` et `export_flashcards.py --check` sont verts.
- **Les notebooks sont reproductibles** : les 13 scripts redonnent exactement les 26 notebooks publiés. Les notebooks de solutions sont exécutés, propres et largement sous le budget du §4.
- **Les fiches sont conformes** : 10 rubriques, ⏩ égaux au SYLLABUS, en-têtes de temps cohérents avec le calcul à 125 mots/min.
- **Les indices 3 respectent la règle Q1** sur l'essentiel de l'échantillon : premier item résolu, autres items posés, aucun 🔮 révélé, aucun 🔨 qui recopie la référence.
- **Les checkpoints suivent le §17** : durée, types, 10 %, barèmes qui totalisent 20 question par question, Mermaid, `mpN_<sujet>.ipynb`, `requires`.

**Les 3 risques principaux**

1. **L'échafaudage ne décroît pas** (constat 1). Les énoncés sont devenus des « contrats », mais aussi longs qu'avant, et en partie IV les 📦 et leurs docstrings donnent encore la recette. Il faut trancher avant la partie V, où PyTorch rend les 📦 nombreux.
2. **Le style dérive en fin de partie IV** (constats 2, 3 et 5). Les rubriques 💼 et 🗣️ s'appauvrissent aux ch. 17 à 20, les indices de code s'allongent au ch. 19, et les icônes de 04 disparaissent dans six chapitres. Aucun test ne voit ces formats : un test d'en-têtes 04/05 et de rubriques 05 (« Relances possibles » pour 💼, « Erreurs fréquentes » pour 🗣️) les fixerait.
3. **Des promesses de l'audit précédent sont restées sans suite** (constats 6, 15 et 17) : un ⚠️ d'arrondi revenu dans un corrigé, le kit `exercise_table` jamais écrit, des conventions de rubriques appliquées sans être écrites. Le §22 et les tests doivent suivre les pratiques réelles, sinon chaque nouveau chapitre recopie l'écart.

**Sorties brutes** (dans `/tmp/claude-0/-home-claude/d0ad1154-96ff-53e5-aebd-0edf604016be/scratchpad/s43/A03_structure/`) : `raw_check1.txt`, `raw_check1b.txt`, `raw_check1c.txt`, `raw_check1d_headers.txt`, `raw_cmp_styles.txt`, `raw_check2.txt`, `raw_check4.txt`, `raw_check5.txt`, `raw_check6.txt`, `raw_check6_hints.json`, `raw_check6b.txt`, `raw_check7.txt` et `raw_check7.json`, `raw_check9.txt`, `raw_check9b.txt`, `raw_check10_stars.txt`, `raw_rebuild.txt`, `raw_pytest_infra.txt`, `sample_short.txt` et `sample_predict.txt` (échantillon relu à la main).
