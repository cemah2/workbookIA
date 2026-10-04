# 1 · Introduction au machine learning et au deep learning — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses des exercices ✏️ se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ Papier-crayon](#papier) · [🧮 🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à D](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 1.Q1 — Règles écrites ou règles apprises ?
1. **É** : le taux de TVA est une règle écrite. 2. **A** : le filtre apprend de tes exemples (tes e-mails marqués). 3. **É** : un seuil fixé à l'avance. 4. **A** : personne ne lui a décrit les visages de tes proches ; il a appris à reconnaître les visages et regroupe ceux qui se ressemblent. 5. **É** : une liste écrite (le dictionnaire).

**À retenir** : ce qui distingue le ML n'est pas la complexité du programme, mais l'**origine des règles** : écrites par un humain, ou tirées d'exemples.

### 1.Q2 — Pourquoi les systèmes experts ont calé
1. **Faux** : un système expert n'apprend rien ; ses règles sont écrites par des humains, d'après ce que disent les experts. 2. **Vrai**. 3. **Faux** : après le 7 barré viendront le 7 penché, le 7 à crochet, le 7 griffonné… chaque cas imprévu demande une règle. 4. **Faux** : beaucoup de savoir-faire est intuitif (reconnaître un visage, juger une radio) et ne s'écrit pas en règles. 5. **Vrai**.

### 1.Q3 — Échantillon, feature ou label ?
1. Un **appartement** (une ligne du tableau). 2. La surface, le nombre de pièces, l'étage et le quartier. 3. Le **loyer**. 4. Des loyers réels, observés (les annonces passées) : ici, l'« expert » qui fournit les labels, c'est le marché. 5. Le quartier devient le label et le loyer une feature ; on passe d'une **régression** (prédire une quantité) à une **classification** (choisir un quartier dans une liste).

**Erreurs fréquentes** : croire qu'une colonne est « par nature » une feature ou un label. C'est la question posée qui décide (1.9 f).

### 1.Q4 — L'école absurde : ce qui marche pour une machine
1. Le premier test mesure la **mémoire** (les faits récités) ; le second, la **compréhension**, c'est-à-dire la capacité à répondre à des questions nouvelles.
2. Un enfant s'ennuie et ne comprend rien à force de répétitions ; un ordinateur ne s'ennuie pas, et chaque nouveau passage sur les mêmes exemples lui permet d'ajuster un peu ses paramètres.
3. **Faux** : il a peut-être seulement appris par cœur (tu le verras en 1.14).
4. Premier test : le score sur les **données d'entraînement** ; second test : le score sur le **jeu de test** (la généralisation).

### 1.Q5 — Paramètre ou hyperparamètre ?
**Paramètres** (appris) : 1 (les poids), 4 (le seuil trouvé par l'arbre), 6 (la pente $w$). **Hyperparamètres** (choisis) : 2 (le learning rate), 3 (le nombre d'epochs), 5 (`max_depth`).
**Moyen mnémotechnique** : un hyperparamètre se règle **avant** `fit` ; un paramètre sort **de** `fit`.

### 1.Q6 — À quoi sert le jeu de test
1. Pour mesurer la généralisation sur des données que le modèle n'a **jamais vues** : s'il avait servi à l'entraînement, le score serait trop optimiste.
2. **Non** : pendant l'évaluation, rien n'est appris, quelle que soit la qualité des prédictions.
3. Le modèle a appris des détails propres à ses exemples plutôt qu'une règle générale : il généralise mal : c'est l'overfitting (*surapprentissage*) du ch. 9. Il faut un modèle plus simple, plus de données, ou d'autres features.
4. Un passage complet sur tout le jeu d'entraînement.
5. **Faux** : en choisissant le learning rate d'après le test, on « apprend » un peu du test, et le score annoncé devient optimiste. On règle les hyperparamètres sur un **jeu de validation** distinct, et l'on ne regarde le test qu'une fois, à la fin (ch. 8).

### 1.Q7 — Classification ou régression : six situations
1. **R** (un prix). 2. **C** (spam ou non). 3. **R** (une température). 4. **C** (une espèce dans une liste). 5. **R** (un âge en années se mesure). 6. **C** : un code postal s'écrit avec des chiffres, mais c'est une **catégorie** ; la « moyenne » de deux codes postaux n'a aucun sens.

**À retenir** : la question n'est pas « la réponse est-elle un nombre ? » mais « la réponse est-elle une **quantité** ? ».

### 1.Q8 — Clustering, débruitage ou réduction de dimension ?
1. **Clustering**. 2. **Débruitage**. 3. **Réduction de dimension**. 4. **Débruitage** : le livre traite les valeurs manquantes comme une forme de bruit (§1.4.2 ; on pourrait aussi les reconstituer par régression). 5. **Réduction de dimension** : une feature constante n'apporte aucune information (§1.4.3).
6. Aucune ne dispose de labels : personne ne dit à l'algorithme le « bon groupe », le « bon son » ou les « bonnes features ». Il n'a que les données.

### 1.Q9 — Générateurs et renforcement : sans labels, mais pas sans retour
1. De **nouvelles données**, qui ressemblent aux exemples sans les recopier (des images, des textes, des sons).
2. Il n'a pas de labels (comme le non supervisé), mais reçoit un retour sur la qualité de ses productions (comme le supervisé). Aujourd'hui, on dirait plutôt que la plupart des générateurs sont entraînés de façon **auto-supervisée** : les données fournissent elles-mêmes la réponse attendue (prédire la suite d'un texte, retrouver une image à partir d'une version bruitée). « Semi-supervisé » désigne en général (et c'était déjà le cas en 2018) l'apprentissage avec peu de labels et beaucoup de données non étiquetées.
3. **Agent** : le thermostat. **Environnement** : la maison et ses habitants (et la météo). **Action** : le réglage du chauffage. **Récompense** : tes réactions (négative si tu corriges la consigne, positive si tu n'y touches pas), éventuellement moins la consommation d'énergie.
4. Un label dit **quelle était la bonne réponse** ; une récompense dit seulement **si l'action choisie était bonne**, sans dire ce qu'il aurait fallu faire.
5. Pour **explorer** : l'action qui paraît la meilleure ne l'est peut-être que par hasard, et une autre, jamais ou trop peu essayée, pourrait être meilleure (1.22).

### 1.Q10 — Profond, capacité et GPU
1. Le nombre de **couches** empilées les unes sur les autres.
2. Il fait une somme pondérée des nombres qu'il reçoit (plus un biais), puis applique une fonction à ce total, et transmet le résultat (ch. 10).
3. Le fait que le réseau **apprend lui-même** ses features (traits, formes, motifs), au lieu qu'on les fabrique à la main (§1.1.2, §1.7).
4. **Faux** : plus de capacité permet d'apprendre plus… y compris le bruit et les détails des exemples ; le score sur le test peut baisser (le mémoriseur de 1.14 a une capacité énorme et généralise très mal ; ch. 9).
5. L'entraînement répète des milliards de multiplications et d'additions indépendantes (des produits de matrices, 0B) ; un GPU en fait des milliers **en parallèle**.

### 1.Q11 — Panorama 2026 : où ranger ChatGPT et Stable Diffusion ?
1. **Auto-supervisé** : la « bonne réponse » (le token suivant) vient du texte lui-même, sans étiquetage humain. (Viennent ensuite un ajustement supervisé sur des exemples de dialogues et un alignement par renforcement ou par des méthodes voisines, RLHF puis DPO.)
2. Un **générateur** (§1.5), construit par **deep learning** (§1.7) : un modèle de diffusion.
3. Un très grand modèle entraîné une fois sur des données très variées, puis **réutilisé** pour de nombreuses tâches (par prompting, RAG ou fine-tuning léger).
4. Un morceau de texte (un mot, un bout de mot, un signe de ponctuation) : l'unité que le modèle lit et prédit (bonus B2).
5. **Faux** : un LLM s'entraîne sur des échantillons (des textes), minimise une loss, avec un learning rate, et l'on mesure sa généralisation sur des données nouvelles. C'est le même socle, à une autre échelle.

<a id="rappels"></a>

## 🔁 Rappels

### 1.R1 — 0B : distance entre deux manchots vus comme des vecteurs
1. $\mathbf{p} - \mathbf{a} = (-0{,}4 ;\ 1{,}3 ;\ -5 ;\ -50)$, donc $\lVert \mathbf{p} - \mathbf{a} \rVert = \sqrt{0{,}16 + 1{,}69 + 25 + 2\,500} = \sqrt{2\,526{,}85} \approx$ **50,3**. $\mathbf{p} - \mathbf{c} = (-13{,}6 ;\ -1{,}1 ;\ -16 ;\ 25)$, donc $\lVert \mathbf{p} - \mathbf{c} \rVert = \sqrt{184{,}96 + 1{,}21 + 256 + 625} = \sqrt{1\,067{,}17} \approx$ **32,7**.
2. Le **Chinstrap** $\mathbf{c}$ est le plus proche de l'Adélie $\mathbf{p}$ ! Surprenant : $\mathbf{p}$ et $\mathbf{a}$ ont des becs et des nageoires presque identiques, mais 50 g d'écart de masse pèsent plus que 13,6 mm de bec et 16 mm de nageoire.
3. Sans la masse : $\sqrt{26{,}85} \approx$ **5,2** pour $\mathbf{a}$, et $\sqrt{442{,}17} \approx$ **21,0** pour $\mathbf{c}$ : cette fois, l'Adélie $\mathbf{a}$ est de loin le plus proche.
4. Une distance additionne des carrés : la feature qui a les plus grands nombres (la masse, en grammes) écrase les autres. Un algorithme de regroupement sur les mesures brutes regroupera surtout par masse (1.21). Il faut mettre les features à la même échelle (1.21, ch. 12).

### 1.R2 — 0A : compter les espèces avec `value_counts`
1. `df["species"].value_counts()` (trié du plus fréquent au moins fréquent).
2. Des **proportions** au lieu de comptes (la somme vaut 1) : `df["species"].value_counts(normalize=True)`.
3. Avec `value_counts` sur deux colonnes : `df[["island", "species"]].value_counts()`, ou `df.groupby("island")["species"].value_counts()` ; sous forme de tableau : `pd.crosstab(df["island"], df["species"])`, ou `df.groupby(["island", "species"]).size()` (0A.57). On y voit que les Gentoo ne viennent que de Biscoe et les Chinstrap que de Dream (data card, 1.13).
4. Les classes n'ont pas le même effectif : les Chinstrap sont environ deux fois moins nombreux que chacune des deux autres espèces. Un classifieur « paresseux » qui répond toujours l'espèce la plus fréquente obtient déjà environ 44 % d'accuracy : c'est la **référence** à battre, et l'accuracy seule peut tromper quand une classe est rare (ch. 3).

### 1.R3 — 0B : la droite qui passe par deux points
1. Pente $w = \frac{13 - 5}{6 - 2} = 2$ ; puis $5 = 2 \times 2 + b$ donne $b = 1$ : la droite $y = 2x + 1$.
2. $2 \times 10 + 1 = $ **21**.
3. **Oui** : $2 \times 4 + 1 = 9$.
4. `def line(x): return 2 * x + 1` (ou `lambda x: 2 * x + 1`).

**Lien avec le chapitre** : une régression linéaire cherche une telle droite, mais passant **au plus près** de nombreux points, qui ne sont jamais tous alignés (1.2, 1.16, ch. 9).

<a id="papier"></a>

## ✏️ Papier-crayon

### Ex 1.1 — Accuracy et erreurs à l'échelle d'un centre de tri ✏️
a) $\frac{9\,905}{10\,000} = $ **0,9905** · b) $1 - 0{,}9905 = 0{,}0095$, soit **0,95 %** · c) $1\,200\,000 \times 0{,}0095 = $ **11 400** chiffres mal lus par jour · d) $0{,}9905^5 \approx 0{,}953394$, soit **0,9534** · e) $240\,000 \times (1 - 0{,}953394) \approx 11\,185{,}4$, soit **11 185** codes faux par jour.
**Pourquoi** : en d, un code est juste si ses 5 chiffres le sont **tous** ; avec l'indépendance, on multiplie les 5 probabilités. En e, un code faux est l'événement contraire : $1 - 0{,}9905^5 \approx 4{,}7\,\%$.
**Erreurs fréquentes** : en d, l'approximation $1 - 5 \times 0{,}0095 = 0{,}9525$ (on additionne les risques ; c'est proche, mais pas exact, car elle compte deux fois les codes avec deux erreurs) ; en e, arrondir d avant de multiplier (à 0,9534 on trouve 11 184, à 0,953 on trouve 11 280 : un arrondi à peine visible, multiplié par 240 000) ; ou compter les chiffres faux ($240\,000 \times 5 \times 0{,}0095 = 11\,400$) au lieu des codes faux.
**À retenir** : 99 % d'accuracy « par chiffre » devient environ 95 % « par code », et plus de 11 000 lettres mal routées chaque jour. Une petite erreur, répétée à grande échelle, devient un vrai problème : c'est pourquoi les systèmes réels combinent plusieurs contrôles (le nom de la ville doit correspondre au code postal, par exemple).
**Variante** : quelle accuracy par chiffre faudrait-il pour que 99 % des codes postaux soient lus sans aucune erreur ? (Il faut $p^5 = 0{,}99$, soit $p = 0{,}99^{1/5} \approx 0{,}998$, la racine cinquième de 0,99 (`0.99 ** (1 / 5)` en Python) : environ 2 chiffres faux pour 1 000, près de cinq fois moins que les 9,5 pour 1 000 du réseau du livre.)

### Ex 1.2 — Concerts : la valeur manquante et celle de demain ✏️
a) $\frac{1\,290 + 1\,390}{2} = $ **1 340** · b) $\frac{1\,550 - 1\,200}{12 - 5} = \frac{350}{7} = $ **50** spectateurs par jour · c) $1\,200 + 50 \times (8 - 5) = $ **1 350** · d) $1\,550 + 50 = $ **1 600** · e) pente des deux derniers jours $1\,550 - 1\,520 = 30$, donc **1 580** · f) $1\,600 \times 25 \times 0{,}10 = $ **4 000 €**.
**Pourquoi** : interpoler au milieu de deux jours, c'est prendre la moyenne de leurs deux valeurs. Une droite monte chaque jour de la même quantité, sa pente : sa valeur un jour donné est celle d'un point connu, plus la pente multipliée par le nombre de jours d'écart.
**Laquelle choisir ?** Aucune n'est « la vraie » : la droite du premier au dernier jour résume la tendance de toute la semaine, celle des deux derniers jours suit la toute dernière évolution (elle ralentit). Deux méthodes raisonnables donnent 20 spectateurs d'écart, soit 50 € pour le groupe. Une bonne pratique : prévoir avec la tendance d'ensemble et **donner une fourchette** plutôt qu'un nombre unique. La régression linéaire du ch. 9 trace la droite la plus proche de **tous** les points, pas seulement de deux.
**Erreurs fréquentes** : diviser par 8 (le nombre de jours de 5 à 12 inclus) au lieu de 7 (le nombre d'intervalles) ; oublier que le 8 est à trois jours du 5.
**Variante** : prévois le 13 mai avec la tendance des **trois** derniers jours, du 10 au 12 mai. (Pente $\frac{1\,550 - 1\,460}{12 - 10} = 45$, donc $1\,550 + 45 = 1\,595$ : encore une autre prévision, entre celles de d et de e.)

### Ex 1.3 — Compter les connexions d'un réseau en couches ✏️
a) $4 \times 3 + 3 \times 2 = $ **18** · b) $18 + 3 + 2 = $ **23** · c) $784 \times 128 + 128 + 128 \times 10 + 10 = 100\,352 + 128 + 1\,280 + 10 = $ **101 770** · d) $784 \times 256 + 256 + 256 \times 10 + 10 = $ **203 530** · e) $101\,770 + 128 \times 128 + 128 = $ **118 282**.
**Pourquoi** : dans une couche pleine, chacun des $m$ neurones reçoit une connexion de **chacune** des $n$ valeurs de la couche précédente, d'où $n \times m$ poids ; s'y ajoute un biais par neurone qui calcule (couches cachées et sortie), jamais pour les entrées.
**Doubler la largeur ou ajouter une couche ?** Doubler la couche cachée ajoute environ 101 760 paramètres (presque le double) ; ajouter une seconde couche de 128 n'en ajoute que 16 512. La première couche domine, car elle est connectée aux 784 pixels. C'est une raison pour laquelle les réseaux profonds préfèrent souvent **plusieurs couches modestes** à une seule couche énorme ; et les réseaux convolutifs (ch. 21) réduisent encore ce coût en partageant les poids.
**Erreurs fréquentes** : additionner les tailles des couches au lieu de les multiplier (12 en a) ; donner un biais aux entrées (27 en b) ; oublier les biais (101 632 en c).
**À retenir** : une couche pleine de $n$ entrées et $m$ neurones a $n \times m + m$ paramètres ; à cause de ce produit, la couche branchée sur les 784 pixels porte à elle seule près de 99 % des paramètres du réseau de c.
**Lien** : scikit-learn compte exactement 101 770 paramètres pour le réseau de 1.23.
**Variante** : combien de couches cachées de 128 neurones faut-il empiler (784 → 128 → … → 128 → 10) pour dépasser les 203 530 paramètres du réseau de d ? (Chaque couche de 128 ajoutée coûte $128 \times 128 + 128 = 16\,512$ paramètres : avec 7 couches cachées, on en compte $101\,770 + 6 \times 16\,512 = 200\,842$, avec 8, $217\,354$. Il en faut donc 8 pour dépasser le réseau à une seule couche de 256.)

### Ex 1.4 — Moins de nombres pour dire la même chose ✏️
a) **2** : la pluie, toujours nulle, ne distingue aucun jour d'un autre · b) $68 \times 2{,}2046 \approx$ **149,9** lb · c) **1** : l'une se calcule à partir de l'autre · d) $\sqrt{240^2 + 320^2} = \sqrt{160\,000} = $ **400** m · e) le vecteur unitaire de la route est $(0{,}6 ;\ 0{,}8)$, donc $750 \times (0{,}6 ;\ 0{,}8) = $ **[450, 600]** · f) $250 \times 0{,}6 + 300 \times 0{,}8 = $ **390** m · g) $\lVert Q \rVert^2 = 250^2 + 300^2 = 152\,500$, et $152\,500 - 390^2 = 400$, donc la distance à l'axe vaut **20** m.
**Pourquoi** : d et e montrent une réduction **sans perte** (sur une route à une voie, un nombre suffit, et l'on retrouve les deux coordonnées) ; f et g une réduction **avec perte** : garder seulement 390 m fait oublier que la voiture roule à 20 m de l'axe, peut-être sur l'autre voie. C'est le compromis de §1.4.3 : moins de nombres, un peu moins de précision.
**Erreurs fréquentes** : en d, additionner les coordonnées (560) ; en f, donner la distance de $Q$ au départ à vol d'oiseau (≈ 390,5 m, arrondie à 391) au lieu de sa position le long de la route.
**Variante** : la réduction de dimension automatique (l'analyse en composantes principales, ch. 12) trouve seule l'axe « le long duquel » les données varient le plus, exactement comme l'axe de la route ici.

<a id="reflexion"></a>

## 🧮 🗣️ ⚖️ 📄 Réflexion

### Ex 1.5 — Fermi : combien coûte l'étiquetage de MNIST ? 🧮
Une estimation possible (d'autres hypothèses raisonnables sont aussi justes) :
1. **2 secondes** par chiffre (regarder, taper une touche) : $70\,000 \times 2 = 140\,000$ s, soit environ **40 heures** pour un passage.
2. Deux passages indépendants, plus un arbitrage pour 2 % des images (1 400 images, disons 5 s chacune, soit 2 h) : environ **80 heures**.
3. À 25 € de l'heure, charges comprises : environ **2 000 €** ; sur une plateforme de micro-travail, beaucoup moins (et c'est un vrai sujet éthique : la rémunération des annotateurs).
4. ImageNet : $\frac{14 \times 10^6}{7 \times 10^4} = $ **200 fois** plus d'images. Et chaque image est plus longue à étiqueter : il faut choisir parmi des milliers de catégories, souvent proches (des dizaines de races de chiens), au lieu de 10 chiffres. Si une image prend 10 fois plus de temps, on arrive à environ $80 \times 200 \times 10 \approx 160\,000$ heures, soit une centaine de personnes à plein temps pendant un an : c'est pourquoi ImageNet a été étiqueté par de très nombreux travailleurs en ligne, sur Amazon Mechanical Turk.
5. Parce que l'étiquetage humain ne passe pas à l'échelle : à une minute par texte, un milliard de textes représentent environ 17 millions d'heures, soit plus de 10 000 années de travail d'une personne à plein temps. L'apprentissage auto-supervisé tire la « réponse » des données elles-mêmes (le mot suivant d'un texte), gratuitement et en quantité illimitée. Les humains n'interviennent plus qu'à la fin, sur quelques centaines de milliers d'exemples de dialogues et de préférences.

**Pourquoi** : le coût d'étiquetage est un produit (nombre d'images × temps par image × nombre de passages × coût horaire) : il suffit qu'un facteur grandisse, 200 fois plus d'images ou 10 fois plus de temps par image, pour que le total explose. L'auto-supervisé supprime ce produit, en tirant la réponse des données elles-mêmes.
**Critères** : les hypothèses sont écrites et raisonnables ; les ordres de grandeur sont justes à un facteur 10 près ; le rapport 200 est trouvé ; la réponse 5 cite l'absence de labels humains.
**Erreurs fréquentes** : diviser les secondes par 60 au lieu de 3 600 pour obtenir des heures ; oublier que chaque image est étiquetée **deux** fois ; se tromper de puissance de 10 en 4 ($\frac{14 \times 10^6}{7 \times 10^4} = 2 \times 10^2$, ni 20 ni 2 000).
**Variante** : refais l'estimation pour CIFAR-10 (60 000 petites photos en couleur, 10 classes), avec 3 s par image et le même protocole : deux passages, et 2 % d'arbitrages de 5 s. ($60\,000 \times 3 \times 2 = 360\,000$ s, soit 100 h, plus $1\,200 \times 5 = 6\,000$ s, moins de 2 h : une centaine d'heures, environ 2 500 € à 25 € de l'heure.)

### Ex 1.6 — Le machine learning en cinq lignes 🗣️
« D'habitude, pour qu'un ordinateur fasse une tâche, on lui écrit toutes les règles. Le machine learning, c'est lui montrer des **exemples** à la place : des milliers de photos de chats et de chiens, chacune avec la bonne réponse. Au début, il répond au hasard ; à chaque **erreur**, il ajuste un peu ses réglages pour se tromper moins. Après des milliers d'essais, il a trouvé tout seul les indices qui comptent. On vérifie enfin qu'il reconnaît bien des photos **nouvelles**, qu'il n'a jamais vues : sinon, il aurait juste appris par cœur. »
**Pourquoi** : le texte suit les trois temps de l'apprentissage supervisé : des exemples avec la bonne réponse, une correction guidée par l'erreur, puis un test sur des cas nouveaux. Sa dernière phrase dit, sans le mot, ce qu'est la généralisation et pourquoi on la vérifie.
**Critères** : un exemple concret ; les trois mots demandés ; l'idée de correction par l'erreur ; la vérification sur du nouveau ; pas de jargon (« algorithme », « paramètre », « modèle » sont soit évités, soit expliqués).
**Erreurs fréquentes** : dépasser cinq lignes ; employer « algorithme », « modèle » ou « données d'entraînement » sans les expliquer ; oublier la vérification sur des cas nouveaux (le texte décrit alors un apprentissage par cœur).
**Variante** : écris cinq lignes de la même façon sur l'apprentissage **par renforcement**, avec un exemple de la vie courante et les mots « essai », « récompense » et « explorer ». (Piste : un enfant qui apprend à faire du vélo ; personne ne lui donne la bonne position, mais chaque chute ou chaque mètre parcouru lui dit si c'était mieux, et il lui faut parfois oser autre chose que ce qui a marché jusque-là.)

### Ex 1.7 — Reconnaissance faciale : utile, risquée, encadrée ⚖️
Il n'y a pas de réponse unique ; voici les points qu'une bonne réponse aborde.
1. **Bénéfices.** A : plus de badge perdu ou prêté, entrée plus rapide (pour les abonnés et la salle). B : moins de vols (pour le magasin).
2. **Risques.** Des erreurs dans les deux sens : un abonné refusé à tort, ou, bien plus grave, un client **accusé à tort** devant tout le monde (projet B). Des taux d'erreur qui peuvent être plus élevés pour certains groupes de personnes, si ceux-ci étaient moins bien représentés dans les données d'entraînement (1.13 : lire la data card). Une base de visages est une cible : un visage volé ne se change pas comme un mot de passe. Enfin, le risque de surveillance généralisée, et d'un usage détourné de la base.
3. **Le cadre.** Les données biométriques qui servent à identifier une personne sont des données **sensibles** : le RGPD en interdit le traitement, sauf exceptions, notamment le **consentement explicite** (article 9). A : les abonnés peuvent consentir, à condition que ce soit un vrai choix, avec une **alternative** sans biométrie (garder le badge). B : chaque client qui entre dans le magasin ne peut pas donner un consentement explicite ; le projet se heurte directement à l'article 9. L'AI Act interdit en outre, depuis février 2025, de constituer une base de visages en moissonnant des images sans cible (article 5), et encadre strictement l'identification biométrique à distance. En cas de doute, la CNIL et un juriste tranchent, pas l'ingénieur.
4. **Avant de coder.** Refuser de commencer sans une analyse juridique ; une analyse d'impact sur la protection des données (AIPD, article 35 du RGPD) ; mesurer les taux d'erreur par groupe de personnes ; prévoir une décision **humaine** avant toute accusation. **Alternatives sans biométrie** pour B : plus de personnel, des étiquettes antivol, un meilleur agencement du magasin, une caméra classique sans identification automatique.

**Pourquoi** : un visage sert à identifier une personne et ne se change pas ; c'est pour cela que le droit range ces données parmi les plus protégées, et que le consentement doit être un vrai choix. Une erreur du modèle n'a pas le même prix dans les deux projets : une porte qui ne s'ouvre pas d'un côté, une accusation publique de l'autre.
**Critères** : les deux types d'erreur sont distingués ; le consentement est discuté pour chaque projet ; au moins une règle (RGPD ou AI Act) est citée correctement ; une alternative est proposée.
**Erreurs fréquentes** : ne voir qu'une erreur, l'abonné refusé, en oubliant le client accusé à tort ; croire qu'un panneau « magasin sous vidéosurveillance » vaut consentement explicite ; réduire la question à la précision du modèle : même sans aucune erreur, le projet B resterait une surveillance biométrique de tous les clients.
**Variante** : projet C, une entreprise veut mesurer l'attention de ses salariés en réunion en analysant les émotions sur leur visage. Qu'en dis-tu ? (C'est interdit dans l'Union européenne depuis le 2 février 2025 : l'AI Act, article 5, interdit la reconnaissance des émotions sur le lieu de travail, sauf pour des raisons médicales ou de sécurité ; voir l'encadré de la fiche.)

### Ex 1.8 — Galton (1886) : l'origine du mot « régression » 📄
1. Les tailles de **930 enfants adultes** et de leurs parents, soit **205** couples (ses « parentages ») ; le total du tableau I donne 928 enfants, un petit écart fréquent dans les données anciennes.
2. Les femmes sont en moyenne plus petites que les hommes : pour comparer toutes les tailles sur la même échelle, Galton « transmute » la taille des femmes en l'équivalent masculin en la multipliant par **1,08** (il dit que 1,08 convient mieux à ses données que 1,07 ou 1,09). C'est ce qui permet de faire la moyenne d'un père et d'une mère, la taille mi-parentale.
3. 72 pouces, c'est $72 - 68{,}25 = 3{,}75$ pouces au-dessus de la moyenne. Écart de l'enfant : $\frac{2}{3} \times 3{,}75 = 2{,}5$ pouces, donc **70,75 pouces**, soit $70{,}75 \times 2{,}54 \approx$ **179,7 cm**. Pour 64,5 pouces (3,75 pouces en dessous) : **65,75 pouces**, soit **167,0 cm**. Les enfants sont plus proches de la moyenne que leurs parents, dans les deux sens.
4. $\hat{y} = 68{,}25 + \frac{2}{3}(x - 68{,}25) = \frac{2}{3}x + 22{,}75$ : c'est une droite de pente $w = \frac{2}{3}$ et d'ordonnée à l'origine $b = 22{,}75$. Une pente plus petite que 1 : c'est cela, la « régression vers la moyenne ». C'est une des premières droites de régression de l'histoire.
5. **Non.** Les enfants de parents extrêmes sont moins extrêmes en moyenne, mais des parents moyens ont aussi des enfants très grands ou très petits (la taille dépend d'autres facteurs que les parents : l'alimentation, le hasard génétique…). La dispersion de la population reste la même d'une génération à l'autre. La régression vers la moyenne est un **effet statistique**, qui apparaît dès que deux mesures ne sont pas parfaitement liées, pas une force biologique qui rapproche tout le monde. On la retrouve partout : un élève exceptionnellement bon à un examen le sera en moyenne un peu moins au suivant.
6. Galton est aussi l'inventeur du mot « **eugénisme** » (1883) : le mouvement eugéniste qu'il a lancé a inspiré, au XXᵉ siècle, des politiques racistes et des stérilisations forcées. Lire une source ancienne demande de séparer la méthode (la droite de régression, toujours utile) des intentions et des conclusions de l'auteur. Pour le ML, le parallèle est direct : un dataset reflète les choix, les préjugés et le contexte de ceux qui l'ont constitué (les data cards de 1.13, et le chapitre B6).

**Pourquoi** : la taille d'un enfant dépend de celle de ses parents, mais pas seulement (l'alimentation, le hasard génétique) ; la meilleure prévision ne garde donc qu'une partie de l'écart des parents à la moyenne, ici les deux tiers. Une pente plus petite que 1 en est la trace : c'est ce que Galton a appelé « régression ».
**Erreurs fréquentes** : en 3, prendre les deux tiers de la **taille** des parents au lieu des deux tiers de leur **écart** à la moyenne ($\frac{2}{3} \times 72 = 48$ pouces, absurde) ; en 4, garder $b = 68{,}25$ en oubliant de développer $-\frac{2}{3} \times 68{,}25$ ; en 5, voir dans la régression vers la moyenne une force qui rapprocherait peu à peu tout le monde de la moyenne.
**Variante** : pour quelle taille mi-parentale la prévision est-elle égale à la taille des parents ? (Il faut $\frac{2}{3}x + 22{,}75 = x$, soit $x = 68{,}25$ pouces, la moyenne : seuls des parents exactement moyens ont, en moyenne, des enfants de leur taille.)
**Source** : F. Galton, « [Regression Towards Mediocrity in Hereditary Stature](https://galton.org/essays/1880-1889/galton-1886-jaigi-regression-stature.pdf) », *Journal of the Anthropological Institute*, vol. 15, 1886, p. 246-263 ; sur l'eugénisme : [NHGRI, « Eugenics and Scientific Racism »](https://www.genome.gov/about-genomics/fact-sheets/Eugenics-and-Scientific-Racism).

<a id="entretien"></a>

## 💼 Entretien

### 1.E1 — Expliquer le machine learning à un recruteur non technique

**Réponse modèle en 60 secondes** : « Le machine learning, c'est apprendre à un ordinateur à faire une tâche à partir d'exemples plutôt qu'en lui écrivant toutes les règles. Prenons la détection de fraude bancaire : écrire à la main toutes les règles d'une fraude est impossible, les fraudeurs changent sans cesse. On donne donc au programme des milliers de transactions passées, chacune marquée « fraude » ou « normale ». Il fait des prédictions, on mesure ses erreurs, et il ajuste ses réglages pour se tromper de moins en moins. Ensuite, et c'est essentiel, on le teste sur des transactions qu'il n'a jamais vues, pour vérifier qu'il a appris une règle générale et pas juste mémorisé les exemples. Un modèle n'est jamais parfait : il faut le surveiller une fois en production, et le réentraîner quand les données changent. »
**Relances possibles** : « Et le deep learning, c'est quoi la différence ? » (une façon de construire les modèles en couches de neurones, qui apprennent elles-mêmes leurs indices ; indispensable pour les images, le son, le texte) · « Quand ne faut-il pas utiliser de machine learning ? » (quand des règles simples suffisent, quand on n'a pas de données, ou quand une erreur coûte trop cher et doit s'expliquer) · « Qu'est-ce qui fait un bon modèle ? » (de bonnes données avant tout, une évaluation honnête sur des données nouvelles).

### 1.E2 — Paramètres, hyperparamètres et jeu de test

**Réponse modèle en 60 secondes** : « Les paramètres sont les valeurs que l'algorithme apprend lui-même pendant l'entraînement : les poids d'un réseau de neurones, les seuils d'un arbre de décision. Les hyperparamètres sont les réglages que je choisis avant l'entraînement : le learning rate, le nombre d'epochs, la profondeur maximale d'un arbre. Pour savoir si le modèle généralise, je mets de côté dès le départ un jeu de test qu'il ne voit jamais pendant l'apprentissage. Si l'on triche, par exemple en entraînant sur les données de test, en réglant les hyperparamètres d'après le score de test, ou en laissant une feature qui contient la réponse, le score devient optimiste, parfois parfait, et s'effondre en production. Pour régler les hyperparamètres proprement, j'utilise un troisième jeu, la validation, ou une validation croisée, et je ne regarde le test qu'une seule fois, à la fin. »
**Relances possibles** : « Comment choisissez-vous le learning rate ? » (quelques valeurs sur une échelle logarithmique, en regardant la loss sur la validation ; ch. 19) · « Qu'est-ce qu'une fuite de données ? » (une information interdite qui se glisse dans l'entraînement : le jeu de test utilisé pour entraîner ou régler le modèle, ou une feature qu'on n'aurait pas au moment de prédire, comme `species_code` en 1.20 ; ch. 8 et 12) · « Quelle taille pour le jeu de test ? » (souvent 20 à 30 % des données ; assez pour que le score soit stable, ch. 8).

### 1.E3 — Supervisé, non supervisé, auto-supervisé, renforcement : un exemple chacun

**Réponse modèle en 60 secondes** : « En supervisé, chaque exemple a sa bonne réponse, son label : prédire si un e-mail est un spam à partir d'e-mails déjà triés, ou prédire le prix d'un appartement ; c'est de la classification ou de la régression. En non supervisé, il n'y a pas de labels : on cherche une structure, par exemple regrouper des clients selon leurs habitudes d'achat pour adapter le marketing, ou réduire le nombre de variables. En auto-supervisé, les données fabriquent leurs propres labels : un modèle de langage apprend à prédire le mot suivant d'un texte, ce qui permet d'apprendre sur des milliards de textes sans annotation ; c'est ainsi qu'on pré-entraîne les LLM. En renforcement, un agent agit dans un environnement et reçoit des récompenses : un programme qui apprend à jouer au go, un robot qui apprend à marcher, ou l'alignement d'un LLM sur les préférences humaines. »
**Relances possibles** : « Et le semi-supervisé ? » (peu de données étiquetées et beaucoup de non étiquetées, qu'on exploite aussi) · « Quelle famille pour un système de recommandation ? » (souvent du supervisé à partir des clics passés ; parfois du renforcement quand on optimise l'engagement dans la durée) · « Pourquoi le renforcement est-il difficile ? » (récompenses rares et tardives, compromis exploration-exploitation, 1.22).

### 1.E4 — Un LLM, c'est quoi ? Réponse en une minute

**Réponse modèle en 60 secondes** : « Un LLM, un grand modèle de langage, est un réseau de neurones profond, de l'architecture Transformer, avec des milliards de paramètres. Il est d'abord pré-entraîné sur une immense quantité de textes à une tâche très simple : prédire le morceau de mot suivant, le token, à partir de tout ce qui précède. C'est de l'apprentissage auto-supervisé : le texte fournit lui-même la réponse, pas besoin de labels humains. En apprenant à bien prédire la suite, il capte la grammaire, beaucoup de connaissances et des capacités de raisonnement. Ensuite, on l'affine sur des exemples de dialogues, puis on l'aligne sur les préférences humaines, par apprentissage par renforcement ou des méthodes proches comme DPO, pour qu'il soit utile et évite les réponses dangereuses. Les modèles les plus récents apprennent aussi, par renforcement, à raisonner par étapes sur des problèmes dont la réponse se vérifie, comme les mathématiques ou le code. Ses limites : il peut produire des affirmations fausses avec aplomb, et ses connaissances s'arrêtent à ses données d'entraînement ; d'où des techniques comme le RAG, qui lui donne des documents à consulter. »
**Relances possibles** : « Qu'est-ce qu'un token ? » (un morceau de texte, souvent un bout de mot ; bonus B2) · « Pourquoi dit-on qu'un LLM hallucine ? » (il génère la suite la plus plausible, pas la plus vraie) · « Quelle différence entre fine-tuning et RAG ? » (le fine-tuning modifie les paramètres ; le RAG ajoute des documents dans le prompt sans toucher au modèle ; bonus B4).

<a id="notebook"></a>

## Notebook, parties A à D

### Ex 1.9 — Penguins : échantillons, features et labels 📦
a) **344** · b) **7** · c) **[152, 124, 68]** · d) **333** · e) **[39,1 ; 18,7 ; 181 ; 3 750]** · f) **`"body_mass_g"`**.
**Démarche** : `len(penguins_all)`, `penguins_all.shape[1] - 1`, `penguins_all["species"].value_counts().tolist()`, `len(penguins_all.dropna())`, `penguins_all.loc[0, MEASURES].astype(float).tolist()`.
**Erreurs fréquentes** : compter le label parmi les features (8) ; oublier que `dropna()` retire aussi les 9 manchots mesurés mais au sexe inconnu (342 au lieu de 333, 0A) ; en f, répondre `"species"` (l'ancien label).
**Variante** : parmi les 7 features, combien sont des nombres ? (`penguins_all.drop(columns="species").select_dtypes("number").shape[1]` vaut 5 : les quatre mesures et `year`. `island` et `sex` sont du texte : il faudra les coder en nombres avant de les donner à la plupart des modèles, ch. 12.)

### Ex 1.10 — MNIST : une image, 784 nombres 📦
a) **(60000, 28, 28)** · b) **784** · c) **[0, 255]** · d) **5** · e) **166** · f) **1**.
**Pourquoi** : une image est une grille de 28 × 28 nombres de 0 (fond) à 255 (encre) ; seulement 166 des 784 pixels de la première image portent de l'encre, le reste est du fond. Le chiffre 1 est le plus fréquent (6 742 images), le 5 le moins (5 421) : un léger déséquilibre.
**Erreurs fréquentes** : répondre `(60000, 784)` en a (c'est la forme **aplatie**, avec `flatten=True`) ; compter les pixels nuls en e.
**Variante** : combien de pixels valent 0 sur **toutes** les images d'entraînement ? (`int((X_mnist.max(axis=0) == 0).sum())` vaut 67, surtout sur la rangée du haut et dans les coins : des features constantes, qui n'apportent aucune information, fiche §1.4.3.)

### Ex 1.11 — Holmes et Verne : le texte devient des nombres 📦
a) **[562 203, 421 336]** · b) **[65, 97, 32, 233]** · c) **[88, 103]** · d) et e) les deux vecteurs de 26 fréquences · f) **[0,0042 ; 0,0875]**.
**Une solution pour `letter_freq`** :
```python
def letter_freq(text):
    counts = Counter(c for c in text.lower() if "a" <= c <= "z")
    total = sum(counts.values())
    return np.array([counts[chr(k)] / total for k in range(ord("a"), ord("z") + 1)])
```
**Pourquoi** : le français a plus de caractères différents (les lettres accentuées). Les fréquences des lettres sont une signature de la langue : le `e` fait environ 12 % des lettres de *Holmes* et 14 % de celles de Verne, le `w` 2,6 % contre 0,08 %. Les deux moitiés de *Holmes* sont à 0,004 l'une de l'autre, Holmes et Verne à 0,088 : vingt fois plus loin. Un texte devient un vecteur, et la distance entre vecteurs (0B) mesure la ressemblance des textes.
**Erreurs fréquentes** : oublier `.lower()` (les majuscules ne sont pas comptées) ; diviser par la longueur totale du texte au lieu du nombre de lettres gardées (les fréquences ne font plus 1 en tout) ; inverser l'ordre des deux distances en f.
**Variante** : compter aussi les lettres accentuées en les ramenant à leur lettre de base (`unicodedata.normalize("NFD", …)`) : l'écart entre les deux langues sur le `e` grandit.

### Ex 1.12 — Taches solaires : tracer, lisser, repérer le cycle 📦
a) **3 327** · b) **1778** · c) **12** · d) **1958** · e) **24** · f) **11,0** ans.
**Démarche** : `sun.loc[sun["sunspots"].idxmax(), "year"]` ; `smooth = sun["sunspots"].rolling(13).mean()` ; `smooth.isna().sum()` ; `sun.loc[smooth.idxmax(), "year"]` ; `peaks = local_maxima(smooth.to_numpy(), 60)` ; `np.diff(sun["decimal_year"].to_numpy()[peaks]).mean()`.
**Pourquoi** : une moyenne sur 13 mois n'existe qu'à partir du 13ᵉ mois, d'où 12 valeurs manquantes. Le mois record (mai 1778, 398 taches) n'est pas au sommet du plus grand cycle lissé (1958) : un mois isolé peut être exceptionnel sans que le cycle entier le soit. Les 24 sommets correspondent aux cycles solaires numérotés 1 à 24 ; le 25ᵉ, en cours, n'est pas encore détecté, car il faut 5 ans de données après un sommet. Comme la moyenne porte sur le mois et les 12 mois **précédents**, la courbe lissée est décalée d'environ 6 mois vers la droite (une moyenne centrée, `rolling(13, center=True)`, place le sommet en 1958 aussi, mais en mars au lieu de septembre).
**Erreurs fréquentes** : lire l'index (un numéro de ligne) au lieu de l'année ; répondre 13 en c ; calculer la période avec les positions (en mois) au lieu des années.
**Variante (régression)** : prévoir chaque mois par la valeur du mois précédent (la prévision « naïve ») et mesurer l'écart moyen en valeur absolue depuis 1900 : environ 19 taches. Tout modèle de prévision du ch. 22 devra faire mieux que cette référence.

### Ex 1.13 — Lire les data cards des quatre fils rouges 🛠️
Réponses dans le notebook de solutions (cellule « Réponses (1.13) ») : Penguins CC0 ; MNIST CC BY-SA 3.0 ; Holmes et Verne domaine public ; taches solaires **CC BY-NC 4.0**, le seul qu'on ne peut pas utiliser dans un produit vendu. Le raccourci de Penguins : l'espèce se devine à l'île. MNIST est « résolu ». Holmes et Verne : un auteur, un siècle, une taille minuscule face aux corpus des LLM. Les taches solaires : exclure les mois provisoires (`definitive_only=True`).
**Pourquoi** : dans une licence Creative Commons, seule la clause NC (*non commercial*) interdit la vente ; BY (citer les auteurs) et SA (partager un dérivé sous la même licence) posent des conditions sans l'interdire. Les biais et les limites viennent de la façon dont les données ont été recueillies : des espèces qui ne vivent pas sur toutes les îles, un dataset que les modèles réussissent presque parfaitement, un seul auteur par roman, des mois encore provisoires.
**Erreurs fréquentes** : croire que CC BY-SA interdit l'usage commercial (il l'autorise, en citant les auteurs et en partageant tout dataset dérivé sous la même licence) ; ne lire que la licence et sauter « Biais et limites » ; garder les mois provisoires des taches solaires, qui peuvent encore être révisés.
**À retenir** : avant d'utiliser un dataset, lire sa fiche : **provenance, licence, biais, limites**. C'est une question classique d'un recruteur sur un projet de portfolio : « D'où viennent vos données, et avez-vous le droit de vous en servir ? »
**Variante** : lis la data card de CIFAR-10 (`wb.datasets.dataset_card("cifar10")`). Quelle licence, et quel problème de provenance signale-t-elle ? (Aucune licence formelle : les auteurs demandent seulement de citer leur rapport technique. Et la base dont CIFAR-10 est tiré, « 80 Million Tiny Images », a été retirée par ses auteurs en 2020, après la découverte de labels offensants.)

### Ex 1.14 — Mémoriser n'est pas apprendre 🔮
a) **`"parfaite"`** (100 %) · b) **`"mauvaise"`** (40 %) · c) **0** · d) **`"Adelie"`**.
**Pourquoi** : sur l'entraînement, le mémoriseur retrouve chaque manchot dans sa table. Mais aucun manchot du test n'a exactement les mêmes quatre mesures qu'un manchot appris : pour tous, il répond l'espèce la plus fréquente de l'entraînement, Adélie, et son accuracy est simplement la part des Adélie dans le test (40 sur 100). Le modèle a une **capacité** immense (il retient tout) et ne **généralise** rien.
**Erreurs fréquentes** : prédire « bonne » en b, en imaginant que le mémoriseur retrouvera des manchots semblables. Il ne cherche que des mesures **identiques** ; chercher le manchot **le plus proche** serait déjà apprendre quelque chose (les k plus proches voisins, ch. 13).
**Variante** : fais apprendre le mémoriseur sur 5 000 images de MNIST (`X, y = wb.datasets.load_mnist("train", n=5000, seed=0, flatten=True)`, puis `Memorizer().fit(pd.DataFrame(X), pd.Series(y))`) et évalue-le sur les 10 000 images de test (`wb.datasets.load_mnist("test", flatten=True)`). (100 % sur l'entraînement. Sur le test, aucune image n'est identique, pixel pour pixel, à une image apprise : il répond toujours « 1 », le chiffre le plus fréquent, et fait 11,35 %, la part des 1 dans le test ; à peine mieux que le hasard, 10 %.)

### Ex 1.15 — Un système expert pour les manchots 🔨
a) **0,850** · b) **0,92** · c) **`"Gentoo"`**.
**Démarche** :
```python
def expert_rule(bill_length, bill_depth, flipper_length, body_mass):
    if body_mass > 4700:
        return "Gentoo"
    if bill_length > 45:
        return "Chinstrap"
    return "Adelie"
```
**Pourquoi** : 24 des 78 Gentoo d'entraînement (31 %) sont mal classés : les plus légers (souvent des femelles) passent sous les 4 700 g, puis deviennent Chinstrap (bec long) ou Adélie. Le score sur le test (92 %) est meilleur que sur l'entraînement (85 %) par hasard : 100 manchots, c'est peu, et ce test-ci contient moins de Gentoo légers.
**Erreurs fréquentes** : `>=` au lieu de `>` : six manchots pèsent exactement 4 700 g (cinq Gentoo et un Adélie), et l'accuracy devient 0,871 sur l'entraînement et 0,91 sur le test ; or la consigne dit « plus de », strictement ; comparer les nombres d'erreurs bruts plutôt que les proportions en c (les espèces n'ont pas le même effectif).
**Variante** : ajoute une règle pour les Gentoo légers (la nageoire, graphique de droite) : c'est exactement le cycle sans fin des systèmes experts (§1.1.2), et le début du défi 1.25.

### Ex 1.16 — La boucle d'entraînement à la main 🔨
a) **14,812** · b) **[0,0232 ; 0,0282]** · c) **[1,6622 ; 0,4421]** · d) **[2,037 ; 1,010]** · e) **0,2485**.
**Démarche** :
```python
def predict_line(w, b, x):
    return w * x + b

def mse(w, b, x, y):
    return float(np.mean((predict_line(w, b, x) - y) ** 2))

def train_step(w, b, x_i, y_i, eta):
    error = y_i - predict_line(w, b, x_i)
    return w + eta * error * x_i, b + eta * error

def train_line(x, y, eta, n_epochs):
    w, b = 0.0, 0.0
    losses = [mse(w, b, x, y)]
    for _ in range(n_epochs):
        for x_i, y_i in zip(x, y):
            w, b = train_step(w, b, x_i, y_i, eta)
        losses.append(mse(w, b, x, y))
    return w, b, losses
```
**Pourquoi** : c'est la figure 1.8 du livre en code : prédire, comparer, corriger, un échantillon à la fois. L'erreur $y - \hat{y}$ règle la taille **et le sens** de la correction : quand la prédiction est trop basse, l'erreur est positive et $w$ et $b$ augmentent. La loss chute de 14,8 à 1,09 en une epoch, puis à 0,25 : la droite apprise ($w \approx 2{,}04$, $b \approx 1{,}01$) est très proche de celle qui a servi à fabriquer les données ($w = 2$, $b = 1$) ; l'écart vient du bruit. Sa loss (0,2485) est d'ailleurs à peine au-dessus de celle de la meilleure droite possible (0,2483), que la régression linéaire du ch. 9 calcule directement.
**Pourquoi cette règle ?** Avec $L = (\hat{y} - y)^2$, 0B (101.5.3) a montré que $\frac{\partial L}{\partial w} = 2(\hat{y} - y)\,x$ et $\frac{\partial L}{\partial b} = 2(\hat{y} - y)$. La règle fait donc un pas de $\frac{\eta}{2}$ fois **moins** la dérivée : c'est une descente de gradient, sur un échantillon à la fois (ch. 5 et 19).
**Erreurs fréquentes** : écrire l'erreur $\hat{y} - y$ (les corrections partent dans le mauvais sens et la loss explose) ; recalculer $\hat{y}$ avec le $w$ déjà corrigé avant de corriger $b$ (les deux corrections doivent utiliser la **même** erreur ; sinon $b$ change un peu, et 1.16 b et c échouent) ; oublier `losses[0]` (la loss avant tout entraînement) ; repartir de $w = b = 0$ à chaque epoch.
**Variante** : calcule la meilleure droite d'un seul coup avec `np.polyfit(x_line, y_line, 1)`, qui renvoie `[w, b]`. ($w \approx 2{,}044$ et $b \approx 1{,}010$ : après 20 epochs, ta boucle en est à moins de 0,01 près. C'est la « formule » de la fiche §1.2, établie au ch. 9 ; pour un réseau de millions de paramètres, il n'y en a pas, et il ne reste que les retouches.)

### Ex 1.17 — Learning rate : trop prudent, trop pressé 🔬
Réponses dans le notebook de solutions (cellule « Réponses (1.17) ») ; le graphique de droite, sans $\eta = 0{,}5$, permet de comparer les trois autres courbes. En bref : $\eta = 0{,}001$ est trop prudent (loss encore à 0,38 après 20 epochs) ; $\eta = 0{,}5$ diverge (loss d'environ $10^{53}$, $w \approx 10^{26}$) ; $\eta = 0{,}1$ descend le plus vite mais reste au-dessus de $\eta = 0{,}01$ (0,28 contre 0,25) ; réduire $\eta$ au fil des epochs combine vitesse et précision.
**Pourquoi ça diverge** : pour un échantillon d'abscisse $x$, une correction multiplie l'erreur sur cet échantillon par $1 - \eta(x^2 + 1)$. Dès que $\eta(x^2 + 1) > 2$, la correction fait plus que traverser la cible : l'erreur change de signe et **grandit**. Avec des $x$ jusqu'à 3, cela commence dès $\eta > 0{,}2$ pour les échantillons les plus éloignés ; à $\eta = 0{,}5$, c'est le cas de tous ceux où $\lvert x \rvert > 1{,}73$, près de la moitié, et l'entraînement explose. Le bon learning rate dépend donc de l'échelle des données (encore une raison de les mettre à l'échelle, ch. 12).
**Erreurs fréquentes** : renvoyer tout le tuple de `train_line` au lieu de la liste des losses (l'élément `[2]`) ; déclarer $\eta = 0{,}1$ le meilleur parce qu'il descend le plus vite au début, alors qu'il finit au-dessus de $\eta = 0{,}01$ ; lire les courbes sans voir que l'axe vertical des deux graphiques est logarithmique.
**Variante** : écris `train_line_decay`, avec $\eta_k = \frac{\eta_0}{1 + k}$ à l'epoch $k$, et compare pour $\eta_0 = 0{,}1$.

### Ex 1.18 — Un arbre de décision apprend les règles à ta place 📦
a) **0,961** · b) **0,97** · c) **`"flipper_length_mm"`** · d) **206,5** · e) **5**.
**Pourquoi** : l'arbre a trouvé seul trois questions : la nageoire (≤ 206,5 mm), puis la longueur du bec (≤ 43,35 mm) d'un côté, l'épaisseur du bec (≤ 17,65 mm) de l'autre. Il fait 97 % sur le test, 5 manchots de mieux que la biologiste (92 %). `max_depth=2` est un hyperparamètre ; les questions et leurs seuils sont des paramètres, appris sur les 233 manchots d'entraînement.
**Erreurs fréquentes** : en c, répondre `bill_length_mm` (c'est la **deuxième** question de la branche de gauche) ; en e, un nombre à virgule (0,05) au lieu d'un nombre de manchots.
**Variante** : essaie `max_depth=None` (pas de limite) : 100 % sur l'entraînement, mais 95 % sur le test. Plus de capacité, meilleure mémoire, moins bonne généralisation : un avant-goût de l'overfitting (ch. 9).

### Ex 1.19 — Un manchot d'une espèce jamais vue 🔮
a) **`"Chinstrap"`** · b) **`"Adelie"`** · c) **0,67**.
**Pourquoi** : l'empereur a une nageoire de 340 mm (> 206,5 : branche de droite), puis un bec épais de 22 mm (> 17,65) : l'arbre conclut « Chinstrap ». Le poussin a une petite nageoire, puis un bec court : « Adélie ». Un arbre n'a pas de feuille « inconnu » : il répond **toujours** une classe connue. Ses « probabilités » ne sont que les proportions d'espèces dans la feuille atteinte (2 Adélie et 4 Chinstrap pour l'empereur, d'où 4/6 ≈ 0,67).
**Erreurs fréquentes** : répondre « inconnu » (un classifieur ne répond qu'avec les classes vues à l'entraînement) ; s'arrêter à la première question pour l'empereur (« longue nageoire, donc Gentoo ») alors que l'arbre en pose une seconde ; en c, lire la première colonne de `predict_proba` (Adélie, 0,33) au lieu de la probabilité de l'espèce prédite.
**À retenir** : un modèle n'est fiable que sur des données qui ressemblent à celles de l'entraînement. Repérer les entrées étranges (détection d'anomalies, données « hors distribution ») est un problème à part entière.
**Variante** : ajoute un garde-fou qui répond « inconnu » dès qu'une mesure sort de l'intervalle [minimum, maximum] du jeu d'entraînement (`X_train.min()`, `X_train.max()`). Quels visiteurs refuse-t-il ? (Les deux, et sur leurs quatre mesures : le bec de l'empereur (80 mm) dépasse le maximum de l'entraînement, 59,6 mm, et celui du poussin (20 mm) est sous le minimum, 32,1 mm. C'est une détection d'anomalies très simple ; elle laisserait passer un intrus dont chaque mesure, prise seule, reste dans les intervalles.)

### Ex 1.20 — Le score trop beau pour être vrai 🐛
Deux erreurs : la **fuite du label**, une forme de fuite de données (`species_code`, l'espèce en chiffres, dans les features) et le **test vu à l'entraînement** (`fit` sur les 333 manchots). Après correction : **95 %** sur le test.
**Démarche** :
```python
features_20 = MEASURES

def train_and_evaluate_20():
    model = DecisionTreeClassifier(random_state=0)
    model.fit(train[features_20], train["species"])
    return model, model.score(test[features_20], test["species"])
```
**Pourquoi il faut corriger les deux** : avec seulement la fuite retirée, l'arbre sans limite de profondeur a appris par cœur les 100 manchots du test pendant l'entraînement : encore 100 %. Avec seulement le découpage corrigé, `species_code` donne la réponse : encore 100 %. Chaque erreur suffit à fausser le score.
**Erreurs fréquentes** : s'arrêter après une seule correction (le score reste à 100 %, voir ci-dessus) ; mesurer le score sur `train` au lieu de `test` (encore 100 % : l'arbre sans limite de profondeur connaît par cœur ses 233 manchots d'entraînement) ; garder `species_code` parce qu'il « sert aux couleurs » : une colonne pratique pour un graphique n'a pas sa place parmi les features si elle contient la réponse.
**Réflexe professionnel** : devant un score parfait, se demander (1) si une feature contient la réponse, ou une information qu'on n'aura pas au moment de prédire, et (2) si le jeu de test a servi, de près ou de loin, à l'entraînement ou aux réglages.
**Variante** : ajoute aux features le rapport `bill_length_mm / bill_depth_mm` (la forme du bec), calculé sur `train` et sur `test`. Est-ce une fuite ? (Non : il se calcule à partir de mesures connues au moment de prédire. L'arbre passe ici à 98 % sur le test ; mais 3 manchots de plus sur 100, c'est peu : il faudrait le confirmer par une validation croisée, ch. 8.)

### Ex 1.21 — Regrouper les manchots sans leurs labels 📦
Pureté **≈ 0,68** sur les mesures brutes, **≈ 0,92** sur les mesures mises à l'échelle.
**Démarche** :
```python
def purity(groups, labels):
    table = pd.crosstab(np.asarray(groups), np.asarray(labels))
    return float(table.max(axis=1).sum() / len(labels))

X_scaled = (penguins[MEASURES] - penguins[MEASURES].mean()) / penguins[MEASURES].std()
```
**Pourquoi** : sur les mesures brutes, la masse (des milliers de grammes) domine les distances (1.R1) : les groupes sont des tranches de poids, et les Gentoo sont coupés en deux. Une fois chaque mesure divisée par son écart-type, toutes comptent autant : les Gentoo forment un groupe à eux seuls, et il ne reste que des Adélie et des Chinstrap mélangés. Les groupes n'ont pas de nom : c'est nous qui décidons, après coup, que « le groupe 1 » correspond aux Gentoo.
**Erreurs fréquentes** : diviser par le nombre de groupes au lieu du nombre de manchots ; utiliser `max(axis=0)` (par espèce au lieu de par groupe) ; standardiser avec la moyenne et l'écart-type d'une seule colonne.
**Variante** : relance k-means sur les mesures mises à l'échelle avec une autre graine, `KMeans(n_clusters=3, n_init=10, random_state=2)`. Les groupes changent-ils ? (Ce sont exactement les mêmes groupes de manchots, et la pureté reste 0,919 ; seuls les numéros changent : les Gentoo forment le groupe 0 au lieu du groupe 1. Un numéro de groupe est arbitraire : on ne lui donne un sens qu'après avoir regardé qui est dedans.)

### Ex 1.22 — L'agent cuisinier : apprendre par la récompense 🔬
Une solution pour `cook_year` est dans le notebook de solutions. L'année contrôlée (`explore = 0.1`, `default_rng(0)`) donne **274** repas mangés sur 365. Résultats sur 200 années par valeur :

| explore | repas mangés | années finies sur la recette préférée |
|---|---|---|
| 0 | 72,2 % | 69 % |
| 0,05 | 74,1 % | 90 % |
| 0,1 | 73,4 % | 96 % |
| 0,2 | 71,0 % | 99 % |
| 0,5 | 60,9 % | 100 % |
| 1 | 43,9 % | 22 % |

**Pourquoi** : sans exploration, un premier essai malchanceux condamne une recette pour toujours (son taux tombe à 0 %) : près d'une année sur trois, le cuisinier ne découvre jamais les pâtes au fromage. En explorant tout le temps, il ne profite jamais de ce qu'il sait (44 %, la moyenne des goûts). Entre les deux, 5 à 10 % d'exploration trouve presque toujours la meilleure recette et mange le plus. Le maximum possible serait 80 % (toujours les pâtes au fromage), inaccessible sans connaître les goûts à l'avance : il faut payer un peu pour apprendre.
**Erreurs fréquentes** : tirer les nombres aléatoires dans un autre ordre (les résultats changent, même si la stratégie est juste) ; diviser par zéro si l'on n'essaie pas chaque recette au début ; mettre à jour les taux avant de choisir.
**Variante** : l'enfant se lasse (le livre, §1.6) ; multiplie le goût d'une recette par 0,9 chaque fois qu'elle est servie deux soirs de suite. Quelle stratégie s'adapte le mieux ? (Le monde qui change est un vrai défi du renforcement, ch. 11 et 26.)

### Ex 1.23 — Un réseau de neurones en boîte noire sur MNIST 📦
a) **101 770** · b) **3** · c) environ **93,6 %** et 644 erreurs sur 10 000 ici (en `FAST_MODE`, 5 000 images d'entraînement ; un peu plus ou un peu moins selon la machine).
**Pourquoi** : `mlp.coefs_` contient deux matrices de formes (784, 128) et (128, 10), `mlp.intercepts_` deux vecteurs de 128 et 10 biais : $100\,352 + 1\,280 + 128 + 10 = 101\,770$, exactement 1.3 c. scikit-learn compte **3 couches** : l'entrée, la couche cachée et la sortie. Le score dépend légèrement de la machine (l'ordre des calculs en virgule flottante) : c'est pourquoi il n'est vérifié que par un seuil (plus de 90 %). Avec les 60 000 images (`FAST_MODE = False`), environ 97,8 %. Pour atteindre les 99,05 % du petit réseau du livre (95 erreurs), il faut entraîner plus longtemps, ou mieux, utiliser un réseau **convolutif**, conçu pour les images (ch. 21).
**Erreurs fréquentes** : oublier les biais (101 632) ; répondre 2 couches en b (les couches « qui calculent ») : les deux conventions existent, lis la documentation de l'outil.
**À retenir** : sans une seule règle écrite, trois lignes de code reconnaissent plus de 9 chiffres sur 10. Beaucoup d'erreurs restantes sont des chiffres réellement ambigus.
**Variante** : essaie `hidden_layer_sizes=(64, 32)`, deux couches cachées plus étroites. Combien de paramètres et de couches pour scikit-learn, et quelle accuracy ? ($784 \times 64 + 64 + 64 \times 32 + 32 + 32 \times 10 + 10 = 52\,650$ paramètres, à peu près moitié moins ; `n_layers_` vaut 4 ; environ 93,4 % ici, avec 662 erreurs : presque autant, avec deux fois moins de paramètres.)

### Ex 1.24 — Fabriquer du faux Holmes et du faux Verne 🔨
Une solution pour `bigram_counts` et `generate` est dans le notebook de solutions. Dans *Holmes*, « q » est suivi 405 fois de « u » (et 2 fois d'un point) ; le faux texte ressemble à de l'anglais de loin (« Tha Euthinto », « An than asiomysirk? »), mais la plupart des « mots » n'existent pas.
**Pourquoi** : le modèle ne connaît que le caractère précédent. Il reproduit les paires fréquentes (« th », « he », « qu », les accents du français) mais ne peut pas faire de mots, et encore moins des phrases. Avec 2 ou 3 caractères de contexte (trigrammes), les mots apparaissent ; un LLM utilise des milliers de tokens de contexte et des milliards de paramètres, mais son pré-entraînement repose sur la même tâche : prédire la suite. Et comme ici, **aucun label humain** : le texte fournit la réponse (apprentissage auto-supervisé).
**Erreurs fréquentes** : oublier de trier les candidats (le tirage n'est plus reproductible d'une machine à l'autre) ; construire les poids avec des entiers et oublier de normaliser (`p` doit sommer à 1) ; générer `length` caractères **en plus** de `start`.
**Variante** : écris `ngram_counts(text, n)`, qui prend pour contexte les $n - 1$ derniers caractères ; compare les textes générés pour $n = 2, 3, 5$. À $n = 5$, le modèle reproduit de longs morceaux du livre : avec beaucoup de contexte et peu de texte, il se met à mémoriser (1.14).

### Ex 1.25 — Battre l'expert : 95 % avec tes propres règles 🏆
Une solution (d'autres existent) : « nageoire > 206 mm **et** épaisseur du bec < 17,5 mm : Gentoo ; sinon, bec > 44 mm : Chinstrap ; sinon : Adélie », soit **96,1 %** sur l'entraînement et **97 %** sur le test.
**Pourquoi** : sur les graphiques d'entraînement, les Gentoo se détachent avec leur longue nageoire **et** leur bec fin (les rares Chinstrap à longue nageoire ont un bec épais) ; parmi les autres, la longueur du bec sépare assez bien Chinstrap et Adélie. C'est presque l'arbre de 1.18 : tu as refait, à la main et en regardant des graphiques, le travail qu'un algorithme fait seul en une fraction de seconde.
**Erreurs fréquentes** : régler ses seuils en regardant le score **sur le test** (on obtient facilement 99 %, mais ce score ne mesure plus rien : l'erreur de 1.20) ; utiliser l'île (le raccourci signalé par la data card, 1.13) ; empiler dix règles pour les cas particuliers (on colle aux exemples, comme le mémoriseur).
**Pour aller plus loin** : fais la même chose sur MNIST, avec 784 pixels… Tu comprends pourquoi personne n'écrit ces règles à la main, et pourquoi le machine learning a gagné.
