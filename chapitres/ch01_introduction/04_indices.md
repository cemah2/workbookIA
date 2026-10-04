# 1 · Introduction au machine learning et au deep learning — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ Papier-crayon](#papier) · [🧮 🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook, parties A à D](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 1.Q1 — Règles écrites ou règles apprises ?

<details><summary>Indice 1</summary>

Demande-toi : quelqu'un a-t-il **écrit** la règle à l'avance (un taux, un seuil, une liste), ou le programme **change-t-il** de comportement en voyant des exemples ?

</details>
<details><summary>Indice 2</summary>

Un seuil fixé à l'avance, un taux ou une liste de mots sont écrits par des humains. Un programme qui change de comportement quand on lui montre des exemples, ou qui reconnaît ce qu'on ne lui a jamais décrit, a dû apprendre.

</details>
<details><summary>Indice 3</summary>

Cherche qui a fixé la règle : un humain qui l'a écrite (un taux, un seuil, une liste) ou des exemples dont le programme l'a tirée. Le 1 applique un taux de TVA écrit dans la loi : règle écrite (É). Fais de même pour les quatre autres.

</details>

### 1.Q2 — Pourquoi les systèmes experts ont calé

<details><summary>Indice 1</summary>

Relis la fiche §1.1.2 : qui écrit les règles d'un système expert, et que se passe-t-il quand un cas imprévu arrive ?

</details>
<details><summary>Indice 2</summary>

Pense au 7 barré : une règle en plus règle-t-elle **tous** les cas imprévus ? Et saurais-tu écrire en règles comment tu reconnais un visage familier ?

</details>
<details><summary>Indice 3</summary>

1 : un système expert applique des règles que des humains ont écrites d'après ce que disent les experts ; il n'apprend rien des exemples : **faux**. Juge les quatre autres avec la fiche §1.1.2 : qui écrit les règles, ce qu'elles laissent de côté, et ce que le ML demande en échange. Une affirmation qui dit « suffit » ou « toujours » tombe dès qu'on lui trouve un seul contre-exemple : en trouves-tu un ?

</details>

### 1.Q3 — Échantillon, feature ou label ?

<details><summary>Indice 1</summary>

Le label est ce que l'on veut **prédire** ; les features sont ce que l'on connaît **au moment de prédire** ; un échantillon est **une ligne**.

</details>
<details><summary>Indice 2</summary>

Ici, on veut prédire le loyer. Pour la question 5, demande-toi quelle colonne devient la cible, et ce que devient l'ancienne cible.

</details>
<details><summary>Indice 3</summary>

1 : un échantillon est une ligne du tableau, et chaque ligne décrit un appartement : l'échantillon, c'est **un appartement**. Pour 2 et 3, classe chaque colonne avec le critère de l'indice 1 : à prédire, ou connue au moment de prédire ? Pour 4, demande-toi qui connaît le vrai loyer d'un appartement, et à quel moment. Pour 5, refais 2 et 3 avec le quartier comme colonne à prédire, puis regarde si la réponse attendue est une quantité ou une catégorie (1.Q7).

</details>

### 1.Q4 — L'école absurde : ce qui marche pour une machine

<details><summary>Indice 1</summary>

Relis la fiche §1.2.1 : l'un des deux tests reprend les faits récités, l'autre pose des questions nouvelles.

</details>
<details><summary>Indice 2</summary>

Pour 2, compare ce que cent répétitions de la même leçon produisent chez un enfant, puis chez un programme (la fin de la fiche §1.2.1 le dit en une phrase). En ML, qu'est-ce qui joue le rôle des « questions nouvelles » ?

</details>
<details><summary>Indice 3</summary>

1 : le premier test redemande exactement les faits récités toute la semaine : il mesure la **mémoire**. Pour le second, demande-toi ce qu'il faut savoir faire pour répondre à une question qu'on n'a jamais entendue. 2 : un ordinateur s'ennuie-t-il, et que fait-il de ses paramètres à chaque passage ? 3 : un élève qui a 20/20 au premier test réussit-il forcément le second ? 4 : associe chaque test à un jeu : celui sur lequel le modèle a appris, ou celui qu'on a mis de côté.

</details>

### 1.Q5 — Paramètre ou hyperparamètre ?

<details><summary>Indice 1</summary>

Un seul critère : qui fixe la valeur ? L'algorithme pendant l'entraînement (paramètre), ou toi avant l'entraînement (hyperparamètre) ?

</details>
<details><summary>Indice 2</summary>

Demande-toi, pour chaque nombre : change-t-il pendant `fit` ? Si oui, c'est l'algorithme qui le fixe ; sinon, c'est toi, au moment d'écrire `Model(...)` ou la boucle d'entraînement.

</details>
<details><summary>Indice 3</summary>

1 : les poids d'un réseau partent de valeurs tirées au hasard, puis changent à chaque correction pendant `fit` : c'est l'algorithme qui les fixe, ce sont des **paramètres**. Pose la même question pour les cinq autres : la valeur sort-elle de `fit`, ou l'écris-tu toi-même avant, dans `Model(...)` ou dans la boucle d'entraînement ?

</details>

### 1.Q6 — À quoi sert le jeu de test

<details><summary>Indice 1</summary>

Le jeu de test joue le rôle des « questions nouvelles » de l'école absurde (fiche §1.2.1 et §1.2.3).

</details>
<details><summary>Indice 2</summary>

3 : que dit un grand écart entre le score d'entraînement et celui du test sur ce que le modèle a appris ? 5 : si le choix du learning rate dépend du score sur le test, le test sert-il encore à mesurer des données **jamais vues** ?

</details>
<details><summary>Indice 3</summary>

1 : le jeu de test joue les « questions nouvelles » ; s'il avait servi à l'entraînement, son score mesurerait la mémoire du modèle, pas sa généralisation : c'est pourquoi on le met de côté **avant**. Pour les autres : 2, l'évaluation appelle-t-elle `fit` ou seulement `predict` ? 3, le modèle réussit-il aussi bien sur ce qu'il n'a jamais vu, et qu'en déduis-tu de ce qu'il a retenu ? 4, dans la boucle d'entraînement, qu'est-ce qu'on répète, et sur quelles données ? 5, relis l'encadré ⚠️ de la fiche §1.2.3 : sur quel jeu doit-on choisir un hyperparamètre ?

</details>

### 1.Q7 — Classification ou régression : six situations

<details><summary>Indice 1</summary>

La réponse est-elle une **catégorie** prise dans une liste, ou une **quantité** qui se mesure et qu'on peut additionner ou comparer ?

</details>
<details><summary>Indice 2</summary>

Un nombre n'est pas toujours une quantité : additionner deux codes postaux ou calculer « la moyenne » de 75001 et 75020 a-t-il un sens ?

</details>
<details><summary>Indice 3</summary>

1 : un prix se mesure, et la moyenne de deux prix a un sens : c'est une quantité, donc une **régression** (R). Pose la même question pour les cinq autres : la moyenne de deux réponses possibles a-t-elle un sens, ou faut-il choisir la réponse dans une liste ?

</details>

### 1.Q8 — Clustering, débruitage ou réduction de dimension ?

<details><summary>Indice 1</summary>

Clustering : on forme des **groupes** d'échantillons. Débruitage : on **nettoie** chaque échantillon. Réduction de dimension : on garde **moins de features** par échantillon.

</details>
<details><summary>Indice 2</summary>

Pour 4, regarde ce que la fiche §1.4.2 range parmi les formes de bruit. Pour 6, regarde ce qu'on donne à l'algorithme : y a-t-il une « bonne réponse » pour chaque échantillon ?

</details>
<details><summary>Indice 3</summary>

1 : on forme des groupes de clients, sans liste de catégories fixée à l'avance : **clustering**. Pour 2 à 5, applique les définitions de l'indice 1 : la tâche forme-t-elle des groupes d'échantillons, nettoie-t-elle chaque échantillon, ou garde-t-elle moins de features par échantillon ? Pour 6, demande-toi ce que serait le label d'un échantillon dans chacune de ces tâches, et qui le fournirait.

</details>

### 1.Q9 — Générateurs et renforcement : sans labels, mais pas sans retour

<details><summary>Indice 1</summary>

Relis les fiches §1.5 et §1.6, et l'encadré 🕰️ « familles d'apprentissage » : un générateur n'a pas de labels, mais il reçoit un retour ; un agent aussi.

</details>
<details><summary>Indice 2</summary>

Pour le thermostat : qui **décide** (l'agent) ? Sur quoi agit-il (l'environnement) ? Que choisit-il à chaque instant (l'action) ? Quel signal lui dit que c'était bien ou mal (la récompense) ? Pour 5, pense au cuisinier de 1.22.

</details>
<details><summary>Indice 3</summary>

1 : un générateur fabrique de **nouvelles** données (images, textes, sons) qui ressemblent à ses exemples sans les recopier. Pour les autres : 2, compare ce qu'il reçoit (des labels ? un retour ?) avec ce que reçoivent le supervisé et le non supervisé, puis cherche dans l'encadré 🕰️ le mot qu'on emploie aujourd'hui ; 3, applique au thermostat les quatre questions de l'indice 2 ; 4, quand l'agent s'est trompé, que lui apprend une récompense, et que lui aurait appris un label ? 5, si l'agent choisit toujours la meilleure action qu'il **connaît**, comment découvrirait-il qu'une autre est meilleure ?

</details>

### 1.Q10 — Profond, capacité et GPU

<details><summary>Indice 1</summary>

Relis la fiche §1.7 et §1.2.4 (capacité).

</details>
<details><summary>Indice 2</summary>

4 : repense au mémoriseur de 1.14, qui a une capacité énorme (il retient tout). 5 : quel type de calcul un réseau fait-il des milliards de fois, et un GPU sait-il le faire en parallèle ?

</details>
<details><summary>Indice 3</summary>

1 : « profond » se compte en **couches** : un réseau est profond quand il empile beaucoup de couches les unes après les autres. Pour les autres : 2, relis la fiche §1.7 : que fait un neurone de ses entrées, puis du total obtenu ? 3, qui fabrique les features : un humain, ou le réseau lui-même ? 4, un modèle qui apprend tout par cœur a une capacité énorme : d'après l'école absurde (fiche §1.2.1), cela lui garantit-il de réussir le second test ? 5, ces milliards de multiplications dépendent-elles les unes des autres, ou peut-on en faire beaucoup en même temps ?

</details>

### 1.Q11 — Panorama 2026 : où ranger ChatGPT et Stable Diffusion ?

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ « Panorama 2026 » et le schéma de la carte des familles.

</details>
<details><summary>Indice 2</summary>

1 : pour prédire le token suivant, qui fournit la « bonne réponse » ? Un humain, ou le texte lui-même ? 2 : Stable Diffusion **fabrique** des données nouvelles.

</details>
<details><summary>Indice 3</summary>

1 : la « bonne réponse » (le token suivant) est déjà dans le texte, et personne n'a besoin de l'étiqueter : c'est de l'apprentissage **auto-supervisé**. Pour les autres : 2, relie ce que fait Stable Diffusion (indice 2) à l'une des familles du chapitre, puis demande-toi avec quels modèles il est construit (§1.7) ; 3, relis la définition de l'encadré « Panorama 2026 » : entraîné une fois sur quoi, puis utilisé comment ? 4, regarde en quels morceaux l'encadré découpe le texte ; 5, cherche dans l'entraînement d'un LLM l'équivalent de chacun des quatre mots cités.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 1.R1 — 0B : distance entre deux manchots vus comme des vecteurs

<details><summary>Indice 1</summary>

La distance entre deux vecteurs est la norme de leur différence : $\sqrt{\sum_i (p_i - q_i)^2}$ (0B, 101.3.2).

</details>
<details><summary>Indice 2</summary>

Calcule d'abord les différences composante par composante, puis leurs carrés. Regarde lequel des quatre carrés domine la somme.

</details>
<details><summary>Indice 3</summary>

$\mathbf{p} - \mathbf{a} = (-0{,}4 ;\ 1{,}3 ;\ -5 ;\ -50)$ et $\mathbf{p} - \mathbf{c} = (-13{,}6 ;\ -1{,}1 ;\ -16 ;\ 25)$. Dans chacune des deux sommes de carrés, regarde le terme de la masse : c'est le plus grand des quatre.

</details>

### 1.R2 — 0A : compter les espèces avec `value_counts`

<details><summary>Indice 1</summary>

`value_counts()` s'applique à **une colonne** (une Series), pas à tout le DataFrame.

</details>
<details><summary>Indice 2</summary>

`normalize=True` transforme des comptes en… ? Pour croiser deux colonnes : `groupby` sur deux colonnes, ou une fonction pandas qui construit un tableau croisé.

</details>
<details><summary>Indice 3</summary>

1 : `df["species"].value_counts()`. Pour les autres : 2, compare la somme des valeurs avec et sans l'option ; 3, il faut croiser deux colonnes : `pd.crosstab(…, …)` attend la Series des lignes, puis celle des colonnes ; 4, imagine un classifieur « paresseux » qui répond toujours l'espèce la plus fréquente, et calcule son accuracy avec les comptes de 1.

</details>

### 1.R3 — 0B : la droite qui passe par deux points

<details><summary>Indice 1</summary>

Pente $= \frac{\Delta y}{\Delta x}$, puis l'ordonnée à l'origine se trouve en écrivant qu'un des points est sur la droite (0B.6).

</details>
<details><summary>Indice 2</summary>

$w = \frac{13 - 5}{6 - 2}$ ; puis $5 = w \times 2 + b$ donne $b$.

</details>
<details><summary>Indice 3</summary>

1 : $w = \frac{13 - 5}{6 - 2} = 2$, puis $5 = 2 \times 2 + b$ donne $b = 1$ : la droite est $y = 2x + 1$. Pour 2 et 3, remplace $x$ par 10, puis par 4, et compare au $y$ du point $C$. Pour 4, une fonction d'une ligne, `def line(x): return ...`, avec la pente et l'ordonnée à l'origine trouvées en 1.

</details>

<a id="papier"></a>

## ✏️ Papier-crayon

### Ex 1.1 — Accuracy et erreurs à l'échelle d'un centre de tri ✏️

<details><summary>Indice 1</summary>

accuracy $=$ bonnes réponses / total ; taux d'erreur $= 1 -$ accuracy (fiche §1.2.3). Un nombre moyen d'erreurs, c'est le nombre d'essais multiplié par la probabilité d'erreur.

</details>
<details><summary>Indice 2</summary>

d) Pour que le code soit juste, il faut que le 1ᵉʳ chiffre soit juste **et** le 2ᵉ **et**… : avec l'indépendance, on **multiplie** les probabilités (0B, 101.7.2). e) Un code faux, c'est le contraire (le complémentaire) d'un code juste.

</details>
<details><summary>Indice 3</summary>

a) $\frac{9\,905}{10\,000} = 0{,}9905$. b) $1 -$ a), à écrire en pourcentage. c) $1\,200\,000 \times (1 - 0{,}9905)$. d) $0{,}9905^5$. e) $240\,000 \times (1 - 0{,}9905^5)$ : garde toutes les décimales de $0{,}9905^5$ sur ta calculatrice avant de multiplier.

</details>

### Ex 1.2 — Concerts : la valeur manquante et celle de demain ✏️

<details><summary>Indice 1</summary>

Interpoler entre deux points, c'est prendre la valeur de la droite qui les relie ; au milieu (le 8 est à mi-chemin entre le 7 et le 9), c'est la moyenne des deux valeurs.

</details>
<details><summary>Indice 2</summary>

b) pente $= \frac{\text{spectateurs du 12} - \text{spectateurs du 5}}{12 - 5}$. c) et d) : pars du 5 mai (ou du 12) et ajoute la pente autant de fois qu'il y a de jours d'écart. e) même idée avec la pente des deux derniers jours.

</details>
<details><summary>Indice 3</summary>

a) $\frac{1\,290 + 1\,390}{2} = 1\,340$. b) $\frac{1\,550 - 1\,200}{12 - 5}$. c) $1\,200 + 3 \times$ pente. d) $1\,550 + 1 \times$ pente. e) pente des deux derniers jours $= 1\,550 - 1\,520$, à ajouter à la valeur du 12. f) spectateurs prévus en d) $\times 25 \times 0{,}10$.

</details>

### Ex 1.3 — Compter les connexions d'un réseau en couches ✏️

<details><summary>Indice 1</summary>

Entre deux couches pleines de tailles $n$ et $m$, il y a $n \times m$ connexions (fiche §1.7). Calcule ce produit pour chaque paire de couches voisines, puis additionne les résultats.

</details>
<details><summary>Indice 2</summary>

Les biais : un par neurone des couches **qui calculent** (cachées et sortie), aucun pour les entrées.

</details>
<details><summary>Indice 3</summary>

a) $4 \times 3 + 3 \times 2$. c) $784 \times 128 + 128 + 128 \times 10 + 10$. e) ajoute $128 \times 128 + 128$ au réseau de c.

</details>

### Ex 1.4 — Moins de nombres pour dire la même chose ✏️

<details><summary>Indice 1</summary>

La pluie, toujours nulle, t'apprend-elle quelque chose sur un jour plutôt qu'un autre ? Si tu connais le poids en kg, celui en livres t'apprend-il quelque chose de plus ? Pour la route, pense à la distance entre deux points (0B).

</details>
<details><summary>Indice 2</summary>

d) $\sqrt{240^2 + 320^2}$. e) le vecteur unitaire de la route est $\frac{(600 ;\ 800)}{1\,000}$ ; un point à 750 m est 750 fois ce vecteur. f) produit scalaire de $Q$ avec ce vecteur unitaire. g) Pythagore dans le triangle départ–$Q$–projection de $Q$.

</details>
<details><summary>Indice 3</summary>

e) $750 \times (0{,}6 ;\ 0{,}8)$. f) $250 \times 0{,}6 + 300 \times 0{,}8$. g) $\sqrt{\lVert Q \rVert^2 - f^2}$, avec $\lVert Q \rVert^2 = 250^2 + 300^2$.

</details>

<a id="reflexion"></a>

## 🧮 🗣️ ⚖️ 📄 Réflexion

### Ex 1.5 — Fermi : combien coûte l'étiquetage de MNIST ? 🧮

<details><summary>Indice 1</summary>

Un calcul de Fermi : des hypothèses simples, des nombres ronds, puis des multiplications. Note chaque hypothèse avant de calculer.

</details>
<details><summary>Indice 2</summary>

Temps total = nombre d'images × secondes par image × nombre d'annotateurs par image (+ les arbitrages). Coût = heures × coût horaire (salaire et charges, ou tarif d'une plateforme).

</details>
<details><summary>Indice 3</summary>

1 : avec 2 s par image, $70\,000 \times 2 = 140\,000$ s, soit environ 39 h pour un passage. Pour les autres : 2, deux passages, plus l'arbitrage de 2 % des images ; 3, heures × coût horaire ; 4, divise $14 \times 10^6$ par $7 \times 10^4$ (divise les nombres, soustrais les exposants), puis compare le nombre de réponses possibles pour un chiffre et pour une photo ; 5, demande-toi qui fournit la « bonne réponse » quand on prédit le mot suivant, et combien d'heures coûterait l'étiquetage humain d'un milliard de textes.

</details>

### Ex 1.6 — Le machine learning en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Pars d'un exemple que tout le monde connaît : un filtre anti-spam, les suggestions d'une plateforme de musique, la reconnaissance des visages dans l'application photos.

</details>
<details><summary>Indice 2</summary>

Trois temps : on montre des **exemples** avec la bonne réponse ; le programme ajuste ses réglages pour réduire son **erreur** ; on vérifie qu'il répond bien sur des cas **nouveaux**.

</details>
<details><summary>Indice 3</summary>

Structure possible : « Au lieu d'écrire les règles… On montre à l'ordinateur des milliers d'exemples… Il se trompe, on mesure son erreur, il se corrige… Au bout du compte… On le teste sur des cas nouveaux… »

</details>

### Ex 1.7 — Reconnaissance faciale : utile, risquée, encadrée ⚖️

<details><summary>Indice 1</summary>

Relis l'encadré ⚖️ de la fiche : RGPD (article 9, données biométriques) et AI Act (article 5, pratiques interdites depuis février 2025).

</details>
<details><summary>Indice 2</summary>

Pour chaque projet : qui est filmé ? le sait-il ? peut-il refuser et garder une autre solution ? que se passe-t-il en cas d'erreur (une personne refusée à l'entrée, une personne accusée à tort) ?

</details>
<details><summary>Indice 3</summary>

Le consentement, pour A : les abonnés peuvent donner un consentement explicite, à condition que ce soit un vrai choix, avec le badge comme alternative. Pour B, pose les mêmes questions aux clients du supermarché : sont-ils tous prévenus, peuvent-ils refuser et entrer quand même ? Et que coûte une fausse alerte à la personne visée ? Pour 4, cherche dans le RGPD l'analyse à mener avant un traitement à risque (article 35), demande-toi qui tranche une question juridique, puis cherche ce qui réduit les vols sans identifier personne.

</details>

### Ex 1.8 — Galton (1886) : l'origine du mot « régression » 📄

<details><summary>Indice 1</summary>

Pour les questions 1 et 2, cherche dans le texte le nombre d'enfants adultes et le mot *transmuted* (la « transmutation » des tailles féminines), puis additionne la colonne des totaux du tableau I.

</details>
<details><summary>Indice 2</summary>

3 : écart des parents $= 72 - 68{,}25$ ; écart de l'enfant $= \frac{2}{3}$ de cet écart ; taille $=$ moyenne $+$ écart de l'enfant. 4 : développe $\hat{y} = 68{,}25 + \frac{2}{3}(x - 68{,}25)$.

</details>
<details><summary>Indice 3</summary>

3 : $68{,}25 + \frac{2}{3} \times 3{,}75$, puis multiplie par 2,54 ; même démarche pour 64,5 pouces, qui est sous la moyenne. 4 : en développant, $\hat{y} = \frac{2}{3}x + 68{,}25 \times \left(1 - \frac{2}{3}\right)$ : $w$ est le coefficient de $x$, $b$ la constante, à calculer. 5 : des parents moyens ont aussi des enfants très grands ou très petits.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 1.E1 — Expliquer le machine learning à un recruteur non technique

<details><summary>Indice 1</summary>

Reprends ton texte de 1.6, et ajoute un exemple **métier** (prévoir des ventes, détecter une fraude, trier des documents).

</details>
<details><summary>Indice 2</summary>

Trois idées : on apprend à partir d'exemples plutôt que d'écrire des règles ; on mesure l'erreur et on corrige ; on vérifie sur des données nouvelles.

</details>
<details><summary>Indice 3</summary>

Finis par une limite honnête : un modèle ne vaut que par ses données, et il se trompe parfois ; c'est pour cela qu'on le teste et qu'on le surveille.

</details>

### 1.E2 — Paramètres, hyperparamètres et jeu de test

<details><summary>Indice 1</summary>

Qui fixe la valeur ? Et à quoi sert une mesure faite sur des données déjà vues ?

</details>
<details><summary>Indice 2</summary>

Donne un exemple de chaque (les poids d'un réseau ; le learning rate ou `max_depth`). Pour la triche : pense aux deux erreurs de 1.20.

</details>
<details><summary>Indice 3</summary>

Mentionne le troisième jeu, le jeu de **validation**, qui sert à régler les hyperparamètres sans toucher au test (ch. 8).

</details>

### 1.E3 — Supervisé, non supervisé, auto-supervisé, renforcement : un exemple chacun

<details><summary>Indice 1</summary>

Quatre familles : supervisé, non supervisé, auto-supervisé, renforcement. Pour chacune : ce que reçoit l'algorithme, et un exemple.

</details>
<details><summary>Indice 2</summary>

Supervisé : des labels. Non supervisé : pas de labels. Auto-supervisé : les données fabriquent leurs propres labels. Renforcement : des récompenses.

</details>
<details><summary>Indice 3</summary>

Exemples possibles : détection de spam ; segmentation de clients ; pré-entraînement d'un LLM à prédire le mot suivant ; un programme qui apprend à jouer, ou l'alignement d'un LLM sur des préférences humaines.

</details>

### 1.E4 — Un LLM, c'est quoi ? Réponse en une minute

<details><summary>Indice 1</summary>

Relis l'encadré 🕰️ « Panorama 2026 » : ce qu'est un LLM, comment il est pré-entraîné, puis aligné.

</details>
<details><summary>Indice 2</summary>

Trois étapes : pré-entraînement auto-supervisé (prédire le token suivant) ; ajustement sur des exemples de dialogues ; alignement sur des préférences humaines (RLHF, DPO).

</details>
<details><summary>Indice 3</summary>

Relie-le à ton générateur de 1.24 : même idée (prédire la suite), mais des tokens au lieu de caractères, un contexte immense, des milliards de paramètres. Finis par une limite : il peut affirmer des choses fausses avec aplomb.

</details>

<a id="notebook"></a>

## Notebook, parties A à D

### Ex 1.9 — Penguins : échantillons, features et labels 📦

<details><summary>Indice 1</summary>

`len(df)` compte les lignes, `df.shape` donne (lignes, colonnes), `df["species"].value_counts()` compte les espèces.

</details>
<details><summary>Indice 2</summary>

b) toutes les colonnes sauf le label : `df.shape[1] - 1`. d) `len(df.dropna())`. e) `df.loc[0, MEASURES]` sélectionne la ligne 0 et quatre colonnes.

</details>
<details><summary>Indice 3</summary>

c) `penguins_all["species"].value_counts().tolist()` (déjà trié du plus fréquent au moins fréquent). e) ajoute `.tolist()`. f) le nom exact de la colonne de la masse.

</details>

### Ex 1.10 — MNIST : une image, 784 nombres 📦

<details><summary>Indice 1</summary>

`X_mnist.shape` ; une image est `X_mnist[0]`, un tableau 28 × 28 ; `X_mnist.min()` et `.max()`.

</details>
<details><summary>Indice 2</summary>

e) `X_mnist[0] > 0` est un masque de booléens ; `.sum()` compte les `True`. f) `np.bincount(y_mnist)` compte chaque chiffre de 0 à 9.

</details>
<details><summary>Indice 3</summary>

b) `28 * 28`. f) `int(np.bincount(y_mnist).argmax())`.

</details>

### Ex 1.11 — Holmes et Verne : le texte devient des nombres 📦

<details><summary>Indice 1</summary>

`len(texte)` compte les caractères ; `ord(c)` donne le code d'un caractère ; `set(texte)` garde chaque caractère une seule fois.

</details>
<details><summary>Indice 2</summary>

`letter_freq` : un `Counter` des caractères de `text.lower()` gardés si `"a" <= c <= "z"` ; divise chaque compte par le total des lettres gardées ; parcours les lettres avec `for k in range(ord("a"), ord("z") + 1)` et `chr(k)`.

</details>
<details><summary>Indice 3</summary>

f) `half = len(holmes) // 2`, puis `np.linalg.norm(letter_freq(holmes[:half]) - letter_freq(holmes[half:]))`, et de même pour Holmes contre Verne.

</details>

### Ex 1.12 — Taches solaires : tracer, lisser, repérer le cycle 📦

<details><summary>Indice 1</summary>

`idxmax()` renvoie le **label de ligne** (l'index) du maximum ; `sun.loc[index, "year"]` donne ensuite l'année.

</details>
<details><summary>Indice 2</summary>

c) `smooth.isna().sum()`. d) comme b, mais avec `smooth.idxmax()`. f) `sun["decimal_year"].to_numpy()[peaks]` donne les années des sommets ; `np.diff` les écarts entre sommets successifs.

</details>
<details><summary>Indice 3</summary>

e) `peaks = local_maxima(smooth.to_numpy(), 60)` et `n_peaks = len(peaks)`. f) `np.diff(...).mean()`.

</details>

### Ex 1.13 — Lire les data cards des quatre fils rouges 🛠️

<details><summary>Indice 1</summary>

Chaque fiche a les mêmes rubriques : cherche « Licence » et « Biais et limites ».

</details>
<details><summary>Indice 2</summary>

Pour l'usage commercial, cherche le sigle **NC** (*non commercial*) dans les licences Creative Commons. Pour Penguins, lis ce qui est dit des îles.

</details>
<details><summary>Indice 3</summary>

1, en modèle pour Penguins : licence CC0 (domaine public), donc oui, même dans un produit vendu. Pour les quatre autres, lis la ligne « Licence » de chaque fiche : NC (*non commercial*) interdit la vente ; BY (citer les auteurs) et SA (partager un dérivé sous la même licence) posent des conditions sans l'interdire. Pour 2, lis dans « Biais et limites » ce qui est dit des îles. Pour 3, cherche le mot qui résume le statut de MNIST, puis demande-toi ce que prouve un bon score sur un tel dataset. Pour 4, compare la taille, le siècle et le nombre d'auteurs de ces textes à ceux d'un corpus de LLM. Pour 5, que peut-il arriver à un mois encore provisoire ?

</details>

### Ex 1.14 — Mémoriser n'est pas apprendre 🔮

<details><summary>Indice 1</summary>

Relis la fiche §1.2.2 (mémoriser ou généraliser). Pour chaque question, demande-toi si le mémoriseur a déjà vu, dans sa table, les manchots sur lesquels on le teste.

</details>
<details><summary>Indice 2</summary>

b) Deux manchots différents ont-ils exactement les mêmes 4 mesures au dixième de millimètre près ? Si non, que répond le mémoriseur pour les manchots du test ?

</details>
<details><summary>Indice 3</summary>

Pour tous les manchots inconnus, il répond la même espèce ; son accuracy est donc la part de cette espèce dans le jeu de test. c) `sum(tuple(row) in memorizer.table_ for row in X_test.to_numpy())`.

</details>

### Ex 1.15 — Un système expert pour les manchots 🔨

<details><summary>Indice 1</summary>

Trois règles dans l'ordre : `if … return` ; puis un second `if … return` ; puis un `return` final.

</details>
<details><summary>Indice 2</summary>

a) `accuracy(y_train, predict_with(expert_rule, X_train))`. c) Dans le tableau croisé, chaque **ligne** est une vraie espèce : compare, pour chaque ligne, les manchots hors de la diagonale au total de la ligne.

</details>
<details><summary>Indice 3</summary>

La première règle : `if body_mass > 4700: return "Gentoo"` (« plus de » : strictement plus grand) ; écris la deuxième de la même façon, avec son seuil, puis le `return` final. c) `errors = predict_with(expert_rule, X_train) != y_train`, puis le taux d'erreur de chaque vraie espèce, `errors.groupby(y_train).mean()` : garde l'espèce du plus grand taux.

</details>

### Ex 1.16 — La boucle d'entraînement à la main 🔨

<details><summary>Indice 1</summary>

Écris les fonctions dans l'ordre : `predict_line` (une ligne), `mse` (la moyenne des carrés des écarts), `train_step` (prédire, erreur, deux corrections), puis `train_line` (deux boucles imbriquées : les epochs, puis les échantillons).

</details>
<details><summary>Indice 2</summary>

`train_step` : calcule l'erreur **une seule fois**, avec les anciens $w$ et $b$, puis applique les deux corrections de l'énoncé et renvoie le couple corrigé. `train_line` : la liste `losses` commence par la loss **avant** tout entraînement, puis reçoit une valeur après chaque epoch (21 valeurs pour 20 epochs).

</details>
<details><summary>Indice 3</summary>

`train_step` : `error = y_i - predict_line(w, b, x_i)`, puis les deux corrections utilisent cette même `error`. Squelette de `train_line` :

```python
def train_line(x, y, eta, n_epochs):
    w, b = 0.0, 0.0
    losses = [mse(w, b, x, y)]            # the loss before any training
    # for each epoch:
    #     for x_i, y_i in zip(x, y):      the samples, in order
    #         one correction with train_step (it returns the new w, b)
    #     append the loss of the current line to losses
    return w, b, losses
```

</details>

### Ex 1.17 — Learning rate : trop prudent, trop pressé 🔬

<details><summary>Indice 1</summary>

Un dictionnaire en compréhension : `{key: value for eta in ETAS_17}`.

</details>
<details><summary>Indice 2</summary>

La valeur est la liste des losses, c'est-à-dire le troisième élément de ce que renvoie `train_line(x_line, y_line, eta, 20)`.

</details>
<details><summary>Indice 3</summary>

Squelette : `return {eta: ... for eta in ETAS_17}`, où `...` est la liste des losses de `train_line(x_line, y_line, eta, 20)` ; la fonction renvoie le tuple `(w, b, losses)` : prends-en le bon élément. Pour lire les courbes : sur le graphique de gauche, celle qui monte ; sur celui de droite, la plus haute à la fin (trop lente) et celle qui descend le plus vite au début ; le tableau affiché donne les valeurs exactes.

</details>

### Ex 1.18 — Un arbre de décision apprend les règles à ta place 📦

<details><summary>Indice 1</summary>

`tree.score(X, y)` donne directement l'accuracy ; `rules` se lit de haut en bas, la première ligne est la première question.

</details>
<details><summary>Indice 2</summary>

d) le nombre qui suit `<=` sur la première ligne. e) (accuracy de l'arbre − accuracy de la biologiste) × nombre de manchots de test.

</details>
<details><summary>Indice 3</summary>

e) `round((tree_test_acc - expert_test_acc) * len(test))`. On peut aussi lire c et d dans `tree.tree_.feature[0]` et `tree.tree_.threshold[0]`.

</details>

### Ex 1.19 — Un manchot d'une espèce jamais vue 🔮

<details><summary>Indice 1</summary>

Suis les règles de `rules` comme un chemin : à chaque question, va dans la branche qui correspond aux mesures du visiteur.

</details>
<details><summary>Indice 2</summary>

Pour l'empereur : sa nageoire passe-t-elle le seuil de la première question ? Puis, dans cette branche, quelle est la seconde question, et que répondent ses mesures ? Un arbre a-t-il une feuille « inconnu » ?

</details>
<details><summary>Indice 3</summary>

L'empereur passe dans la branche des longues nageoires, où la seconde question porte sur l'épaisseur du bec (22 mm). c) `float(tree.predict_proba(visitors)[0].max())`.

</details>

### Ex 1.20 — Le score trop beau pour être vrai 🐛

<details><summary>Indice 1</summary>

Deux questions : une feature contient-elle la réponse ? Sur quels manchots le modèle a-t-il été entraîné ?

</details>
<details><summary>Indice 2</summary>

Pour chaque colonne de `FEATURES_20`, demande-toi si tu la connaîtrais pour un manchot inconnu, avant de savoir son espèce. Puis regarde quelles lignes du tableau reçoit `fit`.

</details>
<details><summary>Indice 3</summary>

Première erreur, en modèle : `species_code` n'est que l'espèce écrite en chiffres ; une feature qui contient la réponse est une fuite du label, et `features_20` ne doit garder que les mesures. Pour la seconde, compte les lignes que reçoit `model_20.fit` et demande-toi si les 100 manchots de `test_coded` en font partie. Squelette de la correction :

```python
def train_and_evaluate_20():
    model = DecisionTreeClassifier(random_state=0)
    model.fit(...)                       # the training set only, the columns features_20
    test_accuracy = model.score(...)     # the test set, the same columns
    return model, test_accuracy
```

</details>

### Ex 1.21 — Regrouper les manchots sans leurs labels 📦

<details><summary>Indice 1</summary>

`pd.crosstab(groups, labels)` donne une ligne par groupe et une colonne par espèce.

</details>
<details><summary>Indice 2</summary>

Pureté : `.max(axis=1)` garde le plus grand compte de chaque ligne ; additionne-les et divise par le nombre de manchots. `X_scaled` : les opérations de pandas se font colonne par colonne.

</details>
<details><summary>Indice 3</summary>

```python
def purity(groups, labels):
    table = pd.crosstab(np.asarray(groups), np.asarray(labels))   # one row per group
    # the largest count of each ROW, summed, then divided by the number of samples
    return ...
```

`X_scaled` : la formule de l'énoncé, avec `X = penguins[MEASURES]` ; `X.mean()` et `X.std()` donnent une valeur par colonne, et pandas fait le calcul colonne par colonne.

</details>

### Ex 1.22 — L'agent cuisinier : apprendre par la récompense 🔬

<details><summary>Indice 1</summary>

Garde deux tableaux de 5 nombres : combien de fois chaque recette a été cuisinée, et combien de fois elle a été mangée. Le taux d'une recette est le second divisé par le premier.

</details>
<details><summary>Indice 2</summary>

Chaque soir : choisir la recette (trois cas : les 5 premiers soirs ; exploration ; exploitation), puis `reward = child_eats(recipe, rng)`, puis mettre à jour les deux tableaux, le total et la liste des recettes.

</details>
<details><summary>Indice 3</summary>

```python
if evening < 5:
    recipe = evening
elif rng.random() < explore:
    recipe = int(rng.integers(5))
else:
    recipe = int(np.argmax(n_eaten_by_recipe / n_tried))
```

</details>

### Ex 1.23 — Un réseau de neurones en boîte noire sur MNIST 📦

<details><summary>Indice 1</summary>

`mlp.coefs_` est une liste de matrices de poids, `mlp.intercepts_` une liste de vecteurs de biais : regarde leurs formes (`.shape`).

</details>
<details><summary>Indice 2</summary>

a) additionne les `.size` des éléments des deux listes. c) `mlp.score(X_te, y_te)` ; les erreurs : `predictions_23 != y_te`.

</details>
<details><summary>Indice 3</summary>

`sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)` ; `int((predictions_23 != y_te).sum())`.

</details>

### Ex 1.24 — Fabriquer du faux Holmes et du faux Verne 🔨

<details><summary>Indice 1</summary>

Les paires de caractères consécutifs sont `zip(text, text[1:])`. Un `defaultdict(Counter)` crée un `Counter` vide la première fois qu'on utilise une clé.

</details>
<details><summary>Indice 2</summary>

`generate` : une liste `characters = [start]` ; à chaque pas, le caractère courant est `current = characters[-1]` ; les candidats `followers = sorted(counts[current])` ; leurs poids dans un array NumPy `weights` ; `rng.choice(followers, p=weights / weights.sum())`.

</details>
<details><summary>Indice 3</summary>

`for current, following in zip(text, text[1:]): counts[current][following] += 1`. Dans `generate`, fais `length - 1` tirages, puis `"".join(characters)` (convertis le tirage avec `str(...)`).

</details>

### Ex 1.25 — Battre l'expert : 95 % avec tes propres règles 🏆

<details><summary>Indice 1</summary>

Sur le graphique « nageoire contre épaisseur du bec », une espèce se détache nettement : sépare-la d'abord.

</details>
<details><summary>Indice 2</summary>

Gentoo : longue nageoire **et** bec fin. Parmi les autres, sur le graphique « épaisseur contre longueur du bec », une seule mesure sépare assez bien Chinstrap et Adélie.

</details>
<details><summary>Indice 3</summary>

Essaie une règle de la forme `if flipper_length > ... and bill_depth < ...: return "Gentoo"`, puis `if bill_length > ...: return "Chinstrap"`, puis `return "Adelie"`, en réglant les seuils sur l'entraînement.

</details>
