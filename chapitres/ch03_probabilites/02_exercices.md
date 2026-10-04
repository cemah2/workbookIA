# 3 · Probabilités et mesure de la qualité — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch03_probabilites/06_mes_reponses.md` (créée par `python tools/start_chapter.py 3`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses des exercices ✏️ se vérifient dans la **partie 0** du notebook (`wb.check`) ; les quiz, les rappels, les exercices ∂ 🗣️ ⚖️ 📄 et les questions d'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 ou 4 minutes chacun. Réponds vite, puis vérifie dans `05_solutions.md`.

### 3.Q1 — Probabilité, pourcentage, degré de confiance 🧠 ⏱️ 3 min
*Fiche §3.1 · livre §3.1 · parcours R*

1. Écris 0,35 en pourcentage, et 7 % comme une probabilité.
2. Vrai ou faux : un événement « très probable » peut avoir une probabilité de 1,2.
3. Pour une variable continue (ch. 2), un événement de probabilité 0 peut-il se produire ?
4. Tu dis de 200 fruits, un par un : « je suis sûr à 80 % qu'il est mûr ». Que faudrait-il observer pour que cette confiance soit méritée ?
5. Un classifieur annonce `predict_proba = 0.97` pour un e-mail. Est-ce forcément la probabilité que ce soit un spam ?
6. Pourquoi faut-il connaître un peu de probabilités pour bien utiliser les bibliothèques de machine learning ?

### 3.Q2 — Fléchettes : pourquoi chaque point du mur doit être aussi probable 🧠 ⏱️ 3 min
*Fiche §3.2, §3.3 · livre §3.2, §3.3 · parcours R*

1. Le mur mesure 4 m² et une tache 1 m². Quelle est la probabilité qu'une fléchette touche la tache ?
2. Pourquoi l'hypothèse « chaque point du mur a la même chance d'être touché » est-elle indispensable pour dire « probabilité = rapport d'aires » ?
3. Un lanceur vise le centre du mur et touche plus souvent le milieu. Le raisonnement tient-il encore ?
4. On double la largeur et la hauteur du mur et de toutes les taches. Les probabilités changent-elles ?
5. Des taches qui ne se chevauchent pas couvrent 30 % du mur. Quelle est la probabilité de toucher le fond ?
6. Sur 500 fléchettes, 130 tombent dans la tache A. Estime $P(A)$. Pourquoi n'est-ce qu'une estimation ?

### 3.Q3 — P(A|B) ou P(B|A) ? 🧠 ⏱️ 3 min
*Fiche §3.4 · livre §3.4 · parcours R*

Pour les phrases 1 à 3, écris la probabilité conditionnelle correspondante sous la forme $P(\ldots \mid \ldots)$.
1. « La probabilité qu'une personne ait soif, sachant qu'elle boit de l'eau. »
2. « Parmi les malades, 99 % ont un test positif. »
3. « Parmi les tests positifs, un tiers seulement concernent des malades. »
4. Chez les manchots, laquelle est la plus proche de 1 : $P(\text{nageoire} > 210\ \text{mm} \mid \text{Gentoo})$ ou $P(\text{Gentoo} \mid \text{nageoire} > 210\ \text{mm})$ ? (Souviens-toi du ch. 2 : les Gentoo ont les nageoires les plus longues, autour de 217 mm en moyenne, mais certaines mesurent 210 mm ou moins.)
5. Si la tache A contient entièrement la tache B, que vaut $P(A \mid B)$ ? Et si A et B ne se touchent pas ?
6. Vrai ou faux : si $P(A) = P(B)$, alors $P(A \mid B) = P(B \mid A)$.

### 3.Q4 — Jointe = conditionnelle × simple 🧠 ⏱️ 3 min
*Fiche §3.5 · livre §3.5 · parcours R, M*

La tache B couvre 40 % du mur, et la tache A couvre le quart de la tache B (en plus d'autres parties du mur). On lance 200 fléchettes.
1. Combien de fléchettes attend-on dans B ? Dans A et B à la fois ?
2. Que vaut $P(A, B)$ ?
3. Écris la règle du produit dans ses deux versions.
4. Vrai ou faux : $P(A, B) = P(B, A)$.
5. Vrai ou faux : $P(A, B)$ ne peut pas dépasser $P(A)$.
6. Deux événements indépendants ont pour probabilités 0,4 et 0,5. Que vaut leur probabilité jointe ?

### 3.Q5 — D'où vient le mot « marginale » 🧠 ⏱️ 3 min
*Fiche §3.6 · livre §3.6*

1. D'où vient le mot « marginale », d'après le livre ?
2. Dans une table de contingence de deux variables, où lit-on les probabilités marginales ?
3. Que vaut la somme de toutes les probabilités jointes d'une table ? Et la somme des probabilités marginales d'une même variable ?
4. Donne une formule qui relie $P(A)$ aux probabilités jointes $P(A, B = b)$.
5. Vrai ou faux : connaître les totaux des lignes et des colonnes suffit pour reconstruire toute la table.
6. Quelle commande pandas donne les probabilités marginales de la colonne `species` des manchots ?

### 3.Q6 — Vérité terrain et prédiction 🧠 ⏱️ 3 min
*Fiche §3.7, §3.7.1 · livre §3.7, §3.7.1 · parcours R*

1. Qu'est-ce que la vérité terrain (*ground truth*) d'un échantillon ? Qui la fournit ?
2. Que représente la frontière de décision d'un classifieur à deux classes ?
3. Pour un filtre anti-spam, quelle classe appelle-t-on d'habitude « positive » ? Est-ce un jugement de valeur ?
4. Vrai ou faux : si l'on échange les rôles de « positif » et de « négatif », la precision et le recall ne changent pas.
5. Sur quelles données faut-il mesurer ces scores pour qu'ils disent quelque chose des données futures (ch. 1) ?
6. La vérité terrain peut-elle être fausse ? Donne un exemple.

### 3.Q7 — Lire une matrice de confusion (et vérifier ses axes) 🧠 ⏱️ 3 min
*Fiche §3.7.2 · livre §3.7.2 · parcours R*

scikit-learn affiche la matrice de confusion d'un test médical (labels 0 = sain, 1 = malade) : `[[50, 5], [10, 35]]`.
1. Combien de TP, FP, FN et TN ?
2. Combien de malades dans les données ? Combien de prédictions « malade » ?
3. Un collègue lit ce tableau comme dans le livre, avec TP en haut à gauche. Quelles valeurs croit-il lire pour TP et TN ?
4. Où se trouvent les bonnes réponses dans une matrice de confusion, quelle que soit sa disposition ?
5. Que vaut la somme de toutes les cases ?
6. Pourquoi faut-il toujours lire les étiquettes des axes ?

### 3.Q8 — Faux positif ou faux négatif : lequel coûte le plus ? 🧠 ⏱️ 3 min
*Fiche §3.7.3, §3.7.4 · livre §3.7.3, §3.7.4 · parcours R*

Pour chaque situation, dis quelle erreur est la plus grave, et s'il faut surtout surveiller la precision ou le recall.
1. Détecter les fraudes à la carte bancaire (positif = fraude).
2. Filtre anti-spam (positif = spam).
3. Retirer de la chaîne de montage toute figurine d'un personnage qu'on n'a plus le droit de vendre (livre §3.7.4, positif = figurine interdite).
4. Ne laisser partir que les figurines aux yeux bien peints (livre §3.7.4, positif = « yeux présents »).
5. Premier test de dépistage d'une maladie grave, suivi d'un examen plus précis pour les positifs.
6. Qu'est-ce qui décide, au fond, qu'une erreur est « acceptable » ?

### 3.Q9 — L'accuracy face aux classes déséquilibrées 🧠 ⏱️ 3 min
*Fiche §3.7.5 · livre §3.7.5 · parcours R*

1. Sur 1 000 transactions, 10 sont frauduleuses. Un modèle répond toujours « pas de fraude ». Quelle est son accuracy ? Son recall ?
2. Pourquoi ce chiffre est-il trompeur ?
3. Quelle référence faut-il toujours donner à côté d'une accuracy (ch. 1) ?
4. Une accuracy de 0,7 mesurée sur 20 exemples et une autre sur 20 000 exemples méritent-elles la même confiance (ch. 2) ?
5. Vrai ou faux : l'accuracy compte de la même façon une erreur sur un malade et une erreur sur une personne saine.
6. Quelle mesure du tableau de la fiche donne le même poids aux deux classes, quelle que soit leur taille ?

### 3.Q10 — Precision ou recall : le moteur de recherche du wiki 🧠 ⏱️ 4 min
*Fiche §3.7.6 à §3.7.8 · livre §3.7.6 à §3.7.8 · parcours R*

Le wiki d'une entreprise compte 500 pages, dont 40 parlent vraiment de dressage de chiens. Une recherche « dressage de chiens » renvoie 25 pages, dont 20 pertinentes.
1. Combien de TP, FP, FN et TN ?
2. Quelles sont la precision et le recall de cette recherche ?
3. Quelle est son accuracy ? Pourquoi ce chiffre est-il peu utile ici ?
4. D'où vient le nom « recall » ?
5. Le moteur renvoie maintenant 80 pages, parmi lesquelles les 40 pages pertinentes. Precision ? Recall ?
6. Un avocat ne doit manquer aucun document utile d'un dossier. Quelle mesure privilégie-t-il ? Et le moteur de recherche grand public, pour sa première page de résultats ?

### 3.Q11 — Tricher sur une seule mesure 🧠 ⏱️ 3 min
*Fiche §3.7.9, §3.7.10 · livre §3.7.9, §3.7.10 · parcours R*

On reprend 20 points, dont 10 positifs.
1. Comment obtenir un recall de 1 sans rien apprendre ? Que vaut alors la precision ?
2. Comment obtenir une precision de 1 (ou presque) ? Que vaut alors le recall, au pire ?
3. Pourquoi faut-il toujours annoncer au moins deux mesures, ou une mesure qui les combine ?
4. Vrai ou faux : si l'on monte le seuil de décision d'un classifieur à scores, le recall ne peut pas augmenter.
5. Vrai ou faux : si l'on monte le seuil, la precision augmente toujours.
6. Quelle mesure vaut $1 - \text{recall}$ ? Et $1 - \text{spécificité}$ ?

### 3.Q12 — F1 et test « fiable à 99 % » 🧠 ⏱️ 4 min
*Fiche §3.7.11, §3.8 · livre §3.7.11, §3.8 · parcours R*

1. Precision 0,9 et recall 0,1 : que vaut leur moyenne ordinaire ? Et le F1 ?
2. Vrai ou faux : le F1 est toujours compris entre la precision et le recall.
3. Le F1 utilise-t-il le nombre de vrais négatifs ?
4. « Ce test détecte 99 % des malades. » Quelle mesure est donnée ? Que faut-il savoir en plus pour dire à une personne positive si elle est probablement malade ?
5. Pourquoi une maladie rare donne-t-elle beaucoup de faux positifs, même avec un bon test ?
6. Que doit-on faire après un test de dépistage positif, d'après le livre ?

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 3.R1 — Ch. 2 : deux cartes rouges de suite, avec ou sans remise 🔁 ★ ⏱️ 5 min
*Ch. 2 (§2.5, tirages) · parcours R, M*

Un jeu de 32 cartes contient 16 cartes rouges. On tire deux cartes.
1. Quelle est la probabilité que la première carte soit rouge ?
2. Avec remise (on remet la première carte et on mélange), quelle est la probabilité de tirer deux rouges ?
3. Sans remise, quelle est la probabilité que la seconde soit rouge, sachant que la première l'était ? Déduis-en la probabilité de tirer deux rouges (3 décimales).
4. Dans lequel des deux cas les deux tirages sont-ils indépendants ?
5. Quelle instruction NumPy tire 5 cartes distinctes parmi 32 (ch. 2) ?

### 3.R2 — 0B : ensembles, intersection, union, complémentaire 🔁 ★ ⏱️ 5 min
*0B (ensembles et probabilités) · parcours R, M*

Dans un lycée de 100 élèves, 40 font du sport (S), 30 de la musique (M), et 12 font les deux.
1. Combien d'élèves sont dans $S \cap M$ ? Dans $S \cup M$ ?
2. Combien ne font ni sport ni musique ?
3. On tire un élève au hasard : que valent $P(S)$, $P(S \cap M)$ et $P(S \cup M)$ ?
4. Que vaut $P(\text{non } S)$ ?
5. Faire du sport et faire de la musique sont-ils indépendants dans ce lycée ?
6. Dans la formule $P(S \cup M) = P(S) + P(M) - P(S \cap M)$, pourquoi soustrait-on $P(S \cap M)$ ?

### 3.R3 — 0A : pandas, `value_counts(normalize=True)` et `groupby` 🔁 ★ ⏱️ 5 min
*0A (pandas) · parcours R, C*

`df` contient les manchots (colonnes `species`, `island`, `sex`, `body_mass_g`…). Sans exécuter de code :
1. Que renvoie `df["species"].value_counts()` ? Et avec `normalize=True` ?
2. Que vaut la somme des valeurs de `df["species"].value_counts(normalize=True)` ?
3. Que renvoie `df.groupby("species")["body_mass_g"].mean()` ?
4. Comment obtenir, pour chaque espèce, la proportion de mâles ?
5. Comment ne garder que les manchots de l'île Biscoe ?

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats ✏️ dans la partie 0 du notebook. Garde les valeurs exactes (des fractions) pendant tes calculs et **n'arrondis qu'à la fin** : une moyenne de valeurs déjà arrondies peut changer la dernière décimale.

### Ex 3.1 — Fléchettes et aires : probabilités simples et conditionnelles ✏️ ★ ⏱️ 15 min
**Objectif :** calculer des probabilités simples, jointes et conditionnelles comme des rapports d'aires, puis les estimer par des comptages.
**Prérequis :** fiche §3.2 à §3.5 · 0B (aires, ensembles) · **Parcours :** M

Un mur rectangulaire mesure 4 m de large et 2,5 m de haut. On repère un point du mur par ses coordonnées $(x, y)$ en mètres, l'origine dans le coin en bas à gauche. Deux taches rectangulaires y sont peintes :
- la tache A occupe $0{,}5 \le x \le 2{,}5$ et $0{,}5 \le y \le 2$ ;
- la tache B occupe $1{,}5 \le x \le 3{,}5$ et $1 \le y \le 2$.

Chaque point du mur a la même chance d'être touché.

a) $P(A)$.
b) $P(B)$.
c) $P(A, B)$ (dessine d'abord la partie commune).
d) $P(A \mid B)$.
e) $P(B \mid A)$ (3 décimales).
f) La probabilité de ne toucher ni A ni B.
g) On lance 400 fléchettes. Combien en attend-on dans B ?
h) Et dans A et B à la fois ?
i) Sur ses 400 fléchettes, une élève en compte 76 dans B, dont 41 aussi dans A. Quelle est son estimation de $P(A \mid B)$ (3 décimales) ?
j) Les événements A et B sont-ils indépendants ? Réponds `True` ou `False`.
k) Pourquoi l'estimation de i ne donne-t-elle pas exactement la valeur de d ? Comment l'améliorer ? (réponds dans ta copie)

### Ex 3.2 — Les 20 points : matrice de confusion et quatre mesures ✏️ ★ ⏱️ 15 min
**Objectif :** construire une matrice de confusion à partir de prédictions, puis en tirer accuracy, precision, recall et F1.
**Prérequis :** fiche §3.7.2 à §3.7.11 · **Parcours :** R, M

Un classifieur a jugé 20 points ; 1 veut dire « positif », 0 « négatif ». Voici la vérité terrain et la prédiction de chaque point, dans l'ordre :

| point | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **vérité** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| **prédiction** | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |

a) Le nombre de vrais positifs TP.
b) Le nombre de faux positifs FP.
c) Le nombre de faux négatifs FN.
d) Le nombre de vrais négatifs TN.
e) La matrice de confusion telle que la renvoie scikit-learn (labels 0 puis 1, vérité en lignes), sous la forme d'une liste de deux lignes, par exemple `[[1, 2], [3, 4]]`.
f) L'accuracy (2 décimales).
g) La precision (3 décimales).
h) Le recall (2 décimales).
i) Le F1 (3 décimales).
j) Si les positifs sont des transactions frauduleuses, laquelle des deux erreurs de ce classifieur coûte le plus cher, et quelle mesure surveilles-tu ? (réponds dans ta copie)

### Ex 3.3 — Le glacier : jointes, marginales et conditionnelles ✏️ ★★ ⏱️ 20 min
**Objectif :** lire probabilités jointes, marginales et conditionnelles dans une table de contingence.
**Prérequis :** Ex 3.1 · fiche §3.4 à §3.6 · livre §3.5, §3.6 · **Parcours :** M

Hier, un glacier qui ne vend que de la vanille et du chocolat, en cornet ou en pot, a servi 150 clients :

| | cornet | pot | total |
|---|---|---|---|
| **vanille** | 42 | 18 | 60 |
| **chocolat** | 38 | 52 | 90 |
| **total** | 80 | 70 | 150 |

On tire un client au hasard. On note V l'événement « il a pris de la vanille » et C l'événement « il a pris un cornet ».

a) $P(V)$.
b) $P(C)$ (3 décimales).
c) $P(V, C)$.
d) $P(V \mid C)$ (3 décimales).
e) $P(C \mid V)$.
f) $P(\text{chocolat} \mid \text{pot})$ (3 décimales).
g) Le parfum et le contenant sont-ils indépendants ? Réponds `True` ou `False`.
h) Demain, 300 clients viendront, avec les mêmes habitudes. Combien de cornets à la vanille le glacier doit-il prévoir ?
i) Explique en une phrase, sur cet exemple, la différence entre $P(V \mid C)$ et $P(C \mid V)$. (réponds dans ta copie)

### Ex 3.4 — Règle du produit et formule des probabilités totales ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer les règles de calcul utilisées tout le chapitre (et au ch. 4).
**Prérequis :** Ex 3.3 · fiche §3.4 à §3.6 · 0B (ensembles) · **Parcours :** M

On part de la définition $P(A \mid B) = \frac{P(A, B)}{P(B)}$ quand $P(B) > 0$.
1. Montre que $P(A, B) = P(A \mid B)\,P(B)$, puis que $P(A, B) = P(B \mid A)\,P(A)$ quand $P(A) > 0$.
2. Déduis-en que $P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$. À quelle condition sur $P(A)$ et $P(B)$ a-t-on $P(A \mid B) = P(B \mid A)$ ? (Cette formule est la règle de Bayes du ch. 4.)
3. B et « non B » découpent le mur en deux régions qui ne se chevauchent pas. Explique avec les aires pourquoi $P(A) = P(A, B) + P(A, \text{non } B)$, puis montre que $P(A) = P(A \mid B)\,P(B) + P(A \mid \text{non } B)\,P(\text{non } B)$ quand $0 < P(B) < 1$ : c'est la **formule des probabilités totales**.
4. Généralise la question 3 : si une variable $B$ prend les valeurs $b_1, \ldots, b_K$ (des régions sans chevauchement qui couvrent tout le mur, chacune de probabilité non nulle), démontre la formule générale de la fiche (§3.6). Vérifie-la ensuite sur la ligne « femelle » de la table des manchots de la fiche.
5. Montre que si A et B sont indépendants, c'est-à-dire si $P(A, B) = P(A)\,P(B)$, alors $P(A \mid B) = P(A)$ (avec $P(B) > 0$).
6. Vérifie les formules des questions 1 et 3 sur les chiffres du glacier (Ex 3.3), avec A = V et B = C.

### Ex 3.5 — Toutes les mesures du tableau récapitulatif ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer et interpréter toutes les mesures d'une matrice de confusion binaire.
**Prérequis :** Ex 3.2 · fiche §3.7.9 · livre §3.7.9 (figures 3.32 à 3.34) · **Parcours :** M

Un détecteur de fraude est évalué sur 1 000 transactions (positif = fraude) : TP = 40, FN = 10, FP = 60, TN = 890. Donne chaque mesure avec 3 décimales.

a) La prévalence.
b) L'accuracy.
c) La precision.
d) Le recall.
e) La spécificité.
f) La NPV (valeur prédictive négative).
g) Le FPR (taux de faux positifs).
h) Le FNR (taux de faux négatifs).
i) Le FDR (taux de fausses découvertes).
j) Le FOR (taux de fausses omissions).
k) Le F1.
l) La balanced accuracy.
m) Le MCC (coefficient de Matthews).
n) Vérifie sur tes valeurs les quatre paires de la fiche qui s'additionnent à 1. Lesquelles de tes treize mesures changeraient si l'on testait le même détecteur sur des transactions deux fois plus souvent frauduleuses ? (réponds dans ta copie)

### Ex 3.6 — Trois espèces : moyennes macro, micro et pondérée ✏️ ★★ ⏱️ 25 min
**Objectif :** calculer precision, recall et F1 par classe, puis les combiner en moyennes macro, micro et pondérée.
**Prérequis :** Ex 3.2 · fiche §3.7.9 (plusieurs classes), §3.7.11 · **Fil rouge :** Penguins · **Parcours :** M

Un classifieur de manchots est testé sur 100 manchots dont on connaît l'espèce. Sa matrice de confusion (vérité en lignes, prédiction en colonnes) :

| vérité \ prédiction | Adélie | Chinstrap | Gentoo |
|---|---|---|---|
| **Adélie** | 41 | 3 | 1 |
| **Chinstrap** | 7 | 13 | 0 |
| **Gentoo** | 0 | 2 | 33 |

Donne toutes les valeurs avec 3 décimales ; pour les listes, les trois valeurs dans l'ordre Adélie, Chinstrap, Gentoo. Pour les moyennes (e à h), repars des fractions exactes, pas des valeurs arrondies de b à d.

a) Le *support* de chaque espèce (le nombre de manchots de cette espèce), une liste.
b) La precision de chaque espèce, une liste.
c) Le recall de chaque espèce, une liste.
d) Le F1 de chaque espèce, une liste.
e) La precision macro.
f) Le F1 macro.
g) Le F1 pondéré (*weighted*).
h) Le F1 micro.
i) Pourquoi le F1 micro est-il égal à l'accuracy ? Quelle espèce le classifieur réussit-il le moins bien, et quelle moyenne le montre le mieux ? (réponds dans ta copie)

### Ex 3.7 — Le test « fiable à 99 % » dans une ville à 1 % de malades ✏️ ★★ ⏱️ 25 min
**Objectif :** mesurer l'effet de la prévalence sur la precision d'un test, et corriger deux phrases du livre.
**Prérequis :** Ex 3.2 · fiche §3.7.9, §3.8 · livre §3.8 · **Parcours :** R, M

**Partie 1.** Une ville compte 50 000 habitants, dont 1 % sont malades. Le test est « fiable à 99 % » : sa sensibilité (recall) et sa spécificité valent toutes deux 0,99.
a) Le nombre de malades.
b) Le nombre de vrais positifs TP attendus.
c) Le nombre de faux négatifs FN.
d) Le nombre de faux positifs FP.
e) Le nombre de vrais négatifs TN.
f) La precision du test, $P(\text{malade} \mid \text{positif})$ (2 décimales).
g) La NPV, $P(\text{sain} \mid \text{négatif})$ (4 décimales).
h) L'accuracy (2 décimales).
i) Le même test dans une autre ville de 50 000 habitants, où la prévalence vaut 0,2 % : sa precision (3 décimales) ?
j) Et dans une ville où la prévalence vaut 10 % (3 décimales) ?

**Partie 2 : les chiffres du livre.** Le livre (§3.8) prend un test de sensibilité 0,99 et de spécificité 0,98, dans une ville de 10 000 habitants dont 1 % sont malades. Construis sa matrice de confusion, puis calcule avec ses **comptages** :
k) la spécificité, $\frac{TN}{TN + FP}$ (2 décimales) ;
l) la NPV, $\frac{TN}{TN + FN}$ (4 décimales).

m) Le livre affirme d'abord qu'un taux de vrais négatifs de 0,98 signifie que 98 fois sur 100, quand le test déclare une personne non malade, elle l'est vraiment. Plus loin, il juge la spécificité presque égale à 1, parce qu'une seule personne malade a reçu un résultat négatif. Quelle mesure chacune de ces deux phrases décrit-elle vraiment ? Corrige-les. Calcule aussi l'accuracy dans les villes de i et j : pourquoi vaut-elle la même chose qu'en h ? (réponds dans ta copie)

### Ex 3.8 — F1, moyenne harmonique : pourquoi elle punit le maillon faible ∂ ★★ ⏱️ 25 min
**Objectif :** démontrer les propriétés de la moyenne harmonique qui font du F1 une bonne mesure de synthèse.
**Prérequis :** Ex 3.2 · fiche §3.7.11 (encadré 🧮) · **Parcours :** M

Pour deux nombres $a > 0$ et $b > 0$, la moyenne harmonique est $H = \frac{2}{\frac{1}{a} + \frac{1}{b}}$ et la moyenne arithmétique (ordinaire) est $M = \frac{a + b}{2}$.
1. Montre que $H = \frac{2ab}{a + b}$. Déduis-en que le F1 (moyenne harmonique de $P$ et $R$) vaut $\frac{2\,TP}{2\,TP + FP + FN}$.
2. Montre que $M - H = \frac{(a - b)^2}{2(a + b)}$. Déduis-en que $H \le M$, avec égalité seulement si $a = b$.
3. Montre que $\min(a, b) \le H \le 2 \min(a, b)$. Que veut dire cette inégalité pour un modèle dont le recall vaut 0,05 ?
4. La precision vaut 1. Que deviennent $M$ et le F1 quand le recall tend vers 0 ?
5. À partir de $F_\beta = \frac{(1 + \beta^2)\,P R}{\beta^2 P + R}$, retrouve $F_\beta = \frac{(1 + \beta^2)\,TP}{(1 + \beta^2)\,TP + \beta^2 FN + FP}$. Sur le même jeu de test, qui compte 55 positifs, le premier modèle en trouve 45 (10 FN) avec 10 FP, le second en trouve 50 (5 FN) avec 25 FP. Lequel a le meilleur $F_2$, alors qu'il fait plus d'erreurs ? Et le meilleur $F_1$ ? Explique-le avec la formule.
6. Tu fais un aller à 60 km/h et le retour, sur la même route, à 20 km/h. Quelle est ta vitesse moyenne sur l'aller-retour ? Pourquoi est-ce une moyenne harmonique ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 3.9 — Precision et recall expliqués à une médecin 🗣️ ★ ⏱️ 10 min
**Objectif :** relier precision et recall au vocabulaire médical, simplement.
**Prérequis :** fiche §3.7.6 à §3.7.8 · **Parcours :** R

Une médecin généraliste connaît très bien la « sensibilité » et la « valeur prédictive positive » d'un examen, mais n'a jamais entendu parler de machine learning. Explique-lui, en **cinq lignes au plus**, ce que mesurent la precision et le recall d'un modèle qui repère des grains de beauté suspects sur des photos. Contraintes :
- les mots « faux positif » et « faux négatif » ;
- le lien avec son propre vocabulaire ;
- aucune formule.

Relis-toi à voix haute, puis compare avec la réponse modèle de `05_solutions.md`.

### Ex 3.10 — Dépistage de masse : que dire à une personne testée positive ? ⚖️ ★★ ⏱️ 20 min
**Objectif :** mesurer les conséquences d'un dépistage à grande échelle pour les personnes, et décider comment l'organiser.
**Prérequis :** Ex 3.7 · fiche §3.8 · **Parcours :** R

Une ville propose, par une application, un autotest gratuit d'une maladie rare : sa prévalence est de 0,5 %. Le test a une sensibilité de 0,95 et une spécificité de 0,97. Le résultat est envoyé par SMS.
1. Pour 100 000 personnes testées, construis l'arbre des fréquences naturelles (malades, TP, FN, sains, FP, TN). Quelle est la precision du test ?
2. Rédige le SMS envoyé à une personne dont le test est positif : trois phrases au plus, honnêtes, compréhensibles, qui n'affolent pas et disent quoi faire.
3. Quels risques le dépistage fait-il courir aux personnes faussement positives ? Et aux faux négatifs ?
4. Qui devrait décider du seuil du test, et quelles informations faut-il publier (sensibilité, spécificité, prévalence, precision) ?
5. Le dépistage est réservé aux personnes à risque, chez qui la prévalence atteint 5 %. Que devient la precision ? En quoi cibler le dépistage est-il à la fois plus efficace et plus juste pour les personnes ?
6. Les résultats de ce test sont des données de santé. Quelles précautions l'application doit-elle prendre ?

### Ex 3.11 — Fawcett (2006) : une introduction à l'analyse ROC 📄 ★★ ⏱️ 30 min
**Objectif :** lire un article de référence sur les courbes ROC et en retenir l'essentiel.
**Prérequis :** Ex 3.5 · fiche, section « au-delà du livre (1) : la courbe ROC et l'AUC » · **Parcours :** complet seulement (lecture conseillée à tous)

L'article : T. Fawcett, « An introduction to ROC analysis », *Pattern Recognition Letters*, vol. 27, n° 8, 2006, p. 861-874 ([DOI 10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010)). L'accès est payant chez l'éditeur ; une bibliothèque universitaire y donne souvent accès, et une version de travail de l'auteur circule (cherche le titre sur un moteur de recherche académique). Lis au moins les sections 1 à 5, 7 et 9.

1. Dans quels domaines utilisait-on les graphes ROC avant le machine learning ?
2. Dans l'espace ROC, que représentent les points $(0, 0)$, $(1, 1)$ et $(0, 1)$, et la diagonale ?
3. Qu'appelle-t-il un classifieur « conservateur » et un classifieur « libéral » ? Où les trouve-t-on sur le graphe ?
4. Pourquoi une courbe ROC ne change-t-elle pas quand la proportion de positifs change, alors que la precision change ?
5. Quelle interprétation statistique donne-t-il de l'AUC ? À quel test statistique est-elle équivalente ? Quelle relation la relie au coefficient de Gini ?
6. Décris en trois étapes son algorithme de construction d'une courbe ROC. Pourquoi faut-il traiter ensemble les échantillons de même score ?
7. Comment propose-t-il d'étendre l'analyse ROC à plus de deux classes ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 3.E1 — 99 % d'accuracy sur la détection de fraude : bonne nouvelle ? 💼 ★★ ⏱️ 10 min
*Fiche §3.7.5, §3.8 · prérequis 3.7 · parcours R*

« Votre modèle de détection de fraude a 99 % d'accuracy. Est-ce une bonne nouvelle ? Qu'est-ce que vous regardez d'autre ? »

### 3.E2 — Precision ou recall : anti-spam, dépistage, modération 💼 ★★ ⏱️ 10 min
*Fiche §3.7.4, §3.7.8 · parcours R*

« Pour un filtre anti-spam, un dépistage médical et la modération automatique de commentaires, vous privilégiez la precision ou le recall ? Pourquoi ? »

### 3.E3 — Expliquer la courbe ROC et l'AUC 💼 ★★ ⏱️ 10 min
*Fiche, au-delà du livre (1) · prérequis 3.24 · parcours R*

« Expliquez-moi la courbe ROC et l'AUC. Une AUC de 0,5, ça veut dire quoi ? Quand préférez-vous une autre courbe ? »

### 3.E4 — F1 macro ou micro : lequel choisir ? 💼 ★★ ⏱️ 10 min
*Fiche §3.7.9 (plusieurs classes), §3.7.11 · prérequis 3.22 · parcours R*

« Sur un problème à plusieurs classes déséquilibrées, vous rapportez le F1 macro, micro ou pondéré ? Pourquoi ? »

### 3.E5 — Qu'est-ce qu'un modèle bien calibré ? 💼 ★★ ⏱️ 10 min
*Fiche §3.1, au-delà du livre (3) · parcours R*

« Qu'est-ce qu'un modèle bien calibré ? Comment le vérifiez-vous, et comment corrigez-vous un modèle mal calibré ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch03_probabilites/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie `mylearn/metrics.py`. La partie 0 du notebook vérifie tes réponses aux exercices ✏️ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 3.12 | Dix mille fléchettes : estimer des aires (et π) | 🔬 | ★ | 15 |
| 3.13 | Deux disques : P(A\|B) = P(B\|A) ? | 🔮 | ★ | 10 |
| 3.14 | Penguins : espèce × île avec pd.crosstab | 📦 | ★★ | 20 |
| 3.15 | confusion_matrix à la manière de scikit-learn | 🔨 | ★★ | 20 |
| 3.16 | accuracy, precision, recall, F-beta et F1 (cas binaire) | 🔨 | ★★★ | 40 |
| 3.17 | La matrice à l'envers | 🐛 | ★★ | 15 |
| 3.18 | Tout positif, un seul positif : prédire les scores | 🔮 | ★★ | 15 |
| 3.19 | Le tableau de bord complet : classification_rates | 🔨 | ★★ | 20 |
| 3.20 | Un seuil sur la nageoire : precision et recall en balance | 🔬 | ★★ | 25 |
| 3.21 | Simuler le dépistage : la prévalence fait la precision | 🔬 | ★★ | 30 |
| 3.22 | Vérifier avec scikit-learn : classification_report et affichages | 📦 | ★★ | 20 |
| 3.23 | Lire la documentation de sklearn.metrics | 🛠️ | ★★ | 25 |
| 3.24 | Courbe ROC et AUC | 🔨 | ★★★ | 40 |
| 3.25 | Moyennes macro, micro et pondérée | 🔨 | ★★★ | 35 |
| 3.26 | Courbe precision-recall et average precision | 🔨 | ★★★ | 40 |
| 3.27 | ROC ou PR ? Lire les courbes d'un problème déséquilibré | 📈 | ★★ | 25 |
| 3.28 | Calibration : quand la météo annonce 70 % | 🔨 | ★★★ | 35 |
| 3.29 | Recall ≥ 0,99 au meilleur prix | 🏆 | ★★★ | 45 |
