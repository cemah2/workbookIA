# 3 · Probabilités et mesure de la qualité — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 3.Q1 — Probabilité, pourcentage, degré de confiance

<details><summary>Indice 1</summary>

Une probabilité et un pourcentage disent la même chose à un facteur 100 près. Pour les questions 4 et 5, demande-toi comment on pourrait **vérifier** une confiance annoncée sur beaucoup de cas.

</details>
<details><summary>Indice 2</summary>

Pour la question 3, relis le quiz Q2 du ch. 2 : que vaut $P(X = 1{,}5)$ pour une variable continue ? Pour les questions 4 et 5, relis le début de la fiche (§3.1) et sa dernière section (calibration).

</details>
<details><summary>Indice 3</summary>

Question 1 : pour passer d'une probabilité à un pourcentage, multiplie par 100 : $0{,}35 \times 100 = 35$, soit 35 % ; pour revenir à une probabilité, divise par 100 : 7 % donne $\frac{7}{100} = 0{,}07$. Pour les autres, un critère chacune. 2 : entre quelles bornes une probabilité est-elle toujours comprise, et de laquelle « très probable » est-il proche ? 3 : pour une variable continue, que vaut $P(X = 1{,}5)$ (ch. 2, Q2) ? Et pourtant, le tirage donne-t-il une valeur ? 4 : si ta confiance de 80 % est juste, combien de fruits mûrs devrais-tu trouver parmi les 200 ? 5 : fais le même contrôle pour le modèle : parmi tous les e-mails notés 0,97, quelle part devrait être des spams, et tous les modèles le garantissent-ils (fiche, au-delà du livre 3) ? 6 : relis la fiche §3.1 : avec quel vocabulaire les documentations décrivent-elles leurs méthodes ?

</details>

### 3.Q2 — Fléchettes : pourquoi chaque point du mur doit être aussi probable

<details><summary>Indice 1</summary>

Une probabilité égale à un rapport d'aires suppose que la position de l'impact suit la loi uniforme du ch. 2.

</details>
<details><summary>Indice 2</summary>

Imagine un lanceur qui vise le centre : une petite tache au centre et une grande tache dans un coin recevraient-elles des fléchettes en proportion de leur aire ? Pour la question 4, calcule ce que deviennent les aires quand toutes les longueurs doublent.

</details>
<details><summary>Indice 3</summary>

Question 1 : la probabilité est le rapport des aires, $\frac{1}{4} = 0{,}25$. Pour les autres, un critère chacune. 2 : sans cette hypothèse, une petite tache au centre pourrait-elle recevoir plus de fléchettes qu'une grande tache dans un coin ? La probabilité dépendrait-elle encore seulement de l'aire ? 3 : la même question avec ce lanceur : la position d'une tache compte-t-elle ? 4 : quand toutes les longueurs doublent, chaque aire est multipliée par $2 \times 2$ ; que deviennent les rapports d'aires ? 5 : le fond et les taches couvrent tout le mur sans se chevaucher : que vaut la somme de leurs probabilités ? 6 : divise les fléchettes tombées dans A par le nombre total de fléchettes ; puis demande-toi si un autre lot de 500 fléchettes donnerait exactement le même compte (ch. 2).

</details>

### 3.Q3 — P(A|B) ou P(B|A) ?

<details><summary>Indice 1</summary>

Dans $P(A \mid B)$, ce qui est **après** la barre est ce qu'on sait déjà ; ce qui est **avant**, ce dont on cherche la probabilité.

</details>
<details><summary>Indice 2</summary>

Un exemple voisin : « parmi les e-mails avec lien, 20 % sont des spams » (fiche §3.5) : on sait que l'e-mail a un lien, donc « lien » va après la barre, et l'on écrit $P(\text{spam} \mid \text{lien}) = 0{,}2$. Pour la question 4, demande-toi quelle espèce peut avoir des nageoires de plus de 210 mm, et si tous les Gentoo en ont d'aussi longues.

</details>
<details><summary>Indice 3</summary>

Phrase 1 : ce qu'on sait déjà (elle boit de l'eau) va après la barre, ce qu'on cherche (elle a soif) avant : $P(\text{soif} \mid \text{boit de l'eau})$. Pour les autres, un critère chacune. 2 et 3 : la même méthode ; le groupe qui suit « parmi » est ce qu'on sait déjà. 4 : les deux probabilités ont le même numérateur, les Gentoo dont la nageoire dépasse 210 mm, mais pas le même groupe de référence ; les manchots des autres espèces dépassent-ils souvent 210 mm, et tous les Gentoo dépassent-ils 210 mm ? 5 : si A contient B, quelle part de B est dans A ? Et si A et B ne se touchent pas ? 6 : écris $P(A \mid B) = \frac{P(A, B)}{P(B)}$ et $P(B \mid A) = \frac{P(A, B)}{P(A)}$, remplace $P(A)$ par $P(B)$, puis compare les deux fractions.

</details>

### 3.Q4 — Jointe = conditionnelle × simple

<details><summary>Indice 1</summary>

Suis les fléchettes en deux étapes : d'abord combien tombent dans B, puis quelle part de celles-ci tombe aussi dans A.

</details>
<details><summary>Indice 2</summary>

Dans B : 40 % de 200. Dans A et B : le quart de ce nombre. $P(A, B)$ est ce dernier nombre divisé par 200.

</details>
<details><summary>Indice 3</summary>

Question 1 : B reçoit 40 % des 200 fléchettes, $0{,}4 \times 200 = 80$ ; A et B à la fois en reçoivent le quart, soit 20. Pour les autres : 2 : divise ce dernier nombre par 200, ou calcule $P(A \mid B)\,P(B)$ ; 3 : multiplie par $P(B)$ la définition $P(A \mid B) = \frac{P(A, B)}{P(B)}$, puis fais de même avec $P(B \mid A)$ ; 4 : « A et B » et « B et A » désignent-ils la même région du mur ? 5 : la partie commune est-elle incluse dans A ? Compare leurs aires ; 6 : pour des événements indépendants, la règle du produit devient $P(A, B) = P(A)\,P(B)$.

</details>

### 3.Q5 — D'où vient le mot « marginale »

<details><summary>Indice 1</summary>

Relis le début du §3.6 du livre et l'encadré `pd.crosstab` de la fiche.

</details>
<details><summary>Indice 2</summary>

Une marge est un total de ligne ou de colonne. Pour la question 5, cherche deux tables différentes qui ont les mêmes totaux (par exemple, ajoute 1 à deux cases opposées en diagonale et retire 1 aux deux autres : aucun total ne bouge).

</details>
<details><summary>Indice 3</summary>

Question 1 : le livre (§3.6) rapporte une légende : les anciens recueils de tables de probabilités reportaient les totaux dans la marge de la page, d'où le mot. Pour les autres, un critère chacune. 2 : où écrit-on les totaux des lignes et des colonnes d'une table, et par quoi les divise-t-on pour obtenir des probabilités ? 3 : dans combien de cases chaque individu est-il compté ? Et combien de valeurs d'une même variable peut-il avoir ? 4 : la marge de $A$ est le total de sa ligne dans la table des probabilités jointes : quelles cases additionne-t-on, et comment l'écrire avec $\sum$ ? 5 : pars de `[[1, 1], [1, 1]]` et applique l'astuce de l'indice 2 : la nouvelle table a-t-elle les mêmes totaux ? Est-elle la même table ? 6 : relis le rappel R3 : quelle méthode d'une colonne compte ses valeurs, et quel argument divise ces comptes par le total ?

</details>

### 3.Q6 — Vérité terrain et prédiction

<details><summary>Indice 1</summary>

Relis le §3.7.1 de la fiche : vérité terrain, prédiction, classe positive, frontière de décision.

</details>
<details><summary>Indice 2</summary>

Pour la question 4, calcule la precision et le recall du filtre anti-spam de la fiche en prenant « normal » comme classe positive : qu'est-ce qui devient TP, FP, FN, TN ?

</details>
<details><summary>Indice 3</summary>

Question 1 : la vérité terrain est le label qu'on tient pour correct ; elle vient d'un humain qui a vérifié, ou d'un test fiable mais coûteux (le test lent du livre). Pour les autres, un critère chacune. 2 : relis la fiche §3.7.1 : dans quel espace la frontière est-elle tracée, et que sépare-t-elle ? 3 : quelle classe le filtre cherche-t-il à repérer ? Un « positif » est-il forcément une bonne nouvelle (pense à une maladie) ? 4 : échange les rôles dans le filtre de la fiche (TP = 8, FN = 4, FP = 2, TN = 36) : les anciens TN deviennent les TP, les anciens FN deviennent les FP, et inversement ; recalcule $\frac{TP}{TP + FP}$ et $\frac{TP}{TP + FN}$, puis compare ; 5 : relis le ch. 1 : quelles données le modèle ne doit-il jamais avoir vues ? 6 : qui pose les labels, et dans quelles situations peut-il se tromper ?

</details>

### 3.Q7 — Lire une matrice de confusion (et vérifier ses axes)

<details><summary>Indice 1</summary>

scikit-learn trie les labels (0 puis 1) et met la vérité en lignes, la prédiction en colonnes.

</details>
<details><summary>Indice 2</summary>

La case en haut à gauche est donc (vérité 0, prédiction 0), celle en bas à droite (vérité 1, prédiction 1). Les malades sont la ligne du bas, les prédictions « malade » la colonne de droite.

</details>
<details><summary>Indice 3</summary>

Question 1 : scikit-learn range `[[TN, FP], [FN, TP]]`, donc TN = 50, FP = 5, FN = 10 et TP = 35. Pour les autres : 2 : les malades sont la ligne du bas, les prédictions « malade » la colonne de droite : additionne leurs cases ; 3 : le collègue lit le tableau à la manière du livre, TP en haut à gauche et TN en bas à droite : quels nombres y trouve-t-il ? 4 : où sont les cases « vérité = prédiction » quand les lignes et les colonnes rangent les classes dans le même ordre ? 5 : dans combien de cases chaque échantillon est-il compté ? 6 : relis l'encadré 🕰️ de la fiche §3.7.2 : la disposition est-elle la même d'un outil à l'autre ?

</details>

### 3.Q8 — Faux positif ou faux négatif : lequel coûte le plus ?

<details><summary>Indice 1</summary>

Pour chaque situation, écris ce qui se passe concrètement après un faux positif, puis après un faux négatif.

</details>
<details><summary>Indice 2</summary>

La precision juge les alertes (les faux positifs), le recall juge les oublis (les faux négatifs). Pour les figurines aux yeux peints, attention : le positif est « yeux présents ».

</details>
<details><summary>Indice 3</summary>

Situation 1 : un faux négatif est une fraude non détectée, de l'argent volé ; un faux positif, un appel de vérification au client. L'oubli coûte le plus : on surveille le recall. Pour les autres, la même démarche : écris la conséquence concrète d'un faux positif, puis celle d'un faux négatif, compare-les, et souviens-toi que la precision juge les fausses alertes, le recall les oublis. 2 : que devient un vrai e-mail classé en spam, et un spam qui passe ? 3 : une figurine interdite qui part chez un client, une figurine autorisée retirée de la chaîne : laquelle est un faux positif, laquelle un faux négatif ? 4 : attention, le positif est « yeux présents » : décris ce que devient la figurine dans un faux positif, puis dans un faux négatif. 5 : que se passe-t-il ensuite pour une personne positive (l'examen plus précis), et pour une personne négative ? 6 : relis la fiche §3.7.4 : qu'écrit-on avant de choisir une mesure ?

</details>

### 3.Q9 — L'accuracy face aux classes déséquilibrées

<details><summary>Indice 1</summary>

Compte les bonnes réponses d'un modèle qui répond toujours « pas de fraude ».

</details>
<details><summary>Indice 2</summary>

Quelles transactions ce modèle classe-t-il bien, et lesquelles rate-t-il ? L'accuracy compte les premières, le recall ne regarde que les fraudes. Pour la question 4, souviens-toi du bootstrap du ch. 2 : l'incertitude d'une proportion dépend du nombre d'exemples.

</details>
<details><summary>Indice 3</summary>

Question 1 : le modèle classe bien les 990 transactions normales et rate les 10 fraudes : accuracy $\frac{990}{1\,000} = 0{,}99$, recall $\frac{0}{10} = 0$. Pour les autres : 2 : que mesure ici l'accuracy : ce que le modèle a appris, ou la rareté des fraudes ? 3 : relis le ch. 1 : quel score un modèle doit-il battre pour avoir appris quelque chose, et comment l'obtient-on sans rien apprendre ? 4 : l'écart-type d'une proportion $p$ mesurée sur $n$ exemples vaut environ $\sqrt{p(1 - p)/n}$ : calcule-le avec $p = 0{,}7$, pour $n = 20$ puis pour $n = 20\,000$ ; 5 : dans $\frac{TP + TN}{n}$, une erreur sur un malade et une erreur sur une personne saine retirent-elles la même chose ? 6 : dans le tableau de la fiche §3.7.9, cherche la mesure qui calcule un score sur les positifs, un autre sur les négatifs, puis fait la moyenne des deux.

</details>

### 3.Q10 — Precision ou recall : le moteur de recherche du wiki

<details><summary>Indice 1</summary>

Une page renvoyée et pertinente est un TP ; renvoyée et non pertinente, un FP ; pertinente mais non renvoyée, un FN ; ni l'un ni l'autre, un TN.

</details>
<details><summary>Indice 2</summary>

Les 25 pages renvoyées sont des TP ou des FP ; les 40 pages pertinentes sont des TP ou des FN. Commence par les TP, déduis-en FP et FN, puis les TN à partir du total de 500 pages.

</details>
<details><summary>Indice 3</summary>

Question 1 : les 20 pages renvoyées et pertinentes sont les TP ; FP = 25 − 20 = 5 (renvoyées, non pertinentes) ; FN = 40 − 20 = 20 (pertinentes, non renvoyées) ; TN = 500 − 20 − 5 − 20 = 455. Pour les autres : 2 : precision $= \frac{TP}{TP + FP}$ et recall $= \frac{TP}{TP + FN}$, avec les cases de la question 1 ; 3 : $\frac{TP + TN}{500}$, puis regarde quelle case pèse le plus dans ce calcul ; 4 : relis la fiche §3.7.8 : de quel domaine vient le mot, et que « fait revenir » un moteur ? 5 : refais la question 1 avec 80 pages renvoyées, dont les 40 pertinentes, puis les deux fractions ; 6 : qui paie le plus cher un document manqué, et qui paie le plus cher un mauvais résultat en haut de page ?

</details>

### 3.Q11 — Tricher sur une seule mesure

<details><summary>Indice 1</summary>

Relis le §3.7.10 du livre (figures 3.35 à 3.38) : deux frontières extrêmes.

</details>
<details><summary>Indice 2</summary>

Tout déclarer positif : combien de TP, de FP, de FN ? Ne déclarer positif qu'un seul point, juste : même question. Pour les questions 4 et 5, écris ce que devient chaque case quand le seuil monte.

</details>
<details><summary>Indice 3</summary>

Question 1 : déclarer tout positif trouve les 10 positifs, donc recall = 1 ; les 20 points sont alors des alertes, dont 10 justes : precision $\frac{10}{20} = 0{,}5$, la prévalence. Pour les autres : 2 : ne déclare positif qu'un seul point, juste : combien de TP, de FP et de FN ? 3 : chacune de ces deux stratégies a-t-elle appris quelque chose ? Que révèle l'autre mesure ? 4 : quand le seuil monte, que peuvent faire les TP, et le dénominateur TP + FN dépend-il du seuil ? 5 : reprends les six scores du mini-exemple de la fiche (courbe ROC) : calcule la precision au seuil 0,4, puis au seuil 0,7, et compare ; 6 : relis les quatre paires de la fiche §3.7.9 qui s'additionnent à 1.

</details>

### 3.Q12 — F1 et test « fiable à 99 % »

<details><summary>Indice 1</summary>

Le F1 est la moyenne harmonique de la precision et du recall (fiche §3.7.11).

</details>
<details><summary>Indice 2</summary>

$F_1 = \frac{2PR}{P + R}$. Pour la question 4, écris l'information donnée avec « sachant », puis celle que veut le patient.

</details>
<details><summary>Indice 3</summary>

Question 1 : la moyenne ordinaire vaut $\frac{0{,}9 + 0{,}1}{2} = 0{,}5$ et le F1 $\frac{2 \times 0{,}9 \times 0{,}1}{0{,}9 + 0{,}1} = 0{,}18$. Pour les autres : 2 : relis l'encadré 🧮 de la fiche §3.7.11 : entre quels nombres la moyenne harmonique reste-t-elle ? 3 : écris le F1 avec les cases de la matrice (fiche §3.7.11) : lesquelles y entrent ? 4 : écris la phrase avec « sachant » (on connaît la maladie, on regarde le test), puis la probabilité que veut le patient ; pour la calculer, refais l'arbre des fréquences naturelles de la fiche §3.8 : de quels nombres as-tu besoin pour le remplir ? 5 : compare la taille du groupe des personnes saines à celle du groupe des malades, puis applique à chacun son taux d'erreur ; 6 : relis la fin de la fiche §3.8 (les conséquences pratiques).

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 3.R1 — Ch. 2 : deux cartes rouges de suite, avec ou sans remise

<details><summary>Indice 1</summary>

Avec remise, le paquet est le même au second tirage ; sans remise, il manque une carte, et elle était rouge.

</details>
<details><summary>Indice 2</summary>

Sans remise : après une rouge, combien reste-t-il de cartes dans le paquet, et combien de rouges ? La probabilité de deux rouges est le produit des deux étapes (règle du produit, §3.5).

</details>
<details><summary>Indice 3</summary>

Question 1 : 16 cartes rouges sur 32, $\frac{16}{32} = 0{,}5$. Pour les autres : 2 : avec remise, le second tirage se fait dans le même paquet : multiplie les probabilités des deux étapes ; 3 : sans remise, après une rouge, il reste 31 cartes, dont 15 rouges ; la probabilité de deux rouges est (probabilité de la 1ʳᵉ rouge) × (probabilité de la 2ᵉ rouge sachant la 1ʳᵉ) ; 4 : dans quel cas le premier tirage change-t-il le paquet du second ? 5 : relis le ch. 2 (§2.5) : la méthode `choice` d'un générateur, avec `size=5` et l'argument qui interdit de tirer deux fois la même carte (vérifie sa valeur par défaut avec `help(rng.choice)`).

</details>

### 3.R2 — 0B : ensembles, intersection, union, complémentaire

<details><summary>Indice 1</summary>

Dessine deux cercles qui se chevauchent, S et M, et place 12 dans la partie commune.

</details>
<details><summary>Indice 2</summary>

Sport seulement : 40 − 12 ; musique seulement : 30 − 12. L'union est la somme des trois morceaux.

</details>
<details><summary>Indice 3</summary>

Question 1 : les 12 élèves qui font les deux forment $S \cap M$, et $|S \cup M| = 40 + 30 - 12 = 58$. Pour les autres : 2 : retire de 100 les élèves de l'union ; 3 : divise chaque effectif (celui de S, celui de $S \cap M$, puis celui de l'union) par 100 ; 4 : la probabilité du complémentaire vaut $1 - P(S)$ ; 5 : compare $P(S \cap M)$ au produit $P(S)\,P(M)$ ; 6 : dans $|S| + |M|$, combien de fois chaque élève qui fait les deux est-il compté ?

</details>

### 3.R3 — 0A : pandas, `value_counts(normalize=True)` et `groupby`

<details><summary>Indice 1</summary>

`value_counts` compte les valeurs d'une colonne ; `normalize=True` divise par le total.

</details>
<details><summary>Indice 2</summary>

`groupby("species")` coupe le tableau en un groupe par espèce, puis applique la fonction à chaque groupe.

</details>
<details><summary>Indice 3</summary>

Question 1 : `df["species"].value_counts()` renvoie une Series, le nombre de manchots de chaque espèce, de la plus fréquente à la moins fréquente ; avec `normalize=True`, chaque nombre est divisé par le total : ce sont des proportions. Pour les autres : 2 : que vaut la somme des proportions de toutes les valeurs (les valeurs manquantes sont ignorées) ? 3 : `groupby("species")` fait un groupe par espèce ; que donne `.mean()` de la colonne choisie dans chaque groupe, et quel est l'index du résultat ? 4 : combine les deux outils : un groupe par espèce, puis, dans chaque groupe, les proportions des valeurs de la colonne `sex` (attention aux sexes manquants) ; 5 : un masque booléen, qui compare la colonne `island` à `"Biscoe"`, placé entre les crochets de `df`.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 3.1 — Fléchettes et aires : probabilités simples et conditionnelles ✏️

<details><summary>Indice 1</summary>

Dessine le mur et les deux rectangles à l'échelle, puis calcule trois aires : A, B, et leur partie commune.

</details>
<details><summary>Indice 2</summary>

La partie commune est le rectangle où les deux conditions sur $x$ et les deux conditions sur $y$ sont vraies à la fois : $x$ entre 1,5 et 2,5, $y$ entre 1 et 2. Une probabilité simple ou jointe divise par l'aire du mur ; une conditionnelle divise par l'aire de ce qu'on sait.

</details>
<details><summary>Indice 3</summary>

a) l'aire de A vaut $2 \times 1{,}5 = 3$ m², celle du mur $4 \times 2{,}5 = 10$ m² : $P(A) = \frac{3}{10} = 0{,}3$. Pour les autres : b) la même division avec l'aire de B ; c) l'aire de la partie commune de l'indice 2, divisée par l'aire du mur ; d) $P(A \mid B) = \frac{\text{aire commune}}{\text{aire de B}}$ : sachant B, B devient le nouveau mur ; e) la même méthode, sachant A ; f) $1 - P(A \text{ ou } B)$, avec $P(A \text{ ou } B) = P(A) + P(B) - P(A, B)$ ; g) et h) 400 × la probabilité de b, puis celle de c ; i) les fléchettes dans A et B, divisées par les fléchettes dans B ; j) compare $P(A, B)$ au produit $P(A)\,P(B)$ ; k) un comptage sur 400 tirages au hasard tombe-t-il toujours pile sur la valeur attendue (ch. 2) ? Comment son erreur typique varie-t-elle avec le nombre de fléchettes ?

</details>

### Ex 3.2 — Les 20 points : matrice de confusion et quatre mesures ✏️

<details><summary>Indice 1</summary>

Pour chaque point, le couple (vérité, prédiction) dit dans quelle case il tombe : (1, 1) TP, (0, 1) FP, (1, 0) FN, (0, 0) TN.

</details>
<details><summary>Indice 2</summary>

Fais un tableau 2 × 2 et une croix par point. Vérifie que le total fait 20 et que TP + FN vaut le nombre de 1 dans la ligne « vérité ». Pour e, scikit-learn met la vérité 0 en première ligne.

</details>
<details><summary>Indice 3</summary>

a) les couples (vérité 1, prédiction 1) sont les points 5, 9, 10, 11, 14, 17 et 19 : TP = 7. Pour les autres : b) à d) la même méthode avec les couples (0, 1), (1, 0) et (0, 0) ; vérifie que TP + FN est le nombre de 1 de la ligne « vérité », TP + FP celui de la ligne « prédiction », et que les quatre cases font 20 ; e) `[[TN, FP], [FN, TP]]`, avec tes quatre cases ; f) $\frac{TP + TN}{20}$ ; g) $\frac{TP}{TP + FP}$ ; h) $\frac{TP}{TP + FN}$ ; i) $F_1 = \frac{2\,TP}{2\,TP + FP + FN}$ : cette forme évite de partir d'une precision arrondie ; j) si les positifs sont des fraudes, que coûte un faux négatif, et que coûte un faux positif ?

</details>

### Ex 3.3 — Le glacier : jointes, marginales et conditionnelles ✏️

<details><summary>Indice 1</summary>

Une probabilité jointe divise une case par le total général ; une conditionnelle divise une case par le total de ce qu'on sait déjà.

</details>
<details><summary>Indice 2</summary>

« Sachant un cornet » : divise par le total de la colonne « cornet » (80). « Sachant la vanille » : divise par le total de la ligne « vanille » (60).

</details>
<details><summary>Indice 3</summary>

a) la ligne « vanille » totalise 60 clients sur 150 : $P(V) = \frac{60}{150} = 0{,}4$. Pour les autres : b) la même division avec le total de la colonne « cornet » ; c) une probabilité jointe : la case (vanille, cornet) divisée par le total général ; d) la même case, divisée par le total de la colonne « cornet » ; e) la même case, divisée par le total de la ligne « vanille » ; f) la case (chocolat, pot), divisée par le total de la colonne « pot » ; g) compare ta réponse c au produit de a et b, calculé avec les fractions exactes ; h) 300 × ta réponse c ; i) une même case, deux groupes de référence : lesquels ?

</details>

### Ex 3.4 — Règle du produit et formule des probabilités totales ∂

<details><summary>Indice 1</summary>

Toutes les questions partent de la définition $P(A \mid B) = \frac{P(A, B)}{P(B)}$ et du fait que $P(A, B) = P(B, A)$ ; la question 3 utilise en plus l'additivité des aires (deux morceaux sans partie commune).

</details>
<details><summary>Indice 2</summary>

Pour 2, égalise les deux écritures de $P(A, B)$. Pour 3, découpe la tache A en deux morceaux sans partie commune : ce qui est dans B, et ce qui n'y est pas. Pour 4, découpe A en $K$ morceaux au lieu de deux ; pour la vérification, la ligne « femelle » et la ligne des totaux de la table de la fiche suffisent.

</details>
<details><summary>Indice 3</summary>

1. Multiplie par $P(B) > 0$ les deux membres de la définition : $P(A, B) = P(A \mid B)\,P(B)$. Pour la seconde égalité, écris $P(B \mid A) = \frac{P(B, A)}{P(A)}$, multiplie par $P(A) > 0$, et utilise $P(B, A) = P(A, B)$. Pour les autres : 2 : égalise les deux expressions de 1, puis divise par $P(B)$ ; pour la condition, écris $\frac{P(A, B)}{P(B)} = \frac{P(A, B)}{P(A)}$ et distingue les cas $P(A, B) > 0$ et $P(A, B) = 0$ ; 3 : $\text{aire}(A) = \text{aire}(A \cap B) + \text{aire}(A \cap \text{non } B)$ ; divise par l'aire du mur, puis applique la règle du produit à chaque terme ; 4 : le même découpage en $K$ morceaux, et une somme $\sum_{j=1}^{K}$ ; pour la vérification, chaque terme est (case « femelle » / total de sa colonne) × (total de sa colonne / 333) ; 5 : remplace $P(A, B)$ par $P(A)\,P(B)$ dans la définition de $P(A \mid B)$ ; 6 : prends A = V et B = C dans les formules de 1 et de 3, avec les nombres de la table de 3.3.

</details>

### Ex 3.5 — Toutes les mesures du tableau récapitulatif ✏️

<details><summary>Indice 1</summary>

Écris d'abord la matrice avec les totaux des lignes (positifs réels : 50, négatifs réels : 950) et des colonnes (prédictions positives : 100, négatives : 900).

</details>
<details><summary>Indice 2</summary>

Chaque mesure est une case divisée par un total : les **taux** (recall, FNR, spécificité, FPR) divisent par une ligne, les **valeurs prédictives** (precision, FDR, NPV, FOR) par une colonne. Aide-toi du tableau de la fiche (§3.7.9).

</details>
<details><summary>Indice 3</summary>

a) la prévalence est la part de positifs réels : $\frac{TP + FN}{1\,000} = \frac{50}{1\,000} = 0{,}050$. Pour les autres, une case divisée par un total : b) $\frac{TP + TN}{1\,000}$ ; c) et d) TP divisé par la colonne « prédit positif » (precision), puis par la ligne « positif réel » (recall) ; e) à j) chaque **taux** divise par une ligne (les 50 positifs réels ou les 950 négatifs réels), chaque **valeur prédictive** par une colonne (les 100 prédictions positives ou les 900 négatives) : par exemple, la spécificité et la NPV ont toutes deux TN au numérateur, mais l'une divise par une ligne, l'autre par une colonne ; k) $F_1 = \frac{2\,TP}{2\,TP + FP + FN}$ ; l) $\frac{\text{recall} + \text{spécificité}}{2}$, avec la spécificité exacte (une fraction), pas arrondie ; m) au numérateur $TP \cdot TN - FP \cdot FN$, au dénominateur la racine du produit des totaux des deux lignes et des deux colonnes ; n) pour chaque mesure, regarde si elle divise par une ligne (elle décrit le détecteur) ou par une colonne (elle dépend de la part de fraudes), puis pense à celles qui mélangent les deux.

</details>

### Ex 3.6 — Trois espèces : moyennes macro, micro et pondérée ✏️

<details><summary>Indice 1</summary>

Chaque espèce devient tour à tour la classe positive. Pour l'Adélie, que sont TP, FP et FN dans la matrice ?

</details>
<details><summary>Indice 2</summary>

TP : la case de la diagonale. FP : le reste de la **colonne** (des manchots prédits dans cette espèce à tort). FN : le reste de la **ligne**. Le support est le total de la ligne.

</details>
<details><summary>Indice 3</summary>

a) le support est le total de chaque ligne : [45, 20, 35]. Pour les autres, l'Adélie donne la méthode : TP = 41 (la diagonale), FP = 7 (le reste de la colonne Adélie), FN = 4 (le reste de la ligne Adélie). b) pour chaque espèce, TP divisé par le total de sa colonne ; c) TP divisé par le total de sa ligne ; d) $F_1 = \frac{2\,TP}{2\,TP + FP + FN}$, espèce par espèce ; e) et f) la moyenne simple des trois valeurs, calculée avec les fractions exactes, pas avec les valeurs arrondies de b et d ; g) la moyenne des trois F1 pondérée par les supports de a ; h) additionne d'abord les TP, les FP et les FN des trois espèces, puis une seule fraction ; i) chaque erreur, hors de la diagonale, est un FP pour une espèce et un FN pour une autre : compare le total des FP et celui des FN.

</details>

### Ex 3.7 — Le test « fiable à 99 % » dans une ville à 1 % de malades ✏️

<details><summary>Indice 1</summary>

Construis l'arbre des fréquences naturelles : pars de toute la population, sépare malades et personnes saines, puis applique le test à chaque groupe (fiche §3.8).

</details>
<details><summary>Indice 2</summary>

Les TP et FN viennent des malades (99 % et 1 % de ce groupe) ; les TN et FP viennent des personnes **saines** (99 % et 1 % de ce groupe, pas de la ville entière).

</details>
<details><summary>Indice 3</summary>

a) 1 % de 50 000 habitants : 500 malades, et donc 49 500 personnes saines. Pour les autres : b) et c) 99 % et 1 % des 500 malades ; d) et e) 1 % et 99 % des 49 500 personnes **saines**, pas de toute la ville ; f) $\frac{TP}{TP + FP}$ ; g) $\frac{TN}{TN + FN}$ ; h) $\frac{TP + TN}{50\,000}$ ; i) et j) refais l'arbre avec 0,2 % puis 10 % de malades (les personnes saines sont tous les autres habitants), puis $\frac{TP}{TP + FP}$ ; k) et l) construis la matrice du livre : 1 % de 10 000 habitants sont malades ; applique la sensibilité 0,99 aux malades et la spécificité 0,98 aux personnes saines, puis compare les dénominateurs : $TN + FP$ pour la spécificité, $TN + FN$ pour la NPV ; m) pour chaque phrase du livre, quelles cases de la matrice entrent dans le calcul qu'elle décrit ?

</details>

### Ex 3.8 — F1, moyenne harmonique : pourquoi elle punit le maillon faible ∂

<details><summary>Indice 1</summary>

Toutes les questions se ramènent à des calculs de fractions : mets au même dénominateur.

</details>
<details><summary>Indice 2</summary>

Pour 2, $(a + b)^2 - 4ab = (a - b)^2$. Pour 3, suppose $a \le b$ et compare $H$ à $a$, puis majore le dénominateur $a + b$. Pour 5, calcule les deux $F_2$, puis les deux $F_1$, avec la forme en comptages : que pèse un FN au dénominateur, et un FP ? Pour 6, calcule les durées de l'aller et du retour sur une distance $d$.

</details>
<details><summary>Indice 3</summary>

1. $\frac{1}{a} + \frac{1}{b} = \frac{a + b}{ab}$, donc $H = \frac{2}{(a + b)/(ab)} = \frac{2ab}{a + b}$ ; pour le F1, remplace $a$ et $b$ par $P = \frac{TP}{TP + FP}$ et $R = \frac{TP}{TP + FN}$, multiplie le numérateur et le dénominateur par $(TP + FP)(TP + FN)$, puis simplifie par $TP$. Pour les autres : 2 : mets $\frac{a + b}{2} - \frac{2ab}{a + b}$ au même dénominateur, et reconnais une identité remarquable au numérateur ; 3 : avec $a \le b$, $H - a = \frac{a(b - a)}{a + b}$ ; pour l'autre inégalité, $a + b \ge b$ donne $H \le \frac{2ab}{b}$ ; 4 : écris $M$ et $F_1$ avec $P = 1$, puis fais tendre $R$ vers 0 ; 5 : multiplie le numérateur et le dénominateur de $F_\beta$ par $\frac{(TP + FP)(TP + FN)}{TP}$, puis calcule les deux $F_2$ et les deux $F_1$ avec la forme en comptages ; 6 : pour une distance $d$ dans chaque sens, la durée totale vaut $\frac{d}{60} + \frac{d}{20}$ : divise la distance totale, $2d$, par cette durée.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 3.9 — Precision et recall expliqués à une médecin 🗣️

<details><summary>Indice 1</summary>

La médecin connaît déjà les deux notions sous d'autres noms : lesquels ?

</details>
<details><summary>Indice 2</summary>

Recall = sensibilité ; precision = valeur prédictive positive. Donne pour chacune une conséquence concrète d'un score faible.

</details>
<details><summary>Indice 3</summary>

Structure possible : une phrase pour le recall (et les faux négatifs), une pour la precision (et les faux positifs), une sur la prévalence, qu'elle connaît.

</details>

### Ex 3.10 — Dépistage de masse : que dire à une personne testée positive ? ⚖️

<details><summary>Indice 1</summary>

Commence par les chiffres : l'arbre des fréquences naturelles, comme en 3.7.

</details>
<details><summary>Indice 2</summary>

500 malades et 99 500 personnes saines. Le SMS doit donner le résultat, un ordre de grandeur de sa fiabilité, et la marche à suivre.

</details>
<details><summary>Indice 3</summary>

Question 1 : sur 100 000 personnes, les 500 malades donnent $0{,}95 \times 500 = 475$ TP (et 25 FN) ; les 99 500 personnes saines donnent $0{,}03 \times 99\,500 = 2\,985$ FP (et 96 515 TN). Precision : $\frac{475}{475 + 2\,985} \approx 0{,}137$, environ une personne positive sur sept est malade. Pour la question 2, tire de ce calcul un ordre de grandeur facile à lire. Pour la question 5, refais l'arbre avec 5 % de malades. Pour la question 6, cherche ce que dit le RGPD des données de santé (article 9).

</details>

### Ex 3.11 — Fawcett (2006) : une introduction à l'analyse ROC 📄

<details><summary>Indice 1</summary>

Les réponses se trouvent dans les sections 1 (introduction), 3 (espace ROC), 4 (courbes), 5 (construction efficace), 7 (AUC) et 9 (plus de deux classes).

</details>
<details><summary>Indice 2</summary>

Pour la question 4, regarde les dénominateurs du TPR et du FPR : dans quelle classe réelle sont-ils calculés ? Attention, la matrice de l'article met les classes réelles en colonnes.

</details>
<details><summary>Indice 3</summary>

Question 1 : l'introduction cite la théorie de la détection du signal (le compromis entre taux de détection et taux de fausses alarmes), puis la décision médicale, avant le machine learning. Pour les autres, où chercher et quoi regarder. 2 et 3 : la section 3 ; où se place un classifieur qui ne déclare jamais positif, ou toujours ? Et un classifieur prudent, qui fait peu de fausses alertes ? 4 : la section 4 : dans quelle classe réelle le TPR et le FPR sont-ils calculés ? Et la precision ? 5 : la section 7 : une interprétation par des paires (un positif, un négatif), le nom d'un test statistique sur les rangs, et une formule avec le coefficient de Gini ; 6 : la section 5 : que se passerait-il si deux échantillons de même score, un positif et un négatif, étaient traités l'un après l'autre, dans un ordre ou dans l'autre ? 7 : la section 9 : avec $K$ classes, combien de courbes trace-t-on, et quelle classe y joue chaque fois le rôle de la classe positive ? Quelle autre AUC, calculée sur deux classes à la fois, propose-t-elle ?

</details>

<a id="entretien"></a>

## 💼 Entretien

### 3.E1 — 99 % d'accuracy sur la détection de fraude : bonne nouvelle ?

<details><summary>Indice 1</summary>

Quelle accuracy obtient un modèle qui ne détecte jamais de fraude ?

</details>
<details><summary>Indice 2</summary>

Cite la prévalence, la classe majoritaire, puis les mesures qui regardent les fraudes : recall, precision, courbe precision-recall.

</details>
<details><summary>Indice 3</summary>

Termine par les coûts (fraudes manquées contre vérifications) et le choix du seuil sur un jeu de validation.

</details>

### 3.E2 — Precision ou recall : anti-spam, dépistage, modération

<details><summary>Indice 1</summary>

Pour chaque cas, quelle erreur coûte le plus : la fausse alerte ou l'oubli ?

</details>
<details><summary>Indice 2</summary>

Anti-spam : un vrai e-mail perdu. Dépistage : un malade rassuré à tort. Modération : dépend de ce que fait le modèle (suppression automatique, ou file d'attente humaine).

</details>
<details><summary>Indice 3</summary>

Conclus avec le seuil de décision et le F-beta ($\beta > 1$ favorise le recall).

</details>

### 3.E3 — Expliquer la courbe ROC et l'AUC

<details><summary>Indice 1</summary>

Pars du score et du seuil : chaque seuil donne un point de la courbe.

</details>
<details><summary>Indice 2</summary>

Axes : FPR en abscisse, TPR (recall) en ordonnée ; de (0, 0) à (1, 1) ; la diagonale est le hasard.

</details>
<details><summary>Indice 3</summary>

AUC = probabilité qu'un positif soit mieux classé qu'un négatif. Donne une limite : classes rares (courbe PR) et calibration.

</details>

### 3.E4 — F1 macro ou micro : lequel choisir ?

<details><summary>Indice 1</summary>

Qu'est-ce qui est additionné avant de calculer, et qu'est-ce qui est moyenné après ?

</details>
<details><summary>Indice 2</summary>

Micro : on additionne les comptages des classes, c'est l'accuracy. Macro : moyenne simple des F1 des classes. Pondéré : moyenne pondérée par l'effectif.

</details>
<details><summary>Indice 3</summary>

Le choix dépend de ce qui compte pour le métier : chaque classe autant (macro) ou chaque exemple autant (micro, pondéré). Mentionne le rapport par classe.

</details>

### 3.E5 — Qu'est-ce qu'un modèle bien calibré ?

<details><summary>Indice 1</summary>

Pars de l'exemple « parmi les cas annoncés à 0,8… ».

</details>
<details><summary>Indice 2</summary>

Vérifier : diagramme de fiabilité, score de Brier. Corriger : recalibrer sur un jeu de validation.

</details>
<details><summary>Indice 3</summary>

Méthodes : Platt (sigmoïde), régression isotonique, *temperature scaling* ; `CalibratedClassifierCV`. Distingue calibration et qualité du classement (AUC).

</details>

<a id="notebook"></a>

## Notebook, parties A à D

Les exercices du notebook (`03_notebook.ipynb`). Pour ceux qui complètent `mylearn/metrics.py`, la docstring de chaque fonction décrit déjà ce qu'elle doit faire, cas d'erreur compris : relis-la avant d'ouvrir un indice.

### Ex 3.12 — Dix mille fléchettes : estimer des aires (et π) 🔬

<details><summary>Indice 1</summary>

La probabilité de toucher le disque vaut $\frac{\pi}{4}$ (rapport des aires) ; la part des fléchettes qui le touchent l'estime. Une seule expression NumPy teste toutes les fléchettes d'un coup, sans boucle.

</details>
<details><summary>Indice 2</summary>

`points = 2 * rng.random((n, 2))` est un tableau de forme (n, 2). `(points - 1) ** 2` retire le centre (1, 1) à chaque coordonnée, et `.sum(axis=1)` donne $(x - 1)^2 + (y - 1)^2$ pour chaque fléchette ; comparé à 1, on obtient un tableau de booléens, dont la moyenne est une part. Pour b, le générateur est créé **une fois**, avant la liste, puis passé à chaque appel ; `np.std` de la liste. Pour c, l'erreur typique varie comme $1/\sqrt{n}$ : par combien faut-il multiplier $n$ pour multiplier $\sqrt{n}$ par 10 ?

</details>
<details><summary>Indice 3</summary>

`estimate_pi` : la signature, le squelette, puis la ligne clé.
```python
def estimate_pi(n, rng):
    points = 2 * rng.random((n, 2))                  # as in the statement: one row per dart
    inside = ((points - 1) ** 2).sum(axis=1) <= 1    # one boolean per dart: is it in the disc?
    # return 4 times the share of True, as a Python float
```
a) le même calcul, avec `np.random.default_rng(3)` et $n = 10\,000$ ; b) crée le générateur `np.random.default_rng(12)` **avant** la liste de l'énoncé, puis prends `np.std` de cette liste ; c) si l'erreur typique vaut $\frac{C}{\sqrt{n}}$, écris-la pour $k\,n$ fléchettes, et cherche $k$ pour qu'elle soit divisée par 10.

</details>

### Ex 3.13 — Deux disques : P(A|B) = P(B|A) ? 🔮

<details><summary>Indice 1</summary>

Écris les deux probabilités conditionnelles comme des rapports d'aires (fiche §3.4) : qu'ont-elles en commun, et en quoi diffèrent-elles ?

</details>
<details><summary>Indice 2</summary>

$P(A \mid B) = \frac{\text{aire}(A \cap B)}{\text{aire}(B)}$ et $P(B \mid A) = \frac{\text{aire}(A \cap B)}{\text{aire}(A)}$ : compare leurs numérateurs, puis leurs dénominateurs. Écris aussi leur rapport : que devient la partie commune ?

</details>
<details><summary>Indice 3</summary>

Même numérateur, la partie commune : la plus grande des deux probabilités est celle qui a le plus petit dénominateur. Lequel des deux disques a la plus petite aire ? Et que deviennent les dénominateurs quand les deux rayons sont égaux ? Pour c, $\frac{P(A \mid B)}{P(B \mid A)} = \frac{\text{aire}(A)}{\text{aire}(B)}$, et l'aire d'un disque de rayon $r$ vaut $\pi r^2$. Pour d, divise l'une par l'autre les deux estimations de l'expérience, $P(A \mid B)$ au numérateur, comme en c.

</details>

### Ex 3.14 — Penguins : espèce × île avec pd.crosstab 📦

<details><summary>Indice 1</summary>

`pd.crosstab(lignes, colonnes)` : la première série donne les lignes. Une probabilité conditionnelle « sachant X » divise une case par le total de X ; une probabilité jointe divise une case par le total général (fiche §3.6).

</details>
<details><summary>Indice 2</summary>

`table_14.loc["Gentoo", "Biscoe"]` lit une case ; `table_14["Biscoe"].sum()` donne le total d'une colonne, `table_14.loc["Chinstrap"].sum()` celui d'une ligne, `table_14.to_numpy().sum()` le total général. Pour f, compare chaque probabilité jointe au produit des deux marginales (fiche §3.6) : une seule case très différente suffit pour répondre. Pour g, relis l'encadré 🧮 « Rappel outil » de la fiche.

</details>
<details><summary>Indice 3</summary>

`p_gentoo_given_biscoe = table_14.loc["Gentoo", "Biscoe"] / table_14["Biscoe"].sum()`, et de même pour c à e. Pour f : avec `n = table_14.to_numpy().sum()`, `np.outer(table_14.sum(axis=1), table_14.sum(axis=0)) / n ** 2` est la table qu'on aurait si les variables étaient indépendantes ; compare-la à `table_14 / n`, en commençant par les cases nulles.

</details>

### Ex 3.15 — confusion_matrix à la manière de scikit-learn 🔨

<details><summary>Indice 1</summary>

Trois étapes : vérifier les entrées, choisir l'ordre des labels, compter les couples (vérité, prédiction). La ligne vient de la vérité, la colonne de la prédiction.

</details>
<details><summary>Indice 2</summary>

Sans `labels` : `np.unique(np.concatenate([y_true, y_pred]))` donne les labels triés, sans doublon. Un dictionnaire `index = {label: i for i, label in enumerate(labels.tolist())}` donne le numéro de chaque label ; un label absent de `index` doit lever une `ValueError`. Puis `C = np.zeros((k, k), dtype=int)` et, pour chaque couple, `C[index[t], index[p]] += 1`. Pour c, les labels sont triés : la ligne des Gentoo est la 3ᵉ, la colonne des Chinstrap la 2ᵉ.

</details>
<details><summary>Indice 3</summary>

`confusion_matrix` : la signature, le squelette, puis les lignes clés.
```python
def confusion_matrix(y_true, y_pred, labels=None):
    # 1. y_true, y_pred = _check_pair(y_true, y_pred): np.asarray, same lengths, not empty (ValueError otherwise)
    # 2. labels = np.asarray(labels) or, without labels, the sorted labels of y_true AND y_pred (hint 2)
    index = {label: i for i, label in enumerate(labels.tolist())}    # label -> row and column number
    # 3. C = a k x k array of zeros, dtype=int (k = number of labels)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        # 4. a label missing from index: ValueError
        C[index[t], index[p]] += 1                                     # row = truth, column = prediction
    # 5. return C
```

</details>

### Ex 3.16 — accuracy, precision, recall, F-beta et F1 (cas binaire) 🔨

<details><summary>Indice 1</summary>

Toutes ces mesures sont des fractions de TP, FP et FN (et de TN pour l'accuracy). Écris d'abord une fonction d'aide qui vérifie les labels et compte ces cases pour la classe `pos_label` ; les quatre mesures l'appellent.

</details>
<details><summary>Indice 2</summary>

Dans l'aide : `a, b = _check_pair(...)` (3.15) ; les labels présents, `np.unique(np.concatenate([a, b]))` ; plus de deux, ou deux dont aucun n'est `pos_label` : `ValueError`. Puis `t = a == pos_label`, `p = b == pos_label`, et `tp = np.sum(t & p)`, `fp = np.sum(~t & p)`, `fn = np.sum(t & ~p)`. Une petite fonction `_ratio(num, den, zero_division)` renvoie `zero_division` quand `den` vaut 0. `fbeta` vérifie d'abord `beta > 0` ; `accuracy` compare simplement `a == b`.

</details>
<details><summary>Indice 3</summary>

La fonction d'aide qui compte, son squelette et ses deux lignes clés :
```python
def _binary_counts(y_true, y_pred, pos_label):
    # 1. a, b = _check_pair(y_true, y_pred)  (3.15)
    # 2. present = the sorted labels of a AND b: more than two, or two without pos_label -> ValueError
    t, p = a == pos_label, b == pos_label                                   # truly positive, predicted positive
    return int(np.sum(t & p)), int(np.sum(~t & p)), int(np.sum(t & ~p))    # tp, fp, fn
```
Chaque mesure vérifie ensuite `average` (autre chose que `"binary"` : `NotImplementedError` jusqu'à 3.25), appelle cette aide, puis renvoie un rapport qui passe par `_ratio(num, den, zero_division)`, la valeur `zero_division` quand `den` vaut 0. Pour le F-beta, après le contrôle `beta > 0`, avec `b2 = beta ** 2` : le numérateur est `(1 + b2) * tp`, le dénominateur `(1 + b2) * tp + b2 * fn + fp`. `f1` appelle `fbeta` ; `accuracy` compare `a == b` après `_check_pair`.

</details>

### Ex 3.17 — La matrice à l'envers 🐛

<details><summary>Indice 1</summary>

Calcule d'abord les vraies valeurs avec tes fonctions : la sensibilité est le recall de la classe « malade », la precision se calcule aussi pour la classe « malade ». Puis compare-les au rapport du collègue.

</details>
<details><summary>Indice 2</summary>

Affiche `skm.confusion_matrix(test_truth, test_result)` et demande-toi dans quel ordre scikit-learn range les deux labels (fiche §3.7.2, 🕰️). Que contient alors chacune des variables `tn, fp, fn, tp` du collègue, et quelles mesures a-t-il donc calculées ? Pour la correction, ne dépends plus de cet ordre : travaille avec des masques booléens, `truth = y_true == positive` et `alarm = y_pred == positive`.

</details>
<details><summary>Indice 3</summary>

`true_sensitivity_17 = mylearn.metrics.recall(test_truth, test_result, pos_label="malade")`, et de même avec `precision`. Dans `screening_report_fixed`, avec les masques de l'indice 2 : `tp = np.sum(truth & alarm)` ; écris FN et FP de la même façon, en mettant `~` (« non ») devant le bon masque, puis les deux rapports en `float`. Autre correction : `skm.confusion_matrix(y_true, y_pred, labels=[negative, positive]).ravel()`, en trouvant d'abord le label négatif.

</details>

### Ex 3.18 — Tout positif, un seul positif : prédire les scores 🔮

<details><summary>Indice 1</summary>

Écris la matrice de confusion de chaque classifieur : combien de TP, de FP, de FN et de TN ?

</details>
<details><summary>Indice 2</summary>

Classifieur 1 : tous les positifs sont trouvés, et tous les négatifs deviennent de fausses alertes. Classifieur 2 : un seul vrai positif, aucune fausse alerte, les autres positifs sont manqués. Applique ensuite les formules : accuracy $\frac{TP + TN}{n}$, $F_1 = \frac{2\,TP}{2\,TP + FP + FN}$, balanced accuracy $\frac{\text{recall} + \text{spécificité}}{2}$.

</details>
<details><summary>Indice 3</summary>

Classifieur 1 : TP = 8, FP = 32, FN = 0, TN = 0. Classifieur 2 : TP = 1, FP = 0, FN = 7, TN = 32. Il ne reste que des fractions à calculer ; pour e, que vaut la spécificité du classifieur 1 ?

</details>

### Ex 3.19 — Le tableau de bord complet : classification_rates 🔨

<details><summary>Indice 1</summary>

Reprends les contrôles de labels de 3.16, compte les quatre cases une seule fois, puis remplis le dictionnaire dans l'ordre de la docstring, chaque rapport passant par la même petite fonction.

</details>
<details><summary>Indice 2</summary>

`t = a == pos_label`, `p = b == pos_label`, puis `tp`, `fn`, `fp`, `tn` avec `&` et `~`, convertis en `int` Python (les produits du MCC ne débordent pas). Une fonction interne `ratio(num, den)` renvoie `float(zero_division)` si `den` vaut 0. La balanced accuracy réutilise le recall et la spécificité ; le MCC a pour dénominateur la racine du produit des quatre sommes. Pour b, écris la precision avec « ham » comme classe positive : quelles cases de la matrice « spam » utilise-t-elle ?

</details>
<details><summary>Indice 3</summary>

Le squelette, puis les lignes clés :
```python
def classification_rates(y_true, y_pred, pos_label=1, zero_division=0.0):
    # 1. the label checks and the counts of 3.16 (your helper gives tp, fp, fn), then n and tn
    def ratio(num, den):                                  # the single rule for every division
        return float(num / den) if den != 0 else float(zero_division)
    # 2. recall_ and specificity first: balanced_accuracy reuses them
    # 3. the 13 keys, in the order of the docstring, every value through ratio(...), for example
    #    "mcc": ratio(tp * tn - fp * fn, np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
```
b) avec « ham » positif, la precision divise les « ham » bien prédits par toutes les prédictions « ham » : à quelles cases de la matrice « spam » correspondent-ils ?

</details>

### Ex 3.20 — Un seuil sur la nageoire : precision et recall en balance 🔬

<details><summary>Indice 1</summary>

Pour chaque seuil, les prédictions sont `(flipper >= t).astype(int)` ; mesure-les contre `is_gentoo` avec tes fonctions de 3.16. Range un dictionnaire par seuil dans une liste, puis fais-en un DataFrame.

</details>
<details><summary>Indice 2</summary>

Dans la boucle : precision, recall et F1 (tes fonctions, classe positive 1), `fn = np.sum((is_gentoo == 1) & (pred == 0))` et `fp = np.sum((is_gentoo == 0) & (pred == 1))`. Pour a, prends la ligne du seuil 205 : `table.set_index("threshold").loc[205]`. Pour b, `table.loc[table["f1"].idxmax(), "threshold"]` ; pour c, ajoute une colonne `cost = 10 * fn + fp`, puis `idxmin`.

</details>
<details><summary>Indice 3</summary>

Le squelette de `sweep_20`, puis les lignes clés :
```python
def sweep_20(thresholds):
    rows = []
    for t in thresholds:
        pred = (flipper >= t).astype(int)                 # >= : "Gentoo" from t mm on
        # one dict per threshold: threshold, precision, recall, f1 (your functions, positive class 1),
        # fn = np.sum((is_gentoo == 1) & (pred == 0)), and fp the other way round
    return pd.DataFrame(rows)                             # one row per dict
```
Pour b et c, attention : `idxmax` et `idxmin` renvoient le **label de la ligne**, pas le seuil ; lis ensuite la colonne `threshold` de cette ligne avec `.loc`.

</details>

### Ex 3.21 — Simuler le dépistage : la prévalence fait la precision 🔬

<details><summary>Indice 1</summary>

La simulation suit la recette de l'énoncé à la lettre (deux appels à `rng.random`, dans cet ordre). La precision théorique se lit sur l'arbre des fréquences naturelles de la fiche : vrais positifs divisés par tous les positifs.

</details>
<details><summary>Indice 2</summary>

En parts de la population : vrais positifs = sensibilité × prévalence ; faux positifs = (1 − spécificité) × (1 − prévalence). Pour c, la precision vaut 0,5 quand ces deux quantités sont égales : une équation du premier degré en $p$. Pour d, tire `u2` juste après la simulation, avec le même `rng`, applique la même règle, puis garde les personnes positives aux deux tests (`positive & positive_2`).

</details>
<details><summary>Indice 3</summary>

`simulate_screening` suit la recette de l'énoncé ; la ligne clé est celle du test.
```python
def simulate_screening(n, prevalence, sensitivity, specificity, rng):
    # 1. sick = rng.random(n) < prevalence    (the sick people first)
    # 2. u = rng.random(n)                    (then ONE array u, with the same generator)
    positive = np.where(sick, u < sensitivity, u < 1 - specificity)    # the rule of the sick, or of the healthy
    # 3. return sick, positive
```
c) résous $0{,}99\,p = 0{,}02\,(1 - p)$. d) tire `u2 = rng.random(100_000)` juste après la simulation de a, avec le même `rng` ; applique-lui la même règle (`np.where`) pour obtenir `positive_2` ; les personnes positives aux deux tests sont `positive & positive_2`, et la precision est la part de malades parmi elles.

</details>

### Ex 3.22 — Vérifier avec scikit-learn : classification_report et affichages 📦

<details><summary>Indice 1</summary>

`print(skm.classification_report(species, expert_pred, digits=3))` affiche le tableau : une ligne par espèce, une colonne par mesure, et les moyennes sur les dernières lignes.

</details>
<details><summary>Indice 2</summary>

`skm.ConfusionMatrixDisplay.from_predictions(species, expert_pred, normalize=...)`, puis `plt.show()`. D'après sa documentation, `normalize` accepte `"true"`, `"pred"` ou `"all"` : laquelle divise chaque **ligne** (une vraie espèce) par son total ?

</details>
<details><summary>Indice 3</summary>

`report_22 = skm.classification_report(species, expert_pred, output_dict=True)` est un dictionnaire de dictionnaires : une clé par espèce, puis `"accuracy"`, `"macro avg"` et `"weighted avg"` (affiche `report_22.keys()`). a) `report_22["Gentoo"]["recall"]` ; b) et c) la même lecture, dans les entrées des deux moyennes : la colonne du F1 s'appelle `"f1-score"`.

</details>

### Ex 3.23 — Lire la documentation de sklearn.metrics 🛠️

<details><summary>Indice 1</summary>

`help(skm.precision_score)` affiche la documentation dans le notebook : cherche la section `Parameters`, puis le paragraphe du paramètre concerné. Sur le site de scikit-learn, vérifie que la version affichée est la 1.6.

</details>
<details><summary>Indice 2</summary>

a) Que vaut la precision quand aucune prédiction n'est positive ? C'est `zero_division` qui décide. b) D'après la documentation, `labels` peut servir à choisir un sous-ensemble des labels : que deviennent les échantillons des autres ? c) Quelle est la valeur par défaut de `pos_label`, et existe-t-elle parmi `"spam"` et `"ham"` ? d) Lis la phrase qui dit sur quoi `normalize` divise : les vraies classes, les classes prédites ou toute la population. e) Chaque échantillon compte avec son poids. f) Sans `labels`, dans quel ordre scikit-learn range-t-il les classes ?

</details>
<details><summary>Indice 3</summary>

a) aucune prédiction n'est positive : la precision vaut $\frac{0}{0}$, et la documentation dit que `zero_division` fixe la valeur renvoyée dans ce cas ; ici, `1.0`. Pour les autres : b) que fait la documentation des échantillons dont la vérité ou la prédiction n'est pas dans `labels` ? Applique ta lecture aux quatre couples (vérité, prédiction) de l'appel ; c) quelle est la valeur par défaut de `pos_label`, et figure-t-elle parmi `"spam"` et `"ham"` ? Quelle erreur Python signale une valeur d'argument invalide ? d) `"pred"` renvoie aux classes **prédites** : dans la matrice de scikit-learn, sont-elles rangées en lignes ou en colonnes ? e) $\frac{1 \times 1 + 1 \times 0 + 2 \times 1}{1 + 1 + 2}$ ; f) les classes sont rangées dans l'ordre trié des labels ; pour chacune, parmi les échantillons prédits dans cette classe, quelle part est juste ?

</details>

### Ex 3.24 — Courbe ROC et AUC 🔨

<details><summary>Indice 1</summary>

La courbe se construit en descendant la liste des échantillons triés par score décroissant : chaque positif fait monter, chaque négatif fait avancer vers la droite (fiche, mini-exemple). Des scores égaux ne donnent qu'un seul point.

</details>
<details><summary>Indice 2</summary>

Vérifie d'abord les entrées : mêmes longueurs, pas de NaN (`np.isnan`), exactement deux classes (`np.unique`) dont `pos_label`. Après `order = np.argsort(-scores, kind="mergesort")` et `pos = positive[order]`, `np.cumsum(pos)` compte les TP et `np.cumsum(~pos)` les FP à chaque rang. Le dernier rang de chaque groupe de scores égaux se repère avec `np.flatnonzero(np.diff(sorted_scores))`, plus le tout dernier rang. Ajoute (0, 0) au début (seuil `np.inf`), puis divise par le nombre total de positifs et de négatifs. `auc` : la somme de `np.diff(x) * (y[1:] + y[:-1]) / 2`, changée de signe si `x` décroît. Pour b, compare $\max(\text{AUC}, 1 - \text{AUC})$ des quatre mesures.

</details>
<details><summary>Indice 3</summary>

Une fonction d'aide trie et cumule une fois pour toutes (elle resservira en 3.26) ; son squelette et ses deux lignes clés :
```python
def _sorted_counts(y_true, y_score, pos_label):
    # 1. checks: same lengths, no NaN (np.isnan), exactly two classes (np.unique), pos_label among them
    # 2. sort by decreasing score (hint 2): s = the sorted scores, pos = the sorted booleans "is positive"
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]            # last rank of each group of equal scores
    return np.cumsum(pos)[last], np.cumsum(~pos)[last], s[last]     # tps, fps, thresholds (decreasing)
```
`roc_curve` : ajoute 0 devant `tps` et devant `fps` (le point $(0, 0)$, de seuil `np.inf`), puis divise par les totaux `tps[-1]` et `fps[-1]`. `auc` : après les contrôles, la somme des trapèzes `np.diff(x) * (y[1:] + y[:-1]) / 2`, changée de signe si `x` décroît. `roc_auc` : `auc` des deux premiers résultats de `roc_curve`.

</details>

### Ex 3.25 — Moyennes macro, micro et pondérée 🔨

<details><summary>Indice 1</summary>

Avec plusieurs classes, compte TP, FP et FN **pour chaque classe**, puis combine selon `average`. Ta matrice de confusion donne tout d'un coup.

</details>
<details><summary>Indice 2</summary>

`C = confusion_matrix(a, b, labels=present)` ; `tp = np.diag(C)` ; `fp = C.sum(axis=0) - tp` (le reste de chaque colonne) ; `fn = C.sum(axis=1) - tp` (le reste de chaque ligne) ; `support = C.sum(axis=1)`. Calcule les valeurs par classe avec une division qui applique `zero_division` case par case, puis : `None` → le tableau ; `"macro"` → `np.mean` ; `"weighted"` → `np.average(values, weights=support)` ; `"micro"` → additionne d'abord `tp`, `fp` et `fn`, puis une seule division. Pour d, calcule les trois baisses.

</details>
<details><summary>Indice 3</summary>

Une seule fonction d'aide pour les quatre mesures ; son squelette :
```python
def _score(y_true, y_pred, pos_label, average, zero_division, kind, beta=1.0):
    # 1. an unknown average -> ValueError
    # 2. tp, fp, fn, support: one value per class from your confusion_matrix (hint 2),
    #    or arrays of a single cell for "binary" (the counts of 3.16)
    # 3. "micro": pool the counts first, tp.sum(keepdims=True), and the same for fp and fn
    # 4. values = the precision, the recall or the F-beta of every cell, with the division below
    # 5. None -> values; "macro" -> np.mean; "weighted" -> np.average(..., weights=support); otherwise values[0]
```
La division qui applique `zero_division` case par case tient en deux lignes clés, `num` et `den` étant des tableaux de `float` : `out = np.full(num.shape, float(zero_division))`, puis `np.divide(num, den, out=out, where=den != 0)`. Les quatre fonctions publiques n'ont plus qu'à appeler l'aide avec leur `kind`. Pour d, calcule la baisse de chacune des trois moyennes d'un classifieur à l'autre.

</details>

### Ex 3.26 — Courbe precision-recall et average precision 🔨

<details><summary>Indice 1</summary>

Même tri et mêmes cumuls que pour la ROC ; seuls les rapports changent : precision = TP / (TP + FP), recall = TP / (nombre total de positifs). Il reste à ranger les points dans l'ordre de scikit-learn.

</details>
<details><summary>Indice 2</summary>

Avec `tps`, `fps` et les seuils (scores distincts, **décroissants**) de ta fonction d'aide de 3.24 : `precision = tps / (tps + fps)`, `recall = tps / tps[-1]`. scikit-learn range les seuils par ordre **croissant** : inverse les trois tableaux (`[::-1]`), puis ajoute le point final (precision 1, recall 0) aux deux premiers. L'AP repart du seuil le plus haut, avec $R_0 = 0$ : la somme de `np.diff(r) * p[1:]`. Pour b et c, `np.argsort(-score)[:k]` donne les k plus hauts scores ; la precision est la part de fraudes parmi eux.

</details>
<details><summary>Indice 3</summary>

Avec ta fonction d'aide de 3.24, le squelette et les lignes clés :
```python
def precision_recall_curve(y_true, y_score, pos_label=1):
    tps, fps, thresholds = _sorted_counts(y_true, y_score, pos_label)   # decreasing thresholds
    # precision = tps / (tps + fps), recall = tps / tps[-1], one value per threshold
    # scikit-learn's order: reverse the three arrays ([::-1]), then add precision 1 and recall 0 at the end (np.r_)


def average_precision(y_true, y_score, pos_label=1):
    # 1. precision, recall from your precision_recall_curve
    # 2. read both from the highest threshold (reverse them): r[0] is then 0, the closing point
    return float(np.sum(np.diff(r) * p[1:]))      # each gain of recall times the precision reached
```
b) et c) `np.argsort(-score)[:k]` donne les indices des k plus hauts scores ; la precision parmi eux est la part de fraudes, la moyenne de `y_26` sur ces indices (pas la part de toutes les fraudes que ces k alertes trouvent : ce serait un recall).

</details>

### Ex 3.27 — ROC ou PR ? Lire les courbes d'un problème déséquilibré 📈

<details><summary>Indice 1</summary>

Sur la ROC, l'abscisse est le taux de faux positifs et l'ordonnée le recall ; sur la courbe PR, l'abscisse est le recall et l'ordonnée la precision. Les lignes pointillées marquent le recall 0,5.

</details>
<details><summary>Indice 2</summary>

b) Sur le zoom, suis la ligne horizontale du recall 0,5 jusqu'aux courbes, puis descends lire l'abscisse. c) Sur le graphique de droite, suis la ligne verticale du recall 0,5. d) Un classifieur au hasard n'utilise pas les scores : ses alertes sont des cas tirés au hasard dans la population. Quelle part de positifs contiennent-elles, en moyenne ?

</details>
<details><summary>Indice 3</summary>

a) sur le graphique de gauche, les deux courbes ROC se superposent presque : `True`. Pour les autres : b) sur le zoom, l'abscisse du point où les courbes coupent la ligne pointillée du recall 0,5, au centième près ; c) sur le graphique de droite, la hauteur de chaque courbe sur la ligne verticale du recall 0,5, au dixième près, d'abord pour 2 % de positifs ; d) des alertes tirées au hasard contiennent, en moyenne, la même part de positifs que toute la population, quel que soit leur nombre : relis cette part dans l'énoncé, et écris-la comme une proportion, pas en pourcentage ; e) laquelle des deux courbes change d'une population à l'autre ? Pour les notes : négatifs = 99,5 % de 200 000 ; fausses alertes = FPR × négatifs ; positifs trouvés = 0,5 × positifs ; precision = trouvés / (trouvés + fausses alertes).

</details>

### Ex 3.28 — Calibration : quand la météo annonce 70 % 🔨

<details><summary>Indice 1</summary>

Le diagramme de fiabilité découpe [0, 1] en intervalles égaux ; dans chaque intervalle non vide, deux moyennes : celle des probabilités annoncées, et celle des résultats (la fréquence des 1). Le score de Brier est une moyenne de carrés.

</details>
<details><summary>Indice 2</summary>

Vérifie les entrées : résultats 0 ou 1 (`np.isin`), probabilités dans [0, 1] et sans NaN, mêmes longueurs, `n_bins` entier au moins égal à 1. `edges = np.linspace(0, 1, n_bins + 1)`, puis `bins = np.searchsorted(edges[1:-1], p)` donne le numéro de l'intervalle de chaque probabilité. `np.bincount(bins, minlength=n_bins)` compte les cas de chaque intervalle ; avec `weights=y` ou `weights=p`, il somme les résultats ou les probabilités. Ne garde que les intervalles où le compte est positif. Pour b, un masque : `(forecast >= 0.65) & (forecast < 0.75)`.

</details>
<details><summary>Indice 3</summary>

Le squelette, puis les lignes clés :
```python
def calibration_curve(y_true, y_prob, n_bins=10):
    # 1. checks (hint 2), then t and p as float arrays
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.searchsorted(edges[1:-1], p)                         # a value on an inner edge goes to the lower bin
    positives = np.bincount(bins, weights=t, minlength=n_bins)     # number of 1s in each bin
    # 2. the same without weights (the counts) and with weights=p (the sums of the probabilities)
    # 3. keep the bins whose count is > 0, and return (share of 1s, mean probability) for them
```
`brier_score` : la moyenne des carrés `(p - t) ** 2`, après les mêmes contrôles. b) la fréquence de la pluie ces jours-là est la moyenne de `rain` sur le masque de l'indice 2, pour A, puis pour B.

</details>

### Ex 3.29 — Recall ≥ 0,99 au meilleur prix 🏆

<details><summary>Indice 1</summary>

Explore d'abord la validation : pour chaque modèle, l'AP, et les scores des fraudes les plus basses (`np.sort(...)[:10]`). Pour un recall de 0,99, c'est le **bas** de la distribution des scores des fraudes qui compte, pas le haut.

</details>
<details><summary>Indice 2</summary>

Le modèle A laisse une partie des fraudes au milieu des transactions normales : pour les attraper, il faudrait alerter presque partout. Avec B, un seuil qui garde 99 % des fraudes de validation vise juste… sur la validation. Mesure combien ce seuil varie d'un échantillon d'environ 500 fraudes à l'autre (bootstrap du ch. 2 : rééchantillonne les scores des fraudes, recalcule le quantile à 1 %), et prends un seuil un peu plus bas.

</details>
<details><summary>Indice 3</summary>

Le squelette :
```python
def choose_29(val):
    column = ...                                              # "score_a" or "score_b": the model you keep
    scores = val.loc[val["fraud"] == 1, column].to_numpy()    # the scores of its validation frauds
    # threshold = a low quantile of these scores, a little BELOW the 1 % one (the margin): np.quantile(scores, q)
    return column, float(threshold)
```
Le modèle : celui dont les fraudes les plus basses restent au-dessus de la masse des transactions normales (indice 2). La marge : assez pour couvrir la variabilité que mesure le bootstrap, pas davantage, sinon la precision passe sous 0,10 ; vérifie ton choix sur la validation, jamais sur le test. Plus stable, si les scores des fraudes suivent à peu près une loi normale (trace leur histogramme) : leur moyenne moins $z$ écarts-types, où $z$ = `-scipy.stats.norm.ppf(q)` pour le quantile visé ; ce seuil utilise toutes les fraudes, et pas seulement les plus basses.

</details>
