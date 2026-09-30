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

Une confiance de 80 % est méritée si, parmi les cas jugés « à 80 % », environ 80 % donnent raison. Un score `predict_proba` n'est une vraie probabilité que si le modèle a cette propriété : on dit qu'il est calibré.

</details>

### 3.Q2 — Fléchettes : pourquoi chaque point du mur doit être aussi probable

<details><summary>Indice 1</summary>

Une probabilité égale à un rapport d'aires suppose que la position de l'impact suit la loi uniforme du ch. 2.

</details>
<details><summary>Indice 2</summary>

Imagine un lanceur qui vise le centre : une petite tache au centre et une grande tache dans un coin recevraient-elles des fléchettes en proportion de leur aire ? Pour la question 4, calcule ce que deviennent les aires quand toutes les longueurs doublent.

</details>
<details><summary>Indice 3</summary>

Doubler les longueurs multiplie toutes les aires par 4, et les rapports restent les mêmes. Le fond et les taches couvrent tout le mur sans se chevaucher : leurs probabilités s'additionnent à 1. Pour la question 6, divise le nombre de fléchettes dans A par le nombre total.

</details>

### 3.Q3 — P(A|B) ou P(B|A) ?

<details><summary>Indice 1</summary>

Dans $P(A \mid B)$, ce qui est **après** la barre est ce qu'on sait déjà ; ce qui est **avant**, ce dont on cherche la probabilité.

</details>
<details><summary>Indice 2</summary>

Un exemple voisin : « parmi les e-mails avec lien, 20 % sont des spams » (fiche §3.5) : on sait que l'e-mail a un lien, donc « lien » va après la barre, et l'on écrit $P(\text{spam} \mid \text{lien}) = 0{,}2$. Pour la question 4, demande-toi quelle espèce peut avoir des nageoires de plus de 210 mm, et si tous les Gentoo en ont d'aussi longues.

</details>
<details><summary>Indice 3</summary>

$P(A \mid B) = \frac{P(A, B)}{P(B)}$ et $P(B \mid A) = \frac{P(A, B)}{P(A)}$ : même numérateur, dénominateurs différents. Les Gentoo descendent jusqu'à environ 203 mm, alors que les autres espèces dépassent à peine 210 mm.

</details>

### 3.Q4 — Jointe = conditionnelle × simple

<details><summary>Indice 1</summary>

Suis les fléchettes en deux étapes : d'abord combien tombent dans B, puis quelle part de celles-ci tombe aussi dans A.

</details>
<details><summary>Indice 2</summary>

Dans B : 40 % de 200. Dans A et B : le quart de ce nombre. $P(A, B)$ est ce dernier nombre divisé par 200.

</details>
<details><summary>Indice 3</summary>

$P(A, B) = P(A \mid B)\,P(B) = 0{,}25 \times 0{,}4$. Pour des événements indépendants, la règle du produit devient $P(A, B) = P(A)\,P(B)$.

</details>

### 3.Q5 — D'où vient le mot « marginale »

<details><summary>Indice 1</summary>

Relis le début du §3.6 du livre et l'encadré `pd.crosstab` de la fiche.

</details>
<details><summary>Indice 2</summary>

Une marge est un total de ligne ou de colonne. Pour la question 5, cherche deux tables différentes qui ont les mêmes totaux (par exemple, ajoute 1 à deux cases opposées en diagonale et retire 1 aux deux autres : aucun total ne bouge).

</details>
<details><summary>Indice 3</summary>

$P(A) = \sum_b P(A, B = b)$ : la marge est la somme de la ligne. Deux tables comme `[[2, 0], [0, 2]]` et `[[1, 1], [1, 1]]` ont les mêmes marges. En pandas : `value_counts(normalize=True)`.

</details>

### 3.Q6 — Vérité terrain et prédiction

<details><summary>Indice 1</summary>

Relis le §3.7.1 de la fiche : vérité terrain, prédiction, classe positive, frontière de décision.

</details>
<details><summary>Indice 2</summary>

Pour la question 4, calcule la precision et le recall du filtre anti-spam de la fiche en prenant « normal » comme classe positive : qu'est-ce qui devient TP, FP, FN, TN ?

</details>
<details><summary>Indice 3</summary>

Échanger les rôles échange TP avec TN et FP avec FN : la precision devient la NPV, le recall devient la spécificité. Pour la question 6, pense aux étiquettes posées à la main.

</details>

### 3.Q7 — Lire une matrice de confusion (et vérifier ses axes)

<details><summary>Indice 1</summary>

scikit-learn trie les étiquettes (0 puis 1) et met la vérité en lignes, la prédiction en colonnes.

</details>
<details><summary>Indice 2</summary>

La case en haut à gauche est donc (vérité 0, prédiction 0), celle en bas à droite (vérité 1, prédiction 1). Les malades sont la ligne du bas, les prédictions « malade » la colonne de droite.

</details>
<details><summary>Indice 3</summary>

`[[TN, FP], [FN, TP]]`. Le collègue, qui lit à la manière du livre, prend la case en haut à gauche pour les TP.

</details>

### 3.Q8 — Faux positif ou faux négatif : lequel coûte le plus ?

<details><summary>Indice 1</summary>

Pour chaque situation, écris ce qui se passe concrètement après un faux positif, puis après un faux négatif.

</details>
<details><summary>Indice 2</summary>

La precision juge les alertes (les faux positifs), le recall juge les oublis (les faux négatifs). Pour les figurines aux yeux peints, attention : le positif est « yeux présents ».

</details>
<details><summary>Indice 3</summary>

Fraude, figurine interdite, dépistage : l'oubli coûte le plus (recall). Anti-spam, figurine sans yeux expédiée : la fausse alerte coûte le plus (precision).

</details>

### 3.Q9 — L'accuracy face aux classes déséquilibrées

<details><summary>Indice 1</summary>

Compte les bonnes réponses d'un modèle qui répond toujours « pas de fraude ».

</details>
<details><summary>Indice 2</summary>

Quelles transactions ce modèle classe-t-il bien, et lesquelles rate-t-il ? L'accuracy compte les premières, le recall ne regarde que les fraudes. Pour la question 4, souviens-toi du bootstrap du ch. 2 : l'incertitude d'une proportion dépend du nombre d'exemples.

</details>
<details><summary>Indice 3</summary>

Accuracy 0,99, recall 0. L'écart-type d'une proportion $p$ mesurée sur $n$ exemples vaut environ $\sqrt{p(1 - p)/n}$. La mesure qui donne le même poids aux deux classes est la moyenne des recalls de chaque classe.

</details>

### 3.Q10 — Precision ou recall : le moteur de recherche du wiki

<details><summary>Indice 1</summary>

Une page renvoyée et pertinente est un TP ; renvoyée et non pertinente, un FP ; pertinente mais non renvoyée, un FN ; ni l'un ni l'autre, un TN.

</details>
<details><summary>Indice 2</summary>

Les 25 pages renvoyées sont des TP ou des FP ; les 40 pages pertinentes sont des TP ou des FN. Commence par les TP, déduis-en FP et FN, puis les TN à partir du total de 500 pages.

</details>
<details><summary>Indice 3</summary>

Precision $= \frac{20}{25}$, recall $= \frac{20}{40}$. L'accuracy est dominée par les TN, très nombreux. Pour la question 6 : qui paie le plus cher un document manqué, et qui paie le plus cher un mauvais résultat en haut de page ?

</details>

### 3.Q11 — Tricher sur une seule mesure

<details><summary>Indice 1</summary>

Relis le §3.7.10 du livre (figures 3.35 à 3.38) : deux frontières extrêmes.

</details>
<details><summary>Indice 2</summary>

Tout déclarer positif : combien de TP, de FP, de FN ? Ne déclarer positif qu'un seul point, juste : même question. Pour les questions 4 et 5, écris ce que devient chaque case quand le seuil monte.

</details>
<details><summary>Indice 3</summary>

Quand le seuil monte, TP et FP ne peuvent que baisser ; TP + FN ne change pas. La precision peut baisser si l'on perd un TP sans perdre de FP : essaie avec les six scores de la fiche (seuils 0,4 puis 0,7).

</details>

### 3.Q12 — F1 et test « fiable à 99 % »

<details><summary>Indice 1</summary>

Le F1 est la moyenne harmonique de la precision et du recall (fiche §3.7.11).

</details>
<details><summary>Indice 2</summary>

$F_1 = \frac{2PR}{P + R}$. Pour la question 4, écris l'information donnée avec « sachant », puis celle que veut le patient.

</details>
<details><summary>Indice 3</summary>

« Détecte 99 % des malades » donne $P(\text{positif} \mid \text{malade})$. Le patient veut $P(\text{malade} \mid \text{positif})$ : il faut la prévalence et le taux de faux positifs (fiche §3.8).

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

$\frac{16}{32} \times \frac{15}{31}$. En NumPy : `rng.choice(32, size=5, replace=False)`.

</details>

### 3.R2 — 0B : ensembles, intersection, union, complémentaire

<details><summary>Indice 1</summary>

Dessine deux cercles qui se chevauchent, S et M, et place 12 dans la partie commune.

</details>
<details><summary>Indice 2</summary>

Sport seulement : 40 − 12 ; musique seulement : 30 − 12. L'union est la somme des trois morceaux.

</details>
<details><summary>Indice 3</summary>

$|S \cup M| = 40 + 30 - 12$. Pour l'indépendance, compare $P(S \cap M)$ au produit $P(S)\,P(M)$.

</details>

### 3.R3 — 0A : pandas, `value_counts(normalize=True)` et `groupby`

<details><summary>Indice 1</summary>

`value_counts` compte les valeurs d'une colonne ; `normalize=True` divise par le total.

</details>
<details><summary>Indice 2</summary>

`groupby("species")` coupe le tableau en un groupe par espèce, puis applique la fonction à chaque groupe.

</details>
<details><summary>Indice 3</summary>

`df.groupby("species")["sex"].value_counts(normalize=True)` ; pour filtrer, un masque booléen : `df[df["island"] == ...]`.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 3.1 — Fléchettes et aires : probabilités simples et conditionnelles

<details><summary>Indice 1</summary>

Dessine le mur et les deux rectangles à l'échelle, puis calcule trois aires : A, B, et leur partie commune.

</details>
<details><summary>Indice 2</summary>

La partie commune est le rectangle où les deux conditions sur $x$ et les deux conditions sur $y$ sont vraies à la fois : $x$ entre 1,5 et 2,5, $y$ entre 1 et 2. Une probabilité simple ou jointe divise par l'aire du mur ; une conditionnelle divise par l'aire de ce qu'on sait.

</details>
<details><summary>Indice 3</summary>

Aires : mur 10, A 3, B 2, commune 1. $P(A \mid B) = \frac{1}{2}$, $P(B \mid A) = \frac{1}{3}$. Pour f, $P(A \text{ ou } B) = P(A) + P(B) - P(A, B)$. Pour j, compare $P(A, B)$ à $P(A)\,P(B)$.

</details>

### Ex 3.2 — Les 20 points : matrice de confusion et quatre mesures

<details><summary>Indice 1</summary>

Pour chaque point, le couple (vérité, prédiction) dit dans quelle case il tombe : (1, 1) TP, (0, 1) FP, (1, 0) FN, (0, 0) TN.

</details>
<details><summary>Indice 2</summary>

Fais un tableau 2 × 2 et une croix par point. Vérifie que le total fait 20 et que TP + FN vaut le nombre de 1 dans la ligne « vérité ». Pour e, scikit-learn met la vérité 0 en première ligne.

</details>
<details><summary>Indice 3</summary>

Tu dois trouver 10 positifs et 11 prédictions positives. Matrice de scikit-learn : `[[TN, FP], [FN, TP]]`. $F_1 = \frac{2\,TP}{2\,TP + FP + FN}$.

</details>

### Ex 3.3 — Le glacier : jointes, marginales et conditionnelles

<details><summary>Indice 1</summary>

Une probabilité jointe divise une case par le total général ; une conditionnelle divise une case par le total de ce qu'on sait déjà.

</details>
<details><summary>Indice 2</summary>

« Sachant un cornet » : divise par le total de la colonne « cornet » (80). « Sachant la vanille » : divise par le total de la ligne « vanille » (60).

</details>
<details><summary>Indice 3</summary>

$P(V, C) = \frac{42}{150}$, $P(V \mid C) = \frac{42}{80}$, $P(C \mid V) = \frac{42}{60}$, $P(\text{chocolat} \mid \text{pot}) = \frac{52}{70}$. Pour g, compare $\frac{42}{150}$ à $\frac{60}{150} \times \frac{80}{150}$.

</details>

### Ex 3.4 — Règle du produit et formule des probabilités totales

<details><summary>Indice 1</summary>

Toutes les questions partent de la définition $P(A \mid B) = \frac{P(A, B)}{P(B)}$ et du fait que $P(A, B) = P(B, A)$ ; la question 3 utilise en plus l'additivité des aires (deux morceaux sans partie commune).

</details>
<details><summary>Indice 2</summary>

Pour 2, égalise les deux écritures de $P(A, B)$. Pour 3, découpe la tache A en deux morceaux sans partie commune : ce qui est dans B, et ce qui n'y est pas. Pour 4, découpe A en $K$ morceaux au lieu de deux ; pour la vérification, la ligne « femelle » et la ligne des totaux de la table de la fiche suffisent.

</details>
<details><summary>Indice 3</summary>

$\text{aire}(A) = \text{aire}(A \cap B) + \text{aire}(A \cap \text{non } B)$ ; divise par l'aire du mur, puis applique la règle du produit à chaque terme. Pour 5, remplace $P(A, B)$ par $P(A)\,P(B)$ dans la définition.

</details>

### Ex 3.5 — Toutes les mesures du tableau récapitulatif

<details><summary>Indice 1</summary>

Écris d'abord la matrice avec les totaux des lignes (positifs réels : 50, négatifs réels : 950) et des colonnes (prédictions positives : 100, négatives : 900).

</details>
<details><summary>Indice 2</summary>

Chaque mesure est une case divisée par un total : les **taux** (recall, FNR, spécificité, FPR) divisent par une ligne, les **valeurs prédictives** (precision, FDR, NPV, FOR) par une colonne. Aide-toi du tableau de la fiche (§3.7.9).

</details>
<details><summary>Indice 3</summary>

Spécificité $= \frac{890}{950}$, NPV $= \frac{890}{900}$, FPR $= \frac{60}{950}$, FDR $= \frac{60}{100}$. MCC : numérateur $40 \times 890 - 60 \times 10$, dénominateur $\sqrt{100 \times 50 \times 950 \times 900}$.

</details>

### Ex 3.6 — Trois espèces : moyennes macro, micro et pondérée

<details><summary>Indice 1</summary>

Chaque espèce devient tour à tour la classe positive. Pour l'Adélie, que sont TP, FP et FN dans la matrice ?

</details>
<details><summary>Indice 2</summary>

TP : la case de la diagonale. FP : le reste de la **colonne** (des manchots prédits dans cette espèce à tort). FN : le reste de la **ligne**. Le support est le total de la ligne.

</details>
<details><summary>Indice 3</summary>

Adélie : TP = 41, FP = 7, FN = 4. Macro : moyenne simple des trois valeurs ; pondérée : moyenne avec les poids 45, 20 et 35 ; micro : additionne TP, FP et FN des trois espèces avant de calculer. Calcule les moyennes avec les fractions exactes ($\frac{82}{93}$…), pas avec les F1 arrondis.

</details>

### Ex 3.7 — Le test « fiable à 99 % » dans une ville à 1 % de malades

<details><summary>Indice 1</summary>

Construis l'arbre des fréquences naturelles : pars de toute la population, sépare malades et personnes saines, puis applique le test à chaque groupe (fiche §3.8).

</details>
<details><summary>Indice 2</summary>

Les TP et FN viennent des malades (99 % et 1 % de ce groupe) ; les TN et FP viennent des personnes **saines** (99 % et 1 % de ce groupe, pas de la ville entière).

</details>
<details><summary>Indice 3</summary>

500 malades et 49 500 personnes saines : TP = 495, FP = 495. Precision $= \frac{TP}{TP + FP}$, NPV $= \frac{TN}{TN + FN}$. Pour la partie 2, les comptages du livre sont TP = 99, FN = 1, FP = 198, TN = 9 702 ; compare les dénominateurs de la spécificité et de la NPV.

</details>

### Ex 3.8 — F1, moyenne harmonique : pourquoi elle punit le maillon faible

<details><summary>Indice 1</summary>

Toutes les questions se ramènent à des calculs de fractions : mets au même dénominateur.

</details>
<details><summary>Indice 2</summary>

Pour 2, $(a + b)^2 - 4ab = (a - b)^2$. Pour 3, suppose $a \le b$ et compare $H$ à $a$, puis majore le dénominateur $a + b$. Pour 5, calcule les deux $F_2$, puis les deux $F_1$, avec la forme en comptages : que pèse un FN au dénominateur, et un FP ? Pour 6, calcule les durées de l'aller et du retour sur une distance $d$.

</details>
<details><summary>Indice 3</summary>

$H - a = \frac{a(b - a)}{a + b}$ et $H = \frac{2ab}{a + b} \le \frac{2ab}{b}$. Pour 5, multiplie le numérateur et le dénominateur par $\frac{(TP + FP)(TP + FN)}{TP}$.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 3.9 — Precision et recall expliqués à une médecin

<details><summary>Indice 1</summary>

La médecin connaît déjà les deux notions sous d'autres noms : lesquels ?

</details>
<details><summary>Indice 2</summary>

Recall = sensibilité ; precision = valeur prédictive positive. Donne pour chacune une conséquence concrète d'un score faible.

</details>
<details><summary>Indice 3</summary>

Structure possible : une phrase pour le recall (et les faux négatifs), une pour la precision (et les faux positifs), une sur la prévalence, qu'elle connaît.

</details>

### Ex 3.10 — Dépistage de masse : que dire à une personne testée positive ?

<details><summary>Indice 1</summary>

Commence par les chiffres : l'arbre des fréquences naturelles, comme en 3.7.

</details>
<details><summary>Indice 2</summary>

500 malades et 99 500 personnes saines. Le SMS doit donner le résultat, un ordre de grandeur de sa fiabilité, et la marche à suivre.

</details>
<details><summary>Indice 3</summary>

Environ une personne positive sur sept est malade. Pour la question 6, cherche ce que dit le RGPD des données de santé (article 9).

</details>

### Ex 3.11 — Fawcett (2006) : une introduction à l'analyse ROC

<details><summary>Indice 1</summary>

Les réponses se trouvent dans les sections 1 (introduction), 3 (espace ROC), 4 (courbes), 5 (construction efficace), 7 (AUC) et 9 (plus de deux classes).

</details>
<details><summary>Indice 2</summary>

Pour la question 4, regarde les dénominateurs du TPR et du FPR : dans quelle classe réelle sont-ils calculés ? Attention, la matrice de l'article met les classes réelles en colonnes.

</details>
<details><summary>Indice 3</summary>

AUC = probabilité qu'un positif tiré au hasard ait un score plus élevé qu'un négatif tiré au hasard ; test de Wilcoxon ; $\text{Gini} + 1 = 2 \times \text{AUC}$. Pour les ex-æquo, on émet un seul point après tout le groupe.

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

## Notebook

Les indices des exercices du notebook (3.12 à 3.29) seront ajoutés à la prochaine session de génération, avec le notebook complet.
