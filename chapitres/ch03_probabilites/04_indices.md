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

$P(A \mid B) = \frac{P(A, B)}{P(B)}$ et $P(B \mid A) = \frac{P(A, B)}{P(A)}$ : même numérateur, dénominateurs différents. Une partie des Gentoo ont une nageoire de 210 mm ou moins, alors que les autres espèces dépassent à peine 210 mm.

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

```python
def estimate_pi(n, rng):
    points = 2 * rng.random((n, 2))
    inside = ((points - 1) ** 2).sum(axis=1) <= 1
    return float(4 * inside.mean())
```
a) le même calcul avec `rng = np.random.default_rng(3)` et $n = 10\,000$ ; b) `rng = np.random.default_rng(12)`, puis `np.std([estimate_pi(10_000, rng) for _ in range(200)])`.

</details>

### Ex 3.13 — Deux disques : P(A|B) = P(B|A) ? 🔮

<details><summary>Indice 1</summary>

Écris les deux probabilités conditionnelles comme des rapports d'aires (fiche §3.4) : qu'ont-elles en commun, et en quoi diffèrent-elles ?

</details>
<details><summary>Indice 2</summary>

$P(A \mid B) = \frac{\text{aire}(A \cap B)}{\text{aire}(B)}$ et $P(B \mid A) = \frac{\text{aire}(A \cap B)}{\text{aire}(A)}$ : compare leurs numérateurs, puis leurs dénominateurs. Écris aussi leur rapport : que devient la partie commune ?

</details>
<details><summary>Indice 3</summary>

Même numérateur : la plus grande des deux est celle qui a le plus petit dénominateur. $\frac{P(A \mid B)}{P(B \mid A)} = \frac{\text{aire}(A)}{\text{aire}(B)}$, et l'aire d'un disque de rayon $r$ vaut $\pi r^2$. Pour d : `ratio_13 = p_a_given_b / p_b_given_a`, avec les variables de l'expérience.

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

Trois étapes : vérifier les entrées, choisir l'ordre des étiquettes, compter les couples (vérité, prédiction). La ligne vient de la vérité, la colonne de la prédiction.

</details>
<details><summary>Indice 2</summary>

Sans `labels` : `np.unique(np.concatenate([y_true, y_pred]))` donne les étiquettes triées, sans doublon. Un dictionnaire `index = {label: i for i, label in enumerate(labels.tolist())}` donne le numéro de chaque étiquette ; une étiquette absente de `index` doit lever une `ValueError`. Puis `C = np.zeros((k, k), dtype=int)` et, pour chaque couple, `C[index[t], index[p]] += 1`. Pour c, les étiquettes sont triées : la ligne des Gentoo est la 3ᵉ, la colonne des Chinstrap la 2ᵉ.

</details>
<details><summary>Indice 3</summary>

```python
def _check_pair(y_true, y_pred):
    a, b = np.asarray(y_true), np.asarray(y_pred)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) == 0:
        raise ValueError("y_true and y_pred must be non-empty 1-D arrays of the same length")
    return a, b


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = _check_pair(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    index = {label: i for i, label in enumerate(labels.tolist())}
    C = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t not in index or p not in index:
            raise ValueError(f"the label {t!r} or {p!r} is missing from labels")
        C[index[t], index[p]] += 1
    return C
```

</details>

### Ex 3.16 — accuracy, precision, recall, F-beta et F1 (cas binaire) 🔨

<details><summary>Indice 1</summary>

Toutes ces mesures sont des fractions de TP, FP et FN (et de TN pour l'accuracy). Écris d'abord une fonction d'aide qui vérifie les étiquettes et compte ces cases pour la classe `pos_label` ; les quatre mesures l'appellent.

</details>
<details><summary>Indice 2</summary>

Dans l'aide : `a, b = _check_pair(...)` (3.15) ; les étiquettes présentes, `np.unique(np.concatenate([a, b]))` ; plus de deux, ou deux dont aucune n'est `pos_label` : `ValueError`. Puis `t = a == pos_label`, `p = b == pos_label`, et `tp = np.sum(t & p)`, `fp = np.sum(~t & p)`, `fn = np.sum(t & ~p)`. Une petite fonction `_ratio(num, den, zero_division)` renvoie `zero_division` quand `den` vaut 0. `fbeta` vérifie d'abord `beta > 0` ; `accuracy` compare simplement `a == b`.

</details>
<details><summary>Indice 3</summary>

```python
def _binary_counts(y_true, y_pred, pos_label):
    a, b = _check_pair(y_true, y_pred)
    present = np.unique(np.concatenate([a, b]))
    if len(present) > 2:
        raise ValueError(f"{len(present)} labels found: the binary case allows at most two")
    if len(present) == 2 and pos_label not in present.tolist():
        raise ValueError(f"pos_label={pos_label!r} is not one of the labels")
    t, p = a == pos_label, b == pos_label
    return int(np.sum(t & p)), int(np.sum(~t & p)), int(np.sum(t & ~p))    # tp, fp, fn


def _ratio(num, den, zero_division):
    return float(num / den) if den != 0 else float(zero_division)


def precision(y_true, y_pred, pos_label=1, average="binary", zero_division=0.0):
    if average != "binary":
        raise NotImplementedError("average=... comes with exercise 3.25")
    tp, fp, fn = _binary_counts(y_true, y_pred, pos_label)
    return _ratio(tp, tp + fp, zero_division)
```
`recall` : la même chose avec FN ; `fbeta` : `b2 = beta ** 2`, puis `_ratio((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp, zero_division)` ; `f1` appelle `fbeta` ; `accuracy` : `float(np.mean(a == b))` après `_check_pair`.

</details>

### Ex 3.17 — La matrice à l'envers 🐛

<details><summary>Indice 1</summary>

Calcule d'abord les vraies valeurs avec tes fonctions : la sensibilité est le recall de la classe « malade », la precision se calcule aussi pour la classe « malade ». Puis compare-les au rapport du collègue.

</details>
<details><summary>Indice 2</summary>

Affiche `skm.confusion_matrix(test_truth, test_result)` et demande-toi dans quel ordre scikit-learn range les deux étiquettes (fiche §3.7.2, 🕰️). Que contient alors chacune des variables `tn, fp, fn, tp` du collègue, et quelles mesures a-t-il donc calculées ? Pour la correction, ne dépends plus de cet ordre : travaille avec des masques booléens, `truth = y_true == positive` et `alarm = y_pred == positive`.

</details>
<details><summary>Indice 3</summary>

`true_sensitivity_17 = mylearn.metrics.recall(test_truth, test_result, pos_label="malade")`, et de même avec `precision`. Dans `screening_report_fixed` : `tp = np.sum(truth & alarm)`, `fn = np.sum(truth & ~alarm)`, `fp = np.sum(~truth & alarm)`, puis les deux rapports en `float`. Autre correction : `skm.confusion_matrix(y_true, y_pred, labels=[negative, positive]).ravel()`, en trouvant d'abord l'étiquette négative.

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

Reprends les contrôles d'étiquettes de 3.16, compte les quatre cases une seule fois, puis remplis le dictionnaire dans l'ordre de la docstring, chaque rapport passant par la même petite fonction.

</details>
<details><summary>Indice 2</summary>

`t = a == pos_label`, `p = b == pos_label`, puis `tp`, `fn`, `fp`, `tn` avec `&` et `~`, convertis en `int` Python (les produits du MCC ne débordent pas). Une fonction interne `ratio(num, den)` renvoie `float(zero_division)` si `den` vaut 0. La balanced accuracy réutilise le recall et la spécificité ; le MCC a pour dénominateur la racine du produit des quatre sommes. Pour b, écris la precision avec « ham » comme classe positive : quelles cases de la matrice « spam » utilise-t-elle ?

</details>
<details><summary>Indice 3</summary>

```python
    def ratio(num, den):
        return float(num / den) if den != 0 else float(zero_division)

    recall_ = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {"accuracy": ratio(tp + tn, n), "balanced_accuracy": (recall_ + specificity) / 2,
            "precision": ratio(tp, tp + fp), "recall": recall_, "specificity": specificity,
            "npv": ratio(tn, tn + fn), "fpr": ratio(fp, fp + tn), "fnr": ratio(fn, fn + tp),
            "fdr": ratio(fp, fp + tp), "false_omission_rate": ratio(fn, fn + tn),
            "prevalence": ratio(tp + fn, n), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "mcc": ratio(tp * tn - fp * fn, np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))}
```
(avec `n = tp + fn + fp + tn`). Pour b : avec « ham » positif, la precision divise les « ham » bien prédits par toutes les prédictions « ham ».

</details>

### Ex 3.20 — Un seuil sur la nageoire : precision et recall en balance 🔬

<details><summary>Indice 1</summary>

Pour chaque seuil, les prédictions sont `(flipper >= t).astype(int)` ; mesure-les contre `is_gentoo` avec tes fonctions de 3.16. Range un dictionnaire par seuil dans une liste, puis fais-en un DataFrame.

</details>
<details><summary>Indice 2</summary>

Dans la boucle : precision, recall et F1 (tes fonctions, classe positive 1), `fn = np.sum((is_gentoo == 1) & (pred == 0))` et `fp = np.sum((is_gentoo == 0) & (pred == 1))`. Pour a, prends la ligne du seuil 205 : `table.set_index("threshold").loc[205]`. Pour b, `table.loc[table["f1"].idxmax(), "threshold"]` ; pour c, ajoute une colonne `cost = 10 * fn + fp`, puis `idxmin`.

</details>
<details><summary>Indice 3</summary>

```python
def sweep_20(thresholds):
    rows = []
    for t in thresholds:
        pred = (flipper >= t).astype(int)
        rows.append({"threshold": t,
                     "precision": mylearn.metrics.precision(is_gentoo, pred),
                     "recall": mylearn.metrics.recall(is_gentoo, pred),
                     "f1": mylearn.metrics.f1(is_gentoo, pred),
                     "fn": int(np.sum((is_gentoo == 1) & (pred == 0))),
                     "fp": int(np.sum((is_gentoo == 0) & (pred == 1)))})
    return pd.DataFrame(rows)
```

</details>

### Ex 3.21 — Simuler le dépistage : la prévalence fait la precision 🔬

<details><summary>Indice 1</summary>

La simulation suit la recette de l'énoncé à la lettre (deux appels à `rng.random`, dans cet ordre). La precision théorique se lit sur l'arbre des fréquences naturelles de la fiche : vrais positifs divisés par tous les positifs.

</details>
<details><summary>Indice 2</summary>

En parts de la population : vrais positifs = sensibilité × prévalence ; faux positifs = (1 − spécificité) × (1 − prévalence). Pour c, la precision vaut 0,5 quand ces deux quantités sont égales : une équation du premier degré en $p$. Pour d, tire `u2` juste après la simulation, avec le même `rng`, applique la même règle, puis garde les personnes positives aux deux tests (`positive & positive_2`).

</details>
<details><summary>Indice 3</summary>

```python
def simulate_screening(n, prevalence, sensitivity, specificity, rng):
    sick = rng.random(n) < prevalence
    u = rng.random(n)
    return sick, np.where(sick, u < sensitivity, u < 1 - specificity)
```
c) résous $0{,}99\,p = 0{,}02\,(1 - p)$. d) `u2 = rng.random(100_000)`, `positive_2 = np.where(sick, u2 < 0.99, u2 < 0.02)`, `both = positive & positive_2`, puis `np.sum(sick & both) / np.sum(both)`.

</details>

### Ex 3.22 — Vérifier avec scikit-learn : classification_report et affichages 📦

<details><summary>Indice 1</summary>

`print(skm.classification_report(species, expert_pred, digits=3))` affiche le tableau : une ligne par espèce, une colonne par mesure, et les moyennes sur les dernières lignes.

</details>
<details><summary>Indice 2</summary>

`skm.ConfusionMatrixDisplay.from_predictions(species, expert_pred, normalize=...)`, puis `plt.show()`. D'après sa documentation, `normalize` accepte `"true"`, `"pred"` ou `"all"` : laquelle divise chaque **ligne** (une vraie espèce) par son total ?

</details>
<details><summary>Indice 3</summary>

`report_22 = skm.classification_report(species, expert_pred, output_dict=True)`, puis `report_22["Gentoo"]["recall"]`, `report_22["macro avg"]["f1-score"]` et `report_22["weighted avg"]["f1-score"]`.

</details>

### Ex 3.23 — Lire la documentation de sklearn.metrics 🛠️

<details><summary>Indice 1</summary>

`help(skm.precision_score)` affiche la documentation dans le notebook : cherche la section `Parameters`, puis le paragraphe du paramètre concerné. Sur le site de scikit-learn, vérifie que la version affichée est la 1.6.

</details>
<details><summary>Indice 2</summary>

a) Que vaut la precision quand aucune prédiction n'est positive ? C'est `zero_division` qui décide. b) D'après la documentation, `labels` peut servir à choisir un sous-ensemble des étiquettes : que deviennent les échantillons des autres ? c) Quelle est la valeur par défaut de `pos_label`, et existe-t-elle parmi `"spam"` et `"ham"` ? d) Lis la phrase qui dit sur quoi `normalize` divise : les vraies classes, les classes prédites ou toute la population. e) Chaque échantillon compte avec son poids. f) Sans `labels`, dans quel ordre scikit-learn range-t-il les classes ?

</details>
<details><summary>Indice 3</summary>

a) une precision 0/0 prend la valeur de `zero_division` ; b) les échantillons dont une étiquette n'est pas dans `labels` sont retirés du comptage ; c) la classe positive par défaut est 1 ; d) avec `"pred"`, chaque colonne (une classe prédite) est divisée par son total ; e) $\frac{1 \times 1 + 1 \times 0 + 2 \times 1}{1 + 1 + 2}$ ; f) les classes sont triées : `"a"`, `"b"`, `"c"`.

</details>

### Ex 3.24 — Courbe ROC et AUC 🔨

<details><summary>Indice 1</summary>

La courbe se construit en descendant la liste des échantillons triés par score décroissant : chaque positif fait monter, chaque négatif fait avancer vers la droite (fiche, mini-exemple). Des scores égaux ne donnent qu'un seul point.

</details>
<details><summary>Indice 2</summary>

Vérifie d'abord les entrées : mêmes longueurs, pas de NaN (`np.isnan`), exactement deux classes (`np.unique`) dont `pos_label`. Après `order = np.argsort(-scores, kind="mergesort")` et `pos = positive[order]`, `np.cumsum(pos)` compte les TP et `np.cumsum(~pos)` les FP à chaque rang. Le dernier rang de chaque groupe de scores égaux se repère avec `np.flatnonzero(np.diff(sorted_scores))`, plus le tout dernier rang. Ajoute (0, 0) au début (seuil `np.inf`), puis divise par le nombre total de positifs et de négatifs. `auc` : la somme de `np.diff(x) * (y[1:] + y[:-1]) / 2`, changée de signe si `x` décroît. Pour b, compare $\max(\text{AUC}, 1 - \text{AUC})$ des quatre mesures.

</details>
<details><summary>Indice 3</summary>

```python
def _sorted_counts(y_true, y_score, pos_label):
    t, s = np.asarray(y_true), np.asarray(y_score, dtype=float)
    if t.ndim != 1 or s.ndim != 1 or len(t) != len(s):
        raise ValueError("y_true and y_score must be 1-D arrays of the same length")
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN")
    classes = np.unique(t)
    if len(classes) != 2 or pos_label not in classes.tolist():
        raise ValueError("y_true must contain exactly two classes, pos_label among them")
    positive = t == pos_label
    order = np.argsort(-s, kind="mergesort")
    s_sorted, pos = s[order], positive[order]
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s) - 1]   # last rank of each distinct score
    return np.cumsum(pos)[last], np.cumsum(~pos)[last], s_sorted[last]


def roc_curve(y_true, y_score, pos_label=1):
    tps, fps, thresholds = _sorted_counts(y_true, y_score, pos_label)
    tps, fps = np.r_[0, tps], np.r_[0, fps]
    return fps / fps[-1], tps / tps[-1], np.r_[np.inf, thresholds]


def auc(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(x) != len(y):
        raise ValueError("at least 2 points, and x and y of the same length")
    dx = np.diff(x)
    if not (np.all(dx >= 0) or np.all(dx <= 0)):
        raise ValueError("x must be non-decreasing or non-increasing")
    direction = 1.0 if np.all(dx >= 0) else -1.0
    return float(direction * np.sum(dx * (y[1:] + y[:-1]) / 2))
```
`roc_auc` : `auc(*roc_curve(y_true, y_score, pos_label)[:2])`.

</details>

### Ex 3.25 — Moyennes macro, micro et pondérée 🔨

<details><summary>Indice 1</summary>

Avec plusieurs classes, compte TP, FP et FN **pour chaque classe**, puis combine selon `average`. Ta matrice de confusion donne tout d'un coup.

</details>
<details><summary>Indice 2</summary>

`C = confusion_matrix(a, b, labels=present)` ; `tp = np.diag(C)` ; `fp = C.sum(axis=0) - tp` (le reste de chaque colonne) ; `fn = C.sum(axis=1) - tp` (le reste de chaque ligne) ; `support = C.sum(axis=1)`. Calcule les valeurs par classe avec une division qui applique `zero_division` case par case, puis : `None` → le tableau ; `"macro"` → `np.mean` ; `"weighted"` → `np.average(values, weights=support)` ; `"micro"` → additionne d'abord `tp`, `fp` et `fn`, puis une seule division. Pour d, calcule les trois baisses.

</details>
<details><summary>Indice 3</summary>

```python
def _divide(num, den, zero_division):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.full(num.shape, float(zero_division))
    np.divide(num, den, out=out, where=den != 0)
    return out
```
Puis une seule fonction d'aide pour les quatre mesures : elle vérifie `average` (sinon `ValueError`), compte `tp`, `fp`, `fn` et `support` (le cas `"binary"` de 3.16 donne des tableaux d'une seule case), les additionne pour `"micro"` (`tp.sum(keepdims=True)`…), calcule `_divide(...)`, puis combine : le tableau pour `None`, `float(np.mean(values))` pour `"macro"`, `float(np.average(values, weights=support))` pour `"weighted"`, `float(values[0])` sinon.

</details>

### Ex 3.26 — Courbe precision-recall et average precision 🔨

<details><summary>Indice 1</summary>

Même tri et mêmes cumuls que pour la ROC ; seuls les rapports changent : precision = TP / (TP + FP), recall = TP / (nombre total de positifs). Il reste à ranger les points dans l'ordre de scikit-learn.

</details>
<details><summary>Indice 2</summary>

Avec `tps`, `fps` et les seuils (scores distincts, **décroissants**) de ta fonction d'aide de 3.24 : `precision = tps / (tps + fps)`, `recall = tps / tps[-1]`. scikit-learn range les seuils par ordre **croissant** : inverse les trois tableaux (`[::-1]`), puis ajoute le point final (precision 1, recall 0) aux deux premiers. L'AP repart du seuil le plus haut, avec $R_0 = 0$ : la somme de `np.diff(r) * p[1:]`. Pour b et c, `np.argsort(-score)[:k]` donne les k plus hauts scores ; la precision est la part de fraudes parmi eux.

</details>
<details><summary>Indice 3</summary>

```python
def precision_recall_curve(y_true, y_score, pos_label=1):
    tps, fps, thresholds = _sorted_counts(y_true, y_score, pos_label)
    return (np.r_[(tps / (tps + fps))[::-1], 1.0], np.r_[(tps / tps[-1])[::-1], 0.0], thresholds[::-1])


def average_precision(y_true, y_score, pos_label=1):
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label)
    p, r = precision[::-1], recall[::-1]        # from the highest threshold: r[0] = 0 (the closing point)
    return float(np.sum(np.diff(r) * p[1:]))
```
b) `float(np.mean(y_26[np.argsort(-score_a)[:50]]))`, et de même pour B.

</details>

### Ex 3.27 — ROC ou PR ? Lire les courbes d'un problème déséquilibré 📈

<details><summary>Indice 1</summary>

Sur la ROC, l'abscisse est le taux de faux positifs et l'ordonnée le recall ; sur la courbe PR, l'abscisse est le recall et l'ordonnée la precision. Les lignes pointillées marquent le recall 0,5.

</details>
<details><summary>Indice 2</summary>

b) Sur le zoom, suis la ligne horizontale du recall 0,5 jusqu'aux courbes, puis descends lire l'abscisse. c) Sur le graphique de droite, suis la ligne verticale du recall 0,5. d) Un classifieur au hasard n'utilise pas les scores : ses alertes sont des cas tirés au hasard dans la population. Quelle part de positifs contiennent-elles, en moyenne ?

</details>
<details><summary>Indice 3</summary>

d) La precision d'un classifieur au hasard est la part des positifs. Pour les notes : négatifs = 99,5 % de 200 000 ; fausses alertes = FPR × négatifs ; positifs trouvés = 0,5 × positifs ; precision = trouvés / (trouvés + fausses alertes).

</details>

### Ex 3.28 — Calibration : quand la météo annonce 70 % 🔨

<details><summary>Indice 1</summary>

Le diagramme de fiabilité découpe [0, 1] en intervalles égaux ; dans chaque intervalle non vide, deux moyennes : celle des probabilités annoncées, et celle des résultats (la fréquence des 1). Le score de Brier est une moyenne de carrés.

</details>
<details><summary>Indice 2</summary>

Vérifie les entrées : résultats 0 ou 1 (`np.isin`), probabilités dans [0, 1] et sans NaN, mêmes longueurs, `n_bins` entier au moins égal à 1. `edges = np.linspace(0, 1, n_bins + 1)`, puis `bins = np.searchsorted(edges[1:-1], p)` donne le numéro de l'intervalle de chaque probabilité. `np.bincount(bins, minlength=n_bins)` compte les cas de chaque intervalle ; avec `weights=y` ou `weights=p`, il somme les résultats ou les probabilités. Ne garde que les intervalles où le compte est positif. Pour b, un masque : `(forecast >= 0.65) & (forecast < 0.75)`.

</details>
<details><summary>Indice 3</summary>

```python
def calibration_curve(y_true, y_prob, n_bins=10):
    t, p = _check_outcomes(y_true, y_prob)     # your checks: float arrays of 0/1 and of probabilities
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.searchsorted(edges[1:-1], p)     # a value on an inner edge goes to the lower bin
    counts = np.bincount(bins, minlength=n_bins)
    positives = np.bincount(bins, weights=t, minlength=n_bins)
    sums = np.bincount(bins, weights=p, minlength=n_bins)
    keep = counts > 0
    return positives[keep] / counts[keep], sums[keep] / counts[keep]
```
`brier_score` : `float(np.mean((p - t) ** 2))` après les mêmes contrôles. b) `rain[mask].mean()` pour A, puis pour B.

</details>

### Ex 3.29 — Recall ≥ 0,99 au meilleur prix 🏆

<details><summary>Indice 1</summary>

Explore d'abord la validation : pour chaque modèle, l'AP, et les scores des fraudes les plus basses (`np.sort(...)[:10]`). Pour un recall de 0,99, c'est le **bas** de la distribution des scores des fraudes qui compte, pas le haut.

</details>
<details><summary>Indice 2</summary>

Le modèle A laisse une partie des fraudes au milieu des transactions normales : pour les attraper, il faudrait alerter presque partout. Avec B, un seuil qui garde 99 % des fraudes de validation vise juste… sur la validation. Mesure combien ce seuil varie d'un échantillon d'environ 500 fraudes à l'autre (bootstrap du ch. 2 : rééchantillonne les scores des fraudes, recalcule le quantile à 1 %), et prends un seuil un peu plus bas.

</details>
<details><summary>Indice 3</summary>

Une méthode simple : `"score_b"`, et le quantile à 0,5 % des scores des fraudes de validation (`np.quantile(scores, 0.005)`) au lieu du quantile à 1 %. Plus stable si les scores des fraudes suivent à peu près une loi normale (trace leur histogramme) : leur moyenne moins 2,58 écarts-types (le quantile à 0,5 % d'une loi normale), qui utilise toutes les fraudes et pas seulement les plus basses.

</details>
