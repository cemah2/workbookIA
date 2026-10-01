# 4 · Règle de Bayes — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 4.Q1 — Deux écoles pour une même probabilité

<details><summary>Indice 1</summary>

Relis le §4.1 et le §4.2 de la fiche : l'atout de l'approche bayésienne y est nommé, et les deux définitions y sont données. Pour la question 4, demande-toi ce qu'on pourrait **répéter** pour attacher une fréquence à « le biais de cette pièce dépasse 0,6 ».

</details>
<details><summary>Indice 2</summary>

Une définition parle de ce qui arrive quand on répète une expérience un très grand nombre de fois ; l'autre, de ce qu'on sait d'une affirmation. Le biais d'**une** pièce donnée est un nombre fixe : une seule des deux définitions permet de lui donner une distribution. Pour la question 5, d'où vient le nombre 0,3 ?

</details>
<details><summary>Indice 3</summary>

Fréquentiste : la limite d'une fréquence sur des répétitions. Bayésien : un degré de certitude, qui peut porter sur un paramètre inconnu. L'atout : avant les données, le bayésien écrit ce qu'il croit, son **prior** ; pour l'exemple, pense à une connaissance qu'on a déjà avant de commencer (la part de spams connue d'un filtre, les taux de conversion des tests A/B précédents). Pour la question 6 : le bootstrap du ch. 2 est-il bayésien ?

</details>

### 4.Q2 — Le fréquentiste et la hauteur de la montagne

<details><summary>Indice 1</summary>

Relis le §4.2 de la fiche, en particulier l'encadré ⚠️ sur les raccourcis du livre.

</details>
<details><summary>Indice 2</summary>

Pour les questions 1 et 2, pour le fréquentiste du livre, existe-t-il une valeur exacte ? Que représente alors chaque mesure par rapport à elle ? Pour la question 3, cherche le mot « fréquence » dans la définition fréquentiste de la probabilité (Q1). Pour la question 5, un fréquentiste strict donne-t-il une probabilité à une grandeur fixe ? Pour la question 6, l'erreur typique d'une moyenne varie comme $\frac{1}{\sqrt{n}}$ (ch. 2).

</details>
<details><summary>Indice 3</summary>

Chaque mesure manque un peu la valeur exacte ; en combinant beaucoup de mesures, il espère s'en approcher. Le nom vient de la probabilité vue comme **fréquence à long terme** ; l'explication du livre, par la valeur mesurée le plus souvent, est un raccourci. Le fréquentiste résume ses mesures par leur moyenne, avec son erreur typique. $\sqrt{100} = 10$.

</details>

### 4.Q3 — Le bayésien et la longueur du crayon

<details><summary>Indice 1</summary>

Relis le §4.2 de la fiche : le portrait du bayésien que donne le livre, puis sa correction dans l'encadré ⚠️.

</details>
<details><summary>Indice 2</summary>

Pour la question 1 : le flou d'un objet (une pointe émoussée, une gomme usée) gêne-t-il seulement le bayésien ? Une distribution peut décrire ce qu'on **sait** d'une grandeur sans que la grandeur elle-même varie. Pour la question 6, que devient la largeur du posterior quand les mesures s'accumulent, et à quel point réagit-il alors à une nouvelle mesure ?

</details>
<details><summary>Indice 3</summary>

Un fréquentiste aussi doit décider ce qu'il mesure : un objet mal défini complique la mesure pour tout le monde. Ce n'est donc pas ce qui sépare les deux écoles, et le portrait force le trait. Le bayésien peut croire à une longueur précise : sa distribution mesure son incertitude. Avant la première mesure, il lui faut un prior. Un posterior très étroit ne bouge presque plus : pour suivre une grandeur qui change, il faut un modèle qui prévoit ce changement.

</details>

### 4.Q4 — Biais d'une pièce : le vocabulaire

<details><summary>Indice 1</summary>

Le biais est une probabilité : celle de quelle issue ?

</details>
<details><summary>Indice 2</summary>

Le nombre moyen de faces vaut biais × nombre de lancers. L'estimation fréquentiste du biais est une proportion. Pour la question 6, cherche au ch. 2 la loi d'une variable qui vaut 1 avec la probabilité $\theta$, et 0 sinon.

</details>
<details><summary>Indice 3</summary>

$0{,}15 \times 200$ et $\frac{31}{50}$. Au début, chaque lancer pèse $\frac{1}{n}$ dans la proportion, et l'erreur typique ne diminue que comme $\frac{1}{\sqrt{n}}$. La loi cherchée porte le nom d'un mathématicien suisse (ch. 2).

</details>

### 4.Q5 — Une seule face change déjà le verdict

<details><summary>Indice 1</summary>

Laquelle des deux pièces donne le plus souvent face ? Sur le mur peint de la fiche (§4.4), ne garde que les zones où la fléchette donne face.

</details>
<details><summary>Indice 2</summary>

Le posterior de la pièce équilibrée est l'aire de sa zone « face » divisée par l'aire des deux zones « face » réunies ; chaque aire vaut prior × vraisemblance. Pour pile, la pièce truquée donne pile avec la probabilité 0,2.

</details>
<details><summary>Indice 3</summary>

$P(\text{équilibrée} \mid \text{face}) = \frac{0{,}5 \times 0{,}5}{0{,}5 \times 0{,}5 + 0{,}5 \times 0{,}8}$, et de même pour pile avec 0,5 et 0,2. Pour la question 5, compare les rapports de vraisemblance « truquée contre équilibrée » : $\frac{0{,}8}{0{,}5}$ pour une face, $\frac{0{,}2}{0{,}5}$ pour une pile. Lequel s'éloigne le plus de 1, en facteur ?

</details>

### 4.Q6 — Prior, vraisemblance, évidence, posterior : qui est qui ?

<details><summary>Indice 1</summary>

Pour chaque nombre : est-ce une probabilité **avant** ou **après** avoir entendu le détecteur ? Porte-t-elle sur l'hypothèse (le feu) ou sur l'observation (la sonnerie) ?

</details>
<details><summary>Indice 2</summary>

Ce qui est après la barre est ce qu'on sait déjà. Le détecteur peut sonner de deux façons : pour un vrai feu, ou pour une fausse alerte (une cuisson, une douche très chaude).

</details>
<details><summary>Indice 3</summary>

Dans l'ordre : prior, vraisemblance, évidence, posterior. $P(S) = P(S \mid F)\,P(F) + P(S \mid \text{non } F)\,P(\text{non } F)$. Le fabricant teste son détecteur avec et sans feu ; il ne connaît pas la fréquence des feux chez toi. Pour la question 6 : que veut dire *evidence* en anglais, et que représente ce nombre ici ?

</details>

### 4.Q7 — La vraisemblance n'a pas à sommer à 1

<details><summary>Indice 1</summary>

Une vraisemblance $P(\text{face} \mid H)$ se lit hypothèse par hypothèse : chaque hypothèse dit avec quelle probabilité elle donne face.

</details>
<details><summary>Indice 2</summary>

Une hypothèse de biais $\theta$ donne face avec quelle probabilité ? Et pile ? Additionne une fois sur les trois hypothèses (questions 1 et 2), une fois sur les deux issues (question 3) : laquelle de ces sommes doit valoir 1 ?

</details>
<details><summary>Indice 3</summary>

Pour face, les vraisemblances sont les biais eux-mêmes ; pour pile, ce sont les $1 - \theta$. Pour une hypothèse fixée, la somme sur les issues vaut 1 ; la somme sur les hypothèses n'a aucune raison de valoir 1. Le dénominateur de la règle de Bayes est la somme des produits vraisemblance × prior : un facteur commun à toutes les vraisemblances se retrouve au numérateur et au dénominateur.

</details>

### 4.Q8 — Sonde spatiale : ne pas inverser la condition

<details><summary>Indice 1</summary>

Écris chaque phrase comme une probabilité conditionnelle : que sait-on (après la barre) et que cherche-t-on ? Puis retrouve son nom dans le tableau du §4.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

La règle de Bayes passe de $P(\text{détecté} \mid \text{vie})$ à $P(\text{vie} \mid \text{détecté})$ grâce au prior. Pour la question 4, la probabilité du livre est-elle « sachant la vie » ou « sachant rien détecté » ? Pour la question 5, compare deux nombres d'erreurs : les planètes stériles qui déclenchent la sonde, et les planètes habitées qu'elle rate.

</details>
<details><summary>Indice 3</summary>

Le test du livre : 101 planètes habitées (1 ratée) et 899 stériles, dont 30 déclenchent la sonde. Le FNR vaut $\frac{1}{101}$, alors que $P(\text{vie} \mid \text{rien})$ est le FOR. Pour la question 6, refais le calcul de la precision avec un prior de 0,001 : que deviennent les vrais positifs face aux fausses alertes ?

</details>

### 4.Q9 — La boucle posterior → prior

<details><summary>Indice 1</summary>

Relis le §4.6.1 de la fiche et son encadré ⚠️ « Indépendants sachant la pièce ».

</details>
<details><summary>Indice 2</summary>

Pour la question 3, écris la vraisemblance de la suite entière sous une hypothèse : c'est un produit. Pour la question 4, compare $P(\text{face, face})$ et $P(\text{face})^2$ dans l'exemple de l'encadré. Pour la question 5, que vaut en machine un produit de milliers de nombres plus petits que 1 ? Pour la question 6, de quoi le produit de la question 3 dépend-il ?

</details>
<details><summary>Indice 3</summary>

Le posterior résume tout ce qu'on sait une fois l'observation faite. L'hypothèse nécessaire est l'indépendance **sachant l'hypothèse** ; sans connaître la pièce, une face rend la truquée plus probable, donc une seconde face aussi. $0{,}5^{2\,000}$ vaut 0 en `float64`. Pour une pièce, seuls le nombre de faces et le nombre de piles comptent.

</details>

### 4.Q10 — Plus de lancers, plus de certitude ?

<details><summary>Indice 1</summary>

Relis le §4.6.2 et le §4.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Pour la question 3, calcule le rapport de vraisemblance d'une face, puis celui d'une pile, entre la pièce équilibrée et la pièce de biais 0,05 : avec 950 faces et 50 piles, lequel l'emporte ? Pour la question 4, que vaut un produit dont un facteur est nul ? Pour la question 5, le prior multiplie les vraisemblances : peut-il les annuler s'il n'est nulle part nul ? Pour la question 6, souviens-toi de la vitesse $\frac{1}{\sqrt{n}}$ du ch. 2.

</details>
<details><summary>Indice 3</summary>

Une face favorise l'équilibrée d'un facteur $\frac{0{,}5}{0{,}05} = 10$ ; une pile la défavorise d'un facteur $\frac{0{,}95}{0{,}5} = 1{,}9$ : 950 facteurs 10 contre 50 facteurs 1,9. Bayes ne compare que les hypothèses proposées, sans dire si l'une d'elles convient. Un prior nul reste nul ; un prior trompeur mais jamais nul finit par céder.

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 4.R1 — Ch. 3 : precision = P(malade | test positif)

<details><summary>Indice 1</summary>

Range les 1 000 personnes en deux groupes (50 malades, 950 personnes saines), puis applique le test à chaque groupe.

</details>
<details><summary>Indice 2</summary>

TP : malades positifs ; FN : malades négatifs ; FP : personnes saines positives ; TN : personnes saines négatives. Precision $= \frac{TP}{TP + FP}$, sensibilité $= \frac{TP}{TP + FN}$, FPR $= \frac{FP}{FP + TN}$, prévalence $= \frac{TP + FN}{1\,000}$.

</details>
<details><summary>Indice 3</summary>

TP = 45, FN = 5, FP = 95, TN = 855. Avec la règle de Bayes : $\frac{0{,}9 \times 0{,}05}{0{,}9 \times 0{,}05 + 0{,}1 \times 0{,}95}$. Son dénominateur, $P(\text{positif})$, est le total de la colonne « positif » divisé par 1 000.

</details>

### 4.R2 — Ch. 1 : un filtre anti-spam apprend-il avec des étiquettes ?

<details><summary>Indice 1</summary>

Relis les §1.2 à §1.4 de la fiche du ch. 1 : les labels, le jeu de test et les grandes familles d'apprentissage.

</details>
<details><summary>Indice 2</summary>

Avec des étiquettes : supervisé ; sans étiquette : non supervisé. Pour la question 3, relis la séparation entre jeu d'entraînement et jeu de test. Pour la question 5, la règle de Bayes a besoin d'un prior et de vraisemblances : qu'est-ce qui joue le rôle de l'hypothèse, et celui de l'observation ?

</details>
<details><summary>Indice 3</summary>

L'hypothèse est « spam » ou « normal », l'observation est la présence d'un mot. Prior : la part des spams parmi les e-mails ; vraisemblances : la fréquence de chaque mot dans les spams, et dans les e-mails normaux.

</details>

### 4.R3 — 0B : trois faces de suite avec une pièce truquée

<details><summary>Indice 1</summary>

Des lancers indépendants d'une pièce de biais **connu** : on multiplie les probabilités.

</details>
<details><summary>Indice 2</summary>

« Au moins une pile » est l'événement contraire de « trois faces ». $\ln(ab) = \ln a + \ln b$. Pour la question 4, combien d'ordres différents donnent deux faces et une pile ?

</details>
<details><summary>Indice 3</summary>

$0{,}7^3$, puis $1 - 0{,}7^3$ ; $3 \ln 0{,}7$ ; trois ordres (FFP, FPF, PFF), chacun de probabilité $0{,}7^2 \times 0{,}3$. Pour la question 5, $\theta^{x_i}(1 - \theta)^{1 - x_i}$ vaut $\theta$ si $x_i = 1$ et $1 - \theta$ si $x_i = 0$ : regroupe ensuite les puissances de $\theta$ et de $1 - \theta$.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 4.1 — Une face : la pièce est-elle équilibrée ?

<details><summary>Indice 1</summary>

Deux hypothèses, chacune avec son prior et sa vraisemblance pour « face » : écris-les, puis applique la règle de Bayes (fiche §4.4.1).

</details>
<details><summary>Indice 2</summary>

L'évidence additionne les deux façons d'obtenir face : (équilibrée, face) et (truquée, face), chacune « prior × vraisemblance ». Pour f et g, chaque pièce est choisie dans la moitié des 800 essais, en moyenne.

</details>
<details><summary>Indice 3</summary>

$P(\text{face}) = 0{,}5 \times 0{,}5 + 0{,}5 \times 0{,}75$, puis $P(\text{équilibrée} \mid \text{face}) = \frac{0{,}5 \times 0{,}5}{P(\text{face})}$. En effectifs : 400 essais avec chaque pièce, qui donnent 200 faces d'un côté et 300 de l'autre. Pour h, compare les aires des deux zones « face » du mur.

</details>

### Ex 4.2 — Une pile : le verdict s'inverse

<details><summary>Indice 1</summary>

Même démarche qu'en 4.1, avec les vraisemblances de « pile ».

</details>
<details><summary>Indice 2</summary>

Pour chaque pièce, la vraisemblance de pile est le complément de celle de face. L'évidence additionne les deux zones « pile » du mur, chacune « prior × vraisemblance ». Pour d, compare $P(\text{équilibrée} \mid \text{pile})$ au prior.

</details>
<details><summary>Indice 3</summary>

$P(\text{pile} \mid \text{truquée}) = 1 - 0{,}75$ ; évidence $0{,}5 \times 0{,}5 + 0{,}5 \times 0{,}25$ ; $P(\text{équilibrée} \mid \text{pile}) = \frac{0{,}25}{0{,}375}$, dont tu retranches 0,5 pour d. Pour e, garde les fractions exactes $\frac{2}{5}$ et $\frac{2}{3}$ : tu dois retrouver un nombre déjà vu. Pour f, compare d) à la baisse de 4.1 (de 0,5 à 0,4), puis $\frac{P(\text{face} \mid \text{truquée})}{P(\text{face} \mid \text{équilibrée})} = \frac{0{,}75}{0{,}5}$ et $\frac{P(\text{pile} \mid \text{équilibrée})}{P(\text{pile} \mid \text{truquée})} = \frac{0{,}5}{0{,}25}$.

</details>

### Ex 4.3 — Retrouver la règle de Bayes en trois lignes

<details><summary>Indice 1</summary>

La règle du produit s'écrit de deux façons pour la même probabilité jointe $P(H, O)$ : c'est tout ce qu'il faut pour la question 1.

</details>
<details><summary>Indice 2</summary>

1 : égale les deux écritures, puis divise par $P(O)$. 2 : découpe l'événement $O$ selon les hypothèses (∂ 3.4, formule des probabilités totales). 3 : additionne les numérateurs. 4 : divise la règle pour $H_1$ par la règle pour $H_2$. 5 : sors la constante de la somme du dénominateur.

</details>
<details><summary>Indice 3</summary>

2 : $P(O) = \sum_j P(O \mid H_j)\,P(H_j)$. 3 : le coefficient est $\frac{1}{P(O)}$. 4 : dans l'exercice 4.1, la cote a priori vaut 1 et le rapport de vraisemblance $\frac{0{,}75}{0{,}5}$ ; pour revenir à une probabilité, $P = \frac{\text{cote}}{1 + \text{cote}}$. 5 : si $P(O \mid H_j) = q$ pour tout $j$, alors $P(O) = q \sum_j P(H_j) = q$.

</details>

### Ex 4.4 — Vie extraterrestre : lire la sonde avec Bayes

<details><summary>Indice 1</summary>

Lis les taux de la sonde en **lignes** dans le tableau du test, puis applique la règle de Bayes avec le prior de la **région** (5 %), pas avec la proportion de planètes habitées du test.

</details>
<details><summary>Indice 2</summary>

La sensibilité divise les planètes habitées détectées par toutes les planètes habitées du test ; la spécificité fait de même avec les planètes stériles. Pour c, deux façons de ne rien détecter : une planète habitée ratée ($1 - $ sensibilité), pondérée par 0,05, ou une planète stérile (la spécificité), pondérée par 0,95 ; n'oublie pas de multiplier le résultat par 1 000. Pour e, la precision se lit dans la **colonne** « détecte » du tableau. Pour f, ne compte que les planètes stériles.

</details>
<details><summary>Indice 3</summary>

a $= \frac{240}{250}$ et b $= \frac{1\,645}{1\,750}$. c : $1\,000 \times \frac{0{,}04 \times 0{,}05}{0{,}04 \times 0{,}05 + 0{,}94 \times 0{,}95}$ ; d $= \frac{0{,}96 \times 0{,}05}{0{,}96 \times 0{,}05 + 0{,}06 \times 0{,}95}$ ; e $= \frac{240}{240 + 105}$ ; f : 6 % des 9 500 planètes stériles. Pour g, compare la part de planètes habitées dans le test ($\frac{250}{2\,000}$) et dans la région.

</details>

### Ex 4.5 — Deux faces : une mise à jour double ou deux simples ?

<details><summary>Indice 1</summary>

En deux temps : refais 4.1 en prenant pour prior le posterior de 4.1. D'un coup : la vraisemblance de deux faces sous une hypothèse est un produit.

</details>
<details><summary>Indice 2</summary>

a : même calcul qu'en 4.1, avec le prior mis à jour. b : les lancers sont indépendants **sachant la pièce** : multiplie les vraisemblances des deux lancers. c : un prior de 0,5 pour chaque pièce, et les vraisemblances de la paire. e : écris la vraisemblance de face, face, pile sous chaque pièce.

</details>
<details><summary>Indice 3</summary>

a $= \frac{0{,}4 \times 0{,}5}{0{,}4 \times 0{,}5 + 0{,}6 \times 0{,}75}$ ; b $= 0{,}75 \times 0{,}75$ ; d $= \frac{0{,}5 \times 0{,}5^2}{P(\text{face, face})}$ ; e $= \frac{0{,}5 \times 0{,}5^3}{0{,}5 \times 0{,}5^3 + 0{,}5 \times 0{,}75^2 \times 0{,}25}$ ; f : les mêmes facteurs, dans un autre ordre. Garde des fractions jusqu'au bout. Pour c, ne calcule **pas** $P(\text{face})^2$ : deux lancers d'une pièce inconnue ne sont pas indépendants.

</details>

### Ex 4.6 — Cinq hypothèses de biais après face, pile, face

<details><summary>Indice 1</summary>

Un tableau à cinq colonnes, une par hypothèse : à chaque lancer, multiplie la ligne courante par la vraisemblance du lancer, puis divise par la somme de la ligne.

</details>
<details><summary>Indice 2</summary>

Face : multiplie par $\theta$ ; pile : par $1 - \theta$. Pour d, l'évidence de la suite entière vaut $\sum \text{prior} \times \theta^2 (1 - \theta)$, ou le produit des trois sommes par lesquelles tu as divisé en chemin. Pour e, réponds une valeur de $\theta$, pas une probabilité.

</details>
<details><summary>Indice 3</summary>

Après face, les produits valent $[0 ;\ 0{,}05 ;\ 0{,}2 ;\ 0{,}15 ;\ 0{,}1]$, de somme 0,5. Après pile, multiplie le résultat de a par $[1 ;\ 0{,}75 ;\ 0{,}5 ;\ 0{,}25 ;\ 0]$, puis normalise. Pour f, refais le calcul avec le prior $[0{,}2 ;\ 0{,}2 ;\ 0{,}2 ;\ 0{,}2 ;\ 0{,}2]$ : seuls les facteurs $\theta^2 (1 - \theta)$ comptent.

</details>

### Ex 4.7 — Le posterior reste une distribution, un prior nul reste nul

<details><summary>Indice 1</summary>

Toutes les questions partent de la formule de ∂ 4.3 : un numérateur $P(O \mid H_i)\,P(H_i)$ et un dénominateur commun, $P(O)$.

</details>
<details><summary>Indice 2</summary>

1 : additionne les numérateurs. 2 et 3 : un produit qui contient un facteur nul est nul ; pour « toujours », raisonne par récurrence (le posterior devient le prior). 4 : hypothèse de récurrence : le prior du tour $n$ vaut $\frac{N_i}{\sum_j N_j}$, avec $N_i = P(H_i) \prod_{k < n} P(o_k \mid H_i)$. 5 : $\log \prod = \sum \log$. 6 : $e^{\ell_i + c} = e^c\,e^{\ell_i}$.

</details>
<details><summary>Indice 3</summary>

4 : applique la règle de Bayes avec ce prior et l'observation $o_n$ : les $\sum_j N_j$ se simplifient. L'indépendance sachant l'hypothèse sert à écrire la vraisemblance de toute la suite comme un produit. 5 : le plus petit `float64` positif vaut environ $5 \times 10^{-324}$ ; un produit de milliers de facteurs plus petits que 1 tombe en dessous. 6 : le facteur $e^c$ se simplifie entre le numérateur et le dénominateur ; avec $c = -\max_i \ell_i$, la plus grande exponentielle vaut 1.

</details>

### Ex 4.8 — Combien de sondes pour descendre sous un sur un million ?

<details><summary>Indice 1</summary>

Avec la forme « cotes » (fiche, 🧮 cotes), chaque sonde **multiplie** la cote « habitée contre stérile » par son rapport de vraisemblance : inutile de recalculer l'évidence à chaque sonde.

</details>
<details><summary>Indice 2</summary>

La cote a priori divise $P(\text{habitée})$ par $P(\text{stérile})$. Rapport d'une sonde négative : $\frac{P(\text{rien} \mid \text{habitée})}{P(\text{rien} \mid \text{stérile})}$ ; d'une sonde positive : $\frac{P(\text{détecté} \mid \text{habitée})}{P(\text{détecté} \mid \text{stérile})}$. Pour revenir à une probabilité : $P = \frac{\text{cote}}{1 + \text{cote}}$. Pour d, essaie 1, 2, 3, 4… sondes.

</details>
<details><summary>Indice 3</summary>

a $= \frac{0{,}05}{0{,}95}$. Rapports : $\frac{0{,}04}{0{,}94} = \frac{1}{23{,}5}$ (sonde négative) et $\frac{0{,}96}{0{,}06} = 16$ (sonde positive). c : cote $\frac{1}{19} \times \frac{1}{23{,}5^2}$, convertie en probabilité, puis multipliée par $10^6$. d : quand une probabilité est minuscule, elle est presque égale à sa cote : cherche le plus petit $k$ tel que $19 \times 23{,}5^k > 10^6$. e : cote $\frac{16^2}{19}$ ; f : cote $\frac{16}{19 \times 23{,}5}$ (convertis-les en probabilités). Pour g : les deux rapports sont-ils inverses l'un de l'autre ?

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 4.9 — La règle de Bayes sans formule, en cinq lignes

<details><summary>Indice 1</summary>

Pars d'une population en **effectifs** (10 000 personnes, par exemple) et d'une maladie rare.

</details>
<details><summary>Indice 2</summary>

Suis les deux groupes : combien de malades le test repère-t-il ? Combien de fausses alertes chez les personnes saines ? Compare ces deux nombres.

</details>
<details><summary>Indice 3</summary>

Les fausses alertes viennent du **grand** groupe, celui des personnes saines : même un petit taux d'erreur y fait beaucoup de monde. Par exemple, 10 malades sur 10 000, un test qui repère 9 malades sur 10 et sonne pour 5 % des personnes saines : 9 vrais positifs contre environ 500 fausses alertes. Place ensuite les trois mots : avant (la maladie est rare), indice (le test positif), mise à jour (le risque après le test).

</details>

### Ex 4.10 — Le prior est un choix : erreur du procureur et priors partiaux

<details><summary>Indice 1</summary>

Pour les questions 1 et 2, écris les deux probabilités conditionnelles avec la barre : laquelle a « innocent » après la barre ? Relis les pièges de la fiche (l'inversion de la condition).

</details>
<details><summary>Indice 2</summary>

2 : combien d'innocents compatibles attend-on parmi 3 millions de personnes, si chacune l'est avec la probabilité $10^{-6}$ ? 3 : élever au carré suppose deux événements indépendants : est-ce raisonnable dans une même famille ? Et « 1 sur 73 millions » est-il la probabilité des décès sachant l'innocence, ou l'inverse ? 4 : qu'est-ce que l'adresse révèle d'autre qu'un risque, et le client peut-il agir dessus ? 5 et 6 : que faire si la décision change quand on change le prior ?

</details>
<details><summary>Indice 3</summary>

2 : environ 3 innocents compatibles, plus le coupable. 3 : (1) l'indépendance des deux décès ; (2) l'inversion de la condition : il fallait comparer deux explications, toutes deux très rares. 4 : un substitut de caractéristiques protégées, et une boucle qui se renforce d'elle-même. 5 : documenter le prior, puis refaire le calcul avec plusieurs priors raisonnables (une analyse de sensibilité).

</details>

### Ex 4.11 — VanderPlas (2014) : fréquentisme et bayésianisme

<details><summary>Indice 1</summary>

Les questions suivent l'ordre de l'article : lis l'introduction, puis repère les mots-clés *frequentism*, *bayesianism*, *nuisance parameter*, *confidence* et *credibility*.

</details>
<details><summary>Indice 2</summary>

2 : compare les deux estimations du flux et leurs incertitudes. 3 : cherche le billard de Bayes (*Bayes' Billiards Game*) : quelle cote chaque approche donne-t-elle, et que donne la simulation ? 4 : cherche l'exponentielle tronquée de Jaynes : où tombe l'intervalle de confiance par rapport à la plus petite observation ? 5 : cherche les noms de paquets dans les blocs de code.

</details>
<details><summary>Indice 3</summary>

3 : l'approche naïve remplace le paramètre de nuisance par son estimation ; l'approche bayésienne en fait la moyenne sur toutes ses valeurs possibles, pondérées par leur probabilité (elle le **marginalise**). 4 : les deux phrases ne portent pas sur la même chose : la **procédure** répétée, ou **ce** paramètre sachant **ces** données. 7 : `mylearn.bayes` manipule des priors et des posteriors ; le bootstrap évalue une méthode sur des rééchantillons.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 4.E1 — Expliquer la règle de Bayes avec un test médical

<details><summary>Indice 1</summary>

Un plan en quatre temps : à quoi sert la règle (inverser une probabilité conditionnelle), la formule dite avec des mots, un exemple chiffré, ce que ça change en pratique.

</details>
<details><summary>Indice 2</summary>

Choisis un exemple qui frappe : une maladie rare et un test bon mais pas parfait, en effectifs sur 10 000 personnes. Termine sur le lien avec la precision d'un classifieur (ch. 3).

</details>
<details><summary>Indice 3</summary>

La phrase clé : « le posterior est proportionnel à la vraisemblance fois le prior ». Le prior est ici la prévalence : le même test, dans un service où la maladie est fréquente, se tromperait bien moins souvent quand il est positif.

</details>

### 4.E2 — Fréquentiste ou bayésien : quelle différence en pratique ?

<details><summary>Indice 1</summary>

Pars de la définition de la probabilité dans chaque école (Q1), puis passe aux intervalles.

</details>
<details><summary>Indice 2</summary>

Un intervalle de confiance est une promesse sur la **méthode** ; un intervalle de crédibilité, une probabilité sur le **paramètre**, sachant les données. Dans quels cas donnent-ils les mêmes nombres ?

</details>
<details><summary>Indice 3</summary>

Avec beaucoup de données et un prior plat, ils coïncident souvent (4.24, 4.25) ; avec peu de données, ils peuvent diverger (les dix piles de 4.25). Donne un exemple concret de paramètre, un taux de conversion par exemple.

</details>

### 4.E3 — Qu'est-ce qu'un prior et comment le choisir ?

<details><summary>Indice 1</summary>

Définis le prior en une phrase, puis dis d'où tu le tires, et ce qui se passe s'il est mauvais.

</details>
<details><summary>Indice 2</summary>

Sources : études passées, données historiques, contraintes physiques ; sans information, un prior peu informatif. Deux règles : jamais de probabilité nulle sur ce qui est possible (∂ 4.7), et une analyse de sensibilité.

</details>
<details><summary>Indice 3</summary>

Un prior trompeur se corrige, mais il coûte des données (4.21) ; un prior nul ne se corrige jamais. Cite un prior conjugué (la loi Beta pour une proportion) et, si tu le connais, le lien entre un prior gaussien sur les poids et la régularisation L2.

</details>

### 4.E4 — Où utiliser des méthodes bayésiennes en data science ?

<details><summary>Indice 1</summary>

Deux exemples bayésiens concrets, un exemple fréquentiste, et pour chacun, pourquoi ce choix.

</details>
<details><summary>Indice 2</summary>

Pense aux tests A/B, au réglage d'hyperparamètres et à la classification de textes ; côté fréquentiste, à l'incertitude d'une métrique de modèle (ch. 2). La fiche (« Bayes dans le ML d'aujourd'hui ») cite des outils.

</details>
<details><summary>Indice 3</summary>

Un test A/B bayésien donne une phrase que le métier comprend (« B a 93 % de chances d'être meilleure que A ») ; Optuna règle les hyperparamètres avec TPE, une méthode bayésienne ; Naive Bayes fait une baseline rapide. Côté fréquentiste : un intervalle bootstrap pour une AUC, simple et sans prior à justifier.

</details>

<a id="notebook"></a>

## Notebook, parties A à D

Les exercices du notebook (`03_notebook.ipynb`). Pour ceux qui complètent `mylearn/bayes.py`, la docstring de chaque fonction décrit déjà ce qu'elle doit faire, cas d'erreur compris : relis-la avant d'ouvrir un indice.

### Ex 4.12 — Combien de lancers pour démasquer la pièce truquée ? 🔮

<details><summary>Indice 1</summary>

Pense en cotes (fiche, 🧮 cotes) : chaque lancer multiplie la cote « truquée contre équilibrée » par un facteur, et l'on s'arrête quand elle dépasse 19 ($\frac{0{,}95}{0{,}05}$) ou descend sous $\frac{1}{19}$. Combien de facteurs faut-il, en moyenne ?

</details>
<details><summary>Indice 2</summary>

Avec un biais de 0,75, une face multiplie la cote par 1,5 et une pile par 0,5. En logarithmes, les facteurs s'additionnent : quand on lance la truquée, la log-cote monte en moyenne de $0{,}75 \ln 1{,}5 + 0{,}25 \ln 0{,}5$ par lancer, et il faut atteindre $\ln 19 \approx 2{,}9$. Fais le même calcul pour le biais 0,55. Pour c, que promet un seuil de 0,95 ?

</details>
<details><summary>Indice 3</summary>

Biais 0,75 : de l'ordre de $\frac{2{,}9}{0{,}13} \approx 23$ lancers. Biais 0,55 : $0{,}55 \ln 1{,}1 + 0{,}45 \ln 0{,}9 \approx 0{,}005$, soit de l'ordre de $\frac{2{,}9}{0{,}005} \approx 600$ lancers. Ces calculs donnent un ordre de grandeur de la moyenne ; la médiane est plus basse, car quelques essais très longs tirent la moyenne vers le haut. S'arrêter à 0,95, c'est accepter de se tromper environ une fois sur vingt.

</details>

### Ex 4.13 — L'estimation fréquentiste : la moyenne courante des faces 🔬

<details><summary>Indice 1</summary>

La $k$-ième estimation est un nombre de faces (un cumul) divisé par $k$ : `np.cumsum` donne tous les cumuls d'un coup, sans boucle.

</details>
<details><summary>Indice 2</summary>

`np.cumsum(flips)` donne le nombre de faces après chaque lancer ; il reste à le diviser par le nombre de lancers correspondant (1, 2, …, n), un tableau que `np.arange` fabrique : attention à son point de départ. Pour b, le 50ᵉ lancer est à l'indice 49. Pour c, applique la formule de l'énoncé avec $\theta = 0{,}8$ et $n = 100$.

</details>
<details><summary>Indice 3</summary>

```python
def running_estimate(flips):
    flips = np.asarray(flips)
    return np.cumsum(flips) / np.arange(1, len(flips) + 1)
```
b) `float(np.max(np.abs(running_estimate(flips_13[0.8])[49:] - 0.8)))` ; c) `float(np.sqrt(0.8 * 0.2 / 100))`.

</details>

### Ex 4.14 — evidence et bayes_posterior 🔨

<details><summary>Indice 1</summary>

Deux étapes dans chaque fonction : valider les entrées, puis calculer. Écris d'abord deux petites fonctions d'aide, l'une qui vérifie une distribution, l'autre des probabilités : les cinq fonctions du module les réutiliseront.

</details>
<details><summary>Indice 2</summary>

`np.asarray(p, dtype=float)`, puis, pour une distribution : une dimension, non vide, aucun NaN (`np.isnan`), aucune valeur négative, `abs(p.sum() - 1) <= 1e-8`. Attention : `nan < 0` vaut `False`, et une somme qui contient un NaN échappe aussi au test de la somme : teste NaN à part. Les vraisemblances : entre 0 et 1, sans NaN, et `likelihood.shape == prior.shape` (pas de broadcasting). `evidence` renvoie `float(np.sum(prior * likelihood))` ; `bayes_posterior` appelle `evidence`, refuse une évidence nulle, puis calcule un **nouveau** tableau (pas de `*=` sur une entrée).

</details>
<details><summary>Indice 3</summary>

```python
def _check_distribution(p, name):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if np.isnan(p).any() or (p < 0).any() or abs(p.sum() - 1) > 1e-8:
        raise ValueError(f"{name} must be a probability distribution")
    return p


def _check_probabilities(q, name):
    q = np.asarray(q, dtype=float)
    if np.isnan(q).any() or (q < 0).any() or (q > 1).any():
        raise ValueError(f"every value of {name} must be in [0, 1]")
    return q


def evidence(prior, likelihood):
    prior = _check_distribution(prior, "prior")
    likelihood = _check_probabilities(likelihood, "likelihood")
    if likelihood.shape != prior.shape:
        raise ValueError("prior and likelihood must have the same shape")
    return float(np.sum(prior * likelihood))


def bayes_posterior(prior, likelihood):
    total = evidence(prior, likelihood)            # also checks both inputs
    if total == 0:
        raise ValueError("the evidence is 0: the observation is impossible")
    return np.asarray(prior, dtype=float) * np.asarray(likelihood, dtype=float) / total
```

</details>

### Ex 4.15 — Bayes chez les manchots : l'espèce sachant l'île 📦

<details><summary>Indice 1</summary>

Le prior est la part de chaque espèce parmi les 344 manchots ; la vraisemblance, pour chaque espèce, la part de **ses** manchots qui ont été vus sur Dream.

</details>
<details><summary>Indice 2</summary>

`pd.crosstab(penguins["species"], penguins["island"])` donne les effectifs. Avec `normalize="index"`, chaque **ligne** (une espèce) est divisée par son total : la colonne `"Dream"` devient $P(\text{Dream} \mid \text{espèce})$. `.reindex(SPECIES)` impose l'ordre des lignes et `.to_numpy()` donne un tableau. Le prior : les totaux des lignes divisés par 344.

</details>
<details><summary>Indice 3</summary>

```python
counts = pd.crosstab(penguins["species"], penguins["island"]).reindex(SPECIES)
prior_15 = (counts.sum(axis=1) / counts.to_numpy().sum()).to_numpy()
likelihood_15 = (counts["Dream"] / counts.sum(axis=1)).to_numpy()
other_prior_15 = np.array([0.60, 0.15, 0.25])
```

</details>

### Ex 4.16 — La boucle posterior-prior : update_discrete 🔨

<details><summary>Indice 1</summary>

Une boucle sur les observations : à chaque tour, la **colonne** du tableau qui correspond à l'issue observée donne les vraisemblances, `bayes_posterior` fait la mise à jour, et le posterior remplace le prior.

</details>
<details><summary>Indice 2</summary>

Valide d'abord : le prior avec ton aide de 4.14 ; le tableau avec ton aide des probabilités, puis `table.ndim == 2` et `table.shape[0] == len(prior)`. `observations = np.asarray(observations)` doit avoir une dimension et contenir des nombres (`observations.dtype.kind` parmi `"biuf"`) ; dans la boucle, refuse `o` si `o != int(o)` (1,5) ou s'il n'est pas entre 0 et `table.shape[1] - 1` (−1 compris). La colonne : `table[:, int(o)]`. L'historique : une liste qui commence par le prior, convertie par `np.array(history)` à la fin. Dans la vérification, la face $k$ devient l'issue $k - 1$ (`rolls_16 - 1`) ; pour c, lis le posterior final qu'elle affiche.

</details>
<details><summary>Indice 3</summary>

```python
def update_discrete(prior, likelihoods, observations, return_history=False):
    posterior = _check_distribution(prior, "prior")
    table = _check_probabilities(likelihoods, "likelihoods")
    if table.ndim != 2 or table.shape[0] != posterior.size:
        raise ValueError("likelihoods must have one row per hypothesis")
    observations = np.asarray(observations)
    if observations.ndim != 1 or observations.dtype.kind not in "biuf":   # ['1'] or [None]: not indices
        raise ValueError("observations must be a 1-D sequence of outcome indices")
    history = [posterior]
    for o in observations:
        if not np.isfinite(o) or o != int(o) or not 0 <= o < table.shape[1]:
            raise ValueError(f"{o} is not an outcome index")
        posterior = bayes_posterior(posterior, table[:, int(o)])   # it becomes the next prior
        history.append(posterior)
    return np.array(history) if return_history else posterior
```

</details>

### Ex 4.17 — Reproduire les trente lancers de la figure 4.24 🎨

<details><summary>Indice 1</summary>

L'historique vient tout droit de `update_discrete(..., return_history=True)` : il reste à convertir la chaîne en issues. Le dessin, ce sont deux appels à `ax.bar`, le second posé sur le premier.

</details>
<details><summary>Indice 2</summary>

`np.array([1 if c == "F" else 0 for c in sequence])` donne les issues ; les colonnes de `COINS_17` sont dans l'ordre (pile, face), comme les issues 0 et 1. Pour empiler : `ax.bar(x, bas)`, puis `ax.bar(x, haut, bottom=bas)`. Pour les étiquettes : `ax.set_xticks(x)`, puis `ax.set_xticklabels(["before"] + list(sequence))`.

</details>
<details><summary>Indice 3</summary>

```python
def stacked_history_17(sequence):
    flips = np.array([1 if c == "F" else 0 for c in sequence])
    return mylearn.bayes.update_discrete([0.5, 0.5], COINS_17, flips, return_history=True)
```
Dans `draw_stacked_17` : `history = stacked_history_17(sequence)`, `x = np.arange(len(history))`, puis les deux `ax.bar` avec `history[:, 0]` et `history[:, 1]`. Pour tes notes, `np.argmax(history[:, 1] > 0.9)` donne la première ligne où la truquée dépasse 0,9 : son numéro est le nombre de lancers.

</details>

### Ex 4.18 — Le posterior qui s'évanouit : underflow 🐛

<details><summary>Indice 1</summary>

a : une boucle `while` qui augmente $n$ tant que `0.5 ** n != 0.0`. b : le `nan` vient d'un $\frac{0}{0}$ : à partir de quand le tableau `joint` de la fonction (le prior multiplié par le produit des vraisemblances) vaut-il 0 pour les **trois** hypothèses ?

</details>
<details><summary>Indice 2</summary>

b : `np.where(flips_18[:, None] == 1, biases_18, 1 - biases_18)` donne une ligne de trois vraisemblances par lancer ; `np.cumprod(..., axis=0)` donne leurs produits après chaque lancer ; multiplie-les par `prior_18`, comme la fonction. Cherche la première ligne où les trois valeurs sont nulles (`(joints == 0).all(axis=1)`) : la ligne 0 correspond à un lancer. c : compte les faces $h$ et les piles $t$, puis `np.log(prior) + h * np.log(biases) + t * np.log(1 - biases)`. d : $\log(e^{a} + e^{b}) = m + \log(e^{a - m} + e^{b - m})$, avec $m = \max(a, b)$.

</details>
<details><summary>Indice 3</summary>

```python
def posterior_fixed_18(prior, biases, flips):
    prior, biases, flips = np.asarray(prior, dtype=float), np.asarray(biases, dtype=float), np.asarray(flips)
    heads = int(flips.sum())
    tails = len(flips) - heads
    log_post = np.log(prior) + heads * np.log(biases) + tails * np.log(1 - biases)
    weights = np.exp(log_post - log_post.max())       # the largest becomes exp(0) = 1
    return weights / weights.sum()
```
b) `joints = prior_18 * np.cumprod(np.where(flips_18[:, None] == 1, biases_18, 1 - biases_18), axis=0)`, puis `int(np.argmax((joints == 0).all(axis=1))) + 1`. d) $-1\,000 + \ln(1 + e^{-1})$, soit `-1000 + np.log(1 + np.exp(-1.0))`.

</details>

### Ex 4.19 — La grille biais × proportion de faces 🔬

<details><summary>Indice 1</summary>

Deux boucles : l'une sur les proportions (les lignes), l'autre sur les biais (les colonnes). Chaque case est un problème à deux pièces, comme en 4.17, avec une série de $h$ faces suivies de $n - h$ piles.

</details>
<details><summary>Indice 2</summary>

`heads = round(proportion * n)` (`round`, pas `int`, qui tronque) ; `flips = np.array([1] * heads + [0] * (n - heads))` ; `table = np.array([[0.5, 0.5], [1 - bias, bias]])` (lignes : équilibrée, truquée ; colonnes : pile, face) ; puis `update_discrete([0.5, 0.5], table, flips)[0]`. Pour b, `np.sum((grid < 0.05) | (grid > 0.95))` sur chacune des deux grilles.

</details>
<details><summary>Indice 3</summary>

```python
def grid_19(n):
    grid = np.zeros((10, 10))
    for i, proportion in enumerate(PROPORTIONS_19):
        heads = round(proportion * n)
        flips = np.array([1] * heads + [0] * (n - heads))
        for j, bias in enumerate(BIASES_19):
            table = np.array([[0.5, 0.5], [1 - bias, bias]])
            grid[i, j] = mylearn.bayes.update_discrete([0.5, 0.5], table, flips)[0]
    return grid
```
b) `[int(np.sum((g < 0.05) | (g > 0.95))) for g in (grid_19(40), grid_19(1000))]`.

</details>

### Ex 4.20 — Envoyer des sondes jusqu'à la décision 🔬

<details><summary>Indice 1</summary>

Une boucle `for` sur les sondes, de 1 à 20 : tirer, mettre à jour, tester les deux seuils. Si aucun seuil n'est atteint après la 20ᵉ sonde, la décision est `"undecided"`.

</details>
<details><summary>Indice 2</summary>

Le posterior de départ est `np.array([PRIOR_20, 1 - PRIOR_20])` (habitée, stérile). **Un seul** tirage par sonde, avec le générateur reçu : `detected = rng.random() < (SENS_20 if inhabited else 1 - SPEC_20)`. Mise à jour : `update_discrete(posterior, TABLE_20, [int(detected)])`. Puis teste `posterior[0] < low` et `posterior[0] > high`, et renvoie aussi le nombre de sondes envoyées.

</details>
<details><summary>Indice 3</summary>

```python
def explore_20(inhabited, rng, low=1e-6, high=0.99, max_probes=20):
    posterior = np.array([PRIOR_20, 1 - PRIOR_20])
    for probes in range(1, max_probes + 1):
        detected = rng.random() < (SENS_20 if inhabited else 1 - SPEC_20)
        posterior = mylearn.bayes.update_discrete(posterior, TABLE_20, [int(detected)])
        if posterior[0] < low:
            return "mine", probes
        if posterior[0] > high:
            return "protect", probes
    return "undecided", max_probes
```

</details>

### Ex 4.21 — Un prior trompeur centré sur 0,8 🔮

<details><summary>Indice 1</summary>

Le posterior est proportionnel à prior × vraisemblance. Compare, en logarithmes, ce que le prior enlève à l'hypothèse 0,3 et ce que les lancers lui apportent face à 0,8.

</details>
<details><summary>Indice 2</summary>

Le prior de 0,3 vaut $e^{-0{,}5^2 / (2 \times 0{,}1^2)} = e^{-12{,}5}$ fois celui de 0,8. Avec 100 lancers et environ 30 faces, le log du rapport de vraisemblance entre 0,3 et 0,8 vaut $30 \ln\frac{0{,}3}{0{,}8} + 70 \ln\frac{0{,}7}{0{,}2}$ : compare-le à 12,5. Mais le MAP n'est pas forcément 0,3 : tant que le posterior est large, la pente du prior le tire vers 0,8. Pour c, divise le posterior obtenu avec la bosse par celui d'un prior uniforme : que reste-t-il ?

</details>
<details><summary>Indice 3</summary>

Le MAP est décalé vers la bosse d'environ $\frac{0{,}8 - 0{,}3}{0{,}1^2} \times \frac{0{,}3 \times 0{,}7}{n} \approx \frac{10}{n}$ : de l'ordre de 0,1 pour 100 lancers, de 0,01 pour 1 000. Le rapport des deux posteriors est proportionnel au prior, qui n'est pas constant. Pour d et e : `maps_21[100]` (la ligne 0 est le prior) ; `far = np.abs(maps_21 - 0.3) >= 0.015`, puis le dernier indice où `far` est vrai, plus 1.

</details>

### Ex 4.22 — Le posterior continu : vérifier avec scipy.stats.beta 📦

<details><summary>Indice 1</summary>

Fiche, au-delà du livre (2) : avec un prior uniforme, $h$ faces et $t$ piles donnent une loi Beta dont les deux paramètres dépendent de $h$ et $t$. Ensuite, l'objet `stats.beta(a, b)` a une méthode pour chaque question.

</details>
<details><summary>Indice 2</summary>

Le prior uniforme est $\mathrm{Beta}(1, 1)$ ; chaque face ajoute 1 au premier paramètre, chaque pile 1 au second. Méthodes : `.mean()` ; `.sf(x)`, qui donne $P(\theta > x)$ (c'est `1 - .cdf(x)`) ; `.interval(0.95)`, l'intervalle à queues égales.

</details>
<details><summary>Indice 3</summary>

`posterior = stats.beta(HEADS_22 + 1, TAILS_22 + 1)`, puis `posterior.mean()`, `posterior.sf(0.5)` et `list(posterior.interval(0.95))`. Pour tes notes : compare la moyenne $\frac{a}{a + b}$ et le mode $\frac{13}{20}$.

</details>

### Ex 4.23 — Refactoriser : du copier-coller à une fonction testée 🛠️

<details><summary>Indice 1</summary>

a : dans chaque bloc, deux choses doivent parler de la même île : le groupe filtré et le nombre par lequel on divise. Ensuite, une seule fonction : filtrer, compter, normaliser, compléter les espèces absentes.

</details>
<details><summary>Indice 2</summary>

`chosen = df.loc[df[column] == value, "species"]`, puis `chosen.value_counts(normalize=True)`, puis `.reindex(sorted(df["species"].unique()), fill_value=0.0)`. Pour les tests : un DataFrame de 3 à 6 lignes, écrit dans chaque test, et une propriété par test : la somme vaut 1 (bug 1) ; une espèce absente du groupe est dans l'index avec 0 (bug 2) ; des valeurs calculées à la main, sur un exemple où $P(\text{espèce} \mid \text{île}) \neq P(\text{île} \mid \text{espèce})$ (bug 3). Compare des flottants avec `pytest.approx`.

</details>
<details><summary>Indice 3</summary>

```python
def species_given(df, column, value):
    """Probability of each species among the rows where ``df[column] == value``.

    Parameters
    ----------
    df : pandas.DataFrame
        ...
    column : str
        ...
    value : object
        ...

    Returns
    -------
    pandas.Series
        ...

    Examples
    --------
    >>> ...
    """
    chosen = df.loc[df[column] == value, "species"]
    return chosen.value_counts(normalize=True).reindex(sorted(df["species"].unique()), fill_value=0.0)


def test_absent_species_is_zero():
    df = pd.DataFrame({"species": ["A", "A", "B"], "island": ["x", "x", "y"]})
    result = species_given(df, "island", "x")
    assert list(result.index) == ["A", "B"]
    assert result["B"] == 0
```
Complète les `...` de la docstring, puis écris sur le même modèle le test de la somme et le test calculé à la main.

</details>

### Ex 4.24 — coin_bias_posterior : 500 hypothèses en log-probabilités 🔨

<details><summary>Indice 1</summary>

Plus de boucle : compte les faces et les piles, puis une seule formule pour toute la grille, en logarithmes. Le plus délicat est le cas $\theta = 0$ ou $\theta = 1$.

</details>
<details><summary>Indice 2</summary>

Validation : `np.isin(flips, [0, 1]).all()` sur des lancers à une dimension ; une grille non vide, à une dimension, dont les valeurs sont des probabilités ; le prior (uniforme par défaut : `np.full(n, 1 / n)`) validé par ton aide, de la même forme que la grille. Calcul, sous `with np.errstate(divide="ignore"):` : `log_post = np.log(prior)` ; `if heads > 0:` ajoute `heads * np.log(grid)` ; de même pour les piles avec `np.log(1 - grid)`. Si tout vaut `-np.inf` : `ValueError`. Sinon, `weights = np.exp(log_post - log_post.max())`, puis divise par leur somme.

</details>
<details><summary>Indice 3</summary>

```python
def coin_bias_posterior(flips, grid, prior=None):
    flips = np.asarray(flips)
    if flips.ndim != 1 or not np.isin(flips, [0, 1]).all():
        raise ValueError("flips must be a 1-D sequence of 0 and 1")
    grid = _check_probabilities(grid, "grid")
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("grid must be a non-empty 1-D array")
    prior = np.full(grid.size, 1 / grid.size) if prior is None else _check_distribution(prior, "prior")
    if prior.shape != grid.shape:
        raise ValueError("prior and grid must have the same shape")
    heads = int(np.sum(flips == 1))
    tails = len(flips) - heads
    with np.errstate(divide="ignore"):               # log(0) = -inf is what we want here
        log_post = np.log(prior)
        if heads > 0:                                 # theta ** 0 = 1, even for theta = 0
            log_post = log_post + heads * np.log(grid)
        if tails > 0:
            log_post = log_post + tails * np.log(1 - grid)
    if np.all(log_post == -np.inf):
        raise ValueError("every hypothesis gets probability 0")
    weights = np.exp(log_post - log_post.max())      # the largest becomes exp(0) = 1
    return weights / weights.sum()
```

</details>

### Ex 4.25 — Intervalle de crédibilité contre intervalle bootstrap 🔨

<details><summary>Indice 1</summary>

Cumule le posterior : la borne basse est la première valeur de la grille où le cumul atteint $\frac{1 - \text{mass}}{2}$, la borne haute la première où il atteint $\frac{1 + \text{mass}}{2}$. `np.searchsorted` trouve ces deux indices sans boucle.

</details>
<details><summary>Indice 2</summary>

`cdf = np.cumsum(posterior)` est croissant ; `np.searchsorted(cdf, level)` (côté `"left"`, par défaut) renvoie le premier indice `i` tel que `cdf[i] >= level`. Les arrondis peuvent laisser le dernier cumul un peu sous 1 : borne l'indice par `len(grid) - 1`. Contrôles : `np.all(np.diff(grid) > 0)`, ton aide de distribution pour le posterior, des formes égales, `0 < mass < 1`. Renvoie `(float(grid[low]), float(grid[high]))`.

</details>
<details><summary>Indice 3</summary>

```python
def credible_interval(grid, posterior, mass=0.95):
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.diff(grid) > 0):
        raise ValueError("grid must be a non-empty, strictly increasing 1-D array")
    posterior = _check_distribution(posterior, "posterior")
    if posterior.shape != grid.shape:
        raise ValueError("grid and posterior must have the same shape")
    if not 0 < mass < 1:
        raise ValueError("mass must be strictly between 0 and 1")
    cdf = np.cumsum(posterior)
    last = len(grid) - 1                              # the total may be 0.9999999999999998
    low = min(int(np.searchsorted(cdf, (1 - mass) / 2)), last)    # first index where cdf >= level
    high = min(int(np.searchsorted(cdf, (1 + mass) / 2)), last)
    return float(grid[low]), float(grid[high])
```
Pour tes notes : que contiennent tous les rééchantillons bootstrap de dix piles ?

</details>

### Ex 4.26 — Le détective de pièces : vingt pièces, le moins de lancers possible 🏆

<details><summary>Indice 1</summary>

Commence par la stratégie de base de l'énoncé et regarde où elle dépense ses lancers : toutes les pièces en demandent-elles autant ? Lesquelles sont difficiles à identifier ?

</details>
<details><summary>Indice 2</summary>

Deux leviers. Le seuil : plus bas, il coûte moins de lancers, mais combien de pièces mal identifiées de plus ? La répartition : au lieu de finir une pièce avant de passer à la suivante, garde un posterior par pièce (`np.column_stack([1 - CANDIDATES_26, CANDIDATES_26])` comme tableau des vraisemblances), lance chaque pièce quelques fois, puis lance toujours celle dont le plus grand posterior est le plus petit, jusqu'à ce que toutes dépassent 0,99 ou que le budget soit épuisé.

</details>
<details><summary>Indice 3</summary>

```python
def detective_26(bag, threshold=0.99, warm_up=5):
    table = np.column_stack([1 - CANDIDATES_26, CANDIDATES_26])        # columns: P(tails), P(heads)
    uniform = np.full(len(CANDIDATES_26), 1 / len(CANDIDATES_26))
    posteriors = [mylearn.bayes.update_discrete(uniform, table, [bag.flip(i) for _ in range(warm_up)])
                  for i in range(20)]
    sure = np.array([posterior.max() for posterior in posteriors])
    while bag.flips_used < BUDGET_26 and sure.min() <= threshold:
        i = int(np.argmin(sure))                                     # the coin we know least about
        posteriors[i] = mylearn.bayes.update_discrete(posteriors[i], table, [bag.flip(i)])
        sure[i] = posteriors[i].max()
    return [float(CANDIDATES_26[np.argmax(posterior)]) for posterior in posteriors]
```
Les pièces difficiles sont celles des candidats intérieurs (0,35, 0,5 et 0,65), encadrés par un voisin de chaque côté : il leur faut souvent 120 à 140 lancers pour dépasser 0,99, contre 70 environ pour 0,2 et 0,8. C'est vers elles que cette stratégie envoie les lancers.

</details>
