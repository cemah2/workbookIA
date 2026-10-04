# 11 · Apprentissage et raisonnement — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 📈 ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 11.Q1 — Représentation, évaluation, optimisation : associer

<details><summary>Indice 1</summary>

Relis le tableau de la section 11.2 de la fiche : pour chaque élément, pose les trois questions « que peut-on exprimer ? », « comment juge-t-on ? », « comment cherche-t-on ? ».

</details>
<details><summary>Indice 2</summary>

Un algorithme qui modifie quelque chose pas à pas est une optimisation. Un nombre qui note une solution est une évaluation. Une famille de modèles ou une structure de paramètres est une représentation. Pour b), lequel des quatre morceaux ne fait que **juger** l'exemple, sans rien modifier ?

</details>
<details><summary>Indice 3</summary>

a) L'élément 1, la descente de gradient, modifie les poids pas à pas pour faire baisser la loss : il cherche une meilleure solution, c'est donc O. Pose la même question aux cinq autres : décrit-il ce que le modèle peut exprimer (R), note-t-il une solution (E), ou fait-il avancer la recherche (O) ? Deux pièges : pour 5, distingue le nombre qu'on maximise et la méthode qui le maximise ; pour 6, la limite « profondeur au plus 3 » porte-t-elle sur ce que l'arbre peut exprimer, ou sur la façon de le chercher ? b) Même grille pour les quatre morceaux : la mise à jour (B) modifie les poids, elle cherche. Classe de même A, C et D, et garde le seul qui se contente de juger l'exemple.

</details>

### 11.Q2 — Puissance de représentation : ce qu'un perceptron ne peut pas « savoir »

<details><summary>Indice 1</summary>

Un perceptron répond selon le signe de $w_1 x_1 + w_2 x_2 + b$ : sa frontière est toujours une **droite**. Pour chaque règle, la frontière est-elle une droite ?

</details>
<details><summary>Indice 2</summary>

Écris la frontière de chaque règle (remplace « < » ou « > » par « = »), puis demande-toi si c'est une droite, c'est-à-dire une équation de la forme $a x_1 + b x_2 = c$ ($a$ ou $b$ peut être nul, pas les deux). Pour la règle D, dessine la zone de la classe 1 : une seule droite peut-elle la séparer du reste ? Pour b), réécris chaque règle comme une somme pondérée de $x_1$, $x_2$, $x_1^2$, $x_2^2$, si c'est possible.

</details>
<details><summary>Indice 3</summary>

a) Remplace « > » ou « < » par « = » : la frontière de A, $x_1 + x_2 = 3$, est une droite, donc un perceptron représente A. Fais de même pour B, C et E. Pour D, la classe 1 occupe deux quarts de plan opposés : une seule droite peut-elle les séparer des deux autres ? Pense à XOR. b) Pour chaque règle que le perceptron ne représentait pas, essaie de l'écrire $w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_2^2 + b > 0$, avec des poids fixes. Pour « même signe », demande-toi si $x_1^2$ et $x_2^2$ gardent la trace du signe de $x_1$ et de $x_2$. c) Repense au polynôme de degré 15 du ch. 9 : un modèle capable de dessiner toutes sortes de frontières peut aussi suivre le bruit de ses données d'entraînement. Que devient alors son erreur sur des données nouvelles ?

</details>

### 11.Q3 — Représentable mais pas apprenable : le problème de l'arrêt

<details><summary>Indice 1</summary>

Relis l'encadré ⚠️ sur le problème de l'arrêt (fiche §11.2.1) : sur quoi porte exactement l'impossibilité, un programme précis ou une méthode pour tous les programmes ?

</details>
<details><summary>Indice 2</summary>

Pour c), distingue « on ne peut jamais le prouver pour un programme donné » et « aucune méthode ne le prouve pour tous les programmes ». Pour d), qu'est-ce qui empêcherait le programme de s'arrêter juste après qu'on a cessé d'attendre ? Pour e), la réponse « oui » ou « non » se range-t-elle facilement ? Quelqu'un peut-il la calculer partout ?

</details>
<details><summary>Indice 3</summary>

a) Lancé sur une entrée précise, un programme précis s'arrête ou tourne sans fin : la question a une réponse, oui ou non, même si personne ne la connaît. a) est donc vrai. b) et c) Distingue une **méthode** qui répondrait juste pour tous les couples (programme, entrée) et une **preuve** pour un programme précis : laquelle Turing a-t-il déclarée impossible ? Pour c), un seul programme dont on prouve qu'il tourne sans fin suffirait à rendre la phrase fausse : peux-tu en écrire un, même très court, et le prouver ? d) Après dix ans d'attente, qu'est-ce qui empêche le programme de s'arrêter la seconde suivante ? e) Deux questions, dans l'ordre des boîtes du livre : chaque réponse tient-elle en un bit (représentable) ? Un algorithme peut-il calculer ce bit pour tous les couples ? Sans cela, aucun ne peut l'apprendre exactement.

</details>

### 11.Q4 — Loss, métrique, objectif : qui sert à quoi ?

<details><summary>Indice 1</summary>

Relis les trois définitions de la section 11.2.2 de la fiche, puis l'encadré ⚠️ sur l'accuracy, la precision et le recall.

</details>
<details><summary>Indice 2</summary>

Qui fait baisser quoi pendant l'entraînement ? Quel nombre calcule-t-on sur des données mises de côté ? Lequel parle d'argent ou de délais ? Pour b), que devient l'accuracy quand on change un peu un poids ? Pour c) et d) : la precision part des alertes données, le recall des vraies fraudes ; de quoi part chaque souhait ?

</details>
<details><summary>Indice 3</summary>

a) Trois questions pour chaque élément : l'optimiseur le fait-il baisser pendant l'entraînement (L) ? Le calcule-t-on sur des données mises de côté, pour juger le modèle (M) ? Parle-t-il le langage du projet, en argent, en délais ou en fraudes évitées (O) ? L'élément 1, la cross-entropy que la descente de gradient fait baisser, est donc une loss : L. Classe les trois autres de la même façon. b) Change un poids d'un tout petit peu : combien de prédictions changent, et de combien bouge l'accuracy ? Que peut tirer la descente de gradient d'une dérivée qui vaut presque partout ce nombre-là ? c) et d) La precision part des alertes données (parmi elles, quelle part de vraies fraudes ?) ; le recall part des vraies fraudes (parmi elles, quelle part d'alertes ?). Pour chaque souhait, repère l'ensemble dont il parle : les alertes, ou les fraudes ?

</details>

### 11.Q5 — Optimiser n'est pas être optimal ; pas de repas gratuit

<details><summary>Indice 1</summary>

Relis la section 11.2.3 de la fiche : l'image du vélo et du train, le minimum local, et le paragraphe qui dit comment on cite souvent mal le théorème.

</details>
<details><summary>Indice 2</summary>

Une suite de pas qui améliorent peut-elle s'arrêter ailleurs qu'au meilleur endroit (ch. 5) ? Sur quoi le théorème No Free Lunch fait-il une moyenne : sur ton problème, ou sur tous les problèmes possibles ?

</details>
<details><summary>Indice 3</summary>

a) Une descente qui améliore à chaque pas s'arrête là où plus aucun petit pas n'améliore : un minimum local, pas forcément le point le plus bas de toute la surface (ch. 5). a) est donc faux, comme le vélo de la fiche, qui ne devient jamais un train. b) Sur quoi le théorème fait-il sa moyenne : sur ton problème, ou sur tous les problèmes possibles, y compris l'immense majorité qui n'a aucune structure ? Un problème réel ressemble-t-il à cette moyenne ? c) Si aucun algorithme ne gagne partout, qu'est-ce qui fait gagner un algorithme sur un problème précis, et comment le vérifies-tu sur tes propres données ?

</details>

### 11.Q6 — Déduction ou induction ? Six situations

<details><summary>Indice 1</summary>

Pour chaque situation, demande-toi : si les prémisses sont vraies, la conclusion peut-elle être fausse ? Si oui, c'est une induction.

</details>
<details><summary>Indice 2</summary>

Pour chaque situation, repère les prémisses et la conclusion. La conclusion dit-elle quelque chose que les prémisses ne contenaient pas déjà (un cas non observé, une règle générale, l'avenir) ? Si oui, elle peut être fausse alors que les prémisses sont vraies. Si elle ne fait que tirer ce que les prémisses disaient déjà, elle est nécessaire.

</details>
<details><summary>Indice 3</summary>

Le test : si les prémisses sont vraies, la conclusion peut-elle encore être fausse ? Situation 1 : même si les 50 000 e-mails ont tous le bon label, le filtre peut se tromper sur le suivant ; la règle qu'il tire des exemples va au-delà de ce qu'ils contiennent : c'est une induction, I. Applique le même test aux cinq autres. La conclusion ajoute-t-elle quelque chose que les prémisses ne contenaient pas (une règle générale, un cas futur, une cause probable) ? Ou se contente-t-elle de tirer ce qu'elles disaient déjà ?

</details>

### 11.Q7 — Valide, solide, ou ni l'un ni l'autre ?

<details><summary>Indice 1</summary>

D'abord la forme (valide ?), avec les règles de distribution de la fiche ; ensuite seulement les prémisses (vraies ?).

</details>
<details><summary>Indice 2</summary>

Pour la forme des syllogismes catégoriques, applique les quatre règles de distribution de l'encadré 🧮 du §11.4, en commençant par le moyen terme (est-il distribué au moins une fois ?). Le 6 est un syllogisme conditionnel : compare-le aux quatre formes du §11.4, deux valides et deux sophismes. Pour chaque syllogisme valide, vérifie ensuite ses prémisses une par une : connais-tu un cas réel qui en contredit une ?

</details>
<details><summary>Indice 3</summary>

a) La forme d'abord, les prémisses ensuite. Syllogisme 1 : « tout $M$ est $P$ ; tout $S$ est $M$ ; donc tout $S$ est $P$ » ($M$ = les mammifères, $P$ = ce qui respire de l'air, $S$ = les dauphins) est Barbara, une forme valide, et ses deux prémisses sont vraies : S. Pour 2 à 5, fais l'inventaire des termes distribués et passe les quatre règles, le moyen terme d'abord ; si la forme est valide, cherche un cas réel qui contredit une prémisse. Pour 6, compare-le aux quatre formes du syllogisme conditionnel du §11.4. b) Relis, dans la fiche (§11.4.1), le paragraphe qui suit le tableau des sophismes : une forme invalide empêche-t-elle sa conclusion d'être vraie ?

</details>

### 11.Q8 — Nommer le sophisme syllogistique

<details><summary>Indice 1</summary>

Sépare d'abord les raisonnements conditionnels (« si… alors… ») des raisonnements catégoriques (« tous les… », « aucun… »).

</details>
<details><summary>Indice 2</summary>

Conditionnels : part-on du conséquent (le « alors » est vrai) ou nie-t-on l'antécédent (le « si » est faux) ? Catégoriques : le moyen terme est-il distribué ? Le sujet ou le prédicat de la conclusion est-il distribué dans la conclusion sans l'être dans sa prémisse ?

</details>
<details><summary>Indice 3</summary>

Sépare d'abord les conditionnels (1 et 3) des catégoriques (2, 4 et 5). Raisonnement 1 : « si le serveur est surchargé, la page est lente ; elle est lente ; donc il est surchargé » part du « alors » pour conclure le « si » : c'est l'affirmation du conséquent, A. Pour 3, regarde de même quelle partie du « si… alors… » la seconde prémisse reprend, et si elle l'affirme ou la nie. Pour 2, 4 et 5, demande-toi d'abord si le moyen terme est distribué au moins une fois ; s'il l'est, cherche le terme de la conclusion qui y est distribué sans l'être dans sa prémisse : son prédicat (majeur illicite) ou son sujet (mineur illicite) ?

</details>

### 11.Q9 — Généralisation, syllogisme statistique, prédiction

<details><summary>Indice 1</summary>

Relis le tableau de la section 11.5 de la fiche : de quoi part chaque raisonnement (échantillon ou population) et vers quoi va-t-il (population, individu, prochain cas) ?

</details>
<details><summary>Indice 2</summary>

Pour chaque phrase, souligne d'où part le raisonnement (un échantillon, ou toute la population ?) et de qui parle la conclusion (toute la population, un individu tiré au hasard, ou le prochain cas observé ?). Pour c), relis la section 11.5.1 sur la dérive des données.

</details>
<details><summary>Indice 3</summary>

a) Pour chaque phrase : de quoi part-on (un échantillon, ou toute la population ?) et de qui parle la conclusion (toute la population, un individu tiré au hasard, ou le prochain cas observé ?). Phrase 1 : les 12 % portent sur toutes les transactions de la banque, la population, et la conclusion sur une transaction tirée au hasard, un individu : c'est un syllogisme statistique, S. Classe les trois autres de la même façon ; pour 3, la conclusion parle-t-elle de toute la base d'images, ou de la prochaine image tirée ? b) Relis la phrase qui suit le tableau des trois principes (fiche §11.5) : sur quelle hypothèse reposent-ils tous ? c) Une généralisation suppose que l'échantillon ressemble à la population sur laquelle on conclut. La clientèle de 2026 est-elle encore celle qu'on a échantillonnée en 2019 ?

</details>

### 11.Q10 — Sophismes inductifs chez les data scientists

<details><summary>Indice 1</summary>

Prends les définitions **usuelles** du tableau de la section 11.5.2 de la fiche, pas celles du livre.

</details>
<details><summary>Indice 2</summary>

Pour chaque situation, cherche le défaut : trop peu de cas ? une collecte qui choisit certains individus ? une anecdote frappante ? des données nettes qu'on refuse ? une règle criblée d'exceptions ? une exception qu'on s'accorde à soi-même ?

</details>
<details><summary>Indice 3</summary>

Avec les définitions usuelles de la fiche (§11.5.2), cherche le défaut de chaque situation. Situation 1 : un seul accident, spectaculaire et partagé des millions de fois, pèse plus que les statistiques d'accidents : c'est la vivacité trompeuse, E. Pour les cinq autres, pose les questions de l'indice 2 : trop peu de cas ? une collecte qui choisit certains individus ? une exception qu'on s'accorde à soi-même ? une règle criblée d'exceptions ? des données nettes qu'on écarte en invoquant le hasard ? En 2, est-ce le nombre d'avis qui pose problème, ou la façon dont ils sont recueillis ?

</details>

### 11.Q11 — Prémisses rationnelles, empiriques, et la fourche de Hume

<details><summary>Indice 1</summary>

Pour chaque prémisse : peut-on la savoir vraie sans rien observer, par la seule raison ou par définition ?

</details>
<details><summary>Indice 2</summary>

Pour chaque prémisse, imagine comment tu la vérifierais : par une démonstration ou par le sens des mots, ou par une mesure ? Une prémisse qu'une expérience pourrait un jour contredire est empirique. Pour b) et c), relis l'encadré ⚠️ sur Hume (fiche §11.6).

</details>
<details><summary>Indice 3</summary>

a) Pour chaque prémisse : peux-tu la savoir vraie sans rien observer, par une démonstration ou par le seul sens des mots (R) ? Ou faut-il mesurer, observer (E) ? Prémisse 1 : la somme des angles d'un triangle du plan se démontre en géométrie euclidienne, sans mesurer un seul triangle : R. Fais de même pour les quatre autres ; une prémisse qu'une expérience pourrait un jour contredire est empirique. b) Relis l'encadré ⚠️ sur Hume (fiche §11.6) : comment connaît-on un fait, selon lui, et quel statut a-t-il à côté d'une vérité de mathématiques ? c) Une déduction valide garantit la conclusion **si** les prémisses sont vraies. Si la majeure n'est que probable, la conclusion peut-elle être plus sûre qu'elle ?

</details>

### 11.Q12 — Holmes déduit-il vraiment ?

<details><summary>Indice 1</summary>

Relis la section 11.6.1 de la fiche, et la définition de l'abduction.

</details>
<details><summary>Indice 2</summary>

Les entailles d'une chaussure ont-elles une seule cause possible ? Pour b), si la liste des possibles est complète, l'élimination peut-elle se tromper ? Pour c), d'où part-on : des données ou d'une théorie ?

</details>
<details><summary>Indice 3</summary>

a) Des entailles sur le cuir d'une chaussure peuvent-elles avoir une autre cause qu'une bonne maladroite qui gratte de la boue séchée ? Oui : la conclusion de Holmes n'est pas nécessaire, c'est l'explication la plus plausible, une abduction : B. b) Écris une étape d'élimination sous forme logique : « $A$ ou $B$ ; pas $A$ ; donc $B$ ». Avec une liste des possibles complète, si toutes les prémisses sont vraies, la conclusion peut-elle être fausse ? Applique le test de la fiche (§11.3). c) Le conseil dit de recueillir les données avant de bâtir une théorie : quelle façon de raisonner part des observations pour en tirer une règle (fiche §11.3) ? d) Suivre les regards et les expressions de Watson pour deviner sa pensée, est-ce tirer une conclusion nécessaire, ou choisir l'explication la plus plausible ? Compare avec le sens strict de « déduction » (fiche §11.3).

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 11.R1 — Ch. 10 : la règle du perceptron et le cas XOR

<details><summary>Indice 1</summary>

Calcule $z = \mathbf{w}\cdot\mathbf{x} + b$, puis $y\,z$ : la règle ne corrige que si $y\,z \le 0$.

</details>
<details><summary>Indice 2</summary>

Fais les deux exemples dans l'ordre : le second part des poids laissés par le premier. En cas d'erreur : $\mathbf{w} \leftarrow \mathbf{w} + \eta\,y\,\mathbf{x}$ et $b \leftarrow b + \eta\,y$. Pour d), XOR est-il linéairement séparable ?

</details>
<details><summary>Indice 3</summary>

a) $z = 1 \times 2 + (-1) \times 1 + 0 = 1$, donc $y\,z = 1 > 0$ : l'exemple est bien classé, et la règle ne touche à rien ; a) est donc faux. b) Même méthode avec $\mathbf{x} = (1, 2)$ et les poids laissés par a) : calcule $z$, puis $y\,z$ ; si $y\,z \le 0$, applique $\mathbf{w} \leftarrow \mathbf{w} + \eta\,y\,\mathbf{x}$. c) Au même pas, si l'exemple est corrigé : $b \leftarrow b + \eta\,y$. d) La règle cesse de corriger quand une droite sépare parfaitement les deux classes : place les quatre points de XOR sur un dessin et cherche une telle droite.

</details>

### 11.R2 — Ch. 8 : représentativité du jeu d'entraînement et fuite de données

<details><summary>Indice 1</summary>

Un jeu de test estime l'erreur sur la population dont il est tiré. En 2026, est-ce encore la même population ?

</details>
<details><summary>Indice 2</summary>

Pour b), la moyenne et l'écart-type calculés sur tout le jeu contiennent-ils de l'information du test ? Pour c), que garde un découpage stratifié d'un jeu à l'autre ? Pour d), quelle feature n'est connue qu'après la transaction ?

</details>
<details><summary>Indice 3</summary>

a) Le test est tiré des ventes de 2015 à 2019 : il estime l'erreur sur cette population-là. En 2026, les prix ont dérivé, la population n'est plus celle qu'on a échantillonnée, et rien ne dit dans quel sens l'erreur bougera : c'est la réponse B. b) La moyenne et l'écart-type calculés sur **toutes** les données ont-ils « vu » les exemples du test ? c) Un découpage stratifié garde la même part de malades dans chaque jeu : le nombre de malades du test est la taille du test multipliée par cette part. d) Pour chaque feature, demande-toi si on la connaît **au moment** de la transaction, ou seulement après.

</details>

### 11.R3 — Ch. 4 : mettre à jour sa croyance sur une pièce avec Bayes

<details><summary>Indice 1</summary>

Avec le prior $\mathrm{Beta}(1, 1)$, après $h$ faces et $t$ piles, le posterior est $\mathrm{Beta}(1 + h, 1 + t)$.

</details>
<details><summary>Indice 2</summary>

La moyenne de $\mathrm{Beta}(a, b)$ vaut $a/(a + b)$ ; son mode, $(a - 1)/(a + b - 2)$. Pour e), compare $\frac{h+1}{n+2} - \frac{1}{2}$ et $\frac{h}{n} - \frac{1}{2}$. Pour f), relis la règle de succession de Laplace dans la fiche du ch. 4, section « la limite continue, la loi Beta ».

</details>
<details><summary>Indice 3</summary>

a) Le prior $\mathrm{Beta}(1, 1)$ reçoit les 7 faces dans son premier paramètre et les 3 piles dans le second : $[1 + 7, 1 + 3] = [8, 4]$. b) et c) Applique au posterior de a) la moyenne $\frac{a}{a + b}$ et le mode $\frac{a - 1}{a + b - 2}$ (fiche du ch. 4, « la limite continue, la loi Beta »). d) Ajoute les 2 piles de plus au second paramètre, puis reprends la formule de la moyenne. e) Mets les deux écarts à 1/2 sur le même modèle : $\frac{h+1}{n+2} - \frac{1}{2} = \frac{2h - n}{2(n+2)}$ ; fais de même pour $\frac{h}{n} - \frac{1}{2}$, puis compare les deux fractions. f) Relis le mini-exemple de la même section de la fiche du ch. 4 : pour un bayésien, la probabilité que le prochain lancer donne face est la moyenne de $P(\text{face} \mid \theta) = \theta$ sur tous les biais possibles, chacun pondéré par son posterior. Quelle grandeur du posterior de a) est-ce, et laquelle des quatre fractions lui est égale ?

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 11.1 — Représentable sur n bits : compter, puis conclure ✏️

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « compter ce qu'une représentation peut contenir » (fiche §11.2.1).

</details>
<details><summary>Indice 2</summary>

$n$ bits écrivent $2^n$ valeurs, de 0 à $2^n - 1$ sans signe, de $-2^{n-1}$ à $2^{n-1} - 1$ en complément à deux. Pour d), combien de valeurs de 0 à 1 000 ? Une fonction booléenne choisit 0 ou 1 pour chacune des $2^n$ entrées possibles. Pour h), un symbole parmi 10 équiprobables porte $\log_2 10$ bits (ch. 6).

</details>
<details><summary>Indice 3</summary>

a) $n$ bits écrivent $2^n$ valeurs : avec 8 bits, $2^8 = 256$. b) et c) Lis les plages de l'encadré 🧮 avec $n = 8$ : de 0 à $2^n - 1$ sans signe, de $-2^{n-1}$ à $2^{n-1} - 1$ en complément à deux. d) Compte les valeurs de 0 à 1 000, le 0 compris, puis cherche la plus petite puissance de 2 qui les contient toutes : le nombre de bits est son exposant, $\lceil \log_2(\text{nombre de valeurs}) \rceil$. e) Compte les combinaisons de 3 entrées binaires, puis les façons de choisir une sortie, 0 ou 1, pour chacune d'elles : $2^{\text{nombre de combinaisons}}$. f) et g) Divise le nombre de fonctions représentables par le nombre **total** de fonctions booléennes de 3, puis de 4 entrées (la formule de e), pas par le nombre de combinaisons. h) Un chiffre parmi 10 équiprobables porte $\log_2 10$ bits (ch. 6) : multiplie par le nombre de chiffres, puis exprime le résultat en millions de bits. i) Écris le rapport $\frac{2^{n^2}}{2^{2^n}} = 2^{n^2 - 2^n}$ et compare la croissance de $n^2$ et de $2^n$ : vers quoi va l'exposant, et donc le rapport ?

</details>

### Ex 11.2 — Moyenne incrémentale : Qₙ₊₁ = Qₙ + (Rₙ − Qₙ)/n ∂

<details><summary>Indice 1</summary>

Sépare la dernière récompense de la somme : $\sum_{i=1}^{n} R_i = R_n + \sum_{i=1}^{n-1} R_i$, et exprime la seconde somme avec $Q_n$.

</details>
<details><summary>Indice 2</summary>

$\sum_{i=1}^{n-1} R_i = (n-1)\,Q_n$. Pour 3), suppose la formule vraie au rang $n$, applique $Q_{n+2} = (1 - \alpha) Q_{n+1} + \alpha R_{n+1}$ et développe. Pour 4), la somme des $\alpha(1-\alpha)^{n-i}$ est une somme géométrique.

</details>
<details><summary>Indice 3</summary>

1) $Q_{n+1} = \frac{1}{n}\big(R_n + (n-1)\,Q_n\big) = Q_n + \frac{1}{n}(R_n - Q_n)$ ; pour $n = 1$, $Q_2 = Q_1 + (R_1 - Q_1) = R_1$. 2) Pour chaque méthode, compte les nombres à garder par bras et les opérations de chaque mise à jour. 3) Au rang suivant, $Q_{n+2} = (1-\alpha)\,Q_{n+1} + \alpha R_{n+1}$ : remplace $Q_{n+1}$ par la formule du rang $n$, puis range les termes. 4) $\sum_{i=1}^{n} \alpha(1-\alpha)^{n-i} = \alpha \sum_{k=0}^{n-1} (1-\alpha)^k$ : applique la somme géométrique $\sum_{k=0}^{n-1} q^k = \frac{1 - q^n}{1 - q}$ avec $q = 1 - \alpha$, puis ajoute $(1-\alpha)^n$. Pour le poids le plus fort, regarde comment $\alpha(1-\alpha)^{n-i}$ varie quand $i$ augmente. 5) Lis le coefficient de $Q_1$ dans la formule de 3) ; pour le pas $1/n$, relis la fin de 1).

</details>

### Ex 11.3 — Syllogismes : valides ? solides ? ✏️

<details><summary>Indice 1</summary>

Pour chaque syllogisme, écris les trois propositions sous leurs formes A, E, I ou O, repère le moyen terme (absent de la conclusion), puis applique les quatre règles de l'encadré 🧮 sur la distribution.

</details>
<details><summary>Indice 2</summary>

Une phrase sur un individu (« 9 », « 2 ») se traite comme une universelle. Pour chaque syllogisme, fais l'inventaire des termes distribués, dans chaque prémisse et dans la conclusion (A : le sujet ; E : les deux ; I : aucun ; O : le prédicat), puis passe les quatre règles dans l'ordre. Pour la solidité, cherche une prémisse fausse parmi les syllogismes valides. Pour e), « aucun $M$ n'est $P$ » vide quelles régions ?

</details>
<details><summary>Indice 3</summary>

a) Le sujet et le prédicat se lisent dans la conclusion ; le moyen terme est le seul terme qui en est absent. Dans S1, la conclusion est « aucun serpent n'a de poils » : son sujet est « les serpents » (A), son prédicat « les choses qui ont des poils » (B), et le moyen terme « les reptiles » (C) : CAB. b) Pour chaque syllogisme : la forme (A, E, I ou O) de chaque proposition, l'inventaire des termes distribués (A : le sujet ; E : les deux ; I : aucun ; O : le prédicat), puis les quatre règles dans l'ordre ; si la forme est valide, cherche une prémisse fausse. c) Pour chaque syllogisme non valide, repère la règle qu'il viole, puis lis sa lettre dans la liste de l'énoncé. d) La validité dépend-elle de la vérité de la conclusion, ou seulement de la forme ? Juge la forme de S6 avec les règles de b), ou garde-la et remplace ses catégories par d'autres : des prémisses vraies et une conclusion fausse sont-elles possibles ? e) Sur la figure (a) de la fiche, colorie les régions qui sont à la fois dans $M$ et dans $P$ : ce sont celles que vide « aucun $M$ n'est $P$ ». Le cercle $S$ les coupe-t-il ? Compte-les.

</details>

### Ex 11.4 — Six raisonnements fautifs à diagnostiquer et à réfuter ✏️

<details><summary>Indice 1</summary>

Pour chacun, commence par son type : si les prémisses étaient vraies, la conclusion suivrait-elle nécessairement (déductif), ou seulement probablement (inductif) ? Attention : un raisonnement fautif n'a pas forcément une forme invalide ; une prémisse fausse suffit.

</details>
<details><summary>Indice 2</summary>

1 : où est le moyen terme, et est-il distribué ? 2 : combien de prémisses négatives ? 3 : compare la forme aux quatre formes du syllogisme conditionnel ; si elle est valide, que vaut la prémisse « une fuite rend toujours le score anormal » ? 4 à 6 : quel sophisme de la fiche ? Que faudrait-il mesurer pour conclure ?

</details>
<details><summary>Indice 3</summary>

1 : déductif. Sa forme est « tout $P$ est $M$ ; $S$ est $M$ ; donc $S$ est $P$ », avec $M$ = avoir une loss d'entraînement très basse : $M$ est deux fois le prédicat d'une proposition A, il n'est jamais distribué. C'est un moyen terme non distribué, que réfute un contre-exemple de même forme aux prémisses vraies : « tous les chats sont des mammifères ; mon chien est un mammifère ; donc mon chien est un chat ». Pour les cinq autres, même démarche. 2 : écris les trois propositions sous leurs formes A, E, I ou O, passe les quatre règles (surtout les deux dernières), puis construis un contre-exemple de même forme avec des animaux. 3 : si la forme est l'une des deux formes valides du syllogisme conditionnel, aucun contre-exemple de même forme n'existe : cherche plutôt une situation où il y a une fuite et où le score reste normal. 4 à 6 : nomme le sophisme inductif (fiche §11.5.2) en te demandant d'où vient l'échantillon (4), combien de cas on a observés (5) et ce qu'on fait des cas gênants (6) ; puis dis ce qu'il faudrait mesurer pour conclure.

</details>

### Ex 11.5 — Enquête au phare : réduire le domaine du discours ✏️

<details><summary>Indice 1</summary>

Pars de l'ensemble des six suspects, et retire après chaque indice ceux qu'il innocente. Garde en tête la prémisse de départ : le coupable est sur l'île.

</details>
<details><summary>Indice 2</summary>

L'indice 2 dit « si X est entré, alors peinture » et « pas de peinture sous les semelles de Diego » : quelle forme ? Que prouve exactement l'élimination sur les suspects qui restent : qu'ils ont agi ensemble ? qu'au moins l'un d'eux est coupable ? Pour g), un groupe de coupables est un sous-ensemble non vide des six personnes.

</details>
<details><summary>Indice 3</summary>

a) Pars de {A, B, C, D, E, F}. L'indice 1 innocente Chloé et Elsa : leurs badges les placent sur la falaise nord à 1 h 50 et à 2 h 05, à vingt minutes de marche du phare. Fais de même avec les indices 2 et 3, puis écris les initiales qui restent. b) Écris l'indice 2 sous la forme « si X est entré, alors… ; or… ; donc… » et compare-le aux formes du syllogisme conditionnel (fiche §11.4). c) et d) Écris l'élimination comme un syllogisme disjonctif : « au moins une des six personnes est coupable ; ni Chloé, ni Elsa, ni Diego, ni Félix ne l'est ; donc… ». Complète ce « donc », puis compare-le aux deux affirmations de c) et de d) : laquelle en découle, et un indice dit-il quelque chose de plus ? e) Si Anne était au sommet à 1 h 56 et qu'il lui faut au moins dix minutes pour descendre, peut-elle être dans la salle des machines à 2 h ? f) Quelle prémisse, posée dès le départ, fait des six personnes la liste complète des possibles ? g) à i) Une hypothèse est un sous-ensemble non vide des suspects : avec $k$ suspects, il y en a $2^k - 1$. Applique-le au départ, après l'indice 3, puis après l'indice 4.

</details>

### Ex 11.6 — Syllogisme statistique et prédiction : 15 % de pommes mûres ✏️

<details><summary>Indice 1</summary>

a) à d) : des probabilités de tirage (ch. 3). e) à h) : une proportion estimée sur un panier, son erreur type (ch. 8) et la probabilité du prochain tirage (ch. 4). i) et j) : la façon dont le client remplit son panier.

</details>
<details><summary>Indice 2</summary>

Sans remise, la seconde pomme est tirée parmi 1 999, dont 299 mûres. « Au moins une » : passe par le contraire, « aucune ». Erreur type : $\sqrt{\hat{p}(1-\hat{p})/n}$ ; pour g), isole $n$. Règle de succession : $(h + 1)/(n + 2)$.

</details>
<details><summary>Indice 3</summary>

a) 300 pommes mûres sur 2 000 : $300 / 2\,000 = 0{,}15$ ; on va de la population (l'épicerie) à une pomme tirée au hasard : c'est un syllogisme statistique. b) Sans remise : $P(\text{1re mûre}) \times P(\text{2e mûre} \mid \text{1re mûre})$, la seconde étant tirée parmi les pommes qui restent. c) Avec remise, les deux tirages sont identiques et indépendants : la probabilité de a), deux fois. d) Passe par le contraire : $1 - P(\text{aucune mûre})$, avec $P(\text{aucune mûre}) = (1 - p)^5$. e) La part de pommes mûres dans le panier. f) $\sqrt{\hat{p}(1 - \hat{p})/n}$, avec le $\hat{p}$ de e) et la taille du panier. g) Isole $n$ dans $\sqrt{p(1-p)/n} \le 0{,}01$ en élevant au carré, avec $p = 0{,}15$, puis arrondis à l'entier supérieur. h) $(h + 1)/(n + 2)$, avec $h$ les pommes mûres du panier et $n$ sa taille. i) Où sont les pommes mûres, et où le client se sert-il ? j) L'erreur type mesure le hasard du tirage : dit-elle quelque chose de la **façon** de tirer ?

</details>

### Ex 11.7 — Renforcement ou punition, positif ou négatif : classer huit situations ✏️

<details><summary>Indice 1</summary>

Deux questions pour chaque situation : on ajoute ou on retire quelque chose ? Le comportement en italique devient plus ou moins fréquent ?

</details>
<details><summary>Indice 2</summary>

Pour chaque situation, remplis deux colonnes : « ajouté ou retiré ? » (qu'est-ce qui apparaît, ou disparaît, juste après le comportement en italique ?) et « plus ou moins fréquent ? ». Puis lis la case dans le tableau du §11.7. Attention : en conditionnement opérant, « positif » et « négatif » ont un sens technique ; relis-le dans ce tableau avant de classer. Pour b), relis la fin de la section 11.7 de la fiche.

</details>
<details><summary>Indice 3</summary>

a) Situation 1 : la friandise est **ajoutée** juste après que le chien s'est assis, et il s'assoit plus volontiers : un stimulus ajouté qui rend le comportement plus fréquent, c'est un renforcement positif, A. Fais de même pour les sept autres, en ne jugeant que « ajouté ou retiré ? » et « plus ou moins fréquent ? », jamais « agréable ou désagréable ? ». b) Selon le livre, la correction des poids est-elle ajoutée ou retirée, et doit-elle rendre les erreurs plus fréquentes ou plus rares ? c) Compare le sens que ces deux colonnes donnent à « négatif » avec son sens courant. d) La récompense de 1 est-elle ajoutée ou retirée, et le choix de ce bras devient-il plus ou moins fréquent ?

</details>

### Ex 11.8 — Un bandit à la main : ε-greedy, moyennes et regret ✏️

<details><summary>Indice 1</summary>

Fais le tableau de ta copie pas à pas : explore-t-il ($u_t < 0{,}2$) ? quel bras ? quelle récompense ? quelle nouvelle estimation (moyenne des récompenses du bras) ?

</details>
<details><summary>Indice 2</summary>

Compare chaque $u_t$ à ε pour savoir à quels pas l'agent explore. Quand il exploite, prends le bras de plus grande estimation, et applique la règle d'égalité de l'énoncé. Pseudo-regret : additionne $0{,}8 - q_*(A_t)$ sur les 8 pas. Pour h), le meilleur bras sort quand l'agent exploite, ou quand il explore et tombe dessus. Pour i), l'agent ne perd que quand il explore.

</details>
<details><summary>Indice 3</summary>

a) $u_1 = 0{,}65 \ge 0{,}2$ : l'agent exploite ; les trois estimations valent 0, et l'égalité va au plus petit numéro : il joue le bras 0. Continue le tableau de la même façon, pas à pas : explore-t-il ($u_t < 0{,}2$) ? quel bras joue-t-il (celui du tableau s'il explore, sinon celui de plus grande estimation) ? quelle récompense ? quelle nouvelle moyenne pour ce bras (la somme de ses récompenses divisée par son nombre de tirages) ? b) à e) se lisent dans ce tableau. f) Additionne $0{,}8 - q_*(A_t)$ sur les 8 bras joués. g) $8 \times 0{,}8$ moins la somme des récompenses obtenues. h) Le meilleur bras sort quand l'agent exploite (probabilité $1 - \varepsilon$), ou quand il explore et tombe dessus ($\varepsilon / K$). i) L'agent ne perd qu'en explorant : $\varepsilon$ fois la moyenne des écarts $q_* - q_*(a)$ sur les trois bras. j) Avec $\varepsilon = 0$, l'agent ne fait qu'exploiter : l'estimation du bras 0 peut-elle devenir négative ? Celles des bras 1 et 2 bougent-elles s'il ne les joue jamais ?

</details>

<a id="reflexion"></a>

## 🗣️ 📈 ⚖️ 📄 Réflexion

### Ex 11.9 — Déduction et induction dans un projet de ML, en cinq lignes 🗣️

<details><summary>Indice 1</summary>

Relis la fin de la section 11.6 de la fiche : l'entraînement et l'utilisation d'un modèle.

</details>
<details><summary>Indice 2</summary>

Une ligne pour l'entraînement (induction), une pour l'application du modèle (déduction à partir d'une règle apprise), une pour ce qui garantit (rien, sauf si les données sont représentatives), une pour ce qu'on fait pour s'en assurer (un test), une pour conclure.

</details>
<details><summary>Indice 3</summary>

Les deux premières lignes, comme modèle : « Un modèle apprend une règle à partir d'exemples : c'est une induction, une conclusion probable, jamais certaine. Ensuite, il applique cette règle à chaque nouveau cas : c'est une déduction, qui n'est pas plus sûre que la règle apprise. » Écris les trois autres toi-même : à quelle condition sur les exemples la règle vaut-elle (le mot « représentatif ») ? Comment vérifie-t-on que le modèle sait « généraliser » ? Que réponds-tu, en une phrase, à la question « raisonne-t-il » ?

</details>

### Ex 11.10 — Lire les courbes d'un bandit : ε = 0, 0,01 et 0,1 📈

<details><summary>Indice 1</summary>

Lis d'abord les fins de courbes (au pas 1 000), en haut puis en bas. Pour d) et e), relis l'encadré 🧮 sur ε-greedy.

</details>
<details><summary>Indice 2</summary>

Le plafond de ε-greedy est $1 - \varepsilon + \varepsilon/K$ avec $K = 10$. À très long terme, chaque agent atteint son plafond : lequel est le plus haut ? Pour h), lis sur la courbe du haut la récompense du glouton au pas 1 000 et la valeur de la ligne en tirets, puis fais le rapport.

</details>
<details><summary>Indice 3</summary>

a) Au pas 1 000, sur le graphique du haut, la courbe de ε = 0,1 est au-dessus des deux autres : a) vaut 0,1. b) et c) Sur le graphique du bas, lis le plafond du glouton, puis la valeur de ε = 0,1 au pas 1 000, avec la graduation de l'axe, et range chaque lecture dans les intervalles proposés. d) et e) Le plafond de ε-greedy, $1 - \varepsilon + \varepsilon/K$ avec $K = 10$, pour chaque valeur de ε. f) À très long terme, un agent qui explore atteint son plafond, et le glouton reste bloqué : compare les plafonds de d) et e). g) Que deviennent les estimations des bras que le glouton ne tire plus jamais ? h) Lis au pas 1 000 la récompense moyenne du glouton et la valeur de la ligne en tirets, puis fais le rapport.

</details>

### Ex 11.11 — Explorer sur des humains : essais adaptatifs, recommandation, A/B tests ⚖️

<details><summary>Indice 1</summary>

Un bras est une option qu'on propose (un traitement, un prix, une vidéo) ; la récompense, ce qu'on observe ensuite. Explorer, c'est proposer une option peut-être moins bonne.

</details>
<details><summary>Indice 2</summary>

Pour 2) : un patient de plus reçoit-il le meilleur traitement ? Mais un seul patient dans un bras permet-il de conclure ? Pour 3) : l'équité entre clients, la transparence, la loi. Pour 4) : ce que la récompense mesure vraiment, et ce qu'elle ignore.

</details>
<details><summary>Indice 3</summary>

Pistes : l'exploration a un coût humain réel ; une allocation adaptative profite aux patients de l'essai, mais affaiblit la preuve (petits effectifs, dérive dans le temps, biais) ; un test A/B à proportions fixes donne une estimation plus nette ; une récompense comme le temps de visionnage pousse vers les contenus addictifs ou extrêmes. Règles possibles : un comité qui valide, une exploration bornée, des garde-fous (un plancher par bras, un arrêt pour nocivité), une récompense qui mesure le bien-être.

</details>

### Ex 11.12 — Domingos (2012) : représentation, évaluation, optimisation et autres leçons 📄

<details><summary>Indice 1</summary>

Lis d'abord le tableau 1 de l'article, puis les sections citées dans l'énoncé.

</details>
<details><summary>Indice 2</summary>

Pour 3), cherche la phrase où Domingos cite Wolpert. Pour 4), cherche ce qu'il dit des bornes théoriques et de leur usage. Pour 6), compare les leçons sur l'ingénierie des features et sur la quantité de données à ce que fait l'apprentissage profond.

</details>
<details><summary>Indice 3</summary>

1 : le perceptron, comme modèle : représentation, un hyperplan ; évaluation, le nombre d'erreurs ; optimisation, la règle de correction. Range de même les moindres carrés et le k-means avec le tableau du §11.2 de la fiche, puis retrouve chaque case dans le tableau 1 de l'article. 3 : dans la phrase qui cite Wolpert, sur quel ensemble de fonctions se fait la comparaison avec le hasard ? Qu'en conclut-il sur ce que l'apprenant doit apporter en plus des données ? Pour Hume, relis l'encadré ⚠️ du §11.6. 4 : une borne théorique parle-t-elle du pire cas ou du cas typique ? Que dit-elle de ton problème précis ? 5 : cherche, dans le §11.2.1 de la fiche, l'exemple que le livre range parmi ce qui est représentable sans être apprenable. 6 : pour chaque leçon, demande-toi si l'apprentissage profond l'a affaiblie (il apprend lui-même ses représentations à partir des données brutes) ou renforcée (des modèles entraînés sur d'énormes quantités de données), puis choisis et justifie.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 11.E1 — Exploration contre exploitation : expliquer avec un exemple métier

<details><summary>Indice 1</summary>

Une définition en une phrase, un exemple concret (un restaurant, une recommandation, une campagne marketing), une méthode.

</details>
<details><summary>Indice 2</summary>

Exploiter, c'est jouer la meilleure option connue ; explorer, essayer une option mal connue qui pourrait être meilleure. Trop exploiter fige sur une option médiocre ; trop explorer gaspille. Cite ε-greedy, UCB ou Thompson.

</details>
<details><summary>Indice 3</summary>

Exemple : une plateforme qui choisit la bannière à montrer. Toujours la meilleure connue : on ne découvre jamais une meilleure. Des tests au hasard pour toujours : on perd des clics. Thompson : montrer chaque bannière avec la probabilité qu'elle soit la meilleure ; l'exploration diminue d'elle-même à mesure que les estimations se précisent.

</details>

### 11.E2 — A/B test ou bandit : lequel choisir ?

<details><summary>Indice 1</summary>

Qu'est-ce qu'on veut : une **décision** fiable et une estimation de l'effet, ou **gagner** le plus pendant l'expérience ?

</details>
<details><summary>Indice 2</summary>

Le test A/B, avec des proportions fixes et une taille prévue d'avance, mesure l'écart avec un intervalle de confiance. Le bandit réduit le coût de l'expérience, mais estime moins bien la variante qu'il délaisse, et réagit mal aux effets qui changent avec le temps.

</details>
<details><summary>Indice 3</summary>

Applique le critère de l'indice 1 à la page de paiement : le changement est-il durable ou éphémère ? Faudra-t-il mesurer l'effet précisément, et peut-être le justifier ? Le trafic perdu pendant l'expérience compte-t-il plus que la précision de l'estimation ? Termine par le cas inverse, où l'autre méthode gagne : des choix nombreux, courts ou renouvelés, comme des titres d'articles ou des promotions d'une semaine.

</details>

### 11.E3 — Que dit le théorème « No Free Lunch » pour le choix d'un modèle ?

<details><summary>Indice 1</summary>

L'énoncé, puis ce qu'il ne dit pas, puis la pratique.

</details>
<details><summary>Indice 2</summary>

En moyenne sur tous les problèmes possibles, tous les algorithmes se valent. Les problèmes réels ont une structure : un modèle gagne quand son biais inductif lui correspond. Comment le savoir pour ton problème ?

</details>
<details><summary>Indice 3</summary>

« Aucun modèle n'est le meilleur partout : je pars de choix par défaut éprouvés pour mon type de données (arbres boostés pour le tabulaire, réseaux pré-entraînés pour les images et le texte), avec une baseline simple, et je compare quelques candidats en validation croisée, sans toucher au test. »

</details>

### 11.E4 — Un biais d'échantillonnage qui a fait échouer un modèle : exemple et parade

<details><summary>Indice 1</summary>

Un cas plausible suffit : un modèle entraîné sur une population, utilisé sur une autre.

</details>
<details><summary>Indice 2</summary>

Raconte : la collecte (qui est dans les données, qui n'y est pas), le symptôme en production, la détection (performances par sous-groupe, comparaison des distributions), la parade (collecte, stratification, repondération, surveillance).

</details>
<details><summary>Indice 3</summary>

Exemple : un modèle de détection de maladies de peau entraîné surtout sur des photos de peaux claires, moins bon sur les peaux foncées. Détection : mesurer la performance par sous-groupe dès la validation, comparer la distribution des données de service à celle de l'entraînement. Parade : compléter la collecte, stratifier, repondérer, et surveiller en production.

</details>

<a id="notebook"></a>

## Notebook, parties A à E

### Ex 11.13 — Glouton pur sur trois bras : que va-t-il se passer ? 🔮

<details><summary>Indice 1</summary>

Imagine les premiers pas d'une partie : les trois estimations valent 0. Que se passe-t-il quand un bras rapporte 1 ?

</details>
<details><summary>Indice 2</summary>

Déroule une partie à la main. Que vaut l'estimation d'un bras après un échec ? Après un succès ? Peut-elle redescendre à 0 ensuite ? Compare alors les trois estimations, et applique la règle du glouton.

</details>
<details><summary>Indice 3</summary>

Tout se joue au premier succès. Avant lui, les trois estimations valent 0 et l'agent tire au sort. Après lui, l'estimation du bras gagnant est une moyenne de 0 et de 1 qui contient au moins un 1 : compare-la, à chaque pas suivant, à celles des deux autres bras, puis réponds à b). Pour a), à chaque pas avant ce premier succès, le bras $a$ est tiré avec la probabilité 1/3 et rapporte 1 avec la probabilité $p_a$ : la probabilité que le premier succès tombe sur $a$ est donc proportionnelle à quoi ? Fais le calcul pour le meilleur bras. Pour c), la récompense moyenne finale est la moyenne des $p_a$, chacun pondéré par la probabilité que le bras $a$ soit celui du premier succès.

</details>

### Ex 11.14 — Holmes déduit-il ? Compter et citer le vocabulaire du raisonnement 🔨

<details><summary>Indice 1</summary>

`re.findall` avec un motif qui commence par `\b` (début de mot), le radical, puis `\w*` (la fin du mot), et l'option `re.IGNORECASE`.

</details>
<details><summary>Indice 2</summary>

`collections.Counter(mot.lower() for mot in re.findall(rf"\b{stem}\w*", text, flags=re.IGNORECASE))`. Pour Verne, vérifie que ton motif garde les lettres accentuées à l'intérieur des mots (en Python, `\w` les accepte ; `[a-z]` non). Pour `story_of_14`, construis le motif `r"\s+".join(re.escape(partie) for partie in phrase.split())` et cherche-le dans le texte de chaque nouvelle.

</details>
<details><summary>Indice 3</summary>

Pour b), affiche le `Counter` de la famille « infer » et cherche le mot qui commence par ces lettres sans venir du verbe *to infer* (un dictionnaire tranche en cas de doute). La fréquence pour 10 000 mots : `sum(counts.values()) / len(words(holmes)) * 10_000`. Pour `story_of_14`, parcours `stories.items()` et renvoie le numéro de la première nouvelle où `re.search(pattern, story, flags=re.IGNORECASE)` trouve quelque chose, `None` sinon.

</details>

### Ex 11.15 — Valider un syllogisme par force brute : 256 mondes de Venn 🔨

<details><summary>Indice 1</summary>

Un monde garde ou non chacune des 8 régions : `itertools.product([False, True], repeat=8)` donne les 256 masques ; garde les régions dont le masque vaut `True`.

</details>
<details><summary>Indice 2</summary>

Avec `i, j = TERMS_15[x], TERMS_15[y]` : « quelque $x$ est $y$ » = `any(r[i] and r[j] for r in world)` ; « quelque $x$ n'est pas $y$ » = `any(r[i] and not r[j] for r in world)` ; A est la négation de O, E celle de I. Pour `valid_forms_15`, trois boucles : la figure, puis les trois formes (`itertools.product("AEIO", repeat=3)`).

</details>
<details><summary>Indice 3</summary>

`counterexamples_15` : `[w for w in worlds_15() if all(holds_15(p, w) for p in premises) and not holds_15(conclusion, w)]`. Codage de S2, comme modèle : `([("I", "M", "P"), ("A", "S", "M")], ("I", "S", "P"))`, avec $S$ = les rapports d'audit (le sujet de la conclusion), $P$ = les brouillons (son prédicat) et $M$ = les documents confidentiels ; la majeure est la prémisse qui contient $P$, même si l'énoncé l'écrit en second. Pour S3, S6 et S7, même méthode : repère $S$ et $P$ dans la conclusion, puis écris chaque proposition `(forme, x, y)` dans le sens de la phrase (dans S1, « aucun reptile n'a de poils » donne `("E", "M", "P")`). Avec `existence=True`, ajoute `("I", t, t)` pour `t` dans `"SMP"`.

</details>

### Ex 11.16 — Reproduire la figure 11.5 : les cinq sophismes en diagrammes 🎨

<details><summary>Indice 1</summary>

Commence par le cadre commun aux cinq diagrammes : le grand cadre « animaux et meubles », la boîte « à quatre pattes », l'ellipse « chats » dans la boîte.

</details>
<details><summary>Indice 2</summary>

Pour chaque sophisme, demande-toi où doit être le contre-exemple : dans la boîte « à quatre pattes » mais hors des chats (un chien, une table), ou dans les mammifères mais hors de la boîte (la baleine). Ajoute l'ellipse dont le sophisme parle (chiens, mammifères, tables).

</details>
<details><summary>Indice 3</summary>

Les deux premiers : le point violet dans la boîte, hors de l'ellipse des chats. Majeur illicite : une ellipse « chiens » dans la boîte, à côté des chats, le point dedans. Moyen terme non distribué : pareil avec « tables ». Mineur illicite : une boîte « à quatre pattes » plus étroite, une grande ellipse « mammifères » qui contient les chats et déborde de la boîte, le point dans la partie qui déborde.

</details>

### Ex 11.17 — Généralisation hâtive et échantillon biaisé chez les manchots 🔬

<details><summary>Indice 1</summary>

`is_gentoo.mean()` donne une proportion. Pour les échantillons, tire des **indices** avec remise, puis fais la moyenne de `is_gentoo` sur ces indices.

</details>
<details><summary>Indice 2</summary>

`rng.integers(len(is_gentoo), size=(n_samples, n))` donne tous les indices d'un coup ; `is_gentoo[indices].mean(axis=1)`, les parts. Pour Biscoe, un masque `island == "Biscoe"`. Pour la capture proportionnelle à la masse, l'espérance vaut $\sum_i m_i \, P(m_i)$ avec $P(m_i) = m_i / \sum_j m_j$.

</details>
<details><summary>Indice 3</summary>

`np.sum(m * m / m.sum())`, soit $\sum m^2 / \sum m$. Pour f), demande-toi si dix fois plus de manchots capturés de la même façon changeraient la loi des manchots capturés.

</details>

### Ex 11.18 — Des points sur un cercle : quand le modèle trahit l'induction 🔬

<details><summary>Indice 1</summary>

L'équation $x^2 + y^2 + Dx + Ey + F = 0$ est linéaire en $D$, $E$, $F$ : c'est une régression linéaire à trois inconnues.

</details>
<details><summary>Indice 2</summary>

`A = np.column_stack([x, y, np.ones_like(x)])` et le second membre `-(x**2 + y**2)` ; `np.linalg.lstsq(A, b, rcond=None)[0]` donne $(D, E, F)$.

</details>
<details><summary>Indice 3</summary>

Centre $(-D/2, -E/2)$ ; rayon $\sqrt{c_x^2 + c_y^2 - F}$, car $(x - c_x)^2 + (y - c_y)^2 = r^2$ se développe en $x^2 + y^2 - 2c_x x - 2c_y y + c_x^2 + c_y^2 - r^2 = 0$.

</details>

### Ex 11.19 — BernoulliBandit et GaussianBandit 🔨

<details><summary>Indice 1</summary>

Les constructeurs sont fournis : il ne reste que `pull`. Utilise le générateur `self._rng`, jamais `np.random`.

</details>
<details><summary>Indice 2</summary>

Vérifie d'abord `0 <= arm < self.n_arms`, sinon `raise IndexError(...)`. Bernoulli : `self._rng.random() < self.means[arm]` vaut `True` avec la probabilité voulue. Gaussien : `self._rng.normal(moyenne, écart-type)`.

</details>
<details><summary>Indice 3</summary>

`return float(self._rng.random() < self.means[arm])` et `return float(self._rng.normal(self.means[arm], self.std))`. Le `float(...)` transforme le booléen ou le flottant NumPy en `float` Python.

</details>

### Ex 11.20 — argmax_random_tie, epsilon_greedy_action et incremental_update 🔨

<details><summary>Indice 1</summary>

Trois fonctions courtes. Commence par les vérifications (`ValueError`), puis le cas `rng is None`.

</details>
<details><summary>Indice 2</summary>

Ex aequo : `best = np.flatnonzero(values == values.max())`. ε-greedy : `if rng.random() < epsilon:` un bras `rng.integers(n_bras)`, sinon `argmax_random_tie`. Un `nan` se détecte avec `np.isnan(values).any()`.

</details>
<details><summary>Indice 3</summary>

`return int(best[0]) if best.size == 1 else int(rng.choice(best))` ; `return estimate + step_size * (target - estimate)` après avoir vérifié `0 < step_size <= 1`.

</details>

### Ex 11.21 — run_bandit : la boucle d'interaction et ses courbes 🔨

<details><summary>Indice 1</summary>

Prépare `q_values` (rempli de `initial_value`), `counts` (des zéros entiers) et des tableaux pour les bras joués et les récompenses ; puis une boucle `for t in range(1, n_steps + 1)`.

</details>
<details><summary>Indice 2</summary>

Dans la boucle : choisir, vérifier le bras, tirer, compter, mettre à jour. Après la boucle : `optimal = means[actions] == bandit.best_mean`, `regret = np.cumsum(bandit.best_mean - means[actions])`. Pour `testbed_21`, une boucle sur les parties qui additionne `history["rewards"]` et `history["optimal"]`.

</details>
<details><summary>Indice 3</summary>

`step = 1.0 / counts[arm] if step_size is None else step_size` puis `q_values[arm] = incremental_update(q_values[arm], reward, step)`. Dans `testbed_21` : `GaussianBandit(rng.normal(0, 1, size=10), std=1.0, random_state=int(rng.integers(2**32)))` et `run_bandit(bandit, policy, n_steps, rng=np.random.default_rng(int(rng.integers(2**32))))`.

</details>

### Ex 11.22 — Initialisation optimiste sans ε : prédire, puis mesurer 🔮

<details><summary>Indice 1</summary>

Suis les estimations de l'agent optimiste pendant les premiers pas : toutes valent 5, et chaque tirage fait baisser celle du bras tiré.

</details>
<details><summary>Indice 2</summary>

Avec le pas 0,1, que devient l'estimation d'un bras après son premier tirage, comparée à celle d'un bras jamais tiré ? Déroule les premiers pas d'une partie à la main, avec 10 bras. Avec la moyenne exacte, que reste-t-il de la valeur 5 après un tirage (∂ 11.2, question 5) ?

</details>
<details><summary>Indice 3</summary>

Avec le pas constant, le premier tirage d'un bras fait passer son estimation de 5 à $5 + 0{,}1\,(R - 5)$, autour de 4,5 : sous la valeur 5 des bras jamais tirés. Combien de pas faut-il pour que chacun des 10 bras ait été tiré une fois ? Au pas suivant, quel bras l'agent choisit-il, et est-ce souvent le meilleur ? Pour a), compare où chaque agent dépense son exploration : l'agent réaliste y consacre 10 % de ses choix, à chaque pas, jusqu'au bout ; l'agent optimiste explore tant que ses estimations n'ont pas oublié la valeur 5 (le poids $(1 - \alpha)^n$ de ∂ 11.2). Pour c), avec la moyenne exacte, le premier tirage efface la valeur 5 (∂ 11.2, question 5) : l'agent essaie chaque bras une fois, puis joue en glouton sur des estimations tirées d'une seule récompense, d'écart-type 1. Rattrape-t-il un bon bras sous-estimé par une première récompense malchanceuse ? Compare-le au glouton de 📈 11.10 et à l'agent optimiste à pas constant.

</details>

### Ex 11.23 — ucb_action et thompson_action 🔨

<details><summary>Indice 1</summary>

UCB : un cas particulier (un bras jamais tiré) avant la formule. Thompson : un tirage par bras, puis un argmax.

</details>
<details><summary>Indice 2</summary>

`untried = np.flatnonzero(counts == 0)` ; s'il n'est pas vide, renvoie `int(untried[0])`. Sinon, `np.argmax(q + c * np.sqrt(np.log(t) / counts))`. Thompson : `rng.beta(1 + successes, 1 + failures)` accepte des tableaux.

</details>
<details><summary>Indice 3</summary>

Vérifie les formes (`np.shape`), `t >= 1`, `c >= 0` et l'absence de compteurs négatifs, puis `return int(np.argmax(rng.beta(1.0 + successes, 1.0 + failures)))`. Pour retrouver les succès dans `run_bandit` : `np.rint(q * n)`.

</details>

### Ex 11.24 — Bandit piégé : l'agent qui n'explore jamais 🐛

<details><summary>Indice 1</summary>

Appelle plusieurs fois `select` sur l'agent du collègue sans rien mettre à jour : le hasard change-t-il d'un appel à l'autre ?

</details>
<details><summary>Indice 2</summary>

Lis chaque ligne en te demandant « que vaut-elle pour le dernier bras ? », « que donne `np.argmax` sur des égalités ? », « par quoi divise-t-on : le nombre total de pas ou le nombre de tirages du bras ? ».

</details>
<details><summary>Indice 3</summary>

Le squelette de la classe corrigée, avec sa ligne clé, celle du bug qui masque les autres :

```python
class FixedAgent24(BuggyAgent24):
    def __init__(self, n_arms, epsilon=0.1, seed=0):
        super().__init__(n_arms, epsilon, seed)
        self.rng = np.random.default_rng(seed)          # bug 1: ONE generator, created once, used by select
    # select: explore with self.rng.integers(...), whose upper bound is excluded: can every arm come out?
    #         exploit with a random choice among np.flatnonzero(self.q == self.q.max())
    # update: the mean of the rewards of THIS arm: count its pull first, then divide by what?
```

</details>

### Ex 11.25 — Un journal d'expériences reproductible (JSON) 🛠️

<details><summary>Indice 1</summary>

Un seul générateur, créé avec `config["seed"]`, d'où l'on tire tout le reste, toujours dans le même ordre.

</details>
<details><summary>Indice 2</summary>

`json.dumps(record, indent=2)` ; les versions avec `platform.python_version()` et `np.__version__` ; la date avec `datetime.datetime.now(datetime.timezone.utc).isoformat()` ; le commit avec `subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)`.

</details>
<details><summary>Indice 3</summary>

Convertis chaque résultat avec `float(...)`. `rerun_25` : `record = json.loads(Path(path).read_text())`, puis `return run_experiment_25(record["config"]) == record["results"]`.

</details>

### Ex 11.26 — Tournoi : ε-greedy, optimiste, UCB et Thompson 🔬

<details><summary>Indice 1</summary>

Une politique est une fonction `(q, n, t, g) -> bras` ; les réglages de la boucle (valeur initiale, pas) vont dans les arguments de `run_bandit`.

</details>
<details><summary>Indice 2</summary>

`"optimiste": (lambda q, n, t, g: mylearn.bandit.argmax_random_tie(q, g), {"initial_value": 1.0, "step_size": 0.1})`. Dans `tournament_26`, pour chaque agent : `for i, bandit in enumerate(make_bench(n_bandits, seed))`, puis `run_bandit(..., rng=np.random.default_rng(i), **kwargs)["regret"][-1]`.

</details>
<details><summary>Indice 3</summary>

L'erreur type : `np.std(finals, ddof=1) / np.sqrt(len(finals))`. Lis ensuite le classement affiché pour remplir `best_26` et `ucb1_beats_eps_26`.

</details>

### Ex 11.27 — Défi : battre UCB1 sur un banc de bandits de Bernoulli 🏆

<details><summary>Indice 1</summary>

Un agent au hasard perd, à chaque pas, l'écart entre le meilleur bras et la moyenne de tous les bras. Pour la stratégie : le tournoi de 11.26 a-t-il un vainqueur qui suffirait déjà ?

</details>
<details><summary>Indice 2</summary>

`random_regret_27` : la moyenne sur les bandits de `n_steps * (bandit.best_mean - np.mean(bandit.means))`. Les moyennes du banc sont uniformes : quel prior Beta leur correspond exactement ?

</details>
<details><summary>Indice 3</summary>

Thompson avec le prior $\mathrm{Beta}(1, 1)$ atteint l'objectif. Pour le palier 🌟, essaie un UCB moins prudent ($c$ autour de 0,3, réglé sur le banc public seulement) ou un UCB bayésien (le quantile $1 - 1/t$ du posterior, `scipy.stats.beta.ppf`).

</details>
