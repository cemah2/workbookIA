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

a) 1 et 4 cherchent (O), 2 et 5 jugent (E), 3 et 6 décrivent ce que le modèle peut exprimer (R). b) Le test « mal classé ? » juge ; la mise à jour et le pas cherchent ; l'hyperplan est la représentation.

</details>

### 11.Q2 — Puissance de représentation : ce qu'un perceptron ne peut pas « savoir »

<details><summary>Indice 1</summary>

Un perceptron répond selon le signe de $w_1 x_1 + w_2 x_2 + b$ : sa frontière est toujours une **droite**. Pour chaque règle, la frontière est-elle une droite ?

</details>
<details><summary>Indice 2</summary>

Écris la frontière de chaque règle (remplace « < » ou « > » par « = »), puis demande-toi si c'est une droite, c'est-à-dire une équation de la forme $a x_1 + b x_2 = c$ ($a$ ou $b$ peut être nul, pas les deux). Pour la règle D, dessine la zone de la classe 1 : une seule droite peut-elle la séparer du reste ? Pour b), réécris chaque règle comme une somme pondérée de $x_1$, $x_2$, $x_1^2$, $x_2^2$, si c'est possible.

</details>
<details><summary>Indice 3</summary>

a) A, C et E. b) Le disque : $x_1^2 + x_2^2 < 1$ est linéaire en $x_1^2$ et $x_2^2$ (poids $-1$, $-1$, biais 1) ; « même signe » demanderait la feature $x_1 x_2$. c) Faux : l'overfitting du ch. 9.

</details>

### 11.Q3 — Représentable mais pas apprenable : le problème de l'arrêt

<details><summary>Indice 1</summary>

Relis l'encadré ⚠️ sur le problème de l'arrêt (fiche §11.2.1) : sur quoi porte exactement l'impossibilité, un programme précis ou une méthode pour tous les programmes ?

</details>
<details><summary>Indice 2</summary>

Pour c), distingue « on ne peut jamais le prouver pour un programme donné » et « aucune méthode ne le prouve pour tous les programmes ». Pour d), qu'est-ce qui empêcherait le programme de s'arrêter juste après qu'on a cessé d'attendre ? Pour e), la réponse « oui » ou « non » se range-t-elle facilement ? Quelqu'un peut-il la calculer partout ?

</details>
<details><summary>Indice 3</summary>

a) Vrai. b) Vrai. c) Faux : `while True: pass`. d) Faux. e) B : représentable (un oui ou un non par couple), mais aucun algorithme ne la calcule partout, donc aucun ne l'apprend exactement.

</details>

### 11.Q4 — Loss, métrique, objectif : qui sert à quoi ?

<details><summary>Indice 1</summary>

Relis les trois définitions de la section 11.2.2 de la fiche, puis l'encadré ⚠️ sur l'accuracy, la precision et le recall.

</details>
<details><summary>Indice 2</summary>

Qui fait baisser quoi pendant l'entraînement ? Quel nombre calcule-t-on sur des données mises de côté ? Lequel parle d'argent ou de délais ? Pour b), que devient l'accuracy quand on change un peu un poids ? Pour c) et d) : la precision part des alertes données, le recall des vraies fraudes ; de quoi part chaque souhait ?

</details>
<details><summary>Indice 3</summary>

a) Entropie croisée : L ; recall sur le test et precision sur la validation : M ; diviser les fraudes : O. b) A : l'accuracy est en escalier. c) La precision. d) Le recall.

</details>

### 11.Q5 — Optimiser n'est pas être optimal ; pas de repas gratuit

<details><summary>Indice 1</summary>

Relis la section 11.2.3 de la fiche : l'image du vélo et du train, le minimum local, et le paragraphe qui dit comment on cite souvent mal le théorème.

</details>
<details><summary>Indice 2</summary>

Une suite de pas qui améliorent peut-elle s'arrêter ailleurs qu'au meilleur endroit (ch. 5) ? Sur quoi le théorème No Free Lunch fait-il une moyenne : sur ton problème, ou sur tous les problèmes possibles ?

</details>
<details><summary>Indice 3</summary>

a) Faux (minimum local). b) Faux : la moyenne porte sur tous les problèmes possibles, et les problèmes réels ont une structure. c) B.

</details>

### 11.Q6 — Déduction ou induction ? Six situations

<details><summary>Indice 1</summary>

Pour chaque situation, demande-toi : si les prémisses sont vraies, la conclusion peut-elle être fausse ? Si oui, c'est une induction.

</details>
<details><summary>Indice 2</summary>

Pour chaque situation, repère les prémisses et la conclusion. La conclusion dit-elle quelque chose que les prémisses ne contenaient pas déjà (un cas non observé, une règle générale, l'avenir) ? Si oui, elle peut être fausse alors que les prémisses sont vraies. Si elle ne fait que tirer ce que les prémisses disaient déjà, elle est nécessaire.

</details>
<details><summary>Indice 3</summary>

Situations 1, 3 et 4 : induction ; 2, 5 et 6 : déduction.

</details>

### 11.Q7 — Valide, solide, ou ni l'un ni l'autre ?

<details><summary>Indice 1</summary>

D'abord la forme (valide ?), avec les règles de distribution de la fiche ; ensuite seulement les prémisses (vraies ?).

</details>
<details><summary>Indice 2</summary>

Pour la forme des syllogismes catégoriques, applique les quatre règles de distribution de l'encadré 🧮 du §11.4, en commençant par le moyen terme (est-il distribué au moins une fois ?). Le 6 est un syllogisme conditionnel : compare-le aux quatre formes du §11.4, deux valides et deux sophismes. Pour chaque syllogisme valide, vérifie ensuite ses prémisses une par une : connais-tu un cas réel qui en contredit une ?

</details>
<details><summary>Indice 3</summary>

1 : S. 2 : V (les manchots ne volent pas). 3 : N. 4 : V (2 est pair et premier). 5 : N. 6 : N (affirmation du conséquent). b) Vrai.

</details>

### 11.Q8 — Nommer le sophisme syllogistique

<details><summary>Indice 1</summary>

Sépare d'abord les raisonnements conditionnels (« si… alors… ») des raisonnements catégoriques (« tous les… », « aucun… »).

</details>
<details><summary>Indice 2</summary>

Conditionnels : part-on du conséquent (le « alors » est vrai) ou nie-t-on l'antécédent (le « si » est faux) ? Catégoriques : le moyen terme est-il distribué ? Le sujet ou le prédicat de la conclusion est-il distribué dans la conclusion sans l'être dans sa prémisse ?

</details>
<details><summary>Indice 3</summary>

1 : affirmation du conséquent. 3 : négation de l'antécédent. 2 : majeur illicite (« mammifère », prédicat de la conclusion, distribué à tort). 5 : mineur illicite (« rectangles », sujet de la conclusion, distribué à tort). 4 : moyen terme non distribué.

</details>

### 11.Q9 — Généralisation, syllogisme statistique, prédiction

<details><summary>Indice 1</summary>

Relis le tableau de la section 11.5 de la fiche : de quoi part chaque raisonnement (échantillon ou population) et vers quoi va-t-il (population, individu, prochain cas) ?

</details>
<details><summary>Indice 2</summary>

Pour chaque phrase, souligne d'où part le raisonnement (un échantillon, ou toute la population ?) et de qui parle la conclusion (toute la population, un individu tiré au hasard, ou le prochain cas observé ?). Pour c), relis la section 11.5.1 sur la dérive des données.

</details>
<details><summary>Indice 3</summary>

a) S, G, P, G. b) A (un échantillon représentatif). c) Faux.

</details>

### 11.Q10 — Sophismes inductifs chez les data scientists

<details><summary>Indice 1</summary>

Prends les définitions **usuelles** du tableau de la section 11.5.2 de la fiche, pas celles du livre.

</details>
<details><summary>Indice 2</summary>

Pour chaque situation, cherche le défaut : trop peu de cas ? une collecte qui choisit certains individus ? une anecdote frappante ? des données nettes qu'on refuse ? une règle criblée d'exceptions ? une exception qu'on s'accorde à soi-même ?

</details>
<details><summary>Indice 3</summary>

1 : vivacité trompeuse. 2 : échantillon biaisé (ceux qui laissent un avis). 3 : plaidoyer spécial. 4 : généralisation hâtive. 5 : exception écrasante. 6 : induction paresseuse.

</details>

### 11.Q11 — Prémisses rationnelles, empiriques, et la fourche de Hume

<details><summary>Indice 1</summary>

Pour chaque prémisse : peut-on la savoir vraie sans rien observer, par la seule raison ou par définition ?

</details>
<details><summary>Indice 2</summary>

Pour chaque prémisse, imagine comment tu la vérifierais : par une démonstration ou par le sens des mots, ou par une mesure ? Une prémisse qu'une expérience pourrait un jour contredire est empirique. Pour b) et c), relis l'encadré ⚠️ sur Hume (fiche §11.6).

</details>
<details><summary>Indice 3</summary>

a) R, E, R, E, R. b) B. c) Vrai : la déduction transmet la vérité des prémisses, elle ne l'augmente pas.

</details>

### 11.Q12 — Holmes déduit-il vraiment ?

<details><summary>Indice 1</summary>

Relis la section 11.6.1 de la fiche, et la définition de l'abduction.

</details>
<details><summary>Indice 2</summary>

Les entailles d'une chaussure ont-elles une seule cause possible ? Pour b), si la liste des possibles est complète, l'élimination peut-elle se tromper ? Pour c), d'où part-on : des données ou d'une théorie ?

</details>
<details><summary>Indice 3</summary>

a) B. b) A. c) A. d) Faux : suivre des regards pour deviner une pensée donne une explication plausible, pas une conclusion nécessaire.

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

a) Faux ($y\,z = 1 > 0$). b) $(1 + 1, -1 + 2) = (2, 1)$. c) 1. d) Faux.

</details>

### 11.R2 — Ch. 8 : représentativité du jeu d'entraînement et fuite de données

<details><summary>Indice 1</summary>

Un jeu de test estime l'erreur sur la population dont il est tiré. En 2026, est-ce encore la même population ?

</details>
<details><summary>Indice 2</summary>

Pour b), la moyenne et l'écart-type calculés sur tout le jeu contiennent-ils de l'information du test ? Pour c), que garde un découpage stratifié d'un jeu à l'autre ? Pour d), quelle feature n'est connue qu'après la transaction ?

</details>
<details><summary>Indice 3</summary>

a) B. b) Vrai. c) $200 \times 0{,}3 = 60$. d) C, la date de clôture du dossier.

</details>

### 11.R3 — Ch. 4 : mettre à jour sa croyance sur une pièce avec Bayes

<details><summary>Indice 1</summary>

Avec le prior $\mathrm{Beta}(1, 1)$, après $h$ faces et $t$ piles, le posterior est $\mathrm{Beta}(1 + h, 1 + t)$.

</details>
<details><summary>Indice 2</summary>

La moyenne de $\mathrm{Beta}(a, b)$ vaut $a/(a + b)$ ; son mode, $(a - 1)/(a + b - 2)$. Pour e), compare $\frac{h+1}{n+2} - \frac{1}{2}$ et $\frac{h}{n} - \frac{1}{2}$. Pour f), relis la règle de succession de Laplace (ch. 4).

</details>
<details><summary>Indice 3</summary>

a) $[8, 4]$. b) $8/12 \approx 0{,}67$. c) $7/10 = 0{,}70$. d) $8/14 \approx 0{,}57$. e) Vrai : les deux écarts ont le même numérateur $2h - n$, et le premier un plus grand dénominateur. f) B.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 11.1 — Représentable sur n bits : compter, puis conclure

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « compter ce qu'une représentation peut contenir » (fiche §11.2.1).

</details>
<details><summary>Indice 2</summary>

$n$ bits écrivent $2^n$ valeurs, de 0 à $2^n - 1$ sans signe, de $-2^{n-1}$ à $2^{n-1} - 1$ en complément à deux. Pour d), combien de valeurs de 0 à 1 000 ? Une fonction booléenne choisit 0 ou 1 pour chacune des $2^n$ entrées possibles. Pour h), un symbole parmi 10 équiprobables porte $\log_2 10$ bits (ch. 6).

</details>
<details><summary>Indice 3</summary>

a) 256. b) 255. c) $-128$. d) $\lceil \log_2 1001 \rceil = 10$. e) $2^{2^3} = 256$. f) $104/256$. g) $1882/2^{16}$. h) $\log_2 10 \approx 3{,}32$ millions de bits. i) Vrai : $2^{n^2} / 2^{2^n} \to 0$.

</details>

### Ex 11.2 — Moyenne incrémentale : Qₙ₊₁ = Qₙ + (Rₙ − Qₙ)/n

<details><summary>Indice 1</summary>

Sépare la dernière récompense de la somme : $\sum_{i=1}^{n} R_i = R_n + \sum_{i=1}^{n-1} R_i$, et exprime la seconde somme avec $Q_n$.

</details>
<details><summary>Indice 2</summary>

$\sum_{i=1}^{n-1} R_i = (n-1)\,Q_n$. Pour 3), suppose la formule vraie au rang $n$, applique $Q_{n+2} = (1 - \alpha) Q_{n+1} + \alpha R_{n+1}$ et développe. Pour 4), la somme des $\alpha(1-\alpha)^{n-i}$ est une somme géométrique.

</details>
<details><summary>Indice 3</summary>

$Q_{n+1} = \frac{1}{n}\big(R_n + (n-1) Q_n\big) = Q_n + \frac{1}{n}(R_n - Q_n)$ ; pour $n = 1$, $Q_2 = Q_1 + (R_1 - Q_1) = R_1$. Somme des poids : $(1-\alpha)^n + \alpha\,\frac{1 - (1-\alpha)^n}{\alpha} = 1$. Le poids de $Q_1$ vaut $(1-\alpha)^n$ ; avec le pas $1/n$, il est nul dès le premier tirage.

</details>

### Ex 11.3 — Syllogismes : valides ? solides ?

<details><summary>Indice 1</summary>

Pour chaque syllogisme, écris les trois propositions sous leurs formes A, E, I ou O, repère le moyen terme (absent de la conclusion), puis applique les quatre règles de l'encadré 🧮 sur la distribution.

</details>
<details><summary>Indice 2</summary>

Une phrase sur un individu (« 9 », « 2 ») se traite comme une universelle. Pour chaque syllogisme, fais l'inventaire des termes distribués, dans chaque prémisse et dans la conclusion (A : le sujet ; E : les deux ; I : aucun ; O : le prédicat), puis passe les quatre règles dans l'ordre. Pour la solidité, cherche une prémisse fausse parmi les syllogismes valides. Pour e), « aucun $M$ n'est $P$ » vide quelles régions ?

</details>
<details><summary>Indice 3</summary>

a) C, A, B. b) S1 : S ; S2 : N ; S3 : S ; S4 : N ; S5 : V ; S6 : N ; S7 : N. c) S2 : A ; S4 : B ; S6 : A ; S7 : D. d) Faux. e) 2 (l'intersection de $M$ et de $P$, dans $S$ et hors de $S$).

</details>

### Ex 11.4 — Six raisonnements fautifs à diagnostiquer et à réfuter

<details><summary>Indice 1</summary>

Les trois premiers sont déductifs, les trois derniers inductifs. Attention : un raisonnement fautif n'a pas forcément une forme invalide ; une prémisse fausse suffit.

</details>
<details><summary>Indice 2</summary>

1 : où est le moyen terme, et est-il distribué ? 2 : combien de prémisses négatives ? 3 : compare la forme aux quatre formes du syllogisme conditionnel ; si elle est valide, que vaut la prémisse « une fuite rend toujours le score anormal » ? 4 à 6 : quel sophisme de la fiche ? Que faudrait-il mesurer pour conclure ?

</details>
<details><summary>Indice 3</summary>

1 : moyen terme non distribué (un modèle simple bien réglé peut avoir une loss basse sans surapprendre). 2 : prémisses exclusives ; la conclusion ne suit de rien (« aucun poisson n'aboie ; le chat n'est pas un poisson ; donc le chat aboie »). 3 : forme valide (*modus tollens*), mais majeure fausse : une petite fuite ne se voit pas dans le score. 4 : échantillon biaisé. 5 : généralisation hâtive. 6 : sélection des données favorables (« exception écrasante » au sens du livre), ou plaidoyer spécial : les deux lectures se défendent.

</details>

### Ex 11.5 — Enquête au phare : réduire le domaine du discours

<details><summary>Indice 1</summary>

Pars de l'ensemble des six suspects, et retire après chaque indice ceux qu'il innocente. Garde en tête la prémisse de départ : le coupable est sur l'île.

</details>
<details><summary>Indice 2</summary>

L'indice 2 dit « si X est entré, alors peinture » et « pas de peinture sous les semelles de Diego » : quelle forme ? L'élimination prouve-t-elle une complicité, ou seulement « au moins un » ? Pour g), un groupe de coupables est un sous-ensemble non vide des six personnes.

</details>
<details><summary>Indice 3</summary>

Indice 1 : Chloé et Elsa ; indice 2 : Diego (modus tollens) ; indice 3 : Félix. Restent A et B. c) Faux ; d) Vrai. e) B. f) A. g) $2^6 - 1 = 63$. h) $2^2 - 1 = 3$. i) 1.

</details>

### Ex 11.6 — Syllogisme statistique et prédiction : 15 % de pommes mûres

<details><summary>Indice 1</summary>

a) à d) : des probabilités de tirage (ch. 3). e) à h) : une généralisation, son erreur type (ch. 8) et une prédiction (ch. 4). i) et j) : un échantillon biaisé.

</details>
<details><summary>Indice 2</summary>

Sans remise, la seconde pomme est tirée parmi 1 999, dont 299 mûres. « Au moins une » : passe par le contraire, « aucune ». Erreur type : $\sqrt{\hat{p}(1-\hat{p})/n}$ ; pour g), isole $n$. Règle de succession : $(h + 1)/(n + 2)$.

</details>
<details><summary>Indice 3</summary>

a) 0,15. b) $\frac{300}{2000} \times \frac{299}{1999} \approx 0{,}02244$. c) $0{,}15^2 = 0{,}02250$. d) $1 - 0{,}85^5 \approx 0{,}556$. e) 0,15. f) $\sqrt{0{,}15 \times 0{,}85 / 40} \approx 0{,}0565$. g) $0{,}1275 / 0{,}01^2 = 1\,275$ (avec remise). h) $7/42 \approx 0{,}167$. i) Vrai. j) Faux.

</details>

### Ex 11.7 — Renforcement ou punition, positif ou négatif : classer huit situations

<details><summary>Indice 1</summary>

Deux questions pour chaque situation : on ajoute ou on retire quelque chose ? Le comportement en italique devient plus ou moins fréquent ?

</details>
<details><summary>Indice 2</summary>

Pour chaque situation, remplis deux colonnes : « ajouté ou retiré ? » (qu'est-ce qui apparaît, ou disparaît, juste après le comportement en italique ?) et « plus ou moins fréquent ? ». Puis lis la case dans le tableau du §11.7. Attention : « négatif » veut dire « retiré », pas « désagréable ». Pour b), relis la fin de la section 11.7 de la fiche.

</details>
<details><summary>Indice 3</summary>

a) A, B, D, D, C, A, B, C. b) C (punition positive, selon le livre). c) Faux. d) A.

</details>

### Ex 11.8 — Un bandit à la main : ε-greedy, moyennes et regret

<details><summary>Indice 1</summary>

Fais le tableau de ta copie pas à pas : explore-t-il ($u_t < 0{,}2$) ? quel bras ? quelle récompense ? quelle nouvelle estimation (moyenne des récompenses du bras) ?

</details>
<details><summary>Indice 2</summary>

Compare chaque $u_t$ à ε pour savoir à quels pas l'agent explore. Quand il exploite, prends le bras de plus grande estimation, et applique la règle d'égalité de l'énoncé. Pseudo-regret : additionne $0{,}8 - q_*(A_t)$ sur les 8 pas. Pour h), le meilleur bras sort quand l'agent exploite, ou quand il explore et tombe dessus. Pour i), l'agent ne perd que quand il explore.

</details>
<details><summary>Indice 3</summary>

Bras joués : 0, 2, 2, 2, 1, 1, 2, 2. b) $[0, 0, 2/3]$. d) $[0;\ 0{,}5;\ 0{,}8]$. e) $[1, 2, 5]$. f) $0{,}5 + 0{,}3 + 0{,}3 = 1{,}1$. g) $6{,}4 - 5 = 1{,}4$. h) $1 - 0{,}2 + 0{,}2/3 \approx 0{,}867$. i) $0{,}2 \times (0{,}5 + 0{,}3 + 0)/3 \approx 0{,}053$. j) Vrai.

</details>

<a id="reflexion"></a>

## 🗣️ 📈 ⚖️ 📄 Réflexion

### Ex 11.9 — Déduction et induction dans un projet de ML, en cinq lignes

<details><summary>Indice 1</summary>

Relis la fin de la section 11.6 de la fiche : l'entraînement et l'utilisation d'un modèle.

</details>
<details><summary>Indice 2</summary>

Une ligne pour l'entraînement (induction), une pour l'application du modèle (déduction à partir d'une règle apprise), une pour ce qui garantit (rien, sauf si les données sont représentatives), une pour ce qu'on fait pour s'en assurer (un test), une pour conclure.

</details>
<details><summary>Indice 3</summary>

« Un modèle apprend une règle à partir d'exemples : c'est une induction, probable mais jamais certaine. Ensuite, il applique cette règle à chaque nouveau cas : c'est une déduction, aussi solide que la règle. La règle ne vaut que si les exemples sont représentatifs des cas réels. On le vérifie en testant le modèle sur des données qu'il n'a pas vues, pour estimer s'il sait généraliser. Il ne raisonne donc pas comme un juriste : il généralise. »

</details>

### Ex 11.10 — Lire les courbes d'un bandit : ε = 0, 0,01 et 0,1

<details><summary>Indice 1</summary>

Lis d'abord les fins de courbes (au pas 1 000), en haut puis en bas. Pour d) et e), relis l'encadré 🧮 sur ε-greedy.

</details>
<details><summary>Indice 2</summary>

Le plafond de ε-greedy est $1 - \varepsilon + \varepsilon/K$ avec $K = 10$. À très long terme, chaque agent atteint son plafond : lequel est le plus haut ? Pour h), lis sur la courbe du haut la récompense du glouton au pas 1 000 et la valeur de la ligne en tirets, puis fais le rapport.

</details>
<details><summary>Indice 3</summary>

a) 0,1. b) A (environ un tiers). c) C (environ 80 %). d) 0,91. e) 0,991. f) 0,01. g) Il se fixe sur le premier bras dont l'estimation devient la plus grande et ne vérifie jamais les autres. h) B.

</details>

### Ex 11.11 — Explorer sur des humains : essais adaptatifs, recommandation, A/B tests

<details><summary>Indice 1</summary>

Un bras est une option qu'on propose (un traitement, un prix, une vidéo) ; la récompense, ce qu'on observe ensuite. Explorer, c'est proposer une option peut-être moins bonne.

</details>
<details><summary>Indice 2</summary>

Pour 2) : un patient de plus reçoit-il le meilleur traitement ? Mais un seul patient dans un bras permet-il de conclure ? Pour 3) : l'équité entre clients, la transparence, la loi. Pour 4) : ce que la récompense mesure vraiment, et ce qu'elle ignore.

</details>
<details><summary>Indice 3</summary>

Pistes : l'exploration a un coût humain réel ; une allocation adaptative profite aux patients de l'essai, mais affaiblit la preuve (petits effectifs, dérive dans le temps, biais) ; un test A/B à proportions fixes donne une estimation plus nette ; une récompense comme le temps de visionnage pousse vers les contenus addictifs ou extrêmes. Règles possibles : un comité qui valide, une exploration bornée, des garde-fous (un plancher par bras, un arrêt pour nocivité), une récompense qui mesure le bien-être.

</details>

### Ex 11.12 — Domingos (2012) : représentation, évaluation, optimisation et autres leçons

<details><summary>Indice 1</summary>

Lis d'abord le tableau 1 de l'article, puis les sections citées dans l'énoncé.

</details>
<details><summary>Indice 2</summary>

Pour 3), cherche la phrase où Domingos cite Wolpert. Pour 4), cherche ce qu'il dit des bornes théoriques et de leur usage. Pour 6), compare les leçons sur l'ingénierie des features et sur la quantité de données à ce que fait l'apprentissage profond.

</details>
<details><summary>Indice 3</summary>

1 : perceptron (hyperplan, erreurs, règle de correction), moindres carrés (hyperplan, erreur quadratique, solution exacte ou descente de gradient), k-means (instances ou centroïdes, inertie, recherche gloutonne). 3 : aucun apprenant ne bat le hasard sur toutes les fonctions possibles ; il faut des hypothèses au-delà des données (le problème de Hume). 4 : une borne dit ce qui est garanti dans le pire cas, pas ce qui marche le mieux. 5 : représentable n'implique pas apprenable. 6 : « feature engineering is the key » a vieilli ; « it's generalization that counts » et « more data beats a cleverer algorithm » restent actuels.

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

Page de paiement : un changement durable, qu'on veut mesurer précisément (et parfois justifier) : test A/B, éventuellement séquentiel. Bandit pour des choix nombreux, courts ou renouvelés (titres d'articles, promotions d'une semaine), où le coût de l'exploration compte plus que la précision de l'estimation.

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

## Notebook

### 11.13 — Glouton pur sur trois bras : que va-t-il se passer ?

<details><summary>Indice 1</summary>

Imagine les premiers pas d'une partie : les trois estimations valent 0. Que se passe-t-il quand un bras rapporte 1 ?

</details>
<details><summary>Indice 2</summary>

Déroule une partie à la main. Que vaut l'estimation d'un bras après un échec ? Après un succès ? Peut-elle redescendre à 0 ensuite ? Compare alors les trois estimations, et applique la règle du glouton.

</details>
<details><summary>Indice 3</summary>

Le premier bras qui rapporte 1 est joué pour toujours. Le premier succès tombe sur le bras $a$ avec une probabilité proportionnelle à $p_a$ : $0{,}7 / 1{,}4 = 1/2$ pour le meilleur. La récompense finale moyenne vaut $\sum_a p_a^2 / \sum_a p_a$.

</details>

### 11.14 — Holmes déduit-il ? Compter et citer le vocabulaire du raisonnement

<details><summary>Indice 1</summary>

`re.findall` avec un motif qui commence par `\b` (début de mot), le radical, puis `\w*` (la fin du mot), et l'option `re.IGNORECASE`.

</details>
<details><summary>Indice 2</summary>

`collections.Counter(mot.lower() for mot in re.findall(rf"\b{stem}\w*", text, flags=re.IGNORECASE))`. Pour Verne, vérifie que ton motif garde les lettres accentuées à l'intérieur des mots (en Python, `\w` les accepte ; `[a-z]` non). Pour `story_of_14`, construis le motif `r"\s+".join(re.escape(partie) for partie in phrase.split())` et cherche-le dans le texte de chaque nouvelle.

</details>
<details><summary>Indice 3</summary>

Le faux positif de « infer » est un adjectif qui parle d'enfer. La fréquence pour 10 000 mots : `sum(compte.values()) / len(words(holmes)) * 10_000`. Parcours `stories.items()`, et renvoie le numéro de la première nouvelle où `re.search(motif, texte, flags=re.IGNORECASE)` trouve quelque chose.

</details>

### 11.15 — Valider un syllogisme par force brute : 256 mondes de Venn

<details><summary>Indice 1</summary>

Un monde garde ou non chacune des 8 régions : `itertools.product([False, True], repeat=8)` donne les 256 masques ; garde les régions dont le masque vaut `True`.

</details>
<details><summary>Indice 2</summary>

Avec `i, j = TERMS_15[x], TERMS_15[y]` : « quelque $x$ est $y$ » = `any(r[i] and r[j] for r in world)` ; « quelque $x$ n'est pas $y$ » = `any(r[i] and not r[j] for r in world)` ; A est la négation de O, E celle de I. Pour `valid_forms_15`, trois boucles : la figure, puis les trois formes (`itertools.product("AEIO", repeat=3)`).

</details>
<details><summary>Indice 3</summary>

`counterexamples_15` : `[w for w in worlds_15() if all(holds_15(p, w) for p in premises) and not holds_15(conclusion, w)]`. Codage : S2 = `([("I", "M", "P"), ("A", "S", "M")], ("I", "S", "P"))`, S3 = `([("E", "M", "P"), ("I", "S", "M")], ("O", "S", "P"))`, S6 = `([("O", "M", "P"), ("A", "S", "M")], ("O", "S", "P"))`, S7 = `([("E", "P", "M"), ("E", "S", "M")], ("E", "S", "P"))`. Avec `existence=True`, ajoute `("I", t, t)` pour `t` dans `"SMP"`.

</details>

### 11.16 — Reproduire la figure 11.5 : les cinq sophismes en diagrammes

<details><summary>Indice 1</summary>

Commence par le cadre commun aux cinq diagrammes : le grand cadre « animaux et meubles », la boîte « à quatre pattes », l'ellipse « chats » dans la boîte.

</details>
<details><summary>Indice 2</summary>

Pour chaque sophisme, demande-toi où doit être le contre-exemple : dans la boîte « à quatre pattes » mais hors des chats (un chien, une table), ou dans les mammifères mais hors de la boîte (la baleine). Ajoute l'ellipse dont le sophisme parle (chiens, mammifères, tables).

</details>
<details><summary>Indice 3</summary>

Les deux premiers : le point violet dans la boîte, hors de l'ellipse des chats. Majeur illicite : une ellipse « chiens » dans la boîte, à côté des chats, le point dedans. Moyen terme non distribué : pareil avec « tables ». Mineur illicite : une boîte « à quatre pattes » plus étroite, une grande ellipse « mammifères » qui contient les chats et déborde de la boîte, le point dans la partie qui déborde.

</details>

### 11.17 — Généralisation hâtive et échantillon biaisé chez les manchots

<details><summary>Indice 1</summary>

`is_gentoo.mean()` donne une proportion. Pour les échantillons, tire des **indices** avec remise, puis fais la moyenne de `is_gentoo` sur ces indices.

</details>
<details><summary>Indice 2</summary>

`rng.integers(len(is_gentoo), size=(n_samples, n))` donne tous les indices d'un coup ; `is_gentoo[indices].mean(axis=1)`, les parts. Pour Biscoe, un masque `island == "Biscoe"`. Pour la capture proportionnelle à la masse, l'espérance vaut $\sum_i m_i \, P(m_i)$ avec $P(m_i) = m_i / \sum_j m_j$.

</details>
<details><summary>Indice 3</summary>

`np.sum(m * m / m.sum())`, soit $\sum m^2 / \sum m$. Pour f), demande-toi si dix fois plus de manchots capturés de la même façon changeraient la loi des manchots capturés.

</details>

### 11.18 — Des points sur un cercle : quand le modèle trahit l'induction

<details><summary>Indice 1</summary>

L'équation $x^2 + y^2 + Dx + Ey + F = 0$ est linéaire en $D$, $E$, $F$ : c'est une régression linéaire à trois inconnues.

</details>
<details><summary>Indice 2</summary>

`A = np.column_stack([x, y, np.ones_like(x)])` et le second membre `-(x**2 + y**2)` ; `np.linalg.lstsq(A, b, rcond=None)[0]` donne $(D, E, F)$.

</details>
<details><summary>Indice 3</summary>

Centre $(-D/2, -E/2)$ ; rayon $\sqrt{c_x^2 + c_y^2 - F}$, car $(x - c_x)^2 + (y - c_y)^2 = r^2$ se développe en $x^2 + y^2 - 2c_x x - 2c_y y + c_x^2 + c_y^2 - r^2 = 0$.

</details>

### 11.19 — BernoulliBandit et GaussianBandit

<details><summary>Indice 1</summary>

Les constructeurs sont fournis : il ne reste que `pull`. Utilise le générateur `self._rng`, jamais `np.random`.

</details>
<details><summary>Indice 2</summary>

Vérifie d'abord `0 <= arm < self.n_arms`, sinon `raise IndexError(...)`. Bernoulli : `self._rng.random() < self.means[arm]` vaut `True` avec la probabilité voulue. Gaussien : `self._rng.normal(moyenne, écart-type)`.

</details>
<details><summary>Indice 3</summary>

`return float(self._rng.random() < self.means[arm])` et `return float(self._rng.normal(self.means[arm], self.std))`. Le `float(...)` transforme le booléen ou le flottant NumPy en `float` Python.

</details>

### 11.20 — argmax_random_tie, epsilon_greedy_action et incremental_update

<details><summary>Indice 1</summary>

Trois fonctions courtes. Commence par les vérifications (`ValueError`), puis le cas `rng is None`.

</details>
<details><summary>Indice 2</summary>

Ex aequo : `best = np.flatnonzero(values == values.max())`. ε-greedy : `if rng.random() < epsilon:` un bras `rng.integers(n_bras)`, sinon `argmax_random_tie`. Un `nan` se détecte avec `np.isnan(values).any()`.

</details>
<details><summary>Indice 3</summary>

`return int(best[0]) if best.size == 1 else int(rng.choice(best))` ; `return estimate + step_size * (target - estimate)` après avoir vérifié `0 < step_size <= 1`.

</details>

### 11.21 — run_bandit : la boucle d'interaction et ses courbes

<details><summary>Indice 1</summary>

Prépare `q_values` (rempli de `initial_value`), `counts` (des zéros entiers) et des tableaux pour les bras joués et les récompenses ; puis une boucle `for t in range(1, n_steps + 1)`.

</details>
<details><summary>Indice 2</summary>

Dans la boucle : choisir, vérifier le bras, tirer, compter, mettre à jour. Après la boucle : `optimal = means[actions] == bandit.best_mean`, `regret = np.cumsum(bandit.best_mean - means[actions])`. Pour `testbed_21`, une boucle sur les parties qui additionne `history["rewards"]` et `history["optimal"]`.

</details>
<details><summary>Indice 3</summary>

`step = 1.0 / counts[arm] if step_size is None else step_size` puis `q_values[arm] = incremental_update(q_values[arm], reward, step)`. Dans `testbed_21` : `GaussianBandit(rng.normal(0, 1, size=10), std=1.0, random_state=int(rng.integers(2**32)))` et `run_bandit(bandit, policy, n_steps, rng=np.random.default_rng(int(rng.integers(2**32))))`.

</details>

### 11.22 — Initialisation optimiste sans ε : prédire, puis mesurer

<details><summary>Indice 1</summary>

Suis les estimations de l'agent optimiste pendant les premiers pas : toutes valent 5, et chaque tirage fait baisser celle du bras tiré.

</details>
<details><summary>Indice 2</summary>

Avec le pas 0,1, que devient l'estimation d'un bras après son premier tirage, comparée à celle d'un bras jamais tiré ? Déroule les premiers pas d'une partie à la main, avec 10 bras. Avec la moyenne exacte, que reste-t-il de la valeur 5 après un tirage (∂ 11.2, question 5) ?

</details>
<details><summary>Indice 3</summary>

Les 10 premiers pas essaient chaque bras une fois ; au pas 11, l'agent prend le bras de meilleure première récompense, souvent le meilleur. Avec la moyenne exacte, l'agent devient glouton après une seule récompense par bras : bien mieux que le glouton de 📈 11.10, moins bien que l'agent à pas constant, qui continue d'explorer un moment.

</details>

### 11.23 — ucb_action et thompson_action

<details><summary>Indice 1</summary>

UCB : un cas particulier (un bras jamais tiré) avant la formule. Thompson : un tirage par bras, puis un argmax.

</details>
<details><summary>Indice 2</summary>

`untried = np.flatnonzero(counts == 0)` ; s'il n'est pas vide, renvoie `int(untried[0])`. Sinon, `np.argmax(q + c * np.sqrt(np.log(t) / counts))`. Thompson : `rng.beta(1 + successes, 1 + failures)` accepte des tableaux.

</details>
<details><summary>Indice 3</summary>

Vérifie les formes (`np.shape`), `t >= 1`, `c >= 0` et l'absence de compteurs négatifs, puis `return int(np.argmax(rng.beta(1.0 + successes, 1.0 + failures)))`. Pour retrouver les succès dans `run_bandit` : `np.rint(q * n)`.

</details>

### 11.24 — Bandit piégé : l'agent qui n'explore jamais

<details><summary>Indice 1</summary>

Appelle plusieurs fois `select` sur l'agent du collègue sans rien mettre à jour : le hasard change-t-il d'un appel à l'autre ?

</details>
<details><summary>Indice 2</summary>

Lis chaque ligne en te demandant « que vaut-elle pour le dernier bras ? », « que donne `np.argmax` sur des égalités ? », « par quoi divise-t-on : le nombre total de pas ou le nombre de tirages du bras ? ».

</details>
<details><summary>Indice 3</summary>

Crée `self.rng = np.random.default_rng(seed)` dans un `__init__` qui appelle `super().__init__(...)` ; explore avec `self.rng.integers(self.n_arms)` ; tire au sort parmi `np.flatnonzero(self.q == self.q.max())` ; incrémente `self.n[arm]`, puis divise par `self.n[arm]`.

</details>

### 11.25 — Un journal d'expériences reproductible (JSON)

<details><summary>Indice 1</summary>

Un seul générateur, créé avec `config["seed"]`, d'où l'on tire tout le reste, toujours dans le même ordre.

</details>
<details><summary>Indice 2</summary>

`json.dumps(record, indent=2)` ; les versions avec `platform.python_version()` et `np.__version__` ; la date avec `datetime.datetime.now(datetime.timezone.utc).isoformat()` ; le commit avec `subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)`.

</details>
<details><summary>Indice 3</summary>

Convertis chaque résultat avec `float(...)`. `rerun_25` : `record = json.loads(Path(path).read_text())`, puis `return run_experiment_25(record["config"]) == record["results"]`.

</details>

### 11.26 — Tournoi : ε-greedy, optimiste, UCB et Thompson

<details><summary>Indice 1</summary>

Une politique est une fonction `(q, n, t, g) -> bras` ; les réglages de la boucle (valeur initiale, pas) vont dans les arguments de `run_bandit`.

</details>
<details><summary>Indice 2</summary>

`"optimiste": (lambda q, n, t, g: mylearn.bandit.argmax_random_tie(q, g), {"initial_value": 1.0, "step_size": 0.1})`. Dans `tournament_26`, pour chaque agent : `for i, bandit in enumerate(make_bench(n_bandits, seed))`, puis `run_bandit(..., rng=np.random.default_rng(i), **kwargs)["regret"][-1]`.

</details>
<details><summary>Indice 3</summary>

L'erreur type : `np.std(finals, ddof=1) / np.sqrt(len(finals))`. Lis ensuite le classement affiché pour remplir `best_26` et `ucb1_beats_eps_26`.

</details>

### 11.27 — Défi : battre UCB1 sur un banc de bandits de Bernoulli

<details><summary>Indice 1</summary>

Un agent au hasard perd, à chaque pas, l'écart entre le meilleur bras et la moyenne de tous les bras. Pour la stratégie : le tournoi de 11.26 a-t-il un vainqueur qui suffirait déjà ?

</details>
<details><summary>Indice 2</summary>

`random_regret_27` : la moyenne sur les bandits de `n_steps * (bandit.best_mean - np.mean(bandit.means))`. Les moyennes du banc sont uniformes : quel prior Beta leur correspond exactement ?

</details>
<details><summary>Indice 3</summary>

Thompson avec le prior $\mathrm{Beta}(1, 1)$ atteint l'objectif. Pour le palier 🌟, essaie un UCB moins prudent ($c$ autour de 0,3, réglé sur le banc public seulement) ou un UCB bayésien (le quantile $1 - 1/t$ du posterior, `scipy.stats.beta.ppf`).

</details>
