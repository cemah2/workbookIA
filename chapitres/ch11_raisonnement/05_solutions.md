# 11 · Apprentissage et raisonnement — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses courtes des quiz, des rappels et des exercices ✏️ 11.1, 11.3, 11.5 à 11.8 et 📈 11.10 se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 📈 ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 11.Q1 — Représentation, évaluation, optimisation : associer
a) **OEROER** : la descente de gradient et l'algorithme de Lloyd **cherchent** (O) ; l'erreur quadratique sur la validation et la vraisemblance **jugent** (E) ; les poids et la règle du perceptron, comme un arbre de profondeur 3, disent ce que le modèle **peut exprimer** (R). b) **A** : le test « mal classé ? » ne fait que juger l'exemple ; la mise à jour et le pas cherchent, l'hyperplan est la représentation.
**Erreurs fréquentes** : classer la vraisemblance en O parce qu'on « maximise la vraisemblance ». Ce qu'on maximise est un critère (E) ; la méthode qui le maximise est l'optimisation.

### 11.Q2 — Puissance de représentation : ce qu'un perceptron ne peut pas « savoir »
a) **ACE** : les trois frontières sont des droites ($x_1 + x_2 = 3$, $x_1 = 2$, $2x_1 - x_2 = 0{,}5$). Le disque a une frontière courbe ; « même signe » regroupe deux quarts de plan opposés, la situation de XOR. b) **B** : $x_1^2 + x_2^2 < 1$ s'écrit $-x_1^2 - x_2^2 + 1 > 0$, une somme pondérée des nouvelles entrées. « Même signe » s'écrit $x_1 x_2 > 0$ : il faudrait la feature $x_1 x_2$ (c'est l'astuce de ∂ 10.4). c) **Faux** : plus de puissance, c'est aussi plus de risque de surapprendre (ch. 9).
**À retenir** : on augmente la puissance d'un modèle linéaire en lui donnant des **features** non linéaires ; c'est le principe des noyaux (ch. 13) et, en un sens, des couches cachées.

### 11.Q3 — Représentable mais pas apprenable : le problème de l'arrêt
a) **Vrai** : un programme précis sur une entrée précise s'arrête ou non. b) **Vrai** : c'est le théorème de Turing (1936). c) **Faux** : `while True: pass` ne s'arrête jamais, et c'est évident ; beaucoup de programmes se prouvent. d) **Faux** : il pourrait s'arrêter juste après. e) **B** : la fonction se représente (un oui ou un non par couple), mais aucun algorithme ne la calcule partout ; aucun système ne l'apprendra donc exactement.
**Erreurs fréquentes** : c) Vrai, en suivant le livre, qui place l'impossibilité au niveau d'un programme précis (encadré ⚠️ de la fiche).

### 11.Q4 — Loss, métrique, objectif : qui sert à quoi ?
a) **LMOM** : la cross-entropy est la loss ; le recall sur le test et la precision sur la validation sont des métriques ; diviser les fraudes par deux est l'objectif. b) **A** : une petite modification des poids ne change aucune prédiction, donc ne change pas l'accuracy ; sa dérivée est nulle presque partout, et la descente de gradient n'a aucune direction. On minimise une loss dérivable qui la remplace. c) **B** : la precision, parmi les alertes, la part de vraies fraudes. d) **C** : le recall, parmi les vraies fraudes, la part détectée.
**Erreurs fréquentes** : c) A, en suivant le livre, qui attribue ce souhait à l'accuracy (encadré ⚠️ de la fiche). Avec une fraude rare, un modèle qui ne détecte rien a une accuracy de 99 %.

### 11.Q5 — Optimiser n'est pas être optimal ; pas de repas gratuit
a) **Faux** : améliorer à chaque pas mène souvent à un minimum local (ch. 5). b) **Faux** : la moyenne porte sur tous les problèmes possibles, dont l'immense majorité n'a aucune structure ; sur un problème réel, un algorithme adapté fait bien mieux que le hasard. c) **B**.
**À retenir** : No Free Lunch ne dit pas « tout se vaut », il dit « rien ne gagne partout » ; c'est un argument pour la validation, pas contre le choix.

### 11.Q6 — Déduction ou induction ? Six situations
a) **IDIIDD** : entraîner un filtre (1), conclure qu'une pièce est truquée (3) et prolonger une tendance (4) donnent des conclusions probables ; les situations 2, 5 et 6 tirent une conclusion nécessaire de leurs prémisses.
**Erreurs fréquentes** : 3 classé D, parce que 140 faces sur 200 « prouvent » le trucage. Une pièce équilibrée peut, très rarement, le faire : la conclusion reste probable (le ch. 4 en chiffre la probabilité).

### 11.Q7 — Valide, solide, ou ni l'un ni l'autre ?
a) **SVNVNN** : 1 est valide (Barbara) et ses prémisses sont vraies. 2 est valide, mais « tous les oiseaux volent » est faux. 3 a un moyen terme (« mammifères ») jamais distribué. 4 est valide (Celarent), mais « aucun nombre pair n'est premier » est faux (2). 5 a un moyen terme (« rectangles ») jamais distribué : prédicat d'une A, sujet d'une O. 6 affirme le conséquent : l'erreur de validation dépasse presque toujours un peu l'erreur d'entraînement, même sans overfitting. b) **Vrai** : « aucun chat n'est un oiseau ; aucun oiseau n'est un chien ; donc aucun chat n'est un chien » (fiche §11.4.1).

### 11.Q8 — Nommer le sophisme syllogistique
a) **ACBED** : 1 part du conséquent (la page est lente) ; 3 nie l'antécédent (pas de standardisation) ; 2 distribue « mammifère » dans la conclusion négative sans qu'il le soit dans la majeure, où il est prédicat d'une A (majeur illicite) ; 5 distribue « rectangles », sujet de la conclusion, alors que la mineure en fait le prédicat d'une A (mineur illicite) ; 4 relie baleines et requins par un terme jamais distribué.
**Erreurs fréquentes** : confondre majeur et mineur illicite. Le **majeur** est le prédicat de la conclusion, le **mineur** son sujet.

### 11.Q9 — Généralisation, syllogisme statistique, prédiction
a) **SGPG** : 1 va de la population (toute la banque) à un individu tiré au hasard ; 2 et 4 vont de l'échantillon à la population ; 3, de l'échantillon au prochain tirage. b) **A**. c) **Faux** : la population de 2026 n'est plus celle qu'on a échantillonnée ; l'accuracy de test ne dit plus rien de fiable (dérive des données, fiche §11.5.1).

### 11.Q10 — Sophismes inductifs chez les data scientists
a) **EBFADC** : 1, une anecdote frappante contre les statistiques (vivacité trompeuse) ; 2, ceux qui laissent un avis ne sont pas représentatifs (échantillon biaisé, par auto-sélection) ; 3, une exception qu'on s'accorde (plaidoyer spécial) ; 4, trois exemples (généralisation hâtive) ; 5, une règle qui exclut 70 % du trafic (exception écrasante) ; 6, un écart net et persistant attribué au hasard (induction paresseuse).
**Erreurs fréquentes** : 2 classé A. Ce n'est pas le nombre d'avis qui pose problème (il peut y en avoir des milliers), c'est **qui** les écrit : les clients très contents ou très mécontents.

### 11.Q11 — Prémisses rationnelles, empiriques, et la fourche de Hume
a) **RERER** : la somme des angles se démontre (géométrie euclidienne) ; la transitivité de l'implication est une loi de logique ; « célibataire » veut dire « non marié ». Le point d'ébullition et la masse des manchots se mesurent. b) **B**. c) **Vrai** : une déduction transmet la vérité de ses prémisses sans l'augmenter.

### 11.Q12 — Holmes déduit-il vraiment ?
a) **B** : des entailles sur une chaussure peuvent avoir d'autres causes ; Holmes choisit la meilleure explication. b) **A** : avec une liste de possibles complète, éliminer tous les possibles sauf un force la conclusion (un syllogisme disjonctif répété) ; la faiblesse est dans la prémisse « la liste est complète ». c) **A**. d) **Faux** : deviner une pensée d'après un regard et des expressions est une abduction, plausible et rien de plus.

<a id="rappels"></a>

## 🔁 Rappels

### 11.R1 — Ch. 10 : la règle du perceptron et le cas XOR
a) **Faux** : $z = 1 \times 2 + (-1) \times 1 + 0 = 1$, $y\,z = 1 > 0$, rien à corriger. b) **[2, 1]** : $z = 1 - 2 = -1$, $y\,z = -1 \le 0$, donc $\mathbf{w} \leftarrow (1, -1) + 1 \times (1, 2) = (2, 1)$. c) **1** : $b \leftarrow 0 + 1$. d) **Faux** : XOR n'est pas linéairement séparable ; la règle corrige sans fin.

### 11.R2 — Ch. 8 : représentativité du jeu d'entraînement et fuite de données
a) **B** : le test estime l'erreur sur la population dont il est tiré, celle de 2015 à 2019 ; en 2026, les prix ont dérivé. On ne sait même pas dans quel sens l'erreur bougera (C est faux), et un test plus grand ne réduit que le hasard (D). b) **Vrai** : la moyenne et l'écart-type de tout le jeu contiennent de l'information du test ; il faut les calculer sur l'entraînement seulement. c) **60** : 200 patients dans le test, dont 30 %. d) **C** : la date de clôture du dossier n'existe qu'après l'enquête, donc après le moment de la prédiction.

### 11.R3 — Ch. 4 : mettre à jour sa croyance sur une pièce avec Bayes
a) **[8, 4]**. b) **0,67** ($8/12$). c) **0,70** ($7/10$). d) **0,57** ($8/14$, avec $\mathrm{Beta}(8, 6)$). e) **Vrai** : $\frac{h+1}{n+2} - \frac{1}{2} = \frac{2h - n}{2(n+2)}$ et $\frac{h}{n} - \frac{1}{2} = \frac{2h - n}{2n}$ : même numérateur, dénominateur plus grand pour la première, qui est donc plus proche de 0. f) **B** : avec le prior uniforme, la probabilité du prochain face est la moyenne du posterior, $(h+1)/(n+2) = 8/12$ (la règle de succession de Laplace).
**À retenir** : c'est ce posterior que l'échantillonnage de Thompson tire au hasard pour chaque bras (fiche §11.7).

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 11.1 — Représentable sur n bits : compter, puis conclure ✏️
**Réponses** : a) **256** · b) **255** · c) **−128** · d) **10** · e) **256** · f) **0,406** · g) **0,0287** · h) **3,32** · i) **Vrai**.
**Démarche** : $2^8 = 256$ valeurs ; sans signe, de 0 à 255 ; en complément à deux, de $-2^7$ à $2^7 - 1$. De 0 à 1 000, il y a 1 001 valeurs, et $2^9 = 512 < 1\,001 \le 2^{10}$. Une fonction de 3 entrées binaires choisit une sortie pour chacune des $2^3 = 8$ combinaisons : $2^8 = 256$ fonctions. $104/256 = 0{,}406$ ; $1\,882/65\,536 \approx 0{,}0287$. Un chiffre décimal porte $\log_2 10 \approx 3{,}32$ bits (ch. 6) : $3{,}32$ millions de bits pour un million de chiffres, contre 8 millions à un octet par chiffre. i) $2^{n^2}/2^{2^n} = 2^{n^2 - 2^n} \to 0$, car $2^n$ croît bien plus vite que $n^2$ ; la part des fonctions à seuil s'effondre (déjà $2 \times 10^{-5}$ pour 5 entrées).
**Ce que dit h)** : la table des chiffres de π demande toujours plus de place ; le programme qui les calcule, quelques centaines d'octets. Une représentation peut être un **calcul** ; c'est aussi l'idée de la complexité de Kolmogorov (la longueur du plus court programme qui produit une suite).
**Erreurs fréquentes** : a) 255 (la plus grande valeur au lieu du nombre de valeurs) ; c) $-127$ ; e) 8 (le nombre de combinaisons d'entrées) ; f) $104/8$.

### Ex 11.2 — Moyenne incrémentale : Qₙ₊₁ = Qₙ + (Rₙ − Qₙ)/n ∂
1. $Q_{n+1} = \frac{1}{n}\sum_{i=1}^{n} R_i = \frac{1}{n}\Big(R_n + \sum_{i=1}^{n-1} R_i\Big) = \frac{1}{n}\big(R_n + (n-1)\,Q_n\big) = Q_n + \frac{1}{n}(R_n - Q_n)$ pour $n \ge 2$, puisque $Q_n$ est la moyenne des $n - 1$ premières récompenses. Pour $n = 1$ : $Q_2 = Q_1 + (R_1 - Q_1) = R_1$, quelle que soit $Q_1$.
2. On ne garde que deux nombres par bras, l'estimation et le compteur, au lieu de toutes les récompenses ; chaque mise à jour coûte un nombre fixe d'opérations, au lieu d'une somme de $n$ termes.
3. Au rang 1 : $Q_2 = (1-\alpha) Q_1 + \alpha R_1$. Si la formule est vraie au rang $n$, alors $Q_{n+2} = (1-\alpha) Q_{n+1} + \alpha R_{n+1} = (1-\alpha)^{n+1} Q_1 + \sum_{i=1}^{n} \alpha(1-\alpha)^{n+1-i} R_i + \alpha R_{n+1}$, qui est la formule au rang $n + 1$ (le dernier terme est celui de $i = n + 1$).
4. $\sum_{i=1}^{n} \alpha(1-\alpha)^{n-i} = \alpha \sum_{k=0}^{n-1} (1-\alpha)^k = \alpha \cdot \frac{1 - (1-\alpha)^n}{\alpha} = 1 - (1-\alpha)^n$ ; avec $(1-\alpha)^n$, la somme vaut 1 : $Q_{n+1}$ est une moyenne pondérée. Le poids $\alpha(1-\alpha)^{n-i}$ est le plus grand pour $i = n$ : la récompense la plus récente compte le plus, et chaque récompense perd un facteur $1 - \alpha$ à chaque nouveau tirage, d'où l'« oubli exponentiel ». Utile quand les bras changent avec le temps (problème **non stationnaire**).
5. Avec un pas constant, $Q_1$ pèse $(1-\alpha)^n$, qui ne s'annule jamais : avec $\alpha = 0{,}1$, encore 35 % après 10 tirages. Avec le pas $1/n$, son poids est nul dès le premier tirage.
**Erreurs fréquentes** : diviser par $n$ au lieu de $n - 1$ en exprimant la somme des premières récompenses ; oublier le terme $(1-\alpha)^n Q_1$.

### Ex 11.3 — Syllogismes : valides ? solides ? ✏️
**Réponses** : a) **CAB** · b) **SNSNVNN** · c) **ABAD** · d) **Faux** · e) **2**.
**Démarche** (forme majeure-mineure-conclusion, puis les règles) :

| | Formes | Diagnostic | Prémisses | Code |
|---|---|---|---|---|
| S1 | E, A ⇒ E | valide (Celarent) | vraies | S |
| S2 | I ($M$-$P$), A ($S$-$M$) ⇒ I | « confidentiels » : sujet d'une I, prédicat d'une A, jamais distribué | | N (A) |
| S3 | E, I ⇒ O | valide (Ferio) | vraies : la majeure ne parle que des modèles linéaires de $x_1$ et $x_2$ seuls | S |
| S4 | A, E (individu) ⇒ E | « impair », prédicat d'une conclusion négative, est distribué ; dans la majeure, prédicat d'une A, il ne l'est pas : majeur illicite | | N (B) |
| S5 | A, A (individu) ⇒ A | valide (Barbara) | « tous les premiers sont impairs » est faux (2) | V |
| S6 | O, A ⇒ O | « oiseaux » : sujet d'une O, prédicat d'une A, jamais distribué | | N (A) |
| S7 | E, E ⇒ E | deux prémisses négatives | | N (D) |

e) « Aucun modèle linéaire de $x_1$ et $x_2$ seuls ne calcule XOR » vide l'intersection de $M$ et de $P$, que le cercle $S$ coupe en deux régions. d) S6 a une conclusion vraie (les manchots), mais sa forme ne la garantit pas : « certains animaux ne sont pas des mammifères ; tous les chiens sont des animaux ; donc certains chiens ne sont pas des mammifères » a la même forme, des prémisses vraies et une conclusion fausse.
**Un contre-exemple** (S2) : un monde où les rapports d'audit sont des documents confidentiels qui ne sont jamais des brouillons, et où les brouillons confidentiels sont d'autres documents : prémisses vraies, conclusion fausse.
**Erreurs fréquentes** : S6 jugé valide parce que sa conclusion est vraie ; S3 jugé non valide parce que « certains perceptrons » paraît vague ; S3 jugé non solide en pensant à ∂ 10.4 e : avec la feature $x_1 x_2$, un modèle linéaire calcule bien XOR, mais ce n'est plus un modèle de $x_1$ et $x_2$ **seuls**, d'où la précision de la majeure (sans elle, la majeure serait fausse et S3 vaudrait V) ; S7 jugé valide parce que sa conclusion paraît plausible (alors qu'elle est fausse : un requin **est** un poisson). 🔨 11.15 vérifie tout cela par force brute.

### Ex 11.4 — Six raisonnements fautifs à diagnostiquer et à réfuter ✏️
1. **Déductif ; moyen terme non distribué** (ou affirmation du conséquent). Contre-exemple de même forme : « tous les chats sont des mammifères ; mon chien est un mammifère ; donc mon chien est un chat ». Sur le fond, un modèle simple et bien régularisé peut avoir une loss d'entraînement basse sans surapprendre : c'est l'écart avec la validation qui compte.
2. **Déductif ; prémisses exclusives** : deux prémisses négatives ne relient rien, et une conclusion affirmative ne peut pas sortir de prémisses négatives (règles 3 et 4 de la fiche). La conclusion n'est vraie que pour des arbres d'au moins deux niveaux : un boosting de souches (des arbres à une seule question) additionne des effets de variables prises une à une, et ne capture aucune interaction. Contre-exemple de même forme : « aucun poisson n'aboie ; le chat n'est pas un poisson ; donc le chat aboie ».
3. **Déductif, forme valide** (*modus tollens*), donc pas de contre-exemple de même forme : la faute est que **la majeure est fausse**. Un cas qui la réfute : une petite fuite (une feature qui n'apporte qu'un peu d'information sur la cible), ou une fuite sur un problème difficile, laisse un score « normal ». Le raisonnement est valide, pas solide. Mieux : chercher la fuite directement (dates, features illégitimes, prétraitement fait avant le découpage, ch. 8).
4. **Inductif ; échantillon biaisé** (et généralisation abusive) : Paris n'est pas représentatif des autres villes (revenus, logements, habitudes). Il faut valider sur des données de ces villes, ou stratifier la validation par ville.
5. **Inductif ; généralisation hâtive** : trois cas. Il faut le taux d'échec de **toutes** les mises en production, jour par jour, et des effectifs suffisants ; et chercher une cause commune (le vendredi, on déploie peut-être les changements les plus gros).
6. **Inductif ; sélection des données favorables** (ce que le livre appelle « exception écrasante », ou sophisme de l'exclusion) ; « plaidoyer spécial » se défend aussi (on s'accorde une exception, « des cas à part », sans justification) : retirer les cas difficiles gonfle le score. Il faut garder ces cas, mesurer la performance par condition (jour, nuit), et dire clairement les limites du modèle.
**À retenir** : un raisonnement peut être fautif par sa **forme** (sophisme) ou par ses **prémisses** (non solide) ; un raisonnement inductif l'est le plus souvent par ses **données**.

### Ex 11.5 — Enquête au phare : réduire le domaine du discours ✏️
**Réponses** : a) **AB** · b) **B** · c) **Faux** · d) **Vrai** · e) **B** · f) **A** · g) **63** · h) **3** · i) **1**.
**Démarche** : départ {A, B, C, D, E, F} ; l'indice 1 retire C et E (des badges enregistrés loin du phare, à 1 h 50 et 2 h 05) ; l'indice 2 retire D : « si X est entré, alors peinture sous ses semelles ; pas de peinture sous celles de Diego ; donc Diego n'est pas entré », c'est le *modus tollens* ; l'indice 3 retire F. Restent A et B : « au moins l'un des deux » est démontré, la complicité ne l'est pas (le livre fait cette erreur avec son cuisinier et son majordome). L'indice 4 retire A : il reste B. Toute l'élimination repose sur « personne d'autre n'est sur l'île » (le monde clos). Un coupable est une personne qui a abaissé la manette : chaque indice dit qui n'a pas pu le faire. Hypothèses : les groupes non vides de 6 personnes, $2^6 - 1 = 63$ ; puis de 2 personnes, $2^2 - 1 = 3$ ({A}, {B}, {A, B}) ; puis 1 ({B}).
**Prémisses empiriques** : les badges ne mentent pas et ne se prêtent pas ; la peinture marque toujours les semelles ; Diego n'a pas d'autre paire ; la caméra donne la bonne heure ; la cheville d'Anne l'empêche vraiment de descendre vite. Chacune est une induction, et peut être fausse.
**Lien avec un classifieur** : un modèle entraîné sur dix classes range toute image dans l'une des dix, même une image d'une onzième sorte : il fait l'hypothèse du monde clos, sans le dire. La détection d'exemples « hors distribution » sert à lever cette hypothèse.
**Erreurs fréquentes** : c) Vrai, en suivant le livre.

### Ex 11.6 — Syllogisme statistique et prédiction : 15 % de pommes mûres ✏️
**Réponses** : a) **0,15** · b) **0,02244** · c) **0,02250** · d) **0,556** · e) **0,15** · f) **0,0565** · g) **1 275** · h) **0,167** · i) **Vrai** · j) **Faux**.
**Démarche** : a) $300/2\,000$ (syllogisme statistique). b) $\frac{300}{2000} \times \frac{299}{1999} \approx 0{,}02244$ ; c) $0{,}15^2 = 0{,}02250$ : la différence est minime (elle n'apparaît qu'à la cinquième décimale), car retirer une pomme sur 2 000 change peu la proportion. d) $1 - 0{,}85^5 \approx 1 - 0{,}4437 = 0{,}556$. e) $6/40 = 0{,}15$ (généralisation). f) $\sqrt{0{,}15 \times 0{,}85 / 40} = \sqrt{0{,}0031875} \approx 0{,}0565$ : la vraie proportion est probablement entre 4 % et 26 %, un panier de 40 pommes est imprécis. g) $\sqrt{0{,}1275/n} \le 0{,}01 \iff n \ge 0{,}1275/0{,}0001 = 1\,275$ pommes, tirées avec remise. Sans remise, 1 275 pommes feraient 64 % du stock : la correction pour population finie, $n = n_0 / (1 + (n_0 - 1)/N)$, ramènerait le besoin à 779 pommes. h) $(6 + 1)/(40 + 2) = 1/6 \approx 0{,}167$ (prédiction, règle de succession). i) Vrai : le client puise là où les pommes mûres sont concentrées. j) Faux : 1 275 pommes prises sur le dessus donnent une estimation **précise** (erreur type de 0,01) d'une **mauvaise** quantité, la proportion de pommes mûres sur le dessus.
**Erreurs fréquentes** : d) $5 \times 0{,}15 = 0{,}75$ (on n'additionne pas des probabilités d'événements compatibles) ; f) oublier la racine ; g) oublier d'élever 0,01 au carré (13 pommes).

### Ex 11.7 — Renforcement ou punition, positif ou négatif : classer huit situations ✏️
**Réponses** : a) **ABDDCABC** · b) **C** · c) **Faux** · d) **A**.
**Démarche** : 1 et 6 ajoutent quelque chose d'agréable, le comportement augmente (A). 2 et 7 retirent quelque chose de désagréable (le bip, la migraine), le comportement augmente (B, le renforcement négatif). 3 et 4 retirent quelque chose d'agréable (de l'argent, le téléphone), le comportement diminue (D, la punition négative). 5 et 8 ajoutent quelque chose de désagréable (la brûlure, le jet d'eau), le comportement diminue (C, la punition positive). d) La récompense est ajoutée et le choix devient plus fréquent : renforcement positif.
**Le perceptron (b)** : **pour** la lecture du livre, on n'agit qu'après une erreur, on ajoute une correction, et l'on veut des erreurs plus rares. **Contre** : la correction n'est pas un stimulus que l'apprenant perçoit, c'est un changement de l'apprenant lui-même ; et elle augmente aussi la probabilité de la bonne réponse sur l'exemple corrigé, ce qui ressemble à un renforcement de la bonne réponse. L'analogie est lâche ; ce qui tient, c'est la loi de l'effet : les conséquences d'une action modifient sa fréquence.
**Erreurs fréquentes** : 3 classé C, en lisant « négatif » comme « désagréable ». Une amende est désagréable, mais elle **retire** de l'argent : négatif.

### Ex 11.8 — Un bandit à la main : ε-greedy, moyennes et regret ✏️
**Réponses** : a) **0** · b) **[0 ; 0 ; 0,667]** · c) **1** · d) **[0 ; 0,5 ; 0,8]** · e) **[1, 2, 5]** · f) **1,1** · g) **1,4** · h) **0,867** · i) **0,053** · j) **Vrai**.
**Démarche** :

| Pas | $u_t$ | Explore ? | Bras | Récompense | $Q(0)$ | $Q(1)$ | $Q(2)$ | Regret du pas |
|---|---|---|---|---|---|---|---|---|
| 1 | 0,65 | non (égalité, bras 0) | 0 | 0 | 0 | 0 | 0 | 0,5 |
| 2 | 0,12 | oui | 2 | 1 | 0 | 0 | 1 | 0 |
| 3 | 0,47 | non | 2 | 1 | 0 | 0 | 1 | 0 |
| 4 | 0,83 | non | 2 | 0 | 0 | 0 | 2/3 | 0 |
| 5 | 0,05 | oui | 1 | 1 | 0 | 1 | 2/3 | 0,3 |
| 6 | 0,71 | non | 1 | 0 | 0 | 0,5 | 2/3 | 0,3 |
| 7 | 0,38 | non | 2 | 1 | 0 | 0,5 | 0,75 | 0 |
| 8 | 0,91 | non | 2 | 1 | 0 | 0,5 | 0,8 | 0 |

f) $0{,}5 + 0{,}3 + 0{,}3 = 1{,}1$. g) $8 \times 0{,}8 - 5 = 1{,}4$ : le regret réalisé dépend aussi de la chance des tirages (au pas 4, le meilleur bras a donné 0) ; le pseudo-regret ne juge que les **décisions**. h) $1 - 0{,}2 + 0{,}2/3 \approx 0{,}867$. i) L'agent ne perd qu'en explorant : $\varepsilon \times \frac{0{,}5 + 0{,}3 + 0}{3} \approx 0{,}053$ par pas, soit environ 533 après 10 000 pas : le regret d'ε-greedy croît **linéairement**. j) Vrai : avec $\varepsilon = 0$, $Q(0) \ge 0 = Q(1) = Q(2)$ pour toujours (les récompenses sont positives ou nulles), et l'égalité va au bras 0.
**Erreurs fréquentes** : a) 1 (le bras au hasard, alors que l'agent n'explore pas) ; b) $[0, 0, 0{,}5]$ (la moyenne sur 4 pas au lieu des 3 tirages du bras 2) ; f) 1,4 (le regret réalisé).
**Variante** : refais la trace avec des ex aequo tirés au sort ; quels pas changent ?

<a id="reflexion"></a>

## 🗣️ 📈 ⚖️ 📄 Réflexion

### Ex 11.9 — Déduction et induction dans un projet de ML, en cinq lignes 🗣️
**Réponse modèle** : « Un modèle apprend une règle à partir d'exemples : c'est une **induction**, une règle probable, jamais certaine. Une fois entraîné, il l'applique à chaque nouveau cas : c'est une **déduction**, mais elle n'est pas plus sûre que la règle apprise. Cette règle ne vaut que si les exemples sont **représentatifs** des cas réels : mêmes types de clients, même époque, même collecte. On vérifie qu'il sait **généraliser** en le testant sur des données qu'il n'a jamais vues. Il ne raisonne donc pas comme un juriste qui applique la loi : il généralise ce qu'il a vu, et peut se tromper dès que le monde change. »
**Ce qui compte** : séparer l'apprentissage (inductif) de l'application (déductive), et dire où est le risque (les données).

### Ex 11.10 — Lire les courbes d'un bandit : ε = 0, 0,01 et 0,1 📈
**Réponses** : a) **0,1** · b) **A** · c) **C** · d) **0,91** · e) **0,991** · f) **0,01** · h) **B**.
**Démarche** : au pas 1 000, la courbe ε = 0,1 est la plus haute dans les deux graphiques (environ 1,36 de récompense et 80 % d'action optimale) ; ε = 0,01 monte encore (environ 58 %) ; le glouton plafonne vers 33 %. Plafonds : $1 - \varepsilon + \varepsilon/10$, soit 0,91 et 0,991 ; à très long terme, ε = 0,01 atteint le sien et dépasse ε = 0,1 (qui continue de perdre 9 % de ses tirages à explorer). h) Environ 1,0 pour le glouton contre 1,53 pour le meilleur bras : deux tiers.
**g)** Le glouton se fixe sur le premier bras dont l'estimation devient la plus grande, souvent un bras correct mais pas le meilleur. Les autres bras gardent une estimation fausse (0 s'ils n'ont jamais été tirés, ou une estimation basse tirée d'un seul mauvais résultat), et il ne les vérifie jamais.
**Erreurs fréquentes** : d) 0,9, en oubliant que l'exploration tombe aussi sur le meilleur bras.

### Ex 11.11 — Explorer sur des humains : essais adaptatifs, recommandation, A/B tests ⚖️
**Éléments de réponse** :
1. Bras : l'ECMO et le traitement classique ; récompense : la survie ; exploration : donner le traitement le moins prometteur pour apprendre ; exploitation : donner celui qui semble le meilleur. La règle « jouer le gagnant » ressemble à un bandit qui exploite très vite.
2. **Pour** : chaque patient de l'essai a plus de chances de recevoir le meilleur traitement ; c'est l'argument éthique des essais adaptatifs. **Contre** : un seul patient dans un bras ne prouve rien ; les effectifs déséquilibrés, la dérive dans le temps (les patients du début et de la fin diffèrent) et les biais de sélection affaiblissent la preuve, et la preuve faible a coûté de nouveaux essais (celui de 1996 a inclus 185 nouveau-nés). Les essais adaptatifs modernes encadrent l'adaptation (règles écrites à l'avance, période de randomisation fixe, analyses intermédiaires).
3. Des prix différents pour un même produit posent des questions d'équité et de transparence (un client paie plus cher qu'un autre au hasard d'un algorithme), parfois de droit (discrimination si le prix suit des caractéristiques personnelles). Un test A/B à proportions fixes, court, annoncé et suivi d'un prix unique, mesure l'effet plus proprement.
4. Maximiser le temps de visionnage favorise les contenus qui retiennent le plus : sensationnels, extrêmes, addictifs ; la récompense mesure l'attention, pas le bien-être. Autres récompenses : la satisfaction déclarée, le retour du lendemain, la diversité ; ou des contraintes (plafonds de temps, exclusion de contenus).
5. Exemples de règles : une validation par un comité indépendant (et le consentement quand il s'agit de personnes) ; une exploration bornée, avec un plancher de trafic par bras et un arrêt automatique en cas de nocivité ; une récompense choisie pour ce qu'on veut vraiment améliorer, publiée à l'avance avec le plan d'analyse.

### Ex 11.12 — Domingos (2012) : représentation, évaluation, optimisation et autres leçons 📄
**Éléments de réponse** :
1. Perceptron : représentation hyperplan, évaluation nombre d'erreurs (accuracy), optimisation par corrections successives. Moindres carrés : hyperplan, erreur quadratique, optimisation continue (solution exacte ou descente de gradient). k-means : des centroïdes (proches des « instances »), l'inertie (erreur quadratique), une recherche gloutonne. Le tableau 1 cite justement les hyperplans, l'erreur quadratique et la descente de gradient.
2. Parce que le but est de **généraliser** : un score mesuré sur les données d'entraînement est trompeur. Domingos insiste aussi sur la contamination involontaire du test par les réglages ; c'est la règle du test unique du ch. 8.
3. Il écrit que, d'après les théorèmes No Free Lunch de Wolpert, aucun apprenant ne bat le hasard en moyenne sur toutes les fonctions possibles ; il en conclut que tout apprenant doit incorporer des connaissances ou des hypothèses au-delà des données (des fonctions lisses, des exemples semblables qui ont des classes semblables…). C'est la réponse pratique au problème de Hume : on ne justifie pas l'induction, on choisit des hypothèses raisonnables pour le monde réel.
4. Les garanties théoriques sont des bornes, souvent très larges, sur le pire cas ou à l'infini : elles servent à comprendre et à concevoir, pas à prédire la performance d'un algorithme sur un problème donné. Dans 🔬 11.26, UCB1 et Thompson ont tous deux un regret garanti en $\ln T$ (pour Thompson, Agrawal et Goyal, 2012) ; pourtant UCB1 fait bien plus de regret qu'ε-greedy ou que l'agent optimiste, qui n'ont pas cette garantie, à l'horizon de 1 000 pas. La garantie d'UCB1 est simplement la plus facile à démontrer.
5. Qu'une fonction soit représentable par un modèle ne veut pas dire qu'on saura trouver les bons paramètres avec les données et le temps disponibles. Exemple du chapitre : la fonction de l'arrêt (représentable, jamais calculable partout) ; en pratique, un réseau peut représenter une fonction que la descente de gradient ne trouve pas.
6. Plus datée : « Feature engineering is the key », puisque l'apprentissage profond apprend des représentations à partir des données brutes (même si le soin des données reste décisif). Plus actuelles : « It's generalization that counts », « More data beats a cleverer algorithm » (les grands modèles en sont la démonstration) et « Correlation does not imply causation ».

<a id="entretien"></a>

## 💼 Entretien

### 11.E1 — Exploration contre exploitation : expliquer avec un exemple métier
**Réponse modèle en 60 secondes** : « Quand on doit choisir plusieurs fois entre des options dont on ne connaît pas la valeur, on fait face à un dilemme : exploiter la meilleure option connue, ou explorer une option moins connue qui pourrait être meilleure. Exemple : un site qui choisit, pour chaque visiteur, une bannière parmi cinq. S'il montre toujours celle qui a le meilleur taux de clic après les premiers jours, il peut se figer sur une bannière moyenne, simplement chanceuse au début. S'il les teste au hasard pour toujours, il perd des clics. Les algorithmes de bandits règlent ce compromis : ε-greedy explore une petite fraction du temps, UCB explore les options dont l'estimation est la plus incertaine, et l'échantillonnage de Thompson montre chaque bannière avec la probabilité qu'elle soit la meilleure. On les compare par leur regret, ce qu'on a perdu par rapport à toujours choisir la meilleure. »
**Relances possibles** : « Et si les préférences changent avec le temps ? » (un pas constant, une fenêtre glissante, ou une exploration qui ne s'éteint jamais) · « Et si l'on connaît le visiteur ? » (un bandit contextuel, comme LinUCB pour Yahoo! en 2010) · « Quel lien avec l'apprentissage par renforcement ? » (un bandit est un renforcement à un seul état ; le RL ajoute des états et des conséquences à long terme).

### 11.E2 — A/B test ou bandit : lequel choisir ?
**Réponse modèle en 60 secondes** : « Ça dépend de ce qu'on veut. Un test A/B, avec des proportions fixes et une taille d'échantillon calculée à l'avance, donne une estimation non biaisée de la différence entre les deux pages, avec un intervalle de confiance : c'est ce qu'il faut pour une décision durable, qu'on doit pouvoir justifier, comme une page de paiement. Un bandit déplace le trafic vers la meilleure variante pendant l'expérience : il perd moins de conversions, mais estime moins bien la variante délaissée, et supporte mal un effet qui change avec le temps ou qui n'apparaît qu'au bout de plusieurs jours. Je choisirais donc un test A/B pour la page de paiement, et un bandit pour des choix nombreux et éphémères, comme les titres d'articles ou les promotions de la semaine, où le coût de l'exploration compte plus que la précision de l'estimation. »
**Relances possibles** : « Peut-on arrêter un test A/B dès qu'il est significatif ? » (non sans méthode séquentielle : regarder souvent gonfle les faux positifs) · « Et avec dix variantes ? » (le bandit devient plus intéressant : un test A/B à dix bras coûte cher).

### 11.E3 — Que dit le théorème « No Free Lunch » pour le choix d'un modèle ?
**Réponse modèle en 60 secondes** : « Le théorème dit qu'en moyenne sur tous les problèmes possibles, aucun algorithme d'apprentissage ou d'optimisation n'est meilleur qu'un autre : ce qu'il gagne quelque part, il le perd ailleurs. Il ne dit pas que tous les modèles se valent sur mon problème : les problèmes réels ont une structure, et un modèle gagne quand ses hypothèses, son biais inductif, collent à cette structure. En pratique, j'en tire trois choses : je pars de choix éprouvés pour mon type de données (des arbres boostés pour du tabulaire, des réseaux pré-entraînés pour les images ou le texte), je garde toujours une baseline simple, et je compare quelques candidats en validation croisée, sans regarder le test avant la fin. Aucun benchmark publié ne remplace une comparaison sur mes données. »
**Relances possibles** : « Alors pourquoi tout le monde utilise AdamW ou XGBoost ? » (parce que les problèmes courants se ressemblent, et que ces méthodes ont fait leurs preuves sur eux) · « Et TabPFN ? » (un modèle pré-entraîné qui bat les arbres boostés sur de petits jeux tabulaires : la frontière bouge, d'où l'intérêt de comparer).

### 11.E4 — Un biais d'échantillonnage qui a fait échouer un modèle : exemple et parade
**Réponse modèle en 60 secondes** : « Exemple plausible : une banque entraîne un modèle de risque de crédit sur ses clients **acceptés** par le passé, les seuls dont on connaît le remboursement. Les clients refusés n'y sont pas : le modèle n'a jamais vu le profil des demandeurs les plus risqués, et il se trompe sur eux quand la banque élargit sa clientèle. C'est un biais de sélection, que plus de données de même provenance ne corrigent pas. Je l'aurais détecté en comparant la distribution des demandes en production à celle de l'entraînement, et en suivant la performance par segment de clientèle. Les parades : collecter des données sur la population réelle (par exemple accepter une petite part de demandes au hasard, comme une exploration), repondérer les exemples, et surveiller le modèle après son déploiement. »
**Relances possibles** : « Comment mesurer la dérive ? » (comparer les distributions des features, par un test ou un indice de stabilité, et la performance par période) · « Qu'est-ce que la repondération ? » (donner à chaque exemple un poids inverse de sa probabilité d'être dans l'échantillon, comme $1/m$ pour les manchots de 🔬 11.17).

<a id="notebook"></a>

## Notebook, parties A à E

Les réponses ci-dessous sont celles de `05_solutions.ipynb` (exécuté, FAST_MODE). La référence de `mylearn.bandit` est dans `solutions/mylearn_ref/` : lis-la **après** avoir réussi les tests. Des méthodes différentes des corrigés sont acceptées tant que les valeurs sont les mêmes.

### Ex 11.13 — Glouton pur sur trois bras : que va-t-il se passer ? 🔮
**Réponses** (l'expérience) : a) **"C"** (le meilleur bras est le plus joué dans 48 % des parties) · b) **False** (aucune partie ne change de bras après le pas 100) · c) **0,56** (0,553 mesuré).
**Démarche** : tant qu'aucun bras n'a rapporté 1, les trois estimations valent 0 et l'agent tire au sort ; le premier bras qui rapporte 1 prend une estimation positive, qu'il garde pour toujours, et les autres restent à 0. Le premier succès tombe sur le bras $a$ avec une probabilité proportionnelle à $p_a$ : $0{,}2/1{,}4 \approx 0{,}14$, $0{,}5/1{,}4 \approx 0{,}36$, $0{,}7/1{,}4 = 0{,}5$ (mesuré : 0,145, 0,371, 0,484). Récompense finale moyenne : $\sum p_a^2 / \sum p_a = 0{,}78/1{,}4 \approx 0{,}557$.
**Erreurs fréquentes de prédiction** : a) A, en croyant que le glouton « finit par trouver » le meilleur bras ; b) True, en croyant que les estimations finissent par se croiser.
**Variante** : refais l'expérience avec des estimations initiales à 1 au lieu de 0 : que change l'optimisme ? (Chaque bras est essayé jusqu'à son premier échec, et le meilleur gagne beaucoup plus souvent.)

### Ex 11.14 — Holmes déduit-il ? Compter et citer le vocabulaire du raisonnement 🔨
**Réponses** : a) **29** · b) **"infernal"** · c) **8** · d) **2,7398** · e) **2** · f) **3,3164** · g) **11** · h) **1**.
**Démarche** : `re.findall(rf"\b{stem}\w*", text, flags=re.IGNORECASE)`, puis un `Counter` des mots en minuscules. Holmes : 29 « deduc » (13 *deduce*), 57 « observ », 54 « reason », 18 « theor », seulement 2 « induc » (*induce*, sans rapport avec l'induction). Verne : aucun « déduction » ni « déduire » ; la famille « infér » ne contient que *inférieurs* et *inférieures*, deux faux positifs ; 24 « observ ». Les deux citations sont coupées par des retours à la ligne : d'où `\s+` entre les mots.
**Erreurs fréquentes** : un découpage en mots limité aux lettres a à z (`[a-z]+`), qui coupe « inférieurs » en deux et donne 0 en e) ; compter `"deduc"` comme sous-chaîne sans `\b` (sans conséquence ici, mais « sub-deduction » ou un mot composé ferait un faux positif) ; oublier `re.IGNORECASE` (un mot en début de phrase) ; chercher la phrase avec des espaces simples (aucune correspondance).
**Variante** : compte les familles nouvelle par nouvelle ; dans quelle nouvelle Holmes « déduit »-il le plus ?

### Ex 11.15 — Valider un syllogisme par force brute : 256 mondes de Venn 🔨
**Réponses** : a) **16** · b) **16** · c) **8** · d) **[16, 0, 16, 16]** · e) **15** · f) **24**.
**Démarche** : un monde est l'ensemble des régions non vides ; « tout $x$ est $y$ » est vrai s'il n'y a aucune région non vide dans $x$ et hors de $y$ (la négation de « quelque $x$ n'est pas $y$ ») ; E est la négation de I. Darapti (« tout $M$ est $P$ ; tout $M$ est $S$ ; donc quelque $S$ est $P$ ») a 8 contre-exemples, tous des mondes où $M$ est vide (parmi les 16 où $M$ est vide, ceux où, en plus, aucun $S$ n'est $P$), et devient valide quand on ajoute « $M$ a au moins un membre » : l'hypothèse d'Aristote. Les 15 formes valides en lecture moderne, et 24 avec l'hypothèse d'existence, sont les nombres classiques.
**Erreurs fréquentes** : coder A comme « quelque $x$ est $y$, et aucun $x$ n'est hors de $y$ », ce qui ajoute en douce l'hypothèse d'existence (A doit être vrai quand $x$ est vide) ; inverser $S$ et $P$ dans la conclusion ; oublier une prémisse en codant S2 ou S7.
**Variante** : ajoute les propositions sur un individu (« Socrate est un homme ») en exigeant qu'un terme ait exactement une région non vide ; S4 et S5 de ✏️ 11.3 donnent-ils les verdicts attendus ?

### Ex 11.16 — Reproduire la figure 11.5 : les cinq sophismes en diagrammes 🎨
**Réponse** : cinq diagrammes (voir le notebook de solutions), un point violet chacun.
**Démarche** : un cadre commun (« animaux et meubles », la boîte « à quatre pattes », les chats dans la boîte), puis, pour chaque sophisme, l'ensemble dont il parle et un point violet qui rend les prémisses vraies et la conclusion fausse : un chien à quatre pattes (affirmation du conséquent, négation de l'antécédent, majeur illicite), une baleine, mammifère sans pattes (mineur illicite), une table à quatre pattes (moyen terme non distribué).
**Erreurs fréquentes** : dessiner l'ellipse des chats en partie hors de la boîte (la prémisse « tout chat a quatre pattes » devient fausse) ; placer le point violet là où la conclusion est vraie.
**Variante** : dessine les prémisses exclusives : trois ellipses disjointes, puis montre que l'ellipse des chats peut entrer dans celle des mammifères sans violer les prémisses.

### Ex 11.17 — Généralisation hâtive et échantillon biaisé chez les manchots 🔬
**Réponses** : a) **0,3574** · b) **0,2143** · c) **0,0479** · d) **0,7301** · e) **4 360,7097** · f) **False**.
**Démarche** : 119 Gentoo sur 333. Erreur type $\sqrt{p(1-p)/n}$ : 0,214 pour 5, 0,048 pour 100 ; la dispersion des 4 000 estimations colle à ces valeurs (0,216 et 0,047). Avec 5 manchots, 66 % des échantillons se trompent de plus de 10 points ; avec 100, 3 %. À Biscoe, 119 Gentoo sur 163 : 0,730 ; à Torgersen et à Dream, aucun. Avec une capture proportionnelle à la masse, l'espérance vaut $\sum m^2/\sum m \approx 4\,361$ g, contre 4 207 g dans la population ; la simulation donne 4 360 g.
**Erreurs fréquentes** : tirer sans remise (la dispersion est alors plus petite que l'erreur type, d'un facteur $\sqrt{(N-n)/(N-1)}$, 0,84 pour 100 manchots sur 333) ; prendre la moyenne simple des masses en e).
**Variante** : corrige le biais de capture en pondérant chaque manchot capturé par $1/m$ : retrouves-tu 4 207 g ?

### Ex 11.18 — Des points sur un cercle : quand le modèle trahit l'induction 🔬
**Réponse** : erreurs du cercle ajusté : 0,943 (3 points proches), 0,008 (4 points répartis), 0,836 (30 points d'un petit arc), 0,009 (30 points tout autour).
**Démarche** : $x^2 + y^2 + Dx + Ey + F = 0$ est linéaire en $(D, E, F)$ : `np.linalg.lstsq` sur les colonnes $x$, $y$, 1 et le second membre $-(x^2 + y^2)$, puis centre $(-D/2, -E/2)$ et rayon $\sqrt{c_x^2 + c_y^2 - F}$. Trois points proches : un bruit minuscule suffit à tordre la courbure (généralisation hâtive). Quatre points répartis : le cercle est retrouvé, mais le polynôme de degré 3 dessine une courbe en S (généralisation abusive, avec une représentation inadaptée). Trente points d'un petit arc : erreur forte (échantillon biaisé). Les scénarios 1 et 3 ont le même défaut, un arc trop court : avec cet ajustement, 300 ou 3 000 points du même arc laissent l'erreur vers 0,86 pour l'arc du scénario 1 et vers 0,94 pour celui du scénario 3 (médianes sur 200 tirages). Plus de données réduisent le hasard ; elles ne remplacent pas les parties du cercle jamais observées.
**Erreurs fréquentes** : oublier le signe moins du second membre ; prendre la racine de $F$ au lieu de $c_x^2 + c_y^2 - F$.
**Variante** : avec 30 points du petit arc, double puis quadruple le nombre de points : l'erreur baisse-t-elle ? (Peu : c'est la couverture qui manque.)

### Ex 11.19 — BernoulliBandit et GaussianBandit 🔨
**Réponses** : `BernoulliBandit([0.2, 0.5, 0.9], random_state=0)` a 3 bras, le meilleur est le bras 2 (0,9) ; 4 000 tirages de chaque bras donnent des moyennes de 0,208, 0,494 et 0,900 ; le bras gaussien a une moyenne de 1,485 et un écart-type de 0,998 ; les 14 tests passent.
**Démarche** : vérifier le bras (`IndexError`), puis `float(self._rng.random() < self.means[arm])` ou `float(self._rng.normal(self.means[arm], self.std))`.
**Erreurs fréquentes** : utiliser `np.random.random()` (les tirages ne dépendent plus de la graine du bandit) ; `rng.random() > p` (la probabilité de succès devient $1 - p$) ; renvoyer un booléen ou un `np.float64` ; accepter `arm = -1`.
**Variante** : ajoute un `NonStationaryBandit` dont les moyennes font une petite marche aléatoire à chaque tirage ; c'est le cas où le pas constant de ∂ 11.2 devient utile.

### Ex 11.20 — argmax_random_tie, epsilon_greedy_action et incremental_update 🔨
**Réponses** : `argmax_random_tie([1, 3, 2])` vaut 1 ; les ex aequo 1, 2 et 4 sortent 962, 995 et 1 043 fois sur 3 000 ; ε-greedy (ε = 0,2, 4 bras) choisit le meilleur bras 8 473 fois sur 10 000 (85 % attendus) ; la moyenne courante de 4, 2, 6 vaut 4, 3, puis 4 ; les 27 tests passent.
**Démarche** : `np.flatnonzero(values == values.max())`, puis `rng.choice` s'il y a plusieurs maxima ; `rng.random() < epsilon` pour explorer, `rng.integers(n)` pour le bras ; `estimate + step_size * (target - estimate)`.
**Erreurs fréquentes** : explorer parmi les **autres** bras seulement (le meilleur est alors choisi avec la probabilité $1 - \varepsilon$, pas $1 - \varepsilon + \varepsilon/K$ : le test du χ² le voit) ; `np.argmax` sans tirage au sort ; oublier de rejeter `nan` (`values.max()` vaut alors `nan`, et aucun bras n'est égal au maximum).

### Ex 11.21 — run_bandit : la boucle d'interaction et ses courbes 🔨
**Réponses** : l'exemple de la docstring donne `counts = [4, 0]` et un regret `[0.5, 1., 1.5, 2.]` ; les 13 tests passent ; sur 200 parties, parts d'action optimale sur les 100 derniers pas : 30 % (ε = 0), 53 % (ε = 0,01), 78 % (ε = 0,1).
**Démarche** : la boucle de la docstring, puis `optimal = means[actions] == bandit.best_mean` et `regret = np.cumsum(bandit.best_mean - means[actions])`. `testbed_21` tire d'un seul générateur les moyennes des bras, la graine de chaque bandit et celle de chaque politique.
**Erreurs fréquentes** : passer à la politique les estimations **après** la mise à jour, ou des copies modifiées ; mettre à jour avec `1 / t` au lieu de `1 / counts[arm]` ; calculer le regret avec les récompenses ; oublier la `ValueError` pour un bras inexistant.
**Variante** : ajoute à `run_bandit` (dans une copie) un argument `callback` appelé à chaque pas, pour tracer l'évolution des estimations d'une partie.

### Ex 11.22 — Initialisation optimiste sans ε : prédire, puis mesurer 🔮
**Réponses** (l'expérience) : a) **True** (84 % contre 72 % sur les 100 derniers pas) · b) **11** (47 % d'action optimale au pas 11, 10 % aux pas 1 à 10) · c) **"B"** (69 %).
**Démarche** : avec $Q_1 = 5$, chaque tirage fait baisser l'estimation du bras tiré sous celle des bras jamais tirés : les 10 premiers pas essaient chaque bras une fois. Au pas 11, l'agent choisit le bras de meilleure première récompense, souvent le meilleur bras : le pic. Le tirage suivant le fait redescendre, et l'exploration reprend jusqu'à ce que les estimations aient oublié leur valeur initiale (le poids $(1 - \alpha)^n$ de ∂ 11.2). Avec la moyenne exacte, la valeur initiale disparaît au premier tirage : un tour d'exploration, puis un glouton sur des estimations d'un seul tirage.
**Erreurs fréquentes de prédiction** : a) False, en croyant qu'un agent sans ε ne peut pas explorer ; b) 10 ou 20 ; c) A, en croyant que l'optimisme ne sert à rien avec la moyenne exacte (il force quand même un tirage de chaque bras, ce qui vaut beaucoup mieux que le glouton à zéro).
**Variante** : essaie $Q_1 = 1$ et $Q_1 = 20$ : un optimisme trop faible n'explore pas assez, un optimisme trop fort explore trop longtemps.

### Ex 11.23 — ucb_action et thompson_action 🔨
**Réponses** : les exemples de la docstring donnent 1, 1 et 0 ; sur les trois bras de la fiche, Thompson choisit chaque bras avec les fréquences 0,545, 0,356 et 0,099 (la figure de la fiche : 0,54, 0,36, 0,10) ; les 17 tests passent. Sur le bandit difficile (0,45, 0,5, 0,55, 0,6), regret final après 2 000 pas : 100,7 pour ε-greedy, 83,8 pour UCB1, 26,5 pour Thompson.
**Démarche** : UCB joue d'abord un bras jamais tiré, puis `np.argmax(q + c * np.sqrt(np.log(t) / counts))` ; Thompson tire `rng.beta(1 + successes, 1 + failures)` et prend l'argmax. Avec des moyennes exactes, `np.rint(q * n)` redonne les succès.
**Erreurs fréquentes** : `np.log(t + 1)` ou `np.log2(t)` (les tests le voient sur des cas choisis) ; mettre $c$ sous la racine ; diviser par zéro pour un bras jamais tiré ; `rng.beta(successes, failures)` (paramètres nuls : erreur) ; un tirage Beta unique pour tous les bras.
**Variante** : écris Bayes-UCB, qui joue le bras dont le quantile $1 - 1/t$ du posterior est le plus grand (`scipy.stats.beta.ppf`) : c'est une piste pour le palier 🌟 de 11.27.

### Ex 11.24 — Bandit piégé : l'agent qui n'explore jamais 🐛
**Réponse** : les quatre diagnostics passent, et le regret moyen tombe de 1 200 à 70 sur 2 000 pas.
**Démarche** : (1) le générateur était recréé à chaque appel avec la même graine : il redonnait toujours 0,637, au-dessus de ε ; on le crée une fois, dans `__init__`. (2) `rng.integers(n_arms - 1)` n'atteint jamais le dernier bras. (3) `np.argmax` donne toujours le premier ex aequo : tirer au sort. (4) La moyenne divisait par le nombre total de pas : diviser par le nombre de tirages **du bras**. Les bugs se masquent : un agent qui n'explore jamais ne tire jamais au hasard (2 invisible) ; un agent qui ne joue qu'un bras a $t = n(a)$ (4 invisible).
**Erreurs fréquentes** : corriger (1) en mettant la graine à `None` : l'agent explore, mais n'est plus reproductible ; corriger (4) en divisant par `self.n[arm]` **avant** de l'incrémenter (division par zéro au premier tirage).

### Ex 11.25 — Un journal d'expériences reproductible (JSON) 🛠️
**Réponse** : un fichier JSON avec la configuration, les résultats (regret final moyen 35,4, erreur type 6,3, 72 % d'action optimale sur les 100 derniers pas), les versions de Python et de NumPy, la date ISO 8601 et le commit git ; l'expérience relancée depuis le journal redonne exactement les mêmes résultats, une autre graine en donne d'autres, et un journal falsifié est détecté.
**Démarche** : un générateur unique créé avec la graine, d'où l'on tire, dans un ordre fixe, les moyennes, puis une graine par bandit et par politique ; des `float` Python dans les résultats ; `json.dumps(..., indent=2, ensure_ascii=False)` ; `subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], ...)` pour le commit.
**Erreurs fréquentes** : utiliser `np.random.default_rng()` sans graine quelque part (le rerun échoue) ; mettre des `np.float32` dans le JSON ; comparer les résultats avec une tolérance dans `rerun_25` (on veut ici une égalité exacte : même code, même graine, même machine).
**Variante** : ajoute au journal l'empreinte (SHA-256) de ton fichier `mylearn/bandit.py`, pour savoir avec quel code une expérience a tourné.

### Ex 11.26 — Tournoi : ε-greedy, optimiste, UCB et Thompson 🔬
**Réponses** : a) **178** · b) **129** · c) **"Thompson"** · d) **False**.
**Démarche** : sur 100 bandits de Bernoulli à 10 bras, 1 000 pas, regret final moyen (± erreur type) : Thompson 28,1 (± 1,4), optimiste 40,7 (± 2,6), ε-greedy 67,9 (± 3,5), UCB1 129,1 (± 2,0), UCB avec $c = 2$ 178,0 (± 3,0). Les bandits « à bande » donnent à tous les agents les mêmes récompenses pour les mêmes tirages : la comparaison n'est pas brouillée par la chance.
**Pourquoi UCB1 perd** : sa garantie, valable à tout horizon, est une borne lâche et prudente. Le bonus $\sqrt{2 \ln t / N}$ vient de l'inégalité de Hoeffding, valable pour toute loi à valeurs dans $[0, 1]$ : il est taillé pour le pire cas, avec une marge de sécurité, et reste grand pendant des milliers de pas ; or beaucoup de bras du banc ont une moyenne proche de 0 ou de 1, donc une variance $p(1 - p)$ bien plus petite que le pire cas, 1/4. Sur un horizon de 100 000 pas, le regret d'ε-greedy continuerait de croître linéairement et UCB1 finirait devant. Thompson gagne parce que son prior uniforme est exactement la loi des moyennes du banc.
**Erreurs fréquentes** : réutiliser le même banc pour tous les agents (les bandes sont consommées : le deuxième agent reçoit d'autres récompenses) ; donner le même générateur à toutes les parties ; conclure sur une différence plus petite que deux erreurs types.

### Ex 11.27 — Défi : battre UCB1 sur un banc de bandits de Bernoulli 🏆
**Réponses** : a) **421,9657** ; la stratégie du corrigé (Thompson, prior uniforme) fait 26,5 sur le banc public (21 % du regret d'UCB1) et 29,2 sur le banc caché (22 %) : objectif atteint.
**Démarche** : un agent au hasard perd, à chaque pas, l'écart entre le meilleur bras et la moyenne des bras : $10/11 - 1/2 \approx 0{,}409$ en espérance (l'espérance du maximum de 10 uniformes, moins 1/2), et 0,422 sur les 100 bandits de ce banc, soit 422 sur 1 000 pas. Thompson avec $\mathrm{Beta}(1, 1)$, le prior exact du banc, suffit pour l'objectif. Pour le palier 🌟 (au plus 20 %) : un UCB réglé ($c \approx 0{,}3$ : 20,5 en public, 24,9 sur le banc caché, 19 %), ou Bayes-UCB. On règle sur le banc public seulement, avec des valeurs grossières : l'erreur type y est d'environ 2.
**Erreurs fréquentes** : régler un paramètre en regardant le banc caché (il ne sert plus alors à rien) ; prendre `step_size=0.1` avec Thompson (les succès ne se retrouvent plus à partir de $q \times n$) ; une politique qui lit les moyennes du bandit (interdit).
**Variante** : avec l'horizon connu (1 000 pas), une stratégie peut explorer davantage au début et exploiter à la fin ; essaie un Thompson qui joue la moyenne du posterior après le pas 600.
