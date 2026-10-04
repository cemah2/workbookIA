# 4 · Règle de Bayes — solutions

> Lis une solution **après** avoir vraiment essayé (règle des 15 minutes, puis les indices de `04_indices.md`). Pour chaque exercice : la réponse, la démarche (le *pourquoi*), les erreurs fréquentes et une variante pour aller plus loin. Les réponses des exercices ✏️ se vérifient aussi dans la partie 0 du notebook ; les exercices du notebook sont résolus et exécutés dans `05_solutions.ipynb`.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 4.Q1 — Deux écoles pour une même probabilité
1. Elle oblige à **écrire ce qu'on croit avant de voir les données** (le prior) : ce qu'on sait déjà sert ouvertement, au lieu d'être caché dans des choix implicites (fiche §4.1). Exemples : un filtre anti-spam qui part de la part de spams qu'on connaît déjà, ou un test A/B qui s'appuie sur les taux de conversion des tests précédents. 2. La **fréquence limite** d'un événement quand on répète l'expérience un très grand nombre de fois. 3. Un **degré de certitude** sur une affirmation, qui peut porter sur une grandeur inconnue (un paramètre). 4. Le **bayésien** : il donne une distribution au biais inconnu. Pour le fréquentiste, le biais est un nombre fixe, qui dépasse 0,6 ou non ; il n'y a rien de « répétable » à quoi attacher une fréquence. Il parlera plutôt d'un intervalle de confiance ou d'un test. 5. La définition **fréquentiste** : une fréquence observée sur de nombreux mois de novembre (un bayésien peut d'ailleurs s'en servir comme prior). 6. **Faux** : le ML utilise les deux, par exemple le bootstrap et les tests (fréquentistes) d'un côté, Naive Bayes et l'optimisation bayésienne de l'autre.

### 4.Q2 — Le fréquentiste et la hauteur de la montagne
1. Pour lui, la montagne a une hauteur **exacte**, et chaque mesure la manque un peu, à cause du bruit de mesure. 2. En combinant beaucoup de mesures, il espère s'approcher de cette valeur exacte. 3. Le nom vient de la définition de la probabilité comme **fréquence** à long terme d'un événement répété. Le livre (§4.2.1) l'explique autrement, par l'importance donnée à la valeur mesurée le plus souvent : c'est un raccourci (fiche, ⚠️). 4. Le plus souvent la **moyenne** des mesures, avec son erreur typique (ch. 2), plutôt que la valeur la plus fréquente. 5. **Vrai** : pour un fréquentiste strict, la hauteur est fixe ; elle dépasse 4 800 m ou non. Ses probabilités portent sur la **procédure** de mesure répétée, pas sur la montagne. 6. Elle est **divisée par 10** ($\sqrt{100}$, vitesse en $1/\sqrt{n}$).

### 4.Q3 — Le bayésien et la longueur du crayon
1. **Non** : un objet mal défini complique la mesure pour tout le monde, fréquentiste compris ; ce n'est pas ce qui distingue les deux écoles. Le portrait force le trait : il attribue au bayésien un doute qui tient à l'objet mesuré, pas à sa façon de définir la probabilité (§4.2.2 et §4.2.3). 2. Un bayésien peut très bien croire que le crayon a une longueur précise. Sa distribution décrit son **incertitude** sur cette longueur, pas une longueur qui changerait. 3. Ce qu'il **sait** de la longueur : quelles valeurs sont plausibles, et à quel point, au vu des mesures. 4. Un **prior** : une distribution de départ sur les longueurs possibles. 5. **Vrai**, en général : chaque mesure élimine des valeurs, et la distribution se resserre. 6. **Faux** : avec un modèle où la grandeur est fixe, le posterior se resserre de plus en plus et réagit de moins en moins à un changement (fiche, ⚠️). Pour suivre une grandeur qui bouge, il faut un modèle qui prévoit ce mouvement.

### 4.Q4 — Biais d'une pièce : le vocabulaire
1. Sa probabilité de tomber sur **face**. 2. $0{,}15 \times 200 = $ **30** faces en moyenne. 3. **0,5**. 4. $\frac{31}{50} = $ **0,62**. 5. Au début, chaque lancer change beaucoup la proportion (il compte pour $\frac{1}{n}$ du total) ; l'erreur typique de l'estimation diminue comme $\frac{1}{\sqrt{n}}$ (ch. 2). 6. La **loi de Bernoulli** de paramètre $\theta$.

### 4.Q5 — Une seule face change déjà le verdict
1. La pièce **truquée** : elle donne plus souvent face. 2. Les deux zones « face » : (équilibrée, face) et (truquée, face). 3. $\frac{0{,}5 \times 0{,}5}{0{,}5 \times 0{,}5 + 0{,}5 \times 0{,}8} = \frac{0{,}25}{0{,}65} \approx $ **0,385**. 4. $\frac{0{,}25}{0{,}25 + 0{,}5 \times 0{,}2} = \frac{0{,}25}{0{,}35} \approx $ **0,714**. 5. Les rapports de vraisemblance diffèrent : une face multiplie la cote « truquée contre équilibrée » par $\frac{0{,}8}{0{,}5} = 1{,}6$, une pile la divise par $\frac{0{,}5}{0{,}2} = 2{,}5$. Pile est l'indice le plus fort, car la pièce truquée donne rarement pile. 6. **Faux** : la pièce équilibrée garde 38,5 % de chances.

### 4.Q6 — Prior, vraisemblance, évidence, posterior : qui est qui ?
1. La probabilité qu'il y ait un feu, avant d'entendre le détecteur : le **prior**. 2. La probabilité que le détecteur sonne **s'il y a** un feu : la **vraisemblance** (sa sensibilité). 3. La probabilité qu'il sonne, pour un feu ou pour une fausse alerte (une cuisson, une douche chaude) : l'**évidence**, $P(S) = P(S \mid F)\,P(F) + P(S \mid \text{non } F)\,P(\text{non } F)$. 4. La probabilité d'un feu **sachant** que le détecteur sonne : le **posterior**. 5. Le fabricant mesure en laboratoire $P(S \mid F)$ et $P(S \mid \text{non } F)$ ; les habitants veulent $P(F \mid S)$, qui dépend aussi du prior, c'est-à-dire de la fréquence des feux chez eux. 6. Ce n'est ni une preuve ni une chose évidente : c'est un nombre, la probabilité de l'observation toutes hypothèses confondues, qui sert à normaliser.

**À retenir** : celui qui construit un test connaît les vraisemblances ; celui qui reçoit un résultat a besoin du posterior.

### 4.Q7 — La vraisemblance n'a pas à sommer à 1
1. 0,2 ; 0,5 ; 0,9 : somme **1,6**. 2. 0,8 ; 0,5 ; 0,1 : somme **1,4**. 3. **1** : pour une hypothèse donnée, face et pile couvrent toutes les issues. 4. **Vrai** : on divise par l'évidence pour cela. 5. **Vrai** : le facteur 10 multiplie aussi l'évidence et disparaît à la normalisation (seuls les **rapports** de vraisemblances comptent). 6. L'**évidence**, le dénominateur.

### 4.Q8 — Sonde spatiale : ne pas inverser la condition
1. $P(\text{détecté} \mid \text{vie}) = \frac{100}{101} \approx 0{,}99$, le **recall** (sensibilité). 2. La **precision** (valeur prédictive positive). 3. Le **prior** $P(\text{vie})$, la part de planètes habitées dans la région (la prévalence du ch. 3), avec le taux de fausses alertes de la sonde. 4. **Non** : le FNR vaut $P(\text{rien} \mid \text{vie}) = \frac{1}{101} \approx 0{,}0099$. $P(\text{vie} \mid \text{rien}) \approx 0{,}0011$ est l'autre conditionnelle, le **FOR** (taux de fausses omissions, $1 - $ NPV). 5. Les fausses alertes sont bien plus nombreuses que les ratés : 30 planètes stériles sur 899 déclenchent la sonde, contre une seule planète habitée manquée sur 101, et les planètes stériles sont neuf fois plus nombreuses. Une alerte est donc souvent fausse, alors qu'un « rien » vient très rarement d'une planète habitée. 6. Elle **s'effondrerait** vers 0 : presque toutes les alertes viendraient de planètes stériles (le piège de la prévalence, ch. 3).

### 4.Q9 — La boucle posterior → prior
1. Parce qu'une fois l'observation faite, le posterior décrit tout ce qu'on sait : il réunit le prior **et** l'observation (§4.6.1). 2. Les observations doivent être **indépendantes sachant l'hypothèse**. 3. **Vrai** : seul le contenu compte (deux faces, une pile), pas l'ordre (∂ 4.7). 4. **Faux** : ils ne sont indépendants que **sachant** la pièce ; une face rend la pièce truquée, donc une autre face, plus probable (fiche, ⚠️ §4.6.1). 5. À chaque tour, juste après le produit prior × vraisemblance. Mathématiquement, on pourrait ne normaliser qu'à la fin et obtenir le même résultat ; mais le produit de milliers de nombres plus petits que 1 finit par valoir 0 en machine (underflow, 4.18). 6. **Non** : pour une pièce, seuls le nombre de faces et le nombre de piles comptent ; un seul calcul suffit, $P(\theta) \times \theta^h (1 - \theta)^t$, en logarithmes (4.24).

### 4.Q10 — Plus de lancers, plus de certitude ?
1. Il se **resserre** autour de l'hypothèse la plus compatible avec les données. 2. **Faux** : une série inhabituelle peut tromper, et avec des biais proches 30 lancers ne suffisent souvent pas (4.12). 3. Vers la pièce **équilibrée**, avec une probabilité proche de 1 : 95 % de faces sont presque impossibles avec les deux pièces, mais encore moins avec la pièce de biais 0,05. Ce n'est **pas rassurant** : Bayes choisit la moins mauvaise des hypothèses proposées, sans dire qu'aucune ne convient. Il faut vérifier le modèle lui-même (4.19). 4. Elle reste à **0 pour toujours** (∂ 4.7). 5. **Non**, tant qu'il n'est nulle part nul : les données finissent par l'emporter, mais il en faut davantage (4.21). 6. Une forme en **cloche**, presque gaussienne (c'est une loi Beta) ; sa largeur diminue comme $\frac{1}{\sqrt{n}}$.

<a id="rappels"></a>

## 🔁 Rappels

### 4.R1 — Ch. 3 : precision = P(malade | test positif)
1. TP = **45**, FN = **5**, FP = **95**, TN = **855**. 2. $\frac{45}{45 + 95} = \frac{45}{140} \approx $ **0,321**. 3. $P(\text{malade} \mid \text{positif})$. 4. Sensibilité $\frac{45}{50} = 0{,}9$ ; FPR $\frac{95}{950} = 0{,}1$ ; prévalence $\frac{50}{1\,000} = 0{,}05$. 5. $\frac{0{,}9 \times 0{,}05}{0{,}9 \times 0{,}05 + 0{,}1 \times 0{,}95} = \frac{0{,}045}{0{,}14} \approx 0{,}321$ : la même valeur. 6. L'**évidence** $P(\text{positif}) = \frac{140}{1\,000} = 0{,}14$.

**À retenir** : la precision est un posterior ; la règle de Bayes la calcule à partir des taux du test et de la prévalence.

### 4.R2 — Ch. 1 : un filtre anti-spam apprend-il avec des labels ?
1. De l'apprentissage **supervisé**. 2. Des **labels** (*étiquettes*), le terme retenu dans ce workbook. 3. Sur un **jeu de test** mis de côté, jamais utilisé pour l'entraînement : sur les e-mails d'entraînement, le filtre peut avoir appris par cœur, et son score serait trop beau. 4. De l'apprentissage **non supervisé** (du *clustering*). 5. La part de spams parmi les e-mails, $P(\text{spam})$ (le prior), et, pour chaque mot, sa fréquence dans les spams et dans les e-mails normaux, $P(\text{mot} \mid \text{spam})$ et $P(\text{mot} \mid \text{normal})$ (les vraisemblances).

### 4.R3 — 0B : trois faces de suite avec une pièce truquée
1. $0{,}7^3 = $ **0,343**. 2. $1 - 0{,}343 = $ **0,657** (l'événement contraire de « trois faces »). 3. $\ln(0{,}7^3) = \ln 0{,}7 + \ln 0{,}7 + \ln 0{,}7 = 3 \ln 0{,}7 \approx 3 \times (-0{,}3567) \approx $ **−1,070**. 4. $0{,}7 \times 0{,}3 \times 0{,}7 = $ **0,147** ; exactement deux faces : trois ordres possibles (FFP, FPF, PFF), soit $3 \times 0{,}147 = $ **0,441**. 5. $\prod_{i=1}^{3} \theta^{x_i}(1 - \theta)^{1 - x_i} = \theta^{x_1 + x_2 + x_3}(1 - \theta)^{3 - (x_1 + x_2 + x_3)} = 0{,}7^2 \times 0{,}3$ : un facteur $\theta$ par face, un facteur $1 - \theta$ par pile.

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 4.1 — Une face : la pièce est-elle équilibrée ? ✏️
a) **0,5** · b) **[0,5 ; 0,75]** · c) $0{,}5 \times 0{,}5 + 0{,}5 \times 0{,}75 = $ **0,625** · d) $\frac{0{,}25}{0{,}625} = $ **0,4** · e) **0,6** · f) $800 \times 0{,}625 = $ **500** · g) $800 \times 0{,}5 \times 0{,}5 = $ **200**.
**Pourquoi** : l'évidence additionne les deux façons d'obtenir face. Les 400 essais faits avec la pièce équilibrée donnent 200 faces ; les 400 faits avec la pièce truquée en donnent 300. Parmi les 500 faces, 200 viennent de la pièce équilibrée : $\frac{200}{500} = 0{,}4$, le posterior de d. C'est le mur peint de la fiche avec des effectifs.
h) Réponse modèle : « Sur le mur, la zone (truquée, face) est plus grande que la zone (équilibrée, face), 0,375 contre 0,25 : une fléchette qui tombe dans une zone « face » a plus de chances d'être dans la zone de la truquée. Une seule face suffit donc à faire passer la truquée de 50 % à 60 %. »
**Erreurs fréquentes** : additionner les vraisemblances sans les pondérer par les priors (1,25, qui dépasse 1) ; répondre 0,75 en e (c'est $P(\text{face} \mid \text{truquée})$, l'erreur du procureur) ; oublier de diviser par l'évidence (0,25).
**Variante** : le sac contient 2 pièces équilibrées et 1 truquée. Recalcule $P(\text{équilibrée} \mid \text{face})$ : $\frac{2/3 \times 0{,}5}{2/3 \times 0{,}5 + 1/3 \times 0{,}75} = \frac{4}{7} \approx 0{,}571$. Le prior compte.

### Ex 4.2 — Une pile : le verdict s'inverse ✏️
a) **0,25** · b) $0{,}5 \times 0{,}5 + 0{,}5 \times 0{,}25 = $ **0,375** · c) $\frac{0{,}25}{0{,}375} = \frac{2}{3} \approx $ **0,667** · d) $\frac{2}{3} - \frac{1}{2} = \frac{1}{6} \approx $ **0,167** · e) $0{,}4 \times 0{,}625 + \frac{2}{3} \times 0{,}375 = 0{,}25 + 0{,}25 = $ **0,500**.
**Pourquoi** : après une pile, la pièce équilibrée devient la plus probable (0,667), alors qu'après une face c'était la truquée (0,6) : le verdict s'inverse. Les deux mises à jour ne sont pas symétriques : face fait perdre 0,1 à « équilibrée », pile lui fait gagner 0,167.
f) e redonne le **prior**, 0,5. En moyenne sur les observations possibles, le posterior vaut le prior. On sait que face fera baisser $P(\text{équilibrée})$ et que pile la fera monter, mais la moyenne des déplacements, pondérée par la probabilité de chaque observation, est nulle : $-0{,}1 \times 0{,}625 + \frac{1}{6} \times 0{,}375 = 0$. C'est la formule des probabilités totales, $P(H) = P(H \mid \text{face})\,P(\text{face}) + P(H \mid \text{pile})\,P(\text{pile})$. Pile est un indice plus fort parce que son rapport de vraisemblance s'éloigne plus de 1 : $\frac{P(\text{pile} \mid \text{équilibrée})}{P(\text{pile} \mid \text{truquée})} = \frac{0{,}5}{0{,}25} = 2$, contre $\frac{0{,}75}{0{,}5} = 1{,}5$ en faveur de la truquée pour face. Une observation **rare** sous une hypothèse est un indice fort contre elle. En contrepartie, la face, plus fréquente, arrive plus souvent : c'est ce qui équilibre la moyenne de e.
**Erreurs fréquentes** : reprendre l'évidence de face (0,625) ; répondre 0,1 en d, par symétrie avec la baisse de 4.1 ; donner en d le posterior lui-même (0,667) au lieu de son écart au prior, ou une hausse relative (0,333, soit +33 % du prior) au lieu de la différence.
**Variante** : avec une pièce truquée de biais 0,9, calcule les deux posteriors après face et après pile (fiche §4.4 : 0,357 et 0,833), et vérifie encore que leur moyenne pondérée vaut 0,5.

### Ex 4.3 — Retrouver la règle de Bayes en trois lignes ∂
1. Règle du produit, deux fois : $P(H, O) = P(H \mid O)\,P(O)$ et $P(H, O) = P(O \mid H)\,P(H)$. Donc $P(H \mid O)\,P(O) = P(O \mid H)\,P(H)$, et, si $P(O) > 0$, on divise : $P(H \mid O) = \frac{P(O \mid H)\,P(H)}{P(O)}$. Si $P(O) = 0$, l'observation est impossible, et la question « sachant $O$ » n'a pas de sens.
2. Les $H_i$ découpent tous les cas sans chevauchement, donc $O$ est la réunion disjointe des événements « $O$ et $H_i$ » : $P(O) = \sum_j P(O, H_j) = \sum_j P(O \mid H_j)\,P(H_j)$. D'où $P(H_i \mid O) = \frac{P(O \mid H_i)\,P(H_i)}{\sum_j P(O \mid H_j)\,P(H_j)}$.
3. La somme des numérateurs est exactement le dénominateur : les posteriors somment à 1. Et $P(H_i \mid O) = c \cdot P(O \mid H_i)\,P(H_i)$, avec $c = \frac{1}{P(O)}$, le même pour tous les $i$.
4. Si $P(O \mid H_2)\,P(H_2) > 0$, on divise la règle de Bayes pour $H_1$ par celle pour $H_2$ : les deux $P(O)$ se simplifient. Exercice 4.1 : cote a priori « truquée contre équilibrée » = 1 ; rapport de vraisemblance $\frac{0{,}75}{0{,}5} = 1{,}5$ ; cote a posteriori 1,5, soit $P(\text{truquée} \mid \text{face}) = \frac{1{,}5}{1 + 1{,}5} = 0{,}6$. Aucune évidence à calculer.
5. Si $P(O \mid H_i) = q > 0$ pour tout $i$ (avec $q = 0$, l'observation serait impossible), alors $P(O) = q \sum_j P(H_j) = q$, et $P(H_i \mid O) = \frac{q\,P(H_i)}{q} = P(H_i)$. Une observation également probable sous toutes les hypothèses n'apprend **rien** sur elles. Exemple : la couleur de la pièce, si les deux pièces sont identiques d'aspect.

**Erreurs fréquentes** : démontrer sur un exemple chiffré au lieu du cas général ; oublier la condition $P(O) > 0$ ; en 2, oublier que les hypothèses doivent couvrir tous les cas sans se chevaucher.
**Variante** : avec la forme « cotes », montre qu'une observation deux fois plus probable sous $H_1$ que sous $H_2$ double la cote de $H_1$ contre $H_2$, quel que soit le prior. (Le rapport de vraisemblance vaut 2, et il multiplie la cote a priori, quelle qu'elle soit. En probabilité, avec deux hypothèses seulement, l'effet dépend du prior : 0,5 devient $\frac{2}{3}$, mais 0,1 ne devient que $\frac{2}{11} \approx 0{,}18$.)

### Ex 4.4 — Vie extraterrestre : lire la sonde avec Bayes ✏️
a) $\frac{240}{250} = $ **0,96** · b) $\frac{1\,645}{1\,750} = $ **0,94** · c) $\frac{0{,}04 \times 0{,}05}{0{,}04 \times 0{,}05 + 0{,}94 \times 0{,}95} = \frac{0{,}002}{0{,}895} \approx 0{,}0022346$, soit **2,23** millièmes · d) $\frac{0{,}96 \times 0{,}05}{0{,}96 \times 0{,}05 + 0{,}06 \times 0{,}95} = \frac{0{,}048}{0{,}105} \approx $ **0,457** · e) $\frac{240}{240 + 105} = \frac{240}{345} \approx $ **0,696** · f) $0{,}06 \times 9\,500 = $ **570**.
**Pourquoi** : une sonde négative rassure beaucoup (une chance sur 450 environ que la planète soit habitée), mais une sonde positive ne suffit pas : moins d'une chance sur deux ! Sur 10 000 planètes de la région, 500 sont habitées et 480 d'entre elles déclenchent la sonde ; mais 6 % des 9 500 planètes stériles la déclenchent aussi, soit 570 fausses alertes, plus que de vraies.
g) Réponse modèle : « La precision dépend de la proportion de planètes habitées. Dans le test, il y en avait 12,5 % (250 sur 2 000) ; dans la région, 5 %. Avec moins de planètes habitées, les fausses alertes pèsent plus lourd, et $P(\text{habitée} \mid \text{détecté})$ baisse. Les deux nombres seraient égaux si le prior de la région valait la proportion du test, 12,5 %. C'est le cas dans le livre, où l'expérience du capitaine (environ 10 %) est proche de la proportion de son test (101 sur 1 000). »
**Erreurs fréquentes** : prendre 12,5 % comme prior (c donnerait 6,04 millièmes et d 0,696) ; oublier un des deux termes de l'évidence (2,24 millièmes : c'est la cote « habitée contre stérile ») ; arrondir en deux fois (0,002235, puis 2,24 en arrondissant le 5 vers le haut : n'arrondis qu'à la fin) ; confondre spécificité et NPV ($\frac{1\,645}{1\,655} \approx 0{,}994$) ; répondre la sensibilité (0,96) en d, l'erreur du procureur ; compter les fausses alertes sur les 10 000 planètes au lieu des 9 500 planètes stériles (600 au lieu de 570).
**Variante** : quelle sensibilité, à spécificité égale, faudrait-il pour que $P(\text{habitée} \mid \text{détecté})$ atteigne 0,5 dans la région ? (Il faut $0{,}05\,s = 0{,}06 \times 0{,}95$, soit $s = 1{,}14$ : impossible ! C'est la spécificité qu'il faut améliorer.)

### Ex 4.5 — Deux faces : une mise à jour double ou deux simples ? ✏️
a) Prior 0,4 (le posterior de 4.1) : $\frac{0{,}4 \times 0{,}5}{0{,}4 \times 0{,}5 + 0{,}6 \times 0{,}75} = \frac{0{,}2}{0{,}65} \approx $ **0,308** · b) $0{,}75^2 = $ **0,5625** · c) $0{,}5 \times 0{,}25 + 0{,}5 \times 0{,}5625 = 0{,}40625 \approx $ **0,406** · d) $\frac{0{,}125}{0{,}40625} = \frac{4}{13} \approx $ **0,308** · e) $\frac{0{,}5 \times 0{,}5^3}{0{,}5 \times 0{,}5^3 + 0{,}5 \times 0{,}75^2 \times 0{,}25} = \frac{8}{17} \approx $ **0,471** · f) **0,471**.
**Pourquoi** : en deux temps ou d'un coup, on calcule le même rapport. En deux temps, le dénominateur du premier lancer se simplifie avec le numérateur du second (∂ 4.7, question 4). L'ordre ne compte pas, car seule la vraisemblance de la suite entière intervient, $\theta^h (1 - \theta)^t$ : le produit ne dépend pas de l'ordre des facteurs.
g) L'hypothèse : les lancers sont **indépendants sachant la pièce** (si l'on sait quelle pièce on tient, un lancer ne change rien au suivant).
**Erreurs fréquentes** : calculer c comme $P(\text{face})^2 = 0{,}625^2 \approx 0{,}391$, comme si les deux lancers d'une pièce **inconnue** étaient indépendants (fiche, ⚠️ §4.6.1) ; repartir du prior 0,5 au second lancer ; additionner les vraisemblances (1,5) au lieu de les multiplier.
**Variante** : combien de faces de suite faut-il pour que $P(\text{équilibrée})$ passe sous 0,05 ? (Il faut $1{,}5^k > 19$, soit $k = 8$ faces.)

### Ex 4.6 — Cinq hypothèses de biais après face, pile, face ✏️
a) **[0 ; 0,1 ; 0,4 ; 0,3 ; 0,2]** · b) **[0 ; 0,214 ; 0,571 ; 0,214 ; 0]** · c) **[0 ; 0,107 ; 0,571 ; 0,321 ; 0]** · d) **0,0875** · e) **0,5**.
**Pourquoi** : après face, les produits prior × $\theta$ valent $[0 ;\ 0{,}05 ;\ 0{,}2 ;\ 0{,}15 ;\ 0{,}1]$ (somme 0,5) ; après pile, on multiplie par $1 - \theta$, ce qui donne $[0 ;\ 0{,}075 ;\ 0{,}2 ;\ 0{,}075 ;\ 0]$ (somme 0,35), soit $[0 ;\ \frac{3}{14} ;\ \frac{4}{7} ;\ \frac{3}{14} ;\ 0]$ ; après la seconde face, $[0 ;\ \frac{3}{28} ;\ \frac{4}{7} ;\ \frac{9}{28} ;\ 0]$. d : $\sum \text{prior} \times \theta^2(1 - \theta) = 0{,}2 \times 0{,}046875 + 0{,}4 \times 0{,}125 + 0{,}2 \times 0{,}140625 = 0{,}0875$ ; c'est aussi le produit des trois évidences successives, $0{,}5 \times 0{,}35 \times 0{,}5$.
f) $\theta = 0$ est éliminée à la première face, $\theta = 1$ à la première pile, pour toujours (vraisemblance nulle). Avec un prior uniforme, le posterior serait $[0 ;\ 0{,}15 ;\ 0{,}4 ;\ 0{,}45 ;\ 0]$ et le MAP passerait à **0,75** (c'est l'exemple de la docstring de `coin_bias_posterior`). Avec trois lancers, le prior pèse encore autant que les données : le prior de l'archéologue, qui favorise 0,5, suffit à changer la conclusion.
**Erreurs fréquentes** : repartir du prior de l'énoncé à chaque lancer ; oublier de normaliser (les posteriors ne somment plus à 1) ; confondre le MAP (une valeur de $\theta$) avec la plus grande probabilité (0,571).
**Variante** : continue avec les lancers pile, pile, face : quelles hypothèses restent possibles, et laquelle domine ? (Toujours 0,25, 0,5 et 0,75 : le posterior devient $[0 ;\ 0{,}148 ;\ 0{,}703 ;\ 0{,}148 ;\ 0]$, et 0,5 domine nettement, avec 3 faces et 3 piles.)

### Ex 4.7 — Le posterior reste une distribution, un prior nul reste nul ∂
1. Chaque posterior est un quotient de deux nombres positifs ou nuls, $P(O \mid H_i)\,P(H_i) \ge 0$ et $P(O) > 0$ : il est positif ou nul (la vraisemblance $P(O \mid H_i)$ est fixée par le modèle, même quand $P(H_i) = 0$ : le produit est bien défini, et nul). Leur somme vaut $\frac{\sum_i P(O \mid H_i)\,P(H_i)}{P(O)} = \frac{P(O)}{P(O)} = 1$.
2. Si $P(H_i) = 0$, le numérateur $P(O \mid H_i) \times 0$ est nul, donc $P(H_i \mid O) = 0$. Ce posterior devient le prior suivant, et le raisonnement se répète (récurrence) : l'hypothèse reste à 0 pour toujours, quelles que soient les données. Conséquence : ne donner un prior **nul** qu'à une hypothèse **certainement** impossible ; sinon, un prior petit mais positif, que les données pourront corriger (c'est la « règle de Cromwell » des statisticiens).
3. Si $P(O \mid H_i) = 0$, le numérateur est nul, donc le posterior aussi ; puis la question 2 s'applique. Exemple : après une face, l'hypothèse $\theta = 0$ (« la pièce ne tombe jamais sur face ») est éliminée pour toujours.
4. Pour $n = 1$, c'est la règle de Bayes (∂ 4.3). Supposons la formule vraie après $n - 1$ observations, et notons $N_i = P(H_i) \prod_{k < n} P(o_k \mid H_i)$, de sorte que le prior du tour $n$ vaut $\frac{N_i}{\sum_j N_j}$. La règle de Bayes avec ce prior et l'observation $o_n$ donne $\frac{P(o_n \mid H_i)\,N_i / \sum_j N_j}{\sum_l P(o_n \mid H_l)\,N_l / \sum_j N_j}$ ; les $\sum_j N_j$ se simplifient, et il reste $\frac{N_i\,P(o_n \mid H_i)}{\sum_l N_l\,P(o_n \mid H_l)}$, la formule au rang $n$. L'indépendance sachant l'hypothèse sert à écrire $P(o_1, \ldots, o_n \mid H_i) = \prod_k P(o_k \mid H_i)$. Un produit ne dépend pas de l'ordre de ses facteurs : l'ordre des observations ne compte pas.
5. $\log P(H_i) + \sum_{k=1}^{n} \log P(o_k \mid H_i)$. Avec $n$ = plusieurs milliers, le produit de la question 4 descend sous le plus petit nombre représentable ($\approx 5 \times 10^{-324}$) et vaut 0 en machine ; la somme des logarithmes, elle, vaut quelques milliers en valeur absolue, sans aucun problème.
6. Si l'on ajoute $c$ à tous les logarithmes, toutes les exponentielles sont multipliées par $e^c$ ; ce facteur apparaît au numérateur et au dénominateur de la normalisation, et se simplifie. Avec $c = -\max_i \ell_i$, la plus grande exponentielle vaut 1 : on évite à la fois l'underflow (tout à 0) et l'overflow (l'infini).

**Erreurs fréquentes** : en 4, oublier de dire où sert l'indépendance ; en 2, ne traiter qu'une observation au lieu de faire la récurrence.
**Variante** : deux hypothèses ont la même vraisemblance pour toutes les issues possibles. Montre que le rapport de leurs posteriors reste égal au rapport de leurs priors, quelles que soient les observations : les données ne peuvent pas les départager. (D'après la question 4, chaque posterior est proportionnel à son prior multiplié par $\prod_k P(o_k \mid H_i)$ ; ces deux produits sont égaux, et ils se simplifient dans le rapport.)

### Ex 4.8 — Combien de sondes pour descendre sous un sur un million ? ✏️
a) $\frac{0{,}05}{0{,}95} = \frac{1}{19} \approx $ **0,0526** · b) $\frac{P(\text{rien} \mid \text{stérile})}{P(\text{rien} \mid \text{habitée})} = \frac{0{,}94}{0{,}04} = $ **23,5** · c) cote $\frac{1}{19} \times \frac{1}{23{,}5^2} \approx 9{,}5304 \times 10^{-5}$, soit $P = \frac{\text{cote}}{1 + \text{cote}} \approx 9{,}5295 \times 10^{-5}$, donc **95,295** millionièmes (avec $\frac{1}{19}$ et $23{,}5$ exacts jusqu'au bout) · d) **4** sondes · e) cote $\frac{1}{19} \times 16^2 \approx 13{,}5$, soit $P = \frac{256}{275} \approx $ **0,931** · f) cote $\frac{1}{19} \times \frac{16}{23{,}5} \approx 0{,}0358$, soit $P \approx $ **0,035**.
**Pourquoi** : avec la forme « cotes », chaque sonde **multiplie** la cote par son rapport de vraisemblance : $\frac{0{,}96}{0{,}06} = 16$ pour une sonde positive, $\frac{0{,}04}{0{,}94} = \frac{1}{23{,}5}$ pour une négative. d : après 3 négatives, $P \approx 4{,}1 \times 10^{-6}$, encore au-dessus de $10^{-6}$ ; après 4, $P \approx 1{,}7 \times 10^{-7}$.
g) f n'est pas le prior parce que les deux rapports ne sont **pas inverses** : $16 \times \frac{1}{23{,}5} \approx 0{,}68 < 1$. Une sonde négative pèse un peu plus qu'une positive, car la sonde rate moins souvent la vie (4 %) qu'elle ne déclenche de fausses alertes (6 %). Avec une sensibilité égale à la spécificité, les deux s'annuleraient exactement. Deux sondes peuvent se tromper **ensemble** : une vie cachée sous la glace échappera à toutes les sondes du même modèle, et une même perturbation peut tromper deux mesures. Leurs erreurs sont alors liées, et le produit des rapports de vraisemblance exagère la preuve : le calcul de d est **trop optimiste**. En pratique, on envoie des sondes de modèles différents, ou on vérifie l'indépendance sur des planètes connues.
**Erreurs fréquentes** : en b, comparer deux issues différentes ($\frac{0{,}96}{0{,}04} = 24$) au lieu de la même issue sous les deux hypothèses ; donner la cote au lieu de la probabilité (95,304 millionièmes en c, 13,47 en e) ; en c, repartir de valeurs arrondies (la cote 0,0526 donne 95,238 ; la réponse de 4.4 c arrondie, 0,00223, donne 95,097 ; un rapport de vraisemblance arrondi à 0,0426 donne 95,505) : à 3 décimales, il faut garder les fractions ; en d, arrondir au plus proche la solution $k \approx 3{,}44$ de $19 \times 23{,}5^k = 10^6$ au lieu de prendre l'entier supérieur ; supposer que deux sondes contraires s'annulent (f = 0,05).
**Variante** : avec la sonde du livre (recall $\frac{100}{101}$, faux positifs $\frac{30}{899}$) et un prior de 10 %, combien de sondes négatives faut-il ? (Trois, car chaque négative divise la cote par environ 98.)

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 4.9 — La règle de Bayes sans formule, en cinq lignes 🗣️
« Prenons 10 000 personnes, dont 20 ont la maladie : c'est ce qu'on sait **avant** le test. Le test repère 19 de ces 20 malades, mais il se trompe aussi sur 5 % des 9 980 personnes saines, soit 499 fausses alertes. Un résultat positif est un **indice**, pas une preuve : sur les 518 personnes testées positives, seules 19 sont malades. Après le test, la **mise à jour** fait passer le risque de 2 sur 1 000 à environ 1 sur 27, ce qui justifie un second examen, pas une panique. »
**Ce qui compte** : partir d'une population en effectifs (les fréquences naturelles du ch. 3) ; montrer que les fausses alertes viennent du **grand** groupe des personnes saines ; dire ce que le résultat change (de 2 sur 1 000 à 1 sur 27) et ce qu'il faut faire ensuite.
**Erreurs fréquentes** : parler en pourcentages de pourcentages (« 95 % de 0,2 % »), que ton ami ne peut pas suivre ; oublier le grand groupe des personnes saines, d'où viennent presque toutes les fausses alertes ; confondre « le test se trompe rarement sur une personne saine » et « un test positif se trompe rarement » (l'inversion de la condition) ; dépasser les cinq lignes, ou glisser une formule.
**Variante** : ton ami demande pourquoi son médecin fait pourtant confiance au même test chez un patient qui a déjà des symptômes. Refais l'explication en effectifs, avec 1 malade sur 5 parmi les patients qui consultent. (Sur 10 000 patients, 2 000 sont malades et le test en repère 1 900 ; il sonne aussi pour 5 % des 8 000 autres, soit 400 fausses alertes. Sur 2 300 positifs, 1 900 sont malades, plus de 8 sur 10. Le test n'a pas changé, seul l'« avant » a changé.)

### Ex 4.10 — Le prior est un choix : erreur du procureur et priors partiaux ⚖️
1. L'expert donne $P(\text{compatible} \mid \text{innocent}) = 10^{-6}$ ; le procureur affirme $P(\text{innocent} \mid \text{compatible}) = 10^{-6}$. Il inverse la condition : c'est l'**erreur du procureur** (*prosecutor's fallacy*).
2. Parmi les 3 millions de personnes du fichier, on attend environ $3 \times 10^6 \times 10^{-6} = 3$ innocents compatibles, plus le coupable : environ 4 personnes compatibles. Sans autre indice, l'accusé est l'une de ces 4 personnes : $P(\text{innocent} \mid \text{compatible}) \approx \frac{3}{4}$. Très loin d'un sur un million ! Le prior, ici une personne parmi 3 millions, change tout ; un indice à charge indépendant de l'ADN (un témoin, un mobile) changerait encore le calcul.
3. Les deux erreurs dénoncées par la Royal Statistical Society (déclaration du 23 octobre 2001). (1) Élever 1 sur 8 543 au carré suppose les deux décès **indépendants**, alors que des facteurs génétiques ou environnementaux communs à une famille rendent un second décès plus probable. (2) L'erreur du procureur : le nombre de 1 sur 73 millions répond à la question « deux morts subites, si la mère est innocente ? », alors que la question du procès est « innocente, sachant les deux décès ? ». Il fallait comparer les probabilités des deux explications, toutes deux très rares (un double meurtre de ses enfants est lui aussi exceptionnel), et non regarder seulement la rareté de l'une. La condamnation a été annulée en janvier 2003, principalement parce que des analyses médicales, qui suggéraient une cause naturelle de la mort du second bébé, n'avaient pas été communiquées à la défense.
4. Le quartier sert de substitut à des caractéristiques protégées (origine, revenu) : deux clients identiques seraient traités différemment selon leur adresse. C'est de la discrimination indirecte, et ce prior se renforce lui-même : moins de crédits dans un quartier, plus de difficultés, plus de défauts mesurés. Le client ne peut pas agir sur ce prior, et ne sait souvent pas qu'il existe. En Europe, le RGPD encadre les décisions automatisées, et le règlement européen sur l'IA classe l'évaluation de la solvabilité parmi les usages à haut risque.
5. Le prior doit être choisi **ouvertement**, par l'équipe qui porte la décision, avec le métier, le juriste et, si possible, des représentants des personnes concernées ; il doit être documenté (d'où vient-il ? sur quelles données ?). Pour vérifier qu'il ne décide pas tout seul, on fait une **analyse de sensibilité** : on refait le calcul avec plusieurs priors raisonnables, et on vérifie que la décision ne change pas (ou on dit clairement quand elle change).
6. Le bayésianisme **automatique** (une règle connue d'avance, par exemple un prior uniforme ou un prior tiré de données publiques) est plus facile à défendre, parce qu'il ne dépend pas de l'opinion de la personne qui décide. Le subjectif peut être légitime (une expertise réelle), à condition d'être écrit, justifié et soumis à une analyse de sensibilité.

**Erreurs fréquentes** : en 1, écrire deux fois la même probabilité, alors que les deux phrases n'ont pas la même chose après la barre ; en 2, oublier le coupable (3 personnes compatibles au lieu d'environ 4), ou garder la réponse du procureur, un sur un million ; en 3, ne relever qu'une des deux erreurs, l'élévation au carré (l'indépendance) ou l'inversion de la condition ; en 4, croire qu'il suffit de retirer l'origine des clients du modèle, alors que l'adresse joue le même rôle.
**Variante** : la trace est comparée à un fichier de 30 millions de personnes au lieu de 3 millions ; le coupable y figure toujours, et rien d'autre n'accuse la personne retrouvée. Que vaut, à peu près, $P(\text{innocent} \mid \text{compatible})$ ? (Environ 30 innocents compatibles, plus le coupable : $\frac{30}{31} \approx 0{,}97$. Plus le fichier est grand, plus une correspondance due au hasard devient probable : une correspondance trouvée en fouillant un fichier pèse moins qu'une correspondance avec un suspect que d'autres indices désignaient déjà.)

### Ex 4.11 — VanderPlas (2014) : fréquentisme et bayésianisme 📄
1. Fréquentiste : une probabilité n'a de sens que comme **limite de mesures répétées**. Bayésien : la notion de probabilité est **étendue aux degrés de certitude** sur des affirmations.
2. Le fréquentiste estime le flux par le maximum de vraisemblance (une moyenne pondérée des mesures, avec son erreur) ; le bayésien, avec un prior plat, obtient un posterior proportionnel à la vraisemblance. Les deux résultats sont pratiquement identiques : pour un problème simple avec un prior plat, les deux approches calculent la même chose sous deux interprétations.
3. Un paramètre de nuisance est un paramètre nécessaire au modèle, mais qui n'intéresse pas en lui-même : dans le billard de Bayes, la position inconnue de la marque qui décide qui gagne chaque manche. L'approche fréquentiste **naïve** remplace ce paramètre par son estimation, et trouve une cote d'environ 18 contre 1 contre Bob. L'approche bayésienne **intègre** (moyenne) sur toutes les positions possibles, pondérées par leur posterior, et trouve environ 10 contre 1, ce que confirme une simulation.
4. Fréquentiste : « si l'on répétait l'expérience de nombreuses fois, l'intervalle calculé contiendrait la vraie valeur dans 95 % des cas ». Bayésien : « au vu des données observées, la vraie valeur est dans la région de crédibilité avec une probabilité de 95 % ». Dans l'exemple de Jaynes, les données imposent une borne (le paramètre doit être inférieur à la plus petite observation, 10), et pourtant l'intervalle de confiance à 95 % calculé, environ (10,2 ; 12,2), est entièrement au-delà : il est absurde pour ces données, tout en respectant sa promesse sur l'ensemble des expériences répétées. La région de crédibilité respecte la borne.
5. Fréquentiste : `statsmodels`. Bayésien par MCMC : `emcee`, `PyMC` et `PyStan`.
6. Le fréquentisme se calcule souvent facilement et convient aux processus et aux mesures vraiment **répétables**, mais il peut buter sur des petits datasets et des modèles très éloignés de la loi normale. Le bayésianisme demande un prior, qui peut être subjectif, et souvent des calculs lourds (MCMC), mais il est souvent plus direct dans son principe : pour VanderPlas, ses résultats répondent plus directement aux questions que se pose un scientifique.
7. `mylearn.bayes` est **bayésien** : priors, posteriors, MAP et intervalle de crédibilité. Le `bootstrap_ci` du ch. 2 est **fréquentiste** : son intervalle est une promesse sur la méthode, qu'on évalue sur des rééchantillons (4.25 compare les deux).

**Erreurs fréquentes** : lire l'intervalle de confiance comme une probabilité sur le paramètre (« 95 % de chances que le flux soit dans cet intervalle ») ; en 3, confondre marginaliser (faire la moyenne sur toutes les positions de la marque) et remplacer la position par sa meilleure estimation ; conclure de l'exemple de Jaynes que l'approche fréquentiste est fausse, alors que son intervalle tient sa promesse sur les expériences répétées ; en 7, classer `bootstrap_ci` parmi les méthodes bayésiennes parce qu'il fait des tirages au hasard.
**Variante** : un rapport d'analyse écrit, à propos d'un intervalle bootstrap à 95 % sur un taux de conversion : « il y a 95 % de chances que le vrai taux soit dans cet intervalle ». Corrige la phrase, puis dis quel intervalle permettrait de l'écrire. (« Si l'on refaisait l'étude de nombreuses fois, la méthode donnerait un intervalle qui contient le vrai taux dans environ 95 % des cas. » La phrase du rapport est celle d'un intervalle de crédibilité, qui demande un prior, par exemple une loi Beta comme en 4.22.)

<a id="entretien"></a>

## 💼 Entretien

### 4.E1 — Expliquer la règle de Bayes avec un test médical

**Réponse modèle en 60 secondes** : « La règle de Bayes sert à inverser une probabilité conditionnelle. Le laboratoire connaît $P(\text{positif} \mid \text{malade})$, la sensibilité de son test ; le patient veut $P(\text{malade} \mid \text{positif})$. La règle dit que le posterior est proportionnel à la vraisemblance fois le prior : $P(\text{malade} \mid +) = \frac{P(+ \mid \text{malade})\,P(\text{malade})}{P(+)}$, où $P(+)$ additionne les vrais et les faux positifs. Exemple : maladie à 1 %, test sensible à 99 % et spécifique à 95 %. Sur 10 000 personnes, 99 malades sont détectés, mais 495 personnes saines aussi : un positif n'est malade qu'une fois sur six environ. Tout tient au prior, la prévalence : dans un service où la moitié des patients sont malades, le même test serait presque toujours juste. C'est aussi pour ça que la precision d'un classifieur dépend de la population où on le déploie. »
**Relances possibles** : « Que faire après un positif ? » (un second test, indépendant : le posterior du premier devient le prior du second) · « Et un test négatif ? » (la NPV est excellente ici, environ 0,9999 : un seul faux négatif parmi 9 406 négatifs) · « Comment l'appliquer à un classifieur ? » (recalculer la precision avec la prévalence de la population visée ; recalibrer les probabilités si la prévalence change).

### 4.E2 — Fréquentiste ou bayésien : quelle différence en pratique ?

**Réponse modèle en 60 secondes** : « Tout part de la définition de la probabilité. Pour un fréquentiste, c'est une fréquence sur des répétitions ; un paramètre, comme le taux de conversion d'un site, est fixe et inconnu, et on étudie les propriétés de nos méthodes sur des échantillons répétés : intervalles de confiance, p-valeurs. Pour un bayésien, une probabilité mesure aussi un degré de certitude : on met un prior sur le paramètre, et les données donnent un posterior. Pour les intervalles, la différence est nette. Un intervalle de confiance à 95 % promet que **la méthode** capture la vraie valeur dans 95 % des expériences répétées ; il ne dit pas qu'il y a 95 % de chances que la valeur soit dans **cet** intervalle-là. Un intervalle de crédibilité dit exactement ça, sachant les données et le prior. Avec beaucoup de données et un prior peu informatif, les deux coïncident souvent ; avec peu de données, ils peuvent diverger. »
**Relances possibles** : « Lequel utilisez-vous ? » (selon la question : bootstrap pour une métrique de modèle, bayésien pour peu de données, un prior utile ou un besoin de probabilité sur le paramètre) · « Les deux peuvent-ils se contredire ? » (oui, avec peu de données ou un prior fort ; l'exemple de Jaynes chez VanderPlas) · « Qu'est-ce qu'une p-valeur ? » (la probabilité, si l'hypothèse nulle était vraie, d'observer des données au moins aussi extrêmes ; pas la probabilité que l'hypothèse nulle soit vraie).

### 4.E3 — Qu'est-ce qu'un prior et comment le choisir ?

**Réponse modèle en 60 secondes** : « Le prior est la distribution qui décrit ce qu'on sait d'un paramètre avant de voir les nouvelles données. Je le choisis à partir de connaissances réelles : des études passées, des données historiques, des contraintes physiques (un taux est entre 0 et 1). Sans information, je prends un prior peu informatif, uniforme ou large. Quand c'est possible, je prends un prior conjugué, par exemple une loi Beta pour une proportion, parce que le calcul est alors exact. Deux règles. Ne jamais mettre une probabilité nulle sur ce qui n'est pas impossible : aucune donnée ne la ferait remonter. Et faire une analyse de sensibilité : recalculer avec plusieurs priors raisonnables. Si la conclusion change, les données ne suffisent pas, et il faut le dire. Un prior trompeur se corrige avec assez de données, mais il en coûte. Enfin, je documente mon prior : c'est un choix, qu'on doit pouvoir discuter. »
**Relances possibles** : « Un exemple de prior en ML ? » (la régularisation L2, ridge, correspond à un prior gaussien centré sur 0 sur les poids, ch. 9) · « Qu'est-ce qu'un prior conjugué ? » (le posterior reste dans la même famille : Beta → Beta) · « Et si on n'a vraiment aucune idée ? » (un prior faiblement informatif, plutôt que parfaitement plat sur un domaine immense).

### 4.E4 — Où utiliser des méthodes bayésiennes en data science ?

**Réponse modèle en 60 secondes** : « Deux exemples bayésiens. Un test A/B : avec une loi Beta par variante, je peux dire au métier « la variante B a 93 % de chances d'être meilleure que A », une phrase qu'il comprend, et l'on peut mettre à jour au fil de l'eau. Le réglage d'hyperparamètres : Optuna, par défaut, utilise TPE, une optimisation bayésienne qui choisit le prochain essai d'après les précédents ; c'est bien plus économe qu'une grille. On peut ajouter Naive Bayes comme baseline très rapide pour du texte. Pour la méthode fréquentiste : pour donner l'incertitude d'une métrique de modèle (une AUC sur un jeu de test), je prends un intervalle bootstrap. C'est simple, sans prior à justifier, et c'est le standard dans les rapports. Même chose pour un test A/B réglementé, avec un plan d'expérience fixé d'avance et un risque d'erreur contrôlé. »
**Relances possibles** : « Le piège des tests A/B fréquentistes ? » (regarder les résultats tous les jours et s'arrêter dès que c'est significatif fausse le risque d'erreur : il faut un plan fixé ou une méthode séquentielle) · « Les probabilités de Naive Bayes sont-elles fiables ? » (non, la documentation de scikit-learn le dit : bon classifieur, mauvais estimateur de probabilités ; recalibrer si besoin) · « Et en deep learning ? » (les VAE, ch. 25, l'incertitude par ensembles de modèles).

<a id="notebook"></a>

## Notebook, parties A à D

Les réponses ci-dessous sont celles de `05_solutions.ipynb` (exécuté). La référence complète de `mylearn.bayes` est dans `solutions/mylearn_ref/bayes.py` : lis-la **après** avoir réussi les tests. Des méthodes différentes des corrigés sont acceptées tant que les valeurs sont les mêmes ; quand un tirage aléatoire intervient, l'énoncé impose la méthode.

### Ex 4.12 — Combien de lancers pour démasquer la pièce truquée ? 🔮
Mesuré sur 1 000 essais : avec la pièce de biais 0,75, une médiane de **19** lancers (16 quand on lance la pièce équilibrée) et **35** annonces fausses ; avec le biais 0,55, une médiane de **429** lancers, et 43 annonces fausses.
**Pourquoi** : chaque lancer apporte d'autant moins d'information que les deux pièces se ressemblent. L'écart au biais 0,5 passe de 0,25 à 0,05, cinq fois moins, et il faut environ 25 fois plus de lancers : l'information d'un lancer est à peu près proportionnelle au **carré** de l'écart. Quant aux erreurs : s'arrêter quand une hypothèse dépasse 0,95, c'est accepter qu'elle soit fausse dans environ 5 % des cas, et on observe bien de 3,5 % à 6 % d'annonces fausses selon les cas. Le posterior est honnête : il est calibré (ch. 3).
**Erreurs fréquentes** : prévoir une proportion (le nombre de lancers croît bien plus vite quand les biais se rapprochent) ; croire que la règle ne se trompe jamais.
**Variante** : refais l'expérience avec un seuil de 0,99 : combien de lancers en plus ? combien d'erreurs en moins ? (Avec la même graine : une médiane de 31 lancers au lieu de 19 pour le biais 0,75 (28 au lieu de 16 avec la pièce équilibrée), et de 747 au lieu de 429 pour le biais 0,55, soit entre 1,6 et 1,75 fois plus de lancers : la log-cote doit atteindre $\ln 99 \approx 4{,}6$ au lieu de $\ln 19 \approx 2{,}9$. Les annonces fausses, entre 35 et 58 sur 1 000 au seuil de 0,95, ne sont plus que 8 à 11, environ 1 % : ce que promet le seuil.)

### Ex 4.13 — L'estimation fréquentiste : la moyenne courante des faces 🔬
a) **[0,20 ; 0,55 ; 0,85]** · b) **0,062** · c) $\sqrt{0{,}8 \times 0{,}2 / 100} = $ **0,040**.
**Démarche** :
```python
def running_estimate(flips):
    flips = np.asarray(flips)
    return np.cumsum(flips) / np.arange(1, len(flips) + 1)    # heads so far / number of flips


max_error_13 = float(np.max(np.abs(running_estimate(flips_13[0.8])[49:] - 0.8)))   # flips 50 to 100
se_13 = float(np.sqrt(0.8 * 0.2 / 100))
```
**Pourquoi** : la $k$-ième estimation est la proportion de faces parmi les $k$ premiers lancers. Entre le 50ᵉ et le 100ᵉ lancer, l'estimation de la pièce de biais 0,8 s'écarte jusqu'à 0,062 (au 58ᵉ lancer, elle vaut 0,862), et finit à 0,85 : c'est de l'ordre de l'erreur typique, 0,04. Rien d'anormal : environ une fois sur trois, une estimation tombe à plus d'une erreur typique de la vraie valeur. Pour diviser l'erreur typique par 10, il faut 100 fois plus de lancers (10 000).
**Erreurs fréquentes** : diviser par `np.arange(len(flips))` (qui commence à 0 : division par zéro au premier lancer) ; prendre les indices 50 à 99 (le 50ᵉ lancer est l'indice 49) ; donner la variance (0,0016) au lieu de l'écart-type.
**Variante** : refais la figure avec 10 000 lancers par pièce, en échelle logarithmique en abscisse : l'écart à la vraie valeur se resserre comme $\frac{1}{\sqrt{n}}$.

### Ex 4.14 — evidence et bayes_posterior 🔨
a) **0,45** · b) **[0,5556 ; 0,3889 ; 0,0556]**, puis les 33 tests passent.
**Démarche** : deux aides, l'une pour une distribution, l'autre pour des probabilités, puis deux fonctions de quelques lignes.
```python
def _check_distribution(p, name):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if np.isnan(p).any() or (p < 0).any() or abs(p.sum() - 1) > 1e-8:   # NaN tested apart: nan < 0 is False
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
    if likelihood.shape != prior.shape:                              # no broadcasting
        raise ValueError("prior and likelihood must have the same shape")
    return float(np.sum(prior * likelihood))


def bayes_posterior(prior, likelihood):
    total = evidence(prior, likelihood)                              # also checks both inputs
    if total == 0:
        raise ValueError("the evidence is 0: the observation is impossible")
    return np.asarray(prior, dtype=float) * np.asarray(likelihood, dtype=float) / total   # a new array
```
**Pourquoi** : $P(\text{pile}) = 0{,}5 \times 0{,}5 + 0{,}25 \times 0{,}7 + 0{,}25 \times 0{,}1 = 0{,}45$ ; le posterior vaut $\frac{(0{,}25 ;\ 0{,}175 ;\ 0{,}025)}{0{,}45}$. La pièce de biais 0,9, qui donne rarement pile, tombe de 0,25 à 0,056. `bayes_posterior` réutilise `evidence` : la validation n'est écrite qu'une fois. Les tests vérifient aussi qu'un prior nul reste nul, qu'une vraisemblance uniforme ne change rien, et que les tableaux reçus ne sont pas modifiés. Ce dernier point est un piège classique : `prior *= likelihood` modifie le tableau de l'appelant quand `np.asarray` ne le copie pas.
**Erreurs fréquentes** : exiger que les vraisemblances somment à 1 ; refuser un prior comme `np.array([0.6, 0.3, 0.1])`, dont la somme vaut 0,9999999999999999 en flottants (d'où la tolérance de `1e-8`) ; oublier les NaN (`nan < 0` vaut `False`, et une somme qui contient un NaN échappe à la comparaison avec 1) ; laisser le broadcasting de NumPy étirer une vraisemblance de longueur 1 ; renvoyer `nan` au lieu de lever une `ValueError` quand l'évidence est nulle.
**Variante** : écris `odds_update(prior_odds, likelihood_ratio)` pour deux hypothèses (fiche, 🧮 cotes), et vérifie qu'elle redonne `bayes_posterior` sur l'exercice 4.1.

### Ex 4.15 — Bayes chez les manchots : l'espèce sachant l'île 📦
a) **[0,442 ; 0,198 ; 0,360]** · b) **[0,368 ; 1,000 ; 0,000]** · c) **[0,452 ; 0,548 ; 0]** · d) $\frac{124}{344} \approx $ **0,3605** · e) **[0,596 ; 0,404 ; 0]**.
**Démarche** :
```python
counts = pd.crosstab(penguins["species"], penguins["island"]).reindex(SPECIES)
prior_15 = (counts.sum(axis=1) / counts.to_numpy().sum()).to_numpy()                  # 152, 68, 124 out of 344
likelihood_15 = pd.crosstab(penguins["species"], penguins["island"], normalize="index").reindex(SPECIES)["Dream"].to_numpy()
other_prior_15 = np.array([0.60, 0.15, 0.25])
```
**Pourquoi** : $P(\text{Dream} \mid \text{Adélie}) = \frac{56}{152}$, $P(\text{Dream} \mid \text{Chinstrap}) = \frac{68}{68} = 1$ et $P(\text{Dream} \mid \text{Gentoo}) = 0$. Bayes redonne exactement la colonne Dream de la table normalisée par colonne : $\frac{56}{124}$ et $\frac{68}{124}$. Rien de neuf, mais les deux ingrédients sont séparés : avec le prior de l'autre équipe, la part des Adélie sur Dream passe à 0,596, sans rien changer aux vraisemblances.
**Erreurs fréquentes** : arrondir a) et b) avant de les réutiliser (c à e sont vérifiés à 4 décimales) ; laisser l'ordre de `value_counts` (Adelie, Gentoo, Chinstrap, par effectif décroissant) ; normaliser par colonne en b (c'est déjà le posterior) ; diviser par les 344 manchots en b (la probabilité jointe).
**Pour le ML** : ce raisonnement suppose que $P(\text{île} \mid \text{espèce})$ reste vrai dans la nouvelle population. C'est faux si l'échantillonnage a privilégié certaines îles pour certaines espèces : les données de Palmer décrivent où l'équipe a compté, pas forcément où vivent les manchots. Le même raisonnement sert à corriger un classifieur quand la proportion des classes change entre l'entraînement et le déploiement (*prior shift*).
**Variante** : refais le calcul avec l'observation « nageoire de plus de 205 mm » à la place de l'île.

### Ex 4.16 — La boucle posterior-prior : update_discrete 🔨
a) **[0,4495 ; 0,5461 ; 0,0044]** · b) **[0,4608 ; 0,5184 ; 0,0207]** · c) **1**, puis les 21 tests passent.
**Démarche** : pour chaque observation, la colonne `likelihoods[:, o]` donne les vraisemblances, et `bayes_posterior` (4.14) fait le reste ; on garde chaque posterior dans une liste si `return_history`.
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
        if not np.isfinite(o) or o != int(o) or not 0 <= o < table.shape[1]:   # 1.5 and -1 are refused
            raise ValueError(f"{o} is not an outcome index")
        posterior = bayes_posterior(posterior, table[:, int(o)])     # it becomes the next prior
        history.append(posterior)
    return np.array(history) if return_history else posterior
```
**Pourquoi** : trois 6 en cinq lancers font passer le dé pipé sur le 6 de 0,1 à 0,55, malgré le prior de 0,8 du dé équilibré ; le dé pipé sur le 1, qui n'a donné aucun 1, tombe sous 1 %. Le rapport de vraisemblance d'un 6 entre le dé pipé et le dé équilibré vaut $\frac{0{,}5}{1/6} = 3$ ; celui d'une autre face vaut $\frac{0{,}1}{1/6} = 0{,}6$. Trois 6 et deux autres faces multiplient la cote « pipé sur le 6 contre équilibré » par $3^3 \times 0{,}6^2 \approx 9{,}7$ : plus que le prior de 8 contre 1 ne la divisait. La cote finale vaut environ 1,2 en faveur du dé pipé (0,546 contre 0,450). La même fonction resservira avec des sondes (4.20) et 501 hypothèses (4.21).
**Erreurs fréquentes** : utiliser les faces comme indices (6 n'existe pas : la face $k$ est l'indice $k - 1$) ; prendre une **ligne** du tableau au lieu d'une colonne ; ajouter à l'historique le même tableau modifié en place (toutes les lignes deviennent identiques) ; laisser passer l'indice −1, que NumPy lit comme la dernière colonne.
**Variante** : ajoute un quatrième dé, pipé sur le 2, de prior 0,05 (et 0,75 pour l'équilibré) : que devient le posterior après les mêmes lancers ? (Environ [0,429 ; 0,556 ; 0,004 ; 0,011] : le dé pipé sur le 6 reste en tête, et le nouveau dé tombe de 0,05 à 0,011. Un seul 2 ne compense pas les trois 6, qu'il ne donne qu'une fois sur dix.)

### Ex 4.17 — Reproduire les trente lancers de la figure 4.24 🎨
Les historiques sont vérifiés (forme `(31, 2)`, colonne 0 = $P(\text{équilibrée})$) ; la figure se juge à l'œil. Dans la suite à 3 faces, la pièce truquée dépasse 0,9 au **8ᵉ** lancer.
**Démarche** :
```python
def stacked_history_17(sequence):
    flips = np.array([1 if c == "F" else 0 for c in sequence])
    return mylearn.bayes.update_discrete([0.5, 0.5], COINS_17, flips, return_history=True)


def draw_stacked_17(ax, sequence):
    history = stacked_history_17(sequence)
    x = np.arange(len(history))
    ax.bar(x, history[:, 0], color="C0", label="P(fair)")
    ax.bar(x, history[:, 1], bottom=history[:, 0], color="C1", label="P(rigged)")   # stacked on top
    ax.set_xticks(x)
    ax.set_xticklabels(["before"] + list(sequence))
```
**Pourquoi** : chaque pile multiplie la cote « truquée contre équilibrée » par $\frac{0{,}8}{0{,}5} = 1{,}6$, chaque face la divise par $\frac{0{,}5}{0{,}2} = 2{,}5$. Dans la suite à 3 faces, quatre piles la portent à 0,87, la face du 5ᵉ lancer la fait redescendre à 0,72, et il faut trois piles de plus pour dépasser 0,9. Avec 6 faces, la pièce truquée l'emporte aussi ($P(\text{équilibrée}) \approx 0{,}003$ à la fin). Avec 24 faces, c'est la pièce équilibrée, bien que 24 faces sur 30 soient aussi très improbables pour elle : Bayes compare les hypothèses entre elles.
**Erreurs fréquentes** : tracer les deux séries côte à côte au lieu de les empiler (oubli de `bottom=`) ; oublier la barre « avant » (le prior) ; inverser les colonnes du tableau `COINS_17`.
**Variante** : ajoute une ligne horizontale à 0,95 et marque le premier lancer où une hypothèse la dépasse : c'est la règle d'arrêt de 4.12.

### Ex 4.18 — Le posterior qui s'évanouit : underflow 🐛
a) **1075** · b) **1099** · c) **[0,0785 ; 0,8586 ; 0,0629]** · d) **−999,687**.
**Démarche** :
```python
n_zero_18 = 1
while 0.5 ** n_zero_18 != 0.0:
    n_zero_18 += 1

products = np.cumprod(np.where(flips_18[:, None] == 1, biases_18, 1 - biases_18), axis=0)   # row k - 1: k flips
joints = prior_18 * products                              # the `joint` of the colleague's function
first_nan_18 = int(np.argmax((joints == 0).all(axis=1))) + 1


def posterior_fixed_18(prior, biases, flips):
    prior, biases, flips = np.asarray(prior, dtype=float), np.asarray(biases, dtype=float), np.asarray(flips)
    heads = int(flips.sum())
    tails = len(flips) - heads
    log_post = np.log(prior) + heads * np.log(biases) + tails * np.log(1 - biases)
    weights = np.exp(log_post - log_post.max())       # the largest becomes exp(0) = 1
    return weights / weights.sum()


lse_18 = -1000 + np.log(1 + np.exp(-1.0))             # max + log(sum of exp(l - max))
```
**Diagnostic** : `0.5 ** 1074` vaut encore environ $5 \times 10^{-324}$ (un nombre dénormalisé) ; à 1 075, le résultat est arrondi à 0. Le produit des vraisemblances de chaque hypothèse décroît d'environ un facteur $e^{0{,}67}$ par lancer, et la fonction le multiplie encore par le prior $\frac{1}{3}$ : $5 \times 10^{-324} \times \frac{1}{3}$ est arrondi à 0. À partir de 1 099 lancers, les trois produits prior × vraisemblances (le tableau `joint`) valent 0, et la normalisation calcule $\frac{0}{0}$ ; les produits de vraisemblances seuls ne s'annulent tous qu'à 1 103 lancers. Aucune erreur au moment de l'underflow : une multiplication qui donne 0 n'est pas une erreur, et seul le `nan` final trahit le bug, bien plus tard.
**Correction** : en logarithmes, les trois log-vraisemblances valent entre −2 022 et −2 019, sans problème ; après soustraction du maximum, on revient à des rapports raisonnables. Après 3 000 lancers (1 799 faces), le vrai biais 0,60 l'emporte (0,86), mais ses voisins 0,58 et 0,62 gardent 6 à 8 %. d : $e^{-1000}$ vaut 0 en `float64`, donc `np.log(np.exp(-1000) + np.exp(-1001))` donne $-\infty$ ; avec le maximum, $-1\,000 + \ln(1 + e^{-1}) \approx -999{,}687$.
**Erreurs fréquentes** : répondre 1 023 en a (le seuil des nombres normaux, `np.finfo(float).tiny` ≈ $2{,}2 \times 10^{-308}$, n'est pas celui de 0) ; en b, s'arrêter à la première hypothèse nulle au lieu d'attendre les trois (1 095), ou oublier le prior que la fonction multiplie encore (1 103) ; les deux erreurs à la fois redonnent 1 099, par pure coïncidence (le produit de 0,62 seul s'annule à 1 099) : vérifie ton raisonnement, pas seulement le ✅ ; en c, `heads * np.log(biases)` avec un biais nul et zéro face donne `nan` (ici, les biais candidats ne sont pas nuls) ; oublier de soustraire le maximum (tout redevient 0).
**Variante** : refais le calcul de c avec `scipy.special.logsumexp` pour normaliser : `np.exp(log_post - special.logsumexp(log_post))`.

### Ex 4.19 — La grille biais × proportion de faces 🔬
a) la grille de `grid_19(40)` (voir le notebook ; par exemple 0,45 dans les deux cases où biais et proportion valent 0,45 ou 0,55) · b) **[80, 100]**.
**Démarche** :
```python
def grid_19(n):
    grid = np.zeros((10, 10))
    for i, proportion in enumerate(PROPORTIONS_19):
        heads = round(proportion * n)
        flips = np.array([1] * heads + [0] * (n - heads))
        for j, bias in enumerate(BIASES_19):
            table = np.array([[0.5, 0.5], [1 - bias, bias]])        # rows: fair, rigged; columns: tails, heads
            grid[i, j] = mylearn.bayes.update_discrete([0.5, 0.5], table, flips)[0]
    return grid
```
**Pourquoi** : avec 40 lancers, 20 cases restent indécises. Quatorze sont dans les deux colonnes centrales, où la pièce truquée a un biais de 0,45 ou 0,55 : presque une pièce équilibrée, que 40 lancers ne suffisent pas à distinguer, pour la plupart des proportions de faces. Les six autres sont voisines, avec un biais de 0,25 à 0,75 proche de la proportion observée. Avec 1 000 lancers, les 100 cases sont tranchées. Mais le coin en haut à gauche (95 % de faces, pièce truquée de biais 0,05) donne $P(\text{équilibrée}) \approx 1$ : Bayes y est certain que la pièce est équilibrée, alors qu'aucune des deux hypothèses ne peut produire 95 % de faces. Il choisit la moins mauvaise. Il faut toujours vérifier que le modèle lui-même tient la route.
**Erreurs fréquentes** : inverser lignes et colonnes (la grille n'est pas symétrique) ; compter les cases indécises au lieu des cases tranchées.
**Variante** : ajoute une troisième hypothèse, « une autre pièce de biais égal à la proportion observée » : que devient le coin en haut à gauche ? (Avec un prior de $\frac{1}{3}$ pour chaque pièce, la troisième, de biais 0,95, prend presque toute la probabilité : dès 40 lancers, $P(\text{équilibrée})$ tombe d'environ 1 à $2{,}6 \times 10^{-9}$. Bayes ne choisit que parmi les hypothèses qu'on lui propose : ajoutes-en une qui explique les données, et elle l'emporte.)

### Ex 4.20 — Envoyer des sondes jusqu'à la décision 🔬
a) **4,464** sondes en moyenne · b) **467** planètes protégées · c) **4** planètes stériles protégées par erreur.
**Démarche** :
```python
def explore_20(inhabited, rng, low=1e-6, high=0.99, max_probes=20):
    posterior = np.array([PRIOR_20, 1 - PRIOR_20])
    for probes in range(1, max_probes + 1):
        detected = rng.random() < (SENS_20 if inhabited else 1 - SPEC_20)   # exactly one draw per probe
        posterior = mylearn.bayes.update_discrete(posterior, TABLE_20, [int(detected)])
        if posterior[0] < low:
            return "mine", probes
        if posterior[0] > high:
            return "protect", probes
    return "undecided", max_probes
```
**Pourquoi** : une planète stérile demande le plus souvent quatre sondes négatives de suite (4.8) ; une planète habitée, trois positives. Une seule sonde contraire allonge l'exploration, d'où un maximum de 13 sondes. Sur 10 000 planètes, 463 sont habitées : toutes sont protégées, et 4 planètes stériles le sont aussi. Aucune planète habitée n'est exploitée, et c'était prévisible : par construction, parmi les planètes exploitées, la part de planètes habitées reste sous un millionième, soit moins d'une erreur attendue sur 9 533 planètes. Mais cette garantie suppose des sondes **indépendantes sachant l'état** de la planète : si les sondes ratent toutes la même vie cachée, leurs erreurs sont liées, et la garantie s'effondre (4.8 g).
**Erreurs fréquentes** : tirer plusieurs nombres par sonde, ou recréer le générateur dans `explore_20` (les nombres ne sont plus ceux de la vérification) ; repartir du prior à chaque sonde ; compter les sondes à partir de 0.
**Variante** : remplace le seuil d'exploitation $10^{-6}$ par $10^{-3}$ : combien de sondes économises-tu, et combien de planètes habitées exploites-tu ? (Avec la graine de la vérification : 2,302 sondes par planète en moyenne au lieu de 4,464 (deux sondes négatives suffisent désormais), et toujours aucune planète habitée exploitée. C'est de la chance : après deux sondes négatives, une planète reste habitée avec une probabilité d'environ $9{,}5 \times 10^{-5}$ (4.8 c), soit environ une planète habitée attendue parmi les quelque 9 500 planètes exploitées. Sur les graines 20 à 39, on en exploite de 0 à 3, une en moyenne.)

### Ex 4.21 — Un prior trompeur centré sur 0,8 🔮
a) **`True`** · b) **`1000`** · c) **`False`** · d) **0,422** · e) **815**.
**Pourquoi** : le prior ne donne que $3 \times 10^{-8}$ à l'hypothèse 0,3 (contre $2 \times 10^{-3}$ pour un prior uniforme), mais chaque lancer multiplie les rapports de vraisemblance. Après 100 lancers (33 faces), le MAP vaut 0,422 : plus près de 0,3 que de 0,8, mais encore tiré vers la bosse (avec un prior uniforme, il vaudrait 0,33). Il passe une première fois à moins de 0,015 de 0,3 vers 780 lancers, s'en éloigne encore un peu, puis y reste à partir de 815 lancers : un ordre de grandeur de 1 000. Ce nombre ne mesure pas seulement le coût du prior : à 1 000 lancers, l'erreur typique de la proportion de faces, $\sqrt{0{,}3 \times 0{,}7 / 1\,000} \approx 0{,}014$, est elle-même de l'ordre de la tolérance de 0,015, et un prior uniforme demanderait lui aussi de l'ordre de 1 000 lancers sur ces données (variante). Après 3 000 lancers, le MAP vaut 0,298 contre 0,296 pour un prior uniforme : le prior a presque disparu, mais pas tout à fait. Un prior nulle part nul finit toujours par céder ; un prior nul ne céderait jamais (∂ 4.7).
**Erreurs fréquentes** : en d, lire la ligne 101 (la ligne 0 est le prior) ; en e, donner le dernier nombre de lancers où le MAP est encore trop loin (814), ou la première fois où il s'approche (780).
**Variante** : refais l'expérience avec une bosse plus large (écart-type 0,3) puis plus étroite (0,05) : comment change e ? (Sur les mêmes lancers, e passe à 2 562 avec la bosse étroite, qui ne laisse qu'environ $3 \times 10^{-24}$ à l'hypothèse 0,3. Avec la bosse large, e passe à 1 342, ce qui surprend : presque sans prior, le MAP suit la proportion de faces de ces lancers, qui descend vers 0,285 autour de 1 300 lancers ; avec un prior uniforme, e vaudrait 1 357. Le 815 de 4.21 doit donc un peu au hasard : la pente de la bosse y compensait cette baisse passagère.)

### Ex 4.22 — Le posterior continu : vérifier avec scipy.stats.beta 📦
a) **[14, 8]** · b) **0,636** · c) **0,905** · d) **[0,430 ; 0,819]**.
**Démarche** :
```python
posterior_22 = stats.beta(HEADS_22 + 1, TAILS_22 + 1)       # uniform prior: Beta(h + 1, t + 1)
mean_22 = posterior_22.mean()                                # 14 / 22
p_above_half_22 = posterior_22.sf(0.5)                       # 1 - cdf(0.5)
interval_22 = list(posterior_22.interval(0.95))              # equal tails: ppf(0.025), ppf(0.975)
```
**Pourquoi** : le prior uniforme est $\mathrm{Beta}(1, 1)$ ; chaque face ajoute 1 à $a$ et chaque pile 1 à $b$. La moyenne $\frac{14}{22}$ est tirée vers 0,5 par rapport au mode $\frac{13}{20} = 0{,}65$ : le prior uniforme compte comme une face et une pile « fictives » (règle de succession de Laplace). La grille de 1 001 points donne exactement la densité Beta normalisée, à $10^{-17}$ près : la loi Beta est la limite continue des figures du livre.
**Erreurs fréquentes** : prendre $\mathrm{Beta}(13, 7)$ (sans le prior) ; utiliser `cdf(0.5)`, qui donne la probabilité à gauche (0,095) ; prendre `pdf(0.5)` pour une probabilité.
**Variante** : avec un prior $\mathrm{Beta}(10, 10)$ (« la pièce est probablement presque équilibrée »), le posterior est $\mathrm{Beta}(23, 17)$ : compare sa moyenne et son intervalle à ceux de b et d. (Moyenne $\frac{23}{40} = 0{,}575$ au lieu de 0,636 ; intervalle d'environ 0,421 à 0,722 au lieu de 0,430 à 0,819 : plus étroit, et tiré vers 0,5. Ce prior compte comme 9 faces et 9 piles fictives de plus que le prior uniforme.)

### Ex 4.23 — Refactoriser : du copier-coller à une fonction testée 🛠️
a) **`"Torgersen"`**.
**Diagnostic** : le bloc de Torgersen divise par `len(biscoe)` au lieu de `len(torgersen)` : les « probabilités » de Torgersen somment à 0,31 (52 Adélie divisés par 168). C'est le risque du copier-coller : la troisième copie a gardé un morceau de la première. Rien ne plante, et chaque nombre pris seul a l'air plausible. Un contrôle d'une ligne, sur les résultats affichés, l'aurait révélé : des probabilités conditionnelles doivent sommer à 1.
**Démarche** :
```python
def species_given(df, column, value):
    """Probability of each species among the rows where ``df[column] == value``.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per animal, with a ``"species"`` column and the column ``column``.
    column : str
        The column that defines the group (for example ``"island"``).
    value : object
        The value of ``column`` that selects the group (for example ``"Dream"``).

    Returns
    -------
    pandas.Series
        P(species | df[column] == value), indexed by every species of ``df`` in
        alphabetical order (0 for a species absent from the group); it sums to 1.

    Examples
    --------
    >>> df = pd.DataFrame({"species": ["A", "A", "B", "C"], "island": ["x", "x", "x", "y"]})
    >>> species_given(df, "island", "x").round(3).tolist()
    [0.667, 0.333, 0.0]
    """
    chosen = df.loc[df[column] == value, "species"]
    return chosen.value_counts(normalize=True).reindex(sorted(df["species"].unique()), fill_value=0.0)


def test_probabilities_sum_to_one():
    df = pd.DataFrame({"species": ["A", "B", "B", "C", "A"], "island": ["x", "x", "y", "y", "y"]})
    for island in ["x", "y"]:
        assert species_given(df, "island", island).sum() == pytest.approx(1)


def test_absent_species_has_probability_zero():
    df = pd.DataFrame({"species": ["A", "A", "B"], "island": ["x", "x", "y"]})
    result = species_given(df, "island", "x")
    assert list(result.index) == ["A", "B"]
    assert result["B"] == 0


def test_hand_computed_case():
    df = pd.DataFrame({"species": ["A", "A", "A", "B", "B", "C"], "island": ["x", "x", "y", "x", "y", "y"]})
    assert species_given(df, "island", "x").tolist() == pytest.approx([2 / 3, 1 / 3, 0])


my_tests_23 = [test_probabilities_sum_to_one, test_absent_species_has_probability_zero, test_hand_computed_case]
```
**Pourquoi** : une seule fonction, le calcul n'est écrit qu'une fois, et il se teste. Chaque test vise un bug : la somme à 1 attrape le bug 1 (division par la taille de toute la table), l'espèce absente attrape le bug 2 (espèces manquantes dans l'index), et le cas calculé à la main attrape le bug 3 (la condition inversée, $P(\text{île} \mid \text{espèce})$). Des petits DataFrames écrits à la main rendent les tests rapides, lisibles et indépendants des données réelles.
**Erreurs fréquentes** : tester seulement sur les 344 manchots (un test lent, qui casse si les données changent) ; oublier `reindex` (une espèce absente disparaît de l'index) ; un test qui ne vérifie que le type du résultat (il passe sur les trois versions buggées).
**Variante** : généralise : `share_given(df, target, column, value)`, pour n'importe quelle colonne cible (le sexe sachant l'île, par exemple), et adapte tes tests.

### Ex 4.24 — coin_bias_posterior : 500 hypothèses en log-probabilités 🔨
a) **0,37** · b) **0,0034**, puis les 25 tests passent.
**Démarche** : compter $h$ et $t$, additionner $\log P(\theta)$, $h\log\theta$ si $h > 0$ et $t\log(1 - \theta)$ si $t > 0$, retrancher le maximum, prendre l'exponentielle et normaliser ; les deux aides viennent de 4.14.
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
**Pourquoi** : 7 396 faces sur 20 000 lancers (0,3698) : le MAP est la valeur de la grille la plus proche, 0,37. L'écart-type du posterior, 0,0034, est presque exactement l'erreur typique fréquentiste $\sqrt{0{,}37 \times 0{,}63 / 20\,000} \approx 0{,}0034$ : avec beaucoup de données et un prior plat, les deux écoles donnent les mêmes nombres. Le calcul en un passage est des centaines de fois plus rapide que la boucle (de l'ordre d'une seconde contre quelques millisecondes ici) : la boucle sert quand les observations arrivent une par une. Les tests vérifient les extrémités de la grille : avec $\theta = 0$ et aucune face, le terme $\theta^0 = 1$ ne doit pas donner `nan`.
**Erreurs fréquentes** : `h * np.log(grid)` sans précaution quand $h = 0$ et que la grille contient 0 (`0 * -inf = nan`) ; oublier de soustraire le maximum (tout vaut 0 après 20 000 lancers) ; ignorer le prior fourni ; ne pas lever d'erreur quand toutes les hypothèses sont éliminées.
**Variante** : ajoute un paramètre `log_prior` qui accepte directement des log-probabilités (utile quand le prior lui-même est minuscule, comme en 4.21).

### Ex 4.25 — Intervalle de crédibilité contre intervalle bootstrap 🔨
a) **(0,293 ; 0,558)** · b) **(0,28 ; 0,56)** · c) **(0,002 ; 0,285)**, puis les 20 tests passent ; couverture sur 200 échantillons de 10 lancers d'une pièce de biais 0,1 : environ **0,91** pour l'intervalle de crédibilité, **0,63** pour le bootstrap.
**Démarche** : `np.cumsum`, puis `np.searchsorted(cdf, level)`, qui renvoie le premier indice où le cumul atteint le niveau.
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
**Pourquoi** : sur 50 lancers (21 faces), les deux intervalles se ressemblent : avec assez de données, le bootstrap et un prior plat racontent la même histoire. Sur dix piles, le bootstrap ne rééchantillonne que des piles et répond (0 ; 0) : « le biais vaut 0, sans aucun doute ». L'intervalle de crédibilité va de presque 0 à environ 0,285 : dix lancers ne permettent pas d'exclure un biais de 0,2. Sur les petits échantillons, 73 sur 200 n'ont aucune face ; pour eux, l'intervalle bootstrap (0 ; 0) ne contient jamais 0,1. Sa couverture tombe à 63 %, loin des 95 % promis. L'intervalle de crédibilité, lui, couvre le vrai biais 91 % du temps. Ce n'est pas exactement 95 % : une promesse bayésienne porte sur le paramètre sachant les données, pas sur des répétitions, mais elle se tient bien mieux ici.
**Erreurs fréquentes** : utiliser `>` au lieu de `>=` (un niveau atteint exactement doit compter) ; renvoyer des indices au lieu de valeurs de la grille ; interpoler entre deux valeurs de la grille ; renvoyer une liste au lieu d'un tuple.
**Variante** : avec une grille de 101 points au lieu de 1 001, de combien bougent les bornes de a ? (Au plus d'un pas, 0,01.)

### Ex 4.26 — Le détective de pièces : vingt pièces, le moins de lancers possible 🏆
La stratégie de base de l'énoncé (règle d'arrêt « posterior > 0,99 », pièce après pièce) identifie les 20 pièces du sac de la graine 434, mais en **2 446** lancers : au-delà du budget de 2 400. La solution ci-dessous répartit le budget : **20 pièces sur 20** en 2 400 lancers sur ce sac, et l'objectif tenu sur les 20 autres sacs (il en faut au moins 15).
**Démarche** :
```python
def detective_26(bag, threshold=0.99, warm_up=5):
    table = np.column_stack([1 - CANDIDATES_26, CANDIDATES_26])        # columns: P(tails), P(heads)
    uniform = np.full(len(CANDIDATES_26), 1 / len(CANDIDATES_26))
    # a few flips of every coin first, then always the coin we are least sure about
    posteriors = [mylearn.bayes.update_discrete(uniform, table, [bag.flip(i) for _ in range(warm_up)])
                  for i in range(20)]
    sure = np.array([posterior.max() for posterior in posteriors])
    while bag.flips_used < BUDGET_26 and sure.min() <= threshold:
        i = int(np.argmin(sure))
        posteriors[i] = mylearn.bayes.update_discrete(posteriors[i], table, [bag.flip(i)])
        sure[i] = posteriors[i].max()
    return [float(CANDIDATES_26[np.argmax(posterior)]) for posterior in posteriors]
```
**Pourquoi** : les pièces difficiles sont celles des candidats intérieurs (0,35, 0,5 et 0,65), encadrés par un voisin de chaque côté. Pour dépasser 0,99, il leur faut en médiane de 120 à 140 lancers, contre 70 environ pour 0,2 et 0,8 (médianes mesurées sur les 200 sacs de graines 5000 à 5199, qui servent aussi aux taux ci-dessous). La stratégie de base finit chaque pièce avant de passer à la suivante : sur un sac où plusieurs pièces sont difficiles, elle épuise le budget, et elle ne réussit qu'un peu plus d'un sac sur trois. Baisser le seuil coûte moins de lancers mais plus d'erreurs. À 0,98, elle passe le défi (20 sur 20 en 2 137 lancers sur le sac 434, l'objectif tenu sur 19 des 20 autres sacs), mais ne réussit qu'environ 4 sacs sur 5 ; à 0,97, 19 sur 20 en 1 917 lancers sur le sac 434, et 9 sacs sur 10. S'arrêter simplement quand le budget est épuisé passe le sac 434, mais ne tient l'objectif que sur 14 des 20 autres sacs : le défi est manqué. Répartir le budget sur la pièce la plus incertaine réussit environ 95 % des 200 sacs : les lancers vont là où l'incertitude reste. C'est la logique des tests séquentiels et des tests A/B bayésiens.
**Erreurs fréquentes** : relancer une pièce déjà identifiée ; repartir d'un prior uniforme à chaque lancer (le posterior doit devenir le prior) ; annoncer le plus grand posterior au lieu de la valeur du biais qui l'atteint ; dépasser le budget sans le vérifier (`bag.flips_used`) ; regarder `bag._biases`.
**Variante** : fais varier le seuil et le nombre de lancers d'échauffement, et mesure le taux de réussite sur les sacs de graines 5000 à 5199 : jusqu'où peux-tu baisser le budget en gardant 19 pièces sur 20 dans 9 sacs sur 10 ?
