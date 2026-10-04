# 6 · Théorie de l'information — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 🧮 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 6.Q1 — Information : sens courant, sens de Shannon

<details><summary>Indice 1</summary>

Relis le §6.1 de la fiche, et surtout « Information : un mot, deux sens ».

</details>
<details><summary>Indice 2</summary>

Pour la question 2, demande-toi ce qui distingue, pour Shannon, un message d'un autre : ce qu'il raconte, ou sa place parmi tous les messages possibles ? Pour la question 5, pense au contexte global (§6.2) : deux destinataires attendent-ils toujours les mêmes messages ?

</details>
<details><summary>Indice 3</summary>

1. Shannon voulait transmettre un message d'un point à un autre efficacement (avec peu de symboles) et fiablement, malgré le bruit du canal ; pour cela, il lui fallait d'abord savoir mesurer l'information. C'est le modèle de réponse. 2. Relis « Information : un mot, deux sens » (fiche §6.1) : que laisse de côté l'ingénieur, selon Shannon, et que regarde-t-il à la place ? 3. Applique ta réponse à 2. à ce message : pour un adulte, est-il attendu ou surprenant ? Conclus avec ce que la fiche (§6.1) dit d'un message attendu. 4. Le premier paragraphe du §6.1 de la fiche nomme plusieurs quantités du chapitre, chacune avec le chapitre où elle revient : choisis-en deux. 5. Reprends ta réponse à 2. : ce qu'elle mesure dépend-il du message seul, ou aussi de ce qu'attend celui qui le reçoit (le contexte global, §6.2) ? Pense à deux destinataires qui n'attendent pas les mêmes choses.

</details>

### 6.Q2 — Plus c'est rare, plus ça informe

<details><summary>Indice 1</summary>

Relis « La surprise » (§6.2 de la fiche), puis la formule du §6.4.

</details>
<details><summary>Indice 2</summary>

La surprise vaut $I = -\log_2 p$. Que vaut $\log_2 1$ ? Pour la question 5, compare $-\log_2 \frac{p}{2}$ et $-\log_2 p$ avec la règle $\log \frac{a}{b} = \log a - \log b$.

</details>
<details><summary>Indice 3</summary>

1. Au début d'un SMS, « Ornithorynque » est bien plus rare que « Bonjour » : sa probabilité est plus petite, donc sa surprise plus grande, et c'est lui qui apporte le plus d'information. C'est le modèle de réponse. 2. Un événement certain a la probabilité 1 : pose $I = -\log_2 1$, et cherche l'exposant $k$ tel que $2^k = 1$. 3. Écris $\frac{1}{16}$ comme une puissance de 2, puis applique $I = -\log_2 p$. 4. Deux personnes donneraient-elles la même note de 0 à 100 au même mot ? Relis ensuite le §6.4 de la fiche : sur quelle grandeur repose la formule qui remplace cette échelle ? 5. Écris $-\log_2 \frac{p}{2}$ avec la règle $\log \frac{a}{b} = \log a - \log b$, et compare-le à $-\log_2 p$.

</details>

### 6.Q3 — Contexte global, contexte local

<details><summary>Indice 1</summary>

Relis « Le contexte » (§6.2 de la fiche) et l'encadré ⚠️ « Une construction à oublier ».

</details>
<details><summary>Indice 2</summary>

Global : ce que l'émetteur et le récepteur partagent avant le message ; local : ce qui précède, dans le message lui-même. Pour la question 4, un mot surprenant est-il un mot fréquent ou un mot rare ? Que donnerait un tirage où il sortirait souvent ?

</details>
<details><summary>Indice 3</summary>

1. Le contexte global, c'est tout ce que l'émetteur et le récepteur partagent avant le message : la langue, la culture, le métier (« levain » est un mot banal pour une boulangère). Le contexte local, ce sont les mots qui précèdent, dans le message lui-même (après « Joyeux », « anniversaire » est presque certain). C'est le modèle de réponse. 2. Ce qui rend « sucre » probable se trouve-t-il dans la phrase elle-même, ou hors d'elle ? Classe-le avec les définitions de 1. 3. Cherche, au ch. 4, la quantité qui résume ce qu'on attend **avant** d'observer quoi que ce soit. 4. Avec une pmf proportionnelle à la surprise, quels mots sortiraient le plus souvent ? Un mot surprenant est-il fréquent ou rare ? Compare les textes tirés ainsi à une vraie langue. 5. Le critère : le contexte local n'aide que si un caractère dépend de ceux qui le précèdent. Applique-le aux caractères d'un numéro tiré au hasard, puis aux lettres d'une phrase en français (pense à celles qui suivent un « q »).

</details>

### 6.Q4 — Le bit est une unité

<details><summary>Indice 1</summary>

Relis le §6.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un chiffre binaire est un support ; le bit d'information mesure ce qu'on apprend en le lisant. Une case qui vaut toujours la même chose : quelle est la probabilité de ce qu'on y lit ? Pour la pièce à 99 %, compare la surprise de pile et celle de face, puis tiens compte de la fréquence de chacune.

</details>
<details><summary>Indice 3</summary>

1. Un chiffre binaire est un **support** : un 0 ou un 1 stocké dans un circuit. Un bit d'information est une **quantité** : ce qu'on apprend en lisant ce support. C'est le modèle de réponse. 2. Quelle est la probabilité de lire `00000000` dans cette case ? Applique $I = -\log_2 p$. 3. Pour la pièce équilibrée, $-\log_2 \frac{1}{2}$. Pour la pièce à 99 %, la moyenne des deux surprises pondérées par leurs fréquences, $-0{,}99 \log_2 0{,}99 - 0{,}01 \log_2 0{,}01$, à comparer à la pièce équilibrée ; pour le « pourquoi », compare la surprise de pile à celle de face, puis la fréquence de chacune. 4. Le §6.3 de la fiche nomme deux autres unités : avec quelle base de logarithme va chacune ? 5. Reprends la distinction de 1. : une mémoire de 1 000 chiffres binaires apporte-t-elle toujours 1 000 bits quand on la lit ? Pense au cas de 2.

</details>

### 6.Q5 — Les quatre propriétés de l'information

<details><summary>Indice 1</summary>

Relis la liste des quatre propriétés du §6.4 de la fiche, et son mini-exemple.

</details>
<details><summary>Indice 2</summary>

Calcule $-\log_2 1$, $-\log_2 \frac{1}{2}$, $-\log_2 \frac{1}{4}$, puis l'information de l'événement « les deux à la fois », de probabilité $\frac{1}{2} \times \frac{1}{16}$. Pour la question 4, demande-toi ce que le début de la phrase fait à la surprise du dernier mot.

</details>
<details><summary>Indice 3</summary>

1. Les quatre propriétés : un événement probable apporte peu d'information (un événement certain, aucune) ; un événement improbable en apporte beaucoup ; plus il est improbable, plus il en apporte ; les informations de deux événements indépendants s'additionnent. C'est le modèle de réponse. 2. Calcule $-\log_2 1$, $-\log_2 \frac{1}{2}$ et $-\log_2 \frac{1}{4}$, et vérifie qu'ils respectent les propriétés 1 à 3 ; pour l'événement double, compare $-\log_2\left(\frac{1}{2} \times \frac{1}{16}\right)$ à $-\log_2 \frac{1}{2} - \log_2 \frac{1}{16}$. 3. L'information de la paire vaut $-\log_2 P(\text{classeur}, \text{vert})$ : à quelle condition peut-on écrire cette probabilité comme un produit ? Et si tous les classeurs du bureau sont verts, que devient la surprise de « vert » après « classeur » ? 4. Compare la surprise de « parapluie » pris seul et celle de « parapluie » après « Il pleut, n'oublie pas ton… », puis raisonne de même sur chaque mot de la phrase. 5. Parmi les fonctions de 0B, laquelle vérifie $f(ab) = f(a) + f(b)$ ? Pour le signe, regarde le signe de $\log_2 p$ quand $0 < p \le 1$.

</details>

### 6.Q6 — Taille du vocabulaire et bits par mot

<details><summary>Indice 1</summary>

Relis le §6.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Avec $k$ chiffres binaires, on écrit $2^k$ numéros différents. Cherche le plus petit $k$ tel que $2^k \ge N$.

</details>
<details><summary>Indice 3</summary>

1. 2 mots : $2^1 = 2 \ge 2$, donc 1 bit par mot (`0` et `1`). C'est le modèle. Pour 3, 1 000, 1 024 et 1 025 mots, cherche de même le plus petit $k$ tel que $2^k \ge N$ : situe chaque nombre entre deux puissances de 2 consécutives ($2^1, 2^2, \ldots, 2^{10} = 1\,024, 2^{11}$), et regarde de près le cas où $N$ est exactement une puissance de 2. 2. Traduis « le plus petit entier $k$ tel que $2^k \ge N$ » avec un logarithme ; pour l'arrondi, demande-toi si un mot de code peut avoir un nombre non entier de chiffres. 3. Compare le nombre de mots différents d'un livre à celui d'un dictionnaire, et ce que ce nombre change au nombre de bits par mot (calcule-le pour 1 000 mots et pour 200 000). 4. Pour retrouver le mot à partir de son numéro, que doit avoir le récepteur sous les yeux ? Ce qui est partagé **avant** le message, quel contexte est-ce (§6.2) ? 5. Lequel des deux nombres peut ne pas être entier ? Un nombre de chiffres binaires le peut-il ?

</details>

### 6.Q7 — Morse, Vail et les codes adaptatifs

<details><summary>Indice 1</summary>

Relis le §6.6 de la fiche, l'encadré ⚠️ « Le Morse a besoin de ses silences » et l'encadré 🧮 sur les codes préfixes.

</details>
<details><summary>Indice 2</summary>

E et T sont les lettres les plus fréquentes de l'anglais. Pour la question 4, découpe `· · · −` en morceaux qui sont des lettres de la liste : le dernier morceau finit forcément par `−`, c'est donc T, A, U ou V. Pour la question 6, compare, pour un symbole rare, la longueur de son mot dans un code adaptatif préfixe (la figure du code de Huffman de la fiche) et dans un code fixe.

</details>
<details><summary>Indice 3</summary>

1. E et T sont les deux lettres les plus fréquentes de l'anglais, et un code adaptatif donne les mots les plus courts aux symboles les plus fréquents. C'est le modèle de réponse. 2. Relis le paragraphe du Morse au §6.6 de la fiche : quel métier Vail a-t-il observé, et qu'y a-t-il compté ? 3. Pour chaque sorte de code, regarde la longueur des mots de code : la même pour tous les symboles, ou liée à leur fréquence ? 4. Choisis d'abord la dernière lettre parmi T, A, U et V (elles seules finissent par `−`), puis découpe les points qui restent devant elle avec E, I et S ; chaque choix qui marche est une lecture. 5. Relis la définition d'un code préfixe dans l'encadré 🧮 du §6.6, puis applique-la aux mots de E, I et S, sans silences. 6. Un code adaptatif raccourcit-il chaque message, ou les messages en moyenne ? Pour trancher, compare, dans la figure du code de Huffman (§6.6), la longueur des mots des deux symboles les plus rares à celle d'un code fixe pour six symboles, puis imagine un message fait de ces deux symboles.

</details>

### 6.Q8 — Entropie nulle, entropie maximale

<details><summary>Indice 1</summary>

Relis le §6.7 de la fiche et l'encadré ⚠️ « L'entropie est une propriété de la distribution ».

</details>
<details><summary>Indice 2</summary>

L'entropie est la moyenne des surprises, chacune pondérée par sa probabilité. Elle est nulle quand aucune issue ne surprend, maximale quand on ne peut rien deviner. Pour la question 4, calcule les quatre entropies : elles sont toutes entre 0 et $\log_2 3$.

</details>
<details><summary>Indice 3</summary>

1. L'entropie est la surprise moyenne d'un tirage selon $p$ : $H(p) = -\sum_i p_i \log_2 p_i$ bits, avec la convention $0 \log_2 0 = 0$. C'est le modèle de réponse. 2. Elle est nulle quand aucune issue ne surprend : que doit valoir l'une des probabilités, et donc les autres ? Trouve deux objets, textes ou distributions qui ont cette forme. 3. Quelle distribution sur 16 issues ne permet de rien deviner ? Calcule son entropie : 16 termes égaux. 4. Calcule les quatre entropies, par exemple $H([0{,}7 ; 0{,}3]) = -0{,}7 \log_2 0{,}7 - 0{,}3 \log_2 0{,}3$ ; les trois autres (une issue certaine, deux issues équiprobables, trois issues équiprobables) se calculent de tête. Range-les ensuite. 5. Relis l'encadré ⚠️ du §6.7 sur l'entropie et la distribution : de quoi dépend $H(p)$, et qu'est-ce qui, en revanche, dépend du message ? 6. Relis la définition de l'espérance (ch. 2) : quelle variable aléatoire prend la valeur $-\log_2 p_i$, et avec quelle probabilité ?

</details>

### 6.Q9 — Entropie et « organisation » : attention au faux ami

<details><summary>Indice 1</summary>

Relis l'encadré ⚠️ « Entropie et organisation : un faux ami » du §6.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une source structurée est prévisible : sa surprise moyenne est-elle grande ou petite ? Pour des lettres tirées uniformément, quelle est l'entropie maximale sur 26 lettres ?

</details>
<details><summary>Indice 3</summary>

1. Une source structurée est prévisible : chaque symbole surprend peu, donc sa surprise moyenne, l'entropie, est faible. C'est le modèle de réponse. 2. Pour les lettres uniformes, pose l'entropie de la loi uniforme sur 26 issues, $\log_2 26$ : c'est le maximum possible sur 26 lettres. Les lettres de Holmes sont-elles équiprobables ? Applique ensuite le critère de 1. pour dire lequel des deux textes est le plus organisé. 3. Même critère pour le désordre : plus il y a d'issues également possibles, plus il y a de désordre ; y a-t-il alors plus ou moins d'entropie ? 4. Relis le début du §6.7 de la fiche : il donne trois lectures d'un même nombre. Compose ta phrase avec l'une d'elles, sans le mot « organisation ».

</details>

### 6.Q10 — Le mauvais code coûte plus cher

<details><summary>Indice 1</summary>

Relis « Mélanger les codes » et « Probabilités nulles et lissage de Laplace » (§6.8 de la fiche).

</details>
<details><summary>Indice 2</summary>

Dans $H(p, q) = -\sum_i p_i \log_2 q_i$, les fréquences viennent de $p$, les longueurs des mots de code de $q$. Pour la question 2, demande-toi si un code fait pour d'autres fréquences peut battre, en moyenne, celui qui est fait pour les vraies. Pour la question 3, un taux compare deux nombres de bits pour le même message : obtenus avec quels codes ?

</details>
<details><summary>Indice 3</summary>

1. $H(p, q) = -\sum_i p_i \log_2 q_i$ est le nombre moyen de bits par symbole quand les données suivent $p$ et que le code est idéal pour $q$ : $p$ décrit les fréquences réelles des données, $q$ celles qu'a prévues le code (ou le modèle). C'est le modèle de réponse. 2. Écris $H(p, q) - H(p)$ : quelle quantité du §6.9 est-ce ? Ce que ∂ 6.7 démontre sur son signe, et sur le cas où elle s'annule, répond aux deux questions. 3. Relis la définition du taux au §6.8 de la fiche (« Mélanger les codes ») : quel nombre de bits est au numérateur ? Plus on comprime, que devient-il ? Et un taux égal à 1, que dirait-il ? 4. Calcule le coût de ce mot avec un code idéal pour $q$, $-\log_2 q_i$, quand $q_i = 0$ ; pour le contournement, relis « Deux codes adaptatifs » (§6.8) : qu'ajoute le livre à chaque code ? 5. Pour un exemple de classe $y$, quelle distribution est connue d'avance (la bonne réponse), et laquelle sort du modèle ?

</details>

### 6.Q11 — KL : positive, asymétrique, nulle quand… ?

<details><summary>Indice 1</summary>

Relis le §6.9 de la fiche, son encadré ⚠️ « L'ordre des arguments » et le paragraphe sur Jensen-Shannon.

</details>
<details><summary>Indice 2</summary>

$\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p)$. Chaque écart entre $p_i$ et $q_i$ est pondéré par la fréquence dans la **première** distribution. Une distance doit être symétrique.

</details>
<details><summary>Indice 3</summary>

1. $\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p)$ : le surcoût, en bits par symbole, d'un code fait pour $q$ quand les données suivent $p$. C'est le modèle de réponse. 2. Relis ∂ 6.7, question 5 : ce qui y est démontré sur le signe de la KL, et le cas d'égalité, répondent aux deux questions. 3. Dans $\sum_i p_i \log_2 \frac{p_i}{q_i}$, chaque écart est pondéré par $p_i$. Suis un mot fréquent dans un livre A et rare dans un livre B : combien pèse-t-il quand on envoie A avec le code de B, et quand on envoie B avec le code de A ? Compare aussi les deux chiffres que trouve le livre (fin du mini-exemple du §6.9 de la fiche). 4. Écris la propriété qu'on exige de toute distance, puis confronte-la à ta réponse à 3. 5. Écris cet envoi comme une cross-entropy $H(p, q)$, avec la convention de la fiche (« Lire les formules ») : $p$ décrit les données, $q$ le code. Quel livre prend la place de $p$ ? Garde le même ordre pour la KL, puis compare aux deux notations du livre. 6. Pour obtenir une divergence symétrique, on compare $p$ et $q$ à une même troisième distribution, construite à partir des deux : laquelle ? Cherche le nom de cette divergence au §6.9 de la fiche.

</details>

### 6.Q12 — Bits, nats et la loss d'un LLM

<details><summary>Indice 1</summary>

Relis « Au-delà du livre (1) » de la fiche et ses deux encadrés 🕰️.

</details>
<details><summary>Indice 2</summary>

Bits : $\log_2$ ; nats : $\ln$. Pour passer de $L$ nats à des bits, divise par $\ln 2$. La perplexité vaut $e^{L}$ quand $L$ est en nats.

</details>
<details><summary>Indice 3</summary>

1. Bits avec $\log_2$, nats avec $\ln$ ; 1 nat $= \frac{1}{\ln 2} \approx 1{,}443$ bit. C'est le modèle de réponse. 2. Relis l'encadré 🕰️ « l'unité des losses » : quel logarithme `torch.nn.CrossEntropyLoss` et `sklearn.metrics.log_loss` utilisent-ils ? Déduis-en l'unité. 3. Pose $\frac{2{,}3}{\ln 2}$ pour les bits et $e^{2{,}3}$ pour la perplexité, puis arrondis comme demandé. 4. Un modèle qui hésite uniformément entre $V$ tokens a une perplexité de $V$ : traduis « 10 » dans ce langage. 5. La perplexité se compte **par token**. Pour un même texte, que changent des tokens plus longs : leur nombre, et l'information que porte chacun ? Une perplexité par token mesure-t-elle alors la même chose pour les deux modèles ? 6. Relis l'encadré 🕰️ « la loss et la perplexité des LLM » : comment le tokenizer de GPT-2 découpe-t-il le texte ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 6.R1 — Ch. 5 : un pas de descente de gradient à la main

<details><summary>Indice 1</summary>

La règle du ch. 5 : $w_{t+1} = w_t - \eta\,f'(w_t)$, appliquée trois fois.

</details>
<details><summary>Indice 2</summary>

Dérive $(w - 3)^2$ avec la règle de la chaîne. Pour le facteur, écris $w_{t+1} - 3$ en fonction de $w_t - 3$.

</details>
<details><summary>Indice 3</summary>

1. $f'(w) = 2(w - 3)$, donc $f'(0) = 2 \times (0 - 3) = -6$ : c'est le modèle. 2. Applique $w_1 = w_0 - \eta\,f'(w_0)$ avec ces nombres, puis calcule $f(w_0)$ et $f(w_1)$. 3. Même règle, deux fois, avec la dérivée **recalculée** au nouveau point. Pour le facteur, écris $w_{t+1} - 3 = (w_t - 3) - 2\eta\,(w_t - 3)$ et mets $w_t - 3$ en facteur. 4. Quel $\eta$ annule ce facteur ? Vérifie en faisant le pas depuis $w_0 = 0$. 5. Pendant l'entraînement, quelle quantité cherche-t-on à rendre petite, et que modifie-t-on pour y arriver ?

</details>

### 6.R2 — Ch. 3 : événements indépendants, P(A, B) = P(A) P(B)

<details><summary>Indice 1</summary>

Relis la définition de l'indépendance (ch. 3), puis la quatrième propriété du §6.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Deux tirages avec remise : $\frac{1}{52} \times \frac{1}{52}$. Sans remise : $P(\text{1er as}) \times P(\text{2e as} \mid \text{1er as})$. Deux tirages sont indépendants si la probabilité du second ne dépend pas du premier.

</details>
<details><summary>Indice 3</summary>

1. $A$ et $B$ sont indépendants si $P(A, B) = P(A)\,P(B)$ : savoir que $A$ s'est produit ne change pas la probabilité de $B$. C'est le modèle de réponse. 2. Pose $P = \frac{1}{52} \times \frac{1}{52}$, puis $I = -\log_2 P$ ; calcule à part $-\log_2 \frac{1}{52}$, et compare. 3. Pose $\frac{4}{52} \times \frac{3}{51}$. Pour l'indépendance, compare la probabilité d'un as au second tirage selon que le premier était un as ou non. 4. Pars de $I(A, B) = -\log_2\left(P(A)\,P(B)\right)$ et applique $\log(ab) = \log a + \log b$. 5. Cherche une lettre après laquelle la suivante est presque imposée en français : la probabilité de la lettre suivante dépend-elle alors de la précédente ?

</details>

### 6.R3 — 0B : logarithmes, log₂ 8, log₂ ¼ et log(ab)

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « logarithmes » du §6.4 de la fiche (et 0B, §101.2.4).

</details>
<details><summary>Indice 2</summary>

$\log_2 x$ est l'exposant : à quelle puissance faut-il élever 2 pour obtenir $x$ ? Changement de base : $\log_2 x = \frac{\ln x}{\ln 2}$.

</details>
<details><summary>Indice 3</summary>

1. $2^3 = 8$, donc $\log_2 8 = 3$ : c'est le modèle. Pour $\frac{1}{4}$, 1 et 1 024, cherche de même l'exposant $k$ tel que $2^k$ donne le nombre (il peut être négatif ou nul). 2. Écris $a = 2^x$ et $b = 2^y$, c'est-à-dire $x = \log_2 a$ et $y = \log_2 b$ : que valent $ab$, $\frac{a}{b}$ et $a^k$ comme puissances de 2 ? Lis-en les logarithmes. 3. $\log_2 5 = \frac{\ln 5}{\ln 2}$, avec $\ln 5 \approx 1{,}609438$ et $\ln 2 \approx 0{,}693147$ : fais la division, et arrondis à 3 décimales. 4. Même méthode, avec $\ln 12 \approx 2{,}484907$. 5. Pour obtenir un nombre entre 0 et 1, à quelle sorte de puissance faut-il élever 2 ? Et pour obtenir 1 ?

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 6.1 — Combien de bits pour une pièce, un dé, une lettre E ? ✏️

<details><summary>Indice 1</summary>

Applique $I = -\log_2 p$ à chaque événement ; pour des événements indépendants, les informations s'additionnent (fiche §6.4, quatrième propriété).

</details>
<details><summary>Indice 2</summary>

$-\log_2 2^{-k} = k$ : c) et d) se font de tête. Pour b), $\log_2 6 = \frac{\ln 6}{\ln 2}$. Pour f), la probabilité de **ce** résultat précis, avec trois dés indépendants. Pour g), remplace $\log_2$ par $\ln$. Pour h), combien de numéros différents donnent $k$ chiffres binaires ? Pour i), quelle est la probabilité de face ?

</details>
<details><summary>Indice 3</summary>

a) $-\log_2 \frac{1}{2} = \log_2 2 = 1$ bit : c'est le modèle. b) $-\log_2 \frac{1}{6} = \log_2 6 = \frac{\ln 6}{\ln 2}$, à arrondir à 3 décimales. c) et d) Écris $\frac{1}{8}$ et $\frac{1}{1\,024}$ comme des puissances de 2. e) Les deux tirages sont indépendants : additionne c) et d). f) La probabilité de **ce** résultat vaut $\left(\frac{1}{6}\right)^3$, quels que soient les numéros sortis : $-\log_2 \left(\frac{1}{6}\right)^3 = 3 \log_2 6$. g) Le calcul de b), avec $\ln$ au lieu de $\log_2$. h) Le plus petit $k$ tel que $2^k \ge 6$. i) Face a la probabilité 0,1 : $-\log_2 0{,}1 = \log_2 10$. j) Compare la nature des deux nombres : b) est un logarithme, h) un nombre de chiffres binaires. Lequel doit être entier, et pourquoi l'autre ne l'est-il pas ici ?

</details>

### Ex 6.2 — Bits par mot : Seuss, Holmes et l'alphabet ✏️

<details><summary>Indice 1</summary>

Avec $k$ chiffres binaires, on écrit $2^k$ numéros, de 0 à $2^k - 1$ : cherche le plus petit $k$ tel que $2^k \ge N$ (fiche §6.5).

</details>
<details><summary>Indice 2</summary>

Écris la liste des puissances de 2 : 32, 64, 128, 256, …, 4 096, 8 192, …, 32 768, 65 536. Pour e), compte d'abord les symboles. Pour g), multiplie le nombre de mots par la longueur **entière** du code de d).

</details>
<details><summary>Indice 3</summary>

a) $64 = 2^6$ exactement : 6 chiffres binaires donnent 64 numéros, de 0 à 63, juste assez pour 64 mots ; 6 bits par mot. C'est le modèle. b) à f) Même critère, le plus petit $k$ tel que $2^k \ge N$ : situe chaque $N$ entre deux puissances de 2 consécutives de la liste de l'indice 2 (pour e), compte d'abord les symboles : 26 lettres, l'espace et 10 chiffres). g) Multiplie le nombre de mots par la longueur **entière** de d). h) Avec 14 chiffres binaires, combien de numéros différents peut-on écrire ? i) Trouve la longueur pour 50 mots comme en b), puis soustrais-la de celle de c).

</details>

### Ex 6.3 — Entropie de quelques distributions ✏️

<details><summary>Indice 1</summary>

$H(p) = -\sum_i p_i \log_2 p_i$ : une surprise par issue, pondérée par sa probabilité. Les issues de probabilité 0 ne comptent pas.

</details>
<details><summary>Indice 2</summary>

Pour a), les surprises valent 1, 2, 3 et 3 bits. Pour b) et f), toutes les issues ont la même surprise. Pour e), 1 bit $= \ln 2$ nat : multiplie. Pour g), il y a deux sortes de termes, celui du 6 et cinq termes identiques. Pour h), commence par la question qui a une chance sur deux d'obtenir « oui ».

</details>
<details><summary>Indice 3</summary>

a) $\frac{1}{2} \times 1 + \frac{1}{4} \times 2 + 2 \times \frac{1}{8} \times 3 = 0{,}5 + 0{,}5 + 0{,}75 = 1{,}75$ bit : c'est le modèle. b) Huit issues de probabilité $\frac{1}{8}$, chacune de surprise $-\log_2 \frac{1}{8}$ : la moyenne de huit nombres égaux. c) $-0{,}9 \log_2 0{,}9 - 0{,}1 \log_2 0{,}1$. d) Laisse tomber l'issue de probabilité 0 ($0 \log_2 0 = 0$) : quelle distribution reste-t-il ? e) Multiplie a) par $\ln 2$. f) Quelle distribution sur 4 issues est la plus imprévisible ? Calcule son entropie comme en b). g) $0{,}5 \times \log_2 2 + 5 \times 0{,}1 \times \log_2 10$. h) Première question : « Est-ce la première issue ? », qui obtient « oui » une fois sur deux. Si c'est non, pose la même sorte de question sur ce qui reste ; compte les questions de chaque issue, fais-en la moyenne pondérée par les probabilités, et compare-la à a).

</details>

### Ex 6.4 — Morse contre code fixe : SHERLOCK HOLMES ✏️

<details><summary>Indice 1</summary>

Combien de lettres, sans l'espace ? Le code fixe en donne 5 symboles à chacune ; en Morse, lis la longueur de chaque lettre dans la table.

</details>
<details><summary>Indice 2</summary>

Regroupe les lettres : S, H, E, O et L apparaissent deux fois chacune. Pour d), combien de silences entre $n$ lettres qui se suivent ? Pour f), découpe `····` en morceaux de 1, 2, 3 ou 4 points (E, I, S, H) : l'ordre compte.

</details>
<details><summary>Indice 3</summary>

a) SHERLOCK HOLMES compte 14 lettres sans l'espace : $14 \times 5 = 70$ symboles avec le code fixe. C'est le modèle. b) Additionne les longueurs Morse des 14 lettres, chacune autant de fois qu'elle apparaît : S 3, H 4, E 1, R 3, L 4, O 3, C 4, K 3, M 2. c) b) divisé par a). d) Combien d'intervalles y a-t-il entre 14 lettres qui se suivent ? Ajoute autant de silences à b). e) d) divisé par a). f) Écris 4 comme une somme ordonnée de morceaux de 1 à 4 points (1 = E, 2 = I, 3 = S, 4 = H), par exemple $1 + 3$, qui se lit ES. Compte toutes les écritures : l'ordre compte ($1 + 3$ et $3 + 1$ sont deux lectures). g) Applique la définition d'un code préfixe aux mots de E et de I.

</details>

### Ex 6.5 — Cross-entropy et KL dans les deux sens ✏️

<details><summary>Indice 1</summary>

Écris les trois formules : $H(p)$ ; $H(p, q) = -\sum_i p_i \log_2 q_i$ (les poids viennent de $p$, les logarithmes de $q$) ; $\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p)$. Pour l'autre sens, échange les rôles de $p$ et $q$.

</details>
<details><summary>Indice 2</summary>

$\log_2 0{,}5 = -1$ : b) se fait de tête. Pour d), les poids sont 0,5 et 0,5, et les logarithmes ceux de 0,8 et 0,2. Pour e), il faut aussi $H(q)$. Pour g), que vaut $\log_2 0$ ?

</details>
<details><summary>Indice 3</summary>

$\log_2 0{,}8 \approx -0{,}3219$ et $\log_2 0{,}2 \approx -2{,}3219$. a) $H(p) = 0{,}8 \times 0{,}3219 + 0{,}2 \times 2{,}3219 \approx 0{,}2575 + 0{,}4644 \approx 0{,}722$ bit : c'est le modèle. b) Les logarithmes de $q$ valent tous $\log_2 0{,}5 = -1$ : pondère-les par $p$. c) $H(p, q) - H(p)$, avec b) et a). d) $0{,}5 \times 0{,}3219 + 0{,}5 \times 2{,}3219$. e) d) moins $H(q)$, l'entropie d'une pièce équilibrée. f) Compare c) et e). g) Face arrive une fois sur cinq, et $r$ lui donne la probabilité 0 : son terme vaut $-0{,}2 \log_2 0$. Que vaut $\log_2 0$ ? h) Additionne a) et c), et compare à b) ; pour g), écris $H(p, r)$ terme à terme, et repère celui qui pose problème.

</details>

### Ex 6.6 — Un code de Huffman à la main ✏️

<details><summary>Indice 1</summary>

Suis l'algorithme de la fiche (§6.6, « Le code de Huffman ») : fusionne les deux groupes les moins probables, note la probabilité du nouveau groupe, recommence. La longueur du mot de code d'un symbole est le nombre de fusions que subit son groupe.

</details>
<details><summary>Indice 2</summary>

Première fusion : neige et vent. Range ensuite les groupes restants par probabilité, et recommence. Chaque fois qu'un groupe est fusionné, chacun de ses symboles gagne un bit. c) $\sum_i p_i \ell_i$. e) $2^{-\ell}$ pour chaque mot de code. f) Additionne les longueurs des cinq mots du message.

</details>
<details><summary>Indice 3</summary>

a) 5 états : $2^2 = 4 < 5 \le 8 = 2^3$, donc 3 bits par état avec un code fixe. C'est le modèle. b) Première fusion : $0{,}08 + 0{,}12 = 0{,}20$ (neige et vent). Il reste quatre groupes, de probabilités 0,40 ; 0,25 ; 0,15 et 0,20 : fusionne de nouveau les deux plus petits, et continue jusqu'à un seul groupe, en notant les états de chaque groupe fusionné. La longueur d'un état est le nombre de fusions que subit son groupe. c) $\sum_i p_i \ell_i$ avec tes longueurs. d) $-\sum_i p_i \log_2 p_i$ sur les cinq probabilités. e) $\sum_i 2^{-\ell_i}$ avec tes longueurs. f) Additionne les longueurs des mots de soleil, soleil, pluie, neige et nuages. g) c) divisé par a). h) À chaque fusion, donne un `0` à l'un des deux groupes et un `1` à l'autre : le mot d'un état se lit de la dernière fusion à la première. Lis ensuite `0110` de gauche à droite, en t'arrêtant dès que les bits lus forment un mot de code. Pour l'unicité, demande-toi ce qui t'empêche d'échanger le `0` et le `1` d'une fusion.

</details>

### Ex 6.7 — H(p, q) = H(p) + KL(p‖q), et KL(p‖p) = 0 ∂

<details><summary>Indice 1</summary>

Écris les trois sommes sur le même ensemble $S$, puis utilise $\log \frac{a}{b} = \log a - \log b$.

</details>
<details><summary>Indice 2</summary>

$\sum_{i \in S} p_i \log_2 \frac{p_i}{q_i} = \sum_{i \in S} p_i \log_2 p_i - \sum_{i \in S} p_i \log_2 q_i$. Pour la question 4, $H(p)$ ne dépend pas de $q$. Pour la question 5, écris $-\mathrm{KL}(p \,\|\, q) = \sum_{i \in S} p_i \log_2 \frac{q_i}{p_i}$, passe en $\ln$ (un facteur $\frac{1}{\ln 2} > 0$ ne change pas le signe), puis applique l'inégalité terme à terme.

</details>
<details><summary>Indice 3</summary>

1. $H(p) = -\sum_{i \in S} p_i \log_2 p_i$, $H(p, q) = -\sum_{i \in S} p_i \log_2 q_i$, et la KL de l'énoncé ; hors de $S$, $p_i = 0$ et les termes sont nuls par convention. C'est le modèle. 2. Dans la KL, écris $\log_2 \frac{p_i}{q_i} = \log_2 p_i - \log_2 q_i$, coupe la somme en deux, et reconnais chacune des deux sommes. 3. Avec $q = p$, que vaut chaque $\log_2 \frac{p_i}{p_i}$ ? Utilise ensuite 2. 4. $H(p)$ ne dépend pas de $q$ : vues comme des fonctions de $q$, $H(p, q)$ et $\mathrm{KL}(p \,\|\, q)$ diffèrent d'une constante. Qu'en déduis-tu sur leurs minimums ? Pour « rassurant », demande-toi laquelle des deux on sait estimer sur des exemples sans connaître $p$, et où se trouve le minimum (question 5). 5. $\ln 2 \times \left(-\mathrm{KL}(p \,\|\, q)\right) = \sum_{i \in S} p_i \ln \frac{q_i}{p_i} \le \sum_{i \in S} p_i \left(\frac{q_i}{p_i} - 1\right)$ : simplifie le membre de droite, puis majore-le avec $\sum_{i \in S} q_i \le 1$ et $\sum_{i \in S} p_i = 1$. Pour l'égalité, regarde quand chacune des deux inégalités devient une égalité. 6. Écris le terme $-p_i \log_2 q_i$ de cette issue, avec $p_i > 0$ et $q_i = 0$.

</details>

### Ex 6.8 — L'entropie d'une pièce est maximale à p = 1/2 ∂

<details><summary>Indice 1</summary>

Question 1 : remplace $p$ par $1 - p$ dans la formule. Question 2 : dérive terme à terme, avec la règle de la chaîne pour $(1 - p) \log_2 (1 - p)$.

</details>
<details><summary>Indice 2</summary>

La dérivée de $(1 - p) \log_2 (1 - p)$ par rapport à $p$ vaut $-\log_2 (1 - p) - \frac{1}{\ln 2}$ : les constantes $\frac{1}{\ln 2}$ se compensent. $\log_2 x > 0$ si et seulement si $x > 1$. Pour la question 6, $\log_2 \frac{p_i}{1/n} = \log_2 p_i + \log_2 n$.

</details>
<details><summary>Indice 3</summary>

1. $h(1 - p) = -(1 - p) \log_2 (1 - p) - p \log_2 p = h(p)$ : les deux termes s'échangent. Échanger pile et face ne change pas l'incertitude. C'est le modèle. 2. Dérive $-p \log_2 p$ avec la formule donnée, puis $-(1 - p) \log_2 (1 - p)$ avec la règle de la chaîne (la dérivée de $1 - p$ vaut $-1$) ; additionne, puis regroupe les deux logarithmes en un seul avec $\log a - \log b = \log \frac{a}{b}$. 3. Compare $\frac{1 - p}{p}$ à 1 sur chaque intervalle, puis lis le signe de $h'$ (indice 2) ; termine par le calcul de $h(\frac{1}{2})$. 4. Écris $p \log_2 p = \frac{p \ln p}{\ln 2}$ pour utiliser la limite admise ; pour le second terme, que vaut $\log_2 1$ ? Pour $p \to 1$, sers-toi de la question 1. 5. Dérive $h'(p) = \log_2 (1 - p) - \log_2 p$ terme à terme : la dérivée de $\log_2 x$ vaut $\frac{1}{x \ln 2}$, et la règle de la chaîne s'applique encore à $\log_2 (1 - p)$. Regarde ensuite le signe de chaque morceau sur $]0 ; 1[$, puis ce que le signe de $h''$ dit de la forme de la courbe. 6. Avec $u_i = \frac{1}{n}$, $\log_2 \frac{p_i}{1/n} = \log_2 p_i + \log_2 n$ : coupe la somme en deux, sors $\log_2 n$ de la seconde, et utilise $\sum_i p_i = 1$. Pour l'inégalité, reprends la positivité de la KL (∂ 6.7, question 5).

</details>

<a id="reflexion"></a>

## 🗣️ 🧮 📄 Réflexion

### Ex 6.9 — L'entropie expliquée avec un jeu de devinettes 🗣️

<details><summary>Indice 1</summary>

Relis le paragraphe du §6.7 de la fiche sur le jeu de questions oui/non, et refais ✏️ 6.3 h).

</details>
<details><summary>Indice 2</summary>

Avec 8 nombres équiprobables, combien de questions faut-il si chacune coupe les possibilités en deux ? Si le 8 sort une fois sur deux, quelle première question poser ?

</details>
<details><summary>Indice 3</summary>

Le premier point, en modèle : avec 8 nombres équiprobables, « plus grand que 4 ? » coupe les chances en deux moitiés égales, puis on recommence sur la moitié qui reste ; il faut 3 questions à chaque partie. Pour le 8 favori, cherche la première question qui coupe encore les chances en deux moitiés égales ; puis, quand la réponse est non, compte les questions qu'il faut, en moyenne, pour départager 7 nombres équiprobables, et fais la moyenne des deux cas. Pour la limite, essaie de partager 7 nombres équiprobables en deux groupes de même probabilité : est-ce possible ? Compare enfin ta moyenne à l'entropie de ce choix, $\frac{1}{2} \times 1 + 7 \times \frac{1}{14} \log_2 14$ (pour toi seulement : l'explication, elle, reste sans formule).

</details>

### Ex 6.10 — Fermi : combien de bits pour envoyer tout Holmes ? 🧮

<details><summary>Indice 1</summary>

Avance pas à pas : nombre de caractères × bits par caractère, pour chaque façon de coder.

</details>
<details><summary>Indice 2</summary>

1 octet = 8 bits, 1 ko = 1 000 octets. Pour 2., un code de longueur fixe pour $N$ symboles demande $\lceil \log_2 N \rceil$ bits par symbole (fiche §6.5). Pour 5., le nombre de mots divisé par 25 donne des minutes ; le nombre de bits divisé par $10^8$ donne des secondes.

</details>
<details><summary>Indice 3</summary>

1. $560\,000 \times 8 \approx 4{,}5 \times 10^6$ bits, soit 560 000 octets, 560 ko : c'est le modèle. 2. Situe 90 entre deux puissances de 2 consécutives : l'exposant de la plus grande donne le nombre de bits par caractère ; multiplie-le ensuite par 560 000. 3. $560\,000 \times 4{,}5$. 4. $560\,000 \times 1$, puis divise le résultat de 1. par celui-ci. 5. $\frac{106\,000}{25}$ donne des minutes, à convertir en heures ; pour la connexion, divise le nombre de bits de 1. par $10^8$ bits par seconde. 6. Un code de caractères voit chaque caractère seul ; un lecteur voit les lettres et les mots qui précèdent, et connaît la langue : quel mot du §6.2 de la fiche nomme ce qui fait la différence ? Cherche ensuite, dans le notebook, les exercices qui descendent sous l'entropie des caractères pris un par un.

</details>

### Ex 6.11 — Shannon (1948) : l'introduction et le schéma de communication 📄

<details><summary>Indice 1</summary>

Tout est dans les deux premières pages : la figure 1, puis la liste numérotée de ses cinq éléments ; les trois raisons du logarithme sont numérotées aussi.

</details>
<details><summary>Indice 2</summary>

Pour la question 2, cherche la phrase où Shannon dit que le message est « choisi parmi un ensemble de messages possibles ». Pour la question 5, il donne la conversion entre bases ; tu peux la retrouver avec $\log_2 10$.

</details>
<details><summary>Indice 3</summary>

1. Les cinq éléments de la figure 1, dans l'ordre : la source d'information, l'émetteur (*transmitter*), le canal, le récepteur et le destinataire ; la source de bruit agit sur le canal. Pour un SMS, la source est la personne qui écrit : place de même les quatre autres. C'est le modèle. 2. Cherche la phrase où le message réel est « choisi parmi un ensemble de messages possibles » : pour quels messages l'ingénieur doit-il concevoir son système, et sait-il, au moment de le concevoir, lequel sera envoyé ? 3. Les trois raisons sont numérotées dans le texte : pour chacune, note l'adjectif qui l'ouvre et l'exemple qui l'illustre. 4. Le nom figure juste après le mot « bits », dans la phrase sur le choix de la base 2 ; pour les relais, compte les états possibles de $N$ relais à deux positions, puis prends-en le $\log_2$. 5. $\log_2 10 = \frac{\ln 10}{\ln 2}$, à arrondir à 2 décimales. 6. Un modèle de langage prédit le token suivant d'un texte : lequel des cinq éléments produit ce texte ? Pour la compression, relis le lien entre cross-entropy et longueur de code (fiche §6.8) et l'encadré 🕰️ « prédire, c'est compresser ».

</details>

<a id="entretien"></a>

## 💼 Entretien

### 6.E1 — Pourquoi la cross-entropy comme loss de classification ?

<details><summary>Indice 1</summary>

Un plan en trois temps : ce que vaut la cross-entropy pour un exemple, pourquoi elle plutôt que l'accuracy, pourquoi elle plutôt que l'erreur quadratique.

</details>
<details><summary>Indice 2</summary>

L'accuracy est-elle dérivable ? Que coûte un exemple où le modèle donnait 0,01 à la bonne classe ? Que se passe-t-il quand les probabilités prédites sont les vraies (calibration, ch. 3) ? Et le lien avec le maximum de vraisemblance ?

</details>
<details><summary>Indice 3</summary>

$-\ln q_y$, la surprise devant la bonne réponse. L'accuracy est en escalier : son gradient est nul presque partout. La cross-entropy punit très fort les erreurs sûres d'elles ; avec une softmax, son gradient par rapport aux logits vaut simplement $q - p$, et il ne s'éteint pas quand le modèle se trompe lourdement, contrairement à l'erreur quadratique après une sigmoïde. La minimiser revient à maximiser la vraisemblance, et elle récompense les probabilités calibrées.

</details>

### 6.E2 — Entropie, cross-entropy, KL : les différences

<details><summary>Indice 1</summary>

Une phrase par quantité avec l'image du code : le meilleur code, le code fait pour une autre distribution, le surcoût.

</details>
<details><summary>Indice 2</summary>

$H(p)$ dépend d'une distribution, $H(p, q)$ et la KL de deux. Écris la relation qui les lie, puis les propriétés : positivité, asymétrie, cas infini.

</details>
<details><summary>Indice 3</summary>

$H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q)$. Quand $p$ est fixée (les données), minimiser la cross-entropy revient à minimiser la KL (∂ 6.7, question 4) : c'est ce que fait l'entraînement.

</details>

### 6.E3 — La perplexité d'un modèle de langage

<details><summary>Indice 1</summary>

Définition (l'exponentielle de la loss moyenne par token), interprétation (un nombre de choix équivalent), puis les conditions d'une comparaison honnête.

</details>
<details><summary>Indice 2</summary>

Un modèle uniforme sur $V$ tokens a une perplexité de $V$. Pour comparer 20 et 30 : mêmes données de test, même tokenizer, même façon de découper les textes longs.

</details>
<details><summary>Indice 3</summary>

Pour la seconde question, sépare deux cas. Les deux perplexités ont-elles été mesurées avec le même protocole (les trois conditions de l'indice 2) ? Si oui, que dit un nombre de choix équivalent plus petit ? Si non, la comparaison a-t-elle encore un sens ? Termine par une limite : la perplexité mesure la prédiction du texte ; dit-elle quelque chose de l'exactitude des faits, ou de l'aide apportée à un utilisateur ?

</details>

### 6.E4 — Où rencontre-t-on la divergence KL en ML moderne ?

<details><summary>Indice 1</summary>

Trois ou quatre usages (fiche, 🕰️ « la KL aujourd'hui »), puis, pour chacun, dans quel sens la KL est prise.

</details>
<details><summary>Indice 2</summary>

Maximum de vraisemblance : $\mathrm{KL}(\text{données} \,\|\, \text{modèle})$. VAE : la KL entre la distribution de l'encodeur et le prior. Distillation : les probabilités du grand modèle comme cibles. RLHF : une pénalité $\mathrm{KL}(\text{modèle} \,\|\, \text{référence})$ ; DPO : la même contrainte, intégrée à sa loss. Pour le sens : sur les échantillons de quelle distribution peut-on estimer chacune ?

</details>
<details><summary>Indice 3</summary>

$\mathrm{KL}(p \,\|\, q)$ est une moyenne sous $p$ : on l'estime avec des échantillons de $p$. Pour les données, on a des exemples, pas leurs probabilités : on minimise donc $-\log q$ sur ces exemples. Pour le RLHF, on a les textes générés par le modèle : la KL se prend sous le modèle. Le sens change aussi le comportement : $\mathrm{KL}(p \,\|\, q)$ punit $q$ quand il oublie ce que $p$ produit ; $\mathrm{KL}(q \,\|\, p)$ punit $q$ quand il produit ce que $p$ juge improbable.

</details>

<a id="notebook"></a>

## Notebook, parties A à D

Les exercices du notebook (`03_notebook.ipynb`). Pour ceux qui complètent `mylearn/info.py`, la docstring de chaque fonction décrit déjà ce qu'elle doit faire, cas d'erreur compris : relis-la avant d'ouvrir un indice.

### Ex 6.12 — self_information et entropy 🔨

<details><summary>Indice 1</summary>

Écris d'abord les deux fonctions d'aide de l'énoncé, `_check_base` et `_as_distribution` : `self_information` et `entropy` tiennent ensuite en quelques lignes (contrôler, calculer $-\log p$, renvoyer le bon type).

</details>
<details><summary>Indice 2</summary>

Dans une base quelconque : `np.log(p) / np.log(base)` (pour la base 2, `np.log2(p)` est exact sur les puissances de 2). `not base > 0 or base == 1` refuse aussi `NaN`, comme `np.any(~(p > 0))` pour les probabilités. `np.ndim(p) == 0` repère un scalaire : renvoie alors `float(...)`. Dans `entropy`, garde `p[p > 0]` **avant** le logarithme : `0 * np.log2(0)` donne `nan`. Ajouter `+ 0.0` transforme un `-0.0` (la surprise d'un événement certain) en `0.0`.

</details>
<details><summary>Indice 3</summary>

```python
def self_information(p, base=2.0):
    _check_base(base)                       # ValueError unless base > 0 and base != 1 (NaN refused too)
    arr = np.asarray(p, dtype=float)
    # 1. ValueError if np.any(~(arr > 0)) or np.any(arr > 1): zero, negative, NaN or above 1
    # 2. the surprise in the requested base, then + 0.0 (the -0.0 of a certain event becomes 0.0)
    # 3. a Python float when arr.ndim == 0, else the array


def entropy(p, base=2.0):
    _check_base(base)
    arr = _as_distribution(p)               # 1-D, non-empty, finite, >= 0, sum within 1e-6 of 1
    present = arr[arr > 0]                  # BEFORE the logarithm: 0 * log(0) would give nan
    # return minus the sum of present * log(present) in the base, as a Python float (+ 0.0)
```
Le logarithme dans une base quelconque tient en une ligne, que les deux fonctions partagent : `np.log2(x) if base == 2 else np.log(x) / np.log(base)` (`np.log2` est exact sur les puissances de 2). `_as_distribution(p, name="p")` commence par `np.asarray(p, dtype=float)`, puis enchaîne trois contrôles, chacun avec une `ValueError` qui dit ce qui ne va pas : `p.ndim != 1 or p.size == 0`, puis `not np.all(np.isfinite(p)) or np.any(p < 0)`, puis `abs(p.sum() - 1) > 1e-6` ; elle renvoie le tableau.

</details>

### Ex 6.13 — Distributions de caractères et de mots 🔨

<details><summary>Indice 1</summary>

`token_distribution` fait tout le travail : choisir le vocabulaire, compter, ajouter le lissage, diviser par le total. `char_distribution` ne fait que préparer le texte et l'appeler.

</details>
<details><summary>Indice 2</summary>

Ajoute `from collections import Counter` en haut de ton fichier. `Counter(tokens)` lit les tokens une seule fois, et `counts[token]` vaut 0 pour un token absent. Sans vocabulaire : `sorted(counts)`. Avec : `list(vocabulary)`, refusé si `len(set(vocab)) != len(vocab)`. Les comptes : `np.array([counts[t] for t in vocab], dtype=float) + smoothing`. Refuse `not smoothing >= 0` et un total nul. Dans la vérification, `vowels_13[1]` est la probabilité de « e », la deuxième lettre de `"aeiou"`.

</details>
<details><summary>Indice 3</summary>

```python
def token_distribution(tokens, vocabulary=None, smoothing=0.0):
    # 1. ValueError if not smoothing >= 0 (NaN included)
    counts = Counter(tokens)                # ONE pass: tokens may be a generator
    # 2. vocab: sorted(counts) without a vocabulary; else list(vocabulary), in its order,
    #    with a ValueError if it repeats a token (len(set(vocab)) != len(vocab))
    weights = np.array([counts[token] for token in vocab], dtype=float) + smoothing
    # 3. ValueError if the total is not > 0; return vocab and the weights divided by their total
```
`char_distribution` : passe `text` en minuscules si `lowercase` (jamais l'alphabet), transforme `alphabet` en liste s'il est donné (sinon, garde `None`), puis renvoie l'appel de `token_distribution` sur le texte, avec ce vocabulaire et le même `smoothing` : une chaîne est déjà une suite de caractères.

</details>

### Ex 6.14 — Qui a l'entropie par lettre la plus haute : Holmes ou Verne ? 🔮

<details><summary>Indice 1</summary>

L'entropie est grande quand les lettres sont employées de façon équilibrée, petite quand quelques lettres concentrent la masse. Laquelle des deux langues concentre le plus ses lettres ? Et que se passe-t-il quand on ajoute des « lettres » à l'alphabet ?

</details>
<details><summary>Indice 2</summary>

Compare les fréquences des lettres des deux langues (ch. 1, 1.11) : laquelle met le plus de masse sur quelques lettres ? Sur 42 lettres, que deviennent les lettres accentuées de chaque texte ? Pour c), rappelle-toi la plus grande entropie possible sur 26 issues, et quand elle est atteinte.

</details>
<details><summary>Indice 3</summary>

Raisonne sans rien calculer. a) Sur 26 lettres, les lettres accentuées sont ignorées. Rappelle-toi les fréquences des lettres des deux langues (1.11) : laquelle met le plus de masse sur ses lettres favorites, et laquelle laisse presque inutilisées certaines lettres ? Une distribution plus concentrée a une entropie plus basse. b) Sur 42 lettres, chaque lettre accentuée devient une issue à part. Estime la part des lettres accentuées dans chaque texte (en anglais, presque aucune), puis demande-toi ce que des issues de plus, qui prennent une part de la masse, font à l'entropie de chacun des deux livres. c) Situe les quatre choix : 4,7 bits est le maximum sur 26 lettres, atteint seulement si elles sont équiprobables, et 5 bits le prix d'un code fixe. Pour juger de la baisse que causent des lettres inégales, compare, en ✏️ 6.3 g), le dé truqué au dé équilibré : quelle part de son entropie a-t-il perdue ?

</details>

### Ex 6.15 — Fréquences des lettres en anglais et en français 🎨

<details><summary>Indice 1</summary>

Trois étapes : l'ordre des lettres (`np.argsort`), les positions $y$ des lignes, puis deux appels à `ax.barh`, décalés de part et d'autre de chaque ligne.

</details>
<details><summary>Indice 2</summary>

`order = np.argsort(-p_holmes)` va de la lettre la plus fréquente à la moins fréquente ; `y = np.arange(len(order))`. Les longueurs des barres : `p_holmes[order]` et `p_verne[order]` ; les étiquettes : `[alphabet[i] for i in order]`. `ax.invert_yaxis()` met la ligne 0 en haut.

</details>
<details><summary>Indice 3</summary>

```python
def draw_letters_15(ax, alphabet, p_holmes, p_verne):
    order = np.argsort(-np.asarray(p_holmes), kind="stable")   # the letters, most frequent in Holmes first
    y = np.arange(len(order))
    ax.barh(y - 0.2, np.asarray(p_holmes)[order], height=0.4, label="Holmes (English)")
    # 1. the same call for Verne at y + 0.2, reordered by the SAME order: each row compares one letter
    # 2. the letters as tick labels of the y axis: ax.set_yticks(y), then ax.set_yticklabels(...)
    # 3. ax.invert_yaxis() puts row 0 on top; then a name for the x axis, a title and ax.legend()
```

</details>

### Ex 6.16 — cross_entropy, kl_divergence et js_divergence 🔨

<details><summary>Indice 1</summary>

Les trois fonctions commencent pareil : contrôler la base, valider $p$ et $q$ (ton `_as_distribution`, puis les formes). Les deux premières ne somment que sur le **support** de $p$, les issues où $p_i > 0$ ; la troisième réutilise la deuxième.

</details>
<details><summary>Indice 2</summary>

`support = p > 0` ; si `np.any(q[support] == 0)`, renvoie `float("inf")` **avant** tout logarithme. Sinon, la somme sur `p[support]` et `q[support]`. Pour la KL, `value if value > 0 else 0.0` élimine les $-10^{-17}$ d'arrondi. Pour JS, le mélange `(p + q) / 2` n'est jamais nul là où $p$ ou $q$ ne l'est pas.

</details>
<details><summary>Indice 3</summary>

```python
def cross_entropy(p, q, base=2.0):
    _check_base(base)
    # 1. p and q through your _as_distribution (names "p" and "q"); ValueError if p.shape != q.shape
    support = p > 0                         # only the outcomes that the data produce
    if np.any(q[support] == 0):
        return float("inf")                 # BEFORE any logarithm: no nan, no warning
    # 2. minus the sum of p[support] * log(q[support]) in the base, as a Python float (+ 0.0)
```
`kl_divergence` reprend les mêmes contrôles et le même `inf`, puis somme `ps * (log(ps) - log(qs))` sur le support (`ps, qs = p[support], q[support]`), et renvoie `0.0` si un arrondi rend le résultat légèrement négatif. `js_divergence` : après les contrôles, le mélange `m = (p + q) / 2`, puis `0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)`.

</details>

### Ex 6.17 — La cross-entropy infinie : la lettre qui manque 🐛

<details><summary>Indice 1</summary>

Une cross-entropy infinie veut dire qu'une issue que $p$ produit reçoit la probabilité 0 dans $q$ (fiche §6.8). Compare `p_verne_17` et `q_holmes_17`, lettre par lettre.

</details>
<details><summary>Indice 2</summary>

Les lettres de a) : `p_verne_17[i] > 0 and q_holmes_17[i] == 0`, pour `i, c in enumerate(LETTERS_FR)`. Pour b), compte-les dans `verne.lower()` avec `collections.Counter`. Pour c), additionne leurs probabilités dans `p_verne_17`. Pour d), `char_distribution(..., smoothing=1)` ; essaie les trois endroits possibles, regarde ce qui donne une valeur finie, puis demande-toi, dans chaque cas, quel envoi tu mesures vraiment.

</details>
<details><summary>Indice 3</summary>

Un cas en modèle : lisser Verne laisse les zéros du code de Holmes, et la cross-entropy reste `inf`. Pour les deux autres choix, lisser Holmes seul ou lisser les deux, pose-toi deux questions : la valeur est-elle finie ? Mesure-t-elle encore l'envoi du **vrai** texte de Verne, tel qu'il est ? La correction tient en deux lignes : une distribution lissée, `mylearn.info.char_distribution(..., alphabet=LETTERS_FR, smoothing=1)`, puis `mylearn.info.cross_entropy(données, code)`, les données en premier (6.16).

</details>

### Ex 6.18 — Coder le français avec le code de l'anglais, et l'inverse 🔬

<details><summary>Indice 1</summary>

Le premier argument est toujours la distribution des données envoyées, le second celle du code : `cross_entropy(données, code)`, `kl_divergence(données, code)`.

</details>
<details><summary>Indice 2</summary>

Deux distributions sur `LETTERS` : `char_distribution(holmes, alphabet=LETTERS)` et la même chose pour Verne. Pour f), coupe le texte en deux avec `len(holmes) // 2`, une distribution par moitié, puis la KL dans l'ordre (première moitié, seconde moitié).

</details>
<details><summary>Indice 3</summary>

```python
_, p_h = mylearn.info.char_distribution(holmes, alphabet=LETTERS)
_, p_v = mylearn.info.char_distribution(verne, alphabet=LETTERS)
ce_verne_holmes_18 = mylearn.info.cross_entropy(p_v, p_h)   # (data, code): Verne sent with the code of Holmes
# b) the other order; c) and d) the same two orders with kl_divergence; e) js_divergence of the two
# f) middle = len(holmes) // 2: one distribution per half of the TEXT, then the KL (first half, second half)
```

</details>

### Ex 6.19 — scipy.stats.entropy et un vrai compresseur (zlib) 📦

<details><summary>Indice 1</summary>

Dans `help(scipy.stats.entropy)`, regarde ce que la fonction fait de `pk` (« normalized ») et la valeur par défaut de `base`. Pour zlib, deux conversions : du texte vers des octets (`encode`), puis des octets vers des bits.

</details>
<details><summary>Indice 2</summary>

Les comptes de `collections.Counter(holmes)` suffisent à `scipy.stats.entropy`, qui les normalise ; pense à sa base par défaut. La longueur du résultat de `zlib.compress` est un nombre d'**octets** : on veut des bits, divisés par le nombre de **caractères** du texte d'origine (quelques caractères de Holmes prennent plusieurs octets en UTF-8). Fais de même avec le texte mélangé.

</details>
<details><summary>Indice 3</summary>

```python
h_chars_19 = scipy.stats.entropy(list(collections.Counter(holmes).values()), base=2)   # counts are fine
zlib_bits_19 = 8 * len(zlib.compress(holmes.encode("utf-8"), 9)) / len(holmes)        # bits per CHARACTER
# c) the same measure on the shuffled text of the statement (np.random.default_rng(19)), divided by len(holmes)
# d) a comparison between b) and a)
```

</details>

### Ex 6.20 — Lire une courbe de loss : nats, bits et perplexité 📈

<details><summary>Indice 1</summary>

Tout part de la relation perplexité $= e^{\text{loss en nats}}$. Au pas 0, un modèle uniforme sur $V$ caractères a une perplexité de $V$.

</details>
<details><summary>Indice 2</summary>

`np.argmin(val_20)` donne l'**indice** du minimum, et `steps_20[...]` le pas correspondant. En bits : divise par `np.log(2)`. `np.exp(val_20) < 3` est un tableau de booléens ; `np.argmax` d'un tableau de booléens renvoie l'indice du premier `True`. N'oublie pas d'arrondir a) à l'entier.

</details>
<details><summary>Indice 3</summary>

```python
vocab_20 = round(float(np.exp(val_20[0])))      # a) the perplexity of a uniform guess = the number of choices
best = int(np.argmin(val_20))                    # the POSITION of the validation minimum, not a step
# b) the step at that position; c) its loss in bits (divide by ln 2); d) its perplexity, np.exp(loss)
# e) the first position where np.exp(val_20) < 3 (np.argmax of the boolean array), then its step
# f) the perplexity at the last position, val_20[-1]
```

</details>

### Ex 6.21 — Mesurer avant d'optimiser : compter des caractères vite 🛠️

<details><summary>Indice 1</summary>

Les quatre fonctions renvoient la même chose, un tableau NumPy de 26 entiers dans l'ordre de `LETTERS` ; seule la façon de compter change.

</details>
<details><summary>Indice 2</summary>

Boucle : un dictionnaire qui part de 0 pour chaque lettre, et `if ch in counts: counts[ch] += 1`. `Counter` : `counts[letter]` vaut 0 pour une lettre absente. `str.count` : une compréhension de liste sur `LETTERS`. NumPy : `np.frombuffer(..., dtype=np.uint8)` donne les codes des octets (97 pour « a ») ; `np.bincount(codes, minlength=123)` compte chaque code de 0 à 122 ; la tranche `[97:123]` garde a à z.

</details>
<details><summary>Indice 3</summary>

```python
def count_loop_21(text):
    counts = {letter: 0 for letter in LETTERS}
    # for each character ch of text: if ch in counts, add 1 to counts[ch]
    # return np.array of the 26 counts, in the order of LETTERS


def count_numpy_21(text):
    codes = np.frombuffer(text.encode("ascii", "ignore"), dtype=np.uint8)   # 97 for "a", ..., 122 for "z"
    # np.bincount(codes, minlength=123) counts every code from 0 to 122, even a "z" that never occurs;
    # return its slice from ord("a") to ord("z"), "z" included
```
`count_counter_21` : `collections.Counter(text)`, puis le même tableau des 26 comptes que dans la boucle (un `Counter` renvoie 0 pour une lettre absente). `count_str_21` : une compréhension de liste sur `LETTERS` qui appelle `text.count`, convertie en tableau.

</details>

### Ex 6.22 — perplexity et log_loss 🔨

<details><summary>Indice 1</summary>

`perplexity` : contrôler, puis une formule d'une ligne. `log_loss` : contrôler, choisir la probabilité de la **vraie** classe de chaque exemple, la couper dans $[\varepsilon ; 1 - \varepsilon]$, puis la moyenne des $-\log$.

</details>
<details><summary>Indice 2</summary>

Perplexité : `np.exp(-np.mean(np.log(probs)))`, après avoir refusé une liste vide et toute valeur hors de $]0 ; 1]$ (avec `~(probs > 0)` pour attraper `NaN`). Log loss : en binaire (`y_prob` à une dimension), `np.where(labels == 1, probs, 1 - probs)` ; en multiclasse (deux dimensions), `probs[np.arange(n), labels]`. Les contrôles : longueurs égales, labels entiers dans le bon intervalle (0 ou 1 en binaire), probabilités dans $[0 ; 1]$ (`NaN` compris : `np.isfinite`), lignes de somme 1 (tolérance $10^{-6}$). `np.clip` renvoie une copie : tes arguments ne bougent pas.

</details>
<details><summary>Indice 3</summary>

```python
def perplexity(token_probs):
    probs = np.asarray(token_probs, dtype=float)
    # ValueError if probs is empty, or if np.any(~(probs > 0)) or np.any(probs > 1)
    return float(np.exp(-np.mean(np.log(probs))))   # exp of the mean surprise in nats


def log_loss(y_true, y_prob, base=np.e):
    labels, probs = np.asarray(y_true), np.asarray(y_prob, dtype=float)
    # 1. checks, each with a ValueError: the base; labels 1-D, probs 1-D or 2-D, same non-zero length;
    #    integer labels (then labels.astype(int)); probabilities finite and in [0, 1]; labels 0 or 1
    #    in binary, below the number of columns in multiclass; rows of sum 1 within 1e-6
    # 2. the probability of the TRUE class of each sample, in p_true:
    #    binary (y_prob 1-D): np.where(labels == 1, probs, 1.0 - probs)
    #    multiclass (2-D):    probs[np.arange(labels.size), labels]
    # 3. np.clip(p_true, eps, 1 - eps) with eps = np.finfo(float).eps (a copy: the caller's array
    #    does not move), then the mean of -log(p_true) in the base, as a Python float
```

</details>

### Ex 6.23 — Huffman : construire, encoder, décoder 🔨

<details><summary>Indice 1</summary>

Ajoute `import heapq` en haut de ton fichier. `huffman_code` : un dictionnaire `{symbole: ""}` que les fusions allongent par la gauche. `huffman_encode` : une recherche par symbole. `huffman_decode` : inverser le code (mot de code vers symbole), puis lire les bits un par un.

</details>
<details><summary>Indice 2</summary>

La file contient des triplets `(probabilité, numéro, liste de symboles)` : `heapq.heapify`, puis, tant qu'il reste plus d'un élément, deux `heapq.heappop`, une boucle qui ajoute `"0"` devant les mots du premier groupe et `"1"` devant ceux du second, et un `heapq.heappush` de la fusion avec un numéro neuf. Valide avec ton `_as_distribution` (après les longueurs et les répétitions). Pour décoder : l'ensemble des **débuts** stricts des mots de code (`word[:k]` pour `k` de 1 à `len(word) - 1`) ; s'il contient un mot de code, le code n'est pas préfixe ; si le mot en cours n'est ni un mot de code ni un début, aucun mot ne peut lui correspondre.

</details>
<details><summary>Indice 3</summary>

```python
def huffman_code(symbols, probs):
    # 1. checks: symbols = list(symbols); same length as probs; symbols non-empty and distinct;
    #    then weights = your _as_distribution(probs, "probs")
    # 2. a single symbol gets the codeword "0"
    codes = {symbol: "" for symbol in symbols}
    heap = [(float(w), i, [s]) for i, (s, w) in enumerate(zip(symbols, weights))]   # (prob, number, group)
    heapq.heapify(heap)
    # 3. while len(heap) > 1: heappop the two least probable groups; put "0" IN FRONT of the codewords
    #    of the first group and "1" in front of those of the second; heappush their merge, with the
    #    probability w0 + w1, a NEW number (a counter that only goes up) and the group group0 + group1
    # 4. return codes
```
`huffman_encode` : `"".join(code[s] for s in symbols)` dans un `try`, dont le `except KeyError` lève à la place une `ValueError` qui nomme le symbole. `huffman_decode` : après les contrôles (des bits faits de `"0"` et de `"1"`, des mots de code distincts et non vides), le code inversé `{word: symbol for symbol, word in code.items()}` et l'ensemble des débuts stricts `{word[:k] for word in inverse for k in range(1, len(word))}` : s'il contient un mot de code, le code n'est pas préfixe. Lis ensuite les bits un par un avec `current += bit` : un mot de code donne un symbole et vide `current`, un `current` qui n'est ni un mot ni un début est une erreur, et `current` doit être vide à la fin.

</details>

### Ex 6.24 — Compresser Holmes : code fixe, Morse, Huffman et entropie 🔬

<details><summary>Indice 1</summary>

Chaque mesure est un nombre de symboles (ou de bits) divisé par le nombre de lettres. En Morse, additionne les longueurs des mots ; avec Huffman, encode tout le texte et mesure la longueur de la chaîne de bits.

</details>
<details><summary>Indice 2</summary>

`sum(len(MORSE[c]) for c in holmes_letters_24)` ; $n$ lettres qui se suivent ont $n - 1$ silences entre elles. c) la distribution `char_distribution(holmes_letters_24, alphabet=LETTERS)`, puis `huffman_code(list(LETTERS), p)`, puis `len(huffman_encode(holmes_letters_24, code)) / n`. e) Le taux du livre : Huffman divisé par le code fixe.

</details>
<details><summary>Indice 3</summary>

```python
dots = sum(len(MORSE[c]) for c in holmes_letters_24)          # all the dots and dashes of the text
_, p = mylearn.info.char_distribution(holmes_letters_24, alphabet=LETTERS)
code = mylearn.info.huffman_code(list(LETTERS), p)
# a) dots per letter; b) dots plus the n - 1 silences between n letters (n = len(holmes_letters_24))
# c) the length of huffman_encode(holmes_letters_24, code), per letter; d) c) minus the entropy of p
# e) c) divided by the 5 symbols of the fixed code; f) is a) below the entropy of p?
```

</details>

### Ex 6.25 — Le code de Huffman de Holmes pour envoyer Verne 🔮

<details><summary>Indice 1</summary>

Relis ∂ 6.7 : aucun code ne bat, en moyenne, celui qui est fait pour les vraies fréquences. Et pense au prix du code fixe.

</details>
<details><summary>Indice 2</summary>

Le coût moyen d'un code idéal fait pour $q$, sur des données $p$, vaut $H(p) + \mathrm{KL}(p \,\|\, q)$, et le code de Huffman en est en pratique très proche. Pour b), compare ce coût au prix du code fixe. Pour c) et d), il te faut un ordre de grandeur de la KL entre l'anglais et le français, dans chaque sens.

</details>
<details><summary>Indice 3</summary>

Prévois avec ce que tu as déjà mesuré, sans rien exécuter de nouveau. a) Relis ∂ 6.7 : en moyenne, un code fait pour d'autres fréquences peut-il battre le code fait pour les vraies ? b) Le coût attendu est proche de $H(p_{\text{Verne}}) + \mathrm{KL}(\text{Verne} \,\|\, \text{Holmes})$ : additionne ta mesure de 🔮 6.14 e) et celle de 6.18 c), puis compare la somme à 5. c) Le surcoût d'un code idéal est la KL de 6.18 c) : laquelle des trois valeurs proposées en est la plus proche ? d) Compare 6.18 c) et 6.18 d), en vérifiant quel livre est envoyé avec le code de quel autre.

</details>

### Ex 6.26 — Le contexte local réduit la surprise : les bigrammes 🔬

<details><summary>Indice 1</summary>

Deux modèles, deux tableaux : `unigram` (26 probabilités) et `bigram_26` (26 × 26). La surprise moyenne d'un texte est la moyenne des $-\log_2$ des probabilités que le modèle donne aux lettres réellement observées.

</details>
<details><summary>Indice 2</summary>

Construis `followers[a]`, la liste des lettres qui suivent `a` dans `train_26` (`collections.defaultdict(list)` et `zip(train_26, train_26[1:])`), puis une ligne par lettre avec `token_distribution(followers[a], vocabulary=list(LETTERS), smoothing=1)`. Avec le tableau `idx` des numéros des lettres de `test_26` (`INDEX`), `bigram_26[idx[:-1], idx[1:]]` donne d'un coup tous les $P(x_t \mid x_{t-1})$. Des surprises en bits donnent une perplexité de $2^{\text{bits}}$.

</details>
<details><summary>Indice 3</summary>

```python
followers = collections.defaultdict(list)
for before, after in zip(train_26, train_26[1:]):
    followers[before].append(after)         # followers["q"]: every letter that follows a "q" in train_26
# bigram_26: one row per letter c of LETTERS, the probabilities of
#            token_distribution(followers[c], vocabulary=list(LETTERS), smoothing=1)
idx = np.array([INDEX[c] for c in test_26])
# a) the mean of self_information(unigram[idx]), with unigram the probabilities of 1. of the statement
# b) the same with bigram_26[idx[:-1], idx[1:]]: row = previous letter, column = next letter
# c) 2 ** each of them (the surprises are in bits); d) b) with the indices of verne_letters_26
```

</details>

### Ex 6.27 — Passer sous la barre de Huffman lettre à lettre 🏆

<details><summary>Indice 1</summary>

Une piste : des blocs de longueur fixe, des paires puis des triplets. Le code de Huffman des blocs tient compte du contexte à l'intérieur de chaque bloc : « th » ou « the » deviennent des symboles fréquents, avec des mots de code courts.

</details>
<details><summary>Indice 2</summary>

`split_27` : `[letters[i:i + k] for i in range(0, len(letters), k)]`, avec $k \le 8$. `model_27` doit donner une probabilité non nulle à **tous** les blocs possibles, y compris le dernier, qui peut être plus court : un vocabulaire de tous les blocs de 1 à $k$ lettres (`"".join(t)` pour `t` dans `itertools.product(LETTERS, repeat=size)`), puis `token_distribution(train_blocks, vocabulary=..., smoothing=...)`. Essaie plusieurs valeurs de `smoothing`.

</details>
<details><summary>Indice 3</summary>

Des paires ne suffisent pas tout à fait (vers 3,9 bits par lettre). Des triplets passent : il y a 17 576 triplets possibles, dont beaucoup ne sont jamais vus ; un lissage de 1 leur donne trop de masse, un lissage de 0,1 bien moins, et le résultat s'améliore nettement.

</details>
