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

Shannon voulait transmettre un message d'un point à un autre, efficacement, malgré le bruit du canal. Sa mesure ne dépend que de la **probabilité** du message : un message attendu n'apprend presque rien. Le §6.1 de la fiche cite plusieurs usages en deep learning (la loss des classifieurs, la perplexité, le VAE, les arbres de décision). La probabilité qu'on prête à un message dépend de ce qu'on attendait.

</details>

### 6.Q2 — Plus c'est rare, plus ça informe

<details><summary>Indice 1</summary>

Relis « La surprise » (§6.2 de la fiche), puis la formule du §6.4.

</details>
<details><summary>Indice 2</summary>

La surprise vaut $I = -\log_2 p$. Que vaut $\log_2 1$ ? Pour la question 5, compare $-\log_2 \frac{p}{2}$ et $-\log_2 p$ avec la règle $\log \frac{a}{b} = \log a - \log b$.

</details>
<details><summary>Indice 3</summary>

« Ornithorynque » est bien plus rare que « Bonjour » au début d'un SMS. Un événement certain : $-\log_2 1$ ; un événement de probabilité $\frac{1}{16} = 2^{-4}$ : l'exposant, au signe près. L'échelle de 0 à 100 est subjective ; la théorie la remplace par une formule fondée sur la probabilité. Diviser $p$ par 2 ajoute $\log_2 2$.

</details>

### 6.Q3 — Contexte global, contexte local

<details><summary>Indice 1</summary>

Relis « Le contexte » (§6.2 de la fiche) et l'encadré ⚠️ « Une construction du livre à oublier ».

</details>
<details><summary>Indice 2</summary>

Global : ce que l'émetteur et le récepteur partagent avant le message ; local : ce qui précède, dans le message lui-même. Pour la question 4, un mot surprenant est-il un mot fréquent ou un mot rare ? Que donnerait un tirage où il sortirait souvent ?

</details>
<details><summary>Indice 3</summary>

Le prior du ch. 4 résume les attentes avant d'observer. Une pmf proportionnelle à la surprise ferait sortir le plus souvent les mots les plus rares : l'inverse d'une vraie langue. Dans un numéro de série, les caractères sont indépendants : celui d'avant ne dit rien du suivant ; en français, un « q » appelle presque toujours un « u ».

</details>

### 6.Q4 — Le bit est une unité

<details><summary>Indice 1</summary>

Relis le §6.3 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un chiffre binaire est un support ; le bit d'information mesure ce qu'on apprend en le lisant. Une case qui vaut toujours la même chose : quelle est la probabilité de ce qu'on y lit ? Pour la pièce à 99 %, compare la surprise de pile et celle de face, puis tiens compte de la fréquence de chacune.

</details>
<details><summary>Indice 3</summary>

La case constante : une probabilité de 1, donc une surprise nulle. La pièce équilibrée : 1 bit ; la pièce à 99 % : moins d'un dixième de bit en moyenne ($-0{,}99 \log_2 0{,}99 - 0{,}01 \log_2 0{,}01$), car pile, presque certain, n'apprend presque rien. Le nat va avec $\ln$ (et le hartley avec $\log_{10}$). « Contenir » : un registre de 8 chiffres binaires peut porter jusqu'à 8 bits, et aucun s'il vaut toujours la même chose.

</details>

### 6.Q5 — Les quatre propriétés de l'information

<details><summary>Indice 1</summary>

Relis la liste des quatre propriétés du §6.4 de la fiche, et son mini-exemple.

</details>
<details><summary>Indice 2</summary>

Calcule $-\log_2 1$, $-\log_2 \frac{1}{2}$, $-\log_2 \frac{1}{4}$, puis l'information de l'événement « les deux à la fois », de probabilité $\frac{1}{2} \times \frac{1}{16}$. Pour la question 4, demande-toi ce que le début de la phrase fait à la surprise du dernier mot.

</details>
<details><summary>Indice 3</summary>

Les surprises valent 0, 1 et 2 bits, et l'événement double $-\log_2 \frac{1}{32}$ : la somme. Si « classeur » rendait « vert » plus probable (tous les classeurs du bureau sont verts), la surprise de « vert » après « classeur » ne serait plus celle de « vert » seul. Après « Il pleut, n'oublie pas ton… », « parapluie » surprend peu ; pris seul, ce mot paraîtrait bien plus surprenant. Le logarithme transforme un produit en somme ; $\log_2 p \le 0$ pour $p \le 1$.

</details>

### 6.Q6 — Taille du vocabulaire et bits par mot

<details><summary>Indice 1</summary>

Relis le §6.5 de la fiche.

</details>
<details><summary>Indice 2</summary>

Avec $k$ chiffres binaires, on écrit $2^k$ numéros différents. Cherche le plus petit $k$ tel que $2^k \ge N$.

</details>
<details><summary>Indice 3</summary>

$2^1 = 2$, $2^2 = 4$, $2^{10} = 1\,024$ : 1 025 mots demandent un chiffre de plus que 1 024. Un code s'écrit avec un nombre **entier** de chiffres, d'où $\lceil \log_2 N \rceil$. La liste partagée à l'avance fait partie du contexte global. $\log_2 N$ est une information, $\lceil \log_2 N \rceil$ une longueur.

</details>

### 6.Q7 — Morse, Vail et les codes adaptatifs

<details><summary>Indice 1</summary>

Relis le §6.6 de la fiche, l'encadré ⚠️ « Le Morse a besoin de ses silences » et l'encadré 🧮 sur les codes préfixes.

</details>
<details><summary>Indice 2</summary>

E et T sont les lettres les plus fréquentes de l'anglais. Pour la question 4, découpe `· · · −` en morceaux qui sont des lettres de la liste : le dernier morceau finit forcément par `−`, c'est donc T, A, U ou V. Pour la question 6, compare, pour un symbole rare, la longueur de son mot dans un code adaptatif préfixe (la figure du code de Huffman de la fiche) et dans un code fixe.

</details>
<details><summary>Indice 3</summary>

Vail comptait les caractères en plomb des casses d'un imprimeur. Lectures possibles : V ; S puis T ; I puis A ; E puis U ; E, I, T… Sans silences, `·` (E) est le début de `··` (I) : le Morse n'est pas préfixe. Dans la figure du code de Huffman (§6.6), les deux symboles les plus rares ont 4 bits, contre 3 pour un code fixe de six symboles : un message fait de ces symboles coûte plus cher.

</details>

### 6.Q8 — Entropie nulle, entropie maximale

<details><summary>Indice 1</summary>

Relis le §6.7 de la fiche et l'encadré ⚠️ « L'entropie est une propriété de la distribution ».

</details>
<details><summary>Indice 2</summary>

L'entropie est la moyenne des surprises, chacune pondérée par sa probabilité. Elle est nulle quand aucune issue ne surprend, maximale quand on ne peut rien deviner. Pour la question 4, calcule les quatre entropies : elles sont toutes entre 0 et $\log_2 3$.

</details>
<details><summary>Indice 3</summary>

$H([1 ; 0]) = 0$, $H([0{,}7 ; 0{,}3]) \approx 0{,}88$, $H([0{,}5 ; 0{,}5]) = 1$, $H([\frac{1}{3} ; \frac{1}{3} ; \frac{1}{3}]) \approx 1{,}58$. Sur 16 issues, l'entropie maximale est celle de la loi uniforme, $\log_2 16$. L'entropie est l'espérance de la variable aléatoire « surprise » $-\log_2 p(X)$ : elle ne dépend que de $p$.

</details>

### 6.Q9 — Entropie et « organisation » : attention au faux ami

<details><summary>Indice 1</summary>

Relis l'encadré ⚠️ « Entropie et organisation : un faux ami » du §6.7 de la fiche.

</details>
<details><summary>Indice 2</summary>

Une source structurée est prévisible : sa surprise moyenne est-elle grande ou petite ? Pour des lettres tirées uniformément, quelle est l'entropie maximale sur 26 lettres ?

</details>
<details><summary>Indice 3</summary>

Prévisible, donc entropie faible. Des lettres uniformes atteignent le maximum, $\log_2 26 \approx 4{,}7$ bits ; Holmes est plus bas (🔮 6.14), et c'est lui le plus organisé. L'entropie de Shannon grandit avec le désordre, comme celle des physiciens. Une bonne fin de phrase : « … la surprise moyenne d'un symbole, c'est-à-dire l'incertitude avant de le lire ».

</details>

### 6.Q10 — Le mauvais code coûte plus cher

<details><summary>Indice 1</summary>

Relis « Mélanger les codes » et « Probabilités nulles et lissage de Laplace » (§6.8 de la fiche).

</details>
<details><summary>Indice 2</summary>

Dans $H(p, q) = -\sum_i p_i \log_2 q_i$, les fréquences viennent de $p$, les longueurs des mots de code de $q$. Aucun code ne bat celui qui est fait pour les vraies fréquences. Le taux de compression : bits du code adaptatif divisés par bits du code fixe.

</details>
<details><summary>Indice 3</summary>

Égalité seulement si $q = p$. Un taux proche de 0 signifie une forte compression ; le livre trouve un peu moins de 0,5 avec un code adapté. Un mot que $q$ juge impossible coûte $-\log_2 0$ : le livre ajoute à chaque code une occurrence de chaque mot de l'autre livre qui lui manque. En classification, $p$ est le vecteur *one-hot* de la vraie classe, $q$ les probabilités prédites.

</details>

### 6.Q11 — KL : positive, asymétrique, nulle quand… ?

<details><summary>Indice 1</summary>

Relis le §6.9 de la fiche, son encadré ⚠️ « L'ordre des arguments » et le paragraphe sur Jensen-Shannon.

</details>
<details><summary>Indice 2</summary>

$\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p)$. Chaque écart entre $p_i$ et $q_i$ est pondéré par la fréquence dans la **première** distribution. Une distance doit être symétrique.

</details>
<details><summary>Indice 3</summary>

Toujours $\ge 0$, nulle seulement si $q = p$. Pas symétrique : le livre trouve environ 0,29 bit par mot dans un sens et 0,5 dans l'autre. On écrit $\mathrm{KL}(\text{données} \,\|\, \text{code})$, comme $H(\text{données}, \text{code})$. La divergence de Jensen-Shannon compare chaque distribution à leur mélange.

</details>

### 6.Q12 — Bits, nats et la loss d'un LLM

<details><summary>Indice 1</summary>

Relis « Au-delà du livre (1) » de la fiche et ses deux encadrés 🕰️.

</details>
<details><summary>Indice 2</summary>

Bits : $\log_2$ ; nats : $\ln$. Pour passer de $L$ nats à des bits, divise par $\ln 2$. La perplexité vaut $e^{L}$ quand $L$ est en nats.

</details>
<details><summary>Indice 3</summary>

$\frac{2{,}3}{0{,}693}$ et $e^{2{,}3}$. Une perplexité de 10 : le modèle hésite en moyenne comme entre 10 tokens équiprobables. Si un tokenizer découpe en morceaux plus longs, chaque token porte plus d'information, et le nombre de tokens change : les perplexités ne mesurent plus la même chose. Les tokens sont des sous-mots (BPE).

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

$f'(w) = 2(w - 3)$, d'où $w_{t+1} - 3 = (1 - 2\eta)(w_t - 3)$ : le facteur vaut $1 - 0{,}2$. Le minimum en un pas : un facteur nul. Pour la question 5 : $f$ mesure l'erreur du réseau sur les données, $w$ regroupe tout ce qu'on ajuste.

</details>

### 6.R2 — Ch. 3 : événements indépendants, P(A, B) = P(A) P(B)

<details><summary>Indice 1</summary>

Relis la définition de l'indépendance (ch. 3), puis la quatrième propriété du §6.4 de la fiche.

</details>
<details><summary>Indice 2</summary>

Deux tirages avec remise : $\frac{1}{52} \times \frac{1}{52}$. Sans remise : $P(\text{1er as}) \times P(\text{2e as} \mid \text{1er as})$. Deux tirages sont indépendants si la probabilité du second ne dépend pas du premier.

</details>
<details><summary>Indice 3</summary>

$-\log_2 \frac{1}{52^2} = 2 \log_2 52$ : deux fois l'information d'un seul tirage. Deux as sans remise : $\frac{4}{52} \times \frac{3}{51}$ ; après un as, il n'en reste que 3 sur 51, donc les tirages ne sont pas indépendants. Pour la question 4, $-\log_2(P(A)P(B))$ et la règle du produit. En français, « q » suivi de « u ».

</details>

### 6.R3 — 0B : logarithmes, log₂ 8, log₂ ¼ et log(ab)

<details><summary>Indice 1</summary>

Relis l'encadré 🧮 « logarithmes » du §6.4 de la fiche (et 0B, §101.2.4).

</details>
<details><summary>Indice 2</summary>

$\log_2 x$ est l'exposant : à quelle puissance faut-il élever 2 pour obtenir $x$ ? Changement de base : $\log_2 x = \frac{\ln x}{\ln 2}$.

</details>
<details><summary>Indice 3</summary>

$2^3 = 8$, $2^{-2} = \frac{1}{4}$, $2^0 = 1$, $2^{10} = 1\,024$. $\ln 5 \approx 1{,}609438$, $\ln 12 \approx 2{,}484907$, $\ln 2 \approx 0{,}693147$. Pour $0 < p < 1$, il faut un exposant négatif.

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 6.1 — Combien de bits pour une pièce, un dé, une lettre E ?

<details><summary>Indice 1</summary>

Applique $I = -\log_2 p$ à chaque événement ; pour des événements indépendants, les informations s'additionnent (fiche §6.4, quatrième propriété).

</details>
<details><summary>Indice 2</summary>

$-\log_2 2^{-k} = k$ : c) et d) se font de tête. Pour b), $\log_2 6 = \frac{\ln 6}{\ln 2}$. Pour f), la probabilité de **ce** résultat précis, avec trois dés indépendants. Pour g), remplace $\log_2$ par $\ln$. Pour h), combien de numéros différents donnent $k$ chiffres binaires ? Pour i), quelle est la probabilité de face ?

</details>
<details><summary>Indice 3</summary>

e) est la somme de c) et d). f) La probabilité vaut $\left(\frac{1}{6}\right)^3$ quels que soient les numéros sortis : trois fois b). h) $2^2 = 4 < 6 \le 8 = 2^3$. i) $-\log_2 0{,}1 = \log_2 10$. j) b) est une quantité d'information (un logarithme), h) un nombre de chiffres, forcément entier.

</details>

### Ex 6.2 — Bits par mot : Seuss, Holmes et l'alphabet

<details><summary>Indice 1</summary>

Avec $k$ chiffres binaires, on écrit $2^k$ numéros, de 0 à $2^k - 1$ : cherche le plus petit $k$ tel que $2^k \ge N$ (fiche §6.5).

</details>
<details><summary>Indice 2</summary>

Écris la liste des puissances de 2 : 32, 64, 128, 256, …, 4 096, 8 192, …, 32 768, 65 536. Pour e), compte d'abord les symboles. Pour g), multiplie le nombre de mots par la longueur **entière** du code de d).

</details>
<details><summary>Indice 3</summary>

64 est exactement $2^6$ ; un mot de plus demande un chiffre de plus. 236 est entre 128 et 256 ; 7 819 entre 4 096 et 8 192 ; 37 symboles entre 32 et 64 ; 50 257 entre 32 768 et 65 536. h) 14 chiffres binaires donnent $2^{14}$ numéros. i) 50 mots tiennent entre 32 et 64.

</details>

### Ex 6.3 — Entropie de quelques distributions

<details><summary>Indice 1</summary>

$H(p) = -\sum_i p_i \log_2 p_i$ : une surprise par issue, pondérée par sa probabilité. Les issues de probabilité 0 ne comptent pas.

</details>
<details><summary>Indice 2</summary>

Pour a), les surprises valent 1, 2, 3 et 3 bits. Pour b) et f), toutes les issues ont la même surprise. Pour e), 1 bit $= \ln 2$ nat : multiplie. Pour g), il y a deux sortes de termes, celui du 6 et cinq termes identiques. Pour h), commence par la question qui a une chance sur deux d'obtenir « oui ».

</details>
<details><summary>Indice 3</summary>

a) $\frac{1}{2} \times 1 + \frac{1}{4} \times 2 + 2 \times \frac{1}{8} \times 3$. c) $-0{,}9 \log_2 0{,}9 - 0{,}1 \log_2 0{,}1$. g) $0{,}5 \times 1 + 5 \times 0{,}1 \times \log_2 10$. h) « Est-ce la première issue ? », puis « la deuxième ? », puis « la troisième ? » : 1, 2, 3 ou 3 questions selon l'issue.

</details>

### Ex 6.4 — Morse contre code fixe : SHERLOCK HOLMES

<details><summary>Indice 1</summary>

Combien de lettres, sans l'espace ? Le code fixe en donne 5 symboles à chacune ; en Morse, lis la longueur de chaque lettre dans la table.

</details>
<details><summary>Indice 2</summary>

Regroupe les lettres : S, H, E, O et L apparaissent deux fois chacune. Pour d), combien de silences entre $n$ lettres qui se suivent ? Pour f), découpe `····` en morceaux de 1, 2, 3 ou 4 points (E, I, S, H) : l'ordre compte.

</details>
<details><summary>Indice 3</summary>

14 lettres. Longueurs Morse : S 3, H 4, E 1, R 3, L 4, O 3, C 4, K 3, M 2. 13 silences. f) Les découpages de 4 en parts de 1 à 4, dans l'ordre : 1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2, 1+3, 3+1, 4. g) Le mot de E est le début de celui de I.

</details>

### Ex 6.5 — Cross-entropy et KL dans les deux sens

<details><summary>Indice 1</summary>

Écris les trois formules : $H(p)$ ; $H(p, q) = -\sum_i p_i \log_2 q_i$ (les poids viennent de $p$, les logarithmes de $q$) ; $\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p)$. Pour l'autre sens, échange les rôles de $p$ et $q$.

</details>
<details><summary>Indice 2</summary>

$\log_2 0{,}5 = -1$ : b) se fait de tête. Pour d), les poids sont 0,5 et 0,5, et les logarithmes ceux de 0,8 et 0,2. Pour e), il faut aussi $H(q)$. Pour g), que vaut $\log_2 0$ ?

</details>
<details><summary>Indice 3</summary>

$\log_2 0{,}8 \approx -0{,}3219$ et $\log_2 0{,}2 \approx -2{,}3219$. a) $0{,}8 \times 0{,}3219 + 0{,}2 \times 2{,}3219$. d) $0{,}5 \times 0{,}3219 + 0{,}5 \times 2{,}3219$. e) d) moins $H(q) = 1$. g) Face arrive une fois sur cinq, et $r$ lui donne la probabilité 0 : son terme vaut $-0{,}2 \log_2 0$.

</details>

### Ex 6.6 — Un code de Huffman à la main

<details><summary>Indice 1</summary>

Suis l'algorithme de la fiche (§6.6, « Le code de Huffman ») : fusionne les deux groupes les moins probables, note la probabilité du nouveau groupe, recommence. La longueur du mot de code d'un symbole est le nombre de fusions que subit son groupe.

</details>
<details><summary>Indice 2</summary>

Première fusion : neige et vent. Range ensuite les groupes restants par probabilité, et recommence. Chaque fois qu'un groupe est fusionné, chacun de ses symboles gagne un bit. c) $\sum_i p_i \ell_i$. e) $2^{-\ell}$ pour chaque mot de code. f) Additionne les longueurs des cinq mots du message.

</details>
<details><summary>Indice 3</summary>

Fusions : $0{,}08 + 0{,}12 = 0{,}20$ ; $0{,}15 + 0{,}20 = 0{,}35$ ; $0{,}25 + 0{,}35 = 0{,}60$ ; $0{,}40 + 0{,}60 = 1$. Soleil n'est fusionné qu'une fois, neige et vent quatre fois. Un code possible : soleil `0`, nuages `10`, pluie `110`, neige `1110`, vent `1111`.

</details>

### Ex 6.7 — H(p, q) = H(p) + KL(p‖q), et KL(p‖p) = 0

<details><summary>Indice 1</summary>

Écris les trois sommes sur le même ensemble $S$, puis utilise $\log \frac{a}{b} = \log a - \log b$.

</details>
<details><summary>Indice 2</summary>

$\sum_{i \in S} p_i \log_2 \frac{p_i}{q_i} = \sum_{i \in S} p_i \log_2 p_i - \sum_{i \in S} p_i \log_2 q_i$. Pour la question 4, $H(p)$ ne dépend pas de $q$. Pour la question 5, écris $-\mathrm{KL}(p \,\|\, q) = \sum_{i \in S} p_i \log_2 \frac{q_i}{p_i}$, passe en $\ln$ (un facteur $\frac{1}{\ln 2} > 0$ ne change pas le signe), puis applique l'inégalité terme à terme.

</details>
<details><summary>Indice 3</summary>

$\ln 2 \times \left(-\mathrm{KL}(p \,\|\, q)\right) = \sum_{i \in S} p_i \ln \frac{q_i}{p_i} \le \sum_{i \in S} p_i \left(\frac{q_i}{p_i} - 1\right) = \sum_{i \in S} q_i - \sum_{i \in S} p_i$. La première somme vaut au plus 1, la seconde exactement 1. Pour l'égalité, regarde quand chaque inégalité devient une égalité. Question 6 : le terme $-p_i \log_2 0$.

</details>

### Ex 6.8 — L'entropie d'une pièce est maximale à p = 1/2

<details><summary>Indice 1</summary>

Question 1 : remplace $p$ par $1 - p$ dans la formule. Question 2 : dérive terme à terme, avec la règle de la chaîne pour $(1 - p) \log_2 (1 - p)$.

</details>
<details><summary>Indice 2</summary>

La dérivée de $(1 - p) \log_2 (1 - p)$ par rapport à $p$ vaut $-\log_2 (1 - p) - \frac{1}{\ln 2}$ : les constantes $\frac{1}{\ln 2}$ se compensent. $\log_2 x > 0$ si et seulement si $x > 1$. Pour la question 6, $\log_2 \frac{p_i}{1/n} = \log_2 p_i + \log_2 n$.

</details>
<details><summary>Indice 3</summary>

$h'(p) = -\log_2 p + \log_2 (1 - p)$ ; $\frac{1 - p}{p} > 1$ si et seulement si $p < \frac{1}{2}$. $h''(p) = -\frac{1}{\ln 2}\left(\frac{1}{p} + \frac{1}{1 - p}\right)$. Question 6 : $\mathrm{KL}(p \,\|\, u) = \sum_i p_i \log_2 p_i + \log_2 n \sum_i p_i$, puis ∂ 6.7, question 5.

</details>

<a id="reflexion"></a>

## 🗣️ 🧮 📄 Réflexion

### Ex 6.9 — L'entropie expliquée avec un jeu de devinettes

<details><summary>Indice 1</summary>

Relis le paragraphe du §6.7 de la fiche sur le jeu de questions oui/non, et refais ✏️ 6.3 h).

</details>
<details><summary>Indice 2</summary>

Avec 8 nombres équiprobables, combien de questions faut-il si chacune coupe les possibilités en deux ? Si le 8 sort une fois sur deux, quelle première question poser ?

</details>
<details><summary>Indice 3</summary>

8 nombres équiprobables : 3 questions, à chaque partie. Si le 8 sort une fois sur deux, « Est-ce le 8 ? » d'abord : une fois sur deux, une seule question suffit ; sinon, il reste 7 nombres équiprobables. En moyenne, environ 2,4 questions. La limite : parmi 7 nombres équiprobables, aucune question ne coupe les chances en deux moitiés égales ; la meilleure stratégie fait alors un peu plus que l'entropie.

</details>

### Ex 6.10 — Fermi : combien de bits pour envoyer tout Holmes ?

<details><summary>Indice 1</summary>

Avance pas à pas : nombre de caractères × bits par caractère, pour chaque façon de coder.

</details>
<details><summary>Indice 2</summary>

1 octet = 8 bits, 1 ko = 1 000 octets. $2^6 = 64 < 90 \le 128 = 2^7$. Pour 5., le nombre de mots divisé par 25 donne des minutes ; le nombre de bits divisé par $10^8$ donne des secondes.

</details>
<details><summary>Indice 3</summary>

1. Environ 4,5 millions de bits. 2. 7 bits par caractère. 3. $560\,000 \times 4{,}5$. 4. $560\,000 \times 1$, à comparer à 1. 5. Environ 4 200 minutes ; quelques centièmes de seconde. 6. Ce que le contexte local (§6.2) et la connaissance de la langue permettent de deviner (fiche, au-delà du livre (2)).

</details>

### Ex 6.11 — Shannon (1948) : l'introduction et le schéma de communication

<details><summary>Indice 1</summary>

Tout est dans les deux premières pages : la figure 1, puis la liste numérotée de ses cinq éléments ; les trois raisons du logarithme sont numérotées aussi.

</details>
<details><summary>Indice 2</summary>

Pour la question 2, cherche la phrase où Shannon dit que le message est « choisi parmi un ensemble de messages possibles ». Pour la question 5, il donne la conversion entre bases ; tu peux la retrouver avec $\log_2 10$.

</details>
<details><summary>Indice 3</summary>

Source, émetteur, canal, récepteur, destinataire, et la source de bruit sur le canal. Le système doit fonctionner pour **tous** les messages possibles, puisqu'on ne sait pas à l'avance lequel sera envoyé. Les raisons : pratique, intuitive, mathématique. Le mot « bit » vient de J. W. Tukey ; $\log_2 2^N = N$. Un modèle de langage cherche à modéliser la source.

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

20 est meilleur que 30 seulement à protocole égal ; sinon, la comparaison ne veut rien dire. Et une perplexité plus basse ne garantit pas un modèle plus utile : elle mesure la prédiction du texte, pas l'exactitude des faits ni l'aide apportée.

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
_SUM_TOLERANCE = 1e-6


def _check_base(base):
    if not base > 0 or base == 1:
        raise ValueError(f"base must be > 0 and != 1, got {base!r}")


def _log(x, base):
    return np.log2(x) if base == 2 else np.log(x) / np.log(base)


def _as_distribution(p, name="p"):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError(f"{name} must contain finite probabilities >= 0")
    if abs(p.sum() - 1) > _SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to 1, got {p.sum()!r}")
    return p


def self_information(p, base=2.0):
    _check_base(base)
    arr = np.asarray(p, dtype=float)
    if np.any(~(arr > 0)) or np.any(arr > 1):
        raise ValueError(f"probabilities must be in (0, 1], got {p!r}")
    surprise = -_log(arr, base) + 0.0
    return float(surprise) if arr.ndim == 0 else surprise


def entropy(p, base=2.0):
    _check_base(base)
    arr = _as_distribution(p)
    present = arr[arr > 0]
    return float(-np.sum(present * _log(present, base))) + 0.0
```

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
    if not smoothing >= 0:
        raise ValueError(f"smoothing must be >= 0, got {smoothing!r}")
    counts = Counter(tokens)
    if vocabulary is None:
        vocab = sorted(counts)
    else:
        vocab = list(vocabulary)
        if len(set(vocab)) != len(vocab):
            raise ValueError("vocabulary contains a repeated token")
    weights = np.array([counts[token] for token in vocab], dtype=float) + smoothing
    total = weights.sum()
    if not total > 0:
        raise ValueError("no token of the vocabulary occurs and smoothing == 0")
    return vocab, weights / total


def char_distribution(text, alphabet=None, lowercase=True, smoothing=0.0):
    if lowercase:
        text = text.lower()
    vocabulary = None if alphabet is None else list(alphabet)
    return token_distribution(text, vocabulary=vocabulary, smoothing=smoothing)
```

</details>

### Ex 6.14 — Qui a l'entropie par lettre la plus haute : Holmes ou Verne ? 🔮

<details><summary>Indice 1</summary>

L'entropie est grande quand les lettres sont employées de façon équilibrée, petite quand quelques lettres concentrent la masse. Laquelle des deux langues concentre le plus ses lettres ? Et que se passe-t-il quand on ajoute des « lettres » à l'alphabet ?

</details>
<details><summary>Indice 2</summary>

Compare les fréquences des lettres des deux langues (ch. 1, 1.11) : laquelle met le plus de masse sur quelques lettres ? Sur 42 lettres, que deviennent les lettres accentuées de chaque texte ? Pour c), rappelle-toi la plus grande entropie possible sur 26 issues, et quand elle est atteinte.

</details>
<details><summary>Indice 3</summary>

Sur les 26 lettres, les lettres accentuées sont ignorées ; sur les 42, chacune devient une issue de plus, qui prend une part de la masse. En français, « e », « a », « s » pèsent plus lourd qu'en anglais, et « w », « k », « y » presque rien : la distribution est plus concentrée. Sur 42 lettres, « é » à lui seul fait près de 2 % des lettres de Verne, et plus de 3 % de ses lettres sont accentuées : la masse s'étale, l'entropie de Verne monte, celle de Holmes ne bouge presque pas. L'entropie de Holmes est un peu au-dessus de 4 bits.

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
    order = np.argsort(-np.asarray(p_holmes), kind="stable")
    y = np.arange(len(order))
    ax.barh(y - 0.2, np.asarray(p_holmes)[order], height=0.4, label="Holmes (English)")
    ax.barh(y + 0.2, np.asarray(p_verne)[order], height=0.4, label="Verne (French)")
    ax.set_yticks(y)
    ax.set_yticklabels([alphabet[i] for i in order])
    ax.invert_yaxis()
    ax.set(xlabel="probability", title="Letters of Holmes and Verne, sorted by their frequency in Holmes")
    ax.legend(loc="lower right")
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
def _two_distributions(p, q):
    p, q = _as_distribution(p, "p"), _as_distribution(q, "q")
    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape, got {p.shape} and {q.shape}")
    return p, q


def cross_entropy(p, q, base=2.0):
    _check_base(base)
    p, q = _two_distributions(p, q)
    support = p > 0
    if np.any(q[support] == 0):
        return float("inf")
    return float(-np.sum(p[support] * _log(q[support], base))) + 0.0


def kl_divergence(p, q, base=2.0):
    _check_base(base)
    p, q = _two_distributions(p, q)
    support = p > 0
    if np.any(q[support] == 0):
        return float("inf")
    ps, qs = p[support], q[support]
    value = float(np.sum(ps * (_log(ps, base) - _log(qs, base))))
    return value if value > 0 else 0.0


def js_divergence(p, q, base=2.0):
    _check_base(base)
    p, q = _two_distributions(p, q)
    m = (p + q) / 2
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)
```

</details>

### Ex 6.17 — La cross-entropy infinie : la lettre qui manque 🐛

<details><summary>Indice 1</summary>

Une cross-entropy infinie veut dire qu'une issue que $p$ produit reçoit la probabilité 0 dans $q$ (fiche §6.8). Compare `p_verne_17` et `q_holmes_17`, lettre par lettre.

</details>
<details><summary>Indice 2</summary>

Les lettres de a) : `p_verne_17[i] > 0 and q_holmes_17[i] == 0`, pour `i, c in enumerate(LETTERS_FR)`. Pour b), compte-les dans `verne.lower()` avec `collections.Counter`. Pour c), additionne leurs probabilités dans `p_verne_17`. Pour d), `char_distribution(..., smoothing=1)` ; essaie les trois endroits possibles, regarde ce qui donne une valeur finie, puis demande-toi, dans chaque cas, quel envoi tu mesures vraiment.

</details>
<details><summary>Indice 3</summary>

Lisser Verne laisse les zéros du code : toujours `inf`. Lisser les deux donne une valeur finie, mais pour un texte que personne n'envoie. Il faut lisser le **code**, et lui seul : `_, q = mylearn.info.char_distribution(holmes, alphabet=LETTERS_FR, smoothing=1)`, puis `mylearn.info.cross_entropy(p_verne_17, q)`.

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
ce_verne_holmes_18 = mylearn.info.cross_entropy(p_v, p_h)
ce_holmes_verne_18 = mylearn.info.cross_entropy(p_h, p_v)
kl_verne_holmes_18 = mylearn.info.kl_divergence(p_v, p_h)
kl_holmes_verne_18 = mylearn.info.kl_divergence(p_h, p_v)
js_18 = mylearn.info.js_divergence(p_h, p_v)
middle = len(holmes) // 2
_, first = mylearn.info.char_distribution(holmes[:middle], alphabet=LETTERS)
_, second = mylearn.info.char_distribution(holmes[middle:], alphabet=LETTERS)
kl_halves_18 = mylearn.info.kl_divergence(first, second)
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
h_chars_19 = scipy.stats.entropy(list(collections.Counter(holmes).values()), base=2)
zlib_bits_19 = 8 * len(zlib.compress(holmes.encode("utf-8"), 9)) / len(holmes)
shuffled = "".join(np.random.default_rng(19).permutation(list(holmes)))
shuffled_bits_19 = 8 * len(zlib.compress(shuffled.encode("utf-8"), 9)) / len(holmes)
zlib_wins_19 = zlib_bits_19 < h_chars_19
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
vocab_20 = round(float(np.exp(val_20[0])))
best = int(np.argmin(val_20))
best_step_20 = int(steps_20[best])
best_bits_20 = val_20[best] / np.log(2)
best_ppl_20 = np.exp(val_20[best])
ppl3_step_20 = int(steps_20[np.argmax(np.exp(val_20) < 3)])
last_ppl_20 = np.exp(val_20[-1])
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
    for ch in text:
        if ch in counts:
            counts[ch] += 1
    return np.array([counts[letter] for letter in LETTERS])


def count_counter_21(text):
    counts = collections.Counter(text)
    return np.array([counts[letter] for letter in LETTERS])


def count_str_21(text):
    return np.array([text.count(letter) for letter in LETTERS])


def count_numpy_21(text):
    codes = np.frombuffer(text.encode("ascii", "ignore"), dtype=np.uint8)
    return np.bincount(codes, minlength=123)[ord("a"):ord("z") + 1]
```

</details>

### Ex 6.22 — perplexity et log_loss 🔨

<details><summary>Indice 1</summary>

`perplexity` : contrôler, puis une formule d'une ligne. `log_loss` : contrôler, choisir la probabilité de la **vraie** classe de chaque exemple, la couper dans $[\varepsilon ; 1 - \varepsilon]$, puis la moyenne des $-\log$.

</details>
<details><summary>Indice 2</summary>

Perplexité : `np.exp(-np.mean(np.log(probs)))`, après avoir refusé une liste vide et toute valeur hors de $]0 ; 1]$ (avec `~(probs > 0)` pour attraper `NaN`). Log loss : en binaire (`y_prob` à une dimension), `np.where(labels == 1, probs, 1 - probs)` ; en multiclasse (deux dimensions), `probs[np.arange(n), labels]`. Les contrôles : longueurs égales, étiquettes entières dans le bon intervalle (0 ou 1 en binaire), probabilités dans $[0 ; 1]$ (`NaN` compris : `np.isfinite`), lignes de somme 1 (tolérance $10^{-6}$). `np.clip` renvoie une copie : tes arguments ne bougent pas.

</details>
<details><summary>Indice 3</summary>

```python
def perplexity(token_probs):
    probs = np.asarray(token_probs, dtype=float)
    if probs.size == 0:
        raise ValueError("token_probs is empty")
    if np.any(~(probs > 0)) or np.any(probs > 1):
        raise ValueError("every probability must be in (0, 1]")
    return float(np.exp(-np.mean(np.log(probs))))


def log_loss(y_true, y_prob, base=np.e):
    _check_base(base)
    labels = np.asarray(y_true)
    probs = np.asarray(y_prob, dtype=float)
    if labels.ndim != 1 or probs.ndim not in (1, 2) or probs.shape[0] != labels.shape[0] or labels.size == 0:
        raise ValueError("y_true must be 1-D, y_prob 1-D or 2-D, with the same non-zero length")
    if np.any(labels != np.round(labels)):
        raise ValueError("y_true must contain integer labels")
    labels = labels.astype(int)
    if not np.all(np.isfinite(probs)) or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("every probability must be in [0, 1]")
    if probs.ndim == 1:
        if np.any((labels < 0) | (labels > 1)):
            raise ValueError("binary labels must be 0 or 1")
        p_true = np.where(labels == 1, probs, 1.0 - probs)
    else:
        if np.any((labels < 0) | (labels >= probs.shape[1])):
            raise ValueError("a label is out of range")
        if np.any(np.abs(probs.sum(axis=1) - 1) > _SUM_TOLERANCE):
            raise ValueError("each row of y_prob must sum to 1")
        p_true = probs[np.arange(labels.size), labels]
    eps = np.finfo(float).eps
    p_true = np.clip(p_true, eps, 1 - eps)
    return float(-np.mean(_log(p_true, base)))
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
    symbols = list(symbols)
    weights = np.asarray(probs, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != len(symbols):
        raise ValueError("symbols and probs must have the same length")
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be non-empty and distinct")
    weights = _as_distribution(weights, "probs")
    if len(symbols) == 1:
        return {symbols[0]: "0"}
    codes = {symbol: "" for symbol in symbols}
    heap = [(float(w), i, [s]) for i, (s, w) in enumerate(zip(symbols, weights))]
    heapq.heapify(heap)
    order = len(heap)
    while len(heap) > 1:
        w0, _, group0 = heapq.heappop(heap)
        w1, _, group1 = heapq.heappop(heap)
        for s in group0:
            codes[s] = "0" + codes[s]
        for s in group1:
            codes[s] = "1" + codes[s]
        heapq.heappush(heap, (w0 + w1, order, group0 + group1))
        order += 1
    return codes


def huffman_encode(symbols, code):
    try:
        return "".join(code[s] for s in symbols)
    except KeyError as error:
        raise ValueError(f"the symbol {error.args[0]!r} has no codeword") from None


def huffman_decode(bits, code):
    if set(bits) - {"0", "1"}:
        raise ValueError("bits must contain only '0' and '1'")
    inverse = {word: symbol for symbol, word in code.items()}
    if len(inverse) != len(code) or any(not word or set(word) - {"0", "1"} for word in inverse):
        raise ValueError("codewords must be distinct non-empty strings of 0 and 1")
    prefixes = {word[:k] for word in inverse for k in range(1, len(word))}
    if prefixes & inverse.keys():
        raise ValueError("the code is not prefix-free")
    decoded, current = [], ""
    for bit in bits:
        current += bit
        if current in inverse:
            decoded.append(inverse[current])
            current = ""
        elif current not in prefixes:
            raise ValueError(f"the bits {current!r} match no codeword")
    if current:
        raise ValueError("bits end in the middle of a codeword")
    return decoded
```

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
n = len(holmes_letters_24)
dots = sum(len(MORSE[c]) for c in holmes_letters_24)
morse_24 = dots / n
morse_total_24 = dots + n - 1
_, p = mylearn.info.char_distribution(holmes_letters_24, alphabet=LETTERS)
code = mylearn.info.huffman_code(list(LETTERS), p)
huffman_24 = len(mylearn.info.huffman_encode(holmes_letters_24, code)) / n
gap_24 = huffman_24 - mylearn.info.entropy(p)
ratio_24 = huffman_24 / 5
morse_beats_entropy_24 = morse_24 < mylearn.info.entropy(p)
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

Tu as mesuré ces KL en 6.18 c) et d) : environ 0,2 bit dans un sens, 0,3 dans l'autre. L'entropie des lettres de Verne est autour de 4,1 bits : on reste bien sous 5 bits. Dans l'autre sens, les « h », « w », « y » de Holmes coûtent cher avec un code fait pour le français.

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
_, unigram = mylearn.info.char_distribution(train_26, alphabet=LETTERS, smoothing=1)
followers = collections.defaultdict(list)
for before, after in zip(train_26, train_26[1:]):
    followers[before].append(after)
bigram_26 = np.array([mylearn.info.token_distribution(followers[c], vocabulary=list(LETTERS), smoothing=1)[1]
                      for c in LETTERS])
idx = np.array([INDEX[c] for c in test_26])
uni_bits_26 = float(np.mean(mylearn.info.self_information(unigram[idx])))
bi_bits_26 = float(np.mean(mylearn.info.self_information(bigram_26[idx[:-1], idx[1:]])))
ppl_26 = [2 ** uni_bits_26, 2 ** bi_bits_26]
v = np.array([INDEX[c] for c in verne_letters_26])
bi_verne_26 = float(np.mean(mylearn.info.self_information(bigram_26[v[:-1], v[1:]])))
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
