# 6 · Théorie de l'information — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch06_information/06_mes_reponses.md` (créée par `python tools/start_chapter.py 6`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses des exercices ✏️ se vérifient dans la **partie 0** du notebook (`wb.check`) ; les quiz, les rappels, les exercices ∂ 🗣️ 🧮 📄 et les questions d'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée : si elle n'a pas de touche $\log_2$, utilise $\log_2 x = \frac{\ln x}{\ln 2}$.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 🧮 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, puis vérifie dans `05_solutions.md`.

### 6.Q1 — Information : sens courant, sens de Shannon 🧠 ⏱️ 3 min
*Fiche §6.1 · livre §6.1 · parcours R*

1. En une phrase : quel problème l'article de Shannon de 1948 cherchait-il à résoudre ?
2. Au sens courant, une information a un sens, une utilité, parfois une valeur de vérité. La mesure de Shannon tient-elle compte du **sens** d'un message ? De quoi dépend-elle ?
3. Vrai ou faux : pour Shannon, la phrase « deux plus deux font quatre », envoyée à un adulte, apporte beaucoup d'information.
4. Cite deux endroits du deep learning où les mesures de ce chapitre reviennent.
5. Vrai ou faux : la quantité d'information d'un même message peut changer d'un destinataire à l'autre. Justifie.

### 6.Q2 — Plus c'est rare, plus ça informe 🧠 ⏱️ 3 min
*Fiche §6.2, §6.4 · livre §6.2, §6.2.1 · parcours R*

1. Deux SMS d'un numéro inconnu : le premier commence par « Bonjour », le second par « Ornithorynque ». Lequel apporte le plus d'information au sens de Shannon, et pourquoi ?
2. Combien d'information apporte un événement certain ?
3. Et un événement de probabilité $\frac{1}{16}$, en bits ?
4. L'« échelle de surprise » de 0 à 100 du livre est-elle une mesure ? Par quoi la théorie la remplace-t-elle ?
5. Si la probabilité d'un événement est divisée par deux, de combien sa surprise augmente-t-elle ?

### 6.Q3 — Contexte global, contexte local 🧠 ⏱️ 3 min
*Fiche §6.2 · livre §6.2.2 · parcours R*

1. Définis le contexte global et le contexte local d'un mot, avec un exemple de chaque.
2. Dans « Je bois mon café sans… », pourquoi le mot « sucre » surprend-il peu ? Quel contexte joue ?
3. Quelle notion du ch. 4 capture une partie du contexte global ?
4. Le livre propose de transformer les surprises des mots en une pmf, puis de tirer des mots au hasard : « les plus surprenants sortiraient le plus souvent ». Qu'est-ce qui cloche dans cette idée ?
5. Un numéro de série tiré au hasard, comme « 7QX2LM9C » : le contexte local aide-t-il à deviner le caractère suivant ? Et dans une phrase en français ?

### 6.Q4 — Le bit est une unité 🧠 ⏱️ 3 min
*Fiche §6.3 · livre §6.3*

1. Quelle différence fais-tu entre un chiffre binaire (un 0 ou un 1 en mémoire) et un bit d'information ?
2. Une case mémoire de 8 chiffres binaires qui contient toujours `00000000` : combien de bits d'information t'apprend sa lecture ?
3. Combien de bits d'information apporte le résultat d'une pièce équilibrée ? Et celui d'une pièce qui tombe sur pile 99 fois sur 100 : plus ou moins, en moyenne ? Pourquoi ?
4. Cite une autre unité d'information que le bit, et la base de logarithme qui lui correspond.
5. Pourquoi le livre insiste-t-il : « une mémoire peut **contenir** 1 000 bits » plutôt que « une mémoire **est** de 1 000 bits » ?

### 6.Q5 — Les quatre propriétés de l'information 🧠 ⏱️ 3 min
*Fiche §6.4 · livre §6.4 · parcours M*

1. Énonce les quatre propriétés que le livre demande à la mesure de l'information.
2. Vérifie-les avec $I(p) = -\log_2 p$ pour $p = 1$, $\frac{1}{2}$ et $\frac{1}{4}$, puis pour deux événements indépendants de probabilités $\frac{1}{2}$ et $\frac{1}{16}$.
3. Dans « Passe-moi le classeur vert », pourquoi ne peut-on additionner les informations de « classeur » et de « vert » que si les deux mots sont indépendants ?
4. Dans « Il pleut, n'oublie pas ton parapluie », la somme des surprises des mots pris un par un est-elle plus grande ou plus petite que la surprise de la phrase entière ?
5. Quelle fonction transforme un produit en somme ? Pourquoi un signe « moins » dans $-\log_2 p$ ?

### 6.Q6 — Taille du vocabulaire et bits par mot 🧠 ⏱️ 3 min
*Fiche §6.5 · livre §6.5 · parcours R*

1. Avec un code de longueur fixe, combien de bits faut-il pour numéroter 2 mots ? 3 mots ? 1 000 mots ? 1 024 mots ? 1 025 mots ?
2. Écris la formule générale. Pourquoi l'arrondi à l'entier supérieur ?
3. Pourquoi est-il plus économique de numéroter seulement les mots du livre que l'on envoie, plutôt que tous les mots du dictionnaire ?
4. Que doivent partager l'émetteur et le récepteur avant l'envoi ? Quel contexte est-ce ?
5. Entre $\log_2 N$ et $\lceil \log_2 N \rceil$ : lequel est une quantité d'information, lequel une longueur de code ?

### 6.Q7 — Morse, Vail et les codes adaptatifs 🧠 ⏱️ 3 min
*Fiche §6.6 · livre §6.6*

1. Pourquoi le E est-il un seul point, et le T un seul trait ?
2. D'après le livre, comment Alfred Vail a-t-il estimé la fréquence des lettres en anglais ?
3. Qu'est-ce qu'un code de longueur fixe ? Un code adaptatif (à longueur variable) ?
4. En Morse, un silence sépare les lettres. Sans ces silences, trouve trois façons de lire `· · · −` (E = `·`, I = `· ·`, S = `· · ·`, T = `−`, A = `· −`, U = `· · −`, V = `· · · −`).
5. Qu'est-ce qu'un code préfixe ? Le Morse privé de ses silences en est-il un ?
6. Vrai ou faux : un code adaptatif envoie **n'importe quel** message en moins de symboles qu'un code fixe.

### 6.Q8 — Entropie nulle, entropie maximale 🧠 ⏱️ 3 min
*Fiche §6.7 · livre §6.7 · parcours R, M*

1. Définis l'entropie d'une distribution en une phrase, puis écris sa formule.
2. Quand vaut-elle 0 ? Donne deux exemples.
3. Pour 16 issues, quelle distribution a la plus grande entropie, et combien vaut-elle ?
4. Range par entropie croissante : $[0{,}5 ; 0{,}5]$, $[0{,}7 ; 0{,}3]$, $[\frac{1}{3} ; \frac{1}{3} ; \frac{1}{3}]$, $[1 ; 0]$.
5. Le livre écrit que l'entropie dépend « du message et de la distribution ». Corrige cette phrase.
6. En quoi l'entropie est-elle une **espérance** (ch. 2) ?

### 6.Q9 — Entropie et « organisation » : attention au faux ami 🧠 ⏱️ 3 min
*Fiche §6.7 · livre §6.7*

1. Le livre relie l'entropie à l'organisation d'un système : plus il y a de structure, plus il y aurait d'information. Une source très structurée (prévisible) a-t-elle une entropie élevée ou faible ?
2. Un texte de lettres tirées au hasard, uniformément, et un texte de Holmes : lequel a la plus grande entropie par lettre ? Lequel est le plus « organisé » ?
3. En physique, l'entropie mesure le « désordre ». Dans quel sens l'entropie de Shannon va-t-elle avec le désordre ?
4. Complète correctement : « L'entropie d'une source mesure… ».

### 6.Q10 — Le mauvais code coûte plus cher 🧠 ⏱️ 3 min
*Fiche §6.8 · livre §6.8, §6.8.1, §6.8.2 · parcours R*

1. Que mesure la cross-entropy $H(p, q)$, en une phrase ? Que représentent $p$ et $q$ ?
2. Pourquoi $H(p, q) \ge H(p)$ ? Quand y a-t-il égalité ?
3. Qu'est-ce que le taux de compression du livre ? Un bon code donne-t-il un taux proche de 0 ou de 1 ?
4. Que se passe-t-il si le code $q$ n'a rien prévu pour un mot qui apparaît dans le message ? Comment le livre contourne-t-il le problème ?
5. Dans la loss d'un classifieur, qui joue le rôle de $p$ et qui joue celui de $q$ ?

### 6.Q11 — KL : positive, asymétrique, nulle quand… ? 🧠 ⏱️ 3 min
*Fiche §6.9 · livre §6.9 · parcours R, M*

1. Écris $\mathrm{KL}(p \,\|\, q)$ comme la différence de deux quantités du chapitre. Que mesure-t-elle ?
2. Vrai ou faux : $\mathrm{KL}(p \,\|\, q) \ge 0$. Quand vaut-elle 0 ?
3. Vrai ou faux : $\mathrm{KL}(p \,\|\, q) = \mathrm{KL}(q \,\|\, p)$. Donne l'intuition avec deux livres.
4. Pourquoi parle-t-on de « divergence » et pas de « distance » ?
5. Le livre note « KL(Treasure Island‖Huckleberry Finn) » l'envoi de *Treasure Island* avec le code de *Huckleberry Finn*, puis décrit ce même envoi une seconde fois en écrivant « KL(Huckleberry Finn‖Treasure Island) ». Laquelle des deux notations est la bonne pour cet envoi ? À quel envoi correspond l'autre ? Qui est en premier : la distribution des données envoyées, ou celle du code ?
6. Cite une divergence symétrique et toujours finie.

### 6.Q12 — Bits, nats et la loss d'un LLM 🧠 ⏱️ 4 min
*Fiche §6.3, §6.8, au-delà du livre (1) · parcours R*

1. Quelle base de logarithme donne des bits ? des nats ? Combien de bits vaut un nat ?
2. En quelle unité PyTorch et scikit-learn calculent-ils la cross-entropy et la log loss ?
3. Un modèle de langage a une loss de 2,3 nats par token. Combien de bits par token (2 décimales) ? Quelle perplexité (à l'unité près) ?
4. Que veut dire, intuitivement, une perplexité de 10 ?
5. Pourquoi est-il trompeur de comparer les perplexités de deux modèles qui découpent le texte en tokens de façons différentes ?
6. Les tokens d'un LLM sont-ils des mots, des lettres, ou autre chose ?

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 6.R1 — Ch. 5 : un pas de descente de gradient à la main 🔁 ★ ⏱️ 5 min
*Ch. 5 (§5.3, §5.4) · parcours R, M*

Soit $f(w) = (w - 3)^2 + 1$. On part de $w_0 = 0$ avec un learning rate $\eta = 0{,}1$.
1. Calcule $f'(w)$, puis $f'(0)$.
2. Fais un pas de descente : que vaut $w_1$ ? Compare $f(w_0)$ et $f(w_1)$.
3. Fais deux pas de plus : $w_2$ et $w_3$. De quel facteur la distance au minimum est-elle multipliée à chaque pas ?
4. Quel learning rate atteint le minimum en un seul pas depuis $w_0$ ?
5. Quand on entraîne un réseau de neurones, quelle quantité joue le rôle de $f$, et quelles quantités jouent le rôle de $w$ ?

### 6.R2 — Ch. 3 : événements indépendants, P(A, B) = P(A) P(B) 🔁 ★ ⏱️ 5 min
*Ch. 3 (§3.4, §3.5) · parcours R, M*

1. Rappelle la définition de deux événements indépendants.
2. On tire une carte d'un jeu de 52 cartes, on la remet, on mélange, et l'on en tire une seconde. Probabilité de tirer deux fois l'as de pique ? Combien de bits d'information apporte ce résultat (3 décimales) ? Compare avec deux fois l'information d'un seul tirage.
3. On tire deux cartes **sans remise** d'un jeu de 52 cartes. Probabilité de tirer deux as ? Les deux tirages sont-ils indépendants ?
4. Montre que l'information de deux événements indépendants est la somme de leurs informations.
5. Dans un texte français, les lettres successives sont-elles indépendantes ? Donne un exemple qui le montre.

### 6.R3 — 0B : logarithmes, log₂ 8, log₂ ¼ et log(ab) 🔁 ★ ⏱️ 5 min
*0B (§101.2.4) · parcours R, M*

1. Calcule sans calculatrice $\log_2 8$, $\log_2 \frac{1}{4}$, $\log_2 1$ et $\log_2 1024$.
2. Complète : $\log_2(ab) = \ldots$, $\log_2 \frac{a}{b} = \ldots$, $\log_2(a^k) = \ldots$
3. Avec la formule de changement de base, calcule $\log_2 5$ (3 décimales).
4. Même question pour $\log_2 12$, avec seulement la touche $\ln$.
5. Quel est le signe de $\log_2 p$ pour $0 < p < 1$ ? Que vaut $\log_2 p$ pour $p = 1$ ?

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats ✏️ dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 6.1 — Combien de bits pour une pièce, un dé, une lettre E ? ✏️ ★ ⏱️ 10 min
**Objectif :** calculer l'information d'un événement en bits et en nats, et l'additionner pour des événements indépendants.
**Prérequis :** 0B (§101.2.4) · fiche §6.4 · **Parcours :** R, M

L'information (la surprise) d'un événement de probabilité $p$ vaut $I = -\log_2 p$ bits, ou $-\ln p$ nats.

a) Combien de bits apporte « pile » avec une pièce équilibrée ?
b) Et le résultat « 6 » avec un dé équilibré à six faces (3 décimales) ?
c) Dans un texte, une lettre tirée au hasard est un « e » avec la probabilité $\frac{1}{8}$. Combien de bits apporte un « e » ?
d) Un « z » a la probabilité $\frac{1}{1024}$. Combien de bits ?
e) On tire deux lettres indépendamment : un « e », puis un « z ». Combien de bits en tout ?
f) Trois dés équilibrés donnent (5, 2, 6). Combien de bits apporte ce résultat (3 décimales) ?
g) Le résultat « 6 » de b), en **nats** (3 décimales).
h) Pour écrire le résultat d'un dé avec un code binaire de longueur fixe (une suite de 0 et de 1, toujours de la même longueur), combien de chiffres binaires faut-il au minimum ?
i) Une pièce truquée tombe sur pile 9 fois sur 10. Combien de bits apporte « face » (3 décimales) ?
j) Pourquoi la réponse de b) n'est-elle pas entière, alors que celle de h) l'est ? (réponds dans ta copie)

### Ex 6.2 — Bits par mot : Seuss, Holmes et l'alphabet ✏️ ★ ⏱️ 10 min
**Objectif :** relier la taille d'un vocabulaire au nombre de bits d'un code de longueur fixe.
**Prérequis :** Ex 6.1 · fiche §6.5 · **Parcours :** M

On numérote les mots d'un vocabulaire de $N$ mots différents, et l'on envoie chaque mot par son numéro écrit en binaire, **toujours avec le même nombre de chiffres**.

a) Combien de bits par mot pour un vocabulaire d'exactement 64 mots ?
b) Et pour 65 mots ?
c) *The Cat in the Hat*, du Dr Seuss, utilise 236 mots différents. Combien de bits par mot ?
d) Le recueil *The Adventures of Sherlock Holmes* (notre fil rouge) utilise 7 819 mots différents (en comptant les mots comme dans le notebook). Combien de bits par mot ?
e) Combien de bits par symbole pour les 26 lettres, l'espace et les 10 chiffres ?
f) Le tokenizer de GPT-2 a un vocabulaire de 50 257 tokens. Combien de bits par token ?
g) Holmes compte 105 849 mots. Combien de bits pour l'envoyer en entier avec le code de d) ?
h) Quel est le plus grand vocabulaire que l'on peut numéroter avec 14 bits ?
i) Le Dr Seuss a écrit *Green Eggs and Ham* avec seulement 50 mots différents. Combien de bits par mot aurait-il économisé par rapport à *The Cat in the Hat* ? (réponds dans ta copie)

### Ex 6.3 — Entropie de quelques distributions ✏️ ★ ⏱️ 15 min
**Objectif :** calculer l'entropie d'une distribution et reconnaître ses valeurs extrêmes.
**Prérequis :** Ex 6.1 · fiche §6.7 · **Parcours :** R, M

L'entropie d'une distribution $p = (p_1, \ldots, p_n)$ est $H(p) = -\sum_i p_i \log_2 p_i$, avec la convention $0 \log_2 0 = 0$.

a) $H$ de $[\frac{1}{2} ; \frac{1}{4} ; \frac{1}{8} ; \frac{1}{8}]$ (valeur exacte).
b) $H$ de la distribution uniforme sur 8 issues.
c) $H$ de $[0{,}9 ; 0{,}1]$ (3 décimales).
d) $H$ de $[\frac{1}{2} ; \frac{1}{2} ; 0]$. Et celle de $[1 ; 0 ; 0]$ ? (la seconde dans ta copie)
e) L'entropie de a), en **nats** (3 décimales).
f) La plus grande entropie possible pour une distribution sur 4 issues.
g) Un dé truqué donne 6 une fois sur deux, et chacune des cinq autres faces avec la probabilité $\frac{1}{10}$. Son entropie (3 décimales) ?
h) Pour a), propose une stratégie de questions oui/non pour deviner l'issue tirée, et calcule le nombre moyen de questions. Que remarques-tu ? (réponds dans ta copie)

### Ex 6.4 — Morse contre code fixe : SHERLOCK HOLMES ✏️ ★★ ⏱️ 20 min
**Objectif :** compter le coût d'un message avec un code fixe et avec le Morse, et voir ce que coûtent les silences.
**Prérequis :** Ex 6.2 · fiche §6.6 · **Parcours :** M

Le code Morse international des 26 lettres (point `·`, trait `−`) :

| A `·−` | B `−···` | C `−·−·` | D `−··` | E `·` | F `··−·` | G `−−·` | H `····` | I `··` |
|---|---|---|---|---|---|---|---|---|
| **J** `·−−−` | **K** `−·−` | **L** `·−··` | **M** `−−` | **N** `−·` | **O** `−−−` | **P** `·−−·` | **Q** `−−·−` | **R** `·−·` |
| **S** `···` | **T** `−` | **U** `··−` | **V** `···−` | **W** `·−−` | **X** `−··−` | **Y** `−·−−` | **Z** `−−··` | |

Comme le livre, on donne la même durée au point et au trait (deux tons différents), on ignore l'espace entre les mots, et l'on compare avec un **code fixe** qui donne 5 symboles à chaque lettre. Le message : SHERLOCK HOLMES.

a) Combien de symboles avec le code fixe ?
b) Combien de points et de traits en Morse ?
c) Le rapport b) / a) (2 décimales).
d) Pour être lu sans ambiguïté, le Morse a besoin d'un silence entre deux lettres qui se suivent (y compris entre K et H, puisqu'on ignore l'espace). En comptant chaque silence comme un symbole, combien de symboles en tout ?
e) Le rapport d) / a) (2 décimales).
f) Sans aucun silence, en combien de façons peut-on lire `····` (avec E, I, S et H) ?
g) Le Morse **sans** ses silences est-il un code préfixe ? (`True` ou `False`)
h) Pourquoi la comparaison « points et traits » du livre (son exemple donne un rapport d'environ 0,5) est-elle trop favorable au Morse ? Le code fixe a-t-il besoin de silences ? (réponds dans ta copie)

### Ex 6.5 — Cross-entropy et KL dans les deux sens ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer une cross-entropy et une divergence KL, et constater que la KL n'est pas symétrique.
**Prérequis :** Ex 6.3 · fiche §6.8, §6.9 · **Parcours :** R, M

Une pièce tombe sur pile avec la probabilité 0,8 : $p = [0{,}8 ; 0{,}2]$. On l'encode avec un code prévu pour une pièce équilibrée : $q = [0{,}5 ; 0{,}5]$. En bits :

a) $H(p)$ (3 décimales).
b) $H(p, q) = -\sum_i p_i \log_2 q_i$.
c) $\mathrm{KL}(p \,\|\, q)$ (3 décimales).
d) Dans l'autre sens : $H(q, p)$ (3 décimales).
e) $\mathrm{KL}(q \,\|\, p)$ (3 décimales).
f) A-t-on $\mathrm{KL}(p \,\|\, q) < \mathrm{KL}(q \,\|\, p)$ ? (`True` ou `False`)
g) Un modèle sûr de lui annonce $r = [1 ; 0]$ : pile à coup sûr. $H(p, r)$ est-elle finie ? (`True` ou `False`)
h) Vérifie sur a) à c) la relation $H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q)$, et explique g) avec la formule. (réponds dans ta copie)

### Ex 6.6 — Un code de Huffman à la main ✏️ ★★ ⏱️ 25 min
**Objectif :** construire un code de Huffman, calculer sa longueur moyenne et la comparer à l'entropie et au code fixe.
**Prérequis :** Ex 6.3, Ex 6.4 · fiche §6.6, §6.7 · **Parcours :** M

Une station météo envoie chaque heure l'un de cinq états, avec ces probabilités :

| soleil | nuages | pluie | vent | neige |
|---|---|---|---|---|
| 0,40 | 0,25 | 0,15 | 0,12 | 0,08 |

a) Combien de bits par état avec un code de longueur fixe ?
b) Construis le code de Huffman : à chaque étape, fusionne les deux groupes **les moins probables**. Donne la liste des longueurs des mots de code, dans l'ordre [soleil, nuages, pluie, vent, neige].
c) La longueur moyenne du code, en bits par état.
d) L'entropie de la distribution (3 décimales).
e) La somme de Kraft $\sum_i 2^{-\ell_i}$, où $\ell_i$ est la longueur du mot de code $i$.
f) Combien de bits pour le message « soleil, soleil, pluie, neige, nuages » ?
g) Le rapport c) / a) (3 décimales).
h) Écris un code possible (des 0 et des 1), vérifie qu'aucun mot n'est le début d'un autre, puis décode `0110` avec ton code. Le code est-il unique ? (réponds dans ta copie)

### Ex 6.7 — H(p, q) = H(p) + KL(p‖q), et KL(p‖p) = 0 ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer la relation entre cross-entropy, entropie et KL, et en tirer la conséquence pour l'entraînement d'un modèle.
**Prérequis :** Ex 6.5 · fiche §6.8, §6.9 · **Parcours :** M

On note $S$ l'ensemble des issues où $p_i > 0$, et l'on suppose $q_i > 0$ sur $S$.
1. Écris les définitions de $H(p)$, $H(p, q)$ et $\mathrm{KL}(p \,\|\, q) = \sum_{i \in S} p_i \log_2 \frac{p_i}{q_i}$, en sommant sur $S$.
2. Démontre que $H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q)$.
3. Déduis-en que $\mathrm{KL}(p \,\|\, p) = 0$ et $H(p, p) = H(p)$.
4. À l'entraînement, $p$ (la distribution des données) est fixée et l'on fait varier le modèle $q$. Montre que minimiser $H(p, q)$ revient à minimiser $\mathrm{KL}(p \,\|\, q)$. Pourquoi est-ce rassurant ?
5. (★) On admet que $\ln x \le x - 1$ pour tout $x > 0$, avec égalité seulement en $x = 1$. Montre que $\mathrm{KL}(p \,\|\, q) \ge 0$ (applique l'inégalité à $x = \frac{q_i}{p_i}$). Quand y a-t-il égalité ?
6. Que devient $H(p, q)$ si $q_i = 0$ pour une issue de $S$ ?

### Ex 6.8 — L'entropie d'une pièce est maximale à p = 1/2 ∂ ★★★ ⏱️ 30 min
**Objectif :** étudier l'entropie d'une pièce comme une fonction de $p$, et montrer que la loi uniforme maximise l'entropie.
**Prérequis :** Ex 6.3 · ∂ 6.7 (pour la question 6) · ch. 5 (dérivée, maximum) · fiche §6.7, §6.9 · **Parcours :** M

Pour une pièce qui tombe sur pile avec la probabilité $p \in \,]0 ; 1[$, l'entropie vaut $h(p) = -p \log_2 p - (1 - p) \log_2 (1 - p)$.
1. Montre que $h(1 - p) = h(p)$. Que signifie cette symétrie ?
2. Sachant que la dérivée de $x \log_2 x$ est $\log_2 x + \frac{1}{\ln 2}$, montre que $h'(p) = \log_2 \frac{1 - p}{p}$.
3. Étudie le signe de $h'$ sur $]0 ; \frac{1}{2}[$ et sur $]\frac{1}{2} ; 1[$. Conclus : où $h$ est-elle maximale, et que vaut son maximum ?
4. On admet que $x \ln x$ tend vers 0 quand $x$ tend vers 0. Vers quoi tend $h(p)$ quand $p$ tend vers 0 ? vers 1 ? Pourquoi la convention $0 \log_2 0 = 0$ est-elle naturelle ?
5. Calcule $h''(p)$ et montre que $h''(p) < 0$. Qu'en déduis-tu sur la forme de la courbe ?
6. Généralisation : pour $n$ issues, $u$ désigne la distribution uniforme. Montre que $\mathrm{KL}(p \,\|\, u) = \log_2 n - H(p)$, puis, avec ∂ 6.7, que $H(p) \le \log_2 n$.

<a id="reflexion"></a>

## 🗣️ 🧮 📄 Réflexion

### Ex 6.9 — L'entropie expliquée avec un jeu de devinettes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer l'entropie à un débutant avec un jeu de questions oui/non.
**Prérequis :** fiche §6.7 · **Parcours :** R

Un ami joue à « devine le nombre » : tu en choisis un entre 1 et 8, il pose des questions auxquelles tu réponds par oui ou non. Explique-lui ce qu'est l'entropie, en **cinq lignes au plus**. Contraintes :
- les mots « question oui/non », « en moyenne » et « probable » ;
- ce qui change si tu choisis le 8 une fois sur deux, les sept autres nombres restant équiprobables ;
- une limite de l'image (on ne peut pas toujours couper les chances en deux moitiés égales) ;
- aucune formule.

Relis-toi à voix haute, puis compare avec la réponse modèle de `05_solutions.md`.

### Ex 6.10 — Fermi : combien de bits pour envoyer tout Holmes ? 🧮 ★★ ⏱️ 20 min
**Objectif :** estimer la taille d'un texte selon la façon de le coder, du code fixe à la prédiction.
**Prérequis :** Ex 6.3 · fiche §6.5, §6.7 et au-delà du livre (2) · **Fil rouge :** Holmes · **Parcours :** M

Données : Holmes compte environ 560 000 caractères (lettres, espaces, ponctuation, retours à la ligne) et environ 106 000 mots. Donne des ordres de grandeur, en justifiant chaque étape.
1. En ASCII, un caractère occupe un octet (8 bits). Combien de bits pour tout le texte ? Combien de kilo-octets ?
2. Le texte utilise un peu moins de 90 caractères différents (minuscules, majuscules, chiffres, ponctuation…). Avec un code de longueur fixe adapté, combien de bits par caractère, et combien en tout ?
3. Un code de Huffman construit sur les caractères de Holmes tombe à environ 4,5 bits par caractère. Combien en tout ?
4. Shannon (1951) estimait qu'un lecteur anglophone, qui tient compte d'un long contexte (jusqu'à 100 lettres), ne « dépense » qu'environ 1 bit par lettre. Combien en tout ? Combien de fois moins qu'en 1. ?
5. Un télégraphiste expérimenté envoie environ 25 mots par minute. Combien d'heures pour transmettre Holmes ? Et avec une connexion à 100 mégabits par seconde, pour le fichier de 1. ?
6. Qu'est-ce qui permet de passer de 2. à 4. ? Où, dans ce chapitre, vois-tu ce phénomène à l'œuvre ?

### Ex 6.11 — Shannon (1948) : l'introduction et le schéma de communication 📄 ★★ ⏱️ 30 min
**Objectif :** lire l'introduction de l'article fondateur de la théorie de l'information, et relier son schéma au machine learning.
**Prérequis :** Ex 6.1 · fiche §6.1, §6.3, §6.4 · **Parcours :** complet seulement (lecture conseillée à tous)

L'article : C. E. Shannon, « A Mathematical Theory of Communication », *Bell System Technical Journal*, vol. 27, 1948 (le livre écrit à tort *Bell Labs Technical Journal*). Une réimpression est en accès libre : [PDF hébergé par l'université Harvard](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf). Lis l'introduction (les deux premières pages) et regarde sa figure 1.

1. Nomme les cinq éléments du système de communication de la figure 1, et la source de bruit. Associe-les à l'envoi d'un SMS : qui est la source, l'émetteur, le canal… ?
2. Pourquoi Shannon écrit-il que les aspects sémantiques (le sens) n'intéressent pas l'ingénieur ? Qu'est-ce qui compte à la place ?
3. Quelles sont les trois raisons qu'il donne pour préférer une mesure logarithmique ?
4. Qui a proposé le mot « bit » ? Combien de bits peuvent stocker $N$ relais à deux positions ? Pourquoi est-ce cohérent avec le logarithme ?
5. Shannon parle aussi d'unités en base 10 (les chiffres décimaux) et en base $e$ (les unités « naturelles », nos nats). Combien de bits vaut un chiffre décimal (2 décimales) ?
6. Quel élément du schéma un modèle de langage cherche-t-il, en un sens, à modéliser ? Pourquoi un bon modèle de la source permet-il de mieux comprimer ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 6.E1 — Pourquoi la cross-entropy comme loss de classification ? 💼 ★★ ⏱️ 10 min
*Fiche §6.8, au-delà du livre (1) · prérequis 6.22 · parcours R*

« Pourquoi entraîne-t-on presque tous les classifieurs avec la cross-entropy, plutôt qu'avec l'erreur quadratique ou l'accuracy ? »

### 6.E2 — Entropie, cross-entropy, KL : les différences 💼 ★★ ⏱️ 10 min
*Fiche §6.7 à §6.9 · prérequis 6.16 · parcours R*

« Expliquez la différence entre entropie, cross-entropy et divergence KL, et comment elles sont liées. »

### 6.E3 — La perplexité d'un modèle de langage 💼 ★★ ⏱️ 10 min
*Fiche, au-delà du livre (1) · prérequis 6.22 · parcours R*

« Qu'est-ce que la perplexité d'un modèle de langage ? Un modèle de perplexité 20 est-il meilleur qu'un modèle de perplexité 30 ? »

### 6.E4 — Où rencontre-t-on la divergence KL en ML moderne ? 💼 ★★ ⏱️ 10 min
*Fiche §6.9 (🕰️ la KL aujourd'hui) · prérequis 6.16 · parcours R*

« Où utilise-t-on la divergence KL en machine learning aujourd'hui ? Pourquoi dans ce sens-là et pas dans l'autre ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch06_information/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie `mylearn/info.py`. La partie 0 du notebook vérifie tes réponses aux exercices ✏️ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 6.12 | self_information et entropy | 🔨 | ★ | 15 |
| 6.13 | Distributions de caractères et de mots | 🔨 | ★★ | 20 |
| 6.14 | Qui a l'entropie par lettre la plus haute : Holmes ou Verne ? | 🔮 | ★★ | 15 |
| 6.15 | Fréquences des lettres en anglais et en français | 🎨 | ★★ | 20 |
| 6.16 | cross_entropy, kl_divergence et js_divergence | 🔨 | ★★ | 25 |
| 6.17 | La cross-entropy infinie : la lettre qui manque | 🐛 | ★★ | 20 |
| 6.18 | Coder le français avec le code de l'anglais, et l'inverse | 🔬 | ★★ | 25 |
| 6.19 | scipy.stats.entropy et un vrai compresseur (zlib) | 📦 | ★★ | 20 |
| 6.20 | Lire une courbe de loss : nats, bits et perplexité | 📈 | ★★ | 20 |
| 6.21 | Mesurer avant d'optimiser : compter des caractères vite | 🛠️ | ★★ | 20 |
| 6.22 | perplexity et log_loss | 🔨 | ★★ | 30 |
| 6.23 | Huffman : construire, encoder, décoder | 🔨 | ★★★ | 45 |
| 6.24 | Compresser Holmes : code fixe, Morse, Huffman et entropie | 🔬 | ★★★ | 30 |
| 6.25 | Le code de Huffman de Holmes pour envoyer Verne | 🔮 | ★★★ | 30 |
| 6.26 | Le contexte local réduit la surprise : les bigrammes | 🔬 | ★★★ | 40 |
| 6.27 | Passer sous la barre de Huffman lettre à lettre | 🏆 | ★★★ | 60 |
