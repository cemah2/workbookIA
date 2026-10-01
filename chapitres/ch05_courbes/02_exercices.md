# 5 · Courbes et surfaces — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch05_courbes/06_mes_reponses.md` (créée par `python tools/start_chapter.py 5`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses des exercices ✏️ se vérifient dans la **partie 0** du notebook (`wb.check`) ; les quiz, les rappels, les exercices ∂ 🗣️ 🧮 📄 et les questions d'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ 🧮 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, puis vérifie dans `05_solutions.md`.

### 5.Q1 — Pourquoi dérivée et gradient sont au cœur de l'apprentissage 🧠 ⏱️ 3 min
*Fiche §5.1 · livre §5.1 · parcours R*

1. Que nous apprennent la dérivée et le gradient sur une courbe ou une surface ?
2. Quand on entraîne un réseau, quelle « surface » cherche-t-on à descendre ? Quelles sont ses entrées et sa sortie ?
3. Quel algorithme, détaillé au ch. 18, calcule le gradient pour un réseau de neurones ?
4. Vrai ou faux : ce chapitre du livre donne les formules de calcul des dérivées.
5. Pourquoi ne pas chercher les meilleurs poids en essayant des valeurs au hasard ?

### 5.Q2 — Une fonction, une table d'entrées et de sorties 🧠 ⏱️ 3 min
*Fiche §5.2 · livre §5.2*

1. Combien de nombres en entrée, et combien en sortie, pour une courbe ? Pour une surface ?
2. Un modèle prédit le prix d'un appartement à partir de 8 features. Sa fonction est-elle une courbe, une surface, ou autre chose ?
3. La loss d'un réseau à 1 million de poids, calculée sur un dataset fixé : combien d'entrées, combien de sorties ?
4. Vrai ou faux : une fonction renvoie toujours la même sortie pour les mêmes entrées. Cite un cas, en entraînement, où l'on introduit du hasard volontairement.
5. Pourquoi le livre parle-t-il d'une table « potentiellement infinie » ?

### 5.Q3 — Continue, lisse, univoque : reconnaître les courbes 🧠 ⏱️ 3 min
*Fiche §5.2 · livre §5.2, §5.3 · parcours R, M*

Pour chaque courbe, dis quelle règle du livre elle enfreint (continuité, absence de point anguleux, une seule valeur par abscisse, pas de tangente verticale), ou si elle les respecte toutes.
1. La partie entière, $y = \lfloor x \rfloor$ (le plus grand entier inférieur ou égal à $x$).
2. $y = |x - 1|$.
3. Le cercle $x^2 + y^2 = 4$.
4. $y = \sqrt[3]{x}$ (racine cubique), autour de $x = 0$.
5. $y = x^2 + \sin x$.
6. Pourquoi le livre tient-il à ces règles ? Que deviendrait l'algorithme qui suit la pente sans elles ?

### 5.Q4 — Minimum local ou minimum global ? 🧠 ⏱️ 3 min
*Fiche §5.3 · livre §5.3 · parcours R*

1. $f(x) = \cos x$ sur tous les réels : que vaut son maximum global ? En combien de points est-il atteint ?
2. Le livre écrit qu'il n'y a « qu'un seul » maximum global. Corrige cette phrase.
3. $f(x) = x^3$ sur tous les réels : a-t-elle un minimum global ? Un minimum local ?
4. $f(x) = (x^2 - 4)^2$ : où sont ses minima globaux, et que valent-ils ? A-t-elle un maximum local ?
5. Sur l'intervalle $[0 ; 3]$, $f(x) = x^2 - 2x$ : où sont son minimum global et son maximum global ?
6. Une descente qui s'arrête dans un creux a-t-elle trouvé le minimum global ?

### 5.Q5 — Le signe de la dérivée indique le chemin 🧠 ⏱️ 3 min
*Fiche §5.3 · livre §5.3 · parcours R, M*

1. En $x = 2$, $f'(2) = -3$. Pour trouver des valeurs **plus grandes** de $f$, faut-il aller à droite ou à gauche ?
2. Même question pour des valeurs plus petites.
3. Que valent $\mathrm{sign}(-0{,}001)$, $\mathrm{sign}(0)$ et $\mathrm{sign}(42)$ ?
4. Avec la règle « un pas fixe dans le sens du signe de la dérivée », pourquoi risque-t-on de tourner autour d'un maximum sans jamais s'y poser ?
5. Pourquoi la descente de gradient, qui fait des pas proportionnels à la dérivée, ralentit-elle d'elle-même près d'un minimum ?
6. Vrai ou faux : une dérivée grande en valeur absolue signale une pente raide.

### 5.Q6 — Dérivée nulle : sommet, creux ou plateau ? 🧠 ⏱️ 3 min
*Fiche §5.3 · livre §5.3 · parcours M*

Les quatre fonctions suivantes ont une dérivée nulle en $x = 0$. Pour chacune : maximum, minimum, ni l'un ni l'autre (ou les deux ?)
1. $f(x) = -x^2$.
2. $f(x) = x^4$.
3. $f(x) = x^3$.
4. $f(x) = 5$ (constante).
5. Que vaut $f''(0)$ pour chacune des trois premières ? Dans quels cas la dérivée seconde permet-elle de conclure ?
6. Le livre appelle *plateau* tout point de pente nulle qui n'est ni un sommet ni un creux. Comment appelle-t-on, en mathématiques, un tel point isolé, comme celui de la question 3 ?

### 5.Q7 — Le gradient : une direction et une longueur 🧠 ⏱️ 3 min
*Fiche §5.4 · livre §5.4 · parcours R, M*

Soit $f(x, y) = 3x - 4y + 1$.
1. Que vaut $\nabla f$, en tout point ?
2. Que vaut sa norme ?
3. Quel vecteur unitaire indique la plus grande montée ?
4. Quelle est la pente de $f$ si l'on se déplace dans la direction $(0, 1)$ ?
5. Donne un vecteur unitaire dans la direction duquel $f$ ne change pas. Que représente cette direction sur la carte des lignes de niveau ?
6. Vrai ou faux : pour descendre le plus vite, on se déplace dans le sens du gradient.

### 5.Q8 — Point selle : un gradient nul sans extremum 🧠 ⏱️ 3 min
*Fiche §5.4 · livre §5.4 · parcours R*

1. Qu'est-ce qu'un point selle ?
2. $f(x, y) = y^2 - x^2$ en $(0, 0)$ : que vaut le gradient ? Que fait $f$ le long de l'axe des $x$ ? Le long de l'axe des $y$ ?
3. Vrai ou faux : $f(x, y) = xy$ n'a pas de point selle en $(0, 0)$, puisqu'elle est plate le long des deux axes.
4. Pour $y^2 - x^2$, dans quelles directions faut-il quitter $(0, 0)$ pour descendre ? Le gradient en $(0, 0)$ l'indique-t-il ?
5. D'après la fiche (🕰️ points selles), pourquoi parle-t-on davantage des points selles que des minima locaux pour les réseaux de neurones ?
6. Existe-t-il des points selles pour une fonction d'une seule variable ? Quel est l'équivalent le plus proche ?

### 5.Q9 — La fonction max(0, x) a un coin en 0 : est-ce grave ? 🧠 ⏱️ 3 min
*Fiche §5.2, §5.3 · livre §5.2, §5.3 · parcours M*

On note $\mathrm{relu}(x) = \max(0, x)$, la fonction d'activation ReLU (ch. 17).
1. Que vaut sa pente juste à gauche de 0 ? Juste à droite ?
2. Quelle règle du livre ReLU enfreint-elle en 0 ?
3. Que donne la pente centrée $\frac{\mathrm{relu}(h) - \mathrm{relu}(-h)}{2h}$ en 0, pour tout $h > 0$ ?
4. Quelle valeur PyTorch donne-t-il à la dérivée de `torch.relu` en 0, et selon quelle règle ?
5. Ce choix compte-t-il pour l'entraînement d'un réseau ? Que répond la théorie, et qu'ont observé Bertoin et coll. (2021) ?
6. Cite une autre fonction anguleuse utilisée en machine learning.

### 5.Q10 — Ce que fait un réseau quand il « descend le gradient » 🧠 ⏱️ 3 min
*Fiche §5.1, §5.4 · livre §5.1, §5.4 · parcours R*

1. Dans l'image du paysage, quelles sont les « coordonnées » du point qui se déplace ? Quelle est son « altitude » ?
2. Écris la mise à jour d'un poids $w$ avec le learning rate $\eta$ et la dérivée partielle $\frac{\partial L}{\partial w}$.
3. Pourquoi n'estime-t-on pas ce gradient par des différences finies pour un réseau d'un million de poids ?
4. Que se passe-t-il quand le gradient devient presque nul, alors que la loss est encore élevée ?
5. Vrai ou faux : la descente de gradient garantit de trouver le minimum global de la loss.
6. Quel chapitre du livre détaille la façon de calculer ce gradient ? Lequel compare les façons de descendre ?

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 5.R1 — Ch. 4 : mettre à jour un prior après une observation 🔁 ★ ⏱️ 5 min
*Ch. 4 (§4.4, §4.5) · parcours R*

Une usine a deux machines : A fabrique 60 % des pièces, dont 2 % sont défectueuses ; B en fabrique 40 %, dont 5 % sont défectueuses. On prélève une pièce au hasard : elle est défectueuse.
1. Quelle est la probabilité qu'une pièce prise au hasard soit défectueuse ? Quel est le nom de ce terme dans la règle de Bayes ?
2. Quelle est la probabilité que la pièce défectueuse vienne de B (3 décimales) ?
3. Si B fabriquait 50 % des pièces, cette probabilité monterait-elle ou baisserait-elle ? Calcule-la.
4. Quel posterior obtiendrait-on pour B avec un prior de 0 ? Quelle leçon du ch. 4 retrouves-tu ?

### 5.R2 — Ch. 2 : où la courbe en cloche atteint-elle son maximum ? 🔁 ★ ⏱️ 5 min
*Ch. 2 (loi normale) · 0B (règle de la chaîne) · parcours R, M*

La densité de la loi normale de moyenne $\mu$ et d'écart-type $\sigma$ est $f(x) = \frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x - \mu)^2}{2\sigma^2}}$.
1. Calcule $f'(x)$ avec la règle de la chaîne.
2. En quel $x$ la dérivée s'annule-t-elle ? Quel est son signe avant et après ? Conclus.
3. Que vaut le maximum de $f$ pour $\sigma = 2$ (3 décimales) ?
4. Que devient la hauteur de ce maximum si l'on double $\sigma$ ?
5. La courbure de la cloche change de signe en deux points (la dérivée seconde s'y annule). Lesquels ? (Tu peux le vérifier en dérivant $f'$, ou le retrouver sur une figure du ch. 2.)

### 5.R3 — 0B : règle de la chaîne, dériver (3x + 1)² 🔁 ★ ⏱️ 5 min
*0B (§101.5.3) · parcours R, M*

Soit $g(x) = (3x + 1)^2$.
1. Dérive $g$ avec la règle de la chaîne.
2. Vérifie en développant d'abord $g(x)$.
3. Que vaut $g'(1)$ ?
4. Où $g'$ s'annule-t-elle ? Que vaut $g$ en ce point ? Est-ce un minimum ou un maximum ?
5. Fais un pas de descente de gradient depuis $x = 0$ avec $\eta = 0{,}05$ : où arrives-tu ? La valeur de $g$ a-t-elle baissé ?

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats ✏️ dans la partie 0 du notebook. Garde les valeurs exactes (des fractions) pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 5.1 — La sécante qui se resserre sur la tangente ✏️ ★ ⏱️ 10 min
**Objectif :** approcher une dérivée par des sécantes symétriques de plus en plus courtes, et comparer avec la sécante avant.
**Prérequis :** 0B (§101.5.1) · fiche §5.3 · **Fil rouge :** synthétique · **Parcours :** R, M

Soit $f(x) = \frac{1}{x}$, au point $a = 2$. La **sécante symétrique** passe par les points de la courbe d'abscisses $2 - h$ et $2 + h$ (la construction du livre, figure 5.8).

a) Sa pente pour $h = 1$ (3 décimales).
b) Sa pente pour $h = 0{,}5$ (4 décimales).
c) Sa pente pour $h = 0{,}1$ (5 décimales).
d) Dans ta copie, écris la pente de la sécante symétrique comme une fraction qui ne dépend que de $h$, puis déduis-en $f'(2)$ : c'est la valeur à reporter.
e) La pente de la sécante « avant », qui passe par les points d'abscisses $2$ et $2{,}1$ (4 décimales).
f) Pour $h = 0{,}1$, l'écart entre la pente avant et $f'(2)$, divisé par l'écart entre la pente symétrique et $f'(2)$ (écarts en valeur absolue, 1 décimale). Garde les fractions exactes jusqu'au bout.
g) Vérifie $f'(2)$ avec les règles de dérivation de 0B. (réponds dans ta copie)

### Ex 5.2 — Gradient à la main et direction de plus grande pente ✏️ ★ ⏱️ 15 min
**Objectif :** calculer un gradient, sa norme, les directions de plus grande pente et la pente dans une direction donnée.
**Prérequis :** 0B (§101.6, §101.3.3) · fiche §5.4 (🧮 pente dans une direction) · **Fil rouge :** synthétique · **Parcours :** R, M

Soit $f(x, y) = x^2 + xy + 2y^2$ et le point $P = (1, -1)$.

a) $f(P)$.
b) Le gradient $\nabla f(P)$, sous forme de liste $[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]$.
c) Sa norme (3 décimales).
d) Le vecteur unitaire qui indique la plus grande **descente** en $P$ (liste, 3 décimales).
e) La pente de $f$ en $P$ dans la direction du vecteur unitaire $\mathbf{u} = (0{,}6 ; 0{,}8)$ (1 décimale).
f) Un vecteur unitaire dans la direction duquel la pente est nulle en $P$, de première composante positive (liste, 3 décimales).
g) Fais un dessin à main levée : le point $P$, la flèche du gradient, la direction de f), et l'allure de la ligne de niveau qui passe par $P$. (réponds dans ta copie)

### Ex 5.3 — Trois pas de descente de gradient à la main ✏️ ★★ ⏱️ 15 min
**Objectif :** faire des pas de descente et de montée de gradient, et trouver pour quels learning rates la descente converge.
**Prérequis :** Ex 5.1 · fiche §5.3 (suivre la pente) · **Fil rouge :** synthétique · **Parcours :** R, M

On cherche le minimum de $f(x) = 2(x - 1)^2$ par descente de gradient, $x_{t+1} = x_t - \eta\, f'(x_t)$, en partant de $x_0 = 5$ avec $\eta = 0{,}125$.

a) $f'(5)$.
b) Les trois points suivants, $[x_1, x_2, x_3]$ (valeurs exactes).
c) Les valeurs $[f(x_0), f(x_1), f(x_2), f(x_3)]$ (valeurs exactes).
d) **Montée** : on cherche le maximum de $g(x) = 4x - x^2$, avec $x_{t+1} = x_t + \eta\, g'(x_t)$, en partant de $x_0 = -1$ avec $\eta = 0{,}25$. Donne $[x_1, x_2, x_3]$ (valeurs exactes).
e) Retour à $f$, en partant de $x_0 = 5$ mais avec $\eta = 0{,}5$ : donne $[x_1, x_2]$. Que se passe-t-il ensuite ?
f) Montre dans ta copie que $x_{t+1} - 1 = (1 - 4\eta)(x_t - 1)$. Pour quels $\eta > 0$ la descente se rapproche-t-elle du minimum à chaque pas ? Donne la borne supérieure de ces $\eta$.
g) Le learning rate qui atteint le minimum en un seul pas, quel que soit le départ.
h) Dans b), pourquoi les pas rétrécissent-ils, alors que $\eta$ ne change pas ? (réponds dans ta copie)

### Ex 5.4 — Tableau de variations : extrema locaux et globaux de x³ − 3x ✏️ ★★ ⏱️ 20 min
**Objectif :** trouver les extrema d'une fonction avec sa dérivée et sa dérivée seconde, et distinguer local et global sur un intervalle.
**Prérequis :** 0B (§101.5.4) · fiche §5.3 (🧮 dérivée seconde) · **Fil rouge :** synthétique · **Parcours :** M

Soit $f(x) = x^3 - 3x$.

a) Les points où $f'(x) = 0$, dans l'ordre croissant (liste).
b) Les valeurs de $f$ en ces points, dans le même ordre (liste).
c) La dérivée seconde $f''$ au plus petit des points de a). Que dit son signe ?
d) Sur l'intervalle $[-2{,}5 ; 2{,}5]$, la valeur du maximum global (3 décimales).
e) L'abscisse où il est atteint.
f) Sur l'intervalle $[-2 ; 2]$, en combien de points $f$ atteint-elle son maximum global ?
g) L'abscisse où $f''$ s'annule (le point d'inflexion).
h) Dresse le tableau de variations de $f$ sur tous les réels. $f$ y a-t-elle un maximum global ? Applique la construction du livre (marcher à gauche, puis à droite) depuis le point de départ $x = 0{,}5$ : quels sont « son » maximum local et « son » minimum local ? (réponds dans ta copie)

### Ex 5.5 — Point selle : x² − y² vu dans deux directions ✏️ ★★ ⏱️ 20 min
**Objectif :** reconnaître un point selle par la courbure dans plusieurs directions, et voir ce qu'en fait la descente de gradient.
**Prérequis :** Ex 5.2 · fiche §5.4 (points critiques) · **Fil rouge :** synthétique · **Parcours :** M

Soit $f(x, y) = x^2 - y^2$.

a) $\nabla f(0, 0)$ (liste).
b) La dérivée seconde de $t \mapsto f(t, 0)$ (le long de l'axe des $x$), puis celle de $t \mapsto f(0, t)$ (le long de l'axe des $y$), en $t = 0$ (liste de deux nombres).
c) Pour le vecteur unitaire $\mathbf{u} = (\cos\theta, \sin\theta)$, écris $f(t\cos\theta, t\sin\theta)$ en fonction de $t$ et de $\theta$ dans ta copie. Que vaut la dérivée seconde de $t \mapsto f(t\cos 30°, t\sin 30°)$ ?
d) L'angle $\theta$ de $[0° ; 90°]$, en degrés, pour lequel cette dérivée seconde est nulle.
e) Deux pas de descente de gradient depuis $(0{,}5 ; 0{,}1)$ avec $\eta = 0{,}25$ : le point atteint (liste, valeurs exactes).
f) La valeur de $f$ en ce point (3 décimales). Compare-la à $f(0{,}5 ; 0{,}1)$.
g) Trois pas depuis $(0{,}5 ; 0)$ exactement, avec le même $\eta$ : le point atteint (liste, valeurs exactes). Où va cette descente si on la poursuit ? Pourquoi est-ce un mauvais signe ? (la question se répond dans ta copie)

### Ex 5.6 — Pourquoi la différence centrée est plus précise (calcul exact sur x³) ∂ ★★ ⏱️ 25 min
**Objectif :** démontrer, par un calcul exact, que l'erreur de la différence centrée est en $h^2$ et celle de la différence avant en $h$.
**Prérequis :** Ex 5.1 · fiche, au-delà du livre (1) · **Parcours :** M

Soit $f(x) = x^3$ et un point $a$ ; $f'(a) = 3a^2$. On note $D_+(h) = \frac{f(a + h) - f(a)}{h}$ (avant), $D_-(h) = \frac{f(a) - f(a - h)}{h}$ (arrière) et $D_0(h) = \frac{f(a + h) - f(a - h)}{2h}$ (centrée).
1. Développe $(a + h)^3$ et $(a - h)^3$.
2. Montre que $D_+(h) = 3a^2 + 3ah + h^2$. Calcule de même $D_-(h)$.
3. Montre que $D_0(h) = 3a^2 + h^2$, et que $D_0$ est la moyenne de $D_+$ et $D_-$. Quels termes se compensent ?
4. Écris l'erreur $D_+(h) - f'(a)$ et l'erreur $D_0(h) - f'(a)$. Par combien chacune est-elle divisée quand on divise $h$ par 10 (pour $a \neq 0$ et $h$ petit) ?
5. Montre que, pour $f(x) = x^2$, la différence centrée donne **exactement** $f'(a)$, quel que soit $h$. Pourquoi est-ce normal, vu la question 3 ?
6. Montre que la différence seconde $\frac{f(a + h) - 2f(a) + f(a - h)}{h^2}$ donne exactement $f''(a) = 6a$ pour $f(x) = x^3$.
7. **Les arrondis.** On suppose que l'ordinateur calcule $f(a + h)$ et $f(a - h)$ chacun avec une erreur d'au plus $\delta$ (de l'ordre de $10^{-16}$ fois $|f(a)|$). Montre que l'erreur due aux arrondis sur $D_0(h)$ peut atteindre $\frac{\delta}{h}$. Pour $a = 1$, l'erreur totale est donc au plus de l'ordre de $h^2 + \frac{\delta}{h}$ : esquisse cette fonction de $h$. Pourquoi existe-t-il un meilleur $h$ ?

### Ex 5.7 — Rosenbrock : gradient et minimum à la main ∂ ★★★ ⏱️ 30 min
**Objectif :** calculer le gradient de la fonction de Rosenbrock, trouver son unique point critique et comprendre pourquoi sa vallée est difficile à descendre.
**Prérequis :** Ex 5.2 · 0B (règle de la chaîne, dérivées partielles) · fiche §5.4 · **Fil rouge :** Rosenbrock · **Parcours :** M

Soit $f(x, y) = (a - x)^2 + b\,(y - x^2)^2$, avec $a$ et $b > 0$ des constantes.
1. Calcule $\frac{\partial f}{\partial x}$ et $\frac{\partial f}{\partial y}$ (règle de la chaîne pour le second terme).
2. Résous $\nabla f = (0, 0)$ : commence par $\frac{\partial f}{\partial y} = 0$. Montre que l'unique point critique est $(a, a^2)$.
3. Pourquoi $f(x, y) \ge 0$ partout ? Déduis-en que $(a, a^2)$ est le minimum global.
4. Avec $a = 1$ et $b = 100$ : calcule le gradient en $(-1{,}5 ; 2)$, puis sa norme. Dans quelle direction la descente part-elle ?
5. Toujours avec $a = 1$ et $b = 100$, autour du minimum $(1, 1)$ : calcule la dérivée seconde en 0 de $u \mapsto f(1 + u, 1)$ (on ne bouge que $x$), de $u \mapsto f(1, 1 + u)$ (on ne bouge que $y$), et de $t \mapsto f(1 + t, 1 + 2t)$ (on suit la direction $(1, 2)$, celle du fond de la vallée en $(1, 1)$, tangente à la courbe $y = x^2$). Ce vecteur n'est pas unitaire (sa norme vaut $\sqrt 5$) : pour une courbure par unité de longueur, divise cette troisième dérivée seconde par 5. Compare les trois courbures.
6. Généralise ✏️ 5.3 f) : pour une parabole de dérivée seconde $c > 0$, pour quels learning rates la descente de gradient converge-t-elle ? Quelle borne obtiens-tu avec les deux premières courbures de la question 5 ? (La direction la plus courbée traverse en fait la vallée en biais, avec une courbure d'environ 1 000 : la vraie borne est encore un peu plus basse.) Avec un learning rate sous cette borne, que se passe-t-il le long du fond de la vallée, où la courbure est plus de mille fois plus faible ? Conclus : pourquoi la descente de gradient avance-t-elle si lentement sur Rosenbrock ?

<a id="reflexion"></a>

## 🗣️ 🧮 📄 Réflexion

### Ex 5.8 — Le gradient expliqué avec de l'eau sur un drap 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer le gradient et la descente de gradient à un débutant, avec l'image du livre, et en dire les limites.
**Prérequis :** fiche §5.4 · **Parcours :** R

Une amie n'a jamais entendu parler de dérivées. Explique-lui ce qu'est le gradient et comment un réseau « apprend en descendant », en **cinq lignes au plus**, avec l'image du drap et de l'eau. Contraintes :
- les mots « pente », « direction » et « pas » ;
- dire où l'eau s'arrête, et pourquoi ce n'est pas forcément le point le plus bas ;
- une limite de l'image (une chose que l'eau fait et que la descente de gradient ne fait pas, ou l'inverse) ;
- aucune formule.

Relis-toi à voix haute, puis compare avec la réponse modèle de `05_solutions.md`.

### Ex 5.9 — Fermi : le prix d'un gradient numérique pour un million de paramètres 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer le coût d'un gradient par différences finies pour un réseau, et le comparer à la différentiation automatique.
**Prérequis :** Ex 5.2 · fiche §5.4 et au-delà du livre (3) · **Parcours :** M

Un réseau a $P = 10^6$ paramètres. Calculer sa loss sur un mini-batch de 64 exemples coûte environ $2P$ opérations par exemple (une multiplication et une addition par poids). Donne des ordres de grandeur, en justifiant chaque étape.
1. Combien d'opérations pour **une** évaluation de la loss sur le mini-batch ?
2. Combien d'évaluations de la loss faut-il pour estimer le gradient par différences centrées ? Combien d'opérations au total ?
3. Sur un processeur qui fait $10^{11}$ opérations par seconde, combien de temps dure **un** pas de descente de gradient ?
4. La rétropropagation coûte environ trois évaluations de la loss. Combien de temps par pas ? Combien de fois plus rapide ?
5. Un entraînement compte $10^4$ pas. Combien de temps dans chaque cas ?
6. Comment ces deux coûts évoluent-ils si le réseau a 1 000 fois plus de paramètres ?

### Ex 5.10 — Dauphin et al. (2014) : les points selles en grande dimension 📄 ★★ ⏱️ 30 min
**Objectif :** lire un article fondateur sur les points selles et en retenir ce qui reste vrai aujourd'hui.
**Prérequis :** Ex 5.5 · fiche §5.4 (🕰️ points selles)

L'article : Y. Dauphin, R. Pascanu, C. Gulcehre, K. Cho, S. Ganguli et Y. Bengio, « Identifying and attacking the saddle point problem in high-dimensional non-convex optimization », *NeurIPS 2014*, en accès libre : [arXiv:1406.2572](https://arxiv.org/abs/1406.2572). Lis le résumé, l'introduction et les sections 2 et 3 (avec la figure 1), puis parcours les sections 4 (les méthodes près d'un point selle, dont celle de Newton) et 6 (*saddle-free Newton*).

1. Selon les auteurs, quelle est la principale difficulté de l'optimisation en grande dimension : les minima locaux ou autre chose ?
2. Qu'appellent-ils l'**indice** (*index*) d'un point critique ? Que vaut-il pour un minimum ? Pour un maximum ?
3. Que montre la figure 1, et sur quelles données ? Quelle relation entre l'erreur et l'indice observent-ils ?
4. Pourquoi un point selle entouré d'un plateau ralentit-il la descente de gradient, et pourquoi peut-il faire croire à un minimum ?
5. Pourquoi la méthode de Newton est-elle **attirée** par les points selles ? Quelle idée simple la méthode *saddle-free Newton* utilise-t-elle pour l'éviter ?
6. Les mesures de la figure 1 portent sur de petits réseaux (images réduites à 10 × 10 pixels). Peut-on généraliser à un réseau moderne ? Que disent les travaux plus récents cités dans la fiche (Lee et coll. 2016, Du et coll. 2017) ?
7. Quel lien fais-tu avec ✏️ 5.5 g) et avec la prédiction 🔮 5.20 du notebook ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 5.E1 — Descente de gradient : que se passe-t-il avec un learning rate trop grand, puis trop petit ? 💼 ★★ ⏱️ 10 min
*Fiche §5.3, §5.4 · parcours R*

« Expliquez la descente de gradient. Que se passe-t-il si le learning rate est trop grand ? Trop petit ? Comment le choisissez-vous ? »

### 5.E2 — Minimum local : un vrai problème en deep learning ? 💼 ★★ ⏱️ 10 min
*Fiche §5.3, §5.4 · prérequis Q4 · parcours R*

« Les réseaux de neurones ont des loss non convexes. Les minima locaux sont-ils un vrai problème en pratique ? »

### 5.E3 — Comment vérifier un gradient calculé ? 💼 ★★ ⏱️ 10 min
*Fiche, au-delà du livre (1) à (3) · prérequis 5.15 · parcours R*

« Vous avez écrit vous-même le calcul du gradient d'une loss. Comment vérifiez-vous qu'il est juste ? »

### 5.E4 — Point selle : définition et effet sur l'optimisation 💼 ★★ ⏱️ 10 min
*Fiche §5.4 · prérequis Q8 · parcours R*

« Qu'est-ce qu'un point selle, et pourquoi en parle-t-on en optimisation de réseaux de neurones ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch05_courbes/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie `mylearn/calculus.py`. La partie 0 du notebook vérifie tes réponses aux exercices ✏️ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 5.11 | Dérivées numériques : première et seconde | 🔨 | ★★ | 25 |
| 5.12 | Quel pas h choisir ? Prédire la courbe d'erreur | 🔮 | ★★ | 15 |
| 5.13 | Erreur de troncature contre erreur d'arrondi | 🔬 | ★★ | 25 |
| 5.14 | Les maxima des cycles solaires | 🔨 | ★★ | 30 |
| 5.15 | numerical_gradient sur la vallée de Rosenbrock | 🔨 | ★★ | 25 |
| 5.16 | Le gradient qui abîme son entrée | 🐛 | ★★ | 20 |
| 5.17 | Lire des lignes de niveau : où pointe le gradient ? | 📈 | ★★ | 20 |
| 5.18 | gradient_descent, et sa version qui monte | 🔨 | ★★ | 30 |
| 5.19 | Learning rate sur un bol : trop petit, juste, trop grand | 🔬 | ★★ | 25 |
| 5.20 | Démarrer pile sur un point selle | 🔮 | ★★ | 15 |
| 5.21 | Le même gradient avec torch.autograd | 📦 | ★★ | 20 |
| 5.22 | L'eau qui descend la surface (figure 5.18) | 🎨 | ★★ | 30 |
| 5.23 | Tests de propriétés paramétrés avec pytest | 🛠️ | ★★ | 20 |
| 5.24 | Minimum, maximum, selle ou plat : classify_critical_point | 🔨 | ★★★ | 35 |
| 5.25 | Atteindre le fond de la vallée de Rosenbrock | 🏆 | ★★★ | 60 |
