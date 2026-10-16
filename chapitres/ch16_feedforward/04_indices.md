# 16 · Réseaux feed-forward — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 16.Q1 — Qu'est-ce qui rend un réseau « feed-forward » ?

<details><summary>Indice 1</summary>

Relis la définition du §16.2 de la fiche (« Pas de boucle : le DAG ») et le paragraphe du §16.1 sur les couches.

</details>
<details><summary>Indice 2</summary>

Pour b), compte les **matrices de poids** : une couche de neurones qui calcule a des poids entrants ; la « couche d'entrée » n'en a pas. Pour c) et d), le §16.1 dit comment les couches communiquent, et pourquoi on parle de traitement hiérarchique.

</details>
<details><summary>Indice 3</summary>

a) La règle d'un réseau feed-forward est l'absence de boucle : une donnée qui revient à un nœud qu'elle a quitté suivrait une boucle ; l'affirmation décrit exactement cette règle, donc elle est vraie. b) Entre 784, 128 et 10, combien de passages d'une taille à la suivante ? c) Les neurones d'une même couche ont-ils des arêtes entre eux dans un MLP ? d) Sur quoi chaque couche fait-elle son calcul ?

</details>

### 16.Q2 — Vocabulaire des graphes : nœuds, arêtes, parents, descendants

<details><summary>Indice 1</summary>

Dessine le graphe : cinq cercles et cinq flèches. Les définitions sont au §16.2 de la fiche.

</details>
<details><summary>Indice 2</summary>

Un parent de E est le départ d'une flèche **qui arrive** à E. Un descendant de A est un nœud où l'on peut arriver depuis A **en suivant les flèches**, en une ou plusieurs étapes. Un nœud sans parent n'a aucune flèche qui arrive.

</details>
<details><summary>Indice 3</summary>

a) Les flèches qui arrivent à E partent de B (B → E) et de C (C → E) : les parents de E sont B et C, soit `"BC"`. b) Depuis A, suis A → C, puis les flèches qui partent de C, et ainsi de suite. c) Pour chaque nœud, cherche une flèche qui y arrive. d) Existe-t-il un chemin de B à D, par un nœud intermédiaire ? e) Relis la convention du §16.2 sur les dessins sans pointes.

</details>

### 16.Q3 — Vrai ou faux : la passe arrière crée une boucle de rétroaction

<details><summary>Indice 1</summary>

Relis le paragraphe « Pourquoi s'interdire les boucles » du §16.2 de la fiche : il distingue deux choses que l'on confond souvent.

</details>
<details><summary>Indice 2</summary>

Une boucle est une suite d'arêtes qui revient à son point de départ. Parcourir les arêtes à l'envers crée-t-il une nouvelle arête ? Un nœud reçoit-il alors sa propre sortie ?

</details>
<details><summary>Indice 3</summary>

a) La passe arrière parcourt les arêtes du même graphe dans l'autre sens, sans en ajouter : le graphe n'a toujours aucune boucle, et aucun nœud ne reçoit sa propre sortie, donc l'affirmation est fausse. b) À quel cas le livre réserve-t-il le mot *feedback* ? c) Dans l'effet Larsen, le son ressort du haut-parleur, rentre dans le micro, ressort du haut-parleur : quel nom porte ce circuit ?

</details>

### 16.Q4 — Pourquoi des graphes sans boucle, et où est l'exception des RNN ?

<details><summary>Indice 1</summary>

Un cycle est un chemin qui **revient à son point de départ**. Deux chemins parallèles, une arête qui saute une couche : sont-ils des cycles ?

</details>
<details><summary>Indice 2</summary>

Pour c), suppose que chaque nœud ait un parent, et remonte de parent en parent dans un graphe fini : que finis-tu par rencontrer ? Pour d), range les nœuds dans un ordre topologique : une arête ne peut aller que d'un nœud vers un nœud **placé après** lui.

</details>
<details><summary>Indice 3</summary>

a) Deux chemins A → B → D et A → C → D vont tous deux de A à D, mais aucun ne revient à son départ : ils ne forment pas de cycle, donc l'affirmation est vraie. b) Relis l'exception des RNN au §16.2 : que fait-on pour supprimer la boucle du dessin sans supprimer la mémoire ? c) En remontant indéfiniment dans un graphe fini, on repasse par un nœud déjà vu : qu'est-ce que cela forme ? d) Dans l'ordre 1, 2, 3, 4, le nœud 1 peut viser 3 nœuds, le nœud 2 en viser 2… : additionne.

</details>

### 16.Q5 — Réseau synchrone ou asynchrone : reconnaître les deux

<details><summary>Indice 1</summary>

Le critère du §16.3 : y a-t-il un signal commun (une horloge, un gong) qui déclenche les échanges, ou chaque élément agit-il dès qu'il a du nouveau ?

</details>
<details><summary>Indice 2</summary>

Pour b), demande-toi ce qui arriverait au nœud le plus lent si l'horloge battait plus vite que lui. Pour c), relis le passage où le livre dit lequel des deux cas est le plus général.

</details>
<details><summary>Indice 3</summary>

a) Dans le système 1, tous les musiciens suivent la même battue : un signal commun cadence tout, c'est le synchrone, d'où un S en première lettre. Pour les trois autres, cherche le signal commun : un tableur attend-il un battement pour recalculer ? Des collègues attendent-ils un signal pour répondre à un e-mail ? Des joueurs qui révèlent leur coup au gong ? b) Le plus lent doit avoir fini avant le battement suivant : sur qui règle-t-on la période ? c) Un système synchrone est-il un système où les mises à jour tombent ensemble ?

</details>

### 16.Q6 — Horodatage : quand un nœud doit-il recalculer sa sortie ?

<details><summary>Indice 1</summary>

Relis la règle du §16.3 de la fiche : ce sont les **horodatages** qui comptent, pas les valeurs.

</details>
<details><summary>Indice 2</summary>

Compare chaque horodatage à 10, l'heure du dernier calcul : une entrée est nouvelle si elle a été envoyée **après**. Le nœud recalcule quand **toutes** ses entrées sont nouvelles. Pour c), relis qui écrit l'horodatage, et à quel moment.

</details>
<details><summary>Indice 3</summary>

a) 12, 15 et 11 sont tous postérieurs à 10 : les trois entrées sont nouvelles, et le nœud doit recalculer : vrai. b) Compare de même 12, 9 et 14 à 10. c) L'horodatage est-il écrit par le nœud qui envoie, au moment du calcul, ou par celui qui reçoit ? d) La règle regarde-t-elle les valeurs ?

</details>

### 16.Q7 — Dans un programme, qui garantit l'ordre d'évaluation ?

<details><summary>Indice 1</summary>

Relis le §16.3.1 de la fiche et son encadré 🕰️.

</details>
<details><summary>Indice 2</summary>

Un programme Python exécute ses instructions dans l'ordre où elles sont écrites. Sur GPU, l'appel d'une opération rend la main **avant** la fin du calcul : que mesure alors une horloge lue juste après ?

</details>
<details><summary>Indice 3</summary>

a) Dans un programme NumPy, la couche 2 est calculée par une ligne écrite après celle de la couche 1 : c'est l'ordre des instructions qui garantit qu'elle lit des valeurs prêtes, d'où `"B"`. b) Que fait `torch.cuda.synchronize()`, et pourquoi faut-il attendre la fin du calcul avant de lire l'heure ? c) Dans quel ordre Python exécute-t-il deux lignes successives ?

</details>

### 16.Q8 — Un neurone, trois enfants : combien de poids ?

<details><summary>Indice 1</summary>

Relis la règle du §16.4 : chaque arête porte son propre poids.

</details>
<details><summary>Indice 2</summary>

Compte les arêtes : un neurone qui a trois enfants a trois arêtes sortantes. Entre deux couches entièrement connectées, chaque neurone de la première est relié à chaque neurone de la seconde.

</details>
<details><summary>Indice 3</summary>

a) Trois enfants, donc trois arêtes sortantes, chacune avec son poids : 3. b) La même valeur multipliée par trois poids donne-t-elle forcément trois produits égaux ? c) Chacun des 4 neurones a combien d'arêtes vers la couche de 3 ? d) Relis la phrase du §16.4 sur les schémas sans poids écrits.

</details>

### 16.Q9 — Pourquoi pas le même poids partout au départ ?

<details><summary>Indice 1</summary>

Relis le premier paragraphe du §16.4.1 de la fiche : la symétrie des neurones.

</details>
<details><summary>Indice 2</summary>

Deux neurones qui ont les mêmes poids entrants, le même biais et les mêmes entrées calculent la même somme pondérée. Qu'est-ce qui pourrait les rendre différents pendant l'entraînement, s'ils reçoivent la même correction ? Pour d), $\mathcal{U}(-a, a)$ a la variance $a^2/3$.

</details>
<details><summary>Indice 3</summary>

a) Mêmes entrées, mêmes poids, même biais : la même somme pondérée, donc la même sortie, pour chaque exemple : vrai. b) Élimine les options qui ne parlent pas des corrections reçues pendant l'entraînement. c) Si les poids entrants diffèrent, les sommes pondérées diffèrent-elles, même avec des biais égaux ? d) Calcule $a/\sqrt{3}$ avec $a = 0{,}05$.

</details>

### 16.Q10 — LeCun, Glorot, He : quel fan-in, pour quelle activation ?

<details><summary>Indice 1</summary>

Utilise la table des trois règles du §16.4.1 de la fiche. Ici $n_{in} = 400$ et $n_{out} = 100$.

</details>
<details><summary>Indice 2</summary>

La version normale tire un écart-type $\sqrt{\mathrm{Var}(w)}$ ; la version uniforme une borne $\sqrt{3\,\mathrm{Var}(w)}$. He n'utilise que $n_{in}$ ; Glorot additionne les deux fans.

</details>
<details><summary>Indice 3</summary>

a) He normale : $\mathrm{Var}(w) = 2/n_{in} = 2/400 = 0{,}005$, et l'écart-type est sa racine, $\sqrt{0{,}005} \approx 0{,}071$. b) Glorot uniforme : borne $\sqrt{6/(n_{in} + n_{out})}$ ; calcule-la avec 400 + 100. c) et d) Pour quelle activation chaque règle a-t-elle été conçue, et laquelle utilise les deux fans ? e) Relis la formule de He : $n_{out}$ y figure-t-il ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 16.R1 — Ch. 15 : Que contient un estimateur après fit() ?

<details><summary>Indice 1</summary>

Relis la convention de scikit-learn du ch. 15 sur les attributs appris, et l'encadré sur le `MLPClassifier` de la fiche (section sur la passe avant).

</details>
<details><summary>Indice 2</summary>

Un attribut suffixé par `_` n'existe qu'après `fit`. `coefs_` contient une matrice de poids par couche de neurones (cachées et sortie), de forme `(n_in, n_out)` ; la sortie a un neurone par classe dès qu'il y a plus de deux classes.

</details>
<details><summary>Indice 3</summary>

a) `coefs_` finit par `_` : c'est un attribut appris, créé par `fit` ; avant `fit`, il n'existe pas : faux. b) Que signale le suffixe `_` dans scikit-learn ? c) Combien de couches de neurones a un réseau à une couche cachée ? d) Que relie la première matrice, et dans quel ordre scikit-learn range-t-il ses deux dimensions ? e) Combien de neurones de sortie pour 3 classes ?

</details>

### 16.R2 — Ch. 13 : Combien de régions pour un arbre de profondeur 2 ?

<details><summary>Indice 1</summary>

Dessine l'arbre : une racine (profondeur 0), ses enfants (profondeur 1), leurs enfants (profondeur 2).

</details>
<details><summary>Indice 2</summary>

Dans un arbre binaire complet, chaque niveau a deux fois plus de nœuds que le précédent ; les feuilles sont au dernier niveau, les nœuds internes (les questions) au-dessus.

</details>
<details><summary>Indice 3</summary>

a) La racine a 2 enfants, chacun a 2 enfants : 4 feuilles au niveau 2. b) Le même raisonnement, avec 5 doublements. c) Une question « $x_j \le s$ » coupe l'espace par un plan perpendiculaire à l'axe $j$ : quelle forme ont les morceaux, coupe après coupe ? d) Compte les nœuds des niveaux 0 et 1.

</details>

### 16.R3 — Ch. 0B : Formes compatibles dans un produit matriciel

<details><summary>Indice 1</summary>

Le produit $(a, b) \times (c, d)$ n'existe que si $b = c$, et il a la forme $(a, d)$.

</details>
<details><summary>Indice 2</summary>

Pour un vecteur 1-D de forme $(b,)$, NumPy calcule `A @ v` si `A` a $b$ colonnes, et rend un vecteur 1-D. Chaque élément d'un produit $(a, b) \times (b, d)$ demande $b$ multiplications.

</details>
<details><summary>Indice 3</summary>

a) $(3, 4) \times (4, 2)$ : 4 = 4, le produit existe et a la forme $(3, 2)$, soit `"(3, 2)"`. b) Compare le nombre de colonnes de B au nombre de lignes de A. c) A a-t-il autant de colonnes que v a d'éléments, et quelle forme a le résultat d'un produit par un vecteur 1-D ? d) Combien d'éléments a le résultat, et combien de multiplications chacun ? e) Vérifie au moins sur les formes : $(\mathbf{A}\mathbf{B})^\top$ a la forme $(2, 3)$ ; quelle forme a $\mathbf{B}^\top\mathbf{A}^\top$ ?

</details>

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

### Ex 16.1 — Compter les poids et les biais d'un MLP

<details><summary>Indice 1</summary>

Une couche de $n_{in}$ entrées et $n_{out}$ neurones a $n_{in} n_{out}$ poids et $n_{out}$ biais (fiche, section sur la passe avant).

</details>
<details><summary>Indice 2</summary>

Liste les couches deux par deux : (10, 8), (8, 6), (6, 3). Additionne les produits pour les poids, les tailles des couches de neurones (pas l'entrée) pour les biais.

</details>
<details><summary>Indice 3</summary>

a) La première couche relie 10 entrées à 8 neurones : $10 \times 8 = 80$ poids. b) Ajoute les poids des couches (8, 6) et (6, 3). c) Un biais par neurone : quelles couches ont des neurones ? d) b) + c). e) et f) La même formule, $\sum (n_{\ell-1} n_\ell + n_\ell)$, sur les nouvelles tailles. g) Compare $400 \times 400$ à $200 \times 200$.

</details>

### Ex 16.2 — Passe avant à la main dans un réseau 2-2-1

<details><summary>Indice 1</summary>

Une somme pondérée : chaque entrée multipliée par son poids, puis la somme, plus le biais. Pour le réseau, $\mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b}$ : le neurone $k$ utilise la **colonne** $k$ de $\mathbf{W}$.

</details>
<details><summary>Indice 2</summary>

Calcule $z_1 = x_1 W_{11} + x_2 W_{21} + b_1$ et $z_2 = x_1 W_{12} + x_2 W_{22} + b_2$, applique l'activation, puis la sortie $1{,}5\,h_1 - 2\,h_2 - 0{,}5$. Pour le mini-batch, fais le même calcul pour la seconde ligne.

</details>
<details><summary>Indice 3</summary>

a) $0{,}2 \times 1 + 0{,}6 \times (-0{,}5) + 0{,}4 \times 2 + 0{,}1 = 0{,}2 - 0{,}3 + 0{,}8 + 0{,}1 = 0{,}8$. b) $z_1 = 2 \times 0{,}5 + 1 \times 1 + 0$ ; calcule de même $z_2$ avec la seconde colonne et le biais 1. c) Applique ReLU à $(z_1, z_2)$, puis la sortie. d) Remplace ReLU par tanh ; garde les valeurs exactes jusqu'au bout. e) $W_{jk}$ va de $x_j$ vers $h_k$ : quelle ligne et quelle colonne ? f) Que vaut $h_2$ avec ReLU pour cet exemple ? g) Refais b) et c) pour $(0 ;\ 1)$.

</details>

### Ex 16.3 — Formes et broadcasting d'un mini-batch à travers les couches

<details><summary>Indice 1</summary>

Une ligne par exemple, $\mathbf{W}$ en $(n_{in}, n_{out})$, $\mathbf{b}$ en $(n_{out},)$. Le broadcasting aligne les formes **par la droite** : deux dimensions sont compatibles si elles sont égales ou si l'une vaut 1.

</details>
<details><summary>Indice 2</summary>

Suis le batch couche par couche : (32, 784) → (32, 256) → (32, 64) → (32, 10). Le cache garde, pour chaque couche, l'entrée `a_prev` (32 × $n_{in}$ valeurs) et la somme pondérée `z` (32 × $n_{out}$ valeurs).

</details>
<details><summary>Indice 3</summary>

a) 32 images, 784 pixels chacune, une par ligne : `"(32, 784)"`. b) à e) Applique la convention à chaque tableau. f) Additionne $32\,(n_{in} + n_{out})$ sur les trois couches. g) Aligne `(32, 256)` et `(32,)` par la droite : 256 et 32 sont-elles compatibles ? h) $\mathbf{X}$ a 784 colonnes ; combien de lignes a ce `W1` ? i) Quelle forme a `W1.T` ?

</details>

### Ex 16.4 — Ordre d'évaluation d'un graphe et détection de boucle

<details><summary>Indice 1</summary>

Relis l'encadré sur l'ordre topologique (§16.2 de la fiche) : compte les parents de chaque nœud, puis range un nœud prêt à la fois.

</details>
<details><summary>Indice 2</summary>

Comptes de parents de départ : U 0, V 0, A 2, B 1, C 2, Y 2. À chaque étape, range le plus petit nom prêt et retire 1 au compte de ses enfants. Pour c), énumère systématiquement, à chaque étape, **tous** les nœuds prêts. Pour d) et e), une arête crée un cycle si et seulement s'il existait déjà un chemin de son arrivée à son départ.

</details>
<details><summary>Indice 3</summary>

a) Au départ, U et V sont prêts ; on range U (le plus petit). A a encore un parent non rangé (V) : seul V est prêt, on le range, ce qui rend A et B prêts. Continue : à chaque étape, le plus petit nom prêt. b) Compte les flèches de l'énoncé. c) Après U et V (dans un ordre ou l'autre), ou dès V, quels nœuds peuvent venir, et où B peut-il se placer ? d) Existe-t-il un chemin de B à Y ? e) Existe-t-il un chemin de Y à U ? f) Cherche le chemin d'une entrée à Y qui passe par le plus de nœuds.

</details>

### Ex 16.5 — Simuler un réseau asynchrone avec horodatage

<details><summary>Indice 1</summary>

Fais un tableau : une ligne par message, avec la valeur et l'horodatage de P, de Q et de R après ce message, et l'heure du dernier calcul de S.

</details>
<details><summary>Indice 2</summary>

Après chaque message, demande-toi si les **trois** horodatages sont postérieurs au dernier calcul de S. Si oui, S calcule (la somme des valeurs actuelles) et note l'heure. Pour la règle naïve, regarde seulement si la valeur d'une entrée a changé.

</details>
<details><summary>Indice 3</summary>

a) Après $t = 5$, P a l'horodatage 2, Q 3 et R 5, tous postérieurs à 1 : S calcule $1 + 4 + 2 = 7$, et note $t = 5$. b) Continue : après quel message les trois horodatages dépassent-ils 5 ? Quelle est alors la somme ? c) Compte les calculs de a) et b), et vérifie le message de $t = 10$. d) Pour chaque message, la valeur change-t-elle ? e) La somme après le dernier message qui change une valeur.

</details>

### Ex 16.6 — Symétrie : pourquoi une initialisation constante ne se brise jamais

<details><summary>Indice 1</summary>

Tout repose sur une **symétrie** : échanger deux neurones jumeaux ne change rien au réseau. Une fonction qui ne change pas quand on échange deux de ses variables a des dérivées partielles égales en un point où ces variables sont égales.

</details>
<details><summary>Indice 2</summary>

1. Écris $h_1$ et $h_2$ avec des paramètres égaux. 2. Dans $\hat y = \sum_k v_k h_k + c$, échanger les neurones 1 et 2 échange deux termes d'une somme. 3. Écris $\partial \mathcal{L} / \partial v_1$ comme la limite de $[\mathcal{L}(\dots, v_1 + \varepsilon, v_2, \dots) - \mathcal{L}(\dots)]/\varepsilon$, et applique l'échange au point perturbé. 4. Une mise à jour soustrait $\eta$ fois le gradient. 5. Somme de $m$ termes égaux. 6. $\partial \hat y / \partial v_k = h_k$ et $\partial \hat y / \partial w_{jk} = v_k f'(z_k) x_j$.

</details>
<details><summary>Indice 3</summary>

1. $h_1 = f(\mathbf{x}\cdot\mathbf{w}_1 + b_1) = f(\mathbf{x}\cdot\mathbf{w}_2 + b_2) = h_2$, puisque $\mathbf{w}_1 = \mathbf{w}_2$ et $b_1 = b_2$. Pour la suite : 2. que deviennent les termes $v_1 h_1 + v_2 h_2$ après l'échange ? 3. Au point jumeau $\theta$, perturber $v_1$ de $\varepsilon$ donne un point $\theta'$ ; l'échange envoie $\theta'$ sur le point où c'est $v_2$ qui est perturbé : que dit la question 2 de leurs losses ? 4. Si les paramètres et les gradients des deux neurones sont égaux, que vaut chaque paramètre après le pas ? 5. Avec $m$ neurones jumeaux de poids sortant $v_1$, combien vaut $\sum_k v_k h_k$ ? 6. Évalue les deux dérivées avec $v_k = 0$ et $\tanh(0) = 0$. 7. Pense à ce que le chapitre propose, et à une technique qui éteint des neurones au hasard (ch. 20).

</details>

### Ex 16.7 — Conserver la variance : retrouver les règles de LeCun, Glorot et He

<details><summary>Indice 1</summary>

Relis l'encadré sur la variance d'une somme de produits (§16.4.1) : il fait la question 1 dans le cas général.

</details>
<details><summary>Indice 2</summary>

1. Linéarité de l'espérance, puis $\mathbb{E}[w_j x_j] = \mathbb{E}[w_j]\mathbb{E}[x_j]$ par indépendance ; la variance d'une somme de termes deux à deux décorrélés est la somme des variances (montre-le d'abord : calcule $\mathbb{E}[w_j x_j\, w_k x_k]$ pour $j \ne k$). 3. Si $z$ est symétrique, $z^2$ a la même loi sur les valeurs positives et négatives, et ReLU ne garde que les positives. 5. Calcule $\int_{-a}^{a} x^2 \frac{1}{2a}\,dx$. Pour les vérifications, applique les formules ; pour g), chaque couche ReLU en LeCun divise la variance par 2.

</details>
<details><summary>Indice 3</summary>

a) $\mathrm{Var}(z) = n\,\mathrm{Var}(w)\,\mathbb{E}[x^2] = 256 \times \frac{1}{256} \times 1 = 1$. Pour les autres : b) et d) la variance de chaque règle avec le fan-in donné ; c) le résultat de la question 3 avec $\mathrm{Var}(z) = 4$ ; e) $\sqrt{6/(n_{in} + n_{out})}$ ; f) la variance est multipliée par 1,5 à chacune des 10 couches : quelle puissance ? g) La première couche a déjà la variance 1 : combien de couches la divisent ensuite par 2 jusqu'à la dixième ?

</details>

### Ex 16.8 — Fermi : taille, mémoire et coût d'un MLP pour MNIST

<details><summary>Indice 1</summary>

Reprends le nombre de paramètres de 16.1 e) et la section « Mémoire et calcul » de la fiche : 4 octets par nombre en `float32`, une multiplication par poids et par exemple.

</details>
<details><summary>Indice 2</summary>

Les multiplications d'une image : la somme des produits $n_{\ell-1} n_\ell$ (sans les biais). Le cache d'un batch : pour chaque couche, $128 \times n_{in}$ valeurs pour `a_prev` et $128 \times n_{out}$ pour `z`.

</details>
<details><summary>Indice 3</summary>

a) $669\,706 \times 4 = 2\,678\,824$ octets, soit environ 2,7 Mo. b) Calcule $784 \times 512 + 512 \times 512 + 512 \times 10$ et divise par $10^6$. c) Multiplie par 60 000 et divise par $10^9$. d) Divise le nombre de c) (en multiplications) par $10^{10}$. e) Trois fois d). f) Additionne $128\,(n_{in} + n_{out})$ sur les trois couches, multiplie par 4 octets.

</details>

<a id="reflexion"></a>

## 🗣️ 📄 Réflexion

### Ex 16.9 — Expliquer un réseau feed-forward avec une chaîne d'usine

<details><summary>Indice 1</summary>

Pars de l'image du livre (§16.2) : les matières premières, les machines, les produits finis. À quoi correspond chacun dans un réseau ?

</details>
<details><summary>Indice 2</summary>

Les postes de travail sont les neurones, rangés par étapes (les couches) ; chaque poste combine ce qu'il reçoit avec des réglages (les poids). Apprendre, c'est ajuster les réglages pour que les produits finis soient bons. Pour la limite, pense à qui règle les machines d'une usine, ou à ce qui circule entre les postes.

</details>
<details><summary>Indice 3</summary>

Structure en cinq phrases : ce qui entre et sort (une image, une prédiction) ; les postes rangés en étapes, qui ne renvoient jamais une pièce en arrière ; chaque poste mélange ses entrées selon des réglages, les poids ; apprendre, c'est régler ces milliers de boutons à partir d'exemples ; une limite de l'image (dans une usine, chaque poste a une tâche connue, choisie par un ingénieur ; dans un réseau, personne ne choisit ce que fait chaque neurone ; ou : un réseau traite des nombres, pas des objets).

</details>

### Ex 16.10 — Glorot & Bengio (2010) : pourquoi les réseaux profonds apprenaient mal

<details><summary>Indice 1</summary>

Lis d'abord le résumé et la conclusion, puis cherche dans l'article les figures qui suivent les activations de chaque couche pendant l'entraînement.

</details>
<details><summary>Indice 2</summary>

Pour 2), cherche la figure des activations de la sigmoïde au fil de l'entraînement : que fait la dernière couche cachée au début ? Pour 3), l'article écrit la variance des activations et celle des gradients à travers les couches, comme ∂ 16.7, et en tire une condition pour chaque passe. Pour 5), quelle activation l'article n'étudie-t-il pas, et qu'ont apporté He et coll. (2015) ?

</details>
<details><summary>Indice 3</summary>

Plan de réponse : 1) la question (pourquoi la descente de gradient depuis une initialisation au hasard marche mal sur les réseaux profonds), les données (des images synthétiques, et des datasets d'images classiques) et les réseaux (une à cinq couches cachées de 1 000 neurones, sigmoïde, tanh, softsign) ; 2) la saturation de la dernière couche cachée et son explication par la sortie softmax ; 3) l'initialisation normalisée, $\mathcal{U}\big(\pm\sqrt{6/(n_{in} + n_{out})}\big)$, comparée à l'initialisation $\mathcal{U}(\pm 1/\sqrt{n_{in}})$ ; 4) les histogrammes d'activations et de gradients par couche, les écarts-types, les valeurs singulières de la jacobienne ; 5) les limites (ni ReLU, ni normalisation, ni connexions résiduelles) et ce qui a suivi.

</details>

<a id="entretien"></a>

## 💼 Entretien

### 16.E1 — Pourquoi ne pas initialiser tous les poids à zéro ?

<details><summary>Indice 1</summary>

Deux idées : la symétrie des neurones, et le cas particulier de zéro.

</details>
<details><summary>Indice 2</summary>

Des neurones identiques reçoivent les mêmes gradients et restent identiques ; avec des zéros, les gradients des poids sont même nuls dans un réseau tanh (∂ 16.6, question 6). Les biais, eux, peuvent partir de 0.

</details>
<details><summary>Indice 3</summary>

Structure : le problème (symétrie : mêmes sorties, mêmes gradients, une couche qui vaut un neurone) ; le cas de zéro (rien n'apprend, sauf les biais de sortie) ; la solution (des poids aléatoires, d'échelle choisie selon le fan-in et l'activation) ; les biais (0, sauf exceptions) ; une relance possible (un tout petit bruit autour d'une constante : voir 🔮 16.19).

</details>

### 16.E2 — He ou Glorot : quelle initialisation pour quelle activation ?

<details><summary>Indice 1</summary>

Le principe commun : garder la variance des activations (et des gradients) d'une couche à l'autre.

</details>
<details><summary>Indice 2</summary>

Le calcul $\mathrm{Var}(z) = n_{in}\mathrm{Var}(w)\mathbb{E}[x^2]$, le facteur $1/2$ de ReLU, le compromis de Glorot entre les deux passes. Puis le défaut de PyTorch (encadré 🕰️ du §16.4.1).

</details>
<details><summary>Indice 3</summary>

Structure : la règle (He pour ReLU, Glorot pour tanh et la sigmoïde) ; d'où elle vient (la variance multipliée par $n_{in}\mathrm{Var}(w)$ à chaque couche ; ReLU en garde la moitié) ; ce qui arrive sinon dans un réseau profond (évanouissement ou explosion, 🔬 16.23) ; PyTorch (`nn.Linear` : $\mathcal{U}(\pm 1/\sqrt{n_{in}})$, pas He ; `nn.init.kaiming_normal_` pour He) ; comment vérifier (mesurer l'écart-type par couche).

</details>

### 16.E3 — Calculer de tête le nombre de paramètres d'un MLP

<details><summary>Indice 1</summary>

$P = \sum_\ell (n_{\ell-1} n_\ell + n_\ell)$, puis 4 octets par paramètre en `float32`.

</details>
<details><summary>Indice 2</summary>

Calcule les produits couche par couche, ajoute les biais. Pour la dernière question, distingue ce qui dépend du réseau (les paramètres) de ce qui dépend des données traitées (les activations, le cache).

</details>
<details><summary>Indice 3</summary>

$1000 \times 500 + 500$ pour la première couche ; fais de même pour (500, 100) et (100, 10), additionne, multiplie par 4 octets. Puis : la taille du batch change-t-elle le nombre de poids ? Et la mémoire des activations ?

</details>

### 16.E4 — Feed-forward, récurrent, graphe acyclique : expliquer la différence

<details><summary>Indice 1</summary>

Un mot clé : la boucle. Un deuxième : le déroulage.

</details>
<details><summary>Indice 2</summary>

Définis le DAG, puis le RNN (un état qui revient à l'entrée au pas suivant), puis ce qu'on fait pour l'entraîner (le dérouler). Termine par la rétropropagation, qui parcourt le graphe à l'envers.

</details>
<details><summary>Indice 3</summary>

Structure : feed-forward = un DAG, chaque exemple traité indépendamment, une sortie fonction de l'entrée seule ; récurrent = une boucle, un état qui garde la mémoire des pas précédents, une sortie qui dépend de toute la séquence ; déroulé dans le temps, un RNN redevient un DAG (une copie par pas, des poids partagés) ; la rétropropagation n'ajoute aucune boucle : elle parcourt les arêtes à l'envers ; un exemple concret de chaque.

</details>

<a id="notebook"></a>

## Notebook

### Ex 16.11 — Prédire les formes des sorties avant d'exécuter 🔮

<details><summary>Indice 1</summary>

Deux règles : le produit $(a, b) \times (b, c)$ a la forme $(a, c)$ ; le broadcasting aligne les formes par la droite, et une dimension 1 se recopie.

</details>
<details><summary>Indice 2</summary>

Écris les deux formes l'une sous l'autre, alignées à droite. Pour un tableau 1-D, `x @ W` donne un résultat 1-D. Pour d) et f), la forme de `b1.reshape(-1, 1)` est `(128, 1)` : compare-la à `(64, 128)`, puis à `(128, 128)`.

</details>
<details><summary>Indice 3</summary>

a) Écris (64, 784) sous (784, 128) : quelles dimensions doivent coïncider, et lesquelles restent ? Puis aligne le biais (128,) par la droite. Pour les autres : b) que donne (784,) × (784, 128) ? c) quelle forme a `W1_11.T`, et le produit est-il possible ? d) et f) aligne `(128, 1)` sous la forme du produit : les dimensions sont-elles égales, ou l'une vaut-elle 1 ? e) suis les formes de gauche à droite. N'exécute l'expérience qu'après avoir écrit tes six prédictions.

</details>

### Ex 16.12 — Compter les paramètres par programme : count_parameters 🔨

<details><summary>Indice 1</summary>

Parcours les couches par paires de tailles voisines : `zip(layer_sizes[:-1], layer_sizes[1:])`.

</details>
<details><summary>Indice 2</summary>

Valide d'abord : au moins 2 tailles, chacune un entier (Python ou NumPy : `isinstance(n, (int, np.integer))`), pas un booléen (`isinstance(n, bool)` est vrai pour `True`), et ≥ 1. Puis additionne `n_in * n_out`, plus `n_out` si `bias`, et convertis le total avec `int(...)`.

</details>
<details><summary>Indice 3</summary>

```python
def count_parameters(layer_sizes, bias=True):
    sizes = list(layer_sizes)
    ...                                          # ValueError: fewer than 2 sizes, a bool, a non-integer, a size < 1
    total = 0
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        total += ...                             # weights, plus biases if bias
    return int(total)
```

</details>

### Ex 16.13 — La couche dense : dense_forward 🔨

<details><summary>Indice 1</summary>

Le calcul tient en une ligne, `x @ W + b` ; l'essentiel du travail est dans les vérifications de forme.

</details>
<details><summary>Indice 2</summary>

Convertis : `x = np.asarray(x, dtype=np.float64)`, `W = np.asarray(W, dtype=np.float64)`. Vérifie `W.ndim == 2`, `x.ndim in (1, 2)`, `x.shape[-1] == W.shape[0]` ; si `b` n'est pas `None`, convertis-le et vérifie `b.shape == (W.shape[1],)`. Un message qui cite les formes aide l'utilisateur.

</details>
<details><summary>Indice 3</summary>

```python
def dense_forward(x, W, b=None):
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    ...                                          # ValueError: W not 2-D, x not 1-D or 2-D, x.shape[-1] != W.shape[0]
    z = x @ W
    if b is not None:
        ...                                      # b as float64, of shape (W.shape[1],) exactly, then added
    return z
```

</details>

### Ex 16.14 — Dix façons d'initialiser : init_weights 🔨

<details><summary>Indice 1</summary>

Toutes les méthodes aléatoires se ramènent à deux tirages : `rng.uniform(-a, a, size)` ou `rng.normal(0.0, s, size)`. Il suffit de calculer la borne $a$ ou l'écart-type $s$ de chaque méthode (table du §16.4.1).

</details>
<details><summary>Indice 2</summary>

Range les méthodes dans un dictionnaire `nom → (loi, fonction qui calcule a ou s à partir de n_in et n_out)`, avec les alias `xavier_*` qui renvoient aux `glorot_*`. Vérifie la méthode **avant** les tailles, puis les tailles (des entiers ≥ 1, pas des booléens). `rng = np.random.default_rng(rng)` accepte `None` ou un générateur. Traite à part `"zeros"` et `"constant"` (`np.zeros`, `np.full`).

</details>
<details><summary>Indice 3</summary>

```python
SPREADS = {"lecun_uniform": ("uniform", lambda n_in, n_out: np.sqrt(3 / n_in)),
           "lecun_normal": ("normal", lambda n_in, n_out: np.sqrt(1 / n_in)),
           ...}                                      # glorot_*, he_*, and the aliases xavier_*

def init_weights(n_in, n_out, method="he_normal", rng=None, *, scale=0.05):
    ...                                              # ValueError: unknown method (first), sizes not ints >= 1
    if method == "zeros":
        return np.zeros((n_in, n_out))
    ...                                              # "constant", "uniform" and "normal" use scale
    law, spread = SPREADS[method]
    rng = np.random.default_rng(rng)
    ...                                              # one draw of shape (n_in, n_out) with rng.uniform or rng.normal
```

</details>

### Ex 16.15 — Reproduire les figures 16.5 et 16.6, puis comparer Glorot et He 🎨

<details><summary>Indice 1</summary>

`ax.hist(values, bins=..., density=True)` normalise l'histogramme pour que son aire fasse 1 : il se compare alors directement à une densité.

</details>
<details><summary>Indice 2</summary>

Tire avec `init_weights(200, 100, "uniform", rng, scale=0.05)` (20 000 poids) et `init_weights(200, 100, "normal", rng, scale=0.05 / 3)`, puis `.ravel()`. Densités : $1/(2a)$ sur $[-a, a]$, et $\frac{1}{\sigma\sqrt{2\pi}} e^{-x^2/(2\sigma^2)}$. Pour le troisième panneau, deux histogrammes `histtype="step"` sur les mêmes axes.

</details>
<details><summary>Indice 3</summary>

```python
def draw_15(rng):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    uniform = mylearn.nn.layers.init_weights(200, 100, "uniform", rng, scale=0.05).ravel()
    axes[0].hist(uniform, bins=50, density=True, alpha=0.5)
    ...                                              # the flat density 1/(2a) on top; then the normal law; then Glorot vs He
    return fig
```
Pour a) à c), applique les formules : hauteur $1/(2a)$, pic $1/(\sigma\sqrt{2\pi})$ avec $\sigma = 0{,}05/3$, et le rapport $\sqrt{2/n} \,/\, \sqrt{2/(2n)}$.

</details>

### Ex 16.16 — Ordre topologique d'un graphe : topological_order 🔨

<details><summary>Indice 1</summary>

L'algorithme de Kahn de l'encadré du §16.2 ; pour toujours sortir le plus petit nœud prêt, un **tas** (`heapq`) garde les nœuds prêts triés.

</details>
<details><summary>Indice 2</summary>

Commence par rassembler **tous** les nœuds : les clés et tous les enfants. Compte les parents de chacun (une arête parent → enfant ajoute 1 à l'enfant). Mets dans le tas les nœuds sans parent. Puis, tant que le tas n'est pas vide : sors le plus petit, ajoute-le à l'ordre, retire 1 à chacun de ses enfants et pousse ceux qui tombent à 0. À la fin, si l'ordre est plus court que la liste des nœuds, il y a un cycle.

</details>
<details><summary>Indice 3</summary>

```python
import heapq

def topological_order(graph):
    nodes = set(graph) | {child for children in graph.values() for child in children}
    parents = {node: 0 for node in nodes}
    ...                                              # count the parents of every child
    ready = [node for node in nodes if parents[node] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)                  # the smallest ready node
        ...                                          # place it, then free its children
    ...                                              # ValueError if some nodes were never placed (a cycle)
```

</details>

### Ex 16.17 — Empiler les couches : init_mlp et mlp_forward 🔨

<details><summary>Indice 1</summary>

`init_mlp` appelle `init_weights` une fois par couche, avec le **même** générateur ; `mlp_forward` appelle `dense_forward` une fois par couche, dans une boucle.

</details>
<details><summary>Indice 2</summary>

`init_mlp` : valide les tailles (comme `count_parameters`), crée `rng = np.random.default_rng(rng)` une seule fois, puis une liste de `(init_weights(n_in, n_out, method, rng), np.zeros(n_out))`. `mlp_forward` : refuse une liste vide ; `a = x` ; pour chaque couche, `z = dense_forward(a, W, b)`, ajoute `(a, z)` au cache **avant** l'activation, puis `a = activation(z)` pour une couche cachée, `output_activation(z)` pour la dernière (`None` : garder `z`).

</details>
<details><summary>Indice 3</summary>

```python
def mlp_forward(x, params, activation=None, output_activation=None, return_cache=False):
    ...                                              # ValueError if params is empty; a <- x as float64; cache <- []
    for k, (W, b) in enumerate(params):
        ...                                          # z of the layer, with dense_forward
        cache.append((a, z))                         # BEFORE the activation
        f = output_activation if k == len(params) - 1 else activation
        ...                                          # the new a (z itself when f is None)
    ...                                              # (a, cache) or a
```

</details>

### Ex 16.18 — Relire un MLPClassifier de scikit-learn avec ton mlp_forward 📦

<details><summary>Indice 1</summary>

`coefs_` et `intercepts_` sont deux listes de même longueur : `zip` les assemble en couples `(W, b)`.

</details>
<details><summary>Indice 2</summary>

Pour a), additionne `W.size + b.size`. Pour les probabilités : `mylearn.nn.layers.mlp_forward(X_TEST_18, params_18, activation=relu, output_activation=softmax_18)`. `proba.argmax(axis=1)` donne la classe prédite ; compare-la à `CLF_18.predict(X_TEST_18)` avec `np.mean(... == ...)`.

</details>
<details><summary>Indice 3</summary>

`params_18 = list(zip(CLF_18.coefs_, CLF_18.intercepts_))` ; puis une seule passe avant avec **ton** `mlp_forward` donne `proba_18`. Pour c), compare l'argmax de chaque ligne à `CLF_18.predict` ; pour d), le plus grand écart absolu avec `predict_proba` ; pour e), compare l'argmax aux vrais labels ; pour f), refais la passe avant avec une autre activation.

</details>

### Ex 16.19 — Initialisation constante : les neurones restent-ils jumeaux ? 🔮

<details><summary>Indice 1</summary>

Relis ∂ 16.6 : ses hypothèses supposent-elles une rétropropagation, ou seulement des dérivées partielles ?

</details>
<details><summary>Indice 2</summary>

Pour b), demande-toi à combien de neurones cachés équivaut un réseau dont les quatre neurones sont jumeaux, et si un tel réseau peut séparer un « ou exclusif » (il faut au moins deux frontières). Pour c), un point symétrique est-il stable (les petits écarts diminuent) ou instable (ils grandissent) ? Et à quelle vitesse un écart de $10^{-4}$ peut-il devenir assez grand ?

</details>
<details><summary>Indice 3</summary>

a) Les différences finies perturbent un paramètre à la fois et mesurent la loss : traitent-elles $v_1$ et $v_2$ de la même façon à un point jumeau ? Relis ce que ∂ 16.6 en déduit. b) Que peut faire, au mieux, un réseau à un seul neurone caché sur quatre amas en « ou exclusif », comparé à un réseau à quatre neurones différents ? c) Raisonne en deux temps : la symétrie est-elle brisée ? Et que faut-il pour que des neurones presque jumeaux deviennent assez différents pour se partager le travail ? N'exécute l'expérience qu'après avoir écrit tes trois prédictions.

</details>

### Ex 16.20 — Chasse au bug : transposée, biais mal diffusé, variance prise pour un écart-type 🐛

<details><summary>Indice 1</summary>

Le titre nomme les trois erreurs. Pour chacune, demande-toi pourquoi un réseau **carré** et un batch de **64** images l'ont cachée.

</details>
<details><summary>Indice 2</summary>

Point 1 : que reçoit `rng.normal` en deuxième argument, une variance ou un écart-type ? Point 2 : quelle forme a `b.reshape(-1, 1)`, et sur quel axe s'aligne-t-elle ? Point 3 : avec `W` de forme `(n_in, n_out)` et une ligne par exemple, faut-il transposer ?

</details>
<details><summary>Indice 3</summary>

Première erreur : `rng.normal(0.0, 2 / n_in, ...)` donne l'écart-type $2/n_{in}$, alors que He demande une **variance** $2/n_{in}$ : il faut `rng.normal(0.0, np.sqrt(2 / n_in), size=(n_in, n_out))`. Pour les deux autres, regarde la ligne `z = a @ W.T + b.reshape(-1, 1)` : quelle forme doit avoir le biais pour s'ajouter à chaque ligne, et `W` doit-il être transposé dans la convention du chapitre ?

</details>

### Ex 16.21 — Rendre l'aléatoire reproductible et le prouver avec pytest 🛠️

<details><summary>Indice 1</summary>

Lis les trois versions buggées : chacune trahit le contrat d'une façon différente. Un bon test est une propriété du contrat que la version buggée viole.

</details>
<details><summary>Indice 2</summary>

Quatre propriétés : la même graine redonne les mêmes paramètres ; deux graines différentes donnent des poids différents ; deux couches de même forme ont des poids différents ; le résultat ne dépend pas de l'état global `np.random` (fixe-le à deux valeurs différentes avec `np.random.seed`, et appelle avec le même `rng`).

</details>
<details><summary>Indice 3</summary>

```python
def test_same_seed_gives_the_same_parameters():
    first = init_mlp([5, 4, 4, 2], rng=np.random.default_rng(42))
    second = init_mlp([5, 4, 4, 2], rng=np.random.default_rng(42))
    assert all(np.array_equal(W1, W2) for (W1, _), (W2, _) in zip(first, second))
```
Écris sur le même modèle un test pour deux graines différentes, un pour deux couches de même forme (par exemple `[4, 4, 4]`), et un pour l'état global. Puis range-les dans `my_tests_21`.

</details>

### Ex 16.22 — Lire des histogrammes d'activations couche par couche 📈

<details><summary>Indice 1</summary>

Trois signatures à chercher : des histogrammes qui **rétrécissent** de couche en couche (l'évanouissement), des **pics en ±1** (la saturation de tanh), et des histogrammes qui gardent à peu près leur largeur.

</details>
<details><summary>Indice 2</summary>

$\mathcal{N}(0 ;\ 1)$ donne des sommes pondérées d'écart-type $\sqrt{784}$ dès la couche 1 : tanh sature. $\mathcal{N}(0 ;\ 0{,}05^2)$ sur 256 entrées a le gain $256 \times 0{,}0025 = 0{,}64 < 1$ : chaque couche rétrécit. Pour c), l'écart-type d'un histogramme en cloche se lit à peu près comme la moitié de sa largeur à mi-hauteur.

</details>
<details><summary>Indice 3</summary>

a) Le panneau dont les six histogrammes sont des pics en $-1$ et $+1$ est celui de $\mathcal{N}(0 ;\ 1)$ : il donne la troisième lettre. Pour les deux autres : lequel rétrécit le plus vite de la couche 1 à la couche 6 ? b) Au dernier histogramme du panneau saturé, où est presque toute la masse ? c) Estime la demi-largeur de l'histogramme de la couche 6 du panneau qui rétrécit le plus vite, et compare-la aux quatre intervalles. d) Compare la largeur des histogrammes de Glorot entre la couche 1 et la couche 6.

</details>

### Ex 16.23 — Activations à travers 20 couches selon l'initialisation 🔬

<details><summary>Indice 1</summary>

Tout se fait avec tes fonctions : `init_weights` couche par couche (pour pouvoir passer `scale=0.05`), puis `mlp_forward(..., return_cache=True)`, dont le cache contient les `z`.

</details>
<details><summary>Indice 2</summary>

`profile_23` : une couche à la fois avec `init_weights` (le même `rng`, `scale=0.05` pour `"normal"` seulement) et des biais nuls ; puis une passe avant avec le cache, et l'écart-type de chaque `z`. `dead_23` : un réseau de `init_mlp` en He, la passe avant en ReLU, puis la dernière somme pondérée du cache : un neurone est mort si sa colonne est négative ou nulle pour toutes les images.

</details>
<details><summary>Indice 3</summary>

```python
def profile_23(activation, method, X, rng):
    extra = {"scale": 0.05} if method == "normal" else {}
    params = ...                                     # one (init_weights(...), zeros) per layer, in order, with rng
    ...                                              # mlp_forward with the cache, then the 20 stds of z
```
Pour `dead_23`, la ligne clé : `float(np.mean((z_last <= 0).all(axis=0)))`, où `z_last` est la somme pondérée de la dernière couche, de forme `(512, 256)`.

</details>

### Ex 16.24 — Défi : garder vivant un réseau ReLU de 50 couches 🏆

<details><summary>Indice 1</summary>

Mesure d'abord : écris une boucle qui passe `X_sample` couche par couche et affiche l'écart-type des `z` et la part des neurones morts. Essaie `init_mlp(SIZES_24, "he_normal", rng)` : quel problème vient de l'échelle des pixels, et lequel n'en vient pas ?

</details>
<details><summary>Indice 2</summary>

Deux problèmes séparés. L'**échelle** : rien n'empêche de régler chaque couche sur les données, en divisant ses poids par l'écart-type mesuré de ses sorties sur `X_sample` (LSUV). Les **morts** : des entrées toutes positives (les pixels, puis les sorties de ReLU) donnent à chaque neurone une somme pondérée de moyenne loin de 0 ; le biais peut corriger cette moyenne. Enfin, mesure tes résultats sur une **autre** partie de `X_sample` que celle qui t'a servi à régler, pour ne pas te faire d'illusions.

</details>
<details><summary>Indice 3</summary>

```python
def init_24(layer_sizes, X_sample, rng):
    params, a = [], np.asarray(X_sample, dtype=float)
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = ...                                      # random weights (He, or orthogonal: np.linalg.qr of a Gaussian matrix)
        z = a @ W
        ...                                          # a bias that sets the mean of each neuron, then a scale that gives std 1 on the sample
        params.append((W, b))
        a = np.maximum(0.0, a @ W + b)
    return params
```
Si les écarts-types de contrôle s'éloignent de 1 dans les dernières couches, demande-toi où placer la moyenne de chaque neurone : exactement à 0, ReLU coupe la moitié de ses valeurs à chaque couche ; un peu au-dessus, il reste actif sur la plupart des images. Essaie plusieurs décalages, mesurés en écarts-types du neurone.

</details>
