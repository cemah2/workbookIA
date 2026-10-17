# 12 · Préparation des données — indices

> **Mode d'emploi.** Cherche d'abord seul pendant 15 minutes. Si tu bloques, ouvre **l'indice 1** (la direction), cherche encore 5 minutes, puis l'indice 2 (la méthode), puis l'indice 3 (presque la solution). Ouvrir un indice n'est pas un échec : c'est ne pas chercher avant qui en est un. Note dans ton journal les exercices où tu as eu besoin de l'indice 3 : ce sont eux qu'il faudra refaire dans une semaine.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ ⚖️ Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Notebook](#notebook)

<a id="quiz"></a>

## 🧠 Quiz

### 12.Q1 — La règle d'or de la préparation

<details><summary>Indice 1</summary>

Relis les deux temps de la règle d'or (fiche §12.2) : où les paramètres sont-ils **appris**, et sur quelles données sont-ils **appliqués** ?

</details>
<details><summary>Indice 2</summary>

Pour a) et b), demande-toi quelles données le modèle a vues pendant son entraînement, et sous quelle forme : sous quelle forme doit-il recevoir les données suivantes ? Pour c) et d), une étape de préparation n'est pas forcément une mise à l'échelle : découper, retirer, recadrer en sont aussi.

</details>
<details><summary>Indice 3</summary>

a) La moyenne et l'écart-type doivent venir des seules données que le modèle a eues pour apprendre : l'entraînement, sans le test (A ferait entrer le test dans les paramètres, D donnerait deux transformations différentes). b) Pose la même question : la mesure de production doit-elle recevoir des paramètres neufs, ou ceux de l'entraînement ? c) Le découpage des silences a-t-il été rejoué en service ? d) Retirer une colonne est-il une étape de préparation comme une autre ?

</details>

### 12.Q2 — Numérique, ordinale ou nominale ?

<details><summary>Indice 1</summary>

Pour chaque feature, deux questions : est-ce un nombre sur lequel une moyenne a un sens ? Sinon, les catégories ont-elles un ordre qui a un sens pour le problème ?

</details>
<details><summary>Indice 2</summary>

Méfie-toi des chiffres : un code peut s'écrire avec des chiffres sans être un nombre. Pour b), l'encodeur ne connaît pas le sens des catégories : quand on ne lui donne aucun ordre, il les trie selon un critère purement mécanique, appliqué à du texte. Pour c), relis l'encadré ⚠️ du §12.3.

</details>
<details><summary>Indice 3</summary>

a) Élément 1 : une masse en grammes se compare et se moyenne, c'est **Q**. Pour les cinq autres, applique les deux questions de l'indice 1 ; pour le code postal, demande-toi ce que vaudrait la moyenne de deux codes postaux. b) Trie les quatre textes « S », « M », « L », « XL » comme le ferait `sorted()` en Python, caractère par caractère. c) Un ordre choisi au hasard permet-il de dire qu'une île est « entre » les deux autres ?

</details>

### 12.Q3 — Pourquoi un one-hot plutôt qu'un entier ?

<details><summary>Indice 1</summary>

Avec une seule colonne codée par des entiers, la prédiction contient le terme $w \times \text{code}$ : que fait ce terme quand le code augmente de 1 ?

</details>
<details><summary>Indice 2</summary>

Un one-hot crée une colonne par catégorie ; `drop="first"` retire celle de la première catégorie (dans l'ordre alphabétique). Pour d), range les trois catégories comme le ferait `sorted()`, puis place le 1 à la position de « tram ». Pour e), une distance entre deux codes entiers est la valeur absolue de leur différence.

</details>
<details><summary>Indice 3</summary>

a) De bus (0) à train (1), le terme $w \times \text{code}$ passe de $0$ à $w$ : il augmente de $w$. Fais le même calcul de train (1) à tram (2), et compare. b) et c) Compte une colonne par catégorie, puis retire celle de la première. d) Compare « train » et « tram » lettre à lettre jusqu'à la première différence. e) Calcule $|0 - 2|$ et $|0 - 1|$.

</details>

### 12.Q4 — Doublons, NaN et points aberrants

<details><summary>Indice 1</summary>

Relis les cinq vérifications de la fiche (§12.4) : formats, orthographes, doublons, valeurs manquantes, points aberrants.

</details>
<details><summary>Indice 2</summary>

Pour a), compare 38 000 g aux autres valeurs et à la masse d'un Adélie, puis demande-toi ce que supposerait chacune des explications A à D. Pour c), demande-toi ce que pandas sait, seul, des conventions d'un fichier. Pour d), Python attend un point comme séparateur décimal. Pour e), relis ce que la fiche dit de l'égalité avec `NaN`.

</details>
<details><summary>Indice 3</summary>

a) 38 000 g vaut dix fois les autres masses, bien au-delà des 3 à 5 kg d'un Adélie : une erreur de saisie (B), à vérifier dans la source avant toute correction. b) Une ligne répétée entre-t-elle une ou deux fois dans une somme ? c) Pour pandas, un « ? » est-il différent d'un mot quelconque, sans l'option `na_values` ? d) Que fait `float()` d'un texte qu'il ne sait pas lire ? e) Que vaut `np.nan == np.nan` ?

</details>

### 12.Q5 — Normaliser ou standardiser ?

<details><summary>Indice 1</summary>

Le min-max divise par l'étendue, qui ne dépend que du minimum et du maximum ; la standardisation divise par l'écart-type. Écris chacune sous la forme $a\,x + b$.

</details>
<details><summary>Indice 2</summary>

Pour a), calcule l'étendue, puis place les petites valeurs. Pour b), imagine un dataset de 1 000 valeurs dont une seule est très loin des autres : son z-score peut-il dépasser 3 ? Pour c) et d), standardise à la main les cinq valeurs 1, 1, 1, 1 et 10, puis compare la forme des deux listes : où se trouve la valeur isolée, avant et après ?

</details>
<details><summary>Indice 3</summary>

a) L'étendue vaut $1\,000 - 1 = 999$ : la valeur 4 devient $3/999$. Placent-elles les quatre premières valeurs régulièrement entre 0 et 1, ou toutes près de 0 ? b) Avec $n$ valeurs ordinaires et une seule très éloignée, l'écart-type ne vaut qu'environ $1/\sqrt{n}$ de l'écart entre cette valeur et la moyenne : que devient son z-score quand $n = 1\,000$ ? c) et d) Les z-scores de 1, 1, 1, 1 et 10 valent $-0{,}5$ quatre fois et $2$ : la forme a-t-elle changé ? Que dire alors d'une feature très asymétrique ?

</details>

### 12.Q6 — Données de test hors de [0, 1] : bug ou normal ?

<details><summary>Indice 1</summary>

`transform` applique la formule du min-max avec le minimum et le maximum de l'**entraînement**, quelle que soit la valeur reçue.

</details>
<details><summary>Indice 2</summary>

a) $x' = (x - \min)/(\max - \min)$, avec $\min = -5$ et $\max = 25$. c) Relis ce que fait l'option `clip` (fiche §12.8). d) Écris la moyenne du test transformé en fonction de la moyenne du test et de celle de l'entraînement.

</details>
<details><summary>Indice 3</summary>

a) $\frac{30 - (-5)}{25 - (-5)} = \frac{35}{30} \approx 1{,}17$. b) Une valeur plus chaude que tout l'entraînement doit-elle tomber dans $[0, 1]$ ? c) Applique `clip` au résultat de a). d) Le test a-t-il servi à calculer la moyenne retirée ? Que vaut alors, en général, la moyenne du test transformé ?

</details>

### 12.Q7 — Univarié ou multivarié ?

<details><summary>Indice 1</summary>

Relis les définitions d'une transformation univariée et d'une transformation multivariée (fiche §12.5.4), puis applique-les à chaque élément.

</details>
<details><summary>Indice 2</summary>

Pour chaque élément, demande-toi : si je change les valeurs d'**une autre** colonne de l'entraînement, la transformation de celle-ci change-t-elle ? Pour l'élément 5, rappelle-toi comment on trouve les plus proches voisins d'un exemple.

</details>
<details><summary>Indice 3</summary>

a) Élément 1 : `StandardScaler` calcule une moyenne et un écart-type par colonne, avec les seules valeurs de cette colonne : **U**. Pour les cinq autres, pose la question de l'indice 2 ; pour le min-max global, d'où viennent le minimum et le maximum ? Pour la PCA, une composante mélange-t-elle des colonnes ? b) D'après la définition, de quelles valeurs dépend la transformation de la colonne 1 ?

</details>

### 12.Q8 — Sélectionner ou réduire la dimension ?

<details><summary>Indice 1</summary>

Relis les deux façons de réduire le nombre de features (fiche §12.6 et §12.7) : que deviennent les colonnes d'origine dans chacune ?

</details>
<details><summary>Indice 2</summary>

Pour chaque opération de a), regarde les colonnes qui en sortent : sont-elles des colonnes d'origine, ou des combinaisons ? Pour b), une sélection qui regarde la cible apprend-elle quelque chose des données ? Sur quelles données ? Pour c), relis la définition d'une composante.

</details>
<details><summary>Indice 3</summary>

a) Élément 1 : retirer une colonne constante laisse les autres intactes : **S**. Même question pour la PCA, les 5 features corrélées, l'IMC et `VarianceThreshold` : les colonnes qui sortent sont-elles celles d'origine ? b) Si la cible du test sert à choisir les colonnes, qu'arrive-t-il au score de test (rappelle-toi 8.25) ? c) Une composante est-elle une colonne, ou un mélange de colonnes ?

</details>

### 12.Q9 — Ce que fait (et ne fait pas) une PCA

<details><summary>Indice 1</summary>

Relis « La PCA en une phrase » et « Ce que la PCA ne fait pas » (fiche §12.7.1).

</details>
<details><summary>Indice 2</summary>

Pour a) et b), relis comment la première composante est définie, puis la deuxième par rapport à elle. Pour c), regarde les arguments que reçoit `fit` dans le pseudo-code de la PCA de la fiche (§12.7.1). Pour d), écris la coordonnée d'un exemple sur une composante. Pour e), que devient la variance d'une feature quand on multiplie toutes ses valeurs par 1 000 ? Pour f), une direction de faible variance peut-elle séparer deux classes ?

</details>
<details><summary>Indice 3</summary>

a) La première composante est, par définition, la direction unitaire qui maximise la variance des projections des données centrées : **Vrai**. b) Comment cherche-t-on la deuxième composante par rapport à la première ? c) La méthode reçoit-elle `y` ? d) Une composante est-elle une somme pondérée ? e) La variance est multipliée par $1\,000^2$ : vers quelle feature la direction de variance maximale tourne-t-elle ? f) Pense à deux classes allongées côte à côte, séparées seulement dans la petite direction.

</details>

### 12.Q10 — Quelle découpe pour ces données ?

<details><summary>Indice 1</summary>

Relis les trois découpes du §12.9 de la fiche : chacune se définit par la tranche du tableau qu'elle utilise.

</details>
<details><summary>Indice 2</summary>

Pour chaque élément, cherche **ce qu'il faut connaître** pour transformer une case : d'autres cases de la même ligne, d'autres cases de la même colonne, ou seulement une constante fixée d'avance. Pour b), une telle transformation utilise-t-elle d'autres exemples que celui qu'elle transforme ?

</details>
<details><summary>Indice 3</summary>

a) Élément 1 : diviser par 255 n'utilise que la case et une constante : **E**. Pour les cinq autres, cherche de quoi dépend la transformation d'une case : du maximum de sa ligne, des valeurs de sa colonne, ou d'une constante ? Pour `Normalizer`, la norme de quel vecteur ? b) Si la transformation d'un exemple n'utilise que cet exemple, par où une information du test atteindrait-elle l'entraînement ?

</details>

### 12.Q11 — Où se cache la fuite ?

<details><summary>Indice 1</summary>

Une fuite par le prétraitement apparaît dès qu'une étape **apprend** quelque chose (une statistique, un choix de colonnes) sur des données qui servent ensuite à évaluer.

</details>
<details><summary>Indice 2</summary>

Pour chaque procédure : l'étape apprend-elle quelque chose des données ? Si oui, sur quelles lignes, et ces lignes servent-elles ensuite à évaluer (le test, ou le fold de validation de chaque tour) ? Pour b), relis l'encadré ⚠️ du §12.10.

</details>
<details><summary>Indice 3</summary>

(A) La moyenne et l'écart-type sont calculés sur toutes les lignes, y compris celles de chaque futur fold de validation : c'est une fuite. Pose la même question à B (le pipeline est-il réajusté dans chaque fold ?), C (la médiane voit-elle le test ?), D (le scaler voit-il la validation ?), E (le choix des features voit-il la cible de la validation ?) et F (diviser par 255 apprend-il quelque chose ?). b) `cross_val_score` peut-il défaire un `fit` fait avant son appel ?

</details>

<a id="rappels"></a>

## 🔁 Rappels

### 12.R1 — Ch. 11 : représentation, évaluation, optimisation — où placer la préparation ?

<details><summary>Indice 1</summary>

Relis les trois ingrédients du ch. 11 : ce que le modèle **peut exprimer**, ce qui **juge** une solution, la méthode qui **cherche**.

</details>
<details><summary>Indice 2</summary>

Pour chaque modification, demande-toi ce qu'elle change : l'ensemble des fonctions possibles, la mesure de leur qualité, ou le chemin pour trouver la meilleure ? Pour b), une régression linéaire sur $(x - \mu)/\sigma$ peut-elle exprimer d'autres fonctions qu'une régression sur $x$ ?

</details>
<details><summary>Indice 3</summary>

a) Avec $x_1^2$, la régression peut tracer des paraboles qu'elle ne pouvait pas tracer avant : l'ensemble des fonctions possibles change, c'est la représentation (**A**). b) $w\,\frac{x - \mu}{\sigma} + b$ se réécrit $w' x + b'$ : les fonctions possibles sont-elles différentes ? Qu'est-ce qui change alors pour la descente de gradient (ch. 5) ? c) Le F1-score sert-il à exprimer, à juger ou à chercher ? d) Une information absente de toutes les features peut-elle apparaître dans une prédiction ?

</details>

### 12.R2 — Ch. 9 : pourquoi la pénalité ridge dépend de l'échelle des features

<details><summary>Indice 1</summary>

La prédiction contient le terme $w\,x$ ; la pénalité contient $\alpha\,w^2$. Change d'unité dans le premier, puis regarde ce que devient le second.

</details>
<details><summary>Indice 2</summary>

Pour garder les mêmes prédictions, il faut $w'\,x' = w\,x$ avec $x' = 100\,x$. Une fois $w'$ trouvé, calcule $w'^2$ en fonction de $w^2$. Pour d), que valent les z-scores de $100\,x$ comparés à ceux de $x$ ?

</details>
<details><summary>Indice 3</summary>

a) $w'\,x' = w\,x$ et $x' = 100\,x$ donnent $w' \times 100\,x = w\,x$, d'où $w' = w / 100$ : c'est **C**. b) Élève ce $w'$ au carré et compare à $w^2$. c) Une pénalité plus petite pour le même effet sur les prédictions : la feature est-elle plus ou moins freinée ? d) Le z-score de $100\,x$ est $\frac{100\,x - 100\,\mu}{100\,\sigma}$ : simplifie.

</details>

### 12.R3 — Ch. 5 : descente de gradient dans une vallée très allongée

<details><summary>Indice 1</summary>

Le gradient de $x^2 + 100\,y^2$ se calcule terme par terme ; un pas de descente remplace chaque coordonnée par elle-même moins $\eta$ fois sa dérivée.

</details>
<details><summary>Indice 2</summary>

Un pas fait $y \leftarrow y - \eta \times 200\,y = (1 - 200\,\eta)\,y$ : la descente converge sur $y$ quand le facteur est strictement entre $-1$ et $1$. Pour $x$, fais le même calcul avec la dérivée de $x^2$. Pour d), réécris $f$ avec $y'$, puis refais le même raisonnement.

</details>
<details><summary>Indice 3</summary>

a) $\nabla f(x, y) = (2x,\ 200\,y)$, soit $(2, 200)$ en $(1, 1)$. b) Résous $-1 < 1 - 200\,\eta < 1$ pour $\eta > 0$ et garde la borne supérieure. c) Calcule $1 - 2 \times 0{,}009$. d) Avec $f = x^2 + y'^2$, le facteur de chaque coordonnée vaut $1 - 2\eta$ : résous $-1 < 1 - 2\eta < 1$. e) Compare les bornes de b) et de d).

</details>

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

### Ex 12.1 — One-hot à la main sur Penguins ✏️

<details><summary>Indice 1</summary>

Range d'abord les catégories de chaque feature par ordre alphabétique : elles donnent l'ordre des colonnes de chaque bloc.

</details>
<details><summary>Indice 2</summary>

Le bloc de `island` a une colonne par île, celui de `sex` une colonne par sexe ; `flipper_length_mm` vient en dernier, sans changement. `drop="first"` retire la colonne de la première catégorie de chaque bloc ; `drop="if_binary"` ne le fait que pour les features à deux catégories. Une catégorie inconnue, avec `handle_unknown="ignore"`, reçoit un bloc de zéros. L'encodage ordinal donne à chaque catégorie sa position dans l'ordre, en commençant à 0.

</details>
<details><summary>Indice 3</summary>

a) Trois îles (Biscoe, Dream, Torgersen) donnent 3 colonnes, deux sexes (female, male) 2 colonnes, et la nageoire 1 colonne : **6**. b) Écris le bloc de Torgersen (un seul 1, à sa position), puis celui de female, puis la nageoire. c) Retire une colonne à chaque bloc catégoriel. d) Après `drop="first"`, il reste les colonnes de Dream, de Torgersen et de male : remplis-les pour le manchot 2, puis la nageoire. e) Quelle feature a exactement deux catégories ? f) Bloc `island` d'une île inconnue, puis le bloc de female, puis la nageoire. g) Position de Torgersen dans l'ordre alphabétique, en comptant à partir de 0.

</details>

### Ex 12.2 — Min-max et z-score de cinq valeurs ✏️

<details><summary>Indice 1</summary>

Le min-max a besoin du minimum et du maximum ; le z-score de la moyenne et de l'écart-type. Calcule ces quatre nombres d'abord.

</details>
<details><summary>Indice 2</summary>

L'écart-type avec ddof = 0 est la racine de la moyenne des carrés des écarts à la moyenne ; avec ddof = 1 (pandas), on divise la somme des carrés par $n - 1$ au lieu de $n$. Vers $[a, b]$ : $x' = a + (b - a)\,\frac{x - \min}{\max - \min}$.

</details>
<details><summary>Indice 3</summary>

a) $(14 + 17 + 21 + 23 + 25)/5 = 100/5 = 20$. b) Les écarts à la moyenne valent $-6$, $-3$, $1$, $3$ et $5$ : additionne leurs carrés, divise par 5, prends la racine. c) Pose $\frac{21 - 14}{25 - 14}$. d) et e) Pose $\frac{x - 20}{\sigma}$ avec le $\sigma$ de b). f) Recalcule $\sigma$ en divisant la somme des carrés par 4, puis le z-score de 25. g) Pose $-1 + 2 \times \frac{21 - 14}{25 - 14}$.

</details>

### Ex 12.3 — Mise à l'échelle univariée ou multivariée ✏️

<details><summary>Indice 1</summary>

En univarié, chaque feature a son minimum et son maximum ; en multivarié, un seul minimum (le plus petit des trois) et un seul maximum (le plus grand des trois) servent à toutes.

</details>
<details><summary>Indice 2</summary>

Le minimum global est celui de f3, le maximum global celui de f1. Pour d), écris l'étendue transformée d'une feature en fonction de son étendue d'origine. Pour e), prends une même valeur transformée, 0,7, dans deux colonnes : à quels montants correspond-elle avec chaque mise à l'échelle ?

</details>
<details><summary>Indice 3</summary>

a) f1 : $\frac{20 - 10}{30 - 10} = 0{,}5$ ; f2 : $\frac{4 - 0}{5 - 0} = 0{,}8$ ; f3 : $\frac{0 + 20}{40} = 0{,}5$ ; d'où $[0{,}5 ;\ 0{,}8 ;\ 0{,}5]$. b) Avec $\min = -20$ et $\max = 30$, pose $\frac{x + 20}{50}$ pour chacune des trois valeurs. c) L'étendue de f2 (5) divisée par l'étendue globale. d) Toutes les étendues sont divisées par le même nombre : que devient leur rapport ? e) Avec quelle transformation 0,7 veut-il dire le même montant dans la colonne du magasin 1 et dans celle du magasin 3 ?

</details>

### Ex 12.4 — Réappliquer la transformation : −10 °C, −50 °C et retour aux voitures ✏️

<details><summary>Indice 1</summary>

Deux transformateurs : un pour la température (appris sur −18 à 12), un pour les voitures (appris sur 120 à 870). Pour chaque question, demande-toi lequel s'applique, et dans quel sens (aller ou retour).

</details>
<details><summary>Indice 2</summary>

L'aller : $x' = \frac{x - \min}{\max - \min}$, avec le minimum et le maximum appris. Le retour : $x = \min + x'\,(\max - \min)$. Pour g) et h), mêmes idées avec $z = (x - \mu)/\sigma$ et $x = \mu + z\,\sigma$.

</details>
<details><summary>Indice 3</summary>

a) $\frac{-10 - (-18)}{12 - (-18)} = \frac{8}{30}$, à arrondir à 3 décimales. b) Pose $120 + 0{,}40 \times (870 - 120)$. c) et d) Même formule qu'en a), avec −50 et 20. e) Une valeur hors de la plage d'entraînement doit-elle rester dans $[0, 1]$ ? f) Pose $120 + (-0{,}1) \times 750$. g) Pose $\frac{-10 - (-3)}{6}$. h) Pose $480 + 0{,}5 \times 150$. i) Quel transformateur a appris le minimum et le maximum des voitures ?

</details>

### Ex 12.5 — Trois découpes d'un même tableau : échantillon, feature, élément ✏️

<details><summary>Indice 1</summary>

Par feature : chaque colonne avec son minimum et son maximum. Par échantillon : chaque ligne avec les siens. Par élément : la même formule pour chaque case.

</details>
<details><summary>Indice 2</summary>

Note d'abord les minimums et les maximums des trois colonnes, puis ceux des lignes demandées. La norme L2 d'une ligne est la racine de la somme des carrés de ses valeurs. Pour e), une transformation s'apprend si elle a besoin des **autres** exemples de l'entraînement.

</details>
<details><summary>Indice 3</summary>

a) Colonne f1 : de 1 à 5, donc $\frac{3 - 1}{5 - 1} = 0{,}5$ ; f2 : de 4 à 8, donc $\frac{8 - 4}{4} = 1$ ; f3 : de 10 à 30, donc $\frac{20 - 10}{20} = 0{,}5$ ; l'exemple 2 devient $[0{,}5 ;\ 1 ;\ 0{,}5]$. b) Ligne 1 : de 1 à 10 ; pose $\frac{x - 1}{10 - 1}$ pour chacune de ses trois valeurs. c) Calcule la norme de $(3, 8, 20)$, puis divise 3 par elle. d) Multiplie chaque valeur de l'exemple 3 par 0,9. e) Lesquelles utilisent les valeurs des autres lignes ? f) Applique les minimums et maximums des colonnes de a) à $(2, 2, 40)$, sans borner. g) Le minimum et le maximum de la ligne $(2, 2, 40)$ elle-même.

</details>

### Ex 12.6 — Montrer que la standardisation donne moyenne 0 et variance 1 ∂

<details><summary>Indice 1</summary>

Utilise les deux règles de 0B : la moyenne d'une transformation affine $a\,x + b$ vaut $a$ fois la moyenne plus $b$ ; sa variance vaut $a^2$ fois la variance. Écris $z_i$ sous la forme $a\,x_i + b$.

</details>
<details><summary>Indice 2</summary>

$z_i = \frac{1}{\sigma}\,x_i - \frac{\mu}{\sigma}$, avec $a = 1/\sigma$ et $b = -\mu/\sigma$. Pour 3, écris $\sigma_1^2$ (ddof = 1) en fonction de $\sigma_0^2$ (ddof = 0) : les deux ont la même somme de carrés. Pour 4, écris la moyenne et l'écart-type des $x'_i$ en fonction de $\mu$ et $\sigma$. Pour 5, une fonction affine croissante conserve l'ordre et les rapports d'écarts. Pour 6, qu'arrive-t-il quand on divise par 0 ?

</details>
<details><summary>Indice 3</summary>

1. La moyenne des $z_i$ vaut $\frac{1}{n}\sum_i \frac{x_i - \mu}{\sigma} = \frac{1}{\sigma}\left(\frac{1}{n}\sum_i x_i - \mu\right) = \frac{\mu - \mu}{\sigma} = 0$. 2. Pose $\frac{1}{n}\sum_i z_i^2$ (la moyenne est nulle) et fais sortir $\frac{1}{\sigma^2}$ de la somme. 3. La variance (ddof = 0) des $z_i$ vaut $\sigma_0^2/\sigma_1^2$, et $\sigma_1^2 = \frac{n}{n - 1}\,\sigma_0^2$ : simplifie, puis évalue pour $n = 5$. 4. La moyenne des $x'_i$ vaut $a\mu + b$ et leur écart-type $|a|\,\sigma$ : écris $z'_i$ et simplifie ; regarde le signe quand $a < 0$. 5. Évalue $x'$ en $m$ et en $M$, puis écris $x'_i - x'_j$ en fonction de $x_i - x_j$. 6. Tous les écarts à la moyenne sont nuls : que devient le numérateur avec le dénominateur remplacé par 1 ?

</details>

### Ex 12.7 — PCA à la main en 2D : covariance, axe principal, projection ✏️

<details><summary>Indice 1</summary>

Suis les étapes de la PCA (fiche §12.7.1) : centrer, matrice de covariance, direction principale, coordonnées, reconstruction.

</details>
<details><summary>Indice 2</summary>

Centre les cinq points (retire le point moyen) ; la covariance est la somme des produits des écarts divisée par $n - 1 = 4$. Un vecteur propre vérifie $\boldsymbol{\Sigma}\,\mathbf{v} = \lambda\,\mathbf{v}$. La variance totale est la somme de la diagonale, égale à la somme des deux valeurs propres. La coordonnée d'un point centré sur $\mathbf{u}$ est un produit scalaire ; la reconstruction est $\boldsymbol{\mu} + t\,\mathbf{u}$ ; le whitening divise $t$ par $\sqrt{\lambda}$.

</details>
<details><summary>Indice 3</summary>

a) La moyenne des abscisses vaut $(1 + 3 + 4 + 5 + 7)/5 = 4$, celle des ordonnées $(1 + 3 + 3 + 5 + 3)/5 = 3$ : le point moyen est $(4, 3)$. b) Points centrés : A devient $(-3, -2)$ ; calcule les quatre autres, puis $\sum x^2$, $\sum y^2$ et $\sum xy$, divisés par 4. c) Multiplie la matrice de b) par $(2, 1)$, ligne par ligne. d) Compare le produit à $(2, 1)$. e) Additionne la diagonale de b). f) Divise d) par e). g) $t = \frac{2x_c + y_c}{\sqrt{5}}$ pour chaque point centré. h) Somme des carrés des $t$, divisée par 4. i) Pose $\boldsymbol{\mu} + t_D\,\mathbf{u}$. j) Distance entre D et i). k) $t_D / \sqrt{\lambda}$. l) $\frac{-x_c + 2y_c}{\sqrt{5}}$ pour chaque point, puis la variance (ddof = 1). m) Que devient la variance de la première feature, et vers quelle feature la direction principale tourne-t-elle ?

</details>

### Ex 12.8 — Variance d'une projection et axe de variance maximale ∂

<details><summary>Indice 1</summary>

Tout se ramène au produit matriciel : $\sum_i t_i^2 = \mathbf{t}^\top \mathbf{t}$ et $(\mathbf{X}_c\,\mathbf{u})^\top = \mathbf{u}^\top \mathbf{X}_c^\top$. Pour les questions suivantes, développe $\mathbf{u}^\top \boldsymbol{\Sigma}\,\mathbf{u}$ coordonnée par coordonnée, puis utilise $u_1^2 + u_2^2 = 1$.

</details>
<details><summary>Indice 2</summary>

1. La moyenne des $t_i$ est une somme pondérée des moyennes des colonnes de $\mathbf{X}_c$. 2. Remplace $u_2^2$ par $1 - u_1^2$ : la variance devient une fonction de $u_1^2$ seul, qui varie entre 0 et 1. 3. Que devient cette fonction quand $a = b$ ? 4. Même idée avec $u_1^2 + u_2^2 + u_3^2 = 1$ ; pour la deuxième composante, impose $u_1 = 0$. 5. $\lVert \mathbf{v} \rVert^2 = \mathbf{v}^\top \mathbf{v}$ ; ramène-toi à la question 2 avec $\mathbf{v}$. 6. Développe les deux formes quadratiques avec $\mathbf{u} = (c, s)$ et $\mathbf{w} = (-s, c)$.

</details>
<details><summary>Indice 3</summary>

1. La moyenne des $t_i = \sum_j x_{c,ij}\,u_j$ vaut $\sum_j u_j \times (\text{moyenne de la colonne } j \text{ de } \mathbf{X}_c) = 0$, puisque chaque colonne est centrée. Puis la variance vaut $\frac{1}{n - 1}\,\mathbf{t}^\top \mathbf{t} = \frac{1}{n - 1}\,\mathbf{u}^\top \mathbf{X}_c^\top \mathbf{X}_c\,\mathbf{u}$ : reconnais $\boldsymbol{\Sigma}$. 2. Développe $(u_1, u_2)\begin{pmatrix} a & 0 \\ 0 & b \end{pmatrix}\begin{pmatrix} u_1 \\ u_2 \end{pmatrix}$, remplace $u_2^2$, puis cherche les valeurs extrêmes de $b + (a - b)\,u_1^2$ quand $u_1^2$ va de 0 à 1. 3. Pose $a = b$ dans l'expression de 2. 4. Écris $a u_1^2 + b u_2^2 + c u_3^2 \le a(u_1^2 + u_2^2 + u_3^2)$ et cherche le cas d'égalité ; puis recommence avec $u_1 = 0$. 5. $\mathbf{v}^\top \mathbf{v} = \mathbf{u}^\top \mathbf{R}\,\mathbf{R}^\top \mathbf{u}$ (admets que $\mathbf{R}\,\mathbf{R}^\top = \mathbf{I}$ aussi) ; puis $\mathbf{u}^\top \boldsymbol{\Sigma}\,\mathbf{u} = (\mathbf{R}^\top \mathbf{u})^\top \mathrm{diag}(a, b)\,(\mathbf{R}^\top \mathbf{u})$ ; d'après la question 2, pour quel $\mathbf{v}$ le maximum est-il atteint ? Déduis-en $\mathbf{u}$, puis calcule $\boldsymbol{\Sigma}\,\mathbf{u}$. 6. Écris les deux formes avec $\Sigma_{11}$, $\Sigma_{12}$ et $\Sigma_{22}$, additionne, et utilise $c^2 + s^2 = 1$.

</details>

<a id="reflexion"></a>

## 🗣️ ⚖️ Réflexion

### Ex 12.9 — La fuite de données expliquée en 5 lignes 🗣️

<details><summary>Indice 1</summary>

Une bonne image : un examen dont on a vu les questions, ou une recette ajustée en goûtant le plat du jury. Qu'est-ce qui a « vu » le test, ici ?

</details>
<details><summary>Indice 2</summary>

Construis les cinq lignes ainsi : l'image ; ce qui s'est passé dans son notebook (la moyenne, l'écart-type ou la médiane calculés sur toutes les lignes, test compris) ; pourquoi le score de validation est alors trop beau ; la règle d'or ; la parade (`Pipeline`) et pourquoi elle marche.

</details>
<details><summary>Indice 3</summary>

Première ligne possible : « C'est comme réviser un examen avec une partie des questions de l'examen : on a l'air meilleur qu'on ne l'est. » Écris les quatre autres en suivant le plan de l'indice 2, sans jargon non défini. Pour la question sur la figure, demande-toi combien de fois la flèche « fit » doit apparaître dans une validation croisée à 5 folds.

</details>

### Ex 12.10 — Supprimer ou imputer : qui disparaît des données ? ⚖️

<details><summary>Indice 1</summary>

Lis le tableau ligne par ligne : la part des lignes incomplètes est-elle la même pour toutes les espèces et toutes les saisons ?

</details>
<details><summary>Indice 2</summary>

Une proportion par groupe se compare à la proportion globale. Pour 2, une valeur qui manque « complètement au hasard » manquerait à peu près autant dans chaque groupe. Pour 3, une suppression qui frappe un groupe et une saison change ce qu'on mesure sur ce groupe et cette saison. Pour 4 et 5, demande-toi à chaque fois ce que le modèle « voit » de la valeur inventée, et où la statistique est apprise.

</details>
<details><summary>Indice 3</summary>

1. En tout, 20 manchots sur 344 ont une valeur manquante, soit environ 5,8 %. Calcule de même la part chez les Adélie de 2007-08, à partir de leur ligne du tableau, et compare. 2. Les commentaires parlent du sang prélevé et des analyses : ces problèmes touchent-ils toutes les saisons de la même façon ? Situe la situation parmi les trois cas de la fiche (complètement au hasard ; au hasard une fois connues d'autres variables comme la saison ou l'espèce ; en fonction de la valeur elle-même). 3. Si un groupe d'une saison perd un quart de ses membres, l'effectif et peut-être la moyenne de cette saison changent-ils ? 4. Pour chaque option, écris ce qui est conservé, ce qui est inventé, et sur quelles lignes la statistique doit être calculée. 5. Qui serait représenté par la moyenne imputée, et qui serait mal décrit ?

</details>

<a id="entretien"></a>

## 💼 Entretien

### 12.E1 — Qu'est-ce qu'une fuite de données ? Donne deux exemples

<details><summary>Indice 1</summary>

Une définition en une phrase, puis deux exemples de familles différentes : une fuite par le prétraitement (ce chapitre) et une fuite par une feature connue trop tard ou par des groupes (ch. 8).

</details>
<details><summary>Indice 2</summary>

Pour chaque exemple : ce qui fuit, pourquoi le score devient trop beau, et la parade (découper d'abord, `Pipeline`, découpage par groupe ou dans le temps, revue des features).

</details>
<details><summary>Indice 3</summary>

Structure : « Une fuite, c'est quand le modèle, ou les choix qui le construisent, profitent d'une information qu'ils n'auraient pas en usage réel. Exemple 1 : une standardisation ou une imputation calculée sur tout le dataset avant la validation croisée… Exemple 2 : … ». Complète avec une feature calculée après la cible (une date de clôture, un traitement prescrit après le diagnostic), puis termine par la façon de détecter une fuite (un score trop beau, une feature trop importante).

</details>

### 12.E2 — Standardisation ou normalisation min-max : laquelle, et pourquoi ?

<details><summary>Indice 1</summary>

Commence par dire ce que fait chacune, puis quand il faut une mise à l'échelle et quand elle est inutile.

</details>
<details><summary>Indice 2</summary>

Trois critères : ce qu'attend l'algorithme (un intervalle fixe, des features comparables), la nature des données (bornées comme des pixels, ou non), et la présence de points aberrants. N'oublie pas les modèles insensibles à l'échelle et la règle d'or.

</details>
<details><summary>Indice 3</summary>

Plan : (1) min-max vers $[0, 1]$, fondé sur le minimum et le maximum, contre z-score, fondé sur la moyenne et l'écart-type ; (2) qui en a besoin : distances, descente de gradient, pénalités, PCA, mais pas les arbres ; (3) le choix : données bornées ou algorithme qui exige un intervalle → min-max ; par défaut → standardisation ; points aberrants → version robuste ; (4) dans tous les cas, `fit` sur l'entraînement et un `Pipeline`, et le choix se valide comme un hyperparamètre.

</details>

### 12.E3 — Comment traites-tu les valeurs manquantes ?

<details><summary>Indice 1</summary>

Avant de choisir une méthode, comprends **pourquoi** les valeurs manquent, et **qui** est concerné.

</details>
<details><summary>Indice 2</summary>

Trois étapes : diagnostiquer (combien, où, chez qui, pourquoi) ; choisir (supprimer des lignes ou une colonne, imputer simplement avec un indicateur, imputer par un modèle, ou un modèle qui accepte les NaN) ; valider (comparer les options en validation croisée, imputation apprise dans le `Pipeline`).

</details>
<details><summary>Indice 3</summary>

Commence par : « D'abord, je regarde comment elles se répartissent : par groupe, par période, et si l'absence peut dépendre de la valeur elle-même. » Enchaîne sur les options de l'indice 2, avec pour chacune son risque (des exemples perdus et un biais possible pour la suppression, variance écrasée pour la moyenne, fuite si l'imputation est apprise hors du `Pipeline`), et termine par la comparaison en validation croisée.

</details>

### 12.E4 — À quoi sert une PCA et quelles sont ses limites ?

<details><summary>Indice 1</summary>

Une image simple de la PCA, puis trois usages, puis les limites.

</details>
<details><summary>Indice 2</summary>

Image : trouver les directions où les données s'étalent le plus, et ne garder que celles-là. Usages : compresser, visualiser, débruiter, accélérer un modèle. Limites : linéaire, non supervisée, sensible aux unités et aux points aberrants, composantes difficiles à interpréter, à apprendre sur l'entraînement.

</details>
<details><summary>Indice 3</summary>

Commence par : « La PCA cherche les directions où les données varient le plus, orthogonales entre elles, et décrit chaque exemple par ses coordonnées sur les premières. » Puis : comment choisir le nombre de composantes (variance expliquée, ou validation du modèle qui suit) ; un exemple d'usage ; et deux limites illustrées (deux classes séparées dans une petite direction ; une feature en grammes qui accapare la première composante sans standardisation).

</details>

### 12.E5 — Encoder une variable catégorielle à 10 000 modalités

<details><summary>Indice 1</summary>

Un one-hot donnerait 10 000 colonnes presque vides : quelles autres familles de solutions existe-t-il ?

</details>
<details><summary>Indice 2</summary>

Quatre pistes : regrouper les modalités rares ; encoder par une statistique de la cible (avec les précautions contre la fuite) ; un hachage ou un embedding appris ; un modèle qui gère les catégories nativement. Pour chacune : coût, risque, quand l'utiliser.

</details>
<details><summary>Indice 3</summary>

Structure : (1) regarder la distribution des modalités (souvent, quelques centaines couvrent l'essentiel) ; (2) one-hot des modalités fréquentes et une colonne « autres » (`min_frequency`) ; (3) encodage par la cible avec cross fitting (`TargetEncoder`), jamais appris sur les lignes qu'il encode ; (4) au-delà : hachage, embedding appris par un réseau, ou modèle qui accepte les catégories ; (5) et toujours : que faire d'une modalité jamais vue ?

</details>

<a id="notebook"></a>

## Notebook

### Ex 12.11 — Diagnostic de Penguins brut avec pandas 📦

<details><summary>Indice 1</summary>

Chaque question tient en une ligne de pandas : compter des valeurs distinctes (colonne par colonne, ou identifiant par identifiant), ou compter les lignes qui vérifient deux conditions sur les valeurs manquantes.

</details>
<details><summary>Indice 2</summary>

`RAW.nunique(dropna=False)` donne, pour chaque colonne, son nombre de valeurs distinctes, `NaN` compté comme une valeur. Pour c) et d), regroupe les lignes par identifiant (`RAW.groupby("Individual ID")`), compte dans chaque groupe les saisons ou les espèces distinctes (`nunique()` d'une colonne), puis compte les groupes où ce nombre dépasse 1. Pour e), combine deux masques avec `&`.

</details>
<details><summary>Indice 3</summary>

a) `counts = RAW.nunique(dropna=False)`, puis `sorted(counts[counts == 1].index)`. b) le nombre de valeurs distinctes d'une seule colonne. c) un `groupby` sur l'identifiant, puis un `nunique` des saisons par groupe, comparé à 1, et le nombre de groupes qui dépassent 1. d) le même raisonnement, avec l'espèce. e) deux masques sur les deux colonnes d'isotopes, combinés par `&`, puis leur somme.

</details>

### Ex 12.12 — CSV à la française : virgules décimales et « 7e-3 » 🐛

<details><summary>Indice 1</summary>

Compare quelques cases du fichier à ce qu'en fait le chargeur du collègue : « 39,1 » devient 391, mais « 50 » reste 50. Quel caractère son code supprime-t-il, et que voulait dire ce caractère dans un fichier français ?

</details>
<details><summary>Indice 2</summary>

`pd.read_csv` connaît les conventions françaises : `sep=";"` pour le séparateur de colonnes, `decimal=","` pour la virgule décimale (la notation scientifique « 3,75e-3 » est lue avec), et `na_values=[...]` pour les marques de valeur absente qu'il ne connaît pas seul. Reste les unités : combien de grammes dans une tonne ?

</details>
<details><summary>Indice 3</summary>

```python
def load_12(text):
    df = pd.read_csv(io.StringIO(text), sep=";", decimal=",", na_values=[...])   # the two markers of the file
    df["masse_g"] = df["masse_t"] * ...                                         # grams in one tonne
    return df[[...]]                                                            # the 4 columns, in this order
```

</details>

### Ex 12.13 — Où tombent les données de test après un MinMaxScaler ? 🔮

<details><summary>Indice 1</summary>

Une valeur de test ne sort de $[0, 1]$ que si elle est plus petite que le minimum d'entraînement ou plus grande que son maximum. Combien de districts de test peuvent battre le plus extrême de 16 500 districts d'entraînement ?

</details>
<details><summary>Indice 2</summary>

Pour b), qu'est-ce qui empêcherait une feature de sortir de $[0, 1]$ sur 200 découpages ? Regarde, dans la data card, si certaines valeurs extrêmes sont partagées par beaucoup de districts. Pour c), la normalisation divise par l'étendue d'entraînement : qu'est-ce qui fixe celle d'`AveOccup` (data card) ?

</details>
<details><summary>Indice 3</summary>

Pour c), situe un ménage ordinaire (environ 3 personnes) entre le minimum et le maximum d'`AveOccup` que donne la data card : est-il au milieu, au premier dixième, ou tout au bord ? Pour a), une case du test ne sort que si elle bat les extrêmes de l'entraînement : sur 33 000 cases, combien peuvent le faire, au plus, pour chacune des 8 features ?

</details>

### Ex 12.14 — pandas pour préparer : dates (« Date Egg »), jointure merge avec une table des îles, pivot_table 📦

<details><summary>Indice 1</summary>

Une colonne de texte devient une colonne de dates avec `pd.to_datetime` ; l'accesseur `.dt` en tire le mois, et la différence de deux dates est une durée, dont `.dt.days` donne le nombre de jours. Pour c) et d), compte les lignes du résultat de `merge`.

</details>
<details><summary>Indice 2</summary>

b) Regroupe les dates par saison (`dates_14.groupby(RAW["studyName"])`), puis `max() - min()`. c) Une clé qui se répète dans la table de droite associe chaque ligne de gauche à **toutes** ses correspondances. d) `merge(..., on=["Island", "studyName"], how="left")`, puis compte les `NaN` de `team`. e) `RAW.pivot_table(index="Species", columns="Island", values="Body Mass (g)", aggfunc="mean")`, puis compte ses `NaN`.

</details>
<details><summary>Indice 3</summary>

a) `dates_14 = pd.to_datetime(RAW["Date Egg"])` et `n_december_14 = int((dates_14.dt.month == 12).sum())`. b) un `groupby` des dates par saison, la différence du maximum et du minimum, puis `.dt.days`, en liste d'entiers. c) le nombre de lignes de la jointure sur l'île seule. d) une jointure à gauche sur les deux clés, puis les `NaN` de la colonne `team`. e) le nombre total de `NaN` du tableau croisé.

</details>

### Ex 12.15 — Lire des données dans une base SQL : sqlite3, SELECT, JOIN, GROUP BY et pd.read_sql 📦

<details><summary>Indice 1</summary>

Écris chaque requête dans l'ordre logique de l'encadré 🧮 : `FROM` (et `JOIN`), `WHERE`, `GROUP BY`, `SELECT`, `ORDER BY`. Nomme les colonnes du résultat avec `AS`.

</details>
<details><summary>Indice 2</summary>

a) Une valeur absente se teste avec `IS NULL`, jamais avec `= NULL`. b) Deux conditions de jointure, reliées par `AND` : `JOIN teams AS t ON p.island = t.island AND p.study = t.study`. c) `WHERE sex = 'FEMALE'`, l'agrégat `AVG(body_mass_g)` et un tri décroissant (`DESC`).

</details>
<details><summary>Indice 3</summary>

a), en entier :

```sql
SELECT island, COUNT(*) AS n
FROM penguins
WHERE sex IS NULL
GROUP BY island
ORDER BY island
```

b) et c) ont la même structure : pour b), quelle table dans `FROM`, quelle condition de `JOIN`, quelle colonne dans `GROUP BY` et `ORDER BY` ; pour c), quel filtre, quel agrégat renommé `mean_mass`, quel tri.

</details>

### Ex 12.16 — Coder StandardScaler (fit, transform, inverse) 🔨

<details><summary>Indice 1</summary>

`fit` convertit `X` en tableau de flottants (`np.asarray(X, dtype=float)`), vérifie sa dimension (`X.ndim`) et ses valeurs (`np.isfinite(X).all()`), puis calcule ses statistiques colonne par colonne (`axis=0`). `transform` et `inverse_transform` ne recalculent rien : ils lisent les attributs appris. Pour savoir si `fit` a eu lieu, teste un attribut appris (`hasattr(self, "mean_")`).

</details>
<details><summary>Indice 2</summary>

`mean_` : la moyenne de chaque colonne ; `var_` : sa variance (ddof = 0), mise à 0 pour une colonne constante, repérée par `X.max(axis=0) == X.min(axis=0)` ; `scale_` : la racine de `var_`, remplacée par 1 là où la colonne est constante, et 1 partout si `with_std=False`. `transform` retire `mean_` (si `with_mean`) puis divise par `scale_` ; `inverse_transform` fait l'inverse, dans l'ordre inverse. Une méthode privée qui vérifie un tableau (2-D, valeurs finies, bon nombre de colonnes) évite d'écrire trois fois les mêmes tests.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y=None):
    X = self._check(X)                                   # 2-D, finite, float64: ValueError otherwise
    constant = X.max(axis=0) == X.min(axis=0)
    self.var_ = np.where(constant, 0.0, X.var(axis=0))
    ...                                                  # mean_, scale_, n_features_in_, n_samples_seen_
    return self

def transform(self, X):
    if not hasattr(self, "mean_"):
        raise RuntimeError("call fit first")
    X = self._check(X, n_features=self.n_features_in_)
    ...
```

</details>

### Ex 12.17 — Coder MinMaxScaler avec feature_range et clip 🔨

<details><summary>Indice 1</summary>

La structure de `StandardScaler` convient : `fit` apprend des statistiques par colonne, `transform` les lit. Les deux formules de la docstring disent tout : `scale_ = (b - a) / data_range_` et `min_ = a - data_min_ * scale_`, avec `(a, b) = feature_range`.

</details>
<details><summary>Indice 2</summary>

`a, b = self.feature_range`, et `ValueError` si `a >= b`. `data_range_ = data_max_ - data_min_` reste tel quel ; seul le calcul de `scale_` remplace une étendue nulle par 1 (`np.where(data_range_ == 0, 1.0, data_range_)`). `transform` : `X * scale_ + min_`, puis `np.clip(..., a, b)` si `clip`. `inverse_transform` : `(Z - min_) / scale_`, jamais borné. Pour refuser les valeurs infinies sans refuser les `NaN` dans `transform`, teste `np.isinf(X).any()`.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y=None):
    a, b = self.feature_range
    if a >= b:
        raise ValueError("feature_range must be (a, b) with a < b")
    X = self._check(X)
    self.data_min_, self.data_max_ = X.min(axis=0), X.max(axis=0)
    self.data_range_ = self.data_max_ - self.data_min_
    ...                                                  # scale_ (range 0 counted as 1), min_, n_features_in_
    return self
```

</details>

### Ex 12.18 — Coder SimpleImputer (moyenne, médiane, mode, constante) 🔨

<details><summary>Indice 1</summary>

Chaque statistique se calcule sur les valeurs **observées** de sa colonne : `column[~np.isnan(column)]`, ou les fonctions de NumPy qui ignorent les `NaN` (`np.nanmean`, `np.nanmedian`).

</details>
<details><summary>Indice 2</summary>

`"mean"` et `"median"` : `np.nanmean(X, axis=0)` et `np.nanmedian(X, axis=0)` (la médiane d'un nombre pair de valeurs est déjà la moyenne des deux du milieu). `"most_frequent"` : pour chaque colonne, `values, counts = np.unique(observed, return_counts=True)`, puis `values[np.argmax(counts)]` : `np.unique` trie les valeurs et `np.argmax` rend le premier maximum, donc la plus petite valeur parmi les ex aequo. `"constant"` : `fill_value`, ou 0.0 s'il vaut `None`. Une colonne sans aucune valeur observée lève une `ValueError`, sauf avec `"constant"`. `transform` : `np.where(np.isnan(X), self.statistics_, X)` rend un nouveau tableau, sans toucher à `X`.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y=None):
    if self.strategy not in ("mean", "median", "most_frequent", "constant"):
        raise ValueError(f"unknown strategy: {self.strategy!r}")
    X = self._check(X)                                   # 2-D float64, NaN allowed
    if self.strategy == "constant":
        ...                                              # one fill_value per column (0.0 if None)
    else:
        observed = [column[~np.isnan(column)] for column in X.T]
        ...                                              # ValueError if one is empty; then one statistic each
    self.n_features_in_ = X.shape[1]
    return self
```

</details>

### Ex 12.19 — Scalers de scikit-learn face aux points aberrants de California 📦

<details><summary>Indice 1</summary>

La règle de Tukey (fiche §12.4) : les quartiles de `np.percentile`, l'écart interquartile, puis l'intervalle $[Q_1 - 1{,}5\,\mathrm{IQR} ;\ Q_3 + 1{,}5\,\mathrm{IQR}]$, calculés sur l'entraînement. Pour b) et c), chaque transformateur s'ajuste sur `X_train_19` et s'applique à `X_test_19`.

</details>
<details><summary>Indice 2</summary>

Pour un tableau transformé `T` à deux colonnes, `np.percentile(T, [25, 75], axis=0)` donne les deux quartiles de chaque colonne : le rapport cherché est l'écart interquartile de la colonne 0 divisé par celui de la colonne 1. Une boucle sur `SCALERS_19.items()` donne les quatre rapports dans l'ordre A, B, C, D. Pour d), combine b) (des rapports proches de 1 ?) et c) (des valeurs bornées ?).

</details>
<details><summary>Indice 3</summary>

a) `q1, q3 = np.percentile(X_train_19[:, 1], [25, 75])`, `iqr = q3 - q1`, puis compte les valeurs `x` de la colonne telles que `(x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)`. b) Pour A : `T = SkMinMaxScaler().fit(X_train_19).transform(X_test_19)`, puis le rapport des deux écarts interquartiles ; même calcul pour B, C et D. c) la plus grande valeur de la colonne 1 de `T` pour C. d) le transformateur dont le rapport est proche de 1 **et** dont les sorties restent dans des bornes fixes.

</details>

### Ex 12.20 — Le scaler réentraîné sur le test et les get_dummies désalignés 🐛

<details><summary>Indice 1</summary>

Relis la cellule qui entraîne `MODEL_20`. Pour chaque ligne de `predict_buggy_20`, demande-toi : ce paramètre (une moyenne, un écart-type, une liste de colonnes) vient-il de l'entraînement, ou du batch reçu ?

</details>
<details><summary>Indice 2</summary>

Deux choses sont réapprises sur le batch : la moyenne et l'écart-type du scaler, et la liste des colonnes d'îles que crée `get_dummies`. Celles de l'entraînement sont déjà rangées dans `SCALER_20` et `COLUMNS_20`. Un DataFrame se réaligne sur une liste de colonnes avec `reindex(columns=..., fill_value=0.0)`.

</details>
<details><summary>Indice 3</summary>

```python
def prepare_20(batch):
    X = SCALER_20.transform(batch[MEASURES])                 # the parameters learnt on P_TRAIN
    islands = pd.get_dummies(batch["island"], dtype=float)   # then: exactly the columns of COLUMNS_20, in order
    ...
    return np.hstack([...])
```

</details>

### Ex 12.21 — Trafic et température : transformer la cible et revenir aux voitures 📦

<details><summary>Indice 1</summary>

Un transformateur par grandeur (fiche §12.8) : le premier appris sur les températures, le second sur les voitures ; le modèle apprend et prédit dans les unités transformées. Un `MinMaxScaler` attend un tableau 2-D : `reshape(-1, 1)`.

</details>
<details><summary>Indice 2</summary>

Pour revenir aux voitures : `inverse_transform` du transformateur des **voitures**, sur la prédiction remise en colonne, puis `ravel()`. `TransformedTargetRegressor(regressor=..., transformer=...)` reçoit le pipeline de la feature et le transformateur de la cible. Pour e), entraîne `make_pipeline(SkMinMaxScaler(), Ridge(alpha=1.0))` directement sur les voitures et compare ses prédictions aux tiennes avec `np.allclose`.

</details>
<details><summary>Indice 3</summary>

```python
def fit_predict_21(T_train, cars_train, T_new):
    temp_scaler = SkMinMaxScaler().fit(np.reshape(T_train, (-1, 1)))
    cars_scaler = SkMinMaxScaler().fit(np.reshape(cars_train, (-1, 1)))
    model = Ridge(alpha=1.0).fit(...)                  # scaled temperatures, scaled cars (1-D)
    z_hat = model.predict(...)                          # in the scaled units of the target
    return ...                                          # back to cars: which transformer?
```

c) la prédiction pour −2 °C passée dans `transform` du transformateur des voitures (ou `z_hat` lui-même) ; d) compare les trois prédictions à `CARS_21.min()` et à `CARS_21.max()`.

</details>

### Ex 12.22 — Projeter un nuage 2D : axe horizontal contre axe de variance maximale 🎨

<details><summary>Indice 1</summary>

La coordonnée d'un point sur une droite est son produit scalaire avec un vecteur **unitaire** de la droite (encadré 🧮 du §12.7.1). La variance totale est la somme des variances des deux colonnes. L'axe principal est le vecteur propre de la plus grande valeur propre de la matrice de covariance.

</details>
<details><summary>Indice 2</summary>

Divise `u` par `np.linalg.norm(u)` avant le produit scalaire. `np.linalg.eigh(np.cov(cloud_s_22, rowvar=False))` rend les valeurs propres par ordre **croissant** et les vecteurs propres en **colonnes**. L'angle d'un vecteur : `np.degrees(np.arctan2(u[1], u[0]))`, ramené entre 0 et 180 avec `% 180`. Pour la figure, les points projetés sont `t[:, None] * u_unitaire`, où `t` sont les coordonnées.

</details>
<details><summary>Indice 3</summary>

`project_22` : normalise `u` (`np.linalg.norm`), puis un produit matriciel. a) la variance des coordonnées sur `[1, 0]`, divisée par `np.var(cloud_s_22, axis=0).sum()` ; b) le même calcul avec la dernière colonne de `eigh`, au signe fixé par sa première coordonnée ; c) `np.arctan2`, en degrés, `% 180`.

</details>

### Ex 12.23 — Lire une courbe de variance expliquée cumulée 📈

<details><summary>Indice 1</summary>

Le panneau (a) donne la part de **chaque** composante, le panneau (b) la part **cumulée** : choisis le bon panneau pour chaque question.

</details>
<details><summary>Indice 2</summary>

Pour b), suis la courbe des 1 sur le panneau (b) et repère le premier point au-dessus de la ligne horizontale de 0,6. Pour c), lis la courbe des 8 à $k = 20$ ; pour d), la fin de la courbe des 1, à $k = 40$, par rapport à la ligne de 0,9.

</details>
<details><summary>Indice 3</summary>

a) Sur le panneau (a), le premier point des 1 est un peu au-dessus de 0,35 : `"B"`. b) Sur le panneau (b), quel est le premier point de la courbe des 1 au-dessus de 0,6 ? c) Quelle valeur lis-tu sur la courbe des 8 à l'abscisse 20, et de quel choix est-elle la plus proche ? d) Le dernier point de la courbe des 1 est-il au-dessus ou en dessous de 0,9 ?

</details>

### Ex 12.24 — Combien de composantes pour 90 % de la variance de MNIST ? 🔮

<details><summary>Indice 1</summary>

12.23 a mesuré les courbes des 1 et des 8 **seuls**. Un mélange des dix chiffres est-il plus ou moins varié qu'un seul chiffre ? Il faut donc plus ou moins de composantes ?

</details>
<details><summary>Indice 2</summary>

De 90 % à 99 %, on ajoute des détails : une courbe cumulée monte de plus en plus lentement. Pense aussi aux pixels du bord, presque toujours noirs : combien de directions portent vraiment de la variance, sur 784 ?

</details>
<details><summary>Indice 3</summary>

Pour c), la standardisation donne une variance 1 à **chaque** pixel. Que devient, dans la variance totale, la part d'un pixel du bord dont la variance était minuscule ? Combien de pixels sont dans ce cas, sur 784 ?

</details>

### Ex 12.25 — Sélection de features : colonnes constantes, VarianceThreshold, SelectKBest 📦

<details><summary>Indice 1</summary>

`VarianceThreshold` et `SelectKBest` sont des transformateurs : on les ajuste avec `fit`, puis `get_support()` rend un masque de booléens, `True` pour chaque colonne gardée.

</details>
<details><summary>Indice 2</summary>

a) les noms `SEL_25.columns[~masque]` ; b) le nombre de `False` du masque ; c) une masse en kilogrammes vaut la masse en grammes divisée par 1 000 : sa variance est divisée par $1\,000^2$ ; d) `SelectKBest(f_classif, k=2).fit(SEL_25[NUMERIC_25], Y_25)` ; e) `cross_val_score(make_pipeline(...), SEL_25[NUMERIC_25], Y_25, cv=CV_25).mean()`, avec `from sklearn.pipeline import make_pipeline`.

</details>
<details><summary>Indice 3</summary>

a) `constant_25 = list(SEL_25.columns[~VarianceThreshold().fit(SEL_25).get_support()])`. b) le même masque avec `threshold=1.0`, et le nombre de `False`. c) la variance de la masse en kilogrammes, `np.var(SEL_25["Body Mass (g)"] / 1000)`, comparée au seuil. d) le masque de `SelectKBest` appliqué à `NUMERIC_25`. e) deux pipelines, l'un sans sélection, l'autre avec `SelectKBest` entre le scaler et la régression, chacun dans `cross_val_score` avec `CV_25`.

</details>

### Ex 12.26 — Coder OrdinalEncoder et OneHotEncoder 🔨

<details><summary>Indice 1</summary>

Les deux encodeurs ont le même `fit` : pour chaque colonne $j$, la liste de ses catégories. Avec `"auto"`, `np.unique(X[:, j])` les rend triées ; avec des listes données, vérifie que chaque valeur de la colonne est dans la sienne. Travaille sur `X = np.asarray(X, dtype=object)`, et range les catégories dans des tableaux NumPy.

</details>
<details><summary>Indice 2</summary>

`OrdinalEncoder.transform` : un dictionnaire `{catégorie: position}` par colonne ; une valeur absente du dictionnaire lève une `ValueError`. `OneHotEncoder.transform` : pour la colonne $j$, `(X[:, j][:, None] == self.categories_[j][None, :]).astype(float)` donne le bloc d'un coup ; une ligne sans aucun 1 est une catégorie inconnue (erreur, ou zéros avec `"ignore"`) ; retire la colonne `drop_idx_[j]` si elle n'est pas `None`, puis concatène les blocs (`np.hstack`). `inverse_transform` : découpe `X` en blocs, `argmax` de chaque bloc, et traite à part les blocs nuls. `drop_idx_` se construit avec `np.array([...], dtype=object)`.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y=None):                              # OneHotEncoder
    if self.drop not in (None, "first", "if_binary") or self.handle_unknown not in ("error", "ignore"):
        raise ValueError("invalid option")
    X = self._check(X)                                 # 2-D object array: ValueError otherwise
    self.categories_ = self._learn_categories(X)       # the same helper as OrdinalEncoder
    if self.drop is None:
        self.drop_idx_ = None
    else:
        self.drop_idx_ = np.array([...], dtype=object)   # 0, or None for a column where nothing is dropped
    ...
    return self
```

</details>

### Ex 12.27 — Coder la PCA (SVD, variance expliquée, whitening, reconstruction) 🔨

<details><summary>Indice 1</summary>

Suis le pseudo-code de la fiche (§12.7.1) : centrer, `np.linalg.svd(Xc, full_matrices=False)`, les lignes de `Vt` sont les composantes et `S**2 / (n - 1)` leurs variances. Tout le reste (signe, parts, nombre de composantes, whitening) se greffe sur ces trois lignes.

</details>
<details><summary>Indice 2</summary>

Le signe : pour chaque ligne de `Vt`, `idx = np.argmax(np.abs(row))`, puis multiplie la ligne par `np.sign(row[idx])`. Les parts : les variances divisées par leur somme, calculées sur **toutes** les composantes avant de couper. Le nombre de composantes : pour un réel, `np.searchsorted(np.cumsum(ratios), seuil, side="right") + 1` donne le plus petit $k$ dont la part cumulée dépasse strictement le seuil ; pour un entier, vérifie qu'il est entre 1 et $\min(n, p)$. Whitening : `transform` divise les coordonnées par `np.sqrt(self.explained_variance_)`, et `inverse_transform` commence par les multiplier.

</details>
<details><summary>Indice 3</summary>

```python
def fit(self, X, y=None):
    X = self._check(X)
    self.mean_ = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
    signs = np.sign(Vt[np.arange(len(Vt)), np.argmax(np.abs(Vt), axis=1)])
    Vt = Vt * signs[:, None]
    variances = S**2 / (len(X) - 1)
    ...                                                # ratios over ALL components, k, then keep the first k
    return self
```

</details>

### Ex 12.28 — Chiffres propres : PCA sur MNIST et reconstructions 📦

<details><summary>Indice 1</summary>

Une reconstruction s'obtient par `pca.inverse_transform(pca.transform(images))`. L'erreur demandée est la moyenne du carré de l'écart sur toutes les images et tous les pixels : un seul `np.mean` sur un tableau 2-D.

</details>
<details><summary>Indice 2</summary>

Pour chaque $k$, ajuste **une** PCA sur `TRAIN_28`, puis calcule l'erreur sur `TRAIN_28` et sur `TEST_28` avec cette même PCA. `baseline_28` : l'image moyenne de l'entraînement, `TRAIN_28.mean(axis=0)`, retirée de chaque image de test, puis la moyenne des carrés.

</details>
<details><summary>Indice 3</summary>

```python
def errors_28(k):
    pca = SkPCA(n_components=k, svd_solver="full").fit(TRAIN_28)
    train_error = float(np.mean((pca.inverse_transform(pca.transform(TRAIN_28)) - TRAIN_28) ** 2))
    ...                                                # the same with TEST_28, then return the pair
```

</details>

### Ex 12.29 — Échelle des features et descente de gradient 🔬

<details><summary>Indice 1</summary>

Une descente de gradient (ch. 5) part d'un point et répète : le gradient au point courant, puis un petit pas en sens inverse. Garde la MSE avant le premier pas, puis après chacun. Pour a) et b), essaie les learning rates de `LRS_29` du plus grand au plus petit, et arrête-toi au premier qui ne diverge pas.

</details>
<details><summary>Indice 2</summary>

À chaque pas : `grad_w, grad_b = mse_grad_29(X, y, w, b)`, puis `w, b = w - lr * grad_w, b - lr * grad_b` (les deux mises à jour à partir du même point). Une descente qui diverge produit `inf`, puis `nan` : teste `np.isfinite(losses[-1]) and losses[-1] < losses[0]` (et `np.errstate` fait taire les avertissements). Pour c), le premier indice `t` tel que `losses[t] <= 1.001 * BEST_29`, sur au plus 20 000 pas. `X_std_29` : `SkStandardScaler().fit_transform(X_29)`.

</details>
<details><summary>Indice 3</summary>

```python
def gd_29(X, y, lr, n_steps):
    w, b = np.zeros(X.shape[1]), 0.0
    losses = [float(np.mean((X @ w + b - y) ** 2))]
    for _ in range(n_steps):
        ...                                            # the two gradients, then the two updates
        losses.append(...)
    return losses
```

d) Compare la dernière MSE de chaque descente de 20 000 pas avec `np.isclose(..., rtol=1e-4)`.

</details>

### Ex 12.30 — PCA, t-SNE ou UMAP pour voir MNIST en 2D 📦

<details><summary>Indice 1</summary>

Les trois méthodes ont un `fit_transform` qui rend directement la carte en 2D.

</details>
<details><summary>Indice 2</summary>

PCA : `SkPCA(n_components=2)`. t-SNE : d'abord `SkPCA(n_components=50).fit_transform(X)`, puis `TSNE(n_components=2, perplexity=30, init="pca", random_state=0)` sur ce résultat. UMAP : `umap.UMAP(n_neighbors=15, random_state=0)`. Une méthode inconnue lève une `ValueError`.

</details>
<details><summary>Indice 3</summary>

```python
def embed_30(method, X):
    if method == "PCA":
        return SkPCA(n_components=2).fit_transform(X)
    if method == "t-SNE":
        ...                                            # 50 PCA components first, then TSNE
    ...
```

</details>

### Ex 12.31 — Fuite de données par le prétraitement : scaler, imputation et TargetEncoder ajustés avant le split 🔬

<details><summary>Indice 1</summary>

Une variante se décrit par ce qui est ajusté **avant** `cv_auc_31`, sur toutes les lignes, et ce qui l'est **dedans**, dans le pipeline, refait à chaque fold. Ce qui est déjà préparé traverse le pipeline avec `"passthrough"`.

</details>
<details><summary>Indice 2</summary>

B : `numeric_31().fit_transform(DATA_31[NUM_31])`, rangé dans un DataFrame aux mêmes noms de colonnes, plus la colonne `code` ; dans le pipeline, `make_column_transformer(("passthrough", NUM_31), (encoder_31(), ["code"]))`. C : une colonne `code_te` égale à `encoder_31().fit(codes, Y_31).transform(codes).ravel()`, avec `codes = DATA_31[["code"]]`, et dans le pipeline `(numeric_31(), NUM_31), ("passthrough", ["code_te"])`. D : comme C, avec `fit_transform(codes, Y_31)`. Puis b), c) et d) se lisent sur les quatre AUC.

</details>
<details><summary>Indice 3</summary>

```python
def auc_31(variant):
    if variant == "A":
        prep = make_column_transformer((numeric_31(), NUM_31), (encoder_31(), ["code"]))
        return cv_auc_31(make_pipeline(prep, LogisticRegression()), DATA_31)
    if variant == "B":
        X = pd.DataFrame(numeric_31().fit_transform(DATA_31[NUM_31]), columns=NUM_31)
        X["code"] = DATA_31["code"].to_numpy()
        ...                                            # passthrough for NUM_31, the encoder for "code"
    ...                                                # C and D: the encoded column first, then the pipeline
```

</details>

### Ex 12.32 — Une fonction de préparation documentée et son test anti-fuite 🛠️

<details><summary>Indice 1</summary>

Commence par la fonction : trois transformateurs ajustés sur `train` seul, puis une fonction intérieure qui prépare un DataFrame avec eux, appelée sur `train` et sur `other`. `fillna` rend une copie : il ne modifie pas le DataFrame reçu.

</details>
<details><summary>Indice 2</summary>

Pour les tests, pense en **propriétés**, chacune sur un petit DataFrame dont tu connais la réponse : que doit-il se passer si `other` change ? si l'on prépare une ligne de `other` seule ? si `other` n'a pas toutes les îles ? si une mesure manque, dans un exemple où la médiane et la moyenne diffèrent ? que deviennent les DataFrames reçus après l'appel ? Une aide `small_32(rows)` qui construit un DataFrame à partir de tuples rend les tests courts.

</details>
<details><summary>Indice 3</summary>

```python
def small_32(rows):
    return pd.DataFrame(rows, columns=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
                                       "island", "sex"])


def test_train_part_does_not_depend_on_other():
    train = small_32([...])                            # three ordinary penguins
    other, huge = small_32([...]), small_32([...])     # one ordinary penguin, one with absurd measures
    assert np.allclose(prepare_32(train, other)[0], prepare_32(train, huge)[0])
```

Les autres tests suivent le même modèle, une propriété chacun : une ligne de `other` seule ou dans un batch, un `other` d'une seule île, une mesure manquante, des DataFrames intacts (`pd.testing.assert_frame_equal` sur une copie faite avant l'appel).

</details>

### Ex 12.33 — Défi : Penguins brut prêt pour l'entraînement, sans fuite 🏆

<details><summary>Indice 1</summary>

Le modèle ne trace qu'une frontière **linéaire** dans l'espace de tes features. Une masse de 4 500 g : est-ce un mâle ? Ça dépend de l'espèce. Comment donner au modèle l'information « lourd pour son espèce » ?

</details>
<details><summary>Indice 2</summary>

Calcule sur `raw_train`, espèce par espèce (`groupby("Species")`), la médiane, la moyenne et l'écart-type de chaque mesure ; prépare chaque ligne avec les statistiques de **son** espèce. Ajoute l'espèce en one-hot, puis essaie les isotopes, en imputant leurs valeurs manquantes. Vérifie les règles et le score public avec `grade_33` à chaque essai.

</details>
<details><summary>Indice 3</summary>

```python
def prepare_33(raw_train, raw_new):
    columns = [...]                                    # the measures (and the isotopes?)
    stats = {}
    for name, group in raw_train.groupby("Species"):
        median = group[columns].median()
        filled = group[columns].fillna(median)
        stats[name] = (median, filled.mean(), filled.std(ddof=0))
    ...                                                # prepare(df): each row with the stats of ITS species,
                                                       # then the species in one-hot; return prepare(raw_train), prepare(raw_new)
```

</details>
