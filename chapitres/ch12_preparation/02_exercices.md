# 12 · Préparation des données — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch12_preparation/06_mes_reponses.md` (créée par `python tools/start_chapter.py 12`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ 12.1, 12.2, 12.3, 12.4, 12.5 et 12.7 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les preuves ∂ 12.6 et 12.8, la réflexion (🗣️ ⚖️) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.25` ou `"0,25"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), des lettres collées pour plusieurs choix (`"AE"`), une suite de codes, une lettre par élément et dans l'ordre (`"XYZ"`), éventuellement séparées par des virgules (`"X, Y, Z"`), une liste pour plusieurs nombres (`[2, 5]`), une liste de listes pour une matrice (`[[1, 0], [0, 1]]`).

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ Papier-crayon](#papier) · [🗣️ ⚖️ Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 ou 4 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 12.Q1 — La règle d'or de la préparation 🧠 ⏱️ 3 min
*Fiche §12.1, §12.2 · livre §12.1, §12.2 · parcours R*

a) On standardise une feature avant d'entraîner un modèle, qu'on évaluera ensuite sur un jeu de test. Sur quelles données calcule-t-on la moyenne et l'écart-type ? (A) sur tout le dataset, avant le découpage ; (B) sur l'entraînement seulement ; (C) sur le test seulement ; (D) sur l'entraînement pour transformer l'entraînement, et sur le test pour transformer le test.
b) Le modèle est en production ; une nouvelle mesure arrive. Que fait-on avant de la lui donner ? (A) un `fit` du scaler sur cette seule mesure ; (B) un `transform` avec les paramètres appris à l'entraînement ; (C) rien : le modèle sait lire les données brutes ; (D) on attend cent mesures pour refaire un `fit` sur elles.
c) Vrai ou faux : un système de reconnaissance vocale est entraîné sur des enregistrements dont on a coupé les silences du début et de la fin ; en service, il reçoit les enregistrements bruts. La règle d'or est respectée.
d) Vrai ou faux : une colonne retirée des données d'entraînement doit aussi être retirée des données de test et de production.

### 12.Q2 — Numérique, ordinale ou nominale ? 🧠 ⏱️ 3 min
*Fiche §12.3 · livre §12.3 · parcours R*

a) Classe chaque feature : Q (numérique, ou quantitative), O (ordinale) ou N (nominale). Six lettres, dans l'ordre.
1. la masse d'un manchot, en grammes ;
2. un avis noté « insatisfait », « neutre » ou « satisfait » ;
3. l'île d'origine d'un manchot ;
4. le code postal d'une commune ;
5. la taille d'un t-shirt : S, M, L ou XL ;
6. le nombre d'enfants d'un foyer.

b) `OrdinalEncoder()` de scikit-learn, avec ses réglages par défaut, reçoit des tailles S, M, L et XL. Dans quel ordre range-t-il les catégories (codes 0, 1, 2 et 3) ? (A) S, M, L, XL ; (B) L, M, S, XL ; (C) XL, L, M, S ; (D) dans l'ordre où elles apparaissent dans les données.
c) Vrai ou faux : coder les îles Biscoe, Dream et Torgersen par 0, 1 et 2 en fait une donnée ordinale.

### 12.Q3 — Pourquoi un one-hot plutôt qu'un entier ? 🧠 ⏱️ 3 min
*Fiche §12.3, §12.3.1 · livre §12.3.1 · parcours R*

Une régression linéaire prédit la durée d'un trajet. L'une de ses features, le mode de transport, prend trois valeurs : bus, train et tram.

a) On code bus = 0, train = 1 et tram = 2 dans une seule colonne, de poids $w$. De combien la prédiction change-t-elle de bus à train, puis de train à tram ? (A) de $w$ dans les deux cas ; (B) de $w$, puis de $2w$ ; (C) de 0 dans les deux cas ; (D) d'une quantité propre à chaque catégorie.
b) Combien de colonnes donne un one-hot de cette feature, sans `drop` ?
c) Et avec `drop="first"` ?
d) Les catégories étant rangées comme le fait `OneHotEncoder` (ordre alphabétique), le vecteur one-hot de « tram », sans `drop`.
e) Vrai ou faux : avec le codage 0, 1, 2, la distance entre bus et tram est le double de la distance entre bus et train.

### 12.Q4 — Doublons, NaN et points aberrants 🧠 ⏱️ 3 min
*Fiche §12.4 · livre §12.4 à §12.4.2 · parcours R*

a) Une colonne de masses de manchots Adélie, une espèce qui pèse de 3 à 5 kg, contient 3 800, 4 150, 38 000 et 3 950 grammes. La valeur 38 000 est très probablement : (A) un manchot exceptionnel, à garder tel quel ; (B) une erreur de saisie, à vérifier puis à corriger ; (C) une valeur manquante ; (D) un doublon.
b) Vrai ou faux : une ligne présente deux fois dans l'entraînement compte deux fois dans une moyenne ou dans une loss.
c) Vrai ou faux : après un `pd.read_csv` sans option, `df.isna()` repère les « ? » qu'un fichier utilise pour noter les valeurs absentes.
d) Que donne `float("0,007")` en Python ? (A) `0.007` ; (B) `7.0` ; (C) une `ValueError` ; (D) `0.0`.
e) Vrai ou faux : `df[df["sex"] == np.nan]` sélectionne les lignes où le sexe manque.

### 12.Q5 — Normaliser ou standardiser ? 🧠 ⏱️ 3 min
*Fiche §12.5 à §12.5.2 · livre §12.5 à §12.5.2 · parcours R, M*

a) Une feature vaut 1, 2, 3, 4 et 1 000 sur l'entraînement. Après une normalisation min-max vers $[0, 1]$, où tombent les quatre premières valeurs ? (A) régulièrement réparties entre 0 et 1 ; (B) toutes entre 0 et 0,01 ; (C) toutes à 0,5 ; (D) entre −1 et 1.
b) Vrai ou faux : après une standardisation, les valeurs de l'entraînement sont toujours comprises entre −3 et 3.
c) Vrai ou faux : la standardisation rend la distribution d'une feature normale (gaussienne).
d) Vrai ou faux : une feature très asymétrique, comme des revenus, reste aussi asymétrique après une normalisation min-max ou une standardisation.

### 12.Q6 — Données de test hors de [0, 1] : bug ou normal ? 🧠 ⏱️ 3 min
*Fiche §12.5.3, §12.8 · livre §12.5.3, §12.8 · parcours R*

Un `MinMaxScaler` est appris sur des températures d'entraînement qui vont de −5 °C à 25 °C.

a) Que donne `transform` pour 30 °C (2 décimales) ?
b) Vrai ou faux : c'est un bug ; il faut refaire un `fit` sur les données de test pour revenir dans $[0, 1]$.
c) Que donne `transform` pour 30 °C avec `MinMaxScaler(clip=True)` ?
d) Après un `StandardScaler` appris sur l'entraînement, la moyenne des données de test transformées est : (A) exactement 0 ; (B) en général proche de 0, sans être 0, si le test ressemble à l'entraînement ; (C) toujours égale à 1 ; (D) toujours négative.

### 12.Q7 — Univarié ou multivarié ? 🧠 ⏱️ 3 min
*Fiche §12.5.4 · livre §12.5.4 · parcours R, M*

a) Classe chaque transformation : U (univariée) ou M (multivariée). Six lettres, dans l'ordre.
1. `StandardScaler` (une moyenne et un écart-type par colonne) ;
2. une PCA ;
3. un min-max global : un seul minimum et un seul maximum, calculés sur toutes les colonnes ensemble ;
4. `SimpleImputer(strategy="median")` ;
5. remplacer une valeur manquante par la moyenne des 5 exemples les plus proches (`KNNImputer`) ;
6. remplacer chaque valeur par son logarithme.

b) Vrai ou faux : dans une transformation univariée apprise sur l'entraînement, changer les valeurs de la colonne 2 de l'entraînement peut changer la transformation de la colonne 1.

### 12.Q8 — Sélectionner ou réduire la dimension ? 🧠 ⏱️ 3 min
*Fiche §12.6, §12.7 · livre §12.6, §12.7 · parcours R*

a) Classe chaque opération : S (sélection : on garde certaines colonnes telles quelles) ou R (réduction de dimension : on fabrique de nouvelles features). Cinq lettres, dans l'ordre.
1. retirer une colonne constante ;
2. une PCA à 10 composantes ;
3. garder les 5 features les plus corrélées à la cible ;
4. remplacer la taille et la masse par l'IMC ;
5. `VarianceThreshold(threshold=0.01)`.

b) Vrai ou faux : choisir les features les plus corrélées à la cible sur tout le dataset, avant le découpage, ne crée aucune fuite.
c) Vrai ou faux : après une PCA, chaque nouvelle feature correspond à l'une des colonnes d'origine.

### 12.Q9 — Ce que fait (et ne fait pas) une PCA 🧠 ⏱️ 4 min
*Fiche §12.7.1, §12.7.2 · livre §12.7.1, §12.7.2 · parcours R, M*

Vrai ou faux ?
a) La première composante principale est la direction de variance maximale des données centrées.
b) Deux composantes principales différentes sont orthogonales.
c) La PCA se sert des labels pour choisir ses directions.
d) Une PCA ne fabrique que des combinaisons linéaires des features.
e) Si l'on exprime une feature en grammes plutôt qu'en kilogrammes, sans standardiser, la première composante tend à s'aligner sur elle.
f) Garder les composantes qui expliquent 95 % de la variance garantit de garder ce qui sert à séparer les classes.

### 12.Q10 — Quelle découpe pour ces données ? 🧠 ⏱️ 3 min
*Fiche §12.9 · livre §12.9 à §12.9.3 · parcours R*

a) Classe chaque transformation : L (par échantillon : chaque ligne à partir d'elle-même), C (par feature : chaque colonne, à partir de ses valeurs sur l'entraînement) ou E (par élément : la même formule pour chaque case). Six lettres, dans l'ordre.
1. diviser chaque pixel d'une image par 255 ;
2. ramener chaque enregistrement d'électrocardiogramme à son propre maximum ;
3. standardiser chaque colonne d'un tableau de mesures de manchots ;
4. convertir des prix de dollars en euros, à un taux fixé ;
5. diviser chaque vecteur de comptes de mots par sa norme (`Normalizer`) ;
6. imputer la médiane de chaque colonne.

b) Vrai ou faux : une transformation par échantillon qui n'utilise que la ligne elle-même ne peut pas faire fuiter d'information du test vers l'entraînement.

### 12.Q11 — Où se cache la fuite ? 🧠 ⏱️ 4 min
*Fiche §12.10 · livre §12.10 · parcours R*

a) Lesquelles de ces procédures contiennent une fuite de données ? Les lettres, dans l'ordre alphabétique.
(A) standardiser tout le dataset, puis lancer une validation croisée à 5 folds ;
(B) passer `make_pipeline(StandardScaler(), LogisticRegression())` à `cross_val_score` ;
(C) imputer les valeurs manquantes par la médiane de tout le dataset, puis découper en entraînement et test ;
(D) ajuster le scaler sur l'entraînement, puis transformer la validation et le test ;
(E) garder les 20 features les plus corrélées à la cible sur tout le dataset, puis faire une validation croisée ;
(F) diviser tous les pixels par 255, puis découper.

b) Vrai ou faux : si l'on a standardisé `X` avant d'appeler `cross_val_score`, la fonction refait un `fit` du scaler dans chaque fold.

<a id="rappels"></a>

## 🔁 Rappels

Des questions sur les chapitres précédents, pour ne pas oublier.

### 12.R1 — Ch. 11 : Représentation, évaluation, optimisation — où placer la préparation ? 🔁 ★ ⏱️ 5 min
*Ch. 11 (fiche §11.2, les trois ingrédients de l'apprentissage) · parcours R, M, C*

Au ch. 11, un apprentissage combine une **représentation** (ce que le modèle peut exprimer), une **évaluation** (ce qui dit qu'une solution est bonne) et une **optimisation** (la méthode qui cherche une bonne solution).

a) Ajouter la feature $x_1^2$ à une régression linéaire change surtout : (A) sa représentation ; (B) son évaluation ; (C) son optimisation.
b) Standardiser les features d'une régression linéaire **sans pénalité**, entraînée par descente de gradient, sans rien changer d'autre, change surtout : (A) sa représentation ; (B) son évaluation ; (C) son optimisation.
c) Choisir le modèle d'après le F1-score plutôt que d'après l'accuracy change : (A) la représentation ; (B) l'évaluation ; (C) l'optimisation.
d) Vrai ou faux : retirer une feature à la préparation peut empêcher le modèle d'exprimer la bonne règle, quelle que soit la qualité de l'optimisation.

### 12.R2 — Ch. 9 : Pourquoi la pénalité ridge dépend de l'échelle des features 🔁 ★ ⏱️ 5 min
*Ch. 9 (fiche §9.5, Ridge) · parcours R, M, C*

Ridge minimise $\lVert \mathbf{y} - \mathbf{X}\mathbf{w} - b\,\mathbf{1} \rVert^2 + \alpha\,\lVert \mathbf{w} \rVert^2$. Une feature $x$, mesurée en mètres, reçoit le poids $w$. On l'exprime maintenant en centimètres : $x' = 100\,x$.

a) Sans pénalité ($\alpha = 0$), quel poids $w'$ donne exactement les mêmes prédictions ? (A) $w$ ; (B) $100\,w$ ; (C) $w / 100$ ; (D) $w / 10\,000$.
b) Par combien le terme de pénalité de ce poids, $\alpha\,w'^2$, est-il divisé par rapport à $\alpha\,w^2$ ?
c) En centimètres, Ridge pénalise donc cette feature : (A) plus ; (B) moins ; (C) autant.
d) Vrai ou faux : standardiser les features avant Ridge rend le résultat indépendant de l'unité choisie pour chacune.

### 12.R3 — Ch. 5 : Descente de gradient dans une vallée très allongée 🔁 ★ ⏱️ 5 min
*Ch. 5 (fiche, la descente de gradient et la vallée de Rosenbrock) · fil rouge Rosenbrock · parcours R, M, C*

Au ch. 5, la vallée de Rosenbrock était longue et étroite, et la descente de gradient y avançait mal. On étudie ici une vallée plus simple, $f(x, y) = x^2 + 100\,y^2$, de minimum $(0, 0)$, avec la descente de gradient de learning rate $\eta$ : $(x, y) \leftarrow (x, y) - \eta\,\nabla f(x, y)$.

a) Le gradient de $f$ en $(1, 1)$.
b) Un pas multiplie $y$ par $1 - 200\,\eta$ (vérifie-le). Au-delà de quel learning rate la descente diverge-t-elle sur $y$ ? (La descente converge pour tout $\eta$ positif strictement inférieur à cette valeur.)
c) Avec $\eta = 0{,}009$, par quel facteur un pas multiplie-t-il $x$ ?
d) On change d'échelle : $y' = 10\,y$, si bien que $f = x^2 + y'^2$. Au-delà de quel learning rate la descente diverge-t-elle maintenant ?
e) Vrai ou faux : mettre les features à la même échelle rend la vallée moins allongée et permet un learning rate plus grand.

<a id="papier"></a>

## ✏️ ∂ Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 12.1 — One-hot à la main sur Penguins ✏️ ★ ⏱️ 10 min
**Objectif :** encoder à la main des catégories en one-hot, avec et sans colonne retirée, et une catégorie inconnue.
**Prérequis :** 0A (tableaux) · fiche §12.3, §12.3.1 · **Fil rouge :** Penguins · **Parcours :** R, M

Cinq manchots, décrits par leur île, leur sexe et la longueur de leur nageoire :

| manchot | island | sex | flipper_length_mm |
|---|---|---|---|
| 1 | Dream | female | 190 |
| 2 | Biscoe | male | 220 |
| 3 | Torgersen | female | 185 |
| 4 | Biscoe | female | 215 |
| 5 | Dream | male | 200 |

On encode `island` et `sex` en one-hot, catégories rangées par ordre alphabétique (comme `OneHotEncoder`), et l'on garde `flipper_length_mm` telle quelle. Les colonnes sont, dans l'ordre : le bloc de `island`, puis celui de `sex`, puis `flipper_length_mm`.

a) Le nombre de colonnes, sans `drop`.
b) La ligne du manchot 3, sans `drop` (une liste).
c) Le nombre de colonnes avec `drop="first"`.
d) La ligne du manchot 2 avec `drop="first"`.
e) Le nombre de colonnes avec `drop="if_binary"`.
f) Un nouveau manchot arrive : île « Anvers » (absente de l'entraînement), femelle, nageoire de 195 mm. Sa ligne avec l'encodeur sans `drop` et `handle_unknown="ignore"`.
g) Avec un encodage **ordinal** de `island` (ordre alphabétique), le code de Torgersen.

### Ex 12.2 — Min-max et z-score de cinq valeurs ✏️ ★ ⏱️ 10 min
**Objectif :** calculer à la main une normalisation min-max et une standardisation, et voir l'effet du choix de l'écart-type.
**Prérequis :** ch. 2 (moyenne, écart-type, z-score) · fiche §12.5.1, §12.5.2 · **Parcours :** R, M

Cinq températures d'entraînement, en °C : 14, 17, 21, 23 et 25.

a) Leur moyenne.
b) Leur écart-type, en divisant par $n$ (ddof = 0, comme `StandardScaler`).
c) La valeur 21 après une normalisation min-max vers $[0, 1]$ (3 décimales).
d) Le z-score de 17.
e) Le z-score de 25.
f) Le z-score de 25 calculé avec l'écart-type de pandas, `Series.std()`, qui divise par $n - 1$ (3 décimales).
g) La valeur 21 après une normalisation min-max vers $[-1, 1]$ (3 décimales).

### Ex 12.3 — Mise à l'échelle univariée ou multivariée ✏️ ★ ⏱️ 10 min
**Objectif :** appliquer un min-max univarié puis multivarié, et voir ce que chacun conserve.
**Prérequis :** Ex 12.2 · fiche §12.5.4 · **Parcours :** M

Sur l'entraînement, trois features vont de 10 à 30 (f1), de 0 à 5 (f2) et de −20 à 20 (f3). On transforme l'exemple $(20, 4, 0)$.

a) L'exemple après un min-max **univarié** vers $[0, 1]$ (chaque feature avec son propre minimum et son propre maximum).
b) L'exemple après un min-max **multivarié** vers $[0, 1]$ (un seul minimum et un seul maximum, calculés sur les trois features ensemble).
c) Après le min-max multivarié, la largeur de l'intervalle qu'occupe f2 (de sa plus petite à sa plus grande valeur d'entraînement transformée).
d) Vrai ou faux : avec le min-max multivarié, le rapport des étendues de deux features est le même avant et après la transformation.
e) Quatre colonnes donnent les ventes quotidiennes d'un même produit, en euros, dans quatre magasins. Pour que deux valeurs transformées se comparent directement, même quand elles viennent de magasins différents, quelle mise à l'échelle choisir ? (A) univariée ; (B) multivariée.

### Ex 12.4 — Réappliquer la transformation : −10 °C, −50 °C et retour aux voitures ✏️ ★★ ⏱️ 15 min
**Objectif :** appliquer une transformation apprise à des données nouvelles, y compris hors de la plage d'entraînement, puis revenir aux unités d'origine.
**Prérequis :** Ex 12.2 · fiche §12.5.3, §12.8 · **Parcours :** R, M

Une ville prédit le nombre de voitures sur son autoroute, entre 7 h et 8 h, à partir de la température mesurée à minuit (l'idée et les nuits à −10 °C et à −50 °C viennent du §12.8 du livre ; les plages d'entraînement sont à nous). À l'entraînement, les températures vont de −18 °C à 12 °C et les comptages de 120 à 870 voitures ; les deux grandeurs sont ramenées dans $[0, 1]$ par deux `MinMaxScaler`, appris sur ces données.

a) La valeur transformée de −10 °C (3 décimales).
b) Le modèle prédit 0,40 (dans l'échelle transformée des comptages). Combien de voitures cela fait-il ?
c) La valeur transformée de −50 °C (3 décimales).
d) La valeur transformée de 20 °C (3 décimales).
e) Vrai ou faux : pour −50 °C, il faut refaire un `fit` du scaler en ajoutant cette mesure, pour rester dans $[0, 1]$.
f) Pour une nuit très chaude, le modèle prédit −0,1. Combien de voitures cela fait-il ?
g) Une autre équipe standardise plutôt la température : $\mu = -3$ °C et $\sigma = 6$ °C sur l'entraînement. Le z-score de −10 °C (3 décimales).
h) Elle standardise aussi la cible ($\mu = 480$ voitures, $\sigma = 150$) ; son modèle prédit 0,5. Combien de voitures ?
i) Vrai ou faux : pour passer de 0,40 à un nombre de voitures (question b), on applique l'inverse du scaler de la température.

Dans ta copie : la valeur de c) et la prédiction de f) sortent de $[0, 1]$. Est-ce un problème pour la transformation ? Et pour le modèle ?

### Ex 12.5 — Trois découpes d'un même tableau : échantillon, feature, élément ✏️ ★★ ⏱️ 15 min
**Objectif :** appliquer une transformation par feature, par échantillon et par élément, et dire lesquelles s'apprennent.
**Prérequis :** Ex 12.2 · fiche §12.9 · **Parcours :** M

Un tableau d'entraînement de trois exemples et trois features :

| | f1 | f2 | f3 |
|---|---|---|---|
| exemple 1 | 1 | 4 | 10 |
| exemple 2 | 3 | 8 | 20 |
| exemple 3 | 5 | 6 | 30 |

a) Min-max **par feature** vers $[0, 1]$ (chaque colonne avec son minimum et son maximum) : l'exemple 2 transformé.
b) Min-max **par échantillon** vers $[0, 1]$ (chaque ligne avec son propre minimum et son propre maximum) : l'exemple 1 transformé (3 décimales).
c) Normalisation L2 **par échantillon** (chaque ligne divisée par sa norme) : la première valeur de l'exemple 2 transformé (3 décimales).
d) Les valeurs sont des prix en dollars ; on les convertit **par élément** en euros, à 0,9 € pour 1 \$. L'exemple 3 converti.
e) Lesquelles de ces transformations doivent être apprises (`fit`) sur l'entraînement ? (A) le min-max par feature ; (B) le min-max par échantillon ; (C) la normalisation L2 par échantillon ; (D) la conversion par élément. Les lettres, dans l'ordre alphabétique.
f) Un nouvel exemple $(2, 2, 40)$ arrive. Transformé par le min-max **par feature** appris sur le tableau.
g) Le même exemple transformé par le min-max **par échantillon**.

### Ex 12.6 — Montrer que la standardisation donne moyenne 0 et variance 1 ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer les propriétés de la standardisation et de la normalisation min-max, et ce que change la convention de l'écart-type.
**Prérequis :** ch. 2 (moyenne, variance) · 0B ($\mathbb{E}[aX + b]$, $\mathrm{Var}(aX + b)$, sommes) · fiche §12.5.1, §12.5.2 · **Parcours :** M

Une feature prend les valeurs $x_1, \dots, x_n$ sur l'entraînement, de moyenne $\mu$ et d'écart-type $\sigma > 0$ (ddof = 0). On pose $z_i = (x_i - \mu)/\sigma$.

1. Montre que la moyenne des $z_i$ vaut 0.
2. Montre que leur variance (ddof = 0) vaut 1.
3. On calcule $\sigma$ avec ddof = 1 (comme pandas), puis la variance des $z_i$ avec ddof = 0 (comme NumPy). Que trouve-t-on ? Que vaut ce résultat pour $n = 5$ ?
4. Changement d'unité : on mesure $x'_i = a\,x_i + b$, avec $a > 0$. Montre que les z-scores ne changent pas. Que se passe-t-il si $a < 0$ ?
5. Normalisation min-max : $x'_i = (x_i - m)/(M - m)$, où $m < M$ sont le minimum et le maximum. Montre que les $x'_i$ vont exactement de 0 à 1, que l'ordre des valeurs est conservé, et que le rapport $\frac{x'_i - x'_j}{x'_k - x'_l}$ de deux écarts est le même qu'avec les $x$ (pour $x_k \neq x_l$).
6. Une feature constante : que vaut $\sigma$, et pourquoi la formule échoue-t-elle ? Que donne la convention de scikit-learn (diviser par 1 au lieu de 0) pour la standardisation ? Et pour la normalisation min-max vers $[a, b]$ ?

### Ex 12.7 — PCA à la main en 2D : covariance, axe principal, projection ✏️ ★★★ ⏱️ 35 min
**Objectif :** mener une PCA complète à la main sur cinq points : covariance, direction principale, variance expliquée, projection, reconstruction et whitening.
**Prérequis :** Ex 12.2 · ch. 2 (covariance) · 0B (produit scalaire, produit matrice-vecteur) · fiche §12.7.1 (encadrés 🧮) · **Parcours :** M

Cinq points du plan : A$(1, 1)$, B$(3, 3)$, C$(4, 3)$, D$(5, 5)$ et E$(7, 3)$.

a) Le point moyen.
b) La matrice de covariance (ddof = 1), en liste de lignes.
c) Le produit de cette matrice par le vecteur $(2, 1)$.
d) Ce produit montre que $(2, 1)$ est un vecteur propre. Sa valeur propre.
e) La variance totale (la somme des variances des deux features).
f) La part de variance expliquée par la première composante principale, de direction $\mathbf{u} = (2, 1)/\sqrt{5}$ (3 décimales). (On admet que l'autre valeur propre est la variance totale moins celle de d.)
g) Les coordonnées $t$ des cinq points sur $\mathbf{u}$, après centrage, de A à E (3 décimales).
h) La variance (ddof = 1) de ces cinq coordonnées.
i) La reconstruction de D à partir de sa seule coordonnée $t$ : le point $\boldsymbol{\mu} + t\,\mathbf{u}$.
j) La distance entre D et sa reconstruction (3 décimales).
k) La coordonnée de D sur $\mathbf{u}$ après whitening (3 décimales).
l) La variance (ddof = 1) des projections des cinq points sur $\mathbf{v} = (-1, 2)/\sqrt{5}$, direction orthogonale à $\mathbf{u}$.
m) Vrai ou faux : si l'on multiplie toutes les valeurs de la première feature par 10 (une autre unité), l'axe principal reste $(2, 1)/\sqrt{5}$.

Dans ta copie : compare j) à la valeur absolue de la coordonnée de D sur $\mathbf{v}$. Pourquoi sont-elles égales ?

### Ex 12.8 — Variance d'une projection et axe de variance maximale ∂ ★★★ ⏱️ 30 min
**Objectif :** démontrer que la variance des projections sur $\mathbf{u}$ vaut $\mathbf{u}^\top \boldsymbol{\Sigma}\,\mathbf{u}$, puis trouver l'axe de variance maximale, d'abord pour une covariance diagonale, puis après une rotation.
**Prérequis :** Ex 12.7 · 0B (produit scalaire, produit matriciel, transposée) · fiche §12.7.1 (encadrés 🧮) · **Parcours :** M

Soit $\mathbf{X}_c$ des données centrées, de forme $(n, p)$, $\boldsymbol{\Sigma} = \frac{1}{n - 1}\,\mathbf{X}_c^\top \mathbf{X}_c$ leur matrice de covariance, et $\mathbf{u}$ un vecteur unitaire. On note $\mathbf{t} = \mathbf{X}_c\,\mathbf{u}$ les coordonnées des projections.

1. Montre que la moyenne des $t_i$ est nulle, puis que leur variance (ddof = 1) vaut $\mathbf{u}^\top \boldsymbol{\Sigma}\,\mathbf{u}$. (Écris $\sum_i t_i^2 = \mathbf{t}^\top \mathbf{t}$.)
2. En dimension 2, avec $\boldsymbol{\Sigma} = \begin{pmatrix} a & 0 \\ 0 & b \end{pmatrix}$, $a > b > 0$, et $\mathbf{u} = (u_1, u_2)$, $u_1^2 + u_2^2 = 1$ : montre que $\mathbf{u}^\top \boldsymbol{\Sigma}\,\mathbf{u} = b + (a - b)\,u_1^2$. Déduis-en la plus grande et la plus petite variance des projections, et les directions qui les atteignent.
3. Que se passe-t-il si $a = b$ ? Pourquoi dit-on alors que les composantes principales ne sont pas uniques ?
4. En dimension 3, avec $\boldsymbol{\Sigma} = \mathrm{diag}(a, b, c)$ et $a > b > c > 0$ : quelle est la première composante principale ? Parmi les directions unitaires orthogonales à la première, laquelle maximise la variance ? Justifie.
5. Rotation : soit $\mathbf{R}$ une matrice $2 \times 2$ telle que $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$ (une matrice orthogonale : une rotation, par exemple), et $\boldsymbol{\Sigma} = \mathbf{R}\,\mathrm{diag}(a, b)\,\mathbf{R}^\top$, avec $a > b > 0$. On pose $\mathbf{v} = \mathbf{R}^\top \mathbf{u}$. Montre que $\lVert \mathbf{v} \rVert = 1$ et que $\mathbf{u}^\top \boldsymbol{\Sigma}\,\mathbf{u} = \mathbf{v}^\top \mathrm{diag}(a, b)\,\mathbf{v}$. Déduis-en la direction $\mathbf{u}$ de variance maximale, et vérifie que c'est un vecteur propre de $\boldsymbol{\Sigma}$, de valeur propre $a$.
6. En dimension 2, toute direction unitaire s'écrit $\mathbf{u} = (c, s)$ avec $c^2 + s^2 = 1$, et $\mathbf{w} = (-s, c)$ est une direction unitaire orthogonale. Montre que $\mathbf{u}^\top \boldsymbol{\Sigma}\,\mathbf{u} + \mathbf{w}^\top \boldsymbol{\Sigma}\,\mathbf{w} = \Sigma_{11} + \Sigma_{22}$ pour toute matrice de covariance $\boldsymbol{\Sigma}$. Interprète : que fait une PCA de la variance totale ?

<a id="reflexion"></a>

## 🗣️ ⚖️ Réflexion

### Ex 12.9 — La fuite de données expliquée en 5 lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer à un débutant ce qu'est une fuite de données par le prétraitement, et comment l'éviter.
**Prérequis :** ch. 8 (découpage, validation croisée, fuite de données) · fiche §12.2, §12.10 · **Parcours :** R

Une amie qui débute en data science te demande pourquoi son modèle, si bon en validation croisée, déçoit en production. Elle a standardisé et imputé ses données au tout début de son notebook. Explique-lui ce qu'est une fuite de données par le prétraitement, en **cinq lignes** au plus. Contraintes :
- une image de la vie courante ;
- la règle d'or, en une phrase ;
- un exemple concret avec un scaler ou une imputation ;
- le moyen de l'éviter, et pourquoi il marche.

Enregistre-toi ou écris ta réponse, puis compare avec la réponse modèle de `05_solutions.md`. Relis ensuite la figure de la règle d'or de la fiche : que faudrait-il y ajouter pour une validation croisée ?

### Ex 12.10 — Supprimer ou imputer : qui disparaît des données ? ⚖️ ★★ ⏱️ 20 min
**Objectif :** peser ce que coûtent la suppression et l'imputation des valeurs manquantes, et pour qui.
**Prérequis :** ch. 2 (proportions, moyenne) · fiche §12.4, §12.6 · **Fil rouge :** Penguins · **Parcours :** R

Les données brutes des manchots de Palmer (`penguins_raw.csv`, 344 manchots suivis sur trois saisons de reproduction) ont des valeurs manquantes dans les mesures, le sexe et les isotopes du sang (δ¹⁵N, δ¹³C, des indices du régime alimentaire). Le tableau donne, par espèce et par saison, le nombre de manchots qui ont **au moins une** valeur manquante, sur l'effectif du groupe :

| | 2007-08 | 2008-09 | 2009-10 | total |
|---|---|---|---|---|
| Adélie | 13 sur 50 | 0 sur 50 | 0 sur 52 | 13 sur 152 |
| Chinstrap | 0 sur 26 | 0 sur 18 | 1 sur 24 | 1 sur 68 |
| Gentoo | 2 sur 34 | 1 sur 46 | 3 sur 44 | 6 sur 124 |
| total | 15 sur 110 | 1 sur 114 | 4 sur 120 | 20 sur 344 |

La colonne des commentaires donne les raisons : « *Not enough blood for isotopes* » (pas assez de sang pour les isotopes), « *Sexing primers did not amplify* » (l'analyse génétique du sexe a échoué), « *No blood sample obtained* » (aucune prise de sang), « *Adult not sampled* » (adulte non mesuré). Réponds dans ta copie.

1. Un `dropna()` sur ces colonnes : quelle part des manchots disparaît en tout ? Et parmi les Adélie de la première saison ?
2. Les valeurs manquent-elles complètement au hasard ? D'après les commentaires et le tableau, de quoi dépend l'absence d'une mesure ? Classe la situation parmi les trois de la fiche, en justifiant.
3. Tu étudies l'évolution de la masse des Adélie d'une saison à l'autre. Que risques-tu en supprimant les lignes incomplètes ? Et si ta question porte sur la différence entre espèces ?
4. Les isotopes manquent pour une quinzaine de manchots, surtout de la première saison. Compare quatre options : retirer les deux colonnes d'isotopes (une sélection de features) ; imputer leur médiane sur l'entraînement ; imputer la médiane de l'espèce et de la saison ; ajouter une colonne indicatrice « isotopes manquants ». Pour chacune : qu'est-ce que le modèle risque d'apprendre, et quelle règle de préparation faut-il respecter ?
5. Transpose à un cas humain : dans une enquête de santé, le revenu manque surtout chez les ménages les plus pauvres et les plus riches. Pourquoi une imputation par la moyenne peut-elle fausser une décision d'attribution d'aides ? Que faudrait-il écrire dans la documentation du dataset ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 12.E1 — Qu'est-ce qu'une fuite de données ? Donne deux exemples 💼 ★★ ⏱️ 10 min
*Fiche §12.10 · ch. 8 · parcours R*

« Qu'est-ce qu'une fuite de données ? Donnez deux exemples différents, et dites comment vous les auriez évitées. »

### 12.E2 — Standardisation ou normalisation min-max : laquelle, et pourquoi ? 💼 ★★ ⏱️ 10 min
*Fiche §12.5 · parcours R*

« Standardisation ou normalisation min-max : laquelle utilisez-vous, et quand ? Faut-il toujours mettre les features à l'échelle ? »

### 12.E3 — Comment traites-tu les valeurs manquantes ? 💼 ★★ ⏱️ 10 min
*Fiche §12.4 · prérequis 12.10 · parcours R*

« Votre dataset a 15 % de valeurs manquantes sur une feature importante. Comment les traitez-vous ? »

### 12.E4 — À quoi sert une PCA et quelles sont ses limites ? 💼 ★★ ⏱️ 10 min
*Fiche §12.7 · parcours R*

« Expliquez la PCA à un collègue qui ne la connaît pas. À quoi sert-elle, et quand ne faut-il pas l'utiliser ? »

### 12.E5 — Encoder une variable catégorielle à 10 000 modalités 💼 ★★ ⏱️ 10 min
*Fiche §12.3.1 (encadré 🕰️) · parcours R*

« Une feature a 10 000 catégories possibles, par exemple un identifiant de produit. Comment l'encodez-vous ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch12_preparation/03_notebook.ipynb`) ; ceux marqués 🔨 complètent ta librairie `mylearn/preprocessing.py`. La partie 0 du notebook vérifie tes réponses courtes aux quiz, aux rappels et aux exercices ✏️ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 12.11 | Diagnostic de Penguins brut avec pandas | 📦 | ★ | 15 |
| 12.12 | CSV à la française : virgules décimales et « 7e-3 » | 🐛 | ★ | 15 |
| 12.13 | Où tombent les données de test après un MinMaxScaler ? | 🔮 | ★ | 10 |
| 12.14 | pandas pour préparer : dates (« Date Egg »), jointure merge avec une table des îles, pivot_table | 📦 | ★★ | 30 |
| 12.15 | Lire des données dans une base SQL : sqlite3, SELECT, JOIN, GROUP BY et pd.read_sql | 📦 | ★★ | 30 |
| 12.16 | Coder StandardScaler (fit, transform, inverse) | 🔨 | ★★ | 25 |
| 12.17 | Coder MinMaxScaler avec feature_range et clip | 🔨 | ★★ | 20 |
| 12.18 | Coder SimpleImputer (moyenne, médiane, mode, constante) | 🔨 | ★★ | 25 |
| 12.19 | Scalers de scikit-learn face aux points aberrants de California | 📦 | ★★ | 25 |
| 12.20 | Le scaler réentraîné sur le test et les get_dummies désalignés | 🐛 | ★★ | 20 |
| 12.21 | Trafic et température : transformer la cible et revenir aux voitures | 📦 | ★★ | 25 |
| 12.22 | Projeter un nuage 2D : axe horizontal contre axe de variance maximale | 🎨 | ★★ | 25 |
| 12.23 | Lire une courbe de variance expliquée cumulée | 📈 | ★★ | 15 |
| 12.24 | Combien de composantes pour 90 % de la variance de MNIST ? | 🔮 | ★★ | 20 |
| 12.25 | Sélection de features : colonnes constantes, VarianceThreshold, SelectKBest | 📦 | ★★ | 25 |
| 12.26 | Coder OrdinalEncoder et OneHotEncoder | 🔨 | ★★★ | 45 |
| 12.27 | Coder la PCA (SVD, variance expliquée, whitening, reconstruction) | 🔨 | ★★★ | 60 |
| 12.28 | Chiffres propres : PCA sur MNIST et reconstructions | 📦 | ★★★ | 40 |
| 12.29 | Échelle des features et descente de gradient | 🔬 | ★★★ | 40 |
| 12.30 | PCA, t-SNE ou UMAP pour voir MNIST en 2D | 📦 | ★★★ | 40 |
| 12.31 | Fuite de données par le prétraitement : scaler, imputation et TargetEncoder ajustés avant le split | 🔬 | ★★★ | 45 |
| 12.32 | Une fonction de préparation documentée et son test anti-fuite | 🛠️ | ★★★ | 30 |
| 12.33 | Défi : Penguins brut prêt pour l'entraînement, sans fuite | 🏆 | ★★★★ | 100 |
