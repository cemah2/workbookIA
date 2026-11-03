# B6 · Explicabilité, équité et éthique — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/b6_explicabilite/06_mes_reponses.md` (créée par `python tools/start_chapter.py B6`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels** et des exercices ✏️ B6.1, B6.3, B6.4 et B6.5 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les dérivations ∂ B6.2 et B6.6, l'estimation 🧮 B6.7, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), une liste de nombres pour un vecteur (`[0.5, -1.25]`). Un taux s'écrit comme une proportion (`0.25`), pas en pourcentage.
> Conventions du chapitre (fiche, encadré 📐) : dans les formules, les features sont numérotées **de 1 à $M$** ; dans le code (ta librairie, shap, scikit-learn), ce sont les **colonnes 0 à $M - 1$** : la feature 1 est la colonne 0, et `phi[0]` est $\phi_1$. Les canaux d'une couche s'écrivent $A^1, A^2, \dots$ dans les formules et `A[0], A[1], …` dans le code. $v(S)$ est la valeur de la coalition $S$ ; $v(\varnothing)$, la valeur de base. Pour l'équité, $\hat{y} \in \{0, 1\}$ est la décision, $y$ la vérité, $p$ un taux de base. Les références juridiques sont un repère de culture professionnelle, pas un conseil juridique.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### B6.Q1 — Interprétable par construction ou expliqué après coup ? 🧠 ⏱️ 3 min
*Fiche §206.1 · parcours R*

a) Lequel de ces trois modèles est interprétable par construction ? (A) une forêt de 500 arbres, expliquée par TreeSHAP ; (B) une régression logistique à six features standardisées ; (C) un réseau dense accompagné de la carte de saillance de chaque prédiction.
b) Vrai ou faux : une explication post hoc reproduit exactement la règle que le modèle applique.
c) Grad-CAM est une méthode : (A) spécifique, qui lit les cartes d'activation et les gradients d'un CNN ; (B) agnostique, qui n'interroge le modèle que par ses prédictions ; (C) interprétable par construction.
d) Vrai ou faux : l'importance par permutation s'applique telle quelle à un SVM, à une forêt ou à un réseau, sans rien savoir de leur structure.
e) Un arbre de décision binaire de profondeur 4 : combien de feuilles au plus, c'est-à-dire de règles « si … alors … » distinctes ?

### B6.Q2 — Explication globale ou locale : quel outil répond à quelle question ? 🧠 ⏱️ 3 min
*Fiche §206.1, §206.2 · parcours R*

a) « Pourquoi le dossier n° 4512 a-t-il été refusé ? » appelle : (A) une explication globale ; (B) une explication par groupe ; (C) une explication locale.
b) « Quelles features ce modèle utilise-t-il le plus, sur l'ensemble des clients ? » : (A) l'importance par permutation, sur la validation ; (B) le waterfall du client le plus récent ; (C) la courbe ICE d'un client tiré au hasard.
c) Vrai ou faux : une feature `pret_en_cours`, qui vaut 1 pour 0,5 % des clients et dont l'importance par permutation vaut 0,001, ne peut pas être la raison principale d'un refus.
d) La moyenne des $\lvert\phi_i\rvert$ d'une feature sur 1 000 clients est : (A) une explication locale de chacun des 1 000 clients ; (B) une explication globale construite à partir d'explications locales ; (C) une autre façon de calculer l'importance par permutation.
e) Un modèle a un R² de validation de 0,83. Après cinq mélanges de la colonne `surface`, il vaut 0,66 ; 0,62 ; 0,65 ; 0,63 et 0,64. Que vaut l'importance par permutation de `surface`, à 2 décimales ?

### B6.Q3 — Dépendance partielle et courbes ICE : ce que montre la moyenne, ce qu'elle cache 🧠 ★★ ⏱️ 3 min
*Fiche §206.2 · parcours R*

a) Le modèle $f(x_1, x_2) = x_1 x_2$ ; $x_2$ vaut $-1$ pour la moitié des exemples, $+1$ pour l'autre moitié. La dépendance partielle de $x_1$ est : (A) une droite de pente 1 ; (B) une droite de pente $-1$ ; (C) plate.
b) Combien de pentes différentes ont les courbes ICE de $x_1$ dans le cas a) ?
c) Le modèle $f(\mathbf{x}) = 3x_1 + x_2^2$ ; $x_2$ prend les valeurs 0, 1 et 2, chacune pour un tiers des exemples. Que vaut $\operatorname{PD}_1(2)$, la dépendance partielle de $x_1$ en $v = 2$, à 2 décimales ?
d) Vrai ou faux : avec `centered=True`, des courbes ICE parallèles se superposent toutes en une seule courbe.
e) La surface et le nombre de pièces d'un logement sont très corrélés. La dépendance partielle de la surface en 200 m² fait la moyenne des prédictions sur : (A) tous les logements, mis à 200 m² en gardant chacun son nombre de pièces, parfois 2 ; (B) les seuls logements de 190 à 210 m² ; (C) tous les logements, mis à 200 m² avec un nombre de pièces ajusté à cette surface.

### B6.Q4 — Les quatre propriétés de Shapley : efficacité, symétrie, joueur nul, additivité 🧠 ⏱️ 3 min
*Fiche §206.3 · parcours R, M*

a) Un modèle prédit $f(\mathbf{x}) = 0{,}72$ pour un client ; la valeur de base vaut 0,40 ; deux de ses trois features reçoivent $\phi_1 = 0{,}15$ et $\phi_2 = -0{,}05$. Que vaut $\phi_3$, à 2 décimales ?
b) Une feature que le modèle ne lit jamais reçoit 0 : quelle propriété ? (A) le joueur nul ; (B) l'efficacité ; (C) la symétrie.
c) Un modèle est la somme de deux arbres, $f = g + h$ ; pour le même client et le même fond, $\phi_1$ vaut 0,3 pour $g$ et $-0{,}1$ pour $h$. Que vaut $\phi_1$ pour $f$, à 1 décimale ?
d) Vrai ou faux : si $f(a, b) = f(b, a)$ pour tous $a$ et $b$, alors $\phi_1 = \phi_2$ pour un client tel que $\mathbf{x} = (1, 5)$.
e) Avec $M = 5$ features, combien de coalitions ne contiennent pas une feature $i$ donnée ?

### B6.Q5 — Changer le fond (background) change les valeurs SHAP : vrai ou faux 🧠 ★★ ⏱️ 3 min
*Fiche §206.3, §206.4 · parcours R*

a) Vrai ou faux : pour un même modèle et un même client, deux fonds différents donnent la même somme $\phi_1 + \dots + \phi_M$.
b) Le fond se réduit à une seule ligne $\mathbf{b}$. La valeur de base vaut alors : (A) la prédiction moyenne sur l'entraînement ; (B) 0 ; (C) $f(\mathbf{b})$.
c) Modèle linéaire $f(\mathbf{x}) = 2x_1 - 3x_2$, client $\mathbf{x} = (4, 1)$, fond A dont les colonnes ont pour moyennes $(3, 1)$. Que vaut $\phi_1$ ?
d) Même modèle, même client, fond B de moyennes $(5, 2)$. Que vaut $\phi_1$ ?
e) Vrai ou faux : `shap.TreeExplainer(model)` sans fond et `shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")` donnent les mêmes valeurs.
f) Tu passes un fond de 300 lignes à `shap.maskers.Independent(bg)`, sans autre argument. Combien de lignes du fond shap utilise-t-il ?

### B6.Q6 — Features corrélées : ce que SHAP répartit et ce qu'il ne dit pas 🧠 ★★ ⏱️ 3 min
*Fiche §206.4, §206.7 · parcours —*

$x_2$ est une copie presque parfaite de $x_1$. Le client vaut $\mathbf{x} = (1{,}5 ;\ 1{,}5)$ ; les deux colonnes du fond ont pour moyenne 0,5. Deux modèles, aussi bons l'un que l'autre : A, $f(\mathbf{x}) = 2x_1 + 2x_2$ ; B, $f(\mathbf{x}) = 4x_1$ (une L1 n'a gardé qu'une copie).

a) Shapley interventionnel : que vaut $\phi_1$ pour le modèle A ?
b) Que vaut $\phi_2$ pour le modèle B ?
c) Vrai ou faux : $\phi_1 + \phi_2$ est le même pour A et pour B.
d) Le Shapley observationnel (conditionnel) du modèle B : (A) donne 0 à $x_2$, comme l'interventionnel ; (B) n'est pas défini quand des features sont corrélées, faute de pouvoir séparer leurs effets ; (C) peut créditer $x_2$, que le modèle B ne lit pas.
e) Pour rendre compte à un comité de l'effet de ces deux colonnes, le plus solide est : (A) la somme de leurs valeurs SHAP ; (B) la plus grande des deux valeurs ; (C) la valeur de chacune, telle quelle.

### B6.Q7 — Gradient, gradient × entrée, gradients intégrés : pourquoi une ligne de référence ? 🧠 ★★ ⏱️ 3 min
*Fiche §206.5 · parcours —*

Un modèle à une entrée sature : $f(x) = \min(x, 1)$, et l'on explique $x = 3$.

a) Que vaut le gradient $f'(x)$ en $x = 3$ ?
b) Que vaut l'attribution des gradients intégrés avec la référence $x' = 0$ ?
c) Et avec la référence $x' = 2$ ?
d) Vrai ou faux : pour un modèle linéaire $f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x}$ et la référence $\mathbf{0}$, gradient × entrée et gradients intégrés donnent les mêmes attributions.
e) Une radiographie montre une lésion blanche (pixels à 1) sur un fond gris uniforme (pixels à 0,5) ; la référence est l'image uniforme à 0,5. Les gradients intégrés : (A) n'attribuent rien aux pixels du fond ; (B) attribuent le plus aux pixels du fond ; (C) attribuent la même valeur à tous les pixels.

### B6.Q8 — Grad-CAM : pourquoi la dernière couche convolutive ? 🧠 ⏱️ 3 min
*Fiche §206.6 · parcours R*

a) Un ResNet-18 de torchvision reçoit une image de 160 × 160 pixels. Quelle est la largeur des cartes d'activation de `layer4` ?
b) Juste après `layer4` vient : (A) une convolution de plus, de stride 2 ; (B) un pooling moyen global ; (C) une softmax, qui garde la position.
c) La sortie de `layer1` divise la taille de l'image par 4. Pour une image de 224 pixels, quelle est la largeur d'une carte Grad-CAM calculée sur `layer1` ?
d) Vrai ou faux : sans le ReLU final, une carte Grad-CAM peut contenir des valeurs négatives.
e) Le score $y^k$ dont on prend le gradient est : (A) la loss de cross-entropy de l'image ; (B) la probabilité de la classe $k$ après la softmax ; (C) le logit de la classe $k$, avant la softmax.

### B6.Q9 — Une carte de saillance qui ne dépend pas du modèle : que prouve le test de randomisation ? 🧠 ★★ ⏱️ 3 min
*Fiche §206.6, §206.7 · parcours R*

a) Une carte d'attribution ne change presque pas quand on randomise toutes les couches du réseau. On en conclut : (A) qu'elle décrit bien le modèle, puisqu'elle est stable ; (B) qu'elle ne dépend pas de ce que le réseau a appris ; (C) que le réseau résiste aux attaques.
b) Vrai ou faux : une méthode qui réussit le test de randomisation est donc fidèle au modèle.
c) La randomisation en cascade : (A) randomise une seule couche à la fois, les autres gardent leurs poids appris ; (B) mélange les labels, puis réentraîne le réseau ; (C) randomise les couches de la sortie vers l'entrée, en les cumulant.
d) Vrai ou faux : une corrélation de rangs de Spearman égale à 1 entre deux cartes signifie que leurs valeurs sont égales, pixel par pixel.
e) Un CNN CIFAR-10 est réentraîné sur des labels mélangés au hasard. Quelle accuracy de test attends-tu, à 1 décimale ?

### B6.Q10 — Sources de biais : données, labels, variables de substitution, boucles de rétroaction 🧠 ⏱️ 3 min
*Fiche §206.8 · parcours R*

a) Un modèle de reconnaissance vocale, entraîné à 90 % sur des voix d'hommes, se trompe plus souvent sur des voix de femmes : (A) un biais de représentation ; (B) une boucle de rétroaction entre décisions et données ; (C) un biais de labels, posés par les annotateurs.
b) Un modèle de recrutement apprend la note de « potentiel » que des managers donnaient à leurs équipes : (A) un biais de mesure d'une feature ; (B) un biais de labels ; (C) un biais de déploiement.
c) Un modèle de détection de fraude n'apprend que sur les dossiers contrôlés, et les contrôles visent les quartiers que le modèle signale : (A) un biais de labels ; (B) une variable de substitution ; (C) une boucle de rétroaction.
d) Vrai ou faux : dans un CV, la liste des sports pratiqués peut servir de variable de substitution au sexe.
e) Un capteur médical mesure moins bien un paramètre chez certains patients, et cette mesure est une feature du modèle : (A) un biais de mesure ; (B) un biais de déploiement ; (C) une boucle de rétroaction.

### B6.Q11 — Parité démographique, égalité des chances, calibration : peut-on tout avoir ? 🧠 ★★ ⏱️ 3 min
*Fiche §206.9 · parcours R*

Deux groupes ont des taux de base de 0,6 et 0,25. Un classifieur **parfait** décide $\hat{y} = y$ pour tout le monde.

a) Que vaut son écart de parité démographique $\Delta_{\text{DP}}$, à 2 décimales ?
b) Vrai ou faux : son écart d'equalized odds est nul.
c) Vrai ou faux : il vérifie aussi la parité prédictive.
d) Un autre classifieur a, dans un groupe de taux de base 0,5, une precision (PPV) de 0,8 et un FNR de 0,2. Que vaut son FPR dans ce groupe, à 2 décimales ?
e) Une banque accorde des prêts ; le positif est « rembourserait ». Pour les demandeurs, l'erreur la plus coûteuse est d'être refusé alors qu'ils auraient remboursé. Quel critère privilégier ? (A) l'égalité des chances ; (B) la parité démographique ; (C) l'égalité des FPR.

### B6.Q12 — AI Act : dans quelle catégorie de risque tombe un scoring de crédit ? 🧠 ⏱️ 3 min
*Fiche §206.11 · parcours R*

a) Une banque note la solvabilité des particuliers qui demandent un prêt : (A) risque minimal ; (B) haut risque (annexe III) ; (C) pratique interdite (art. 5).
b) Une banque détecte les paiements frauduleux par carte : (A) hors du point 5 b) de l'annexe III ; (B) haut risque au titre du point 5 b) de l'annexe III ; (C) pratique interdite (art. 5).
c) Une autorité publique note les citoyens d'après leurs comportements sociaux, et la note conditionne l'accès à des services : (A) risque minimal ; (B) transparence (art. 50) ; (C) pratique interdite (art. 5).
d) Vrai ou faux : depuis le règlement 2026/1744, les obligations des systèmes à haut risque de l'annexe III s'appliquent dès le 2 août 2026.
e) Pour un scoring de crédit, l'analyse d'impact sur les droits fondamentaux (art. 27) revient : (A) à l'entreprise qui a développé le système (le fournisseur) ; (B) à la banque qui l'utilise (le déployeur) ; (C) à la personne notée.
f) Vrai ou faux : le point 5 b) vise aussi la notation de crédit des entreprises.

<a id="rappels"></a>

## 🔁 Rappels

### B6.R1 — Ch. 21 : visualiser ce qu'a appris un CNN, et l'attaque FGSM 🔁 ★ ⏱️ 5 min
*Ch. 21 (maximisation d'activation, FGSM) · fiche §206.5, §206.6 · parcours R*

a) Une image de trois pixels $\mathbf{x} = (0{,}40 ;\ 0{,}98 ;\ 0{,}10)$, dans $[0, 1]$, et le gradient de la loss $\nabla_{\mathbf{x}} L = (-0{,}7 ;\ 0{,}2 ;\ 0)$. Donne l'image attaquée par FGSM avec $\varepsilon = 0{,}03$, pixels ramenés dans $[0, 1]$, sous la forme d'une liste de trois nombres à 2 décimales.
b) Vrai ou faux : la passe arrière de FGSM (ch. 21) donne une carte de saillance si l'on y dérive le score de la classe au lieu de la loss, et si l'on garde la valeur absolue du gradient au lieu de son signe.
c) La maximisation d'activation modifie, pas à pas : (A) les poids du réseau, par descente de gradient ; (B) les pixels d'une image ; (C) les labels des images d'entraînement.
d) Un score linéaire $\mathbf{w}^\top\mathbf{x}$ sur 3 072 pixels, avec $\lvert w_j \rvert = 0{,}01$ pour tous. De combien la perturbation $\varepsilon\,\operatorname{sign}(\mathbf{w})$, avec $\varepsilon = 0{,}01$, change-t-elle le score, à 2 décimales ?
e) Vrai ou faux : la montée de gradient qui visualise un filtre part d'un bruit et cherche un minimum de l'activation.

### B6.R2 — Ch. 14 : feature_importances_ contre importance par permutation 🔁 ★ ⏱️ 5 min
*Ch. 14 (importance MDI, importance par permutation) · fiche §206.2 · parcours R*

Une forêt profonde a pour features l'identifiant de chaque ligne (une valeur différente par ligne) et des mesures utiles.

a) L'importance MDI (`feature_importances_`) de l'identifiant est : (A) parmi les plus grandes ; (B) proche de 0, faute d'information utile ; (C) négative, parce qu'elle gêne l'apprentissage.
b) Vrai ou faux : un `feature_importances_` élevé peut venir de ce que la forêt a appris par cœur les exemples d'entraînement.
c) Que vaut la somme des `feature_importances_` d'une forêt de scikit-learn ?
d) L'importance par permutation de l'identifiant, sur la validation, est : (A) la plus grande ; (B) proche de 0 ; (C) négative et grande.
e) Sur 400 exemples de validation, l'accuracy passe de 0,88 à 0,86 en moyenne quand on mélange une feature. Combien d'exemples de plus, en moyenne, sont mal classés ?

### B6.R3 — Ch. 3 : TPR, FPR et precision à partir d'une matrice de confusion 🔁 ★ ⏱️ 5 min
*Ch. 3 (matrice de confusion, `confusion_matrix`) · fiche §206.9 · parcours R*

`confusion_matrix(y_true, y_pred)` rend `[[88, 12], [10, 40]]` (lignes : la vérité 0 puis 1 ; colonnes : la prédiction 0 puis 1).

a) Le TPR (recall), à 2 décimales.
b) Le FPR, à 2 décimales.
c) La precision, à 3 décimales.
d) Le taux de sélection $P(\hat{y} = 1)$, à 3 décimales.
e) La case `[0, 1]` de cette matrice contient : (A) les faux positifs ; (B) les faux négatifs ; (C) les vrais positifs.

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes (des fractions si besoin) pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex B6.1 — Valeurs de Shapley exactes à la main : trois features, huit coalitions ✏️ ★★ ⏱️ 25 min
**Objectif :** calculer les huit valeurs de coalition d'un jeu à trois features, en tirer les valeurs de Shapley et voir comment le fond les change.
**Prérequis :** 0B (fonctions de plusieurs variables) · fiche §206.3 (jeu coopératif, $v(S)$ avec un fond, poids de Shapley) · **Parcours :** R, M

Un score $f(\mathbf{x}) = 2x_1 + x_1x_3 + x_2$, un client $\mathbf{x} = (2, 1, 3)$ et un fond de deux lignes, $\mathbf{b}^{(1)} = (0, 2, 0)$ et $\mathbf{b}^{(2)} = (2, 2, 2)$. Une coalition $S$ vaut $v(S)$, la moyenne de $f$ sur les deux lignes du fond où les features de $S$ prennent les valeurs du client.

a) $v(\varnothing)$, la valeur de base.
b) $v(\{1\})$.
c) $v(\{1, 3\})$.
d) Le poids de Shapley de la coalition $\{3\}$ dans le calcul de $\phi_1$, à 2 décimales.
e) $\phi_1$, à 1 décimale.
f) $\phi_2$.
g) $\phi_3$, à 1 décimale.
h) Vrai ou faux : l'écart du terme d'interaction $x_1x_3$ à sa moyenne sur le fond est partagé à parts égales entre les features 1 et 3.
i) Combien d'évaluations de $f$ (une ligne = une évaluation) faut-il pour calculer ces trois valeurs de Shapley exactement, si chaque $v(S)$ n'est calculé qu'une fois, sur les deux lignes du fond, la coalition pleine comprise ?
j) On remplace le fond par **une seule ligne**, sa ligne moyenne $(1, 2, 1)$. Que vaut alors $\phi_1$ ?
k) Avec ta librairie, `phi, base = shapley_values(f, [2, 1, 3], [[0, 2, 0], [2, 2, 2]])`. Que contient `phi[2]` ? (A) $\phi_2$, la valeur de la deuxième feature ; (B) $\phi_3$ ; (C) la valeur de base.

Dans ta copie : dresse le tableau des huit coalitions (masque, $v(S)$) et, pour chaque feature, ses quatre contributions marginales avec leurs poids. Vérifie l'efficacité. Pourquoi $\phi_2$ ne dépend-il pas de la coalition ? Compare a) à $f(1, 2, 1)$ : la valeur de base est-elle la prédiction de la ligne moyenne ? Pourquoi le partage de h) n'est-il pas égal ? Fais le calcul ligne du fond par ligne du fond (les valeurs de Shapley d'un fond de deux lignes sont la moyenne de celles de chaque ligne, puisque $v(S)$ est une moyenne), et compare $x_1 - b_1$ et $x_3 - b_3$.

### Ex B6.2 — Modèle linéaire et fond indépendant : φ_i = w_i (x_i − E[x_i]) ∂ ★★ ⏱️ 20 min
**Objectif :** démontrer la formule des valeurs de Shapley d'un modèle linéaire, puis voir ce qu'elle devient avec une interaction.
**Prérequis :** Ex B6.1 · ch. 2 (moyenne) · fiche §206.3 (fonction de valeur avec un fond, poids de Shapley, le cas linéaire) · **Parcours :** M

Un modèle linéaire $f(\mathbf{x}) = w_0 + \sum_{j=1}^{M} w_j x_j$ et un fond de $B$ lignes $\mathbf{b}^{(1)}, \dots, \mathbf{b}^{(B)}$ ; $\bar b_j = \frac{1}{B}\sum_k b^{(k)}_j$ est la moyenne de la feature $j$ sur le fond (le $\mathbb{E}[x_j]$ de la formule). Deux faits sont admis : **(i)** la moyenne est linéaire, $\frac{1}{B}\sum_k (a + c\,u_k) = a + c\,\bar u$ ; **(ii)** pour une feature $i$ fixée, les poids de Shapley $\frac{|S|!\,(M - |S| - 1)!}{M!}$ des $2^{M-1}$ coalitions qui ne contiennent pas $i$ somment à 1 (encadré 🧮 de 206.3). Réponds dans ta copie.

1. **Une ligne du fond.** Écris $f(\mathbf{x}_S, \mathbf{b}_{\bar S})$ pour une ligne $\mathbf{b}$, en séparant les features de $S$ des autres. Déduis-en $v(S)$ avec (i).
2. **La contribution marginale.** Pour $i \notin S$, calcule $v(S \cup \{i\}) - v(S)$. De quoi dépend-elle, et de quoi ne dépend-elle pas ?
3. **La formule.** Conclus avec (ii) : $\phi_i = w_i\,(x_i - \bar b_i)$.
4. **L'efficacité.** Montre que $v(\varnothing) = f(\bar{\mathbf{b}})$, la prédiction de la ligne moyenne, puis que $\sum_i \phi_i = f(\mathbf{x}) - v(\varnothing)$. Pourquoi ✏️ B6.1 a) ne vérifiait-il pas $v(\varnothing) = f(\bar{\mathbf{b}})$ ?
5. **Application.** $w_0 = 1$, $\mathbf{w} = (0{,}5 ;\ -2 ;\ 0)$, client $\mathbf{x} = (4, 1, 7)$, fond de trois lignes $(2, 0, 1)$, $(4, 1, 5)$, $(0, 2, 3)$. Calcule les trois $\phi_i$, la valeur de base et $f(\mathbf{x})$. Quelles features reçoivent 0, et pour deux raisons différentes ?
6. **Un fond corrélé.** On remplace le fond par trois lignes de mêmes moyennes, mais où $x_1$ et $x_2$ montent ensemble. Les $\phi_i$ changent-ils ? Qu'est-ce que le Shapley interventionnel ignore du fond ?
7. **La régression logistique.** $f$ rend $\sigma(w_0 + \mathbf{w}^\top\mathbf{x})$. Sur quelle sortie la formule de 3 reste-t-elle vraie : la probabilité ou le logit ? Pourquoi pas l'autre ?
8. **Une interaction.** Le modèle $f(\mathbf{x}) = x_1 x_2$ et un fond d'**une** ligne $\mathbf{b}$. Montre que $\phi_1 = \frac{1}{2}(x_1 - b_1)(x_2 + b_2)$ et $\phi_2 = \frac{1}{2}(x_2 - b_2)(x_1 + b_1)$, et vérifie l'efficacité. Retrouve le second exemple de la docstring de `shapley_values` (client $(1, 1)$, fond $(0, 0)$), puis la part de l'interaction $x_1x_3$ dans ✏️ B6.1 j) (la feature 2 n'entre pas dans ce terme : un joueur nul ajouté au jeu ne change pas les parts des autres, ce que tu peux vérifier sur les huit coalitions).

### Ex B6.3 — Gradients intégrés à la main sur f(x) = x₁² + 3x₂ et vérification de la complétude ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer à la main gradient, gradient × entrée et gradients intégrés, voir l'effet de la règle d'intégration et de la ligne de référence, et traduire un calcul en appel de ta librairie.
**Prérequis :** ch. 5 (dérivées) · 0B (dérivées partielles) · fiche §206.5 (gradients intégrés, complétude, sommes de Riemann et trapèzes), §206.6 (encadré 🕰️ sur captum) · **Parcours :** M

Le modèle $f(\mathbf{x}) = x_1^2 + 3x_2$, en **numérotation des formules** ($x_1$ est la colonne 0 du code ; la docstring de `integrated_gradients` écrit le même modèle $x_0^2 + 3x_1$). On explique $\mathbf{x} = (2, -1)$, avec la référence $\mathbf{x}' = \mathbf{0}$ sauf en i) et j). $n$ est le nombre d'**intervalles**, comme dans ta docstring : 'left' prend les points $k/n$ pour $k = 0, \dots, n - 1$, 'right' $k = 1, \dots, n$, chacun de poids $1/n$, et 'trapezoid' les $n + 1$ points $k/n$, de poids $1/n$ divisés par deux aux extrémités.

a) Le gradient $\nabla f(\mathbf{x})$, sous la forme d'une liste de deux nombres.
b) Gradient × entrée en $\mathbf{x}$, une liste de deux nombres.
c) L'écart à expliquer, $f(\mathbf{x}) - f(\mathbf{x}')$.
d) Les gradients intégrés exacts (l'intégrale calculée exactement), une liste de deux nombres.
e) $\operatorname{IG}_1$ par la règle 'left' avec $n = 5$, à 1 décimale.
f) $\operatorname{IG}_1$ par la règle 'right' avec $n = 5$, à 1 décimale.
g) $\operatorname{IG}_1$ par la règle 'trapezoid' avec $n = 5$, à 1 décimale.
h) Avec la règle 'left', le plus petit $n$ pour lequel l'erreur de complétude, $\lvert f(\mathbf{x}) - f(\mathbf{x}') - (\operatorname{IG}_1 + \operatorname{IG}_2) \rvert$, vaut au plus 0,03.
i) Avec la référence $\mathbf{x}' = (1, 0)$ : $\operatorname{IG}_1$ exact.
j) Avec la référence $\mathbf{x}' = (1, 0)$ : la somme des deux attributions exactes.
k) captum compte des **points** : son `riemann_trapezoid` avec 6 points, sur ce modèle et avec la référence $\mathbf{0}$, donne quelle valeur de $\operatorname{IG}_1$, à 2 décimales ?
l) Quel appel de ta librairie calcule e) (et son $\operatorname{IG}_2$) ? (A) `integrated_gradients(grad_fn, [2.0, -1.0], n_steps=6, method="left")` ; (B) `integrated_gradients(grad_fn, [2.0, -1.0], n_steps=5, method="right")` ; (C) `integrated_gradients(grad_fn, [2.0, -1.0], n_steps=5, method="left")`.

Dans ta copie : pourquoi la règle des trapèzes est-elle exacte ici, quel que soit $n$, alors que 'left' et 'right' se trompent dans des sens opposés ? Pourquoi $\operatorname{IG}_2$ est-il exact avec toutes les règles ? Avec la référence de i), la somme j) est-elle le signe que les features ne comptent pas ? Que dit b) de gradient × entrée, comparé à c) ?

### Ex B6.4 — Grad-CAM à la main : deux cartes 3×3 et leurs gradients ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer à la main les poids des canaux, la carte Grad-CAM avant et après le ReLU, sa normalisation, et voir qu'une autre classe éclaire une autre région.
**Prérequis :** ch. 21 (cartes d'activation) · fiche §206.6 (Grad-CAM, l'exemple à trois canaux, la normalisation de `grad_cam`) · **Parcours :** M

Une couche a deux canaux de 3 × 3. Ses activations, et les gradients du score $y^k$ d'une classe $k$ par rapport à elles :

$$A^1 = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 3 & 2 & 1 \end{pmatrix},\quad A^2 = \begin{pmatrix} 0 & 1 & 2 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix};\qquad \frac{\partial y^k}{\partial A^1} = \begin{pmatrix} 0 & 1 & 2 \\ 1 & 1 & 1 \\ 2 & 1 & 0 \end{pmatrix},\quad \frac{\partial y^k}{\partial A^2} = \begin{pmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ -1 & -0{,}5 & 0 \end{pmatrix}.$$

La normalisation est celle de ta `grad_cam` : chaque carte divisée par sa plus grande valeur absolue.

a) Le poids $\alpha_1$, à 1 décimale.
b) Le poids $\alpha_2$, à 1 décimale.
c) La valeur de la somme pondérée $\alpha_1 A^1 + \alpha_2 A^2$ dans la case en haut à droite, **avant** le ReLU, à 1 décimale.
d) Combien de cases le ReLU met-il à 0 ?
e) La valeur de la carte normalisée (après ReLU) dans la case du centre, à 2 décimales.
f) Où est le maximum de la carte de la classe $k$ ? (A) en haut à gauche ; (B) en bas à gauche ; (C) en haut à droite.
g) Sans le ReLU (`relu=False`), la carte normalisée dans la case en haut à droite, à 2 décimales.
h) Pour une autre classe $k'$, les poids valent $\alpha'_1 = -0{,}5$ et $\alpha'_2 = 1$. Où est le maximum de sa carte ? (A) en haut à gauche ; (B) en bas à gauche ; (C) en haut à droite.
i) Vrai ou faux : si l'on multiplie les activations $A^1$ et $A^2$ par 10, sans toucher aux gradients, la carte normalisée change.
j) La carte de 3 × 3 est agrandie à une image de 72 × 72 pixels. Une convention d'alignement (`align_corners`) différente pour la carte et pour l'image décale la carte d'un demi-pixel de la couche : combien de pixels de l'image ?

Dans ta copie : dessine les deux cartes (classes $k$ et $k'$). Que fait le canal 2 pour la classe $k$ ? Pourquoi multiplier les activations par 10 ne change-t-il rien après normalisation, mais changerait la carte non normalisée ? Comment s'écrirait `A` pour ta `grad_cam` (sa forme) ?

### Ex B6.5 — Audit à la main : deux groupes, deux matrices de confusion, cinq indicateurs ✏️ ★★ ⏱️ 20 min
**Objectif :** calculer à la main les taux d'un audit par groupe, les écarts de parité démographique et d'equalized odds et la règle des quatre cinquièmes, et voir ce qu'un seuil par groupe changerait.
**Prérequis :** ch. 3 (matrice de confusion, TPR, FPR, precision) · fiche §206.9 (taux par groupe, critères, $\Delta_{\text{DP}}$, $\Delta_{\text{EO}}$, règle des quatre cinquièmes), §206.10 (seuils par groupe) · **Parcours :** R, M

Un modèle présélectionne des candidatures pour un entretien ($\hat{y} = 1$) ; $y = 1$ quand un jury indépendant, qui a vu tous les dossiers, juge le candidat qualifié. Les matrices de confusion des femmes (F) et des hommes (H) :

| Groupe | TP | FN | FP | TN |
|---|---|---|---|---|
| F | 72 | 48 | 19 | 361 |
| H | 256 | 64 | 40 | 280 |

a) Le taux de base des femmes, à 2 décimales.
b) Le taux de sélection des femmes, à 3 décimales.
c) Le TPR des femmes, à 2 décimales.
d) Le FPR des hommes, à 3 décimales.
e) La precision des femmes, à 3 décimales.
f) L'écart de parité démographique $\Delta_{\text{DP}}$, à 2 décimales.
g) L'écart d'equalized odds $\Delta_{\text{EO}}$, à 2 décimales.
h) Le rapport du taux de sélection des femmes à celui des hommes, à 2 décimales.
i) Vrai ou faux : la règle des quatre cinquièmes est respectée.
j) Vrai ou faux : les accuracies des deux groupes diffèrent de moins de 5 points.
k) Pour donner aux femmes le TPR des hommes, l'entreprise baisse le seuil de leur score. Combien de candidates qualifiées de plus seront alors sélectionnées ?

Dans ta copie : quel critère te semble le plus pertinent pour une présélection d'entretiens, et pourquoi ? Que coûterait k) en faux positifs (le FPR des femmes va-t-il monter) ? Les effectifs suffisent-ils pour conclure sans intervalle ? Que t'apprend la comparaison de a) avec le taux de base des hommes ?

### Ex B6.6 — Impossibilité : taux de base différents, calibration et égalité des erreurs incompatibles ∂ ★★★ ⏱️ 40 min
**Objectif :** démontrer la relation de Chouldechova, puis en déduire que parité prédictive et égalité des taux d'erreur sont incompatibles quand les taux de base diffèrent.
**Prérequis :** Ex B6.5 · ch. 4 (règle de Bayes) · fiche §206.9 (indicateurs, l'impossibilité) · **Parcours :** M

Dans un groupe de $n$ personnes, de taux de base $p$ (avec $0 < p < 1$), un classifieur a un TPR, un FPR, un taux de faux négatifs $\operatorname{FNR} = 1 - \operatorname{TPR}$ et une precision $\operatorname{PPV} = \frac{TP}{TP + FP}$. On admet les définitions du tableau de 206.9 et la règle de Bayes du ch. 4. Réponds dans ta copie.

1. **Les effectifs.** Combien de positifs réels et de négatifs réels dans le groupe ? Écris $TP$ et $FN$ avec $p$, $n$ et le FNR.
2. **Les faux positifs.** À partir de la définition de la PPV, écris $FP$ en fonction de $TP$ et de la PPV (on suppose $TP > 0$).
3. **La relation.** Divise $FP$ par le nombre de négatifs réels et montre que $\operatorname{FPR} = \frac{p}{1 - p}\cdot\frac{1 - \operatorname{PPV}}{\operatorname{PPV}}\cdot(1 - \operatorname{FNR})$.
4. **Par Bayes.** Écris $\operatorname{PPV} = P(y = 1 \mid \hat{y} = 1)$ par la règle de Bayes, avec $p$, le TPR et le FPR, puis retrouve la relation en isolant le FPR.
5. **Contrôle.** Vérifie la relation sur les deux groupes de ✏️ B6.5 : calcule le membre de droite avec leurs $p$, PPV et FNR, et compare au FPR de chaque groupe.
6. **Parité prédictive et même FNR.** Deux groupes ont la même PPV et le même FNR. Montre que le rapport de leurs FPR vaut le rapport de leurs **cotes** $\frac{p}{1 - p}$. Application : PPV = 0,8, FNR = 0,3 dans les deux groupes, taux de base 0,24 et 0,5 (ceux de ✏️ B6.5). Calcule les deux FPR.
7. **Equalized odds.** Deux groupes ont maintenant le même FPR et le même FNR. Écris la PPV en fonction de $p$, du FPR et du FNR, et montre qu'elle croît avec $p$ (on suppose FPR > 0). Application : FPR = 0,1, FNR = 0,2, taux de base 0,24 et 0,5.
8. **Les issues.** Pour quels taux de base, ou quel classifieur, les trois égalités (PPV, FNR, FPR) peuvent-elles tenir ensemble ? Que devient la relation de 3 pour un classifieur parfait ?
9. **COMPAS.** Que mesurait l'éditeur du score, que mesuraient les journalistes (206.8), et pourquoi la relation de 3 dit-elle qu'ils ne pouvaient pas être d'accord ? Qu'ajoute le résultat de Kleinberg et coll. (206.9), qui porte sur des scores et non sur des décisions ?

### Ex B6.7 — Fermi : coût de Shapley exact, de KernelSHAP et de TreeSHAP pour 10, 20 et 50 features 🧮 ★★ ⏱️ 15 min
**Objectif :** chiffrer en ordres de grandeur ce que coûtent les trois façons de calculer des valeurs de Shapley, et en tirer quand utiliser chacune.
**Prérequis :** Ex B6.1 · fiche §206.3 (« Combien ça coûte »), §206.4 (KernelSHAP, TreeSHAP) · **Parcours :** M

Hypothèses (des ordres de grandeur) : un gradient boosting de $T = 100$ arbres d'au plus $L = 31$ feuilles prédit environ 150 000 lignes par seconde sur les deux cœurs du workbook ; le fond a $B = 50$ lignes ; KernelSHAP de shap 0.54.1 tire par défaut $2M + 2\,048$ coalitions, mais jamais plus que les $2^M - 2$ coalitions qui ne sont ni vide ni pleine (il les prend alors toutes), chacune évaluée sur les $B$ lignes du fond ; TreeSHAP interventionnel fait de l'ordre de $T \times L \times B$ pas élémentaires par ligne expliquée, à une dizaine de nanosecondes chacun. Réponds dans ta copie, avec des ordres de grandeur ; des fourchettes sont bienvenues.

1. **Shapley exact.** Pour $M = 10$, 20 et 50 : combien de prédictions par ligne expliquée, et combien de temps ? Exprime le cas $M = 50$ dans une unité parlante. Et pour expliquer 1 000 lignes (un beeswarm) ?
2. **Le seuil.** À partir de quel $M$ le calcul exact d'une seule ligne dépasse-t-il une minute ?
3. **KernelSHAP.** Pour les trois valeurs de $M$ : combien de coalitions, de prédictions, de temps par ligne ? Quelle part des $2^M$ coalitions voit-il pour $M = 20$ et $M = 50$ ? Qu'en déduis-tu de la précision de ses valeurs ? Pour $M = 10$, combien de coalitions KernelSHAP évalue-t-il ? Compare son coût à celui du calcul exact : lequel choisirais-tu, et pourquoi ?
4. **TreeSHAP.** Combien de pas élémentaires par ligne, et de temps ? Où $M$ apparaît-il dans ce coût ? Combien de temps pour 1 000 lignes ?
5. **Le piège des permutations.** Une autre écriture de Shapley fait la moyenne sur les $M!$ ordres d'arrivée. Combien d'ordres pour $M = 10$ et $M = 12$ ? Si l'on range chaque ordre dans un tuple Python d'environ 140 octets, combien de mémoire pour $M = 12$ ? Compare à la mémoire d'une session Colab gratuite (une douzaine de Go).
6. **Le notebook.** Ta `shapley_values` de 🔨 B6.14 explique un district de California avec 8 features et un fond de 50 lignes. Pour expliquer 20 districts de la même façon, combien de prédictions, et de temps ?
7. Dis, en une phrase chacune, quand utiliser le calcul exact, KernelSHAP et TreeSHAP.

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex B6.8 — Expliquer une valeur SHAP à un client dont le prêt est refusé, en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** traduire un waterfall SHAP en une explication honnête et utile pour une personne qui n'est pas du métier.
**Prérequis :** fiche §206.3, §206.4 (waterfall, valeur de base, explication contrefactuelle), §206.11 (RGPD, AI Act art. 86) · **Parcours :** R

Un modèle estime la probabilité de défaut d'un demandeur ; un dossier est refusé au-dessus de 0,30. Pour Mme D., le waterfall donne : valeur de base (probabilité moyenne de défaut sur le fond, les clients de l'an dernier) 0,12 ; taux d'endettement de 42 % : +0,17 ; ancienneté dans l'emploi de 4 mois : +0,09 ; deux incidents de paiement en 2025 : +0,06 ; revenu de 2 400 € par mois : −0,03 ; apport de 15 % : 0,00 ; prédiction 0,41.

Explique-lui en cinq lignes au plus, sans jargon (ni « SHAP », ni « valeur de base », ni « feature »), pourquoi son dossier a été refusé. Ton explication doit faire apparaître (1) la décision et ce qu'est le score, en mots simples ; (2) les deux ou trois éléments qui ont le plus pesé, et dans quel sens ; (3) ce qui aurait pu changer la décision, sans promettre ce que le modèle ne garantit pas ; (4) que ces éléments décrivent le raisonnement du modèle, pas un jugement sur elle ; (5) son droit de demander un réexamen par une personne et de présenter ses observations.

### Ex B6.9 — Scoring de crédit : le code postal est-il une variable de substitution ? Droits du client, obligations de l'entreprise ⚖️ ★★★ ⏱️ 35 min
**Objectif :** analyser un scoring de crédit qui utilise le lieu de résidence, sous l'angle technique (variable de substitution, audit) et réglementaire (droit de la non-discrimination, RGPD, AI Act), et proposer un plan d'action.
**Prérequis :** ch. 29 (données personnelles, data card) · MP3 (audit par sous-groupe, équité par aveuglement) · fiche §206.8 (biais, variables de substitution), §206.9, §206.10, §206.11 (RGPD art. 22, SCHUFA, Dun & Bradstreet, AI Act) · **Parcours :** R

Une fintech française accorde en ligne des prêts personnels de 1 000 à 15 000 €. Son modèle, un gradient boosting, utilise les revenus et charges déclarés, l'ancienneté dans l'emploi, les incidents de paiement, l'âge, la situation familiale et les **deux premiers chiffres du code postal** (en métropole, le département de résidence). La décision est prise par le modèle seul, en une minute ; un refus s'affiche à l'écran avec la mention « Votre dossier ne correspond pas à nos critères ». La fintech a développé le modèle elle-même et l'utilise.

Un audit interne trouve que le département est la 3ᵉ feature par moyenne des $\lvert\phi_i\rvert$, et que le taux d'acceptation des demandeurs domiciliés dans trois départements à forte population immigrée est de 41 %, contre 63 % ailleurs ; à revenus, charges et incidents comparables (même tranche), l'écart reste de 9 points. Retirer le département fait passer l'AUC de 0,781 à 0,774. (Chiffres inventés pour l'exercice.) Réponds dans ta copie.

1. **Variable de substitution ?** Le département est-il une variable de substitution, et de quoi ? Comment le mesurer sans collecter l'origine des clients ? Que permettrait l'article 4 bis créé par le règlement 2026/1744, et à quelles conditions ?
2. **Le droit de la non-discrimination.** Le code pénal (art. 225-1 et 225-2) fait du « lieu de résidence » un critère de discrimination, y compris pour le refus de fournir un service. Qu'est-ce que cela change pour l'usage du département ? Que recommande la CNIL sur ce point (recommandation sur l'évaluation de la solvabilité, mai 2026) ?
3. **Le RGPD.** La décision relève-t-elle de l'article 22 ? Sur quelle exception la fintech peut-elle s'appuyer, et quels droits doit-elle alors garantir ? Que changerait l'achat du score à une agence externe (arrêt SCHUFA) ? Que doit contenir une réponse à une demande d'accès (art. 15, arrêt Dun & Bradstreet), et que faire si la fintech invoque le secret des affaires ?
4. **L'AI Act.** Dans quelle catégorie tombe ce système ? La fintech est-elle fournisseur, déployeur ou les deux ? Cite quatre obligations, avec leurs articles, et la date à laquelle elles s'appliqueront. Que dit l'article 86 à la personne refusée ?
5. **Le plan technique.** Quel audit (groupes, indicateurs, intervalles) mener avant et après le retrait du département ? Quelles variables pourraient porter la même information, et comment le vérifier ? Le coût de 0,007 d'AUC est-il un argument pour garder la variable ?
6. **Le message de refus.** Réécris le message affiché au client refusé, en quatre lignes au plus, pour qu'il respecte ce que tu as dit en 3 et 4.
7. Rédige les cinq lignes « Données et équité » de la fiche modèle.

### Ex B6.10 — Reconnaissance faciale : erreurs par sous-groupe et décision de déploiement ⚖️ ★★ ⏱️ 25 min
**Objectif :** lire des taux d'erreur par sous-groupe, voir ce qu'une moyenne cache selon la population, et argumenter une décision de déploiement.
**Prérequis :** Ex B6.5 · fiche §206.8 (Gender Shades, analyse intersectionnelle), §206.9, §206.11 · **Parcours :** R

**Partie 1 : relire Gender Shades.** Dans l'article de J. Buolamwini et T. Gebru (2018), le classifieur de genre de Microsoft a, sur les 1 270 visages du benchmark PPB, les taux d'erreur suivants (tableau 4 de l'article) : 6,3 % en tout ; 20,8 % pour les femmes à peau foncée (DF), 6,0 % pour les hommes à peau foncée (DM), 1,7 % pour les femmes à peau claire (LF), 0,0 % pour les hommes à peau claire (LM). Le benchmark compte 21,3 % de DF, 25,0 % de DM, 23,3 % de LF et 30,3 % de LM (tableau 2).

1. Recalcule l'erreur totale à partir des quatre sous-groupes. Que se passerait-il pour un benchmark composé à 80 % de LM et à 20 % des trois autres sous-groupes à parts égales ? Et pour une population à 50 % de DF, les trois autres sous-groupes gardant entre eux leurs proportions du benchmark ?
2. Les erreurs par sexe seul (10,7 % et 2,6 %) ou par couleur de peau seule (12,9 % et 0,7 %) auraient-elles suffi à voir le problème ? Qu'apporte l'analyse intersectionnelle, et quel est son prix (effectifs, comparaisons multiples) ?

**Partie 2 : une décision de déploiement.** Une banque en ligne veut vérifier l'identité de ses nouveaux clients en comparant un selfie à la photo de leur pièce d'identité (une **vérification** 1:1). Le fournisseur annonce, au seuil choisi, un taux de faux rejets (un vrai client refusé) de 0,5 % pour LM, 1,0 % pour LF, 1,5 % pour DM et 4,0 % pour DF, et un taux de fausses acceptations (un imposteur accepté) de 0,01 %. La banque attend 200 000 ouvertures de compte par an, dont 10 % de DF. (Chiffres inventés pour l'exercice.)

3. Combien de clientes DF seront rejetées à tort chaque année, et combien de fois plus souvent qu'un client LM ? Qui paie un faux rejet, qui paie une fausse acceptation ?
4. La banque peut : (a) baisser le seuil pour tout le monde ; (b) prendre un seuil par sous-groupe ; (c) envoyer tout rejet vers une vérification par un conseiller, sous 24 h ; (d) exiger du fournisseur un modèle réentraîné et un rapport par sous-groupe ; (e) proposer une autre voie sans biométrie. Discute chaque option (erreurs, équité, faisabilité, droit).
5. **Le cadre juridique.** La vérification 1:1 est-elle un système à haut risque au titre du point 1 de l'annexe III de l'AI Act ? Quelles données traite-t-elle au sens de l'article 9 du RGPD, et qu'est-ce que cela impose ? Un système qui **classerait** les clients par genre ou par origine à partir de leur visage serait-il permis (art. 5, paragraphe 1, g) et annexe III, point 1 b)) ?
6. Recommande une décision, en cinq lignes, avec la mesure de suivi que tu exiges après le déploiement.

### Ex B6.11 — Lundberg et Lee (2017) : « A Unified Approach to Interpreting Model Predictions » 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a fondé SHAP : sa définition des explications additives, ses trois propriétés, ses approximations et ses expériences.
**Prérequis :** ∂ B6.2 · fiche §206.3, §206.4 · **Parcours :** M

Lis S. Lundberg et S.-I. Lee, « A Unified Approach to Interpreting Model Predictions » ([arXiv:1705.07874](https://arxiv.org/abs/1705.07874v2), version 2 de novembre 2017, NIPS 2017) : le résumé, les sections 1 à 4 et les figures 1, 2 et 4, puis la section 5. Réponds dans ta copie.

1. **Le problème** (résumé, section 1). Quelle tension les auteurs veulent-ils résoudre ? Quelles sont les deux « composantes nouvelles » annoncées dans le résumé ?
2. **La définition** (section 2, définition 1, équation 1). Que sont $z'$ et l'application $h_x$ ? Fais le lien avec les coalitions et la valeur de base de la fiche : que vaut $z'$ pour la coalition $\{1, 3\}$ de ✏️ B6.1, et que vaut $g$ en $z' = (0, 0, 0)$ ?
3. **Les six méthodes** (sections 2.1 à 2.4). Nomme-les. Pour chacune, comment représente-t-elle une feature « absente » ? Laquelle demande de **réentraîner** un modèle par sous-ensemble de features, et combien de modèles pour $M = 10$ ?
4. **Les trois propriétés** (section 3, équations 5 à 7, théorème 1). Traduis la précision locale, l'absence et la cohérence en mots simples. Laquelle correspond à l'efficacité de la fiche ? Pourquoi les auteurs disent-ils que la propriété d'absence est nécessaire ici, alors que la théorie des jeux classique s'en passe ?
5. **Les valeurs SHAP** (section 4, équations 9 à 12). Quelle hypothèse fait passer de l'équation 10 à l'équation 11, et laquelle de 11 à 12 ? Laquelle correspond au $v(S)$ de la fiche et de ta `shapley_values` ? Montre avec ✏️ B6.1 que 11 et 12 donnent des valeurs de base différentes, et avec ∂ B6.2 qu'elles coïncident pour un modèle linéaire.
6. **Kernel SHAP** (théorème 2). Calcule le noyau de Shapley $\pi_x(z')$ pour $M = 4$ et $|z'| = 1, 2, 3$. Que vaut-il pour $|z'| = 0$ et $M$, et comment les auteurs le gèrent-ils ? Combien de vecteurs $z'$ la figure 2A range-t-elle, et pour quel $M$ ?
7. **Linear SHAP** (corollaire 1). Compare à ∂ B6.2. Relève l'incohérence de notation dans la formule.
8. **Deux jeux de la figure 4.** (A) Un score de maladie vaut 2 quand la fièvre et la toux sont toutes deux présentes, 5 quand une seule l'est, 0 sinon. Calcule les valeurs de Shapley de la fièvre et de la toux pour un patient qui a les deux, la valeur de base étant le cas « aucun symptôme ». (B) Trois hommes gagnent autant de dollars que le meilleur score obtenu par l'un d'eux ; ils ont répondu juste à 5, 4 et 0 questions. Calcule les trois valeurs de Shapley, la coalition vide valant 0. Compare tes résultats aux réponses des participants humains de la figure.
9. **Les expériences** (section 5). Que montre la figure 3 sur le nombre d'évaluations du modèle ? Combien d'échantillons KernelSHAP utilise-t-il pour la figure 5, et quelle part des pixels est masquée ?
10. **Et aujourd'hui** (fiche §206.4). Donne deux points que l'article laisse ouverts ou que l'usage de 2026 fait autrement (le choix entre espérance conditionnelle et fond indépendant, les arbres, la bibliothèque).

<a id="entretien"></a>

## 💼 Entretien

Réponds à voix haute en une minute, puis compare avec `05_solutions.md` : réponse modèle et relances possibles.

### B6.E1 — Comment expliquer les prédictions d'un gradient boosting à une équipe métier ? 💼 ★★ ⏱️ 10 min
*Fiche §206.2, §206.4 · parcours R*

« Notre modèle de churn est un gradient boosting. L'équipe commerciale ne lui fait pas confiance parce qu'elle ne comprend pas ses prédictions. Que lui montrez-vous ? »

### B6.E2 — Qu'est-ce qu'une valeur SHAP, et quelles sont ses limites ? 💼 ★★ ⏱️ 10 min
*Fiche §206.3, §206.4, §206.7 · parcours R*

« Vous citez SHAP dans votre CV. Expliquez-moi ce qu'est une valeur SHAP, et dans quels cas vous vous en méfieriez. »

### B6.E3 — Comment audites-tu un modèle pour détecter des biais ? 💼 ★★ ⏱️ 10 min
*Fiche §206.8, §206.9 · parcours R*

« On vous confie un modèle de présélection de CV déjà en production. Comment vérifiez-vous qu'il ne discrimine pas ? »

### B6.E4 — Grad-CAM : principe, usage et limites 💼 ★★ ⏱️ 10 min
*Fiche §206.6, §206.7 · parcours R*

« Votre classifieur d'images médicales atteint 95 % d'accuracy. Un radiologue demande où le modèle regarde. Que lui proposez-vous, et avec quelles précautions ? »

### B6.E5 — Retirer la variable sensible suffit-il à rendre un modèle équitable ? 💼 ★★ ⏱️ 10 min
*Fiche §206.8, §206.10 · parcours R*

« Pour éviter toute discrimination, notre équipe a retiré le sexe et l'âge des features. Le modèle est donc équitable, non ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`, en trois phases, de difficulté croissante dans chacune : **les modèles tabulaires** (B6.12 à B6.17 : explications globales sur California, Shapley exact avec ta `shapley_values`, shap sur un gradient boosting, le rôle du fond, Penguins), **les réseaux** (B6.18 à B6.22 : `integrated_gradients` et `grad_cam`, attributions sur un CNN CIFAR-10, Grad-CAM par hooks sur le ResNet fine-tuné du B1, un Grad-CAM à réparer, le test de santé), **l'équité** (B6.23 à B6.29 : tes indicateurs par groupe, l'audit d'*Adult* avec intervalles, l'atténuation et son coût, la fiche modèle et le défi).

| ID | Titre | Type | ★ | ⏱️ | 🚀 |
|---|---|---|---|---|---|
| B6.12 | Explications globales sur California : coefficients, permutation, dépendance partielle et ICE | 📦 | ★★ | 30 | |
| B6.13 | Prédire la forme des courbes ICE du revenu médian et de la latitude | 🔮 | ★ | 10 | |
| B6.14 | shapley_values : Shapley exact par énumération des coalitions | 🔨 | ★★★ | 45 | |
| B6.15 | SHAP sur un gradient boosting California : TreeExplainer, waterfall, beeswarm, dépendance | 📦 | ★★ | 30 | |
| B6.16 | Le fond change tout : trois fonds différents, puis une feature dupliquée | 🔬 | ★★★ | 40 | |
| B6.17 | Penguins : SHAP par classe pour une régression logistique et une forêt aléatoire | 📦 | ★★ | 25 | |
| B6.18 | integrated_gradients et grad_cam | 🔨 | ★★ | 30 | |
| B6.19 | Saillance, gradient × entrée et gradients intégrés sur un CNN CIFAR-10 | 📦 | ★★ | 30 | |
| B6.20 | Grad-CAM par hooks sur le ResNet fine-tuné du B1 : bonnes prédictions et erreurs | 📦 | ★★★ | 45 | |
| B6.21 | Grad-CAM à l'envers : mauvaise couche, mauvaise classe, ReLU oubliée, carte mal alignée | 🐛 | ★★ | 25 | |
| B6.22 | Test de santé d'Adebayo : randomiser les poids couche par couche, la carte change-t-elle ? | 🔬 | ★★★ | 40 | |
| B6.23 | group_rates, demographic_parity_difference et equalized_odds_difference | 🔨 | ★★ | 25 | |
| B6.24 | Audit d'équité sur Adult : indicateurs par sexe et par âge, intervalles de confiance bootstrap | 📦 | ★★★ | 45 | |
| B6.25 | Lire un tableau d'audit : où est l'écart, est-il significatif, que conclure ? | 📈 | ★★ | 15 | |
| B6.26 | Prédire : retirer la variable « sexe » rend-il le modèle équitable ? | 🔮 | ★ | 10 | |
| B6.27 | Atténuer : retrait des variables de substitution, repondération, seuils par groupe, et ce que cela coûte | 🔬 | ★★★ | 50 | |
| B6.28 | Fiche modèle avec sections explicabilité et équité : indicateurs par groupe, limites, usage prévu | 🛠️ | ★★ | 25 | |
| B6.29 | Défi : écart d'equalized odds sous 0,07 sur Adult, en perdant moins de 2 points d'accuracy | 🏆 | ★★★★ | 90 | |

Le notebook utilise California, Penguins, CIFAR-10 et *Adult* (téléchargés aux chapitres précédents, ou les copies du dépôt) et ton ResNet fine-tuné du B1 s'il existe, sinon un petit CNN entraîné en une minute environ. Tout tourne sur CPU ; aucun exercice n'est marqué 🚀.
