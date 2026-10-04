# 1 · Introduction au machine learning et au deep learning — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch01_introduction/06_mes_reponses.md` (créée par `python tools/start_chapter.py 1`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses des exercices ✏️ se vérifient dans la **partie 0** du notebook (`wb.check`) ; les quiz, les rappels, les exercices 🧮 🗣️ ⚖️ 📄 et les questions d'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ Papier-crayon](#papier) · [🧮 🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 ou 4 minutes chacun. Réponds vite, puis vérifie dans `05_solutions.md`.

### 1.Q1 — Règles écrites ou règles apprises ? 🧠 ⏱️ 3 min
*Fiche §1.1 · livre §1.1, §1.1.1 · parcours R*

Pour chaque programme, dis s'il applique des règles **écrites** par un humain (É) ou des règles **apprises** à partir d'exemples (A).
1. Un tableur calcule la TVA de chaque facture.
2. Un filtre anti-spam s'améliore chaque fois que tu marques un e-mail comme indésirable.
3. Un thermostat allume le chauffage dès que la température passe sous 19 °C.
4. Une application de photos regroupe automatiquement les visages de tes proches.
5. Un correcteur orthographique signale tout mot absent de son dictionnaire.

### 1.Q2 — Pourquoi les systèmes experts ont calé 🧠 ⏱️ 3 min
*Fiche §1.1.2 · livre §1.1.2 · parcours R*

Vrai ou faux ?
1. Un système expert apprend ses règles à partir d'exemples étiquetés.
2. Le *feature engineering* consiste à fabriquer à la main les indices que le programme doit surveiller.
3. Pour reconnaître les 7, ajouter une règle pour le 7 barré suffit à couvrir tous les cas.
4. Le savoir-faire d'un expert s'écrit toujours facilement sous forme de règles.
5. Le machine learning évite d'écrire les règles, mais il demande beaucoup de données.

### 1.Q3 — Échantillon, feature ou label ? 🧠 ⏱️ 3 min
*Fiche §1.2.2 · livre §1.2, §1.2.2 · parcours R*

Un site d'annonces veut prédire le **loyer** d'un appartement. Son tableau contient une ligne par appartement, avec les colonnes : surface (m²), nombre de pièces, étage, quartier, loyer (€).
1. Qu'est-ce qu'un échantillon, ici ?
2. Quelles sont les features ?
3. Quel est le label ?
4. D'où vient le label de chaque échantillon ?
5. Si l'on voulait plutôt prédire le **quartier** d'un appartement à partir des autres colonnes, qu'est-ce qui changerait ?

### 1.Q4 — L'école absurde : ce qui marche pour une machine 🧠 ⏱️ 3 min
*Fiche §1.2.1 · livre §1.2.1 · parcours R*

1. Dans l'école imaginée par le livre, que mesure le premier test du vendredi ? Et le second ?
2. Pourquoi cette méthode serait-elle mauvaise pour des enfants, mais convient-elle à un ordinateur ?
3. Vrai ou faux : un modèle qui a 100 % de bonnes réponses sur les exemples qu'il a appris généralisera bien.
4. À quoi correspondent, en machine learning, le premier et le second test ?

### 1.Q5 — Paramètre ou hyperparamètre ? 🧠 ⏱️ 3 min
*Fiche §1.2.4 · livre §1.2.4 · parcours R*

Pour chaque nombre, dis s'il est un **paramètre** (appris par l'algorithme) ou un **hyperparamètre** (choisi par toi).
1. Les poids d'un réseau de neurones.
2. Le learning rate.
3. Le nombre d'epochs.
4. Le seuil « bec ≤ 40 mm » trouvé par un arbre de décision pendant son entraînement.
5. La profondeur maximale d'un arbre (`max_depth`).
6. La pente $w$ d'une droite de régression entraînée.

### 1.Q6 — À quoi sert le jeu de test 🧠 ⏱️ 3 min
*Fiche §1.2.3 · livre §1.2.3 · parcours R*

1. Pourquoi met-on le jeu de test de côté **avant** d'entraîner le modèle ?
2. Pendant l'évaluation sur le jeu de test, le modèle modifie-t-il ses paramètres ?
3. Un modèle fait 98 % d'accuracy sur l'entraînement et 70 % sur le test. Que conclus-tu ?
4. Qu'est-ce qu'une epoch ?
5. Vrai ou faux : on peut essayer plusieurs learning rates, garder celui qui donne le meilleur score **sur le jeu de test**, puis annoncer ce score comme la performance du modèle.

### 1.Q7 — Classification ou régression : six situations 🧠 ⏱️ 4 min
*Fiche §1.3 · livre §1.3, §1.3.1, §1.3.2 · parcours R*

Classification (C) ou régression (R) ?
1. Prédire le prix de vente d'une maison.
2. Dire si un e-mail est un spam.
3. Prévoir la température de demain à midi.
4. Reconnaître l'espèce d'une fleur sur une photo.
5. Estimer l'âge, en années, d'une personne sur une photo.
6. Lire le code postal manuscrit d'une enveloppe.

### 1.Q8 — Clustering, débruitage ou réduction de dimension ? 🧠 ⏱️ 4 min
*Fiche §1.4 · livre §1.4 à §1.4.3 · parcours R*

Pour chaque tâche : clustering, débruitage ou réduction de dimension ?
1. Regrouper les clients d'un magasin selon leurs achats, sans catégories fixées à l'avance.
2. Retirer le souffle d'un vieil enregistrement.
3. Résumer les 300 mesures d'un capteur par 10 nombres qui en gardent l'essentiel.
4. Reconstituer les pixels effacés d'une photo abîmée.
5. Supprimer une colonne qui vaut toujours 0.
6. Pourquoi ces trois tâches sont-elles dites **non supervisées** ?

### 1.Q9 — Générateurs et renforcement : sans labels, mais pas sans retour 🧠 ⏱️ 3 min
*Fiche §1.5, §1.6 · livre §1.5, §1.6 · parcours R*

1. Que produit un générateur ?
2. Pourquoi le livre range-t-il les générateurs dans le « semi-supervisé » ? Comment le dirait-on aujourd'hui (fiche, 🕰️) ?
3. Un thermostat intelligent apprend à régler le chauffage selon tes réactions (tu montes ou baisses la consigne, ou tu ne touches à rien). Qui est l'agent ? l'environnement ? l'action ? la récompense ?
4. Quelle différence y a-t-il entre une récompense et un label ?
5. Pourquoi un agent doit-il parfois essayer une action qui n'est pas la meilleure qu'il connaisse ?

### 1.Q10 — Profond, capacité et GPU 🧠 ⏱️ 3 min
*Fiche §1.7, §1.2.4 · livre §1.7 · parcours R*

1. Qu'est-ce qui rend un réseau de neurones « profond » ?
2. Que calcule un neurone artificiel, en une phrase ?
3. Qu'appelle-t-on *feature learning* ?
4. Vrai ou faux : un modèle de plus grande capacité donne toujours un meilleur score sur le jeu de test.
5. Pourquoi les GPU accélèrent-ils l'entraînement des réseaux ?

### 1.Q11 — Panorama 2026 : où ranger ChatGPT et Stable Diffusion ? 🧠 ⏱️ 4 min
*Fiche §1.5, §1.7, « Panorama 2026 » · livre §1.5, §1.7, §1.8 · parcours R*

1. Pendant son pré-entraînement, un grand modèle de langage (LLM) apprend à prédire le token suivant d'un texte. Est-ce supervisé, non supervisé, auto-supervisé ou par renforcement ?
2. À quelle famille du chapitre appartient Stable Diffusion, qui produit des images à partir d'une phrase ?
3. Qu'est-ce qu'un *foundation model* ?
4. Qu'est-ce qu'un token ?
5. Vrai ou faux : le vocabulaire de ce chapitre (échantillon, loss, learning rate, généralisation) ne s'applique plus aux LLM.

<a id="rappels"></a>

## 🔁 Rappels

Deux notions de 0B et une de 0A, revues avec les yeux du ch. 1. Réponds sur papier, puis vérifie dans une cellule de notebook et dans `05_solutions.md`.

### 1.R1 — 0B : distance entre deux manchots vus comme des vecteurs 🔁 ★ ⏱️ 5 min
*Chapitre 0B (0B.8, 101.3.2) · fiche §1.2.2 · parcours R, M*

Trois manchots, décrits par (longueur du bec en mm ; épaisseur du bec en mm ; nageoire en mm ; masse en g) :
$\mathbf{p} = (39{,}1 ;\ 18{,}7 ;\ 181 ;\ 3\,750)$ (un Adélie), $\mathbf{a} = (39{,}5 ;\ 17{,}4 ;\ 186 ;\ 3\,800)$ (un autre Adélie), $\mathbf{c} = (52{,}7 ;\ 19{,}8 ;\ 197 ;\ 3\,725)$ (un Chinstrap).
1. Calcule les distances $\lVert \mathbf{p} - \mathbf{a} \rVert$ et $\lVert \mathbf{p} - \mathbf{c} \rVert$ (1 décimale).
2. Lequel des deux est le plus proche de $\mathbf{p}$ ? Est-ce surprenant ?
3. Recalcule les deux distances **sans la masse** (3 composantes).
4. Qu'en conclus-tu pour un algorithme qui regroupe les manchots « proches » (1.21) ?

### 1.R2 — 0A : compter les espèces avec `value_counts` 🔁 ★ ⏱️ 5 min
*Chapitre 0A (0A.4, 0A.57) · fiche §1.3.1 · parcours R, C*

`df` est le DataFrame des manchots (`wb.datasets.load_penguins()`).
1. Écris la ligne pandas qui compte les manchots de chaque espèce.
2. Qu'ajoute l'option `normalize=True` ?
3. Comment compter les manchots de chaque espèce **sur chaque île**, en une ligne ?
4. Les trois espèces n'ont pas le même nombre de manchots. Pourquoi dit-on alors que les classes sont **déséquilibrées**, et pourquoi faudra-t-il s'en souvenir en évaluant un classifieur ?

### 1.R3 — 0B : la droite qui passe par deux points 🔁 ★ ⏱️ 5 min
*Chapitre 0B (0B.6, 101.2.1) · fiche §1.3.2 · parcours R, M*

1. Donne la pente et l'ordonnée à l'origine de la droite qui passe par $A(2 ;\ 5)$ et $B(6 ;\ 13)$.
2. Que vaut-elle en $x = 10$ ?
3. Le point $C(4 ;\ 9)$ est-il sur cette droite ?
4. Écris en Python une fonction `line(x)` qui renvoie la valeur de cette droite en `x`.

<a id="papier"></a>

## ✏️ Papier-crayon

Fais-les **à la main** (calculatrice autorisée), en écrivant chaque étape dans ta copie de `06_mes_reponses.md`. Reporte ensuite **la valeur** trouvée dans la **partie 0 du notebook**, qui la vérifie avec `wb.check`. Arrondis comme indiqué ; sinon, donne la valeur exacte. Écris une liste comme `[7, -2]`.

### Ex 1.1 — Accuracy et erreurs à l'échelle d'un centre de tri ✏️ ★ ⏱️ 10 min
**Objectif :** passer d'une accuracy à des nombres d'erreurs concrets, pour un chiffre puis pour un code entier.
**Prérequis :** fiche §1.2.3 · 0B (indépendance, 101.7.2) · **Fil rouge :** MNIST · **Parcours :** R, M

Le petit réseau du livre (§1.7) reconnaît correctement 9 905 des 10 000 chiffres du jeu de test de MNIST.

a) Son accuracy, sous forme de proportion (4 décimales).
b) Son taux d'erreur, en pourcentage (2 décimales).
c) Un centre de tri lit 1 200 000 chiffres manuscrits par jour. Avec ce taux d'erreur, combien de chiffres sont mal lus chaque jour, en moyenne ?
d) Un code postal compte 5 chiffres. On suppose que les erreurs sur les chiffres sont **indépendantes** (0B). Quelle est la probabilité qu'un code soit lu **sans aucune** erreur (4 décimales) ?
e) Le centre trie 240 000 lettres par jour. Combien de codes postaux, en moyenne, contiennent au moins une erreur ? Arrondis à l'entier, **sans arrondir les résultats intermédiaires**.

### Ex 1.2 — Concerts : la valeur manquante et celle de demain ✏️ ★ ⏱️ 15 min
**Objectif :** estimer une valeur manquante et prévoir la suivante avec des droites, et voir que la prévision dépend de la méthode.
**Prérequis :** 0B (droites, 101.2.1) · Rappel 1.R3 · fiche §1.3.2 · **Parcours :** M

Une salle de concert a noté la fréquentation de chaque soir de mai, mais la feuille du 8 mai a été perdue :

| jour de mai | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| spectateurs | 1 200 | 1 270 | 1 290 | ? | 1 390 | 1 460 | 1 520 | 1 550 |

a) Estime le 8 mai par **interpolation linéaire** entre le 7 et le 9 (la valeur au milieu du segment qui les relie).
b) La pente, en spectateurs par jour, de la droite qui passe par le **premier** (5 mai) et le **dernier** point (12 mai).
c) La valeur de cette droite au 8 mai.
d) La prévision de cette droite pour le 13 mai.
e) La prévision pour le 13 mai si l'on prolonge seulement la tendance des **deux derniers jours** (11 et 12 mai).
f) Le groupe touche 10 % de la recette, et un billet coûte 25 €. Combien peut-il espérer toucher le 13 mai, avec la prévision de d) ?

Laquelle des deux prévisions du 13 mai choisirais-tu ? (réponds dans ta copie)

### Ex 1.3 — Compter les connexions d'un réseau en couches ✏️ ★ ⏱️ 10 min
**Objectif :** compter les paramètres d'un réseau de couches pleines, et voir comment leur nombre grandit.
**Prérequis :** fiche §1.7 · 0B (somme pondérée et biais, 0B.4) · **Fil rouge :** MNIST · **Parcours :** M

Dans une couche **pleine**, chaque neurone reçoit **toutes** les valeurs de la couche précédente : chaque connexion porte un poids. Chaque neurone a en plus un biais. Les entrées ne sont pas des neurones : elles n'ont pas de biais.

a) Le réseau de la figure 1.21 du livre : 4 entrées, une couche de 3 neurones, puis une couche de 2 neurones (les sorties). Combien de connexions (de poids) ?
b) Combien de paramètres en tout, poids et biais ?
c) Un réseau pour MNIST : 784 entrées (les pixels), une couche cachée de 128 neurones, 10 sorties. Combien de paramètres ?
d) Même réseau avec 256 neurones cachés au lieu de 128 ?
e) Et avec **deux** couches cachées de 128 neurones (784 → 128 → 128 → 10) ?

Qu'est-ce qui coûte le plus de paramètres : doubler la largeur de la couche cachée, ou ajouter une seconde couche ? (réponds dans ta copie)

### Ex 1.4 — Moins de nombres pour dire la même chose ✏️ ★★ ⏱️ 15 min
**Objectif :** repérer les features inutiles ou redondantes, et décrire une position avec moins de nombres.
**Prérequis :** fiche §1.4.3 · 0B (norme, produit scalaire, 101.3) · **Parcours :** M

Toutes les situations de cet exercice reprennent des exemples du livre (§1.4.3).

a) Chaque jour d'été, une station du désert enregistre la vitesse du vent, sa direction et la pluie tombée, qui vaut toujours 0. Combien de features apportent réellement une information ?
b) Une clinique note le poids de chaque patient en kilogrammes, puis en livres (1 kg = 2,2046 lb). Combien de livres pour un patient de 68 kg (1 décimale) ?
c) Combien d'informations **indépendantes** contient le couple (poids en kg, poids en lb) ?

Une route rectiligne à une seule voie va du point $(0 ;\ 0)$ au point $(600 ;\ 800)$ d'une carte (en mètres).
d) Une voiture est sur la route, au point $P(240 ;\ 320)$. À quelle distance du départ se trouve-t-elle ? Ce seul nombre remplace les deux coordonnées.
e) Une autre voiture est sur la route, à 750 m du départ. Quelles sont ses coordonnées sur la carte, sous forme de liste `[x, y]` ?

La route est élargie à deux voies ; son axe central reste le même. Une voiture est au point $Q(250 ;\ 300)$, un peu à côté de l'axe.

> 🧮 **Rappel maths — projeter sur un axe** — Soit $\mathbf{u}$ le vecteur **unitaire** de l'axe (de norme 1, 0B). La position d'un point $\mathbf{q}$ le long de l'axe, c'est-à-dire la longueur de sa projection sur l'axe, vaut $\mathbf{q} \cdot \mathbf{u} = \lVert \mathbf{q} \rVert \cos \theta$, où $\theta$ est l'angle entre $\mathbf{q}$ et l'axe (0B, 101.3.3). Avec la distance $d$ entre le point et l'axe, on a un triangle rectangle : $\lVert \mathbf{q} \rVert^2 = (\mathbf{q} \cdot \mathbf{u})^2 + d^2$ (Pythagore).
f) Sa position **le long** de la route (la projection de $Q$ sur l'axe, rappel ci-dessus).
g) La distance entre $Q$ et l'axe : l'information que l'on perd en ne gardant que la réponse f.

<a id="reflexion"></a>

## 🧮 🗣️ ⚖️ 📄 Réflexion

### Ex 1.5 — Fermi : combien coûte l'étiquetage de MNIST ? 🧮 ★★ ⏱️ 15 min
**Objectif :** estimer par un ordre de grandeur le coût humain de l'étiquetage d'un dataset.
**Prérequis :** fiche §1.1.1, §1.2.2 · **Fil rouge :** MNIST · **Parcours :** M

On veut faire étiqueter **aujourd'hui** les 70 000 images de MNIST par des annotateurs payés. Pas besoin de chiffres exacts : des hypothèses raisonnables et un calcul d'ordre de grandeur (une estimation « de Fermi »).
1. Combien de secondes faut-il pour regarder un chiffre et taper son label ? Combien d'heures de travail pour tout le dataset ?
2. Pour limiter les erreurs, chaque image est étiquetée par **deux** personnes, et une troisième tranche les désaccords (disons 2 % des images). Combien d'heures au total ?
3. Avec un coût horaire de ton choix (justifie-le), combien coûte l'étiquetage ?
4. ImageNet, le dataset qui a lancé l'essor du deep learning en vision, compte plus de 14 millions d'images, étiquetées avec l'aide de la plateforme de micro-travail Amazon Mechanical Turk. Combien de fois plus d'images que MNIST ? Pourquoi une image d'ImageNet prend-elle plus de temps qu'un chiffre ?
5. Pourquoi l'apprentissage **auto-supervisé** (fiche, 🕰️), qui se passe de labels humains, a-t-il permis d'entraîner des modèles sur des milliards de textes ?

### Ex 1.6 — Le machine learning en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer le machine learning simplement, sans jargon.
**Prérequis :** fiche §1.1, §1.2 · **Parcours :** R

Explique à quelqu'un qui n'a jamais programmé ce qu'est le machine learning, en **cinq lignes au plus**. Contraintes :
- un exemple concret de la vie courante ;
- les mots « exemples », « erreur » et « nouveau » ;
- aucun mot technique qui ne soit expliqué.

Relis-toi ensuite à voix haute : un collégien comprendrait-il ? Puis compare avec la réponse modèle de `05_solutions.md`.

### Ex 1.7 — Reconnaissance faciale : utile, risquée, encadrée ⚖️ ★★ ⏱️ 20 min
**Objectif :** peser les bénéfices, les risques et le cadre légal d'une application de machine learning.
**Prérequis :** fiche §1.1.1, encadrés ⚖️ et 🕰️ sur la reconnaissance faciale · **Parcours :** R

Deux projets arrivent sur ton bureau :
- **A.** Une salle de sport veut remplacer les badges d'entrée de ses abonnés par la reconnaissance de leur visage.
- **B.** Un supermarché veut repérer automatiquement, à l'entrée, les personnes déjà surprises en train de voler.

1. Quels bénéfices chaque projet apporte-t-il, et à qui ?
2. Quels risques ? Pense aux erreurs du modèle (dans les deux sens : reconnaître à tort, ne pas reconnaître), aux personnes pour qui il pourrait se tromper plus souvent, à ce qui se passe si la base de visages est volée (un visage ne se change pas comme un mot de passe), et à la surveillance.
3. Quelles règles s'appliquent en France et dans l'Union européenne (fiche : RGPD, AI Act) ? Pour chaque projet, qui doit donner son accord, et peut-on vraiment le demander ?
4. Tu es l'ingénieur·e ML chargé·e du projet B. Que fais-tu avant d'écrire la moindre ligne de code ? Propose au moins une alternative sans biométrie.

### Ex 1.8 — Galton (1886) : l'origine du mot « régression » 📄 ★★ ⏱️ 25 min
**Objectif :** lire un article fondateur, retrouver sa règle chiffrée et la relier à la régression linéaire.
**Prérequis :** Ex 1.2 · fiche §1.3.2 · livre §1.3.2 · **Parcours :** complet seulement (lecture conseillée à tous)

Lis l'introduction de l'article de Francis Galton, « Regression Towards Mediocrity in Hereditary Stature » (*Journal of the Anthropological Institute*, vol. 15, 1886, p. 246-263), [disponible gratuitement sur galton.org](https://galton.org/essays/1880-1889/galton-1886-jaigi-regression-stature.pdf), et regarde son tableau I (tailles des enfants selon celle des parents). Pas besoin de tout comprendre : l'anglais est ancien et les calculs se font à la main.

Pour lire le tableau, il faut une définition que Galton donne à peine : la taille **mi-parentale** (*mid-parent*) d'un enfant est la moyenne de la taille de son père et de celle de sa mère, après avoir multiplié la taille de la mère par 1,08 : $\frac{\text{père} + 1{,}08 \times \text{mère}}{2}$.

1. Quelles données Galton a-t-il rassemblées ? Combien d'enfants adultes, et combien de couples de parents ? (Compare le nombre annoncé dans le texte avec le total du tableau I.)
2. Pourquoi multiplie-t-il la taille des femmes par 1,08 ?
3. Galton constate que l'écart d'un enfant à la moyenne vaut en moyenne les **deux tiers** de l'écart de ses parents (mi-parent). La moyenne de sa population est de 68¼ pouces (1 pouce = 2,54 cm). Quelle taille moyenne prévoir pour les enfants de parents de taille mi-parentale 72 pouces ? Et pour 64,5 pouces ? Donne les deux réponses en pouces (valeur exacte), puis en centimètres (1 décimale).
4. Écris cette règle sous la forme d'une droite $\hat{y} = w\,x + b$, où $x$ est la taille mi-parentale et $\hat{y}$ la taille prévue pour l'enfant : que valent $w$ et $b$ ?
5. Si les enfants des parents très grands sont plus petits qu'eux, la population finit-elle par avoir partout la même taille ? Explique pourquoi non (pense aux enfants de parents moyens).
6. Galton est aussi l'inventeur du mot « eugénisme » (1883). Pourquoi faut-il lire un article scientifique ancien, et les usages faits de ses résultats, avec un regard critique ? Quel parallèle avec les datasets d'aujourd'hui ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 1.E1 — Expliquer le machine learning à un recruteur non technique 💼 ★★ ⏱️ 10 min
*Fiche §1.1, §1.2 · prérequis 1.6 · parcours R*

« Pouvez-vous m'expliquer ce qu'est le machine learning, comme vous le feriez pour un client qui n'est pas du métier ? »

### 1.E2 — Paramètres, hyperparamètres et jeu de test 💼 ★★ ⏱️ 10 min
*Fiche §1.2.3, §1.2.4 · prérequis 1.14 (notebook) · parcours R*

« Quelle différence faites-vous entre un paramètre et un hyperparamètre ? Et pourquoi garder un jeu de test séparé : que se passe-t-il si l'on triche ? »

### 1.E3 — Supervisé, non supervisé, auto-supervisé, renforcement : un exemple chacun 💼 ★★ ⏱️ 10 min
*Fiche §1.3 à §1.6, 🕰️ familles · parcours R*

« Citez-moi les grandes familles d'apprentissage, avec un exemple concret pour chacune. »

### 1.E4 — Un LLM, c'est quoi ? Réponse en une minute 💼 ★★ ⏱️ 10 min
*Fiche « Panorama 2026 » · prérequis 1.Q11 · parcours R*

« Qu'est-ce qu'un grand modèle de langage, comme ceux derrière ChatGPT ou Claude, et comment est-il entraîné ? »

<a id="notebook"></a>

## Exercices du notebook

Les exercices suivants se font dans `03_notebook.ipynb` (ta copie : `mon_travail/ch01_introduction/03_notebook.ipynb`). La partie 0 du notebook vérifie tes réponses aux exercices ✏️ ci-dessus.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 1.9 | Penguins : échantillons, features et labels | 📦 | ★ | 10 |
| 1.10 | MNIST : une image, 784 nombres | 📦 | ★ | 10 |
| 1.11 | Holmes et Verne : le texte devient des nombres | 📦 | ★★ | 15 |
| 1.12 | Taches solaires : tracer, lisser, repérer le cycle | 📦 | ★★ | 20 |
| 1.13 | Lire les data cards des quatre fils rouges | 🛠️ | ★★ | 15 |
| 1.14 | Mémoriser n'est pas apprendre | 🔮 | ★★ | 15 |
| 1.15 | Un système expert pour les manchots | 🔨 | ★★ | 20 |
| 1.16 | La boucle d'entraînement à la main | 🔨 | ★★ | 30 |
| 1.17 | Learning rate : trop prudent, trop pressé | 🔬 | ★★ | 20 |
| 1.18 | Un arbre de décision apprend les règles à ta place | 📦 | ★★ | 20 |
| 1.19 | Un manchot d'une espèce jamais vue | 🔮 | ★★ | 15 |
| 1.20 | Le score trop beau pour être vrai | 🐛 | ★★ | 20 |
| 1.21 | Regrouper les manchots sans leurs labels | 📦 | ★★ | 20 |
| 1.22 | L'agent cuisinier : apprendre par la récompense | 🔬 | ★★ | 30 |
| 1.23 | Un réseau de neurones en boîte noire sur MNIST | 📦 | ★★ | 25 |
| 1.24 | Fabriquer du faux Holmes et du faux Verne | 🔨 | ★★★ | 35 |
| 1.25 | Battre l'expert : 95 % avec tes propres règles | 🏆 | ★★★ | 40 |

Indices : `04_indices.md` (section « Notebook ») ; solutions commentées : `05_solutions.md` et `05_solutions.ipynb`.
