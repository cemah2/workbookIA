# 10 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/ch10_neurones/06_mes_reponses.md` (créée par `python tools/start_chapter.py 10`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_10_Q1a = ...`, `answer_10_1a = ...`) pour les vérifier.

## 🧠 Quiz

### 10.Q1 — Neurones artificiels : où sont-ils indispensables, où s'en passe-t-on ?

- a) :
- b) :
- c) :


### 10.Q2 — Le neurone biologique en quatre étapes

- a) :
- b) :
- c) :


### 10.Q3 — Connectome et émulation du cerveau : vrai ou faux justifié

- a) :
- b) :
- c) :
- d) :
- e) :


### 10.Q4 — Neurone, unité, « cerveau électronique » : bien nommer les choses

- a) :
- b) :
- c) :


### 10.Q5 — McCulloch et Pitts (1943) : ce qu'ils ont démontré

- a) :
- b) :
- c) :


### 10.Q6 — Anatomie d'un perceptron

- a) :
- b) :
- c) :


### 10.Q7 — Mark I, Minsky-Papert, renaissance : la chronologie

- a) :
- b) :
- c) :
- d) :


### 10.Q8 — Du perceptron au neurone moderne : les deux changements

- a) :
- b) :
- c) :
- d) :


### 10.Q9 — Lire un schéma de réseau : poids implicites et convention AD/DA

- a) :
- b) :
- c) :
- d) :
- e) :


## 🔁 Rappels

### 10.R1 — Ch. 9 : régularisation L2, que deviennent les poids ?

- a) :
- b) :
- c) :
- d) :

Réponse rédigée (e) :


### 10.R2 — Ch. 7 : la frontière du centroïde le plus proche, w·x + b = 0

Démarche (le développement des deux carrés) :


- a) :
- b) :
- c) :
- d) :
- e) :


### 10.R3 — Ch. 3 : matrice de confusion et accuracy d'un classifieur binaire

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


## ✏️ ∂ Papier-crayon

Reporte ensuite chaque réponse dans la partie 0 du notebook (`answer_10_1a = ...`) pour la vérifier.

### Ex 10.1 — Sortie d'un perceptron à quatre entrées, avec et sans biais

Démarche :


Réponses :

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :
- i) :


### Ex 10.2 — L'astuce du biais : même neurone, une entrée de plus

Démarche :


Réponses :

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Réponse rédigée (h) :


### Ex 10.3 — Portes logiques à la main : AND, OR, NOT, NAND

| Porte | $\mathbf{w}$ | $b$ | Vérification (toutes les entrées) |
|---|---|---|---|
| OR | | | |
| NOT | | | |
| NAND | | | |
| NOR | | | |
| majorité (3 entrées) | | | |

c) Pourquoi changer les signes marche ici, et le cas d'une somme nulle :


f) Multiplier par $c > 0$, puis par $c < 0$ :


g) Mon dessin (ou sa description) :


### Ex 10.4 — Pourquoi un seul perceptron ne peut pas calculer XOR

**a)** Les quatre inégalités :


**b)** La contradiction :


**c)** La preuve géométrique :


**d)** XNOR :


**e)** XOR avec $x_3 = x_1 x_2$ :


**f)** Les fonctions logiques à deux entrées :


### Ex 10.5 — XOR en deux couches : câbler et nommer les poids

Démarche (le tableau des sorties de C, D et E) :


Réponses :

- a) :
- b) :
- c) :
- e) :
- f) :
- g) :
- h) :

Réponse rédigée (d) :


Réponse rédigée (i) :


### Ex 10.6 — Une epoch de la règle du perceptron à la main

| Epoch | Exemple | $z$ | Erreur ? | $\mathbf{w}$ après | $b$ après |
|---|---|---|---|---|---|
| 1 | (0, 0) | | | | |
| 1 | (0, 1) | | | | |
| 1 | (1, 0) | | | | |
| 1 | (1, 1) | | | | |

Réponses :

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :

Réponse rédigée (i) :


### Ex 10.7 — Le théorème de convergence du perceptron, guidé pas à pas

**a)**


**b)**


**c)**


**d)**


**e)**


**f)**


**g)**


**h)**


**i)**


## 🗣️ 🧮 ⚖️ 📄 Réflexion

### Ex 10.8 — Pourquoi un neurone artificiel n'est pas un neurone

Mes mots-clés (une ressemblance, trois différences) :


### Ex 10.9 — Fermi : cerveau humain contre grands modèles

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :

Réponse rédigée (g) :


### Ex 10.10 — « Cerveaux électroniques » : hype, hivers de l'IA et responsabilité

**1.**


**2.**


**3.**


**4.**


**5.**


### Ex 10.11 — Rosenblatt (1958) : le perceptron dans le texte

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


## 💼 Entretien

Écris les mots-clés de ta réponse, puis entraîne-toi à la dire en une minute.

### 10.E1 — Qu'est-ce qu'un perceptron, et quelle est sa limite fondamentale ?


### 10.E2 — À quoi sert le biais d'un neurone ?


### 10.E3 — Pourquoi remplacer le seuil par une activation dérivable ?


### 10.E4 — Un réseau de neurones ressemble-t-il au cerveau ?


## 📝 Notes sur le notebook

Ce qui m'a bloqué, ce que j'ai compris, les questions à poser (par exercice) :
