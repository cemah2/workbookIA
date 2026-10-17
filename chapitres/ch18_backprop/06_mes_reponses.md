# 18 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/ch18_backprop/06_mes_reponses.md` (créée par `python tools/start_chapter.py 18`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_18_Q1a = ...`, `answer_18_1a = ...`) pour les vérifier.

## 🧠 Quiz

### 18.Q1 — Pourquoi apprendre la rétropropagation si PyTorch la fait pour toi ?

- a) :
- b) :
- c) :

Dans ta copie :



### 18.Q2 — Perturber un poids au hasard : pourquoi est-ce si lent ?

- a) :
- b) :
- c) :
- d) :

### 18.Q3 — Ce que le gradient dit à chaque poids, et le cas du biais

- a) :
- b) :
- c) :
- d) :

### 18.Q4 — Le delta d'un neurone : signe et effet sur l'erreur

- a) :
- b) :
- c) :
- d) :
- e) :

### 18.Q5 — Le réseau minuscule : couches, poids et leurs noms

- a) :
- b) :
- c) :
- d) :
- e) :

### 18.Q6 — Delta de sortie, delta caché, mise à jour d'un poids

- a) :
- b) :
- c) :
- d) :

### 18.Q7 — Rétropropager couche par couche, puis avec une activation

- a) :
- b) :
- c) :
- d) :
- e) :

### 18.Q8 — Learning rate trop grand ou trop petit : reconnaître les symptômes

- a) :
- b) :
- c) :
- d) :
- e) :

### 18.Q9 — Ce que la rétropropagation propage vraiment

- a) :
- b) :
- c) :
- d) :

### 18.Q10 — Neurones saturés et ReLU mortes

- a) :
- b) :
- c) :
- d) :

### 18.Q11 — Mini-batches, mises à jour parallèles et coût du backward

- a) :
- b) :
- c) :
- d) :

### 18.Q12 — Aucune garantie, une longue histoire, des équations à apprivoiser

- a) :
- b) :
- c) :
- d) :
- e) :

## 🔁 Rappels

### 18.R1 — Ch. 17 : la dérivée de la sigmoïde à partir de sa sortie

- a) :
- b) :
- c) :
- d) :
- e) :

### 18.R2 — Ch. 15 : chercher le meilleur learning rate par recherche en grille

- a) :
- b) :
- c) :
- d) :

### 18.R3 — Ch. 11 : représentation, évaluation, optimisation dans un MLP

- a) :
- b) :
- c) :
- d) :

## ✏️ ∂ 🧮 Papier-crayon

### Ex 18.1 — Le delta, amplificateur de changement

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :



### Ex 18.2 — Deltas de sortie avec l'erreur quadratique : attention au signe

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :



### Ex 18.3 — Descente sur une parabole : quel learning rate ?

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :

Dans ta copie :



### Ex 18.4 — Rétropropagation complète du réseau minuscule, sans activation

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :
- i) :

Dans ta copie :



### Ex 18.5 — Le même réseau avec une sigmoïde cachée : le facteur σ'(z)

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :

Dans ta copie :



### Ex 18.6 — La règle de la chaîne derrière les deltas

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**



### Ex 18.7 — Forme matricielle : XᵀΔ, ΔWᵀ et somme des lignes

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**



### Ex 18.8 — Softmax et cross-entropy : le gradient vaut p − y

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**



### Ex 18.9 — Fermi : rétropropagation contre force brute

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :

Dans ta copie :



## 🗣️ 📄 Réflexion

### Ex 18.10 — Expliquer la rétropropagation sans équation

Mes cinq phrases :




### Ex 18.11 — Rumelhart, Hinton & Williams (1986) : l'article qui a relancé les réseaux

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


**8.**


**9.**



## 💼 Entretien

### 18.E1 — Explique la rétropropagation en une minute




### 18.E2 — Pourquoi la rétropropagation bat-elle les différences finies ?




### 18.E3 — Gradient évanescent : causes, symptômes, remèdes




### 18.E4 — Comment vérifies-tu une implémentation de gradient ?




### 18.E5 — Pourquoi calculer la loss sur les logits plutôt qu'après la softmax ?




## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook qui n'ont pas de cellule 📝.

- **18.12** :
- **18.13** :
- **18.14** :
- **18.15** :
- **18.16** :
- **18.17** :
- **18.18** :
- **18.19** :
- **18.20** :
- **18.21** :
- **18.22** :
- **18.23** :
- **18.24** :
- **18.25** :
- **18.26** :
- **18.27** :
- **18.28** :
- **18.29** :
