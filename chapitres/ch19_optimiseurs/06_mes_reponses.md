# 19 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/ch19_optimiseurs/06_mes_reponses.md` (créée par `python tools/start_chapter.py 19`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_19_Q1a = ...`, `answer_19_1a = ...`) pour les vérifier.

## 🧠 Quiz

### 19.Q1 — Qu'appelle-t-on un optimiseur ?

- a) :
- b) :
- c) :
- d) :


### 19.Q2 — Minimum, maximum, plateau, selle : où le gradient s'annule-t-il ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 19.Q3 — Minima locaux ou points selles : le vrai piège en grande dimension

- a) :
- b) :
- c) :
- d) :


### 19.Q4 — Pourquoi lisser une erreur de comptage ?

- a) :
- b) :
- c) :
- d) :


### 19.Q5 — Pas constant : pourquoi rebondit-on au fond de la vallée ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 19.Q6 — Décroissance exponentielle : que vaut η après 50 pas ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 19.Q7 — Plannings : exponentiel, retardé, par paliers, sur l'erreur, bold driver

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 19.Q8 — Batch, stochastique, mini-batch : compter les mises à jour

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 19.Q9 — Momentum : que se passe-t-il si γ = 0 ? si γ = 1 ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 19.Q10 — Nesterov : regarder là où l'on va

- a) :
- b) :
- c) :
- d) :


### 19.Q11 — Adagrad, Adadelta, RMSprop : un learning rate par poids

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 19.Q12 — Adam, No Free Lunch et choix d'un optimiseur

- a) :
- b) :
- c) :
- d) :
- e) :


## 🔁 Rappels

### 19.R1 — Du delta au gradient d'un poids

- a) :
- b) :
- c) :
- d) :


### 19.R2 — Pourquoi l'initialisation He garde la variance stable

- a) :
- b) :
- c) :
- d) :


### 19.R3 — Standardiser les features, et l'effet sur la descente

- a) :
- b) :
- c) :
- d) :


## ✏️ ∂ 🧮 Papier-crayon

### Ex 19.1 — Plannings exponentiel et par paliers à la main

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :




### Ex 19.2 — Trois pas de momentum, deux conventions

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :

Dans ta copie :




### Ex 19.3 — Nesterov : gradient anticipé et reformulation de PyTorch

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :




### Ex 19.4 — Adagrad et RMSprop sur une suite de gradients

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




### Ex 19.5 — Un pas d'Adam et la correction de biais

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :
- i) :
- j) :

Dans ta copie :




### Ex 19.6 — Adam + L2 ou AdamW : la même pénalité ? Convertir alpha, lambda et weight_decay d'une convention à l'autre

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :
- i) :
- j) :

Dans ta copie :




### Ex 19.7 — Sur une parabole, η doit rester sous 2/a

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**



### Ex 19.8 — Moyenne mobile exponentielle : mémoire effective et biais initial

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**



### Ex 19.9 — Fermi : la mémoire des optimiseurs, du MLP au LLM

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :




## 🗣️ ⚖️ 📄 🛠️ Réflexion

### Ex 19.10 — Expliquer le momentum avec une bille qui roule

Mes cinq lignes :




### Ex 19.11 — Le coût énergétique d'une recherche d'hyperparamètres

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**



### Ex 19.12 — Kingma & Ba (2014) : l'algorithme Adam ligne par ligne

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


**8.**



### Ex 19.13 — Lire la doc de torch.optim : correspondance livre ↔ PyTorch

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


**8.**


Dans ta copie :




## 💼 Entretien

### 19.E1 — Adam ou SGD avec momentum : lequel choisir, et pourquoi ?

Ma réponse en une minute :




### 19.E2 — Adam avec weight_decay contre AdamW

Ma réponse en une minute :




### 19.E3 — À quoi servent le warmup et la décroissance en cosinus ?

Ma réponse en une minute :




### 19.E4 — Qu'est-ce que le gradient clipping, et quand l'utiliser ?

Ma réponse en une minute :




### 19.E5 — Taille du mini-batch : quels compromis ?

Ma réponse en une minute :




## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook qui n'ont pas de cellule 📝.

- **19.14** :
- **19.15** :
- **19.16** :
- **19.17** :
- **19.18** :
- **19.19** :
- **19.20** :
- **19.21** :
- **19.22** :
- **19.23** :
- **19.24** :
- **19.25** :
- **19.26** :
- **19.27** :
- **19.28** :
