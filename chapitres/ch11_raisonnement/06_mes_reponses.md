# 11 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/ch11_raisonnement/06_mes_reponses.md` (créée par `python tools/start_chapter.py 11`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_11_Q1a = ...`, `answer_11_1a = ...`) pour les vérifier.

## 🧠 Quiz

### 11.Q1 — Représentation, évaluation, optimisation : associer

- a) :
- b) :


### 11.Q2 — Puissance de représentation : ce qu'un perceptron ne peut pas « savoir »

- a) :
- b) :
- c) :


### 11.Q3 — Représentable mais pas apprenable : le problème de l'arrêt

- a) :
- b) :
- c) :
- d) :
- e) :


### 11.Q4 — Loss, métrique, objectif : qui sert à quoi ?

- a) :
- b) :
- c) :
- d) :


### 11.Q5 — Optimiser n'est pas être optimal ; pas de repas gratuit

- a) :
- b) :
- c) :


### 11.Q6 — Déduction ou induction ? Six situations

- a) :


### 11.Q7 — Valide, solide, ou ni l'un ni l'autre ?

- a) :
- b) :


### 11.Q8 — Nommer le sophisme syllogistique

- a) :


### 11.Q9 — Généralisation, syllogisme statistique, prédiction

- a) :
- b) :
- c) :


### 11.Q10 — Sophismes inductifs chez les data scientists

- a) :


### 11.Q11 — Prémisses rationnelles, empiriques, et la fourche de Hume

- a) :
- b) :
- c) :


### 11.Q12 — Holmes déduit-il vraiment ?

- a) :
- b) :
- c) :
- d) :


## 🔁 Rappels

### 11.R1 — Ch. 10 : la règle du perceptron et le cas XOR

- a) :
- b) :
- c) :
- d) :


### 11.R2 — Ch. 8 : représentativité du jeu d'entraînement et fuite de données

- a) :
- b) :
- c) :
- d) :


### 11.R3 — Ch. 4 : mettre à jour sa croyance sur une pièce avec Bayes

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


## ✏️ ∂ Papier-crayon

Reporte ensuite chaque réponse dans la partie 0 du notebook (`answer_11_1a = ...`) pour la vérifier.

### Ex 11.1 — Représentable sur n bits : compter, puis conclure

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

Stocker une valeur ou un programme qui la calcule :


### Ex 11.2 — Moyenne incrémentale : Qₙ₊₁ = Qₙ + (Rₙ − Qₙ)/n

**1.**


**2.**


**3.**


**4.**


**5.**


### Ex 11.3 — Syllogismes : valides ? solides ?

| Syllogisme | Formes (majeure, mineure, conclusion) | Moyen terme distribué ? | Valide ? | Prémisses vraies ? | Code (S, V, N) |
|---|---|---|---|---|---|
| S1 | | | | | |
| S2 | | | | | |
| S3 | | | | | |
| S4 | | | | | |
| S5 | | | | | |
| S6 | | | | | |
| S7 | | | | | |

- a) :
- b) :
- c) :
- d) :
- e) :

Un contre-exemple pour un syllogisme non valide :


### Ex 11.4 — Six raisonnements fautifs à diagnostiquer et à réfuter

| N° | Déductif ou inductif ? | Faute | Contre-exemple, prémisse réfutée ou bonne méthode |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |


### Ex 11.5 — Enquête au phare : réduire le domaine du discours

Démarche (les suspects après chaque indice) :


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

Les prémisses empiriques de chaque indice, et le lien avec un classifieur :


### Ex 11.6 — Syllogisme statistique et prédiction : 15 % de pommes mûres

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
- j) :

Le principe inductif utilisé en a), e) et h) :


### Ex 11.7 — Renforcement ou punition, positif ou négatif : classer huit situations

| Situation | On ajoute ou on retire ? | Le comportement devient plus ou moins fréquent ? | Code |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |

- a) :
- b) :
- c) :
- d) :

Le perceptron : un argument pour, un argument contre :


### Ex 11.8 — Un bandit à la main : ε-greedy, moyennes et regret

| Pas | Explore ? | Bras joué | Récompense | $Q(0)$ | $Q(1)$ | $Q(2)$ | Regret du pas |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |
| 8 | | | | | | | |

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

Pseudo-regret et regret réalisé ; le regret après 10 000 pas :


## 🗣️ 📈 ⚖️ 📄 Réflexion

### Ex 11.9 — Déduction et induction dans un projet de ML, en cinq lignes

Mes cinq lignes :


### Ex 11.10 — Lire les courbes d'un bandit : ε = 0, 0,01 et 0,1

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :

Réponse rédigée (g) :


- h) :


### Ex 11.11 — Explorer sur des humains : essais adaptatifs, recommandation, A/B tests

**1.**


**2.**


**3.**


**4.**


**5.**


### Ex 11.12 — Domingos (2012) : représentation, évaluation, optimisation et autres leçons

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


## 💼 Entretien

Écris les mots-clés de ta réponse, puis entraîne-toi à la dire en une minute.

### 11.E1 — Exploration contre exploitation : expliquer avec un exemple métier


### 11.E2 — A/B test ou bandit : lequel choisir ?


### 11.E3 — Que dit le théorème « No Free Lunch » pour le choix d'un modèle ?


### 11.E4 — Un biais d'échantillonnage qui a fait échouer un modèle : exemple et parade


## 📝 Notes sur le notebook

Ce qui m'a bloqué, ce que j'ai compris, les questions à poser (par exercice) :

- **11.13** :
- **11.14** :
- **11.15** :
- **11.16** :
- **11.17** :
- **11.18** :
- **11.19** :
- **11.20** :
- **11.21** :
- **11.22** :
- **11.23** :
- **11.24** :
- **11.25** :
- **11.26** :
- **11.27** :
