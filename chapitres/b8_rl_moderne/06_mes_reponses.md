# B8 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/b8_rl_moderne/06_mes_reponses.md` (créée par `python tools/start_chapter.py B8`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_B8_Q1a = ...`, `answer_B8_1a = ...`) pour les vérifier.

## 🧠 Quiz

### B8.Q1 — Pourquoi une table Q ne suffit plus pour CartPole

- a) :
- b) :
- c) :
- d) :
- e) :



### B8.Q2 — La triade mortelle : approximation, amorçage et apprentissage off-policy

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :



### B8.Q3 — Réseau cible et mémoire de rejeu : quel problème chacun résout-il ? Vrai ou faux

- a) :
- b) :
- c) :
- d) :
- e) :



### B8.Q4 — Double DQN : d'où vient la surestimation des valeurs Q ?

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :



### B8.Q5 — Méthodes par valeur ou par politique : actions continues et politiques aléatoires

- a) :
- b) :
- c) :
- d) :
- e) :



### B8.Q6 — REINFORCE : pourquoi soustraire une baseline ne biaise pas le gradient

- a) :
- b) :
- c) :
- d) :
- e) :



### B8.Q7 — Avantage, erreur TD et GAE : ce que règle λ

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :



### B8.Q8 — PPO : à quoi sert le clip, et pourquoi réutiliser plusieurs fois le même lot ?

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :



### B8.Q9 — On-policy ou off-policy : DQN, REINFORCE et PPO

- a) :
- b) :
- c) :
- d) :
- e) :



### B8.Q10 — Évaluer un agent : graines, épisodes, politique déterministe

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :



### B8.Q11 — RLHF, DPO, GRPO : qui a besoin d'un modèle de récompense ou d'un critique ?

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :



### B8.Q12 — Piratage de la récompense (reward hacking) : quand l'agent optimise la lettre et pas l'esprit

- a) :
- b) :
- c) :
- d) :
- e) :



## 🔁 Rappels

### B8.R1 — Ch. 26 : la mise à jour du Q-learning et le rôle de la mémoire de rejeu

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :



### B8.R2 — Ch. 19 : Adam et le gradient clipping

- a) :
- b) :
- c) :
- d) :
- e) :



### B8.R3 — Ch. 11 : exploration, softmax et bandits

- a) :
- b) :
- c) :
- d) :
- e) :



## ✏️ ∂ 🧮 Papier-crayon

### Ex B8.1 — Cible DQN à la main : état terminal, puis cible Double DQN

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



### Ex B8.2 — Loss de Huber et son gradient pour trois erreurs TD

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



### Ex B8.3 — Gradient du log d'une politique softmax à trois actions

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



### Ex B8.4 — GAE à la main sur quatre pas, pour λ = 0, 0,95 et 1

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
- j) :

Dans ta copie :



### Ex B8.5 — Objectif clippé de PPO : quatre cas, valeur et gradient nul ou non

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



### Ex B8.6 — Préférences : loss de Bradley-Terry puis loss DPO sur une paire de réponses

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
- j) :

Dans ta copie :



### Ex B8.7 — Le gradient de politique en une étape : l'astuce de la log-dérivée, puis la baseline sans biais

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**



### Ex B8.8 — Fermi : combien d'interactions ? CartPole, Atari (DQN 2015), AlphaGo Zero et le temps CPU correspondant

Mes calculs :


**1.**


**2.**


**3.**


**4.**



## 🗣️ ⚖️ 📄 Réflexion

### Ex B8.9 — PPO expliqué en cinq lignes




### Ex B8.10 — RLHF : qui choisit les préférences ? Annotateurs, valeurs et flagornerie des modèles

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**



### Ex B8.11 — Schulman et coll. (2017) : « Proximal Policy Optimization Algorithms »

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

### B8.E1 — Méthodes par valeur ou par gradient de politique : avantages et inconvénients




### B8.E2 — Explique PPO et le rôle du clipping




### B8.E3 — Ton agent RL n'apprend pas : comment le débogues-tu ?




### B8.E4 — PPO avec un modèle de récompense ou DPO pour aligner un modèle de langue : que choisis-tu ?




### B8.E5 — Pourquoi faut-il plusieurs graines pour comparer deux algorithmes de RL ?




## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook.

- **B8.12** :
- **B8.13** :
- **B8.14** :
- **B8.15** :
- **B8.16** :
- **B8.17** :
- **B8.18** :
- **B8.19** :
- **B8.20** :
- **B8.21** :
- **B8.22** :
- **B8.23** :
- **B8.24** :
- **B8.25** :
- **B8.26** :
- **B8.27** :
- **B8.28** :
