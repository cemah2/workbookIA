# B3 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/b3_transformers/06_mes_reponses.md` (créée par `python tools/start_chapter.py B3`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_B3_Q1a = ...`, `answer_B3_1a = ...`) pour les vérifier.

## 🧠 Quiz

### B3.Q1 — Le goulot de l'encodeur-décodeur récurrent : pourquoi un seul vecteur ne suffit pas

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q2 — Requête, clé, valeur : un dictionnaire où l'on cherche « à peu près »

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q3 — Pourquoi diviser les scores par √d_k ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q4 — Masque causal : qui a le droit de regarder qui ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q5 — Auto-attention ou attention croisée : d'où viennent Q, K et V ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q6 — Plusieurs têtes pour le prix d'une : que gagne-t-on ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q7 — Sans encodage de position, l'attention voit-elle l'ordre des mots ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q8 — Résiduel, pré-LN ou post-LN : ce qui rend un Transformer profond entraînable

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q9 — BERT, GPT, T5 : quel masque, quel objectif, quel usage ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q10 — Décalage d'un token : ce que chaque position apprend à prédire

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q11 — KV cache : ce que l'on garde et pourquoi la génération s'accélère

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.Q12 — Coût quadratique du contexte et ce que change FlashAttention

- a) :
- b) :
- c) :
- d) :
- e) :


## 🔁 Rappels

### B3.R1 — Ch. B2 : de quoi dépend le nombre de tokens d'un texte ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.R2 — Ch. 22 : pourquoi le gradient s'évanouit dans un RNN déroulé

- a) :
- b) :
- c) :
- d) :
- e) :


### B3.R3 — Ch. 20 : layer norm et connexion résiduelle, à quoi sert chacune ?

- a) :
- b) :
- c) :
- d) :
- e) :


## ✏️ ∂ 🧮 Papier-crayon

### Ex B3.1 — L'attention à la main : trois tokens, d_k = 2, scores, softmax et sortie

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :


### Ex B3.2 — Le même calcul avec un masque causal

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


### Ex B3.3 — Suivre les formes dans l'attention multi-têtes et compter ses paramètres

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


### Ex B3.4 — Encodage sinusoïdal : quelques valeurs et l'effet d'un décalage de position

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


### Ex B3.5 — Compter les paramètres d'un GPT et retrouver les 124 millions de GPT-2 small

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


### Ex B3.6 — Pourquoi √d_k : variance du produit scalaire de deux vecteurs aléatoires

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B3.7 — RoPE : la rotation rend q_m · k_n fonction de m − n seulement

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B3.8 — Fermi : FLOPs d'entraînement (≈ 6 N D), ton mini-GPT sur CPU, GPT-3, et la mémoire du KV cache

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


## 🗣️ ⚖️ 📄 Réflexion

### Ex B3.9 — L'attention expliquée en cinq lignes

Mes cinq lignes :


### Ex B3.10 — Mémorisation : quand un modèle de langue recrache ses données d'entraînement

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B3.11 — Vaswani et coll. (2017) : « Attention Is All You Need »

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

### B3.E1 — Explique le mécanisme d'attention, de la requête à la sortie


### B3.E2 — Quel est le coût de l'auto-attention en fonction de la longueur, et comment traite-t-on les contextes longs ?


### B3.E3 — Encodeur, décodeur ou encodeur-décodeur : lequel pour quelle tâche ?


### B3.E4 — À quoi sert le KV cache, et combien de mémoire consomme-t-il ?


### B3.E5 — Pourquoi un Transformer a-t-il besoin d'encodages de position, et qu'apporte RoPE ?


## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook.

- **B3.12** :
- **B3.13** :
- **B3.14** :
- **B3.15** :
- **B3.16** :
- **B3.17** :
- **B3.18** :
- **B3.19** :
- **B3.20** :
- **B3.21** :
- **B3.22** :
- **B3.23** :
- **B3.24** :
- **B3.25** :
- **B3.26** :
- **B3.27** :
- **B3.28** :
- **B3.29** :
- **B3.30** :
