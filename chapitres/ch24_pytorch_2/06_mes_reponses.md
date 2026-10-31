# 24 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/ch24_pytorch_2/06_mes_reponses.md` (créée par `python tools/start_chapter.py 24`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_24_Q1a = ...`, `answer_24_1a = ...`) pour les vérifier.

## 🧠 Quiz

### 24.Q1 — Compter les hyperparamètres et n'en changer qu'un

- a) :
- b) :
- c) :
- d) :
- e) :


### 24.Q2 — Taille de batch : bruit, overfitting et temps

- a) :
- b) :
- c) :
- d) :
- e) :


### 24.Q3 — Plus de neurones, moins de neurones, dropout

- a) :
- b) :
- c) :
- d) :
- e) :


### 24.Q4 — Valider un réseau : coût, jeu non représentatif, normalisation dans chaque fold

- a) :
- b) :
- c) :
- d) :
- e) :


### 24.Q5 — 🕰️ KerasClassifier, grille, hasard ou zoom : chercher des hyperparamètres

- a) :
- b) :
- c) :
- d) :
- e) :


### 24.Q6 — 🕰️ Couches utilitaires de Keras et leurs équivalents PyTorch

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 24.Q7 — Préparer des images pour nn.Conv2d

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 24.Q8 — Augmentation et données synthétiques : quelles transformations sont plausibles ?

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 24.Q9 — Motifs conv-conv-pool, stride contre pooling, batchnorm contre dropout

- a) :
- b) :
- c) :
- d) :
- e) :


### 24.Q10 — Séries pour un RNN : fenêtres, scaler, inverse_transform, RMSE

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 24.Q11 — 🕰️ return_sequences, stateful, TimeDistributed : que devient chaque option en PyTorch ?

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 24.Q12 — 🕰️ API fonctionnelle de Keras 3 et forward() de PyTorch : branches, partage, gel

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


## 🔁 Rappels

### 24.R1 — Ch. 23 : ce qu'un historique d'entraînement doit contenir pour comparer deux runs

- a) :
- b) :
- c) :
- d) :


### 24.R2 — Ch. 21 : taille de sortie d'une convolution à padding 'same' et stride 2

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 24.R3 — Ch. 17 : ce que change la division des logits avant le softmax

- a) :
- b) :
- c) :
- d) :
- e) :


## ✏️ ∂ 🧮 Papier-crayon

### Ex 24.1 — Le coût d'une grille (27 hyperparamètres, 54 entraînements) contre une recherche aléatoire : probabilité de tomber dans les 5 % meilleurs réglages en n essais

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


### Ex 24.2 — Formes et paramètres du CNN à pooling du livre

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


### Ex 24.3 — Stride ou pooling : compter les multiplications

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


### Ex 24.4 — Ce que renvoie nn.LSTM : output, h_n et c_n

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


### Ex 24.5 — Fenêtres, pas d'optimisation et durée : de 200 à 20 000 points

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


### Ex 24.6 — Dropout inversé et max-norm : espérance et projection

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


### Ex 24.7 — Fermi : le budget d'une recherche d'hyperparamètres sur CPU, sur GPU et en précision mixte

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


## 🗣️ ⚖️ 📄 Réflexion

### Ex 24.8 — La data augmentation en cinq lignes

Mes cinq lignes :


### Ex 24.9 — Une seule graine suffit-elle ? Reproductibilité et honnêteté des comparaisons d'hyperparamètres

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**



### Ex 24.10 — Lire He et coll. (2016) : les réseaux résiduels

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**



## 💼 Entretien

### 24.E1 — Comment organises-tu une recherche d'hyperparamètres avec un budget limité ?


### 24.E2 — Pourquoi la data augmentation améliore-t-elle la généralisation ? Donne des transformations à éviter


### 24.E3 — Dropout ou batchnorm : lequel, où, et peut-on combiner les deux ?


### 24.E4 — Comment éviter la fuite de données dans une validation croisée avec prétraitement ?


### 24.E5 — Batchnorm : que fait-elle, et pourquoi se comporte-t-elle différemment en train et en eval ?


## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook qui n'ont pas de cellule 📝.

- **24.11** :
- **24.12** :
- **24.13** :
- **24.14** :
- **24.15** :
- **24.16** :
- **24.17** :
- **24.18** :
- **24.19** :
- **24.20** :
- **24.21** :
- **24.22** :
- **24.23** :
- **24.24** :
- **24.25** :
- **24.26** :
- **24.27** :
- **24.28** :
- **24.29** :
- **24.30** :
- **24.31** :
