# 23 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/ch23_pytorch_1/06_mes_reponses.md` (créée par `python tools/start_chapter.py 23`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_23_Q1a = ...`, `answer_23_1a = ...`) pour les vérifier.

## 🧠 Quiz

### 23.Q1 — Pourquoi utiliser une bibliothèque plutôt que tout coder soi-même ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q2 — Avertissement ou erreur ? Versions et dépréciations

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q3 — Déboguer méthodiquement : réduire, isoler, reconstruire

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q4 — Les trois sens du mot « modèle »

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q5 — Tenseurs, arrays et type float32

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q6 — channels_first, channels_last et NCHW

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q7 — CPU, GPU, MPS, TPU : où calcule PyTorch ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q8 — Iris, MNIST, Fashion-MNIST : les « hello world » du ML

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q9 — Labels entiers, one-hot et logits

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q10 — fit() décortiqué : batch, epoch, validation, train() et eval()

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q11 — Sauvegarder tout, les poids ou l'architecture ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.Q12 — 🕰️ Que devient le code Keras du livre en 2026 ?

- a) :
- b) :
- c) :
- d) :
- e) :


## 🔁 Rappels

### 23.R1 — Ch. 22 : (N, T, D) et batch_first

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.R2 — Ch. 20 : les cinq lignes d'une étape d'entraînement

- a) :
- b) :
- c) :
- d) :
- e) :


### 23.R3 — Ch. 2 : moyenne et écart-type du train pour normaliser les pixels

- a) :
- b) :
- c) :
- d) :
- e) :


## ✏️ 🧮 Papier-crayon

### Ex 23.1 — Remodeler 12 éléments : indices et ordre de lecture

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


### Ex 23.2 — De NHWC à NCHW : où atterrit chaque pixel ?

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


### Ex 23.3 — 623 290 paramètres : refaire le compte du livre

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


### Ex 23.4 — Une cross-entropy à la main : indice de classe ou one-hot

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


### Ex 23.5 — Batches par epoch et nombre de mises à jour

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


### Ex 23.6 — Early stopping à la main : patience et min_delta

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


### Ex 23.7 — Plannings de learning rate : le planning du livre avec LambdaLR

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :


Dans ta copie :


### Ex 23.8 — Fermi : durée d'une epoch et poids d'un checkpoint

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :


Dans ta copie :


## 🗣️ ⚖️ 📄 Réflexion

### Ex 23.9 — Pourquoi le prétraitement doit voyager avec le modèle

Mes cinq lignes :


### Ex 23.10 — Réutiliser un modèle pré-entraîné : licence, données d'origine, biais hérités

**1.**


**2.**


**3.**


**4.**


**5.**



### Ex 23.11 — Lire Krizhevsky et coll. (2012) : la recette d'entraînement d'AlexNet

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**



## 💼 Entretien

### 23.E1 — Ton entraînement PyTorch est deux fois plus lent que prévu : où cherches-tu, et dans quel ordre ?


### 23.E2 — Pourquoi appeler model.eval() et torch.no_grad() (ou inference_mode) à l'inférence ?


### 23.E3 — Comment sauvegarder un modèle PyTorch pour reprendre l'entraînement plus tard ou le déployer ?


### 23.E4 — La loss de validation remonte alors que l'accuracy de validation progresse encore : que se passe-t-il ?


### 23.E5 — Keras ou PyTorch : comment choisir un framework aujourd'hui ?


## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook qui n'ont pas de cellule 📝.

- **23.12** :
- **23.13** :
- **23.14** :
- **23.15** :
- **23.16** :
- **23.17** :
- **23.18** :
- **23.19** :
- **23.20** :
- **23.21** :
- **23.22** :
- **23.23** :
- **23.24** :
- **23.25** :
- **23.26** :
- **23.27** :
- **23.28** :
- **23.29** :
