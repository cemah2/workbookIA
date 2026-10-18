# 21 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/ch21_cnn/06_mes_reponses.md` (créée par `python tools/start_chapter.py 21`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_21_Q1a = ...`, `answer_21_1a = ...`) pour les vérifier.

## 🧠 Quiz

### 21.Q1 — Les deux sens de « profondeur »

- a) :
- b) :
- c) :
- d) :
- e) :


### 21.Q2 — Un seul neurone promené sur toute l'image

- a) :
- b) :
- c) :
- d) :
- e) :


### 21.Q3 — Convolution ou corrélation croisée ?

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 21.Q4 — Détecteurs de motifs et hiérarchie de filtres

- a) :
- b) :
- c) :
- d) :
- e) :


### 21.Q5 — Padding et stride : que devient la taille ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 21.Q6 — Canaux d'entrée, nombre de filtres, canaux de sortie

- a) :
- b) :
- c) :
- d) :
- e) :


### 21.Q7 — Convolution 1D ou convolution 1×1 ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 21.Q8 — Couche de convolution et initialisation des filtres

- a) :
- b) :
- c) :
- d) :
- e) :


### 21.Q9 — Convolution transposée : les mots piégés

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### 21.Q10 — VGG16 et ce que voient ses filtres

- a) :
- b) :
- c) :
- d) :
- e) :


### 21.Q11 — Exemples adversariaux : vrai ou faux ?

- a) :
- b) :
- c) :
- d) :
- e) :


### 21.Q12 — 🕰️ Après VGG16 : ResNet, ViT et modèles pré-entraînés

- a) :
- b) :
- c) :
- d) :


## 🔁 Rappels

### 21.R1 — Ch. 20 : ce que changent model.train() et model.eval() pour le dropout

- a) :
- b) :
- c) :
- d) :


### 21.R2 — Ch. 18 : le gradient d'un poids utilisé à plusieurs endroits

- a) :
- b) :
- c) :


### 21.R3 — 0B : le produit scalaire, ce que calcule un filtre en chaque position

- a) :
- b) :
- c) :


## ✏️ ∂ 🧮 Papier-crayon

### Ex 21.1 — Le détecteur de jaune à la main

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


### Ex 21.2 — Une corrélation 5×5 par 3×3 à la main

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


### Ex 21.3 — Tailles de sortie : padding, stride et la hiérarchie 600 → 300 → 100

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


### Ex 21.4 — Compter les paramètres : convolution contre couche dense

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


### Ex 21.5 — Champ récepteur effectif d'une pile convolution + pooling

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


### Ex 21.6 — Convolution transposée à la main : insérer des zéros puis corréler

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


### Ex 21.7 — Rétropropager dans une convolution 1D : la somme sur les positions

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex 21.8 — Fermi : paramètres, mémoire et calcul de VGG16

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

### Ex 21.9 — Expliquer la convolution en cinq lignes

Mes cinq lignes :


### Ex 21.10 — Deviner l'âge et le genre sur une photo : jusqu'où aller ?

**1.**


**2.**


**3.**


**4.**


**5.**


### Ex 21.11 — Lire LeCun et coll. (1998) : LeNet-5

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


**8.**


## 💼 Entretien

### 21.E1 — Pourquoi un CNN plutôt qu'un MLP pour des images ?


### 21.E2 — Taille de sortie et paramètres d'une Conv2d(3, 64, 3, stride=2, padding=1) sur une image 224×224


### 21.E3 — À quoi sert une convolution 1×1 ?


### 21.E4 — Exemples adversariaux : quels risques, quelles défenses ?


### 21.E5 — CNN ou Vision Transformer pour un nouveau projet de vision ?


## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook qui n'ont pas de cellule 📝.

- **21.12** :
- **21.13** :
- **21.14** :
- **21.15** :
- **21.16** :
- **21.17** :
- **21.18** :
- **21.19** :
- **21.20** :
- **21.21** :
- **21.22** :
- **21.23** :
- **21.24** :
- **21.25** :
- **21.26** :
- **21.27** :
- **21.28** :
- **21.29** :
- **21.30** :
- **21.31** :
- **21.32** :
