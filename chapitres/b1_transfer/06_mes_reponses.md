# B1 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/b1_transfer/06_mes_reponses.md` (créée par `python tools/start_chapter.py B1`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_B1_Q1a = ...`, `answer_B1_1a = ...`) pour les vérifier.

## 🧠 Quiz

### B1.Q1 — Réutiliser plutôt que repartir de zéro : trois raisons et une limite

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q2 — Premières couches génériques, dernières couches spécialisées : vrai ou faux

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q3 — Où trouver un modèle pré-entraîné, et quoi lire avant de s'en servir

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q4 — weights.transforms() : pourquoi le prétraitement fait partie du modèle

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q5 — Extraction de features, linear probe, fine-tuning : qui s'entraîne ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q6 — Pourquoi un learning rate 10 à 100 fois plus petit pour fine-tuner ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q7 — Batchnorm pendant le fine-tuning sur de tout petits batches

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q8 — Peu ou beaucoup de données, domaine proche ou lointain : la bonne stratégie

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q9 — Transfert négatif et oubli catastrophique : reconnaître les symptômes

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q10 — Classer sans exemple d'entraînement : le zéro-shot de CLIP

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.Q11 — Auto-supervisé (DINOv2) ou supervisé sur ImageNet : quels embeddings ?

- a) :
- b) :
- c) :
- d) :
- e) :


## 🔁 Rappels

### B1.R1 — Ch. 29 : licence d'un dataset et décalage de distribution

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.R2 — Ch. 23 : geler un réseau et remplacer sa tête en PyTorch

- a) :
- b) :
- c) :
- d) :
- e) :


### B1.R3 — Ch. 21 : ce que détectent les filtres des premières et des dernières couches

- a) :
- b) :
- c) :
- d) :
- e) :


## ✏️ ∂ 🧮 Papier-crayon

### Ex B1.1 — Nouvelle tête : formes des tenseurs de l'image 32×32 redimensionnée jusqu'aux logits

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


### Ex B1.2 — k plus proches voisins cosinus sur six embeddings donnés

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :


### Ex B1.3 — Learning rates discriminatifs et warmup : le learning rate de chaque groupe, pas à pas

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


### Ex B1.4 — Compter les paramètres entraînables : tête seule, dernier bloc, réseau entier

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


### Ex B1.5 — Zéro-shot CLIP à la main : cosinus, logit_scale et softmax

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :

Dans ta copie :


### Ex B1.6 — Rester près des poids pré-entraînés : le gradient de la pénalité L2-SP

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B1.7 — Fermi : pré-entraîner sur ImageNet ou fine-tuner sur 5 000 images (GPU-heures, énergie, coût)

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


## 🗣️ ⚖️ 📄 Réflexion

### Ex B1.8 — Le transfer learning expliqué en cinq lignes

Mes cinq lignes :


### Ex B1.9 — Biais et licences hérités d'un modèle pré-entraîné : un projet d'imagerie médicale

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B1.10 — Yosinski et coll. (2014) : jusqu'où les features se transfèrent-elles ?

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


## 💼 Entretien

### B1.E1 — Feature extraction ou fine-tuning : comment choisis-tu ?


### B1.E2 — Tu as 500 images médicales étiquetées : quelle stratégie de transfer learning ?


### B1.E3 — Pourquoi un learning rate plus faible pour les couches pré-entraînées ?


### B1.E4 — Qu'est-ce qu'un modèle de fondation, et que change le zéro-shot ?


### B1.E5 — Comment vérifier qu'un modèle pré-entraîné peut servir dans un produit commercial ?


## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook.

- **B1.11** :
- **B1.12** :
- **B1.13** :
- **B1.14** :
- **B1.15** :
- **B1.16** :
- **B1.17** :
- **B1.18** :
- **B1.19** :
- **B1.20** :
- **B1.21** :
- **B1.22** :
- **B1.23** :
- **B1.24** :
- **B1.25** :
- **B1.26** :
- **B1.27** :
- **B1.28** :
