# B5 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/b5_diffusion/06_mes_reponses.md` (créée par `python tools/start_chapter.py B5`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_B5_Q1a = ...`, `answer_B5_1a = ...`) pour les vérifier.

## 🧠 Quiz

### B5.Q1 — Diffusion, VAE, GAN : comment chacun fabrique-t-il une image ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q2 — Pourquoi détruire l'image en mille petits pas de bruit plutôt qu'en un seul ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q3 — ᾱ_t : quelle part de l'image reste-t-il au pas t ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q4 — Prédire le bruit, l'image ou la vitesse : la même information ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q5 — Le pas de temps est une entrée du réseau : pourquoi, et comment l'encoder ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q6 — U-Net : à quoi servent les connexions de saut ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q7 — Échantillonnage DDPM : pourquoi est-il lent et aléatoire ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q8 — DDIM : moins de pas avec le même réseau, vrai ou faux

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q9 — Guidance sans classifieur : ce que règle l'échelle w

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q10 — Un bon FID peut-il cacher de la copie des données d'entraînement ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.Q11 — Diffusion latente, transformers de diffusion, flow matching : ce qui a changé depuis 2020

- a) :
- b) :
- c) :
- d) :
- e) :


## 🔁 Rappels

### B5.R1 — Ch. B3 : l'encodage sinusoïdal des positions

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.R2 — Ch. 27 : ce que mesure la distance de Fréchet et l'effondrement de mode

- a) :
- b) :
- c) :
- d) :
- e) :


### B5.R3 — Ch. 25 : l'astuce de reparamétrisation et le terme KL du VAE

- a) :
- b) :
- c) :
- d) :
- e) :


## ✏️ ∂ 🧮 Papier-crayon

### Ex B5.1 — Guidance sans classifieur : combiner les deux prédictions pour w = 0, 1 et 3

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



### Ex B5.2 — Planning linéaire : β_t, α_t et ᾱ_t pour T = 4, puis ᾱ_T pour T = 1 000

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
- k) :
- l) :

Dans ta copie :



### Ex B5.3 — Rapport signal sur bruit : à quel pas un chiffre devient-il illisible, planning linéaire ou cosinus ?

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



### Ex B5.4 — Un pas DDPM à la main, en dimension 1, avec un bruit prédit donné

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



### Ex B5.5 — Un pas DDIM à la main et l'image estimée x̂_0

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



### Ex B5.6 — Forme fermée : deux pas de bruit gaussien n'en font qu'un, d'où q(x_t | x_0)

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B5.7 — Moyenne et variance du posterior q(x_{t−1} | x_t, x_0) en dimension 1

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B5.8 — Fermi : coût d'échantillonnage, de MNIST sur CPU à une image 1024×1024, et ce que gagnent DDIM et l'espace latent

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


## 🗣️ ⚖️ 📄 Réflexion

### Ex B5.9 — La diffusion expliquée en cinq lignes

Mes cinq lignes :


### Ex B5.10 — Un générateur peut-il recopier ses images d'entraînement ? Mémorisation, droits et traçabilité

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


### Ex B5.11 — Ho, Jain et Abbeel (2020) : « Denoising Diffusion Probabilistic Models »

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

### B5.E1 — Explique le principe d'un modèle de diffusion, de l'entraînement à la génération


### B5.E2 — Pourquoi le réseau prédit-il le bruit plutôt que l'image ?


### B5.E3 — Comment accélérer l'échantillonnage d'un modèle de diffusion ?


### B5.E4 — Qu'est-ce que la guidance sans classifieur et quel compromis règle-t-elle ?


### B5.E5 — Diffusion ou flow matching : qu'est-ce qui a changé dans les générateurs récents ?


## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook.

- **B5.12** :
- **B5.13** :
- **B5.14** :
- **B5.15** :
- **B5.16** :
- **B5.17** :
- **B5.18** :
- **B5.19** :
- **B5.20** :
- **B5.21** :
- **B5.22** :
- **B5.23** :
- **B5.24** :
- **B5.25** :
- **B5.26** :
- **B5.27** :
- **B5.28** :
- **B5.29** :
- **B5.30** :
- **B5.31** :
