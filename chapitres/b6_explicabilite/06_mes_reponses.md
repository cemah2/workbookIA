# B6 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/b6_explicabilite/06_mes_reponses.md` (créée par `python tools/start_chapter.py B6`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_B6_Q1a = ...`, `answer_B6_1a = ...`) pour les vérifier.

## 🧠 Quiz

### B6.Q1 — Interprétable par construction ou expliqué après coup ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q2 — Explication globale ou locale : quel outil répond à quelle question ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q3 — Dépendance partielle et courbes ICE : ce que montre la moyenne, ce qu'elle cache

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q4 — Les quatre propriétés de Shapley : efficacité, symétrie, joueur nul, additivité

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q5 — Changer le fond (background) change les valeurs SHAP : vrai ou faux

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


### B6.Q6 — Features corrélées : ce que SHAP répartit et ce qu'il ne dit pas

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q7 — Gradient, gradient × entrée, gradients intégrés : pourquoi une ligne de référence ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q8 — Grad-CAM : pourquoi la dernière couche convolutive ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q9 — Une carte de saillance qui ne dépend pas du modèle : que prouve le test de randomisation ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q10 — Sources de biais : données, labels, variables de substitution, boucles de rétroaction

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q11 — Parité démographique, égalité des chances, calibration : peut-on tout avoir ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.Q12 — AI Act : dans quelle catégorie de risque tombe un scoring de crédit ?

- a) :
- b) :
- c) :
- d) :
- e) :
- f) :


## 🔁 Rappels

### B6.R1 — Ch. 21 : visualiser ce qu'a appris un CNN, et l'attaque FGSM

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.R2 — Ch. 14 : feature_importances_ contre importance par permutation

- a) :
- b) :
- c) :
- d) :
- e) :


### B6.R3 — Ch. 3 : TPR, FPR et precision à partir d'une matrice de confusion

- a) :
- b) :
- c) :
- d) :
- e) :


## ✏️ ∂ 🧮 Papier-crayon

### Ex B6.1 — Valeurs de Shapley exactes à la main : trois features, huit coalitions

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

Dans ta copie :



### Ex B6.2 — Modèle linéaire et fond indépendant : φ_i = w_i (x_i − E[x_i])

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


**8.**



### Ex B6.3 — Gradients intégrés à la main sur f(x) = x₁² + 3x₂ et vérification de la complétude

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



### Ex B6.4 — Grad-CAM à la main : deux cartes 3×3 et leurs gradients

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



### Ex B6.5 — Audit à la main : deux groupes, deux matrices de confusion, cinq indicateurs

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

Dans ta copie :



### Ex B6.6 — Impossibilité : taux de base différents, calibration et égalité des erreurs incompatibles

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


**8.**


**9.**



### Ex B6.7 — Fermi : coût de Shapley exact, de KernelSHAP et de TreeSHAP pour 10, 20 et 50 features

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**



## 🗣️ ⚖️ 📄 Réflexion

### Ex B6.8 — Expliquer une valeur SHAP à un client dont le prêt est refusé, en cinq lignes

Mes cinq lignes :



### Ex B6.9 — Scoring de crédit : le code postal est-il une variable de substitution ? Droits du client, obligations de l'entreprise

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**



### Ex B6.10 — Reconnaissance faciale : erreurs par sous-groupe et décision de déploiement

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**



### Ex B6.11 — Lundberg et Lee (2017) : « A Unified Approach to Interpreting Model Predictions »

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


**8.**


**9.**


**10.**



## 💼 Entretien

### B6.E1 — Comment expliquer les prédictions d'un gradient boosting à une équipe métier ?



### B6.E2 — Qu'est-ce qu'une valeur SHAP, et quelles sont ses limites ?



### B6.E3 — Comment audites-tu un modèle pour détecter des biais ?



### B6.E4 — Grad-CAM : principe, usage et limites



### B6.E5 — Retirer la variable sensible suffit-il à rendre un modèle équitable ?



## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook.

- **B6.12** :
- **B6.13** :
- **B6.14** :
- **B6.15** :
- **B6.16** :
- **B6.17** :
- **B6.18** :
- **B6.19** :
- **B6.20** :
- **B6.21** :
- **B6.22** :
- **B6.23** :
- **B6.24** :
- **B6.25** :
- **B6.26** :
- **B6.27** :
- **B6.28** :
- **B6.29** :
