# PARCOURS

> Généré par `python tools/syllabus.py build` à partir du syllabus (`docs/SYLLABUS.md`). Chaque exercice y porte ses parcours (colonne « Parcours » : R, M, C).

Quel que soit le parcours : fiche du chapitre, flashcards chaque jour, **checkpoints, mini-projets et projet final en entier** (ils font partie des quatre parcours). Les listes ci-dessous donnent les **exercices** à faire ; les plages `18.Q1–Q12` se lisent « de 18.Q1 à 18.Q12 ».

**Règles des prérequis.** Un exercice de notebook qui réutilise le code d'un exercice de notebook antérieur (fonction `mylearn`, modèle, données préparées) a toujours ce prérequis dans son parcours : c'est vérifié automatiquement. Les autres prérequis sont des connaissances : quand l'un d'eux n'est pas dans ton parcours, lis son corrigé (`05_solutions`), compté pour un tiers de son temps (lignes « corrigés à lire »). Si tu sautes un chapitre entier, ses modules `mylearn` sont pris dans la référence pour les chapitres suivants (voir le README, § « Ta librairie mylearn »).

**Lecture.** Le parcours rapide lit toute la fiche, mais seulement les sections du livre couvertes par ses exercices pratiques (la liste figure dans le SYLLABUS, chapitre par chapitre, et le guide de lecture de la fiche les signale ⏩) ; ses 🧠 et 💼 se préparent avec la fiche. Les autres parcours lisent tout. Les chapitres sans livre (0A, 0B, bonus) sont lus en entier par tous : la fiche y est le cours.

| Parcours | Exercices | Exercices (temps) | Corrigés à lire | Lecture | Flashcards, synthèses, projets | **Total** |
|---|---|---|---|---|---|---|
| [Parcours complet](#complet) | 2019 | 661 h | 0 (0,0 h) | 134 h | 95 h | **891 h** |
| [Parcours rapide](#rapide) | 1299 | 313 h | 58 (7,5 h) | 103 h | 95 h | **519 h** |
| [Parcours orienté maths](#maths) | 782 | 298 h | 8 (0,3 h) | 134 h | 95 h | **528 h** |
| [Parcours orienté code](#code) | 881 | 458 h | 188 (20 h) | 134 h | 95 h | **708 h** |

À 10 h par semaine : complet ≈ 89 semaines, rapide ≈ 52 semaines.

<a id="complet"></a>

## Parcours complet

Tout le workbook, dans l'ordre : l'objectif d'exhaustivité de la bible.

**2019 exercices, 661 h d'exercices, ≈ 891 h au total.**

### Partie 0 · Prérequis

- **0A** Python, notebooks et outils (84 ex., 22 h) : 0A.Q1–Q12, 0A.1–67, 0A.E1–E5
- **0B** Maths du lycée au ML (74 ex., 17 h) : 0B.Q1–Q12, 0B.R1–R3, 0B.1–54, 0B.E1–E5

### Partie I · Fondations

- **1** Introduction au machine learning et au deep learning (43 ex., 9,5 h) : 1.Q1–Q11, 1.R1–R3, 1.1–25, 1.E1–E4
- **2** Hasard et statistiques de base (52 ex., 13 h) : 2.Q1–Q12, 2.R1–R3, 2.1–32, 2.E1–E5
- **3** Probabilités et mesure de la qualité (49 ex., 13 h) : 3.Q1–Q12, 3.R1–R3, 3.1–29, 3.E1–E5
- **4** Règle de Bayes (43 ex., 11 h) : 4.Q1–Q10, 4.R1–R3, 4.1–26, 4.E1–E4
- **5** Courbes et surfaces (42 ex., 11 h) : 5.Q1–Q10, 5.R1–R3, 5.1–25, 5.E1–E4
- **6** Théorie de l'information (46 ex., 12 h) : 6.Q1–Q12, 6.R1–R3, 6.1–27, 6.E1–E4
- **CP1** Checkpoint I — Fondations (14 ex., 1,9 h) : CP1.1–14

### Partie II · Concepts

- **7** Classification (49 ex., 16 h) : 7.Q1–Q11, 7.R1–R3, 7.1–31, 7.E1–E4
- **8** Entraînement et test (46 ex., 13 h) : 8.Q1–Q11, 8.R1–R3, 8.1–27, 8.E1–E5
- **9** Surapprentissage et sous-apprentissage (50 ex., 17 h) : 9.Q1–Q11, 9.R1–R3, 9.1–31, 9.E1–E5
- **10** Neurones (41 ex., 11 h) : 10.Q1–Q9, 10.R1–R3, 10.1–25, 10.E1–E4
- **11** Apprentissage et raisonnement (46 ex., 13 h) : 11.Q1–Q12, 11.R1–R3, 11.1–27, 11.E1–E4
- **CP2** Checkpoint II — Concepts (13 ex., 1,9 h) : CP2.1–13

### Partie III · ML classique

- **12** Préparation des données (52 ex., 17 h) : 12.Q1–Q11, 12.R1–R3, 12.1–33, 12.E1–E5
- **13** Classifieurs (56 ex., 21 h) : 13.Q1–Q12, 13.R1–R3, 13.1–36, 13.E1–E5
- **14** Ensembles (47 ex., 16 h) : 14.Q1–Q11, 14.R1–R3, 14.1–28, 14.E1–E5
- **15** scikit-learn (51 ex., 15 h) : 15.Q1–Q12, 15.R1–R3, 15.1–31, 15.E1–E5
- **CP3** Checkpoint III — ML classique (11 ex., 1,6 h) : CP3.1–11

### Partie IV · Réseaux

- **16** Réseaux feed-forward (41 ex., 11 h) : 16.Q1–Q10, 16.R1–R3, 16.1–24, 16.E1–E4
- **17** Fonctions d'activation (43 ex., 11 h) : 17.Q1–Q12, 17.R1–R3, 17.1–24, 17.E1–E4
- **18** Rétropropagation (49 ex., 17 h) : 18.Q1–Q12, 18.R1–R3, 18.1–29, 18.E1–E5
- **19** Optimiseurs (48 ex., 15 h) : 19.Q1–Q12, 19.R1–R3, 19.1–28, 19.E1–E5
- **20** Deep learning et premiers pas en PyTorch (51 ex., 18 h) : 20.Q1–Q12, 20.R1–R3, 20.1–31, 20.E1–E5
- **CP4** Checkpoint IV — Réseaux de neurones (12 ex., 1,9 h) : CP4.1–12

### Partie V · Architectures

- **21** Réseaux convolutifs (CNN) (52 ex., 20 h) : 21.Q1–Q12, 21.R1–R3, 21.1–32, 21.E1–E5
- **22** Réseaux récurrents (RNN, LSTM, GRU) (49 ex., 16 h) : 22.Q1–Q12, 22.R1–R3, 22.1–29, 22.E1–E5
- **23** PyTorch en pratique 1 : du jeu de données au modèle sauvegardé (49 ex., 14 h) : 23.Q1–Q12, 23.R1–R3, 23.1–29, 23.E1–E5
- **24** PyTorch en pratique 2 : améliorer, chercher, CNN et RNN (51 ex., 20 h) : 24.Q1–Q12, 24.R1–R3, 24.1–31, 24.E1–E5
- **CP5** Checkpoint V — Architectures (CNN, RNN, PyTorch en pratique) (10 ex., 1,8 h) : CP5.1–10

### Partie VI · Génératif et RL

- **25** Autoencodeurs et VAE (59 ex., 22 h) : 25.Q1–Q12, 25.R1–R3, 25.1–39, 25.E1–E5
- **26** Apprentissage par renforcement (58 ex., 20 h) : 26.Q1–Q12, 26.R1–R3, 26.1–38, 26.E1–E5
- **27** Réseaux antagonistes génératifs (GAN) (46 ex., 16 h) : 27.Q1–Q10, 27.R1–R3, 27.1–28, 27.E1–E5
- **28** Applications créatives (46 ex., 15 h) : 28.Q1–Q10, 28.R1–R3, 28.1–29, 28.E1–E4
- **29** Datasets et préparation du projet final (31 ex., 7,9 h) : 29.Q1–Q8, 29.R1–R3, 29.1–16, 29.E1–E4
- **CP6** Checkpoint VI — Génératif et apprentissage par renforcement (13 ex., 1,9 h) : CP6.1–13

### Partie VII · Bonus

- **B1** Transfer learning et modèles pré-entraînés (47 ex., 16 h) : B1.Q1–Q11, B1.R1–R3, B1.1–28, B1.E1–E5
- **B2** Tokenisation et embeddings (51 ex., 15 h) : B2.Q1–Q12, B2.R1–R3, B2.1–31, B2.E1–E5
- **B3** Attention et Transformers : un mini-GPT from scratch (50 ex., 19 h) : B3.Q1–Q12, B3.R1–R3, B3.1–30, B3.E1–E5
- **B4** LLM en pratique : Hugging Face, prompting, RAG et LoRA (50 ex., 18 h) : B4.Q1–Q12, B4.R1–R3, B4.1–30, B4.E1–E5
- **B5** Modèles de diffusion : un DDPM minimal (50 ex., 18 h) : B5.Q1–Q11, B5.R1–R3, B5.1–31, B5.E1–E5
- **B6** Explicabilité, équité et éthique (49 ex., 16 h) : B6.Q1–Q12, B6.R1–R3, B6.1–29, B6.E1–E5
- **B7** Du notebook à la production (44 ex., 15 h) : B7.Q1–Q11, B7.R1–R3, B7.1–25, B7.E1–E5
- **B8** RL moderne : DQN, gradient de politique, PPO et RLHF (48 ex., 18 h) : B8.Q1–Q12, B8.R1–R3, B8.1–28, B8.E1–E5

### Partie PF · Projet final

- **PF** Projet final : un projet de bout en bout sur ton propre dataset (13 ex., 41 h) : PF.1–13

<a id="rapide"></a>

## Parcours rapide

L'essentiel pour être employable (data scientist, ML engineer) : concepts centraux, pratique scikit-learn et PyTorch, toutes les questions d'entretien, les implémentations clés. On peut revenir plus tard sur le reste.

**1299 exercices, 313 h d'exercices, ≈ 519 h au total.**

### Partie 0 · Prérequis

- **0A** Python, notebooks et outils (49 ex., 12 h) : 0A.Q1–Q6, 0A.Q10–Q12, 0A.5, 0A.7–8, 0A.10, 0A.13–14, 0A.16, 0A.18, 0A.20–28, 0A.30–36, 0A.44, 0A.47, 0A.49, 0A.51–54, 0A.57, 0A.61–62, 0A.66, 0A.E1–E5
  - corrigés à lire : 0A.1–4, 0A.12, 0A.55
- **0B** Maths du lycée au ML (44 ex., 8,8 h) : 0B.Q1–Q12, 0B.R1–R3, 0B.3–4, 0B.6, 0B.8–11, 0B.15, 0B.18, 0B.20, 0B.22–23, 0B.25, 0B.27, 0B.31–32, 0B.39–40, 0B.42–45, 0B.52–53, 0B.E1–E5
  - corrigés à lire : 0B.1, 0B.5, 0B.7, 0B.26, 0B.29, 0B.41

### Partie I · Fondations

- **1** Introduction au machine learning et au deep learning (28 ex., 4,3 h) : 1.Q1–Q11, 1.R1–R3, 1.1, 1.6–7, 1.9–11, 1.14, 1.16, 1.18, 1.23, 1.E1–E4
- **2** Hasard et statistiques de base (29 ex., 5,8 h) : 2.Q1–Q3, 2.Q5–Q6, 2.Q8–Q10, 2.Q12, 2.R1–R3, 2.1, 2.4, 2.7, 2.9–10, 2.13, 2.15, 2.21–22, 2.24, 2.26–27, 2.E1–E5
- **3** Probabilités et mesure de la qualité (29 ex., 5,7 h) : 3.Q1–Q4, 3.Q6–Q12, 3.R1–R3, 3.2, 3.7, 3.9–10, 3.15–16, 3.20, 3.22, 3.24, 3.27, 3.E1–E5
- **4** Règle de Bayes (23 ex., 4,1 h) : 4.Q1, 4.Q3, 4.Q5–Q6, 4.Q8–Q10, 4.R1–R3, 4.1, 4.4, 4.9–10, 4.14–16, 4.18, 4.23, 4.E1–E4
- **5** Courbes et surfaces (23 ex., 4,1 h) : 5.Q1, 5.Q3–Q5, 5.Q7–Q8, 5.Q10, 5.R1–R3, 5.1–3, 5.8, 5.11, 5.15, 5.17–18, 5.21, 5.E1–E4
- **6** Théorie de l'information (25 ex., 4,5 h) : 6.Q1–Q3, 6.Q6, 6.Q8, 6.Q10–Q12, 6.R1–R3, 6.1, 6.3, 6.5, 6.9, 6.12–13, 6.16, 6.18, 6.20, 6.22, 6.E1–E4
- **CP1** Checkpoint I — Fondations (14 ex., 1,9 h) : CP1.1–14

### Partie II · Concepts

- **7** Classification (32 ex., 7,4 h) : 7.Q1–Q11, 7.R1–R3, 7.1, 7.3, 7.8, 7.10–14, 7.17–18, 7.25–26, 7.28–29, 7.E1–E4
  - corrigés à lire : 7.7, 7.20, 7.23
- **8** Entraînement et test (33 ex., 8,2 h) : 8.Q1–Q11, 8.R1–R3, 8.1, 8.3, 8.7, 8.9, 8.11, 8.13–14, 8.16, 8.18, 8.21–22, 8.24–26, 8.E1–E5
  - corrigés à lire : 8.23
- **9** Surapprentissage et sous-apprentissage (32 ex., 6,6 h) : 9.Q1–Q11, 9.R1–R3, 9.1, 9.5, 9.8–9, 9.12, 9.14–18, 9.20–22, 9.E1–E5
  - corrigés à lire : 9.2–3, 9.11, 9.24
- **10** Neurones (26 ex., 4,5 h) : 10.Q1–Q9, 10.R1–R3, 10.1, 10.3–4, 10.8, 10.12–14, 10.18–19, 10.21, 10.E1–E4
  - corrigés à lire : 10.2, 10.6
- **11** Apprentissage et raisonnement (29 ex., 5,0 h) : 11.Q1–Q12, 11.R1–R3, 11.2–3, 11.7–10, 11.17, 11.19–21, 11.E1–E4
  - corrigés à lire : 11.6, 11.11, 11.26
- **CP2** Checkpoint II — Concepts (13 ex., 1,9 h) : CP2.1–13

### Partie III · ML classique

- **12** Préparation des données (36 ex., 7,8 h) : 12.Q1–Q11, 12.R1–R3, 12.1–2, 12.4, 12.9–11, 12.13–17, 12.19–20, 12.22–23, 12.28, 12.31, 12.E1–E5
  - corrigés à lire : 12.7
- **13** Classifieurs (33 ex., 7,0 h) : 13.Q1–Q12, 13.R1–R3, 13.1–2, 13.10–14, 13.16, 13.19, 13.23, 13.30–31, 13.33, 13.E1–E5
  - corrigés à lire : 13.4
- **14** Ensembles (29 ex., 5,4 h) : 14.Q1–Q11, 14.R1–R3, 14.1, 14.5, 14.7–9, 14.11, 14.13, 14.15, 14.20, 14.26, 14.E1–E5
  - corrigés à lire : 14.4
- **15** scikit-learn (35 ex., 7,5 h) : 15.Q1–Q12, 15.R1–R3, 15.1–2, 15.6–7, 15.9–10, 15.12, 15.17–18, 15.22, 15.24–26, 15.28, 15.30, 15.E1–E5
  - corrigés à lire : 15.4
- **CP3** Checkpoint III — ML classique (11 ex., 1,6 h) : CP3.1–11

### Partie IV · Réseaux

- **16** Réseaux feed-forward (23 ex., 3,7 h) : 16.Q1–Q4, 16.Q8–Q10, 16.R1–R3, 16.1–3, 16.8–9, 16.12–14, 16.17, 16.E1–E4
  - corrigés à lire : 16.23
- **17** Fonctions d'activation (24 ex., 5,0 h) : 17.Q1, 17.Q3, 17.Q6, 17.Q8–Q9, 17.Q11–Q12, 17.R1–R3, 17.1–2, 17.6, 17.11–15, 17.20, 17.23, 17.E1–E4
- **18** Rétropropagation (31 ex., 6,6 h) : 18.Q1–Q4, 18.Q6–Q11, 18.R1–R3, 18.1–4, 18.9–10, 18.12, 18.14–16, 18.18–19, 18.22, 18.E1–E5
  - corrigés à lire : 18.24
- **19** Optimiseurs (29 ex., 5,5 h) : 19.Q1–Q3, 19.Q5, 19.Q7–Q9, 19.Q11–Q12, 19.R1–R3, 19.1, 19.4–5, 19.9–11, 19.14, 19.16, 19.18–19, 19.23–24, 19.E1–E5
  - corrigés à lire : 19.25
- **20** Deep learning et premiers pas en PyTorch (32 ex., 8,0 h) : 20.Q1–Q7, 20.Q11–Q12, 20.R1–R3, 20.1–3, 20.10–11, 20.13, 20.15, 20.17–19, 20.23–25, 20.27–28, 20.E1–E5
- **CP4** Checkpoint IV — Réseaux de neurones (12 ex., 1,9 h) : CP4.1–12

### Partie V · Architectures

- **21** Réseaux convolutifs (CNN) (29 ex., 6,2 h) : 21.Q1–Q6, 21.Q11–Q12, 21.R1–R3, 21.1–4, 21.8–10, 21.12, 21.22–24, 21.27, 21.29, 21.E1–E5
- **22** Réseaux récurrents (RNN, LSTM, GRU) (29 ex., 6,0 h) : 22.Q1–Q3, 22.Q5–Q6, 22.Q8, 22.Q10, 22.Q12, 22.R1–R3, 22.1–3, 22.5, 22.8–9, 22.12–13, 22.15, 22.20–23, 22.E1–E5
- **23** PyTorch en pratique 1 : du jeu de données au modèle sauvegardé (28 ex., 6,9 h) : 23.Q6, 23.Q9–Q12, 23.R1–R3, 23.1, 23.3–4, 23.9, 23.12–13, 23.15–19, 23.21–23, 23.25, 23.E1–E5
  - corrigés à lire : 23.2, 23.6–7
- **24** PyTorch en pratique 2 : améliorer, chercher, CNN et RNN (34 ex., 9,5 h) : 24.Q1–Q4, 24.Q7–Q8, 24.Q10–Q11, 24.R1–R3, 24.1–2, 24.4, 24.7–9, 24.11–14, 24.16–20, 24.22–23, 24.28, 24.E1–E5
  - corrigés à lire : 24.6
- **CP5** Checkpoint V — Architectures (CNN, RNN, PyTorch en pratique) (10 ex., 1,8 h) : CP5.1–10

### Partie VI · Génératif et RL

- **25** Autoencodeurs et VAE (41 ex., 9,1 h) : 25.Q1–Q12, 25.R1–R3, 25.1, 25.3, 25.5, 25.9–12, 25.14–16, 25.20–21, 25.24, 25.26, 25.28–31, 25.34–35, 25.37, 25.E1–E5
  - corrigés à lire : 25.6–7
- **26** Apprentissage par renforcement (39 ex., 7,6 h) : 26.Q1–Q12, 26.R1–R3, 26.1–4, 26.9–11, 26.13–14, 26.16, 26.19–23, 26.28, 26.31, 26.34, 26.36, 26.E1–E5
  - corrigés à lire : 26.5
- **27** Réseaux antagonistes génératifs (GAN) (28 ex., 4,8 h) : 27.Q1–Q10, 27.R1–R3, 27.1, 27.3–4, 27.7–9, 27.12–14, 27.20, 27.E1–E5
- **28** Applications créatives (28 ex., 5,1 h) : 28.Q1–Q10, 28.R1–R3, 28.1–2, 28.6, 28.8–9, 28.11–13, 28.18, 28.20, 28.23, 28.E1–E4
- **29** Datasets et préparation du projet final (26 ex., 5,7 h) : 29.Q1–Q8, 29.R1–R3, 29.4–5, 29.7–15, 29.E1–E4
  - corrigés à lire : 29.6
- **CP6** Checkpoint VI — Génératif et apprentissage par renforcement (13 ex., 1,9 h) : CP6.1–13

### Partie VII · Bonus

- **B1** Transfer learning et modèles pré-entraînés (33 ex., 7,8 h) : B1.Q1–Q10, B1.R1–R3, B1.1, B1.4–5, B1.7–9, B1.11, B1.13, B1.17–20, B1.24, B1.26–27, B1.E1–E5
  - corrigés à lire : B1.2–3
- **B2** Tokenisation et embeddings (33 ex., 7,1 h) : B2.Q1–Q2, B2.Q4–Q11, B2.R1–R3, B2.1, B2.3, B2.5, B2.8, B2.10–11, B2.13–18, B2.23, B2.28, B2.30, B2.E1–E5
  - corrigés à lire : B2.2, B2.4
- **B3** Attention et Transformers : un mini-GPT from scratch (32 ex., 8,7 h) : B3.Q1–Q7, B3.Q9–Q11, B3.R1–R3, B3.1–3, B3.5, B3.8–9, B3.12, B3.14–15, B3.17–20, B3.23, B3.E1–E5
  - corrigés à lire : B3.4, B3.7
- **B4** LLM en pratique : Hugging Face, prompting, RAG et LoRA (37 ex., 10 h) : B4.Q1–Q12, B4.R1–R3, B4.1, B4.3–4, B4.7–9, B4.12, B4.14–15, B4.17–22, B4.25–26, B4.E1–E5
  - corrigés à lire : B4.2, B4.5–6
- **B5** Modèles de diffusion : un DDPM minimal (34 ex., 8,4 h) : B5.Q1–Q3, B5.Q5–Q11, B5.R1–R3, B5.1–2, B5.4, B5.8–10, B5.12–14, B5.16–17, B5.19, B5.21–24, B5.E1–E5
  - corrigés à lire : B5.5–7
- **B6** Explicabilité, équité et éthique (33 ex., 9,1 h) : B6.Q1–Q5, B6.Q8–Q12, B6.R1–R3, B6.1, B6.5, B6.8–10, B6.12, B6.14–15, B6.18, B6.20, B6.23–24, B6.26–28, B6.E1–E5
  - corrigés à lire : B6.2–4
- **B7** Du notebook à la production (28 ex., 7,3 h) : B7.Q1–Q8, B7.Q11, B7.R1–R3, B7.5–7, B7.9–11, B7.14–15, B7.17–18, B7.23, B7.E1–E5
  - corrigés à lire : B7.3
- **B8** RL moderne : DQN, gradient de politique, PPO et RLHF (27 ex., 4,6 h) : B8.Q1, B8.Q3–Q12, B8.R1–R3, B8.1, B8.4–5, B8.8–10, B8.12–13, B8.E1–E5
  - corrigés à lire : B8.2–3

### Partie PF · Projet final

- **PF** Projet final : un projet de bout en bout sur ton propre dataset (13 ex., 41 h) : PF.1–13

<a id="maths"></a>

## Parcours orienté maths

Pour comprendre en profondeur : calculs à la main, démonstrations, estimations de Fermi et implémentations à forte composante mathématique.

**782 exercices, 298 h d'exercices, ≈ 528 h au total.**

### Partie 0 · Prérequis

- **0A** Python, notebooks et outils (34 ex., 9,3 h) : 0A.Q11, 0A.1–8, 0A.14, 0A.16, 0A.18, 0A.20–28, 0A.30, 0A.32–35, 0A.44, 0A.47, 0A.49, 0A.51–53, 0A.63–64
- **0B** Maths du lycée au ML (64 ex., 16 h) : 0B.Q1–Q12, 0B.1–30, 0B.32–44, 0B.46–54
  - corrigés à lire : 0B.R1–R3

### Partie I · Fondations

- **1** Introduction au machine learning et au deep learning (9 ex., 2,0 h) : 1.R1, 1.R3, 1.1–5, 1.11, 1.16
- **2** Hasard et statistiques de base (29 ex., 9,2 h) : 2.Q2, 2.Q4–Q5, 2.Q7, 2.Q11, 2.R3, 2.1–8, 2.11, 2.13, 2.15–16, 2.18–19, 2.21–23, 2.25–28, 2.30, 2.32
  - corrigés à lire : 2.Q9, 2.12
- **3** Probabilités et mesure de la qualité (23 ex., 8,6 h) : 3.Q4, 3.R1–R2, 3.1–8, 3.12–13, 3.15–16, 3.19–21, 3.24–28
- **4** Règle de Bayes (19 ex., 5,6 h) : 4.Q6–Q7, 4.R1, 4.R3, 4.1–8, 4.14, 4.16, 4.18, 4.21–22, 4.24–25
- **5** Courbes et surfaces (22 ex., 5,8 h) : 5.Q3, 5.Q5–Q7, 5.Q9, 5.R2–R3, 5.1–7, 5.9, 5.11–13, 5.15, 5.17–18, 5.24
- **6** Théorie de l'information (23 ex., 7,0 h) : 6.Q5, 6.Q8, 6.Q11, 6.R1–R3, 6.1–8, 6.10, 6.12–13, 6.16, 6.20, 6.22–24, 6.26
- **CP1** Checkpoint I — Fondations (14 ex., 1,9 h) : CP1.1–14

### Partie II · Concepts

- **7** Classification (18 ex., 5,4 h) : 7.Q9, 7.Q11, 7.R1–R3, 7.1–7, 7.13, 7.15, 7.19–21, 7.28
- **8** Entraînement et test (16 ex., 6,0 h) : 8.Q10, 8.R2, 8.1–6, 8.8, 8.11, 8.13–14, 8.21–23, 8.26
- **9** Surapprentissage et sous-apprentissage (21 ex., 9,3 h) : 9.Q7, 9.R2–R3, 9.1–7, 9.9, 9.11, 9.14–17, 9.23–24, 9.26–27, 9.30
- **10** Neurones (15 ex., 5,5 h) : 10.R2, 10.1–7, 10.9, 10.12, 10.14–15, 10.17, 10.21, 10.23
- **11** Apprentissage et raisonnement (14 ex., 5,2 h) : 11.R3, 11.1–8, 11.15, 11.19–21, 11.23
- **CP2** Checkpoint II — Concepts (13 ex., 1,9 h) : CP2.1–13

### Partie III · ML classique

- **12** Préparation des données (20 ex., 5,9 h) : 12.Q5, 12.Q7, 12.Q9, 12.R1–R3, 12.1–8, 12.16, 12.22–24, 12.27, 12.29
- **13** Classifieurs (24 ex., 9,5 h) : 13.Q5–Q6, 13.Q9, 13.Q11, 13.R1–R3, 13.1–9, 13.16–17, 13.20, 13.22, 13.24, 13.26, 13.28, 13.32
- **14** Ensembles (15 ex., 4,3 h) : 14.Q4, 14.Q9, 14.Q11, 14.R1–R3, 14.1–7, 14.13, 14.25
- **15** scikit-learn (15 ex., 3,6 h) : 15.Q8, 15.R1–R3, 15.1–5, 15.9, 15.15, 15.17–20
- **CP3** Checkpoint III — ML classique (11 ex., 1,6 h) : CP3.1–11

### Partie IV · Réseaux

- **16** Réseaux feed-forward (17 ex., 6,0 h) : 16.Q10, 16.R3, 16.1–8, 16.10, 16.13–14, 16.17, 16.19, 16.22–23
- **17** Fonctions d'activation (18 ex., 6,8 h) : 17.R3, 17.1–8, 17.10, 17.12–16, 17.19, 17.21, 17.23
- **18** Rétropropagation (19 ex., 8,8 h) : 18.R1, 18.1–9, 18.11–12, 18.14–16, 18.18, 18.22, 18.26–27
  - corrigés à lire : 18.Q8
- **19** Optimiseurs (17 ex., 6,5 h) : 19.Q6, 19.R1, 19.1–9, 19.12, 19.16–18, 19.23, 19.26
- **20** Deep learning et premiers pas en PyTorch (11 ex., 5,2 h) : 20.1, 20.3–9, 20.12, 20.22, 20.25
  - corrigés à lire : 20.Q11
- **CP4** Checkpoint IV — Réseaux de neurones (12 ex., 1,9 h) : CP4.1–12

### Partie V · Architectures

- **21** Réseaux convolutifs (CNN) (21 ex., 11 h) : 21.Q3, 21.Q9, 21.R2, 21.1–8, 21.11–12, 21.14–15, 21.18, 21.22, 21.27–28, 21.30–31
- **22** Réseaux récurrents (RNN, LSTM, GRU) (14 ex., 5,6 h) : 22.Q6, 22.1–8, 22.11, 22.13, 22.15–16, 22.19
- **23** PyTorch en pratique 1 : du jeu de données au modèle sauvegardé (8 ex., 2,4 h) : 23.R3, 23.1–4, 23.7–8, 23.11
- **24** PyTorch en pratique 2 : améliorer, chercher, CNN et RNN (9 ex., 2,8 h) : 24.R2, 24.1–7, 24.10
- **CP5** Checkpoint V — Architectures (CNN, RNN, PyTorch en pratique) (10 ex., 1,8 h) : CP5.1–10

### Partie VI · Génératif et RL

- **25** Autoencodeurs et VAE (24 ex., 11 h) : 25.Q9, 25.Q12, 25.R3, 25.1–10, 25.13, 25.15–16, 25.20, 25.23, 25.28–30, 25.34, 25.36, 25.38
- **26** Apprentissage par renforcement (19 ex., 6,4 h) : 26.Q6, 26.Q8, 26.1–9, 26.12–14, 26.19–22, 26.26
- **27** Réseaux antagonistes génératifs (GAN) (13 ex., 4,8 h) : 27.Q5, 27.1–7, 27.10–12, 27.18, 27.22
- **28** Applications créatives (13 ex., 5,0 h) : 28.Q7, 28.1–7, 28.10–11, 28.18, 28.23, 28.25
- **29** Datasets et préparation du projet final (13 ex., 5,0 h) : 29.1–3, 29.6–15
  - corrigés à lire : 29.4
- **CP6** Checkpoint VI — Génératif et apprentissage par renforcement (13 ex., 1,9 h) : CP6.1–13

### Partie VII · Bonus

- **B1** Transfer learning et modèles pré-entraînés (11 ex., 4,4 h) : B1.1–7, B1.10–11, B1.13, B1.16
- **B2** Tokenisation et embeddings (17 ex., 6,7 h) : B2.Q12, B2.1–9, B2.12, B2.15–17, B2.19–20, B2.22
- **B3** Attention et Transformers : un mini-GPT from scratch (17 ex., 8,6 h) : B3.Q3, B3.1–8, B3.11–12, B3.14–15, B3.18–20, B3.26
- **B4** LLM en pratique : Hugging Face, prompting, RAG et LoRA (8 ex., 3,2 h) : B4.1–3, B4.5–7, B4.11, B4.25
- **B5** Modèles de diffusion : un DDPM minimal (17 ex., 7,2 h) : B5.Q3, B5.1–8, B5.11–12, B5.14, B5.16–20
- **B6** Explicabilité, équité et éthique (11 ex., 4,6 h) : B6.Q4, B6.1–7, B6.11, B6.14, B6.18
- **B7** Du notebook à la production (5 ex., 1,4 h) : B7.1–5
- **B8** RL moderne : DQN, gradient de politique, PPO et RLHF (13 ex., 4,5 h) : B8.Q6, B8.1–8, B8.11, B8.18–19, B8.25

### Partie PF · Projet final

- **PF** Projet final : un projet de bout en bout sur ton propre dataset (13 ex., 41 h) : PF.1–13

<a id="code"></a>

## Parcours orienté code

Pour devenir solide en implémentation : from scratch, bibliothèques, chasses au bug, défis, expériences et compétences pro.

**881 exercices, 458 h d'exercices, ≈ 708 h au total.**

### Partie 0 · Prérequis

- **0A** Python, notebooks et outils (75 ex., 21 h) : 0A.Q3–Q12, 0A.1–4, 0A.6–8, 0A.10–67
  - corrigés à lire : 0A.5
- **0B** Maths du lycée au ML (25 ex., 7,1 h) : 0B.R1–R3, 0B.32–49, 0B.51–54
  - corrigés à lire : 0B.2–4, 0B.6–9, 0B.12–13, 0B.15, 0B.17–18, 0B.20–23, 0B.25–27, 0B.29–30

### Partie I · Fondations

- **1** Introduction au machine learning et au deep learning (18 ex., 6,1 h) : 1.R2, 1.9–25
  - corrigés à lire : 1.Q10, 1.R3
- **2** Hasard et statistiques de base (20 ex., 8,2 h) : 2.R2, 2.13–26, 2.28–32
  - corrigés à lire : 2.Q9, 2.1–4, 2.7, 2.11–12
- **3** Probabilités et mesure de la qualité (18 ex., 7,6 h) : 3.R3, 3.12–26, 3.28–29
  - corrigés à lire : 3.2–3, 3.5–7
- **4** Règle de Bayes (15 ex., 6,4 h) : 4.12–26
  - corrigés à lire : 4.1, 4.4
- **5** Courbes et surfaces (14 ex., 6,2 h) : 5.11–16, 5.18–25
  - corrigés à lire : 5.1–3, 5.5
- **6** Théorie de l'information (15 ex., 6,9 h) : 6.12–19, 6.21–27
  - corrigés à lire : 6.1, 6.3–6
- **CP1** Checkpoint I — Fondations (14 ex., 1,9 h) : CP1.1–14

### Partie II · Concepts

- **7** Classification (22 ex., 11 h) : 7.9, 7.11–31
  - corrigés à lire : 7.R2–R3, 7.1–7
- **8** Entraînement et test (17 ex., 8,7 h) : 8.11–27
  - corrigés à lire : 8.1, 8.3, 8.6
- **9** Surapprentissage et sous-apprentissage (20 ex., 12 h) : 9.12–31
  - corrigés à lire : 9.1–7, 9.11
- **10** Neurones (14 ex., 6,1 h) : 10.12–25
  - corrigés à lire : 10.4–7
- **11** Apprentissage et raisonnement (15 ex., 7,7 h) : 11.13–27
  - corrigés à lire : 11.2–4, 11.6, 11.8
- **CP2** Checkpoint II — Concepts (13 ex., 1,9 h) : CP2.1–13

### Partie III · ML classique

- **12** Préparation des données (26 ex., 12 h) : 12.R1–R3, 12.11–33
  - corrigés à lire : 12.1–2, 12.4, 12.7
- **13** Classifieurs (27 ex., 16 h) : 13.R1–R3, 13.12–32, 13.34–36
  - corrigés à lire : 13.1, 13.3–8
- **14** Ensembles (22 ex., 11 h) : 14.R1–R3, 14.10–28
  - corrigés à lire : 14.1, 14.3, 14.5–7
- **15** scikit-learn (31 ex., 12 h) : 15.Q2–Q4, 15.Q7, 15.R1–R3, 15.8–31
  - corrigés à lire : 15.1–6
- **CP3** Checkpoint III — ML classique (11 ex., 1,6 h) : CP3.1–11

### Partie IV · Réseaux

- **16** Réseaux feed-forward (13 ex., 5,5 h) : 16.11–21, 16.23–24
  - corrigés à lire : 16.1, 16.3–4
- **17** Fonctions d'activation (13 ex., 5,9 h) : 17.11–20, 17.22–24
  - corrigés à lire : 17.1–2
- **18** Rétropropagation (17 ex., 11 h) : 18.12–21, 18.23–29
  - corrigés à lire : 18.4
- **19** Optimiseurs (15 ex., 8,4 h) : 19.13–22, 19.24–28
  - corrigés à lire : 19.1–2
- **20** Deep learning et premiers pas en PyTorch (18 ex., 12 h) : 20.13–24, 20.26–31
  - corrigés à lire : 20.1
- **CP4** Checkpoint IV — Réseaux de neurones (12 ex., 1,9 h) : CP4.1–12

### Partie V · Architectures

- **21** Réseaux convolutifs (CNN) (21 ex., 14 h) : 21.12–32
  - corrigés à lire : 21.1–7
- **22** Réseaux récurrents (RNN, LSTM, GRU) (18 ex., 10 h) : 22.12–29
  - corrigés à lire : 22.1–2, 22.4–6
- **23** PyTorch en pratique 1 : du jeu de données au modèle sauvegardé (18 ex., 8,7 h) : 23.12–29
  - corrigés à lire : 23.2–4, 23.6–7
- **24** PyTorch en pratique 2 : améliorer, chercher, CNN et RNN (21 ex., 15 h) : 24.11–31
  - corrigés à lire : 24.1–6
- **CP5** Checkpoint V — Architectures (CNN, RNN, PyTorch en pratique) (10 ex., 1,8 h) : CP5.1–10

### Partie VI · Génératif et RL

- **25** Autoencodeurs et VAE (26 ex., 15 h) : 25.14–39
  - corrigés à lire : 25.1, 25.3–8
- **26** Apprentissage par renforcement (26 ex., 15 h) : 26.13–38
  - corrigés à lire : 26.1, 26.3–5, 26.8
- **27** Réseaux antagonistes génératifs (GAN) (18 ex., 11 h) : 27.11–28
  - corrigés à lire : 27.1, 27.3–4, 27.6
- **28** Applications créatives (19 ex., 10 h) : 28.11–29
  - corrigés à lire : 28.2–3
- **29** Datasets et préparation du projet final (10 ex., 4,8 h) : 29.7–16
  - corrigés à lire : 29.4, 29.6
- **CP6** Checkpoint VI — Génératif et apprentissage par renforcement (13 ex., 1,9 h) : CP6.1–13

### Partie VII · Bonus

- **B1** Transfer learning et modèles pré-entraînés (18 ex., 11 h) : B1.11–28
  - corrigés à lire : B1.3, B1.5, B1.10
- **B2** Tokenisation et embeddings (19 ex., 9,8 h) : B2.13–31
  - corrigés à lire : B2.2–4, B2.6–7
- **B3** Attention et Transformers : un mini-GPT from scratch (19 ex., 13 h) : B3.12–30
  - corrigés à lire : B3.2–4, B3.6–7
- **B4** LLM en pratique : Hugging Face, prompting, RAG et LoRA (20 ex., 13 h) : B4.4, B4.12–30
  - corrigés à lire : B4.2, B4.5–6
- **B5** Modèles de diffusion : un DDPM minimal (20 ex., 12 h) : B5.12–31
  - corrigés à lire : B5.1–3, B5.5–7
- **B6** Explicabilité, équité et éthique (18 ex., 10 h) : B6.12–29
  - corrigés à lire : B6.1–5
- **B7** Du notebook à la production (17 ex., 11 h) : B7.9–25
  - corrigés à lire : B7.1–4
- **B8** RL moderne : DQN, gradient de politique, PPO et RLHF (17 ex., 13 h) : B8.12–28
  - corrigés à lire : B8.1–2, B8.4–7

### Partie PF · Projet final

- **PF** Projet final : un projet de bout en bout sur ton propre dataset (13 ex., 41 h) : PF.1–13
