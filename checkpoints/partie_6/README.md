# Checkpoint VI — Génératif et apprentissage par renforcement (chapitres 25 à 29)

Le checkpoint clôt la partie VI. Il vérifie que les modèles génératifs et l'apprentissage par renforcement tiennent avant les chapitres bonus et le projet final : tirer un code par reparamétrisation et calculer sa KL, mettre à jour une table Q par Q-learning et par SARSA, justifier le discriminateur optimal et les losses à l'équilibre d'un GAN, calculer une matrice de Gram, écrire la loss d'un VAE et réparer un pas de GAN en PyTorch, diagnostiquer des courbes de VAE, de GAN et d'agent, argumenter un choix de modèle génératif et de dataset. Compte environ **13 h 30** : la synthèse (1 h 30), l'examen blanc (2 h), le mini-projet (10 h), sans compter la correction et la remédiation.

| Fichier | Contenu | Quand |
|---|---|---|
| [`05_synthese.md`](05_synthese.md) | carte mentale, les 18 formules clés et leurs fonctions `mylearn`, le tableau AE, VAE, GAN et diffusion, le RL en une page, le gradient par rapport à l'entrée, la check-list d'un dataset, les pièges, le vocabulaire à dire à l'oral | **avant** l'examen |
| [`01_examen_sujet.md`](01_examen_sujet.md) | le sujet de l'examen blanc : 13 questions, 114 minutes, noté sur 20 | le jour de l'examen |
| [`02_examen_notebook.ipynb`](02_examen_notebook.ipynb) | partie A : les deux questions de code (CP6.8, la loss d'un VAE ; CP6.9, un pas de GAN à réparer) ; partie B : la vérification automatique, après l'examen | pendant, puis après |
| [`04_mes_reponses.md`](04_mes_reponses.md) | ta copie : une section par question, et le tableau de ta note | pendant, puis après |
| [`03_examen_corrige.md`](03_examen_corrige.md) | le corrigé détaillé, le barème sur 20 et la remédiation | **après** l'examen |
| [`03_examen_solutions.ipynb`](03_examen_solutions.ipynb) | les solutions exécutées des questions de code (le pas de GAN réparé, entraîné jusqu'à l'équilibre) | après l'examen |
| [`../../projets/partie_6_generatif/`](../../projets/partie_6_generatif/) | le mini-projet : un banc d'essai qui compare un VAE convolutif et un DCGAN sur Fashion-MNIST, avec un protocole d'évaluation fixé à l'avance, une galerie et une fiche modèle, pour ton portfolio | après l'examen |

## Déroulé conseillé

1. **Prépare** : `python tools/start_chapter.py CP6` copie dans `mon_travail/` le notebook de l'examen, ta copie (`04_mes_reponses.md`) et le kit de départ du mini-projet. Ce sont ces copies que tu remplis : les fichiers de ce dossier sont mis à jour par Claude.
2. **Révise** avec la synthèse (environ 90 minutes), un autre jour que l'examen si possible.
3. **Passe l'examen en conditions réelles** : 114 minutes d'une traite, livre, fiches et corrigés fermés, sans assistant IA (les règles sont en tête du sujet). Les deux questions de code se font dans la partie A de ta copie du notebook (exécute d'abord sa cellule de données) ; le reste, dans `04_mes_reponses.md` ou sur papier.
4. **Vérifie** : passe `EXAM_OVER` à `True` dans la partie B du notebook. Elle vérifie ton code (CP6.9 entraîne le GAN, une dizaine de secondes) et les réponses chiffrées de ta copie (✅ ou ❌ avec une piste).
5. **Note ta copie** avec le barème du corrigé : il donne aussi les points des démarches, des justifications et des questions rédigées. Remplis le tableau en fin de copie, avec le temps réellement passé sur chaque question : ces temps aideront à calibrer les examens suivants.
6. **Remédie** : pour chaque question où tu as eu moins de la moitié des points, refais les exercices indiqués par le corrigé ; une semaine plus tard, refais ces questions sans le corrigé.
7. **Le mini-projet** : commence par son cahier des charges.

Tu peux aussi faire corriger ta copie par Claude (prompt P9) : il lit `mon_travail/checkpoints/partie_6/`, sans jamais y écrire.

Reporte tes notes (examen et mini-projet) et tes points faibles dans la section CP6 de `mon_travail/suivi/auto_evaluation.md`, ajoute une ligne dans `journal.md`, puis coche le checkpoint dans ton tableau de bord.
