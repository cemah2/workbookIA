# Checkpoint IV — Réseaux de neurones (chapitres 16 à 20)

Le checkpoint clôt la partie IV. Il vérifie que les réseaux de neurones tiennent avant la partie V, où viennent les architectures (CNN, RNN, PyTorch en pratique) : propager et rétropropager à la main dans un petit réseau, dérouler les premiers pas des optimiseurs, distinguer l'entraînement et l'évaluation du dropout et de la batchnorm, écrire et réparer une boucle d'entraînement, lire des courbes de loss. Compte environ **11 h 30** : la synthèse (1 h 30), l'examen blanc (2 h), le mini-projet (8 h), sans compter la correction et la remédiation.

| Fichier | Contenu | Quand |
|---|---|---|
| [`05_synthese.md`](05_synthese.md) | carte mentale, les 24 formules clés et leurs fonctions `mylearn`, diagnostiquer un entraînement, la boucle PyTorch et ses pièges, les choix par défaut, le vocabulaire à dire à l'oral | **avant** l'examen |
| [`01_examen_sujet.md`](01_examen_sujet.md) | le sujet de l'examen blanc : 14 questions, 114 minutes, noté sur 20 | le jour de l'examen |
| [`02_examen_notebook.ipynb`](02_examen_notebook.ipynb) | partie A : la question de code (CP4.9) ; partie B : la vérification automatique, après l'examen | pendant, puis après |
| [`04_mes_reponses.md`](04_mes_reponses.md) | ta copie : une section par question, et le tableau de ta note | pendant, puis après |
| [`03_examen_corrige.md`](03_examen_corrige.md) | le corrigé détaillé, le barème sur 20 et la remédiation | **après** l'examen |
| [`03_examen_solutions.ipynb`](03_examen_solutions.ipynb) | la solution exécutée de la question de code, comparée à PyTorch | après l'examen |
| [`../../projets/partie_4_mini_framework/`](../../projets/partie_4_mini_framework/) | le mini-projet : un mini-framework de deep learning en NumPy, assemblé à partir de ta librairie et validé contre PyTorch, pour ton portfolio | après l'examen |

## Déroulé conseillé

1. **Prépare** : `python tools/start_chapter.py CP4` copie dans `mon_travail/` le notebook de l'examen, ta copie (`04_mes_reponses.md`) et le kit de départ du mini-projet. Ce sont ces copies que tu remplis : les fichiers de ce dossier sont mis à jour par Claude.
2. **Révise** avec la synthèse (environ 90 minutes), un autre jour que l'examen si possible.
3. **Passe l'examen en conditions réelles** : 114 minutes d'une traite, livre, fiches et corrigés fermés, sans assistant IA (les règles sont en tête du sujet). La question de code se fait dans la partie A de ta copie du notebook ; le reste, dans `04_mes_reponses.md` ou sur papier.
4. **Vérifie** : passe `EXAM_OVER` à `True` dans la partie B du notebook. Elle vérifie ton code et les réponses chiffrées de ta copie (✅ ou ❌ avec une piste).
5. **Note ta copie** avec le barème du corrigé : il donne aussi les points des démarches, des justifications et des questions rédigées. Remplis le tableau en fin de copie, avec le temps réellement passé sur chaque question : ces temps aideront à calibrer les examens suivants.
6. **Remédie** : pour chaque question où tu as eu moins de la moitié des points, refais les exercices indiqués par le corrigé ; une semaine plus tard, refais ces questions sans le corrigé.
7. **Le mini-projet** : commence par son cahier des charges.

Tu peux aussi faire corriger ta copie par Claude (prompt P9) : il lit `mon_travail/checkpoints/partie_4/`, sans jamais y écrire.

Reporte tes notes (examen et mini-projet) et tes points faibles dans la section CP4 de `mon_travail/suivi/auto_evaluation.md`, ajoute une ligne dans `journal.md`, puis coche le checkpoint dans ton tableau de bord.
