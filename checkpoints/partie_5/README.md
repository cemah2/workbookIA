# Checkpoint V — Architectures : CNN, RNN, PyTorch en pratique (chapitres 21 à 24)

Le checkpoint clôt la partie V. Il vérifie que les architectures tiennent avant la partie VI : calculer les formes, les paramètres et le champ récepteur d'un CNN, dérouler une couche récurrente et un pas de LSTM, lire les sorties de `nn.LSTM`, écrire une boucle d'entraînement PyTorch avec early stopping, diagnostiquer un entraînement d'après ses courbes, choisir une architecture et comparer deux modèles honnêtement. Compte environ **13 h 30** : la synthèse (1 h 30), l'examen blanc (2 h), le mini-projet (10 h), sans compter la correction et la remédiation.

| Fichier | Contenu | Quand |
|---|---|---|
| [`05_synthese.md`](05_synthese.md) | carte mentale, les 22 formules clés et leurs fonctions `mylearn`, diagnostiquer formes et courbes, le workflow PyTorch et ses pièges, les choix par défaut, le vocabulaire à dire à l'oral | **avant** l'examen |
| [`01_examen_sujet.md`](01_examen_sujet.md) | le sujet de l'examen blanc : 14 questions, 116 minutes, noté sur 20 | le jour de l'examen |
| [`02_examen_notebook.ipynb`](02_examen_notebook.ipynb) | partie A : les deux questions de code (CP5.5 en NumPy, CP5.6 en PyTorch) ; partie B : la vérification automatique, après l'examen | pendant, puis après |
| [`04_mes_reponses.md`](04_mes_reponses.md) | ta copie : une section par question, et le tableau de ta note | pendant, puis après |
| [`03_examen_corrige.md`](03_examen_corrige.md) | le corrigé détaillé, le barème sur 20 et la remédiation | **après** l'examen |
| [`03_examen_solutions.ipynb`](03_examen_solutions.ipynb) | les solutions exécutées des questions de code (CP5.5 comparée à `nn.RNN`) | après l'examen |
| [`../../projets/partie_5_classifieur_cifar10/`](../../projets/partie_5_classifieur_cifar10/) | le mini-projet : un classifieur CIFAR-10 de bout en bout, de la baseline au réseau résiduel, avec son journal d'expériences, son analyse d'erreurs et sa fiche de modèle, pour ton portfolio | après l'examen |

## Déroulé conseillé

1. **Prépare** : `python tools/start_chapter.py CP5` copie dans `mon_travail/` le notebook de l'examen, ta copie (`04_mes_reponses.md`) et le kit de départ du mini-projet. Ce sont ces copies que tu remplis : les fichiers de ce dossier sont mis à jour par Claude.
2. **Révise** avec la synthèse (environ 90 minutes), un autre jour que l'examen si possible.
3. **Passe l'examen en conditions réelles** : 116 minutes d'une traite, livre, fiches et corrigés fermés, sans assistant IA (les règles sont en tête du sujet). Les deux questions de code se font dans la partie A de ta copie du notebook (exécute d'abord sa cellule de données : elle charge 3 072 images de Fashion-MNIST) ; le reste, dans `04_mes_reponses.md` ou sur papier.
4. **Vérifie** : passe `EXAM_OVER` à `True` dans la partie B du notebook. Elle vérifie ton code (CP5.6 réentraîne le CNN plusieurs fois, une trentaine de secondes) et les réponses chiffrées de ta copie (✅ ou ❌ avec une piste).
5. **Note ta copie** avec le barème du corrigé : il donne aussi les points des démarches, des justifications et des questions rédigées. Remplis le tableau en fin de copie, avec le temps réellement passé sur chaque question : ces temps aideront à calibrer les examens suivants.
6. **Remédie** : pour chaque question où tu as eu moins de la moitié des points, refais les exercices indiqués par le corrigé ; une semaine plus tard, refais ces questions sans le corrigé.
7. **Le mini-projet** : commence par son cahier des charges.

Tu peux aussi faire corriger ta copie par Claude (prompt P9) : il lit `mon_travail/checkpoints/partie_5/`, sans jamais y écrire.

Reporte tes notes (examen et mini-projet) et tes points faibles dans la section CP5 de `mon_travail/suivi/auto_evaluation.md`, ajoute une ligne dans `journal.md`, puis coche le checkpoint dans ton tableau de bord.
