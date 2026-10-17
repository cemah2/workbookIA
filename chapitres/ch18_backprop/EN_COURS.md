# ⏳ Chapitre 18 en cours de génération

Ce chapitre est généré en deux sessions (voir `suivi/PROGRESS.md`).

| Fichier | État |
|---|---|
| `01_fiche.md` + `figures/` | ✅ complet (tout le cours, §18.1 à §18.12.10 : la méthode très lente, le delta d'un neurone, le réseau minuscule et ses trois étapes, les activations et le facteur $f'(z)$, le learning rate et ses plateaux, la discussion ; plus la forme matricielle, les losses calculées sur les logits, la vérification d'un gradient et le micro-autograd, ajoutés par le workbook) |
| `02_exercices.md`, `06_mes_reponses.md` | ✅ complets (quiz, rappels, papier, réflexion, entretien) ; la liste des exercices du notebook est déjà dans `02_exercices.md` |
| `03_notebook.ipynb`, `05_solutions.ipynb` | 🛠️ parties 0 à D écrites ; vérification indépendante en cours (session 37) |
| `04_indices.md`, `05_solutions.md` | 🛠️ complets ; vérification indépendante en cours (session 37) |
| `flashcards.csv` | ✅ complet |
| `mylearn` : `nn/backward.py`, `autograd_mini.py` | ✅ références et tests prêts (`tests/test_ch18_backward.py`, `tests/test_ch18_autograd_mini.py`) ; les exercices qui te les font écrire arrivent avec le notebook |

Tant que ce fichier existe, `python tools/start_chapter.py 18` copie dans `mon_travail/` ton fichier de réponses (`06_mes_reponses.md`) et tes squelettes `mylearn/nn/backward.py` et `mylearn/autograd_mini.py`, mais **pas encore le notebook** : une copie à moitié écrite ne serait jamais complétée, puisque tes fichiers ne sont jamais écrasés. Tu peux déjà lire toute la fiche, faire tous les exercices papier et même commencer les modules (leurs tests : `python -m pytest tests/test_ch18_backward.py tests/test_ch18_autograd_mini.py -q`). `backward.py` s'appuie sur **ton** `nn/activations.py` du ch. 17 (`sigmoid`, `log_softmax`) : si tu ne l'as pas encore, la référence le remplace. Quand le chapitre sera terminé, ce fichier disparaîtra : relance alors la même commande pour obtenir le notebook complet, dont la partie 0 vérifie tes réponses courtes.
