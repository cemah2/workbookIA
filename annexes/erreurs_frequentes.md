# Erreurs fréquentes

Les messages d'erreur rencontrés le plus souvent, avec leur cause et la solution. Complété au fil des chapitres (et après chaque correction P9). Pour les problèmes d'installation, voir aussi `00_setup/INSTALL_LOCAL.md` et `00_setup/COLAB.md`.

**Méthode face à une erreur Python** : lis la **dernière ligne** du message (le type d'erreur et sa description), puis remonte jusqu'à la première ligne qui mentionne **ton** code.

## Mise en place et outils

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'wb'` | la cellule de setup n'a pas été exécutée, ou l'environnement virtuel n'est pas activé | exécute la cellule de setup (Colab) ; en local, active `.venv` et fais `pip install -e .` |
| `⏳ Ex 3.4 : pas encore fait` | la réponse vaut encore `...`, ou la fonction lève `NotImplementedError` | ce n'est pas une erreur : écris ton code, puis relance la cellule |
| `❓ Ex 3.4 : aucune réponse enregistrée` | ID mal tapé, ou `answers.json` pas à jour | vérifie l'ID dans l'énoncé, puis `git pull` |
| `ℹ️ Ta librairie mylearn n'existe pas encore` | `mon_travail/mylearn/` n'a pas été créé | `python tools/start_chapter.py --init` (ou le numéro du chapitre) |
| tests `SKIPPED ... n'existe pas encore` | le module mylearn du chapitre n'a pas été copié | `python tools/start_chapter.py <chapitre>` |
| test `FAILED ... ⏳ pas encore implémenté` | la fonction contient encore `raise NotImplementedError` | implémente-la |
| `git pull` refuse de s'exécuter | un fichier hors de `mon_travail/` a été modifié | `git status`, puis `git restore <fichier>` (copie d'abord ta modification si tu y tiens) |
| `fatal: Need to specify how to reconcile divergent branches` | tu as des commits locaux et Claude en a poussé d'autres | `git config --global pull.rebase true` et `git config --global rebase.autoStash true`, puis `git pull` |

## Python et NumPy
*(à compléter à partir du ch. 0A)*

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| | | |

## pandas
*(à compléter)*

## scikit-learn
*(à compléter)*

## PyTorch
*(à compléter à partir du ch. 20)*

## Erreurs de raisonnement (ML)
*(fuite de données, évaluation sur l'entraînement, classes déséquilibrées… : à compléter)*
