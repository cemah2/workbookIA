# Cheatsheet git

> Aide-mémoire rempli au fil des chapitres (0A, puis chaque chapitre (🛠️)). Une ligne = une commande utile + ce qu'elle fait.

## Configurer

| Code | Effet | Ch. |
|---|---|---|
| `git config --global user.name "Ton Nom"` | ton nom, écrit dans chaque commit (une fois par ordinateur) | 0A |
| `git config --global user.email "ton@email"` | ton e-mail (celui de GitHub) | 0A |
| `git config --global pull.rebase true` puis `git config --global rebase.autoStash true` | `git pull` range tes commits après ceux de Claude, sans conflit inutile | 0A |

## Cloner et mettre à jour (clone, pull)

| Code | Effet | Ch. |
|---|---|---|
| `git clone URL` | copie un dépôt GitHub sur ton ordinateur (une seule fois) | 0A |
| `git pull --rebase --autostash` | récupère les nouveautés (nouveaux chapitres) sans toucher à `mon_travail/` | 0A |

## Suivre ses modifications (status, diff, add, commit)

| Code | Effet | Ch. |
|---|---|---|
| `git status` | l'état des fichiers : non suivis et modifiés (rouge), préparés (vert) | 0A |
| `git diff` | les lignes modifiées pas encore préparées (`git diff --staged` : les lignes préparées ; `git diff HEAD` : tout depuis le dernier commit) | 0A |
| `git add mon_travail/ch00a_python/06_mes_reponses.md` | prépare **ce** fichier pour le prochain commit (évite `git add .`) | 0A |
| `git commit -m "0A: answer exercise 0A.9"` | enregistre la photo des fichiers préparés, avec un message qui dit ce que fait le commit | 0A |
| `.gitignore` (`__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.pt`) | fichiers que git ignore : générés, lourds, secrets (à écrire **avant** le premier commit) | 0A |
| `git check-ignore -v fichier` | quelle ligne du `.gitignore` ignore ce fichier ? | 0A |
| `git rm --cached fichier` | arrête de suivre un fichier (sans le supprimer du disque) | 0A |

## Historique (log)

| Code | Effet | Ch. |
|---|---|---|
| `git log --oneline -3` | les trois derniers commits : identifiant court et message | 0A |

## Annuler (restore, revert)

| Code | Effet | Ch. |
|---|---|---|
| `git restore fichier` | annule les modifications non préparées d'un fichier (⚠️ définitif) ; `git restore --staged fichier` retire un fichier de la préparation | 0A |

## Branches et GitHub (push, pull request)

| Code | Effet | Ch. |
|---|---|---|
| `git push` | envoie tes commits sur GitHub | 0A |
| `git switch -c essai` | crée une branche et s'y place | 0A |
| `git switch main` | revient sur `main` (les fichiers reprennent l'état de `main`) | 0A |
| `git merge essai` | intègre les commits de la branche `essai` dans la branche courante | 0A |
| `git branch -d essai` / `-D` | supprime une branche fusionnée / force la suppression | 0A |

