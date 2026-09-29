# Travailler sur Google Colab (avec Google Drive)

Colab exécute les notebooks dans ton navigateur, sur une machine de Google, avec toutes les bibliothèques déjà installées et un **GPU gratuit** (selon disponibilité). Le dépôt est gardé dans ton **Google Drive** : ton travail y est sauvegardé et `git pull` y récupère les nouveaux chapitres.

## 1. Première ouverture

1. Connecte-toi à ton compte Google.
2. Ouvre la démo : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cemah2/workbookIA/blob/main/00_setup/demo.ipynb)
   (ou, sur <https://colab.research.google.com> : *Fichier → Ouvrir un notebook → GitHub*, saisis `cemah2/workbookIA`, choisis `00_setup/demo.ipynb`).
3. Exécute la première cellule (**setup**, `Maj + Entrée`). Colab demande l'accès à Google Drive : accepte.
4. La cellule **clone le dépôt** dans `Mon Drive/workbookIA` (une seule fois, environ 30 secondes), puis affiche un rapport de l'environnement.
5. *Exécution → Tout exécuter* : tout doit aller jusqu'au bout.

Ce que fait la cellule de setup, à chaque ouverture de n'importe quel notebook :

| Action | Détail |
|---|---|
| monte Google Drive | dans `/content/drive/MyDrive` |
| met le dépôt à jour | `git pull --rebase --autostash` dans `Mon Drive/workbookIA` (ou le clone la première fois) : tes fichiers de `mon_travail/` sont mis de côté pendant la mise à jour puis remis en place |
| rend `wb` importable | ajoute `workbookIA/src` au chemin de Python |
| `wb.setup(seed=42, fast=FAST_MODE)` | graines aléatoires, device (GPU si activé), style des graphiques, rapport des versions |

Rien n'est installé : Colab a déjà tout. Si un jour Colab retire une bibliothèque, `wb.ensure("nom")` l'installe à la version du workbook.

## 2. Démarrer un chapitre

Tes copies de travail vont dans `mon_travail/`, dans ton Drive. Dans n'importe quel notebook où le setup a tourné, crée une cellule et exécute :

```python
!python /content/drive/MyDrive/workbookIA/tools/start_chapter.py 3
```

Puis, dans Google Drive, ouvre `workbookIA/mon_travail/ch03_…/03_notebook.ipynb` → clic droit → *Ouvrir avec → Google Colaboratory*. Ce notebook est **ta copie** : Colab l'enregistre automatiquement dans ton Drive.

La première fois (ou avec `--init`), `start_chapter.py` crée aussi `mon_travail/suivi/` : ton tableau de bord, ton journal et ton auto-évaluation, que tu modifies directement dans Drive (ou dans Colab via *Fichier → Ouvrir*). Les modèles de `suivi/` sont tenus par Claude : n'y écris pas.

> ⚠️ Travaille toujours dans `mon_travail/`. Si tu modifies un fichier ailleurs (par exemple `chapitres/`), le prochain `git pull` pourra refuser de s'exécuter.

Pour tester ta librairie `mylearn` depuis Colab :
```python
!cd /content/drive/MyDrive/workbookIA && python -m pytest tests/ -q
```

## 3. Activer le GPU (cellules 🚀)

*Exécution → Modifier le type d'exécution → Accélérateur matériel : GPU T4*, puis relance la cellule de setup : le rapport affiche `Device : cuda`. Le GPU gratuit est limité en durée : ne l'active que pour les cellules marquées 🚀 ou le mode complet (`FAST_MODE = False`).

## 4. Données et cache

- Les petits datasets (Penguins, MNIST, textes, taches solaires, California) sont dans le dépôt : rien à télécharger.
- Fashion-MNIST et CIFAR-10 sont téléchargés dans `/content/wb_cache` (disque local de la machine Colab, bien plus rapide que Drive). Ce cache disparaît quand la session se termine ; il est retéléchargé en moins d'une minute.

## 5. Sauvegarder ton travail sur GitHub (optionnel, pour ton portfolio)

Tes notebooks sont déjà sauvegardés dans Drive. Pour les publier aussi sur GitHub :

1. Sur GitHub : *Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token*. Limite-le au dépôt `workbookIA`, permission **Contents : Read and write**.
2. Dans Colab : icône 🔑 (*Secrets*) à gauche → ajoute un secret nommé `GITHUB_TOKEN` avec ce jeton, et active l'accès pour le notebook.
3. Dans une cellule :
   ```python
   import subprocess
   from google.colab import userdata

   repo = "/content/drive/MyDrive/workbookIA"
   token = userdata.get("GITHUB_TOKEN")
   git = lambda *args: subprocess.run(["git", "-C", repo, *args], check=True)
   git("config", "user.name", "Prénom Nom")
   git("config", "user.email", "ton.email@example.com")
   git("pull", "--rebase", "--autostash")
   git("add", "mon_travail")
   git("commit", "-m", "ch03: exercices 3.1 à 3.8")
   git("push", f"https://x-access-token:{token}@github.com/cemah2/workbookIA.git", "main")
   ```
   N'affiche jamais le jeton et ne l'écris jamais dans un notebook.

## 6. Problèmes fréquents

| Symptôme | Solution |
|---|---|
| « ⚠️ git pull a échoué » | Le plus souvent, un fichier **hors de `mon_travail/`** a été modifié. Dans une cellule : `!git -C /content/drive/MyDrive/workbookIA status` montre lequel ; copie ta modification ailleurs si tu y tiens, puis `!git -C /content/drive/MyDrive/workbookIA restore <fichier>`. |
| `Unable to create '.../.git/index.lock': File exists` | Une session Colab coupée pendant une opération git a laissé un verrou : `!rm /content/drive/MyDrive/workbookIA/.git/index.lock`, puis relance la cellule de setup. |
| `detected dubious ownership in repository` | La cellule de setup le règle (`safe.directory`) ; si le message persiste, relance-la. |
| `fatal: destination path ... already exists` | Un dossier `workbookIA` existe déjà dans ton Drive sans être un dépôt git : renomme-le, puis relance la cellule de setup. |
| « Drive already mounted » | Normal quand on relance la cellule : ce n'est pas une erreur. |
| La session se déconnecte | Colab coupe après une période d'inactivité (et au bout de quelques heures au maximum). Relance la cellule de setup puis les cellules nécessaires. |
| `Device : cpu` alors que je veux un GPU | Active le GPU (section 3) puis relance la cellule de setup. Le GPU gratuit n'est pas toujours disponible. |
| Une version différente est signalée ⚠️ | Colab a mis à jour ses bibliothèques. En général ça fonctionne ; sinon, *Exécution → Modifier le type d'exécution → Version de l'environnement d'exécution* permet de revenir à une version antérieure. Signale-le à Claude (prompt P6) pour mettre le workbook à jour. |
