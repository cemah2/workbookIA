# Installer le workbook sur ton ordinateur (sans GPU)

Ce guide part de zéro. Compte **30 à 45 minutes** la première fois et **environ 3 Go** d'espace disque. Tu peux aussi tout faire sur Google Colab ([COLAB.md](COLAB.md)) ; l'installation locale sert à travailler hors ligne et à apprendre les gestes d'un développeur.

> 💡 **Terminal** : c'est la fenêtre où l'on tape des commandes.
> - **Windows** : menu Démarrer → tape « PowerShell » → *Windows PowerShell*.
> - **Mac** : `Cmd + Espace` → tape « Terminal ».
> - **Linux** : `Ctrl + Alt + T`.
>
> Dans ce guide, une ligne dans un bloc gris est une commande : copie-la, colle-la dans le terminal, appuie sur Entrée.

## Ce que tu vas installer

| Étape | Outil | Pourquoi |
|---|---|---|
| 1 | **Python 3.13** | le langage du workbook (même version que Colab) |
| 2 | **Git** | récupérer le dépôt et ses mises à jour, sauvegarder ton travail |
| 3 | le **dépôt** workbookIA | les chapitres, les outils, les données |
| 4 | un **environnement virtuel** | isoler les bibliothèques du workbook du reste de ton ordinateur |
| 5-6 | **PyTorch CPU** et les bibliothèques | numpy, pandas, scikit-learn, PyTorch, Jupyter… |
| 7-9 | vérification et démo | s'assurer que tout fonctionne |

---

## Étape 1 · Installer Python 3.13

### Windows
1. Va sur <https://www.python.org/downloads/windows/> et télécharge le dernier **Python 3.13** (*Windows installer 64-bit*).
2. Lance l'installateur. **Coche la case « Add python.exe to PATH »** en bas de la première fenêtre (très important), puis *Install Now*.
3. Ferme puis rouvre PowerShell, et vérifie :
   ```powershell
   py -3.13 --version
   ```
   Tu dois voir `Python 3.13.x`.

### Mac
1. Va sur <https://www.python.org/downloads/macos/> et télécharge le dernier **Python 3.13** (*macOS 64-bit universal2 installer*).
2. Ouvre le fichier `.pkg` et suis l'installation.
3. Dans le Terminal :
   ```bash
   python3.13 --version
   ```
> ⚠️ **Mac Intel** (processeur Intel, et non M1/M2/M3/M4) : PyTorch ne publie plus de versions pour ces Mac depuis 2024. Tout le reste fonctionne ; les exercices qui se servent de PyTorch (5.21, 10.14 et 10.15, puis les chapitres 20 et suivants) se font sur Colab, et les tests qui le prennent pour oracle sont ignorés (*skipped*). Pour savoir quel Mac tu as : menu  → *À propos de ce Mac* (« Puce Apple » ou « Processeur Intel »).

### Linux (Ubuntu, Debian…)
Il faut **Python 3.12 ou 3.13** (3.11 minimum ; les versions plus anciennes ne suffisent pas). Ubuntu 24.04 fournit Python 3.12, qui convient :
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 --version
```
Sur une distribution plus ancienne (Ubuntu 22.04 fournit Python 3.10), installe Python 3.13 à côté :
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.13 python3.13-venv
```
et remplace ensuite `python3` par `python3.13` à l'étape 4.

## Étape 2 · Installer Git

- **Windows** : télécharge l'installateur sur <https://git-scm.com/download/win> et garde toutes les options par défaut. Ferme puis rouvre PowerShell.
- **Mac** : dans le Terminal, tape `git --version`. Si Git n'est pas installé, macOS propose d'installer les « outils de ligne de commande » : accepte.
- **Linux** : `sudo apt install -y git`

Puis présente-toi à Git (une seule fois, avec ton nom et l'e-mail de ton compte GitHub), et demande-lui de mettre ton travail de côté pendant les mises à jour :
```bash
git config --global user.name "Prénom Nom"
git config --global user.email "ton.email@example.com"
git config --global pull.rebase true
git config --global rebase.autoStash true
```
Les deux dernières lignes rendent `git pull` sûr même si tu as des modifications en cours dans `mon_travail/`.

## Étape 3 · Récupérer le dépôt

Choisis où ranger le workbook (par exemple ton dossier personnel), puis :
```bash
cd ~
git clone https://github.com/cemah2/workbookIA.git
cd workbookIA
```
(Sous Windows, `cd ~` mène à `C:\Users\<toi>`.) Toutes les commandes suivantes se tapent **depuis ce dossier `workbookIA`**.

## Étape 4 · Créer et activer l'environnement virtuel

Un environnement virtuel est un dossier (`.venv`) qui contient une copie de Python et les bibliothèques du workbook, sans toucher au reste de l'ordinateur.

| Système | Créer (une seule fois) | Activer (à chaque nouvelle session) |
|---|---|---|
| Windows (PowerShell) | `py -3.13 -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| Mac | `python3.13 -m venv .venv` | `source .venv/bin/activate` |
| Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |

Une fois activé, le début de la ligne du terminal affiche `(.venv)`. Mets ensuite pip à jour :
```bash
python -m pip install --upgrade pip
```

> ⚠️ **Windows** : si PowerShell refuse d'exécuter `Activate.ps1` (« l'exécution de scripts est désactivée »), tape une fois `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, réponds `O`, puis relance l'activation.

## Étape 5 · Installer PyTorch (version CPU)

On installe PyTorch **avant** le reste pour obtenir la version CPU, bien plus légère.

- **Windows et Linux** :
  ```bash
  pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
  ```
- **Mac Apple Silicon (M1 à M4)** :
  ```bash
  pip install torch==2.11.0 torchvision==0.26.0
  ```
- **Mac Intel** : saute cette étape (voir l'avertissement de l'étape 1).

## Étape 6 · Installer les autres bibliothèques

```bash
pip install -r requirements.txt
pip install -e .
```
La première commande installe les versions exactes utilisées par le workbook (les mêmes que sur Colab) ; comptes 5 à 10 minutes. La seconde rend le package `wb` importable depuis n'importe quel dossier.

> **Mac Intel uniquement** : installe tout sauf PyTorch et les paquets qui en dépendent :
> ```bash
> grep -v -E "^(torch|torchvision|accelerate|peft)==" requirements.txt > requirements-mac-intel.txt
> pip install -r requirements-mac-intel.txt
> pip install -e .
> ```

## Étape 7 · Vérifier l'installation

```bash
python 00_setup/check_env.py
```
Tout doit être ✅. Un seul ⚠️ est normal à ce stade : « mon_travail/mylearn pas encore créé ». En cas de ❌, lis le message, puis le tableau « Problèmes fréquents » ci-dessous.

## Étape 8 · Créer ta librairie `mylearn`

```bash
python tools/start_chapter.py --init
```
Cela crée `mon_travail/mylearn/`, ton espace de code, et `mon_travail/suivi/`, tes fichiers de suivi (tableau de bord, journal, auto-évaluation). Par la suite, `python tools/start_chapter.py 3` (par exemple) copiera le notebook et les squelettes du chapitre 3, **sans jamais écraser** ce que tu as déjà écrit.

## Étape 9 · Lancer Jupyter et la démo

```bash
jupyter lab
```
Ton navigateur s'ouvre. Dans le panneau de gauche, ouvre `00_setup/demo.ipynb`, puis menu **Run → Run All Cells**. Tout doit s'exécuter jusqu'au bout. Pour arrêter Jupyter : `Ctrl + C` dans le terminal.

> 💡 Tu préfères un éditeur ? **VS Code** (<https://code.visualstudio.com/>) avec les extensions *Python* et *Jupyter* ouvre aussi les notebooks. Choisis l'interpréteur `.venv` quand VS Code le demande.

---

## Au quotidien

```bash
cd ~/workbookIA
source .venv/bin/activate          # Windows : .venv\Scripts\Activate.ps1
git pull                           # récupère les nouveaux chapitres
jupyter lab
```

## Problèmes fréquents

| Symptôme | Solution |
|---|---|
| `python` ou `py` « n'est pas reconnu » (Windows) | Python n'est pas dans le PATH : relance l'installateur, choisis *Modify* puis coche *Add Python to environment variables*. Ferme et rouvre PowerShell. |
| `ModuleNotFoundError: No module named 'wb'` | Environnement virtuel pas activé (pas de `(.venv)` en début de ligne), ou `pip install -e .` oublié. |
| `OSError: [WinError 126]` ou `c10.dll` en important torch (Windows) | Installe « Microsoft Visual C++ Redistributable » (x64) depuis le site de Microsoft, puis redémarre. Les chapitres sans PyTorch fonctionnent en attendant. |
| `pip` télécharge un fichier torch de plus de 2 Go (Linux) | Tu as oublié `--index-url https://download.pytorch.org/whl/cpu` à l'étape 5 : `pip uninstall torch torchvision`, puis refais l'étape 5. |
| `ERROR: No matching distribution found for torch==2.11.0` | Mauvaise version de Python (vérifie `python --version` : 3.12 ou 3.13), ou Mac Intel (voir étape 6). |
| Conflit de versions pendant `pip install -r requirements.txt` | Repars d'un environnement propre : supprime le dossier `.venv`, puis refais les étapes 4 à 6. |
| Erreur SSL ou proxy (réseau d'entreprise) | Essaie depuis un autre réseau, ou demande à ton service informatique la configuration pip. |
| Jupyter n'utilise pas le bon Python | Lance `jupyter lab` depuis le terminal où `(.venv)` est actif. |
| `git pull` refuse de s'exécuter, ou laisse des marqueurs `<<<<<<<` dans un fichier | Tu as modifié un fichier en dehors de `mon_travail/`, que Claude a modifié aussi. `git status` montre lequel ; `git restore --source=HEAD --staged --worktree <fichier>` reprend la version du dépôt (copie ta modification avant si tu y tiens). Si git l'a mise de côté (« autostash »), `git stash show -p` l'affiche, puis `git stash drop` quand tu l'as recopiée dans `mon_travail/`. Si le message parle de « divergent branches », lance les deux commandes `git config --global pull.rebase true` et `git config --global rebase.autoStash true` (étape 2). |

Toujours bloqué ? Copie le message d'erreur complet et demande de l'aide (voir `annexes/erreurs_frequentes.md`).
