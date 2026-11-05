# Réviser avec Anki : importer les flashcards du workbook

Chaque chapitre publié a son fichier `flashcards.csv` (15 à 30 cartes, BIBLE §12). L'outil `tools/export_flashcards.py` les rassemble en **un seul deck** pour [Anki](https://apps.ankiweb.net/), rangé en **sous-paquets** par partie et par chapitre :

```
Deep Learning Workbook
├── Partie 0
│   ├── Ch. 0A Python, notebooks et outils
│   └── Ch. 0B Maths du lycée au ML
├── Partie I
│   ├── Ch. 01 Introduction au machine learning et au deep learning
│   └── …
├── Partie IV
│   ├── Ch. 18 Rétropropagation
│   └── …
└── Partie VII
    └── Ch. B8 RL moderne : DQN, gradient de politique, PPO et RLHF
```

Les numéros ont deux chiffres (« Ch. 07 ») parce qu'Anki range les paquets par ordre alphabétique : ainsi, la liste suit l'ordre du workbook. Chaque carte garde son tag de chapitre (`dlwb::ch18::delta`) et reçoit en plus le tag de sa partie (`dlwb::partie_IV::ch18`), pour chercher ou suspendre une partie entière.

## 1. Fabriquer le fichier à importer

Dans un terminal, à la racine du dépôt. Sur Colab, une cellule `!python /content/drive/MyDrive/workbookIA/tools/export_flashcards.py --chapter 18` écrit le fichier dans `Mon Drive/workbookIA/exports/` : télécharge-le depuis [drive.google.com](https://drive.google.com), ou dans le panneau **Fichiers** de Colab (icône 📁 à gauche, `drive/MyDrive/workbookIA/exports/`, clic droit > **Télécharger**).

| Commande | Fichier créé dans `exports/` | Quand ? |
|---|---|---|
| `python tools/export_flashcards.py --chapter 18` | `dlwb_anki_ch18_backprop.csv` : les cartes d'un chapitre | à la fin de chaque chapitre (routine conseillée) |
| `python tools/export_flashcards.py` | `dlwb_anki.csv` : toutes les cartes | pour rattraper plusieurs chapitres, ou sur un nouvel ordinateur |
| `python tools/export_flashcards.py --apkg` | `dlwb_anki.apkg` : un paquet Anki (toutes les cartes, ou un chapitre avec `--chapter`) | si tu préfères un double-clic à la fenêtre d'import |
| `python tools/export_flashcards.py --check` | aucun : vérifie seulement les fichiers | après une mise à jour du dépôt |

Le CSV d'un chapitre et le CSV global rangent une même carte dans **le même sous-paquet** : tu peux importer chapitre par chapitre, puis le deck global, sans rien déplacer.

`--apkg` a besoin de la bibliothèque [genanki](https://github.com/kerrickstaley/genanki), qui ne fait pas partie des dépendances du workbook : `pip install genanki` (ou `pip install -e ".[anki]"`). Sans elle, la commande s'arrête avec un message et n'écrit rien ; le CSV contient exactement les mêmes cartes.

## 2. Importer le CSV (méthode conseillée)

Il faut **Anki 2.1.54 ou plus récent** (toute version actuelle convient : depuis 23.10, les numéros suivent l'année, et la version de novembre 2026 est la 26.09) : les premières lignes du fichier (`#separator:semicolon`, `#html:true`, `#tags column:3`, `#deck column:4`) règlent la fenêtre d'import à ta place ([manuel d'Anki, *Text Files*](https://docs.ankiweb.net/manual/importing/text-files)).

1. Ouvre Anki, puis **Fichier > Importer** (*File > Import*) et choisis le fichier `.csv` du dossier `exports/`.
2. Vérifie la fenêtre d'import :
   - **Type de note** : « Basique » (« Basic » si Anki est en anglais) : le recto dans le premier champ, le verso dans le second ;
   - **Séparateur** : point-virgule, et **HTML autorisé** coché (les deux sont lus dans le fichier) ;
   - **Paquet** : la colonne 4 (« Deck column ») ; les sous-paquets manquants sont créés ;
   - **Tags** : la colonne 3 ;
   - **Notes existantes** (*Existing notes*) : « Mettre à jour » (*Update*, le réglage par défaut). Quand tu réimportes un fichier après une correction des cartes, Anki reconnaît une carte à son **recto** (le premier champ, dans le même type de note) : il la met à jour sur place, dans son paquet actuel, avec son historique de révision. Une carte dont le recto a changé arrive comme une nouvelle carte : supprime l'ancienne dans **Parcourir**.
3. Clique sur **Importer**. Le résumé indique le nombre de cartes ajoutées ou mises à jour.

Les formules s'affichent grâce à MathJax, intégré à Anki (`\( … \)` dans les cartes) : rien à installer. Dans le fichier exporté, un « < » ou un « & » du texte est écrit `&lt;` ou `&amp;`, comme le demande le manuel quand le HTML est autorisé ; la carte affiche bien « < » et « & ».

> ⚠️ **Piège classique — des paquets « ch18_backprop » d'avant la session 79** — Les anciennes versions de l'outil rangeaient un chapitre dans `Deep Learning Workbook::ch18_backprop`. Une réimportation met les cartes à jour **dans leur paquet actuel**, sans les déplacer. Pour adopter le nouveau rangement : dans **Parcourir** (*Browse*), cherche `"deck:Deep Learning Workbook::ch18_backprop"`, sélectionne tout (Ctrl+A), puis **Cartes > Changer de paquet** (*Change Deck*, Ctrl+D) vers `Deep Learning Workbook::Partie IV::Ch. 18 Rétropropagation`, et supprime le paquet vide.

## 3. Importer le paquet `.apkg` (autre méthode)

1. Double-clique sur `exports/dlwb_anki.apkg`, ou **Fichier > Importer** dans Anki.
2. C'est tout : les sous-paquets, les tags et le type de note « Deep Learning Workbook (recto/verso) » sont créés.

Une réimportation d'un `.apkg` plus récent met à jour les cartes déjà présentes (réglage par défaut de la fenêtre d'import : mettre à jour les notes plus récentes), car chaque carte garde le même identifiant d'une exportation à l'autre (tant que son recto ne change pas). **Choisis une seule méthode** : un `.apkg` importé après le CSV (ou l'inverse) donne des doublons, puisque ce ne sont pas les mêmes types de notes.

## 4. Sur ton téléphone (AnkiDroid, AnkiMobile)

- **Le plus simple : la synchronisation.** Crée un compte gratuit sur [AnkiWeb](https://ankiweb.net), importe les cartes sur ton ordinateur (section 2 ou 3), clique sur **Synchroniser** (*Sync*, en haut de la fenêtre d'Anki), puis synchronise AnkiDroid (le bouton aux deux flèches en cercle, en haut de la liste des paquets) ou AnkiMobile sur iPhone avec le même compte. Les paquets, les cartes, tes révisions et tes suspensions suivent, dans les deux sens.
- **Sans ordinateur, sur Android** : envoie-toi le `.apkg` (pièce jointe, Google Drive) et ouvre-le depuis le téléphone : AnkiDroid propose de l'importer ; ou, dans AnkiDroid, menu de la liste des paquets > **Importer**, puis choisis le fichier ([manuel d'AnkiDroid, *Importing Anki Files*](https://ankidroid.org/docs/manual.html#importing)). AnkiDroid importe aussi le CSV depuis sa version 2.17 (la même fenêtre d'import que sur ordinateur), mais le `.apkg` est plus simple sur un petit écran.
- **Sur iPhone (AnkiMobile)** : ouvre le fichier depuis l'app **Fichiers**, puis « Ouvrir avec » AnkiMobile.
- **Importe sur un seul appareil**, puis synchronise : un même CSV importé sur l'ordinateur **et** sur le téléphone donne des doublons à la synchronisation (ses cartes n'ont pas d'identifiant fixe, contrairement à celles du `.apkg`).

## 5. Réviser au fil du parcours

- **Un chapitre à la fois** : clique sur le sous-paquet du chapitre que tu étudies (par exemple *Ch. 18 Rétropropagation*). Un clic sur *Partie IV* révise toute la partie, un clic sur *Deep Learning Workbook* tout ce que tu as importé.
- **Chaque jour** : fais tes révisions sur le paquet racine *Deep Learning Workbook* : Anki mélange les chapitres et te présente les cartes dues, ce qui entretient les chapitres anciens (la répétition espacée, c'est ce qui rend les flashcards rentables).
- **Nouvelles cartes par jour** : dans les options du paquet racine (roue dentée > *Options*), règle « Nouvelles cartes/jour » vers 20 : un chapitre (≈ 25 cartes) entre en un ou deux jours.

## 6. Suspendre les chapitres pas encore étudiés

Si tu importes le deck global au début, suspends ce que tu n'as pas encore étudié, puis réactive chapitre par chapitre :

1. **Parcourir** (*Browse*), puis cherche tout le deck : `"deck:Deep Learning Workbook"`.
2. Sélectionne tout (Ctrl+A), puis **Cartes > Suspendre** (*Toggle Suspend*, Ctrl+J) : les cartes suspendues apparaissent en jaune et ne sont plus proposées.
3. Quand tu commences un chapitre, cherche son sous-paquet, par exemple `"deck:Deep Learning Workbook::Partie IV::Ch. 18*"` (l'étoile évite de recopier le titre), ou son tag `tag:dlwb::ch18` ; sélectionne tout et refais **Cartes > Suspendre** (Ctrl+J) : la commande réactive les cartes suspendues.
4. Pour toute une partie : `tag:dlwb::partie_IV` (Anki inclut les tags enfants, donc les chapitres de la partie).

Les cartes de 0A et 0B se révisent dès le début ; celles d'un chapitre bonus, quand tu arrives au bonus (les parcours sont dans [`docs/PARCOURS.md`](../docs/PARCOURS.md)).

## 7. Ce qui ne change pas

- Le format des fichiers du workbook (`front;back;tags`, BIBLE §12) : la colonne du paquet n'est ajoutée qu'à l'export, dans `exports/` (non versionné, régénérable).
- Le tableau de bord (`suivi/tableau_de_bord.md`) garde sa case « Flashcards importées dans Anki » par chapitre.
- Les cartes ne donnent jamais la réponse d'une prédiction 🔮 ni d'un exercice : tu peux les réviser avant de faire le chapitre suivant.
