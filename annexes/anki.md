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

Dans un terminal, à la racine du dépôt. Sur Colab, une cellule `!python /content/drive/MyDrive/workbookIA/tools/export_flashcards.py --chapter 18` écrit le fichier dans `Mon Drive/workbookIA/exports/`, où tu le télécharges depuis Google Drive (`00_setup/COLAB.md`).

| Commande | Fichier créé dans `exports/` | Quand ? |
|---|---|---|
| `python tools/export_flashcards.py --chapter 18` | `dlwb_anki_ch18_backprop.csv` : les cartes d'un chapitre | à la fin de chaque chapitre (routine conseillée) |
| `python tools/export_flashcards.py` | `dlwb_anki.csv` : toutes les cartes | pour rattraper plusieurs chapitres, ou sur un nouvel ordinateur |
| `python tools/export_flashcards.py --apkg` | `dlwb_anki.apkg` : un paquet Anki (toutes les cartes, ou un chapitre avec `--chapter`) | si tu préfères un double-clic à la fenêtre d'import |
| `python tools/export_flashcards.py --check` | aucun : vérifie seulement les fichiers | après une mise à jour du dépôt |

Le CSV d'un chapitre et le CSV global rangent une même carte dans **le même sous-paquet** : tu peux importer chapitre par chapitre, puis le deck global, sans rien déplacer.

`--apkg` a besoin de la bibliothèque [genanki](https://github.com/kerrickstaley/genanki), qui ne fait pas partie des dépendances du workbook : `pip install genanki` (ou `pip install -e ".[anki]"`). Sans elle, la commande s'arrête avec un message et n'écrit rien ; le CSV contient exactement les mêmes cartes.

## 2. Importer le CSV (méthode conseillée)

Il faut **Anki 2.1.54 ou plus récent** : les premières lignes du fichier (`#separator`, `#html`, `#tags column`, `#deck column`) règlent la fenêtre d'import à ta place ([manuel d'Anki, *Text Files*](https://docs.ankiweb.net/manual/importing/text-files)).

1. Ouvre Anki, puis **Fichier > Importer** (*File > Import*) et choisis le fichier `.csv` du dossier `exports/`.
2. Vérifie la fenêtre d'import :
   - **Type de note** : « Basique » (« Basic » si Anki est en anglais) : le recto dans le premier champ, le verso dans le second ;
   - **Séparateur** : point-virgule, et **HTML autorisé** coché (les deux sont lus dans le fichier) ;
   - **Paquet** : la colonne 4 (« Deck column ») ; les sous-paquets manquants sont créés ;
   - **Tags** : la colonne 3 ;
   - **Doublons** : « Mettre à jour » (*Update*) si tu réimportes un fichier après une correction des cartes : la carte est mise à jour sur place et garde son historique de révision.
3. Clique sur **Importer**. Le résumé indique le nombre de cartes ajoutées ou mises à jour.

Les formules s'affichent grâce à MathJax, intégré à Anki (`\( … \)` dans les cartes) : rien à installer.

> ⚠️ **Piège classique — des paquets « ch18_backprop » d'avant la session 79** — Les anciennes versions de l'outil rangeaient un chapitre dans `Deep Learning Workbook::ch18_backprop`. Une réimportation met les cartes à jour **dans leur paquet actuel**, sans les déplacer. Pour adopter le nouveau rangement : dans **Parcourir** (*Browse*), cherche `"deck:Deep Learning Workbook::ch18_backprop"`, sélectionne tout (Ctrl+A), puis **Cartes > Changer de paquet** (*Change Deck*, Ctrl+D) vers `Deep Learning Workbook::Partie IV::Ch. 18 Rétropropagation`, et supprime le paquet vide.

## 3. Importer le paquet `.apkg` (autre méthode)

1. Double-clique sur `exports/dlwb_anki.apkg`, ou **Fichier > Importer** dans Anki.
2. C'est tout : les sous-paquets, les tags et le type de note « Deep Learning Workbook (recto/verso) » sont créés.

Une réimportation d'un `.apkg` plus récent met à jour les cartes déjà présentes, car chaque carte garde le même identifiant d'une exportation à l'autre (tant que son recto ne change pas). **Choisis une seule méthode** : un `.apkg` importé après le CSV (ou l'inverse) donne des doublons, puisque ce ne sont pas les mêmes types de notes.

## 4. Réviser au fil du parcours

- **Un chapitre à la fois** : clique sur le sous-paquet du chapitre que tu étudies (par exemple *Ch. 18 Rétropropagation*). Un clic sur *Partie IV* révise toute la partie, un clic sur *Deep Learning Workbook* tout ce que tu as importé.
- **Chaque jour** : fais tes révisions sur le paquet racine *Deep Learning Workbook* : Anki mélange les chapitres et te présente les cartes dues, ce qui entretient les chapitres anciens (la répétition espacée, c'est ce qui rend les flashcards rentables).
- **Nouvelles cartes par jour** : dans les options du paquet racine (roue dentée > *Options*), règle « Nouvelles cartes/jour » vers 20 : un chapitre (≈ 25 cartes) entre en un ou deux jours.

## 5. Suspendre les chapitres pas encore étudiés

Si tu importes le deck global au début, suspends ce que tu n'as pas encore étudié, puis réactive chapitre par chapitre :

1. **Parcourir** (*Browse*), puis cherche tout le deck : `"deck:Deep Learning Workbook"`.
2. Sélectionne tout (Ctrl+A), puis **Cartes > Suspendre** (*Toggle Suspend*, Ctrl+J) : les cartes suspendues apparaissent en jaune et ne sont plus proposées.
3. Quand tu commences un chapitre, cherche son sous-paquet, par exemple `"deck:Deep Learning Workbook::Partie IV::Ch. 18*"` (l'étoile évite de recopier le titre), ou son tag `tag:dlwb::ch18` ; sélectionne tout et refais **Cartes > Suspendre** (Ctrl+J) : la commande réactive les cartes suspendues.
4. Pour toute une partie : `tag:dlwb::partie_IV` (Anki inclut les tags enfants, donc les chapitres de la partie).

Les cartes de 0A et 0B se révisent dès le début ; celles d'un chapitre bonus, quand tu arrives au bonus (les parcours sont dans [`docs/PARCOURS.md`](../docs/PARCOURS.md)).

## 6. Ce qui ne change pas

- Le format des fichiers du workbook (`front;back;tags`, BIBLE §12) : la colonne du paquet n'est ajoutée qu'à l'export, dans `exports/` (non versionné, régénérable).
- Le tableau de bord (`suivi/tableau_de_bord.md`) garde sa case « Flashcards importées dans Anki » par chapitre.
- Les cartes ne donnent jamais la réponse d'une prédiction 🔮 ni d'un exercice : tu peux les réviser avant de faire le chapitre suivant.
