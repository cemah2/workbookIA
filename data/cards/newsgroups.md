# 20 Newsgroups (classification de textes, bonus B4)

| | |
|---|---|
| **Chargement** | `wb.datasets.load_20newsgroups(categories, split="train", n=None, seed=0)` → `(textes, labels)` ; par `sklearn.datasets.fetch_20newsgroups(subset=split, categories=…, remove=("headers", "footers", "quotes"))` |
| **Stockage** | téléchargé une fois par scikit-learn (`20news-bydate_py3.pkz`, ≈ 14,5 Mio (15,2 Mo), dans `data/downloads/sklearn/`, sur Colab `/content/wb_cache/sklearn/`) ; jamais versionné |
| **Source** | la version « bydate » de Jason Rennie, que scikit-learn télécharge ([documentation de scikit-learn, « The 20 newsgroups text dataset »](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)) ; messages collectés par Ken Lang pour NewsWeeder (1995) ; [UCI Machine Learning Repository, « Twenty Newsgroups »](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups) (Tom Mitchell, 1999) |
| **Taille** | 18 846 messages de 20 groupes de discussion Usenet : 11 314 d'entraînement et 7 532 de test, coupés à une date (les messages de test sont postérieurs) |
| **Tâche type** | classification de textes (le groupe d'un message) |
| **Licence** | Creative Commons Attribution 4.0 (CC BY 4.0) d'après l'UCI (page consultée le 2 novembre 2026) |
| **Téléchargé le** | 2026-11-02 |

## Ce que le workbook en prend
- **Quatre catégories** (📦 B4.28) : `comp.graphics`, `rec.autos`, `sci.med`, `sci.space`, dans cet ordre (labels 0 à 3). Sans les en-têtes, les signatures et les citations, et une fois retirés les messages de moins de 20 caractères : 2 275 messages d'entraînement (567, 556, 576 et 576 par catégorie) et 1 508 de test (383, 370, 379, 376). Médiane : 84 mots par message de test.
- `n` tire un sous-échantillon **stratifié** (autant de messages par catégorie, à une unité près) dans un ordre mélangé, à graine fixe : 1 000 messages d'entraînement et 400 de test en mode rapide, avec `seed=28`.

## Biais et limites
- **Fuite par les métadonnées** : les en-têtes (expéditeur, organisation, groupe), les signatures et les citations des messages précédents trahissent le sujet ; un classifieur y apprend des noms et des adresses, et son score est trop beau. scikit-learn recommande de les retirer (`remove=("headers", "footers", "quotes")`), ce que fait toujours `load_20newsgroups` (fiche B4, §204.9). Le retrait est heuristique : il reste des restes de signatures.
- **Daté** : des messages publics de 1992-1993, en anglais, écrits surtout par des universitaires et des ingénieurs américains ; vocabulaire technique de ces années-là (formats d'image, modèles de voitures).
- **Données personnelles** : des messages publics, mais nominatifs à l'origine ; le retrait des en-têtes en ôte la plupart des noms et adresses, pas tous.
- **Contamination** : un dataset si répandu est dans les données d'entraînement des LLM ; un LLM en zéro-shot peut en avoir vu des messages.

## Utilisation
Bonus B4 : 📦 B4.28 (TF-IDF et régression logistique, MiniLM fine-tuné par LoRA, LLM en zéro-shot, sur les mêmes messages).
