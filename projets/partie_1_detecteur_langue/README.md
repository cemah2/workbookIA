# Mini-projet MP1 — Un détecteur de langue anglais / français *from scratch*

**Cahier des charges** du mini-projet du checkpoint I. Compte environ **8 heures** : sept étapes (7 h 30), plus la lecture du cahier des charges et, à la fin, celle de la solution ; c'est un projet de portfolio : à la fin, un module testé, un notebook propre, deux figures et un README qu'un recruteur peut lire.

| | |
|---|---|
| **Problème** | dire si un court extrait de texte est en anglais ou en français, avec une probabilité honnête |
| **Données** | *The Adventures of Sherlock Holmes* (anglais) et *Le Tour du monde en quatre-vingts jours* (français), découpés en chapitres puis en extraits de 5 à 200 caractères |
| **Méthode imposée** | un classifieur *Naive Bayes* sur les lettres, écrit sans bibliothèque de machine learning, avec ta librairie `mylearn` |
| **Prérequis** | chapitres 1 à 6 : `mylearn.stats`, `metrics`, `bayes`, `calculus` et `info` (si tu n'as pas écrit l'un de ces modules, celui de la référence est utilisé à sa place) |
| **Évaluation** | sur 20, avec la grille ci-dessous |

## Le contexte

Un service client reçoit des messages en anglais et en français, souvent très courts, et veut les envoyer à la bonne équipe. On te demande un premier détecteur, simple et explicable, et surtout **évalué comme un professionnel** : selon la longueur des messages, avec des intervalles de confiance, et avec des probabilités dont on a vérifié qu'elles sont honnêtes. C'est l'occasion de réunir tout ce que la partie I t'a appris : statistiques et bootstrap (ch. 2), mesures de qualité et calibration (ch. 3), règle de Bayes (ch. 4), descente de gradient (ch. 5), entropie et cross-entropy (ch. 6).

## Le modèle

Chaque langue est décrite par la fréquence de ses lettres, estimée sur des chapitres d'entraînement et lissée (aucune lettre à probabilité nulle). Un extrait reçoit, pour chaque langue, sa **log-vraisemblance**, la somme des logarithmes des probabilités de ses lettres : c'est l'hypothèse i.i.d. La **règle de Bayes** combine les log-vraisemblances avec un **prior**. Une **température**, ajustée par descente de gradient sur la log loss d'un jeu de validation, corrige au besoin la confiance des probabilités.

## Les contraintes

- **Pas de bibliothèque de machine learning pour le modèle** : NumPy et ta librairie `mylearn`. scikit-learn ne sert qu'à **vérifier** tes chiffres (étape MP1.4).
- **Pas de fuite** : chaque chapitre sert à un seul ensemble (entraînement, validation ou test) ; le test ne sert qu'à la mesure finale.
- **Reproductible** : graines fixées, et le notebook s'exécute d'un bout à l'autre en moins de 10 minutes sur un CPU.
- **Le code durable va dans un module**, `langid.py`, testé avec pytest ; le notebook l'appelle, le vérifie et trace les figures.

## Les livrables

1. `langid.py` : `LanguageDetector` (`fit`, `log_likelihood`, `predict_proba`, `predict`), `posterior_from_loglik`, `fit_temperature` et `evaluate`, qui réutilisent `mylearn.info`, `mylearn.bayes`, `mylearn.metrics` et `mylearn.calculus`.
2. `test_langid.py` : au moins **six** tests pytest (formes, probabilités qui somment à 1, lissage, log-vraisemblance, reproductibilité, prior, température, entrées invalides…).
3. `notebook.ipynb` : propre, exécuté de bout en bout, avec tes commentaires.
4. La **figure principale**, `figures/accuracy_auc_longueur.png` : l'accuracy et la ROC-AUC selon la longueur de l'extrait, avec leurs intervalles bootstrap.
5. Le **diagramme de fiabilité** avant et après la température, `figures/fiabilite.png`.
6. Le **README de portfolio** (`README.md` de ton dossier) : le problème, les données et leur licence, la méthode, les résultats chiffrés, les limites, les pistes, la façon de reproduire.

## Les étapes

| Étape | Ce que tu fais | Ce que tu produis | ⏱️ |
|---|---|---|---|
| MP1.1 | Cadrer le problème ; découper les chapitres en entraînement, validation et test ; tirer les extraits (graines imposées) | `SPLIT`, `make_set`, les sections « Le problème » et « Les données » du README | 45 min |
| MP1.2 | Explorer : fréquences des lettres, entropies, table des cross-entropies anglais/français | `letter_distribution`, `cross_entropy_table`, tes observations | 60 min |
| MP1.3 | Modéliser : distributions lissées, log-vraisemblance, règle de Bayes avec un prior, astuce log-sum-exp | `langid.py` (le modèle), tes premiers tests, la section « La méthode » | 90 min |
| MP1.4 | Évaluer selon la longueur : matrice de confusion, precision, recall, F1, ROC-AUC | `langid.evaluate`, `evaluate_by_length`, le tableau des scores | 60 min |
| MP1.5 | Quantifier l'incertitude : intervalles bootstrap de l'accuracy et de la ROC-AUC | `bootstrap_by_length`, la figure principale | 45 min |
| MP1.6 | Calibrer : diagramme de fiabilité, score de Brier, température ; l'effet du prior | `langid.fit_temperature`, `reliability_data`, le diagramme de fiabilité | 75 min |
| MP1.7 | Emballer : six tests au moins, README complet, notebook relancé de zéro, commit | le projet complet | 75 min |

Chaque étape du notebook se termine par des **garde-fous** (✅ ou ❌) : des propriétés qu'un travail juste doit vérifier (un découpage sans chapitre commun, des probabilités qui somment à 1, des intervalles qui contiennent leur score…). Ce ne sont pas des notes : deux projets justes peuvent avoir des chiffres différents.

## La grille d'évaluation (sur 20)

| Critère | Points | Tous les points | La moitié des points |
|---|---|---|---|
| **Découpage sans fuite**, justifié par écrit | 2 | trois ensembles de chapitres disjoints, le test jamais utilisé pour régler quoi que ce soit, et une phrase qui explique pourquoi on découpe par chapitres | des ensembles disjoints, mais pas de justification, ou la température réglée sur le test |
| **Modèle correct** : distributions lissées, log-probabilités, règle de Bayes | 4 | lissage > 0, log-vraisemblance en somme de logarithmes, posterior par la règle de Bayes avec le prior, aucun underflow sur un chapitre entier | le modèle marche sur des textes courts mais pas sur les longs (underflow), ou le prior est ignoré |
| **Évaluation complète selon la longueur**, bien interprétée | 4 | precision, recall, F1 et ROC-AUC pour chaque longueur, la matrice de confusion d'au moins une longueur courte, et une interprétation (à partir de quelle longueur, quelles erreurs) | les mesures sans interprétation, ou une seule longueur |
| **Incertitude** : intervalles bootstrap construits et commentés | 2 | un intervalle de l'accuracy et de la ROC-AUC pour chaque longueur, construit sur les extraits de test, et commenté (largeur, ce qu'il permet de conclure) | des intervalles sans commentaire, ou sur une seule longueur |
| **Calibration** : diagramme de fiabilité, Brier, température et effet mesuré | 3 | diagramme avant et après, Brier et log loss avant et après sur le test, température ajustée sur la validation, conclusion honnête (même si l'effet est petit) ; l'effet du prior discuté | une partie seulement (le diagramme sans la température, ou l'inverse) |
| **Code professionnel** | 3 | module documenté, au moins six tests verts, graines fixées, notebook qui s'exécute d'un bout à l'autre | moins de six tests, ou un notebook qui ne s'exécute plus de zéro |
| **README de portfolio** | 2 | clair et chiffré, avec la figure principale, les limites (textes du XIXᵉ siècle, deux langues, hypothèse i.i.d.) et des pistes | des résultats sans limites ni pistes, ou sans chiffres |

## Les extensions (pour aller plus loin)

- **Bigrammes de caractères** (le contexte local du ch. 6) : le gain sur les extraits très courts, et la perplexité des deux modèles.
- **Une troisième langue** (un roman allemand ou espagnol du Project Gutenberg) : un problème multiclasse, un F1 macro et une matrice 3 × 3.
- **Comparer avec `MultinomialNB`** de scikit-learn, après le ch. 13.
- **Comparer avec un détecteur pré-entraîné** 🕰️ (vérifié le 1ᵉʳ octobre 2026) : le modèle `lid.176` de fastText reconnaît 176 langues (appris sur Wikipédia, Tatoeba et SETimes, licence CC BY-SA 3.0 ; 917 ko dans sa version compressée `lid.176.ftz`). Mais le dépôt de fastText est archivé depuis mars 2024, et sa dernière version (`fasttext` 0.9.3, juin 2024) date d'avant NumPy 2 : avec la version de NumPy du workbook, `model.predict("un texte")` lève `ValueError: Unable to avoid copy…` ; passe une liste, `model.predict(["un texte"])`. La bibliothèque `langdetect` reconnaît 55 langues, n'a plus de nouvelle version depuis 2021, et ses réponses changent d'un appel à l'autre sur les textes courts tant qu'on n'a pas fixé `DetectorFactory.seed = 0`. Mesure-les sur tes extraits, selon la longueur. *Sources :* [fastText, « Language identification »](https://fasttext.cc/docs/en/language-identification.html) ; [dépôt fastText (archivé)](https://github.com/facebookresearch/fastText) ; [fasttext sur PyPI](https://pypi.org/project/fasttext/) ; [discussion « numpy incompatibility » du modèle fastText sur Hugging Face](https://huggingface.co/facebook/fasttext-language-identification/discussions/9) ; [langdetect sur PyPI](https://pypi.org/project/langdetect/).
- **Tes propres phrases** : des noms propres, des textes mélangés (on passe d'une langue à l'autre), des messages d'aujourd'hui ; documente les échecs.

## Démarrer

```bash
python tools/start_chapter.py CP1
```

copie le kit de départ dans `mon_travail/projets/partie_1_detecteur_langue/` : `notebook.ipynb` (les sept étapes), `langid.py` (à écrire), `test_langid.py` (deux tests d'exemple, à compléter), `data.py` (fourni : les chapitres et les extraits), `conftest.py` (fourni : pytest trouve ta librairie) et `README.md` (le modèle de ton README de portfolio). Travaille **dans ce dossier-là** ; les tests se lancent depuis la racine du dépôt :

```bash
python -m pytest mon_travail/projets/partie_1_detecteur_langue -q
```

Tant que `langid.py` n'est pas écrit, les tests affichent « ⏳ pas encore écrit : … » avec le nom de la fonction qui manque : c'est normal.

Tu peux faire corriger ton projet par Claude avec le prompt P9 : il lit ton dossier, sans jamais y écrire, et le note avec cette grille.

## La solution de référence

Le dossier [`solution/`](solution/) contient un projet complet : le module, 15 tests, le notebook exécuté et un README de portfolio rédigé. **Ne l'ouvre qu'après avoir fini le tien** : c'est une façon de faire parmi d'autres, et tes chiffres n'ont pas à être les mêmes.
