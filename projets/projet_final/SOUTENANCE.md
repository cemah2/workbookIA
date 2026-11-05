# Projet final · La soutenance (PF.13)

> Dix minutes de présentation, cinq minutes de questions façon entretien technique, une démonstration en direct. C'est l'étape PF.13 (1 point sur 20, 120 min de préparation), et c'est surtout la répétition de ce que tu feras en entretien : raconter un projet en partant du problème, défendre tes choix chiffres à l'appui, dire ce que tu ne sais pas.

**Le livrable** : les diapositives en PDF (dans le dépôt, `docs/soutenance.pdf`) et une répétition chronométrée **enregistrée** (une visio enregistrée, ou la caméra de ton téléphone : la vidéo reste chez toi, note dans `docs/rapport.md` sa durée et ce que tu as changé après l'avoir regardée).

## Le plan des dix minutes

| Temps | Partie | Ce que tu montres | Diapositives |
|---|---|---|---|
| 0:00 – 1:00 | **Le problème** | pour qui, quelle décision, la métrique métier, le critère de succès fixé avant les expériences (ton pitch de 29.4) | 1 |
| 1:00 – 2:30 | **Les données** | la source, la licence, la taille ; **un** défaut trouvé à l'audit et ce que tu en as fait ; la règle du découpage | 1 à 2 |
| 2:30 – 3:30 | **Le protocole** | le test scellé (le tag `protocole`, l'ouverture unique), les deux baselines, la métrique et son intervalle | 1 |
| 3:30 – 5:30 | **Le modèle et les expériences** | pourquoi ce modèle pour ces données ; **une** ablation qui a appris quelque chose, sur trois graines | 2 |
| 5:30 – 7:00 | **Les résultats** | le tableau des trois modèles avec leurs intervalles, le critère atteint ou non ; deux erreurs typiques | 1 à 2 |
| 7:00 – 8:00 | **Limites, explicabilité, équité** | ce que montrent les explications, le sous-groupe le moins bien servi, l'usage à proscrire | 1 |
| 8:00 – 9:30 | **La démonstration** | une prédiction en ligne de commande, puis l'API (`/health`, `/predict`), la latence mesurée | 0 (terminal) |
| 9:30 – 10:00 | **Ce que tu as appris** | une phrase honnête, et ce que tu ferais avec un mois de plus | 1 |

Dix diapositives au plus. Une idée par diapositive, un titre qui dit la conclusion (« Le fine-tuning bat le gradient boosting de 4 points [1 ; 7] » plutôt que « Résultats »), des chiffres arrondis à ce que l'intervalle permet (pas « 0,8734 » si l'intervalle fait ± 0,02), des figures lisibles depuis le fond de la salle. Le détail reste dans le rapport : la soutenance donne envie de l'ouvrir.

## Conseils de démonstration

- **Prépare-la comme un test** : un fichier `examples.jsonl` de trois ou quatre entrées choisies (une facile, une difficile, une qui se trompe, et tu dis pourquoi), les commandes dans l'ordre dans un fichier texte, le serveur lancé et chauffé avant de commencer (le premier appel est toujours plus lent).
- **Rien qui dure** : jamais d'entraînement en direct, jamais de téléchargement ; l'artefact est déjà là.
- **Un plan B** : une capture d'écran de chaque étape, ou une courte vidéo, si le réseau ou la machine lâchent. Dis-le calmement et passe au plan B : un recruteur juge ta réaction, pas la panne.
- **Lisible** : la police du terminal agrandie, un fond clair, les sorties JSON passées à `python -m json.tool`.
- **Vérifie la reproductibilité la veille** : un `git clone` neuf de ton dépôt, l'installation, les tests, une prédiction (c'est l'esprit du défi 🏆 B7.25). Ce qui casse ce soir-là aurait cassé chez le recruteur.
- **Mesure, n'affirme pas** : montre la latence que mesure `projet latency` et la machine, plutôt que « c'est rapide ».

## Les cinq minutes de questions

Un recruteur part de ton projet, puis élargit. Deux sortes de questions, à préparer toutes les deux.

**Sur ton projet** (prépare une réponse de 60 secondes à chacune) :

- Pourquoi cette métrique, et pas l'accuracy ? Que se passe-t-il pour l'utilisateur quand le modèle se trompe dans chaque sens ? (PF.1, PF.4)
- Comment sais-tu que ton test n'a pas fui dans tes choix ? Montre-moi la preuve. (PF.4, PF.8 : le tag, `experiments/test_access.jsonl`)
- Qu'est-ce qui te dit que ton modèle n'a pas appris un raccourci ? (PF.3, PF.9)
- Ton modèle deep learning bat-il vraiment la baseline classique ? De combien, avec quelle incertitude ? Valait-il son coût ? (PF.5, PF.8)
- Quelle ablation t'a le plus surpris ? (PF.7)
- Que ferais-tu si les données de production dérivaient ? Comment le verrais-tu ? (PF.11, B7.22)
- Qui ne devrait pas utiliser ce modèle, et pourquoi ? (PF.9, model card)
- Si tu recommençais, que changerais-tu en premier ?

**Élargies aux chapitres** : la banque ci-dessous, tirée des questions 💼 des chapitres que le projet mobilise. Chaque question renvoie à son ID ; la réponse modèle en 60 secondes et les relances possibles sont dans le `05_solutions.md` du chapitre. Choisis-en d'abord dans les chapitres de **tes** étapes (la ligne « Étapes du projet » de chaque chapitre), puis tire au hasard dans les autres.

**Une soutenance simulée avec Claude** : donne-lui ton README, ton rapport et ta model card, puis demande-lui : « Joue un recruteur ML. Pose-moi une question sur mon projet, puis une question de la banque de projets/projet_final/SOUTENANCE.md liée à mon type de données, avec une relance chacune ; attends ma réponse orale retranscrite avant la suivante ; à la fin, dis-moi ce qui manquait par rapport aux réponses modèles des 05_solutions.md, sans me les recopier. »

## Les critères de la grille (1 point)

| Critère | Points |
|---|---|
| Le temps tenu (10 minutes, ± 1) et le plan suivi : problème, données, protocole, modèle, résultats, limites, démonstration | 0,25 |
| Des résultats chiffrés **avec** leurs intervalles et la baseline, le critère de succès atteint ou non dit clairement | 0,25 |
| La démonstration en direct qui marche (ou le plan B déroulé sans perdre le fil) | 0,25 |
| Des réponses aux questions : précises, qui disent « je ne sais pas » quand c'est le cas, qui renvoient à une preuve du dépôt | 0,25 |

## La banque de questions d'entretien

<!-- pf:bank -->

*Liste générée par `python tools/chapters/build_pf.py` à partir des `02_exercices.md` des chapitres mobilisés (champ `chapters` du contrat) : ne la modifie pas à la main.*

### 2 · Hasard et statistiques de base

Étapes du projet : PF.3, PF.8 · réponses modèles : [`chapitres/ch02_stats/05_solutions.md`](../../chapitres/ch02_stats/05_solutions.md)

- **2.E1** — Moyenne ou médiane pour résumer des salaires ? · « Pour résumer les salaires d'une entreprise dans un tableau de bord, vous prenez la moyenne ou la médiane ? Pourquoi ? »
- **2.E2** — i.i.d. : définition et pourquoi le ML en a besoin · « Que veut dire i.i.d. ? Pourquoi est-ce si important en machine learning ? Donnez-moi un cas où l'hypothèse ne tient pas. »
- **2.E3** — Expliquer un intervalle de confiance bootstrap · « Votre modèle obtient 87 % d'accuracy sur un jeu de test de 400 exemples. Comment donneriez-vous une marge d'incertitude sur ce chiffre, sans collecter de nouvelles données ? Et comment l'expliqueriez-vous à un chef de projet ? »
- **2.E4** — Corrélation nulle veut-elle dire indépendance ? · « Deux variables ont une corrélation nulle : sont-elles indépendantes ? Et si leur corrélation vaut 0,9, l'une cause-t-elle l'autre ? »
- **2.E5** — Pourquoi fixer la graine aléatoire d'une expérience ? · « Pourquoi fixe-t-on la graine aléatoire dans une expérience de machine learning ? Est-ce que cela suffit pour que vos résultats soient fiables ? »

### 3 · Probabilités et mesure de la qualité

Étapes du projet : PF.1, PF.4, PF.8 · réponses modèles : [`chapitres/ch03_probabilites/05_solutions.md`](../../chapitres/ch03_probabilites/05_solutions.md)

- **3.E1** — 99 % d'accuracy sur la détection de fraude : bonne nouvelle ? · « Votre modèle de détection de fraude a 99 % d'accuracy. Est-ce une bonne nouvelle ? Qu'est-ce que vous regardez d'autre ? »
- **3.E2** — Precision ou recall : anti-spam, dépistage, modération · « Pour un filtre anti-spam, un dépistage médical et la modération automatique de commentaires, vous privilégiez la precision ou le recall ? Pourquoi ? »
- **3.E3** — Expliquer la courbe ROC et l'AUC · « Expliquez-moi la courbe ROC et l'AUC. Une AUC de 0,5, ça veut dire quoi ? Quand préférez-vous une autre courbe ? »
- **3.E4** — F1 macro ou micro : lequel choisir ? · « Sur un problème à plusieurs classes déséquilibrées, vous rapportez le F1 macro, micro ou pondéré ? Pourquoi ? »
- **3.E5** — Qu'est-ce qu'un modèle bien calibré ? · « Qu'est-ce qu'un modèle bien calibré ? Comment le vérifiez-vous, et comment corrigez-vous un modèle mal calibré ? »

### 8 · Entraînement et test

Étapes du projet : PF.4 · réponses modèles : [`chapitres/ch08_train_test/05_solutions.md`](../../chapitres/ch08_train_test/05_solutions.md)

- **8.E1** — Pourquoi trois jeux : entraînement, validation et test ? · « Pourquoi découpe-t-on les données en trois jeux, et pas seulement en deux ? Que se passe-t-il si l'on règle les hyperparamètres sur le jeu de test ? »
- **8.E2** — Qu'est-ce qu'une p-valeur ? Comment savoir si le modèle B bat vraiment le modèle A ? · « Votre nouveau modèle B obtient 91,2 % d'accuracy, contre 90,5 % pour le modèle A en production, sur le même jeu de test de 2 000 exemples. Peut-on dire que B est meilleur ? Qu'est-ce qu'une p-valeur, et comment l'obtiendriez-vous ici ? »
- **8.E3** — Validation croisée ou simple hold-out : quand choisir quoi ? · « Quand utilisez-vous une validation croisée, et quand un simple découpage entraînement / validation suffit-il ? Et en deep learning ? »
- **8.E4** — 99 % en test, échec en production : vos hypothèses · « Votre modèle obtenait 99 % sur le jeu de test, et il déçoit en production. Quelles hypothèses examinez-vous, et dans quel ordre ? »
- **8.E5** — Stratifier un découpage : quand et pourquoi ? · « Qu'est-ce qu'un découpage stratifié ? Quand est-il indispensable, et dans quels cas ne suffit-il pas ? »

### 12 · Préparation des données

Étapes du projet : PF.3, PF.4 · réponses modèles : [`chapitres/ch12_preparation/05_solutions.md`](../../chapitres/ch12_preparation/05_solutions.md)

- **12.E1** — Qu'est-ce qu'une fuite de données ? Donne deux exemples · « Qu'est-ce qu'une fuite de données ? Donnez deux exemples différents, et dites comment vous les auriez évitées. »
- **12.E2** — Standardisation ou normalisation min-max : laquelle, et pourquoi ? · « Standardisation ou normalisation min-max : laquelle utilisez-vous, et quand ? Faut-il toujours mettre les features à l'échelle ? »
- **12.E3** — Comment traites-tu les valeurs manquantes ? · « Votre dataset a 15 % de valeurs manquantes sur une feature importante. Comment les traitez-vous ? »
- **12.E4** — À quoi sert une PCA et quelles sont ses limites ? · « Expliquez la PCA à un collègue qui ne la connaît pas. À quoi sert-elle, et quand ne faut-il pas l'utiliser ? »
- **12.E5** — Encoder une variable catégorielle à 10 000 modalités · « Une feature a 10 000 catégories possibles, par exemple un identifiant de produit. Comment l'encodez-vous ? »

### 13 · Classifieurs

Étapes du projet : PF.5 · réponses modèles : [`chapitres/ch13_classifieurs/05_solutions.md`](../../chapitres/ch13_classifieurs/05_solutions.md)

- **13.E1** — Forces et faiblesses du kNN ; pourquoi standardiser · « Quelles sont les forces et les faiblesses des k plus proches voisins ? Pourquoi faut-il standardiser les données avant ? »
- **13.E2** — Comment un arbre de décision choisit-il ses splits ? · « Expliquez comment un arbre de décision choisit la question qu'il pose à chaque nœud. Quelle différence entre Gini et l'entropie ? »
- **13.E3** — Comment limiter l'overfitting d'un arbre ? · « Votre arbre de décision a 100 % d'accuracy sur l'entraînement et 78 % sur la validation. Que faites-vous ? »
- **13.E4** — L'hypothèse de Naïve Bayes, et pourquoi il marche quand même · « Quelle hypothèse fait Naïve Bayes ? Elle est presque toujours fausse : pourquoi le modèle marche-t-il quand même, par exemple pour filtrer le spam ? »
- **13.E5** — Quelle baseline : régression logistique, SVM ou arbre ? · « Vous recevez un nouveau problème de classification sur des données tabulaires. Quelle baseline entraînez-vous en premier, et pourquoi ? Quand passeriez-vous à un SVM ou à un arbre ? »

### 14 · Ensembles

Étapes du projet : PF.5 · réponses modèles : [`chapitres/ch14_ensembles/05_solutions.md`](../../chapitres/ch14_ensembles/05_solutions.md)

- **14.E1** — Bagging contre boosting : différences et cas d'usage · « Quelle est la différence entre le bagging et le boosting ? Dans quel cas utilisez-vous l'un ou l'autre ? »
- **14.E2** — Ajouter des arbres à une forêt fait-il surapprendre ? · « Si j'augmente le nombre d'arbres d'une forêt aléatoire, est-ce que je risque l'overfitting ? Et pour un gradient boosting ? »
- **14.E3** — XGBoost ou réseau de neurones sur un tableau de données ? · « On vous donne un tableau de 50 000 lignes et 40 colonnes, numériques et catégorielles, pour prédire un défaut de paiement. Partez-vous sur XGBoost ou sur un réseau de neurones ? Pourquoi ? »
- **14.E4** — Peut-on se fier à feature_importances_ ? · « Votre forêt aléatoire classe en tête de ses `feature_importances_` une colonne qui ne devrait rien dire. Comment l'expliquez-vous, et comment mesurez-vous l'importance des features de façon plus fiable ? »
- **14.E5** — Comment régler un gradient boosting ? · « Vous avez un gradient boosting à régler. Quels hyperparamètres touchez-vous, dans quel ordre, et comment évitez-vous l'overfitting ? »

### 15 · scikit-learn

Étapes du projet : PF.5 · réponses modèles : [`chapitres/ch15_sklearn/05_solutions.md`](../../chapitres/ch15_sklearn/05_solutions.md)

- **15.E1** — fit, transform, predict : pourquoi jamais fit sur le test ? · « Expliquez la différence entre `fit`, `transform`, `fit_transform` et `predict`. Pourquoi ne faut-il jamais appeler `fit` sur le jeu de test ? »
- **15.E2** — Grille, hasard ou optimisation bayésienne ? · « Vous devez régler une dizaine d'hyperparamètres d'un gradient boosting avec un budget de calcul limité. Recherche en grille, au hasard, ou autre chose ? Comment évaluez-vous le modèle final ? »
- **15.E3** — Pipeline et ColumnTransformer sur des données hétérogènes · « Votre dataset a des colonnes numériques avec des valeurs manquantes et des colonnes catégorielles. Comment construisez-vous la préparation dans scikit-learn, et pourquoi ? »
- **15.E4** — predict_proba donne-t-il une vraie probabilité ? · « Votre modèle renvoie 0,8 pour un client. Peut-on dire qu'il a 80 % de chances de résilier ? Comment le vérifier, et comment choisir le seuil de décision ? »
- **15.E5** — Mettre un modèle scikit-learn en production · « Votre modèle est prêt dans un notebook. Comment le livrez-vous pour qu'il serve en production et qu'on puisse reproduire ses résultats dans six mois ? »

### 19 · Optimiseurs

Étapes du projet : PF.7 · réponses modèles : [`chapitres/ch19_optimiseurs/05_solutions.md`](../../chapitres/ch19_optimiseurs/05_solutions.md)

- **19.E1** — Adam ou SGD avec momentum : lequel choisir, et pourquoi ? · « Pour entraîner un nouveau modèle, vous partez sur Adam ou sur SGD avec momentum ? Qu'est-ce qui guiderait votre choix ? »
- **19.E2** — Adam avec weight_decay contre AdamW · « Quelle est la différence entre `Adam(weight_decay=...)` et `AdamW` ? Pourquoi est-ce important ? »
- **19.E3** — À quoi servent le warmup et la décroissance en cosinus ? · « Dans la plupart des recettes d'entraînement de Transformers, on voit un warmup puis une décroissance en cosinus. À quoi sert chacun ? »
- **19.E4** — Qu'est-ce que le gradient clipping, et quand l'utiliser ? · « Qu'est-ce que le gradient clipping ? Par la norme ou par la valeur ? Dans quels cas est-il indispensable, et comment choisir le seuil ? »
- **19.E5** — Taille du mini-batch : quels compromis ? · « Comment choisissez-vous la taille du mini-batch ? Que se passe-t-il si vous la multipliez par 8 sans rien changer d'autre ? »

### 20 · Deep learning et premiers pas en PyTorch

Étapes du projet : PF.6 · réponses modèles : [`chapitres/ch20_deep_learning/05_solutions.md`](../../chapitres/ch20_deep_learning/05_solutions.md)

- **20.E1** — Dropout : fonctionnement, et ce que change model.eval() · « Expliquez le dropout. Que se passe-t-il quand on appelle `model.eval()`, et que se passe-t-il si on l'oublie ? »
- **20.E2** — Batchnorm ou layer norm : différences et usages · « Quelle est la différence entre la batchnorm et la layer norm ? Pourquoi les Transformers utilisent-ils la seconde ? »
- **20.E3** — Décrire une boucle d'entraînement PyTorch, étape par étape · « Décrivez-moi une boucle d'entraînement PyTorch, étape par étape. Que fait chaque ligne, et dans quel ordre doivent-elles venir ? »
- **20.E4** — Ton modèle surapprend : que fais-tu, et dans quel ordre ? · « Votre modèle a 99 % d'accuracy en entraînement et 85 % en validation. Que faites-vous, et dans quel ordre ? »
- **20.E5** — Pourquoi les connexions résiduelles rendent-elles les réseaux très profonds entraînables ? · « Pourquoi les connexions résiduelles permettent-elles d'entraîner des réseaux de cent couches ou plus ? »

### 21 · Réseaux convolutifs (CNN)

Étapes du projet : PF.6 · réponses modèles : [`chapitres/ch21_cnn/05_solutions.md`](../../chapitres/ch21_cnn/05_solutions.md)

- **21.E1** — Pourquoi un CNN plutôt qu'un MLP pour des images ? · « Pourquoi utilise-t-on des réseaux convolutifs plutôt que des réseaux denses pour des images ? »
- **21.E2** — Taille de sortie et paramètres d'une Conv2d(3, 64, 3, stride=2, padding=1) sur une image 224×224 · « J'ai une couche `nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)` et des images de 224 × 224. Quelle est la forme de la sortie pour un mini-batch de 32 images, et combien de paramètres a la couche ? »
- **21.E3** — À quoi sert une convolution 1×1 ? · « À quoi sert une convolution 1 × 1 ? Elle ne regarde qu'un pixel : qu'est-ce qu'elle peut bien apprendre ? »
- **21.E4** — Exemples adversariaux : quels risques, quelles défenses ? · « Qu'est-ce qu'un exemple adversarial ? Si vous déployez un modèle de vision, comment évaluez-vous et réduisez-vous ce risque ? »
- **21.E5** — CNN ou Vision Transformer pour un nouveau projet de vision ? · « On démarre un projet de classification d'images industrielles, avec 5 000 images étiquetées. Vous partez sur un CNN ou sur un Vision Transformer ? »

### 22 · Réseaux récurrents (RNN, LSTM, GRU)

Étapes du projet : PF.6 · réponses modèles : [`chapitres/ch22_rnn/05_solutions.md`](../../chapitres/ch22_rnn/05_solutions.md)

- **22.E1** — Pourquoi les Transformers ont-ils remplacé les RNN pour le texte ? Quand garderais-tu un LSTM ? · « Pourquoi les Transformers ont-ils remplacé les RNN pour le texte ? Est-ce qu'il vous arrive encore d'utiliser un LSTM, et pour quoi ? »
- **22.E2** — Explique le gradient évanescent et deux façons d'y remédier · « Expliquez-moi le problème du gradient évanescent dans un RNN, et donnez deux façons d'y remédier. »
- **22.E3** — LSTM ou GRU : différences et critères de choix · « Quelle différence entre un LSTM et un GRU ? Comment choisissez-vous ? »
- **22.E4** — Comment valider un modèle de série temporelle sans fuite de données ? · « Vous devez prévoir les ventes hebdomadaires d'un magasin avec un LSTM. Comment validez-vous votre modèle sans fuite de données, et à quoi le comparez-vous ? »
- **22.E5** — À quoi sert la température quand on génère du texte ? · « Dans une API de modèle de langue, il y a un paramètre `temperature`. À quoi sert-il, et quelle valeur choisiriez-vous pour un assistant qui extrait des informations d'un contrat ? Et pour écrire un slogan ? »

### 23 · PyTorch en pratique 1 : du dataset au modèle sauvegardé

Étapes du projet : PF.6, PF.11, PF.12 · réponses modèles : [`chapitres/ch23_pytorch_1/05_solutions.md`](../../chapitres/ch23_pytorch_1/05_solutions.md)

- **23.E1** — Ton entraînement PyTorch est deux fois plus lent que prévu : où cherches-tu, et dans quel ordre ? · « Votre entraînement PyTorch prend deux fois plus de temps que prévu. Comment cherchez-vous la cause, et dans quel ordre ? »
- **23.E2** — Pourquoi appeler model.eval() et torch.no_grad() (ou inference_mode) à l'inférence ? · « Que font `model.eval()` et `torch.no_grad()` ? Pourquoi faut-il les deux pour prédire, et que se passe-t-il si l'on oublie l'un ou l'autre ? »
- **23.E3** — Comment sauvegarder un modèle PyTorch pour reprendre l'entraînement plus tard ou le déployer ? · « Comment sauvegardez-vous un modèle PyTorch ? Est-ce la même chose pour reprendre un entraînement demain et pour le mettre en production ? »
- **23.E4** — La loss de validation remonte alors que l'accuracy de validation progresse encore : que se passe-t-il ? · « Sur vos courbes, la loss de validation remonte depuis quelques epochs, mais l'accuracy de validation continue de monter. Que se passe-t-il, et que faites-vous ? »
- **23.E5** — Keras ou PyTorch : comment choisir un framework aujourd'hui ? · « Keras ou PyTorch : lequel choisiriez-vous pour un nouveau projet, et sur quels critères ? »

### 24 · PyTorch en pratique 2 : améliorer, chercher, CNN et RNN

Étapes du projet : PF.6, PF.7 · réponses modèles : [`chapitres/ch24_pytorch_2/05_solutions.md`](../../chapitres/ch24_pytorch_2/05_solutions.md)

- **24.E1** — Comment organises-tu une recherche d'hyperparamètres avec un budget limité ? · « Vous avez deux jours de GPU pour régler un réseau. Comment organisez-vous la recherche d'hyperparamètres ? »
- **24.E2** — Pourquoi l'augmentation de données améliore-t-elle la généralisation ? Donne des transformations à éviter · « Pourquoi l'augmentation de données améliore-t-elle la généralisation ? Donnez des transformations à éviter, et dites pourquoi. »
- **24.E3** — Dropout ou batchnorm : lequel, où, et peut-on combiner les deux ? · « Dropout ou batchnorm : lequel utilisez-vous, à quel endroit du réseau, et peut-on combiner les deux ? »
- **24.E4** — Comment éviter la fuite de données dans une validation croisée avec prétraitement ? · « Votre validation croisée donne un score bien meilleur que le test final. Votre pipeline normalise les données et choisit des features. Où cherchez-vous une fuite de données, et comment l'évitez-vous ? »
- **24.E5** — Batchnorm : que fait-elle, et pourquoi se comporte-t-elle différemment en train et en eval ? · « Que fait une couche de batchnorm ? Pourquoi se comporte-t-elle différemment en entraînement et en évaluation, et quels bugs cela peut-il causer ? »

### 29 · Datasets et préparation du projet final

Étapes du projet : PF.1, PF.2, PF.3, PF.8, PF.12, PF.13 · réponses modèles : [`chapitres/ch29_datasets/05_solutions.md`](../../chapitres/ch29_datasets/05_solutions.md)

- **29.E1** — Comment choisir et valider un dataset pour un nouveau projet ? · « On vous confie un nouveau projet de classification. Comment choisissez-vous les données, et comment vous assurez-vous qu'elles conviennent ? »
- **29.E2** — Qu'est-ce qu'une data card et que doit-elle contenir ? · « Qu'est-ce qu'une data card (ou datasheet) ? Que mettez-vous dedans, et à quoi sert-elle ? »
- **29.E3** — Comment détecter une fuite de données entre entraînement et test ? · « Votre modèle obtient 99,5 % sur le test. Comment vérifiez-vous qu'il n'y a pas de fuite ? »
- **29.E4** — Données publiques et usage commercial : que vérifier ? · « Nous voulons utiliser un dataset public trouvé en ligne pour un produit. Que vérifiez-vous avant ? »

### B1 · Transfer learning et modèles pré-entraînés

Étapes du projet : PF.6 · réponses modèles : [`chapitres/b1_transfer/05_solutions.md`](../../chapitres/b1_transfer/05_solutions.md)

- **B1.E1** — Extraction de features ou fine-tuning : comment choisis-tu ? · « Vous partez d'un modèle pré-entraîné pour une tâche de classification d'images. Comment décidez-vous entre extraire des features et fine-tuner, et que mesurez-vous pour trancher ? »
- **B1.E2** — Tu as 500 images médicales étiquetées : quelle stratégie de transfer learning ? · « Un hôpital vous confie 500 radiographies étiquetées, en deux classes. Quelle stratégie proposez-vous, et comment savez-vous qu'elle marche ? »
- **B1.E3** — Pourquoi un learning rate plus faible pour les couches pré-entraînées ? · « Pourquoi utilise-t-on un learning rate plus faible pour les couches pré-entraînées que pour la nouvelle tête ? Comment le mettez-vous en place en PyTorch ? »
- **B1.E4** — Qu'est-ce qu'un modèle de fondation, et que change le zéro-shot ? · « Qu'est-ce qu'un modèle de fondation ? Que veut dire classer en zéro-shot, et qu'est-ce que cela change pour un projet ? »
- **B1.E5** — Comment vérifier qu'un modèle pré-entraîné peut servir dans un produit commercial ? · « Vous avez trouvé sur le Hugging Face Hub le modèle idéal pour votre produit. Que vérifiez-vous avant de vous en servir ? »

### B2 · Tokenisation et embeddings

Étapes du projet : PF.6 · réponses modèles : [`chapitres/b2_tokenisation/05_solutions.md`](../../chapitres/b2_tokenisation/05_solutions.md)

- **B2.E1** — Pourquoi les LLM découpent-ils le texte en sous-mots plutôt qu'en mots ou en caractères ? · « Les LLM ne lisent ni des mots ni des lettres, mais des sous-mots. Pourquoi ce choix, et qu'est-ce qu'il coûte ? »
- **B2.E2** — Explique BPE, puis ce qui le distingue de WordPiece · « Expliquez-moi comment BPE construit son vocabulaire et encode un mot nouveau. En quoi WordPiece est-il différent ? »
- **B2.E3** — Que se passe-t-il si l'on change de tokenizer sans ré-entraîner le modèle ? · « Un collègue veut remplacer le tokenizer d'un modèle déjà entraîné par un tokenizer plus récent, plus économe en français. Qu'en pensez-vous ? »
- **B2.E4** — Qu'est-ce qu'un embedding, et comment mesurer la similarité de deux textes ? · « Qu'est-ce qu'un embedding ? Comment mesureriez-vous la similarité de deux textes, et à quoi faut-il faire attention ? »
- **B2.E5** — Comment construirais-tu une recherche sémantique sur des documents internes ? · « Nous avons 200 000 documents internes, en français et en anglais. Comment construiriez-vous une recherche par le sens, et comment sauriez-vous qu'elle marche ? »

### B3 · Attention et Transformers : un mini-GPT from scratch

Étapes du projet : PF.6 · réponses modèles : [`chapitres/b3_transformers/05_solutions.md`](../../chapitres/b3_transformers/05_solutions.md)

- **B3.E1** — Explique le mécanisme d'attention, de la requête à la sortie · « Expliquez-moi l'attention d'un Transformer, depuis les vecteurs des tokens jusqu'à la sortie d'une couche multi-têtes. »
- **B3.E2** — Quel est le coût de l'auto-attention en fonction de la longueur, et comment traite-t-on les contextes longs ? · « Que coûte l'auto-attention quand le contexte s'allonge, en calcul et en mémoire ? Comment les modèles actuels tiennent-ils 128 000 tokens ? »
- **B3.E3** — Encodeur, décodeur ou encodeur-décodeur : lequel pour quelle tâche ? · « Nous avons trois projets : classer des e-mails, résumer des contrats, et un assistant qui répond aux clients. Quelle famille de Transformer pour chacun, et pourquoi ? »
- **B3.E4** — À quoi sert le KV cache, et combien de mémoire consomme-t-il ? · « Expliquez le KV cache. Combien de mémoire prend-il pour un modèle de 8 milliards de paramètres à 32 000 tokens de contexte, et comment la réduit-on ? »
- **B3.E5** — Pourquoi un Transformer a-t-il besoin d'encodages de position, et qu'apporte RoPE ? · « Pourquoi faut-il donner les positions à un Transformer, et pourquoi presque tous les LLM récents utilisent-ils RoPE ? »

### B4 · LLM en pratique : Hugging Face, prompting, RAG et LoRA

Étapes du projet : PF.6 · réponses modèles : [`chapitres/b4_llm/05_solutions.md`](../../chapitres/b4_llm/05_solutions.md)

- **B4.E1** — Comment réduis-tu les hallucinations d'un assistant fondé sur un LLM ? · « Notre assistant invente parfois des réponses avec beaucoup d'assurance. Que faites-vous pour le réduire, et comment savez-vous que ça marche ? »
- **B4.E2** — RAG ou fine-tuning : comment choisis-tu ? · « Nous voulons un assistant qui connaisse notre catalogue de 50 000 produits et qui réponde dans le ton de la marque. RAG, fine-tuning, ou les deux ? »
- **B4.E3** — Explique LoRA et pourquoi c'est si économique · « Expliquez-moi LoRA. Pourquoi est-ce si peu coûteux, et qu'est-ce qu'on perd par rapport à un fine-tuning complet ? »
- **B4.E4** — Comment évalues-tu un système RAG, de la recherche à la réponse ? · « Vous avez construit un RAG sur notre documentation. Comment mesurez-vous qu'il est bon, et comment trouvez-vous ce qui ne va pas quand il se trompe ? »
- **B4.E5** — Quels risques de sécurité avec un LLM en production, et quelles parades ? · « Nous allons brancher un LLM sur nos e-mails clients et lui donner accès à notre outil de tickets. Quels risques de sécurité voyez-vous, et que mettez-vous en place ? »

### B6 · Explicabilité, équité et éthique

Étapes du projet : PF.8, PF.9, PF.12 · réponses modèles : [`chapitres/b6_explicabilite/05_solutions.md`](../../chapitres/b6_explicabilite/05_solutions.md)

- **B6.E1** — Comment expliquer les prédictions d'un gradient boosting à une équipe métier ? · « Notre modèle de churn est un gradient boosting. L'équipe commerciale ne lui fait pas confiance parce qu'elle ne comprend pas ses prédictions. Que lui montrez-vous ? »
- **B6.E2** — Qu'est-ce qu'une valeur SHAP, et quelles sont ses limites ? · « Vous citez SHAP dans votre CV. Expliquez-moi ce qu'est une valeur SHAP, et dans quels cas vous vous en méfieriez. »
- **B6.E3** — Comment audites-tu un modèle pour détecter des biais ? · « On vous confie un modèle de présélection de CV déjà en production. Comment vérifiez-vous qu'il ne discrimine pas ? »
- **B6.E4** — Grad-CAM : principe, usage et limites · « Votre classifieur d'images médicales atteint 95 % d'accuracy. Un radiologue demande où le modèle regarde. Que lui proposez-vous, et avec quelles précautions ? »
- **B6.E5** — Retirer la variable sensible suffit-il à rendre un modèle équitable ? · « Pour éviter toute discrimination, notre équipe a retiré le sexe et l'âge des features. Le modèle est donc équitable, non ? »

### B7 · Du notebook à la production

Étapes du projet : PF.7, PF.10, PF.11, PF.12, PF.13 · réponses modèles : [`chapitres/b7_production/05_solutions.md`](../../chapitres/b7_production/05_solutions.md)

- **B7.E1** — Comment passes-tu d'un notebook à du code prêt pour la production ? · « Vous avez un notebook d'exploration qui donne de bons résultats. Comment le transformez-vous en code qu'une équipe peut maintenir et déployer ? »
- **B7.E2** — Comment garantis-tu la reproductibilité d'un entraînement ? · « Un collègue n'arrive pas à retrouver votre accuracy de 85 %. Que faites-vous pour qu'un entraînement soit reproductible, et comment rapportez-vous le résultat ? »
- **B7.E3** — Que testes-tu dans un projet de machine learning ? · « Au-delà des tests unitaires habituels, que testez-vous dans un projet de machine learning, et quand ces tests tournent-ils ? »
- **B7.E4** — Comment surveilles-tu un modèle en production, et que fais-tu en cas de dérive ? · « Votre modèle est en production depuis six mois. Comment savez-vous qu'il fonctionne encore, et que faites-vous si les données dérivent ? »
- **B7.E5** — Tu dois exposer ton modèle par une API : quels choix fais-tu (batches, latence, versions) ? · « Vous devez exposer votre classifieur d'images par une API. Requêtes une par une ou par batches, latence, versions du modèle : quels choix faites-vous, et comment les vérifiez-vous ? »

### B8 · RL moderne : DQN, gradient de politique, PPO et RLHF

Étapes du projet : PF.6 · réponses modèles : [`chapitres/b8_rl_moderne/05_solutions.md`](../../chapitres/b8_rl_moderne/05_solutions.md)

- **B8.E1** — Méthodes par valeur ou par gradient de politique : avantages et inconvénients · « Quand choisiriez-vous une méthode par valeur, comme DQN, plutôt qu'une méthode par gradient de politique, comme PPO ? Donnez les avantages et les inconvénients de chacune. »
- **B8.E2** — Explique PPO et le rôle du clipping · « Expliquez-moi PPO. Que fait exactement le clipping, et que se passerait-il sans lui ? »
- **B8.E3** — Ton agent RL n'apprend pas : comment le débogues-tu ? · « Votre agent de RL tourne depuis une heure et son retour ne monte pas. Aucune erreur. Que faites-vous, dans quel ordre ? »
- **B8.E4** — PPO avec un modèle de récompense ou DPO pour aligner un modèle de langue : que choisis-tu ? · « Vous devez aligner un modèle de langue de 7 milliards de paramètres sur les préférences de vos utilisateurs. RLHF avec PPO et un modèle de récompense, ou DPO ? Qu'est-ce qui guide votre choix ? »
- **B8.E5** — Pourquoi faut-il plusieurs graines pour comparer deux algorithmes de RL ? · « Un collègue annonce que sa variante de PPO bat la version standard : sa courbe est au-dessus. Que lui demandez-vous avant d'y croire, et comment rapporteriez-vous la comparaison ? »

*104 questions, 21 chapitres.*

<!-- pf:end bank -->
