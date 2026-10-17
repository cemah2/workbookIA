# Audit P5 parties III et IV — relecteur A9 « checkpoints » (CP3, CP4, MP3, MP4)

Périmètre : `checkpoints/partie_3`, `checkpoints/partie_4`, `projets/partie_3_revenus` (MP3), `projets/partie_4_mini_framework` (MP4), `docs/syllabus/data/cp3.json` et `cp4.json`, scripts `build_cp3/cp4/mp3/mp4.py` et `figures_cp3/cp4.py`, et leur cohérence avec CP1, CP2, MP1, MP2, les ch. 12 à 20, les stubs et le formulaire. Les décisions du §22 (types des examens, CP3.7 et CP4.9 en 🔨, durées, choix de MP3 et MP4…) ne sont pas re-signalées.

**Ce que j'ai vérifié.** Tout s'est fait en lecture seule. Ce qui s'exécute a tourné dans une copie `tar`, avec le `WB_ROOT` de la copie. À la fin, `git status --short` du dépôt est vide et aucun processus ne reste.
- **Barèmes** : sommes par sous-question recalculées par script, dans le sujet et dans le corrigé. Les points donnés en prose (« 0,25 chacun », « 0,15 pour a à d ») ont été comptés à la main. Résultat : 20/20 pour CP3 et pour CP4. Durées : 111 et 114 min. Les deux examens ont 2 points (10 %) sur les parties antérieures.
- **Réponses papier** : toutes les réponses de CP3 et CP4 ont été recalculées en Python (NumPy ; PyTorch en float64 pour CP4.2, CP4.3, CP4.4 et CP4.13). Elles sont toutes justes, et les valeurs des erreurs fréquentes aussi, avec **une exception** (constat 2).
- **Valeurs de code** :
  - CP3.6 : k = 10 ; R² = 0,7367 ; `best_score_` = 0,6882 ; collègue 0,9337 ; sans (A), 0,7388 et 0,7367.
  - CP3.7 : (0,23 ; 173) ; test 163 et 231 ; avec `>`, (0,22 ; 176) ; coûts échangés, (0,75 ; 120) ; coûts cités en f).
  - CP3.1 : `decision_function` en 1D, `UserWarning` puis 3 candidats, `ValueError`, vérifiés sous scikit-learn 1.6.1.
  - CP3.8 : figure régénérée par `figures_cp3.py`. Meilleure profondeur 8, 56 arbres ; 0,493 à 395 arbres, 0,4933 à 400.
- **Remédiations** : tous les ID cités par les deux corrigés existent (script sur `ch*.json`), et chacun porte sur la notion visée.
- **Synthèses** :
  - toutes les fonctions `mylearn` citées existent dans les stubs, dans le bon module (script) ;
  - les formules ont été comparées aux fiches, au formulaire et aux docstrings ;
  - les deux cartes Mermaid s'affichent (mermaid-cli, avec le Chromium de Playwright).
- **`wb.check`** : une trentaine de saisies essayées (formats « 2,5 », « vrai », « adelie », matrices, entiers). Bonnes réponses acceptées, erreurs classiques reconnues.
- **« Démarrer » dans la copie** :
  - `start_chapter.py --init`, puis `CP3` et `CP4` ;
  - tests des kits vides : les ⏳ nomment la fonction qui manque ;
  - « Run all » avec nbclient des deux examens vides et des deux kits vides : aucune erreur ;
  - les deux kits **remplis** avec le code et les cellules de la solution : MP3 ouvre son coffre une fois et écrit `results.json` ; MP4 attrape 6 mutants sur 6 et garde le test fermé en mode rapide ;
  - `tests/infra/test_projects.py` est vert, et `syllabus.py check` signale 0 problème ;
  - les huit notebooks reconstruits par `build_cp3/cp4/mp3/mp4.py` ont des sources identiques à celles du dépôt ;
  - tests des solutions verts : 26 (MP3) et 23 (MP4), comme l'annoncent les README.
- **README de solution** : les chiffres ont été comparés à `results.json` et aux sorties des notebooks. MP3 : tous concordent. MP4 : constat 7.

## Constats

### MAJEUR

**1. MAJEUR (contrat ; enjeu pédagogique faible) · étoiles appliquées différemment en CP3 et CP4 (§17, « ★★ = deux notions combinées, une interprétation, une preuve ») · `docs/syllabus/data/cp4.json` (CP4.3, CP4.4), `checkpoints/partie_4/01_examen_sujet.md:69` et `:90`, contre `cp3.json` (CP3.5), `checkpoints/partie_3/01_examen_sujet.md:141`**
- **Extraits** :
  - `### CP4.3 — Deux pas de momentum, deux pas d'Adam, et le weight decay d'AdamW ✏️ ★` ;
  - `### CP4.4 — Dropout et batchnorm : mode entraînement contre mode évaluation ✏️ ★` ;
  - `### CP3.5 — Une itération d'AdaBoost et une de gradient boosting ✏️ ★★`. La note de `cp3.json` dit : « ★★ pour deux notions […] : CP3.4, CP3.5 ».
- **Problème** : CP3.5 juxtapose deux calculs (AdaBoost puis gradient boosting) et reçoit ★★. CP4.3 en juxtapose trois (momentum, Adam, AdamW) et, en plus, **combine** Adam et weight decay (f, g), avec une interprétation (h) ; il reçoit ★. CP4.4 (dropout et batchnorm, avec les « pourquoi » de c) et h)) reçoit ★ aussi. La même règle donne donc deux étoiles différentes à deux questions de même forme, dans deux checkpoints consécutifs. Les étoiles sont affichées dans le sujet, le tableau de bord et le SYLLABUS.
- **Correction** : appliquer une seule lecture de la règle, en prenant « combinées » au sens strict, puis corriger `cp3.json` ou `cp4.json`, les en-têtes du sujet et le tableau de bord (`syllabus.py build`), avec une ligne dans PROGRESS § Écarts :
  - soit CP3.5 passe à ★ (deux notions juxtaposées, chacune appliquée à un cas travaillé en 14.6 et 14.7) ;
  - soit CP4.3 passe à ★★ (f-g combinent Adam et weight decay, h est une interprétation).
  - La proposition : **CP4.3 à ★★**, CP3.5 et CP4.4 inchangés (en CP4.4, les deux couches restent séparées).

### MINEUR

**2. MINEUR · `checkpoints/partie_3/03_examen_corrige.md:52` (erreurs fréquentes de CP3.3)**
- **Extrait** : « prendre le seuil du milieu de l'intervalle (5,5 : Gini pondéré 0,4, loin du meilleur) ».
- **Problème** : le seuil 5,5 donne un Gini pondéré de **0,5**, pas 0,4. La ligne 48 du même corrigé le dit (« 5,5 → 0,5 »). Calcul : à gauche, {A, A, B, B}, Gini 0,5 ; à droite, {A, B}, Gini 0,5 ; donc (4 × 0,5 + 2 × 0,5)/6 = 0,5. Par ailleurs, le « milieu de l'intervalle » des x (de 1 à 8) vaut 4,5, ce qui désigne le candidat s = 4 (0,444), pas 5,5.
- **Correction** : « prendre le seuil du milieu de l'intervalle des x (4,5, donc le candidat s = 4 : Gini pondéré 0,444, loin du meilleur) ».

**3. MINEUR · CP3.6 c) : le sujet et le notebook ne donnent pas la même consigne · `checkpoints/partie_3/01_examen_sujet.md:170` contre `tools/chapters/build_cp3.py:76` (→ cellule 4 de `02_examen_notebook.ipynb` et de `03_examen_solutions.ipynb`)**
- **Extraits** :
  - sujet : « garde la ligne (B) telle quelle (d) se calcule avec elle) et retire les fuites » ;
  - notebook : « garde la transformation (B), qui n'est pas une fuite, et retire ce qui en est une ».
- **Problème** : le notebook, ouvert pendant l'examen, affirme que (B) n'est pas une fuite. Il retire ainsi un candidat de a), alors que le sujet laisse la question ouverte. La parenthèse imbriquée du sujet, « (d) se calcule avec elle) », se lit mal.
- **Correction** : la même phrase aux deux endroits, sans verdict, par exemple « garde la transformation de la ligne (B) telle quelle (la valeur de d) en dépend), et retire ce qui fuit ». Puis reconstruire, réexécuter les solutions et lancer `build_answers.py`.

**4. MINEUR · structure du corrigé de CP4, différente de celle de CP1, CP2 et CP3 · `checkpoints/partie_4/03_examen_corrige.md:220-228`, `:169-175`, `:199-216`**
- **Extraits** : la section finale s'intitule « ## Après la correction » et contient un tableau à trois lignes : « 16 ou plus », « de 12 à moins de 16 », « moins de 12 ». Ailleurs, on a « ## Barème récapitulatif et remédiation » (CP3 l. 234, CP2 l. 271, CP1 l. 246) et « **Lire ta note** » en quatre tranches.
- **Problème** :
  - le tableau récapitulatif (question, points, chapitre, exercices à refaire) manque, alors que la copie (`04_mes_reponses.md:89`) a une colonne « À revoir (remédiation du corrigé) » et que le README (l. 23) dit de refaire « les exercices indiqués par le corrigé » ;
  - CP4.10, CP4.13 et CP4.14 n'ont pas de ligne « **Erreurs fréquentes** » (11 lignes pour 14 questions, contre 14/14 en CP1 et CP3), alors que BIBLE §12 demande « les erreurs fréquentes » pour chaque exercice.
- **Correction** :
  - ajouter « ## Barème récapitulatif et remédiation », avec un tableau `| Question | Points | Ch. | Si tu as moins de la moitié des points, refais… |` sur les 14 questions (reprendre leurs lignes « Remédiation »), puis « **Lire ta note** » en quatre tranches, comme en CP3 ;
  - garder la consigne particulière de la tranche « moins de 12 » (🔨 des ch. 18 et 19, 📦 20.23) ;
  - ajouter les trois lignes manquantes, par exemple :
    - CP4.13 : « croire que le graphe libéré empêche un second `backward()` (chaque tour en construit un neuf) ; multiplier par $1 - p$ en évaluation (le dropout du livre) ; lire `nn.Linear.weight` en $(n_{\text{in}}, n_{\text{out}})$ » ;
    - CP4.10 : « réciter la formule du delta sans dire pourquoi c'est efficace ; confondre rétropropagation et descente de gradient » ;
    - CP4.14 : « dire que le hasard suffit, sans parler de l'échelle ; confondre symétrie et gradient évanescent ».

**5. MINEUR · le commit de `test_indices.npy` n'est pas un garde-fou du coffre (MP3, et déjà MP2) · `tools/chapters/build_mp3.py` (cellule 5 du notebook : `print(... "⏳ MP3.1 : test_indices.npy {detail} : commite-le avant d'aller plus loin")` ; cellule 34 : `blocking = sorted(cell for cell, failures in FAILED.items() if failures and cell < "MP3.6 test")`)**
- **Constaté** : j'ai rempli le kit avec la solution, sans commiter `test_indices.npy`. Résultat :
  - la cellule 5 affiche « ⏳ MP3.1 : test_indices.npy n'est pas encore commité : commite-le avant d'aller plus loin » ;
  - la cellule 34 affiche ensuite « 🔓 Coffre ouvert (1 ouverture(s) enregistrée(s) dans vault.json) » ;
  - seule la vérification finale de MP3.8 signale « ❌ test_indices.npy n'est pas encore commité ».
- **Problème** : le cahier des charges et la grille (« test stratifié commité avant toute modélisation », 4 points) en font la règle centrale du protocole, mais la porte irréversible ne la vérifie pas. Le cas « a changé depuis son commit » (`committed()` rend `False`) ne bloque pas non plus : un `test_indices.npy` réécrit après le commit laisse le coffre s'ouvrir. MP2 a la même faille (`build_mp2.py`, cellule MP2.1, un ⏳ seulement).
- **Correction** : dans la cellule du coffre (MP3.6 test), avant `vault.open`, appeler `state, detail = committed(INDICES)`. Si `state` est faux : « ⏳ MP3.6 : le coffre reste fermé : test_indices.npy {detail} ; commite-le (MP3.1, consigne d) et relance cette cellule. » Faire de même dans MP2, et ajouter un test dans `tests/infra/test_projects.py` si le kit le permet.

**6. MINEUR · message du coffre de MP4 en anglais · `projets/partie_4_mini_framework/solution/data.py:59` et `depart/data.py:59`**
- **Extrait** : `print("⚠️ The test set was already opened for another model: its score is no longer an honest measure " "(say so in your README).")`.
- **Problème** : ce message s'affiche dans le notebook de l'apprenant. Les coffres de MP2 et MP3 lui parlent en français : « ⚠️ Le coffre a déjà été ouvert le … pour un autre modèle. Ce nouveau score de … » (`partie_3_revenus/solution/data.py:87`). MP4 n'affiche pas non plus de confirmation (« 🔓 Coffre ouvert (n ouverture(s)…) »).
- **Correction** :
  - `print(f"⚠️ Le test a déjà été ouvert le {log[0]['date']} pour un autre modèle : ce nouveau score n'est plus une mesure honnête (dis-le dans ton README).")` ;
  - après l'écriture du journal, `print(f"🔓 Test ouvert ({len(log)} ouverture(s) enregistrée(s) dans test_log.json).")` ;
  - même fichier dans `depart/` et `solution/`, puisque `test_projects` le vérifie.

**7. MINEUR · README de la solution de MP4 : une valeur absente de `results.json` · `projets/partie_4_mini_framework/solution/README.md:68` et `:3` contre `solution/results.json` et `tools/chapters/build_mp4.py` (cellule MP4.6 e, dictionnaire `results`)**
- **Extraits** :
  - README : « | Meilleure accuracy de validation (15 epochs) | 0,8994 | 0,8997 | » et « à 0,03 point de PyTorch en validation » ;
  - `results.json` contient `"torch_best_val_accuracy": 0.8996999859809875` et `"val_accuracy": 0.8976` (le modèle final, batchnorm seule), mais aucune trace de la meilleure validation de minifw sur l'entraînement de comparaison.
- **Problème** :
  - le livrable 4 du cahier (l. 50) et `cp4.json` demandent dans `results.json` « minifw contre PyTorch : accuracy de validation et de test ». Ici, le seul chiffre minifw de la comparaison ne s'y trouve pas, et le notebook de solution (exécuté en mode rapide) ne le montre pas non plus. Le 0,8994 est invérifiable ;
  - « (15 epochs) » est inexact : l'early stopping (patience 3) peut s'arrêter avant.
- **Correction** :
  - ajouter au dictionnaire `"minifw_best_val_accuracy": max(main_history["val_acc"])` (et `"main_config": "les deux"`) ;
  - réexécuter en mode complet, ou reporter la valeur à la main dans `results.json` en le disant ;
  - écrire « (15 epochs au plus) ».

**8. MINEUR · CP4.9 a) contredit la fonction du ch. 19 sans le dire · `checkpoints/partie_4/01_examen_sujet.md:167`, `03_examen_corrige.md:166`, contre `templates/mylearn_stubs/optim.py:876` (`clip_grad_norm`, « Rescale all the gradients IN PLACE »)**
- **Extraits** :
  - sujet : « `clipped` une liste de **nouveaux** tableaux […] (la règle de `torch.nn.utils.clip_grad_norm_`). Elle ne modifie pas `grads` » ;
  - corrigé : « **Erreurs fréquentes** : […] `g *= coef`, qui modifie les gradients reçus ».
- **Problème** : l'apprenant a écrit en 19.19 (et retrouvera dans MP4) une version **sur place**, comme `clip_grad_norm_`. L'examen, « de mémoire », demande l'inverse, et le corrigé classe parmi les erreurs le comportement de sa librairie et de PyTorch. La consigne est écrite, mais rien ne signale ce contre-pied.
- **Correction** :
  - dans le sujet (et la cellule 4 du notebook, `build_cp4.py`), ajouter : « (à la différence de `clip_grad_norm` du ch. 19 et de `clip_grad_norm_`, qui modifient les gradients sur place) » ;
  - dans le corrigé : « `g *= coef` (la version sur place du ch. 19 et de PyTorch : ici, le contrat demande de nouveaux tableaux) ».

**9. MINEUR · notation de RMSprop dans la synthèse de CP4 · `checkpoints/partie_4/05_synthese.md:58` et `:64` contre `chapitres/ch19_optimiseurs/01_fiche.md:202`, `annexes/formulaire.md:546` et `templates/mylearn_stubs/optim.py:387`**
- **Extraits** :
  - synthèse : « RMSprop | $v \leftarrow \rho v + (1 - \rho)g^2$ » et, pour la batchnorm, « ($\rho$ = `momentum` = 0,1 : ici le poids de la **nouvelle** valeur, à l'inverse du $\rho$ de RMSprop) » ;
  - fiche, formulaire et docstring : $v_t = \alpha\, v_{t-1} + (1 - \alpha)\,g_t^2$, avec `alpha` dans PyTorch. Le $\rho$ du ch. 19 est celui d'Adadelta (fiche l. 212).
- **Problème** : la synthèse renomme le coefficient de RMSprop, et la remarque de la ligne 64 s'appuie sur ce nom inventé.
- **Correction** :
  - ligne 58 : « $v \leftarrow \alpha v + (1 - \alpha)g^2$ (`alpha` = 0,99 dans PyTorch) » ;
  - ligne 64 : « à l'inverse de l'`alpha` de RMSprop et des $\beta$ d'Adam, poids de l'**ancienne** valeur ».

**10. MINEUR · seuil de coût minimal : « > » au ch. 7, « ≥ » en CP3 · `annexes/formulaire.md:293` et `chapitres/ch07_classification/01_fiche.md:105` contre `checkpoints/partie_3/01_examen_sujet.md:176`, `05_synthese.md:60` et `03_examen_corrige.md:138`**
- **Extraits** :
  - formulaire : « positif si $p > t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ | `y_pred = (p >= t).astype(int)` » ;
  - CP3 : « classe 1 quand la probabilité est **supérieure ou égale** au seuil » ; synthèse : « décider 1 si $p \ge \frac{c_{FP}}{c_{FP} + c_{FN}}$ ».
- **Problème** : la ligne du formulaire se contredit (`>` dans la formule, `>=` dans le code), et la synthèse de CP3 écrit $c$ minuscule là où le ch. 7 écrit $C$. À l'égalité, les deux décisions coûtent autant : rien de faux, mais deux conventions pour la même règle.
- **Correction** : formulaire et fiche du ch. 7 : « positif si $p \ge t^*$ (à l'égalité, les deux décisions coûtent autant) ». Synthèse et sujet de CP3 : $C_{FP}$, $C_{FN}$.

### SUGGESTIONS

**11. SUGGESTION · `checkpoints/partie_4/05_synthese.md:73-82` et `:169` — la synthèse, révisée juste avant l'examen, amorce trois questions**
- **Extraits** :
  - section 3 : « La loss ne descend pas sous un plancher, autour de 1,5 à 10 classes | une softmax avant `CrossEntropyLoss` », « La mémoire grossit au fil de l'epoch | `total += loss` », « Chaque epoch commence par un saut | l'optimiseur recréé à chaque epoch » ;
  - mode d'emploi : « (une passe avant et une passe arrière sur un réseau 2-2-1, deux pas d'Adam) ».
- **Problème** : les quatre bugs de CP4.7 (D, I, K, M) sont exactement ceux des lignes 74, 76, 81 et 82, et l'exemple d'entraînement annonce la forme de CP4.2 et CP4.3. Ce constat est de même nature que le n° 21 de l'audit des parties I-II pour CP2.
- **Correction** : dans le mode d'emploi, écrire « un petit réseau de ton choix, quelques pas d'un optimiseur ». Garder le tableau de diagnostic (c'est la matière), ou remplacer la ligne « Chaque epoch commence par un saut » par un symptôme qui n'est pas à l'examen (par exemple « la loss de validation est `nan` | une batchnorm sur un mini-batch d'un exemple, un `log(0)` »).

**12. SUGGESTION · `checkpoints/partie_3/03_examen_corrige.md:153` (CP3.8 g) et le message de `CP3.8e` dans `build_cp3.py`**
- **Extrait** : « **Vrai** : la validation du learning rate 0,1 baisse encore à 400 arbres (0,493) ».
- **Mesuré** (`figures_cp3.py`) : 0,4984 à 300 arbres, 0,4934 à 380, minimum 0,4932 à 395, puis 0,4933 à 400. La courbe est plate à la fin. « Vrai » reste défendable (« pourraient »), mais pas l'affirmation « baisse encore à 400 ».
- **Correction** : « **Vrai** : la validation du learning rate 0,1 n'est jamais remontée (0,498 à 300 arbres, 0,493 à 400 : elle baisse encore, de plus en plus lentement), et l'écart à l'entraînement (0,35) reste modéré ; l'early stopping trouverait le bon nombre d'arbres. »

**13. SUGGESTION · `projets/partie_3_revenus/solution/results.json` contre `solution/README.md:56` et `:73`, `MODEL_CARD.md:8`, `:53`, `:57`**
- **Problème** : les chiffres de la décision de seuil (le seuil de la prévalence, 0,416, et son F1 hors fold, 0,726) et ceux de l'expérience « sans `sex` » (PR-AUC 0,829, 0,829, 0,828 ; écarts de sélection 0,244, 0,234, 0,247) ne figurent que dans les sorties du notebook. Or la model card les recommande et le cahier demande « les choix […], l'audit » dans `results.json`.
- **Correction** : ajouter `"thresholds": {"best_f1": …, "prevalence": …}` et `"blind_sex": {...}` au dictionnaire écrit en MP3.7.

**14. SUGGESTION · `projets/partie_4_mini_framework/depart/README.md` : le modèle de README n'a pas de section « ## Pistes »**
- MP1, MP2 (ajoutée après l'audit précédent, n° 25) et MP3 (`depart/README.md:47`) en ont une, alors que le cahier de MP4 liste cinq extensions.
- Ajouter : « ## Pistes — TODO : ce que je ferais ensuite (une couche Conv2d par im2col, un autograd vectoriel, des connexions résiduelles…) ».

**15. SUGGESTION · porte du coffre de MP3 : `cell < "MP3.6 test"` (`build_mp3.py`, cellule 34)**
- Le filtre des garde-fous bloquants compare des **chaînes** : il marche aujourd'hui, parce que « MP3.6 hors fold » et « MP3.6 seuil » sont avant « MP3.6 test » dans l'ordre alphabétique. Mais une cellule renommée « MP3.6 validation » sortirait du filtre sans bruit.
- Remplacer par une liste explicite : `BEFORE_VAULT = ("MP3.1", "MP3.2", "MP3.3", "MP3.4", "MP3.5 défauts", "MP3.5 recherche", "MP3.5 choix", "MP3.6 hors fold", "MP3.6 seuil")`, puis `cell in BEFORE_VAULT`.

**16. SUGGESTION · calibrage de MP3.8 (`projets/partie_3_revenus/README.md:65`, 60 min)**
- L'étape demande en une heure : six tests au moins, `library_versions`, `save_model`, `load_model`, le README, la model card et la data card, la relance complète du notebook et le commit. Pour un débutant, c'est la plus dense des huit.
- Noter ce point dans PROGRESS (Calibrage) pour le retour de l'apprenant, ou répartir les documents sur MP3.1, MP3.2 et MP3.7, où le cahier les fait déjà commencer.

## Bilan

**Ce qui est solide.**
- Les deux examens sont justes : chaque valeur a été recalculée, une seule erreur fréquente est fausse (constat 2), et les barèmes, les temps et les 10 % tombent juste.
- Les vérifications de la partie B résistent aux formats naturels et nomment les erreurs classiques.
- Les remédiations pointent vers des exercices existants et pertinents.
- Les synthèses citent les bonnes fonctions de `mylearn`, et leurs cartes s'affichent.
- Les deux kits démarrent proprement : ⏳ explicites, « Run all » sans erreur, notebooks reconstruits à l'identique.
- Remplis avec la solution, ils vont jusqu'au bout : coffre de MP3 ouvert une fois, six mutants sur six attrapés en MP4, test de MP4 fermé en mode rapide.
- Les README des solutions concordent avec `results.json`, sauf une valeur (constat 7).

**Les trois risques principaux.**
1. **Le protocole du test n'est pas entièrement verrouillé** : le coffre de MP3 (et de MP2) s'ouvre sans que `test_indices.npy` soit commité, ou alors qu'il a changé depuis son commit (constat 5). C'est la règle que la grille note le plus lourdement.
2. **La dérive entre checkpoints** : les étoiles du §17 sont appliquées différemment en CP3 et CP4 (constat 1). Le corrigé de CP4 n'a ni le tableau de remédiation ni toutes ses erreurs fréquentes (4). Le message du test de MP4 est en anglais (6), et MP4 n'a pas de « Pistes » (14).
3. **Des conventions qui divergent des chapitres** : `clip_grad_norm` sur place au ch. 19, contre de nouveaux tableaux à l'examen (8) ; $\alpha$ ou $\rho$ pour RMSprop (9) ; $>$ ou $\ge$ pour le seuil de coût (10). S'y ajoute une synthèse de CP4 qui annonce l'examen (11).
