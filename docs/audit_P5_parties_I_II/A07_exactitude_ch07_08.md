# Audit P5, tour 2 — A7 (exact3) : exactitude des chapitres 7 et 8

*Relecteur : sous-agent exact3 · 2026-10-04 · dépôt lu au commit `0863f10`, sans modification (`git -C /home/claude/workbookia status --short` vide à la fin).*

**Méthode.** Lecture intégrale des fiches, énoncés, indices, solutions, flashcards et figures des ch. 7 et 8 ; Markdown et sorties des deux `05_solutions.ipynb` extraits par script ; sections 7 et 8 du formulaire, entrées du glossaire, des erreurs fréquentes et de l'antisèche scikit-learn. Tous les calculs papier (✏️, ∂, 🧮, quiz, rappels, mini-exemples de la fiche) refaits en Python. Affirmations sur le code rejouées avec la référence `mylearn_ref`, scikit-learn 1.6.1 et NumPy 2.1.3 dans une copie (`exact3/copy/`, scripts dans `exact3/py/`). Variantes rejouées (dont 7.25 avec $D$ et $D^4$, et 8.27). 14 affirmations datées vérifiées en ligne (tableau en fin de rapport). Je ne re-signale pas les constats des §7–8 de `docs/AUDIT_parties_I_II.md`, les décisions du §22 ni les chantiers P1–P5.

**Aucun constat MAJEUR.** 6 MINEURS, 5 SUGGESTIONS.

## Constats

### MINEURS

**1. MINEUR · `chapitres/ch08_train_test/flashcards.csv:18` et `:19` · tag qui contient une espace : le `--check` des flashcards échoue (régression du commit e96506d)**
- Extrait (fin des deux lignes) : `;dlwb::ch08::erreur type`
- Problème : dans Anki, la colonne des tags sépare les tags par des espaces. Ces deux cartes reçoivent donc les tags `dlwb::ch08::erreur` et `type`, ce dernier étant un tag racine parasite, hors de `dlwb::`. BIBLE §12 (l. 270) : « Tags : `dlwb::chXX::concept` ». Le commit e96506d (« "erreur type" without hyphen (… chapter 8 syllabus and flashcards …) ») a remplacé `dlwb::ch08::erreur-type` (version précédente) par `dlwb::ch08::erreur type`. Or la règle de vocabulaire s'applique au texte, pas à un identifiant. Vérifié : `python tools/export_flashcards.py --check` affiche « ⚠️ chapitres/ch08_train_test/flashcards.csv:18 : tags attendus sous la forme dlwb::chXX::concept » (puis la même chose pour `:19`), puis « 320 carte(s) dans 13 fichier(s). », et sort avec le code 1. Cette commande figure parmi les vérifications de `CLAUDE.md`. Aucun test pytest ne lit les vrais `flashcards.csv` (`tests/infra/test_tools.py:266-281` ne teste que des fichiers temporaires) : la régression est passée inaperçue.
- Correction : mettre le tag `dlwb::ch08::erreur-type` (ou `dlwb::ch08::erreur_type`) sur les deux lignes, et ajouter un garde-fou dans `tests/infra/test_repo_structure.py` :
  ```python
  def test_published_flashcards_pass_the_check():
      import export_flashcards
      lines = []
      code = export_flashcards.export(ROOT / "exports" / "unused.csv", check_only=True, out=lines.append)
      problems = [line for line in lines if line.startswith("⚠️")]
      assert code == 0, f"expected no flashcard problem, got {len(problems)}: " + " | ".join(problems)
  ```
  (Avec `check_only=True`, `export` rend la main avant d'écrire.)

**2. MINEUR · `chapitres/ch08_train_test/05_solutions.md:299` ; `05_solutions.ipynb`, cellule 120 (💡, `tools/chapters/build_ch08.py:2105`) · « tout plus proche voisin vaut 1 » est faux**
- Extrait : « le point de départ échoue : l'accuracy d'entraînement de tout plus proche voisin vaut 1 (8.12), et il retient donc le premier 1-NN de la liste (longueur et épaisseur du bec, brutes), qui fait 0,81 sur le test. »
- Problème : `OneNN` (`build_ch08.py:454`) recopie le label du point le plus proche, « the first one on ties ». Un manchot d'entraînement qui a exactement les deux mêmes mesures qu'un manchot de l'autre sexe placé avant lui reçoit donc le mauvais label. J'ai recalculé avec le `PairModel` et le `OneNN` du notebook (233 manchots d'entraînement, `default_rng(42)`, script `exact3/py/ch8c.py`). Accuracy d'entraînement des 12 candidats 1-NN :
  - 1,0 pour les paires (0, 1) et (0, 3), brutes ou standardisées ;
  - 0,9785 pour (0, 2), (1, 2) et (2, 3) : 5 groupes de doublons de sexes différents (la nageoire est mesurée au millimètre, la masse à 25 g près) ;
  - 0,9871 pour (1, 3).

  La conclusion tient : le premier candidat, (0, 1) brut, est à 1,0, et l'argmax donne bien 0,81 sur le test. Mais la justification est fausse, et un apprenant qui affiche les 24 scores d'entraînement verra le démenti. 8.12 portait sur les quatre mesures, sans doublon.
- Correction : « le point de départ échoue : un plus proche voisin recopie le label de chaque manchot d'entraînement, sauf quand deux manchots de sexes différents ont exactement les mêmes mesures (8.12) ; son accuracy d'entraînement vaut donc 1 ou presque (1 pour la longueur du bec associée à son épaisseur ou à la masse, 0,98 ou 0,99 pour les autres paires), et le point de départ retient le premier 1-NN de la liste (longueur et épaisseur du bec, brutes), qui fait 0,81 sur le test. » Même texte dans la `note=` de `build_ch08.py:2105`, puis reconstruire le notebook.

**3. MINEUR · `chapitres/ch08_train_test/04_indices.md` et `chapitres/ch07_classification/04_indices.md` (indice 3 des quiz et rappels) · l'indice 3 donne les réponses finales, comme aux ch. 10 et 11**
- Extraits :
  - ch. 8, l. 115 (8.Q6) : « A empêche une fuite (des doublons des deux côtés du découpage) ; C est la règle à suivre. B : la sélection a regardé les labels de toutes les lignes. D : un même patient peut se retrouver des deux côtés. E : la date de sortie n'est pas connue au moment de l'ad[mission] ». La réponse attendue est `"BDE"`.
  - ch. 8, l. 79 (8.Q4) : « Le décor. Dans le même lot, il accompagne toujours la même classe : le raccourci réussit aussi sur le test. Pour le démasquer, il faut une photo où le décor et la classe se contredisent. » Cela donne a) B, b) Faux et c) C.
  - ch. 8, l. 187 (8.Q10) : « trois folds de 201, deux de 200 ». Cela donne b) 201 et c) 3.
  - ch. 7, l. 97 (7.Q5) : « Cinq features : un point de $\mathbb{R}^5$, et une frontière de dimension $5 - 1$ ». Cela donne a) 5 et d) 4.
  - ch. 7, l. 115 (7.Q6) : « 7 classes, 7 modèles ».
  - ch. 7, l. 133 (7.Q7) : « En d), A et B sont à égalité : la règle de `mylearn` garde le plus petit indice ».
  - ch. 7, l. 151 (7.Q8) : « k-means travaille sans labels, et rend exactement $k$ clusters. $k$ est fixé avant l'entraînement. Le résultat dépend du départ […] ». Cela donne a) à d).
- Occurrences :
  - ch. 8 : 8.Q1 (l. 25), Q2 (l. 43, b et c), Q4 (l. 79), Q5 (l. 97), Q6 (l. 115), Q9 (l. 169), Q10 (l. 187), Q11 (l. 205) et R1 (l. 227), soit 9 des 14 quiz et rappels ;
  - ch. 7 : Q2 (l. 43, c et d), Q5 (l. 97), Q6 (l. 115, a et c, « elles font 1,47 »), Q7 (l. 133, d) et Q8 (l. 151).

  Les indices 3 des exercices papier et des autres quiz (7.Q4, Q9, Q11, 7.R1 à R3, 8.Q7, 8.R2) posent le calcul sans la valeur, comme le veut BIBLE §12 (l. 261 : « presque la solution : pseudo-code ou première ligne »).
- Problème : c'est le défaut du constat 13 du §8 de `docs/AUDIT_parties_I_II.md`, reporté pour les ch. 10 et 11. Ce constat affirme que « les ch. 4 à 8 s'arrêtent le plus souvent au calcul posé » : c'est faux pour les quiz du ch. 8 et pour cinq quiz du ch. 7. Qui ouvre l'indice 3 recopie les réponses que vérifie la partie 0.
- Correction : à traiter avec le constat 13 (même contrôle sur tous les chapitres). Exemples de réécriture sur le modèle du ch. 9 :
  - 8.Q6 : « Pour chaque pratique, une question. A : que deviennent deux doublons si on les supprime **avant** le découpage ? B : sur quelles lignes la corrélation a-t-elle été calculée ? C : d'où viennent la moyenne et l'écart-type ? D : où peuvent tomber deux clichés du même patient ? E : la date de sortie existe-t-elle au moment de l'admission ? »
  - 8.Q10 : « $1\,003 = 5 \times 200 + 3$ : répartis le reste sur les premiers folds. Au dernier tour, l'entraînement réunit tous les folds sauf le cinquième. Le *leave-one-out* fait un tour par exemple. »
  - 8.Q4 : « Cherche ce qui, sur ces photos, accompagne toujours la même classe sans être l'animal. Un test tiré du même lot contient-il le même biais ? Pour c), cherche une photo où cet élément et la classe se contredisent. »
  - 7.Q5 : « Un œuf décrit par $d$ mesures est un point de $\mathbb{R}^d$ ; une frontière a une dimension de moins que l'espace qu'elle coupe. »
  - 7.Q6 : « Un modèle par classe. En b) et d), la décision est un argmax des scores, même quand aucun n'est positif ; en c), additionne les quatre probabilités de b). »

**4. MINEUR · `annexes/glossaire.md:183` · la définition de « fuite de données » ne suit pas la définition retenue au ch. 8 (§22)**
- Extrait : « | fuite de données | data leakage | une information sur la réponse qui se glisse dans les features ou l'entraînement : le score devient trop beau | 1 | »
- Problème : le §22 (2026-10-02, « Fuite de données (définition du workbook, ch. 8) ») et la fiche du ch. 8 (l. 51 et l. 123) définissent la fuite ainsi : « le modèle, ou les choix qui l'ont construit, profitent d'une information qu'ils n'auraient pas dans l'usage réel : le plus souvent une information venue du jeu de test, parfois une information qui n'existera qu'après le moment de la prédiction ». Le ch. 1 (`05_solutions.md:164`) va dans le même sens. Les formes les plus courantes enseignées au ch. 8 ne sont pas « une information sur la réponse » : standardisation calculée test compris, doublons ou patients des deux côtés du découpage, réglages choisis en regardant le test (8.Q5 c, 8.Q6 A, C et D, 8.24, 8.25). Un apprenant qui révise avec le glossaire ne les reconnaîtra pas comme des fuites, alors que 8.Q6 attend D parmi elles.
- Correction : « | fuite de données | data leakage | le modèle, ou les choix qui l'ont construit, profitent d'une information qu'ils n'auraient pas en usage réel : le plus souvent venue du jeu de test (statistiques, doublons, choix faits en le regardant), parfois une feature connue seulement après la prédiction ; le score devient trop beau | 1, 8 | »

**5. MINEUR · `chapitres/ch08_train_test/01_fiche.md:237` (🕰️ tests et contamination) · LiveBench ne renouvelle pas ses questions « chaque mois »**
- Extrait : « des benchmarks comme LiveBench renouvellent leurs questions chaque mois, avec des réponses vérifiables, pour la limiter. »
- Problème : c'est l'intention affichée (README, l. 36 : « releasing new questions monthly »), pas la pratique. Le changelog officiel liste 12 versions entre juin 2024 et janvier 2026 (2024-06-12, 06-24, 07-26, 08-31, 11-25 ; 2025-04-02, 04-25, 05-30, 10-03, 11-25, 12-23 ; 2026-01-08), avec des trous de quatre mois, et aucune depuis le 2026-01-08 (vérifié le 2026-10-04). « Chaque mois » est donc faux à cette date.
- Source : [changelog de LiveBench](https://github.com/LiveBench/LiveBench/blob/main/changelog.md) ; README du même dépôt.
- Correction : « des benchmarks comme LiveBench renouvellent régulièrement une partie de leurs questions à partir de sources récentes (concours, articles, actualités), avec des réponses vérifiables, pour la limiter. »

**6. MINEUR · `chapitres/ch07_classification/05_solutions.md:278` (Ex 7.24) · des durées « de l'exécution du corrigé » qui ne sont pas celles du corrigé**
- Extrait : « Lors de l'exécution du corrigé, l'un-contre-un s'entraîne même plus vite que l'un-contre-tous (de l'ordre de 0,2 s contre 0,7 s) […] 45 duels (de l'ordre de 0,17 s) coûtent plus que 10 scores (0,05 s), et bien plus qu'un seul modèle natif (0,01 s). »
- Problème : le tableau imprimé par `05_solutions.ipynb` (cellule 111) donne :
  - `fit_s` : 0,1319 pour l'un-contre-un et 0,6247 pour l'un-contre-tous ;
  - `predict_s` : 0,1521 pour l'un-contre-un, 0,0412 pour l'un-contre-tous et 0,0094 pour le modèle natif.

  Le texte annonce donc 0,2 au lieu de 0,13, 0,17 au lieu de 0,15 et 0,05 au lieu de 0,04. Les conclusions qualitatives restent justes, mais la phrase présente ces durées comme celles du corrigé, et l'apprenant les compare au notebook. La note du notebook (`build_ch07.py:1968-1979`) évite à raison les chiffres : « Les durées varient d'une machine et d'une exécution à l'autre : compare leurs ordres de grandeur. »
- Correction : « Lors de l'exécution du corrigé, l'un-contre-un s'est même entraîné plus vite que l'un-contre-tous (0,13 s contre 0,62 s) […] À la prédiction, en revanche, chaque modèle coûte : 45 duels (0,15 s) coûtent plus que 10 scores (0,04 s), et bien plus qu'un seul modèle natif (0,01 s). Ces durées changent d'une machine et d'une exécution à l'autre : compare leurs ordres de grandeur. »

### SUGGESTIONS

**7. SUGGESTION · `chapitres/ch08_train_test/01_fiche.md:135` (🕰️ la taxonomie des fuites) · le décompte de Kapoor et Narayanan a été mis à jour depuis**
- Extrait : « S. Kapoor et A. Narayanan ont recensé des fuites dans **294 articles de 17 disciplines** scientifiques »
- Problème : le chiffre est exact pour l'article de 2023 (*Patterns* 4 (9), 100804 : « 17 fields … 294 papers », vérifié sur le texte intégral, Europe PMC PMC10499856 ; les huit types répartis en trois familles le sont aussi). Mais la liste que tiennent les auteurs a grandi : « 41 papers from 30 fields where errors have been found, collectively affecting 648 papers » (« Table updated in May 2024 », https://reproducible.cs.princeton.edu/).
- Correction : « ont recensé en 2023 des fuites dans **294 articles de 17 disciplines** scientifiques (648 articles dans 30 disciplines dans leur liste mise à jour en mai 2024) ». Ajouter la source [S. Kapoor et A. Narayanan, « Leakage and the Reproducibility Crisis in ML-based Science »](https://reproducible.cs.princeton.edu/).

**8. SUGGESTION · `chapitres/ch08_train_test/01_fiche.md:210` et `flashcards.csv:19` · il manque « universel » au résultat de Bengio et Grandvalet**
- Extraits : fiche, « ont montré qu'il n'existe **aucun** estimateur sans biais de la variance de la k-fold » ; flashcard, « Il n'existe pas d'estimateur sans biais de la variance de la k-fold (Bengio et Grandvalet, 2004) ».
- Problème : le théorème dit « there exists no universal (valid under all distributions) unbiased estimator of the variance of K-fold cross-validation » (résumé, *JMLR* 5, 2004). Le corrigé du CP2 le dit bien (`checkpoints/partie_2/03_examen_corrige.md:25` : « qui vaille pour toutes les distributions des données »). Le ch. 8 est plus absolu que sa source et que le checkpoint.
- Correction : dans la fiche, « ont montré qu'il n'existe **aucun** estimateur sans biais de la variance de la k-fold qui vaille pour toutes les distributions des données » ; dans la flashcard, « Il n'existe pas d'estimateur sans biais de cette variance valable pour toutes les distributions (Bengio et Grandvalet, 2004) ».

**9. SUGGESTION · `chapitres/ch07_classification/flashcards.csv:18` · la définition de $b$ (silhouette) est ambiguë**
- Extrait : « avec \(a\) la distance moyenne aux autres points de son cluster et \(b\) celle au cluster voisin le plus proche »
- Problème : « le cluster voisin le plus proche » se lit facilement comme « le cluster dont le centre est le plus proche ». Or $b$ est la plus petite des distances moyennes aux points de chaque autre cluster (fiche, l. 217 ; glossaire, l. 368). Les deux ne coïncident pas toujours. Exemple : un point en $(0, 0)$, un cluster $\{(-3, 0), (3, 0)\}$ (centre à distance 0, distance moyenne 3) et un cluster $\{(2, 0)\}$ (distance 2) ; on a alors $b = 2$, donné par le second cluster.
- Correction : « et \(b\) la plus petite de ses distances moyennes aux points d'un autre cluster ».

**10. SUGGESTION · `chapitres/ch07_classification/01_fiche.md:56` et `flashcards.csv:14` · « ne peut que baisser »**
- Extraits : « L'**inertie** ne peut que baisser. » ; « Pourquoi l'inertie de k-means ne peut-elle que baisser, et pourquoi n'atteint-on pas forcément son minimum ? »
- Problème : à chaque étape de Lloyd, l'inertie baisse **ou reste égale** (elle reste égale à la convergence). Le verso de la carte le dit bien (« aucune étape n'augmente l'inertie »), et 8.R1 b) aussi (« ne peut que baisser, ou rester égale »).
- Correction : « L'**inertie** ne peut pas augmenter d'une étape à l'autre. » Recto de la carte : « Pourquoi l'inertie ne peut-elle pas augmenter pendant les itérations de k-means, et pourquoi n'atteint-on pas forcément son minimum ? »

**11. SUGGESTION · `annexes/erreurs_frequentes.md:236` · faire le lien avec la règle d'une erreur type de MP2**
- Extrait : « présenter la moyenne et la dispersion, sans en tirer un intervalle ; comparer les modèles sur les mêmes folds (fiche du ch. 8, §8.5.1) »
- Problème : ce n'est pas une contradiction. MP2 (`projets/partie_2_california_validation/README.md:33`, `solution/README.md:24`) se sert de $\sigma/\sqrt{k}$ pour la règle d'une erreur type, en le disant « une approximation optimiste » (décision du §22), et le corrigé de CP2.1 d fait le lien (« ne sert que d'ordre de grandeur »). Il manque seulement cette passerelle dans l'annexe, que l'apprenant consulte justement quand il cherche une erreur.
- Correction : ajouter à la fin de la cellule « ; s'en servir tout au plus comme ordre de grandeur, en le disant optimiste (règle d'une erreur type de MP2) ».

## Affirmations datées vérifiées

| # | Affirmation (fichier:ligne) | Verdict | Source |
|---|---|---|---|
| 1 | Détecteur de chars : légende urbaine, partie d'une question hypothétique du début des années 1960, première version imprimée retrouvée de 1992 (ch. 8, fiche:99) | exact | [gwern.net/tank](https://gwern.net/tank) |
| 2 | Geirhos et coll., *Nature Machine Intelligence* 2, 665-673, 2020 (ch. 8, fiche:99) | exact | nature.com |
| 3 | Kapoor et Narayanan : 294 articles, 17 disciplines, huit types en trois familles ; *Patterns* 4 (9), 100804 (ch. 8, fiche:135 ; 📄 8.10) | exact pour 2023 ; liste mise à jour depuis (suggestion 7) | Europe PMC PMC10499856 ; arXiv 2207.07048 ; reproducible.cs.princeton.edu |
| 4 | Zech et coll. 2018 : prévalences de 34,2 %, 1,2 % et 1,0 % ; hôpital reconnu à 99,95 % et 99,98 % ; expérience de prévalence accentuée (meilleur score interne, pas de gain externe) (ch. 8, `05_solutions.md:146`, ⚖️ 8.9) | exact (AUC interne 0,739 → 0,899, externe 0,641) | PMC6219764 |
| 5 | CS231n : un seul découpage de validation plutôt qu'une validation croisée en deep learning (ch. 8, fiche:212) | exact | cs231n.github.io/classification |
| 6 | `GroupKFold(shuffle=True)` depuis la 1.6, `StratifiedGroupKFold` depuis la 1.0, `TimeSeriesSplit` (`gap`, `max_train_size`, `test_size`), `cv` entier → `StratifiedKFold` sans mélange pour un classifieur, $\lceil t \cdot n \rceil$ et refus de `stratify` avec `shuffle=False` (ch. 8, fiche:225) | exact | notes de version 1.6 ; vérifié avec scikit-learn 1.6.1 |
| 7 | `permutation_test_score`, $p = (C + 1)/(n_{\text{perm}} + 1)$, Ojala et Garriga 2010 (ch. 8, fiche:237) | exact | documentation 1.6 ; vérifié localement |
| 8 | Barz et Denzler 2020 : environ 3 % (CIFAR-10) et 10 % (CIFAR-100) de quasi-doublons dans le test (ch. 8, fiche:237) | exact | *J. Imaging* 6 (6), 41 |
| 9 | LiveBench renouvelle ses questions chaque mois (ch. 8, fiche:237) | inexact au 2026-10-04 (constat 5) | changelog GitHub |
| 10 | `KMeans` : k-means++ glouton, `n_init="auto"` depuis la 1.4 (un seul départ), `"full"` renommé `"lloyd"` en 1.1 (ch. 7, fiche:202) | exact | documentation de `KMeans` 1.6 ; vérifié localement |
| 11 | `DBSCAN` (`eps=0.5`, `min_samples=5`, le point compté) ; `HDBSCAN` depuis la 1.3, labels −1, −2 et −3, pas de `predict` (ch. 7, fiche:241) | exact | documentation 1.6 ; vérifié localement |
| 12 | Tous les classifieurs gèrent plusieurs classes d'office ; `SVC` et `NuSVC` en un-contre-un, `LinearSVC` en un-contre-tous ; l'OvR comme « fair default », l'OvO plus lent (ch. 7, fiche:152) | exact | guide « Multiclass » 1.6 |
| 13 | HNSW (*TPAMI* 42 (4), 2020, p. 824-836) ; « The Faiss library » (*IEEE TBD* 12 (2), 2026, p. 346-361) ; Beyer et coll., ICDT 1999 (ch. 7, fiche:304) | exact | IEEE ; arXiv 2401.08281 |
| 14 | Bengio et Grandvalet 2004 (ch. 8, fiche:210) | exact, mais sans « universel » (suggestion 8) | *JMLR* 5 |

## Vérifié sans constat

- **Calculs.**
  - ch. 7 : toutes les réponses des quiz, des rappels, des ✏️, du ∂ et des 🧮 (seuil $t^* = C_{FP}/(C_{FP} + C_{FN})$, Bayes, densité $n/b^d$, récurrence $V_d = \frac{2\pi}{d} V_{d-2}$, coquille $1 - (1 - \varepsilon)^d$, hyper-orange $\sqrt{d} - 1$, OvR et OvO, Lloyd à la main, silhouette, pureté).
  - ch. 8 : toutes les réponses aussi ($\lceil t \cdot n \rceil$, plus fort reste, tailles des folds, validation croisée imbriquée, erreur type, maximum de $K$ scores, $R^2$, McNemar exact 0,25 et $9 \times 10^{-10}$).
- **Sorties des notebooks et `05_solutions.md`.** Cohérents, hors constats 2 et 6.
- **Variantes rejouées.**
  - 7.25 : $D$ donne 4,69 paquets couverts ; $D^4$ donne 5,91 paquets et 97 % de meilleurs découpages. La solution annonce 4,7, 5,9 et 97 %.
  - 8.27 : le choix est `((1, 3), True, "centroid")` (0,880 contre 0,832), comme annoncé.
  - Les autres variantes de 8.24 à 8.26 concordent aussi.
- **API.** Les affirmations sur scikit-learn 1.6.1 et sur `mylearn_ref` (règles d'égalité OvR et OvO, `NearestCentroid`, `clone`, `cross_val_score`) sont justes.
- **Figures.** Elles concordent avec le texte :
  - `optimisme` : moyennes 0,750, 0,791 et 0,817 pour $K = 1$, 5 et 25 ;
  - `boites_8_8` : 3 valeurs aberrantes de C entre 0,688 et 0,735, 5 points sous zéro, écart B − A maximal d'environ 0,028 ;
  - `donnees_dependantes` (`TimeSeriesSplit`), `densite`, `silhouette`.
- **Formulaire, glossaire et annexes.** Les sections 7 et 8 du formulaire, l'antisèche scikit-learn et les autres entrées du glossaire et des erreurs fréquentes sont exactes.
- **Contradictions.** Aucune avec les ch. 1 à 6 et 9 à 11, CP2 ou MP2 ; pour l'usage de $\sigma/\sqrt{k}$, la passerelle reste à ajouter (suggestion 11).

## Bilan

**Solide.**
- Tous les calculs et toutes les réponses vérifiées des ch. 7 et 8 sont justes, et les nombres des solutions concordent avec les sorties du notebook (hors durées de 7.24).
- La référence suit les docstrings figées.
- Les affirmations sur scikit-learn 1.6 sont exactes.
- Les 🕰️ sont précis et bien sourcés, jusqu'aux chiffres fins (Zech, Barz et Denzler, Kapoor et Narayanan).
- Les figures concordent avec le texte.

**Trois risques principaux.**
1. **Les corrections transversales d'un tour d'audit peuvent casser un garde-fou que pytest ne couvre pas.** Ici, le tag `erreur type` fait échouer `export_flashcards.py --check` sans qu'aucun test ne le voie (constat 1). Il faut un test qui valide les vrais fichiers publiés.
2. **Les indices de niveau 3 donnent les réponses des quiz** dans tout le ch. 8 et dans une partie du ch. 7, contrairement à ce que suppose le constat 13 (constat 3). Le contrôle annoncé doit couvrir tous les chapitres.
3. **Le texte rédigé à la main peut diverger des sorties, et le contenu daté vieillit.** C'est le cas de la justification de 8.27 et des durées de 7.24 dans `05_solutions.md` (constats 2 et 6), de LiveBench et du décompte de Kapoor (constat 5, suggestion 7). Il faut tirer ces phrases des sorties, ou éviter les chiffres non reproductibles, et dater les décomptes.

*État : `git -C /home/claude/workbookia status --short` est vide ; aucun processus lancé par cet audit n'est actif. Je n'ai écrit que dans `scratchpad/s24/audit2/exact3/` (copie, scripts et ce rapport).*
