# Audit P5 (tour 2) — checkpoints I et II, mini-projets MP1 et MP2

Relecteur « checkpoints ». Périmètre : `checkpoints/partie_1/*`, `checkpoints/partie_2/*`, `projets/partie_1_detecteur_langue/`, `projets/partie_2_california_validation/`, `docs/syllabus/data/cp1.json` et `cp2.json`, BIBLE §17 et §22 (lignes des checkpoints), scripts `build_cp1/cp2/mp1/mp2.py`, `figures_cp1/cp2.py`. Les points des sections 7 et 8 de `docs/AUDIT_parties_I_II.md` (P1 à P5, dont « époque », « surapprend », « lot », etc.) et les décisions du §22 ne sont pas re-signalés.

**Ce qui a été vérifié** (tout dans une copie `tar`, `mon_travail/` jamais touché ; dépôt propre à la fin, `git status --short` vide, aucun processus laissé) :
- barèmes : somme par question (sujet et corrigé) recalculée par script, 20/20 pour les deux examens ; répartitions par chapitre de `cp1.json`/`cp2.json` recalculées (CP1 : 1,25 + 4,25 + 3,75 + 3,25 + 2,75 + 2,75 + 2 ; CP2 : 4 + 2,75 + 4,75 + 2 + 2 + 2,5 + 2) ; 117 min chacun ;
- toutes les réponses chiffrées des deux corrigés recalculées en Python (dont CP1.3 −0,5685/−0,5676 et X.mean() ≈ 1 117, CP1.5 d percentiles 2,5/97,5 des Gentoo = [4 200 ; 6 000], CP1.7 AUC Φ(2,5/√2) = 0,9615 et FPR ≈ 0,049, CP1.13 intervalle de Fisher [0,10 ; 0,66], CP2.3 inertie 10 = optimum global par force brute, CP2.8 convergence à la 9ᵉ époque avec w = (−3, −2), b = 4, CP2.10 0,7163/0,6811/0,7285 et 0,68108 contre 0,68107) ; lectures des figures CP1.7 et CP2.7 contrôlées contre leurs scripts ;
- remédiations : tous les renvois d'exercices des deux corrigés existent (script sur les `chXX.json` : 56 pour CP1, une soixantaine pour CP2), et les sections de fiche citées aussi ;
- synthèses : chaque fonction `mylearn` citée existe dans `templates/mylearn_stubs/` (bon module, bonne signature) ; formules comparées au formulaire et aux fiches ; cartes Mermaid rendues avec mermaid-cli (les deux s'affichent) ;
- mini-projets : README de portfolio confrontés à `results.json` et aux sorties (MP2 : tous les chiffres concordent, données recalculées sur les 16 512 districts : 809 plafonnés, 1 018 à 52 ans, 44 à 15, maximum d'`AveOccup` 599,7) ; versions plotly 5.24.1 et folium 0.20.0 revérifiées dans le `pip-freeze.txt` de Colab ;
- « Démarrer » dans la copie : `start_chapter.py --init`, `CP1`, `CP2` ; tests des projets depuis la racine ; « Run all » (nbclient, FAST_MODE) des deux notebooks d'examen et des deux kits **vierges**, puis des deux kits **remplis** avec le code de la solution (commit de `test_indices.npy` fait comme l'indique le notebook, coffre ouvert une fois, `results.json` identique) ; les quatre scripts `build_*` reconstruisent des notebooks sans aucune différence de source ; ≈ 40 saisies de formats essayées dans `wb.check`.

## Constats

### MAJEUR

**1. MAJEUR (convention à décider) · `projets/partie_1_detecteur_langue/README.md:32` et `:77`, `depart/README.md:43`, `solution/README.md:77`, `tools/chapters/build_mp1.py:5-6, 25-26, 644` / `projets/partie_2_california_validation/README.md:48`**
- Extraits : MP1 « 3. `notebook.ipynb` : propre, exécuté de bout en bout, avec tes commentaires. » ; MP2 « 4. `mp2_california.ipynb` : propre, exécuté de bout en bout (« Run all ») ».
- Problème : deux conventions pour le même livrable. Dans un portfolio GitHub, `notebook.ipynb` ne dit rien du projet, et MP3 à MP6 n'ont pas de règle. `tests/infra/test_projects.py` déduit les noms de `solution/` : les deux marchent, mais le choix doit être fait une fois.
- Correction : fixer au §22 la convention `mpN_<sujet>.ipynb` (dépôt et kit), renommer maintenant `notebook.ipynb` en `mp1_detecteur_langue.ipynb` dans `depart/` et `solution/` (`STARTER`/`SOLUTION` de `build_mp1.py`, la cellule d'en-tête l. 644, les cinq mentions des README), relancer la solution, et une ligne dans PROGRESS (Écarts). À faire avant que l'apprenant copie le kit (`start_chapter.py` n'écrase jamais : une copie déjà faite garderait l'ancien nom).

### MINEUR

**2. MINEUR · `docs/syllabus/data/cp2.json:403` → `docs/SYLLABUS.md:2093` ; `cp1.json:439` → `SYLLABUS.md:1404`** — le paragraphe `study_notes`, publié dans le SYLLABUS (que le README, l. 52, donne à l'apprenant comme « plan détaillé »), dévoile des réponses et garde l'historique de génération.
- Extraits (CP2) : « MP2 : le nombre de zones se choisit par validation croisée et non par la silhouette, qui préfère k = 2 (0,755) alors que 2 zones n'apportent rien (0,5816 contre 0,5815) ; […] référence : degré 3, Ridge α = 300, 256 zones, 0,498 ± 0,010 en validation croisée et 0,478 [0,459 ; 0,498] sur le test » ; « CP2.10 vérifie la standardisation par fold sur ce que reçoit ridge_fit » ; « CP2.1 f porte désormais sur le R² en validation (le diagnostic des courbes proches était déjà demandé en CP2.7 e) » ; « Types présents : ✏️, ∂, 📈, 🐛, 🔨, 🗣️ (🔮, 🔬, 🎨 et 🏆 relèvent du mini-projet) […] les mentions de 🗣️ et de « 🔮, 🔬, 🎨 et 🏆 relèvent du mini-projet » ci-dessus sont caduques ». (CP1) « Les 🧠 et 💼 n'ont pas de type propre dans l'examen parce que le validateur impose les suffixes Q et E […] (les mentions ✏️ ou 🗣️ ci-dessus pour ces items sont caduques). »
- Problème : la question centrale de MP2.4 (silhouette contre validation croisée) et les choix de la solution, que le cahier des charges demande de n'ouvrir qu'à la fin, sont lisibles dans le plan ; CP2.10 a) (quelles étapes fuient) est en partie donnée ; le paragraphe se contredit (types) et parle de « validateur », « brief », « session 2 ».
- Correction : réécrire les deux `study_notes` à l'état final, sans résultats ni historique (déplacer l'historique dans `notes`, non publié, ou dans PROGRESS). Pour CP2 : « Examen blanc de 117 minutes (2 h au plus), noté sur 20 : ch. 7 = 4 pts (CP2.2-2.4), ch. 8 = 1,5 pt (CP2.5) + moitié de CP2.10 et CP2.12, ch. 9 = 3,5 pts (CP2.6-2.7) + moitié de CP2.10 et CP2.12, ch. 10 = 2 pts (CP2.8), ch. 11 = 2 pts (CP2.9, CP2.11), transversal 2,5 pts (CP2.1) ; 10 % (2 pts, CP2.13) sur la partie I. Types : 🧠, ✏️, ∂, 📈, 🐛, 🔨, 💼. Les questions de code (CP2.10, CP2.11) se font dans la partie A du notebook ; la partie B vérifie les réponses chiffrées après l'examen, le barème du corrigé fait la note. Mini-projet MP2 (≈ 10 h) après l'examen. Examen, synthèse et mini-projet font partie des quatre parcours. » Même traitement pour CP1 (types : 🧠 ✏️ ∂ 🔨 🐛 📈 🔮 🗣️ ⚖️ 💼), puis `syllabus.py build`.

**3. MINEUR · `tools/syllabus.py:667` → `docs/SYLLABUS.md:1354, 2044` et colonnes « Points » des deux examens**
- Extraits : « examen blanc 117 min sur 20.0 points » ; lignes CP1 « 1.5 », CP2 « 1.0 », « 2.0 », « 2.5 » (CP4, l. 3431 : « sur 20 points »).
- Problème : point décimal anglais et « .0 » parasite dans un texte français ; CP1 (`1`, `2` entiers dans `cp1.json`) et CP2 (`1.0`, `2.0` dans `cp2.json`) s'affichent différemment.
- Correction : formater les points dans `syllabus.py` (`f"{v:g}".replace(".", ",")`) pour l'en-tête et la colonne, puis `syllabus.py build`.

**4. MINEUR · `tools/chapters/build_cp1.py:497-499` → `checkpoints/partie_1/02_examen_notebook.ipynb`, cellule 37 (« Bilan »)**
- Extrait : `right = sorted(k for k, r in done.items() if r)` (idem `wrong`, `pending`).
- Problème : tri alphabétique ; `sorted(['CP1.1a', 'CP1.10a', 'CP1.11b', 'CP1.2c', 'CP1.9e'])` donne `['CP1.10a', 'CP1.11b', 'CP1.1a', 'CP1.2c', 'CP1.9e']` : la liste « À revoir avec le corrigé » sort dans le désordre. CP2 le fait déjà bien (`natural`, `build_cp2.py:765-775`).
- Correction : reprendre `natural` (regex `CP1\.(\d+)(.*)`) et `key=natural` pour les trois listes ; reconstruire, réexécuter les solutions, `build_answers.py`.

**5. MINEUR · partie B des notebooks d'examen : formats annoncés différents d'un checkpoint à l'autre, et un exemple non neutre**
- Extraits : `build_cp1.py:379-384` `("a", "True or False", …)` (cellules 19, 21 et 35 du notebook CP1) contre `build_cp2.py:579` `'True / False (ou "vrai" / "faux")'` ; CP1 normalise les lettres (`as_letters`, `WRAP = {"CP1.10a": "as_letters"}`, l. 227 et 509), pas CP2 : `wb.check("CP2.9a", ["N","S","N","V","N","N"])` répond « J'attends une chaîne de caractères (str) ; j'ai reçu un objet de type `list`. » (essayé) ; `build_cp2.py:265` : « les lettres s'écrivent entre guillemets (`"NSN"`) ».
- Problème : un apprenant français ne sait pas, en CP1, que « vrai »/« faux » sont acceptés ; une liste de lettres passe en CP1, pas en CP2 ; `"NSN"` est le début de la bonne réponse de CP2.9 a (`"NSNVNN"`, l. 723), contraire à la règle du §22 (2026-09-30) « Exemples de format […] toujours neutres, jamais une vraie réponse ».
- Correction : CP1, commentaires `'True / False (ou "vrai" / "faux")'` ; CP2, copier `as_letters` et `WRAP = {"CP2.9a": "as_letters", "CP2.9b": "as_letters"}` ; remplacer `"NSN"` par `"XYZ"`.

**6. MINEUR · `checkpoints/partie_2/03_examen_corrige.md:284` contre `:212`**
- Extraits : tableau « | CP2.10 | 1,5 | 8, 9 | 8.24, 8.25, 9.28 | » ; question « **Remédiation** : 8.24, 8.25, 8.3, 8.Q6 ; fiche du ch. 8, §8.5.1 […] ».
- Problème : seule ligne des deux corrigés dont le tableau cite un exercice absent de la remédiation de la question (contrôle par script).
- Correction (l. 212) : « **Remédiation** : 8.24, 8.25, 9.28 (la standardisation faite hors des folds), 8.3, 8.Q6 ; fiche du ch. 8, §8.5.1 (encadré ⚠️ « Pas de fuite, puisqu'on crée un modèle neuf à chaque tour ») ».

**7. MINEUR · `checkpoints/partie_1/05_synthese.md:75`**
- Extrait : « | **Pas $h$ trop petit** | en dessous de $10^{-6}$ environ, la dérivée numérique devient plus fausse, pas plus juste | $h \approx 10^{-5}$ en différence centrée (5.12) | »
- Problème : le meilleur pas centré est $10^{-5}$ (formulaire l. 248, §22 « k = 5 en différence centrée », corrigé CP1.1 e l. 27 : « de l'ordre de $10^{-5}$ […] ; en dessous, l'estimation se dégrade »). Avec l'erreur $h^2 + \varepsilon/h$ : 1,2·10⁻¹⁰ à $h = 10^{-5}$, 2,2·10⁻¹⁰ à $10^{-6}$ : la dégradation commence sous $10^{-5}$, pas sous $10^{-6}$.
- Correction : « en dessous du meilleur pas (vers $10^{-5}$ en différence centrée, $10^{-8}$ en différence avant), la dérivée numérique devient plus fausse, pas plus juste ».

**8. MINEUR · `checkpoints/partie_1/05_synthese.md:50`**
- Extrait : « $P(H \mid D) = \frac{P(D \mid H)\, P(H)}{P(D)}$, $P(D) = \sum_k P(D \mid H_k)\, P(H_k)$ »
- Problème : la fiche du ch. 4 (8 fois $P(H \mid O)$) et le formulaire (l. 218-223 : « $O$ : une observation », $\sum_j$) notent l'observation $O$ et l'indice $j$.
- Correction : « $P(H \mid O) = \frac{P(O \mid H)\, P(H)}{P(O)}$, $P(O) = \sum_j P(O \mid H_j)\, P(H_j)$ ».

**9. MINEUR · `checkpoints/partie_2/05_synthese.md:56-57`**
- Extraits : « | 15 | Moindres carrés | $a = \frac{S_{xy}}{S_{xx}}$, $b = \bar{y} - a\,\bar{x}$ ; en général $\mathbf{X}^\top\mathbf{X}\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}$ | » ; « | 16 | Ridge | […] ; en 1D $w^* = \frac{S_{xy}}{S_{xx} + \lambda}$ ($\lambda$ : l'`alpha` du code) | »
- Problème : $S_{xy}$ et $S_{xx}$ n'apparaissent nulle part dans le ch. 9 ni au formulaire (seulement dans le sujet de CP2.6, lu après la synthèse) ; la formule 1D est celle **avec** ordonnée non pénalisée (le résultat de CP2.6), alors que le ch. 9 enseigne (9.3, formulaire l. 341) « en dimension 1, sans ordonnée : $w^* = \frac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda}$ » ; l'équation normale $\mathbf{X}^\top\mathbf{X}\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}$ oublie l'ordonnée (formulaire l. 340 : données centrées, puis $b = \bar{y} - \bar{\mathbf{x}} \cdot \hat{\mathbf{w}}$).
- Correction : l. 56 « $a = \frac{S_{xy}}{S_{xx}}$ avec $S_{xx} = \sum_i (x_i - \bar{x})^2$ et $S_{xy} = \sum_i (x_i - \bar{x})(y_i - \bar{y})$, $b = \bar{y} - a\,\bar{x}$ ; en général $\mathbf{X}_c^\top\mathbf{X}_c\,\mathbf{w} = \mathbf{X}_c^\top\mathbf{y}_c$, puis $b = \bar{y} - \bar{\mathbf{x}} \cdot \mathbf{w}$ » ; l. 57 « en 1D, sans ordonnée (9.3) $w^* = \frac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda}$ ; avec une ordonnée non pénalisée, $w^* = \frac{S_{xy}}{S_{xx} + \lambda}$ », et cette seconde forme au formulaire (ch. 9).

**10. MINEUR · `annexes/formulaire.md:332` et `:345` contre `chapitres/ch09_overfitting/01_fiche.md` (8 occurrences) et `checkpoints/partie_2/05_synthese.md:59`**
- Extraits : formulaire « $\hat{f}_m$ : le modèle entraîné sur le $m$-ième jeu » ; fiche du ch. 9 et synthèse « $\hat{f}_D$ ».
- Problème : la synthèse ne peut pas être « identique au formulaire et à la fiche » : les deux diffèrent (la synthèse suit la fiche).
- Correction : aligner le formulaire sur la fiche (« $\hat{f}_D$ : le modèle entraîné sur le dataset $D$ ; $\bar{f}$ : la moyenne sur $M$ datasets »), ou l'inverse.

**11. MINEUR · `checkpoints/partie_1/03_examen_corrige.md:223, 228, 232`**
- Extraits : « Un facteur de confusion peut créer le lien » ; « Une **expérience randomisée** (un test A/B) […] Le tirage au sort équilibre les facteurs de confusion » ; « un grand échantillon d'observation garde ses facteurs de confusion ».
- Problème : le ch. 2 (fiche l. 323) et le glossaire (l. 228) ont retenu « **variable de confusion** (*confounder*) » et parlent d'« expérience contrôlée (tirer au sort qui reçoit un traitement) » (§5 : on s'en tient au terme retenu).
- Correction : « variable(s) de confusion » aux trois endroits ; l. 228 « Une **expérience contrôlée**, randomisée (un test A/B) : ».

**12. MINEUR · `annexes/glossaire.md` (aucune occurrence)** — termes introduits par les checkpoints et les mini-projets, absents du glossaire (§5 : « Tout terme ajouté va dans `annexes/glossaire.md` ») :
- « Naive Bayes » (`projets/partie_1_detecteur_langue/README.md:9`, la méthode imposée de MP1 ; défini dans un encadré du ch. 4) ; « règle d'une erreur type » (`projets/partie_2_california_validation/README.md:33`, le cœur de MP2) ; « encodage par la cible (*target encoding*) » (`checkpoints/partie_2/03_examen_corrige.md:209`) ; « règle de trois » (`projets/partie_1_detecteur_langue/solution/README.md:42`) ; « test A/B » (`checkpoints/partie_1/03_examen_corrige.md:228`) ; « *successive halving* » (`projets/partie_2_california_validation/README.md:86`).
- Correction : six lignes, colonne « Ch. » = CP1/MP1 ou CP2/MP2. Par exemple : « règle d'une erreur type | one-standard-error rule | parmi des modèles rangés du plus simple au plus complexe, garder le premier dont le score moyen ne dépasse pas le meilleur de plus d'une erreur type (en validation croisée, $\sigma/\sqrt{k}$ : une approximation optimiste) | MP2 » ; « règle de trois | rule of three | 0 erreur sur $n$ essais indépendants : le taux d'erreur est inférieur à $3/n$ avec 95 % de confiance | MP1 ».

**13. MINEUR · `annexes/formulaire.md:149-386`** — aucune formule propre aux checkpoints et mini-projets (§6 : le formulaire « recense toutes les formules du workbook »).
- Manquent : la température de MP1 (posterior ∝ prior × $e^{\ell/T}$, docstring de `posterior_from_loglik`), la règle de trois ($3/n$), la règle d'une erreur type (choix : moyenne ≤ meilleure moyenne + $\sigma_{\text{meilleur}}/\sqrt{k}$), Ridge 1D avec ordonnée ($S_{xy}/(S_{xx} + \lambda)$, CP2.6).
- Correction : une section « Checkpoints et mini-projets » (ou une ligne au ch. 3, au ch. 2, au ch. 8 et au ch. 9, marquée MP1, MP2 ou CP2).

**14. MINEUR · data cards : `data/cards/penguins.md:38`, `california_housing.md:38`, `holmes.md:17`, `verne.md:17`** — les usages des checkpoints et mini-projets n'y figurent pas (§9 : « chapitres qui l'utilisent » ; cela confirme le résultat brut « data cards : 1 » de la section 5 de l'audit).
- Correction : penguins « … ; CP1 (CP1.3 standardisation colonne par colonne, CP1.5 bootstrap de la masse des 119 Gentoo), CP2 (CP2.5 plan d'évaluation sur les 333 manchots) » ; california « … ; CP2 (CP2.10 : 2 000 districts, une fuite par la moyenne de la cible par case), MP2 (tous les districts, test gelé de 4 128 districts, graine 2026, coffre ouvert une fois) » ; holmes et verne « … ; MP1 (détecteur de langue : nouvelles 1-8 / 9-10 / 11-12 et chapitres 1-25 / 26-31 / 32-37 pour l'entraînement, la validation et le test) ».

**15. MINEUR · `projets/partie_1_detecteur_langue/depart/langid.py:55, 107, 122, 133, 137, 163, 177` et `projets/partie_2_california_validation/depart/housing.py:65, 70, 152, 164, 168, 172, 187, 199, 213, 231` (générés par `build_mp1.py`/`build_mp2.py`), et cellules TODO des deux kits**
- Extrait : `raise NotImplementedError  # TODO MP1.3` (sans message).
- Problème : `wb.attempt` n'affiche que le message de l'exception (`src/wb/checker.py:993-995`) : un apprenant qui a écrit trois des cinq fonctions de MP1.3 lit « ⏳ Ex MP1.3 : pas encore fait. » sans savoir laquelle manque (essayé) ; même chose pour MP2.2 (cinq fonctions). La règle de PROGRESS (« chaque stub nomme la sienne ») est appliquée aux examens (« ⏳ Ex CP2.10 : pas encore fait (cv_rmse_fixed). ») mais pas aux kits.
- Correction : générer `raise NotImplementedError("LanguageDetector.fit (MP1.3)")`, `"posterior_from_loglik (MP1.3)"`, `"HousingModel.transform (MP2.2)"`… dans les modules et dans les cellules TODO (`make_set`, `letter_distribution`, `cross_entropy_table`, `mean_baseline`, `pick_simplest`, `residuals_by_group`).

**16. MINEUR · `projets/partie_1_detecteur_langue/depart/conftest.py` et `projets/partie_2_california_validation/depart/conftest.py` ; « Démarrer » des cahiers des charges (`partie_1…/README.md:79-81`, `partie_2…/README.md:97-100`)**
- Constat (copie, kit vierge) : `python -m pytest mon_travail/projets/partie_1_detecteur_langue -q` affiche « 2 errors » et une trentaine de lignes de traceback (`NotImplementedError` dans la fixture `model`) ; MP2 « 2 failed », idem.
- Problème : les tests des chapitres affichent « ⏳ pas encore implémenté » (`tests/conftest.py:204-212`, décision du §22 de 2026-09-29) ; ceux des projets, une erreur brute que rien n'annonce à un débutant qui n'a encore rien écrit.
- Correction : reprendre le hook `pytest_runtest_makereport` de `tests/conftest.py` dans les deux `conftest.py` des kits (en traitant aussi la phase `setup`, pour la fixture), ou au moins ajouter dans « Démarrer » : « Tant que `langid.py` n'est pas écrit, les deux tests d'exemple échouent avec `NotImplementedError` : c'est normal. »

**17. MINEUR · en-tête du notebook de MP1 (`tools/chapters/build_mp1.py`, cellule 0) contre MP2 (`build_mp2.py:1143`)**
- Extrait MP2 : « **Mode rapide.** Avec `FAST_MODE = True` (la cellule de setup), seules les courbes d'apprentissage et la mesure de la variance (MP2.5) passent de 5 à 3 folds […] ».
- Problème : MP1 ne dit pas ce que change son mode rapide, alors qu'il change la taille des jeux (`N_VAL = 200 if FAST_MODE else 1000`, `N_TEST = 500 if FAST_MODE else 2000`) et que le README rédigé donne ses résultats avec les tailles du mode rapide (`solution/README.md:23` : « 500 par langue et par longueur ») et « en mode rapide » (l. 77).
- Correction (en-tête de MP1) : « **Mode rapide.** Avec `FAST_MODE = True` (la cellule de setup), 200 extraits de validation et 500 de test par langue et par longueur (1 000 et 2 000 en mode complet) ; le notebook tourne en une demi-minute environ sur un CPU. »

**18. MINEUR · durées annoncées : `projets/partie_2_california_validation/README.md:3`, `projets/partie_1_detecteur_langue/README.md:3`, `checkpoints/partie_1/README.md:3` et `partie_2/README.md:3`**
- Extraits : MP2 « Compte environ **10 heures**, en sept étapes » (les étapes, l. 57-63, font 60 + 60 + 120 + 90 + 90 + 45 + 75 = 540 min, 9 h) ; MP1 « environ **8 heures** » (450 min, 7 h 30) ; README des checkpoints « Compte environ **11 h 30** » / « **13 h 30** » contre `docs/SYLLABUS.md:1354, 2044` et `suivi/tableau_de_bord.md:391, 597` « 11 h » / « 13 h ».
- Correction, sans toucher aux durées du contrat : « Compte environ **10 heures** : sept étapes (9 h), plus la lecture du cahier des charges et, à la fin, celle de la solution » (MP1 : « 8 heures : sept étapes (7 h 30), plus … ») ; un même arrondi partout (`fmt_hours` de `syllabus.py` : « 11 h 30 »).

**19. MINEUR · `tools/chapters/build_mp2.py:1136` (cellule « Bilan » des deux notebooks de MP2) et `docs/syllabus/data/cp2.json:397`**
- Extraits : « un réglage de α par *successive halving* avec ta librairie `bandit` » ; « Choisir α par une recherche « bandit » (successive halving) avec bandit.py ».
- Problème : le *successive halving* n'utilise pas `bandit.py` ; le cahier des charges (l. 86) sépare bien les deux : « `ucb_action` de ta librairie `bandit` choisit le bras suivant. Ou bien le *successive halving* […] ».
- Correction : « un réglage de α comme un bandit (`ucb_action` de ta librairie `bandit`) ou par *successive halving* » (notebook et `cp2.json`).

**20. MINEUR · `checkpoints/partie_1/01_examen_sujet.md:39` et `:156`**
- Extraits : « ### CP1.1 — Questions flash 🧠 ★ ⏱️ 10 min · 1,5 point » ; « ### CP1.12 — Un LLM expliqué en cinq lignes 🗣️ ★ ⏱️ 5 min · 1 point ».
- Problème : seuls titres du sujet qui diffèrent du contrat (« Questions flash sur toute la partie (vrai ou faux, justifié) », « Un LLM expliqué en cinq lignes : données, loss, perplexité ») et du tableau du même sujet (l. 21, 32) ; le sujet de CP2 reprend partout les titres complets. Probablement les « 2 » écarts du tableau brut A3 pour CP1.
- Correction : reprendre les deux titres complets.

### SUGGESTIONS

**21. SUGGESTION · `checkpoints/partie_2/05_synthese.md:72-73`** — la synthèse, révisée juste avant l'examen, amorce deux réponses : « une standardisation, une sélection de features ou une moyenne par groupe calculées sur toutes les données » (la « moyenne par groupe » ne figure pas dans la liste des fuites du ch. 8, fiche l. 123-126, ni en 8.24 : c'est exactement la fuite de CP2.10) et « « le score de test déçoit, je change le degré et je recommence » » (≈ CP2.5 f, protocole 5 : « Le score de test déçoit : changer de degré, réévaluer sur le test, et recommencer »), alors que PROGRESS (l. 388) annonce « aucune valeur ni exemple de l'examen ». Proposition : « une standardisation, une imputation ou une sélection de features calculées sur toutes les données » et « « le test donne 0,81 ; j'ajoute une feature et je reteste » ».

**22. SUGGESTION · `checkpoints/partie_*/04_mes_reponses.md` (tableau « Après l'examen : ma note », l. 125-141 et 135-150)** — CP2 est plus dense que CP1 : 75 réponses notées en 117 min (1,56 min chacune) contre 65 (1,80 min) ; CP2.2 demande 7 réponses en 5 min, CP2.1 huit vrai/faux justifiés (dont un calcul et une récurrence) en 10 min, CP2.10 la lecture de sept fonctions, deux fuites et une fonction à écrire en 9 min. Ajouter une colonne « ⏱️ réel » au tableau, et au README (étape 5) « note aussi le temps passé sur chaque question » : ces temps alimenteront le Calibrage avant CP3.

**23. SUGGESTION · BIBLE §17 et §22 (2026-10-01 « Types de l'examen blanc I » ; 2026-10-03 « Types de l'examen blanc II »)** — deux règles différentes (CP1 : « tous les types qui ont un sens pour la partie et pour un examen de deux heures », 10 types dont 🔮 🗣️ ⚖️ ; CP2 : « les 7 types du contrat », sans 🔮 ni 🗣️ ni ⚖️, alors que la partie II en a). Décidé pour ces deux checkpoints ; pour CP3 à CP6, écrire une seule règle au §17. De même pour les étoiles : CP2.10 (🐛 : deux fuites et une fonction) et CP2.1 (dont une preuve) sont notés ★, alors que la règle de `cp1.json:438` (« ★★ = question qui combine deux notions ou demande une interprétation ») les classerait ★★.

**24. SUGGESTION · `checkpoints/partie_1/README.md:27` et `partie_2/README.md:27`** — « Reporte ta note et tes points faibles dans `mon_travail/suivi/journal.md` » ne renvoie pas à la section CP du modèle `suivi/auto_evaluation.md` (l. 144-156 et 227-240), faite pour cela (note de l'examen, du mini-projet, reprise une semaine plus tard). Proposition : « Reporte tes notes (examen et mini-projet) et tes points faibles dans la section CP1 de `mon_travail/suivi/auto_evaluation.md`, une ligne dans `journal.md`, puis coche le checkpoint dans ton tableau de bord. »

**25. SUGGESTION · `projets/partie_2_california_validation/depart/README.md`** — le modèle de README n'a pas de section « ## Pistes », alors que celui de MP1 (l. 31) et le README rédigé de MP2 (`solution/README.md:67`) en ont une ; l'ajouter (« TODO : ce que je ferais ensuite (validation croisée spatiale, distances à la côte, cible plafonnée, gradient boosting…) »).

**26. SUGGESTION · `tools/chapters/build_mp1.py:606` (observations MP1.5 de la solution)** — « on évalue les deux versions sur les mêmes extraits et l'on bootstrappe la différence (un rééchantillonnage apparié) » : notion du ch. 8 (formulaire, « bootstrap apparié »), pas encore vue à la fin de la partie I. Ajouter « (tu le pratiqueras au ch. 8) ».

**27. SUGGESTION · fonction `ready` des notebooks des deux kits** — « ⏳ il manque train_texts, train_labels : fais d'abord les étapes précédentes. » (sortie du kit vierge) nomme des variables que l'apprenant n'écrit pas (la vérification de MP1.1 les crée). Associer chaque variable à son étape : « ⏳ il manque train_texts, train_labels (créés par la vérification de MP1.1) ».

**28. SUGGESTION · `checkpoints/partie_2/05_synthese.md:19` (carte Mermaid)** — « densité n / b^d » s'affiche littéralement « b^d » (rendu mermaid-cli) ; écrire « n / bᵈ ».

**29. SUGGESTION · `tools/start_chapter.py` (sortie de `CP1`/`CP2`)** — la sortie finit par « 👉 Teste ta librairie : python -m pytest tests/ -q » mais ne donne pas la commande des tests du mini-projet ; ajouter « 👉 Tests du mini-projet : python -m pytest mon_travail/projets/partie_1_detecteur_langue -q ».

## Bilan

**Ce qui est solide.** Les deux examens sont justes : chaque valeur des corrigés a été recalculée sans écart, chaque question admet une seule réponse, les barèmes totalisent 20 question par question et les répartitions par chapitre de `cp1.json`/`cp2.json` sont exactes ; tous les renvois de remédiation existent et sont pertinents, les sections de fiche citées aussi. Les deux checkpoints ont la même architecture (README et déroulé mot pour mot parallèles, sujet, notebook A/B avec `EXAM_OVER`, corrigé avec règles de notation, copie avec tableau de note, synthèse en cinq sections, « Lire ta note »). Les synthèses citent des fonctions `mylearn` qui existent toutes, et les deux cartes Mermaid se rendent. Les mini-projets fonctionnent de bout en bout depuis `mon_travail/` (kit vierge : uniquement ⏳ ; kit rempli : garde-fous verts, commit et coffre, `results.json` identique à la solution) ; les README de portfolio sont honnêtes et concordent chiffre par chiffre avec `results.json` et les sorties ; les scripts `build_*` reconstruisent exactement les notebooks publiés.

**Les trois risques principaux.**
1. **Des réponses lisibles avant l'heure** : le SYLLABUS publie les choix et les scores de référence de MP2 et des indices sur CP2.10, au milieu d'un historique de génération contradictoire (constat 2) ; la synthèse de CP2 amorce deux réponses de l'examen (21).
2. **L'expérience d'un débutant dans les kits** : fonctions à écrire qui ne se nomment pas dans les ⏳ (15), tracebacks bruts des tests au départ (16), mode rapide non documenté en MP1 (17), nommage du notebook (1) ; et un CP2 plus dense que CP1 dont le temps n'a jamais été mesuré sur un humain (22).
3. **Les dérives de notation et de vocabulaire** entre synthèses, corrigés, chapitres et annexes ($S_{xy}$ non défini et Ridge 1D avec ordonnée, $P(H \mid D)$, $\hat{f}_m$/$\hat{f}_D$, seuil du pas $h$, « facteur de confusion ») et les annexes qui ignorent les checkpoints (glossaire, formulaire, data cards : 9 à 14).
