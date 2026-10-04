# Audit P5, tour 2 : exactitude des chapitres 0A et 0B (relecteur exact0)

**Périmètre lu en entier** : `chapitres/ch00a_python/` et `chapitres/ch00b_maths/` (fiche, exercices, indices, solutions, flashcards, Markdown des deux notebooks, sorties de `05_solutions.ipynb`), les 13 figures de 0B (0A n'en a pas), les sections 0A et 0B de `annexes/formulaire.md`, les entrées 0A et 0B de `annexes/glossaire.md`, `annexes/erreurs_frequentes.md` et les cheatsheets git, NumPy et pandas.

**Méthode** : copie `tar` du dépôt dans le dossier de travail. Python 3.13.13, numpy 2.1.3, pandas 2.2.3, torch 2.11.0+cpu, git 2.43, `OMP_NUM_THREADS=1`.
- Blocs `>>>` rejoués avec `doctest`. 0A : 66 blocs, 322 exemples, 4 écarts, tous de présentation (une docstring citée dans un bloc, l'espace final d'un message NumPy, la largeur d'affichage de pandas en console, le nom de fichier dans une sortie de doctest). 0B : 21 blocs, 71 exemples, aucun écart.
- Exercices ✏️, ∂ et 🧮 recalculés en Python, et sorties des notebooks de solutions comparées aux solutions écrites.
- Historique de `src/wb/answers.json` suivi commit par commit.
- 14 affirmations datées vérifiées sur le web (tableau après les constats).
- 23 liens externes testés : tous répondent 200.

Les constats sont classés par sévérité, les majeurs d'abord. Je ne reprends pas les sections 7 et 8 de `docs/AUDIT_parties_I_II.md` (P1 à P5, constat 13 des ch. 10 et 11), ni les points « à valider sur Colab ».

## Constats

### MAJEUR

**1. MAJEUR · `src/wb/answers.json` (clé `"0B.40e"`), produit par `tools/chapters/build_ch00b.py:901` · une bonne réponse est refusée en 0B.40 e**
- Extraits : `wb.record("0B.40e", bool(speedup >= 10))` ; sortie de `chapitres/ch00b_maths/05_solutions.ipynb`, cellule d'index 93 (exécution [49]) : « `-7.0 [4.61 7.5  4.  ] -0.277 0.0 NumPy is 1 times faster` » ; `05_solutions.md:747` : « e) **True** (NumPy est ici environ 300 fois plus rapide) ».
- Problème : la réponse attendue est enregistrée à partir d'une **mesure de temps**. Le commit `e96506d` (2026-10-04, notebooks réexécutés sur la machine partagée chargée) a mesuré « 1 fois plus rapide », donc enregistré `False`.
  - L'empreinte est passée de `2e31a45b…` (commits `e28bca6`, `a3e40b0`, `384b50c`) à `eb59ba4f…` (`e96506d`, `HEAD`). C'est la seule empreinte que ce commit a changée.
  - La vérification du notebook d'exercices (`03_notebook.ipynb`, cellule d'index 85) est `wb.check("0B.40e", bool(speedup >= 10))`. Un apprenant qui a un code juste, sur une machine normale, reçoit donc ❌, et une machine lente reçoit ✅.
  - Preuve avec les réponses de `HEAD` : `wb.check("0B.40e", True)` donne « ❌ Ex 0B.40e : Ce n'est pas la bonne réponse : relis l'énoncé et justifie ton choix. », et `wb.check("0B.40e", False)` donne « ✅ Ex 0B.40e : Exact ! ». Avec le `answers.json` de `384b50c`, c'est l'inverse.
  - Vitesse mesurée ici avec la référence et une fonction `measure` identique à celle du notebook (meilleur de 3 essais) : 145, puis 183 à la relance (charge ≈ 1,2).
  - Ce n'est pas le point 8 de « À valider sur Colab » (qui demande si Colab atteint le seuil de 10). La section 5 de l'audit avait noté un ❌ intermittent **avant** `e96506d` ; depuis ce commit, l'échec est systématique.
- Correction : enregistrer la réponse que l'énoncé attend, et non la mesure : `wb.record("0B.40e", True)`. Faire échouer la construction si la mesure est aberrante (par exemple `assert speedup >= 10, "machine trop chargée : relance"` dans le script, hors du notebook publié). Ensuite, réexécuter `05_solutions.ipynb` sur une machine au repos, lancer `python tools/build_answers.py` et vérifier que `wb.check("0B.40e", True)` donne ✅.

### MINEUR

**2. MINEUR · `chapitres/ch00b_maths/05_solutions.ipynb`, cellules d'index 93, 124 et 170 (scripts : `build_ch00b.py:840-905`, `1216-1270`, `1814-1880`) · les sorties de vitesse publiées contredisent les solutions**
- Extraits :
  - Cellule 93 : « NumPy is 1 times faster ». Avant `e96506d` : 285.
  - Cellule 124 : « solve: gap 1.0e-11, 7.9 ms » puis « inv:   gap 1.8e-11, 196.2 ms ». Avant : 3,8 ms et 12,4 ms.
  - Cellule 170 : « left to right: 108.0 ms, right to left: 22.23 ms, speedup: 5 ». Avant : 33,5 ms, 0,66 ms, 51.
  - En face, `05_solutions.md:828` dit « c) un gain de l'ordre de 50 sur la machine de test (30 à 100 selon les machines) : objectif atteint », et la note de 0B.46 (`build_ch00b.py:1266`) parle d'« environ trois fois moins de calculs ».
- Problème : même cause que le constat 1. Le corrigé publié montre le défi 🏆 0B.54 **raté** (5 contre un objectif de 20), un `inv` 25 fois plus lent que `solve`, et NumPy pas plus rapide qu'une boucle Python. Mesures refaites ici : 0B.54, 75,2 ms contre 1,39 ms (vitesse 54) ; 0B.46, `solve` 4,1 ms et `inv` 21,4 ms. Les sorties de temps des autres notebooks réexécutés par `e96506d` (ch. 3, 5, 6, 9, 10, 11) ne sont pas dégradées.
- Correction : la réexécution du constat 1 suffit. En option, la cellule de solution de 0B.54 peut afficher un avertissement si `speedup < 20` (« mesure faussée par une machine chargée : relance »), pour que `run_all_notebooks.py` le rende visible.

**3. MINEUR · `chapitres/ch00a_python/01_fiche.md:161` · fait dépassé sur Colab**
- Extrait : « Sur Colab, il n'y a pas de terminal dans la version gratuite : on lance les commandes depuis une cellule avec le module `subprocess` (le workbook n'utilise jamais les raccourcis `!` et `%`, qui ne fonctionnent pas partout). »
- Problème : c'est faux au 2026-10-04. Les notes de version de Colab disent, au 2025-07-22 : « Terminal is now available to all users, free of charge! » ([Colab release notes](https://developers.google.cn/colab/release-notes) ; voir aussi le billet « Colab Terminal Is Now Free For All Users », 2025-06-06, [medium.com/@spkshumway](https://medium.com/@spkshumway)). Les exercices n'en dépendent pas, mais un débutant croit qu'il n'a pas de terminal pour git.
- Correction : « Colab propose aussi un terminal, gratuit pour tous depuis 2025, mais le workbook lance ses commandes depuis une cellule, avec le module `subprocess`, pour qu'elles restent dans le notebook (il n'utilise jamais les raccourcis `!` et `%`, qui ne fonctionnent pas partout). »

**4. MINEUR · `chapitres/ch00a_python/01_fiche.md:1814-1815`, `annexes/cheatsheets/git.md:25` et `:42` · `git diff` et `git restore` décrits de façon inexacte**
- Extraits :
  - Fiche : « `git diff` | les lignes modifiées depuis le dernier commit » et « `git restore fichier` | annule les modifications non commitées d'un fichier ».
  - Cheatsheet : « `git diff` | les lignes modifiées depuis le dernier commit, pas encore préparées » et « `git restore fichier` | annule les modifications non commitées d'un fichier (⚠️ définitif) ».
- Problème : `git diff` compare à l'index, pas au dernier commit. `git restore fichier` remet le fichier dans l'état de l'index : une modification déjà passée par `git add` **n'est pas** annulée. La fiche ne dit pas non plus que l'opération est irréversible.
  - Démonstration (dépôt jetable, git 2.43) : on commite `v1` ; on écrit `v2` puis on fait `git add`. `git diff` est alors vide, et `git diff HEAD` montre le changement. Après `git restore f.txt`, le fichier contient toujours `v2`, avec le statut `M `. Il faut `git restore --staged --worktree f.txt` pour revenir à `v1`.
- Correction :
  - Fiche, ligne 1814 : « `git diff` | les lignes modifiées **pas encore ajoutées** avec `git add` (`git diff --staged` : celles qui sont ajoutées ; `git diff HEAD` : tout depuis le dernier commit) ».
  - Fiche, ligne 1815 : « `git restore fichier` | ⚠️ efface **définitivement** les modifications pas encore ajoutées d'un fichier (après un `git add` : `git restore --staged --worktree fichier`) ».
  - Cheatsheet, ligne 25 : « les lignes modifiées pas encore préparées (`git diff --staged` : les lignes préparées) ».
  - Cheatsheet, ligne 42 : « annule les modifications non préparées d'un fichier (⚠️ définitif) ; `git restore --staged fichier` retire un fichier de la préparation ».

**5. MINEUR · `annexes/cheatsheets/pandas.md:65`, et aussi `chapitres/ch02_stats/01_fiche.md:329` et `annexes/formulaire.md:187` · `df.cov()` et `df.corr()` décrits comme sous pandas 1.x**
- Extraits :
  - Cheatsheet : « `df.cov()`, `df.corr()` | matrices de covariance (`ddof=1`) et de corrélation de Pearson des colonnes numériques ».
  - Fiche du ch. 2 : « En pratique : `df.cov()` et `df.corr()` en pandas ».
  - Formulaire : « `np.cov(X, rowvar=False)`, `df.corr()` ».
- Problème : depuis pandas 2.0, `numeric_only` vaut `False` par défaut. Avec pandas 2.2.3, `penguins.corr()` et `penguins.cov()` lèvent `ValueError: could not convert string to float: 'Adelie'`, alors que `penguins.corr(numeric_only=True).shape` vaut `(5, 5)`. Source : [doc `DataFrame.corr` (2.2)](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.DataFrame.corr.html), « Changed in version 2.0.0: The default value of `numeric_only` is now `False` ».
- Correction : écrire `df.cov(numeric_only=True)` et `df.corr(numeric_only=True)` (ou `df[cols].corr()`) aux trois endroits. Dans la cheatsheet, ajouter : « sans `numeric_only=True`, une colonne de texte lève une `ValueError` ».

**6. MINEUR · `annexes/formulaire.md:98`, `chapitres/ch00b_maths/04_indices.md:388` et `:393`, `chapitres/ch00b_maths/05_solutions.md:222` · notation de la droite qui n'est pas celle du cours**
- Extraits :
  - Formulaire : « droite | $y = m x + p$ ; pente $m = \frac{y_B - y_A}{x_B - x_A}$ ».
  - Indice 2 de 0B.6 : « b : écris $y = mx + p$ et remplace par le point $(-1, 5)$. c : résous $mx + p = 0$. d : ici $a = 2$, $b = -8$, $c = 6$. »
  - Solution de 0B.6 : « $5 = -2 \times (-1) + p$, donc $p = 3$ ».
  - Fiche, au contraire (101.2.1, ligne 264) : « Une fonction **affine** s'écrit $f(x) = a\,x + b$ ». Même écriture au ch. 9 (`01_fiche.md:135`).
- Problème : $m$ et $p$ ne sont introduits nulle part dans le cours. Le débutant qui ouvre l'indice découvre une notation nouvelle, et $p$ entre en conflit avec la lettre retenue pour le nombre de features (P4).
- Correction :
  - Formulaire : « droite (fonction affine) | $f(x) = a\,x + b$ ; pente $a = \frac{y_2 - y_1}{x_2 - x_1}$ ; ordonnée à l'origine $b = f(0)$ ».
  - Indice 2 : « b : écris $y = a\,x + b$ (101.2.1) avec la pente trouvée en a, puis remplace par le point $(-1, 5)$. c : résous $a\,x + b = 0$. d : pour la parabole, les lettres changent de rôle : ici $a = 2$, $b = -8$, $c = 6$. »
  - Indice 3 et solution : remplacer $p$ par $b$.

**7. MINEUR · `chapitres/ch00b_maths/01_fiche.md:322` · mauvais renvoi pour la log-loss**
- Extrait : « C'est pourquoi on travaille en **log-probabilités** (0B.E3) et que la loss de classification est une « log-loss » (ch. 13). »
- Problème : la log loss est enseignée au ch. 6 (`ch06.json`, notions : « log loss » ; 🔨 6.22, `info.log_loss`). Le ch. 13 la réutilise pour la régression logistique (« La log-loss binaire est réutilisée depuis info.log_loss (ch. 6) »).
- Correction : « … une « log-loss » (ch. 6, puis la régression logistique au ch. 13). »

**8. MINEUR · `chapitres/ch00b_maths/05_solutions.md:631` (🧮 0B.30, question 4) · « 700 fois plus rapide », démenti par 0B.54 d dans le même chapitre**
- Extrait : « mais le calcul de droite à gauche est environ $\frac{2 \times 10^9}{3 \times 10^6} \approx 700$ fois plus rapide. »
- Problème : $2{,}001 \times 10^9 / 3 \times 10^6 \approx 667$ est un rapport de **nombres de multiplications**. Or `05_solutions.md:830` (0B.54 d) enseigne que « le gain en temps (≈ 50) est plus petit que le gain en multiplications (≈ 670) ». L'apprenant lit donc deux réponses contradictoires à « combien de fois plus rapide ».
- Correction, dans la solution seulement (le contrat ne change pas) : « … demande environ $\frac{2 \times 10^9}{3 \times 10^6} \approx 670$ fois moins de multiplications. En temps, le gain est plus petit, de l'ordre de 50, car un produit matrice-vecteur est limité par la mémoire (tu le mesureras en 0B.54). » En option, au contrat : ajouter « (en nombre de multiplications) » à la question 4 de l'énoncé.

**9. MINEUR · `chapitres/ch00a_python/05_solutions.md:410` et `tools/chapters/build_ch00a.py:1177` (note de 0A.36 dans `05_solutions.ipynb`) · un chiffre contredit ses propres données**
- Extrait : « d'une année à l'autre, la masse moyenne de chaque espèce varie de 120 g au plus (Gentoo : 5071 g en 2007, 5020 g en 2008, 5141 g en 2009), soit moins de 3 % ».
- Problème : $5140{,}70 - 5019{,}57 = 121{,}13$ g. Calcul : `p.groupby(["year", "species"])["body_mass_g"].mean().unstack()` ; écarts maximaux : Gentoo 121,13 g (2,41 %), Chinstrap 105,77 g (2,86 %), Adelie 77,10 g (2,10 %).
- Correction : « varie de 121 g au plus (…), soit moins de 3 % », dans la solution et dans la note du script.

**10. MINEUR · exemples de format qui sont des réponses, en 0A : `tools/chapters/build_ch00a.py:216-217`, `:139` et `:3173` ; `chapitres/ch00a_python/02_exercices.md:328`**
- Extraits :
  - Partie 0 des deux notebooks : « Reporte ensuite chaque réponse ici : **la valeur** que tu as trouvée (`3`, `"float"`, `[1, 2]`, `(4, 6)`…) ».
  - Énoncé de 0A.8 et commentaire du notebook : « réponds par une **chaîne** : `"(5, 3)"` ou `"erreur"` » ; `# a) (5, 3) + (3,), a string such as "(5, 3)" or "erreur"`.
  - 0A.65 d : « le type des éléments de `one_hot([1, 0], n_classes=3, dtype=int)` (son nom, par exemple `"int64"`) ».
- Problème :
  - `3` est la réponse de 0A.1 a (et de 0A.4 c), et `"float"` celle de 0A.1 g. Le commentaire de 0A.1 g, lui, prend exprès la mauvaise réponse comme exemple (« a string such as "int" »).
  - `"(5, 3)"` est la réponse de 0A.8 a.
  - `"int64"` est la réponse de 0A.65 d (`05_solutions.md:582` : « d) **int64** »).
- Correction :
  - Partie 0 : « (`7`, `"int"`, `[1, 2]`, `(4, 6)`…) ». Aucune réponse de 0A.1 à 0A.8 ne vaut l'une de ces valeurs.
  - 0A.8 : `"(7, 2)"` ou `"erreur"`.
  - 0A.65 d : « par exemple `"float32"` ».

**11. MINEUR · indices de niveau 3 qui donnent la réponse finale en 0A et 0B (même défaut que le constat 13 du tour 1, ici hors des ch. 10 et 11)**
- Occurrences dans `chapitres/ch00a_python/04_indices.md` :
  - ligne 25 (0A.Q1) : « Deux réponses sont « vrai » : celle sur *Restart and run all* et celle sur `cd ..` » ;
  - ligne 97 (0A.Q5, item 1) : « 1 : `1 4 7` » ;
  - ligne 115 (0A.Q6, item 2) : « `a=1`, `b=5`, et il reste `6, 7` » ;
  - ligne 169 (0A.Q9, item 2) : « `__len__`, `__call__`, `__rmul__` » ;
  - ligne 205 (0A.Q11, item 1) : « 1 : `(4,)` » ;
  - ligne 353 (0A.7) : « b : `(4,)` mais c : `(4, 1)` » ;
  - ligne 371 (0A.8) : « a : `3 = 3` → OK, résultat `(5, 3)` » ;
  - ligne 900 (🔮 0A.29) : « À la fin, `a` vaut `[0 1 100 3 4 5]` ».
- Occurrences dans `chapitres/ch00b_maths/04_indices.md` :
  - lignes 25, 43, 133 et 205 (0B.Q1, Q2, Q7, Q11) : « Trois affirmations sont vraies : la 2, la 4 et la 5 » (puis « la 3, la 4 et la 5 », « la 2 et la 5 », « la 2, la 4 et la 5 ») ;
  - ligne 245 (0B.R1) : « Tu dois trouver 55, puis 120. » ;
  - ligne 429 (0B.8 h) : « h : $\left(\frac{3}{5}, \frac{-4}{5}\right)$ » ;
  - ligne 465 (0B.10 e) : « e : $y = 5x - 4$ » ;
  - ligne 735 (0B.25) : « f : $(2{,}6 ; 0{,}6)$ … h : $x = 1$, $y = 0$ ».
- Problème : BIBLE §12 définit l'indice 3 comme « presque la solution : pseudo-code ou première ligne ». Ici, la vérification de la partie 0 (ou l'autocorrection du quiz) n'a plus d'objet.
- Correction : poser le calcul sans la valeur, comme aux ch. 4 à 9. Exemples :
  - 0B.Q1 : « 1 : on additionne les exposants. 3 : essaie $a = 9$, $b = 16$. 4 : $2^{20} = (2^{10})^2$. »
  - 0B.R1 : supprimer la phrase « Tu dois trouver… » et contrôler sur $1 + 2 + 3 = 6$.
  - 0B.25 f : « $(3, 1) - 0{,}1 \times \nabla f(3, 1)$ ».
  - 0A.7 : « b : un indice entier retire l'axe ; c : une tranche le garde ».
  - 0A.29 : « quels noms partagent la mémoire de `a` ? ».

**12. MINEUR · `chapitres/ch00a_python/01_fiche.md:109` · le message de démonstration n'est pas celui que l'outil affiche**
- Extrait : « ❌ Ex demo.1 : Tu es à 1 près : erreur de bornes (off-by-one) ? Vérifie si les bornes sont incluses. »
- Problème : la sortie réelle (`src/wb/checker.py:569`, rejouée ici) est « ❌ Ex demo.1 : Tu es à 1 près : une petite erreur de calcul, ou, si tu comptes des éléments, une erreur de bornes (off-by-one : bornes incluses ou exclues ?). » C'est la première sortie de `wb.check` que lit l'apprenant.
- Correction : recopier ce message tel quel à la ligne 109.

**13. MINEUR · `tools/chapters/build_ch00a.py:2487-2488` (note de 0A.54 dans `05_solutions.ipynb`) · justification fausse des réseaux convolutifs**
- Extrait : « 81 % des pixels sont nuls : l'information tient dans une petite partie de l'image, ce qu'exploiteront les réseaux convolutifs (ch. 21). »
- Problème : un CNN n'exploite pas la proportion de pixels nuls. Il exploite la structure **locale** (des pixels voisins qui forment un trait) et le partage des poids (le même motif cherché partout). Le débutant retient une mauvaise raison.
- Correction : « 81 % des pixels sont nuls : l'information tient dans les traits du chiffre, des groupes de pixels voisins ; c'est cette structure locale, où un même motif peut apparaître n'importe où dans l'image, qu'exploiteront les réseaux convolutifs (ch. 21). »

**14. MINEUR · `annexes/glossaire.md:87` · deux notions confondues**
- Extrait : « | pile d'appels | traceback | message qui liste les appels en cours au moment d'une erreur (lire la dernière ligne d'abord) | 0A | »
- Problème : dans la section « Retenu | Autre langue », cette ligne fait de « pile d'appels » le terme retenu pour « traceback ». Or la pile d'appels est le *call stack*, c'est-à-dire les appels en cours, et le traceback est le message qui l'affiche. La fiche 0A les distingue bien : ligne 843 (« Chaque appel en cours occupe une place dans la **pile d'appels** ») et ligne 884 (« Python … affiche un **traceback** »).
- Correction : deux entrées.
  - « | pile d'appels | call stack | les appels de fonctions en cours, chacun en attente du suivant ; profondeur limitée (`RecursionError`) | 0A | »
  - « | traceback | trace de la pile d'appels | message d'erreur qui liste les appels en cours au moment de l'erreur (lire la dernière ligne d'abord) | 0A | »

**15. MINEUR · `annexes/cheatsheets/numpy.md:239-249` · lignes rangées sous le mauvais titre**
- Extrait : sous « ## Moindres carrés, pénalités et grilles de droites (ch. 9) » (ligne 223) figurent `np.where(z > 0, 1.0, -1.0)` … (colonne Ch. : 10) et `best = np.flatnonzero(q == q.max())` … `np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), …)` (colonne Ch. : 11).
- Correction : insérer « ## Perceptron : seuil, biais, ordre aléatoire (ch. 10) » avant la ligne 239, et « ## Bandits et tirages pondérés (ch. 11) » avant la ligne 243.

**16. MINEUR · `annexes/cheatsheets/git.md:11` · commande incomplète**
- Extrait : « `git config --global pull.rebase true` et `rebase.autoStash true` »
- Problème : la seconde partie n'est pas une commande. Tapée telle quelle, elle échoue. `erreurs_frequentes.md:18`, lui, donne bien les deux commandes complètes.
- Correction : « `git config --global pull.rebase true` puis `git config --global rebase.autoStash true` ».

**17. MINEUR · `chapitres/ch00b_maths/flashcards.csv:8` · coquille dans la question**
- Extrait : « Pour quelles raisons \(q\) la suite \(q^k\) tend-elle vers 0 ? »
- Correction : « Pour quelles valeurs de \(q\) la suite \(q^k\) tend-elle vers 0 ? »

**18. MINEUR · `tools/chapters/build_ch00b.py:531` et `:566` (0B.35 f) · avertissement d'arrondi visible dans le corrigé publié**
- Extraits : `wb.check("0B.35f", np.asarray(moving_average(series, 9))[:3])` et `wb.record("0B.35f", moving_average(series, 9)[:3], decimals=3)`. Sortie de `05_solutions.ipynb`, cellule d'index 70 : « ⚠️ Ex 0B.35f : un élément est proche d'une limite d'arrondi ; envisage un autre nombre de décimales. »
- Problème : $0{,}34048053$ est à $2 \times 10^{-5}$ de la limite $0{,}3405$. La règle actuelle (CLAUDE.md) demande `computed=True` et 4 décimales pour une valeur calculée par la fonction de l'apprenant dans la cellule de vérification. 0A et 0B n'emploient jamais `computed=True` (0 occurrence, contre 14 au ch. 3 et 22 au ch. 5). Sans effet sur l'apprenant : boucle, `np.convolve`, `cumsum`, `sum/k` et `rolling` de pandas donnent tous ✅.
- Correction : `wb.check("0B.35f", …, computed=True)` et `wb.record("0B.35f", …, decimals=4)`. Le même passage vaut pour les autres valeurs calculées par la librairie de l'apprenant en 0A et 0B.

### SUGGESTION

**19. SUGGESTION · `chapitres/ch00a_python/01_fiche.md:1006` · `\d` n'est pas exactement `[0-9]`**
- Extrait : « | `\d` | un chiffre (`[0-9]`) | »
- Problème : sur des chaînes `str`, `\d` reconnaît tous les chiffres Unicode. `re.fullmatch(r"\d", "٣")` réussit, et `re.findall(r"\d+", "masse ３７５０ g")` donne `['３７５０']`.
- Correction : « un chiffre (`0` à `9` pour du texte courant ; exactement `[0-9]` avec `flags=re.ASCII`) ».

**20. SUGGESTION · ordre de grandeur du gain de NumPy, annoncé différemment selon les endroits**
- Extraits :
  - `01_fiche.md:57` (0A) : « souvent 50 à 100 fois plus rapide » ;
  - `04_indices.md:505` et `05_solutions.md:277` (0A) : « souvent 50 à 100 » ;
  - `build_ch00a.py:2582` : « de l'ordre de 100 à 300 fois » ;
  - `build_ch00b.py:903` : « souvent 50 à 500 » ;
  - `05_solutions.md:747` (0B) : « environ 300 fois ».
- Problème : le chapitre mesure lui-même 123 à 191 fois en 0A.55.
- Correction : harmoniser en « de quelques dizaines à quelques centaines de fois selon l'opération (souvent autour de 100) ».

**21. SUGGESTION · `chapitres/ch00a_python/01_fiche.md:1999` · la doc de pandas 3 n'est pas si proche de la 2.2**
- Extrait : « la documentation en ligne décrit pandas 3, proche de la version 2.2 du workbook ».
- Problème : pandas 3.0.0 (2026-01-21) infère le type `str` pour le texte, là où la 2.2 affiche `object`, et active Copy-on-Write par défaut ([whatsnew 3.0.0](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html)). Le débutant verra `dtype: str` dans le tutoriel.
- Correction : « … décrit pandas 3 ; différences visibles : le texte y est de type `str` (et non `object`), et Copy-on-Write est actif (100.9) ; doc de la 2.2 : https://pandas.pydata.org/pandas-docs/version/2.2/ ».

**22. SUGGESTION · `tools/chapters/figures_ch00b.py:352` (`figures/grands_nombres.png`) · titre coupé au bord droit**
- Extrait : `ax.set_title("Loi des grands nombres : la fréquence se rapproche de la probabilité")`
- Problème : la fin de « probabilité » touche le bord de l'image (figsize `(7, 3.6)`).
- Correction : raccourcir le titre (« Loi des grands nombres : la fréquence tend vers 1/6 ») ou appeler `fig.tight_layout()`.

**23. SUGGESTION · `annexes/cheatsheets/pandas.md:3` · en-tête périmé**
- Extrait : « Aide-mémoire rempli au fil des chapitres (0A, 2, 12). »
- Problème : la cheatsheet contient déjà des lignes des ch. 3 et 4.
- Correction : « (0A, puis une section par chapitre) », comme pour la cheatsheet NumPy.

## Affirmations datées vérifiées sur le web

| # | Affirmation (fichier:ligne) | Verdict au 2026-10-04 | Source |
|---|---|---|---|
| 1 | Colab gratuit sans terminal (0A `01_fiche.md:161`) | **dépassé** (constat 3) | [Colab release notes](https://developers.google.cn/colab/release-notes), 2025-07-22 |
| 2 | Copy-on-Write : option en pandas 2.2, défaut en pandas 3.0 ; l'affectation en chaîne ne marche plus (0A `:1724`) | exact ; pandas 3.0.0 publié le 2026-01-21, `SettingWithCopyWarning` retiré | [whatsnew 3.0.0](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html) |
| 3 | `torch.load(..., weights_only=True)` par défaut depuis PyTorch 2.6 (0A `:998`) | exact (PyTorch 2.6 sorti le 2025-01-29) | [blog PyTorch 2.6](https://pytorch.org/blog/pytorch2-6/) |
| 4 | `main`, branche par défaut des nouveaux dépôts GitHub depuis le 2020-10-01 (0A `:1851`) | exact | [GitHub Changelog](https://github.blog/changelog/2020-10-01-the-default-branch-for-newly-created-repositories-is-now-main/) |
| 5 | `git switch` et `git restore` apparus dans Git 2.23, en 2019 (0A `:1851`) | exact | [notes de version de Git 2.23](https://github.com/git/git/blob/master/Documentation/RelNotes/2.23.0.adoc) |
| 6 | `int \| None` depuis Python 3.10, PEP 604 (0A `:812`) | exact | [PEP 604](https://peps.python.org/pep-0604/) |
| 7 | `list[float]` depuis Python 3.9, PEP 585 (0A `:812`) | exact | [PEP 585](https://peps.python.org/pep-0585/) |
| 8 | `default_rng` recommandé pour tout code neuf ; NEP 19 (0A `:1482`) | exact (`Generator` existe depuis NumPy 1.17) | [doc NumPy « Random sampling »](https://numpy.org/doc/stable/reference/random/index.html) |
| 9 | NumPy 2 affiche `np.float64(181.0)` (0A `:1289`) | exact | [notes de NumPy 2.0.0](https://numpy.org/doc/stable/release/2.0.0-notes.html) |
| 10 | Sous Windows avec NumPy 2, le type par défaut est `int64` (0A `05_solutions.md:582`) | exact | même source |
| 11 | `np.log` calcule le logarithme népérien (0B `:336`) | exact (« Natural logarithm, element-wise ») | [doc `numpy.log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html) |
| 12 | Poids de `torch.nn.Linear` de forme `(out_features, in_features)`, $y = xA^\top + b$ (0B `:549`) | exact ; vérifié avec torch 2.11 : `Linear(3, 2).weight.shape` vaut `(2, 3)` | [doc `torch.nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) |
| 13 | `df.cov()`, `df.corr()` « des colonnes numériques » (cheatsheet pandas `:65`) | **faux depuis pandas 2.0** (constat 5) | [doc `DataFrame.corr` (2.2)](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.DataFrame.corr.html) |
| 14 | La doc de pandas 3 est « proche de la version 2.2 » (0A `:1999`) | à nuancer (constat 21) | [whatsnew 3.0.0](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html) |

## Bilan

**Ce qui est solide.**
- Les 393 exemples `>>>` sont exacts avec les versions figées, et toutes les valeurs des exercices papier et du notebook 0A concordent avec les solutions.
- 11 affirmations datées sur 14 sont exactes au 2026-10-04, avec des sources à jour, et les 23 liens externes répondent.
- Les conventions sont cohérentes avec les ch. 1 à 11 et les annexes :
  - variance divisée par $n$ en 0B, `ddof` expliqué au ch. 2 ;
  - `log` = ln, avec l'exception des bits au ch. 6 ;
  - $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$ avec des poids `(n_in, n_out)`, et `(out, in)` pour PyTorch : 0B, ch. 10 et formulaire disent la même chose ;
  - `default_rng`, et l'arrondi au pair de `round`.
- Les erreurs fréquentes, les flashcards (une coquille), les 13 figures et les réponses modèles 💼 sont exactes.
- `wb.check` ignore bien la casse et les accents, comme le cours l'annonce.

**Les trois risques principaux.**
1. **Des réponses enregistrées à partir d'une mesure de temps.** Une seule réexécution sur une machine chargée a inversé la réponse attendue de 0B.40 e, et aucun contrôle ne l'a vu : `build_answers.py` enregistre fidèlement ce que dit le notebook. Le risque revient à chaque reconstruction (P1 en prévoit une dizaine). Il faut enregistrer des constantes et faire échouer la construction quand une mesure est aberrante (constats 1 et 2).
2. **Des affirmations sur les outils qui vieillissent hors des encadrés 🕰️**, et qu'aucune revérification ne couvre : terminal de Colab, `df.corr()` décrit comme sous pandas 1.x, documentation de pandas 3, sémantique de `git diff` et `git restore`. Il faudrait relire la prose « outillage » et les cheatsheets à chaque checkpoint, comme les versions de Colab (§21) (constats 3, 4, 5 et 21).
3. **Des fuites de réponses légères mais nombreuses dans la partie 0 et les indices de niveau 3**, alors que 0A et 0B sont la première expérience de `wb.check` : exemples de format qui sont des réponses, indices qui donnent la valeur, notation et message de démonstration qui divergent du cours. Corriger 0A et 0B en même temps que le constat 13 des ch. 10 et 11 (constats 6, 10, 11 et 12).
