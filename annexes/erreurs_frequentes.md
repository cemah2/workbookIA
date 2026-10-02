# Erreurs fréquentes

Les messages d'erreur rencontrés le plus souvent, avec leur cause et la solution. Complété au fil des chapitres (et après chaque correction P9). Pour les problèmes d'installation, voir aussi `00_setup/INSTALL_LOCAL.md` et `00_setup/COLAB.md`.

**Méthode face à une erreur Python** : lis la **dernière ligne** du message (le type d'erreur et sa description), puis remonte jusqu'à la première ligne qui mentionne **ton** code.

## Mise en place et outils

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'wb'` | la cellule de setup n'a pas été exécutée, ou l'environnement virtuel n'est pas activé | exécute la cellule de setup (Colab) ; en local, active `.venv` et fais `pip install -e .` |
| `⏳ Ex 3.4 : pas encore fait` | la réponse vaut encore `...`, ou la fonction lève `NotImplementedError` | ce n'est pas une erreur : écris ton code, puis relance la cellule |
| `❓ Ex 3.4 : aucune réponse enregistrée` | ID mal tapé, ou `answers.json` pas à jour | vérifie l'ID dans l'énoncé, puis `git pull` |
| `ℹ️ Ta librairie mylearn n'existe pas encore` | `mon_travail/mylearn/` n'a pas été créé | `python tools/start_chapter.py --init` (ou le numéro du chapitre) |
| tests `SKIPPED ... n'existe pas encore` | le module mylearn du chapitre n'a pas été copié | `python tools/start_chapter.py <chapitre>` |
| test `FAILED ... ⏳ pas encore implémenté` | la fonction contient encore `raise NotImplementedError` | implémente-la |
| `git pull` refuse de s'exécuter | un fichier hors de `mon_travail/` a été modifié | `git status`, puis `git restore <fichier>` (copie d'abord ta modification si tu y tiens) |
| `fatal: Need to specify how to reconcile divergent branches` | tu as des commits locaux et Claude en a poussé d'autres | `git config --global pull.rebase true` et `git config --global rebase.autoStash true`, puis `git pull` |

## Python et NumPy

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| `NameError: name 'x' is not defined` | la cellule qui crée `x` n'a pas été exécutée (ou le noyau a redémarré), ou faute de frappe | exécute les cellules au-dessus (*Run all* jusqu'ici) ; vérifie l'orthographe |
| le notebook marche chez toi, pas après redémarrage | cellules exécutées dans le désordre, variable créée par une cellule supprimée | *Restart and run all* avant de partager (0A.13) |
| `IndentationError`, `SyntaxError: expected ':'` | indentation incohérente, `:` oublié après `if`, `for`, `def` | 4 espaces par niveau ; `:` à la fin de la ligne d'en-tête |
| `TypeError: can only concatenate str (not "int") to str` | `"3" + 1` : un nombre lu dans un fichier est encore du texte | convertis : `int(text)`, `float(text)`, ou une f-string |
| `TypeError: '<' not supported between instances of 'NoneType' and 'int'` | comparaison avec une valeur manquante (`None`) | teste `x is None` **avant** la comparaison (0A.20) |
| `TypeError: f() missing 1 required positional argument` / `takes 2 positional arguments but 3 were given` | mauvais nombre d'arguments ; argument *keyword-only* passé par position | relis la signature (`help(f)`) ; nomme les arguments après `*` |
| `IndexError: list index out of range` | indice ≥ `len(l)` (les indices vont de 0 à `len(l) - 1`) | vérifie `len(l)` ; `l[-1]` pour le dernier |
| `KeyError: 'sex'` | la clé n'existe pas dans le dictionnaire (ou la colonne dans le DataFrame) | `d.get(k, défaut)`, `k in d`, `df.columns` |
| `ValueError: could not convert string to float: '3,450'` | virgule décimale, texte vide, `"NA"` | nettoie le texte (`strip`, `replace(",", ".")`) ou rattrape la `ValueError` (0A.25) |
| une liste vaut `None` | `l = l.sort()` : `sort` trie sur place et renvoie `None` | `l.sort()` seul, ou `l = sorted(l)` |
| modifier `b` modifie aussi `a` | `b = a` (même liste) ou `b = a[2:5]` (vue NumPy) | `b = a.copy()` |
| `0.1 + 0.2 == 0.3` vaut `False` | flottants approchés | `math.isclose`, `np.isclose`, `pytest.approx` |
| `ValueError: The truth value of an array with more than one element is ambiguous` | `and`, `or`, `not` ou `if` sur un array ou une colonne | `&`, `\|`, `~` avec des parenthèses ; `if len(a) == 0` ; `.any()` / `.all()` |
| `ValueError: operands could not be broadcast together with shapes (5,3) (5,)` | formes incompatibles (dimensions de droite différentes) | aligne les formes à droite ; `reshape(-1, 1)` (0A.8) |
| résultat de forme `(n, n)` au lieu de `(n,)` | broadcasting silencieux entre `(n,)` et `(n, 1)` | vérifie `.shape` ; `ravel()` ou `reshape` explicite |
| un nombre au lieu d'une valeur par colonne (ou l'inverse) | `axis` oublié ou inversé | `axis=0` : une valeur par colonne ; vérifie la forme du résultat |
| `x == np.nan` toujours `False` | NaN n'est égal à rien | `np.isnan(x)`, `pd.isna(x)` |
| `FileNotFoundError: [Errno 2] No such file or directory` | chemin relatif lancé depuis un autre dossier | `Path.cwd()` ; chemin construit depuis la racine du dépôt (`wb.datasets.data_dir()`) |
| ⏳ ou `NotImplementedError` persiste alors que ta fonction mylearn est écrite | fichier pas enregistré, ou le noyau garde l'ancienne version en mémoire | enregistre le fichier, redémarre le noyau, relance la cellule de setup (0A.26) |
| `TypeError: Object of type int64 is not JSON serializable` | un nombre NumPy (`np.int64`) dans des données à enregistrer en JSON | convertir avec `int(v)`, `float(v)` ou `v.item()` (0A.39) |
| `UnboundLocalError: cannot access local variable 'total'` | une fonction intérieure modifie une variable de la fonction englobante | la déclarer `nonlocal` (0A.42) |
| des fonctions créées dans une boucle donnent toutes le même résultat | une fermeture lit la variable de boucle quand on l'appelle, pas quand on la crée | `lambda x, k=k: ...`, ou une fabrique de fonctions (0A.43) |
| `RecursionError: maximum recursion depth exceeded` | fonction récursive sans cas de base (ou qui ne se rapproche jamais de lui) | écrire d'abord le cas de base, le tester sur une petite entrée (0A.44) |
| `TypeError: unsupported operand type(s) for +: 'int' and 'Vector2D'` | `sum(objets)` commence par `0 + objet` | définir `__radd__` (0A.48) ou donner la valeur de départ à `sum` |
| somme de pixels fausse, ou `RuntimeWarning: overflow encountered` | des entiers `uint8` (0 à 255) qui débordent | `int(pixel)`, ou `images.astype(np.int64)` avant de sommer (0A.55) |
| des tirages qui changent à chaque exécution malgré `np.random.seed(0)` | `np.random.seed` ne règle que l'ancien générateur global : `np.random.default_rng()` sans argument n'en dépend pas | `rng = np.random.default_rng(0)`, puis passer `rng` aux fonctions (2.14) |
| sur un tableau à 4 colonnes, `np.percentile(X, [25, 50, 75], axis=0)` a la forme `(3, 4)` et non `(4, 3)` | les quantiles demandés forment le **premier** axe du résultat | lire la ligne `k` pour le k-ième quantile, ou transposer (`.T`) (2.15) |
| le plus proche voisin d'un point est… lui-même | la distance d'un point à lui-même vaut 0 et `argmin` la choisit | mettre cette distance à `np.inf` avant `argmin` (`np.fill_diagonal(D, np.inf)` pour une matrice) (2.25) |
| `AxisError: axis is out of bounds for array of dimension 0` avec `scipy.stats.bootstrap` | SciPy attend une **séquence** d'échantillons, pas le tableau seul | `bootstrap((x,), np.mean, ...)` : noter la virgule (2.24) |
| un test pytest passe aussi sur une fonction fausse | le test ne vise pas le cas qui fait la différence (par exemple, des colonnes semblables pour tester `axis`) | un test par cas limite ; essayer chaque test sur une version volontairement fausse de la fonction (2.31) |
| un prior comme `np.array([0.6, 0.3, 0.1])` est refusé : « ne somme pas à 1 » | sa somme vaut 0,9999999999999999 : les flottants sont approchés | comparer la somme à 1 avec une tolérance, `abs(p.sum() - 1) <= 1e-8` (4.14) |
| une fonction accepte une vraisemblance `[0.7]` pour deux hypothèses et renvoie un résultat | le broadcasting étire un tableau de longueur 1 sans rien dire | vérifier `likelihood.shape == prior.shape` avant de calculer (4.14) |
| l'issue −1 est acceptée, et lit la dernière colonne d'un tableau | NumPy accepte les indices négatifs (`table[:, -1]`) | vérifier `0 <= o < n_outcomes` avant d'indexer (4.16) |
| toutes les lignes d'un historique sont identiques | la liste contient plusieurs fois **le même** tableau, modifié ensuite sur place (`posterior *= ...`) | créer un nouveau tableau à chaque tour (`posterior = posterior * ...`), ou ajouter une copie (4.16) |
| un tableau reçu en argument est modifié par la fonction | `np.asarray` ne copie pas un tableau NumPy, et `p *= ...` le modifie sur place | `p = p * ...` (un nouveau tableau), ou `np.array(p, dtype=float)` qui copie (4.14) |
| `ValueError: Integers to negative integer powers are not allowed` | `10 ** -k` avec un entier NumPy `k` (une valeur de `np.arange`) | écrire `10.0 ** -k` (5.13) |
| un gradient numérique absurde (des 0 et des millions) pour un point écrit `np.array([-1, 1])` | le tableau est d'**entiers** : `x[i] += 1e-5` est tronqué, et le point devient `[0, 0]` | travailler sur une copie en flottants, `np.array(x, dtype=float)` (5.16) |
| le point de départ de l'appelant a « bougé » après un calcul de gradient, d'un arrondi invisible à l'affichage | la fonction modifie le tableau reçu (`np.asarray` ne copie pas), et `+ h`, `- 2 * h`, `+ h` ne redonnent pas exactement la valeur | une copie, et remettre l'ancienne valeur telle quelle ; comparer avec `==`, pas avec `print` (5.15, 5.16) |
| une dérivée numérique très fausse avec un pas minuscule (`h = 1e-15`) | l'erreur d'arrondi, de l'ordre de $\frac{\varepsilon}{h}$, domine | `h` vers `1e-5` pour la différence centrée, vers `1e-4` pour la dérivée seconde (5.12, 5.13) |
| tous les points d'un chemin de descente sont identiques | `path.append(x)` puis une mise à jour en place (`x -= lr * g`) : la liste contient le même tableau | `x = x - lr * g` (un nouveau tableau), ou `path.append(x.copy())` (5.18) |
| `find_local_extrema` trouve des extrema sur un plateau, ou n'en trouve pas sur une série qui en a | comparaisons non strictes (`<=`), ou au contraire un vrai creux plat que la définition stricte ignore | la définition est stricte ; pour les plateaux, un traitement à part (5.14) |
| `RuntimeWarning: divide by zero encountered in log2`, puis une entropie `nan` | `p * np.log2(p)` calculé sur toutes les issues : $0 \times (-\infty)$ donne `nan` | ne sommer que sur `p[p > 0]` ($0 \log 0 = 0$) (6.12) |
| une distribution valide est refusée : « la somme ne vaut pas 1 » | `p.sum() == 1` sur des flottants : `np.full(7, 1 / 7).sum()` vaut 0.9999999999999998 | une tolérance : `abs(p.sum() - 1) <= 1e-6` (6.12) |
| une probabilité `NaN` passe les contrôles | `p <= 0` est faux pour `NaN`, comme toute comparaison avec `NaN` | tester `~(p > 0)`, ou `np.isfinite(p)` (6.12) |
| un générateur de tokens semble vide | il a été lu deux fois ; un générateur ne se lit qu'une fois | compter en un seul passage, `collections.Counter(tokens)` (6.13) |
| `TypeError: '<' not supported between instances of 'str' and 'int'` (ou d'autres types) dans `heapq` | deux groupes de même probabilité : Python compare l'élément suivant du tuple, les listes de symboles, élément par élément, et bute sur deux symboles de types différents | un compteur entre la probabilité et le groupe, `(p, numéro, groupe)` : deux groupes ne sont jamais comparés (6.23) |
| `np.sqrt` d'une matrice de distances au carré donne des `nan` | l'identité $\lVert \mathbf{a} \rVert^2 - 2\,\mathbf{a} \cdot \mathbf{b} + \lVert \mathbf{b} \rVert^2$ peut donner de minuscules valeurs négatives (arrondis) | `np.maximum(D, 0.0)` avant la racine (7.13) |
| les modèles d'un méta-estimateur donnent tous les mêmes scores | la boucle entraîne **le même** objet à chaque tour (`fit` renvoie `self`) : la liste contient plusieurs fois le dernier modèle | une copie neuve par modèle, `copy.deepcopy(estimator)` (7.22) |
| les centres de k-means ont toutes leurs coordonnées égales | `X[labels == j].mean()` fait la moyenne de toutes les valeurs du groupe, toutes features confondues | `X[labels == j].mean(axis=0)` (7.27) |
| la boucle de k-means s'arrête toujours après une itération | l'ancienne affectation est remplacée par la nouvelle **avant** d'être comparée : la comparaison est toujours vraie | comparer, puis mémoriser (7.27) |
| un découpage `X_train`, `y_train` donne des scores au niveau du hasard | `X` et `y` ont été mélangés chacun de leur côté (deux permutations) : les lignes ne correspondent plus | tirer **une** permutation d'indices et indexer tous les tableaux avec elle (8.13, 8.20) |
| `df[idx]` sélectionne des colonnes au lieu de lignes (`KeyError`, ou un tableau de la mauvaise forme) | sur un `DataFrame`, les crochets sélectionnent des colonnes | `np.asarray(df)[idx]`, ou `df.iloc[idx]` (8.22) |
| un générateur d'indices de folds semble vide au deuxième usage | un générateur (`yield`, `cv.split(X)`) ne se parcourt qu'une fois | le convertir en liste, `list(cv.split(X))` (8.14) |

## Maths et algèbre linéaire

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0` | `A @ B` avec un nombre de colonnes de `A` différent du nombre de lignes de `B` | écris les formes : `(m, n) @ (n, p)` ; transpose si besoin (`A @ B.T`) (0B.20) |
| un vecteur au lieu d'un nombre, ou des valeurs fausses sans message d'erreur | `*` (élément par élément) confondu avec `@` (produit scalaire ou matriciel) | `*` : Hadamard ; `@` : produit scalaire ou matriciel (0B.18) |
| `v.T` ne change rien | un vecteur `(n,)` n'a qu'un axe : sa transposée est lui-même | `v.reshape(-1, 1)` pour une colonne `(n, 1)` ; vérifie `.shape` |
| `RuntimeWarning: divide by zero encountered in log`, résultat `-inf` | `np.log(0)` : une probabilité nulle | ajoute un petit `eps` (`np.log(p + 1e-12)`) ou travaille en log-probabilités (0B.15) |
| `RuntimeWarning: invalid value encountered in log`, résultat `nan` | logarithme d'un nombre négatif | vérifie le signe des entrées avant le `log` |
| `RuntimeWarning: overflow encountered in exp`, résultat `inf` | `np.exp` d'un grand nombre (au-delà de 709 environ en `float64`) | réécris la formule (par exemple la sigmoïde avec `np.exp(-abs(x))`) ou utilise `scipy.special.expit` |
| un produit de probabilités vaut `0.0` | underflow (*sous-dépassement*) : le produit est trop petit pour un `float64` (sous $10^{-308}$ environ, il perd des chiffres ; sous $5 \times 10^{-324}$, il devient 0) | additionne les logarithmes au lieu de multiplier (0B.15, 0B.E3, 4.18) |
| un posterior vaut `[nan nan nan]`, sans aucun message au moment du problème | les produits prior × vraisemblances de **toutes** les hypothèses sont tombés à 0 (underflow), puis la normalisation a calculé 0/0 | calculer en log-probabilités et soustraire le maximum avant l'exponentielle (log-sum-exp) (4.18, 4.24) |
| `RuntimeWarning: invalid value encountered in multiply` et un `nan` dans un posterior sur une grille | `h * np.log(grid)` avec $h = 0$ et un $\theta = 0$ dans la grille : $0 \times (-\infty)$ n'est pas défini | n'ajouter $h \log\theta$ que si $h > 0$, puisque $\theta^0 = 1$ (4.24) |
| `np.log(np.exp(-1000) + np.exp(-1001))` donne `-inf` | `np.exp(-1000)` vaut 0 en `float64` | `scipy.special.logsumexp([-1000, -1001])`, ou soustraire le maximum avant l'exponentielle (4.18) |
| `stats.beta(a, b).cdf(0.5)` pour $P(\theta > 0{,}5)$ donne la probabilité du mauvais côté | `cdf` donne la probabilité à **gauche** du seuil | `stats.beta(a, b).sf(0.5)`, c'est-à-dire `1 - cdf(0.5)` (4.22) |
| le posterior Beta d'une pièce ne correspond pas à la grille | `stats.beta(h, t)` : le prior uniforme $\mathrm{Beta}(1, 1)$ a été oublié | $\mathrm{Beta}(h + 1, t + 1)$ (4.22) |
| `RuntimeWarning: overflow encountered in reduce` et un résultat `inf` | `np.prod` de beaucoup de grands nombres | passe par les logarithmes : `np.exp(np.log(x).sum())`, ou garde le résultat en logarithme (0B.37) |
| `u @ v.T` donne un nombre au lieu d'une matrice | `.T` ne change rien à un vecteur `(n,)` : c'est un produit scalaire | `u[:, None] @ v[None, :]` pour le produit extérieur (0B.45) |
| `np.log(100)` donne 4,6 au lieu de 2 | `np.log` est le logarithme **népérien** | `np.log10` ou `np.log2` selon la base voulue |
| `np.linalg.LinAlgError: Singular matrix` | la matrice n'est pas inversible (déterminant nul : une ligne proportionnelle à une autre) | vérifie les données (colonne dupliquée ?) ; `np.linalg.lstsq` pour un système sans solution unique (0B.21) |
| une somme « de 1 à n » est fausse d'un terme | `range(1, n)` s'arrête à `n - 1` | `range(1, n + 1)` : la borne haute d'un $\Sigma$ est incluse (0B.R1) |
| `math.floor(-3.7)` et `int(-3.7)` ne donnent pas la même chose | `floor` va vers le bas (−4), `int` tronque vers zéro (−3) | choisis selon le sens voulu (0B.2) |
| des distances dominées par une seule feature | features dans des unités très différentes (grammes et millimètres) | standardise les colonnes avant de calculer des distances (0B.8, 0A.52) |
| `LinAlgError: Singular matrix` avec `np.linalg.solve(X.T @ X, X.T @ y)`, ou, plus souvent, **aucun message** mais des poids énormes ou absurdes | features redondantes (une colonne copiée, une feature somme de deux autres) ou plus de features que d'exemples : $\mathbf{X}^\top\mathbf{X}$ est singulière, mais les arrondis la font souvent paraître inversible | vérifier `np.linalg.matrix_rank(X) < X.shape[1]` ; `np.linalg.lstsq` (la solution la plus courte), ou Ridge avec `alpha > 0` (fiche du ch. 9, §9.3 et §9.5) |
| `RankWarning: Polyfit may be poorly conditioned` | `np.polyfit` de degré élevé sur des $x$ loin de 0 et peu étalés (des années : 1990 à 2020), ou trop peu de points pour le degré | ramener $x$ vers $[-1, 1]$ (ou standardiser), baisser le degré (fiche du ch. 9, §9.3) |

## pandas

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| `KeyError: 'body_mass'` | nom de colonne mal tapé | `df.columns` ; copie le nom exact (`body_mass_g`) |
| `SettingWithCopyWarning` ou `FutureWarning: ChainedAssignmentError` ; selon les cas, rien ne change | affectation en chaîne : `df[masque]["col"] = v` ou `df["col"][masque] = v` | `df.loc[masque, "col"] = v` |
| `ValueError: The truth value of a Series is ambiguous` | `and` / `or` entre deux conditions | `(cond1) & (cond2)`, `(cond1) \| (cond2)` |
| `y` de forme `(n, 1)` au lieu de `(n,)` | `df[["species"]]` (doubles crochets) renvoie un DataFrame | `df["species"].to_numpy()` (0A.35) |
| moyenne `NaN` pour un groupe | toutes les valeurs du groupe sont manquantes | `groupby(...).agg(["count", "mean"])` pour voir les effectifs |
| moins de lignes que prévu après `dropna()` | `dropna()` retire toute ligne avec **au moins une** valeur manquante | `dropna(subset=[...])` si seules certaines colonnes comptent |
| `df["f1"].idxmax()` renvoie 17 au lieu du seuil cherché | `idxmax` renvoie l'**étiquette** de la ligne (son index), pas la valeur d'une autre colonne | `df.loc[df["f1"].idxmax(), "threshold"]` (3.20) |
| les espèces sortent dans le désordre (Adelie, Gentoo, Chinstrap) | `value_counts` trie par effectif décroissant | `.sort_index()` ou `.reindex(SPECIES)` (4.15) |
| une espèce absente d'un groupe disparaît du résultat | `value_counts` ne compte que les valeurs présentes | `.reindex(toutes_les_espèces, fill_value=0)` (4.23) |
| `pd.crosstab(..., normalize="columns")` donne $P(\text{espèce} \mid \text{île})$ alors qu'on voulait $P(\text{île} \mid \text{espèce})$ | `normalize="columns"` divise par le total de chaque colonne, `normalize="index"` par celui de chaque ligne | la variable **après** la barre est celle dont on divise le total (4.15) |
| une pureté de clustering trop basse (ou trop haute) | `pd.crosstab(clusters, classes).max()` prend le maximum de chaque **colonne** (classe), pas de chaque ligne (cluster) | `.max(axis=1).sum() / n` (7.17) |

## scikit-learn

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| `ConvergenceWarning: Stochastic Optimizer: Maximum iterations (30) reached and the optimization hasn't converged yet.` | l'entraînement s'est arrêté au bout de `max_iter` passages alors que la loss baissait encore | voulu en `FAST_MODE` (1.23) ; sinon, augmente `max_iter`. Un avertissement se lit toujours, il ne s'ignore pas |
| `ValueError: Expected 2D array, got 1D array instead` | `fit` ou `predict` reçoit un seul échantillon, ou une seule feature, sous forme de vecteur | les features sont toujours un tableau 2D `(n_samples, n_features)` : `X.reshape(-1, 1)` pour une seule feature, `X.reshape(1, -1)` pour un seul échantillon |
| `NotFittedError: This DecisionTreeClassifier instance is not fitted yet` | `predict` ou `score` appelé avant `fit` | appelle `model.fit(X_train, y_train)` d'abord (encadré 🧮 de la fiche du ch. 1) |
| `UserWarning: X does not have valid feature names` (ou l'inverse) | modèle entraîné sur un DataFrame et utilisé sur un array NumPy (ou l'inverse) | garde le même type et les mêmes colonnes, dans le même ordre, à l'entraînement et à la prédiction |
| une « probabilité » de 1,0 pour une entrée absurde | `predict_proba` d'un arbre renvoie les proportions de la feuille atteinte, pas une vraie mesure de confiance | ne pas lire `predict_proba` comme une certitude ; un classifieur ne sait pas dire « je ne sais pas » (1.19) |
| `ValueError: Target is multiclass but average='binary'` | `precision_score`, `recall_score` ou `f1_score` sur plus de deux classes, avec le réglage par défaut | choisir `average="macro"`, `"weighted"`, `"micro"` ou `None` (une valeur par classe) (3.25) |
| `ValueError: pos_label=1 is not a valid label` | étiquettes textuelles (`"spam"`, `"ham"`) ou autres que 0/1 | préciser `pos_label="spam"` (3.16) |
| `UndefinedMetricWarning: Precision is ill-defined and being set to 0.0` | aucune prédiction positive : la precision vaut 0/0 | vérifier le modèle (prédit-il toujours la même classe ?) ; `zero_division=0` ou `1` fixe la valeur et fait taire l'avertissement (3.16) |
| TP et TN échangés en lisant une matrice de `confusion_matrix` | scikit-learn trie les étiquettes : `[[TN, FP], [FN, TP]]` pour 0/1, pas TP en haut à gauche comme dans le livre | lire les étiquettes des axes ; `labels=[1, 0]` impose un autre ordre (3.17) |
| `tn, fp, fn, tp = confusion_matrix(...).ravel()` donne des mesures échangées (sensibilité ↔ spécificité, precision ↔ NPV) | avec des étiquettes textuelles, l'ordre trié peut mettre la classe positive en premier (`"malade" < "sain"`) | `confusion_matrix(y_true, y_pred, labels=[negative, positive]).ravel()`, ou des masques booléens ; tester sur un petit exemple calculé à la main (3.17) |
| `roc_auc_score` renvoie une valeur sous 0,5 | scores inversés (probabilité de la mauvaise classe, `predict_proba(X)[:, 0]` au lieu de `[:, 1]`) ou étiquettes inversées | passer la probabilité de la classe positive, `model.predict_proba(X)[:, 1]` (3.24) |
| la courbe ROC ou PR ne va pas jusqu'au bout, ou a trop peu de points | `roc_curve` retire par défaut des points inutiles au dessin (`drop_intermediate=True`) ; on a passé des classes prédites au lieu de scores | passer des **scores** (`predict_proba`, `decision_function`), pas `predict` ; `drop_intermediate=False` pour tous les seuils (3.24) |
| `KMeans` ne donne pas le même résultat qu'avant une mise à jour de scikit-learn | depuis la version 1.4, `n_init="auto"` : un seul départ avec k-means++ au lieu de 10 | fixer `n_init` et `random_state` explicitement ; lire la documentation de la version installée (7.9) |
| `DBSCAN(...).predict(X_new)` : `AttributeError` | DBSCAN et HDBSCAN n'ont pas de `predict` : ils ne savent étiqueter que les données de `fit` | `fit_predict(X)` ou `labels_` ; pour de nouveaux points, relancer le clustering ou entraîner un classifieur sur les clusters (7.18) |
| DBSCAN met presque tout en bruit (−1), ou tout dans un seul cluster | `eps` mal choisi pour l'échelle des données (la valeur par défaut, 0,5, n'a de sens qu'à l'échelle des données ; après une standardisation, c'est un point de départ) | standardiser, puis régler `eps` (courbe des distances au k-ième voisin), ou essayer HDBSCAN (7.18) |
| `KMeans.score(X)` est négatif | convention de scikit-learn : un score est « plus grand = meilleur », et `score` renvoie l'**opposé** de l'inertie | utiliser `inertia_` pour l'inertie, `score` pour comparer des modèles (7.9) |
| `ValueError: The least populated class in y has only 1 member, which is too few` | `train_test_split(..., stratify=y)` avec une classe d'un seul exemple : impossible de la mettre des deux côtés | regrouper les classes trop rares, ou découper sans stratifier (8.13) |
| `UserWarning: The least populated class in y has only 3 members, which is less than n_splits=5` | `StratifiedKFold` (ou `cross_val_score` d'un classifieur) avec une classe plus petite que le nombre de folds : certains folds ne la verront pas | réduire `n_splits`, regrouper des classes, ou accepter des folds sans cette classe en connaissance de cause (8.21) |
| `cross_val_score(..., cv=5)` de scikit-learn ne donne pas les mêmes scores que `KFold(5)` | avec un entier, scikit-learn utilise `StratifiedKFold` pour un classifieur (et `KFold` sinon), sans mélange | passer le découpeur explicitement, `cv=KFold(5)` ou une liste de paires d'indices (8.22) |
| des scores de validation croisée très irréguliers (0,97 puis 0,64) | fichier trié par classe et folds consécutifs sans mélange : chaque fold contient surtout une classe | `shuffle=True`, ou une k-fold stratifiée (8.14, 8.22) |
| `ValueError: The 'groups' parameter should not be None` | `GroupKFold` (ou `cross_val_score` avec lui) sans le tableau des groupes | `cross_val_score(model, X, y, cv=GroupKFold(5), groups=groups)` (8.19) |
| `ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: …, tolerance: …` | `Lasso` ou `ElasticNet` : features d'échelles très différentes, features presque identiques, ou `alpha` très petit | standardiser les features (z-score calculé sur l'entraînement seul), augmenter `max_iter`, ou prendre un `alpha` plus grand (fiche du ch. 9, §9.5) |
| `TypeError: got an unexpected keyword argument 'squared'` | `mean_squared_error(..., squared=False)`, retiré en version 1.6 | `root_mean_squared_error(y_true, y_pred)` (depuis la version 1.4) |
| des scores négatifs avec `cross_val_score(..., scoring="neg_mean_squared_error")` | scikit-learn maximise toujours un score : il renvoie l'**opposé** des erreurs | lire `-scores` (ou `"neg_root_mean_squared_error"` pour la RMSE) |
| `Ridge(alpha=1.0)` et `Lasso(alpha=1.0)` n'ont pas du tout la même force | dans `Ridge`, le terme d'erreur est une **somme** de carrés ; dans `Lasso`, il est divisé par $2n$ ; et les pénalités elles-mêmes diffèrent ($\lVert w \rVert^2$ et $\lVert w \rVert_1$) | régler `alpha` séparément pour chaque modèle, par validation croisée (fiche du ch. 9, §9.5) |
| les prédictions d'une régression pénalisée changent quand une feature passe des mètres aux kilomètres (sans pénalité, seul son poids serait multiplié par 1 000) | la pénalité traite tous les poids de la même façon, alors qu'un poids dépend de l'unité de sa feature | standardiser les features avant `Ridge` ou `Lasso`, avec les statistiques de l'entraînement seul |

## PyTorch

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| `RuntimeError: Can't call numpy() on Tensor that requires grad` | la fonction passe par NumPy (`np.asarray`, `wb.synth.rosenbrock`…) : PyTorch ne peut plus suivre les opérations | n'écrire la fonction qu'avec des opérations de tenseurs (`+`, `*`, `**`, `torch.exp`…) (5.21) |
| un gradient d'autograd et un gradient numérique s'écartent de $10^{-4}$ | calcul en `float32` (7 chiffres) | comparer en `float64` : `torch.tensor(x, dtype=torch.float64, requires_grad=True)` (5.21) |
| la « dérivée » de ReLU en 0 change selon la façon d'écrire la fonction | en un point anguleux, chaque opération a sa convention (`torch.relu` : 0, `torch.clamp` : 1, `torch.maximum` : 0,5) | ne pas tester un gradient sur un point anguleux (5.21) |

*(complété à partir du ch. 20)*

## Erreurs de raisonnement (ML)

| Symptôme | Cause probable | Solution |
|---|---|---|
| NumPy et pandas donnent deux écarts-types différents pour la même colonne | NumPy divise par `n` (`ddof=0`), pandas par `n - 1` (`ddof=1`) | préciser `ddof` à chaque calcul (2.4, 2.29) |
| `np.cov(X)` renvoie une matrice énorme ($n \times n$) | NumPy traite chaque **ligne** comme une variable | `np.cov(X, rowvar=False)` (2.28) |
| une moyenne ou une variance vaut `nan` | une valeur manquante dans la colonne | `dropna()` d'abord, ou `np.nanmean` en connaissance de cause (0A) |
| deux exécutions « identiques » donnent des tirages différents | pas de graine, ou un générateur global modifié ailleurs dans le code | `rng = np.random.default_rng(seed)`, passé explicitement aux fonctions (2.14) |
| un « sous-échantillon » contient des doublons | `rng.choice` tire **avec** remise par défaut | `replace=False` (2.21) |
| un intervalle de confiance bootstrap très large | des rééchantillons plus petits que l'échantillon (comme dans le livre) | rééchantillons de taille $n$ (2.23) |
| la moyenne décrit mal « l'individu typique » | distribution asymétrique ou valeurs extrêmes | donner la médiane et regarder l'histogramme (2.1, 2.13) |
| la règle 68-95-99,7 donne des parts fausses | la distribution n'est pas normale (asymétrique, à plusieurs bosses) | regarder l'histogramme avant d'appliquer la règle (2.18) |
| « corrélation nulle, donc aucun lien » | un lien non linéaire (en U, en cercle) | tracer le nuage de points (2.27) |
| « forte corrélation, donc cause » | variable de confusion, causalité inversée ou coïncidence | chercher une troisième variable ; seule une expérience contrôlée établit une cause (2.10) |
| un intervalle de confiance étroit, mais une conclusion fausse | échantillon biaisé : le bootstrap ne mesure que le hasard de l'échantillonnage | se demander d'abord qui est dans l'échantillon (2.10) |
| l'intervalle bootstrap bouge d'une exécution à l'autre | bruit de Monte-Carlo : trop peu de rééchantillons, ou pas de graine | quelques milliers de rééchantillons (`n_boot`) et une graine fixée (2.22, 2.23) |
| `scipy.stats.bootstrap` et ton code ne donnent pas le même intervalle | SciPy utilise par défaut la méthode BCa et 9 999 rééchantillons | `method="percentile"` et le même nombre de rééchantillons pour comparer (2.24) |
| une corrélation change de signe quand on sépare les groupes | paradoxe de Simpson : les groupes (les espèces) sont décalés les uns par rapport aux autres | calculer la corrélation dans chaque groupe et colorer le nuage par groupe (2.28) |
| un seul point fait basculer une droite ou une corrélation | point influent (valeur aberrante isolée) | tracer le nuage ; comparer les résultats avec et sans ce point (2.30) |
| « le plus proche voisin » ne veut plus rien dire en grande dimension | fléau de la dimension : pour des points au hasard, toutes les distances se ressemblent (le contraste s'effondre) | vérifier le contraste sur les vraies données ; réduire la dimension (ch. 12) (2.25) |
| un score de test très bon, puis décevant en production, sur des données datées | découpage au hasard d'une série temporelle : les données ne sont pas i.i.d. | découper dans le temps (ch. 8, 22) |
| 100 % (ou presque) sur les données d'entraînement, beaucoup moins sur de nouvelles données | le modèle a mémorisé ses exemples au lieu de généraliser | juger un modèle **uniquement** sur un jeu de test mis de côté avant l'entraînement (1.14) |
| la loss oscille, puis explose (jusqu'à `nan`) | learning rate trop grand pour la direction la plus courbée | le diviser par 2 ou 10, tracer la loss (5.19, 5.25) |
| la loss baisse, mais avec une lenteur désespérante | learning rate trop petit, ou surface mal conditionnée (une vallée étroite) | augmenter le learning rate tant qu'il reste stable ; au ch. 19, momentum et Adam (5.19, 5.25) |
| la loss stagne longtemps, puis repart | un plateau autour d'un point selle | patience, un peu de bruit (descente stochastique), du momentum (5.20) |
| « le gradient est nul, donc c'est un minimum » | un point selle (ou un maximum) a aussi un gradient nul | regarder la courbure dans plusieurs directions, axes **et** diagonales (5.24) |
| un learning rate qui atteint le minimum en premier, mais n'y reste pas | juste au-dessus de la limite : l'oscillation en travers de la vallée ne s'amortit plus | vérifier qu'on **reste** au minimum, pas seulement qu'on l'atteint (5.25) |
| une cross-entropy (ou une KL) infinie | une issue qui se produit a la probabilité 0 dans le code, ou dans le modèle | un lissage de Laplace, placé au bon endroit (6.17) |
| une KL qui ne correspond pas à l'envoi voulu (plus grande ou plus petite que prévu) | arguments inversés : `kl(code, données)` | $\mathrm{KL}(\text{données} \,\|\, \text{code})$, comme $H(\text{données}, \text{code})$ (6.5, 6.18) |
| une perplexité absurde (55 au lieu de 16) | bases mélangées : `np.exp` d'une moyenne de `np.log2` | bits avec `np.log2` et `2 **`, nats avec `np.log` et `np.exp` (6.22) |
| une perplexité calculée avec `2 ** loss` sur une loss de PyTorch | la loss de PyTorch est en nats | `np.exp(loss)` (6.20) |
| `scipy.stats.entropy` renvoie 3,11 au lieu de 4,49 bits | sa base par défaut est $e$ (des nats) | `scipy.stats.entropy(p, base=2)` (6.19) |
| « le Morse descend sous l'entropie » | on compare des points et des traits (et un silence oublié) à des bits | compter les silences ; comparer des symboles binaires avec des symboles binaires (6.4, 6.24) |
| deux entropies de lettres « incomparables » | calculées sur des alphabets différents (26 ou 42 lettres) : ce ne sont pas les mêmes distributions | mesurer sur le même alphabet (6.14) |
| « mon modèle de perplexité 15 bat le leur, à 20 » | tokenizers ou données de test différents : la perplexité se compte par token | même tokenizer et mêmes données, ou des bits par caractère (6.E3) |
| un score de test parfait, trop beau pour être vrai | **fuite de données** : une feature contient la réponse (le label recodé), ou le jeu de test a servi à l'entraînement | pour chaque feature : l'aurai-je au moment de prédire ? Découper train/test **avant** tout traitement (1.20, ch. 8 et 12) |
| le score de test baisse dès qu'on essaie le modèle en vrai | les hyperparamètres ont été réglés d'après le score **sur le test** | régler sur un jeu de validation (ou par validation croisée) ; ne regarder le test qu'une fois, à la fin (1.Q6, ch. 8) |
| des groupes (clustering) ou des voisins « absurdes » | une feature à grands nombres (des grammes) domine les distances | mettre les features à la même échelle (standardisation) avant de calculer des distances (1.R1, 1.21, ch. 12) |
| la loss grandit d'epoch en epoch, puis devient `inf` ou `nan` | learning rate trop grand : chaque correction dépasse la cible | diviser le learning rate par 10 et regarder la courbe de loss (1.17) |
| la loss baisse très lentement | learning rate trop petit | le multiplier par 10 ; essayer quelques valeurs sur une échelle logarithmique (1.17, ch. 19) |
| une accuracy de 44 % semble « pas si mal » sur Penguins | la classe majoritaire fait déjà 44 % : une accuracy s'interprète toujours par rapport à la référence la plus simple | comparer à un modèle qui prédit toujours la classe la plus fréquente (1.R2, ch. 3) |
| « le test repère 99 % des malades, donc un positif est malade à 99 % » | confusion de $P(\text{positif} \mid \text{malade})$ et de $P(\text{malade} \mid \text{positif})$ ; oubli de la prévalence | arbre des fréquences naturelles : la precision dépend de la prévalence (3.7, 3.21, ch. 4) |
| une accuracy de 99 % qui ne détecte aucune fraude | classes très déséquilibrées : répondre toujours « négatif » suffit | recall, precision, courbe precision-recall, MCC ; comparer à la classe majoritaire (3.Q9, E1) |
| un recall de 100 % annoncé comme un succès | tout est déclaré positif : la precision vaut la prévalence | toujours annoncer precision **et** recall, ou le F1 (3.Q11) |
| le F1 d'une precision de 0,8 et d'un recall de 0,2 calculé à 0,5 | moyenne ordinaire au lieu d'une moyenne harmonique | $F_1 = \frac{2PR}{P + R}$ = 0,32 (3.8) |
| un F1 micro élevé cache une classe rare mal reconnue | le micro (= accuracy) est dominé par les grandes classes | F1 macro et rapport par classe (`classification_report`) (3.6, 3.25) |
| la precision mesurée à l'hôpital ne se retrouve pas dans un dépistage de masse | la prévalence est beaucoup plus faible dans la population générale | recalculer la precision avec la prévalence de la population visée (3.8, 3.21) |
| une belle AUC, mais la plupart des alertes sont fausses | la ROC ne dépend pas de la prévalence ; avec peu de positifs, la precision s'effondre | courbe precision-recall et average precision (3.26, 3.27) |
| « le modèle annonce 0,97, donc 97 % de chances » | le modèle n'est pas calibré (souvent trop sûr de lui) | diagramme de fiabilité, score de Brier ; recalibrer sur un jeu de validation (3.28) |
| le seuil de décision réglé sur le jeu de test | le test a servi à choisir : son score est trop optimiste | choisir le seuil sur un jeu de validation, garder le test pour la fin (3.29, ch. 8) |
| « une chance sur un million qu'un innocent corresponde, donc une chance sur un million que l'accusé soit innocent » | erreur du procureur : $P(O \mid H)$ confondu avec $P(H \mid O)$ | écrire la question avec « sachant » ; compter les innocents compatibles dans toute la population examinée (4.10) |
| $P(\text{face, face})$ calculé comme $P(\text{face})^2$ pour une pièce dont on ignore le biais | les lancers ne sont indépendants que **sachant** la pièce | passer par les hypothèses : $\sum_H P(H)\,P(\text{face} \mid H)^2$ (4.5, 4.Q9) |
| des vraisemblances « corrigées » pour qu'elles somment à 1 | confusion avec le prior : les vraisemblances de plusieurs hypothèses n'ont pas à sommer à 1 | seuls les priors et les posteriors somment à 1 ; l'évidence s'en charge (4.Q7, 4.14) |
| une hypothèse ne remonte jamais, quelles que soient les données | son prior vaut exactement 0 (ou une observation l'a rendue impossible) | un prior petit, mais jamais nul, sauf impossibilité certaine (∂ 4.7, 4.21) |
| un posterior sûr à 99,99 % d'une hypothèse qui explique mal les données | Bayes compare seulement les hypothèses proposées : il choisit la moins mauvaise | vérifier que le modèle tient la route (fréquences observées contre prédites) ; ajouter des hypothèses (4.Q10, 4.19) |
| l'intervalle bootstrap de dix piles vaut (0 ; 0) | tous les rééchantillons ne contiennent que des piles : le bootstrap ne voit aucune variabilité | avec très peu de données, un intervalle de crédibilité (prior raisonnable) (4.25) |
| des sondes « indépendantes » promettent une erreur minuscule, que les faits démentent | leurs erreurs sont liées : elles se trompent pour la même raison | vérifier l'indépendance sachant l'état sur des cas connus ; varier les modèles de sondes (4.8, 4.20) |
| un seuil qui garde 99 % des positifs de validation n'en garde que 98,8 % sur le test | le bas de la distribution des scores de quelques centaines de positifs varie beaucoup d'un échantillon à l'autre | prendre une marge (quantile plus bas, mesurée par bootstrap) ou estimer ce bas avec tous les positifs ; annoncer une garantie avec sa marge d'erreur (3.29) |
| des scores un-contre-tous comparés comme des probabilités (« 0,9 pour A, donc A à 90 % ») | chaque modèle binaire est entraîné séparément : ses scores n'ont ni la même échelle que les autres, ni une somme de 1 | calibrer les scores, ou prendre un modèle multi-classe natif (softmax) ; ne garder que l'argmax pour décider (7.Q6) |
| un un-contre-un qui donne deux classes différentes d'une exécution ou d'une bibliothèque à l'autre | égalité de votes tranchée par des règles différentes (plus petit indice, confiance cumulée) | fixer la règle et la documenter ; regarder les votes (7.2, 7.23) |
| « le seuil de 0,5 est le bon » | 0,5 ne minimise le coût que si un faux positif et un faux négatif coûtent autant | $t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ pour des probabilités calibrées, puis vérification sur la validation (7.Q4, 7.10) |
| « k = 10 donne l'inertie la plus basse, c'est le meilleur k » | la meilleure inertie possible baisse toujours quand $k$ augmente (0 quand chaque point est seul) | coude, silhouette, stabilité, besoin métier (7.Q8, 7.29, 7.E3) |
| deux lancers de k-means donnent des clusters différents | minima locaux : le résultat dépend des centres de départ | k-means++ et plusieurs départs (`n_init`), une graine fixée (7.3, 7.26) |
| k-means coupe en travers des groupes allongés ou emboîtés | il fait des régions convexes (cellules de Voronoï) autour de centres : il suppose des groupes ronds | regarder les données ; DBSCAN ou HDBSCAN pour des formes quelconques (7.18) |
| la pureté (ou l'inertie) s'améliore « toute seule » quand on ajoute des clusters | avec $k = n$, chaque point est seul : pureté 1, inertie 0 | comparer à $k$ fixé, ou avec une mesure qui pénalise les clusters inutiles (silhouette, indice de Rand ajusté) (7.17, 7.31) |
| ajouter des features fait baisser l'accuracy de validation | phénomène de Hughes : la densité des exemples s'effondre et le modèle apprend le bruit | sélection de features, PCA, régularisation, plus d'exemples (7.30, ch. 9 et 12) |
| « une densité de 0,08 = 8 % de chances qu'une case soit occupée » | une densité est un nombre moyen d'échantillons par case, pas une probabilité (elle peut dépasser 1) | calculer la vraie probabilité qu'une case soit vide (7.4) |
| raisonner sur un espace à 100 features comme sur le plan | en grande dimension, les volumes et les distances ne se comportent pas comme en 2D ou en 3D | calculer (densités, distances, volumes) plutôt qu'imaginer (7.19 à 7.21) |
| standardiser le jeu de test avec sa propre moyenne et son propre écart-type | le test sert alors un peu à l'entraînement, et les deux jeux ne sont plus mis à la même échelle | les statistiques des seules données d'entraînement, appliquées aux deux jeux (7.24, ch. 8) |
| annoncer le score de validation du meilleur réglage | ce score est le maximum de plusieurs scores bruités : il contient une part de chance | noter le réglage retenu une seule fois sur un test jamais consulté (∂ 8.6, 🔮 8.17) |
| choisir le modèle (ou le seuil, ou le nombre d'epochs) en regardant le test | le test devient un jeu de validation, et son score n'est plus une estimation honnête | une validation, ou une validation croisée, pour choisir ; le test une seule fois, à la fin (8.24) |
| standardiser, imputer ou sélectionner des features sur toutes les données avant la validation croisée | chaque fold de validation a déjà influencé ce prétraitement : les scores sont optimistes, jusqu'à 90 % sur du bruit | refaire chaque étape dans chaque tour, sur la partie d'entraînement (un `Pipeline`) (8.25) |
| un plus proche voisin excellent en validation croisée sur une série temporelle | les mois mélangés ont leurs voisins dans l'entraînement : le modèle recopie le mois voisin | `TimeSeriesSplit`, ou des groupes de périodes entières (8.19) |
| « B fait 0,98, A 0,95 : B est meilleur » | sur 100 exemples, l'écart peut venir du hasard ; seuls les désaccords comptent | un test apparié (permutation, McNemar) et un intervalle bootstrap apparié (8.26) |
| diviser l'écart-type des scores de folds par $\sqrt{k}$ pour avoir une erreur type | les scores des folds ne sont pas indépendants (entraînements qui se recouvrent) | présenter la moyenne et la dispersion, sans en tirer un intervalle ; comparer les modèles sur les mêmes folds (fiche du ch. 8, §8.5.1) |
| « collectons plus de données » pour un modèle qui se trompe déjà sur son entraînement | l'underfitting ne vient pas d'un manque d'exemples : erreurs d'entraînement et de validation hautes et proches | plus de capacité (features, degré, couches) ou moins de régularisation ; plus de données soigne surtout l'overfitting (9.Q3, fiche du ch. 9, §9.2.2) |
| le $\lambda$ (ou l'`alpha`) qui minimise l'erreur d'entraînement vaut toujours 0 | sans pénalité, l'erreur d'entraînement est la plus basse possible | choisir $\lambda$ sur la validation ou par validation croisée (9.Q7) |
| l'early stopping rend un modèle moins bon que le meilleur vu pendant l'entraînement | on a gardé les poids de la **dernière** epoch, pas ceux de la meilleure | sauvegarder les poids à chaque amélioration de la validation et les recharger à l'arrêt (9.5) |
| l'entraînement s'arrête au premier soubresaut de la courbe de validation | patience trop petite pour une courbe bruitée | une patience de quelques epochs ; pas de `min_delta`, qui fait s'arrêter encore plus tôt (9.Q6, 9.5) |
| l'entraînement ne s'arrête presque jamais, alors que la validation plafonne | avec `min_delta = 0`, la moindre baisse due au bruit remet l'attente à zéro | un petit `min_delta` : une baisse plus faible ne compte plus comme une amélioration (9.5) |
| un R² négatif sur le test, pris pour un bug | le modèle prédit moins bien que la simple moyenne des cibles | ce n'est pas un bug : le modèle est mauvais sur ces données (9.1) |
| « ce modèle a un biais élevé », dit d'une seule courbe ajustée | le biais et la variance décrivent une **famille** de modèles, entraînés sur beaucoup de jeux tirés de la même source | simuler des jeux (ou rééchantillonner) et mesurer la dispersion des modèles, c'est-à-dire la variance ; le biais, lui, ne se mesure que si l'on connaît la courbe idéale (simulation) (9.Q9) |
| un polynôme de haut degré prédit des valeurs folles aux bords des données | peu de points retiennent la courbe aux extrémités de l'intervalle | un degré plus faible ou une pénalité (Ridge) ; ne pas extrapoler (9.Q10) |
| la loss de validation reste sous la loss d'entraînement | dropout ou augmentation actifs seulement pendant l'entraînement, ou validation plus facile (cas plus nets, moins de labels faux) | mesurer aussi la loss d'entraînement en mode évaluation (dropout éteint), puis comparer la difficulté des deux jeux (9.9) |
| un point gênant retiré des données, et des résultats soudain bien meilleurs | le point était peut-être une vraie mesure, rare : l'écarter fausse la conclusion | chercher d'où il vient ; montrer les résultats avec et sans lui (9.10) |
