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

## Maths et algèbre linéaire

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0` | `A @ B` avec un nombre de colonnes de `A` différent du nombre de lignes de `B` | écris les formes : `(m, n) @ (n, p)` ; transpose si besoin (`A @ B.T`) (0B.20) |
| un vecteur au lieu d'un nombre, ou des valeurs fausses sans message d'erreur | `*` (élément par élément) confondu avec `@` (produit scalaire ou matriciel) | `*` : Hadamard ; `@` : produit scalaire ou matriciel (0B.18) |
| `v.T` ne change rien | un vecteur `(n,)` n'a qu'un axe : sa transposée est lui-même | `v.reshape(-1, 1)` pour une colonne `(n, 1)` ; vérifie `.shape` |
| `RuntimeWarning: divide by zero encountered in log`, résultat `-inf` | `np.log(0)` : une probabilité nulle | ajoute un petit `eps` (`np.log(p + 1e-12)`) ou travaille en log-probabilités (0B.15) |
| `RuntimeWarning: invalid value encountered in log`, résultat `nan` | logarithme d'un nombre négatif | vérifie le signe des entrées avant le `log` |
| `RuntimeWarning: overflow encountered in exp`, résultat `inf` | `np.exp` d'un grand nombre (au-delà de 709 environ en `float64`) | réécris la formule (par exemple la sigmoïde avec `np.exp(-abs(x))`) ou utilise `scipy.special.expit` |
| un produit de probabilités vaut `0.0` | sous-dépassement (*underflow*) : le produit est trop petit pour un `float64` (sous $10^{-308}$ environ, il perd des chiffres ; sous $5 \times 10^{-324}$, il devient 0) | additionne les logarithmes au lieu de multiplier (0B.15, 0B.E3) |
| `RuntimeWarning: overflow encountered in reduce` et un résultat `inf` | `np.prod` de beaucoup de grands nombres | passe par les logarithmes : `np.exp(np.log(x).sum())`, ou garde le résultat en logarithme (0B.37) |
| `u @ v.T` donne un nombre au lieu d'une matrice | `.T` ne change rien à un vecteur `(n,)` : c'est un produit scalaire | `u[:, None] @ v[None, :]` pour le produit extérieur (0B.45) |
| `np.log(100)` donne 4,6 au lieu de 2 | `np.log` est le logarithme **népérien** | `np.log10` ou `np.log2` selon la base voulue |
| `np.linalg.LinAlgError: Singular matrix` | la matrice n'est pas inversible (déterminant nul : une ligne proportionnelle à une autre) | vérifie les données (colonne dupliquée ?) ; `np.linalg.lstsq` pour un système sans solution unique (0B.21) |
| une somme « de 1 à n » est fausse d'un terme | `range(1, n)` s'arrête à `n - 1` | `range(1, n + 1)` : la borne haute d'un $\Sigma$ est incluse (0B.R1) |
| `math.floor(-3.7)` et `int(-3.7)` ne donnent pas la même chose | `floor` va vers le bas (−4), `int` tronque vers zéro (−3) | choisis selon le sens voulu (0B.2) |
| des distances dominées par une seule feature | features dans des unités très différentes (grammes et millimètres) | standardise les colonnes avant de calculer des distances (0B.8, 0A.52) |

## pandas

| Message / symptôme | Cause probable | Solution |
|---|---|---|
| `KeyError: 'body_mass'` | nom de colonne mal tapé | `df.columns` ; copie le nom exact (`body_mass_g`) |
| `SettingWithCopyWarning` ou `FutureWarning: ChainedAssignmentError` ; selon les cas, rien ne change | affectation en chaîne : `df[masque]["col"] = v` ou `df["col"][masque] = v` | `df.loc[masque, "col"] = v` |
| `ValueError: The truth value of a Series is ambiguous` | `and` / `or` entre deux conditions | `(cond1) & (cond2)`, `(cond1) \| (cond2)` |
| `y` de forme `(n, 1)` au lieu de `(n,)` | `df[["species"]]` (doubles crochets) renvoie un DataFrame | `df["species"].to_numpy()` (0A.35) |
| moyenne `NaN` pour un groupe | toutes les valeurs du groupe sont manquantes | `groupby(...).agg(["count", "mean"])` pour voir les effectifs |
| moins de lignes que prévu après `dropna()` | `dropna()` retire toute ligne avec **au moins une** valeur manquante | `dropna(subset=[...])` si seules certaines colonnes comptent |

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
| `roc_auc_score` renvoie une valeur sous 0,5 | scores inversés (probabilité de la mauvaise classe, `predict_proba(X)[:, 0]` au lieu de `[:, 1]`) ou étiquettes inversées | passer la probabilité de la classe positive, `model.predict_proba(X)[:, 1]` (3.24) |
| la courbe ROC ou PR ne va pas jusqu'au bout, ou a trop peu de points | `roc_curve` retire par défaut des points inutiles au dessin (`drop_intermediate=True`) ; on a passé des classes prédites au lieu de scores | passer des **scores** (`predict_proba`, `decision_function`), pas `predict` ; `drop_intermediate=False` pour tous les seuils (3.24) |

## PyTorch
*(à compléter à partir du ch. 20)*

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
