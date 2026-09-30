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
*(à compléter)*

## PyTorch
*(à compléter à partir du ch. 20)*

## Erreurs de raisonnement (ML)
*(fuite de données, évaluation sur l'entraînement, classes déséquilibrées… : à compléter)*
