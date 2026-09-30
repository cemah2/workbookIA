# Glossaire français ↔ anglais

Règle du workbook (BIBLE §5) : on garde le terme anglais quand c'est l'usage professionnel. À sa première apparition dans un chapitre, un terme est écrit « terme retenu (*autre langue*) », puis seul le terme retenu est utilisé. Chaque nouveau terme d'un chapitre est ajouté ici.

**Colonne « Retenu »** : le terme utilisé dans le workbook. **Ch.** : chapitre où il est introduit (complété au fil de la génération).

## Termes gardés en anglais

| Retenu | En français | Définition courte | Ch. |
|---|---|---|---|
| dataset | jeu de données | ensemble d'exemples utilisés pour entraîner ou évaluer un modèle | 0A |
| feature | caractéristique, variable explicative | une information mesurée sur chaque exemple (ex. la longueur du bec d'un manchot) | 0A |
| label | étiquette | la réponse attendue pour un exemple (ex. l'espèce du manchot) | 0A |
| batch | lot | groupe d'exemples traités ensemble en une étape de calcul | 0A |
| mini-batch | mini-lot | petit lot (quelques dizaines d'exemples) utilisé à chaque mise à jour des poids | 0A |
| epoch | époque | un passage complet sur tout le dataset d'entraînement | 0A |
| learning rate | taux d'apprentissage | taille des pas faits à chaque mise à jour des poids | |
| loss | perte, fonction de coût | nombre qui mesure à quel point les prédictions sont mauvaises ; on cherche à le minimiser | |
| pipeline | chaîne de traitement | suite d'étapes (préparation, modèle…) enchaînées automatiquement | |
| framework | cadriciel | bibliothèque qui fournit la structure d'un programme (ex. PyTorch) | |
| fine-tuning | réglage fin, affinage | réentraîner un peu un modèle déjà entraîné sur une nouvelle tâche | |
| embedding | plongement, représentation vectorielle | vecteur de nombres qui représente un objet (mot, image…) | |
| dropout | abandon | désactiver au hasard des neurones pendant l'entraînement pour limiter l'overfitting | |
| pooling | agrégation, sous-échantillonnage | résumer une zone d'une image par un seul nombre (maximum, moyenne) | |
| padding | remplissage, marge | ajouter des valeurs (souvent des zéros) autour d'une donnée pour garder sa taille | |
| stride | pas | décalage entre deux positions successives d'un filtre de convolution | |
| kernel | noyau, filtre | petite grille de poids que l'on fait glisser sur une image (CNN) | |
| token | jeton, unité lexicale | morceau de texte (mot, sous-mot, caractère) traité par un modèle de langage | |
| prompt | invite, instruction | texte donné en entrée à un modèle de langage | |
| overfitting | surapprentissage, surajustement | le modèle apprend par cœur l'entraînement et généralise mal | |
| underfitting | sous-apprentissage | le modèle est trop simple pour capter la structure des données | |
| benchmark | banc d'essai, référence | jeu de test standard pour comparer des méthodes | |
| notebook | carnet | document qui mêle texte, code exécutable et résultats (Jupyter, Colab) | 0A |
| accuracy | exactitude, taux de bonnes réponses | proportion de prédictions correctes | |
| precision | précision (ambigu) | parmi les exemples prédits positifs, proportion vraiment positive | |
| recall | rappel, sensibilité | parmi les exemples vraiment positifs, proportion retrouvée | |
| F1-score | score F1 | moyenne harmonique de la precision et du recall | |

> ⚠️ **accuracy, precision et recall restent en anglais** : en français, « précision » peut désigner l'une ou l'autre notion.

## Termes dits en français

| Retenu | En anglais | Définition courte | Ch. |
|---|---|---|---|
| réseau de neurones | neural network | modèle fait de couches de neurones artificiels reliés entre eux | |
| couche | layer | ensemble de neurones qui reçoivent les mêmes entrées | |
| poids | weights | coefficients appris qui multiplient les entrées d'un neurone | |
| biais | bias | nombre appris ajouté à la somme pondérée d'un neurone (à ne pas confondre avec un biais statistique) | |
| neurone | neuron, unit | calcule une somme pondérée de ses entrées puis applique une fonction d'activation | |
| fonction d'activation | activation function | fonction non linéaire appliquée à la sortie d'un neurone (ReLU, sigmoïde…) | |
| descente de gradient | gradient descent | méthode qui ajuste les poids par petits pas dans la direction qui fait baisser la loss | |
| rétropropagation | backpropagation | algorithme qui calcule efficacement le gradient de la loss par rapport à tous les poids | |
| entraînement | training | phase où le modèle ajuste ses poids à partir des données | |
| validation croisée | cross-validation | évaluer un modèle en le réentraînant sur plusieurs découpages des données | |
| apprentissage supervisé | supervised learning | apprendre à partir d'exemples étiquetés | |
| apprentissage non supervisé | unsupervised learning | trouver une structure dans des données sans étiquettes | |
| apprentissage par renforcement | reinforcement learning | apprendre par essais et erreurs grâce à des récompenses | |
| matrice de confusion | confusion matrix | tableau qui croise les classes réelles et les classes prédites | |

## Termes ajoutés au fil des chapitres

| Retenu | Autre langue | Définition courte | Ch. |
|---|---|---|---|
| noyau | kernel | le programme qui exécute les cellules d'un notebook et garde les variables en mémoire (à ne pas confondre avec le *kernel* d'une convolution) | 0A |
| terminal | terminal, shell | fenêtre où l'on tape des commandes (`cd`, `python`, `git`) | 0A |
| chemin relatif / absolu | relative / absolute path | chemin qui part du dossier courant / de la racine du disque | 0A |
| variable | variable | nom qui désigne une valeur en mémoire | 0A |
| type | type | nature d'une valeur (`int`, `float`, `str`, `bool`, `list`…) | 0A |
| chaîne de caractères | string (`str`) | texte ; les f-strings (`f"{x:.1f}"`) y insèrent des valeurs | 0A |
| flottant | float | nombre à virgule, stocké avec une précision limitée | 0A |
| liste, tuple | list, tuple | séquences ordonnées : modifiable pour la liste, figée pour le tuple | 0A |
| dictionnaire | dictionary (`dict`) | associe des clés à des valeurs | 0A |
| ensemble | set | collection sans ordre ni doublon | 0A |
| mutable / immuable | mutable / immutable | qu'on peut modifier sur place (liste, dict, array) / non (int, str, tuple) | 0A |
| compréhension | comprehension | construction d'une liste, d'un dict ou d'un ensemble en une ligne : `[f(x) for x in xs if ...]` | 0A |
| fonction, paramètre, argument | function, parameter, argument | bloc de code réutilisable ; le paramètre est le nom dans la définition, l'argument la valeur passée à l'appel | 0A |
| portée | scope | zone du code où un nom est visible (locale à une fonction, ou globale) | 0A |
| fermeture | closure | fonction qui se souvient des variables de la fonction qui l'a créée | 0A |
| docstring | docstring | texte de documentation placé au début d'une fonction ou d'une classe | 0A |
| annotation de type | type hint | indication du type attendu : `def f(x: float) -> str` | 0A |
| exception | exception | erreur signalée par `raise`, rattrapée par `try` / `except` | 0A |
| pile d'appels | traceback | message qui liste les appels en cours au moment d'une erreur (lire la dernière ligne d'abord) | 0A |
| module, package | module, package | fichier `.py` / dossier de modules qu'on importe | 0A |
| classe, objet, méthode, attribut | class, object, method, attribute | modèle d'objets ; un objet ; une fonction d'un objet ; une valeur d'un objet | 0A |
| héritage | inheritance | une classe fille reprend et spécialise une classe mère (`super()`) | 0A |
| générateur | generator | fonction avec `yield`, qui produit ses valeurs une par une, à la demande | 0A |
| sérialiser | serialize | enregistrer un objet dans un fichier (JSON, pickle) pour le relire plus tard | 0A |
| expression régulière | regular expression (regex) | motif qui décrit du texte à chercher (`re.findall`) | 0A |
| array | tableau NumPy (`ndarray`) | tableau de nombres d'un seul type, de forme quelconque | 0A |
| forme | shape | tuple des tailles de chaque dimension d'un array : `(333, 4)` | 0A |
| axe | axis | une des dimensions d'un array ; `axis=0` = les lignes | 0A |
| vectorisation | vectorization | calculer sur un tableau entier plutôt qu'avec une boucle | 0A |
| broadcasting | diffusion | règle qui étire automatiquement des arrays de formes compatibles | 0A |
| vue / copie | view / copy | array qui partage la mémoire d'un autre / array indépendant | 0A |
| masque booléen | boolean mask | tableau de `True`/`False` qui sélectionne des éléments | 0A |
| graine | seed | nombre qui fixe la suite produite par un générateur aléatoire (reproductibilité) | 0A |
| DataFrame, Series | DataFrame, Series | tableau pandas à colonnes nommées ; une colonne | 0A |
| valeur manquante | missing value (NaN, `None`) | mesure absente ; NaN = *Not a Number* | 0A |
| one-hot | encodage one-hot, codage disjonctif | vecteur de 0 avec un seul 1, à la position de la classe | 0A |
| dépôt | repository (repo) | dossier suivi par git, avec tout son historique | 0A |
| commit | commit | photo enregistrée de fichiers, avec un message | 0A |
| branche | branch | ligne d'historique parallèle, fusionnée ensuite (`merge`) | 0A |
| pull request | demande de fusion | proposition de fusionner une branche, relue avant d'être acceptée (GitHub) | 0A |
| test unitaire | unit test | petite fonction qui vérifie automatiquement un comportement (pytest) | 0A |
| oracle | oracle (test) | implémentation de confiance (NumPy, scikit-learn, PyTorch) à laquelle on compare son code | 0A |
| stub | squelette | fonction dont seule la signature et la docstring sont écrites, à compléter | 0A |
| ordre de grandeur | order of magnitude | la puissance de 10 la plus proche d'un nombre : $2^{30} \approx 10^9$ | 0B |
| valeur absolue | absolute value | distance d'un nombre à 0 : $\lvert -3 \rvert = 3$ | 0B |
| partie entière (plancher / plafond) | floor / ceiling | plus grand entier $\le x$ / plus petit entier $\ge x$ : $\lfloor 2{,}7 \rfloor = 2$, $\lceil 2{,}1 \rceil = 3$, $\lceil 4 \rceil = 4$ | 0B |
| moyenne pondérée | weighted average | moyenne où chaque valeur compte selon son poids : $\frac{\sum_i w_i x_i}{\sum_i w_i}$ | 0B |
| moyenne mobile | moving average | moyenne des $k$ dernières valeurs d'une série, recalculée à chaque pas | 0B |
| suite géométrique, raison | geometric sequence, ratio | suite où l'on multiplie toujours par le même nombre $q$ : $u_n = u_0\,q^n$ | 0B |
| factorielle | factorial | $n! = n \times (n - 1) \times \dots \times 1$ : les façons de ranger $n$ objets | 0B |
| coefficient binomial | binomial coefficient, « n choose k » | $\binom{n}{k}$ : les façons de choisir $k$ objets parmi $n$, sans ordre | 0B |
| cardinal | cardinality | nombre d'éléments d'un ensemble fini : $\lvert A \rvert$ | 0B |
| logarithme népérien | natural logarithm | $\ln$, le logarithme de base $e$ ; `np.log` en NumPy | 0B |
| sigmoïde | sigmoid, logistic function | $\sigma(x) = \frac{1}{1 + e^{-x}}$, courbe en S de 0 à 1 | 0B |
| tangente hyperbolique | hyperbolic tangent (tanh) | courbe en S de −1 à 1 : $\tanh(x) = 2\sigma(2x) - 1$ | 0B |
| logit | logit | inverse de la sigmoïde : $\ln\frac{p}{1 - p}$ ; le score avant la sigmoïde | 0B |
| planning en cosinus | cosine schedule, cosine annealing | learning rate qui décroît comme une demi-période de cosinus | 0B |
| composante | component, entry | un des nombres d'un vecteur | 0B |
| norme | norm | longueur d'un vecteur : $\lVert \mathbf{x} \rVert_2 = \sqrt{\sum_i x_i^2}$ (aussi L1, L∞) | 0B |
| produit scalaire | dot product, inner product | $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$, un nombre | 0B |
| similarité cosinus | cosine similarity | cosinus de l'angle entre deux vecteurs, entre −1 et 1 ; ignore leur longueur | 0B |
| orthogonaux | orthogonal | de produit scalaire nul (perpendiculaires) | 0B |
| produit de Hadamard | element-wise product, Hadamard product | $\mathbf{a} \odot \mathbf{b}$ : produit composante par composante (`a * b`) | 0B |
| transposée | transpose | matrice dont les lignes sont les colonnes de la matrice de départ ($\mathbf{A}^\top$, `A.T`) | 0B |
| produit matriciel | matrix multiplication (matmul) | $(\mathbf{A}\mathbf{B})_{ij}$ = ligne $i$ de $\mathbf{A}$ · colonne $j$ de $\mathbf{B}$ (`A @ B`) | 0B |
| matrice identité | identity matrix | $\mathbf{I}$ : des 1 sur la diagonale, 0 ailleurs ; $\mathbf{A}\mathbf{I} = \mathbf{A}$ | 0B |
| inverse, déterminant | inverse, determinant | $\mathbf{M}^{-1}\mathbf{M} = \mathbf{I}$ ; existe si et seulement si $\det \mathbf{M} \neq 0$ | 0B |
| taux d'accroissement | difference quotient | $\frac{f(a + h) - f(a)}{h}$ : pente de la sécante | 0B |
| dérivée | derivative | pente de la tangente : $f'(a)$ | 0B |
| règle de la chaîne | chain rule | $(g \circ f)' = (g' \circ f) \times f'$ : on multiplie les dérivées des étapes | 0B |
| dérivée partielle | partial derivative | dérivée par rapport à une variable, les autres étant fixées : $\frac{\partial f}{\partial x}$ | 0B |
| gradient | gradient | vecteur des dérivées partielles ; direction de plus forte montée | 0B |
| ligne de niveau | level curve, contour line | points où une fonction de deux variables garde la même valeur | 0B |
| graphe de calcul | computational graph | schéma des étapes d'un calcul ; la dérivée est la somme sur les chemins | 0B |
| événement, issue | event, outcome | une issue est un résultat possible ; un événement, un ensemble d'issues | 0B |
| indépendance | independence | $P(A \cap B) = P(A)\,P(B)$ : savoir que l'un est arrivé ne change rien à l'autre | 0B |
| variable aléatoire | random variable | nombre qui dépend du résultat d'une expérience aléatoire | 0B |
| espérance | expected value, expectation | moyenne des valeurs pondérée par leurs probabilités : $\mathbb{E}[X]$ | 0B |
| variance, écart-type | variance, standard deviation | dispersion autour de l'espérance ; l'écart-type est sa racine | 0B |
| loi des grands nombres | law of large numbers | la fréquence observée tend vers la probabilité quand on répète l'expérience | 0B |
| baseline | modèle de référence | le modèle le plus simple (par exemple, prédire toujours la moyenne), à battre | 0B |
| log-probabilité, vraisemblance | log-probability, likelihood | logarithme d'une probabilité ; probabilité des données selon le modèle | 0B |
| sous-dépassement | underflow | nombre trop petit pour un `float`, arrondi à 0 | 0B |
