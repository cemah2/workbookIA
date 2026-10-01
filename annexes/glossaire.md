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
| learning rate | taux d'apprentissage | taille des pas faits à chaque mise à jour des poids | 0B |
| loss | perte, fonction de coût | nombre qui mesure à quel point les prédictions sont mauvaises ; on cherche à le minimiser | 0B |
| pipeline | chaîne de traitement | suite d'étapes (préparation, modèle…) enchaînées automatiquement | |
| framework | cadriciel | bibliothèque qui fournit la structure d'un programme (ex. PyTorch) | |
| fine-tuning | réglage fin, affinage | réentraîner un peu un modèle déjà entraîné sur une nouvelle tâche | 1 |
| embedding | plongement, représentation vectorielle | vecteur de nombres qui représente un objet (mot, image…) | |
| dropout | abandon | désactiver au hasard des neurones pendant l'entraînement pour limiter l'overfitting | |
| pooling | agrégation, sous-échantillonnage | résumer une zone d'une image par un seul nombre (maximum, moyenne) | |
| padding | remplissage, marge | ajouter des valeurs (souvent des zéros) autour d'une donnée pour garder sa taille | |
| stride | pas | décalage entre deux positions successives d'un filtre de convolution | |
| kernel | noyau, filtre | petite grille de poids que l'on fait glisser sur une image (CNN) | |
| token | jeton, unité lexicale | morceau de texte (mot, sous-mot, caractère) traité par un modèle de langage | 1 |
| prompt | invite, instruction | texte donné en entrée à un modèle de langage | 1 |
| overfitting | surapprentissage, surajustement | le modèle apprend par cœur l'entraînement et généralise mal | |
| underfitting | sous-apprentissage | le modèle est trop simple pour capter la structure des données | |
| benchmark | banc d'essai, référence | jeu de test standard pour comparer des méthodes | 1 |
| notebook | carnet | document qui mêle texte, code exécutable et résultats (Jupyter, Colab) | 0A |
| accuracy | exactitude, taux de bonnes réponses | proportion de prédictions correctes | 1 |
| precision | précision (ambigu), valeur prédictive positive (VPP) | parmi les exemples prédits positifs, proportion vraiment positive : $\frac{TP}{TP + FP}$ | 3 |
| recall | rappel, sensibilité, taux de vrais positifs (TPR) | parmi les exemples vraiment positifs, proportion retrouvée : $\frac{TP}{TP + FN}$ | 3 |
| F1-score | score F1 | moyenne harmonique de la precision et du recall : $\frac{2\,TP}{2\,TP + FP + FN}$ | 3 |
| underflow | sous-dépassement, dépassement par le bas | nombre trop petit pour un `float`, arrondi à 0 sans erreur ni avertissement ; on l'évite en additionnant des logarithmes | 0B |
| prior | a priori, loi a priori | ce qu'on croit des hypothèses **avant** les nouvelles données : la distribution $P(H)$ | 4 |
| posterior | a posteriori, loi a posteriori | ce qu'on croit **après** l'observation : $P(H \mid O)$ ; il sert de prior à l'observation suivante | 4 |
| MAP | maximum a posteriori | l'hypothèse (la valeur du paramètre) de plus grand posterior ; avec un prior uniforme, c'est le maximum de vraisemblance | 4 |
| log-sum-exp | astuce du log-somme-exp | $\log\sum_j e^{\ell_j} = m + \log\sum_j e^{\ell_j - m}$, avec $m = \max_j \ell_j$ : normaliser des log-probabilités sans underflow (`scipy.special.logsumexp`) | 4 |

> ⚠️ **accuracy, precision et recall restent en anglais** : en français, « précision » peut désigner l'une ou l'autre notion.

## Termes dits en français

| Retenu | En anglais | Définition courte | Ch. |
|---|---|---|---|
| réseau de neurones | neural network | modèle fait de couches de neurones artificiels reliés entre eux | 1 |
| couche | layer | ensemble de neurones qui reçoivent les mêmes entrées | 1 |
| poids | weights | coefficients appris qui multiplient les entrées d'un neurone | 0B |
| biais | bias | nombre appris ajouté à la somme pondérée d'un neurone (à ne pas confondre avec un biais statistique) | 0B |
| neurone | neuron, unit | calcule une somme pondérée de ses entrées puis applique une fonction d'activation | 1 |
| fonction d'activation | activation function | fonction non linéaire appliquée à la sortie d'un neurone (ReLU, sigmoïde…) | |
| descente de gradient | gradient descent | méthode qui ajuste les poids par petits pas dans la direction qui fait baisser la loss | |
| rétropropagation | backpropagation | algorithme qui calcule efficacement le gradient de la loss par rapport à tous les poids | |
| entraînement | training | phase où le modèle ajuste ses poids à partir des données | 1 |
| validation croisée | cross-validation | évaluer un modèle en le réentraînant sur plusieurs découpages des données | |
| apprentissage supervisé | supervised learning | apprendre à partir d'exemples étiquetés | 1 |
| apprentissage non supervisé | unsupervised learning | trouver une structure dans des données sans étiquettes | 1 |
| apprentissage par renforcement | reinforcement learning | apprendre par essais et erreurs grâce à des récompenses | 1 |
| matrice de confusion | confusion matrix | tableau qui croise les classes réelles et les classes prédites ; scikit-learn : vérité en lignes, étiquettes triées | 3 |

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
| pente centrée | central difference | estimation de $f'(a)$ par $\frac{f(a + h) - f(a - h)}{2h}$ | 0B |
| gradient checking | vérification du gradient | comparer une dérivée calculée (à la main ou par rétropropagation) à une pente numérique | 0B |
| produit extérieur | outer product | $\mathbf{u}\mathbf{v}^\top$ : la matrice de tous les produits $u_i v_j$ | 0B |
| sac de mots | bag of words | représentation d'un texte par le nombre d'occurrences de chaque mot d'un vocabulaire | 0B |
| échantillon | sample | une observation du dataset : une ligne du tableau (un manchot, une image) | 1 |
| système expert | expert system | programme qui applique des règles écrites à la main d'après des experts humains | 1 |
| feature engineering | ingénierie des features | fabriquer à la main les indices (features, règles) qu'un programme doit surveiller | 1 |
| feature learning | apprentissage des features | un réseau profond apprend lui-même les features utiles, au lieu qu'on les fabrique | 1 |
| paramètre (d'un modèle) | parameter | valeur apprise par l'algorithme pendant l'entraînement (un poids, un seuil) | 1 |
| hyperparamètre | hyperparameter | réglage choisi avant l'entraînement (learning rate, nombre d'epochs, `max_depth`) | 1 |
| modèle | model | la structure d'un programme plus les valeurs de ses paramètres : une représentation simplifiée des données | 1 |
| capacité | capacity, representational power | richesse de ce qu'un modèle peut représenter | 1 |
| généralisation | generalization | le fait de réussir sur des données nouvelles, pas seulement sur les exemples appris | 1 |
| jeu d'entraînement / jeu de test | training set / test set | les exemples qui servent à apprendre / ceux mis de côté pour mesurer la généralisation | 1 |
| déployer | deploy | mettre un modèle en service pour de vrais utilisateurs | 1 |
| classifieur, classe | classifier, class | modèle qui range chaque échantillon dans une catégorie (une classe) d'une liste connue | 1 |
| classification / régression | classification / regression | prédire une catégorie / prédire une quantité (un nombre) | 1 |
| régression vers la moyenne | regression to the mean | une valeur extrême est en moyenne suivie d'une valeur moins extrême (Galton, 1886) | 1 |
| taux d'erreur | error rate | proportion de prédictions fausses : $1 - \text{accuracy}$ | 1 |
| interpolation linéaire | linear interpolation | estimer une valeur manquante sur la droite qui relie ses deux voisines | 1 |
| clustering | partitionnement, regroupement | former des groupes d'échantillons qui se ressemblent, sans labels | 1 |
| débruitage | denoising, noise reduction | retirer le bruit d'un échantillon, ou combler ses valeurs manquantes | 1 |
| réduction de dimension | dimensionality reduction | décrire les échantillons avec moins de features, en gardant l'essentiel | 1 |
| générateur | generator, generative model | modèle qui fabrique de nouvelles données ressemblant aux exemples | 1 |
| apprentissage auto-supervisé | self-supervised learning | les données fournissent elles-mêmes la réponse (le mot suivant, une zone masquée) : pas de labels humains | 1 |
| apprentissage semi-supervisé | semi-supervised learning | apprendre avec peu d'exemples étiquetés et beaucoup de non étiquetés | 1 |
| agent, environnement, action, récompense | agent, environment, action, reward | vocabulaire du renforcement : qui décide ; le reste du monde ; son choix ; le nombre qui évalue ce choix | 1 |
| exploration / exploitation | exploration / exploitation | essayer d'autres actions / refaire la meilleure connue : le compromis du renforcement | 1 |
| bandit manchot | multi-armed bandit | choisir sans cesse entre plusieurs options au gain inconnu (le cuisinier de 1.22, ch. 11) | 1 |
| deep learning | apprentissage profond | construire des modèles en couches de neurones empilées, qui apprennent leurs propres features | 1 |
| couche pleine | dense layer, fully connected layer | couche dont chaque neurone reçoit toutes les valeurs de la couche précédente | 1 |
| GPU, TPU, NPU | GPU, TPU, NPU | processeurs spécialisés qui font des milliers de calculs en parallèle (graphique, tensoriel, neuronal) | 1 |
| fuite de données | data leakage | une information sur la réponse qui se glisse dans les features ou l'entraînement : le score devient trop beau | 1 |
| data card | fiche de données | fiche d'un dataset : provenance, licence, taille, variables, biais et limites | 1 |
| bigramme | bigram | paire de symboles consécutifs (deux caractères, deux mots) | 1 |
| LLM | grand modèle de langage (large language model) | très grand réseau pré-entraîné à prédire le token suivant sur d'immenses textes, puis aligné | 1 |
| alignement, RLHF | alignment, reinforcement learning from human feedback | ajuster un LLM sur des préférences humaines, par renforcement ou des méthodes voisines (DPO) | 1 |
| foundation model | modèle de fondation | modèle géant entraîné une fois sur des données très variées, puis réutilisé pour de nombreuses tâches | 1 |
| modèle de diffusion | diffusion model | générateur qui apprend à reconstruire une image à partir d'une version bruitée | 1 |
| machine learning (ML) | apprentissage automatique | ensemble des méthodes qui apprennent à partir d'exemples au lieu d'appliquer des règles écrites à la main | 1 |
| jeu de validation | validation set | exemples mis de côté pour comparer des modèles et régler les hyperparamètres, sans toucher au jeu de test | 1 |
| arbre de décision | decision tree | modèle qui pose une suite de questions du type « bec ≤ 40 mm ? » et en déduit une classe (ch. 13) | 1 |
| k-means | k-moyennes | algorithme de clustering qui répartit les échantillons en $k$ groupes autour de $k$ centres (ch. 7) | 1 |
| CPU | processeur central (Central Processing Unit) | le processeur principal d'un ordinateur, polyvalent mais peu parallèle | 1 |
| pré-entraînement | pre-training | premier entraînement, long et général (souvent auto-supervisé), avant un ajustement à une tâche précise | 1 |
| Transformer | Transformer | architecture de réseau fondée sur un mécanisme d'attention, à la base des LLM (2017, bonus B3) | 1 |
| RAG | retrieval-augmented generation, génération augmentée par recherche | donner à un LLM des documents trouvés par une recherche, à consulter avant de répondre (bonus B4) | 1 |
| LoRA | low-rank adaptation | méthode de fine-tuning léger qui n'entraîne que de petites matrices ajoutées au modèle (bonus B4) | 1 |
| hallucination | hallucination | affirmation fausse produite avec aplomb par un modèle génératif | 1 |
| moyenne, médiane, mode | mean, median, mode | le centre d'une liste : somme divisée par l'effectif ; valeur du milieu une fois triée ; valeur la plus fréquente | 2 |
| robuste | robust | se dit d'une statistique peu sensible aux valeurs extrêmes (la médiane l'est, pas la moyenne) | 2 |
| distribution de probabilité, loi | probability distribution | façon de répartir une probabilité totale de 1 entre les valeurs possibles | 2 |
| normaliser (une distribution) | normalize | diviser par le total pour obtenir une somme de 1 (au ch. 12, le mot désigne aussi une mise à l'échelle) | 2 |
| fonction de masse (pmf) | probability mass function | loi discrète : la probabilité de chaque valeur | 2 |
| densité (pdf) | probability density function | loi continue : courbe dont l'aire entre deux bornes est une probabilité ; elle peut dépasser 1 | 2 |
| tirage | draw, sample | produire une valeur au hasard selon une loi | 2 |
| pseudo-aléatoire, générateur | pseudo-random, pseudo-random number generator (PRNG) | suite calculée qui imite le hasard ; même graine, même suite | 2 |
| loi uniforme | uniform distribution | toutes les valeurs d'un intervalle (ou d'une liste) ont la même chance | 2 |
| loi normale, gaussienne | normal distribution, Gaussian distribution | la « courbe en cloche », définie par sa moyenne $\mu$ et son écart-type $\sigma$ | 2 |
| règle 68-95-99,7 | 68-95-99.7 rule, three-sigma rule | pour une loi normale, part des valeurs à moins de 1, 2 et 3 écarts-types de la moyenne | 2 |
| loi de Bernoulli | Bernoulli distribution | variable qui vaut 1 avec la probabilité $p$, 0 sinon | 2 |
| loi catégorielle, multinoulli | categorical distribution | une issue parmi $K$, chacune avec sa probabilité ; codée par un indice ou un vecteur one-hot | 2 |
| ddof | delta degrees of freedom | le diviseur d'une variance est $n - \mathrm{ddof}$ : 0 divise par $n$, 1 par $n - 1$ | 2 |
| percentile, quantile, quartile | percentile, quantile, quartile | valeur sous laquelle se trouve une part donnée des données (quartiles : 25 %, 50 %, 75 %) | 2 |
| z-score, standardiser | z-score, standardize | nombre d'écarts-types entre une valeur et la moyenne ; standardiser, c'est remplacer chaque valeur par son z-score | 2 |
| i.i.d. | independent and identically distributed | des variables indépendantes qui suivent toutes la même loi | 2 |
| tirage avec remise, sans remise | sampling with replacement, without replacement | l'élément tiré peut ressortir / sort au plus une fois | 2 |
| population, échantillon (en statistique) | population, sample | l'ensemble qu'on veut décrire / les observations qu'on a mesurées (au ch. 1, « échantillon » désigne une seule ligne) | 2 |
| bootstrap, rééchantillon | bootstrap, bootstrap sample | tirer $n$ éléments parmi $n$ avec remise, pour mesurer la variabilité d'une statistique | 2 |
| intervalle de confiance | confidence interval | intervalle construit par une méthode qui contient la vraie valeur dans, par exemple, 95 % des cas | 2 |
| biais d'échantillonnage | sampling bias, selection bias | échantillon qui ne ressemble pas à la population visée ; aucun calcul ne le corrige | 2 |
| fléau de la dimension | curse of dimensionality | en grande dimension, les données sont toujours clairsemées et les distances trompeuses | 2 |
| histogramme | histogram | barres qui comptent les valeurs tombant dans des intervalles de même largeur | 2 |
| nuage de points | scatter plot | graphique avec un point par individu et une variable par axe | 2 |
| covariance | covariance | moyenne des produits des écarts à la moyenne de deux variables : leur tendance à varier ensemble | 2 |
| corrélation (de Pearson) | Pearson correlation coefficient | covariance divisée par les deux écarts-types : entre −1 et 1, sans unité, mesure le lien linéaire | 2 |
| matrice de covariance, de corrélation | covariance matrix, correlation matrix | tableau des covariances (ou des corrélations) de toutes les paires de colonnes | 2 |
| variable de confusion | confounder, confounding variable | troisième variable qui influence les deux autres et crée une corrélation sans causalité | 2 |
| analyse exploratoire | exploratory data analysis (EDA) | regarder les données (histogrammes, nuages de points, statistiques) avant de les modéliser | 2 |
| quartet d'Anscombe, Datasaurus | Anscombe's quartet, Datasaurus Dozen | jeux de points aux statistiques identiques mais aux nuages très différents | 2 |
| erreur type | standard error | écart-type d'une statistique d'un échantillon à l'autre ; pour une moyenne, $\sigma/\sqrt{n}$ ; le bootstrap l'estime avec un seul échantillon | 2 |
| couverture (d'un intervalle) | coverage | part des répétitions où l'intervalle de confiance contient la vraie valeur ; elle devrait égaler le niveau annoncé (95 %) | 2 |
| bruit de Monte-Carlo | Monte Carlo error | variation d'un résultat obtenu par tirages au hasard ; elle diminue quand on fait plus de tirages | 2 |
| BCa | bias-corrected and accelerated bootstrap | intervalle bootstrap qui corrige les percentiles du biais et de l'asymétrie ; méthode par défaut de `scipy.stats.bootstrap` | 2 |
| contraste des distances | relative contrast | écart relatif $(d_{\max} - d_{\min})/d_{\min}$ entre la plus grande et la plus petite distance d'un point aux autres ; il s'effondre en grande dimension | 2 |
| paradoxe de Simpson | Simpson's paradox | une corrélation qui change de signe quand on sépare les données en groupes (longueur et profondeur du bec des manchots : corrélation négative sur l'ensemble, positive dans chaque espèce) | 2 |
| point influent | influential point | point isolé qui, à lui seul, déplace beaucoup une droite ajustée ou une corrélation | 2 |
| probabilité conditionnelle | conditional probability | $P(A \mid B) = \frac{P(A, B)}{P(B)}$ : probabilité de A quand on sait déjà que B s'est produit | 3 |
| probabilité jointe | joint probability | $P(A, B)$ : probabilité que A et B se produisent tous les deux | 3 |
| probabilité simple, marginale | simple probability, marginal probability | probabilité d'un seul événement ; « marginale » parce qu'elle se lit dans les marges (totaux) d'une table de contingence | 3 |
| règle du produit | product rule | $P(A, B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$ | 3 |
| formule des probabilités totales | law of total probability | $P(A) = \sum_b P(A \mid B = b)\,P(B = b)$ | 3 |
| table de contingence | contingency table, cross-tabulation | comptages croisés de deux variables catégorielles (`pd.crosstab`) | 3 |
| vérité terrain | ground truth | l'étiquette qu'on tient pour correcte (elle peut contenir des erreurs), à laquelle on compare les prédictions | 3 |
| classe positive, négative | positive class, negative class | la classe qu'on cherche à détecter (spam, maladie), et l'autre ; pas un jugement de valeur | 3 |
| frontière de décision | decision boundary | limite, dans l'espace des features, entre les régions prédites positives et négatives | 3 |
| vrai positif, faux positif, faux négatif, vrai négatif | true positive (TP), false positive (FP), false negative (FN), true negative (TN) | les quatre cases d'une matrice de confusion binaire ; le second mot est la prédiction | 3 |
| spécificité | specificity, true negative rate (TNR) | parmi les négatifs réels, proportion bien reconnue : $\frac{TN}{TN + FP}$ | 3 |
| valeur prédictive négative (VPN) | negative predictive value (NPV) | parmi les prédictions négatives, proportion juste : $\frac{TN}{TN + FN}$ | 3 |
| taux de faux positifs, de faux négatifs | false positive rate (FPR, fall-out), false negative rate (FNR, miss rate) | $1 - \text{spécificité}$ et $1 - \text{recall}$ | 3 |
| taux de fausses découvertes, de fausses omissions | false discovery rate (FDR), false omission rate (FOR) | $1 - \text{precision}$ et $1 - \text{NPV}$ | 3 |
| prévalence, taux de base | prevalence, base rate | proportion de positifs dans la population | 3 |
| erreur du taux de base | base rate fallacy | oublier la prévalence : croire qu'un test positif « fiable à 99 % » veut dire 99 % de chances d'être malade | 3 |
| fréquences naturelles, arbre des fréquences naturelles | natural frequencies, natural frequency tree | raisonner en effectifs (sur 10 000 personnes…) plutôt qu'en probabilités, avec un arbre | 3 |
| balanced accuracy | accuracy équilibrée | moyenne des recalls de chaque classe ; en binaire, $\frac{\text{recall} + \text{spécificité}}{2}$ | 3 |
| MCC | Matthews correlation coefficient | corrélation entre vérité et prédiction, de −1 à 1, qui utilise les quatre cases | 3 |
| F-beta | F-beta score | moyenne harmonique pondérée : $\beta > 1$ favorise le recall, $\beta < 1$ la precision | 3 |
| moyenne harmonique | harmonic mean | $\frac{2ab}{a + b}$ : inverse de la moyenne des inverses ; colle au plus petit des deux nombres | 3 |
| moyenne macro, micro, pondérée | macro, micro, weighted average | combiner une mesure sur plusieurs classes : moyenne simple, comptages additionnés, ou moyenne pondérée par le support | 3 |
| one-vs-rest (OvR) | un contre tous | chaque classe devient tour à tour la classe positive face à toutes les autres | 3 |
| support (d'une classe) | support | nombre d'exemples réels de cette classe dans les données évaluées | 3 |
| score, seuil de décision | score, decision threshold | nombre donné par un classifieur, et valeur à partir de laquelle on prédit « positif » | 3 |
| courbe ROC | ROC curve (receiver operating characteristic) | taux de vrais positifs en fonction du taux de faux positifs, pour tous les seuils | 3 |
| AUC | area under the (ROC) curve | aire sous la courbe ROC : probabilité qu'un positif tiré au hasard ait un score plus élevé qu'un négatif | 3 |
| courbe precision-recall | precision-recall curve | precision en fonction du recall, pour tous les seuils ; à montrer à côté de la ROC quand les positifs sont rares | 3 |
| average precision (AP) | précision moyenne | aire en escalier sous la courbe precision-recall : $\sum_j (R_j - R_{j-1})\,P_j$ | 3 |
| méthode des trapèzes | trapezoidal rule | aire sous une courbe approchée par des trapèzes entre points consécutifs | 3 |
| calibration, calibré | calibration, calibrated | un modèle est calibré si, parmi les cas annoncés à $p$, une proportion $p$ est positive | 3 |
| diagramme de fiabilité | reliability diagram | fréquence observée des positifs en fonction de la probabilité annoncée, par intervalles | 3 |
| méthode de Monte-Carlo | Monte Carlo method | estimer une quantité (une aire, une probabilité) par la moyenne de nombreux tirages au hasard ; l'erreur typique diminue comme $1/\sqrt{n}$ | 3 |
| test de confirmation | confirmatory test | second test, plus fiable, fait aux seuls positifs d'un dépistage : il augmente la precision | 3 |
| point de fonctionnement | operating point | le seuil (ou le nombre d'alertes) auquel un classifieur sera vraiment utilisé ; le meilleur modèle peut en dépendre | 3 |
| ligne de base d'une courbe precision-recall | PR baseline | precision d'un classifieur au hasard : la prévalence, quel que soit le seuil | 3 |
| score de Brier | Brier score | $\frac{1}{n}\sum_i (p_i - y_i)^2$ : écart quadratique moyen entre probabilité annoncée et résultat 0/1 | 3 |
| fréquentiste, bayésien | frequentist, Bayesian | deux lectures de la probabilité : une fréquence limite sur des expériences répétées, ou un degré de certitude, qui peut porter sur un paramètre inconnu | 4 |
| hypothèse, observation | hypothesis, observation | ce dont on cherche la probabilité (« la pièce est truquée ») et ce qu'on a vu (« face ») ; notées $H$ et $O$ dans la fiche du ch. 4 | 4 |
| règle de Bayes, théorème de Bayes | Bayes' rule, Bayes' theorem | $P(H \mid O) = \frac{P(O \mid H)\,P(H)}{P(O)}$ : passer de la vraisemblance au posterior grâce au prior | 4 |
| vraisemblance (d'une hypothèse) | likelihood | $P(O \mid H)$ : probabilité de l'observation si l'hypothèse est vraie ; les vraisemblances de plusieurs hypothèses n'ont pas à sommer à 1 | 4 |
| évidence | evidence, marginal likelihood | $P(O) = \sum_j P(O \mid H_j)\,P(H_j)$ : probabilité de l'observation, toutes hypothèses confondues ; faux ami, ni une preuve ni une chose évidente | 4 |
| biais (d'une pièce) | bias (of a coin) | probabilité qu'une pièce tombe sur face, notée $\theta$ ; rien à voir avec le biais d'un neurone ni avec un biais d'échantillonnage | 4 |
| mise à jour bayésienne, séquentielle | Bayesian updating, sequential updating | appliquer la règle de Bayes observation après observation : le posterior de l'une devient le prior de la suivante | 4 |
| indépendance conditionnelle | conditional independence | indépendance **sachant** une autre variable : $P(o_1, o_2 \mid H) = P(o_1 \mid H)\,P(o_2 \mid H)$ ; elle n'entraîne pas l'indépendance tout court | 4 |
| cote | odds | $\frac{P(H)}{P(\text{non } H)}$ : une probabilité de 0,8 donne une cote de 4 (« 4 contre 1 ») ; $P = \frac{\text{cote}}{1 + \text{cote}}$ | 4 |
| rapport de vraisemblance | likelihood ratio | $\frac{P(O \mid H_1)}{P(O \mid H_2)}$ : le facteur par lequel l'observation multiplie la cote de $H_1$ contre $H_2$ | 4 |
| erreur du procureur | prosecutor's fallacy | confondre $P(O \mid H)$ et $P(H \mid O)$ : « un innocent a une chance sur un million de correspondre » ne veut pas dire « l'accusé a une chance sur un million d'être innocent » | 4 |
| intervalle de crédibilité | credible interval | intervalle qui contient le paramètre avec une probabilité donnée (95 %), sachant les données ; « à queues égales » s'il laisse la même probabilité de chaque côté | 4 |
| loi Beta | Beta distribution | loi continue sur $[0, 1]$, de densité proportionnelle à $\theta^{a-1}(1 - \theta)^{b-1}$ ; posterior du biais d'une pièce sous un prior uniforme : $\mathrm{Beta}(h + 1, t + 1)$ | 4 |
| prior conjugué | conjugate prior | prior pour lequel le posterior reste dans la même famille de lois : un prior Beta donne un posterior Beta | 4 |
| règle de succession de Laplace | rule of succession | avec un prior uniforme, après $h$ succès en $n$ essais, la probabilité du succès suivant vaut $\frac{h + 1}{n + 2}$ | 4 |
| nombre dénormalisé | subnormal number, denormal number | `float64` positif plus petit que $2{,}2 \times 10^{-308}$ : il perd des chiffres, jusqu'à environ $5 \times 10^{-324}$ ; en dessous, c'est 0 | 4 |
| grille (d'hypothèses) | grid | valeurs régulièrement espacées d'un paramètre, chacune traitée comme une hypothèse ; leur nombre explose avec la dimension | 4 |
| analyse de sensibilité | sensitivity analysis | refaire un calcul avec plusieurs choix raisonnables (plusieurs priors, par exemple) pour vérifier que la conclusion n'en dépend pas trop | 4 |
| test séquentiel, règle d'arrêt | sequential test, stopping rule | décider après chaque observation de s'arrêter ou de continuer, par exemple dès qu'une hypothèse dépasse 0,95 (A. Wald) | 4 |
| paramètre de nuisance | nuisance parameter | paramètre nécessaire au modèle, mais qui n'intéresse pas en lui-même ; l'approche bayésienne en fait la moyenne (elle le marginalise) | 4 |
| programmation probabiliste | probabilistic programming | décrire un modèle et laisser une bibliothèque (PyMC, Stan) produire des échantillons du posterior | 4 |
| MCMC | Markov chain Monte Carlo (méthodes de Monte-Carlo par chaînes de Markov) | algorithmes qui tirent des échantillons du posterior quand une grille est impossible (trop de paramètres) | 4 |
| refactoriser | refactor | réécrire du code sans changer ce qu'il fait, pour le rendre plus simple et plus sûr (une fonction testée plutôt que trois copies) | 4 |
| courbe continue, lisse, univoque | continuous, smooth, single-valued curve | les trois règles du livre (ch. 5) : sans saut, sans point anguleux, une seule valeur par abscisse ; ajoutons « jamais verticale » ; alors chaque point a une seule dérivée, finie | 5 |
| point anguleux | corner, kink (*cusp* dans le livre) | point où la pente à gauche et la pente à droite diffèrent, comme $|x|$ ou ReLU en 0 ; le point de rebroussement (*cusp* au sens strict) est un cas plus pointu | 5 |
| extremum local, extremum global | local extremum, global extremum | la plus grande (ou plus petite) valeur dans un voisinage, ou sur tout le domaine ; la valeur d'un extremum global est unique, mais elle peut être atteinte en plusieurs points | 5 |
| sécante | secant | droite qui passe par deux points d'une courbe ; quand les deux points se rapprochent, elle tend vers la tangente | 5 |
| différence finie (avant, arrière, centrée) | finite difference (forward, backward, central) | pente approchée avec un pas $h$ : $\frac{f(x + h) - f(x)}{h}$, $\frac{f(x) - f(x - h)}{h}$, $\frac{f(x + h) - f(x - h)}{2h}$ ; la centrée a une erreur en $h^2$, les deux autres en $h$ | 5 |
| erreur de troncature | truncation error | l'erreur de la formule elle-même (le pas $h$ n'est pas nul) ; elle diminue avec $h$ | 5 |
| erreur d'arrondi | round-off error | l'erreur due aux flottants, qui ne gardent qu'environ 16 chiffres ; dans une différence finie, elle grandit comme $\frac{\varepsilon}{h}$ quand $h$ diminue | 5 |
| epsilon machine | machine epsilon | précision relative d'un `float64`, environ $2{,}2 \times 10^{-16}$ (`np.finfo(float).eps`) | 5 |
| annulation catastrophique | catastrophic cancellation | perte de chiffres exacts quand on soustrait deux nombres presque égaux : il ne reste que les derniers chiffres, ceux qui portent l'erreur d'arrondi | 5 |
| dérivée seconde, courbure | second derivative, curvature | dérivée de la dérivée : positive dans une cuvette (convexe), négative sous un dôme (concave) ; différence seconde $\frac{f(x + h) - 2f(x) + f(x - h)}{h^2}$ | 5 |
| point critique, point stationnaire | critical point, stationary point | point où la dérivée (le gradient) s'annule : maximum, minimum, point selle ou plateau | 5 |
| point d'inflexion à tangente horizontale | stationary inflection point | point de pente nulle où la courbe continue de monter (ou de descendre), comme $x^3$ en 0 ; le livre l'appelle *plateau* | 5 |
| point selle, col | saddle point | point critique où la surface monte dans certaines directions et descend dans d'autres, comme $x^2 - y^2$ en $(0, 0)$ | 5 |
| plateau | plateau | zone où la surface est presque plate : la descente y avance à tout petits pas, et la loss semble avoir convergé | 5 |
| pente dans une direction, dérivée directionnelle | directional derivative | $\nabla f(\mathbf{x}) \cdot \mathbf{u}$ pour un vecteur unitaire $\mathbf{u}$ : maximale dans le sens du gradient, nulle le long d'une ligne de niveau | 5 |
| montée de gradient | gradient ascent | $\mathbf{x} \leftarrow \mathbf{x} + \eta\,\nabla f(\mathbf{x})$ : la descente de gradient pour chercher un maximum (`maximize=True` dans `torch.optim.SGD`) | 5 |
| matrice hessienne | Hessian matrix | matrice des dérivées secondes d'une fonction de plusieurs variables ; les signes de ses valeurs propres classent un point critique (ch. 19) | 5 |
| conditionnement | conditioning | rapport entre la plus forte et la plus faible courbure : une surface mal conditionnée (une vallée étroite comme Rosenbrock) force un petit learning rate et ralentit la descente | 5 |
| fonction de Rosenbrock | Rosenbrock function | $(a - x)^2 + b\,(y - x^2)^2$, de minimum $(a, a^2)$ au fond d'une vallée étroite et courbe : le banc d'essai classique des optimiseurs | 5 |
| différentiation automatique | automatic differentiation, autodiff | calcul exact (aux arrondis près) d'une dérivée en enregistrant les opérations élémentaires et en appliquant la règle de la chaîne ; en mode inverse, pour quelques évaluations, quel que soit le nombre de paramètres | 5 |
| tenseur | tensor | le tableau de PyTorch, l'équivalent d'un tableau NumPy ; avec `requires_grad=True`, PyTorch enregistre les opérations faites avec lui | 5 |
| sous-gradient | subgradient | en un point anguleux d'une fonction convexe, toute pente comprise entre la pente de gauche et celle de droite ; PyTorch prend celle de plus petite norme (0 pour ReLU en 0) | 5 |
| test de propriétés | property-based testing | tester une propriété vraie pour beaucoup d'entrées (par exemple, la forme du gradient) plutôt qu'une seule valeur calculée à la main ; *Hypothesis* choisit lui-même les entrées | 5 |
| information (de Shannon), surprise | self-information, surprisal | $-\log_2 p$ : ce qu'apprend un événement de probabilité $p$ ; nulle pour un événement certain, elle s'additionne pour des événements indépendants ; elle ne dépend pas du sens du message | 6 |
| bit (d'information) | bit, shannon (Sh) | unité d'information, avec le logarithme en base 2 ; à distinguer du chiffre binaire, qui peut porter un bit ou moins | 6 |
| nat | nat | unité d'information avec le logarithme népérien : 1 nat $\approx 1{,}443$ bit ; l'unité des losses de PyTorch et de scikit-learn | 6 |
| contexte global, contexte local | global context, local context | ce que l'émetteur et le récepteur partagent avant le message (la langue, la culture, un prior) ; les symboles qui le précèdent dans le message | 6 |
| code de longueur fixe | fixed-length code | chaque symbole reçoit un mot de même longueur : $\lceil \log_2 N \rceil$ bits pour $N$ symboles | 6 |
| code à longueur variable, code adaptatif | variable-length code (*variable-bitrate code* dans le livre) | mots courts pour les symboles fréquents, longs pour les rares : le Morse, le code de Huffman | 6 |
| code préfixe | prefix code, prefix-free code | aucun mot de code n'est le début d'un autre : une suite de bits se lit sans séparateur | 6 |
| inégalité de Kraft | Kraft inequality | $\sum_i 2^{-\ell_i} \le 1$ pour les longueurs d'un code préfixe binaire ; égalité pour un code complet | 6 |
| code de Huffman | Huffman code | le meilleur code préfixe symbole par symbole : on fusionne les deux groupes les moins probables jusqu'à n'en avoir qu'un ; $H \le \bar{L} < H + 1$ | 6 |
| entropie (de Shannon) | (Shannon) entropy | surprise moyenne d'une distribution, $-\sum_i p_i \log_2 p_i$ : 0 pour une issue certaine, au plus $\log_2 n$ (loi uniforme) ; la borne inférieure du nombre moyen de bits par symbole de tout code, qu'on approche en codant de longs blocs | 6 |
| cross-entropy, entropie croisée | cross-entropy | $-\sum_i p_i \log_2 q_i$ : coût moyen de données tirées de $p$ envoyées avec un code fait pour $q$ ; la loss de la classification | 6 |
| divergence KL, divergence de Kullback-Leibler | KL divergence, relative entropy | $\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p)$ : le surcoût du mauvais code ; positive, nulle seulement si $q = p$, pas symétrique ; on écrit $\mathrm{KL}(\text{données} \,\|\, \text{modèle})$ | 6 |
| divergence de Jensen-Shannon | Jensen-Shannon divergence | moyenne des KL de $p$ et de $q$ vers leur mélange $\frac{p + q}{2}$ : symétrique, toujours finie, au plus 1 bit ; sa racine est une distance | 6 |
| taux de compression | compression ratio | taille comprimée divisée par la taille d'origine (dans le livre : bits du code adaptatif sur bits du code fixe) ; certains auteurs prennent l'inverse | 6 |
| compression sans perte | lossless compression | on retrouve exactement les données d'origine (zlib, ZIP, PNG, FLAC) | 6 |
| lissage de Laplace | Laplace smoothing, add-one smoothing | ajouter un pseudo-compte $\alpha$ à chaque élément d'un vocabulaire avant de normaliser, $\frac{n_i + \alpha}{n + \alpha V}$ : plus aucune probabilité nulle | 6 |
| log loss | log loss, logistic loss | moyenne des $-\ln$ de la probabilité donnée à la vraie classe : la cross-entropy d'un classifieur, en nats | 6 |
| perplexité | perplexity | $e^{\text{loss moyenne par token en nats}}$ : un nombre de choix équivalent ; ne se compare qu'avec le même tokenizer et les mêmes données | 6 |
| entropie conditionnelle | conditional entropy | surprise moyenne d'un symbole quand on connaît son contexte (la lettre précédente) ; jamais plus grande que l'entropie | 6 |
| modèle unigramme, bigramme, trigramme | unigram, bigram, trigram model | modèle qui prédit chaque symbole sans contexte, sachant le précédent, sachant les deux précédents | 6 |
| codage arithmétique | arithmetic coding | code tout un message presque au prix de sa surprise totale, sans arrondir symbole par symbole : il transforme un modèle de prédiction en compresseur | 6 |
| ANS | asymmetric numeral systems | famille de codes aussi efficaces que le codage arithmétique, et plus rapides (FSE, dans zstd) | 6 |
| BPE | byte-pair encoding | algorithme qui construit un vocabulaire de sous-mots en fusionnant les paires les plus fréquentes ; le tokenizer de GPT-2 (50 257 tokens) | 6 |
| label smoothing | lissage des étiquettes | remplacer la cible *one-hot* par un mélange avec la loi uniforme, pour que le modèle ne vise pas 100 % ; option de `torch.nn.CrossEntropyLoss` | 6 |
| distillation | knowledge distillation | entraîner un petit modèle sur les probabilités adoucies (par une température) d'un grand modèle | 6 |
| softmax | softmax | transforme un vecteur de scores en probabilités positives de somme 1 : $\frac{e^{z_k}}{\sum_j e^{z_j}}$ ; la dernière couche d'un classifieur | 6 |
| logits (d'un classifieur) | logits | les scores d'un classifieur avant la softmax, ni positifs ni de somme 1 ; `torch.nn.CrossEntropyLoss` les attend à la place des probabilités (pour deux classes, le logit du 0B) | 6 |
| tokenizer | tokeniseur | programme qui découpe un texte en tokens et les numérote ; la perplexité d'un modèle de langage dépend de ce découpage | 6 |
| redondance (d'une source) | redundancy | ce que l'on peut deviner d'avance : $1 - \frac{H}{\log_2 n}$ ; une langue très redondante se comprime bien | 6 |
