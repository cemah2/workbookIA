# Audit P5, tour 2 : exactitude des ch. 4, 5 et 6 (relecteur « exact2 »)

Date : 2026-10-04. Lecture seule : aucun fichier du dépôt n'a été modifié, et `git -C /home/claude/workbookia status --short` est vide à la fin. Calculs refaits en Python 3.13 (venv du projet, `OMP_NUM_THREADS=1`), sans processus restant.

**Périmètre relu.** Pour les ch. 4, 5 et 6 : `01_fiche.md`, `02_exercices.md`, `04_indices.md`, `05_solutions.md`, `flashcards.csv` (22, 20 et 25 cartes), les 18 figures (PNG ouverts), les cellules Markdown des deux notebooks (scripts `tools/chapters/build_ch04.py` à `build_ch06.py`) et les sorties de `05_solutions.ipynb`. Également les sections 4 à 6 de `annexes/formulaire.md` et les entrées de ces chapitres dans `annexes/glossaire.md` et `annexes/erreurs_frequentes.md`. Contradictions cherchées par grep dans 0A, 0B, ch. 1 à 3, 7 à 11, les checkpoints et les annexes (base des logarithmes, bits et nats, gradient, learning rate, convexité, vraisemblance, posterior, intervalle de confiance). Je ne re-signale ni ce qui figure dans `docs/AUDIT_parties_I_II.md` §7 et §8, ni les décisions P1 à P5.

## Constats

### MAJEUR

1. **MAJEUR (contrat, sans effet pour l'apprenant)** · `docs/BIBLE.md:487` (§22, « Alphabets et lissage du ch. 6 »)
   - **Extrait** : « `LETTERS_FR` (26 + 16 lettres accentuées et ligatures du français, 42 en tout) dès que Holmes et Verne sont comparés sur les mêmes issues (🔮 6.14 b, 🐛 6.17, 🔬 6.18, 🔮 6.25) »
   - **Problème** : 🔬 6.18 et 🔮 6.25, tels que publiés, travaillent sur les 26 lettres de `LETTERS`, pas sur `LETTERS_FR` :
     - énoncé de 6.18 (`tools/chapters/build_ch06.py:633`) : « Sur les 26 lettres (`LETTERS`, accents ignorés, toutes présentes dans les deux livres) » ;
     - énoncé de 6.25 (`build_ch06.py:1152`) : « On construit deux codes de Huffman sur les 26 lettres : celui des lettres de Holmes et celui des lettres de Verne (a–z, accents ignorés, comme en 6.18) » ;
     - solutions : `char_distribution(…, alphabet=LETTERS)` (`build_ch06.py:656-657`, `1004`).

     Les réponses enregistrées de 6.18 a–d et 6.25 e–f sont calculées sur 26 lettres. Sur 42 lettres, la cross-entropy de Verne codé avec Holmes serait infinie, puisque 9 lettres accentuées de Verne manquent chez Holmes (🐛 6.17). La bible, qui fait foi, décrit donc ces deux exercices autrement qu'ils ne sont publiés. Une session qui l'appliquerait à la lettre changerait des réponses publiées.
   - **Correction** (dans la bible, pas dans les exercices) : « … `LETTERS_FR` (26 + 16 lettres accentuées et ligatures du français, 42 en tout) quand la comparaison doit garder les accents (🔮 6.14 b, 🐛 6.17) ; 🔬 6.18 et 🔮 6.25 comparent Holmes et Verne sur les 26 lettres de `LETTERS`, toutes présentes dans les deux livres, pour que la cross-entropy reste finie sans lissage ; … ».

### MINEUR

2. **MINEUR** · La règle « axes et diagonales » est présentée comme décisive, alors qu'elle peut manquer une selle.
   - **Extraits** :
     - `chapitres/ch05_courbes/01_fiche.md:182` : « positive partout, c'est un minimum ; négative partout, un maximum ; […] `classify_critical_point` (5.24) regarde ainsi les axes et les diagonales. » ;
     - `annexes/formulaire.md:256` : « dérivées secondes selon les axes $\mathbf{e}_i$ et les diagonales $\mathbf{e}_i \pm \mathbf{e}_j$ : toutes $> 0$ minimum, toutes $< 0$ maximum, des deux signes selle, sinon on ne conclut pas ; en toute rigueur, les signes des valeurs propres de la hessienne » ;
     - `annexes/erreurs_frequentes.md:187` (remède) : « regarder la courbure dans plusieurs directions, axes **et** diagonales (5.24) ».
   - **Problème** : la référence (`solutions/mylearn_ref/calculus.py`) répond `"minimum"` pour $\mathbf{v}^\top A\,\mathbf{v}$ en $(0, 0)$ dans deux cas qui sont des selles :
     - $A = \begin{pmatrix} 1 & 11 \\ 11 & 100 \end{pmatrix}$ : valeurs propres −0,207 et 101,2 ; la forme vaut −20 dans la direction (10, −1) ;
     - $A = \begin{pmatrix} 1 & 20 \\ 20 & 100 \end{pmatrix}$ : valeurs propres −2,89 et 103,9.

     Seuls la variante de 5.24 (`05_solutions.md:264`) et la note de solution du notebook (`build_ch05.py:1460-1462`) le signalent. Qui révise avec le formulaire ou les erreurs fréquentes retient « toutes > 0 : minimum ». La docstring du stub, figée, n'en parle pas non plus : l'avertissement doit donc figurer dans la fiche et dans les annexes.
   - **Correction** :
     - fiche, fin de L182 : « `classify_critical_point` (5.24) regarde ainsi les axes et les diagonales : cela suffit pour $xy$, pas pour toutes les selles (variante de 5.24). L'outil complet, … » ;
     - formulaire L256 : « … : toutes $> 0$, minimum probable (une selle peut échapper à ces directions, variante de 5.24) ; […] ; seuls les signes des valeurs propres de la hessienne tranchent à coup sûr » ;
     - erreurs fréquentes L187 : « …, axes **et** diagonales (5.24), qui peuvent encore manquer une selle ; la hessienne (ch. 19) tranche ».

3. **MINEUR** · `annexes/formulaire.md:245`
   - **Extrait** : « dérivée (sécante symétrique) | $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x - h)}{2h}$ »
   - **Problème** : écrite comme une définition, sans condition, la formule est fausse en un point anguleux. Pour $\lvert x \rvert$ en 0, la limite existe et vaut 0 (½ pour ReLU), alors que la dérivée n'existe pas. La fiche le dit : `ch05_courbes/01_fiche.md:76` (« la sécante symétrique […] donne 0, une valeur qui ne décrit ni la pente de gauche ni celle de droite ») et L102 (« pour une courbe lisse, les deux limites coïncident »). Le formulaire ne le dit pas.
   - **Correction** : « dérivée (sécante symétrique), si $f$ est dérivable en $x$ | $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x - h)}{2h}$ ; en un point anguleux, cette limite peut exister sans que $f$ soit dérivable (0 pour $\lvert x \rvert$ en 0) ».

4. **MINEUR** · La correction de la session 15 n'a pas été reportée hors de la fiche. Elle est consignée à `docs/BIBLE.md:499` : « Une ligne de la fiche du ch. 5 corrigée : « la loss grimpe, souvent en dents de scie, puis explose » (sur une parabole, seule la variable oscille) ».
   - **Extraits** :
     - `chapitres/ch05_courbes/flashcards.csv:14` : « Trop grand : chaque pas dépasse le minimum, la loss oscille puis explose (divergence). » ;
     - `annexes/erreurs_frequentes.md:184` (symptôme) : « la loss oscille, puis explose (jusqu'à `nan`) » ;
     - `chapitres/ch05_courbes/05_solutions.md:146` (réponse modèle de 5.E1) : « les pas enjambent le fond de la vallée : la loss oscille, puis peut exploser, jusqu'à `nan` ».
   - **Problème** : sur une parabole $L = \frac{c}{2}x^2$ avec $\eta > \frac{2}{c}$, $x$ change de signe à chaque pas, mais $L$ est multipliée par $(1 - \eta c)^2 > 1$ : elle croît à chaque pas. C'est la variable qui oscille, pas la loss. La fiche dit maintenant « la loss grimpe, souvent en dents de scie, puis explose » (`01_fiche.md:238`). De plus, « chaque pas dépasse le minimum » est déjà vrai pour $\frac{1}{c} < \eta < \frac{2}{c}$, où la descente converge.
   - **Correction** :
     - flashcard : « Trop grand : chaque pas enjambe le minimum ; au-delà de \(2/c\), la variable oscille de plus en plus loin, et la loss grimpe (souvent en dents de scie) puis explose. Trop petit : la loss baisse, mais très lentement. » ;
     - erreurs fréquentes : « la loss grimpe, souvent en dents de scie, puis explose (jusqu'à `nan`) » ;
     - 5.E1 : « … les pas enjambent le fond de la vallée : la loss grimpe, souvent en dents de scie, puis explose, jusqu'à `nan`. »

5. **MINEUR** · Les durées citées ne sont pas celles du corrigé exécuté.
   - **Extraits** :
     - note de solution de 6.21 (`tools/chapters/build_ch06.py:892-895`, cellule « 💡 » de `05_solutions.ipynb`) : « Sur la machine où ce corrigé a tourné, la version NumPy est de loin la plus rapide (environ 2 ms) ; `str.count` vient ensuite (environ 8 fois plus lente, […]), puis `Counter` (environ 15 fois) et la boucle Python (environ 30 fois) ». La sortie juste au-dessus affiche `count_numpy_21 1.44 ms`, `count_str_21 … ( 6.7 × the fastest)`, `count_counter_21 … ( 12.0 × the fastest)` et `count_loop_21 … ( 22.9 × the fastest)` ;
     - `chapitres/ch06_information/05_solutions.md:234` : « Lors de l'exécution du corrigé : NumPy **1,6 ms**, `str.count` environ **8 fois** plus lente, `Counter` environ **14 fois**, la boucle environ **28 fois**. » C'est un troisième jeu de chiffres ;
     - `chapitres/ch04_bayes/05_solutions.md:392` : « (0,7 s contre quelques millisecondes ici) », alors que la sortie de 4.24 affiche « ⏱️ update_discrete, 20 000 flips : 0.90 s ».
   - **Problème** : le texte dit décrire l'exécution du corrigé, et la contredit (pour la boucle : « 30 fois » et « 28 fois » contre 22,9). La note ajoute elle-même : « Les rapports exacts dépendent de la machine » ; chaque réexécution change donc la sortie, jamais le texte. Le constat compare le texte aux sorties publiées et ne dépend d'aucune mesure faite pendant l'audit.
   - **Correction** : ne citer que des ordres de grandeur.
     - Note de 6.21 : « Sur la machine où ce corrigé a tourné, la version NumPy est de loin la plus rapide (de l'ordre de la milliseconde) ; `str.count` vient ensuite (quelques fois plus lente, […]), puis `Counter` (une dizaine de fois) et la boucle Python (plus de vingt fois) » ;
     - `05_solutions.md:234`, même formulation ;
     - ch. 4 : « (de l'ordre d'une seconde contre quelques millisecondes ici) ».

6. **MINEUR** · Des indices de niveau 3 donnent des réponses vérifiées par `wb.check`. La règle est celle du BIBLE §12 : « **Indice 3** (presque la solution : pseudo-code ou première ligne) ». Les vérifications des ch. 5 et 6 en avaient déjà retiré (BIBLE §22, L484 : « trois indices qui donnaient des valeurs vérifiées » ; L493 : « réponses de 🔮 données par les indices »). Il en reste :
   - `ch04_bayes/04_indices.md:805` (🔮 4.21) : « de l'ordre de 0,1 pour 100 lancers, de 0,01 pour 1 000. Le rapport des deux posteriors est proportionnel au prior, qui n'est pas constant. » → donne 4.21 a (`True`), b (`1000`) et c (`False`) ;
   - `ch05_courbes/04_indices.md:285` (✏️ 5.2) : « en $P$, le gradient vaut $(1, -3)$, de norme $\sqrt{10}$ » → donne 5.2 b et c ;
   - `ch05_courbes/04_indices.md:321` (✏️ 5.4) : « $f(-1) = 2$, $f(1) = -2$, $f(2{,}5) = 8{,}125$ […]. Sur $[-2 ; 2]$, la valeur maximale 2 est atteinte deux fois. » → donne 5.4 b, d et f ;
   - `ch05_courbes/04_indices.md:710` (📈 5.17) : « En D, les couleurs claires sont vers le haut et vers la gauche. » → donne 5.17 e (`"NO"`) ;
   - `ch05_courbes/04_indices.md:788` (🔮 5.20) : « le signe de $y$ ne change pas […], de l'ordre de la centaine […]. Pour d), il faut gagner un facteur $10^6$ de plus : $n = \frac{\ln 10^6}{\ln 1{,}1}$ » → donne 5.20 b (`1`), c (`100`) et d ;
   - `ch05_courbes/04_indices.md:945` (🏆 5.25) : « Un learning rate un peu en dessous, vers 0,0019, arrive à temps et reste. » → c'est une solution complète. Calcul : `[(0.0019, 10000)]` passe sous $10^{-3}$ de (1, 1) au pas 8 222, finit à $2{,}6 \times 10^{-4}$, ne s'en écarte jamais de plus de $2{,}6 \times 10^{-4}$ pendant les 10 000 pas suivants et finit à $1{,}3 \times 10^{-7}$ ;
   - `ch06_information/04_indices.md:393` (✏️ 6.6) : « Un code possible : soleil `0`, nuages `10`, pluie `110`, neige `1110`, vent `1111`. » → donne 6.6 b ([1, 2, 3, 4, 4]), puis c, e et f par une simple addition ;
   - `ch06_information/04_indices.md:684` (🔮 6.14) : « l'entropie de Verne monte, celle de Holmes ne bouge presque pas. L'entropie de Holmes est un peu au-dessus de 4 bits. » → donne 6.14 c (`4`) et le sens de a et b ;
   - `ch06_information/04_indices.md:1072` (🔮 6.25) : « environ 0,2 bit dans un sens, 0,3 dans l'autre. […] on reste bien sous 5 bits. Dans l'autre sens, les « h », « w », « y » de Holmes coûtent cher » → donne 6.25 a, b, c (`0.2`) et d ;
   - `ch06_information/04_indices.md:1121` (🏆 6.27) : « Des triplets passent […] un lissage de 0,1 bien moins » → c'est exactement la solution de référence (`BLOCK_27 = 3`, `smoothing=0.1`, `build_ch06.py:1272-1283`).

   **Moins grave** (réponses non vérifiées par `wb.check`, mais données telles quelles) : les quiz `ch05_courbes/04_indices.md:43, 61, 115, 133, 151` (5.Q2, Q3, Q6, Q7, Q8), `ch06_information/04_indices.md:151` (6.Q8 : les quatre entropies, $\log_2 16$, « elle ne dépend que de $p$ ») et le 🧮 `:469` (6.10 : « 1. Environ 4,5 millions de bits. 2. 7 bits par caractère. […] 5. Environ 4 200 minutes ; quelques centièmes de seconde. »).

   **Correction** : s'arrêter au calcul posé, comme le font la plupart des indices des ch. 4 à 6. Par exemple :
   - 5.2 : « … : remplace $x$ par 1 et $y$ par $-1$, puis calcule la norme » ;
   - 5.4 : « … calcule $f$ aux deux points critiques et aux bornes, puis compare » ;
   - 5.17 : « En D, cherche de quel côté sont les couleurs claires : le gradient pointe vers elles » ;
   - 4.21 : « Le MAP est décalé vers la bosse d'environ $\frac{0{,}8 - 0{,}3}{0{,}1^2} \times \frac{0{,}3 \times 0{,}7}{n}$ : calcule ce décalage pour $n = 100$ et pour $n = 1\,000$. Que vaut le rapport des deux posteriors ? » ;
   - 5.20 : garder la récurrence $y \leftarrow 1{,}1\,y$ sans son résultat ;
   - 5.25 : « … au-delà de $\frac{2}{1\,000}$, la descente oscille en travers de la vallée sans s'amortir : essaie un learning rate juste en dessous » ;
   - 6.6 : « Soleil n'est fusionné qu'une fois, neige et vent quatre fois : déduis-en les longueurs », sans donner le code ;
   - 6.14 : retirer la dernière phrase ;
   - 6.25 : ne garder que « Tu as mesuré ces KL en 6.18 c) et d) » ;
   - 6.27 : « Des paires ne suffisent pas tout à fait ; essaie des blocs plus longs, et regarde ce que le lissage fait aux blocs jamais vus ».

7. **MINEUR** · `chapitres/ch05_courbes/05_solutions.md:262`, et note de solution de 5.24 (`tools/chapters/build_ch05.py:1452-1453`)
   - **Extraits** : « en $(0, 0)$, $h$ ne varie le long des axes qu'en $v^4$ : la différence seconde y vaut $2h^2 = 2 \times 10^{-6}$ » ; dans la note : « En $(0, 0)$, $h$ vaut $v^4$ le long de chaque axe : la différence seconde y est minuscule ($2h^2 = 2 \times 10^{-6}$ avec $h = 10^{-3}$, […]) »
   - **Problème** : dans la même phrase, la lettre $h$ désigne à la fois la fonction testée, $h(\mathbf{v}) = v_0^4 + v_1^4 - 4v_0v_1$ (`h_24` dans l'énoncé), et le pas `h=1e-3` de `classify_critical_point` (stub figé). « $2h^2$ » se lit donc comme $2\,h(\mathbf{v})^2$.
   - **Correction** (dans le texte seulement ; l'énoncé et le stub ne changent pas) : « … la différence seconde y vaut $2 \times (10^{-3})^2 = 2 \times 10^{-6}$ (avec le pas par défaut `h=1e-3`) ».

8. **MINEUR** · Note de solution de 5.15 (`tools/chapters/build_ch05.py:706-709`, cellule « 💡 Il faut deux évaluations… » de `05_solutions.ipynb`)
   - **Extrait** : « Le gradient de b) a la forme de $W$ […] $W\mathbf{v} = (-2 ;\ 4{,}5)$, donc $W\mathbf{v} - \mathbf{t} = (-2 ;\ 3{,}5)$ et $2\,(W\mathbf{v} - \mathbf{t})\,\mathbf{v}^\top$ »
   - **Problème** : reste du constat A1 n° 24 (« $\mathbf{W}$ en gras (ch. 5) », noté « appliqué »). L'énoncé (L670, L673) et `05_solutions.md:196` écrivent $\mathbf{W}$, cette note écrit $W$.
   - **Correction** : remplacer les quatre `W` de la note par `\\mathbf{W}`.

9. **MINEUR** · `annexes/glossaire.md:319`
   - **Extrait** : « sous-gradient | […] PyTorch prend celle de plus petite norme (0 pour ReLU en 0) »
   - **Problème** : c'est vrai pour `torch.relu`, et c'est ce que dit la documentation, mais pas en général. `annexes/erreurs_frequentes.md:158` donne `torch.clamp` : 1 et `torch.maximum` : 0,5 (5.21, sorties du corrigé, reproduites). La fiche (`01_fiche.md:80`, « selon la documentation de PyTorch ») et la flashcard `flashcards.csv:19` (« Toutes les opérations ne suivent pas cette règle ») nuancent. Lu seul, le glossaire contredit l'annexe des erreurs fréquentes.
   - **Correction** : « … ; selon la documentation de PyTorch, celle de plus petite norme (0 pour `torch.relu` en 0), mais chaque opération a sa convention (`torch.clamp` : 1, `torch.maximum` : 0,5 ; 5.21) ».

10. **MINEUR** · Énoncé de 🔬 6.26 (`tools/chapters/build_ch06.py:1195`, dans les deux notebooks)
    - **Extrait** : « Un modèle **unigramme** donne à chaque lettre la même probabilité, quelle que soit la lettre d'avant. »
    - **Problème** : on lit d'abord « la même probabilité pour toutes les lettres », c'est-à-dire un modèle uniforme. Or le modèle demandé est `char_distribution(train_26, alphabet=LETTERS, smoothing=1)`, qui donne à chaque lettre sa fréquence (glossaire L340 : « modèle qui prédit chaque symbole sans contexte »). Pour un débutant, « unigramme » et « uniforme » se confondent facilement.
    - **Correction** (formulation seule, la tâche ne change pas) : « Un modèle **unigramme** donne à chaque lettre sa fréquence, la même quelle que soit la lettre d'avant. »

### SUGGESTION

11. **SUGGESTION** · `chapitres/ch06_information/05_solutions.md:39` (6.Q10, question 3), et l'indice `04_indices.md:187`
    - **Extraits** : « Un bon code donne un taux **proche de 0** : le livre trouve un peu moins de 0,5 pour *Huckleberry Finn* envoyé avec son propre code » ; dans l'indice : « Un taux proche de 0 signifie une forte compression »
    - **Problème** : 0,5 n'est pas « proche de 0 », et un code lettre à lettre sans perte ne peut pas descendre sous l'entropie divisée par la longueur fixe. Pour Holmes : 4,17/5 ≈ 0,83, et Huffman donne 0,8404 (6.24 e).
    - **Correction** (corrigé et indice seulement ; l'énoncé ne change pas) : « Un bon code donne un taux nettement **sous 1** : plus il est petit, plus on comprime, mais un code lettre à lettre ne descend pas sous l'entropie divisée par la longueur fixe ; le livre trouve un peu moins de 0,5 … ».

## Encadrés 🕰️ : affirmations datées vérifiées sur le web (au 2026-10-04)

Aucune n'est fausse ni dépassée.

| # | Affirmation (lieu) | Source consultée | Verdict |
|---|---|---|---|
| 1 | SciPy 1.15 ajoute `scipy.differentiate` et retire `scipy.misc` (dont `derivative`) (ch. 5 L150) | Notes de version de SciPy 1.15.0 : « All functions in the `scipy.misc` submodule have been removed » | exact |
| 2 | PyTorch prend le sous-gradient de plus petite norme d'une fonction convexe (ReLU en 0 : 0) (ch. 5 L80, 5.Q9, flashcard 19) | PyTorch, « Autograd mechanics », *Gradients for non-differentiable functions* (docs 2.14) | exact ; 5.21 reproduit 0 (`relu`), 1 (`clamp`), 0,5 (`maximum`) |
| 3 | Bertoin et coll. (2021) : en float32, des entrées exactement nulles assez fréquentes pour que ReLU'(0) change l'entraînement ; systématique en 16 bits, disparaît en 64 bits ; plus de 10 points sur ImageNet ; batch norm et Adam atténuent (ch. 5 L80, `05_solutions.md:36`) | arXiv:2106.12915 (NeurIPS 2021) | exact |
| 4 | `gradcheck` doit s'utiliser en float64 (ch. 5 L215) | documentation de `torch.autograd.gradcheck` | exact |
| 5 | Dauphin et coll. (2014) : en grande dimension, prolifération des selles entourées de plateaux, qui donnent l'illusion d'un minimum (ch. 5 L184) | arXiv:1406.2572, résumé | exact |
| 6 | Lee et coll. (2016) : la descente de gradient depuis un point aléatoire converge presque sûrement vers un minimiseur local (ch. 5 L184) | arXiv:1602.04915 (COLT 2016), résumé | exact |
| 7 | Du et coll. (2017) : temps exponentiel pour s'échapper d'une selle ; une version bruitée s'échappe en temps polynomial (ch. 5 L184) | arXiv:1705.10412, résumé | exact |
| 8 | Optuna utilise TPE par défaut (ch. 4 L247) | documentation de `optuna.create_study` (5.0.0) : TPESampler par défaut | exact |
| 9 | Naive Bayes, d'après scikit-learn : rapide et bon classifieur, mais probabilités peu fiables (ch. 4 L247) | scikit-learn 1.6, « Naive Bayes » : « known to be a bad estimator » | exact |
| 10 | PyMC (et Stan) : NUTS par défaut, inférence variationnelle en option (ch. 4 L233) | PyMC, « Introductory Overview of PyMC » (v5.28 : `NUTS[nutpie]`, ADVI par `fit()`) | exact |
| 11 | `log_loss` de scikit-learn : logarithme népérien, probabilités écrêtées (ch. 6 L221, formulaire L55) | documentation de `sklearn.metrics.log_loss` (1.6) | exact |
| 12 | `CrossEntropyLoss` de PyTorch en nats (ch. 6 L221) | documentation de `torch.nn.CrossEntropyLoss` (2.11) | exact |
| 13 | La perplexité dépend de la tokenisation ; calcul par fenêtre glissante (ch. 6 L235) | Hugging Face, « Perplexity of fixed-length models » | exact |
| 14 | GPT-2 : 50 257 tokens, BPE au niveau des octets (ch. 6 L235, ✏️ 6.2) | documentation Hugging Face de GPT-2 : `vocab_size` 50257, « byte-level Byte-Pair-Encoding » | exact |
| 15 | Delétang et coll. (ICLR 2024) : Chinchilla 70B comprime ImageNet à 43,4 % et LibriSpeech à 16,4 %, contre 58,5 % (PNG) et 30,3 % (FLAC) ; le tableau 1 donne 48,0 % et 21,0 % ; l'avantage disparaît si l'on compte la taille du modèle (ch. 6 L249) | arXiv:2309.10668v2, résumé et Table 1 | exact |
| 16 | zstd : Huffman pour les littéraux, FSE (ANS) pour les séquences (ch. 6 L249) | RFC 8878 | exact |
| 17 | Brotli : codes de Huffman et modélisation de contexte, sans ANS (ch. 6 L249) | RFC 7932 | exact |
| 18 | VanderPlas (2014) : intervalles exact (10,2 ; 12,2) et approché (10,2 ; 12,5) (ch. 4 L80, 📄 4.11) | arXiv:1411.5018 (PDF) | exact ; la solution de 4.11 cite bien (10,2 ; 12,2) |

## Vérifié sans constat (pour le tri)

- **Réponses vérifiées.** ✏️ et ∂ 4.1 à 4.8, 5.1 à 5.7, 6.1 à 6.8, quiz et rappels chiffrés, 🧮 5.9 et 6.10 : tout a été recalculé, et tout concorde avec les solutions et les enregistrements. Les sorties de `05_solutions.ipynb` concordent avec `05_solutions.md` dans les trois chapitres, sauf les durées (constat 5).
- **Ch. 4.** Posteriors et cotes, sondes de 4.8, loi Beta, intervalle de crédibilité par `np.searchsorted`, écart-type 0,0034 en 4.24 ; underflow : `0.5**1074` ≈ 4,9e-324 et `0.5**1075 == 0`.
- **Ch. 5.**
  - Meilleurs pas : $(\varepsilon/2)^{1/3} \approx 4{,}8 \times 10^{-6}$ pour la différence centrée et $\sqrt{\varepsilon} \approx 1{,}5 \times 10^{-8}$ pour la différence avant (formulaire L248 cohérent).
  - Hessienne de Rosenbrock en (1, 1) : $\begin{pmatrix} 802 & -400 \\ -400 & 200 \end{pmatrix}$, valeurs propres 1 001,6 et 0,3994, d'où la limite $\eta < 0{,}001997$ (formulaire L255 : « ≈ 1 002 et 0,4 »).
  - 🏆 5.25 : $\eta = 0{,}002$ atteint le minimum mais n'y reste pas (écart maximal 4,3e-3), comme le dit `erreurs_frequentes.md:188`.
  - 5.24 : en (1, 1), valeurs propres 8 et 16 ; en (1, 0), gradient (4, −4).
- **Ch. 6, sur les textes.**
  - Hapax : 3 454 sur 7 819 mots différents (Holmes), 4 558 sur 8 793 (Verne).
  - Mots : entropie 9,210 bits, Huffman 9,236.
  - Lettres : part du « e », 12,30 % (Holmes) et 14,15 % (Verne) ; lettres accentuées, 3,28 % chez Verne (é : 1,82 %) contre 23 occurrences chez Holmes (é : 12) ; Morse, 2,547 et 2,484 signes (points et traits) par lettre.
  - KL de 6.17 : 0,4543 ; les neuf lettres manquantes y comptent pour 0,048 bit (18,72 bits par occurrence) ; « é » pèse à lui seul 0,168 bit (15,02 bits par occurrence).
  - Mélange : 4,525 pour w = 0,01, 4,450 pour w = 0,1.
  - Contributions de 6.18 : h 0,183, w 0,132, u 0,063, y 0,060, q 0,043.
  - Compresseurs : bz2 2,28 et lzma 2,58 bits par caractère ; texte mélangé avec Huffman seul : 4,73.
  - Taille de Holmes : 575 794 octets, 562 203 caractères.
- **Figures.** Les 18 PNG ont des axes, des légendes et des unités conformes au texte. `cinq_hypotheses.png`, `barres_lancers.png` (26 faces), `morse_fixe.png` (WATSON), `huffman_arbre.png` (codes) et les barres KL de `cross_entropie.png` ont été reproduits.
- **Flashcards.** Les 67 cartes sont exactes, sauf la carte du constat 4.
- **Cohérence entre chapitres.** Bits, nats et base du logarithme (0B `01_fiche.md:321`, formulaire L55 et L262–267) ; sens de l'intervalle de confiance (ch. 2 `01_fiche.md:269` et ch. 4 L80) ; vraisemblance et posterior (glossaire L40–41, L150, L278) ; learning rate (ch. 1, ch. 5, checkpoint I `03_examen_corrige.md:179-188`, `05_synthese.md:57, 97`) : aucune contradiction.

## Bilan

**Ce qui est solide.** Toutes les valeurs vérifiées que j'ai recalculées sont justes, et les sorties du corrigé concordent avec les solutions. Les figures sont fidèles au texte et les flashcards exactes. Les 18 affirmations datées vérifiées sur le web sont toutes à jour au 2026-10-04 (SciPy 1.15, PyTorch, Optuna, scikit-learn, RFC de zstd et de Brotli, Delétang et coll.). Les conventions transversales (bits ou nats, base du logarithme, gradient, learning rate, vraisemblance, posterior) ne se contredisent pas d'un chapitre à l'autre. Aucune erreur de fond n'atteint l'apprenant : le seul constat MAJEUR est un désaccord entre la bible et deux exercices publiés.

**Les 3 risques principaux.**

1. **Les annexes simplifient plus que la fiche et perdent ses réserves.** C'est le cas des axes et des diagonales, de la sécante symétrique, du sous-gradient et de « la loss oscille » (constats 2, 3, 4 et 9). Qui révise par le formulaire, le glossaire ou les erreurs fréquentes y apprend des règles trop fortes.
2. **Les indices de niveau 3 donnent encore des valeurs vérifiées** pour quatre 🔮, trois ✏️, un 📈 et deux 🏆 (constat 6), malgré les nettoyages des vérifications des ch. 5 et 6. La valeur d'entraînement des prédictions et des défis s'use pour qui ouvre le troisième niveau.
3. **Certains textes ne suivent pas le contenu publié qu'ils commentent.** Des chiffres de durée sont figés dans le texte alors qu'ils changent à chaque exécution (constat 5), et la bible décrit des alphabets que les exercices n'utilisent pas (constat 1). Une reconstruction, ou une session qui suit la bible à la lettre, peut créer de nouvelles incohérences ou changer des réponses publiées.
