# Audit P5 — exactitude des chapitres 9, 10 et 11 (relecteur « exact4 »)

**Périmètre lu en entier** : `01_fiche.md`, `02_exercices.md`, `05_solutions.md` et `flashcards.csv` des ch. 9, 10 et 11 ; `04_indices.md` (papier et quiz en entier, notebook par sondage) ; Markdown et sorties de `05_solutions.ipynb` (extraits par script), Markdown de `03_notebook.ipynb` (comparé à celui des solutions : identique, hors titre et cellules 📝) ; les 18 figures (ouvertes, et leur code dans `tools/chapters/figures_ch09.py` pour les points douteux) ; sections 9 à 11 de `annexes/formulaire.md` (et le tableau des conventions), entrées 9 à 11 de `annexes/glossaire.md` et de `annexes/erreurs_frequentes.md`.

**Recalculé en Python** (scripts dans `exact4/py/`) : toutes les réponses papier (9.1 à 9.7, 9.R2, 9.R3 ; 10.1, 10.2, 10.4 f, 10.5, 10.6 jusqu'à la convergence, 10.7 h, 10.9, 10.R1 à 10.R3 ; 11.1, 11.3, 11.5 à 11.8, 11.R1 à 11.R3), leurs variantes (9.4, 9.5, 9.7, 10.3, 10.5, 10.6), les mini-exemples des trois fiches (MSE/R², features polynomiales, droite des moindres carrés, Ridge, seuillage doux, biais²/variance, early stopping, perceptron, UCB, regret, erreur type, Laplace…), la boutique de 9.12-9.14 (prédiction de 16 h 15 : −1,274, 98,7 % de la MSE ; conditionnements 2,3·10²⁵ et 2,9·10⁶), le seuil α_max de 9.19 (0,129) et la corrélation 0,837, le posterior de `droite_bayes.png` (centre (0,855 ; −0,393), pente surestimée de 1,65 écart-type), le Lasso de 9.6 avec scikit-learn, les comptes de Darapti (8), de Barbara (16) et du moyen terme non distribué (16), les 14 fonctions à seuil de 2 entrées. **Tout concorde** avec les énoncés, les solutions et les sorties exécutées : aucune erreur de calcul trouvée.

**Vérifié sur le web ou dans le code source** (🕰️, au 2026-10-03) : 17 affirmations, détail en fin de rapport ; une seule est dépassée (constat 3).

`git -C /home/claude/workbookia status --short` : vide à la fin de l'audit.

---

## Constats

### MAJEUR

**1. MAJEUR · `chapitres/ch11_raisonnement/02_exercices.md:228` (✏️ 11.3, S3) · exercice à deux réponses défendables**
- Extrait : « **S3.** Aucun modèle linéaire ne calcule XOR. Certains perceptrons sont des modèles linéaires. Donc certains perceptrons ne calculent pas XOR. »
- Problème : la réponse vérifiée de 11.3 b est `"SNSNVNN"` (`tools/chapters/build_ch11.py:289`), donc S3 = **S** (valide **et** prémisses vraies ; `05_solutions.md:95` : « S3 | E, I ⇒ O | valide (Ferio) | vraies | S »). Or le workbook enseigne lui-même que la majeure est fausse dès qu'on ajoute une feature : ∂ 10.4 e (« On ajoute une troisième entrée, le produit $x_3 = x_1 x_2$. Trouve $w_1$, $w_2$, $w_3$ et $b$ qui calculent XOR »), 11.Q2 b (des entrées $x_1^2$, $x_2^2$ rendent un disque représentable), fiche du ch. 9 §9.3 (« un modèle linéaire devient polynomial […] mais elle reste **linéaire en ses poids** »). Le même défaut a d'ailleurs été corrigé dans 11.4 (2) (« Aucune régression linéaire sur les variables brutes (sans produit comme $x_1 x_2$) », écart de la session 22), mais pas dans S3. Un apprenant attentif répond V (`"SNVNVNN"`) : le message reçu est générique, et le corrigé n'en parle pas.
- Correction (énoncé d'un exercice publié, d'où la sévérité ; réponse inchangée) : « **S3.** Aucun modèle linéaire de $x_1$ et $x_2$ seuls (sans feature comme $x_1 x_2$) ne calcule XOR. Certains perceptrons sont des modèles linéaires de $x_1$ et $x_2$. Donc certains perceptrons ne calculent pas XOR. » ; même texte dans `SYLLOGISMS_15` (codage inchangé) ; dans la solution, ajouter à la ligne S3 : « la précision « de $x_1$ et $x_2$ seuls » compte : avec la feature $x_1 x_2$, un modèle linéaire calcule XOR (∂ 10.4 e), et la majeure deviendrait fausse (V) » ; ajouter l'erreur classique `"SNVNVNN"` avec ce message.

### MINEURS

**2. MINEUR · `chapitres/ch10_neurones/02_exercices.md:128-129` (+ `04_indices.md:222, 227`, `05_solutions.md:56-57`, `tools/chapters/build_ch10.py:209, 211`) · precision et recall en français, contre la bible et le ch. 11**
- Extraits : « f) Le rappel (*recall*), avec 2 décimales. » · « g) La précision (*precision*). » ; indices : « Accuracy $7/10$, rappel $3/4$, précision $3/5$ » ; message du notebook : `"c'est la précision : le rappel divise par TP + FN, les vrais +1"`.
- Problème : BIBLE §5 (« accuracy / precision / recall restent en anglais : « précision » est ambigu en français ») ; la fiche du ch. 11, le chapitre suivant, l'écrit noir sur blanc (`01_fiche.md:100` : « ces noms restent en anglais, « précision » serait ambigu »), et le ch. 11 a été corrigé pour cela (§22, session 22). Le ch. 10 est le seul de la partie II à dire « précision » et « rappel ».
- Correction : « f) Le recall, avec 2 décimales. » · « g) La precision. » ; remplacer « rappel » par « recall » et « précision » par « precision » dans les indices, la solution (« Erreur fréquente : échanger le recall et la precision ; le recall regarde les vrais positifs, la precision les prédictions positives ») et les deux messages de `build_ch10.py` (puis reconstruire et réexécuter le notebook, `build_answers.py`).

**3. MINEUR · `chapitres/ch11_raisonnement/01_fiche.md:244` (🕰️ « les modèles qui « raisonnent » ») · encadré dépassé au 2026-10-03, et une source mal liée**
- Extrait : « D'où les approches **neuro-symboliques**, qui confient la déduction à un moteur logique exact et l'intuition à un réseau : AlphaGeometry résout 25 des 30 problèmes de géométrie d'olympiade de son test, près des 25,9 d'un médaillé d'or moyen (Trinh et al., *Nature*, 2024). » ; sources : « [DeepSeek-AI (2025)](https://arxiv.org/abs/2501.12948) ».
- Problème : le chiffre de 2024 est exact (25/30, vérifié dans l'article), mais « Aujourd'hui » ne l'est plus : en juillet 2025, une version de Gemini Deep Think a atteint le niveau d'une médaille d'or à l'OIM 2025 « end-to-end in natural language », sans traduction formelle, alors qu'en 2024 AlphaGeometry et AlphaProof en avaient besoin ([Google DeepMind, 21 juillet 2025](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)) ; AlphaGeometry 2 résout 84 % des problèmes de géométrie de l'OIM des 25 dernières années ([arXiv 2502.03544](https://arxiv.org/abs/2502.03544)). Le « D'où » présente le neuro-symbolique comme la réponse actuelle au débat, ce que 2025 nuance fortement. Par ailleurs, le texte cite « DeepSeek-AI, *Nature*, 2025 » mais le lien pointe vers la prépublication arXiv.
- Correction : après la phrase sur AlphaGeometry, ajouter « En 2025, AlphaGeometry 2 résout 84 % des problèmes de géométrie de l'OIM des 25 dernières années, et une version de Gemini (Deep Think) a atteint le niveau d'une médaille d'or à l'OIM en rédigeant ses preuves en langue naturelle, sans moteur formel : le débat reste ouvert, mais la frontière bouge vite. » ; sources : ajouter les deux liens ci-dessus et remplacer le lien DeepSeek par [DeepSeek-AI, *Nature* 645, 2025](https://doi.org/10.1038/s41586-025-09422-z) (vérifié : publié le 17 septembre 2025, R1-Zero sans démonstrations humaines, vérification et réflexion émergentes).

**4. MINEUR · ch. 11 et annexes · « erreur-type » au lieu du terme retenu « erreur type »**
- Occurrences : `chapitres/ch11_raisonnement/01_fiche.md:9, 200, 292, 309` ; `02_exercices.md:281, 282, 294, 295` ; `04_indices.md:383, 837` ; `05_solutions.md:123, 235, 279, 286, 288` ; `flashcards.csv:18` ; `tools/chapters/build_ch11.py` (≈ 25 lignes, dont 347-360, 1061-1194, 2030-2176 : énoncés et messages des deux notebooks) ; `annexes/formulaire.md:380` ; `annexes/erreurs_frequentes.md:261, 271` ; `suivi/auto_evaluation.md:222`.
- Problème : le glossaire (`annexes/glossaire.md:231`, « erreur type | standard error »), le ch. 8 (12 occurrences dans la fiche), le formulaire du ch. 8 (`formulaire.md:321`), le checkpoint II et MP2 écrivent « erreur type » ; le §22 (session 23) a aligné la synthèse du CP2 sur la forme sans trait d'union. Deux graphies pour la même notion, à deux chapitres d'écart (BIBLE §5 : s'en tenir au terme retenu).
- Correction : remplacer partout « erreur-type » par « erreur type » et « erreurs-types » par « erreurs types » (fichiers ci-dessus, puis reconstruction du notebook du ch. 11 et `build_answers.py`, les messages faisant partie des empreintes).

**5. MINEUR · `annexes/glossaire.md:410`, `chapitres/ch09_overfitting/01_fiche.md:52, 306, 307` · « les poids de la meilleure epoch », contre la règle du workbook**
- Extraits : glossaire « arrêter l'entraînement quand l'erreur de validation ne s'améliore plus, puis reprendre les poids de la meilleure epoch » ; fiche (L'essentiel 3) « il garde les poids de la meilleure epoch de validation » ; pièges : « une patience, et les poids de la meilleure epoch » et « recharger la meilleure epoch (`restore_best_weights=True` dans Keras) ».
- Problème : la règle fixée au §22 (et dans la fiche §9.4, le formulaire `formulaire.md:344`, la flashcard 10, la synthèse du CP2) est « on recharge les poids de la **dernière amélioration** (la meilleure epoch quand `min_delta` = 0) ». Les exercices 9.5 g (poids de l'epoch 4 alors que la meilleure est la 5ᵉ) et 9.20 d (epoch 37, pas 56) testent précisément le cas où les deux diffèrent : l'apprenant qui révise par le glossaire ou « L'essentiel » apprend la mauvaise règle. Keras, avec `restore_best_weights=True` et un `min_delta` > 0, rend aussi les poids de la dernière amélioration (code source de `EarlyStopping` relu : `best_weights` n'est mis à jour qu'après `_is_improvement`).
- Correction : glossaire « … puis reprendre les poids de la dernière amélioration (ceux de la meilleure epoch quand `min_delta` = 0) » ; fiche l. 52 « il recharge les poids de la dernière amélioration (la meilleure epoch avec `min_delta` = 0, le réglage par défaut) » ; l. 306 « une patience, et les poids de la dernière amélioration » ; l. 307 « recharger les poids de la dernière amélioration (`restore_best_weights=True` dans Keras) ».

**6. MINEUR · `chapitres/ch09_overfitting/01_fiche.md:57` (L'essentiel 8) · « sans toucher au biais »**
- Extrait : « Mais ce n'est pas une loi : plus de données fait baisser la variance sans toucher au biais, et la **double descente** … »
- Problème : faux en général, et contraire à la convention du chapitre : `Ridge` minimise une **somme** de carrés plus $\alpha\lVert \mathbf{w}\rVert^2$, si bien qu'à `alpha` fixé, plus d'exemples réduit le poids relatif de la pénalité, donc le biais (c'est le cas de `PolyRidge`, 9.18 à 9.29) ; même les moindres carrés mal spécifiés ont un biais qui dépend un peu de $n$. La fiche le dit correctement ailleurs (l. 267 : « font baisser la variance **sans** faire monter le biais » ; flashcard 23 ; solution de 9.Q10 d).
- Correction : « plus de données fait baisser la variance sans faire monter le biais ».

**7. MINEUR · `chapitres/ch09_overfitting/flashcards.csv:21` · une décomposition sans son repère**
- Extrait : « Le bruit est un plancher ; l'erreur du modèle moyen ne contient que le biais² (idée du bagging). »
- Problème : la carte vient d'écrire l'erreur attendue sur une **nouvelle mesure bruitée**, où l'erreur du modèle moyen vaut biais² + bruit, pas le biais² seul. La fiche fait la distinction (`01_fiche.md:245` : « face à $f$, le modèle moyen $\bar{f}$ n'a plus que le biais² (face à de nouvelles mesures, biais² + bruit) »).
- Correction : « Le bruit est un plancher ; face à \(f\), l'erreur du modèle moyen se réduit au biais² (face à de nouvelles mesures : biais² + bruit) : c'est l'idée du bagging. »

**8. MINEUR · `chapitres/ch09_overfitting/01_fiche.md:241`, `annexes/glossaire.md:280, 437` · trois « biais », deux seulement reliés**
- Extraits : fiche « Ne confonds pas ce biais avec le biais d'un neurone (le nombre ajouté à sa somme pondérée, ch. 10), qui porte le même nom en français. » ; glossaire, « biais (statistique) » : « … rien à voir avec le biais d'un neurone » ; « biais (d'une pièce) » : « … rien à voir avec le biais d'un neurone ni avec un biais d'échantillonnage ».
- Problème : le sens le plus proche dans le temps, le « biais θ d'une pièce » (probabilité de face, ch. 4), n'est mentionné ni par l'encadré du ch. 9 ni par l'entrée « biais (statistique) », et l'entrée « biais (d'une pièce) » ne renvoie pas au biais d'un estimateur. Or le ch. 11 réemploie les deux sens à quelques pages d'écart (11.R3 : « le biais $\theta$ d'une pièce » ; §11.7 : « chaque bras est une pièce biaisée ») juste après que le ch. 9 a défini le biais comme $\mathbb{E}[\hat{\theta}] - \theta$ (et 9.R3 parle du « biais de cet estimateur »).
- Correction : fiche l. 241 « … ni avec le « biais » d'une pièce du ch. 4, qui est sa probabilité de tomber sur face ($\theta$) : trois sens pour un même mot. » ; glossaire 437 « … rien à voir avec le biais d'un neurone ni avec le biais d'une pièce (ch. 4) » ; glossaire 280 « … ni avec un biais d'échantillonnage, ni avec le biais statistique d'un estimateur (ch. 9) ».

**9. MINEUR · `annexes/glossaire.md:466` · théorème du cycle du perceptron énoncé trop fort**
- Extrait : « sur des données non séparables, les poids du perceptron restent bornés et repassent par les mêmes valeurs (Block et Levin, 1970) ».
- Problème : le théorème garantit que les poids restent **bornés** ; le retour périodique aux mêmes valeurs n'est garanti que pour des entrées entières (ou rationnelles), comme XOR en 10.13. Avec des mesures réelles (les manchots de 10.19 E), rien ne dit que les poids repassent exactement par les mêmes valeurs. La solution de 10.13 dit d'ailleurs seulement « Les poids restent bornés ».
- Correction : « sur des données non séparables, les poids du perceptron restent bornés (Block et Levin, 1970) ; avec des entrées entières, ils finissent par repasser périodiquement par les mêmes valeurs (XOR, 10.13) ».

**10. MINEUR · `chapitres/ch00b_maths/01_fiche.md:106` · la fonction signe annoncée pour les perceptrons**
- Extrait : « La **fonction signe** vaut $\mathrm{sign}(x) = -1$ si $x < 0$, $0$ si $x = 0$, $1$ si $x > 0$. […] Elle reviendra avec la régularisation L1 (ch. 9) et les perceptrons (ch. 10). »
- Problème : le ch. 10 enseigne l'inverse : « ce n'est pas `np.sign`, qui renvoie 0 en 0 » (`ch10/01_fiche.md:102`), et le premier piège de son tableau est « utiliser `np.sign` comme seuil » (`:172`) ; la synthèse du CP2 le reprend (`05_synthese.md:79`). L'annonce de 0B prépare exactement l'erreur que le ch. 10 corrige.
- Correction : « Elle reviendra avec la régularisation L1 (ch. 9) ; le perceptron (ch. 10) en utilise une variante qui vaut −1 en 0 (`sign_step`), car `np.sign(0)` vaut 0. »

**11. MINEUR · `chapitres/ch10_neurones/05_solutions.md:12` (🧠 10.Q1 a) · la régression linéaire, neurone ou pas ?**
- Extrait : « a) **ABD** : le k-means calcule des moyennes et des distances, la régression linéaire résout un système d'équations, l'arbre de décision enchaîne des questions sur les variables ; aucun neurone. »
- Problème : la fiche du même chapitre identifie le biais d'un neurone à « l'ordonnée à l'origine (*intercept*) de la régression du ch. 9 » (`01_fiche.md:138`), et un neurone à activation identité calcule exactement $\mathbf{w}\cdot\mathbf{x} + b$ : un apprenant qui répond « AD » (« une régression linéaire est un neurone linéaire ») a un argument. Le message de `wb.check` pour `"AD"` (`build_ch10.py:81`, « la régression par moindres carrés est une formule (ch. 9) ») et la solution répondent « formule », sans reconnaître la ressemblance. La réponse se défend avec la lecture de la fiche (§10.1 : « calculent leur résultat par une formule […] sans aucun neurone »), mais il faut le dire.
- Correction : ajouter à la solution « La régression linéaire a la **forme** d'un neurone à activation identité ($\hat{y} = \mathbf{w}\cdot\mathbf{x} + b$, fiche §10.3.3) ; mais elle se calcule par une formule, sans réseau ni règle d'apprentissage de neurone : c'est le sens de « n'utilise aucun neurone » (livre §10.1). » ; message de `"AD"` : « la régression par moindres carrés a la forme d'un neurone linéaire, mais elle se calcule par une formule (ch. 9), sans neurone : relis la fiche §10.1 ».

**12. MINEUR · `chapitres/ch09_overfitting/01_fiche.md:7, 15, 333` et `chapitres/ch11_raisonnement/01_fiche.md:7, 15, 317` · nombre de pages incohérent avec la plage citée**
- Extraits : ch. 9 « p. 338-373 » puis « Ses 35 pages » et « 35 pages, vingt figures » (338 à 373 font 36 pages) ; ch. 11 « p. 394-430 » puis « ses 36 pages » et « 36 pages, douze figures » (394 à 430 font 37 pages). Les ch. 5, 6, 8 et 10 sont cohérents (26, 34, 27, 18).
- Correction : vérifier contre le livre et aligner (« 36 pages » au ch. 9 et « 37 pages » au ch. 11 si les plages sont justes, ou corriger la plage).

**13. MINEUR · `chapitres/ch10_neurones/04_indices.md` et `chapitres/ch11_raisonnement/04_indices.md` (indices de niveau 3 des quiz, rappels et papier) · l'indice 3 donne les réponses finales**
- Exemples : ch. 10, l. 61 (10.Q3) « Vrai, faux, faux, vrai, vrai. » ; ch. 11, l. 61 (11.Q3) « a) Vrai. b) Vrai. c) Faux : `while True: pass`. d) Faux. e) B […] » ; l. 303 (11.1) « a) 256. b) 255. c) $-128$. d) […] = 10 » ; l. 339 (11.3) « b) S1 : S ; S2 : N ; S3 : S ; S4 : N ; S5 : V ; S6 : N ; S7 : N. c) S2 : A ; S4 : B ; S6 : A ; S7 : D. »
- Problème : BIBLE §12 : « Indice 3 (presque la solution : pseudo-code ou première ligne) ». Au ch. 9, l'indice 3 pose le calcul ou la méthode sans la valeur (9.1 : « corrige la somme des carrés et la somme des valeurs absolues, puis divise par 5 » ; 9.5, 9.Q1 à Q11 de même), et les ch. 4 à 8 s'arrêtent le plus souvent au calcul posé ; aux ch. 10 et 11, l'indice 3 recopie, sous-question par sous-question, les réponses que vérifie la partie 0, qui ne vérifie alors plus rien pour qui ouvre le troisième niveau. Dérive de style entre chapitres voisins (à croiser avec l'audit de structure).
- Correction : aux ch. 10 et 11, réécrire les indices 3 des exercices à réponse courte sur le modèle du ch. 9 : le calcul posé pour chaque sous-question (« d) $\lceil \log_2 \dots \rceil$ », « S3 : inventaire des termes distribués : … »), sans la valeur finale ni la lettre ; les valeurs restent dans `05_solutions.md`.

**14. MINEUR · `chapitres/ch09_overfitting/figures/biais_variance.png`, panneau (c) (`tools/chapters/figures_ch09.py:295-311`) · axe vertical sans titre**
- Problème : le panneau trace biais², variance et erreur totale en échelle logarithmique (de 10⁻³ à 1) sans titre d'axe vertical ; seul le titre du panneau dit « Erreur attendue sur une nouvelle mesure ». Les autres figures du chapitre nomment leurs deux axes.
- Correction : après `ax = axes[2]`, ajouter `ax.set_ylabel("erreur quadratique (échelle log)")`, puis régénérer la figure.

### SUGGESTIONS

**15. SUGGESTION · `chapitres/ch11_raisonnement/01_fiche.md:115` (🕰️ No Free Lunch) · TabPFN a progressé**
- Extrait : « en 2025, TabPFN […] a dépassé les arbres boostés sur des jeux d'au plus 10 000 exemples (Hollmann et al., *Nature*, 2025). » (exact : vérifié dans l'article, publié le 8 janvier 2025).
- Proposition : ajouter « ; sa version 2.5 (novembre 2025) monte à 50 000 exemples et 2 000 features » ([Grinsztajn et al., arXiv 2511.08667](https://arxiv.org/abs/2511.08667)), ce qui renforce le « ces positions bougent ».

**16. SUGGESTION · `chapitres/ch09_overfitting/01_fiche.md:229` (🕰️ régularisation des réseaux) · AdamW n'est plus seul**
- Extrait : « d'où **AdamW**, l'optimiseur par défaut de la plupart des grands modèles (`torch.optim.AdamW`, `weight_decay=0.01` par défaut dans PyTorch 2.11, alors qu'`Adam` a `weight_decay=0`) » (valeurs exactes, relues dans la signature de PyTorch 2.11).
- Proposition : ajouter « ; des concurrents apparaissent, comme Muon (Kimi K2, 2025, entraîné avec MuonClip : [arXiv 2507.20534](https://arxiv.org/abs/2507.20534)) » — c'est justement le modèle de 10¹² paramètres dont 3,2·10¹⁰ actifs de 🧮 10.9 —, et signaler que `torch.optim.Adam(..., decoupled_weight_decay=True)` (PyTorch 2.11) équivaut à AdamW.

**17. SUGGESTION · `chapitres/ch09_overfitting/01_fiche.md:54`, `flashcards.csv:17`, `annexes/glossaire.md:417` · « Ridge … sans les annuler »**
- Extraits : « La pénalité **L2** (Ridge) rétrécit les poids dans leur ensemble sans les annuler » ; flashcard « la norme des poids diminue, sans qu'aucun ne s'annule exactement ».
- Proposition : 🔮 9.19 montre trois coefficients Ridge qui changent de signe, donc passent par 0 en un point du chemin ; écrire « sans en mettre à zéro sur toute une plage de $\lambda$ (un poids qui change de signe ne s'annule qu'en un point, 9.19) », ou « en général sans les annuler ».

**18. SUGGESTION · `chapitres/ch11_raisonnement/01_fiche.md:79` · l'« évaluation » du perceptron**
- Extrait : « | **Évaluation** | Comment juge-t-on une solution ? | le nombre d'exemples mal classés | … »
- Proposition : le ch. 10 présente la règle comme une descente de sous-gradient sur $\max(0, -y z)$ (`ch10/01_fiche.md:114`) et dit qu'« il n'optimise rien sur des données non séparables » (`ch10/05_solutions.md:202`) : il ne minimise pas le nombre d'erreurs (c'est le rôle de l'algorithme pocket). Écrire « les exemples mal classés ($y z \le 0$), via la perte $\max(0, -y z)$ ».

**19. SUGGESTION · `annexes/formulaire.md:371` et `:382`, `checkpoints/partie_2/05_synthese.md:57` · le symbole α**
- Problème : au formulaire, la légende du ch. 11 ne définit pas α, qui y désigne le pas constant (« pas constant : $Q_{n+1} = (1-\alpha)^n Q_1 + \dots$ »), alors que deux sections plus haut α est la force de Ridge et du Lasso ; la synthèse du CP2 mélange α et λ dans la même ligne (« Ridge | $(\mathbf{X}_c^\top\mathbf{X}_c + \alpha\,\mathbf{I})\,\mathbf{w} = \dots$ ; en 1D $w^* = \frac{S_{xy}}{S_{xx} + \lambda}$ »).
- Proposition : légende du ch. 11 : « … ; $\alpha$ : le pas constant d'une estimation (sans rapport avec l'`alpha` de Ridge, ch. 9) » ; CP2 : une seule lettre dans la ligne 16 (« $S_{xx} + \alpha$ »).

---

## Affirmations 🕰️ et datées vérifiées (sources consultées le 2026-10-03)

| Affirmation (fichier) | Verdict | Source |
|---|---|---|
| Keras `EarlyStopping` : `patience=0`, `restore_best_weights=False` ; meilleurs poids mis à jour seulement à une amélioration (ch. 9 §9.4) | exact | [code source Keras](https://github.com/keras-team/keras/blob/master/keras/src/callbacks/early_stopping.py) |
| Lightning `EarlyStopping(monitor, min_delta=0.0, patience=3, mode="min")`, sans restauration des poids (ch. 9) | exact | [code source Lightning](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/callbacks/early_stopping.py) |
| scikit-learn 1.6 : `HistGradientBoosting*` `early_stopping="auto"` au-delà de 10 000 exemples ; `MLPClassifier`/`SGDClassifier` `validation_fraction=0.1`, `n_iter_no_change` 10/5 ; `LearningCurveDisplay` 1.2, `ValidationCurveDisplay` 1.3 | exact | docstrings de scikit-learn 1.6.1 installé |
| PyTorch 2.11 : `AdamW(weight_decay=0.01)`, `Adam(weight_decay=0)` (ch. 9) | exact (et `Adam` a désormais `decoupled_weight_decay`, constat 16) | signatures de torch 2.11.0 installé |
| Curth, Jeffares, van der Schaar, NeurIPS 2023 : la seconde descente des méthodes classiques redevient un U avec un nombre effectif de paramètres (ch. 9) | exact | [arXiv 2310.18988](https://arxiv.org/abs/2310.18988) |
| FlyWire 2024 : 139 255 neurones, 54,5 millions de synapses (ch. 10, flashcard 3) | exact | [Dorkenwald et al., *Nature* 634](https://www.nature.com/articles/s41586-024-07558-y) |
| MICrONS 2025 : plus de 200 000 cellules, 0,5 milliard de synapses (ch. 10) | exact | [*Nature* 640, 9 avril 2025](https://www.nature.com/articles/s41586-025-08790-w) |
| Hala Point : 1 152 Loihi 2, 1,15 milliard de neurones, 128 milliards de synapses, 2 600 W (ch. 10) | exact | [Intel, 17 avril 2024](https://www.intc.com/news-events/press-releases/detail/1691/intel-builds-worlds-largest-neuromorphic-system-to) |
| SpiNNcloud, TU Dresde, 2025 : 35 000 puces SpiNNaker2 (ch. 10) | exact | [TU Dresden, 14 avril 2025](https://tu-dresden.de/ing/der-bereich/news/milestone-for-energy-efficient-ai-systems-tud-launches-spinncloud-supercomputer) |
| Darwin Monkey, université du Zhejiang, août 2025, plus de 2 milliards de neurones (ch. 10) | exact | [ZJU, 26 août 2025](https://www.zju.edu.cn/english/2025/0910/c19573a3079424/page.htm) |
| Rosenblatt : perceptron simulé sur IBM 704 au Cornell Aeronautical Laboratory ; Mark I à 400 photocellules, 512 unités d'association, 8 réponses (ch. 10, 10.11) | exact | [Wikipédia, Perceptron](https://en.wikipedia.org/wiki/Perceptron) et [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) |
| GSM-Symbolic (ICLR 2025) : chute jusqu'à 65 % avec une phrase inutile (ch. 11) | exact | [arXiv 2410.05229](https://arxiv.org/abs/2410.05229) |
| AlphaGeometry : 25 des 30 problèmes (ch. 11) | exact en 2024, **dépassé** comme « aujourd'hui » (constat 3) | [*Nature* 2024](https://www.nature.com/articles/s41586-023-06747-5) ; [DeepMind 2025](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) ; [AlphaGeometry 2](https://arxiv.org/abs/2502.03544) |
| DeepSeek-R1 (*Nature*, 2025), R1-Zero sans exemples humains, auto-vérification émergente (ch. 11) | exact (lien à corriger, constat 3) | [*Nature* 645](https://www.nature.com/articles/s41586-025-09422-z) |
| TabPFN (*Nature*, 2025) : meilleur que les arbres boostés jusqu'à 10 000 exemples (ch. 11) | exact (complément, constat 15) | [*Nature*, 8 janvier 2025](https://www.nature.com/articles/s41586-024-08328-6) |
| Li et al. 2010 : 33 millions d'événements, +12,5 % de clics (ch. 11) | exact | [arXiv 1003.0146](https://arxiv.org/abs/1003.0146) |
| InstructGPT (RLHF) 1,3 milliard de paramètres préféré à GPT-3 175 milliards (ch. 11) | exact | [arXiv 2203.02155](https://arxiv.org/abs/2203.02155) |

---

## Bilan

**Ce qui est solide.** Les trois chapitres sont exacts sur le fond : formules (moindres carrés, Ridge et Lasso avec leurs conventions d'échelle, seuillage doux et descente de coordonnées, décomposition biais² + variance + bruit, règle `min_delta`, posterior et MAP = Ridge ; règle du perceptron, preuve de Novikoff, XOR ; règles de distribution, Venn et force brute, Laplace, moyenne incrémentale, ε-greedy, UCB1, Thompson, pseudo-regret), mini-exemples, réponses papier et variantes, chiffres des solutions contre les sorties exécutées : tout a été recalculé sans écart. Les figures sont cohérentes avec le texte (les optima L1/L2, la vraisemblance « en bande », le centre du posterior, les panneaux de 9.9, la figure de Thompson). Les encadrés ⚠️ sur le livre sont justes et bien sourcés, et 16 des 17 affirmations datées vérifiées tiennent au 2026-10-03.

**Les trois risques principaux.**
1. **Une réponse vérifiée contestable** (constat 1) : S3 de ✏️ 11.3 attend « solide » pour une prémisse que le workbook réfute lui-même deux chapitres plus tôt ; c'est le seul point qui touche un contrat.
2. **Une terminologie qui dérive d'un chapitre à l'autre** : precision/recall francisés au ch. 10 (constat 2), « erreur-type » au ch. 11 (constat 4), trois sens de « biais » mal reliés (constat 8), « meilleure epoch » contre « dernière amélioration » (constat 5) — chacun bénin, mais l'apprenant révise justement avec le glossaire, le formulaire et les flashcards, où ces écarts se cumulent.
3. **Des indices de niveau 3 qui donnent les réponses** aux ch. 10 et 11 (constat 13), et un encadré 🕰️ sur le raisonnement des modèles déjà daté par l'OIM 2025 (constat 3) : la valeur d'entraînement de la partie 0 et l'actualité des encadrés s'usent plus vite que le reste.
