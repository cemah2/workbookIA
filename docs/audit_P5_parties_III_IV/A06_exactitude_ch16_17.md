# A6 · Exactitude des ch. 16 et 17 — rapport (audit P5, parties III et IV)

Périmètre : `chapitres/ch16_feedforward`, `chapitres/ch17_activations`, leurs lignes dans `annexes/` (formulaire, glossaire, erreurs fréquentes, cheatsheets) et `data/cards/mnist.md`, cohérence avec les ch. 18 à 20. Commit audité : `5a17b91`. Scripts de vérification dans ce dossier : `verif16.py`, `verif17.py` (toutes les réponses papier recalculées), `twins.py`, `twins2.py` (constat 1), `nbdump.py` (extraction des cellules des notebooks).

## Méthode

- Fiches lues en entier et comparées au texte du livre (vol. 1, p. 655-702) : chaque renvoi de page, chaque encadré « Le livre : » et « Le livre, corrigé ».
- Toutes les réponses courtes des quiz, rappels, ✏️, vérifications de ∂ et 🧮 recalculées à partir des énoncés (`verif16.py`, `verif17.py`), puis comparées à `05_solutions.md` et aux `wb.record` des notebooks de solutions exécutés : **aucune réponse enregistrée fausse**. Démarches, erreurs fréquentes chiffrées et variantes recalculées (16.1 à 16.8, 16.10, 16.14, 16.15, 16.19, 16.20, 16.22, 16.23, 17.1 à 17.8, 17.10, 17.13, 17.15 à 17.19, 17.21, 17.24) : justes, sauf le constat 2.
- Preuves ∂ 16.6, 16.7, 17.3, 17.4, 17.7 relues ligne à ligne : justes, hypothèses bien posées (poids indépendants **du vecteur** des entrées et centrés, produits décorrélés même pour des entrées corrélées ; échange $\tau$ ; $z$ symétrique pour le facteur 1/2).
- Conventions de PyTorch rejouées avec torch 2.11 CPU : `relu'(0) = 0`, `leaky_relu'(0) = negative_slope`, `elu'(0) = alpha`, `abs'(0) = 0`, `softplus` linéaire seulement si `beta*z > threshold`, `nn.Linear.reset_parameters` = `kaiming_uniform_(a=√5)` (variance mesurée 3,35e-4 pour $1/(3 \times 1000) = 3{,}33$e-4), docstrings de `CrossEntropyLoss` et `LSTM`. scikit-learn 1.6.1 : `_init_coef` (facteur 6, ou 2 pour `logistic`, biais tirés), `coefs_`, `n_layers_ = 4`, `out_activation_`. transformers 5.18 (installé) : `_init_weights` de GPT-2 ($\mathcal{N}(0, 0{,}02^2)$, `c_proj` à $0{,}02/\sqrt{2 N}$), `gelu_new`, `hidden_act="silu"` de LLaMA.
- Les 18 + 22 flashcards relues : justes (voir la suggestion 13).
- 15 affirmations 🕰️ (ou de type 🕰️) revérifiées sur le web ou dans le code source des versions figées : liste à la fin.

## Constats

### 1. MAJEUR · 🔮 16.19 a) : « exactement jumeaux » dépend de l'ordre des additions en virgule flottante

- `chapitres/ch16_feedforward/03_notebook.ipynb` (énoncé, produit par `tools/chapters/build_ch16.py:1199`) : « a) `twins_19` : après 200 pas, les quatre neurones partis de 0,5 sont-ils encore identiques (`spread_19` exactement nul) ? `True` ou `False` » ; réponse enregistrée `True` (`build_ch16.py:1208`, `wb.record("16.19a", True, …)`).
- `05_solutions.md:241` : « partis de 0,5, les quatre neurones restent exactement jumeaux (écart 0), même avec des différences finies » ; `05_solutions.md:243` (erreurs fréquentes) : « croire que les erreurs d'arrondi des différences finies briseront la symétrie (elles sont symétriques elles aussi) » ; note de `build_ch16.py:1220` : « restent **exactement** jumeaux (écart nul) ».
- **Problème.** L'expérience appelle la `mlp_forward` **de l'apprenant**. Mathématiquement, la symétrie est exacte (∂ 16.6) ; en virgule flottante, elle ne l'est que si la somme $\sum_k v_k h_k$ de la couche de sortie est calculée dans un ordre symétrique. Perturber $v_1$ ou $v_3$ change un terme différent d'une somme séquentielle $((t_0 + t_1) + t_2) + t_3$, et l'arrondi n'est pas le même. Le point symétrique étant **instable** (c'est la leçon de c)), ces écarts d'un ulp grandissent. Rejoué sur le même protocole (`twins2.py`, données `make_xor(n=200, noise=0.15, seed=1619)`, 200 pas, lr 0,5, différences centrées $h = 10^{-6}$) avec quatre `dense_forward` justes (toutes passent les tests, qui comparent avec `allclose`) :
  ```
  matmul  (x @ W)                          spread 0.0
  einsum                                   spread 0.0
  (x[:, :, None] * W[None]).sum(axis=1)    spread 2.13e-06
  boucle sur les entrées j                 spread 2.13e-06
  ```
  Un apprenant qui a écrit sa couche par diffusion ou par boucle (ou, plus rarement, une autre machine dont le noyau BLAS somme dans un autre ordre) voit `spread 0.00e+00 -> 2.13e-06` à l'écran, alors que sa prédiction `True` reçoit ✅ et que la solution affirme « écart 0 ». L'affirmation des erreurs fréquentes (« elles sont symétriques elles aussi ») est fausse en virgule flottante.
- **Correction proposée** (énoncé et réponse d'un exercice publié : contrat). Énoncé a) : « …sont-ils encore identiques **à $10^{-5}$ près** (`spread_19 < 1e-5`) ? ». Solution : « partis de 0,5, les quatre neurones restent jumeaux : l'écart, nul en arithmétique exacte (∂ 16.6), reste au plus de l'ordre de $10^{-6}$ (selon l'ordre dans lequel ta couche additionne ses produits, l'arrondi peut créer un écart d'un ulp, que l'instabilité du point symétrique amplifie un peu) ». Erreur fréquente : « croire que les différences finies traitent deux jumeaux différemment : elles perturbent chaque paramètre de la même façon ; seuls les arrondis, de l'ordre de $10^{-16}$, peuvent les distinguer ». Garder `wb.check` booléen.

### 2. MINEUR · 17.13, variante : `tanh(19)` n'est pas « exactement 1 » pour la raison donnée

- `chapitres/ch17_activations/05_solutions.md:212` : « `tanh(19)` vaut déjà exactement **1.0** en `float64` (l'écart $1 - \tanh 19 \approx 6 \times 10^{-17}$ est sous la précision), et `tanh_derivative(19)` vaut donc **0.0** ».
- **Problème.** $1 - \tanh 19 = 6{,}3 \times 10^{-17}$ dépasse le demi-ulp sous 1 ($5{,}55 \times 10^{-17}$) : la valeur correctement arrondie est `0.9999999999999999`. C'est ce que rendent `math.tanh(19)` et `torch.tanh` en `float64` ; seule l'implémentation de `np.tanh` rend `1.0` :
  ```
  np.tanh(19) = 1.0     math.tanh(19) = 0.9999999999999999     torch.tanh(19., float64) == 1.0 -> False
  ```
  Un apprenant qui vérifie avec `math.tanh` ou PyTorch obtient une dérivée de $2{,}2 \times 10^{-16}$, pas 0, et l'explication est fausse.
- **Correction** : « `tanh(20)` vaut exactement **1.0** en `float64` (l'écart $1 - \tanh 20 \approx 8{,}5 \times 10^{-18}$ est sous le demi-ulp de 1, $5{,}6 \times 10^{-17}$), et `tanh_derivative(20)` vaut donc **0.0** » (vérifié : `np.tanh`, `math.tanh` et `torch.tanh` rendent tous 1.0 en 20).

### 3. MINEUR · Fiche 16, tableau synchrone/asynchrone : l'horloge de parquet n'est pas « opposée » au convoyeur

- `chapitres/ch16_feedforward/01_fiche.md:93` : « un convoyeur d'usine qui avance par à-coups (le livre l'oppose à l'horloge de parquet qui bat la mesure) ».
- **Problème.** Dans le livre (p. 662), l'horloge de parquet est l'image de l'**horloge maître** d'un système synchrone (« a master clock somewhere, ticking along forever like a grandfather clock ») ; le convoyeur est l'exemple de ce même système. Rien n'est opposé.
- **Correction** : « un convoyeur d'usine qui avance par à-coups (le livre compare l'horloge maître à une horloge de parquet) ».

### 4. MINEUR · Fiche 16 : justice envers le livre sur l'échelle fixe ±0,05

- `01_fiche.md:176` (« Le livre, corrigé — l'influence de l'initialisation ne s'efface pas toujours ») : « Une échelle fixe comme $\pm 0{,}05$, qui ne dépend pas de la largeur des couches, est justement ce qu'il ne faut pas faire dans un réseau profond » ; `01_fiche.md:204` (🕰️) : « **Le livre :** cite les règles de LeCun, Glorot et He, et un exemple uniforme sur $\pm 0{,}05$, sans dire ce que font les bibliothèques » ; énoncé de 📈 16.22 : « $\mathcal{N}(0 ;\ 0{,}05^2)$ (une échelle fixe, comme l'exemple du livre) ».
- **Problème.** Le livre dit lui-même (p. 669) : « We rarely use uniform or normal initialization directly, because we need to pick their parameters », et renvoie aux règles nommées ; il écrit aussi (p. 667) que ces méthodes « are supported by most machine-learning libraries » et (p. 670) que les détails sont « noted in the documentation ». La fiche ne le mentionne nulle part (aucune occurrence de « rarement » à ce sujet) : le lecteur peut croire que le livre recommande ±0,05.
- **Correction** : ajouter à la fin de l'encadré de la ligne 176 : « Le livre le sait, d'ailleurs : il note qu'on emploie rarement ces lois telles quelles, faute de savoir choisir leurs paramètres (p. 669), et renvoie aux règles nommées ; ce qui lui manque, c'est la raison de ces règles. » ; ligne 204 : « …sans dire ce que font les bibliothèques **par défaut** ».

### 5. MINEUR · Paraphrases proches du livre (§3), fiche 17

- `chapitres/ch17_activations/01_fiche.md:172` : « Loin de 0 (le livre cite $|z| > 6$ environ ; tanh, plus raide, plafonne plus tôt), leur sortie est presque constante et leur dérivée presque nulle : on dit qu'elles **saturent**. […] ReLU souffre du même mal pour $z < 0$ (une pente exactement nulle), mais pas pour $z > 0$, où sa pente reste 1 ».
  Livre p. 696 : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session)
  La dernière phrase suit le livre mot à mot (« souffre du même mal » = « suffers the same problem », même structure).
  **Correction** : « Pour ReLU, le côté négatif est pire encore (une pente exactement nulle), mais le côté positif ne s'aplatit jamais : sa pente y vaut 1 quel que soit $z$. »
- `01_fiche.md:101` : « La couche de **sortie** fait exception (p. 681-682) : aucune couche ne la suit, elle ne peut donc s'effondrer avec personne » ; livre p. 682 : « … » (citation en anglais retirée de la version versionnée ; texte complet dans le rapport brut de la session) **Correction** : « La couche de sortie est l'exception (p. 681-682) : rien ne vient après elle, et l'effondrement ne concerne que deux couches qui se suivent. »

### 6. MINEUR · Notebook de solutions du ch. 17 : un `DeprecationWarning` affiché dans la partie 0

- `chapitres/ch17_activations/05_solutions.ipynb`, cellule des outils de la partie 0, sortie conservée : « DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future […] `B_EQ_171 = float((B1_171 @ W2_171 + B2_171) @ W3_171 + B3_171)` » ; source `tools/chapters/build_ch17.py:64`.
- **Problème.** `(…) @ W3_171` a la forme `(1,)` ; `float()` d'un tableau 1-D est déprécié (NumPy 1.25) et lèvera une erreur dans une version future de NumPy (le notebook de solutions cesserait de s'exécuter). La sortie affichée trouble aussi l'apprenant qui lit le corrigé.
- **Correction** (`build_ch17.py:64`) : `B_EQ_171 = float(((B1_171 @ W2_171 + B2_171) @ W3_171).item() + B3_171)`, puis réexécuter `05_solutions.ipynb`.

### 7. MINEUR · `annexes/erreurs_frequentes.md:100` : la forme du symptôme est fausse

- Extrait : « une couche dense donne un tableau `(n, n)` au lieu de `(n, n_out)`, ou des valeurs fausses sans erreur | biais écrit en colonne (`b.reshape(-1, 1)`) ».
- **Problème.** `(n, n_out) + (n_out, 1)` ne se diffuse que si $n = n_{out}$ (forme `(n, n_out)`, valeurs fausses) ou $n = 1$ (forme `(n_out, n_out)`) ; ailleurs, NumPy lève une erreur. Le tableau `(n, n)` n'apparaît jamais comme forme « inattendue » (pour $n = n_{out}$, c'est la bonne forme). C'est 16.20 qui le dit bien : « une image seule, `(1, 64) + (64, 1)`, donnerait un tableau `(64, 64)` au lieu de `(1, 64)` ».
- **Correction** : « une couche dense donne un tableau `(n_out, n_out)` pour un seul exemple au lieu de `(1, n_out)`, ou des valeurs fausses sans erreur quand le batch a `n_out` exemples ».

### 8. MINEUR · `annexes/cheatsheets/pytorch.md:44` : renvoi vers un détail qui n'existe pas

- Extrait : « `nn.Threshold(t, v)` n'est **pas** le seuil du perceptron : il garde $x$ si $x > t$, et met `v` ailleurs (ch. 10, détails au ch. 17) ».
- **Problème.** Aucun fichier de `chapitres/ch17_activations/` ne mentionne `Threshold` (grep vide).
- **Correction** : « (ch. 10) », ou ajouter une phrase dans la fiche 17, §17.4 : « Attention : `torch.nn.Threshold(t, v)` n'est pas une marche, il garde $z$ au-dessus de $t$ et met `v` ailleurs. »

### 9. MINEUR · Glossaire : termes des ch. 16 et 17 absents

- `annexes/glossaire.md` n'a pas d'entrée pour **SwiGLU** (fiche 16 l. 68, fiche 17 l. 58, 190, 212, 268, flashcards, 17.E4), **échantillonneur-bloqueur** (*sample-and-hold*, fiche 16 l. 86) ni **fonction de transfert** (*transfer function*, fiche 17 l. 87, 17.Q2 c). BIBLE §5 : « Tout terme ajouté va dans `annexes/glossaire.md` ».
- **Correction** (section du ch. 16 et 17) :
  `| SwiGLU | SwiGLU (gated linear unit à porte SiLU) | bloc feed-forward à porte : $(\mathrm{SiLU}(\mathbf{x}\mathbf{W}) \odot \mathbf{x}\mathbf{V})\mathbf{W}_2$, largeur cachée réduite aux 2/3 de $4d$ (LLaMA) | 17 |`
  `| échantillonneur-bloqueur | sample-and-hold | une valeur arrivée sur une arête y reste jusqu'à ce qu'une autre la remplace | 16 |`
  `| fonction de transfert | transfer function | autre nom de la fonction d'activation dans le livre | 17 |`

### 10. SUGGESTION · Glossaire, « affine par morceaux »

- `annexes/glossaire.md:643` : « affine par morceaux | piecewise linear | fonction **continue** faite de morceaux de droite raccordés en des coudes ».
- Le livre et la fiche 17 (§17.3.2, §17.5) appellent aussi « linéaire par morceaux » l'escalier, discontinu (17.Q3, 17.Q4 b). **Correction** : « fonction faite de morceaux de droite (continue quand les morceaux se raccordent en des coudes, comme ReLU ; l'escalier ne l'est pas) ».

### 11. SUGGESTION · Fiche 17, seuil de softplus

- `01_fiche.md:156` : « au-dessus duquel, quand $\beta z$ le dépasse, la fonction rend $z$ tel quel : la différence est sous $10^{-8}$ ».
- L'écart vaut $\log(1 + e^{-\beta z})/\beta \le 2{,}06 \times 10^{-9}/\beta$ : sous $10^{-8}$ pour $\beta \ge 0{,}21$ seulement. **Correction** : « la différence est sous $2 \times 10^{-9}$ pour $\beta = 1$ ».

### 12. SUGGESTION · Liens de documentation PyTorch de versions différentes

- Fiche 16 l. 115 et 204 : `docs.pytorch.org/docs/2.14/…` (trois liens) ; fiche 17 l. 176 et 246 : `docs.pytorch.org/docs/2.11/…`. La version figée est 2.11 (BIBLE §21), et les deux chapitres affirment « vérifié sur PyTorch 2.11 ». **Correction** : remplacer `2.14` par `2.11` dans la fiche 16 (vérifié : la page « CUDA semantics » de la 2.11 existe, avec la même section « Asynchronous execution » ; `nn.Linear` et `nn.init` existent dans toutes les versions). Le mélange `stable` / `2.11` / `2.14` touche aussi d'autres chapitres (23 liens `stable`, 10 `2.11`, 6 `2.14`) : à harmoniser globalement (relecteur des liens).

### 13. SUGGESTION · Flashcard « Variance d'une somme pondérée »

- `ch16_feedforward/flashcards.csv`, carte 16 : « Chaque couche multiplie donc la variance par \(n_{in}\mathrm{Var}(w)\) : après L couches, par sa puissance L. »
- Vrai pour des entrées centrées et sans activation (sinon c'est $\mathbb{E}[x^2]$, et ReLU ajoute le facteur 1/2, ce que la carte suivante et la fiche disent). **Correction** : « Dans un réseau linéaire aux entrées centrées, chaque couche multiplie donc la variance par… ».

## Ce qui a été vérifié et trouvé juste (pour mémoire)

- **Fiche 16** : passe avant matricielle et mini-exemples (46 paramètres du 3-4-4-2 ; 4,2 M paramètres du 3072-1024-1024-10, dont 75 % dans la première couche ; 0,577 et 1,83 d'écart-type ; 0,058 / 0,1 / 0,082 pour la couche 300 × 300 ; $1{,}25^{15} = 28{,}4$ ; $2^{20} \approx 10^6$ ; pic $1/(\sigma\sqrt{2\pi})$ ; 0,27 % au-delà de 3σ) ; Kahn et l'ordre P, Q, R, S, T ; règle des horodatages et mini-exemple $t = 4, 6, 7, 8$ ; figure (b) des graphes (ordre i1, h1, i2, h2, h3, s, différent du rangement couche par couche) ; figure de la variance en profondeur.
- **Fiche 17** : toutes les formules et dérivées (SiLU′, GELU′, tanh = 2σ(2z) − 1, Jacobienne), limites de débordement (709,78 ; −745 ; 88,7 ; −104), $\sigma(40) = 1$ exactement, $\sigma'(10) = 4{,}5\times10^{-5}$, l'exemple $\sigma(\sigma(3))$ (0,20 contre 0,045), le creux de SiLU (−0,278 en −1,278) et de GELU (−0,170 en −0,752), les dérivées maximales 1,0998 et 1,1289 ($z = \sqrt2$), l'écart de l'approximation tanh de GELU (max 4,7e-4), le test « GELU avec erf(z/2) » qui vérifie encore $\mathrm{GELU}(z) - \mathrm{GELU}(-z) = z$.
- **Encadrés « Le livre, corrigé »** : densité/probabilité (fig. 16.5-16.6), influence de l'initialisation (citation fidèle de p. 668), « en essayer quelques-unes » (p. 670, fidèle), PReLU (le livre dit « lets us choose », p. 687), « la softmax vue plus haut » (aucune softmax avant p. 699 dans tout le volume 1 : vérifié), rapport C/B (livre p. 701 « a little less than 2 », scores 0,8 et 0,1 donnés par le livre : $e^{0{,}7} = 2{,}014$). Tous fidèles et justes envers le livre, sauf la nuance du constat 4. Renvois de pages tous exacts.
- **Cohérence entre chapitres** : la convention « dérivées en $z$ » et `ReLU'(0) = 0` est celle du ch. 18 (`mlp_backward`, `Value`) ; « un pas d'entraînement ≈ trois passes avant » (fiche 16, 16.8 e) est la conclusion du ch. 18 (« Le livre, corrigé — à peu près le temps d'une passe avant ») ; le défaut de `nn.Linear` (variance $1/(3 n_{in})$, six fois plus petite que He) est dit de la même façon au ch. 20 (fiche l. 125) et dans `cheatsheets/pytorch.md:47` ; $\sigma'(6) \approx 2{,}5\times10^{-3}$ (ch. 18 l. 256 et 17.Q8 b) ; le `sign_step` du ch. 10 (−1 en 0) est correctement cité par la fiche 17 (l. 116), qui distingue les deux conventions du livre (−1 en 0 au ch. 10, valeur de droite au ch. 17). Formulaire (ch. 16 et 17), erreurs fréquentes (l. 77-82, 101-102, 221-222), cheatsheets NumPy (l. 262-286) et PyTorch (l. 41-52) : justes, hors constats 7 et 8. Data card MNIST : usages du ch. 16 exacts, le ch. 17 n'utilise que des données synthétiques.

## Affirmations 🕰️ (et de même nature) revérifiées

| # | Affirmation (fichier) | Verdict | Source |
|---|---|---|---|
| 1 | Bloc FFN du Transformer : $\max(0, xW_1 + b_1)W_2 + b_2$, 512 → 2 048 → 512 ; $8d^2$ poids contre $4d^2$ (fiche 16 l. 68) | juste | [Vaswani et coll., arXiv:1706.03762, §3.3](https://arxiv.org/pdf/1706.03762) |
| 2 | Opérations CUDA asynchrones par défaut, ordonnées dans une file, chronométrer avec `synchronize()` ou `torch.cuda.Event(enable_timing=True)` (fiche 16 l. 115, 16.Q7 b) | juste | [PyTorch, « CUDA semantics »](https://docs.pytorch.org/docs/2.14/notes/cuda.html) |
| 3 | `nn.Linear` : `kaiming_uniform_(a=√5)` = $\mathcal{U}(\pm 1/\sqrt{n_{in}})$, biais dans le même intervalle (fiche 16 l. 204, flashcard, 16.E2) | juste | code source de `torch.nn.Linear.reset_parameters`, torch 2.11 (rejoué) ; [`torch.nn.Linear`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html) |
| 4 | Keras : `kernel_initializer="glorot_uniform"`, biais nuls ; `GlorotNormal`, `HeNormal`, `LecunNormal` tirent une normale tronquée réglée pour garder la variance (fiche 16 l. 204) | juste | [Keras, `Dense`](https://keras.io/api/layers/core_layers/dense/) ; [Keras, initializers](https://keras.io/api/layers/initializers/) (« stddev … after truncation ») |
| 5 | scikit-learn : Glorot uniforme biais compris, borne $\sqrt{2/(n_{in}+n_{out})}$ avec `logistic` (fiche 16 l. 204) | juste | `_init_coef`, scikit-learn 1.6.1 (code source rejoué) |
| 6 | GPT-2 : $\mathcal{N}(0, 0{,}02^2)$, projections résiduelles réduites de $1/\sqrt{2N}$ ; `gelu_new` (fiches 16 et 17) | juste | [Hugging Face, `GPT2Config`](https://huggingface.co/docs/transformers/model_doc/gpt2) ; `modeling_gpt2.py` de transformers 5.18 (`_init_weights`) |
| 7 | BERT : « a gelu activation rather than the standard relu, following OpenAI GPT », annexe A.2 (fiche 17 l. 190) | juste | [Devlin et coll., arXiv:1810.04805](https://arxiv.org/pdf/1810.04805) |
| 8 | SwiGLU $(\mathrm{Swish}_1(xW) \otimes xV)W_2$, largeur réduite, « divine benevolence » (fiche 17 l. 190) | juste | [Shazeer, arXiv:2002.05202](https://arxiv.org/pdf/2002.05202) |
| 9 | LLaMA : « a dimension of 2/3·4d instead of 4d as in PaLM » ; `hidden_act="silu"` (fiche 17 l. 190) | juste | [Touvron et coll., arXiv:2302.13971, §2.2](https://arxiv.org/pdf/2302.13971) ; `configuration_llama.py` (transformers 5.18) |
| 10 | `nn.CrossEntropyLoss` attend « the unnormalized logits for each class », équivaut à `LogSoftmax` + `NLLLoss` (fiche 17 l. 246) | juste | docstring de torch 2.11 ; [PyTorch, `CrossEntropyLoss`](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html) |
| 11 | `torch.nn.LSTM` : portes en σ, état candidat en tanh (fiche 17 l. 176) | juste | docstring de `torch.nn.LSTM`, torch 2.11 |
| 12 | Aux points anguleux, les logiciels rendent une dérivée d'un côté ; unités sigmoïdes cachées « now discouraged » (fiche 17 l. 128 et 176) | juste | [Goodfellow, Bengio, Courville, §6.3 et §6.3.2](https://www.deeplearningbook.org/contents/mlp.html) ; valeurs de PyTorch 2.11 rejouées |
| 13 | PReLU apprise, par canal ou partagée, initialisée à 0,25 ; He : une seule des deux conditions suffit (fiche 17 l. 136, 16.10 variante) | juste | [He et coll., arXiv:1502.01852, §2.1-2.2](https://arxiv.org/pdf/1502.01852) |
| 14 | GELU : approximations tanh et $x\sigma(1{,}702x)$, tâches, conseil du momentum, SiLU entre GELU et ReLU/ELU, swish publié « without citing » (17.10) | juste | [Hendrycks et Gimpel, arXiv:1606.08415](https://arxiv.org/pdf/1606.08415) |
| 15 | Glorot et Bengio : 1 à 5 couches de 1 000 neurones, Shapeset-3×2, MNIST, CIFAR-10, Small-ImageNet ; dernière couche sigmoïde poussée vers 0, explication par les biais de la softmax ; valeurs singulières 0,8 contre 0,5 ; saturation de tanh à partir de la couche 1 ; variance des gradients des poids à peu près constante (fiche 16 l. 176, 16.10) | juste | [Glorot et Bengio, AISTATS 2010 (PDF)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) |

Non revérifiable sur le texte du livre : la sortie « $47A + 6B + 61C$ » de la figure 17.5 (fiche 17 l. 71), qui n'est que dans l'image.

## Bilan

**Solide.** Les deux chapitres sont d'une exactitude remarquable : aucune réponse enregistrée fausse sur environ 230 réponses courtes recalculées, des preuves ∂ rigoureuses (hypothèses d'indépendance énoncées au bon niveau, cas des entrées corrélées traité), des conventions PyTorch toutes confirmées sur torch 2.11, des encadrés « Le livre, corrigé » fidèles au texte et des 🕰️ tous confirmés par leur source. Les chiffres de la fiche, des solutions, des sorties exécutées et des figures concordent. Les conventions reprises aux ch. 18 à 20 (dérivées en $z$, `ReLU'(0) = 0`, coût ≈ 3 passes avant, défaut de `nn.Linear`) ne se contredisent nulle part.

**Les 3 risques principaux.**
1. Le 🔮 16.19 a) (constat 1) : une réponse « exacte » qui dépend de l'implémentation de l'apprenant ; c'est le seul point où un apprenant au code juste peut voir le corrigé démenti à l'écran.
2. Les arguments de précision numérique « au bit près » (constats 1 et 2) : affirmés sans marge, ils sont vrais pour l'implémentation de référence (BLAS, `np.tanh`) mais pas en général ; à formuler avec une tolérance.
3. Des annexes qui dérivent légèrement des chapitres (constats 7, 8, 9) : symptôme de forme inexact, renvoi vers un détail absent, termes non reportés au glossaire.
