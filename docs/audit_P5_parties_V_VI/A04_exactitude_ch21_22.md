# Audit P5 parties V-VI — A4 : exactitude des ch. 21 (CNN) et 22 (RNN)

**Périmètre** : `chapitres/ch21_cnn` et `chapitres/ch22_rnn` (fiches, exercices, indices, solutions md et notebooks exécutés, flashcards, figures et `tools/chapters/figures_ch21.py`, `figures_ch22.py`, historiques `history_ch21.json` et `history_ch22.json`) ; leurs lignes dans `annexes/formulaire.md`, `glossaire.md`, `erreurs_frequentes.md`, `cheatsheets/pytorch.md` ; data cards `sunspots.md`, `holmes.md`, `verne.md`, `cifar10.md` ; renvois des ch. 20, 23 à 28 et CP5 vers 21 et 22. Commit `b961aa3`, torch 2.11.0+cpu, torchvision 0.26.0. **Scripts de vérification** (dans `A04_exactitude_ch21_22/scripts/`) : `bgrep.py` (recherche paginée dans le texte du livre), `nbcells.py` (cellules d'un notebook), `dump_records.py` (exécute la partie 0 d'un notebook de solutions avec un `wb.record` qui affiche les valeurs enregistrées ; sorties `rec21_part0.txt`, `rec22_part0.txt`), `indep21.py` et `indep22.py` (re-résolution indépendante des ✏️, ∂ et 🧮 avec NumPy et PyTorch), `md2mod.py` (reconstruit `conv.py`, `rnn.py` et `sequences.py` à partir du code de `05_solutions.md`, puis les 449 tests des ch. 21 et 22 lancés sur ce code par `--learner-dir` : **449 passés** ; un mutant de contrôle fait échouer 118 tests). Autres contrôles : `pytest --impl=ref` des ch. 21-22 (449 passés), `build_answers.py --check` (à jour), `export_flashcards.py --check`, API vérifiée dans le code source de torch 2.11 et torchvision 0.26, 11 affirmations 🕰️ revérifiées sur le web.

**Ce qui a été recalculé.** Ch. 21 : les 133 valeurs de la partie 0 (Q1 à Q12, R1 à R3, ✏️ 21.1 à 21.6, 🧮 21.8), la démarche et les variantes de chaque ✏️/∂/🧮 de `05_solutions.md` (dont ∂ 21.7 question 4 et sa différence finie), les 21.23 depuis `history_ch21.json`, les comptes de paramètres et de MAC des solutions du notebook (21.15, 21.22, 21.24, 21.26, 21.28, 21.32) et les nombres de la fiche (exemples de tailles, champ récepteur, He, kaiming_uniform, têtes de VGG16 et ResNet-18). Ch. 22 : les 119 valeurs de la partie 0, chaque démarche et variante de ✏️ 22.1 à 22.5, ∂ 22.6, ∂ 22.7, 🧮 22.8, 22.E3, les nombres de 22.12 à 22.29 contre les sorties exécutées (dont 22.23 b, persistance 41,10 recalculée sur SILSO, et 22.26 depuis `history_ch22.json`), et les nombres de la fiche. **Aucune réponse enregistrée fausse, aucune valeur de solution en désaccord avec le calcul ou avec les sorties exécutées.**

## Constats

### 1. MINEUR · `chapitres/ch21_cnn/01_fiche.md:15` et `:370` (temps de lecture non reportés)
**Extrait :** « Compte environ sept heures de lecture, livre et fiche. » (l. 15) ; « Le **parcours rapide** lit toute la fiche et les sections ⏩ du livre, soit environ 4,2 h » (l. 370).
**Problème :** l'en-tête de la même fiche (l. 8) et `docs/SYLLABUS.md` (l. 3507 et 3526) donnent « lecture 8,0 h » et « Lecture du parcours rapide … ≈ 5,3 h ». La fiche a grandi après le calcul, et CLAUDE.md demande de reporter ces deux temps après `syllabus.py build`. Les ch. 20, 22, 23 et 24 sont cohérents (ch. 22 : 5,8 h / « six heures » / 4,6 h).
**Correction :** l. 15 « Compte environ huit heures de lecture, livre et fiche. » ; l. 370 « soit environ 5,3 h ».

### 2. MINEUR · `chapitres/ch22_rnn/02_exercices.md:142` et `chapitres/ch22_rnn/04_indices.md:235` (renvoi faux vers le §21.3.4)
**Extrait :** « ce que change une **dilatation** qui double à chaque couche (ch. 21, §21.3.4) » ; indice 1 de 22.R1 : « Relis la convolution 1D (§21.5) et le champ récepteur (§21.3.4) de la fiche du ch. 21. »
**Problème :** le §21.3.4 de la fiche du ch. 21 est le padding. Le champ récepteur (et le rôle de la dilatation dans sa formule) est au §21.3.3 ; la dilatation est définie dans l'encadré « Le livre, corrigé — la convolution dilatée » du §21.8.
**Correction :** 02, l. 142 : « (ch. 21, §21.3.3 et encadré du §21.8) » ; indices, l. 235 : « … et le champ récepteur (§21.3.3) de la fiche du ch. 21, avec l'encadré du §21.8 sur la dilatation. »

### 3. MINEUR · `chapitres/ch24_pytorch_2/02_exercices.md:154` (renvoi vers le ch. 21)
**Extrait :** « ### 24.R2 — Ch. 21 : Taille de sortie d'une convolution à padding 'same' et stride 2 … *Ch. 21 (§21.5) · fiche §24.4.3* »
**Problème :** le §21.5 de la fiche du ch. 21 est la convolution 1D ; la taille de sortie, le padding et le stride sont aux §21.3.4 et §21.3.5. (Le même renvoi « §21.5 » est juste en 22.R1, qui porte sur la convolution 1D.) Le défaut se corrige dans `tools/chapters/build_ch24.py` ou dans la source du `02` selon la chaîne de génération du ch. 24.
**Correction :** « *Ch. 21 (§21.3.4, §21.3.5) · fiche §24.4.3 · parcours R, M* ».

### 4. MINEUR · `annexes/formulaire.md:601` et `chapitres/ch22_rnn/flashcards.csv:7` (le « biais unique » de Keras étendu au GRU)
**Extrait :** formulaire : « $k\,(DH + H^2 + 2H)$, $k = 1$ (RNN), 3 (GRU), 4 (LSTM) ; Keras : un seul biais, $k\,(DH + H^2 + H)$ » ; flashcard : « k (DH + H² + 2H) avec k = 1 (RNN), 3 (GRU), 4 (LSTM). Les deux vecteurs de biais viennent de cuDNN. Keras n'en compte qu'un. »
**Problème :** vrai pour le Keras 2.0 du livre et pour `SimpleRNN` et `LSTM`, faux pour le GRU de Keras depuis TensorFlow 2 et dans Keras 3 : `reset_after=True` par défaut, deux vecteurs de biais, donc le même compte que PyTorch (source : réponse « calculating the number of parameters of a GRU layer », [kiwix StackOverflow 57318930](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/57318930/calculating-the-number-of-parameters-of-a-gru-layer-keras) : 9 600 paramètres avec `reset_after=True` contre 9 408 avec `False`). Un apprenant qui compare un `model.summary()` Keras 3 à la formule se trompera pour le GRU.
**Correction :** formulaire : « … ; Keras : un seul biais pour `SimpleRNN` et `LSTM`, $k\,(DH + H^2 + H)$ (le GRU de Keras 3, `reset_after=True` par défaut, en a deux, comme PyTorch) ; bidirectionnelle : $\times 2$ ». Flashcard : « … Le Keras du livre n'en compte qu'un (Keras 3 en garde deux pour le GRU). »

### 5. MINEUR · `annexes/cheatsheets/pytorch.md:92` (biais d'oubli à 1 : une seule tranche)
**Extrait :** « `lstm.bias_ih_l0.data[H:2 * H].fill_(1.0)` | biais de la porte d'oubli à 1 (PyTorch ne le fait pas) »
**Problème :** la porte voit $b_{ih} + b_{hh}$ ; la fiche du ch. 22 (l. 181 : « on met à 1 la tranche `[H:2H]` de `bias_ih_l0` et à 0 celle de `bias_hh_l0` ») et le notebook (`build_ch22.py:849-850`) remettent aussi la tranche de `bias_hh_l0` à 0. La ligne de l'aide-mémoire laisse cette tranche tirée dans $\pm 1/\sqrt{H}$ (code de `RNNBase.reset_parameters`, torch 2.11) : le biais total n'est pas 1. Elle passe aussi par `.data`, que les ch. 20 à 24 remplacent par `torch.no_grad()`.
**Correction :** « `with torch.no_grad(): lstm.bias_ih_l0[H:2 * H].fill_(1.0); lstm.bias_hh_l0[H:2 * H].zero_()` | biais total de la porte d'oubli à 1 (la porte voit `bias_ih + bias_hh` ; PyTorch ne le fait pas ; à répéter pour chaque couche `_l1`, `_l2`…) ».

### 6. MINEUR · `annexes/cheatsheets/pytorch.md:88-89` (la lettre D pour deux choses dans le même tableau)
**Extrait :** l. 88 « entrée `(T, N, D)`, ou `(N, T, D)` avec `batch_first=True` » ; l. 89 « `output` : … `(N, T, D·H)` avec `D = 2` si bidirectionnel ; `h_n` : … `(couches·D, N, H)`, indice `couche·D + sens` ».
**Problème :** deux lignes consécutives donnent à $D$ deux sens : le nombre de features (convention de la fiche du ch. 22, encadré « les conventions de ce chapitre ») et le nombre de sens (notation de la documentation de PyTorch). Contradictoire avec la fiche, et piégeant pour l'apprenant qui lit `(N, T, D·H)`.
**Correction :** l. 89 : « `output` : … `(N, T, n_dir·H)` avec `n_dir = 2` si bidirectionnel ; `h_n` : … `(couches·n_dir, N, H)`, indice `couche·n_dir + sens` ».

### 7. MINEUR · `chapitres/ch22_rnn/flashcards.csv` (maths sans `\( … \)`, accolades affichées telles quelles)
**Extrait :** l. 4 « <b>h_t = tanh(x_t W_ih + b_ih + h_{t−1} W_hh + b_hh)</b> » ; l. 15 « <b>c_t = f ⊙ c_{t−1} + i ⊙ g</b> » ; l. 21 « h_t = (1 − z) ⊙ n + z ⊙ h_{t−1} ».
**Problème :** BIBLE §12 : « maths en `\( … \)` ». Le ch. 22 est le seul des ch. 16 à 22 dont les flashcards n'ont aucune formule MathJax (0 contre 6 à 19 ailleurs ; ch. 21 : 13) : dans Anki, `h_{t−1}` s'affiche avec ses accolades et ses soulignés. Même dérive au ch. 23 et au ch. 29 (0 formule), hors de mon périmètre. Une coquille aussi, l. 10 : « …au réseau déroulé. les gradients d'un même poids… » (minuscule après un point).
**Correction :** passer les formules des cartes 3, 4, 7, 11, 15, 19 et 21 en `\( … \)` (par exemple l. 15 : « \( c_t = f \odot c_{t-1} + i \odot g \) » et « \( h_t = o \odot \tanh(c_t) \) ») ; l. 10 : « … au réseau déroulé : les gradients d'un même poids s'additionnent sur tous les pas. » Ajouter à `export_flashcards.py --check` un avertissement pour un `_{` hors de `\( … \)`.

### 8. MINEUR · `chapitres/ch21_cnn/01_fiche.md:129` (le score maximal du filtre de trait horizontal)
**Extrait :** « le score maximal, 3, n'est atteint que pour un trait de trois pixels exactement entouré de noir. »
**Problème :** le filtre ne contraint que la fenêtre 3 × 3 : un trait horizontal plus long, d'un pixel d'épaisseur, donne 3 à chaque position le long du trait. Le notebook le montre : la solution de 🎨 21.17 (`05_solutions.md:386`) explique que « les deux correspondances voisines viennent d'une rayure de **quatre** pixels ». La phrase de la fiche contredit donc ce que l'apprenant mesurera.
**Correction :** « le score maximal, 3, n'est atteint que lorsque la rangée du milieu de la fenêtre est entièrement blanche et les deux autres entièrement noires : sur un trait horizontal d'un pixel d'épaisseur (un trait plus long que trois pixels répond à plusieurs positions voisines). »

### 9. MINEUR · `chapitres/ch21_cnn/01_fiche.md:139` (im2col présenté comme LA méthode des bibliothèques sur CPU)
**Extrait :** « C'est plus qu'une image : c'est la façon dont les bibliothèques calculent une convolution sur CPU. »
**Problème :** affirmation trop forte, et en tension avec la variante de 21.14 (`05_solutions.md:295` : « la raison pour laquelle les bibliothèques ne la construisent pas toujours »). Sur CPU, PyTorch passe en général par oneDNN (convolutions directes ou Winograd), et ne prend la voie im2col + GEMM (`slow_conv2d`) qu'en repli ; Caffe, en revanche, l'utilisait partout.
**Correction :** « C'est plus qu'une image : c'est une façon classique de calculer une convolution (Caffe, et la voie de repli de PyTorch sur CPU) ; les bibliothèques récentes lui préfèrent souvent des algorithmes qui évitent de recopier l'image. »

### 10. MINEUR · `chapitres/ch21_cnn/01_fiche.md:318` (BIBLE §3 : paraphrase proche du livre)
**Extrait :** « Sur une photo de canard, un filtre du premier bloc de VGG16 répond aux bords verticaux, d'autres à l'orange des pattes, au bleu de l'eau ou à la neige ; plus loin dans le réseau, les cartes sont plus petites (chaque pooling divise leur côté par 2) et de moins en moins liées à la forme du canard ; d'autres cartes restent presque vides : le motif de leur filtre est absent de cette photo. »
**Problème :** la phrase suit pas à pas, dans le même ordre, quatre phrases consécutives du livre (p. 1009-1012) : « … » (citation du livre retirée de la version versionnée, §3) ; « … » (citation du livre retirée de la version versionnée, §3) ; « … » (citation du livre retirée de la version versionnée, §3) ; « … » (citation du livre retirée de la version versionnée, §3). C'est une paraphrase rapprochée au sens du §3.
**Correction :** résumer sans reprendre les observations une à une : « Le livre fait passer une photo de canard dans VGG16 et affiche les cartes de plusieurs couches (p. 1008-1012) : dans le premier bloc, chaque carte ressemble encore à la photo, filtrée selon un bord ou une couleur ; plus loin, les cartes rapetissent à chaque pooling et deviennent des taches abstraites, et certaines restent vides parce que le motif de leur filtre manque dans l'image. »

### 11. MINEUR · `annexes/glossaire.md:27` (entrée « kernel » contraire à la dérogation du §22)
**Extrait :** « | kernel | noyau, filtre | petite grille de poids que l'on fait glisser sur une image (CNN) | | »
**Problème :** la colonne « Retenu » dit *kernel*, alors que la DÉROGATION du §22 (2026-10-18) retient « noyau » dans la prose des CNN (le code garde `kernel_size`) ; l'entrée l. 722 « noyau (d'une convolution) » le dit bien. Les deux lignes se contredisent sur le terme retenu, et la colonne « Ch. » est vide.
**Correction :** l. 27 : « | kernel (dans le code : `kernel_size`) | noyau, filtre | dans la prose des CNN, on dit « noyau » ou « filtre » (BIBLE §22, voir « noyau (d'une convolution) ») | 21 | ».

### 12. SUGGESTION · `annexes/glossaire.md:669` et `:753` (même terme défini deux fois)
**Extrait :** l. 669 « | gradient évanescent | vanishing gradient | … | 0B | » ; l. 753 « | gradient évanescent, explosif | vanishing, exploding gradient | … | 22 | ».
**Problème :** deux entrées pour la même notion, avec des chapitres d'introduction différents ; la première parle de couches, la seconde de pas de temps.
**Correction :** garder une seule entrée « gradient évanescent, explosif » (Ch. « 0B, 22 »), dont la définition couvre les couches (0B, ch. 17-18) et les pas de temps (ch. 22, jacobiennes, $\sigma_{\max}(W_{hh})$).

### 13. SUGGESTION · `chapitres/ch21_cnn/02_exercices.md:91` (21.Q8 a, cas du dropout)
**Extrait :** « Vrai ou faux : si **tous** les poids et les biais d'une couche de convolution et des couches suivantes partent d'une même valeur (une pour les poids, une pour les biais), les filtres de cette couche restent identiques pendant tout l'entraînement. »
**Problème :** la réponse « Vrai » suppose un réseau sans dropout : un dropout après la convolution (le convnet du livre en a un, §21.9) tire un masque différent pour chaque canal et sépare les filtres dès le premier pas. Le §22 a déjà précisé cet énoncé pour les biais ; la réponse ne change pas.
**Correction :** ajouter « (sans dropout) » après « des couches suivantes », et dans la solution (`05_solutions.md:40`) : « (un dropout, dont le masque diffère d'un canal à l'autre, suffirait à briser la symétrie) ».

### 14. SUGGESTION · `chapitres/ch22_rnn/05_solutions.md:153` (domaine public de Verne)
**Extrait :** « les droits patrimoniaux s'éteignent 70 ans après la mort de l'auteur, et leurs œuvres sont dans le domaine public depuis 2001 et 1976. »
**Problème :** juste pour Conan Doyle (protégé en Allemagne au 1er juillet 1995, donc couvert par l'harmonisation de la directive 93/98 jusqu'à fin 2000). Pour Verne, mort en 1905, les œuvres étaient déjà libres avant le passage à 70 ans (en France, 50 ans plus les prorogations de guerre, vers 1970), et la règle des 70 ans ne les a pas fait renaître : « depuis 1976 » est une date calculée, pas la date réelle.
**Correction :** « … leurs œuvres sont dans le domaine public dans toute l'Union européenne, celles de Conan Doyle depuis 2001, celles de Verne depuis les années 1970 au plus tard (la date exacte dépend des règles de chaque pays avant l'harmonisation). »

## Vérifications sans constat (pour mémoire)

- **Fiches** : toutes les formules contrôlées sont justes (taille de sortie avec $p$, $s$, $d$ ; champ récepteur ; im2col ; convolution transposée et recette $k - 1 - p$ ; vue « tampon » ; passe arrière $\partial L/\partial x_j = \sum_i g_i w_{j-i}$ ; Glorot, He, `kaiming_uniform_(a=√5)` = variance $1/(3\,\text{fan-in})$ ; LSTM et GRU dans l'ordre de PyTorch ; jacobienne $\operatorname{diag}(1 - h_k^2)W_{hh}^\top$ et son ordre de produit ; chemin additif ; température ; perplexité). Tous les exemples chiffrés recalculés (32 → 32 → 16, $r_3 = 15$, $3\,216$ paramètres, $\sqrt{2/288} = 0{,}083$, $0{,}604$ et $0{,}707$, $0{,}7^{20}$, $1{,}2^{20}$, $0{,}95^{100}$ et $1{,}05^{100}$, $c_t = 2{,}9$ et $h_t = 0{,}497$, $0{,}95^{50}$, $0{,}665$ et $0{,}867$, $N = 6$ fenêtres, têtes de VGG16 (123,6 M) et de ResNet-18 (513 000), CNN-LSTM du livre 16,8 M).
- **Encadrés « Le livre, corrigé »** : les 12 du ch. 21 et les 7 du ch. 22 sont justes et leurs pages exactes (p. 931, 963, 976-977, 978, 985, 988, 993-994, 996, 1002, 1012-1013, 1020 ; p. 1043, 1052-1053, 1054, 1065, 1072, 1078, 1079), y compris la numérotation « 24.4.2 » à « 24.9.1 » du livre. Les citations du livre vérifiées : « … » (citation du livre retirée de la version versionnée, §3) (p. 931), « … » (citation du livre retirée de la version versionnée, §3) (p. 963), « … » (citation du livre retirée de la version versionnée, §3) (p. 988), « … » (citation du livre retirée de la version versionnée, §3) (p. 996), « … » (citation du livre retirée de la version versionnée, §3) (p. 1012), « … » (citation du livre retirée de la version versionnée, §3) (p. 1043), « … » (citation du livre retirée de la version versionnée, §3) (p. 1065), « … » (citation du livre retirée de la version versionnée, §3)] (p. 1065).
- **Solutions** : démarches, erreurs fréquentes et variantes recalculées une à une ; les réponses 💼 et 🗣️ sont techniquement justes (E2 : 112, 1 792 et 1 728 paramètres, 21,7 M MAC, 113 pour 225 ; E3 du ch. 22 : 42 496 et 31 872 ; refus de `padding="same"` avec stride vérifié dans torch 2.11). Les fonctions de `05_solutions.md` (ch. 21 et 22), reconstituées en modules, passent les 449 tests.
- **Notebooks** : chaque nombre cité dans `05_solutions.md` pour 21.12 à 21.32 et 22.12 à 22.29 se retrouve dans les sorties exécutées ou se recalcule (21.23 depuis `history_ch21.json` : epoch 7, 0,075, Shirt, 100/212, 0,528, precision 0,830 ; 22.26 : epoch 9, 4,71, 2,235, 22,41). Seules les durées diffèrent (21.26 « 4 à 5 s » contre 5,2 s exécutées ; 22.18 « 20 ms et 7 ms » contre 25 et 9,5 ms ; 22.25 « 75 s » contre 84 s), ce qui reste dans la variation annoncée (consigne : ne pas juger les durées).
- **Flashcards** : les 25 du ch. 21 et les 25 du ch. 22 relues : contenu juste (hors constats 4 et 7).
- **Figures** : les 12 figures relues contre leurs légendes et leurs scripts : tailles (62, 60 ; 5 × 5, 7 × 7, 4 × 4 ; 11 × 11 et 12 × 12 ; 10 cases de champ récepteur), signes des bords (rouge positif), comptes de chevauchement (1, 2, 4), $0{,}7^{30}$ et $1{,}2^{30}$, fenêtres $T = 10$, $w = 3$, $h = 2$ : justes.
- **Data cards** : `sunspots.md` (3 332 mois, persistance 41,1, test caché jusqu'à mars 2026 avec `definitive_only=True`), `holmes.md` (88 caractères, perplexités 4,7 et 22,4), `verne.md` (13,5 % de `<unk>`, 2,85 et 3,07 bits), `cifar10.md` (21.13, 21.19, 21.26, 21.27) concordent avec les notebooks.
- **Contradictions** : aucune entre les ch. 21-22 et les ch. 20, 23, 24, 25, 27, 28, le CP5 (CP5.2, CP5.3, CP5.12 reprennent les conventions et les valeurs des ch. 21-22) et les annexes, hors constats 4 à 6 et 11.
- **🕰️ revérifiés sur le web ou dans le code source (11)** :
  1. `nn.Conv2d` calcule une corrélation croisée (docstring « the valid 2D cross-correlation operator », code source de torch 2.11.0).
  2. Initialisation par défaut `init.kaiming_uniform_(self.weight, a=math.sqrt(5))`, biais dans $\pm 1/\sqrt{\text{fan-in}}$ (`_ConvNd.reset_parameters`, torch 2.11.0).
  3. ResNet de torchvision : `nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")` (`ResNet.__init__`, torchvision 0.26.0).
  4. `padding="same"` refusé avec un stride : `ValueError: padding='same' is not supported for strided convolutions` (torch 2.11.0).
  5. `h_n` reste `(couches × sens, N, H)` avec `batch_first=True` ; `clip_grad_norm_` rend la norme totale (torch 2.11.0, exécuté).
  6. RNN de PyTorch initialisés uniformément dans $\pm 1/\sqrt{H}$, sans traitement du biais d'oubli (`RNNBase.reset_parameters`, torch 2.11.0).
  7. M5 : « 1,972 (35.8%) outperformed the sNaive benchmark » (Makridakis et coll., *M5 accuracy competition*, [PDF](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf)) : la fiche (« 36 % ») et 22.E4 (« une équipe sur trois ») sont justes.
  8. AI Act, omnibus : règlement (UE) 2026/1744, en vigueur le 27 juillet 2026 ; annexe III reportée au 2 décembre 2027 ; marquage de l'art. 50, § 2 pour les systèmes déjà sur le marché au 2 décembre 2026 (art. 111, § 4) ([euaiact.com](https://www.euaiact.com/digital-omnibus-ai)) : 21.10 et 22.10 sont justes.
  9. TiRex, modèle de fondation pour séries fondé sur xLSTM, 2025 ([arXiv 2505.23719](https://arxiv.org/abs/2505.23719v1)).
  10. Microsoft retire l'inférence de l'âge, du genre et des émotions d'Azure Face en juin 2022 ([blog Azure, « Responsible AI investments and safeguards for facial recognition »](https://azure.microsoft.com/en-us/blog/responsible-ai-investments-and-safeguards-for-facial-recognition/)).
  11. GRU de Keras : `reset_after=True` par défaut depuis TensorFlow 2, deux vecteurs de biais (constat 4).

## Bilan

**Solide.** Aucune réponse enregistrée fausse sur les 252 valeurs de la partie 0 des deux chapitres, recalculées indépendamment ; toutes les démarches, variantes et nombres des solutions se retrouvent, dans les calculs comme dans les sorties exécutées et les historiques précalculés. Le code des solutions passe les 449 tests. Les encadrés « Le livre, corrigé » sont exacts et leurs pages justes ; les 🕰️ revérifiés tiennent, y compris les dates de l'omnibus de l'AI Act et la statistique du M5.

**Trois risques principaux.**
1. Les **annexes** dérivent des fiches : le biais d'oubli à 1 sur une seule tranche (cheatsheet), le « biais unique » de Keras étendu au GRU (formulaire, flashcard), la lettre $D$ prise dans deux sens : l'apprenant qui révise sur les annexes retient une recette ou un compte faux.
2. Les **métadonnées non reportées** après une réécriture (temps de lecture du ch. 21) et les **renvois de sections** faux d'un chapitre à l'autre (22.R1, 24.R2) : rien ne les vérifie automatiquement.
3. Le **format des flashcards** n'est plus tenu à partir du ch. 22 (maths sans MathJax), une dérive de convention qui touche aussi les ch. 23 et 29.

| Sévérité | Nombre |
|---|---|
| MAJEUR | 0 |
| MINEUR | 11 |
| SUGGESTION | 3 |
