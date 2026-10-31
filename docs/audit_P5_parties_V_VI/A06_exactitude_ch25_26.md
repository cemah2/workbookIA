# Rapport A6 — exactitude des ch. 25 et 26 (audit P5, commit b961aa3)

**Périmètre :** `chapitres/ch25_autoencodeurs`, `chapitres/ch26_rl` (fiches, exercices, indices, solutions, notebooks, flashcards, figures), lignes ch. 25-26 de `annexes/formulaire.md`, `glossaire.md`, `erreurs_frequentes.md`, data cards `rl_envs.md`, `mnist.md`, `fashion_mnist.md`, `src/wb/envs.py` (ce que les fiches en disent), renvois des autres chapitres. Scripts de vérification : `scripts/` de ce dossier (liste en fin de rapport).

Scripts : `scripts/recalc25.py` (≈ 75 réponses ✏️ ∂ 🧮 du ch. 25 recalculées avec NumPy, PyTorch 2.11 et `torch.distributions`), `scripts/recalc26.py` (≈ 80 réponses et variantes du ch. 26 : retours, Q-learning, SARSA, ε-greedy, softmax, chaîne, Bellman, itération sur les valeurs de FrozenLake avec gymnasium 1.4.0, Flippers par recherche exhaustive), `scripts/nbdump.py` (sorties des deux `05_solutions.ipynb`, confrontées à `05_solutions.md`), `scripts/pg.sh` (pages du livre). Les extraits de code des fiches ont été rejoués dans `/root/venv313`.

## Constats

### 1. MINEUR · `annexes/erreurs_frequentes.md:278` (ligne « un autoencodeur qui ressort la même image »)
**Extrait :** « entraîné sur une seule image (ou trop peu), il l'a rangée dans ses biais »
**Problème :** l'annexe affirme comme un fait l'explication du livre que le chapitre réfute. Le 🔮 25.14 (`05_solutions.md`, Ex 25.14) mesure que la botte est rangée **dans les poids du décodeur** : un réseau sans aucun biais la reconstruit aussi (MSE 0,00000, ligne « sans biais » de la sortie de la cellule 53 de `05_solutions.ipynb`). La solution classe même « croire que "mémorisé" veut dire "dans les biais" » parmi les **erreurs fréquentes**. La fiche (§25.3) et la flashcard 7 disent bien « selon le livre ». L'annexe enseigne donc l'erreur que 25.14 corrige.
**Correction :** « entraîné sur une seule image (ou trop peu), il l'a mémorisée (dans ses biais selon le livre ; dans les poids du décodeur au 🔮 25.14) ».

### 2. MINEUR · `annexes/erreurs_frequentes.md` (ligne « `TypeError` ou état différent à chaque `reset` de Gymnasium »)
**Extrait :** « `TypeError` ou état différent à chaque `reset` de Gymnasium | graine passée en `np.int64` »
**Problème :** avec gymnasium 1.4.0 (version figée), `env.reset(seed=np.int64(3))` lève `gymnasium.error.Error: Seed must be a python integer, actual type: <class 'numpy.int64'>`, pas un `TypeError` (vérifié dans `/root/venv313`). L'apprenant qui cherche « TypeError » dans l'annexe ne trouvera pas sa ligne.
**Correction :** « `gymnasium.error.Error: Seed must be a python integer` (ou un état différent à chaque `reset`) ».

### 3. MINEUR · `annexes/formulaire.md:655` et `:658` (ch. 25, lignes « cross-entropy binaire » et « PSNR »)
**Extrait :** « $-\sum_j [x_j \ln \hat{x}_j + \dots]$ | `F.binary_cross_entropy_with_logits(logits, x)` » ; « PSNR | $10\log_{10}(1/\text{MSE})$ | `10 * torch.log10(1 / F.mse_loss(out, clean))` »
**Problème :** (a) la formule est une **somme** sur les pixels, le code donné calcule une **moyenne** (`reduction="mean"` par défaut) : exactement le « β caché de 784 » que la fiche (§25.9.2, pièges) et CP6.8 interdisent pour la loss du VAE. (b) Le code du PSNR prend la MSE de tout le batch ; la solution de 25.26 range justement « calculer la MSE sur tout le tableau au lieu d'image par image » parmi les erreurs fréquentes.
**Correction :** (a) `F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)` (par image) ; (b) « pour une image ; pour un batch de lignes : `(10 * torch.log10(1 / ((out - clean) ** 2).mean(dim=1))).mean()` ».

### 4. MINEUR · `chapitres/ch25_autoencodeurs/05_solutions.ipynb`, cellule 143 (vérification de 25.38), produite par `tools/chapters/build_ch25.py:2892-2893` ; même cellule dans `03_notebook.ipynb` (cellule 162)
**Extrait :** « `[float(rec_t38 + kl_t38), float(rec_t38), float(kl_t38)], rtol=1e-8)` » ; sortie : « UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior. Consider using tensor.detach() first. »
**Problème :** la vérification de l'apprenant affiche un avertissement PyTorch dès que 25.38 est fait (le §22, session 54, en avait déjà retiré un autre). L'apprenant croira que son code est en cause.
**Correction :** `[(rec_t38 + kl_t38).item(), rec_t38.item(), kl_t38.item()]` (ou `.detach()`), puis reconstruire et réexécuter.

### 5. MINEUR · `chapitres/ch25_autoencodeurs/05_solutions.md`, Ex 25.20, Démarche
**Extrait :** « celui d'un VAE ou d'un GAN, où le milieu d'une ligne droite tombe là où il n'y a presque jamais de code (25.Q9). »
**Problème :** 25.Q9 porte sur l'autoencodeur linéaire et l'ACP. Le renvoi voulu est ✏️ 25.6 (f, g : norme du milieu $\sqrt{10}$ contre $\sqrt{20}$), déjà cité dans la phrase précédente.
**Correction :** remplacer « (25.Q9) » par « (✏️ 25.6 f et g) ».

### 6. MINEUR · `chapitres/ch25_autoencodeurs/05_solutions.md`, Ex 25.21, Démarche
**Extrait :** « entre les deux 5, le fondu (0,017 à 0,022) est même plus « proche » que les deux 5 réels (0,020 et 0,036). »
**Problème :** la sortie exécutée (cellule 80) donne pour le fondu des deux 5 : « 0.020 0.017 0.017 0.019 0.022 0.027 0.036 ». Les étapes intermédiaires vont de 0,017 à **0,027**, et l'étape à 0,027 n'est pas plus proche que le premier 5 (0,020).
**Correction :** « entre les deux 5, le fondu (0,017 à 0,027) est presque partout plus « proche » que le second 5 réel (0,036), et souvent que le premier (0,020). »

### 7. MINEUR · `chapitres/ch26_rl/01_fiche.md:166` (§26.5) et `chapitres/ch26_rl/flashcards.csv:14` (carte « L-learning »)
**Extrait :** fiche : « Dans un monde parfaitement prévisible, ça marche : après 3 000 parties, l'agent joue encore en six coups des plateaux qui en demandent deux ; après 6 000, il les joue en deux (fig. 26.23 à 26.26 du livre, sur deux plateaux ; 🔬 26.18 vérifie sur tous). » ; flashcard : « Il marche dans un monde prévisible, avec du hasard (le camion de Flippers), une seule partie malchanceuse efface la bonne valeur d'un coup. »
**Problème :** le chapitre réfute lui-même ce « ça marche ». Sortie de 26.18 (cellule 69 de `05_solutions.ipynb`) : sans camion, 7,7 % des plateaux joués au minimum, **autant après 3 000 qu'après 6 000 parties**, longueur moyenne 15,6 contre un optimum de 2,70 ; 🔮 26.16 : 2 % à 14 % sur cinq graines ; §22 (session 56) : « le L-learning sans exploration ne progresse plus entre 3 000 et 6 000 parties ». La fiche présente comme un fait ce qui n'est vrai que de deux plateaux choisis par le livre, et la flashcard le fait mémoriser sans réserve. La flashcard a en plus une ponctuation qui fausse le sens (« Il marche dans un monde prévisible, avec du hasard (…), une seule partie… » se lit « il marche avec du hasard »).
**Correction :** fiche : « Sans hasard, le livre montre que ça semble marcher, sur deux plateaux qu'il a choisis : … (fig. 26.23 à 26.26). Le 🔮 26.16 et le 🔬 26.18 te font vérifier si c'est vrai des 506 plateaux. » (ne donne pas l'issue du 🔮). Flashcard : « Un algorithme volontairement naïf : à la fin de chaque partie, le retour total de chaque coup <b>remplace</b> la valeur de sa case. Sans exploration, il garde ses premiers chemins, souvent longs ; avec du hasard (le camion de Flippers), une seule partie malchanceuse efface la bonne valeur d'un coup. »

### 8. MINEUR · `chapitres/ch26_rl/01_fiche.md:247` (⚠️ « Le livre, corrigé — Flippers en chiffres »)
**Extrait :** « en comptant les 512 plateaux (🔬 26.17), 3 coups est le cas le plus fréquent, mais 2 coups l'est plus que 4, et une partie optimale est en moyenne plus courte que 3 coups. »
**Problème :** la fiche donne d'avance la réponse vérifiée `most_common_17` de 🔬 26.17 (3, clé `26.17a`) et une borne de `mean_17` (2,70, clé `26.17b`). Le guide de lecture fait lire ce passage (étape 4) avant le notebook (étape 7). C'est la règle notée en session 55 pour 26.2 f (« une correction du livre dans la fiche ne doit pas donner la réponse ») ; ici, c'est une réponse de 🔬.
**Correction :** « C'est vrai de justesse, et la répartition n'est pas celle qu'on imagine : le 🔬 26.17 te la fait calculer sur les 512 plateaux. »

### 9. MINEUR · `chapitres/ch26_rl/01_fiche.md:297` et `:335` (renvois au bonus du RLHF)
**Extrait :** « l'usage le plus visible du RL aujourd'hui : l'entraînement des grands modèles de langage (ch. B4). » ; « **B4 et B8** : le RLHF des modèles de langage ; DQN, gradient de politique et PPO ».
**Problème :** d'après le contrat, le RLHF, DPO et GRPO sont au bonus **B8** (`docs/syllabus/data/chB8.json` : « préférences : losses et RLHF miniature », « RLHF (modèle de récompense + PPO avec pénalité KL), DPO sans RL explicite, GRPO »). `chB4.json` ne mentionne ni RLHF ni DPO (0 occurrence) : B4 couvre Hugging Face, le prompting, le RAG et LoRA.
**Correction :** l. 297 : « (bonus B8 ; les LLM eux-mêmes sont au B4) » ; l. 335 : « **B4** : les grands modèles de langage ; **B8** : DQN, gradient de politique, PPO et le RLHF, où ta mémoire de rejeu resservira. »

### 10. MINEUR · `chapitres/ch26_rl/figures/politiques.png`, produite par `tools/chapters/figures_ch26.py:181-194`
**Extrait :** `fig.suptitle("Une même ligne de la table (valeurs Q sous chaque action), trois politiques", fontsize=10)` sans `tight_layout` ni `subplots_adjust`.
**Problème :** le titre général chevauche les titres des trois panneaux (« ε-greedy, ε = 0,2 », « softmax, τ = 0,5 », « softmax, τ = 0,1 ») : on lit les deux lignes l'une sur l'autre. Les valeurs tracées sont justes (recalculées : 0,84/0,04 ; 0,16, 0,36, 0,07, 0,30, 0,11 ; 0,01, 0,72, 0,00, 0,27, 0,00).
**Correction :** `fig.suptitle(...)` puis `fig.tight_layout(rect=(0, 0, 1, 0.92))` (ou `plt.subplots(..., layout="constrained")`), et régénérer la figure.

### 11. MINEUR · `annexes/glossaire.md:811` (et ch. 25 : `01_fiche.md` ×3, `02_exercices.md`, `04_indices.md`, `05_solutions.md` ×3, `tools/chapters/build_ch25.py` ×5)
**Extrait :** « | sur-échantillonnage | upsampling | agrandir une image en répétant (ou en interpolant) ses valeurs, sans paramètres (`nn.Upsample`) | 25 | » ; la ligne 708 du même glossaire : « | suréchantillonnage | upsampling | agrandir un tenseur, en général en répétant chaque valeur (`nn.Upsample`) ; l'inverse du pooling | 20 | ».
**Problème :** le même terme anglais a deux entrées et deux graphies ; le ch. 25 le redéfinit avec glose (« deux **sur-échantillonnages** (*upsampling*) », fiche §25.7) alors que les ch. 20 et 21 ont fixé « suréchantillonnage ». BIBLE §5 : un terme est défini une fois, puis employé tel quel partout.
**Correction :** supprimer la ligne 811 (ou fusionner sa précision « sans paramètres » dans la ligne 708) ; remplacer « sur-échantillonnage » par « suréchantillonnage » dans les fichiers du ch. 25 et dans `build_ch25.py`, et retirer la glose (*upsampling*) de la fiche §25.7.

### 12. MINEUR · `chapitres/ch26_rl/01_fiche.md` (§26.3.7 et encadré 📐 « Conventions de la fiche »)
**Extrait :** « On note $s$ l'état, $a$ l'action, $r$ la récompense … comme Sutton et Barto et la bibliothèque Gymnasium » ; « $G_t = r_t + \gamma\, r_{t+1} + \dots$ ».
**Problème :** l'indice de la récompense diffère de Sutton et Barto, que la fiche recommande (« Pour aller plus loin ») : chez eux, la récompense qui suit l'action $A_t$ est $R_{t+1}$ et $G_t = R_{t+1} + \gamma R_{t+2} + \dots$ (2ᵉ éd., §3.3). La fiche annonce les notations « comme Sutton et Barto » sans signaler cet écart, que BIBLE §6 demande de signaler une fois. Un apprenant qui ouvre le livre de référence lira une formule décalée d'un indice.
**Correction :** ajouter à l'encadré 📐 : « Sutton et Barto notent $R_{t+1}$ la récompense qui suit l'action $A_t$ : leur $G_t = R_{t+1} + \gamma R_{t+2} + \dots$ est le même retour que notre $G_t = r_t + \gamma r_{t+1} + \dots$ (et celui de Gymnasium, où `step` rend la récompense de l'action jouée). »

## Suggestions

### 13. SUGGESTION · `chapitres/ch25_autoencodeurs/01_fiche.md` (§25.9.2, la KL)
**Extrait :** « $\mathrm{KL}\left(q(\mathbf{z} \mid \mathbf{x}) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\right)$ » ; le ch. 6 a fixé « $p$ décrit les données, $q$ le code ou le modèle » et « Retiens : $\mathrm{KL}(\text{données} \,\|\, \text{code})$ » (`ch06_information/01_fiche.md:38` et `:211`).
**Problème :** dans le VAE, les lettres s'inversent ($q$, la loi de l'encodeur, est en premier ; $p$ est le prior) et la KL est prise « à l'envers » de la règle mnémotechnique du ch. 6. Rien n'est faux, mais l'apprenant qui applique la règle du ch. 6 sera dérouté ; le ch. 6 le signale pour le RLHF, pas le ch. 25 pour le VAE.
**Correction :** une phrase après la formule fermée : « Attention aux lettres : ici $q$ est la loi que l'encodeur apprend et $p$ le prior fixé ; c'est la loi apprise qui est en premier, comme dans la pénalité du RLHF (ch. 6, 🕰️ « la KL aujourd'hui »). »

### 14. SUGGESTION · `chapitres/ch26_rl/05_solutions.md`, Ex 26.26, Variante
**Extrait :** « avec $\gamma = 0{,}9$, $V^*(0)$ ne vaut plus que **0,069** et la politique optimale réussit **71,6 %** des épisodes : elle change dans un état sur seize. Un but lointain compte moins : le long détour sûr coûte trop cher, et l'agent prend un chemin plus court et plus risqué. »
**Problème :** $V^*(0) = 0{,}0689$ et « un état sur seize » sont justes (itération sur les valeurs recalculée, `scripts/recalc26.py`). Mais l'écart 73,2 % → 71,6 % est mesuré sur 2 000 épisodes, soit une erreur type d'environ 1 point : il est du même ordre que le bruit. Sur 20 000 épisodes, je mesure 74,1 % (γ = 0,99) contre 73,3 % (γ = 0,9). La conclusion tient, mais le chiffre exagère l'effet, dans le chapitre même qui enseigne de rapporter un résultat avec son incertitude (🛠️ 26.34).
**Correction :** « …réussit environ 73 % des épisodes au lieu de 74 % (20 000 épisodes ; l'écart est petit) : elle change dans un état sur seize… ».


## Bilan

**Ce qui est solide.**
- **Les calculs.** Toutes les réponses recalculées sont justes, soit environ 75 pour le ch. 25 et 80 pour le ch. 26 : quiz chiffrés, ✏️, ∂ 25.5, 25.7, 25.8, 26.7, 26.8, 🧮 25.9, 25.10, 26.9, et toutes les variantes vérifiées. C'est le cas de la KL et de son gradient, de la reparamétrisation, des formes des convolutions transposées, des retours, des mises à jour Q et SARSA, de la remontée des valeurs, du point fixe de Bellman, et de $V^*(0) = 0{,}542$ et $0{,}069$ sur FrozenLake.
- **Les solutions et les sorties exécutées concordent**, sauf deux chiffres (constats 6 et 14). Cela vaut pour 25.14 à 25.39 et 26.13 à 26.38, et pour les historiques `history_ch25.json` et `history_ch26.json` (25.31, 25.34, 25.39, 26.16, 26.28).
- **Les encadrés « Le livre, corrigé » sont exacts.** J'ai vérifié chaque page citée et chaque renvoi de figure contre `book_ch25.txt` et `book_ch26.txt`. Les erreurs du livre sur le RL sont bien signalées et jamais reproduites :
  - γ présenté comme une « confiance » dans l'environnement ;
  - ε-greedy parmi les « autres » actions, avec ε de 0,01 ;
  - « softmax » $Q/\sum Q$ ;
  - α de 0,95 ;
  - le max présenté comme un « défaut » du Q-learning ;
  - 9 colonnes au morpion 4 × 4 ;
  - « un milliard d'univers » ;
  - « Baro, 2017 » ;
  - arXiv 1712.07403 ;
  - « Wired, 2017 ».
- **Aucune paraphrase proche du livre** dans les passages comparés (le banquier, le musicien, le peintre de cerisiers, le pissenlit) : ce sont des reprises d'idées, permises par le §3.
- **Les 🕰️ revérifiés sur le web, sans erreur :**
  - Gym n'est plus maintenu et Gymnasium lui succède ;
  - Stable-Baselines3 fait décroître ε de 1 à 0,05 sur 10 % de l'entraînement ;
  - InstructGPT : classements de K = 4 à 9 réponses, PPO, KL par token ;
  - GRPO se passe de réseau critique ;
  - AI Act, art. 50 : en vigueur le 2 août 2026 ; l'omnibus reporte au 2 décembre 2026 le seul marquage des systèmes déjà sur le marché et interdit la nudification (25.12) ;
  - T. White (2016) : l'exemple en 100 dimensions, des normes de 10 et 7 ;
  - `torch.load(weights_only=True)` par défaut depuis PyTorch 2.6 ;
  - latents à 16 canaux de SD3 et de FLUX.
- **Les flashcards sont justes** (25 et 30 cartes), sauf la carte du L-learning (constat 7).
- **Aucune contradiction de fond** avec les ch. 6, 11 et 12, CP6 ou MP6 :
  - la loss du VAE est sommée sur les pixels dans le ch. 25, CP6.8 et `genbench` ;
  - CP6.2 et CP6.3 recalculés.

**Les trois risques principaux.**
1. **Des annexes et une flashcard qui énoncent comme des faits ce que le chapitre réfute** (constats 1 et 7) : la mémorisation « dans les biais », le L-learning qui « marche » sans hasard. L'apprenant qui révise par les annexes retient l'erreur que le notebook corrige.
2. **Une fuite de réponse et des renvois faux** (constats 5, 8 et 9) : la fiche donne la réponse de 26.17 a, et le RLHF est renvoyé au B4 alors qu'il est au B8.
3. **Des codes de formulaire qui contredisent les conventions du chapitre** (constat 3) : une BCE moyennée, qui donne le β caché de 784, et un PSNR calculé sur le batch entier.

| Sévérité | Nombre |
|---|---|
| MAJEUR | 0 |
| MINEUR | 12 |
| SUGGESTION | 2 |
