# Checkpoint VI · Corrigé détaillé et barème de l'examen blanc

> À lire **après** l'examen, et après la partie B du notebook (vérification automatique du code et des réponses chiffrées). Pour chaque question : la réponse, la démarche, le barème, les erreurs fréquentes et les exercices à refaire si tu as eu moins de la moitié des points. Les solutions exécutées des deux questions de code sont dans `03_examen_solutions.ipynb`.

## Règles de notation

- **Barème par sous-question**, indiqué entre crochets. Une réponse juste sans démarche, quand la question en demande une, rapporte la moitié des points.
- **Méthode juste, erreur de calcul** : la moitié des points de la sous-question. **Erreur reportée** : si une réponse fausse est réutilisée correctement plus loin (en CP6.2, CP6.3 et CP6.13 surtout), la suite n'est pas pénalisée une seconde fois.
- **Arrondis** : une décimale de trop ou de moins coûte au plus 0,05 point par question ; un arrondi fait trop tôt, qui change la dernière décimale demandée, aussi.
- **Estimations** (CP6.10) : toute valeur qui s'arrondit à la réponse attendue rapporte tous les points ; un ordre de grandeur juste mais une valeur fausse, la moitié.
- **Code** (CP6.8, CP6.9) : les points vont aux propriétés du contrat (listées ci-dessous), que la partie B vérifie une à une ; un code qui lève une erreur garde les points des parties justes que tu peux montrer en le relisant.
- **Questions rédigées** (CP6.2 e, CP6.3 e, CP6.4 a, d et e, CP6.5 c, CP6.6 c, CP6.10 e, CP6.11, CP6.12, CP6.13 f) : les critères sont listés ; une idée juste formulée autrement compte autant.

---

## CP6.1 — Questions flash 🧠 · 2,5 points

0,25 point par item.

a) **24,5** : $784 / 32$.
b) **Faux** : un autoencodeur ordinaire n'oblige pas ses codes à remplir $\mathcal{N}(\mathbf{0}, \mathbf{I})$ ; un code tiré au hasard tombe souvent loin des codes des vraies images, et le décodeur rend une tache.
c) **B** : le prior est fixé d'avance, $\mathcal{N}(\mathbf{0}, \mathbf{I})$ ; la gaussienne de l'encodeur est le posterior approché $q(\mathbf{z} \mid \mathbf{x})$, propre à chaque image.
d) **0,81** : $0 + 0{,}9 \times 0 + 0{,}9^2 \times 1$.
e) **(16, 4)** : une ligne par case (état), une colonne par direction (action).
f) **0,5** : quand $p_g = p_{\text{data}}$, $D^* = \frac{p_{\text{data}}}{p_{\text{data}} + p_g} = \frac{1}{2}$ partout.
g) **Vrai**.
h) **(64, 64)** : une ligne et une colonne par canal ; les $14 \times 14 = 196$ positions sont sommées.
i) **70 000** (60 000 + 10 000).
j) **Vrai** : CC BY demande de citer la source et la licence (et d'indiquer les modifications) ; c'est NC qui interdit l'usage commercial (ce qui ne dispense pas de vérifier les droits des personnes photographiées, CP6.11).

**Erreurs fréquentes** : diviser dans l'autre sens en a) ; confondre prior et posterior en c) ; oublier le facteur $\gamma^2$ en d) ; écrire (4, 16) en e) ; répondre 1,386 en f) (c'est la loss de $D$, pas sa sortie) ; (196, 196) en h) ; 60 000 en i).
**Remédiation** : 25.1, 25.Q7, 25.Q12, 25.31, 26.1, 26.13, 27.4, 27.Q10, 28.1, 29.Q4, 29.Q2.

## CP6.2 — VAE : un tirage reparamétrisé et sa divergence KL ✏️ · 2 points

$\boldsymbol{\sigma} = e^{\frac{1}{2}\log\boldsymbol{\sigma}^2} = (1 ;\ 0{,}5)$.

a) **[0,4]** $\mathbf{z} = (0{,}5 + 1 \times 1{,}2 ;\ -1 + 0{,}5 \times (-0{,}4)) =$ **[1,7 ; −1,2]**.
b) **[0,6]** $\mathrm{KL} = \frac{1}{2}(\mu^2 + \sigma^2 - 1 - \ln\sigma^2)$ par latent : $\frac{1}{2}(0{,}25 + 1 - 1 - 0) = 0{,}125$ ; $\frac{1}{2}(1 + 0{,}25 - 1 - \ln 0{,}25) = \frac{1}{2}(0{,}25 + 1{,}3863) = 0{,}8181$ : **[0,1250 ; 0,8181]** (total 0,9431 nat). 0,3 pour la formule, 0,3 pour les valeurs.
c) **[0,4]** $z_2 = \mu_2 + e^{\frac{1}{2}\log\sigma_2^2}\,\varepsilon_2$, donc $\dfrac{\partial z_2}{\partial \log\sigma_2^2} = \frac{1}{2}\,\sigma_2\,\varepsilon_2 = \frac{1}{2} \times 0{,}5 \times (-0{,}4) =$ **−0,1000**.
d) **[0,3]** $\frac{1}{2}(0 + e^{-10} - 1 + 10) \approx$ **4,5000** nats.
e) **[0,3]** Un tirage $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ n'est pas une fonction dérivable de $\boldsymbol{\mu}$ et $\boldsymbol{\sigma}$ : la rétropropagation ne peut pas le traverser. En tirant $\boldsymbol{\varepsilon}$, qui ne dépend d'aucun poids, $\mathbf{z}$ devient une fonction dérivable de l'encodeur (0,15). d) montre que la KL pénalise lourdement une variance minuscule : sans elle, l'encodeur rendrait $\sigma \to 0$ et le VAE redeviendrait un autoencodeur ordinaire, aux codes isolés (0,15).

**Erreurs fréquentes** : prendre $\log\sigma^2$ pour $\sigma$ ($z_2 = -0{,}4455$) ou multiplier $\varepsilon$ par la variance ($z_2 = -1{,}1$) ; oublier le facteur $\frac{1}{2}$ ou le $-1$ de la KL ; oublier le $\frac{1}{2}$ de la dérivée en c).
**Remédiation** : 25.5, 25.7, 25.28, 25.29, 25.R3, 25.E3.

## CP6.3 — Deux transitions : Q-learning puis SARSA ✏️ · 2 points

Règle commune : $Q \leftarrow Q + \alpha\,(y - Q)$.

a) **[0,4]** Cible du Q-learning : $y = 0 + 0{,}9 \times \max(0{,}5 ;\ 0{,}3 ;\ 0{,}8) = 0{,}72$ ; $Q(A, 1) = 0{,}4 + 0{,}5 \times (0{,}72 - 0{,}4) =$ **0,5600**.
b) **[0,4]** C est terminal : pas d'amorçage, $y = r = 1$ ; $Q(B, 2) = 0{,}8 + 0{,}5 \times 0{,}2 =$ **0,9000**. Amorcer sur la ligne de C (valeur 0,6) donnerait 1,17 : la table « croirait » qu'il reste quelque chose à gagner après la fin de la partie.
c) **[0,4]** Cible de SARSA : $y = 0 + 0{,}9 \times Q(B, 0) = 0{,}45$ ; $Q(A, 1) = 0{,}4 + 0{,}5 \times 0{,}05 =$ **0,4250**.
d) **[0,4]** Gloutonne avec probabilité $0{,}8$, plus le tirage au hasard qui tombe sur elle : $0{,}8 + 0{,}2 / 3 =$ **0,8667**.
e) **[0,4]** SARSA amorce sur l'action **réellement** choisie par sa politique ε-greedy : il apprend la valeur de cette politique, exploration comprise (*on-policy*). Le Q-learning amorce sur le maximum : il apprend la valeur de la politique gloutonne, quelle que soit celle qui génère les données (*off-policy*) (0,2). Conséquence, au choix (0,2) : au bord d'une falaise, SARSA apprend un chemin plus sûr, qui tient compte des pas d'exploration ; le Q-learning peut apprendre à partir de données d'une autre politique (rejeu d'expérience, démonstrations).

**Erreurs fréquentes** : le max à la place de l'action jouée (ou l'inverse) ; oublier $\gamma$ ; remplacer au lieu de mélanger (0,72) ; amorcer sur un état terminal ; oublier que l'exploration peut tirer l'action gloutonne (0,8 en d).
**Remédiation** : 26.3, 26.4, 26.5, 26.20, 26.Q8, 26.Q11, 26.E1.

## CP6.4 — Discriminateur optimal et losses à l'équilibre ∂ · 2 points

a) **[0,7]** $V(D) = \int \bigl[p_{\text{data}}(\mathbf{x}) \ln D(\mathbf{x}) + p_g(\mathbf{x}) \ln(1 - D(\mathbf{x}))\bigr]\,d\mathbf{x}$ : on maximise point par point (0,2). Avec $a = p_{\text{data}}(\mathbf{x})$, $b = p_g(\mathbf{x})$ et $f(y) = a \ln y + b \ln(1 - y)$ : $f'(y) = \frac{a}{y} - \frac{b}{1 - y} = 0 \iff y = \frac{a}{a + b}$ (0,3) ; $f''(y) = -\frac{a}{y^2} - \frac{b}{(1 - y)^2} < 0$, c'est un maximum (0,2).
b) **[0,4]** $D^* = \frac{1}{2}$ partout : la loss de $D$ vaut $-\ln\frac{1}{2} - \ln\frac{1}{2} = 2\ln 2$, celle de $G$, $-\ln\frac{1}{2} = \ln 2$ : **[1,386 ; 0,693]**.
c) **[0,3]** $D^* = 0{,}3 / (0{,}3 + 0{,}1) =$ **0,7500**.
d) **[0,3]** Avec $D = \sigma(\ell)$ ($\ell$, le logit), $\frac{\partial}{\partial \ell} \ln(1 - \sigma(\ell)) = -\sigma(\ell) \approx 0$ quand $D \approx 0$ : le générateur, que $D$ rejette facilement au début, ne reçoit presque aucun signal (0,15). La loss non saturante $-\ln D(G(\mathbf{z}))$ a pour dérivée $-(1 - \sigma(\ell)) \approx -1$ : un gradient fort justement quand $G$ est mauvais ; même point fixe, autre dynamique (0,15).
e) **[0,3]** La loss de $D$ (somme des deux BCE) vaut $2 \ln 2 \approx 1{,}39$ à l'équilibre, pas 0,5 ; 0,5 est la **sortie** de $D$ (0,1). Et la valeur des losses ne mesure pas la qualité : elles oscillent, et un $D$ qui gagne (loss vers 0) peut cacher un $G$ bloqué (0,1). On juge sur des grilles d'échantillons pour un bruit fixé, une mesure comme la distance de Fréchet (FID), la couverture des classes ou des modes (0,1).

**Erreurs fréquentes** : dériver $V$ par rapport à $\mathbf{x}$ au lieu de $D(\mathbf{x})$ ; oublier de vérifier le maximum ; prendre la moyenne des deux BCE pour la loss de $D$ (0,693) ; écrire $D^* = p_g / (p_{\text{data}} + p_g)$ (0,25).
**Remédiation** : 27.3, 27.4, 27.5, 27.6, 27.14, 27.18, 27.E1.

## CP6.5 — Matrice de Gram de trois cartes d'activation ✏️ · 1 point

a) **[0,4]** Produits scalaires des lignes : **[[6, 2, 3], [2, 2, 1], [3, 1, 6]]**.
b) **[0,3]** $CHW = 3 \times 2 \times 2 = 12$ : $G_{00} = 6/12 =$ **0,5000** et $G_{02} = 3/12 =$ **0,2500**.
c) **[0,3]** Rien ne change : chaque coefficient est une somme sur les positions, et une même permutation des positions dans toutes les cartes ne change aucune somme (0,15). La matrice de Gram retient quels filtres s'activent **ensemble** et avec quelle énergie, pas **où** : c'est une statistique de texture, indépendante de la composition de la scène, d'où son usage comme description du style (0,15).

**Erreurs fréquentes** : calculer $F^\top F$ (4 × 4, les positions au lieu des canaux) ; diviser par $HW$ ou par $C$ seulement ; croire qu'une permutation des positions change $G$.
**Remédiation** : 28.2, 28.3, 28.18, 28.Q7, 28.8.

## CP6.6 — Prédire : γ → 0 puis ε = 0 sur Flippers 🔮 · 1 point

Mesures réelles (`tools/chapters/figures_cp6.py`, trois graines, `mylearn` de référence) : avec $\gamma = 0{,}9$ et $\varepsilon = 0{,}1$, la politique gloutonne gagne 82 % à 93 % des 506 plateaux non gagnants.

a) **[0,4]** **C** : avec $\gamma = 0$, elle n'en gagne que **20,2 %** (pour les trois graines), en 1,5 coup en moyenne. La cible est $y = r$ : seules les actions qui gagnent immédiatement reçoivent une valeur non nulle ; ailleurs, toute la ligne reste à 0, la politique gloutonne joue l'action 0 (`np.argmax`) et tourne en rond.
b) **[0,4]** **A** : avec $\varepsilon = 0$, elle gagne **100 %** des plateaux (pour les trois graines), en 5 coups environ au lieu de 2,7 au mieux. Tant que la table est nulle, les égalités se départagent au hasard pendant l'entraînement : l'agent explore quand même, comme une marche au hasard ; dès qu'une victoire est trouvée, $\gamma = 0{,}9$ propage sa valeur vers les plateaux d'avant. Ses parties restent pourtant presque deux fois plus longues que l'optimum, comme celles de l'agent ε = 0,1 (4,2 à 5,3 coups) : 3 000 parties ne suffisent à aucun des deux pour apprendre les chemins les plus courts.
c) **[0,2]** Une phrase juste pour chacune (0,1 chacune) : γ = 0 rend l'agent myope, l'avenir ne compte plus ; ε = 0 n'empêche pas l'exploration quand la table est vide et que les égalités se tirent au hasard.

**Erreurs fréquentes** : croire que sans ε l'agent n'explore jamais (c'est vrai avec des égalités départagées par l'indice, pas ici) ; croire que γ = 0 ne fait que « raccourcir » les parties.
**Remédiation** : 26.7, 26.27, 26.Q6, 26.Q9, 26.Q10, 26.24.

## CP6.7 — Trois courbes, trois diagnostics 📈 · 1 point

Un tiers de point par panneau (0,33 ; 1 point pour les trois).

a) **A** : la KL tombe à 0,004 nat et y reste, la reconstruction plafonne vers 208 nats par image (un VAE normal descend vers 120 sur ces données). Ce VAE avait $\beta = 60$ : le code ne porte plus d'information, le décodeur rend une image moyenne (l'effondrement du posterior).
b) **C** : la loss de $D$ est presque nulle dès les premiers pas, celle de $G$ reste entre 5 et 10. Ici, le discriminateur avait un learning rate de $10^{-2}$ et le générateur de $10^{-5}$ : les exemples générés restent autour de −0,2, loin de la moyenne 4 des données. À l'équilibre (D), on aurait 1,39 et 0,69.
c) **F** : 64 victoires en 3 000 épisodes ; le retour moyen reste sous 0,06 (il monte à peine : l'agent apprend un tout petit peu, très loin d'une bonne politique). Sur la grande carte glissante, avec des épisodes coupés à 60 pas, l'agent n'atteint presque jamais le but : sa table reste presque vide. Une récompense façonnée, plus d'épisodes ou une carte plus petite le débloqueraient.

**Erreurs fréquentes** : lire en ① une « convergence rapide » sans regarder la KL ; prendre ② pour un équilibre parce que les courbes sont plates ; prendre ③ pour une politique déjà optimale parce que la courbe est plate.
**Remédiation** : 25.34, 25.36, 27.14, 27.17, 26.23, 26.32.

## CP6.8 — Écrire la loss d'un VAE en PyTorch 🔨 · 2 points

```python
def vae_loss(logits, x, mu, log_var, beta=1.0):
    n = x.shape[0]
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / n     # sum over pixels, mean over images
    kl = 0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum() / n                 # sum over latents, mean over images
    return rec + beta * kl, rec, kl
```

a) **[1,4]** Barème des propriétés vérifiées par la partie B : trois tenseurs à 0 dimension (0,15) ; `rec` sur les logits, sommée sur les pixels et moyennée sur les images (0,35) ; `kl` avec son facteur $\frac{1}{2}$ et $\sigma^2 = e^{\log\sigma^2}$, sommée sur les latents et moyennée sur les images (0,35) ; `loss = rec + beta * kl`, `beta` venant de l'argument (0,2) ; d'autres tailles, images aplaties (0,15) ; le gradient remonte jusqu'à `logits`, `mu` et `log_var` (0,2).
b) **[0,3]** **[829,45 ; 2,75]**. La reconstruction est grande parce que ces logits sont tirés au hasard : un décodeur entraîné descendrait vers 100 nats par image.
c) **[0,3]** **837,68** ($829{,}45 + 3 \times 2{,}75$).

**Erreurs fréquentes** : `reduction="mean"` (une reconstruction 784 fois trop petite : un β caché, ch. 25) ; `binary_cross_entropy` sur des logits, ou une sigmoïde appliquée deux fois ; lire `log_var` comme $\log\sigma$ ; moyenner la KL sur les latents ; `.item()` dans la fonction (plus de gradient) ; oublier la division par le nombre d'images ; `beta` qui multiplie la reconstruction.
**Remédiation** : 25.30, 25.29, 25.36, 25.34, 25.38.

## CP6.9 — Réparer un pas d'entraînement de GAN 🐛 · 1,5 point

a) **[0,6]** **« CGJ »** (0,2 par lettre juste, −0,1 par lettre en trop, sans descendre sous 0). (I) à la place de (C) désigne l'endroit où l'erreur se déclare, pas la ligne à corriger : 0,1 pour cette lettre, sans la pénalité d'une lettre en trop.
- **C** : `bce(D(fake.detach()), zeros)`. Sans `detach`, le `backward` de E traverse aussi le générateur et **libère son graphe** : le `backward` de I échoue alors (« Trying to backward through the graph a second time »). L'erreur se déclare en I, mais la cause est en C.
- **G** : `bce(D(fake), ones)`. Le générateur veut que $D$ prenne ses faux pour des vrais : label 1 (la loss non saturante). Avec le label 0, il apprendrait à être reconnu, l'inverse de son but.
- **J** : `opt_g.step()`. Avec `opt_d.step()`, le générateur ne bouge jamais, et le discriminateur fait un second pas avec les gradients de la loss de $G$, qui le poussent à se tromper.

b) **[0,9]** Propriétés vérifiées par la partie B : le pas ne lève pas d'erreur et rend deux `float` (0,2) ; après trois pas, les poids de $D$ (0,25) et de $G$ (0,25) sont exactement ceux du pas de référence ; après 1 500 pas, la moyenne générée est proche de 4 (0,2). La solution exécutée obtient une moyenne de 4,05 et un écart-type de 1,26 (données : 4 et 1,25) ; sur les 100 derniers pas, la loss de $D$ vaut 1,386 et celle de $G$ 0,696 : l'équilibre de CP6.4 b).

**Erreurs fréquentes** : désigner I, la ligne où l'erreur apparaît ; corriger C par `retain_graph=True` au lieu de `detach` (les poids obtenus sont les mêmes, car `opt_g.zero_grad()` efface ensuite les gradients que le pas de $D$ a laissés dans $G$ : la partie B l'accepte, mais le pas de $D$ rétropropage inutilement jusqu'au générateur et garde son graphe en mémoire ; la lettre attendue reste C) ; tirer un second `torch.randn` pour le pas de $G$ (une variante correcte, mais elle change la référence ; l'énoncé l'interdisait) ; oublier `opt_g.zero_grad()` en recopiant.
**Remédiation** : 27.12, 27.16, 27.19, 27.Q4, 27.E1.

## CP6.10 — Table Q ou réseau 🧮 · 1 point

0,2 point par item.

a) **0,71 Mo** : $3^9 \times 9 \times 4 = 19\,683 \times 36 = 708\,588$ octets.
b) **0,20 Mo** : $5\,478 \times 9 \times 4 = 197\,208$ octets.
c) **2,755 Go** : $3^{16} \times 16 \times 4 = 43\,046\,721 \times 64 \approx 2{,}755 \times 10^9$ octets.
d) **24 848** : $(48 \times 128 + 128) + (128 \times 128 + 128) + (128 \times 16 + 16) = 6\,272 + 16\,512 + 2\,064$.
e) La **généralisation** : des plateaux voisins ont des valeurs voisines, et le réseau en estime pour des plateaux qu'il n'a jamais vus, là où une table n'apprend rien d'une case jamais visitée (au prix de la garantie de convergence du tabulaire).

**Erreurs fréquentes** : $2^9$ au lieu de $3^9$ ; 9 colonnes pour le 4 × 4 ; oublier les 4 octets ; une entrée de 16 valeurs au lieu de 48 ; oublier les biais.
**Remédiation** : 26.2, 26.9, 26.Q12, 26.E4.

## CP6.11 — Un dataset de visages pour un produit commercial ⚖️ · 1 point

**Réponse modèle** : avant d'utiliser un dataset, on vérifie sa **licence** (usage commercial permis ?), les **droits des personnes** (visages = données personnelles, consentement, base légale au sens du RGPD, droit à l'image), la **provenance** et l'**histoire** du dataset (retraits, versions corrigées), et ses **biais** (qui est représenté). CelebA est réservé à la recherche non commerciale : non pour l'application. Ses photos de célébrités ont été collectées en ligne sans leur consentement, et un générateur entraîné dessus peut reproduire des visages reconnaissables. Proposer à la place : un dataset sous licence commerciale explicite et documenté, ou des photos collectées avec un consentement écrit pour cet usage, avec une data card ; et signaler les images générées comme telles (le règlement européen sur l'IA impose de marquer les contenus générés).

**Critères** (1 point) : trois vérifications pertinentes (0,4) ; la conclusion juste sur CelebA, avec sa raison (0,3) ; une alternative réaliste (0,3). « Public, donc libre » ne rapporte rien.
**Erreurs fréquentes** : confondre « téléchargeable » et « libre d'usage » ; ne vérifier que la licence, en oubliant que des visages sont des données personnelles (consentement, RGPD) ; croire que citer la source rend permis un usage que la licence interdit.
**Remédiation** : 29.5, 29.Q2, 29.Q8, 29.E4, 27.9.

## CP6.12 — GAN, VAE ou diffusion, en 60 secondes 💼 · 1 point

**Réponse modèle en 60 secondes** : « Un VAE apprend un encodeur et un décodeur avec un espace latent régulier : entraînement stable, une vraie loss à suivre, des interpolations faciles, mais des images souvent floues. Un GAN met un générateur face à un discriminateur : des images nettes et une génération en une passe, mais un entraînement instable, de l'effondrement de mode, et pas de mesure de vraisemblance. La diffusion apprend à retirer du bruit pas à pas : la meilleure qualité et la meilleure diversité aujourd'hui, un entraînement stable, mais une génération lente, en plusieurs dizaines de pas, et un coût de calcul élevé. En pratique, la diffusion travaille souvent dans l'espace latent d'un autoencodeur, comme Stable Diffusion. Pour générer des images de produits en haute qualité, je partirais d'un modèle de diffusion latente pré-entraîné, avec un fine-tuning sur nos données ; pour de la génération en temps réel sur mobile, un GAN distillé ou un modèle de diffusion en peu de pas. »

**Critères** (1 point) : le principe de chacun (0,3) ; une force et une faiblesse de chacun (0,4) ; un choix justifié par un cas concret (0,3).
**Relances possibles** : « Pourquoi la diffusion a-t-elle supplanté les GAN pour les images ? » (un entraînement stable, une meilleure couverture des modes, le contrôle par le texte, au prix de nombreuses passes à la génération) ; « Et pour générer en temps réel ? » (un GAN, ou un modèle de diffusion distillé, en une ou quelques passes).
**Erreurs fréquentes** : confondre le VAE et l'autoencodeur simple, qui ne sait pas générer ; dire que la diffusion est rapide à l'échantillonnage ; réciter les trois principes sans jamais choisir pour un cas concret.
**Remédiation** : 27.E3, 25.E5, 28.E4, 25.11, 27.8.

## CP6.13 — Retour sur les parties IV et V ✏️ · 2 points

a) **[0,3]** $\lfloor (64 + 2 \times 2 - 5)/2 \rfloor + 1 = 32$ : **(1, 32, 32, 32)**.
b) **[0,3]** $32 \times 3 \times 5 \times 5 + 32 =$ **2 432**.
c) **[0,3]** $(32 - 1) \times 2 - 2 \times 1 + 4 = 64$ : **(1, 16, 64, 64)**.
d) **[0,4]** $m_1 = 0{,}1 \times 0{,}5 = 0{,}05$, $v_1 = 0{,}001 \times 0{,}25 = 0{,}00025$ ; corrigés : $\hat m_1 = 0{,}05 / 0{,}1 = 0{,}5$, $\hat v_1 = 0{,}00025 / 0{,}001 = 0{,}25$ ; pas $= 0{,}01 \times 0{,}5 / (0{,}5 + 10^{-8}) \approx 0{,}01$ : $w_1 =$ **0,9900**.
e) **[0,5]** $m_2 = 0{,}9 \times 0{,}05 + 0{,}1 \times (-0{,}1) = 0{,}035$, $v_2 = 0{,}999 \times 0{,}00025 + 0{,}001 \times 0{,}01 = 0{,}00025975$ ; corrigés par $1 - 0{,}9^2 = 0{,}19$ et $1 - 0{,}999^2 = 0{,}001999$ : $\hat m_2 = 0{,}18421$, $\hat v_2 = 0{,}12994$, $\sqrt{\hat v_2} = 0{,}36047$ ; pas $= 0{,}01 \times 0{,}18421 / 0{,}36047 = 0{,}00511$ : $w_2 =$ **0,9849**. Le gradient a changé de signe, mais le momentum garde la direction du premier : le poids continue de baisser.
f) **[0,2]** Au premier pas, $\hat m_1 = g_1$ et $\hat v_1 = g_1^2$ : le pas vaut $\eta\, g_1 / (|g_1| + \epsilon) \approx \eta \times \operatorname{signe}(g_1)$, quelle que soit la taille de $g_1$.

**Erreurs fréquentes** : oublier le padding des deux côtés (30) ou le stride (64) en a) ; oublier les canaux d'entrée en b) ; appliquer la formule de la convolution ordinaire en c) (16) ; oublier la correction du biais (0,9684) ou faire un pas de SGD (0,995) en d) ; corriger le 2ᵉ pas avec $1 - \beta$ au lieu de $1 - \beta^2$ (0,9831) en e).
**Remédiation** : 21.3, 21.4, 21.6, 25.4, 19.5, 19.8, 19.18.

---

## Barème récapitulatif et remédiation

| Question | Points | Ch. | Si tu as moins de la moitié des points, refais… |
|---|---|---|---|
| CP6.1 | 2,5 | 25 à 29 | 25.1, 25.Q7, 25.Q12, 25.31, 26.1, 26.13, 27.4, 27.Q10, 28.1, 29.Q4, 29.Q2 |
| CP6.2 | 2 | 25 | 25.5, 25.7, 25.28, 25.29, 25.R3, 25.E3 |
| CP6.3 | 2 | 26 | 26.3, 26.4, 26.5, 26.20, 26.Q8, 26.Q11, 26.E1 |
| CP6.4 | 2 | 27 | 27.3, 27.4, 27.5, 27.6, 27.14, 27.18, 27.E1 |
| CP6.5 | 1 | 28 | 28.2, 28.3, 28.18, 28.Q7, 28.8 |
| CP6.6 | 1 | 26 | 26.7, 26.27, 26.Q6, 26.Q9, 26.Q10, 26.24 |
| CP6.7 | 1 | 25 à 27 | 25.34, 25.36, 27.14, 27.17, 26.23, 26.32 |
| CP6.8 | 2 | 25 | 25.30, 25.29, 25.36, 25.34, 25.38 |
| CP6.9 | 1,5 | 27 | 27.12, 27.16, 27.19, 27.Q4, 27.E1 |
| CP6.10 | 1 | 26 | 26.2, 26.9, 26.Q12, 26.E4 |
| CP6.11 | 1 | 29, 27 | 29.5, 29.Q2, 29.Q8, 29.E4, 27.9 |
| CP6.12 | 1 | 25, 27, 28 | 27.E3, 25.E5, 28.E4, 25.11, 27.8 |
| CP6.13 | 2 | 19, 21 | 21.3, 21.4, 21.6, 25.4, 19.5, 19.8, 19.18 |
| **Total** | **20** | | |

**Lire ta note**
- **16 à 20** : la partie VI est solide. Passe au mini-projet MP6 ; garde les flashcards des chapitres 25 à 29 dans tes révisions.
- **12 à moins de 16** : c'est bien. Fais la remédiation des questions où tu as eu moins de la moitié des points, puis le mini-projet.
- **8 à moins de 12** : avant le mini-projet, refais (ou fais, s'il n'est pas dans ton parcours) le 📦 25.30 (un VAE en PyTorch), les 🔨 26.20 et 26.21 (Q-learning, SARSA et la boucle tabulaire), le 🐛 27.16 et le 📦 27.12 (un GAN et ses pièges) ; relis la synthèse (`05_synthese.md`), puis repasse les questions ratées sans le corrigé.
- **Moins de 8** : fais tout ce qui précède, reprends en plus les exercices ★★ des chapitres où tu as perdu le plus de points, puis refais l'examen avant le mini-projet.

Dans tous les cas, **refais dans une semaine** les questions où tu as eu moins de la moitié des points, sans regarder ce corrigé : c'est l'effort de rappel qui fixe les notions. Reporte ta note, le temps réellement passé sur chaque question et tes points faibles dans la section CP6 de `mon_travail/suivi/auto_evaluation.md`.
