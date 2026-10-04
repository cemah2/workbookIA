# Audit P5, tour 2 : exactitude des chapitres 1, 2 et 3 (relecteur « exact1 »)

**Périmètre lu**
- Les trois chapitres `chapitres/ch01_introduction/`, `ch02_stats/` et `ch03_probabilites/` : fiche, exercices, indices, solutions .md, flashcards et les 19 figures, ouvertes une à une.
- Pour les notebooks : les cellules Markdown des deux notebooks et les sorties de `05_solutions.ipynb`.
- Le formulaire, sections 0 à 3 (L1–L215), et les entrées des ch. 1 à 3 du glossaire.
- Dans les annexes : `erreurs_frequentes.md` et `cheatsheets/sklearn.md`.

**Méthode**
- Le dépôt n'a été que lu. Les calculs ont été refaits avec la pile figée (`/home/claude/venv313` : NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, SciPy 1.16.3) ; les scripts de figures ont été relus.
- 13 affirmations datées ont été vérifiées sur le web.
- La cohérence avec 0A, 0B, les ch. 4 à 11, CP1, CP2 et les annexes a été cherchée par grep.

**Non re-signalé** : ce que traitent les §7–8 de `docs/AUDIT_parties_I_II.md` (P1 à P5, et les constats A1 déjà appliqués) et les décisions du §22.

**Résultat : aucun constat MAJEUR ; 10 MINEURS, 7 SUGGESTIONS.** Toutes les valeurs numériques recalculées des trois chapitres sont justes (voir « Contrôles sans constat »).

## Constats

### 1. MINEUR · `chapitres/ch02_stats/05_solutions.md:147` (Ex 2.10, point 4) · fait daté dépassé (AI Act)
- **Extrait** : « Dans l'Union européenne, les systèmes d'IA utilisés pour recruter ou trier des candidatures sont classés **à haut risque** par l'AI Act (règlement (UE) 2024/1689, annexe III, point 4) : gestion des risques, qualité et gouvernance des données, contrôle humain et documentation y sont obligatoires. »
- **Problème** : le présent « y sont obligatoires » est faux au 2026-10-04, et l'apprenant peut le répéter en entretien.
  - L'« omnibus numérique sur l'IA » a été publié au JO le 24 juillet 2026 et est en vigueur depuis le 27 juillet 2026.
  - Il repousse l'application des obligations des systèmes à haut risque de l'annexe III (emploi compris) du 2 août 2026 au **2 décembre 2027**. Pour les composants de sécurité des produits de l'annexe I, la nouvelle date est le 2 août 2028.
  - Le classement « à haut risque » reste juste : l'indice 3 (`04_indices.md:469`), qui ne dit que cela, est juste.
- **Correction** : « … sont classés **à haut risque** par l'AI Act (règlement (UE) 2024/1689, annexe III, point 4) : gestion des risques, qualité et gouvernance des données, contrôle humain et documentation y seront obligatoires à partir du **2 décembre 2027** (date repoussée en 2026 par l'« omnibus numérique » ; elle était fixée au 2 août 2026). Le texte évolue : consulte sa version consolidée. »
  - Ajouter une source à la ligne *Sources* (L149), par exemple White & Case, « EU AI Omnibus enters into force, amending the AI Act » (2026).

### 2. MINEUR · indices de niveau 3 qui donnent la réponse (ch. 1, 2 et 3) · groupé
La BIBLE (§12) demande : « Indice 3 (presque la solution : pseudo-code ou première ligne) ». Les ch. 4 à 9 s'arrêtent au calcul posé (constat A1 n° 13). Les ch. 1 à 3, eux, donnent souvent la réponse elle-même.

**(a) Valeurs que `wb.check` vérifie en partie 0** (même défaut que celui, déjà connu, des ch. 10 et 11) :
- `chapitres/ch03_probabilites/04_indices.md:303` (Ex 3.1) : « Aires : mur 10, A 3, B 2, commune 1. $P(A \mid B) = \frac{1}{2}$, $P(B \mid A) = \frac{1}{3}$. »
  - Ce sont les réponses de 3.1 d et 3.1 e (`3.1d`, `3.1e` dans `src/wb/answers.json`).
- `chapitres/ch03_probabilites/04_indices.md:411` (Ex 3.7) : « 500 malades et 49 500 personnes saines : TP = 495, FP = 495. »
  - Ce sont les réponses de 3.7 a, b et d ; c et e s'en déduisent par soustraction (`3.7a` à `3.7e`).

**(b) Quiz et rappels** : l'indice 3 énumère les items vrais ou les résultats.
- ch. 1, `04_indices.md` :
  - L25 (Q1 « Deux programmes apprennent (le 2 et le 4) ») ;
  - L43 (Q2 « Deux affirmations sont vraies (la 2 et la 5). ») ;
  - L61 (Q3) ;
  - L97 (Q5 « Paramètres : 1, 4 et 6. Hyperparamètres : 2, 3 et 5. ») ;
  - L133 (Q7 « Régression : 1, 3 et 5. Classification : 2, 4 et 6 ») ;
  - L151 (Q8) ;
  - L205 (Q11) ;
  - L263 (R3 « $w = 2$ et $b = 1$ : la droite est $y = 2x + 1$ ») ;
  - L585 (Ex 1.13 « Un seul dataset interdit l'usage commercial : celui des taches solaires (CC BY-NC 4.0). Le raccourci de Penguins : l'espèce se devine à l'île. »).
- ch. 2, `04_indices.md` :
  - L43 (Q2 « Trois affirmations sont vraies : la 3, la 5 et la 6. ») ;
  - L61 (Q3 « Deux affirmations sont vraies : la 1 et la 2. ») ;
  - L115 (Q6) ;
  - L133 (Q7 : les cinq valeurs) ;
  - L151 (Q8 « Seules les situations 1 et 5 sont i.i.d. ») ;
  - L169 (Q9) ;
  - L187 (Q10 « Trois affirmations sont vraies : la 2, la 3 et la 6. »).
- ch. 3, `04_indices.md` :
  - L151 (Q8 : le classement des cinq situations) ;
  - L169 (Q9 « Accuracy 0,99, recall 0. »).

Le modèle à suivre existe déjà : les autres indices 3 des ch. 2 et 3 (papier 2.1 à 2.8, 3.2 à 3.6 et 3.8 ; quiz 3.Q1 à Q7 et Q10 à Q12) posent le calcul ou le raisonnement sans énoncer la réponse.

**Correction**
- (a) Ex 3.1 : « Aires : mur 10, A 3, B 2, commune 1. $P(A \mid B) = \frac{\text{aire commune}}{\text{aire de } B}$, $P(B \mid A) = \frac{\text{aire commune}}{\text{aire de } A}$. … » (la suite inchangée).
- (a) Ex 3.7 : « Malades : 1 % de 50 000 ; TP : 99 % des malades ; FP : 1 % des personnes saines. Precision $= \frac{TP}{TP + FP}$… » (la suite inchangée).
- (b) Donner le critère et l'appliquer à **un** item, sans désigner les items vrais. Exemples :
  - ch. 1 Q5 : « Le learning rate (2) est fixé avant l'entraînement : c'est un hyperparamètre. Pose la même question (« qui le fixe, et quand ? ») aux cinq autres. »
  - ch. 2 Q8 : « La situation 1 est i.i.d. : même dé, et un lancer ne renseigne pas sur le suivant. Pose les deux mêmes questions aux autres. »
  - Ex 1.13 : « Cherche la ligne « Licence » de chaque data card : une seule mentionne NC (*non commercial*). »

### 3. MINEUR · `chapitres/ch03_probabilites/flashcards.csv:4` · carte fausse dans un cas
- **Extrait** : « Pourquoi \(P(A \mid B)\) et \(P(B \mid A)\) diffèrent-elles en général ?;Même numérateur \(P(A, B)\), mais dénominateurs différents : \(P(B)\) pour l'une, \(P(A)\) pour l'autre. Elles ne sont égales que si \(P(A) = P(B)\). »
- **Problème** : la carte est fausse quand $P(A, B) = 0$.
  - Exemple : deux taches disjointes avec $P(A) = 0{,}3$ et $P(B) = 0{,}2$ donnent $P(A \mid B) = P(B \mid A) = 0$.
  - Le chapitre le dit lui-même : `05_solutions.md:91` (∂ 3.4, question 2) « Si $P(A, B) > 0$, alors $P(A \mid B) = P(B \mid A)$ exactement quand $P(A) = P(B)$ ; si $P(A, B) = 0$, les deux valent 0. »
  - Le 💡 de 3.13 aussi (`tools/chapters/build_ch03.py:531`) : « (ou si les disques ne se touchent pas : 0 = 0) ».
- **Correction** : « … Quand \(P(A, B) > 0\), elles ne sont égales que si \(P(A) = P(B)\) (si A et B ne se chevauchent pas, les deux valent 0). »

### 4. MINEUR · `chapitres/ch03_probabilites/01_fiche.md:436` · nombre de cas par intervalle de `figures/calibration.png`
- **Extrait** : « Un diagramme de fiabilité demande beaucoup de données : ici, les intervalles comptent de 120 à 600 cas environ. »
- **Problème** : c'est vrai pour le modèle calibré seulement. Sur la même figure, le modèle trop sûr de lui a de 224 à 872 cas par intervalle.
- Calcul, avec le code de `tools/chapters/figures_ch03.py:265-275` (graine 9) :
  ```python
  rng = np.random.default_rng(9); n = 4000
  q = rng.beta(2, 2, n); y = (rng.random(n) < q).astype(int)
  over = 1 / (1 + np.exp(-2.5 * np.log(q / (1 - q))))
  edges = np.linspace(0, 1, 11)
  np.bincount(np.searchsorted(edges[1:-1], p, side="right"), minlength=10)   # p = q, puis over
  # calibré  : [121, 319, 449, 529, 593, 535, 559, 470, 306, 119]
  # trop sûr : [851, 380, 299, 235, 246, 232, 224, 269, 392, 872]
  ```
- **Correction** : « ici, les intervalles du modèle calibré comptent de 120 à 600 cas environ (ceux du modèle trop sûr de lui, de 220 à 870). »

### 5. MINEUR · `annexes/cheatsheets/sklearn.md:102` · valeur par défaut de `calibration_curve`
- **Extrait** : « | `from sklearn.calibration import calibration_curve, CalibrationDisplay` | diagramme de fiabilité (`n_bins=10`, `strategy="uniform"`) | »
- **Problème** : la ligne se lit comme les valeurs par défaut, qui sont fausses.
  - Dans scikit-learn 1.6.1, `inspect.signature` donne `(y_true, y_prob, *, pos_label=None, n_bins=5, strategy='uniform')`, et `CalibrationDisplay.from_predictions(..., n_bins=5, ...)`.
  - `mylearn.metrics.calibration_curve` prend, elle, `n_bins=10` par défaut (`templates/mylearn_stubs/metrics.py:636`).
  - L'apprenant qui compare les deux sans argument obtient 5 intervalles contre 10.
- **Correction** : « diagramme de fiabilité : `calibration_curve(y_true, proba, n_bins=10)` ; par défaut `n_bins=5` (10 dans `mylearn.metrics.calibration_curve`) et `strategy="uniform"` ».

### 6. MINEUR · « régression = prédire un nombre », contraire à la leçon de 1.Q7 · groupé
- **Extraits** :
  - `chapitres/ch01_introduction/flashcards.csv:11` : « …a donné son nom aux méthodes qui prédisent un nombre. » ;
  - `chapitres/ch01_introduction/05_solutions.md:19` (1.Q3) : « on passe d'une **régression** (prédire un nombre) à une **classification** » ;
  - titre de `chapitres/ch01_introduction/figures/regression_droite_courbe.png` (`tools/chapters/figures_ch01.py:81`) : « Régression : prédire un nombre à partir d'une entrée » ;
  - `annexes/glossaire.md:167` : « prédire une catégorie / prédire une quantité (un nombre) ».
- **Problème** : la fiche dit « quantité » (L55 « **régression** (prédire une quantité) »), et le chapitre insiste sur la différence :
  - un code postal est « une catégorie écrite avec des chiffres » (1.Q7, `04_indices.md:133`) ;
  - le tableau des pièges (`01_fiche.md:349`) dit « une quantité qui se mesure : régression ».
  - « Prédire un nombre » réintroduit exactement l'erreur que 1.Q7 corrige.
- **Correction** :
  - flashcard : « …aux méthodes qui prédisent une quantité. » ;
  - 1.Q3 : « (prédire une quantité) » ;
  - figure : titre « Régression : prédire une quantité à partir d'une entrée », puis régénérer la figure ;
  - glossaire : « prédire une catégorie / prédire une quantité (un nombre qui se mesure) ».

### 7. MINEUR · `annexes/glossaire.md:236` (entrée « paradoxe de Simpson », ch. 2) · « profondeur » au lieu d'« épaisseur » du bec
- **Extrait** : « (longueur et profondeur du bec des manchots : corrélation négative sur l'ensemble, positive dans chaque espèce) »
- **Problème** : `bill_depth_mm` se dit « épaisseur du bec » dans les ch. 0B, 1, 2, 3, 7, 8 et 10 et dans la data card `data/cards/penguins.md`. « Profondeur » ne reste qu'ici et en 0A :
  - `tools/chapters/build_ch00a.py:831`, `:864`, `:935`, `:1141` et `:2832`, donc dans les deux notebooks de 0A ;
  - `chapitres/ch00a_python/05_solutions.md:546`.
- **Correction** : « épaisseur du bec » partout, puis reconstruire 0A (`build_ch00a.py`, `run_all_notebooks.py … --inplace`, `build_answers.py`).

### 8. MINEUR · `tools/chapters/build_ch02.py:1432`, d'où `chapitres/ch02_stats/05_solutions.ipynb`, cellule 94 (💡 de 2.25) · correction A1 n° 14 non propagée au notebook
- **Extrait** : « C'est un visage du **fléau de la dimension** : « le plus proche » perd son sens. »
- **Problème** : la fiche (L285), `05_solutions.md` et `erreurs_frequentes.md` disent désormais « malédiction de la dimension », le terme principal (§22). Le script du notebook n'a pas été modifié et le notebook n'a pas été reconstruit.
  - Même renvoi vieilli au ch. 7 : `chapitres/ch07_classification/01_fiche.md:327`, « la malédiction de la dimension (le « fléau » du ch. 2) ».
- **Correction** :
  - dans `build_ch02.py` : « C'est un visage de la **malédiction de la dimension** : … », puis reconstruction, `run_all_notebooks.py chapitres/ch02_stats/05_solutions.ipynb --inplace` et `build_answers.py` ;
  - au ch. 7 : « la malédiction de la dimension (ch. 2, §2.7 et 2.25) ».

### 9. MINEUR · `chapitres/ch01_introduction/01_fiche.md:347` et `:44` · « fuite de données » jamais nommée dans le chapitre censé l'introduire
- **Extraits** :
  - objectifs, L44 : « **diagnostiquer** une évaluation faussée (mémorisation, jeu de test vu pendant l'entraînement, fuite du label) » ;
  - tableau des pièges, L347 : « | une feature qui contient la réponse | une colonne dérivée du label, ou une information connue seulement après coup | … » ;
  - `chapitres/ch01_introduction/flashcards.csv:13` : « (<b>fuite du label</b>, <i>data leakage</i>) ».
- **Problème** : les autres fichiers renvoient au ch. 1 pour ce terme, mais la fiche du ch. 1 ne l'emploie pas.
  - Le glossaire attribue « fuite de données (*data leakage*) » au ch. 1 (`annexes/glossaire.md:183`, colonne Ch. = 1).
  - Les ch. 2 et 3 y renvoient : `ch02_stats/01_fiche.md:217` « c'est une fuite de données (ch. 1) » ; `ch03_probabilites/01_fiche.md:460` « le score « trop beau » d'une fuite de données ».
  - Or le corps de la fiche du ch. 1 n'emploie ni « fuite de données » ni *data leakage*. La seule occurrence de « fuite » est dans les objectifs (L44), sous la forme « fuite du label ».
  - La flashcard traduit en outre « fuite du label » par *data leakage*, qui est le terme général (ch. 8, glossaire).
- **Correction** :
  - L347 : « | une feature qui contient la réponse (une **fuite de données**) | … » ;
  - à la fin du ⚠️ de la L150 : « Une information sur la réponse qui se glisse ainsi dans l'entraînement s'appelle une **fuite de données** (*data leakage*). » ;
  - flashcard : « (<b>fuite de données</b>, <i>data leakage</i> : ici, le label caché dans une feature) » ;
  - L44 : « fuite de données ».

### 10. MINEUR · `annexes/cheatsheets/sklearn.md:3` · chapitres annoncés faux
- **Extrait** : « Aide-mémoire rempli au fil des chapitres (8, 12-15). »
- **Problème** : l'aide-mémoire contient déjà des lignes des ch. 1 (`fit`, `predict`, `DecisionTreeClassifier`, `MLPClassifier`), 3 (section des mesures, L86–L105), 6 (`log_loss`), 7, 8, 9 et 10.
- **Correction** : « Aide-mémoire rempli au fil des chapitres (1, 3, 6 à 10, puis 12 à 15). »

### 11. SUGGESTION · `chapitres/ch02_stats/01_fiche.md`, §2.6 (L255–L275) · « erreur type » et vitesse en $1/\sqrt{n}$ absentes de la fiche
- **Constat** : plusieurs fichiers attribuent ces notions au ch. 2 :
  - `annexes/glossaire.md:231` : « erreur type | standard error | … | 2 » ;
  - `ch03_probabilites/01_fiche.md:436` : « l'écart typique d'une proportion diminue comme $1/\sqrt{n}$, ch. 2 » ;
  - `ch04_bayes/01_fiche.md:266` : « Ch. 2 : … vitesse en $1/\sqrt{n}$ » ;
  - `ch08_train_test/01_fiche.md:9` et `:258` : « Ch. 2 : … l'erreur type et le bootstrap (2.22 à 2.24) ».
- Dans le ch. 2 lui-même, ces notions ne figurent pas dans la fiche : l'apprenant qui révise par la fiche ne les trouve pas.
  - Le terme « erreur type » n'apparaît que dans le notebook : 2.22 e (`build_ch02.py:1061`) et son 💡 (`:1095-1098`).
  - $1/\sqrt{n}$ n'apparaît que dans la solution de 2.10 (`05_solutions.md:139`) et, sous la forme $s/\sqrt{n}$, dans le même 💡 de 2.22.
- **Correction** : ajouter après l'étape 4 du §2.6 (L263) : « L'écart-type de la distribution bootstrap estime l'**erreur type** (*standard error*) de la statistique : de combien elle varie d'un échantillon à l'autre. Pour une moyenne, elle vaut environ $\sigma / \sqrt{n}$ : il faut quatre fois plus de données pour diviser l'incertitude par deux (tu le vérifies en 2.22 e). »

### 12. SUGGESTION · panorama 2026 sans les modèles « de raisonnement »
- **Extraits** :
  - `chapitres/ch01_introduction/01_fiche.md:303` : « …sont **pré-entraînés** à prédire le **token** suivant … puis **alignés** sur des préférences humaines » ;
  - `05_solutions.md:173` (1.E4) : « Ensuite, on l'affine sur des exemples de dialogues, puis on l'aligne sur les préférences humaines, par apprentissage par renforcement ou des méthodes proches comme DPO » ;
  - `flashcards.csv:21` : « puis <b>aligné</b> sur des préférences humaines (RLHF, DPO) » ;
  - `annexes/glossaire.md:186-187`.
- **Problème** : ce panorama « 2026 » et la réponse modèle 💼 décrivent l'état de 2023.
  - Depuis OpenAI o1 (septembre 2024) et DeepSeek-R1 (*Nature*, 2025), une étape d'entraînement supplémentaire est courante : le renforcement sur des tâches dont la réponse se vérifie automatiquement (mathématiques, code).
  - Le ch. 11 le dit déjà (`ch11_raisonnement/01_fiche.md:244`).
- **Correction** :
  - à la puce LLM de la fiche, ajouter : « ; depuis 2024, des modèles « de raisonnement » sont en plus entraînés par **renforcement** sur des tâches dont la réponse se vérifie automatiquement (mathématiques, code), ch. 11 » ;
  - dans 1.E4, après DPO : « Les modèles les plus récents apprennent aussi, par renforcement, à raisonner par étapes sur des problèmes dont la réponse se vérifie, comme les mathématiques ou le code. »

### 13. SUGGESTION · `chapitres/ch01_introduction/01_fiche.md:334-339` (🕰️ reconnaissance faciale) · mentionner l'omnibus de 2026
- **Constat** : l'encadré est juste au 2026-10-04, et « Le texte évolue » y figure déjà.
  - Les interdictions de l'article 5 s'appliquent depuis le 2 février 2025.
  - Les amendes de Clearview sont confirmées : 20 M€ par la CNIL (2022), 30,5 M€ par l'autorité néerlandaise (2024).
  - La solution de 1.7 (`05_solutions.md:139`) est juste aussi.
- **Correction** : ajouter à la puce AI Act : « En 2026, un règlement « omnibus » a repoussé au 2 décembre 2027 les obligations des systèmes à haut risque (dont l'identification biométrique à distance hors cas interdits), sans toucher aux interdictions. » Ajouter aussi la source White & Case.

### 14. SUGGESTION · `chapitres/ch03_probabilites/05_solutions.md:183` (3.E5, relance) · l'AUC après recalibration
- **Extrait** : « (une recalibration croissante, comme Platt, garde l'ordre des scores, donc l'AUC ; la régression isotonique peut créer des ex-æquo et la modifier un peu) »
- **Problème** : c'est vrai pour **une** sigmoïde appliquée aux scores d'un modèle figé, mais pas pour `CalibratedClassifierCV` par défaut.
  - Son défaut est `ensemble="auto"`, donc `True` dans scikit-learn 1.6.1 : il réentraîne le modèle sur chaque fold et moyenne les probabilités de 5 couples (modèle, calibrateur).
  - Le résultat n'est plus une fonction croissante des scores d'origine, et l'AUC peut bouger.
  - Essai sur `make_classification` avec `GaussianNB` : AUC 0,763522 pour le modèle ; 0,763514 avec `CalibratedClassifierCV(GaussianNB(), method="sigmoid")` ; 0,763522 avec `CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")`.
- **Correction** : ajouter « (avec `CalibratedClassifierCV` par défaut, qui moyenne plusieurs modèles réentraînés, l'AUC peut légèrement changer ; pour seulement recalibrer un modèle déjà entraîné : `FrozenEstimator`) ».

### 15. SUGGESTION · `chapitres/ch02_stats/05_solutions.md:278` et cellule 39 du notebook (`build_ch02.py:567`) · Ex 2.16
- **Extrait** : « le creux entre elles (vers 205 mm) tombe au milieu d'un intervalle » (le notebook dit « autour de 205 mm »).
- **Problème** : le creux ne tombe pas au milieu de l'intervalle.
  - Avec 4 intervalles, les bords sont 172 ; 186,75 ; 201,5 ; 216,25 ; 231 (`np.histogram(flipper, bins=4)`, effectifs 48, 152, 83, 59).
  - Le creux (≈ 203–206 mm) tombe dans le 3ᵉ intervalle, près de son début, et non en son milieu (208,9 mm).
- **Correction** : « tombe à l'intérieur d'un intervalle (de 201,5 à 216,25 mm) ».

### 16. SUGGESTION · `chapitres/ch01_introduction/05_solutions.md:327` (Ex 1.24)
- **Extrait** : « le faux texte ressemble à de l'anglais de loin (« Tha Euthinto », « An than asiomysirk? »), sans un vrai mot sur deux. »
- **Problème** : « sans un vrai mot sur deux » est ambigu : on peut lire « pas un mot sur deux » ou « moins d'un mot sur deux ».
- **Correction** : « …, mais la plupart des « mots » n'existent pas. »

### 17. SUGGESTION · `annexes/formulaire.md:187-188` · mise en forme
- **Extrait** : la dernière ligne du tableau du ch. 2 (L187 « | matrices | case $(j, k)$ : … ») est immédiatement suivie du titre L188 « ### Ch. 3 · Probabilités et mesure de la qualité ».
- **Problème** : il manque la ligne vide que les autres sections ont, et certains rendus Markdown collent le titre au tableau.
- **Correction** : insérer une ligne vide.

## Affirmations datées vérifiées sur le web (13)

| # | Affirmation (fichier:ligne) | Verdict au 2026-10-04 |
|---|---|---|
| 1 | AI Index 2026 : « l'industrie a produit plus de 90 % des modèles d'IA marquants de 2025 » (`ch01/01_fiche.md:307`) | juste (93 modèles de l'industrie contre 2 de l'université ; « over 90% ») |
| 2 | AI Act, interdictions de l'article 5 depuis le 2 février 2025 (`ch01/01_fiche.md:336`, `05_solutions.md:139`, `04_indices.md:387`) | juste ; l'omnibus de 2026 ne les retarde pas |
| 3 | AI Act, annexe III : obligations « y sont obligatoires » (`ch02/05_solutions.md:147`) | **dépassé** (constat 1) |
| 4 | Clearview AI : 20 M€ (CNIL, 2022), 30,5 M€ (autorité néerlandaise, 2024) (`ch01/01_fiche.md:337`) | juste |
| 5 | PC Copilot+ : NPU de 40 TOPS et plus (`ch01/01_fiche.md:264`) | juste |
| 6 | Keras 3 multi-backend : JAX, TensorFlow, PyTorch, et OpenVINO pour l'inférence (`ch01/01_fiche.md:275`) | juste |
| 7 | Les créateurs de Fashion-MNIST jugent MNIST trop facile et trop utilisé (`ch01/01_fiche.md:256`) | juste |
| 8 | MNIST « résolu » dès 2012 par un ensemble de réseaux convolutifs (Cireşan et coll.) (`ch01/01_fiche.md:255`) | juste (« near-human performance ») |
| 9 | `CalibratedClassifierCV(method="temperature")` depuis scikit-learn 1.8 (`ch03/01_fiche.md:440`) | juste |
| 10 | La régression isotonique demande au moins un millier d'exemples d'après scikit-learn (`ch03/01_fiche.md:440`) | juste (« >> 1000 samples ») |
| 11 | Guo et coll. 2017 (réseaux modernes mal calibrés) ; Minderer et coll. 2021 (architectures sans convolutions parmi les mieux calibrées) (`ch03/01_fiche.md:440`) | juste |
| 12 | McDermott et coll., NeurIPS 2024 : pas de supériorité générale de l'AUPRC en cas de déséquilibre (`ch03/01_fiche.md:422`) | juste |
| 13 | `scipy.stats.bootstrap` : 9 999 rééchantillons et BCa par défaut (`ch02/01_fiche.md:275`) | juste |

Le site original de MNIST (yann.lecun.com) est signalé comme peu fiable : ce n'est pas réfuté (incident pytorch/vision n° 8568).

## Contrôles sans constat

- **Calculs** : toutes les valeurs recalculées sont justes et identiques d'un fichier à l'autre (fiche, énoncés, indices, `05_solutions.md`, sorties de `05_solutions.ipynb`). Cela couvre les ✏️, ∂, 🧮 et quiz des trois chapitres, les exemples chiffrés des fiches et les valeurs annoncées dans les 💡.
- **Notebooks** : les blocs `>>>` des fiches se reproduisent à l'identique avec la pile figée. Les cellules Markdown des deux notebooks sont identiques, hors cellules 📝.
- **Figures** : les 19 figures (axes, unités, légendes) sont conformes au texte, à l'effectif de la calibration près (constat 4).
- **Cohérence entre chapitres** (grep sur 0A, 0B, ch. 4 à 11, CP1, CP2, MP1, MP2 et annexes) :
  - **ddof** : NumPy 0, pandas 1, identique partout.
  - **Prévalence** : $\frac{TP + FN}{n}$ ; la precision est un posterior dont le prior est la prévalence (ch. 4).
  - **Seuil** : « positif si score ≥ t » ; le ch. 7 écrit « p > t* » mais précise « à égalité, les deux choix coûtent autant ».
  - **Matrice de confusion** : `[[TN, FP], [FN, TP]]` partout.
  - **Intervalle de confiance** : une promesse sur la méthode (ch. 2, 4, 8 ; flashcards ; glossaire).
  - **AP** : en escalier.
  - **Fuite de données** : même définition au ch. 8, au glossaire et dans `erreurs_frequentes.md`.
  - Aucune contradiction de fond trouvée.
- **Indices** : hors les cas du constat 2, les indices 3 posent le calcul : papier des ch. 1 et 2, papier du ch. 3 sauf 3.1 et 3.7. Le défi 3.29 donne une méthode, ce qui correspond au « pseudo-code » du §12.

## Bilan

**Ce qui est solide**
- L'exactitude numérique : aucun calcul faux trouvé dans les trois chapitres, sorties de notebooks comprises.
- Des définitions homogènes avec les chapitres suivants : ddof, prévalence, seuil, matrice de confusion, intervalle de confiance, AP.
- Des encadrés 🕰️ précis et sourcés : 12 affirmations datées sur 13 restent justes.
- Des figures fidèles au texte.

**Les 3 risques principaux**
1. **Les faits réglementaires datés** : l'omnibus de juillet 2026 rend déjà faux un présent (constat 1). Les passages juridiques (AI Act, RGPD) vieillissent plus vite que le reste et demandent une relecture à chaque échéance du calendrier de l'AI Act (2 décembre 2027, 2 août 2028).
2. **Des corrections faites dans le Markdown sans passer par les scripts `build_*.py`** : le notebook du ch. 2 garde le « fléau » corrigé ailleurs (constat 8). Un test qui chercherait les termes interdits dans les notebooks et les messages des scripts éviterait ces restes.
3. **Des indices de niveau 3 qui livrent la réponse**, dont deux exercices vérifiés en partie 0 (constat 2), et **de petites dérives de vocabulaire** (« nombre » / « quantité », « profondeur » / « épaisseur », « fuite du label » / « fuite de données ») qui se propagent aux flashcards, au glossaire et aux figures (constats 6, 7 et 9).

## Sources

- [White & Case, « EU AI Omnibus enters into force, amending the AI Act » (2026)](https://www.whitecase.com/insight-alert/eu-ai-omnibus-enters-force-amending-ai-act)
- [Lewis Silkin, « The Digital Omnibus on AI enters into force today » (27 juillet 2026)](https://www.lewissilkin.com/insights/2026/07/27/the-digital-omnibus-on-ai-enters-into-force-today-102nedo)
- [Cuatrecasas, « Council of the EU approves Digital Omnibus on AI »](https://www.cuatrecasas.com/en/latam/intellectual-property/art/council-eu-approves-digital-omnibus-ai)
- [National Law Review, « EU Digital Omnibus on AI enters into force »](https://natlawreview.com/article/eu-digital-omnibus-ai-enters-force)
- [Stanford HAI, AI Index 2026, chapitre 1 (PDF)](https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_1_research_development.pdf)
- [JD Supra, « Stanford's 2026 AI Index highlights »](https://www.jdsupra.com/legalnews/stanford-s-2026-ai-index-highlights-7540386/)
- [Autoriteit Persoonsgegevens, amende Clearview (2024)](https://www.autoriteitpersoonsgegevens.nl/en/current/dutch-dpa-imposes-a-fine-on-clearview-because-of-illegal-data-collection-for-facial-recognition)
- [CNIL, délibération SAN-2022-019 (Clearview AI), Légifrance](https://www.legifrance.gouv.fr/cnil/id/CNILTEXT000046444859)
- [Microsoft Learn, « NPU devices » (Copilot+ PC, 40+ TOPS)](https://learn.microsoft.com/en-us/windows/ai/npu-devices/)
- [scikit-learn 1.8, release highlights (temperature scaling)](https://scikit-learn.org/1.8/auto_examples/release_highlights/plot_release_highlights_1_8_0.html)
- [scikit-learn 1.6, guide de la calibration](https://scikit-learn.org/1.6/modules/calibration.html)
- [SciPy, `scipy.stats.bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [keras.io, « Introducing Keras 3.0 »](https://keras.io/keras_3/)
- [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- [D. Cireşan, U. Meier, J. Schmidhuber, « Multi-column Deep Neural Networks for Image Classification », arXiv 1202.2745](https://arxiv.org/abs/1202.2745)
- [M. McDermott et coll., NeurIPS 2024](https://neurips.cc/virtual/2024/poster/95133)
