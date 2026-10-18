# Audit P5 des parties III et IV

*Chapitres 12 à 20, checkpoints III et IV, mini-projets MP3 et MP4. Prompt P5 (`docs/METHODE.md` §3), session 43.*

> **Statut : ✅ constats mineurs appliqués (2026-10-18) ; ⏸️ propositions R1 à R11 en attente de ta décision (§3).** Onze relecteurs indépendants en lecture seule sur le commit `5a17b91` (rapports bruts dans `docs/audit_P5_parties_III_IV/`, A01 à A11), puis sept correcteurs par zone (F1 à F7) ; tri au §4, vérification finale au §5.

## 1. Méthode

Onze relecteurs indépendants (sous-agents), en lecture seule, chacun sur une dimension de l'audit, avec les consignes communes de l'audit des parties 0 à II (`docs/AUDIT_parties_I_II.md`, annexe A), adaptées au périmètre (annexe A ci-dessous). Nouveauté : les relecteurs ont le texte du livre (volumes 1 et 2, usage interne de l'audit, jamais versionné) pour juger les encadrés « Le livre : » et « Le livre, corrigé », les renvois et la règle du §3.

| # | Relecteur | Dimension |
|---|---|---|
| A1 | termes | terminologie (§5, §22), titres, tutoiement, notations (§6), encadrés (§14), en-têtes (§11), typographie, d'un chapitre à l'autre |
| A2 | liens | renvois entre chapitres et leur sens, renvois vers l'avant, rappels 🔁, liens relatifs et externes, notions utilisées avant d'être enseignées (§15.2), remédiations, `requires` |
| A3 | structure | conformité aux §11, §12, §17 et §19 par script ; indices de niveau 3 ; échafaudage décroissant de la partie III |
| A4 | exactitude | ch. 12 et 13 |
| A5 | exactitude | ch. 14 et 15 |
| A6 | exactitude | ch. 16 et 17 |
| A7 | exactitude | ch. 18 et 19 |
| A8 | exactitude | ch. 20 et la cheatsheet PyTorch |
| A9 | checkpoints | CP3, CP4, MP3, MP4 : cohérence entre eux, avec CP1, CP2, MP1, MP2 et avec les chapitres ; corrigés recalculés ; kits essayés |
| A10 | code | API `mylearn` des parties III et IV, stubs ↔ référence ↔ tests ↔ énoncés, mutations, `wb`, cellules de vérification, « Run all » |
| A11 | parcours | expérience de l'apprenant de bout en bout (copie du dépôt), consignes, progression, parcours, suivi, Colab |

**Sévérités** (comme en session 24). MAJEUR : erreur de fond qui trompe ou bloque l'apprenant, ou correction qui changerait le contrat (énoncé ou réponse d'un exercice publié, ID, titre, type, durée, stub, règle de la bible, convention transversale) : **proposée, à valider par toi avant d'être appliquée**. MINEUR : défaut local sans effet sur le contrat : appliqué directement après vérification. SUGGESTION : facultatif.

## 2. Résultats d'ensemble

Les onze rapports bruts sont dans `docs/audit_P5_parties_III_IV/` (les citations du livre en anglais y sont retirées, §3). Ils comptent 200 constats : 7 majeurs, 149 mineurs et 44 suggestions.

| Relecteur | Majeurs | Mineurs | Suggestions | En une phrase |
|---|---|---|---|---|
| A1 termes | 0 | 31 | 2 | les termes du §5 tiennent ; les formats fixés par P5 (fil rouge, « Tout exécuter », icônes des 04, rubriques 💼/🗣️, préfixe des rappels) dérivent à partir du ch. 13 ; quelques termes à deux formes |
| A2 liens | 0 | 17 | 2 | 13 749 ID cités, aucun inconnu ; quelques notions attribuées au mauvais chapitre, des mots de maths hors programme dans deux corrigés, des liens `stable` |
| A3 structure | 1 | 14 | 6 | les 437 exercices concordent avec le contrat sur tous les supports ; l'échafaudage ne décroît pas ; quelques indices 3 trop complets |
| A4 ch. 12-13 | 0 | 7 | 5 | aucune erreur de calcul sur plus de 250 nombres ; deux définitions qui divergent entre chapitres (un-contre-un, lissage de Laplace) |
| A5 ch. 14-15 | 0 | 9 | 3 | aucune réponse fausse ; trois résumés du livre trop rapides, des durées commentées à la main |
| A6 ch. 16-17 | 1 | 8 | 4 | aucune réponse fausse, preuves justes ; le 🔮 16.19 a) dépend de l'implémentation |
| A7 ch. 18-19 | 0 | 8 | 3 | environ 230 nombres recalculés, tous justes à quelques détails près |
| A8 ch. 20 | 1 | 17 | 2 | le fond est juste ; des affirmations PyTorch trop absolues (labels one-hot de `CrossEntropyLoss`) et des chiffres d'annexes en retard |
| A9 checkpoints | 1 | 9 | 6 | examens justes et barèmes exacts ; le coffre de MP3 (et MP2) s'ouvre sans commit du test |
| A10 code | 1 | 16 | 8 | 1 775 tests de la référence verts, 91 mutants sur 94 tués ; les outils de notebook recopiés par chapitre divergent |
| A11 parcours | 2 | 13 | 3 | aucun exercice bloqué dans un parcours ; le parcours maths n'a pas de PyTorch au ch. 20 ; trois durées de la partie III sous-estimées |

## 3. Corrections majeures proposées (à valider)

Ces onze points changent le contrat (énoncés, réponses, titres, ★, durées, parcours), une convention transversale ou la structure des outils. Ils sont exclus des corrections directes (section 4) et attendent ta décision. Pour chacun, la recommandation est en tête.

**R1 — 🔮 16.19 a) : une symétrie « exacte » qui dépend du code de l'apprenant.** *Recommandé : A.* (A6 1)
L'expérience appelle ta `mlp_forward`. Avec `x @ W` ou `einsum`, les quatre neurones restent jumeaux à l'écart 0 près ; avec une somme par diffusion ou une boucle (deux écritures justes), l'écart vaut 2·10⁻⁶ après 200 pas, et ta prédiction `True` reçoit pourtant ✅ pendant que le corrigé affirme « écart 0 ».
- **A** : énoncé « identiques à 10⁻⁵ près (`spread_19 < 1e-5`) », réponse `True` inchangée ; solution et erreur fréquente expliquent que l'arrondi d'une somme dépend de l'ordre des additions, et que le point symétrique est instable.
- **B** : laisser tel quel.

**R2 — L'échafaudage ne décroît pas (§12 point 8, §22 Q10).** *Recommandé : C.* (A3 1)
Les énoncés 🔨/📦 des parties III et IV font 198 à 290 mots en moyenne, comme aux ch. 5 à 11. Les 🔨 ont un bloc « Contrat », mais leurs contrats décrivent l'ordre des opérations (les tests comparent pas à pas à un oracle) ; quelques conseils de méthode restent dans l'énoncé (19.16 : `np.asarray` pour un tableau 0-D, `g.copy()`) ; les 📦 de la partie IV donnent la recette (20.18, 20.23, 20.28), et en 20.23 l'énoncé et la docstring en disent plus que l'indice 3.
- **A** : écrire au §22 que le contrat d'un 🔨 comparé pas à pas inclut l'ordre des opérations ; déplacer les conseils de mise en œuvre de 19.16 (et des énoncés semblables) vers les indices 1 et 2.
- **B** : A, plus un bloc « Contrat » au lieu de la recette pour les 📦 de la partie IV (20.18, 20.23, 20.28), les appels passant dans l'indice 2.
- **C** : A pour les 🔨 ; garder la recette des 📦 du ch. 20, premier contact avec PyTorch (l'apprenant y apprend justement l'enchaînement), mais l'écrire comme exception au §22 et appliquer B à partir de la partie V ; un contrôle indicatif de la longueur des énoncés par partie dans `syllabus.py stats`.

**R3 — Les outils des cellules de vérification, recopiés par chapitre, divergent.** *Recommandé : B.* (A10 20)
`checked` existe en 4 versions, `same` en 3, `run_*_tests` en 6 : une amélioration faite dans un chapitre manque aux suivants (l'indice « NoneType » des ch. 13 et 14 a disparu aux ch. 16 à 20). Les améliorations existantes sont déjà reportées dans les chapitres (section 4, option a).
- **A** : en rester là (option a seule).
- **B** : un module `wb.nbtools`, testé dans `tests/infra/`, pour `checked`, `written`, `same`, `verdict`, `returned`, `run_pytest_group`, utilisé à partir du ch. 21 ; les ch. 12 à 20 y passent quand on les reconstruit pour une autre raison.
- **C** : B, et migrer tout de suite les ch. 12 à 20 (neuf reconstructions, notebooks réexécutés).

**R4 — Titres publiés non conformes au vocabulaire.** *Recommandé : A.* (A1 8, 12, 24, 33 ; A3 9)
« LayerNorm » dans les titres de 20.7, 20.30 et 20.E2 (le terme retenu de la fiche est « layer norm ») ; « coût du backward » dans 18.Q11 (« passe arrière ») ; « & » dans les titres des 📄 8.10, 16.10, 17.10, 18.11, 19.12, 20.12 (« et », comme les contrats des ch. 21 et suivants) ; titres raccourcis des questions dans les corrigés et copies de CP3 et CP4.
- **A** : changer les trois premiers groupes de titres (ID inchangés, sur tous les supports, contrat compris) ; garder les titres courts des checkpoints et l'écrire au §22.
- **B** : ne rien changer.

**R5 — Termes à deux formes.** *Recommandé : les choix ci-dessous.* (A1 8 à 11, 14, 15, 28)
- « **Naïve Bayes** » (≈ 220 occurrences et six titres des parties III et IV) plutôt que « Naive Bayes » (≈ 20 au ch. 4 et en MP1) ; une seule ligne au glossaire.
- « **couche dense** » (≈ 125 occurrences) plutôt que « couche pleine » (glossaire, ch. 1).
- « **variable de substitution** (*proxy*) » plutôt que « substitut » et « indicateur indirect ».
- « **vérification du gradient** (*gradient checking*) », la forme du ch. 18, au glossaire.
- « **baseline** » (le terme du glossaire), glose « *modèle de référence* » au ch. 13.
- Gloses à la première occurrence : « **split** (*découpe* d'un nœud) » au ch. 13 et « avant le découpage » dans le titre de 12.31 ; « nuage gaussien (*blob*) » ; « scaler » glosé au ch. 12.
- « **layer norm** » dans la prose (ch. 16, 18, 20), `nn.LayerNorm` dans le code.

**R6 — Étoiles de CP4.3.** *Recommandé : ★★.* (A9 1)
CP4.3 combine momentum, Adam et weight decay et demande une interprétation (h) : deux notions et une interprétation, comme CP3.5 (★★) ; il a ★.

**R7 — Le parcours maths n'a aucun exercice PyTorch du ch. 20.** *Recommandé : A.* (A11 7)
CP4.7, CP4.13 et le MP4, communs aux quatre parcours, puis 21.22 (parcours maths) supposent la pratique de PyTorch ; le parcours maths ne fait au ch. 20 que les exercices NumPy.
- **A** : ajouter 20.13, 20.15, 20.18, 20.19 et 20.23 au parcours maths (+ 130 min ; parcours maths ≈ 552 h).
- **B** : déclarer des prérequis d'exercice (`"20.23"`, `"20.24"` pour CP4.7 ; `"20.13"`, `"20.18"` pour CP4.13 ; `"20.23"` pour 21.22), comptés en « corrigés à lire », et une phrase dans le cahier des charges de MP4.

**R8 — Trois durées de la partie III sous-estimées pour un débutant.** *Recommandé : appliquer.* (A11 8)
13.24 (arbre récursif, ≈ 110 lignes, 49 tests) : 100 → 150 min ; 13.32 (régression logistique, 48 lignes) : 50 → 70 min ; 14.24 (AdaBoost SAMME, 57 lignes) : 50 → 75 min. Les autres se recaleront avec tes temps réels (journal, P7).

**R9 — Énoncés à préciser, réponses inchangées.** *Recommandé : appliquer.* (A2, A4, A7, A9)
- 18.R2 d : le solveur `lbfgs` n'est enseigné nulle part ; en faire une lecture de la documentation de `MLPClassifier` 1.6 (réponse « Vrai » inchangée).
- 13.18 : expliquer `GridSearchCV` et `best_params_` dans l'énoncé (le prérequis « ch. 8 » est faux).
- 13.14 : « pourquoi conseille-t-on souvent un $k$ impair » (le conseil n'est pas de scikit-learn).
- 📄 16.10, question 4 : gloser « jacobienne » (définie au ch. 17).
- 18.16 : poser le cas d'un facteur $k$ quelconque, dont 🐛 18.17 a besoin.
- 19.Q12 e : « lequel l'emporte ? » (le livre donne Nesterov nettement devant Adam) ; réponse C inchangée.
- CP3.6 c) : une même phrase neutre au sujet et au notebook (le notebook retire un candidat de a)).
- CP4.9 a) : préciser que, à la différence de `clip_grad_norm` (ch. 19) et de `clip_grad_norm_`, la fonction rend de nouveaux tableaux.
- 20.15 : dire que la comparaison à 18.4 passe par son corrigé si ton parcours ne l'a pas fait.

**R10 — Conventions de la référence `mylearn` sans test.** *Recommandé : A.* (A10 3, 5, 9)
- **A** : le kNN, Naïve Bayes et la régression logistique refusent ±inf dans `X` et NaN dans `y`, comme `preprocessing`, `tree`, `ensemble` et scikit-learn (référence seulement, docstrings figées) ; écrire au §22 que `softmax_cross_entropy` ne vérifie pas des cibles de probabilités et qu'un `init_mlp` qui recrée son générateur n'est pas testable sans oracle.
- **B** : seulement les lignes du §22.

**R11 — Les diagnostics des 🐛 « corrigés sur place » s'affichent au premier « Tout exécuter ».** *Recommandé : B.* (A11 2)
Pour 14.23, 15.12, 16.20, 17.17 et 20.24, le code fautif est fourni et l'apprenant le corrige sur place : un « Tout exécuter » à vide montre ses diagnostics avant tout essai ; en 14.23, un message nomme le correctif (`sample_weight`) ; 14.23 a/b et 20.24 (points 3 à 5) tournent même quand le point 1 échoue, contre la règle de 10.22 (§22).
- **A** : une variable `FIXED_N = False` que l'apprenant passe à `True` quand il a corrigé ; jusque-là, ⏳.
- **B** : garder les diagnostics (le symptôme du code fourni est le point de départ du 🐛, comme en 12.12), mais enchaîner les points comme en 10.22 et retirer de tout message le nom du correctif.

## 4. Tri et application des constats mineurs (2026-10-18)

Sept correcteurs (sous-agents), un par zone du dépôt, ont lu les onze rapports, vérifié chaque constat de leur zone dans les fichiers (calculs refaits, code rejoué, livre relu), puis appliqué ce qui ne change pas le contrat ; les constats R1 à R11 leur étaient interdits. Chaque correcteur a reconstruit ses notebooks par leur script, réexécuté les solutions et fait un « Tout exécuter » à vide dans une copie. Leurs tableaux de décision (constat → appliqué, non retenu avec la raison, reporté Rn, proposé) sont dans `docs/audit_P5_parties_III_IV/F1_ch12_13.md` à `F7_transversal.md`.

| Zone | Correcteur | L'essentiel |
|---|---|---|
| ch. 12 et 13 | F1 | formats P5 (fil rouge « synthétique », « Tout exécuter » et phrase Colab, préfixe des rappels), liens figés sur scikit-learn 1.6, indices qui donnaient une valeur, définitions alignées sur les ch. 6 et 7 (un-contre-un, lissage de Laplace), paraphrase du livre réécrite ; les cinq 🔨 du ch. 12 passent au modèle `checked`/`written` du ch. 13 ; 12.20 d enregistré à 6 décimales (seul changement de réponse, par l'arrondi : la matrice est la même) |
| ch. 14 et 15 | F2 | trois résumés « Le livre : » corrigés après relecture du livre, durées et proportions des solutions recalées sur les sorties, une fluctuation de la figure des importances expliquée, paraphrases réécrites |
| ch. 16 et 17 | F3 | horloge de parquet et initialisation ±0,05 rendues justes envers le livre, variante de 17.13 avec `tanh(20)`, ∂ 17.7 sans « noyau » ni « rang », `coefs_` rattaché à 📦 1.23, rubriques 💼 et 🗣️, indice 3 de 17.12 réduit au squelette, messages sans PyTorch |
| ch. 18 et 19 | F4 | « cent fois » en 19.22, demi-vie de 19.8, Nesterov nettement devant Adam (fiche et solution), μ = 1 accepté par `torch.optim.SGD`, flashcard du signe du gradient, comptes de tests, indices 3 de 18.26 et 19.16 à 19.18 réduits, titre de 19.6 aligné sur le contrat, préfixe des rappels du ch. 19, liens figés sur PyTorch 2.11 |
| ch. 20 | F5 | `CrossEntropyLoss` et les labels one-hot (constat majeur A8 1, une erreur de fait corrigée dans la fiche, le tableau des pièges et les solutions, sans toucher au contrat), variante de 20.23 rejouée, figure du dropout redessinée, helper `pair` pour 20.16 et 20.22, le ch. 20 sans PyTorch va au bout avec des ⏳, rubriques 💼 et 🗣️, notations ρ et $H \times W$ |
| checkpoints et mini-projets | F6 | le coffre de MP3 et de MP2 ne s'ouvre plus sans commit de `test_indices.npy` ; le kit vierge de MP4 va au bout avec des ⏳ ; corrigé de CP4 avec son barème récapitulatif et ses erreurs fréquentes ; une erreur fréquente fausse de CP3.3 ; RMSprop en α et λ = 2α/n dans la synthèse de CP4 ; messages de MP4 en français ; `results.json` de MP3 et MP4 complétés |
| annexes, code, tests, suivi | F7 | référence `mylearn` ramenée à ses conventions (dropout et `rng` entier, `RuntimeError` de Naïve Bayes, messages des ensembles) ; 7 tests et messages de tests de plus ; `wb.attempt` sur un `AttributeError` de `None` ; une `SyntaxError` de l'apprenant en deux lignes de pytest au lieu de 5 400 ; glossaire, formulaire, erreurs fréquentes, cheatsheets et data cards mis à jour ; test de terminologie étendu (data cards, model cards, `minifw`) ; METHODE et README sur le temps de lecture des fiches ; lien du code du volume 1 (dépôt d'origine retiré, copie publique) |

**Réponses enregistrées.** Aucune bonne réponse ne change de valeur. 12.20 d change de nombre de décimales (4 → 6) ; 20.Q10 d, 20.13 b et CP3.8 e changent seulement par leurs messages d'erreur classique.

**Laissés pour plus tard (notés).** La place du 🧮 dans `02_exercices.md` (papier ou réflexion selon le chapitre, A3 7) ; `exercise_table` avec la colonne 🚀 (A3 15) ; un délai dans les `run_*_tests` des chapitres qui n'en ont pas encore (A10 25 : appliqué aux ch. 14 à 19) ; deux tests d'infrastructure des kits (coffre de MP2 et MP3, « Tout exécuter » du kit MP4 avec les squelettes) ; 18.19 d, proche d'une limite d'arrondi mais dont les deux arrondis sont déjà acceptés ; une piste « Que vaut J𝟏 ? » dans l'énoncé de ∂ 17.7 (elle faciliterait la question).

## 5. Vérification finale (2026-10-18)

- **Reconstruction** : 13 scripts (`build_mp4.py` une fois de plus après la vérification ci-dessous) (`build_ch12.py` à `build_ch20.py`, `build_cp3.py`, `build_mp2.py`, `build_mp3.py`, `build_mp4.py`) et la figure du dropout (`figures_ch20.py`).
- **Notebooks de solutions** : tous réexécutés après leur script. Les cinq qui avaient été mesurés pendant que les correcteurs chargeaient la machine l'ont été à nouveau sur une machine calme : ch. 14 en 167 s (1 019 s sous charge), ch. 12 en 72 s, ch. 17 en 38 s, MP3 en 129 s, MP2 en 97 s. Aucun n'est hors budget (§4) ; leurs sorties, comme celles des ch. 13, 15, 16, 18 à 20, de CP3, CP4 et MP4, n'ont ni ❌, ni ⏳, ni erreur. Dans `results.json` de MP3, seuls les temps changent.
- **Réponses** : `answers.json` régénéré (3 348 réponses). Aucune bonne réponse ne change de valeur : 12.20d passe de 4 à 6 décimales (même matrice), et 20.Q10d, 20.13b et CP3.8e ne changent que par leurs messages.
- **« Tout exécuter » à vide**, dans une copie du dépôt : les neuf notebooks d'exercices des ch. 12 à 20, les deux examens et les kits de MP3 et MP4, sans erreur, en 95 s au total ; les seuls ❌ sont ceux des 🐛 « corrigés sur place » (14.23, 15.12, 16.20, 17.17, 20.24 : R11). Le kit de MP4 lancé hors de `mon_travail/` (sans librairie) affichait « ❌ 0 test(s) échouent, ou moins de 10 tests » : son étape MP4.7 dit maintenant qu'aucun test n'a tourné et renvoie à l'étape MP4.4 (`build_mp4.py`, solution réexécutée en 58 s). Copié par `start_chapter.py CP4` avec les squelettes de la librairie, le kit va au bout avec des ⏳ et aucun ❌, comme l'examen.
- **Tests** : `--impl=ref` 3 411 réussis (4 ignorés) ; mode apprenant 300 réussis (3 115 ignorés, la librairie n'existe pas encore) ; `--impl=stubs` sur les tests de chapitre : 3 105 échecs attendus (le seul test qui passe, `sigmoid` du ch. 13, porte sur une fonction fournie). `syllabus.py check` (2 024 exercices, 0 problème), `build_answers.py --check`, `export_flashcards.py --check` (548 cartes) et `check_env.py` au vert ; `mon_travail/` intact.
- **Suivi** : `suivi/PROGRESS.md` (session 43, écarts, lignes 24 à 26 « À valider sur Colab », calibrage de MP3.8), `suivi/tableau_de_bord.md` (temps réel au bout de la ligne), huit lignes au §22 de la bible.

## Annexe A — consignes des relecteurs

Les consignes communes reprennent celles de l'audit des parties 0 à II (`docs/AUDIT_parties_I_II.md`, annexe A.1), avec trois changements : le périmètre (ch. 12 à 20, CP3, CP4, MP3, MP4, modules `mylearn` et tests de ces chapitres ; les parties 0 à II ne sont relues que pour la cohérence) ; l'accès au texte du livre ; un rapport écrit au fur et à mesure, pour qu'une interruption ne perde rien.
