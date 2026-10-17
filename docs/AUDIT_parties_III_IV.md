# Audit P5 des parties III et IV

*Chapitres 12 à 20, checkpoints III et IV, mini-projets MP3 et MP4. Prompt P5 (`docs/METHODE.md` §3), session 43.*

> **Statut : 🛠️ en cours (2026-10-17).** Onze relecteurs indépendants lancés en lecture seule sur le commit `5a17b91` ; leurs rapports bruts iront dans `docs/audit_P5_parties_III_IV/`, le tri et les propositions dans ce fichier.

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

## Annexe A — consignes des relecteurs

Les consignes communes reprennent celles de l'audit des parties 0 à II (`docs/AUDIT_parties_I_II.md`, annexe A.1), avec trois changements : le périmètre (ch. 12 à 20, CP3, CP4, MP3, MP4, modules `mylearn` et tests de ces chapitres ; les parties 0 à II ne sont relues que pour la cohérence) ; l'accès au texte du livre ; un rapport écrit au fur et à mesure, pour qu'une interruption ne perde rien.
