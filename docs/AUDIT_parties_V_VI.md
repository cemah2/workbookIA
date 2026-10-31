# Audit P5 des parties V et VI

*Chapitres 21 à 29, checkpoints V et VI, mini-projets MP5 et MP6. Prompt P5 (`docs/METHODE.md` §3), session 62.*

> **Statut : ✅ terminé (2026-10-31) : constats mineurs appliqués (§4), propositions R1 à R9 validées et appliquées (§6).**
> - Onze relecteurs indépendants ont travaillé en lecture seule sur le commit `b961aa3`. Leurs rapports bruts sont dans `docs/audit_P5_parties_V_VI/`, de A01 à A11.
> - Sept correcteurs par zone (F1 à F7) ont vérifié puis appliqué les constats mineurs : voir le §4. La vérification finale est au §5.

## 1. Méthode

### Les relecteurs

Onze relecteurs indépendants (des sous-agents) ont travaillé en lecture seule, chacun sur une dimension de l'audit. Ils ont suivi les consignes communes des audits précédents, adaptées au périmètre (annexe A). Comme en session 43, ils avaient le texte des deux volumes du livre, pour un usage interne à l'audit, jamais versionné.

| # | Relecteur | Dimension |
|---|---|---|
| A1 | termes | terminologie (§5, §22), titres, tutoiement, notations (§6), encadrés (§14), en-têtes (§11), typographie, formats fixés par l'audit de la session 43 |
| A2 | liens | renvois entre chapitres et leur sens, renvois vers l'avant, rappels 🔁, liens relatifs et externes (figeage des versions), notions utilisées avant d'être enseignées (§15.2), remédiations, `requires`, pages du livre |
| A3 | structure | conformité aux §11, §12, §17 et §19, vérifiée par script ; indices de niveau 3 ; échafaudage décroissant (R2 C de la session 43) |
| A4 | exactitude | ch. 21 et 22 |
| A5 | exactitude | ch. 23, 24 et l'aide-mémoire PyTorch |
| A6 | exactitude | ch. 25 et 26 |
| A7 | exactitude | ch. 27 et 28 |
| A8 | exactitude | ch. 29, data cards, annexes des parties V et VI |
| A9 | checkpoints | CP5, CP6, MP5 et MP6 : cohérence avec CP3, CP4, MP3, MP4 et avec les chapitres ; corrigés recalculés ; kits essayés |
| A10 | code | modules `mylearn` des parties V et VI (stubs, référence, tests, énoncés), mutations, `wb` (`nbtools`, `envs`), cellules de vérification, « Tout exécuter » |
| A11 | parcours | expérience de l'apprenant de bout en bout (copie du dépôt), consignes, progression, parcours, suivi, Colab |

Les onze relecteurs tournaient en même temps sur une machine de 2 cœurs. Pour cette raison, aucune durée mesurée pendant l'audit n'est jugée.

### Les sévérités

Elles sont les mêmes qu'aux sessions 24 et 43.
- **MAJEUR** : une erreur de fond qui trompe ou bloque l'apprenant, ou une correction qui changerait le contrat (énoncé ou réponse d'un exercice publié, ID, titre, type, ★, durée, parcours, stub, règle de la bible, convention transversale). Une telle correction est **proposée, et attend ta validation avant d'être appliquée**.
- **MINEUR** : un défaut local sans effet sur le contrat. Il est appliqué directement, après vérification.
- **SUGGESTION** : facultative.

## 2. Résultats d'ensemble

Les onze rapports bruts sont dans `docs/audit_P5_parties_V_VI/`. Les citations du livre en anglais en sont retirées (§3). Ils comptent **199 constats** : 10 majeurs, 161 mineurs et 28 suggestions. Aucune réponse enregistrée n'est fausse.

| Relecteur | Majeurs | Mineurs | Suggestions | En une phrase |
|---|---|---|---|---|
| A1 termes | 1 | 23 | 4 | les en-têtes du §11, les titres de chapitres, l'ordre des rubriques et le tutoiement tiennent. Onze titres publiés emploient le terme écarté (« data augmentation », « replay buffer », « model card », « split »…). Les formats fixés en session 43 dérivent de nouveau à partir du ch. 25, faute de test (icônes des 04, préfixe des rappels, rubriques des 🗣️/⚖️/📄 et des 💼). `test_terminology.py` ne lit ni les ch. 28-29 ni les termes des parties V et VI. |
| A2 liens | 0 | 16 | 0 | plus de 2 500 renvois vérifiés, aucun ID, aucune section, aucune page du livre ni aucun lien relatif manquant. Huit sources PyTorch du ch. 23 (et d'autres) sont en `stable`. Quatre renvois ont un sens faux (§21.3.4, §23.10.2, « inégalité de Gibbs », RLHF au B4). Le minimax sert au ch. 26 avant le ch. 27. Deux remédiations manquent. |
| A3 structure | 2 | 19 | 3 | les 441 exercices concordent sur tous les supports ; composition, ⏩, budget et barèmes sont conformes. La hausse de longueur des énoncés 🔨 vient d'un « Contrat commun » recopié jusqu'à six fois, mais des 📦 donnent encore la méthode (24.24) ou la solution (24.27). CP6 s'écarte du §17 (📦 à l'examen, étoiles, pas de `notes`). |
| A4 ch. 21-22 | 0 | 11 | 3 | 252 valeurs recalculées, toutes justes. Le code des solutions passe les 449 tests. Il y a des temps de lecture non reportés au ch. 21, le « biais unique » de Keras étendu à tort au GRU, et des flashcards sans MathJax. |
| A5 ch. 23-24 | 0 | 9 | 3 | 255 réponses re-résolues à l'aveugle, toutes acceptées. Toutes les affirmations PyTorch sont justes, sauf « `view` refuse tout tenseur non contigu ». Quelques extraits de la fiche 24 ne tournent pas tels quels. |
| A6 ch. 25-26 | 0 | 12 | 2 | environ 155 réponses recalculées, toutes justes ; les erreurs du livre sur le RL sont signalées. La fiche 26 dit que le L-learning « marche » (le chapitre le réfute) et donne d'avance la réponse de 26.17 a. Il y a deux erreurs de code dans le formulaire (BCE, PSNR). |
| A7 ch. 27-28 | 0 | 11 | 2 | 112 valeurs recalculées, toutes justes. Deux corrections du livre sont injustes (l'origine du « scénario Helvetica », la normalisation du gradient du deep dream). La flashcard « minimax » définit le maximin. |
| A8 ch. 29 et annexes | 0 | 16 | 3 | toutes les réponses du ch. 29 et le formulaire des ch. 21 à 29 sont justes. Le dataset piégé a trois labels faux, entérinés par le nettoyage de référence. Les listes de chapitres et les tailles des data cards sont fausses. `ressources.md` et `metiers.md` ne couvrent pas les parties V et VI. |
| A9 checkpoints | 4 | 16 | 3 | toutes les réponses chiffrées sont justes ; la partie B, remplie par des solutions écrites autrement, donne 40/40 et 39/39. Mais la partie B des examens contient en clair la solution des questions de code. `model.pt` est ignoré par git. CP6 a un 📦 et des étoiles hors §17. Le kit MP5 range son modèle dans un dossier temporaire partagé. |
| A10 code | 0 | 18 | 4 | les stubs, la référence et les docstrings concordent ; 845 tests de la référence sont verts ; 50 mutants sur 53 sont tués (1 survivant réel). Le `rng` de `rl.py` refuse un entier, et `same` et `attempt` se trompent sur des tenseurs. |
| A11 parcours | 3 | 10 | 1 | `start_chapter.py`, les notebooks 21, 25 et 26 remplis par d'autres méthodes, les tests et le suivi tiennent. Mais une quinzaine de messages d'erreur classique du ch. 21 écrivent le calcul juste. Les petits 🔨 de `mylearn` sont sous-estimés d'un facteur 2 à 4. Le parcours maths arrive à CP5, MP5 et MP6 sans les notebooks des ch. 23 et 24. |

**Un majeur corrigé sans toucher au contrat** (A11 1). Les messages `mistakes` de la partie 0 du ch. 21 posent le calcul juste, à une addition de la réponse. C'est une fuite, qui va contre le §13 et la règle « Toujours valable » des messages. Les messages ne font pas partie du contrat (la bonne réponse ne change pas) : ils sont réécrits directement (§4).

## 3. Corrections majeures proposées (à valider)

> **Décision (2026-10-31)** : toutes les recommandations sont retenues (R1 A, R2 A, R3 A, R4 A, R5 A et B pour 24.24 et 24.27, R6 A, R7 A, R8 sans 25.25, R9 A). Application : §6.

Ces neuf points changent le contrat (titres, termes retenus, types, étoiles, durées, parcours, énoncés), une convention transversale ou la structure des outils. Ils sont exclus des corrections directes du §4 et attendent ta décision. Pour chacun, la recommandation est donnée en tête.

**R1 — Titres et termes des parties V et VI.** *Recommandé : A.* (A1 6, 7, 8, 9, 10 ; A3 24)

Onze titres publiés emploient la forme que le glossaire écarte, alors que R4 et R5 (session 43) avaient fixé « un terme par notion, titres compris ». Exemples : 24.8 et 24.E2 « data augmentation » (24.23 dit « augmentation de données ») ; 26.E4 et 26.31 « replay buffer » (le glossaire retient « mémoire de rejeu ») ; 23.28 et MP6.7 « model card » (« fiche modèle » ailleurs, jusque dans le cahier de MP6) ; 25.27 « convolutions à pas » (le §5 garde *stride*) et « sur-échantillonnage » ; 23.15 et MP5.1 « split stratifié » (R5 a réservé *split* au nœud d'un arbre) ; 27.R1 « (self-play) » (le glossaire retient « auto-apprentissage ») ; 29.R3, un tiret cadratin dans le contrat et « : » sur les supports. Trois notions ont deux noms sans que le glossaire tranche dans le sens de l'usage :
- « carte de caractéristiques » est retenue au glossaire, mais « carte d'activation » est dans les titres de 21.27 et de CP6.5 et dans tout le ch. 28 ;
- « sur politique » est retenu, mais *on-policy* est dans le titre de 26.E1 et dans CP6 ;
- « apprentissage par transfert » est retenu, mais *transfer learning* sert aux ch. 24, 28 et 29 et dans le titre du bonus B1.

Le §5 garde aussi *benchmark* en anglais, alors que le titre du MP6 dit « banc d'essai ».

- **A** :
  - **Titres** : aligner les onze titres (ID inchangés) sur le terme retenu, sur tous les supports et dans le JSON :
    - « L'augmentation de données en cinq lignes » ;
    - « … une mémoire de rejeu et un réseau cible ? », « Une mémoire de rejeu » ;
    - « Rédiger la fiche modèle », « README, fiche modèle, tests pytest… » ;
    - « Suréchantillonnage explicite ou convolutions à stride… » ;
    - « … découpage stratifié… » (23.15, MP5.1) ;
    - « … (auto-apprentissage) » ;
    - 29.R3 avec « : ».
  - **Termes** : trancher en faveur de l'usage. Retenir « carte d'activation » (*feature map*). Garder *on-policy* et *off-policy* en anglais, glosés « sur politique » et « hors politique ». Garder *transfer learning* en anglais, glosé « apprentissage par transfert ». Pour le MP6, garder « banc d'essai » dans la prose, et ajouter *benchmark* en glose au titre (« Banc d'essai génératif (*benchmark*)… ») ; ou bien, si tu préfères la règle stricte du §5, « Benchmark génératif : VAE contre GAN sur Fashion-MNIST ».
  - **Mise en œuvre** : glossaire et prose alignés, `test_terminology.py` étendu, une ligne au §22.
- **B** : aligner la prose seulement (les titres publiés gardent leur forme). Deux formes subsistent alors dans les titres.

**R2 — CP6 et le §17 : un 📦 à l'examen, deux étoiles, un titre et des types absents non justifiés.** *Recommandé : A.* (A3 13, A9 1 et 2)

- CP6.8 est un 📦, alors que le §17 exclut ce type de l'examen. CP4.9 et CP5.6 avaient justement été convertis en 🔨 « au contrat écrit ».
- CP6.4 (une preuve suivie d'interprétations) et CP6.9 (un 🐛 à trois erreurs) sont en ★. Le §17 demande ★★, comme pour CP4.5, CP4.7, CP5.4 et CP5.12.
- Le titre de CP6.9 dit « un pas d'entraînement » dans le sujet, les notebooks et le corrigé, mais « une boucle » dans le contrat.
- `cp6.json` n'a pas de champ `notes` pour justifier les types pratiqués mais absents (🔨, 🗣️, 📄), et son `study_notes` est périmé.

Options :
- **A** : faire comme pour CP5.
  - CP6.8 devient un 🔨 « au contrat écrit » ; l'énoncé, les réponses et les vérifications ne changent pas.
  - CP6.4 et CP6.9 passent en ★★.
  - Le contrat prend le titre « Réparer un pas d'entraînement de GAN ».
  - Un champ `notes` justifie les types absents : 🗣️, dont CP6.12 💼 tient lieu ; 📄, car lire un article demande l'article ; 📦, 🔬, 🛠️, 🔁, 🎨 et 🏆, exclus par le §17.
  - `study_notes` est réécrit, et l'écart 61 est complété.
- **B** : garder le 📦 par une dérogation écrite au §22, avec le même champ `notes`. Les étoiles et le titre sont corrigés comme en A.

**R3 — La partie B des notebooks d'examen contient en clair la solution des questions de code.** *Recommandé : A.* (A9 18)

L'apprenant a le notebook d'examen ouvert pendant l'épreuve, puisque la partie A s'y fait. La partie B, juste en dessous et non repliée, contient en clair :
- en CP6, `oracle_8` (la loss du VAE de CP6.8), `reference_step` (le pas de GAN réparé de CP6.9) et les trois lettres de CP6.9 a avec leur correction ;
- en CP5, `reference_fit` (la solution de CP5.6).

Cela fait environ 3,5 points sur 20 par examen. CP4 suit le même schéma, mais son énoncé donnait déjà les formules d'AdamW.

Options :
- **A** : déplacer les oracles dans un module `wb.exams`, testé dans `tests/infra/`. La partie B les importe, et ses messages restent des pistes. On l'applique à CP5 et CP6 (et à CP4, pour l'uniformité), avec une règle au §22 pour les examens suivants : « aucun oracle de code en clair dans un notebook d'examen ».
- **B** : sortir la partie B dans un second notebook, `02b_examen_verification.ipynb`, copié par `start_chapter.py`.
- **C** (le minimum) : replier les cellules de la partie B (métadonnée `source_hidden`) et ajouter un avertissement.

**R4 — `model.pt` est ignoré par git.** *Recommandé : A.* (A9 17, A11 8)

La règle `*.pt` du `.gitignore` écarte deux fichiers :
- le `model.pt` de la solution de MP5 n'a jamais été versionné, si bien qu'après un clone, la commande « Reproduire » de la solution échoue ;
- le livrable 4 de l'apprenant (`model.pt`) est écarté sans avertissement par `git add`, alors que la grille note « un modèle qu'on sait recharger ».

Options :
- **A** : des exceptions dans le `.gitignore`, `!projets/partie_5_classifieur_cifar10/solution/model.pt` et `!mon_travail/projets/*/model.pt`, et le commit du `model.pt` de la solution (300 Ko, sous le seuil des données lourdes). Une ligne au §22.
- **B** : garder l'exclusion, mais le dire dans le cahier des charges et le README de la solution (`git add -f`), et faire afficher un ⚠️ par MP5.8.

**R5 — L'échafaudage des parties V et VI.** *Recommandé : A, puis B pour 24.24 et 24.27 ; le reste de B à la prochaine reconstruction de chaque chapitre.* (A3 12)

La hausse de longueur des énoncés 🔨 mesurée par `syllabus.py stats` (310 et 313 mots, contre 251 en partie IV) est surtout un artefact. Le « Contrat commun » de `conv.py`, `rl.py`, `sequences.py`, `vae.py` et `gen_metrics.py` (110 à 125 mots) est recopié dans chaque 🔨 du module. Sans lui, le contrat propre d'un 🔨 reste à 152 mots, contre 142 en partie IV. Les 📦 de la partie V grossissent en revanche vraiment (189 mots, contre 144). Plusieurs vont contre R2 C de la session 43 (la méthode passe dans les indices) :
- 24.27 donne dans l'énoncé et la docstring la solution de `last_steps_27` (`output[:, -1, :H]` et `h_n[-2]`) ;
- 24.24 donne une recette de dessin que la vérification ne contrôle pas ;
- 21.22, 22.25, 24.30 et 27.20 donnent les appels ou les réglages qui se déduisent du contrat.

Options :
- **A** (outil et convention, sans toucher aux énoncés) :
  - `syllabus.py stats` exclut le paragraphe « Contrat commun », comme l'encadré « Mode d'emploi » ;
  - ligne au §22 : « le contrat commun d'un module s'écrit une fois, dans l'introduction de la partie du notebook qui ouvre le module ; chaque énoncé y renvoie ».
- **B** (énoncés, réponses inchangées) :
  - écrire une seule fois le contrat commun ;
  - réduire les énoncés de 24.27 et 24.24 à leur contrat ;
  - passer dans les indices 1 et 2 les appels et les réglages déductibles de 21.22, 22.25, 24.30 et 27.20 (liste détaillée dans A03, constat 12).
- **C** : ne rien changer, et écrire au §22 que les 📦 de la partie V restent longs pour fixer la reproductibilité.

**R6 — Les durées des petits 🔨 de `mylearn` (parties V et VI) et celle du MP5.** *Recommandé : A.* (A11 12, A9 10)

Les contrats des modules V et VI ont doublé ou triplé le code à écrire, sans que les durées suivent. Ces contrats imposent : refus des NaN, des booléens et des indices −1, formes diffusables, nouveaux tableaux `float64`. Mesures, sur la référence et dans la copie de l'apprenant simulé, à 1,3 à 1,9 min par ligne :

| Exercice | Lignes à écrire | Durée annoncée |
|---|---|---|
| 26.20 | environ 90 | 25 min |
| 26.21 | environ 110 | 45 min |
| 22.12 | environ 50 | 15 min |
| 26.14 | environ 30 | 15 min |

Le ch. 26 est ainsi sous-estimé de 4 à 5 h. Par ailleurs, les étapes du MP5 font 9 h 30, et non les 10 h du contrat, alors que la règle veut qu'une durée de mini-projet soit la somme de ses étapes.

Options :
- **A** :
  - recaler les durées : 21.12 → 30 min, 22.12 → 45, 22.13 → 40, 25.28 → 35, 26.14 → 35, 26.19 → 35, 26.20 → 75 (★★★), 26.21 → 120 (★★★, en gardant le plafond de l'étoile), 26.31 → 40, 21.23 → 25 ;
  - MP5 à 9 h 30, ou 30 min de plus sur MP5.8 ;
  - `syllabus.py build`, puis les en-têtes « Temps total estimé » et une ligne au §22.
- **B** : alléger le contrat avec un module de validation fourni (`_checks`). Cela change des énoncés publiés et des stubs : déconseillé.
- **C** : ne rien changer, et recaler en P7 sur les temps réels notés au tableau de bord.

**R7 — Le parcours maths sans pratique de PyTorch avant CP5, MP5 et MP6.** *Recommandé : A.* (A11 13)

C'est le défaut que R8 avait corrigé au ch. 20, et il revient aux ch. 23 et 24. Les 36 exercices de notebook de ces deux chapitres sont tous au parcours R ou C. Or l'apprenant maths doit écrire à CP5.6 une boucle avec early stopping et meilleur modèle, puis, dans MP5, un `fit()` complet, l'augmentation `transforms.v2`, la sauvegarde et la reprise exacte. Au ch. 27, il ne fait ni le DCGAN (27.20) ni le checkpoint d'un GAN (27.27), que MP6.4 suppose. `syllabus.py check` ne le voit pas, car il ne compte que les fonctions `mylearn`.

Options :
- **A** : ajouter au parcours maths 23.15, 23.16, 23.18, 23.23, 23.25 (environ 2 h 25), 24.23 (45 min), 27.20 et 27.27 (75 min), puis `syllabus.py build` et une ligne au §22.
- **B** (le minimum) :
  - déclarer des prérequis d'exercice dans `cp5.json` ;
  - écrire dans les cahiers des charges de MP5 et MP6 : « fais d'abord 23.18, 23.23, 23.25 et 24.23 (MP5), 27.20 et 27.27 (MP6) ».

**R8 — Neuf énoncés à préciser (réponses inchangées).** *Recommandé : tous, sauf 25.25 (facultatif).* (propositions des correcteurs F1 à F7, rapports `F*.md`)

| Exercice | Ce qui change dans l'énoncé | Raison |
|---|---|---|
| 21.Q8 a | « … et des couches suivantes (**sans dropout**) partent d'une même valeur… » | le dropout casse la symétrie ; la solution le disait, pas l'énoncé (A4 13) |
| 22.20 | le contrat ajoute : « elle rend un **nouveau** tableau NumPy `float64` de la forme des logits » | ce que dit la docstring figée et ce que vérifie le test ajouté (A10 12) |
| 24.R1 a | option D : « le learning rate de chaque epoch (utile dès qu'un planning le fait varier) » | l'historique du ch. 23 n'a pas de learning rate : « AB » se défendait (A5 12) |
| 26.17 | « N'utilise pas `Flippers().min_moves` : elle donne la réponse. » (la phrase qui la nommait comme outil est déjà retirée par F3, sans changer le sens) | la méthode fournie calculait les trois réponses en une ligne (A11 4) |
| 26.35 | « … et contre minimax, un adversaire **parfait** (fiche §26.10) : il joue toujours le coup qui donne le meilleur résultat dans le pire des cas » | minimax était utilisé avant le ch. 27 ; la fiche a maintenant son encadré (A2 12) |
| 25.25 (facultatif) | décrire des symptômes au lieu de nommer les trois diagnostics ; les corrections passent à l'indice 2 | l'énoncé nommait chaque bug (A11 11, suggestion) |
| 29.15 | « Modèle : les six sections ci-dessus ; les fiches de `data/cards/` montrent le niveau de détail attendu (elles n'ont pas toutes ces six titres). » | un apprenant qui recopie `cifar10.md` échouait à `check_card` (A8 18) |
| CP6.8 | « des **logits** tirés au hasard, comme ceux d'un décodeur pas encore entraîné… : la reconstruction sera bien plus grande que celle d'un VAE entraîné » | « un décodeur a rendus » laissait croire à un VAE entraîné (A9 19 ; à faire avec R2) |
| CP6.7 | « (une lettre par panneau, **1/3 de point chacune** ; …) » | le sous-barème n'était que dans le corrigé (A9 20) |

**R9 — Trois conventions de forme à écrire dans la bible.** *Recommandé : A pour chacune.*

- **Encadré 📐 « Conventions de la fiche »** (ch. 22 à 29) : il est absent du §14 (A1 18).
  - A : l'ajouter au §14 (« notations du chapitre, et tout écart au §6, signalé une fois ») ;
  - B : le remplacer par l'encadré 🧮.
- **Gloses répétées** (A1 26). Le §5 veut une glose à la première occurrence, dans le chapitre qui définit le terme ; or les parties V et VI rappellent des gloses d'autres parties (« le stride (*pas*), ch. 20 »).
  - A : écrire au §5 qu'un chapitre peut rappeler une fois la glose d'un terme défini ailleurs, avec le renvoi ;
  - B : retirer ces rappels.
- **Relances des 💼** (A1 5). Les ch. 12 à 28 les écrivent sur une ligne ; le ch. 29, en liste à puces.
  - A : la ligne partout, et le ch. 29 aligné ;
  - B : la liste partout, à la prochaine reconstruction de chaque chapitre.

Une fois ces choix faits (et R1), trois tests de format pourront être ajoutés à `tests/infra/test_syllabus.py` (proposition de F7) :
- l'icône du type à la fin des en-têtes de `04_indices.md` et `05_solutions.md` ;
- le préfixe « Ch. N : » et « 🔁 ★ » des rappels ;
- le temps du parcours rapide du guide de lecture.

Ils empêcheraient la dérive constatée par A1 et A3, la deuxième d'un audit à l'autre.

## 4. Tri et application des constats mineurs (2026-10-31)

### Les correcteurs

Sept correcteurs (des sous-agents) ont travaillé, chacun sur une zone du dépôt.
- **Ce qu'ils ont fait.** Ils ont lu les onze rapports et vérifié chaque constat de leur zone dans les fichiers : calculs refaits, code rejoué, livre relu, et pour le ch. 27 le code de l'auteur. Ils ont ensuite appliqué ce qui ne change pas le contrat.
- **Ce qui leur était interdit.** Les propositions R1 à R7 ; ils ont reporté au §3 (R8, R9) ce qui touchait un énoncé.
- **Leurs vérifications.** Chacun a reconstruit ses notebooks par leur script. Dans une copie du dépôt, il a ensuite fait un « Tout exécuter » à vide et exécuté les solutions.
- **Le partage de la machine.** Les commandes lourdes passaient par un verrou commun, une seule à la fois.
- **Leurs rapports.** Leurs tableaux de décision (constat → appliqué, non retenu avec la raison, reporté Rn, proposé, hors zone) sont dans `docs/audit_P5_parties_V_VI/F1_ch21_22.md` à `F7_transversal.md`.

| Zone | Correcteur | Décisions | L'essentiel |
|---|---|---|---|
| ch. 21 et 22 | F1 | 28 appliqués, 1 non retenu, 4 reportés | Environ 70 messages `mistakes` du ch. 21 et une trentaine du ch. 22 réécrits : ils nomment l'oubli sans poser le calcul (A11 1). Deux d'entre eux donnaient la réponse elle-même (22.R1 c, 22.4 a), et 21.13 certifiait les réponses de 21.1 c et f. 1 199 882 n'est plus écrit en clair en 21.22 et 21.24. Les cellules des 🔨 passent une copie de l'entrée et signalent une entrée modifiée. 22.14 ne note ses prédictions qu'après l'expérience. Temps de lecture du ch. 21 recalculés (8 h et 5,3 h). « État caché » défini. Renvoi de 22.R1. Flashcards du ch. 22 en MathJax. Formats. |
| ch. 23 et 24 | F2 | 24 appliqués, 2 non retenus, 2 reportés | `view` et la contiguïté. La confusion *channels_last* de la p. 1106 signalée. Extraits scikit-learn de la fiche 24 qui tournent tels quels. `num_workers` sous Windows et macOS. Indices 3 de 23.5, 23.25 et 24.2 réduits. Variante de 24.20 (rapport 3,5). Liens figés sur PyTorch 2.11 et torchvision 0.26. 🐛 24.19 enchaîné selon R11 B, rejoué sur ses 64 versions partiellement corrigées. 24.16 à 24.18 attendent le code de 24.16 au lieu d'accuser le leur. *Run* et *worker* glosés. |
| ch. 25 et 26 | F3 | 27 appliqués, 3 non retenus, 3 reportés | Le L-learning du livre n'est plus dit « marcher ». La fiche ne donne plus la réponse de 26.17. Le RLHF renvoie au B8. Encadré « minimax, le joueur parfait » au §26.10. Notation de Sutton et Barto. Solutions de 25.20, 25.21 et 26.26. Plus d'avertissement en 25.38. 26.17 comparé à une recherche privée de `wb.envs`. 26.19 et 26.28 mieux gardés (`prereq` de 26.28 complété). Onze flashcards en questions actives. Formats. |
| ch. 27 et 28 | F4 | 25 appliqués, 2 reportés | Bruit gaussien de 27.15 (code de l'auteur). Le « scénario Helvetica » : le livre avait raison. Normalisation du gradient du deep dream. FID « pour peu d'images ». Dimension du bruit. L'erreur « backward through the graph a second time ». Flashcard « minimax ». Deux figures. 28.6. Paraphrases réécrites. Indice 3 de 27.22. Rubriques des 🗣️, ⚖️, 📄 et 💼. 🐛 28.21 enchaîné. `prereq` de 27.20. « mini-batches » et « suréchantillonnage » dans la prose. |
| ch. 29, data cards, `wb.datasets` | F5 | 24 appliqués, 2 reportés | Le dataset piégé écorche maintenant le vrai fruit de la ligne : trois labels n'étaient plus faux, et un test le garde. Les réponses sont inchangées ; seule une précision affichée passe de 0,963 à 0,967. Fiche du ch. 29 (prérequis Fashion-MNIST, liens, Boston). Solution 29.11 (CC BY-SA). Data cards : chapitres, tailles en Mio (Mo), liens, empreintes SHA-256 vérifiées par un test. `pretrained.md` couvre ResNet-18 et MobileNetV3. CIFAR-10 se charge du Hub à une révision figée. Formats. |
| checkpoints et mini-projets | F6 | 24 appliqués, 3 non retenus, 10 reportés | La synthèse de CP6 ne donne plus l'issue du 🔮 CP6.6. Le kit MP5 range son modèle dans un dossier propre à la session, et son coffre ne s'ouvre plus sans commit (sept cas essayés). Messages de CP5.6 et CP6.9. Erreurs fréquentes de CP5.2 et CP6.6 (2,7). Remédiations de CP5.11 d et CP6.1 a. Rubriques de CP6.11 et CP6.12. Titres des sujets alignés sur le contrat (sauf CP6.9, R2). « fine-tuning » et « marche à suivre ». Commandes de reproduction de MP5 et MP6. |
| annexes, code, tests, outils | F7 | 42 appliqués, 2 reportés | Glossaire (sept doublons, une dizaine de termes), formulaire (GRU de Keras, σ, BCE, PSNR), erreurs fréquentes, aide-mémoire PyTorch (sections des ch. 26 et 29), ressources et métiers des parties V et VI. Référence : `rng` entier dans `rl.py`, NaN refusés par `sample_from_logits` (Q9). Nouveaux tests : le slerp à 10⁻³ rad tue le mutant survivant, le type de `temperature_softmax`, les groupes `-k` des notebooks couvrent tout leur fichier de tests. `wb` : `attempt`, `checked`, `same` et `as_numpy` face aux tenseurs PyTorch. `chapter_kit` : 🚀 et fil rouge lus dans le contrat. `test_terminology.py` lit les `ch*_parts.py` et `ch*_code.py`. README (partie VI ✅) et `00_setup/` (hors ligne, Colab, entraînements longs). |

### Ce que le coordinateur a fait en plus

- **Ch. 27 et 28** : il a appliqué les formats que F4 avait laissés hors de sa liste. Les icônes des en-têtes de `04_indices.md` (57 en-têtes) ; le préfixe « Ch. N : », l'étoile et le parcours des rappels ; la ligne « parcours » des quiz et des 💼 dans `02_exercices.md`.
- **Autres fichiers** :
  - la taille de CIFAR-10 (≈ 145 Mo depuis le Hub) dans le kit et le cahier des charges de MP5, et dans `00_setup/demo.ipynb` ;
  - deux lignes de `suivi/PROGRESS.md`.

### Réponses enregistrées

Aucune bonne réponse ne change de valeur. Seuls changent des messages d'erreur classique :
- au ch. 21, environ 70 sous-ID ;
- au ch. 22, environ 30 sous-ID ;
- les messages de 25.R3 d, 29.Q7 e et CP5.2 c.

### Laissés pour plus tard (notés)

- **Termes en attente de R1.** « convolutions à pas » et « sur-échantillonnage » restent dans la prose des ch. 20, 21, 25, 27 et 28. « affiné » reste au ch. 11. Ils seront remplacés avec R1, et les motifs de `test_terminology.py` ajoutés à ce moment-là (texte prêt dans F7).
- **Tableau des exercices.** La ligne du tableau de la cellule 0 par `Ex.table_row` (🚀 dans le tableau) n'est pas encore adoptée par les ch. 21 à 26 ; l'en-tête de chaque exercice porte déjà le 🚀.
- **Garde-fous des mini-projets.** Un module `wb.projects`, qui réunirait les garde-fous des mini-projets (MP4 à MP6), reste à créer (A3 8).
- **Coffre de MP3.** Le `committed` de MP3 prend un dépôt sans aucun commit pour « pas de dépôt git » ; la version corrigée de MP5 est à y reporter.
- **Infinis dans `vae`.** Le refus de ±inf par `vae` (cohérence avec `gen_metrics`) n'est pas fait : il faut d'abord le vérifier sur les expériences du ch. 25 qui font diverger `log_var`.

## 5. Vérification finale (2026-10-31)

### Reconstruction et exécution

- **Reconstruction.** Les treize scripts (`build_ch21.py` à `build_ch29.py`, `build_cp5.py`, `build_cp6.py`, `build_mp5.py`, `build_mp6.py`) ont été relancés après la modification de `chapter_kit.py` par F7 (🚀, fil rouge). Les notebooks reconstruits ne diffèrent des versions des correcteurs que par les identifiants de cellules.
- **Notebooks de solutions.** Les treize ont été réexécutés à la suite, une fois les correcteurs arrêtés, sur une machine calme :

  | Notebook | Durée |
  |---|---|
  | ch. 21 | 155 s |
  | ch. 22 | 171 s |
  | ch. 23 | 121 s |
  | ch. 24 | 331 s |
  | ch. 25 | 183 s |
  | ch. 26 | 138 s |
  | ch. 27 | 206 s |
  | ch. 28 | 131 s |
  | ch. 29 | 13 s |
  | CP5 | 44 s |
  | CP6 | 5 s |
  | MP5 | 373 s |
  | MP6 | 219 s |

  Aucun n'est hors budget (§4), et aucune sortie n'a d'erreur ni d'avertissement sur stderr. Les seuls ❌ et ⏳ sont ceux des cellules données des 🐛 corrigés sur place (25.25, 26.24, 27.16, 28.21, R11). Dans `results_rapide.json` (MP6) et `runs_rapide.csv` (MP5), seuls les temps et les dates changent.
- **« Tout exécuter » à vide.** Les neuf notebooks d'exercices, les deux examens et les deux kits sont allés au bout sans erreur.

### Réponses, tests et cohérence

- **Réponses.** `answers.json` a été régénéré (4 590 réponses). Aucune bonne réponse ne change. 122 sous-ID ne changent que par leurs messages : ceux des ch. 21 et 22 (A11 1), ainsi que 25.R3 d, 29.Q7 e et CP5.2 c.
- **Tests.**

  | Mode | Résultat |
  |---|---|
  | apprenant | 504 réussis, 3 964 ignorés |
  | `--impl=ref` | 4 463 réussis, 5 ignorés |
  | `--impl=stubs` (tests de chapitre) | 3 953 échecs attendus ; le seul test qui passe porte sur une fonction fournie |

- **Cohérence.**
  - `syllabus.py build` a été relancé, puis `syllabus.py check` (2 028 exercices, 0 problème).
  - `build_answers.py --check`, `export_flashcards.py --check` (757 cartes) et `check_env.py` sont au vert.
  - L'en-tête du ch. 26 a été recalé sur le SYLLABUS (lecture 7,9 h, parcours rapide 5,9 h).
  - `mon_travail/` est intact.

### Suivi

- `suivi/PROGRESS.md` : session 62, les écarts, la ligne 35 « À valider sur Colab » (CIFAR-10 à sa révision figée, durée du mode rapide du MP5), deux lignes corrigées.
- Sept lignes au §22 de la bible.

## Annexe A — consignes des relecteurs

Les consignes communes reprennent celles des audits précédents (`docs/AUDIT_parties_I_II.md`, annexe A.1 ; `docs/AUDIT_parties_III_IV.md`, annexe A), avec quatre changements :
- **le périmètre** : ch. 21 à 29, CP5, CP6, MP5, MP6, les modules `mylearn` `conv`, `rnn`, `sequences`, `vae`, `rl` et `gen_metrics` et leurs tests, `wb.nbtools` et `wb.envs`, les data cards et les lignes des annexes de ces chapitres. Les parties 0 à IV ne sont relues que pour la cohérence ;
- **les conventions de la session 43** (R1 à R11) : une dérive de ces conventions dans les parties V et VI est un constat ;
- **la machine partagée** : un seul processus lourd à la fois, aucune durée jugée ;
- **les citations du livre en anglais**, balisées dans les rapports bruts, sont retirées de la version versionnée.

Le texte complet des consignes est dans `docs/audit_P5_parties_V_VI/00_consignes_communes.md` (relecteurs) et `00_consignes_F.md` (correcteurs). Chaque relecteur a écrit son rapport au fur et à mesure (constat, sévérité, fichier et ligne, extrait exact, problème, correction précise), puis un bilan.

## 6. Application des propositions R1 à R9 (2026-10-31)

> **Décision (2026-10-31)** : l'utilisateur m'a laissé décider. Toutes les recommandations sont retenues, avec ces choix :
> - **R1** : « carte d'activation » ; *on-policy*, *off-policy* et *transfer learning* gardés en anglais et glosés ; « banc d'essai » garde le titre du MP6 et se glose *benchmark* dans la prose.
> - **R3** : `wb.exams` sert aussi à CP4.
> - **R5** : A, plus 24.24 et 24.27 ; les autres énoncés seront repris à la prochaine reconstruction de leur chapitre.
> - **R6** : les durées restent dans les bornes d'étoiles du §11 ; le MP5 garde 10 h, avec 30 min de plus pour MP5.8.
> - **R8** : sans 25.25.
> - **R9** : A pour les trois conventions.

Sept agents (G1 à G7) ont appliqué ces décisions zone par zone, chacun avec ses vérifications dans une copie. Leurs rapports et leurs consignes sont dans `docs/audit_P5_parties_V_VI/` (`G1` à `G7`, `00_consignes_G.md`). Un relecteur de contrôle indépendant (C1) a ensuite tout revu en lecture seule.

| Décision | Ce qui a changé |
|---|---|
| **R1** | Douze titres, sur tous les supports et dans les JSON : 23.15, 23.28, 24.8, 24.E2, 25.27, 26.E4, 26.31, 27.R1, 27.Q7, 29.R3, MP5.1 et MP6.7 (et sa grille). Les termes retenus remplacent les formes écartées dans toute la prose, ch. 1, 11 et 20 compris : carte d'activation, stride, suréchantillonnage, mini-batches, fine-tuning, découpage, augmentation de données, mémoire de rejeu, fiche modèle, auto-apprentissage. *on-policy*, *off-policy* et *transfer learning* sont glosés à leur première occurrence. Le glossaire suit. `test_terminology.py` refuse maintenant les formes écartées, avec quatre sens courants de « affiner » en exception. |
| **R2** | CP6.8 devient un 🔨 ; CP6.4 et CP6.9 passent en ★★ ; CP6.9 prend le titre « Réparer un pas d'entraînement de GAN » ; `cp6.json` reçoit `notes` et un `study_notes` réécrit. |
| **R3** | `src/wb/exams.py` (sept oracles et les pistes qui donnent une formule), testé par `tests/infra/test_wb_exams.py` : 17 tests, dont un qui refuse toute solution en clair dans les trois notebooks d'examen. La partie B a été rejouée avant et après, à vide, avec la solution et avec 17 variantes fautives : les verdicts sont identiques. Deux fuites ont été trouvées en passant, dans des docstrings d'exemple (« CGJ », « BDFI », la réponse de CP5.4 a) : elles sont neutralisées. |
| **R4** | Deux exceptions dans `.gitignore`. Le `model.pt` de la solution de MP5 est versionné, et le cahier des charges dit que le modèle livré se commite. |
| **R5** | `syllabus.py stats` ne compte plus le « Contrat commun » : les 🔨 passent à 215, 200 et 227 mots en parties IV, V et VI, contre 251, 310 et 313. 24.24 garde son contrat, et sa recette passe à l'indice 2 ; 24.27 garde « le sens avant seulement », et les deux expressions passent à l'indice 3. |
| **R6** | 21.12 ★★ 30, 21.23 ★★ 25, 22.12 ★★ 30, 22.13 ★★ 30, 25.28 ★★★ 35, 26.14 ★★ 30, 26.19 ★★★ 35, 26.20 ★★★ 75, 26.21 ★★★ 90, 26.31 ★★★ 40 ; MP5.8 passe à 90 min, et les étapes du MP5 font 10 h. |
| **R7** | 23.15, 23.16, 23.18, 23.23, 23.25, 24.23, 27.20 et 27.27 entrent au parcours maths, avec les prérequis de code que `syllabus.py check` exige : 23.12, 23.13, 24.19, 24.20 et 24.22. |
| **R8** | Huit énoncés précisés (21.Q8 a, 22.20, 24.R1 a, 26.17, 26.35, 29.15, CP6.7, CP6.8), réponses inchangées. |
| **R9** | Les §14 (📐), §5 (gloses rappelées) et §12 (relances sur une ligne) de la bible ; les relances du ch. 29 ; l'encadré 📐 du ch. 22. Trois tests de format dans `test_syllabus.py` : icône du type dans les en-têtes de 04 et 05, préfixe « Ch. N : » des rappels (à partir du ch. 12), « 🔁 ★ » dans 02. Seul 21.R3 était hors format. |

**Contrôle (C1).**
- **Ce qui est conforme.** C1 a comparé 2 342 en-têtes au JSON : aucune divergence. Les durées respectent les bornes. Aucune solution ne reste en clair dans les examens. Les `wb.record` ne changent ni de valeur, ni de décimales, ni de tolérance.
- **Les restes corrigés.** C1 a relevé huit restes, tous corrigés avant le commit :
  - le « pas » au sens de stride dans quatre phrases des ch. 27 et 28, et au ch. 20 (pooling de 20.14 et un quiz) ;
  - « on l'affine » au ch. 1 ;
  - « Data augmentation » dans la synthèse de CP5 ;
  - les comptes d'énoncés dans la bible ;
  - les écarts dans PROGRESS ;
  - `answers.json`.
- **Laissés pour plus tard.** `help(wb.exams)`, permis pendant l'examen, afficherait encore la formule de la KL et les corrections de CP6.9 dans les docstrings. Il faudra les réduire à une phrase lors de la prochaine reconstruction des examens. Restent aussi le « banc d'essai » sans glose avant le MP6 et le 🚀 dans le tableau de la cellule 0 des ch. 21 à 26.

**Vérification.**
- **Notebooks de solutions.** Les quatorze notebooks reconstruits ont été réexécutés : ch. 21 à 29, CP4 à CP6, MP5, MP6, puis le ch. 20, dont un message change. Aucun n'a d'erreur. Le conteneur a redémarré en cours de session, et la nouvelle machine est plus lente que celle du §5 (même code : ch. 21 en 299 s sous charge, 222 s ensuite, contre 155 s). Les notebooks qui dépassaient le budget du §4 sous charge ont été relancés sur la machine calme, et tous passent : ch. 21 en 222 s (le défi 21.32 s'entraîne en 132 s, sous sa limite de 150 s, alors qu'il en prenait 179 sous charge), ch. 22 en 217 s, MP5 en 581 s. MP5 reste le plus proche du budget (ligne 35 « À valider sur Colab »). Aucune sortie n'a d'erreur ni de stderr ; les seuls ❌ sont ceux des 🐛 corrigés sur place. Les « Tout exécuter » à vide de quinze notebooks (03 des ch. 20 à 29, trois examens, deux kits) vont au bout sans erreur.
- **Réponses.** Aucune bonne réponse ne change ; seuls changent des messages (21.Q2 d, 25.3 e, 27.1 c, 20.14 c).
- **Tests et cohérence.** `answers.json` est à jour (4 590 réponses). Tests : 530 réussis en mode apprenant, 4 489 en `--impl=ref`, et 3 953 échecs attendus en `--impl=stubs`. `syllabus.py build` et `check` (0 problème), flashcards (757 cartes) et `check_env.py` sont au vert. Les en-têtes « Temps total estimé » concordent avec le SYLLABUS (ch. 21 : 29 h, ch. 26 : 32 h). `mon_travail/` est intact.
