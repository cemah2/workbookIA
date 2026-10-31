# Audit P5 des parties V et VI

*Chapitres 21 à 29, checkpoints V et VI, mini-projets MP5 et MP6. Prompt P5 (`docs/METHODE.md` §3), session 62.*

> **Statut : 🔍 en cours (2026-10-31).**
> - Onze relecteurs indépendants ont travaillé en lecture seule sur le commit `b961aa3`. Leurs rapports bruts sont dans `docs/audit_P5_parties_V_VI/`, de A01 à A11.
> - Des correcteurs par zone (F1 à F7) vérifient puis appliquent les constats mineurs : voir le §4.
> - Les propositions R1 à R7 (§3) attendent ta décision.

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

Ces sept points changent le contrat (titres, termes retenus, types, étoiles, durées, parcours, énoncés), une convention transversale ou la structure des outils. Ils sont exclus des corrections directes du §4 et attendent ta décision. Pour chacun, la recommandation est donnée en tête.

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

*D'autres propositions peuvent venir des correcteurs du §4 : ce seront des énoncés à préciser qu'ils ne peuvent pas changer seuls.*

## 4. Tri et application des constats mineurs

*En cours.*

## 5. Vérification finale

*À venir.*

## Annexe A — consignes des relecteurs

Les consignes communes reprennent celles des audits précédents (`docs/AUDIT_parties_I_II.md`, annexe A.1 ; `docs/AUDIT_parties_III_IV.md`, annexe A), avec quatre changements :
- **le périmètre** : ch. 21 à 29, CP5, CP6, MP5, MP6, les modules `mylearn` `conv`, `rnn`, `sequences`, `vae`, `rl` et `gen_metrics` et leurs tests, `wb.nbtools` et `wb.envs`, les data cards et les lignes des annexes de ces chapitres. Les parties 0 à IV ne sont relues que pour la cohérence ;
- **les conventions de la session 43** (R1 à R11) : une dérive de ces conventions dans les parties V et VI est un constat ;
- **la machine partagée** : un seul processus lourd à la fois, aucune durée jugée ;
- **les citations du livre en anglais**, balisées dans les rapports bruts, sont retirées de la version versionnée.

Chaque relecteur a écrit son rapport au fur et à mesure (constat, sévérité, fichier et ligne, extrait exact, problème, correction précise), puis un bilan.
