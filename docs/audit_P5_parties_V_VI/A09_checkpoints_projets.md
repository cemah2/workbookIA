# Audit P5 parties V et VI — A9 checkpoints et mini-projets (commit b961aa3)

**Périmètre :** CP5, CP6 (`checkpoints/partie_5`, `partie_6`), MP5 (`projets/partie_5_classifieur_cifar10`, `cifarkit`), MP6 (`projets/partie_6_generatif`, `genbench`), `tools/chapters/build_cp5.py`, `build_cp6.py`, `build_mp5.py`, `build_mp6.py`, `figures_cp5.py`, `figures_cp6.py`, `docs/syllabus/data/cp5.json`, `cp6.json` ; comparaison avec CP3, CP4, MP3, MP4. Scripts de vérification (dans ce dossier) : listés au fil des constats.

## Constats

Scripts de vérification : `scripts/cp5_recalc.py`, `scripts/cp6_recalc.py` (toutes les réponses chiffrées recalculées, avec PyTorch comme oracle quand c'est possible), `scripts/contract.py` (en-têtes et tableau du sujet contre `cpN.json`), `scripts/remed.py` (remédiations), `scripts/forms.py` (formes de réponses acceptées par `wb.check`), `scripts/fill_cp5.py` (notebook d'examen rempli par une solution écrite autrement, puis trois mutants), `scripts/nbdump.py`. Copie : `COPIE/` (`start_chapter.py --init`, `CP5`, `CP6`, puis exécutions dans `COPIE/runs/`).

### 1. MAJEUR · `docs/syllabus/data/cp6.json` (CP6.8) et `checkpoints/partie_6/01_examen_sujet.md:23` (tableau)
**Extrait :** « | CP6.8 | 📦 | Écrire la loss d'un VAE en PyTorch et la tester sur un batch fourni (notebook) | 2 | 15 min | »
**Problème :** le §17 exclut le 📦 de l'examen (« sauf 📦, 🔬 et 🛠️, qui vont dans le mini-projet »). CP4 (CP4.9) et CP5 (CP5.6) ont justement converti leur 📦 du contrat de la session 2 en 🔨 « au contrat écrit » pour cette raison (notes de `cp4.json` et `cp5.json`, écart 52). CP6 garde le 📦 sans dérogation : la ligne §22 du 2026-10-31 dit seulement « l'examen n'a ni 📄, ni 🔬, ni 🛠️ (§17) ». Le même écart touche les types absents. Les ch. 25, 26 et 27 pratiquent le 🔨 (12 exercices), et les cinq chapitres ont chacun un 🗣️ et un 📄 ; l'examen n'a ni 🔨, ni 🗣️, ni 📄. De plus, `cp6.json` n'a **pas de champ `notes`**, alors que le §17 veut que « un type pratiqué mais absent de l'examen [soit] justifié dans les `notes` ». Enfin, son `study_notes` est périmé et se contredit : « Types couverts : ✏️ ∂ 🔮 📈 📦 🐛 🧮 ⚖️ 🗣️ » (🗣️ absent de l'examen, 🧠 et 💼 oubliés), puis « questions flash typées ✏️, question d'entretien typée 🗣️ […] caduques ».
**Correction :** options.
- (A, recommandée) Faire comme CP4 et CP5 : CP6.8 devient un 🔨, la loss écrite d'après son contrat avec PyTorch comme bibliothèque de tenseurs (énoncé, réponses et vérifications inchangés). Changer le type dans `cp6.json`, le sujet, le corrigé, `build_cp6.py` et le tableau de bord. Ajouter un champ `notes` sur le modèle de `cp5.json` : « Types pratiqués mais absents : 🗣️ (l'oral est évalué par CP6.12 💼), 📄 (lire un article demande l'article) ; 📦, 🔬, 🛠️, 🔁, 🎨 et 🏆 sont exclus par le §17 ». Réécrire `study_notes` (types réels : 🧠 ✏️ ∂ 🔮 📈 🔨 🐛 🧮 ⚖️ 💼), une ligne d'écart dans `PROGRESS.md`, une ligne au §22.
- (B) Garder le 📦 par une DÉROGATION explicite au §22, avec ses raisons, et ajouter le même champ `notes`.

### 2. MAJEUR · `docs/syllabus/data/cp6.json` (étoiles de CP6.4 et CP6.9) ; `checkpoints/partie_6/01_examen_sujet.md:99,155`
**Extrait :** « ### CP6.4 — Discriminateur optimal et losses à l'équilibre ∂ ★ » ; « ### CP6.9 — Réparer un pas d'entraînement de GAN 🐛 ★ ⏱️ 10 min · 1,5 point » ; « Le pas contient **trois** lignes fautives »
**Problème :** le §17 donne « ★★ = deux notions combinées, une interprétation, une preuve, ou un 🐛 à plusieurs erreurs ». CP6.4 a) est une preuve (« Montre que le discriminateur optimal est… ») et CP6.9 un 🐛 à trois erreurs. Les checkpoints précédents suivent la règle : CP4.5 ∂ ★★, CP4.7 🐛 ★★, CP5.4 🐛 ★★ (« un 🐛 à plusieurs erreurs »), CP5.12 ∂ ★★. Douze des treize questions de CP6 sont ★, et seul le 📦 CP6.8 est ★★. L'examen paraît ainsi plus facile que CP5 sur le tableau de bord, et la remédiation « reprends en plus les exercices ★★ » vise mal.
**Correction :** (A, recommandée) CP6.4 et CP6.9 en ★★ dans `cp6.json`, le sujet (en-têtes) et les copies de `build_cp6.py` (en-tête de CP6.9 dans le notebook) ; `syllabus.py build` ; une ligne d'écart. (B) Garder ★ et justifier dans les `notes` (constat 1) pourquoi la règle ne s'applique pas.

### 3. MINEUR · `checkpoints/partie_6/05_synthese.md` §4 (« Le retour ») et §4 (« Explorer »)
**Extrait :** « un petit $\gamma$ rend l'agent myope (CP6.6 : avec $\gamma = 0$, il ne voit que les victoires à un coup) » ; « des égalités départagées au hasard explorent aussi tant que la table est vide »
**Problème :** la synthèse se lit **avant** l'examen (README, déroulé 2 ; en-tête de la synthèse). Ces deux phrases donnent le raisonnement, et en fait l'issue, des deux prédictions du 🔮 CP6.6 (a : C, b : A). La première nomme même la question. Le 🔮 ne mesure plus une prédiction. Aucune synthèse précédente ne renvoie à une question de son examen.
**Correction :** retirer la parenthèse « (CP6.6 : … à un coup) » et la proposition « des égalités départagées au hasard explorent aussi tant que la table est vide ». Écrire par exemple : « un petit $\gamma$ rend l'agent myope : l'avenir compte peu » et « Explorer : ε-greedy (souvent décroissant), softmax, ou une table initialisée de façon optimiste. » Le corrigé de CP6.6 garde l'explication.

### 4. MINEUR · `checkpoints/partie_6/03_examen_corrige.md` (CP6.6 b)
**Extrait :** « elle gagne **100 %** des plateaux (pour les trois graines), en 5 coups environ au lieu de 2,6 au mieux »
**Problème :** la longueur optimale moyenne sur les 506 plateaux non gagnants vaut 2,70 (`wb.envs._flippers_distances(3)`, moyenne sur les états non gagnants : 2,7036), et le ch. 26 l'écrit (« la moyenne sur les 506 plateaux non gagnants vaut **2,70** », `chapitres/ch26_rl/05_solutions.md:306`). De plus, la phrase suivante (« plus rien ne pousse l'agent à essayer d'autres chemins : ses parties restent deux fois plus longues que l'optimum ») laisse entendre que l'agent ε = 0,1 joue plus court. Or il fait 4,2 à 5,3 coups (`history_cp6.json`, `mean_moves`), presque autant.
**Correction :** « en 5 coups environ au lieu de 2,7 au mieux » ; et « ses parties restent deux fois plus longues que l'optimum (comme celles de l'agent ε = 0,1, 4,2 à 5,3 coups : 3 000 parties ne suffisent à aucun des deux pour jouer court) ».

### 5. MINEUR · `checkpoints/partie_5/03_examen_corrige.md` (CP5.2, erreurs fréquentes)
**Extrait :** « ignorer le stride de la convolution C, ou oublier que le padding s'ajoute des deux côtés (3 au lieu de 4) »
**Problème :** compter le padding d'un seul côté donne $\lfloor (8 + 1 - 3)/2 \rfloor + 1 = 4$, la bonne valeur. 3 vient d'un padding **oublié** : $\lfloor (8 - 3)/2 \rfloor + 1 = 3$.
**Correction :** « ignorer le stride de la convolution C (8 au lieu de 4), ou oublier le padding (3 au lieu de 4) ».

### 6. MINEUR · `checkpoints/partie_5/02_examen_notebook.ipynb`, partie B, cellule « Checks of fit_early_stopping (CP5.6) » (`tools/chapters/build_cp5.py`)
**Extrait :** « avec max_epochs = 2, patience = 5 et lr = 1e-3 sur 256 images, attendu les deux epochs du contrat : fit_early_stopping utilise-t-elle ses arguments (pas 30 epochs ni lr = 3e-3 écrits en dur) ? » ; « le fichier path ne contient pas les poids de la meilleure epoch : torch.save(model.state_dict(), path) à chaque progrès strict, et seulement alors. »
**Problème :** j'ai rejoué trois erreurs fréquentes du corrigé dans une copie (`scripts/fill_cp5.py`). Les valeurs du corrigé sont exactes : [12 ; 0,5124], [9 ; 0,4709] et [23 ; 0,4351]. Mais deux messages secondaires accusent la mauvaise étape, contre la règle « une propriété calculée sur la trajectoire de référence ne se vérifie qu'une fois la trajectoire de l'apprenant identique » (`PROGRESS.md`, « Toujours valable », CP4.9 d).
- **`model.eval()` oublié.** Le premier message est juste (« model.eval() avant la validation »). La propriété « max_epochs = 2 » échoue pourtant avec la piste « des arguments écrits en dur », car le dropout rend la validation aléatoire. CP6.8 évite ce piège avec `main_ok` (« c'est la conséquence des erreurs ci-dessus »).
- **`waited > patience`.** En plus du message juste (27 epochs au lieu de 12), deux ❌ disent que la sauvegarde et le rechargement sont faux (« torch.save … à chaque progrès strict, et seulement alors » ; « model.load_state_dict(…) avant le return »). Or ils sont justes : seul l'arrêt est tardif.
**Correction :** dans `build_cp5.py`, ne tester `best_epoch`, le fichier `path` et les poids finaux que si `len(mine_va) == len(ref_va)`. Sinon, une ligne : « (best_epoch et les poids gardés se vérifient une fois l'arrêt juste) ». Pour la propriété `max_epochs = 2`, reprendre la logique de CP6.8 : si les losses de la trajectoire principale sont déjà fausses, écrire « c'est la conséquence de l'écart ci-dessus », sinon garder la piste des arguments écrits en dur.

### 7. MINEUR · `checkpoints/partie_6/02_examen_notebook.ipynb`, partie B, cellule « Checks of gan_step_fixed (CP6.9) » (`tools/chapters/build_cp6.py`)
**Extrait :** « après trois pas, les poids du discriminateur diffèrent de ceux du pas de référence : vrais exemples avec le label 1, faux DÉTACHÉS avec le label 0, et un seul tirage torch.randn (ligne B). »
**Problème :** un pas corrigé partout sauf en J (`opt_d.step()` laissé à la place de `opt_g.step()`, `detach` et label 1 corrects ; variante rejouée dans une copie) reçoit cette piste. Elle désigne le pas de D, qui est juste. La branche prévue pour ce cas (« le premier pas de D est juste ; l'écart vient ensuite du pas du générateur (un opt_d.step() de trop…) ») ne peut pas se déclencher : `run_steps(…, n=1)` compare les poids de D **après un pas complet**, J compris, donc `first_d` est faux. Le bug le plus classique de la question est ainsi attribué à la mauvaise ligne (seul le message sur G cite `opt_g`).
**Correction :** détecter le générateur immobile. Si les poids de G après trois pas sont égaux à ceux de `make_gan(69)[0]`, écrire pour D : « l'écart vient du pas du générateur : le générateur n'a pas bougé et D a fait un pas de trop (ligne J : c'est opt_g qui fait le pas) ». Garder la piste actuelle pour les autres cas. Une seconde voie : comparer D après un pas à la référence suivie d'un `opt_d.step()` supplémentaire.

### 8. MINEUR · `projets/partie_5_classifieur_cifar10/depart/mp5_classifieur_cifar10.ipynb`, cellule de configuration et cellule de vérification MP5.8 (`tools/chapters/build_mp5.py:231`, et la cellule `checks_of("MP5.8")`)
**Extrait :** « MODEL_DIR = Path(tempfile.gettempdir()) / "cifarkit_rapide" if FAST_MODE else PROJECT   # where the final model goes » ; sortie du kit vierge : « ❌ MP5.8 : ton code lève NameError: name 'y_val' is not defined (la vérification de la cellule s'arrête là ; corrige, puis relance-la) »
**Problème :** en mode rapide, le modèle livré va dans un dossier temporaire **au nom fixe**, partagé par tous les notebooks de la machine : la solution, les essais précédents de l'apprenant, et tout autre utilisateur de `/tmp` sous Linux. Sur cette machine, `/tmp/cifarkit_rapide/model.json` date du 30 octobre (une exécution de la solution). J'ai copié le kit par `start_chapter.py CP5` et l'ai exécuté vierge (« Tout exécuter », mode rapide). La vérification MP5.8 a trouvé ce modèle étranger et l'a utilisé : faute de `X_val`, elle affiche un ❌ qui accuse « ton code ». Le §22 (2026-10-18) veut au contraire que « ce qui est encore au modèle s'affiche ⏳, pas ❌ » dans un kit. Pire, si l'étape MP5.5 de l'apprenant échoue après une exécution de la solution, MP5.8 valide `classify` et la ligne de commande **sur le modèle de la solution** (✅ trompeurs).
**Correction :** dans `build_mp5.py`, un dossier propre à la session : `MODEL_DIR = Path(tempfile.mkdtemp(prefix="cifarkit_rapide_")) if FAST_MODE else PROJECT`. Ou bien vider ce dossier dans la cellule de configuration. Dans la cellule MP5.8, tester `if final_ok and (MODEL_DIR / "model.json").exists():`, sinon `pending("MP5.8", "pas encore de modèle livré : l'étape MP5.5 l'écrit.")`. Reconstruire ensuite le kit et la solution.

### 9. MINEUR · `tools/chapters/build_mp5.py` (cellule `MP55_TEST`) et `projets/partie_5_classifieur_cifar10/depart/data.py` (`open_test`)
**Extrait :** « X_test, y_test = data.open_test(PROJECT, description) » (aucune condition de commit) ; `open_test` : « Every opening is logged in folder / "test_log.json" »
**Problème :** le coffre de MP5 s'ouvre au bon moment : mode complet seulement, et après toutes les vérifications. Mais rien n'exige un commit. La seule trace des ouvertures est `test_log.json`, un fichier non versionné au moment de l'ouverture. Supprimer `test_log.json` puis rouvrir le test pour un autre modèle ne laisse aucune trace, et aucun avertissement ne s'affiche. La règle posée par l'audit précédent pour MP2 et MP3 (§22, 2026-10-18 : « un test ouvert avant le commit n'est plus traçable ») n'a pas d'équivalent ici, et la grille note justement « le test ouvert une seule fois ».
**Correction :** reprendre `committed` et `commit_file` de `build_mp3.py`. Avant l'ouverture, si `test_log.json` existe, exiger qu'il soit commité et inchangé (sinon ⏳ « commite test_log.json avant de rouvrir le test »). Juste après l'ouverture, `commit_file(PROJECT / "test_log.json", "MP5: open the test set")`, avec le repli « ce dossier n'est pas dans un dépôt git » de MP3. MP6 n'utilise pas le test : rien à faire.

### 10. MINEUR · `projets/partie_5_classifieur_cifar10/README.md` (tableau des étapes) et `docs/syllabus/data/cp5.json` (`"hours": 10`)
**Extrait :** « Le mini-projet du checkpoint V, à faire après l'examen blanc (environ 10 h) » ; étapes : 60 + 60 + 90 + 90 + 90 + 60 + 60 + 60 min
**Problème :** la somme des étapes vaut 570 min (9 h 30), pas 10 h. Or la règle de `PROGRESS.md` (« Toujours valable ») dit : « une durée de mini-projet est la somme de ses étapes ». MP6 la respecte (600 min = 10 h). Le README du checkpoint (« le mini-projet (10 h) », total 13 h 30) et le tableau de bord (« ≈ 10 h ») héritent de l'écart.
**Correction :** soit `"hours": 9.5` (README : « environ 9 h 30 » ; README du checkpoint : « environ **13 h** … le mini-projet (9 h 30) »), soit 30 min de plus sur l'étape la plus chargée (MP5.8 Livraison, 90 min). Lancer ensuite `syllabus.py build` et ajouter une ligne d'écart.

### 11. MINEUR · titres des sujets contre les contrats (`checkpoints/partie_6/01_examen_sujet.md`, `checkpoints/partie_5/01_examen_sujet.md`)
**Extrait :** CP6.1 « Questions flash : dix réponses courtes sur la partie VI » (contrat : « … (valeurs, formes, vrai/faux) ») ; CP6.9 « Réparer un pas d'entraînement de GAN » (contrat et tableau de bord : « Réparer une boucle d'entraînement de GAN ») ; CP6.13 « Retour sur les parties IV et V : convolution et pas d'Adam » (contrat : « … taille de sortie d'une convolution et un pas d'Adam ») ; en-têtes de CP5.5 et CP5.6 sans le « (notebook, NumPy) » et le « (notebook, PyTorch) » du contrat (le tableau du sujet les a).
**Problème :** le §22 (2026-10-18) dit que « le titre complet est dans le sujet et le contrat » : seuls les corrigés et les copies abrègent. Pour CP6.9, le titre a changé de sens (« un pas » contre « une boucle »), et l'écart 61 dit pourtant « aucun écart d'ID, de type ou de points » sans mentionner les titres. `scripts/contract.py` relève ces écarts. CP3.1 et CP3.8 ont le même défaut, mais ils sont hors périmètre.
**Correction :** reprendre les titres du contrat dans les en-têtes et le tableau du sujet. Pour CP6.9, garder plutôt « Réparer un pas d'entraînement de GAN », qui est plus juste : le corriger dans `cp6.json`, `syllabus.py build`, et le noter dans l'écart 61.

### 12. MINEUR · `checkpoints/partie_5/05_synthese.md` §4 (le workflow PyTorch)
**Extrait :** « x01 = X_train / 255.0 » ; « mean, std = x01.mean(axis=(0, 2, 3)), x01.std(axis=(0, 2, 3))   # per channel, (N, C, H, W) » ; « train_tf = v2.Compose([v2.ToImage(), … ]) »
**Problème :** l'extrait se contredit. Les statistiques supposent des images NumPy `(N, C, H, W)`, mais `v2.ToImage()` lit un tableau NumPy en `(H, W, C)` : sur un tableau `(3, 32, 32)`, il rend `(32, 3, 32)` (vérifié avec torchvision 0.26). Dans le workbook, CIFAR-10 se charge en `(N, 32, 32, 3)` (`data.load_cifar10`, `wb.datasets`), et `channel_stats` de MP5 réduit sur `(N, H, W)`. Recopié de mémoire sur ces images, l'extrait donne des « statistiques par canal » calculées sur des lignes de pixels.
**Correction :** « x01 = X_train / 255.0                                       # uint8 images (N, H, W, C), as loaded » et « mean, std = x01.mean(axis=(0, 1, 2)), x01.std(axis=(0, 1, 2))   # per channel ».

### 13. MINEUR · `projets/partie_5_classifieur_cifar10/solution/README.md` (« Reproduire ») et `tools/chapters/build_mp5.py:16`
**Extrait :** « avec `FAST_MODE = False` pour les chiffres de ce README, quelques minutes sur un GPU de Colab, environ une heure et demie sur un CPU à deux cœurs » ; docstring : « on a CPU with 2 cores, about an hour and a half »
**Problème :** le cahier des charges (« environ une heure vingt sur un CPU à deux cœurs »), `PROGRESS.md` (ligne 29 : « ≈ 80 min sur CPU ici ») et le §22 (2026-10-30 : « durées du mode complet (≈ 1 h 20 …) ») disent 1 h 20. La correction de la session 52 n'a pas atteint le README de la solution.
**Correction :** « environ une heure vingt sur un CPU à deux cœurs », dans le README de la solution et la docstring de `build_mp5.py`.

### 14. MINEUR · `checkpoints/partie_6/03_examen_corrige.md` (tableau « Barème récapitulatif », ligne CP6.1)
**Extrait :** section CP6.1 : « **Remédiation** : 25.Q7, 25.Q12, 25.31, 26.1, … » ; tableau : « | CP6.1 | 2,5 | 25 à 29 | 25.Q7, 25.Q12, 26.1, 26.13, 27.4, 27.Q10, 28.1, 29.Q4, 29.Q2 | »
**Problème :** le tableau récapitulatif oublie 25.31, le 🔮 qui fait décoder des tirages $\mathcal{N}(\mathbf{0}, \mathbf{I})$ avec un AE puis un VAE (item b). Partout ailleurs, les deux listes sont identiques (`scripts/remed.py`). Tous les ID de remédiation de CP5 et CP6 existent et traitent la notion.
**Correction :** ajouter 25.31 après 25.Q12 dans la ligne CP6.1 du tableau.

### 15. MINEUR · `checkpoints/partie_5/02_examen_notebook.ipynb`, cellule « Bilan » (`build_cp5.py`) contre CP6
**Extrait :** CP5, exécution à vide avec `EXAM_OVER = True` : « Réponses vérifiées : 0 juste(s) sur 0 ; pas remplies : 38. » ; CP6 : « … pas remplies : 39. »
**Problème :** CP6 compte dans « pas remplies » les réponses des questions de code non écrites (CP6.8b et CP6.8c, `RESULTS[key] = wb.check(key, ..., quiet=True)`), une correction de la vérification indépendante de la session 61 (§22 : « les réponses pas remplies du code sont comptées »). CP5 ne l'a pas reçue. Ses 40 réponses (`study_notes`) deviennent 38, car CP5.5b et CP5.6b manquent tant que le code n'est pas écrit.
**Correction :** dans les branches `⏳` des cellules CP5.5 et CP5.6 de `build_cp5.py`, ajouter `RESULTS["CP5.5b"] = wb.check("CP5.5b", ..., quiet=True)` (et de même pour CP5.6b), comme dans `build_cp6.py`.

### 16. MINEUR · `projets/partie_5_classifieur_cifar10/solution/mp5_classifieur_cifar10.ipynb`, cellule `checks_of("MP5.5 test")` (`build_mp5.py`, `MP55_TEST`)
**Extrait :** « ✅ MP5.5 : toutes les vérifications des étapes précédentes passent : le test peut s'ouvrir. » puis, aussitôt : « ⏸️ MP5.5 : en mode rapide, le test reste fermé »
**Problème :** en mode rapide (le mode de l'apprenant sans GPU), les deux lignes se contredisent.
**Correction :** en mode rapide, écrire « toutes les vérifications des étapes précédentes passent : en mode complet, le test pourrait s'ouvrir. »

### 17. MAJEUR · `.gitignore:40` (`*.pt`) contre `projets/partie_5_classifieur_cifar10/README.md:51,116,122` et `solution/README.md`
**Extrait :** `.gitignore` : « *.pt » ; cahier des charges : « Le dossier `solution/` contient […] les résultats du mode complet (`results.json`, `runs.csv`, `model.pt`, `model.json`, les figures, `test_log.json`) » ; « enregistre ensuite ta copie et les fichiers produits (`results.json`, `runs.csv`, `model.pt`, `model.json`, `figures/`, `test_log.json`) dans ton dépôt » ; README de la solution : « Le modèle livré : `model.pt` (le `state_dict`, 300 Ko) » et « python -m cifarkit.predict photo.jpg --model . »
**Problème :** `git check-ignore -v` le montre : `projets/partie_5_classifieur_cifar10/solution/model.pt` et `mon_travail/projets/partie_5_classifieur_cifar10/model.pt` sont ignorés par la règle `*.pt`. Le `model.pt` de la solution n'a jamais été commité (`git ls-files` : seul `model.json` est versionné ; le fichier existe sur la machine de génération seulement). Trois conséquences :
- après un clone, en particulier sur Colab, la commande « Reproduire » de la solution (`python -m cifarkit.predict … --model .`) et `load_model(solution)` échouent ;
- le livrable 4 de l'apprenant (`model.pt`) est **silencieusement** écarté par `git add` : son portfolio poussé n'a pas de modèle livré, alors que la grille note « un modèle qu'on ne sait pas recharger » ;
- la vérification MP5.8 (« results.json, model.pt et model.json sont livrés ») passe en local et ne voit rien.

MP3 avait prévenu pour `model.joblib` (« le dépôt du workbook l'ignore ») ; MP5 ne dit rien.
**Correction :** options.
- (A, recommandée) Ajouter des exceptions à `.gitignore` : `!projets/partie_5_classifieur_cifar10/solution/model.pt` et `!mon_travail/projets/*/model.pt`, puis commiter le `model.pt` de la solution (300 Ko, sous le seuil des données lourdes). Le livrable et la solution redeviennent ce que le cahier annonce.
- (B) Garder l'exclusion, mais le dire : dans le cahier (« `model.pt` est ignoré par le `.gitignore` du workbook : `git add -f …/model.pt`, ou commite-le dans ton dépôt de portfolio ») ; dans le README de la solution (« `model.pt` n'est pas versionné : relance le mode complet pour le recréer ») ; et dans MP5.8, une vérification `git check-ignore` qui affiche ⚠️.

Une ligne au §22 dans les deux cas.

### 18. MAJEUR · `checkpoints/partie_6/02_examen_notebook.ipynb` et `checkpoints/partie_5/02_examen_notebook.ipynb`, partie B (`build_cp6.py`, `build_cp5.py`)
**Extrait :** CP6, cellule « Checks of gan_step_fixed (CP6.9) » : « def reference_step(G, D, real, opt_d, opt_g): """The GAN step of chapter 27: D on real and detached fakes, then G with the label 1 (non-saturating).""" … d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros) … g_loss = bce(D(fake), ones) … opt_g.step() » ; cellule CP6.8 : « def oracle_8(…): rec = float(F.binary_cross_entropy_with_logits(logits, x, reduction="sum")) / n ; kl = float(0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum()) / n » ; CP5, cellule CP5.6 : « def reference_fit(model, train_loader, val_loader, max_epochs, patience, path, lr=1e-3): """The contract of CP5.6 a), the oracle of these checks.""" » (la solution complète).
**Problème :** l'apprenant a ce notebook ouvert **pendant** l'examen : la partie A s'y fait, et les règles l'autorisent. La partie B, juste en dessous, n'est pas repliée (aucune métadonnée `source_hidden`). Elle contient en clair la solution complète des deux questions de code de CP6 et de CP5.6, ainsi que les trois lettres de CP6.9 a, avec leur correction. Au total, 3,5 points sur 20 dans chaque examen. Un débutant qui bloque en partie A fait défiler le notebook et tombe dessus. C'est la règle « un code fourni ne contient jamais la réponse » (`PROGRESS.md`, « Toujours valable », 18.22) appliquée à l'examen. Le hash de `wb.check` protège les réponses papier ; ici, la réponse se lit sans effort. CP4 (`oracle_step` de CP4.9) suit le même schéma, mais son énoncé donnait déjà les formules d'AdamW, si bien que la fuite y était faible. Les ch. 25 à 27 font écrire ces mêmes codes : leur valeur d'examen est perdue.
**Correction :** options.
- (A, recommandée) Déplacer les oracles hors du notebook, dans un module de `wb`, par exemple `src/wb/exams.py` avec `cp5_fit_reference`, `cp5_rnn_oracle`, `cp6_vae_loss_reference` et `cp6_gan_step_reference`, testés dans `tests/infra/`. La partie B les importe (`from wb.exams import cp6_gan_step_reference`) ; ses messages restent des pistes. C'est déjà le principe de `wb.synth` pour 18.22.
- (B) Sortir la partie B dans un second notebook, `02b_examen_verification.ipynb`, que `start_chapter.py CPn` copie aussi, à ouvrir après l'examen.
- (C) Au minimum, replier les cellules de vérification (`"jupyter": {"source_hidden": true}` dans les métadonnées, respecté par Jupyter et Colab), avec une ligne d'avertissement en tête de la partie B.

Une ligne au §22 et la même règle pour les checkpoints suivants (PF).

### 19. MINEUR · `checkpoints/partie_6/01_examen_sujet.md` (CP6.8) et la cellule des données du notebook (`build_cp6.py`)
**Extrait :** sujet : « les **logits** qu'un décodeur a rendus pour elles (`LOGITS_V`, même forme) » ; cellule : « LOGITS_V = 2.0 * torch.randn(16, 1, 28, 28, generator=_g8, dtype=torch.float64) » ; corrigé : « La reconstruction est grande parce que ces logits sont tirés au hasard : un décodeur entraîné descendrait vers 100 nats par image. »
**Problème :** l'énoncé présente des logits tirés au hasard comme ceux d'un décodeur. Pendant l'examen, l'apprenant obtient 829 nats par image alors que le ch. 25 lui a appris à attendre une centaine de nats. Il risque de croire son code faux et de « corriger » une reconstruction juste, par exemple en la moyennant sur les pixels (l'erreur que la question veut éviter). Le corrigé le reconnaît, mais trop tard.
**Correction :** « des **logits** tirés au hasard, comme ceux d'un décodeur pas encore entraîné (`LOGITS_V`, même forme) : la reconstruction sera bien plus grande que celle d'un VAE entraîné ». La même phrase va dans le notebook (`build_cp6.py`, énoncé de CP6.8).

### 20. MINEUR · barème de CP6.7 et de CP6.9 a (`checkpoints/partie_6/01_examen_sujet.md`, `03_examen_corrige.md`)
**Extrait :** corrigé de CP6.7 : « Un tiers de point par panneau (0,33 ; 1 point pour les trois). » (le sujet ne donne pas de points par panneau) ; CP6.9 a : « (0,2 par lettre juste, −0,1 par lettre en trop). »
**Problème :** CP5.7 donne ses points par item dans le sujet (« 0,25 point chacune »), et CP4.7 et CP5.4 notent un 🐛 à lettres par « −0,25 par lettre en trop, sans descendre sous 0 », avec une règle pour la ligne où l'erreur se déclare. CP6.9 a n'a ni plancher à 0, ni règle pour (I), alors que le corrigé cite justement « désigner I » parmi les erreurs fréquentes. Avec 0,33 par panneau, le maximum est 0,99.
**Correction :** dans le sujet, CP6.7 : « (une lettre par panneau, 1/3 de point chacune ; …) ». Dans le corrigé, CP6.9 a : « (0,2 par lettre juste, −0,1 par lettre en trop, sans descendre sous 0 ; (I) à la place de (C) désigne l'endroit où l'erreur se déclare, pas la ligne à corriger : 0,1 pour cette lettre, sans la pénalité d'une lettre en trop) ».

### 21. SUGGESTION · « Lire ta note », tranche 8 à 12 (`checkpoints/partie_5/03_examen_corrige.md`, `checkpoints/partie_6/03_examen_corrige.md`)
**Extrait :** CP6 : « refais […] le 🐛 27.16 et le 📦 27.12 » ; CP5 : « refais le 🔨 22.15 (`rnn_forward`), les 📦 23.18 et 23.25 […] et les ✏️ 21.3 à 21.5 »
**Problème :** « refais » suppose un exercice déjà fait. Or 27.16 n'est que dans le parcours code ; 23.18 et 23.25 ne sont pas dans le parcours maths ; 21.5 n'est que dans le parcours maths. Les questions d'examen, elles, sont dans les quatre parcours.
**Correction :** écrire « refais (ou fais, s'il n'est pas dans ton parcours) », ou choisir pour chaque tranche des exercices communs aux quatre parcours (par exemple 27.12 et 25.30 pour CP6, 22.15 et 23.6 pour CP5).

### 22. SUGGESTION · `checkpoints/partie_6/05_synthese.md` §2 et `checkpoints/partie_6/README.md`
**Extrait :** « ## 2. Fiche d'une page : les formules clés » (CP4 : « les 24 formules clés », CP5 : « les 22 formules clés »)
**Problème :** une petite dérive de format entre checkpoints. Le tableau a 18 lignes.
**Correction :** « ## 2. Fiche d'une page : les 18 formules clés », et la même chose dans le tableau du README.

### 23. SUGGESTION · `projets/partie_6_generatif/depart/genbench/train.py:52` (`save_config`) et l'étape MP6.1 (`build_mp6.py`)
**Extrait :** « Write the configuration of the experiments to the JSON file path, BEFORE they run: the keys of config, plus "date" (today […]) » ; cahier : « Il s'écrit **avant** les expériences, dans ton README (étape MP6.1), et ne change plus ensuite »
**Problème :** chaque « Tout exécuter » réécrit `config.json` avec la date du jour, et MP6.7 demande de relancer tout le notebook. La date finale est donc celle de la dernière exécution, pas celle du protocole, et rien ne prouve que le protocole précède les mesures. C'est la même question de traçabilité que le coffre de MP2 et MP3 (constat 9).
**Correction :** à l'étape MP6.1, proposer (sans l'exiger, faute de dépôt sur certaines installations) de commiter `README.md` (sections objectif et protocole) et `config.json`, avec `committed` et `commit_file` de `build_mp3.py`. Autre option : `save_config` garde la date existante quand le contenu n'a pas changé, en le disant dans sa docstring (le kit est publié : à décider avec le contrat de `genbench`).

## Bilan

**Ce qui est solide.**
- **Les réponses chiffrées.** Toutes celles des deux corrigés sont justes, recalculées en Python et avec PyTorch comme oracle (formes, paramètres, champ récepteur, LSTM, Fermi, Pipeline, GridSearchCV, gradients, VAE, Q-learning et SARSA, D*, Gram, Adam, mémoire des tables Q), de même que les valeurs des erreurs fréquentes (seule exception : le constat 5).
- **La partie B des notebooks d'examen.** Remplie par des solutions écrites autrement, elle donne 40/40 et 39/39, avec 14 et 13 propriétés de code sur 14 et 13. Elle accepte les formes naturelles : « faux », « (B) », « 16 x 4 », « J, C et G », « 1/2 », `retain_graph`, `inference_mode`, une boucle `while`. Les trois erreurs fréquentes de CP5.6 rejouées redonnent les valeurs du corrigé.
- **Les sujets.** Les barèmes totalisent 20 question par question, les durées (119 et 114 min) et les 10 % sont conformes au §17, et toutes les remédiations renvoient à des exercices existants et pertinents.
- **« Tout exécuter » à vide.** Les deux examens passent, avec `EXAM_OVER = False` puis `True` : aucune erreur, aucune réponse affichée.
- **Le reste.** Les cartes Mermaid se rendent (`mmdc`). Les `requires` sont écrits par les quatre parcours (`syllabus.py check` : 0 problème). Les kits copiés par `start_chapter.py` mettent leurs tests en ⏳. Les README des solutions concordent avec `results.json`, et le modèle livré de MP5, réévalué sur le test, redonne 84,99 %, les accuracies par classe et les confusions (213, 88, 85) du README. Les tests des deux solutions sont verts (19 et 19).

**Les trois risques principaux.**
1. **Les solutions des questions de code** de CP5 et CP6 sont lisibles en clair dans la partie B du notebook que l'apprenant a ouvert pendant l'examen (constat 18).
2. **CP6 s'écarte du §17 et de CP4 et CP5** : un 📦 à l'examen, des ★ trop basses pour une preuve et un 🐛 à trois erreurs, et pas de `notes` pour justifier les types absents (constats 1 et 2).
3. **Le modèle livré de MP5 (`model.pt`) est ignoré par git**, dans la solution comme dans la copie de l'apprenant : le livrable disparaît sans bruit (constat 17). S'y ajoutent un dossier temporaire partagé qui fait valider un modèle étranger (constat 8) et un coffre du test sans commit (constat 9).

| Sévérité | Nombre |
|---|---|
| MAJEUR | 4 (1, 2, 17, 18) |
| MINEUR | 16 (3 à 16, 19, 20) |
| SUGGESTION | 3 (21 à 23) |
| **Total** | **23** |

`git -C /home/claude/workbookia status --short` : vide à la fin de l'audit. Tout ce qui a été exécuté l'a été dans `COPIE/`.
