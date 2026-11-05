# Index des notions

> **Généré** par `python tools/syllabus.py build` (`tools/build_index.py`) à partir du glossaire (`annexes/glossaire.md`), du contrat de chaque chapitre (`docs/syllabus/data/*.json` : notions enseignées, concepts, notions mobilisées, exercices) et des titres des fiches : on ne l'édite pas à la main.

**1570 notions**, et **1618 renvois** depuis l'autre langue ou un synonyme, dans l'ordre alphabétique (accents ignorés). Pour chaque notion :

- **enseignée** : le chapitre qui l'introduit (colonne « Ch. » du glossaire), puis ceux qui l'enseignent (leurs « notions enseignées »), chacun avec un lien vers la section de sa fiche qui la présente : un titre qui la nomme, sinon sa première définition (en gras ou suivie de sa glose), sinon la section qui en parle le plus ;
- **exercices** : ceux qui couvrent cette section du livre, puis ceux dont le titre la nomme (10 au plus, le nombre des autres entre parenthèses) ;
- **approfondie** : les chapitres dont une notion enseignée la contient ; **réutilisée** : les chapitres qui la mobilisent comme prérequis.

Entre parenthèses et en italique, l'autre langue. Une ligne « *terme* → voir … » renvoie de l'autre langue (le plus souvent l'anglais) vers le terme retenu (BIBLE §5) ; une ligne « terme → voir … », d'un synonyme vers l'entrée qui le contient. Les définitions sont dans le [glossaire](glossaire.md), les formules dans le [formulaire](formulaire.md). La recherche est automatique : un lien qui tombe à côté se signale avec P6.

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [X](#x) · [Y](#y) · [Z](#z) · [0-9](#symboles)

## A

- *à quelques exemples* → voir [few-shot](#f)
- *A/B test* → voir [test A/B](#t)
- **abandon du label (guidance sans classifieur)** (*label dropout, condition dropout*) — enseignée : [ch. B5, 205.7 · Conditionner : label de classe et guidance sans classifieur][s565] · exercices : B5.1, B5.26–27, B5.E4, B5.Q9
- **abduction** (*inference to the best explanation*) — enseignée : [ch. 11, 11.6.1 · Sherlock Holmes, « maître de la déduction »][s253] · exercices : 11.14, 11.Q12
- **ablation** (*ablation study*) — enseignée : [ch. 20, Au-delà du livre : régulariser un réseau][s227] · exercices : 20.29, B3.25, B8.15, PF.7
- *ablation study* → voir [ablation](#a)
- absence → voir [précision locale, absence, cohérence](#p)
- *absolute value* → voir [valeur absolue](#v)
- **abstention** — enseignée : [ch. B4, Les hallucinations][s546] · exercices : B4.9–10, B4.24, B4.E1, B4.E5, B4.Q7, B4.Q12
- **accès contrôlé (dépôt)** (*gated, gated repository*) — enseignée : [ch. B2, Les tokenizers des modèles, lus dans leur `tokenizer.json`][s476] · exercices : B2.13, B2.19, B2.30, B2.E2, B2.Q7
- **accord top-1** (*top-1 agreement*) — enseignée : [ch. B1, L'oubli catastrophique][s451] · exercices : B1.6, B1.9–10, B1.16, B1.22–23, B1.E2, B1.Q8–Q9, B1.R1
- *accumulated local effects* → voir [courbe ALE](#c)
- **accumulation des gradients** (*gradient accumulation*) — enseignée : [ch. 18][s311]
- **accuracy** (*exactitude, taux de bonnes réponses*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s29] ; [ch. 3, 3.7.5 · Accuracy][s30] · exercices : 1.14, 1.18, 1.20, 1.25, 1.E2, 1.Q6, 3.2, 3.16, 3.E1, 3.Q9 (+ 19) · réutilisée : ch. 8, 10, 20, 23
- *accuracy équilibrée* → voir [balanced accuracy](#b)
- **accuracy top-5** (*top-5 accuracy*) — enseignée : [ch. 20][s332]
- *ACP* → voir [PCA](#p)
- acteur → voir [acteur-critique, acteur, critique](#a)
- **acteur-critique et GAE** — enseignée : [ch. B8, 208.4 · Acteur-critique : fonction de valeur, erreur TD, avantage, GAE][s409] · exercices : B8.4, B8.18, B8.20, B8.Q7
- **acteur-critique, acteur, critique** (*actor-critic, actor, critic*) — enseignée : [ch. B8, 208.4 · Acteur-critique : fonction de valeur, erreur TD, avantage, GAE][s409] · exercices : 20.12, B8.4, B8.18, B8.20, B8.Q7, B8.Q11 · approfondie : ch. 5, 19
- *action* → voir [agent, environnement, action, récompense](#a), [workflow, job, étape, action](#w)
- action → voir [agent, environnement, action, récompense](#a), [workflow, job, étape, action](#w)
- *action mask* → voir [masque d'actions](#m)
- *activation function* → voir [fonction d'activation](#f)
- *activation map* → voir [carte d'activation](#c)
- *activation maximization* → voir [maximisation d'activation](#m)
- **activations linéaires par morceaux (ReLU et variantes, maxout)** — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s299] · exercices : 17.2, 17.5, 17.12, 17.18, 17.20, 17.23–24, 17.Q6–Q7, 17.R1 (+ 2)
- **activations lisses (softplus, ELU, sigmoïde, tanh, SiLU, GELU)** — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s85] · exercices : 17.2, 17.4, 17.8, 17.10, 17.13–14, 17.17–18, 17.Q8–Q9 (+ 6)
- *actor* → voir [acteur-critique, acteur, critique](#a)
- *actor-critic* → voir [acteur-critique, acteur, critique](#a)
- **AdaBoost** (*adaptive boosting*) — enseignée : [ch. 14, 14.7 · Le boosting][s275] · exercices : 14.2, 14.4, 14.6–8, 14.10, 14.Q8–Q11 (+ 13)
- **Adadelta** — enseignée : [ch. 19, 19.5.4 · Adadelta et RMSprop][s322] · exercices : 19.4, 19.8, 19.13, 19.17, 19.Q11
- **Adagrad** (*adaptive gradient*) — enseignée : [ch. 19, 19.5.3 · Adagrad][s321] · exercices : 19.4, 19.13, 19.17, 19.23, 19.Q11
- **AdaIN** (*adaptive instance normalization*) — enseignée : [ch. 28, 28.4.6 · Discussion][s438] · exercices : 28.3, 28.8–9, 28.24–25, 28.E4, 28.Q7
- **adaLN-Zero** (*adaptive layer norm, zero-initialized*) — enseignée : [ch. B5, Les transformers de diffusion][s570] · exercices : B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11
- **Adam** (*adaptive moment estimation*) — enseignée : [ch. 19, 19.5.5 · Adam][s323] · exercices : 19.5–6, 19.8–9, 19.12–13, 19.18, 19.22, 19.24, 19.Q12 (+ 5) · réutilisée : ch. 21, 22, 23, 24, 25, 27, B8
- **AdamW** — enseignée : [ch. 19, Au-delà du livre : AdamW, le weight decay découplé][s324] · exercices : 19.6, 19.18, 19.E2, 20.28, 20.R1, CP4.3, CP4.6, CP4.9, B3.20 · réutilisée : ch. 20, B1, B3, B4, B5
- **adaptateur** (*adapter, wrapper*) — enseignée : [ch. 15, Au-delà du livre : brancher ton propre estimateur][s290]
- **adaptateur (PEFT)** (*adapter*) — enseignée : [ch. B4, 204.8 · Fine-tuning : complet ou PEFT ; LoRA (rang, alpha, modules…][s539] · exercices : B4.3, B4.6, B4.11, B4.25–27, B4.E2, B4.Q10–Q11, B4.R2 (+ 1) · réutilisée : ch. 15
- *adapter* → voir [adaptateur](#a), [adaptateur (PEFT)](#a)
- *adaptive boosting* → voir [AdaBoost](#a)
- *adaptive gradient* → voir [Adagrad](#a)
- *adaptive instance normalization* → voir [AdaIN](#a)
- *adaptive layer norm* → voir [adaLN-Zero](#a)
- *adaptive moment estimation* → voir [Adam](#a)
- *add-one smoothing* → voir [lissage de Laplace](#l)
- **`add_generation_prompt`** — enseignée : [ch. B4, Rôles et chat template][s521] · exercices : B4.4, B4.12–14, B4.Q4–Q5, B4.R1
- *additive attention* → voir [attention additive](#a)
- *additive smoothing* → voir [lissage additif](#l)
- additivité → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *additivity* → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *adjusted Rand index* → voir [indice de Rand ajusté (ARI)](#i)
- *advantage* → voir [avantage](#a)
- *adversarial diffusion distillation* → voir [distillation adversariale](#d)
- *adversarial example* → voir [exemple adversarial](#e)
- *adversarial training* → voir [entraînement adversarial](#e)
- **affine par morceaux** (*piecewise linear*) — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s299] · exercices : 17.2, 17.5, 17.12, 17.18, 17.20, 17.23–24, 17.Q6–Q7, 17.R1 (+ 2)
- **affirmation du conséquent** (*affirming the consequent*) — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5, 11.15, 11.Q7
- *affirming the consequent* → voir [affirmation du conséquent](#a)
- *agent* → voir [agent, environnement, action, récompense](#a)
- **agent libre, déclenché** (*free-running, triggered agent*) — enseignée : [ch. 26, 26.3.4 · Des variantes de la version simple][s403] · exercices : 26.21, 26.35, 26.Q4
- **agent, environnement, action, récompense** (*agent, environment, action, reward*) — enseignée : [ch. 1, 1.6 · Apprentissage par renforcement][s46] · exercices : 1.22, 1.E3, 1.Q9, 11.24, 20.10, 23.12, 26.Q1–Q4 (+ 20) · approfondie : ch. 26, B8 · réutilisée : ch. B8
- **agentivité excessive** (*excessive agency*) — enseignée : [ch. B4, L'injection de prompt][s543] · exercices : B4.9–10, B4.24, B4.E1, B4.E5, B4.Q7, B4.Q12
- *AI Act (règlement 2024/1689)* → voir [règlement européen sur l'IA](#r)
- *AI winter* → voir [hiver de l'IA](#h)
- **ajustement bayésien d'une droite** — enseignée : [ch. 9, 9.7 · Ajuster une droite avec la règle de Bayes][s233] · exercices : 9.2, 9.7, 9.26–27, 9.Q11
- *ajustement croisé* → voir [cross fitting](#c)
- **algorithme de Lloyd** (*Lloyd's algorithm*) — enseignée : [ch. 7, Sans labels : k-means][s131] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 9)
- **algorithme de Viterbi** (*Viterbi algorithm*) — enseignée : [ch. B2, Unigram : partir de trop et élaguer][s471] · exercices : B2.13, B2.19, B2.30, B2.E2, B2.Q7
- **algorithme EM** (*expectation-maximization*) — enseignée : [ch. B2, Unigram : partir de trop et élaguer][s471] · exercices : B2.13, B2.19, B2.30, B2.E2, B2.Q7
- **algorithme en ligne, hors ligne** (*online, offline algorithm*) — enseignée : [ch. 19, 19.4.1 · La descente de gradient par batch][s7] · exercices : 19.25, 19.Q8
- **algorithme glouton** (*greedy algorithm*) — enseignée : [ch. 13][s274]
- **algorithme paresseux** (*lazy learner*) — enseignée : [ch. 13, 13.3 · Les k plus proches voisins][s211] · exercices : 13.1, 13.9, 13.13–15, 13.34, 13.36, 13.E1, 13.Q3–Q4
- **algorithme pocket** (*pocket algorithm*) — enseignée : [ch. 10][s245]
- **algorithmes déterministes** (*deterministic algorithms*) — enseignée : [ch. B7][s614] · exercices : B7.13
- **ALiBi** (*attention with linear biases*) — enseignée : [ch. B3, Allonger le contexte][s518] · exercices : B3.4, B3.7, B3.11, B3.15, B3.25, B3.E5, B3.Q7
- **alignement (traduction)** (*alignment*) — enseignée : [ch. B3, 203.1 · Des RNN à l'attention : goulot de l'encodeur-décodeur et…][s491] · exercices : B3.Q1, B3.R2
- **alignement, RLHF** (*alignment, reinforcement learning from human feedback*) — enseignée : [ch. 1, 1.5 · Générateurs][s118] ; [ch. B8, 208.7 · RL et modèles de langue : RLHF, modèle de récompense, DPO,…][s126] · exercices : 1.24, 1.E4, 1.Q9, 1.Q11, B8.6, B8.10, B8.25–26, B8.Q11–Q12 (+ 2)
- *alignment* → voir [alignement, RLHF](#a), [alignement (traduction)](#a)
- **amorçage** (*bootstrapping, RL*) — enseignée : [ch. 26, 26.6.1 · Les valeurs Q et leur mise à jour][s406] · exercices : 26.3, 26.8, 26.20, 26.Q8, B8.Q2
- **amorce** (*seed*) — enseignée : [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s203] · exercices : 22.1, 22.5, 22.8, 22.12, 22.20, 22.24–27, 22.Q12 (+ 2)
- **analyse d'erreurs** — enseignée : [ch. 23, 23.8.3 · Analyser l'historique][s374] · exercices : 23.6, 23.19, 23.27, 23.E4, PF.8
- **analyse d'impact sur les droits fondamentaux (FRIA)** (*fundamental rights impact assessment*) — enseignée : [ch. B6, L'AI Act][s602] · exercices : B6.8–9, B6.28, B6.Q12
- **analyse de sensibilité** (*sensitivity analysis*) — enseignée : [ch. 4][s176]
- **analyse des erreurs** (*error analysis*) — enseignée : [ch. 23, 23.8.1 · Lire la sortie][s375] · exercices : 23.5, 23.19–20
- *analyse en composantes principales* → voir [PCA](#p)
- **analyse exploratoire** (*exploratory data analysis, EDA*) — enseignée : [ch. 2, 2.9 · Le quartet d'Anscombe][s149] · exercices : 2.12, 2.30, 2.32, 2.E4, 2.Q12
- **analyse intersectionnelle** (*intersectional analysis*) — enseignée : [ch. B6, Cas célèbres][s595] · exercices : B6.9–10, B6.24, B6.26, B6.E3, B6.E5, B6.Q10
- *ancestor* → voir [parent, enfant ; ancêtre, descendant](#p)
- *ancestral sampling* → voir [échantillonnage ancestral](#e)
- ancêtre → voir [parent, enfant ; ancêtre, descendant](#p)
- *anchor* → voir [ancre, pixel focal](#a)
- **ancre, pixel focal** (*anchor, focal pixel*) — enseignée : [ch. 21, 21.2.4 · Le champ récepteur local][s339] · exercices : 21.Q2
- *ANN* → voir [recherche approchée des plus proches voisins](#r)
- **annotation de type** (*type hint*) — enseignée : [ch. 0A, 100.5.5 · Docstrings au format NumPy et annotations de type][s61] · exercices : 0A.61, 0A.Q7
- **annulation catastrophique** (*catastrophic cancellation*) — enseignée : [ch. 5, Au-delà du livre (2) : les arrondis et le choix du pas][s183]
- *anomaly detection* → voir [détection d'anomalies](#d)
- **ANS** (*asymmetric numeral systems*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s202]
- *Anscombe's quartet* → voir [quartet d'Anscombe, Datasaurus](#q)
- AP → voir [average precision (AP)](#a)
- **API de prédiction** — enseignée : [ch. B7, En ligne ou par batches, et une API HTTP][s621] · exercices : B7.3, B7.5, B7.16–19, B7.Q8–Q9, B7.R2–R3 (+ 2)
- **API fonctionnelle** (*functional API*) — enseignée : [ch. 24, 24.6 · L'API fonctionnelle][s334] · exercices : 24.10, 24.28, 24.Q12
- **API gymnasium (reset, step, terminated, truncated)** — enseignée : [ch. 26, 26.3 · La structure du RL][s402] · exercices : 26.13, 26.Q3
- **API HTTP, route** (*HTTP API, endpoint*) — enseignée : [ch. B7, 207.7 · Empaqueter et servir : artefact de modèle, inférence par…][s620] · exercices : B7.3, B7.5, B7.16–19, B7.Q8–Q9, B7.R2–R3 (+ 3)
- **API scikit-learn** — enseignée : [ch. 15, 15.2 · scikit-learn, NumPy et SciPy][s17] ; [ch. 29, 29.3.1 · scikit-learn][s691] · exercices : 15.9, 15.Q1, 29.5, 29.7, 29.Q5 · réutilisée : ch. 16
- **aplatissement** (*flatten*) — enseignée : [ch. 20, 20.4.8 · Les autres couches utilitaires][s330] · exercices : 20.1, 20.7, 20.14, 20.17, 20.20, 20.30, 20.E2, 20.Q9
- *appareil* → voir [device](#d)
- *appeal to coincidence* → voir [induction paresseuse](#i)
- **apprenant faible, apprenant fort** (*weak learner, strong learner*) — enseignée : [ch. 14, 14.7 · Le boosting][s275] · exercices : 14.2, 14.4, 14.6–8, 14.10, 14.Q8–Q11 (+ 12)
- apprenant fort → voir [apprenant faible, apprenant fort](#a)
- **apprentissage auto-supervisé** (*self-supervised learning*) — enseignée : [ch. 1, 1.5 · Générateurs][s118] ; [ch. 25, 25.3 · Le plus simple des autoencodeurs][s120] · exercices : 1.24, 1.E4, 1.Q9, 1.Q11, 25.1, 25.9, 25.14, 25.Q4–Q5, 25.R1 (+ 1)
- *apprentissage automatique* → voir [machine learning (ML)](#m)
- **apprentissage contrastif** (*contrastive learning*) — enseignée : [ch. B1, La courbe d'apprentissage selon la taille du train][s457] · exercices : B1.20–21, B1.25, B1.28, B1.E1–E2, B1.Q8
- *apprentissage des features* → voir [feature learning](#f)
- **apprentissage en contexte** (*in-context learning*) — enseignée : [ch. B4, Zéro-shot et few-shot][s525] · exercices : B4.15–16, B4.E1, B4.Q6
- **apprentissage hiérarchique** (*hierarchical learning*) — enseignée : [ch. 20, 20.8 · Interpréter les résultats][s326] · exercices : 20.10–11, 20.26, 20.Q12
- **apprentissage non supervisé** (*unsupervised learning*) — enseignée : [ch. 1, 1.4 · Apprentissage non supervisé][s45] · exercices : 1.21, 1.E3, 1.Q8
- **apprentissage par renforcement** (*reinforcement learning*) — enseignée : [ch. 1, 1.6 · Apprentissage par renforcement][s46] · exercices : 1.22, 1.E3, 1.Q9, CP6.7
- *apprentissage par transfert* → voir [transfer learning](#t)
- *apprentissage profond* → voir [deep learning](#d)
- **apprentissage semi-supervisé** (*semi-supervised learning*) — enseignée : [ch. 1, 1.3 · Apprentissage supervisé][s44] · exercices : 1.E3, 1.Q7
- **apprentissage supervisé** (*supervised learning*) — enseignée : [ch. 1, 1.3 · Apprentissage supervisé][s44] · exercices : 1.E3, 1.Q7 · réutilisée : ch. 7
- *approximate KL* → voir [KL approchée](#k)
- *approximate nearest neighbor search* → voir [recherche approchée des plus proches voisins](#r)
- **approximation de fonction** (*function approximation*) — enseignée : [ch. B8, 208.1 · Des tables aux réseaux : approximation de fonction, états…][s636] · exercices : B8.12, B8.Q1–Q2
- **arbre de décision** (*decision tree*) — enseignée : [ch. 1, 1.1.2 · Les systèmes experts et leurs limites][s110] ; [ch. 13, 13.5 · Les arbres de décision][s130] · exercices : 1.15, 1.18, 1.25, 1.Q2, 13.11, 13.24, 13.26, 13.E2, 13.Q7
- **arbre de régression** — enseignée : [ch. 13, 13.5 · Les arbres de décision][s130] · exercices : 13.11, 13.24, 13.26, 13.Q7
- arbre des fréquences naturelles → voir [fréquences naturelles, arbre des fréquences naturelles](#f)
- **architecture d'un réseau profond** — enseignée : [ch. 20, 20.7 · Construire un réseau profond][s702] · exercices : 20.9, 20.17, 20.19, 20.23–25, 20.27–28, 20.31, 20.Q11 (+ 1)
- **architectures BERT/GPT/T5** — enseignée : [ch. B3, 203.7 · Familles : encodeur (BERT), décodeur (GPT),…][s728] · exercices : B3.5, B3.11, B3.19, B3.E3, B3.Q5, B3.Q9
- *area under the curve* → voir [AUC](#a)
- **argparse et logging** — enseignée : [ch. B7, 207.2 · Scripts et ligne de commande : argparse, point d'entrée,…][s606] · exercices : B7.10, B7.25, B7.E1, B7.Q3
- ***args, **kwargs, arguments keyword-only** — enseignée : [ch. 0A, 100.5.2 · Arguments positionnels et nommés, `*args`, `**kwargs`,…][s669] · exercices : 0A.6, 0A.40, 0A.Q6 · réutilisée : ch. 24
- *argument* → voir [fonction, paramètre, argument](#f)
- argument → voir [fonction, paramètre, argument](#f)
- ARI → voir [indice de Rand ajusté (ARI)](#i)
- *arithmetic coding* → voir [codage arithmétique](#c)
- **array** (*tableau NumPy*) — enseignée : [ch. 0A, 100.8.1 · Créer des arrays : `dtype`, `shape`, `ndim`][s68] · exercices : 0A.9, 0A.27, 0A.Q10, 0B.R2, 23.Q5 · réutilisée : ch. 0B, 16, 20
- **artefact** (*artifact*) — enseignée : [ch. B7, 207.5 · Suivi d'expériences : runs, métriques, artefacts, comparaison][s616] · exercices : B7.1–2, B7.14, B7.17, B7.25, B7.E2, B7.Q8, B7.R1
- **artefact de modèle** (*model artifact*) — enseignée : [ch. B7, 207.7 · Empaqueter et servir : artefact de modèle, inférence par…][s620] · exercices : B7.3, B7.5, B7.16–19, B7.Q8–Q9, B7.R2–R3 (+ 2)
- **artefacts en damier** (*checkerboard artifacts*) — enseignée : [ch. 21, 21.8 · La convolution transposée][s343] · exercices : 21.3, 21.6, 21.30, 21.Q9
- *artifact* → voir [artefact](#a)
- **astuce de la log-dérivée** (*log-derivative trick*) — enseignée : [ch. B8, Apprendre la politique elle-même][s641] · exercices : B8.3, B8.7, B8.16–17, B8.E1, B8.Q5–Q6, B8.Q9, B8.R3
- **astuce de reparamétrisation** (*reparameterization trick*) — enseignée : [ch. 25, 25.9.2 · La structure d'un VAE][s399] · exercices : 25.5, 25.13, 25.28, 25.30, 25.38, 25.E3, 25.Q12, 25.R3, B5.R3
- **astuce du biais** (*bias trick*) — enseignée : [ch. 10, 10.3.3 · Les neurones modernes][s40] ; [ch. 18, 18.2.1 · Apprendre en tâtonnant][s246] · exercices : 10.1–2, 10.5, 10.12, 10.14–15, 10.20, 10.Q8–Q9, 10.R1 (+ 7) · réutilisée : ch. 16
- **astuce du noyau** (*kernel trick*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s244] · exercices : 13.4–5, 13.10, 13.16–18, 13.36, 13.Q5–Q6, CP3.9
- *asymmetric numeral systems* → voir [ANS](#a)
- asynchrone → voir [synchrone, asynchrone](#s)
- *asynchronous* → voir [synchrone, asynchrone](#s)
- *atomic write* → voir [écriture atomique](#e)
- *atrous convolution* → voir [dilatation](#d)
- **attaque adversariale FGSM** — enseignée : [ch. 21, Au-delà du livre : l'attaque FGSM][s352]
- **attaque ciblée, non ciblée** (*targeted, untargeted attack*) — enseignée : [ch. 21, 21.10 · Les adversaires][s351] · exercices : 21.10, 21.29, 21.E4, 21.Q11
- **attention additive** (*additive attention*) — enseignée : [ch. B3, L'idée de Bahdanau : relire la source à chaque pas][s493] · exercices : B3.Q1, B3.R2
- **attention croisée** (*cross-attention*) — enseignée : [ch. B3, 203.3 · Masques : causal, padding ; auto-attention et attention…][s496] · exercices : B3.2, B3.11–12, B3.17–18, B3.29, B3.Q4–Q5
- *attention head* → voir [tête d'attention](#t)
- *attention map* → voir [carte d'attention](#c)
- *attention mask* → voir [masque d'attention](#m)
- **attention multi-têtes (MHA)** (*multi-head attention*) — enseignée : [ch. B3, 203.4 · Attention multi-têtes : projections, découpage en têtes,…][s500] · exercices : B3.3, B3.11, B3.14, B3.18, B3.24, B3.E1, B3.Q6
- **attention par produit scalaire** — enseignée : [ch. B3, 203.2 · Attention par produit scalaire : requêtes, clés, valeurs,…][s494] · exercices : B3.1, B3.6, B3.9, B3.11–13, B3.18, B3.E1, B3.Q2–Q3
- **attention par produit scalaire mis à l'échelle** (*scaled dot-product attention*) — enseignée : [ch. B3, La formule][s495] · exercices : B3.1, B3.6, B3.9, B3.11–13, B3.18, B3.E1, B3.Q2–Q3
- *attention with linear biases* → voir [ALiBi](#a)
- **atténuation des biais** — enseignée : [ch. B6, D'où vient un biais][s594] · exercices : B6.9–10, B6.24, B6.26, B6.E3, B6.E5, B6.Q10
- attribut → voir [classe, objet, méthode, attribut](#c)
- **attribut appris** (*fitted attribute*) — enseignée : [ch. 15, 15.4.2 · Apprendre avec fit][s286] · exercices : 15.9, 15.Q4
- **attribut sensible** (*sensitive attribute, protected attribute*) — enseignée : [ch. B6, D'où vient un biais][s594] · exercices : B6.9–10, B6.24, B6.26, B6.E3, B6.E5, B6.Q10
- *attribute* → voir [classe, objet, méthode, attribut](#c)
- **attribution du crédit** (*credit assignment*) — enseignée : [ch. 26, 26.3.5 · La vue d'ensemble][s404] · exercices : 26.10, 26.26, 26.Q5
- **AUC** (*area under the curve, ROC*) — enseignée : [ch. 3, Au-delà du livre (1) : la courbe ROC et l'AUC][s167] · exercices : 3.24, 3.E3 · réutilisée : ch. B6
- **audit de qualité** — enseignée : [ch. 29, Au-delà du livre : auditer un dataset][s439] · exercices : PF.3
- **augmentation de données** (*data augmentation*) — enseignée : [ch. 9, Pénalité sur les poids, dropout, batchnorm][s20] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 5) · réutilisée : ch. 25, B1
- **auto-apprentissage** (*self-play*) — enseignée : [ch. 26, 26.10 · Deux applications][s414] · exercices : 26.37, 26.E5, 26.Q12, 27.R1
- **auto-attention** (*self-attention*) — enseignée : [ch. B3, 203.3 · Masques : causal, padding ; auto-attention et attention…][s496] · exercices : B3.2, B3.11–12, B3.17–18, B3.29, B3.E2, B3.Q4–Q5
- **auto-cohérence** (*self-consistency*) — enseignée : [ch. B4, Voter : l'auto-cohérence][s529] · exercices : B4.15–16, B4.E1, B4.Q6
- *autodiff* → voir [différentiation automatique](#d)
- *autoencoder* → voir [autoencodeur](#a)
- **autoencodeur** (*autoencoder*) — enseignée : [ch. 25, 25.3 · Le plus simple des autoencodeurs][s120] · exercices : 25.1, 25.9, 25.14–15, 25.Q2, 25.Q4–Q5, 25.Q7, 25.Q9, 25.R1 (+ 12)
- **autoencodeur convolutif** — enseignée : [ch. 25, 25.7 · Autoencodeurs convolutifs][s331] · exercices : 25.3, 25.12, 25.24, 25.Q10
- **autoencodeur débruiteur** (*denoising autoencoder*) — enseignée : [ch. 25, 25.3 · Le plus simple des autoencodeurs][s120] · exercices : 25.1, 25.9, 25.14–15, 25.26, 25.Q4–Q5, 25.R1
- **autoencodeur linéaire et ACP** — enseignée : [ch. 25, 25.6 · Discussion : l'autoencodeur et l'ACP][s720] · exercices : 25.8, 25.23, 25.E1, 25.Q9
- **autoencodeur parcimonieux** (*sparse autoencoder*) — enseignée : [ch. B6, Global ou local][s575] · exercices : B6.Q1–Q2
- **autograd (aperçu)** — enseignée : [ch. 5, Au-delà du livre (3) : la différentiation automatique][s188] · exercices : 5.21, 20.15 · approfondie : ch. 20 · réutilisée : ch. 21, 23, 25, 27, 28, B8
- **autograd PyTorch** — enseignée : [ch. 20][s332] · réutilisée : ch. 27
- *automatic differentiation* → voir [différentiation automatique](#d)
- *automation bias* → voir [biais d'automatisation](#b)
- *autoreset* → voir [autoreset (next-step, same-step)](#a)
- **autoreset (next-step, same-step)** (*autoreset*) — enseignée : [ch. B8, Les environnements vectorisés][s646] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- **avantage** (*advantage*) — enseignée : [ch. B8, 208.4 · Acteur-critique : fonction de valeur, erreur TD, avantage, GAE][s409] · exercices : B8.4, B8.18, B8.20, B8.E1, B8.Q7
- **average precision (AP)** (*précision moyenne*) — enseignée : [ch. 3, Au-delà du livre (2) : la courbe precision-recall et l'average…][s168] · exercices : 3.26
- *averaged perceptron* → voir [perceptron moyenné](#p)
- **avertissement** (*warning*) — enseignée : [ch. 23, 23.1.3 · Les avertissements de Python][s368] · exercices : CP3.1, 23.12, 23.Q2
- **axe** (*axis*) — enseignée : [ch. 0A, 100.8.5 · Réductions par axe, tri, `argmax`, `unique`][s70] · exercices : 0A.7, 0A.51, 0A.56, 0A.64, 0A.67, 0A.Q11, 3.Q7, 12.7–8, 12.22 (+ 2) · réutilisée : ch. 2, 7, B3
- *axis* → voir [axe](#a)

## B

- **backbone** (*tronc*) — enseignée : [ch. B1, Backbone et tête][s444] · exercices : B1.7–8, B1.10, B1.16, B1.Q1–Q2, B1.R1, B1.R3
- *backend* → voir [backend (de Keras)](#b)
- **backend (de Keras)** (*backend*) — enseignée : [ch. 23, 23.3.3 · Installer et configurer la bibliothèque][s370] · exercices : 23.12, 23.E5, 23.Q5, 23.Q12
- *background* → voir [fond](#f)
- *backprop* → voir [rétropropagation](#r)
- *backpropagation* → voir [rétropropagation](#r)
- *backpropagation through time* → voir [rétropropagation dans le temps (BPTT)](#r)
- *backward pass* → voir [passe avant, passe arrière](#p)
- *bag of words* → voir [sac de mots](#s)
- **bagging** (*bootstrap aggregating*) — enseignée : [ch. 14, 14.4 · Le bagging][s146] · exercices : 14.3, 14.8, 14.12, 14.16–17, 14.E1, 14.Q4–Q5, 14.Q10
- **balanced accuracy** (*accuracy équilibrée*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s161] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11
- **banc d'essai** (*benchmark*) — enseignée : [mini-projet 6][s667]
- **bandit contextuel** (*contextual bandit*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s121] · exercices : 11.2, 11.7–8, 11.10–11, 11.13, 11.19–21, 11.R1 (+ 8)
- **bandit manchot** (*multi-armed bandit*) — enseignée : [ch. 1][s111] ; [ch. 11, Au-delà du livre : le bandit manchot][s121] · exercices : 11.2, 11.7–8, 11.10–11, 11.13, 11.19–21, 11.R1 (+ 8)
- **base de données vectorielle** (*vector database*) — enseignée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s214] ; [ch. B4, Les bases de données vectorielles][s217] · exercices : 7.5–6, 7.19–21, 7.E4, 7.Q11, B4.5, B4.Q9, B4.R3 (+ 5)
- *base model* → voir [modèle de base](#m)
- *base rate* → voir [prévalence, taux de base](#p)
- *base rate fallacy* → voir [erreur du taux de base](#e)
- *base value* → voir [valeur de base](#v)
- **baseline** (*modèle de référence*) — enseignée : [ch. 0A, 100.6.4 · Sérialiser : `json` et `pickle`][s4] ; [ch. 13, 13.1 · Pourquoi ce chapitre ?][s107] ; [ch. B8, 208.3 · Gradient de politique : politique paramétrée, REINFORCE,…][s108] · exercices : 0A.39, 0A.Q8, 13.12, 13.Q1, B8.3, B8.7, B8.Q5–Q6, B8.Q9, B8.R3 (+ 6) · approfondie : ch. 22
- **baseline (d'un gradient de politique)** (*baseline*) — enseignée : [ch. B8, 208.3 · Gradient de politique : politique paramétrée, REINFORCE,…][s108] · exercices : 13.E5, B8.3, B8.7, B8.16–17, B8.E1, B8.Q5–Q6, B8.Q9, B8.R3 (+ 2) · approfondie : ch. 22
- **batch** (*lot*) — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s6] ; [ch. 19, 19.4.1 · La descente de gradient par batch][s7] · exercices : 0A.5, 0A.31, 0A.66, 0A.E4, 0A.Q11, 10.14, 19.25, 19.Q8, 23.5, 23.Q10 (+ 11) · réutilisée : ch. B3
- *batch gradient descent* → voir [descente par batch](#d)
- *batch inference* → voir [inférence par batches, en ligne](#i)
- **batchnorm** (*normalisation par lot*) — enseignée : [ch. 9, Pénalité sur les poids, dropout, batchnorm][s20] ; [ch. 20, 20.4.4 · La batchnorm][s240] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 18) · réutilisée : ch. 24, 27, B1
- *Bayes' rule* → voir [règle de Bayes, théorème de Bayes](#r)
- *Bayes' theorem* → voir [règle de Bayes, théorème de Bayes](#r)
- *Bayesian* → voir [fréquentiste, bayésien](#f)
- *Bayesian linear regression* → voir [régression linéaire bayésienne](#r)
- *Bayesian optimization* → voir [optimisation bayésienne](#o)
- *Bayesian updating* → voir [mise à jour bayésienne, séquentielle](#m)
- bayésien → voir [fréquentiste, bayésien](#f)
- *bayésien naïf* → voir [Naïve Bayes (gaussien, multinomial)](#n)
- **BCa** (*bias-corrected and accelerated bootstrap*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s145] · exercices : 2.6, 2.9–10, 2.22–24, 2.E3, 2.Q10
- **beeswarm** (*essaim d'abeilles*) — enseignée : [ch. B6, Les trois graphiques à connaître][s584] · exercices : B6.2, B6.7, B6.11, B6.15–17, B6.E1–E2, B6.Q5–Q6
- *Bellman equation* → voir [équation de Bellman](#e)
- *Bellman residual* → voir [résidu de Bellman](#r)
- **benchmark** (*banc d'essai, référence*) — enseignée : [ch. 1, 1.7 · Deep learning][s27] · exercices : 1.1, 1.3, 1.10, 1.23, 1.E4, 1.Q10–Q11, 14.27, 22.18, B4.Q8
- *benchmark contamination* → voir [contamination (d'un benchmark)](#c)
- **bénédiction de la non-uniformité, de la structure** (*blessing of non-uniformity*) — enseignée : [ch. 7, 7.6 · La malédiction de la dimension][s148] · exercices : 7.4, 7.8, 7.20, 7.30–31, 7.E4, 7.Q9–Q10
- *Bernoulli distribution* → voir [loi de Bernoulli](#l)
- *best checkpoint* → voir [meilleur checkpoint](#m)
- *Beta distribution* → voir [loi Beta](#l)
- **bfloat16** (*brain floating point*) — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s391] · exercices : B7.1, B7.20
- *bi-encoder* → voir [bi-encodeur](#b)
- **bi-encodeur** (*bi-encoder*) — enseignée : [ch. B4, La recherche dense][s535] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **biais** (*bias*) — enseignée : [ch. 0B, 101.1.4 · Moyenne, moyenne pondérée, somme pondérée, moyenne mobile][s39] ; [ch. 10, 10.3.3 · Les neurones modernes][s40] · exercices : 0B.4, 0B.35, 0B.Q3, 10.1–2, 10.5, 10.12, 10.Q8–Q9, 10.R1 (+ 29) · approfondie : ch. 11, B6 · réutilisée : ch. 9, 13, 14, 16, 20
- **biais (d'une pièce)** (*bias*) — enseignée : [ch. 4, 4.3 · Lancer des pièces][s174] · exercices : 4.6, 4.13, 4.19, 4.Q4, 4.R3, 8.6, 9.24, 9.Q9–Q10, 9.R3 (+ 18) · approfondie : ch. 10, 11, B6 · réutilisée : ch. 9, 13, 14, 16, 20
- **biais (équité)** (*bias*) — enseignée : [ch. B6, 206.8 · Biais : données, labels, mesure, variables de…][s593] · exercices : 4.6, 4.19, 4.Q4, B6.9–10, B6.24, B6.26, B6.E3, B6.E5, B6.Q10 (+ 21) · approfondie : ch. 10, 11 · réutilisée : ch. 9, 13, 14, 16, 20
- **biais (statistique)** (*bias*) — enseignée : [ch. 9, 9.6 · Biais et variance][s238] · exercices : 4.6, 4.19, 4.Q4, 9.4, 9.8, 9.24, 9.29, 9.E1, 9.Q9, 9.R3 (+ 20) · approfondie : ch. 10, 11, B6 · réutilisée : ch. 13, 14, 16, 20
- **biais d'automatisation** (*automation bias*) — enseignée : [ch. B6, La supervision humaine][s600] · exercices : B6.8–9, B6.28, B6.Q12
- **biais d'échantillonnage** (*sampling bias, selection bias*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s145] ; [ch. 11, Au-delà du livre : le bandit manchot][s121] · exercices : 2.6, 2.9–10, 2.22–24, 2.E3, 2.Q10, 11.2, 11.R1 (+ 18)
- **biais d'exposition** (*exposure bias*) — enseignée : [ch. B3, Teacher forcing et décalage d'un token][s510] · exercices : B3.5, B3.11, B3.19, B3.E3, B3.Q5, B3.Q9
- **biais d'optimisme (du gagnant)** (*optimistic bias, winner's curse*) — enseignée : [ch. 8, 8.4 · Le jeu de validation][s129] · exercices : 8.1–3, 8.6, 8.10, 8.17–18, 8.Q7–Q8, 8.R1 (+ 3)
- **biais de fréquence (embeddings)** (*frequency bias*) — enseignée : [ch. B2, Les biais des embeddings][s490] · exercices : B2.2, B2.11, B2.23–24, B2.26–27, B2.E4, B2.Q11–Q12
- **biais de labels, de mesure, de déploiement** (*label, measurement, deployment bias*) — enseignée : [ch. B6, D'où vient un biais][s594] · exercices : B6.9–10, B6.24, B6.26, B6.E3, B6.E5, B6.Q10
- **biais de longueur (d'un modèle de récompense)** (*length bias*) — enseignée : [ch. B8, GAE : une moyenne de toutes les longueurs][s657] · exercices : B8.4, B8.18, B8.20, B8.Q7
- **biais de prédiction** (*prediction bias*) — enseignée : [ch. B7, Pourquoi un modèle se dégrade sans changer, et que surveiller][s630] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- **biais de sélection** (*selection bias*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s336] · exercices : 20.11, 20.Q12
- **biais inductif** (*inductive bias*) — enseignée : [ch. 11, 11.2.3 · L'optimisation][s249] ; [ch. 13, 13.2 · Paramétrique ou non paramétrique ?][s251] · exercices : 11.12, 11.E3, 11.Q5, 13.8, 13.12, 13.31–32, 13.35, 13.E5, 13.Q2
- *bias* → voir [biais](#b), [biais (d'une pièce)](#b), [biais (statistique)](#b), [biais (équité)](#b)
- *bias correction* → voir [correction de biais](#c)
- *bias trick* → voir [astuce du biais](#a)
- *bias-corrected and accelerated bootstrap* → voir [BCa](#b)
- *bias-variance decomposition* → voir [décomposition biais-variance](#d)
- *bias-variance trade-off* → voir [compromis biais-variance](#c)
- *biased sample* → voir [échantillon biaisé](#e)
- **bibliothèque tokenizers** — enseignée : [ch. B2, 202.5 · WordPiece, Unigram, SentencePiece ; la bibliothèque…][s470] · exercices : B2.13, B2.19, B2.30, B2.E2, B2.Q7
- *bidirectional RNN* → voir [RNN profond, bidirectionnel](#r)
- bidirectionnel → voir [RNN profond, bidirectionnel](#r)
- *bigram* → voir [bigramme](#b), [modèle unigramme, bigramme, trigramme](#m)
- **bigramme** (*bigram*) — enseignée : [ch. 1][s111] · exercices : 6.26 · réutilisée : ch. 6
- *binary* → voir [classification binaire, multi-classe, multi-étiquette](#c)
- **binary relevance** (*pertinence binaire*) — enseignée : [ch. 7, 7.4.1 · Un-contre-tous (OvR)][s165] · exercices : 7.1, 7.16, 7.22, 7.Q6
- *binomial coefficient* → voir [coefficient binomial](#c)
- *bit* → voir [bit (d'information)](#b)
- **bit (d'information)** (*bit, shannon, Sh*) — enseignée : [ch. 6, 6.3 · Le bit, une unité][s190] · exercices : 0B.16, 6.1–2, 6.10, 6.20, 6.Q4, 6.Q6, 6.Q12, 6.R3, 11.1 · approfondie : ch. 0B · réutilisée : ch. 13, 18
- **bits et nats** — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s191] · exercices : 0B.16 · approfondie : ch. 0B · réutilisée : ch. 18
- *black box* → voir [boîte noire](#b)
- *blanchiment* → voir [whitening](#w)
- *blessing of non-uniformity* → voir [bénédiction de la non-uniformité, de la structure](#b)
- *blob* → voir [nuage gaussien](#n)
- **bloc feed-forward (FFN)** (*position-wise feed-forward network*) — enseignée : [ch. 16, 16.1 · Pourquoi ce chapitre ?][s293] · exercices : 16.9, 16.Q1, 16.R2
- **bloc Transformer** (*Transformer block*) — enseignée : [ch. B3, 203.6 · Le bloc Transformer : résiduel, layer norm (pré ou post),…][s506] · exercices : B3.5, B3.11, B3.16, B3.19, B3.25, B3.Q8, B3.R3
- **bloc Transformer pré-LN** — enseignée : [ch. B3, 203.6 · Le bloc Transformer : résiduel, layer norm (pré ou post),…][s506] · exercices : B3.5, B3.11, B3.16, B3.19, B3.25, B3.Q8, B3.R3
- **BM25** (*Okapi BM25*) — enseignée : [ch. B4, 204.6 · Recherche d'information : découpage, BM25, recherche…][s533] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **boîte noire** (*black box*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s336] · exercices : 1.23, 20.11, 20.Q12 · réutilisée : ch. 1, 9, 12, 16
- **bold driver** — enseignée : [ch. 19, 19.3.3 · Les plannings de décroissance][s314] · exercices : 19.1, 19.15, 19.19, 19.24, 19.27, 19.E3, 19.Q7
- *Boltzmann* → voir [politique softmax, de Boltzmann](#p)
- *Bonferroni correction* → voir [correction de Bonferroni](#c)
- **bonus d'entropie** (*entropy bonus*) — enseignée : [ch. B8, La baseline et la variance][s643] · exercices : B8.3, B8.7, B8.16–17, B8.E1, B8.Q5–Q6, B8.Q9, B8.R3
- *boolean mask* → voir [masque booléen](#m)
- **boosting** — enseignée : [ch. 14, 14.7 · Le boosting][s275] · exercices : 14.2, 14.4, 14.6–8, 14.10, 14.Q8–Q11 (+ 17) · réutilisée : ch. 15, B6
- *bootstrap* → voir [bootstrap, rééchantillon](#b)
- *bootstrap aggregating* → voir [bagging](#b)
- **bootstrap apparié** (*paired bootstrap*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s222] · exercices : 8.2, 8.4, 8.8, 8.16, 8.18, 8.26–27, 8.E2, 8.E4, 8.Q11
- *bootstrap sample* → voir [bootstrap, rééchantillon](#b)
- **bootstrap stratifié (de runs)** (*stratified bootstrap*) — enseignée : [ch. B8, Rapporter plusieurs graines][s649] · exercices : B8.8, B8.15, B8.21–22, B8.27–28, B8.E3, B8.E5, B8.Q10
- **bootstrap stratifié (par groupe)** (*stratified bootstrap*) — enseignée : [ch. B6][s576]
- **bootstrap, rééchantillon** (*bootstrap, bootstrap sample*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s145] ; [ch. 14, 14.4 · Le bagging][s146] · exercices : 2.6, 2.9–10, 2.22–24, 2.E3, 2.Q10, 14.Q4–Q5 (+ 13) · réutilisée : ch. 9, 26, B6, B7, B8
- *bootstrapping* → voir [amorçage](#a)
- *border point* → voir [point cœur, point de bord, bruit](#p)
- *bottleneck* → voir [goulot](#g)
- **boucle agent-environnement** — enseignée : [ch. 26, 26.3 · La structure du RL][s402] · exercices : 26.13, 26.Q3
- **boucle d'entraînement** — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s29] ; [ch. 8, 8.2 · La boucle d'entraînement][s9] · exercices : 1.14, 1.18, 1.20, 1.25, 1.E2, 1.Q6, 8.16, 8.Q1–Q2, 8.R2 (+ 5) · approfondie : ch. 18, 20, 27 · réutilisée : ch. 5, 19, 21, 22, 23, 25, B2, B4, B5, B8
- **boucle d'entraînement alternée (detach, deux optimiseurs)** — enseignée : [ch. 27][s722]
- **boucle d'entraînement from scratch** — enseignée : [ch. 18][s311]
- **boucle d'entraînement PyTorch** — enseignée : [ch. 20, 20.1 · Pourquoi ce chapitre ?][s703] · exercices : 20.15, 20.23, 20.E3, 20.Q1, CP4.7 · réutilisée : ch. B2, B4, B5, B8
- **boucle de rétroaction** (*feedback loop*) — enseignée : [ch. B6, D'où vient un biais][s594] · exercices : 16.Q3, B6.9–10, B6.24, B6.26, B6.E3, B6.E5, B6.Q10
- *boundary method* → voir [méthode à frontière](#m)
- **BPE** (*byte-pair encoding*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s191] · exercices : B2.3, B2.10, B2.19, B2.E2, B2.Q4–Q5, B3.26 · approfondie : ch. B2 · réutilisée : ch. B3
- **BPE au niveau octet** (*byte-level BPE*) — enseignée : [ch. B2, 202.4 · BPE au niveau octet, pré-tokenisation et tokens spéciaux][s468] · exercices : B2.4, B2.16–18, B2.30, B2.E3, B2.Q5–Q6
- **BPE from scratch** — enseignée : [ch. B2, 202.4 · BPE au niveau octet, pré-tokenisation et tokens spéciaux][s468] · exercices : B2.4, B2.16–18, B2.30, B2.E3, B2.Q5–Q6
- BPTT → voir [rétropropagation dans le temps (BPTT)](#r)
- *Bradley-Terry model* → voir [modèle de Bradley-Terry](#m)
- *brain floating point* → voir [bfloat16](#b)
- *branch* → voir [branche](#b)
- **branche** (*branch*) — enseignée : [ch. 0A, 100.11.2 · git : branches (aperçu)][s76] · exercices : 0A.12, 0A.E5, 0A.Q12, 24.28, 24.Q12 · approfondie : ch. 24 · réutilisée : ch. B7
- *breadth-first search* → voir [recherche en largeur](#r)
- *breakpoint* → voir [point d'arrêt, débogueur](#p)
- *Brier score* → voir [score de Brier](#s)
- **brisure de symétrie** (*symmetry breaking*) — enseignée : [ch. 16][s297]
- **broadcasting** (*diffusion*) — enseignée : [ch. 0A, 100.8.4 · Broadcasting][s69] · exercices : 0A.8, 0A.52, 0A.56, 0A.Q11, 16.3 · réutilisée : ch. 0B, 7, 16, 21
- bruit → voir [point cœur, point de bord, bruit](#p)
- **bruit de Monte-Carlo** (*Monte Carlo error*) — enseignée : [ch. 2][s135]
- **budget de pas (d'environnement)** (*environment step budget*) — enseignée : [ch. B8][s656]
- *buffer* → voir [tampon (d'un optimiseur)](#t), [buffer (d'un module)](#b)
- **buffer (d'un module)** (*buffer*) — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s2]
- *byte fallback* → voir [repli sur les octets](#r)
- *byte-level BPE* → voir [BPE au niveau octet](#b)
- *byte-pair encoding* → voir [BPE](#b)
- *bytes per token* → voir [compression (d'un tokenizer)](#c)

## C

- *C* → voir [marge souple, paramètre C](#m)
- *C order* → voir [ordre C, ligne par ligne](#o)
- *C2PA* → voir [métadonnées de provenance, C2PA](#m)
- C2PA → voir [métadonnées de provenance, C2PA](#m)
- **CACE** (*Changing Anything Changes Everything*) — enseignée : [ch. B7, La dette technique du ML][s634] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- *calibrated* → voir [calibration, calibré](#c)
- *calibration* → voir [calibration, calibré](#c)
- **calibration de Platt, régression isotone** (*Platt scaling, isotonic regression*) — enseignée : [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s166]
- **calibration par groupe** (*calibration within groups*) — enseignée : [ch. B6, Les critères][s597] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- *calibration within groups* → voir [calibration par groupe](#c)
- **calibration, calibré** (*calibration, calibrated*) — enseignée : [ch. 3, Au-delà du livre (3) : la calibration des probabilités][s169] · exercices : 3.28, 3.E5, B6.6, B6.Q11 · réutilisée : ch. 6, B6
- calibré → voir [calibration, calibré](#c)
- *call stack* → voir [pile d'appels](#p)
- **callback** — enseignée : [ch. 23, 23.10 · Callbacks][s376] · exercices : 23.25
- **canari** (*canary*) — enseignée : [ch. B3][s502]
- *canary* → voir [canari](#c), [déploiement canari, en ombre](#d)
- **candidat (d'une recherche)** (*candidate*) — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s287] · exercices : 15.1, 15.12, 15.18, 15.23, 15.E2, 15.Q10
- *candidate* → voir [candidat (d'une recherche)](#c)
- *canonical composition* → voir [NFC, NFD, NFKC](#n)
- *canonical decomposition* → voir [NFC, NFD, NFKC](#n)
- **capacité** (*capacity, representational power*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s38] · exercices : 1.1, 1.16–17, 1.E2, 1.Q5, 1.Q10
- *capacity* → voir [capacité](#c)
- **caractère combinant** (*combining character*) — enseignée : [ch. B2, Un caractère affiché, plusieurs points de code][s465] · exercices : B2.1, B2.18, B2.Q2–Q3
- **cardinal** (*cardinality*) — enseignée : [ch. 0B, 101.1.6 · Ensembles : union, intersection, complémentaire, cardinal][s81] · exercices : 0B.5, 0B.Q5
- *cardinality* → voir [cardinal](#c)
- *carried state* → voir [état porté](#e)
- **carrousel à erreur constante** (*constant error carousel, CEC*) — enseignée : [ch. 22, 22.7.2 · Le LSTM][s363] · exercices : 22.3–4, 22.7, 22.11, 22.16–18, 22.E3, 22.Q8–Q9
- **CART** (*classification and regression trees*) — enseignée : [ch. 13, 13.5 · Les arbres de décision][s130] · exercices : 3.R1, 13.11, 13.24, 13.26, 13.Q7, 21.27, CP6.5, B3.24, B6.4 · approfondie : ch. 21 · réutilisée : ch. 28, B1
- **carte d'activation** (*feature map, activation map, « carte de caractéristiques »*) — enseignée : [ch. 21, 21.7 · Une couche de convolution][s340] ; [ch. 28, 28.2.1 · Choisir un réseau][s341] · exercices : 21.4, 21.7, 21.22, 21.24, 21.31, 21.E2, 21.Q8, 28.1, 28.Q1, 28.R3 (+ 1)
- **carte d'attention** (*attention map*) — enseignée : [ch. B3, L'idée de Bahdanau : relire la source à chaque pas][s493] · exercices : B3.13, B3.Q1, B3.R2
- **carte de saillance** (*saliency map*) — enseignée : [ch. B6, Le gradient comme explication][s587] · exercices : B6.3, B6.18–19, B6.Q7, B6.Q9, B6.R1
- *cascading* → voir [randomisation en cascade, indépendante](#r)
- *catastrophic cancellation* → voir [annulation catastrophique](#a)
- *catastrophic forgetting* → voir [oubli catastrophique](#o)
- *categorical distribution* → voir [loi catégorielle, multinoulli](#l)
- *causal language model* → voir [modèle de langue causal (auto-régressif)](#m)
- *causal mask* → voir [masque causal](#m)
- CBOW → voir [skip-gram, CBOW](#s)
- *CEC* → voir [carrousel à erreur constante](#c)
- *cell state* → voir [mémoire de cellule](#m)
- **cellule RNN et BPTT** — enseignée : [ch. 22, 22.4 · La structure d'une cellule RNN][s357] · exercices : 22.2, 22.15, 22.18, 22.Q3
- **cellules de Voronoï** (*Voronoi cells*) — enseignée : [ch. 7, Avec des labels : le centroïde le plus proche][s210] ; [ch. 13, 13.3 · Les k plus proches voisins][s211] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 19)
- *central difference* → voir [pente centrée](#p)
- *centroid* → voir [centroïde](#c)
- **centroïde** (*centroid*) — enseignée : [ch. 7, Avec des labels : le centroïde le plus proche][s210] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 11) · réutilisée : ch. 8
- CFG → voir [guidance sans classifieur (CFG)](#g)
- *chain rule* → voir [règle de la chaîne](#r)
- **chain-of-thought** (*chaîne de pensée, raisonnement pas à pas*) — enseignée : [ch. B4, Le raisonnement pas à pas][s528] · exercices : B4.15–16, B4.E1, B4.Q6
- **chaîne d'arrêt** (*stop string*) — enseignée : [ch. B4, S'arrêter][s524] · exercices : B4.4, B4.12–14, B4.Q4–Q5, B4.R1
- **chaîne de caractères** (*string*) — enseignée : [ch. 0A, 100.2.2 · Chaînes de caractères et f-strings][s51] · exercices : 0A.2, 0A.14–15, 0A.Q3
- *chaîne de pensée* → voir [chain-of-thought](#c)
- **champ de vitesses** (*velocity field*) — enseignée : [ch. B5, Flow matching et rectified flow][s571] · exercices : B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11
- **champ récepteur** (*receptive field*) — enseignée : [ch. 21, 21.2.4 · Le champ récepteur local][s339] · exercices : 21.5, 21.Q2, CP5.2, 28.R3
- *Changing Anything Changes Everything* → voir [CACE](#c)
- *channels first* → voir [channels_first, channels_last](#c)
- *channels last* → voir [channels_first, channels_last](#c)
- **channels_first, channels_last** (*channels first, channels last*) — enseignée : [ch. 23, 23.3.4 · La forme des tenseurs d'images][s373] · exercices : 23.2, 23.Q6, 23.R1
- channels_last → voir [channels_first, channels_last](#c)
- *chat model* → voir [modèle de chat](#m)
- **chat template** (*modèle de conversation*) — enseignée : [ch. B4, 204.3 · Génération : generate(), décodage, arrêt, streaming ; chat…][s520] · exercices : B4.4, B4.12–14, B4.Q4–Q5, B4.R1
- **chat templates** — enseignée : [ch. B4, 204.3 · Génération : generate(), décodage, arrêt, streaming ; chat…][s520] · exercices : B4.4, B4.12–14, B4.Q4–Q5, B4.R1
- **ChatML** — enseignée : [ch. B4, Rôles et chat template][s521] · exercices : B4.4, B4.12–14, B4.Q4–Q5, B4.R1
- *checkerboard artifacts* → voir [artefacts en damier](#a)
- **checkpoint (d'entraînement)** (*training checkpoint*) — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s2] · exercices : 23.8, 23.23, 23.25, 27.27, B1.19, B7.R2 · approfondie : ch. 23 · réutilisée : ch. 24, 27, B1, B7
- *checksum* → voir [somme de contrôle](#s)
- **chemin de régularisation** (*regularization path*) — enseignée : [ch. 9, 9.5 · La régularisation][s229] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 2)
- **chemin relatif / absolu** (*relative / absolute path*) — enseignée : [ch. 0A, 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest…][s49] · exercices : 0A.10, 0A.Q1
- *child* → voir [parent, enfant ; ancêtre, descendant](#p)
- **choix d'un optimiseur** — enseignée : [ch. 19, 19.6 · Choisir un optimiseur][s250] · exercices : 19.9, 19.11, 19.23, 19.26–28, 19.E1, 19.Q12
- **choix du pas h** — enseignée : [ch. 5, Au-delà du livre (2) : les arrondis et le choix du pas][s183]
- **chunk** (*morceau*) — enseignée : [ch. B4, Découper][s532] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- CI → voir [intégration continue (CI)](#i)
- *class* → voir [classe, objet, méthode, attribut](#c), [classifieur, classe](#c)
- *class prototype* → voir [prototype de classe](#p)
- *class weighting* → voir [pondération des classes](#p)
- classe → voir [classifieur, classe](#c)
- **classe positive, négative** (*positive class, negative class*) — enseignée : [ch. 3, 3.7.1 · Classer des échantillons][s159] · exercices : 3.15, 3.Q6
- **classe, objet, méthode, attribut** (*class, object, method, attribute*) — enseignée : [ch. 0A, 100.3.1 · Listes : indices, tranches, méthodes, mutabilité][s53] · exercices : 0A.2, 0A.9, 0A.16, 0A.47–48, 0A.E2, 0A.Q4, 0A.Q9, 3.Q9, 4.E4 (+ 22) · approfondie : ch. 15 · réutilisée : ch. 3, 7, 8, 10, 12, 15, 16, 20, 24, 26, B5
- **classes : __init__, attributs, méthodes** — enseignée : [ch. 0A, 100.3.1 · Listes : indices, tranches, méthodes, mutabilité][s53] · exercices : 0A.2, 0A.9, 0A.16, 0A.47–48, 0A.E2, 0A.Q4, 0A.Q9, 4.E4, 15.Q3 (+ 2) · réutilisée : ch. 7, 12, 16, 20
- **classes déséquilibrées : class_weight, rééchantillonnage** — enseignée : [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s166] · exercices : 15.26
- *classification / regression* → voir [classification / régression](#c)
- **classification / régression** (*classification / regression*) — enseignée : [ch. 1, 1.3.1 · Classification][s112] · exercices : 1.8–9, 1.18–19, 1.Q7, 1.R2, 6.E1, 13.8, 13.26, 13.31 (+ 5) · approfondie : ch. 7, 13, B4 · réutilisée : ch. 3, 8, 13, 14, 15, 18, B1, B4
- *classification and regression trees* → voir [CART](#c)
- **classification binaire et multi-classe** — enseignée : [ch. 7, 7.4 · Plusieurs classes avec des classifieurs binaires][s688] · exercices : 7.1, 7.22, 7.24, 7.E1, 7.Q6
- **classification binaire, multi-classe, multi-étiquette** (*binary, multi-class, multi-label classification*) — enseignée : [ch. 7, 7.2.1 · Classification binaire en 2D][s160] · exercices : 7.7, 7.10–11, 7.14–15, 7.24, 7.Q2–Q4, 7.R2 (+ 1)
- **classification de textes** — enseignée : [ch. B4, 204.9 · Classer des textes : encodeur fine-tuné, LLM en zéro-shot…][s733] · exercices : B4.28, B4.R2
- *classification token* → voir [token `[CLS]`](#t)
- *classifier* → voir [classifieur, classe](#c)
- *classifier guidance* → voir [guidance par classifieur](#g)
- *classifier-free guidance* → voir [guidance sans classifieur (CFG)](#g)
- **classifieur du centroïde le plus proche** (*nearest centroid classifier*) — enseignée : [ch. 7, Avec des labels : le centroïde le plus proche][s210] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 9) · réutilisée : ch. 8
- **classifieur paramétrique, non paramétrique** (*parametric, non-parametric classifier*) — enseignée : [ch. 13, 13.2 · Paramétrique ou non paramétrique ?][s251] · exercices : 13.8, 13.12, 13.31–32, 13.35, 13.E5, 13.Q2
- **classifieur, classe** (*classifier, class*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s29] · exercices : 0A.47–48, 0A.Q9, 1.14, 1.18, 1.20, 1.25, 1.E2, 1.Q6, 3.Q9 (+ 34) · approfondie : ch. 0A, 7, 15, B5 · réutilisée : ch. 7, 8, 10, 12, 15, 20, 24, 26, 27, 29, B7
- clé → voir [requête, clé, valeur](#r)
- CLIP → voir [échelle des logits (CLIP)](#e)
- *clip fraction* → voir [fraction clippée](#f)
- **CLIP score** — enseignée : [ch. B5, La mémorisation][s572] · exercices : B5.10, B5.25, B5.28–29, B5.31, B5.Q10, B5.R2
- *clipped objective* → voir [objectif clippé, PPO](#o)
- *clone* → voir [clone (d'un estimateur)](#c)
- **clone (d'un estimateur)** (*clone*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s218] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10 (+ 1) · approfondie : ch. 0A, 15 · réutilisée : ch. 14
- *closed form* → voir [forme fermée (du processus direct)](#f)
- *closed-world assumption* → voir [hypothèse du monde clos](#h)
- *closure* → voir [fermeture](#f)
- **clustering** (*partitionnement, regroupement*) — enseignée : [ch. 1, 1.4.1 · Clustering][s114] · exercices : 1.21, 1.Q8, 15.13
- **clustering par densité** (*density-based clustering*) — enseignée : [ch. 7, Quand k-means échoue, et le clustering par densité][s215] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 9)
- **CNN PyTorch (nn.Conv2d, nn.MaxPool2d)** — enseignée : [ch. 21][s707]
- **CNN-LSTM** — enseignée : [ch. 22, 22.8.1 · Une ou plusieurs entrées et sorties][s364] · exercices : 22.8, 22.10, 22.23, 22.28–29, 22.Q10
- *co-occurrence matrix* → voir [matrice de co-occurrences](#m)
- **coalition** — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s578] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- *coarse-to-fine search* → voir [recherche multirésolution](#r)
- **codage arithmétique** (*arithmetic coding*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s202]
- *codage disjonctif* → voir [one-hot](#o)
- *code* → voir [code, variables latentes](#c)
- **code à longueur variable, code adaptatif** (*variable-length code*) — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s194] · exercices : 6.4, 6.6, 6.13, 6.15, 6.23–24, 6.27, 6.Q7
- code adaptatif → voir [code à longueur variable, code adaptatif](#c)
- **code de colle** (*glue code*) — enseignée : [ch. B7, La dette technique du ML][s634] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- **code de Huffman** (*Huffman code*) — enseignée : [ch. 6, Le code de Huffman][s195] · exercices : 6.4, 6.6, 6.13, 6.15, 6.23–25, 6.27, 6.Q7
- **code de longueur fixe** (*fixed-length code*) — enseignée : [ch. 6, 6.5 · La taille d'un événement][s193] · exercices : 6.2, 6.10, 6.13, 6.21, 6.Q6
- **code de sortie** (*exit code*) — enseignée : [ch. B7, Quatre familles de tests][s607] · exercices : B7.15, B7.23, B7.25, B7.E3, B7.Q7
- **code de statut** (*status code*) — enseignée : [ch. B7, En ligne ou par batches, et une API HTTP][s621] · exercices : B7.3, B7.5, B7.16–19, B7.Q8–Q9, B7.R2–R3 (+ 2)
- *code point* → voir [point de code](#p)
- **code préfixe** (*prefix code, prefix-free code*) — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s194] · exercices : 6.4, 6.6, 6.13, 6.15, 6.23–24, 6.27, 6.Q7
- *code review* → voir [revue de code](#r)
- **code, variables latentes** (*code, latent variables*) — enseignée : [ch. 25, 25.5.1 · Les variables latentes de près][s393] · exercices : 0A.Q5, 0B.35, 6.4, 6.6, 6.18, 6.24, 6.Q7, 6.Q10, 25.19, 25.Q7 (+ 13) · approfondie : ch. 6 · réutilisée : ch. 6
- **codes adaptatifs** — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s194] · exercices : 6.4, 6.6, 6.13, 6.15, 6.23–24, 6.27, 6.Q7
- **codes de longueur fixe** — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s194] · exercices : 6.4, 6.6, 6.13, 6.15, 6.23–24, 6.27, 6.Q7
- **coefficient binomial** (*binomial coefficient, « n choose k »*) — enseignée : [ch. 0B, 101.1.7 · Dénombrement : principe multiplicatif, factorielle,…][s11] · exercices : 0B.12, 0B.33, 0B.Q5 · réutilisée : ch. 7, 14, 15
- **coefficient de détermination ($R^2$)** (*coefficient of determination*) — enseignée : [ch. 8][s220]
- **coefficient de silhouette** (*silhouette coefficient*) — enseignée : [ch. 7, Combien de clusters ?][s213] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 9)
- *coefficient of determination* → voir [coefficient de détermination ($R^2$)](#c)
- **cognition incarnée** (*embodied cognition*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s241] · exercices : 10.8–9, 10.E4, 10.Q2–Q3
- cohérence → voir [précision locale, absence, cohérence](#p)
- col → voir [point selle, col](#p)
- **Colab / Jupyter : cellules, Markdown, noyau, « Tout exécuter »** — enseignée : [ch. 0A, 100.1.1 · Cellules, noyau, ordre d'exécution et « Tout exécuter »][s48] · exercices : 0A.13, 0A.E4, 0A.Q1, 0B.32, 13.5, 13.18, 13.Q6, CP3.9, 21.20 · approfondie : ch. 0B, 13 · réutilisée : ch. 0B, 28
- *cold start* → voir [démarrage à froid](#d)
- *collapse of a linear network* → voir [effondrement (d'un réseau linéaire)](#e)
- **colonne indicatrice (de valeur manquante)** (*missing indicator*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s264] · exercices : 12.11, 12.Q4
- *column transformer* → voir [ColumnTransformer](#c)
- **ColumnTransformer** (*column transformer*) — enseignée : [ch. 15, Au-delà du livre : les données hétérogènes][s291] · exercices : 15.28, 15.E3 · réutilisée : ch. B6
- *combining character* → voir [caractère combinant](#c)
- **commit** — enseignée : [ch. 0A, 100.11.1 · git : `status`, `add`, `commit`, `push`, `pull` ;…][s75] · exercices : 0A.10–11, 0A.E5, 0A.Q12 · réutilisée : ch. B7
- **comparaison appariée** (*paired comparison*) — enseignée : [ch. 8][s220]
- **comparaisons multiples** (*multiple comparisons*) — enseignée : [ch. B6, Des écarts avec leurs intervalles][s598] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- *compatibility normalization* → voir [NFC, NFD, NFKC](#n)
- *compilation* → voir [compilation (d'un modèle)](#c)
- **compilation (d'un modèle)** (*compilation, torch.compile*) — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s391]
- complet → voir [fine-tuning partiel, complet](#f)
- *completeness* → voir [complétude](#c)
- **complétude** (*completeness*) — enseignée : [ch. B6, Les gradients intégrés][s590] · exercices : B6.3, B6.18–19, B6.Q7, B6.R1
- *component* → voir [composante](#c)
- **composante** (*component, entry*) — enseignée : [ch. 0B, 101.3.1 · Composantes, somme, produit par un scalaire][s87] · exercices : 0B.8, 0B.38, 0B.40, 0B.Q8, 12.24, 25.19
- **composante principale** (*principal component*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s152] · exercices : 12.7–8, 12.22–23, 12.27, 12.Q9
- **composition de fonctions** — enseignée : [ch. 0B, 101.2.7 · Composition de fonctions][s682] · exercices : 0B.23, 0B.47, 0B.Q7, 0B.R3
- *comprehension* → voir [compréhension](#c)
- **compréhension** (*comprehension*) — enseignée : [ch. 0A, 100.4.3 · Compréhensions de listes, de dictionnaires et d'ensembles][s57] · exercices : 0A.3, 0A.22, 0A.Q5, 0B.R1 · réutilisée : ch. 0B
- **compression (d'un tokenizer)** (*bytes per token*) — enseignée : [ch. B2, Un algorithme de compression devenu tokenizer][s474] · exercices : B2.3–4, B2.10, B2.12, B2.15–17, B2.20, B2.Q4, B2.R3 (+ 5) · réutilisée : ch. 6
- **compression avec ou sans perte** (*lossy, lossless compression*) — enseignée : [ch. 25, 25.2.1 · Avec ou sans perte][s395] · exercices : 25.Q1
- *compression ratio* → voir [taux de compression](#t)
- **compression sans perte** (*lossless compression*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s202] · exercices : B2.31
- **compromis biais-variance** (*bias-variance trade-off*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s239] · exercices : 9.4, 9.8, 9.11, 9.25, 9.30–31, 9.E1, 9.E5, 9.Q9
- **comptage des paramètres** — enseignée : [ch. 16][s297] · réutilisée : ch. 21, 22, 23
- *computational graph* → voir [graphe de calcul](#g)
- **concentration des distances** (*distance concentration*) — enseignée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s214] · exercices : 7.5–6, 7.19–21, 7.E4, 7.Q11
- **conception de la récompense** — enseignée : [ch. 26, 26.3.7 · Les récompenses][s405] · exercices : 26.1, 26.7, 26.11, 26.14–15, 26.32, 26.E2, 26.Q6
- *condition dropout* → voir [abandon du label (guidance sans classifieur)](#a)
- *condition number* → voir [conditionnement d'une matrice](#c)
- **condition vide** (*null condition*) — enseignée : [ch. B5, Un bloc résiduel conditionné par le pas][s562] · exercices : B5.11, B5.19, B5.21, B5.Q5–Q6, B5.R1
- *conditional entropy* → voir [entropie conditionnelle](#e)
- *conditional GAN* → voir [GAN conditionnel](#g)
- *conditional independence* → voir [indépendance conditionnelle](#i)
- *conditional probability* → voir [probabilité conditionnelle](#p)
- *conditioning* → voir [conditionnement](#c)
- **conditionnement** (*conditioning*) — enseignée : [ch. 5][s184] · approfondie : ch. 11 · réutilisée : ch. 26
- **conditionnement d'une matrice** (*condition number*) — enseignée : [ch. 9, Les moindres carrés (au-delà du livre)][s232] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 2)
- **conditionnement opérant** (*operant conditioning*) — enseignée : [ch. 11, 11.7 · Le conditionnement opérant][s259] · exercices : 11.2, 11.7–8, 11.10–11, 11.13, 11.19–21, 11.R1 (+ 8) · réutilisée : ch. 26
- *Condorcet's jury theorem* → voir [théorème du jury](#t)
- *confidence interval* → voir [intervalle de confiance](#i)
- **confidentialité différentielle (DP-SGD)** (*differential privacy*) — enseignée : [ch. B3][s502]
- **configuration typée** (*typed configuration*) — enseignée : [ch. B7, 207.3 · Configuration : dataclasses, YAML ou JSON, surcharges,…][s610] · exercices : B7.11, B7.25, B7.E1, B7.Q4
- *confirmatory test* → voir [test de confirmation](#t)
- *confounder* → voir [variable de confusion](#v)
- *confounding variable* → voir [variable de confusion](#v)
- *confusion matrix* → voir [matrice de confusion](#m)
- *conjugate prior* → voir [prior conjugué](#p)
- **connectome** — enseignée : [ch. 10, 10.2 · Les vrais neurones][s241] · exercices : 10.8–9, 10.E4, 10.Q2–Q3
- **connexion de saut** (*skip connection*) — enseignée : [ch. B5, Le U-Net et ses connexions de saut][s561] · exercices : B5.11, B5.19, B5.21, B5.Q5–Q6, B5.R1
- **connexion persistante** (*keep-alive*) — enseignée : [ch. B7][s614]
- **connexion résiduelle** (*residual connection*) — enseignée : [ch. 20, Au-delà du livre : les connexions résiduelles][s333] ; [ch. 24, 24.6 · L'API fonctionnelle][s334] ; [ch. B3, Deux sous-couches][s335] · exercices : 24.10, 24.28, 24.Q12, B3.5, B3.11, B3.16, B3.19, B3.25, B3.Q8, B3.R3 (+ 1)
- **connexions résiduelles** — enseignée : [ch. 20, Au-delà du livre : les connexions résiduelles][s333] · exercices : 20.30, 20.E5 · réutilisée : ch. B3
- **conservation de la variance** — enseignée : [ch. 16][s297]
- *consistency* → voir [précision locale, absence, cohérence](#p)
- *consistency model* → voir [modèle de cohérence](#m)
- *constant error carousel* → voir [carrousel à erreur constante](#c)
- *constrained decoding* → voir [génération contrainte](#g)
- *container* → voir [conteneur, image](#c)
- **contamination (d'un benchmark)** (*benchmark contamination*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s222] · exercices : 8.2, 8.4, 8.8, 8.16, 8.18, 8.26–27, 8.E2, 8.E4, 8.Q11 (+ 1)
- **conteneur, image** (*container, image*) — enseignée : [ch. B7, 207.9 · Conteneurs et déploiement : Dockerfile, variables…][s626] · exercices : 0A.54, 0A.Q10, 1.10, 2.11, 2.Q11, 20.26, B7.23–25, B7.Q4 (+ 37) · approfondie : ch. 0A · réutilisée : ch. B2
- *content credentials* → voir [métadonnées de provenance, C2PA](#m)
- *content image* → voir [image de contenu, référence de style, image générée](#i)
- *content loss* → voir [loss de contenu](#l)
- *context distribution smoothing* → voir [lissage des contextes](#l)
- *context manager* → voir [gestionnaire de contexte](#g)
- *context vector* → voir [vecteur de contexte](#v)
- **contexte global, contexte local** (*global context, local context*) — enseignée : [ch. 6, Le contexte][s192] · exercices : 6.26, 6.Q2–Q3
- contexte local → voir [contexte global, contexte local](#c)
- *contextual bandit* → voir [bandit contextuel](#b)
- *contextual embedding* → voir [embedding statique, contextuel](#e)
- contextuel → voir [embedding statique, contextuel](#e)
- *contiguous tensor* → voir [tenseur contigu](#t)
- *contingency table* → voir [table de contingence](#t)
- *continuous* → voir [courbe continue, lisse, univoque](#c)
- *continuous bag of words* → voir [skip-gram, CBOW](#s)
- *continuous integration* → voir [intégration continue (CI)](#i)
- *contour line* → voir [ligne de niveau](#l)
- **contraste des distances** (*relative contrast*) — enseignée : [ch. 2][s135]
- *contrastive learning* → voir [apprentissage contrastif](#a)
- **contribution marginale** (*marginal contribution*) — enseignée : [ch. B6, Partager un gain entre joueurs][s579] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- **convention AD (ou DA)** (*weight naming convention*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s208] · exercices : 10.1, 10.3, 10.6, 10.11–12, 10.17, 10.Q5–Q6, 10.R2–R3 (+ 5)
- **convention des poids** — enseignée : [ch. 10, 10.3.1 · Le perceptron][s208] · exercices : 10.1, 10.3, 10.6, 10.11–12, 10.17, 10.Q5–Q6, 10.R2–R3 (+ 4)
- **convention W (n_in, n_out)** — enseignée : [ch. 16][s297]
- **convolution 1 × 1** (*pointwise convolution*) — enseignée : [ch. 21, 21.4 · La convolution en plusieurs dimensions][s344] · exercices : 21.15, 21.19, 21.Q6
- **convolution 1D** (*1D convolution*) — enseignée : [ch. 21, 21.5 · La convolution 1D][s345] · exercices : 21.7, 21.20, 21.Q7, 22.R1, CP5.12
- **convolution 2D (corrélation croisée)** — enseignée : [ch. 21, 21.3 · La convolution][s337] · exercices : 21.2, 21.9, 21.15, 21.Q3
- **convolution par canal** (*depthwise convolution*) — enseignée : [ch. 21, 21.4 · La convolution en plusieurs dimensions][s344] · exercices : 21.15, 21.19, 21.Q6
- **convolution séparable** (*depthwise separable convolution*) — enseignée : [ch. 21, 21.3 · La convolution][s337] · exercices : 21.2, 21.9, 21.15, 21.Q3
- **convolution transposée** (*transposed convolution, fractionally strided convolution*) — enseignée : [ch. 21, 21.8 · La convolution transposée][s343] · exercices : 21.3, 21.6, 21.30, 21.Q9, 25.4 · réutilisée : ch. 25, 27, B5
- *cooperative game* → voir [jeu coopératif](#j)
- *coordinate descent* → voir [descente de coordonnées](#d)
- *core point* → voir [point cœur, point de bord, bruit](#p)
- *corner* → voir [point anguleux](#p)
- **correction de biais** (*bias correction*) — enseignée : [ch. 19, 19.5.5 · Adam][s323] · exercices : 19.5–6, 19.8–9, 19.12–13, 19.18, 19.22, 19.24, 19.Q12 (+ 1)
- **correction de Bonferroni** (*Bonferroni correction*) — enseignée : [ch. B7, Le test de Kolmogorov-Smirnov][s629] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- **corrélation (de Pearson)** (*Pearson correlation coefficient*) — enseignée : [ch. 2, 2.8 · Covariance et corrélation][s150] · exercices : 2.7–8, 2.10, 2.26–28, 2.E4, 2.Q12, CP1.13, 21.Q3 (+ 1) · approfondie : ch. 21 · réutilisée : ch. 8, 12
- **corrélation croisée** (*cross-correlation*) — enseignée : [ch. 21, 21.3 · La convolution][s337] · exercices : 21.2, 21.9, 21.15, 21.Q3
- **corrélation de rangs (de Spearman)** (*Spearman rank correlation*) — enseignée : [ch. B6, Les tests de santé][s592] · exercices : B6.16, B6.22, B6.E2, B6.E4, B6.Q6, B6.Q9
- *correlation matrix* → voir [matrice de covariance, de corrélation](#m)
- *correspondance exacte* → voir [exact match](#e)
- **corruption de segments** (*span corruption*) — enseignée : [ch. B3, L'encodeur-décodeur : T5 et l'attention croisée][s509] · exercices : B3.5, B3.11, B3.19, B3.E3, B3.Q5, B3.Q9
- *cosine annealing* → voir [planning en cosinus](#p)
- *cosine noise schedule* → voir [planning linéaire, planning cosinus (de bruit)](#p)
- *cosine schedule* → voir [planning en cosinus](#p)
- *cosine similarity* → voir [similarité cosinus](#s)
- **cosinus (aperçu)** — enseignée : [ch. 0B, 101.2.6 · Cosinus (aperçu)][s86] · exercices : 0B.7, 0B.18, 0B.36, 0B.39, 0B.41, 0B.E2, 0B.Q6, 0B.Q8, 19.19, 19.27 (+ 10) · approfondie : ch. 19, B2, B5 · réutilisée : ch. 19, B1, B2, B3
- **cote** (*odds*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s173] · exercices : 4.3, 4.10, 4.14, 4.23, 4.E3, 4.Q6–Q7
- **couche** (*layer*) — enseignée : [ch. 1, 1.7 · Deep learning][s27] · exercices : 1.1, 1.3, 1.10, 1.23, 1.E4, 1.Q10–Q11, 10.5, 16.3, 16.13 (+ 43) · approfondie : ch. 20 · réutilisée : ch. 20, 21, 22, 25, 27
- **couche appliquée à chaque pas** (*time-distributed layer*) — enseignée : [ch. 24, 24.5.10 · Une couche appliquée à chaque pas de temps][s390] · exercices : 24.4, 24.27, 24.Q11
- **couche cachée** (*hidden layer*) — enseignée : [ch. 16, 16.1 · Pourquoi ce chapitre ?][s293] · exercices : 16.9, 16.Q1, 16.R2
- **couche d'entrée** (*input layer*) — enseignée : [ch. 20, 20.3.1 · La couche d'entrée][s328] · exercices : 20.19, 20.Q3
- **couche de sortie** (*output layer*) — enseignée : [ch. 20, 20.3.2 · La couche de sortie][s329] · exercices : 20.2, 20.23, 20.Q3, 28.7
- **couche dense** (*dense layer, fully connected layer, couche pleine, couche entièrement connectée*) — enseignée : [ch. 1, 1.7 · Deep learning][s27] · exercices : 1.1, 1.3, 1.10, 1.23, 1.E4, 1.Q10–Q11, 16.13, 20.Q4, 21.4 (+ 1) · réutilisée : ch. 20, 21, 22, 25
- *couche entièrement connectée* → voir [couche dense](#c)
- **couche partagée** (*shared layer*) — enseignée : [ch. 24, 24.5.10 · Une couche appliquée à chaque pas de temps][s390] · exercices : 24.4, 24.27, 24.Q11
- **couche utilitaire** (*utility layer*) — enseignée : [ch. 20, 20.4.8 · Les autres couches utilitaires][s330] · exercices : 20.1, 20.7, 20.14, 20.17, 20.20, 20.30, 20.E2, 20.Q9
- *counterfactual explanation* → voir [explication contrefactuelle](#e)
- **courbe ALE** (*accumulated local effects*) — enseignée : [ch. B6, Les courbes ICE][s577] · exercices : B6.12–13, B6.E1, B6.Q2–Q3, B6.R2
- **courbe continue, lisse, univoque** (*continuous, smooth, single-valued curve*) — enseignée : [ch. 5, 5.2 · Introduction : fonctions, courbes et surfaces][s179] · exercices : 5.14, 5.Q2–Q3, 5.Q9 · approfondie : ch. 17
- **courbe d'apprentissage** (*learning curve*) — enseignée : [ch. 9, Courbes de validation et courbes d'apprentissage (au-delà du livre)][s237] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 2)
- **courbe de validation** (*validation curve*) — enseignée : [ch. 9, Courbes de validation et courbes d'apprentissage (au-delà du livre)][s237] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 3)
- **courbe ICE** (*individual conditional expectation*) — enseignée : [ch. B6, Les courbes ICE][s577] · exercices : B6.12–13, B6.E1, B6.Q2–Q3, B6.R2
- **courbe precision-recall** (*precision-recall curve*) — enseignée : [ch. 3, Au-delà du livre (2) : la courbe precision-recall et l'average…][s168] · exercices : 3.26
- **courbe ROC** (*ROC curve*) — enseignée : [ch. 3, Au-delà du livre (1) : la courbe ROC et l'AUC][s167] · exercices : 3.24, 3.E3, CP1.7 · réutilisée : ch. 15
- **courbes d'apprentissage** — enseignée : [ch. 9, Courbes de validation et courbes d'apprentissage (au-delà du livre)][s237] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 5)
- **courbes de validation** — enseignée : [ch. 9, Courbes de validation et courbes d'apprentissage (au-delà du livre)][s237] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 2)
- courbure → voir [dérivée seconde, courbure](#d)
- **coût quadratique et FlashAttention** — enseignée : [ch. B3, 203.10 · Coût et passage à l'échelle : complexité quadratique,…][s354] · exercices : B3.8, B3.10, B3.27–28, B3.E2, B3.Q12
- **couverture (d'un intervalle)** (*coverage*) — enseignée : [ch. 2][s135]
- **covariance** — enseignée : [ch. 2, 2.8 · Covariance et corrélation][s150] · exercices : 2.7–8, 2.26, 2.28, 2.Q12, 12.7 · approfondie : ch. 12 · réutilisée : ch. 12, 25, 27
- *covariance matrix* → voir [matrice de covariance, de corrélation](#m)
- *coverage* → voir [couverture (d'un intervalle)](#c)
- **CPU** (*processeur central*) — enseignée : [ch. 1, 1.7 · Deep learning][s27] · exercices : 1.1, 1.3, 1.10, 1.23, 1.E4, 1.Q10–Q11, 23.29, 23.Q7, 24.7 (+ 15) · approfondie : ch. 20
- *Creative Commons license* → voir [licence Creative Commons](#l)
- *credible interval* → voir [intervalle de crédibilité](#i)
- *credit assignment* → voir [attribution du crédit](#a)
- *critic* → voir [acteur-critique, acteur, critique](#a)
- *critical point* → voir [point critique, point stationnaire](#p)
- critique → voir [acteur-critique, acteur, critique](#a)
- **croissance progressive** (*progressive growing*) — enseignée : [ch. 27, 27.7.1 · Les grandes images][s424] · exercices : 27.7, 27.9, 27.Q9
- **Croissant** — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s125]
- **cross fitting** (*ajustement croisé*) — enseignée : [ch. 12, 12.3.1 · Le one-hot][s263] · exercices : 12.1, 12.20, 12.26, 12.33, 12.E5, 12.Q3
- *cross-attention* → voir [attention croisée](#a)
- *cross-correlation* → voir [corrélation croisée](#c)
- **cross-encoder** — enseignée : [ch. B2, Une recherche sémantique en vingt lignes][s488] · exercices : B2.28–29, B2.E4–E5, B2.Q11, B2.R1
- **cross-entropy** (*entropie croisée*) — enseignée : [ch. 6, 6.8 · La cross-entropy][s198] · exercices : 6.5, 6.7, 6.16, 6.18, 6.20, 6.22, 6.26, 6.E1, 6.Q10, 6.Q12 (+ 8) · approfondie : ch. 18 · réutilisée : ch. 9, 13, 18, 22, 23, 25, 27, B3, B4, B8
- *cross-tabulation* → voir [table de contingence](#t)
- *cross-validation* → voir [validation croisée](#v)
- **cross_val_score** — enseignée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s221] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10 · réutilisée : ch. 9, 10
- **cross_validate** — enseignée : [ch. 15, 15.9.1 · La validation croisée][s694] · exercices : 15.17, 15.Q10
- **CrossEntropyLoss et logits** — enseignée : [ch. 23, 23.5.7 · Corriger les labels][s709] · exercices : 23.4, 23.Q9
- *curse of dimensionality* → voir [malédiction de la dimension, fléau de la dimension](#m)
- *curvature* → voir [dérivée seconde, courbure](#d)
- **cv_results_** — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s287] · exercices : 15.1, 15.12, 15.18, 15.23, 15.E2, 15.Q10
- **cycle de dépréciation** (*deprecation cycle*) — enseignée : [ch. 7][s216]

## D

- DAG → voir [graphe orienté acyclique (DAG)](#g)
- *data augmentation* → voir [augmentation de données](#a)
- **data card** (*fiche de données*) — enseignée : [ch. 1, Les quatre fils rouges du workbook][s124] ; [ch. 29, Au-delà du livre : documenter et télécharger][s125] · exercices : 1.13, 29.15, 29.E2, PF.2, PF.12 · réutilisée : ch. B6
- *data cleaning* → voir [préparation des données](#p)
- *data drift* → voir [dérive des données](#d)
- *data fingerprint* → voir [empreinte (des données)](#e)
- *data leakage* → voir [fuite de données](#f)
- *data loader* → voir [`Dataset`, `DataLoader`](#d)
- *data minimisation* → voir [minimisation, limitation de la conservation](#m)
- *data preparation* → voir [préparation des données](#p)
- *data split* → voir [découpage (des données)](#d)
- *data test* → voir [test de données](#t)
- *data type* → voir [dtype](#d)
- **dataclasses (@dataclass)** — enseignée : [ch. 0A, 100.7.1 · `class`, `__init__`, attributs et méthodes ; `@dataclass`][s679] · exercices : 0A.47, 0A.Q9 · réutilisée : ch. B4, B7
- *DataFrame* → voir [DataFrame, Series](#d)
- **DataFrame, Series** (*DataFrame, Series*) — enseignée : [ch. 0A, 100.9.1 · `DataFrame` et `Series` : `read_csv`, `head`, `info`,…][s73] · exercices : 0A.32, 0A.Q12
- DataLoader → voir [`Dataset`, `DataLoader`](#d)
- Datasaurus → voir [quartet d'Anscombe, Datasaurus](#q)
- *Datasaurus Dozen* → voir [quartet d'Anscombe, Datasaurus](#q)
- **dataset** (*jeu de données*) — enseignée : [ch. 0A, 100.1.2 · Le workbook : setup, `wb.check`, `wb.attempt`,…][s1] ; [ch. 20, Au-delà du livre : PyTorch en pratique][s2] ; [ch. 23, 23.5.2 · Charger les données][s3] · exercices : 0A.26, 0A.66, 0A.E1, 0A.Q2, 15.7, 15.11, 15.Q12, 20.19, 23.13, 23.Q8 (+ 15) · approfondie : ch. 24, 29 · réutilisée : ch. 29, B1
- **dataset card** — enseignée : [ch. 29, 29.4 · Les collections de datasets][s441] · exercices : 29.6, 29.11, 29.15, 29.E2, 29.Q7
- **Dataset généré à la volée** — enseignée : [ch. 24][s713]
- **dataset retiré** (*retracted dataset*) — enseignée : [ch. 29, 29.5 · Des datasets plus récents][s443] · exercices : 15.7, 29.3, 29.5, 29.9, 29.11, 29.14, 29.E4, 29.Q8
- *dataset shift* → voir [décalage de distribution](#d)
- **`Dataset`, `DataLoader`** (*dataset, data loader*) — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s2] · exercices : 0A.66, 15.7, 15.11, 15.Q12, 20.19, 23.15, 29.Q1, 29.Q5–Q7 (+ 11) · approfondie : ch. 23, 24, 29 · réutilisée : ch. 21, 22, 23, 25, 27, 28, 29, B1
- **datasheet (d'un dataset)** (*datasheet for datasets*) — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s125] · exercices : 29.6
- *datasheet for datasets* → voir [datasheet (d'un dataset)](#d)
- **dates avec pandas (to_datetime, dt)** — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s272]
- **DBSCAN et HDBSCAN (usage)** — enseignée : [ch. 7, Quand k-means échoue, et le clustering par densité][s215] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 9)
- **DCGAN** (*deep convolutional GAN*) — enseignée : [ch. 27, 27.6 · DCGAN][s421] · exercices : 27.1, 27.20–21, 27.25, 27.28, 27.Q8, 27.R3, B5.29 · réutilisée : ch. B5
- **DDIM** (*denoising diffusion implicit model*) — enseignée : [ch. B5, 205.6 · Accélérer : DDIM déterministe, moins de pas, distillation][s557] · exercices : B5.5, B5.8, B5.17–18, B5.24–25, B5.30–31, B5.Q7–Q8 (+ 1)
- *DDIM inversion* → voir [inversion DDIM](#i)
- **ddof** (*delta degrees of freedom*) — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s142] · exercices : 2.3–4, 2.15, 2.17–18, 2.29, 2.31, 2.Q5
- **DDPM** (*denoising diffusion probabilistic model*) — enseignée : [ch. B5, 205.5 · Échantillonnage DDPM : moyenne du posterior, variance, T pas][s556] · exercices : B5.4, B5.7, B5.11, B5.16–19, B5.22, B5.24, B5.Q7 (+ 3)
- de Boltzmann → voir [politique softmax, de Boltzmann](#p)
- de corrélation → voir [matrice de covariance, de corrélation](#m)
- de déploiement → voir [biais de labels, de mesure, de déploiement](#b)
- de fausses omissions → voir [taux de fausses découvertes, de fausses omissions](#t)
- de faux négatifs → voir [taux de faux positifs, de faux négatifs](#t)
- de la structure → voir [bénédiction de la non-uniformité, de la structure](#b)
- de mesure → voir [biais de labels, de mesure, de déploiement](#b)
- *dead neuron* → voir [neurone mort](#n)
- *deadly triad* → voir [triade mortelle](#t)
- **débit** (*throughput*) — enseignée : [ch. B1, Quatre mesures à ne pas confondre][s459] · exercices : B1.7, B1.14, B1.26–27, B1.E5, B1.Q11, B7.3, B7.Q9
- débogueur → voir [point d'arrêt, débogueur](#p)
- **débruitage** (*denoising, noise reduction*) — enseignée : [ch. 1, 1.4.2 · Débruitage][s115] · exercices : 1.12, 1.Q8, 25.39
- **débruiteur parfait (d'un dataset fini)** (*optimal denoiser, ideal denoiser*) — enseignée : [ch. B5, 205.4 · Le débruiteur : U-Net minuscule et encodage du pas de temps][s560] · exercices : B5.11, B5.19, B5.21, B5.Q5–Q6, B5.R1
- *debugger* → voir [point d'arrêt, débogueur](#p)
- **décalage de distribution** (*distribution shift, dataset shift*) — enseignée : [ch. 29, Au-delà du livre : le décalage de distribution][s440] · exercices : 29.14, B1.R1 · réutilisée : ch. B1
- *décalage entre entraînement et service* → voir [training-serving skew](#t)
- *décalages* → voir [offsets](#o)
- *decay parameter* → voir [paramètre de décroissance](#p)
- *decision boundary* → voir [frontière de décision](#f)
- *decision region* → voir [région de décision](#r)
- *decision stump* → voir [souche](#s)
- *decision threshold* → voir [score, seuil de décision](#s)
- *decision tree* → voir [arbre de décision](#a)
- **decision_function** — enseignée : [ch. 15, 15.4.4 · decision_function et predict_proba][s692] · exercices : 15.2, 15.10, 15.25–26, 15.E4, 15.Q5
- **DecisionBoundaryDisplay** — enseignée : [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s166] · exercices : 15.21
- déclenché → voir [agent libre, déclenché](#a)
- **décodage glouton** (*greedy decoding*) — enseignée : [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s203] ; [ch. 28, 28.5 · Générer la suite du livre][s204] · exercices : 22.1, 22.5, 22.8, 22.12, 22.20, 22.24–27, 22.Q12 (+ 7)
- **décodage spéculatif** (*speculative decoding*) — enseignée : [ch. B3, Ce que coûte le cache][s517] · exercices : B3.8, B3.23, B3.E4, B3.Q11
- *decoder* → voir [encodeur, décodeur](#e)
- décodeur → voir [encodeur, décodeur](#e)
- **décomposition biais-variance** (*bias-variance decomposition*) — enseignée : [ch. 9, 9.6 · Biais et variance][s238] · exercices : 9.4, 9.8, 9.24, 9.29, 9.E1, 9.Q9, 9.R3
- *décomposition en valeurs singulières* → voir [SVD](#s)
- **découpage (des données)** (*data split*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s218] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10 (+ 6) · approfondie : ch. 23, 29 · réutilisée : ch. 12, 22, 29
- **découpage par groupes** (*group k-fold*) — enseignée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s221] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10
- *découpe* → voir [split](#s)
- *deduction* → voir [déduction](#d)
- **déduction** (*deduction*) — enseignée : [ch. 11, 11.3 · Déduction et induction][s252] · exercices : 11.5, 11.9, 11.Q6
- *deep convolutional GAN* → voir [DCGAN](#d)
- **deep dream** (*inceptionism*) — enseignée : [ch. 28, 28.3 · Le deep dream][s431] · exercices : 28.16–17, 28.E4, 28.Q4–Q5
- **deep learning** (*apprentissage profond*) — enseignée : [ch. 1, 1.7 · Deep learning][s27] · exercices : 1.1, 1.3, 1.10, 1.23, 1.E4, 1.Q10–Q11, 5.E2, 14.10, 20.Q1 (+ 1)
- *deep Q-network* → voir [deep RL, DQN](#d)
- *deep reinforcement learning* → voir [deep RL, DQN](#d)
- **deep RL, DQN** (*deep reinforcement learning, deep Q-network*) — enseignée : [ch. 26, 26.8 · La vue d'ensemble][s412] ; [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s234] · exercices : 26.2, 26.9, 26.12, 26.36, 26.38, 26.E4, 26.Q12, 26.R1, B8.Q3–Q4 (+ 10)
- *deep RNN* → voir [RNN profond, bidirectionnel](#r)
- **deepfake** (*hypertrucage*) — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s394] · exercices : 25.12, 27.9
- **dégel progressif** (*gradual unfreezing*) — enseignée : [ch. B1, 201.4 · Fine-tuning : partiel, complet, dégel progressif, learning…][s447] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 3)
- *delimiters* → voir [délimiteurs (d'un prompt)](#d)
- **délimiteurs (d'un prompt)** (*delimiters*) — enseignée : [ch. B4, L'anatomie d'un prompt][s545] · exercices : B4.15–16, B4.E1, B4.Q6
- *delta* → voir [delta (d'un neurone)](#d)
- **delta (d'un neurone)** (*delta, error term*) — enseignée : [ch. 18, 18.6 · Étape 1 : les deltas des neurones de sortie][s304] · exercices : 18.2, 18.4, 18.8, 18.12, 18.14, 18.17, 18.E5, 18.Q4–Q6 (+ 3)
- **delta d'un neurone** — enseignée : [ch. 18, 18.6 · Étape 1 : les deltas des neurones de sortie][s304] · exercices : 18.2, 18.4, 18.8, 18.12, 18.14, 18.17, 18.E5, 18.Q4–Q5
- *delta degrees of freedom* → voir [ddof](#d)
- *demande de fusion* → voir [pull request](#p)
- **démarche expérimentale et journal d'expériences** — enseignée : [ch. 24][s713]
- **démarrage à froid** (*cold start*) — enseignée : [ch. B1][s461]
- **démo hébergée** (*hosted demo*) — enseignée : [ch. B7, Une démo hébergée][s627] · exercices : B7.23–25, B7.Q4
- *demographic parity* → voir [parité démographique](#p)
- *denoising* → voir [débruitage](#d)
- *denoising autoencoder* → voir [autoencodeur débruiteur](#a)
- *denoising diffusion implicit model* → voir [DDIM](#d)
- *denoising diffusion probabilistic model* → voir [DDPM](#d)
- **dénombrement, factorielle, coefficient binomial C(n, k)** — enseignée : [ch. 0B, 101.1 · Nombres, notations et dénombrement][s680] · exercices : 0B.12, 0B.Q1, 0B.Q5 · réutilisée : ch. 2, 7, B6
- *denormal number* → voir [nombre dénormalisé](#n)
- *dense layer* → voir [couche dense](#c)
- *dense retrieval* → voir [recherche dense](#r)
- **densité (pdf)** (*probability density function*) — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s136] · exercices : 2.1–2, 2.13, 2.16, 2.19, 2.E1, 2.Q1–Q2, 2.R2, 7.Q9 (+ 3) · réutilisée : ch. 4, 13, 25, B5
- **densité d'échantillons** (*sample density*) — enseignée : [ch. 7, 7.6 · La malédiction de la dimension][s148] · exercices : 7.4, 7.8, 7.20, 7.30–31, 7.E4, 7.Q9–Q10, CP2.4
- *density-based clustering* → voir [clustering par densité](#c)
- *denying the antecedent* → voir [négation de l'antécédent](#n)
- **dépendance à long terme** (*long-term dependency*) — enseignée : [ch. 22, 22.6 · Entraîner un RNN][s309] · exercices : 22.6–7, 22.19, 22.23, 22.E2, 22.Q6, 22.R2
- **dépendance partielle (PDP)** (*partial dependence*) — enseignée : [ch. B6, 206.2 · Explications globales : coefficients, arbres, importance…][s284] · exercices : B6.12–13, B6.E1, B6.Q2–Q3, B6.R2
- **dépendance partielle et ICE** — enseignée : [ch. B6, 206.2 · Explications globales : coefficients, arbres, importance…][s284] · exercices : B6.12–13, B6.E1, B6.Q2–Q3, B6.R2
- *dependence plot* → voir [graphique de dépendance (SHAP)](#g)
- **déploiement canari, en ombre** (*canary, shadow deployment*) — enseignée : [ch. B7, Agir : alerter, réentraîner, revenir en arrière][s632] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- *deploy* → voir [déployer](#d)
- *deployer* → voir [déployeur](#d)
- **déployer** (*deploy*) — enseignée : [ch. 1][s111] · exercices : CP3.10, 23.E3
- **déployeur** (*deployer*) — enseignée : [ch. B6, L'AI Act][s602] · exercices : B6.8–9, B6.28, B6.Q12
- *deployment bias* → voir [biais de labels, de mesure, de déploiement](#b)
- **dépôt** (*repository*) — enseignée : [ch. 0A, 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest…][s49] · exercices : 0A.10–11, 0A.Q1, B7.25, PF.12
- *deprecation cycle* → voir [cycle de dépréciation](#c)
- *depth* → voir [profondeur (d'un arbre)](#p)
- *depthwise convolution* → voir [convolution par canal](#c)
- *depthwise separable convolution* → voir [convolution séparable](#c)
- *derivative* → voir [dérivée](#d)
- **dérive des données** (*data drift*) — enseignée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s221] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10 (+ 1) · réutilisée : ch. 20
- *derived model* → voir [modèle dérivé](#m)
- **dérivée** (*derivative*) — enseignée : [ch. 0B, 101.5 · Dérivées][s96] · exercices : 0B.25, 0B.47, 0B.51, 0B.Q10, 5.11, 5.Q1, 5.Q5–Q6, 9.2, 17.12 (+ 4) · approfondie : ch. 5, 17 · réutilisée : ch. 5, 6, 9, 13, 16, 17, 18, 22, 27, B6
- dérivée directionnelle → voir [pente dans une direction, dérivée directionnelle](#p)
- **dérivée numérique** — enseignée : [ch. 5, Tangente et dérivée][s181] · exercices : 5.1, 5.3–4, 5.6, 5.Q4–Q6, 5.Q9, 5.R2–R3 (+ 8) · réutilisée : ch. 18
- **dérivée partielle** (*partial derivative*) — enseignée : [ch. 0B, 101.6.2 · Dérivées partielles][s98] · exercices : 0B.25, 0B.29, 0B.51, 0B.Q11
- **dérivée seconde, courbure** (*second derivative, curvature*) — enseignée : [ch. 5, Tangente et dérivée][s181] · exercices : 5.1, 5.3–4, 5.6, 5.Q4–Q6, 5.Q9, 5.R2–R3 (+ 8)
- **dérivées des activations** — enseignée : [ch. 17, 17.7 · La galerie des activations][s697] · exercices : 17.10, 17.14, 17.18, 17.22, 17.24, 17.Q9–Q10
- **dérivées partielles, gradient, pas de descente** — enseignée : [ch. 0B, 101.6.2 · Dérivées partielles][s98] · exercices : 0B.25, 0B.29, 0B.31, 0B.49–51, 0B.E1, 0B.Q11, 5.Q1, 5.Q7 (+ 81) · approfondie : ch. 5, 14, 18, 19, 22, 28, B6, B8 · réutilisée : ch. 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, B3, B6, B8
- **dérivées usuelles (puissances, exp, ln) et règles (somme, produit, quotient)** — enseignée : [ch. 0B, 101.5.2 · Dérivées usuelles et règles (somme, produit, quotient,…][s683] · exercices : 0B.22, 0B.47, 0B.Q10
- **déroulement** (*unrolling*) — enseignée : [ch. 22][s360] · exercices : 22.Q3
- *descendant* → voir [parent, enfant ; ancêtre, descendant](#p)
- descendant → voir [parent, enfant ; ancêtre, descendant](#p)
- **descente de coordonnées** (*coordinate descent*) — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s231] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 2)
- **descente de gradient** (*gradient descent*) — enseignée : [ch. 0B, 101.6.3 · Gradient et direction de plus grande pente][s41] ; [ch. 5, Suivre la pente jusqu'à un extremum][s13] · exercices : 0B.25, 0B.31, 0B.49–50, 0B.E1, 0B.Q11, 5.Q4–Q6, 5.Q9 (+ 21) · réutilisée : ch. 8, 9, 12, 13, 14, 18, 19, 27, 28
- **descente de gradient stochastique (SGD)** (*stochastic gradient descent*) — enseignée : [ch. 19, 19.4.2 · La descente stochastique (SGD)][s313] · exercices : 18.20, 19.16, 19.23–25, 19.E1, 19.Q8, CP4.2 · réutilisée : ch. 23
- **descente par batch** (*batch gradient descent*) — enseignée : [ch. 19, 19.4.1 · La descente de gradient par batch][s7] · exercices : 19.25, 19.Q8
- **descente par mini-batches** (*mini-batch gradient descent*) — enseignée : [ch. 19, 19.4.3 · La descente par mini-batches][s8] · exercices : 19.25, 19.E5, 19.Q8
- **détecteur de motif** (*feature detector*) — enseignée : [ch. 21, 21.3.1 · Les filtres][s342] · exercices : 21.2, 21.17, 21.19, 21.31, 21.Q4
- **détection d'anomalies** (*anomaly detection*) — enseignée : [ch. 25, 25.5.4 · Une entrée jamais vue][s400] · exercices : 25.17, 25.37, 25.E4, 25.Q11
- **détection d'anomalies par reconstruction** — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s394]
- *determinant* → voir [inverse, déterminant](#i)
- déterminant → voir [inverse, déterminant](#i)
- *deterministic algorithms* → voir [algorithmes déterministes](#a)
- **dette technique (du ML)** (*technical debt*) — enseignée : [ch. B7, La dette technique du ML][s634] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- **device** (*appareil*) — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s327] · exercices : 20.1, 20.13, 20.Q2, 23.12, B4.Q3
- **diagnostic par les journaux RL** — enseignée : [ch. B8, Les journaux de PPO][s639] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- **diagramme de fiabilité** (*reliability diagram*) — enseignée : [ch. 3, Au-delà du livre (3) : la calibration des probabilités][s169]
- **diagramme pente-ordonnée** (*slope-intercept diagram*) — enseignée : [ch. 9, 9.7 · Ajuster une droite avec la règle de Bayes][s233] · exercices : 9.2, 9.7, 9.26–27, 9.Q11
- *dictionary* → voir [dictionnaire](#d)
- **dictionnaire** (*dictionary*) — enseignée : [ch. 0A, 100.3.3 · Dictionnaires][s54] · exercices : 0A.18, 0A.44, 0A.63, 0A.E2, 0A.Q4, B3.Q2 · réutilisée : ch. 26
- **différence centrée** — enseignée : [ch. 5, Au-delà du livre (1) : calculer une dérivée sur ordinateur][s182] · exercices : 5.6
- **différence finie (avant, arrière, centrée)** (*finite difference*) — enseignée : [ch. 5, Au-delà du livre (1) : calculer une dérivée sur ordinateur][s182]
- *difference quotient* → voir [taux d'accroissement](#t)
- *differential privacy* → voir [confidentialité différentielle (DP-SGD)](#c)
- **différentiation automatique** (*automatic differentiation, autodiff*) — enseignée : [ch. 5, Au-delà du livre (3) : la différentiation automatique][s188] · approfondie : ch. 18
- **différentiation automatique en mode inverse** (*reverse-mode automatic differentiation*) — enseignée : [ch. 18, 18.12.3 · Ce qu'elle fait][s308] · exercices : 18.10, 18.26–27, 18.Q9
- *diffusion* → voir [broadcasting](#b)
- **diffusion à variance préservée** (*variance preserving diffusion*) — enseignée : [ch. B5, Un pas de bruit][s550] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 2)
- *diffusion au fil de l'eau* → voir [streaming](#s)
- **diffusion latente** (*latent diffusion*) — enseignée : [ch. B5, 205.9 · Paysage actuel : diffusion latente, transformers de…][s569] · exercices : B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11
- *diffusion model* → voir [modèle de diffusion](#m)
- *diffusion transformer* → voir [transformer de diffusion (DiT)](#t)
- **dilatation** (*dilation, atrous convolution*) — enseignée : [ch. 21, 21.8 · La convolution transposée][s343] · exercices : 21.3, 21.6, 21.15, 21.30, 21.Q9
- *dilation* → voir [dilatation](#d)
- **dimension effective** (*effective dimensionality*) — enseignée : [ch. 15, 15.9.2 · Chercher des hyperparamètres][s289] · exercices : 15.1, 15.5, 15.18, 15.20, 15.Q10
- *dimensionality reduction* → voir [réduction de dimension](#r)
- *direct preference optimization* → voir [DPO](#d)
- *directed acyclic graph* → voir [graphe orienté acyclique (DAG)](#g)
- *directional derivative* → voir [pente dans une direction, dérivée directionnelle](#p)
- *discount factor* → voir [facteur d'actualisation](#f)
- *discounted return* → voir [retour actualisé](#r)
- **discrétisation d'un état continu** — enseignée : [ch. 26][s416]
- **discrétiser** (*discretize*) — enseignée : [ch. 26][s416] · exercices : 26.36
- *discretize* → voir [discrétiser](#d)
- **discriminateur** (*discriminator*) — enseignée : [ch. 27, 27.4.1 · Le discriminateur][s418] · exercices : 27.3, 27.5, 27.12, 27.17, 27.Q6, 28.R1, CP6.4
- **discriminateur optimal** (*optimal discriminator*) — enseignée : [ch. 27, 27.4.4 · Le jeu][s420] · exercices : 27.2, 27.4–5, 27.10, 27.E5, 27.Q5, 27.R1, CP6.4
- **discrimination par mini-batch** (*minibatch discrimination*) — enseignée : [ch. 27, 27.7.2 · L'effondrement de mode][s422] · exercices : 27.23–24, 27.E2, 27.Q10, 27.R2
- *discriminative fine-tuning* → voir [learning rates discriminatifs](#l)
- *discriminative learning rates* → voir [learning rates discriminatifs](#l)
- *discriminator* → voir [discriminateur](#d)
- *distance concentration* → voir [concentration des distances](#c)
- **distance de domaine** (*domain gap*) — enseignée : [ch. B1, 201.7 · Quand le transfert échoue : distance de domaine, transfert…][s452] · exercices : B1.6, B1.9–10, B1.16, B1.22–23, B1.E2, B1.Q8–Q9, B1.R1
- **distance de Fréchet** (*Fréchet distance*) — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s426] ; [ch. B5, 205.8 · Évaluer : distance de Fréchet, diversité, mémorisation][s427] · exercices : 27.22–23, 27.28, B5.10, B5.25, B5.28–29, B5.31, B5.Q10, B5.R2
- **distance de Manhattan** (*Manhattan distance, $L^1$ distance*) — enseignée : [ch. 13, 13.3 · Les k plus proches voisins][s211] · exercices : 13.1, 13.9, 13.13–15, 13.34, 13.36, 13.E1, 13.Q3–Q4
- **distance de Wasserstein (1D)** (*Wasserstein distance, earth mover's distance*) — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s426] · exercices : 27.11
- **distance de Wasserstein 1D** — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s426] · exercices : 27.11
- **distances** — enseignée : [ch. 13, 13.3 · Les k plus proches voisins][s211] · exercices : 13.1, 13.9, 13.13–15, 13.34, 13.36, 13.E1, 13.Q3–Q4 (+ 4) · approfondie : ch. 7
- **distillation** (*knowledge distillation*) — enseignée : [ch. 6, 6.9 · La divergence KL][s199] · exercices : 6.5, 6.7, 6.16, 6.18, 6.25, 6.E2, 6.E4, 6.Q11
- **distillation (de l'échantillonneur)** (*step distillation*) — enseignée : [ch. B5, 205.6 · Accélérer : DDIM déterministe, moins de pas, distillation][s557] · exercices : B5.5, B5.8, B5.17–18, B5.24–25, B5.30–31, B5.Q7–Q8 (+ 1)
- **distillation adversariale** (*adversarial diffusion distillation*) — enseignée : [ch. B5, La distillation][s563] · exercices : B5.5, B5.8, B5.17–18, B5.24–25, B5.30–31, B5.Q7–Q8 (+ 1)
- **distillation progressive** (*progressive distillation*) — enseignée : [ch. B5, La distillation][s563] · exercices : B5.5, B5.8, B5.17–18, B5.24–25, B5.30–31, B5.Q7–Q8 (+ 1)
- *distributed term* → voir [terme distribué](#t)
- **distribution de probabilité, loi** (*probability distribution*) — enseignée : [ch. 2, Des comptages à une distribution de probabilité][s134] · exercices : 0B.52, 2.1–2, 2.13, 2.16, 2.19, 2.E1, 2.Q1–Q2, 2.R2 (+ 2) · approfondie : ch. 0B, 4 · réutilisée : ch. 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 25, B5, B8
- **distribution log-uniforme** (*log-uniform distribution*) — enseignée : [ch. 15, 15.9.4 · La recherche aléatoire][s288] · exercices : 15.5, 15.19–20, 15.E2, 15.Q10
- *distribution shift* → voir [décalage de distribution](#d)
- *distributional hypothesis* → voir [hypothèse distributionnelle](#h)
- DiT → voir [transformer de diffusion (DiT)](#t)
- **divergence de Jensen-Shannon** (*Jensen-Shannon divergence*) — enseignée : [ch. 6, 6.9 · La divergence KL][s199] · exercices : 6.5, 6.7, 6.16, 6.18, 6.25, 6.E2, 6.E4, 6.Q11
- divergence de Kullback-Leibler → voir [divergence KL, divergence de Kullback-Leibler](#d)
- **divergence KL entre gaussiennes** — enseignée : [ch. 25][s721]
- **divergence KL, divergence de Kullback-Leibler** (*KL divergence, relative entropy*) — enseignée : [ch. 6, 6.9 · La divergence KL][s199] · exercices : 6.5, 6.7, 6.16, 6.18, 6.25, 6.E2, 6.E4, 6.Q11, 25.7, 25.R3 (+ 2) · approfondie : ch. 25 · réutilisée : ch. 25, B7, B8
- *Dockerfile* → voir [Dockerfile, couche](#d)
- **Dockerfile, couche** (*Dockerfile, layer*) — enseignée : [ch. B7, 207.9 · Conteneurs et déploiement : Dockerfile, variables…][s626] · exercices : 1.3, 10.5, 16.3, 16.13, 16.17, 16.22, B7.23–25, B7.Q4 (+ 41) · approfondie : ch. 20 · réutilisée : ch. 20, 21, 22, 25, 27
- **docstring** — enseignée : [ch. 0A, 100.5.5 · Docstrings au format NumPy et annotations de type][s61] · exercices : 0A.61, 0A.Q7, 2.31, 10.20 · réutilisée : ch. 10
- **docstrings au format NumPy, annotations de type** — enseignée : [ch. 0A, 100.5.5 · Docstrings au format NumPy et annotations de type][s61] · exercices : 0A.61, 0A.Q7
- **documentation de provenance et de licences** — enseignée : [ch. 28][s349]
- *domain gap* → voir [distance de domaine](#d)
- *domain of discourse* → voir [domaine du discours](#d)
- *domain randomization* → voir [randomisation de domaine](#r)
- **domaine du discours** (*domain of discourse*) — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5, 11.15, 11.Q7
- **domaine public** (*public domain*) — enseignée : [ch. 28, 28.4.5 · Réaliser le transfert][s437] · exercices : 28.4, 28.6, 28.9, 28.20–21, 28.23–24, 28.26, 28.29, 28.Q9 (+ 1) · réutilisée : ch. 29
- **donnée nominale** (*nominal data*) — enseignée : [ch. 12, 12.3 · Les types de données][s262] · exercices : 12.1, 12.14, 12.26, 12.Q2–Q3
- **donnée numérique (quantitative)** (*numerical data*) — enseignée : [ch. 12, 12.3 · Les types de données][s262] · exercices : 12.1, 12.14, 12.26, 12.Q2–Q3
- **donnée ordinale** (*ordinal data*) — enseignée : [ch. 12, 12.3 · Les types de données][s262] · exercices : 12.1, 12.14, 12.26, 12.Q2–Q3
- **données synthétiques** (*synthetic data*) — enseignée : [ch. 24, 24.4.7 · Des données synthétiques][s386] · exercices : 24.24, 24.Q8
- **DoRA** (*weight-decomposed low-rank adaptation*) — enseignée : [ch. B4, QLoRA][s541] · exercices : B4.3, B4.6, B4.11, B4.25–27, B4.E2, B4.Q10–Q11, B4.R2 (+ 1)
- *dot product* → voir [produit scalaire](#p)
- *double descent* → voir [double descente](#d)
- **double descente** (*double descent*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s239] · exercices : 9.4, 9.11, 9.25, 9.30–31, 9.E5, 9.Q9
- *Double DQN* → voir [double estimateur, Double DQN](#d)
- Double DQN → voir [double estimateur, Double DQN](#d)
- **double estimateur, Double DQN** (*double estimator, Double DQN*) — enseignée : [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s234] · exercices : B8.1–2, B8.13–15, B8.E1, B8.Q3–Q4, B8.Q9, B8.R1
- *double estimator* → voir [double estimateur, Double DQN](#d)
- **doublons et fuites entre découpages** — enseignée : [ch. 29][s442]
- DP-SGD → voir [confidentialité différentielle (DP-SGD)](#c)
- **DPO** (*direct preference optimization*) — enseignée : [ch. B8, 208.7 · RL et modèles de langue : RLHF, modèle de récompense, DPO,…][s126] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- DQN → voir [deep RL, DQN](#d)
- **DQN et ses stabilisateurs** — enseignée : [ch. B8, La surestimation, et Double DQN][s640] · exercices : B8.1–2, B8.13–15, B8.E1, B8.Q3–Q4, B8.Q9, B8.R1
- *draw* → voir [tirage](#t)
- *drop-column importance* → voir [importance par retrait](#i)
- **dropout** (*abandon*) — enseignée : [ch. 9, Pénalité sur les poids, dropout, batchnorm][s20] ; [ch. 20, 20.4.3 · Le dropout][s21] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 22) · approfondie : ch. 24 · réutilisée : ch. 21, 24, B3
- **dropout et max-norm en PyTorch** — enseignée : [ch. 24, 24.2.6 · Ajouter du dropout][s382] · exercices : 24.6, 24.13–14, 24.E3, 24.Q3
- **dropout inversé** (*inverted dropout*) — enseignée : [ch. 20, 20.4.3 · Le dropout][s21] · exercices : 20.3, 20.5, 20.16, 20.24–25, 20.28–29, 20.Q5, 20.R1, 20.R3 (+ 2)
- **dtype** (*data type*) — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s327] · exercices : 0A.27, 0A.Q10, 20.1, 20.13, 20.Q2, B4.1, B4.Q3 · approfondie : ch. 0A
- **duck typing** (*typage canard*) — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s287] · exercices : 15.1, 15.12, 15.18, 15.23, 15.E2, 15.Q10
- *dummy variable* → voir [variable indicatrice](#v)
- *dynamic* → voir [quantification dynamique, statique](#q)
- *dynamic batching* → voir [regroupement dynamique](#r)
- *dynamic programming* → voir [programmation dynamique](#p)

## E

- **early stopping** (*arrêt anticipé*) — enseignée : [ch. 9, 9.4 · L'early stopping][s225] ; [ch. 14, Au-delà du livre : le gradient boosting][s226] ; [ch. 20, Au-delà du livre : régulariser un réseau][s227] · exercices : 9.5, 9.9, 9.20, 9.Q6, 20.28, 23.6, 23.25, 23.27, CP5.6, B1.19 · approfondie : ch. 23 · réutilisée : ch. 23, 24, B1
- **early stopping et meilleur checkpoint** — enseignée : [ch. 23, 23.10.3 · Early stopping][s377] · exercices : 23.6, 23.25, 23.27, 23.E4
- *earth mover's distance* → voir [distance de Wasserstein (1D)](#d)
- **écart de généralisation** (*generalization gap*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s26] · exercices : 9.1, 9.12, 9.14, 9.E2, 9.Q1
- **écart interquartile** (*interquartile range, IQR*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s264] · exercices : 12.11, 12.Q4
- **écart simulation-réalité, sim-to-real** (*reality gap, sim-to-real*) — enseignée : [ch. B8, 208.8 · Limites : efficacité en échantillons, généralisation,…][s648] · exercices : B8.8, B8.23–24, B8.Q12
- écart-type → voir [variance, écart-type](#v)
- **écart-type par mini-batch** (*minibatch standard deviation*) — enseignée : [ch. 27, 27.5 · Un GAN en action][s423] · exercices : 27.4, 27.11, 27.13–15, 27.Q7
- **échantillon** (*sample*) — enseignée : [ch. 1, 1.2.2 · La version pour ordinateur : échantillons, features, labels][s12] · exercices : 0B.E5, 1.5, 1.9, 1.14, 1.16–17, 1.Q3, 1.R1, 2.10, CP1.13 (+ 8)
- **échantillon biaisé** (*biased sample*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s257] · exercices : 2.10, 11.17–18, 11.E4, 11.Q10
- **échantillonnage ancestral** (*ancestral sampling*) — enseignée : [ch. B5, L'algorithme][s554] · exercices : B5.4, B5.7, B5.11, B5.16–19, B5.22, B5.24, B5.Q7 (+ 1)
- **échantillonnage avec ou sans remise** — enseignée : [ch. 2, 2.5.1 · Avec remise, 2.5.2 Sans remise][s687] · exercices : 2.21, 2.Q9
- **échantillonnage DDPM** — enseignée : [ch. B5, 205.5 · Échantillonnage DDPM : moyenne du posterior, variance, T pas][s556] · exercices : B5.4, B5.7, B5.11, B5.16–19, B5.22, B5.24, B5.Q7 (+ 1)
- **échantillonnage de Thompson** (*Thompson sampling*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s121] · exercices : 11.2, 11.7–8, 11.10–11, 11.13, 11.19–21, 11.R1 (+ 8)
- **échantillonnage négatif** (*negative sampling*) — enseignée : [ch. B2, word2vec : prédire plutôt que compter][s482] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- **échantillonneur** (*sampler*) — enseignée : [ch. B5, Ce que coûte un échantillon][s555] · exercices : B5.4, B5.7, B5.11, B5.16–19, B5.22, B5.24, B5.Q7 (+ 2)
- **échantillonneur-bloqueur** (*sample-and-hold*) — enseignée : [ch. 16, 16.3 · Flux synchrone et asynchrone][s295] · exercices : 16.5, 16.16, 16.E4, 16.Q5–Q6
- **échelle de guidance** (*guidance scale*) — enseignée : [ch. B5, La guidance sans classifieur][s566] · exercices : B5.1, B5.26–27, B5.E4, B5.Q9
- **échelle des logits (CLIP)** (*logit scale*) — enseignée : [ch. B1, 201.8 · Modèles de fondation : auto-supervisé (DINOv2),…][s455] · exercices : 12.17, B1.5, B1.15, B1.24–25, B1.E4, B1.Q10–Q11, B8.20, B8.Q8 (+ 1)
- *écrêtage du gradient* → voir [gradient clipping](#g)
- **écriture atomique** (*atomic write*) — enseignée : [ch. B4][s544]
- *EDA* → voir [analyse exploratoire](#a)
- *editable install* → voir [installation éditable](#i)
- *effective dimensionality* → voir [dimension effective](#d)
- **efficacité en échantillons** (*sample efficiency*) — enseignée : [ch. B8, 208.8 · Limites : efficacité en échantillons, généralisation,…][s648] · exercices : B8.8, B8.23–24, B8.Q12
- **efficacité, symétrie, joueur nul, additivité** (*efficiency, symmetry, null player, additivity*) — enseignée : [ch. B6, Quatre propriétés, une seule solution][s580] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5 (+ 3) · approfondie : ch. 0B, 16
- *efficiency* → voir [efficacité, symétrie, joueur nul, additivité](#e)
- **effondrement (d'un réseau linéaire)** (*collapse of a linear network*) — enseignée : [ch. 17, 17.2 · Ce que fait une fonction d'activation][s301] · exercices : 17.1, 17.3, 17.9, 17.11, 17.20, 17.Q1, 25.34, 27.24, 27.E2, 27.Q10 (+ 1) · approfondie : ch. 27 · réutilisée : ch. B5
- **effondrement de mode** (*mode collapse*) — enseignée : [ch. 27, 27.7.2 · L'effondrement de mode][s422] · exercices : 27.23–24, 27.E2, 27.Q10, 27.R2, B5.R2 · réutilisée : ch. B5
- **effondrement du posterior** (*posterior collapse*) — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s394] · exercices : 25.34
- **égalité des chances** (*equal opportunity*) — enseignée : [ch. B6, 206.9 · Équité : indicateurs de groupe (parité démographique,…][s596] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- *égalité des taux d'erreur* → voir [equalized odds](#e)
- *eigenvalue* → voir [vecteur propre, valeur propre](#v)
- *eigenvector* → voir [vecteur propre, valeur propre](#v)
- élagage → voir [pré-élagage, élagage](#p)
- **élagage coût-complexité** (*minimal cost-complexity pruning*) — enseignée : [ch. 13, 13.5.3 · Limiter l'overfitting][s276] · exercices : 13.19, 13.21, 13.23–25, 13.E3, 13.Q8, 13.Q10
- *élan* → voir [momentum](#m)
- **Elastic Net** — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s231] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 2)
- *elastic weight consolidation* → voir [EWC](#e)
- **ELBO** (*evidence lower bound*) — enseignée : [ch. 25, 25.9.2 · La structure d'un VAE][s399] · exercices : 25.5, 25.13, 25.28, 25.30, 25.38, 25.E3, 25.Q12, 25.R3 · réutilisée : ch. B5
- *elbow method* → voir [méthode du coude](#m)
- *element-wise product* → voir [produit de Hadamard](#p)
- *elementwise* → voir [par échantillon, par feature, par élément](#p)
- **ELU** (*exponential linear unit*) — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s299] · exercices : 17.2, 17.5, 17.12, 17.18, 17.20, 17.23–24, 17.Q6–Q7, 17.R1 (+ 3)
- *EMA* → voir [moyenne mobile exponentielle](#m)
- **EMA des poids** — enseignée : [ch. B5, En pratique : la moyenne exponentielle des poids][s736] · exercices : B5.7, B5.11, B5.16, B5.19, B5.23, B5.E1–E2, B5.Q4, B5.R3
- **embedding** (*plongement, représentation vectorielle*) — enseignée : [ch. 12, 12.7.2 · Standardisation et PCA pour des images][s19] · exercices : 12.23–24, 12.28, 12.Q9, 24.30, B1.2, B1.13, B1.Q11, B2.Q9, B2.Q12 (+ 8) · approfondie : ch. B2 · réutilisée : ch. 24, 28, B2, B3, B4
- **embedding de phrase** (*sentence embedding*) — enseignée : [ch. B2, Deux modèles d'embedding de phrases][s487] · exercices : B2.28–29, B2.E4–E5, B2.Q11, B2.R1
- **embedding figé** (*frozen embedding*) — enseignée : [ch. B1, Calculer les embeddings une fois][s446] · exercices : B1.2, B1.4, B1.13, B1.15, B1.E1, B1.Q5, B1.R2
- **embedding statique, contextuel** (*static, contextual embedding*) — enseignée : [ch. B2, 202.10 · Embeddings contextuels et embeddings de phrases ;…][s485] · exercices : B2.28–29, B2.E4–E5, B2.Q11, B2.R1
- **embeddings de phrases** — enseignée : [ch. B2, 202.10 · Embeddings contextuels et embeddings de phrases ;…][s485] · exercices : B2.28–29, B2.E4–E5, B2.Q11, B2.R1 · réutilisée : ch. B4
- *embodied cognition* → voir [cognition incarnée](#c)
- **empreinte (des données)** (*data fingerprint*) — enseignée : [ch. B7, 207.4 · Reproductibilité : graines, déterminisme, versions figées,…][s613] · exercices : B7.2, B7.12–13, B7.17, B7.25, B7.E2, B7.Q5–Q6, B7.R3
- en ligne → voir [inférence par batches, en ligne](#i)
- en ombre → voir [déploiement canari, en ombre](#d)
- **encodage de position** (*positional encoding*) — enseignée : [ch. B3, L'encodage sinusoïdal (2017)][s503] · exercices : B3.4, B3.7, B3.11, B3.15, B3.25, B3.E5, B3.Q7
- **encodage de position rotatif (RoPE)** (*rotary position embedding*) — enseignée : [ch. B3, 203.5 · Positions : encodage sinusoïdal, positions apprises, RoPE][s504] · exercices : B3.4, B3.7, B3.11, B3.15, B3.25, B3.E5, B3.Q7
- **encodage du pas de temps** (*timestep embedding*) — enseignée : [ch. B5, 205.4 · Le débruiteur : U-Net minuscule et encodage du pas de temps][s560] · exercices : B5.11, B5.19, B5.21, B5.Q5–Q6, B5.R1
- *encodage one-hot* → voir [one-hot](#o)
- **encodage ordinal** (*ordinal encoding*) — enseignée : [ch. 12, 12.3 · Les types de données][s262] · exercices : 12.1, 12.14, 12.26, 12.Q2–Q3
- **encodage par la cible** (*target encoding*) — enseignée : [checkpoint 2][s661]
- **encodages de position (sinusoïdal, RoPE)** — enseignée : [ch. B3, Les positions apprises][s727] · exercices : B3.4, B3.7, B3.11, B3.15, B3.25, B3.E5, B3.Q7
- *encoder* → voir [encodeur, décodeur](#e)
- *encoder-decoder* → voir [encodeur-décodeur](#e)
- **encodeur, décodeur** (*encoder, decoder*) — enseignée : [ch. 25, 25.2 · Introduction : comprimer][s392] · exercices : 25.9, 25.12, 25.35, 25.Q1, 26.R1, B3.E3, B4.28 · réutilisée : ch. B4
- **encodeur-décodeur** (*encoder-decoder, seq2seq*) — enseignée : [ch. B3, 203.1 · Des RNN à l'attention : goulot de l'encodeur-décodeur et…][s491] · exercices : B3.E3, B3.Q1, B3.R2
- *endpoint* → voir [API HTTP, route](#a)
- enfant → voir [parent, enfant ; ancêtre, descendant](#p)
- **ensemble** (*set*) — enseignée : [ch. 0A, 100.3.4 · Ensembles][s55] ; [ch. 14, 14.2 · Les ensembles][s56] · exercices : 0A.19, 0A.E2, 0A.Q4, 0B.5, 0B.Q5, 14.5, 14.9, 14.13, 14.Q1–Q2 (+ 6) · approfondie : ch. 0B · réutilisée : ch. 3, 11, 15
- **ensemble (de modèles)** (*ensemble, ensemble learning*) — enseignée : [ch. 14, 14.2 · Les ensembles][s56] · exercices : 0A.19, 0A.E2, 0B.5, 0B.Q5, 3.R2, 14.5, 14.9, 14.13, 14.Q1–Q2 (+ 5) · approfondie : ch. 0A, 0B · réutilisée : ch. 3, 11, 15
- **ensemble de prompts** (*prompt ensembling*) — enseignée : [ch. B1, Le zéro-shot][s456] · exercices : B1.5, B1.15, B1.24–25, B1.E4, B1.Q10–Q11
- *ensemble learning* → voir [ensemble (de modèles)](#e)
- **ensembles : union, intersection, complémentaire, cardinal** — enseignée : [ch. 0B, 101.1.6 · Ensembles : union, intersection, complémentaire, cardinal][s81] · exercices : 0B.5, 0B.11, 0B.Q5, 3.R2 · réutilisée : ch. 3
- **entraînement** (*training*) — enseignée : [ch. 1, 1.2.2 · La version pour ordinateur : échantillons, features, labels][s12] · exercices : 1.5, 1.9, 1.14, 1.16–17, 1.Q3, 1.R1, 8.Q1, 8.Q3, 8.Q10 (+ 47) · approfondie : ch. 8, 18, 20, 27 · réutilisée : ch. 5, 9, 12, 19, 21, 22, 23, 25, 26, 29, B2, B4, B5, B8
- **entraînement adversarial** (*adversarial training*) — enseignée : [ch. 21, Au-delà du livre : l'attaque FGSM][s352]
- **entraînement fumée** (*smoke test*) — enseignée : [ch. B7, Quatre familles de tests][s607] · exercices : B7.15, B7.23, B7.25, B7.E3, B7.Q7
- *entrées* → voir [fan-in, fan-out](#f)
- **entropie (de Shannon)** (*entropy, Shannon*) — enseignée : [ch. 6, 6.7 · L'entropie][s196] ; [ch. 13, 13.5.2 · Choisir les splits][s197] · exercices : 6.3, 6.6, 6.8–10, 6.12, 6.14, 6.19, 6.Q8–Q9 (+ 18) · réutilisée : ch. 7, 17, B2
- **entropie conditionnelle** (*conditional entropy*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s202]
- *entropy* → voir [entropie (de Shannon)](#e)
- *entropy bonus* → voir [bonus d'entropie](#b)
- *entry* → voir [composante](#c)
- *entry point* → voir [point d'entrée](#p)
- *environment* → voir [agent, environnement, action, récompense](#a)
- *environment step budget* → voir [budget de pas (d'environnement)](#b)
- *environment variable* → voir [variable d'environnement](#v)
- environnement → voir [agent, environnement, action, récompense](#a)
- **environnements vectorisés** (*vectorized environments*) — enseignée : [ch. B8, 208.5 · PPO : ratio de probabilités, objectif clippé, entropie,…][s645] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- **environnements vectorisés gymnasium** — enseignée : [ch. B8, Les environnements vectorisés][s646] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- *episode* → voir [épisode](#e)
- **épisode** (*episode*) — enseignée : [ch. 26, 26.3 · La structure du RL][s402] · exercices : 26.13, 26.33, 26.Q3–Q4, B8.14, B8.28, B8.Q10
- épisode tronqué → voir [état terminal, épisode tronqué](#e)
- **epoch** (*époque*) — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s6] ; [ch. 8, 8.2 · La boucle d'entraînement][s9] ; [ch. 18, 18.11.1 · Explorer le learning rate][s10] · exercices : 0A.5, 0A.31, 0A.66, 0A.E4, 0A.Q11, 8.16, 8.Q1–Q2, 8.R2, 18.Q8 (+ 20) · réutilisée : ch. 19
- **epsilon machine** (*machine epsilon*) — enseignée : [ch. 5][s184]
- *epsilon-greedy* → voir [ε-greedy](#g)
- *equal opportunity* → voir [égalité des chances](#e)
- **equalized odds** (*égalité des taux d'erreur*) — enseignée : [ch. B6, 206.9 · Équité : indicateurs de groupe (parité démographique,…][s596] · exercices : B6.5–6, B6.10, B6.23–25, B6.29, B6.E3, B6.Q11, B6.R3
- **équation de Bellman** (*Bellman equation*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s405] · exercices : 26.1, 26.7–8, 26.11, 26.14–15, 26.32, 26.E2, 26.Q6
- **équations normales** (*normal equations*) — enseignée : [ch. 9, Les moindres carrés (au-delà du livre)][s232] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 2)
- **équilibre de Nash** (*Nash equilibrium*) — enseignée : [ch. 27, 27.4.4 · Le jeu][s420] · exercices : 27.2, 27.4–5, 27.10, 27.E5, 27.Q5, 27.R1
- **équité par aveuglement** (*fairness through unawareness*) — enseignée : [mini-projet 3][s279]
- *equivalence test* → voir [test d'équivalence](#t)
- **équivariance par permutation** (*permutation equivariance*) — enseignée : [ch. B3][s502]
- *erreur absolue moyenne* → voir [MSE, RMSE, MAE](#m)
- *erreur absolue moyenne en pourcentage* → voir [MAPE](#m)
- **erreur d'arrondi** (*round-off error*) — enseignée : [ch. 5, Au-delà du livre (2) : les arrondis et le choix du pas][s183] · exercices : 5.13
- **erreur d'entraînement, erreur de généralisation** (*training error, generalization error*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s26] · exercices : 9.1, 9.12, 9.14, 9.E2, 9.Q1
- erreur de généralisation → voir [erreur d'entraînement, erreur de généralisation](#e)
- **erreur de reconstruction** (*reconstruction error*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s152] · exercices : 12.7–8, 12.22–23, 12.27, 12.Q9, 25.22, 25.37
- **erreur de troncature** (*truncation error*) — enseignée : [ch. 5, Au-delà du livre (1) : calculer une dérivée sur ordinateur][s182] · exercices : 5.13
- **erreur du procureur** (*prosecutor's fallacy*) — enseignée : [ch. 4][s176] · exercices : 4.10
- **erreur du taux de base** (*base rate fallacy*) — enseignée : [ch. 3, 3.8 · Appliquer la matrice de confusion : le piège de la prévalence][s162] · exercices : 3.7, 3.10, 3.21, 3.26–27, 3.E1, 3.Q12
- *erreur quadratique moyenne* → voir [MSE, RMSE, MAE](#m)
- **erreur relative (d'un gradient)** (*relative error*) — enseignée : [ch. 18, Au-delà du livre : vérifier un gradient][s109]
- **erreur TD** (*temporal-difference error*) — enseignée : [ch. 26, 26.6.1 · Les valeurs Q et leur mise à jour][s406] ; [ch. B8, 208.4 · Acteur-critique : fonction de valeur, erreur TD, avantage, GAE][s409] · exercices : 26.3, 26.8, 26.20, 26.Q8, B8.4, B8.18, B8.20, B8.Q7
- **erreur type** (*standard error*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s145] · exercices : 2.6, 2.9–10, 2.22–24, 2.E3, 2.Q10, 8.4 · approfondie : ch. 8
- **erreur type d'un score** — enseignée : [ch. 8, 8.2.1 · Pourquoi le score d'entraînement trompe][s219] · exercices : 8.9, 8.12, 8.E4, 8.Q3–Q4, 8.R3
- *error analysis* → voir [analyse des erreurs](#a)
- *error rate* → voir [taux d'erreur](#t)
- *error term* → voir [delta (d'un neurone)](#d)
- **espace de recherche** (*search space*) — enseignée : [ch. 24, 24.3.4 · La recherche d'hyperparamètres][s383] · exercices : 24.1, 24.7, 24.9, 24.18, 24.E1, 24.Q5
- **espace latent** (*latent space*) — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s394] · exercices : 25.10, 25.21, 25.E5, 25.Q7, 27.26, B5.8 · réutilisée : ch. B5
- **espérance** (*expected value, expectation*) — enseignée : [ch. 0B, 101.7.3 · Variable aléatoire discrète et espérance][s103] ; [ch. 2, 2.3.5 · L'espérance][s104] · exercices : 0B.27–28, 0B.53, 0B.E5, 0B.Q12, 2.5, 2.Q7, 2.R3, 20.5, 24.6 · réutilisée : ch. 4, 6, 11, 20, 25, 26, B8
- *essaim d'abeilles* → voir [beeswarm](#b)
- **estimateur (scikit-learn)** (*estimator*) — enseignée : [ch. 15, 15.4 · Les estimateurs][s285] · exercices : 8.16, 9.R3, 15.8–9, 15.29, 15.Q3, 15.Q9, 16.R1, 24.16 · approfondie : ch. 24 · réutilisée : ch. 9, 14, 29
- **estimateur *straight-through*** (*straight-through estimator, STE*) — enseignée : [ch. 10][s245]
- **estimateur scikit-learn autour d'un réseau PyTorch** — enseignée : [ch. 24, 24.3.1 · Habiller un réseau en estimateur][s714] · exercices : 24.16, 24.Q5
- *estimator* → voir [estimateur (scikit-learn)](#e)
- étape → voir [workflow, job, étape, action](#w)
- **étape de mise à jour** (*update step*) — enseignée : [ch. 18, 18.4 · Sorties des neurones et erreur du réseau][s306] · exercices : 18.1, 18.Q4
- **état caché** (*hidden state*) — enseignée : [ch. 22, 22.3 · L'état][s358] · exercices : 22.9, 22.Q2, B7.Q1
- **état caché (d'un notebook)** (*hidden state*) — enseignée : [ch. B7, Ce que cache un notebook][s604] · exercices : 22.9, B7.6, B7.8–9, B7.25, B7.E1, B7.Q1–Q2
- **état porté** (*stateful RNN, carried state*) — enseignée : [ch. 24, 24.5.9 · Les RNN « stateful »][s389] · exercices : 24.27, 24.Q11
- **état terminal, épisode tronqué** (*terminal state, truncation*) — enseignée : [ch. 26, 26.6.1 · Les valeurs Q et leur mise à jour][s406] · exercices : 26.3, 26.8, 26.20, 26.Q8, B8.1
- **état, transition** (*state, transition*) — enseignée : [ch. 26, 26.3.1 · Premier temps : l'agent choisit une action][s401] · exercices : 22.9, 22.Q2, 22.Q4, 22.Q12, 24.27, 25.R2, 26.4, 26.13, 26.Q3, 26.Q8 (+ 7) · approfondie : ch. 24
- *Euler method* → voir [méthode d'Euler](#m)
- *evaluation* → voir [évaluation](#e)
- **évaluation** (*evaluation*) — enseignée : [ch. 11, 11.2.2 · L'évaluation][s248] · exercices : 8.23, 11.12, 11.Q1, 11.Q4, CP2.5, 12.R1, 16.4, 16.Q7, 18.R3, CP4.4 (+ 4) · approfondie : ch. B4, B5, B8
- **évaluation de LLM** — enseignée : [ch. B4, Ce que fait vraiment un LLM][s731] · exercices : B4.10, B4.Q1–Q2, B4.Q7
- *evaluation mode* → voir [mode entraînement, mode évaluation](#m)
- **évaluation multi-graines** — enseignée : [ch. B8, 208.6 · Pratique du deep RL : hyperparamètres, graines,…][s740] · exercices : B8.8, B8.15, B8.21–22, B8.27–28, B8.E3, B8.E5, B8.Q10
- **évaluation par distance de Fréchet** — enseignée : [ch. B5, La distance de Fréchet][s737] · exercices : B5.10, B5.25, B5.28–29, B5.31, B5.Q10, B5.R2
- **événement, issue** (*event, outcome*) — enseignée : [ch. 0B, 101.7.1 · Expérience aléatoire, événement, probabilité, complémentaire][s101] · exercices : 0B.11, 0B.52, 0B.Q12, 6.R2 · réutilisée : ch. 2, 14
- *event* → voir [événement, issue](#e)
- *evidence* → voir [évidence](#e)
- **évidence** (*evidence, marginal likelihood*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s173] · exercices : 4.3, 4.10, 4.14, 4.23, 4.E3, 4.Q6–Q7
- *evidence lower bound* → voir [ELBO](#e)
- **EWC** (*elastic weight consolidation*) — enseignée : [ch. B1, L'oubli catastrophique][s451] · exercices : B1.6, B1.9–10, B1.16, B1.22–23, B1.E2, B1.Q8–Q9, B1.R1
- **exact match** (*correspondance exacte*) — enseignée : [ch. B4, Les métriques][s530] · exercices : B4.2, B4.15–16, B4.18, B4.23, B4.29, B4.E4, B4.Q8
- **exception** — enseignée : [ch. 0A, 100.6.2 · Exceptions : lire un traceback, `raise`, `try`/`except`][s62] · exercices : 0A.25, 0A.37, 0A.Q8, 16.Q4
- **exception écrasante** (*overwhelming exception*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s257] · exercices : 11.17–18, 11.E4, 11.Q10
- **exceptions (raise, try/except), lecture d'un traceback** — enseignée : [ch. 0A, 100.6.2 · Exceptions : lire un traceback, `raise`, `try`/`except`][s62] · exercices : 0A.25, 0A.37, 0A.Q8
- *excessive agency* → voir [agentivité excessive](#a)
- *exclusive or* → voir [XOR (ou exclusif)](#x)
- *exec form* → voir [forme exec, forme shell](#f)
- *exécution* → voir [run](#r)
- **exemple adversarial** (*adversarial example*) — enseignée : [ch. 21, 21.10 · Les adversaires][s351] · exercices : 21.10, 21.29, 21.E4, 21.Q11
- *exit code* → voir [code de sortie](#c)
- *expectation* → voir [espérance](#e)
- *expectation-maximization* → voir [algorithme EM](#a)
- *expected SARSA* → voir [SARSA espéré](#s)
- *expected value* → voir [espérance](#e)
- **experience replay** — enseignée : [ch. 26, 26.9 · Le rejeu d'expérience][s413] · exercices : 26.12, 26.31, 26.33, 26.E4, 26.Q12
- *experiment log* → voir [journal d'expériences](#j)
- *experiment tracking* → voir [journal d'expériences](#j)
- *expert system* → voir [système expert](#s)
- *explainability* → voir [explicabilité](#e)
- *explained variance* → voir [variance expliquée (d'un critique)](#v)
- *explained variance ratio* → voir [part de variance expliquée](#p)
- **explicabilité** (*explainability, interpretability*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s336] · exercices : 20.11, 20.Q12, B6.28, PF.9 · approfondie : ch. B6
- **explicabilité et dérive** — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s336] · exercices : 20.11, 20.Q12
- **explication contrefactuelle** (*counterfactual explanation*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s336] · exercices : 20.11, 20.Q12
- **explication globale, explication locale** (*global, local explanation*) — enseignée : [ch. B6, Global ou local][s575] · exercices : B6.Q1–Q2
- explication locale → voir [explication globale, explication locale](#e)
- **explication post hoc** (*post-hoc explanation*) — enseignée : [ch. 13, 13.7 · Bilan : choisir un classifieur][s273] · exercices : 13.9, 13.11–12, 13.23, 13.33, 13.35–36, 13.E5, 13.Q12
- *exploding gradient* → voir [gradient évanescent, explosif](#g)
- *exploitation* → voir [exploration, exploitation](#e)
- exploitation → voir [exploration, exploitation](#e)
- *exploration* → voir [exploration, exploitation](#e)
- **exploration / exploitation** — enseignée : [ch. 1][s111] · exercices : 11.E1, 26.10, 26.E3, 26.Q5, B8.14, B8.R3, PF.3 · réutilisée : ch. 11, B8
- **exploration, exploitation** (*exploration, exploitation*) — enseignée : [ch. 26, 26.6.2 · La politique du Q-learning][s260] · exercices : 11.E1, 26.5, 26.10, 26.19, 26.E3, 26.Q5, 26.Q9, 26.R3, B8.14, B8.R3 (+ 1) · réutilisée : ch. 11, B8
- *exploratory data analysis* → voir [analyse exploratoire](#a)
- explosif → voir [gradient évanescent, explosif](#g)
- *exponential linear unit* → voir [ELU](#e)
- *exponential moving average* → voir [moyenne mobile exponentielle](#m)
- **exponentielle et logarithmes (ln, log₂, log₁₀), changement de base, bits et nats** — enseignée : [ch. 0B, 101.2.4 · Logarithmes : ln, log₂, log₁₀, changement de base][s82] · exercices : 0B.15–16, 0B.36–37, 0B.E3, 0B.Q7 · approfondie : ch. 6 · réutilisée : ch. 6, 18
- *export* → voir [exporter (un modèle), `torch.export`](#e)
- **exporter (un modèle), `torch.export`** (*export*) — enseignée : [ch. B7, 207.8 · Exporter et optimiser l'inférence : torch.export, ONNX,…][s623] · exercices : 25.35, B7.1, B7.19–21, B7.25, B7.E5, B7.Q9–Q10
- *exposure bias* → voir [biais d'exposition](#b)
- **expression régulière** (*regular expression*) — enseignée : [ch. 0A, 100.6.5 · Expressions régulières (`re`) : les bases][s67] · exercices : 0A.45, 0A.Q8
- **expressions régulières (re) : bases** — enseignée : [ch. 0A, 100.6.5 · Expressions régulières (`re`) : les bases][s67] · exercices : 0A.45, 0A.Q8
- *external validation* → voir [validation externe](#v)
- **extraction de features** (*feature extraction*) — enseignée : [ch. B1, 201.3 · Extraction de features : embeddings figés, linear probe, k…][s445] · exercices : B1.2, B1.4, B1.13, B1.15, B1.E1, B1.Q5, B1.R2, B4.R2 · réutilisée : ch. B4
- **ExtraTrees** (*extremely randomized trees*) — enseignée : [ch. 14, 14.6 · ExtraTrees][s283] · exercices : 14.15–16, 14.18, 14.Q7
- **extrema locaux et globaux** — enseignée : [ch. 5, Extrema globaux et locaux][s180] · exercices : 5.1, 5.3–4, 5.6, 5.Q4–Q6, 5.Q9, 5.R2–R3 (+ 8)
- *extremely randomized trees* → voir [ExtraTrees](#e)
- extremum global → voir [extremum local, extremum global](#e)
- **extremum local, extremum global** (*local extremum, global extremum*) — enseignée : [ch. 5, Extrema globaux et locaux][s180] · exercices : 5.1, 5.3–4, 5.6, 5.Q4–Q6, 5.Q9, 5.R2–R3 (+ 8)

## F

- **F-beta** (*F-beta score*) — enseignée : [ch. 3, 3.7.11 · Le score F1][s164] · exercices : 3.2, 3.6, 3.8, 3.16, 3.25, 3.E4, 3.Q12
- *F-beta score* → voir [F-beta](#f)
- **F1** — enseignée : [ch. 3, 3.7.11 · Le score F1][s164] · exercices : 3.2, 3.6, 3.8, 3.16, 3.25, 3.E4, 3.Q12, CP3.11
- **F1-score** (*score F1*) — enseignée : [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s33] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11
- **facteur d'actualisation** (*discount factor*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s405] · exercices : 26.1, 26.7, 26.11, 26.14–15, 26.32, 26.E2, 26.Q6
- *factorial* → voir [factorielle](#f)
- **factorielle** (*factorial*) — enseignée : [ch. 0B, 101.1.7 · Dénombrement : principe multiplicatif, factorielle,…][s11] · exercices : 0B.12, 0B.33, 0B.Q5 · réutilisée : ch. B6
- *fairness through unawareness* → voir [équité par aveuglement](#e)
- *fallacy* → voir [sophisme](#s)
- *false discovery rate* → voir [taux de fausses découvertes, de fausses omissions](#t)
- *false negative* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *false negative rate* → voir [taux de faux positifs, de faux négatifs](#t)
- *false omission rate* → voir [taux de fausses découvertes, de fausses omissions](#t)
- *false positive* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *false positive rate* → voir [taux de faux positifs, de faux négatifs](#t)
- **familles d'apprentissage** — enseignée : [ch. 1, 1.5 · Générateurs][s118] · exercices : 1.24, 1.E4, 1.Q9, 1.Q11 · réutilisée : ch. 2
- **fan-in, fan-out** (*entrées, sorties d'une couche*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s294] · exercices : 16.2, 16.4, 16.9, 16.16–18, 16.Q1–Q4 (+ 2)
- fan-out → voir [fan-in, fan-out](#f)
- *fast gradient sign method* → voir [FGSM, PGD](#f)
- *FastAPI* → voir [FastAPI, uvicorn, pydantic](#f)
- **FastAPI, uvicorn, pydantic** (*FastAPI, uvicorn, pydantic*) — enseignée : [ch. B7, 207.3 · Configuration : dataclasses, YAML ou JSON, surcharges,…][s610] · exercices : B7.11, B7.18, B7.25, B7.E1, B7.Q4
- faux négatif → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- faux positif → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *FDR* → voir [taux de fausses découvertes, de fausses omissions](#t)
- **feature** (*caractéristique, variable explicative*) — enseignée : [ch. 0A, 100.6.4 · Sérialiser : `json` et `pickle`][s4] · exercices : 0A.39, 0A.46, 0A.Q8, 1.9, 1.Q3, 7.30, 7.Q9, 8.25, 8.27, 9.30 (+ 20) · approfondie : ch. 9, 12, B1 · réutilisée : ch. 15, 21, B4
- **feature bagging** (*tirage de features*) — enseignée : [ch. 14, 14.5 · Les forêts aléatoires][s282] · exercices : 14.15–16, 14.18–19, 14.28, 14.E2, 14.E4, 14.Q6
- *feature detector* → voir [détecteur de motif](#d)
- **feature engineering** (*ingénierie des features*) — enseignée : [ch. 1, 1.1.2 · Les systèmes experts et leurs limites][s110] · exercices : 1.15, 1.18, 1.25, 1.Q2
- *feature extraction* → voir [extraction de features](#e)
- **feature learning** (*apprentissage des features*) — enseignée : [ch. 1, 1.7 · Deep learning][s27] · exercices : 1.1, 1.3, 1.10, 1.23, 1.E4, 1.Q10–Q11
- *feature map* → voir [carte d'activation](#c)
- *feature selection* → voir [sélection de features](#s)
- *feature visualization* → voir [visualisation de features](#v)
- **features aléatoires** (*random features*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s239] · exercices : 9.4, 9.11, 9.25, 9.30–31, 9.E5, 9.Q9
- **features polynomiales** (*polynomial features*) — enseignée : [ch. 9, Ajuster une courbe : complexité et capacité][s236] ; [ch. 15, 15.9.5 · Les pipelines][s16] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 11)
- *featurewise* → voir [par échantillon, par feature, par élément](#p)
- *feed-forward network* → voir [réseau feed-forward, MLP](#r)
- *feedback* → voir [rétroaction](#r)
- *feedback loop* → voir [boucle de rétroaction](#b)
- **fenêtre glissante** (*sliding window*) — enseignée : [ch. 22, 22.2 · Introduction : des fenêtres pour une couche dense][s356] · exercices : 22.1, 22.13–14, 22.Q1, 22.R1
- **fenêtre glissante (attention)** (*sliding window attention*) — enseignée : [ch. B3, Les autres voies vers les contextes longs][s515] · exercices : B3.8, B3.10, B3.27–28, B3.E2, B3.Q12
- **fenêtres de séries temporelles** — enseignée : [ch. 22, 22.2 · Introduction : des fenêtres pour une couche dense][s356] · exercices : 22.1, 22.13–14, 22.Q1, 22.R1
- **fermeture** (*closure*) — enseignée : [ch. 0A, 100.5.4 · Fermetures (*closures*)][s60] · exercices : 0A.42–43, 0A.Q7 · réutilisée : ch. 18
- **fermetures (closures)** — enseignée : [ch. 0A, 100.5.4 · Fermetures (*closures*)][s60] · exercices : 0A.42–43, 0A.Q7 · réutilisée : ch. 18
- **fertilité** (*fertility*) — enseignée : [ch. B2, 202.6 · Coûts et biais de la tokenisation : fertilité, contexte,…][s473] · exercices : B2.8, B2.11, B2.14, B2.20–21, B2.31, B2.Q8, B2.R3
- **fertilité et coût des tokens** — enseignée : [ch. B2, 202.6 · Coûts et biais de la tokenisation : fertilité, contexte,…][s473] · exercices : B2.8, B2.11, B2.14, B2.20–21, B2.31, B2.Q8, B2.R3
- *fertility* → voir [fertilité](#f)
- feuille → voir [racine, feuille, nœud interne](#r)
- **few-shot** (*à quelques exemples*) — enseignée : [ch. B1, 201.6 · Peu de données : courbes d'apprentissage, augmentation,…][s453] · exercices : B1.20–21, B1.25, B1.28, B1.E1–E2, B1.Q8, B4.15, B4.Q6
- **few-shot par prototypes** — enseignée : [ch. B1, Le few-shot et les prototypes de classe][s454] · exercices : B1.20–21, B1.25, B1.28, B1.E1–E2, B1.Q8
- FFN → voir [bloc feed-forward (FFN)](#b)
- **FGSM, PGD** (*fast gradient sign method, projected gradient descent*) — enseignée : [ch. 21, Au-delà du livre : l'attaque FGSM][s352] · exercices : 21.29, B6.R1 · réutilisée : ch. B6
- *fiche de données* → voir [data card](#d)
- **fiche modèle** (*model card*) — enseignée : [ch. 8, 8.3 · Le jeu de test][s122] ; [ch. 23, 23.9.5 · Sauvegarder le prétraitement][s223] ; [ch. B6, La fiche modèle][s224] · exercices : 8.1, 8.3–4, 8.7, 8.10–13, 8.Q5–Q6 (+ 15) · approfondie : ch. B1 · réutilisée : ch. B1
- **fiche modèle d'un modèle dérivé** — enseignée : [ch. B1, Publier un modèle dérivé][s460] · exercices : B1.7, B1.14, B1.26–27, B1.E5, B1.Q11
- **fiche modèle équité et explicabilité** — enseignée : [ch. B6, La fiche modèle][s224] · exercices : B6.8–9, B6.28, B6.Q12
- **fichier de référence figé** (*golden file*) — enseignée : [ch. B2][s480]
- **fichier de verrouillage** (*lock file*) — enseignée : [ch. B7, Les versions figées et l'empreinte des données][s611] · exercices : B7.2, B7.12–13, B7.17, B7.25, B7.E2, B7.Q5–Q6, B7.R3
- **fichiers texte, pathlib** — enseignée : [ch. 0A, 100.6.3 · Fichiers texte et chemins avec `pathlib`][s672] · exercices : 0A.38, 0A.Q8 · réutilisée : ch. 11, 23, B7
- **FID** (*Fréchet Inception distance*) — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s426] · exercices : 27.22, 27.E4, B5.Q10
- **filigrane** (*watermark*) — enseignée : [ch. B5, Transparence et provenance des contenus générés][s573] · exercices : B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11
- **fils rouges du workbook** — enseignée : [ch. 1, Les quatre fils rouges du workbook][s124]
- *filter* → voir [noyau (d'une convolution)](#n)
- **fine-tuning** (*réglage fin, affinage, affiner*) — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s18] · exercices : B1.17–19, B1.21, B1.23, B1.E1, B1.Q5, B1.Q7, B4.Q10, B4.R2 (+ 1) · approfondie : ch. B1 · réutilisée : ch. B4, B8
- **fine-tuning partiel et complet** — enseignée : [ch. B1, 201.4 · Fine-tuning : partiel, complet, dégel progressif, learning…][s447] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 3)
- **fine-tuning partiel, complet** (*partial, full fine-tuning*) — enseignée : [ch. B1, 201.4 · Fine-tuning : partiel, complet, dégel progressif, learning…][s447] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 9) · approfondie : ch. 23 · réutilisée : ch. 27
- *fine-tuning supervisé* → voir [SFT](#s)
- *finite difference* → voir [différence finie (avant, arrière, centrée)](#d)
- *fitted attribute* → voir [attribut appris](#a)
- *fixed-length code* → voir [code de longueur fixe](#c)
- *fixture* → voir [fixture (pytest)](#f)
- **fixture (pytest)** (*fixture*) — enseignée : [ch. B7, Quatre familles de tests][s607] · exercices : B7.15, B7.23, B7.25, B7.E3, B7.Q7
- **flagornerie** (*sycophancy*) — enseignée : [ch. B8, Le piratage de la récompense][s653] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- **FlashAttention** — enseignée : [ch. B3, 203.10 · Coût et passage à l'échelle : complexité quadratique,…][s354] · exercices : B3.8, B3.10, B3.27–28, B3.E2, B3.Q12
- *flatten* → voir [aplatissement](#a)
- fléau de la dimension → voir [malédiction de la dimension, fléau de la dimension](#m)
- *float* → voir [flottant](#f)
- *floor / ceiling* → voir [partie entière (plancher / plafond)](#p)
- **flottant** (*float*) — enseignée : [ch. 0A][s52]
- **flow matching** — enseignée : [ch. B5, 205.9 · Paysage actuel : diffusion latente, transformers de…][s569] · exercices : B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11
- **flux synchrone et asynchrone** — enseignée : [ch. 16, 16.3 · Flux synchrone et asynchrone][s295] · exercices : 16.5, 16.16, 16.E4, 16.Q5–Q6
- *FN* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *focal pixel* → voir [ancre, pixel focal](#a)
- **fold** (*pli*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s218] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10 (+ 3)
- **fonction d'activation** (*activation function*) — enseignée : [ch. 10, 10.3.3 · Les neurones modernes][s40] · exercices : 10.1–2, 10.5, 10.12, 10.14–15, 10.20, 10.Q8–Q9, 10.R1 (+ 4)
- **fonction de masse (pmf)** (*probability mass function*) — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s136] · exercices : 2.1–2, 2.13, 2.16, 2.19, 2.E1, 2.Q1–Q2, 2.R2
- **fonction de Rosenbrock** (*Rosenbrock function*) — enseignée : [ch. 5, Descendre une surface pas à pas][s187] · exercices : 5.2–3, 5.5, 5.7–10, 5.Q7–Q8, 5.Q10 (+ 16) · réutilisée : ch. 19
- **fonction de transfert** (*transfer function*) — enseignée : [ch. 17, 17.2.1 · La forme d'une activation][s300] · exercices : 17.8, 17.Q2
- **fonction de valeur** (*value function*) — enseignée : [ch. B6, Partager un gain entre joueurs][s579] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- **fonction de valeur (RL)** (*value function*) — enseignée : [ch. B8, 208.4 · Acteur-critique : fonction de valeur, erreur TD, avantage, GAE][s409] · exercices : 26.23, 26.34, 26.E5, 29.R2, B8.4, B8.12–13, B8.18, B8.20, B8.Q7 (+ 2) · approfondie : ch. 26
- **fonction fit() avec historique** — enseignée : [ch. 23, 23.8.3 · Analyser l'historique][s374] · exercices : 23.6, 23.19, 23.27, 23.E4
- **fonction score** (*score function*) — enseignée : [ch. B8, Apprendre la politique elle-même][s641] · exercices : B8.3, B8.7, B8.16–17, B8.E1, B8.Q5–Q6, B8.Q9, B8.R3
- fonction signe → voir [marche unité, marche de Heaviside, fonction signe](#m)
- **fonction, paramètre, argument** (*function, parameter, argument*) — enseignée : [ch. 0A, 100.5 · Fonctions][s58] · exercices : 0A.6, 0A.23, 0A.30, 0A.40–42, 0A.44, 0A.Q6, 0B.36, 0B.R3 (+ 49) · approfondie : ch. 0B, 10, 16, 23 · réutilisée : ch. 0B, 1, 2, 5, 6, 7, 9, 10, 11, 13, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, B1, B8
- **fonctions affines et polynômes** — enseignée : [ch. 0B, 101.2 · Fonctions usuelles][s681] · exercices : 0B.Q6
- **fonctions de deux variables, lignes de niveau (lecture)** — enseignée : [ch. 0B, 101.6.1 · Fonctions de deux variables et lignes de niveau][s99] · exercices : 0B.25, 0B.49, 0B.Q11, 5.17 · approfondie : ch. 5 · réutilisée : ch. 5
- **fonctions, valeurs par défaut, portée** — enseignée : [ch. 0A, 100.5 · Fonctions][s58] · exercices : 0A.6, 0A.23, 0A.30, 0A.41–42, 0A.44, 0A.Q6, 0B.36, 0B.40, 17.Q8 · approfondie : ch. 0B · réutilisée : ch. 0B, 6, 10, 11, 13, 16, 18
- **fond** (*background*) — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s578] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5 (+ 6)
- *FOR* → voir [taux de fausses découvertes, de fausses omissions](#t)
- *forçage par l'enseignant* → voir [teacher forcing](#t)
- **force de régularisation ($\lambda$, `alpha`, `C`)** (*regularization strength*) — enseignée : [ch. 9, 9.5 · La régularisation][s229] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 2)
- **forêt aléatoire** (*random forest*) — enseignée : [ch. 14, 14.5 · Les forêts aléatoires][s282] · exercices : 14.15–16, 14.18–19, 14.28, 14.E2, 14.E4, 14.Q6, 15.R1, 17.R2 (+ 1) · réutilisée : ch. B6
- **forme** (*shape*) — enseignée : [ch. 0A, 100.8.4 · Broadcasting][s69] · exercices : 0A.7–8, 0A.52, 0A.56, 0A.Q11, 0B.20, 0B.42–43, 0B.Q9, 7.18 (+ 26) · approfondie : ch. 0B, 18, B5 · réutilisée : ch. 11, 16, 18, 20, 25
- **forme (d'un tenseur)** (*shape*) — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s327] · exercices : 0A.7–8, 0A.56, 0B.20, 0B.42–43, 0B.Q9, 20.1, 20.13, 20.Q2 (+ 25) · approfondie : ch. 0B, 18, B5 · réutilisée : ch. 11, 16, 18, 25
- **forme exec, forme shell** (*exec form, shell form*) — enseignée : [ch. B7, Image, conteneur et Dockerfile][s633] · exercices : B7.23–25, B7.Q4
- **forme fermée (du processus direct)** (*closed form*) — enseignée : [ch. B5, 205.2 · Le processus direct : planning de bruit, ᾱ_t et forme…][s547] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 3) · réutilisée : ch. 11, 25
- **forme matricielle des gradients** — enseignée : [ch. 18, 18.12.1 · La rétropropagation en un seul endroit][s699] · exercices : 18.10, 18.19, 18.28, 18.E1, 18.Q9
- forme shell → voir [forme exec, forme shell](#f)
- **formule des probabilités totales** (*law of total probability*) — enseignée : [ch. 3, 3.6 · Probabilité marginale et tables de contingence][s158] · exercices : 3.3–4, 3.14, 3.Q5, 3.R3
- *forward pass* → voir [passe avant, passe arrière](#p)
- *forward process* → voir [processus direct](#p)
- *forward process posterior* → voir [posterior (d'un pas)](#p)
- **forward() avec branches et bloc résiduel** — enseignée : [ch. 24, 24.6.2 · Construire un modèle fonctionnel][s715] · exercices : 24.28, 24.31, 24.Q12
- **fouille de textes et de données** (*text and data mining, TDM*) — enseignée : [ch. 28, Au-delà du livre : la génération d'images aujourd'hui][s430]
- *foundation model* → voir [modèle de fondation](#m)
- *four-fifths rule* → voir [règle des quatre cinquièmes](#r)
- **fourche de Hume** (*Hume's fork*) — enseignée : [ch. 11, 11.6 · Raisonnement combiné][s258] · exercices : 11.9, 11.Q11
- *FP* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- **fraction clippée** (*clip fraction*) — enseignée : [ch. B8, Les journaux de PPO][s639] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- *fractionally strided convolution* → voir [convolution transposée](#c)
- **framework** (*cadriciel*) — enseignée : [ch. 15, 15.2 · scikit-learn, NumPy et SciPy][s17] · exercices : 15.9, 15.Q1, 23.E5
- *Fréchet distance* → voir [distance de Fréchet](#d)
- *Fréchet Inception distance* → voir [FID](#f)
- *free-running* → voir [agent libre, déclenché](#a)
- *freeze* → voir [geler (des couches)](#g)
- **fréquences naturelles, arbre des fréquences naturelles** (*natural frequencies, natural frequency tree*) — enseignée : [ch. 3, 3.8 · Appliquer la matrice de confusion : le piège de la prévalence][s162] · exercices : 3.7, 3.10, 3.21, 3.26–27, 3.E1, 3.Q12
- *frequency bias* → voir [biais de fréquence (embeddings)](#b)
- **fréquentisme contre bayésianisme** — enseignée : [ch. 4][s176]
- *frequentist* → voir [fréquentiste, bayésien](#f)
- **fréquentiste, bayésien** (*frequentist, Bayesian*) — enseignée : [ch. 4, 4.2 · Fréquentiste et bayésien : deux façons de penser les…][s171] · exercices : 4.11, 4.13, 4.E2, 4.E4, 4.Q1–Q3 · approfondie : ch. 9
- FRIA → voir [analyse d'impact sur les droits fondamentaux (FRIA)](#a)
- *Frobenius norm* → voir [norme de Frobenius](#n)
- **frontière de décision** (*decision boundary*) — enseignée : [ch. 3, 3.7.1 · Classer des échantillons][s159] ; [ch. 7, 7.2.1 · Classification binaire en 2D][s160] · exercices : 3.15, 3.Q6, 7.7, 7.10–11, 7.14–15, 7.Q3–Q4, 7.R2 · réutilisée : ch. 10
- *frozen embedding* → voir [embedding figé](#e)
- **fuite de données** (*data leakage*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s29] ; [ch. 8, 8.3 · Le jeu de test][s122] ; [ch. 12, 12.2 · La règle d'or : apprendre la transformation sur l'entraînement][s123] · exercices : 1.14, 1.18, 1.20, 1.25, 1.E2, 1.Q6, 8.1, 8.3, 8.Q5–Q6 (+ 22) · approfondie : ch. B4 · réutilisée : ch. 11, 13, 15, 29
- *full fine-tuning* → voir [fine-tuning partiel, complet](#f)
- *fully connected layer* → voir [couche dense](#c)
- *function* → voir [fonction, paramètre, argument](#f)
- *function approximation* → voir [approximation de fonction](#a)
- *functional API* → voir [API fonctionnelle](#a)
- *fundamental rights impact assessment* → voir [analyse d'impact sur les droits fondamentaux (FRIA)](#a)
- **fusion (BPE), rang** (*merge, rank*) — enseignée : [ch. B2, 202.3 · Byte Pair Encoding : apprendre et appliquer les fusions][s467] · exercices : B2.3–4, B2.10, B2.12, B2.15–17, B2.20, B2.Q4, B2.R3 (+ 12) · réutilisée : ch. B3, B4
- **fusion d'adaptateur** (*merging*) — enseignée : [ch. B4, Fusionner l'adaptateur][s542] · exercices : B4.3, B4.6, B4.11, B4.25–27, B4.E2, B4.Q10–Q11, B4.R2 (+ 1)
- *fusion par rangs réciproques* → voir [reciprocal rank fusion (RRF)](#r)

## G

- **gabarit (de prompt)** (*prompt template*) — enseignée : [ch. B1, Le zéro-shot][s456] · exercices : B1.5, B1.15, B1.24–25, B1.E4, B1.Q10–Q11
- **GAE** (*generalized advantage estimation*) — enseignée : [ch. B8, 208.4 · Acteur-critique : fonction de valeur, erreur TD, avantage, GAE][s409] · exercices : B8.4, B8.18, B8.20, B8.Q7
- **gain (d'un split), gain d'information** (*impurity decrease, information gain*) — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s197] · exercices : 0A.55, 13.2–3, 13.20, 13.22, 13.25, 13.E2, 13.Q9, 27.2, B3.23
- gain d'information → voir [gain (d'un split), gain d'information](#g)
- GAN → voir [réseau antagoniste génératif (GAN)](#r)
- **GAN : générateur et discriminateur** — enseignée : [ch. 27, 27.4 · Implémenter un GAN][s417] · exercices : 27.12, 27.Q6
- **GAN conditionnel** (*conditional GAN*) — enseignée : [ch. 27, Au-delà du livre : le GAN conditionnel][s425] · exercices : 27.25
- **garde** (*guard*) — enseignée : [ch. B7, torch.export et ONNX][s624] · exercices : 19.R2, B3.Q11, B7.1, B7.19–21, B7.25, B7.E5, B7.Q9–Q10
- *gate* → voir [porte](#p)
- *gated* → voir [accès contrôlé (dépôt)](#a)
- *gated recurrent unit* → voir [GRU](#g)
- *gated repository* → voir [accès contrôlé (dépôt)](#a)
- *Gaussian* → voir [Naïve Bayes (gaussien, multinomial)](#n)
- *Gaussian distribution* → voir [loi normale, gaussienne](#l)
- *Gaussian error linear unit* → voir [GELU](#g)
- gaussienne → voir [loi normale, gaussienne](#l)
- *GDPR (règlement 2016/679)* → voir [RGPD](#r)
- **geler (des couches)** (*freeze*) — enseignée : [ch. 23][s380] · exercices : B1.R2 · réutilisée : ch. 27
- **GELU** (*Gaussian error linear unit*) — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s85] · exercices : 17.2, 17.4, 17.8, 17.10, 17.13–14, 17.17–18, 17.Q8–Q9 (+ 6) · réutilisée : ch. B3
- *general-purpose AI model* → voir [modèle d'IA à usage général](#m)
- **généralisation** (*generalization*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s29] · exercices : 1.14, 1.18, 1.20, 1.25, 1.E2, 1.Q6, 8.R3, 11.17, 11.Q9, 14.R2 (+ 1) · réutilisée : ch. 8
- **généralisation (principe inductif)** (*generalization*) — enseignée : [ch. 11, 11.5 · L'induction][s256] · exercices : 8.R3, 11.6, 11.17, 11.Q9, 14.R2, 24.E2 · approfondie : ch. 1 · réutilisée : ch. 8
- **généralisation hâtive** (*hasty generalization*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s257] · exercices : 11.17–18, 11.E4, 11.Q10
- *generalization* → voir [généralisation](#g), [généralisation (principe inductif)](#g)
- *generalization error* → voir [erreur d'entraînement, erreur de généralisation](#e)
- *generalization gap* → voir [écart de généralisation](#e)
- *generalized advantage estimation* → voir [GAE](#g)
- *generated image* → voir [image de contenu, référence de style, image générée](#i)
- **générateur** (*generator*) — enseignée : [ch. 0A, 100.7.4 · Itérables et générateurs (`yield`)][s66] · exercices : 0A.50, 0A.Q9, 1.Q9, 22.26, 23.23, CP5.8, 25.35, 27.3, 27.Q3, 27.R2 (+ 4) · approfondie : ch. 27 · réutilisée : ch. 24, 26
- **générateur (modèle génératif)** (*generator, generative model*) — enseignée : [ch. 1, 1.5 · Générateurs][s118] ; [ch. 27, 27.4.2 · Le générateur][s119] · exercices : 0A.50, 1.24, 1.E4, 1.Q9, 1.Q11, 22.26, 27.12, 27.25–26, 27.Q6 (+ 9) · approfondie : ch. 0A · réutilisée : ch. 24, 26
- *génération augmentée par recherche* → voir [RAG](#r)
- **génération contrainte** (*constrained decoding*) — enseignée : [ch. B4, La sortie structurée][s527] · exercices : B4.15–16, B4.E1, B4.Q6
- **génération de texte avec température** — enseignée : [ch. 22][s360]
- **génération de texte mot à mot** — enseignée : [ch. 28, 28.3 · Le deep dream][s431] · exercices : 28.16–17, 28.E4, 28.Q4–Q5
- **génération de texte par caractère et par mot** — enseignée : [ch. 24, 24.5.11 · Générer du texte][s719] · exercices : 24.30, 24.R3
- *generative adversarial network* → voir [réseau antagoniste génératif (GAN)](#r)
- *generative model* → voir [générateur (modèle génératif)](#g)
- *generator* → voir [générateur](#g), [générateur (modèle génératif)](#g)
- *geometric sequence* → voir [suite géométrique, raison](#s)
- **géométrie de la surface d'erreur** — enseignée : [ch. 19, 19.2 · L'erreur comme géométrie][s701] · exercices : 19.Q1
- **gestionnaire de contexte** (*context manager*) — enseignée : [ch. B7][s614]
- **get_params/set_params/clone** — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s287] · exercices : 15.1, 15.12, 15.18, 15.23, 15.E2, 15.Q10
- **Gini** — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s197] · exercices : 13.2–3, 13.20, 13.22, 13.25, 13.E2, 13.Q9, 14.R1, CP3.3
- *Gini impurity* → voir [impureté de Gini](#i)
- **git : clone, status, add, commit, push, pull --rebase, .gitignore, branches (aperçu)** — enseignée : [ch. 0A, 100.11.1 · git : `status`, `add`, `commit`, `push`, `pull` ;…][s75] · exercices : 0A.10–11, 0A.E5, 0A.Q12, 24.28, 24.Q12 · approfondie : ch. 24 · réutilisée : ch. B7
- *global* → voir [explication globale, explication locale](#e)
- *global average pooling* → voir [pooling moyen global](#p)
- *global context* → voir [contexte global, contexte local](#c)
- *global extremum* → voir [extremum local, extremum global](#e)
- *global vectors* → voir [GloVe](#g)
- *Glorot* → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- Glorot → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- **GloVe** (*global vectors*) — enseignée : [ch. B2, Compter ou prédire : les deux se rejoignent][s483] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- *glue code* → voir [code de colle](#c)
- *golden file* → voir [fichier de référence figé](#f)
- *Goodhart's law* → voir [loi de Goodhart](#l)
- **goulot** (*bottleneck*) — enseignée : [ch. 25, 25.3 · Le plus simple des autoencodeurs][s120] · exercices : 25.1, 25.9, 25.14–15, 25.22, 25.Q4–Q5, 25.R1, B3.Q1
- *GPU* → voir [GPU, TPU, NPU](#g)
- **GPU, TPU, NPU** (*GPU, TPU, NPU*) — enseignée : [ch. 1, 1.7 · Deep learning][s27] · exercices : 1.1, 1.3, 1.10, 1.23, 1.E4, 1.Q10–Q11, 23.Q7, 24.7, B7.12 · approfondie : ch. 20
- GQA → voir [MQA, GQA](#m)
- **Grad-CAM** (*gradient-weighted class activation mapping*) — enseignée : [ch. 21, 21.9.3 · Regarder les filtres (2) : les cartes d'activation][s350] · exercices : 21.27, 21.Q10, B6.4, B6.20–21, B6.E4, B6.Q8 · approfondie : ch. B6
- **Grad-CAM par hooks** — enseignée : [ch. B6, 206.6 · Grad-CAM : cartes d'activation pondérées par les gradients][s738] · exercices : B6.4, B6.18, B6.20–22, B6.E4, B6.Q8–Q9, B6.R1
- **gradient** — enseignée : [ch. 0B, 101.6.3 · Gradient et direction de plus grande pente][s41] · exercices : 0B.25, 0B.31, 0B.49–50, 0B.E1, 0B.Q11, 5.Q1, 5.Q7–Q8, 5.Q10 (+ 78) · approfondie : ch. 5, 14, 18, 19, 22, 28, B6, B8 · réutilisée : ch. 5, 8, 9, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 27, 28, B3, B6, B8
- *gradient accumulation* → voir [accumulation des gradients](#a)
- *gradient ascent* → voir [montée de gradient](#m)
- *gradient ascent on the input* → voir [montée de gradient sur l'entrée](#m)
- **gradient boosting** — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s226] · exercices : 14.7, 14.25, 14.E5, 14.Q11, 15.R1, CP3.5, B6.15, B6.E1 · réutilisée : ch. 15, B6
- **gradient boosting par histogrammes** (*histogram-based gradient boosting*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s226]
- **gradient boosting stochastique** (*stochastic gradient boosting*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s226]
- **gradient clipping** (*écrêtage du gradient*) — enseignée : [ch. 19, Au-delà du livre : le gradient clipping][s325] ; [ch. 22, 22.6 · Entraîner un RNN][s309] · exercices : 19.19, 19.E4, 22.6–7, 22.19, 22.23, 22.E2, 22.Q6, 22.R2, B8.R2 · réutilisée : ch. B3, B8
- **gradient de politique** (*policy gradient*) — enseignée : [ch. 26, 26.10 · Deux applications][s414] · exercices : 26.37, 26.E5, 26.Q12, B8.7, B8.E1 · approfondie : ch. B8
- *gradient descent* → voir [descente de gradient](#d)
- **gradient évanescent et clipping** — enseignée : [ch. 22, 22.6 · Entraîner un RNN][s309] · exercices : 22.6–7, 22.19, 22.23, 22.E2, 22.Q6, 22.R2
- **gradient évanescent, explosif** (*vanishing, exploding gradient*) — enseignée : [ch. 0B, 101.1.5 · Suites géométriques et leur somme][s80] ; [ch. 22, 22.6 · Entraîner un RNN][s309] · exercices : 0B.13–14, 0B.34, 0B.Q4, 22.6–7, 22.19, 22.23, 22.Q6, 22.R2 (+ 2)
- **gradient numérique** — enseignée : [ch. 5, Au-delà du livre (3) : la différentiation automatique][s188] · exercices : 5.9 · réutilisée : ch. 16
- *gradient penalty* → voir [pénalité de gradient (WGAN-GP)](#p)
- **gradient × entrée** (*gradient × input*) — enseignée : [ch. B6, 206.5 · Attributions pour les réseaux : saillance, gradient ×…][s588] · exercices : B6.3, B6.18–19, B6.Q7, B6.R1
- *gradient × input* → voir [gradient × entrée](#g)
- *gradient-weighted class activation mapping* → voir [Grad-CAM](#g)
- **gradients intégrés** (*integrated gradients*) — enseignée : [ch. B6, 206.5 · Attributions pour les réseaux : saillance, gradient ×…][s588] · exercices : B6.3, B6.18–19, B6.Q7, B6.R1
- *gradual unfreezing* → voir [dégel progressif](#d)
- **graine** (*seed*) — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s6] · exercices : 0A.5, 0A.31, 0A.66, 0A.E4, 0A.Q11, 2.14, 2.E5, 2.Q3, 14.27, 18.23 (+ 15) · approfondie : ch. 2, 26 · réutilisée : ch. 3
- *Gram matrix* → voir [matrice de Gram](#m)
- *grand modèle de langage* → voir [LLM](#l)
- **grande dimension** — enseignée : [ch. 2, 2.7 · Espaces de grande dimension][s147] · exercices : 2.11, 2.25, 2.Q11, 5.10, 7.Q11, 19.Q3, 25.6 · réutilisée : ch. 25
- **graphe de calcul** (*computational graph*) — enseignée : [ch. 0B, 101.6.4 · Règle de la chaîne à plusieurs variables : la somme sur…][s100] · exercices : 0B.29, 0B.51, 0B.E4, 0B.Q11
- **graphe orienté acyclique (DAG)** (*directed acyclic graph*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s294] · exercices : 16.2, 16.4, 16.9, 16.16–18, 16.Q1–Q4 (+ 1)
- **graphique de dépendance (SHAP)** (*dependence plot*) — enseignée : [ch. B6, 206.4 · SHAP en pratique : TreeExplainer, Explainer, graphiques,…][s585] · exercices : B6.2, B6.7, B6.11, B6.15–17, B6.E1–E2, B6.Q5–Q6 (+ 1)
- *graphique en cascade* → voir [waterfall](#w)
- **ε-greedy** (*epsilon-greedy*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s121] ; [ch. 26, 26.6.2 · La politique du Q-learning][s260] · exercices : 11.2, 11.7–8, 11.10–11, 11.13, 11.19–21, 11.R1 (+ 13)
- *greedy algorithm* → voir [algorithme glouton](#a)
- *greedy decoding* → voir [décodage glouton](#d)
- **ε-greedy et softmax (Boltzmann)** — enseignée : [ch. 26, 26.6.2 · La politique du Q-learning][s260] · exercices : 26.5, 26.19, 26.E3, 26.Q9, 26.R3
- *greedy policy* → voir [politique gloutonne](#p)
- *grid* → voir [grille (d'hypothèses)](#g)
- *grid search* → voir [recherche en grille](#r)
- **GridSearchCV** — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s287] · exercices : 15.1, 15.12, 15.18, 15.23, 15.E2, 15.Q10 · réutilisée : ch. 24
- **grille (d'hypothèses)** (*grid*) — enseignée : [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s178] · exercices : 0A.46, 4.19, 9.7, 9.26, 15.1, 15.20, 15.E2, 15.Q10, CP3.14, 18.R2 (+ 7) · réutilisée : ch. 19
- *ground truth* → voir [vérité terrain](#v)
- *group k-fold* → voir [découpage par groupes](#d)
- **group norm** (*group normalization*) — enseignée : [ch. B5, Un bloc résiduel conditionné par le pas][s562] · exercices : B5.11, B5.19, B5.21, B5.Q5–Q6, B5.R1
- *group normalization* → voir [group norm](#g)
- *group relative policy optimization* → voir [GRPO](#g)
- *group-specific thresholds* → voir [seuils par groupe](#s)
- **groupe de paramètres** (*parameter group, param group*) — enseignée : [ch. B1, Learning rates discriminatifs][s448] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 3)
- *grouped-query attention* → voir [MQA, GQA](#m)
- **GroupKFold et TimeSeriesSplit (usage)** — enseignée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s221] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10
- **GRPO** (*group relative policy optimization*) — enseignée : [ch. B8, 208.7 · RL et modèles de langue : RLHF, modèle de récompense, DPO,…][s126] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- **GRU** (*gated recurrent unit*) — enseignée : [ch. 22, 22.7 · LSTM et GRU][s362] · exercices : 22.11, 22.19, 22.E2–E3, 22.Q7, 22.Q9
- *guard* → voir [garde](#g)
- **guidance par classifieur** (*classifier guidance*) — enseignée : [ch. B5, Donner la classe au réseau][s564] · exercices : B5.1, B5.26–27, B5.E4, B5.Q9
- **guidance sans classifieur (CFG)** (*classifier-free guidance*) — enseignée : [ch. B5, 205.7 · Conditionner : label de classe et guidance sans classifieur][s565] · exercices : B5.1, B5.26–27, B5.E4, B5.Q9
- *guidance scale* → voir [échelle de guidance](#e)
- **Guided Grad-CAM** — enseignée : [ch. B6, Avec PyTorch : deux hooks][s591] · exercices : B6.4, B6.18, B6.20–22, B6.E4, B6.Q8–Q9, B6.R1

## H

- *Hadamard product* → voir [produit de Hadamard](#p)
- **hallucination** — enseignée : [ch. 1][s111] ; [ch. B4, 204.10 · Usage responsable : hallucinations, injection de prompt,…][s132] · exercices : B4.9–10, B4.24, B4.E1, B4.E5, B4.Q7, B4.Q12
- *halting problem* → voir [problème de l'arrêt](#p)
- *handler* → voir [logger, handler](#l)
- handler → voir [logger, handler](#l)
- *hard voting* → voir [vote dur, vote souple](#v)
- *harmonic mean* → voir [moyenne harmonique](#m)
- *hash* → voir [somme de contrôle](#s)
- *hasty generalization* → voir [généralisation hâtive](#g)
- He → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- *He initialization* → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- *head* → voir [tête (d'un modèle)](#t)
- *Heaviside step* → voir [marche unité, marche de Heaviside, fonction signe](#m)
- **héritage** (*inheritance*) — enseignée : [ch. 0A, 100.7.3 · Héritage et `super()`][s65] · exercices : 0A.49, 0A.Q9 · réutilisée : ch. 15, 20, 24
- **héritage et super()** — enseignée : [ch. 0A, 100.7.3 · Héritage et `super()`][s65] · exercices : 0A.49, 0A.Q9
- *Hessian matrix* → voir [matrice hessienne](#m)
- *hidden layer* → voir [couche cachée](#c)
- *hidden size* → voir [taille de l'état](#t)
- *hidden state* → voir [état caché](#e), [état caché (d'un notebook)](#e)
- *hierarchical learning* → voir [apprentissage hiérarchique](#a)
- *hierarchical navigable small world* → voir [HNSW](#h)
- **hinge loss** (*perte charnière*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s244] · exercices : 13.4–5, 13.10, 13.16–18, 13.36, 13.Q5–Q6
- **HistGradientBoosting** — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s226] · exercices : 14.26
- *histogram* → voir [histogramme](#h)
- *histogram-based gradient boosting* → voir [gradient boosting par histogrammes](#g)
- **histogramme** (*histogram*) — enseignée : [ch. 2, 2.9 · Le quartet d'Anscombe][s149] · exercices : 2.12, 2.16, 2.30, 2.32, 2.E4, 2.Q12, 16.22, 17.21 · réutilisée : ch. 3
- **historique (d'entraînement)** (*training history*) — enseignée : [ch. 23, 23.8.3 · Analyser l'historique][s374] · exercices : 23.6, 23.18–19, 23.27, 23.E4, 24.R1
- **hiver de l'IA** (*AI winter*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s243] · exercices : 10.4–5, 10.7, 10.10–11, 10.13, 10.16, 10.19, 10.21, 10.Q7 (+ 4)
- **HNSW** (*hierarchical navigable small world*) — enseignée : [ch. B4, Les bases de données vectorielles][s217] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **hold-out** (*mise de côté*) — enseignée : [ch. 8, 8.3 · Le jeu de test][s122] · exercices : 8.1, 8.3–4, 8.7, 8.10–13, 8.Q5–Q6 (+ 6)
- **hook** — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s2] · exercices : 28.11, B6.20 · approfondie : ch. 28, B6 · réutilisée : ch. 28, B6
- **hooks PyTorch** — enseignée : [ch. 28][s349] · réutilisée : ch. B6
- **horodatage** (*time stamp*) — enseignée : [ch. 16, 16.3 · Flux synchrone et asynchrone][s295] · exercices : 16.5, 16.16, 16.E4, 16.Q5–Q6
- *hors du sac* → voir [out-of-bag (OOB)](#o)
- hors ligne → voir [algorithme en ligne, hors ligne](#a)
- *hors politique* → voir [on-policy, off-policy](#o)
- *hosted demo* → voir [démo hébergée](#d)
- *HTTP API* → voir [API HTTP, route](#a)
- *Huber loss* → voir [loss de Huber](#l)
- *Huffman code* → voir [code de Huffman](#c)
- *Hughes phenomenon* → voir [phénomène de Hughes](#p)
- *human oversight* → voir [supervision humaine](#s)
- *Hume's fork* → voir [fourche de Hume](#f)
- *hybrid search* → voir [recherche hybride](#r)
- *hyperbolic tangent* → voir [tangente hyperbolique](#t)
- *hypercube* → voir [hypersphère, hypercube](#h)
- hypercube → voir [hypersphère, hypercube](#h)
- *hyperparameter* → voir [hyperparamètre](#h)
- **hyperparamètre** (*hyperparameter*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s38] · exercices : 1.1, 1.16–17, 1.E2, 1.Q5, 8.2, 15.5, 15.Q3, 19.11, 24.Q1 (+ 10) · approfondie : ch. 8, 24 · réutilisée : ch. 7, 8, 19
- **hyperplan** (*hyperplane*) — enseignée : [ch. 7, 7.3 · Plusieurs classes, plusieurs dimensions][s207] ; [ch. 10, 10.3.1 · Le perceptron][s208] · exercices : 7.14, 7.24, 7.Q2, 7.Q5, 10.1, 10.3, 10.Q5–Q6, 10.R2–R3 (+ 8)
- *hyperplane* → voir [hyperplan](#h)
- *hypersphere* → voir [hypersphère, hypercube](#h)
- **hypersphère, hypercube** (*hypersphere, hypercube*) — enseignée : [ch. 7][s216]
- *hypertrucage* → voir [deepfake](#d)
- **hypothèse de la variété** (*manifold hypothesis*) — enseignée : [ch. 7, 7.6 · La malédiction de la dimension][s148] · exercices : 7.4, 7.8, 7.20, 7.30–31, 7.E4, 7.Q9–Q10
- **hypothèse distributionnelle** (*distributional hypothesis*) — enseignée : [ch. B2, L'hypothèse distributionnelle][s477] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- **hypothèse du monde clos** (*closed-world assumption*) — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5, 11.15, 11.Q7
- **hypothèse nulle** (*null hypothesis*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s222] · exercices : 8.2, 8.4, 8.8, 8.16, 8.18, 8.26–27, 8.E2, 8.E4, 8.Q11
- **hypothèse, observation** (*hypothesis, observation*) — enseignée : [ch. 4, 4.7 · Plusieurs hypothèses][s172] · exercices : 4.6–7, 4.15, 4.18, 4.21–22, 4.24–25, 4.Q7, 4.Q10 (+ 7) · réutilisée : ch. 8
- **hypothèses multiples** — enseignée : [ch. 4, 4.7 · Plusieurs hypothèses][s172] · exercices : 4.6–7, 4.15, 4.18, 4.21–22, 4.24–25, 4.Q7, 4.Q10 (+ 1)
- *hypothesis* → voir [hypothèse, observation](#h)
- *hypothetico-deductive method* → voir [méthode hypothético-déductive](#m)

## I

- **i.i.d.** (*independent and identically distributed*) — enseignée : [ch. 2, 2.4.1 · Des variables i.i.d.][s143] · exercices : 2.20, 2.E2, 2.Q8
- *id* → voir [identifiant (d'un token)](#i)
- *ideal denoiser* → voir [débruiteur parfait (d'un dataset fini)](#d)
- **identifiant (d'un token)** (*id*) — enseignée : [ch. B2, Un modèle ne lit que des identifiants][s462] · exercices : 0A.45, 13.21, B2.13, B2.E1, B2.Q1, B2.R2, B7.14
- **identifiant de run** (*run ID*) — enseignée : [ch. B7][s614] · exercices : B7.14
- *identity matrix* → voir [matrice identité](#m)
- **if/for/while, enumerate, zip, compréhensions** — enseignée : [ch. 0A, 100.4.2 · Boucles : `for`, `range`, `enumerate`, `zip`, `while`,…][s668] · exercices : 0A.3, 0A.21–22, 0A.Q5 · réutilisée : ch. 0B
- *illicit major* → voir [majeur illicite, mineur illicite](#m)
- *illicit minor* → voir [majeur illicite, mineur illicite](#m)
- **im2col** (*unfold*) — enseignée : [ch. 21, 21.3.2 · La vue « œil de mouche »][s346] · exercices : 21.14, 21.Q3
- *image* → voir [conteneur, image](#c)
- image → voir [conteneur, image](#c)
- **image de contenu, référence de style, image générée** (*content image, style reference, generated image*) — enseignée : [ch. 28, 28.4 · Le transfert de style][s432] · exercices : 28.9–10, 28.E2, 28.Q6, 28.R2
- image générée → voir [image de contenu, référence de style, image générée](#i)
- **images en arrays (N, H, W), (N, C, H, W)** — enseignée : [ch. 0A, 100.8.8 · Les images sont des arrays : MNIST][s676] · exercices : 0A.54, 0A.Q10
- *implementation invariance* → voir [sensibilité, invariance à l'implémentation](#s)
- *implicit weights* → voir [poids implicites](#p)
- **importance des features (MDI)** (*mean decrease in impurity*) — enseignée : [ch. 13, 13.5.3 · Limiter l'overfitting][s276] · exercices : 13.19, 13.21, 13.23–25, 13.E3, 13.Q8, 13.Q10, 14.19
- **importance par permutation** (*permutation importance*) — enseignée : [ch. 14, 14.5 · Les forêts aléatoires][s282] ; [ch. B6, 206.2 · Explications globales : coefficients, arbres, importance…][s284] · exercices : 14.15–16, 14.18–19, 14.28, 14.E2, 14.E4, 14.Q6, B6.Q2–Q3 (+ 4)
- **importance par retrait** (*drop-column importance*) — enseignée : [ch. 14][s281]
- *impossibility theorem* → voir [théorème d'impossibilité](#t)
- **impureté** (*impurity*) — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s197] · exercices : 13.2–3, 13.20, 13.22, 13.25, 13.E2, 13.Q9, 14.19, 14.R1
- **impureté de Gini** (*Gini impurity*) — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s197] · exercices : 13.2–3, 13.20, 13.22, 13.25, 13.E2, 13.Q9, 14.R1
- *impurity* → voir [impureté](#i)
- *impurity decrease* → voir [gain (d'un split), gain d'information](#g)
- **imputation** — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s264] · exercices : 12.11, 12.31, 12.Q4
- **imputation (d'enquête)** (*survey imputation*) — enseignée : [mini-projet 3, Le contexte][s665] · exercices : 12.31 · approfondie : ch. 12
- *in-context learning* → voir [apprentissage en contexte](#a)
- *inceptionism* → voir [deep dream](#d)
- **indépendance** (*independence*) — enseignée : [ch. 0B, 101.7.2 · Indépendance][s102] · exercices : 0B.26, 0B.52, 0B.E3, 0B.Q12, 2.E4 · réutilisée : ch. 2, 3, 6, 13, 14
- **indépendance conditionnelle** (*conditional independence*) — enseignée : [ch. 4][s176] ; [ch. 13, 13.6 · Naïve Bayes][s177] · exercices : 13.6–7, 13.27–30, 13.E4, 13.Q11
- indépendante → voir [randomisation en cascade, indépendante](#r)
- *independence* → voir [indépendance](#i)
- *independent and identically distributed* → voir [i.i.d.](#i)
- *independent randomization* → voir [randomisation en cascade, indépendante](#r)
- *index* → voir [index (recherche sémantique)](#i)
- **index (recherche sémantique)** (*index*) — enseignée : [ch. B2, Un numéro n'est pas une représentation][s489] · exercices : B2.5, B2.9, B2.25, B2.E3, B2.Q9, B2.R1
- **indicateurs d'équité par groupe** — enseignée : [ch. B6, 206.9 · Équité : indicateurs de groupe (parité démographique,…][s596] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- **indice de Rand ajusté (ARI)** (*adjusted Rand index*) — enseignée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s214] · exercices : 7.5–6, 7.19–21, 7.E4, 7.Q11
- *individual conditional expectation* → voir [courbe ICE](#c)
- **induction** — enseignée : [ch. 11, 11.3 · Déduction et induction][s252] · exercices : 11.5, 11.9, 11.18, 11.Q6, 14.R2
- **induction paresseuse** (*slothful induction, appeal to coincidence*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s257] · exercices : 11.17–18, 11.E4, 11.Q10
- *inductive bias* → voir [biais inductif](#b)
- **inégalité de Kraft** (*Kraft inequality*) — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s194] · exercices : 6.4, 6.6, 6.13, 6.15, 6.23–24, 6.27, 6.Q7
- *inertia* → voir [inertie](#i)
- **inertie** (*inertia, within-cluster sum of squares*) — enseignée : [ch. 7, Sans labels : k-means][s131] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 11)
- **inférence avec eval et inference_mode** — enseignée : [ch. 23, 23.7 · Entraîner le modèle][s710] · exercices : 23.5, 23.18, 23.E1, 23.Q10, 23.R2
- *inference mode* → voir [mode inférence](#m)
- **inférence par batches, en ligne** (*batch inference, online inference*) — enseignée : [ch. B7, 207.7 · Empaqueter et servir : artefact de modèle, inférence par…][s620] · exercices : B7.3, B7.5, B7.16–19, B7.Q8–Q9, B7.R2–R3 (+ 5)
- *inference to the best explanation* → voir [abduction](#a)
- *influential point* → voir [point influent](#p)
- **information (de Shannon), surprise** (*self-information, surprisal*) — enseignée : [ch. 6, Information : un mot, deux sens][s189] · exercices : 6.11, 6.26, 6.Q1, 6.Q5, 6.R1, 13.Q9, 26.Q2–Q3, B4.15, B5.Q4 · approfondie : ch. 13 · réutilisée : ch. 22, B2
- **information de Shannon** — enseignée : [ch. 6, Information : un mot, deux sens][s189] · exercices : 6.11, 6.Q1, 6.R1
- *information gain* → voir [gain (d'un split), gain d'information](#g)
- **information mutuelle** (*mutual information*) — enseignée : [ch. 12, 12.6 · La sélection de features][s270] · exercices : 12.10–11, 12.25, 12.Q8 · réutilisée : ch. B2
- *ingénierie des features* → voir [feature engineering](#f)
- *inheritance* → voir [héritage](#h)
- **initialisation des poids (LeCun, Glorot, He)** — enseignée : [ch. 16, 16.4.1 · L'initialisation][s296] · exercices : 16.6–7, 16.10, 16.14–15, 16.19–21, 16.Q9–Q10 (+ 5)
- **initialisation LeCun, Glorot (Xavier), He (Kaiming)** (*LeCun, Glorot, He initialization*) — enseignée : [ch. 16, 16.4.1 · L'initialisation][s296] · exercices : 16.6–7, 16.10, 16.14–15, 16.19–21, 16.Q9–Q10 (+ 8) · réutilisée : ch. 21
- **initialisation optimiste** (*optimistic initial values*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s121] · exercices : 11.2, 11.7–8, 11.10–11, 11.13, 11.19–21, 11.R1 (+ 8)
- **injection de prompt** (*prompt injection*) — enseignée : [ch. B4, 204.10 · Usage responsable : hallucinations, injection de prompt,…][s132] · exercices : B4.9–10, B4.24, B4.E1, B4.E5, B4.Q7, B4.Q12
- *inner product* → voir [produit scalaire](#p)
- *input layer* → voir [couche d'entrée](#c)
- **installation éditable** (*editable install*) — enseignée : [ch. B7, 207.1 · Du notebook au projet : limites du notebook, structure…][s605] · exercices : B7.6, B7.8–9, B7.25, B7.E1, B7.Q1–Q2
- *instruct model* → voir [modèle instruct](#m)
- *integrated gradients* → voir [gradients intégrés](#g)
- **intégration continue (CI)** (*continuous integration*) — enseignée : [ch. B7, 207.6 · Tester un projet ML : tests unitaires, tests de données,…][s617] · exercices : B7.15, B7.23, B7.25, B7.E3, B7.Q7, PF.10
- **interaction** (*interaction term*) — enseignée : [ch. 9, Ajuster une courbe : complexité et capacité][s236] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 4)
- *interaction term* → voir [interaction](#i)
- *intercept* → voir [ordonnée à l'origine](#o)
- *internal node* → voir [racine, feuille, nœud interne](#r)
- **interpolation linéaire** (*linear interpolation*) — enseignée : [ch. 1][s111] · réutilisée : ch. 2
- **interpolation sphérique (slerp)** (*spherical linear interpolation*) — enseignée : [ch. 25, 25.5.3 · Mélanger des variables latentes][s397] · exercices : 25.6, 25.20–21, 25.Q8
- *interpolation threshold* → voir [seuil d'interpolation](#s)
- **interprétabilité mécaniste** (*mechanistic interpretability*) — enseignée : [ch. B6, Global ou local][s575] · exercices : B6.Q1–Q2
- *interpretability* → voir [explicabilité](#e)
- *interpretable by design* → voir [interprétable par construction](#i)
- **interprétable par construction** (*interpretable by design*) — enseignée : [ch. B6, Interprétable par construction, ou expliqué après coup][s574] · exercices : B6.Q1–Q2
- *interquartile mean* → voir [moyenne interquartile (IQM)](#m)
- *interquartile range* → voir [écart interquartile](#e)
- *intersectional analysis* → voir [analyse intersectionnelle](#a)
- *interval decay* → voir [planning par paliers](#p)
- **intervalle de confiance** (*confidence interval*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s145] · exercices : 2.6, 2.9–10, 2.22–24, 2.E3, 2.Q10, CP1.5 · réutilisée : ch. 26, B6
- **intervalle de crédibilité** (*credible interval*) — enseignée : [ch. 4, Au-delà du livre (3) : MAP et intervalle de crédibilité][s37] · exercices : 4.25
- *interventional* → voir [Shapley interventionnel, observationnel](#s)
- invariance à l'implémentation → voir [sensibilité, invariance à l'implémentation](#s)
- *inverse* → voir [inverse, déterminant](#i)
- *inverse document frequency* → voir [TF-IDF](#t)
- **inverse, déterminant** (*inverse, determinant*) — enseignée : [ch. 0B, 101.4.4 · Identité et idée de l'inverse][s94] · exercices : 0B.21, 0B.42, 0B.46, 0B.Q9, 4.2, 6.18, 12.16, B2.21 · approfondie : ch. 12, 18 · réutilisée : ch. 9
- **inversion DDIM** (*DDIM inversion*) — enseignée : [ch. B5, Pourquoi c'est plus rapide][s559] · exercices : B5.5, B5.8, B5.17–18, B5.24–25, B5.30–31, B5.Q7–Q8 (+ 1)
- *inverted dropout* → voir [dropout inversé](#d)
- IQM → voir [moyenne interquartile (IQM)](#m)
- *IQR* → voir [écart interquartile](#e)
- *isotonic regression* → voir [calibration de Platt, régression isotone](#c)
- issue → voir [événement, issue](#e)
- **itérables et générateurs (yield)** — enseignée : [ch. 0A, 100.7.4 · Itérables et générateurs (`yield`)][s66] · exercices : 0A.50, 0A.Q9
- **itération sur les valeurs** (*value iteration*) — enseignée : [ch. 26, 26.6.4 · L'éléphant dans la pièce][s411] · exercices : 26.6, 26.8, 26.21, 26.24, 26.26, 26.Q10
- **itertools, heapq (aperçu)** — enseignée : [ch. 0A, 100.6.6 · `itertools` et `heapq` (aperçu)][s673] · exercices : 0A.46, 0A.Q8 · réutilisée : ch. 6, 11

## J

- *Jacobian* → voir [Jacobienne](#j)
- **Jacobienne** (*Jacobian*) — enseignée : [ch. 16][s297] ; [ch. 17, 17.8 · La softmax][s205] · exercices : 17.6–7, 17.15–17, 17.19, 17.E3, 17.Q11–Q12, 17.R2
- **Jacobienne de la softmax** (*softmax Jacobian*) — enseignée : [ch. 17, 17.8 · La softmax][s205] · exercices : 17.6–7, 17.15–17, 17.19, 17.E3, 17.Q11–Q12, 17.R2
- *Jensen-Shannon divergence* → voir [divergence de Jensen-Shannon](#d)
- **jeu à somme nulle** (*zero-sum game*) — enseignée : [ch. 27, 27.4.4 · Le jeu][s420] · exercices : 27.2, 27.4–5, 27.10, 27.E5, 27.Q5, 27.R1
- **jeu coopératif** (*cooperative game*) — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s578] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- **jeu d'entraînement / jeu de test** (*training set / test set*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s29] · exercices : 1.14, 1.18, 1.20, 1.25, 1.E2, 1.Q6, 8.7, 8.Q5, 11.R2 · approfondie : ch. 8 · réutilisée : ch. 3
- **jeu de test** — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s29] ; [ch. 8, 8.3 · Le jeu de test][s122] · exercices : 1.14, 1.18, 1.20, 1.25, 1.E2, 1.Q6, 8.1, 8.3, 8.Q5–Q6 (+ 11) · réutilisée : ch. 3
- **jeu de validation** (*validation set*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s29] ; [ch. 8, 8.4 · Le jeu de validation][s129] · exercices : 1.14, 1.18, 1.20, 1.25, 1.E2, 1.Q6, 8.1, 8.Q7–Q8, 8.R1 (+ 9) · réutilisée : ch. 13, 14
- *job* → voir [workflow, job, étape, action](#w)
- job → voir [workflow, job, étape, action](#w)
- *join* → voir [jointure](#j)
- *joint probability* → voir [probabilité jointe](#p)
- **jointure** (*join, merge*) — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s272] · exercices : 12.14
- joueur nul → voir [efficacité, symétrie, joueur nul, additivité](#e)
- **journal d'expériences** (*experiment log, experiment tracking*) — enseignée : [ch. 24, 24.2.7 · Observations][s384] · exercices : 11.25, 24.12, 24.14–15, 24.Q3, B7.R1 · réutilisée : ch. B7
- *journalisation* → voir [logging](#l)
- *JSON Lines* → voir [JSONL](#j)
- **json, pickle** — enseignée : [ch. 0A, 100.6.4 · Sérialiser : `json` et `pickle`][s4] · exercices : 0A.39, 0A.Q8, 11.25, B2.30, B4.15 · réutilisée : ch. 11, 23, B4, B7
- **JSONL** (*JSON Lines*) — enseignée : [ch. B7, 207.5 · Suivi d'expériences : runs, métriques, artefacts, comparaison][s616] · exercices : B7.2, B7.14, B7.25, B7.E2, B7.R1
- **juge LLM** (*LLM-as-a-judge*) — enseignée : [ch. B4, 204.5 · Évaluer un LLM : jeux de test, métriques, juge LLM,…][s531] · exercices : B4.2, B4.15–16, B4.18, B4.23, B4.29, B4.E4, B4.Q8

## K

- **k plus proches voisins (kNN)** (*k-nearest neighbors*) — enseignée : [ch. 13, 13.3 · Les k plus proches voisins][s211] · exercices : 13.1, 13.9, 13.13–15, 13.34, 13.36, 13.E1, 13.Q3–Q4 (+ 3) · réutilisée : ch. B1, B4
- **k-fold** (*validation croisée à $k$ plis*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s218] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10 (+ 1) · réutilisée : ch. 12, 15
- **k-fold répétée** (*repeated k-fold*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s218] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10
- **k-fold stratifiée** — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s218] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10
- **k-means** (*k-moyennes*) — enseignée : [ch. 1, Les quatre fils rouges du workbook][s124] ; [ch. 7, Sans labels : k-means][s131] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 11) · réutilisée : ch. 8, 15
- **k-means++** — enseignée : [ch. 7, Le départ compte : minima locaux et k-means++][s212] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 9)
- *k-moyennes* → voir [k-means](#k)
- *k-nearest neighbors* → voir [k plus proches voisins (kNN)](#k)
- Kaiming → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- *keep-alive* → voir [connexion persistante](#c)
- *kernel* → voir [noyau](#n), [noyau (d'un SVM)](#n), [noyau (d'une convolution)](#n), [noyau (d'une opération)](#n)
- **kernel (dans le code : `kernel_size`)** (*noyau, filtre*) — enseignée : [ch. 21, 21.2.5 · Le noyau][s25] · exercices : 21.Q2
- *Kernel SHAP* → voir [KernelSHAP](#k)
- *kernel trick* → voir [astuce du noyau](#a)
- **KernelSHAP** (*Kernel SHAP*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s582] · exercices : B6.2, B6.7, B6.11, B6.15–17, B6.E1–E2, B6.Q5–Q6
- *key* → voir [requête, clé, valeur](#r)
- *key-value cache* → voir [KV cache](#k)
- *kink* → voir [point anguleux](#p)
- **KL approchée** (*approximate KL*) — enseignée : [ch. B8, Les journaux de PPO][s639] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- *KL divergence* → voir [divergence KL, divergence de Kullback-Leibler](#d)
- *KL penalty* → voir [pénalité KL](#p)
- **kNN** — enseignée : [ch. 13, 13.3 · Les k plus proches voisins][s211] · exercices : 13.1, 13.9, 13.13–15, 13.34, 13.36, 13.E1, 13.Q3–Q4 (+ 1)
- *knowledge distillation* → voir [distillation](#d)
- *Kolmogorov-Smirnov test* → voir [test de Kolmogorov-Smirnov](#t)
- *Kraft inequality* → voir [inégalité de Kraft](#i)
- **KV cache** (*key-value cache*) — enseignée : [ch. B3, 203.9 · Génération : KV cache et échantillonnage (température,…][s511] · exercices : B3.8, B3.23, B3.E4, B3.Q11, B4.1, B4.R1 · réutilisée : ch. B4

## L

- **L-learning** — enseignée : [ch. 26, 26.5 · Le L-learning][s407] · exercices : 26.18, 26.32, 26.Q7
- *L1 penalty* → voir [Lasso (pénalité L1)](#l)
- *L2 penalty* → voir [Ridge (pénalité L2)](#r)
- *L2 starting point* → voir [L2-SP](#l)
- **L2-SP** (*L2 starting point*) — enseignée : [ch. B1, Rester près des poids pré-entraînés : la pénalité L2-SP][s450] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 3)
- *$L^1$ distance* → voir [distance de Manhattan](#d)
- *$L_\text{simple}$* → voir [loss simplifiée](#l)
- **label** (*étiquette*) — enseignée : [ch. 0A, 100.9.4 · Compter et regrouper : `value_counts`, `groupby`, tri][s5] · exercices : 0A.4, 0A.57, 0A.60, 0A.65, 0A.67, 0A.Q12, 1.9, 1.21, 1.Q3, 1.Q9 (+ 7)
- *label dropout* → voir [abandon du label (guidance sans classifieur)](#a)
- **label smoothing** (*lissage des étiquettes*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s191]
- **lambda, fonctions en argument, Callable** — enseignée : [ch. 0A, 100.5.3 · Fonctions comme valeurs : `lambda`, `key=`, `Callable`][s670] · exercices : 0A.41, 0A.43, 0A.Q7, 0B.R3, 19.6 · réutilisée : ch. 0B, 2, 23
- *Laplace prior* → voir [prior de Laplace](#p)
- *Laplace smoothing* → voir [lissage de Laplace](#l)
- *largest remainder method* → voir [règle du plus fort reste](#r)
- *lasso* → voir [Lasso (pénalité L1)](#l)
- **Lasso (pénalité L1)** (*lasso, L1 penalty*) — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s231] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 2)
- **latence** (*latency*) — enseignée : [ch. B1, 201.9 · Coût d'usage : taille, latence CPU, choix d'un backbone…][s458] · exercices : B1.7, B1.14, B1.26–27, B1.E5, B1.Q11, B4.7, B7.3, B7.19, B7.Q9 (+ 2) · approfondie : ch. B7
- **latence et taille des modèles** — enseignée : [ch. B1, 201.9 · Coût d'usage : taille, latence CPU, choix d'un backbone…][s458] · exercices : B1.7, B1.14, B1.26–27, B1.E5, B1.Q11
- *latency* → voir [latence](#l)
- *latent diffusion* → voir [diffusion latente](#d)
- *latent space* → voir [espace latent](#e)
- *latent variables* → voir [code, variables latentes](#c)
- **LaTeX dans Markdown** — enseignée : [ch. 0B][s34] · exercices : 0B.32
- *law of large numbers* → voir [loi des grands nombres](#l)
- *law of total probability* → voir [formule des probabilités totales](#f)
- *layer* → voir [couche](#c), [Dockerfile, couche](#d)
- **layer norm** (*normalisation par couche*) — enseignée : [ch. 20, 20.4.4 · La batchnorm][s240] · exercices : 20.4, 20.7–8, 20.12, 20.22, 20.24, 20.28–29, 20.E2, 20.Q6 (+ 2) · réutilisée : ch. B3
- *layer-sequential unit-variance* → voir [LSUV](#l)
- **layout `src/`** (*src layout*) — enseignée : [ch. B7, Une structure de dépôt][s603] · exercices : B7.6, B7.8–9, B7.25, B7.E1, B7.Q1–Q2
- *lazy learner* → voir [algorithme paresseux](#a)
- *leaf* → voir [racine, feuille, nœud interne](#r)
- *leaky ReLU* → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- leaky ReLU → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- *learned positional embeddings* → voir [positions apprises](#p)
- *learning curve* → voir [courbe d'apprentissage](#c)
- **learning rate** (*taux d'apprentissage*) — enseignée : [ch. 0B, 101.1.7 · Dénombrement : principe multiplicatif, factorielle,…][s11] ; [ch. 1, 1.2.2 · La version pour ordinateur : échantillons, features, labels][s12] ; [ch. 5, Suivre la pente jusqu'à un extremum][s13] · exercices : 0B.12, 0B.33, 0B.Q5, 1.5, 1.9, 1.14, 1.16–17, 1.Q3, 1.R1 (+ 35) · approfondie : ch. 14, 18, 19, 23, B1 · réutilisée : ch. 21, 23, 24
- **learning rate critique** — enseignée : [ch. 19, 19.3.2 · Changer le learning rate au fil du temps][s315] · exercices : 19.1, 19.15, 19.21, 19.Q6
- **learning rates discriminatifs** (*discriminative learning rates, discriminative fine-tuning*) — enseignée : [ch. B1, 201.4 · Fine-tuning : partiel, complet, dégel progressif, learning…][s447] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 3)
- *learning-rate schedule* → voir [planning (de learning rate)](#p)
- *least squares* → voir [moindres carrés](#m)
- **leave-one-out** (*validation croisée par exclusion d'un exemple, LOOCV*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s218] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10
- **lecture de dépréciations** — enseignée : [ch. 15, 15.3 · Conventions : importer, lire la documentation, suivre les…][s695] · exercices : 15.8, 15.12, 15.Q2
- *LeCun* → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- *length bias* → voir [biais de longueur (d'un modèle de récompense)](#b)
- *level curve* → voir [ligne de niveau](#l)
- **licence Creative Commons** (*Creative Commons license*) — enseignée : [ch. 29][s442]
- **licences et retraits de datasets** — enseignée : [ch. 29, 29.1 · Les datasets publics][s723] · exercices : 29.2, 29.4–6, 29.Q1–Q3, 29.R1–R3 (+ 7)
- *Lidstone smoothing* → voir [lissage additif](#l)
- **LightGBM** — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s226] · exercices : 14.26
- **ligne de base d'une courbe precision-recall** (*PR baseline*) — enseignée : [ch. 3, Au-delà du livre (2) : la courbe precision-recall et l'average…][s168]
- **ligne de niveau** (*level curve, contour line*) — enseignée : [ch. 0B, 101.6.1 · Fonctions de deux variables et lignes de niveau][s99] · exercices : 0B.25, 0B.49–50, 0B.Q11
- **ligne de référence (d'une attribution)** (*baseline*) — enseignée : [ch. B6, Le choix de la ligne de référence][s589] · exercices : B6.3, B6.18–19, B6.Q7, B6.R1
- ligne par ligne → voir [ordre C, ligne par ligne](#o)
- **lignes de niveau** — enseignée : [ch. 5, 5.4 · Le gradient][s186] · exercices : 5.2–3, 5.5, 5.7–10, 5.Q7–Q8, 5.Q10 (+ 18) · approfondie : ch. 0B
- *likelihood* → voir [log-probabilité, vraisemblance](#l), [vraisemblance (d'une hypothèse)](#v)
- *likelihood ratio* → voir [rapport de vraisemblance](#r)
- LIME → voir [substitut local, LIME](#s)
- limitation de la conservation → voir [minimisation, limitation de la conservation](#m)
- *linear* → voir [planning linéaire, planning cosinus (de bruit)](#p)
- *linear interpolation* → voir [interpolation linéaire](#i)
- **linear probe** (*sonde linéaire*) — enseignée : [ch. B1, 201.3 · Extraction de features : embeddings figés, linear probe, k…][s445] · exercices : B1.2, B1.4, B1.13–16, B1.21, B1.E1, B1.Q5, B1.R2
- *linear probing then fine-tuning* → voir [LP-FT](#l)
- *linear separability* → voir [séparabilité linéaire](#s)
- **lissage additif** (*additive smoothing, Lidstone smoothing*) — enseignée : [ch. 13, 13.6 · Naïve Bayes][s177] · exercices : 13.6–7, 13.27–30, 13.E4, 13.Q11
- **lissage de Laplace** (*Laplace smoothing, add-one smoothing*) — enseignée : [ch. 6, Mélanger les codes][s200] ; [ch. 13, 13.6 · Naïve Bayes][s177] · exercices : 6.5, 6.7, 6.16, 6.18, 6.20, 6.22, 6.26, 6.E1, 6.Q10, 6.Q12 (+ 10)
- **lissage des contextes** (*context distribution smoothing*) — enseignée : [ch. B2][s480]
- lisse → voir [courbe continue, lisse, univoque](#c)
- *list* → voir [liste, tuple](#l)
- **liste, tuple** (*list, tuple*) — enseignée : [ch. 0A, 100.3.1 · Listes : indices, tranches, méthodes, mutabilité][s53] · exercices : 0A.2, 0A.9, 0A.16–17, 0A.E2, 0A.Q4, 2.1, B2.Q4 · réutilisée : ch. 0B
- **listes, tuples, dictionnaires, ensembles, collections.Counter** — enseignée : [ch. 0A, 100.3.1 · Listes : indices, tranches, méthodes, mutabilité][s53] · exercices : 0A.2, 0A.9, 0A.16–19, 0A.44, 0A.E2, 0A.Q4, 0B.Q5 (+ 4) · approfondie : ch. 0B · réutilisée : ch. 0B, 3, 11, 15, 26
- *Little's law* → voir [loi de Little](#l)
- **LLM** (*grand modèle de langage*) — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s18] · exercices : 1.E4, 6.Q12, CP1.12, 19.9, 26.E5, B2.9, B2.E1, B2.Q8, B4.Q7–Q8 (+ 5) · approfondie : ch. B4 · réutilisée : ch. B8
- *LLM-as-a-judge* → voir [juge LLM](#j)
- *Lloyd's algorithm* → voir [algorithme de Lloyd](#a)
- *local accuracy* → voir [précision locale, absence, cohérence](#p)
- *local context* → voir [contexte global, contexte local](#c)
- *local explanation* → voir [explication globale, explication locale](#e)
- *local extremum* → voir [extremum local, extremum global](#e)
- *local interpretable model-agnostic explanations* → voir [substitut local, LIME](#s)
- *local minimum* → voir [minimum local](#m)
- *local surrogate* → voir [substitut local, LIME](#s)
- *lock file* → voir [fichier de verrouillage](#f)
- *log level* → voir [niveau de log](#n)
- **log loss** (*logistic loss*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s191]
- *log-derivative trick* → voir [astuce de la log-dérivée](#a)
- **log-loss** — enseignée : [ch. 13, Au-delà du livre : la régression logistique, le classifieur…][s84] · exercices : 13.8 · réutilisée : ch. 18
- **log-odds** — enseignée : [ch. 13, Au-delà du livre : la régression logistique, le classifieur…][s84]
- **log-probabilité, vraisemblance** (*log-probability, likelihood*) — enseignée : [ch. 0B, 101.2.4 · Logarithmes : ln, log₂, log₁₀, changement de base][s82] ; [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s35] · exercices : 0B.15–16, 0B.36–37, 0B.E3, 0B.Q7, 4.24, 4.Q6–Q7, 9.27 · réutilisée : ch. 13
- **log-probabilités** — enseignée : [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s35] · exercices : 0B.E3, 4.24 · réutilisée : ch. 13
- *log-probability* → voir [log-probabilité, vraisemblance](#l)
- **log-softmax** — enseignée : [ch. 17, 17.8 · La softmax][s205] · exercices : 17.6–7, 17.15–17, 17.19, 17.E3, 17.Q11–Q12, 17.R2 · réutilisée : ch. B8
- **log-sum-exp** (*astuce du log-somme-exp*) — enseignée : [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s35] · exercices : 17.15 · approfondie : ch. 17 · réutilisée : ch. B2
- *log-uniform distribution* → voir [distribution log-uniforme](#d)
- **logarithme népérien** (*natural logarithm*) — enseignée : [ch. 0B, 101.2.4 · Logarithmes : ln, log₂, log₁₀, changement de base][s82] · exercices : 0B.15–16, 0B.36–37, 0B.E3, 0B.Q7 · réutilisée : ch. 11
- *logger* → voir [logger, handler](#l)
- **logger, handler** (*logger, handler*) — enseignée : [ch. B7, Le logging plutôt que print][s609] · exercices : B7.10, B7.25, B7.E1, B7.Q3
- **logging** (*journalisation*) — enseignée : [ch. B7, 207.2 · Scripts et ligne de commande : argparse, point d'entrée,…][s606] · exercices : B7.10, B7.25, B7.E1, B7.Q3
- *logistic function* → voir [sigmoïde](#s)
- *logistic loss* → voir [log loss](#l)
- *logistic regression* → voir [régression logistique](#r)
- **logit** — enseignée : [ch. 0B][s34] · exercices : 18.14, 18.E5, 20.R2, 23.Q9, 24.R3, B1.1, B3.29 · approfondie : ch. 18, 23 · réutilisée : ch. 20
- *logit scale* → voir [échelle des logits (CLIP)](#e)
- *logits* → voir [logits (d'un classifieur)](#l)
- **logits (d'un classifieur)** (*logits*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s191] ; [ch. 17, 17.8 · La softmax][s205] · exercices : 17.6–7, 17.15–17, 17.19, 17.E3, 17.Q11–Q12, 17.R2 (+ 7) · approfondie : ch. 18, 23 · réutilisée : ch. 20
- **logsumexp** — enseignée : [ch. 13, 13.6 · Naïve Bayes][s177] · exercices : 13.6–7, 13.27–30, 13.E4, 13.Q11 · réutilisée : ch. 17
- loi → voir [distribution de probabilité, loi](#d)
- **loi Beta** (*Beta distribution*) — enseignée : [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s178] · réutilisée : ch. 11
- **loi catégorielle, multinoulli** (*categorical distribution*) — enseignée : [ch. 2, 2.3.4 · La loi catégorielle (multinoulli)][s141] · exercices : 2.5, 2.19, 2.Q6 · réutilisée : ch. 6
- **loi de Bernoulli** (*Bernoulli distribution*) — enseignée : [ch. 2, 2.3.3 · La loi de Bernoulli][s140] · exercices : 2.5, 2.17, 2.Q6 · réutilisée : ch. 4, 8, 11, 14
- **loi de Goodhart** (*Goodhart's law*) — enseignée : [ch. B8, Le piratage de la récompense][s653] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- **loi de Little** (*Little's law*) — enseignée : [ch. B7, Latence, percentiles et débit][s622] · exercices : B7.3, B7.5, B7.16–19, B7.Q8–Q9, B7.R2–R3 (+ 2)
- **loi de Student** (*Student's t distribution*) — enseignée : [ch. B7, Rapporter un score : plusieurs graines et un intervalle][s615] · exercices : B7.2, B7.12–13, B7.17, B7.25, B7.E2, B7.Q5–Q6, B7.R3
- **loi des grands nombres** (*law of large numbers*) — enseignée : [ch. 0B, 101.7.5 · Simuler pour estimer : fréquences et loi des grands nombres][s106] · exercices : 0B.52–53, 0B.E5, 0B.Q12
- **loi normale, gaussienne** (*normal distribution, Gaussian distribution*) — enseignée : [ch. 2, 2.3.2 · La loi normale][s139] · exercices : 2.3–4, 2.15, 2.17–18, 2.29, 2.31, 2.Q5, 25.6, 25.29 (+ 3) · approfondie : ch. 25 · réutilisée : ch. 5, 7, 9, 12, 13, 17, 25, B5
- **loi uniforme** (*uniform distribution*) — enseignée : [ch. 2, 2.3.1 · La loi uniforme][s138] · exercices : 2.17, 2.Q4 · réutilisée : ch. 3
- **lois d'échelle** (*scaling laws*) — enseignée : [ch. B3, 203.10 · Coût et passage à l'échelle : complexité quadratique,…][s354] · exercices : B3.8, B3.10, B3.27–28, B3.E2, B3.Q12
- **lois usuelles** — enseignée : [ch. 2, 2.3 · Quelques lois usuelles][s686] · exercices : 2.16–17, 2.Q4
- *long short-term memory* → voir [LSTM](#l)
- *long-term dependency* → voir [dépendance à long terme](#d)
- *LOOCV* → voir [leave-one-out](#l)
- **LoRA** (*low-rank adaptation*) — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s18] · exercices : B4.3, B4.6, B4.11, B4.25–26, B4.28, B4.E3, B4.Q11, PF.6 · approfondie : ch. B4
- **LoRA from scratch** — enseignée : [ch. B4, 204.8 · Fine-tuning : complet ou PEFT ; LoRA (rang, alpha, modules…][s539] · exercices : B4.3, B4.6, B4.11, B4.25–27, B4.E2, B4.Q10–Q11, B4.R2 (+ 1)
- **loss** (*perte, fonction de coût*) — enseignée : [ch. 0B, 101.1.3 · Notations Σ et Π][s14] · exercices : 0B.3, 0B.24, 0B.32, 0B.35, 0B.Q3, 0B.R1, 6.20, 6.E1, 6.Q12, CP1.12 (+ 30) · approfondie : ch. 6, 18, 27, 28, B5 · réutilisée : ch. 18, 19, B8
- **loss calculée sur les logits** (*loss on logits*) — enseignée : [ch. 18, Au-delà du livre : les losses calculées sur les logits][s310]
- **loss de contenu** (*content loss*) — enseignée : [ch. 28, 28.4.3 · La loss de contenu][s434] · exercices : 28.4, 28.19, 28.22, 28.Q8
- **loss de Huber** (*Huber loss*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s26] ; [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s234] · exercices : 9.1, 9.12, 9.14, 9.E2, 9.Q1, B8.1, B8.Q3–Q4, B8.Q9, B8.R1 (+ 5)
- **loss de prédiction du bruit** — enseignée : [ch. B5, 205.3 · Le processus inverse et la loss simplifiée : prédire le…][s548] · exercices : B5.7, B5.11, B5.16, B5.19, B5.23, B5.E1–E2, B5.Q4, B5.R3
- **loss de style** (*style loss*) — enseignée : [ch. 28, 28.4.4 · La loss de style][s435] · exercices : 28.1, 28.4–5, 28.19, 28.22, 28.29, 28.Q9
- **loss minimax et loss non saturante** — enseignée : [ch. 27, 27.4.4 · Le jeu][s420] · exercices : 27.2, 27.4–5, 27.10, 27.E5, 27.Q5, 27.R1
- **loss non saturante** (*non-saturating loss*) — enseignée : [ch. 27, 27.4.3 · Entraîner le GAN][s419] · exercices : 27.3, 27.6, 27.12, 27.16, 27.18–19, 27.27, 27.E1, 27.Q6
- *loss on logits* → voir [loss calculée sur les logits](#l)
- **loss perceptuelle** (*perceptual loss*) — enseignée : [ch. 28, 28.4.2 · Vue d'ensemble][s436] · exercices : 28.23, 28.Q6, 28.R1
- **loss quantile** (*pinball loss, quantile loss*) — enseignée : [ch. 9][s235]
- *loss scaling* → voir [mise à l'échelle de la loss](#m)
- **loss simplifiée** (*simplified objective, $L_\text{simple}$*) — enseignée : [ch. B5, 205.3 · Le processus inverse et la loss simplifiée : prédire le…][s548] · exercices : B5.7, B5.11, B5.16, B5.19, B5.23, B5.E1–E2, B5.Q4, B5.R3
- **losses de contenu, de style et de variation totale** — enseignée : [ch. 28, 28.4 · Le transfert de style][s432] · exercices : 28.4, 28.9–10, 28.E2, 28.Q6, 28.R2
- **losses et leurs gradients (MSE, BCE sur logits, cross-entropy softmax)** — enseignée : [ch. 18, Au-delà du livre : les losses calculées sur les logits][s310]
- *lossless compression* → voir [compression sans perte](#c), [compression avec ou sans perte](#c)
- *lossy* → voir [compression avec ou sans perte](#c)
- *low-rank adaptation* → voir [LoRA](#l)
- **LP-FT** (*linear probing then fine-tuning*) — enseignée : [ch. B1, Pourquoi un learning rate beaucoup plus petit][s449] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 3)
- **LSTM** (*long short-term memory*) — enseignée : [ch. 22, 22.7 · LSTM et GRU][s362] · exercices : 22.3–4, 22.7, 22.11, 22.18–19, 22.23, 22.E2, 22.Q7, 22.Q9 (+ 11) · approfondie : ch. 24 · réutilisée : ch. 24, 28, B3
- **LSTM et GRU (ordre des portes PyTorch)** — enseignée : [ch. 22, 22.7 · LSTM et GRU][s362] · exercices : 22.11, 22.19, 22.E2, 22.Q7
- **LSTM pour séries (fenêtres, scaler, RMSE)** — enseignée : [ch. 24, 24.5.1 · Générer des séries][s717] · exercices : 24.25, 24.Q10
- **LSUV** (*layer-sequential unit-variance*) — enseignée : [ch. 16, Au-delà du livre : diagnostiquer l'évanouissement et l'explosion…][s298]

## M

- MAC → voir [multiplication-addition (MAC)](#m)
- machine à vecteurs de support → voir [SVM, machine à vecteurs de support](#s)
- *machine epsilon* → voir [epsilon machine](#e)
- **machine learning (ML)** (*apprentissage automatique*) — enseignée : [ch. 1, 1.1 · Le machine learning : extraire du sens des données][s128] · exercices : 0A.E1, 0B.E2, 1.6, 1.E1, 1.Q1, 2.E2, 6.E4, 11.9, 20.Q1, 23.Q8 (+ 4) · approfondie : ch. B7
- *macro* → voir [moyenne macro, micro, pondérée](#m)
- MAE → voir [MSE, RMSE, MAE](#m)
- **majeur illicite, mineur illicite** (*illicit major, illicit minor*) — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s255] · exercices : 11.4, 11.15–16, 11.Q8
- **malédiction de la dimension, fléau de la dimension** (*curse of dimensionality*) — enseignée : [ch. 2, 2.7 · Espaces de grande dimension][s147] ; [ch. 7, 7.6 · La malédiction de la dimension][s148] · exercices : 2.11, 2.25, 2.Q11, 7.4, 7.8, 7.20, 7.30–31, 7.Q9–Q10 (+ 2) · réutilisée : ch. 12, 13
- *Manhattan distance* → voir [distance de Manhattan](#d)
- *manifold hypothesis* → voir [hypothèse de la variété](#h)
- *many-to-many* → voir [one-to-many, many-to-one, many-to-many](#o)
- many-to-many → voir [one-to-many, many-to-one, many-to-many](#o)
- *many-to-one* → voir [one-to-many, many-to-one, many-to-many](#o)
- many-to-one → voir [one-to-many, many-to-one, many-to-many](#o)
- **MAP** (*maximum a posteriori*) — enseignée : [ch. 4, Au-delà du livre (3) : MAP et intervalle de crédibilité][s37]
- **MAPE** (*erreur absolue moyenne en pourcentage*) — enseignée : [ch. 9][s235]
- MAR → voir [MCAR, MAR, MNAR](#m)
- marche de Heaviside → voir [marche unité, marche de Heaviside, fonction signe](#m)
- **marche unité, marche de Heaviside, fonction signe** (*unit step, Heaviside step, sign function*) — enseignée : [ch. 17, 17.4 · Les marches][s302] ; [ch. 0B, 101.1.2 · Valeur absolue, inégalités, partie entière, fonction signe][s79] · exercices : 0B.2, 0B.33, 0B.36, 0B.Q2, 17.2, 17.12, 17.18, 17.Q5, 17.R3 · réutilisée : ch. 21
- **marge** (*margin*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s243] ; [ch. 13, 13.4 · Les machines à vecteurs de support][s244] · exercices : 10.4–5, 10.7, 10.10–11, 10.13, 10.16, 10.19, 10.21, 10.Q7 (+ 13)
- **marge (d'un SVM)** (*margin*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s244] · exercices : 10.23, 13.4–5, 13.10, 13.16–18, 13.36, 13.Q5–Q6 · réutilisée : ch. 10
- **marge souple, paramètre C** (*soft margin, C*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s244] · exercices : 13.4–5, 13.10, 13.16–18, 13.36, 13.Q5–Q6
- *margin* → voir [marge](#m), [marge (d'un SVM)](#m)
- *marginal contribution* → voir [contribution marginale](#c)
- *marginal likelihood* → voir [évidence](#e)
- *marginal probability* → voir [probabilité simple, marginale](#p)
- marginale → voir [probabilité simple, marginale](#p)
- *Markov chain Monte Carlo* → voir [MCMC](#m)
- *Markov decision process* → voir [processus de décision markovien (MDP)](#p)
- *masked language model* → voir [modèle de langue masqué (MLM)](#m)
- *masker* → voir [masqueur (shap)](#m)
- **masque booléen** (*boolean mask*) — enseignée : [ch. 0A, 100.8.2 · Indexation, tranches, masques booléens ; vues et copies][s72] · exercices : 0A.7, 0A.28–29, 0A.56, 0A.65, 0A.Q10
- **masque causal** (*causal mask*) — enseignée : [ch. B3, Le masque causal][s497] · exercices : B3.2, B3.11–12, B3.17–18, B3.29, B3.Q4–Q5, B4.R1 · réutilisée : ch. B4
- **masque d'actions** (*action mask*) — enseignée : [ch. 26][s416] · exercices : 26.35
- **masque d'attention** (*attention mask*) — enseignée : [ch. B2, Utiliser un tokenizer avec `transformers`][s475] · exercices : B2.13, B2.19, B2.30, B2.E2, B2.Q7
- **masque de padding** (*padding mask*) — enseignée : [ch. B3, Le masque de padding][s498] · exercices : B3.2, B3.11–12, B3.17–18, B3.29, B3.Q4–Q5
- **masques causal et de padding** — enseignée : [ch. B3, 203.3 · Masques : causal, padding ; auto-attention et attention…][s496] · exercices : B3.2, B3.11–12, B3.17–18, B3.29, B3.Q4–Q5
- **masques d'actions** — enseignée : [ch. 26][s416]
- **masqueur (shap)** (*masker*) — enseignée : [ch. B6, Un exemple à deux features][s583] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- **matplotlib : plot, scatter, hist, subplots, imshow, légendes ; wb.plot.show_images** — enseignée : [ch. 0A, 100.10.1 · `plot`, `scatter`, `hist` ; titres, axes et légendes][s677] · exercices : 0A.36, 0A.59–60, 0A.67, 0A.Q12 · réutilisée : ch. 0B, 1
- **matrice de co-occurrences** (*co-occurrence matrix*) — enseignée : [ch. B2, La matrice de co-occurrences][s478] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- **matrice de confusion** (*confusion matrix*) — enseignée : [ch. 3, 3.7.2 · La matrice de confusion][s47] · exercices : 3.2, 3.15, 3.17, 3.Q7, CP1.6, 10.R3, CP2.13, 21.23, 23.20, B6.R3 · réutilisée : ch. 4, 7, 8, 10, 13, 20, 21, 23, B6
- **matrice de corrélation** — enseignée : [ch. 2, Matrices de covariance et de corrélation][s151] · exercices : 2.7–8, 2.10, 2.26–27, 2.32, 2.E4, 2.Q12
- **matrice de covariance, de corrélation** (*covariance matrix, correlation matrix*) — enseignée : [ch. 2, Matrices de covariance et de corrélation][s151] ; [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s152] · exercices : 2.7–8, 2.10, 2.26–27, 2.32, 2.E4, 2.Q12, 12.7, 12.Q9 (+ 5) · réutilisée : ch. 25, 27
- **matrice de Gram** (*Gram matrix*) — enseignée : [ch. 28, 28.4.1 · Le style dans une matrice][s433] · exercices : 28.2–3, 28.8, 28.10, 28.18, 28.E2, 28.Q7, CP6.5
- **matrice hessienne** (*Hessian matrix*) — enseignée : [ch. 5, Les points où le gradient s'annule][s185] · exercices : 5.2–3, 5.5, 5.7–10, 5.Q7–Q8, 5.Q10 (+ 15)
- **matrice identité** (*identity matrix*) — enseignée : [ch. 0B, 101.4.4 · Identité et idée de l'inverse][s94] · exercices : 0B.21, 0B.42, 0B.46, 0B.Q9
- **matrices : forme, transposée, produit matrice-vecteur, produit matriciel, vérification des formes** — enseignée : [ch. 0B, 101.4.1 · Forme, éléments, transposée ; un dataset est une matrice][s92] · exercices : 0A.53, 0B.9, 0B.20, 0B.30, 0B.42–43, 0B.45, 0B.Q9, 0B.R2, 16.R3 (+ 5) · approfondie : ch. 21 · réutilisée : ch. 9, 12, 16, 18, 25, 27, 28, B3, B5
- *matrix multiplication* → voir [produit matriciel](#p)
- *Matthews correlation coefficient* → voir [MCC](#m)
- **max et average pooling** — enseignée : [ch. 21, 21.4.2 · Le stride pour construire une hiérarchie][s347] · exercices : 21.3, 21.5, 21.26, 21.Q6
- **max-norm (contrainte)** (*max-norm constraint*) — enseignée : [ch. 24, 24.2.6 · Ajouter du dropout][s382] · exercices : 24.6, 24.13–14, 24.E3, 24.Q3
- *max-norm constraint* → voir [max-norm (contrainte)](#m)
- *maximin* → voir [minimax, maximin](#m)
- maximin → voir [minimax, maximin](#m)
- **maximisation d'activation** (*activation maximization*) — enseignée : [ch. 21, 21.9.2 · Regarder les filtres (1) : ce qui les excite le plus][s348] ; [ch. 28][s349] · exercices : 21.28, 21.Q10
- *maximum mean discrepancy* → voir [MMD](#m)
- **maxout** — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s299] · exercices : 17.2, 17.5, 17.12, 17.18, 17.20, 17.23–24, 17.Q6–Q7, 17.R1 (+ 2)
- **MCAR, MAR, MNAR** (*missing completely at random, missing at random, missing not at random*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s264] · exercices : 12.11, 12.Q4
- **MCC** (*Matthews correlation coefficient*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s161] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11
- *McCulloch-Pitts neuron* → voir [neurone formel](#n)
- **MCMC** (*Markov chain Monte Carlo*) — enseignée : [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s178]
- *McNemar's test* → voir [test de McNemar](#t)
- MDI → voir [importance des features (MDI)](#i)
- MDP → voir [processus de décision markovien (MDP)](#p)
- *mean* → voir [moyenne, médiane, mode](#m)
- *mean decrease in impurity* → voir [importance des features (MDI)](#i)
- **mean pooling** (*moyenne des tokens*) — enseignée : [ch. B2, D'une suite de vecteurs à un vecteur de phrase][s486] · exercices : B2.28–29, B2.E4–E5, B2.Q11, B2.R1
- *mean reciprocal rank* → voir [MRR](#m)
- *measurement* → voir [biais de labels, de mesure, de déploiement](#b)
- *mechanistic interpretability* → voir [interprétabilité mécaniste](#i)
- *median* → voir [moyenne, médiane, mode](#m)
- médiane → voir [moyenne, médiane, mode](#m)
- **médiatrice** (*perpendicular bisector*) — enseignée : [ch. 7, Avec des labels : le centroïde le plus proche][s210] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 9)
- **meilleur checkpoint** (*best checkpoint*) — enseignée : [ch. 23, 23.10.3 · Early stopping][s377] · exercices : 23.6, 23.25, 23.27, 23.E4, B1.19 · réutilisée : ch. B1
- **mélange d'experts (MoE)** (*mixture of experts*) — enseignée : [ch. B3, Compter les paramètres d'un bloc][s516] · exercices : B3.5, B3.11, B3.16, B3.19, B3.25, B3.Q8, B3.R3
- **mélange paramétrique** (*parametric blending*) — enseignée : [ch. 25, 25.2.3 · Mélanger deux représentations][s396] · exercices : 25.2, 25.21, 25.Q3
- **mémoire de cellule** (*cell state*) — enseignée : [ch. 22, 22.7 · LSTM et GRU][s362] · exercices : 22.11, 22.19, 22.E2, 22.Q7
- **mémoire de rejeu** (*replay buffer, experience replay*) — enseignée : [ch. 26, 26.9 · Le rejeu d'expérience][s413] ; [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s234] · exercices : 26.12, 26.31, 26.33, 26.E4, 26.Q12, B8.1, B8.Q3–Q4, B8.Q9, B8.R1 (+ 5)
- **mémoire paramétrique, non paramétrique** (*parametric, non-parametric memory*) — enseignée : [ch. B4, L'idée][s538] · exercices : B4.2, B4.7–8, B4.21–23, B4.30, B4.E1–E2, B4.Q10 (+ 2) · approfondie : ch. 13
- **mémorisation (générateur d'images)** (*memorization*) — enseignée : [ch. B5, 205.8 · Évaluer : distance de Fréchet, diversité, mémorisation][s427] · exercices : B3.10, B5.10, B5.25, B5.28–29, B5.31, B5.Q10, B5.R2
- **mémorisation (modèle de langue)** (*memorization*) — enseignée : [ch. B3][s502] ; [ch. B5, 205.8 · Évaluer : distance de Fréchet, diversité, mémorisation][s427] · exercices : B3.10, B5.10, B5.25, B5.28–29, B5.31, B5.Q10, B5.R2
- *memorization* → voir [mémorisation (modèle de langue)](#m), [mémorisation (générateur d'images)](#m)
- *merge* → voir [jointure](#j), [fusion (BPE), rang](#f)
- **merge et pivot_table** — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s272]
- *merging* → voir [fusion d'adaptateur](#f)
- **mesure de latence** — enseignée : [ch. B7, Latence, percentiles et débit][s622] · exercices : B7.3, B7.5, B7.16–19, B7.Q8–Q9, B7.R2–R3 (+ 2)
- **méta-estimateur** (*meta-estimator*) — enseignée : [ch. 7, 7.4.2 · Un-contre-un (OvO)][s209] · exercices : 7.1–2, 7.23, 7.Q7
- *meta-estimator* → voir [méta-estimateur](#m)
- *meta-learner* → voir [stacking, méta-modèle](#s)
- méta-modèle → voir [stacking, méta-modèle](#s)
- **métadonnées de provenance, C2PA** (*content credentials, C2PA*) — enseignée : [ch. B5, Transparence et provenance des contenus générés][s573] · exercices : B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11
- *method* → voir [classe, objet, méthode, attribut](#c)
- méthode → voir [classe, objet, méthode, attribut](#c)
- **méthode à frontière** (*boundary method*) — enseignée : [ch. 7, 7.2.1 · Classification binaire en 2D][s160] · exercices : 7.7, 7.10–11, 7.14–15, 7.Q3–Q4, 7.R2
- **méthode agnostique, méthode spécifique** (*model-agnostic, model-specific method*) — enseignée : [ch. B6][s576]
- **méthode d'Euler** (*Euler method*) — enseignée : [ch. B5, Flow matching et rectified flow][s571] · exercices : B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11
- **méthode de Monte-Carlo** (*Monte Carlo method*) — enseignée : [ch. 3][s170]
- **méthode des trapèzes** (*trapezoidal rule*) — enseignée : [ch. 3, Au-delà du livre (1) : la courbe ROC et l'AUC][s167]
- **méthode du coude** (*elbow method*) — enseignée : [ch. 7, Combien de clusters ?][s213] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 9)
- **méthode hypothético-déductive** (*hypothetico-deductive method*) — enseignée : [ch. 11, 11.3 · Déduction et induction][s252] · exercices : 11.5, 11.9, 11.Q6
- méthode spécifique → voir [méthode agnostique, méthode spécifique](#m)
- **méthodes spéciales : __repr__, __len__, __getitem__, __call__, surcharge d'opérateurs (__add__, __radd__, __mul__, __eq__)** — enseignée : [ch. 0A, 100.7.2 · Méthodes spéciales : `__repr__`, `__len__`,…][s674] · exercices : 0A.48–50, 0A.Q9 · réutilisée : ch. 18, 20
- MHA → voir [attention multi-têtes (MHA)](#a)
- *micro* → voir [moyenne macro, micro, pondérée](#m)
- micro → voir [moyenne macro, micro, pondérée](#m)
- **micro-autograd** — enseignée : [ch. 18, Au-delà du livre : un micro-autograd][s312] · exercices : 18.26
- *middle term* → voir [moyen terme](#m)
- *min-max scaling* → voir [normalisation min-max](#n)
- **min-SNR** (*min-SNR weighting*) — enseignée : [ch. B5, Le lien avec l'ELBO][s568] · exercices : B5.7, B5.11, B5.16, B5.19, B5.23, B5.E1–E2, B5.Q4, B5.R3
- *min-SNR weighting* → voir [min-SNR](#m)
- min_delta → voir [patience, `min_delta`](#p)
- mineur illicite → voir [majeur illicite, mineur illicite](#m)
- **mini-batch** (*mini-lot*) — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s6] ; [ch. 19, 19.4.3 · La descente par mini-batches][s8] · exercices : 0A.5, 0A.31, 0A.66, 0A.E4, 0A.Q11, 16.3, 18.Q11, 19.25, 19.E5, 19.Q8 (+ 4) · approfondie : ch. 18 · réutilisée : ch. 18
- *mini-batch gradient descent* → voir [descente par mini-batches](#d)
- **mini-batches** — enseignée : [ch. 18, 18.12.5 · Les mini-batches][s700] ; [ch. 27, 27.5 · Un GAN en action][s423] · exercices : 18.7, 18.17, 18.19, 18.29, 18.Q11, 27.4, 27.11, 27.13–14, 27.Q7 (+ 4) · approfondie : ch. 0A
- **mini-batches : iterate_minibatches, drop_last** — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s6] · exercices : 0A.5, 0A.31, 0A.66, 0A.E4, 0A.Q11
- **mini-GPT PyTorch from scratch** — enseignée : [ch. B3, 203.8 · Mini-GPT from scratch : données, entraînement, loss,…][s729] · exercices : B3.10, B3.19–22, B3.26, B3.29–30, B3.Q10, B3.R1
- *minibatch discrimination* → voir [discrimination par mini-batch](#d)
- *minibatch standard deviation* → voir [écart-type par mini-batch](#e)
- *minimal cost-complexity pruning* → voir [élagage coût-complexité](#e)
- *minimax* → voir [minimax, maximin](#m)
- **minimax, maximin** (*minimax, maximin*) — enseignée : [ch. 27, 27.4.4 · Le jeu][s420] · exercices : 27.2, 27.4–5, 27.10, 27.E5, 27.Q5, 27.R1
- **minimisation, limitation de la conservation** (*data minimisation, storage limitation*) — enseignée : [ch. B7, Journaliser les prédictions, et le RGPD][s635] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- **minimum local** (*local minimum*) — enseignée : [ch. 7, Sans labels : k-means][s131] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 11) · réutilisée : ch. 11
- *minimum-norm solution* → voir [solution de norme minimale, pseudo-inverse](#s)
- **mise à jour bayésienne, séquentielle** (*Bayesian updating, sequential updating*) — enseignée : [ch. 4, 4.6 · Répéter la règle de Bayes][s175] · exercices : 4.5, 4.8, 4.16, 4.20, 4.26, 4.Q9, 9.7
- **mise à jour douce (de Polyak)** (*soft update, Polyak*) — enseignée : [ch. B8, Les journaux de PPO][s639] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- **mise à jour séquentielle** — enseignée : [ch. 4, 4.6 · Répéter la règle de Bayes][s175] · exercices : 4.5, 4.8, 4.16, 4.20, 4.26, 4.Q9
- **mise à l'échelle de la loss** (*loss scaling*) — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s391]
- **mise à l'échelle robuste** (*robust scaling*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s264] · exercices : 12.11, 12.Q4
- *mise de côté* → voir [hold-out](#h)
- *misleading vividness* → voir [vivacité trompeuse](#v)
- *missing at random* → voir [MCAR, MAR, MNAR](#m)
- *missing completely at random* → voir [MCAR, MAR, MNAR](#m)
- *missing indicator* → voir [colonne indicatrice (de valeur manquante)](#c)
- *missing not at random* → voir [MCAR, MAR, MNAR](#m)
- *missing value* → voir [valeur manquante](#v)
- *missingness* → voir [précision locale, absence, cohérence](#p)
- *mixed precision* → voir [précision mixte](#p)
- *mixture of experts* → voir [mélange d'experts (MoE)](#m)
- ML → voir [machine learning (ML)](#m)
- **ML Test Score** — enseignée : [ch. B7, Une grille : le ML Test Score][s619] · exercices : B7.15, B7.23, B7.25, B7.E3, B7.Q7
- **MLflow** — enseignée : [ch. B7, 207.5 · Suivi d'expériences : runs, métriques, artefacts, comparaison][s616] · exercices : B7.2, B7.14, B7.25, B7.E2, B7.R1
- MLM → voir [modèle de langue masqué (MLM)](#m)
- MLP → voir [réseau feed-forward, MLP](#r)
- **MM-DiT** (*multimodal diffusion transformer*) — enseignée : [ch. B5, Les transformers de diffusion][s570] · exercices : B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11
- **MMD** (*maximum mean discrepancy*) — enseignée : [ch. 28, 28.4.6 · Discussion][s438] · exercices : 28.3, 28.8–9, 28.24–25, 28.E4, 28.Q7
- MNAR → voir [MCAR, MAR, MNAR](#m)
- *mode* → voir [moyenne, médiane, mode](#m)
- mode → voir [moyenne, médiane, mode](#m)
- *mode collapse* → voir [effondrement de mode](#e)
- **mode entraînement, mode évaluation** (*training mode, evaluation mode*) — enseignée : [ch. 20, 20.4.3 · Le dropout][s21] · exercices : 20.3, 20.5, 20.16, 20.24–25, 20.28–29, 20.Q5, 20.R1, 20.R3 (+ 2)
- mode évaluation → voir [mode entraînement, mode évaluation](#m)
- **mode inférence** (*inference mode*) — enseignée : [ch. 23, 23.5.1 · Remodeler][s371] · exercices : 23.1, 23.12, 23.Q6
- *model* → voir [modèle](#m)
- *model artifact* → voir [artefact de modèle](#a)
- *model card* → voir [fiche modèle](#f)
- *model persistence* → voir [persistance (d'un modèle)](#p)
- *model-agnostic* → voir [méthode agnostique, méthode spécifique](#m)
- *model-specific method* → voir [méthode agnostique, méthode spécifique](#m)
- **modèle** (*model*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s38] · exercices : 0B.E1, 1.1, 1.16–17, 1.E2, 1.Q5, 3.E5, 6.E3, 7.24, 7.Q6 (+ 83) · approfondie : ch. 10, 20, 23, B1, B6, B7, B8 · réutilisée : ch. 23, 27, 28, B1, B2, B6, B7, B8
- **modèle à espace d'états (SSM)** (*state space model*) — enseignée : [ch. 22, Au-delà du livre : après les RNN, Transformers et modèles à espace…][s367]
- **modèle d'IA à usage général** (*general-purpose AI model*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s519] · exercices : B4.10, B4.Q1–Q2, B4.Q7
- **modèle de base** (*base model*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s519] · exercices : B1.27, B4.10, B4.Q1–Q2, B4.Q7
- **modèle de Bradley-Terry** (*Bradley-Terry model*) — enseignée : [ch. B8, Le modèle de récompense de Bradley-Terry][s650] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- **modèle de chat** (*chat model*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s519] · exercices : B4.10, B4.Q1–Q2, B4.Q7
- **modèle de cohérence** (*consistency model*) — enseignée : [ch. B5, La distillation][s563] · exercices : B5.5, B5.8, B5.17–18, B5.24–25, B5.30–31, B5.Q7–Q8 (+ 1)
- *modèle de conversation* → voir [chat template](#c)
- **modèle de diffusion** (*diffusion model*) — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s18] · exercices : B5.E1, B5.E3
- **modèle de fondation** (*foundation model*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s38] ; [ch. B1, Qu'est-ce qu'un modèle de fondation ?][s127] · exercices : 1.1, 1.16–17, 1.E2, 1.Q5, B1.5, B1.15, B1.24, B1.Q10–Q11 (+ 2)
- **modèle de fondation tabulaire** (*tabular foundation model*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s226]
- **modèle de langue causal (auto-régressif)** (*causal language model*) — enseignée : [ch. B3, Le décodeur : GPT et le modèle de langue causal][s507] · exercices : B3.5, B3.11, B3.19, B3.E3, B3.Q5, B3.Q9
- **modèle de langue masqué (MLM)** (*masked language model*) — enseignée : [ch. B3, L'encodeur : BERT et le modèle de langue masqué][s508] · exercices : B3.5, B3.11, B3.19, B3.E3, B3.Q5, B3.Q9
- **modèle de raisonnement** (*reasoning model*) — enseignée : [ch. B4, Le raisonnement pas à pas][s528] · exercices : B4.15–16, B4.E1, B4.Q6
- **modèle de récompense** (*reward model*) — enseignée : [ch. B8, 208.7 · RL et modèles de langue : RLHF, modèle de récompense, DPO,…][s126] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- **modèle de récompense Bradley-Terry** — enseignée : [ch. B8, Le modèle de récompense de Bradley-Terry][s650] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- *modèle de référence* → voir [baseline](#b)
- **modèle dérivé** (*derived model*) — enseignée : [ch. B1, Publier un modèle dérivé][s460] · exercices : B1.7, B1.14, B1.26–27, B1.E5, B1.Q11
- **modèle instruct** (*instruct model*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s519] · exercices : B4.10, B4.14, B4.Q1–Q2, B4.Q7
- **modèle pré-entraîné** (*pretrained model*) — enseignée : [ch. 23, 23.9.4 · Utiliser un modèle pré-entraîné][s378] · exercices : 23.10–11, 23.24, 29.Q1, B1.9, B1.E5, B1.Q3 · approfondie : ch. 20 · réutilisée : ch. B1, B2
- **modèle pré-entraîné torchvision** — enseignée : [ch. 20, 20.6 · Quelques exemples][s705] ; [ch. 23, 23.9.4 · Utiliser un modèle pré-entraîné][s378] · exercices : 20.6, 20.9, 20.18, 20.21, 20.26, 20.Q10, 23.10–11, 23.24 · réutilisée : ch. B1
- **modèle unigramme, bigramme, trigramme** (*unigram, bigram, trigram model*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s202] · exercices : 6.26
- **modèles de fondation (DINOv2, CLIP)** — enseignée : [ch. B1, 201.8 · Modèles de fondation : auto-supervisé (DINOv2),…][s455] · exercices : B1.5, B1.15, B1.24–25, B1.E4, B1.Q10–Q11
- *module* → voir [module, paquet](#m), [`nn.Module`](#n)
- **module, paquet** (*module, package*) — enseignée : [ch. 0A, 100.6 · Modules, erreurs et fichiers][s64] · exercices : 0A.24, 0A.26, 0A.Q8, 20.18, B3.18, B4.27, B7.9, PF.10 · approfondie : ch. 20 · réutilisée : ch. 11, 21, 22, 23, 25, B2, B4, B7
- **modules et imports** — enseignée : [ch. 0A, 100.6.1 · Modules, imports et bibliothèque standard][s671] · exercices : 0A.24, 0A.Q8
- *modus ponens* → voir [*modus ponens*, *modus tollens*](#m)
- ***modus ponens*, *modus tollens*** (*modus ponens, modus tollens*) — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5, 11.15, 11.Q7
- *modus tollens* → voir [*modus ponens*, *modus tollens*](#m)
- modus tollens → voir [*modus ponens*, *modus tollens*](#m)
- MoE → voir [mélange d'experts (MoE)](#m)
- **moindres carrés** (*least squares, ordinary least squares, OLS*) — enseignée : [ch. 9, Les moindres carrés (au-delà du livre)][s232] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 4)
- **momentum** (*élan*) — enseignée : [ch. 19, 19.5.1 · Le momentum][s318] · exercices : 19.2, 19.10, 19.13, 19.16, 19.20–21, 19.24, 19.E1, 19.Q9, CP4.3 · réutilisée : ch. 23, 26, B5
- **momentum de Nesterov** (*Nesterov momentum, Nesterov accelerated gradient*) — enseignée : [ch. 19, 19.5.2 · Le momentum de Nesterov][s319] · exercices : 19.3, 19.13, 19.16, 19.21, 19.Q10
- **monôme** (*monomial*) — enseignée : [ch. 15, 15.9.5 · Les pipelines][s16] · exercices : 15.1, 15.4, 15.6, 15.22–23, 15.28–29, 15.E3, 15.Q11
- *monomial* → voir [monôme](#m)
- *Monte Carlo error* → voir [bruit de Monte-Carlo](#b)
- *Monte Carlo method* → voir [méthode de Monte-Carlo](#m)
- **montée de gradient** (*gradient ascent*) — enseignée : [ch. 5, Descendre une surface pas à pas][s187] · exercices : 5.2–3, 5.5, 5.7–10, 5.Q7–Q8, 5.Q10 (+ 18) · approfondie : ch. 28 · réutilisée : ch. 21, 28
- **montée de gradient sur l'entrée** (*gradient ascent on the input*) — enseignée : [ch. 28, 28.2.2 · Visualiser un filtre][s429] · exercices : 21.28, 28.12–13, 28.15, 28.Q2
- *Moore-Penrose pseudo-inverse* → voir [solution de norme minimale, pseudo-inverse](#s)
- *morceau* → voir [chunk](#c)
- *moving average* → voir [moyenne mobile](#m)
- **moyen terme** (*middle term*) — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5, 11.15, 11.Q7
- **moyen terme non distribué** (*undistributed middle*) — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s255] · exercices : 11.4, 11.15–16, 11.Q8
- *moyenne des tokens* → voir [mean pooling](#m)
- **moyenne harmonique** (*harmonic mean*) — enseignée : [ch. 3, 3.7.11 · Le score F1][s164] · exercices : 3.2, 3.6, 3.8, 3.16, 3.25, 3.E4, 3.Q12
- **moyenne incrémentale** — enseignée : [ch. 11][s689] · exercices : 11.2, CP2.11
- **moyenne interquartile (IQM)** (*interquartile mean*) — enseignée : [ch. B8, Rapporter plusieurs graines][s649] · exercices : B8.8, B8.15, B8.21–22, B8.27–28, B8.E3, B8.E5, B8.Q10
- **moyenne macro, micro, pondérée** (*macro, micro, weighted average*) — enseignée : [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s33] · exercices : 0B.4, 0B.Q3, 3.5–6, 3.19, 3.22, 3.25, 3.E4, 3.Q11, 14.24 · approfondie : ch. 0B · réutilisée : ch. 16, 21
- **moyenne mobile** (*moving average*) — enseignée : [ch. 0B, 101.1.4 · Moyenne, moyenne pondérée, somme pondérée, moyenne mobile][s39] · exercices : 0B.4, 0B.35, 0B.Q3, 19.8 · réutilisée : ch. 19, 20, 26, B5
- **moyenne mobile exponentielle** (*exponential moving average, EMA*) — enseignée : [ch. 19, 19.5.4 · Adadelta et RMSprop][s322] · exercices : 19.4, 19.8, 19.13, 19.17, 19.Q11 · réutilisée : ch. 20, 26, B5
- **moyenne pondérée** (*weighted average*) — enseignée : [ch. 0B, 101.1.4 · Moyenne, moyenne pondérée, somme pondérée, moyenne mobile][s39] · exercices : 0B.4, 0B.35, 0B.Q3
- **moyenne, médiane, mode** (*mean, median, mode*) — enseignée : [ch. 2, Résumer une liste de nombres : moyenne, médiane, mode][s133] · exercices : 0B.Q3, 2.1–2, 2.13, 2.16, 2.19, 2.E1, 2.Q1–Q2, 2.R2 (+ 30) · approfondie : ch. 0B, 3, 11, 18, 27 · réutilisée : ch. 1, 3, 5, 7, 8, 9, 12, 19, 20, 25, 26, 27, B5
- **moyennes macro/micro/pondérée** — enseignée : [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s33] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11
- **MQA, GQA** (*multi-query attention, grouped-query attention*) — enseignée : [ch. B3, Moins de têtes pour les clés : MQA et GQA][s501] · exercices : B3.3, B3.11, B3.14, B3.18, B3.24, B3.E1, B3.Q6
- **MRR** (*mean reciprocal rank, rang réciproque moyen*) — enseignée : [ch. B4, Mesurer une recherche : recall@k et MRR][s537] · exercices : B4.2, B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **MSE, RMSE, MAE** (*erreur quadratique moyenne, sa racine, erreur absolue moyenne*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s26] · exercices : 9.1, 9.12, 9.14, 9.E2, 9.Q1, 14.4, 14.28, 24.25, 24.Q10 · approfondie : ch. 18, 24 · réutilisée : ch. 14, 24
- *multi-armed bandit* → voir [bandit manchot](#b)
- *multi-class* → voir [classification binaire, multi-classe, multi-étiquette](#c)
- multi-classe → voir [classification binaire, multi-classe, multi-étiquette](#c)
- *multi-head attention* → voir [attention multi-têtes (MHA)](#a)
- *multi-label classification* → voir [classification binaire, multi-classe, multi-étiquette](#c)
- *multi-query attention* → voir [MQA, GQA](#m)
- *multilayer perceptron* → voir [réseau feed-forward, MLP](#r)
- *multimodal diffusion transformer* → voir [MM-DiT](#m)
- *multinomial naive Bayes* → voir [Naïve Bayes (gaussien, multinomial)](#n)
- multinoulli → voir [loi catégorielle, multinoulli](#l)
- *multiple comparisons* → voir [comparaisons multiples](#c)
- **multiplication-addition (MAC)** (*multiply-accumulate*) — enseignée : [ch. 24, 24.4.3 · Les couches de convolution][s388] · exercices : 24.2, 24.19–20, 24.Q7, 24.R2
- *multiply-accumulate* → voir [multiplication-addition (MAC)](#m)
- *multivariable chain rule* → voir [règle de la chaîne à plusieurs chemins](#r)
- *multivariate transformation* → voir [transformation univariée, multivariée](#t)
- multivariée → voir [transformation univariée, multivariée](#t)
- **mutable / immuable** (*mutable / immutable*) — enseignée : [ch. 0A, 100.2.2 · Chaînes de caractères et f-strings][s51] · exercices : 0A.2, 0A.14–15, 0A.Q3
- *mutable / immutable* → voir [mutable / immuable](#m)
- *mutation testing* → voir [test par mutation](#t)
- *mutual information* → voir [information mutuelle](#i)
- **mylearn.linalg_basics** — enseignée : [ch. 0B, 101.3.1 · Composantes, somme, produit par un scalaire][s87] · exercices : 0B.8, 0B.38, 0B.40, 0B.Q8

## N

- *« n choose k »* → voir [coefficient binomial](#c)
- *naive Bayes* → voir [Naïve Bayes (gaussien, multinomial)](#n)
- **Naïve Bayes (gaussien, multinomial)** (*naive Bayes, bayésien naïf, Gaussian, multinomial naive Bayes*) — enseignée : [ch. 4, Bayes dans le ML d'aujourd'hui][s277] ; [ch. 13, 13.6 · Naïve Bayes][s177] ; [mini-projet 1][s278] · exercices : 13.6–7, 13.27–30, 13.E4, 13.Q11 · réutilisée : ch. B4
- *naive forecast* → voir [persistance (en prévision)](#p)
- *Nash equilibrium* → voir [équilibre de Nash](#e)
- **nat** — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s191] · exercices : 0B.16, 6.20, 6.Q12 · approfondie : ch. 0B · réutilisée : ch. 18
- *natural frequencies* → voir [fréquences naturelles, arbre des fréquences naturelles](#f)
- *natural frequency tree* → voir [fréquences naturelles, arbre des fréquences naturelles](#f)
- *natural logarithm* → voir [logarithme népérien](#l)
- **NCHW** — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s327] · exercices : 20.1, 20.13, 20.Q2, 23.2, 23.Q6
- *near-duplicate* → voir [quasi-doublon](#q)
- *nearest centroid classifier* → voir [classifieur du centroïde le plus proche](#c)
- *nearest neighbour* → voir [plus proche voisin (1-NN)](#p)
- **négation de l'antécédent** (*denying the antecedent*) — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s255] · exercices : 11.4, 11.15–16, 11.Q8
- négative → voir [classe positive, négative](#c)
- *negative class* → voir [classe positive, négative](#c)
- *negative predictive value* → voir [valeur prédictive négative (VPN)](#v)
- *negative sampling* → voir [échantillonnage négatif](#e)
- *negative transfer* → voir [transfert négatif](#t)
- *nested cross-validation* → voir [validation croisée imbriquée](#v)
- **Nesterov** — enseignée : [ch. 19, 19.5.2 · Le momentum de Nesterov][s319] · exercices : 19.3, 19.13, 19.16, 19.21, 19.Q10
- *Nesterov accelerated gradient* → voir [momentum de Nesterov](#m)
- *Nesterov momentum* → voir [momentum de Nesterov](#m)
- *neural network* → voir [réseau de neurones](#r)
- *neural style transfer* → voir [transfert de style neuronal](#t)
- *neuromorphic chip* → voir [puce neuromorphique](#p)
- *neuron* → voir [neurone](#n), [neurone biologique](#n)
- **neurone** (*neuron, unit*) — enseignée : [ch. 1, 1.7 · Deep learning][s27] · exercices : 1.1, 1.3, 1.10, 1.23, 1.E4, 1.Q10–Q11, 10.Q1–Q2, 10.Q4 (+ 19) · approfondie : ch. 10, 17, 18 · réutilisée : ch. 16, 26
- **neurone artificiel** — enseignée : [ch. 10, 10.3 · Les neurones artificiels][s242] · exercices : 10.8–10, 10.E4, 10.Q4 · réutilisée : ch. 16
- **neurone biologique** (*neuron*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s241] · exercices : 10.8–9, 10.E4, 10.Q2–Q3
- **neurone formel** (*McCulloch-Pitts neuron, threshold logic unit*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s208] · exercices : 10.1, 10.3, 10.6, 10.11–12, 10.17, 10.Q5–Q6, 10.R2–R3 (+ 4)
- **neurone mort** (*dead neuron*) — enseignée : [ch. 16, Au-delà du livre : diagnostiquer l'évanouissement et l'explosion…][s298] ; [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s299] · exercices : 17.2, 17.5, 17.12, 17.18, 17.20, 17.23–24, 17.Q6–Q7, 17.R1 (+ 2)
- **neurotransmetteur** (*neurotransmitter*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s241] · exercices : 10.8–9, 10.E4, 10.Q2–Q3
- *neurotransmitter* → voir [neurotransmetteur](#n)
- **NFC, NFD, NFKC** (*canonical composition, canonical decomposition, compatibility normalization*) — enseignée : [ch. B2, Un caractère affiché, plusieurs points de code][s465] · exercices : B2.1, B2.18, B2.Q2–Q3
- NFD → voir [NFC, NFD, NFKC](#n)
- NFKC → voir [NFC, NFD, NFKC](#n)
- **niveau de log** (*log level*) — enseignée : [ch. B7, Variables d'environnement et secrets][s608] · exercices : B7.23–25, B7.Q4
- **nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d** — enseignée : [ch. 24, 24.6.2 · Construire un modèle fonctionnel][s715] · exercices : 24.28, 24.31, 24.Q7, 24.Q12 · approfondie : ch. 21 · réutilisée : ch. 22
- **nn.Embedding** — enseignée : [ch. B2, 202.7 · Embeddings : de one-hot à `nn.Embedding`, entraînement et…][s74] · exercices : B2.5, B2.9, B2.25, B2.E3, B2.Q9, B2.R1 · réutilisée : ch. 24, B3
- **`nn.Module`** (*module*) — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s2] · exercices : 20.18 · réutilisée : ch. 21, 22, 23, 25
- **nn.RNN, nn.LSTM, nn.GRU** — enseignée : [ch. 22, 22.5 · Organiser les entrées][s355] · exercices : 22.3, 22.13, 22.15, 22.21–23, 22.29, 22.E4, 22.Q5, 22.R3 (+ 2) · approfondie : ch. 24
- **nn.Sequential, nn.Flatten, nn.Linear** — enseignée : [ch. 23, 23.6.2 · Créer le modèle][s708] · exercices : 23.3, 23.8, 23.16, 23.29, 24.28
- **No Free Lunch** — enseignée : [ch. 11, 11.2.3 · L'optimisation][s249] · exercices : 11.12, 11.E3, 11.Q5, 19.Q12 · réutilisée : ch. 19
- *no free lunch theorem* → voir [théorème No Free Lunch](#t)
- nœud interne → voir [racine, feuille, nœud interne](#r)
- *noise* → voir [point cœur, point de bord, bruit](#p)
- *noise prediction* → voir [prédiction du bruit](#p)
- *noise reduction* → voir [débruitage](#d)
- *noise schedule* → voir [planning de bruit](#p)
- *noisy ReLU* → voir [ReLU bruitée](#r)
- **nombre dénormalisé** (*subnormal number, denormal number*) — enseignée : [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s35]
- *nominal data* → voir [donnée nominale](#d)
- non ciblée → voir [attaque ciblée, non ciblée](#a)
- non paramétrique → voir [classifieur paramétrique, non paramétrique](#c), [mémoire paramétrique, non paramétrique](#m)
- *non-parametric classifier* → voir [classifieur paramétrique, non paramétrique](#c)
- *non-parametric memory* → voir [mémoire paramétrique, non paramétrique](#m)
- *non-saturating loss* → voir [loss non saturante](#l)
- *norm* → voir [norme](#n)
- *normal distribution* → voir [loi normale, gaussienne](#l)
- *normal equations* → voir [équations normales](#e)
- **normalisation de texte** — enseignée : [ch. B2, Le même texte, un autre prix selon la langue][s724] · exercices : B2.8, B2.11, B2.14, B2.20–21, B2.31, B2.Q8, B2.R3
- **normalisation et découpage stratifié** — enseignée : [ch. 23][s380]
- **normalisation min-max** (*min-max scaling*) — enseignée : [ch. 12, 12.5.1 · La normalisation min-max][s265] · exercices : 12.2, 12.6, 12.17, 12.E2, 12.Q5
- **normaliser (une distribution)** (*normalize*) — enseignée : [ch. 2][s135] · exercices : 12.Q5, 23.R3 · réutilisée : ch. 4, 13
- *normalize* → voir [normaliser (une distribution)](#n)
- **norme** (*norm*) — enseignée : [ch. 0B, 101.3.2 · Norme et distance][s88] · exercices : 0B.8, 0B.19, 0B.39–41, 0B.Q8, CP1.2 · réutilisée : ch. 1, 2, 5, 7, 9, 10, 12, 13, 19, 25, 28, B2
- **norme de Frobenius** (*Frobenius norm*) — enseignée : [ch. 28, 28.4.4 · La loss de style][s435] · exercices : 28.1, 28.4–5, 28.19, 28.22, 28.Q9
- **notations Σ et Π** — enseignée : [ch. 0B, 101.1.3 · Notations Σ et Π][s14] · exercices : 0B.3, 0B.24, 0B.32, 0B.35, 0B.Q3, 0B.R1
- **notebook** (*carnet*) — enseignée : [ch. 0A, 100.1 · Prise en main : notebooks, terminal et workbook][s28] · exercices : 0A.E4, 0A.Q1, 8.24, CP4.9, CP5.5–6, B7.6, B7.9, B7.16, B7.Q1 (+ 1) · réutilisée : ch. 0B
- **noyau** (*kernel*) — enseignée : [ch. 0A, 100.1.1 · Cellules, noyau, ordre d'exécution et « Tout exécuter »][s48] · exercices : 0A.13, 0A.E4, 0A.Q1, 13.5, 13.18, 13.Q6, CP3.9, 21.20 · approfondie : ch. 13 · réutilisée : ch. 28
- **noyau (d'un SVM)** (*kernel*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s244] · exercices : 0A.Q1, 13.4–5, 13.10, 13.16–18, 13.36, 13.Q5–Q6 (+ 2) · approfondie : ch. 0A · réutilisée : ch. 28
- **noyau (d'une convolution)** (*kernel, filter*) — enseignée : [ch. 21, 21.2.5 · Le noyau][s25] · exercices : 0A.Q1, 13.5, 13.18, 13.Q6, CP3.9, 21.20, 21.Q2 · approfondie : ch. 0A, 13 · réutilisée : ch. 28
- **noyau (d'une opération)** (*kernel*) — enseignée : [ch. B3, Le masque causal quand il y a moins de requêtes que de clés][s513] · exercices : 0A.Q1, 13.Q6, B3.2, B3.11–12, B3.17–18, B3.29, B3.Q4–Q5 (+ 4) · approfondie : ch. 0A, 13 · réutilisée : ch. 28
- **noyau de Shapley** (*Shapley kernel*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s582] · exercices : B6.2, B6.7, B6.11, B6.15–17, B6.E1–E2, B6.Q5–Q6
- **noyau RBF (gaussien)** (*RBF kernel*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s244] · exercices : 13.4–5, 13.10, 13.16–18, 13.36, 13.Q5–Q6
- **np.random.default_rng, graine, permutation, choice** — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s6] · exercices : 0A.5, 0A.31, 0A.66, 0A.E4, 0A.Q11, 2.14, 2.E5, 2.Q3, 8.26, 14.19 (+ 19) · approfondie : ch. 2, 8, 14, 26 · réutilisée : ch. 0B, 1, 2, 3, 7, 8, 16, B6
- *NPU* → voir [GPU, TPU, NPU](#g)
- NPU → voir [GPU, TPU, NPU](#g)
- *NPV* → voir [valeur prédictive négative (VPN)](#v)
- **nuage de points** (*scatter plot*) — enseignée : [ch. 2, 2.8 · Covariance et corrélation][s150] · exercices : 2.26–28, 2.Q12
- **nuage gaussien** (*blob*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s244] · exercices : 13.4–5, 13.10, 13.16–18, 13.36, 13.Q5–Q6
- *nucleus sampling* → voir [top-k, top-p](#t)
- *nuisance parameter* → voir [paramètre de nuisance](#p)
- *null condition* → voir [condition vide](#c)
- *null hypothesis* → voir [hypothèse nulle](#h)
- *null player* → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *numerical data* → voir [donnée numérique (quantitative)](#d)
- **NumPy : arrays, dtype, shape, indexation, masques, vues/copies** — enseignée : [ch. 0A, 100.8.1 · Créer des arrays : `dtype`, `shape`, `ndim`][s68] · exercices : 0A.7, 0A.27–28, 0A.Q10, 0B.R2, 20.13, B3.17, B4.1, B4.Q3 · approfondie : ch. 26, B3 · réutilisée : ch. 0B, 1, 20, 21
- **NumPy : calcul vectorisé, broadcasting, réductions par axe, tri, argmax, unique** — enseignée : [ch. 0A, 100.8.4 · Broadcasting][s69] · exercices : 0A.8, 0A.51–52, 0A.56, 0A.64, 0A.Q11, 1.1, 16.3 · réutilisée : ch. 0B, 2, 7, 16, 21
- **NumPy : reshape, transpose, empilement** — enseignée : [ch. 0A, 100.8.6 · `reshape`, transposition, empilement][s675] · exercices : 0A.7, 0A.53, 0A.Q11 · réutilisée : ch. 1, 20

## O

- *object* → voir [classe, objet, méthode, attribut](#c)
- **objectif clippé, PPO** (*clipped objective, proximal policy optimization*) — enseignée : [ch. B8, 208.5 · PPO : ratio de probabilités, objectif clippé, entropie,…][s645] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 6)
- **objectif de substitution** (*surrogate objective*) — enseignée : [ch. B8, Réutiliser un rollout sans tout casser][s644] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- objet → voir [classe, objet, méthode, attribut](#c)
- *objet de mise à l'échelle* → voir [scaler](#s)
- **observabilité partielle** (*partial observability*) — enseignée : [ch. 26, 26.3.5 · La vue d'ensemble][s404] · exercices : 26.10, 26.26, 26.Q5
- *observation* → voir [hypothèse, observation](#h)
- observation → voir [hypothèse, observation](#h)
- *observational Shapley* → voir [Shapley interventionnel, observationnel](#s)
- observationnel → voir [Shapley interventionnel, observationnel](#s)
- **octave** — enseignée : [ch. 28, 28.2.2 · Visualiser un filtre][s429] · exercices : 28.12–13, 28.15, 28.Q2
- *odds* → voir [cote](#c)
- off-policy → voir [on-policy, off-policy](#o)
- *offline algorithm* → voir [algorithme en ligne, hors ligne](#a)
- *offline RL* → voir [RL hors ligne](#r)
- **offsets** (*décalages*) — enseignée : [ch. B2, La pré-tokenisation][s466] · exercices : B2.4, B2.16–18, B2.30, B2.E3, B2.Q5–Q6
- *Okapi BM25* → voir [BM25](#b)
- *OLS* → voir [moindres carrés](#m)
- **on-policy, off-policy** (*sur politique, hors politique*) — enseignée : [ch. 26, 26.7 · SARSA][s410] · exercices : 26.4, 26.20, 26.28, 26.E1, 26.Q11, B8.Q2, B8.Q9
- **one-hot** (*encodage one-hot, codage disjonctif*) — enseignée : [ch. 0A][s52] ; [ch. B2, 202.7 · Embeddings : de one-hot à `nn.Embedding`, entraînement et…][s74] · exercices : 12.1, 12.Q3, 22.Q4, 23.Q9, B2.5, B2.9, B2.25, B2.E3, B2.Q9, B2.R1 (+ 1) · approfondie : ch. 12 · réutilisée : ch. 18, 20, 22, 23, 24
- **one-hot encoding** — enseignée : [ch. 12, 12.3.1 · Le one-hot][s263] · exercices : 12.1, 12.20, 12.26, 12.33, 12.E5, 12.Q3 · réutilisée : ch. 18
- *one-standard-error rule* → voir [règle d'une erreur type](#r)
- *one-to-many* → voir [one-to-many, many-to-one, many-to-many](#o)
- **one-to-many, many-to-one, many-to-many** (*one-to-many, many-to-one, many-to-many*) — enseignée : [ch. 22, 22.8.1 · Une ou plusieurs entrées et sorties][s364] · exercices : 22.8, 22.10, 22.23, 22.25, 22.28–29, 22.Q10
- *one-versus-one* → voir [un-contre-un (OvO)](#u)
- *one-versus-rest* → voir [un-contre-tous (OvR)](#u)
- *one-vs-all* → voir [un-contre-tous (OvR)](#u)
- *online* → voir [algorithme en ligne, hors ligne](#a)
- *online inference* → voir [inférence par batches, en ligne](#i)
- *online softmax* → voir [softmax en ligne](#s)
- **ONNX** (*Open Neural Network Exchange*) — enseignée : [ch. 15, Au-delà du livre : sauvegarder un modèle et ses versions][s292] · exercices : B7.21, B7.Q10
- OOB → voir [out-of-bag (OOB)](#o)
- *Open Neural Network Exchange* → voir [ONNX](#o)
- **open weights** (*poids ouverts*) — enseignée : [ch. B4, Licences : « open weights » n'est pas « open source »][s522] · exercices : B4.10, B4.Q1–Q2, B4.Q7
- *operant conditioning* → voir [conditionnement opérant](#c)
- *operating point* → voir [point de fonctionnement](#p)
- *optimal denoiser* → voir [débruiteur parfait (d'un dataset fini)](#d)
- *optimal discriminator* → voir [discriminateur optimal](#d)
- **optimisation** (*optimization*) — enseignée : [ch. 11, 11.2.3 · L'optimisation][s249] · exercices : 5.E4, 11.12, 11.E3, 11.Q1, 11.Q5, 12.R1, 15.E2, 18.R3, 24.5
- **optimisation bayésienne** (*Bayesian optimization*) — enseignée : [ch. 15, 15.9.4 · La recherche aléatoire][s288] · exercices : 15.5, 15.19–20, 15.E2, 15.Q10
- **optimiseur** (*optimizer*) — enseignée : [ch. 8, 8.2 · La boucle d'entraînement][s9] · exercices : 8.16, 8.Q1–Q2, 8.R2, 19.9, 19.23, 19.26–27, 19.Q1, 19.Q12 (+ 4) · approfondie : ch. 19, 27 · réutilisée : ch. 20, 25, 27, 28
- *optimistic bias* → voir [biais d'optimisme (du gagnant)](#b)
- *optimistic initial values* → voir [initialisation optimiste](#i)
- *optimization* → voir [optimisation](#o)
- *optimizer* → voir [optimiseur](#o)
- *optimizer state* → voir [tampon (d'un optimiseur)](#t)
- **oracle** — enseignée : [ch. 0A, 100.11.5 · mylearn : stubs, référence et tests fondés sur un oracle][s77] · exercices : 0A.26, 0A.63–66, 0A.Q2
- *order of magnitude* → voir [ordre de grandeur](#o)
- *ordinal data* → voir [donnée ordinale](#d)
- *ordinal encoding* → voir [encodage ordinal](#e)
- *ordinary least squares* → voir [moindres carrés](#m)
- **ordonnée à l'origine** (*intercept*) — enseignée : [ch. 9, 9.7 · Ajuster une droite avec la règle de Bayes][s233] · exercices : 0B.6, 9.2, 9.7, 9.26–27, 9.Q11 · réutilisée : ch. 1
- **ordre C, ligne par ligne** (*C order, row-major order*) — enseignée : [ch. 23, 23.5.1 · Remodeler][s371] · exercices : 19.12, 23.1, 23.12, 23.Q6
- **ordre de grandeur** (*order of magnitude*) — enseignée : [ch. 0B, 101.1.1 · Puissances, racines, notation scientifique, ordres de…][s78] · exercices : 0B.1, 0B.30, 0B.33, 0B.54, 0B.Q1
- **ordre topologique** (*topological order*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s294] · exercices : 16.2, 16.4, 16.9, 16.16–18, 16.Q1–Q4 (+ 1) · réutilisée : ch. 18
- *orthogonal* → voir [orthogonaux](#o)
- **orthogonaux** (*orthogonal*) — enseignée : [ch. 0B, 101.3.3 · Produit scalaire, angle, similarité cosinus][s89] · exercices : 0B.18–19, 0B.39–41, 0B.E2, 0B.Q8
- **oubli catastrophique** (*catastrophic forgetting*) — enseignée : [ch. B1, 201.7 · Quand le transfert échoue : distance de domaine, transfert…][s452] · exercices : B1.6, B1.9–10, B1.16, B1.22–23, B1.E2, B1.Q8–Q9, B1.R1
- **out-of-bag (OOB)** (*hors du sac*) — enseignée : [ch. 14, 14.4 · Le bagging][s146] · exercices : 14.3, 14.8, 14.12, 14.15–17, 14.E1, 14.Q4–Q5, 14.Q10
- *outcome* → voir [événement, issue](#e)
- *outer product* → voir [produit extérieur](#p)
- *outlier* → voir [point aberrant](#p)
- *output layer* → voir [couche de sortie](#c)
- *output projection* → voir [projection de sortie](#p)
- *OvA* → voir [un-contre-tous (OvR)](#u)
- *overestimation bias* → voir [surestimation](#s)
- *overfit-one-batch test* → voir [test d'overfitting d'un mini-batch](#t)
- **overfitting** (*surapprentissage, surajustement*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s26] · exercices : 9.1, 9.12, 9.14, 9.E2, 9.E4, 9.Q1, 9.Q4, 13.E3, 20.25, 20.R3 (+ 3) · réutilisée : ch. 11, 13, 20, 23, 24
- *overlap* → voir [recouvrement](#r)
- *override* → voir [surcharge (de configuration)](#s)
- *overwhelming exception* → voir [exception écrasante](#e)
- OvO → voir [un-contre-un (OvO)](#u)
- OvR → voir [un-contre-tous (OvR)](#u)

## P

- **p-valeur** (*p-value*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s222] · exercices : 8.2, 8.4, 8.8, 8.16, 8.18, 8.26–27, 8.E2, 8.E4, 8.Q11
- *p-value* → voir [p-valeur](#p)
- *p50* → voir [p50, p95, p99](#p)
- **p50, p95, p99** (*p50, p95, p99*) — enseignée : [ch. B7, Latence, percentiles et débit][s622] · exercices : B7.3, B7.5, B7.16–19, B7.Q8–Q9, B7.R2–R3 (+ 2)
- *p95* → voir [p50, p95, p99](#p)
- p95 → voir [p50, p95, p99](#p)
- *p99* → voir [p50, p95, p99](#p)
- p99 → voir [p50, p95, p99](#p)
- *package* → voir [module, paquet](#m), [paquet](#p)
- **padding** (*remplissage, marge*) — enseignée : [ch. 20, 20.4.5 · La convolution (aperçu)][s23] ; [ch. 21, 21.3.4 · Le padding][s24] · exercices : 20.6, 20.20, 20.Q7, 21.3, 21.12, 21.15–16, 21.E2, 21.Q5, 24.R2 (+ 2) · approfondie : ch. B3
- *padding mask* → voir [masque de padding](#m)
- *PagedAttention* → voir [PagedAttention (attention paginée)](#p)
- **PagedAttention (attention paginée)** (*PagedAttention*) — enseignée : [ch. B3, Ce que coûte le cache][s517] · exercices : B3.8, B3.23, B3.E4, B3.Q11
- *paired bootstrap* → voir [bootstrap apparié](#b)
- *paired comparison* → voir [comparaison appariée](#c)
- **pandas : read_csv, head, info, describe, loc, iloc, filtres, isna, dropna, fillna, duplicated, value_counts, groupby, to_numpy** — enseignée : [ch. 0A, 100.9.1 · `DataFrame` et `Series` : `read_csv`, `head`, `info`,…][s73] · exercices : 0A.4, 0A.32–34, 0A.57, 0A.Q12, 1.R2, 3.R3, 21.Q4, 21.Q6 (+ 8) · approfondie : ch. 21 · réutilisée : ch. 1, 2, 3, 12, 29, B1, B6
- **panorama 2026** — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s18] · exercices : 1.Q11
- **paquet** (*package*) — enseignée : [ch. B7, Une structure de dépôt][s603] · exercices : B7.6, B7.8–9, B7.25, B7.E1, B7.Q1–Q2, PF.10
- **par échantillon, par feature, par élément** (*samplewise, featurewise, elementwise*) — enseignée : [ch. 12, 12.9 · Par échantillon, par feature, par élément][s269] · exercices : 12.5, 12.Q10 · réutilisée : ch. 22
- par élément → voir [par échantillon, par feature, par élément](#p)
- par feature → voir [par échantillon, par feature, par élément](#p)
- **paradoxe de Simpson** (*Simpson's paradox*) — enseignée : [ch. 2][s135]
- *param group* → voir [groupe de paramètres](#g)
- *parameter* → voir [fonction, paramètre, argument](#f), [paramètre (d'un modèle)](#p)
- *parameter group* → voir [groupe de paramètres](#g)
- *parameter-efficient fine-tuning* → voir [PEFT](#p)
- *parameterized policy* → voir [politique paramétrée](#p)
- paramètre → voir [fonction, paramètre, argument](#f)
- **paramètre (d'un modèle)** (*parameter*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s38] · exercices : 0A.23, 0A.Q6, 1.1, 1.16–17, 1.E2, 1.Q5, 5.9, 13.16, 16.12 (+ 31) · approfondie : ch. 16 · réutilisée : ch. 2, 21, 22, 23, 25, B8
- paramètre C → voir [marge souple, paramètre C](#m)
- **paramètre de décroissance** (*decay parameter*) — enseignée : [ch. 19, 19.3.2 · Changer le learning rate au fil du temps][s315] · exercices : 19.1, 19.15, 19.21, 19.Q6
- **paramètre de nuisance** (*nuisance parameter*) — enseignée : [ch. 4][s176]
- *parametric* → voir [classifieur paramétrique, non paramétrique](#c), [mémoire paramétrique, non paramétrique](#m)
- *parametric blending* → voir [mélange paramétrique](#m)
- *parametric ReLU* → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- **paramétrique vs non paramétrique** — enseignée : [ch. 13, 13.2 · Paramétrique ou non paramétrique ?][s251] · exercices : 13.8, 13.12, 13.31–32, 13.35, 13.E5, 13.Q2
- *parametrized test* → voir [test paramétré](#t)
- **parcimonieux** (*sparse*) — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s231] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 2)
- *parent* → voir [parent, enfant ; ancêtre, descendant](#p)
- **parent, enfant ; ancêtre, descendant** (*parent, child, ancestor, descendant*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s294] · exercices : 16.2, 16.4, 16.9, 16.16–18, 16.Q1–Q4 (+ 2)
- **parité démographique** (*demographic parity, statistical parity*) — enseignée : [ch. B6, 206.9 · Équité : indicateurs de groupe (parité démographique,…][s596] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- **parité prédictive** (*predictive parity*) — enseignée : [ch. B6, Les critères][s597] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- **part de variance expliquée** (*explained variance ratio*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s152] · exercices : 12.7–8, 12.22–23, 12.27, 12.Q9
- **partage de poids** (*weight sharing*) — enseignée : [ch. 21, 21.2.3 · Le partage de poids][s338] · exercices : 21.4, 21.7, 21.31, 21.E1, 21.Q2, 21.R2
- *partial* → voir [fine-tuning partiel, complet](#f)
- *partial dependence* → voir [dépendance partielle (PDP)](#d)
- *partial derivative* → voir [dérivée partielle](#d)
- *partial observability* → voir [observabilité partielle](#o)
- **partie entière (plancher / plafond)** (*floor / ceiling*) — enseignée : [ch. 0B, 101.1.2 · Valeur absolue, inégalités, partie entière, fonction signe][s79] · exercices : 0B.2, 0B.33, 0B.36, 0B.Q2 · réutilisée : ch. 21
- *partitionnement* → voir [clustering](#c)
- **pas de temps** (*time step*) — enseignée : [ch. 22, 22.4 · La structure d'une cellule RNN][s357] · exercices : 22.2, 22.15, 22.18, 22.Q3, 22.Q5, 24.27, B5.19, B5.21, B5.Q5
- passe arrière → voir [passe avant, passe arrière](#p)
- **passe avant matricielle** — enseignée : [ch. 16, Au-delà du livre : la passe avant en matrices][s305]
- **passe avant, passe arrière** (*forward pass, backward pass*) — enseignée : [ch. 16, Au-delà du livre : la passe avant en matrices][s305] · exercices : 16.2, 16.Q3, 18.Q11, CP4.2, 25.28 · réutilisée : ch. 18
- **pasting** — enseignée : [ch. 14][s281]
- *patch* → voir [patch (ViT)](#p)
- **patch (ViT)** (*patch*) — enseignée : [ch. B3, 203.10 · Coût et passage à l'échelle : complexité quadratique,…][s354] · exercices : 21.Q12, B3.8, B3.10, B3.27–28, B3.E2, B3.Q12
- *patience* → voir [patience, `min_delta`](#p)
- **patience, `min_delta`** (*patience*) — enseignée : [ch. 9, 9.4 · L'early stopping][s225] · exercices : 9.5, 9.9, 9.20, 9.Q6, 23.6, 23.27 · réutilisée : ch. 20
- **PCA** (*analyse en composantes principales, principal component analysis, ACP*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s152] · exercices : 12.7–8, 12.22–23, 12.27–28, 12.30, 12.E4, 12.Q9, 15.Q8 (+ 3) · réutilisée : ch. 20, 21, B1, B2
- PDP → voir [dépendance partielle (PDP)](#d)
- *peak signal-to-noise ratio* → voir [PSNR](#p)
- *peaking phenomenon* → voir [phénomène de Hughes](#p)
- *Pearson correlation coefficient* → voir [corrélation (de Pearson)](#c)
- **PEFT** (*parameter-efficient fine-tuning*) — enseignée : [ch. B4, 204.8 · Fine-tuning : complet ou PEFT ; LoRA (rang, alpha, modules…][s539] · exercices : B4.3, B4.6, B4.11, B4.25–27, B4.E2, B4.Q10–Q11, B4.R2 (+ 1)
- **pénalité de gradient (WGAN-GP)** (*gradient penalty*) — enseignée : [ch. 27, 27.7 · Les difficultés][s428] · exercices : 27.6, 27.17–18, 27.22–23, 27.28, 27.E4–E5, 27.Q9
- **pénalité KL** (*KL penalty*) — enseignée : [ch. B8, Le RLHF : PPO sous une laisse KL][s651] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- **pénalité L2 et weight decay** — enseignée : [ch. 20, Au-delà du livre : régulariser un réseau][s227]
- **pente centrée** (*central difference*) — enseignée : [ch. 0B, 101.5.1 · Taux d'accroissement, tangente, dérivée][s95] · exercices : 0B.10, 0B.47, 0B.Q10
- **pente dans une direction, dérivée directionnelle** (*directional derivative*) — enseignée : [ch. 5, 5.4 · Le gradient][s186] · exercices : 5.2–3, 5.5, 5.7–10, 5.Q7–Q8, 5.Q10 (+ 15)
- *percentile* → voir [percentile, quantile, quartile](#p)
- **percentile, quantile, quartile** (*percentile, quantile, quartile*) — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s142] · exercices : 2.3–4, 2.15, 2.17–18, 2.29, 2.31, 2.Q5 · réutilisée : ch. B7
- **percentiles** — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s142] · exercices : 2.3–4, 2.15, 2.17–18, 2.29, 2.31, 2.Q5 · réutilisée : ch. B7
- **perceptron** — enseignée : [ch. 10, 10.3.1 · Le perceptron][s208] · exercices : 10.1, 10.3, 10.6, 10.11–12, 10.17, 10.Q5–Q6, 10.R2–R3 (+ 15) · réutilisée : ch. 11, 13, 16, 17
- *perceptron convergence theorem* → voir [théorème de convergence du perceptron](#t)
- *perceptron cycling theorem* → voir [théorème du cycle du perceptron](#t)
- *perceptron learning rule* → voir [règle d'apprentissage du perceptron](#r)
- **perceptron moyenné** (*averaged perceptron*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s243] · exercices : 10.4–5, 10.7, 10.10–11, 10.13, 10.16, 10.19, 10.21, 10.Q7 (+ 4)
- *perceptual loss* → voir [loss perceptuelle](#l)
- *permutation equivariance* → voir [équivariance par permutation](#e)
- *permutation importance* → voir [importance par permutation](#i)
- *permutation test* → voir [test par permutation](#t)
- *perpendicular bisector* → voir [médiatrice](#m)
- **perplexité** (*perplexity*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s191] ; [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s203] ; [ch. 28, 28.5 · Générer la suite du livre][s204] · exercices : 22.1, 22.5, 22.8, 22.12, 22.20, 22.24–27, 22.Q12 (+ 15) · approfondie : ch. B3 · réutilisée : ch. B2, B4
- *perplexity* → voir [perplexité](#p)
- **persistance (d'un modèle)** (*model persistence*) — enseignée : [ch. 15, Au-delà du livre : sauvegarder un modèle et ses versions][s292] · exercices : 22.23, 22.29
- **persistance (en prévision)** (*persistence, naive forecast*) — enseignée : [ch. 22, Au-delà du livre : prévoir une série temporelle][s366] · exercices : 22.23, 22.29 · approfondie : ch. 15
- **persistance joblib** — enseignée : [ch. 15, Au-delà du livre : sauvegarder un modèle et ses versions][s292]
- *persistence* → voir [persistance (en prévision)](#p)
- *pertinence binaire* → voir [binary relevance](#b)
- **perturbation universelle** (*universal adversarial perturbation*) — enseignée : [ch. 21, 21.10 · Les adversaires][s351] · exercices : 21.10, 21.29, 21.E4, 21.Q11
- PGD → voir [FGSM, PGD](#f)
- **phénomène de Hughes** (*Hughes phenomenon, peaking phenomenon*) — enseignée : [ch. 7, 7.6 · La malédiction de la dimension][s148] · exercices : 7.4, 7.8, 7.20, 7.30–31, 7.E4, 7.Q9–Q10
- *piecewise linear* → voir [affine par morceaux](#a)
- **pile d'appels** (*call stack*) — enseignée : [ch. 0A, 100.5.6 · Fonctions récursives][s63] · exercices : 0A.44
- *pinball loss* → voir [loss quantile](#l)
- *pip-tools* → voir [uv, pip-tools, ruff, pre-commit](#u)
- pip-tools → voir [uv, pip-tools, ruff, pre-commit](#u)
- **pipeline** (*chaîne de traitement*) — enseignée : [ch. 12, 12.10 · Transformations et validation croisée][s15] ; [ch. 15, 15.9.5 · Les pipelines][s16] · exercices : 12.9, 12.31, 12.E1, 12.Q11, 15.1, 15.4, 15.6, 15.22–23, 15.Q11 (+ 12) · approfondie : ch. B4 · réutilisée : ch. 22, 24, B1, B4, B6, B7
- **piratage de la récompense** (*reward hacking, specification gaming*) — enseignée : [ch. 26, Au-delà du livre : le RL et les modèles de langage][s415] ; [ch. B8, 208.7 · RL et modèles de langue : RLHF, modèle de récompense, DPO,…][s126] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- *pivot table* → voir [tableau croisé](#t)
- pixel focal → voir [ancre, pixel focal](#a)
- **plaidoyer spécial** (*special pleading*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s257] · exercices : 11.17–18, 11.E4, 11.Q10
- **planning (de learning rate)** (*learning-rate schedule*) — enseignée : [ch. 19, 19.3.3 · Les plannings de décroissance][s314] · exercices : 0B.7, 19.1, 19.15, 19.19, 19.24, 19.27, 19.E3, 19.Q7, 23.7, 23.25 (+ 6) · approfondie : ch. B5 · réutilisée : ch. 20, 23
- planning cosinus → voir [planning linéaire, planning cosinus (de bruit)](#p)
- **planning de bruit** (*noise schedule*) — enseignée : [ch. B5, 205.2 · Le processus direct : planning de bruit, ᾱ_t et forme…][s547] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 2)
- **planning en cosinus** (*cosine schedule, cosine annealing*) — enseignée : [ch. 0B, 101.2.6 · Cosinus (aperçu)][s86] · exercices : 0B.7, 0B.36, 0B.Q6
- **planning linéaire, planning cosinus (de bruit)** (*linear, cosine noise schedule*) — enseignée : [ch. B5, Le planning linéaire de Ho et coll.][s549] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 3)
- **planning par paliers** (*step decay, interval decay*) — enseignée : [ch. 19, 19.3.3 · Les plannings de décroissance][s314] · exercices : 19.1, 19.15, 19.19, 19.24, 19.27, 19.E3, 19.Q7
- **plannings de bruit linéaire et cosinus** — enseignée : [ch. B5, Le planning cosinus de Nichol et Dhariwal][s734] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 2)
- **plannings de learning rate** — enseignée : [ch. 19, 19.3.3 · Les plannings de décroissance][s314] · exercices : 19.1, 19.15, 19.19, 19.24, 19.27, 19.E3, 19.Q7, 23.7 · réutilisée : ch. 23
- **plateau** — enseignée : [ch. 5, Tangente et dérivée][s181] · exercices : 5.1, 5.3–4, 5.6, 5.Q4–Q6, 5.Q9, 5.R2–R3 (+ 11) · réutilisée : ch. 19
- *Platt scaling* → voir [calibration de Platt, régression isotone](#c)
- *pli* → voir [fold](#f)
- *plurality voting* → voir [vote à la pluralité, vote pondéré](#v)
- **plus proche voisin (1-NN)** (*nearest neighbour*) — enseignée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s214] · exercices : 7.5–6, 7.19–21, 7.E4, 7.Q11
- **PMI, PPMI** (*pointwise mutual information, positive PMI*) — enseignée : [ch. B2, 202.8 · Embeddings distributionnels : co-occurrences, PPMI, SVD,…][s479] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- *pocket algorithm* → voir [algorithme pocket](#a)
- **poids** (*weights*) — enseignée : [ch. 0B, 101.1.4 · Moyenne, moyenne pondérée, somme pondérée, moyenne mobile][s39] · exercices : 0B.4, 0B.35, 0B.Q3, 9.Q8, 10.5, 10.15, 10.Q9, 10.R1, 13.25, 16.Q8 (+ 27) · approfondie : ch. 10, 16, B5 · réutilisée : ch. 10, 21, B1, B6
- **poids implicites** (*implicit weights*) — enseignée : [ch. 10, 10.3.3 · Les neurones modernes][s40] · exercices : 10.1–2, 10.5, 10.12, 10.14–15, 10.20, 10.Q8–Q9, 10.R1 (+ 3)
- **poids liés** (*weight tying*) — enseignée : [ch. B2, 202.7 · Embeddings : de one-hot à `nn.Embedding`, entraînement et…][s74] · exercices : B2.5, B2.9, B2.25, B2.E3, B2.Q9, B2.R1
- *poids ouverts* → voir [open weights](#o)
- **point aberrant** (*outlier*) — enseignée : [ch. 9, Le point isolé][s228] · exercices : 9.8–10, 9.12, 9.15, 9.18, 9.20, 9.Q4–Q5, 9.R1 (+ 2)
- **point anguleux** (*corner, kink*) — enseignée : [ch. 5, 5.2 · Introduction : fonctions, courbes et surfaces][s179] · exercices : 5.14, 5.Q2–Q3, 5.Q9 · réutilisée : ch. 17
- **point cœur, point de bord, bruit** (*core point, border point, noise*) — enseignée : [ch. 7, Quand k-means échoue, et le clustering par densité][s215] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 29) · approfondie : ch. B5 · réutilisée : ch. 9
- **point critique, point stationnaire** (*critical point, stationary point*) — enseignée : [ch. 5, Les points où le gradient s'annule][s185] · exercices : 5.2–3, 5.5, 5.7–10, 5.Q7–Q8, 5.Q10 (+ 15)
- **point d'arrêt, débogueur** (*breakpoint, debugger*) — enseignée : [ch. 23, 23.2.2 · Programmer et déboguer en Python][s369] · exercices : 23.Q3
- **point d'entrée** (*entry point*) — enseignée : [ch. B7, 207.2 · Scripts et ligne de commande : argparse, point d'entrée,…][s606] · exercices : B7.10, B7.25, B7.E1, B7.Q3
- **point d'inflexion à tangente horizontale** (*stationary inflection point*) — enseignée : [ch. 5, Tangente et dérivée][s181] · exercices : 5.1, 5.3–4, 5.6, 5.Q4–Q6, 5.Q9, 5.R2–R3 (+ 8)
- point de bord → voir [point cœur, point de bord, bruit](#p)
- **point de code** (*code point*) — enseignée : [ch. B2, Points de code et octets][s463] · exercices : B2.1, B2.18, B2.Q2–Q3
- **point de fonctionnement** (*operating point*) — enseignée : [ch. 3][s170]
- **point influent** (*influential point*) — enseignée : [ch. 2][s135]
- **point selle, col** (*saddle point*) — enseignée : [ch. 5, Les points où le gradient s'annule][s185] · exercices : 5.2–3, 5.5, 5.7–10, 5.Q7–Q8, 5.Q10 (+ 16) · réutilisée : ch. 19
- point stationnaire → voir [point critique, point stationnaire](#p)
- **points critiques** — enseignée : [ch. 5, Les points où le gradient s'annule][s185] · exercices : 5.2–3, 5.5, 5.7–10, 5.Q7–Q8, 5.Q10 (+ 15)
- *pointwise convolution* → voir [convolution 1 × 1](#c)
- *pointwise mutual information* → voir [PMI, PPMI](#p)
- *policy* → voir [politique](#p)
- *policy gradient* → voir [gradient de politique](#g)
- *policy gradient theorem* → voir [théorème du gradient de politique](#t)
- **politique** (*policy*) — enseignée : [ch. 26, 26.6.2 · La politique du Q-learning][s260] · exercices : 7.15, 7.Q4, 26.5, 26.19, 26.21, 26.E3, 26.Q3, 26.Q9, 26.R3, B8.Q5 (+ 5) · approfondie : ch. 7, B8
- **politique de seuil** (*threshold policy*) — enseignée : [ch. 7, 7.2.1 · Classification binaire en 2D][s160] · exercices : 7.7, 7.10–11, 7.14–15, 7.Q3–Q4, 7.R2
- **politique gloutonne** (*greedy policy*) — enseignée : [ch. 26, 26.7 · SARSA][s410] · exercices : 26.4, 26.20, 26.28, 26.Q11
- **politique paramétrée** (*parameterized policy*) — enseignée : [ch. B8, 208.3 · Gradient de politique : politique paramétrée, REINFORCE,…][s108] · exercices : B8.3, B8.7, B8.16–17, B8.E1, B8.Q5–Q6, B8.Q9, B8.R3
- **politique softmax, de Boltzmann** (*softmax policy, Boltzmann*) — enseignée : [ch. 26, 26.6.2 · La politique du Q-learning][s260] · exercices : 26.5, 26.19, 26.E3, 26.Q9, 26.R3, B8.3
- *Polyak* → voir [mise à jour douce (de Polyak)](#m)
- *polynomial features* → voir [features polynomiales](#f)
- **PolynomialFeatures** — enseignée : [ch. 15, 15.9.5 · Les pipelines][s16] · exercices : 15.1, 15.4, 15.6, 15.22–23, 15.28–29, 15.E3, 15.Q11
- **pondération des classes** (*class weighting*) — enseignée : [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s166]
- pondérée → voir [moyenne macro, micro, pondérée](#m)
- **pooling** (*agrégation, sous-échantillonnage*) — enseignée : [ch. 20, 20.4.6 · Le pooling (aperçu)][s22] · exercices : 20.14, 20.20, 20.Q7, 21.5, 21.26, 24.2–3, 24.Q9, B2.28 · approfondie : ch. 21 · réutilisée : ch. 24, 25
- **pooling moyen global** (*global average pooling*) — enseignée : [ch. 21, 21.4.2 · Le stride pour construire une hiérarchie][s347] · exercices : 21.3, 21.5, 21.26, 21.Q6
- *population* → voir [population, échantillon (en statistique)](#p)
- *population stability index* → voir [PSI (indice de stabilité de population)](#p)
- **population, échantillon (en statistique)** (*population, sample*) — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s142] · exercices : 0B.E5, 1.Q3, 2.3–4, 2.15, 2.17–18, 2.29, 2.31, 2.Q5 (+ 13)
- **porte** (*gate*) — enseignée : [ch. 22, 22.7.1 · Les portes][s361] · exercices : 10.3, 10.18, 22.4, 22.16–17, 22.Q7
- **portée** (*scope*) — enseignée : [ch. 0A, 100.5.1 · Définir une fonction : paramètres, valeurs par défaut,…][s59] · exercices : 0A.6, 0A.23, 0A.37, 0A.Q6
- *position-wise feed-forward network* → voir [bloc feed-forward (FFN)](#b)
- *positional encoding* → voir [encodage de position](#e)
- **positions apprises** (*learned positional embeddings*) — enseignée : [ch. B3, 203.5 · Positions : encodage sinusoïdal, positions apprises, RoPE][s504] · exercices : B3.4, B3.7, B3.11, B3.15, B3.25, B3.E5, B3.Q7
- *positive class* → voir [classe positive, négative](#c)
- *positive PMI* → voir [PMI, PPMI](#p)
- *post-hoc explanation* → voir [explication post hoc](#e)
- *post-LN* → voir [pré-LN, post-LN](#p)
- post-LN → voir [pré-LN, post-LN](#p)
- **posterior** (*a posteriori, loi a posteriori*) — enseignée : [ch. 4, 4.6.1 · La boucle posterior → prior][s36] · exercices : 4.5, 4.7, 4.16, 4.18, 4.22, 4.Q6, 4.Q9, 9.26–27, 13.7 (+ 2) · approfondie : ch. B5 · réutilisée : ch. 11, 13
- **posterior (d'un pas)** (*forward process posterior*) — enseignée : [ch. B5, Le posterior : un pas en arrière quand on connaît $\mathbf{x}_0$][s552] · exercices : 4.Q6, B5.7, B5.11, B5.16, B5.19, B5.23, B5.E1–E2, B5.Q4, B5.R3 (+ 8) · approfondie : ch. 4 · réutilisée : ch. 11, 13
- *posterior collapse* → voir [effondrement du posterior](#e)
- **posterior gaussien** — enseignée : [ch. B5, Le posterior : un pas en arrière quand on connaît $\mathbf{x}_0$][s552] · exercices : B5.7, B5.11, B5.16, B5.19, B5.23, B5.E1–E2, B5.Q4, B5.R3
- PPMI → voir [PMI, PPMI](#p)
- **PPMI + SVD** — enseignée : [ch. B2, Les plus proches voisins][s725] · exercices : B2.2, B2.11, B2.23–24, B2.26–27, B2.E4, B2.Q11–Q12
- PPO → voir [objectif clippé, PPO](#o)
- **PPO from scratch** — enseignée : [ch. B8, 208.5 · PPO : ratio de probabilités, objectif clippé, entropie,…][s645] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- *PR baseline* → voir [ligne de base d'une courbe precision-recall](#l)
- *pre-activation* → voir [somme pondérée, pré-activation](#s)
- pré-activation → voir [somme pondérée, pré-activation](#s)
- *pre-commit* → voir [uv, pip-tools, ruff, pre-commit](#u)
- pre-commit → voir [uv, pip-tools, ruff, pre-commit](#u)
- **pré-élagage, élagage** (*pre-pruning, pruning*) — enseignée : [ch. 13, 13.5.3 · Limiter l'overfitting][s276] · exercices : 13.19, 13.21, 13.23–25, 13.E3, 13.Q8, 13.Q10
- **pré-entraînement** (*pre-training*) — enseignée : [ch. 1][s111] · exercices : B1.20
- *pre-LN* → voir [pré-LN, post-LN](#p)
- **pré-LN, post-LN** (*pre-LN, post-LN*) — enseignée : [ch. B3, Pré-LN ou post-LN][s505] · exercices : B3.5, B3.11, B3.16, B3.19, B3.25, B3.Q8, B3.R3
- *pre-pruning* → voir [pré-élagage, élagage](#p)
- *pre-tokenizer* → voir [pré-tokenizer](#p)
- **pré-tokenizer** (*pre-tokenizer*) — enseignée : [ch. B2, La pré-tokenisation][s466] · exercices : B2.4, B2.16–18, B2.30, B2.E3, B2.Q5–Q6
- *pre-training* → voir [pré-entraînement](#p)
- **precision** (*précision, valeur prédictive positive, VPP*) — enseignée : [ch. 3, 3.7.6 · Precision][s31] · exercices : 3.2, 3.9, 3.16, 3.20–21, 3.23, 3.26, 3.E2, 3.Q10, 4.R1 (+ 4) · réutilisée : ch. 4, 7, 15, B6
- *precision and recall for generative models* → voir [precision et recall génératifs](#p)
- **precision et recall génératifs** (*precision and recall for generative models*) — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s426]
- **précision locale, absence, cohérence** (*local accuracy, missingness, consistency*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s582] · exercices : B6.2, B6.7, B6.11, B6.15–17, B6.E1–E2, B6.Q5–Q6 (+ 1)
- **précision mixte** (*mixed precision*) — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s391] · exercices : 24.7, 24.29
- **précision mixte et torch.compile** — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s391] · exercices : 24.29
- *précision moyenne* → voir [average precision (AP)](#a)
- *precision-recall curve* → voir [courbe precision-recall](#c)
- **predict_proba** — enseignée : [ch. 15, 15.4.4 · decision_function et predict_proba][s692] · exercices : 15.2, 15.10, 15.25–26, 15.E4, 15.Q5, CP3.10
- *predicted value* → voir [valeur prédite](#v)
- **prédicteur** (*predictor*) — enseignée : [ch. 15, 15.4 · Les estimateurs][s285] · exercices : 15.8–9, 15.29, 15.Q3, PF.5
- *prediction* → voir [prédiction (principe inductif)](#p)
- *ε-prediction* → voir [prédiction du bruit](#p)
- **prédiction (principe inductif)** (*prediction*) — enseignée : [ch. 11, 11.5 · L'induction][s256] · exercices : 3.Q6, 7.Q1, 9.4, 11.6, 11.17, 11.Q9, B1.23, B5.1, B6.20, B6.E1 (+ 4) · approfondie : ch. B5, B7
- *prediction bias* → voir [biais de prédiction](#b)
- **prédiction du bruit** (*ε-prediction, noise prediction*) — enseignée : [ch. B5, Trois façons de fabriquer une image][s553] · exercices : B5.9, B5.E1, B5.Q1
- *predictive parity* → voir [parité prédictive](#p)
- *predictor* → voir [prédicteur](#p)
- **prefill** — enseignée : [ch. B3, Le KV cache][s512] · exercices : B3.8, B3.23, B3.E4, B3.Q11
- *prefix code* → voir [code préfixe](#c)
- *prefix-free code* → voir [code préfixe](#c)
- PReLU → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- **préparation des données** (*data preparation, data cleaning*) — enseignée : [ch. 12, 12.1 · Pourquoi ce chapitre ?][s261] · exercices : 12.Q1 · réutilisée : ch. 29
- *pretrained model* → voir [modèle pré-entraîné](#m)
- *prevalence* → voir [prévalence, taux de base](#p)
- **prévalence, taux de base** (*prevalence, base rate*) — enseignée : [ch. 3, 3.8 · Appliquer la matrice de confusion : le piège de la prévalence][s162] ; [ch. B6, L'impossibilité][s163] · exercices : 3.7, 3.10, 3.21, 3.26–27, 3.E1, 3.Q12, B6.5, B6.Q11, B6.R3 (+ 6) · réutilisée : ch. 4
- **prévision de série temporelle avec baseline** — enseignée : [ch. 22, Au-delà du livre : prévoir une série temporelle][s366]
- *principal component* → voir [composante principale](#c)
- *principal component analysis* → voir [PCA](#p)
- **prior** (*a priori, loi a priori*) — enseignée : [ch. 4, 4.6.1 · La boucle posterior → prior][s36] · exercices : 4.5, 4.7, 4.10, 4.16, 4.18, 4.21, 4.E3, 4.Q6, 4.Q9, 5.R1 (+ 1) · réutilisée : ch. 6, 13
- **prior conjugué** (*conjugate prior*) — enseignée : [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s178]
- **prior de Laplace** (*Laplace prior*) — enseignée : [ch. 9][s235]
- *PRNG* → voir [pseudo-aléatoire, générateur](#p)
- *probabilistic programming* → voir [programmation probabiliste](#p)
- **probabilité conditionnelle** (*conditional probability*) — enseignée : [ch. 3, 3.4 · Probabilité conditionnelle][s154] · exercices : 3.1, 3.3, 3.13–14, 3.Q3, 3.R1 · réutilisée : ch. 6, 11, 13
- **probabilité jointe** (*joint probability*) — enseignée : [ch. 3, 3.5 · Probabilité jointe][s155] · exercices : 3.3–4, 3.14, 3.Q4, 3.R2
- **probabilité marginale** — enseignée : [ch. 3, 3.6 · Probabilité marginale et tables de contingence][s158] · exercices : 3.3–4, 3.14, 3.Q5, 3.R3
- **probabilité simple, marginale** (*simple probability, marginal probability*) — enseignée : [ch. 3, 3.3 · Probabilité simple][s156] · exercices : 3.1, 3.3, 3.12, 3.Q2, 3.Q5 · réutilisée : ch. 4
- **probabilités : événement, complémentaire, union, indépendance** — enseignée : [ch. 0B, 101.1.6 · Ensembles : union, intersection, complémentaire, cardinal][s81] · exercices : 0B.5, 0B.11, 0B.26, 0B.52, 0B.Q5, 0B.Q12, 2.E4, 3.R2 · réutilisée : ch. 2, 3, 6, 13, 14
- *probability density function* → voir [densité (pdf)](#d)
- *probability distribution* → voir [distribution de probabilité, loi](#d)
- *probability mass function* → voir [fonction de masse (pmf)](#f)
- *probability ratio* → voir [ratio de probabilités](#r)
- *problem of induction* → voir [problème de l'induction](#p)
- **problème de l'arrêt** (*halting problem*) — enseignée : [ch. 11, 11.2.1 · La représentation][s247] · exercices : 11.1, 11.12, 11.Q2–Q3
- **problème de l'induction** (*problem of induction*) — enseignée : [ch. 11, 11.6 · Raisonnement combiné][s258] · exercices : 11.9, 11.Q11
- *processeur central* → voir [CPU](#c)
- *processus de chargement* → voir [worker (d'un `DataLoader`)](#w)
- **processus de décision markovien (MDP)** (*Markov decision process*) — enseignée : [ch. 26, 26.3.5 · La vue d'ensemble][s404] · exercices : 26.10, 26.26, 26.Q5
- **processus direct** (*forward process*) — enseignée : [ch. B5, 205.2 · Le processus direct : planning de bruit, ᾱ_t et forme…][s547] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 2)
- **processus direct et forme fermée** — enseignée : [ch. B5, 205.2 · Le processus direct : planning de bruit, ᾱ_t et forme…][s547] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 2)
- **processus inverse** (*reverse process*) — enseignée : [ch. B5, 205.3 · Le processus inverse et la loss simplifiée : prédire le…][s548] · exercices : B5.7, B5.11, B5.16, B5.19, B5.23, B5.E1–E2, B5.Q4, B5.R3
- *product rule* → voir [règle du produit](#r)
- **produit de Hadamard** (*element-wise product, Hadamard product*) — enseignée : [ch. 0B, 101.3.4 · Produit élément par élément (Hadamard)][s91] · exercices : 0B.18, 0B.38, 0B.40, 0B.45, 0B.Q8
- **produit extérieur** (*outer product*) — enseignée : [ch. 0B, 101.4.3 · Produit matriciel et vérification des formes][s93] · exercices : 0B.20, 0B.30, 0B.43–45, 0B.54, 0B.Q9
- **produit matriciel** (*matrix multiplication*) — enseignée : [ch. 0B, 101.4.3 · Produit matriciel et vérification des formes][s93] · exercices : 0B.20, 0B.30, 0B.43–45, 0B.54, 0B.Q9, 16.R3, 17.8 · réutilisée : ch. 9, 16, 28, B3
- **produit scalaire** (*dot product, inner product*) — enseignée : [ch. 0B, 101.3.3 · Produit scalaire, angle, similarité cosinus][s89] · exercices : 0B.18–19, 0B.39–41, 0B.E2, 0B.Q8, CP1.2, 7.R3, 21.R3 (+ 1) · approfondie : ch. B3 · réutilisée : ch. 7, 10, 12, 13, 21, B2
- **profondeur (d'un arbre)** (*depth*) — enseignée : [ch. 13, 13.5 · Les arbres de décision][s130] · exercices : 0A.44, 13.11, 13.19, 13.24, 13.26, 13.Q7, 13.Q10, 16.R2, 21.Q1, 24.26 (+ 2)
- **programmation dynamique** (*dynamic programming*) — enseignée : [ch. B2][s480]
- **programmation probabiliste** (*probabilistic programming*) — enseignée : [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s178]
- *progressive distillation* → voir [distillation progressive](#d)
- *progressive growing* → voir [croissance progressive](#c)
- *projected gradient descent* → voir [FGSM, PGD](#f)
- **projection de sortie** (*output projection*) — enseignée : [ch. B3, Plusieurs questions à la fois][s499] · exercices : B3.3, B3.11, B3.14, B3.18, B3.24, B3.E1, B3.Q6
- **prompt** (*invite, instruction*) — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s18] · exercices : B1.24, B4.4, B4.13, B4.16, B4.21, B4.24, B4.29, B4.Q12 · approfondie : ch. B4
- *prompt ensembling* → voir [ensemble de prompts](#e)
- *prompt injection* → voir [injection de prompt](#i)
- *prompt template* → voir [gabarit (de prompt)](#g)
- **prompting et sortie structurée** — enseignée : [ch. B4, 204.4 · Prompting : zéro-shot, few-shot, instructions, sortie…][s526] · exercices : B4.15–16, B4.E1, B4.Q6
- *property-based test* → voir [test de propriété](#t)
- *property-based testing* → voir [test de propriétés](#t)
- *prosecutor's fallacy* → voir [erreur du procureur](#e)
- *protected attribute* → voir [attribut sensible](#a)
- **prototype de classe** (*class prototype*) — enseignée : [ch. B1, Le few-shot et les prototypes de classe][s454] · exercices : B1.20–21, B1.25, B1.28, B1.E1–E2, B1.Q8
- *proximal policy optimization* → voir [objectif clippé, PPO](#o)
- *proxy* → voir [variable de substitution](#v)
- *proxy variable* → voir [variable de substitution](#v)
- *pruning* → voir [pré-élagage, élagage](#p)
- **pseudo-aléatoire, générateur** (*pseudo-random, pseudo-random number generator, PRNG*) — enseignée : [ch. 2, 2.2.1 · Nombres aléatoires en pratique][s137] · exercices : 0A.50, 1.Q9, 2.14, 2.E5, 2.Q3, 22.26, 23.23, CP5.8, 25.35, 27.Q3 (+ 6) · approfondie : ch. 0A, 27 · réutilisée : ch. 24, 26
- pseudo-inverse → voir [solution de norme minimale, pseudo-inverse](#s)
- *pseudo-random* → voir [pseudo-aléatoire, générateur](#p)
- *pseudo-random number generator* → voir [pseudo-aléatoire, générateur](#p)
- *pseudo-residuals* → voir [pseudo-résidus](#p)
- **pseudo-résidus** (*pseudo-residuals*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s226]
- **pseudonymisation** — enseignée : [ch. B7][s614]
- **PSI (indice de stabilité de population)** (*population stability index*) — enseignée : [ch. B7, L'indice de stabilité de population (PSI)][s628] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- **PSNR** (*peak signal-to-noise ratio*) — enseignée : [ch. 25, 25.8 · Débruiter][s398] · exercices : 25.3–4, 25.25–27, 25.39, 25.Q10
- *public domain* → voir [domaine public](#d)
- **puce neuromorphique** (*neuromorphic chip*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s241] · exercices : 10.8–9, 10.E4, 10.Q2–Q3
- **puissance de représentation** (*representational power*) — enseignée : [ch. 11, 11.2.1 · La représentation][s247] · exercices : 11.1, 11.12, 11.Q2–Q3
- **puissances, racines, notation scientifique, ordres de grandeur** — enseignée : [ch. 0B, 101.1.1 · Puissances, racines, notation scientifique, ordres de…][s78] · exercices : 0B.1, 0B.6, 0B.30, 0B.33, 0B.54, 0B.Q1, B5.22 · réutilisée : ch. 11
- **pull request** (*demande de fusion*) — enseignée : [ch. 0A, 100.11.2 · git : branches (aperçu)][s76] · exercices : 0A.12, 0A.E5, 0A.Q12
- *punishment* → voir [renforcement, punition](#r)
- punition → voir [renforcement, punition](#r)
- **pureté (d'un clustering)** (*purity*) — enseignée : [ch. 7, Combien de clusters ?][s213] · exercices : 7.3, 7.7, 7.9, 7.12–14, 7.17, 7.Q8, 7.R1, 7.R3 (+ 9)
- *purity* → voir [pureté (d'un clustering)](#p)
- *pydantic* → voir [FastAPI, uvicorn, pydantic](#f)
- pydantic → voir [FastAPI, uvicorn, pydantic](#f)
- **pytest : assert, approx, raises, parametrize ; doctest** — enseignée : [ch. 0A, 100.11.4 · doctest : des exemples exécutables dans les docstrings][s678] · exercices : 0A.61–62, 0A.Q12, 10.20 · réutilisée : ch. 5, 10, 17

## Q

- **Q-learning** — enseignée : [ch. 26, 26.6 · Le Q-learning][s408] · exercices : 26.3, 26.8, 26.20, 26.22, 26.25, 26.28–29, 26.36, 26.Q8, 26.Q11 (+ 4) · réutilisée : ch. B8
- **Q-learning tabulaire** — enseignée : [ch. 26, 26.5 · Le L-learning][s407] · exercices : 26.18, 26.32, 26.36, 26.Q7
- *Q-table* → voir [table Q, valeur Q](#t)
- *Q-value* → voir [table Q, valeur Q](#t)
- **QK-Norm** (*query-key normalization*) — enseignée : [ch. B3, Compter les paramètres d'un bloc][s516] · exercices : B3.5, B3.11, B3.16, B3.19, B3.25, B3.Q8, B3.R3
- **QLoRA** (*quantized LoRA*) — enseignée : [ch. B4, 204.8 · Fine-tuning : complet ou PEFT ; LoRA (rang, alpha, modules…][s539] · exercices : B4.3, B4.6, B4.11, B4.25–27, B4.E2, B4.Q10–Q11, B4.R2 (+ 1)
- **quantification** (*quantization*) — enseignée : [ch. B4, La mémoire des poids][s523] · exercices : B4.1, B4.7, B4.12, B4.29, B4.Q3, B7.20
- **quantification dynamique, statique** (*dynamic, static quantization*) — enseignée : [ch. B7, Accélérer sur CPU][s625] · exercices : B2.29, B2.Q11, B7.1, B7.19–21, B7.25, B7.E5, B7.Q9–Q10
- *quantile* → voir [percentile, quantile, quartile](#p)
- quantile → voir [percentile, quantile, quartile](#p)
- *quantile loss* → voir [loss quantile](#l)
- *quantization* → voir [quantification](#q)
- *quantized LoRA* → voir [QLoRA](#q)
- **quartet d'Anscombe, Datasaurus** (*Anscombe's quartet, Datasaurus Dozen*) — enseignée : [ch. 2, 2.9 · Le quartet d'Anscombe][s149] · exercices : 2.12, 2.30, 2.32, 2.E4, 2.Q12
- *quartile* → voir [percentile, quantile, quartile](#p)
- quartile → voir [percentile, quantile, quartile](#p)
- **quasi-doublon** (*near-duplicate*) — enseignée : [ch. 29, Au-delà du livre : auditer un dataset][s439]
- *query* → voir [requête, clé, valeur](#r)
- *query-key normalization* → voir [QK-Norm](#q)

## R

- **raccourci appris** (*shortcut learning*) — enseignée : [ch. 8, 8.2.1 · Pourquoi le score d'entraînement trompe][s219] · exercices : 8.9, 8.12, 8.E4, 8.Q3–Q4, 8.R3
- **raccourcis appris** — enseignée : [ch. 8, 8.2.1 · Pourquoi le score d'entraînement trompe][s219] · exercices : 8.9, 8.12, 8.E4, 8.Q3–Q4, 8.R3
- **racine, feuille, nœud interne** (*root, leaf, internal node*) — enseignée : [ch. 13, 13.5 · Les arbres de décision][s130] · exercices : 0A.44, 0B.1, 0B.6, 0B.Q1, 13.2, 13.11, 13.24, 13.26, 13.Q7, B5.22 · approfondie : ch. 0B · réutilisée : ch. 2, 27
- **racinisation** (*stemming*) — enseignée : [ch. B4, De TF-IDF à BM25][s534] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **RAG** (*retrieval-augmented generation, génération augmentée par recherche*) — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s18] · exercices : B4.7–8, B4.21–23, B4.30, B4.E2, B4.E4, B4.Q10 · approfondie : ch. B4
- **RAG avec citations** — enseignée : [ch. B4, RAG ou fine-tuning ?][s732] · exercices : B4.2, B4.7–8, B4.21–23, B4.30, B4.E1–E2, B4.Q10 (+ 1)
- raison → voir [suite géométrique, raison](#s)
- *raisonnement pas à pas* → voir [chain-of-thought](#c)
- *random features* → voir [features aléatoires](#f)
- *random forest* → voir [forêt aléatoire](#f)
- *random search* → voir [recherche au hasard, recherche aléatoire](#r)
- *random variable* → voir [variable aléatoire](#v)
- **randomisation de domaine** (*domain randomization*) — enseignée : [ch. B8, De la simulation au réel][s654] · exercices : B8.8, B8.23–24, B8.Q12
- **randomisation en cascade, indépendante** (*cascading, independent randomization*) — enseignée : [ch. B6, Les tests de santé][s592] · exercices : B6.16, B6.22, B6.E2, B6.E4, B6.Q6, B6.Q9 · réutilisée : ch. 8, 16, B3, B4, B5
- **RandomizedSearchCV** — enseignée : [ch. 15, 15.9.4 · La recherche aléatoire][s288] · exercices : 15.5, 15.19–20, 15.E2, 15.Q10 · réutilisée : ch. 24
- rang → voir [fusion (BPE), rang](#f)
- **rang (LoRA)** (*rank*) — enseignée : [ch. B4, 204.8 · Fine-tuning : complet ou PEFT ; LoRA (rang, alpha, modules…][s539] · exercices : B4.3, B4.6, B4.11, B4.25–27, B4.E2, B4.Q10–Q11, B4.R2 (+ 4)
- *rang réciproque moyen* → voir [MRR](#m)
- *rank* → voir [fusion (BPE), rang](#f), [rang (LoRA)](#r)
- *rank-stabilized LoRA* → voir [rsLoRA](#r)
- *rappel aux k premiers* → voir [recall@k](#r)
- **rapport de vraisemblance** (*likelihood ratio*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s173] · exercices : 4.3, 4.10, 4.14, 4.23, 4.E3, 4.Q6–Q7
- **rapport signal sur bruit (SNR)** (*signal-to-noise ratio*) — enseignée : [ch. B5, Le rapport signal sur bruit][s551] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 2)
- *ratio* → voir [suite géométrique, raison](#s)
- **ratio de probabilités** (*probability ratio*) — enseignée : [ch. B8, 208.5 · PPO : ratio de probabilités, objectif clippé, entropie,…][s645] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- *RBF kernel* → voir [noyau RBF (gaussien)](#n)
- *reality gap* → voir [écart simulation-réalité, sim-to-real](#e)
- *reasoning model* → voir [modèle de raisonnement](#m)
- **recall** (*rappel, sensibilité, taux de vrais positifs, TPR*) — enseignée : [ch. 3, 3.7.7 · Recall][s32] · exercices : 3.2, 3.9, 3.16, 3.20, 3.23, 3.29, 3.E2, 3.Q10, 15.26, CP3.11 (+ 4) · approfondie : ch. B4 · réutilisée : ch. 4, 7, 15
- **recall@k** (*rappel aux k premiers*) — enseignée : [ch. B4, Mesurer une recherche : recall@k et MRR][s537] · exercices : B4.2, B4.5, B4.17–20, B4.22–23, B4.Q9, B4.R3
- **recall@k et MRR** — enseignée : [ch. B4, Mesurer une recherche : recall@k et MRR][s537] · exercices : B4.2, B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- *receptive field* → voir [champ récepteur](#c)
- recherche aléatoire → voir [recherche au hasard, recherche aléatoire](#r)
- **recherche approchée des plus proches voisins** (*approximate nearest neighbor search, ANN*) — enseignée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s214] ; [ch. 13, 13.3 · Les k plus proches voisins][s211] · exercices : 7.5–6, 7.19–21, 7.E4, 7.Q11, 13.1, 13.Q3–Q4 (+ 7)
- **recherche au hasard, recherche aléatoire** (*random search*) — enseignée : [ch. 15, 15.9.4 · La recherche aléatoire][s288] · exercices : 15.5, 15.19–20, 15.E2, 15.Q10, 24.1
- **recherche d'hyperparamètres** — enseignée : [ch. 8, 8.4 · Le jeu de validation][s129] · exercices : 8.1–3, 8.6, 8.10, 8.17–18, 8.Q7–Q8, 8.R1 (+ 8) · approfondie : ch. 24 · réutilisée : ch. 19
- **recherche dense** (*dense retrieval*) — enseignée : [ch. B4, 204.6 · Recherche d'information : découpage, BM25, recherche…][s533] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **recherche dense et hybride** — enseignée : [ch. B4, 204.6 · Recherche d'information : découpage, BM25, recherche…][s533] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **recherche en grille** (*grid search*) — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s287] · exercices : 15.1, 15.12, 15.18, 15.23, 15.E2, 15.Q10, CP3.14, 18.R2
- **recherche en largeur** (*breadth-first search*) — enseignée : [ch. 26][s416]
- **recherche hybride** (*hybrid search*) — enseignée : [ch. B4, La recherche hybride et la fusion RRF][s536] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **recherche multirésolution** (*coarse-to-fine search*) — enseignée : [ch. 24, 24.3.4 · La recherche d'hyperparamètres][s383] · exercices : 24.1, 24.7, 24.9, 24.18, 24.E1, 24.Q5
- **recherche sémantique** (*semantic search*) — enseignée : [ch. B2, 202.10 · Embeddings contextuels et embeddings de phrases ;…][s485] · exercices : B2.28–29, B2.E4–E5, B2.Q11, B2.R1
- **reciprocal rank fusion (RRF)** (*fusion par rangs réciproques*) — enseignée : [ch. B4, La recherche hybride et la fusion RRF][s536] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- récompense → voir [agent, environnement, action, récompense](#a)
- **récompense façonnée** (*reward shaping*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s405] · exercices : 26.1, 26.7, 26.11, 26.14–15, 26.32, 26.E2, 26.Q6
- *reconstruction error* → voir [erreur de reconstruction](#e)
- **recouvrement** (*overlap*) — enseignée : [ch. B4, Découper][s532] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **rectified flow** — enseignée : [ch. B5, Flow matching et rectified flow][s571] · exercices : B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11
- *recurrent neural network* → voir [réseau récurrent (RNN)](#r)
- *recursive feature elimination* → voir [RFE](#r)
- **récursivité** — enseignée : [ch. 0A, 100.5.6 · Fonctions récursives][s63] · exercices : 0A.44
- **redondance (d'une source)** (*redundancy*) — enseignée : [ch. 6, 6.7 · L'entropie][s196] · exercices : 6.3, 6.6, 6.8–10, 6.12, 6.14, 6.19, 6.Q8–Q9 (+ 3)
- **réduction de dimension** (*dimensionality reduction*) — enseignée : [ch. 1, 1.4.3 · Réduction de dimension][s116] ; [ch. 12, 12.7 · La réduction de dimension][s117] · exercices : 1.4, 1.Q8, 12.7, 12.22, 12.27, 12.30, 12.E4, 12.Q8
- *réduction de moitié successive* → voir [successive halving](#s)
- *redundancy* → voir [redondance (d'une source)](#r)
- rééchantillon → voir [bootstrap, rééchantillon](#b)
- **réentraînement final** (*refit*) — enseignée : [ch. 15, 15.9.2 · Chercher des hyperparamètres][s289] · exercices : 15.1, 15.5, 15.18, 15.20, 15.Q10
- *refactor* → voir [refactoriser](#r)
- **refactoriser** (*refactor*) — enseignée : [ch. 4][s176] · exercices : 4.23, 9.29
- référence de style → voir [image de contenu, référence de style, image générée](#i)
- *refit* → voir [réentraînement final](#r)
- **région de confiance, TRPO** (*trust region*) — enseignée : [ch. B8, Réutiliser un rollout sans tout casser][s644] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- **région de décision** (*decision region*) — enseignée : [ch. 7, 7.2.1 · Classification binaire en 2D][s160] · exercices : 7.7, 7.10–11, 7.14–15, 7.Q3–Q4, 7.R2
- **règle 68-95-99,7** (*68-95-99.7 rule, three-sigma rule*) — enseignée : [ch. 2, 2.3.2 · La loi normale][s139] · exercices : 2.3–4, 2.15, 2.17–18, 2.29, 2.31, 2.Q5
- **règle d'apprentissage du perceptron** (*perceptron learning rule*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s208] · exercices : 10.1, 10.3, 10.6, 10.11–12, 10.17, 10.Q5–Q6, 10.R2–R3 (+ 4)
- règle d'arrêt → voir [test séquentiel, règle d'arrêt](#t)
- **règle d'or (de la préparation)** — enseignée : [ch. 12, 12.2 · La règle d'or : apprendre la transformation sur l'entraînement][s123] · exercices : 8.Q5, 12.9, 12.20, 12.31–33, 12.Q1
- **règle d'une erreur type** (*one-standard-error rule*) — enseignée : [mini-projet 2, Le protocole, le cœur du projet][s662]
- **règle de Bayes, théorème de Bayes** (*Bayes' rule, Bayes' theorem*) — enseignée : [ch. 4, 4.4.1 · La règle de Bayes][s157] · exercices : 4.1, 4.3, 4.9, 4.14–15, 4.E1, 4.Q6, CP1.6, 7.R2 · réutilisée : ch. 7, 9, 11, 13
- **règle de la chaîne** (*chain rule*) — enseignée : [ch. 0B, 101.5.3 · Règle de la chaîne][s97] · exercices : 0B.23, 0B.29, 0B.47, 0B.E4, 0B.Q10, 5.R3, 18.6 · approfondie : ch. 18 · réutilisée : ch. 5, 13, 17, 18, 22, 25, 28, B2
- **règle de la chaîne à plusieurs chemins** (*multivariable chain rule*) — enseignée : [ch. 18, 18.8 · Étape 3 : les deltas des autres neurones][s307] · exercices : 18.4, 18.6, 18.15, 18.18, 18.26, 18.Q6
- **règle de la chaîne à plusieurs variables (somme sur les chemins, version simple)** — enseignée : [ch. 0B, 101.6.4 · Règle de la chaîne à plusieurs variables : la somme sur…][s100] · exercices : 0B.29, 0B.51, 0B.E4, 0B.Q11
- **règle de la chaîne multivariée** — enseignée : [ch. 18][s311]
- **règle de succession de Laplace** (*rule of succession*) — enseignée : [ch. 4, 4.4.1 · La règle de Bayes][s157] · exercices : 4.1, 4.9, 4.14–15, 4.E1, 4.Q6
- **règle de trois** (*rule of three*) — enseignée : [mini-projet 1, La grille d'évaluation (sur 20)][s659]
- **règle de Tukey** (*Tukey's fences*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s264] · exercices : 12.11, 12.Q4
- **règle des quatre cinquièmes** (*four-fifths rule*) — enseignée : [ch. B6, Les critères][s597] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- **règle du plus fort reste** (*largest remainder method*) — enseignée : [ch. 8, 8.3 · Le jeu de test][s122] · exercices : 8.1, 8.3–4, 8.7, 8.10–13, 8.Q5–Q6 (+ 5)
- **règle du produit** (*product rule*) — enseignée : [ch. 3, 3.5 · Probabilité jointe][s155] ; [ch. 4, 4.4.1 · La règle de Bayes][s157] · exercices : 3.3–4, 3.14, 3.Q4, 3.R2, 4.1, 4.9, 4.14–15, 4.Q6 (+ 1)
- **règlement européen sur l'IA** (*AI Act (règlement 2024/1689), UE*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s336] · exercices : 20.11, 20.Q12
- régression isotone → voir [calibration de Platt, régression isotone](#c)
- **régression linéaire bayésienne** (*Bayesian linear regression*) — enseignée : [ch. 9, 9.7 · Ajuster une droite avec la règle de Bayes][s233] · exercices : 9.2, 9.7, 9.26–27, 9.Q11
- **régression logistique** (*logistic regression*) — enseignée : [ch. 13, Au-delà du livre : la régression logistique, le classifieur…][s84] · exercices : 13.8, 13.31–32, 13.E5, B4.28, B6.17 · réutilisée : ch. 15, 18, B1, B4
- *regression test* → voir [test de non-régression](#t)
- *regression to the mean* → voir [régression vers la moyenne](#r)
- **régression vers la moyenne** (*regression to the mean*) — enseignée : [ch. 1, 1.3.2 · Régression][s113] · exercices : 1.2, 1.8, 1.12, 1.Q7, 1.R3
- **regret** — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s121] · exercices : 11.2, 11.7–8, 11.10–11, 11.13, 11.19–21, 11.R1 (+ 8)
- *regroupement* → voir [clustering](#c)
- **regroupement dynamique** (*dynamic batching*) — enseignée : [ch. B7][s614]
- *regular expression* → voir [expression régulière](#e)
- **régularisation** (*regularization*) — enseignée : [ch. 9, 9.5 · La régularisation][s229] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 5) · réutilisée : ch. 10, 15, 18, 19, 20
- **régularisation L1 (Lasso)** — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s231] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 2)
- **régularisation L2 (Ridge)** — enseignée : [ch. 9, La pénalité L2 : Ridge][s230] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 8) · réutilisée : ch. 12, 15, 19, 20
- *regularization* → voir [régularisation](#r)
- *regularization path* → voir [chemin de régularisation](#c)
- *regularization strength* → voir [force de régularisation ($\lambda$, `alpha`, `C`)](#f)
- **REINFORCE** — enseignée : [ch. B8, 208.3 · Gradient de politique : politique paramétrée, REINFORCE,…][s108] · exercices : B8.3, B8.7, B8.16–17, B8.E1, B8.Q5–Q6, B8.Q9, B8.R3
- **REINFORCE avec baseline** — enseignée : [ch. B8, 208.3 · Gradient de politique : politique paramétrée, REINFORCE,…][s108] · exercices : B8.3, B8.7, B8.16–17, B8.E1, B8.Q5–Q6, B8.Q9, B8.R3
- *reinforcement* → voir [renforcement, punition](#r)
- *reinforcement learning* → voir [apprentissage par renforcement](#a)
- *reinforcement learning from human feedback* → voir [alignement, RLHF](#a), [RLHF](#r)
- *relative / absolute path* → voir [chemin relatif / absolu](#c)
- *relative contrast* → voir [contraste des distances](#c)
- *relative entropy* → voir [divergence KL, divergence de Kullback-Leibler](#d)
- *relative error* → voir [erreur relative (d'un gradient)](#e)
- *reliability diagram* → voir [diagramme de fiabilité](#d)
- *ReLU* → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- **ReLU bruitée** (*noisy ReLU*) — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s299] · exercices : 17.2, 17.5, 17.12, 17.18, 17.20, 17.23–24, 17.Q6–Q7, 17.R1 (+ 2)
- ReLU décalée → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- **ReLU, leaky ReLU, PReLU, ReLU décalée** (*ReLU, leaky ReLU, parametric ReLU, shifted ReLU*) — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s299] · exercices : 17.2, 17.5, 17.12, 17.18, 17.20, 17.23–24, 17.Q6–Q7, 17.R1 (+ 9) · réutilisée : ch. 16, 21, 23, 25, 27
- *remplissage de zéros* → voir [zero-padding](#z)
- **renforcement, punition** (*reinforcement, punishment*) — enseignée : [ch. 11, 11.6.1 · Sherlock Holmes, « maître de la déduction »][s253] · exercices : 1.E3, 1.Q9, 11.7, 11.14, 11.Q12, CP6.7
- *reparameterization trick* → voir [astuce de reparamétrisation](#a)
- *repeated k-fold* → voir [k-fold répétée](#k)
- *replay buffer* → voir [mémoire de rejeu](#m)
- **repli sur les octets** (*byte fallback*) — enseignée : [ch. B2, SentencePiece : sans pré-tokenisation][s472] · exercices : B2.13, B2.19, B2.30, B2.E2, B2.Q7
- **repondération** (*reweighing*) — enseignée : [ch. B6][s576] · exercices : B6.27
- *repository* → voir [dépôt](#d)
- *representation* → voir [représentation](#r)
- **représentation** (*representation*) — enseignée : [ch. 11, 11.2.1 · La représentation][s247] · exercices : 11.1, 11.12, 11.Q1–Q3, 12.R1, 18.R3, 28.R2
- *representational power* → voir [capacité](#c), [puissance de représentation](#p)
- **reproductibilité** — enseignée : [ch. B7, 207.4 · Reproductibilité : graines, déterminisme, versions figées,…][s613] · exercices : 23.29, B7.2, B7.12–13, B7.17, B7.25, B7.E2, B7.Q5–Q6, B7.R3 (+ 1) · approfondie : ch. 2, 20, 26, B4
- **reproductibilité (graine)** — enseignée : [ch. 2][s135] · exercices : 23.29, 24.9, B7.E2 · approfondie : ch. 20, 26, B4, B7
- **reproductibilité d'un pipeline LLM** — enseignée : [ch. B4, 204.9 · Classer des textes : encodeur fine-tuné, LLM en zéro-shot…][s733] · exercices : B4.28, B4.R2
- **reproductibilité en RL (graines multiples)** — enseignée : [ch. 26, Au-delà du livre : le RL et les modèles de langage][s415]
- **requête, clé, valeur** (*query, key, value*) — enseignée : [ch. B3, 203.2 · Attention par produit scalaire : requêtes, clés, valeurs,…][s494] · exercices : B3.1, B3.6, B3.9, B3.11–13, B3.18, B3.E1, B3.Q2–Q3 (+ 38) · approfondie : ch. 0A, 0B, 12, 26, B6 · réutilisée : ch. 9, 12, 27
- **réseau antagoniste génératif (GAN)** (*generative adversarial network*) — enseignée : [ch. 27, 27.4 · Implémenter un GAN][s417] · exercices : 27.8, 27.12, 27.14, 27.16, 27.25, 27.27, 27.E1, 27.E3, 27.E5, 27.Q6 (+ 4)
- **réseau cible** (*target network*) — enseignée : [ch. 26, 26.9 · Le rejeu d'expérience][s413] ; [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s234] · exercices : 26.12, 26.31, 26.33, 26.E4, 26.Q12, B8.1, B8.Q3–Q4, B8.Q9, B8.R1 (+ 5)
- **réseau de neurones** (*neural network*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s38] · exercices : 1.1, 1.16–17, 1.23, 1.E2, 1.Q5, 10.E4, 14.E3 · réutilisée : ch. 26
- **réseau feed-forward, MLP** (*feed-forward network, multilayer perceptron*) — enseignée : [ch. 16, 16.1 · Pourquoi ce chapitre ?][s293] · exercices : 16.1, 16.8–9, 16.E3, 16.Q1, 16.R2, 17.11, 18.29, 18.R3, 19.9 (+ 10) · réutilisée : ch. 23
- **réseau impulsionnel** (*spiking neural network, SNN*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s241] · exercices : 10.8–9, 10.E4, 10.Q2–Q3
- **réseau récurrent (RNN)** (*recurrent neural network*) — enseignée : [ch. 22, 22.4 · La structure d'une cellule RNN][s357] · exercices : 16.Q4, 22.2–3, 22.9, 22.15, 22.18–19, 22.Q3, 22.Q11–Q12 (+ 5) · approfondie : ch. 20 · réutilisée : ch. B3
- **résidu** (*residual*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s26] · exercices : 9.1, 9.12, 9.14, 9.E2, 9.Q1, 14.4, 14.Q11
- **résidu de Bellman** (*Bellman residual*) — enseignée : [ch. B8][s656]
- *residual* → voir [résidu](#r)
- *residual connection* → voir [connexion résiduelle](#c)
- **retour actualisé** (*discounted return*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s405] · exercices : 26.1, 26.7, 26.11, 26.14–15, 26.32, 26.E2, 26.Q6
- **retour actualisé et facteur γ** — enseignée : [ch. 26, 26.3.7 · Les récompenses][s405] · exercices : 26.1, 26.7, 26.11, 26.14–15, 26.32, 26.E2, 26.Q6
- **retour arrière** (*rollback*) — enseignée : [ch. B7, 207.10 · Surveiller en production : dérive des données et des…][s631] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- retour total → voir [retour, retour total (TFR)](#r)
- **retour, retour total (TFR)** (*return, total future reward*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s405] · exercices : 1.Q9, 12.4, 26.1, 26.7, 26.11, 26.14–15, 26.32, 26.E2, 26.Q6 (+ 6) · réutilisée : ch. B8
- *retracted dataset* → voir [dataset retiré](#d)
- *rétrécissement* → voir [shrinkage](#s)
- *retrieval-augmented generation* → voir [RAG](#r)
- **rétroaction** (*feedback*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s294] · exercices : 16.2, 16.4, 16.9, 16.16–18, 16.Q1–Q4 (+ 2)
- **rétropropagation** (*backpropagation, backprop*) — enseignée : [ch. 18, 18.9 · La rétropropagation en action][s42] · exercices : 0B.E4, 18.4, 18.9–10, 18.18–19, 18.28, 18.Q1, 18.Q7, 18.Q9 (+ 4) · approfondie : ch. 21 · réutilisée : ch. 19, 20, 21, 22, 25, B2, B8
- **rétropropagation d'une convolution** — enseignée : [ch. 21, Au-delà du livre : la rétropropagation d'une convolution][s706]
- **rétropropagation dans le temps (BPTT)** (*backpropagation through time*) — enseignée : [ch. 22, 22.6 · Entraîner un RNN][s309] · exercices : 22.6–7, 22.19, 22.23, 22.E2, 22.Q6, 22.R2
- **rétropropagation tronquée dans le temps** (*truncated BPTT*) — enseignée : [ch. 24, 24.5.9 · Les RNN « stateful »][s389] · exercices : 24.27, 24.Q11
- *return* → voir [retour, retour total (TFR)](#r)
- *reverse process* → voir [processus inverse](#p)
- *reverse-mode automatic differentiation* → voir [différentiation automatique en mode inverse](#d)
- *revision* → voir [révision (d'un dataset)](#r)
- **révision (d'un dataset)** (*revision*) — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s125] · exercices : B4.29, PF.2
- **revue de code** (*code review*) — enseignée : [ch. B8, Déboguer un agent qui n'apprend pas][s658] · exercices : B8.8, B8.15, B8.21–22, B8.27–28, B8.E3, B8.E5, B8.Q10
- *reward* → voir [agent, environnement, action, récompense](#a)
- *reward hacking* → voir [piratage de la récompense](#p)
- *reward model* → voir [modèle de récompense](#m)
- *reward model overoptimization* → voir [sur-optimisation du modèle de récompense](#s)
- *reward shaping* → voir [récompense façonnée](#r)
- *reweighing* → voir [repondération](#r)
- **RFE** (*recursive feature elimination*) — enseignée : [ch. 12, 12.6 · La sélection de features][s270] · exercices : 12.10–11, 12.25, 12.Q8
- **RGPD** (*GDPR (règlement 2016/679), UE*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s336] · exercices : 20.11, 20.Q12 · approfondie : ch. B6
- **RGPD et AI Act appliqués** — enseignée : [ch. B6, 206.11 · Éthique et régulation en pratique : fiches modèles,…][s601] · exercices : B6.8–9, B6.28, B6.Q12
- **Ridge (pénalité L2)** (*ridge regression, L2 penalty, Tikhonov regularization*) — enseignée : [ch. 9, La pénalité L2 : Ridge][s230] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 6) · réutilisée : ch. 12, 15, 19, 20
- *ridge regression* → voir [Ridge (pénalité L2)](#r)
- *RL* → voir [amorçage](#a)
- RL → voir [fonction de valeur (RL)](#f)
- **RL à récompenses vérifiables (RLVR)** (*RL with verifiable rewards*) — enseignée : [ch. B8, GRPO et les récompenses vérifiables][s652] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- **RL hors ligne** (*offline RL*) — enseignée : [ch. B8, Sécurité][s655] · exercices : B8.8, B8.23–24, B8.Q12
- *RL with verifiable rewards* → voir [RL à récompenses vérifiables (RLVR)](#r)
- **RLHF** (*reinforcement learning from human feedback*) — enseignée : [ch. 26, Au-delà du livre : le RL et les modèles de langage][s415] · exercices : 26.E5, B8.10, B8.26, B8.Q11 · approfondie : ch. B8
- RLVR → voir [RL à récompenses vérifiables (RLVR)](#r)
- RMSE → voir [MSE, RMSE, MAE](#m)
- **RMSNorm** (*root mean square normalization*) — enseignée : [ch. 20, 20.4.4 · La batchnorm][s240] · exercices : 20.4, 20.8, 20.12, 20.22, 20.24, 20.28–29, 20.E2, 20.Q6
- **RMSprop** — enseignée : [ch. 19, 19.5.4 · Adadelta et RMSprop][s322] · exercices : 19.4, 19.8, 19.13, 19.17, 19.Q11 · réutilisée : ch. 22
- RNN → voir [réseau récurrent (RNN)](#r)
- **RNN profond, bidirectionnel** (*deep RNN, bidirectional RNN*) — enseignée : [ch. 22, 22.8.2 · Le RNN profond][s365] · exercices : 22.22, 22.25, 22.Q11
- *robust* → voir [robuste](#r)
- *robust scaling* → voir [mise à l'échelle robuste](#m)
- **robuste** (*robust*) — enseignée : [ch. 2, Résumer une liste de nombres : moyenne, médiane, mode][s133] · exercices : 2.1–2, 2.13, 2.16, 2.19, 2.E1, 2.Q1–Q2, 2.R2
- *ROC* → voir [AUC](#a)
- *ROC curve* → voir [courbe ROC](#c)
- **rôle de la non-linéarité** — enseignée : [ch. 17][s696]
- *rollback* → voir [retour arrière](#r)
- *rolling origin* → voir [validation glissante](#v)
- **rollout** — enseignée : [ch. B8, Réutiliser un rollout sans tout casser][s644] · exercices : B8.5, B8.9, B8.11, B8.19–22, B8.Q8–Q9, B8.R2 (+ 2)
- *root* → voir [racine, feuille, nœud interne](#r)
- *root mean square normalization* → voir [RMSNorm](#r)
- RoPE → voir [encodage de position rotatif (RoPE)](#e)
- *Rosenbrock function* → voir [fonction de Rosenbrock](#f)
- *rotary position embedding* → voir [encodage de position rotatif (RoPE)](#e)
- *round-off error* → voir [erreur d'arrondi](#e)
- route → voir [API HTTP, route](#a)
- *row-major order* → voir [ordre C, ligne par ligne](#o)
- RRF → voir [reciprocal rank fusion (RRF)](#r)
- **rsLoRA** (*rank-stabilized LoRA*) — enseignée : [ch. B4, LoRA][s540] · exercices : B4.3, B4.6, B4.11, B4.25–27, B4.E2, B4.Q10–Q11, B4.R2 (+ 1)
- *ruff* → voir [uv, pip-tools, ruff, pre-commit](#u)
- ruff → voir [uv, pip-tools, ruff, pre-commit](#u)
- *rule of succession* → voir [règle de succession de Laplace](#r)
- *rule of three* → voir [règle de trois](#r)
- **run** (*exécution*) — enseignée : [ch. 24, 24.2.2 · Changer un seul hyperparamètre][s385] · exercices : 24.11, 24.15, 24.29, 24.Q1–Q2, 24.R1, B7.14
- *run ID* → voir [identifiant de run](#i)
- *running statistics* → voir [statistiques glissantes](#s)
- **R²** — enseignée : [ch. 9][s235] · exercices : 9.1 · réutilisée : ch. 8

## S

- *sa racine* → voir [MSE, RMSE, MAE](#m)
- **sac de mots** (*bag of words*) — enseignée : [ch. 0B][s34] · exercices : 13.30
- *saddle point* → voir [point selle, col](#p)
- **safetensors** — enseignée : [ch. 23, 23.9.2 · Les poids seuls][s381] · exercices : 23.23, 23.Q11, B1.27, B7.17
- *saliency map* → voir [carte de saillance](#c)
- **SAMME** (*stagewise additive modeling using a multi-class exponential loss*) — enseignée : [ch. 14, 14.7 · Le boosting][s275] · exercices : 14.2, 14.4, 14.6–8, 14.10, 14.Q8–Q11 (+ 12)
- *sample* → voir [échantillon](#e), [tirage](#t), [population, échantillon (en statistique)](#p)
- *sample density* → voir [densité d'échantillons](#d)
- *sample efficiency* → voir [efficacité en échantillons](#e)
- *sample-and-hold* → voir [échantillonneur-bloqueur](#e)
- *sampler* → voir [échantillonneur](#e)
- *samplewise* → voir [par échantillon, par feature, par élément](#p)
- **samplewise/featurewise/elementwise** — enseignée : [ch. 12, 12.9 · Par échantillon, par feature, par élément][s269] · exercices : 12.5, 12.Q10
- *sampling bias* → voir [biais d'échantillonnage](#b)
- *sampling with replacement* → voir [tirage avec remise, sans remise](#t)
- *sanity check* → voir [test de santé](#t)
- sans remise → voir [tirage avec remise, sans remise](#t)
- **SARSA** — enseignée : [ch. 26, 26.7 · SARSA][s410] · exercices : 26.4, 26.20, 26.28–29, 26.E1, 26.Q11, CP6.3
- **SARSA espéré** (*expected SARSA*) — enseignée : [ch. 26, 26.7 · SARSA][s410] · exercices : 26.4, 26.20, 26.28, 26.Q11
- **saturation** — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s85] · exercices : 17.2, 17.4, 17.8, 17.10, 17.13–14, 17.17–18, 17.Q8–Q9 (+ 6) · approfondie : ch. 18
- **saturation et neurones morts** — enseignée : [ch. 17, 17.2 · Ce que fait une fonction d'activation][s301] ; [ch. 18, 18.12.4 · Garder les neurones en vie][s698] · exercices : 17.1, 17.3, 17.9, 17.11, 17.20, 17.Q1, 18.24, 18.E3, 18.Q10
- **sauvegarde et reproductibilité** — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s2]
- *scaled dot-product attention* → voir [attention par produit scalaire mis à l'échelle](#a)
- **scaler** (*objet de mise à l'échelle*) — enseignée : [ch. 12, 12.5.3 · Retenir la transformation][s267] · exercices : 12.4, 12.13, 12.16, 12.19–20, 12.31, 12.33, 12.Q6, 15.R2, 22.R3 (+ 2) · approfondie : ch. 24
- *scaling laws* → voir [lois d'échelle](#l)
- *scatter plot* → voir [nuage de points](#n)
- **schedulers de learning rate** — enseignée : [ch. 23, 23.10.2 · Le learning rate][s712] · exercices : 23.7, 23.11, 23.25–26
- *scope* → voir [portée](#p)
- *score* → voir [score, seuil de décision](#s), [score (d'une densité)](#s)
- **score (d'une densité)** (*score*) — enseignée : [ch. B5, La guidance sans classifieur][s566] · exercices : 1.20, 3.18, 8.5, 8.8, 8.Q8, B5.1, B5.26–27, B5.E4, B5.Q9 (+ 9) · approfondie : ch. 3, 8 · réutilisée : ch. 3
- **score de Brier** (*Brier score*) — enseignée : [ch. 3, Au-delà du livre (3) : la calibration des probabilités][s169]
- *score function* → voir [fonction score](#f)
- **score, seuil de décision** (*score, decision threshold*) — enseignée : [ch. 3, 3.7.11 · Le score F1][s164] ; [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s166] · exercices : 1.20, 3.2, 3.6, 3.8, 3.16, 3.18, 3.25, 3.E4, 3.Q12, 8.Q8 (+ 13) · approfondie : ch. 8
- *search space* → voir [espace de recherche](#e)
- *secant* → voir [sécante](#s)
- **sécante** (*secant*) — enseignée : [ch. 5, Tangente et dérivée][s181] · exercices : 5.1, 5.3–4, 5.6, 5.Q4–Q6, 5.Q9, 5.R2–R3 (+ 9)
- *second derivative* → voir [dérivée seconde, courbure](#d)
- **secret** — enseignée : [ch. B7, 207.9 · Conteneurs et déploiement : Dockerfile, variables…][s626] · exercices : B7.23–25, B7.Q4
- *seed* → voir [graine](#g), [amorce](#a)
- **segmentation la plus courte** (*shortest tokenization*) — enseignée : [ch. B2][s480]
- *selection bias* → voir [biais d'échantillonnage](#b), [biais de sélection](#b)
- **sélection de features** (*feature selection*) — enseignée : [ch. 12, 12.6 · La sélection de features][s270] · exercices : 12.10–11, 12.25, 12.Q8, 15.24
- *selection rate* → voir [taux de sélection](#t)
- *self-attention* → voir [auto-attention](#a)
- *self-consistency* → voir [auto-cohérence](#a)
- *self-information* → voir [information (de Shannon), surprise](#i)
- **self-play** — enseignée : [ch. 26, 26.10 · Deux applications][s414] · exercices : 26.37, 26.E5, 26.Q12
- *self-supervised learning* → voir [apprentissage auto-supervisé](#a)
- *semantic search* → voir [recherche sémantique](#r)
- **semi-gradient** — enseignée : [ch. B8, Apprendre une valeur par régression][s638] · exercices : B8.12, B8.Q1–Q2
- *semi-supervised learning* → voir [apprentissage semi-supervisé](#a)
- **sensibilité, invariance à l'implémentation** (*sensitivity, implementation invariance*) — enseignée : [ch. B6, Les gradients intégrés][s590] · exercices : B4.16, B6.3, B6.18–19, B6.Q7, B6.R1
- *sensitive attribute* → voir [attribut sensible](#a)
- *sensitivity* → voir [sensibilité, invariance à l'implémentation](#s)
- *sensitivity analysis* → voir [analyse de sensibilité](#a)
- *sentence embedding* → voir [embedding de phrase](#e)
- **SentencePiece** — enseignée : [ch. B2, 202.5 · WordPiece, Unigram, SentencePiece ; la bibliothèque…][s470] · exercices : B2.13, B2.19, B2.30, B2.E2, B2.Q7
- *sentinel value* → voir [valeur sentinelle](#v)
- **séparabilité linéaire** (*linear separability*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s208] · exercices : 10.1, 10.3, 10.6, 10.11–12, 10.17, 10.Q5–Q6, 10.R2–R3 (+ 4)
- *seq2seq* → voir [encodeur-décodeur](#e)
- *sequence* → voir [séquence](#s)
- **séquence** (*sequence*) — enseignée : [ch. 22, 22.5 · Organiser les entrées][s355] · exercices : 22.13, 22.15, 22.21–23, 22.29, 22.E4, 22.Q5, 22.R1, 22.R3 (+ 4)
- *sequential test* → voir [test séquentiel, règle d'arrêt](#t)
- *sequential updating* → voir [mise à jour bayésienne, séquentielle](#m)
- séquentielle → voir [mise à jour bayésienne, séquentielle](#m)
- **sérialiser** (*serialize*) — enseignée : [ch. 0A, 100.6.4 · Sérialiser : `json` et `pickle`][s4] · exercices : 0A.39, 0A.Q8
- *serialize* → voir [sérialiser](#s)
- **série temporelle** (*time series*) — enseignée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s221] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10 (+ 2) · approfondie : ch. 22
- *Series* → voir [DataFrame, Series](#d)
- Series → voir [DataFrame, Series](#d)
- *service level objective* → voir [SLO](#s)
- *set* → voir [ensemble](#e)
- **set_output** — enseignée : [ch. 15, Au-delà du livre : les données hétérogènes][s291]
- **seuil d'interpolation** (*interpolation threshold*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s239] · exercices : 9.4, 9.11, 9.25, 9.30–31, 9.E5, 9.Q9
- seuil de décision → voir [score, seuil de décision](#s)
- **seuillage doux** (*soft thresholding*) — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s231] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 2)
- **seuils par groupe** (*group-specific thresholds*) — enseignée : [ch. B6, Des taux par groupe][s599] · exercices : B6.5–6, B6.10, B6.23–25, B6.27, B6.E3, B6.Q11, B6.R3
- **SFT** (*supervised fine-tuning, fine-tuning supervisé*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s519] · exercices : B4.10, B4.Q1–Q2, B4.Q7
- SGD → voir [descente de gradient stochastique (SGD)](#d)
- *Sh* → voir [bit (d'information)](#b)
- *shadow deployment* → voir [déploiement canari, en ombre](#d)
- *Shannon* → voir [entropie (de Shannon)](#e)
- *shannon* → voir [bit (d'information)](#b)
- SHAP → voir [graphique de dépendance (SHAP)](#g)
- *shape* → voir [forme](#f), [forme (d'un tenseur)](#f)
- *SHapley Additive exPlanations* → voir [valeurs SHAP](#v)
- **Shapley interventionnel, observationnel** (*interventional, observational Shapley*) — enseignée : [ch. B6, Les features comme joueurs][s581] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- *Shapley kernel* → voir [noyau de Shapley](#n)
- *Shapley value* → voir [valeur de Shapley](#v)
- *shared layer* → voir [couche partagée](#c)
- *shell* → voir [terminal](#t)
- *shell form* → voir [forme exec, forme shell](#f)
- *shifted ReLU* → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- *shortcut learning* → voir [raccourci appris](#r)
- *shortest tokenization* → voir [segmentation la plus courte](#s)
- **shrinkage** (*rétrécissement*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s226]
- *siamese network* → voir [siamois (réseau)](#s)
- **siamois (réseau)** (*siamese network*) — enseignée : [ch. B2, D'une suite de vecteurs à un vecteur de phrase][s486] · exercices : B2.28–29, B2.E4–E5, B2.Q11, B2.R1
- *sigmoid* → voir [sigmoïde](#s)
- *sigmoid linear unit* → voir [SiLU, swish](#s)
- **sigmoïde** (*sigmoid, logistic function*) — enseignée : [ch. 0B, 101.2.5 · Sigmoïde et tangente hyperbolique : définitions et allure][s83] ; [ch. 13, Au-delà du livre : la régression logistique, le classifieur…][s84] · exercices : 0B.17, 0B.36, 0B.Q6, 17.4, 17.13, 17.23, 17.E1, 17.E3, 18.5, 18.R1 (+ 1) · approfondie : ch. 17 · réutilisée : ch. 15, 17, 22, 25, 27, B2
- **sigmoïde et tanh : définitions, allure, symétries** — enseignée : [ch. 0B, 101.2.5 · Sigmoïde et tangente hyperbolique : définitions et allure][s83] · exercices : 0B.17, 0B.36, 0B.Q6, 22.24, B5.13
- *sign function* → voir [marche unité, marche de Heaviside, fonction signe](#m)
- *signal-to-noise ratio* → voir [rapport signal sur bruit (SNR)](#r)
- **silhouette** — enseignée : [ch. 15, 15.5 · Le clustering][s693] · exercices : 7.28–29, 15.13, 15.Q6 · approfondie : ch. 7
- *silhouette coefficient* → voir [coefficient de silhouette](#c)
- **SiLU, swish** (*sigmoid linear unit, swish*) — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s85] · exercices : 17.2, 17.4, 17.8, 17.10, 17.13–14, 17.17–18, 17.Q8–Q9 (+ 6)
- *sim-to-real* → voir [écart simulation-réalité, sim-to-real](#e)
- sim-to-real → voir [écart simulation-réalité, sim-to-real](#e)
- **similarité cosinus** (*cosine similarity*) — enseignée : [ch. 0B, 101.3.3 · Produit scalaire, angle, similarité cosinus][s89] ; [ch. B2, Le cosinus][s90] · exercices : 0B.18–19, 0B.39–41, 0B.E2, 0B.Q8, B2.2, B2.Q11–Q12 (+ 6) · réutilisée : ch. B1
- **similarité cosinus et analogies** — enseignée : [ch. B2, 202.9 · Géométrie des embeddings : cosinus, voisins, analogies,…][s726] · exercices : B2.2, B2.11, B2.23–24, B2.26–27, B2.E4, B2.Q11–Q12
- *simple probability* → voir [probabilité simple, marginale](#p)
- *simplified objective* → voir [loss simplifiée](#l)
- *Simpson's paradox* → voir [paradoxe de Simpson](#p)
- **simulation et loi des grands nombres (intuition)** — enseignée : [ch. 0B, 101.7.5 · Simuler pour estimer : fréquences et loi des grands nombres][s106] · exercices : 0B.52–53, 0B.E5, 0B.Q12
- *single-valued curve* → voir [courbe continue, lisse, univoque](#c)
- *singular value decomposition* → voir [SVD](#s)
- *skip connection* → voir [connexion de saut](#c)
- *skip-gram* → voir [skip-gram, CBOW](#s)
- **skip-gram, CBOW** (*skip-gram, continuous bag of words*) — enseignée : [ch. B2, word2vec : prédire plutôt que compter][s482] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- *sliding window* → voir [fenêtre glissante](#f)
- *sliding window attention* → voir [fenêtre glissante (attention)](#f)
- **SLO** (*service level objective*) — enseignée : [ch. B7, Latence, percentiles et débit][s622] · exercices : B7.3, B7.5, B7.16–19, B7.Q8–Q9, B7.R2–R3 (+ 2)
- *slope-intercept diagram* → voir [diagramme pente-ordonnée](#d)
- *slothful induction* → voir [induction paresseuse](#i)
- *smoke test* → voir [entraînement fumée](#e)
- *smooth* → voir [courbe continue, lisse, univoque](#c)
- **SmoothGrad** — enseignée : [ch. B6, Le choix de la ligne de référence][s589] · exercices : B6.3, B6.18–19, B6.Q7, B6.R1
- *SNN* → voir [réseau impulsionnel](#r)
- SNR → voir [rapport signal sur bruit (SNR)](#r)
- **SNR final nul** (*zero terminal SNR*) — enseignée : [ch. B5, Le rapport signal sur bruit][s551] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 2)
- *soft margin* → voir [marge souple, paramètre C](#m)
- *soft thresholding* → voir [seuillage doux](#s)
- *soft update* → voir [mise à jour douce (de Polyak)](#m)
- *soft voting* → voir [vote dur, vote souple](#v)
- **softmax** — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s191] ; [ch. 17, 17.8 · La softmax][s205] · exercices : 17.6–7, 17.15–17, 17.19, 17.E3, 17.Q11–Q12, 17.R2 (+ 12) · approfondie : ch. 18, 26 · réutilisée : ch. 18, 20, 21, 22, 23, 24, 26, B2, B3, B8
- **softmax en ligne** (*online softmax*) — enseignée : [ch. B3, FlashAttention : le même résultat, sans la matrice][s514] · exercices : B3.8, B3.10, B3.27–28, B3.E2, B3.Q12
- *softmax Jacobian* → voir [Jacobienne de la softmax](#j)
- *softmax policy* → voir [politique softmax, de Boltzmann](#p)
- **softplus** — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s85] · exercices : 17.2, 17.4, 17.8, 17.10, 17.13–14, 17.17–18, 17.Q8–Q9 (+ 6)
- **solide** (*sound*) — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5, 11.15, 11.Q7, CP2.9
- **solution de norme minimale, pseudo-inverse** (*minimum-norm solution, Moore-Penrose pseudo-inverse*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s239] · exercices : 9.4, 9.11, 9.25, 9.30–31, 9.E5, 9.Q9
- **somme de contrôle** (*checksum, hash*) — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s125]
- **somme pondérée, pré-activation** (*weighted sum, pre-activation*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s208] · exercices : 10.1, 10.3, 10.6, 10.11–12, 10.17, 10.Q5–Q6, 10.R2–R3 (+ 6) · approfondie : ch. 0B · réutilisée : ch. 16, 21
- *sonde linéaire* → voir [linear probe](#l)
- **sophisme** (*fallacy*) — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s255] · exercices : 11.4, 11.15–16, 11.Q8, 11.Q10, CP2.9
- **sophismes inductifs** — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s257] · exercices : 11.17–18, 11.E4, 11.Q10
- **sophismes syllogistiques** — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s255] · exercices : 11.4, 11.15–16, 11.Q8
- **sortie structurée** (*structured output*) — enseignée : [ch. B4, 204.4 · Prompting : zéro-shot, few-shot, instructions, sortie…][s526] · exercices : B4.15–16, B4.E1, B4.Q6
- *sorties d'une couche* → voir [fan-in, fan-out](#f)
- **sorties de nn.LSTM et état porté** — enseignée : [ch. 24, 24.5.6 · Les RNN profonds][s718] · exercices : 24.4, 24.26
- **souche** (*decision stump*) — enseignée : [ch. 13][s274] ; [ch. 14, 14.7 · Le boosting][s275] · exercices : 14.2, 14.4, 14.6–8, 14.10, 14.Q8–Q11 (+ 12)
- *sound* → voir [solide](#s)
- **sources de datasets (HF Hub, Kaggle, OpenML, UCI)** — enseignée : [ch. 29, 29.4 · Les collections de datasets][s441] · exercices : 29.6, 29.11, 29.15, 29.E2, 29.Q7
- **sous-échantillonnage des mots fréquents** (*subsampling of frequent words*) — enseignée : [ch. B2, word2vec : prédire plutôt que compter][s482] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- **sous-gradient** (*subgradient*) — enseignée : [ch. 5, 5.2 · Introduction : fonctions, courbes et surfaces][s179] · exercices : 5.14, 5.Q2–Q3, 5.Q9 · réutilisée : ch. 17
- **sous-séquence de pas** (*timestep spacing*) — enseignée : [ch. B5, L'idée de DDIM][s558] · exercices : B5.5, B5.8, B5.17–18, B5.24–25, B5.30–31, B5.Q7–Q8 (+ 1)
- *span corruption* → voir [corruption de segments](#c)
- *sparse* → voir [parcimonieux](#p)
- *sparse autoencoder* → voir [autoencodeur parcimonieux](#a)
- *Spearman rank correlation* → voir [corrélation de rangs (de Spearman)](#c)
- *special pleading* → voir [plaidoyer spécial](#p)
- *special token* → voir [token spécial](#t)
- *specification gaming* → voir [piratage de la récompense](#p)
- **spécificité** (*specificity, true negative rate, TNR*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s161] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11
- *specificity* → voir [spécificité](#s)
- *speculative decoding* → voir [décodage spéculatif](#d)
- *spherical linear interpolation* → voir [interpolation sphérique (slerp)](#i)
- *spiking neural network* → voir [réseau impulsionnel](#r)
- **split** (*découpe*) — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s197] · exercices : 13.2–3, 13.20, 13.22, 13.25, 13.E2, 13.Q9
- **SQL** (*structured query language*) — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s272] · exercices : 12.15
- **SQL de base (sqlite3, pd.read_sql)** — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s272]
- *squelette* → voir [stub](#s)
- *src layout* → voir [layout `src/`](#l)
- SSM → voir [modèle à espace d'états (SSM)](#m)
- **stabilité numérique (log-sum-exp)** — enseignée : [ch. 17][s696]
- *stacking* → voir [stacking, méta-modèle](#s)
- **stacking, méta-modèle** (*stacking, meta-learner*) — enseignée : [ch. 14, 14.3 · Le vote][s280] · exercices : 14.1, 14.11–14, 14.Q3
- *stagewise additive modeling using a multi-class exponential loss* → voir [SAMME](#s)
- *standard deviation* → voir [variance, écart-type](#v)
- *standard error* → voir [erreur type](#e)
- **standardisation (centrage, réduction)** (*standardization*) — enseignée : [ch. 12, 12.5.2 · La standardisation][s266] · exercices : 12.2, 12.6, 12.16, 12.29, 12.E2, 12.Q5, CP3.4, CP4.12 · réutilisée : ch. 13, 20
- standardiser → voir [z-score, standardiser](#z)
- *standardization* → voir [standardisation (centrage, réduction)](#s)
- *standardize* → voir [z-score, standardiser](#z)
- *state* → voir [état, transition](#e)
- *state dictionary* → voir [`state_dict`](#s)
- *state space model* → voir [modèle à espace d'états (SSM)](#m)
- **state_dict et checkpoints** — enseignée : [ch. 23, 23.10.1 · Checkpoints][s711] · exercices : 23.25, 23.E3
- **`state_dict`** (*state dictionary*) — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s2] · exercices : B7.R2 · approfondie : ch. 23 · réutilisée : ch. 25, 27, B7
- *stateful RNN* → voir [état porté](#e)
- *static* → voir [embedding statique, contextuel](#e)
- *static quantization* → voir [quantification dynamique, statique](#q)
- *stationary inflection point* → voir [point d'inflexion à tangente horizontale](#p)
- *stationary point* → voir [point critique, point stationnaire](#p)
- statique → voir [quantification dynamique, statique](#q)
- *statistical parity* → voir [parité démographique](#p)
- *statistical syllogism* → voir [syllogisme statistique](#s)
- **statistiques descriptives** — enseignée : [ch. 2, 2.1 · Pourquoi des statistiques ?][s685] · exercices : 2.Q1, 2.R1
- **statistiques glissantes** (*running statistics*) — enseignée : [ch. 20, 20.4.4 · La batchnorm][s240] · exercices : 20.4, 20.8, 20.12, 20.22, 20.24, 20.28–29, 20.E2, 20.Q6 · réutilisée : ch. B1
- *status code* → voir [code de statut](#c)
- *STE* → voir [estimateur *straight-through*](#e)
- *stemming* → voir [racinisation](#r)
- *step* → voir [workflow, job, étape, action](#w)
- *step decay* → voir [planning par paliers](#p)
- *step distillation* → voir [distillation (de l'échantillonneur)](#d)
- *stochastic gradient boosting* → voir [gradient boosting stochastique](#g)
- *stochastic gradient descent* → voir [descente de gradient stochastique (SGD)](#d)
- *stop string* → voir [chaîne d'arrêt](#c)
- *stopping rule* → voir [test séquentiel, règle d'arrêt](#t)
- *storage limitation* → voir [minimisation, limitation de la conservation](#m)
- *straight-through estimator* → voir [estimateur *straight-through*](#e)
- **stratégies de décodage (glouton, température, top-k, top-p)** — enseignée : [ch. 28][s349] · exercices : 28.E3
- **stratification** — enseignée : [ch. 8, 8.3 · Le jeu de test][s122] · exercices : 8.1, 8.3–4, 8.7, 8.10–13, 8.Q5–Q6 (+ 5) · réutilisée : ch. 15, 23, 24, 29
- *stratified bootstrap* → voir [bootstrap stratifié (par groupe)](#b), [bootstrap stratifié (de runs)](#b)
- **StratifiedKFold** — enseignée : [ch. 15, 15.9.1 · La validation croisée][s694] · exercices : 15.17, 15.Q10
- **streaming** (*diffusion au fil de l'eau*) — enseignée : [ch. B4, 204.3 · Génération : generate(), décodage, arrêt, streaming ; chat…][s520] · exercices : B4.4, B4.12–14, B4.Q4–Q5, B4.R1
- **stride** (*pas*) — enseignée : [ch. 20, 20.4.6 · Le pooling (aperçu)][s22] · exercices : 20.14, 20.20, 20.Q7, 21.3, 21.15, 21.26, 21.E2, 21.Q5, 24.Q9, 24.R2 (+ 2) · approfondie : ch. 21 · réutilisée : ch. 24, 25, 27
- *string* → voir [chaîne de caractères](#c)
- *strong learner* → voir [apprenant faible, apprenant fort](#a)
- **structure d'un projet Python** — enseignée : [ch. B7, 207.1 · Du notebook au projet : limites du notebook, structure…][s605] · exercices : B7.6, B7.8–9, B7.25, B7.E1, B7.Q1–Q2
- *structured output* → voir [sortie structurée](#s)
- *structured query language* → voir [SQL](#s)
- **stub** (*squelette*) — enseignée : [ch. 0A, 100.11.5 · mylearn : stubs, référence et tests fondés sur un oracle][s77] · exercices : 0A.26, 0A.63–66, 0A.Q2 · réutilisée : ch. 0B
- *Student's t distribution* → voir [loi de Student](#l)
- *style loss* → voir [loss de style](#l)
- *style reference* → voir [image de contenu, référence de style, image générée](#i)
- *subgradient* → voir [sous-gradient](#s)
- *subnormal number* → voir [nombre dénormalisé](#n)
- *subsampling of frequent words* → voir [sous-échantillonnage des mots fréquents](#s)
- *substitut* → voir [variable de substitution](#v)
- **substitut local, LIME** (*local surrogate, local interpretable model-agnostic explanations*) — enseignée : [ch. B6, Deux autres explications locales : substitut et contrefactuel][s586] · exercices : B6.2, B6.7, B6.11, B6.15–17, B6.E1–E2, B6.Q5–Q6
- **successive halving** (*réduction de moitié successive*) — enseignée : [mini-projet 2, Les extensions (pour aller plus loin)][s663] ; [ch. 15, 15.9.4 · La recherche aléatoire][s288] · exercices : 15.5, 15.19–20, 15.E2, 15.Q10
- **suite géométrique, raison** (*geometric sequence, ratio*) — enseignée : [ch. 0B, 101.1.5 · Suites géométriques et leur somme][s80] · exercices : 0B.13–14, 0B.34, 0B.Q4, 15.7, B1.Q1 · réutilisée : ch. 22, 26
- **suites géométriques et leur somme (finie et infinie)** — enseignée : [ch. 0B, 101.1.5 · Suites géométriques et leur somme][s80] · exercices : 0B.13–14, 0B.34, 0B.Q4
- **suivi d'expériences** — enseignée : [ch. B7, 207.5 · Suivi d'expériences : runs, métriques, artefacts, comparaison][s616] · exercices : B7.2, B7.14, B7.25, B7.E2, B7.R1
- *supervised fine-tuning* → voir [SFT](#s)
- *supervised learning* → voir [apprentissage supervisé](#a)
- **supervision humaine** (*human oversight*) — enseignée : [ch. B6, 206.11 · Éthique et régulation en pratique : fiches modèles,…][s601] · exercices : B6.8–9, B6.28, B6.Q12
- *support* → voir [support (d'une classe)](#s)
- **support (d'une classe)** (*support*) — enseignée : [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s33] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11, 13.16, 13.Q5
- **support (few-shot)** (*support set*) — enseignée : [ch. B1, Le few-shot et les prototypes de classe][s454] · exercices : 13.16, 13.Q5, B1.20–21, B1.25, B1.28, B1.E1–E2, B1.Q8
- *support set* → voir [support (few-shot)](#s)
- *support vector* → voir [vecteur de support](#v)
- *support vector machine* → voir [SVM, machine à vecteurs de support](#s)
- *sur politique* → voir [on-policy, off-policy](#o)
- **sur-optimisation du modèle de récompense** (*reward model overoptimization*) — enseignée : [ch. B8, Le piratage de la récompense][s653] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- **surcharge (de configuration)** (*override*) — enseignée : [ch. B7, 207.3 · Configuration : dataclasses, YAML ou JSON, surcharges,…][s610] · exercices : B7.11, B7.25, B7.E1, B7.Q4 · approfondie : ch. 0A · réutilisée : ch. 18
- **surcoût de tokenisation** (*tokenization premium*) — enseignée : [ch. B2, 202.4 · BPE au niveau octet, pré-tokenisation et tokens spéciaux][s468] · exercices : B2.4, B2.16–18, B2.30, B2.E3, B2.Q5–Q6
- **suréchantillonnage** (*upsampling*) — enseignée : [ch. 20, 20.4.8 · Les autres couches utilitaires][s330] ; [ch. 25, 25.7 · Autoencodeurs convolutifs][s331] · exercices : 20.1, 20.7, 20.14, 20.17, 20.20, 20.30, 20.E2, 20.Q9, 25.3, 25.Q10 (+ 3) · réutilisée : ch. B5
- **surestimation** (*overestimation bias*) — enseignée : [ch. B8, La surestimation, et Double DQN][s640] · exercices : B8.1–2, B8.13–15, B8.E1, B8.Q3–Q4, B8.Q9, B8.R1
- *surprisal* → voir [information (de Shannon), surprise](#i)
- surprise → voir [information (de Shannon), surprise](#i)
- *surrogate objective* → voir [objectif de substitution](#o)
- **surveillance de la dérive** — enseignée : [ch. B7, 207.10 · Surveiller en production : dérive des données et des…][s631] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- **survey des couches (conv, pooling, RNN, utilitaires)** — enseignée : [ch. 20, 20.4 · Un tour d'horizon des couches][s704] · exercices : 20.Q4
- *survey imputation* → voir [imputation (d'enquête)](#i)
- **SVD** (*décomposition en valeurs singulières, singular value decomposition*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s152] · exercices : 12.7–8, 12.22–23, 12.27, 12.Q9, B2.24 · approfondie : ch. B2 · réutilisée : ch. B2
- **SVD tronquée** (*truncated SVD*) — enseignée : [ch. B2, Réduire : la SVD tronquée][s481] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- **SVM, machine à vecteurs de support** (*support vector machine*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s244] · exercices : 13.4–5, 13.10–11, 13.16–18, 13.36, 13.Q5–Q6 (+ 1)
- **SwiGLU** (*unité linéaire à porte avec SiLU*) — enseignée : [ch. 16, 16.1 · Pourquoi ce chapitre ?][s293] ; [ch. 17, 17.6 · Les fonctions lisses][s85] · exercices : 16.9, 16.Q1, 16.R2, 17.2, 17.4, 17.8, 17.10, 17.13, 17.Q8–Q9 (+ 9)
- *swish* → voir [SiLU, swish](#s)
- swish → voir [SiLU, swish](#s)
- *sycophancy* → voir [flagornerie](#f)
- *syllogism* → voir [syllogisme](#s)
- **syllogisme** (*syllogism*) — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5–6, 11.15, 11.Q7, 11.Q9, CP2.9
- **syllogisme statistique** (*statistical syllogism*) — enseignée : [ch. 11, 11.5 · L'induction][s256] · exercices : 11.6, 11.17, 11.Q9
- **syllogismes** — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5, 11.15, 11.Q7, CP2.9
- symétrie → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *symmetry* → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *symmetry breaking* → voir [brisure de symétrie](#b)
- **synapse** — enseignée : [ch. 10, 10.2 · Les vrais neurones][s241] · exercices : 10.8–9, 10.E4, 10.Q2–Q3
- **synchrone, asynchrone** (*synchronous, asynchronous*) — enseignée : [ch. 16, 16.3 · Flux synchrone et asynchrone][s295] · exercices : 16.5, 16.16, 16.E4, 16.Q5–Q6
- *synchronous* → voir [synchrone, asynchrone](#s)
- *synthetic data* → voir [données synthétiques](#d)
- **système expert** (*expert system*) — enseignée : [ch. 1, 1.1.2 · Les systèmes experts et leurs limites][s110] · exercices : 1.15, 1.18, 1.25, 1.Q2

## T

- *t-distributed stochastic neighbor embedding* → voir [t-SNE, UMAP](#t)
- **t-SNE, UMAP** (*t-distributed stochastic neighbor embedding, uniform manifold approximation and projection*) — enseignée : [ch. 12, Au-delà du livre : voir des données en grande dimension][s271] · exercices : 12.30, B1.15, B2.27 · réutilisée : ch. B1, B2
- **table de contingence** (*contingency table, cross-tabulation*) — enseignée : [ch. 3, 3.6 · Probabilité marginale et tables de contingence][s158] · exercices : 3.3–4, 3.14, 3.Q5, 3.R3
- **table Q, valeur Q** (*Q-table, Q-value*) — enseignée : [ch. 26, 26.6 · Le Q-learning][s408] · exercices : 26.9, 26.Q8, CP6.10, B8.Q1
- **tableau croisé** (*pivot table*) — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s272]
- *tableau NumPy* → voir [array](#a)
- *tabular foundation model* → voir [modèle de fondation tabulaire](#m)
- **taille de l'état** (*hidden size*) — enseignée : [ch. 22, 22.4.1 · Une cellule à plus d'état][s359] · exercices : 22.3, 22.12, 22.15, 22.Q4
- **tampon (d'un optimiseur)** (*buffer, optimizer state*) — enseignée : [ch. 19, 19.5 · Les variantes de la descente de gradient][s320] · exercices : 19.13, 19.16, 19.26–27, 19.E4, 19.Q9
- **tangente hyperbolique** (*hyperbolic tangent*) — enseignée : [ch. 0B, 101.2.5 · Sigmoïde et tangente hyperbolique : définitions et allure][s83] ; [ch. 17, 17.6 · Les fonctions lisses][s85] · exercices : 0B.17, 0B.36, 0B.Q6, 17.2, 17.4, 17.8, 17.10, 17.13, 17.Q8–Q9 (+ 9)
- *target encoding* → voir [encodage par la cible](#e)
- *target network* → voir [réseau cible](#r)
- *targeted* → voir [attaque ciblée, non ciblée](#a)
- **taux d'accroissement** (*difference quotient*) — enseignée : [ch. 0B, 101.5.1 · Taux d'accroissement, tangente, dérivée][s95] · exercices : 0B.10, 0B.47, 0B.Q10 · réutilisée : ch. 5
- **taux d'erreur** (*error rate*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s29] · exercices : 1.14, 1.18, 1.20, 1.25, 1.E2, 1.Q6
- taux de base → voir [prévalence, taux de base](#p)
- **taux de compression** (*compression ratio*) — enseignée : [ch. 6, Mélanger les codes][s200] ; [ch. B2, Deux rapports][s201] · exercices : 6.5, 6.7, 6.16, 6.18, 6.20, 6.22, 6.26, 6.E1, 6.Q10, 6.Q12 (+ 10)
- **taux de fausses découvertes, de fausses omissions** (*false discovery rate, false omission rate, FDR, FOR*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s161] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11
- **taux de faux positifs, de faux négatifs** (*false positive rate, false negative rate*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s161] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11
- **taux de sélection** (*selection rate*) — enseignée : [mini-projet 3, La grille d'évaluation (sur 20)][s664]
- *TDM* → voir [fouille de textes et de données](#f)
- **teacher forcing** (*forçage par l'enseignant*) — enseignée : [ch. B3, Teacher forcing et décalage d'un token][s510] · exercices : B3.5, B3.11, B3.19, B3.E3, B3.Q5, B3.Q9
- *technical debt* → voir [dette technique (du ML)](#d)
- **téléchargement reproductible (révision, SHA-256)** — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s125] · exercices : PF.2
- *temperature* → voir [température (d'une softmax)](#t)
- **température (d'une softmax)** (*temperature*) — enseignée : [ch. 17, 17.8 · La softmax][s205] · exercices : 17.6–7, 17.15–17, 17.19, 17.E3, 17.Q11–Q12, 17.R2 (+ 8) · approfondie : ch. 22, 28 · réutilisée : ch. 26, B3, B4, B8
- *temporal-difference error* → voir [erreur TD](#e)
- **tenseur** (*tensor*) — enseignée : [ch. 5, Au-delà du livre (3) : la différentiation automatique][s188] · exercices : 20.1, 20.13, 20.Q2, 23.12, 23.Q5, B1.1 · approfondie : ch. 20 · réutilisée : ch. 21, 23, 25
- **tenseur contigu** (*contiguous tensor*) — enseignée : [ch. 23, 23.3.2 · Tenseurs et tableaux][s372] · exercices : 23.12, 23.Q5
- **tenseurs PyTorch** — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s327] · exercices : 20.1, 20.13, 20.Q2
- *tensor* → voir [tenseur](#t)
- *term frequency* → voir [TF-IDF](#t)
- **terme distribué** (*distributed term*) — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5, 11.15, 11.Q7
- **terminal** (*shell*) — enseignée : [ch. 0A, 100.1 · Prise en main : notebooks, terminal et workbook][s28] · exercices : 0A.10, 0A.Q1, B8.1
- **terminal : cd, chemins, lancer python et pytest** — enseignée : [ch. 0A, 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest…][s49] · exercices : 0A.10, 0A.Q1, 0B.29, 0B.51, 9.22 · approfondie : ch. 0B · réutilisée : ch. 18, 23
- *terminal state* → voir [état terminal, épisode tronqué](#e)
- **test A/B** (*A/B test*) — enseignée : [checkpoint 1][s660]
- **test d'équivalence** (*equivalence test*) — enseignée : [mini-projet 4, Les livrables][s666]
- **test d'overfitting d'un mini-batch** (*overfit-one-batch test*) — enseignée : [ch. B7, Quatre familles de tests][s607] · exercices : B7.15, B7.23, B7.25, B7.E3, B7.Q7
- **test de confirmation** (*confirmatory test*) — enseignée : [ch. 3, 3.8 · Appliquer la matrice de confusion : le piège de la prévalence][s162] · exercices : 3.7, 3.10, 3.21, 3.26–27, 3.E1, 3.Q12
- **test de données** (*data test*) — enseignée : [ch. B7, 207.6 · Tester un projet ML : tests unitaires, tests de données,…][s617] · exercices : B7.15, B7.23, B7.25, B7.E3, B7.Q7
- **test de Kolmogorov-Smirnov** (*Kolmogorov-Smirnov test*) — enseignée : [ch. B7, Le test de Kolmogorov-Smirnov][s629] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- **test de McNemar** (*McNemar's test*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s222] · exercices : 8.2, 8.4, 8.8, 8.16, 8.18, 8.26–27, 8.E2, 8.E4, 8.Q11
- **test de non-régression** (*regression test*) — enseignée : [ch. 7][s216] · exercices : B2.30
- **test de propriété** (*property-based test*) — enseignée : [ch. 17, Au-delà du livre : choisir une activation, et tester une famille…][s303]
- **test de propriétés** (*property-based testing*) — enseignée : [ch. 5][s184]
- **test de santé** (*sanity check*) — enseignée : [ch. B6, Les tests de santé][s592] · exercices : B6.16, B6.22, B6.E2, B6.E4, B6.Q6, B6.Q9
- **test par mutation** (*mutation testing*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s222] · exercices : 8.2, 8.4, 8.8, 8.16, 8.18, 8.26–27, 8.E2, 8.E4, 8.Q11
- **test par permutation** (*permutation test*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s222] · exercices : 8.2, 8.4, 8.8, 8.16, 8.18, 8.26–27, 8.E2, 8.E4, 8.Q11
- **test paramétré** (*parametrized test*) — enseignée : [ch. 17, Au-delà du livre : choisir une activation, et tester une famille…][s303]
- **test séquentiel, règle d'arrêt** (*sequential test, stopping rule*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s173] · exercices : 4.3, 4.10, 4.14, 4.23, 4.E3, 4.Q6–Q7
- **test unitaire** (*unit test*) — enseignée : [ch. 0A, 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest…][s49] · exercices : 0A.10, 0A.Q1
- **tests de santé des explications** — enseignée : [ch. B6, 206.7 · Limites des explications : instabilité, tests de santé,…][s739] · exercices : B6.16, B6.22, B6.E2, B6.E4, B6.Q6, B6.Q9
- **tests paramétrés pytest** — enseignée : [ch. 17][s696]
- **tests pour le ML** — enseignée : [ch. B7, 207.6 · Tester un projet ML : tests unitaires, tests de données,…][s617] · exercices : B7.15, B7.23, B7.25, B7.E3, B7.Q7
- **tête (d'un modèle)** (*head*) — enseignée : [ch. 23, 23.3.4 · La forme des tenseurs d'images][s373] · exercices : 16.E3, 23.2, 23.24, 23.Q6, 23.R1, B1.1, B1.4, B1.R2, B3.19, B3.Q6 (+ 2) · réutilisée : ch. B1
- **tête d'attention** (*attention head*) — enseignée : [ch. B3, Plusieurs questions à la fois][s499] · exercices : B3.3, B3.11, B3.14, B3.18, B3.24, B3.E1, B3.Q6
- *text and data mining* → voir [fouille de textes et de données](#f)
- **TF-IDF** (*term frequency, inverse document frequency*) — enseignée : [ch. 13, 13.6 · Naïve Bayes][s177] · exercices : 13.6–7, 13.27–30, 13.E4, 13.Q11, B4.28 · réutilisée : ch. B4
- TFR → voir [retour, retour total (TFR)](#r)
- **théorème d'impossibilité** (*impossibility theorem*) — enseignée : [ch. B6, L'impossibilité][s163] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- théorème de Bayes → voir [règle de Bayes, théorème de Bayes](#r)
- **théorème de convergence du perceptron** (*perceptron convergence theorem*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s243] · exercices : 10.4–5, 10.7, 10.10–11, 10.13, 10.16, 10.19, 10.21, 10.Q7 (+ 4)
- **théorème du cycle du perceptron** (*perceptron cycling theorem*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s243] · exercices : 10.4–5, 10.7, 10.10–11, 10.13, 10.16, 10.19, 10.21, 10.Q7 (+ 4)
- **théorème du gradient de politique** (*policy gradient theorem*) — enseignée : [ch. B8, Le théorème du gradient de politique][s642] · exercices : B8.3, B8.7, B8.16–17, B8.E1, B8.Q5–Q6, B8.Q9, B8.R3
- **théorème du jury** (*Condorcet's jury theorem*) — enseignée : [ch. 14, 14.2 · Les ensembles][s56] · exercices : 14.5, 14.9, 14.13, 14.Q1–Q2
- **théorème No Free Lunch** (*no free lunch theorem*) — enseignée : [ch. 11, 11.2.3 · L'optimisation][s249] ; [ch. 19, 19.6 · Choisir un optimiseur][s250] · exercices : 11.12, 11.E3, 11.Q5, 19.9, 19.11, 19.23, 19.26–28, 19.Q12 (+ 1)
- *Thompson sampling* → voir [échantillonnage de Thompson](#e)
- *three-sigma rule* → voir [règle 68-95-99,7](#r)
- *threshold logic unit* → voir [neurone formel](#n)
- *threshold policy* → voir [politique de seuil](#p)
- *throughput* → voir [débit](#d)
- *Tikhonov regularization* → voir [Ridge (pénalité L2)](#r)
- *time series* → voir [série temporelle](#s)
- *time series cross-validation* → voir [validation glissante](#v)
- *time stamp* → voir [horodatage](#h)
- *time step* → voir [pas de temps](#p)
- *time-distributed layer* → voir [couche appliquée à chaque pas](#c)
- *timestep embedding* → voir [encodage du pas de temps](#e)
- *timestep spacing* → voir [sous-séquence de pas](#s)
- **tirage** (*draw, sample*) — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s136] · exercices : 2.1–2, 2.6, 2.13, 2.16, 2.19, 2.E1, 2.Q1–Q2, 2.R2 (+ 6) · réutilisée : ch. 3, 7
- **tirage avec remise, sans remise** (*sampling with replacement, without replacement*) — enseignée : [ch. 2, 2.5 · Tirages avec et sans remise][s144] · exercices : 2.6, 2.21, 2.Q9, 3.R1 · réutilisée : ch. 3, 8
- **tirage catégoriel** — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s136] · exercices : 2.1–2, 2.13, 2.16, 2.19, 2.E1, 2.Q1–Q2, 2.R2
- *tirage de features* → voir [feature bagging](#f)
- *TN* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *TNR* → voir [spécificité](#s)
- **token** (*jeton, unité lexicale*) — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s18] · exercices : B2.8–9, B2.14, B2.20, B2.31, B2.Q5–Q6, B2.R2, B3.Q10, B3.R1 (+ 6) · approfondie : ch. B2 · réutilisée : ch. 28, B2, B3
- **token `[CLS]`** (*classification token*) — enseignée : [ch. B3, Teacher forcing et décalage d'un token][s510] · exercices : B3.5, B3.11, B3.19, B3.E3, B3.Q5, B3.Q9
- **token spécial** (*special token*) — enseignée : [ch. B2, Les tokens spéciaux][s469] · exercices : B2.4, B2.16–18, B2.30, B2.E3, B2.Q5–Q6
- *tokeniseur* → voir [tokenizer](#t)
- *tokenization premium* → voir [surcoût de tokenisation](#s)
- **tokenizer** (*tokeniseur*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s191] · exercices : B2.13, B2.18–19, B2.21, B2.30, B2.E3 · approfondie : ch. B2 · réutilisée : ch. B3
- *top-1 agreement* → voir [accord top-1](#a)
- *top-5 accuracy* → voir [accuracy top-5](#a)
- **top-k et top-p** — enseignée : [ch. B3][s502] · réutilisée : ch. 28
- *top-k sampling* → voir [top-k, top-p](#t)
- **top-k, top-p** (*top-k sampling, nucleus sampling*) — enseignée : [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s203] · exercices : 22.1, 22.5, 22.8, 22.12, 22.20, 22.24–27, 22.Q12 (+ 5) · approfondie : ch. 28, B3 · réutilisée : ch. 28
- top-p → voir [top-k, top-p](#t)
- *topological order* → voir [ordre topologique](#o)
- *torch.compile* → voir [compilation (d'un modèle)](#c)
- torch.export → voir [exporter (un modèle), `torch.export`](#e)
- **torchao** — enseignée : [ch. B7, Accélérer sur CPU][s625] · exercices : B7.1, B7.19–21, B7.25, B7.E5, B7.Q9–Q10
- **TorchScript** — enseignée : [ch. B7, torch.export et ONNX][s624] · exercices : B7.1, B7.19–21, B7.25, B7.E5, B7.Q9–Q10
- **torchvision.transforms.v2** — enseignée : [ch. 24, 24.4.2 · Préparer les données pour un CNN][s716] · exercices : 24.19–20, 24.23, 24.Q7
- *total future reward* → voir [retour, retour total (TFR)](#r)
- *total variation* → voir [variation totale](#v)
- *TP* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *TPU* → voir [GPU, TPU, NPU](#g)
- TPU → voir [GPU, TPU, NPU](#g)
- **trace** — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s426]
- *trace de la pile d'appels* → voir [traceback](#t)
- **traceback** (*trace de la pile d'appels*) — enseignée : [ch. 0A, 100.6.2 · Exceptions : lire un traceback, `raise`, `try`/`except`][s62] · exercices : 0A.25, 0A.37, 0A.Q8
- *training* → voir [entraînement](#e)
- *training checkpoint* → voir [checkpoint (d'entraînement)](#c)
- *training error* → voir [erreur d'entraînement, erreur de généralisation](#e)
- *training history* → voir [historique (d'entraînement)](#h)
- *training mode* → voir [mode entraînement, mode évaluation](#m)
- *training set / test set* → voir [jeu d'entraînement / jeu de test](#j)
- **training-serving skew** (*décalage entre entraînement et service*) — enseignée : [ch. 12, 12.2 · La règle d'or : apprendre la transformation sur l'entraînement][s123] · exercices : 12.9, 12.20, 12.31–33, 12.Q1
- *transfer function* → voir [fonction de transfert](#f)
- **transfer learning** (*apprentissage par transfert*) — enseignée : [ch. 23, 23.9.4 · Utiliser un modèle pré-entraîné][s378] ; [ch. B1, Des données et du calcul que tu n'as pas][s379] · exercices : 23.10–11, 23.24, B1.7–8, B1.10, B1.Q1–Q2, B1.R1, B1.R3 (+ 3)
- **transférabilité** (*transferability*) — enseignée : [ch. 21, 21.10 · Les adversaires][s351] · exercices : 21.10, 21.29, 21.E4, 21.Q11
- *transferability* → voir [transférabilité](#t)
- **transfert de style neuronal** (*neural style transfer*) — enseignée : [ch. 28, 28.4 · Le transfert de style][s432] · exercices : 28.9–10, 28.E2, 28.Q6, 28.R2
- **transfert négatif** (*negative transfer*) — enseignée : [ch. B1, 201.7 · Quand le transfert échoue : distance de domaine, transfert…][s452] · exercices : B1.6, B1.9–10, B1.16, B1.22–23, B1.E2, B1.Q8–Q9, B1.R1
- **transformateur (scikit-learn)** (*transformer*) — enseignée : [ch. 12, 12.5.3 · Retenir la transformation][s267] · exercices : 12.4, 12.13, 12.16, 12.20, 12.33, 12.Q6, 15.14 · réutilisée : ch. 15
- **transformation inverse** — enseignée : [ch. 12, 12.5.3 · Retenir la transformation][s267] · exercices : 12.4, 12.13, 12.16, 12.20, 12.33, 12.Q6
- **transformation univariée, multivariée** (*univariate, multivariate transformation*) — enseignée : [ch. 12, 12.5.4 · Univariée ou multivariée][s268] · exercices : 12.3, 12.19, 12.Q7 · approfondie : ch. 18 · réutilisée : ch. 18
- **Transformer** — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s18] · exercices : 0A.22, 12.21, CP3.2, 17.E4, 21.E5, 22.E1, 22.Q12, B3.28, B3.E5, B3.Q8 (+ 2) · approfondie : ch. B3, B4 · réutilisée : ch. B1, B4, B7
- *Transformer block* → voir [bloc Transformer](#b)
- **transformer de diffusion (DiT)** (*diffusion transformer*) — enseignée : [ch. B5, Les transformers de diffusion][s570] · exercices : 11.E3, 18.Q3, B5.8, B5.10, B5.20, B5.E3, B5.E5, B5.Q11, B6.Q6
- **transformers en pratique (5.x)** — enseignée : [ch. B4, 204.2 · Charger et exécuter un modèle avec transformers :…][s730] · exercices : B4.1, B4.7, B4.12, B4.29, B4.Q3
- *transition* → voir [état, transition](#e)
- transition → voir [état, transition](#e)
- *transpose* → voir [transposée](#t)
- *transposed convolution* → voir [convolution transposée](#c)
- **transposée** (*transpose*) — enseignée : [ch. 0B, 101.4.1 · Forme, éléments, transposée ; un dataset est une matrice][s92] · exercices : 0A.53, 0B.9, 0B.42, 0B.45, 0B.Q9, 0B.R2, 16.20, 21.6, 21.Q9, 25.4 · approfondie : ch. 21 · réutilisée : ch. 9, 16, 25, 27, 28, B3, B5
- *trapezoidal rule* → voir [méthode des trapèzes](#m)
- *Tree SHAP* → voir [TreeSHAP](#t)
- **TreeSHAP** (*Tree SHAP*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s582] · exercices : B6.2, B6.7, B6.11, B6.15–17, B6.E1–E2, B6.Q5–Q6
- **triade mortelle** (*deadly triad*) — enseignée : [ch. B8, Instabilité : la triade mortelle][s637] · exercices : B8.12, B8.Q1–Q2
- *triggered agent* → voir [agent libre, déclenché](#a)
- *trigram model* → voir [modèle unigramme, bigramme, trigramme](#m)
- trigramme → voir [modèle unigramme, bigramme, trigramme](#m)
- *tronc* → voir [backbone](#b)
- TRPO → voir [région de confiance, TRPO](#r)
- *true negative* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *true negative rate* → voir [spécificité](#s)
- *true positive* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *truncated BPTT* → voir [rétropropagation tronquée dans le temps](#r)
- *truncated SVD* → voir [SVD tronquée](#s)
- *truncation* → voir [état terminal, épisode tronqué](#e)
- *truncation error* → voir [erreur de troncature](#e)
- *trust region* → voir [région de confiance, TRPO](#r)
- *Tukey's fences* → voir [règle de Tukey](#r)
- *tuple* → voir [liste, tuple](#l)
- tuple → voir [liste, tuple](#l)
- *typage canard* → voir [duck typing](#d)
- **type** — enseignée : [ch. 0A, 100.2 · Valeurs, variables et types][s50] · exercices : 0A.Q3, 8.4, 23.Q5 · approfondie : ch. 8
- *type hint* → voir [annotation de type](#a)
- *typed configuration* → voir [configuration typée](#c)

## U

- **U-Net** — enseignée : [ch. B5, 205.4 · Le débruiteur : U-Net minuscule et encodage du pas de temps][s560] · exercices : B5.11, B5.19, B5.21, B5.Q5–Q6, B5.R1
- **U-Net conditionné par le temps** — enseignée : [ch. B5, Dire au réseau quel est le pas : l'encodage du pas de temps][s735] · exercices : B5.11, B5.19, B5.21, B5.Q5–Q6, B5.R1
- **UCB** (*upper confidence bound*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s121] · exercices : 11.2, 11.7–8, 11.10–11, 11.13, 11.19–21, 11.R1 (+ 9) · réutilisée : ch. 26
- *UE* → voir [règlement européen sur l'IA](#r), [RGPD](#r)
- UMAP → voir [t-SNE, UMAP](#t)
- **un-contre-tous (OvR)** (*one-versus-rest, one-vs-all, OvA*) — enseignée : [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s33] ; [ch. 7, 7.4.1 · Un-contre-tous (OvR)][s165] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11, 7.1, 7.16, 7.22, 7.Q6 (+ 5) · réutilisée : ch. 10, 13, 14
- **un-contre-un (OvO)** (*one-versus-one*) — enseignée : [ch. 7, 7.4.2 · Un-contre-un (OvO)][s209] · exercices : 7.1–2, 7.23–24, 7.E1, 7.Q7, CP2.2, 14.R3
- **underfitting** (*sous-apprentissage*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s26] · exercices : 9.1, 9.12, 9.14, 9.E2, 9.Q1, 9.Q3
- **underflow** (*sous-dépassement, dépassement par le bas*) — enseignée : [ch. 0B][s34] ; [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s35] · exercices : 4.18 · réutilisée : ch. 13
- *undistributed middle* → voir [moyen terme non distribué](#m)
- *unfold* → voir [im2col](#i)
- **Unicode et UTF-8** — enseignée : [ch. B2, 202.2 · Unicode, UTF-8 et normalisation][s464] · exercices : B2.1, B2.18, B2.Q2–Q3
- *uniform distribution* → voir [loi uniforme](#l)
- *uniform manifold approximation and projection* → voir [t-SNE, UMAP](#t)
- *unigram* → voir [modèle unigramme, bigramme, trigramme](#m)
- **Unigram (tokenizer)** (*unigram language model*) — enseignée : [ch. B2, 202.5 · WordPiece, Unigram, SentencePiece ; la bibliothèque…][s470] · exercices : B2.13, B2.19, B2.30, B2.E2, B2.Q7
- *unigram language model* → voir [Unigram (tokenizer)](#u)
- *unit* → voir [neurone](#n), [unité](#u)
- *unit step* → voir [marche unité, marche de Heaviside, fonction signe](#m)
- *unit test* → voir [test unitaire](#t)
- **unité** (*unit*) — enseignée : [ch. 10, 10.3 · Les neurones artificiels][s242] · exercices : 0A.30, 2.8, 6.Q4, 10.8–10, 10.E4, 10.Q4, 17.Q5
- *unité linéaire à porte avec SiLU* → voir [SwiGLU](#s)
- *univariate* → voir [transformation univariée, multivariée](#t)
- *universal adversarial perturbation* → voir [perturbation universelle](#p)
- univoque → voir [courbe continue, lisse, univoque](#c)
- *unrolling* → voir [déroulement](#d)
- *unsupervised learning* → voir [apprentissage non supervisé](#a)
- *untargeted attack* → voir [attaque ciblée, non ciblée](#a)
- *update step* → voir [étape de mise à jour](#e)
- *upper confidence bound* → voir [UCB](#u)
- *upsampling* → voir [suréchantillonnage](#s)
- **UTF-8** — enseignée : [ch. B2, 202.2 · Unicode, UTF-8 et normalisation][s464] · exercices : B2.1, B2.18, B2.Q2–Q3
- *utility layer* → voir [couche utilitaire](#c)
- *uv* → voir [uv, pip-tools, ruff, pre-commit](#u)
- **uv, pip-tools, ruff, pre-commit** (*uv, pip-tools, ruff, pre-commit*) — enseignée : [ch. B7, pyproject.toml et installation éditable][s612] · exercices : B7.6, B7.8–9, B7.25, B7.E1, B7.Q1–Q2
- *uvicorn* → voir [FastAPI, uvicorn, pydantic](#f)
- uvicorn → voir [FastAPI, uvicorn, pydantic](#f)

## V

- *v-prediction* → voir [v-prédiction](#v)
- **v-prédiction** (*v-prediction*) — enseignée : [ch. B5][s567]
- **β-VAE** — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s394] · exercices : 25.36
- **VAE (autoencodeur variationnel)** (*variational autoencoder*) — enseignée : [ch. 25, 25.9.2 · La structure d'un VAE][s399] · exercices : 25.5, 25.11, 25.13, 25.28, 25.30–31, 25.38, 25.E3, 25.Q12, 25.R3 (+ 12) · réutilisée : ch. 27, B5
- valeur → voir [requête, clé, valeur](#r)
- **valeur absolue** (*absolute value*) — enseignée : [ch. 0B, 101.1.2 · Valeur absolue, inégalités, partie entière, fonction signe][s79] · exercices : 0B.2, 0B.33, 0B.36, 0B.Q2 · réutilisée : ch. 9
- **valeur de base** (*base value*) — enseignée : [ch. B6, Les features comme joueurs][s581] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- **valeur de Shapley** (*Shapley value*) — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s578] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- **valeur manquante** (*missing value*) — enseignée : [ch. 0A, 100.9.1 · `DataFrame` et `Series` : `read_csv`, `head`, `info`,…][s73] · exercices : 0A.32, 0A.Q12, 1.2
- **valeur prédictive négative (VPN)** (*negative predictive value, NPV*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s161] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11
- **valeur prédite** (*predicted value*) — enseignée : [ch. 7, 7.1 · Pourquoi ce chapitre ?][s206] · exercices : 7.Q1–Q2
- valeur propre → voir [vecteur propre, valeur propre](#v)
- valeur Q → voir [table Q, valeur Q](#t)
- **valeur sentinelle** (*sentinel value*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s264] · exercices : 12.11, 12.Q4
- **valeurs aberrantes** — enseignée : [ch. 12][s690]
- **valeurs de Shapley exactes** — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s578] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- **valeurs SHAP** (*SHapley Additive exPlanations*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s582] · exercices : B6.2, B6.7, B6.11, B6.15–17, B6.E1–E2, B6.Q5–Q6
- *valid* → voir [valide](#v)
- **validation croisée** (*cross-validation*) — enseignée : [ch. 8, 8.5 · La validation croisée][s43] · exercices : 8.2, 8.5, 8.8, 8.15, 8.19, 8.22–23, 8.25–26, 8.Q9 (+ 11) · approfondie : ch. 24 · réutilisée : ch. 13, 14, 24
- *validation croisée à $k$ plis* → voir [k-fold](#k)
- **validation croisée et recherche d'hyperparamètres** — enseignée : [ch. 24, 24.3.4 · La recherche d'hyperparamètres][s383] · exercices : 24.1, 24.7, 24.9, 24.18, 24.E1, 24.Q5
- **validation croisée imbriquée** (*nested cross-validation*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s218] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10
- **validation croisée k-fold** — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s218] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10
- *validation croisée par exclusion d'un exemple* → voir [leave-one-out](#l)
- *validation curve* → voir [courbe de validation](#c)
- **validation externe** (*external validation*) — enseignée : [ch. 8, 8.4 · Le jeu de validation][s129] · exercices : 8.1–3, 8.6, 8.10, 8.17–18, 8.Q7–Q8, 8.R1 (+ 3)
- **validation glissante** (*time series cross-validation, rolling origin*) — enseignée : [ch. 22][s360]
- *validation set* → voir [jeu de validation](#j)
- **valide** (*valid*) — enseignée : [ch. 11, 11.4 · La déduction][s254] · exercices : 11.3, 11.5, 11.15, 11.Q7, CP2.9
- **validité et solidité** — enseignée : [ch. 11][s689]
- *value* → voir [requête, clé, valeur](#r)
- *value function* → voir [fonction de valeur](#f), [fonction de valeur (RL)](#f)
- *value iteration* → voir [itération sur les valeurs](#i)
- *vanishing* → voir [gradient évanescent, explosif](#g)
- **variable** — enseignée : [ch. 0A, 100.2 · Valeurs, variables et types][s50] · exercices : 0A.40, 0A.Q3, 0B.27, 0B.29, 2.20, 12.E5, 25.19, 25.Q4, B6.9, B6.Q10 (+ 4) · approfondie : ch. 0B · réutilisée : ch. 2, 8, 12, 16, 20, 24, 25, 26, B3
- **variable aléatoire** (*random variable*) — enseignée : [ch. 0B, 101.7.3 · Variable aléatoire discrète et espérance][s103] · exercices : 0B.27–28, 0B.53, 0B.E5, 0B.Q12 · réutilisée : ch. 2, 26
- **variable aléatoire discrète, espérance, variance, écart-type, linéarité de l'espérance** — enseignée : [ch. 0B, 101.7.3 · Variable aléatoire discrète et espérance][s103] · exercices : 0B.27–28, 0B.53, 0B.E5, 0B.Q12, 2.4–5, 2.15, 2.Q7, 2.R3 (+ 26) · approfondie : ch. 2, 12, 16 · réutilisée : ch. 2, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 20, 25, 26, 27, B3, B5, B8
- **variable d'environnement** (*environment variable*) — enseignée : [ch. B7, Variables d'environnement et secrets][s608] · exercices : B7.23–25, B7.Q4
- **variable de confusion** (*confounder, confounding variable*) — enseignée : [ch. 2, 2.8.2 · La corrélation][s153] · exercices : 2.7–8, 2.10, 2.26–27, 2.32, 2.E4, 2.Q12
- **variable de substitution** (*proxy, proxy variable, indicateur indirect, substitut*) — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s136] ; [ch. 13][s274] ; [mini-projet 3][s279] · exercices : 2.1–2, 2.13, 2.16, 2.19, 2.E1, 2.Q1–Q2, 2.R2, B6.9
- **variable indicatrice** (*dummy variable*) — enseignée : [ch. 12, 12.3.1 · Le one-hot][s263] · exercices : 12.1, 12.20, 12.26, 12.33, 12.E5, 12.Q3
- *variable-length code* → voir [code à longueur variable, code adaptatif](#c)
- **variables et types, f-strings** — enseignée : [ch. 0A, 100.2 · Valeurs, variables et types][s50] · exercices : 0A.14, 0A.Q3
- variables latentes → voir [code, variables latentes](#c)
- *variance* → voir [variance, écart-type](#v), [variance (d'une famille de modèles)](#v)
- **variance (d'une famille de modèles)** (*variance*) — enseignée : [ch. 9, 9.6 · Biais et variance][s238] · exercices : 0B.27–28, 0B.Q12, 9.4, 9.8, 9.24, 9.29, 9.E1, 9.Q9, 9.R3 (+ 21) · approfondie : ch. 0B, 2, 12, 16 · réutilisée : ch. 2, 12, 13, 14, 16, 17, 25, B3, B5, B8
- **variance et ddof** — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s142] · exercices : 2.3–4, 2.15, 2.17–18, 2.29, 2.31, 2.Q5
- **variance expliquée (d'un critique)** (*explained variance*) — enseignée : [ch. B8, Le critique et sa variance expliquée][s647] ; [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s152] · exercices : 12.7–8, 12.22–23, 12.27, 12.Q9, B8.4, B8.18, B8.20, B8.Q7 (+ 1) · réutilisée : ch. 25
- *variance preserving diffusion* → voir [diffusion à variance préservée](#d)
- **variance, écart-type** (*variance, standard deviation*) — enseignée : [ch. 0B, 101.7.4 · Variance et écart-type][s105] · exercices : 0B.27–28, 0B.53, 0B.E5, 0B.Q12, 2.4, 2.15, 2.R3, 8.5, 9.Q9 (+ 22) · approfondie : ch. 2, 12, 16 · réutilisée : ch. 2, 7, 8, 9, 12, 13, 14, 16, 17, 25, 27, B3, B5, B8
- **variation totale** (*total variation*) — enseignée : [ch. 28, 28.4.5 · Réaliser le transfert][s437] · exercices : 28.4, 28.6, 28.20–21, 28.23–24, 28.26, 28.29, 28.Q9
- *variational autoencoder* → voir [VAE (autoencodeur variationnel)](#v)
- **variations, minimum et maximum (dérivée nulle)** — enseignée : [ch. 0B, 101.5.4 · Variations, minimum et maximum][s684] · exercices : 0B.24, 0B.48, 0B.Q10, 5.4
- **vecteur de contexte** (*context vector*) — enseignée : [ch. B3, Traduire avec deux RNN][s492] · exercices : B3.Q1, B3.R2
- **vecteur de support** (*support vector*) — enseignée : [ch. 13, 13.7 · Bilan : choisir un classifieur][s273] · exercices : 13.9, 13.11–12, 13.23, 13.33, 13.35–36, 13.E5, 13.Q12
- **vecteur propre, valeur propre** (*eigenvector, eigenvalue*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s152] · exercices : 12.7–8, 12.22–23, 12.27, 12.Q9
- **vecteurs : somme, produit par un scalaire, norme (L1, L2, L∞), distance euclidienne** — enseignée : [ch. 0B, 101.3.1 · Composantes, somme, produit par un scalaire][s87] · exercices : 0B.8, 0B.38–40, 0B.Q8, CP1.2, B2.Q12 · réutilisée : ch. 1, 2, 5, 7, 9, 10, 12, 13, 19, 25, 28, B2
- *vector database* → voir [base de données vectorielle](#b)
- **vectorisation** (*vectorization*) — enseignée : [ch. 0A, 100.8.3 · Calcul vectorisé et fonctions universelles][s71] · exercices : 0A.9, 0A.30, 0A.55, 0A.E3, 0A.Q10
- *vectorization* → voir [vectorisation](#v)
- *vectorized environments* → voir [environnements vectorisés](#e)
- *velocity field* → voir [champ de vitesses](#c)
- **vérification du gradient** (*gradient checking, gradient check*) — enseignée : [ch. 0B, 101.6.3 · Gradient et direction de plus grande pente][s41] ; [ch. 18, Au-delà du livre : vérifier un gradient][s109] · exercices : 0B.25, 0B.31, 0B.49–50, 0B.E1, 0B.Q11
- **vérité terrain** (*ground truth*) — enseignée : [ch. 3, 3.7.1 · Classer des échantillons][s159] · exercices : 3.15, 3.Q6, 7.Q1
- *view / copy* → voir [vue / copie](#v)
- *vision transformer* → voir [Vision Transformer (ViT)](#v)
- **Vision Transformer (ViT)** (*vision transformer*) — enseignée : [ch. 21, 21.9.1 · VGG16][s353] ; [ch. B3, 203.10 · Coût et passage à l'échelle : complexité quadratique,…][s354] · exercices : 21.8, 21.26–27, 21.E5, 21.Q10, 21.Q12, B3.8, B3.10, B3.27, B3.Q12 (+ 2)
- **visualisation de features** (*feature visualization*) — enseignée : [ch. 28, Au-delà du livre : la génération d'images aujourd'hui][s430]
- **visualisation de filtres et de cartes d'activation** — enseignée : [ch. 21, 21.9.3 · Regarder les filtres (2) : les cartes d'activation][s350] · exercices : 21.27, 21.Q10
- ViT → voir [Vision Transformer (ViT)](#v), [patch (ViT)](#p)
- *Viterbi algorithm* → voir [algorithme de Viterbi](#a)
- **vivacité trompeuse** (*misleading vividness*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s257] · exercices : 11.17–18, 11.E4, 11.Q10
- **vocabulaire (d'un modèle de langue)** (*vocabulary*) — enseignée : [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s203] · exercices : 22.1, 22.5, 22.8, 22.12, 22.20, 22.24–27, 22.Q12 (+ 11) · approfondie : ch. 1 · réutilisée : ch. 3, 24, B2, B3
- **vocabulaire caractères/mots, encodage** — enseignée : [ch. 22][s360] · exercices : B3.4, B3.E5, B3.Q7, B4.22, B5.21, B5.R1 · approfondie : ch. 12, B3 · réutilisée : ch. B5
- **vocabulaire du machine learning** — enseignée : [ch. 1, 1.1 · Le machine learning : extraire du sens des données][s128] · exercices : 1.6, 1.E1, 1.Q1
- *vocabulary* → voir [vocabulaire (d'un modèle de langue)](#v)
- *Voronoi cells* → voir [cellules de Voronoï](#c)
- **vote à la pluralité, vote pondéré** (*plurality voting, weighted voting*) — enseignée : [ch. 14, 14.3 · Le vote][s280] · exercices : 14.1, 14.11–14, 14.Q3
- **vote dur, vote souple** (*hard voting, soft voting*) — enseignée : [ch. 14, 14.3 · Le vote][s280] · exercices : 14.1, 14.11–14, 14.Q3
- vote pondéré → voir [vote à la pluralité, vote pondéré](#v)
- vote souple → voir [vote dur, vote souple](#v)
- VPN → voir [valeur prédictive négative (VPN)](#v)
- vrai négatif → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- **vrai positif, faux positif, faux négatif, vrai négatif** (*true positive, false positive, false negative, true negative, TP, FP, FN, TN*) — enseignée : [ch. 3, 3.7.2 · La matrice de confusion][s47] · exercices : 3.2, 3.15, 3.17, 3.Q7–Q8
- vraisemblance → voir [log-probabilité, vraisemblance](#l)
- **vraisemblance (d'une hypothèse)** (*likelihood*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s173] · exercices : 4.3, 4.10, 4.14, 4.23, 4.E3, 4.Q6–Q7, 9.27
- **vue / copie** (*view / copy*) — enseignée : [ch. 0A, 100.8.2 · Indexation, tranches, masques booléens ; vues et copies][s72] · exercices : 0A.7, 0A.28–29, 0A.56, 0A.58, 0A.65, 0A.Q10, 1.19, 20.26, 21.14 (+ 4) · réutilisée : ch. 2, 25, B6

## W

- **warmup** (*échauffement*) — enseignée : [ch. 19, Au-delà du livre : warmup et décroissance en cosinus][s316] ; [ch. B1, Warmup puis cosinus, groupe par groupe][s317] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 7) · réutilisée : ch. B3
- **warmup + cosinus** — enseignée : [ch. 19, Au-delà du livre : warmup et décroissance en cosinus][s316] · exercices : 19.19, B1.19, B3.20 · réutilisée : ch. B1, B3
- *warning* → voir [avertissement](#a)
- *Wasserstein distance* → voir [distance de Wasserstein (1D)](#d)
- **waterfall** (*graphique en cascade*) — enseignée : [ch. B6, Les trois graphiques à connaître][s584] · exercices : B6.2, B6.7, B6.11, B6.15–17, B6.E1–E2, B6.Q5–Q6
- *watermark* → voir [filigrane](#f)
- *weak learner* → voir [apprenant faible, apprenant fort](#a)
- **weight decay** (*décroissance des poids*) — enseignée : [ch. 9, Pénalité sur les poids, dropout, batchnorm][s20] · exercices : 9.3, 9.6, 9.17–19, 9.22–23, 9.28, 9.Q7–Q8 (+ 3) · approfondie : ch. 19, 20 · réutilisée : ch. 20
- *weight naming convention* → voir [convention AD (ou DA)](#c)
- *weight sharing* → voir [partage de poids](#p)
- *weight tying* → voir [poids liés](#p)
- *weight-decomposed low-rank adaptation* → voir [DoRA](#d)
- *weighted average* → voir [moyenne pondérée](#m), [moyenne macro, micro, pondérée](#m)
- *weighted sum* → voir [somme pondérée, pré-activation](#s)
- *weighted voting* → voir [vote à la pluralité, vote pondéré](#v)
- *weights* → voir [poids](#p)
- WGAN-GP → voir [pénalité de gradient (WGAN-GP)](#p)
- **whitening** (*blanchiment*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s152] · exercices : 12.7–8, 12.22–23, 12.27, 12.Q9
- *winner's curse* → voir [biais d'optimisme (du gagnant)](#b)
- *within-cluster sum of squares* → voir [inertie](#i)
- *without replacement* → voir [tirage avec remise, sans remise](#t)
- **word2vec** — enseignée : [ch. B2, 202.8 · Embeddings distributionnels : co-occurrences, PPMI, SVD,…][s479] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- **word2vec skip-gram** — enseignée : [ch. B2, word2vec : prédire plutôt que compter][s482] · exercices : B2.6–7, B2.22, B2.24–25, B2.Q10
- **WordPiece** — enseignée : [ch. B2, 202.5 · WordPiece, Unigram, SentencePiece ; la bibliothèque…][s470] · exercices : B2.13, B2.19, B2.30, B2.E2, B2.Q7
- **workbook : wb.check, wb.attempt, start_chapter.py, mylearn** — enseignée : [ch. 0A, 100.1.2 · Le workbook : setup, `wb.check`, `wb.attempt`,…][s1] · exercices : 0A.26, 0A.E1, 0A.Q2, 13.35, 15.29, 25.38 · approfondie : ch. 0B · réutilisée : ch. 0B, 3, 14, 15, 16, 17, 23, 24, 25, 26, 27, 28
- **worker (d'un `DataLoader`)** (*processus de chargement*) — enseignée : [ch. 24, 24.4.6 · L'augmentation de données][s387] · exercices : 24.8, 24.22–23, 24.31, 24.E2, 24.Q8
- *workflow* → voir [workflow, job, étape, action](#w)
- **workflow PyTorch complet** — enseignée : [ch. 23][s380]
- **workflow, job, étape, action** (*workflow, job, step, action*) — enseignée : [ch. B7, L'intégration continue][s618] · exercices : 0A.E5, 10.Q2, 20.E3, 23.R2, 26.Q3, B7.15, B7.23, B7.25, B7.E3, B7.Q7 (+ 6) · approfondie : ch. 23, 26 · réutilisée : ch. 24, B8
- *wrapper* → voir [adaptateur](#a)

## X

- Xavier → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- **XGBoost** — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s226] · exercices : 14.26, 14.E3
- **XOR (ou exclusif)** (*exclusive or*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s208] · exercices : 10.1, 10.3, 10.6, 10.11–12, 10.17, 10.Q5–Q6, 10.R2–R3 (+ 10)

## Y

- **YAML** — enseignée : [ch. B7, 207.3 · Configuration : dataclasses, YAML ou JSON, surcharges,…][s610] · exercices : B7.11, B7.25, B7.E1, B7.Q4
- **YaRN** (*yet another RoPE extension*) — enseignée : [ch. B3, Allonger le contexte][s518] · exercices : B3.4, B3.7, B3.11, B3.15, B3.25, B3.E5, B3.Q7
- *yet another RoPE extension* → voir [YaRN](#y)

## Z

- *z-score* → voir [z-score, standardiser](#z)
- **z-score, standardiser** (*z-score, standardize*) — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s142] · exercices : 0A.52, 2.3–4, 2.15, 2.17–18, 2.29, 2.31, 2.Q5, 12.Q5 (+ 5) · réutilisée : ch. 7, 9, 12
- *zero terminal SNR* → voir [SNR final nul](#s)
- *zero-initialized* → voir [adaLN-Zero](#a)
- **zero-padding** (*remplissage de zéros*) — enseignée : [ch. 20, 20.4.5 · La convolution (aperçu)][s23] · exercices : 20.6, 20.20, 20.Q7
- *zero-shot* → voir [zéro-shot](#z)
- **zéro-shot** (*zero-shot*) — enseignée : [ch. B1, 201.8 · Modèles de fondation : auto-supervisé (DINOv2),…][s455] · exercices : B1.5, B1.15, B1.24–25, B1.E4, B1.Q10–Q11, B4.15, B4.28, B4.Q6
- **zéro-shot CLIP** — enseignée : [ch. B1, 201.8 · Modèles de fondation : auto-supervisé (DINOv2),…][s455] · exercices : B1.5, B1.15, B1.24–25, B1.E4, B1.Q10–Q11
- *zero-sum game* → voir [jeu à somme nulle](#j)

<a id="symboles"></a>

## 0-9 et symboles

- *1D convolution* → voir [convolution 1D](#c)
- **3CosAdd** — enseignée : [ch. B2, Les analogies][s484] · exercices : B2.2, B2.11, B2.23–24, B2.26–27, B2.E4, B2.Q11–Q12
- *68-95-99.7 rule* → voir [règle 68-95-99,7](#r)

[s1]: ../chapitres/ch00a_python/01_fiche.md#10012--le-workbook--setup-wbcheck-wbattempt-start_chapterpy-mon_travail
[s2]: ../chapitres/ch20_deep_learning/01_fiche.md#au-delà-du-livre--pytorch-en-pratique
[s3]: ../chapitres/ch23_pytorch_1/01_fiche.md#2352--charger-les-données-
[s4]: ../chapitres/ch00a_python/01_fiche.md#10064--sérialiser--json-et-pickle
[s5]: ../chapitres/ch00a_python/01_fiche.md#10094--compter-et-regrouper--value_counts-groupby-tri
[s6]: ../chapitres/ch00a_python/01_fiche.md#10087--aléatoire-reproductible-et-mini-batches
[s7]: ../chapitres/ch19_optimiseurs/01_fiche.md#1941--la-descente-de-gradient-par-batch
[s8]: ../chapitres/ch19_optimiseurs/01_fiche.md#1943--la-descente-par-mini-batches
[s9]: ../chapitres/ch08_train_test/01_fiche.md#82--la-boucle-dentraînement-
[s10]: ../chapitres/ch18_backprop/01_fiche.md#18111--explorer-le-learning-rate-
[s11]: ../chapitres/ch00b_maths/01_fiche.md#10117--dénombrement--principe-multiplicatif-factorielle-coefficient-binomial
[s12]: ../chapitres/ch01_introduction/01_fiche.md#122--la-version-pour-ordinateur--échantillons-features-labels-
[s13]: ../chapitres/ch05_courbes/01_fiche.md#suivre-la-pente-jusquà-un-extremum
[s14]: ../chapitres/ch00b_maths/01_fiche.md#10113--notations-σ-et-π
[s15]: ../chapitres/ch12_preparation/01_fiche.md#1210--transformations-et-validation-croisée-
[s16]: ../chapitres/ch15_sklearn/01_fiche.md#1595--les-pipelines-
[s17]: ../chapitres/ch15_sklearn/01_fiche.md#152--scikit-learn-numpy-et-scipy-
[s18]: ../chapitres/ch01_introduction/01_fiche.md#panorama-2026--où-en-est-lia-
[s19]: ../chapitres/ch12_preparation/01_fiche.md#1272--standardisation-et-pca-pour-des-images-
[s20]: ../chapitres/ch09_overfitting/01_fiche.md#pénalité-sur-les-poids-dropout-batchnorm
[s21]: ../chapitres/ch20_deep_learning/01_fiche.md#2043--le-dropout-
[s22]: ../chapitres/ch20_deep_learning/01_fiche.md#2046--le-pooling-aperçu
[s23]: ../chapitres/ch20_deep_learning/01_fiche.md#2045--la-convolution-aperçu
[s24]: ../chapitres/ch21_cnn/01_fiche.md#2134--le-padding-
[s25]: ../chapitres/ch21_cnn/01_fiche.md#2125--le-noyau
[s26]: ../chapitres/ch09_overfitting/01_fiche.md#92--overfitting-et-underfitting-
[s27]: ../chapitres/ch01_introduction/01_fiche.md#17--deep-learning-
[s28]: ../chapitres/ch00a_python/01_fiche.md#1001--prise-en-main--notebooks-terminal-et-workbook
[s29]: ../chapitres/ch01_introduction/01_fiche.md#123--généraliser--le-jeu-de-test-
[s30]: ../chapitres/ch03_probabilites/01_fiche.md#375--accuracy-
[s31]: ../chapitres/ch03_probabilites/01_fiche.md#376--precision-
[s32]: ../chapitres/ch03_probabilites/01_fiche.md#377--recall-
[s33]: ../chapitres/ch03_probabilites/01_fiche.md#au-delà-du-livre--plusieurs-classes-et-moyennes-macro-micro-pondérée
[s34]: ../chapitres/ch00b_maths/01_fiche.md
[s35]: ../chapitres/ch04_bayes/01_fiche.md#au-delà-du-livre-1--log-probabilités-et-underflow
[s36]: ../chapitres/ch04_bayes/01_fiche.md#461--la-boucle-posterior--prior-
[s37]: ../chapitres/ch04_bayes/01_fiche.md#au-delà-du-livre-3--map-et-intervalle-de-crédibilité
[s38]: ../chapitres/ch01_introduction/01_fiche.md#124--modèle-paramètres-capacité-hyperparamètres-
[s39]: ../chapitres/ch00b_maths/01_fiche.md#10114--moyenne-moyenne-pondérée-somme-pondérée-moyenne-mobile
[s40]: ../chapitres/ch10_neurones/01_fiche.md#1033--les-neurones-modernes-
[s41]: ../chapitres/ch00b_maths/01_fiche.md#10163--gradient-et-direction-de-plus-grande-pente
[s42]: ../chapitres/ch18_backprop/01_fiche.md#189--la-rétropropagation-en-action-
[s43]: ../chapitres/ch08_train_test/01_fiche.md#85--la-validation-croisée-
[s44]: ../chapitres/ch01_introduction/01_fiche.md#13--apprentissage-supervisé
[s45]: ../chapitres/ch01_introduction/01_fiche.md#14--apprentissage-non-supervisé
[s46]: ../chapitres/ch01_introduction/01_fiche.md#16--apprentissage-par-renforcement
[s47]: ../chapitres/ch03_probabilites/01_fiche.md#372--la-matrice-de-confusion-
[s48]: ../chapitres/ch00a_python/01_fiche.md#10011--cellules-noyau-ordre-dexécution-et--tout-exécuter-
[s49]: ../chapitres/ch00a_python/01_fiche.md#10013--le-terminal--dossiers-chemins-lancer-python-pytest-et-git
[s50]: ../chapitres/ch00a_python/01_fiche.md#1002--valeurs-variables-et-types
[s51]: ../chapitres/ch00a_python/01_fiche.md#10022--chaînes-de-caractères-et-f-strings
[s52]: ../chapitres/ch00a_python/01_fiche.md
[s53]: ../chapitres/ch00a_python/01_fiche.md#10031--listes--indices-tranches-méthodes-mutabilité
[s54]: ../chapitres/ch00a_python/01_fiche.md#10033--dictionnaires
[s55]: ../chapitres/ch00a_python/01_fiche.md#10034--ensembles
[s56]: ../chapitres/ch14_ensembles/01_fiche.md#142--les-ensembles-
[s57]: ../chapitres/ch00a_python/01_fiche.md#10043--compréhensions-de-listes-de-dictionnaires-et-densembles
[s58]: ../chapitres/ch00a_python/01_fiche.md#1005--fonctions
[s59]: ../chapitres/ch00a_python/01_fiche.md#10051--définir-une-fonction--paramètres-valeurs-par-défaut-return-portée
[s60]: ../chapitres/ch00a_python/01_fiche.md#10054--fermetures-closures
[s61]: ../chapitres/ch00a_python/01_fiche.md#10055--docstrings-au-format-numpy-et-annotations-de-type
[s62]: ../chapitres/ch00a_python/01_fiche.md#10062--exceptions--lire-un-traceback-raise-tryexcept
[s63]: ../chapitres/ch00a_python/01_fiche.md#10056--fonctions-récursives
[s64]: ../chapitres/ch00a_python/01_fiche.md#1006--modules-erreurs-et-fichiers
[s65]: ../chapitres/ch00a_python/01_fiche.md#10073--héritage-et-super
[s66]: ../chapitres/ch00a_python/01_fiche.md#10074--itérables-et-générateurs-yield
[s67]: ../chapitres/ch00a_python/01_fiche.md#10065--expressions-régulières-re--les-bases
[s68]: ../chapitres/ch00a_python/01_fiche.md#10081--créer-des-arrays--dtype-shape-ndim
[s69]: ../chapitres/ch00a_python/01_fiche.md#10084--broadcasting
[s70]: ../chapitres/ch00a_python/01_fiche.md#10085--réductions-par-axe-tri-argmax-unique
[s71]: ../chapitres/ch00a_python/01_fiche.md#10083--calcul-vectorisé-et-fonctions-universelles
[s72]: ../chapitres/ch00a_python/01_fiche.md#10082--indexation-tranches-masques-booléens--vues-et-copies
[s73]: ../chapitres/ch00a_python/01_fiche.md#10091--dataframe-et-series--read_csv-head-info-describe
[s74]: ../chapitres/b2_tokenisation/01_fiche.md#2027--embeddings--de-one-hot-à-nnembedding-entraînement-et-poids-liés-
[s75]: ../chapitres/ch00a_python/01_fiche.md#100111--git--status-add-commit-push-pull--messages-de-commit--gitignore
[s76]: ../chapitres/ch00a_python/01_fiche.md#100112--git--branches-aperçu
[s77]: ../chapitres/ch00a_python/01_fiche.md#100115--mylearn--stubs-référence-et-tests-fondés-sur-un-oracle
[s78]: ../chapitres/ch00b_maths/01_fiche.md#10111--puissances-racines-notation-scientifique-ordres-de-grandeur
[s79]: ../chapitres/ch00b_maths/01_fiche.md#10112--valeur-absolue-inégalités-partie-entière-fonction-signe
[s80]: ../chapitres/ch00b_maths/01_fiche.md#10115--suites-géométriques-et-leur-somme
[s81]: ../chapitres/ch00b_maths/01_fiche.md#10116--ensembles--union-intersection-complémentaire-cardinal
[s82]: ../chapitres/ch00b_maths/01_fiche.md#10124--logarithmes--ln-log₂-log₁₀-changement-de-base
[s83]: ../chapitres/ch00b_maths/01_fiche.md#10125--sigmoïde-et-tangente-hyperbolique--définitions-et-allure
[s84]: ../chapitres/ch13_classifieurs/01_fiche.md#au-delà-du-livre--la-régression-logistique-le-classifieur-paramétrique-de-référence
[s85]: ../chapitres/ch17_activations/01_fiche.md#176--les-fonctions-lisses-
[s86]: ../chapitres/ch00b_maths/01_fiche.md#10126--cosinus-aperçu
[s87]: ../chapitres/ch00b_maths/01_fiche.md#10131--composantes-somme-produit-par-un-scalaire
[s88]: ../chapitres/ch00b_maths/01_fiche.md#10132--norme-et-distance
[s89]: ../chapitres/ch00b_maths/01_fiche.md#10133--produit-scalaire-angle-similarité-cosinus
[s90]: ../chapitres/b2_tokenisation/01_fiche.md#le-cosinus
[s91]: ../chapitres/ch00b_maths/01_fiche.md#10134--produit-élément-par-élément-hadamard
[s92]: ../chapitres/ch00b_maths/01_fiche.md#10141--forme-éléments-transposée--un-dataset-est-une-matrice
[s93]: ../chapitres/ch00b_maths/01_fiche.md#10143--produit-matriciel-et-vérification-des-formes
[s94]: ../chapitres/ch00b_maths/01_fiche.md#10144--identité-et-idée-de-linverse
[s95]: ../chapitres/ch00b_maths/01_fiche.md#10151--taux-daccroissement-tangente-dérivée
[s96]: ../chapitres/ch00b_maths/01_fiche.md#1015--dérivées
[s97]: ../chapitres/ch00b_maths/01_fiche.md#10153--règle-de-la-chaîne
[s98]: ../chapitres/ch00b_maths/01_fiche.md#10162--dérivées-partielles
[s99]: ../chapitres/ch00b_maths/01_fiche.md#10161--fonctions-de-deux-variables-et-lignes-de-niveau
[s100]: ../chapitres/ch00b_maths/01_fiche.md#10164--règle-de-la-chaîne-à-plusieurs-variables--la-somme-sur-les-chemins
[s101]: ../chapitres/ch00b_maths/01_fiche.md#10171--expérience-aléatoire-événement-probabilité-complémentaire
[s102]: ../chapitres/ch00b_maths/01_fiche.md#10172--indépendance
[s103]: ../chapitres/ch00b_maths/01_fiche.md#10173--variable-aléatoire-discrète-et-espérance
[s104]: ../chapitres/ch02_stats/01_fiche.md#235--lespérance
[s105]: ../chapitres/ch00b_maths/01_fiche.md#10174--variance-et-écart-type
[s106]: ../chapitres/ch00b_maths/01_fiche.md#10175--simuler-pour-estimer--fréquences-et-loi-des-grands-nombres
[s107]: ../chapitres/ch13_classifieurs/01_fiche.md#131--pourquoi-ce-chapitre--
[s108]: ../chapitres/b8_rl_moderne/01_fiche.md#2083--gradient-de-politique--politique-paramétrée-reinforce-baseline-variance-
[s109]: ../chapitres/ch18_backprop/01_fiche.md#au-delà-du-livre--vérifier-un-gradient
[s110]: ../chapitres/ch01_introduction/01_fiche.md#112--les-systèmes-experts-et-leurs-limites-
[s111]: ../chapitres/ch01_introduction/01_fiche.md
[s112]: ../chapitres/ch01_introduction/01_fiche.md#131--classification-
[s113]: ../chapitres/ch01_introduction/01_fiche.md#132--régression
[s114]: ../chapitres/ch01_introduction/01_fiche.md#141--clustering
[s115]: ../chapitres/ch01_introduction/01_fiche.md#142--débruitage
[s116]: ../chapitres/ch01_introduction/01_fiche.md#143--réduction-de-dimension
[s117]: ../chapitres/ch12_preparation/01_fiche.md#127--la-réduction-de-dimension-
[s118]: ../chapitres/ch01_introduction/01_fiche.md#15--générateurs
[s119]: ../chapitres/ch27_gan/01_fiche.md#2742--le-générateur-
[s120]: ../chapitres/ch25_autoencodeurs/01_fiche.md#253--le-plus-simple-des-autoencodeurs-
[s121]: ../chapitres/ch11_raisonnement/01_fiche.md#au-delà-du-livre--le-bandit-manchot
[s122]: ../chapitres/ch08_train_test/01_fiche.md#83--le-jeu-de-test-
[s123]: ../chapitres/ch12_preparation/01_fiche.md#122--la-règle-dor--apprendre-la-transformation-sur-lentraînement-
[s124]: ../chapitres/ch01_introduction/01_fiche.md#les-quatre-fils-rouges-du-workbook
[s125]: ../chapitres/ch29_datasets/01_fiche.md#au-delà-du-livre--documenter-et-télécharger
[s126]: ../chapitres/b8_rl_moderne/01_fiche.md#2087--rl-et-modèles-de-langue--rlhf-modèle-de-récompense-dpo-grpo-piratage-de-la-récompense-
[s127]: ../chapitres/b1_transfer/01_fiche.md#quest-ce-quun-modèle-de-fondation-
[s128]: ../chapitres/ch01_introduction/01_fiche.md#11--le-machine-learning--extraire-du-sens-des-données-
[s129]: ../chapitres/ch08_train_test/01_fiche.md#84--le-jeu-de-validation-
[s130]: ../chapitres/ch13_classifieurs/01_fiche.md#135--les-arbres-de-décision-
[s131]: ../chapitres/ch07_classification/01_fiche.md#sans-labels--k-means
[s132]: ../chapitres/b4_llm/01_fiche.md#20410--usage-responsable--hallucinations-injection-de-prompt-données-licences-coûts-ai-act-
[s133]: ../chapitres/ch02_stats/01_fiche.md#résumer-une-liste-de-nombres--moyenne-médiane-mode
[s134]: ../chapitres/ch02_stats/01_fiche.md#des-comptages-à-une-distribution-de-probabilité
[s135]: ../chapitres/ch02_stats/01_fiche.md
[s136]: ../chapitres/ch02_stats/01_fiche.md#variable-aléatoire-tirage-loi-discrète-et-densité
[s137]: ../chapitres/ch02_stats/01_fiche.md#221--nombres-aléatoires-en-pratique
[s138]: ../chapitres/ch02_stats/01_fiche.md#231--la-loi-uniforme
[s139]: ../chapitres/ch02_stats/01_fiche.md#232--la-loi-normale-
[s140]: ../chapitres/ch02_stats/01_fiche.md#233--la-loi-de-bernoulli
[s141]: ../chapitres/ch02_stats/01_fiche.md#234--la-loi-catégorielle-multinoulli
[s142]: ../chapitres/ch02_stats/01_fiche.md#mesurer-la-dispersion-de-données--variance-écart-type-ddof
[s143]: ../chapitres/ch02_stats/01_fiche.md#241--des-variables-iid
[s144]: ../chapitres/ch02_stats/01_fiche.md#25--tirages-avec-et-sans-remise-
[s145]: ../chapitres/ch02_stats/01_fiche.md#26--le-bootstrap-
[s146]: ../chapitres/ch14_ensembles/01_fiche.md#144--le-bagging-
[s147]: ../chapitres/ch02_stats/01_fiche.md#27--espaces-de-grande-dimension
[s148]: ../chapitres/ch07_classification/01_fiche.md#76--la-malédiction-de-la-dimension-
[s149]: ../chapitres/ch02_stats/01_fiche.md#29--le-quartet-danscombe
[s150]: ../chapitres/ch02_stats/01_fiche.md#28--covariance-et-corrélation-
[s151]: ../chapitres/ch02_stats/01_fiche.md#matrices-de-covariance-et-de-corrélation
[s152]: ../chapitres/ch12_preparation/01_fiche.md#1271--lanalyse-en-composantes-principales-pca-
[s153]: ../chapitres/ch02_stats/01_fiche.md#282--la-corrélation-
[s154]: ../chapitres/ch03_probabilites/01_fiche.md#34--probabilité-conditionnelle
[s155]: ../chapitres/ch03_probabilites/01_fiche.md#35--probabilité-jointe
[s156]: ../chapitres/ch03_probabilites/01_fiche.md#33--probabilité-simple
[s157]: ../chapitres/ch04_bayes/01_fiche.md#441--la-règle-de-bayes-
[s158]: ../chapitres/ch03_probabilites/01_fiche.md#36--probabilité-marginale-et-tables-de-contingence
[s159]: ../chapitres/ch03_probabilites/01_fiche.md#371--classer-des-échantillons-
[s160]: ../chapitres/ch07_classification/01_fiche.md#721--classification-binaire-en-2d-
[s161]: ../chapitres/ch03_probabilites/01_fiche.md#379--les-autres-mesures-
[s162]: ../chapitres/ch03_probabilites/01_fiche.md#38--appliquer-la-matrice-de-confusion--le-piège-de-la-prévalence-
[s163]: ../chapitres/b6_explicabilite/01_fiche.md#limpossibilité
[s164]: ../chapitres/ch03_probabilites/01_fiche.md#3711--le-score-f1-
[s165]: ../chapitres/ch07_classification/01_fiche.md#741--un-contre-tous-ovr-
[s166]: ../chapitres/ch15_sklearn/01_fiche.md#au-delà-du-livre--seuil-de-décision-calibration-et-classes-déséquilibrées
[s167]: ../chapitres/ch03_probabilites/01_fiche.md#au-delà-du-livre-1--la-courbe-roc-et-lauc
[s168]: ../chapitres/ch03_probabilites/01_fiche.md#au-delà-du-livre-2--la-courbe-precision-recall-et-laverage-precision
[s169]: ../chapitres/ch03_probabilites/01_fiche.md#au-delà-du-livre-3--la-calibration-des-probabilités
[s170]: ../chapitres/ch03_probabilites/01_fiche.md
[s171]: ../chapitres/ch04_bayes/01_fiche.md#42--fréquentiste-et-bayésien--deux-façons-de-penser-les-probabilités
[s172]: ../chapitres/ch04_bayes/01_fiche.md#47--plusieurs-hypothèses-
[s173]: ../chapitres/ch04_bayes/01_fiche.md#442--retrouver-la-règle-et-nommer-ses-termes-
[s174]: ../chapitres/ch04_bayes/01_fiche.md#43--lancer-des-pièces
[s175]: ../chapitres/ch04_bayes/01_fiche.md#46--répéter-la-règle-de-bayes-
[s176]: ../chapitres/ch04_bayes/01_fiche.md
[s177]: ../chapitres/ch13_classifieurs/01_fiche.md#136--naïve-bayes-
[s178]: ../chapitres/ch04_bayes/01_fiche.md#au-delà-du-livre-2--la-limite-continue-la-loi-beta
[s179]: ../chapitres/ch05_courbes/01_fiche.md#52--introduction--fonctions-courbes-et-surfaces
[s180]: ../chapitres/ch05_courbes/01_fiche.md#extrema-globaux-et-locaux
[s181]: ../chapitres/ch05_courbes/01_fiche.md#tangente-et-dérivée
[s182]: ../chapitres/ch05_courbes/01_fiche.md#au-delà-du-livre-1--calculer-une-dérivée-sur-ordinateur
[s183]: ../chapitres/ch05_courbes/01_fiche.md#au-delà-du-livre-2--les-arrondis-et-le-choix-du-pas
[s184]: ../chapitres/ch05_courbes/01_fiche.md
[s185]: ../chapitres/ch05_courbes/01_fiche.md#les-points-où-le-gradient-sannule
[s186]: ../chapitres/ch05_courbes/01_fiche.md#54--le-gradient-
[s187]: ../chapitres/ch05_courbes/01_fiche.md#descendre-une-surface-pas-à-pas
[s188]: ../chapitres/ch05_courbes/01_fiche.md#au-delà-du-livre-3--la-différentiation-automatique
[s189]: ../chapitres/ch06_information/01_fiche.md#information--un-mot-deux-sens
[s190]: ../chapitres/ch06_information/01_fiche.md#63--le-bit-une-unité-
[s191]: ../chapitres/ch06_information/01_fiche.md#au-delà-du-livre-1--bits-nats-log-loss-et-perplexité
[s192]: ../chapitres/ch06_information/01_fiche.md#le-contexte
[s193]: ../chapitres/ch06_information/01_fiche.md#65--la-taille-dun-événement-
[s194]: ../chapitres/ch06_information/01_fiche.md#66--les-codes-adaptatifs-
[s195]: ../chapitres/ch06_information/01_fiche.md#le-code-de-huffman
[s196]: ../chapitres/ch06_information/01_fiche.md#67--lentropie-
[s197]: ../chapitres/ch13_classifieurs/01_fiche.md#1352--choisir-les-splits-
[s198]: ../chapitres/ch06_information/01_fiche.md#68--la-cross-entropy-
[s199]: ../chapitres/ch06_information/01_fiche.md#69--la-divergence-kl-
[s200]: ../chapitres/ch06_information/01_fiche.md#mélanger-les-codes
[s201]: ../chapitres/b2_tokenisation/01_fiche.md#deux-rapports
[s202]: ../chapitres/ch06_information/01_fiche.md#au-delà-du-livre-2--le-contexte-réduit-la-surprise
[s203]: ../chapitres/ch22_rnn/01_fiche.md#229--un-exemple--générer-du-holmes-caractère-par-caractère-
[s204]: ../chapitres/ch28_creatif/01_fiche.md#285--générer-la-suite-du-livre
[s205]: ../chapitres/ch17_activations/01_fiche.md#178--la-softmax-
[s206]: ../chapitres/ch07_classification/01_fiche.md#71--pourquoi-ce-chapitre-
[s207]: ../chapitres/ch07_classification/01_fiche.md#73--plusieurs-classes-plusieurs-dimensions-
[s208]: ../chapitres/ch10_neurones/01_fiche.md#1031--le-perceptron-
[s209]: ../chapitres/ch07_classification/01_fiche.md#742--un-contre-un-ovo-
[s210]: ../chapitres/ch07_classification/01_fiche.md#avec-des-labels--le-centroïde-le-plus-proche
[s211]: ../chapitres/ch13_classifieurs/01_fiche.md#133--les-k-plus-proches-voisins-
[s212]: ../chapitres/ch07_classification/01_fiche.md#le-départ-compte--minima-locaux-et-k-means
[s213]: ../chapitres/ch07_classification/01_fiche.md#combien-de-clusters-
[s214]: ../chapitres/ch07_classification/01_fiche.md#761--bizarreries-de-la-grande-dimension
[s215]: ../chapitres/ch07_classification/01_fiche.md#quand-k-means-échoue-et-le-clustering-par-densité
[s216]: ../chapitres/ch07_classification/01_fiche.md
[s217]: ../chapitres/b4_llm/01_fiche.md#les-bases-de-données-vectorielles
[s218]: ../chapitres/ch08_train_test/01_fiche.md#851--la-validation-croisée-à-k-folds-
[s219]: ../chapitres/ch08_train_test/01_fiche.md#821--pourquoi-le-score-dentraînement-trompe-
[s220]: ../chapitres/ch08_train_test/01_fiche.md
[s221]: ../chapitres/ch08_train_test/01_fiche.md#données-dépendantes--groupes-et-séries-temporelles-au-delà-du-livre
[s222]: ../chapitres/ch08_train_test/01_fiche.md#86--utiliser-les-résultats-du-test-
[s223]: ../chapitres/ch23_pytorch_1/01_fiche.md#2395--sauvegarder-le-prétraitement-
[s224]: ../chapitres/b6_explicabilite/01_fiche.md#la-fiche-modèle
[s225]: ../chapitres/ch09_overfitting/01_fiche.md#94--learly-stopping-
[s226]: ../chapitres/ch14_ensembles/01_fiche.md#au-delà-du-livre--le-gradient-boosting
[s227]: ../chapitres/ch20_deep_learning/01_fiche.md#au-delà-du-livre--régulariser-un-réseau
[s228]: ../chapitres/ch09_overfitting/01_fiche.md#le-point-isolé
[s229]: ../chapitres/ch09_overfitting/01_fiche.md#95--la-régularisation-
[s230]: ../chapitres/ch09_overfitting/01_fiche.md#la-pénalité-l2--ridge
[s231]: ../chapitres/ch09_overfitting/01_fiche.md#la-pénalité-l1--lasso-et-les-zéros-exacts
[s232]: ../chapitres/ch09_overfitting/01_fiche.md#les-moindres-carrés-au-delà-du-livre
[s233]: ../chapitres/ch09_overfitting/01_fiche.md#97--ajuster-une-droite-avec-la-règle-de-bayes
[s234]: ../chapitres/b8_rl_moderne/01_fiche.md#2082--dqn--réseau-q-mémoire-de-rejeu-réseau-cible-loss-de-huber-double-dqn-
[s235]: ../chapitres/ch09_overfitting/01_fiche.md
[s236]: ../chapitres/ch09_overfitting/01_fiche.md#ajuster-une-courbe--complexité-et-capacité
[s237]: ../chapitres/ch09_overfitting/01_fiche.md#courbes-de-validation-et-courbes-dapprentissage-au-delà-du-livre
[s238]: ../chapitres/ch09_overfitting/01_fiche.md#96--biais-et-variance-
[s239]: ../chapitres/ch09_overfitting/01_fiche.md#964--le-compromis-biais-variance-et-ses-limites
[s240]: ../chapitres/ch20_deep_learning/01_fiche.md#2044--la-batchnorm-
[s241]: ../chapitres/ch10_neurones/01_fiche.md#102--les-vrais-neurones-
[s242]: ../chapitres/ch10_neurones/01_fiche.md#103--les-neurones-artificiels-
[s243]: ../chapitres/ch10_neurones/01_fiche.md#1032--lhistoire-du-perceptron-
[s244]: ../chapitres/ch13_classifieurs/01_fiche.md#134--les-machines-à-vecteurs-de-support-
[s245]: ../chapitres/ch10_neurones/01_fiche.md
[s246]: ../chapitres/ch18_backprop/01_fiche.md#1821--apprendre-en-tâtonnant-
[s247]: ../chapitres/ch11_raisonnement/01_fiche.md#1121--la-représentation
[s248]: ../chapitres/ch11_raisonnement/01_fiche.md#1122--lévaluation
[s249]: ../chapitres/ch11_raisonnement/01_fiche.md#1123--loptimisation
[s250]: ../chapitres/ch19_optimiseurs/01_fiche.md#196--choisir-un-optimiseur-
[s251]: ../chapitres/ch13_classifieurs/01_fiche.md#132--paramétrique-ou-non-paramétrique--
[s252]: ../chapitres/ch11_raisonnement/01_fiche.md#113--déduction-et-induction-
[s253]: ../chapitres/ch11_raisonnement/01_fiche.md#1161--sherlock-holmes--maître-de-la-déduction-
[s254]: ../chapitres/ch11_raisonnement/01_fiche.md#114--la-déduction-
[s255]: ../chapitres/ch11_raisonnement/01_fiche.md#1141--les-sophismes-syllogistiques
[s256]: ../chapitres/ch11_raisonnement/01_fiche.md#115--linduction-
[s257]: ../chapitres/ch11_raisonnement/01_fiche.md#1152--les-sophismes-inductifs-
[s258]: ../chapitres/ch11_raisonnement/01_fiche.md#116--raisonnement-combiné-
[s259]: ../chapitres/ch11_raisonnement/01_fiche.md#117--le-conditionnement-opérant-
[s260]: ../chapitres/ch26_rl/01_fiche.md#2662--la-politique-du-q-learning-
[s261]: ../chapitres/ch12_preparation/01_fiche.md#121--pourquoi-ce-chapitre-
[s262]: ../chapitres/ch12_preparation/01_fiche.md#123--les-types-de-données-
[s263]: ../chapitres/ch12_preparation/01_fiche.md#1231--le-one-hot-
[s264]: ../chapitres/ch12_preparation/01_fiche.md#124--le-nettoyage-de-base-
[s265]: ../chapitres/ch12_preparation/01_fiche.md#1251--la-normalisation-min-max-
[s266]: ../chapitres/ch12_preparation/01_fiche.md#1252--la-standardisation-
[s267]: ../chapitres/ch12_preparation/01_fiche.md#1253--retenir-la-transformation-
[s268]: ../chapitres/ch12_preparation/01_fiche.md#1254--univariée-ou-multivariée-
[s269]: ../chapitres/ch12_preparation/01_fiche.md#129--par-échantillon-par-feature-par-élément
[s270]: ../chapitres/ch12_preparation/01_fiche.md#126--la-sélection-de-features-
[s271]: ../chapitres/ch12_preparation/01_fiche.md#au-delà-du-livre--voir-des-données-en-grande-dimension
[s272]: ../chapitres/ch12_preparation/01_fiche.md#au-delà-du-livre--assembler-les-données-avec-pandas-et-sql
[s273]: ../chapitres/ch13_classifieurs/01_fiche.md#137--bilan--choisir-un-classifieur-
[s274]: ../chapitres/ch13_classifieurs/01_fiche.md
[s275]: ../chapitres/ch14_ensembles/01_fiche.md#147--le-boosting-
[s276]: ../chapitres/ch13_classifieurs/01_fiche.md#1353--limiter-loverfitting-
[s277]: ../chapitres/ch04_bayes/01_fiche.md#bayes-dans-le-ml-daujourdhui
[s278]: ../projets/partie_1_detecteur_langue/README.md
[s279]: ../projets/partie_3_revenus/README.md
[s280]: ../chapitres/ch14_ensembles/01_fiche.md#143--le-vote-
[s281]: ../chapitres/ch14_ensembles/01_fiche.md
[s282]: ../chapitres/ch14_ensembles/01_fiche.md#145--les-forêts-aléatoires-
[s283]: ../chapitres/ch14_ensembles/01_fiche.md#146--extratrees-
[s284]: ../chapitres/b6_explicabilite/01_fiche.md#2062--explications-globales--coefficients-arbres-importance-par-permutation-dépendance-partielle-et-ice-
[s285]: ../chapitres/ch15_sklearn/01_fiche.md#154--les-estimateurs-
[s286]: ../chapitres/ch15_sklearn/01_fiche.md#1542--apprendre-avec-fit-
[s287]: ../chapitres/ch15_sklearn/01_fiche.md#1593--la-recherche-en-grille-
[s288]: ../chapitres/ch15_sklearn/01_fiche.md#1594--la-recherche-aléatoire
[s289]: ../chapitres/ch15_sklearn/01_fiche.md#1592--chercher-des-hyperparamètres-
[s290]: ../chapitres/ch15_sklearn/01_fiche.md#au-delà-du-livre--brancher-ton-propre-estimateur
[s291]: ../chapitres/ch15_sklearn/01_fiche.md#au-delà-du-livre--les-données-hétérogènes
[s292]: ../chapitres/ch15_sklearn/01_fiche.md#au-delà-du-livre--sauvegarder-un-modèle-et-ses-versions
[s293]: ../chapitres/ch16_feedforward/01_fiche.md#161--pourquoi-ce-chapitre--
[s294]: ../chapitres/ch16_feedforward/01_fiche.md#162--les-graphes-de-réseaux-de-neurones-
[s295]: ../chapitres/ch16_feedforward/01_fiche.md#163--flux-synchrone-et-asynchrone
[s296]: ../chapitres/ch16_feedforward/01_fiche.md#1641--linitialisation-
[s297]: ../chapitres/ch16_feedforward/01_fiche.md
[s298]: ../chapitres/ch16_feedforward/01_fiche.md#au-delà-du-livre--diagnostiquer-lévanouissement-et-lexplosion-des-activations
[s299]: ../chapitres/ch17_activations/01_fiche.md#175--les-fonctions-linéaires-par-morceaux-
[s300]: ../chapitres/ch17_activations/01_fiche.md#1721--la-forme-dune-activation
[s301]: ../chapitres/ch17_activations/01_fiche.md#172--ce-que-fait-une-fonction-dactivation-
[s302]: ../chapitres/ch17_activations/01_fiche.md#174--les-marches-
[s303]: ../chapitres/ch17_activations/01_fiche.md#au-delà-du-livre--choisir-une-activation-et-tester-une-famille-de-fonctions
[s304]: ../chapitres/ch18_backprop/01_fiche.md#186--étape-1--les-deltas-des-neurones-de-sortie-
[s305]: ../chapitres/ch16_feedforward/01_fiche.md#au-delà-du-livre--la-passe-avant-en-matrices
[s306]: ../chapitres/ch18_backprop/01_fiche.md#184--sorties-des-neurones-et-erreur-du-réseau-
[s307]: ../chapitres/ch18_backprop/01_fiche.md#188--étape-3--les-deltas-des-autres-neurones-
[s308]: ../chapitres/ch18_backprop/01_fiche.md#18123--ce-quelle-fait-
[s309]: ../chapitres/ch22_rnn/01_fiche.md#226--entraîner-un-rnn-
[s310]: ../chapitres/ch18_backprop/01_fiche.md#au-delà-du-livre--les-losses-calculées-sur-les-logits
[s311]: ../chapitres/ch18_backprop/01_fiche.md
[s312]: ../chapitres/ch18_backprop/01_fiche.md#au-delà-du-livre--un-micro-autograd
[s313]: ../chapitres/ch19_optimiseurs/01_fiche.md#1942--la-descente-stochastique-sgd-
[s314]: ../chapitres/ch19_optimiseurs/01_fiche.md#1933--les-plannings-de-décroissance-
[s315]: ../chapitres/ch19_optimiseurs/01_fiche.md#1932--changer-le-learning-rate-au-fil-du-temps-
[s316]: ../chapitres/ch19_optimiseurs/01_fiche.md#au-delà-du-livre--warmup-et-décroissance-en-cosinus
[s317]: ../chapitres/b1_transfer/01_fiche.md#warmup-puis-cosinus-groupe-par-groupe
[s318]: ../chapitres/ch19_optimiseurs/01_fiche.md#1951--le-momentum-
[s319]: ../chapitres/ch19_optimiseurs/01_fiche.md#1952--le-momentum-de-nesterov-
[s320]: ../chapitres/ch19_optimiseurs/01_fiche.md#195--les-variantes-de-la-descente-de-gradient-
[s321]: ../chapitres/ch19_optimiseurs/01_fiche.md#1953--adagrad-
[s322]: ../chapitres/ch19_optimiseurs/01_fiche.md#1954--adadelta-et-rmsprop-
[s323]: ../chapitres/ch19_optimiseurs/01_fiche.md#1955--adam-
[s324]: ../chapitres/ch19_optimiseurs/01_fiche.md#au-delà-du-livre--adamw-le-weight-decay-découplé
[s325]: ../chapitres/ch19_optimiseurs/01_fiche.md#au-delà-du-livre--le-gradient-clipping
[s326]: ../chapitres/ch20_deep_learning/01_fiche.md#208--interpréter-les-résultats-
[s327]: ../chapitres/ch20_deep_learning/01_fiche.md#2021--les-tenseurs-
[s328]: ../chapitres/ch20_deep_learning/01_fiche.md#2031--la-couche-dentrée-
[s329]: ../chapitres/ch20_deep_learning/01_fiche.md#2032--la-couche-de-sortie-
[s330]: ../chapitres/ch20_deep_learning/01_fiche.md#2048--les-autres-couches-utilitaires-
[s331]: ../chapitres/ch25_autoencodeurs/01_fiche.md#257--autoencodeurs-convolutifs-
[s332]: ../chapitres/ch20_deep_learning/01_fiche.md
[s333]: ../chapitres/ch20_deep_learning/01_fiche.md#au-delà-du-livre--les-connexions-résiduelles
[s334]: ../chapitres/ch24_pytorch_2/01_fiche.md#246--lapi-fonctionnelle-
[s335]: ../chapitres/b3_transformers/01_fiche.md#deux-sous-couches
[s336]: ../chapitres/ch20_deep_learning/01_fiche.md#2081--une-explication-satisfaisante-
[s337]: ../chapitres/ch21_cnn/01_fiche.md#213--la-convolution-
[s338]: ../chapitres/ch21_cnn/01_fiche.md#2123--le-partage-de-poids-
[s339]: ../chapitres/ch21_cnn/01_fiche.md#2124--le-champ-récepteur-local
[s340]: ../chapitres/ch21_cnn/01_fiche.md#217--une-couche-de-convolution-
[s341]: ../chapitres/ch28_creatif/01_fiche.md#2821--choisir-un-réseau-
[s342]: ../chapitres/ch21_cnn/01_fiche.md#2131--les-filtres-
[s343]: ../chapitres/ch21_cnn/01_fiche.md#218--la-convolution-transposée-
[s344]: ../chapitres/ch21_cnn/01_fiche.md#214--la-convolution-en-plusieurs-dimensions
[s345]: ../chapitres/ch21_cnn/01_fiche.md#215--la-convolution-1d
[s346]: ../chapitres/ch21_cnn/01_fiche.md#2132--la-vue--œil-de-mouche-
[s347]: ../chapitres/ch21_cnn/01_fiche.md#2142--le-stride-pour-construire-une-hiérarchie-
[s348]: ../chapitres/ch21_cnn/01_fiche.md#2192--regarder-les-filtres-1--ce-qui-les-excite-le-plus
[s349]: ../chapitres/ch28_creatif/01_fiche.md
[s350]: ../chapitres/ch21_cnn/01_fiche.md#2193--regarder-les-filtres-2--les-cartes-dactivation-
[s351]: ../chapitres/ch21_cnn/01_fiche.md#2110--les-adversaires-
[s352]: ../chapitres/ch21_cnn/01_fiche.md#au-delà-du-livre--lattaque-fgsm
[s353]: ../chapitres/ch21_cnn/01_fiche.md#2191--vgg16-
[s354]: ../chapitres/b3_transformers/01_fiche.md#20310--coût-et-passage-à-léchelle--complexité-quadratique-flashattention-lois-déchelle-vit-
[s355]: ../chapitres/ch22_rnn/01_fiche.md#225--organiser-les-entrées-
[s356]: ../chapitres/ch22_rnn/01_fiche.md#222--introduction--des-fenêtres-pour-une-couche-dense-
[s357]: ../chapitres/ch22_rnn/01_fiche.md#224--la-structure-dune-cellule-rnn-
[s358]: ../chapitres/ch22_rnn/01_fiche.md#223--létat-
[s359]: ../chapitres/ch22_rnn/01_fiche.md#2241--une-cellule-à-plus-détat-
[s360]: ../chapitres/ch22_rnn/01_fiche.md
[s361]: ../chapitres/ch22_rnn/01_fiche.md#2271--les-portes
[s362]: ../chapitres/ch22_rnn/01_fiche.md#227--lstm-et-gru
[s363]: ../chapitres/ch22_rnn/01_fiche.md#2272--le-lstm-
[s364]: ../chapitres/ch22_rnn/01_fiche.md#2281--une-ou-plusieurs-entrées-et-sorties-
[s365]: ../chapitres/ch22_rnn/01_fiche.md#2282--le-rnn-profond-
[s366]: ../chapitres/ch22_rnn/01_fiche.md#au-delà-du-livre--prévoir-une-série-temporelle
[s367]: ../chapitres/ch22_rnn/01_fiche.md#au-delà-du-livre--après-les-rnn-transformers-et-modèles-à-espace-détats
[s368]: ../chapitres/ch23_pytorch_1/01_fiche.md#2313--les-avertissements-de-python-
[s369]: ../chapitres/ch23_pytorch_1/01_fiche.md#2322--programmer-et-déboguer-en-python
[s370]: ../chapitres/ch23_pytorch_1/01_fiche.md#2333--installer-et-configurer-la-bibliothèque-
[s371]: ../chapitres/ch23_pytorch_1/01_fiche.md#2351--remodeler-
[s372]: ../chapitres/ch23_pytorch_1/01_fiche.md#2332--tenseurs-et-tableaux-
[s373]: ../chapitres/ch23_pytorch_1/01_fiche.md#2334--la-forme-des-tenseurs-dimages
[s374]: ../chapitres/ch23_pytorch_1/01_fiche.md#2383--analyser-lhistorique-
[s375]: ../chapitres/ch23_pytorch_1/01_fiche.md#2381--lire-la-sortie-
[s376]: ../chapitres/ch23_pytorch_1/01_fiche.md#2310--callbacks-
[s377]: ../chapitres/ch23_pytorch_1/01_fiche.md#23103--early-stopping-
[s378]: ../chapitres/ch23_pytorch_1/01_fiche.md#2394--utiliser-un-modèle-pré-entraîné
[s379]: ../chapitres/b1_transfer/01_fiche.md#des-données-et-du-calcul-que-tu-nas-pas
[s380]: ../chapitres/ch23_pytorch_1/01_fiche.md
[s381]: ../chapitres/ch23_pytorch_1/01_fiche.md#2392--les-poids-seuls-
[s382]: ../chapitres/ch24_pytorch_2/01_fiche.md#2426--ajouter-du-dropout-
[s383]: ../chapitres/ch24_pytorch_2/01_fiche.md#2434--la-recherche-dhyperparamètres-
[s384]: ../chapitres/ch24_pytorch_2/01_fiche.md#2427--observations-
[s385]: ../chapitres/ch24_pytorch_2/01_fiche.md#2422--changer-un-seul-hyperparamètre-
[s386]: ../chapitres/ch24_pytorch_2/01_fiche.md#2447--des-données-synthétiques
[s387]: ../chapitres/ch24_pytorch_2/01_fiche.md#2446--laugmentation-de-données-
[s388]: ../chapitres/ch24_pytorch_2/01_fiche.md#2443--les-couches-de-convolution-
[s389]: ../chapitres/ch24_pytorch_2/01_fiche.md#2459--les-rnn--stateful-
[s390]: ../chapitres/ch24_pytorch_2/01_fiche.md#24510--une-couche-appliquée-à-chaque-pas-de-temps-
[s391]: ../chapitres/ch24_pytorch_2/01_fiche.md#au-delà-du-livre--précision-mixte-et-compilation
[s392]: ../chapitres/ch25_autoencodeurs/01_fiche.md#252--introduction--comprimer
[s393]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2551--les-variables-latentes-de-près
[s394]: ../chapitres/ch25_autoencodeurs/01_fiche.md#au-delà-du-livre--le-compromis-β-les-anomalies-et-les-autoencodeurs-aujourdhui
[s395]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2521--avec-ou-sans-perte
[s396]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2523--mélanger-deux-représentations-
[s397]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2553--mélanger-des-variables-latentes-
[s398]: ../chapitres/ch25_autoencodeurs/01_fiche.md#258--débruiter-
[s399]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2592--la-structure-dun-vae-
[s400]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2554--une-entrée-jamais-vue-
[s401]: ../chapitres/ch26_rl/01_fiche.md#2631--premier-temps--lagent-choisit-une-action-
[s402]: ../chapitres/ch26_rl/01_fiche.md#263--la-structure-du-rl-
[s403]: ../chapitres/ch26_rl/01_fiche.md#2634--des-variantes-de-la-version-simple-
[s404]: ../chapitres/ch26_rl/01_fiche.md#2635--la-vue-densemble-
[s405]: ../chapitres/ch26_rl/01_fiche.md#2637--les-récompenses-
[s406]: ../chapitres/ch26_rl/01_fiche.md#2661--les-valeurs-q-et-leur-mise-à-jour-
[s407]: ../chapitres/ch26_rl/01_fiche.md#265--le-l-learning
[s408]: ../chapitres/ch26_rl/01_fiche.md#266--le-q-learning
[s409]: ../chapitres/b8_rl_moderne/01_fiche.md#2084--acteur-critique--fonction-de-valeur-erreur-td-avantage-gae-
[s410]: ../chapitres/ch26_rl/01_fiche.md#267--sarsa-
[s411]: ../chapitres/ch26_rl/01_fiche.md#2664--léléphant-dans-la-pièce-
[s412]: ../chapitres/ch26_rl/01_fiche.md#268--la-vue-densemble-
[s413]: ../chapitres/ch26_rl/01_fiche.md#269--le-rejeu-dexpérience-
[s414]: ../chapitres/ch26_rl/01_fiche.md#2610--deux-applications
[s415]: ../chapitres/ch26_rl/01_fiche.md#au-delà-du-livre--le-rl-et-les-modèles-de-langage
[s416]: ../chapitres/ch26_rl/01_fiche.md
[s417]: ../chapitres/ch27_gan/01_fiche.md#274--implémenter-un-gan-
[s418]: ../chapitres/ch27_gan/01_fiche.md#2741--le-discriminateur-
[s419]: ../chapitres/ch27_gan/01_fiche.md#2743--entraîner-le-gan-
[s420]: ../chapitres/ch27_gan/01_fiche.md#2744--le-jeu-
[s421]: ../chapitres/ch27_gan/01_fiche.md#276--dcgan-
[s422]: ../chapitres/ch27_gan/01_fiche.md#2772--leffondrement-de-mode
[s423]: ../chapitres/ch27_gan/01_fiche.md#275--un-gan-en-action-
[s424]: ../chapitres/ch27_gan/01_fiche.md#2771--les-grandes-images-
[s425]: ../chapitres/ch27_gan/01_fiche.md#au-delà-du-livre--le-gan-conditionnel
[s426]: ../chapitres/ch27_gan/01_fiche.md#au-delà-du-livre--mesurer-un-générateur
[s427]: ../chapitres/b5_diffusion/01_fiche.md#2058--évaluer--distance-de-fréchet-diversité-mémorisation-
[s428]: ../chapitres/ch27_gan/01_fiche.md#277--les-difficultés
[s429]: ../chapitres/ch28_creatif/01_fiche.md#2822--visualiser-un-filtre-
[s430]: ../chapitres/ch28_creatif/01_fiche.md#au-delà-du-livre--la-génération-dimages-aujourdhui
[s431]: ../chapitres/ch28_creatif/01_fiche.md#283--le-deep-dream
[s432]: ../chapitres/ch28_creatif/01_fiche.md#284--le-transfert-de-style-
[s433]: ../chapitres/ch28_creatif/01_fiche.md#2841--le-style-dans-une-matrice-
[s434]: ../chapitres/ch28_creatif/01_fiche.md#2843--la-loss-de-contenu
[s435]: ../chapitres/ch28_creatif/01_fiche.md#2844--la-loss-de-style-
[s436]: ../chapitres/ch28_creatif/01_fiche.md#2842--vue-densemble-
[s437]: ../chapitres/ch28_creatif/01_fiche.md#2845--réaliser-le-transfert-
[s438]: ../chapitres/ch28_creatif/01_fiche.md#2846--discussion-
[s439]: ../chapitres/ch29_datasets/01_fiche.md#au-delà-du-livre--auditer-un-dataset
[s440]: ../chapitres/ch29_datasets/01_fiche.md#au-delà-du-livre--le-décalage-de-distribution
[s441]: ../chapitres/ch29_datasets/01_fiche.md#294--les-collections-de-datasets-
[s442]: ../chapitres/ch29_datasets/01_fiche.md
[s443]: ../chapitres/ch29_datasets/01_fiche.md#295--des-datasets-plus-récents-
[s444]: ../chapitres/b1_transfer/01_fiche.md#backbone-et-tête
[s445]: ../chapitres/b1_transfer/01_fiche.md#2013--extraction-de-features--embeddings-figés-linear-probe-k-plus-proches-voisins-
[s446]: ../chapitres/b1_transfer/01_fiche.md#calculer-les-embeddings-une-fois
[s447]: ../chapitres/b1_transfer/01_fiche.md#2014--fine-tuning--partiel-complet-dégel-progressif-learning-rates-discriminatifs-batchnorm-
[s448]: ../chapitres/b1_transfer/01_fiche.md#learning-rates-discriminatifs
[s449]: ../chapitres/b1_transfer/01_fiche.md#pourquoi-un-learning-rate-beaucoup-plus-petit
[s450]: ../chapitres/b1_transfer/01_fiche.md#rester-près-des-poids-pré-entraînés--la-pénalité-l2-sp
[s451]: ../chapitres/b1_transfer/01_fiche.md#loubli-catastrophique
[s452]: ../chapitres/b1_transfer/01_fiche.md#2017--quand-le-transfert-échoue--distance-de-domaine-transfert-négatif-oubli-catastrophique-
[s453]: ../chapitres/b1_transfer/01_fiche.md#2016--peu-de-données--courbes-dapprentissage-augmentation-few-shot-
[s454]: ../chapitres/b1_transfer/01_fiche.md#le-few-shot-et-les-prototypes-de-classe
[s455]: ../chapitres/b1_transfer/01_fiche.md#2018--modèles-de-fondation--auto-supervisé-dinov2-texte-image-clip-zéro-shot-
[s456]: ../chapitres/b1_transfer/01_fiche.md#le-zéro-shot
[s457]: ../chapitres/b1_transfer/01_fiche.md#la-courbe-dapprentissage-selon-la-taille-du-train
[s458]: ../chapitres/b1_transfer/01_fiche.md#2019--coût-dusage--taille-latence-cpu-choix-dun-backbone-pour-la-production-
[s459]: ../chapitres/b1_transfer/01_fiche.md#quatre-mesures-à-ne-pas-confondre
[s460]: ../chapitres/b1_transfer/01_fiche.md#publier-un-modèle-dérivé
[s461]: ../chapitres/b1_transfer/01_fiche.md
[s462]: ../chapitres/b2_tokenisation/01_fiche.md#un-modèle-ne-lit-que-des-identifiants
[s463]: ../chapitres/b2_tokenisation/01_fiche.md#points-de-code-et-octets
[s464]: ../chapitres/b2_tokenisation/01_fiche.md#2022--unicode-utf-8-et-normalisation-
[s465]: ../chapitres/b2_tokenisation/01_fiche.md#un-caractère-affiché-plusieurs-points-de-code
[s466]: ../chapitres/b2_tokenisation/01_fiche.md#la-pré-tokenisation
[s467]: ../chapitres/b2_tokenisation/01_fiche.md#2023--byte-pair-encoding--apprendre-et-appliquer-les-fusions-
[s468]: ../chapitres/b2_tokenisation/01_fiche.md#2024--bpe-au-niveau-octet-pré-tokenisation-et-tokens-spéciaux-
[s469]: ../chapitres/b2_tokenisation/01_fiche.md#les-tokens-spéciaux
[s470]: ../chapitres/b2_tokenisation/01_fiche.md#2025--wordpiece-unigram-sentencepiece--la-bibliothèque-tokenizers-et-les-tokenizers-des-modèles-
[s471]: ../chapitres/b2_tokenisation/01_fiche.md#unigram--partir-de-trop-et-élaguer
[s472]: ../chapitres/b2_tokenisation/01_fiche.md#sentencepiece--sans-pré-tokenisation
[s473]: ../chapitres/b2_tokenisation/01_fiche.md#2026--coûts-et-biais-de-la-tokenisation--fertilité-contexte-prix-chiffres-
[s474]: ../chapitres/b2_tokenisation/01_fiche.md#un-algorithme-de-compression-devenu-tokenizer
[s475]: ../chapitres/b2_tokenisation/01_fiche.md#utiliser-un-tokenizer-avec-transformers
[s476]: ../chapitres/b2_tokenisation/01_fiche.md#les-tokenizers-des-modèles-lus-dans-leur-tokenizerjson
[s477]: ../chapitres/b2_tokenisation/01_fiche.md#lhypothèse-distributionnelle
[s478]: ../chapitres/b2_tokenisation/01_fiche.md#la-matrice-de-co-occurrences
[s479]: ../chapitres/b2_tokenisation/01_fiche.md#2028--embeddings-distributionnels--co-occurrences-ppmi-svd-word2vec-
[s480]: ../chapitres/b2_tokenisation/01_fiche.md
[s481]: ../chapitres/b2_tokenisation/01_fiche.md#réduire--la-svd-tronquée
[s482]: ../chapitres/b2_tokenisation/01_fiche.md#word2vec--prédire-plutôt-que-compter
[s483]: ../chapitres/b2_tokenisation/01_fiche.md#compter-ou-prédire--les-deux-se-rejoignent
[s484]: ../chapitres/b2_tokenisation/01_fiche.md#les-analogies
[s485]: ../chapitres/b2_tokenisation/01_fiche.md#20210--embeddings-contextuels-et-embeddings-de-phrases--recherche-sémantique-
[s486]: ../chapitres/b2_tokenisation/01_fiche.md#dune-suite-de-vecteurs-à-un-vecteur-de-phrase
[s487]: ../chapitres/b2_tokenisation/01_fiche.md#deux-modèles-dembedding-de-phrases
[s488]: ../chapitres/b2_tokenisation/01_fiche.md#une-recherche-sémantique-en-vingt-lignes
[s489]: ../chapitres/b2_tokenisation/01_fiche.md#un-numéro-nest-pas-une-représentation
[s490]: ../chapitres/b2_tokenisation/01_fiche.md#les-biais-des-embeddings
[s491]: ../chapitres/b3_transformers/01_fiche.md#2031--des-rnn-à-lattention--goulot-de-lencodeur-décodeur-et-alignement-
[s492]: ../chapitres/b3_transformers/01_fiche.md#traduire-avec-deux-rnn
[s493]: ../chapitres/b3_transformers/01_fiche.md#lidée-de-bahdanau--relire-la-source-à-chaque-pas
[s494]: ../chapitres/b3_transformers/01_fiche.md#2032--attention-par-produit-scalaire--requêtes-clés-valeurs-mise-à-léchelle-softmax-
[s495]: ../chapitres/b3_transformers/01_fiche.md#la-formule
[s496]: ../chapitres/b3_transformers/01_fiche.md#2033--masques--causal-padding--auto-attention-et-attention-croisée-
[s497]: ../chapitres/b3_transformers/01_fiche.md#le-masque-causal
[s498]: ../chapitres/b3_transformers/01_fiche.md#le-masque-de-padding
[s499]: ../chapitres/b3_transformers/01_fiche.md#plusieurs-questions-à-la-fois
[s500]: ../chapitres/b3_transformers/01_fiche.md#2034--attention-multi-têtes--projections-découpage-en-têtes-concaténation-
[s501]: ../chapitres/b3_transformers/01_fiche.md#moins-de-têtes-pour-les-clés--mqa-et-gqa
[s502]: ../chapitres/b3_transformers/01_fiche.md
[s503]: ../chapitres/b3_transformers/01_fiche.md#lencodage-sinusoïdal-2017
[s504]: ../chapitres/b3_transformers/01_fiche.md#2035--positions--encodage-sinusoïdal-positions-apprises-rope-
[s505]: ../chapitres/b3_transformers/01_fiche.md#pré-ln-ou-post-ln
[s506]: ../chapitres/b3_transformers/01_fiche.md#2036--le-bloc-transformer--résiduel-layer-norm-pré-ou-post-mlp-gelu-
[s507]: ../chapitres/b3_transformers/01_fiche.md#le-décodeur--gpt-et-le-modèle-de-langue-causal
[s508]: ../chapitres/b3_transformers/01_fiche.md#lencodeur--bert-et-le-modèle-de-langue-masqué
[s509]: ../chapitres/b3_transformers/01_fiche.md#lencodeur-décodeur--t5-et-lattention-croisée
[s510]: ../chapitres/b3_transformers/01_fiche.md#teacher-forcing-et-décalage-dun-token
[s511]: ../chapitres/b3_transformers/01_fiche.md#2039--génération--kv-cache-et-échantillonnage-température-top-k-top-p-
[s512]: ../chapitres/b3_transformers/01_fiche.md#le-kv-cache
[s513]: ../chapitres/b3_transformers/01_fiche.md#le-masque-causal-quand-il-y-a-moins-de-requêtes-que-de-clés
[s514]: ../chapitres/b3_transformers/01_fiche.md#flashattention--le-même-résultat-sans-la-matrice
[s515]: ../chapitres/b3_transformers/01_fiche.md#les-autres-voies-vers-les-contextes-longs
[s516]: ../chapitres/b3_transformers/01_fiche.md#compter-les-paramètres-dun-bloc
[s517]: ../chapitres/b3_transformers/01_fiche.md#ce-que-coûte-le-cache
[s518]: ../chapitres/b3_transformers/01_fiche.md#allonger-le-contexte
[s519]: ../chapitres/b4_llm/01_fiche.md#trois-étapes-trois-sortes-de-modèles
[s520]: ../chapitres/b4_llm/01_fiche.md#2043--génération--generate-décodage-arrêt-streaming--chat-templates-et-rôles-
[s521]: ../chapitres/b4_llm/01_fiche.md#rôles-et-chat-template
[s522]: ../chapitres/b4_llm/01_fiche.md#licences---open-weights--nest-pas--open-source-
[s523]: ../chapitres/b4_llm/01_fiche.md#la-mémoire-des-poids
[s524]: ../chapitres/b4_llm/01_fiche.md#sarrêter
[s525]: ../chapitres/b4_llm/01_fiche.md#zéro-shot-et-few-shot
[s526]: ../chapitres/b4_llm/01_fiche.md#2044--prompting--zéro-shot-few-shot-instructions-sortie-structurée-raisonnement-pas-à-pas-
[s527]: ../chapitres/b4_llm/01_fiche.md#la-sortie-structurée
[s528]: ../chapitres/b4_llm/01_fiche.md#le-raisonnement-pas-à-pas
[s529]: ../chapitres/b4_llm/01_fiche.md#voter--lauto-cohérence
[s530]: ../chapitres/b4_llm/01_fiche.md#les-métriques
[s531]: ../chapitres/b4_llm/01_fiche.md#2045--évaluer-un-llm--jeux-de-test-métriques-juge-llm-contamination-variance-
[s532]: ../chapitres/b4_llm/01_fiche.md#découper
[s533]: ../chapitres/b4_llm/01_fiche.md#2046--recherche-dinformation--découpage-bm25-recherche-dense-hybride-bases-vectorielles-
[s534]: ../chapitres/b4_llm/01_fiche.md#de-tf-idf-à-bm25
[s535]: ../chapitres/b4_llm/01_fiche.md#la-recherche-dense
[s536]: ../chapitres/b4_llm/01_fiche.md#la-recherche-hybride-et-la-fusion-rrf
[s537]: ../chapitres/b4_llm/01_fiche.md#mesurer-une-recherche--recallk-et-mrr
[s538]: ../chapitres/b4_llm/01_fiche.md#lidée
[s539]: ../chapitres/b4_llm/01_fiche.md#2048--fine-tuning--complet-ou-peft--lora-rang-alpha-modules-ciblés-qlora-
[s540]: ../chapitres/b4_llm/01_fiche.md#lora
[s541]: ../chapitres/b4_llm/01_fiche.md#qlora
[s542]: ../chapitres/b4_llm/01_fiche.md#fusionner-ladaptateur
[s543]: ../chapitres/b4_llm/01_fiche.md#linjection-de-prompt
[s544]: ../chapitres/b4_llm/01_fiche.md
[s545]: ../chapitres/b4_llm/01_fiche.md#lanatomie-dun-prompt
[s546]: ../chapitres/b4_llm/01_fiche.md#les-hallucinations
[s547]: ../chapitres/b5_diffusion/01_fiche.md#2052--le-processus-direct--planning-de-bruit-ᾱ_t-et-forme-fermée-qx_t--x_0-
[s548]: ../chapitres/b5_diffusion/01_fiche.md#2053--le-processus-inverse-et-la-loss-simplifiée--prédire-le-bruit-lien-avec-lelbo-
[s549]: ../chapitres/b5_diffusion/01_fiche.md#le-planning-linéaire-de-ho-et-coll
[s550]: ../chapitres/b5_diffusion/01_fiche.md#un-pas-de-bruit
[s551]: ../chapitres/b5_diffusion/01_fiche.md#le-rapport-signal-sur-bruit
[s552]: ../chapitres/b5_diffusion/01_fiche.md#le-posterior--un-pas-en-arrière-quand-on-connaît-mathbfx_0
[s553]: ../chapitres/b5_diffusion/01_fiche.md#trois-façons-de-fabriquer-une-image
[s554]: ../chapitres/b5_diffusion/01_fiche.md#lalgorithme
[s555]: ../chapitres/b5_diffusion/01_fiche.md#ce-que-coûte-un-échantillon
[s556]: ../chapitres/b5_diffusion/01_fiche.md#2055--échantillonnage-ddpm--moyenne-du-posterior-variance-t-pas-
[s557]: ../chapitres/b5_diffusion/01_fiche.md#2056--accélérer--ddim-déterministe-moins-de-pas-distillation-
[s558]: ../chapitres/b5_diffusion/01_fiche.md#lidée-de-ddim
[s559]: ../chapitres/b5_diffusion/01_fiche.md#pourquoi-cest-plus-rapide
[s560]: ../chapitres/b5_diffusion/01_fiche.md#2054--le-débruiteur--u-net-minuscule-et-encodage-du-pas-de-temps-
[s561]: ../chapitres/b5_diffusion/01_fiche.md#le-u-net-et-ses-connexions-de-saut
[s562]: ../chapitres/b5_diffusion/01_fiche.md#un-bloc-résiduel-conditionné-par-le-pas
[s563]: ../chapitres/b5_diffusion/01_fiche.md#la-distillation
[s564]: ../chapitres/b5_diffusion/01_fiche.md#donner-la-classe-au-réseau
[s565]: ../chapitres/b5_diffusion/01_fiche.md#2057--conditionner--label-de-classe-et-guidance-sans-classifieur-
[s566]: ../chapitres/b5_diffusion/01_fiche.md#la-guidance-sans-classifieur
[s567]: ../chapitres/b5_diffusion/01_fiche.md
[s568]: ../chapitres/b5_diffusion/01_fiche.md#le-lien-avec-lelbo
[s569]: ../chapitres/b5_diffusion/01_fiche.md#2059--paysage-actuel--diffusion-latente-transformers-de-diffusion-flow-matching-texte-vers-image-
[s570]: ../chapitres/b5_diffusion/01_fiche.md#les-transformers-de-diffusion
[s571]: ../chapitres/b5_diffusion/01_fiche.md#flow-matching-et-rectified-flow
[s572]: ../chapitres/b5_diffusion/01_fiche.md#la-mémorisation
[s573]: ../chapitres/b5_diffusion/01_fiche.md#transparence-et-provenance-des-contenus-générés
[s574]: ../chapitres/b6_explicabilite/01_fiche.md#interprétable-par-construction-ou-expliqué-après-coup
[s575]: ../chapitres/b6_explicabilite/01_fiche.md#global-ou-local
[s576]: ../chapitres/b6_explicabilite/01_fiche.md
[s577]: ../chapitres/b6_explicabilite/01_fiche.md#les-courbes-ice
[s578]: ../chapitres/b6_explicabilite/01_fiche.md#2063--valeurs-de-shapley--jeu-coopératif-propriétés-calcul-exact-par-coalitions-rôle-du-fond-
[s579]: ../chapitres/b6_explicabilite/01_fiche.md#partager-un-gain-entre-joueurs
[s580]: ../chapitres/b6_explicabilite/01_fiche.md#quatre-propriétés-une-seule-solution
[s581]: ../chapitres/b6_explicabilite/01_fiche.md#les-features-comme-joueurs
[s582]: ../chapitres/b6_explicabilite/01_fiche.md#shap-une-famille-dexplications-additives
[s583]: ../chapitres/b6_explicabilite/01_fiche.md#un-exemple-à-deux-features
[s584]: ../chapitres/b6_explicabilite/01_fiche.md#les-trois-graphiques-à-connaître
[s585]: ../chapitres/b6_explicabilite/01_fiche.md#2064--shap-en-pratique--treeexplainer-explainer-graphiques-features-corrélées-
[s586]: ../chapitres/b6_explicabilite/01_fiche.md#deux-autres-explications-locales--substitut-et-contrefactuel
[s587]: ../chapitres/b6_explicabilite/01_fiche.md#le-gradient-comme-explication
[s588]: ../chapitres/b6_explicabilite/01_fiche.md#2065--attributions-pour-les-réseaux--saillance-gradient--entrée-gradients-intégrés-
[s589]: ../chapitres/b6_explicabilite/01_fiche.md#le-choix-de-la-ligne-de-référence
[s590]: ../chapitres/b6_explicabilite/01_fiche.md#les-gradients-intégrés
[s591]: ../chapitres/b6_explicabilite/01_fiche.md#avec-pytorch--deux-hooks
[s592]: ../chapitres/b6_explicabilite/01_fiche.md#les-tests-de-santé
[s593]: ../chapitres/b6_explicabilite/01_fiche.md#2068--biais--données-labels-mesure-variables-de-substitution-déploiement--cas-célèbres-
[s594]: ../chapitres/b6_explicabilite/01_fiche.md#doù-vient-un-biais
[s595]: ../chapitres/b6_explicabilite/01_fiche.md#cas-célèbres
[s596]: ../chapitres/b6_explicabilite/01_fiche.md#2069--équité--indicateurs-de-groupe-parité-démographique-égalité-des-chances-equalized-odds-calibration-et-impossibilités-
[s597]: ../chapitres/b6_explicabilite/01_fiche.md#les-critères
[s598]: ../chapitres/b6_explicabilite/01_fiche.md#des-écarts-avec-leurs-intervalles
[s599]: ../chapitres/b6_explicabilite/01_fiche.md#des-taux-par-groupe
[s600]: ../chapitres/b6_explicabilite/01_fiche.md#la-supervision-humaine
[s601]: ../chapitres/b6_explicabilite/01_fiche.md#20611--éthique-et-régulation-en-pratique--fiches-modèles-supervision-humaine-rgpd-ai-act-
[s602]: ../chapitres/b6_explicabilite/01_fiche.md#lai-act
[s603]: ../chapitres/b7_production/01_fiche.md#une-structure-de-dépôt
[s604]: ../chapitres/b7_production/01_fiche.md#ce-que-cache-un-notebook
[s605]: ../chapitres/b7_production/01_fiche.md#2071--du-notebook-au-projet--limites-du-notebook-structure-dun-dépôt-pyprojecttoml-installation-éditable-
[s606]: ../chapitres/b7_production/01_fiche.md#2072--scripts-et-ligne-de-commande--argparse-point-dentrée-logging-codes-de-sortie-
[s607]: ../chapitres/b7_production/01_fiche.md#quatre-familles-de-tests
[s608]: ../chapitres/b7_production/01_fiche.md#variables-denvironnement-et-secrets
[s609]: ../chapitres/b7_production/01_fiche.md#le-logging-plutôt-que-print
[s610]: ../chapitres/b7_production/01_fiche.md#2073--configuration--dataclasses-yaml-ou-json-surcharges-validation--séparer-code-et-paramètres-
[s611]: ../chapitres/b7_production/01_fiche.md#les-versions-figées-et-lempreinte-des-données
[s612]: ../chapitres/b7_production/01_fiche.md#pyprojecttoml-et-installation-éditable
[s613]: ../chapitres/b7_production/01_fiche.md#2074--reproductibilité--graines-déterminisme-versions-figées-empreinte-des-données-
[s614]: ../chapitres/b7_production/01_fiche.md
[s615]: ../chapitres/b7_production/01_fiche.md#rapporter-un-score--plusieurs-graines-et-un-intervalle
[s616]: ../chapitres/b7_production/01_fiche.md#2075--suivi-dexpériences--runs-métriques-artefacts-comparaison-
[s617]: ../chapitres/b7_production/01_fiche.md#2076--tester-un-projet-ml--tests-unitaires-tests-de-données-test-doverfitting-intégration-continue-
[s618]: ../chapitres/b7_production/01_fiche.md#lintégration-continue
[s619]: ../chapitres/b7_production/01_fiche.md#une-grille--le-ml-test-score
[s620]: ../chapitres/b7_production/01_fiche.md#2077--empaqueter-et-servir--artefact-de-modèle-inférence-par-batches-ou-en-ligne-api-http-latence-
[s621]: ../chapitres/b7_production/01_fiche.md#en-ligne-ou-par-batches-et-une-api-http
[s622]: ../chapitres/b7_production/01_fiche.md#latence-percentiles-et-débit
[s623]: ../chapitres/b7_production/01_fiche.md#2078--exporter-et-optimiser-linférence--torchexport-onnx-précision-réduite-quantification-
[s624]: ../chapitres/b7_production/01_fiche.md#torchexport-et-onnx
[s625]: ../chapitres/b7_production/01_fiche.md#accélérer-sur-cpu
[s626]: ../chapitres/b7_production/01_fiche.md#2079--conteneurs-et-déploiement--dockerfile-variables-denvironnement-secrets-démos-hébergées-
[s627]: ../chapitres/b7_production/01_fiche.md#une-démo-hébergée
[s628]: ../chapitres/b7_production/01_fiche.md#lindice-de-stabilité-de-population-psi
[s629]: ../chapitres/b7_production/01_fiche.md#le-test-de-kolmogorov-smirnov
[s630]: ../chapitres/b7_production/01_fiche.md#pourquoi-un-modèle-se-dégrade-sans-changer-et-que-surveiller
[s631]: ../chapitres/b7_production/01_fiche.md#20710--surveiller-en-production--dérive-des-données-et-des-performances-journalisation-retour-arrière-
[s632]: ../chapitres/b7_production/01_fiche.md#agir--alerter-réentraîner-revenir-en-arrière
[s633]: ../chapitres/b7_production/01_fiche.md#image-conteneur-et-dockerfile
[s634]: ../chapitres/b7_production/01_fiche.md#la-dette-technique-du-ml
[s635]: ../chapitres/b7_production/01_fiche.md#journaliser-les-prédictions-et-le-rgpd
[s636]: ../chapitres/b8_rl_moderne/01_fiche.md#2081--des-tables-aux-réseaux--approximation-de-fonction-états-continus-instabilité-
[s637]: ../chapitres/b8_rl_moderne/01_fiche.md#instabilité--la-triade-mortelle
[s638]: ../chapitres/b8_rl_moderne/01_fiche.md#apprendre-une-valeur-par-régression
[s639]: ../chapitres/b8_rl_moderne/01_fiche.md#les-journaux-de-ppo
[s640]: ../chapitres/b8_rl_moderne/01_fiche.md#la-surestimation-et-double-dqn
[s641]: ../chapitres/b8_rl_moderne/01_fiche.md#apprendre-la-politique-elle-même
[s642]: ../chapitres/b8_rl_moderne/01_fiche.md#le-théorème-du-gradient-de-politique
[s643]: ../chapitres/b8_rl_moderne/01_fiche.md#la-baseline-et-la-variance
[s644]: ../chapitres/b8_rl_moderne/01_fiche.md#réutiliser-un-rollout-sans-tout-casser
[s645]: ../chapitres/b8_rl_moderne/01_fiche.md#2085--ppo--ratio-de-probabilités-objectif-clippé-entropie-environnements-vectorisés-
[s646]: ../chapitres/b8_rl_moderne/01_fiche.md#les-environnements-vectorisés
[s647]: ../chapitres/b8_rl_moderne/01_fiche.md#le-critique-et-sa-variance-expliquée
[s648]: ../chapitres/b8_rl_moderne/01_fiche.md#2088--limites--efficacité-en-échantillons-généralisation-sim-to-real-sécurité-
[s649]: ../chapitres/b8_rl_moderne/01_fiche.md#rapporter-plusieurs-graines
[s650]: ../chapitres/b8_rl_moderne/01_fiche.md#le-modèle-de-récompense-de-bradley-terry
[s651]: ../chapitres/b8_rl_moderne/01_fiche.md#le-rlhf--ppo-sous-une-laisse-kl
[s652]: ../chapitres/b8_rl_moderne/01_fiche.md#grpo-et-les-récompenses-vérifiables
[s653]: ../chapitres/b8_rl_moderne/01_fiche.md#le-piratage-de-la-récompense
[s654]: ../chapitres/b8_rl_moderne/01_fiche.md#de-la-simulation-au-réel
[s655]: ../chapitres/b8_rl_moderne/01_fiche.md#sécurité
[s656]: ../chapitres/b8_rl_moderne/01_fiche.md
[s657]: ../chapitres/b8_rl_moderne/01_fiche.md#gae--une-moyenne-de-toutes-les-longueurs
[s658]: ../chapitres/b8_rl_moderne/01_fiche.md#déboguer-un-agent-qui-napprend-pas
[s659]: ../projets/partie_1_detecteur_langue/README.md#la-grille-dévaluation-sur-20
[s660]: ../checkpoints/partie_1/05_synthese.md
[s661]: ../checkpoints/partie_2/05_synthese.md
[s662]: ../projets/partie_2_california_validation/README.md#le-protocole-le-cœur-du-projet
[s663]: ../projets/partie_2_california_validation/README.md#les-extensions-pour-aller-plus-loin
[s664]: ../projets/partie_3_revenus/README.md#la-grille-dévaluation-sur-20
[s665]: ../projets/partie_3_revenus/README.md#le-contexte
[s666]: ../projets/partie_4_mini_framework/README.md#les-livrables
[s667]: ../projets/partie_6_generatif/README.md
[s668]: ../chapitres/ch00a_python/01_fiche.md#10042--boucles--for-range-enumerate-zip-while-break-continue
[s669]: ../chapitres/ch00a_python/01_fiche.md#10052--arguments-positionnels-et-nommés-args-kwargs-arguments-keyword-only
[s670]: ../chapitres/ch00a_python/01_fiche.md#10053--fonctions-comme-valeurs--lambda-key-callable
[s671]: ../chapitres/ch00a_python/01_fiche.md#10061--modules-imports-et-bibliothèque-standard
[s672]: ../chapitres/ch00a_python/01_fiche.md#10063--fichiers-texte-et-chemins-avec-pathlib
[s673]: ../chapitres/ch00a_python/01_fiche.md#10066--itertools-et-heapq-aperçu
[s674]: ../chapitres/ch00a_python/01_fiche.md#10072--méthodes-spéciales--__repr__-__len__-__getitem__-__call__-opérateurs
[s675]: ../chapitres/ch00a_python/01_fiche.md#10086--reshape-transposition-empilement
[s676]: ../chapitres/ch00a_python/01_fiche.md#10088--les-images-sont-des-arrays--mnist
[s677]: ../chapitres/ch00a_python/01_fiche.md#100101--plot-scatter-hist--titres-axes-et-légendes
[s678]: ../chapitres/ch00a_python/01_fiche.md#100114--doctest--des-exemples-exécutables-dans-les-docstrings
[s679]: ../chapitres/ch00a_python/01_fiche.md#10071--class-__init__-attributs-et-méthodes--dataclass
[s680]: ../chapitres/ch00b_maths/01_fiche.md#1011--nombres-notations-et-dénombrement
[s681]: ../chapitres/ch00b_maths/01_fiche.md#1012--fonctions-usuelles
[s682]: ../chapitres/ch00b_maths/01_fiche.md#10127--composition-de-fonctions
[s683]: ../chapitres/ch00b_maths/01_fiche.md#10152--dérivées-usuelles-et-règles-somme-produit-quotient-exp-ln
[s684]: ../chapitres/ch00b_maths/01_fiche.md#10154--variations-minimum-et-maximum
[s685]: ../chapitres/ch02_stats/01_fiche.md#21--pourquoi-des-statistiques-
[s686]: ../chapitres/ch02_stats/01_fiche.md#23--quelques-lois-usuelles
[s687]: ../chapitres/ch02_stats/01_fiche.md#251--avec-remise-252-sans-remise-
[s688]: ../chapitres/ch07_classification/01_fiche.md#74--plusieurs-classes-avec-des-classifieurs-binaires-
[s689]: ../chapitres/ch11_raisonnement/01_fiche.md
[s690]: ../chapitres/ch12_preparation/01_fiche.md
[s691]: ../chapitres/ch29_datasets/01_fiche.md#2931--scikit-learn-
[s692]: ../chapitres/ch15_sklearn/01_fiche.md#1544--decision_function-et-predict_proba-
[s693]: ../chapitres/ch15_sklearn/01_fiche.md#155--le-clustering
[s694]: ../chapitres/ch15_sklearn/01_fiche.md#1591--la-validation-croisée-
[s695]: ../chapitres/ch15_sklearn/01_fiche.md#153--conventions--importer-lire-la-documentation-suivre-les-dépréciations-
[s696]: ../chapitres/ch17_activations/01_fiche.md
[s697]: ../chapitres/ch17_activations/01_fiche.md#177--la-galerie-des-activations-
[s698]: ../chapitres/ch18_backprop/01_fiche.md#18124--garder-les-neurones-en-vie
[s699]: ../chapitres/ch18_backprop/01_fiche.md#18121--la-rétropropagation-en-un-seul-endroit-
[s700]: ../chapitres/ch18_backprop/01_fiche.md#18125--les-mini-batches-
[s701]: ../chapitres/ch19_optimiseurs/01_fiche.md#192--lerreur-comme-géométrie
[s702]: ../chapitres/ch20_deep_learning/01_fiche.md#207--construire-un-réseau-profond-
[s703]: ../chapitres/ch20_deep_learning/01_fiche.md#201--pourquoi-ce-chapitre--
[s704]: ../chapitres/ch20_deep_learning/01_fiche.md#204--un-tour-dhorizon-des-couches
[s705]: ../chapitres/ch20_deep_learning/01_fiche.md#206--quelques-exemples-
[s706]: ../chapitres/ch21_cnn/01_fiche.md#au-delà-du-livre--la-rétropropagation-dune-convolution
[s707]: ../chapitres/ch21_cnn/01_fiche.md
[s708]: ../chapitres/ch23_pytorch_1/01_fiche.md#2362--créer-le-modèle-
[s709]: ../chapitres/ch23_pytorch_1/01_fiche.md#2357--corriger-les-labels-
[s710]: ../chapitres/ch23_pytorch_1/01_fiche.md#237--entraîner-le-modèle-
[s711]: ../chapitres/ch23_pytorch_1/01_fiche.md#23101--checkpoints-
[s712]: ../chapitres/ch23_pytorch_1/01_fiche.md#23102--le-learning-rate-
[s713]: ../chapitres/ch24_pytorch_2/01_fiche.md
[s714]: ../chapitres/ch24_pytorch_2/01_fiche.md#2431--habiller-un-réseau-en-estimateur-
[s715]: ../chapitres/ch24_pytorch_2/01_fiche.md#2462--construire-un-modèle-fonctionnel-
[s716]: ../chapitres/ch24_pytorch_2/01_fiche.md#2442--préparer-les-données-pour-un-cnn-
[s717]: ../chapitres/ch24_pytorch_2/01_fiche.md#2451--générer-des-séries
[s718]: ../chapitres/ch24_pytorch_2/01_fiche.md#2456--les-rnn-profonds-
[s719]: ../chapitres/ch24_pytorch_2/01_fiche.md#24511--générer-du-texte
[s720]: ../chapitres/ch25_autoencodeurs/01_fiche.md#256--discussion--lautoencodeur-et-lacp
[s721]: ../chapitres/ch25_autoencodeurs/01_fiche.md
[s722]: ../chapitres/ch27_gan/01_fiche.md
[s723]: ../chapitres/ch29_datasets/01_fiche.md#291--les-datasets-publics-
[s724]: ../chapitres/b2_tokenisation/01_fiche.md#le-même-texte-un-autre-prix-selon-la-langue
[s725]: ../chapitres/b2_tokenisation/01_fiche.md#les-plus-proches-voisins
[s726]: ../chapitres/b2_tokenisation/01_fiche.md#2029--géométrie-des-embeddings--cosinus-voisins-analogies-visualisation-biais-
[s727]: ../chapitres/b3_transformers/01_fiche.md#les-positions-apprises
[s728]: ../chapitres/b3_transformers/01_fiche.md#2037--familles--encodeur-bert-décodeur-gpt-encodeur-décodeur-t5--objectifs-causal-et-masqué-
[s729]: ../chapitres/b3_transformers/01_fiche.md#2038--mini-gpt-from-scratch--données-entraînement-loss-perplexité-
[s730]: ../chapitres/b4_llm/01_fiche.md#2042--charger-et-exécuter-un-modèle-avec-transformers--tokenizer-modèle-pipeline-dtype-mémoire-
[s731]: ../chapitres/b4_llm/01_fiche.md#ce-que-fait-vraiment-un-llm
[s732]: ../chapitres/b4_llm/01_fiche.md#rag-ou-fine-tuning-
[s733]: ../chapitres/b4_llm/01_fiche.md#2049--classer-des-textes--encodeur-fine-tuné-llm-en-zéro-shot-ou-baseline-classique-
[s734]: ../chapitres/b5_diffusion/01_fiche.md#le-planning-cosinus-de-nichol-et-dhariwal
[s735]: ../chapitres/b5_diffusion/01_fiche.md#dire-au-réseau-quel-est-le-pas--lencodage-du-pas-de-temps
[s736]: ../chapitres/b5_diffusion/01_fiche.md#en-pratique--la-moyenne-exponentielle-des-poids
[s737]: ../chapitres/b5_diffusion/01_fiche.md#la-distance-de-fréchet
[s738]: ../chapitres/b6_explicabilite/01_fiche.md#2066--grad-cam--cartes-dactivation-pondérées-par-les-gradients-
[s739]: ../chapitres/b6_explicabilite/01_fiche.md#2067--limites-des-explications--instabilité-tests-de-santé-corrélation-nest-pas-causalité-
[s740]: ../chapitres/b8_rl_moderne/01_fiche.md#2086--pratique-du-deep-rl--hyperparamètres-graines-évaluation-bibliothèques-de-référence-
