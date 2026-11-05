# Index des notions

> **Généré** par `python tools/syllabus.py build` (`tools/build_index.py`) à partir du glossaire (`annexes/glossaire.md`), du contrat de chaque chapitre (`docs/syllabus/data/*.json` : notions enseignées, concepts, notions mobilisées, exercices), des fiches et des énoncés des exercices : on ne l'édite pas à la main.

**1566 notions**, et **1604 renvois** depuis l'autre langue, un synonyme ou un mot commun à plusieurs entrées, dans l'ordre alphabétique (accents ignorés). Pour chaque notion :

- **enseignée** : les chapitres qui la présentent, la leçon principale d'abord, chacun avec un lien vers la section de sa fiche : un titre qui porte sur elle ou qui la nomme, sinon sa définition (en gras ou suivie de sa glose), sinon la section qui en parle le plus ; **évoquée** : un chapitre qui la mentionne seulement, souvent avant la leçon (le panorama du ch. 1, par exemple) ;
- **exercices** : ceux dont le titre la nomme, puis ceux dont l'énoncé la nomme, d'abord dans les chapitres qui l'enseignent (10 au plus, le nombre des autres entre parenthèses) ;
- **approfondie** : les autres chapitres dont une notion enseignée la contient ; **réutilisée** : les chapitres qui la mobilisent comme prérequis.

Entre parenthèses et en italique, l'autre langue. Une ligne « *terme* → voir … » renvoie de l'autre langue (le plus souvent l'anglais) vers le terme retenu (BIBLE §5) ; une ligne « terme → voir … », d'un synonyme ou d'un mot commun vers les entrées qui le contiennent. Les définitions sont dans le [glossaire](glossaire.md), les formules dans le [formulaire](formulaire.md). La recherche est automatique : un lien qui tombe à côté se signale avec P6.

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [X](#x) · [Y](#y) · [Z](#z) · [0-9](#symboles)

## A

- *à quelques exemples* → voir [few-shot](#f)
- *A/B test* → voir [test A/B](#t)
- **abandon du label (guidance sans classifieur)** (*label dropout, condition dropout*) — enseignée : [ch. B5, 205.7 · Conditionner : label de classe et guidance sans classifieur][s624] · exercices : B5.27
- **abduction** (*inference to the best explanation*) — enseignée : [ch. 11, 11.6.1 · Sherlock Holmes, « maître de la déduction »][s296] · exercices : 11.Q12
- **ablation** (*ablation study*) — enseignée : [ch. 20, Au-delà du livre : régulariser un réseau][s381] · exercices : 20.29, B3.25, B3.30, B5.11, B8.15, PF.7
- *ablation study* → voir [ablation](#a)
- absence → voir [précision locale, absence, cohérence](#p)
- *absolute value* → voir [valeur absolue](#v)
- **abstention** — enseignée : [ch. B4, Les hallucinations][s174] · exercices : B4.23
- **accès contrôlé (dépôt)** (*gated, gated repository*) — enseignée : [ch. B2, Les tokenizers des modèles, lus dans leur `tokenizer.json`][s534] · exercices : B4.Q2
- **accord top-1** (*top-1 agreement*) — enseignée : [ch. B1, L'oubli catastrophique][s508] · exercices : B1.23
- *accumulated local effects* → voir [courbe ALE](#c)
- **accumulation des gradients** (*gradient accumulation*) — enseignée : [ch. 18][s355] · exercices : 21.R2
- **accuracy** (*exactitude, taux de bonnes réponses*) — enseignée : [ch. 3, 3.7.5 · Accuracy][s37] ; [ch. 1, 1.2.3 · Généraliser : le jeu de test][s38] · exercices : 1.1, 3.16, 3.E1, 3.Q9, CP1.14, 7.24, 7.30, 8.4, 8.12, 10.R3 (+ 27) · réutilisée : ch. 8, 10, 20, 23
- *accuracy équilibrée* → voir [balanced accuracy](#b)
- **accuracy top-5** (*top-5 accuracy*) — enseignée : [ch. 20][s376] · exercices : 20.Q10
- *ACP* → voir [PCA](#p)
- acteur → voir [acteur-critique, acteur, critique](#a)
- **acteur-critique et GAE** — enseignée : [ch. B8, 208.4 · Acteur-critique : fonction de valeur, erreur TD, avantage, GAE][s689] · exercices : B8.4, B8.18, B8.20, B8.Q7
- **acteur-critique, acteur, critique** (*actor-critic, actor, critic*) — enseignée : [ch. B8, 208.4 · Acteur-critique : fonction de valeur, erreur TD, avantage, GAE][s689] · exercices : B8.20
- action → voir [agent, environnement, action, récompense](#a), [workflow, job, étape, action](#w)
- *action mask* → voir [masque d'actions](#m)
- *activation function* → voir [fonction d'activation](#f)
- *activation map* → voir [carte d'activation](#c)
- *activation maximization* → voir [maximisation d'activation](#m)
- **activations linéaires par morceaux (ReLU et variantes, maxout)** — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s235] · exercices : 17.12
- **activations lisses (softplus, ELU, sigmoïde, tanh, SiLU, GELU)** — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s110] · exercices : 17.4
- *actor* → voir [acteur-critique, acteur, critique](#a)
- *actor-critic* → voir [acteur-critique, acteur, critique](#a)
- **AdaBoost** (*adaptive boosting*) — enseignée : [ch. 14, 14.7 · Le boosting][s316] · exercices : 13.25, 14.2, 14.6, 14.22–24, 14.Q9, 15.16, 15.Q9, CP3.5
- **Adadelta** — enseignée : [ch. 19, 19.5.4 · Adadelta et RMSprop][s366] · exercices : 19.13, 19.17, 19.27, 19.Q11–Q12, 21.22
- **Adagrad** (*adaptive gradient*) — enseignée : [ch. 19, 19.5.3 · Adagrad][s365] · exercices : 19.4, 19.13, 19.17, 19.23, 19.27, 19.Q11–Q12
- **AdaIN** (*adaptive instance normalization*) — enseignée : [ch. 28, 28.4.6 · Discussion][s494] · exercices : 28.25
- **adaLN-Zero** (*adaptive layer norm, zero-initialized*) — enseignée : [ch. B5, Les transformers de diffusion][s626]
- **Adam** (*adaptive moment estimation*) — enseignée : [ch. 19, 19.5.5 · Adam][s367] · exercices : 19.5–6, 19.12, 19.18, 19.22, 19.24, 19.E1–E2, 19.Q12, CP4.3 (+ 62) · réutilisée : ch. 21, 22, 23, 24, 25, 27, B8
- **AdamW** — enseignée : [ch. 19, Au-delà du livre : AdamW, le weight decay découplé][s368] · exercices : 19.6, 19.13, 19.18, 19.E2, 20.28, 20.R1, CP4.3, CP4.6, CP4.9, B3.20 (+ 15) · réutilisée : ch. 20, B1, B3, B4, B5
- **adaptateur** (*adapter, wrapper*) — enseignée : [ch. 15, Au-delà du livre : brancher ton propre estimateur][s333] · exercices : 15.29
- **adaptateur (PEFT)** (*adapter*) — enseignée : [ch. B4, 204.8 · Fine-tuning : complet ou PEFT ; LoRA (rang, alpha, modules…][s597] · exercices : B4.3, B4.10–11, B4.25–28
- *adapter* → voir [adaptateur](#a), [adaptateur (PEFT)](#a)
- *adaptive boosting* → voir [AdaBoost](#a)
- *adaptive gradient* → voir [Adagrad](#a)
- *adaptive instance normalization* → voir [AdaIN](#a)
- *adaptive layer norm* → voir [adaLN-Zero](#a)
- *adaptive moment estimation* → voir [Adam](#a)
- *add-one smoothing* → voir [lissage de Laplace](#l)
- **`add_generation_prompt`** — enseignée : [ch. B4, Rôles et chat template][s582] · exercices : B4.4, B4.14, B4.Q4
- *additive attention* → voir [attention additive](#a)
- *additive smoothing* → voir [lissage additif](#l)
- additivité → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *additivity* → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *adjusted Rand index* → voir [indice de Rand ajusté (ARI)](#i)
- *advantage* → voir [avantage](#a)
- *adversarial diffusion distillation* → voir [distillation adversariale](#d)
- *adversarial example* → voir [exemple adversarial](#e)
- *adversarial training* → voir [entraînement adversarial](#e)
- **affine par morceaux** (*piecewise linear*) — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s235] · exercices : 17.3
- **affirmation du conséquent** (*affirming the consequent*) — enseignée : [ch. 11, 11.4 · La déduction][s297] · exercices : 11.5, 11.Q8
- *affirming the consequent* → voir [affirmation du conséquent](#a)
- **agent libre, déclenché** (*free-running, triggered agent*) — enseignée : [ch. 26, 26.3.4 · Des variantes de la version simple][s460]
- **agent, environnement, action, récompense** (*environment, reward*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s154] ; [ch. 1, 1.6 · Apprentissage par renforcement][s61] · exercices : 1.22, 11.24, 20.10, 26.11, 26.24, 26.37, 26.Q2, 26.Q4, 27.R1, CP6.6 (+ 44)
- **agentivité excessive** (*excessive agency*) — enseignée : [ch. B4, L'injection de prompt][s600] · exercices : B4.Q12
- *AI Act (règlement 2024/1689)* → voir [règlement européen sur l'IA](#r)
- *AI winter* → voir [hiver de l'IA](#h)
- **ajustement bayésien d'une droite** — enseignée : [ch. 9, 9.7 · Ajuster une droite avec la règle de Bayes][s279]
- *ajustement croisé* → voir [cross fitting](#c)
- algorithme → voir [algorithme de Lloyd](#a), [algorithme de Viterbi](#a), [algorithme EM](#a), [algorithme en ligne, hors ligne](#a), [algorithme glouton](#a), [algorithme paresseux](#a), [algorithme pocket](#a), [algorithmes déterministes](#a)
- **algorithme de Lloyd** (*Lloyd's algorithm*) — enseignée : [ch. 7, Sans labels : k-means][s170] · exercices : 7.3, 7.12, 7.25–26, 11.Q1, CP2.3
- **algorithme de Viterbi** (*Viterbi algorithm*) — enseignée : [ch. B2, Unigram : partir de trop et élaguer][s529]
- **algorithme EM** (*expectation-maximization*) — enseignée : [ch. B2, Unigram : partir de trop et élaguer][s529]
- **algorithme en ligne, hors ligne** (*online, offline algorithm*) — enseignée : [ch. 19, 19.4.1 · La descente de gradient par batch][s6] · exercices : B7.18, B7.Q8
- **algorithme glouton** (*greedy algorithm*) — enseignée : [ch. 13][s315]
- **algorithme paresseux** (*lazy learner*) — enseignée : [ch. 13, 13.3 · Les k plus proches voisins][s257]
- **algorithme pocket** (*pocket algorithm*) — enseignée : [ch. 10][s289]
- **algorithmes déterministes** (*deterministic algorithms*) — enseignée : [ch. B7][s665] · exercices : B7.13
- **ALiBi** (*attention with linear biases*) — enseignée : [ch. B3, Allonger le contexte][s576]
- **alignement (traduction)** (*alignment*) — enseignée : [ch. B3, 203.1 · Des RNN à l'attention : goulot de l'encodeur-décodeur et…][s552] · exercices : B3.2
- **alignement, RLHF** (*alignment, reinforcement learning from human feedback*) — enseignée : [ch. B8, Le RLHF : PPO sous une laisse KL][s164] · évoquée : [ch. 1, 1.5 · Générateurs][s152] · exercices : B8.10, B8.20, B8.26, B8.E4, B8.Q11–Q12
- *alignment* → voir [alignement, RLHF](#a), [alignement (traduction)](#a)
- **amorçage** (*bootstrapping, RL*) — enseignée : [ch. 26, 26.6.1 · Les valeurs Q et leur mise à jour][s462] · exercices : 26.21, B8.4, B8.18, B8.Q2
- **amorce** (*seed*) — enseignée : [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s249] · exercices : 22.25, B8.18
- analyse → voir [analyse d'erreurs](#a), [analyse d'impact sur les droits fondamentaux (FRIA)](#a), [analyse de sensibilité](#a), [analyse des erreurs](#a), [analyse exploratoire](#a), [analyse intersectionnelle](#a)
- **analyse d'erreurs** — enseignée : [ch. 23, 23.8.3 · Analyser l'historique][s426] ; [projet final, PF.8 — Évaluation honnête sur le test scellé : une seule passe,…][s759] · exercices : PF.8
- **analyse d'impact sur les droits fondamentaux (FRIA)** (*fundamental rights impact assessment*) — enseignée : [ch. B6, L'AI Act][s652] · exercices : B6.Q12
- **analyse de sensibilité** (*sensitivity analysis*) — enseignée : [ch. 4][s221]
- **analyse des erreurs** (*error analysis*) — enseignée : [ch. 23, 23.8.1 · Lire la sortie][s427]
- *analyse en composantes principales* → voir [PCA](#p)
- **analyse exploratoire** (*exploratory data analysis, EDA*) — enseignée : [ch. 2, 2.9 · Le quartet d'Anscombe][s194]
- **analyse intersectionnelle** (*intersectional analysis*) — enseignée : [ch. B6, Cas célèbres][s646] · exercices : B6.10
- *ancestor* → voir [parent, enfant ; ancêtre, descendant](#p)
- *ancestral sampling* → voir [échantillonnage ancestral](#e)
- ancêtre → voir [parent, enfant ; ancêtre, descendant](#p)
- *anchor* → voir [ancre, pixel focal](#a)
- **ancre, pixel focal** (*anchor, focal pixel*) — enseignée : [ch. 21, 21.2.4 · Le champ récepteur local][s388] · exercices : 21.Q2
- *ANN* → voir [recherche approchée des plus proches voisins](#r)
- **annotation de type** (*type hint*) — enseignée : [ch. 0A, 100.5.5 · Docstrings au format NumPy et annotations de type][s80] · exercices : 0B.R3
- **annulation catastrophique** (*catastrophic cancellation*) — enseignée : [ch. 5, Au-delà du livre (2) : les arrondis et le choix du pas][s228]
- *anomaly detection* → voir [détection d'anomalies](#d)
- **ANS** (*asymmetric numeral systems*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s162] · exercices : 3.28, 4.Q1, 5.14, 11.5, 11.Q3, 11.Q6, 11.Q10, 14.9, 20.11, 20.Q12 (+ 8)
- *Anscombe's quartet* → voir [quartet d'Anscombe, Datasaurus](#q)
- AP → voir [average precision (AP)](#a)
- **API de prédiction** — enseignée : [ch. B7, En ligne ou par batches, et une API HTTP][s670] · exercices : B7.18
- **API fonctionnelle** (*functional API*) — enseignée : [ch. 24, 24.6 · L'API fonctionnelle][s171] · exercices : 24.28, 24.Q12
- **API gymnasium (reset, step, terminated, truncated)** — enseignée : [ch. 26, 26.3 · La structure du RL][s459]
- **API HTTP, route** (*HTTP API, endpoint*) — enseignée : [ch. B7, 207.7 · Empaqueter et servir : artefact de modèle, inférence par…][s669] · exercices : B7.18, PF.11
- **API scikit-learn** — enseignée : [ch. 15, 15.2 · scikit-learn, NumPy et SciPy][s18] ; [checkpoint 3, 6. Anatomie de l'API scikit-learn][s740] ; [ch. 29, 29.3.1 · scikit-learn][s741] · réutilisée : ch. 16
- **aplatissement** (*flatten*) — enseignée : [ch. 20, 20.4.8 · Les autres couches utilitaires][s374] · exercices : 0A.54, 20.6, 20.20–21, 20.28, 20.Q9, 21.4, 21.22, B1.1
- *appareil* → voir [device](#d)
- *appeal to coincidence* → voir [induction paresseuse](#i)
- **apprenant faible, apprenant fort** (*weak learner, strong learner*) — enseignée : [ch. 14, 14.7 · Le boosting][s316] · exercices : 14.Q8
- apprenant fort → voir [apprenant faible, apprenant fort](#a)
- apprentissage → voir [apprentissage auto-supervisé](#a), [apprentissage contrastif](#a), [apprentissage en contexte](#a), [apprentissage hiérarchique](#a), [apprentissage non supervisé](#a), [apprentissage par renforcement](#a), [apprentissage semi-supervisé](#a), [apprentissage supervisé](#a)
- **apprentissage auto-supervisé** (*self-supervised learning*) — enseignée : [ch. 1, 1.5 · Générateurs][s152] ; [ch. 25, 25.3 · Le plus simple des autoencodeurs][s153] · exercices : 1.5
- *apprentissage automatique* → voir [machine learning (ML)](#m)
- **apprentissage contrastif** (*contrastive learning*) — enseignée : [ch. B1, La courbe d'apprentissage selon la taille du train][s514]
- *apprentissage des features* → voir [feature learning](#f)
- **apprentissage en contexte** (*in-context learning*) — enseignée : [ch. B4, Zéro-shot et few-shot][s513]
- **apprentissage hiérarchique** (*hierarchical learning*) — enseignée : [ch. 20, 20.8 · Interpréter les résultats][s370] · exercices : 20.10, 20.Q1
- **apprentissage non supervisé** (*unsupervised learning*) — enseignée : [ch. 1, 1.4 · Apprentissage non supervisé][s60] · exercices : 1.21, 1.E3, 1.Q8
- **apprentissage par renforcement** (*reinforcement learning*) — enseignée : [ch. 1, 1.6 · Apprentissage par renforcement][s61] ; [ch. 26, Apprentissage par renforcement][s62] · exercices : 26.E5, 26.Q1–Q7, 29.R2, CP6.7 (+ 50)
- *apprentissage par transfert* → voir [transfer learning](#t)
- *apprentissage profond* → voir [deep learning](#d)
- **apprentissage semi-supervisé** (*semi-supervised learning*) — enseignée : [ch. 1, 1.3 · Apprentissage supervisé][s59]
- **apprentissage supervisé** (*supervised learning*) — enseignée : [ch. 1, 1.3 · Apprentissage supervisé][s59] · exercices : 2.R1, CP1.1, 26.Q1, B8.9 · réutilisée : ch. 7
- *approximate KL* → voir [KL approchée](#k)
- *approximate nearest neighbor search* → voir [recherche approchée des plus proches voisins](#r)
- **approximation de fonction** (*function approximation*) — enseignée : [ch. B8, 208.1 · Des tables aux réseaux : approximation de fonction, états…][s681] · exercices : B8.Q2
- **arbre de décision** (*decision tree*) — enseignée : [ch. 13, 13.5 · Les arbres de décision][s169] ; [ch. 1, 1.1.2 · Les systèmes experts et leurs limites][s141] · exercices : 1.18, 1.20, 1.Q5, 13.11, 13.24, 13.E2–E3, 13.Q2, 13.Q8, 13.Q12 (+ 8)
- **arbre de régression** — enseignée : [ch. 13, 13.5 · Les arbres de décision][s169] · exercices : 13.26, 14.4, 14.7, 14.25, CP3.8
- arbre des fréquences naturelles → voir [fréquences naturelles, arbre des fréquences naturelles](#f)
- **architecture d'un réseau profond** — enseignée : [ch. 20, 20.7 · Construire un réseau profond][s751]
- **architectures BERT/GPT/T5** — enseignée : [ch. B3, 203.7 · Familles : encodeur (BERT), décodeur (GPT),…][s777]
- *area under the curve* → voir [AUC](#a)
- **argparse et logging** — enseignée : [ch. B7, 207.2 · Scripts et ligne de commande : argparse, point d'entrée,…][s657] · exercices : B7.10, B7.25, B7.E1, B7.Q3
- **\*args, \*\*kwargs, arguments keyword-only** — enseignée : [ch. 0A, 100.5.2 · Arguments positionnels et nommés, `*args`, `**kwargs`,…][s717] · exercices : 0A.40, 0A.Q6, 13.34, 22.18, B7.10
- argument → voir [fonction, paramètre, argument](#f)
- ARI → voir [indice de Rand ajusté (ARI)](#i)
- *arithmetic coding* → voir [codage arithmétique](#c)
- **array** (*tableau NumPy*) — enseignée : [ch. 0A, 100.8.1 · Créer des arrays : `dtype`, `shape`, `ndim`][s90] · exercices : 0A.9, 0A.27–29, 0A.35, 0A.49–50, 0A.Q10, 0B.R2, 23.Q5 (+ 26) · réutilisée : ch. 0B, 16, 20
- **artefact** (*artifact*) — enseignée : [ch. B7, 207.5 · Suivi d'expériences : runs, métriques, artefacts, comparaison][s437] · exercices : B7.1, B7.14, B7.16–18, B7.21, B7.24–25, B7.Q7–Q8 (+ 3)
- **artefact de modèle** (*model artifact*) — enseignée : [ch. B7, 207.7 · Empaqueter et servir : artefact de modèle, inférence par…][s669] · exercices : B7.Q8
- **artefacts en damier** (*checkerboard artifacts*) — enseignée : [ch. 21, 21.8 · La convolution transposée][s392] · exercices : 21.Q9
- *artifact* → voir [artefact](#a)
- astuce → voir [astuce de la log-dérivée](#a), [astuce de reparamétrisation](#a), [astuce du biais](#a), [astuce du noyau](#a)
- **astuce de la log-dérivée** (*log-derivative trick*) — enseignée : [ch. B8, Apprendre la politique elle-même][s686] · exercices : B8.7
- **astuce de reparamétrisation** (*reparameterization trick*) — enseignée : [ch. 25, 25.9.2 · La structure d'un VAE][s455] · exercices : 25.5, 25.E3, 25.Q12, B5.R3
- **astuce du biais** (*bias trick*) — enseignée : [ch. 10, 10.3.3 · Les neurones modernes][s50] ; [ch. 18, 18.2.1 · Apprendre en tâtonnant][s290] · exercices : 10.2, 10.12, 10.20, 10.Q8, 18.Q2 · réutilisée : ch. 16
- **astuce du noyau** (*kernel trick*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s210] · exercices : 13.5, 13.17, 13.Q6, CP3.9
- *asymmetric numeral systems* → voir [ANS](#a)
- asynchrone → voir [synchrone, asynchrone](#s)
- *asynchronous* → voir [synchrone, asynchrone](#s)
- *atomic write* → voir [écriture atomique](#e)
- *atrous convolution* → voir [dilatation](#d)
- **attaque adversariale FGSM** — enseignée : [ch. 21, Au-delà du livre : l'attaque FGSM][s403]
- **attaque ciblée, non ciblée** (*targeted, untargeted attack*) — enseignée : [ch. 21, 21.10 · Les adversaires][s402] · exercices : 21.Q11
- attention → voir [attention additive](#a), [attention croisée](#a), [attention multi-têtes (MHA)](#a), [attention par produit scalaire](#a), [attention par produit scalaire mis à l'échelle](#a)
- **attention additive** (*additive attention*) — enseignée : [ch. B3, L'idée de Bahdanau : relire la source à chaque pas][s553] · exercices : B3.11
- **attention croisée** (*cross-attention*) — enseignée : [ch. B3, 203.3 · Masques : causal, padding ; auto-attention et attention…][s557] · exercices : B3.3, B3.14, B3.Q5
- *attention head* → voir [tête d'attention](#t)
- *attention map* → voir [carte d'attention](#c)
- *attention mask* → voir [masque d'attention](#m)
- **attention multi-têtes (MHA)** (*multi-head attention*) — enseignée : [ch. B3, 203.4 · Attention multi-têtes : projections, découpage en têtes,…][s264] · exercices : B3.3, B3.11, B3.14, B3.16, B3.28, B3.Q8
- **attention par produit scalaire** — enseignée : [ch. B3, 203.2 · Attention par produit scalaire : requêtes, clés, valeurs,…][s554] · exercices : B3.12
- **attention par produit scalaire mis à l'échelle** (*scaled dot-product attention*) — enseignée : [ch. B3, La formule][s555]
- *attention with linear biases* → voir [ALiBi](#a)
- **atténuation des biais** — enseignée : [ch. B6, D'où vient un biais][s645]
- attribut → voir [classe, objet, méthode, attribut](#c)
- **attribut appris** (*fitted attribute*) — enseignée : [ch. 15, 15.4.2 · Apprendre avec fit][s327] · exercices : 16.R1
- **attribut sensible** (*sensitive attribute, protected attribute*) — enseignée : [ch. B6, D'où vient un biais][s645] · exercices : B6.26, B6.29
- *attribute* → voir [classe, objet, méthode, attribut](#c)
- **attribution du crédit** (*credit assignment*) — enseignée : [ch. 26, 26.3.5 · La vue d'ensemble][s461] · exercices : 26.10, 26.12, 26.Q5
- **AUC** (*area under the curve, ROC*) — enseignée : [ch. 3, Au-delà du livre (1) : la courbe ROC et l'AUC][s212] · exercices : 3.11, 3.24, 3.26, 3.28, 3.E3, 8.9, 12.31, 13.11, B6.9, B6.26 (+ 1) · réutilisée : ch. B6
- **audit de qualité** — enseignée : [ch. 29, Au-delà du livre : auditer un dataset][s496] ; [projet final, PF.3 — Exploration et audit de qualité : doublons, fuites,…][s772] · exercices : PF.3
- **augmentation de données** (*data augmentation*) — enseignée : [ch. 9, Pénalité sur les poids, dropout, batchnorm][s25] · exercices : 9.9, 9.Q8, 24.8, 24.23, 24.E2, 24.Q8, 25.R1 · réutilisée : ch. 25, B1
- **auto-apprentissage** (*self-play*) — enseignée : [ch. 26, 26.10 · Deux applications][s470] · exercices : 26.37, 26.Q12, 27.R1
- **auto-attention** (*self-attention*) — enseignée : [ch. B3, Auto-attention et attention croisée][s556] · exercices : B3.1, B3.11, B3.14, B3.16, B3.E2, B3.Q2, B3.Q5, B3.Q7, B3.R2
- **auto-cohérence** (*self-consistency*) — enseignée : [ch. B4, Voter : l'auto-cohérence][s588] · exercices : B4.15
- *autodiff* → voir [différentiation automatique](#d)
- *autoencoder* → voir [autoencodeur](#a)
- **autoencodeur** (*autoencoder*) — enseignée : [ch. 25, 25.3 · Le plus simple des autoencodeurs][s153] · exercices : 25.1, 25.3, 25.8, 25.12, 25.14–15, 25.Q2, 25.Q7, 25.Q9–Q10 (+ 53)
- **autoencodeur convolutif** — enseignée : [ch. 25, 25.7 · Autoencodeurs convolutifs][s375] · exercices : 25.3, 25.24
- **autoencodeur débruiteur** (*denoising autoencoder*) — enseignée : [ch. 25, 25.3 · Le plus simple des autoencodeurs][s153] · exercices : 25.26, 25.Q10
- **autoencodeur linéaire et ACP** — enseignée : [ch. 25, 25.6 · Discussion : l'autoencodeur et l'ACP][s768] · exercices : 25.Q9
- **autoencodeur parcimonieux** (*sparse autoencoder*) — enseignée : [ch. B6, Global ou local][s643]
- **autograd (aperçu)** — enseignée : [ch. 5, Au-delà du livre (3) : la différentiation automatique][s233] · exercices : 5.11, 5.21, 20.15–16, 20.22, 21.27, 21.31, 22.19, 22.25, 22.27 (+ 9) · approfondie : ch. 20 · réutilisée : ch. 21, 23, 25, 27, 28, B8
- **autograd PyTorch** — enseignée : [ch. 20][s376] · réutilisée : ch. 27
- *automatic differentiation* → voir [différentiation automatique](#d)
- *automation bias* → voir [biais d'automatisation](#b)
- **autoreset (next-step, same-step)** — enseignée : [ch. B8, Les environnements vectorisés][s694] · exercices : B8.20, B8.27–28
- **avantage** (*advantage*) — enseignée : [ch. B8, 208.4 · Acteur-critique : fonction de valeur, erreur TD, avantage, GAE][s689] · exercices : B8.3–5, B8.19–20, B8.E1, B8.Q5, B8.Q7–Q8, B8.Q11 (+ 7)
- **average precision (AP)** (*précision moyenne*) — enseignée : [ch. 3, Au-delà du livre (2) : la courbe precision-recall et l'average…][s213] · exercices : 3.26–27, 3.29, 15.26
- *averaged perceptron* → voir [perceptron moyenné](#p)
- **avertissement** (*warning*) — enseignée : [ch. 23, 23.1.3 · Les avertissements de Python][s421] · exercices : 0A.55, 0B.37, 1.23, 4.24, 6.16, 12.18, 14.15, CP3.1, 23.12, 23.Q2 (+ 15)
- **axe** (*axis*) — enseignée : [ch. 0A, 100.8.5 · Réductions par axe, tri, `argmax`, `unique`][s92] · exercices : 0A.36, 0A.49, 0A.51, 0A.64, 3.Q7, 12.7–8, 12.22, 17.17, B2.27 (+ 79) · réutilisée : ch. 2, 7, B3
- *axis* → voir [axe](#a)

## B

- **backbone** (*tronc*) — enseignée : [ch. B1, Backbone et tête][s433] · exercices : B1.6, B1.9, B1.11, B1.13–14, B1.16, B1.20, B1.22–23, B1.Q2 (+ 8)
- **backend (de Keras)** — enseignée : [ch. 23, 23.3.3 · Installer et configurer la bibliothèque][s422] · exercices : 23.Q12, B3.27, B7.9, B7.18
- *background* → voir [fond](#f)
- *backprop* → voir [rétropropagation](#r)
- *backpropagation* → voir [rétropropagation](#r)
- *backpropagation through time* → voir [rétropropagation dans le temps (BPTT)](#r)
- *backward pass* → voir [passe avant, passe arrière](#p)
- *bag of words* → voir [sac de mots](#s)
- **bagging** (*bootstrap aggregating*) — enseignée : [ch. 14, 14.4 · Le bagging][s190] · exercices : 14.8, 14.16–18, 14.24, 14.E1, 14.Q5–Q6, 14.Q10, 15.R1 (+ 1)
- **balanced accuracy** (*accuracy équilibrée*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s205] · exercices : 3.5, 3.18–19, 15.26, B6.29
- **banc d'essai** (*benchmark*) — enseignée : [mini-projet 6][s714] · exercices : 14.27, 19.25–26, 19.Q12, 27.24
- **bandit contextuel** (*contextual bandit*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s156]
- **bandit manchot** (*multi-armed bandit*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s156] · évoquée : [ch. 1][s144] · exercices : 11.11, 11.13, 11.19, B8.R3
- **base de données vectorielle** (*vector database*) — enseignée : [ch. B4, Les bases de données vectorielles][s159] · évoquée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s260] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- *base model* → voir [modèle de base](#m)
- *base rate* → voir [prévalence, taux de base](#p)
- *base rate fallacy* → voir [erreur du taux de base](#e)
- *base value* → voir [valeur de base](#v)
- **baseline** (*modèle de référence*) — enseignée : [ch. B8, La baseline et la variance][s137] ; [ch. 13, 13.1 · Pourquoi ce chapitre ?][s138] · évoquée : [ch. 0A, 100.6.4 · Sérialiser : `json` et `pickle`][s88] · exercices : 13.E5, 13.Q1, 13.Q12, B8.7, B8.16–17, B8.Q6
- **baseline (d'un gradient de politique)** — enseignée : [ch. B8, La baseline et la variance][s137] · exercices : B8.7, B8.16–17, B8.Q6
- **batch** (*lot*) — enseignée : [ch. 19, 19.4.1 · La descente de gradient par batch][s6] · évoquée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s7] · exercices : 0A.5, 10.14, 19.25, 19.Q8, 23.5, 23.Q10, 24.11, 24.27, 24.Q2, CP6.8 (+ 11) · réutilisée : ch. B3
- *batch gradient descent* → voir [descente par batch](#d)
- *batch inference* → voir [inférence par batches, en ligne](#i)
- **batchnorm** (*normalisation par lot*) — enseignée : [ch. 20, 20.4.4 · La batchnorm][s286] ; [ch. 9, Pénalité sur les poids, dropout, batchnorm][s25] · exercices : 9.Q8, 20.4, 20.8, 20.28, 20.E2, 20.Q6, CP4.4, 24.21, 24.E3, 24.Q9 (+ 34) · réutilisée : ch. 24, 27, B1
- *Bayes' rule* → voir [règle de Bayes, théorème de Bayes](#r)
- *Bayes' theorem* → voir [règle de Bayes, théorème de Bayes](#r)
- *Bayesian* → voir [fréquentiste, bayésien](#f)
- *Bayesian linear regression* → voir [régression linéaire bayésienne](#r)
- *Bayesian optimization* → voir [optimisation bayésienne](#o)
- *Bayesian updating* → voir [mise à jour bayésienne, séquentielle](#m)
- bayésien → voir [fréquentiste, bayésien](#f)
- *bayésien naïf* → voir [Naïve Bayes (gaussien, multinomial)](#n)
- **BCa** (*bias-corrected and accelerated bootstrap*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s189] · exercices : 2.24
- **beeswarm** (*essaim d'abeilles*) — enseignée : [ch. B6, Les trois graphiques à connaître][s637] · exercices : B6.7, B6.15
- *Bellman equation* → voir [équation de Bellman](#e)
- *Bellman residual* → voir [résidu de Bellman](#r)
- **benchmark** (*banc d'essai, référence*) — enseignée : [ch. 1, 1.7 · Deep learning][s35] · exercices : 14.10, 14.27, 22.18, 27.24, 29.Q3, B4.Q8, B6.10, PF.5
- *benchmark contamination* → voir [contamination (d'un benchmark)](#c)
- **bénédiction de la non-uniformité, de la structure** (*blessing of non-uniformity*) — enseignée : [ch. 7, 7.6 · La malédiction de la dimension][s192] · exercices : 7.20, 7.Q10, 25.Q2
- *Bernoulli distribution* → voir [loi de Bernoulli](#l)
- *best checkpoint* → voir [meilleur checkpoint](#m)
- *Beta distribution* → voir [loi Beta](#l)
- **bfloat16** (*brain floating point*) — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s444] ; [ch. B4, La mémoire des poids][s445] ; [ch. B7, Accélérer sur CPU][s446] · exercices : 19.9, 24.29, B2.9, B4.1, B4.3, B4.12, B4.Q3, B4.R1, B7.1, B7.20 (+ 1)
- *bi-encoder* → voir [bi-encodeur](#b)
- **bi-encodeur** (*bi-encoder*) — enseignée : [ch. B4, La recherche dense][s547]
- **biais** (*bias*) — enseignée : [ch. 0B, 101.1.4 · Moyenne, moyenne pondérée, somme pondérée, moyenne mobile][s49] ; [ch. 10, 10.3.3 · Les neurones modernes][s50] · exercices : 10.1–2, 10.5, 10.12, 10.15, 10.20, 10.E2, 10.Q8–Q9, 10.R1 (+ 9)
- **biais (d'une pièce)** (*bias*) — enseignée : [ch. 4, 4.3 · Lancer des pièces][s219] · exercices : 4.1, 4.5–6, 4.12–14, 4.17, 4.19, 4.Q4, 4.R3 (+ 10)
- **biais (équité)** (*bias*) — enseignée : [ch. B6, 206.8 · Biais : données, labels, mesure, variables de…][s644] · exercices : B6.9, B6.E3, B6.Q10
- **biais (statistique)** (*bias*) — enseignée : [ch. 9, 9.6 · Biais et variance][s284] · exercices : 9.4, 9.8, 9.24, 9.Q9–Q11, 9.R3
- **biais d'automatisation** (*automation bias*) — enseignée : [ch. B6, La supervision humaine][s651]
- **biais d'échantillonnage** (*sampling bias, selection bias*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s189] ; [ch. 11, Au-delà du livre : le bandit manchot][s156] · exercices : 11.E4, 29.R3
- **biais d'exposition** (*exposure bias*) — enseignée : [ch. B3, Teacher forcing et décalage d'un token][s569]
- **biais d'optimisme (du gagnant)** (*optimistic bias, winner's curse*) — enseignée : [ch. 8, 8.4 · Le jeu de validation][s168] · exercices : 8.6
- **biais de fréquence (embeddings)** (*frequency bias*) — enseignée : [ch. B2, Les biais des embeddings][s549]
- **biais de labels, de mesure, de déploiement** (*label, measurement, deployment bias*) — enseignée : [ch. B6, D'où vient un biais][s645] · exercices : 2.8, 2.28, 3.15, 4.Q2, 7.R2, 9.10, 12.Q10, 13.34, B6.10, B6.Q10 (+ 5)
- **biais de longueur (d'un modèle de récompense)** (*length bias*) — enseignée : [ch. B8, GAE : une moyenne de toutes les longueurs][s691]
- **biais de prédiction** (*prediction bias*) — enseignée : [ch. B7, Pourquoi un modèle se dégrade sans changer, et que surveiller][s677]
- **biais de sélection** (*selection bias*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s383] · exercices : 20.11
- **biais inductif** (*inductive bias*) — enseignée : [ch. 11, 11.2.3 · L'optimisation][s293] ; [ch. 13, 13.2 · Paramétrique ou non paramétrique ?][s294] · exercices : 14.10, B3.28
- *bias* → voir [biais](#b), [biais (d'une pièce)](#b), [biais (statistique)](#b), [biais (équité)](#b)
- *bias correction* → voir [correction de biais](#c)
- *bias trick* → voir [astuce du biais](#a)
- *bias-corrected and accelerated bootstrap* → voir [BCa](#b)
- *bias-variance decomposition* → voir [décomposition biais-variance](#d)
- *bias-variance trade-off* → voir [compromis biais-variance](#c)
- *biased sample* → voir [échantillon biaisé](#e)
- **bibliothèque tokenizers** — enseignée : [ch. B2, 202.5 · WordPiece, Unigram, SentencePiece ; la bibliothèque…][s528] · exercices : B2.19, B2.31, B3.26
- *bidirectional RNN* → voir [RNN profond, bidirectionnel](#r)
- bidirectionnel → voir [RNN profond, bidirectionnel](#r)
- *bigram* → voir [bigramme](#b), [modèle unigramme, bigramme, trigramme](#m)
- **bigramme** (*bigram*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s162] ; [ch. 1][s144] · exercices : 1.24, 6.26, 22.24 · réutilisée : ch. 6
- *binary* → voir [classification binaire, multi-classe, multi-étiquette](#c)
- **binary relevance** (*pertinence binaire*) — enseignée : [ch. 7, 7.4.1 · Un-contre-tous (OvR)][s209] · exercices : 7.Q6
- *binomial coefficient* → voir [coefficient binomial](#c)
- **bit (d'information)** (*shannon, Sh*) — enseignée : [ch. 6, 6.3 · Le bit, une unité][s237] · exercices : 0B.16, 6.1–2, 6.5, 6.10, 6.20, 6.Q4, 6.Q6, 6.Q12, 11.1 (+ 52) · approfondie : ch. 0B · réutilisée : ch. 13, 18
- **bits et nats** — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] · exercices : 0B.16 · approfondie : ch. 0B · réutilisée : ch. 18
- *black box* → voir [boîte noire](#b)
- *blanchiment* → voir [whitening](#w)
- *blessing of non-uniformity* → voir [bénédiction de la non-uniformité, de la structure](#b)
- *blob* → voir [nuage gaussien](#n)
- bloc → voir [bloc feed-forward (FFN)](#b), [bloc Transformer](#b), [bloc Transformer pré-LN](#b)
- **bloc feed-forward (FFN)** (*position-wise feed-forward network*) — enseignée : [ch. 16, 16.1 · Pourquoi ce chapitre ?][s337]
- **bloc Transformer** (*Transformer block*) — enseignée : [ch. B3, 203.6 · Le bloc Transformer : résiduel, layer norm (pré ou post),…][s567] · exercices : B3.16
- **bloc Transformer pré-LN** — enseignée : [ch. B3, 203.6 · Le bloc Transformer : résiduel, layer norm (pré ou post),…][s567] · exercices : B3.16
- **BM25** (*Okapi BM25*) — enseignée : [ch. B4, 204.6 · Recherche d'information : découpage, BM25, recherche…][s592] · exercices : B4.5, B4.17, B4.19–20, B4.30, B4.Q9
- **boîte noire** (*black box*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s383] · exercices : 1.18, 1.23, 16.10, 20.Q12 · réutilisée : ch. 1, 9, 12, 16
- **bold driver** — enseignée : [ch. 19, 19.3.3 · Les plannings de décroissance][s358] · exercices : 19.Q7, 19.Q12
- *Boltzmann* → voir [politique softmax, de Boltzmann](#p)
- *Bonferroni correction* → voir [correction de Bonferroni](#c)
- **bonus d'entropie** (*entropy bonus*) — enseignée : [ch. B8, La baseline et la variance][s137] · exercices : B8.21
- *boolean mask* → voir [masque booléen](#m)
- **boosting** — enseignée : [ch. 14, 14.7 · Le boosting][s316] · exercices : 14.7–8, 14.20, 14.25, 14.E1, 14.E5, 14.Q10–Q11, 15.R1, CP3.5 (+ 25) · réutilisée : ch. 15, B6
- bootstrap → voir [bootstrap apparié](#b), [bootstrap stratifié (de runs)](#b), [bootstrap stratifié (par groupe)](#b)
- *bootstrap aggregating* → voir [bagging](#b)
- **bootstrap apparié** (*paired bootstrap*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s269] · exercices : 8.26, PF.8
- *bootstrap sample* → voir [bootstrap, rééchantillon](#b)
- **bootstrap stratifié (de runs)** (*stratified bootstrap*) — enseignée : [ch. B8, Rapporter plusieurs graines][s653]
- **bootstrap stratifié (par groupe)** (*stratified bootstrap*) — enseignée : [ch. B8, Rapporter plusieurs graines][s653] ; [ch. B6][s382] · exercices : B6.24
- **bootstrap, rééchantillon** (*bootstrap sample*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s189] ; [ch. 14, 14.4 · Le bagging][s190] · exercices : 2.9, 2.22–24, 2.E3, 2.Q10, 4.25, CP1.5, 8.26, 9.R3 (+ 22) · réutilisée : ch. 9, 26, B6, B7, B8
- *bootstrapping* → voir [amorçage](#a)
- *border point* → voir [point cœur, point de bord, bruit](#p)
- *bottleneck* → voir [goulot](#g)
- boucle → voir [boucle agent-environnement](#b), [boucle d'entraînement](#b), [boucle d'entraînement alternée (detach, deux optimiseurs)](#b), [boucle d'entraînement PyTorch](#b), [boucle de rétroaction](#b)
- **boucle agent-environnement** — enseignée : [ch. 26, 26.3 · La structure du RL][s459] · exercices : 11.21, 29.R2
- **boucle d'entraînement** — enseignée : [ch. 8, 8.2 · La boucle d'entraînement][s9] ; [ch. 1, 1.2.3 · Généraliser : le jeu de test][s38] ; [ch. 18][s355] · évoquée : [ch. 20, 20.1 · Pourquoi ce chapitre ?][s731] ; [ch. 23, 23.3.3 · Installer et configurer la bibliothèque][s422] · exercices : 1.16, 8.Q1, 9.20, 18.19, 19.16, 20.15, 20.23, 20.E3, CP4.7, 26.21 (+ 17) · approfondie : ch. 27 · réutilisée : ch. 5, 19, 21, 22, 25, B2, B4, B5, B8
- **boucle d'entraînement alternée (detach, deux optimiseurs)** — enseignée : [ch. 27][s770]
- **boucle d'entraînement PyTorch** — enseignée : [ch. 20, 20.1 · Pourquoi ce chapitre ?][s731] ; [checkpoint 4, 4. La boucle d'entraînement PyTorch, et ses pièges][s752] · exercices : 20.23, 20.E3, CP4.7, CP5.6, 27.12, B2.25 · réutilisée : ch. B2, B4, B5, B8
- **boucle de rétroaction** (*feedback loop*) — enseignée : [ch. B6, D'où vient un biais][s645] · exercices : 16.Q3, B6.Q10
- *boundary method* → voir [méthode à frontière](#m)
- **BPE** (*byte-pair encoding*) — enseignée : [ch. B2, 202.4 · BPE au niveau octet, pré-tokenisation et tokens spéciaux][s31] ; [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] · exercices : B2.3–4, B2.10, B2.16, B2.19, B2.E2, B2.Q4–Q6, B3.26 (+ 12) · réutilisée : ch. B3
- **BPE au niveau octet** (*byte-level BPE*) — enseignée : [ch. B2, 202.4 · BPE au niveau octet, pré-tokenisation et tokens spéciaux][s31] · exercices : B2.4, B2.18, B2.Q5–Q6, B3.26
- BPTT → voir [rétropropagation dans le temps (BPTT)](#r)
- *Bradley-Terry model* → voir [modèle de Bradley-Terry](#m)
- *brain floating point* → voir [bfloat16](#b)
- *branch* → voir [branche](#b)
- **branche** (*branch*) — enseignée : [ch. 0A, 100.11.2 · git : branches (aperçu)][s101] · exercices : 0A.12, 0A.E5, 0A.Q12, 13.Q7, 15.28, 19.19, 20.7, 24.28, 24.Q12, B4.6 (+ 3) · approfondie : ch. 24 · réutilisée : ch. B7
- *breadth-first search* → voir [recherche en largeur](#r)
- *breakpoint* → voir [point d'arrêt, débogueur](#p)
- *Brier score* → voir [score de Brier](#s)
- **brisure de symétrie** (*symmetry breaking*) — enseignée : [ch. 16][s341] · exercices : 18.11
- **broadcasting** (*diffusion*) — enseignée : [ch. 0A, 100.8.4 · Broadcasting][s91] · exercices : 0A.8, 0A.49, 0A.52, 0A.Q11, 2.25, 4.14, 7.13, 10.14, 13.28, 16.3 (+ 6) · réutilisée : ch. 0B, 7, 16, 21
- bruit → voir [point cœur, point de bord, bruit](#p)
- **bruit de Monte-Carlo** (*Monte Carlo error*) — enseignée : [ch. 2][s178]
- **budget de pas (d'environnement)** (*environment step budget*) — enseignée : [ch. B8][s702] · exercices : B8.28
- *buffer* → voir [tampon (d'un optimiseur)](#t)
- **buffer (d'un module)** — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s3] · exercices : 23.Q11, 26.31, 29.R2, B1.4, B3.30, B4.1, B5.19, B5.21, B5.23, B7.1 (+ 1)
- *byte fallback* → voir [repli sur les octets](#r)
- *byte-level BPE* → voir [BPE au niveau octet](#b)
- *byte-pair encoding* → voir [BPE](#b)
- *bytes per token* → voir [compression (d'un tokenizer)](#c)

## C

- *C* → voir [marge souple, paramètre C](#m)
- *C order* → voir [ordre C, ligne par ligne](#o)
- C2PA → voir [métadonnées de provenance, C2PA](#m)
- **CACE** (*Changing Anything Changes Everything*) — enseignée : [ch. B7, La dette technique du ML][s679] · exercices : B7.8
- *calibrated* → voir [calibration, calibré](#c)
- **calibration de Platt, régression isotone** (*Platt scaling, isotonic regression*) — enseignée : [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s211]
- **calibration par groupe** (*calibration within groups*) — enseignée : [ch. B6, Les critères][s648]
- *calibration within groups* → voir [calibration par groupe](#c)
- **calibration, calibré** (*calibrated*) — enseignée : [ch. 3, Au-delà du livre (3) : la calibration des probabilités][s214] · exercices : 3.28, 6.22, 15.10, 15.E4, B6.6, B6.Q11, PF.8 · réutilisée : ch. 6, B6
- calibré → voir [calibration, calibré](#c)
- *call stack* → voir [pile d'appels](#p)
- **callback** — enseignée : [ch. 23, 23.10 · Callbacks][s428] · exercices : 0A.Q7, 23.18, 23.25, PF.6
- **canari** (*canary*) — enseignée : [ch. B7, Agir : alerter, réentraîner, revenir en arrière][s579] ; [ch. B3][s562] · exercices : B3.10
- *canary* → voir [canari](#c), [déploiement canari, en ombre](#d)
- **candidat (d'une recherche)** (*candidate*) — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s328] ; [ch. 22, 22.7.2 · Le LSTM][s331] · exercices : 15.1, 15.5, 15.18–19, 15.22–23, 15.28, 22.4, 22.17, 22.Q8 (+ 18)
- *candidate* → voir [candidat (d'une recherche)](#c)
- *canonical composition* → voir [NFC, NFD, NFKC](#n)
- *canonical decomposition* → voir [NFC, NFD, NFKC](#n)
- **capacité** (*capacity, representational power*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s47] · exercices : 1.Q10, 9.18, 25.16, 25.22, 26.7, 26.31, B8.13, B8.15
- *capacity* → voir [capacité](#c)
- **caractère combinant** (*combining character*) — enseignée : [ch. B2, Un caractère affiché, plusieurs points de code][s524]
- **cardinal** (*cardinality*) — enseignée : [ch. 0B, 101.1.6 · Ensembles : union, intersection, complémentaire, cardinal][s73] · exercices : 0B.5
- *cardinality* → voir [cardinal](#c)
- *carried state* → voir [état porté](#e)
- **carrousel à erreur constante** (*constant error carousel, CEC*) — enseignée : [ch. 22, 22.7.2 · Le LSTM][s331] · exercices : 22.11
- **CART** (*classification and regression trees*) — enseignée : [ch. 13, 13.5 · Les arbres de décision][s169] · exercices : 0B.11, 0B.33, 3.R1, 6.R2, 8.21, 13.24, 21.27, CP6.5, B3.24, B6.4 (+ 29) · approfondie : ch. 21 · réutilisée : ch. 28, B1
- carte → voir [carte d'activation](#c), [carte d'attention](#c), [carte de saillance](#c)
- **carte d'activation** (*feature map, activation map, « carte de caractéristiques »*) — enseignée : [ch. 21, 21.7 · Une couche de convolution][s389] · évoquée : [ch. 28, 28.2.1 · Choisir un réseau][s390] · exercices : 21.Q2, 21.Q8, 28.Q3, 28.Q7, B1.1, B1.R3, B5.Q5
- **carte d'attention** (*attention map*) — enseignée : [ch. B3, L'idée de Bahdanau : relire la source à chaque pas][s553] · exercices : B3.1–2, B3.13, B3.24, B3.Q5
- **carte de saillance** (*saliency map*) — enseignée : [ch. B6, Le gradient comme explication][s638] · exercices : B6.19, B6.Q1, B6.Q9, B6.R1
- *cascading* → voir [randomisation en cascade, indépendante](#r)
- *catastrophic cancellation* → voir [annulation catastrophique](#a)
- *catastrophic forgetting* → voir [oubli catastrophique](#o)
- *categorical distribution* → voir [loi catégorielle, multinoulli](#l)
- *causal language model* → voir [modèle de langue causal (auto-régressif)](#m)
- *causal mask* → voir [masque causal](#m)
- CBOW → voir [skip-gram, CBOW](#s)
- *CEC* → voir [carrousel à erreur constante](#c)
- *cell state* → voir [mémoire de cellule](#m)
- **cellule RNN et BPTT** — enseignée : [ch. 22, 22.4 · La structure d'une cellule RNN][s410]
- **cellules de Voronoï** (*Voronoi cells*) — enseignée : [ch. 7, Avec des labels : le centroïde le plus proche][s256] ; [ch. 13, 13.3 · Les k plus proches voisins][s257]
- *central difference* → voir [pente centrée](#p)
- *centroid* → voir [centroïde](#c)
- **centroïde** (*centroid*) — enseignée : [ch. 7, Avec des labels : le centroïde le plus proche][s256] · exercices : 7.7, 7.11, 7.14–16, 7.23, 7.30, 8.26–27, 10.R2 (+ 2) · réutilisée : ch. 8
- CFG → voir [guidance sans classifieur (CFG)](#g)
- *chain rule* → voir [règle de la chaîne](#r)
- **chain-of-thought** (*chaîne de pensée, raisonnement pas à pas*) — enseignée : [ch. B4, Le raisonnement pas à pas][s587] · exercices : B4.Q6
- **chaîne d'arrêt** (*stop string*) — enseignée : [ch. B4, S'arrêter][s584] · exercices : B4.14
- **chaîne de caractères** (*string*) — enseignée : [ch. 0A, 100.2.2 · Chaînes de caractères et f-strings][s68] · exercices : 0A.2, 0A.14–15, 0A.Q3
- *chaîne de pensée* → voir [chain-of-thought](#c)
- **champ de vitesses** (*velocity field*) — enseignée : [ch. B5, Flow matching et rectified flow][s627] · exercices : B5.Q11
- **champ récepteur** (*receptive field*) — enseignée : [ch. 21, 21.2.4 · Le champ récepteur local][s388] · exercices : 21.5, 21.Q2, 21.Q8, 22.R1, CP5.2, 28.12, 28.17, 28.R3
- *Changing Anything Changes Everything* → voir [CACE](#c)
- *channels first* → voir [channels_first, channels_last](#c)
- *channels last* → voir [channels_first, channels_last](#c)
- **channels_first, channels_last** (*channels first, channels last*) — enseignée : [ch. 23, 23.3.4 · La forme des tenseurs d'images][s425] · exercices : 23.Q6
- channels_last → voir [channels_first, channels_last](#c)
- *chat model* → voir [modèle de chat](#m)
- **chat template** (*modèle de conversation*) — enseignée : [ch. B4, 204.3 · Génération : generate(), décodage, arrêt, streaming ; chat…][s581] · exercices : B4.4, B4.12, B4.14, B4.28–29, B4.Q1, B4.Q4
- **chat templates** — enseignée : [ch. B4, 204.3 · Génération : generate(), décodage, arrêt, streaming ; chat…][s581] · exercices : B4.4, B4.12–14, B4.Q4–Q5, B4.R1
- **ChatML** — enseignée : [ch. B4, Rôles et chat template][s582] · exercices : B4.Q4
- *checkerboard artifacts* → voir [artefacts en damier](#a)
- **checkpoint (d'entraînement)** (*training checkpoint*) — enseignée : [ch. 23, 23.10.1 · Checkpoints][s380] ; [ch. 20, Au-delà du livre : PyTorch en pratique][s3] · exercices : 20.27, 23.8, 23.23, 23.25, 27.27, B1.19, B4.11, B5.30, B7.R2, B8.28 (+ 1) · réutilisée : ch. 24, 27, B1, B7
- *checksum* → voir [somme de contrôle](#s)
- **chemin de régularisation** (*regularization path*) — enseignée : [ch. 9, 9.5 · La régularisation][s276] · exercices : 9.19
- **chemin relatif / absolu** (*relative / absolute path*) — enseignée : [ch. 0A, 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest…][s65] · exercices : 0A.Q1
- *child* → voir [parent, enfant ; ancêtre, descendant](#p)
- **choix d'un optimiseur** — enseignée : [ch. 19, 19.6 · Choisir un optimiseur][s265] · exercices : 19.11, 19.Q12
- **choix du pas h** — enseignée : [ch. 5, Au-delà du livre (2) : les arrondis et le choix du pas][s228]
- **chunk** (*morceau*) — enseignée : [ch. B4, Découper][s591] · exercices : B4.17, B4.19–21, B4.23–24, B4.R3
- CI → voir [intégration continue (CI)](#i)
- *class* → voir [classe, objet, méthode, attribut](#c), [classifieur, classe](#c)
- *class prototype* → voir [prototype de classe](#p)
- *class weighting* → voir [pondération des classes](#p)
- classe → voir [classifieur, classe](#c), [classe positive, négative](#c), [classes : __init__, attributs, méthodes](#c), [classes déséquilibrées : class_weight, rééchantillonnage](#c)
- **classe positive, négative** (*positive class, negative class*) — enseignée : [ch. 3, 3.7.1 · Classer des échantillons][s204] · exercices : 3.16–17, 3.19–20, 3.25, 3.27, 10.R3, CP2.13, 27.R2
- **classe, objet, méthode, attribut** (*class, object, method, attribute*) — enseignée : [ch. 0A, 100.7 · Classes et objets][s84] · exercices : 0A.47–48, 0A.Q9, 3.Q9, 7.2, 7.16, 7.26, 7.Q4, 10.21, 15.26 (+ 17) · approfondie : ch. 15 · réutilisée : ch. 7, 8, 10, 12, 15, 20, 24, 26
- **classes : __init__, attributs, méthodes** — enseignée : [ch. 0A, 100.3.1 · Listes : indices, tranches, méthodes, mutabilité][s70] · exercices : 0A.2, 0A.9, 0A.16, 0A.E2, 0A.Q4
- **classes déséquilibrées : class_weight, rééchantillonnage** — enseignée : [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s211] · exercices : 15.26
- classification → voir [classification / régression](#c), [classification binaire et multi-classe](#c), [classification binaire, multi-classe, multi-étiquette](#c), [classification de textes](#c)
- *classification / regression* → voir [classification / régression](#c)
- **classification / régression** (*classification / regression*) — enseignée : [ch. 1, 1.3.1 · Classification][s145] ; [ch. 7, 7.2.1 · Classification binaire en 2D][s127] · exercices : 1.8, 1.12, 1.25, 1.Q5, 1.Q7, 7.E2, 7.Q1–Q4 (+ 44) · approfondie : ch. 13, B4 · réutilisée : ch. 3, 8, 13, 14, 15, 18, B1, B4
- *classification and regression trees* → voir [CART](#c)
- **classification binaire et multi-classe** — enseignée : [ch. 7, 7.4 · Plusieurs classes avec des classifieurs binaires][s736]
- **classification binaire, multi-classe, multi-étiquette** (*binary, multi-class, multi-label classification*) — enseignée : [ch. 7, 7.2.1 · Classification binaire en 2D][s127] · exercices : 14.4
- **classification de textes** — enseignée : [ch. B4, 204.9 · Classer des textes : encodeur fine-tuné, LLM en zéro-shot…][s782]
- *classification token* → voir [token `[CLS]`](#t)
- *classifier* → voir [classifieur, classe](#c)
- *classifier guidance* → voir [guidance par classifieur](#g)
- *classifier-free guidance* → voir [guidance sans classifieur (CFG)](#g)
- **classifieur du centroïde le plus proche** (*nearest centroid classifier*) — enseignée : [ch. 7, Avec des labels : le centroïde le plus proche][s256] · exercices : 7.11, 7.14 · réutilisée : ch. 8
- **classifieur paramétrique, non paramétrique** (*parametric, non-parametric classifier*) — enseignée : [ch. 13, 13.2 · Paramétrique ou non paramétrique ?][s294] · exercices : 13.Q2
- **classifieur, classe** (*classifier, class*) — enseignée : [ch. 13, Au-delà du livre : la régression logistique, le classifieur…][s109] ; [ch. 1, 1.2.3 · Généraliser : le jeu de test][s38] · exercices : 7.1, 7.14, 10.R3, 13.12, 13.33, 13.36, 13.Q1, 13.Q12, 14.11, 14.14 (+ 60) · approfondie : ch. 7, B5 · réutilisée : ch. 8, 15, 27, 29, B7
- clé → voir [requête, clé, valeur](#r)
- CLIP → voir [échelle des logits (CLIP)](#e)
- *clip fraction* → voir [fraction clippée](#f)
- **CLIP score** — enseignée : [ch. B5, La mémorisation][s578]
- *clipped objective* → voir [objectif clippé, PPO](#o)
- **clone (d'un estimateur)** — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s263] · exercices : 8.13, 8.16, 8.22, 9.16, 14.12, 14.17, 14.24–25, 15.9, B7.25 (+ 6) · approfondie : ch. 0A, 15 · réutilisée : ch. 14
- *closed form* → voir [forme fermée (du processus direct)](#f)
- *closed-world assumption* → voir [hypothèse du monde clos](#h)
- *closure* → voir [fermeture](#f)
- **clustering** (*partitionnement, regroupement*) — enseignée : [ch. 1, 1.4.1 · Clustering][s148] · exercices : 1.21, 1.Q8, 2.R1, 7.17–18, 7.31, 7.R1, 15.13
- **clustering par densité** (*density-based clustering*) — enseignée : [ch. 7, Quand k-means échoue, et le clustering par densité][s261] · exercices : 7.18, 15.13
- **CNN PyTorch (nn.Conv2d, nn.MaxPool2d)** — enseignée : [ch. 21][s756]
- **CNN-LSTM** — enseignée : [ch. 22, 22.8.1 · Une ou plusieurs entrées et sorties][s416] · exercices : 22.8, 22.28
- *co-occurrence matrix* → voir [matrice de co-occurrences](#m)
- **coalition** — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s631] · exercices : B6.1–2, B6.7, B6.11, B6.14, B6.Q4
- *coarse-to-fine search* → voir [recherche multirésolution](#r)
- **codage arithmétique** (*arithmetic coding*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s162]
- *codage disjonctif* → voir [one-hot](#o)
- code → voir [code à longueur variable, code adaptatif](#c), [code de colle](#c), [code de Huffman](#c), [code de longueur fixe](#c), [code de sortie](#c), [code de statut](#c), [code préfixe](#c), [codes adaptatifs](#c), [codes de longueur fixe](#c)
- **code à longueur variable, code adaptatif** (*variable-length code*) — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s240] · exercices : 6.Q7
- code adaptatif → voir [code à longueur variable, code adaptatif](#c)
- **code de colle** (*glue code*) — enseignée : [ch. B7, La dette technique du ML][s679] · exercices : B7.8
- **code de Huffman** (*Huffman code*) — enseignée : [ch. 6, Le code de Huffman][s241] · exercices : 6.6, 6.10, 6.23–25, 6.27, CP1.11, B2.20, B2.R3
- **code de longueur fixe** (*fixed-length code*) — enseignée : [ch. 6, 6.5 · La taille d'un événement][s239] · exercices : 6.2, 6.6, 6.10, 6.Q6–Q7, CP1.11, B2.R3
- **code de sortie** (*exit code*) — enseignée : [ch. B7, Les codes de sortie][s658] · exercices : B5.30, B7.10, B7.13, B7.15, B7.Q3–Q4
- **code de statut** (*status code*) — enseignée : [ch. B7, En ligne ou par batches, et une API HTTP][s670]
- *code point* → voir [point de code](#p)
- **code préfixe** (*prefix code, prefix-free code*) — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s240] · exercices : 6.4, 6.Q7
- *code review* → voir [revue de code](#r)
- **code, variables latentes** (*latent variables*) — enseignée : [ch. 25, 25.5.1 · Les variables latentes de près][s448] · exercices : 0A.Q5, 0B.35, 6.4, 6.6, 6.18, 6.24–25, 6.Q7, 6.Q10, 15.12 (+ 42) · approfondie : ch. 6 · réutilisée : ch. 6
- **codes adaptatifs** — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s240] · exercices : 6.Q7
- **codes de longueur fixe** — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s240]
- coefficient → voir [coefficient binomial](#c), [coefficient de détermination ($R^2$)](#c), [coefficient de silhouette](#c)
- **coefficient binomial** (*binomial coefficient, « n choose k »*) — enseignée : [ch. 0B, 101.1.7 · Dénombrement : principe multiplicatif, factorielle,…][s13] · exercices : 0B.12, 7.1, 15.4 · réutilisée : ch. 7, 14, 15
- **coefficient de détermination ($R^2$)** (*coefficient of determination*) — enseignée : [ch. 8][s267]
- **coefficient de silhouette** (*silhouette coefficient*) — enseignée : [ch. 7, Combien de clusters ?][s259] · exercices : 7.28
- *coefficient of determination* → voir [coefficient de détermination ($R^2$)](#c)
- **cognition incarnée** (*embodied cognition*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s51]
- cohérence → voir [précision locale, absence, cohérence](#p)
- col → voir [point selle, col](#p)
- **Colab / Jupyter : cellules, Markdown, noyau, « Tout exécuter »** — enseignée : [ch. 0A, 100.1.1 · Cellules, noyau, ordre d'exécution et « Tout exécuter »][s64] · exercices : 0A.10, 0A.E4, 0A.Q1, 21.24, 22.21, 23.17, 24.19, B1.18, B2.18, B3.17 (+ 2)
- *cold start* → voir [démarrage à froid](#d)
- *collapse of a linear network* → voir [effondrement (d'un réseau linéaire)](#e)
- **colonne indicatrice (de valeur manquante)** (*missing indicator*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s304] · exercices : 12.10
- *column transformer* → voir [ColumnTransformer](#c)
- **ColumnTransformer** (*column transformer*) — enseignée : [ch. 15, Au-delà du livre : les données hétérogènes][s334] · exercices : 15.8, 15.28, 15.31, 15.E3, CP3.2, PF.5 · réutilisée : ch. B6
- *combining character* → voir [caractère combinant](#c)
- **commit** — enseignée : [ch. 0A, 100.11.1 · git : `status`, `add`, `commit`, `push`, `pull` ;…][s100] · exercices : 0A.10, 0A.E5, 11.25, 14.27, 24.15, 26.34, 29.10, B1.11, B4.29, B7.14 (+ 9) · réutilisée : ch. B7
- **comparaison appariée** (*paired comparison*) — enseignée : [ch. 8][s267]
- **comparaisons multiples** (*multiple comparisons*) — enseignée : [ch. B6, Des écarts avec leurs intervalles][s649] · exercices : B6.10, B6.25
- *compatibility normalization* → voir [NFC, NFD, NFKC](#n)
- **compilation (d'un modèle)** (*torch.compile*) — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s444] · exercices : 24.7, 24.29
- complet → voir [fine-tuning partiel, complet](#f)
- *completeness* → voir [complétude](#c)
- **complétude** (*completeness*) — enseignée : [ch. B6, Les gradients intégrés][s640] · exercices : B6.3, B6.18–19
- *component* → voir [composante](#c)
- **composante** (*component, entry*) — enseignée : [ch. 0B, 101.3.1 · Composantes, somme, produit par un scalaire][s113] · exercices : 0B.18, 0B.38, 0B.40, 0B.42, 1.R1, 2.5, 5.2, 5.18, 12.24, 25.19 (+ 34)
- **composante principale** (*principal component*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s199] · exercices : 12.7–8, 12.Q9, 25.8, 25.Q9
- **composition de fonctions** — enseignée : [ch. 0B, 101.2.7 · Composition de fonctions][s727] · exercices : 0B.23, 0B.47, 0B.Q7, 0B.R3
- *comprehension* → voir [compréhension](#c)
- **compréhension** (*comprehension*) — enseignée : [ch. 0A, 100.4.3 · Compréhensions de listes, de dictionnaires et d'ensembles][s75] · exercices : 0A.3, 0A.22, 0A.51, 0A.Q5, 0B.38–39, 0B.42–43, 0B.R1, 1.17 (+ 1) · réutilisée : ch. 0B
- **compression (d'un tokenizer)** (*bytes per token*) — enseignée : [ch. B2, Un algorithme de compression devenu tokenizer][s251] · exercices : 6.19, 6.24, 6.Q10, 25.9–10, B2.13, B2.20–21, B2.31, B2.R3 (+ 5) · réutilisée : ch. 6
- **compression avec ou sans perte** (*lossy, lossless compression*) — enseignée : [ch. 25, 25.2.1 · Avec ou sans perte][s451]
- *compression ratio* → voir [taux de compression](#t)
- **compression sans perte** (*lossless compression*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s162] · exercices : B2.31
- **compromis biais-variance** (*bias-variance trade-off*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s285] · exercices : 9.8, 9.11, 9.25, 9.E1, 9.E5
- **comptage des paramètres** — enseignée : [ch. 16][s341] · réutilisée : ch. 21, 22, 23
- *computational graph* → voir [graphe de calcul](#g)
- **concentration des distances** (*distance concentration*) — enseignée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s260] · exercices : 7.20, 7.30
- **conception de la récompense** — enseignée : [ch. 26, 26.3.7 · Les récompenses][s154]
- *condition dropout* → voir [abandon du label (guidance sans classifieur)](#a)
- *condition number* → voir [conditionnement d'une matrice](#c)
- **condition vide** (*null condition*) — enseignée : [ch. B5, Un bloc résiduel conditionné par le pas][s619]
- *conditional entropy* → voir [entropie conditionnelle](#e)
- *conditional GAN* → voir [GAN conditionnel](#g)
- *conditional independence* → voir [indépendance conditionnelle](#i)
- *conditional probability* → voir [probabilité conditionnelle](#p)
- *conditioning* → voir [conditionnement](#c)
- **conditionnement** (*conditioning*) — enseignée : [ch. 9, Les moindres carrés (au-delà du livre)][s232] ; [ch. 5][s229] · exercices : 11.7, 19.7, 19.R3, 27.25 · approfondie : ch. 11 · réutilisée : ch. 26
- **conditionnement d'une matrice** (*condition number*) — enseignée : [ch. 9, Les moindres carrés (au-delà du livre)][s232]
- **conditionnement opérant** (*operant conditioning*) — enseignée : [ch. 11, 11.7 · Le conditionnement opérant][s301] · exercices : 11.7 · réutilisée : ch. 26
- *Condorcet's jury theorem* → voir [théorème du jury](#t)
- *confidence interval* → voir [intervalle de confiance](#i)
- **confidentialité différentielle (DP-SGD)** (*differential privacy*) — enseignée : [ch. B3][s562] · exercices : B3.10
- **configuration typée** (*typed configuration*) — enseignée : [ch. B7, 207.3 · Configuration : dataclasses, YAML ou JSON, surcharges,…][s661] · exercices : B7.11
- *confirmatory test* → voir [test de confirmation](#t)
- *confounder* → voir [variable de confusion](#v)
- *confounding variable* → voir [variable de confusion](#v)
- *confusion matrix* → voir [matrice de confusion](#m)
- *conjugate prior* → voir [prior conjugué](#p)
- **connectome** — enseignée : [ch. 10, 10.2 · Les vrais neurones][s51] · exercices : 10.Q3
- connexion → voir [connexion de saut](#c), [connexion persistante](#c), [connexion résiduelle](#c), [connexions résiduelles](#c)
- **connexion de saut** (*skip connection*) — enseignée : [ch. B5, Le U-Net et ses connexions de saut][s618] · exercices : B5.Q6
- **connexion persistante** (*keep-alive*) — enseignée : [ch. B7][s665]
- **connexion résiduelle** (*residual connection*) — enseignée : [ch. 20, Au-delà du livre : les connexions résiduelles][s377] ; [ch. 24, 24.6 · L'API fonctionnelle][s171] ; [ch. B3, Deux sous-couches][s378] · exercices : 20.7, 20.30, B3.R3
- **connexions résiduelles** — enseignée : [ch. 20, Au-delà du livre : les connexions résiduelles][s377] ; [ch. 21, 21.9.1 · VGG16][s405] · exercices : 20.7, 20.30, 20.E5, 21.Q12, 24.10, B3.16, B3.25, B3.Q8, B3.R3, B5.Q6 · réutilisée : ch. B3
- **conservation de la variance** — enseignée : [ch. 16][s341]
- *consistency* → voir [précision locale, absence, cohérence](#p)
- *consistency model* → voir [modèle de cohérence](#m)
- *constant error carousel* → voir [carrousel à erreur constante](#c)
- *constrained decoding* → voir [génération contrainte](#g)
- *container* → voir [conteneur, image](#c)
- **contamination (d'un benchmark)** (*benchmark contamination*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s269] · exercices : B4.21, B4.Q8
- **conteneur, image** (*container*) — enseignée : [ch. B7, 207.9 · Conteneurs et déploiement : Dockerfile, variables…][s673] · exercices : B7.24
- *content credentials* → voir [métadonnées de provenance, C2PA](#m)
- *content image* → voir [image de contenu, référence de style, image générée](#i)
- *content loss* → voir [loss de contenu](#l)
- *context distribution smoothing* → voir [lissage des contextes](#l)
- *context manager* → voir [gestionnaire de contexte](#g)
- *context vector* → voir [vecteur de contexte](#v)
- **contexte global, contexte local** (*global context, local context*) — enseignée : [ch. 6, Le contexte][s238] · exercices : 6.26–27, 6.Q3
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
- **contraste des distances** (*relative contrast*) — enseignée : [ch. 2][s178]
- *contrastive learning* → voir [apprentissage contrastif](#a)
- **contribution marginale** (*marginal contribution*) — enseignée : [ch. B6, Partager un gain entre joueurs][s632] · exercices : B6.2
- convention → voir [convention AD (ou DA)](#c), [convention des poids](#c), [convention W (n_in, n_out)](#c)
- **convention AD (ou DA)** (*weight naming convention*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s253] · exercices : 10.15, 10.Q9
- **convention des poids** — enseignée : [ch. 10, 10.3.1 · Le perceptron][s253]
- **convention W (n_in, n_out)** — enseignée : [ch. 16][s341]
- convolution → voir [convolution 1 × 1](#c), [convolution 1D](#c), [convolution 2D (corrélation croisée)](#c), [convolution par canal](#c), [convolution séparable](#c), [convolution transposée](#c)
- **convolution 1 × 1** (*pointwise convolution*) — enseignée : [ch. 21, 21.6 · Les convolutions 1 × 1][s394] · exercices : 21.1, 21.13, 21.20, 21.E3, 21.Q7, 24.28, B1.4
- **convolution 1D** (*1D convolution*) — enseignée : [ch. 21, 21.5 · La convolution 1D][s395] · exercices : 21.7, 21.Q7, 22.R1, CP5.12
- **convolution 2D (corrélation croisée)** — enseignée : [ch. 21, 21.3 · La convolution][s386] · exercices : 24.Q12
- **convolution par canal** (*depthwise convolution*) — enseignée : [ch. 21, 21.4 · La convolution en plusieurs dimensions][s393] · exercices : 21.15, 21.Q6
- **convolution séparable** (*depthwise separable convolution*) — enseignée : [ch. 21, 21.3 · La convolution][s386]
- **convolution transposée** (*transposed convolution, fractionally strided convolution*) — enseignée : [ch. 21, 21.8 · La convolution transposée][s392] · exercices : 21.3, 21.6–7, 21.30, 21.Q9, 25.4, 25.27, 25.Q10, 27.1, 27.21 (+ 3) · réutilisée : ch. 25, 27, B5
- *cooperative game* → voir [jeu coopératif](#j)
- *coordinate descent* → voir [descente de coordonnées](#d)
- *core point* → voir [point cœur, point de bord, bruit](#p)
- *corner* → voir [point anguleux](#p)
- **correction de biais** (*bias correction*) — enseignée : [ch. 19, 19.5.5 · Adam][s367] · exercices : 19.5, 19.8, 19.12, 19.18, B8.R2
- **correction de Bonferroni** (*Bonferroni correction*) — enseignée : [ch. B7, Le test de Kolmogorov-Smirnov][s676]
- **corrélation (de Pearson)** (*Pearson correlation coefficient*) — enseignée : [ch. 2, 2.8.2 · La corrélation][s197] · exercices : 2.7–8, 2.10, 2.26–28, 2.E4, 2.Q12, CP1.13, 21.Q3 (+ 22) · approfondie : ch. 21 · réutilisée : ch. 8, 12
- **corrélation croisée** (*cross-correlation*) — enseignée : [ch. 21, 21.3 · La convolution][s386] · exercices : 21.2, 21.7, 21.15, 21.Q3, CP5.12
- **corrélation de rangs (de Spearman)** (*Spearman rank correlation*) — enseignée : [ch. B6, Les tests de santé][s642] · exercices : B6.22, B6.Q9
- *correlation matrix* → voir [matrice de covariance, de corrélation](#m)
- *correspondance exacte* → voir [exact match](#e)
- **corruption de segments** (*span corruption*) — enseignée : [ch. B3, L'encodeur-décodeur : T5 et l'attention croisée][s550]
- *cosine annealing* → voir [planning en cosinus](#p)
- *cosine noise schedule* → voir [planning linéaire, planning cosinus (de bruit)](#p)
- *cosine schedule* → voir [planning en cosinus](#p)
- *cosine similarity* → voir [similarité cosinus](#s)
- **cosinus (aperçu)** — enseignée : [ch. 0B, 101.2.6 · Cosinus (aperçu)][s112] ; [ch. B2, Le cosinus][s116] · exercices : 0B.7, 0B.18, 0B.36, 0B.39, 0B.41, 0B.E2, 0B.Q8, 19.19, 19.27, 19.E3 (+ 38) · approfondie : ch. 19, B5 · réutilisée : ch. 19, B1, B2, B3
- **cote** (*odds*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s218] · exercices : 4.2–3, 4.8, 4.12, B6.6
- **couche** (*layer*) — enseignée : [ch. 1, 1.7 · Deep learning][s35] · exercices : 1.3, 10.5, 16.3, 16.13, 16.17, 16.22–24, 17.23, 17.E1 (+ 38) · approfondie : ch. 20 · réutilisée : ch. 20, 21, 22, 25, 27
- **couche appliquée à chaque pas** (*time-distributed layer*) — enseignée : [ch. 24, 24.5.10 · Une couche appliquée à chaque pas de temps][s443]
- **couche cachée** (*hidden layer*) — enseignée : [ch. 16, 16.1 · Pourquoi ce chapitre ?][s337] · exercices : 16.1–3, 16.6, 16.10, 16.17–19, 16.24, 16.Q9 (+ 16)
- **couche d'entrée** (*input layer*) — enseignée : [ch. 20, 20.3 · Les couches d'entrée et de sortie][s372] · évoquée : [ch. 16, 16.1 · Pourquoi ce chapitre ?][s337] · exercices : 18.Q5, 20.30, 20.Q3, 27.20, 27.Q8
- **couche de sortie** (*output layer*) — enseignée : [ch. 20, 20.3.2 · La couche de sortie][s373] · exercices : 17.Q1, 18.19, 18.21, 18.Q7, 18.Q9, 20.2, 20.30, 20.Q3, 24.16, 28.7 (+ 7)
- **couche dense** (*dense layer, fully connected layer, couche pleine, couche entièrement connectée*) — enseignée : [ch. 1, 1.7 · Deep learning][s35] · exercices : 0B.30, 1.3, 16.13, 16.Q10, 17.8, 17.Q1, 18.7, 20.Q4, 21.4, CP5.10 (+ 45) · réutilisée : ch. 20, 21, 22, 25
- *couche entièrement connectée* → voir [couche dense](#c)
- **couche partagée** (*shared layer*) — enseignée : [ch. 24, 24.5.10 · Une couche appliquée à chaque pas de temps][s443] · exercices : 24.28
- **couche utilitaire** (*utility layer*) — enseignée : [ch. 20, 20.4.8 · Les autres couches utilitaires][s374]
- *counterfactual explanation* → voir [explication contrefactuelle](#e)
- courbe → voir [courbe ALE](#c), [courbe continue, lisse, univoque](#c), [courbe d'apprentissage](#c), [courbe de validation](#c), [courbe ICE](#c), [courbe precision-recall](#c), [courbe ROC](#c), [courbes d'apprentissage](#c), [courbes de validation](#c)
- **courbe ALE** (*accumulated local effects*) — enseignée : [ch. B6, Les courbes ICE][s630]
- **courbe continue, lisse, univoque** (*continuous, smooth, single-valued curve*) — enseignée : [ch. 5, 5.2 · Introduction : fonctions, courbes et surfaces][s224]
- **courbe d'apprentissage** (*learning curve*) — enseignée : [ch. 9, Courbes de validation et courbes d'apprentissage (au-delà du livre)][s283] · exercices : 9.21, 26.23, B1.20, B1.Q8
- **courbe de validation** (*validation curve*) — enseignée : [ch. 9, Courbes de validation et courbes d'apprentissage (au-delà du livre)][s283] · exercices : 9.20, 14.20, 14.26, CP3.8, 24.14
- **courbe ICE** (*individual conditional expectation*) — enseignée : [ch. B6, Les courbes ICE][s630] · exercices : B6.13, B6.Q2
- **courbe precision-recall** (*precision-recall curve*) — enseignée : [ch. 3, Au-delà du livre (2) : la courbe precision-recall et l'average…][s213] · exercices : 3.26, 15.26
- **courbe ROC** (*ROC curve*) — enseignée : [ch. 3, Au-delà du livre (1) : la courbe ROC et l'AUC][s212] · exercices : 3.11, 3.24, 3.E3, CP1.7 · réutilisée : ch. 15
- **courbes d'apprentissage** — enseignée : [ch. 9, Courbes de validation et courbes d'apprentissage (au-delà du livre)][s283] · exercices : 9.9, 9.21, CP2.7, 18.21, 22.26, 26.22–23, B1.21
- **courbes de validation** — enseignée : [ch. 9, Courbes de validation et courbes d'apprentissage (au-delà du livre)][s283] · exercices : 9.18, 14.21, 20.29, B3.21
- courbure → voir [dérivée seconde, courbure](#d)
- **coût quadratique et FlashAttention** — enseignée : [ch. B3, 203.10 · Coût et passage à l'échelle : complexité quadratique,…][s577] · exercices : B3.8, B3.10, B3.27–28, B3.E2, B3.Q12
- **couverture (d'un intervalle)** (*coverage*) — enseignée : [ch. 2][s178] · exercices : 2.23
- **covariance** — enseignée : [ch. 2, 2.8 · Covariance et corrélation][s196] · exercices : 2.7–8, 2.26, 2.28–29, 2.32, 2.Q12, CP1.4, 7.15, 12.7 (+ 5) · approfondie : ch. 12 · réutilisée : ch. 12, 25, 27
- *covariance matrix* → voir [matrice de covariance, de corrélation](#m)
- *coverage* → voir [couverture (d'un intervalle)](#c)
- **CPU** (*processeur central*) — enseignée : [ch. 1, 1.7 · Deep learning][s35] · exercices : 23.29, 23.Q7, 24.7, 28.6, 28.23, 28.29, B1.26, B1.28, B2.31, B3.8 (+ 62) · approfondie : ch. 20
- *Creative Commons license* → voir [licence Creative Commons](#l)
- *credible interval* → voir [intervalle de crédibilité](#i)
- *credit assignment* → voir [attribution du crédit](#a)
- *critic* → voir [acteur-critique, acteur, critique](#a)
- *critical point* → voir [point critique, point stationnaire](#p)
- critique → voir [acteur-critique, acteur, critique](#a)
- **croissance progressive** (*progressive growing*) — enseignée : [ch. 27, 27.7.1 · Les grandes images][s479] · exercices : 27.7, 27.Q9
- **Croissant** — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s161] · exercices : 0B.36, 0B.48, 2.27, 2.31, 3.24, 3.26, 4.6, 5.4, CP1.8, 8.14 (+ 6)
- **cross fitting** (*ajustement croisé*) — enseignée : [ch. 12, 12.3.1 · Le one-hot][s98]
- *cross-attention* → voir [attention croisée](#a)
- *cross-correlation* → voir [corrélation croisée](#c)
- **cross-encoder** — enseignée : [ch. B2, Une recherche sémantique en vingt lignes][s546] ; [ch. B4, La recherche dense][s547]
- **cross-entropy** (*entropie croisée*) — enseignée : [ch. 6, 6.8 · La cross-entropy][s243] · exercices : 6.5, 6.17, 6.E1–E2, 6.Q10, CP1.11, 9.R2, 18.8, CP4.12, 23.4 (+ 45) · approfondie : ch. 18 · réutilisée : ch. 9, 13, 18, 22, 23, 25, 27, B3, B4, B8
- *cross-tabulation* → voir [table de contingence](#t)
- *cross-validation* → voir [validation croisée](#v)
- **cross_val_score** — enseignée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s268] · exercices : 8.13, 8.19, 8.22–25, 12.Q11, 14.14, 15.1, 15.6 (+ 9) · réutilisée : ch. 9, 10
- **cross_validate** — enseignée : [ch. 15, 15.9.1 · La validation croisée][s744] · exercices : 15.17, 15.27
- **CrossEntropyLoss et logits** — enseignée : [ch. 23, 23.5.7 · Corriger les labels][s758]
- *curse of dimensionality* → voir [malédiction de la dimension, fléau de la dimension](#m)
- *curvature* → voir [dérivée seconde, courbure](#d)
- **cv_results_** — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s328] · exercices : 15.18, 15.23, 18.R2, 24.18
- **cycle de dépréciation** (*deprecation cycle*) — enseignée : [ch. 7][s262]

## D

- DAG → voir [graphe orienté acyclique (DAG)](#g)
- *data augmentation* → voir [augmentation de données](#a)
- **data card** (*fiche de données*) — enseignée : [ch. 1, Les quatre fils rouges du workbook][s160] ; [ch. 29, Au-delà du livre : documenter et télécharger][s161] · exercices : 1.13, 12.11, 12.13, 15.7, 29.5, 29.15–16, 29.E2, PF.2, PF.12 (+ 7) · réutilisée : ch. B6
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
- **dataclasses (@dataclass)** — enseignée : [ch. 0A, 100.7.1 · `class`, `__init__`, attributs et méthodes ; `@dataclass`][s725] ; [ch. B7, 207.3 · Configuration : dataclasses, YAML ou JSON, surcharges,…][s661] · réutilisée : ch. B4, B7
- **DataFrame, Series** — enseignée : [ch. 0A, 100.9.1 · `DataFrame` et `Series` : `read_csv`, `head`, `info`,…][s96] · exercices : 0A.4, 0A.32, 1.R2, 3.20, 3.29, 4.23, 7.24, 8.22, 12.12, 12.32 (+ 17)
- DataLoader → voir [`Dataset`, `DataLoader`](#d)
- Datasaurus → voir [quartet d'Anscombe, Datasaurus](#q)
- *Datasaurus Dozen* → voir [quartet d'Anscombe, Datasaurus](#q)
- **dataset** (*jeu de données*) — enseignée : [ch. 0A, 100.1.2 · Le workbook : setup, `wb.check`, `wb.attempt`,…][s1] ; [ch. 0B, 101.4.1 · Forme, éléments, transposée ; un dataset est une matrice][s2] ; [ch. 20, Au-delà du livre : PyTorch en pratique][s3] · exercices : 0A.5, 0A.15, 0A.32, 0A.50, 0A.57, 0A.66–67, 0A.Q10, 20.19 · approfondie : ch. 23
- **dataset card** — enseignée : [ch. 29, 29.4 · Les collections de datasets][s498] · exercices : 29.11, 29.Q7
- **Dataset généré à la volée** — enseignée : [ch. 24][s761]
- **dataset retiré** (*retracted dataset*) — enseignée : [ch. 29, 29.5 · Des datasets plus récents][s500] · exercices : 15.7, 29.15, 29.Q8
- *dataset shift* → voir [décalage de distribution](#d)
- **`Dataset`, `DataLoader`** (*data loader*) — enseignée : [ch. 0B, 101.4.1 · Forme, éléments, transposée ; un dataset est une matrice][s2] ; [ch. 20, Au-delà du livre : PyTorch en pratique][s3] · exercices : 20.19
- **datasheet (d'un dataset)** (*datasheet for datasets*) — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s161] · exercices : 29.6, 29.E2, PF.2
- *datasheet for datasets* → voir [datasheet (d'un dataset)](#d)
- **dates avec pandas (to_datetime, dt)** — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s313]
- **DBSCAN et HDBSCAN (usage)** — enseignée : [ch. 7, Quand k-means échoue, et le clustering par densité][s261] · exercices : 7.18
- **DCGAN** (*deep convolutional GAN*) — enseignée : [ch. 27, 27.6 · DCGAN][s477] · exercices : 27.1, 27.3, 27.12, 27.20–21, 27.23, 27.26–28, B5.29 (+ 1) · réutilisée : ch. B5
- **DDIM** (*denoising diffusion implicit model*) — enseignée : [ch. B5, 205.6 · Accélérer : DDIM déterministe, moins de pas, distillation][s615] · exercices : B5.1, B5.5, B5.8, B5.17–18, B5.24–25, B5.30–31, B5.Q8 (+ 7)
- *DDIM inversion* → voir [inversion DDIM](#i)
- **ddof** (*delta degrees of freedom*) — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s185] · exercices : 2.4, 2.7–8, 2.12, 2.15, 2.18, 2.22, 2.26, 2.29, 2.31 (+ 25)
- **DDPM** (*denoising diffusion probabilistic model*) — enseignée : [ch. B5, 205.5 · Échantillonnage DDPM : moyenne du posterior, variance, T pas][s614] · exercices : B5.4, B5.8, B5.17–18, B5.20, B5.22–24, B5.29, B5.Q7 (+ 5)
- de Boltzmann → voir [politique softmax, de Boltzmann](#p)
- de corrélation → voir [matrice de covariance, de corrélation](#m)
- de déploiement → voir [biais de labels, de mesure, de déploiement](#b)
- de fausses omissions → voir [taux de fausses découvertes, de fausses omissions](#t)
- de faux négatifs → voir [taux de faux positifs, de faux négatifs](#t)
- de la structure → voir [bénédiction de la non-uniformité, de la structure](#b)
- de mesure → voir [biais de labels, de mesure, de déploiement](#b)
- *dead neuron* → voir [neurone mort](#n)
- *deadly triad* → voir [triade mortelle](#t)
- **débit** (*throughput*) — enseignée : [ch. B1, Quatre mesures à ne pas confondre][s518] · exercices : 29.3, B1.26, B4.11, B5.8, B7.3, B7.5, B7.19, B7.Q9
- débogueur → voir [point d'arrêt, débogueur](#p)
- **débruitage** (*denoising, noise reduction*) — enseignée : [ch. 1, 1.4.2 · Débruitage][s149] · exercices : 1.12, 1.Q8, 25.39, B5.Q1, B5.Q5
- **débruiteur parfait (d'un dataset fini)** (*optimal denoiser, ideal denoiser*) — enseignée : [ch. B5, 205.4 · Le débruiteur : U-Net minuscule et encodage du pas de temps][s620] · exercices : B5.22, B5.28
- *debugger* → voir [point d'arrêt, débogueur](#p)
- **décalage de distribution** (*distribution shift, dataset shift*) — enseignée : [ch. 29, Au-delà du livre : le décalage de distribution][s497] · exercices : 29.9, 29.14, 29.R3, B1.9, B1.R1, PF.3, PF.8 · réutilisée : ch. B1
- *décalage entre entraînement et service* → voir [training-serving skew](#t)
- *décalages* → voir [offsets](#o)
- *decay parameter* → voir [paramètre de décroissance](#p)
- *decision boundary* → voir [frontière de décision](#f)
- *decision region* → voir [région de décision](#r)
- *decision stump* → voir [souche](#s)
- *decision threshold* → voir [score, seuil de décision](#s)
- *decision tree* → voir [arbre de décision](#a)
- **decision_function** — enseignée : [ch. 15, 15.4.4 · decision_function et predict_proba][s742] · exercices : 7.14, 7.16, 7.22, 10.21, 13.32, 14.24, 15.2, 15.21, 15.Q5, CP3.1 (+ 1)
- **DecisionBoundaryDisplay** — enseignée : [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s211] · exercices : 15.21
- déclenché → voir [agent libre, déclenché](#a)
- **décodage glouton** (*greedy decoding*) — enseignée : [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s249] · évoquée : [ch. 28, 28.5 · Générer la suite du livre][s418] · exercices : 28.Q10
- **décodage spéculatif** (*speculative decoding*) — enseignée : [ch. B3, Ce que coûte le cache][s575]
- *decoder* → voir [encodeur, décodeur](#e)
- décodeur → voir [encodeur, décodeur](#e)
- **décomposition biais-variance** (*bias-variance decomposition*) — enseignée : [ch. 9, 9.6 · Biais et variance][s284]
- *décomposition en valeurs singulières* → voir [SVD](#s)
- **découpage (des données)** (*data split*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s263] ; [ch. B3, 203.4 · Attention multi-têtes : projections, découpage en têtes,…][s264] · exercices : 8.1, 8.19, 8.E3, 8.E5, 12.31, 23.15, 29.2, B4.22, PF.3–4 (+ 48) · approfondie : ch. 23, 29 · réutilisée : ch. 12, 22, 29
- **découpage par groupes** (*group k-fold*) — enseignée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s268]
- *découpe* → voir [split](#s)
- *deduction* → voir [déduction](#d)
- **déduction** (*deduction*) — enseignée : [ch. 11, 11.3 · Déduction et induction][s295] · exercices : 11.9, 11.Q6, 11.Q12, 14.R2
- *deep convolutional GAN* → voir [DCGAN](#d)
- **deep dream** (*inceptionism*) — enseignée : [ch. 28, 28.3 · Le deep dream][s487] · exercices : 28.16, 28.E4, 28.Q4–Q5
- **deep learning** (*apprentissage profond*) — enseignée : [ch. 1, 1.7 · Deep learning][s35] ; [ch. 20, 20.2 · Le deep learning en bref][s157] · exercices : 1.5, 5.E2, 6.Q1, 8.E3, 14.10, 18.Q1, 19.11, 20.Q1, 23.11, PF.6 (+ 56)
- *deep Q-network* → voir [deep RL, DQN](#d)
- *deep reinforcement learning* → voir [deep RL, DQN](#d)
- **deep RL, DQN** (*deep reinforcement learning, deep Q-network*) — enseignée : [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s280] ; [ch. 26, 26.8 · La vue d'ensemble][s468] · exercices : 26.E4, B8.1, B8.8, B8.12–15, B8.Q3–Q4, B8.Q9 (+ 9)
- *deep RNN* → voir [RNN profond, bidirectionnel](#r)
- **deepfake** (*hypertrucage*) — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s449] ; [ch. B5, Transparence et provenance des contenus générés][s457] · exercices : 25.12, 27.9, B5.10
- **dégel progressif** (*gradual unfreezing*) — enseignée : [ch. B1, Le dégel progressif][s505] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 3)
- *delimiters* → voir [délimiteurs (d'un prompt)](#d)
- **délimiteurs (d'un prompt)** (*delimiters*) — enseignée : [ch. B4, L'anatomie d'un prompt][s602] · exercices : B4.24
- **delta (d'un neurone)** (*error term*) — enseignée : [ch. 18, 18.6 · Étape 1 : les deltas des neurones de sortie][s349] · exercices : 18.1–2, 18.4–7, 18.Q4–Q6, 19.R1 (+ 17)
- **delta d'un neurone** — enseignée : [ch. 18, 18.6 · Étape 1 : les deltas des neurones de sortie][s349] · exercices : 18.1, 18.Q4, 18.Q9
- *delta degrees of freedom* → voir [ddof](#d)
- *demande de fusion* → voir [pull request](#p)
- **démarche expérimentale et journal d'expériences** — enseignée : [ch. 24][s761]
- **démarrage à froid** (*cold start*) — enseignée : [ch. B1][s520] · exercices : B1.14
- **démo hébergée** (*hosted demo*) — enseignée : [ch. B7, Une démo hébergée][s674] · exercices : B7.23–25, B7.Q4
- *demographic parity* → voir [parité démographique](#p)
- *denoising* → voir [débruitage](#d)
- *denoising autoencoder* → voir [autoencodeur débruiteur](#a)
- *denoising diffusion implicit model* → voir [DDIM](#d)
- *denoising diffusion probabilistic model* → voir [DDPM](#d)
- **dénombrement, factorielle, coefficient binomial C(n, k)** — enseignée : [ch. 0B, 101.1.7 · Dénombrement : principe multiplicatif, factorielle,…][s13] · exercices : 0B.Q5, 2.6 · réutilisée : ch. 2, 7
- *denormal number* → voir [nombre dénormalisé](#n)
- *dense layer* → voir [couche dense](#c)
- *dense retrieval* → voir [recherche dense](#r)
- **densité (pdf)** (*probability density function*) — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s179] · exercices : 2.16–18, 2.Q2, 2.Q4, 4.22, 7.4, 7.20, 7.Q9, CP2.4 (+ 14) · réutilisée : ch. 4, 13, 25, B5
- **densité d'échantillons** (*sample density*) — enseignée : [ch. 7, 7.6 · La malédiction de la dimension][s192] · exercices : 7.4, 7.20, 7.Q9, CP2.4
- *density-based clustering* → voir [clustering par densité](#c)
- *denying the antecedent* → voir [négation de l'antécédent](#n)
- dépendance → voir [dépendance à long terme](#d), [dépendance partielle (PDP)](#d), [dépendance partielle et ICE](#d)
- **dépendance à long terme** (*long-term dependency*) — enseignée : [ch. 22, 22.6 · Entraîner un RNN][s353] · exercices : 22.Q6
- **dépendance partielle (PDP)** (*partial dependence*) — enseignée : [ch. B6, La dépendance partielle][s629] · exercices : B6.12–13, B6.Q3
- **dépendance partielle et ICE** — enseignée : [ch. B6, 206.2 · Explications globales : coefficients, arbres, importance…][s628] · exercices : B6.12
- *dependence plot* → voir [graphique de dépendance (SHAP)](#g)
- **déploiement canari, en ombre** (*canary, shadow deployment*) — enseignée : [ch. B7, Agir : alerter, réentraîner, revenir en arrière][s579]
- *deploy* → voir [déployer](#d)
- *deployer* → voir [déployeur](#d)
- **déployer** (*deploy*) — enseignée : [ch. 1][s144] · exercices : 11.4, CP3.10, 23.E3, 26.28, B7.E1
- **déployeur** (*deployer*) — enseignée : [ch. B6, L'AI Act][s652] · exercices : B6.9, B6.Q12
- *deployment bias* → voir [biais de labels, de mesure, de déploiement](#b)
- **dépôt** (*repository*) — enseignée : [ch. 0A, 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest…][s65] ; [ch. B7, 207.1 · Du notebook au projet : limites du notebook, structure…][s99] · exercices : 0A.10–12, B7.9, B7.14, B7.23, B7.25, B7.Q2, B7.Q4, PF.12 (+ 16)
- *deprecation cycle* → voir [cycle de dépréciation](#c)
- *depth* → voir [profondeur (d'un arbre)](#p)
- *depthwise convolution* → voir [convolution par canal](#c)
- *depthwise separable convolution* → voir [convolution séparable](#c)
- *derivative* → voir [dérivée](#d)
- **dérive des données** (*data drift*) — enseignée : [ch. B7, 207.10 · Surveiller en production : dérive des données et des…][s270] · évoquée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s268] · exercices : B7.Q11 · réutilisée : ch. 20
- *derived model* → voir [modèle dérivé](#m)
- **dérivée** (*derivative*) — enseignée : [ch. 0B, 101.5 · Dérivées][s121] ; [ch. 5, 5.3 · La dérivée][s122] · exercices : 0B.25, 0B.47, 0B.51, 0B.Q10, 5.11, 5.Q1, 5.Q5–Q6, 9.2, 17.12 (+ 70) · approfondie : ch. 17 · réutilisée : ch. 5, 6, 9, 13, 16, 17, 18, 22, 27, B6
- dérivée directionnelle → voir [pente dans une direction, dérivée directionnelle](#p)
- **dérivée numérique** — enseignée : [ch. 5, Tangente et dérivée][s226] · exercices : 0B.47, 18.16 · réutilisée : ch. 18
- **dérivée partielle** (*partial derivative*) — enseignée : [ch. 0B, 101.6.2 · Dérivées partielles][s124] · exercices : 5.Q10, 16.6
- **dérivée seconde, courbure** (*second derivative, curvature*) — enseignée : [ch. 5, Tangente et dérivée][s226] · exercices : 5.4–5, 5.7, 5.11, 5.Q6, 5.R2, CP1.9, 27.5
- **dérivées des activations** — enseignée : [ch. 17, 17.7 · La galerie des activations][s747] · exercices : 17.4
- **dérivées partielles, gradient, pas de descente** — enseignée : [ch. 0B, 101.6.2 · Dérivées partielles][s124] · exercices : 0B.25, 0B.29, 0B.32, 0B.51, 5.3, 5.5, 6.R1, CP1.9, 8.R2, 9.2 (+ 14) · réutilisée : ch. 5, 9, 17, 18, B6
- **dérivées usuelles (puissances, exp, ln) et règles (somme, produit, quotient)** — enseignée : [ch. 0B, 101.5.2 · Dérivées usuelles et règles (somme, produit, quotient,…][s729] · exercices : 0B.22, 0B.47, 0B.Q10
- **déroulement** (*unrolling*) — enseignée : [ch. 22][s413] · exercices : 22.Q3
- descendant → voir [parent, enfant ; ancêtre, descendant](#p)
- descente → voir [descente de coordonnées](#d), [descente de gradient](#d), [descente de gradient stochastique (SGD)](#d), [descente par batch](#d), [descente par mini-batches](#d)
- **descente de coordonnées** (*coordinate descent*) — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s278] · exercices : 9.23
- **descente de gradient** (*gradient descent*) — enseignée : [ch. 5, Suivre la pente jusqu'à un extremum][s53] · évoquée : [ch. 0B, 101.6.3 · Gradient et direction de plus grande pente][s54] · exercices : 5.3, 5.E1, 5.Q5, 5.R3, 6.R1, 8.R2, 9.20, 12.29, 12.R3, 13.32 (+ 47) · réutilisée : ch. 8, 9, 12, 13, 14, 18, 19, 27, 28
- **descente de gradient stochastique (SGD)** (*stochastic gradient descent*) — enseignée : [ch. 19, 19.4.2 · La descente stochastique (SGD)][s357] · exercices : 18.20, 19.6, 19.9, 19.13, 19.16, 19.23–25, 19.E1, CP4.2 (+ 21) · réutilisée : ch. 23
- **descente par batch** (*batch gradient descent*) — enseignée : [ch. 19, 19.4.1 · La descente de gradient par batch][s6] · exercices : 19.Q8
- **descente par mini-batches** (*mini-batch gradient descent*) — enseignée : [ch. 19, 19.4.3 · La descente par mini-batches][s8] · exercices : 19.Q12
- **détecteur de motif** (*feature detector*) — enseignée : [ch. 21, 21.3.1 · Les filtres][s391]
- **détection d'anomalies** (*anomaly detection*) — enseignée : [ch. 25, 25.5.4 · Une entrée jamais vue][s456]
- **détection d'anomalies par reconstruction** — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s449]
- *determinant* → voir [inverse, déterminant](#i)
- déterminant → voir [inverse, déterminant](#i)
- *deterministic algorithms* → voir [algorithmes déterministes](#a)
- **dette technique (du ML)** (*technical debt*) — enseignée : [ch. B7, La dette technique du ML][s679] · exercices : B7.8
- **device** (*appareil*) — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s234] ; [ch. 23, 23.2.2 · Programmer et déboguer en Python][s371] · exercices : 20.13, 20.23, 20.28, 23.12, 23.18, 23.20, 23.22, 23.Q7, 23.Q10, B4.Q3 (+ 7)
- **diagnostic par les journaux RL** — enseignée : [ch. B8, Les journaux de PPO][s684]
- **diagramme de fiabilité** (*reliability diagram*) — enseignée : [ch. 3, Au-delà du livre (3) : la calibration des probabilités][s214] · exercices : 3.28
- **diagramme pente-ordonnée** (*slope-intercept diagram*) — enseignée : [ch. 9, 9.7 · Ajuster une droite avec la règle de Bayes][s279]
- *dictionary* → voir [dictionnaire](#d)
- **dictionnaire** (*dictionary*) — enseignée : [ch. 0A, 100.3.3 · Dictionnaires][s71] · exercices : 0A.18, 0A.22, 0A.39–40, 0A.44, 0A.63, 0A.E2, 0A.Q8, 0B.47, B3.Q2 (+ 84) · réutilisée : ch. 26
- **différence centrée** — enseignée : [ch. 5, Au-delà du livre (1) : calculer une dérivée sur ordinateur][s227] · exercices : 5.6, 5.12–13, 5.15, 5.21, 5.23, 18.16, 18.27
- **différence finie (avant, arrière, centrée)** (*finite difference*) — enseignée : [ch. 5, Au-delà du livre (1) : calculer une dérivée sur ordinateur][s227] · exercices : 5.12–13, 21.7
- *difference quotient* → voir [taux d'accroissement](#t)
- *differential privacy* → voir [confidentialité différentielle (DP-SGD)](#c)
- **différentiation automatique** (*automatic differentiation, autodiff*) — enseignée : [ch. 5, Au-delà du livre (3) : la différentiation automatique][s233] · exercices : 5.9, 5.21 · approfondie : ch. 18
- **différentiation automatique en mode inverse** (*reverse-mode automatic differentiation*) — enseignée : [ch. 18, 18.12.3 · Ce qu'elle fait][s129] · exercices : 18.10, 18.26–27, 18.Q9
- *diffusion* → voir [broadcasting](#b)
- **diffusion à variance préservée** (*variance preserving diffusion*) — enseignée : [ch. B5, Un pas de bruit][s607]
- *diffusion au fil de l'eau* → voir [streaming](#s)
- **diffusion latente** (*latent diffusion*) — enseignée : [ch. B5, La diffusion latente][s450] · exercices : 25.10, 25.E5, B5.8, B5.Q11
- *diffusion model* → voir [modèle de diffusion](#m)
- *diffusion transformer* → voir [transformer de diffusion (DiT)](#t)
- **dilatation** (*dilation, atrous convolution*) — enseignée : [ch. 21, 21.8 · La convolution transposée][s392] · exercices : 21.3, 21.12, 21.14–15, 21.18, 21.30–31, 21.Q9, 22.R1
- *dilation* → voir [dilatation](#d)
- **dimension effective** (*effective dimensionality*) — enseignée : [ch. 15, 15.9.2 · Chercher des hyperparamètres][s332]
- *dimensionality reduction* → voir [réduction de dimension](#r)
- *direct preference optimization* → voir [DPO](#d)
- *directed acyclic graph* → voir [graphe orienté acyclique (DAG)](#g)
- *directional derivative* → voir [pente dans une direction, dérivée directionnelle](#p)
- *discount factor* → voir [facteur d'actualisation](#f)
- *discounted return* → voir [retour actualisé](#r)
- **discrétisation d'un état continu** — enseignée : [ch. 26][s62]
- **discrétiser** (*discretize*) — enseignée : [ch. 26][s62] · exercices : 26.36
- *discretize* → voir [discrétiser](#d)
- **discriminateur** (*discriminator*) — enseignée : [ch. 27, 27.4.1 · Le discriminateur][s475] · exercices : 27.1, 27.3–6, 27.12, 27.17, 27.Q6, 28.R1, CP6.4 (+ 24)
- **discriminateur optimal** (*optimal discriminator*) — enseignée : [ch. 27, 27.4.4 · Le jeu][s245] · exercices : 27.5, CP6.4
- **discrimination par mini-batch** (*minibatch discrimination*) — enseignée : [ch. 27, 27.7.2 · L'effondrement de mode][s344] · exercices : 27.Q7, 27.Q10
- *discriminative fine-tuning* → voir [learning rates discriminatifs](#l)
- *discriminative learning rates* → voir [learning rates discriminatifs](#l)
- *discriminator* → voir [discriminateur](#d)
- distance → voir [distance de domaine](#d), [distance de Fréchet](#d), [distance de Manhattan](#d), [distance de Wasserstein (1D)](#d), [distance de Wasserstein 1D](#d)
- *distance concentration* → voir [concentration des distances](#c)
- **distance de domaine** (*domain gap*) — enseignée : [ch. B1, La distance de domaine][s509] · exercices : B1.16, B1.22
- **distance de Fréchet** (*Fréchet distance*) — enseignée : [ch. B5, La distance de Fréchet][s482] ; [ch. 27, Au-delà du livre : mesurer un générateur][s481] · exercices : 27.22–23, 27.28, B5.10, B5.25, B5.28–29, B5.31, B5.Q10, B5.R2 (+ 4)
- **distance de Manhattan** (*Manhattan distance, $L^1$ distance*) — enseignée : [ch. 13, 13.3 · Les k plus proches voisins][s257] · exercices : 13.1
- **distance de Wasserstein (1D)** (*Wasserstein distance, earth mover's distance*) — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s481] · exercices : 27.11, 27.19
- **distance de Wasserstein 1D** — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s481] · exercices : 27.11
- **distances** — enseignée : [ch. 12, 12.5 · Normaliser et standardiser][s177] ; [ch. 13, 13.3 · Les k plus proches voisins][s257] · exercices : 2.25, 7.13, 7.20, 13.1, 13.4, 13.9, 13.14–15, 13.34, 13.R1 (+ 26) · approfondie : ch. 7
- **distillation** (*knowledge distillation*) — enseignée : [ch. 6, 6.9 · La divergence KL][s244]
- **distillation (de l'échantillonneur)** (*step distillation*) — enseignée : [ch. B5, La distillation][s621] · exercices : B5.5, B5.8, B5.17–18, B5.24–25, B5.30–31, B5.Q7–Q8 (+ 1)
- **distillation adversariale** (*adversarial diffusion distillation*) — enseignée : [ch. B5, La distillation][s621]
- **distillation progressive** (*progressive distillation*) — enseignée : [ch. B5, La distillation][s621]
- *distributed term* → voir [terme distribué](#t)
- **distribution de probabilité, loi** (*probability distribution*) — enseignée : [ch. 2, Des comptages à une distribution de probabilité][s176] · exercices : 2.2, 2.Q2, 4.Q3, 17.6, 17.19, B4.Q7 · réutilisée : ch. 6, 17
- **distribution log-uniforme** (*log-uniform distribution*) — enseignée : [ch. 15, 15.9.4 · La recherche aléatoire][s330]
- *distribution shift* → voir [décalage de distribution](#d)
- *distributional hypothesis* → voir [hypothèse distributionnelle](#h)
- DiT → voir [transformer de diffusion (DiT)](#t)
- divergence → voir [divergence de Jensen-Shannon](#d), [divergence KL entre gaussiennes](#d), [divergence KL, divergence de Kullback-Leibler](#d)
- **divergence de Jensen-Shannon** (*Jensen-Shannon divergence*) — enseignée : [ch. 6, 6.9 · La divergence KL][s244] ; [ch. 27, 27.4.4 · Le jeu][s245] · exercices : 6.16, 6.18, 27.5
- divergence de Kullback-Leibler → voir [divergence KL, divergence de Kullback-Leibler](#d)
- **divergence KL entre gaussiennes** — enseignée : [ch. 25][s769]
- **divergence KL, divergence de Kullback-Leibler** (*KL divergence, relative entropy*) — enseignée : [ch. 6, 6.9 · La divergence KL][s244] · exercices : 6.5, 6.16, 6.E2, 6.E4, 25.7, 25.29, 25.R3, 27.5, CP6.2, CP6.8 (+ 3) · approfondie : ch. 25 · réutilisée : ch. 25, B7, B8
- **Dockerfile, couche** (*layer*) — enseignée : [ch. B7, 207.9 · Conteneurs et déploiement : Dockerfile, variables…][s673] · exercices : B7.24, B7.Q4, PF.11
- **docstring** — enseignée : [ch. 0A, 100.5.5 · Docstrings au format NumPy et annotations de type][s80] · exercices : 0A.26, 0A.61, 0A.63–66, 2.31, 10.20 · réutilisée : ch. 10
- **docstrings au format NumPy, annotations de type** — enseignée : [ch. 0A, 100.5.5 · Docstrings au format NumPy et annotations de type][s80] · exercices : 10.20
- **documentation de provenance et de licences** — enseignée : [ch. 28][s399]
- *domain gap* → voir [distance de domaine](#d)
- *domain of discourse* → voir [domaine du discours](#d)
- *domain randomization* → voir [randomisation de domaine](#r)
- **domaine du discours** (*domain of discourse*) — enseignée : [ch. 11, 11.4 · La déduction][s297] · exercices : 11.5
- **domaine public** (*public domain*) — enseignée : [ch. 28, 28.4.5 · Réaliser le transfert][s493] ; [ch. B4, Licences et droit d'auteur][s495] · exercices : 1.11, 22.10, 28.9, 28.23, 29.R1, B1.R1, B3.10, B4.10 · réutilisée : ch. 29
- donnée → voir [donnée nominale](#d), [donnée numérique (quantitative)](#d), [donnée ordinale](#d), [données synthétiques](#d)
- **donnée nominale** (*nominal data*) — enseignée : [ch. 12, 12.3 · Les types de données][s303]
- **donnée numérique (quantitative)** (*numerical data*) — enseignée : [ch. 12, 12.3 · Les types de données][s303]
- **donnée ordinale** (*ordinal data*) — enseignée : [ch. 12, 12.3 · Les types de données][s303] · exercices : 12.Q2
- **données synthétiques** (*synthetic data*) — enseignée : [ch. 24, 24.4.7 · Des données synthétiques][s439] · exercices : 14.2, 14.6, 15.3, 24.Q8, B7.23, B7.25
- **DoRA** (*weight-decomposed low-rank adaptation*) — enseignée : [ch. B4, QLoRA][s598]
- *dot product* → voir [produit scalaire](#p)
- *double descent* → voir [double descente](#d)
- **double descente** (*double descent*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s285] · exercices : 9.11, 9.30, 9.E5
- Double DQN → voir [double estimateur, Double DQN](#d)
- **double estimateur, Double DQN** (*double estimator*) — enseignée : [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s280] · exercices : B8.1, B8.15, B8.Q3–Q4, B8.Q10
- *double estimator* → voir [double estimateur, Double DQN](#d)
- **doublons et fuites entre découpages** — enseignée : [ch. 29][s499]
- DP-SGD → voir [confidentialité différentielle (DP-SGD)](#c)
- **DPO** (*direct preference optimization*) — enseignée : [ch. B8, DPO : la récompense est cachée dans la politique][s698] · exercices : B8.6, B8.10, B8.25, B8.E4, B8.Q11
- DQN → voir [deep RL, DQN](#d)
- **DQN et ses stabilisateurs** — enseignée : [ch. B8, La surestimation, et Double DQN][s685]
- *draw* → voir [tirage](#t)
- *drop-column importance* → voir [importance par retrait](#i)
- **dropout** (*abandon*) — enseignée : [ch. 20, 20.4.3 · Le dropout][s24] ; [ch. 9, Pénalité sur les poids, dropout, batchnorm][s25] · exercices : 9.Q8, 20.3, 20.5, 20.28, 20.E1, 20.Q5, CP4.4, 21.R1, 24.Q3, 24.Q9 (+ 56) · approfondie : ch. 24 · réutilisée : ch. 21, 24, B3
- **dropout et max-norm en PyTorch** — enseignée : [ch. 24, 24.2.6 · Ajouter du dropout][s435]
- **dropout inversé** (*inverted dropout*) — enseignée : [ch. 20, 20.4.3 · Le dropout][s24] · exercices : 20.3, 20.5, 20.16, CP4.4, 24.6
- **dtype** (*data type*) — enseignée : [ch. 0A, 100.8.1 · Créer des arrays : `dtype`, `shape`, `ndim`][s90] ; [ch. 20, 20.2.1 · Les tenseurs][s234] · exercices : 0A.27, 0A.54, 0A.65, 0A.Q10, 2.13, 20.2, 20.13, 20.15, B4.1, B4.Q3 (+ 19) · approfondie : ch. 0A
- **duck typing** (*typage canard*) — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s328] · exercices : 15.8
- *dummy variable* → voir [variable indicatrice](#v)
- *dynamic* → voir [quantification dynamique, statique](#q)
- *dynamic batching* → voir [regroupement dynamique](#r)
- *dynamic programming* → voir [programmation dynamique](#p)

## E

- **early stopping** (*arrêt anticipé*) — enseignée : [ch. 9, 9.4 · L'early stopping][s273] ; [ch. 23, 23.10.3 · Early stopping][s274] ; [ch. 14, Au-delà du livre : le gradient boosting][s195] · exercices : 9.5, 9.20, 9.Q6, 9.Q8, 20.28, 23.6, 23.25, 23.27, CP5.6, B1.19 (+ 14) · approfondie : ch. 20 · réutilisée : ch. 23, 24, B1
- **early stopping et meilleur checkpoint** — enseignée : [ch. 23, 23.10.3 · Early stopping][s274]
- *earth mover's distance* → voir [distance de Wasserstein (1D)](#d)
- écart → voir [écart de généralisation](#e), [écart interquartile](#e), [écart simulation-réalité, sim-to-real](#e)
- **écart de généralisation** (*generalization gap*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s33]
- **écart interquartile** (*interquartile range, IQR*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s304] · exercices : 8.8, 12.19
- **écart simulation-réalité, sim-to-real** (*reality gap*) — enseignée : [ch. B8, 208.8 · Limites : efficacité en échantillons, généralisation,…][s696]
- écart-type → voir [variance, écart-type](#v)
- **écart-type par mini-batch** (*minibatch standard deviation*) — enseignée : [ch. 27, 27.5 · Un GAN en action][s478] · exercices : 27.24, 27.Q7, 27.Q10
- **échantillon** (*sample*) — enseignée : [ch. 1, 1.2.2 · La version pour ordinateur : échantillons, features, labels][s56] · exercices : 0B.E5, 1.9, 1.Q3, 2.10, CP1.13, 7.4, 7.Q9, 11.17, CP2.4, 12.5 (+ 61)
- **échantillon biaisé** (*biased sample*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s299] · exercices : 2.10, 9.R3, 11.6, 11.17, 11.Q10, 14.R2
- échantillonnage → voir [échantillonnage ancestral](#e), [échantillonnage avec ou sans remise](#e), [échantillonnage DDPM](#e), [échantillonnage de Thompson](#e), [échantillonnage négatif](#e)
- **échantillonnage ancestral** (*ancestral sampling*) — enseignée : [ch. B5, L'algorithme][s612] · exercices : B5.4, B5.19–20
- **échantillonnage avec ou sans remise** — enseignée : [ch. 2, 2.5.1 · Avec remise, 2.5.2 Sans remise][s734]
- **échantillonnage DDPM** — enseignée : [ch. B5, 205.5 · Échantillonnage DDPM : moyenne du posterior, variance, T pas][s614] · exercices : B5.17, B5.Q7
- **échantillonnage de Thompson** (*Thompson sampling*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s156] · exercices : 11.12
- **échantillonnage négatif** (*negative sampling*) — enseignée : [ch. B2, word2vec : prédire plutôt que compter][s540] · exercices : B2.7, B2.25
- **échantillonneur** (*sampler*) — enseignée : [ch. B5, Ce que coûte un échantillon][s613] · exercices : B5.4, B5.20, B5.24, B5.30, B5.Q7
- **échantillonneur-bloqueur** (*sample-and-hold*) — enseignée : [ch. 16, 16.3 · Flux synchrone et asynchrone][s339]
- **échelle de guidance** (*guidance scale*) — enseignée : [ch. B5, La guidance sans classifieur][s623] · exercices : B5.26
- **échelle des logits (CLIP)** (*logit scale*) — enseignée : [ch. B1, CLIP : rapprocher les images de leur légende][s515] · exercices : 12.17, B1.5, B1.7, B1.9, B1.24–25, B1.Q10, B8.20, B8.22, B8.Q8 (+ 14)
- *écrêtage du gradient* → voir [gradient clipping](#g)
- **écriture atomique** (*atomic write*) — enseignée : [ch. B4][s601] · exercices : B4.29
- *EDA* → voir [analyse exploratoire](#a)
- *editable install* → voir [installation éditable](#i)
- *effective dimensionality* → voir [dimension effective](#d)
- **efficacité en échantillons** (*sample efficiency*) — enseignée : [ch. B8, 208.8 · Limites : efficacité en échantillons, généralisation,…][s696] · exercices : B8.8, B8.28
- **efficacité, symétrie, joueur nul, additivité** (*efficiency, symmetry, null player, additivity*) — enseignée : [ch. B6, Quatre propriétés, une seule solution][s633] · exercices : B6.1–2, B6.11, B6.14–15, B6.17, B6.Q4, B8.8, B8.28
- *efficiency* → voir [efficacité, symétrie, joueur nul, additivité](#e)
- **effondrement (d'un réseau linéaire)** (*collapse of a linear network*) — enseignée : [ch. 17, 17.2 · Ce que fait une fonction d'activation][s52] ; [ch. 27, 27.7.2 · L'effondrement de mode][s344] · exercices : 9.10, 17.3, 17.Q1, 25.34, 27.24, 27.E2, 27.Q10, CP6.1, CP6.7, B5.R2 (+ 1) · approfondie : ch. 27 · réutilisée : ch. B5
- **effondrement de mode** (*mode collapse*) — enseignée : [ch. 27, 27.7.2 · L'effondrement de mode][s344] · exercices : 27.24, 27.E2, 27.Q10, CP6.1, B5.29, B5.R2 · réutilisée : ch. B5
- **effondrement du posterior** (*posterior collapse*) — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s449] · exercices : 25.34, CP6.7
- **égalité des chances** (*equal opportunity*) — enseignée : [ch. B6, 206.9 · Équité : indicateurs de groupe (parité démographique,…][s647] · exercices : B6.28, B6.Q11
- *égalité des taux d'erreur* → voir [equalized odds](#e)
- *eigenvalue* → voir [vecteur propre, valeur propre](#v)
- *eigenvector* → voir [vecteur propre, valeur propre](#v)
- élagage → voir [pré-élagage, élagage](#p)
- **élagage coût-complexité** (*minimal cost-complexity pruning*) — enseignée : [ch. 13, 13.5.3 · Limiter l'overfitting][s317]
- *élan* → voir [momentum](#m)
- **Elastic Net** — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s278] · exercices : 9.E3
- *elastic weight consolidation* → voir [EWC](#e)
- **ELBO** (*evidence lower bound*) — enseignée : [ch. 25, 25.9.2 · La structure d'un VAE][s455] · exercices : B5.R3 · réutilisée : ch. B5
- *elbow method* → voir [méthode du coude](#m)
- *element-wise product* → voir [produit de Hadamard](#p)
- *elementwise* → voir [par échantillon, par feature, par élément](#p)
- **ELU** (*exponential linear unit*) — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s235] · exercices : 17.2, 17.10, 17.14, 17.Q8, 17.Q10
- *EMA* → voir [moyenne mobile exponentielle](#m)
- **EMA des poids** — enseignée : [ch. B5, En pratique : la moyenne exponentielle des poids][s785]
- **embedding** (*plongement, représentation vectorielle*) — enseignée : [ch. B2, 202.7 · Embeddings : de one-hot à `nn.Embedding`, entraînement et…][s22] ; [ch. 12, 12.7.2 · Standardisation et PCA pour des images][s23] · exercices : 24.30, B1.2, B1.13, B1.Q11, B2.5, B2.9, B2.24, B2.Q9, B2.Q12, B2.R1 (+ 51) · réutilisée : ch. 24, 28, B2, B3, B4
- **embedding de phrase** (*sentence embedding*) — enseignée : [ch. B2, Deux modèles d'embedding de phrases][s545]
- **embedding figé** (*frozen embedding*) — enseignée : [ch. B1, Calculer les embeddings une fois][s502] · exercices : B2.R1
- **embedding statique, contextuel** (*static, contextual embedding*) — enseignée : [ch. B2, 202.10 · Embeddings contextuels et embeddings de phrases ;…][s543] · exercices : B2.29
- **embeddings de phrases** — enseignée : [ch. B2, 202.10 · Embeddings contextuels et embeddings de phrases ;…][s543] · exercices : B1.Q10, B2.28, PF.6 · réutilisée : ch. B4
- *embodied cognition* → voir [cognition incarnée](#c)
- **empreinte (des données)** (*data fingerprint*) — enseignée : [ch. B7, 207.4 · Reproductibilité : graines, déterminisme, versions figées,…][s664] · exercices : B7.7–9, B7.14–15, B7.17, B7.25, B7.Q2, B7.Q7–Q8 (+ 3)
- en ligne → voir [inférence par batches, en ligne](#i)
- en ombre → voir [déploiement canari, en ombre](#d)
- encodage → voir [encodage de position](#e), [encodage de position rotatif (RoPE)](#e), [encodage du pas de temps](#e), [encodage ordinal](#e), [encodage par la cible](#e), [encodages de position (sinusoïdal, RoPE)](#e)
- **encodage de position** (*positional encoding*) — enseignée : [ch. B3, L'encodage sinusoïdal (2017)][s563] · exercices : B3.25, B3.Q7
- **encodage de position rotatif (RoPE)** (*rotary position embedding*) — enseignée : [ch. B3, RoPE : tourner les requêtes et les clés][s565] · exercices : B3.7, B3.15, B3.E5, B3.Q7
- **encodage du pas de temps** (*timestep embedding*) — enseignée : [ch. B5, 205.4 · Le débruiteur : U-Net minuscule et encodage du pas de temps][s620] · exercices : B5.21
- *encodage one-hot* → voir [one-hot](#o)
- **encodage ordinal** (*ordinal encoding*) — enseignée : [ch. 12, 12.3 · Les types de données][s303] · exercices : 12.1
- **encodage par la cible** (*target encoding*) — enseignée : [checkpoint 2][s706]
- **encodages de position (sinusoïdal, RoPE)** — enseignée : [ch. B3, Les positions apprises][s564] · exercices : B3.E5
- *encoder* → voir [encodeur, décodeur](#e)
- *encoder-decoder* → voir [encodeur-décodeur](#e)
- **encodeur, décodeur** (*encoder, decoder*) — enseignée : [ch. 25, 25.2 · Introduction : comprimer][s447] · exercices : 25.1, 25.3, 25.5, 25.8, 25.11–13, 25.15, B3.E3, B4.28 (+ 31)
- **encodeur-décodeur** (*encoder-decoder, seq2seq*) — enseignée : [ch. B3, L'encodeur-décodeur : T5 et l'attention croisée][s550] · exercices : B3.E3, B3.Q1, B3.Q9
- *endpoint* → voir [API HTTP, route](#a)
- enfant → voir [parent, enfant ; ancêtre, descendant](#p)
- **ensemble** (*set*) — enseignée : [ch. 0A, 100.3.4 · Ensembles][s72] ; [ch. 0B, 101.1.6 · Ensembles : union, intersection, complémentaire, cardinal][s73] ; [ch. 14, 14.2 · Les ensembles][s74] · exercices : 0A.19, 0A.22, 0A.E2, 0B.5, 0B.12, 0B.Q5, 14.5, 14.9, 14.13, 14.Q2 (+ 13)
- **ensemble (de modèles)** (*ensemble learning*) — enseignée : [ch. 14, 14.2 · Les ensembles][s74] · exercices : 14.3–6, 14.9, 14.12–13, 14.16–17, 14.Q2 (+ 7)
- **ensemble de prompts** (*prompt ensembling*) — enseignée : [ch. B1, Le zéro-shot][s512] · exercices : B1.Q10
- *ensemble learning* → voir [ensemble (de modèles)](#e)
- **ensembles : union, intersection, complémentaire, cardinal** — enseignée : [ch. 0B, 101.1.6 · Ensembles : union, intersection, complémentaire, cardinal][s73] · exercices : 0B.5
- **entraînement** (*training*) — enseignée : [ch. 8, 8.2 · La boucle d'entraînement][s9] ; [ch. 1, 1.2.2 · La version pour ordinateur : échantillons, features, labels][s56] · exercices : 1.16, 8.2, 8.12, 8.E1, 8.Q1, 8.Q3, 8.Q10, 9.9, 9.13, 11.R2 (+ 92) · approfondie : ch. 18, 20, 27 · réutilisée : ch. 5, 9, 12, 19, 21, 22, 23, 25, 26, 29, B2, B4, B5, B8
- **entraînement adversarial** (*adversarial training*) — enseignée : [ch. 21, Au-delà du livre : l'attaque FGSM][s403]
- **entraînement fumée** (*smoke test*) — enseignée : [ch. B7, Quatre familles de tests][s666] · exercices : B7.9, B7.11, B7.15, B7.23, B7.25, B7.Q7
- *entrées* → voir [fan-in, fan-out](#f)
- **entropie (de Shannon)** (*entropy, Shannon*) — enseignée : [ch. 6, 6.7 · L'entropie][s242] ; [ch. 13, 13.5.2 · Choisir les splits][s147] · exercices : 6.3, 6.8–9, 6.14, 6.24, 6.E2, 6.Q8–Q9, CP1.11, 7.R1 (+ 24) · réutilisée : ch. 7, 17, B2
- **entropie conditionnelle** (*conditional entropy*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s162] · exercices : 6.26
- *entropy* → voir [entropie (de Shannon)](#e)
- *entropy bonus* → voir [bonus d'entropie](#b)
- *entry* → voir [composante](#c)
- *entry point* → voir [point d'entrée](#p)
- *environment* → voir [agent, environnement, action, récompense](#a)
- *environment step budget* → voir [budget de pas (d'environnement)](#b)
- *environment variable* → voir [variable d'environnement](#v)
- environnement → voir [agent, environnement, action, récompense](#a)
- **environnements vectorisés** (*vectorized environments*) — enseignée : [ch. B8, Les environnements vectorisés][s694] · exercices : B8.20
- **environnements vectorisés gymnasium** — enseignée : [ch. B8, Les environnements vectorisés][s694]
- *episode* → voir [épisode](#e)
- **épisode** (*episode*) — enseignée : [ch. 26, 26.3 · La structure du RL][s459] · exercices : 26.6, 26.13–14, 26.21, 26.33, 26.Q3–Q4, B8.14, B8.28, B8.Q10 (+ 35)
- épisode tronqué → voir [état terminal, épisode tronqué](#e)
- **epoch** (*époque*) — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s7] ; [ch. 8, 8.2 · La boucle d'entraînement][s9] · évoquée : [ch. 18, 18.11.1 · Explorer le learning rate][s10] · exercices : 8.Q2, 10.6, 10.19, CP2.8, 19.28, 20.Q5, CP4.9, 22.8, 23.5, 23.Q10 (+ 20) · réutilisée : ch. 19
- **epsilon machine** (*machine epsilon*) — enseignée : [ch. 5][s229]
- *epsilon-greedy* → voir [ε-greedy](#g)
- *equal opportunity* → voir [égalité des chances](#e)
- **equalized odds** (*égalité des taux d'erreur*) — enseignée : [ch. B6, 206.9 · Équité : indicateurs de groupe (parité démographique,…][s647] · exercices : B6.5–6, B6.23, B6.28–29, B6.Q11
- **équation de Bellman** (*Bellman equation*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s154] · exercices : 26.8
- **équations normales** (*normal equations*) — enseignée : [ch. 9, Les moindres carrés (au-delà du livre)][s232] · exercices : 9.17
- **équilibre de Nash** (*Nash equilibrium*) — enseignée : [ch. 27, 27.4.4 · Le jeu][s245] · exercices : 27.Q5
- **équité par aveuglement** (*fairness through unawareness*) — enseignée : [ch. B6, L'équité par aveuglement][s709] ; [mini-projet 3][s710] · exercices : B6.9, B6.26
- *equivalence test* → voir [test d'équivalence](#t)
- **équivariance par permutation** (*permutation equivariance*) — enseignée : [ch. B3][s562]
- erreur → voir [erreur d'arrondi](#e), [erreur d'entraînement, erreur de généralisation](#e), [erreur de reconstruction](#e), [erreur de troncature](#e), [erreur du procureur](#e), [erreur du taux de base](#e), [erreur relative (d'un gradient)](#e), [erreur TD](#e), [erreur type](#e), [erreur type d'un score](#e)
- *erreur absolue moyenne* → voir [MSE, RMSE, MAE](#m)
- *erreur absolue moyenne en pourcentage* → voir [MAPE](#m)
- **erreur d'arrondi** (*round-off error*) — enseignée : [ch. 5, Au-delà du livre (2) : les arrondis et le choix du pas][s228] · exercices : 5.13, 25.29
- **erreur d'entraînement, erreur de généralisation** (*training error, generalization error*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s33] · exercices : 9.13, 9.21, 9.30, 9.Q1, 9.Q4–Q7, 9.Q9, 11.Q7 (+ 7)
- erreur de généralisation → voir [erreur d'entraînement, erreur de généralisation](#e)
- **erreur de reconstruction** (*reconstruction error*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s199] · exercices : 15.15, 25.22, 25.37, 25.Q5, 25.Q11
- **erreur de troncature** (*truncation error*) — enseignée : [ch. 5, Au-delà du livre (1) : calculer une dérivée sur ordinateur][s227] · exercices : 5.13
- **erreur du procureur** (*prosecutor's fallacy*) — enseignée : [ch. 4][s221] · exercices : 4.10
- **erreur du taux de base** (*base rate fallacy*) — enseignée : [ch. 3, 3.8 · Appliquer la matrice de confusion : le piège de la prévalence][s206]
- *erreur quadratique moyenne* → voir [MSE, RMSE, MAE](#m)
- **erreur relative (d'un gradient)** (*relative error*) — enseignée : [ch. 18, Au-delà du livre : vérifier un gradient][s139] · exercices : 18.16–17, 18.27
- **erreur TD** (*temporal-difference error*) — enseignée : [ch. B8, L'erreur TD, première estimation de l'avantage][s465] ; [ch. 26, 26.6.1 · Les valeurs Q et leur mise à jour][s462] · exercices : 26.20, B8.4, B8.Q7
- **erreur type** (*standard error*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s189] · exercices : 2.22, 2.24, 8.4, 8.6, 8.23, 11.6, 11.17, 11.25–26, CP2.1 (+ 5) · approfondie : ch. 8
- **erreur type d'un score** — enseignée : [ch. 8, 8.2.1 · Pourquoi le score d'entraînement trompe][s266]
- *error analysis* → voir [analyse des erreurs](#a)
- *error rate* → voir [taux d'erreur](#t)
- *error term* → voir [delta (d'un neurone)](#d)
- **espace de recherche** (*search space*) — enseignée : [ch. 24, 24.3.4 · La recherche d'hyperparamètres][s329]
- **espace latent** (*latent space*) — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s449] ; [ch. B5, La diffusion latente][s450] · exercices : 25.10, 25.18, 25.20–21, 25.31, 25.33, 25.E5, 25.Q7, 27.26, B5.8 (+ 2) · réutilisée : ch. B5
- **espérance** (*expected value, expectation*) — enseignée : [ch. 2, 2.3.5 · L'espérance][s134] ; [ch. 0B, 101.7.3 · Variable aléatoire discrète et espérance][s133] · exercices : 0B.27–28, 0B.53, 0B.E5, 0B.Q12, 2.5, 2.Q7, 2.R3, 20.5, 24.6 (+ 19) · réutilisée : ch. 4, 6, 11, 20, 25, 26, B8
- *essaim d'abeilles* → voir [beeswarm](#b)
- **estimateur (scikit-learn)** (*estimator*) — enseignée : [ch. 15, 15.4 · Les estimateurs][s255] · exercices : 8.16, 9.R3, 15.8–9, 15.17, 15.29, 15.Q3, 15.Q9, 16.R1, 24.16 (+ 9) · approfondie : ch. 24 · réutilisée : ch. 9, 14, 29
- **estimateur \*straight-through\*** (*straight-through estimator, STE*) — enseignée : [ch. 10][s289]
- **estimateur scikit-learn autour d'un réseau PyTorch** — enseignée : [ch. 24, 24.3.1 · Habiller un réseau en estimateur][s762]
- *estimator* → voir [estimateur (scikit-learn)](#e)
- étape → voir [workflow, job, étape, action](#w)
- **étape de mise à jour** (*update step*) — enseignée : [ch. 18, 18.4 · Sorties des neurones et erreur du réseau][s351]
- état → voir [état caché](#e), [état caché (d'un notebook)](#e), [état porté](#e), [état terminal, épisode tronqué](#e)
- **état caché** (*hidden state*) — enseignée : [ch. 22, 22.3 · L'état][s411] · exercices : 22.9, 22.Q7
- **état caché (d'un notebook)** (*hidden state*) — enseignée : [ch. 22, 22.3 · L'état][s411] ; [ch. B7, Ce que cache un notebook][s656] · exercices : 22.9, 22.Q7, B7.Q1
- **état porté** (*stateful RNN, carried state*) — enseignée : [ch. 24, 24.5.9 · Les RNN « stateful »][s442] · exercices : 24.27, 24.30, 24.Q11
- **état terminal, épisode tronqué** (*terminal state, truncation*) — enseignée : [ch. 26, 26.6.1 · Les valeurs Q et leur mise à jour][s462] · exercices : 26.3, 26.8, 26.36, B8.1, B8.4
- **état, transition** (*state*) — enseignée : [ch. 26, 26.3.1 · Premier temps : l'agent choisit une action][s458] · exercices : 22.9, 22.Q2, 22.Q4, 22.Q12, 24.27, 25.R2, 26.6, 26.20, 26.Q3, 26.Q8 (+ 90) · approfondie : ch. 24
- *Euler method* → voir [méthode d'Euler](#m)
- *evaluation* → voir [évaluation](#e)
- **évaluation** (*evaluation*) — enseignée : [ch. 11, 11.2.2 · L'évaluation][s292] · exercices : 8.23, 11.12, 11.Q1, CP2.5, 12.R1, 16.4, 16.Q7, 18.R3, CP4.4, 26.21 (+ 76) · approfondie : ch. B4, B5, B8
- **évaluation de LLM** — enseignée : [ch. B4, Ce que fait vraiment un LLM][s780]
- *evaluation mode* → voir [mode entraînement, mode évaluation](#m)
- **évaluation multi-graines** — enseignée : [ch. B8, 208.6 · Pratique du deep RL : hyperparamètres, graines,…][s788]
- **évaluation par distance de Fréchet** — enseignée : [ch. B5, La distance de Fréchet][s482]
- **événement, issue** (*event, outcome*) — enseignée : [ch. 0B, 101.7.1 · Expérience aléatoire, événement, probabilité, complémentaire][s130] ; [ch. 3, 3.3 · Probabilité simple][s131] · exercices : 0B.16, 0B.26, 0B.52, 0B.Q12, 3.1, 3.3, 3.Q1, 3.Q4, 6.1, 6.R2 (+ 8) · réutilisée : ch. 2, 14
- *event* → voir [événement, issue](#e)
- *evidence* → voir [évidence](#e)
- **évidence** (*evidence, marginal likelihood*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s218] · exercices : 4.1–3, 4.5–6, 4.14, 4.Q6, 4.Q9, CP1.8
- *evidence lower bound* → voir [ELBO](#e)
- **EWC** (*elastic weight consolidation*) — enseignée : [ch. B1, L'oubli catastrophique][s508]
- **exact match** (*correspondance exacte*) — enseignée : [ch. B4, Les métriques][s589] · exercices : B4.Q8
- **exception** — enseignée : [ch. 0A, 100.6.2 · Exceptions : lire un traceback, `raise`, `try`/`except`][s81] · exercices : 0A.25, 0A.37, 0A.39, 0A.58, 0A.63, 0B.38, 0B.43, 0B.45–46, 16.Q4 (+ 13)
- **exception écrasante** (*overwhelming exception*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s299] · exercices : 11.Q10
- **exceptions (raise, try/except), lecture d'un traceback** — enseignée : [ch. 0A, 100.6.2 · Exceptions : lire un traceback, `raise`, `try`/`except`][s81] · exercices : 0A.25, 0A.37, 0A.58
- *excessive agency* → voir [agentivité excessive](#a)
- *exclusive or* → voir [XOR (ou exclusif)](#x)
- *exec form* → voir [forme exec, forme shell](#f)
- *exécution* → voir [run](#r)
- **exemple adversarial** (*adversarial example*) — enseignée : [ch. 21, 21.10 · Les adversaires][s402] · exercices : 21.E4
- *exit code* → voir [code de sortie](#c)
- *expectation* → voir [espérance](#e)
- *expectation-maximization* → voir [algorithme EM](#a)
- *expected SARSA* → voir [SARSA espéré](#s)
- *expected value* → voir [espérance](#e)
- **experience replay** — enseignée : [ch. 26, 26.9 · Le rejeu d'expérience][s469]
- *experiment log* → voir [journal d'expériences](#j)
- *experiment tracking* → voir [journal d'expériences](#j)
- *expert system* → voir [système expert](#s)
- *explainability* → voir [explicabilité](#e)
- *explained variance* → voir [variance expliquée (d'un critique)](#v)
- *explained variance ratio* → voir [part de variance expliquée](#p)
- **explicabilité** (*explainability, interpretability*) — enseignée : [ch. B6, Explicabilité, équité et éthique][s382] ; [ch. 20, 20.8.1 · Une explication satisfaisante][s383] · exercices : 13.11, 14.9, B6.28, B6.Q1–Q6, PF.9 (+ 42)
- **explicabilité et dérive** — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s383]
- explication → voir [explication contrefactuelle](#e), [explication globale, explication locale](#e), [explication post hoc](#e)
- **explication contrefactuelle** (*counterfactual explanation*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s383] ; [ch. B6, Deux autres explications locales : substitut et contrefactuel][s384] · exercices : CP4.11, B6.8
- **explication globale, explication locale** (*global, local explanation*) — enseignée : [ch. B6, 206.2 · Explications globales : coefficients, arbres, importance…][s628] · exercices : B6.Q2
- explication locale → voir [explication globale, explication locale](#e)
- **explication post hoc** (*post-hoc explanation*) — enseignée : [ch. 13, 13.7 · Bilan : choisir un classifieur][s314] ; [ch. B6, Interprétable par construction, ou expliqué après coup][s322] · exercices : 13.11, B6.Q1
- *exploding gradient* → voir [gradient évanescent, explosif](#g)
- exploitation → voir [exploration / exploitation](#e)
- exploration → voir [exploration / exploitation](#e)
- **exploration / exploitation** — enseignée : [ch. 26, 26.6.2 · La politique du Q-learning][s155] ; [ch. 1][s144] · exercices : 1.22, 26.4–5, 26.10, 26.12, 26.16, 26.18, 26.E3, 26.Q5, 26.Q9 (+ 5) · réutilisée : ch. 11, B8
- *exploratory data analysis* → voir [analyse exploratoire](#a)
- explosif → voir [gradient évanescent, explosif](#g)
- *exponential linear unit* → voir [ELU](#e)
- *exponential moving average* → voir [moyenne mobile exponentielle](#m)
- **exponentielle et logarithmes (ln, log₂, log₁₀), changement de base, bits et nats** — enseignée : [ch. 0B, 101.2.4 · Logarithmes : ln, log₂, log₁₀, changement de base][s107] · exercices : 0B.16, 6.R3 · approfondie : ch. 6 · réutilisée : ch. 6, 18
- *export* → voir [exporter (un modèle), `torch.export`](#e)
- **exporter (un modèle), `torch.export`** (*export*) — enseignée : [ch. B7, 207.8 · Exporter et optimiser l'inférence : torch.export, ONNX,…][s671] · exercices : 25.35, B7.21
- *exposure bias* → voir [biais d'exposition](#b)
- **expression régulière** (*regular expression*) — enseignée : [ch. 0A, 100.6.5 · Expressions régulières (`re`) : les bases][s89] · exercices : 11.14
- **expressions régulières (re) : bases** — enseignée : [ch. 0A, 100.6.5 · Expressions régulières (`re`) : les bases][s89] · exercices : 0A.45, 0A.Q8
- *external validation* → voir [validation externe](#v)
- **extraction de features** (*feature extraction*) — enseignée : [ch. B1, 201.3 · Extraction de features : embeddings figés, linear probe, k…][s501] · exercices : B1.10, B1.E1, B1.Q5, B1.Q8, B4.R2 · réutilisée : ch. B4
- **ExtraTrees** (*extremely randomized trees*) — enseignée : [ch. 14, 14.6 · ExtraTrees][s325] · exercices : 13.22, 13.25, 14.16, 14.18, 14.Q7
- **extrema locaux et globaux** — enseignée : [ch. 5, Extrema globaux et locaux][s225] · exercices : 5.4
- *extremely randomized trees* → voir [ExtraTrees](#e)
- extremum global → voir [extremum local, extremum global](#e)
- **extremum local, extremum global** (*local extremum, global extremum*) — enseignée : [ch. 5, Extrema globaux et locaux][s225]

## F

- **F-beta** (*F-beta score*) — enseignée : [ch. 3, 3.7.11 · Le score F1][s208] · exercices : 3.16, 3.25
- *F-beta score* → voir [F-beta](#f)
- **F1** — enseignée : [ch. 3, 3.7.11 · Le score F1][s208] · exercices : 3.2, 3.5–6, 3.8, 3.16, 3.18, 3.25, 3.E4, 3.Q12, CP3.11 (+ 9)
- **F1-score** (*score F1*) — enseignée : [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s41] · exercices : 12.R1, 19.Q4, B4.28
- **facteur d'actualisation** (*discount factor*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s154] · exercices : 26.7, 26.E2
- *factorial* → voir [factorielle](#f)
- **factorielle** (*factorial*) — enseignée : [ch. 0B, 101.1.7 · Dénombrement : principe multiplicatif, factorielle,…][s13] · exercices : 0B.12 · réutilisée : ch. B6
- *fairness through unawareness* → voir [équité par aveuglement](#e)
- *fallacy* → voir [sophisme](#s)
- *false discovery rate* → voir [taux de fausses découvertes, de fausses omissions](#t)
- *false negative* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *false negative rate* → voir [taux de faux positifs, de faux négatifs](#t)
- *false omission rate* → voir [taux de fausses découvertes, de fausses omissions](#t)
- *false positive* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *false positive rate* → voir [taux de faux positifs, de faux négatifs](#t)
- **familles d'apprentissage** — enseignée : [ch. 1, 1.5 · Générateurs][s152] · exercices : 1.E3 · réutilisée : ch. 2
- **fan-in, fan-out** (*entrées, sorties d'une couche*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s338] · exercices : 16.Q10, 21.Q8
- fan-out → voir [fan-in, fan-out](#f)
- *fast gradient sign method* → voir [FGSM, PGD](#f)
- **FastAPI, uvicorn, pydantic** — enseignée : [ch. B7, En ligne ou par batches, et une API HTTP][s670] · exercices : B7.7, B7.9, B7.18–19, B7.24
- faux négatif → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- faux positif → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *FDR* → voir [taux de fausses découvertes, de fausses omissions](#t)
- **feature** (*caractéristique, variable explicative*) — enseignée : [ch. 0A, 100.9.5 · De pandas à NumPy : `X` et `y`][s4] · exercices : 0A.46, 1.9, 1.Q3, 7.30, 7.Q9, 8.25, 8.27, 9.30, 12.5, 12.R2 (+ 22) · approfondie : ch. 9, 12, B1 · réutilisée : ch. 15, 21, B4
- **feature bagging** (*tirage de features*) — enseignée : [ch. 14, 14.5 · Les forêts aléatoires][s318]
- *feature detector* → voir [détecteur de motif](#d)
- **feature engineering** (*ingénierie des features*) — enseignée : [ch. 1, 1.1.2 · Les systèmes experts et leurs limites][s141] · exercices : 1.Q2
- *feature extraction* → voir [extraction de features](#e)
- **feature learning** (*apprentissage des features*) — enseignée : [ch. 1, 1.7 · Deep learning][s35] · exercices : 1.Q10
- *feature map* → voir [carte d'activation](#c)
- *feature selection* → voir [sélection de features](#s)
- *feature visualization* → voir [visualisation de features](#v)
- **features aléatoires** (*random features*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s285] · exercices : 9.11, 9.30
- **features polynomiales** (*polynomial features*) — enseignée : [ch. 9, Ajuster une courbe : complexité et capacité][s282] ; [ch. 15, 15.9.5 · Les pipelines][s15] · exercices : 9.15, 9.17, 9.28, 9.Q7, CP2.5, 15.4, 15.22
- *featurewise* → voir [par échantillon, par feature, par élément](#p)
- *feed-forward network* → voir [réseau feed-forward, MLP](#r)
- *feedback* → voir [rétroaction](#r)
- *feedback loop* → voir [boucle de rétroaction](#b)
- fenêtre → voir [fenêtre glissante](#f), [fenêtre glissante (attention)](#f), [fenêtres de séries temporelles](#f)
- **fenêtre glissante** (*sliding window*) — enseignée : [ch. 22, 22.2 · Introduction : des fenêtres pour une couche dense][s408] ; [ch. B3, Les autres voies vers les contextes longs][s409]
- **fenêtre glissante (attention)** (*sliding window attention*) — enseignée : [ch. B3, Les autres voies vers les contextes longs][s409]
- **fenêtres de séries temporelles** — enseignée : [ch. 22, 22.2 · Introduction : des fenêtres pour une couche dense][s408]
- **fermeture** (*closure*) — enseignée : [ch. 0A, 100.5.4 · Fermetures (*closures*)][s79] · exercices : 0A.42–43, 0A.Q7, 18.26 · réutilisée : ch. 18
- **fermetures (closures)** — enseignée : [ch. 0A, 100.5.4 · Fermetures (*closures*)][s79] · exercices : 0A.42, 18.26 · réutilisée : ch. 18
- **fertilité** (*fertility*) — enseignée : [ch. B2, 202.6 · Coûts et biais de la tokenisation : fertilité, contexte,…][s531] · exercices : B2.8, B2.12–13, B3.R1, B4.7
- **fertilité et coût des tokens** — enseignée : [ch. B2, 202.6 · Coûts et biais de la tokenisation : fertilité, contexte,…][s531]
- *fertility* → voir [fertilité](#f)
- feuille → voir [racine, feuille, nœud interne](#r)
- **few-shot** (*à quelques exemples*) — enseignée : [ch. B1, Le few-shot et les prototypes de classe][s511] · exercices : B1.25, B4.15, B4.Q6
- **few-shot par prototypes** — enseignée : [ch. B1, Le few-shot et les prototypes de classe][s511] · exercices : B1.25
- FFN → voir [bloc feed-forward (FFN)](#b)
- **FGSM, PGD** (*fast gradient sign method, projected gradient descent*) — enseignée : [ch. 21, Au-delà du livre : l'attaque FGSM][s403] · exercices : 21.29, 21.E4, 21.Q11, B6.R1 · réutilisée : ch. B6
- fiche → voir [fiche modèle](#f), [fiche modèle d'un modèle dérivé](#f), [fiche modèle équité et explicabilité](#f)
- *fiche de données* → voir [data card](#d)
- **fiche modèle** (*model card*) — enseignée : [ch. B6, La fiche modèle][s271] ; [ch. 23, 23.9.5 · Sauvegarder le prétraitement][s272] · évoquée : [ch. 8, 8.3 · Le jeu de test][s143] · exercices : 7.10, 8.9, 23.10, 23.28, 24.15, 25.35, 29.Q7, B1.27, B6.9, B6.28 (+ 14) · approfondie : ch. B1 · réutilisée : ch. B1
- **fiche modèle d'un modèle dérivé** — enseignée : [ch. B1, Publier un modèle dérivé][s519]
- **fiche modèle équité et explicabilité** — enseignée : [ch. B6, La fiche modèle][s271]
- fichier → voir [fichier de référence figé](#f), [fichier de verrouillage](#f), [fichiers texte, pathlib](#f)
- **fichier de référence figé** (*golden file*) — enseignée : [ch. B2][s538]
- **fichier de verrouillage** (*lock file*) — enseignée : [ch. B7, Les versions figées et l'empreinte des données][s662] · exercices : B7.Q6
- **fichiers texte, pathlib** — enseignée : [ch. 0A, 100.6.3 · Fichiers texte et chemins avec `pathlib`][s719] · exercices : 0A.38, 0A.Q8 · réutilisée : ch. 11
- **FID** (*Fréchet Inception distance*) — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s481] · exercices : 27.22–23, 27.27, 27.E4, 27.R2, B5.Q10, B5.R2
- **filigrane** (*watermark*) — enseignée : [ch. B5, Transparence et provenance des contenus générés][s457] · exercices : 27.9, B5.10
- **fils rouges du workbook** — enseignée : [ch. 1, Les quatre fils rouges du workbook][s160]
- *filter* → voir [noyau (d'une convolution)](#n)
- **fine-tuning** (*réglage fin, affinage, affiner*) — enseignée : [ch. B1, 201.4 · Fine-tuning : partiel, complet, dégel progressif, learning…][s20] ; [ch. 1, Panorama 2026 : où en est l'IA ?][s21] · exercices : B1.17–19, B1.21, B1.23, B1.E1, B1.Q5, B1.Q7, B4.Q10, B4.R2 (+ 17) · réutilisée : ch. B4, B8
- **fine-tuning partiel et complet** — enseignée : [ch. B1, 201.4 · Fine-tuning : partiel, complet, dégel progressif, learning…][s20] · exercices : B1.3–4, B1.6, B1.17–19, B1.Q5–Q7, B1.R2 (+ 3)
- **fine-tuning partiel, complet** (*partial, full fine-tuning*) — enseignée : [ch. B1, 201.4 · Fine-tuning : partiel, complet, dégel progressif, learning…][s20] · exercices : B1.10, B1.17, B1.22, B1.Q5
- *fine-tuning supervisé* → voir [SFT](#s)
- *finite difference* → voir [différence finie (avant, arrière, centrée)](#d)
- *fitted attribute* → voir [attribut appris](#a)
- *fixed-length code* → voir [code de longueur fixe](#c)
- **fixture (pytest)** — enseignée : [ch. B7, Quatre familles de tests][s666] · exercices : B7.15
- **flagornerie** (*sycophancy*) — enseignée : [ch. B8, Le piratage de la récompense][s472] · exercices : B8.10
- **FlashAttention** — enseignée : [ch. B3, FlashAttention : le même résultat, sans la matrice][s572] · exercices : B3.27, B3.Q12
- *flatten* → voir [aplatissement](#a)
- fléau de la dimension → voir [malédiction de la dimension, fléau de la dimension](#m)
- *float* → voir [flottant](#f)
- *floor / ceiling* → voir [partie entière (plancher / plafond)](#p)
- **flottant** (*float*) — enseignée : [ch. 0A][s69] · exercices : 0A.62, 0B.37, 4.14, 5.11, 5.15–16, 5.18, 5.23, 6.12, 7.13 (+ 14)
- **flow matching** — enseignée : [ch. B5, Flow matching et rectified flow][s627] · exercices : B5.20, B5.E5, B5.Q4, B5.Q11
- **flux synchrone et asynchrone** — enseignée : [ch. 16, 16.3 · Flux synchrone et asynchrone][s339] · exercices : 16.5, 16.16, 16.E4, 16.Q5–Q6
- *FN* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *focal pixel* → voir [ancre, pixel focal](#a)
- **fold** (*pli*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s263] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21, 8.Q10, 15.R3, CP3.11, 24.Q4 (+ 49)
- fonction → voir [fonction d'activation](#f), [fonction de masse (pmf)](#f), [fonction de Rosenbrock](#f), [fonction de transfert](#f), [fonction de valeur](#f), [fonction de valeur (RL)](#f), [fonction fit() avec historique](#f), [fonction score](#f), [fonctions affines et polynômes](#f), [fonctions de deux variables, lignes de niveau (lecture)](#f)
- **fonction d'activation** (*activation function*) — enseignée : [ch. 17, 17.2 · Ce que fait une fonction d'activation][s52] ; [ch. 10, 10.3.3 · Les neurones modernes][s50] · exercices : 0B.54, 5.Q9, 10.E3, 10.Q8, 17.Q1–Q4, 18.29, 19.28 (+ 39)
- **fonction de masse (pmf)** (*probability mass function*) — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s179]
- **fonction de Rosenbrock** (*Rosenbrock function*) — enseignée : [ch. 5, Descendre une surface pas à pas][s231] · exercices : 5.7, 5.15, 5.21, 19.15–19, 19.22, 19.26 · réutilisée : ch. 19
- **fonction de transfert** (*transfer function*) — enseignée : [ch. 17, 17.2.1 · La forme d'une activation][s343] · exercices : 17.Q2
- **fonction de valeur** (*value function*) — enseignée : [ch. B6, Partager un gain entre joueurs][s632] · exercices : B6.2
- **fonction de valeur (RL)** (*value function*) — enseignée : [ch. B8, 208.7 · RL et modèles de langue : RLHF, modèle de récompense, DPO,…][s690] · exercices : B8.8–9, B8.12–13, B8.17, B8.E3, B8.E5, B8.R1
- **fonction fit() avec historique** — enseignée : [ch. 23, 23.8.3 · Analyser l'historique][s426]
- **fonction score** (*score function*) — enseignée : [ch. B8, Apprendre la politique elle-même][s686] · exercices : B8.7, B8.Q6
- fonction signe → voir [marche unité, marche de Heaviside, fonction signe](#m)
- **fonction, paramètre, argument** (*function, parameter*) — enseignée : [ch. 0A, 100.5 · Fonctions][s76] ; [ch. 0B, 101.2.1 · Fonction, graphe, fonction affine][s77] · exercices : 0A.23, 0A.30, 0A.41–42, 0A.44, 0B.36, 0B.40, 0B.R3, 4.23, 5.Q2 (+ 52) · approfondie : ch. 10, 23 · réutilisée : ch. 1, 5, 6, 7, 9, 10, 11, 13, 16, 17, 18, 19, 21, 23, 24, 26, 27, B1
- **fonctions affines et polynômes** — enseignée : [ch. 0B, 101.2 · Fonctions usuelles][s726]
- **fonctions de deux variables, lignes de niveau (lecture)** — enseignée : [ch. 0B, 101.6.1 · Fonctions de deux variables et lignes de niveau][s126] · exercices : 0B.25, 0B.49, 0B.Q11, 5.17, 5.22, 5.Q7, 7.15 · approfondie : ch. 5 · réutilisée : ch. 5
- **fonctions, valeurs par défaut, portée** — enseignée : [ch. 0A, 100.5 · Fonctions][s76] ; [ch. 0B, 101.2.1 · Fonction, graphe, fonction affine][s77] · exercices : 0A.6, 0A.23, 0A.30, 0A.41–42, 0A.44, 0A.Q6, 0B.36, 0B.40, 17.Q8 (+ 14) · réutilisée : ch. 0B, 6, 10, 11, 13, 16, 18
- **fond** (*background*) — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s631] · exercices : 0B.13, 0B.Q4, 5.25, 19.14, 19.Q5, B6.1–2, B6.16, B6.Q4–Q5 (+ 30)
- *FOR* → voir [taux de fausses découvertes, de fausses omissions](#t)
- *forçage par l'enseignant* → voir [teacher forcing](#t)
- **force de régularisation ($\lambda$, `alpha`, `C`)** (*regularization strength*) — enseignée : [ch. 9, 9.5 · La régularisation][s276] · exercices : 9.31, CP2.5
- **forêt aléatoire** (*random forest*) — enseignée : [ch. 14, 14.5 · Les forêts aléatoires][s318] · exercices : 14.8, 14.16, 14.18, 14.E2, 14.E4, 14.Q6–Q7, 15.R1, 17.R2, B6.17 (+ 4) · réutilisée : ch. B6
- **forme** (*shape*) — enseignée : [ch. 0A, 100.8.4 · Broadcasting][s91] ; [ch. 0B, 101.4.1 · Forme, éléments, transposée ; un dataset est une matrice][s2] · exercices : 0A.7–8, 0A.52, 0A.56, 0A.Q11, 0B.20, 0B.42–43, 0B.Q9, 0B.R2 (+ 22)
- **forme (d'un tenseur)** (*shape*) — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s234] · exercices : 20.1–2, 20.8, 20.12–17, 20.Q2 (+ 7)
- **forme exec, forme shell** (*exec form, shell form*) — enseignée : [ch. B7, Image, conteneur et Dockerfile][s678] · exercices : B7.24
- **forme fermée (du processus direct)** (*closed form*) — enseignée : [ch. B5, La forme fermée][s606] · exercices : 9.17, B5.6–7, B5.11–14, B5.18–19, B5.22 (+ 5) · réutilisée : ch. 11, 25
- **forme matricielle des gradients** — enseignée : [ch. 18, 18.12.1 · La rétropropagation en un seul endroit][s748]
- forme shell → voir [forme exec, forme shell](#f)
- **formule des probabilités totales** (*law of total probability*) — enseignée : [ch. 3, 3.6 · Probabilité marginale et tables de contingence][s203] · exercices : 3.4, 4.3
- *forward pass* → voir [passe avant, passe arrière](#p)
- *forward process* → voir [processus direct](#p)
- *forward process posterior* → voir [posterior (d'un pas)](#p)
- **forward() avec branches et bloc résiduel** — enseignée : [ch. 24, 24.6.2 · Construire un modèle fonctionnel][s763]
- **fouille de textes et de données** (*text and data mining, TDM*) — enseignée : [ch. 28, Au-delà du livre : la génération d'images aujourd'hui][s486] · exercices : 28.9, B5.10
- *foundation model* → voir [modèle de fondation](#m)
- *four-fifths rule* → voir [règle des quatre cinquièmes](#r)
- **fourche de Hume** (*Hume's fork*) — enseignée : [ch. 11, 11.6 · Raisonnement combiné][s300] · exercices : 11.Q11
- *FP* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- **fraction clippée** (*clip fraction*) — enseignée : [ch. B8, Les journaux de PPO][s684] · exercices : B8.5, B8.21–22
- *fractionally strided convolution* → voir [convolution transposée](#c)
- **framework** (*cadriciel*) — enseignée : [ch. 15, 15.2 · scikit-learn, NumPy et SciPy][s18] ; [ch. 16, 16.3.1 · Le graphe en pratique][s19] · exercices : 23.E5
- *Fréchet distance* → voir [distance de Fréchet](#d)
- *Fréchet Inception distance* → voir [FID](#f)
- *free-running* → voir [agent libre, déclenché](#a)
- *freeze* → voir [geler (des couches)](#g)
- **fréquences naturelles, arbre des fréquences naturelles** (*natural frequencies, natural frequency tree*) — enseignée : [ch. 3, 3.8 · Appliquer la matrice de confusion : le piège de la prévalence][s206] · exercices : 3.10, 3.21
- *frequency bias* → voir [biais de fréquence (embeddings)](#b)
- **fréquentisme contre bayésianisme** — enseignée : [ch. 4][s221]
- *frequentist* → voir [fréquentiste, bayésien](#f)
- **fréquentiste, bayésien** (*frequentist, Bayesian*) — enseignée : [ch. 4, 4.2 · Fréquentiste et bayésien : deux façons de penser les…][s216] · exercices : 4.11, 4.13, 4.24, 4.E2, 4.E4, 4.Q1–Q4
- FRIA → voir [analyse d'impact sur les droits fondamentaux (FRIA)](#a)
- *Frobenius norm* → voir [norme de Frobenius](#n)
- **frontière de décision** (*decision boundary*) — enseignée : [ch. 3, 3.7.1 · Classer des échantillons][s204] ; [ch. 7, 7.2.1 · Classification binaire en 2D][s127] · exercices : 3.Q6, 7.11, 7.Q3, 7.Q5, 10.2, 10.Q8, 18.21 · réutilisée : ch. 10
- *frozen embedding* → voir [embedding figé](#e)
- **fuite de données** (*data leakage*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s38] ; [ch. 8, 8.3 · Le jeu de test][s143] ; [ch. 12, 12.2 · La règle d'or : apprendre la transformation sur l'entraînement][s158] · évoquée : [ch. B4, Les bases de données vectorielles][s159] · exercices : 8.3, 8.24, 8.Q6, 11.R2, 12.9, 12.31, 12.E1, 22.E4, 24.E4, 29.E3 (+ 9) · réutilisée : ch. 11, 13, 15, 29
- *full fine-tuning* → voir [fine-tuning partiel, complet](#f)
- *fully connected layer* → voir [couche dense](#c)
- *function* → voir [fonction, paramètre, argument](#f)
- *function approximation* → voir [approximation de fonction](#a)
- *functional API* → voir [API fonctionnelle](#a)
- *fundamental rights impact assessment* → voir [analyse d'impact sur les droits fondamentaux (FRIA)](#a)
- **fusion (BPE), rang** (*merge, rank*) — enseignée : [ch. B2, 202.3 · Byte Pair Encoding : apprendre et appliquer les fusions][s526] · exercices : B2.3–4, B2.10, B2.16–19, B2.E2, B2.Q4–Q5 (+ 20) · réutilisée : ch. B3
- **fusion d'adaptateur** (*merging*) — enseignée : [ch. B4, Fusionner l'adaptateur][s599]
- *fusion par rangs réciproques* → voir [reciprocal rank fusion (RRF)](#r)

## G

- **gabarit (de prompt)** (*prompt template*) — enseignée : [ch. B1, Le zéro-shot][s512] · exercices : 14.17, 14.24, B1.24, B1.Q10, B4.4, B4.14
- **GAE** (*generalized advantage estimation*) — enseignée : [ch. B8, GAE : une moyenne de toutes les longueurs][s691] · exercices : B8.4, B8.18, B8.20, B8.22, B8.Q7
- **gain (d'un split), gain d'information** (*impurity decrease, information gain*) — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s147] · exercices : 0A.55, 13.3, 13.20, 13.22, 13.24–25, 13.35, 13.Q9, 27.2, B3.23 (+ 22)
- gain d'information → voir [gain (d'un split), gain d'information](#g)
- GAN → voir [réseau antagoniste génératif (GAN)](#r)
- **GAN : générateur et discriminateur** — enseignée : [ch. 27, 27.4 · Implémenter un GAN][s474]
- **GAN conditionnel** (*conditional GAN*) — enseignée : [ch. 27, Au-delà du livre : le GAN conditionnel][s480] · exercices : 27.25
- **garde** (*guard*) — enseignée : [ch. B7, torch.export et ONNX][s672] · exercices : 19.R2, B3.Q11, B7.2, B7.7, B7.10, B7.14, B7.17, B7.19, B7.21, B7.Q10 (+ 2)
- *gate* → voir [porte](#p)
- *gated* → voir [accès contrôlé (dépôt)](#a)
- *gated recurrent unit* → voir [GRU](#g)
- *gated repository* → voir [accès contrôlé (dépôt)](#a)
- *Gaussian* → voir [Naïve Bayes (gaussien, multinomial)](#n)
- *Gaussian distribution* → voir [loi normale, gaussienne](#l)
- *Gaussian error linear unit* → voir [GELU](#g)
- gaussienne → voir [loi normale, gaussienne](#l)
- *GDPR (règlement 2016/679)* → voir [RGPD](#r)
- **geler (des couches)** (*freeze*) — enseignée : [ch. 24, 24.6 · L'API fonctionnelle][s171] ; [ch. 23][s432] · exercices : 23.24, 24.28, B1.10, B1.R2 · réutilisée : ch. 27
- **GELU** (*Gaussian error linear unit*) — enseignée : [ch. 10, 10.3.3 · Les neurones modernes][s50] ; [ch. 17, 17.6 · Les fonctions lisses][s110] · exercices : 17.2, 17.10, 17.14, 17.18, 17.23, 17.E4, 17.Q9, B3.16, B3.19, B3.28 · réutilisée : ch. B3
- *general-purpose AI model* → voir [modèle d'IA à usage général](#m)
- **généralisation** (*generalization*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s38] ; [ch. 7, 7.1 · Pourquoi ce chapitre ?][s142] · exercices : 1.14, 1.25, 1.Q11, 7.11
- **généralisation (principe inductif)** (*generalization*) — enseignée : [ch. 11, 11.5 · L'induction][s188] · exercices : 11.6, 11.17, 11.Q9–Q10
- **généralisation hâtive** (*hasty generalization*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s299] · exercices : 11.17, 11.Q10
- *generalization* → voir [généralisation](#g), [généralisation (principe inductif)](#g)
- *generalization error* → voir [erreur d'entraînement, erreur de généralisation](#e)
- *generalization gap* → voir [écart de généralisation](#e)
- *generalized advantage estimation* → voir [GAE](#g)
- *generated image* → voir [image de contenu, référence de style, image générée](#i)
- **générateur** (*generator*) — enseignée : [ch. 27, 27.4.2 · Le générateur][s86] ; [ch. 0A, 100.7.4 · Itérables et générateurs (`yield`)][s87] · exercices : 0A.50, 0A.Q9, 27.3, 27.12, 27.23, 27.25–26, 27.Q3, 27.Q6, 27.R2 (+ 28)
- **générateur (modèle génératif)** (*generator, generative model*) — enseignée : [ch. 1, 1.5 · Générateurs][s152] ; [ch. 27, 27.4.2 · Le générateur][s86] · exercices : 1.24, 1.Q9, 27.3, 27.12, 27.23, 27.25–26, 27.Q3, 27.Q6, 27.R2 (+ 25)
- génération → voir [génération contrainte](#g), [génération de texte avec température](#g), [génération de texte mot à mot](#g), [génération de texte par caractère et par mot](#g)
- *génération augmentée par recherche* → voir [RAG](#r)
- **génération contrainte** (*constrained decoding*) — enseignée : [ch. B4, La sortie structurée][s586]
- **génération de texte avec température** — enseignée : [ch. 22][s413]
- **génération de texte mot à mot** — enseignée : [ch. 28, 28.3 · Le deep dream][s487]
- **génération de texte par caractère et par mot** — enseignée : [ch. 24, 24.5.11 · Générer du texte][s767]
- *generative adversarial network* → voir [réseau antagoniste génératif (GAN)](#r)
- *generative model* → voir [générateur (modèle génératif)](#g)
- *generator* → voir [générateur](#g), [générateur (modèle génératif)](#g)
- *geometric sequence* → voir [suite géométrique, raison](#s)
- **géométrie de la surface d'erreur** — enseignée : [ch. 19, 19.2 · L'erreur comme géométrie][s750]
- **gestionnaire de contexte** (*context manager*) — enseignée : [ch. B7][s665] · exercices : B7.18
- **get_params/set_params/clone** — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s328]
- **Gini** — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s147] · exercices : 13.2–3, 13.20, 13.22, 13.24, 13.26, 13.E2, 13.Q9, 14.R1, CP3.3 (+ 1)
- *Gini impurity* → voir [impureté de Gini](#i)
- **git : clone, status, add, commit, push, pull --rebase, .gitignore, branches (aperçu)** — enseignée : [ch. 0A, 100.11.1 · git : `status`, `add`, `commit`, `push`, `pull` ;…][s100] · exercices : 0A.12
- *global* → voir [explication globale, explication locale](#e)
- *global average pooling* → voir [pooling moyen global](#p)
- *global context* → voir [contexte global, contexte local](#c)
- *global extremum* → voir [extremum local, extremum global](#e)
- *global vectors* → voir [GloVe](#g)
- Glorot → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- **GloVe** (*global vectors*) — enseignée : [ch. B2, Compter ou prédire : les deux se rejoignent][s541]
- *glue code* → voir [code de colle](#c)
- *golden file* → voir [fichier de référence figé](#f)
- *Goodhart's law* → voir [loi de Goodhart](#l)
- **goulot** (*bottleneck*) — enseignée : [ch. 25, 25.3 · Le plus simple des autoencodeurs][s153] · exercices : 25.1, 25.16, 25.22, 25.Q4–Q6, B3.Q1, B5.Q6
- **GPU, TPU, NPU** — enseignée : [ch. 1, 1.7 · Deep learning][s35] · exercices : 0B.30, 1.Q10, 10.9, 15.Q1, 16.Q7, 17.8, 19.9, 23.Q7, 24.7, B7.12 (+ 35) · approfondie : ch. 20
- GQA → voir [MQA, GQA](#m)
- **Grad-CAM** (*gradient-weighted class activation mapping*) — enseignée : [ch. B6, 206.6 · Grad-CAM : cartes d'activation pondérées par les gradients][s400] ; [ch. 21, 21.9.3 · Regarder les filtres (2) : les cartes d'activation][s401] · exercices : B6.4, B6.18, B6.20–22, B6.E4, B6.Q1, B6.Q8, PF.9
- **Grad-CAM par hooks** — enseignée : [ch. B6, 206.6 · Grad-CAM : cartes d'activation pondérées par les gradients][s400] · exercices : B6.20
- **gradient** — enseignée : [ch. 0B, 101.6.3 · Gradient et direction de plus grande pente][s54] ; [ch. 5, 5.4 · Le gradient][s125] · exercices : 0B.25, 0B.31, 0B.49–50, 0B.E1, 0B.Q11, 5.Q1, 5.Q7–Q8, 5.Q10 (+ 93) · approfondie : ch. 14, 18, 19, 22, 28, B6, B8 · réutilisée : ch. 5, 8, 9, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 27, 28, B3, B6, B8
- *gradient accumulation* → voir [accumulation des gradients](#a)
- *gradient ascent* → voir [montée de gradient](#m)
- *gradient ascent on the input* → voir [montée de gradient sur l'entrée](#m)
- **gradient boosting** — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s195] · exercices : 14.4, 14.7, 14.9, 14.25, 14.E5, 14.Q11, 15.R1, CP3.5, B6.15, B6.E1 (+ 21) · réutilisée : ch. 15, B6
- **gradient boosting par histogrammes** (*histogram-based gradient boosting*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s195]
- **gradient boosting stochastique** (*stochastic gradient boosting*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s195]
- **gradient clipping** (*écrêtage du gradient*) — enseignée : [ch. 19, Au-delà du livre : le gradient clipping][s369] ; [ch. 22, 22.6 · Entraîner un RNN][s353] · exercices : 19.19, 19.E4, 22.Q6, 22.R2, B3.Q8, B3.R2, B8.R2 · réutilisée : ch. B3, B8
- **gradient de politique** (*policy gradient*) — enseignée : [ch. B8, 208.3 · Gradient de politique : politique paramétrée, REINFORCE,…][s473] ; [ch. 26, 26.10 · Deux applications][s470] · exercices : B8.7, B8.E1, B8.Q5, B8.Q8, B8.R3
- *gradient descent* → voir [descente de gradient](#d)
- **gradient évanescent et clipping** — enseignée : [ch. 22, 22.6 · Entraîner un RNN][s353]
- **gradient évanescent, explosif** (*vanishing, exploding gradient*) — enseignée : [ch. 0B, 101.1.5 · Suites géométriques et leur somme][s106] ; [ch. 22, 22.6 · Entraîner un RNN][s353] · exercices : 18.E3, 20.30, 22.7, 22.E2, 22.Q6, B3.R2
- **gradient numérique** — enseignée : [ch. 5, Au-delà du livre (3) : la différentiation automatique][s233] · exercices : 0B.51, 5.9, 5.15–16, 5.21, 5.23, 16.19, 18.9, 18.16 · réutilisée : ch. 16
- *gradient penalty* → voir [pénalité de gradient (WGAN-GP)](#p)
- **gradient × entrée** (*gradient × input*) — enseignée : [ch. B6, 206.5 · Attributions pour les réseaux : saillance, gradient ×…][s639] · exercices : B6.3, B6.19, B6.Q7
- *gradient × input* → voir [gradient × entrée](#g)
- *gradient-weighted class activation mapping* → voir [Grad-CAM](#g)
- **gradients intégrés** (*integrated gradients*) — enseignée : [ch. B6, Les gradients intégrés][s640] · exercices : B6.3, B6.18–19, B6.22, B6.Q7, PF.9
- *gradual unfreezing* → voir [dégel progressif](#d)
- **graine** (*seed*) — enseignée : [ch. B7, Les graines et le déterminisme][s95] ; [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s7] · exercices : 0A.31, 0A.E4, 0A.Q11, 2.14, 2.E5, 2.Q3, 14.27, 18.23, 23.29, 24.9 (+ 17) · approfondie : ch. 2, 26 · réutilisée : ch. 3
- *Gram matrix* → voir [matrice de Gram](#m)
- *grand modèle de langage* → voir [LLM](#l)
- **grande dimension** — enseignée : [ch. 2, 2.7 · Espaces de grande dimension][s193] · exercices : 2.25, 2.Q11, 5.10, 7.Q10–Q11, 13.15, 19.Q3, 25.6 · réutilisée : ch. 25
- **graphe de calcul** (*computational graph*) — enseignée : [ch. 0B, 101.6.4 · Règle de la chaîne à plusieurs variables : la somme sur…][s128] ; [ch. 18, 18.12.3 · Ce qu'elle fait][s129] · exercices : 0B.29, 23.18, 23.20, 23.22, 23.24
- **graphe orienté acyclique (DAG)** (*directed acyclic graph*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s338] · exercices : 16.Q4, 16.R2
- **graphique de dépendance (SHAP)** (*dependence plot*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s635] · exercices : B6.7–8, B6.11, B6.14–17, B6.E2, B6.Q5–Q6 (+ 3)
- *graphique en cascade* → voir [waterfall](#w)
- **ε-greedy** (*epsilon-greedy*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s156] ; [ch. 26, 26.6.2 · La politique du Q-learning][s155] · exercices : 11.8, 11.10, 11.20–22, 11.26, 26.5, 26.19, 26.Q9, 26.R3 (+ 12)
- *greedy algorithm* → voir [algorithme glouton](#a)
- *greedy decoding* → voir [décodage glouton](#d)
- **ε-greedy et softmax (Boltzmann)** — enseignée : [ch. 26, 26.6.2 · La politique du Q-learning][s155] · exercices : 26.5, 26.19, 26.E3, 26.Q9, 26.R3
- *greedy policy* → voir [politique gloutonne](#p)
- *grid* → voir [grille (d'hypothèses)](#g)
- *grid search* → voir [recherche en grille](#r)
- **GridSearchCV** — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s328] · exercices : 13.18, 15.1, 15.8, 15.18, 15.22, 15.24, 15.28, 15.31, 15.Q10, CP3.1 (+ 7) · réutilisée : ch. 24
- **grille (d'hypothèses)** (*grid*) — enseignée : [ch. 0A, 100.6.6 · `itertools` et `heapq` (aperçu)][s223] ; [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s222] · exercices : 0A.46, 4.19, 9.7, 9.26, 15.1, 15.20, 15.E2, 15.Q10, CP3.14, 18.R2 (+ 53) · réutilisée : ch. 19
- *ground truth* → voir [vérité terrain](#v)
- *group k-fold* → voir [découpage par groupes](#d)
- **group norm** (*group normalization*) — enseignée : [ch. B5, Un bloc résiduel conditionné par le pas][s619] · exercices : B5.21, B5.Q6
- *group normalization* → voir [group norm](#g)
- *group relative policy optimization* → voir [GRPO](#g)
- *group-specific thresholds* → voir [seuils par groupe](#s)
- **groupe de paramètres** (*parameter group, param group*) — enseignée : [ch. B1, Learning rates discriminatifs][s504] · exercices : B1.3
- *grouped-query attention* → voir [MQA, GQA](#m)
- **GroupKFold et TimeSeriesSplit (usage)** — enseignée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s268] · exercices : 8.19
- **GRPO** (*group relative policy optimization*) — enseignée : [ch. B8, GRPO et les récompenses vérifiables][s699] · exercices : B8.Q11
- **GRU** (*gated recurrent unit*) — enseignée : [ch. 22, 22.7 · LSTM et GRU][s415] · exercices : 22.7, 22.17, 22.19, 22.29, 22.E3, 22.Q9, 23.R1, CP5.1, CP5.11
- *guard* → voir [garde](#g)
- **guidance par classifieur** (*classifier guidance*) — enseignée : [ch. B5, Donner la classe au réseau][s622]
- **guidance sans classifieur (CFG)** (*classifier-free guidance*) — enseignée : [ch. B5, La guidance sans classifieur][s623] · exercices : B5.1, B5.21, B5.26–27, B5.E4, B5.Q9
- *guidance scale* → voir [échelle de guidance](#e)
- **Guided Grad-CAM** — enseignée : [ch. B6, Avec PyTorch : deux hooks][s379]

## H

- *Hadamard product* → voir [produit de Hadamard](#p)
- **hallucination** — enseignée : [ch. B4, Les hallucinations][s174] · évoquée : [ch. 1][s144] · exercices : 28.Q5, B4.9, B4.E1, B4.Q7
- *halting problem* → voir [problème de l'arrêt](#p)
- handler → voir [logger, handler](#l)
- *hard voting* → voir [vote dur, vote souple](#v)
- *harmonic mean* → voir [moyenne harmonique](#m)
- *hash* → voir [somme de contrôle](#s)
- *hasty generalization* → voir [généralisation hâtive](#g)
- He → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- *He initialization* → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- *head* → voir [tête (d'un modèle)](#t)
- *Heaviside step* → voir [marche unité, marche de Heaviside, fonction signe](#m)
- *held-out test set* → voir [test scellé](#t)
- **héritage** (*inheritance*) — enseignée : [ch. 0A, 100.7.3 · Héritage et `super()`][s85] · exercices : 0A.49, 0A.Q9, 15.8, 15.29, 20.18 · réutilisée : ch. 15, 20, 24
- **héritage et super()** — enseignée : [ch. 0A, 100.7.3 · Héritage et `super()`][s85] · exercices : 0A.49
- *Hessian matrix* → voir [matrice hessienne](#m)
- *hidden layer* → voir [couche cachée](#c)
- *hidden size* → voir [taille de l'état](#t)
- *hidden state* → voir [état caché](#e), [état caché (d'un notebook)](#e)
- *hierarchical learning* → voir [apprentissage hiérarchique](#a)
- *hierarchical navigable small world* → voir [HNSW](#h)
- **hinge loss** (*perte charnière*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s210]
- **HistGradientBoosting** — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s195] · exercices : 14.26–27, B6.15
- *histogram* → voir [histogramme](#h)
- *histogram-based gradient boosting* → voir [gradient boosting par histogrammes](#g)
- **histogramme** (*histogram*) — enseignée : [ch. 2, 2.9 · Le quartet d'Anscombe][s194] ; [ch. 14, Au-delà du livre : le gradient boosting][s195] · exercices : 0A.36, 0A.59–60, 2.16, 2.18, 2.20, 2.22, 14.26, 16.22, 17.21 (+ 10) · réutilisée : ch. 3
- **historique (d'entraînement)** (*training history*) — enseignée : [ch. 23, 23.8.3 · Analyser l'historique][s426] · exercices : 4.16–17, 7.9, 13.11, 20.11, 23.18–19, 23.25, 23.27, 24.R1 (+ 19)
- **hiver de l'IA** (*AI winter*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s288] · exercices : 10.10
- **HNSW** (*hierarchical navigable small world*) — enseignée : [ch. B4, Les bases de données vectorielles][s159]
- **hold-out** (*mise de côté*) — enseignée : [ch. 8, 8.3 · Le jeu de test][s143] · exercices : 8.1, 8.13, 8.23, 8.E3
- **hook** — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s3] ; [ch. B6, Avec PyTorch : deux hooks][s379] · exercices : 21.24, 21.26–27, 28.11, B3.23, B3.28, B5.21, B6.18, B6.20–21 · approfondie : ch. 28, B6 · réutilisée : ch. 28, B6
- **hooks PyTorch** — enseignée : [ch. 28][s399] · réutilisée : ch. B6
- **horodatage** (*time stamp*) — enseignée : [ch. 16, 16.3 · Flux synchrone et asynchrone][s339] · exercices : 16.5, 16.Q3, 16.Q6–Q7
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
- hypercube → voir [hypersphère, hypercube](#h)
- *hyperparameter* → voir [hyperparamètre](#h)
- **hyperparamètre** (*hyperparameter*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s47] · exercices : 1.E2, 1.Q5, 8.2, 15.5, 15.Q3, 19.11, 24.1, 24.7, 24.Q1, 24.Q5 (+ 50) · approfondie : ch. 8, 24 · réutilisée : ch. 7, 8, 19
- **hyperplan** (*hyperplane*) — enseignée : [ch. 7, 7.3 · Plusieurs classes, plusieurs dimensions][s252] ; [ch. 10, 10.3.1 · Le perceptron][s253] · exercices : 7.7, 10.23, 10.Q7, 11.Q1, 13.4
- *hyperplane* → voir [hyperplan](#h)
- *hypersphere* → voir [hypersphère, hypercube](#h)
- **hypersphère, hypercube** (*hypersphere*) — enseignée : [ch. 7][s262]
- *hypertrucage* → voir [deepfake](#d)
- hypothèse → voir [hypothèse de la variété](#h), [hypothèse distributionnelle](#h), [hypothèse du monde clos](#h), [hypothèse nulle](#h), [hypothèses multiples](#h)
- **hypothèse de la variété** (*manifold hypothesis*) — enseignée : [ch. 7, 7.6 · La malédiction de la dimension][s192]
- **hypothèse distributionnelle** (*distributional hypothesis*) — enseignée : [ch. B2, L'hypothèse distributionnelle][s535] · exercices : B2.26, B2.Q10
- **hypothèse du monde clos** (*closed-world assumption*) — enseignée : [ch. 11, 11.4 · La déduction][s297]
- **hypothèse nulle** (*null hypothesis*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s269]
- **hypothèse, observation** (*hypothesis*) — enseignée : [ch. 4, 4.7 · Plusieurs hypothèses][s217] · exercices : 4.6–7, 4.24, 4.Q7, 4.Q10, CP1.8, 7.E2, 8.E4, 13.E4, B2.Q10 (+ 13)
- **hypothèses multiples** — enseignée : [ch. 4, 4.7 · Plusieurs hypothèses][s217]
- *hypothesis* → voir [hypothèse, observation](#h)
- *hypothetico-deductive method* → voir [méthode hypothético-déductive](#m)

## I

- **i.i.d.** (*independent and identically distributed*) — enseignée : [ch. 2, 2.4.1 · Des variables i.i.d.][s186] · exercices : 2.E2, 2.Q8
- *id* → voir [identifiant (d'un token)](#i)
- *ideal denoiser* → voir [débruiteur parfait (d'un dataset fini)](#d)
- **identifiant (d'un token)** (*id*) — enseignée : [ch. B2, Un modèle ne lit que des identifiants][s521] · exercices : 0A.45, 13.21, B2.3–5, B2.13, B2.15, B2.Q1, B2.R2, B7.14 (+ 24)
- **identifiant de run** (*run ID*) — enseignée : [ch. B7][s665] · exercices : B7.14, B7.R1
- *identity matrix* → voir [matrice identité](#m)
- **if/for/while, enumerate, zip, compréhensions** — enseignée : [ch. 0A, 100.4.2 · Boucles : `for`, `range`, `enumerate`, `zip`, `while`,…][s716] · exercices : 0A.3, 0A.21, 0A.Q5
- *illicit major* → voir [majeur illicite, mineur illicite](#m)
- *illicit minor* → voir [majeur illicite, mineur illicite](#m)
- **im2col** (*unfold*) — enseignée : [ch. 21, 21.3.2 · La vue « œil de mouche »][s396] · exercices : 21.14, 21.Q3
- image → voir [conteneur, image](#c)
- **image de contenu, référence de style, image générée** (*content image, style reference, generated image*) — enseignée : [ch. 28, 28.4 · Le transfert de style][s488] · exercices : 21.6, 28.1, 28.3, 28.9, 28.26, 28.Q6, 28.Q9, B5.10, B5.E1, B5.R2
- image générée → voir [image de contenu, référence de style, image générée](#i)
- **images en arrays (N, H, W), (N, C, H, W)** — enseignée : [ch. 0A, 100.8.8 · Les images sont des arrays : MNIST][s722] · exercices : 0A.54, 0A.Q10
- *implementation invariance* → voir [sensibilité, invariance à l'implémentation](#s)
- *implicit weights* → voir [poids implicites](#p)
- importance → voir [importance des features (MDI)](#i), [importance par permutation](#i), [importance par retrait](#i)
- **importance des features (MDI)** (*mean decrease in impurity*) — enseignée : [ch. 13, 13.5.3 · Limiter l'overfitting][s317] ; [ch. 14, 14.5 · Les forêts aléatoires][s318] · exercices : 14.19, 14.E4, B6.R2
- **importance par permutation** (*permutation importance*) — enseignée : [ch. B6, L'importance par permutation][s326] ; [ch. 14, 14.5 · Les forêts aléatoires][s318] · exercices : 14.19, B6.12, B6.28, B6.Q1–Q2, B6.R2
- **importance par retrait** (*drop-column importance*) — enseignée : [ch. 14][s324]
- *impossibility theorem* → voir [théorème d'impossibilité](#t)
- **impureté** (*impurity*) — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s147] · exercices : 13.2–3, 13.20, 13.24–26, 13.Q9, 14.19, 14.Q7, 14.R1 (+ 1)
- **impureté de Gini** (*Gini impurity*) — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s147] · exercices : 13.2–3, 13.Q9, 14.R1
- *impurity* → voir [impureté](#i)
- *impurity decrease* → voir [gain (d'un split), gain d'information](#g)
- **imputation** — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s304] · exercices : 12.9–10, 12.31
- **imputation (d'enquête)** (*survey imputation*) — enseignée : [mini-projet 3, Le contexte][s712]
- *in-context learning* → voir [apprentissage en contexte](#a)
- *inceptionism* → voir [deep dream](#d)
- **indépendance** (*independence*) — enseignée : [ch. 0B, 101.7.2 · Indépendance][s132] · exercices : 0B.26, 0B.52, 0B.Q12, 1.1, 2.20, 2.E4, 3.14, 4.8, 16.7, 22.28 (+ 1) · réutilisée : ch. 2, 3, 6, 13, 14
- **indépendance conditionnelle** (*conditional independence*) — enseignée : [ch. 13, 13.6 · Naïve Bayes][s140] · évoquée : [ch. 4][s221]
- indépendante → voir [randomisation en cascade, indépendante](#r)
- *independence* → voir [indépendance](#i)
- *independent and identically distributed* → voir [i.i.d.](#i)
- *independent randomization* → voir [randomisation en cascade, indépendante](#r)
- **index (recherche sémantique)** — enseignée : [ch. 0A, 100.9.1 · `DataFrame` et `Series` : `read_csv`, `head`, `info`,…][s96] ; [ch. B2, Un numéro n'est pas une représentation][s548] · exercices : 0A.4, 0A.33, 0A.38, 3.24, 5.10, 7.24, B2.28, B4.20–21, B4.30 (+ 4)
- **indicateurs d'équité par groupe** — enseignée : [ch. B6, 206.9 · Équité : indicateurs de groupe (parité démographique,…][s647] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- **indice de Rand ajusté (ARI)** (*adjusted Rand index*) — enseignée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s260] · exercices : 7.17–18, 15.13, 15.15
- *individual conditional expectation* → voir [courbe ICE](#c)
- **induction** — enseignée : [ch. 11, 11.5 · L'induction][s188] · exercices : 11.6, 11.9, 11.12, 11.18, 11.Q6, 11.Q10, 11.Q12, CP2.9, 14.R2
- **induction paresseuse** (*slothful induction, appeal to coincidence*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s299] · exercices : 11.Q10, 14.R2
- *inductive bias* → voir [biais inductif](#b)
- **inégalité de Kraft** (*Kraft inequality*) — enseignée : [ch. 6, 6.6 · Les codes adaptatifs][s240]
- *inertia* → voir [inertie](#i)
- **inertie** (*inertia, within-cluster sum of squares*) — enseignée : [ch. 7, Sans labels : k-means][s170] · exercices : 7.3, 7.26–27, 7.29, 7.Q8, 8.R1, CP2.3, 15.13, 15.Q6
- **inférence avec eval et inference_mode** — enseignée : [ch. 23, 23.7 · Entraîner le modèle][s760]
- *inference mode* → voir [mode inférence](#m)
- **inférence par batches, en ligne** (*batch inference, online inference*) — enseignée : [ch. B7, 207.7 · Empaqueter et servir : artefact de modèle, inférence par…][s669] · exercices : 0B.32, 2.24, 2.32, 3.2, B7.5–6, B7.10–11, B7.18, PF.11 (+ 29)
- *inference to the best explanation* → voir [abduction](#a)
- *influential point* → voir [point influent](#p)
- **information (de Shannon), surprise** (*self-information, surprisal*) — enseignée : [ch. 6, Information : un mot, deux sens][s236] · exercices : 6.1, 6.11, 6.Q1–Q2, 6.Q5, 13.Q9, 26.Q2–Q3, B4.15, B5.Q4 (+ 43) · approfondie : ch. 13 · réutilisée : ch. 22, B2
- **information de Shannon** — enseignée : [ch. 6, Information : un mot, deux sens][s236]
- *information gain* → voir [gain (d'un split), gain d'information](#g)
- **information mutuelle** (*mutual information*) — enseignée : [ch. 12, 12.6 · La sélection de features][s310] · exercices : B2.22, B5.11 · réutilisée : ch. B2
- *ingénierie des features* → voir [feature engineering](#f)
- *inheritance* → voir [héritage](#h)
- initialisation → voir [initialisation des poids (LeCun, Glorot, He)](#i), [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i), [initialisation optimiste](#i)
- **initialisation des poids (LeCun, Glorot, He)** — enseignée : [ch. 16, 16.4.1 · L'initialisation][s340] · exercices : B7.Q5
- **initialisation LeCun, Glorot (Xavier), He (Kaiming)** (*LeCun, He initialization*) — enseignée : [ch. 16, 16.4.1 · L'initialisation][s340]
- **initialisation optimiste** (*optimistic initial values*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s156] · exercices : 11.22
- **injection de prompt** (*prompt injection*) — enseignée : [ch. B4, L'injection de prompt][s600] · exercices : B4.24, B4.Q12
- *inner product* → voir [produit scalaire](#p)
- *input layer* → voir [couche d'entrée](#c)
- **installation éditable** (*editable install*) — enseignée : [ch. B7, 207.1 · Du notebook au projet : limites du notebook, structure…][s99] · exercices : B7.9, B7.Q1
- *instruct model* → voir [modèle instruct](#m)
- *integrated gradients* → voir [gradients intégrés](#g)
- **intégration continue (CI)** (*continuous integration*) — enseignée : [ch. B7, L'intégration continue][s667] · exercices : B7.6, B7.9–11, B7.15, B7.23, B7.25, B7.Q2, B7.Q7, PF.10
- **interaction** (*interaction term*) — enseignée : [ch. 9, Ajuster une courbe : complexité et capacité][s282] · exercices : 9.15, 11.4, 11.21, 29.R2, B6.1–2, B8.8, B8.Q8
- *interaction term* → voir [interaction](#i)
- *intercept* → voir [ordonnée à l'origine](#o)
- *internal node* → voir [racine, feuille, nœud interne](#r)
- **interpolation linéaire** (*linear interpolation*) — enseignée : [ch. 1][s144] · exercices : 1.2, 25.20 · réutilisée : ch. 2
- **interpolation sphérique (slerp)** (*spherical linear interpolation*) — enseignée : [ch. 25, 25.5.3 · Mélanger des variables latentes][s453] · exercices : 25.6
- *interpolation threshold* → voir [seuil d'interpolation](#s)
- **interprétabilité mécaniste** (*mechanistic interpretability*) — enseignée : [ch. B6, Global ou local][s643]
- *interpretability* → voir [explicabilité](#e)
- *interpretable by design* → voir [interprétable par construction](#i)
- **interprétable par construction** (*interpretable by design*) — enseignée : [ch. B6, Interprétable par construction, ou expliqué après coup][s322] · exercices : B6.Q1
- *interquartile mean* → voir [moyenne interquartile (IQM)](#m)
- *interquartile range* → voir [écart interquartile](#e)
- *intersectional analysis* → voir [analyse intersectionnelle](#a)
- *interval decay* → voir [planning par paliers](#p)
- **intervalle de confiance** (*confidence interval*) — enseignée : [ch. 2, 2.6 · Le bootstrap][s189] ; [ch. B7, Rapporter un score : plusieurs graines et un intervalle][s191] · exercices : 2.10, 2.22–23, 2.E3, 2.Q10, 4.11, 4.25, CP1.5, 8.4 · réutilisée : ch. 26, B6
- **intervalle de crédibilité** (*credible interval*) — enseignée : [ch. 4, Au-delà du livre (3) : MAP et intervalle de crédibilité][s45] · exercices : 4.22, 4.25
- *interventional* → voir [Shapley interventionnel, observationnel](#s)
- invariance à l'implémentation → voir [sensibilité, invariance à l'implémentation](#s)
- *inverse document frequency* → voir [TF-IDF](#t)
- **inverse, déterminant** (*determinant*) — enseignée : [ch. 0B, 101.4.4 · Identité et idée de l'inverse][s119] · exercices : 0B.21, 0B.30, 0B.46, 4.2, 5.8, 6.18, 9.17, 12.4, 12.16, B2.21 (+ 19) · approfondie : ch. 12, 18 · réutilisée : ch. 9
- **inversion DDIM** (*DDIM inversion*) — enseignée : [ch. B5, Pourquoi c'est plus rapide][s617]
- *inverted dropout* → voir [dropout inversé](#d)
- IQM → voir [moyenne interquartile (IQM)](#m)
- *IQR* → voir [écart interquartile](#e)
- *isotonic regression* → voir [calibration de Platt, régression isotone](#c)
- issue → voir [événement, issue](#e)
- **itérables et générateurs (yield)** — enseignée : [ch. 0A, 100.7.4 · Itérables et générateurs (`yield`)][s87] · exercices : 0A.50, 0A.Q9
- **itération sur les valeurs** (*value iteration*) — enseignée : [ch. 26, 26.6.4 · L'éléphant dans la pièce][s467] · exercices : 26.26
- **itertools, heapq (aperçu)** — enseignée : [ch. 0A, 100.6.6 · `itertools` et `heapq` (aperçu)][s223] · exercices : 0A.46, 9.14–15, 11.15 · réutilisée : ch. 11

## J

- *Jacobian* → voir [Jacobienne](#j)
- **Jacobienne** (*Jacobian*) — enseignée : [ch. 17, 17.8 · La softmax][s250] · évoquée : [ch. 16][s341] · exercices : 16.10, 17.7, 17.16, 22.6, 22.19, B3.R2–R3
- **Jacobienne de la softmax** (*softmax Jacobian*) — enseignée : [ch. 17, 17.8 · La softmax][s250] · exercices : 17.7, 17.16
- *Jensen-Shannon divergence* → voir [divergence de Jensen-Shannon](#d)
- **jeu à somme nulle** (*zero-sum game*) — enseignée : [ch. 27, 27.4.4 · Le jeu][s245] · exercices : 26.37, 27.2, 27.Q5
- **jeu coopératif** (*cooperative game*) — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s631] · exercices : B6.1
- **jeu d'entraînement / jeu de test** (*training set / test set*) — enseignée : [ch. 8, 8.3 · Le jeu de test][s143] ; [ch. 1, 1.2.3 · Généraliser : le jeu de test][s38] · exercices : 1.E2, 1.Q6, 8.1, 8.3–4, 8.7, 8.11, 8.E1, 8.Q5–Q6 (+ 17) · réutilisée : ch. 3
- **jeu de test** — enseignée : [ch. 8, 8.3 · Le jeu de test][s143] ; [ch. 1, 1.2.3 · Généraliser : le jeu de test][s38] · exercices : 1.E2, 1.Q6, 8.1, 8.3–4, 8.7, 8.11, 8.E1, 8.Q5–Q6 (+ 43) · réutilisée : ch. 3
- **jeu de validation** (*validation set*) — enseignée : [ch. 8, 8.4 · Le jeu de validation][s168] ; [ch. 1, 1.2.3 · Généraliser : le jeu de test][s38] · exercices : 3.29, 8.1–2, 8.6, 8.18, 8.Q7–Q9, 9.9, 10.18 (+ 6) · réutilisée : ch. 13, 14
- job → voir [workflow, job, étape, action](#w)
- *join* → voir [jointure](#j)
- *joint probability* → voir [probabilité jointe](#p)
- **jointure** (*join, merge*) — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s313] · exercices : 12.14–15
- joueur nul → voir [efficacité, symétrie, joueur nul, additivité](#e)
- **journal d'expériences** (*experiment log, experiment tracking*) — enseignée : [ch. 24, 24.2.7 · Observations][s436] · exercices : 11.25, 24.9, 24.15, B7.14, B7.R1, PF.5 · réutilisée : ch. B7
- *journalisation* → voir [logging](#l)
- *JSON Lines* → voir [JSONL](#j)
- **json, pickle** — enseignée : [ch. 0A, 100.6.4 · Sérialiser : `json` et `pickle`][s88] · exercices : 0A.39, 11.25, 15.30, 23.22, 24.15, 26.34, B1.5, B1.27, B2.30, B4.15 (+ 26) · réutilisée : ch. 11, 23, B4, B7
- **JSONL** (*JSON Lines*) — enseignée : [ch. B7, 207.5 · Suivi d'expériences : runs, métriques, artefacts, comparaison][s437] · exercices : 24.15, B7.14, B7.R1, PF.4–5, PF.7–8, PF.11
- **juge LLM** (*LLM-as-a-judge*) — enseignée : [ch. B4, Le juge LLM][s590] · exercices : B4.Q8

## K

- **k plus proches voisins (kNN)** (*k-nearest neighbors*) — enseignée : [ch. 13, 13.3 · Les k plus proches voisins][s257] · exercices : 13.1, 13.E1, 26.Q1, B1.2, B1.13, B1.Q5, B1.Q8, B2.R1, B4.R3 · réutilisée : ch. B1, B4
- **k-fold** (*validation croisée à $k$ plis*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s263] · exercices : 8.14, 8.19, 8.21, 8.23, 8.Q10 · réutilisée : ch. 12, 15
- **k-fold répétée** (*repeated k-fold*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s263]
- **k-fold stratifiée** — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s263] · exercices : 8.21
- **k-means** (*k-moyennes*) — enseignée : [ch. 7, Sans labels : k-means][s170] · évoquée : [ch. 1, Les quatre fils rouges du workbook][s160] · exercices : 7.3, 7.12, 7.17, 7.25–27, 7.E2, 7.Q8, CP2.3, 13.Q3 (+ 16) · réutilisée : ch. 8, 15
- **k-means++** — enseignée : [ch. 7, Le départ compte : minima locaux et k-means++][s258] · exercices : 7.25–26, CP2.3
- *k-moyennes* → voir [k-means](#k)
- *k-nearest neighbors* → voir [k plus proches voisins (kNN)](#k)
- Kaiming → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- *keep-alive* → voir [connexion persistante](#c)
- *kernel* → voir [noyau](#n), [noyau (d'un SVM)](#n), [noyau (d'une convolution)](#n), [noyau (d'une opération)](#n)
- **kernel (dans le code : `kernel_size`)** (*noyau, filtre*) — enseignée : [ch. 21, 21.2.5 · Le noyau][s30] · exercices : 13.16–18, 15.1, 21.17, B3.27, B6.11
- *Kernel SHAP* → voir [KernelSHAP](#k)
- *kernel trick* → voir [astuce du noyau](#a)
- **KernelSHAP** (*Kernel SHAP*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s635] · exercices : B6.7, B6.11
- *key* → voir [requête, clé, valeur](#r)
- *key-value cache* → voir [KV cache](#k)
- *kink* → voir [point anguleux](#p)
- **KL approchée** (*approximate KL*) — enseignée : [ch. B8, Les journaux de PPO][s684] · exercices : B8.21–22
- *KL divergence* → voir [divergence KL, divergence de Kullback-Leibler](#d)
- *KL penalty* → voir [pénalité KL](#p)
- **kNN** — enseignée : [ch. 13, 13.3 · Les k plus proches voisins][s257] ; [checkpoint 3, 3. Quel classifieur quand ?][s739] · exercices : 13.1, 13.9, 13.13–15, 13.34, 13.E1, 13.Q3–Q4, CP3.4 (+ 14)
- *knowledge distillation* → voir [distillation](#d)
- *Kolmogorov-Smirnov test* → voir [test de Kolmogorov-Smirnov](#t)
- *Kraft inequality* → voir [inégalité de Kraft](#i)
- **KV cache** (*key-value cache*) — enseignée : [ch. B3, Le KV cache][s570] · exercices : B3.2–3, B3.7–8, B3.12, B3.23, B3.E4, B3.Q11, B4.1, B4.R1 (+ 3) · réutilisée : ch. B4

## L

- **L-learning** — enseignée : [ch. 26, 26.5 · Le L-learning][s463] · exercices : 26.16, 26.18, 26.25, 26.27, 26.Q7–Q8
- *L1 penalty* → voir [Lasso (pénalité L1)](#l)
- *L2 penalty* → voir [Ridge (pénalité L2)](#r)
- *L2 starting point* → voir [L2-SP](#l)
- **L2-SP** (*L2 starting point*) — enseignée : [ch. B1, Rester près des poids pré-entraînés : la pénalité L2-SP][s507] · exercices : B1.6, B1.Q9
- *$L^1$ distance* → voir [distance de Manhattan](#d)
- *$L_\text{simple}$* → voir [loss simplifiée](#l)
- **label** (*étiquette*) — enseignée : [ch. 0A, 100.9.2 · Sélectionner : colonnes, `loc`, `iloc`, filtres][s5] · exercices : 0A.65, 1.9, 1.21, 1.Q3, 1.Q9, 4.R2, 7.17, 7.31, 7.E3, 7.Q1 (+ 8)
- *label dropout* → voir [abandon du label (guidance sans classifieur)](#a)
- **label smoothing** (*lissage des étiquettes*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] ; [ch. 9, Pénalité sur les poids, dropout, batchnorm][s25] · exercices : 24.31, B3.11
- **lambda, fonctions en argument, Callable** — enseignée : [ch. 0A, 100.5.3 · Fonctions comme valeurs : `lambda`, `key=`, `Callable`][s718] · exercices : 0A.41, 0A.43, 0A.Q7, 0B.41, 0B.47, 0B.R3, 6.21, 9.3, 9.6, 19.6 (+ 21) · réutilisée : ch. 0B, 23
- *Laplace prior* → voir [prior de Laplace](#p)
- *Laplace smoothing* → voir [lissage de Laplace](#l)
- *largest remainder method* → voir [règle du plus fort reste](#r)
- **Lasso (pénalité L1)** (*L1 penalty*) — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s278] · exercices : 9.6, 9.14, 9.19, 9.22–23, 9.31, 9.Q7–Q8, CP2.9
- **latence** (*latency*) — enseignée : [ch. B7, Latence, percentiles et débit][s516] ; [ch. B1, 201.9 · Coût d'usage : taille, latence CPU, choix d'un backbone…][s517] · exercices : B1.14, B1.26, B4.7, B7.3, B7.5, B7.19, B7.25, B7.E5, B7.Q9, PF.11 (+ 5)
- **latence et taille des modèles** — enseignée : [ch. B1, 201.9 · Coût d'usage : taille, latence CPU, choix d'un backbone…][s517]
- *latency* → voir [latence](#l)
- *latent diffusion* → voir [diffusion latente](#d)
- *latent space* → voir [espace latent](#e)
- *latent variables* → voir [code, variables latentes](#c)
- **LaTeX dans Markdown** — enseignée : [ch. 0B][s43] · exercices : 0B.32
- *law of large numbers* → voir [loi des grands nombres](#l)
- *law of total probability* → voir [formule des probabilités totales](#f)
- *layer* → voir [couche](#c), [Dockerfile, couche](#d)
- **layer norm** (*normalisation par couche*) — enseignée : [ch. 20, 20.4.4 · La batchnorm][s286] · exercices : 20.7, 20.17, 20.30, 20.E2, B3.5, B3.16, B3.19–20, B3.Q8, B3.R3 · réutilisée : ch. B3
- *layer-sequential unit-variance* → voir [LSUV](#l)
- **layout `src/`** (*src layout*) — enseignée : [ch. B7, Une structure de dépôt][s655] · exercices : B7.6, B7.9
- *lazy learner* → voir [algorithme paresseux](#a)
- *leaf* → voir [racine, feuille, nœud interne](#r)
- leaky ReLU → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- *learned positional embeddings* → voir [positions apprises](#p)
- *learning curve* → voir [courbe d'apprentissage](#c)
- **learning rate** (*taux d'apprentissage*) — enseignée : [ch. 18, 18.11 · Le learning rate][s11] ; [ch. 23, 23.10.2 · Le learning rate][s12] ; [ch. 0B, 101.1.7 · Dénombrement : principe multiplicatif, factorielle,…][s13] · exercices : 1.17, 5.19, 5.E1, CP1.10, 10.17, 14.21, 18.3, 18.23, 18.Q8, 18.R2 (+ 38) · approfondie : ch. 1, 5, 14, 19, B1 · réutilisée : ch. 21, 23, 24
- **learning rate critique** — enseignée : [ch. 19, 19.3.2 · Changer le learning rate au fil du temps][s359]
- **learning rates discriminatifs** (*discriminative learning rates, discriminative fine-tuning*) — enseignée : [ch. B1, Learning rates discriminatifs][s504] · exercices : B1.3, B1.Q6, B1.Q9
- *learning-rate schedule* → voir [planning (de learning rate)](#p)
- *least squares* → voir [moindres carrés](#m)
- **leave-one-out** (*validation croisée par exclusion d'un exemple, LOOCV*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s263] · exercices : 8.Q10, 15.R3
- **lecture de dépréciations** — enseignée : [ch. 15, 15.3 · Conventions : importer, lire la documentation, suivre les…][s745]
- *LeCun* → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- *length bias* → voir [biais de longueur (d'un modèle de récompense)](#b)
- *level curve* → voir [ligne de niveau](#l)
- **licence Creative Commons** (*Creative Commons license*) — enseignée : [ch. 29][s499]
- **licences et retraits de datasets** — enseignée : [ch. 29, 29.1 · Les datasets publics][s771]
- *Lidstone smoothing* → voir [lissage additif](#l)
- **LightGBM** — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s195] · exercices : 14.26–28
- ligne → voir [ligne de base d'une courbe precision-recall](#l), [ligne de niveau](#l), [ligne de référence (d'une attribution)](#l), [lignes de niveau](#l)
- **ligne de base d'une courbe precision-recall** (*PR baseline*) — enseignée : [ch. 3, Au-delà du livre (2) : la courbe precision-recall et l'average…][s213]
- **ligne de niveau** (*level curve, contour line*) — enseignée : [ch. 0B, 101.6.1 · Fonctions de deux variables et lignes de niveau][s126] ; [ch. 7, 7.2.1 · Classification binaire en 2D][s127] · exercices : 0B.50, 0B.Q11, 5.2
- **ligne de référence (d'une attribution)** (*baseline*) — enseignée : [ch. B6, Le choix de la ligne de référence][s641] · exercices : B6.3, B6.18–19, B6.Q7
- ligne par ligne → voir [ordre C, ligne par ligne](#o)
- **lignes de niveau** — enseignée : [ch. 0B, 101.6.1 · Fonctions de deux variables et lignes de niveau][s126] ; [ch. 5, 5.4 · Le gradient][s125] · exercices : 0B.25, 0B.49, 0B.Q11, 5.17, 5.22, 5.Q7, 7.15 · approfondie : ch. 0B
- *likelihood* → voir [log-probabilité, vraisemblance](#l), [vraisemblance (d'une hypothèse)](#v)
- *likelihood ratio* → voir [rapport de vraisemblance](#r)
- LIME → voir [substitut local, LIME](#s)
- limitation de la conservation → voir [minimisation, limitation de la conservation](#m)
- *linear* → voir [planning linéaire, planning cosinus (de bruit)](#p)
- *linear interpolation* → voir [interpolation linéaire](#i)
- **linear probe** (*sonde linéaire*) — enseignée : [ch. B1, Le linear probe][s503] · exercices : B1.7, B1.12–14, B1.16–17, B1.19–21, B1.Q5 (+ 7)
- *linear probing then fine-tuning* → voir [LP-FT](#l)
- *linear separability* → voir [séparabilité linéaire](#s)
- lissage → voir [lissage additif](#l), [lissage de Laplace](#l), [lissage des contextes](#l)
- **lissage additif** (*additive smoothing, Lidstone smoothing*) — enseignée : [ch. 13, 13.6 · Naïve Bayes][s140]
- **lissage de Laplace** (*Laplace smoothing, add-one smoothing*) — enseignée : [ch. 6, Mélanger les codes][s246] ; [ch. 13, 13.6 · Naïve Bayes][s140] · exercices : 6.13, 6.17, 6.27, 13.6, 13.29
- **lissage des contextes** (*context distribution smoothing*) — enseignée : [ch. B2][s538] · exercices : B2.6
- lisse → voir [courbe continue, lisse, univoque](#c)
- *list* → voir [liste, tuple](#l)
- **liste, tuple** (*list*) — enseignée : [ch. 0A, 100.3.1 · Listes : indices, tranches, méthodes, mutabilité][s70] · exercices : 0A.2, 0A.9, 0A.15–18, 0A.E2, 0A.Q4, 2.1, B2.Q4 (+ 16) · réutilisée : ch. 0B
- **listes, tuples, dictionnaires, ensembles, collections.Counter** — enseignée : [ch. 0A, 100.3.1 · Listes : indices, tranches, méthodes, mutabilité][s70] · exercices : 0A.2, 0A.9, 0A.16, 0A.62, 0B.38–43 (+ 66) · réutilisée : ch. 0B
- *Little's law* → voir [loi de Little](#l)
- **LLM** (*grand modèle de langage*) — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s21] ; [ch. B4, 204.1 · Panorama : modèles de base, instruct et chat ; tailles,…][s163] · exercices : 1.E4, 6.Q12, CP1.12, 19.9, 26.E5, B2.9, B2.E1, B2.Q8, B4.Q7–Q8 (+ 16) · approfondie : ch. B4 · réutilisée : ch. B8
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
- **log loss** (*logistic loss*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] · exercices : 6.22, 6.Q12, 9.R2, 15.31, 19.Q4, CP4.12
- *log-derivative trick* → voir [astuce de la log-dérivée](#a)
- **log-loss** — enseignée : [ch. 13, Au-delà du livre : la régression logistique, le classifieur…][s109] · exercices : 13.8, 13.32, 14.4 · réutilisée : ch. 18
- **log-odds** — enseignée : [ch. 13, Au-delà du livre : la régression logistique, le classifieur…][s109] · exercices : 14.4
- **log-probabilité, vraisemblance** (*log-probability, likelihood*) — enseignée : [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s42] ; [ch. 0B, 101.2.4 · Logarithmes : ln, log₂, log₁₀, changement de base][s107] · exercices : 0B.E3, 4.18, 4.24, 9.26, 13.6–7, 13.28, B8.6 · réutilisée : ch. 13
- **log-probabilités** — enseignée : [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s42] · exercices : 0B.E3, 4.18, 4.24, 9.26, 13.6, 13.28, B8.6 · réutilisée : ch. 13
- *log-probability* → voir [log-probabilité, vraisemblance](#l)
- **log-softmax** — enseignée : [ch. 17, 17.8 · La softmax][s250] · exercices : 17.15, B8.3 · réutilisée : ch. B8
- **log-sum-exp** (*astuce du log-somme-exp*) — enseignée : [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s42] ; [ch. 17, Au-delà du livre : calculer sans débordement][s46] · exercices : 4.7, 4.18, 17.15 · approfondie : ch. 17 · réutilisée : ch. B2
- *log-uniform distribution* → voir [distribution log-uniforme](#d)
- **logarithme népérien** (*natural logarithm*) — enseignée : [ch. 0B, 101.2.4 · Logarithmes : ln, log₂, log₁₀, changement de base][s107] · exercices : 0B.16, 9.R2, 13.6, 18.12, 18.14 · réutilisée : ch. 11
- **logger, handler** — enseignée : [ch. B7, Le logging plutôt que print][s660] · exercices : B7.10
- **logging** (*journalisation*) — enseignée : [ch. B7, 207.2 · Scripts et ligne de commande : argparse, point d'entrée,…][s657] · exercices : B7.9–10, B7.Q3
- *logistic function* → voir [sigmoïde](#s)
- *logistic loss* → voir [log loss](#l)
- *logistic regression* → voir [régression logistique](#r)
- **logit** — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] ; [ch. 0B][s43] · exercices : 16.3, 16.17, 16.24, 18.14, 18.E5, 20.R2, 23.Q9, 24.R3, B1.1, B3.29 (+ 76) · approfondie : ch. 18, 23 · réutilisée : ch. 20
- *logit scale* → voir [échelle des logits (CLIP)](#e)
- **logits (d'un classifieur)** — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] ; [ch. 17, 17.8 · La softmax][s250] · exercices : 17.7, 17.15–16, 18.14, 18.E5, 20.R2, 23.Q9, 24.R3, B1.1, B3.29 (+ 64) · approfondie : ch. 18, 23 · réutilisée : ch. 20
- **logsumexp** — enseignée : [ch. 13, 13.6 · Naïve Bayes][s140] · exercices : 13.28–29, 17.15 · réutilisée : ch. 17
- loi → voir [distribution de probabilité, loi](#d)
- **loi Beta** (*Beta distribution*) — enseignée : [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s222] · exercices : 4.22, 11.R3 · réutilisée : ch. 11
- **loi catégorielle, multinoulli** (*categorical distribution*) — enseignée : [ch. 2, 2.3.4 · La loi catégorielle (multinoulli)][s184] · exercices : 2.5, 2.19, 2.Q6 · réutilisée : ch. 6
- **loi de Bernoulli** (*Bernoulli distribution*) — enseignée : [ch. 2, 2.3.3 · La loi de Bernoulli][s183] · exercices : 2.Q6, 4.13, 8.4, 11.19 · réutilisée : ch. 4, 8, 11, 14
- **loi de Goodhart** (*Goodhart's law*) — enseignée : [ch. B8, Le piratage de la récompense][s472]
- **loi de Little** (*Little's law*) — enseignée : [ch. B7, Latence, percentiles et débit][s516] · exercices : B7.5
- **loi de Student** (*Student's t distribution*) — enseignée : [ch. B7, Rapporter un score : plusieurs graines et un intervalle][s191] · exercices : B7.2
- **loi des grands nombres** (*law of large numbers*) — enseignée : [ch. 0B, 101.7.5 · Simuler pour estimer : fréquences et loi des grands nombres][s136] · exercices : 0B.52, 0B.Q12
- **loi normale, gaussienne** (*normal distribution, Gaussian distribution*) — enseignée : [ch. 2, 2.3.2 · La loi normale][s182] · exercices : 2.3, 2.20, 2.Q1, 2.Q5, 5.R2, 7.15, 7.30, 8.25, 9.30, 11.10 (+ 9) · réutilisée : ch. 5, 7, 9, 12, 13, 17, 25, B5
- **loi uniforme** (*uniform distribution*) — enseignée : [ch. 2, 2.3.1 · La loi uniforme][s181] · exercices : 2.Q4, 3.12, 6.8, 16.15, 16.Q9, CP6.1 · réutilisée : ch. 3
- **lois d'échelle** (*scaling laws*) — enseignée : [ch. B3, Les lois d'échelle][s574] · exercices : B3.8, B3.10, B3.27–28, B3.E2, B3.Q12
- **lois usuelles** — enseignée : [ch. 2, 2.3 · Quelques lois usuelles][s733] · exercices : 2.5, 2.17
- *long short-term memory* → voir [LSTM](#l)
- *long-term dependency* → voir [dépendance à long terme](#d)
- *LOOCV* → voir [leave-one-out](#l)
- **LoRA** (*low-rank adaptation*) — enseignée : [ch. B4, LoRA][s173] ; [ch. 1, Panorama 2026 : où en est l'IA ?][s21] · exercices : B4.3, B4.6, B4.11, B4.25–26, B4.28, B4.E3, B4.Q10–Q11, PF.6 (+ 3)
- **loss** (*perte, fonction de coût*) — enseignée : [ch. 0B, 101.1.3 · Notations Σ et Π][s14] · exercices : 6.20, 6.E1, 6.Q12, CP1.12, 9.5, 9.R2, 11.Q4, 18.12, 18.22, 18.E5 (+ 28) · approfondie : ch. 6, 18, 27, 28, B5 · réutilisée : ch. 18, 19, B8
- **loss calculée sur les logits** (*loss on logits*) — enseignée : [ch. 18, Au-delà du livre : les losses calculées sur les logits][s354] · exercices : 18.18
- **loss de contenu** (*content loss*) — enseignée : [ch. 28, 28.4.3 · La loss de contenu][s490] · exercices : 28.1, 28.10, 28.19, 28.Q6, 28.Q8
- **loss de Huber** (*Huber loss*) — enseignée : [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s280] ; [ch. 9, 9.2 · Overfitting et underfitting][s33] · exercices : B8.2, B8.13, B8.Q2
- **loss de prédiction du bruit** — enseignée : [ch. B5, 205.3 · Le processus inverse et la loss simplifiée : prédire le…][s604]
- **loss de style** (*style loss*) — enseignée : [ch. 28, 28.4.4 · La loss de style][s491] · exercices : 28.1, 28.3–5, 28.10, 28.19, 28.29, 28.Q6, 28.Q9
- **loss minimax et loss non saturante** — enseignée : [ch. 27, 27.4.4 · Le jeu][s245]
- **loss non saturante** (*non-saturating loss*) — enseignée : [ch. 27, 27.4.3 · Entraîner le GAN][s476] · exercices : 27.3–4, 27.6, 27.12, 27.17–18, CP6.4, CP6.9
- *loss on logits* → voir [loss calculée sur les logits](#l)
- **loss perceptuelle** (*perceptual loss*) — enseignée : [ch. 28, 28.4.2 · Vue d'ensemble][s492] · exercices : 28.R1
- **loss quantile** (*pinball loss, quantile loss*) — enseignée : [ch. 9][s281]
- *loss scaling* → voir [mise à l'échelle de la loss](#m)
- **loss simplifiée** (*simplified objective, $L_\text{simple}$*) — enseignée : [ch. B5, La loss simplifiée][s611] · exercices : B5.19, B5.23
- **losses de contenu, de style et de variation totale** — enseignée : [ch. 28, 28.4 · Le transfert de style][s488] · exercices : 28.4
- **losses et leurs gradients (MSE, BCE sur logits, cross-entropy softmax)** — enseignée : [ch. 18, Au-delà du livre : les losses calculées sur les logits][s354]
- *lossless compression* → voir [compression sans perte](#c), [compression avec ou sans perte](#c)
- *lossy* → voir [compression avec ou sans perte](#c)
- *low-rank adaptation* → voir [LoRA](#l)
- **LP-FT** (*linear probing then fine-tuning*) — enseignée : [ch. B1, Pourquoi un learning rate beaucoup plus petit][s506] · exercices : B1.Q6
- **LSTM** (*long short-term memory*) — enseignée : [ch. 22, 22.7 · LSTM et GRU][s415] · exercices : 22.3–4, 22.7, 22.18–19, 22.23, 22.25, 22.E1, 22.E3, 22.Q9 (+ 28) · approfondie : ch. 24 · réutilisée : ch. 24, 28, B3
- **LSTM et GRU (ordre des portes PyTorch)** — enseignée : [ch. 22, 22.7 · LSTM et GRU][s415] · exercices : 22.11, 22.19, 22.E2, 22.Q7
- **LSTM pour séries (fenêtres, scaler, RMSE)** — enseignée : [ch. 24, 24.5.1 · Générer des séries][s765]
- **LSUV** (*layer-sequential unit-variance*) — enseignée : [ch. 16, Au-delà du livre : diagnostiquer l'évanouissement et l'explosion…][s342]

## M

- MAC → voir [multiplication-addition (MAC)](#m)
- machine à vecteurs de support → voir [SVM, machine à vecteurs de support](#s)
- *machine epsilon* → voir [epsilon machine](#e)
- **machine learning (ML)** (*apprentissage automatique*) — enseignée : [ch. 1, 1.1 · Le machine learning : extraire du sens des données][s167] · exercices : 0A.E1, 0B.E2, 1.6, 1.E1, 2.E2, 6.E4, 11.9, 20.Q1, 23.Q8, B7.Q7 (+ 33) · approfondie : ch. B7
- *macro* → voir [moyenne macro, micro, pondérée](#m)
- MAE → voir [MSE, RMSE, MAE](#m)
- **majeur illicite, mineur illicite** (*illicit major, illicit minor*) — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s298] · exercices : 11.16, 11.Q8, CP2.9
- **malédiction de la dimension, fléau de la dimension** (*curse of dimensionality*) — enseignée : [ch. 7, 7.6 · La malédiction de la dimension][s192] ; [ch. 2, 2.7 · Espaces de grande dimension][s193] · exercices : 7.8, 7.20, 7.E4, 13.15 · réutilisée : ch. 12, 13
- *Manhattan distance* → voir [distance de Manhattan](#d)
- *manifold hypothesis* → voir [hypothèse de la variété](#h)
- many-to-many → voir [one-to-many, many-to-one, many-to-many](#o)
- many-to-one → voir [one-to-many, many-to-one, many-to-many](#o)
- **MAP** (*maximum a posteriori*) — enseignée : [ch. 4, Au-delà du livre (3) : MAP et intervalle de crédibilité][s45] · exercices : 4.6, 4.21, 4.24, 4.26, CP1.1, CP1.8
- **MAPE** (*erreur absolue moyenne en pourcentage*) — enseignée : [ch. 9][s281]
- MAR → voir [MCAR, MAR, MNAR](#m)
- marche de Heaviside → voir [marche unité, marche de Heaviside, fonction signe](#m)
- **marche unité, marche de Heaviside, fonction signe** (*unit step, Heaviside step, sign function*) — enseignée : [ch. 0B, 101.1.2 · Valeur absolue, inégalités, partie entière, fonction signe][s105] ; [ch. 17, 17.4 · Les marches][s345] · exercices : 0B.2, 17.Q5, 17.Q9, 17.R3 · réutilisée : ch. 21
- **marge** (*margin*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s288] ; [ch. 13, 13.4 · Les machines à vecteurs de support][s210] · exercices : 10.7, 10.23, 13.4, 13.10, 13.16, 13.Q5
- **marge (d'un SVM)** (*margin*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s210] · exercices : 13.4, 13.10, 13.16, 13.Q5
- **marge souple, paramètre C** (*soft margin, C*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s210] · exercices : 9.Q7, 13.10, 13.16, 13.Q5
- *margin* → voir [marge](#m), [marge (d'un SVM)](#m)
- *marginal contribution* → voir [contribution marginale](#c)
- *marginal likelihood* → voir [évidence](#e)
- *marginal probability* → voir [probabilité simple, marginale](#p)
- marginale → voir [probabilité simple, marginale](#p)
- *Markov chain Monte Carlo* → voir [MCMC](#m)
- *Markov decision process* → voir [processus de décision markovien (MDP)](#p)
- *masked language model* → voir [modèle de langue masqué (MLM)](#m)
- *masker* → voir [masqueur (shap)](#m)
- masque → voir [masque booléen](#m), [masque causal](#m), [masque d'actions](#m), [masque d'attention](#m), [masque de padding](#m), [masques causal et de padding](#m), [masques d'actions](#m)
- **masque booléen** (*boolean mask*) — enseignée : [ch. 0A, 100.8.2 · Indexation, tranches, masques booléens ; vues et copies][s94] · exercices : 0B.52, 1.10, 7.28
- **masque causal** (*causal mask*) — enseignée : [ch. B3, Le masque causal][s558] · exercices : B3.2, B3.12, B3.16–18, B3.24, B3.29, B3.Q4–Q5, B4.R1 (+ 2) · réutilisée : ch. B4
- **masque d'actions** (*action mask*) — enseignée : [ch. 26][s62] · exercices : 26.35
- **masque d'attention** (*attention mask*) — enseignée : [ch. B2, Utiliser un tokenizer avec `transformers`][s532] · exercices : B2.28
- **masque de padding** (*padding mask*) — enseignée : [ch. B3, Le masque de padding][s559] · exercices : B3.12, B3.14
- **masques causal et de padding** — enseignée : [ch. B3, 203.3 · Masques : causal, padding ; auto-attention et attention…][s557] · exercices : B3.2, B3.11–12, B3.17–18, B3.29, B3.Q4–Q5
- **masques d'actions** — enseignée : [ch. 26][s62] · exercices : 26.20–21
- **masqueur (shap)** (*masker*) — enseignée : [ch. B6, Un exemple à deux features][s636]
- **matplotlib : plot, scatter, hist, subplots, imshow, légendes ; wb.plot.show_images** — enseignée : [ch. 0A, 100.10.1 · `plot`, `scatter`, `hist` ; titres, axes et légendes][s723] · exercices : 0A.36, 0A.60, 0A.67, 0A.Q12 · réutilisée : ch. 0B, 1
- matrice → voir [matrice de co-occurrences](#m), [matrice de confusion](#m), [matrice de corrélation](#m), [matrice de covariance, de corrélation](#m), [matrice de Gram](#m), [matrice hessienne](#m), [matrice identité](#m), [matrices : forme, transposée, produit matrice-vecteur, produit matriciel, vérification des formes](#m)
- **matrice de co-occurrences** (*co-occurrence matrix*) — enseignée : [ch. B2, La matrice de co-occurrences][s536] · exercices : B2.6, B2.22
- **matrice de confusion** (*confusion matrix*) — enseignée : [ch. 3, 3.7.2 · La matrice de confusion][s63] · exercices : 3.2, 3.15, 3.17, 3.Q7, CP1.6, 10.R3, CP2.13, 21.23, 23.20, B6.R3 (+ 12) · réutilisée : ch. 4, 7, 8, 10, 13, 20, 21, 23, B6
- **matrice de corrélation** — enseignée : [ch. 2, Matrices de covariance et de corrélation][s198] · exercices : 2.28, 7.31
- **matrice de covariance, de corrélation** (*covariance matrix, correlation matrix*) — enseignée : [ch. 2, Matrices de covariance et de corrélation][s198] ; [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s199] · exercices : 2.28, 7.31, 12.7–8, 12.22, 14.13, 25.8 · réutilisée : ch. 25, 27
- **matrice de Gram** (*Gram matrix*) — enseignée : [ch. 28, 28.4.1 · Le style dans une matrice][s489] · exercices : 28.1–3, 28.8, 28.18, 28.25, 28.E2, 28.Q7, 28.R2, CP6.5 (+ 1)
- **matrice hessienne** (*Hessian matrix*) — enseignée : [ch. 5, Les points où le gradient s'annule][s230] · exercices : 5.24
- **matrice identité** (*identity matrix*) — enseignée : [ch. 0B, 101.4.4 · Identité et idée de l'inverse][s119] · exercices : 0B.21, 0B.42, 0B.46, 0B.Q9
- **matrices : forme, transposée, produit matrice-vecteur, produit matriciel, vérification des formes** — enseignée : [ch. 0B, 101.4.2 · Produit matrice-vecteur][s728] · exercices : 0B.9, 0B.20, 0B.30, 0B.42–45, 0B.54, 16.R3, 17.8 (+ 19) · réutilisée : ch. 9, 12, 16, 18, 28, B3
- *matrix multiplication* → voir [produit matriciel](#p)
- *Matthews correlation coefficient* → voir [MCC](#m)
- **max et average pooling** — enseignée : [ch. 21, 21.4.2 · Le stride pour construire une hiérarchie][s397]
- **max-norm (contrainte)** (*max-norm constraint*) — enseignée : [ch. 24, 24.2.6 · Ajouter du dropout][s435] · exercices : 24.6, 24.13–14
- *max-norm constraint* → voir [max-norm (contrainte)](#m)
- maximin → voir [minimax, maximin](#m)
- **maximisation d'activation** (*activation maximization*) — enseignée : [ch. 21, 21.9.2 · Regarder les filtres (1) : ce qui les excite le plus][s398] ; [ch. 28][s399] · exercices : B6.R1
- *maximum mean discrepancy* → voir [MMD](#m)
- **maxout** — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s235] · exercices : 17.5, 17.Q7
- **MCAR, MAR, MNAR** (*missing completely at random, missing at random, missing not at random*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s304]
- **MCC** (*Matthews correlation coefficient*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s205] · exercices : 3.5, 3.19
- *McCulloch-Pitts neuron* → voir [neurone formel](#n)
- **MCMC** (*Markov chain Monte Carlo*) — enseignée : [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s222] · exercices : 4.11
- *McNemar's test* → voir [test de McNemar](#t)
- MDI → voir [importance des features (MDI)](#i)
- MDP → voir [processus de décision markovien (MDP)](#p)
- *mean* → voir [moyenne, médiane, mode](#m)
- *mean decrease in impurity* → voir [importance des features (MDI)](#i)
- **mean pooling** (*moyenne des tokens*) — enseignée : [ch. B2, D'une suite de vecteurs à un vecteur de phrase][s544] · exercices : B2.28, B2.Q11
- *mean reciprocal rank* → voir [MRR](#m)
- *measurement* → voir [biais de labels, de mesure, de déploiement](#b)
- *mechanistic interpretability* → voir [interprétabilité mécaniste](#i)
- *median* → voir [moyenne, médiane, mode](#m)
- médiane → voir [moyenne, médiane, mode](#m)
- **médiatrice** (*perpendicular bisector*) — enseignée : [ch. 7, Avec des labels : le centroïde le plus proche][s256] · exercices : 7.7
- **meilleur checkpoint** (*best checkpoint*) — enseignée : [ch. 23, 23.10.3 · Early stopping][s274] · exercices : 23.25, B1.19, PF.6 · réutilisée : ch. B1
- **mélange d'experts (MoE)** (*mixture of experts*) — enseignée : [ch. B3, Compter les paramètres d'un bloc][s573] · exercices : 10.9
- **mélange paramétrique** (*parametric blending*) — enseignée : [ch. 25, 25.2.3 · Mélanger deux représentations][s452] · exercices : 25.2, 25.21
- mémoire → voir [mémoire de cellule](#m), [mémoire de rejeu](#m), [mémoire paramétrique, non paramétrique](#m)
- **mémoire de cellule** (*cell state*) — enseignée : [ch. 22, 22.7 · LSTM et GRU][s415] · exercices : 22.Q8–Q9
- **mémoire de rejeu** (*replay buffer, experience replay*) — enseignée : [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s280] ; [ch. 26, 26.9 · Le rejeu d'expérience][s469] · exercices : 26.12, 26.31, 26.33, 26.E4, 26.Q12, B8.13–14, B8.Q2–Q3, B8.R1 (+ 1)
- **mémoire paramétrique, non paramétrique** (*parametric, non-parametric memory*) — enseignée : [ch. B4, L'idée][s596] · exercices : 13.Q2 · approfondie : ch. 13
- **mémorisation (générateur d'images)** (*memorization*) — enseignée : [ch. B5, La mémorisation][s578] · exercices : B5.10, B5.28, B5.Q10
- **mémorisation (modèle de langue)** (*memorization*) — enseignée : [ch. B5, La mémorisation][s578] · évoquée : [ch. B3][s562] · exercices : B3.10, B5.10, B5.28, B5.Q10
- *memorization* → voir [mémorisation (modèle de langue)](#m), [mémorisation (générateur d'images)](#m)
- *merge* → voir [jointure](#j), [fusion (BPE), rang](#f)
- **merge et pivot_table** — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s313]
- *merging* → voir [fusion d'adaptateur](#f)
- **mesure de latence** — enseignée : [ch. B7, Latence, percentiles et débit][s516] · exercices : PF.11
- **méta-estimateur** (*meta-estimator*) — enseignée : [ch. 7, 7.4.2 · Un-contre-un (OvO)][s254] ; [ch. 15, 15.4 · Les estimateurs][s255] · exercices : 7.22–23, 15.8
- *meta-estimator* → voir [méta-estimateur](#m)
- *meta-learner* → voir [stacking, méta-modèle](#s)
- méta-modèle → voir [stacking, méta-modèle](#s)
- **métadonnées de provenance, C2PA** (*content credentials*) — enseignée : [ch. B5, Transparence et provenance des contenus générés][s457] · exercices : 27.9, B5.10
- *method* → voir [classe, objet, méthode, attribut](#c)
- méthode → voir [classe, objet, méthode, attribut](#c), [méthode à frontière](#m), [méthode agnostique, méthode spécifique](#m), [méthode d'Euler](#m), [méthode de Monte-Carlo](#m), [méthode des trapèzes](#m), [méthode du coude](#m), [méthode hypothético-déductive](#m), [méthodes spéciales : __repr__, __len__, __getitem__, __call__, surcharge d'opérateurs (__add__, __radd__, __mul__, __eq__)](#m)
- **méthode à frontière** (*boundary method*) — enseignée : [ch. 7, 7.2.1 · Classification binaire en 2D][s127]
- **méthode agnostique, méthode spécifique** (*model-agnostic, model-specific method*) — enseignée : [ch. B6][s382]
- **méthode d'Euler** (*Euler method*) — enseignée : [ch. B5, Flow matching et rectified flow][s627] · exercices : B5.20
- **méthode de Monte-Carlo** (*Monte Carlo method*) — enseignée : [ch. 3][s215]
- **méthode des trapèzes** (*trapezoidal rule*) — enseignée : [ch. 3, Au-delà du livre (1) : la courbe ROC et l'AUC][s212] · exercices : 3.24
- **méthode du coude** (*elbow method*) — enseignée : [ch. 7, Combien de clusters ?][s259]
- **méthode hypothético-déductive** (*hypothetico-deductive method*) — enseignée : [ch. 11, 11.3 · Déduction et induction][s295]
- méthode spécifique → voir [méthode agnostique, méthode spécifique](#m)
- **méthodes spéciales : __repr__, __len__, __getitem__, __call__, surcharge d'opérateurs (__add__, __radd__, __mul__, __eq__)** — enseignée : [ch. 0A, 100.7.2 · Méthodes spéciales : `__repr__`, `__len__`,…][s720] · exercices : 18.26 · réutilisée : ch. 18
- MHA → voir [attention multi-têtes (MHA)](#a)
- micro → voir [moyenne macro, micro, pondérée](#m)
- **micro-autograd** — enseignée : [ch. 18, Au-delà du livre : un micro-autograd][s356] · exercices : 18.26–28, 21.R2
- *middle term* → voir [moyen terme](#m)
- *min-max scaling* → voir [normalisation min-max](#n)
- **min-SNR** (*min-SNR weighting*) — enseignée : [ch. B5, Le lien avec l'ELBO][s625]
- *min-SNR weighting* → voir [min-SNR](#m)
- min_delta → voir [patience, `min_delta`](#p)
- mineur illicite → voir [majeur illicite, mineur illicite](#m)
- **mini-batch** (*mini-lot*) — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s7] ; [ch. 19, 19.4.3 · La descente par mini-batches][s8] · exercices : 0A.5, 0A.66, 16.3, 18.Q11, 19.25, 19.E5, 19.Q8, 20.4, 20.19, 27.Q7 (+ 9) · approfondie : ch. 18 · réutilisée : ch. 18
- *mini-batch gradient descent* → voir [descente par mini-batches](#d)
- **mini-batches** — enseignée : [ch. 18, 18.12.5 · Les mini-batches][s749] · évoquée : [ch. 27, 27.5 · Un GAN en action][s478] · exercices : 0A.5, 0A.24, 0A.66, 2.21, 2.Q9, 18.11, 18.29, 18.Q11, 20.19, 27.Q7 (+ 64) · approfondie : ch. 0A
- **mini-batches : iterate_minibatches, drop_last** — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s7]
- **mini-GPT PyTorch from scratch** — enseignée : [ch. B3, 203.8 · Mini-GPT from scratch : données, entraînement, loss,…][s248] · exercices : B3.10, B3.19–22, B3.26, B3.29–30, B3.Q10, B3.R1
- *minibatch discrimination* → voir [discrimination par mini-batch](#d)
- *minibatch standard deviation* → voir [écart-type par mini-batch](#e)
- *minimal cost-complexity pruning* → voir [élagage coût-complexité](#e)
- **minimax, maximin** — enseignée : [ch. 27, 27.4.4 · Le jeu][s245] · exercices : 26.35, 26.37, 27.2–3, 27.6, 27.18, 27.Q5
- **minimisation, limitation de la conservation** (*data minimisation, storage limitation*) — enseignée : [ch. B7, Journaliser les prédictions, et le RGPD][s680] · exercices : B3.10, B4.9, B7.7
- **minimum local** (*local minimum*) — enseignée : [ch. 7, Sans labels : k-means][s170] · exercices : 5.4, 5.14, 5.17, 5.E2, 5.Q4, 18.Q8 · réutilisée : ch. 11
- *minimum-norm solution* → voir [solution de norme minimale, pseudo-inverse](#s)
- **mise à jour bayésienne, séquentielle** (*Bayesian updating, sequential updating*) — enseignée : [ch. 4, 4.6 · Répéter la règle de Bayes][s220] · exercices : 9.7
- **mise à jour douce (de Polyak)** (*soft update, Polyak*) — enseignée : [ch. B8, Les journaux de PPO][s684] · exercices : B8.Q3
- **mise à jour séquentielle** — enseignée : [ch. 4, 4.6 · Répéter la règle de Bayes][s220] · exercices : 4.16
- **mise à l'échelle de la loss** (*loss scaling*) — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s444]
- **mise à l'échelle robuste** (*robust scaling*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s304]
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
- **ML Test Score** — enseignée : [ch. B7, Une grille : le ML Test Score][s668] · exercices : B7.25, B7.Q7
- **MLflow** — enseignée : [ch. B7, 207.5 · Suivi d'expériences : runs, métriques, artefacts, comparaison][s437] · exercices : 11.25
- MLM → voir [modèle de langue masqué (MLM)](#m)
- MLP → voir [réseau feed-forward, MLP](#r)
- **MM-DiT** (*multimodal diffusion transformer*) — enseignée : [ch. B5, Les transformers de diffusion][s626]
- **MMD** (*maximum mean discrepancy*) — enseignée : [ch. 28, 28.4.6 · Discussion][s494]
- MNAR → voir [MCAR, MAR, MNAR](#m)
- mode → voir [moyenne, médiane, mode](#m)
- *mode collapse* → voir [effondrement de mode](#e)
- **mode entraînement, mode évaluation** (*training mode, evaluation mode*) — enseignée : [ch. 20, 20.4.3 · Le dropout][s24] · exercices : 20.3, 20.23, 20.Q5–Q6, CP4.4, CP4.13, 21.22–23, 21.27–28 (+ 20)
- mode évaluation → voir [mode entraînement, mode évaluation](#m)
- **mode inférence** (*inference mode*) — enseignée : [ch. 23, 23.5.1 · Remodeler][s423]
- *model* → voir [modèle](#m)
- *model artifact* → voir [artefact de modèle](#a)
- *model card* → voir [fiche modèle](#f)
- *model persistence* → voir [persistance (d'un modèle)](#p)
- *model-agnostic* → voir [méthode agnostique, méthode spécifique](#m)
- *model-specific method* → voir [méthode agnostique, méthode spécifique](#m)
- **modèle** (*model*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s47] · exercices : 0B.E1, 3.E5, 6.E3, 7.24, 7.Q6, 8.6, 8.8, 8.12, 8.26, 8.Q8 (+ 91) · approfondie : ch. 10, 20, 23, B1, B6, B7, B8 · réutilisée : ch. 23, 27, 28, B1, B2, B6, B7, B8
- **modèle à espace d'états (SSM)** (*state space model*) — enseignée : [ch. 22, Au-delà du livre : après les RNN, Transformers et modèles à espace…][s420]
- **modèle d'IA à usage général** (*general-purpose AI model*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s580] · exercices : B4.10
- **modèle de base** (*base model*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s580] · exercices : 14.Q5, 14.Q10, 15.Q9, B1.9, B1.23, B1.27, B4.10, B4.26, B4.Q1, B4.Q7 (+ 1)
- **modèle de Bradley-Terry** (*Bradley-Terry model*) — enseignée : [ch. B8, Le modèle de récompense de Bradley-Terry][s697]
- **modèle de chat** (*chat model*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s580] · exercices : B4.Q1, B4.Q6
- **modèle de cohérence** (*consistency model*) — enseignée : [ch. B5, La distillation][s621]
- *modèle de conversation* → voir [chat template](#c)
- **modèle de diffusion** (*diffusion model*) — enseignée : [ch. B5, Modèles de diffusion : un DDPM minimal][s166] · évoquée : [ch. 1, Panorama 2026 : où en est l'IA ?][s21] · exercices : 25.10, 27.E3, CP6.12, B5.9, B5.13, B5.E1, B5.E3, B5.Q1–Q2, B5.R3 (+ 43)
- **modèle de fondation** (*foundation model*) — enseignée : [ch. B1, 201.8 · Modèles de fondation : auto-supervisé (DINOv2),…][s165] · évoquée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s47] · exercices : 1.Q11, B1.E4
- **modèle de fondation tabulaire** (*tabular foundation model*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s195]
- **modèle de langue causal (auto-régressif)** (*causal language model*) — enseignée : [ch. B3, Le décodeur : GPT et le modèle de langue causal][s32] · exercices : B3.29
- **modèle de langue masqué (MLM)** (*masked language model*) — enseignée : [ch. B3, L'encodeur : BERT et le modèle de langue masqué][s568]
- **modèle de raisonnement** (*reasoning model*) — enseignée : [ch. B4, Le raisonnement pas à pas][s587]
- **modèle de récompense** (*reward model*) — enseignée : [ch. B8, 208.7 · RL et modèles de langue : RLHF, modèle de récompense, DPO,…][s690] · exercices : 26.11, B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- **modèle de récompense Bradley-Terry** — enseignée : [ch. B8, Le modèle de récompense de Bradley-Terry][s697] · exercices : B8.6, B8.10, B8.25–26, B8.E4, B8.Q11–Q12
- *modèle de référence* → voir [baseline](#b)
- **modèle dérivé** (*derived model*) — enseignée : [ch. B1, Publier un modèle dérivé][s519] · exercices : B1.9, B1.27, B4.Q2
- **modèle instruct** (*instruct model*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s580] · exercices : B4.14, B4.Q1
- **modèle pré-entraîné** (*pretrained model*) — enseignée : [ch. 23, 23.9.4 · Utiliser un modèle pré-entraîné][s429] ; [ch. B1, 201.2 · Trouver et charger un modèle pré-entraîné : torchvision,…][s430] · exercices : 23.10, 23.24, 29.Q1, B1.9, B1.E1, B1.E5, B1.Q1, B1.Q3, B1.Q8–Q9 (+ 2) · approfondie : ch. 20 · réutilisée : ch. B2
- **modèle pré-entraîné torchvision** — enseignée : [ch. 20, 20.6 · Quelques exemples][s754] ; [ch. 23, 23.9.4 · Utiliser un modèle pré-entraîné][s429] · exercices : 23.24 · réutilisée : ch. B1
- **modèle unigramme, bigramme, trigramme** (*unigram, bigram, trigram model*) — enseignée : [ch. 6, Au-delà du livre (2) : le contexte réduit la surprise][s162] · exercices : 6.26
- **modèles de fondation (DINOv2, CLIP)** — enseignée : [ch. B1, 201.8 · Modèles de fondation : auto-supervisé (DINOv2),…][s165] · exercices : B1.25
- *module* → voir [`nn.Module`](#n)
- **module, paquet** (*package*) — enseignée : [ch. 0A, 100.6 · Modules, erreurs et fichiers][s83] · exercices : 0A.24, 0A.26, 0A.45, 0A.Q8, 4.14, 6.12, 6.19, 20.18, B3.18, B4.27 (+ 56) · approfondie : ch. 20 · réutilisée : ch. 11, 21, 22, 23, 25, B2, B4, B7
- **modules et imports** — enseignée : [ch. 0A, 100.6.1 · Modules, imports et bibliothèque standard][s654] · exercices : 0A.24, 0A.Q8
- **\*modus ponens\*, \*modus tollens\*** — enseignée : [ch. 11, 11.4 · La déduction][s297] · exercices : 11.5
- modus tollens → voir [\*modus ponens\*, \*modus tollens\*](#m)
- MoE → voir [mélange d'experts (MoE)](#m)
- **moindres carrés** (*least squares, ordinary least squares, OLS*) — enseignée : [ch. 9, Les moindres carrés (au-delà du livre)][s232] · exercices : 0B.24, 8.16, 9.2, 9.16–17, 9.22, 9.31, 10.Q1, 11.12, 11.18 (+ 2)
- **momentum** (*élan*) — enseignée : [ch. 19, 19.5.1 · Le momentum][s362] · exercices : 19.2, 19.10, 19.13, 19.16, 19.20–21, 19.24, 19.E1, 19.Q9, CP4.3 (+ 18) · réutilisée : ch. 23, 26, B5
- **momentum de Nesterov** (*Nesterov momentum, Nesterov accelerated gradient*) — enseignée : [ch. 19, 19.5.2 · Le momentum de Nesterov][s363] · exercices : 19.3, 19.13, 19.16, 19.21, 19.Q10
- **monôme** (*monomial*) — enseignée : [ch. 15, 15.9.5 · Les pipelines][s15] · exercices : 15.4
- *monomial* → voir [monôme](#m)
- *Monte Carlo error* → voir [bruit de Monte-Carlo](#b)
- *Monte Carlo method* → voir [méthode de Monte-Carlo](#m)
- **montée de gradient** (*gradient ascent*) — enseignée : [ch. 5, Descendre une surface pas à pas][s231] · exercices : 5.3, 21.28, 21.Q10, 28.12–13, 28.Q2, B6.R1, B8.3 · approfondie : ch. 28 · réutilisée : ch. 21, 28
- **montée de gradient sur l'entrée** (*gradient ascent on the input*) — enseignée : [ch. 28, 28.2.2 · Visualiser un filtre][s485] · exercices : 21.28, 28.Q2
- *Moore-Penrose pseudo-inverse* → voir [solution de norme minimale, pseudo-inverse](#s)
- *morceau* → voir [chunk](#c)
- *moving average* → voir [moyenne mobile](#m)
- **moyen terme** (*middle term*) — enseignée : [ch. 11, 11.4 · La déduction][s297] · exercices : 11.3, 11.15, 11.Q8, CP2.9
- **moyen terme non distribué** (*undistributed middle*) — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s298] · exercices : 11.15, 11.Q8, CP2.9
- moyenne → voir [moyenne harmonique](#m), [moyenne incrémentale](#m), [moyenne interquartile (IQM)](#m), [moyenne macro, micro, pondérée](#m), [moyenne mobile](#m), [moyenne mobile exponentielle](#m), [moyenne pondérée](#m), [moyennes macro/micro/pondérée](#m)
- *moyenne des tokens* → voir [mean pooling](#m)
- **moyenne harmonique** (*harmonic mean*) — enseignée : [ch. 3, 3.7.11 · Le score F1][s208] · exercices : 3.8
- **moyenne incrémentale** — enseignée : [ch. 11][s737] · exercices : 11.2, 11.20, CP2.11, B8.R3
- **moyenne interquartile (IQM)** (*interquartile mean*) — enseignée : [ch. B8, Rapporter plusieurs graines][s653]
- **moyenne macro, micro, pondérée** (*macro, weighted average*) — enseignée : [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s41] · exercices : 3.25
- **moyenne mobile** (*moving average*) — enseignée : [ch. 0B, 101.1.4 · Moyenne, moyenne pondérée, somme pondérée, moyenne mobile][s49] · exercices : 0B.4, 0B.35, 1.12, 5.14, 19.8, 19.17, 19.Q12, 20.4, B5.23 · réutilisée : ch. 19, 20, 26, B5
- **moyenne mobile exponentielle** (*exponential moving average, EMA*) — enseignée : [ch. 19, 19.5.4 · Adadelta et RMSprop][s366] · exercices : 19.8, 19.17, 20.4, B5.23 · réutilisée : ch. 20, 26, B5
- **moyenne pondérée** (*weighted average*) — enseignée : [ch. 0B, 101.1.4 · Moyenne, moyenne pondérée, somme pondérée, moyenne mobile][s49] · exercices : 0B.4, 0B.35, 0B.Q3, 3.25, 13.20, B6.18
- **moyenne, médiane, mode** (*mean, median*) — enseignée : [ch. 0B, 101.1.4 · Moyenne, moyenne pondérée, somme pondérée, moyenne mobile][s49] ; [ch. 2, Résumer une liste de nombres : moyenne, médiane, mode][s175] · exercices : 0B.4, 0B.24, 0B.35, 0B.E5, 0B.Q3, 2.1, 2.E1, 2.Q1, 2.R2, 3.6 (+ 50) · approfondie : ch. 3, 11 · réutilisée : ch. 1, 3, 5, 7, 8, 9, 12, 19, 20, 25, 26, 27, B5
- **moyennes macro/micro/pondérée** — enseignée : [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s41] · exercices : 3.5–6, 3.19, 3.22, 3.25, 3.Q11
- **MQA, GQA** (*multi-query attention, grouped-query attention*) — enseignée : [ch. B3, Moins de têtes pour les clés : MQA et GQA][s561] · exercices : B3.3, B3.8, B3.Q6, B4.1, B4.3
- **MRR** (*mean reciprocal rank, rang réciproque moyen*) — enseignée : [ch. B4, Mesurer une recherche : recall@k et MRR][s595] · exercices : B4.2, B4.18, B4.20
- **MSE, RMSE, MAE** (*erreur quadratique moyenne, sa racine, erreur absolue moyenne*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s33] · exercices : 9.1, 9.10, 9.12–14, 9.E2, 14.4, 14.28, 24.25, 24.Q10 (+ 40) · approfondie : ch. 18, 24 · réutilisée : ch. 14, 24
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
- **multiplication-addition (MAC)** (*multiply-accumulate*) — enseignée : [ch. 24, 24.4.3 · Les couches de convolution][s441] · exercices : 0A.Q8, 17.8, 21.8, 23.Q7, 24.3, 24.7, CP5.2, 28.6
- *multiply-accumulate* → voir [multiplication-addition (MAC)](#m)
- *multivariable chain rule* → voir [règle de la chaîne à plusieurs chemins](#r)
- *multivariate transformation* → voir [transformation univariée, multivariée](#t)
- multivariée → voir [transformation univariée, multivariée](#t)
- **mutable / immuable** (*mutable / immutable*) — enseignée : [ch. 0A, 100.2.2 · Chaînes de caractères et f-strings][s68]
- *mutable / immutable* → voir [mutable / immuable](#m)
- *mutation testing* → voir [test par mutation](#t)
- *mutual information* → voir [information mutuelle](#i)
- **mylearn.linalg_basics** — enseignée : [ch. 0B, 101.3.1 · Composantes, somme, produit par un scalaire][s113]

## N

- *« n choose k »* → voir [coefficient binomial](#c)
- *naive Bayes* → voir [Naïve Bayes (gaussien, multinomial)](#n)
- **Naïve Bayes (gaussien, multinomial)** (*naive Bayes, bayésien naïf, Gaussian, multinomial naive Bayes*) — enseignée : [ch. 13, 13.6 · Naïve Bayes][s140] ; [ch. 4, Bayes dans le ML d'aujourd'hui][s319] · évoquée : [mini-projet 1][s320] · exercices : 4.R2, 13.6–7, 13.27–30, 13.35, 13.E4, 13.Q11 (+ 6) · réutilisée : ch. B4
- *naive forecast* → voir [persistance (en prévision)](#p)
- *Nash equilibrium* → voir [équilibre de Nash](#e)
- **nat** — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] · exercices : 0B.15–16, 6.1, 6.3, 6.11–12, 6.20, 6.22, 6.Q12, 9.R2 (+ 23) · approfondie : ch. 0B · réutilisée : ch. 18
- *natural frequencies* → voir [fréquences naturelles, arbre des fréquences naturelles](#f)
- *natural frequency tree* → voir [fréquences naturelles, arbre des fréquences naturelles](#f)
- *natural logarithm* → voir [logarithme népérien](#l)
- **NCHW** — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s234] · exercices : 20.13, 23.2, 23.12, 23.Q6
- *near-duplicate* → voir [quasi-doublon](#q)
- *nearest centroid classifier* → voir [classifieur du centroïde le plus proche](#c)
- *nearest neighbour* → voir [plus proche voisin (1-NN)](#p)
- **négation de l'antécédent** (*denying the antecedent*) — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s298] · exercices : 11.Q8
- négative → voir [classe positive, négative](#c)
- *negative class* → voir [classe positive, négative](#c)
- *negative predictive value* → voir [valeur prédictive négative (VPN)](#v)
- *negative sampling* → voir [échantillonnage négatif](#e)
- *negative transfer* → voir [transfert négatif](#t)
- *nested cross-validation* → voir [validation croisée imbriquée](#v)
- **Nesterov** — enseignée : [ch. 19, 19.5.2 · Le momentum de Nesterov][s363] · exercices : 19.3, 19.13, 19.16, 19.21, 19.24, 19.Q10, 19.Q12
- *Nesterov accelerated gradient* → voir [momentum de Nesterov](#m)
- *Nesterov momentum* → voir [momentum de Nesterov](#m)
- *neural network* → voir [réseau de neurones](#r)
- *neural style transfer* → voir [transfert de style neuronal](#t)
- *neuromorphic chip* → voir [puce neuromorphique](#p)
- *neuron* → voir [neurone](#n), [neurone biologique](#n)
- **neurone** (*neuron, unit*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s51] · évoquée : [ch. 1, 1.7 · Deep learning][s35] · exercices : 1.23, 10.2, 10.8, 10.14, 10.16, 10.E2, 10.Q1–Q2, 10.Q4, 10.Q8 (+ 47) · approfondie : ch. 17, 18 · réutilisée : ch. 16, 26
- **neurone artificiel** — enseignée : [ch. 10, 10.3 · Les neurones artificiels][s287] · exercices : 1.Q10, 10.8, 10.Q1, 10.Q4 · réutilisée : ch. 16
- **neurone biologique** (*neuron*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s51] · exercices : 10.8, 10.Q2
- **neurone formel** (*McCulloch-Pitts neuron, threshold logic unit*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s253]
- **neurone mort** (*dead neuron*) — enseignée : [ch. 16, Au-delà du livre : diagnostiquer l'évanouissement et l'explosion…][s342] ; [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s235]
- **neurotransmetteur** (*neurotransmitter*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s51] · exercices : 10.Q2, 10.Q4
- *neurotransmitter* → voir [neurotransmetteur](#n)
- **NFC, NFD, NFKC** (*canonical composition, canonical decomposition, compatibility normalization*) — enseignée : [ch. B2, Un caractère affiché, plusieurs points de code][s524] · exercices : B2.1, B2.Q2–Q3
- NFD → voir [NFC, NFD, NFKC](#n)
- NFKC → voir [NFC, NFD, NFKC](#n)
- **niveau de log** (*log level*) — enseignée : [ch. B7, Variables d'environnement et secrets][s659]
- **nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d** — enseignée : [ch. 24, 24.6.2 · Construire un modèle fonctionnel][s763] · exercices : 20.Q7, CP4.1, 21.2, 21.11, 21.20, 21.25, 24.6, 24.13, 24.Q7, 24.R2 (+ 11) · approfondie : ch. 21 · réutilisée : ch. 22
- **nn.Embedding** — enseignée : [ch. B2, 202.7 · Embeddings : de one-hot à `nn.Embedding`, entraînement et…][s22] · exercices : CP5.8, CP5.11, 27.25, B2.5, B2.25, B2.Q9, B3.19, B5.21 · réutilisée : ch. 24, B3
- **`nn.Module`** (*module*) — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s3] · exercices : 20.15, 20.18, 20.20, 20.30–31, 21.32, 24.Q5, 25.30, B2.25, B3.18 (+ 8) · réutilisée : ch. 21, 22, 23, 25
- **nn.RNN, nn.LSTM, nn.GRU** — enseignée : [ch. 22, 22.5 · Organiser les entrées][s407] · exercices : 20.20, 22.3, 22.15, 22.19
- **nn.Sequential, nn.Flatten, nn.Linear** — enseignée : [ch. 23, 23.6.2 · Créer le modèle][s757] · exercices : 16.17, 20.21, 20.Q4, 21.22, 21.28, 23.16, 23.Q3, 24.16, 24.20, 24.28 (+ 11)
- **No Free Lunch** — enseignée : [ch. 11, 11.2.3 · L'optimisation][s293] · exercices : 11.12, 11.26, 11.E3, 11.Q5, 19.26, 19.Q12 · réutilisée : ch. 19
- *no free lunch theorem* → voir [théorème No Free Lunch](#t)
- nœud interne → voir [racine, feuille, nœud interne](#r)
- *noise* → voir [point cœur, point de bord, bruit](#p)
- *noise prediction* → voir [prédiction du bruit](#p)
- *noise reduction* → voir [débruitage](#d)
- *noise schedule* → voir [planning de bruit](#p)
- *noisy ReLU* → voir [ReLU bruitée](#r)
- **nombre dénormalisé** (*subnormal number, denormal number*) — enseignée : [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s42]
- *nominal data* → voir [donnée nominale](#d)
- non ciblée → voir [attaque ciblée, non ciblée](#a)
- non paramétrique → voir [classifieur paramétrique, non paramétrique](#c), [mémoire paramétrique, non paramétrique](#m)
- *non-parametric classifier* → voir [classifieur paramétrique, non paramétrique](#c)
- *non-parametric memory* → voir [mémoire paramétrique, non paramétrique](#m)
- *non-saturating loss* → voir [loss non saturante](#l)
- *norm* → voir [norme](#n)
- *normal distribution* → voir [loi normale, gaussienne](#l)
- *normal equations* → voir [équations normales](#e)
- normalisation → voir [normalisation de texte](#n), [normalisation et découpage stratifié](#n), [normalisation min-max](#n)
- **normalisation de texte** — enseignée : [ch. B2, Le même texte, un autre prix selon la langue][s774]
- **normalisation et découpage stratifié** — enseignée : [ch. 23][s432]
- **normalisation min-max** (*min-max scaling*) — enseignée : [ch. 12, 12.5.1 · La normalisation min-max][s305] · exercices : 0A.30, 0A.42, 12.2, 12.6, 12.17, 12.E2, 12.Q5, 15.3
- **normaliser (une distribution)** (*normalize*) — enseignée : [ch. 12, 12.5 · Normaliser et standardiser][s177] ; [ch. 2][s178] · exercices : 4.7, 12.Q5, 14.24, 20.22, 23.R3, 28.2, 28.16, B1.2, B1.5, B1.Q4 (+ 2) · réutilisée : ch. 4, 13
- *normalize* → voir [normaliser (une distribution)](#n)
- **norme** (*norm*) — enseignée : [ch. 0B, 101.3.2 · Norme et distance][s114] · exercices : 0B.8, 0B.18, 0B.25, 0B.39–41, 0B.Q8, 1.4, 1.11, CP1.2 (+ 55) · réutilisée : ch. 1, 2, 5, 7, 9, 10, 12, 13, 19, 25, 28, B2
- **norme de Frobenius** (*Frobenius norm*) — enseignée : [ch. 28, 28.4.4 · La loss de style][s491]
- **notations Σ et Π** — enseignée : [ch. 0B, 101.1.3 · Notations Σ et Π][s14] · exercices : 0B.3, 0B.24, 0B.32, 0B.35, 0B.Q3, 0B.R1
- **notebook** (*carnet*) — enseignée : [ch. 0A, 100.1 · Prise en main : notebooks, terminal et workbook][s36] · exercices : 0A.E4, 8.24, CP4.9, CP5.5–6, B7.6, B7.9, B7.16, B7.E1, B7.Q1 (+ 5) · réutilisée : ch. 0B
- **noyau** (*kernel*) — enseignée : [ch. 0A, 100.1.1 · Cellules, noyau, ordre d'exécution et « Tout exécuter »][s64] · exercices : 0A.13, 0A.26, 0A.63–66, 0A.Q1
- **noyau (d'un SVM)** (*kernel*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s210] · exercices : 13.5, 13.11–12, 13.17–18, 13.Q2, 13.Q5–Q6, 13.Q12
- **noyau (d'une convolution)** (*kernel, filter*) — enseignée : [ch. 21, 21.2.5 · Le noyau][s30] · exercices : 21.2–3, 21.6–7, 21.12, 21.14–15, 21.18, 21.20, 21.Q2 (+ 6)
- **noyau (d'une opération)** (*kernel*) — enseignée : [ch. 0A, 100.1.1 · Cellules, noyau, ordre d'exécution et « Tout exécuter »][s64] ; [ch. B3, Le masque causal quand il y a moins de requêtes que de clés][s571] · exercices : 0A.13, 0A.26, 0A.63–66, 0A.Q1, B3.27
- **noyau de Shapley** (*Shapley kernel*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s635] · exercices : B6.11
- **noyau RBF (gaussien)** (*RBF kernel*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s210] · exercices : 13.5, 13.11–12, 13.17–18, 13.Q2, 13.Q5–Q6, 13.Q12, 15.19
- **np.random.default_rng, graine, permutation, choice** — enseignée : [ch. 0A, 100.8.7 · Aléatoire reproductible et mini-batches][s7] · exercices : 0A.31, 0A.66, 0A.Q11, 0B.52–53, 1.22, 2.14, 2.17–19 (+ 62) · réutilisée : ch. 0B, 1, 2, 8, 16
- NPU → voir [GPU, TPU, NPU](#g)
- *NPV* → voir [valeur prédictive négative (VPN)](#v)
- **nuage de points** (*scatter plot*) — enseignée : [ch. 2, 2.8 · Covariance et corrélation][s196] · exercices : 0A.36, 0A.60, 2.25, 2.27, 10.19
- **nuage gaussien** (*blob*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s210] · exercices : 27.15
- *nucleus sampling* → voir [top-k, top-p](#t)
- *nuisance parameter* → voir [paramètre de nuisance](#p)
- *null condition* → voir [condition vide](#c)
- *null hypothesis* → voir [hypothèse nulle](#h)
- *null player* → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *numerical data* → voir [donnée numérique (quantitative)](#d)
- **NumPy : arrays, dtype, shape, indexation, masques, vues/copies** — enseignée : [ch. 0A, 100.8.2 · Indexation, tranches, masques booléens ; vues et copies][s94] · exercices : 0A.7, 0A.28–29, 0A.56, 0A.65, 0A.Q10 · réutilisée : ch. 0B
- **NumPy : calcul vectorisé, broadcasting, réductions par axe, tri, argmax, unique** — enseignée : [ch. 0A, 100.8.4 · Broadcasting][s91] · exercices : 0A.49, 0A.51, 2.R2, 7.13, 7.27 · réutilisée : ch. 2, 7
- **NumPy : reshape, transpose, empilement** — enseignée : [ch. 0A, 100.8.6 · `reshape`, transposition, empilement][s721] · exercices : 0A.7, 0A.53, 0A.Q11 · réutilisée : ch. 1

## O

- *object* → voir [classe, objet, méthode, attribut](#c)
- **objectif clippé, PPO** (*clipped objective, proximal policy optimization*) — enseignée : [ch. B8, 208.5 · PPO : ratio de probabilités, objectif clippé, entropie,…][s693] · exercices : B8.5, B8.9, B8.20–23, B8.26–27, B8.Q8–Q9 (+ 12)
- **objectif de substitution** (*surrogate objective*) — enseignée : [ch. B8, Réutiliser un rollout sans tout casser][s692]
- objet → voir [classe, objet, méthode, attribut](#c)
- *objet de mise à l'échelle* → voir [scaler](#s)
- **observabilité partielle** (*partial observability*) — enseignée : [ch. 26, 26.3.5 · La vue d'ensemble][s461]
- observation → voir [hypothèse, observation](#h)
- *observational Shapley* → voir [Shapley interventionnel, observationnel](#s)
- observationnel → voir [Shapley interventionnel, observationnel](#s)
- **octave** — enseignée : [ch. 28, 28.2.2 · Visualiser un filtre][s485] · exercices : 28.15–16
- *odds* → voir [cote](#c)
- off-policy → voir [on-policy, off-policy](#o)
- *offline algorithm* → voir [algorithme en ligne, hors ligne](#a)
- *offline RL* → voir [RL hors ligne](#r)
- **offsets** (*décalages*) — enseignée : [ch. B2, La pré-tokenisation][s525]
- *Okapi BM25* → voir [BM25](#b)
- *OLS* → voir [moindres carrés](#m)
- **on-policy, off-policy** (*sur politique, hors politique*) — enseignée : [ch. 26, 26.7 · SARSA][s466] · exercices : 26.E1, CP6.3, B8.Q9
- **one-hot** (*encodage one-hot, codage disjonctif*) — enseignée : [ch. 12, 12.3.1 · Le one-hot][s98] ; [ch. B2, 202.7 · Embeddings : de one-hot à `nn.Embedding`, entraînement et…][s22] ; [ch. 0A][s69] · exercices : 0A.65, 12.1, 12.20, 12.25, 12.32, 12.Q3, 22.Q4, 23.4, 23.Q9, B2.5 (+ 13) · réutilisée : ch. 18, 20, 22, 23, 24
- **one-hot encoding** — enseignée : [ch. 12, 12.3.1 · Le one-hot][s98] · réutilisée : ch. 18
- *one-standard-error rule* → voir [règle d'une erreur type](#r)
- **one-to-many, many-to-one, many-to-many** — enseignée : [ch. 22, 22.8.1 · Une ou plusieurs entrées et sorties][s416] · exercices : 22.Q10
- *one-versus-one* → voir [un-contre-un (OvO)](#u)
- *one-versus-rest* → voir [un-contre-tous (OvR)](#u)
- *one-vs-all* → voir [un-contre-tous (OvR)](#u)
- *online* → voir [algorithme en ligne, hors ligne](#a)
- *online inference* → voir [inférence par batches, en ligne](#i)
- *online softmax* → voir [softmax en ligne](#s)
- **ONNX** (*Open Neural Network Exchange*) — enseignée : [ch. 15, Au-delà du livre : sauvegarder un modèle et ses versions][s335] · exercices : B7.21, B7.Q10, B7.R3
- OOB → voir [out-of-bag (OOB)](#o)
- *Open Neural Network Exchange* → voir [ONNX](#o)
- **open weights** (*poids ouverts*) — enseignée : [ch. B4, Licences : « open weights » n'est pas « open source »][s583] · exercices : B4.9, B4.Q2
- *operant conditioning* → voir [conditionnement opérant](#c)
- *operating point* → voir [point de fonctionnement](#p)
- *optimal denoiser* → voir [débruiteur parfait (d'un dataset fini)](#d)
- *optimal discriminator* → voir [discriminateur optimal](#d)
- **optimisation** (*optimization*) — enseignée : [ch. 11, 11.2.3 · L'optimisation][s293] · exercices : 5.10, 5.E4, 11.12, 11.Q1, 12.R1, 15.E2, 18.R3, 20.12, 23.14, 24.5 (+ 10)
- **optimisation bayésienne** (*Bayesian optimization*) — enseignée : [ch. 15, 15.9.4 · La recherche aléatoire][s330] · exercices : 15.E2
- **optimiseur** (*optimizer*) — enseignée : [ch. 19, 19.6 · Choisir un optimiseur][s265] ; [ch. 8, 8.2 · La boucle d'entraînement][s9] · exercices : 19.9, 19.23, 19.26–27, 19.Q1, 19.Q12, 22.R2, 23.23, 27.27, B8.R2 (+ 79) · approfondie : ch. 27 · réutilisée : ch. 20, 25, 27, 28
- *optimistic bias* → voir [biais d'optimisme (du gagnant)](#b)
- *optimistic initial values* → voir [initialisation optimiste](#i)
- *optimization* → voir [optimisation](#o)
- *optimizer* → voir [optimiseur](#o)
- *optimizer state* → voir [tampon (d'un optimiseur)](#t)
- **oracle** — enseignée : [ch. 0A, 100.11.5 · mylearn : stubs, référence et tests fondés sur un oracle][s103] · exercices : 0A.63–66, 0B.38–39, 0B.42–43, 2.13, 2.15 (+ 62)
- *order of magnitude* → voir [ordre de grandeur](#o)
- *ordinal data* → voir [donnée ordinale](#d)
- *ordinal encoding* → voir [encodage ordinal](#e)
- *ordinary least squares* → voir [moindres carrés](#m)
- **ordonnée à l'origine** (*intercept*) — enseignée : [ch. 0B, 101.2.1 · Fonction, graphe, fonction affine][s77] ; [ch. 9, 9.7 · Ajuster une droite avec la règle de Bayes][s279] · exercices : 0B.6, 0B.10, 1.R3, 9.2–3, 9.6, 9.17, 9.26, 9.28, 9.30 (+ 6) · réutilisée : ch. 1
- ordre → voir [ordre C, ligne par ligne](#o), [ordre de grandeur](#o), [ordre topologique](#o)
- **ordre C, ligne par ligne** (*C order, row-major order*) — enseignée : [ch. 23, 23.5.1 · Remodeler][s423] · exercices : 0A.38, 0B.9, 9.31, 10.Q2, 12.12, 18.4, 18.8, 19.12, 23.1–2 (+ 10)
- **ordre de grandeur** (*order of magnitude*) — enseignée : [ch. 0B, 101.1.1 · Puissances, racines, notation scientifique, ordres de…][s104] · exercices : 0B.1, 0B.34, 1.5, 4.21, 5.20, 7.1, 18.27, 20.5, 20.8–9 (+ 8)
- **ordre topologique** (*topological order*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s338] · exercices : 16.4, 16.16, 18.26 · réutilisée : ch. 18
- *orthogonal* → voir [orthogonaux](#o)
- **orthogonaux** (*orthogonal*) — enseignée : [ch. 0B, 101.3.3 · Produit scalaire, angle, similarité cosinus][s115] · exercices : 0B.18, CP1.2, 25.Q9, B2.Q12
- **oubli catastrophique** (*catastrophic forgetting*) — enseignée : [ch. B1, L'oubli catastrophique][s508] · exercices : B1.23, B1.Q9, B4.R2
- **out-of-bag (OOB)** (*hors du sac*) — enseignée : [ch. 14, 14.4 · Le bagging][s190] · exercices : 14.3, 14.15, 14.17, 14.Q4–Q5, 15.R1, CP3.12
- *outcome* → voir [événement, issue](#e)
- *outer product* → voir [produit extérieur](#p)
- *outlier* → voir [point aberrant](#p)
- *output layer* → voir [couche de sortie](#c)
- *output projection* → voir [projection de sortie](#p)
- *OvA* → voir [un-contre-tous (OvR)](#u)
- *overestimation bias* → voir [surestimation](#s)
- *overfit-one-batch test* → voir [test d'overfitting d'un mini-batch](#t)
- **overfitting** (*surapprentissage, surajustement*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s33] · exercices : 9.9, 9.E4, 9.Q1, 9.Q4, 13.E3, 20.25, 20.29, 20.R3, 24.Q2, B7.15 (+ 72) · réutilisée : ch. 11, 13, 20, 23, 24
- *overlap* → voir [recouvrement](#r)
- *override* → voir [surcharge (de configuration)](#s)
- *overwhelming exception* → voir [exception écrasante](#e)
- OvO → voir [un-contre-un (OvO)](#u)
- OvR → voir [un-contre-tous (OvR)](#u)

## P

- **p-valeur** (*p-value*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s269] · exercices : 8.26, 8.E2, B7.22, B7.Q11
- *p-value* → voir [p-valeur](#p)
- **p50, p95, p99** — enseignée : [ch. B7, Latence, percentiles et débit][s516] · exercices : B7.3, B7.20, B7.25, B7.Q9, PF.11
- p95 → voir [p50, p95, p99](#p)
- p99 → voir [p50, p95, p99](#p)
- *package* → voir [module, paquet](#m), [paquet](#p)
- **padding** (*remplissage, marge*) — enseignée : [ch. 21, 21.3.4 · Le padding][s27] · évoquée : [ch. 20, 20.4.5 · La convolution (aperçu)][s28] · exercices : 21.2–3, 21.12, 21.15–16, 21.E2, 21.Q5, 24.R2, 25.4, B3.17 (+ 52) · approfondie : ch. B3
- *padding mask* → voir [masque de padding](#m)
- **PagedAttention (attention paginée)** — enseignée : [ch. B3, Ce que coûte le cache][s575]
- *paired bootstrap* → voir [bootstrap apparié](#b)
- *paired comparison* → voir [comparaison appariée](#c)
- **pandas : read_csv, head, info, describe, loc, iloc, filtres, isna, dropna, fillna, duplicated, value_counts, groupby, to_numpy** — enseignée : [ch. 0A, 100.9.1 · `DataFrame` et `Series` : `read_csv`, `head`, `info`,…][s96] · exercices : 0A.32, 0A.Q12 · réutilisée : ch. 1, 2, 12, 29
- **panorama 2026** — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s21] · exercices : 1.24, 1.E4, 1.Q11
- **paquet** (*package*) — enseignée : [ch. 0A, 100.6.1 · Modules, imports et bibliothèque standard][s654] ; [ch. B7, Une structure de dépôt][s655] · exercices : B7.9, B7.12–15, B7.17–18, B7.Q1–Q2, PF.10 (+ 9)
- **par échantillon, par feature, par élément** (*samplewise, featurewise, elementwise*) — enseignée : [ch. 12, 12.9 · Par échantillon, par feature, par élément][s309] · exercices : 0B.45, 9.18, 12.5, 12.Q10, 13.22, 13.28, 13.Q11, 15.24, 17.8, 17.Q2 (+ 2) · réutilisée : ch. 22
- par élément → voir [par échantillon, par feature, par élément](#p)
- par feature → voir [par échantillon, par feature, par élément](#p)
- **paradoxe de Simpson** (*Simpson's paradox*) — enseignée : [ch. 2][s178]
- *param group* → voir [groupe de paramètres](#g)
- *parameter* → voir [fonction, paramètre, argument](#f), [paramètre (d'un modèle)](#p)
- *parameter group* → voir [groupe de paramètres](#g)
- *parameter-efficient fine-tuning* → voir [PEFT](#p)
- *parameterized policy* → voir [politique paramétrée](#p)
- paramètre → voir [fonction, paramètre, argument](#f)
- **paramètre (d'un modèle)** (*parameter*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s47] · exercices : 0A.23, 0A.Q6, 1.E2, 1.Q5, 5.9, 13.16, 16.12, 16.E3, 17.24, 20.Q6 (+ 33) · approfondie : ch. 16 · réutilisée : ch. 2, 21, 22, 23, 25, B8
- paramètre C → voir [marge souple, paramètre C](#m)
- **paramètre de décroissance** (*decay parameter*) — enseignée : [ch. 19, 19.3.2 · Changer le learning rate au fil du temps][s359]
- **paramètre de nuisance** (*nuisance parameter*) — enseignée : [ch. 4][s221] · exercices : 4.11
- *parametric* → voir [classifieur paramétrique, non paramétrique](#c), [mémoire paramétrique, non paramétrique](#m)
- *parametric blending* → voir [mélange paramétrique](#m)
- *parametric ReLU* → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- **paramétrique vs non paramétrique** — enseignée : [ch. 13, 13.2 · Paramétrique ou non paramétrique ?][s294] · exercices : 13.8, 13.12, 13.31–32, 13.35, 13.E5, 13.Q2
- *parametrized test* → voir [test paramétré](#t)
- **parcimonieux** (*sparse*) — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s278]
- **parent, enfant ; ancêtre, descendant** (*child, ancestor*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s338] · exercices : 0A.Q1, 1.8, 13.20, 13.Q9, 16.4, 16.16, 16.Q2, 16.Q4, 16.R2, 18.26
- **parité démographique** (*demographic parity, statistical parity*) — enseignée : [ch. B6, 206.9 · Équité : indicateurs de groupe (parité démographique,…][s647] · exercices : B6.5, B6.28, B6.Q11
- **parité prédictive** (*predictive parity*) — enseignée : [ch. B6, Les critères][s648] · exercices : B6.6, B6.Q11
- **part de variance expliquée** (*explained variance ratio*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s199] · exercices : 12.7, 12.23
- **partage de poids** (*weight sharing*) — enseignée : [ch. 21, 21.2.3 · Le partage de poids][s387] · exercices : 21.Q2
- *partial* → voir [fine-tuning partiel, complet](#f)
- *partial dependence* → voir [dépendance partielle (PDP)](#d)
- *partial derivative* → voir [dérivée partielle](#d)
- *partial observability* → voir [observabilité partielle](#o)
- **partie entière (plancher / plafond)** (*floor / ceiling*) — enseignée : [ch. 0B, 101.1.2 · Valeur absolue, inégalités, partie entière, fonction signe][s105] · exercices : 0B.2, 5.Q3, 7.20, 8.13, 21.3 · réutilisée : ch. 21
- *partitionnement* → voir [clustering](#c)
- **pas de temps** (*time step*) — enseignée : [ch. 22, 22.4 · La structure d'une cellule RNN][s410] · exercices : 16.Q4, 20.14, 20.20, 22.Q3, 22.Q5, 24.27, CP5.4, B5.19, B5.21, B5.Q5
- passe arrière → voir [passe avant, passe arrière](#p)
- **passe avant matricielle** — enseignée : [ch. 16, Au-delà du livre : la passe avant en matrices][s350] · exercices : 16.3
- **passe avant, passe arrière** (*forward pass, backward pass*) — enseignée : [ch. 16, Au-delà du livre : la passe avant en matrices][s350] · exercices : 16.1–3, 16.7–8, 16.11, 16.Q3, 18.Q11, CP4.2, 25.28 (+ 56) · réutilisée : ch. 18
- **pasting** — enseignée : [ch. 14][s324]
- **patch (ViT)** — enseignée : [ch. B3, 203.10 · Coût et passage à l'échelle : complexité quadratique,…][s577] · exercices : 4.15, 21.Q12, B3.28, B5.8, B5.Q11, PF.6, PF.10
- **patience, `min_delta`** — enseignée : [ch. 9, 9.4 · L'early stopping][s273] · exercices : 9.5, 9.20, 9.Q6, 14.26, 20.17, 20.28, 23.6, 23.25–27 (+ 2) · réutilisée : ch. 20
- **PCA** (*analyse en composantes principales, principal component analysis, ACP*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s199] · exercices : 12.7, 12.27–28, 12.30, 12.E4, 12.Q9, 15.15, 15.Q8, B1.15, B2.27 (+ 11) · réutilisée : ch. 20, 21, B1, B2
- PDP → voir [dépendance partielle (PDP)](#d)
- *peak signal-to-noise ratio* → voir [PSNR](#p)
- *peaking phenomenon* → voir [phénomène de Hughes](#p)
- *Pearson correlation coefficient* → voir [corrélation (de Pearson)](#c)
- **PEFT** (*parameter-efficient fine-tuning*) — enseignée : [ch. B4, 204.8 · Fine-tuning : complet ou PEFT ; LoRA (rang, alpha, modules…][s597] · exercices : B4.25–28
- pénalité → voir [pénalité de gradient (WGAN-GP)](#p), [pénalité KL](#p), [pénalité L2 et weight decay](#p)
- **pénalité de gradient (WGAN-GP)** (*gradient penalty*) — enseignée : [ch. 27, 27.7 · Les difficultés][s483] · exercices : 27.Q10
- **pénalité KL** (*KL penalty*) — enseignée : [ch. B8, Le RLHF : PPO sous une laisse KL][s164] · exercices : 26.11, B8.10–11, B8.26, B8.Q11–Q12
- **pénalité L2 et weight decay** — enseignée : [ch. 20, Au-delà du livre : régulariser un réseau][s381] · exercices : B1.6
- **pente centrée** (*central difference*) — enseignée : [ch. 0B, 101.5.1 · Taux d'accroissement, tangente, dérivée][s120] · exercices : 0B.47–48, 0B.51, 5.11, 5.Q9
- **pente dans une direction, dérivée directionnelle** (*directional derivative*) — enseignée : [ch. 5, 5.4 · Le gradient][s125] · exercices : 5.2, 5.17
- **percentile, quantile, quartile** — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s185] · exercices : 2.13, 2.15, 2.22, 2.24, 3.29, CP1.5, 8.26, 12.19, B7.3, B7.19 · réutilisée : ch. B7
- **percentiles** — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s185] ; [ch. B7, Latence, percentiles et débit][s516] · exercices : 2.15, 2.22, 3.29, CP1.5, 8.26, B7.3, B7.19 · réutilisée : ch. B7
- **perceptron** — enseignée : [ch. 10, 10.3.1 · Le perceptron][s253] · exercices : 10.1, 10.4, 10.6–7, 10.11, 10.13, 10.17–18, 10.Q6, 10.Q8 (+ 32) · réutilisée : ch. 11, 13, 16, 17
- *perceptron convergence theorem* → voir [théorème de convergence du perceptron](#t)
- *perceptron cycling theorem* → voir [théorème du cycle du perceptron](#t)
- *perceptron learning rule* → voir [règle d'apprentissage du perceptron](#r)
- **perceptron moyenné** (*averaged perceptron*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s288]
- *perceptual loss* → voir [loss perceptuelle](#l)
- *permutation equivariance* → voir [équivariance par permutation](#e)
- *permutation importance* → voir [importance par permutation](#i)
- *permutation test* → voir [test par permutation](#t)
- *perpendicular bisector* → voir [médiatrice](#m)
- **perplexité** (*perplexity*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] ; [ch. B3, 203.8 · Mini-GPT from scratch : données, entraînement, loss,…][s248] ; [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s249] · exercices : 6.20, 6.E3, CP1.12, 22.26, B3.20–21, B3.26, B3.30, B4.26–27 (+ 11) · approfondie : ch. 28 · réutilisée : ch. B2, B4
- *perplexity* → voir [perplexité](#p)
- **persistance (d'un modèle)** (*model persistence*) — enseignée : [ch. 15, Au-delà du livre : sauvegarder un modèle et ses versions][s335] ; [ch. 22, Au-delà du livre : prévoir une série temporelle][s336] · exercices : 22.14, 22.21, 22.23, 22.29, 22.Q12
- **persistance (en prévision)** (*persistence, naive forecast*) — enseignée : [ch. 22, Au-delà du livre : prévoir une série temporelle][s336] · exercices : 22.14, 22.21, 22.23, 22.29, 22.Q12
- **persistance joblib** — enseignée : [ch. 15, Au-delà du livre : sauvegarder un modèle et ses versions][s335]
- *persistence* → voir [persistance (en prévision)](#p)
- *pertinence binaire* → voir [binary relevance](#b)
- **perturbation universelle** (*universal adversarial perturbation*) — enseignée : [ch. 21, 21.10 · Les adversaires][s402] · exercices : 21.Q11
- PGD → voir [FGSM, PGD](#f)
- **phénomène de Hughes** (*Hughes phenomenon, peaking phenomenon*) — enseignée : [ch. 7, 7.6 · La malédiction de la dimension][s192] · exercices : 7.30
- *piecewise linear* → voir [affine par morceaux](#a)
- **pile d'appels** (*call stack*) — enseignée : [ch. 0A, 100.5.6 · Fonctions récursives][s82]
- *pinball loss* → voir [loss quantile](#l)
- pip-tools → voir [uv, pip-tools, ruff, pre-commit](#u)
- **pipeline** (*chaîne de traitement*) — enseignée : [ch. 15, 15.9.5 · Les pipelines][s15] ; [ch. B4, Le pipeline : une ligne pour essayer][s16] ; [ch. 12, 12.10 · Transformations et validation croisée][s17] · exercices : 15.6, 15.22, 15.24, 15.30, 15.E3, 15.Q11, CP3.10, 24.17, CP5.10, B4.12 (+ 34) · réutilisée : ch. 22, 24, B1, B4, B6, B7
- **piratage de la récompense** (*reward hacking, specification gaming*) — enseignée : [ch. B8, Le piratage de la récompense][s472] ; [ch. 26, Au-delà du livre : le RL et les modèles de langage][s471] · exercices : B8.10, B8.26, B8.Q12
- *pivot table* → voir [tableau croisé](#t)
- pixel focal → voir [ancre, pixel focal](#a)
- **plaidoyer spécial** (*special pleading*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s299] · exercices : 11.Q10
- planning → voir [planning de bruit](#p), [planning en cosinus](#p), [planning linéaire, planning cosinus (de bruit)](#p), [planning par paliers](#p), [plannings de bruit linéaire et cosinus](#p), [plannings de learning rate](#p)
- **planning (de learning rate)** (*learning-rate schedule*) — enseignée : [ch. 19, 19.3.3 · Les plannings de décroissance][s358] · exercices : 0B.7, 19.1, 19.15, 19.Q7, 23.7, 23.25–26, B5.2–3, B5.13 (+ 36) · approfondie : ch. B5 · réutilisée : ch. 20, 23
- planning cosinus → voir [planning linéaire, planning cosinus (de bruit)](#p)
- **planning de bruit** (*noise schedule*) — enseignée : [ch. B5, 205.2 · Le processus direct : planning de bruit, ᾱ_t et forme…][s603] · exercices : B5.2, B5.Q7
- **planning en cosinus** (*cosine schedule, cosine annealing*) — enseignée : [ch. 0B, 101.2.6 · Cosinus (aperçu)][s112] · exercices : 0B.7
- **planning linéaire, planning cosinus (de bruit)** (*linear, cosine noise schedule*) — enseignée : [ch. B5, Le planning linéaire de Ho et coll.][s605] · exercices : 19.27, 23.25, B5.2–3, B5.6, B5.11–13, B5.15, B5.25
- **planning par paliers** (*step decay, interval decay*) — enseignée : [ch. 19, 19.3.3 · Les plannings de décroissance][s358] · exercices : 19.1, 19.23
- **plannings de bruit linéaire et cosinus** — enseignée : [ch. B5, Le planning cosinus de Nichol et Dhariwal][s783]
- **plannings de learning rate** — enseignée : [ch. 19, 19.3.3 · Les plannings de décroissance][s358] · exercices : 19.15, 23.7, 23.26 · réutilisée : ch. 23
- **plateau** — enseignée : [ch. 5, Les points où le gradient s'annule][s230] · exercices : 5.10, 5.Q6, 18.11, 18.21–22, 18.Q8, 18.Q10, 19.10, 19.Q2, 23.26 (+ 16) · réutilisée : ch. 19
- *Platt scaling* → voir [calibration de Platt, régression isotone](#c)
- *pli* → voir [fold](#f)
- *plurality voting* → voir [vote à la pluralité, vote pondéré](#v)
- **plus proche voisin (1-NN)** (*nearest neighbour*) — enseignée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s260] · exercices : 0B.19, 7.19–20, 7.Q11, 8.12, 8.19, 8.24, 8.27, 13.15, 13.34 (+ 8)
- **PMI, PPMI** (*pointwise mutual information, positive PMI*) — enseignée : [ch. B2, PMI et PPMI][s537] · exercices : B2.6–7, B2.22, B2.24, B2.Q10
- *pocket algorithm* → voir [algorithme pocket](#a)
- **poids** (*weights*) — enseignée : [ch. 16, 16.4 · Les poids][s48] ; [ch. 0B, 101.1.4 · Moyenne, moyenne pondérée, somme pondérée, moyenne mobile][s49] · exercices : 9.Q8, 10.5, 10.15, 10.Q9, 10.R1, 13.25, 16.1, 16.E1, 16.Q8–Q9 (+ 42) · approfondie : ch. 10, B5 · réutilisée : ch. 10, 21, B1, B6
- **poids implicites** (*implicit weights*) — enseignée : [ch. 10, 10.3.3 · Les neurones modernes][s50] · exercices : 10.Q9
- **poids liés** (*weight tying*) — enseignée : [ch. B2, Les poids liés][s533] · exercices : B2.9, B2.Q9, B3.5, B3.19, B3.29, B4.1, B4.30
- *poids ouverts* → voir [open weights](#o)
- point → voir [point aberrant](#p), [point anguleux](#p), [point cœur, point de bord, bruit](#p), [point critique, point stationnaire](#p), [point d'arrêt, débogueur](#p), [point d'entrée](#p), [point d'inflexion à tangente horizontale](#p), [point de code](#p), [point de fonctionnement](#p), [point influent](#p), [point selle, col](#p), [points critiques](#p)
- **point aberrant** (*outlier*) — enseignée : [ch. 9, Le point isolé][s275] · exercices : 9.10, 9.Q5, 12.19
- **point anguleux** (*corner, kink*) — enseignée : [ch. 5, 5.2 · Introduction : fonctions, courbes et surfaces][s224] · exercices : 5.21, 5.Q3, 17.12 · réutilisée : ch. 17
- **point cœur, point de bord, bruit** (*core point, border point, noise*) — enseignée : [ch. 7, Quand k-means échoue, et le clustering par densité][s261]
- **point critique, point stationnaire** (*critical point, stationary point*) — enseignée : [ch. 5, Les points où le gradient s'annule][s230] · exercices : 5.7, 5.10, 5.24, CP1.9
- **point d'arrêt, débogueur** (*breakpoint, debugger*) — enseignée : [ch. 23, 23.2.2 · Programmer et déboguer en Python][s371]
- **point d'entrée** (*entry point*) — enseignée : [ch. B7, 207.2 · Scripts et ligne de commande : argparse, point d'entrée,…][s657] · exercices : B7.9–10, B7.Q3
- **point d'inflexion à tangente horizontale** (*stationary inflection point*) — enseignée : [ch. 5, Tangente et dérivée][s226]
- point de bord → voir [point cœur, point de bord, bruit](#p)
- **point de code** (*code point*) — enseignée : [ch. B2, Points de code et octets][s522] · exercices : B2.1, B2.3, B2.15–16, B2.Q2–Q3
- **point de fonctionnement** (*operating point*) — enseignée : [ch. 3][s215]
- **point influent** (*influential point*) — enseignée : [ch. 2][s178]
- **point selle, col** (*saddle point*) — enseignée : [ch. 5, Les points où le gradient s'annule][s230] · exercices : 5.5, 5.10, 5.20, 5.E4, 5.Q8, CP1.9, 18.Q8, 19.Q2–Q3, 27.2 · réutilisée : ch. 19
- point stationnaire → voir [point critique, point stationnaire](#p)
- **points critiques** — enseignée : [ch. 5, Les points où le gradient s'annule][s230] · exercices : 5.5, 5.17, 5.24
- *pointwise convolution* → voir [convolution 1 × 1](#c)
- *pointwise mutual information* → voir [PMI, PPMI](#p)
- *policy* → voir [politique](#p)
- *policy gradient* → voir [gradient de politique](#g)
- *policy gradient theorem* → voir [théorème du gradient de politique](#t)
- **politique** (*policy*) — enseignée : [ch. 26, 26.6.2 · La politique du Q-learning][s155] · exercices : 7.15, 7.Q4, 26.19, 26.21, 26.Q3, B8.3, B8.7, B8.12, B8.Q5, B8.Q10 (+ 52) · approfondie : ch. 7, B8
- **politique de seuil** (*threshold policy*) — enseignée : [ch. 7, 7.2.1 · Classification binaire en 2D][s127] · exercices : 7.15, 7.Q4
- **politique gloutonne** (*greedy policy*) — enseignée : [ch. 26, 26.7 · SARSA][s466] · exercices : 26.3, 26.19, 26.22–23, 26.28–29, 26.34–36, 26.38 (+ 9)
- **politique paramétrée** (*parameterized policy*) — enseignée : [ch. B8, 208.3 · Gradient de politique : politique paramétrée, REINFORCE,…][s473]
- **politique softmax, de Boltzmann** (*softmax policy, Boltzmann*) — enseignée : [ch. 26, 26.6.2 · La politique du Q-learning][s155] · exercices : B8.3, B8.7, B8.Q5, B8.R3
- *Polyak* → voir [mise à jour douce (de Polyak)](#m)
- *polynomial features* → voir [features polynomiales](#f)
- **PolynomialFeatures** — enseignée : [ch. 15, 15.9.5 · Les pipelines][s15] · exercices : 8.16, 9.12, 9.14–15, 9.21, 15.1, 15.4, 15.17, 15.21–22 (+ 1)
- **pondération des classes** (*class weighting*) — enseignée : [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s211]
- pondérée → voir [moyenne macro, micro, pondérée](#m)
- **pooling** (*agrégation, sous-échantillonnage*) — enseignée : [ch. 20, 20.4.6 · Le pooling (aperçu)][s26] · exercices : 20.6, 20.20–21, 20.Q7, 21.5, 21.26, 24.2–3, 24.Q9, B2.28 (+ 31) · approfondie : ch. 21 · réutilisée : ch. 24, 25
- **pooling moyen global** (*global average pooling*) — enseignée : [ch. 21, 21.4.2 · Le stride pour construire une hiérarchie][s397] · exercices : 21.22, 21.Q12, 24.22, 24.Q9, B1.16, B1.R3, B6.Q8
- *population stability index* → voir [PSI (indice de stabilité de population)](#p)
- **population, échantillon (en statistique)** (*sample*) — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s185] ; [ch. 11, 11.5 · L'induction][s188] · exercices : 0A.67, 2.21, 2.23, 2.Q8, 11.17–18, 11.Q9, 11.R2, 29.R3, B7.4 (+ 19)
- **porte** (*gate*) — enseignée : [ch. 22, 22.7.1 · Les portes][s414] · exercices : 1.3, 10.3, 10.18, 22.4, 22.7, 22.11, 22.16–17, 22.Q7–Q8 (+ 42)
- **portée** (*scope*) — enseignée : [ch. 0A, 100.5.1 · Définir une fonction : paramètres, valeurs par défaut,…][s78] · exercices : 0A.6, 22.7, 28.23, B7.5
- *position-wise feed-forward network* → voir [bloc feed-forward (FFN)](#b)
- *positional encoding* → voir [encodage de position](#e)
- **positions apprises** (*learned positional embeddings*) — enseignée : [ch. B3, Les positions apprises][s564] · exercices : B3.5, B3.19, B3.25, B3.28, B3.Q7
- *positive class* → voir [classe positive, négative](#c)
- *positive PMI* → voir [PMI, PPMI](#p)
- *post-hoc explanation* → voir [explication post hoc](#e)
- post-LN → voir [pré-LN, post-LN](#p)
- **posterior** (*a posteriori, loi a posteriori*) — enseignée : [ch. 4, 4.6.1 · La boucle posterior → prior][s44] · exercices : 4.1–3, 4.5, 4.7, 4.16, 4.18, 4.22, 4.Q6, 4.Q9 (+ 9)
- **posterior (d'un pas)** (*forward process posterior*) — enseignée : [ch. B5, Le posterior : un pas en arrière quand on connaît $\mathbf{x}_0$][s609] · exercices : B5.4, B5.7, B5.11, B5.16–17, B5.19, B5.Q8
- *posterior collapse* → voir [effondrement du posterior](#e)
- **posterior gaussien** — enseignée : [ch. B5, Le posterior : un pas en arrière quand on connaît $\mathbf{x}_0$][s609]
- PPMI → voir [PMI, PPMI](#p)
- **PPMI + SVD** — enseignée : [ch. B2, Les plus proches voisins][s775] · exercices : B2.24
- PPO → voir [objectif clippé, PPO](#o)
- *PR baseline* → voir [ligne de base d'une courbe precision-recall](#l)
- *pre-activation* → voir [somme pondérée, pré-activation](#s)
- pré-activation → voir [somme pondérée, pré-activation](#s)
- pre-commit → voir [uv, pip-tools, ruff, pre-commit](#u)
- **pré-élagage, élagage** (*pre-pruning, pruning*) — enseignée : [ch. 13, 13.5.3 · Limiter l'overfitting][s317] · exercices : 13.Q10
- **pré-entraînement** (*pre-training*) — enseignée : [ch. 24, 24.6 · L'API fonctionnelle][s171] ; [ch. 1][s144] · exercices : 1.Q11, B1.7, B1.9, B1.20, B1.22, B1.Q1, B1.R1, B3.Q9, B4.Q1, B4.Q7
- *pre-LN* → voir [pré-LN, post-LN](#p)
- **pré-LN, post-LN** (*pre-LN*) — enseignée : [ch. B3, Pré-LN ou post-LN][s566] · exercices : B3.5, B3.11, B3.16, B3.19, B3.25, B3.28, B3.Q8
- *pre-pruning* → voir [pré-élagage, élagage](#p)
- *pre-tokenizer* → voir [pré-tokenizer](#p)
- **pré-tokenizer** (*pre-tokenizer*) — enseignée : [ch. B2, La pré-tokenisation][s525] · exercices : B2.17, B2.19, B2.21, B2.31, B2.Q6, B3.26
- *pre-training* → voir [pré-entraînement](#p)
- **precision** (*précision, valeur prédictive positive, VPP*) — enseignée : [ch. 3, 3.7.6 · Precision][s39] · exercices : 3.9, 3.16, 3.20–21, 3.26, 3.E2, 3.Q10, 4.R1, CP1.6, CP3.11 (+ 33) · réutilisée : ch. 4, 7, 15, B6
- précision → voir [précision locale, absence, cohérence](#p), [précision mixte](#p), [précision mixte et torch.compile](#p)
- *precision and recall for generative models* → voir [precision et recall génératifs](#p)
- **precision et recall génératifs** (*precision and recall for generative models*) — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s481] ; [ch. B5, Fidélité et diversité][s484]
- **précision locale, absence, cohérence** (*local accuracy, missingness, consistency*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s635] · exercices : B6.11
- **précision mixte** (*mixed precision*) — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s444] · exercices : 19.9, 24.7, 24.29
- **précision mixte et torch.compile** — enseignée : [ch. 24, Au-delà du livre : précision mixte et compilation][s444] · exercices : 24.29
- *précision moyenne* → voir [average precision (AP)](#a)
- *precision-recall curve* → voir [courbe precision-recall](#c)
- **predict_proba** — enseignée : [ch. 15, 15.4.4 · decision_function et predict_proba][s742] · exercices : 1.19, 3.Q1, 15.2, 15.10, 15.21, 15.25–26, 15.E4, 15.Q5, CP3.10 (+ 19)
- *predicted value* → voir [valeur prédite](#v)
- **prédicteur** (*predictor*) — enseignée : [ch. 15, 15.4 · Les estimateurs][s255] · exercices : 9.4, 9.11, 15.8, PF.5
- *prediction* → voir [prédiction (principe inductif)](#p)
- *ε-prediction* → voir [prédiction du bruit](#p)
- **prédiction (principe inductif)** (*prediction*) — enseignée : [ch. 11, 11.5 · L'induction][s188] · exercices : 3.Q6, 7.Q1, 9.4, 11.6, 11.Q9, B1.23, B5.1, B6.20, B6.E1, B7.7 (+ 5) · approfondie : ch. B5, B7
- *prediction bias* → voir [biais de prédiction](#b)
- **prédiction du bruit** (*ε-prediction, noise prediction*) — enseignée : [ch. B5, Trois façons de fabriquer une image][s610] · exercices : B5.21
- *predictive parity* → voir [parité prédictive](#p)
- *predictor* → voir [prédicteur](#p)
- **prefill** — enseignée : [ch. B3, Le KV cache][s570] · exercices : B3.23, B4.7
- *prefix code* → voir [code préfixe](#c)
- *prefix-free code* → voir [code préfixe](#c)
- PReLU → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- **préparation des données** (*data preparation, data cleaning*) — enseignée : [ch. 12, 12.1 · Pourquoi ce chapitre ?][s302] · exercices : 12.Q1–Q10 (+ 42) · réutilisée : ch. 29
- *pretrained model* → voir [modèle pré-entraîné](#m)
- *prevalence* → voir [prévalence, taux de base](#p)
- **prévalence, taux de base** (*prevalence, base rate*) — enseignée : [ch. 3, 3.8 · Appliquer la matrice de confusion : le piège de la prévalence][s206] · évoquée : [ch. B6, L'impossibilité][s207] · exercices : 3.5, 3.7, 3.10, 3.21, 4.R1, B6.5–6, B6.24–26 (+ 1) · réutilisée : ch. 4
- **prévision de série temporelle avec baseline** — enseignée : [ch. 22, Au-delà du livre : prévoir une série temporelle][s336]
- *principal component* → voir [composante principale](#c)
- *principal component analysis* → voir [PCA](#p)
- **prior** (*a priori, loi a priori*) — enseignée : [ch. 4, 4.6.1 · La boucle posterior → prior][s44] · exercices : 4.5, 4.7, 4.10, 4.16, 4.21, 4.E3, 4.Q6, 4.Q9, 5.R1, 9.27 (+ 33) · réutilisée : ch. 6, 13
- **prior conjugué** (*conjugate prior*) — enseignée : [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s222]
- **prior de Laplace** (*Laplace prior*) — enseignée : [ch. 9][s281]
- *PRNG* → voir [pseudo-aléatoire, générateur](#p)
- *probabilistic programming* → voir [programmation probabiliste](#p)
- probabilité → voir [probabilité conditionnelle](#p), [probabilité jointe](#p), [probabilité marginale](#p), [probabilité simple, marginale](#p), [probabilités : événement, complémentaire, union, indépendance](#p)
- **probabilité conditionnelle** (*conditional probability*) — enseignée : [ch. 3, 3.4 · Probabilité conditionnelle][s200] · exercices : 3.Q3, 4.15, 4.Q8, 4.R1 · réutilisée : ch. 6, 11, 13
- **probabilité jointe** (*joint probability*) — enseignée : [ch. 3, 3.5 · Probabilité jointe][s201] · exercices : 3.14, 3.Q4
- **probabilité marginale** — enseignée : [ch. 3, 3.6 · Probabilité marginale et tables de contingence][s203] · exercices : 3.3–4, 3.14, 3.Q5, 3.R3
- **probabilité simple, marginale** (*simple probability, marginal probability*) — enseignée : [ch. 3, 3.3 · Probabilité simple][s131] · exercices : 3.1, 3.12, 3.Q2
- **probabilités : événement, complémentaire, union, indépendance** — enseignée : [ch. 0B, 101.7.2 · Indépendance][s132] · exercices : 0B.26, 0B.52, 0B.E3, 0B.Q12
- *probability density function* → voir [densité (pdf)](#d)
- *probability distribution* → voir [distribution de probabilité, loi](#d)
- *probability mass function* → voir [fonction de masse (pmf)](#f)
- *probability ratio* → voir [ratio de probabilités](#r)
- *problem of induction* → voir [problème de l'induction](#p)
- **problème de l'arrêt** (*halting problem*) — enseignée : [ch. 11, 11.2.1 · La représentation][s291] · exercices : 11.Q3
- **problème de l'induction** (*problem of induction*) — enseignée : [ch. 11, 11.6 · Raisonnement combiné][s300] · exercices : 11.12
- *processeur central* → voir [CPU](#c)
- processus → voir [processus de décision markovien (MDP)](#p), [processus direct](#p), [processus direct et forme fermée](#p), [processus inverse](#p)
- *processus de chargement* → voir [worker (d'un `DataLoader`)](#w)
- **processus de décision markovien (MDP)** (*Markov decision process*) — enseignée : [ch. 26, 26.3.5 · La vue d'ensemble][s461] · exercices : 26.8, 26.Q5
- **processus direct** (*forward process*) — enseignée : [ch. B5, 205.2 · Le processus direct : planning de bruit, ᾱ_t et forme…][s603] · exercices : B5.6, B5.22, B5.Q1
- **processus direct et forme fermée** — enseignée : [ch. B5, 205.2 · Le processus direct : planning de bruit, ᾱ_t et forme…][s603] · exercices : B5.2–3, B5.6, B5.11–14, B5.Q2–Q3, B5.R3 (+ 2)
- **processus inverse** (*reverse process*) — enseignée : [ch. B5, 205.3 · Le processus inverse et la loss simplifiée : prédire le…][s604] · exercices : B5.7, B5.11, B5.16, B5.19, B5.23, B5.E1–E2, B5.Q4, B5.R3
- *product rule* → voir [règle du produit](#r)
- produit → voir [produit de Hadamard](#p), [produit extérieur](#p), [produit matriciel](#p), [produit scalaire](#p)
- **produit de Hadamard** (*element-wise product, Hadamard product*) — enseignée : [ch. 0B, 101.3.4 · Produit élément par élément (Hadamard)][s117] · exercices : 0B.18, 0B.Q8
- **produit extérieur** (*outer product*) — enseignée : [ch. 0B, 101.4.3 · Produit matriciel et vérification des formes][s118] · exercices : 0B.45
- **produit matriciel** (*matrix multiplication*) — enseignée : [ch. 0B, 101.4.3 · Produit matriciel et vérification des formes][s118] · exercices : 0B.20, 0B.30, 0B.43–45, 0B.54, 2.28, 7.13, 16.R3, 17.8 (+ 15) · réutilisée : ch. 9, 16, 28, B3
- **produit scalaire** (*dot product, inner product*) — enseignée : [ch. 0B, 101.3.3 · Produit scalaire, angle, similarité cosinus][s115] · exercices : 0B.18–19, 0B.39–40, 0B.E2, 0B.Q8, CP1.2, 7.R3, 21.R3, B3.6 (+ 23) · approfondie : ch. B3 · réutilisée : ch. 7, 10, 12, 13, 21, B2
- **profondeur (d'un arbre)** (*depth*) — enseignée : [ch. 13, 13.5 · Les arbres de décision][s169] · exercices : 0A.44, 13.11, 13.19, 13.Q7, 13.Q10, 16.R2, 21.Q1, 24.26, 25.22, 25.Q6 (+ 29)
- **programmation dynamique** (*dynamic programming*) — enseignée : [ch. B2][s538]
- **programmation probabiliste** (*probabilistic programming*) — enseignée : [ch. 4, Au-delà du livre (2) : la limite continue, la loi Beta][s222]
- *progressive distillation* → voir [distillation progressive](#d)
- *progressive growing* → voir [croissance progressive](#c)
- *projected gradient descent* → voir [FGSM, PGD](#f)
- **projection de sortie** (*output projection*) — enseignée : [ch. B3, Plusieurs questions à la fois][s560] · exercices : B3.14, B3.18
- **prompt** (*invite, instruction*) — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s21] ; [ch. B3, Le décodeur : GPT et le modèle de langue causal][s32] · exercices : B1.24, B3.10, B3.Q9, B4.4, B4.13, B4.16, B4.21, B4.24, B4.29, B4.Q12 (+ 18) · approfondie : ch. B4
- *prompt ensembling* → voir [ensemble de prompts](#e)
- *prompt injection* → voir [injection de prompt](#i)
- *prompt template* → voir [gabarit (de prompt)](#g)
- **prompting et sortie structurée** — enseignée : [ch. B4, 204.4 · Prompting : zéro-shot, few-shot, instructions, sortie…][s779] · exercices : B4.15–16, B4.E1, B4.Q6
- *property-based test* → voir [test de propriété](#t)
- *property-based testing* → voir [test de propriétés](#t)
- *prosecutor's fallacy* → voir [erreur du procureur](#e)
- *protected attribute* → voir [attribut sensible](#a)
- **prototype de classe** (*class prototype*) — enseignée : [ch. B1, Le few-shot et les prototypes de classe][s511]
- *proximal policy optimization* → voir [objectif clippé, PPO](#o)
- *proxy* → voir [variable de substitution](#v)
- *proxy variable* → voir [variable de substitution](#v)
- *pruning* → voir [pré-élagage, élagage](#p)
- **pseudo-aléatoire, générateur** (*pseudo-random, pseudo-random number generator, PRNG*) — enseignée : [ch. 2, 2.2.1 · Nombres aléatoires en pratique][s180] · exercices : 2.14, 2.Q3
- pseudo-inverse → voir [solution de norme minimale, pseudo-inverse](#s)
- *pseudo-random* → voir [pseudo-aléatoire, générateur](#p)
- *pseudo-random number generator* → voir [pseudo-aléatoire, générateur](#p)
- *pseudo-residuals* → voir [pseudo-résidus](#p)
- **pseudo-résidus** (*pseudo-residuals*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s195] · exercices : 14.4
- **pseudonymisation** — enseignée : [ch. B7][s665]
- **PSI (indice de stabilité de population)** (*population stability index*) — enseignée : [ch. B7, L'indice de stabilité de population (PSI)][s675] · exercices : B7.4, B7.22, B7.Q11
- **PSNR** (*peak signal-to-noise ratio*) — enseignée : [ch. 25, 25.8 · Débruiter][s454] · exercices : 25.25–27, 25.39
- *public domain* → voir [domaine public](#d)
- **puce neuromorphique** (*neuromorphic chip*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s51]
- **puissance de représentation** (*representational power*) — enseignée : [ch. 11, 11.2.1 · La représentation][s291] · exercices : 11.Q2
- **puissances, racines, notation scientifique, ordres de grandeur** — enseignée : [ch. 0B, 101.1.1 · Puissances, racines, notation scientifique, ordres de…][s104] · exercices : 0B.1, 0B.30, 0B.33–34, 0B.Q1, 2.11, 5.9, 6.10, 7.4, 9.15 (+ 34) · réutilisée : ch. 11
- **pull request** (*demande de fusion*) — enseignée : [ch. 0A, 100.11.2 · git : branches (aperçu)][s101] · exercices : 0A.E5
- *punishment* → voir [renforcement, punition](#r)
- punition → voir [renforcement, punition](#r)
- **pureté (d'un clustering)** (*purity*) — enseignée : [ch. 7, Combien de clusters ?][s259] ; [ch. 13, 13.5.2 · Choisir les splits][s147] · exercices : 1.21, 7.17, 7.31
- *purity* → voir [pureté (d'un clustering)](#p)
- pydantic → voir [FastAPI, uvicorn, pydantic](#f)
- **pytest : assert, approx, raises, parametrize ; doctest** — enseignée : [ch. 0A, 100.11.4 · doctest : des exemples exécutables dans les docstrings][s724] · exercices : 0A.61, 0A.Q12

## Q

- **Q-learning** — enseignée : [ch. 26, 26.6 · Le Q-learning][s464] · exercices : 26.3, 26.8, 26.20, 26.22, 26.25, 26.28–29, 26.36, 26.Q8, 26.Q11 (+ 22) · réutilisée : ch. B8
- **Q-learning tabulaire** — enseignée : [ch. 26, 26.5 · Le L-learning][s463] · exercices : 26.12, 26.33, 26.36, 26.R2, B8.Q2
- *Q-table* → voir [table Q, valeur Q](#t)
- *Q-value* → voir [table Q, valeur Q](#t)
- **QK-Norm** (*query-key normalization*) — enseignée : [ch. B3, Compter les paramètres d'un bloc][s573]
- **QLoRA** (*quantized LoRA*) — enseignée : [ch. B4, QLoRA][s598] · exercices : B4.3, B4.6, B4.11, B4.25–27, B4.E2, B4.Q10–Q11, B4.R2 (+ 1)
- **quantification** (*quantization*) — enseignée : [ch. B4, La mémoire des poids][s445] · exercices : B7.1, B7.20
- **quantification dynamique, statique** (*dynamic, static quantization*) — enseignée : [ch. B7, Accélérer sur CPU][s446] · exercices : B7.1, B7.20
- quantile → voir [percentile, quantile, quartile](#p)
- *quantile loss* → voir [loss quantile](#l)
- *quantization* → voir [quantification](#q)
- *quantized LoRA* → voir [QLoRA](#q)
- **quartet d'Anscombe, Datasaurus** (*Anscombe's quartet, Datasaurus Dozen*) — enseignée : [ch. 2, 2.9 · Le quartet d'Anscombe][s194] · exercices : 2.30, 2.Q12
- quartile → voir [percentile, quantile, quartile](#p)
- **quasi-doublon** (*near-duplicate*) — enseignée : [ch. 29, Au-delà du livre : auditer un dataset][s496] · exercices : 29.13, 29.16, 29.E3, PF.3
- *query* → voir [requête, clé, valeur](#r)
- *query-key normalization* → voir [QK-Norm](#q)

## R

- **raccourci appris** (*shortcut learning*) — enseignée : [ch. 8, 8.2.1 · Pourquoi le score d'entraînement trompe][s266] · exercices : 8.Q4
- **raccourcis appris** — enseignée : [ch. 8, 8.2.1 · Pourquoi le score d'entraînement trompe][s266] · exercices : 8.9
- **racine, feuille, nœud interne** (*root, leaf, internal node*) — enseignée : [ch. 13, 13.5 · Les arbres de décision][s169] · exercices : 0B.1, 0B.6, 0B.Q1, 13.19–20, 13.23–25, 13.Q7, B5.22 (+ 29) · approfondie : ch. 0B · réutilisée : ch. 2, 27
- **racinisation** (*stemming*) — enseignée : [ch. B4, De TF-IDF à BM25][s593]
- **RAG** (*retrieval-augmented generation, génération augmentée par recherche*) — enseignée : [ch. B4, 204.7 · RAG : récupérer, augmenter le prompt, générer, citer ;…][s172] ; [ch. 1, Panorama 2026 : où en est l'IA ?][s21] · exercices : B4.7–9, B4.21–23, B4.30, B4.E2, B4.E4, B4.Q10 (+ 2)
- **RAG avec citations** — enseignée : [ch. B4, RAG ou fine-tuning ?][s781]
- raison → voir [suite géométrique, raison](#s)
- *raisonnement pas à pas* → voir [chain-of-thought](#c)
- *random features* → voir [features aléatoires](#f)
- *random forest* → voir [forêt aléatoire](#f)
- *random search* → voir [recherche au hasard, recherche aléatoire](#r)
- *random variable* → voir [variable aléatoire](#v)
- **randomisation de domaine** (*domain randomization*) — enseignée : [ch. B8, De la simulation au réel][s700]
- **randomisation en cascade, indépendante** (*cascading, independent randomization*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s635] · exercices : B6.Q9
- **RandomizedSearchCV** — enseignée : [ch. 15, 15.9.4 · La recherche aléatoire][s330] · exercices : 15.1, 15.19, 15.31, 15.Q10, CP3.1, CP3.14, 24.18 · réutilisée : ch. 24
- rang → voir [fusion (BPE), rang](#f)
- **rang (LoRA)** (*rank*) — enseignée : [ch. B4, LoRA][s173] · exercices : B2.17, B4.3, B4.6, B4.11, B4.25–28, B4.E3, B4.Q11 (+ 23)
- *rang réciproque moyen* → voir [MRR](#m)
- *rank* → voir [fusion (BPE), rang](#f), [rang (LoRA)](#r)
- *rank-stabilized LoRA* → voir [rsLoRA](#r)
- *rappel aux k premiers* → voir [recall@k](#r)
- **rapport de vraisemblance** (*likelihood ratio*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s218]
- **rapport signal sur bruit (SNR)** (*signal-to-noise ratio*) — enseignée : [ch. B5, Le rapport signal sur bruit][s608] · exercices : 19.12, B5.3, B5.13, B5.15, B5.Q3
- *ratio* → voir [suite géométrique, raison](#s)
- **ratio de probabilités** (*probability ratio*) — enseignée : [ch. B8, 208.5 · PPO : ratio de probabilités, objectif clippé, entropie,…][s693] · exercices : B8.Q9
- *RBF kernel* → voir [noyau RBF (gaussien)](#n)
- *reality gap* → voir [écart simulation-réalité, sim-to-real](#e)
- *reasoning model* → voir [modèle de raisonnement](#m)
- **recall** (*rappel, sensibilité, taux de vrais positifs, TPR*) — enseignée : [ch. 3, 3.7.7 · Recall][s40] · exercices : 3.9, 3.16, 3.20, 3.29, 3.E2, 3.Q10, 15.26, CP3.11, 27.R2, B4.2 (+ 36) · approfondie : ch. B4 · réutilisée : ch. 4, 7, 15
- **recall@k** (*rappel aux k premiers*) — enseignée : [ch. B4, Mesurer une recherche : recall@k et MRR][s595] · exercices : B4.2, B4.7, B4.18, B4.20, B4.23
- **recall@k et MRR** — enseignée : [ch. B4, Mesurer une recherche : recall@k et MRR][s595] · exercices : B4.2, B4.18, B4.20
- *receptive field* → voir [champ récepteur](#c)
- recherche → voir [recherche approchée des plus proches voisins](#r), [recherche au hasard, recherche aléatoire](#r), [recherche d'hyperparamètres](#r), [recherche dense](#r), [recherche dense et hybride](#r), [recherche en grille](#r), [recherche en largeur](#r), [recherche hybride](#r), [recherche multirésolution](#r), [recherche sémantique](#r)
- recherche aléatoire → voir [recherche au hasard, recherche aléatoire](#r)
- **recherche approchée des plus proches voisins** (*approximate nearest neighbor search, ANN*) — enseignée : [ch. 7, 7.6.1 · Bizarreries de la grande dimension][s260] · évoquée : [ch. 13, 13.3 · Les k plus proches voisins][s257]
- **recherche au hasard, recherche aléatoire** (*random search*) — enseignée : [ch. 15, 15.9.4 · La recherche aléatoire][s330] · exercices : 8.2, 15.5, 15.19, 15.Q10, 19.11, 24.1, 24.7, 24.18
- **recherche d'hyperparamètres** — enseignée : [ch. 24, 24.3.4 · La recherche d'hyperparamètres][s329] ; [ch. 8, 8.4 · Le jeu de validation][s168] · exercices : 8.2, 8.18, 8.Q7, 15.5, 15.31, 19.11, 24.7, 24.E1, PF.7 · réutilisée : ch. 19
- **recherche dense** (*dense retrieval*) — enseignée : [ch. B4, La recherche dense][s547] · exercices : B4.18–20, B4.22, B4.Q9
- **recherche dense et hybride** — enseignée : [ch. B4, 204.6 · Recherche d'information : découpage, BM25, recherche…][s592] · exercices : B4.5, B4.17–20, B4.22, B4.Q9, B4.R3
- **recherche en grille** (*grid search*) — enseignée : [ch. 15, 15.9.3 · La recherche en grille][s328] · évoquée : [ch. 24, 24.3.4 · La recherche d'hyperparamètres][s329] · exercices : 15.1, 15.5, 15.12, 15.18, 15.22, 15.E2, CP3.14, 18.R2, 19.11, 24.1 (+ 1)
- **recherche en largeur** (*breadth-first search*) — enseignée : [ch. 26][s62] · exercices : 26.17
- **recherche hybride** (*hybrid search*) — enseignée : [ch. B4, La recherche hybride et la fusion RRF][s594] · exercices : B4.18–20, B4.23
- **recherche multirésolution** (*coarse-to-fine search*) — enseignée : [ch. 24, 24.3.4 · La recherche d'hyperparamètres][s329] · exercices : 24.Q5
- **recherche sémantique** (*semantic search*) — enseignée : [ch. B2, 202.10 · Embeddings contextuels et embeddings de phrases ;…][s543] · exercices : B2.28, B2.E5
- **reciprocal rank fusion (RRF)** (*fusion par rangs réciproques*) — enseignée : [ch. B4, La recherche hybride et la fusion RRF][s594] · exercices : B4.18–20, B4.Q9
- récompense → voir [agent, environnement, action, récompense](#a)
- **récompense façonnée** (*reward shaping*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s154] · exercices : 26.32
- *reconstruction error* → voir [erreur de reconstruction](#e)
- **recouvrement** (*overlap*) — enseignée : [ch. B4, Découper][s591] · exercices : 22.13, B4.17, B4.19, B7.15
- **rectified flow** — enseignée : [ch. B5, Flow matching et rectified flow][s627] · exercices : B5.20, B5.E5
- *recurrent neural network* → voir [réseau récurrent (RNN)](#r)
- *recursive feature elimination* → voir [RFE](#r)
- **récursivité** — enseignée : [ch. 0A, 100.5.6 · Fonctions récursives][s82]
- **redondance (d'une source)** (*redundancy*) — enseignée : [ch. 6, 6.7 · L'entropie][s242] · exercices : B7.5
- **réduction de dimension** (*dimensionality reduction*) — enseignée : [ch. 1, 1.4.3 · Réduction de dimension][s150] ; [ch. 12, 12.7 · La réduction de dimension][s151] · exercices : 1.Q8, 2.R1, 12.Q8
- *réduction de moitié successive* → voir [successive halving](#s)
- *redundancy* → voir [redondance (d'une source)](#r)
- rééchantillon → voir [bootstrap, rééchantillon](#b)
- **réentraînement final** (*refit*) — enseignée : [ch. 15, 15.9.2 · Chercher des hyperparamètres][s332] · exercices : 8.2, 15.1, 15.5, 15.R3, CP3.14, 18.R2, 24.Q5
- *refactor* → voir [refactoriser](#r)
- **refactoriser** (*refactor*) — enseignée : [ch. 4][s221] · exercices : 4.23, 9.29
- référence de style → voir [image de contenu, référence de style, image générée](#i)
- *refit* → voir [réentraînement final](#r)
- **région de confiance, TRPO** (*trust region*) — enseignée : [ch. B8, Réutiliser un rollout sans tout casser][s692] · exercices : B8.11, B8.Q8
- **région de décision** (*decision region*) — enseignée : [ch. 7, 7.2.1 · Classification binaire en 2D][s127]
- règle → voir [règle 68-95-99,7](#r), [règle d'apprentissage du perceptron](#r), [règle d'or (de la préparation)](#r), [règle d'une erreur type](#r), [règle de Bayes, théorème de Bayes](#r), [règle de la chaîne](#r), [règle de la chaîne à plusieurs chemins](#r), [règle de la chaîne à plusieurs variables (somme sur les chemins, version simple)](#r), [règle de la chaîne multivariée](#r), [règle de succession de Laplace](#r), [règle de trois](#r), [règle de Tukey](#r), …
- **règle 68-95-99,7** (*68-95-99.7 rule, three-sigma rule*) — enseignée : [ch. 2, 2.3.2 · La loi normale][s182] · exercices : 2.3, 2.Q5
- **règle d'apprentissage du perceptron** (*perceptron learning rule*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s253] · exercices : 10.6, 17.R3
- règle d'arrêt → voir [test séquentiel, règle d'arrêt](#t)
- **règle d'or (de la préparation)** — enseignée : [ch. 12, 12.2 · La règle d'or : apprendre la transformation sur l'entraînement][s158] · exercices : 8.Q5, 12.9, 12.33, 12.Q1, 15.14
- **règle d'une erreur type** (*one-standard-error rule*) — enseignée : [mini-projet 2, Le protocole, le cœur du projet][s707] · exercices : 15.23
- **règle de Bayes, théorème de Bayes** (*Bayes' rule, Bayes' theorem*) — enseignée : [ch. 4, 4.4.1 · La règle de Bayes][s202] · exercices : 4.1, 4.3, 4.9–10, 4.14–15, 4.E1, 4.Q6, CP1.6, 7.R2 (+ 47) · réutilisée : ch. 7, 9, 11, 13
- **règle de la chaîne** (*chain rule*) — enseignée : [ch. 0B, 101.5.3 · Règle de la chaîne][s123] · exercices : 0B.23–24, 0B.29, 0B.E4, 5.7, 5.21, 5.R2–R3, 13.8, 18.6 (+ 13) · approfondie : ch. 18 · réutilisée : ch. 5, 13, 17, 18, 22, 25, 28, B2
- **règle de la chaîne à plusieurs chemins** (*multivariable chain rule*) — enseignée : [ch. 18, 18.8 · Étape 3 : les deltas des autres neurones][s352] · exercices : 18.6, 18.26
- **règle de la chaîne à plusieurs variables (somme sur les chemins, version simple)** — enseignée : [ch. 0B, 101.6.4 · Règle de la chaîne à plusieurs variables : la somme sur…][s128] · exercices : 0B.29, 0B.51, 0B.E4, 0B.Q11
- **règle de la chaîne multivariée** — enseignée : [ch. 18][s355]
- **règle de succession de Laplace** (*rule of succession*) — enseignée : [ch. 4, 4.4.1 · La règle de Bayes][s202] · exercices : 11.6, 11.R3
- **règle de trois** (*rule of three*) — enseignée : [mini-projet 1, La grille d'évaluation (sur 20)][s704]
- **règle de Tukey** (*Tukey's fences*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s304] · exercices : 12.19
- **règle des quatre cinquièmes** (*four-fifths rule*) — enseignée : [ch. B6, Les critères][s648] · exercices : B6.5, B6.26
- **règle du plus fort reste** (*largest remainder method*) — enseignée : [ch. 8, 8.3 · Le jeu de test][s143] · exercices : 8.1, 8.13
- **règle du produit** (*product rule*) — enseignée : [ch. 3, 3.5 · Probabilité jointe][s201] · évoquée : [ch. 4, 4.4.1 · La règle de Bayes][s202] · exercices : 3.4, 3.Q4, 4.1, 4.3, B5.7
- **règlement européen sur l'IA** (*AI Act (règlement 2024/1689), UE*) — enseignée : [ch. 20, 20.8.1 · Une explication satisfaisante][s383] · exercices : 21.10, 22.10, 25.12, 27.9, B3.10
- régression → voir [régression linéaire bayésienne](#r), [régression logistique](#r), [régression vers la moyenne](#r)
- régression isotone → voir [calibration de Platt, régression isotone](#c)
- **régression linéaire bayésienne** (*Bayesian linear regression*) — enseignée : [ch. 9, 9.7 · Ajuster une droite avec la règle de Bayes][s279]
- **régression logistique** (*logistic regression*) — enseignée : [ch. 13, Au-delà du livre : la régression logistique, le classifieur…][s109] · exercices : 13.8–9, 13.11, 13.30–32, 13.36, 13.E5, B4.28, B6.17 (+ 24) · réutilisée : ch. 15, 18, B1, B4
- *regression test* → voir [test de non-régression](#t)
- *regression to the mean* → voir [régression vers la moyenne](#r)
- **régression vers la moyenne** (*regression to the mean*) — enseignée : [ch. 1, 1.3.2 · Régression][s146]
- **regret** — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s156] · exercices : 11.8, 11.12, 11.21, 11.24–27
- *regroupement* → voir [clustering](#c)
- **regroupement dynamique** (*dynamic batching*) — enseignée : [ch. B7][s665] · exercices : B7.19
- *regular expression* → voir [expression régulière](#e)
- **régularisation** (*regularization*) — enseignée : [ch. 9, 9.5 · La régularisation][s276] · exercices : 9.19, 9.22, 9.28, 9.31, 9.E3, 9.Q3, 9.Q7, 10.R1, CP4.12, 25.R3 (+ 22) · réutilisée : ch. 10, 15, 18, 19, 20
- **régularisation L1 (Lasso)** — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s278] · exercices : 9.E3
- **régularisation L2 (Ridge)** — enseignée : [ch. 9, La pénalité L2 : Ridge][s277] · exercices : 9.E3, 10.R1, 19.6, 20.17, CP4.12, B1.6 · réutilisée : ch. 19, 20
- *regularization* → voir [régularisation](#r)
- *regularization path* → voir [chemin de régularisation](#c)
- *regularization strength* → voir [force de régularisation ($\lambda$, `alpha`, `C`)](#f)
- **REINFORCE** — enseignée : [ch. B8, REINFORCE][s688] · exercices : B8.3, B8.7, B8.16–17, B8.Q6, B8.Q9
- **REINFORCE avec baseline** — enseignée : [ch. B8, 208.3 · Gradient de politique : politique paramétrée, REINFORCE,…][s473] · exercices : B8.3, B8.7, B8.16–17, B8.E1, B8.Q5–Q6, B8.Q9, B8.R3
- *reinforcement* → voir [renforcement, punition](#r)
- *reinforcement learning* → voir [apprentissage par renforcement](#a)
- *reinforcement learning from human feedback* → voir [alignement, RLHF](#a), [RLHF](#r)
- *relative / absolute path* → voir [chemin relatif / absolu](#c)
- *relative contrast* → voir [contraste des distances](#c)
- *relative entropy* → voir [divergence KL, divergence de Kullback-Leibler](#d)
- *relative error* → voir [erreur relative (d'un gradient)](#e)
- *reliability diagram* → voir [diagramme de fiabilité](#d)
- **ReLU bruitée** (*noisy ReLU*) — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s235] · exercices : 17.Q7
- ReLU décalée → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- **ReLU, leaky ReLU, PReLU, ReLU décalée** (*parametric ReLU, shifted ReLU*) — enseignée : [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s235] · exercices : 16.24, 17.3, 17.5, 17.12, 17.23, 17.E1, 17.Q6–Q7, 17.Q9, 17.R1 (+ 19) · réutilisée : ch. 16, 21, 23, 25, 27
- *remplissage de zéros* → voir [zero-padding](#z)
- **renforcement, punition** (*reinforcement, punishment*) — enseignée : [ch. 11, 11.6.1 · Sherlock Holmes, « maître de la déduction »][s296] · exercices : 1.E3, 1.Q9, 1.Q11, 10.11, 11.7, 25.Q4, 26.E5, 29.R2, CP6.7
- *reparameterization trick* → voir [astuce de reparamétrisation](#a)
- *repeated k-fold* → voir [k-fold répétée](#k)
- *replay buffer* → voir [mémoire de rejeu](#m)
- **repli sur les octets** (*byte fallback*) — enseignée : [ch. B2, SentencePiece : sans pré-tokenisation][s530]
- **repondération** (*reweighing*) — enseignée : [ch. B6][s382] · exercices : 14.6, B6.27, B6.29
- *repository* → voir [dépôt](#d)
- *representation* → voir [représentation](#r)
- **représentation** (*representation*) — enseignée : [ch. 11, 11.2.1 · La représentation][s291] · exercices : 7.31, 11.1, 11.12, 11.18, 11.Q1–Q2, 12.R1, 15.8, 18.R3, 28.R2 (+ 2)
- *representational power* → voir [capacité](#c), [puissance de représentation](#p)
- **reproductibilité** — enseignée : [ch. B7, 207.4 · Reproductibilité : graines, déterminisme, versions figées,…][s664] · exercices : B7.E2
- **reproductibilité (graine)** — enseignée : [ch. 23, Au-delà du livre : la reproductibilité][s735] ; [ch. 2][s178] · exercices : 23.23, 23.29
- **reproductibilité d'un pipeline LLM** — enseignée : [ch. B4, 204.9 · Classer des textes : encodeur fine-tuné, LLM en zéro-shot…][s782]
- **reproductibilité en RL (graines multiples)** — enseignée : [ch. 26, Au-delà du livre : le RL et les modèles de langage][s471]
- **requête, clé, valeur** (*query, key, value*) — enseignée : [ch. B3, 203.2 · Attention par produit scalaire : requêtes, clés, valeurs,…][s554] · exercices : B3.1, B3.6, B3.13, B3.E1, B3.Q2–Q3, B4.2, B4.5, B7.5, B7.19 (+ 37)
- réseau → voir [réseau antagoniste génératif (GAN)](#r), [réseau cible](#r), [réseau de neurones](#r), [réseau feed-forward, MLP](#r), [réseau impulsionnel](#r), [réseau récurrent (RNN)](#r)
- **réseau antagoniste génératif (GAN)** (*generative adversarial network*) — enseignée : [ch. 27, 27.4 · Implémenter un GAN][s474] · exercices : 27.8, 27.12, 27.14, 27.16, 27.25, 27.27, 27.E1, 27.E3, 27.E5, CP6.7 (+ 47)
- **réseau cible** (*target network*) — enseignée : [ch. B8, 208.2 · DQN : réseau Q, mémoire de rejeu, réseau cible, loss de…][s280] ; [ch. 26, 26.9 · Le rejeu d'expérience][s469] · exercices : 26.12, 26.E4, B8.1, B8.13, B8.15, B8.Q2–Q4, B8.Q9
- **réseau de neurones** (*neural network*) — enseignée : [ch. 1, 1.2.4 · Modèle, paramètres, capacité, hyperparamètres][s47] · exercices : 0B.E1, 1.23, 1.Q5, 1.Q10, 5.Q1, 6.R1, 10.E4, 10.Q1, 10.Q4, 14.E3 (+ 10) · réutilisée : ch. 26
- **réseau feed-forward, MLP** (*feed-forward network, multilayer perceptron*) — enseignée : [ch. 16, 16.1 · Pourquoi ce chapitre ?][s337] · exercices : 16.1, 16.8–9, 16.E3, 17.11, 18.29, 18.R3, 19.9, 20.31, CP4.6 (+ 101) · réutilisée : ch. 23
- **réseau impulsionnel** (*spiking neural network, SNN*) — enseignée : [ch. 10, 10.2 · Les vrais neurones][s51]
- **réseau récurrent (RNN)** (*recurrent neural network*) — enseignée : [ch. 22, 22.4 · La structure d'une cellule RNN][s410] · exercices : 16.Q4, 22.2–3, 22.9, 22.19, 22.22, 22.E1, 22.Q11–Q12, 24.Q10 (+ 50) · approfondie : ch. 20 · réutilisée : ch. B3
- **résidu** (*residual*) — enseignée : [ch. 9, 9.2 · Overfitting et underfitting][s33] · exercices : 9.2–3, 9.23, CP2.1, CP2.6, 14.4, 14.7, 14.25, 14.Q11, CP3.5
- **résidu de Bellman** (*Bellman residual*) — enseignée : [ch. B8][s702]
- *residual* → voir [résidu](#r)
- *residual connection* → voir [connexion résiduelle](#c)
- retour → voir [retour actualisé](#r), [retour actualisé et facteur γ](#r), [retour arrière](#r)
- **retour actualisé** (*discounted return*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s154] · exercices : 26.1, 26.7, 26.14–15, 26.Q6, CP6.1
- **retour actualisé et facteur γ** — enseignée : [ch. 26, 26.3.7 · Les récompenses][s154]
- **retour arrière** (*rollback*) — enseignée : [ch. B7, 207.10 · Surveiller en production : dérive des données et des…][s270]
- retour total → voir [retour, retour total (TFR)](#r)
- **retour, retour total (TFR)** (*return, total future reward*) — enseignée : [ch. 26, 26.3.7 · Les récompenses][s154] · exercices : 1.Q9, 12.4, CP4.12, CP5.10, 26.1, 26.14, 26.Q6, 27.1, CP6.13, B8.16 (+ 37) · réutilisée : ch. B8
- *retracted dataset* → voir [dataset retiré](#d)
- *rétrécissement* → voir [shrinkage](#s)
- *retrieval-augmented generation* → voir [RAG](#r)
- **rétroaction** (*feedback*) — enseignée : [ch. 16, 16.2 · Les graphes de réseaux de neurones][s338] · exercices : 11.7, 16.Q3, B6.Q10, B7.8
- **rétropropagation** (*backpropagation, backprop*) — enseignée : [ch. 18, 18.9 · La rétropropagation en action][s55] · exercices : 0B.E4, 18.4, 18.9–10, 18.E1–E2, 18.Q1, 18.Q9, CP4.2, CP4.10 (+ 64) · approfondie : ch. 21 · réutilisée : ch. 19, 20, 21, 22, 25, B2, B8
- **rétropropagation d'une convolution** — enseignée : [ch. 21, Au-delà du livre : la rétropropagation d'une convolution][s755] · exercices : 21.7, 21.31
- **rétropropagation dans le temps (BPTT)** (*backpropagation through time*) — enseignée : [ch. 22, 22.6 · Entraîner un RNN][s353] · exercices : 22.Q6, 22.Q11, B3.R2
- **rétropropagation tronquée dans le temps** (*truncated BPTT*) — enseignée : [ch. 24, 24.5.9 · Les RNN « stateful »][s442]
- *return* → voir [retour, retour total (TFR)](#r)
- *reverse process* → voir [processus inverse](#p)
- *reverse-mode automatic differentiation* → voir [différentiation automatique en mode inverse](#d)
- *revision* → voir [révision (d'un dataset)](#r)
- **révision (d'un dataset)** (*revision*) — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s161] · exercices : 2.7, 29.10, 29.15, B1.11, B1.24, B2.13, B2.28, B4.4, B4.29, PF.2 (+ 2)
- **revue de code** (*code review*) — enseignée : [ch. B8, Déboguer un agent qui n'apprend pas][s703] · exercices : B8.27
- *reward* → voir [agent, environnement, action, récompense](#a)
- *reward hacking* → voir [piratage de la récompense](#p)
- *reward model* → voir [modèle de récompense](#m)
- *reward model overoptimization* → voir [sur-optimisation du modèle de récompense](#s)
- *reward shaping* → voir [récompense façonnée](#r)
- *reweighing* → voir [repondération](#r)
- **RFE** (*recursive feature elimination*) — enseignée : [ch. 12, 12.6 · La sélection de features][s310]
- **RGPD** (*GDPR (règlement 2016/679), UE*) — enseignée : [ch. B6, Le RGPD, article 22][s385] · évoquée : [ch. 20, 20.8.1 · Une explication satisfaisante][s383] · exercices : 1.7, 14.9, 20.11, 29.Q2, B1.9, B3.10, B4.9, B6.8–10 (+ 2)
- **RGPD et AI Act appliqués** — enseignée : [ch. B6, 206.11 · Éthique et régulation en pratique : fiches modèles,…][s787]
- **Ridge (pénalité L2)** (*ridge regression, L2 penalty, Tikhonov regularization*) — enseignée : [ch. 9, La pénalité L2 : Ridge][s277] · exercices : 9.3, 9.6, 9.17, 9.22, 9.28, 9.Q7, CP2.6, 12.R2, 15.9, 15.22 (+ 17) · réutilisée : ch. 12, 15, 19, 20
- *ridge regression* → voir [Ridge (pénalité L2)](#r)
- *RL* → voir [amorçage](#a)
- RL → voir [fonction de valeur (RL)](#f)
- **RL à récompenses vérifiables (RLVR)** (*RL with verifiable rewards*) — enseignée : [ch. B8, GRPO et les récompenses vérifiables][s699]
- **RL hors ligne** (*offline RL*) — enseignée : [ch. B8, Sécurité][s701]
- *RL with verifiable rewards* → voir [RL à récompenses vérifiables (RLVR)](#r)
- **RLHF** (*reinforcement learning from human feedback*) — enseignée : [ch. B8, Le RLHF : PPO sous une laisse KL][s164] ; [ch. 26, Au-delà du livre : le RL et les modèles de langage][s471] · exercices : 26.11, 26.E5, B8.10, B8.20, B8.26, B8.E4, B8.Q11–Q12
- RLVR → voir [RL à récompenses vérifiables (RLVR)](#r)
- RMSE → voir [MSE, RMSE, MAE](#m)
- **RMSNorm** (*root mean square normalization*) — enseignée : [ch. 20, 20.4.4 · La batchnorm][s286]
- **RMSprop** — enseignée : [ch. 19, 19.5.4 · Adadelta et RMSprop][s366] · exercices : 19.4, 19.8, 19.13, 19.17, 19.Q1, 19.Q11, 19.R3, 22.25, 22.R2 · réutilisée : ch. 22
- RNN → voir [réseau récurrent (RNN)](#r)
- **RNN profond, bidirectionnel** (*deep RNN, bidirectional RNN*) — enseignée : [ch. 22, 22.8.2 · Le RNN profond][s417] · exercices : 22.22, 22.Q11
- *robust* → voir [robuste](#r)
- *robust scaling* → voir [mise à l'échelle robuste](#m)
- **robuste** (*robust*) — enseignée : [ch. 2, Résumer une liste de nombres : moyenne, médiane, mode][s175] · exercices : 9.10, 9.Q2, 12.19, B8.Q9
- *ROC* → voir [AUC](#a)
- *ROC curve* → voir [courbe ROC](#c)
- **rôle de la non-linéarité** — enseignée : [ch. 17][s746]
- *rollback* → voir [retour arrière](#r)
- *rolling origin* → voir [validation glissante](#v)
- **rollout** — enseignée : [ch. B8, Réutiliser un rollout sans tout casser][s692] · exercices : B8.4, B8.8, B8.18, B8.20–21, B8.Q7–Q9
- *root* → voir [racine, feuille, nœud interne](#r)
- *root mean square normalization* → voir [RMSNorm](#r)
- RoPE → voir [encodage de position rotatif (RoPE)](#e)
- *Rosenbrock function* → voir [fonction de Rosenbrock](#f)
- *rotary position embedding* → voir [encodage de position rotatif (RoPE)](#e)
- *round-off error* → voir [erreur d'arrondi](#e)
- route → voir [API HTTP, route](#a)
- *row-major order* → voir [ordre C, ligne par ligne](#o)
- RRF → voir [reciprocal rank fusion (RRF)](#r)
- **rsLoRA** (*rank-stabilized LoRA*) — enseignée : [ch. B4, LoRA][s173] · exercices : B4.6
- ruff → voir [uv, pip-tools, ruff, pre-commit](#u)
- *rule of succession* → voir [règle de succession de Laplace](#r)
- *rule of three* → voir [règle de trois](#r)
- **run** (*exécution*) — enseignée : [ch. B7, 207.5 · Suivi d'expériences : runs, métriques, artefacts, comparaison][s437] ; [ch. 24, 24.2.2 · Changer un seul hyperparamètre][s438] · exercices : 24.11, 24.13, 24.15, 24.26, 24.R1, B7.10, B7.12, B7.14, B7.25, B7.R1 (+ 23)
- *run ID* → voir [identifiant de run](#i)
- *running statistics* → voir [statistiques glissantes](#s)
- **R²** — enseignée : [ch. 9][s281] · exercices : 8.16, 9.1, 13.26, 14.15, 14.25, B6.12, B6.Q2 · réutilisée : ch. 8

## S

- *sa racine* → voir [MSE, RMSE, MAE](#m)
- **sac de mots** (*bag of words*) — enseignée : [ch. 13, 13.6 · Naïve Bayes][s140] ; [ch. 0B][s43] · exercices : 0B.41, 13.30
- *saddle point* → voir [point selle, col](#p)
- **safetensors** — enseignée : [ch. 23, 23.9.2 · Les poids seuls][s434] · exercices : B1.27, B4.30, B6.20, B7.1, B7.9–10, B7.17, B7.21, B7.24, B7.Q8 (+ 1)
- *saliency map* → voir [carte de saillance](#c)
- **SAMME** (*stagewise additive modeling using a multi-class exponential loss*) — enseignée : [ch. 14, 14.7 · Le boosting][s316] · exercices : 14.6, 14.23–24, 14.Q9, 15.16, CP3.5
- *sample* → voir [échantillon](#e), [tirage](#t), [population, échantillon (en statistique)](#p)
- *sample density* → voir [densité d'échantillons](#d)
- *sample efficiency* → voir [efficacité en échantillons](#e)
- *sample-and-hold* → voir [échantillonneur-bloqueur](#e)
- *sampler* → voir [échantillonneur](#e)
- *samplewise* → voir [par échantillon, par feature, par élément](#p)
- **samplewise/featurewise/elementwise** — enseignée : [ch. 12, 12.9 · Par échantillon, par feature, par élément][s309]
- *sampling bias* → voir [biais d'échantillonnage](#b)
- *sampling with replacement* → voir [tirage avec remise, sans remise](#t)
- *sanity check* → voir [test de santé](#t)
- sans remise → voir [tirage avec remise, sans remise](#t)
- **SARSA** — enseignée : [ch. 26, 26.7 · SARSA][s466] · exercices : 26.4, 26.20–21, 26.28–30, 26.E1, 26.Q11, CP6.3, B8.Q2 (+ 1)
- **SARSA espéré** (*expected SARSA*) — enseignée : [ch. 26, 26.7 · SARSA][s466]
- **saturation** — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s110] ; [ch. 18, 18.12.4 · Garder les neurones en vie][s346] ; [ch. B6, Le problème de la saturation][s347] · exercices : 16.10, 16.22, 17.21, 17.23, 17.Q8, 18.24, 18.Q10, 22.6, 22.19, 27.18 (+ 1)
- **saturation et neurones morts** — enseignée : [ch. 17, 17.2 · Ce que fait une fonction d'activation][s52] ; [ch. 18, 18.12.4 · Garder les neurones en vie][s346]
- **sauvegarde et reproductibilité** — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s3]
- *scaled dot-product attention* → voir [attention par produit scalaire mis à l'échelle](#a)
- **scaler** (*objet de mise à l'échelle*) — enseignée : [ch. 12, 12.5.3 · Retenir la transformation][s307] · exercices : 12.4, 12.9, 12.19–20, 12.31, 12.Q1, 15.R2, 22.R3, 24.25, 24.Q10 (+ 6) · approfondie : ch. 24
- *scaling laws* → voir [lois d'échelle](#l)
- *scatter plot* → voir [nuage de points](#n)
- **schedulers de learning rate** — enseignée : [ch. 23, 23.10.2 · Le learning rate][s12]
- *scope* → voir [portée](#p)
- **score (d'une densité)** — enseignée : [ch. B5, La guidance sans classifieur][s623]
- **score de Brier** (*Brier score*) — enseignée : [ch. 3, Au-delà du livre (3) : la calibration des probabilités][s214] · exercices : 3.28, CP1.1
- *score function* → voir [fonction score](#f)
- **score, seuil de décision** (*decision threshold*) — enseignée : [ch. 15, Au-delà du livre : seuil de décision, calibration et classes…][s211] ; [ch. 3, 3.7.11 · Le score F1][s208] · exercices : 3.11, 3.18, 3.20, 3.24, 3.26–28, 15.18, 15.25, 15.27 (+ 22)
- *sealed test* → voir [test scellé](#t)
- *search space* → voir [espace de recherche](#e)
- *secant* → voir [sécante](#s)
- **sécante** (*secant*) — enseignée : [ch. 5, Tangente et dérivée][s226] · exercices : 0B.10, 5.1
- *second derivative* → voir [dérivée seconde, courbure](#d)
- **secret** — enseignée : [ch. B7, 207.9 · Conteneurs et déploiement : Dockerfile, variables…][s673] · exercices : 0A.10, B4.Q2, B6.9, B7.23–24, PF.12
- *seed* → voir [graine](#g), [amorce](#a)
- **segmentation la plus courte** (*shortest tokenization*) — enseignée : [ch. B2][s538]
- *selection bias* → voir [biais d'échantillonnage](#b), [biais de sélection](#b)
- **sélection de features** (*feature selection*) — enseignée : [ch. 12, 12.6 · La sélection de features][s310] · exercices : 8.25, 12.10, 12.25, 14.19, 15.24, 15.R2
- *selection rate* → voir [taux de sélection](#t)
- *self-attention* → voir [auto-attention](#a)
- *self-consistency* → voir [auto-cohérence](#a)
- *self-information* → voir [information (de Shannon), surprise](#i)
- **self-play** — enseignée : [ch. 26, 26.10 · Deux applications][s470] · exercices : 26.Q12, 27.R1
- *self-supervised learning* → voir [apprentissage auto-supervisé](#a)
- *semantic search* → voir [recherche sémantique](#r)
- **semi-gradient** — enseignée : [ch. B8, Apprendre une valeur par régression][s683]
- *semi-supervised learning* → voir [apprentissage semi-supervisé](#a)
- **sensibilité, invariance à l'implémentation** (*sensitivity, implementation invariance*) — enseignée : [ch. B6, Les gradients intégrés][s640] · exercices : 3.7, 3.9–10, 3.17, 3.21, 4.4, 4.8, 4.20, 4.R1, B4.16 (+ 2)
- *sensitive attribute* → voir [attribut sensible](#a)
- *sensitivity* → voir [sensibilité, invariance à l'implémentation](#s)
- *sensitivity analysis* → voir [analyse de sensibilité](#a)
- *sentence embedding* → voir [embedding de phrase](#e)
- **SentencePiece** — enseignée : [ch. B2, SentencePiece : sans pré-tokenisation][s530] · exercices : B2.Q7
- *sentinel value* → voir [valeur sentinelle](#v)
- **séparabilité linéaire** (*linear separability*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s253]
- *seq2seq* → voir [encodeur-décodeur](#e)
- *sequence* → voir [séquence](#s)
- **séquence** (*sequence*) — enseignée : [ch. 20, 20.4.7 · Les couches récurrentes (aperçu)][s406] ; [ch. 22, 22.5 · Organiser les entrées][s407] · exercices : 20.14, 20.20, 22.2, 22.15, 22.22, 22.R1, 24.27, 24.30, 25.R2, B2.Q1 (+ 33)
- *sequential test* → voir [test séquentiel, règle d'arrêt](#t)
- *sequential updating* → voir [mise à jour bayésienne, séquentielle](#m)
- séquentielle → voir [mise à jour bayésienne, séquentielle](#m)
- **sérialiser** (*serialize*) — enseignée : [ch. 0A, 100.6.4 · Sérialiser : `json` et `pickle`][s88] · exercices : 0A.39, 0A.Q8
- *serialize* → voir [sérialiser](#s)
- **série temporelle** (*time series*) — enseignée : [ch. 8, Données dépendantes : groupes et séries temporelles (au-delà du livre)][s268] · exercices : 1.12, 8.19, 22.1, 22.21, 22.E4, 22.Q12, 22.R3, PF.3 · approfondie : ch. 22
- Series → voir [DataFrame, Series](#d)
- *service level objective* → voir [SLO](#s)
- *set* → voir [ensemble](#e)
- **set_output** — enseignée : [ch. 15, Au-delà du livre : les données hétérogènes][s334] · exercices : 15.8
- **seuil d'interpolation** (*interpolation threshold*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s285] · exercices : 9.11, 9.30
- seuil de décision → voir [score, seuil de décision](#s)
- **seuillage doux** (*soft thresholding*) — enseignée : [ch. 9, La pénalité L1 : Lasso et les zéros exacts][s278] · exercices : 9.6, 9.23
- **seuils par groupe** (*group-specific thresholds*) — enseignée : [ch. B6, Des taux par groupe][s650] · exercices : B6.5, B6.27
- **SFT** (*supervised fine-tuning, fine-tuning supervisé*) — enseignée : [ch. B4, Trois étapes, trois sortes de modèles][s580]
- SGD → voir [descente de gradient stochastique (SGD)](#d)
- *Sh* → voir [bit (d'information)](#b)
- *shadow deployment* → voir [déploiement canari, en ombre](#d)
- *Shannon* → voir [entropie (de Shannon)](#e)
- *shannon* → voir [bit (d'information)](#b)
- SHAP → voir [graphique de dépendance (SHAP)](#g)
- *shape* → voir [forme](#f), [forme (d'un tenseur)](#f)
- *SHapley Additive exPlanations* → voir [valeurs SHAP](#v)
- **Shapley interventionnel, observationnel** (*interventional, observational Shapley*) — enseignée : [ch. B6, Les features comme joueurs][s634] · exercices : B6.2, B6.Q6
- *Shapley kernel* → voir [noyau de Shapley](#n)
- *Shapley value* → voir [valeur de Shapley](#v)
- *shared layer* → voir [couche partagée](#c)
- *shell* → voir [terminal](#t)
- *shell form* → voir [forme exec, forme shell](#f)
- *shifted ReLU* → voir [ReLU, leaky ReLU, PReLU, ReLU décalée](#r)
- *shortcut learning* → voir [raccourci appris](#r)
- *shortest tokenization* → voir [segmentation la plus courte](#s)
- **shrinkage** (*rétrécissement*) — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s195]
- *siamese network* → voir [siamois (réseau)](#s)
- **siamois (réseau)** (*siamese network*) — enseignée : [ch. B2, D'une suite de vecteurs à un vecteur de phrase][s544] · exercices : 24.28
- *sigmoid* → voir [sigmoïde](#s)
- *sigmoid linear unit* → voir [SiLU, swish](#s)
- **sigmoïde** (*sigmoid, logistic function*) — enseignée : [ch. 0B, 101.2.5 · Sigmoïde et tangente hyperbolique : définitions et allure][s108] ; [ch. 13, Au-delà du livre : la régression logistique, le classifieur…][s109] · exercices : 0B.17, 0B.32, 17.4, 17.13, 17.23, 17.E1, 17.E3, 18.5, 18.R1, CP4.5 (+ 39) · approfondie : ch. 17 · réutilisée : ch. 15, 17, 22, 25, 27, B2
- **sigmoïde et tanh : définitions, allure, symétries** — enseignée : [ch. 0B, 101.2.5 · Sigmoïde et tangente hyperbolique : définitions et allure][s108] · exercices : 0B.17, 0B.36, 0B.Q6
- *sign function* → voir [marche unité, marche de Heaviside, fonction signe](#m)
- *signal-to-noise ratio* → voir [rapport signal sur bruit (SNR)](#r)
- **silhouette** — enseignée : [ch. 15, 15.5 · Le clustering][s743] · exercices : 7.28–29, 8.R1, 15.13 · approfondie : ch. 7
- *silhouette coefficient* → voir [coefficient de silhouette](#c)
- **SiLU, swish** (*sigmoid linear unit*) — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s110] · exercices : 17.2, 17.4, 17.10, 17.14, 17.18, 17.E4, 17.Q9–Q10, B5.19, B5.21
- sim-to-real → voir [écart simulation-réalité, sim-to-real](#e)
- **similarité cosinus** (*cosine similarity*) — enseignée : [ch. 0B, 101.3.3 · Produit scalaire, angle, similarité cosinus][s115] ; [ch. B2, Le cosinus][s116] · exercices : 0B.18–19, 0B.39–41, 0B.E2, 0B.Q8, 28.2, B1.2, B2.2 (+ 2) · réutilisée : ch. B1
- **similarité cosinus et analogies** — enseignée : [ch. B2, 202.9 · Géométrie des embeddings : cosinus, voisins, analogies,…][s776]
- *simple probability* → voir [probabilité simple, marginale](#p)
- *simplified objective* → voir [loss simplifiée](#l)
- *Simpson's paradox* → voir [paradoxe de Simpson](#p)
- **simulation et loi des grands nombres (intuition)** — enseignée : [ch. 0B, 101.7.5 · Simuler pour estimer : fréquences et loi des grands nombres][s136]
- *single-valued curve* → voir [courbe continue, lisse, univoque](#c)
- *singular value decomposition* → voir [SVD](#s)
- *skip connection* → voir [connexion de saut](#c)
- **skip-gram, CBOW** (*continuous bag of words*) — enseignée : [ch. B2, word2vec : prédire plutôt que compter][s540] · exercices : B2.7, B2.25, B2.Q10
- *sliding window* → voir [fenêtre glissante](#f)
- *sliding window attention* → voir [fenêtre glissante (attention)](#f)
- **SLO** (*service level objective*) — enseignée : [ch. B7, Latence, percentiles et débit][s516] · exercices : B7.19
- *slope-intercept diagram* → voir [diagramme pente-ordonnée](#d)
- *slothful induction* → voir [induction paresseuse](#i)
- *smoke test* → voir [entraînement fumée](#e)
- *smooth* → voir [courbe continue, lisse, univoque](#c)
- **SmoothGrad** — enseignée : [ch. B6, Le choix de la ligne de référence][s641]
- *SNN* → voir [réseau impulsionnel](#r)
- SNR → voir [rapport signal sur bruit (SNR)](#r)
- **SNR final nul** (*zero terminal SNR*) — enseignée : [ch. B5, Le rapport signal sur bruit][s608]
- *soft margin* → voir [marge souple, paramètre C](#m)
- *soft thresholding* → voir [seuillage doux](#s)
- *soft update* → voir [mise à jour douce (de Polyak)](#m)
- *soft voting* → voir [vote dur, vote souple](#v)
- **softmax** — enseignée : [ch. 17, 17.8 · La softmax][s250] · évoquée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] · exercices : 17.6–7, 17.15–16, 17.19, 17.E3, 17.Q11, 18.8, 18.14, 18.E5 (+ 47) · approfondie : ch. 18, 26 · réutilisée : ch. 18, 20, 21, 22, 23, 24, 26, B2, B3, B8
- **softmax en ligne** (*online softmax*) — enseignée : [ch. B3, FlashAttention : le même résultat, sans la matrice][s572]
- *softmax Jacobian* → voir [Jacobienne de la softmax](#j)
- *softmax policy* → voir [politique softmax, de Boltzmann](#p)
- **softplus** — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s110] · exercices : 17.2, 17.4, 17.13, 17.22, 17.Q10
- **solide** (*sound*) — enseignée : [ch. 11, 11.4 · La déduction][s297] · exercices : 11.3, 11.Q7, CP2.9, B6.Q6, B8.11
- **solution de norme minimale, pseudo-inverse** (*minimum-norm solution, Moore-Penrose pseudo-inverse*) — enseignée : [ch. 9, 9.6.4 · Le compromis biais-variance, et ses limites][s285] · exercices : 9.11, 9.16
- **somme de contrôle** (*checksum, hash*) — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s161] · exercices : 29.15
- **somme pondérée, pré-activation** (*weighted sum, pre-activation*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s253] · exercices : 0B.4, 0B.32, 1.3, 10.1, 10.5, 10.Q6, 10.Q8, 16.2–3, 16.7 (+ 9) · approfondie : ch. 0B · réutilisée : ch. 16, 21
- *sonde linéaire* → voir [linear probe](#l)
- **sophisme** (*fallacy*) — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s298] · exercices : 11.4, 11.16–18, 11.Q8, 11.Q10, CP2.9, 14.R2
- **sophismes inductifs** — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s299] · exercices : 11.Q10
- **sophismes syllogistiques** — enseignée : [ch. 11, 11.4.1 · Les sophismes syllogistiques][s298] · exercices : 11.4, 11.15–16, 11.Q8
- **sortie structurée** (*structured output*) — enseignée : [ch. B4, La sortie structurée][s586] · exercices : B4.15
- *sorties d'une couche* → voir [fan-in, fan-out](#f)
- **sorties de nn.LSTM et état porté** — enseignée : [ch. 24, 24.5.6 · Les RNN profonds][s766]
- **souche** (*decision stump*) — enseignée : [ch. 14, 14.7 · Le boosting][s316] · évoquée : [ch. 13][s315] · exercices : 13.3, 14.6–7, 14.22–24, 14.Q5, 14.Q8, 14.Q10, 15.16 (+ 2)
- *sound* → voir [solide](#s)
- **sources de datasets (HF Hub, Kaggle, OpenML, UCI)** — enseignée : [ch. 29, 29.4 · Les collections de datasets][s498] · exercices : 29.Q7
- **sous-échantillonnage des mots fréquents** (*subsampling of frequent words*) — enseignée : [ch. B2, word2vec : prédire plutôt que compter][s540]
- **sous-gradient** (*subgradient*) — enseignée : [ch. 5, 5.2 · Introduction : fonctions, courbes et surfaces][s224] ; [ch. 17, 17.5 · Les fonctions linéaires par morceaux][s235] · réutilisée : ch. 17
- **sous-séquence de pas** (*timestep spacing*) — enseignée : [ch. B5, L'idée de DDIM][s616]
- *span corruption* → voir [corruption de segments](#c)
- *sparse* → voir [parcimonieux](#p)
- *sparse autoencoder* → voir [autoencodeur parcimonieux](#a)
- *Spearman rank correlation* → voir [corrélation de rangs (de Spearman)](#c)
- *special pleading* → voir [plaidoyer spécial](#p)
- *special token* → voir [token spécial](#t)
- *specification gaming* → voir [piratage de la récompense](#p)
- **spécificité** (*specificity, true negative rate, TNR*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s205] · exercices : 3.5, 3.7, 3.10, 3.21, 3.Q11, 4.4, 4.8, 4.20, CP1.6
- *specificity* → voir [spécificité](#s)
- *speculative decoding* → voir [décodage spéculatif](#d)
- *spherical linear interpolation* → voir [interpolation sphérique (slerp)](#i)
- *spiking neural network* → voir [réseau impulsionnel](#r)
- **split** (*découpe*) — enseignée : [ch. 13, 13.5.2 · Choisir les splits][s147] · exercices : 0A.15, 0A.38, 8.15, 9.28, 13.3, 13.20, 13.22, 13.E2, 13.Q9, 14.R1 (+ 7)
- **SQL** (*structured query language*) — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s313] · exercices : 12.15
- **SQL de base (sqlite3, pd.read_sql)** — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s313]
- *squelette* → voir [stub](#s)
- *src layout* → voir [layout `src/`](#l)
- SSM → voir [modèle à espace d'états (SSM)](#m)
- **stabilité numérique (log-sum-exp)** — enseignée : [ch. 17][s746] · exercices : 23.Q1
- **stacking, méta-modèle** (*meta-learner*) — enseignée : [ch. 14, 14.3 · Le vote][s323] · exercices : 14.14
- *stagewise additive modeling using a multi-class exponential loss* → voir [SAMME](#s)
- *standard deviation* → voir [variance, écart-type](#v)
- *standard error* → voir [erreur type](#e)
- **standardisation (centrage, réduction)** (*standardization*) — enseignée : [ch. 12, 12.5.2 · La standardisation][s306] · exercices : 1.21, 12.2, 12.6, 12.24, 12.29, 12.31, 12.E2, 12.Q5, CP3.4, CP4.12 (+ 19) · réutilisée : ch. 13, 20
- standardiser → voir [z-score, standardiser](#z)
- *standardization* → voir [standardisation (centrage, réduction)](#s)
- *standardize* → voir [z-score, standardiser](#z)
- *state* → voir [état, transition](#e)
- *state dictionary* → voir [`state_dict`](#s)
- *state space model* → voir [modèle à espace d'états (SSM)](#m)
- **state_dict et checkpoints** — enseignée : [ch. 23, 23.10.1 · Checkpoints][s380]
- **`state_dict`** (*state dictionary*) — enseignée : [ch. 20, Au-delà du livre : PyTorch en pratique][s3] · exercices : 20.17, 20.27–28, 23.23, 23.25, 23.Q4, 23.Q11, CP5.6, 25.35, B7.R2 (+ 13) · approfondie : ch. 23 · réutilisée : ch. 25, 27, B7
- *stateful RNN* → voir [état porté](#e)
- *static* → voir [embedding statique, contextuel](#e)
- *static quantization* → voir [quantification dynamique, statique](#q)
- *stationary inflection point* → voir [point d'inflexion à tangente horizontale](#p)
- *stationary point* → voir [point critique, point stationnaire](#p)
- statique → voir [quantification dynamique, statique](#q)
- *statistical parity* → voir [parité démographique](#p)
- *statistical syllogism* → voir [syllogisme statistique](#s)
- **statistiques descriptives** — enseignée : [ch. 2, 2.1 · Pourquoi des statistiques ?][s732]
- **statistiques glissantes** (*running statistics*) — enseignée : [ch. 20, 20.4.4 · La batchnorm][s286] · exercices : 20.4, 20.8, 20.22, 20.Q6, 24.28, B1.4, B1.17–18, B1.Q7 · réutilisée : ch. B1
- *status code* → voir [code de statut](#c)
- *STE* → voir [estimateur \*straight-through\*](#e)
- *stemming* → voir [racinisation](#r)
- *step* → voir [workflow, job, étape, action](#w)
- *step decay* → voir [planning par paliers](#p)
- *step distillation* → voir [distillation (de l'échantillonneur)](#d)
- *stochastic gradient boosting* → voir [gradient boosting stochastique](#g)
- *stochastic gradient descent* → voir [descente de gradient stochastique (SGD)](#d)
- *stop string* → voir [chaîne d'arrêt](#c)
- *stopping rule* → voir [test séquentiel, règle d'arrêt](#t)
- *storage limitation* → voir [minimisation, limitation de la conservation](#m)
- *straight-through estimator* → voir [estimateur \*straight-through\*](#e)
- **stratégies de décodage (glouton, température, top-k, top-p)** — enseignée : [ch. 28][s399] · exercices : 28.28, 28.E3
- **stratification** — enseignée : [ch. 8, 8.3 · Le jeu de test][s143] · exercices : 8.11, 8.13, 8.15, 15.11, 15.17, 23.15, 29.2 · réutilisée : ch. 15, 23, 24, 29
- *stratified bootstrap* → voir [bootstrap stratifié (par groupe)](#b), [bootstrap stratifié (de runs)](#b)
- **StratifiedKFold** — enseignée : [ch. 15, 15.9.1 · La validation croisée][s744] · exercices : 8.13, 8.21, 15.17, 15.24–25, 15.R3, CP3.11, 24.17–18, PF.5
- **streaming** (*diffusion au fil de l'eau*) — enseignée : [ch. B4, Le streaming][s585] · exercices : B4.7, B4.14
- **stride** (*pas*) — enseignée : [ch. 21, 21.3.5 · Le stride][s29] · évoquée : [ch. 20, 20.4.6 · Le pooling (aperçu)][s26] · exercices : 21.3, 21.12, 21.15, 21.26, 21.E2, 21.Q5, 24.3, 24.Q9, 24.R2, 25.27 (+ 42) · réutilisée : ch. 24, 25, 27
- *string* → voir [chaîne de caractères](#c)
- *strong learner* → voir [apprenant faible, apprenant fort](#a)
- **structure d'un projet Python** — enseignée : [ch. B7, 207.1 · Du notebook au projet : limites du notebook, structure…][s99]
- *structured output* → voir [sortie structurée](#s)
- *structured query language* → voir [SQL](#s)
- **stub** (*squelette*) — enseignée : [ch. 0A, 100.11.5 · mylearn : stubs, référence et tests fondés sur un oracle][s103] · exercices : B3.16 · réutilisée : ch. 0B
- *Student's t distribution* → voir [loi de Student](#l)
- *style loss* → voir [loss de style](#l)
- *style reference* → voir [image de contenu, référence de style, image générée](#i)
- *subgradient* → voir [sous-gradient](#s)
- *subnormal number* → voir [nombre dénormalisé](#n)
- *subsampling of frequent words* → voir [sous-échantillonnage des mots fréquents](#s)
- *substitut* → voir [variable de substitution](#v)
- **substitut local, LIME** (*local surrogate, local interpretable model-agnostic explanations*) — enseignée : [ch. B6, Deux autres explications locales : substitut et contrefactuel][s384]
- **successive halving** (*réduction de moitié successive*) — enseignée : [ch. 15, 15.9.4 · La recherche aléatoire][s330] · évoquée : [mini-projet 2, Les extensions (pour aller plus loin)][s708] · exercices : 15.5, 15.31, 19.11
- **suite géométrique, raison** (*geometric sequence, ratio*) — enseignée : [ch. 0B, 101.1.5 · Suites géométriques et leur somme][s106] · exercices : 0B.Q4 · réutilisée : ch. 22, 26
- **suites géométriques et leur somme (finie et infinie)** — enseignée : [ch. 0B, 101.1.5 · Suites géométriques et leur somme][s106] · exercices : 0B.13–14, 0B.34, 0B.Q4
- **suivi d'expériences** — enseignée : [ch. B7, 207.5 · Suivi d'expériences : runs, métriques, artefacts, comparaison][s437] · exercices : B7.14, PF.7
- *supervised fine-tuning* → voir [SFT](#s)
- *supervised learning* → voir [apprentissage supervisé](#a)
- **supervision humaine** (*human oversight*) — enseignée : [ch. B6, La supervision humaine][s651] · exercices : B6.8–9, B6.28, B6.Q12
- **support (d'une classe)** — enseignée : [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s41] ; [ch. 13, 13.4 · Les machines à vecteurs de support][s210] · exercices : 3.6, 3.25, 13.4, 13.10, 13.16, 13.18, 13.Q5
- **support (few-shot)** (*support set*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s210] ; [ch. B1, Le few-shot et les prototypes de classe][s511] · exercices : 13.4, 13.10, 13.16, 13.18, 13.Q5, B1.25–26
- *support set* → voir [support (few-shot)](#s)
- *support vector* → voir [vecteur de support](#v)
- *support vector machine* → voir [SVM, machine à vecteurs de support](#s)
- *sur politique* → voir [on-policy, off-policy](#o)
- **sur-optimisation du modèle de récompense** (*reward model overoptimization*) — enseignée : [ch. B8, Le piratage de la récompense][s472]
- **surcharge (de configuration)** (*override*) — enseignée : [ch. B7, 207.3 · Configuration : dataclasses, YAML ou JSON, surcharges,…][s661] · exercices : 18.26, B7.8, B7.11 · approfondie : ch. 0A · réutilisée : ch. 18
- **surcoût de tokenisation** (*tokenization premium*) — enseignée : [ch. B2, 202.4 · BPE au niveau octet, pré-tokenisation et tokens spéciaux][s31] · exercices : B2.11
- **suréchantillonnage** (*upsampling*) — enseignée : [ch. 20, 20.4.8 · Les autres couches utilitaires][s374] ; [ch. 25, 25.7 · Autoencodeurs convolutifs][s375] · exercices : 20.Q9, 25.27, 25.Q10, 27.1, 27.21 · réutilisée : ch. B5
- **surestimation** (*overestimation bias*) — enseignée : [ch. B8, La surestimation, et Double DQN][s685] · exercices : B8.15, B8.Q4
- *surprisal* → voir [information (de Shannon), surprise](#i)
- surprise → voir [information (de Shannon), surprise](#i)
- *surrogate objective* → voir [objectif de substitution](#o)
- **surveillance de la dérive** — enseignée : [ch. B7, 207.10 · Surveiller en production : dérive des données et des…][s270]
- **survey des couches (conv, pooling, RNN, utilitaires)** — enseignée : [ch. 20, 20.4 · Un tour d'horizon des couches][s753]
- *survey imputation* → voir [imputation (d'enquête)](#i)
- **SVD** (*décomposition en valeurs singulières, singular value decomposition*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s199] ; [ch. B2, 202.8 · Embeddings distributionnels : co-occurrences, PPMI, SVD,…][s311] · exercices : 12.27, 15.1, B2.24 · approfondie : ch. B2 · réutilisée : ch. B2
- **SVD tronquée** (*truncated SVD*) — enseignée : [ch. B2, Réduire : la SVD tronquée][s539] · exercices : B2.24
- **SVM, machine à vecteurs de support** (*support vector machine*) — enseignée : [ch. 13, 13.4 · Les machines à vecteurs de support][s210] · exercices : 13.4, 13.10–11, 13.16–17, 13.36, 13.E5, 13.Q1, 13.Q5–Q6 (+ 8)
- **SwiGLU** (*unité linéaire à porte avec SiLU*) — enseignée : [ch. 17, 17.6 · Les fonctions lisses][s110] · évoquée : [ch. 16, 16.1 · Pourquoi ce chapitre ?][s337] · exercices : 17.E4
- swish → voir [SiLU, swish](#s)
- *sycophancy* → voir [flagornerie](#f)
- *syllogism* → voir [syllogisme](#s)
- **syllogisme** (*syllogism*) — enseignée : [ch. 11, 11.4 · La déduction][s297] · exercices : 11.3, 11.5–6, 11.15, 11.Q7, 11.Q9, 11.Q11–Q12, CP2.1, CP2.9
- **syllogisme statistique** (*statistical syllogism*) — enseignée : [ch. 11, 11.5 · L'induction][s188] · exercices : 11.6, 11.Q9
- **syllogismes** — enseignée : [ch. 11, 11.4 · La déduction][s297] · exercices : 11.3, 11.5, 11.15, CP2.9
- symétrie → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *symmetry* → voir [efficacité, symétrie, joueur nul, additivité](#e)
- *symmetry breaking* → voir [brisure de symétrie](#b)
- **synapse** — enseignée : [ch. 10, 10.2 · Les vrais neurones][s51] · exercices : 10.9
- **synchrone, asynchrone** (*synchronous, asynchronous*) — enseignée : [ch. 16, 16.3 · Flux synchrone et asynchrone][s339] · exercices : 16.Q3, 16.Q5, 26.26
- *synchronous* → voir [synchrone, asynchrone](#s)
- *synthetic data* → voir [données synthétiques](#d)
- **système expert** (*expert system*) — enseignée : [ch. 1, 1.1.2 · Les systèmes experts et leurs limites][s141] · exercices : 1.15, 1.Q2

## T

- *t-distributed stochastic neighbor embedding* → voir [t-SNE, UMAP](#t)
- **t-SNE, UMAP** (*t-distributed stochastic neighbor embedding, uniform manifold approximation and projection*) — enseignée : [ch. 12, Au-delà du livre : voir des données en grande dimension][s312] · exercices : 12.30, B1.15, B2.27 · réutilisée : ch. B1, B2
- **table de contingence** (*contingency table, cross-tabulation*) — enseignée : [ch. 3, 3.6 · Probabilité marginale et tables de contingence][s203] · exercices : 3.3, 3.14, 3.Q5, 4.15
- **table Q, valeur Q** (*Q-table, Q-value*) — enseignée : [ch. 26, 26.6.1 · Les valeurs Q et leur mise à jour][s462] · exercices : 26.2, 26.9, 26.32, 26.Q4, 26.Q10, 26.Q12, 26.R2, CP6.1, CP6.10, B8.Q1 (+ 3)
- **tableau croisé** (*pivot table*) — enseignée : [ch. 12, Au-delà du livre : assembler les données avec pandas et SQL][s313] · exercices : 7.17, 12.14
- *tableau NumPy* → voir [array](#a)
- *tabular foundation model* → voir [modèle de fondation tabulaire](#m)
- **taille de l'état** (*hidden size*) — enseignée : [ch. 22, 22.4.1 · Une cellule à plus d'état][s412] · exercices : 22.Q4
- **tampon (d'un optimiseur)** (*buffer, optimizer state*) — enseignée : [ch. 19, 19.5 · Les variantes de la descente de gradient][s364] · exercices : 19.9, 19.16–17, 21.6–7, 21.9
- **tangente hyperbolique** (*hyperbolic tangent*) — enseignée : [ch. 0B, 101.2.5 · Sigmoïde et tangente hyperbolique : définitions et allure][s108] ; [ch. 17, 17.6 · Les fonctions lisses][s110] · exercices : 17.2
- *target encoding* → voir [encodage par la cible](#e)
- *target network* → voir [réseau cible](#r)
- *targeted* → voir [attaque ciblée, non ciblée](#a)
- taux → voir [taux d'accroissement](#t), [taux d'erreur](#t), [taux de compression](#t), [taux de fausses découvertes, de fausses omissions](#t), [taux de faux positifs, de faux négatifs](#t), [taux de sélection](#t)
- **taux d'accroissement** (*difference quotient*) — enseignée : [ch. 0B, 101.5.1 · Taux d'accroissement, tangente, dérivée][s120] · exercices : 0B.10, 16.6 · réutilisée : ch. 5
- **taux d'erreur** (*error rate*) — enseignée : [ch. 1, 1.2.3 · Généraliser : le jeu de test][s38] ; [ch. 13, 13.5.2 · Choisir les splits][s147] · exercices : 1.1, 13.2–3, 20.11, 21.10–11, B6.6, B6.10, B6.29
- taux de base → voir [prévalence, taux de base](#p)
- **taux de compression** (*compression ratio*) — enseignée : [ch. 6, Mélanger les codes][s246] ; [ch. B2, Deux rapports][s247] · exercices : 6.24, 6.Q10, B2.R3
- **taux de fausses découvertes, de fausses omissions** (*false discovery rate, false omission rate, FDR, FOR*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s205] · exercices : 3.5
- **taux de faux positifs, de faux négatifs** (*false positive rate, false negative rate*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s205] · exercices : 3.2, 3.5, 3.7, 3.27, 4.Q8, 4.R1, CP1.7, 10.R3, 11.Q4, CP2.13 (+ 1)
- **taux de sélection** (*selection rate*) — enseignée : [mini-projet 3, La grille d'évaluation (sur 20)][s711] · exercices : B6.5, B6.23, B6.25–26, B6.R3
- *TDM* → voir [fouille de textes et de données](#f)
- **teacher forcing** (*forçage par l'enseignant*) — enseignée : [ch. B3, Teacher forcing et décalage d'un token][s569] · exercices : B3.Q10
- *technical debt* → voir [dette technique (du ML)](#d)
- **téléchargement reproductible (révision, SHA-256)** — enseignée : [ch. 29, Au-delà du livre : documenter et télécharger][s161] ; [projet final, PF.2 — Données : data card complète, licence vérifiée, script de…][s773] · exercices : 29.15, PF.2
- *temperature* → voir [température (d'une softmax)](#t)
- **température (d'une softmax)** (*temperature*) — enseignée : [ch. 17, 17.8 · La softmax][s250] · exercices : 12.21, 17.6, 17.19, 22.1, 22.5, 22.24, 22.E5, 28.28, B4.13, B4.Q5 (+ 30) · approfondie : ch. 22, 28 · réutilisée : ch. 26, B3, B4, B8
- *temporal-difference error* → voir [erreur TD](#e)
- **tenseur** (*tensor*) — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s234] ; [ch. 5, Au-delà du livre (3) : la différentiation automatique][s233] · exercices : 5.21, 20.1, 20.13–15, 20.19, 20.Q2, 23.12, 23.Q5, B1.1 (+ 69) · réutilisée : ch. 21, 23, 25
- **tenseur contigu** (*contiguous tensor*) — enseignée : [ch. 23, 23.3.2 · Tenseurs et tableaux][s424]
- **tenseurs PyTorch** — enseignée : [ch. 20, 20.2.1 · Les tenseurs][s234] · exercices : 20.13
- *tensor* → voir [tenseur](#t)
- *term frequency* → voir [TF-IDF](#t)
- **terme distribué** (*distributed term*) — enseignée : [ch. 11, 11.4 · La déduction][s297]
- **terminal** (*shell*) — enseignée : [ch. 0A, 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest…][s65] · exercices : 0A.10, 0A.12, 0A.Q1, 2.13, 3.15, 4.14, 5.11, 6.12, 7.13, B8.1 (+ 85)
- **terminal : cd, chemins, lancer python et pytest** — enseignée : [ch. 0A, 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest…][s65] · exercices : 0A.10, 0A.Q1
- *terminal state* → voir [état terminal, épisode tronqué](#e)
- test → voir [test A/B](#t), [test d'équivalence](#t), [test d'overfitting d'un mini-batch](#t), [test de confirmation](#t), [test de données](#t), [test de Kolmogorov-Smirnov](#t), [test de McNemar](#t), [test de non-régression](#t), [test de propriété](#t), [test de propriétés](#t), [test de santé](#t), [test par mutation](#t), …
- **test A/B** (*A/B test*) — enseignée : [checkpoint 1][s705] · exercices : 11.11, 11.E2
- **test d'équivalence** (*equivalence test*) — enseignée : [mini-projet 4, Les livrables][s713]
- **test d'overfitting d'un mini-batch** (*overfit-one-batch test*) — enseignée : [ch. B7, Quatre familles de tests][s666]
- **test de confirmation** (*confirmatory test*) — enseignée : [ch. 3, 3.8 · Appliquer la matrice de confusion : le piège de la prévalence][s206]
- **test de données** (*data test*) — enseignée : [ch. B7, 207.6 · Tester un projet ML : tests unitaires, tests de données,…][s102] · exercices : B7.Q7
- **test de Kolmogorov-Smirnov** (*Kolmogorov-Smirnov test*) — enseignée : [ch. B7, Le test de Kolmogorov-Smirnov][s676] · exercices : B7.4, B7.7–8, B7.16, B7.22, B7.E4, B7.Q11
- **test de McNemar** (*McNemar's test*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s269] · exercices : 8.26
- **test de non-régression** (*regression test*) — enseignée : [ch. 7][s262] · exercices : B2.30, B7.16, B7.18
- **test de propriété** (*property-based test*) — enseignée : [ch. 17, Au-delà du livre : choisir une activation, et tester une famille…][s348]
- **test de propriétés** (*property-based testing*) — enseignée : [ch. 5][s229]
- **test de santé** (*sanity check*) — enseignée : [ch. B6, Les tests de santé][s642] · exercices : B6.22, PF.9
- **test par mutation** (*mutation testing*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s269]
- **test par permutation** (*permutation test*) — enseignée : [ch. 8, 8.6 · Utiliser les résultats du test][s269] · exercices : 8.26
- **test paramétré** (*parametrized test*) — enseignée : [ch. 17, Au-delà du livre : choisir une activation, et tester une famille…][s348] · exercices : 0A.62, 17.12–16, 18.25, 21.16
- **test scellé** (*held-out test set, sealed test*) — enseignée : [projet final, La règle du test scellé][s715] · exercices : PF.4, PF.8
- **test séquentiel, règle d'arrêt** (*sequential test, stopping rule*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s218] · exercices : 7.26, 23.6
- **test unitaire** (*unit test*) — enseignée : [ch. B7, 207.6 · Tester un projet ML : tests unitaires, tests de données,…][s102] · évoquée : [ch. 0A, 100.1.3 · Le terminal : dossiers, chemins, lancer python, pytest…][s65] · exercices : 4.23
- **tests de santé des explications** — enseignée : [ch. B6, 206.7 · Limites des explications : instabilité, tests de santé,…][s786] · exercices : B6.16, B6.22, B6.E2, B6.E4, B6.Q6, B6.Q9
- **tests paramétrés pytest** — enseignée : [ch. 17][s746]
- **tests pour le ML** — enseignée : [ch. B7, 207.6 · Tester un projet ML : tests unitaires, tests de données,…][s102] · exercices : B7.15, B7.23, B7.25, B7.E3, B7.Q7
- **tête (d'un modèle)** (*head*) — enseignée : [ch. 23, 23.3.4 · La forme des tenseurs d'images][s425] ; [ch. B1, Backbone et tête][s433] · exercices : 16.E3, 23.24, B1.1, B1.4, B1.Q1, B1.R2, B3.19, B3.24–25, B3.Q6 (+ 61) · réutilisée : ch. B1
- **tête d'attention** (*attention head*) — enseignée : [ch. B3, Plusieurs questions à la fois][s560]
- *text and data mining* → voir [fouille de textes et de données](#f)
- **TF-IDF** (*term frequency, inverse document frequency*) — enseignée : [ch. 13, 13.6 · Naïve Bayes][s140] · exercices : 13.30, B4.5, B4.17, B4.28, B4.R3 · réutilisée : ch. B4
- TFR → voir [retour, retour total (TFR)](#r)
- théorème → voir [théorème d'impossibilité](#t), [théorème de convergence du perceptron](#t), [théorème du cycle du perceptron](#t), [théorème du gradient de politique](#t), [théorème du jury](#t), [théorème No Free Lunch](#t)
- **théorème d'impossibilité** (*impossibility theorem*) — enseignée : [ch. B6, L'impossibilité][s207] · exercices : B6.5–6, B6.10, B6.23–25, B6.E3, B6.Q11, B6.R3
- théorème de Bayes → voir [règle de Bayes, théorème de Bayes](#r)
- **théorème de convergence du perceptron** (*perceptron convergence theorem*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s288] · exercices : 10.7
- **théorème du cycle du perceptron** (*perceptron cycling theorem*) — enseignée : [ch. 10, 10.3.2 · L'histoire du perceptron][s288]
- **théorème du gradient de politique** (*policy gradient theorem*) — enseignée : [ch. B8, Le théorème du gradient de politique][s687] · exercices : B8.7
- **théorème du jury** (*Condorcet's jury theorem*) — enseignée : [ch. 14, 14.2 · Les ensembles][s74]
- **théorème No Free Lunch** (*no free lunch theorem*) — enseignée : [ch. 19, 19.6 · Choisir un optimiseur][s265] · évoquée : [ch. 11, 11.2.3 · L'optimisation][s293] · exercices : 11.12, 11.E3, 11.Q5, 19.Q12
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
- **tirage** (*draw, sample*) — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s179] · exercices : 2.2, 2.5–6, 2.18–19, 14.12, 25.5–6, 25.31, CP6.2 (+ 5) · réutilisée : ch. 3, 7
- **tirage avec remise, sans remise** (*sampling with replacement, without replacement*) — enseignée : [ch. 2, 2.5 · Tirages avec et sans remise][s187] · exercices : 0A.31, 2.6, 2.21, 2.23, 2.Q9, 3.R1, 6.R2, 11.6, 13.25, 14.12 (+ 5) · réutilisée : ch. 3, 8
- **tirage catégoriel** — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s179]
- *tirage de features* → voir [feature bagging](#f)
- *TN* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- *TNR* → voir [spécificité](#s)
- **token** (*jeton, unité lexicale*) — enseignée : [ch. B2, 202.4 · BPE au niveau octet, pré-tokenisation et tokens spéciaux][s31] ; [ch. 1, Panorama 2026 : où en est l'IA ?][s21] · exercices : B2.8–9, B2.14, B2.20, B2.31, B2.Q5–Q6, B2.R2, B3.Q10, B3.R1 (+ 30) · réutilisée : ch. 28, B3
- **token `[CLS]`** (*classification token*) — enseignée : [ch. B3, Teacher forcing et décalage d'un token][s569] · exercices : B3.28
- **token spécial** (*special token*) — enseignée : [ch. B2, Les tokens spéciaux][s527] · exercices : B2.3–4, B2.16, B2.19, B2.Q6–Q7, B4.4, B4.Q4
- *tokeniseur* → voir [tokenizer](#t)
- *tokenization premium* → voir [surcoût de tokenisation](#s)
- **tokenizer** (*tokeniseur*) — enseignée : [ch. 6, Au-delà du livre (1) : bits, nats, log loss et perplexité][s111] ; [ch. B2, Un algorithme de compression devenu tokenizer][s251] · exercices : B2.3–4, B2.10, B2.13, B2.18–19, B2.21, B2.30, B2.E3, B2.Q4 (+ 28) · approfondie : ch. B2 · réutilisée : ch. B3
- *top-1 agreement* → voir [accord top-1](#a)
- *top-5 accuracy* → voir [accuracy top-5](#a)
- **top-k et top-p** — enseignée : [ch. B3][s562] · réutilisée : ch. 28
- *top-k sampling* → voir [top-k, top-p](#t)
- **top-k, top-p** (*top-k sampling, nucleus sampling*) — enseignée : [ch. B3, 203.9 · Génération : KV cache et échantillonnage (température,…][s419] ; [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s249] · exercices : 0A.46, 22.20, 28.28, B4.13 · approfondie : ch. 28 · réutilisée : ch. 28
- top-p → voir [top-k, top-p](#t)
- *topological order* → voir [ordre topologique](#o)
- *torch.compile* → voir [compilation (d'un modèle)](#c)
- torch.export → voir [exporter (un modèle), `torch.export`](#e)
- **torchao** — enseignée : [ch. B7, Accélérer sur CPU][s446]
- **TorchScript** — enseignée : [ch. B7, torch.export et ONNX][s672] · exercices : B7.Q10
- **torchvision.transforms.v2** — enseignée : [ch. 24, 24.4.2 · Préparer les données pour un CNN][s764] · exercices : 24.23
- *total future reward* → voir [retour, retour total (TFR)](#r)
- *total variation* → voir [variation totale](#v)
- *TP* → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- TPU → voir [GPU, TPU, NPU](#g)
- **trace** — enseignée : [ch. 27, Au-delà du livre : mesurer un générateur][s481] · exercices : 27.14–15, 27.22, 27.26
- *trace de la pile d'appels* → voir [traceback](#t)
- **traceback** (*trace de la pile d'appels*) — enseignée : [ch. 0A, 100.6.2 · Exceptions : lire un traceback, `raise`, `try`/`except`][s81] · exercices : 0A.37, 0A.61
- *training* → voir [entraînement](#e)
- *training checkpoint* → voir [checkpoint (d'entraînement)](#c)
- *training error* → voir [erreur d'entraînement, erreur de généralisation](#e)
- *training history* → voir [historique (d'entraînement)](#h)
- *training mode* → voir [mode entraînement, mode évaluation](#m)
- *training set / test set* → voir [jeu d'entraînement / jeu de test](#j)
- **training-serving skew** (*décalage entre entraînement et service*) — enseignée : [ch. 12, 12.2 · La règle d'or : apprendre la transformation sur l'entraînement][s158] · exercices : B7.16, B7.Q11
- *transfer function* → voir [fonction de transfert](#f)
- **transfer learning** (*apprentissage par transfert*) — enseignée : [ch. B1, Des données et du calcul que tu n'as pas][s431] ; [ch. 23, 23.9.4 · Utiliser un modèle pré-entraîné][s429] · exercices : 23.24, 29.Q1, B1.7–8, B1.E2, B1.Q1–Q4, PF.6 (+ 40)
- **transférabilité** (*transferability*) — enseignée : [ch. 21, 21.10 · Les adversaires][s402]
- *transferability* → voir [transférabilité](#t)
- **transfert de style neuronal** (*neural style transfer*) — enseignée : [ch. 28, 28.4 · Le transfert de style][s488] · exercices : 28.E2
- **transfert négatif** (*negative transfer*) — enseignée : [ch. B1, Le transfert négatif][s510] · exercices : B1.22, B1.Q9
- **transformateur (scikit-learn)** (*transformer*) — enseignée : [ch. 12, 12.5.3 · Retenir la transformation][s307] · exercices : 12.16, 12.19, 12.21, 15.8, 15.14, 15.Q7, 15.Q11 · réutilisée : ch. 15
- **transformation inverse** — enseignée : [ch. 12, 12.5.3 · Retenir la transformation][s307]
- **transformation univariée, multivariée** (*univariate, multivariate transformation*) — enseignée : [ch. 12, 12.5.4 · Univariée ou multivariée][s308] · exercices : 12.Q7
- **Transformer** — enseignée : [ch. 1, Panorama 2026 : où en est l'IA ?][s21] · exercices : 0A.22, 12.21, CP3.2, 17.E4, 21.E5, 22.E1, 22.Q12, B3.28, B3.E5, B3.Q8 (+ 40) · approfondie : ch. B3, B4 · réutilisée : ch. B1, B4, B7
- *Transformer block* → voir [bloc Transformer](#b)
- **transformer de diffusion (DiT)** (*diffusion transformer*) — enseignée : [ch. B5, Les transformers de diffusion][s626] · exercices : 11.E3, 18.Q3, B5.8, B5.10, B5.13, B5.18–20, B5.Q11, B6.Q6 (+ 5)
- **transformers en pratique (5.x)** — enseignée : [ch. B4, 204.2 · Charger et exécuter un modèle avec transformers :…][s778]
- transition → voir [état, transition](#e)
- *transpose* → voir [transposée](#t)
- *transposed convolution* → voir [convolution transposée](#c)
- **transposée** (*transpose*) — enseignée : [ch. 0B, 101.4.1 · Forme, éléments, transposée ; un dataset est une matrice][s2] · exercices : 0A.53, 0A.64, 0B.9, 0B.42, 9.16, 12.8, 16.20, 21.6, 21.Q9, 25.4 (+ 16) · approfondie : ch. 21 · réutilisée : ch. 9, 16, 25, 27, 28, B3, B5
- *trapezoidal rule* → voir [méthode des trapèzes](#m)
- *Tree SHAP* → voir [TreeSHAP](#t)
- **TreeSHAP** (*Tree SHAP*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s635] · exercices : B6.7, B6.15–17, B6.Q1
- **triade mortelle** (*deadly triad*) — enseignée : [ch. B8, Instabilité : la triade mortelle][s682] · exercices : B8.Q2
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
- tuple → voir [liste, tuple](#l)
- *typage canard* → voir [duck typing](#d)
- **type** — enseignée : [ch. 0A, 100.2.3 · Types, conversions et erreurs de type][s67] · exercices : 0A.1, 0A.9, 0A.27, 0A.39, 0A.48, 0A.54, 0A.65, 0A.Q3, 8.4, 23.Q5 (+ 67) · approfondie : ch. 8
- *type hint* → voir [annotation de type](#a)
- *typed configuration* → voir [configuration typée](#c)

## U

- **U-Net** — enseignée : [ch. B5, Le U-Net et ses connexions de saut][s618] · exercices : B5.8, B5.21, B5.23–24, B5.26–28, B5.30–31, B5.Q6 (+ 1)
- **U-Net conditionné par le temps** — enseignée : [ch. B5, Dire au réseau quel est le pas : l'encodage du pas de temps][s784]
- **UCB** (*upper confidence bound*) — enseignée : [ch. 11, Au-delà du livre : le bandit manchot][s156] · exercices : 11.23, 11.26–27, 26.R3, B8.R3 · réutilisée : ch. 26
- *UE* → voir [règlement européen sur l'IA](#r), [RGPD](#r)
- UMAP → voir [t-SNE, UMAP](#t)
- **un-contre-tous (OvR)** (*one-versus-rest, one-vs-all, OvA*) — enseignée : [ch. 7, 7.4.1 · Un-contre-tous (OvR)][s209] ; [ch. 3, Au-delà du livre : plusieurs classes et moyennes macro, micro,…][s41] · exercices : 3.25, 7.1, 7.16, 7.22, 7.24, 7.E1, 7.Q6, 10.24, CP2.2, 14.R3 (+ 2) · réutilisée : ch. 10, 13, 14
- **un-contre-un (OvO)** (*one-versus-one*) — enseignée : [ch. 7, 7.4.2 · Un-contre-un (OvO)][s254] · exercices : 0B.12, 7.1–2, 7.23–24, 7.E1, 7.Q7, 10.24, CP2.2, 14.R3
- **underfitting** (*sous-apprentissage*) — enseignée : [ch. 9, 9.2.2 · L'underfitting][s34] · exercices : 9.9, 9.Q1, 9.Q3–Q4, CP2.7, 14.20, CP3.8, 20.25
- **underflow** (*sous-dépassement, dépassement par le bas*) — enseignée : [ch. 4, Au-delà du livre (1) : log-probabilités et underflow][s42] · évoquée : [ch. 0B][s43] · exercices : 4.18, 4.24, 9.26 · réutilisée : ch. 13
- *undistributed middle* → voir [moyen terme non distribué](#m)
- *unfold* → voir [im2col](#i)
- **Unicode et UTF-8** — enseignée : [ch. B2, 202.2 · Unicode, UTF-8 et normalisation][s523] · exercices : B2.1, B2.18, B2.Q2–Q3
- *uniform distribution* → voir [loi uniforme](#l)
- *uniform manifold approximation and projection* → voir [t-SNE, UMAP](#t)
- *unigram* → voir [modèle unigramme, bigramme, trigramme](#m)
- **Unigram (tokenizer)** (*unigram language model*) — enseignée : [ch. B2, Unigram : partir de trop et élaguer][s529] · exercices : B2.Q7
- *unigram language model* → voir [Unigram (tokenizer)](#u)
- *unit* → voir [neurone](#n), [unité](#u)
- *unit step* → voir [marche unité, marche de Heaviside, fonction signe](#m)
- *unit test* → voir [test unitaire](#t)
- **unité** (*unit*) — enseignée : [ch. 10, 10.3 · Les neurones artificiels][s287] · exercices : 0A.23, 0A.30, 2.8, 6.Q4, 10.8, 10.11, 10.23, 10.25, 10.Q4, 17.Q5 (+ 38)
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
- **UTF-8** — enseignée : [ch. B2, 202.2 · Unicode, UTF-8 et normalisation][s523] · exercices : B2.1, B2.13–14, B2.16–17, B2.20, B2.31, B2.Q2, B2.Q5, B2.Q8 (+ 3)
- *utility layer* → voir [couche utilitaire](#c)
- **uv, pip-tools, ruff, pre-commit** — enseignée : [ch. B7, pyproject.toml et installation éditable][s663] · exercices : 0B.51, B7.Q6
- uvicorn → voir [FastAPI, uvicorn, pydantic](#f)

## V

- *v-prediction* → voir [v-prédiction](#v)
- **v-prédiction** (*v-prediction*) — enseignée : [ch. B5][s166]
- **β-VAE** — enseignée : [ch. 25, Au-delà du livre : le compromis β, les anomalies, et les…][s449] · exercices : 25.34, 25.36
- **VAE (autoencodeur variationnel)** (*variational autoencoder*) — enseignée : [ch. 25, 25.9.2 · La structure d'un VAE][s455] · exercices : 25.11, 25.30–33, 25.38, 25.E2, 25.Q12, 25.R3, 26.R1 (+ 23) · réutilisée : ch. 27, B5
- valeur → voir [requête, clé, valeur](#r), [valeur absolue](#v), [valeur de base](#v), [valeur de Shapley](#v), [valeur manquante](#v), [valeur prédictive négative (VPN)](#v), [valeur prédite](#v), [valeur sentinelle](#v), [valeurs aberrantes](#v), [valeurs de Shapley exactes](#v), [valeurs SHAP](#v)
- **valeur absolue** (*absolute value*) — enseignée : [ch. 0B, 101.1.2 · Valeur absolue, inégalités, partie entière, fonction signe][s105] · exercices : 0B.2, 0B.39, 0B.Q4, 2.31, 5.1, 5.Q5, 8.25–26, 9.6, 9.16 (+ 16) · réutilisée : ch. 9
- **valeur de base** (*base value*) — enseignée : [ch. B6, Les features comme joueurs][s634] · exercices : B6.1–2, B6.8, B6.11, B6.14–16, B6.Q4–Q5
- **valeur de Shapley** (*Shapley value*) — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s631] · exercices : B6.1–2, B6.7–8, B6.11, B6.14, B6.17, B6.E2, B6.Q4–Q5
- **valeur manquante** (*missing value*) — enseignée : [ch. 0A, 100.9.3 · Valeurs manquantes et doublons][s97] · exercices : 0A.20, 0A.35, 0A.38, 1.2, 1.9, CP2.5, 12.10–11, 12.18, 12.33 (+ 3)
- **valeur prédictive négative (VPN)** (*negative predictive value, NPV*) — enseignée : [ch. 3, 3.7.9 · Les autres mesures][s205] · exercices : 3.5
- **valeur prédite** (*predicted value*) — enseignée : [ch. 7, 7.1 · Pourquoi ce chapitre ?][s142] · exercices : 7.Q1
- valeur propre → voir [vecteur propre, valeur propre](#v)
- valeur Q → voir [table Q, valeur Q](#t)
- **valeur sentinelle** (*sentinel value*) — enseignée : [ch. 12, 12.4 · Le nettoyage de base][s304]
- **valeurs aberrantes** — enseignée : [ch. 12][s738] · exercices : 9.10
- **valeurs de Shapley exactes** — enseignée : [ch. B6, 206.3 · Valeurs de Shapley : jeu coopératif, propriétés, calcul…][s631] · exercices : B6.1
- **valeurs SHAP** (*SHapley Additive exPlanations*) — enseignée : [ch. B6, SHAP, une famille d'explications additives][s635] · exercices : B6.11, B6.Q5–Q6
- *valid* → voir [valide](#v)
- validation → voir [validation croisée](#v), [validation croisée et recherche d'hyperparamètres](#v), [validation croisée imbriquée](#v), [validation croisée k-fold](#v), [validation externe](#v), [validation glissante](#v)
- **validation croisée** (*cross-validation*) — enseignée : [ch. 8, 8.5 · La validation croisée][s57] ; [ch. 24, 24.3.2 · La validation croisée][s58] · exercices : 8.5, 8.25, 8.E3, 8.Q9, CP2.10, 13.18, 15.27, 15.R2–R3, CP3.11 (+ 55) · réutilisée : ch. 13, 14, 24
- *validation croisée à $k$ plis* → voir [k-fold](#k)
- **validation croisée et recherche d'hyperparamètres** — enseignée : [ch. 24, 24.3.4 · La recherche d'hyperparamètres][s329]
- **validation croisée imbriquée** (*nested cross-validation*) — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s263] · exercices : 8.2, CP2.5, 15.1, 15.24, 15.31, CP3.14
- **validation croisée k-fold** — enseignée : [ch. 8, 8.5.1 · La validation croisée à k folds][s263] · exercices : 8.1, 8.5, 8.14–15, 8.19, 8.21–22, 8.E3, 8.E5, 8.Q10
- *validation croisée par exclusion d'un exemple* → voir [leave-one-out](#l)
- *validation curve* → voir [courbe de validation](#c)
- **validation externe** (*external validation*) — enseignée : [ch. 8, 8.4 · Le jeu de validation][s168]
- **validation glissante** (*time series cross-validation, rolling origin*) — enseignée : [ch. 22][s413]
- *validation set* → voir [jeu de validation](#j)
- **valide** (*valid*) — enseignée : [ch. 11, 11.4 · La déduction][s297] · exercices : 0A.25, 0A.45, 0A.65, 4.14, 11.3–4, 11.15, 11.Q7, 11.Q11, CP2.9 (+ 19)
- **validité et solidité** — enseignée : [ch. 11][s737]
- *value* → voir [requête, clé, valeur](#r)
- *value function* → voir [fonction de valeur](#f), [fonction de valeur (RL)](#f)
- *value iteration* → voir [itération sur les valeurs](#i)
- *vanishing* → voir [gradient évanescent, explosif](#g)
- **variable** — enseignée : [ch. 0A, 100.2 · Valeurs, variables et types][s66] · exercices : 0A.40, 0B.27, 0B.29, 2.20, 12.E5, 25.19, 25.Q4, B6.9, B6.26, B6.Q10 (+ 76) · approfondie : ch. 0B · réutilisée : ch. 2, 8, 12, 16, 20, 24, 25, 26, B3
- **variable aléatoire** (*random variable*) — enseignée : [ch. 0B, 101.7.3 · Variable aléatoire discrète et espérance][s133] · exercices : 0B.E5, 2.Q7 · réutilisée : ch. 2, 26
- **variable aléatoire discrète, espérance, variance, écart-type, linéarité de l'espérance** — enseignée : [ch. 0B, 101.7.3 · Variable aléatoire discrète et espérance][s133] · exercices : 0B.28
- **variable d'environnement** (*environment variable*) — enseignée : [ch. B7, Variables d'environnement et secrets][s659] · exercices : B7.18
- **variable de confusion** (*confounder, confounding variable*) — enseignée : [ch. 2, 2.8.2 · La corrélation][s197] · exercices : 2.10
- **variable de substitution** (*proxy, proxy variable, indicateur indirect, substitut*) — enseignée : [ch. 2, Variable aléatoire, tirage, loi discrète et densité][s179] ; [ch. B6, Corrélation n'est pas causalité][s321] ; [ch. 13][s315] · exercices : 14.9, 20.11, B6.9, B6.Q10 · approfondie : ch. mini-projet 3
- **variable indicatrice** (*dummy variable*) — enseignée : [ch. 12, 12.3.1 · Le one-hot][s98]
- *variable-length code* → voir [code à longueur variable, code adaptatif](#c)
- **variables et types, f-strings** — enseignée : [ch. 0A, 100.2 · Valeurs, variables et types][s66] · exercices : 0A.Q3
- variables latentes → voir [code, variables latentes](#c)
- **variance (d'une famille de modèles)** — enseignée : [ch. 9, 9.6 · Biais et variance][s284] · exercices : 9.4, 9.8, 9.11, 9.24–25, 9.29, 9.Q9–Q11, 9.R3
- **variance et ddof** — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s185] · exercices : 9.R3
- **variance expliquée (d'un critique)** (*explained variance*) — enseignée : [ch. B8, Le critique et sa variance expliquée][s695] ; [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s199] · exercices : 12.7, 12.23, 12.27, B8.18, B8.20–21 · réutilisée : ch. 25
- *variance preserving diffusion* → voir [diffusion à variance préservée](#d)
- **variance, écart-type** (*standard deviation*) — enseignée : [ch. 0B, 101.7.4 · Variance et écart-type][s135] · exercices : 0B.24, 0B.27–28, 0B.53, 0B.E5, 0B.Q12
- **variation totale** (*total variation*) — enseignée : [ch. 28, 28.4.5 · Réaliser le transfert][s493] · exercices : 28.4, 28.18, 28.20, 28.24, 28.Q9
- *variational autoencoder* → voir [VAE (autoencodeur variationnel)](#v)
- **variations, minimum et maximum (dérivée nulle)** — enseignée : [ch. 0B, 101.5.4 · Variations, minimum et maximum][s730] · exercices : 0A.47, 0B.48, 0B.50, 5.4, 21.Q11
- vecteur → voir [vecteur de contexte](#v), [vecteur de support](#v), [vecteur propre, valeur propre](#v), [vecteurs : somme, produit par un scalaire, norme (L1, L2, L∞), distance euclidienne](#v)
- **vecteur de contexte** (*context vector*) — enseignée : [ch. B3, Traduire avec deux RNN][s551] · exercices : B3.Q1
- **vecteur de support** (*support vector*) — enseignée : [ch. 13, 13.7 · Bilan : choisir un classifieur][s314] · exercices : 9.Q5, 13.18, 13.Q5
- **vecteur propre, valeur propre** (*eigenvector, eigenvalue*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s199] · exercices : 12.7–8, 12.22, 25.8
- **vecteurs : somme, produit par un scalaire, norme (L1, L2, L∞), distance euclidienne** — enseignée : [ch. 0B, 101.3.2 · Norme et distance][s114] · exercices : 0A.49, 0A.52, 0B.8, 0B.41, 0B.E2, 2.25, 5.25, 7.Q5, 13.1, B2.Q12 (+ 6) · réutilisée : ch. 7
- *vector database* → voir [base de données vectorielle](#b)
- **vectorisation** (*vectorization*) — enseignée : [ch. 0A, 100.8.3 · Calcul vectorisé et fonctions universelles][s93]
- *vectorization* → voir [vectorisation](#v)
- *vectorized environments* → voir [environnements vectorisés](#e)
- *velocity field* → voir [champ de vitesses](#c)
- **vérification du gradient** (*gradient checking, gradient check*) — enseignée : [ch. 18, Au-delà du livre : vérifier un gradient][s139] · évoquée : [ch. 0B, 101.6.3 · Gradient et direction de plus grande pente][s54] · exercices : 0B.47
- **vérité terrain** (*ground truth*) — enseignée : [ch. 3, 3.7.1 · Classer des échantillons][s204] · exercices : 3.2, 3.Q6, 7.Q1
- *view / copy* → voir [vue / copie](#v)
- **Vision Transformer (ViT)** — enseignée : [ch. B3, Le Vision Transformer][s404] ; [ch. 21, 21.9.1 · VGG16][s405] · exercices : 4.15, 21.E5, 21.Q12, B1.7, B3.28, B5.Q11, PF.6, PF.10
- **visualisation de features** (*feature visualization*) — enseignée : [ch. 28, Au-delà du livre : la génération d'images aujourd'hui][s486]
- **visualisation de filtres et de cartes d'activation** — enseignée : [ch. 21, 21.9.3 · Regarder les filtres (2) : les cartes d'activation][s401]
- ViT → voir [Vision Transformer (ViT)](#v), [patch (ViT)](#p)
- *Viterbi algorithm* → voir [algorithme de Viterbi](#a)
- **vivacité trompeuse** (*misleading vividness*) — enseignée : [ch. 11, 11.5.2 · Les sophismes inductifs][s299] · exercices : 11.Q10
- **vocabulaire (d'un modèle de langue)** (*vocabulary*) — enseignée : [ch. 22, 22.9 · Un exemple : générer du Holmes caractère par caractère][s249] · exercices : 4.Q4, 6.Q6, 7.Q1, 11.14, 13.Q7, 16.Q2, 22.12, 28.7, B2.20, B2.Q1 (+ 36) · approfondie : ch. 1 · réutilisée : ch. 3, 24, B2, B3
- **vocabulaire caractères/mots, encodage** — enseignée : [ch. 22][s413]
- **vocabulaire du machine learning** — enseignée : [ch. 1, 1.1 · Le machine learning : extraire du sens des données][s167] · exercices : 1.9
- *vocabulary* → voir [vocabulaire (d'un modèle de langue)](#v)
- *Voronoi cells* → voir [cellules de Voronoï](#c)
- **vote à la pluralité, vote pondéré** (*plurality voting, weighted voting*) — enseignée : [ch. 14, 14.3 · Le vote][s323] · exercices : 14.1–2, 14.12, 14.23, 14.Q3, 15.Q9
- **vote dur, vote souple** (*hard voting, soft voting*) — enseignée : [ch. 14, 14.3 · Le vote][s323] · exercices : 14.14, 14.17–18
- vote pondéré → voir [vote à la pluralité, vote pondéré](#v)
- vote souple → voir [vote dur, vote souple](#v)
- VPN → voir [valeur prédictive négative (VPN)](#v)
- vrai négatif → voir [vrai positif, faux positif, faux négatif, vrai négatif](#v)
- **vrai positif, faux positif, faux négatif, vrai négatif** (*true positive, false positive, false negative, true negative, TP, FP, FN, TN*) — enseignée : [ch. 3, 3.7.2 · La matrice de confusion][s63] · exercices : 3.2, 3.5, 3.7, 3.9–10, 3.27, 3.Q8, 3.Q12, 4.Q8, 4.R1 (+ 16)
- vraisemblance → voir [log-probabilité, vraisemblance](#l)
- **vraisemblance (d'une hypothèse)** (*likelihood*) — enseignée : [ch. 4, 4.4.2 · Retrouver la règle et nommer ses termes][s218] · exercices : 4.1–2, 4.5, 4.7, 4.14–16, 4.Q6–Q7, 9.27 (+ 6)
- **vue / copie** (*view / copy*) — enseignée : [ch. 0A, 100.8.2 · Indexation, tranches, masques booléens ; vues et copies][s94] · exercices : 0A.29, 0A.39, 0A.56, 0A.58, 0A.62, 0A.Q10 · réutilisée : ch. 2, 25, B6

## W

- **warmup** (*échauffement*) — enseignée : [ch. 19, Au-delà du livre : warmup et décroissance en cosinus][s360] ; [ch. B1, Warmup puis cosinus, groupe par groupe][s361] · exercices : 19.19, 19.23, 19.27, 19.E3, 19.Q7, 19.Q12, B1.3, B1.19, B1.Q6, B3.20 (+ 10) · réutilisée : ch. B3
- **warmup + cosinus** — enseignée : [ch. 19, Au-delà du livre : warmup et décroissance en cosinus][s360] · exercices : 19.19, B1.19, B3.20 · réutilisée : ch. B1, B3
- *warning* → voir [avertissement](#a)
- *Wasserstein distance* → voir [distance de Wasserstein (1D)](#d)
- **waterfall** (*graphique en cascade*) — enseignée : [ch. B6, Les trois graphiques à connaître][s637] · exercices : B6.8, B6.15–16, B6.Q2
- *watermark* → voir [filigrane](#f)
- *weak learner* → voir [apprenant faible, apprenant fort](#a)
- **weight decay** (*décroissance des poids*) — enseignée : [ch. 9, Pénalité sur les poids, dropout, batchnorm][s25] · exercices : 19.6, 19.13, 19.16, 19.18, 20.28–29, 20.R1, CP4.3, 23.11, 24.31 (+ 5) · approfondie : ch. 19, 20 · réutilisée : ch. 20
- *weight naming convention* → voir [convention AD (ou DA)](#c)
- *weight sharing* → voir [partage de poids](#p)
- *weight tying* → voir [poids liés](#p)
- *weight-decomposed low-rank adaptation* → voir [DoRA](#d)
- *weighted average* → voir [moyenne pondérée](#m), [moyenne macro, micro, pondérée](#m)
- *weighted sum* → voir [somme pondérée, pré-activation](#s)
- *weighted voting* → voir [vote à la pluralité, vote pondéré](#v)
- *weights* → voir [poids](#p)
- WGAN-GP → voir [pénalité de gradient (WGAN-GP)](#p)
- **whitening** (*blanchiment*) — enseignée : [ch. 12, 12.7.1 · L'analyse en composantes principales (PCA)][s199] · exercices : 12.7, 12.27
- *winner's curse* → voir [biais d'optimisme (du gagnant)](#b)
- *within-cluster sum of squares* → voir [inertie](#i)
- *without replacement* → voir [tirage avec remise, sans remise](#t)
- **word2vec** — enseignée : [ch. B2, word2vec : prédire plutôt que compter][s540] · exercices : B2.7, B2.25–27, B2.Q10–Q11
- **word2vec skip-gram** — enseignée : [ch. B2, word2vec : prédire plutôt que compter][s540]
- **WordPiece** — enseignée : [ch. B2, 202.5 · WordPiece, Unigram, SentencePiece ; la bibliothèque…][s528] · exercices : B2.13–14, B2.E2, B2.Q7
- **workbook : wb.check, wb.attempt, start_chapter.py, mylearn** — enseignée : [ch. 0A, 100.11.5 · mylearn : stubs, référence et tests fondés sur un oracle][s103] · exercices : 0A.26, 0A.63–66, 0A.Q2
- **worker (d'un `DataLoader`)** (*processus de chargement*) — enseignée : [ch. 24, 24.4.6 · L'augmentation de données][s440] · exercices : 24.24, B7.12
- **workflow PyTorch complet** — enseignée : [ch. 23][s432]
- **workflow, job, étape, action** (*step*) — enseignée : [ch. B7, L'intégration continue][s667] · exercices : 0A.E5, B7.23, B7.25, PF.10 · approfondie : ch. 23 · réutilisée : ch. 24
- *wrapper* → voir [adaptateur](#a)

## X

- Xavier → voir [initialisation LeCun, Glorot (Xavier), He (Kaiming)](#i)
- **XGBoost** — enseignée : [ch. 14, Au-delà du livre : le gradient boosting][s195] · exercices : 14.8, 14.26–28, 14.E3
- **XOR (ou exclusif)** (*exclusive or*) — enseignée : [ch. 10, 10.3.1 · Le perceptron][s253] · exercices : 10.4–5, 10.13, 10.16, 10.18–19, 10.21, 10.Q5, 11.R1, CP2.8 (+ 3)

## Y

- **YAML** — enseignée : [ch. B7, 207.3 · Configuration : dataclasses, YAML ou JSON, surcharges,…][s661] · exercices : 23.Q12, B7.8, B7.11, B7.13, B7.23, B7.Q2, B7.Q4, PF.6
- **YaRN** (*yet another RoPE extension*) — enseignée : [ch. B3, Allonger le contexte][s576]
- *yet another RoPE extension* → voir [YaRN](#y)

## Z

- **z-score, standardiser** (*standardize*) — enseignée : [ch. 2, Mesurer la dispersion de données : variance, écart-type, ddof][s185] · exercices : 2.3, 2.15, 2.31, CP1.3–4, 7.17, 9.22, 9.28, 9.31, 12.2 (+ 5) · réutilisée : ch. 7, 9, 12
- *zero terminal SNR* → voir [SNR final nul](#s)
- *zero-initialized* → voir [adaLN-Zero](#a)
- **zero-padding** (*remplissage de zéros*) — enseignée : [ch. 20, 20.4.5 · La convolution (aperçu)][s28] · exercices : 20.6, 20.20, 20.Q7, 20.Q9, 21.2
- *zero-shot* → voir [zéro-shot](#z)
- **zéro-shot** (*zero-shot*) — enseignée : [ch. B1, Le zéro-shot][s512] ; [ch. B4, Zéro-shot et few-shot][s513] · exercices : B1.5, B1.24–25, B1.28, B1.E4, B1.Q10, B4.15, B4.28, B4.Q6
- **zéro-shot CLIP** — enseignée : [ch. B1, 201.8 · Modèles de fondation : auto-supervisé (DINOv2),…][s165] · exercices : B1.5
- *zero-sum game* → voir [jeu à somme nulle](#j)

<a id="symboles"></a>

## 0-9 et symboles

- *1D convolution* → voir [convolution 1D](#c)
- **3CosAdd** — enseignée : [ch. B2, Les analogies][s542] · exercices : B2.2
- *68-95-99.7 rule* → voir [règle 68-95-99,7](#r)

[s1]: ../chapitres/ch00a_python/01_fiche.md#10012--le-workbook--setup-wbcheck-wbattempt-start_chapterpy-mon_travail
[s2]: ../chapitres/ch00b_maths/01_fiche.md#10141--forme-éléments-transposée--un-dataset-est-une-matrice
[s3]: ../chapitres/ch20_deep_learning/01_fiche.md#au-delà-du-livre--pytorch-en-pratique
[s4]: ../chapitres/ch00a_python/01_fiche.md#10095--de-pandas-à-numpy--x-et-y
[s5]: ../chapitres/ch00a_python/01_fiche.md#10092--sélectionner--colonnes-loc-iloc-filtres
[s6]: ../chapitres/ch19_optimiseurs/01_fiche.md#1941--la-descente-de-gradient-par-batch
[s7]: ../chapitres/ch00a_python/01_fiche.md#10087--aléatoire-reproductible-et-mini-batches
[s8]: ../chapitres/ch19_optimiseurs/01_fiche.md#1943--la-descente-par-mini-batches
[s9]: ../chapitres/ch08_train_test/01_fiche.md#82--la-boucle-dentraînement-
[s10]: ../chapitres/ch18_backprop/01_fiche.md#18111--explorer-le-learning-rate-
[s11]: ../chapitres/ch18_backprop/01_fiche.md#1811--le-learning-rate-
[s12]: ../chapitres/ch23_pytorch_1/01_fiche.md#23102--le-learning-rate-
[s13]: ../chapitres/ch00b_maths/01_fiche.md#10117--dénombrement--principe-multiplicatif-factorielle-coefficient-binomial
[s14]: ../chapitres/ch00b_maths/01_fiche.md#10113--notations-σ-et-π
[s15]: ../chapitres/ch15_sklearn/01_fiche.md#1595--les-pipelines-
[s16]: ../chapitres/b4_llm/01_fiche.md#le-pipeline--une-ligne-pour-essayer
[s17]: ../chapitres/ch12_preparation/01_fiche.md#1210--transformations-et-validation-croisée-
[s18]: ../chapitres/ch15_sklearn/01_fiche.md#152--scikit-learn-numpy-et-scipy-
[s19]: ../chapitres/ch16_feedforward/01_fiche.md#1631--le-graphe-en-pratique-
[s20]: ../chapitres/b1_transfer/01_fiche.md#2014--fine-tuning--partiel-complet-dégel-progressif-learning-rates-discriminatifs-batchnorm-
[s21]: ../chapitres/ch01_introduction/01_fiche.md#panorama-2026--où-en-est-lia-
[s22]: ../chapitres/b2_tokenisation/01_fiche.md#2027--embeddings--de-one-hot-à-nnembedding-entraînement-et-poids-liés-
[s23]: ../chapitres/ch12_preparation/01_fiche.md#1272--standardisation-et-pca-pour-des-images-
[s24]: ../chapitres/ch20_deep_learning/01_fiche.md#2043--le-dropout-
[s25]: ../chapitres/ch09_overfitting/01_fiche.md#pénalité-sur-les-poids-dropout-batchnorm
[s26]: ../chapitres/ch20_deep_learning/01_fiche.md#2046--le-pooling-aperçu
[s27]: ../chapitres/ch21_cnn/01_fiche.md#2134--le-padding-
[s28]: ../chapitres/ch20_deep_learning/01_fiche.md#2045--la-convolution-aperçu
[s29]: ../chapitres/ch21_cnn/01_fiche.md#2135--le-stride-
[s30]: ../chapitres/ch21_cnn/01_fiche.md#2125--le-noyau
[s31]: ../chapitres/b2_tokenisation/01_fiche.md#2024--bpe-au-niveau-octet-pré-tokenisation-et-tokens-spéciaux-
[s32]: ../chapitres/b3_transformers/01_fiche.md#le-décodeur--gpt-et-le-modèle-de-langue-causal
[s33]: ../chapitres/ch09_overfitting/01_fiche.md#92--overfitting-et-underfitting-
[s34]: ../chapitres/ch09_overfitting/01_fiche.md#922--lunderfitting-
[s35]: ../chapitres/ch01_introduction/01_fiche.md#17--deep-learning-
[s36]: ../chapitres/ch00a_python/01_fiche.md#1001--prise-en-main--notebooks-terminal-et-workbook
[s37]: ../chapitres/ch03_probabilites/01_fiche.md#375--accuracy-
[s38]: ../chapitres/ch01_introduction/01_fiche.md#123--généraliser--le-jeu-de-test-
[s39]: ../chapitres/ch03_probabilites/01_fiche.md#376--precision-
[s40]: ../chapitres/ch03_probabilites/01_fiche.md#377--recall-
[s41]: ../chapitres/ch03_probabilites/01_fiche.md#au-delà-du-livre--plusieurs-classes-et-moyennes-macro-micro-pondérée
[s42]: ../chapitres/ch04_bayes/01_fiche.md#au-delà-du-livre-1--log-probabilités-et-underflow
[s43]: ../chapitres/ch00b_maths/01_fiche.md
[s44]: ../chapitres/ch04_bayes/01_fiche.md#461--la-boucle-posterior--prior-
[s45]: ../chapitres/ch04_bayes/01_fiche.md#au-delà-du-livre-3--map-et-intervalle-de-crédibilité
[s46]: ../chapitres/ch17_activations/01_fiche.md#au-delà-du-livre--calculer-sans-débordement
[s47]: ../chapitres/ch01_introduction/01_fiche.md#124--modèle-paramètres-capacité-hyperparamètres-
[s48]: ../chapitres/ch16_feedforward/01_fiche.md#164--les-poids-
[s49]: ../chapitres/ch00b_maths/01_fiche.md#10114--moyenne-moyenne-pondérée-somme-pondérée-moyenne-mobile
[s50]: ../chapitres/ch10_neurones/01_fiche.md#1033--les-neurones-modernes-
[s51]: ../chapitres/ch10_neurones/01_fiche.md#102--les-vrais-neurones-
[s52]: ../chapitres/ch17_activations/01_fiche.md#172--ce-que-fait-une-fonction-dactivation-
[s53]: ../chapitres/ch05_courbes/01_fiche.md#suivre-la-pente-jusquà-un-extremum
[s54]: ../chapitres/ch00b_maths/01_fiche.md#10163--gradient-et-direction-de-plus-grande-pente
[s55]: ../chapitres/ch18_backprop/01_fiche.md#189--la-rétropropagation-en-action-
[s56]: ../chapitres/ch01_introduction/01_fiche.md#122--la-version-pour-ordinateur--échantillons-features-labels-
[s57]: ../chapitres/ch08_train_test/01_fiche.md#85--la-validation-croisée-
[s58]: ../chapitres/ch24_pytorch_2/01_fiche.md#2432--la-validation-croisée-
[s59]: ../chapitres/ch01_introduction/01_fiche.md#13--apprentissage-supervisé
[s60]: ../chapitres/ch01_introduction/01_fiche.md#14--apprentissage-non-supervisé
[s61]: ../chapitres/ch01_introduction/01_fiche.md#16--apprentissage-par-renforcement
[s62]: ../chapitres/ch26_rl/01_fiche.md
[s63]: ../chapitres/ch03_probabilites/01_fiche.md#372--la-matrice-de-confusion-
[s64]: ../chapitres/ch00a_python/01_fiche.md#10011--cellules-noyau-ordre-dexécution-et--tout-exécuter-
[s65]: ../chapitres/ch00a_python/01_fiche.md#10013--le-terminal--dossiers-chemins-lancer-python-pytest-et-git
[s66]: ../chapitres/ch00a_python/01_fiche.md#1002--valeurs-variables-et-types
[s67]: ../chapitres/ch00a_python/01_fiche.md#10023--types-conversions-et-erreurs-de-type
[s68]: ../chapitres/ch00a_python/01_fiche.md#10022--chaînes-de-caractères-et-f-strings
[s69]: ../chapitres/ch00a_python/01_fiche.md
[s70]: ../chapitres/ch00a_python/01_fiche.md#10031--listes--indices-tranches-méthodes-mutabilité
[s71]: ../chapitres/ch00a_python/01_fiche.md#10033--dictionnaires
[s72]: ../chapitres/ch00a_python/01_fiche.md#10034--ensembles
[s73]: ../chapitres/ch00b_maths/01_fiche.md#10116--ensembles--union-intersection-complémentaire-cardinal
[s74]: ../chapitres/ch14_ensembles/01_fiche.md#142--les-ensembles-
[s75]: ../chapitres/ch00a_python/01_fiche.md#10043--compréhensions-de-listes-de-dictionnaires-et-densembles
[s76]: ../chapitres/ch00a_python/01_fiche.md#1005--fonctions
[s77]: ../chapitres/ch00b_maths/01_fiche.md#10121--fonction-graphe-fonction-affine
[s78]: ../chapitres/ch00a_python/01_fiche.md#10051--définir-une-fonction--paramètres-valeurs-par-défaut-return-portée
[s79]: ../chapitres/ch00a_python/01_fiche.md#10054--fermetures-closures
[s80]: ../chapitres/ch00a_python/01_fiche.md#10055--docstrings-au-format-numpy-et-annotations-de-type
[s81]: ../chapitres/ch00a_python/01_fiche.md#10062--exceptions--lire-un-traceback-raise-tryexcept
[s82]: ../chapitres/ch00a_python/01_fiche.md#10056--fonctions-récursives
[s83]: ../chapitres/ch00a_python/01_fiche.md#1006--modules-erreurs-et-fichiers
[s84]: ../chapitres/ch00a_python/01_fiche.md#1007--classes-et-objets
[s85]: ../chapitres/ch00a_python/01_fiche.md#10073--héritage-et-super
[s86]: ../chapitres/ch27_gan/01_fiche.md#2742--le-générateur-
[s87]: ../chapitres/ch00a_python/01_fiche.md#10074--itérables-et-générateurs-yield
[s88]: ../chapitres/ch00a_python/01_fiche.md#10064--sérialiser--json-et-pickle
[s89]: ../chapitres/ch00a_python/01_fiche.md#10065--expressions-régulières-re--les-bases
[s90]: ../chapitres/ch00a_python/01_fiche.md#10081--créer-des-arrays--dtype-shape-ndim
[s91]: ../chapitres/ch00a_python/01_fiche.md#10084--broadcasting
[s92]: ../chapitres/ch00a_python/01_fiche.md#10085--réductions-par-axe-tri-argmax-unique
[s93]: ../chapitres/ch00a_python/01_fiche.md#10083--calcul-vectorisé-et-fonctions-universelles
[s94]: ../chapitres/ch00a_python/01_fiche.md#10082--indexation-tranches-masques-booléens--vues-et-copies
[s95]: ../chapitres/b7_production/01_fiche.md#les-graines-et-le-déterminisme
[s96]: ../chapitres/ch00a_python/01_fiche.md#10091--dataframe-et-series--read_csv-head-info-describe
[s97]: ../chapitres/ch00a_python/01_fiche.md#10093--valeurs-manquantes-et-doublons
[s98]: ../chapitres/ch12_preparation/01_fiche.md#1231--le-one-hot-
[s99]: ../chapitres/b7_production/01_fiche.md#2071--du-notebook-au-projet--limites-du-notebook-structure-dun-dépôt-pyprojecttoml-installation-éditable-
[s100]: ../chapitres/ch00a_python/01_fiche.md#100111--git--status-add-commit-push-pull--messages-de-commit--gitignore
[s101]: ../chapitres/ch00a_python/01_fiche.md#100112--git--branches-aperçu
[s102]: ../chapitres/b7_production/01_fiche.md#2076--tester-un-projet-ml--tests-unitaires-tests-de-données-test-doverfitting-intégration-continue-
[s103]: ../chapitres/ch00a_python/01_fiche.md#100115--mylearn--stubs-référence-et-tests-fondés-sur-un-oracle
[s104]: ../chapitres/ch00b_maths/01_fiche.md#10111--puissances-racines-notation-scientifique-ordres-de-grandeur
[s105]: ../chapitres/ch00b_maths/01_fiche.md#10112--valeur-absolue-inégalités-partie-entière-fonction-signe
[s106]: ../chapitres/ch00b_maths/01_fiche.md#10115--suites-géométriques-et-leur-somme
[s107]: ../chapitres/ch00b_maths/01_fiche.md#10124--logarithmes--ln-log-log-changement-de-base
[s108]: ../chapitres/ch00b_maths/01_fiche.md#10125--sigmoïde-et-tangente-hyperbolique--définitions-et-allure
[s109]: ../chapitres/ch13_classifieurs/01_fiche.md#au-delà-du-livre--la-régression-logistique-le-classifieur-paramétrique-de-référence
[s110]: ../chapitres/ch17_activations/01_fiche.md#176--les-fonctions-lisses-
[s111]: ../chapitres/ch06_information/01_fiche.md#au-delà-du-livre-1--bits-nats-log-loss-et-perplexité
[s112]: ../chapitres/ch00b_maths/01_fiche.md#10126--cosinus-aperçu
[s113]: ../chapitres/ch00b_maths/01_fiche.md#10131--composantes-somme-produit-par-un-scalaire
[s114]: ../chapitres/ch00b_maths/01_fiche.md#10132--norme-et-distance
[s115]: ../chapitres/ch00b_maths/01_fiche.md#10133--produit-scalaire-angle-similarité-cosinus
[s116]: ../chapitres/b2_tokenisation/01_fiche.md#le-cosinus
[s117]: ../chapitres/ch00b_maths/01_fiche.md#10134--produit-élément-par-élément-hadamard
[s118]: ../chapitres/ch00b_maths/01_fiche.md#10143--produit-matriciel-et-vérification-des-formes
[s119]: ../chapitres/ch00b_maths/01_fiche.md#10144--identité-et-idée-de-linverse
[s120]: ../chapitres/ch00b_maths/01_fiche.md#10151--taux-daccroissement-tangente-dérivée
[s121]: ../chapitres/ch00b_maths/01_fiche.md#1015--dérivées
[s122]: ../chapitres/ch05_courbes/01_fiche.md#53--la-dérivée-
[s123]: ../chapitres/ch00b_maths/01_fiche.md#10153--règle-de-la-chaîne
[s124]: ../chapitres/ch00b_maths/01_fiche.md#10162--dérivées-partielles
[s125]: ../chapitres/ch05_courbes/01_fiche.md#54--le-gradient-
[s126]: ../chapitres/ch00b_maths/01_fiche.md#10161--fonctions-de-deux-variables-et-lignes-de-niveau
[s127]: ../chapitres/ch07_classification/01_fiche.md#721--classification-binaire-en-2d-
[s128]: ../chapitres/ch00b_maths/01_fiche.md#10164--règle-de-la-chaîne-à-plusieurs-variables--la-somme-sur-les-chemins
[s129]: ../chapitres/ch18_backprop/01_fiche.md#18123--ce-quelle-fait-
[s130]: ../chapitres/ch00b_maths/01_fiche.md#10171--expérience-aléatoire-événement-probabilité-complémentaire
[s131]: ../chapitres/ch03_probabilites/01_fiche.md#33--probabilité-simple
[s132]: ../chapitres/ch00b_maths/01_fiche.md#10172--indépendance
[s133]: ../chapitres/ch00b_maths/01_fiche.md#10173--variable-aléatoire-discrète-et-espérance
[s134]: ../chapitres/ch02_stats/01_fiche.md#235--lespérance
[s135]: ../chapitres/ch00b_maths/01_fiche.md#10174--variance-et-écart-type
[s136]: ../chapitres/ch00b_maths/01_fiche.md#10175--simuler-pour-estimer--fréquences-et-loi-des-grands-nombres
[s137]: ../chapitres/b8_rl_moderne/01_fiche.md#la-baseline-et-la-variance
[s138]: ../chapitres/ch13_classifieurs/01_fiche.md#131--pourquoi-ce-chapitre--
[s139]: ../chapitres/ch18_backprop/01_fiche.md#au-delà-du-livre--vérifier-un-gradient
[s140]: ../chapitres/ch13_classifieurs/01_fiche.md#136--naïve-bayes-
[s141]: ../chapitres/ch01_introduction/01_fiche.md#112--les-systèmes-experts-et-leurs-limites-
[s142]: ../chapitres/ch07_classification/01_fiche.md#71--pourquoi-ce-chapitre-
[s143]: ../chapitres/ch08_train_test/01_fiche.md#83--le-jeu-de-test-
[s144]: ../chapitres/ch01_introduction/01_fiche.md
[s145]: ../chapitres/ch01_introduction/01_fiche.md#131--classification-
[s146]: ../chapitres/ch01_introduction/01_fiche.md#132--régression
[s147]: ../chapitres/ch13_classifieurs/01_fiche.md#1352--choisir-les-splits-
[s148]: ../chapitres/ch01_introduction/01_fiche.md#141--clustering
[s149]: ../chapitres/ch01_introduction/01_fiche.md#142--débruitage
[s150]: ../chapitres/ch01_introduction/01_fiche.md#143--réduction-de-dimension
[s151]: ../chapitres/ch12_preparation/01_fiche.md#127--la-réduction-de-dimension-
[s152]: ../chapitres/ch01_introduction/01_fiche.md#15--générateurs
[s153]: ../chapitres/ch25_autoencodeurs/01_fiche.md#253--le-plus-simple-des-autoencodeurs-
[s154]: ../chapitres/ch26_rl/01_fiche.md#2637--les-récompenses-
[s155]: ../chapitres/ch26_rl/01_fiche.md#2662--la-politique-du-q-learning-
[s156]: ../chapitres/ch11_raisonnement/01_fiche.md#au-delà-du-livre--le-bandit-manchot
[s157]: ../chapitres/ch20_deep_learning/01_fiche.md#202--le-deep-learning-en-bref-
[s158]: ../chapitres/ch12_preparation/01_fiche.md#122--la-règle-dor--apprendre-la-transformation-sur-lentraînement-
[s159]: ../chapitres/b4_llm/01_fiche.md#les-bases-de-données-vectorielles
[s160]: ../chapitres/ch01_introduction/01_fiche.md#les-quatre-fils-rouges-du-workbook
[s161]: ../chapitres/ch29_datasets/01_fiche.md#au-delà-du-livre--documenter-et-télécharger
[s162]: ../chapitres/ch06_information/01_fiche.md#au-delà-du-livre-2--le-contexte-réduit-la-surprise
[s163]: ../chapitres/b4_llm/01_fiche.md#2041--panorama--modèles-de-base-instruct-et-chat--tailles-licences-hub--ce-que-fait-vraiment-un-llm-
[s164]: ../chapitres/b8_rl_moderne/01_fiche.md#le-rlhf--ppo-sous-une-laisse-kl
[s165]: ../chapitres/b1_transfer/01_fiche.md#2018--modèles-de-fondation--auto-supervisé-dinov2-texte-image-clip-zéro-shot-
[s166]: ../chapitres/b5_diffusion/01_fiche.md
[s167]: ../chapitres/ch01_introduction/01_fiche.md#11--le-machine-learning--extraire-du-sens-des-données-
[s168]: ../chapitres/ch08_train_test/01_fiche.md#84--le-jeu-de-validation-
[s169]: ../chapitres/ch13_classifieurs/01_fiche.md#135--les-arbres-de-décision-
[s170]: ../chapitres/ch07_classification/01_fiche.md#sans-labels--k-means
[s171]: ../chapitres/ch24_pytorch_2/01_fiche.md#246--lapi-fonctionnelle-
[s172]: ../chapitres/b4_llm/01_fiche.md#2047--rag--récupérer-augmenter-le-prompt-générer-citer--évaluer-et-diagnostiquer-
[s173]: ../chapitres/b4_llm/01_fiche.md#lora
[s174]: ../chapitres/b4_llm/01_fiche.md#les-hallucinations
[s175]: ../chapitres/ch02_stats/01_fiche.md#résumer-une-liste-de-nombres--moyenne-médiane-mode
[s176]: ../chapitres/ch02_stats/01_fiche.md#des-comptages-à-une-distribution-de-probabilité
[s177]: ../chapitres/ch12_preparation/01_fiche.md#125--normaliser-et-standardiser-
[s178]: ../chapitres/ch02_stats/01_fiche.md
[s179]: ../chapitres/ch02_stats/01_fiche.md#variable-aléatoire-tirage-loi-discrète-et-densité
[s180]: ../chapitres/ch02_stats/01_fiche.md#221--nombres-aléatoires-en-pratique
[s181]: ../chapitres/ch02_stats/01_fiche.md#231--la-loi-uniforme
[s182]: ../chapitres/ch02_stats/01_fiche.md#232--la-loi-normale-
[s183]: ../chapitres/ch02_stats/01_fiche.md#233--la-loi-de-bernoulli
[s184]: ../chapitres/ch02_stats/01_fiche.md#234--la-loi-catégorielle-multinoulli
[s185]: ../chapitres/ch02_stats/01_fiche.md#mesurer-la-dispersion-de-données--variance-écart-type-ddof
[s186]: ../chapitres/ch02_stats/01_fiche.md#241--des-variables-iid
[s187]: ../chapitres/ch02_stats/01_fiche.md#25--tirages-avec-et-sans-remise-
[s188]: ../chapitres/ch11_raisonnement/01_fiche.md#115--linduction-
[s189]: ../chapitres/ch02_stats/01_fiche.md#26--le-bootstrap-
[s190]: ../chapitres/ch14_ensembles/01_fiche.md#144--le-bagging-
[s191]: ../chapitres/b7_production/01_fiche.md#rapporter-un-score--plusieurs-graines-et-un-intervalle
[s192]: ../chapitres/ch07_classification/01_fiche.md#76--la-malédiction-de-la-dimension-
[s193]: ../chapitres/ch02_stats/01_fiche.md#27--espaces-de-grande-dimension
[s194]: ../chapitres/ch02_stats/01_fiche.md#29--le-quartet-danscombe
[s195]: ../chapitres/ch14_ensembles/01_fiche.md#au-delà-du-livre--le-gradient-boosting
[s196]: ../chapitres/ch02_stats/01_fiche.md#28--covariance-et-corrélation-
[s197]: ../chapitres/ch02_stats/01_fiche.md#282--la-corrélation-
[s198]: ../chapitres/ch02_stats/01_fiche.md#matrices-de-covariance-et-de-corrélation
[s199]: ../chapitres/ch12_preparation/01_fiche.md#1271--lanalyse-en-composantes-principales-pca-
[s200]: ../chapitres/ch03_probabilites/01_fiche.md#34--probabilité-conditionnelle
[s201]: ../chapitres/ch03_probabilites/01_fiche.md#35--probabilité-jointe
[s202]: ../chapitres/ch04_bayes/01_fiche.md#441--la-règle-de-bayes-
[s203]: ../chapitres/ch03_probabilites/01_fiche.md#36--probabilité-marginale-et-tables-de-contingence
[s204]: ../chapitres/ch03_probabilites/01_fiche.md#371--classer-des-échantillons-
[s205]: ../chapitres/ch03_probabilites/01_fiche.md#379--les-autres-mesures-
[s206]: ../chapitres/ch03_probabilites/01_fiche.md#38--appliquer-la-matrice-de-confusion--le-piège-de-la-prévalence-
[s207]: ../chapitres/b6_explicabilite/01_fiche.md#limpossibilité
[s208]: ../chapitres/ch03_probabilites/01_fiche.md#3711--le-score-f1-
[s209]: ../chapitres/ch07_classification/01_fiche.md#741--un-contre-tous-ovr-
[s210]: ../chapitres/ch13_classifieurs/01_fiche.md#134--les-machines-à-vecteurs-de-support-
[s211]: ../chapitres/ch15_sklearn/01_fiche.md#au-delà-du-livre--seuil-de-décision-calibration-et-classes-déséquilibrées
[s212]: ../chapitres/ch03_probabilites/01_fiche.md#au-delà-du-livre-1--la-courbe-roc-et-lauc
[s213]: ../chapitres/ch03_probabilites/01_fiche.md#au-delà-du-livre-2--la-courbe-precision-recall-et-laverage-precision
[s214]: ../chapitres/ch03_probabilites/01_fiche.md#au-delà-du-livre-3--la-calibration-des-probabilités
[s215]: ../chapitres/ch03_probabilites/01_fiche.md
[s216]: ../chapitres/ch04_bayes/01_fiche.md#42--fréquentiste-et-bayésien--deux-façons-de-penser-les-probabilités
[s217]: ../chapitres/ch04_bayes/01_fiche.md#47--plusieurs-hypothèses-
[s218]: ../chapitres/ch04_bayes/01_fiche.md#442--retrouver-la-règle-et-nommer-ses-termes-
[s219]: ../chapitres/ch04_bayes/01_fiche.md#43--lancer-des-pièces
[s220]: ../chapitres/ch04_bayes/01_fiche.md#46--répéter-la-règle-de-bayes-
[s221]: ../chapitres/ch04_bayes/01_fiche.md
[s222]: ../chapitres/ch04_bayes/01_fiche.md#au-delà-du-livre-2--la-limite-continue-la-loi-beta
[s223]: ../chapitres/ch00a_python/01_fiche.md#10066--itertools-et-heapq-aperçu
[s224]: ../chapitres/ch05_courbes/01_fiche.md#52--introduction--fonctions-courbes-et-surfaces
[s225]: ../chapitres/ch05_courbes/01_fiche.md#extrema-globaux-et-locaux
[s226]: ../chapitres/ch05_courbes/01_fiche.md#tangente-et-dérivée
[s227]: ../chapitres/ch05_courbes/01_fiche.md#au-delà-du-livre-1--calculer-une-dérivée-sur-ordinateur
[s228]: ../chapitres/ch05_courbes/01_fiche.md#au-delà-du-livre-2--les-arrondis-et-le-choix-du-pas
[s229]: ../chapitres/ch05_courbes/01_fiche.md
[s230]: ../chapitres/ch05_courbes/01_fiche.md#les-points-où-le-gradient-sannule
[s231]: ../chapitres/ch05_courbes/01_fiche.md#descendre-une-surface-pas-à-pas
[s232]: ../chapitres/ch09_overfitting/01_fiche.md#les-moindres-carrés-au-delà-du-livre
[s233]: ../chapitres/ch05_courbes/01_fiche.md#au-delà-du-livre-3--la-différentiation-automatique
[s234]: ../chapitres/ch20_deep_learning/01_fiche.md#2021--les-tenseurs-
[s235]: ../chapitres/ch17_activations/01_fiche.md#175--les-fonctions-linéaires-par-morceaux-
[s236]: ../chapitres/ch06_information/01_fiche.md#information--un-mot-deux-sens
[s237]: ../chapitres/ch06_information/01_fiche.md#63--le-bit-une-unité-
[s238]: ../chapitres/ch06_information/01_fiche.md#le-contexte
[s239]: ../chapitres/ch06_information/01_fiche.md#65--la-taille-dun-événement-
[s240]: ../chapitres/ch06_information/01_fiche.md#66--les-codes-adaptatifs-
[s241]: ../chapitres/ch06_information/01_fiche.md#le-code-de-huffman
[s242]: ../chapitres/ch06_information/01_fiche.md#67--lentropie-
[s243]: ../chapitres/ch06_information/01_fiche.md#68--la-cross-entropy-
[s244]: ../chapitres/ch06_information/01_fiche.md#69--la-divergence-kl-
[s245]: ../chapitres/ch27_gan/01_fiche.md#2744--le-jeu-
[s246]: ../chapitres/ch06_information/01_fiche.md#mélanger-les-codes
[s247]: ../chapitres/b2_tokenisation/01_fiche.md#deux-rapports
[s248]: ../chapitres/b3_transformers/01_fiche.md#2038--mini-gpt-from-scratch--données-entraînement-loss-perplexité-
[s249]: ../chapitres/ch22_rnn/01_fiche.md#229--un-exemple--générer-du-holmes-caractère-par-caractère-
[s250]: ../chapitres/ch17_activations/01_fiche.md#178--la-softmax-
[s251]: ../chapitres/b2_tokenisation/01_fiche.md#un-algorithme-de-compression-devenu-tokenizer
[s252]: ../chapitres/ch07_classification/01_fiche.md#73--plusieurs-classes-plusieurs-dimensions-
[s253]: ../chapitres/ch10_neurones/01_fiche.md#1031--le-perceptron-
[s254]: ../chapitres/ch07_classification/01_fiche.md#742--un-contre-un-ovo-
[s255]: ../chapitres/ch15_sklearn/01_fiche.md#154--les-estimateurs-
[s256]: ../chapitres/ch07_classification/01_fiche.md#avec-des-labels--le-centroïde-le-plus-proche
[s257]: ../chapitres/ch13_classifieurs/01_fiche.md#133--les-k-plus-proches-voisins-
[s258]: ../chapitres/ch07_classification/01_fiche.md#le-départ-compte--minima-locaux-et-k-means
[s259]: ../chapitres/ch07_classification/01_fiche.md#combien-de-clusters-
[s260]: ../chapitres/ch07_classification/01_fiche.md#761--bizarreries-de-la-grande-dimension
[s261]: ../chapitres/ch07_classification/01_fiche.md#quand-k-means-échoue-et-le-clustering-par-densité
[s262]: ../chapitres/ch07_classification/01_fiche.md
[s263]: ../chapitres/ch08_train_test/01_fiche.md#851--la-validation-croisée-à-k-folds-
[s264]: ../chapitres/b3_transformers/01_fiche.md#2034--attention-multi-têtes--projections-découpage-en-têtes-concaténation-
[s265]: ../chapitres/ch19_optimiseurs/01_fiche.md#196--choisir-un-optimiseur-
[s266]: ../chapitres/ch08_train_test/01_fiche.md#821--pourquoi-le-score-dentraînement-trompe-
[s267]: ../chapitres/ch08_train_test/01_fiche.md
[s268]: ../chapitres/ch08_train_test/01_fiche.md#données-dépendantes--groupes-et-séries-temporelles-au-delà-du-livre
[s269]: ../chapitres/ch08_train_test/01_fiche.md#86--utiliser-les-résultats-du-test-
[s270]: ../chapitres/b7_production/01_fiche.md#20710--surveiller-en-production--dérive-des-données-et-des-performances-journalisation-retour-arrière-
[s271]: ../chapitres/b6_explicabilite/01_fiche.md#la-fiche-modèle
[s272]: ../chapitres/ch23_pytorch_1/01_fiche.md#2395--sauvegarder-le-prétraitement-
[s273]: ../chapitres/ch09_overfitting/01_fiche.md#94--learly-stopping-
[s274]: ../chapitres/ch23_pytorch_1/01_fiche.md#23103--early-stopping-
[s275]: ../chapitres/ch09_overfitting/01_fiche.md#le-point-isolé
[s276]: ../chapitres/ch09_overfitting/01_fiche.md#95--la-régularisation-
[s277]: ../chapitres/ch09_overfitting/01_fiche.md#la-pénalité-l2--ridge
[s278]: ../chapitres/ch09_overfitting/01_fiche.md#la-pénalité-l1--lasso-et-les-zéros-exacts
[s279]: ../chapitres/ch09_overfitting/01_fiche.md#97--ajuster-une-droite-avec-la-règle-de-bayes
[s280]: ../chapitres/b8_rl_moderne/01_fiche.md#2082--dqn--réseau-q-mémoire-de-rejeu-réseau-cible-loss-de-huber-double-dqn-
[s281]: ../chapitres/ch09_overfitting/01_fiche.md
[s282]: ../chapitres/ch09_overfitting/01_fiche.md#ajuster-une-courbe--complexité-et-capacité
[s283]: ../chapitres/ch09_overfitting/01_fiche.md#courbes-de-validation-et-courbes-dapprentissage-au-delà-du-livre
[s284]: ../chapitres/ch09_overfitting/01_fiche.md#96--biais-et-variance-
[s285]: ../chapitres/ch09_overfitting/01_fiche.md#964--le-compromis-biais-variance-et-ses-limites
[s286]: ../chapitres/ch20_deep_learning/01_fiche.md#2044--la-batchnorm-
[s287]: ../chapitres/ch10_neurones/01_fiche.md#103--les-neurones-artificiels-
[s288]: ../chapitres/ch10_neurones/01_fiche.md#1032--lhistoire-du-perceptron-
[s289]: ../chapitres/ch10_neurones/01_fiche.md
[s290]: ../chapitres/ch18_backprop/01_fiche.md#1821--apprendre-en-tâtonnant-
[s291]: ../chapitres/ch11_raisonnement/01_fiche.md#1121--la-représentation
[s292]: ../chapitres/ch11_raisonnement/01_fiche.md#1122--lévaluation
[s293]: ../chapitres/ch11_raisonnement/01_fiche.md#1123--loptimisation
[s294]: ../chapitres/ch13_classifieurs/01_fiche.md#132--paramétrique-ou-non-paramétrique--
[s295]: ../chapitres/ch11_raisonnement/01_fiche.md#113--déduction-et-induction-
[s296]: ../chapitres/ch11_raisonnement/01_fiche.md#1161--sherlock-holmes--maître-de-la-déduction-
[s297]: ../chapitres/ch11_raisonnement/01_fiche.md#114--la-déduction-
[s298]: ../chapitres/ch11_raisonnement/01_fiche.md#1141--les-sophismes-syllogistiques
[s299]: ../chapitres/ch11_raisonnement/01_fiche.md#1152--les-sophismes-inductifs-
[s300]: ../chapitres/ch11_raisonnement/01_fiche.md#116--raisonnement-combiné-
[s301]: ../chapitres/ch11_raisonnement/01_fiche.md#117--le-conditionnement-opérant-
[s302]: ../chapitres/ch12_preparation/01_fiche.md#121--pourquoi-ce-chapitre-
[s303]: ../chapitres/ch12_preparation/01_fiche.md#123--les-types-de-données-
[s304]: ../chapitres/ch12_preparation/01_fiche.md#124--le-nettoyage-de-base-
[s305]: ../chapitres/ch12_preparation/01_fiche.md#1251--la-normalisation-min-max-
[s306]: ../chapitres/ch12_preparation/01_fiche.md#1252--la-standardisation-
[s307]: ../chapitres/ch12_preparation/01_fiche.md#1253--retenir-la-transformation-
[s308]: ../chapitres/ch12_preparation/01_fiche.md#1254--univariée-ou-multivariée-
[s309]: ../chapitres/ch12_preparation/01_fiche.md#129--par-échantillon-par-feature-par-élément
[s310]: ../chapitres/ch12_preparation/01_fiche.md#126--la-sélection-de-features-
[s311]: ../chapitres/b2_tokenisation/01_fiche.md#2028--embeddings-distributionnels--co-occurrences-ppmi-svd-word2vec-
[s312]: ../chapitres/ch12_preparation/01_fiche.md#au-delà-du-livre--voir-des-données-en-grande-dimension
[s313]: ../chapitres/ch12_preparation/01_fiche.md#au-delà-du-livre--assembler-les-données-avec-pandas-et-sql
[s314]: ../chapitres/ch13_classifieurs/01_fiche.md#137--bilan--choisir-un-classifieur-
[s315]: ../chapitres/ch13_classifieurs/01_fiche.md
[s316]: ../chapitres/ch14_ensembles/01_fiche.md#147--le-boosting-
[s317]: ../chapitres/ch13_classifieurs/01_fiche.md#1353--limiter-loverfitting-
[s318]: ../chapitres/ch14_ensembles/01_fiche.md#145--les-forêts-aléatoires-
[s319]: ../chapitres/ch04_bayes/01_fiche.md#bayes-dans-le-ml-daujourdhui
[s320]: ../projets/partie_1_detecteur_langue/README.md
[s321]: ../chapitres/b6_explicabilite/01_fiche.md#corrélation-nest-pas-causalité
[s322]: ../chapitres/b6_explicabilite/01_fiche.md#interprétable-par-construction-ou-expliqué-après-coup
[s323]: ../chapitres/ch14_ensembles/01_fiche.md#143--le-vote-
[s324]: ../chapitres/ch14_ensembles/01_fiche.md
[s325]: ../chapitres/ch14_ensembles/01_fiche.md#146--extratrees-
[s326]: ../chapitres/b6_explicabilite/01_fiche.md#limportance-par-permutation
[s327]: ../chapitres/ch15_sklearn/01_fiche.md#1542--apprendre-avec-fit-
[s328]: ../chapitres/ch15_sklearn/01_fiche.md#1593--la-recherche-en-grille-
[s329]: ../chapitres/ch24_pytorch_2/01_fiche.md#2434--la-recherche-dhyperparamètres-
[s330]: ../chapitres/ch15_sklearn/01_fiche.md#1594--la-recherche-aléatoire
[s331]: ../chapitres/ch22_rnn/01_fiche.md#2272--le-lstm-
[s332]: ../chapitres/ch15_sklearn/01_fiche.md#1592--chercher-des-hyperparamètres-
[s333]: ../chapitres/ch15_sklearn/01_fiche.md#au-delà-du-livre--brancher-ton-propre-estimateur
[s334]: ../chapitres/ch15_sklearn/01_fiche.md#au-delà-du-livre--les-données-hétérogènes
[s335]: ../chapitres/ch15_sklearn/01_fiche.md#au-delà-du-livre--sauvegarder-un-modèle-et-ses-versions
[s336]: ../chapitres/ch22_rnn/01_fiche.md#au-delà-du-livre--prévoir-une-série-temporelle
[s337]: ../chapitres/ch16_feedforward/01_fiche.md#161--pourquoi-ce-chapitre--
[s338]: ../chapitres/ch16_feedforward/01_fiche.md#162--les-graphes-de-réseaux-de-neurones-
[s339]: ../chapitres/ch16_feedforward/01_fiche.md#163--flux-synchrone-et-asynchrone
[s340]: ../chapitres/ch16_feedforward/01_fiche.md#1641--linitialisation-
[s341]: ../chapitres/ch16_feedforward/01_fiche.md
[s342]: ../chapitres/ch16_feedforward/01_fiche.md#au-delà-du-livre--diagnostiquer-lévanouissement-et-lexplosion-des-activations
[s343]: ../chapitres/ch17_activations/01_fiche.md#1721--la-forme-dune-activation
[s344]: ../chapitres/ch27_gan/01_fiche.md#2772--leffondrement-de-mode
[s345]: ../chapitres/ch17_activations/01_fiche.md#174--les-marches-
[s346]: ../chapitres/ch18_backprop/01_fiche.md#18124--garder-les-neurones-en-vie
[s347]: ../chapitres/b6_explicabilite/01_fiche.md#le-problème-de-la-saturation
[s348]: ../chapitres/ch17_activations/01_fiche.md#au-delà-du-livre--choisir-une-activation-et-tester-une-famille-de-fonctions
[s349]: ../chapitres/ch18_backprop/01_fiche.md#186--étape-1--les-deltas-des-neurones-de-sortie-
[s350]: ../chapitres/ch16_feedforward/01_fiche.md#au-delà-du-livre--la-passe-avant-en-matrices
[s351]: ../chapitres/ch18_backprop/01_fiche.md#184--sorties-des-neurones-et-erreur-du-réseau-
[s352]: ../chapitres/ch18_backprop/01_fiche.md#188--étape-3--les-deltas-des-autres-neurones-
[s353]: ../chapitres/ch22_rnn/01_fiche.md#226--entraîner-un-rnn-
[s354]: ../chapitres/ch18_backprop/01_fiche.md#au-delà-du-livre--les-losses-calculées-sur-les-logits
[s355]: ../chapitres/ch18_backprop/01_fiche.md
[s356]: ../chapitres/ch18_backprop/01_fiche.md#au-delà-du-livre--un-micro-autograd
[s357]: ../chapitres/ch19_optimiseurs/01_fiche.md#1942--la-descente-stochastique-sgd-
[s358]: ../chapitres/ch19_optimiseurs/01_fiche.md#1933--les-plannings-de-décroissance-
[s359]: ../chapitres/ch19_optimiseurs/01_fiche.md#1932--changer-le-learning-rate-au-fil-du-temps-
[s360]: ../chapitres/ch19_optimiseurs/01_fiche.md#au-delà-du-livre--warmup-et-décroissance-en-cosinus
[s361]: ../chapitres/b1_transfer/01_fiche.md#warmup-puis-cosinus-groupe-par-groupe
[s362]: ../chapitres/ch19_optimiseurs/01_fiche.md#1951--le-momentum-
[s363]: ../chapitres/ch19_optimiseurs/01_fiche.md#1952--le-momentum-de-nesterov-
[s364]: ../chapitres/ch19_optimiseurs/01_fiche.md#195--les-variantes-de-la-descente-de-gradient-
[s365]: ../chapitres/ch19_optimiseurs/01_fiche.md#1953--adagrad-
[s366]: ../chapitres/ch19_optimiseurs/01_fiche.md#1954--adadelta-et-rmsprop-
[s367]: ../chapitres/ch19_optimiseurs/01_fiche.md#1955--adam-
[s368]: ../chapitres/ch19_optimiseurs/01_fiche.md#au-delà-du-livre--adamw-le-weight-decay-découplé
[s369]: ../chapitres/ch19_optimiseurs/01_fiche.md#au-delà-du-livre--le-gradient-clipping
[s370]: ../chapitres/ch20_deep_learning/01_fiche.md#208--interpréter-les-résultats-
[s371]: ../chapitres/ch23_pytorch_1/01_fiche.md#2322--programmer-et-déboguer-en-python
[s372]: ../chapitres/ch20_deep_learning/01_fiche.md#203--les-couches-dentrée-et-de-sortie
[s373]: ../chapitres/ch20_deep_learning/01_fiche.md#2032--la-couche-de-sortie-
[s374]: ../chapitres/ch20_deep_learning/01_fiche.md#2048--les-autres-couches-utilitaires-
[s375]: ../chapitres/ch25_autoencodeurs/01_fiche.md#257--autoencodeurs-convolutifs-
[s376]: ../chapitres/ch20_deep_learning/01_fiche.md
[s377]: ../chapitres/ch20_deep_learning/01_fiche.md#au-delà-du-livre--les-connexions-résiduelles
[s378]: ../chapitres/b3_transformers/01_fiche.md#deux-sous-couches
[s379]: ../chapitres/b6_explicabilite/01_fiche.md#avec-pytorch--deux-hooks
[s380]: ../chapitres/ch23_pytorch_1/01_fiche.md#23101--checkpoints-
[s381]: ../chapitres/ch20_deep_learning/01_fiche.md#au-delà-du-livre--régulariser-un-réseau
[s382]: ../chapitres/b6_explicabilite/01_fiche.md
[s383]: ../chapitres/ch20_deep_learning/01_fiche.md#2081--une-explication-satisfaisante-
[s384]: ../chapitres/b6_explicabilite/01_fiche.md#deux-autres-explications-locales--substitut-et-contrefactuel
[s385]: ../chapitres/b6_explicabilite/01_fiche.md#le-rgpd-article-22
[s386]: ../chapitres/ch21_cnn/01_fiche.md#213--la-convolution-
[s387]: ../chapitres/ch21_cnn/01_fiche.md#2123--le-partage-de-poids-
[s388]: ../chapitres/ch21_cnn/01_fiche.md#2124--le-champ-récepteur-local
[s389]: ../chapitres/ch21_cnn/01_fiche.md#217--une-couche-de-convolution-
[s390]: ../chapitres/ch28_creatif/01_fiche.md#2821--choisir-un-réseau-
[s391]: ../chapitres/ch21_cnn/01_fiche.md#2131--les-filtres-
[s392]: ../chapitres/ch21_cnn/01_fiche.md#218--la-convolution-transposée-
[s393]: ../chapitres/ch21_cnn/01_fiche.md#214--la-convolution-en-plusieurs-dimensions
[s394]: ../chapitres/ch21_cnn/01_fiche.md#216--les-convolutions-1--1
[s395]: ../chapitres/ch21_cnn/01_fiche.md#215--la-convolution-1d
[s396]: ../chapitres/ch21_cnn/01_fiche.md#2132--la-vue--œil-de-mouche-
[s397]: ../chapitres/ch21_cnn/01_fiche.md#2142--le-stride-pour-construire-une-hiérarchie-
[s398]: ../chapitres/ch21_cnn/01_fiche.md#2192--regarder-les-filtres-1--ce-qui-les-excite-le-plus
[s399]: ../chapitres/ch28_creatif/01_fiche.md
[s400]: ../chapitres/b6_explicabilite/01_fiche.md#2066--grad-cam--cartes-dactivation-pondérées-par-les-gradients-
[s401]: ../chapitres/ch21_cnn/01_fiche.md#2193--regarder-les-filtres-2--les-cartes-dactivation-
[s402]: ../chapitres/ch21_cnn/01_fiche.md#2110--les-adversaires-
[s403]: ../chapitres/ch21_cnn/01_fiche.md#au-delà-du-livre--lattaque-fgsm
[s404]: ../chapitres/b3_transformers/01_fiche.md#le-vision-transformer
[s405]: ../chapitres/ch21_cnn/01_fiche.md#2191--vgg16-
[s406]: ../chapitres/ch20_deep_learning/01_fiche.md#2047--les-couches-récurrentes-aperçu
[s407]: ../chapitres/ch22_rnn/01_fiche.md#225--organiser-les-entrées-
[s408]: ../chapitres/ch22_rnn/01_fiche.md#222--introduction--des-fenêtres-pour-une-couche-dense-
[s409]: ../chapitres/b3_transformers/01_fiche.md#les-autres-voies-vers-les-contextes-longs
[s410]: ../chapitres/ch22_rnn/01_fiche.md#224--la-structure-dune-cellule-rnn-
[s411]: ../chapitres/ch22_rnn/01_fiche.md#223--létat-
[s412]: ../chapitres/ch22_rnn/01_fiche.md#2241--une-cellule-à-plus-détat-
[s413]: ../chapitres/ch22_rnn/01_fiche.md
[s414]: ../chapitres/ch22_rnn/01_fiche.md#2271--les-portes
[s415]: ../chapitres/ch22_rnn/01_fiche.md#227--lstm-et-gru
[s416]: ../chapitres/ch22_rnn/01_fiche.md#2281--une-ou-plusieurs-entrées-et-sorties-
[s417]: ../chapitres/ch22_rnn/01_fiche.md#2282--le-rnn-profond-
[s418]: ../chapitres/ch28_creatif/01_fiche.md#285--générer-la-suite-du-livre
[s419]: ../chapitres/b3_transformers/01_fiche.md#2039--génération--kv-cache-et-échantillonnage-température-top-k-top-p-
[s420]: ../chapitres/ch22_rnn/01_fiche.md#au-delà-du-livre--après-les-rnn-transformers-et-modèles-à-espace-détats
[s421]: ../chapitres/ch23_pytorch_1/01_fiche.md#2313--les-avertissements-de-python-
[s422]: ../chapitres/ch23_pytorch_1/01_fiche.md#2333--installer-et-configurer-la-bibliothèque-
[s423]: ../chapitres/ch23_pytorch_1/01_fiche.md#2351--remodeler-
[s424]: ../chapitres/ch23_pytorch_1/01_fiche.md#2332--tenseurs-et-tableaux-
[s425]: ../chapitres/ch23_pytorch_1/01_fiche.md#2334--la-forme-des-tenseurs-dimages
[s426]: ../chapitres/ch23_pytorch_1/01_fiche.md#2383--analyser-lhistorique-
[s427]: ../chapitres/ch23_pytorch_1/01_fiche.md#2381--lire-la-sortie-
[s428]: ../chapitres/ch23_pytorch_1/01_fiche.md#2310--callbacks-
[s429]: ../chapitres/ch23_pytorch_1/01_fiche.md#2394--utiliser-un-modèle-pré-entraîné
[s430]: ../chapitres/b1_transfer/01_fiche.md#2012--trouver-et-charger-un-modèle-pré-entraîné--torchvision-hugging-face-hub-fiches-modèles-licences-
[s431]: ../chapitres/b1_transfer/01_fiche.md#des-données-et-du-calcul-que-tu-nas-pas
[s432]: ../chapitres/ch23_pytorch_1/01_fiche.md
[s433]: ../chapitres/b1_transfer/01_fiche.md#backbone-et-tête
[s434]: ../chapitres/ch23_pytorch_1/01_fiche.md#2392--les-poids-seuls-
[s435]: ../chapitres/ch24_pytorch_2/01_fiche.md#2426--ajouter-du-dropout-
[s436]: ../chapitres/ch24_pytorch_2/01_fiche.md#2427--observations-
[s437]: ../chapitres/b7_production/01_fiche.md#2075--suivi-dexpériences--runs-métriques-artefacts-comparaison-
[s438]: ../chapitres/ch24_pytorch_2/01_fiche.md#2422--changer-un-seul-hyperparamètre-
[s439]: ../chapitres/ch24_pytorch_2/01_fiche.md#2447--des-données-synthétiques
[s440]: ../chapitres/ch24_pytorch_2/01_fiche.md#2446--laugmentation-de-données-
[s441]: ../chapitres/ch24_pytorch_2/01_fiche.md#2443--les-couches-de-convolution-
[s442]: ../chapitres/ch24_pytorch_2/01_fiche.md#2459--les-rnn--stateful-
[s443]: ../chapitres/ch24_pytorch_2/01_fiche.md#24510--une-couche-appliquée-à-chaque-pas-de-temps-
[s444]: ../chapitres/ch24_pytorch_2/01_fiche.md#au-delà-du-livre--précision-mixte-et-compilation
[s445]: ../chapitres/b4_llm/01_fiche.md#la-mémoire-des-poids
[s446]: ../chapitres/b7_production/01_fiche.md#accélérer-sur-cpu
[s447]: ../chapitres/ch25_autoencodeurs/01_fiche.md#252--introduction--comprimer
[s448]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2551--les-variables-latentes-de-près
[s449]: ../chapitres/ch25_autoencodeurs/01_fiche.md#au-delà-du-livre--le-compromis-β-les-anomalies-et-les-autoencodeurs-aujourdhui
[s450]: ../chapitres/b5_diffusion/01_fiche.md#la-diffusion-latente
[s451]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2521--avec-ou-sans-perte
[s452]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2523--mélanger-deux-représentations-
[s453]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2553--mélanger-des-variables-latentes-
[s454]: ../chapitres/ch25_autoencodeurs/01_fiche.md#258--débruiter-
[s455]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2592--la-structure-dun-vae-
[s456]: ../chapitres/ch25_autoencodeurs/01_fiche.md#2554--une-entrée-jamais-vue-
[s457]: ../chapitres/b5_diffusion/01_fiche.md#transparence-et-provenance-des-contenus-générés
[s458]: ../chapitres/ch26_rl/01_fiche.md#2631--premier-temps--lagent-choisit-une-action-
[s459]: ../chapitres/ch26_rl/01_fiche.md#263--la-structure-du-rl-
[s460]: ../chapitres/ch26_rl/01_fiche.md#2634--des-variantes-de-la-version-simple-
[s461]: ../chapitres/ch26_rl/01_fiche.md#2635--la-vue-densemble-
[s462]: ../chapitres/ch26_rl/01_fiche.md#2661--les-valeurs-q-et-leur-mise-à-jour-
[s463]: ../chapitres/ch26_rl/01_fiche.md#265--le-l-learning
[s464]: ../chapitres/ch26_rl/01_fiche.md#266--le-q-learning
[s465]: ../chapitres/b8_rl_moderne/01_fiche.md#lerreur-td-première-estimation-de-lavantage
[s466]: ../chapitres/ch26_rl/01_fiche.md#267--sarsa-
[s467]: ../chapitres/ch26_rl/01_fiche.md#2664--léléphant-dans-la-pièce-
[s468]: ../chapitres/ch26_rl/01_fiche.md#268--la-vue-densemble-
[s469]: ../chapitres/ch26_rl/01_fiche.md#269--le-rejeu-dexpérience-
[s470]: ../chapitres/ch26_rl/01_fiche.md#2610--deux-applications
[s471]: ../chapitres/ch26_rl/01_fiche.md#au-delà-du-livre--le-rl-et-les-modèles-de-langage
[s472]: ../chapitres/b8_rl_moderne/01_fiche.md#le-piratage-de-la-récompense
[s473]: ../chapitres/b8_rl_moderne/01_fiche.md#2083--gradient-de-politique--politique-paramétrée-reinforce-baseline-variance-
[s474]: ../chapitres/ch27_gan/01_fiche.md#274--implémenter-un-gan-
[s475]: ../chapitres/ch27_gan/01_fiche.md#2741--le-discriminateur-
[s476]: ../chapitres/ch27_gan/01_fiche.md#2743--entraîner-le-gan-
[s477]: ../chapitres/ch27_gan/01_fiche.md#276--dcgan-
[s478]: ../chapitres/ch27_gan/01_fiche.md#275--un-gan-en-action-
[s479]: ../chapitres/ch27_gan/01_fiche.md#2771--les-grandes-images-
[s480]: ../chapitres/ch27_gan/01_fiche.md#au-delà-du-livre--le-gan-conditionnel
[s481]: ../chapitres/ch27_gan/01_fiche.md#au-delà-du-livre--mesurer-un-générateur
[s482]: ../chapitres/b5_diffusion/01_fiche.md#la-distance-de-fréchet
[s483]: ../chapitres/ch27_gan/01_fiche.md#277--les-difficultés
[s484]: ../chapitres/b5_diffusion/01_fiche.md#fidélité-et-diversité
[s485]: ../chapitres/ch28_creatif/01_fiche.md#2822--visualiser-un-filtre-
[s486]: ../chapitres/ch28_creatif/01_fiche.md#au-delà-du-livre--la-génération-dimages-aujourdhui
[s487]: ../chapitres/ch28_creatif/01_fiche.md#283--le-deep-dream
[s488]: ../chapitres/ch28_creatif/01_fiche.md#284--le-transfert-de-style-
[s489]: ../chapitres/ch28_creatif/01_fiche.md#2841--le-style-dans-une-matrice-
[s490]: ../chapitres/ch28_creatif/01_fiche.md#2843--la-loss-de-contenu
[s491]: ../chapitres/ch28_creatif/01_fiche.md#2844--la-loss-de-style-
[s492]: ../chapitres/ch28_creatif/01_fiche.md#2842--vue-densemble-
[s493]: ../chapitres/ch28_creatif/01_fiche.md#2845--réaliser-le-transfert-
[s494]: ../chapitres/ch28_creatif/01_fiche.md#2846--discussion-
[s495]: ../chapitres/b4_llm/01_fiche.md#licences-et-droit-dauteur
[s496]: ../chapitres/ch29_datasets/01_fiche.md#au-delà-du-livre--auditer-un-dataset
[s497]: ../chapitres/ch29_datasets/01_fiche.md#au-delà-du-livre--le-décalage-de-distribution
[s498]: ../chapitres/ch29_datasets/01_fiche.md#294--les-collections-de-datasets-
[s499]: ../chapitres/ch29_datasets/01_fiche.md
[s500]: ../chapitres/ch29_datasets/01_fiche.md#295--des-datasets-plus-récents-
[s501]: ../chapitres/b1_transfer/01_fiche.md#2013--extraction-de-features--embeddings-figés-linear-probe-k-plus-proches-voisins-
[s502]: ../chapitres/b1_transfer/01_fiche.md#calculer-les-embeddings-une-fois
[s503]: ../chapitres/b1_transfer/01_fiche.md#le-linear-probe
[s504]: ../chapitres/b1_transfer/01_fiche.md#learning-rates-discriminatifs
[s505]: ../chapitres/b1_transfer/01_fiche.md#le-dégel-progressif
[s506]: ../chapitres/b1_transfer/01_fiche.md#pourquoi-un-learning-rate-beaucoup-plus-petit
[s507]: ../chapitres/b1_transfer/01_fiche.md#rester-près-des-poids-pré-entraînés--la-pénalité-l2-sp
[s508]: ../chapitres/b1_transfer/01_fiche.md#loubli-catastrophique
[s509]: ../chapitres/b1_transfer/01_fiche.md#la-distance-de-domaine
[s510]: ../chapitres/b1_transfer/01_fiche.md#le-transfert-négatif
[s511]: ../chapitres/b1_transfer/01_fiche.md#le-few-shot-et-les-prototypes-de-classe
[s512]: ../chapitres/b1_transfer/01_fiche.md#le-zéro-shot
[s513]: ../chapitres/b4_llm/01_fiche.md#zéro-shot-et-few-shot
[s514]: ../chapitres/b1_transfer/01_fiche.md#la-courbe-dapprentissage-selon-la-taille-du-train
[s515]: ../chapitres/b1_transfer/01_fiche.md#clip--rapprocher-les-images-de-leur-légende
[s516]: ../chapitres/b7_production/01_fiche.md#latence-percentiles-et-débit
[s517]: ../chapitres/b1_transfer/01_fiche.md#2019--coût-dusage--taille-latence-cpu-choix-dun-backbone-pour-la-production-
[s518]: ../chapitres/b1_transfer/01_fiche.md#quatre-mesures-à-ne-pas-confondre
[s519]: ../chapitres/b1_transfer/01_fiche.md#publier-un-modèle-dérivé
[s520]: ../chapitres/b1_transfer/01_fiche.md
[s521]: ../chapitres/b2_tokenisation/01_fiche.md#un-modèle-ne-lit-que-des-identifiants
[s522]: ../chapitres/b2_tokenisation/01_fiche.md#points-de-code-et-octets
[s523]: ../chapitres/b2_tokenisation/01_fiche.md#2022--unicode-utf-8-et-normalisation-
[s524]: ../chapitres/b2_tokenisation/01_fiche.md#un-caractère-affiché-plusieurs-points-de-code
[s525]: ../chapitres/b2_tokenisation/01_fiche.md#la-pré-tokenisation
[s526]: ../chapitres/b2_tokenisation/01_fiche.md#2023--byte-pair-encoding--apprendre-et-appliquer-les-fusions-
[s527]: ../chapitres/b2_tokenisation/01_fiche.md#les-tokens-spéciaux
[s528]: ../chapitres/b2_tokenisation/01_fiche.md#2025--wordpiece-unigram-sentencepiece--la-bibliothèque-tokenizers-et-les-tokenizers-des-modèles-
[s529]: ../chapitres/b2_tokenisation/01_fiche.md#unigram--partir-de-trop-et-élaguer
[s530]: ../chapitres/b2_tokenisation/01_fiche.md#sentencepiece--sans-pré-tokenisation
[s531]: ../chapitres/b2_tokenisation/01_fiche.md#2026--coûts-et-biais-de-la-tokenisation--fertilité-contexte-prix-chiffres-
[s532]: ../chapitres/b2_tokenisation/01_fiche.md#utiliser-un-tokenizer-avec-transformers
[s533]: ../chapitres/b2_tokenisation/01_fiche.md#les-poids-liés
[s534]: ../chapitres/b2_tokenisation/01_fiche.md#les-tokenizers-des-modèles-lus-dans-leur-tokenizerjson
[s535]: ../chapitres/b2_tokenisation/01_fiche.md#lhypothèse-distributionnelle
[s536]: ../chapitres/b2_tokenisation/01_fiche.md#la-matrice-de-co-occurrences
[s537]: ../chapitres/b2_tokenisation/01_fiche.md#pmi-et-ppmi
[s538]: ../chapitres/b2_tokenisation/01_fiche.md
[s539]: ../chapitres/b2_tokenisation/01_fiche.md#réduire--la-svd-tronquée
[s540]: ../chapitres/b2_tokenisation/01_fiche.md#word2vec--prédire-plutôt-que-compter
[s541]: ../chapitres/b2_tokenisation/01_fiche.md#compter-ou-prédire--les-deux-se-rejoignent
[s542]: ../chapitres/b2_tokenisation/01_fiche.md#les-analogies
[s543]: ../chapitres/b2_tokenisation/01_fiche.md#20210--embeddings-contextuels-et-embeddings-de-phrases--recherche-sémantique-
[s544]: ../chapitres/b2_tokenisation/01_fiche.md#dune-suite-de-vecteurs-à-un-vecteur-de-phrase
[s545]: ../chapitres/b2_tokenisation/01_fiche.md#deux-modèles-dembedding-de-phrases
[s546]: ../chapitres/b2_tokenisation/01_fiche.md#une-recherche-sémantique-en-vingt-lignes
[s547]: ../chapitres/b4_llm/01_fiche.md#la-recherche-dense
[s548]: ../chapitres/b2_tokenisation/01_fiche.md#un-numéro-nest-pas-une-représentation
[s549]: ../chapitres/b2_tokenisation/01_fiche.md#les-biais-des-embeddings
[s550]: ../chapitres/b3_transformers/01_fiche.md#lencodeur-décodeur--t5-et-lattention-croisée
[s551]: ../chapitres/b3_transformers/01_fiche.md#traduire-avec-deux-rnn
[s552]: ../chapitres/b3_transformers/01_fiche.md#2031--des-rnn-à-lattention--goulot-de-lencodeur-décodeur-et-alignement-
[s553]: ../chapitres/b3_transformers/01_fiche.md#lidée-de-bahdanau--relire-la-source-à-chaque-pas
[s554]: ../chapitres/b3_transformers/01_fiche.md#2032--attention-par-produit-scalaire--requêtes-clés-valeurs-mise-à-léchelle-softmax-
[s555]: ../chapitres/b3_transformers/01_fiche.md#la-formule
[s556]: ../chapitres/b3_transformers/01_fiche.md#auto-attention-et-attention-croisée
[s557]: ../chapitres/b3_transformers/01_fiche.md#2033--masques--causal-padding--auto-attention-et-attention-croisée-
[s558]: ../chapitres/b3_transformers/01_fiche.md#le-masque-causal
[s559]: ../chapitres/b3_transformers/01_fiche.md#le-masque-de-padding
[s560]: ../chapitres/b3_transformers/01_fiche.md#plusieurs-questions-à-la-fois
[s561]: ../chapitres/b3_transformers/01_fiche.md#moins-de-têtes-pour-les-clés--mqa-et-gqa
[s562]: ../chapitres/b3_transformers/01_fiche.md
[s563]: ../chapitres/b3_transformers/01_fiche.md#lencodage-sinusoïdal-2017
[s564]: ../chapitres/b3_transformers/01_fiche.md#les-positions-apprises
[s565]: ../chapitres/b3_transformers/01_fiche.md#rope--tourner-les-requêtes-et-les-clés
[s566]: ../chapitres/b3_transformers/01_fiche.md#pré-ln-ou-post-ln
[s567]: ../chapitres/b3_transformers/01_fiche.md#2036--le-bloc-transformer--résiduel-layer-norm-pré-ou-post-mlp-gelu-
[s568]: ../chapitres/b3_transformers/01_fiche.md#lencodeur--bert-et-le-modèle-de-langue-masqué
[s569]: ../chapitres/b3_transformers/01_fiche.md#teacher-forcing-et-décalage-dun-token
[s570]: ../chapitres/b3_transformers/01_fiche.md#le-kv-cache
[s571]: ../chapitres/b3_transformers/01_fiche.md#le-masque-causal-quand-il-y-a-moins-de-requêtes-que-de-clés
[s572]: ../chapitres/b3_transformers/01_fiche.md#flashattention--le-même-résultat-sans-la-matrice
[s573]: ../chapitres/b3_transformers/01_fiche.md#compter-les-paramètres-dun-bloc
[s574]: ../chapitres/b3_transformers/01_fiche.md#les-lois-déchelle
[s575]: ../chapitres/b3_transformers/01_fiche.md#ce-que-coûte-le-cache
[s576]: ../chapitres/b3_transformers/01_fiche.md#allonger-le-contexte
[s577]: ../chapitres/b3_transformers/01_fiche.md#20310--coût-et-passage-à-léchelle--complexité-quadratique-flashattention-lois-déchelle-vit-
[s578]: ../chapitres/b5_diffusion/01_fiche.md#la-mémorisation
[s579]: ../chapitres/b7_production/01_fiche.md#agir--alerter-réentraîner-revenir-en-arrière
[s580]: ../chapitres/b4_llm/01_fiche.md#trois-étapes-trois-sortes-de-modèles
[s581]: ../chapitres/b4_llm/01_fiche.md#2043--génération--generate-décodage-arrêt-streaming--chat-templates-et-rôles-
[s582]: ../chapitres/b4_llm/01_fiche.md#rôles-et-chat-template
[s583]: ../chapitres/b4_llm/01_fiche.md#licences---open-weights--nest-pas--open-source-
[s584]: ../chapitres/b4_llm/01_fiche.md#sarrêter
[s585]: ../chapitres/b4_llm/01_fiche.md#le-streaming
[s586]: ../chapitres/b4_llm/01_fiche.md#la-sortie-structurée
[s587]: ../chapitres/b4_llm/01_fiche.md#le-raisonnement-pas-à-pas
[s588]: ../chapitres/b4_llm/01_fiche.md#voter--lauto-cohérence
[s589]: ../chapitres/b4_llm/01_fiche.md#les-métriques
[s590]: ../chapitres/b4_llm/01_fiche.md#le-juge-llm
[s591]: ../chapitres/b4_llm/01_fiche.md#découper
[s592]: ../chapitres/b4_llm/01_fiche.md#2046--recherche-dinformation--découpage-bm25-recherche-dense-hybride-bases-vectorielles-
[s593]: ../chapitres/b4_llm/01_fiche.md#de-tf-idf-à-bm25
[s594]: ../chapitres/b4_llm/01_fiche.md#la-recherche-hybride-et-la-fusion-rrf
[s595]: ../chapitres/b4_llm/01_fiche.md#mesurer-une-recherche--recallk-et-mrr
[s596]: ../chapitres/b4_llm/01_fiche.md#lidée
[s597]: ../chapitres/b4_llm/01_fiche.md#2048--fine-tuning--complet-ou-peft--lora-rang-alpha-modules-ciblés-qlora-
[s598]: ../chapitres/b4_llm/01_fiche.md#qlora
[s599]: ../chapitres/b4_llm/01_fiche.md#fusionner-ladaptateur
[s600]: ../chapitres/b4_llm/01_fiche.md#linjection-de-prompt
[s601]: ../chapitres/b4_llm/01_fiche.md
[s602]: ../chapitres/b4_llm/01_fiche.md#lanatomie-dun-prompt
[s603]: ../chapitres/b5_diffusion/01_fiche.md#2052--le-processus-direct--planning-de-bruit-ᾱ_t-et-forme-fermée-qx_t--x_0-
[s604]: ../chapitres/b5_diffusion/01_fiche.md#2053--le-processus-inverse-et-la-loss-simplifiée--prédire-le-bruit-lien-avec-lelbo-
[s605]: ../chapitres/b5_diffusion/01_fiche.md#le-planning-linéaire-de-ho-et-coll
[s606]: ../chapitres/b5_diffusion/01_fiche.md#la-forme-fermée
[s607]: ../chapitres/b5_diffusion/01_fiche.md#un-pas-de-bruit
[s608]: ../chapitres/b5_diffusion/01_fiche.md#le-rapport-signal-sur-bruit
[s609]: ../chapitres/b5_diffusion/01_fiche.md#le-posterior--un-pas-en-arrière-quand-on-connaît-mathbfx_0
[s610]: ../chapitres/b5_diffusion/01_fiche.md#trois-façons-de-fabriquer-une-image
[s611]: ../chapitres/b5_diffusion/01_fiche.md#la-loss-simplifiée
[s612]: ../chapitres/b5_diffusion/01_fiche.md#lalgorithme
[s613]: ../chapitres/b5_diffusion/01_fiche.md#ce-que-coûte-un-échantillon
[s614]: ../chapitres/b5_diffusion/01_fiche.md#2055--échantillonnage-ddpm--moyenne-du-posterior-variance-t-pas-
[s615]: ../chapitres/b5_diffusion/01_fiche.md#2056--accélérer--ddim-déterministe-moins-de-pas-distillation-
[s616]: ../chapitres/b5_diffusion/01_fiche.md#lidée-de-ddim
[s617]: ../chapitres/b5_diffusion/01_fiche.md#pourquoi-cest-plus-rapide
[s618]: ../chapitres/b5_diffusion/01_fiche.md#le-u-net-et-ses-connexions-de-saut
[s619]: ../chapitres/b5_diffusion/01_fiche.md#un-bloc-résiduel-conditionné-par-le-pas
[s620]: ../chapitres/b5_diffusion/01_fiche.md#2054--le-débruiteur--u-net-minuscule-et-encodage-du-pas-de-temps-
[s621]: ../chapitres/b5_diffusion/01_fiche.md#la-distillation
[s622]: ../chapitres/b5_diffusion/01_fiche.md#donner-la-classe-au-réseau
[s623]: ../chapitres/b5_diffusion/01_fiche.md#la-guidance-sans-classifieur
[s624]: ../chapitres/b5_diffusion/01_fiche.md#2057--conditionner--label-de-classe-et-guidance-sans-classifieur-
[s625]: ../chapitres/b5_diffusion/01_fiche.md#le-lien-avec-lelbo
[s626]: ../chapitres/b5_diffusion/01_fiche.md#les-transformers-de-diffusion
[s627]: ../chapitres/b5_diffusion/01_fiche.md#flow-matching-et-rectified-flow
[s628]: ../chapitres/b6_explicabilite/01_fiche.md#2062--explications-globales--coefficients-arbres-importance-par-permutation-dépendance-partielle-et-ice-
[s629]: ../chapitres/b6_explicabilite/01_fiche.md#la-dépendance-partielle
[s630]: ../chapitres/b6_explicabilite/01_fiche.md#les-courbes-ice
[s631]: ../chapitres/b6_explicabilite/01_fiche.md#2063--valeurs-de-shapley--jeu-coopératif-propriétés-calcul-exact-par-coalitions-rôle-du-fond-
[s632]: ../chapitres/b6_explicabilite/01_fiche.md#partager-un-gain-entre-joueurs
[s633]: ../chapitres/b6_explicabilite/01_fiche.md#quatre-propriétés-une-seule-solution
[s634]: ../chapitres/b6_explicabilite/01_fiche.md#les-features-comme-joueurs
[s635]: ../chapitres/b6_explicabilite/01_fiche.md#shap-une-famille-dexplications-additives
[s636]: ../chapitres/b6_explicabilite/01_fiche.md#un-exemple-à-deux-features
[s637]: ../chapitres/b6_explicabilite/01_fiche.md#les-trois-graphiques-à-connaître
[s638]: ../chapitres/b6_explicabilite/01_fiche.md#le-gradient-comme-explication
[s639]: ../chapitres/b6_explicabilite/01_fiche.md#2065--attributions-pour-les-réseaux--saillance-gradient--entrée-gradients-intégrés-
[s640]: ../chapitres/b6_explicabilite/01_fiche.md#les-gradients-intégrés
[s641]: ../chapitres/b6_explicabilite/01_fiche.md#le-choix-de-la-ligne-de-référence
[s642]: ../chapitres/b6_explicabilite/01_fiche.md#les-tests-de-santé
[s643]: ../chapitres/b6_explicabilite/01_fiche.md#global-ou-local
[s644]: ../chapitres/b6_explicabilite/01_fiche.md#2068--biais--données-labels-mesure-variables-de-substitution-déploiement--cas-célèbres-
[s645]: ../chapitres/b6_explicabilite/01_fiche.md#doù-vient-un-biais
[s646]: ../chapitres/b6_explicabilite/01_fiche.md#cas-célèbres
[s647]: ../chapitres/b6_explicabilite/01_fiche.md#2069--équité--indicateurs-de-groupe-parité-démographique-égalité-des-chances-equalized-odds-calibration-et-impossibilités-
[s648]: ../chapitres/b6_explicabilite/01_fiche.md#les-critères
[s649]: ../chapitres/b6_explicabilite/01_fiche.md#des-écarts-avec-leurs-intervalles
[s650]: ../chapitres/b6_explicabilite/01_fiche.md#des-taux-par-groupe
[s651]: ../chapitres/b6_explicabilite/01_fiche.md#la-supervision-humaine
[s652]: ../chapitres/b6_explicabilite/01_fiche.md#lai-act
[s653]: ../chapitres/b8_rl_moderne/01_fiche.md#rapporter-plusieurs-graines
[s654]: ../chapitres/ch00a_python/01_fiche.md#10061--modules-imports-et-bibliothèque-standard
[s655]: ../chapitres/b7_production/01_fiche.md#une-structure-de-dépôt
[s656]: ../chapitres/b7_production/01_fiche.md#ce-que-cache-un-notebook
[s657]: ../chapitres/b7_production/01_fiche.md#2072--scripts-et-ligne-de-commande--argparse-point-dentrée-logging-codes-de-sortie-
[s658]: ../chapitres/b7_production/01_fiche.md#les-codes-de-sortie
[s659]: ../chapitres/b7_production/01_fiche.md#variables-denvironnement-et-secrets
[s660]: ../chapitres/b7_production/01_fiche.md#le-logging-plutôt-que-print
[s661]: ../chapitres/b7_production/01_fiche.md#2073--configuration--dataclasses-yaml-ou-json-surcharges-validation--séparer-code-et-paramètres-
[s662]: ../chapitres/b7_production/01_fiche.md#les-versions-figées-et-lempreinte-des-données
[s663]: ../chapitres/b7_production/01_fiche.md#pyprojecttoml-et-installation-éditable
[s664]: ../chapitres/b7_production/01_fiche.md#2074--reproductibilité--graines-déterminisme-versions-figées-empreinte-des-données-
[s665]: ../chapitres/b7_production/01_fiche.md
[s666]: ../chapitres/b7_production/01_fiche.md#quatre-familles-de-tests
[s667]: ../chapitres/b7_production/01_fiche.md#lintégration-continue
[s668]: ../chapitres/b7_production/01_fiche.md#une-grille--le-ml-test-score
[s669]: ../chapitres/b7_production/01_fiche.md#2077--empaqueter-et-servir--artefact-de-modèle-inférence-par-batches-ou-en-ligne-api-http-latence-
[s670]: ../chapitres/b7_production/01_fiche.md#en-ligne-ou-par-batches-et-une-api-http
[s671]: ../chapitres/b7_production/01_fiche.md#2078--exporter-et-optimiser-linférence--torchexport-onnx-précision-réduite-quantification-
[s672]: ../chapitres/b7_production/01_fiche.md#torchexport-et-onnx
[s673]: ../chapitres/b7_production/01_fiche.md#2079--conteneurs-et-déploiement--dockerfile-variables-denvironnement-secrets-démos-hébergées-
[s674]: ../chapitres/b7_production/01_fiche.md#une-démo-hébergée
[s675]: ../chapitres/b7_production/01_fiche.md#lindice-de-stabilité-de-population-psi
[s676]: ../chapitres/b7_production/01_fiche.md#le-test-de-kolmogorov-smirnov
[s677]: ../chapitres/b7_production/01_fiche.md#pourquoi-un-modèle-se-dégrade-sans-changer-et-que-surveiller
[s678]: ../chapitres/b7_production/01_fiche.md#image-conteneur-et-dockerfile
[s679]: ../chapitres/b7_production/01_fiche.md#la-dette-technique-du-ml
[s680]: ../chapitres/b7_production/01_fiche.md#journaliser-les-prédictions-et-le-rgpd
[s681]: ../chapitres/b8_rl_moderne/01_fiche.md#2081--des-tables-aux-réseaux--approximation-de-fonction-états-continus-instabilité-
[s682]: ../chapitres/b8_rl_moderne/01_fiche.md#instabilité--la-triade-mortelle
[s683]: ../chapitres/b8_rl_moderne/01_fiche.md#apprendre-une-valeur-par-régression
[s684]: ../chapitres/b8_rl_moderne/01_fiche.md#les-journaux-de-ppo
[s685]: ../chapitres/b8_rl_moderne/01_fiche.md#la-surestimation-et-double-dqn
[s686]: ../chapitres/b8_rl_moderne/01_fiche.md#apprendre-la-politique-elle-même
[s687]: ../chapitres/b8_rl_moderne/01_fiche.md#le-théorème-du-gradient-de-politique
[s688]: ../chapitres/b8_rl_moderne/01_fiche.md#reinforce
[s689]: ../chapitres/b8_rl_moderne/01_fiche.md#2084--acteur-critique--fonction-de-valeur-erreur-td-avantage-gae-
[s690]: ../chapitres/b8_rl_moderne/01_fiche.md#2087--rl-et-modèles-de-langue--rlhf-modèle-de-récompense-dpo-grpo-piratage-de-la-récompense-
[s691]: ../chapitres/b8_rl_moderne/01_fiche.md#gae--une-moyenne-de-toutes-les-longueurs
[s692]: ../chapitres/b8_rl_moderne/01_fiche.md#réutiliser-un-rollout-sans-tout-casser
[s693]: ../chapitres/b8_rl_moderne/01_fiche.md#2085--ppo--ratio-de-probabilités-objectif-clippé-entropie-environnements-vectorisés-
[s694]: ../chapitres/b8_rl_moderne/01_fiche.md#les-environnements-vectorisés
[s695]: ../chapitres/b8_rl_moderne/01_fiche.md#le-critique-et-sa-variance-expliquée
[s696]: ../chapitres/b8_rl_moderne/01_fiche.md#2088--limites--efficacité-en-échantillons-généralisation-sim-to-real-sécurité-
[s697]: ../chapitres/b8_rl_moderne/01_fiche.md#le-modèle-de-récompense-de-bradley-terry
[s698]: ../chapitres/b8_rl_moderne/01_fiche.md#dpo--la-récompense-est-cachée-dans-la-politique
[s699]: ../chapitres/b8_rl_moderne/01_fiche.md#grpo-et-les-récompenses-vérifiables
[s700]: ../chapitres/b8_rl_moderne/01_fiche.md#de-la-simulation-au-réel
[s701]: ../chapitres/b8_rl_moderne/01_fiche.md#sécurité
[s702]: ../chapitres/b8_rl_moderne/01_fiche.md
[s703]: ../chapitres/b8_rl_moderne/01_fiche.md#déboguer-un-agent-qui-napprend-pas
[s704]: ../projets/partie_1_detecteur_langue/README.md#la-grille-dévaluation-sur-20
[s705]: ../checkpoints/partie_1/05_synthese.md
[s706]: ../checkpoints/partie_2/05_synthese.md
[s707]: ../projets/partie_2_california_validation/README.md#le-protocole-le-cœur-du-projet
[s708]: ../projets/partie_2_california_validation/README.md#les-extensions-pour-aller-plus-loin
[s709]: ../chapitres/b6_explicabilite/01_fiche.md#léquité-par-aveuglement
[s710]: ../projets/partie_3_revenus/README.md
[s711]: ../projets/partie_3_revenus/README.md#la-grille-dévaluation-sur-20
[s712]: ../projets/partie_3_revenus/README.md#le-contexte
[s713]: ../projets/partie_4_mini_framework/README.md#les-livrables
[s714]: ../projets/partie_6_generatif/README.md
[s715]: ../projets/projet_final/README.md#la-règle-du-test-scellé
[s716]: ../chapitres/ch00a_python/01_fiche.md#10042--boucles--for-range-enumerate-zip-while-break-continue
[s717]: ../chapitres/ch00a_python/01_fiche.md#10052--arguments-positionnels-et-nommés-args-kwargs-arguments-keyword-only
[s718]: ../chapitres/ch00a_python/01_fiche.md#10053--fonctions-comme-valeurs--lambda-key-callable
[s719]: ../chapitres/ch00a_python/01_fiche.md#10063--fichiers-texte-et-chemins-avec-pathlib
[s720]: ../chapitres/ch00a_python/01_fiche.md#10072--méthodes-spéciales--__repr__-__len__-__getitem__-__call__-opérateurs
[s721]: ../chapitres/ch00a_python/01_fiche.md#10086--reshape-transposition-empilement
[s722]: ../chapitres/ch00a_python/01_fiche.md#10088--les-images-sont-des-arrays--mnist
[s723]: ../chapitres/ch00a_python/01_fiche.md#100101--plot-scatter-hist--titres-axes-et-légendes
[s724]: ../chapitres/ch00a_python/01_fiche.md#100114--doctest--des-exemples-exécutables-dans-les-docstrings
[s725]: ../chapitres/ch00a_python/01_fiche.md#10071--class-__init__-attributs-et-méthodes--dataclass
[s726]: ../chapitres/ch00b_maths/01_fiche.md#1012--fonctions-usuelles
[s727]: ../chapitres/ch00b_maths/01_fiche.md#10127--composition-de-fonctions
[s728]: ../chapitres/ch00b_maths/01_fiche.md#10142--produit-matrice-vecteur
[s729]: ../chapitres/ch00b_maths/01_fiche.md#10152--dérivées-usuelles-et-règles-somme-produit-quotient-exp-ln
[s730]: ../chapitres/ch00b_maths/01_fiche.md#10154--variations-minimum-et-maximum
[s731]: ../chapitres/ch20_deep_learning/01_fiche.md#201--pourquoi-ce-chapitre--
[s732]: ../chapitres/ch02_stats/01_fiche.md#21--pourquoi-des-statistiques-
[s733]: ../chapitres/ch02_stats/01_fiche.md#23--quelques-lois-usuelles
[s734]: ../chapitres/ch02_stats/01_fiche.md#251--avec-remise-252-sans-remise-
[s735]: ../chapitres/ch23_pytorch_1/01_fiche.md#au-delà-du-livre--la-reproductibilité
[s736]: ../chapitres/ch07_classification/01_fiche.md#74--plusieurs-classes-avec-des-classifieurs-binaires-
[s737]: ../chapitres/ch11_raisonnement/01_fiche.md
[s738]: ../chapitres/ch12_preparation/01_fiche.md
[s739]: ../checkpoints/partie_3/05_synthese.md#3-quel-classifieur-quand-
[s740]: ../checkpoints/partie_3/05_synthese.md#6-anatomie-de-lapi-scikit-learn
[s741]: ../chapitres/ch29_datasets/01_fiche.md#2931--scikit-learn-
[s742]: ../chapitres/ch15_sklearn/01_fiche.md#1544--decision_function-et-predict_proba-
[s743]: ../chapitres/ch15_sklearn/01_fiche.md#155--le-clustering
[s744]: ../chapitres/ch15_sklearn/01_fiche.md#1591--la-validation-croisée-
[s745]: ../chapitres/ch15_sklearn/01_fiche.md#153--conventions--importer-lire-la-documentation-suivre-les-dépréciations-
[s746]: ../chapitres/ch17_activations/01_fiche.md
[s747]: ../chapitres/ch17_activations/01_fiche.md#177--la-galerie-des-activations-
[s748]: ../chapitres/ch18_backprop/01_fiche.md#18121--la-rétropropagation-en-un-seul-endroit-
[s749]: ../chapitres/ch18_backprop/01_fiche.md#18125--les-mini-batches-
[s750]: ../chapitres/ch19_optimiseurs/01_fiche.md#192--lerreur-comme-géométrie
[s751]: ../chapitres/ch20_deep_learning/01_fiche.md#207--construire-un-réseau-profond-
[s752]: ../checkpoints/partie_4/05_synthese.md#4-la-boucle-dentraînement-pytorch-et-ses-pièges
[s753]: ../chapitres/ch20_deep_learning/01_fiche.md#204--un-tour-dhorizon-des-couches
[s754]: ../chapitres/ch20_deep_learning/01_fiche.md#206--quelques-exemples-
[s755]: ../chapitres/ch21_cnn/01_fiche.md#au-delà-du-livre--la-rétropropagation-dune-convolution
[s756]: ../chapitres/ch21_cnn/01_fiche.md
[s757]: ../chapitres/ch23_pytorch_1/01_fiche.md#2362--créer-le-modèle-
[s758]: ../chapitres/ch23_pytorch_1/01_fiche.md#2357--corriger-les-labels-
[s759]: ../projets/projet_final/README.md#pf8--évaluation-honnête-sur-le-test-scellé--une-seule-passe-intervalles-bootstrap-analyse-derreurs-sous-groupes-robustesse---️-180-min
[s760]: ../chapitres/ch23_pytorch_1/01_fiche.md#237--entraîner-le-modèle-
[s761]: ../chapitres/ch24_pytorch_2/01_fiche.md
[s762]: ../chapitres/ch24_pytorch_2/01_fiche.md#2431--habiller-un-réseau-en-estimateur-
[s763]: ../chapitres/ch24_pytorch_2/01_fiche.md#2462--construire-un-modèle-fonctionnel-
[s764]: ../chapitres/ch24_pytorch_2/01_fiche.md#2442--préparer-les-données-pour-un-cnn-
[s765]: ../chapitres/ch24_pytorch_2/01_fiche.md#2451--générer-des-séries
[s766]: ../chapitres/ch24_pytorch_2/01_fiche.md#2456--les-rnn-profonds-
[s767]: ../chapitres/ch24_pytorch_2/01_fiche.md#24511--générer-du-texte
[s768]: ../chapitres/ch25_autoencodeurs/01_fiche.md#256--discussion--lautoencodeur-et-lacp
[s769]: ../chapitres/ch25_autoencodeurs/01_fiche.md
[s770]: ../chapitres/ch27_gan/01_fiche.md
[s771]: ../chapitres/ch29_datasets/01_fiche.md#291--les-datasets-publics-
[s772]: ../projets/projet_final/README.md#pf3--exploration-et-audit-de-qualité--doublons-fuites-déséquilibre-valeurs-manquantes-décalage-entre-découpages---️-180-min
[s773]: ../projets/projet_final/README.md#pf2--données--data-card-complète-licence-vérifiée-script-de-téléchargement-reproductible-révision-figée-sha-256-️--️-90-min
[s774]: ../chapitres/b2_tokenisation/01_fiche.md#le-même-texte-un-autre-prix-selon-la-langue
[s775]: ../chapitres/b2_tokenisation/01_fiche.md#les-plus-proches-voisins
[s776]: ../chapitres/b2_tokenisation/01_fiche.md#2029--géométrie-des-embeddings--cosinus-voisins-analogies-visualisation-biais-
[s777]: ../chapitres/b3_transformers/01_fiche.md#2037--familles--encodeur-bert-décodeur-gpt-encodeur-décodeur-t5--objectifs-causal-et-masqué-
[s778]: ../chapitres/b4_llm/01_fiche.md#2042--charger-et-exécuter-un-modèle-avec-transformers--tokenizer-modèle-pipeline-dtype-mémoire-
[s779]: ../chapitres/b4_llm/01_fiche.md#2044--prompting--zéro-shot-few-shot-instructions-sortie-structurée-raisonnement-pas-à-pas-
[s780]: ../chapitres/b4_llm/01_fiche.md#ce-que-fait-vraiment-un-llm
[s781]: ../chapitres/b4_llm/01_fiche.md#rag-ou-fine-tuning-
[s782]: ../chapitres/b4_llm/01_fiche.md#2049--classer-des-textes--encodeur-fine-tuné-llm-en-zéro-shot-ou-baseline-classique-
[s783]: ../chapitres/b5_diffusion/01_fiche.md#le-planning-cosinus-de-nichol-et-dhariwal
[s784]: ../chapitres/b5_diffusion/01_fiche.md#dire-au-réseau-quel-est-le-pas--lencodage-du-pas-de-temps
[s785]: ../chapitres/b5_diffusion/01_fiche.md#en-pratique--la-moyenne-exponentielle-des-poids
[s786]: ../chapitres/b6_explicabilite/01_fiche.md#2067--limites-des-explications--instabilité-tests-de-santé-corrélation-nest-pas-causalité-
[s787]: ../chapitres/b6_explicabilite/01_fiche.md#20611--éthique-et-régulation-en-pratique--fiches-modèles-supervision-humaine-rgpd-ai-act-
[s788]: ../chapitres/b8_rl_moderne/01_fiche.md#2086--pratique-du-deep-rl--hyperparamètres-graines-évaluation-bibliothèques-de-référence-
