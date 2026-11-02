# B4 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/b4_llm/06_mes_reponses.md` (créée par `python tools/start_chapter.py B4`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_B4_Q1a = ...`, `answer_B4_1a = ...`) pour les vérifier.

## 🧠 Quiz

### B4.Q1 — Modèle de base, modèle instruct, modèle de chat : lequel pour quoi ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q2 — « Open weights » n'est pas « open source » : lire la licence d'un modèle

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q3 — dtype, device et mémoire : ce qui décide si un modèle tient sur ton ordinateur

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q4 — Chat template : pourquoi ne pas écrire soi-même « User: … Assistant: … »

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q5 — Glouton, température, top-p : lequel pour extraire une date, lequel pour écrire un poème ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q6 — Zéro-shot, few-shot, chain-of-thought : vrai ou faux

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q7 — Pourquoi un LLM invente : hallucination et token le plus probable

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q8 — Évaluer un LLM : exact match, juge LLM, contamination des benchmarks

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q9 — BM25 ou recherche dense : qui retrouve « Watson » et qui retrouve « le docteur » ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q10 — RAG ou fine-tuning pour ajouter des connaissances ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q11 — LoRA : ce qui est gelé, ce qui est appris, ce que l'on fusionne

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.Q12 — Injection de prompt : quand un document récupéré devient une instruction

- a) :
- b) :
- c) :
- d) :
- e) :


## 🔁 Rappels

### B4.R1 — Ch. B3 : ce que le masque causal et le KV cache imposent à la génération

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.R2 — Ch. B1 : extraction de features ou fine-tuning : que change le nombre d'exemples ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B4.R3 — Ch. 13 : les k plus proches voisins et le choix de la distance

- a) :
- b) :
- c) :
- d) :
- e) :


## ✏️ ∂ 🧮 Papier-crayon

### Ex B4.1 — Mémoire d'un modèle : paramètres, dtype et KV cache — tient-il sur ton ordinateur ?

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :
- i) :
- j) :
- k) :

Dans ta copie :


### Ex B4.2 — Recall@k et MRR pour cinq requêtes

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :

Dans ta copie :


### Ex B4.3 — LoRA : paramètres entraînables d'une couche 896×896 de rang 8, puis de tout un modèle

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :
- i) :
- j) :
- k) :

Dans ta copie :


### Ex B4.4 — Un chat template à la main : rôles, tokens spéciaux et prompt de génération

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :

Dans ta copie :


### Ex B4.5 — BM25 à la main : trois documents, une requête de deux mots

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :
- h) :
- i) :

Dans ta copie :


### Ex B4.6 — LoRA : ΔW = BA de rang au plus r, B nulle au départ, gradients de A et de B, facteur α/r

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


### Ex B4.7 — Fermi : coût et latence d'un assistant RAG, API facturée au token contre petit modèle local

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


## 🗣️ ⚖️ 📄 Réflexion

### Ex B4.8 — Le RAG expliqué en cinq lignes

Mes cinq lignes :


### Ex B4.9 — Un assistant RH qui répond sur les salaires : données personnelles, hallucinations, injection, transparence

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B4.10 — Fine-tuner sur des romans : domaine public, œuvres protégées et droit de retrait (opt-out)

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


### Ex B4.11 — Hu et coll. (2021) : « LoRA: Low-Rank Adaptation of Large Language Models »

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


**8.**


**9.**


## 💼 Entretien

### B4.E1 — Comment réduis-tu les hallucinations d'un assistant fondé sur un LLM ?


### B4.E2 — RAG ou fine-tuning : comment choisis-tu ?


### B4.E3 — Explique LoRA et pourquoi c'est si économique


### B4.E4 — Comment évalues-tu un système RAG, de la recherche à la réponse ?


### B4.E5 — Quels risques de sécurité avec un LLM en production, et quelles parades ?


## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook.

- **B4.12** :
- **B4.13** :
- **B4.14** :
- **B4.15** :
- **B4.16** :
- **B4.17** :
- **B4.18** :
- **B4.19** :
- **B4.20** :
- **B4.21** :
- **B4.22** :
- **B4.23** :
- **B4.24** :
- **B4.25** :
- **B4.26** :
- **B4.27** :
- **B4.28** :
- **B4.29** :
- **B4.30** :
