# B2 · Mes réponses

> Ta copie de ce fichier est dans `mon_travail/b2_tokenisation/06_mes_reponses.md` (créée par `python tools/start_chapter.py B2`) : c'est **elle** que tu remplis, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat (en LaTeX pour les formules, 0B.32) : une réponse juste pour une mauvaise raison ne vaut rien en entretien. Reporte les réponses courtes dans la partie 0 du notebook (`answer_B2_Q1a = ...`, `answer_B2_1a = ...`) pour les vérifier.

## 🧠 Quiz

### B2.Q1 — Caractères, mots ou sous-mots : taille du vocabulaire contre longueur des séquences

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q2 — Octets, points de code, caractères affichés : combien « é » compte-t-il ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q3 — NFC ou NFD : deux chaînes identiques à l'écran, différentes en mémoire

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q4 — Fusions BPE : pourquoi la liste ordonnée des fusions est le modèle

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q5 — BPE au niveau octet : jamais de token inconnu, vrai ou faux ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q6 — Tokens spéciaux et espace initial : « Ġthe » n'est pas « the »

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q7 — WordPiece, Unigram, SentencePiece : quel critère pour quelle méthode ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q8 — Pourquoi un LLM compte mal les lettres d'un mot et calcule mal avec les chiffres

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q9 — nn.Embedding : une table de consultation entraînable, pas une couche magique

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q10 — L'hypothèse distributionnelle : un mot se définit par ses voisins

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q11 — Statique ou contextuel : l'« avocat » du tribunal et celui du marché

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.Q12 — Cosinus ou distance euclidienne pour des embeddings normalisés ?

- a) :
- b) :
- c) :
- d) :
- e) :


## 🔁 Rappels

### B2.R1 — Ch. B1 : à quoi sert l'embedding d'image extrait d'un réseau pré-entraîné ?

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.R2 — Ch. 22 : build_vocab, encode, decode et le token inconnu

- a) :
- b) :
- c) :
- d) :
- e) :


### B2.R3 — Ch. 6 : entropie, Huffman et ce que l'on compresse vraiment

- a) :
- b) :
- c) :
- d) :
- e) :


## ✏️ ∂ 🧮 Papier-crayon

### Ex B2.1 — « Noël à Paris » en UTF-8 : octets, points de code, longueur

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :


### Ex B2.2 — Analogie à la main : roi − homme + femme sur des vecteurs de dimension 3

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :


### Ex B2.3 — Trois fusions BPE à la main sur un mini-corpus

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


### Ex B2.4 — Encoder un mot jamais vu avec des fusions déjà apprises

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :


### Ex B2.5 — One-hot fois matrice : nn.Embedding à la main, et quelles lignes reçoivent un gradient

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


### Ex B2.6 — Co-occurrences et PPMI à la main (fenêtre de 1, quatre mots)

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


### Ex B2.7 — Skip-gram avec échantillonnage négatif : gradients par rapport au mot et au contexte

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B2.8 — Fermi : fertilité française et anglaise, pages dans 8 192 tokens, prix d'un roman traduit par API

Mes calculs :


**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B2.9 — Fermi : poids de la table d'embedding de GPT-2 et d'un LLM à 128 000 tokens

Mes calculs :


- a) :
- b) :
- c) :
- d) :
- e) :
- f) :
- g) :

Dans ta copie :


## 🗣️ ⚖️ 📄 Réflexion

### Ex B2.10 — BPE en cinq lignes, sans jargon

Mes cinq lignes :


### Ex B2.11 — Une taxe cachée sur certaines langues, des stéréotypes dans les vecteurs : tokenisation et biais

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


### Ex B2.12 — Sennrich, Haddow et Birch (2016) : les sous-mots pour traduire les mots rares

**1.**


**2.**


**3.**


**4.**


**5.**


**6.**


**7.**


## 💼 Entretien

### B2.E1 — Pourquoi les LLM découpent-ils le texte en sous-mots plutôt qu'en mots ou en caractères ?


### B2.E2 — Explique BPE, puis ce qui le distingue de WordPiece


### B2.E3 — Que se passe-t-il si l'on change de tokenizer sans ré-entraîner le modèle ?


### B2.E4 — Qu'est-ce qu'un embedding, et comment mesurer la similarité de deux textes ?


### B2.E5 — Comment construirais-tu une recherche sémantique sur des documents internes ?


## 📝 Notes sur le notebook

Les réponses aux questions « dans tes notes » du notebook.

- **B2.13** :
- **B2.14** :
- **B2.15** :
- **B2.16** :
- **B2.17** :
- **B2.18** :
- **B2.19** :
- **B2.20** :
- **B2.21** :
- **B2.22** :
- **B2.23** :
- **B2.24** :
- **B2.25** :
- **B2.26** :
- **B2.27** :
- **B2.28** :
- **B2.29** :
- **B2.30** :
- **B2.31** :
