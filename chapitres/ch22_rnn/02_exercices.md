# 22 · Réseaux récurrents (RNN, LSTM, GRU) — quiz, rappels, exercices papier, réflexion et entretien

> Écris tes réponses dans **ta copie** `mon_travail/ch22_rnn/06_mes_reponses.md` (créée par `python tools/start_chapter.py 22`), jamais dans ce fichier : il est mis à jour par Claude.
> Les réponses courtes des **quiz**, des **rappels**, des exercices ✏️ 22.1 à 22.5 et de l'estimation 🧮 22.8 se vérifient dans la **partie 0** du notebook (`wb.check`). Les questions marquées « dans ta copie », les preuves ∂ 22.6 et 22.7, la réflexion (🗣️ ⚖️ 📄) et l'entretien se corrigent avec `05_solutions.md`. Indices : `04_indices.md`. Calculatrice autorisée.
> Formats de réponse : un nombre (`0.3` ou `"0,3"`), `True` ou `False` pour un vrai ou faux, une lettre seule entre guillemets pour un choix (`"E"`), une forme entre guillemets (`"(3, 5, 1)"`), un mot entre guillemets (`"fit"`).
> Conventions du chapitre : celles de PyTorch et de `mylearn` (fiche). Un état $h_t$ et une entrée $x_t$ sont des vecteurs lignes ; une cellule calcule $x_t W_{ih} + b_{ih} + h_{t-1} W_{hh} + b_{hh}$ ; les portes du LSTM sont rangées $i, f, g, o$, celles du GRU $r, z, n$ ; les pas de temps se comptent à partir de 1 ($h_0$ est l'état de départ), les positions d'une série à partir de 0.

**Sommaire** : [🧠 Quiz](#quiz) · [🔁 Rappels](#rappels) · [✏️ ∂ 🧮 Papier-crayon](#papier) · [🗣️ ⚖️ 📄 Réflexion](#reflexion) · [💼 Entretien](#entretien) · [Exercices du notebook](#notebook)

Légende : ★ application directe · ★★ standard · ★★★ approfondi · ★★★★ défi · ⏱️ durée indicative · parcours R = rapide, M = maths, C = code.

<a id="quiz"></a>

## 🧠 Quiz

Sans la fiche, en 3 minutes chacun. Réponds vite, vérifie les réponses courtes dans la partie 0 du notebook, puis lis les explications de `05_solutions.md`.

### 22.Q1 — Pourquoi un réseau dense ignore l'ordre 🧠 ⏱️ 3 min
*Fiche §22.1, §22.2 · livre §22.1, §22.2 · parcours R*

On entraîne une petite couche dense à prédire une valeur d'une série à partir des 4 valeurs qui la précèdent, découpées en fenêtres.

a) Vrai ou faux : si l'on range **toujours de la même façon** les 4 valeurs de chaque fenêtre dans un ordre mélangé (la 3ᵉ, puis la 1ʳᵉ, la 4ᵉ et la 2ᵉ), le réseau peut atteindre exactement la même loss qu'avec l'ordre naturel.
b) Pourquoi ? (A) un réseau dense trie ses entrées ; (B) chaque entrée a ses propres poids : permuter les entrées revient à permuter les poids de la première couche ; (C) la loss ne dépend pas des entrées ; (D) un réseau dense garde un état.
c) Vrai ou faux : si l'on mélange les valeurs dans un ordre **différent pour chaque fenêtre**, tiré au hasard, le réseau apprend aussi bien.
d) Ce qui fait d'un texte, d'un film ou de la hauteur de la marée chaque jour une **séquence** : (A) les valeurs sont des nombres ; (B) chaque élément a un lien avec ceux qui le précèdent ou le suivent ; (C) il y a beaucoup de données ; (D) les données sont étiquetées.
e) Vrai ou faux : quand une fenêtre de 4 valeurs glisse d'un pas, deux exemples consécutifs ont 3 valeurs en commun.

### 22.Q2 — État ou poids : qu'est-ce qui change en production ? 🧠 ⏱️ 3 min
*Fiche §22.3, §22.3.1 · livre §22.3, §22.3.1 · parcours R*

a) Vrai ou faux : après l'entraînement, les poids d'un RNN changent à chaque nouvelle entrée.
b) Vrai ou faux : après l'entraînement, l'état d'un RNN change à chaque nouvelle entrée.
c) On donne deux fois la même phrase à un RNN entraîné, en repartant chaque fois d'un état nul. Vrai ou faux : les sorties sont identiques.
d) On lui donne la phrase B juste après la phrase A sans remettre l'état à zéro, puis B seule à partir d'un état nul. Vrai ou faux : les sorties pour B sont forcément identiques.
e) Ce qu'un RNN emporte d'une entrée à la suivante : (A) ses poids ; (B) son état ; (C) ses gradients ; (D) son learning rate.

### 22.Q3 — Délai, déroulement et pas de temps 🧠 ⏱️ 3 min
*Fiche §22.4 · livre §22.4 · parcours R*

a) Dans le schéma enroulé du livre, le petit carré noir placé après l'état représente : (A) une activation ; (B) un délai d'un pas : l'état calculé maintenant ne servira qu'à l'entrée suivante ; (C) un biais ; (D) un dropout.
b) Une séquence de 12 pas de temps traverse une couche RNN. Combien de jeux de poids différents le schéma déroulé contient-il ?
c) Combien de fois la cellule est-elle appliquée ?
d) Vrai ou faux : dans le schéma déroulé, chaque copie de la cellule a ses propres poids.
e) Le nombre de matrices de poids d'une cellule `nn.RNNCell` de PyTorch (sans compter les biais).

### 22.Q4 — Entrée one-hot, taille de l'état et sens de ses valeurs 🧠 ⏱️ 3 min
*Fiche §22.4.1, §22.4.2 · livre §22.4.1, §22.4.2*

a) Un vocabulaire de 3 000 mots codés en one-hot : la longueur du vecteur d'entrée d'un mot.
b) Combien de valeurs non nulles contient ce vecteur ?
c) Vrai ou faux : la taille de l'état d'un RNN doit être égale à la taille de son entrée.
d) Une entrée one-hot de 30 caractères et un état de 64 valeurs : le nombre de poids de la matrice $W_{ih}$ (sans biais).
e) Ce que représente la 17ᵉ valeur de l'état d'un RNN entraîné : (A) une information fixée à l'avance par le programmeur ; (B) ce que l'entraînement a appris à y mettre, qu'on ne sait interpréter que parfois ; (C) le 17ᵉ caractère lu ; (D) la loss au 17ᵉ pas.

### 22.Q5 — (échantillons, pas de temps, features) : la bonne organisation 🧠 ★★ ⏱️ 3 min
*Fiche §22.5 · livre §22.5 · parcours R*

Une station météo prend, chaque jour pendant 10 jours, 6 mesures (une toutes les deux heures, de 8 h à 18 h) de deux grandeurs : la température et la pression. Les nuits ne sont pas mesurées.

a) La forme du tenseur au format `batch_first=True` de PyTorch, $(N, T, D)$ (entre guillemets).
b) La forme qu'attend `nn.LSTM` **par défaut**, sans `batch_first`.
c) Vrai ou faux : `x.reshape(6, 10, 2)` transforme correctement le premier tenseur en le second.
d) Comment ranger les données ? (A) un seul exemple de 60 pas ; (B) 10 exemples de 6 pas ; (C) 60 exemples d'un pas ; (D) un exemple d'un pas à 120 features.
e) Les 6 températures d'une seule journée, rangées par erreur en un tenseur de forme `(1, 1, 6)` : combien de pas de temps le RNN voit-il ?

### 22.Q6 — BPTT, gradients évanescents et explosifs, dépendances longues 🧠 ★★ ⏱️ 3 min
*Fiche §22.6 · livre §22.6 · parcours R, M*

a) La rétropropagation dans le temps (BPTT), c'est : (A) un optimiseur ; (B) la rétropropagation appliquée au réseau déroulé, les contributions d'un même poids s'ajoutant sur tous les pas ; (C) une initialisation ; (D) un dropout appliqué dans le temps.
b) Le gradient qui remonte de 50 pas est multiplié par 50 facteurs d'environ 0,9 chacun. Son ordre de grandeur : (A) 0,5 ; (B) 0,05 ; (C) 0,005 ; (D) 0,0005.
c) Même question avec 50 facteurs d'environ 1,1 : (A) 1 ; (B) 12 ; (C) 120 ; (D) 1 200.
d) Vrai ou faux : le gradient clipping résout le problème du gradient évanescent.
e) Une dépendance à long terme, c'est : (A) un poids qui dépend d'un autre ; (B) une sortie qui dépend d'une entrée lointaine dans la séquence, que la descente de gradient a du mal à relier ; (C) une loss qui baisse lentement ; (D) un état trop petit.

### 22.Q7 — Une porte, c'est une multiplication 🧠 ⏱️ 3 min
*Fiche §22.7, §22.7.1 · livre §22.7, §22.7.1*

a) Une porte vaut 0,25 et la valeur qui la traverse 80 : la valeur en sortie.
b) La fonction qui calcule les valeurs d'une porte dans un LSTM : (A) tanh ; (B) sigmoïde ; (C) ReLU ; (D) softmax.
c) Vrai ou faux : une porte peut changer le signe de la valeur qui la traverse.
d) Vrai ou faux : une porte peut augmenter la valeur absolue de la valeur qui la traverse.
e) Les valeurs des portes se calculent à chaque pas à partir de : (A) l'entrée courante et l'état caché précédent ; (B) le learning rate ; (C) la loss ; (D) un tirage au hasard.

### 22.Q8 — Oublier, se souvenir, sélectionner 🧠 ★★ ⏱️ 3 min
*Fiche §22.7.2 · livre §22.7.2 · parcours R*

a) La porte « se souvenir » (*remember*) du livre s'appelle dans PyTorch : (A) la porte d'oubli $f$ ; (B) la porte d'entrée $i$ ; (C) la porte de sortie $o$ ; (D) le candidat $g$.
b) Et la porte « sélectionner » (*select*) ? Mêmes choix.
c) Vrai ou faux : la porte d'oubli faisait partie du LSTM de 1997.
d) Pour garder une valeur de la mémoire de cellule intacte pendant 100 pas, il faut : (A) $f \approx 0$ et $i \approx 1$ ; (B) $f \approx 1$ et $i \approx 0$ ; (C) $f \approx 1$ et $i \approx 1$ ; (D) $o \approx 0$.
e) Dans `nn.LSTM`, on découpe les colonnes de `weight_ih_l0.T` en 4 blocs de $H$ colonnes, numérotés à partir de 0. Le bloc n° 2 est celui de : (A) $i$ ; (B) $f$ ; (C) $g$ ; (D) $o$.

### 22.Q9 — LSTM ou GRU ? 🧠 ⏱️ 2 min
*Fiche §22.7.2 · livre §22.7.2*

a) Le nombre de blocs de $H$ colonnes de la matrice $W_{ih}$ d'un GRU.
b) Vrai ou faux : un GRU a une mémoire de cellule $c_t$ distincte de son état $h_t$.
c) À tailles égales, lequel a le moins de paramètres ? (A) le LSTM ; (B) le GRU ; (C) ils en ont autant.
d) Vrai ou faux : le GRU est presque toujours nettement meilleur que le LSTM.

### 22.Q10 — One-to-many, many-to-one, many-to-many 🧠 ⏱️ 3 min
*Fiche §22.8, §22.8.1 · livre §22.8, §22.8.1 · parcours R*

Pour chaque tâche, la structure : (A) one-to-many ; (B) many-to-one ; (C) many-to-many synchronisé (une sortie par entrée) ; (D) many-to-many décalé (on lit toute l'entrée, puis on produit la sortie).

a) Classer une critique de film en positive ou négative.
b) Composer une mélodie à partir d'une seule note.
c) Traduire une phrase de l'anglais vers le français.
d) Étiqueter chaque image d'une vidéo (la balle monte, descend, vole à plat).
e) Le générateur de Holmes du livre (§22.9), qui lit 40 caractères pour prédire le suivant.

### 22.Q11 — RNN profond et bidirectionnel : quand et à quel prix ? 🧠 ★★ ⏱️ 3 min
*Fiche §22.8.2 à §22.8.4 · livre §22.8.2 à §22.8.4*

a) Vrai ou faux : un RNN bidirectionnel convient à la reconnaissance de commandes vocales en temps réel, mot après mot.
b) Dans un RNN profond, la deuxième couche reçoit de la première : (A) seulement son dernier état ; (B) toute sa séquence d'états ; (C) ses poids ; (D) rien.
c) Par défaut, `nn.LSTM(..., bidirectional=True)` combine les sorties des deux sens par : (A) une somme ; (B) une moyenne ; (C) une concaténation ; (D) un produit.
d) Vrai ou faux : le sens « arrière » d'une couche bidirectionnelle a ses propres poids.
e) Le prix d'un RNN profond et bidirectionnel : (A) moins de poids ; (B) plus de poids, et plus de données pour les apprendre ; (C) il ne s'entraîne plus par BPTT ; (D) il ne peut traiter que des séquences courtes.

### 22.Q12 — 🕰️ Transformers, RNN légers et modèles à espace d'états 🧠 ⏱️ 3 min
*Fiche §22.1, §22.9, sections « Au-delà du livre » · livre §22.1, §22.9 · parcours R*

a) L'année de l'article « Attention Is All You Need », qui introduit les Transformers.
b) Vrai ou faux : pendant l'entraînement, un Transformer calcule tous les pas d'une séquence en parallèle, un RNN doit les calculer l'un après l'autre.
c) Un cas où un LSTM reste un bon choix aujourd'hui : (A) un grand modèle de langage ; (B) un capteur embarqué qui prévoit une série en continu avec peu de mémoire et de calcul ; (C) la traduction de livres entiers ; (D) la génération d'images.
d) Les modèles à espace d'états comme Mamba : (A) sont des réseaux convolutifs ; (B) reprennent une récurrence linéaire, entraînable en parallèle, au coût linéaire en la longueur de la séquence ; (C) sont des arbres de décision ; (D) n'ont pas d'état.
e) Vrai ou faux : pour une série temporelle, on compare un LSTM à une baseline simple comme la persistance.

<a id="rappels"></a>

## 🔁 Rappels

### 22.R1 — Ch. 21 : une convolution 1D peut-elle lire une séquence ? 🔁 ★ ⏱️ 5 min
*Ch. 21 (§21.5) · fiche §22.2 · parcours R*

a) `nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4)` sur une série de 20 valeurs, sans padding : la longueur de la sortie.
b) Vrai ou faux : une convolution 1D applique les mêmes poids à toutes les positions de la série.
c) Le champ récepteur d'une sortie après deux `Conv1d` de noyau 3 empilées (stride 1, sans dilatation) : le nombre de valeurs de l'entrée dont elle dépend.
d) Avec des noyaux de 3, sans dilatation, combien de couches faut-il empiler pour qu'une sortie dépende d'une valeur placée 1 000 pas plus tôt (un champ récepteur d'au moins 1 001 valeurs) ?
e) Dans ta copie : un avantage d'une pile de convolutions 1D sur un RNN pour lire une séquence, et ce que change une **dilatation** qui double à chaque couche (ch. 21, §21.3.3 et l'encadré ⚠️ « la convolution dilatée » du §21.8).

### 22.R2 — Ch. 19 : gradient clipping et optimiseurs adaptatifs 🔁 ★ ⏱️ 5 min
*Ch. 19 · fiche §22.6 · parcours R*

a) Un gradient $(3, 4)$ et `max_norm = 1` : la norme du gradient avant clipping.
b) La première composante du gradient après un clipping par la norme (`clip_grad_norm_` avec `max_norm = 1`).
c) Vrai ou faux : le clipping par la norme change la direction du gradient.
d) RMSprop divise le pas de chaque poids par : (A) la racine d'une moyenne glissante des carrés de ses gradients ; (B) la norme des poids ; (C) le numéro de l'epoch ; (D) le learning rate.
e) Vrai ou faux : avec Adam, un gradient multiplié par 1 000 donne presque le même premier pas.

### 22.R3 — Ch. 15 : ajuster le scaler sur le train seulement 🔁 ★ ⏱️ 5 min
*Ch. 15, ch. 12 · fiche §22.5, section « Au-delà du livre : prévoir une série temporelle » · parcours R*

Une série de 100 valeurs mensuelles : les 80 premières servent à l'entraînement, les 20 dernières au test.

a) Vrai ou faux : on peut mélanger les 100 valeurs avant de les couper en 80 et 20.
b) Le min-max scaler s'ajuste sur : (A) les 100 valeurs ; (B) les 80 premières ; (C) les 20 dernières ; (D) chaque fenêtre séparément.
c) Les valeurs d'entraînement vont de 0 à 200, et une valeur de test vaut 250 : sa valeur après le min-max scaler.
d) Vrai ou faux : une valeur de test qui sort de $[0, 1]$ après le scaler est une erreur, qu'on corrige en réajustant le scaler sur le test.
e) Dans un `Pipeline` de scikit-learn, la méthode que reçoit le scaler sur les données de test (un mot entre guillemets).

<a id="papier"></a>

## ✏️ ∂ 🧮 Papier-crayon

Calculatrice autorisée. Écris la démarche dans ta copie de `06_mes_reponses.md`, puis reporte les résultats dans la partie 0 du notebook. Garde les valeurs exactes pendant tes calculs et **n'arrondis qu'à la fin**.

### Ex 22.1 — Combien de fenêtres ? Températures, taches solaires et Holmes ✏️ ★ ⏱️ 10 min
**Objectif :** calculer le nombre d'exemples d'un dataset fenêtré, et préparer un découpage sans fuite.
**Prérequis :** 0A (tranches) · fiche §22.2, section « Au-delà du livre : prévoir une série temporelle » · **Fil rouge :** taches solaires · **Parcours :** R, M

Une série de $T$ valeurs (positions $0$ à $T - 1$) donne des fenêtres de $w$ valeurs ; la cible d'une fenêtre est la valeur placée $h$ pas après sa dernière valeur ; deux fenêtres consécutives démarrent à $s$ positions d'écart.

a) 24 températures horaires d'une journée, des fenêtres de 4 valeurs pour prédire la suivante ($h = 1$), $s = 1$ : le nombre de fenêtres.
b) La même série avec $s = 2$.
c) La série mensuelle des taches solaires de janvier 1749 à décembre 2024 : son nombre de mois $T$.
d) Sur cette série, des fenêtres de 120 mois (10 ans) pour prédire la valeur 6 mois après la fin de la fenêtre ($h = 6$), $s = 1$ : le nombre de fenêtres.
e) On garde les 600 derniers mois pour le test. Les fenêtres d'entraînement ne doivent contenir aucune valeur de test, ni comme entrée ni comme cible : elles sont prises dans les $T - 600$ premiers mois seulement. Leur nombre.
f) Un texte de 575 000 caractères, des fenêtres de 40 caractères pour prédire le suivant, $s = 3$ : le nombre de fenêtres.
g) Le livre (§22.9) annonce « environ un demi-million » de fenêtres de 40 caractères, une tous les 3 caractères, sur trois recueils de Holmes. Combien de caractères cela suppose-t-il, environ ? (A) 170 000 ; (B) 500 000 ; (C) 1,5 million ; (D) 15 millions.

Dans ta copie : pourquoi prendre une fenêtre tous les 3 caractères plutôt qu'à chaque caractère ? Que perd-on ?

### Ex 22.2 — Dérouler à la main une cellule RNN scalaire sur trois pas ✏️ ★★ ⏱️ 15 min
**Objectif :** calculer à la main les états successifs d'une cellule RNN, et voir que l'ordre des entrées compte.
**Prérequis :** ch. 17 (tanh) · 0B (règle de la chaîne, puissances) · fiche §22.3.1, §22.4 · **Parcours :** R, M

Une cellule à une seule valeur d'état : $h_t = \tanh(u\,x_t + w\,h_{t-1} + b)$, avec une sortie $y_t = v\,h_t$, et $u = 1$, $w = 0{,}5$, $b = 0$, $v = 2$, $h_0 = 0$. On lui donne la séquence $x = (2,\ 0,\ -1)$. Réponds avec 4 décimales.

a) $h_1$.
b) $h_2$.
c) $h_3$.
d) $y_3$.
e) On donne les mêmes trois valeurs dans l'ordre inverse, $x = (-1,\ 0,\ 2)$ : la nouvelle valeur de $h_3$.
f) Sans activation (la tanh remplacée par l'identité), $h_3$ s'écrit $\alpha_1 x_1 + \alpha_2 x_2 + \alpha_3 x_3$ : la valeur du coefficient $\alpha_1$ (2 décimales).

Dans ta copie : pourquoi la cellule utilise-t-elle les mêmes $u$, $w$ et $b$ à chaque pas ? Que deviendrait le coefficient $\alpha_1$ de f) pour une séquence de 20 pas, et qu'est-ce que cela annonce pour l'apprentissage ?

### Ex 22.3 — Compter les poids : la cellule à 69 poids du livre, puis nn.RNN et nn.LSTM ✏️ ★★ ⏱️ 20 min
**Objectif :** compter les paramètres d'une cellule récurrente, à la manière du livre puis de PyTorch.
**Prérequis :** ch. 16 (paramètres d'une couche dense) · Ex 22.2 · fiche §22.4.1, §22.7.2 · **Parcours :** R, M

La cellule de la figure 22.18 du livre a 69 poids : des neurones A (de l'entrée vers l'état, sans biais), des neurones B (de l'état vers l'état, sans biais), un biais par valeur d'état, et des neurones C (de l'état vers la sortie, avec biais). On garde la même organisation avec une entrée de 10 valeurs, un état de 8 valeurs et une sortie de 3 valeurs.

a) Le nombre de poids des neurones A.
b) Le nombre de poids des neurones B.
c) Le nombre de poids des neurones C, biais compris.
d) Le nombre total de poids de cette cellule (biais de l'état compris).
e) Le nombre de paramètres de `nn.RNN(input_size=10, hidden_size=8)`.
f) Le nombre de paramètres de `nn.RNN(10, 8)` suivi de `nn.Linear(8, 3)` ; puis l'écart avec d) (un seul nombre : l'écart).
g) Le nombre de paramètres de `nn.LSTM(input_size=10, hidden_size=8)`.
h) Le générateur du livre (§22.9) : une entrée one-hot de 89 caractères, deux couches LSTM de 128 valeurs d'état empilées, puis `nn.Linear(128, 89)`. Son nombre de paramètres, compté comme PyTorch (deux vecteurs de biais par couche).
i) Le livre en annonce « un peu plus de 250 000 », compté comme Keras, avec un seul vecteur de biais par couche LSTM. L'écart entre les deux décomptes.

Dans ta copie : pourquoi le nombre de paramètres d'une couche récurrente ne dépend-il pas de la longueur des séquences ?

### Ex 22.4 — Un pas de LSTM à la main ✏️ ★★ ⏱️ 25 min
**Objectif :** calculer à la main les portes, la mémoire et la sortie d'un pas de LSTM, dans l'ordre de PyTorch.
**Prérequis :** ch. 17 (sigmoïde, tanh) · Ex 22.2 · fiche §22.7.2 · **Parcours :** M

Un LSTM à une seule valeur d'état ($H = 1$) et une entrée d'une valeur ($D = 1$). Ses poids, rangés dans l'ordre $i, f, g, o$ : $W_{ih} = (-0{,}5 ;\ 1 ;\ 1 ;\ -1)$, $W_{hh} = (1 ;\ -2 ;\ 0 ;\ 3)$, un biais total (la somme $b_{ih} + b_{hh}$) de $(0 ;\ 1 ;\ 0 ;\ 0)$. À ce pas : $x_t = 1$, $h_{t-1} = 0{,}5$, $c_{t-1} = 2$. Réponds avec 4 décimales.

a) La porte d'entrée $i_t$.
b) La porte d'oubli $f_t$.
c) Le candidat $g_t$.
d) La porte de sortie $o_t$.
e) La nouvelle mémoire $c_t$.
f) La sortie $h_t$.
g) On ajoute $+20$ au biais de la porte d'oubli et $-20$ à celui de la porte d'entrée : la nouvelle valeur de $c_t$.
h) Pour le pas de a) à f), la dérivée $\partial c_t / \partial c_{t-1}$ le long du seul chemin direct (les portes et le candidat traités comme des constantes).
i) La forme de `weight_ih_l0` de `nn.LSTM(input_size=1, hidden_size=1)` (entre guillemets).

Dans ta copie : que devient la mémoire après 30 pas identiques à celui de g), et après 30 pas identiques à celui de a) à f) si l'on ne garde que le terme $f_t\,c_{t-1}$ ? Lequel des deux réglages permet de transporter une information loin ?

### Ex 22.5 — Softmax avec température à la main ✏️ ★ ⏱️ 10 min
**Objectif :** calculer l'effet de la température sur la distribution du caractère suivant.
**Prérequis :** ch. 17 (softmax) · ch. 6 (distribution des caractères) · fiche §22.9 · **Fil rouge :** Holmes · **Parcours :** R, M

Après « Sherloc », un modèle de Holmes donne les logits $z = (3 ;\ 1 ;\ 0{,}5)$ aux trois caractères les plus probables « k », « e » et « a » (on ignore les autres). Réponds avec 4 décimales.

a) La probabilité de « k » à la température $T = 1$.
b) La probabilité de « k » à $T = 0{,}5$.
c) La probabilité de « k » à $T = 2$.
d) La limite de la probabilité de « k » quand $T$ tend vers 0.
e) La limite de la probabilité de « k » quand $T$ tend vers l'infini.
f) Le rapport $p(\text{k}) / p(\text{e})$ à $T = 0{,}5$ (2 décimales).
g) Vrai ou faux : à $T = 2$, le caractère le plus probable reste « k ».

Dans ta copie : pourquoi un générateur qui prend toujours le caractère le plus probable ($T \to 0$) finit-il souvent par tourner en boucle ? Quelle température choisirais-tu pour un texte qui doit rester lisible, et laquelle pour des noms propres inventés ?

### Ex 22.6 — Pourquoi le gradient s'évanouit : un produit de dérivées de tanh ∂ ★★★ ⏱️ 35 min
**Objectif :** démontrer que le gradient d'un RNN est un produit de facteurs, et en déduire quand il disparaît ou explose.
**Prérequis :** ch. 18 (règle de la chaîne) · Ex 22.2 · 0B (suites géométriques) · fiche §22.6 · **Parcours :** M

On prend une cellule scalaire $h_k = \tanh(a_k)$, $a_k = w\,h_{k-1} + u\,x_k + b$, et une loss $L$ qui ne dépend que de $h_T$. Rédige dans ta copie.

1. Montre que $\dfrac{\partial h_k}{\partial h_{k-1}} = w\,(1 - h_k^2)$.
2. En déduire, pour $t < T$, que $\dfrac{\partial h_T}{\partial h_t} = \displaystyle\prod_{k=t+1}^{T} w\,(1 - h_k^2)$, puis $\dfrac{\partial L}{\partial x_t}$ en fonction de $\dfrac{\partial L}{\partial h_T}$.
3. Montre que $\left|\dfrac{\partial h_T}{\partial h_t}\right| \le |w|^{T - t}$. Conclus quand $|w| < 1$. Donne une borne pour $w = 0{,}9$ et $T - t = 50$.
4. Avec $|w| > 1$, la borne de 3. ne dit plus rien. Montre que si les $h_k$ restent proches de 0, le produit vaut environ $|w|^{T-t}$, et calcule cette valeur pour $w = 1{,}5$ et $T - t = 20$. Pourquoi l'explosion est-elle souvent suivie d'une saturation, puis d'une disparition ?
5. Cas vectoriel : avec $h_k = \tanh(x_k W_{ih} + h_{k-1} W_{hh} + b)$, les états en lignes, admets que la jacobienne d'un pas est $J_k = \operatorname{diag}(1 - h_k^2)\,W_{hh}^\top$. En utilisant $\lVert A B \rVert \le \lVert A \rVert\,\lVert B \rVert$ et $\lVert \operatorname{diag}(1 - h_k^2) \rVert \le 1$ (la norme spectrale, la plus grande valeur singulière), montre que $\lVert \partial h_T / \partial h_t \rVert \le \sigma_{\max}(W_{hh})^{T - t}$. Que conclure si $\sigma_{\max}(W_{hh}) < 1$ ?
6. Le poids $w$ est partagé par tous les pas. Écris $\dfrac{\partial L}{\partial w}$ comme une somme de $T$ termes, en notant $\delta_k = \dfrac{\partial L}{\partial h_k}$. Explique pourquoi, quand le gradient s'évanouit, cette somme est dominée par les pas proches de $T$, et ce que cela veut dire pour les dépendances lointaines.
7. Le livre explique la disparition par la composition de tanh avec elle-même. Montre que la fonction $\tanh \circ \tanh \circ \dots \circ \tanh$ ($n$ fois) a une pente égale à 1 en 0, quel que soit $n$. Qu'est-ce que cela dit de son explication ?

### Ex 22.7 — Le chemin additif de la cellule LSTM ∂ ★★★ ⏱️ 30 min
**Objectif :** démontrer que la mémoire d'un LSTM offre au gradient un chemin sans poids ni tanh, et en mesurer la portée.
**Prérequis :** ∂ 22.6 · Ex 22.4 · fiche §22.7.2 · **Parcours :** M

On reprend les équations du LSTM de la fiche, avec $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ et $h_t = o_t \odot \tanh(c_t)$. Rédige dans ta copie.

1. On regarde seulement le chemin direct de $c_{t-1}$ à $c_t$, en traitant $f_t$, $i_t$ et $g_t$ comme des constantes. Montre que $\dfrac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t)$.
2. En déduire, le long de ce chemin, $\dfrac{\partial c_T}{\partial c_t} = \operatorname{diag}\Big(\displaystyle\prod_{k=t+1}^{T} f_k\Big)$. Calcule ce facteur pour des portes d'oubli toutes égales à 0,99 sur 100 pas, et compare au facteur $0{,}9^{100}$ d'un RNN simple aux facteurs de 0,9.
3. Au début de l'entraînement, les pré-activations des portes sont proches de leur biais. Compare le facteur de 2. sur 20 pas avec un biais d'oubli de 0 (donc $f \approx \sigma(0)$) et avec un biais de 1 (donc $f \approx \sigma(1)$). Pourquoi Jozefowicz et coll. (2015) recommandent-ils un biais positif ?
4. En vrai, $c_{t-1}$ agit aussi sur $c_t$ par un autre chemin : par $h_{t-1} = o_{t-1} \odot \tanh(c_{t-1})$, qui entre dans le calcul de $f_t$, $i_t$ et $g_t$. Écris ce chemin en mots (quelles dérivées, quelles matrices de poids) et explique pourquoi il se comporte comme le gradient d'un RNN simple.
5. Le livre dit que le LSTM échappe au gradient évanescent parce que l'activation qu'il « présente au monde » est linéaire. Confronte cette phrase à l'équation de $h_t$. Quelle est la bonne explication ?
6. Le GRU : avec $h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$, donne la dérivée de $h_t$ par rapport à $h_{t-1}$ le long du chemin direct (en traitant $z_t$ et $n_t$ comme des constantes). Quelle porte joue le rôle de $f_t$ ?

### Ex 22.8 — Fermi : les 17 millions de poids du CNN-LSTM et le prix d'une epoch 🧮 ★★ ⏱️ 20 min
**Objectif :** estimer le nombre de poids et le coût de calcul d'un CNN-LSTM, et voir où se trouve l'essentiel.
**Prérequis :** Ex 22.3 · ch. 21 (paramètres d'une convolution) · fiche §22.8.1 · **Parcours :** R, M

Le CNN-LSTM du livre (§22.8.1) : des images en niveaux de gris de 128 × 128 pixels ; une convolution de 8 filtres 5 × 5, de stride 2 et de padding 2 ; sa sortie aplatie, à chaque image, entre dans une couche LSTM de 128 valeurs d'état ; une couche dense de 9 classes lit le dernier état. Compte comme PyTorch (deux vecteurs de biais pour le LSTM).

a) Le nombre de valeurs de la sortie de la convolution pour une image (ce que reçoit le LSTM à chaque pas).
b) Le nombre de paramètres de la convolution.
c) Le nombre de paramètres de la couche LSTM.
d) Le nombre de paramètres de la couche dense.
e) Le nombre total de paramètres.
f) Le nombre de paramètres de la couche LSTM si l'état n'a plus que 32 valeurs.
g) Le nombre de multiplications-additions du seul produit $x_t W_{ih}$ du LSTM (128 valeurs d'état) pour **une** image.
h) Une epoch de 1 000 vidéos de 16 images : l'ordre de grandeur (la puissance de 10) du nombre de multiplications-additions de ce produit, passe avant seule, pour toute l'epoch. Par exemple, $3 \times 10^{7}$ donne 7.

Dans ta copie : sur un ordinateur qui fait environ $10^{10}$ multiplications-additions par seconde, combien de temps prend cette passe avant ? Où gagnerais-tu le plus de poids et de calcul (indique une modification et son nouveau nombre de paramètres pour le LSTM) ?

<a id="reflexion"></a>

## 🗣️ ⚖️ 📄 Réflexion

### Ex 22.9 — Expliquer l'état caché d'un RNN en cinq lignes 🗣️ ★ ⏱️ 10 min
**Objectif :** expliquer ce qu'est l'état d'un RNN à quelqu'un qui n'en a jamais vu.
**Prérequis :** fiche §22.3, §22.3.1 · **Parcours :** R

Dans ta copie, en 5 lignes au plus et sans formule, explique à un ami ce qu'est l'« état caché » d'un réseau récurrent, et en quoi il diffère des poids. Utilise une image du quotidien de ton choix (autre que la boîte de pièces du livre). Puis relis-toi : as-tu dit que l'état change à chaque entrée, même après l'entraînement ? Que sa taille est fixe ? Que c'est le réseau qui apprend quoi y garder ? Où ton image s'arrête-t-elle ?

### Ex 22.10 — Écrire « à la manière de », imiter une voix : usages et abus ⚖️ ★★ ⏱️ 20 min
**Objectif :** analyser les usages légitimes et les risques des modèles qui imitent un style ou une voix.
**Prérequis :** fiche §22.1, §22.8.1, §22.9 · **Fil rouge :** Holmes/Verne

Le livre présente comme des succès des textes générés « dans la voix » d'un écrivain célèbre, des épisodes de série écrits par un réseau, et la transformation d'une voix en celle de quelqu'un d'autre (p. 1026, p. 1068). Réponds dans ta copie.

1. Ton générateur écrit « à la manière » de Conan Doyle et de Jules Verne. Pourquoi est-ce sans risque juridique pour ces deux auteurs (pense au domaine public), et pourquoi la question serait-elle différente pour un auteur vivant ? Qu'est-ce qui distingue un pastiche, que les écrivains pratiquent depuis toujours, d'une copie ?
2. Les modèles de langue ont été entraînés sur des textes dont les auteurs n'ont pas été consultés. Quels arguments avancent les auteurs et les éditeurs, et quels arguments avancent les entreprises ? Cite un débat ou une affaire que tu connais, en le vérifiant.
3. Le clonage de voix : un réseau peut imiter une voix à partir de quelques secondes d'enregistrement. Donne un usage légitime (par exemple pour une personne qui perd la parole) et un abus (escroquerie, désinformation). Quelles précautions un service de clonage devrait-il prendre (consentement, vérification, marquage) ?
4. Le règlement européen sur l'IA impose des obligations de transparence pour les contenus générés ou manipulés (les *deep fakes*). Que doit faire celui qui diffuse un texte ou un son généré ? Vérifie la date d'application de ces obligations, et si elle a changé.
5. Un ami veut publier sur un site de nouvelles, sous un pseudonyme, des « inédits de Sherlock Holmes » produits par ton générateur. Que lui conseilles-tu de dire aux lecteurs, et pourquoi ?

### Ex 22.11 — Lire Hochreiter et Schmidhuber (1997) : Long Short-Term Memory 📄 ★★★ ⏱️ 45 min
**Objectif :** lire l'article qui a introduit le LSTM, et retrouver dans la version d'origine les idées du chapitre et ce qui a changé depuis.
**Prérequis :** Ex 22.4 · ∂ 22.6 (conseillé) · fiche §22.6, §22.7 · **Parcours :** M

Lis S. Hochreiter et J. Schmidhuber, « Long Short-Term Memory », *Neural Computation*, 9(8), p. 1735-1780, 1997 (DOI 10.1162/neco.1997.9.8.1735 ; une copie PDF libre est indiquée dans `04_indices.md`) : le résumé, l'introduction (section 1), la section 3 (*Constant Error Backpropagation* : l'erreur qui décroît exponentiellement, puis le carrousel à erreur constante), la section 4 (*The Concept of Long Short-Term Memory* : cellules mémoire et portes) et la section 5 (les expériences : au moins l'expérience 4, *Adding Problem*), ainsi que le début de l'annexe A.1. Réponds dans ta copie.

1. Quel problème les auteurs veulent-ils résoudre, et quel ordre de grandeur de délai entre une entrée et sa conséquence visent-ils (résumé) ?
2. Section 3 : comment les auteurs décrivent-ils le comportement du gradient qui remonte le temps ? Relie leur analyse à ta preuve ∂ 22.6 (un produit de facteurs).
3. Section 4 : qu'est-ce que le « carrousel à erreur constante » (*constant error carousel*, CEC) ? Quelle condition sur l'unité donne un flux d'erreur constant ? Relie-la au chemin additif de ∂ 22.7.
4. Quelles portes contient la cellule de 1997 ? Laquelle manque par rapport à celle de PyTorch, et qui l'a ajoutée ensuite ?
5. Dans les expériences, choisis un test : que doit apprendre le réseau, sur quelle longueur de séquence, et à quelles méthodes le LSTM est-il comparé ? Que conclure ?
6. L'entraînement de 1997 ne remonte pas exactement tout le gradient (annexe A.1.2 : une approximation qui coupe certains chemins). Lesquels, et que font les bibliothèques d'aujourd'hui à la place ?
7. Bilan : qu'est-ce qui, dans cet article, se retrouve tel quel dans `nn.LSTM`, et qu'est-ce qui a changé (portes, entraînement, initialisation) ?

<a id="entretien"></a>

## 💼 Entretien

Réponds **à voix haute**, en une minute, comme face à un recruteur ; puis compare avec la « réponse modèle en 60 secondes » de `05_solutions.md`.

### 22.E1 — Pourquoi les Transformers ont-ils remplacé les RNN pour le texte ? Quand garderais-tu un LSTM ? 💼 ★★ ⏱️ 10 min
*Fiche §22.1 (encadré 🕰️), section « Au-delà du livre : après les RNN » · parcours R*

« Pourquoi les Transformers ont-ils remplacé les RNN pour le texte ? Est-ce qu'il vous arrive encore d'utiliser un LSTM, et pour quoi ? »

### 22.E2 — Explique le gradient évanescent et deux façons d'y remédier 💼 ★★ ⏱️ 10 min
*Fiche §22.6, §22.7.2 · parcours R*

« Expliquez-moi le problème du gradient évanescent dans un RNN, et donnez deux façons d'y remédier. »

### 22.E3 — LSTM ou GRU : différences et critères de choix 💼 ★★ ⏱️ 10 min
*Fiche §22.7.2 · parcours R*

« Quelle différence entre un LSTM et un GRU ? Comment choisissez-vous ? »

### 22.E4 — Comment valider un modèle de série temporelle sans fuite de données ? 💼 ★★ ⏱️ 10 min
*Fiche §22.5, section « Au-delà du livre : prévoir une série temporelle » · parcours R*

« Vous devez prévoir les ventes hebdomadaires d'un magasin avec un LSTM. Comment validez-vous votre modèle sans fuite de données, et à quoi le comparez-vous ? »

### 22.E5 — À quoi sert la température quand on génère du texte ? 💼 ★★ ⏱️ 10 min
*Fiche §22.9 · parcours R*

« Dans une API de modèle de langue, il y a un paramètre `temperature`. À quoi sert-il, et quelle valeur choisiriez-vous pour un assistant qui extrait des informations d'un contrat ? Et pour écrire un slogan ? »

<a id="notebook"></a>

## Exercices du notebook

Ces exercices sont dans `03_notebook.ipynb`.

| ID | Titre | Type | ★ | ⏱️ |
|---|---|---|---|---|
| 22.12 | build_vocab, encode et decode sur Holmes et Verne | 🔨 | ★★ | 30 |
| 22.13 | make_windows et make_sequence_pairs | 🔨 | ★★ | 30 |
| 22.14 | Un MLP sur des fenêtres : l'ordre compte-t-il ? | 🔮 | ★★ | 20 |
| 22.15 | rnn_cell_forward et rnn_forward | 🔨 | ★★ | 30 |
| 22.16 | lstm_cell_forward et lstm_forward (portes i, f, g, o) | 🔨 | ★★★ | 45 |
| 22.17 | gru_cell_forward (portes r, z, n) | 🔨 | ★★ | 25 |
| 22.18 | Benchmark : ta cellule NumPy contre nn.LSTM | 🛠️ | ★★ | 20 |
| 22.19 | Mesurer la disparition du gradient : RNN tanh contre LSTM | 🔬 | ★★ | 30 |
| 22.20 | temperature_softmax et sample_from_logits | 🔨 | ★★ | 25 |
| 22.21 | Série temporelle piégée : fuite, mélange et batch_first | 🐛 | ★★ | 25 |
| 22.22 | RNN profonds et bidirectionnels : formes de output et h_n | 📦 | ★★ | 25 |
| 22.23 | Prévoir les taches solaires avec nn.LSTM, face à la persistance | 📦 | ★★★ | 45 |
| 22.24 | Température : prédire l'allure du texte généré | 🔮 | ★ | 10 |
| 22.25 | Générer du Holmes caractère par caractère (deux LSTM, many-to-one) | 📦 | ★★★ | 60 |
| 22.26 | Lire la courbe de loss et la perplexité du générateur | 📈 | ★★ | 15 |
| 22.27 | Holmes contre Verne : même réseau, deux langues | 🔬 | ★★★ | 40 |
| 22.28 | CNN-LSTM : la balle monte-t-elle ou descend-elle ? | 📦 | ★★★ | 50 |
| 22.29 | Défi taches solaires : battre la persistance de 20 % à six mois | 🏆 | ★★★★ | 120 |

🚀 *GPU Colab conseillé* pour le mode complet (`FAST_MODE = False`) de 22.25 et 22.27 ; le mode rapide tourne sur CPU.
