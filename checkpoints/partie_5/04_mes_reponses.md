# Checkpoint V · Ma copie de l'examen blanc

> Ta copie de ce fichier est dans `mon_travail/checkpoints/partie_5/04_mes_reponses.md` (créée par `python tools/start_chapter.py CP5`) : c'est **elle** que tu remplis, pendant l'examen, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat : une bonne partie des points d'une question de calcul va à la méthode. Tu peux aussi composer sur papier, puis recopier ici tes réponses finales.

**Date :** · **Début :** · **Fin :** · **Durée :**

## CP5.1 — Forme de sortie ou erreur ?

- a) · b) · c)
- d) · e) · f)
- g) l'extrait qui lève une erreur, pourquoi, la correction et la forme obtenue ; les trois nombres de d) :

## CP5.2 — Formes, paramètres et champ récepteur d'un petit CNN

- a) sortie du bloc A : · b) sortie du bloc B : · c) sortie de la convolution C :
- d) [conv A, conv B, conv C, dense, total] =
- e) MAC de la convolution C :
- f) champ récepteur (calcul couche par couche) :
- g) pourquoi `AdaptiveAvgPool2d(1)`, paramètres de la dense après `Flatten()`, autre conséquence :

## CP5.3 — Un pas de LSTM à la main

- a) sommes pondérées : · [i, f, g, o] =
- b) c₁ = · c) h₁ =
- d) paramètres de `nn.LSTM(10, 20)` :
- e) pourquoi le chemin additif aide le gradient :

## CP5.4 — Quatre bugs de CNN et de RNN

- a) les quatre lettres :
- b) pour chaque ligne : la correction, et ce qui se passe sans elle (erreur, où et pourquoi, ou résultat faux sans message) :

## CP5.5 — `rnn_forward` (notebook, partie A)

- a) le code est dans le notebook ; remarques éventuelles :
- b) `H_all[1, 4, 0]` =

## CP5.6 — Early stopping et meilleur modèle (notebook, partie A)

- a) le code est dans le notebook ; remarques éventuelles :
- b) [best_epoch, loss de validation à cette epoch] =

## CP5.7 — Quatre paires de courbes

- a) ① : · b) ② : · c) ③ : · d) ④ :

## CP5.8 — Fermi : un générateur de texte LSTM

- a) paramètres (millions) : · b) part de la couche de sortie (%) :
- c) FLOP d'une epoch (unités de 10¹³) : · d) durée (s) :

## CP5.9 — Trois projets, trois architectures

1. Les rayures :
2. La consommation électrique :
3. Les messages des clients :

## CP5.10 — Retour sur les parties III et IV

- a) [μ, σ] = · b) z de x = 2 :
- c) nombre d'ajustements du scaler :
- d) ∂L/∂W = · e) ∂L/∂b =
- f) ∂L/∂X =

## CP5.11 — Questions flash

- a) · b) · c) · d)

## CP5.12 — Rétropropager dans une convolution 1D

- a) (y₁, y₂, y₃) =
- b) démonstration, et d'où vient la somme :
- c) [∂L/∂w₁, ∂L/∂w₂, ∂L/∂b] =
- d) (∂L/∂x₁, …, ∂L/∂x₄) =

## CP5.13 — `model.eval()` et `torch.inference_mode()` (comme à l'oral)

## CP5.14 — Une seule graine pour 0,8 point

---

## Après l'examen : ma note

Remplis ce tableau avec le barème de `03_examen_corrige.md` (et la vérification de la partie B du notebook).

| Question | Points | Mes points | ⏱️ réel | À revoir (remédiation du corrigé) |
|---|---|---|---|---|
| CP5.1 | 2 | |  | |
| CP5.2 | 2,5 | |  | |
| CP5.3 | 2 | |  | |
| CP5.4 | 2 | |  | |
| CP5.5 | 1,5 | |  | |
| CP5.6 | 2 | |  | |
| CP5.7 | 1 | |  | |
| CP5.8 | 1 | |  | |
| CP5.9 | 1 | |  | |
| CP5.10 | 2 | |  | |
| CP5.11 | 1 | |  | |
| CP5.12 | 1 | |  | |
| CP5.13 | 0,5 | |  | |
| CP5.14 | 0,5 | |  | |
| **Total** | **20** | |  | |

**Ce que je refais dans une semaine** (les questions où j'ai eu moins de la moitié des points) :
