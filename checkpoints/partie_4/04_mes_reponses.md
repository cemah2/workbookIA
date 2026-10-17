# Checkpoint IV · Ma copie de l'examen blanc

> Ta copie de ce fichier est dans `mon_travail/checkpoints/partie_4/04_mes_reponses.md` (créée par `python tools/start_chapter.py CP4`) : c'est **elle** que tu remplis, pendant l'examen, et c'est elle que Claude lit s'il corrige ton travail (prompt P9).
> Écris ta démarche, pas seulement le résultat : une bonne partie des points d'une question de calcul va à la méthode. Tu peux aussi composer sur papier, puis recopier ici tes réponses finales.

**Date :** · **Début :** · **Fin :** · **Durée :**

## CP4.1 — Questions flash

- a) paramètres du MLP 20-64-32-3 : · b) forme :
- c) [ReLU(−1,5), σ(0), tanh′(0), leaky ReLU(−2)] =
- d) première probabilité : · e) paramètres appris de la batchnorm : · f) paramètres de la convolution :

## CP4.2 — Passe avant, rétropropagation et un pas de SGD

- a) [z₁, z₂] =
- b) [ŷ, L] =
- c) δ =
- d) [∂L/∂W⁽²⁾₁, ∂L/∂W⁽²⁾₂, ∂L/∂b⁽²⁾] =
- e) [∂L/∂z₁, ∂L/∂z₂] =
- f) ∂L/∂W⁽¹⁾ =
- g) nouvelle W⁽¹⁾ =
- h) loss après le pas :
- i) pourquoi le neurone 2 ne bouge pas, et pour tout le dataset :

## CP4.3 — Momentum, Adam et AdamW

- a) b₂ = · b) w₂ =
- c) w₁ (Adam) = · d) [m̂₂, v̂₂] = · e) w₂ (Adam) =
- f) w₁ (AdamW) = · g) w₁ (Adam couplé) =
- h) comparaison et conclusion :

## CP4.4 — Dropout et batchnorm

- a) sortie en entraînement : · b) sortie en évaluation :
- c) moyenne de a) : · pourquoi pas la moyenne de x :
- d) [μ, σ²] = · e) sortie pour 12 :
- f) [μ_run, σ²_run] = · g) sortie en évaluation pour 7 :
- h) pourquoi g) est loin de β :

## CP4.5 — Le gradient à travers des sigmoïdes empilées

- a) démonstration :
- b) ∂L/∂a₀ =
- c) plus petit L :
- d) ReLU, et un autre remède :

## CP4.6 — Fermi : paramètres et mémoire

- a) paramètres (millions) : · b) mémoire (Mo) : · c) facteur : · d) mises à jour :

## CP4.7 — Quatre bugs dans une boucle PyTorch

- a) lettres des quatre lignes fautives :
- b) pour chacune, la correction et le symptôme :

## CP4.8 — Quatre paires de courbes

- a) ① · b) ② · c) ③ · d) ④

## CP4.9 — Une epoch avec AdamW et clipping (notebook, partie A)

- a), b) et c) (dans le notebook)
- d) [loss, accuracy] =

## CP4.10 — Entretien : la rétropropagation

## CP4.11 — Refus de crédit : quelle explication ?

## CP4.12 — Parties I à III

- a) log loss (nats) : · b) log loss (bits) : · c) log loss moyenne :
- d) [w pour α = 0, w pour α = 7] =
- e) z-score du manchot de test :
- f) pourquoi pas les statistiques des cinq manchots :

## CP4.13 — Cinq extraits PyTorch

- a) · b) · c) · d) · e)

## CP4.14 — Pourquoi pas des poids à zéro (5 lignes)

---

## Après l'examen : ma note

Remplis ce tableau avec le barème de `03_examen_corrige.md` (et la vérification de la partie B du notebook).

| Question | Points | Mes points | ⏱️ réel | À revoir (remédiation du corrigé) |
|---|---|---|---|---|
| CP4.1 | 1,5 | |  | |
| CP4.2 | 3 | |  | |
| CP4.3 | 2 | |  | |
| CP4.4 | 1,5 | |  | |
| CP4.5 | 1 | |  | |
| CP4.6 | 1 | |  | |
| CP4.7 | 2 | |  | |
| CP4.8 | 1 | |  | |
| CP4.9 | 2 | |  | |
| CP4.10 | 1 | |  | |
| CP4.11 | 0,5 | |  | |
| CP4.12 | 2 | |  | |
| CP4.13 | 1 | |  | |
| CP4.14 | 0,5 | |  | |
| **Total** | **20** | |  | |

**Ce que je refais dans une semaine** (les questions où j'ai eu moins de la moitié des points) :
