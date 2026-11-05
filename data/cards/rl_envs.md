# Environnements d'apprentissage par renforcement

| Environnement | Origine | Accès | Chapitres |
|---|---|---|---|
| **Flippers** | jeu décrit dans le livre (ch. 26), recodé de façon originale | `wb.envs.Flippers(size=3, truck_prob=0.1)` | 26 |
| **Morpion** (*tic-tac-toe*) | jeu classique, recodé | `wb.envs.TicTacToe(opponent="random" ou "minimax")`, `wb.envs.minimax_policy` | 26 |
| **CliffWalking-v1** | gymnasium (Farama Foundation), l'exemple 6.6 de Sutton et Barto | `wb.datasets.make_env("CliffWalking-v1", seed=0)` | 26 |
| **FrozenLake-v1** | gymnasium (Farama Foundation), [documentation](https://gymnasium.farama.org/v1.4.0/environments/toy_text/frozen_lake/) | `wb.datasets.make_env("FrozenLake-v1", seed=0, is_slippery=False)` | 26, 29 (🔁 29.R2) |
| **CartPole-v1** | gymnasium (Farama Foundation) | `wb.datasets.make_env("CartPole-v1", seed=0)` | 26, B8 |

- **Licence** : gymnasium est sous licence MIT. Les jeux recodés font partie du code du workbook.
- **Version** : gymnasium 1.4.0 (voir BIBLE §21). Gym (OpenAI) n'est plus maintenu depuis 2022 : gymnasium est son successeur officiel.
- `make_env` appelle déjà `env.reset(seed=...)` et fixe la graine de l'espace d'actions, pour des expériences reproductibles.
- Les environnements sont du code : rien n'est téléchargé.
- **Flippers** : un plateau de 3 × 3 tuiles, avec ou sans point ; un coup retourne une tuile ; on gagne quand les points forment exactement une ligne ou une colonne complète. État = un entier de 0 à 511 (bit $i$ = tuile $i$, ligne par ligne) ; récompense finale $1/\sqrt{k}$ pour une victoire en $k$ coups ; le camion (`truck_prob`) retourne une tuile au hasard après un coup qui ne gagne pas ; partie coupée à 100 coups. `min_moves(state)` donne le nombre minimal de coups pour gagner depuis un plateau.
- **Morpion** : état = code en base 3 du plateau (19 683 codes, dont 5 478 positions atteignables) ; `info["action_mask"]` donne les coups permis ; récompenses +1, 0, −1 ; l'adversaire est aléatoire, parfait (minimax) ou une fonction.
- **CartPole-v1** (chapitre B8) : état de 4 variables continues (position et vitesse du chariot, angle et vitesse angulaire du mât), 2 actions (pousser à gauche, à droite), récompense +1 par pas, épisode **terminé** quand le mât dépasse 12° ou le chariot 2,4 m, **tronqué** à 500 pas ; résolu à 475 de moyenne sur 100 épisodes (`env.spec.reward_threshold`). Physique par défaut : `gravity` 9,8, `length` 0,5 (la **moitié** du mât), `masspole` 0,1, `masscart` 1,0, `force_mag` 10 ; `total_mass` et `polemass_length` sont calculés une fois par `__init__` : `wb.deep_rl_env.set_physics` les recalcule (B8.23, B8.24). `wb.deep_rl_env.make_env(...)` (un environnement, une physique au choix) et `make_vec(n, seed)` (`gym.make_vec` synchrone, autoreset next-step explicite) ; les évaluations du notebook tirent leurs graines de `EVAL_SEED = 8000`, celles du défi B8.28 de `CHALLENGE_EVAL_SEED = 28028`, qu'aucun entraînement n'utilise. La « vraie récompense » du RLHF miniature (B8.26, $\exp(-(x/0{,}5)^2)$ par pas) n'est pas celle de l'environnement : l'agent ne la voit jamais.

## Limites
Environnements « jouets » : ils servent à comprendre les algorithmes (Q-learning, SARSA, DQN, PPO), pas à mesurer leurs performances sur des problèmes réels. Un agent qui résout CartPole-v1 n'a appris qu'une physique : B8.23 et B8.24 montrent ce qu'il devient sur une autre.
