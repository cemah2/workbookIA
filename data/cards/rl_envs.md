# Environnements d'apprentissage par renforcement

| Environnement | Origine | Accès | Chapitres |
|---|---|---|---|
| **Flippers** | jeu décrit dans le livre (ch. 26), recodé de façon originale | code à écrire au ch. 26 (prévu dans `wb`) | 26 |
| **Morpion** (*tic-tac-toe*) | jeu classique, recodé | code à écrire au ch. 26 | 26 |
| **FrozenLake-v1** | gymnasium (Farama Foundation) | `wb.datasets.make_env("FrozenLake-v1", seed=0, is_slippery=False)` | 26 |
| **CartPole-v1** | gymnasium (Farama Foundation) | `wb.datasets.make_env("CartPole-v1", seed=0)` | 26, B8 |

- **Licence** : gymnasium est sous licence MIT. Les jeux recodés font partie du code du workbook.
- **Version** : gymnasium 1.3.0 (voir BIBLE §21). Gym (OpenAI) n'est plus maintenu depuis 2022 : gymnasium est son successeur officiel.
- `make_env` appelle déjà `env.reset(seed=...)` et fixe la graine de l'espace d'actions, pour des expériences reproductibles.
- Les environnements sont du code : rien n'est téléchargé.

## Limites
Environnements « jouets » : ils servent à comprendre les algorithmes (Q-learning, SARSA, DQN, PPO), pas à mesurer leurs performances sur des problèmes réels.
