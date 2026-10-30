#!/usr/bin/env python
"""Build the two notebooks of chapter 26 from a single source (used by Claude).

    python tools/chapters/build_ch26.py
    python tools/run_all_notebooks.py chapitres/ch26_rl/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch26_rl/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3 and of the ✏️ paper exercises
26.1 to 26.6 (02_exercices.md). The derivations ∂ 26.7 and 26.8, the estimate 🧮 26.9, the reflection (🗣️ 26.10,
⚖️ 26.11, 📄 26.12) and the interview are corrected with 05_solutions.md.
Parts A to F (exercises 26.13 to 26.38) come at the second generation session: the loop and Gymnasium, Flippers
and L-learning, the learner's mylearn.rl, Q-learning in action, SARSA, replay and rewards, tic-tac-toe, CartPole
and self-play.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, bug_letters_mistakes, check_unique_keys, code, guarded, md,  # noqa: E402,F401
                         paper_cells, part_cells, setup_cell, write_notebook)

CHAPTER = "26"
FOLDER = "chapitres/ch26_rl"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math


def discounted(rewards, gamma):
    """G_0 = sum of gamma**k * r_k."""
    return sum(gamma ** k * r for k, r in enumerate(rewards))


def softmax_prob(q, tau, i):
    """Probability of action i under a softmax of temperature tau."""
    m = max(q)
    weights = [math.exp((v - m) / tau) for v in q]
    return weights[i] / sum(weights)


REWARDS_26_1 = [1, 0, 0, 2, 0, 4]
Q_26_3, R_26_3, NEXT_26_3, GAMMA_26_3, ALPHA_26_3 = 0.5, 0.2, [0.1, 0.6, 0.3, 0.6], 0.9, 0.5
TARGET_26_3 = R_26_3 + GAMMA_26_3 * max(NEXT_26_3)
NEW_26_3 = Q_26_3 + ALPHA_26_3 * (TARGET_26_3 - Q_26_3)
TARGET_26_4 = R_26_3 + GAMMA_26_3 * NEXT_26_3[2]
NEW_26_4 = Q_26_3 + ALPHA_26_3 * (TARGET_26_4 - Q_26_3)
Q_26_5, EPS_26_5 = [1, 2, 4, 0], 0.2


def chain_26_6(alpha, episodes, n_states=3, gamma=0.9):
    """Q of the chain 0 -> 1 -> ... -> n_states (terminal), updated in the order of the steps."""
    q = [0.0] * n_states
    for _ in range(episodes):
        for s in range(n_states):
            target = 1.0 if s == n_states - 1 else gamma * q[s + 1]
            q[s] += alpha * (target - q[s])
    return q
'''

exec(PAPER_CONTEXT)  # the builder computes a few mistakes with the same helpers

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "F"'
LETTERS = 'the letters, e.g. "AF"'
WHOLE = "a whole number"
NUMBER = "a number"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


PAPER = [
    Paper("26.Q1", "Apprendre avec des récompenses : ni supervisé, ni non supervisé", [
        ("a", LETTER, '"B"', letters("B", {"A": "le RL utilise souvent des réseaux : relis ce que l'agent reçoit à la place d'un label",
                                           "C": "un agent apprend de ses propres essais : ce sont des données",
                                           "D": "relis les exemples du livre : ascenseurs, cultures, robot"})),
        ("b", LETTERS, '"ACE"', r'''mistakes={"un des problèmes cités se résout avec des labels, un autre sans aucun signal : ni l'un ni l'autre n'est du RL": "ABCE",
          "il manque un problème où l'on agit et où l'on est jugé sur le résultat": "AC",
          "le regroupement sans labels est de l'apprentissage non supervisé": "ACDE"}'''),
        ("c", TF, "False", r'''mistakes={"relis le §26.2 : quand arrive la récompense d'une partie gagnée ?": True}'''),
        ("d", TF, "False", r'''mistakes={"relis le §26.1 : le livre parle d'une approche générale": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "le livre dit qu'il n'existe souvent pas de réponse objectivement juste",
                                           "C": "relis le §26.2 : le critère est celui qu'on choisit"})),
    ]),
    Paper("26.Q2", "Les quatre informations données à l'agent du morpion", [
        ("a", LETTER, '"B"', letters("B", {"A": "l'amie choisit les coups : c'est l'agent",
                                           "C": "relis qui donne les récompenses et joue en retour"})),
        ("b", LETTERS, '"ABDF"', r'''mistakes={"l'agent ne reçoit pas la règle du jeu : il l'apprend": "ABCDF",
          "il manque une des quatre informations": "ABD",
          "on ne lui dit même pas que c'est un jeu": "ABDEF"}'''),
        ("c", TF, "False", r'''mistakes={"relis ce que le livre dit des informations privées de l'agent": True}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "proposer des coups interdits ne fait pas gagner de temps",
                                           "C": "relis le §26.2.1 : à quoi servent les récompenses d'un coup interdit ?"})),
        ("e", WHOLE, "9 - 2", r'''mistakes={"les coups permis sont les cases vides": 2}'''),
    ]),
    Paper("26.Q3", "État, actions, politique et informations privées", [
        ("a", LETTERS, '"CAB"', r'''mistakes={"l'agent agit avant que l'environnement réponde": "ACB",
          "l'agent se met à jour après la réponse": "CBA"}'''),
        ("b", LETTER, '"B"', letters("B", {"A": "une epoch est un passage sur un dataset, pas une partie",
                                           "C": "la politique est la règle de choix des actions",
                                           "D": "une transition est le passage d'un état à un autre"})),
        ("c", LETTER, '"B"', letters("B", {"A": "une moyenne de récompenses mesure un résultat, pas une règle de choix",
                                           "C": "la liste des états décrit le monde, pas ce qu'on y fait",
                                           "D": "le plateau de départ est un état, pas une règle"})),
        ("d", TF, "False", r'''mistakes={"relis le §26.3.1 : qui exécute l'action, et pourquoi ?": True}'''),
        ("e", TF, "True", r'''mistakes={"relis le §26.3.1 : un état contient tous les nombres qui décrivent l'environnement": False}'''),
    ]),
    Paper("26.Q4", "Réponse de l'environnement, épisode et agents libres ou déclenchés", [
        ("a", LETTER, '"B"', letters("B", {"A": "la politique appartient à l'agent",
                                           "C": "l'environnement ne connaît pas la meilleure action",
                                           "D": "les valeurs Q sont calculées par l'agent"})),
        ("b", LETTERS, '"BA"', r'''mistakes={"le cuisinier attend la sonnerie : il est déclenché ; le minuteur tourne seul": "AB",
          "le minuteur tourne de lui-même": "BB",
          "le cuisinier attend qu'on le sollicite": "AA"}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "au backgammon, chacun attend son tour",
                                           "C": "les deux joueurs attendent leur tour"})),
        ("d", TF, "True", r'''mistakes={"relis le §26.3.4 : lequel des quatre cas le livre dit-il le plus simple ?": False}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "il manque les indicateurs de fin et le dictionnaire info",
                                           "B": "c'est ce que renvoie reset, pas step",
                                           "D": "c'est l'ancienne API de Gym : Gymnasium sépare deux façons de finir"})),
    ]),
    Paper("26.Q5", "MDP, observabilité, attribution du crédit, exploration ou exploitation", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis l'encadré sur les MDP : il s'agit de mémoire, pas d'équiprobabilité",
                                           "C": "la propriété de Markov ne dit rien du signe des récompenses",
                                           "D": "un MDP peut être stochastique"})),
        ("b", LETTER, '"B"', letters("B", {"A": "l'agent voit-il tout l'état ?"})),
        ("c", LETTER, '"B"', letters("B", {"A": "le dilemme porte sur le choix du prochain coup, pas sur le mérite des coups passés",
                                           "C": "il s'agit de partager le mérite d'une victoire entre les coups",
                                           "D": "l'actualisation pèse les récompenses futures, elle ne dit pas quel coup a compté"})),
        ("d", TF, "False", r'''mistakes={"une politique gloutonne rejoue toujours la même action : une action sous-estimée n'est plus jamais essayée": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "la régression logistique ne choisit pas d'actions",
                                           "C": "le k-means regroupe des données, il ne choisit pas d'actions"})),
    ]),
    Paper("26.Q6", "Retour total, retour actualisé et rôle de γ", [
        ("a", WHOLE, "2 + 0 + 1", ""),
        ("b", DEC2, "discounted([2, 0, 1], 0.5)",
         r'''decimals=2, mistakes={"le poids de la 3e récompense est γ², pas γ": 2.5, "c'est le retour total : actualise": 3}'''),
        ("c", TF, "True", r'''mistakes={"avec γ = 0, que valent les poids γ^k pour k ≥ 1 ?": False}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "c'est l'interprétation du livre, que la fiche corrige",
                                           "C": "c'est le rôle du learning rate α"})),
        ("e", WHOLE, "round(1 / (1 - 0.95))", r'''mistakes={"1 / (1 − γ), pas γ / (1 − γ)": 19}'''),
    ]),
    Paper("26.Q7", "Flippers et L-learning : pourquoi « lousy » ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis la règle du jeu au §26.4",
                                           "C": "les diagonales ne comptent pas, et les autres tuiles doivent être vides"})),
        ("b", WHOLE, "6", r'''mistakes={"les diagonales ne comptent pas": 8, "les colonnes comptent aussi": 3}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "la moyenne, c'est justement ce qui manque au L-learning",
                                           "C": "relis la règle de mise à jour du §26.5"})),
        ("d", TF, "True", r'''mistakes={"relis les résultats du livre sans camion, après 6 000 parties": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "le jeu reste gagnable, seulement plus long",
                                           "C": "le camion retourne des tuiles, il n'en ajoute pas"})),
    ]),
    Paper("26.Q8", "La mise à jour du Q-learning : α, γ et la valeur de l'état suivant", [
        ("a", LETTER, '"B"', letters("B", {"A": "la ligne de s porte la valeur qu'on met à jour : la cible regarde ce qui vient après",
                                           "C": "la cible regarde l'état atteint par l'action, pas celui du départ"})),
        ("b", LETTER, '"B"', letters("B", {"A": "c'est le cas α = 1", "C": "α = 0 laisse l'ancienne valeur"})),
        ("c", LETTER, '"B"', letters("B", {"A": "la moyenne, c'est 0 < α < 1", "C": "c'est le cas α = 0"})),
        ("d", TF, "False", r'''mistakes={"relis le §26.6 : quand le Q-learning met-il la table à jour ?": True}'''),
        ("e", DEC2, "0.4 + 0.5 * (0 + 1 * 0.8 - 0.4)",
         r'''decimals=2, mistakes={"c'est la cible : mélange-la avec l'ancienne valeur": 0.8}'''),
    ]),
    Paper("26.Q9", "ε-greedy ou softmax : vrai ou faux", [
        ("a", TF, "True", r'''mistakes={"c'est la version du livre que la fiche corrige": False}'''),
        ("b", DEC2, "1 - 0.1 + 0.1 / 5",
         r'''decimals=2, mistakes={"l'exploration peut aussi tirer l'action gloutonne": 0.9}'''),
        ("c", TF, "False", r'''mistakes={"une grande température rend les probabilités presque égales": True}'''),
        ("d", TF, "True", r'''mistakes={"relis la fin du paragraphe softmax du §26.6.2": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "divise 0 par la somme des zéros : est-ce défini ?",
                                           "C": "aucune action n'est préférée : la formule elle-même pose problème"})),
    ]),
    Paper("26.Q10", "Table initialisée à zéro : comment la valeur remonte et converge", [
        ("a", LETTER, '"A"', letters("A", {"B": "la mise à jour est déterministe : que vaut une cible faite de zéros ?",
                                           "C": "la mise à jour mélange des nombres nuls"})),
        ("b", LETTER, '"B"', letters("B", {"A": "le Q-learning met à jour un seul pas à la fois",
                                           "C": "la première action ne voit la victoire que bien plus tard"})),
        ("c", TF, "True", r'''mistakes={"relis le §26.6.4 : la valeur remonte grâce à la cible qui regarde l'état suivant": False}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "α = 1 remplace à chaque fois : relis l'encadré sur la convergence",
                                           "C": "une politique gloutonne ne visite pas toutes les cases"})),
        ("e", LETTER, '"C"', letters("C", {"A": "relis les résultats du §26.6.5 : la longueur baisse encore après 300 parties",
                                   "B": "relis les résultats du §26.6.5 : la longueur baisse encore après 3 000 parties"})),
    ]),
    Paper("26.Q11", "SARSA contre Q-learning : l'action suivante réellement jouée", [
        ("a", LETTER, '"A"', letters("A", {"B": "le nom liste les cinq éléments d'une mise à jour", "C": "le nom liste les cinq éléments d'une mise à jour, pas un nom propre"})),
        ("b", LETTER, '"B"', letters("B", {"A": "c'est la cible du Q-learning", "C": "relis la formule de SARSA"})),
        ("c", LETTER, '"A"', letters("A", {"B": "SARSA apprend la valeur de la politique qu'il suit"})),
        ("d", TF, "True", r'''mistakes={"si a' est l'action gloutonne, que vaut Q(s', a') comparé au max ?": False}'''),
        ("e", TF, "False", r'''mistakes={"relis l'encadré « un défaut du Q-learning ? »": True}'''),
    ]),
    Paper("26.Q12", "Tables géantes, rejeu d'expérience, AlphaGo Zero et lancer de rayons", [
        ("a", WHOLE, "16", r'''mistakes={"c'est le nombre du livre, pour un plateau de 9 cases": 9}'''),
        ("b", LETTER, '"B"', letters("B", {"A": "relis le §26.8", "C": "une table plus petite ne tiendrait pas les états"})),
        ("c", LETTER, '"B"', letters("B", {"A": "c'est plutôt la version du livre, pas celle de DQN",
                                           "C": "DQN garde toutes les transitions récentes"})),
        ("d", TF, "False", r'''mistakes={"relis le §26.10 : « Zero » veut dire zéro donnée humaine": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "l'adversaire n'est pas une règle de l'agent", "C": "relis le §26.10"})),
    ]),
    Paper("26.R1", "Un réseau comme fonction qui remplace une table (le décodeur d'un VAE)", [
        ("a", LETTER, '"C"', letters("C", {"A": "les codes sont des nombres réels", "B": "les latents ne sont pas des bits"})),
        ("b", TF, "True", r'''mistakes={"relis 25.Q11 et 25.17 : que fait un autoencodeur d'une entrée voisine de ses données ?": False}'''),
        ("c", WHOLE, "4", r'''mistakes={"un DQN rend une valeur par action": 1}'''),
    ]),
    Paper("26.R2", "Que sauvegarder pour reprendre un entraînement ?", [
        ("a", LETTERS, '"ABCD"', r'''mistakes={"l'optimiseur a un état (les moyennes d'Adam)": "ACD",
          "sans l'état des générateurs, la suite ne sera pas la même": "ABC",
          "le code de PyTorch n'est pas un état de l'entraînement": "ABCDE",
          "il faut savoir où l'on s'est arrêté": "ABD"}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "les récompenses sont un historique, pas l'état appris",
                                           "C": "l'environnement n'est pas appris"})),
        ("c", TF, "False", r'''mistakes={"où en était ε ? et le générateur aléatoire ?": True}'''),
    ]),
    Paper("26.R3", "Bandit à bras multiples, ε-greedy et UCB", [
        ("a", LETTER, '"A"', letters("A", {"B": "un bandit n'a pas d'état qui change", "C": "les bras sont les actions, pas des états"})),
        ("b", DEC2, "3 + 0.25 * (5 - 3)", r'''decimals=2, mistakes={"le pas multiplie l'écart au nouveau gain": 3.25}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "c'est la règle gloutonne", "C": "c'est l'exploration d'ε-greedy"})),
    ]),
    Paper("26.1", "Retours total et actualisé d'une partie de six coups", [
        ("a", WHOLE, "sum(REWARDS_26_1)", ""),
        ("b", WHOLE, "sum(REWARDS_26_1[2:])", r'''mistakes={"le TFR du coup 3 ne compte pas les récompenses d'avant": 7}'''),
        ("c", DEC3, "discounted(REWARDS_26_1, 0.5)",
         r'''decimals=3, mistakes={"la récompense du coup 1 n'est pas actualisée (γ⁰ = 1)": discounted(REWARDS_26_1, 0.5) * 0.5}'''),
        ("d", DEC3, "discounted(REWARDS_26_1, 0.9)",
         r'''decimals=3, mistakes={"la récompense du coup 1 n'est pas actualisée (γ⁰ = 1)": discounted(REWARDS_26_1, 0.9) * 0.9}'''),
        ("e", DEC2, "discounted(REWARDS_26_1[3:], 0.9)",
         r'''decimals=2, mistakes={"le retour du coup 4 commence à sa propre récompense": discounted(REWARDS_26_1[4:], 0.9),
          "c'est le retour total du coup 4 : actualise": 6}'''),
        ("f", WHOLE, "REWARDS_26_1[0]", r'''mistakes={"γ⁰ = 1 : la récompense immédiate reste": 0}'''),
        ("g", TF, "True", r'''mistakes={"G_t − G_{t+1} = r_t : quel est son signe ?": False}'''),
    ]),
    Paper("26.2", "Taille des tables : de Flippers au Go", [
        ("a", WHOLE, "2 ** 9", r'''mistakes={"chaque tuile a deux états : 2 puissance 9, pas 9 au carré": 81, "2 × 9 ne compte pas les combinaisons": 18}'''),
        ("b", WHOLE, "2 ** 9 * 9", r'''mistakes={"multiplie par le nombre d'actions": 512}'''),
        ("c", WHOLE, "2 ** 16 * 16", r'''mistakes={"multiplie par le nombre d'actions": 2 ** 16, "16 actions, pas 9": 2 ** 16 * 9}'''),
        ("d", WHOLE, "4 + 4", r'''mistakes={"les colonnes comptent aussi": 4, "les diagonales ne comptent pas": 10}'''),
        ("e", WHOLE, "3 ** 9", r'''mistakes={"trois états par case : 3 puissance 9": 2 ** 9, "3 puissance 9, pas 9 puissance 3": 9 ** 3}'''),
        ("f", WHOLE, "3 ** 16 * 16", r'''mistakes={"c'est le compte du livre, avec 9 colonnes": 3 ** 16 * 9, "multiplie par 16 actions": 3 ** 16}'''),
        ("g", WHOLE, "math.floor(361 * math.log10(3)) + 1", r'''mistakes={"le nombre de chiffres est la partie entière du log, plus 1": 172}'''),
    ]),
    Paper("26.3", "Une mise à jour de Q-learning à la main", [
        ("a", DEC2, "TARGET_26_3", r'''decimals=2, mistakes={"multiplie le max par γ": R_26_3 + max(NEXT_26_3),
          "le max de la ligne de s', pas la moyenne": R_26_3 + GAMMA_26_3 * sum(NEXT_26_3) / 4}'''),
        ("b", DEC2, "TARGET_26_3 - Q_26_3", r'''decimals=2, mistakes={"δ = cible − ancienne valeur, pas l'inverse": Q_26_3 - TARGET_26_3}'''),
        ("c", DEC2, "NEW_26_3", r'''decimals=2, mistakes={"c'est la cible : mélange-la avec l'ancienne valeur": TARGET_26_3,
          "α multiplie δ, puis on l'ajoute à l'ancienne valeur": ALPHA_26_3 * (TARGET_26_3 - Q_26_3)}'''),
        ("d", DEC2, "Q_26_3 + ALPHA_26_3 * (R_26_3 - Q_26_3)", r'''decimals=2, mistakes={"s' terminal : la cible est r seul": NEW_26_3,
          "la cible est r, mais il faut encore mélanger": R_26_3}'''),
        ("e", DEC2, "TARGET_26_3", r'''decimals=2, mistakes={"avec α = 1, la nouvelle valeur est la cible": NEW_26_3}'''),
        ("f", TF, "False", r'''mistakes={"une mise à jour ne touche qu'une case : Q(s, a)": True}'''),
    ]),
    Paper("26.4", "La même transition vue par SARSA", [
        ("a", DEC2, "TARGET_26_4", r'''decimals=2, mistakes={"SARSA prend Q(s', a') de l'action choisie, pas le max": TARGET_26_3}'''),
        ("b", DEC3, "NEW_26_4", r'''decimals=3, mistakes={"c'est la cible : mélange-la avec l'ancienne valeur": TARGET_26_4,
           "SARSA prend Q(s', a') de l'action choisie, pas le max": NEW_26_3}'''),
        ("c", DEC3, "NEW_26_3 - NEW_26_4", r'''decimals=3, mistakes={"Q-learning moins SARSA, pas l'inverse": NEW_26_4 - NEW_26_3}'''),
        ("d", TF, "True", r'''mistakes={"Q(s', 1) vaut 0,6, le max de la ligne": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "le Q-learning apprend la valeur de la politique gloutonne, pas de celle qu'il suit"})),
    ]),
    Paper("26.5", "Probabilités d'action : ε-greedy (standard et version du livre) et softmax", [
        ("a", DEC2, "1 - EPS_26_5 + EPS_26_5 / 4", r'''decimals=2, mistakes={"l'exploration standard peut aussi tirer l'action gloutonne": 1 - EPS_26_5}'''),
        ("b", DEC2, "EPS_26_5 / 4", r'''decimals=2, mistakes={"l'action 0 n'est jouée qu'en exploration : ε divisé par le nombre d'actions": EPS_26_5}'''),
        ("c", DEC2, "1 - EPS_26_5", r'''decimals=2, mistakes={"dans la version du livre, l'exploration ne tire jamais la gloutonne": 1 - EPS_26_5 + EPS_26_5 / 4}'''),
        ("d", DEC3, "EPS_26_5 / 3", r'''decimals=3, mistakes={"le livre tire parmi les trois autres actions": EPS_26_5 / 4}'''),
        ("e", DEC3, "softmax_prob(Q_26_5, 1, 2)", r'''decimals=3, mistakes={"c'est la « softmax » du livre : la vraie passe par l'exponentielle": 4 / 7}'''),
        ("f", DEC3, "softmax_prob(Q_26_5, 2, 2)", r'''decimals=3, mistakes={"divise les valeurs par τ = 2 avant l'exponentielle": softmax_prob(Q_26_5, 1, 2),
          "on divise par τ, on ne multiplie pas": softmax_prob(Q_26_5, 0.5, 2)}'''),
        ("g", TF, "False", r'''mistakes={"e^(Q+10) = e^10 × e^Q : le facteur se simplifie": True}'''),
        ("h", DEC3, "4 / 7", r'''decimals=3, mistakes={"c'est la vraie softmax : le livre divise les valeurs par leur somme": softmax_prob(Q_26_5, 1, 2)}'''),
    ]),
    Paper("26.6", "Regarder la valeur remonter une chaîne d'états", [
        ("a", WHOLE, "round(chain_26_6(1.0, 1)[2])", ""),
        ("b", DEC2, "chain_26_6(1.0, 2)[1]", r'''decimals=2, mistakes={"le pas de 1 à 2 ne rapporte rien : seule la valeur actualisée de l'état 2 compte": 1.0}'''),
        ("c", DEC2, "chain_26_6(1.0, 3)[0]", r'''decimals=2, mistakes={"compte les actualisations entre l'état 0 et la récompense": 0.9}'''),
        ("d", WHOLE, "3", r'''mistakes={"les mises à jour suivent l'ordre des pas : l'état 0 est mis à jour avant que 1 ait une valeur": 1,
          "à cet épisode, l'état 1 avait-il déjà une valeur quand l'état 0 a été mis à jour ?": 2}'''),
        ("e", DEC3, "chain_26_6(0.5, 2)[1]", r'''decimals=3, mistakes={"à l'épisode 2, l'état 1 voit la valeur de l'état 2 d'avant sa propre mise à jour": 0.5 * 0.9 * 0.75,
          "α = 0,5 : on ne garde que la moitié de l'écart": 0.9 * 0.5}'''),
        ("f", DEC3, "chain_26_6(0.5, 3)[2]", r'''decimals=3, mistakes={"compte les mises à jour de l'état 2 pendant les trois épisodes": 0.75}'''),
        ("g", DEC2, "0.9 ** 2", r'''decimals=2, mistakes={"α ne change pas la limite, seulement la vitesse": 0.5 * 0.81,
          "compte les actualisations entre l'état 0 et la récompense": 0.9}'''),
        ("h", WHOLE, "10", r'''mistakes={"refais le raisonnement de d) : à quel épisode chaque état reçoit-il sa première valeur ?": 11,
          "compte les états non terminaux de la chaîne": 9}'''),
    ]),
]

for _paper in PAPER:                      # a dict literal with a repeated message would silently lose one
    for _sub in _paper.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_paper.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 26.1 à 26.6)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"F\"`) ; "
               "plusieurs lettres : entre guillemets (`\"AF\"`), dans l'ordre demandé quand l'énoncé en impose un. "
               "Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore "
               "remplies affichent ⏳. Les questions « dans ta copie », les dérivations ∂ 26.7 et 26.8, "
               "l'estimation 🧮 26.9, la réflexion (🗣️ 26.10, ⚖️ 26.11, 📄 26.12) et l'entretien se corrigent avec "
               "`05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to F (exercises 26.13 to 26.38): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "26.13 à 26.18", "la boucle et Gymnasium, les retours, Flippers et le L-learning"),
    ("B", "26.19 à 26.21", "le cœur de ta librairie `rl.py` : politiques, mises à jour, boucle d'entraînement"),
    ("C", "26.22 à 26.27", "le Q-learning en action : FrozenLake, courbes, un 🐛, Flippers bruité, itération sur les "
                           "valeurs, hyperparamètres"),
    ("D", "26.28 à 26.30", "SARSA contre le Q-learning, au bord de la falaise et sur Flippers"),
    ("E", "26.31 à 26.34", "le rejeu d'expérience, la conception des récompenses, un résultat honnête"),
    ("F", "26.35 à 26.38", "le morpion, CartPole, l'auto-apprentissage et le défi"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 26.Q1 à 26.Q12, 26.R1 à 26.R3, 26.1 à 26.6 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# 26 · Apprentissage par renforcement — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch26_rl/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 26`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 26 · Apprentissage par renforcement — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Manipuler la boucle agent-environnement avec Gymnasium, et calculer des retours actualisés.\n"
               "- Écrire dans `mylearn.rl` un agent tabulaire complet (politiques, mises à jour du Q-learning et de "
               "SARSA, boucle d'entraînement, itération sur les valeurs, mémoire de rejeu) et le comparer sur "
               "Flippers, FrozenLake, CliffWalking, le morpion et CartPole.\n"
               "- Rapporter un résultat de RL sur plusieurs graines.\n\n"
               "**Rappel express.** $G_t = r_t + \\gamma\\,G_{t+1}$. Q-learning : "
               "$Q(s, a) \\leftarrow Q(s, a) + \\alpha\\,[r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)]$, la cible "
               "vaut $r$ si $s'$ est terminal. SARSA : $Q(s', a')$ de l'action réellement choisie au lieu du max. "
               "ε-greedy : l'action gloutonne avec la probabilité $1 - \\varepsilon + \\varepsilon/|\\mathcal{A}|$.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 26.13 à 26.38 (parties A à F : la boucle et Flippers, ta librairie "
             "`rl.py`, le Q-learning, SARSA, le rejeu et les récompenses, le morpion, CartPole et "
             "l'auto-apprentissage) seront ajoutés à la prochaine session de génération ; relance alors "
             "`python tools/start_chapter.py 26` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu écrire et appliquer à la main les mises à jour du Q-learning et de SARSA, état terminal "
               "compris ?\n"
               "2. Sais-tu expliquer ce que règlent $\\alpha$, $\\gamma$ et $\\varepsilon$, et ce qui se passe quand "
               "on les choisit mal ?\n"
               "3. Sais-tu dire pourquoi une table ne passe pas à l'échelle, et ce que DQN y ajoute ?\n\n"
               "**Pour aller plus loin** : les chapitres 2, 3 et 6 de Sutton et Barto (2018), cités dans la fiche. "
               "La suite : le ch. 27, les GAN." + later)]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT, PAPER_INTRO)
    for part in PARTS:
        cells += part_cells(part, kind)
    cells += footer_cells(kind)
    return cells


def main() -> int:
    write_notebook(f"{FOLDER}/03_notebook.ipynb", build("exercise"))
    write_notebook(f"{FOLDER}/05_solutions.ipynb", build("solution"))
    n_checks = sum(len(p.subs) for p in PAPER)
    n_ex = sum(len(part.exercises) for part in PARTS)
    print(f"✅ {FOLDER}/03_notebook.ipynb and 05_solutions.ipynb written ({len(PAPER)} paper exercises, "
          f"{n_checks} paper checks, {n_ex} notebook exercises)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
