"""Tests of wb.envs: Flippers and tic-tac-toe with the gymnasium API (chapter 26)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from wb import envs  # noqa: E402


# ---------------------------------------------------------------------------
# Flippers
# ---------------------------------------------------------------------------
def test_flippers_winning_boards():
    win = envs.flippers_winning_states(3)
    assert len(win) == 6
    assert {0b000000111, 0b000111000, 0b111000000, 0b001001001, 0b010010010, 0b100100100} == set(win)
    assert len(envs.flippers_winning_states(4)) == 8


def test_flippers_optimal_lengths_match_the_enumeration():
    env = envs.Flippers()
    counts = Counter(env.min_moves(s) for s in range(512))
    assert dict(counts) == {0: 6, 1: 54, 2: 162, 3: 187, 4: 87, 5: 15, 6: 1}


def test_flippers_min_moves_is_consistent_with_one_move():
    env = envs.Flippers()
    for s in range(512):
        d = env.min_moves(s)
        neighbours = [env.min_moves(s ^ (1 << a)) for a in range(9)]
        if d == 0:
            continue
        assert min(neighbours) == d - 1, f"board {s}: no neighbour one move closer"
        assert all(abs(n - d) == 1 for n in neighbours)


def test_flippers_board_encode_round_trip_and_render():
    env = envs.Flippers()
    board = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
    s = env.encode(board)
    assert s == 1 + 16 + 128 + 256
    assert np.array_equal(env.board(s), board)
    env.reset(options={"state": s})
    assert env.render().splitlines()[0] == "● · ·"


def test_flippers_win_in_two_moves_gives_one_over_sqrt_two():
    env = envs.Flippers()
    start = env.encode([[1, 1, 0], [0, 0, 0], [0, 0, 1]])     # flip 8 then 2: top row
    state, info = env.reset(options={"state": start})
    assert state == start and info == {"moves": 0, "truck": False}
    s1, r1, term1, trunc1, info1 = env.step(8)
    assert (r1, term1, trunc1, info1["moves"]) == (0.0, False, False, 1)
    s2, r2, term2, trunc2, _ = env.step(2)
    assert term2 and not trunc2 and s2 == 0b111
    assert r2 == pytest.approx(1 / math.sqrt(2))
    with pytest.raises(RuntimeError):
        env.step(0)


def test_flippers_reward_modes_and_penalty():
    start = 0b000000011                                       # one move from the top row
    for mode, expected in (("final", 1.0), ("win", 1.0)):
        env = envs.Flippers(reward=mode, step_penalty=0.05)
        env.reset(options={"state": start})
        assert env.step(2)[1] == pytest.approx(expected - 0.05)
    env = envs.Flippers(reward="shaped", shaping=0.1)
    env.reset(options={"state": start ^ (1 << 8)})          # two moves away
    rewards = [env.step(8)[1], env.step(2)[1]]
    assert rewards[0] == pytest.approx(0.1)                   # one move closer
    assert rewards[1] == pytest.approx(0.1 + 1 / math.sqrt(2))


def test_flippers_truncation_at_max_steps():
    env = envs.Flippers(max_steps=3)
    env.reset(options={"state": 0b000000001})                # flipping tile 4 back and forth never wins
    results = [env.step(4) for _ in range(3)]
    assert [r[3] for r in results] == [False, False, True]
    assert not any(r[2] for r in results)


def test_flippers_truck_flips_one_tile_and_is_reproducible():
    env = envs.Flippers(truck_prob=1.0)
    s0, _ = env.reset(seed=3, options={"state": 0b000010000})
    s1, _, _, _, info = env.step(0)
    assert info["truck"]
    assert bin(s1 ^ s0 ^ 1).count("1") == 1                 # the move, then exactly one other flip
    env2 = envs.Flippers(truck_prob=1.0)
    env2.reset(seed=3, options={"state": 0b000010000})
    assert env2.step(0)[0] == s1


def test_flippers_no_truck_after_a_winning_move():
    env = envs.Flippers(truck_prob=1.0)
    env.reset(seed=0, options={"state": 0b000000011})
    state, reward, terminated, _, info = env.step(2)
    assert terminated and state == 0b111 and not info["truck"] and reward == pytest.approx(1.0)


def test_flippers_reset_draws_non_winning_boards_reproducibly():
    env = envs.Flippers(seed=5)
    first = [env.reset()[0]]
    first += [env.reset()[0] for _ in range(200)]
    assert not set(first) & envs.flippers_winning_states(3)
    env2 = envs.Flippers()
    assert env2.reset(seed=5)[0] == first[0]
    assert len(set(first)) > 100


@pytest.mark.parametrize("kwargs, why", [
    ({"size": 5}, "size 5 has 2**25 states"),
    ({"size": True}, "a bool is not a size"),
    ({"truck_prob": 1.5}, "a probability is in [0, 1]"),
    ({"reward": "dense"}, "unknown reward mode"),
    ({"step_penalty": -0.1}, "a penalty is a cost"),
    ({"max_steps": 0}, "at least one move"),
], ids=["size-5", "size-bool", "truck-1.5", "reward-unknown", "penalty-negative", "max-steps-0"])
def test_flippers_refuses_bad_settings(kwargs, why):
    with pytest.raises(ValueError):
        envs.Flippers(**kwargs)


@pytest.mark.parametrize("action, why", [(9, "only 9 tiles"), (-1, "no negative tile"), (1.0, "a float"),
                                         (True, "a bool")], ids=["9", "minus-1", "float", "bool"])
def test_flippers_refuses_bad_actions(action, why):
    env = envs.Flippers()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


def test_flippers_refuses_a_winning_start():
    with pytest.raises(ValueError):
        envs.Flippers().reset(options={"state": 0b111})


def test_flippers_passes_the_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    check_env(envs.Flippers(truck_prob=0.1), skip_render_check=True)


def test_flippers_size_four():
    env = envs.Flippers(size=4)
    assert env.observation_space.n == 65536 and env.action_space.n == 16
    assert env.min_moves(0) == 4


def test_plot_flippers_game_draws(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    ax = envs.plot_flippers_game([0b000000011, 0b000000111], actions=[2, None], title="test")
    assert len(ax.patches) == 18 + 5


# ---------------------------------------------------------------------------
# Tic-tac-toe
# ---------------------------------------------------------------------------
def test_ttt_codes_round_trip_and_swap():
    board = np.array([1, 0, 2, 0, 1, 0, 0, 0, 2])
    s = envs.ttt_encode(board)
    assert np.array_equal(envs.ttt_decode(s), board)
    assert np.array_equal(envs.ttt_decode(envs.ttt_swap(s)), [2, 0, 1, 0, 2, 0, 0, 0, 1])
    assert envs.ttt_winner([1, 1, 1, 2, 2, 0, 0, 0, 0]) == 1
    assert envs.ttt_winner(envs.ttt_encode([2, 1, 1, 2, 1, 0, 2, 0, 0])) == 2
    assert envs.ttt_legal([1, 1, 1, 2, 2, 0, 0, 0, 0]).sum() == 0
    assert envs.ttt_legal(0).dtype == np.int8 and envs.ttt_legal(0).sum() == 9


def test_ttt_has_5478_reachable_positions_and_255168_games():
    seen, games = set(), 0
    stack = [0]
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        board = envs.ttt_decode(s)
        legal = np.flatnonzero(envs.ttt_legal(board))
        me = 1 if (board == 1).sum() == (board == 2).sum() else 2
        stack.extend(int(s + me * 3 ** c) for c in legal)
    assert len(seen) == 5478

    memo = {}

    def count(s):                                 # the number of games from s (two orders of moves = two games)
        if s in memo:
            return memo[s]
        board = envs.ttt_decode(s)
        legal = np.flatnonzero(envs.ttt_legal(board))
        if legal.size == 0:
            return 1
        me = 1 if (board == 1).sum() == (board == 2).sum() else 2
        memo[s] = sum(count(int(s + me * 3 ** c)) for c in legal)
        return memo[s]

    assert count(0) == 255168


def test_minimax_values():
    assert envs.minimax_value(0) == 0
    # X to move can complete the top row
    assert envs.minimax_value(envs.ttt_encode([1, 1, 0, 2, 2, 0, 0, 0, 0])) == 1
    assert envs.minimax_policy(envs.ttt_encode([1, 1, 0, 2, 2, 0, 0, 0, 0])) == 2
    # O to move must block a double threat it cannot block: loss
    assert envs.minimax_value(envs.ttt_encode([1, 2, 0, 0, 1, 0, 0, 0, 0])) in (-1, 0)
    with pytest.raises(ValueError):
        envs.minimax_policy(envs.ttt_encode([1, 1, 1, 2, 2, 0, 0, 0, 0]))


def test_minimax_never_loses_against_random_play():
    rng = np.random.default_rng(0)
    results = Counter()
    for agent in ("X", "O"):
        env = envs.TicTacToe(opponent="minimax", agent=agent)
        for episode in range(60):
            state, info = env.reset(seed=episode)
            done = False
            while not done:
                action = int(rng.choice(np.flatnonzero(info["action_mask"])))
                state, reward, done, _, info = env.step(action)
            results[info["result"]] += 1
    assert results["win"] == 0 and results["loss"] > 80


def test_minimax_against_minimax_always_draws():
    env = envs.TicTacToe(opponent="minimax")
    for episode in range(20):
        state, info = env.reset(seed=episode)
        done = False
        while not done:
            state, reward, done, _, info = env.step(envs.minimax_policy(state, np.random.default_rng(episode)))
        assert info["result"] == "draw" and reward == 0.0


def test_ttt_rewards_and_results():
    env = envs.TicTacToe(opponent=lambda state, rng: int(np.flatnonzero(envs.ttt_legal(state))[-1]))
    state, info = env.reset(seed=0)
    assert state == 0 and info["action_mask"].tolist() == [1] * 9
    state, r, done, trunc, info = env.step(0)     # the opponent plays the last empty cell, 8
    assert envs.ttt_decode(state).tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 2] and r == 0.0 and not done
    env.step(1)                                   # opponent: 7
    state, r, done, _, info = env.step(2)         # top row
    assert done and r == 1.0 and info["result"] == "win" and info["action_mask"].sum() == 0
    with pytest.raises(RuntimeError):
        env.step(3)


def test_ttt_agent_o_and_loss():
    env = envs.TicTacToe(opponent=lambda state, rng: int(np.flatnonzero(envs.ttt_legal(state))[0]), agent="O")
    state, info = env.reset(seed=0)
    assert envs.ttt_decode(state).tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    env.step(8)                                   # X: 1
    state, r, done, _, info = env.step(7)         # X: 2, top row
    assert done and r == -1.0 and info["result"] == "loss"


def test_ttt_illegal_moves():
    env = envs.TicTacToe(seed=0)
    env.reset()
    env.step(4)
    with pytest.raises(ValueError):
        env.step(4)
    penalty = envs.TicTacToe(illegal="penalty", seed=0)
    state, info = penalty.reset()
    assert "action_mask" not in info
    penalty.step(4)
    state2, r, done, _, info = penalty.step(4)
    assert done and r == -1.0 and info["result"] == "illegal"


def test_ttt_refuses_bad_settings_and_bad_opponents():
    with pytest.raises(ValueError):
        envs.TicTacToe(opponent="perfect")
    with pytest.raises(ValueError):
        envs.TicTacToe(agent="Z")
    with pytest.raises(ValueError):
        envs.TicTacToe(illegal="ignore")
    env = envs.TicTacToe(opponent=lambda state, rng: 0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(0)                               # the opponent then plays the occupied cell 0


def test_ttt_passes_the_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    check_env(envs.TicTacToe(), skip_render_check=True)


def test_reference_train_tabular_runs_on_both_games():
    from wb.impl import load_mylearn
    rl = load_mylearn("ref").rl
    Q, history = rl.train_tabular(envs.TicTacToe(), 200, rng=np.random.default_rng(0))
    assert Q.shape == (3 ** 9, 9) and history["returns"].shape == (200,)
    Q, history = rl.train_tabular(envs.Flippers(truck_prob=0.1), 300, alpha=0.5, gamma=0.9,
                                  rng=np.random.default_rng(0))
    assert Q.shape == (512, 9) and history["lengths"].max() <= 100
