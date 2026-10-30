"""Small games with the gymnasium API, for chapter 26 (reinforcement learning).

>>> from wb import envs
>>> env = envs.Flippers(truck_prob=0.1)            # the book's Flippers, with its truck
>>> state, info = env.reset(seed=0)
>>> state, reward, terminated, truncated, info = env.step(4)
>>> game = envs.TicTacToe(opponent="minimax")      # tic-tac-toe against a perfect player

Both games are written from the description of the book (Glassner, ch. 26), not from its code.

**Flippers** (``Flippers``): a board of ``size x size`` tiles, each showing a dot or not. A move flips one tile. The
game is won when the dots form exactly one full row or one full column, every other tile being empty. The state is
an integer: bit ``i`` is tile ``i``, counted row by row from the top left (``board(state)`` draws it). The final
reward of a win in ``k`` moves is ``1 / sqrt(k)``: shorter games are better (the book's values 0.7 in 2 moves and
0.44 in 5 moves). With ``truck_prob > 0``, after a move that does not win, a truck goes by with that probability
and flips one random tile (the book's "big truck").

**Tic-tac-toe** (``TicTacToe``): the agent plays against a fixed opponent (random, minimax, or a function). The
state is the base-3 code of the board: cell ``i`` (row by row) is worth ``0`` (empty), ``1`` (X) or ``2`` (O) times
``3**i``. X always moves first. ``info["action_mask"]`` lists the legal moves (int8, as in gymnasium); with
``illegal="penalty"`` there is no mask, and an illegal move ends the episode with reward -1.

The helpers ``ttt_decode``, ``ttt_encode``, ``ttt_winner``, ``ttt_legal``, ``ttt_swap``, ``minimax_value`` and
``minimax_policy`` work on these codes (self-play, evaluation).
"""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Integral

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - gymnasium is in requirements.txt
    raise ImportError("wb.envs needs gymnasium: pip install gymnasium==1.4.0") from exc

__all__ = [
    "ttt_tables",
    "Flippers",
    "TicTacToe",
    "flippers_winning_states",
    "plot_flippers_game",
    "ttt_decode",
    "ttt_encode",
    "ttt_winner",
    "ttt_legal",
    "ttt_swap",
    "minimax_value",
    "minimax_policy",
    "TTT_N_STATES",
]


def _as_index(value, n: int, what: str) -> int:
    """An integer in 0..n-1 (bools and floats refused), else ValueError."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Integral, np.integer)):
        raise ValueError(f"{what} must be an integer in 0..{n - 1}, got {value!r}")
    value = int(value)
    if not 0 <= value < n:
        raise ValueError(f"{what} must be in 0..{n - 1}, got {value}")
    return value


# ---------------------------------------------------------------------------
# Flippers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def flippers_winning_states(size: int = 3) -> frozenset:
    """The winning boards of Flippers: one full row or one full column, every other tile empty (2 * size boards)."""
    states = set()
    for r in range(size):
        states.add(sum(1 << (r * size + c) for c in range(size)))
    for c in range(size):
        states.add(sum(1 << (r * size + c) for r in range(size)))
    return frozenset(states)


@lru_cache(maxsize=None)
def _flippers_distances(size: int) -> np.ndarray:
    """The minimal number of moves from every board to a winning board (breadth-first search; a move is reversible,
    so the search starts from the winning boards)."""
    n_tiles = size * size
    dist = np.full(1 << n_tiles, -1, dtype=np.int16)
    frontier = np.array(sorted(flippers_winning_states(size)), dtype=np.int64)
    dist[frontier] = 0
    flips = np.int64(1) << np.arange(n_tiles, dtype=np.int64)
    depth = 0
    while frontier.size:
        depth += 1
        neighbours = np.unique((frontier[:, None] ^ flips[None, :]).ravel())
        neighbours = neighbours[dist[neighbours] < 0]
        dist[neighbours] = depth
        frontier = neighbours
    dist.setflags(write=False)
    return dist


class Flippers(gym.Env):
    """The book's Flippers game, with the gymnasium API (see the module docstring for the rules).

    Parameters
    ----------
    size : int, default=3
        Side of the board (``size * size`` tiles, ``2 ** (size * size)`` states, ``size * size`` actions). 2 to 4.
    truck_prob : float, default=0.0
        Probability that the truck flips one random tile after a move that does not win.
    reward : {"final", "win", "shaped"}, default="final"
        ``"final"``: 0 at each step, ``1 / sqrt(k)`` for a win in ``k`` moves; ``"win"``: 1 for any win;
        ``"shaped"``: the final reward plus ``shaping * (d(s) - d(s'))`` at each step, where ``d`` is the minimal
        number of moves to a win (``min_moves``): a bonus for getting closer.
    step_penalty : float, default=0.0
        Subtracted from the reward of every step (a cost per move), >= 0.
    max_steps : int, default=100
        The episode is truncated after this many moves without a win (the book's cut-off of 100 moves).
    shaping : float, default=0.1
        The weight of the shaping bonus (``reward="shaped"`` only), >= 0.
    seed : int or None, default=None
        Seed of the first ``reset()`` called without a seed.

    Observations are ints in ``0 .. 2 ** (size * size) - 1``; ``reset(options={"state": s})`` starts from the board
    ``s`` (not a winning one) instead of a random non-winning board. ``info`` holds ``"moves"`` (moves played) and
    ``"truck"`` (whether the truck flipped a tile during this step).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 3, truck_prob: float = 0.0, reward: str = "final", step_penalty: float = 0.0,
                 max_steps: int = 100, shaping: float = 0.1, seed: int | None = None):
        if isinstance(size, bool) or not isinstance(size, (Integral, np.integer)) or not 2 <= size <= 4:
            raise ValueError(f"size must be an integer from 2 to 4, got {size!r}")
        if not 0.0 <= float(truck_prob) <= 1.0:
            raise ValueError(f"truck_prob must be in [0, 1], got {truck_prob}")
        if reward not in ("final", "win", "shaped"):
            raise ValueError(f'reward must be "final", "win" or "shaped", got {reward!r}')
        if float(step_penalty) < 0 or float(shaping) < 0:
            raise ValueError("step_penalty and shaping must be >= 0")
        if isinstance(max_steps, bool) or not isinstance(max_steps, (Integral, np.integer)) or max_steps < 1:
            raise ValueError(f"max_steps must be an integer >= 1, got {max_steps!r}")
        self.size = int(size)
        self.n_tiles = self.size * self.size
        self.truck_prob = float(truck_prob)
        self.reward_mode = reward
        self.step_penalty = float(step_penalty)
        self.max_steps = int(max_steps)
        self.shaping = float(shaping)
        self.observation_space = spaces.Discrete(1 << self.n_tiles)
        self.action_space = spaces.Discrete(self.n_tiles)
        self.winning = flippers_winning_states(self.size)
        self._first_seed = seed
        self.state: int | None = None
        self.moves = 0
        self._done = True

    # -- the game -------------------------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None and self._first_seed is not None:
            seed, self._first_seed = self._first_seed, None
        super().reset(seed=seed)
        if options and "state" in options:
            state = _as_index(options["state"], 1 << self.n_tiles, "options['state']")
            if state in self.winning:
                raise ValueError(f"the board {state} is already winning: start from another one")
        else:
            while True:
                state = int(self.np_random.integers(1 << self.n_tiles))
                if state not in self.winning:
                    break
        self.state, self.moves, self._done = state, 0, False
        return state, {"moves": 0, "truck": False}

    def step(self, action):
        if self._done:
            raise RuntimeError("the episode is over: call reset() first")
        action = _as_index(action, self.n_tiles, "action")
        before = self.state
        state = before ^ (1 << action)
        truck = False
        if state not in self.winning and self.truck_prob > 0 and self.np_random.random() < self.truck_prob:
            state ^= 1 << int(self.np_random.integers(self.n_tiles))
            truck = True
        self.moves += 1
        terminated = state in self.winning
        reward = 0.0 - self.step_penalty           # 0.0, not -0.0, without a penalty
        if terminated:
            reward += 1.0 if self.reward_mode == "win" else 1.0 / math.sqrt(self.moves)
        if self.reward_mode == "shaped":
            reward += self.shaping * float(self.min_moves(before) - self.min_moves(state))
        truncated = (not terminated) and self.moves >= self.max_steps
        self.state, self._done = state, terminated or truncated
        return state, float(reward), terminated, truncated, {"moves": self.moves, "truck": truck}

    # -- helpers --------------------------------------------------------------------------------------------
    def board(self, state: int | None = None) -> np.ndarray:
        """The board of ``state`` (default: the current one), a (size, size) array of 0 and 1."""
        state = self.state if state is None else _as_index(state, 1 << self.n_tiles, "state")
        bits = (state >> np.arange(self.n_tiles)) & 1
        return bits.reshape(self.size, self.size).astype(np.int64)

    def encode(self, board) -> int:
        """The state of a (size, size) board of 0 and 1 (the inverse of ``board``)."""
        flat = np.asarray(board).reshape(-1)
        if flat.size != self.n_tiles or not np.isin(flat, (0, 1)).all():
            raise ValueError(f"a board is a {self.size} x {self.size} array of 0 and 1")
        return int(sum(int(b) << i for i, b in enumerate(flat)))

    def min_moves(self, state: int | None = None) -> int:
        """The minimal number of moves from ``state`` (default: the current one) to a winning board."""
        state = self.state if state is None else _as_index(state, 1 << self.n_tiles, "state")
        return int(_flippers_distances(self.size)[state])

    def render(self) -> str:
        """The current board as text: ● a dot, · an empty tile."""
        return "\n".join(" ".join("●" if v else "·" for v in row) for row in self.board())


def plot_flippers_game(states, actions=None, size: int = 3, title: str | None = None, ax=None):
    """Draw a game of Flippers: the boards ``states`` from left to right (ints), the tile flipped by
    ``actions[i]`` outlined on board ``i``. Returns the matplotlib Axes (no ``plt.show()``)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from wb.plot import BAD, CATEGORICAL, TEXT

    states = [int(s) for s in states]
    actions = list(actions) if actions is not None else []
    if ax is None:
        _, ax = plt.subplots(figsize=(1.25 * len(states) + 0.3, 1.6))
    winning = flippers_winning_states(size)
    for k, state in enumerate(states):
        x0 = k * (size + 1)
        for i in range(size * size):
            r, c = divmod(i, size)
            outlined = k < len(actions) and actions[k] is not None and int(actions[k]) == i
            ax.add_patch(Rectangle((x0 + c, size - 1 - r), 1, 1, fc="#f4f4f4", ec=BAD if outlined else TEXT,
                                   lw=2.2 if outlined else 0.6))
            if (state >> i) & 1:
                colour = CATEGORICAL[2] if state in winning else CATEGORICAL[0]
                ax.add_patch(plt.Circle((x0 + c + 0.5, size - 1 - r + 0.5), 0.28, color=colour))
    ax.set_xlim(-0.2, len(states) * (size + 1) - 0.8)
    ax.set_ylim(-0.2, size + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Tic-tac-toe
# ---------------------------------------------------------------------------
TTT_N_STATES = 3 ** 9
_POW3 = 3 ** np.arange(9)
_LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


@lru_cache(maxsize=None)
def ttt_tables() -> dict:
    """Lookup tables over the 3**9 codes (computed once, read-only), for fast loops:
    ``"boards"`` (19683, 9) int8, ``"winner"`` (19683,) int8 (0, 1 or 2), ``"legal"`` (19683, 9) int8 (no legal move
    once the game is won), ``"swap"`` (19683,) int64 (the code with X and O exchanged), ``"to_move"`` (19683,) int8
    (1 if X is to move, i.e. both have as many marks, else 2)."""
    codes = np.arange(TTT_N_STATES)
    boards = ((codes[:, None] // _POW3[None, :]) % 3).astype(np.int8)
    winner = np.zeros(TTT_N_STATES, dtype=np.int8)
    for a, b, c in _LINES:
        line = (boards[:, a] == boards[:, b]) & (boards[:, b] == boards[:, c]) & (boards[:, a] > 0)
        winner = np.where(line & (winner == 0), boards[:, a], winner).astype(np.int8)
    legal = ((boards == 0) & (winner[:, None] == 0)).astype(np.int8)
    swapped = np.where(boards == 1, 2, np.where(boards == 2, 1, 0)).astype(np.int64)
    swap = swapped @ _POW3
    to_move = np.where((boards == 1).sum(1) == (boards == 2).sum(1), 1, 2).astype(np.int8)
    tables = {"boards": boards, "winner": winner, "legal": legal, "swap": swap, "to_move": to_move}
    for array in tables.values():
        array.setflags(write=False)
    return tables


def ttt_decode(state: int) -> np.ndarray:
    """The board of a tic-tac-toe code: 9 ints, row by row, 0 empty, 1 X, 2 O."""
    state = _as_index(state, TTT_N_STATES, "state")
    return (state // _POW3) % 3


def ttt_encode(board) -> int:
    """The code of a board of 9 values in {0, 1, 2} (the inverse of ``ttt_decode``)."""
    board = np.asarray(board).reshape(-1)
    if board.size != 9 or not np.isin(board, (0, 1, 2)).all():
        raise ValueError("a tic-tac-toe board is 9 values in {0, 1, 2}")
    return int((board.astype(np.int64) * _POW3).sum())


def ttt_winner(board) -> int:
    """1 if X has a line, 2 if O has one, 0 otherwise (a board or a code)."""
    if isinstance(board, (Integral, np.integer)) and not isinstance(board, (bool, np.bool_)):
        return int(ttt_tables()["winner"][_as_index(board, TTT_N_STATES, "state")])
    board = np.asarray(board).reshape(-1)
    for a, b, c in _LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return int(board[a])
    return 0


def ttt_legal(state) -> np.ndarray:
    """The legal moves of a board (a code or 9 values): an int8 mask of the empty cells, none if the game is over."""
    if isinstance(state, (Integral, np.integer)) and not isinstance(state, (bool, np.bool_)):
        return ttt_tables()["legal"][_as_index(state, TTT_N_STATES, "state")].copy()
    board = np.asarray(state).reshape(-1)
    if ttt_winner(board):
        return np.zeros(9, dtype=np.int8)
    return (board == 0).astype(np.int8)


def ttt_swap(state: int) -> int:
    """The same board with X and O exchanged (the board seen by the other player)."""
    return int(ttt_tables()["swap"][_as_index(state, TTT_N_STATES, "state")])


def _to_move(board: np.ndarray) -> int:
    """The mark of the player to move: X (1) when both have as many marks, else O (2)."""
    return 1 if (board == 1).sum() == (board == 2).sum() else 2


@lru_cache(maxsize=None)
def minimax_value(state: int) -> int:
    """The value of a board for the player to move, with perfect play on both sides: +1 win, 0 draw, -1 loss."""
    tables = ttt_tables()
    state = _as_index(state, TTT_N_STATES, "state")
    winner, me = int(tables["winner"][state]), int(tables["to_move"][state])
    if winner:
        return 1 if winner == me else -1   # in practice the player to move has just lost
    empty = np.flatnonzero(tables["boards"][state] == 0)
    if empty.size == 0:
        return 0
    return max(-minimax_value(int(state + me * _POW3[cell])) for cell in empty)


def minimax_policy(state: int, rng: np.random.Generator | None = None) -> int:
    """An optimal move for the player to move (X if both have as many marks), drawn at random among the optimal moves
    with ``rng`` (None: the first optimal move)."""
    board = ttt_decode(state)
    empty = np.flatnonzero(board == 0)
    if ttt_winner(board) or empty.size == 0:
        raise ValueError("the game is over: no move to play")
    me = _to_move(board)
    values = np.array([-minimax_value(int(state + me * _POW3[cell])) for cell in empty])
    best = empty[values == values.max()]
    return int(best[0] if rng is None else rng.choice(best))


class TicTacToe(gym.Env):
    """Tic-tac-toe against a fixed opponent, with the gymnasium API (see the module docstring).

    Parameters
    ----------
    opponent : "random", "minimax" or callable, default="random"
        The other player: a uniformly random legal move, a perfect player (random among its optimal moves), or a
        function ``opponent(state, rng) -> cell`` (``state``: the code of the board before its move, ``rng``: the
        environment's generator).
    agent : {"X", "O"}, default="X"
        The agent's mark. X moves first: with ``agent="O"``, the opponent plays its first move in ``reset``.
    illegal : {"mask", "penalty"}, default="mask"
        ``"mask"``: ``info["action_mask"]`` gives the legal moves, and an illegal move raises ValueError;
        ``"penalty"``: no mask, an illegal move ends the episode with reward -1 (the board does not change).
    seed : int or None, default=None
        Seed of the first ``reset()`` called without a seed.

    Rewards: +1 when the agent wins, -1 when it loses, 0 for a draw and for every other step. ``info["result"]`` is
    ``"win"``, ``"loss"``, ``"draw"``, ``"illegal"`` at the end of an episode, else None.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, opponent="random", agent: str = "X", illegal: str = "mask", seed: int | None = None):
        if not (opponent in ("random", "minimax") or callable(opponent)):
            raise ValueError(f'opponent must be "random", "minimax" or a function, got {opponent!r}')
        if agent not in ("X", "O"):
            raise ValueError(f'agent must be "X" or "O", got {agent!r}')
        if illegal not in ("mask", "penalty"):
            raise ValueError(f'illegal must be "mask" or "penalty", got {illegal!r}')
        self.opponent = opponent
        self.agent = agent
        self.me, self.other = (1, 2) if agent == "X" else (2, 1)
        self.illegal = illegal
        self.observation_space = spaces.Discrete(TTT_N_STATES)
        self.action_space = spaces.Discrete(9)
        self._first_seed = seed
        self.state: int | None = None
        self._done = True

    def _info(self, result=None) -> dict:
        info = {"result": result}
        if self.illegal == "mask":
            info["action_mask"] = ttt_legal(self.state) if result is None else np.zeros(9, dtype=np.int8)
        return info

    def _opponent_move(self) -> int:
        if self.opponent == "random":
            return int(self.np_random.choice(np.flatnonzero(ttt_tables()["legal"][self.state])))
        if self.opponent == "minimax":
            return minimax_policy(self.state, self.np_random)
        cell = self.opponent(self.state, self.np_random)
        cell = _as_index(cell, 9, "the opponent's move")
        if not ttt_legal(self.state)[cell]:
            raise ValueError(f"the opponent played the occupied cell {cell}")
        return cell

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None and self._first_seed is not None:
            seed, self._first_seed = self._first_seed, None
        super().reset(seed=seed)
        self.state, self._done = 0, False
        if self.agent == "O":
            self.state += self.other * int(_POW3[self._opponent_move()])
        return self.state, self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("the episode is over: call reset() first")
        action = _as_index(action, 9, "action")
        if not ttt_tables()["legal"][self.state, action]:
            if self.illegal == "mask":
                raise ValueError(f"illegal move: cell {action} is not empty (use info['action_mask'])")
            self._done = True
            return self.state, -1.0, True, False, self._info("illegal")
        tables = ttt_tables()
        self.state += self.me * int(_POW3[action])
        if tables["winner"][self.state] == self.me:
            self._done = True
            return self.state, 1.0, True, False, self._info("win")
        if not (tables["boards"][self.state] == 0).any():
            self._done = True
            return self.state, 0.0, True, False, self._info("draw")
        self.state += self.other * int(_POW3[self._opponent_move()])
        if tables["winner"][self.state] == self.other:
            self._done = True
            return self.state, -1.0, True, False, self._info("loss")
        if not (tables["boards"][self.state] == 0).any():
            self._done = True
            return self.state, 0.0, True, False, self._info("draw")
        return self.state, 0.0, False, False, self._info()

    def render(self) -> str:
        """The current board as text (X, O and · for an empty cell)."""
        marks = {0: "·", 1: "X", 2: "O"}
        board = ttt_decode(self.state)
        return "\n".join(" ".join(marks[int(v)] for v in board[r * 3:r * 3 + 3]) for r in range(3))
